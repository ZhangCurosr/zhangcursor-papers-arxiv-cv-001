# AdaOcc: Adaptive 3D Occupancy Prediction for Embodied Tasks

Jinglong Wang<sup>1,2,∗,§</sup> Yunjie Wang<sup>2,3,∗,§</sup> Zhiyang Zhang<sup>1,2,∗,§</sup> Jiawei He<sup>2,4,†</sup> Ye Yuan<sup>5</sup> Bo Qiu<sup>6,†</sup> Jing Zhang<sup>1,†</sup>

<sup>1</sup>Beihang University <sup>2</sup>Beijing Academy of Artificial Intelligence <sup>3</sup>Hebei University of Technology <sup>4</sup>XYZ Embodied AI <sup>5</sup>ShanghaiTech University <sup>6</sup>University of Science and Technology Beijing

wjlzy@buaa.edu.cn 202421902001@stu.hebut.edu.cn 20377279@buaa.edu.cn jwhe2024@gmail.com yuanye2024@shanghaitech.edu.cn qiubo@ustb.edu.cn zhang\_jing@buaa.edu.cn

<sup>∗</sup>These authors contributed equally to this work. <sup>§</sup>These authors conducted this work during an internship at XYZ Embodied AI. <sup>†</sup>Corresponding authors: Jing Zhang, Bo Qiu, and Jiawei He.

## Abstract

Embodied tasks demand accurate, flexible, and semantically rich 3D scene representations. 3D semantic occupancy is well suited to this requirement, as it can model holistic 3D spaces by encoding geometric occupancy along with semantic categories. However, existing occupancy prediction methods struggle to meet practical deployment requirements, such as adapting to varying computing budgets, sensor setups, and observation views. In this paper, we propose a point-based Adaptive 3D Occupancy Prediction method, called AdaOcc, tailored for embodied scenarios. To accommodate heterogeneous sensor inputs, AdaOcc uses an adaptive geometry-guided dual-branch encoder that can support RGB images in various numbers of views with (estimated) depth maps or LiDAR scans. AdaOcc represents occupied regions via sparse semantic points trained with a progressive query learning strategy, allowing the prediction computational budget to be flexibly adjusted through query point numbers and decoder layers. To facilitate high-fidelity geometric modeling for lightweight point-based occupancy learning, we further propose a novel containment loss that regularizes predicted points to reside within valid occupied regions. Extensive experiments show that our method achieves a new state-of-the-art on Occ-ScanNet with considerable performance improvements over previous methods. Moreover, our framework demonstrates strong practical applicability as an adaptive 3D perception module in real-world embodied systems.

## 1 Introduction

Accurate, flexible, and semantically rich 3D scene representations play a fundamental role in embodied AI tasks, as embodied agents typically make decisions based on their perception of the environment. 3D semantic occupancy can model holistic 3D spaces by encoding both geometric occupancy and semantic categories, providing a general-purpose representation for embodied scene

![](images/234fad00659c89bf51f49666a7c91b706c816d2f2c9353e13186f670193c3817.jpg)  
Figure 1: Overview of AdaOcc. AdaOcc targets adaptive 3D occupancy prediction for embodied scene understanding. It accepts heterogeneous visual and geometric inputs from different embodied platforms and adapts to different computational budgets through adjustable query numbers and decoder depths. AdaOcc predicts sparse semantic points, which can be converted into semantic occupancy to provide structured spatial cues for downstream embodied tasks.

understanding (Song et al., 2017; Cao and De Charette, 2022; Li et al., 2023a; Zhang et al., 2023;   
Wei et al., 2023; Tong et al., 2023).

Early 3D occupancy prediction methods mainly focus on outdoor driving scenes, representing 3D scenes as dense voxel grids (Zhang et al., 2023; Wei et al., 2023; Tong et al., 2023; Huang et al., 2023; Yu et al., 2023; Ma et al., 2024; Lu et al., 2024), 3D Gaussian primitives (Huang et al., 2024, 2025), or sparse points (Wang et al., 2024a). Recent research on embodied perception (Wang et al., 2024b; Wu et al., 2025; Wang et al., 2025a; Zhang et al., 2025) extends these outdoor methods by improving geometric and semantic modeling for more diverse and complex indoor scenes, with concurrent work further exploring adaptive serialization (Wang et al., 2026), geometry-prior sparse Gaussians (Zhou et al., 2026a), and open-vocabulary prediction (Zhou et al., 2026b). Despite these advances, prevailing methods still face inherent limitations in balancing accuracy and latency. Moreover, they cannot adapt to robotic platforms with heterogeneous sensor setups, and show limited generalization acros diverse embodied downstream tasks.

In this work, we introduce AdaOcc, a unified point-based framework for adaptive 3D occupancy prediction in embodied tasks. As illustrated in Fig. 1, AdaOcc supports different robots and sensors, and can be applied to different downstream embodied tasks. Since robotic platforms can have different numbers of cameras and different geometric sensors, we design an adaptive geometry-guided dualbranch encoder that accepts RGB observations with varying numbers of views and incorporates geometric cues from estimated depth maps, depth-camera measurements, or LiDAR scans. We unify different geometric cues into a calibrated 3D point cloud, by calibrating the LiDAR point cloud to the RGB camera coordinate system and lifting depth maps into point clouds, enabling AdaOcc to adapt to different sensing conditions.

We observe that robotic platforms differ in computational capability, and embodied tasks demand different levels of perceptual granularity (e.g., manipulation often requires finer perception than navigation). We design AdaOcc to initialize sparse point queries from geometric inputs and reconstruct occupied semantic points via a multi-layer decoder equipped with progressive query learning. This design supports adaptive inference by adjusting the number of query points or using intermediate decoder outputs, allowing AdaOcc to balance accuracy and latency under varying computational budgets and environmental complexity.

To further enhance geometric fidelity in our point-based pipeline, we introduce a containment loss to regularize predicted points to reside within valid occupied regions, reducing surface-floating artifacts and improving spatial consistency around object boundaries and occupied structures. Experiments on Occ-ScanNet (Yu et al., 2024) show that AdaOcc achieves new state-of-the-art performance, reaching 65.29 IoU and 59.67 mIoU and outperforming the strongest prior method by 2.15 and 3.48 points, respectively. Further efficiency, adaptability, and real-world deployment studies demonstrate that AdaOcc is not only a strong benchmark model, but also a practical spatial perception module for downstream embodied tasks. Our main contributions are summarized as follows:

• We propose AdaOcc, a point-based adaptive 3D occupancy framework for embodied scene understanding. AdaOcc enables flexible occupancy prediction under varying compute budgets, sensor setups, and inference settings.

• For robust point-based occupancy prediction, we introduce key technical designs including an adaptive geometry-guided dual-branch encoder, progressive query learning for adaptive inference, and containment-guided optimization for improved boundary fidelity.

• AdaOcc achieves state-of-the-art performance on Occ-ScanNet with large margins over previous methods. Extensive experiments and real-world embodied-system deployment demonstrate its effectiveness, efficiency, and adaptability for downstream embodied tasks.

## 2 Related work

Semantic occupancy prediction. Early 3D scene understanding research is largely framed as semantic scene completion, which recovers complete scene geometry and semantics from partial RGB-D observations (Song et al., 2017; Garbade et al., 2019; Li et al., 2019; Roldao et al., 2020; Chen et al., 2020; Cai et al., 2021). This formulation later moves toward vision-only scene understanding, where monocular and camera-based methods infer 3D semantic occupancy from RGB observations by learning to lift image features into 3D space (Cao and De Charette, 2022; Yao et al., 2023; Li et al., 2023a; Yu et al., 2024; Wang et al., 2026). As occupancy prediction expands to larger scenes and multi-view settings, the field further shifts from dense voxel prediction toward more efficient representations. Dense voxel and view-transformation methods provide regular 3D supervision but suffer from high memory and computation costs (Zhang et al., 2023; Wei et al., 2023; Tong et al., 2023; Wang et al., 2024c; Li et al., 2023b), motivating compact decompositions, sparse structures, Gaussian primitives, and query-based formulations for scalable 3D reasoning (Yu et al., 2023; Ma et al., 2024; Lu et al., 2024; Tang et al., 2024; Huang et al., 2025; Jia et al., 2023; Li et al., 2024a; Wang et al., 2024a; Zhou et al., 2026a,b). These works establish strong occupancy prediction backbones, but their assumptions on inputs, views, and inference budgets are usually tied to fixed evaluation protocols.

Occupancy for embodied perception. Recent embodied AI studies further extend occupancy prediction from offline scene reconstruction toward deployable spatial perception for robot navigation, interaction, and decision-making. Embodied perception benchmarks emphasize holistic 3D represen tations for realistic agent environments (Wang et al., 2024b; Xu et al., 2024), while robot-oriented occupancy methods begin to consider online perception, temporal scene understanding, and embodied semantic reasoning (Wu et al., 2025; Wang et al., 2025a; Zhang et al., 2025; Li et al., 2025). These works suggest that 3D occupancy is a useful spatial representation for embodied perception and robot scene understanding. However, practical robot deployment introduces additional variability that is less explored in existing studies. Most current embodied occupancy pipelines are still built around fixed sensing inputs, fixed view configurations, and fixed prediction budgets, making them difficult to adapt across platforms and tasks without redesign or retraining. Motivated by this gap, AdaOcc develops adaptive point-based occupancy prediction for embodied deployment.

## 3 Method

## 3.1 Problem Setup and Method Overview

We study semantic occupancy prediction in an agent-centric 3D space. Given one or multiple calibrated RGB images $\dot { \mathcal { T } } = \{ \bar { \mathbf { I } } ^ { v } \} _ { v = 1 } ^ { V }$ and optional geometric observations $\mathcal { G } = \{ \mathbf { g } _ { m } \} _ { m = 1 } ^ { M }$ , the goal is to predict the occupancy state and semantic label within a bounded local 3D region Ω. Here, G is a unified point set in the local agent-centric coordinate system, where M is the number of geometric points and $\mathbf { g } _ { m }$ is the m-th point. When no independent geometric sensor is available, G can be estimated solely from I, as in our Occ-ScanNet experiments. Following standard settings, Ω is discretized into a voxel grid $\mathcal { V } = \{ \mathbf { p } _ { i } \} _ { i = 1 } ^ { H \times W \times D }$ . Each voxel has a label $y _ { i } \in \{ 0 , 1 , \ldots , C \}$ , where 0 denotes empty space and $1 , \ldots , C$ denote occupied semantic classes. Instead of directly classifying all voxels, AdaOcc predicts a sparse semantic point set $\hat { \mathcal { P } } = \{ ( \hat { \bf p } _ { j } , \hat { \bf s } _ { j } ) \} _ { i = 1 } ^ { N }$ , where $\hat { \mathbf { p } } _ { j }$ is a predicted occupied location and $\hat { \bf s } _ { j }$ contains its semantic logits. Finally, the predicted point set is voxelized into $\hat { \mathbf Y }$ for standard occupancy evaluation.

![](images/dfc5f97e37c603728e3b210821497d73e23247b014076e748eabd566866b33ba.jpg)  
Figure 2: Overview of AdaOcc. AdaOcc is an adaptive 3D occupancy prediction framework for embodied scene understanding. It takes RGB observations as the primary input and can optionally incorporate geometric cues from estimated depth maps, depth cameras, or LiDAR scans. AdaOcc also adjusts its prediction budget under different computational constraints. The predicted semantic occupancy provides structured spatial cues for downstream embodied tasks.

As illustrated in Fig. 2, AdaOcc is a lightweight point-based framework for adaptive 3D occupancy prediction in embodied scenes. It uses an adaptive geometry-guided dual-branch encoder to process RGB observations and optional geometric cues from estimated depth maps, depth cameras, or LiDAR scans. Geometry-guided queries are then refined by a progressive point decoder, producing sparse semantic points under adjustable query and decoder budgets. The model is trained with multi-layer point supervision and the proposed containment-guided optimization.

## 3.2 Adaptive Geometry-Guided Dual-Branch Encoding

Embodied occupancy prediction requires semantic cues from RGB observations and spatial grounding from geometric measurements. AdaOcc therefore adopts an adaptive geometry-guided dual-branch encoder that uses RGB observations as the primary input and optionally incorporates geometric observations when available. The two branches produce image and geometric features as:

$$
\mathcal { F } ^ { \mathrm { i m g } } = E _ { \mathrm { i m g } } ( \mathcal { T } ) = \{ \mathbf { F } _ { r } ^ { \mathrm { i m g } } \} _ { r = 1 } ^ { R _ { \mathrm { i m g } } } , \qquad \mathcal { F } ^ { \mathrm { g e o } } = E _ { \mathrm { g e o } } ( \mathcal { G } ) = \{ \mathbf { F } ^ { x y } , \mathbf { F } ^ { x z } , \mathbf { F } ^ { y z } \} .\tag{1}
$$

Here, $\mathbf { F } _ { r } ^ { \mathrm { i m g } }$ denotes the image feature map at the $r { \mathrm { - t h } }$ scale, and $R _ { \mathrm { i m g } }$ is the number of image feature scales. The geometric observations $\mathcal { G }$ are first voxelized and encoded as follows. For depth inputs, including estimated depth maps and depth-camera measurements, AdaOcc lifts depth pixels into 3D points using camera intrinsics and transforms them with the corresponding camera extrinsics. For LiDAR scans, AdaOcc transforms points from the LiDAR coordinate frame to the RGB-camera coordinate frame using calibrated sensor extrinsics, aligning geometric measurements with visual observations. After this normalization, different geometric sources share the same point-set interface and can be processed by the same geometric branch.

To encode geometry, AdaOcc voxelizes $\mathcal { G }$ and models its local 3D structure with a sparse 3D convolutional encoder. Instead of materializing a dense 3D feature volume, the resulting sparse geometric features are compressed into three orthogonal feature planes $\{ \mathbf { F } ^ { x y } , \mathbf { F } ^ { x z } , \mathbf { F } ^ { y z } \}$ . The xy plane captures horizontal layout cues, while the xz and yz planes preserve vertical geometric profiles, providing height-aware spatial context with low memory cost.

## 3.3 Progressive Point Query Decoding

Inspired by OPUS (Wang et al., 2024a), AdaOcc adopts a sparse point decoder and extends it with geometry-guided query initialization and progressive query learning for budget-adaptive inference. Given an active query budget $N _ { q } ,$ , AdaOcc initializes point queries as ${ \mathcal Q } _ { 0 } = \{ ( { \bf q } _ { i } ^ { 0 } , { \bf f } _ { i } ^ { 0 } ) \} _ { i = 1 } ^ { N _ { q } }$ by sampling 3D locations from the geometric point set ${ \mathcal { G } } .$ , where $\mathbf { q } _ { i } ^ { 0 } \in \Omega$ is the initial reference location and ${ \bf f } _ { i } ^ { 0 }$ is the query feature. This provides sensor-agnostic spatial anchors because $\mathcal { G }$ can come from estimated depth maps, depth cameras, or Li-DAR scans.

![](images/588cd31600a50292ff82c89e4f4c7465b00a00fc0836263052ea8aa7b366a830.jpg)  
Figure 3: Illustration of Containment-Guided Point Optimization (C.P.O.), which regularizes predicted outlier points in supervised empty space to reside inside nearby occupied voxels, reducing surface-floating artifacts.

At decoder layer ℓ, each query predicts K 3D sampling coordinates around its current reference location, denoted as $\mathcal { R } _ { i } ^ { ( \ell ) } =$ $\{ \mathbf { r } _ { i , k } ^ { ( \ell ) } \} _ { k = 1 } ^ { K }$ . AdaOcc then samples both ap-

pearance and geometric evidence at these coordinates: $\mathbf { z } _ { i , k } ^ { \mathrm { i m g } } \ = \ \psi _ { \mathrm { i m g } } ( \mathbf { r } _ { i , k } ^ { ( \ell ) } , \mathcal { F } ^ { \mathrm { i m g } } )$ and ${ \bf z } _ { i , k } ^ { \mathrm { g e o } } \ =$ $\psi _ { \mathrm { g e o } } ( \mathbf { r } _ { i , k } ^ { ( \ell ) } , \mathcal { F } ^ { \mathrm { g e o } } )$ . Here, $\psi _ { \mathrm { i m g } }$ projects each sampling coordinate onto calibrated image planes and samples multi-scale image features, while $\psi _ { \mathrm { g e o } }$ samples geometric features from the xy, xz, and $y z$ planes. The sampled appearance and geometric features are aggregated by the decoder and fused with the previous query feature through a residual update:

$$
\mathbf { f } _ { i } ^ { ( \ell ) } = \mathbf { f } _ { i } ^ { ( \ell - 1 ) } + \phi _ { \ell } \left( \mathbf { f } _ { i } ^ { ( \ell - 1 ) } , \{ ( \mathbf { z } _ { i , k } ^ { \mathrm { i m g } } , \mathbf { z } _ { i , k } ^ { \mathrm { g e o } } ) \} _ { k = 1 } ^ { K } \right) ,\tag{2}
$$

where $\phi _ { \ell }$ denotes the feature aggregation and mixing operation in the ℓ-th decoder layer.

The updated query feature predicts $N _ { p } ^ { ( \ell ) }$ semantic points at layer ℓ, denoted as $\{ ( \hat { \mathbf { p } } _ { i , j } ^ { ( \ell ) } , \hat { \mathbf { s } } _ { i , j } ^ { ( \ell ) } ) \} _ { j = 1 } ^ { N _ { p } ^ { ( \ell ) } }$ where $\hat { \mathbf { p } } _ { i , j } ^ { ( \ell ) } = \mathbf { q } _ { i } ^ { ( \ell - 1 ) } + \Delta \mathbf { p } _ { i , j } ^ { ( \ell ) }$ is a predicted occupied location and $\hat { \mathbf { s } } _ { i , j } ^ { ( \ell ) }$ denotes semantic logits. These predicted points serve as both the layer-wise occupancy output and the basis for spatial refinement. We recenter the next-layer query reference location by the mean predicted coordinate, $\begin{array} { r } { \mathbf q _ { i } ^ { ( \ell ) } = \frac { 1 } { N _ { n } ^ { ( \ell ) } } \sum _ { j = 1 } ^ { N _ { p } ^ { ( \ell ) } } \hat { \mathbf p } _ { i , j } ^ { ( \ell ) } } \end{array}$ . This iterative feature and coordinate refinement allows sparse queries to progressively align with occupied regions.

The active query number directly controls the prediction budget. To improve robustness across query budgets, AdaOcc gradually increases the active number of queries during training:

$$
N _ { q } ( t ) = \operatorname* { m i n } \left( N _ { \mathrm { m a x } } , N _ { \mathrm { i n i t } } + \left\lfloor \frac { t - t _ { 0 } } { \Delta t } \right\rfloor \Delta N \right) ,\tag{3}
$$

where t is the training epoch, $N _ { \mathrm { i n i t } }$ is the initial query budget, $\Delta N$ is the query increment, $\Delta t$ is the growth interval, and $N _ { \mathrm { m a x } }$ is the maximum query budget. This curriculum enables the same trained decoder to adapt to different computational budgets by changing $N _ { q }$ or using intermediate decoder outputs at inference time.

To handle varying observation views, we apply view-level masking during training by randomly retaining a subset of RGB views and their associated geometric observations. When $V _ { t }$ views are retained, the total query budget becomes $N _ { q } ( t , V _ { t } ) \stackrel { \smile } { = } V _ { t } N _ { q } ( t )$ , allowing the model to process different numbers of views jointly in a single forward pass.

## 3.4 Containment-Guided Point Optimization

Point-based occupancy prediction is commonly supervised by Chamfer-style point-set reconstruction losses, such as Chamfer Distance (Fan et al., 2017) and Density-aware Chamfer Distance (DCD) (Wu et al., 2021). Although these losses encourage predicted points to cover the target geometry, they do not explicitly enforce whether a point lies inside an occupied region. As shown in Fig. 3, this may produce surface-floating points that are close to occupied voxels but still located in empty space, leading to inaccurate free-space boundaries.

To address this issue, we introduce Containment-Guided Point Optimization (C.P.O.), which adds a containment loss to regularize predicted points to reside within valid occupied regions. Let ${ \mathcal { P } } ^ { + } = \{ { \bf p } _ { i } \in \mathcal { V } | y _ { i } > 0 \}$ denote occupied ground-truth voxel centers. Each occupied voxel center $\mathbf { p } _ { i }$ induces an occupied box with side length $\mathbf { v } ,$ where v is the voxel size. For a predicted point ${ \hat { \mathbf { p } } } ,$ its distance to the nearest occupied box is

$$
d _ { \mathrm { b o x } } ( \hat { \mathbf { p } } , \mathcal { P } ^ { + } ) = \operatorname* { m i n } _ { \mathbf { p } _ { i } \in \mathcal { P } ^ { + } } \left. \mathrm { R e L U } \left( \left. \hat { \mathbf { p } } - \mathbf { p } _ { i } \right. - \frac { \mathbf { v } } { 2 } \right) \right. _ { 2 } ,\tag{4}
$$

where | · | and the subtraction by $\frac { \mathbf { v } } { 2 }$ are applied element-wise. The distance is zero if pˆ lies inside any occupied voxel box and increases as it drifts into empty space.

Let $\Omega _ { \mathrm { o c c } } \subseteq \Omega$ denote the valid occupancy supervision region defined by the evaluation mask. For decoder layer ℓ, we collect predictions that fall into supervised empty space:

$$
\mathcal { O } ^ { ( \ell ) } = \left\{ \hat { \mathbf { p } } ^ { ( \ell ) } \in \hat { \mathcal { P } } ^ { ( \ell ) } \mid \hat { \mathbf { p } } ^ { ( \ell ) } \in \Omega _ { \mathrm { o c c } } , Y ( \hat { \mathbf { p } } ^ { ( \ell ) } ) = 0 \right\} .\tag{5}
$$

The containment loss is defined as

$$
\mathcal { L } _ { \mathrm { c o n } } = \frac { 1 } { \vert S \vert } \sum _ { \ell \in S } \frac { 1 } { \vert O ^ { ( \ell ) } \vert } \sum _ { \hat { \mathbf { p } } \in O ^ { ( \ell ) } } \rho _ { \beta } \left( d _ { \mathrm { b o x } } ( \hat { \mathbf { p } } , \mathcal { P } ^ { + } ) + m \right) ,\tag{6}
$$

where $s$ denotes the supervised decoder layers, $\rho _ { \beta }$ is the Smooth-L1 penalty, and m is a small positive offset that keeps the penalty active near occupied-box boundaries, encouraging predictions to move inside occupied voxels rather than stopping near the surface. By penalizing only predictions in supervised empty regions, the loss complements point-set reconstruction without suppressing valid occupied predictions or unsupervised space.

## 3.5 Training Objective

AdaOcc is trained with multi-layer supervision over semantic point predictions. At each decoder layer $\ell ,$ the predicted point set $\hat { \mathcal { P } } ^ { ( \ell ) }$ is supervised by semantic classification, point-set reconstruction, and containment regularization. The overall objective is

$$
\mathcal { L } = \sum _ { \ell = 1 } ^ { L } \gamma ^ { L - \ell } \left( \lambda _ { \mathrm { s e m } } \mathcal { L } _ { \mathrm { s e m } } ^ { ( \ell ) } + \lambda _ { \mathrm { p t s } } \mathcal { L } _ { \mathrm { p t s } } ^ { ( \ell ) } \right) + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } ,\tag{7}
$$

where $\mathcal { L } _ { \mathrm { s e m } } ^ { ( \ell ) }$ is focal loss (Lin et al., 2017b) over semantic logits, $ { \mathcal { L } } _ { \mathrm { { p t s } } } ^ { ( \ell ) }$ is DCD point reconstruction between $\hat { \mathcal { P } } ^ { ( \ell ) }$ and the occupied ground-truth point set ${ \mathcal { P } } ^ { + }$ , and ${ \mathcal { L } } _ { \mathrm { c o n } }$ is the proposed containment loss. The containment term is applied to selected late decoder layers in C.P.O. L denotes the number of decoder layers, $\gamma$ is the layer-wise decay factor, and $\lambda _ { \mathrm { { s e m } } } , \lambda _ { \mathrm { { p t s } } }$ , and $\lambda _ { \mathrm { c o n } }$ balance the semantic, reconstruction, and containment terms.

## 4 Experiments

## 4.1 Datasets and Evaluation Metrics

We evaluate AdaOcc on Occ-ScanNet, an indoor semantic occupancy prediction benchmark built upon RGB-D scans (Dai et al., 2017). The task requires predicting a semantic occupancy grid for a bounded local 3D volume from calibrated RGB observations. AdaOcc predicts over an agentcentric range that is discretized into a dense voxel grid of size $1 3 0 \times 1 2 0 \times 1 4 0$ with a voxel size of 0.08 m, which fully contains the official Occ-ScanNet evaluation grid of size $6 0 \times 6 0 \times 3 6$ at the same voxel size. Following the benchmark protocol, metrics are computed on this official grid, and predictions falling outside it are not evaluated. The semantic label space contains 11 occupied categories, including ceiling, floor, wall, window, chair, bed, sofa, table, tvs, furniture, and objects, together with an empty class; voxels with unknown ground-truth labels are ignored during evaluation.

We report results on both Occ-ScanNet and Occ-ScanNet-mini. The mini split is used for efficient comparison and ablation studies, while the full split is used to evaluate the final model against existing

Table 1: Local prediction performance on the Occ-ScanNet dataset. Rep. denotes the main scene representation: V for dense voxels, T for TPV features, G for Gaussians, and P for points or point queries. <sup>†</sup> indicates the AdaOcc variant with an EfficientNet image encoder. Bold and underline indicate the best and second-best results.
<table><tr><td rowspan=1 colspan=3>Dataset</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Rep.</td><td rowspan=1 colspan=1>mIoU</td><td rowspan=1 colspan=11>ceiing   00r         w   char              taqble         furureWallbousoamo   obeets</td><td rowspan=1 colspan=1>IoU</td></tr><tr><td rowspan=16 colspan=3>Occ-ScanNet</td><td rowspan=1 colspan=1>TPVFormer</td><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>24.94</td><td rowspan=1 colspan=5>6.96 32.9714.41 9.10 24.01</td><td rowspan=1 colspan=1>41.49</td><td rowspan=1 colspan=1>45.44</td><td rowspan=1 colspan=4>28.6110.6635.3725.31</td><td rowspan=1 colspan=1>33.39</td></tr><tr><td rowspan=2 colspan=1>MonoSceneISO</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>24.62</td><td rowspan=1 colspan=2>15.1744.71</td><td rowspan=1 colspan=1>22.41</td><td rowspan=1 colspan=1>12.55</td><td rowspan=1 colspan=1>26.11</td><td rowspan=1 colspan=1>27.03</td><td rowspan=1 colspan=1>35.91</td><td rowspan=1 colspan=3>28.326.5732.16</td><td rowspan=1 colspan=1>19.84</td><td rowspan=1 colspan=1>41.60</td></tr><tr><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>28.71</td><td rowspan=1 colspan=2>19.8841.88</td><td rowspan=1 colspan=1>22.37</td><td rowspan=1 colspan=1>16.98</td><td rowspan=1 colspan=1>29.09</td><td rowspan=1 colspan=1>42.43</td><td rowspan=1 colspan=1>42.00</td><td rowspan=1 colspan=3>29.6010.6236.36</td><td rowspan=1 colspan=1>24.61</td><td rowspan=1 colspan=1>42.16</td></tr><tr><td rowspan=1 colspan=1>SurroundOcc</td><td rowspan=1 colspan=1>ν</td><td rowspan=1 colspan=1>30.83</td><td rowspan=1 colspan=2>18.9049.30</td><td rowspan=1 colspan=1>24.80</td><td rowspan=1 colspan=1>18.00</td><td rowspan=1 colspan=1>26.80</td><td rowspan=1 colspan=1>42.00</td><td rowspan=1 colspan=1>44.10</td><td rowspan=1 colspan=3>32.9018.6036.80</td><td rowspan=1 colspan=1>26.90</td><td rowspan=1 colspan=1>42.52</td></tr><tr><td rowspan=1 colspan=1>GaussianFormer</td><td rowspan=1 colspan=1>g</td><td rowspan=1 colspan=1>29.93</td><td rowspan=1 colspan=2>20.7042.00</td><td rowspan=1 colspan=1>23.40</td><td rowspan=1 colspan=1>17.40</td><td rowspan=1 colspan=1>27.00</td><td rowspan=1 colspan=1>44.30</td><td rowspan=1 colspan=1>44.80</td><td rowspan=1 colspan=2>32.7015.30</td><td rowspan=1 colspan=1>36.70</td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=1>40.91</td></tr><tr><td rowspan=1 colspan=1>OPUS</td><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1>38.96</td><td rowspan=1 colspan=1>39.06</td><td rowspan=1 colspan=1>45.04</td><td rowspan=1 colspan=1>34.97</td><td rowspan=1 colspan=1>28.63</td><td rowspan=1 colspan=1>35.92</td><td rowspan=1 colspan=1>49.27</td><td rowspan=1 colspan=1>54.39</td><td rowspan=1 colspan=1>37.93</td><td rowspan=1 colspan=1>23.93</td><td rowspan=1 colspan=1>45.04</td><td rowspan=1 colspan=1>34.42</td><td rowspan=1 colspan=1>45.62</td></tr><tr><td rowspan=1 colspan=1>AdaSFormer</td><td rowspan=1 colspan=1>ν</td><td rowspan=1 colspan=1>45.33</td><td rowspan=1 colspan=1>37.66</td><td rowspan=1 colspan=1>57.34</td><td rowspan=1 colspan=1>40.51</td><td rowspan=1 colspan=1>29.49</td><td rowspan=1 colspan=1>43.29</td><td rowspan=1 colspan=1>60.41</td><td rowspan=1 colspan=1>63.08</td><td rowspan=1 colspan=1>47.05</td><td rowspan=1 colspan=1>29.63</td><td rowspan=1 colspan=1>54.08</td><td rowspan=1 colspan=1>36.15</td><td rowspan=1 colspan=1>54.63</td></tr><tr><td rowspan=2 colspan=1>EmbodiedOccEmbodiedOcc++</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>45.48</td><td rowspan=1 colspan=1>40.90</td><td rowspan=1 colspan=1>50.80</td><td rowspan=1 colspan=1>41.90</td><td rowspan=1 colspan=1>33.00</td><td rowspan=1 colspan=1>41.20</td><td rowspan=1 colspan=1>55.20</td><td rowspan=1 colspan=1>61.90</td><td rowspan=1 colspan=1>43.80</td><td rowspan=1 colspan=1>35.40</td><td rowspan=1 colspan=1>53.50</td><td rowspan=1 colspan=1>42.90</td><td rowspan=1 colspan=1>53.95</td></tr><tr><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>46.20</td><td rowspan=1 colspan=1>36.40</td><td rowspan=1 colspan=1>53.10</td><td rowspan=1 colspan=1>41.80</td><td rowspan=1 colspan=1>34.40</td><td rowspan=1 colspan=1>42.90</td><td rowspan=1 colspan=1>57.30</td><td rowspan=1 colspan=1>64.10</td><td rowspan=1 colspan=1>45.20</td><td rowspan=1 colspan=1>34.80</td><td rowspan=1 colspan=1>54.20</td><td rowspan=1 colspan=1>44.10</td><td rowspan=1 colspan=1>54.90</td></tr><tr><td rowspan=1 colspan=1>DiScene</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>47.17</td><td rowspan=1 colspan=1>45.21</td><td rowspan=1 colspan=1>50.63</td><td rowspan=1 colspan=1>40.38</td><td rowspan=1 colspan=1>36.73</td><td rowspan=1 colspan=1>42.28</td><td rowspan=1 colspan=1>59.68</td><td rowspan=1 colspan=1>62.04</td><td rowspan=1 colspan=1>45.60</td><td rowspan=1 colspan=1>41.17</td><td rowspan=1 colspan=1>52.42</td><td rowspan=1 colspan=1>42.72</td><td rowspan=1 colspan=1>51.99</td></tr><tr><td rowspan=1 colspan=1>RoboOcc</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>47.67</td><td rowspan=1 colspan=1>45.36</td><td rowspan=1 colspan=1>53.49</td><td rowspan=1 colspan=1>44.35</td><td rowspan=1 colspan=1>34.81</td><td rowspan=1 colspan=1>43.38</td><td rowspan=1 colspan=1>56.93</td><td rowspan=1 colspan=1>63.35</td><td rowspan=1 colspan=1>46.35</td><td rowspan=1 colspan=1>36.12</td><td rowspan=1 colspan=1>55.48</td><td rowspan=1 colspan=1>44.78</td><td rowspan=1 colspan=1>56.48</td></tr><tr><td rowspan=1 colspan=1>SplatSSC</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>51.83</td><td rowspan=1 colspan=1>49.10</td><td rowspan=1 colspan=1>59.00</td><td rowspan=1 colspan=1>48.30</td><td rowspan=1 colspan=1>38.80</td><td rowspan=1 colspan=1>47.40</td><td rowspan=1 colspan=1>62.40</td><td rowspan=1 colspan=1>67.00</td><td rowspan=1 colspan=1>49.50</td><td rowspan=1 colspan=1>42.60</td><td rowspan=1 colspan=1>60.70</td><td rowspan=1 colspan=1>45.40</td><td rowspan=1 colspan=1>62.83</td></tr><tr><td rowspan=1 colspan=1>GPOcc-DPT</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>51.88</td><td rowspan=1 colspan=1>51.42</td><td rowspan=1 colspan=1>50.35</td><td rowspan=1 colspan=1>46.97</td><td rowspan=1 colspan=1>41.84</td><td rowspan=1 colspan=1>46.98</td><td rowspan=1 colspan=1>60.39</td><td rowspan=1 colspan=1>66.16</td><td rowspan=1 colspan=1>50.51</td><td rowspan=1 colspan=1>47.97</td><td rowspan=1 colspan=1>58.88</td><td rowspan=1 colspan=1>49.23</td><td rowspan=1 colspan=1>56.96</td></tr><tr><td rowspan=1 colspan=1>GPOcc-VGGT</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>56.19</td><td rowspan=1 colspan=1>51.67</td><td rowspan=1 colspan=1>59.93</td><td rowspan=1 colspan=1>52.07</td><td rowspan=1 colspan=1>46.44</td><td rowspan=1 colspan=1>51.35</td><td rowspan=1 colspan=1>64.45</td><td rowspan=1 colspan=1>69.47</td><td rowspan=1 colspan=1>54.30</td><td rowspan=1 colspan=1>51.76</td><td rowspan=1 colspan=1>63.29</td><td rowspan=1 colspan=1>53.36</td><td rowspan=1 colspan=1>63.14</td></tr><tr><td rowspan=2 colspan=1>AdaOcc† (Ours)AdaOcc (Ours)</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>59.03</td><td rowspan=1 colspan=1>53.97</td><td rowspan=1 colspan=1>57.90</td><td rowspan=1 colspan=1>53.14</td><td rowspan=1 colspan=1>47.34</td><td rowspan=1 colspan=1>55.95</td><td rowspan=1 colspan=1>70.38</td><td rowspan=1 colspan=1>74.34</td><td rowspan=1 colspan=1>59.86</td><td rowspan=1 colspan=1>52.95</td><td rowspan=1 colspan=1>66.90</td><td rowspan=1 colspan=1>56.66</td><td rowspan=1 colspan=1>64.60</td></tr><tr><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>59.67</td><td rowspan=1 colspan=2>55.4058.50</td><td rowspan=1 colspan=1>53.63</td><td rowspan=1 colspan=1>48.00</td><td rowspan=1 colspan=1>56.81</td><td rowspan=1 colspan=1>70.82</td><td rowspan=1 colspan=1>74.83</td><td rowspan=1 colspan=1>60.89</td><td rowspan=1 colspan=1>53.15</td><td rowspan=1 colspan=1>67.22</td><td rowspan=1 colspan=1>57.17</td><td rowspan=1 colspan=1>65.29</td></tr><tr><td rowspan=3 colspan=3></td><td rowspan=2 colspan=1>MonoSceneISO</td><td rowspan=1 colspan=1>ν</td><td rowspan=1 colspan=1>25.90</td><td rowspan=1 colspan=2>17.0046.20</td><td rowspan=1 colspan=1>23.90</td><td rowspan=1 colspan=1>12.70</td><td rowspan=1 colspan=1>27.00</td><td rowspan=1 colspan=1>29.10</td><td rowspan=1 colspan=1>34.80</td><td rowspan=1 colspan=1>29.10</td><td rowspan=1 colspan=1>9.70</td><td rowspan=1 colspan=1>34.50</td><td rowspan=1 colspan=1>20.40</td><td rowspan=1 colspan=1>41.90</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>29.40</td><td rowspan=1 colspan=1>21.10</td><td rowspan=1 colspan=1>42.70</td><td rowspan=1 colspan=1>24.60</td><td rowspan=1 colspan=1>15.10</td><td rowspan=1 colspan=1>30.80</td><td rowspan=1 colspan=1>41.00</td><td rowspan=1 colspan=1>43.30</td><td rowspan=1 colspan=1>32.20</td><td rowspan=1 colspan=1>12.10</td><td rowspan=1 colspan=1>35.90</td><td rowspan=1 colspan=1>25.10</td><td rowspan=1 colspan=1>42.90</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>EmbodiedOcc</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>45.57</td><td rowspan=1 colspan=1>29.50</td><td rowspan=1 colspan=1>49.40</td><td rowspan=1 colspan=1>41.70</td><td rowspan=1 colspan=1>36.30</td><td rowspan=1 colspan=1>41.90</td><td rowspan=1 colspan=1>60.40</td><td rowspan=1 colspan=1>59.60</td><td rowspan=1 colspan=1>46.30</td><td rowspan=1 colspan=1>34.50</td><td rowspan=1 colspan=1>58.00</td><td rowspan=1 colspan=1>43.50</td><td rowspan=1 colspan=1>55.13</td></tr><tr><td rowspan=1 colspan=2>Occ-ScanNet-mini</td><td rowspan=1 colspan=1>et-mini</td><td rowspan=1 colspan=1>EmbodiedOcc++</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>48.20</td><td rowspan=1 colspan=1>23.30</td><td rowspan=1 colspan=1>51.00</td><td rowspan=1 colspan=1>42.80</td><td rowspan=1 colspan=1>39.30</td><td rowspan=1 colspan=1>43.50</td><td rowspan=1 colspan=1>65.60</td><td rowspan=1 colspan=1>64.00</td><td rowspan=1 colspan=1>50.70</td><td rowspan=1 colspan=1>40.70</td><td rowspan=1 colspan=1>60.30</td><td rowspan=1 colspan=1>48.90</td><td rowspan=1 colspan=1>55.70</td></tr><tr><td rowspan=3 colspan=3></td><td rowspan=1 colspan=1>SplatSSC</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>48.87</td><td rowspan=1 colspan=1>36.60</td><td rowspan=1 colspan=1>55.70</td><td rowspan=1 colspan=1>46.50</td><td rowspan=1 colspan=1>40.10</td><td rowspan=1 colspan=1>45.60</td><td rowspan=1 colspan=1>64.50</td><td rowspan=1 colspan=1>62.40</td><td rowspan=1 colspan=1>48.60</td><td rowspan=1 colspan=1>30.60</td><td rowspan=1 colspan=1>61.20</td><td rowspan=1 colspan=1>45.39</td><td rowspan=1 colspan=1>61.47</td></tr><tr><td rowspan=2 colspan=1>AdaOcc† (Ours)AdaOcc (Ours)</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>57.97</td><td rowspan=1 colspan=1>45.76</td><td rowspan=1 colspan=1>57.46</td><td rowspan=1 colspan=1>56.15</td><td rowspan=1 colspan=1>47.50</td><td rowspan=1 colspan=1>59.17</td><td rowspan=1 colspan=1>74.91</td><td rowspan=1 colspan=1>75.16</td><td rowspan=1 colspan=1>57.63</td><td rowspan=1 colspan=1>42.20</td><td rowspan=1 colspan=1>64.18</td><td rowspan=1 colspan=1>57.54</td><td rowspan=1 colspan=1>65.16</td></tr><tr><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>58.74</td><td rowspan=1 colspan=1>48.73</td><td rowspan=1 colspan=1>57.61</td><td rowspan=1 colspan=1>56.25</td><td rowspan=1 colspan=1>48.18</td><td rowspan=1 colspan=1>58.90</td><td rowspan=1 colspan=1>75.30</td><td rowspan=1 colspan=1>75.07</td><td rowspan=1 colspan=2>57.5846.52</td><td rowspan=1 colspan=2>64.3757.61</td><td rowspan=1 colspan=1>65.38</td></tr></table>

![](images/74fcdf6f7fa3774b9a449622dba570120fee899b174f9fe1980ff9259a8434a4.jpg)  
Figure 4: Qualitative comparison on Occ-ScanNet. AdaOcc produces cleaner and more complete semantic occupancy compared to the existing approaches.  
methods. Following the standard protocol, we use occupied IoU to measure binary occupancy quality and mean IoU (mIoU) over the 11 occupied semantic classes to evaluate semantic occupancy prediction. We also report per-class IoU to analyze category-level performance.

## 4.2 Experimental Setup

By default, AdaOcc uses C-RADIOv3-B (Heinrich et al., 2025) as the image encoder and a sparse 3D convolutional encoder for geometric feature extraction. Following SplatSSC (Qian et al., 2026), our benchmark experiments use depth maps estimated from the input RGB observations by Depth Anything v2 (Yang et al., 2024); the depth maps are back-projected into 3D points and used only for geometry-guided query initialization and geometric encoding. No ground-truth depth, point cloud, or occupancy information is used as input during inference. We quantify the effect of the depth estimator, including a second monocular estimator and ground-truth depth, in Appendix A.5. Training follows the objective described in Sec. 3, and predicted points are voxelized for standard evaluation. AdaOcc<sup>†</sup> denotes the variant that uses the same EfficientNet image encoder as SplatSSC (Tan and Le, 2019; Qian et al., 2026), replacing the default C-RADIOv3-B encoder for a controlled comparison.

![](images/9ba1d3d73bd959bb9a9af57d2b2f440541ffb05e6d6bec79a7c31a1ad3020060.jpg)  
Figure 5: Qualitative adaptability analysis on TartanGround. AdaOcc accepts geometric inputs from depth maps or LiDAR scans, and processes RGB observations with varying numbers of views jointly in a single forward pass.

## 4.3 Main Results

Table 1 summarizes the main results on the Occ-ScanNet and Occ-ScanNet-mini benchmarks. On the full Occ-ScanNet split, AdaOcc achieves state-of-the-art performance with 59.67 mIoU and 65.29 IoU. The strongest prior method, GPOcc-VGGT (Zhou et al., 2026a), reaches 56.19 mIoU and 63.14 IoU, so AdaOcc improves over it by 3.48 mIoU and 2.15 IoU even though GPOcc-VGGT uses the stronger VGGT geometry (Wang et al., 2025b) prior while AdaOcc estimates depth from RGB alone. For the controlled comparison we retain SplatSSC (Qian et al., 2026) as the primary baseline because it shares our setting, estimating depth from RGB with Depth-Anything-V2, and is the strongest method under this prior: it reaches 62.83 IoU, whereas the Depth-Anything-V2 variant of GPOcc obtains 56.96, and its IoU is within 0.31 of GPOcc-VGGT despite the latter using the much stronger VGGT prior. Swapping the geometry prior inside GPOcc changes its own IoU by 6.18, indicating that the advantage of the strongest competitor is largely attributable to its prior. Relative to SplatSSC, AdaOcc improves by 7.84 mIoU and 2.46 IoU; the larger gain in mIoU than in IoU indicates stronger semantic discrimination across indoor categories. To control for the effect of the image backbone, AdaOcc<sup>†</sup> uses the same EfficientNet image encoder as SplatSSC and still reaches 59.03 mIoU and 64.60 IoU, improving over SplatSSC by 7.20 mIoU and 1.77 IoU. This suggests that the gains mainly come from the progressive point decoder, the adaptive geometry-guided dual-branch encoder, and containment-guided point optimization, rather than from the image encoder. The qualitative results further support this observation: as shown in Fig. 4, AdaOcc produces cleaner object-level structures and more complete indoor layouts.

## 4.4 Ablation Study

Component Ablation Table 2 ablates the key components of AdaOcc on Occ-ScanNet-mini. Starting from the RADIO image encoder baseline, adding C.P.O. and progressive training substantially improves performance from 49.65 mIoU and 56.29 IoU to 55.50 mIoU and 62.92 IoU. Adding geometric initialization further improves the result to 57.06 mIoU and 64.72 IoU, showing the benefit of sensor-guided spatial anchors. With the full 3D encoder,

Table 2: Ablation study of key components in AdaOcc.
<table><tr><td>Image Encoder</td><td>Geometric Initialization</td><td>3D Encoder</td><td>C.P.O.</td><td>Progressive Training</td><td>mIoU</td><td>IoU</td></tr><tr><td rowspan="6">RADIO</td><td>一</td><td></td><td></td><td>一</td><td>49.65</td><td>56.29</td></tr><tr><td></td><td></td><td>√</td><td>√</td><td></td><td>55.50 62.92</td></tr><tr><td>√</td><td>1</td><td>√</td><td>√</td><td></td><td>57.06 64.72</td></tr><tr><td>√</td><td>√</td><td></td><td>√</td><td></td><td>54.54 61.31</td></tr><tr><td>√</td><td>√</td><td>√</td><td>一</td><td></td><td>54.63 62.35</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>58.74 65.38</td></tr><tr><td>EffNet</td><td>√</td><td>√</td><td>√</td><td>√</td><td>57.97</td><td>65.16</td></tr></table>

AdaOcc reaches 58.74 mIoU and 65.38 IoU, indicating that explicit geometric features provide

Table 3: Sensitivity to training hyperparameters on Occ-ScanNet-mini. All other settings are fixed within each study, and bold marks the default configuration. For the query schedule, X/Y denote adding X queries every Y epochs, from X up to 500.
<table><tr><td>Parameter</td><td>Setting</td><td>IoU</td><td>mIoU</td></tr><tr><td>Margin m</td><td>0.00 / 0.04 / 0.08</td><td>64.25 / 65.38 / 65.32</td><td>57.47 / 58.74 / 58.12</td></tr><tr><td>C.P.O. weight</td><td> $0 . 5 \times / 1 \times / 2 \times$ </td><td>65.19 / 65.38 / 65.32</td><td>58.16 / 58.74 / 58.17</td></tr><tr><td>Decay γ</td><td> $0 . 8 0 / { \bf 0 . 9 0 } / 0 . 9 5$ </td><td>65.51 / 65.38 / 65.35</td><td>58.62 / 58.74 / 58.37</td></tr><tr><td>Query schedule</td><td> $\mathbf { - } / 5 0 / 2 0 / 1 0 0 / 4 0$ </td><td>62.35 / 65.17 / 65.38</td><td>54.63 / 58.15 / 58.74</td></tr></table>

Table 4: Efficiency and adaptability analysis on the Occ-ScanNet-mini dataset using a single NVIDIA H20 GPU. All results use the same EfficientNet image encoder as SplatSSC for a controlled comparison. (a) Query-number ablation for AdaOcc<sup>†</sup>. (b) Anchor-number ablation for SplatSSC. (c) Decoder-layer output ablation for AdaOcc<sup>†</sup>. Within each panel only the listed factor is varied while all other settings are fixed.
<table><tr><td>Query</td><td>Time↓ (ms)</td><td>Mem.↓ (MiB)</td><td>mIoU</td><td>IoU</td></tr><tr><td>200</td><td>63.2</td><td>1710.7</td><td>55.13</td><td>62.36</td></tr><tr><td>500</td><td>64.4</td><td>1710.6</td><td>57.97</td><td>65.16</td></tr><tr><td>800</td><td>73.3</td><td>1798.8</td><td>58.32</td><td>64.58</td></tr><tr><td>1000</td><td>75.9</td><td>1863.7</td><td>58.25</td><td>64.29</td></tr></table>

<table><tr><td colspan="4">(b) Anchor (SplatSSC)</td></tr><tr><td>Anchor</td><td>|Time↓ (ms)</td><td>Mem.↓ (MiB)</td><td>mIoU IoU</td></tr><tr><td>600</td><td>91.0</td><td>1837.2</td><td>43.30 53.75</td></tr><tr><td>1131</td><td>85.1</td><td>1826.1</td><td>48.46 60.86</td></tr><tr><td>1911</td><td>85.0</td><td>1835.1 45.26</td><td>58.85</td></tr><tr><td>2596</td><td>85.6</td><td>1830.3</td><td>42.95 56.33</td></tr></table>

<table><tr><td>Layer</td><td>Time↓ (ms)</td><td>Mem.↓ (MiB)</td><td>mIoU</td><td>IoU</td></tr><tr><td>3</td><td>58.2</td><td>1254.6</td><td>49.46</td><td>54.92</td></tr><tr><td>4</td><td>59.9</td><td>1405.9</td><td>54.51</td><td>60.96</td></tr><tr><td>5</td><td>62.5</td><td>1556.4</td><td>56.80</td><td>63.67</td></tr><tr><td>6</td><td>64.4</td><td>1710.6</td><td>57.97</td><td>65.16</td></tr></table>

complementary spatial grounding for point-based occupancy prediction. Removing either C.P.O. or progressive training causes clear performance drops, verifying the contribution of containment-guided optimization and budget-aware training. The EfficientNet variant remains competitive, confirming that the gains are not solely due to the image backbone.

Hyperparameter sensitivity. Table 3 varies the principal training hyperparameters on Occ-ScanNetmini with all other settings fixed. The offset margin trades recall for precision: setting m = 0 lowers IoU and mIoU by 1.13 and 1.27 points, while increasing it to 0.08 changes them by only 0.06 and 0.62 points, so a small positive margin suffices. Halving or doubling the C.P.O. weight schedule changes IoU and mIoU by at most 0.19 and 0.58 points, and varying the decoder decay factor γ between 0.80 and 0.95 changes them by at most 0.13 and 0.37 points. By contrast, removing progressive training costs 3.03 IoU and 4.11 mIoU, while using a finer query schedule changes the result by only 0.21 IoU and 0.59 mIoU. Overall, these results show that AdaOcc is robust to moderate variations of its loss-related hyperparameters.

## 4.5 Efficiency and Adaptability

Table 4 evaluates the efficiency-adaptability trade-off on Occ-ScanNet-mini using a single NVIDIA H20 GPU. We use SplatSSC (Qian et al., 2026) as the main efficiency baseline, since it reports lower latency and memory usage than EmbodiedOcc (Wu et al., 2025) in its Occ-ScanNet-mini comparison. For a controlled comparison with SplatSSC, all results use the same EfficientNet image encoder, corresponding to AdaOcc<sup>†</sup>. AdaOcc<sup>†</sup> is trained with a progressive query schedule from 100 to 500 queries, while SplatSSC is trained with 1131 anchors. When varying the query budget, AdaOcc remains stable across a wide range of query numbers and provides a stronger accuracy-latency trade-off than SplatSSC. For example, AdaOcc with 500 queries achieves 57.97 mIoU and 65.16 IoU at 64.4 ms, whereas the best SplatSSC setting obtains 48.46 mIoU and 60.86 IoU at 85.1 ms. The decoder-layer ablation further shows that the same trained model can produce valid intermediate predictions without retraining, enabling flexible latency-memory-accuracy trade-offs by selecting different decoder depths.

Beyond computational adaptability, Fig. 5 shows that AdaOcc can handle heterogeneous geometric inputs and flexible multi-view observations on TartanGround (Patel et al., 2025). The same trained model produces coherent reconstructions with geometric inputs from either depth maps or LiDAR scans, and processes different numbers of RGB views jointly in a single forward pass rather than stitching per-view predictions.

![](images/b9c75e96cd8a9764aff5052ad9a08be5339df066bae700b8aa8935e055942060.jpg)  
Figure 6: Real-world embodied applications of AdaOcc. AdaOcc provides semantic 3D occupancy for navigation, manipulation, and mobile manipulation. The semantic prompts are plant for navigation, bottle and basket for manipulation, and trash and basket for mobile manipulation.

## 4.6 Embodied Applications

To further examine the practical applicability of AdaOcc, we deploy it in real-world indoor embodied systems for navigation, manipulation, and mobile manipulation. As shown in Fig. 6, AdaOcc reconstructs semantic 3D occupancy from robot observations and provides structured spatial cues for downstream planning and interaction. In navigation, the reconstructed occupancy supports target-aware path planning; in manipulation, it localizes task-relevant objects and receptacles; and in mobile manipulation, it supports navigation toward objects and subsequent placement. These demonstrations suggest that AdaOcc can serve as a practical 3D perception module beyond offline benchmark evaluation. More real-robot demonstrations are available at https://wangjl-nb. github.io/AdaOcc\_web/, and a closed-loop navigation study with quantitative results is reported in Appendix B.2.

## 5 Conclusions

We presented AdaOcc, a unified point-based framework for adaptive 3D occupancy prediction in embodied scenes. AdaOcc normalizes geometric cues from estimated depth maps, depth-camera measurements, or LiDAR scans into a common point-set interface, and combines them with RGB observations from varying numbers of views through an adaptive geometry-guided dual-branch encoder. Its progressive point decoder supports budget-adaptive inference by changing query numbers or using intermediate decoder outputs, while the containment loss regularizes predicted points to reside within valid occupied regions and improves spatial consistency. Experiments on Occ-ScanNet and Occ-ScanNet-mini demonstrate state-of-the-art performance, strong efficiency-adaptability tradeoffs, and practical deployment in a real robot navigation system. Future work will focus on improving the reconstruction of small and thin objects in cluttered embodied scenes.

## Acknowledgments and Disclosure of Funding

Funding. This work was supported by the National Natural Science Foundation of China (No. 62461160331 and No. 62132001).

Competing interests. The first three authors conducted this work during an internship at XYZ Embodied AI. The authors declare no other competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

Yingjie Cai, Xuesong Chen, Chao Zhang, Kwan-Yee Lin, Xiaogang Wang, and Hongsheng Li. Semantic scene completion via integrating instances and scene in-the-loop. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 324–333, 2021.

Anh-Quan Cao and Raoul De Charette. Monoscene: Monocular 3d semantic scene completion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3991–4001, 2022.

Xiaokang Chen, Kwan-Yee Lin, Chen Qian, Gang Zeng, and Hongsheng Li. 3d sketch-aware semantic scene completion via semi-supervised structure prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4193–4202, 2020.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

Haoqiang Fan, Hao Su, and Leonidas J Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 605–613, 2017.

Hao-Shu Fang, Chenxi Wang, Minghao Gou, and Cewu Lu. Graspnet-1billion: A large-scale benchmark for general object grasping. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11444–11453, 2020.

Martin Garbade, Yueh-Tung Chen, Johann Sawatzky, and Juergen Gall. Two stream 3d semantic scene completion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 0–0, 2019.

Peter E Hart, Nils J Nilsson, and Bertram Raphael. A formal basis for the heuristic determination of minimum cost paths. IEEE transactions on Systems Science and Cybernetics, 4(2):100–107, 1968.

Greg Heinrich, Mike Ranzinger, Hongxu Yin, Yao Lu, Jan Kautz, Andrew Tao, Bryan Catanzaro, and Pavlo Molchanov. Radiov2. 5: Improved baselines for agglomerative vision foundation models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 22487–22497, 2025.

Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Tri-perspective view for vision-based 3d semantic occupancy prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9223–9232, 2023.

Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Gaussianformer: Scene as gaussians for vision-based 3d semantic occupancy prediction. In European Conference on Computer Vision, pages 376–393. Springer, 2024.

Yuanhui Huang, Amonnut Thammatadatrakoon, Wenzhao Zheng, Yunpeng Zhang, Dalong Du, and Jiwen Lu. Gaussianformer-2: Probabilistic gaussian superposition for efficient 3d occupancy prediction. In Proceedings of the computer vision and pattern recognition conference, pages 27477–27486, 2025.

Yupeng Jia, Jie He, Runze Chen, Fang Zhao, and Haiyong Luo. Occupancydetr: Using detr for mixed dense-sparse 3d occupancy prediction. arXiv preprint arXiv:2309.08504, 2023.

Jianing Li, Ming Lu, Juntao Liu, Hao Wang, Chenyang Gu, Wenzhao Zheng, Li Du, and Shanghang Zhang. Sliceocc: Indoor 3d semantic occupancy prediction with vertical slice representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 15762–15768. IEEE, 2025.

Jie Li, Yu Liu, Xia Yuan, Chunxia Zhao, Roland Siegwart, Ian Reid, and Cesar Cadena. Depth based semantic scene completion with position importance aware loss. IEEE Robotics and Automation Letters, 5(1):219–226, 2019.

Jinke Li, Xiao He, Chonghua Zhou, Xiaoqiang Cheng, Yang Wen, and Dan Zhang. Viewformer: Exploring spatiotemporal modeling for multi-view 3d occupancy perception via view-guided transformers. In European Conference on Computer Vision, pages 90–106. Springer, 2024a.

Yiming Li, Zhiding Yu, Christopher Choy, Chaowei Xiao, Jose M Alvarez, Sanja Fidler, Chen Feng, and Anima Anandkumar. Voxformer: Sparse voxel transformer for camera-based 3d semantic scene completion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9087–9098, 2023a.

Zhiqi Li, Zhiding Yu, David Austin, Mingsheng Fang, Shiyi Lan, Jan Kautz, and Jose M Alvarez. Fb-occ: 3d occupancy prediction based on forward-backward view transformation. arXiv preprint arXiv:2307.01492, 2023b.

Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(3):2020–2036, 2024b.

Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017a.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017b.

Yuhang Lu, Xinge Zhu, Tai Wang, and Yuexin Ma. Octreeocc: Efficient and multi-granularity occupancy prediction using octree queries. Advances in Neural Information Processing Systems, 37:79618–79641, 2024.

Qihang Ma, Xin Tan, Yanyun Qu, Lizhuang Ma, Zhizhong Zhang, and Yuan Xie. Cotr: Compact occupancy transformer for vision-based 3d occupancy prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19936–19945, 2024.

Mingjie Pan, Jiaming Liu, Renrui Zhang, Peixiang Huang, Xiaoqi Li, Hongwei Xie, Bing Wang, Li Liu, and Shanghang Zhang. Renderocc: Vision-centric 3d occupancy prediction with 2d rendering supervision. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 12404–12411. IEEE, 2024.

Manthan Patel, Fan Yang, Yuheng Qiu, Cesar Cadena, Sebastian Scherer, Marco Hutter, and Wenshan Wang. Tartanground: A large-scale dataset for ground robot perception and navigation. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 20524–20531. IEEE, 2025.

Rui Qian, Haozhi Cao, Tianchen Deng, Shenghai Yuan, and Lihua Xie. Splatssc: Decoupled depthguided gaussian splatting for semantic scene completion. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 8520–8528, 2026.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Luis Roldao, Raoul De Charette, and Anne Verroust-Blondet. Lmscnet: Lightweight multiscale 3d semantic completion. In 2020 International Conference on 3D Vision (3DV), pages 111–119. IEEE, 2020.

Shuran Song, Fisher Yu, Andy Zeng, Angel X Chang, Manolis Savva, and Thomas Funkhouser. Semantic scene completion from a single depth image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1746–1754, 2017.

Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.

Pin Tang, Zhongdao Wang, Guoqing Wang, Jilai Zheng, Xiangxuan Ren, Bailan Feng, and Chao Ma. Sparseocc: Rethinking sparse latent representation for vision-based semantic occupancy prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15035–15044, 2024.

Wenwen Tong, Chonghao Sima, Tai Wang, Li Chen, Silei Wu, Hanming Deng, Yi Gu, Lewei Lu, Ping Luo, Dahua Lin, et al. Scene as occupancy. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 8406–8415, 2023.

Hao Wang, Xiaobao Wei, Xiaoan Zhang, Jianing Li, Chengyu Bai, Ying Li, Ming Lu, Wenzhao Zheng, and Shanghang Zhang. Embodiedocc++: Boosting embodied 3d occupancy prediction with plane regularization and uncertainty sampler. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 925–934, 2025a.

Jiabao Wang, Zhaojiang Liu, Qiang Meng, Liujiang Yan, Ke Wang, Jie Yang, Wei Liu, Qibin Hou, and Ming-Ming Cheng. Opus: occupancy prediction using a sparse set. Advances in Neural Information Processing Systems, 37:119861–119885, 2024a.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5294–5306. IEEE, 2025b.

Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, et al. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19757–19767, 2024b.

Xuzhi Wang, Xinran Wu, Song Wang, Lingdong Kong, and Ziping Zhao. Adasformer: Adaptive serialized transformers for monocular semantic scene completion from indoor environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 34148–34159, 2026.

Yuqi Wang, Yuntao Chen, Xingyu Liao, Lue Fan, and Zhaoxiang Zhang. Panoocc: Unified occupancy representation for camera-based 3d panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17158–17168, 2024c.

Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, and Jiwen Lu. Surroundocc: Multicamera 3d occupancy prediction for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21729–21740, 2023.

Tong Wu, Liang Pan, Junzhe Zhang, Tai Wang, Ziwei Liu, and Dahua Lin. Density-aware chamfer distance as a comprehensive metric for point cloud completion. In Proceedings of the 35th International Conference on Neural Information Processing Systems, pages 29088–29100, 2021.

Yuqi Wu, Wenzhao Zheng, Sicheng Zuo, Yuanhui Huang, Jie Zhou, and Jiwen Lu. Embodiedocc: Embodied 3d occupancy prediction for vision-based online scene understanding. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 26360–26370, 2025.

Xiuwei Xu, Chong Xia, Ziwei Wang, Linqing Zhao, Yueqi Duan, Jie Zhou, and Jiwen Lu. Memorybased adapters for online 3d scene perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21604–21613, 2024.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024.

Jiawei Yao, Chuming Li, Keqiang Sun, Yingjie Cai, Hao Li, Wanli Ouyang, and Hongsheng Li. Ndc-scene: Boost monocular 3d semantic scene completion in normalized device coordinates space. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 9421–9431. IEEE, 2023.

Hongxiao Yu, Yuqi Wang, Yuntao Chen, and Zhaoxiang Zhang. Monocular occupancy prediction for scalable indoor scenes. In European Conference on Computer Vision, pages 38–54. Springer, 2024.

Zichen Yu, Changyong Shu, Jiajun Deng, Kangjie Lu, Zongdai Liu, Jiangyong Yu, Dawei Yang, Hui Li, and Yan Chen. Flashocc: Fast and memory-efficient occupancy prediction via channel-to-height plugin. arXiv preprint arXiv:2311.12058, 2023.

Yunpeng Zhang, Zheng Zhu, and Dalong Du. Occformer: Dual-path transformer for vision-based 3d semantic occupancy prediction. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 9433–9443, 2023.

Zhang Zhang, Qiang Zhang, Wei Cui, Shuai Shi, Yijie Guo, Gang Han, Wen Zhao, Hengle Ren, Renjing Xu, and Jian Tang. Roboocc: Enhancing the geometric and semantic scene understanding for robots. arXiv preprint arXiv:2504.14604, 2025.

Changqing Zhou, Yueru Luo, and Changhao Chen. Generalizing visual geometry priors to sparse gaussian occupancy prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28578–28587, 2026a.

Changqing Zhou, Yueru Luo, Han Zhang, Zeyu Jiang, and Changhao Chen. Monocular open vocabulary occupancy prediction for indoor scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21627–21637, 2026b.

## Supplemental Material

## A Additional Experiments

## A.1 Open-Vocabulary Semantic Occupancy Prediction

We further adapt AdaOcc to the open-vocabulary semantic occupancy setting. Different from conventional semantic occupancy prediction, which predicts a closed-set class label for each occupied voxel, our open-vocabulary occupancy prediction represents each occupied voxel with a text-aligned feature. This formulation enables arbitrary language queries over 3D occupied space, allowing semantically related concepts to activate consistent regions. This makes the occupancy representation more flexible for downstream embodied tasks, where the queried object category may be expressed with synonyms or fine-grained natural-language descriptions.

To achieve this, we modify AdaOcc with a CLIP-aligned semantic feature branch. The original image backbone is replaced by a frozen OpenAI CLIP RN101 visual encoder Radford et al. (2021). We directly connect the multi-stage ResNet features to an FPN Lin et al. (2017a). The decoder is kept query-based, but each query predicts two outputs: a binary occupancy score and a semantic feature. The binary occupancy head models whether a query corresponds to occupied space, while the feature head maps occupied queries into the CLIP text embedding space.

For semantic supervision, we pre-compute a CLIP text prototype bank for the TartanGround vocabulary. For each class name $c ,$ we encode multiple prompt templates using the CLIP RN101 text encoder and average the normalized embeddings:

$$
\mathbf { t } _ { c } = \mathrm { N o r m } \left( \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathrm { N o r m } \left( E _ { \mathrm { t e x t } } ( \pi _ { m } ( c ) ) \right) \right) ,
$$

where $\pi _ { m } ( \cdot )$ denotes a prompt template and $E _ { \mathrm { t e x t } }$ is the CLIP text encoder. During training, each occupied query matched to a ground-truth semantic voxel with label $y _ { i }$ is supervised by the corresponding prototype $\mathbf { t } _ { y _ { i } }$

The overall training loss is

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { o c c } } \mathcal { L } _ { \mathrm { o c c } } + \lambda _ { \mathrm { p t s } } \mathcal { L } _ { \mathrm { p t s } } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } + \lambda _ { \mathrm { f e a t } } \mathcal { L } _ { \mathrm { f e a t } } . } \end{array}
$$

The semantic feature loss is applied only to positive occupied queries:

$$
\mathcal { L } _ { \mathrm { f e a t } } = \lambda _ { \mathrm { c o s } } \left( 1 - \cos ( \mathbf { f } _ { i } , \mathbf { t } _ { y _ { i } } ) \right) + \lambda _ { \mathrm { c e } } \mathrm { C E } \left( \frac { \mathbf { f } _ { i } \mathbf { T } ^ { \top } } { \tau } , y _ { i } \right) ,
$$

where $\mathbf { f } _ { i }$ is the predicted voxel feature, T is the CLIP prototype bank, and τ is the temperature. We use cosine alignment as the main supervision and a lightweight prototype classification term to stabilize the embedding space. Class-balanced weights are used to reduce the dominance of frequent classes, while rare categories are assigned stronger weights according to voxel statistics.

We train this open-vocabulary variant on the TartanGround indoor occupancy dataset Patel et al. (2025). The voxel size is 0.08m, and the model is trained for 200 epochs with the CLIP RN101 image encoder frozen. Only the FPN, occupancy decoder, binary occupancy head, and semantic feature head are optimized. At inference time, an arbitrary text query q is encoded by the same CLIP text encoder, and each occupied voxel is scored by cosine similarity:

$$
s _ { i } ( \ v q ) = \cos \left( \mathrm { N o r m } ( \mathbf { f } _ { i } ) , \mathrm { N o r m } ( E _ { \mathrm { t e x t } } ( \ v q ) ) \right) .
$$

Qualitatively, the trained model shows clear open-vocabulary behavior in 3D occupancy prediction. Given different but semantically related text queries, the CLIP-aligned voxel features tend to activate the same object regions in the reconstructed scene. For example, synonyms of objects such as beds and mattress produce consistent high-response areas. We visualize this by highlighting the top activated occupied points in red and rendering the remaining points in gray, as shown in Fig. 7.

## A.2 Flexible Multi-View and Geometric Input

During open-vocabulary occupancy training, we further introduce a flexible input setting that supports varying numbers of camera views and geometric points from different sources, as illustrated in Fig. 5.

FRONT  
BACK  
LEFT  
RIGHT  
![](images/4116c5ce969f929d4605b9a447157656fed700e42d84f9a07d13cfd286592ab9.jpg)  
Figure 7: Open-vocabulary semantic activation in reconstructed 3D occupancy. Semantically related text queries activate consistent object regions, demonstrating that the learned CLIP-aligned voxel features can respond to free-form language.

Table 5: Comparison of the GT-depth and LiDAR input variants on the TartanGround test set.
<table><tr><td>Variant</td><td>Input Source</td><td>IoU (%)</td><td>mIoU (%)</td></tr><tr><td>GT-depth version</td><td>Depth-derived points</td><td>67.55</td><td>26.13</td></tr><tr><td>LiDAR version</td><td>LiDAR points</td><td>55.65</td><td>19.40</td></tr></table>

Although this is used in our open-vocabulary experiments, the same training strategy is general and can be applied to a broader range of experimental settings.

To enable this flexibility, we use four cameras, including the front, back, left, and right views. During training, we randomly select a subset of the available camera views as input, with the number of selected views ranging from one to four. The model therefore sees different view combinations during training and can be evaluated with one, two, three, or four input views at inference time.

The geometric input can be provided either by depth-derived points or by LiDAR points. For the RGB-D setting, points are generated from the depth maps of the selected camera views. For the RGB-LiDAR setting, the raw LiDAR point cloud is used as the geometric input instead. When only a subset of camera views is selected, the same LiDAR point cloud is paired with the selected RGB views, while the image features are computed only from those views. The point loading and deduplication pipeline is adjusted accordingly, but the occupancy decoder and prediction heads remain unchanged.

With this design, the same training framework supports randomly sampled one-to-four-view input under both RGB-D and RGB-LiDAR settings. The model can thus reconstruct occupancy from different numbers of camera views while using either depth-based points or LiDAR points as geometric support. We compare two settings on TartanGround test set. The results are shown in Table 5.

Table 6 reports the occupancy quality under one to four input views on 1,379 TartanGround test samples from 12 scenes (ground-truth depth). The model is trained once with randomly sampled view subsets and evaluated without retraining. Within the same observed regions, the reconstruction quality changes only slightly as the number of input views varies, indicating that AdaOcc is robust to different view configurations. Additional views mainly expand spatial coverage: under the fixed four-view union, the IoU increases from 17.24 with a single view to 67.55 with four views.

Table 6: Occupancy IoU / mIoU on the TartanGround test set under one to four input views, using the model trained with randomly sampled view subsets and evaluated without retraining. “Current-view union” evaluates the union of the regions covered by the views currently provided as input, whereas “Four-view union” uses the fixed union covered by all four cameras.
<table><tr><td>Input views</td><td>Front</td><td>Left</td><td>Back</td><td>Right</td><td>Current-view union</td><td>Four-view union</td></tr><tr><td>Front</td><td>62.87 / 23.24</td><td></td><td></td><td>一</td><td>62.87 / 23.24</td><td>17.24 / 6.93</td></tr><tr><td>Front + Left</td><td>65.98 / 25.56</td><td>64.43 / 23.95</td><td></td><td></td><td>65.29 / 25.06</td><td>32.34 / 13.29</td></tr><tr><td>Front + Left + Back</td><td>65.88 / 25.15</td><td>64.99 / 24.37</td><td>67.24 / 25.26</td><td></td><td>66.14 /25.04</td><td>52.14 / 19.18</td></tr><tr><td>Four views</td><td>67.57 / 26.55</td><td>66.33 / 25.18</td><td>68.63 / 26.49</td><td>67.29 / 27.03</td><td>67.55 / 26.14</td><td>67.55 / 26.14</td></tr></table>

## A.3 Occ3D-nuScenes Dataset Results

We conduct outdoor 3D semantic occupancy prediction experiments on Occ3D-nuScenes dataset. Compared with other SOTA methods, we achieve 3.8 RayIoU improvement. With LiDAR and surrounding RGB inputs, we have batter performance with 50.9 RayIoU, showing that we can utilize more geometry information and support different sensor and view inputs.

## A.4 Hyperparameter Sensitivity

We study the sensitivity of AdaOcc to its main training hyperparameters on Occ-ScanNet-mini. All variants keep the remaining settings fixed and are trained for 200 epochs.

Margin and C.P.O. weight. Table 8 reports the effect of the C.P.O. offset margin m and of the C.P.O. loss weight schedule. A positive margin converts a moderate recall reduction (87.88 → 83.41) into a clear precision gain $( 7 0 . 5 0  7 5 . 3 2 )$ , and m = 0.04 attains the best IoU, mIoU, and F1 (79.16, against 78.24 for $m = 0$ and 79.02 for $m = 0 . 0 8 )$ . Halving or doubling the weight schedule changes IoU and mIoU by at most 0.19 and 0.58 points, respectively.

Decay factor and query schedule. Table 9 varies the decoder loss decay factor γ and the granularity of the progressive query schedule. Both affect the final result only mildly: within $\gamma \in [ 0 . 8 0 , 0 . 9 5 ]$ the variation is at most 0.13 IoU and 0.37 mIoU, and using a finer query schedule instead of the default changes the result by only 0.21 IoU and 0.59 mIoU. Removing progressive training, by contrast, costs 3.03 IoU and 4.11 mIoU, indicating that the prediction budget, rather than the loss coefficients, is the dominant factor.

Table 7: Comparison of RayIoU results on the Occ3D-nuScenes dataset.
<table><tr><td>Methods</td><td> $\mathbf { R a y I o U } _ { 1 m }$ </td><td> $\mathbf { R a y I o U } _ { 2 m }$ </td><td> $\mathbf { R a y I o U } _ { 4 m }$ </td><td>RayIoU</td></tr><tr><td>RenderOcc (Pan et al., 2024)</td><td>13.1</td><td>19.6</td><td>25.5</td><td>19.5</td></tr><tr><td>BEVFormer (Li et al., 2024b)</td><td>26.1</td><td>32.9</td><td>38.0</td><td>32.4</td></tr><tr><td>BEVDet-Occ</td><td>23.6</td><td>30.0</td><td>35.1</td><td>29.6</td></tr><tr><td>BEVDet-Occ (8f)</td><td>26.6</td><td>33.1</td><td>38.2</td><td>32.6</td></tr><tr><td>FB-Occ (16f) (Li et al., 2023b)</td><td>26.7</td><td>34.1</td><td>39.7</td><td>33.5</td></tr><tr><td>SparseOcc (8f) (Tang et al., 2024)</td><td>28.0</td><td>34.7</td><td>39.4</td><td>34.0</td></tr><tr><td>SparseOcc (16f) (Tang et al., 2024)</td><td>29.1</td><td>35.8</td><td>40.3</td><td>35.1</td></tr><tr><td>OPUS-T (8f) (Wang et al., 2024a)</td><td>31.7</td><td>39.2</td><td>44.3</td><td>38.4</td></tr><tr><td>OPUS-S (8f) (Wang et al., 2024a)</td><td>32.6</td><td>39.9</td><td>44.7</td><td>39.1</td></tr><tr><td>OPUS-M (8f) (Wang et al., 2024a)</td><td>33.7</td><td>41.1</td><td>46.0</td><td>40.3</td></tr><tr><td>OPUS-L (8f) (Wang et al., 2024a)</td><td>34.7</td><td>42.1</td><td>46.7</td><td>41.2</td></tr><tr><td>AdaOcc (Ours) (1f) (RGB+Pseudo depth map)</td><td>37.55</td><td>46.14</td><td>51.20</td><td>44.97</td></tr><tr><td>AdaOcc (Ours) (1f) (RGB+LiDAR)</td><td>46.60</td><td>51.58</td><td>54.48</td><td>50.89</td></tr></table>

Table 8: Effect of the C.P.O. offset margin m and of the C.P.O. loss weight schedule on Occ-ScanNetmini. Bold marks the default configuration.
<table><tr><td>Study</td><td>Setting</td><td>IoU</td><td>mIoU</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td rowspan="3">Margin m</td><td>0.00</td><td>64.25</td><td>57.47</td><td>70.50</td><td>87.88</td><td>78.24</td></tr><tr><td>0.04</td><td>65.38</td><td>58.74</td><td>75.32</td><td>83.41</td><td>79.16</td></tr><tr><td>0.08</td><td>65.32</td><td>58.12</td><td>75.36</td><td>83.06</td><td>79.02</td></tr><tr><td rowspan="3">C.P.O. weight</td><td> $0 . 0 5  0 . 1 5$ </td><td>65.19</td><td>58.16</td><td>74.32</td><td>84.14</td><td>一</td></tr><tr><td> ${ \bf 0 . 1 0 }  { \bf 0 . 3 0 }$ </td><td>65.38</td><td>58.74</td><td>75.32</td><td>83.41</td><td>一</td></tr><tr><td> $0 . 2 0  0 . 6 0$ </td><td>65.32</td><td>58.17</td><td>75.17</td><td>83.29</td><td>一</td></tr></table>

## A.5 Depth Estimator Sensitivity

AdaOcc estimates depth from RGB with Depth Anything v2 (DAv2) by default. To quantify the dependence on this external component, we replace DAv2 with a second monocular estimator, MoGe-2, and, separately, replace the estimated depth with ground-truth depth, keeping the architecture, the evaluation protocol, and all training settings unchanged. We first measure depth quality over all 6,646 unique Occ-ScanNet-mini frames: both estimators process every frame, and predictions are resized to the ground-truth resolution and evaluated over valid depths.

Occupancy accuracy follows depth quality consistently: replacing DAv2 with the less accurate MoGe-2 reduces IoU and mIoU by 3.06 and 2.59 points, whereas ground-truth depth improves them by a further 4.11 and 4.32 points. The remaining headroom therefore lies in the external depth estimator, whose errors concentrate around reflective or low-texture surfaces, thin structures, and occluded boundaries. The DAv2 entry of Table 10 stems from the run shared with the MoGe-2 and ground-truth variants and differs from the DAv2 results of Tables 2 and 3 (65.38 IoU / 58.74 mIoU) by training stochasticity.

## B Embodied System

We use different robots and perception sensors to conduct real-world embodied tasks, as listed in Table 12. The devices used in these tasks are shown in Fig. 8. The details of each task are as following:

Table 9: Effect of the decoder loss decay factor γ and of the progressive query schedule on Occ-ScanNet-mini. The query schedule is written as queries added per epoch interval; bold marks the default configuration.
<table><tr><td>Study</td><td>Setting</td><td>IoU</td><td>mIoU</td></tr><tr><td rowspan="2">Decay γ</td><td>0.80</td><td>65.51</td><td>58.62</td></tr><tr><td>0.90 0.95</td><td>65.38 65.35</td><td>58.74 58.37</td></tr><tr><td rowspan="2">Query schedule</td><td>none</td><td>62.35</td><td>54.63</td></tr><tr><td>50/20 100/40</td><td>65.17 65.38</td><td>58.15 58.74</td></tr></table>

Table 10: Depth quality and the resulting occupancy accuracy on Occ-ScanNet-mini. All three entries are produced in a single controlled run with identical training settings.
<table><tr><td>Depth input</td><td>AbsRel ↓</td><td>RMSE (cm)↓</td><td>IoU↑</td><td>mIoU ↑</td></tr><tr><td>Ground truth</td><td>0.0000</td><td>0.00</td><td>69.61</td><td>62.67</td></tr><tr><td>DAv2</td><td>0.0441</td><td>13.82</td><td>65.50</td><td>58.35</td></tr><tr><td>MoGe-2</td><td>0.0782</td><td>23.43</td><td>62.44</td><td>55.76</td></tr></table>

## B.1 Object Navigation with Go2

We implement a lightweight embodied navigation system to connect AdaOcc predictions with realrobot execution. The platform uses a Unitree Go2 quadruped robot equipped with a D455 camera. AdaOcc reconstructs the observed scene as a semantic PLY point cloud, where each point is associated with its 3D position and semantic label or text-aligned semantic response. This semantic point cloud serves as the spatial representation for downstream navigation.

Given a language-specified target, we use open-vocabulary semantic activation in the reconstructed point cloud to determine the navigation goal. The text query activates semantically relevant occupied regions, and the goal position is obtained from the corresponding activated target region. In this way, the navigation module specifies goals through semantic concepts rather than manually defined metric coordinates.

To connect 3D reconstruction with robot navigation, we convert the semantic PLY map into a 2.5D traversability costmap. Floor or otherwise traversable points are projected onto a ground-plane grid as free space, while non-floor points within the robot collision-height range are treated as obstacles. The obstacles are further inflated according to the robot radius and a safety margin, yielding a conservative costmap for quadruped navigation. Unknown cells are treated as non-traversable during planning.

On the resulting costmap, we run A\* search Hart et al. (1968) from the robot start position to the semantic target. The planner snaps the start and goal positions to nearby traversable cells when necessary, uses clearance-aware costs to prefer safer regions, and resamples the planned path into robot-friendly waypoints. The final output is a waypoint sequence in the map frame, including position, heading, speed, and accumulated path distance, which can be directly used by the Go2 waypoint tracker. Fig. 9 visualizes the costmap and the planned trajectory.

## B.2 Closed-Loop Navigation on FloorPlan-R2R

We evaluate AdaOcc as the occupancy module of a closed-loop navigation policy on the FloorPlan-R2R val-unseen split in Habitat. Unlike standard vision-and-language navigation, the agent receives a semantic floorplan, in which each room is represented by a 2D polygon with a semantic category and a region ID, together with a concise instruction that specifies a start region, a target region, and an object- or relation-level stopping condition (e.g., “start in office 16, go to living room 13, and stop at the couch”). At every step the agent observes egocentric RGB-D and selects among MOVE\_FORWARD, TURN\_LEFT, TURN\_RIGHT, and STOP, so success requires both reaching the correct region and stopping where the local observation satisfies the language condition. The agent is not given its pose in floorplan coordinates or the floorplan-to-world scale.

![](images/3bbd30ae84b3677b1d5058b45213354d8fc7076ebcc0c579c23b35a40c6ce46f.jpg)  
Figure 8: The photo of devices used in real-world embodied tasks.

AdaOcc is used only as the online occupancy module: it converts accumulated egocentric RGB-D observations into a spatial representation used for floorplan grounding, target-region projection, and collision-free planning. The grounding, planning, and fixed zero-shot value-map stopping components are identical for both methods below, so the comparison isolates the utility of the occupancy representation.

Table 11: Closed-loop navigation on the FloorPlan-R2R val-unseen split. Both methods share the same grounding, planning, and stopping backend; FP-Nav additionally uses privileged floorplan alignment, whereas our pipeline estimates pose and scale online.
<table><tr><td>Method</td><td>NE↓</td><td>OSR ↑</td><td>SR↑</td><td>SPL ↑</td></tr><tr><td>FP-Nav</td><td>9.40</td><td>39.80</td><td>28.80</td><td>24.00</td></tr><tr><td>AdaOcc + value-map stopping</td><td>5.38</td><td>63.66</td><td>43.53</td><td>25.04</td></tr></table>

NE is the final distance to the goal, OSR records whether the agent reaches the success region at any time, SR additionally requires a correct final STOP, and SPL weighs SR by path efficiency. The AdaOcc occupancy representation reduces NE by 4.02 m and improves OSR, SR, and SPL by 23.86, 14.73, and 1.04 points. The large OSR gain indicates more reliable cross-region navigation and the SR gain shows that this advantage persists through final task completion, while the smaller SPL gain reflects that stopping accuracy, rather than path efficiency, remains the main bottleneck.

This setting differs from occupancy prediction for autonomous driving, where benchmarks typically use a fixed, calibrated surround-view rig, predict a predefined metric volume around a vehicle in structured outdoor scenes, and evaluate occupancy mainly as a perception output. Here a moving embodied agent incrementally maps an unseen indoor environment from changing egocentric views, and narrow passages, nearby clutter, dense occlusion, and fine object or room boundaries directly affect action selection and stopping. The predicted occupancy therefore acts as persistent spatial memory and traversability evidence inside a closed-loop policy rather than as an endpoint in itself.

## B.3 Pick and Place with ARX X5

We further validate the practical utility of AdaOcc in a real-world manipulation setup for picking bottle and place into basket using the ARX X5 platform. Given an input scene, AdaOcc predicts three object-level point sets, including two point clouds corresponding to the two target bottles and one point cloud corresponding to the basket.

The predicted point cloud of each bottle is independently provided to GraspNet Fang et al. (2020), which outputs an executable grasp pose for the ARX X5 manipulator. This process yields the grasping poses for the two bottles shown in the figure. For placement, we compute the geometric center of the predicted basket point cloud and use it as the target placement position.

Based on the predicted bottle points, the generated grasp poses, and the basket-centered placement coordinate, the ARX X5 platform executes a complete manipulation pipeline for waste-bottle collection, including target grasping and basket placement. This experiment shows that the object-level geometric outputs of AdaOcc can be directly coupled with off-the-shelf grasp planning for real-robot manipulation.

![](images/08bbb29cafd4c992b154612f2c22880797a908e0c3a6ac2492dc382eb1cc6050.jpg)  
Figure 9: Costmap-based navigation from AdaOcc predictions. The semantic PLY prediction is converted into a 2.5D traversability costmap, where free space, inflated obstacles, and unknown regions are represented in different gray levels. Given an open-semantic target activation, A\* plans a collision-aware path from the robot start position to the target region and exports map-frame waypoints for the Unitree Go2 robot.

Table 12: Equipment Used in Our Real-world Experiments
<table><tr><td>Device Name</td><td>Type</td><td>Embodied Tasks</td></tr><tr><td>Unitree Go2</td><td>Quadruped robot</td><td>Navigation/Mobile manipulation</td></tr><tr><td>ARX X5</td><td>Robotic arm</td><td>Maniulation/Mobile manipulation</td></tr><tr><td>Intel RealSense D455</td><td>Long-range stereo depth camera</td><td>Navigation</td></tr><tr><td>Intel RealSense D405</td><td>Short-range depth camera</td><td>Mobile manipulation</td></tr><tr><td>MRDVS M4</td><td>ToF RGB-D sensor</td><td>Maniulation</td></tr></table>

## B.4 Mobile Manipulation with ARX X5 on Go2

Building on the navigation and manipulation pipeline above, we further demonstrate a long-range embodied mobile manipulation task using ARX X5 on Go2 platform. In this setting, the robot first observes a target bottle and a basket from a distance. The bottle location predicted by AdaOcc is used as the semantic navigation target, allowing Go2 to move from the initial observation point to the vicinity of the target object.

After reaching a suitable manipulation range, we apply the same object-centric perception and manipulation logic as above. Specifically, AdaOcc predicts the point sets of the target bottle and the basket in the local scene. The predicted bottle point set is then fed into GraspNet to generate an executable grasp pose, while the center of the predicted basket point set is used as the placement target. The robot then completes the bottle collection task by grasping the bottle and placing it into the basket.

This experiment demonstrates that AdaOcc supports a complete embodied pipeline that starts from long-range semantic target localization, continues with waypoint-based quadruped navigation, and finally enables goal-conditioned bottle grasping and basket placement in a real-world setting.

## C Standard Deviation of Experiments

We repeat the best AdaOcc configuration on Occ-ScanNet-mini with five random seeds. S0–S4 correspond to seeds 0, 1785784487, 1602478221, 1383549716, and 977375669, respectively. As shown in Table 13, AdaOcc shows stable performance across runs, with a standard deviation of 0.20

Table 13: Standard deviation over five random seeds on Occ-ScanNet-mini. S0–S4 correspond to seeds 0, 1785784487, 1602478221, 1383549716, and 977375669, respectively. All results are reported at epoch 200. Bold and underline indicate the best and second-best results among the five runs.
<table><tr><td>Seed</td><td>mIoU</td><td>ceiing</td><td>0or</td><td>Wall</td><td></td><td>char</td><td>pəq</td><td>soa</td><td>table</td><td>SAS</td><td>Furure</td><td>obects</td><td>IoU</td></tr><tr><td>SO</td><td>58.74</td><td>48.73</td><td>57.61</td><td>56.25</td><td>48.18</td><td>58.90</td><td>75.30</td><td>75.07</td><td>57.58</td><td>46.52</td><td>64.37</td><td>57.61</td><td>65.38</td></tr><tr><td>S1</td><td>58.26</td><td>47.13</td><td>57.39</td><td>56.12</td><td>47.64</td><td>58.69</td><td>75.19</td><td>74.99</td><td>57.63</td><td>43.60</td><td>64.47</td><td>58.05</td><td>65.24</td></tr><tr><td>S2</td><td>58.54</td><td>48.69</td><td>57.54</td><td>56.40</td><td>48.15</td><td>58.91</td><td>75.53</td><td>75.27</td><td>57.93</td><td>42.91</td><td>64.82</td><td>57.81</td><td>65.49</td></tr><tr><td>S3</td><td>58.46</td><td>48.03</td><td>57.57</td><td>56.17</td><td>47.94</td><td>58.94</td><td>75.54</td><td>75.03</td><td>57.85</td><td>43.18</td><td>65.19</td><td>57.59</td><td>65.46</td></tr><tr><td>S4</td><td>58.29</td><td>46.71</td><td>57.59</td><td>56.51</td><td>47.52</td><td>58.88</td><td>74.71</td><td>75.28</td><td>57.30</td><td>43.93</td><td>64.92</td><td>57.85</td><td>65.46</td></tr><tr><td>Mean Std.</td><td>58.46 0.20</td><td>47.86 0.91</td><td>57.54 0.09</td><td>56.29 0.16</td><td>47.89 0.30</td><td>58.86 0.10</td><td>75.25 0.34</td><td>75.13 0.14</td><td>57.66 0.25</td><td>44.03 1.45</td><td>64.75 0.34</td><td>57.78 0.19</td><td>65.41 0.10</td></tr></table>

mIoU and 0.10 IoU. Across the five runs, the mIoU ranges from 58.26 to 58.74 and the IoU ranges from 65.24 to 65.49.

## D Implementation Details and Compute Resources

## D.1 Implementation Details

We provide the main implementation details for the Occ-ScanNet-mini AdaOcc configuration in Table 14. The model is trained for 200 epochs with progressive query learning, where the active query budget increases from 100 to 500 queries in five stages. During evaluation, the full query budget is used unless otherwise specified.

Coordinate frames. Each Occ-ScanNet sample provides camera intrinsics, camera-to-world extrinsics, ego-to-world pose, and LiDAR-to-ego pose. We use the ego/occupancy frame as the canonical local 3D frame for training. In the data loader, the original metadata contains the transform from ego to LiDAR, denoted as $T _ { \mathrm { e g o \to l i d a r } }$ , and an identity transform from ego to occupancy, $T _ { \mathrm { e g o \to o c c } } = I$ Depth-derived points are first reconstructed in the camera frame, transformed to the LiDAR frame using the corresponding sensor-to-LiDAR extrinsic, and then transformed into the occupancy frame by

$$
T _ { \mathrm { l i d a r } \to \mathrm { o c c } } = T _ { \mathrm { e g o } \to \mathrm { o c c } } T _ { \mathrm { e g o } \to \mathrm { l i d a r } } ^ { - 1 } .
$$

Since $T _ { \mathrm { e g o \mathrm {  o c c } } }$ is identity in this setting, the occupancy frame is aligned with the current ego frame. For image feature sampling, we construct an ego-to-image projection matrix $T _ { \mathrm { e g o \to i m g } } =$ $K T _ { \mathrm { e g o \to c a m } } .$ , where K is the camera intrinsic matrix. Any image-space resize/crop transform is represented by $\mathbf { a } \ 4 \times 4$ image data augmentation matrix $A _ { \mathrm { i d a } }$ and left-multiplied into the projection matrix, i.e.,

$$
T _ { \mathrm { e g o \to i m g } }  A _ { \mathrm { i d a } } T _ { \mathrm { e g o \to i m g } } ,
$$

so that the projected 3D points remain consistent with the resized input image.

Depth back-projection and pseudo points. For Occ-ScanNet dataset, we use the front camera only. The input RGB image has raw resolution 1296 × 968 and is resized/cropped to $9 6 0 \times 7 2 0$ . The associated depth map is used to construct pseudo points. For each valid depth pixel $( u , v )$ with depth $d ,$ where $d \in \mathsf { [ 0 . 1 , 7 . 5 ] }$ meters, we back-project using the OpenCV camera convention:

$$
x = \frac { ( u + 0 . 5 - c _ { x } ) d } { f _ { x } } , \qquad y = \frac { ( v + 0 . 5 - c _ { y } ) d } { f _ { y } } , \qquad z = d .
$$

The resulting camera-frame point $( x , y , z )$ is transformed to the LiDAR frame and then to the occupancy frame as described above. Each point stores five channels:

$$
( x , y , z , { \mathrm { i n t e n s i t y } } , \Delta t ) ,
$$

where the intensity channel is set to zero and $\Delta t$ is the timestamp offset. In the reported single-frame configuration, only the current frame is kept, so $\Delta t = 0$ for the selected points. Points outside the local occupancy range are removed before voxelization.

Occupancy range and voxelization. The local 3D range is

$$
[ - 3 . 2 , - 4 . 8 , - 5 . 6 , 7 . 2 , 4 . 8 , 5 . 6 ] ,
$$

corresponding to $( x _ { \mathrm { m i n } } ,$ y<sub>min</sub>, z<sub>min</sub>, $x _ { \mathrm { m a x } } , y _ { \mathrm { m a x } } , z _ { \mathrm { m a x } } )$ in meters. We use a voxel size of

$$
\Delta = ( 0 . 0 8 , 0 . 0 8 , 0 . 0 8 ) \mathrm { m } .
$$

Thus the Cartesian grid has size

$$
N _ { x } = 1 3 0 , \quad N _ { y } = 1 2 0 , \quad N _ { z } = 1 4 0 .
$$

This range is the agent-centric volume over which AdaOcc predicts; it fully contains the official Occ-ScanNet evaluation grid of size $6 0 \times 6 0 \times 3 6$ at the same voxel size.

Input points are voxelized with voxel size 0.08 m, maximum 10 points per voxel, and at most 90k/120k voxels for train/test respectively. The voxel coordinates are computed by

$$
\mathbf i = \left\lfloor \frac { \mathbf p _ { \mathrm { o c c } } - \mathbf p _ { \mathrm { m i n } } } { \Delta } \right\rfloor ,
$$

where ${ \bf p } _ { \mathrm { m i n } } = ( - 3 . 2 , - 4 . 8 , - 5 . 6 )$ . Only points whose voxel indices fall inside the valid grid are kept.

Ground-truth occupancy voxels. The Occ-ScanNet ground truth is loaded from compressed labels.npz files. Each file contains semantic occupancy labels, camera visibility masks, LiDAR visibility masks, and, for raw Occ-ScanNet supervision, the raw semantic grid with its voxel origin and voxel size. We use the raw Occ-ScanNet ground truth branch during training and evaluation. The raw semantic grid has size 60 × 60 × 36 at 0.08 m together with a per-frame voxel origin and voxel size, and follows the native $[ y , x , z ]$ axis order. Voxels labelled 0 are empty, labels $1 , \ldots , 1 1$ are the occupied semantic classes, and 255 denotes unknown voxels, which are excluded from both training targets and evaluation. Raw semantic labels in the range $1 , \ldots , 1 1$ are treated as occupied semantic classes and converted to zero-based class IDs $0 , \ldots , 1 0 ;$ free space and unknown voxels are ignored for the point-level semantic targets.

The raw Occ-ScanNet tensor is indexed in $( y , x , z )$ order. Therefore, when constructing 3D target points, a raw voxel coordinate $( i _ { y } , i _ { x } , i _ { z } )$ is converted to Cartesian order as $( i _ { x } , i _ { y } , i _ { z } )$ . The raw voxel origin denotes the center of voxel $( 0 , 0 , 0 )$ rather than the minimum grid corner. Hence the 3D center of a raw occupied voxel is computed as

$$
\mathbf { p } _ { \mathrm { g t } } = \mathbf { o } _ { \mathrm { r a w } } + ( i _ { x } , i _ { y } , i _ { z } ) \odot \Delta _ { \mathrm { r a w } } ,
$$

where $\mathbf { o } _ { \mathrm { r a w } }$ is the raw voxel-origin vector and $\Delta _ { \mathrm { r a w } }$ is the raw voxel size. This convention avoids a half-voxel offset when matching predicted points to raw Occ-ScanNet supervision.

Prediction decoding and voxel aggregation. The decoder predicts normalized 3D refinement points inside the configured local range. We decode them back to metric coordinates in the occupancy frame. For raw Occ-ScanNet evaluation, predicted points are further transformed from the local occupancy frame to the world frame using

$$
T _ { \mathrm { o c c } \to \mathrm { w o r l d } } = T _ { \mathrm { e g o } \to \mathrm { w o r l d } } T _ { \mathrm { o c c } \to \mathrm { e g o } } .
$$

They are then mapped into the raw Occ-ScanNet grid using the raw voxel origin and voxel size. Since the raw grid origin is center-defined, the binning rule is

$$
\mathbf { i } _ { \mathrm { r a w } } = \left\lfloor \frac { \mathbf { p } _ { \mathrm { w o r l d } } - \mathbf { o } _ { \mathrm { r a w } } } { \Delta _ { \mathrm { r a w } } } + 0 . 5 \right\rfloor .
$$

If multiple predicted points fall into the same voxel, we keep the top-k points by their maximum semantic confidence, with $k = 1 0$ , and average their semantic scores to obtain the voxel-level prediction. During evaluation, points with semantic confidence below 0.4 are discarded, and points too far from their query center are removed using a center-distance threshold of 1.0 m. We additionally apply the configured sparse padding step to fill small holes before computing occupancy metrics.

Training target construction. For training, occupied raw voxels are converted to sparse 3D points and used as point-level targets. Predicted query points are matched to ground-truth occupied voxel centers using a bidirectional point matching objective. The point loss is a density-aware Chamfer-style loss implemented with Smooth-L1 distance, and semantic classification is supervised with focal loss. The final loss combines the last decoder layer and auxiliary decoder layers with an exponential layer decay of 0.9. A containment loss is additionally applied on decoder layers d4 and d5 to penalize predictions that fall into known empty regions and pulls them toward nearby occupied voxel boxes. Its weight is linearly increased from 0.1 to 0.3 between epochs 20 and 60.

Evaluation protocol. Evaluation uses the full 500-query budget. Predicted points are mapped from the local occupancy frame into the official Occ-ScanNet raw grid $( 6 0 \times 6 \bar { 0 } \times 3 6 $ , voxel size 0.08 m) using the per-frame voxel origin and voxel size, and compared against its raw semantic labels. Voxels with unknown labels (255) are excluded, which corresponds to the camera-visible region of the selected front camera; predictions falling outside the official grid are ignored. We report occupied IoU and mIoU over the 11 occupied semantic classes.

Table 14: Main implementation details for AdaOcc on Occ-ScanNet-mini.
<table><tr><td colspan="2">Data and Input Configuration</td></tr><tr><td>Dataset</td><td>Occ-ScanNet-mini</td></tr><tr><td>Training / validation samples</td><td>4,639 / 2,007</td></tr><tr><td>RGB observations T</td><td>Single-view RGB, current frame only</td></tr><tr><td>Geometric observations  $\mathcal { G }$ </td><td>Depth-derived 3D points for benchmark experiments</td></tr><tr><td>Image resolution</td><td>960 × 720 after resizing</td></tr><tr><td>Occupancy range Ω</td><td> $[ - 3 . 2 , - 4 . 8 , - 5 . 6 , 7 . 2 , 4 . 8 , 5 . 6 ]$ </td></tr><tr><td>Voxel size v</td><td>[0.08, 0.08, 0.08]</td></tr><tr><td>Semantic classes</td><td>11 occupied classes + free label</td></tr><tr><td colspan="2">Model Configuration</td></tr><tr><td>Image encoder  $E _ { \mathrm { i m g } }$ </td><td>C-RADIOv3-B, last 4 blocks unfrozen</td></tr><tr><td>Image feature dimension</td><td>512</td></tr><tr><td>Geometric encoder  $E _ { \mathrm { g e o } }$ </td><td>Sparse 3D convolutional encoder with multi-plane feature aggregation</td></tr><tr><td>Decoder layers  $L$ </td><td></td></tr><tr><td>Maximum query budget  $N _ { \mathrm { m a x } }$ </td><td>500</td></tr><tr><td>Initial query budget  $N _ { \mathrm { i n i t } }$ </td><td>100</td></tr><tr><td>Progressive query schedule  $N _ { q } ( t )$ </td><td>Increase by 100 queries every 40 epochs until 500</td></tr><tr><td>Queries at evaluation</td><td>Full query budget unless otherwise specified</td></tr><tr><td>Points per query refinement</td><td>[1, 2, 4, 8, 16, 32] across decoder layers</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Base learning rate</td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Batch size</td><td>8 per GPU, 64 global</td></tr><tr><td>Training epochs</td><td>200</td></tr><tr><td>Learning-rate schedule</td><td>500-iteration linear warmup + cosine decay to  $2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Mixed precision</td><td>Enabled with static loss scale 512</td></tr><tr><td>Gradient clipping</td><td>L2 norm clipping with max norm 35</td></tr><tr><td colspan="2">Loss Configuration</td></tr><tr><td>Semantic loss  $\mathcal { L } _ { \mathrm { s e m } }$ </td><td>Focal loss with  $\gamma _ { \mathrm { f o c a l } } = 2 . 0 , \alpha = 0 . 2 5 , \lambda _ { \mathrm { s e m } } = 2 . 0$ </td></tr><tr><td>Point reconstruction  ${ \mathcal { L } } _ { \mathrm { p t s } }$ </td><td>DCD Loss with Smooth L1 regression,  $\lambda _ { \mathrm { { p t s } } } = 0 . 5$ </td></tr><tr><td>Decoder loss decay</td><td> $\gamma = 0 . 9$ </td></tr><tr><td>Containment loss  ${ \mathcal { L } } _ { \mathrm { c o n } }$ </td><td>Applied to layer 5 and  $^ { 6 , }$  with  $\lambda _ { \mathrm { { c o n } } }$  ramped from 0.10 to 0.30</td></tr></table>

Table 15: Experimental Settings on Different Datasets
<table><tr><td>Datasets</td><td>RGB Input</td><td>LiDAR Input</td><td>Open Vocabulary</td><td>Depth Map Type</td><td>#Views</td></tr><tr><td>Occ-ScanNet</td><td>√</td><td></td><td></td><td>Pseudo</td><td>1</td></tr><tr><td>Occ3D-nuScenes (setting1)</td><td>√</td><td>√</td><td></td><td></td><td>6</td></tr><tr><td>Occ3D-nuScenes (setting2)</td><td>V</td><td></td><td></td><td>Pseudo</td><td>6</td></tr><tr><td>TartanGround (setting1)</td><td>√</td><td>V</td><td>√</td><td></td><td>1-4</td></tr><tr><td>TartanGround (setting2)</td><td>√</td><td></td><td>√</td><td>GT</td><td>1-4</td></tr></table>

## D.2 Experimental Settings on Different Datasets

We provide a summary of the experimental settings across datasets in Table 15. The compared settings cover different combinations of RGB input, LiDAR input, open-vocabulary support, depth map sources, and the number of available views.

## D.3 Compute Resources

The compute resources for the Occ-ScanNet-mini RADIO configuration are summarized in Table 16. The model was trained on one internal Linux node with 8 NVIDIA H20 GPUs using PyTorch distributed data parallelism. Each GPU has approximately 97,871 MiB of memory. The 200-epoch training process took about 8.0 hours on 8 GPUs, corresponding to approximately 64 GPU-hours for training. Including the final validation pass, the total recorded run time was about 8.2 hours. The final 500-query training stage used about 11.0 GB memory per GPU, while validation used about 4.8 GB per GPU. Validation latency measured from the training log was approximately 0.34 s per sample per GPU, including data loading and metric computation.

Table 16: Compute resources for the Occ-ScanNet-mini AdaOcc run.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Hardware</td><td>8 × NVIDIA H20 GPUs</td></tr><tr><td>GPU memory</td><td>97,871 MiB per GPU</td></tr><tr><td>Training framework</td><td>PyTorch DDP with NCCL backend</td></tr><tr><td>CUDA / cuDNN</td><td>CUDA 12.1 / cuDNN 8.9.2</td></tr><tr><td>PyTorch / TorchVision</td><td>2.2.0 / 0.17.0</td></tr><tr><td>Training time</td><td>8.0 h on 8 GPUs</td></tr><tr><td>Total recorded run time</td><td>8.2 h including final validation</td></tr><tr><td>Estimated training compute</td><td>64 GPU-hours</td></tr><tr><td>Steady-state training memory</td><td>11.0 GB per GPU at 500 queries</td></tr><tr><td>Validation memory</td><td>4.8 GB per GPU</td></tr><tr><td>Validation latency</td><td>0.34 s per sample per GPU</td></tr><tr><td>Checkpoint size</td><td>3.4 GB per checkpoint</td></tr></table>

## E Limitations

AdaOcc still has limited capability in modeling very small or thin objects. Since the method represents occupied regions with a finite set of semantic points and voxelizes them into a standard occupancy grid for evaluation, fine-grained structures may be under-represented when they occupy only a few voxels or are weakly observed from the input views. This limitation can be further affected by occlusion, noisy geometric inputs, or incomplete depth/LiDAR observations in cluttered indoor scenes. Beyond fine-grained geometry, AdaOcc estimates depth from RGB with an external model (Depth Anything v2), which adds inference latency and introduces failure modes on reflective or low-texture surfaces, thin structures, and occluded boundaries; replacing the estimate with ground-truth depth quantifies both the dependency and the remaining headroom (Appendix A.5). Our evaluation is further limited to indoor benchmarks in the single-frame or direct-aggregation setting, so broader indoor evaluation and explicit temporal memory remain open directions. Future work will mitigate these issues through finer-grained query allocation, stronger local geometric refinement, small-object-aware supervision, and a more robust treatment of the geometric input.

## F Broader Impacts

AdaOcc aims to improve adaptive 3D scene perception for embodied agents. Its potential positive impacts include enabling robots and embodied systems to build more accurate spatial representations under varying sensing and computational conditions, which may benefit indoor navigation, assistive robotics, and other downstream embodied tasks. The budget-adaptive design may also make 3D perception more accessible on platforms with limited onboard computation.

At the same time, occupancy prediction systems may introduce risks when deployed in real-world environments. Incorrect occupancy or semantic predictions could affect downstream planning decisions, especially in safety-critical settings involving humans or fragile objects. In addition, 3D perception systems used in indoor environments may raise privacy concerns if raw sensor data are collected or stored. AdaOcc does not directly address these deployment-level issues. Practical use should therefore include task-specific safety checks, privacy-preserving data handling, and validation under the target sensing and operating conditions.

## G Assets and Licenses

This work uses existing academic datasets, pretrained models, and baseline implementations for research evaluation. We cite the original sources for Occ-ScanNet, Occ-ScanNet-mini, TartanGround, Depth Anything v2, EfficientNet, RADIO/C-RADIO, OPUS, SplatSSC, and other compared methods in the main paper. These assets are used only for academic research and evaluation under their original licenses and terms of use. We do not redistribute the original datasets, pretrained checkpoints, or third-party code as part of this submission.