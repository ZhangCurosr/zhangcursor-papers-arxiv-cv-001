# HiPFuse: Real-Time LiDAR Semantic Segmentation for Robotic

Perception on Edge Devices

Haiyang Wu<sup>1</sup>, George Vosselman<sup>1</sup>, and Ville Lehtola<sup>1</sup>

Abstract—Real-time robotic perception requires accurate Li-DAR semantics under limited resources. Bird’s-eye-view (BEV) aggregation is efficient but can mix differently labeled surfaces in columns. We present HiPFuse, a point–BEV fusion framework that exchanges point and height-partitioned BEV features across horizontal scales, combining point-wise detail with 2D spatial context from one LiDAR scan. Fixed height assignments preserve separate vertical aggregation groups, with auxiliary supervision of initial BEV features using targets from the same 3D cells. A dual-scale BEV context module adds broader context from a coarser grid. On SemanticKITTI and MCD, HiPFuse achieves 63.7% and 62.5% mIoU at 14.2 and 22.4 FPS on Jetson AGX Orin, respectively, exceeding the 10 Hz scan rate. Separate training and evaluation show indoor applicability. Replacing only the baseline semantic front end with HiPFuse in fixed robotic systems reduces odometry absolute trajectory error RMSE by 10.6% on average across two outdoor sequences and increases accepted-loop recall by 8.7 percentage points on SemanticKITTI. Qualitative semantic maps show more continuous road labeling. Our code is available at https://github.com/haiyang2022/HiPFuse.

Index Terms—LiDAR semantic segmentation, point clouds, real-time perception, robotic perception.

## I. INTRODUCTION

Beyond scene understanding, semantic information from LiDAR scans can support state estimation tasks such as LiDAR–inertial odometry by informing how observations are selected, associated, and modeled [1], [2]. Semantic cues also support loop retrieval [3] and semantic mapping [4], allowing point-wise semantic predictions to serve several components of a robotic perception pipeline. The benefit of these predictions depends on both their quality and the cost of producing them, since semantic segmentation must share computational resources with the downstream algorithms, especially on edge devices.

In an online perception pipeline, this calls for a semantic front end that keeps pace with incoming scans within a limited computational budget. Projecting point cloud features onto 2D grids enables efficient convolutional processing, as exemplified by the range-image networks RangeNet++ [5] and SalsaNext [6]. However, range-image projection introduces distance-dependent variations in object scale [7]. A bird’seye-view (BEV) grid instead uses fixed-size horizontal cells, providing a consistent scale for spatial processing without requiring a regular arrangement of scan lines.

![](images/ef4f5ec26138154af50f6503007b486301a793fe32fe6969bece9dd52baee2e5.jpg)  
Fig. 1. SemanticKITTI test accuracy versus measured Jetson AGX Orin throughput (Tables III and V). Shading indicates throughput meeting the 10 Hz scan rate under the measured settings.

This leads to a follow-up problem: surfaces at different heights within a BEV column may belong to different semantic classes, yet methods such as PolarNet [8] combine all observations in the column into one descriptor before 2D spatial processing.

The vertical axis need not be collapsed, however: height partitioning keeps separate descriptors for different height intervals within each column. Related ideas of retaining height information in BEV channels have been used in point cloud registration and LiDAR perception [9], [10]. For segmentation, the challenge is to learn features that distinguish surfaces with different labels within the same BEV column.

A point branch offers a complementary way to preserve local detail, as demonstrated by HARP-NeXt [11] and sparse point–voxel convolution [12]. Keeping individual point descriptors alongside grid context does not itself separate height partitions during BEV aggregation. We bring these ideas together in HiPFuse, a point–BEV fusion framework for semantic segmentation from a single LiDAR scan. Within this framework, point features and height-partitioned BEV context are learned jointly across scales, preserving separate height groups during aggregation and individual point features during fusion.

Even with height partitioning and point-wise detail, similar local surfaces can belong to different semantic classes. Their interpretation can therefore benefit from surrounding structures and spatial relationships. To gather this context at limited additional cost, we further introduce a dual-scale BEV context (DSC) module that processes a coarser spatial grid and fuses the resulting context with features at the current resolution.

To assess HiPFuse as a semantic front end for resourceconstrained robotic perception, we evaluate its accuracy and throughput on the outdoor benchmarks SemanticKITTI [13] and MCD [14]. Our method achieves a favorable accuracy– throughput trade-off relative to the evaluated baselines, with scan-rate segmentation on NVIDIA Jetson AGX Orin (Fig. 1). Separate training and evaluation on Semantic-ITC [15] examine indoor applicability, with segmentation throughput also exceeding the scan rate.

To determine whether this segmentation performance translates into benefits for downstream robotic perception, we replace the semantic inputs to Hybrid-VoxelMap [2] odometry, SGLC [3] loop retrieval, and SG-SLAM [4] semantic mapping while keeping the downstream algorithms and settings fixed. Compared with semantic inputs from SalsaNext [6] and HARP-NeXt [11], HiPFuse predictions reduce outdoor odometry errors and increase loop retrieval recall. Qualitative map comparisons also show more continuous road labeling.

The main contributions are as follows:

• We introduce HiPFuse, a point–BEV fusion framework for real-time LiDAR semantic segmentation whose integrated design combines height partitioning along the LiDAR z-axis, point-wise detail, and 2D spatial context learning.

• We expand the context available for semantic interpretation with a dual-scale BEV module, limiting the added spatial computation by extracting broader context on a coarser grid.

• We demonstrate a favorable accuracy–efficiency balance with scan-rate segmentation throughput on an edge device, and show that HiPFuse, requiring only the current scan, can be plugged in as the semantic front end of existing systems and benefits multiple robotic perception tasks while keeping their algorithms and settings fixed.

## II. RELATED WORK

## A. Efficient LiDAR Semantic Segmentation

Efficient LiDAR segmentation organizes feature aggregation over point neighborhoods, 3D voxel cells, or projected 2D grids, with different consequences for computational cost and geometric detail. RandLA-Net [16] operates on point neighborhoods, combining random sampling with local feature aggregation to retain information after subsampling. Cylinder3D [17] organizes points in cylindrical cells adapted to outdoor scan density and applies asymmetrical 3D convolutions. RangeNet++ [5] instead performs 2D processing on spherical range images and uses range-based postprocessing to recover point labels affected by projection and discretization.

Among projected representations, BEV grids group observations by horizontal location, making the treatment of height central to descriptor construction. PolarNet [8] pools learned point features into one descriptor per polar column for 2D processing. Its output channels encode separate class scores for each height bin and are reshaped into a voxel prediction grid. Layered alternatives retain separate height contributions within the BEV channels: VolMap [18] stacks BEV intensity maps from different LiDAR layers, while LidarMultiNet [10] concatenates height-indexed features from a downsampled voxel grid to provide BEV context within a sparse 3D encoder– decoder.

Grid organization determines which points are aggregated together, but the resulting descriptors can obscure differences needed for point-wise semantic prediction. Hybrid representations retain individual point features alongside grid context to preserve these distinctions. Sparse point–voxel convolution [12] combines a high-resolution point stream with sparse voxel convolutions, while RPVNet [19] exchanges features among range, point, and voxel representations through gated fusion. For 2D spatial processing, WaffleIron [20] repeatedly projects point features onto Cartesian planes and lifts the processed features back to points. HARP-NeXt [11] couples range-image and point features through hierarchical fusion, using pixel groups for local encoding and aggregation together with auxiliary range-image supervision.

Across these representation choices, context extraction must account for both spatial scale and computational cost. EfficientLPS [7] uses range-aware feature fusion and a distancedependent semantic head to handle object-scale variation in Li-DAR panoptic segmentation. SalsaNext [6] expands receptive fields using dilated convolutions without the parameter cost of corresponding dense kernels, while MINet [21] allocates more complex processing to lower-resolution paths.

These designs expose a common tension: grid aggregation supports compact spatial processing but can mix observations requiring different semantic labels. We address this tension through a point–BEV fusion framework that jointly learns height-partitioned BEV features and individual point features across scales, retaining both height and point distinctions within 2D spatial processing.

## B. Semantics for Robotic Perception

For pose estimation, semantic information can guide observation selection and the construction of geometric constraints for registration. SuMa++ [1] uses semantic consistency to filter dynamic surfels and semantic information to weigh scanmatching residuals. Hybrid-VoxelMap [2] uses semantic superlabels to select planar or Gaussian voxel models and combines their residuals for LiDAR–inertial odometry.

Beyond local scan matching, the spatial arrangement of semantic observations can help identify revisited places. Semantic Scan Context [22] encodes categories in a polar descriptor after scan alignment. SGLC [3] combines foreground semantic graphs with background geometry for candidate retrieval, verification, and loop pose estimation.

Alongside their use in pose estimation and loop retrieval, semantic observations can be accumulated across scans to build persistent maps. Bayesian spatial kernel smoothing [23] exploits spatial correlations to estimate semantic occupancy, while the semantic extension of PIN-SLAM [24] learns a continuous semantic field from point-wise labels. SlideSLAM [25] uses segmented objects as landmarks and jointly optimizes their models and robot trajectories. SG-SLAM [4] maintains both semantic point cloud and graph maps, using instance tracking to associate graph nodes across scans.

![](images/e3d909e31359e04b865acd553378b2163ebd5d76bdbfcd80337cb067745612f2.jpg)  
Fig. 2. Overview of the HiPFuse architecture.

We therefore evaluate semantic predictions through the odometry estimates, loop retrieval results, and semantic maps they help produce. Keeping downstream algorithms and settings fixed, we show that replacing only the semantic input benefits these perception tasks.

## III. METHODOLOGY

## A. Overview

Fig. 2 illustrates the overall HiPFuse pipeline. From a single LiDAR scan, HiPFuse encodes individual points and aggregates their features in 3D cells to form a height-partitioned BEV representation. The point and BEV branches then exchange features across scales: 2D processing provides BEV context to individual points, and the updated point features are aggregated back into BEV. Throughout this process, the point branch retains individual descriptors, while BEV aggregation keeps the height assignments fixed as the horizontal grid becomes coarser. Features from multiple levels of both branches are finally combined for point-wise semantic prediction.

We adopt the Conv-SE-NeXt blocks, residual attention fusion, and fusion head from HARP-NeXt [11].

## B. Height-Partitioned Projection and Encoding

1) BEV Grid and Height Partitions: Each input point has coordinates $\mathbf { p } _ { i } ~ = ~ ( x _ { i } , y _ { i } , z _ { i } ) ^ { \mathsf { T } } ~ \in ~ \mathbb { R } ^ { 3 }$ and intensity $\rho _ { i } .$ . We divide a predefined region in the LiDAR coordinate system into a $W \times H$ BEV grid and $Z$ uniform height partitions, forming 3D cells of size $s _ { x } \times s _ { y } \times s _ { z }$ . Each new scan is partitioned directly along the LiDAR z-axis, without gravity alignment or leveling.

Let $x _ { \mathrm { m i n } } , y _ { \mathrm { m i n } } .$ , and $z _ { \mathrm { m i n } }$ denote the region’s lower coordinate bounds. Each point within this region is assigned the cell indices

$$
\begin{array} { r } { ( u _ { i } , v _ { i } , h _ { i } ) = \left( \left\lfloor \frac { x _ { i } - x _ { \mathrm { m i n } } } { s _ { x } } \right\rfloor , \left\lfloor \frac { y _ { i } - y _ { \mathrm { m i n } } } { s _ { y } } \right\rfloor , \left\lfloor \frac { z _ { i } - z _ { \mathrm { m i n } } } { s _ { z } } \right\rfloor \right) , } \end{array}\tag{1}
$$

where $u _ { i } \in \{ 0 , \ldots , W - 1 \} , v _ { i } \in \{ 0 , \ldots , H - 1 \}$ , and $h _ { i } \in$ $\{ 0 , \ldots , Z - 1 \}$

2) Height-Aware Point Encoder: Using these cell assignments, we encode each point relative to its 3D cell. For a batch of B scans containing N retained points, let $\mathcal { V } _ { i } = ( b _ { i } , h _ { i } , v _ { i } , u _ { i } )$ identify point i’s cell, where $b _ { i }$ is the scan index. The cell’s mean point coordinate is $\bar { \mathbf { p } } \nu _ { i }$ , and its geometric center is $\mathbf { c } _ { \mathcal { V } _ { i } }$ . We concatenate each point’s coordinates and intensity with its coordinate offsets from the cell mean and cell center, using brackets to denote concatenation. A shared point-wise multilayer perceptron (MLP) ψ with Linear– BatchNorm–ReLU layers maps this input to a C-dimensional point descriptor $\mathbf { e } _ { i } \mathbf { : }$

$$
\mathbf { e } _ { i } = \psi \left( \left[ \mathbf { p } _ { i } , \rho _ { i } , \mathbf { p } _ { i } - \bar { \mathbf { p } } \nu _ { i } , \mathbf { p } _ { i } - \mathbf { c } _ { \mathcal { V } _ { i } } \right] \right) .\tag{2}
$$

The mean-relative offset describes local point geometry, whereas the center-relative offset locates the point within its 3D cell.

3) Cell Max Pooling and Scatter: The encoded point features $\mathbf { e } _ { i }$ in each 3D cell are aggregated by channel-wise max pooling. We scatter the pooled descriptors to their cell indices and set empty cells to zero, giving $\dot { \mathbf { X } } \in \mathbb { R } ^ { B \times Z \times H \times W \times C }$

At each horizontal cell, the Z height partitions each contribute C features. We arrange these $Z C$ values as channels of a 2D BEV feature map, yielding the input ${ \bf F } _ { \mathrm { i n } } ~ \in  \ L $ $\mathbb { R } ^ { B \times Z C \times H \times W }$ for 2D convolution. The individual descriptors $\mathbf { e } _ { i }$ are retained for the point branch.

## C. Multi-Scale Point–BEV Fusion Backbone

The backbone takes the BEV input $\mathbf { F } _ { \mathrm { i n } }$ and encoded point features $\{ { \mathbf { e } _ { i } } \} _ { i = 1 } ^ { N }$ from the preceding encoder. An initial feature-processing block, referred to as the stem, initializes the point and BEV branches; four fusion stages then update them at multiple BEV resolutions. The backbone then combines these levels within each branch to provide the inputs to the fusion head. We denote the output BEV map and the set of point descriptors at level k by $\mathbf { F } ^ { k }$ and ${ \bf Q } ^ { \bar { k } } \ = \ \{ { \bf q } _ { i } ^ { k } \} _ { i = 1 } ^ { N } ,$ respectively, with $k = 0$ for the stem and $k = 1 , \ldots , 4$ for the fusion stages.

1) Stem: The stem first applies three successive $3 \times 3$ convolutions to $\mathbf { F } _ { \mathrm { i n } }$ . This produces the intermediate BEV map $\widetilde { \mathbf { F } } ^ { 0 }$ at resolution $H \times W$ , before point–BEV fusion. For each point, we sample this map at its horizontal cell, concatenate the sampled features with $\mathbf { e } _ { i } ,$ and apply a shared point-wise MLP $\phi _ { 0 }$ to obtain the updated descriptor ${ \bf q } _ { i } ^ { 0 }$ for point i.

We then aggregate these descriptors, collected in $\mathbf { Q } ^ { 0 }$ , into a BEV tensor $\mathbf { S } ^ { 0 }$ using the same cell max pooling, scatter, and channel arrangement as for $\mathbf { F } _ { \mathrm { i n } } .$ . A 2D convolutional fusion layer $\gamma _ { 0 }$ projects the concatenation of ${ \bf S } ^ { 0 }$ and $\widetilde { \mathbf { F } } ^ { 0 }$ , and the DSC module $\mathcal { D } _ { 0 }$ (Sec. III-C3) adds broader context to produce $\mathbf { F } ^ { 0 }$ . The stem convolutions, point update MLPs $\phi _ { k }$ and BEV fusion layers $\gamma _ { k }$ all use batch normalization and Hardswish [26].

2) Fusion Stages: Each stage k receives the preceding BEV features $\mathbf { F } ^ { k - 1 }$ , point descriptors $\mathbf { Q } ^ { k - 1 }$ , and their BEV aggregate $\mathbf { S } ^ { k - 1 }$ . Stage 1 retains the stem resolution, while stages 2–4 each halve both BEV dimensions, giving cumulative horizontal strides $( r _ { 1 } , r _ { 2 } , r _ { 3 } , r _ { 4 } ) = ( 1 , 2 , 4 , 8 )$ . We choose H and W to be divisible by 8. The point branch retains a descriptor for each of the N points at every stage.

A stack of Conv-SE-NeXt blocks $\boldsymbol { B } _ { k }$ processes $\mathbf { F } ^ { k - 1 }$ through spatial convolution, channel mixing, and channel attention to produce the intermediate BEV features $\widetilde { \mathbf { F } } ^ { k }$ at resolution $\left( H / r _ { k } \right) \times \left( W / r _ { k } \right)$ . These features are shared by the point and BEV updates.

To update the point descriptors, we sample $\widetilde { \mathbf { F } } ^ { k }$ at each point’s horizontal cell. For a BEV feature map $\mathbf { F }$ at cumulative stride $^ { r , }$ the sampled vector for point i is denoted by $\mathcal { G } _ { r } ( \mathbf { F } ) _ { i } \mathbf { : }$

$$
\mathcal { G } _ { r } ( \mathbf { F } ) _ { i } = \mathbf { F } [ b _ { i } , : , \lfloor v _ { i } / r \rfloor , \lfloor u _ { i } / r \rfloor ] .\tag{3}
$$

We concatenate the sampled features with the preceding descriptor of the same point and apply a shared point-wise MLP $\phi _ { k }$ to obtain the updated descriptor $\mathbf { q } _ { i } ^ { k }$ :

$$
\begin{array} { r l } & { \widetilde { \mathbf { F } } ^ { k } = \boldsymbol { B } _ { k } ( \mathbf { F } ^ { k - 1 } ) , } \\ & { \mathbf { q } _ { i } ^ { k } = \phi _ { k } \big ( [ \mathcal { G } _ { r _ { k } } ( \widetilde { \mathbf { F } } ^ { k } ) _ { i } , \mathbf { q } _ { i } ^ { k - 1 } ] \big ) . } \end{array}\tag{4}
$$

These descriptors form $\mathbf { Q } ^ { k }$ , with the same channel width as $\widetilde { \mathbf { F } } ^ { k }$

![](images/d518e935b13077277fad13cd67a219588d9cfb2e1e7034c71eef53c94a9bb9e0.jpg)  
Fig. 3. Dual-scale BEV context (DSC) module.

At stages $k = 1 , 2 , 3 ,$ , we aggregate $\mathbf { Q } ^ { k }$ into a BEV tensor $\mathbf { S } ^ { k }$ for the next stage. At cumulative stride r, cell max pooling and scatter use the grouping indices $\mathcal { V } _ { i } ^ { r }$ for point i:

$$
\mathcal { V } _ { i } ^ { r } = \left( b _ { i } , h _ { i } , \lfloor v _ { i } / r \rfloor , \lfloor u _ { i } / r \rfloor \right) .\tag{5}
$$

As $r$ increases, horizontal cells become coarser while each point’s height partition $h _ { i }$ stays fixed. Let $S _ { r } ( \mathbf { Q } )$ denote the resulting BEV tensor after arranging the pooled height-partition descriptors as channels. The aggregate is then $\mathbf { S } ^ { k } \bar { = } \bar { S _ { r _ { k } } } ( \mathbf { Q } ^ { k } )$

The current stage’s BEV update combines the convolutional features $\widetilde { \mathbf { F } } ^ { k }$ with the incoming aggregate $\mathbf { S } ^ { k - 1 }$ . We use bilinear interpolation (Resize) to align $\bar { \mathbf { S } } ^ { k - 1 }$ with the current BEV resolution, downsampling at stages $2 { \mathrm { - } } 4 .$ . The convolutional fusion layer $\gamma _ { k }$ projects their concatenation to the stage’s BEV feature width (number of channels), producing $\mathbf { U } ^ { k }$ . An attention module $\eta _ { k }$ predicts sigmoid weights $\mathbf { A } ^ { k }$ that scale $\mathbf { U } ^ { k }$ before it is added to $\widetilde { \mathbf { F } } ^ { k }$ . The stage’s DSC module $\mathcal { D } _ { k }$ then produces the updated BEV features $\mathbf { F } ^ { k }$ . With ⊙ denoting element-wise multiplication, these updates are

$$
\begin{array} { r l } & { \mathbf { U } ^ { k } = \gamma _ { k } \big ( [ \mathrm { R e s i z e } ( \mathbf { S } ^ { k - 1 } ) , \widetilde { \mathbf { F } } ^ { k } ] \big ) , } \\ & { \mathbf { A } ^ { k } = \eta _ { k } ( \mathbf { U } ^ { k } ) , } \\ & { \mathbf { F } ^ { k } = \mathcal { D } _ { k } \big ( \widetilde { \mathbf { F } } ^ { k } + \mathbf { A } ^ { k } \odot \mathbf { U } ^ { k } \big ) . } \end{array}\tag{6}
$$

The resulting BEV features $\mathbf { F } ^ { k }$ and point descriptors $\mathbf { Q } ^ { k }$ are retained for fusion across levels. Stages 1–3 also pass them, together with $\mathbf { S } ^ { k }$ , to the next stage.

3) Dual-Scale BEV Context Module: Fig. 3 illustrates the DSC module D, which adds context from a coarser BEV grid to the incoming features.

For input BEV features F, we use C, H, and W for the current input dimensions. Average pooling $\mathrm { P o o l } _ { d }$ with kernel size and stride d produces a grid of $\lfloor H / d \rfloor \times \lfloor W / d \rfloor$ cells. $\mathrm { ~ A ~ 3 ~ } \times \mathrm { ~ 3 ~ }$ convolution maps the pooled features from $C$ channels to hidden width $C _ { h } .$ , followed by batch normalization and Hardswish. The features then pass through L Conv-SE-NeXt blocks with dilation $\delta$ and stride one. $\textbf { A 1 \times 1 }$ convolution followed by batch normalization maps them back to $C$ channels. We denote this sequence of operations by κ. Bilinear interpolation (Resize ${ } _ { : H , W } )$ upsamples the coarse features to the input resolution $H \times W$ , yielding the context features $\mathbf { F } _ { \mathrm { c t x } } { \mathrm { : } }$

$$
\mathbf { F } _ { \mathrm { c t x } } = \operatorname { R e s i z e } _ { H , W } \big ( \kappa ( \operatorname { P o o l } _ { d } ( \mathbf { F } ) ) \big ) ,\tag{7}
$$

We concatenate the input F with $\mathbf { F } _ { \mathrm { c t x } }$ and apply a $1 \times 1$ convolution from $2 C$ to $C$ channels, followed by the sigmoid function $\sigma ,$ to obtain a gate G. The gate weighs the context at each spatial location and channel before it is added to the input, yielding the DSC output ${ \bf F } _ { \mathrm { d s } }$ with the same dimensions as the input:

$$
\begin{array} { r } { \mathbf { G } = \sigma \left( \mathrm { C o n v } _ { 1 \times 1 } \left( \left[ \mathbf { F } , \mathbf { F } _ { \mathrm { c t x } } \right] \right) \right) , } \\ { \mathbf { F } _ { \mathrm { d s } } = \mathcal { D } ( \mathbf { F } ) = \mathbf { F } + \mathbf { G } \odot \mathbf { F } _ { \mathrm { c t x } } . } \end{array}\tag{8}
$$

4) Fuse Levels 0–4: We combine the stem and four stage outputs $\{ \mathbf { F } ^ { j } , \mathbf { Q } ^ { j } \} _ { j = 0 } ^ { 4 }$ within each branch. The coarser BEV feature maps are upsampled to the stem resolution $H \times W$ by bilinear interpolation. We then concatenate all five feature maps along the channel dimension and apply 2D convolutions to obtain the fused BEV features $\mathbf { F } _ { \mathrm { d e c } }$ . The point descriptors remain aligned with the same N points across levels. We concatenate each point’s descriptors from these levels and apply a shared point-wise MLP to obtain the fused point descriptors $\mathbf { Q } _ { \mathrm { d e c } } = \{ \mathbf { q } _ { \mathrm { d e c } , i } \} _ { i = 1 } ^ { N }$

## D. Point–BEV Fusion Head

The fusion head combines the backbone outputs $\mathbf { F } _ { \mathrm { d e c } }$ and $\mathbf { Q } _ { \mathrm { d e c } }$ for point-wise prediction. We sample the BEV features at each point using $\mathcal { G } _ { 1 } . \mathrm { A }$ shared point-wise ${ \mathrm { M L P } } \mu$ maps these sampled features to the same width as $\mathbf { q } _ { \mathrm { d e c } , i } ,$ , and their sum forms the final descriptor $\mathbf { d } _ { i } .$ Let $K$ be the number of semantic classes. Dropout followed by a linear classifier produces the class-logit vector $\mathbf { o } _ { i } \in \mathbb { R } ^ { K }$ for point i:

$$
\begin{array} { r } { \mathbf { d } _ { i } = \mu ( \mathcal { G } _ { 1 } ( \mathbf { F } _ { \mathrm { d e c } } ) _ { i } ) + \mathbf { q } _ { \mathrm { d e c } , i } , } \\ { \mathbf { o } _ { i } = \mathrm { L i n e a r } ( \mathrm { D r o p o u t } ( \mathbf { d } _ { i } ) ) . } \end{array}\tag{9}
$$

Collecting these vectors gives $\mathbf { O } \in \mathbb { R } ^ { N \times K } ;$ ; class-wise argmax gives the predicted label for each retained point.

## E. Auxiliary Supervision and Training Objective

During training, auxiliary heads branch from the encoder and backbone to supervise intermediate features alongside the final point prediction. Each auxiliary classifier uses dropout followed by a convolution.

1) Shared 2D Auxiliary Head: The shared head receives the encoder tensor X. The same 2D classifier processes the features of the 3D cells in each height partition, producing logits $\mathbf { O } ^ { \mathrm { h t } } \in \mathbb { R } ^ { B \times Z \times K \times H \times W }$

Each prediction is supervised by labels from its corresponding 3D cell. For valid ground-truth point labels $y _ { i } \in$ $\{ 0 , \ldots , K - 1 \}$ , the target $\mathbf { Y } _ { b , h , v , u } ^ { \mathrm { 3 D } }$ is the most frequent valid label among points in that cell. Cells without valid labels are ignored.

2) Stage Auxiliary Heads: The three stage heads receive the BEV maps $\mathbf { F } ^ { 2 } , \ \mathbf { F } ^ { 3 }$ , and $\mathbf { F } ^ { 4 }$ (Fig. 2). Each BEV feature map is upsampled to the stem resolution $H \times W$ by bilinear interpolation and passed to its classifier, producing logits $\mathbf { O } ^ { a } \in$ $\mathbb { R } ^ { B \times \dot { K } \times H \times W }$ for $a \in \{ 2 , 3 , 4 \}$ . All three heads use the targets $\mathbf { Y } ^ { \mathrm { 2 D } } \in \mathbb { Z } ^ { B \times H \times W }$ , obtained by majority voting over all valid point labels in each horizontal cell across heights. Horizontal cells without valid labels are ignored.

3) Total Loss: The total objective L combines the point segmentation loss $\mathcal { L } _ { \mathrm { s e g } }$ and the auxiliary loss $\mathcal { L } _ { \mathrm { a u x } }$ , with λ weighting auxiliary supervision:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \lambda \mathcal { L } _ { \mathrm { a u x } } .\tag{10}
$$

The point segmentation loss supervises the fusion head logits O using the ground-truth point labels $\mathbf { y } = ( y _ { 1 } , \dots , y _ { N } )$ . It combines cross-entropy $\mathrm { C E } _ { \mathbf { w } , \mathbf { \xi } }$ with the Lovasz–Softmax loss´ $\mathcal { L } _ { \mathrm { L o v a s z } }$ [27]. Here cross-entropy uses class weights w (unit weights in all experiments) and label smoothing ϵ. Groundtruth labels designated as ignored by the dataset protocol do not contribute to the loss. Softmax acts along the class dimension:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s e g } } = \mathrm { C E } _ { \mathbf { w } , \epsilon } ( \mathbf { O } , \mathbf { y } ) + 0 . 3 \mathcal { L } _ { \mathrm { L o v a s z } } \left( \mathrm { S o f t m a x } ( \mathbf { O } ) , \mathbf { y } \right) . } \end{array}\tag{11}
$$

We compute one loss for the shared 2D auxiliary head using the predictions and corresponding 3D-cell targets from all $Z$ height partitions of each of the B scans. Each of the three stage auxiliary heads has a separate loss. Let $\ell _ { a }$ denote the loss of head $a \in \{ \mathrm { h t } , 2 , 3 , 4 \}$ . Their arithmetic mean over heads with valid targets gives $\mathcal { L } _ { \mathrm { a u x } } ;$ if no head has valid targets, this loss is zero.

For each auxiliary head $^ { a , }$ let $\mathbf { O } ^ { a }$ and $\mathbf { Y } ^ { a }$ denote its logits and targets. The shared head uses $\mathbf { Y } ^ { \mathrm { { h t } } } = \mathbf { Y } ^ { \mathrm { { 3 D } } }$ , while the stage heads use $\mathbf { Y } ^ { a } = \mathbf { Y } ^ { \mathrm { 2 D } }$

Each auxiliary loss combines cross-entropy with Lovasz–´ Softmax and a boundary-F1-based loss $\mathcal { L } _ { \mathrm { B D } }$ [28]. The boundary loss is computed from the precision and recall of prediction–target boundary overlap. With coefficients α and $\beta$ weighting the auxiliary Lovasz–Softmax and boundary terms,´ respectively, the loss for head a is

$$
\begin{array} { r l } & { \ell _ { a } = \mathrm { C E } _ { { \mathbf w } , \epsilon } ( { \mathbf O } ^ { a } , { \mathbf Y } ^ { a } ) } \\ & { \qquad + \alpha \mathcal { L } _ { \mathrm { L o v a s z } } \left( \mathrm { S o f t m a x } ( { \mathbf O } ^ { a } ) , { \mathbf Y } ^ { a } \right) } \\ & { \qquad + \beta \mathcal { L } _ { \mathrm { B D } } \left( \mathrm { S o f t m a x } ( { \mathbf O } ^ { a } ) , { \mathbf Y } ^ { a } \right) . } \end{array}\tag{12}
$$

## IV. EXPERIMENTS

To assess whether HiPFuse can provide accurate semantics at the scan rate under limited computational resources, we measure segmentation accuracy (Sec. IV-B) together with throughput and resource usage (Sec. IV-C) on the outdoor benchmarks SemanticKITTI [13] and MCD [14]. Ablation studies (Sec. IV-D) examine how the proposed components contribute to segmentation accuracy, while separate training and evaluation on Semantic-ITC [15] assess indoor applicability (Sec. IV-B3). To determine whether these predictions benefit downstream robotic perception, we replace only the semantic input in fixed odometry, loop retrieval, and semantic mapping systems (Sec. IV-E).

## A. Experimental Setup

1) Datasets: For MCD [14], we use the NTU subset of outdoor campus scans acquired with a Livox Mid-70 LiDAR using a non-repetitive scan pattern. We use 24 learning classes and the published split of three training sequences and one test sequence. Following the released baselines, the test sequence is also used for checkpoint selection because no separate validation split is provided.

TABLE I  
KEY HIPFUSE SETTINGS. BEV WIDTHS AND CONV-SE-NEXT BLOCK COUNTS FOLLOW STAGES 1–4.
<table><tr><td>Setting</td><td>MCD</td><td>SemanticKITTI</td><td>Semantic-ITC</td></tr><tr><td>z range (m)</td><td> $[ - 4 . 5 , 2 4 . 0 )$ </td><td> $[ - 4 . 0 , 2 . 5 )$ </td><td> $[ - 4 . 0 , 8 . 0 )$ </td></tr><tr><td>Cell size  $( s _ { x } , s _ { y } )$  (m)</td><td> $0 . 6 \times 0 . 4$ </td><td> $\mathrm { { 0 . 3 5 \times 0 . 3 5 } }$ </td><td>0.1875 × 0.1875</td></tr><tr><td>Height partitions Z</td><td>6</td><td>3</td><td>8</td></tr><tr><td>BEV widths (1–4)</td><td>64/128/256/256128/128/128/12864/128/256/256</td><td></td><td></td></tr><tr><td>Blocks (1–4)</td><td>2/2/2/2</td><td>1/1/1/1</td><td>2/2/2/2</td></tr></table>

SemanticKITTI [13] contains urban road scans acquired with a Velodyne HDL-64E spinning LiDAR. We use 19 learning classes and the standard split of ten training, one validation, and eleven test sequences. Main comparisons use the official test set, while component and partition-count ablations use the validation sequence.

Semantic-ITC [15] contains indoor building scans acquired with a Livox Mid-360 LiDAR. We use 16 learning classes and its official sequence-level split of 37 training, 5 validation, and 10 test sequences.

In all segmentation benchmarks, the input consists only of coordinates and intensity from the current scan.

2) Spatial and Network Configurations: Table I summarizes the dataset-specific spatial and backbone settings. All three datasets use a stride-one BEV resolution of 256 × 256. All DSC instances use pooling stride d = 2 and dilation δ = 2, with L = 2 after stem fusion and L = 1 after each stage.

3) Training Protocol: We train our models on a single NVIDIA L40 GPU with mixed precision, using AdamW [29] and a cosine learning-rate schedule with warmup. For MCD, SemanticKITTI, and Semantic-ITC, respectively, batch sizes are 8, 4, and 2; initial learning rates are $8 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 }$ , and $8 \times 1 0 ^ { - 4 }$ . The maximum numbers of training epochs are 300, 100, and 150, respectively.

The same early-stopping rule is used across all three datasets: after at least 30 epochs, training stops if the mIoU used for checkpoint selection does not improve for ten consecutive epochs. Checkpoint selection uses MCD ntu\_day\_10, SemanticKITTI sequence 08, and the official Semantic-ITC validation split, respectively.

4) Augmentation and Losses: We augment training scans on all three datasets with flipping, rotation, scaling, and translation; MCD and Semantic-ITC additionally use point dropout and jitter, while SemanticKITTI uses instance CutMix [20]. The auxiliary loss weight is $\lambda = 0 . 2$ on MCD and Semantic-ITC and $\lambda = 1 . 0$ on SemanticKITTI; only SemanticKITTI adds auxiliary Lovasz–Softmax [27] and boundary-F1-based´ terms [28], with $\alpha = 1 . 5$ and $\beta = 1 . 0 ,$ , respectively.

5) Evaluation Protocol: For the main comparisons, we train our models with a fixed random seed of 0 for reproducibility. For baselines evaluated locally, we use the provided implementations and their default prediction procedures.

We evaluate segmentation accuracy using FP32 inference with batch size one, without test-time augmentation or point subsampling. Following WaffleIron [20], we assign each point outside the predefined 3D region the predicted label of its nearest retained point in the same scan under 3D Euclidean distance. We report class-wise intersection over union (IoU) and mIoU over all valid ground-truth points.

## B. Semantic Segmentation Results

1) MCD: Our method leads the MCD [14] comparison with 62.5% mIoU (Table II) and achieves the highest reported IoU in 19 of the 24 classes, including a tie on stairs. The largest margins over the next-best results occur on fence and other structures, while bike and pedestrian remain well below the best compared results.

Class-wise results for SalsaNext [6], SPVCNN [12], Cylinder3D [17], SDSeg3D [30], WaffleIron [20], and S-Former [31] are taken from the MCD dataset paper [14]. We train HARP-NeXt [11] using the same MCD training protocol and hyperparameters as our method and report its seed-0 result.

2) SemanticKITTI: Our method reaches 63.7% test mIoU on SemanticKITTI [13] (Table III), with the highest IoUs for road and sidewalk and the second-highest IoUs for parking, building, fence, and bicyclist in this comparison. Improvements over HARP-NeXt [11] cover 17 of the 19 classes; truck and traffic-sign are the two exceptions. Relative to WaffleIron [20], larger deficits on motorcyclist, other-vehicle, and person outweigh the gains on road and sidewalk in the overall mIoU.

Results for SalsaNext [6], WaffleIron, Cylinder3D [17], and SPVNAS [12] are taken from WaffleIron [20]. Results for RangeRet [32] and PolarNet [8] are taken from their respective papers. Because HARP-NeXt [11] reports only validation results, we use its released checkpoint and official code. Test results for both HARP-NeXt and our method are obtained from the official SemanticKITTI benchmark server under the singlescan evaluation protocol [13].

3) Indoor Evaluation: On Semantic-ITC [15], our method achieves 71.6% mIoU (Table IV), exceeding SalsaNext [6] but remaining below HARP-NeXt [11] and WaffleIron [20]. It ranks second among the compared methods on floor, table, sofa, cabinet/shelf, and light. Compared with HARP-NeXt [11], it gains 14.8 percentage points on cabinet/shelf, while the deficits on stair and board account for most of its overall mIoU gap.

4) Qualitative Comparison: Fig. 4 compares our predictions with those of SalsaNext [6] and HARP-NeXt [11], using the same points and view within each row. In the highlighted regions, our predictions more closely follow the reference road/sidewalk boundary on SemanticKITTI [13], recover an MCD [14] vehicle largely classified as building by both baselines, and reduce wall-to-door confusion on Semantic-ITC [15].

## C. Efficiency and Resource Analysis

We complement the accuracy comparisons with segmentation throughput and peak GPU memory measurements on an NVIDIA L40 and a 64 GB Jetson AGX Orin (Table V). The Orin runs in MAXN mode with fixed CPU, GPU, and memory clock frequencies. Measurements cover the MCD [14]

<table><tr><td colspan="17">TABLE II CLASS-WISE IOU AND MIOU (%) ON THE MCD NTU TEST SEQUENCE. BOLD: BEST; UNDERLINE: SECOND BEST.</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Method</td><td>Category</td><td>Year Barrier</td><td>Bike Cha Building</td><td>CiF</td><td>Curhb Container</td><td>Fence Hydrant</td><td>Sign</td><td>Lanemark</td><td>Other Parkinglot</td><td>Ppole Pedestrian</td><td>Road Shelter</td><td>Sidewalk</td><td>Stairs</td><td>Structure-O</td><td>Traffic-C</td><td>Trunk</td><td>Vegetation Vehicle</td><td>mIoU (%)</td></tr><tr><td>SalsaNext [6]</td><td>Projection-based</td><td>2020 49.6</td><td>0.2</td><td>81.3 37.2 6.4</td><td>0.048.7</td><td>36.7</td><td>9.3 35.5</td><td>45.7 30.2</td><td>12.2</td><td>11.8 27.5 91.4</td><td>43.7</td><td>66.3 0.0</td><td></td><td>0.0</td><td>8.1 31.1</td><td>84.0</td><td></td><td>33.5</td></tr><tr><td>WaffleIron [20]</td><td>Projection-based</td><td>2023 76.7</td><td>28.3</td><td>81.6 43.9 6.9</td><td>6.9 57.2</td><td>37.7</td><td>14.7 57.9</td><td>53.5 11.7</td><td>39.7</td><td>52.5</td><td>56.7 91.5 67.3</td><td></td><td>77.0 65.8</td><td>42.2</td><td>12.4 59.0</td><td>88.2</td><td>48.2 75.9</td><td>50.2</td></tr><tr><td>Cylinder3D [17]</td><td>Voxel-based</td><td>2021 70.5</td><td>1.4</td><td>90.1 42.3 9.2</td><td>19.5 43.4</td><td>31.6</td><td>16.2 65.4</td><td>48.5</td><td>1.4 13.6</td><td>41.1</td><td>65.1 90.5 60.7</td><td></td><td>58.7 41.3</td><td>0.6</td><td>10.3 57.2</td><td>91.6</td><td>72.1</td><td>43.5</td></tr><tr><td>SDSeg3D [30]</td><td>Voxel-based</td><td>2022</td><td>72.2 57.9 82.2</td><td>57.3 0.8</td><td>0.047.6</td><td>31.4</td><td>24.4 67.1</td><td>49.1</td><td>0.0 12.5</td><td>45.3</td><td>61.091.6 69.0</td><td></td><td>64.8 49.0</td><td>0.0</td><td>11.1 56.0</td><td>89.4</td><td>80.0</td><td>46.7</td></tr><tr><td>S-Former [31]</td><td>Voxel-based</td><td>2023 79.1</td><td>46.1 88.7</td><td>50.4 16.6</td><td>23.1 61.2</td><td>46.7</td><td>51.2 75.4</td><td>55.1 10.4</td><td>45.1</td><td>66.6 73.9</td><td>93.6 78.1</td><td></td><td>77.8 72.6</td><td>37.6</td><td>26.4 63.3</td><td>92.3</td><td>83.7</td><td>59.0</td></tr><tr><td>SPVCNN [12]</td><td>Fusion-based</td><td>2020 65.5</td><td>56.1 81.4 44.9</td><td>8.6</td><td>16.6 50.6</td><td>37.0</td><td>22.0 65.8</td><td>49.7 1.3</td><td>28.2</td><td>46.3 58.1</td><td>91.7 62.9</td><td>93.8 49.1</td><td></td><td>6.1</td><td>8.9 51.3</td><td>89.5</td><td>71.4</td><td></td></tr><tr><td>HARP-NeXt [11] Ours</td><td>Fusion-based Fusion-based</td><td>2025 87.5 2026 93.9</td><td>1.2 90.8 2.4 96.4</td><td>49.6 16.8 59.7 27.1</td><td>23.3 56.3 23.5 61.3</td><td>52.1 26.7 74.0</td><td>66.9 62.1 80.3</td><td>54.3 51.7 55.3 51.4</td><td>20.1 27.4</td><td>20.8 60.3 32.8 78.9</td><td>93.1 64.0 93.7 80.1</td><td>75.2 20.2 78.9 72.6</td><td></td><td>40.2 59.4</td><td>35.6 59.0 70.0</td><td>92.5 94.8</td><td>55.0 50.6</td><td>48.2</td></tr><tr><td colspan="11">37.6 86.1 62.5</td></tr><tr><td colspan="11">TABLE III CLASS-WISE IOU AND MIOU (%) ON THE SEMANTICKITTI TEST SET. BOLD: BEST; UNDERLINE: SECOND BEST.</td></tr><tr><td></td><td></td><td></td><td></td><td>Truck</td><td></td><td>Bicyclist</td><td>Motorcyclist</td><td></td><td></td><td>Other-ground</td><td>Building</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Method</td><td>Category</td><td>Year</td><td>Car Bicycle</td><td>Motorcycle</td><td>Other-veh.</td><td>Person</td><td></td><td>Road</td><td>Parking</td><td>Sidewalk</td><td></td><td>Fence</td><td>Vegetation</td><td>Trunk Terrain</td><td>Pole Traffic-sign</td><td>mIoU (%)</td></tr><tr><td>SalsaNext [6] PolarNet [8]</td><td>Projection-based</td><td>2020 91.9</td><td>48.3</td><td>38.6 38.9</td><td>31.9</td><td>60.2 59.0</td><td></td><td>19.4 91.7</td><td>63.7</td><td>75.8 29.1</td><td>90.2 64.2</td><td></td><td>81.8 63.6</td><td>66.5</td><td>54.3 62.1</td><td>59.5</td></tr><tr><td>WaffleIron [20]</td><td>Projection-based</td><td>2020 93.8</td><td>40.3</td><td>30.1 22.9</td><td>28.5</td><td>43.2 40.2</td><td></td><td>5.6 90.8</td><td>61.7 74.4</td><td>21.7</td><td>90.0 61.3</td><td></td><td>84.0 65.5</td><td>67.8 51.8</td><td>57.5</td><td>54.3</td></tr><tr><td>RangeRet [32]</td><td>Projection-based</td><td>2023 97.2</td><td>70.0</td><td>69.8 40.4</td><td>59.6 77.1</td><td>75.5</td><td>41.5</td><td>90.6</td><td>70.4 76.4</td><td>38.9</td><td>93.5</td><td>72.3</td><td>86.7 75.7</td><td>71.7 66.2</td><td>71.9</td><td>70.8</td></tr><tr><td>Cylinder3D [17]</td><td>Projection-based</td><td>2026 93.5</td><td>50.6</td><td>55.4 51.3</td><td>52.9 67.3</td><td>66.3</td><td>27.3</td><td>90.3</td><td>68.5 73.5</td><td>32.0</td><td>90.1</td><td>63.3</td><td>84.5 65.6</td><td>70.4 56.0</td><td>65.8</td><td>64.5</td></tr><tr><td></td><td>Voxel-based</td><td>2021 97.1</td><td>67.6</td><td>64.0 50.8</td><td>58.6 73.9</td><td>67.9</td><td>36.091.4</td><td>65.1</td><td>75.5</td><td>32.3</td><td>91.0 66.5</td><td></td><td>85.4 71.8</td><td>68.5 62.6</td><td>65.6</td><td>67.8</td></tr><tr><td>SPVNAS [12]</td><td>Fusion-based</td><td>2020 97.3</td><td>51.5</td><td>50.8 59.8</td><td>58.8 65.7</td><td>65.2</td><td>43.7 90.2</td><td>67.6</td><td>75.2</td><td>16.9</td><td>91.3 65.9</td><td>86.1</td><td>73.4</td><td>71.0 64.2</td><td>66.9</td><td>66.4</td></tr><tr><td>HARP-NeXt [11] Fusion-based Ours</td><td>Fusion-based</td><td>2025 95.3 2026 95.5</td><td>53.6 63.4</td><td>48.3 41.8 57.9 37.6</td><td>39.5 59.4 40.5 60.5</td><td>61.6 68.5</td><td>0.8 90.1 16.992.1</td><td>68.2 69.1</td><td>74.0 77.0</td><td>25.7 26.6</td><td>89.6 63.8 92.6 68.0</td><td>84.0 84.6</td><td>68.4 68.5</td><td>68.4 57.2 70.2 61.5</td><td>63.6</td><td>60.7</td></tr><tr><td colspan="11">58.8 63.7 TABLE IV</td></tr><tr><td>CLASS-WISE IOU AND MIOU (%) ON THE SEMANTIC-ITC TEST SET. BOLD: BEST; UNDERLINE: SECOND BEST.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Vegetation</td><td></td><td></td><td></td></tr><tr><td>Method Category</td><td></td><td>Year</td><td>Wall Floor</td><td>Ceiling</td><td>Column</td><td>Beam Door</td><td>Stair</td><td>Table</td><td>Chair Sofa</td><td>Cab./shelf</td><td>Screen</td><td>Liht</td><td>Board</td><td>Other</td><td>mIoU (%)</td></tr><tr><td>SalsaNext [6]</td><td>Projection-based</td><td>2020</td><td>83.1</td><td>90.7</td><td>92.2 43.9</td><td>76.9</td><td>51.0 47.3</td><td>76.9</td><td>69.6</td><td>52.7</td><td>57.9 91.2</td><td>90.0</td><td>89.8</td><td>31.5 72.2</td><td></td><td>69.8</td></tr><tr><td>WaffleIron [20]</td><td>Projection-based</td><td>2023</td><td>88.4</td><td>93.4</td><td>93.8 55.7</td><td>81.0</td><td>60.7</td><td>79.6 83.0</td><td>80.2</td><td>73.8</td><td>67.3 93.8</td><td>97.2</td><td>94.7</td><td>44.8</td><td>81.1</td><td>79.3</td></tr><tr><td>HARP-NeXt [11]</td><td>Fusion-based</td><td>2025</td><td>88.5</td><td>92.1</td><td>94.4 55.5</td><td>80.1</td><td>56.1 63.8</td><td>79.9</td><td>75.6</td><td>68.2</td><td>43.2 86.7</td><td>94.5</td><td>91.2</td><td>55.6</td><td>77.9</td><td>75.2</td></tr><tr><td>Ours</td><td>Fusion-based</td><td>2026</td><td>86.7</td><td>92.6</td><td>93.7 53.2</td><td>76.7</td><td>52.1 37.5</td><td>81.7</td><td>69.8</td><td>71.5</td><td>58.0 85.9</td><td>95.1</td><td>90.1</td><td>27.1 73.4</td><td></td><td>71.6</td></tr></table>

NTU test set, the SemanticKITTI [13] validation set, and the Semantic-ITC [15] test set. Each method uses the same configuration on both devices for a given dataset.

Timing spans scan-file input through recovery of CPU label arrays in original point order, including preprocessing, inference, and postprocessing. Time per scan is total traversal time divided by scan count; FPS is scan count divided by total traversal time in seconds. GPU memory is measured as peak active allocation in MiB.

On Orin, our method reaches 22.4 FPS on MCD and 14.2 FPS on SemanticKITTI, exceeding the 10 Hz scan rate on both outdoor benchmarks. It is more accurate but slower than SalsaNext [6] on these benchmarks. Relative to HARP-NeXt [11], it combines higher test mIoU and throughput with lower GPU memory use on SemanticKITTI; on MCD, higher mIoU accompanies similar throughput and greater GPU memory use. The separately trained indoor model reaches 19.6 FPS on Semantic-ITC, also exceeding the scan rate.

## D. Ablation Studies

1) Component Ablation: All variants in Table VI retain the point–BEV backbone and three stage-level auxiliary heads.

The two baselines compare single-layer and height-partitioned aggregation with relative-z encoding, encoder-level supervision, and DSC disabled. Three further variants each remove one component: the two relative-z offsets, the shared encoderlevel auxiliary head, or the DSC modules. Removing the relative-z offsets preserves the point encoder, raw coordinates, and horizontal offsets.

For the full model, Table VI reports the mean ± sample standard deviation of mIoU over training seeds 0, 1, and 2. The standard deviation is approximately 0.2 percentage points on both datasets, indicating consistent performance across these runs.

With relative-z encoding, encoder-level supervision, and DSC disabled, height-partitioned aggregation does not improve on single-layer aggregation (Table VI). With these components retained, using Z = 6 on MCD and Z = 3 on SemanticKITTI improves mIoU over the single-layer setting (Z = 1) by 0.7 and 0.9 percentage points, respectively (Table VII). These results support the value of height partitioning within the integrated HiPFuse framework. The full model achieves seed-0 mIoUs of 62.5% on MCD and 64.8% on SemanticKITTI, exceeding all five variants. Removing height-aligned supervision causes the largest decrease on MCD. On SemanticKITTI, removing either this supervision or the relative-z offsets reduces mIoU to 64.0%, while the result without DSC remains close to the full-model mean. Component effects therefore differ across datasets.

![](images/a8be36a010c12a4af36b610bf6bd98713b892a6a1fe398b2af379457893e11d7.jpg)  
Fig. 4. Selected semantic predictions on (a) SemanticKITTI validation, (b) MCD test, and (c) Semantic-ITC test scans. Red boxes identify corresponding regions across methods. Row legends identify the colored classes; gray denotes other valid classes.

2) Sensitivity to Height Partitions: We vary the number of height partitions Z while keeping all other network hyperparameters and training settings fixed (Table VII). The highest evaluated mIoU is obtained with $Z = 6$ on MCD [14] and $Z = 3$ on SemanticKITTI [13], which are used in the corresponding models. The Z = 1 setting retains the full model’s encoding, auxiliary supervision, and DSC components. On MCD, Z = 6 and $Z = 8$ yield similar mIoU in this singleseed comparison. Since performance is not monotonic in Z, finer vertical partitioning is not uniformly beneficial.

## E. Downstream Robotic Perception

HiPFuse and SalsaNext [6] are the only evaluated methods with measured segmentation throughput above the 10 Hz scan rate on Orin across all three datasets (Table V). We therefore compare their semantic inputs with those of HARP-NeXt [11], our baseline fusion architecture, in three downstream tasks. All downstream experiments run on Orin under the hardware and clock settings in Sec. IV-C. Within each experiment, input scans, class mappings, downstream algorithms, and their parameters remain fixed while the semantic input changes.

All three tasks use SemanticKITTI [13] sequence 08 (4,071 scans). Odometry additionally uses MCD [14] sequence ntu\_day\_10 (3,232 scans) and a separate 3,155-scan indoor route recorded with our handheld platform, equipped with a Livox Mid-360 and its built-in IMU. The indoor route traverses a campus building with environments similar to those in Semantic-ITC [15]. All runs process every scan in their respective evaluation sequences.

1) Semantic-Assisted Odometry: Hybrid-VoxelMap [2] uses planar/nonplanar semantic group assignments to select planar or Gaussian voxel models and their matching residuals. We compare the three learned semantic inputs with Gaussianonly mapping, and include ground-truth semantics for both outdoor sequences.

On SemanticKITTI, Hybrid-VoxelMap [2] processes Velodyne LiDAR scans without IMU input and uses constantvelocity prediction. MCD uses Livox Mid-70 LiDAR and a VN100 IMU. We report outdoor odometry accuracy as the root mean square error (RMSE) of absolute trajectory error (ATE) after rigid SE(3) alignment.

TABLE V  
SEGMENTATION THROUGHPUT AND RESOURCE USAGE ON NVIDIA L40 AND ORIN. BOLD AND UNDERLINE MARK THE HIGHEST AND SECOND-HIGHEST FPS PER DATASET AND DEVICE.
<table><tr><td rowspan="2">Dataset Method</td><td rowspan="2"></td><td rowspan="2">Category</td><td rowspan="2">Params (M)</td><td colspan="3">NVIDIA L40 GPU</td><td rowspan="2"></td><td colspan="3">Orin</td></tr><tr><td>GPU Memory</td><td>(MiB)</td><td>Time/scan (ms)</td><td>GPU Memory (MiB)</td><td>Time/scan (ms)</td><td>FPS</td></tr><tr><td></td><td></td><td>Projection-based</td><td>6.7</td><td></td><td>294.5</td><td>13.2</td><td>FPS 75.7</td><td>186.7</td><td>29.9</td><td>33.5</td></tr><tr><td>MCD 14]</td><td>SalsaNext [6] SPVCNN [12]</td><td>Fusion-based</td><td>21.8</td><td></td><td>146.1</td><td>77.8</td><td>12.8</td><td>151.2</td><td>61.5</td><td>16.3</td></tr><tr><td></td><td>HARP-NeXt [11]</td><td>Fusion-based</td><td>5.4</td><td></td><td>384.9</td><td>20.3</td><td>49.1</td><td>276.9</td><td>45.8</td><td>21.8</td></tr><tr><td></td><td>Ours</td><td>Fusion-based</td><td>14.3</td><td></td><td>358.4</td><td>18.4</td><td>54.3</td><td>359.4</td><td>44.6</td><td>22.4</td></tr><tr><td></td><td></td><td>Projection-based</td><td>6.7</td><td></td><td>360.1</td><td>13.7</td><td>72.9</td><td>1167.0</td><td>64.2</td><td>15.6</td></tr><tr><td>Sem] an3]</td><td>SalsaNext [6] PolarNet [8]</td><td>Projection-based</td><td>13.6</td><td></td><td>1473.6</td><td>66.0</td><td>15.1</td><td>1345.6</td><td>177.6</td><td>5.6</td></tr><tr><td></td><td>WaffleIron [20]</td><td>Projection-based</td><td>6.8</td><td></td><td>1354.1</td><td>161.9</td><td>6.2</td><td>1235.1</td><td>529.1</td><td>1.9</td></tr><tr><td></td><td>RangeRet [32]</td><td>Projection-based</td><td>3.8</td><td></td><td>760.6</td><td>32.5</td><td>30.8</td><td>761.8</td><td>177.9</td><td>5.6</td></tr><tr><td></td><td>Cylinder3D [17]</td><td>Voxel-based</td><td>55.9</td><td></td><td>1394.6</td><td>107.4</td><td>9.3</td><td>1394.6</td><td>448.9</td><td>2.2</td></tr><tr><td></td><td>SPVNAS [12]</td><td>Fusion-based</td><td>10.8</td><td></td><td>681.7</td><td>29.4</td><td>33.9</td><td>682.1</td><td>212.2</td><td>4.7</td></tr><tr><td></td><td>HARP-NeXt [11]</td><td>Fusion-based</td><td>5.4</td><td></td><td>938.5</td><td>19.8</td><td>50.4</td><td>856.0</td><td>118.3</td><td>8.5</td></tr><tr><td></td><td>Ours</td><td>Fusion-based</td><td>7.2</td><td></td><td>659.1</td><td>12.6</td><td>79.0</td><td>660.1</td><td>70.5</td><td>14.2</td></tr><tr><td></td><td></td><td></td><td>6.7</td><td></td><td></td><td></td><td>111.7</td><td>81.1</td><td>33.5</td><td>29.8</td></tr><tr><td>SeannnITc</td><td>SalsaNext [6] WaffleIron [20]</td><td>Projection-based</td><td>6.8</td><td></td><td>285.2 462.4</td><td>8.9 166.0</td><td>6.0</td><td>256.3</td><td>133.7</td><td>7.5</td></tr><tr><td>[15]</td><td>HARP-NeXt [11]</td><td>Projection-based</td><td>5.4</td><td></td><td>459.8</td><td>12.8</td><td>78.0</td><td>350.4</td><td>54.1</td><td>18.5</td></tr><tr><td></td><td>Ours</td><td>Fusion-based Fusion-based</td><td>16.3</td><td></td><td>441.9</td><td>10.6</td><td>93.9</td><td>442.9</td><td>50.9</td><td>19.6</td></tr></table>

TABLE VI

COMPONENT ABLATION ON MCD TEST AND SEMANTICKITTI VALIDATION WITH FIXED TRAINING SETTINGS PER DATASET. FULL-MODEL RESULTS SUMMARIZE MULTIPLE TRAINING RUNS; OTHER VARIANTS USE SEED 0.
<table><tr><td colspan="2"></td><td rowspan="2">Relative-z encoding</td><td rowspan="2">Height-aligned BEV aux.</td><td rowspan="2">Dual-scale BEV context</td><td rowspan="2">mIoU on MCD (%)</td><td rowspan="2">mIoU on SemanticKITTI (%)</td></tr><tr><td>Variant</td><td>Projection</td></tr><tr><td>Single-layer point-BEV baseline</td><td>Single-layer BEV</td><td>X</td><td>X</td><td>X</td><td>60.6</td><td>64.3</td></tr><tr><td>Height-partitioned point-BEV baseline</td><td>Height-partitioned BEV</td><td>X</td><td>X</td><td>X</td><td>60.5</td><td>63.2</td></tr><tr><td>w/o dual-scale BEV context</td><td>Height-partitioned BEV</td><td>√</td><td>√</td><td>X</td><td>60.9</td><td>64.6</td></tr><tr><td>w/o height-aligned BEV aux.</td><td>Height-partitioned BEV</td><td>√</td><td>X</td><td>√</td><td>60.4</td><td>64.0</td></tr><tr><td>w/o relative-z offsets</td><td>Height-partitioned BEV</td><td>X</td><td>√</td><td>√</td><td>61.2</td><td>64.0</td></tr><tr><td>Full model</td><td>Height-partitioned BEV</td><td>√</td><td>√</td><td>√</td><td> $6 2 . 7 \pm 0 . 2$ </td><td> $6 4 . 6 \pm 0 . 2$ </td></tr></table>

TABLE VII

SENSITIVITY TO THE NUMBER OF HEIGHT PARTITIONS Z ON MCD TEST AND SEMANTICKITTI VALIDATION USING SEED 0. BOLD MARKS THE HIGHEST MIOU PER DATASET; – DENOTES AN UNEVALUATED SETTING.
<table><tr><td>Height partitions Z</td><td>mIoU on MCD (%)</td><td>mIoU on SemanticKITTI (%)</td></tr><tr><td>1</td><td>61.8</td><td>63.9</td></tr><tr><td>3</td><td>61.3</td><td>64.8</td></tr><tr><td>6</td><td>62.5</td><td>63.7</td></tr><tr><td>8</td><td>62.4</td><td>一</td></tr></table>

In this real-world indoor experiment, we evaluate the models trained on Semantic-ITC [15] using our handheld recordings. The route starts and ends at the same physical location. Because trajectory and semantic ground truth are unavailable, we assess endpoint consistency using the endpoint position discrepancy, $e _ { \mathrm { c l o s e } } = \| \mathbf { t } _ { T } - \mathbf { t } _ { 1 } \| _ { 2 }$ . Here $\mathbf { t } _ { 1 }$ and $\mathbf { t } _ { T }$ are the first and final estimated translations after initialization.

Semantic processing and SLAM run concurrently, following the original Hybrid-VoxelMap [2] pipeline. We replay scans at processing speed while preserving their timestamps and IMU order. Odometry system throughput is the number of output poses divided by the elapsed time from the first scan publication to receipt of the final pose, including initialization. SLAM runtime is the mean active processing time per scan. We also report peak CPU memory usage for the SLAM process and peak GPU memory usage for the semantic front end.

![](images/cd4ddd49ce42ca919e8dadf9d071cd99d87c1b62fcf624f2b8dec3eb52efe5ea.jpg)  
Fig. 5. Planar/nonplanar label grouping for Hybrid-VoxelMap [2] on a selected SemanticKITTI 08 scan. Blue and gray indicate correct planar and nonplanar assignments; yellow and red indicate false-planar and missed-planar assignments, respectively. Red boxes mark corresponding regions. The reference groups are derived from semantic labels.

Fig. 5 compares the predicted point-wise groups with those derived from ground-truth semantics. False-planar points are reference nonplanar points assigned to the planar group; missed-planar points are reference planar points assigned to the nonplanar group. Our method shows fewer assignments of both error types than SalsaNext [6] and HARP-NeXt [11] in this scan. In the highlighted region, it retains more of the reference planar structure, preserving planar-model assignments that are lost in the baseline predictions.

All three learned semantic inputs reduce ATE relative to Gaussian-only mapping on both outdoor sequences, with larger reductions on MCD (Table VIII). Our semantic input yields the lowest ATE among the learned inputs on both sequences. The improvement over SalsaNext [6] and HARP-NeXt [11] semantic inputs is modest on SemanticKITTI, with 3.71 m versus 3.76–3.77 m, and larger on MCD, with 1.74 m versus 1.83–2.17 m. Ground-truth semantics yield the lowest overall ATE, at 3.48 and 1.65 m, respectively.

With our semantic input, odometry system throughput reaches 11.1, 19.6, and 15.2 Hz on SemanticKITTI, MCD, and the indoor route, respectively. HARP-NeXt [11] semantic input yields 8.0 Hz on SemanticKITTI, below the 10 Hz scan rate. Its throughput on MCD and the indoor route, 19.4 and 15.3 Hz, is close to that obtained with our semantic input.

On the indoor route (Table IX), HARP-NeXt [11] and HiPFuse yield endpoint discrepancies of 0.02 m and 0.01 m, respectively, versus 5.09 m without semantics and 1.94 m with SalsaNext [6]. Together with the outdoor ATE reductions, these results suggest that the benefit of semantic input varies with the scene. In Hybrid-VoxelMap [2], selecting planar models enables point-to-plane constraints on suitable surfaces, so their contribution depends on the geometry available for scan matching.

The Semantic-ITC [15] results in Table IV show comparable IoUs for wall, floor, and ceiling, with most of our mIoU gap to HARP-NeXt arising from stair and board. All classes contribute equally to mIoU, while the recorded route contains few board objects and limited stair areas. The planar/nonplanar mapping assigns the same geometric model to several semantic classes, so confusions within a group leave model selection unchanged. The similarly small endpoint discrepancies are consistent with this class-wise accuracy pattern, the route’s class distribution, and the planar/nonplanar grouping.

TABLE VIII  
OUTDOOR SEMANTIC-ASSISTED ODOMETRY WITH HYBRID-VOXELMAP [2] ON ORIN. BOLD: LOWEST ATE AMONG LEARNED SEMANTIC INPUTS PER DATASET; –: INAPPLICABLE.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Sequence</td><td rowspan="2">Semantic input</td><td colspan="3">SLAM</td><td>Semantic front end</td><td rowspan="2">System throughput (Hz) ↑</td></tr><tr><td>ATE RMSE (m) ↓</td><td>Runtime (ms) ↓</td><td>CPU Memory (MiB) ↓</td><td>GPU Memory (MiB) ↓</td></tr><tr><td rowspan="5">SemanticKITTI [13]</td><td>08</td><td>Gaussian-only, no semantics</td><td>3.83</td><td>70.1</td><td>855.2</td><td></td><td>14.1</td></tr><tr><td>08</td><td>Ground-truth semantics</td><td>3.48</td><td>75.5</td><td>856.4</td><td></td><td>13.1</td></tr><tr><td>08</td><td>SalsaNext [6] semantics</td><td>3.77</td><td>81.8</td><td>852.2</td><td>184.4</td><td>11.2</td></tr><tr><td>08</td><td>HARP-NeXt [11] semantics</td><td>3.76</td><td>80.5</td><td>858.6</td><td>804.6</td><td>8.0</td></tr><tr><td>08</td><td>Our semantics</td><td>3.71</td><td>84.4</td><td>852.4</td><td>626.9</td><td>11.1</td></tr><tr><td rowspan="5">MCD [14]</td><td>ntu_day_10</td><td>Gaussian-only, no semantics</td><td>2.44</td><td>37.2</td><td>771.3</td><td></td><td>26.3</td></tr><tr><td>ntu_day_10</td><td>Ground-truth semantics</td><td>1.65</td><td>38.0</td><td>772.6</td><td></td><td>25.7</td></tr><tr><td>ntu_day_10</td><td>SalsaNext [6] semantics</td><td>1.83</td><td>43.0</td><td>771.4</td><td>112.0</td><td>22.5</td></tr><tr><td>ntu_day_10</td><td>HARP-NeXt [11] semantics</td><td>2.17</td><td>44.2</td><td>772.1</td><td>231.6</td><td>19.4</td></tr><tr><td>ntu_day_10</td><td>Our semantics</td><td>1.74</td><td>44.1</td><td>769.5</td><td>289.1</td><td>19.6</td></tr></table>

TABLE IX

INDOOR ENDPOINT POSITION DISCREPANCY WITH HYBRID-VOXELMAP [2] ON ORIN. BOLD: LOWEST DISCREPANCY AMONG LEARNED SEMANTIC INPUTS; –: INAPPLICABLE.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Sequence</td><td rowspan="2">Semantic input</td><td colspan="3">SLAM</td><td>Semantic front end</td><td rowspan="2">System throughput (Hz) ↑</td></tr><tr><td>Endpoint discrepancy (m)↓</td><td>Runtime (ms) ↓</td><td>CPU Memory (MiB) ↓</td><td>GPU Memory (MiB) ↓</td></tr><tr><td rowspan="4">Indoor campus</td><td>Campus route</td><td>Gaussian-only, no semantics</td><td>5.09</td><td>52.5</td><td>244.6</td><td></td><td>18.8</td></tr><tr><td>Campus route</td><td>SalsaNext [6] semantics</td><td>1.94</td><td>55.6</td><td>236.9</td><td>76.4</td><td>17.7</td></tr><tr><td>Campus route</td><td>HARP-NeXt [11] semantics</td><td>0.02</td><td>59.1</td><td>212.8</td><td>222.5</td><td>15.3</td></tr><tr><td>Campus route</td><td>Our semantics</td><td>0.01</td><td>59.1</td><td>217.7</td><td>364.0</td><td>15.2</td></tr></table>

2) Loop Retrieval: We evaluate loop retrieval with SGLC [3]. It builds foreground instance graphs from semantic labels, combines their spatial structure with background descriptors for retrieval, and verifies candidates through geometric matching between instances. Scans are replayed chronologically at a scheduled 10 Hz input rate, and semantic predictions are generated during replay. Each query accesses only previously processed scans, excludes the preceding 100 scans, and retains at most five candidates.

We also evaluate SegNet4D [33], the default semantic segmentation method used by SGLC [3]. In our evaluation, it uses eight scans available up to the query time, aligned with KITTI ground-truth poses, whereas SalsaNext [6], HARP-NeXt [11], and our method use single scans. We also evaluate retrieval with ground-truth semantic labels.

Among the 3,970 eligible queries, 321 have at least one eligible historical scan within 3 m, measured as the horizontal distance between their ground-truth positions. For these revisiting queries, Candidate Recall@5 measures the percentage with at least one such scan among the retained candidates. Accepted-loop recall measures the percentage for which such a scan is also accepted after geometric verification.

TABLE X  
LOOP RETRIEVAL WITH SGLC [3] ON SEMANTICKITTI 08, MEASURED ON ORIN. BOLD: BEST AMONG LEARNED SEMANTIC INPUTS.
<table><tr><td>Semantic input</td><td>Candidate Recall@5 (%) ↑</td><td>Accepted-loop recall (%) ↑</td><td>Semantic front-end service (ms) ↓</td></tr><tr><td>Ground-truth semantics</td><td>93.2</td><td>70.4</td><td></td></tr><tr><td>SalsaNext [6]</td><td>81.6</td><td>57.0</td><td>92.8</td></tr><tr><td>HARP-NeXt [11]</td><td>86.9</td><td>56.7</td><td>119.8</td></tr><tr><td>SegNet4D [33]†</td><td>90.7</td><td>63.6</td><td>413.9</td></tr><tr><td>Ours</td><td>91.9</td><td>65.4</td><td>98.5</td></tr></table>

<sup>†</sup>Eight scans aligned using ground-truth poses.

With our semantic input, SGLC [3] achieves 91.9% candidate recall and 65.4% accepted-loop recall (Table X), compared with 93.2% and 70.4% using ground-truth semantic input. Both measures exceed those obtained with all three learned baselines, including the multi-scan SegNet4D [33] reference.

Semantic front-end service time spans scan loading through point-label handoff to SGLC [3], including preprocessing, inference, postprocessing, and label-file writing. The mean semantic front-end service times of our method and SalsaNext [6] are both below the 100 ms scan interval at 10 Hz.

3) Semantic Mapping: We qualitatively compare semantic maps produced by SG-SLAM [4] using the three learned semantic inputs and ground-truth labels. It uses semantic labels to organize and track object instances across scans, maintaining both semantic graph and point cloud maps. We use the default settings with loop closure enabled. Each map combines semantic point clouds from keyframes using the final poses estimated in that run.

![](images/59f8e3c8b6a91bcf12c2fb00d55f2b1744e55fab22f98027b987d25c1de6aefe.jpg)  
Fig. 6. Semantic maps generated with SG-SLAM [4] on SemanticKITTI 08. Left: global map generated using our semantic input; the compared region is boxed in red. Right: local maps from four semantic inputs, each with its own estimated poses. Ground truth in the panel labels denotes the semantic labels used as input.

Around the two road junctions compared in Fig. 6, our result has fewer scattered non-road labels and a more continuous road-labeled surface than the results with SalsaNext [6] and HARP-NeXt [11]. The road-label distribution also more closely resembles that obtained with ground-truth semantic input.

## V. CONCLUSION

We presented HiPFuse, a point–BEV fusion framework for efficient LiDAR semantic segmentation from a single scan. Its integrated design retains height and point distinctions while enabling spatial context extraction through 2D processing for robotic perception.

The downstream evaluations show that HiPFuse’s semantic predictions benefit odometry, loop retrieval, and semantic mapping. Together with the measured edge-device throughput, this supports improving semantic input as a practical route to strengthening robotic perception under limited computational resources. The additional indoor evaluation supports applicability after separate training, although our model remains less accurate than the strongest baselines on that dataset.

Like the downstream systems it serves, HiPFuse is configured per sensor setup. A limitation is that its spatial grid, including the height range, cell size, and number of height partitions, is chosen together with supervised training, so adapting it to a new setup requires labeled data and retraining. This motivates further work on adaptive height partitioning and, more broadly, on universal LiDAR segmentation models across datasets and sensors [34] that maintain scan-rate inference on edge devices.

## REFERENCES

[1] X. Chen, A. Milioto, E. Palazzolo, P. Giguere, J. Behley, and C. Stach-\` niss, “SuMa++: Efficient LiDAR-based semantic SLAM,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2019, pp. 4530–4537, doi: 10.1109/iros40897.2019.8967704.

[2] H. Wu, G. Vosselman, and V. Lehtola, “Gaussian or plane? Both: Semantic-driven voxel representation for LiDAR–inertial odometry,” IEEE Robot. Autom. Lett., vol. 11, no. 1, pp. 161–168, 2026, doi: 10.1109/LRA.2025.3632730.

[3] N. Wang, X. Chen, C. Shi, Z. Zheng, H. Yu, and H. Lu, “SGLC: Semantic graph-guided coarse-fine-refine full loop closing for LiDAR SLAM,” IEEE Robot. Autom. Lett., vol. 9, no. 12, pp. 11 545–11 552, 2024, doi: 10.1109/lra.2024.3495455.

[4] N. Wang, H. Lu, Z. Zheng, Y.-H. Liu, and X. Chen, “Leveraging semantic graphs for efficient and robust LiDAR SLAM,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2025, pp. 1614–1621, doi: 10.1109/iros60139.2025.11246943.

[5] A. Milioto, I. Vizzo, J. Behley, and C. Stachniss, “RangeNet++: Fast and accurate LiDAR semantic segmentation,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2019, pp. 4213–4220, doi: 10.1109/ iros40897.2019.8967762.

[6] T. Cortinhal, G. Tzelepis, and E. E. Aksoy, “SalsaNext: Fast, uncertaintyaware semantic segmentation of LiDAR point clouds,” in Advances in Visual Computing – ISVC 2020, ser. Lecture Notes in Computer Science, vol. 12510. Springer, 2020, pp. 207–222, doi: 10.1007/ 978-3-030-64559-5 16.

[7] K. Sirohi, R. Mohan, D. Buscher, W. Burgard, and A. Valada, “Effi-¨ cientLPS: Efficient LiDAR panoptic segmentation,” IEEE Trans. Robot., vol. 38, no. 3, pp. 1894–1914, 2022, doi: 10.1109/TRO.2021.3122069.

[8] Y. Zhang, Z. Zhou, P. David, X. Yue, Z. Xi, B. Gong, and H. Foroosh, “PolarNet: An improved grid representation for online LiDAR point clouds semantic segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 9601–9610, doi: 10.1109/ cvpr42600.2020.00962.

[9] L. Li, W. Ding, Y. Wen, Y. Liang, Y. Liu, and G. Wan, “A unified BEV model for joint learning of 3D local features and overlap estimation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2023, pp. 8341–8348, doi: 10.1109/ICRA48891.2023.10160492.

[10] D. Ye, Z. Zhou, W. Chen, Y. Xie, Y. Wang, P. Wang, and H. Foroosh, “LidarMultiNet: Towards a unified multi-task network for LiDAR perception,” in Proc. AAAI Conf. Artif. Intell. (AAAI), vol. 37, no. 3, 2023, pp. 3231–3240, doi: 10.1609/aaai.v37i3.25429.

[11] S. Abou Haidar, A. Chariot, M. Darouich, C. Joly, and J.-E. Deschaud, “HARP-NeXt: High-speed and accurate range-point fusion network for 3D LiDAR semantic segmentation,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2025, pp. 5946–5953, doi: 10.1109/iros60139.2025. 11246557.

[12] H. Tang, Z. Liu, S. Zhao, Y. Lin, J. Lin, H. Wang, and S. Han, “Searching efficient 3D architectures with sparse point-voxel convolution,” in Computer Vision – ECCV 2020, ser. Lecture Notes in Computer Science, vol. 12373. Springer, 2020, pp. 685–702, doi: 10.1007/978-3-030-58604-1 41.

[13] J. Behley, M. Garbade, A. Milioto, J. Quenzel, S. Behnke, C. Stachniss, and J. Gall, “SemanticKITTI: A dataset for semantic scene understanding of LiDAR sequences,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 9297–9307, doi: 10.1109/iccv.2019.00939.

[14] T.-M. Nguyen, S. Yuan, T. H. Nguyen, P. Yin, H. Cao, L. Xie, M. Wozniak, P. Jensfelt, M. Thiel, J. Ziegenbein, and N. Blunder, “MCD: Diverse large-scale multi-campus dataset for robot perception,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 22 304–22 313, doi: 10.1109/cvpr52733.2024.02105.

[15] H. Wu, M. Affan, G. Vosselman, and V. Lehtola, “Semantic-ITC: A frame-wise indoor mobile laser scanning dataset and benchmark for semantic segmentation,” 2026, arXiv:2609.18493. [Online]. Available: https://arxiv.org/abs/2609.18493

[16] Q. Hu, B. Yang, L. Xie, S. Rosa, Y. Guo, Z. Wang, N. Trigoni, and A. Markham, “RandLA-Net: Efficient semantic segmentation of largescale point clouds,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 11 105–11 114, doi: 10.1109/cvpr42600. 2020.01112.

[17] X. Zhu, H. Zhou, T. Wang, F. Hong, Y. Ma, W. Li, H. Li, and D. Lin, “Cylindrical and asymmetrical 3D convolution networks for LiDAR segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 9939–9948, doi: 10.1109/cvpr46437.2021.00981.

[18] H. Radi and W. Ali, “VolMap: A real-time model for semantic segmentation of a LiDAR surrounding view,” in ICML Workshop on AI for Autonomous Driving, 2019. [Online]. Available: https: //arxiv.org/abs/1906.11873

[19] J. Xu, R. Zhang, J. Dou, Y. Zhu, J. Sun, and S. Pu, “RPVNet: A deep and efficient range-point-voxel fusion network for LiDAR point cloud segmentation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 16 004–16 013, doi: 10.1109/iccv48922.2021.01572.

[20] G. Puy, A. Boulch, and R. Marlet, “Using a waffle iron for automotive point cloud semantic segmentation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 3379–3389, doi: 10.1109/iccv51070. 2023.00313.

[21] S. Li, X. Chen, Y. Liu, D. Dai, C. Stachniss, and J. Gall, “Multi-scale interaction for real-time LiDAR data segmentation on an embedded platform,” IEEE Robot. Autom. Lett., vol. 7, no. 2, pp. 738–745, 2022, doi: 10.1109/lra.2021.3132059.

[22] L. Li, X. Kong, X. Zhao, T. Huang, W. Li, F. Wen, H. Zhang, and Y. Liu, “SSC: Semantic scan context for large-scale place recognition,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2021, pp. 2092–2099, doi: 10.1109/iros51168.2021.9635904.

[23] L. Gan, R. Zhang, J. W. Grizzle, R. M. Eustice, and M. Ghaffari, “Bayesian spatial kernel smoothing for scalable dense semantic mapping,” IEEE Robot. Autom. Lett., vol. 5, no. 2, pp. 790–797, 2020, doi: 10.1109/lra.2020.2965390.

[24] Y. Pan, X. Zhong, L. Wiesmann, T. Posewsky, J. Behley, and C. Stachniss, “PIN-SLAM: LiDAR SLAM using a point-based implicit neural representation for achieving global map consistency,” IEEE Trans. Robot., vol. 40, pp. 4045–4064, 2024, doi: 10.1109/TRO.2024.3422055.

[25] X. Liu, J. Lei, A. Prabhu, Y. Tao, I. Spasojevic, P. Chaudhari, N. Atanasov, and V. Kumar, “SlideSLAM: Sparse, lightweight, decentralized metric-semantic SLAM for multirobot navigation,” IEEE Trans. Robot., vol. 41, pp. 6529–6548, 2025, doi: 10.1109/tro.2025.3629786.

[26] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, “Searching for MobileNetV3,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 1314–1324, doi: 10.1109/iccv.2019.00140.

[27] M. Berman, A. Rannen Triki, and M. B. Blaschko, “The Lovasz-Softmax´ loss: A tractable surrogate for the optimization of the intersection-overunion measure in neural networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4413–4421, doi: 10.1109/cvpr. 2018.00464.

[28] A. Bokhovkin and E. Burnaev, “Boundary loss for remote sensing imagery semantic segmentation,” in Advances in Neural Networks – ISNN 2019, ser. Lecture Notes in Computer Science, vol. 11555. Springer, 2019, pp. 388–401, doi: 10.1007/978-3-030-22808-8 38.

[29] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7

[30] J. Li, H. Dai, and Y. Ding, “Self-distillation for robust LiDAR semantic segmentation in autonomous driving,” in Computer Vision – ECCV 2022, ser. Lecture Notes in Computer Science, vol. 13688. Springer, 2022, pp. 659–676, doi: 10.1007/978-3-031-19815-1 38.

[31] X. Lai, Y. Chen, F. Lu, J. Liu, and J. Jia, “Spherical transformer for LiDAR-based 3D recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 17 545–17 555, doi: 10.1109/cvpr52729.2023.01683.

[32] S. Mosco, D. Fusaro, W. Li, and A. Pretto, “Revisiting retentive networks for fast range-view 3D LiDAR semantic segmentation,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2026, pp. 2499–2509, doi: 10.1109/wacv61042.2026.00246.

[33] N. Wang, R. Guo, C. Shi, Z. Wang, H. Zhang, H. Lu, Z. Zheng, and X. Chen, “SegNet4D: Efficient instance-aware 4D semantic segmentation for LiDAR point cloud,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 15 339–15 350, 2025, doi: 10.1109/TASE.2025.3568001.

[34] Y. Liu, L. Kong, X. Wu, R. Chen, X. Li, L. Pan, Z. Liu, and Y. Ma, “Multi-space alignments towards universal LiDAR segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 14 648–14 661, doi: 10.1109/CVPR52733.2024.01388.