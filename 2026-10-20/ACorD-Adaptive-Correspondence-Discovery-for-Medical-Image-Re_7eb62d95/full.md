# ACorD: Adaptive Correspondence Discovery for Medical Image Registration

Arghavan Rezvani University of California, Irvine rezvanid@uci.edu

Kun Han University of California, Irvine khan7@uci.edu

Xiaohui Xie University of California, Irvine xhx@uci.edu

## Abstract

Medical image registration is afundamental taskfor align ing anatomical structures across images. Exhaustively searching for point-wise correspondences is prohibitively expensive in both computation and memory, particularly for high-resolution 3D medical images. Most recent learning-based methods directly predict correspondences in a coarse-to-fine manner, but often propagate intermediate estimates across scales without explicitly weighting correspondence evidence, which can allow coarse-scale errors to persist during refinement. To address these challenges, we propose ACorD, an adaptive correspondence discovery framework designed to mitigate error propagation with an attention-guided process. ACorD first predicts multiple potential correspondence candidates through Correlation-guided Cross-scale Offset Prediction (CCOP), which combines correlation maps as spatial matching priors with cross-scalefeatures as global deformation context. Then, Multi-Head Deformable Aggregation (MHDA) aggregates these candidate correspondences using attention, producing a learned, non-uniform weighting over competing matches rather than treating them equally. The multihead design further captures complementary deformation patterns, ensuring both global alignment and local precision. By integrating adaptive offset prediction with selective, attention-based aggregation of correspondence evidence, ACorD enables accurate deformation refinement. Extensive experiments on four public medical registration benchmarks demonstrate that adaptive correspondence discovery within a multi-scale refinementframework improves registration robustness across diverse anatomical structures.

![](images/73cf08396cc72a1a0d542ed511e125ff436fff3889fd4c683089762ec27ccc78.jpg)  
Figure 1. Adaptive Correspondence Discovery. ACorD calculates attention between a feature from the fixed image and a set of sampled features from the moving image using predicted correspondence candidates. Predicted offsets guide the sampling, and the resulting attention weights determine the final correspondence.

## 1. Introduction

Medical image registration [14, 25, 37] aligns images across time points, modalities, or subjects by establishing anatomical correspondences. It supports applications such as disease monitoring, treatment planning, image-guided surgery, and population studies, but remains challenging across both large deformations and subtle local misalignments.

Traditional registration methods optimize image similarity with smoothness constraints [2, 3, 5, 31, 33, 35], but are computationally costly, initialization-sensitive, and challenged by complex deformations. Deep learning methods [9, 15] instead learn registration mappings end-toend, improving efficiency and performance across applications. Deep learning registration began with VoxelMorph [4], establishing encoder-decoder architectures [6, 7, 10, 13, 18, 36], with multi-scale coarse-to-fine frameworks [19, 23, 27, 28, 30, 32, 41, 42, 46] emerging as the dominant paradigm. However, they often propagate deformation estimates across scales without explicitly using correspondence evidence, allowing coarse-level errors to persist during refinement. Recent methods such as Dual-PRNet++ [19], Im2Grid [22], ModeT [40], CorrMLP[29], and SACB-Net [11] try to address this limitation by incorporating correlation layers to model voxelwise correspondences, but remain limited to fixed local neighborhoods, typically $3 \times 3 \times 3$ , limiting their ability to capture distant correspondences for large deformations or adapt search granularity for precise sub-voxel refinements.

The core challenge lies in balancing computational efficiency with correspondence flexibility. Fixed neighborhoods are efficient but rigid, while larger searches increase cost and ambiguity. Existing methods lack an adaptive strategy to selectively allocate resources to likely correspondence regions while dynamically adjusting search granularity to the deformation scale.

We propose a novel framework that addresses these limitations through attention-guided deformation propagation with adaptive correspondence discovery (ACorD). The key idea is that robust multi-scale registration requires deformation propagation guided by correspondence evidence rather than blind refinement. ACorD predicts multiple candidate correspondences and uses sparse attention to weight them. Learned offsets shift candidates beyond a fixed local neighborhood according to the observed deformation, avoiding exhaustive 3D correspondence search.

The main contributions of this work are:

(1) Attention-guided propagation paradigm: We introduce the first framework that employs attention mechanism to weight candidate correspondences before crossscale propagation, transforming coarse-to-fine registration from unweighted field estimation to learned, similaritydriven aggregation. (2) Adaptive correspondence discovery beyond fixed local windows: ACorD decouples candidate generation from aggregation: (1) Correlation-guided Cross-scale Offset Prediction (CCOP) leverages spatial correlation priors and cross-scale deformation context to predict multiple candidate locations, with learned offsets adaptively shifting the sampled candidate set relative to a reference neighborhood and (2) Multi-Head Deformable Aggregation (MHDA) then applies sparse attention at these candidate locations, using multi-head aggregation to weight candidates non-uniformly across the full deformation spectrum. (3) State-of-the-art performance across deformation scales: Extensive experiments on four medical datasets demonstrate consistent improvements-from datasets requiring precise local alignment to the challenging AbdomenCT dataset with large deformations—validating that adaptive and attention-guided registration excels across the entire deformation spectrum.

## 2. Background and Related Works

Let $ { \Omega } \subset \mathbb { R } ^ { 3 }$ denote the spatial domain and $I _ { m } , I _ { f } :$ $\Omega \ \to \ \mathbb { R }$ the moving (source) and fixed (target) images, sampled on an $H \times W \times D$ voxel grid. Registration estimates a transformation $\phi ~ : ~ \Omega ~ \to ~ \Omega$ mapping fixedimage coordinates to corresponding moving-image locations, such that $( I _ { m } \circ \phi ) ( \mathbf { x } ) = I _ { m } ( \phi ( \mathbf { x } ) )$ . We optimize $\phi ^ { * } = \arg \operatorname* { m i n } _ { \phi } ~ \mathcal { L } _ { \mathrm { s i m } } ( I _ { f } , I _ { m } \circ \phi ) + \lambda \mathcal { L } _ { \mathrm { r e g } } ( \phi )$ , where $\mathcal { L } _ { \mathrm { s i m } }$ measures image similarity and $\mathcal { L } _ { \mathrm { r e g } }$ regularizes the transformation.

Deep Learning-based Image Registration: Deep learning enables fast, end-to-end deformable registration. VoxelMorph [4] established the unsupervised paradigm, later extended with cycle consistency [20] and probabilistic modeling [13]. Transformer methods [7, 16, 36, 48] further model long-range dependencies, but mostly regress single-scale deformation fields, limiting simultaneous large-displacement and fine-detail alignment.

Multi-Scale Coarse-to-Fine Registration: Coarse-tofine methods progressively refine deformation fields across resolutions to handle large displacements and local refinements. Early deep models like LapIRN [32] adopted pyramid structures for iterative deformation refinement, followed by more efficient approaches such as NICE-Net [27], SD-Net [46], and PC-Reg [45]. Recursive pyramid designs in RDPNet [41] and IIRP-Net [24] further support dense multi-scale feature fusion. More recently, transformerbased methods such as PIViT [23] and NICE-Trans [28] integrated hierarchical attention to model long-range dependencies. These methods typically propagate intermediate estimates across scales, so coarse errors can affect later refinement. ACorD instead forms each refinement from a learned weighting of candidate correspondence evidence before propagation to the next scale.

Local Correlation Based Approaches: Voxel-wise correlation improves registration, particularly for large deformations, as popularized by RAFT [39]. Due to the memory demands of 3D medical data, recent methods adopt local window-based correlation. Dual-PRNet++ [19] integrated multi-scale features and correlation maps. im2grid [22] and ModeT [40] captured correspondences within local windows, with ModeT using transformer-based attention. CorrMLP [29] improved deformation learning by employing multi-scale MLP blocks for correlation modeling, while SACB-Net [11] used spatial-awareness convolutions to emphasize local structural context during matching. Although these methods enhance local matching, they are inherently constrained by fixed-size windows, which limits their ability to capture correspondences over long distances. ACorD instead learns offsets that shift candidate locations beyond the reference neighborhood, enabling adaptive search.

Deformable Modeling: Deformable modeling enables networks to adaptively sample spatial locations based on content, a concept first introduced in DCN [12]. This principle has been extended to attention mechanisms, such as in Deformable DETR [47] and Deformable ViT [43], which apply deformable sampling to attention queries. In medical image registration, TransDCA [8] adopts deformable crossattention to enhance feature extraction. However, these approaches primarily use deformable sampling to construct stronger feature representations, after which a separate prediction head estimates the deformation field. In contrast, ACorD uses deformable sampling directly for correspondence discovery: CCOP predicts multiple candidate correspondence locations, and MHDA weights them to refine the deformation field. This design lets the attention weights modulate the contribution of competing correspondences, enabling adaptive refinement across both large displacements and fine local misalignments.

## 3. Methodology

## 3.1. Architecture Overview

Fig. 2 illustrates the proposed Adaptive Correspondence Discovery (ACorD) framework, comprising: (1) a shared feature pyramid encoder that extracts hierarchical multiscale representations from moving and fixed images; (2) ACorD blocks that perform adaptive correspondence discovery and selective aggregation at each scale; and (3) coarse-to-fine deformation refinement across scales.

Rather than directly propagating each intermediate deformation field, ACorD refines it using learned correspondence weighting before passing it to the next finer scale, reducing the influence of poorly matched candidates during refinement.

## 3.2. Multi-Scale Feature Extraction

The shared encoder $\mathcal { E }$ extracts a hierarchy of L feature maps from the moving and fixed images:

$$
\{ F _ { m } ^ { ( l ) } \} _ { l = 1 } ^ { L } = E ( I _ { m } ) , \qquad \{ F _ { f } ^ { ( l ) } \} _ { l = 1 } ^ { L } = E ( I _ { f } ) .\tag{1}
$$

where $F _ { m } ^ { ( l ) } , F _ { t } ^ { ( l ) } \in \mathbb { R } ^ { C _ { l } \times H _ { l } \times W _ { l } \times D _ { l } }$ denote features at scale l. Each scale has spatial resolution $\textstyle \left[ { \frac { H } { 2 ^ { l - 1 } } } , { \frac { W } { 2 ^ { l - 1 } } } , { \frac { D } { 2 ^ { l - 1 } } } \right]$ and channel dimension $C _ { l } = C _ { 1 } \times 2 ^ { l - \bar { 1 } }$ , so that deeper levels capture increasingly rich semantic representations. Each encoder stage uses two convolutional layers with instance normalization and LeakyReLU activation. The resulting multi-scale features form the backbone for the subsequent attention-guided, coarse-to-fine registration process.

## 3.3. Correspondence-Aware Deformation Propagation

The decoder performs L stages of coarse-to-fine registration with ACorD Blocks. At the coarsest scale L, features $F _ { m } ^ { ( L ) }$ and $F _ { f } ^ { ( L ) }$ are processed by an ACorD block consisting of (1) Correlation-guided Cross-scale Offset Prediction (CCOP) and (2) Multi-Head Deformable Aggregation (MHDA), yielding the initial deformation field $\bar { \phi } ^ { ( L ) }$ . At each finer scale $l < L$ , the context feature $F _ { c } ^ { ( l + 1 ) }$ from the previous stage, the warped moving feature $\bar { F } _ { w } ^ { ( l ) } = F _ { m } ^ { ( l ) } \circ$ $\phi ^ { ( l + 1 ) }$ , and the fixed feature $F _ { f } ^ { ( l ) }$ are fed into the CCOP module to predict multi-head offsets $\{ \Delta _ { h } ^ { ( l ) } \} _ { h = 1 } ^ { \mathbb { H } _ { l } }$ , where $\mathbb { H } _ { l }$ is the number of heads at scale l. The MHDA module then performs deformable attention at these offset locations, with attention weights $\mathbf { A t t e n t i o n } _ { h } ^ { ( l ) }$ derived from query–key similarity, so that candidates better matching the fixed feature contribute more to the aggregated coordinate. The attended deformation patterns $A _ { h } ^ { \overline { { ( l ) } } }$ from all heads are regressed to produce the refinement $\dot { \Delta } \phi ^ { ( l ) }$ , which is composed with the previous deformation: $\phi ^ { ( l ) } = \phi ^ { ( l + 1 ) } \circ \Delta \phi ^ { ( l ) }$ . This process continues to the finest scale, yielding the final deformation field $\phi ^ { ( 1 ) }$

## 3.4. Adaptive Correspondence Discovery Block

Fig. 3 illustrates the core innovation of our framework: Adaptive Correspondence Discovery (ACorD) mechanism, which uses CCOP and MHDA for attention-guided correspondence discovery.

## 3.4.1. Correlation-guided Cross-scale Offset Prediction:

CCOP predicts correspondence candidates by estimating offsets relative to a reference neighborhood $\mathcal { N } _ { K } ( i , j , k ) =$ $\{ [ i + \delta _ { x } , j + \delta _ { y } , k + \delta _ { z } ] : \delta _ { x y z } \in [ - [ K / 2 ] , \lfloor K / 2 \rfloor ] ^ { 3 } \}$ where K is the search window size. Learning offsets allows the network to adaptively shift the search region toward locations with stronger feature correspondence, leveraging both local similarity and cross-scale motion context.

Feature Projection and Correlation Computation: At scale l, CCOP first applies shared projection layers to decouple feature extraction from deformation prediction. The fixed and warped moving image features are projected into a correspondence space using a shared MLP and layer normalization:

$$
\begin{array} { r } { P _ { f } ^ { ( l ) } = \operatorname { P r o j } ( F _ { f } ^ { ( l ) } ) , \quad P _ { m } ^ { ( l ) } = \operatorname { P r o j } ( F _ { w } ^ { ( l ) } ) } \end{array}\tag{2}
$$

where $F _ { w } ^ { ( l ) } = F _ { m } ^ { ( l ) } \circ \phi ^ { ( l + 1 ) }$ and $\phi ^ { ( l + 1 ) }$ is the accumulated deformation from previous scale $l + 1$ . The number of channels for the projected feature is $\mathbb { H } _ { l } \times C _ { h e a d } .$ , where $\mathbb { H } _ { l }$ is the number of the heads for scale l and $C _ { h e a d }$ represents the single-head dimension. The multi-head mechanism will be further introduced in Sec. 3.4.2 to capture various deformation pattern for robust registration.

Next, a local correlation volume $C ^ { ( l ) } \in \mathbb { R } ^ { H _ { l } \times W _ { l } \times D _ { l } \times L ^ { 3 } }$ is computed within a cubic neighborhood window of size $L \times L \times L$ to encode spatial similarity prior between the projected fixed and moving features [19, 29]. For each voxel location $( i , j , k )$ , the correlation with its neighboring voxel $( r _ { x } , r _ { y } , r _ { z } ) \in \mathcal { N } _ { L } ( i , j , k )$ is given by

$$
\begin{array} { r } { C ^ { ( l ) } [ i , j , k , r ] = \mathrm { m e a n } \Big ( P _ { f } ^ { ( l ) } [ i , j , k ] \odot P _ { m } ^ { ( l ) } [ r _ { x } , r _ { y } , r _ { z } ] \Big ) , } \end{array}\tag{3}
$$

where ⊙ denotes element-wise multiplication. This correlation volume provides the spatial prior necessary for the subsequent offset prediction step.

![](images/0fb23b8ff194fa84fa7552de205176ec152c5676122aadf9563ffe3f6c907aa5.jpg)  
Figure 2. Overview of the Proposed Method. A shared encoder extracts multi-scale representations from both images, which are processed by ACorD blocks in a coarse-to-fine manner. At each scale, ACorD combines Correlation-guided Cross-scale Offset Prediction (CCOP) with Multi-Head Deformable Aggregation (MHDA) to refine the deformation field through attention-guided correspondence discovery and propagation.

![](images/be5f06bf73607cd8e02259947408cc58f68ff8a93067964146cd26d0729b5587.jpg)  
Figure 3. Adaptive Correspondence Discovery (ACorD). ACorD consists of two modules: (1) CCOP predicts multi-head adaptive spatial offsets from projected features, local-window correlations, and cross-scale context, while generating features for the next scale. (2) MHDA uses these offsets to dynamically place attention keys, where queries come from fixed features, keys from warped moving features, and values from sampled positions. Attention weights modulate the contribution of each candidate and the multi-head design captures diverse deformation patterns.

Adaptive Offset Prediction: To integrate coarse-level global motion priors that could guide finer-scale refinement, the context feature $F _ { c } ^ { ( l + 1 ) }$ from the previous scale is first upsampled and then concatenated with the current-scale projected features and the correlation volume:

$$
F _ { C C O P } ^ { ( l ) } = \mathbf { C a t } ( [ P _ { f } ^ { ( l ) } , P _ { m } ^ { ( l ) } , \mathbf { U p } ( F _ { c } ^ { ( l + 1 ) } ) , C ^ { ( l ) } ] )\tag{4}
$$

The concatenated features are processed through a lightweight regression network consisting of three convolutional layers, batch normalization, and ReLU activation to jointly predict multi-head offsets $\{ \Delta _ { h } ^ { ( l ) } \} _ { h = 1 } ^ { \mathbb { H } _ { l } }$ and updated context features $F _ { c } ^ { ( l ) }$

$$
\{ \Delta _ { h } ^ { ( l ) } \} _ { h = 1 } ^ { \mathbb { H } _ { l } } , F _ { c } ^ { ( l ) } = \mathrm { O f f s e t P r e d i c t o r } ( F _ { C C O P } ^ { ( l ) } ) ,\tag{5}
$$

where $\mathbb { H } _ { l }$ denotes the number of attention heads, and each $\Delta _ { h } ^ { ( l ) } \ \in \ \mathring { \mathbb { R } } ^ { H _ { l } \times W _ { l } \times D _ { l } \times ( K \times K \times K ) \times 3 }$ encodes the predicted 3D offset $( \Delta x , \Delta y , \Delta z )$ of $K ^ { 3 }$ search region for each voxel in head h. The updated context feature $\bar { F } _ { c } ^ { ( l ) }$ is then propagated to the next finer scale to provide hierarchical motion context. The predicted offsets allows the network to dynamically adapt the search range according to deformation complexity, thus guide the subsequent MHDA module to capture global motion and fine-grained positioning for local refinements within a unified framework

## 3.4.2. Multi-Head Deformable Aggregation

MHDA performs dynamic correspondence aggregation by positioning attention keys at the updated neighbourhood location using the offset predicted by CCOP. Unlike previous fixed-grid attention, MHDA enables flexible correspondence discovery with sparse attention mechanism.

Dynamic Key Positioning: For each attention head $h \in \{ 1 , \dots , \mathbb { H } _ { l } \}$ at scale l and each query location $( i , j , k )$ the MHDA module samples $K ^ { 3 }$ keys from the projected warped moving features at dynamically updated locations using the predicted offsets from the CCOP module. The queries are derived from the projected fixed features, while keys are sampled from offset locations, and the values encode the spatial coordinates of these positions:

$$
Q _ { h } ^ { ( l ) } [ i , j , k ] = P _ { f } ^ { ( l ) } [ i , j , k ] _ { h } ,\tag{6}
$$

$$
\begin{array} { r } { K _ { h } ^ { ( l ) } [ i , j , k , r ] = \mathrm { S a m p l e } \big ( P _ { m \ h } ^ { ( l ) } , \mathcal { N } _ { K } ( i , j , k ) [ r ] + \Delta _ { h } ^ { ( l ) } [ i , j , k , r ] \big ) , } \end{array}\tag{7}
$$

$$
V _ { h } ^ { ( l ) } [ i , j , k , r ] = \mathcal { N } _ { K } ( i , j , k ) [ r ] + \Delta _ { h } ^ { ( l ) } [ i , j , k , r ] ,\tag{8}
$$

where $P _ { m } ^ { ( l ) } { } _ { h }$ and ${ P _ { f } ^ { ( l ) } } _ { h }$ denote the $C _ { h e a d } .$ -dimensional projected features for head h. The indices $i \in \{ 1 , \ldots , H _ { l } \}$ $j \in \{ 1 , \dots , W _ { l } \}$ , and $k \in \{ 1 , \ldots , D _ { l } \}$ specify the query voxel, while $r \in \{ 1 , \ldots , K ^ { 3 } \}$ enumerates the ${ \dot { K } } ^ { 3 }$ sampled positions. The function Sample(·, ·) performs trilinear interpolation, and $V _ { h } ^ { ( l ) } [ i , j , k , r ]$ stores the spatial coordinates of the sampled key positions for head h.

Attention-Based Correspondence Assessment: Let $d _ { k }$ be the key dimension. For each query position and attention head, the attention mechanism computes correspondence weights across the $K ^ { 3 }$ key positions:

$$
\mathrm { A t t e n t i o n } _ { h } ^ { ( l ) } [ i , j , k ] = \mathrm { S o f t m a x } \left( \frac { Q _ { h } ^ { ( l ) } [ i , j , k ] \cdot K _ { h } ^ { ( l ) } [ i , j , k , : ] } { \sqrt { d _ { k } } } \right)\tag{9}
$$

The attended coordinate for each head is then computed as:

$$
A _ { h } ^ { ( l ) } [ i , j , k ] = \sum _ { r = 1 } ^ { K ^ { 3 } } \mathrm { A t t e n t i o n } _ { h } ^ { ( l ) } [ i , j , k , r ] \cdot V _ { h } ^ { ( l ) } [ i , j , k , r ]\tag{10}
$$

Multi-Head Coordinate Aggregation: For each query position $( i , j , k )$ , we obtain H<sub>l</sub> attended coordinates from all attention heads. These coordinates are concatenated and processed through a simple regression head to produce the final deformation:

$$
\Delta \phi ^ { ( l ) } [ i , j , k ] = \mathrm { R e g H e a d } ( \mathrm { C a t } ( [ A _ { 1 } ^ { ( l ) } [ i , j , k ] , \dots , A _ { \mathbb { H } _ { l } } ^ { ( l ) } [ i , j , k ] ] ) )\tag{11}
$$

$$
\phi ^ { ( l ) } = \phi ^ { ( l + 1 ) } \circ \Delta \phi ^ { ( l ) }\tag{12}
$$

This multi-head design [1] helps to capture robust correspondence through complementary deformation patterns—each head can specialize in different types of anatomical motion while the concatenated representation provides comprehensive correspondence information.

## 3.5. Unsupervised Training Loss Function

Our method is trained with an unsupervised loss combining three components computed on the final deformation field ϕ: ${ \mathcal { L } } = { \mathcal { L } } _ { \mathrm { { s i m } } } + \lambda _ { \mathrm { { s m o o t h } } } { \mathcal { L } } _ { \mathrm { { s m o o t h } } } + \lambda _ { \mathrm { { j a c } } } { \mathcal { L } } _ { \mathrm { { j a c } } }$ . The similarity term uses normalized cross-correlation (NCC) to measure alignment quality: $\mathcal { L } _ { \mathrm { s i m } } ~ = ~ - N C C ( I _ { f } , I _ { m } \circ \phi )$ Smoothness regularization enforces spatially coherent deformations: $\begin{array} { r } { \mathcal { L } _ { \mathrm { s m o o t h } } = \sum _ { d \in \{ x , y , z \} } | | \nabla \phi _ { d } ^ { ( 1 ) } | | _ { 2 } ^ { 2 } } \end{array}$

The Jacobian regularization penalizes negative determinants to prevent folding: $\begin{array} { r l } { \mathcal { L } _ { j a c } } & { { } = } \end{array}$ $\frac { 1 } { N } \sum _ { p \in \Omega }$ relu $\left( - \left| J _ { \phi } ( \pmb { p } ) \right| \right)$ . In the above formula λ<sub>smooth</sub> and $\lambda _ { \mathrm { j a c } }$ balance smoothness and topology preservation.

## 4. Experiments

## 4.1. Datasets

We evaluated our framework on three public 3D brain MRI datasets and one abdominal CT dataset.

MindBoggle [21] consists of 101 brain MRI scans with 97 anatomical ROIs, including both cortical (62) and subcortical (35) structures. We used 70 scans for training, 7 for validation, and a separate set of 24 for cross-site testing to ensure generalizability.

LPBA40 [34] contains 40 healthy adult brain MRI scans with 54 manual anatomical labels. We used 25 scans for training, 5 for validation, and 10 for testing.

OASIS [26] contains 414 brain MRI scans with 35 anatomical labels from a cross-sectional cohort spanning young, middle-aged, nondemented, and demented older adults. We used 200 scans for training, 20 for validation, and the provided test pairs for testing.

AbdomenCT [44] comprises 30 abdominal CT scans with 13 anatomical RoIs. This dataset is particularly challenging due to significant anatomical variations, making it a challenging testbed for large-deformation registration. We used 18 scans for training, 4 for validation, and 8 for testing.

## 4.2. Implementation Details

Our method was implemented in PyTorch and trained on a single NVIDIA RTX A6000 GPU. Runtime, MACs, and peak GPU memory are reported in the supplementary material; ACorD takes 710 ms/pair and 16.9 GB peak memory, and is not the lightest model. We use a 5-scale pyramid and the local correlation neighborhood size is set to $L = 3$ to provide the spatial prior for offset prediction, and the dynamic attention search window K is set to 3. Training was performed using Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ and linear annealing. The loss function hyperparameters were set at $\lambda _ { \mathrm { s m o o t h } } = 1$ and $\lambda _ { \mathrm { j a c } } = 5 0$ after tuning.

## 4.3. Evaluation Metrics

We evaluate registration accuracy using the Dice Similarity Coefficient (DSC), measuring anatomical overlap, and Average Symmetric Surface Distance (ASSD) [17, 38] quantifies surface alignment accuracy. The smoothness and invertibility of the predicted spatial transformations are evaluated using the percentage of Negative Jacobian Determinants (NJD) [13], indicating topological violations.

Table 1. Quantitative comparisons on MindBoggle, LPBA, and OASIS datasets. Red, Green and Blue denote the 1st, 2nd and 3rd best.
<table><tr><td rowspan="2">Group</td><td rowspan="2">Method</td><td colspan="3">MindBoggle (97 ROIs)</td><td colspan="3">LPBA (54 ROIs)</td><td colspan="3">OASIS (35 ROIs)</td><td rowspan="2">#params (M)</td></tr><tr><td>DSC ↑</td><td>ASSD ↓</td><td>NJD(%) ↓</td><td>DSC↑</td><td>ASSD↓</td><td>NJD(%) ↓</td><td>DSC ↑</td><td>ASSD↓</td><td>NJD(%) ↓</td></tr><tr><td>Traditional</td><td>Affine SyN[2]</td><td>0.404±0.021 0.513±0.031</td><td>2.137±0.196 1.692±0.211</td><td>00</td><td>0.635±0.020 0.695±0.017</td><td>2.120±0.125 1.780±0.109</td><td>00</td><td>0.571±0.052 0.755±0.029</td><td>1.618±0.281 0.837±0.123</td><td>00</td><td></td></tr><tr><td rowspan="7">Single-Scale</td><td>VoxelMorph[4]</td><td>0.500±0.108</td><td>2.147±1.092</td><td>0.883±0.289</td><td>0.638±0.031</td><td>2.122±0.242</td><td>0.589±0.273</td><td>0.763±0.031</td><td>0.823±0.145</td><td>0.963±0.179</td><td>0.27</td></tr><tr><td>ViT-V-Net[6]</td><td>0.526±0.112</td><td>2.086±1.128</td><td>0.797±0.332</td><td>0.648±0.030</td><td>2.049±0.225</td><td>0.735±0.247</td><td>0.787±0.031</td><td>0.733±0.136</td><td>0.917±0.151</td><td>31.56</td></tr><tr><td>TransMorph[7]</td><td>0.551±0.118</td><td>1.963±1.083</td><td>0.691±0.181</td><td>0.663±0.030</td><td>1.982±0.232</td><td>0.648±0.199</td><td>0.806±0.025</td><td>0.661±0.103</td><td>0.827±0.169</td><td>46.77</td></tr><tr><td>TransMatch[10]</td><td>0.544±0.112</td><td>1.950±1.051</td><td>0.675±0.200</td><td>0.667±0.018</td><td>1.956±0.218</td><td>0.515±0.162</td><td>0.805±0.022</td><td>0.665±0.096</td><td>0.767±0.138</td><td>112.41</td></tr><tr><td>TransDCA[8]</td><td>0.550±0.118</td><td>1.944±1.071</td><td>0.604±0.161</td><td>0.661±0.031</td><td>1.998±0.243</td><td>0.672±0.240</td><td>0.803±0.023</td><td>0.674±0.097</td><td>0.818±0.166</td><td>219.67</td></tr><tr><td>H-ViT[16]</td><td>0.536±0.113</td><td>2.033±1.137</td><td>0.751±0.342</td><td>0.667±0.030</td><td>1.969±0.235</td><td>0.524±0.177</td><td>0.803±0.021</td><td>0.676±0.095</td><td>0.817±0.156</td><td>18.20</td></tr><tr><td></td><td>0.559±0.075</td><td>1.737±0.694</td><td>0.041±0.025</td><td>0.706±0.013</td><td>1.712±0.113</td><td>0.026±0.012</td><td>0.797±0.022</td><td>0.684±0.096</td><td>0.180±0.044</td><td>0.66</td></tr><tr><td rowspan="6">Multi-Scale</td><td>PIViT[23]</td><td>0.589±0.081</td><td>1.668±0.718</td><td>0.651±0.239</td><td>0.715±0.157</td><td>1.657±0.135</td><td>0.418±0.107</td><td>0.821±0.023</td><td>0.610±0.103</td><td>0.820±0.160</td><td>1.08</td></tr><tr><td>NICE-Net[27] NICE-Trans[28]</td><td>0.608±0.068</td><td>1.568±0.599</td><td>0.668±0.231</td><td>0.726±0.015</td><td>1.591±0.126</td><td>0.389±0.093</td><td>0.828±0.024</td><td>0.588±0.097</td><td>0.822±0.158</td><td>5.69</td></tr><tr><td>RPNet[24]</td><td>0.587±0.076</td><td>1.659±0.676</td><td>0.113±0.056</td><td>0.714±0.015</td><td>1.675±0.136</td><td>0.060±0.020</td><td>0.816±0.023</td><td>0.624±0.099</td><td>0.183±0.057</td><td>0.42</td></tr><tr><td>RDPNet[41]</td><td>0.608±0.0748</td><td>1.606±0.659</td><td>0.003±0.001</td><td>0.729±0.015</td><td>1.589±0.124</td><td>0.001±0.001</td><td>0.830±0.023</td><td>0.581±0.096</td><td>0.008±0.003</td><td>8.92</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dual-PR++[19]</td><td>0.562±0.102</td><td>1.899±0.974</td><td>0.258±0.111</td><td>0.695±0.021</td><td>1.798±0.183</td><td>0.040±0.018</td><td>0.807±0.021</td><td>0.657±0.095</td><td>0.449±0.106</td><td>1.24</td></tr><tr><td rowspan="5">Local-Corr.</td><td>Im2Grid[22]</td><td>0.556±0.098 0.614±0.051</td><td>1.768±0.788 1.472±0.418</td><td>0.019±0.006 0.008±0.006</td><td>0.703±0.015</td><td>1.715±0.129</td><td>0.008±0.003</td><td>0.804±0.021</td><td>0.666±0.095</td><td>0.028±0.012</td><td>0.89 1.03</td></tr><tr><td>ModeT[40]</td><td>0.617±0.057</td><td>1.533±0.528</td><td>0.599±0.291</td><td>0.729±0.013</td><td>1.573±0.110</td><td>0.002±0.001</td><td>0.824±0.023</td><td>0.600±0.095</td><td>0.013±0.006 0.761±0.143</td><td>4.20</td></tr><tr><td>CorrMLP[29] SACB-Net[11]</td><td>0.605±0.055</td><td>1.514±0.455</td><td>0.125±0.041</td><td>0.731±0.015 0.725±0.013</td><td>1.566±0.110 1.581±0.108</td><td>0.307±0.078</td><td>0.830±0.023 0.818±0.023</td><td>0.580±0.093 0.616±0.095</td><td>0.208±0.039</td><td>1.11</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.103±0.019</td><td></td><td></td><td></td><td></td></tr><tr><td>ACorD</td><td>0.624±0.049</td><td>1.468±0.447</td><td>0.005±0.004</td><td>0.736±0.015</td><td>1.549±0.123</td><td>0.002±0.001</td><td>0.830±0.024</td><td>0.580±0.099</td><td>0.006±0.003</td><td>6.47</td></tr></table>

Table 2. Quantitative comparison on AbdomenCT dataset. Red, Green and Blue denote the 1st, 2nd and 3rd best resutls.
<table><tr><td rowspan="2">Group</td><td rowspan="2">Method</td><td colspan="3">AbdomenCT (13 ROIs)</td></tr><tr><td>DSC ↑</td><td>ASSD ↓</td><td>NJD(%) ↓</td></tr><tr><td>Traditional</td><td>Affine SyN[2]</td><td>0.251±0.070 0.319±0.084</td><td>6.629±2.416 5.780±2.443</td><td>0 0</td></tr><tr><td>Single-Scale</td><td>VoxelMorph[4] ViT-V-Net[6] TransMorph[7] TransMatch[10] TransDCA[8]</td><td>0.349±0.051 0.385±0.054 0.409±0.061 0.420±0.064 0.420±0.058 4.317±1.071</td><td>4.793±1.047 4.547±1.032 4.422±1.118 4.258±1.097</td><td>1.746±0.094 4.798±1.400 5.554±1.368 5.188±1.430 4.497±1.375</td></tr><tr><td>Multi-Scale</td><td>PIViT[23] NICE-Net[27] NICE-Trans[28] RPNet[24] RDPNet[41]</td><td>0.416±0.044 4.005±0.714 0.464±0.058 3.666±0.893 0.524±0.047 0.480±0.055 0.554±0.052</td><td>3.034±0.689 3.543±0.723 3.023±0.757</td><td>0.367±0.179 6.375±2.100 14.938±2.943 1.700±0.827 0.227±0.216</td></tr><tr><td>Local-Corr.</td><td>Dual-PR++[19] Im2Grid[22] ModeT[40] CorrMLP[29] SACB-Net[11]</td><td>0.466±0.056 3.883±0.929 0.455±0.041 3.642±0.686 0.554±0.053 2.729±0.646 0.572±0.055 0.577±0.047</td><td>2.645±0.708 2.575±0.603</td><td>2.371±1.208 0.178±0.119 0.061±0.029 12.235±2.686 1.642±0.651</td></tr><tr><td>Proposed</td><td>ACorD</td><td>0.597±0.052 2.411±0.603</td><td></td><td>0.136±0.092</td></tr></table>

## 4.4. Comparison Methods

We compared ACorD with state-of-the-art registration methods, including 2 traditional optimization-based methods and 16 deep registration methods grouped into three categories (as discussed in Sec.2): Single-Scale methods [4, 6–8, 10, 16]; Multi-Scale methods [23, 24, 27, 28, 41]; and Local Correlation methods [11, 19, 22, 29, 40]. All methods were retrained using the same data splits and preprocessing pipelines. To preserve each method’s intended formulation, we follow its original training objective and regularization settings.

## 5. Results and Discussion

## 5.1. Brain Registration Results

Table 1 presents evaluations across three brain MRI datasets. ACorD consistently demonstrates superior performance across all evaluation metrics, including DSC, ASSD, and the percentage NJD. Single-scale methods generally underperform multi-scale and correlation-based approaches. Among multi-scale methods RDPNet achieves strong performance through recursive multi-step refinement and maintains excellent topology preservation via diffeomorphic registration. Local-window correlation methods show promising results, with ModeT and CorrMLP achieving competitive performance, though still constrained to fixed neighborhoods. Our method outperforms all methods across the three datasets, achieving the highest DSC and ASSD scores on MindBoggle and LPBA, while matching the best performance on OASIS. Notably, these improvements are achieved with a moderate model size (6.47M parameters), which is substantially smaller than transformerbased methods such as TransDCA, demonstrating that adaptive correspondence discovery and selective aggregation can improve accuracy without excessive model complexity. Fig. 4 shows qualitative LPBA results with improved anatomical alignment and fewer structural distortions.

## 5.2. AbdomenCT Registration Results

AbdomenCT is challenging due to large, non-linear organ deformations from breathing and positional changes. As such, it serves as a critical testbed for evaluating a registration model’s robustness to large displacements. As presented in Table 2, ACorD achieves state-of-the-art performance on this challenging dataset.

Compared to brain datasets, performance differences between method categories become more pronounced, emphasizing the importance of adaptive correspondence modeling under large deformations. SACB-Net is the strongest competitor, demonstrating the effectiveness of explicit structure modeling and correspondence mechanisms. However, its fixed neighborhood constraint limits effectiveness for the large displacements. By contrast, ACorD benefits from adaptive, context-conditioned correspondence discovery, enabling more effective alignment of distant structures while maintaining reasonable topology preservation. The results underscore that attention-guided propagation becomes increasingly valuable as deformation complexity increases. As further illustrated in Fig. 5, ACorD consistently achieves higher Dice scores across abdominal structures, confirming its robustness under large deformations.

![](images/8fbe2a0bc39c32c441a0de5dc62603ea49092c706475517f52f8c7d3d771602b.jpg)  
Figure 4. Visual comparisons on the Brain LPBA dataset (top two rows) and the Abdomen CT dataset (bottom two rows)

![](images/080ad13d4b439d356e94c4d7f49b82a9d5c1ede632dd888dda25a306f1a8d7c4.jpg)  
Figure 5. Dice score evaluation on AbdomenCT (13 ROIs)

Effect of Jacobian regularization. We additionally tuned $\lambda _ { \mathrm { j a c } }$ for the strongest competitors. CorrMLP reaches a best DSC of 0.580 and ModeT reaches 0.565, while SACB-Net does not improve over its original configuration. ACorD achieves 0.597 DSC under its selected setting. These results indicate that ACorD’s improvement cannot be explained solely by stronger Jacobian regularization.

Table 3. Ablation study on the AbdomenCT dataset. CG: Correlation Guidance; CS: Cross-Scale Feature Propagation; DA: Deformation Aggregation; MH: Multi-Head mechanism; Agg: aggregation rule (Attn: attention-weighted, Avg: uniform averaging); Offset: correspondence offset source (Learned: CCOP-predicted, Default: fixed neighbors).

<table><tr><td></td><td colspan="6">Components</td><td colspan="3">AbdomenCT (13 ROIs)</td></tr><tr><td>Exp</td><td>CG</td><td>CS</td><td>DA</td><td>MH</td><td>Agg</td><td>Offset</td><td>DSC ↑</td><td>ASSD ↓</td><td>NJD(%) ↓</td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.549±0.054</td><td>2.83±0.67</td><td>0.105±0.08</td></tr><tr><td>2</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td>0.570±0.051</td><td>2.731±0.658</td><td>0.103±0.066</td></tr><tr><td>3</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td>0.577±0.046</td><td>2.626±0.646</td><td>0.151±0.105</td></tr><tr><td>4</td><td></td><td></td><td>√</td><td></td><td>Attn</td><td>Learned</td><td>0.564±0.045</td><td>2.649±0.598</td><td>0.098±0.064</td></tr><tr><td>5</td><td></td><td></td><td>√</td><td>√</td><td>Attn</td><td>Learned</td><td>0.579±0.049</td><td>2.585±0.651</td><td>0.084±0.048</td></tr><tr><td>6</td><td></td><td></td><td>√</td><td>√</td><td>Avg</td><td>Learned</td><td>0.557±0.048</td><td>2.686±0.590</td><td>0.048±0.029</td></tr><tr><td>7</td><td>√</td><td>√</td><td>√</td><td>√</td><td>Avg</td><td>Learned</td><td>0.583±0.052</td><td>2.543±0.650</td><td>0.089±0.066</td></tr><tr><td>8</td><td>√</td><td>√</td><td>√</td><td>√</td><td>Attn</td><td>Default</td><td>0.559±0.053</td><td>2.657±0.612</td><td>0.014±0.092</td></tr><tr><td>ACorD</td><td>√</td><td>√</td><td>√</td><td>√</td><td>Attn</td><td>Learned</td><td>0.597±0.052</td><td>2.411±0.603</td><td>0.136±0.092</td></tr></table>

Table 4. Aggregated position label accuracy (%) using uniform and attention-based aggregation over ACorD candidates.
<table><tr><td rowspan="2">Lvl.</td><td colspan="2">MindBoggle</td><td colspan="2">AbdomenCT</td></tr><tr><td>Unif.</td><td>Attn.</td><td>Unif.</td><td>Attn.</td></tr><tr><td></td><td>73.89</td><td>74.40</td><td>78.35</td><td>78.56</td></tr><tr><td></td><td>65.95</td><td>68.07</td><td>74.78</td><td>75.84</td></tr><tr><td></td><td>55.55</td><td>58.03</td><td>68.79</td><td>70.79</td></tr><tr><td></td><td>49.56</td><td>49.64</td><td>59.14</td><td>62.07</td></tr><tr><td>12345</td><td>49.19</td><td>48.91</td><td>48.29</td><td>47.31</td></tr></table>

## 5.3. Ablation Study

## 5.3.1. Components Analysis

Table 3 presents an ablation study that systematically evaluates the contribution of each component in the proposed framework. Correlation guidance (CG) improves DSC and ASSD by providing explicit correspondence cues, while cross-scale feature propagation (CS) adds global deformation context for more informed refinement (Exp. 2 and 3).

Table 5. Common-set recovery on AbdomenCT. For voxels misregistered by both models at the coarser scale, we report the percentage corrected one level finer. Higher is better; p-values use paired Wilcoxon signed-rank tests.
<table><tr><td></td><td>Transition | ACorD(%)</td><td>Exp.7(%)</td><td>|ACorD(%)</td><td>Exp.8(%)</td></tr><tr><td>L4→L3</td><td>40.4</td><td> $3 8 . 3 ( p = 0 . 0 0 4 )$ </td><td>40.2</td><td> $3 7 . 0 \ : ( p < 1 \mathrm { e } - 4 )$ </td></tr><tr><td>L3→L2</td><td>23.6</td><td> $2 2 . 6 \overset { \cdot } { ( } p = 0 . 0 3 2 \overset { \cdot } { ) }$ </td><td>23.8</td><td> $2 0 . 6 \ : \dot { ( p \times 1 0 - 4 ) }$ </td></tr><tr><td>L2→L1</td><td>13.2</td><td> $1 2 . 2 \check { ( p } = 0 . 0 0 1 \check { ) }$ </td><td>12.6</td><td> $1 0 . 4 \ : \ddot { ( p \textmd { p } < 1 \mathrm { e } - 4 ) }$ </td></tr></table>

The deformable aggregation module (DA) improves topology preservation, and its multi-head extension (MH) further enhances performance by capturing diverse correspondence hypotheses (Exp. 4 and 5).

To assess the role of learned candidate weighting, we replace attention-weighted aggregation with uniform averaging while keeping all other components unchanged. As shown in Exp. $^ { 6 , }$ this substitution degrades performance, indicating that selectively weighting correspondence candidates is critical for effective deformation refinement. This is further supported by comparing Exp. 7 with the full ACorD model, further confirming that the performance gain arises from learned candidate weighting rather than generating multiple candidates.

We additionally evaluate adaptive correspondence discovery by replacing learned offsets with fixed local neighbors while retaining attention-based aggregation (Exp. 8). The performance drop confirms that attention alone is insufficient, and adaptive, context-conditioned offset prediction is essential for discovering informative correspondences across varying deformation scales.

Finally, the complete ACorD framework, which integrates correlation guidance, cross-scale context, adaptive correspondence discovery, and learned candidate weighting, achieves the best overall performance across all metrics. These results validate the effectiveness of selectively aggregating adaptively discovered correspondences for robust multi-scale medical image registration.

Beyond the end-task ablation, we verify within a single trained model that replacing uniform averaging with attention weighting over an identical candidate set increases the anatomical correctness of the aggregated position at 4 of 5 scales on both AbdomenCT and MindBoggle. (Table 4).

## 5.3.2. Cross-Scale Error Recovery

We next evaluate registration at each pyramid level using its predicted deformation field. On AbdomenCT, ACorD achieves DSC values of 0.304, 0.408, 0.500, 0.563, and 0.597 from $L _ { 5 }$ to $L _ { 1 }$ , exceeding both Learned+Avg and Default+Attn at each level. To distinguish improved recovery from a stronger predictor, we define a common-error set for each coarse-to-fine transition: voxels that are misregistered by both compared models at the coarser level. We then measure the fraction of these shared errors corrected at the next finer level. ACorD recovers a larger fraction at every evaluated transition (Table 5). We omit $L _ { 5 } \to L _ { 4 }$ because too few common errors remain at $1 / 1 6$ resolution for meaningful significance testing. These results provide direct evidence that ACorD more effectively corrects shared coarse-scale failures during refinement.

Table 6. Statistics of predicted candidate offsets. Ratio: the ratio of predicted offsets to default offset magnitude. Frac. > 1: fraction of predicted candidate offsets with ratios greater than 1.
<table><tr><td rowspan="3"> $_ \mathrm { L v l }$ </td><td colspan="2">AbdomenCT</td><td colspan="2">LPBA</td><td colspan="2">OASIS</td><td colspan="2">MindBoggle</td></tr><tr><td>Ratio</td><td>Frac. &gt; 1 |</td><td>Ratio</td><td>Frac. &gt; 1 |</td><td>Ratio</td><td>Frac. &gt; 1 |</td><td>Ratio</td><td> $\mathbf { F r a c . > 1 }$ </td></tr><tr><td></td><td>1.46±0.02</td><td>0.73±0.01</td><td>1.45±0.07</td><td>0.75±0.03</td><td>1.07±0.01</td><td>0.55±0.00</td><td>1.05±0.02</td><td>0.51±0.05</td></tr><tr><td></td><td>1.14±0.02</td><td>0.55±0.01</td><td>1.18±0.03</td><td>0.68±0.02</td><td>0.68±0.00</td><td>0.11±0.00</td><td>1.09±0.02</td><td>0.58±0.01</td></tr><tr><td></td><td>0.93±0.02</td><td>0.35±0.02</td><td>0.95±0.01</td><td>0.32±0.01</td><td>0.79±0.01</td><td>0.20±0.00</td><td>0.97±0.01</td><td>0.36±0.01</td></tr><tr><td>12345</td><td>1.15±0.03</td><td>0.50±0.02</td><td>0.65±0.01</td><td>0.09±0.01</td><td>0.75±0.00</td><td>0.20±0.00</td><td>0.65±0.03</td><td>0.13±0.03</td></tr><tr><td></td><td>1.54±0.04</td><td>0.73±0.01</td><td>1.10±0.01</td><td>0.49±0.01</td><td>0.97±0.01</td><td>0.29±0.01</td><td>0.91±0.00</td><td>0.19±0.00</td></tr></table>

## 5.3.3. Statistical Analysis on Predicted Offsets

Table 6 compares ACorD’s predicted candidate offsets with the default local offsets across datasets and network levels. AbdomenCT shows the largest and most consistent offset expansion, with predicted offsets exceeding the default magnitude across most levels. This supports the role of adaptive offset prediction for large-deformation registration, where fixed local neighborhoods may be insufficient. In contrast, the brain datasets generally show smaller ratios, suggesting that ACorD adjusts its search range according to deformation complexity.

## 5.4. Limitations

A key direction for future work is improving structure and boundary awareness during training. While ACorD provides effective correspondence discovery, mismatched regions can still occur (see supplementary material). These errors often arise from structure-unaware loss functions that prioritize image similarity, leading to unnecessary or excessive deformation that ignores structural integrity. For example, brain registration results can sometimes show extra deformation that causes structural folding to achieve higher image similarity. Similarly, in AbdomenCT registration, even when images appear visually similar, mask visualizations may reveal structure-level artifacts. Although methods like SACB-Net[11] have attempted to address this with clustering or graph, they often prove ineffective for complex structures like the brain. Therefore, a significant challenge remains: developing a method for structure-aware training that works effectively with non-local deformations. A promising direction is to explore incorporating learned anatomical priors into the attention mechanism, allowing the model to weigh correspondences not just by feature similarity but also by their structural context.

## 6. Conclusion

We presented Adaptive Correspondence Discovery (ACorD), a novel framework for multi-scale medical image registration that enables attention-guided propagation and adaptive correspondence discovery beyond fixed local windows. By predicting correspondence candidates and applying sparse attention adaptively, ACorD achieves accurate alignment across a wide range of deformations. Extensive experiments on four benchmark datasets demonstrate state-of-the-art performance, particularly in handling large non-linear deformations. In future work, we plan to incorporate structure-aware supervision to enhance boundary-level accuracy, extend ACorD to multi-modal registration tasks, and improve scalability.

## References

[1] Vaswani Ashish. Attention is all you need. Advances in neural information processing systems, 30:I, 2017. 5

[2] Brian B Avants, Charles L Epstein, Murray Grossman, and James C Gee. Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Medical image analysis, 12(1):26–41, 2008. 1, 6

[3] Ruzena Bajcsy and Stane Kovaciˇ c. Multiresolution elasticˇ matching. Computer vision, graphics, and image processing, 46(1):1–21, 1989. 1

[4] Guha Balakrishnan, Amy Zhao, Mert R Sabuncu, John Guttag, and Adrian V Dalca. Voxelmorph: a learning framework for deformable medical image registration. IEEE transactions on medical imaging, 38(8):1788–1800, 2019. 1, 2, 6

[5] M Faisal Beg, Michael I Miller, Alain Trouve, and Laurent ´ Younes. Computing large deformation metric mappings via geodesic flows of diffeomorphisms. International journal of computer vision, 61(2):139–157, 2005. 1

[6] Junyu Chen, Yufan He, Eric C Frey, Ye Li, and Yong Du. Vitv-net: Vision transformer for unsupervised volumetric medical image registration. arXiv preprint arXiv:2104.06468, 2021. 1, 6

[7] Junyu Chen, Eric C Frey, Yufan He, William P Segars, Ye Li, and Yong Du. Transmorph: Transformer for unsupervised medical image registration. Medical image analysis, 82:102615, 2022. 1, 2, 6

[8] Junyu Chen, Yihao Liu, Yufan He, and Yong Du. Deformable cross-attention transformer for medical image registration. In International Workshop on Machine Learning in Medical Imaging, pages 115–125. Springer, 2023. 2, 6

[9] Junyu Chen, Yihao Liu, Shuwen Wei, Zhangxing Bian, Shalini Subramanian, Aaron Carass, Jerry L Prince, and Yong Du. A survey on deep learning in medical image registration: New technologies, uncertainty, evaluation metrics, and beyond. Medical Image Analysis, 100:103385, 2025. 1

[10] Zeyuan Chen, Yuanjie Zheng, and James C Gee. Transmatch: a transformer-based multilevel dual-stream feature matching network for unsupervised deformable image registration. IEEE transactions on medical imaging, 43(1):15–27, 2023. 1, 6

[11] Xinxing Cheng, Tianyang Zhang, Wenqi Lu, Qingjie Meng, Alejandro F Frangi, and Jinming Duan. Sacb-net: spatialawareness convolutions for medical image registration. In

Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5227–5237, 2025. 1, 2, 6, 8

[12] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Proceedings of the IEEE international confer ence on computer vision, pages 764–773, 2017. 2

[13] Adrian V Dalca, Guha Balakrishnan, John Guttag, and Mert R Sabuncu. Unsupervised learning of probabilistic diffeomorphic registration for images and surfaces. Medical image analysis, 57:226–236, 2019. 1, 2, 6

[14] Enzo Ferrante and Nikos Paragios. Slice-to-volume medical image registration: A survey. Medical image analysis, 39: 101–123, 2017. 1

[15] Yabo Fu, Yang Lei, Tonghe Wang, Walter J Curran, Tian Liu, and Xiaofeng Yang. Deep learning in medical image registration: a review. Physics in Medicine & Biology, 65 (20):20TR01, 2020. 1

[16] Morteza Ghahremani, Mohammad Khateri, Bailiang Jian, Benedikt Wiestler, Ehsan Adeli, and Christian Wachinger. H-vit: A hierarchical vision transformer for deformable image registration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11513–11523, 2024. 2, 6

[17] Tobias Heimann, Bram Van Ginneken, Martin A Styner, Yulia Arzhaeva, Volker Aurich, Christian Bauer, Andreas Beck, Christoph Becker, Reinhard Beichel, Gyorgy Bekes, et al.¨ Comparison and evaluation of methods for liver segmentation from ct datasets. IEEE transactions on medical imaging, 28(8):1251–1265, 2009. 6

[18] Andrew Hoopes, Malte Hoffmann, Bruce Fischl, John Gut tag, and Adrian V Dalca. Hypermorph: Amortized hyperparameter learning for image registration. In International conference on information processing in medical imaging, pages 3–17. Springer, 2021. 1

[19] Miao Kang, Xiaojun Hu, Weilin Huang, Matthew R Scott, and Mauricio Reyes. Dual-stream pyramid registration network. Medical image analysis, 78:102379, 2022. 1, 2, 3, 6

[20] Boah Kim, Dong Hwan Kim, Seong Ho Park, Jieun Kim, June-Goo Lee, and Jong Chul Ye. Cyclemorph: cycle consistent unsupervised deformable image registration. Medical image analysis, 71:102036, 2021. 2

[21] Arno Klein, Satrajit S Ghosh, Forrest S Bao, Joachim Giard, Yrjo H¨ ame, Eliezer Stavsky, Noah Lee, Brian Rossa, Martin¨ Reuter, Elias Chaibub Neto, et al. Mindboggling morphometry of human brains. PLoS computational biology, 13(2): e1005350, 2017. 5

[22] Yihao Liu, Lianrui Zuo, Shuo Han, Yuan Xue, Jerry L Prince, and Aaron Carass. Coordinate translator for learning deformable medical image registration. In International workshop on multiscale multimodal medical imaging, pages 98–109. Springer, 2022. 1, 2, 6

[23] Tai Ma, Xinru Dai, Suwei Zhang, and Ying Wen. Pivit: Large deformation image registration with pyramid-iterative vision transformer. In International Conference on Medi cal Image Computing and Computer-Assisted Intervention, pages 602–612. Springer, 2023. 1, 2, 6

[24] Tai Ma, Suwei Zhang, Jiafeng Li, and Ying Wen. Iirp-net: Iterative inference residual pyramid network for enhanced image registration. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11546–11555, 2024. 2, 6

[25] JB Antoine Maintz and Max A Viergever. A survey of medical image registration. Medical image analysis, 2(1):1–36, 1998. 1

[26] Daniel S Marcus, Anthony F Fotenos, John G Csernansky, John C Morris, and Randy L Buckner. Open access series of imaging studies: longitudinal mri data in nondemented and demented older adults. Journal ofcognitive neuroscience, 22 (12):2677–2684, 2010. 5

[27] Mingyuan Meng, Lei Bi, Dagan Feng, and Jinman Kim. Non-iterative coarse-to-fine registration based on single-pass deep cumulative learning. In International Conference on Medical Image Computing and Computer-Assisted Interven tion, pages 88–97. Springer, 2022. 1, 2, 6

[28] Mingyuan Meng, Lei Bi, Michael Fulham, Dagan Feng, and Jinman Kim. Non-iterative coarse-to-fine transformer networks for joint affine and deformable image registration. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 750–760. Springer, 2023. 1, 2, 6

[29] Mingyuan Meng, Dagan Feng, Lei Bi, and Jinman Kim. Correlation-aware coarse-to-fine mlps for deformable medical image registration. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9645–9654, 2024. 1, 2, 3, 6

[30] Mingyuan Meng, Michael Fulham, Lei Bi, and Jinman Kim. Advancing deformable medical image registration with multi-axis cross-covariance attention. arXiv preprint arXiv:2412.18545, 2024. 1

[31] Marc Modat, Gerard R Ridgway, Zeike A Taylor, Manja Lehmann, Josephine Barnes, David J Hawkes, Nick C Fox, and Sebastien Ourselin. Fast free-form deformation using´ graphics processing units. Computer methods and programs in biomedicine, 98(3):278–284, 2010. 1

[32] Tony CW Mok and Albert CS Chung. Large deformation diffeomorphic image registration with laplacian pyramid networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 211–221. Springer, 2020. 1, 2

[33] Daniel Rueckert, Luke I Sonoda, Carmel Hayes, Derek LG Hill, Martin O Leach, and David J Hawkes. Nonrigid registration using free-form deformations: application to breast mr images. IEEE transactions on medical imaging, 18(8): 712–721, 2002. 1

[34] David W Shattuck, Mubeena Mirza, Vitria Adisetiyo, Cornelius Hojatkashani, Georges Salamon, Katherine L Narr, Russell A Poldrack, Robert M Bilder, and Arthur W Toga. Construction of a 3d probabilistic atlas of human cortical structures. Neuroimage, 39(3):1064–1080, 2008. 5

[35] Dinggang Shen and Christos Davatzikos. Hammer: hierarchical attribute matching mechanism for elastic registration. IEEE transactions on medical imaging, 21(11):1421–1439, 2002. 1

[36] Jiacheng Shi, Yuting He, Youyong Kong, Jean-Louis Coatrieux, Huazhong Shu, Guanyu Yang, and Shuo Li. Xmorpher: Full transformer for deformable medical image regis tration via cross attention. In International Conference on Medical Image Computing and Computer-Assisted Interven tion, pages 217–226. Springer, 2022. 1, 2

[37] Aristeidis Sotiras, Christos Davatzikos, and Nikos Paragios. Deformable medical image registration: A survey. IEEE transactions on medical imaging, 32(7):1153–1190, 2013. 1

[38] Abdel Aziz Taha and Allan Hanbury. Metrics for evaluating 3d medical image segmentation: analysis, selection, and tool. BMC medical imaging, 15(1):29, 2015. 6

[39] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In European conference on computer vision, pages 402–419. Springer, 2020. 2

[40] Haiqiao Wang, Dong Ni, and Yi Wang. Modet: Learning deformable image registration via motion decomposition transformer. In International conference on medical image computing and computer-assisted intervention, pages 740–749. Springer, 2023. 1, 2, 6

[41] Haiqiao Wang, Dong Ni, and Yi Wang. Recursive deformable pyramid network for unsupervised medical image registration. IEEE Transactions on Medical Imaging, 43(6): 2229–2240, 2024. 1, 2, 6

[42] Zhuoyuan Wang, Haiqiao Wang, and Yi Wang. Pyra mid attention network for medical image registration. In 2024 IEEE International Symposium on Biomedical Imag ing (ISBI), pages 1–5. IEEE, 2024. 1

[43] Zhuofan Xia, Xuran Pan, Shiji Song, Li Erran Li, and Gao Huang. Vision transformer with deformable attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4794–4803, 2022. 2

[44] Zhoubing Xu, Christopher P Lee, Mattias P Heinrich, Marc Modat, Daniel Rueckert, Sebastien Ourselin, Richard G Abramson, and Bennett A Landman. Evaluation of six registration methods for the human abdomen on clinically acquired ct. IEEE Transactions on Biomedical Engineering, 63(8):1563–1572, 2016. 5

[45] Wenzhe Yin, Jan-Jakob Sonke, and Efstratios Gavves. Pcreg: A pyramidal prediction–correction approach for large deformation image registration. Medical image analysis, 90: 102978, 2023. 2

[46] Shenglong Zhou, Bo Hu, Zhiwei Xiong, and Feng Wu. Selfdistilled hierarchical network for unsupervised deformable image registration. IEEE Transactions on Medical Imaging, 42(8):2162–2175, 2023. 1, 2

[47] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020. 2

[48] Yongpei Zhu and Shi Lu. Swin-voxelmorph: A symmetric unsupervised learning model for deformable medical image registration using swin transformer. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 78–87. Springer, 2022. 2

# Supplementary Material

# ACorD: Adaptive Correspondence Discovery for Medical Image Registration

## A. Ablation Study on Deformable Attention Size $K ^ { 3 }$

Based on Table S1, we investigated the impact of the attention window size on our model’s performance. We specifically evaluated small window sizes of $K = 2$ and $K = 3 .$ which both yielded strong results compared with baseline with no attention involved in Table. 3 in the orginal paper. While larger values of K could potentially be explored, we refrained from doing so due to the cubic increase in computational cost and memory requirements. We concluded that our chosen configuration strikes a balance between performance and computational efficiency.

## B. Network Structure for the Encoder layers and Offset Predictor.

![](images/b19c4dfcaa5d00f2f3aea238a4504ec1a684a89d957ca123d9fddce94a608a78.jpg)  
Figure S1. Network Structure for the Encoder layers and Offset Predictor.

## C. Details on Loss function

C.1. NCC

$$
\begin{array} { l } { { \displaystyle N C C \left( I _ { f } , I _ { w } \right) = } } \\ { { \displaystyle \sum _ { p \in \Omega } \frac { \left( \sum _ { p _ { i } } \left( I _ { w } \left( p _ { i } \right) - \bar { I } _ { w } ( p ) \right) \left( I _ { f } \left( p _ { i } \right) - \bar { I } _ { f } ( p ) \right) \right) ^ { 2 } } { \sum _ { p _ { i } } \left( I _ { w } \left( p _ { i } \right) - \bar { I } _ { w } ( p ) \right) ^ { 2 } \sum _ { p _ { i } } \left( I _ { f } \left( p _ { i } \right) - \bar { I } _ { f } ( p ) \right) ^ { 2 } } } } \end{array}\tag{S1}
$$

## C.2. Jacobian Matrix

$$
J _ { \phi } ( \pmb { p } ) = \left[ \begin{array} { c c c } { \frac { \partial \phi _ { x } ( \pmb { p } ) } { \partial x } } & { \frac { \partial \phi _ { x } ( \pmb { p } ) } { \partial y } } & { \frac { \partial \phi _ { x } ( \pmb { p } ) } { \partial z } } \\ { \frac { \partial \phi _ { y } ( \pmb { p } ) } { \partial x } } & { \frac { \partial \phi _ { y } ( \pmb { p } ) } { \partial y } } & { \frac { \partial \phi _ { y } ( \pmb { p } ) } { \partial z } } \\ { \frac { \partial \phi _ { z } ( \pmb { p } ) } { \partial x } } & { \frac { \partial \phi _ { z } ( \pmb { p } ) } { \partial y } } & { \frac { \partial \phi _ { z } ( \pmb { p } ) } { \partial z } } \end{array} \right]\tag{S2}
$$

## D. Deformation Accuracy with $\lambda _ { j a c }$ Tuning

Table S2 presents a parameter tuning analysis for the Jacobian determinant loss, $L _ { j a c } ,$ a regularization term often overlooked in previous works. Increasing $\lambda _ { j a c }$ consistently reduces NJD(%), indicating improved topology preservation after deformation. While DSC remains relatively stable on the MindBoggle brain dataset, moderate Jacobian regularization improves AbdomenCT performance, suggesting that topology preservation is particularly important for large-deformation registration.

## E. Head Allocation Robustness

Alternative head allocations on AbdomenCT result in similar performance as presented in Table S3: [2, 2, 4, 4, 1] achieves 0.595 DSC and [4, 4, 2, 2, 1] achieves 0.594 DSC, close to our default [8, 4, 2, 1, 1] result of 0.597 DSC, indicating that ACorD is not highly sensitive to the exact head allocation, while the default provides the best accuracymemory trade-off.

## F. Dice Scores On LPBA dataset

Fig. S2 presents the dice score evaluation results on different anatomical classes of LPBA dataset.

## G. Limitations of ACorD

Are presented in Fig. S3.

## H. Computational Cost Analysis

Is presented in Table S4.

## I. Per-scale Dice on AbdomenCT

Is presented in Table S5.

Table S1. Deformable Attention Size
<table><tr><td rowspan="2"> $K ^ { 3 }$ </td><td colspan="3">AbdomenCT</td><td colspan="3">MindBoggle</td></tr><tr><td>DSC↑</td><td>ASSD↓</td><td>NJD(%)↓</td><td>DSC↑</td><td>ASSD ↓</td><td>NJD(%)↓</td></tr><tr><td> $2 ^ { 3 } = 8$ </td><td> $0 . 5 9 5 { \scriptstyle \pm 0 . 0 5 6 }$ </td><td>2.475±0.614</td><td>0.135±0.096</td><td>0.621±0.054</td><td>1.512±0.517</td><td>0.005±0.006</td></tr><tr><td> $3 ^ { 3 } = 2 7$ </td><td> $0 . 5 9 7 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $2 . 4 1 1 { \pm } 0 . 6 0 3$ </td><td>0.136±0.092</td><td>0.624±0.049</td><td>1.468±0.447</td><td>0.005±0.004</td></tr></table>

![](images/7daf7066692ed336255edb9d84179b7d04643e2c79ac42faca2476ea2a0534ad.jpg)  
Figure S2. Dice score evaluation on LPBA brain image registration (54 ROIs).

Table S2. Hyperparameter Tuning for Negative Jacobian Loss
<table><tr><td rowspan="3">λjac</td><td colspan="4">ACorD</td></tr><tr><td colspan="2">AbdomenCT</td><td colspan="2">MindBoggle</td></tr><tr><td>1 DSC↑</td><td>NJD(%)↓</td><td>DSC↑</td><td>NJD(%)↓</td></tr><tr><td>0</td><td>0.590</td><td>8.381</td><td>0.625</td><td>0.191</td></tr><tr><td>20</td><td>0.592</td><td>0.317</td><td>0.623</td><td>0.011</td></tr><tr><td>50</td><td>0.597</td><td>0.136</td><td>0.624</td><td>0.005</td></tr><tr><td>80</td><td>0.595</td><td>0.086</td><td>0.623</td><td>0.003</td></tr></table>

Table S3. Ablation Study on Head Allocation
<table><tr><td>Head Allocation</td><td>DSC↑</td><td>ASSD↓</td><td>NJD(%)↓</td></tr><tr><td>[2, 2, 4, 4, 1]</td><td>0.595 ± 0.057</td><td> $2 . 4 2 8 \pm 0 . 6 0 1$ </td><td>0.18 ± 0.14</td></tr><tr><td>[4, 4, 2, 2, 1]</td><td>0.594 ± 0.052</td><td> $2 . 4 1 8 \pm 0 . 6 0 3$ </td><td>0.094 ± 0.05</td></tr><tr><td>default [8, 4, 2, 1, 1]</td><td>0.597 ± 0.052</td><td> $2 . 4 1 1 \pm 0 . 6 0 3$ </td><td>0.136 ± 0.092</td></tr></table>

Table S4. Computational cost comparison on A6000.
<table><tr><td>Model</td><td>MACs (GMac)</td><td>Runtime (ms)</td><td>Memory (MB)</td></tr><tr><td>TransMorph</td><td>714.16</td><td>161.15</td><td>5353.16</td></tr><tr><td>TransMatch</td><td>717</td><td>257</td><td>4768</td></tr><tr><td>TransDCA</td><td>4766</td><td>593</td><td>9431</td></tr><tr><td>NICE-Trans</td><td>313</td><td>415</td><td>6205</td></tr><tr><td>RDPNet</td><td>3109</td><td>428</td><td>9306</td></tr><tr><td>Dual-PR++</td><td>1773</td><td>491</td><td>9461</td></tr><tr><td>Im2Grid</td><td>73</td><td>367</td><td>10906</td></tr><tr><td>ModeT</td><td>83</td><td>870</td><td>11166</td></tr><tr><td>CorrMLP</td><td>1117</td><td>1237</td><td>7004</td></tr><tr><td>SACB-Net</td><td>181</td><td>1119</td><td>2836</td></tr><tr><td>ACorD</td><td>1781</td><td>710</td><td>16946</td></tr></table>

Table S5. Per-scale DSC on AbdomenCT
<table><tr><td>Model</td><td>L5</td><td>L4</td><td>L3</td><td>L2</td><td>L1</td></tr><tr><td>Exp.8 (Default+Attn)</td><td>0.298</td><td>0.403</td><td>0.476</td><td>0.534</td><td>0.562</td></tr><tr><td>Exp.7 (Learned+Avg)</td><td>0.289</td><td>0.403</td><td>0.472</td><td>0.554</td><td>0.587</td></tr><tr><td>ACorD</td><td>0.304</td><td>0.408</td><td>0.500</td><td>0.563</td><td>0.597</td></tr></table>

![](images/e803e279b907dd88aab747cdfb549120d7a938e7fc6f8af8f8100f480f1b944d.jpg)  
Figure S3. Artifact of ACorD registration. Left: In regions highlighted by the red boxes, the cortical surface is shown to have unnecessary folding. Right: While intensity similarity metrics may indicate good alignment, errors revealed through segmentation comparisons demonstrate the shortcomings of structure-unaware supervision. The mismatches show that the model might achieve high image similarity at the cost of structural integrity, leading to inaccuracies at the structure level.