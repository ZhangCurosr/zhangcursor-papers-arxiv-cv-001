# CLFTv2: Efficient Camera-LiDAR Fusion for Semantic Segmentation via Hierarchical Feature Pyramids

Toomas Tahves, Mauro Bellone, and Raivo Sell

Abstract—Semantic segmentation for autonomous driving requires reliable detection of vulnerable road users (VRUs) despite heavy class imbalance. We introduce CLFTv2, a hierarchical camera-LiDAR fusion framework replacing global ViT attention with a Swin-based multi-scale encoder and a lightweight FPNstyle residual decoder. Operating in the 2D perspective domain, CLFTv2 integrates multi-scale geometric cues through shiftedwindow attention and per-scale residual fusion, avoiding the computational overhead of query-matching decoders. Across three driving datasets, CLFTv2 consistently improves VRU recall. On ZOD, CLFTv2-Large achieves 53.5% mIoU, improving pedestrian IoU from 35.5% to 44.9% over the prior CLFT model. On Waymo, CLFTv2 reaches 61.7% mIoU. Additionally, a modalityisolation study suggests ViT’s global receptive field yields stronger fusion gains only under dense LiDAR returns. Compared to a Swin-based Mask2Former adaptation, CLFTv2 requires 1.4× fewer GFLOPs and delivers 2.2× higher throughput, while achieving comparable overall accuracy. These results demonstrate that hierarchical local-attention fusion offers an efficient, scalable alternative to global-attention and query-based decoders for real-time on-vehicle perception in intelligent transportation systems. Source code is publicly available.

Index Terms—Intelligent transportation systems, autonomous driving, camera-LiDAR fusion, semantic segmentation, vulnerable road users, Swin Transformer, feature pyramid network.

## I. INTRODUCTION

Semantic segmentation for autonomous driving must reliably identify vulnerable road users (VRUs) whose pixel footprint in perspective images is tiny (0.3-0.4% of pixels) yet critical for safety. Cameras provide dense color and texture at high spatial resolution but lack metric depth, whereas LiDAR returns 3D geometry but yields point clouds that grow increasingly sparse with distance [1], [2]. Multi-modal fusion bridges these gaps, but extracting reliable features for distant VRUs requires operating at high spatial resolutions. Processing dense, multi-modal sensor streams under the strict latency constraints of autonomous driving requires lightweight architectures to efficiently process multi-modal sensor streams.

Recent vision transformers have advanced multi-modal fusion. Our prior work, CLFT [3], [4], demonstrated that a Vision Transformer (ViT) encoder provides excellent crossmodal context. However, standard ViTs incur quadratic complexity and maintain a single fixed spatial resolution, making it expensive to generate the multi-scale feature pyramids required for small-object detection [5], [6].

To overcome these limitations, we introduce CLFTv2, a hierarchical camera–LiDAR fusion framework built on the Swin Transformer [5]. Shifted-window self-attention restricts computation to local windows, reducing complexity to linear in image resolution, while patch merging produces a four-stage feature pyramid. In our framework, two weightshared Swin encoders process projected 2D camera and LiDAR inputs in parallel. At each scale, modality-specific residual convolution units refine the features, which are then fused via element-wise summation and accumulated through a coarse-to-fine FPN-style decoder. CLFTv2 operates entirely in the 2D perspective-projection domain: LiDAR points are projected onto the image plane and encoded as a three-channel coordinate map, enabling direct use of ImageNet-pretrained backbones without specialized 3D processing.

The main contributions of this paper are:

• We propose CLFTv2, a hierarchical Swin-based fusion architecture with a lightweight per-scale residual decoder. By extracting multi-scale geometric cues natively, it avoids the heavy computational overhead of universal query-based decoders when applied to closed-set semantic segmentation.

• Through a multi-dataset benchmark (ZOD, Waymo, ISEAuto), we demonstrate CLFTv2 achieves competitive accuracy against state-of-the-art query decoders under sparse or pseudo-labeled supervision, while demanding substantially lower GFLOPs and delivering 2.2× higher throughput.

• A modality-isolation study reveals a practical trade-off: while Swin’s hierarchical structure improves small-object resolution, its local-window attention struggles slightly more than ViT’s global receptive field to fully exploit dense, localized LiDAR returns. This suggests hybrid global-local architectures as a direction for future work.

• We demonstrate that our learned multi-modal representations generalize across differing sensor characteristics and geographical datasets, with cross-dataset initialization accelerating convergence and improving early-epoch accuracy.

## II. RELATED WORK

## A. Camera-LiDAR Fusion for Semantic Segmentation

Multi-modal fusion of camera and LiDAR data has been studied across two broadly distinct paradigms: 3D-native representations and 2D perspective-projection representations.

Early deep fusion work such as PointFusion [7] projected image features onto each raw LiDAR point to augment the 3D point representation for bounding-box estimation. PointPainting [8] extended this by back-projecting 2D semantic scores from a pretrained image segmentation network onto the point cloud before 3D detection. More recently, BEVFusion [9] unifies multi-sensor data in a bird’s-eye-view grid before fusing camera and LiDAR features; UniAD [10] builds on BEV for end-to-end multi-task driving. These 3D-native methods preserve metric depth but require specialised voxelization and sparse-convolution pipelines that preclude direct use of ImageNet-pretrained image backbones.

An alternative paradigm encodes both modalities as spatially aligned 2D image tensors by projecting LiDAR points onto the camera image plane. This allows the CNN and Transformer image backbones to be applied without modification, at the cost of discarding out-of-plane 3D geometry. Valada et al. [1] demonstrated self-supervised adaptation of CNN fusion models under this paradigm; SqueezeSegV3 [2] applied spatiallyadaptive convolutions to projected LiDAR range images for efficient point-cloud segmentation. TransFusion [11] introduced transformer cross-attention between image and LiDAR features in the perspective domain for 3D detection. Our prior work, CLFT [3], [4], applied a ViT encoder to this 2D perspective setting. CLFTv2 operates in the same paradigm, enabling direct comparison, while replacing the ViT backbone with a hierarchical Swin encoder to address the aforementioned single-scale limitations. Direct numeric comparison with BEVbased 3D methods is outside the scope of this paper, as the two paradigms discard different geometric information and serve different downstream application constraints.

## B. Hierarchical Vision Transformers for Dense Prediction

ViT’s quadratic attention complexity and single-scale output limit its use for dense prediction; the Swin Transformer [5] addressed both. Shifted-window attention restricts computation to local M×M windows, reducing complexity to linear in image resolution, while Patch Merging layers create a fourstage feature pyramid analogous to the feature hierarchy of ResNets. SwinV2 [12] subsequently extended this architecture with cosine-similarity attention and log-spaced continuous relative position bias, improving stability at larger model sizes and higher resolutions. Swin-based backbones have been applied to medical image segmentation [13] and monocular depth estimation [14]; more directly, Swin serves as the default backbone for both MaskFormer [15] and Mask2Former [16].

Feature Pyramid Networks (FPN) [17] established the standard pattern for combining multi-scale backbone features in dense prediction: lateral 1×1 projections normalize channel dimensions across stages, and top-down upsampling merges coarse semantic and fine spatial information. CLFTv2’s decoder follows this pattern, adding per-scale residual fusion of camera and LiDAR streams before the coarse-to-fine accumulation. The Dense Prediction Transformer (DPT) [18] adapted ViT for dense prediction; CLFT [3], [4] built on DPT for the camera-LiDAR setting, which CLFTv2 extends by replacing the ViT-DPT stack with the Swin-FPN hierarchy for lower complexity and native multi-scale feature access.

## C. Query-Based Universal Segmentation

MaskFormer [15] reformulated semantic segmentation as a set prediction problem: a fixed bank of learned object queries attend to image features via a standard transformer decoder, each producing a class logit and a binary mask; Hungarian matching assigns queries to ground-truth segments during training. This cast semantic segmentation as a form of instance-agnostic mask classification, removing the need for per-pixel class assignment heads. Mask2Former [16] strengthened this design with masked cross-attention, queries attend only within their predicted foreground region, and a multiscale deformable attention pixel decoder [19], achieving stateof-the-art results across panoptic, instance, and semantic segmentation benchmarks.

These architectures are designed for open-ended scene decomposition where the number and identity of segments must be inferred per image. For the closed-set, fixed-class semantic segmentation evaluated in this work, the querymatching mechanism introduces training overhead (Hungarian solver, deep supervision across all decoder layers) that does not directly correspond to a task requirement. We extend both models to the camera-LiDAR fusion setting using the same per-scale residual fusion method as CLFTv2, and evaluate them as strong baselines.

## III. METHODOLOGY

As illustrated in Fig. 1, CLFTv2 comprises three primary components: a Siamese Swin Transformer encoder that independently extracts hierarchical feature pyramids from aligned RGB and LiDAR inputs; a coarse-to-fine residual fusion decoder that integrates these streams across four spatial scales; and a lightweight segmentation head that upsamples the final fused representation to the original input resolution. Detailed schematics of the decoder’s per-scale fusion mechanism and its core residual convolution units are provided in Fig. 2 and Fig. 3, respectively.

## A. Hierarchical Backbone: Swin Transformer

The input tensor is first partitioned into non-overlapping 4×4 patches, then processed through four hierarchical transformer stages. Within each stage, self-attention is computed locally over M×M non-overlapping windows. For our main benchmarks, the Tiny variant uses a window size of M=16 at 256×256 input resolution, while the Base and Large variants natively support scaling M from 12 to 24 at 384×384 resolution (Section V-C1 explicitly investigates varying this window size). A Shifted-Window (SW-MSA) mechanism in alternating transformer blocks cyclically shifts the window partition by $\left( \lfloor M / 2 \rfloor , \lfloor M / 2 \rfloor \right)$ to allow cross-window information flow. For a feature map of spatial size $h \times w$ and channel dimension $C ,$ the W-MSA complexity is:

![](images/0202b8c341d22e49b214affb6c640ab1b3ee93a5a850f2d3d3e6fb1b79a75c54.jpg)  
Fig. 1: CLFTv2 Architecture. Parallel Swin Transformer backbones extract four-level feature pyramids independently from the aligned RGB and LiDAR inputs. Spatial resolution and channel dimensions are first unified using per-scale $1 \times 1$ projections (yellow) that sit between each backbone stage and the fusion decoder. The coarse-to-fine residual fusion decoder (orange) propagates semantic context bottom-up: it begins at Stage 4 $( H / 3 2 )$ and refines the representation via 2× bilinear upsampling and stage-wise feature summation through Stages 3, 2, and 1. The final fused representation $A _ { 1 } ~ ( H / 4 )$ is processed by the segmentation head and upsampled 4× to yield the full-resolution categorical prediction.

$$
\Omega ( \mathrm { W } \mathrm { \bf - M S A } ) = 4 h w C ^ { 2 } + 2 M ^ { 2 } h w C ,\tag{1}
$$

which is $\mathcal { O } ( h w )$ , compared to $\mathcal { O } ( ( h w ) ^ { 2 } )$ for global selfattention. Between stages, Patch Merging halves the spatial resolution and doubles the channel dimension, producing a four-level pyramid $\{ F _ { i } \} _ { i = 1 } ^ { 4 }$ at strides $S _ { i } \in$ $\{ 4 , 8 , 1 6 , 3 2 \}$ . The embedding dimensions $C _ { i }$ scale with model size: {96, 192, 384, 768} for Tiny, {128, 256, 512, 1024} for Base, and {192, 384, 768, 1536} for Large. In all experiments we use SwinV2 initialized with ImageNet-pretrained weights (ImageNet-1k for Tiny models, ImageNet-22k for Base/Large models). These pretrained models incorporate cosine-similarity attention and log-spaced continuous relative position bias for improved large-resolution fine-tuning [12].

## B. Siamese Feature Encoding

Both the camera image $I _ { R G B } \in \mathbb { R } ^ { H \times W \times 3 }$ and the projected LiDAR tensor $I _ { L i D A R } \in \mathbb { R } ^ { H \times W \times 3 }$ are processed by two Swin

Transformer backbones sharing identical parameters $\theta _ { e n c } \colon$

$$
\begin{array} { c } { { \{ F _ { i } ^ { R G B } \} _ { i = 1 } ^ { 4 } = \mathcal { E } ( I _ { R G B } ; \theta _ { e n c } ) } } \\ { { \{ F _ { i } ^ { L i D A R } \} _ { i = 1 } ^ { 4 } = \mathcal { E } ( I _ { L i D A R } ; \theta _ { e n c } ) } } \end{array}\tag{2}
$$

(3)

where $\mathcal { E }$ denotes the hierarchical encoder mapping a 3-channel input to a four-stage feature pyramid $\{ \mathbb { R } ^ { H _ { i } \times \mathbf { \bar { W } } _ { i } \times \mathbf { { \mathscr { C } } } _ { i } } \} _ { i = 1 } ^ { 4 }$ , with spatial dimensions $H _ { i } = H / S _ { i }$ and $W _ { i } = W / S _ { i }$ governed by descending strides $S _ { i } \in \{ 4 , 8 , 1 6 , 3 2 \}$

Rather than visual texture, the LiDAR input encodes structural geometry as a dense coordinate map where each pixel represents a projected 3D point $( X , Y , Z )$ . Formulating this sparse spatial data as a dense 3-channel tensor ensures structural compatibility with the standard vision backbone. A Siamese architecture with tied weights halves the parameter count and enables the use of ImageNet-pretrained weights across both appearance and geometry streams.

## C. Hierarchical Fusion Decoder

1) Channel Projection: Each backbone level i produces features $F _ { m , i } \in \mathbb { R } ^ { \check { H } _ { i } \times W _ { i } \times C _ { i } }$ with varying embedding dimension $C _ { i } . \mathrm { ~ A ~ }$ learned 1×1 convolution projects each level to a unified dimension $D { = } 2 5 6$ while preserving the native spatial resolution:

$$
\begin{array} { r } { F _ { m , i } ^ { \prime } = \operatorname { C o n v } _ { 1 \times 1 } ( F _ { m , i } ) \in \mathbb { R } ^ { H _ { i } \times W _ { i } \times D } . } \end{array}\tag{4}
$$

This normalises the feature scale across levels and enables consistent fusion operations without any spatial resampling at this stage.

2) Coarse-to-Fine Residual Fusion: The decoder processes levels in order from the coarsest (i=4, stride 32) to the finest (i=1, stride 4), accumulating a fused representation $A _ { i }$ at each scale (see Fig. 2). Let $\mathcal { R } _ { \mathrm { R G B } }$ and $\mathcal { R } _ { \mathrm { L i D A R } }$ denote modality-specific Residual Convolution (ResConv) units that independently refine features, and let ${ \mathcal { R } } _ { \mathrm { o u t } }$ denote a shared output ResConv unit applied after fusion (both detailed in Fig. 3).

At the coarsest level, the accumulation is initialised using only the refined features:

$$
A _ { 4 } = \mathcal { R } _ { \mathrm { o u t } } \big ( \mathcal { R } _ { \mathrm { R G B } } ( F _ { R G B , 4 } ^ { \prime } ) + \mathcal { R } _ { \mathrm { L i D A R } } ( F _ { L i D A R , 4 } ^ { \prime } ) \big ) .\tag{5}
$$

At each subsequent finer level $i \in \{ 3 , 2 , 1 \}$ , the accumulated coarser representation is bilinearly upsampled by 2× to match the current spatial resolution. This provides top-down semantic context, which is added to the refined modality features before passing through the final output convolution:

$$
\begin{array} { r l } & { A _ { i } = \mathcal { R } _ { \mathrm { o u t } } \Big ( \mathcal { R } _ { \mathrm { R G B } } ( F _ { R G B , i } ^ { \prime } ) + \mathcal { R } _ { \mathrm { L i D A R } } ( F _ { L i D A R , i } ^ { \prime } ) } \\ & { \qquad + \operatorname { U p } _ { 2 \times } ( A _ { i + 1 } ) \Big ) . } \end{array}\tag{6}
$$

The output of the decoder is $A _ { 1 } \in \mathbb { R } ^ { ( H / 4 ) \times ( W / 4 ) \times D }$ . This coarse-to-fine accumulation combines global semantic context (propagated from $A _ { 4 } )$ with bottom-up fine-grained detail at each level, analogous to the top-down pathway of FPN [17], while the modality-specific ResConvs align the visual and geometric features prior to linear combination.

![](images/87d0f034b0401b19be2a9eb629ca4686424f95a8d4ff647d162204bb0baf010a.jpg)  
Fig. 2: Detailed architecture of the per-scale fusion block. Per-modality Residual Convolution (ResConv) units refine the 1×1 projected features independently. These refined features are then summed together with the upsampled context accumulated from the preceding, coarser stage. A final shared ResConv unit processes the combined representation before it is passed to the next finer stage.

![](images/30beed75cdf49d4419806d231c57f9cc16582e4880d48baaa48b489f277789a6.jpg)  
Fig. 3: Architecture of the Residual Convolution (ResConv) unit. This module consists of two consecutive 3×3 convolutions, each followed by a ReLU activation, with an identity skip connection that mirrors standard ResNet topologies [20].

3) Ablation Experiments for Design Choices: The recurrence term $\mathrm { U p } _ { 2 \times } ( A _ { i + 1 } )$ in Eq. (6) is the dominant contributor to accuracy. Removing it (i.e., $A _ { i } = \mathcal { R } _ { \mathrm { o u t } } ( \mathcal { R } _ { R G B } ( F _ { R G B , i } ^ { \prime } ) +$ $\mathcal { R } _ { L i D A R } ( F _ { L i D A R , i } ^ { \prime } ) )$ , equivalent to the ResConv Fusion row in Table XI) reduces ZOD mIoU from 35.9% to 23.1%, an absolute decrease of 12.8%. By contrast, replacing elementwise summation with any of five gated or attention-weighted alternatives changes mIoU by at most ±0.7% across all single-run configurations, confirming that additional fusion parameters are not warranted under the supervision conditions tested.

4) Segmentation Head: $A _ { 1 }$ is at one-quarter of the input resolution. The segmentation head refines the features with a 3×3 convolution followed by Batch Normalization (BN) and a ReLU activation, then applies a 1×1 convolution and upsamples the result to the original resolution:

$$
\hat { Y } = \mathrm { U p } _ { 4 \times } ( \mathrm { C o n v } _ { 1 \times 1 } ( \mathrm { R e L U } ( \mathrm { B N } ( \mathrm { C o n v } _ { 3 \times 3 } ( A _ { 1 } ) ) ) ) )\tag{7}
$$

producing logits $\hat { Y } \in \mathbb { R } ^ { H \times W \times C _ { c l s } }$ where $C _ { c l s }$ is the number of semantic classes.

## IV. DATASET PREPARATION

## A. Datasets

Three autonomous driving datasets are used in this work, spanning different geographical regions, sensor configurations, annotation methods, and class distributions. Split sizes and perclass pixel distributions are summarised in Table I.

1) Waymo Open Dataset: The Waymo Open Dataset [21] provides large-scale recordings from urban environments across multiple US cities, covering a wide range of weather conditions, occlusion levels, and traffic densities. We use a 22,000-frame split with front-facing camera and LiDAR captures. Per-pixel segmentation annotations covering four classes (background, vehicle, human, sign) were prepared within our research group in prior work [3], [4]; the sensor projection and annotation from that work are unchanged. Waymo’s annotations are dense and manually verified, providing the most reliable ground truth of the three datasets.

2) ISEAuto Dataset: The ISEAuto dataset [22], [23] was collected in Estonian urban and suburban conditions using a vehicle equipped with calibrated camera and LiDAR sensors. We use a 2,400-frame split containing two foreground semantic classes (human, vehicle). Unlike ZOD, ISEAuto provides native pixel-level segmentation labels verified by human annotators. ISEAuto’s high foreground class frequency and manually verified annotations make it a complementary evaluation setting to ZOD, directly exposing the effect of annotation quality on model performance.

3) Zenseact Open Dataset (ZOD): ZOD [24] provides multi-modal frames recorded across diverse Swedish urban and highway environments, encompassing challenging scenarios such as snow, rain, and low-light conditions. We utilize a curated split of 2,300 frames (1,150 for training, 1,150 for validation). Because ZOD natively supplies only 3D boundingbox annotations, dense four-class segmentation masks were generated via a semi-automated SAM-based [25] projection pipeline. This process relied on manually selected frames and a suite of algorithmic quality filters (size-filtering, priority based fusion, sequence-consistency). While this human-in-theloop curation ensures the masks are not purely unsupervised, they remain pseudo-labels subject to minor local inaccuracies. A specific consequence of this generation pipeline is a higher mask rejection rate for occluded or heavily truncated objects. ZOD IoU scores should be interpreted as an upper-bound estimate constrained by geometric visibility rather than pure sensor-level detection capability.

TABLE I: Dataset Pixel Statistics by Train, Validation and Test Split. Values show percentage of total pixels per class; ISEAuto does not include Sign class annotations (denoted –).
<table><tr><td>Dataset</td><td>Split</td><td>Frames</td><td>Background</td><td>Vehicle</td><td>Sign</td><td>Human</td></tr><tr><td rowspan="3">ZOD</td><td>Train</td><td>1,148</td><td>97.8%</td><td>1.5%</td><td>0.4%</td><td>0.3%</td></tr><tr><td>Val.</td><td>573</td><td>97.7%</td><td>1.6%</td><td>0.4%</td><td>0.2%</td></tr><tr><td>Test</td><td>579</td><td>97.8%</td><td>1.5%</td><td>0.4%</td><td>0.3%</td></tr><tr><td rowspan="3">Waymo</td><td>Train</td><td>13,199</td><td>95.5%</td><td>4.1%</td><td>0.1%</td><td>0.3%</td></tr><tr><td>Val.</td><td>4,400</td><td>95.4%</td><td>4.1%</td><td>0.1%</td><td>0.3%</td></tr><tr><td>Test</td><td>4,400</td><td>95.3%</td><td>4.2%</td><td>0.1%</td><td>0.3%</td></tr><tr><td rowspan="3">ISEAuto</td><td>Train</td><td>1,200</td><td>96.7%</td><td>2.9%</td><td>一</td><td>0.3%</td></tr><tr><td>Val.</td><td>600</td><td>96.5%</td><td>3.1%</td><td>一</td><td>0.3%</td></tr><tr><td>Test</td><td>600</td><td>96.8%</td><td>2.9%</td><td>一</td><td>0.4%</td></tr></table>

## B. Data Representation and Preprocessing

We transform the raw camera and LiDAR data into spatially aligned tensors that can be processed by our hierarchical encoder.

1) Input Preprocessing: Camera images are resized to $H \times W$ and normalized with standard ImageNet statistics $( \pmb { \mu } =$ (0.485, 0.456, 0.406), σ = (0.229, 0.224, 0.225)), consistent with all pretrained backbones used.

Raw LiDAR points are projected onto the 2D image plane using the sensor’s extrinsic and intrinsic calibration parameters. For each pixel, the spatial coordinates $( x , y , z )$ of the closest projected return are stored in a three-channel format, natively substituting the standard RGB channels (where Z provides depth). To store these sparse maps efficiently as 8-bit PNGs, we apply dataset-specific quantization rules. For ZOD and ISEAuto, the coordinate distributions are clipped at the 95th percentile, scaled to [−1, 1], and mapped into the integer range [0, 255], assigning empty pixels to a neutral background value of 127. Conversely, for the Waymo dataset, min-max normalization is applied dynamically per-frame to fill the standard [0, 255] bounds. During training, these integer maps are re-normalized into a continuous floating-point format: $I _ { L i D A R } = ( I _ { P N G } / 2 5 5 - \pmb { \mu } _ { L i D A R } ) / \pmb { \sigma } _ { L i D A R }$ , where datasetspecific statistics $( \mu _ { L i D A R } , \pmb { \sigma } _ { L i D A R } )$ are computed across the active training split (Table II). While these different scaling approaches mean the numerical value of empty pixels fluctuates per-frame, the networks successfully learn to distinguish these background artifacts from true geometric foreground structures regardless of the dataset scheme.

TABLE II: Dataset-specific LiDAR PNG normalization statistics $( \mu _ { L i D A R } , \pmb { \sigma } _ { L i D A R } )$ computed over non-zero pixels in the float [0, 1] domain after 8-bit decoding. Channels correspond to R=X (left-right), G=Y (up-down), B=Z (depth).
<table><tr><td rowspan="2">Dataset</td><td colspan="3">µLiDAR (R, G, B)</td><td colspan="3"> $\pmb { \sigma } _ { L i D A R }$  (R, G, B)</td></tr><tr><td>R</td><td>G</td><td>B</td><td>R</td><td>G</td><td>B</td></tr><tr><td>Waymo</td><td>0.461</td><td>0.288</td><td>0.267</td><td>0.115</td><td>0.126</td><td>0.098</td></tr><tr><td>ZOD</td><td>0.253</td><td>0.493</td><td>0.496</td><td>0.234</td><td>0.031</td><td>0.167</td></tr><tr><td>ISEauto</td><td>0.824</td><td>0.430</td><td>0.510</td><td>0.170</td><td>0.339</td><td>0.348</td></tr></table>

## C. Class Weights

We train with Weighted Cross-Entropy Loss:

$$
\mathcal { L } _ { W C E } = - \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \sum _ { c = 1 } ^ { C } w _ { c } \cdot y _ { n , c } \cdot \log \hat { y } _ { n , c }\tag{8}
$$

where N is the number of pixels, $y _ { n , c } \in \{ 0 , 1 \}$ is the groundtruth indicator, $\hat { y } _ { n , c }$ is the predicted probability, and $w _ { c }$ is the class-specific weight. Weights are derived per dataset from the per-class pixel frequency using an approximate inverse-squareroot schedule, as detailed in Table III. The low background weight suppresses the dominant gradient contribution of nonobject pixels.

TABLE III: Class Weights per Dataset
<table><tr><td>Dataset</td><td>Background</td><td>Vehicle</td><td>Human</td><td>Sign</td></tr><tr><td>Waymo</td><td>0.5</td><td>4.0</td><td>10.0</td><td>10.0</td></tr><tr><td>ZOD</td><td>0.1</td><td>10.0</td><td>20.0</td><td>17.0</td></tr><tr><td>ISEAuto</td><td>0.1</td><td>10.0</td><td>20.0</td><td>=</td></tr></table>

## V. EXPERIMENTS AND RESULTS

## A. Experimental Setup

1) Training Configuration: All models were implemented in PyTorch and trained on NVIDIA A100 GPUs (80GB)

provided by the TalTech High Performance Computing Centre [26] with a total training time of over 800 hours. Training configurations (optimizers, learning rate schedules, total epochs, and input resolutions) were tailored to match the optimal setup for each architecture baseline (Table IV). A common batch size of 8 and a base learning rate of $8 \times 1 0 ^ { - 5 }$ were preserved across all setups. General data augmentation across setups included random horizontal flipping $\left( p { = } 0 . 5 \right)$ rotation $( \pm 2 0 ^ { \circ } , p { = } 0 . 4 )$ , and random resized cropping $\left( p { = } 0 . 3 \right)$

TABLE IV: Training configurations across models and datasets.
<table><tr><td rowspan="2">Architecture</td><td colspan="2">Total Epochs</td><td rowspan="2">Resolution</td></tr><tr><td>Waymo</td><td>ZOD / ISEAuto</td></tr><tr><td>CLFTv2 (Base / Large)</td><td>100</td><td>200</td><td>384×384</td></tr><tr><td>CLFTv2 (Tiny)</td><td>100</td><td>200</td><td>256×256</td></tr><tr><td>CLFTv1 [3]</td><td>200</td><td>300</td><td>384×384</td></tr><tr><td>DeepLabV3+</td><td>100</td><td>200</td><td>256×256</td></tr><tr><td>MaskFormer (Base / Large)</td><td>50</td><td>100</td><td>384×384</td></tr><tr><td>MaskFormer (Tiny)</td><td>50</td><td>100</td><td>256×256</td></tr><tr><td>Mask2Former (Base / Large)</td><td>50</td><td>100</td><td>384×384</td></tr><tr><td>Mask2Former (Tiny)</td><td>50</td><td>100</td><td>256×256</td></tr></table>

Although all models use identical seeds and dataset pipelines, training schedules differ to match architecturespecific convergence. Some models (e.g., MaskFormer variants) plateau earlier, so we use fewer epochs to avoid unnecessary compute. CLFT results reported here come from the unified codebase and supersede prior publications [3], [4].

2) Baseline Implementations: MaskFormer and Mask2Former: To reduce framework-induced variance, both MaskFormer [15] and Mask2Former [16] were re-implemented natively in PyTorch, removing external framework dependencies (e.g., Detectron2) in favor of a simpler, pure PyTorch approach. These implementations follow the original architectures while incorporating the adaptations listed in Table V. MaskFormer reproduces the FPN pixel decoder and the six-layer cross-attention transformer decoder $( d _ { \mathrm { m o d e l } } = 2 5 6$ , 8 heads, Q=100 queries). We apply identical Hungarian matching cost weights $( \lambda _ { \mathrm { c l s } } { = } 1$ $\lambda _ { \mathrm { f o c a l } } { = } 2 0 , \lambda _ { \mathrm { d i c e } } { = } 1 )$ and training loss scaling $( \lambda _ { \mathrm { c l s } } { = } 2 .$ $\lambda _ { \mathrm { m a s k } } { = } 5 , \ \lambda _ { \mathrm { d i c e } } { = } 5 )$ with deep supervision enabled. The final segmentation map is assembled using the reference semantic inference formula, mapping predicted class probabilities $\mathbf { c } _ { q }$ and mask logits $m _ { q } .$

$$
\hat { s } _ { c } = \sum _ { q } \operatorname { s o f t m a x } ( \mathbf { c } _ { q } ) _ { c } \cdot \sigma ( m _ { q } )\tag{9}
$$

where the no-object column is dropped without gating.

Mask2Former extends this paradigm via six transformer encoder layers of multi-scale deformable self-attention, and a nine-layer masked cross-attention decoder with auxiliary sequence losses. To avoid custom CUDA dependencies, the multi-scale deformable attention is implemented natively in PyTorch; this gives mathematical equivalence to the original custom CUDA kernel [19] for 2D bilinear sampling. Both baseline models natively ingest the identical Camera-LiDAR early-fusion pipeline utilized by CLFTv2 (residual convolutions and element-wise addition across all backbone scales prior to the pixel decoder). Consequently, comparisons in this paper target the adapted implementations within a common codebase, not direct leaderboard replication of the original single-RGB formulations.

TABLE V: Deviations of our MaskFormer and Mask2Former re-implementations from the published originals [15], [16].
<table><tr><td>#</td><td>Aspect</td><td>Original</td><td>Our adaptation</td><td>Model</td></tr><tr><td>(i)</td><td>CUDA kernel</td><td>Hand-written C++/CUDA extension [19]</td><td>Pure-PyTorch via F.grid_sample; nu- merically equivalent for 2D bilinear sam- pling.</td><td>M2F only</td></tr><tr><td>(ii)</td><td>Backbone</td><td>Swin-T/S/B [5] pretrained on ImageNet- 1k/22k</td><td>SwinV2 [12] pretrained on ImageNet-22k; cosine-similarity attention and log-spaced relative position bias.</td><td>Both</td></tr><tr><td></td><td>(iii) Input modality</td><td>Single RGB stream</td><td>Camera + LiDAR dual stream; residual conv units with element-wise addition per back- bone scale before the pixel decoder.</td><td>Both</td></tr><tr><td></td><td>(iv) Background in matching</td><td>Class 0 excluded from GT segments in Hun- garian matching</td><td>Class 0 included as explicit GT segment; Both motivated by &gt;95% background pixel fre- quency.</td><td></td></tr><tr><td>(v)</td><td>Backbone LR</td><td>Single LR for all parameters</td><td>Backbone at 0.1× base rate; decoder and MF only head at full base rate.</td><td></td></tr></table>

3) DeepLabV3+ Baseline Implementation: To establish a CNN-based comparative baseline, DeepLabV3+ [27] was incorporated. The implementation reconstructs the standard architecture: a ResNet-101 backbone pretrained on ImageNet-1k, an Atrous Spatial Pyramid Pooling (ASPP) module with four parallel heads at dilation rates {6, 12, 18} alongside global-average-pooling, and a standard decoder. This decoder upsamples the ASPP output to $\textstyle { \frac { 1 } { 4 } }$ resolution, concatenates it with low-level layer-1 features (1×1 projected to 48 channels), and applies two 3×3 convolution blocks followed by a 1×1 classifier. For multimodal evaluation, Camera-LiDAR fusion employs the exact same residual-averaging formulation as CLFTv2: two independent modality-specific ResNet-101 streams are processed, and their parallel ASPP representations are summed prior to the decoder.

4) Evaluation Metrics: We evaluate segmentation performance using the Intersection-over-Union (IoU) metric. Because autonomous driving scenes exhibit extreme class imbalance (often > 95% background), standard mIoU is easily skewed. To prevent the static background from inflating aggregate scores, we adopt a Foreground mIoU $( \mathrm { m I o U } _ { f g } )$ which focuses exclusively on dynamic agents:

$$
\mathrm { m I o U } _ { f g } = { \frac { 1 } { | C _ { f g } | } } \sum _ { c \in C _ { f g } } \mathrm { I o U } _ { c }\tag{10}
$$

where $C _ { f g }$ consists of the vehicle, human, and sign classes.

We complement this with Frequency-Weighted Foreground IoU (FW IoU), weighting each class by its pixel frequency to balance common object validation against improvements on sparse classes [28].

Finally, to reduce single-epoch noise while avoiding overfitting to a single peak checkpoint, we report metrics as the average of the top 10 validation checkpoints from each session rather than a single peak score. This top-10 averaging is applied consistently across all models; we report dispersion (±) where available.

TABLE VI: Fusion Results on ZOD mIoU (%).
<table><tr><td>Method</td><td>mIoU</td><td>mRec</td><td>mPrec</td><td>Veh IoU</td><td>Hum IoU</td><td>Sig IoU</td></tr><tr><td>CLFTv2-Large</td><td>53.54</td><td>81.17</td><td>60.08</td><td>72.57</td><td>44.90</td><td>43.15</td></tr><tr><td>MaskFormer-Large</td><td>52.83</td><td>64.70</td><td>72.48</td><td>72.18</td><td>44.33</td><td>41.98</td></tr><tr><td>Mask2Former-Large</td><td>52.52</td><td>64.19</td><td>72.37</td><td>72.27</td><td>43.26</td><td>42.04</td></tr><tr><td>CLFTv2-Base</td><td>52.45</td><td>80.87</td><td>58.78</td><td>71.77</td><td>43.37</td><td>42.20</td></tr><tr><td>MaskFormer-Base</td><td>52.34</td><td>64.43</td><td>71.93</td><td>71.64</td><td>43.50</td><td>41.88</td></tr><tr><td>Mask2Former-Base</td><td>52.07</td><td>64.14</td><td>71.72</td><td>71.31</td><td>43.28</td><td>41.63</td></tr><tr><td>CLFT-Large</td><td>46.82</td><td>65.89</td><td>59.31</td><td>62.39</td><td>35.52</td><td>33.15</td></tr><tr><td>CLFT-Hybrid</td><td>45.79</td><td>72.68</td><td>53.65</td><td>66.90</td><td>36.03</td><td>34.45</td></tr><tr><td>CLFT-Base</td><td>44.64</td><td>67.30</td><td>54.64</td><td>66.58</td><td>33.66</td><td>33.69</td></tr><tr><td>CLFTv2-Tiny</td><td>43.44</td><td>78.06</td><td>48.32</td><td>65.34</td><td>34.10</td><td>30.89</td></tr><tr><td>Mask2Former-Tiny</td><td>43.28</td><td>54.16</td><td>65.32</td><td>65.18</td><td>33.24</td><td>31.43</td></tr><tr><td>MaskFormer-Tiny</td><td>42.90</td><td>53.44</td><td>65.68</td><td>64.44</td><td>32.93</td><td>31.33</td></tr><tr><td>DeepLabV3+</td><td>36.70</td><td>91.17</td><td>37.83</td><td>60.20</td><td>29.10</td><td>20.80</td></tr></table>

## B. Quantitative Analysis

1) Overall Segmentation Performance: We evaluate CLFTv2 against baselines on ZOD, Waymo, and ISEAuto using architecture-appropriate resolutions and a shared batch-size regime. On ZOD (Table VI), CLFTv2-Large reaches 53.5% mIoU (vs. 46.8% for CLFT-Large), with large gains on Human IoU (35.5% → 44.9%) and Sign IoU (33.2% → 43.2%). Mask2Former-Large reaches 52.5% mIoU, within 1% of CLFTv2-Large, but with roughly double training time. Ranking is dataset-dependent: query-based models lead on ISEAuto (MaskFormer-Large 75.6%, Mask2Former-Large 75.4%, CLFTv2-Large 73.3%), while ZOD favors CLFTv2-Large.

2) The Impact ofGlobal vs. Local Attention in Cross-Modal Fusion: An architectural divergence occurs on the Waymo dataset (Table VII), where the ViT-based CLFT family completely outperforms CLFTv2-Large (61.7% mIoU). Specifically, CLFT-Large (68.3%) and CLFT-Base (66.3%) outpace the Swin-based architecture by 6.6% and 4.6% respectively.

Because both architectures achieve nearly identical mIoU (within 0.2–0.4%) when restricted to single-modality inputs (Table X), this gap isolates the fusion interaction mechanism. ViT’s unrestricted global self-attention computes long-range cross-modal associations across the entire dense Waymo Li-DAR point cloud. In contrast, Swin’s local window bounds these associations. While reducing the window size from 24 to 8 degrades accuracy further (Table IX), even the maximum expanded window fails to fully bridge the fusion-efficiency gap against pure global attention on Waymo’s dense geometric point clouds. This confirms that while Swin provides efficiency, unbounded global attention is superior for dense spatial point-cloud alignment.

TABLE VII: Fusion Results on WAYMO mIoU (%).
<table><tr><td>Method</td><td>mIoU</td><td>mRec</td><td>mPrec</td><td>Veh IoU</td><td>Hum IoU</td><td>Sig IoU</td></tr><tr><td>CLFT-Large</td><td>68.26</td><td>91.59</td><td>72.31</td><td>79.75</td><td>64.88</td><td>54.99</td></tr><tr><td>CLFT-Base</td><td>66.32</td><td>91.93</td><td>69.89</td><td>79.80</td><td>64.88</td><td>54.29</td></tr><tr><td>CLFT-Hybrid</td><td>65.50</td><td>92.25</td><td>68.87</td><td>78.10</td><td>63.30</td><td>55.11</td></tr><tr><td>CLFTv2-Large</td><td>61.69</td><td>95.30</td><td>63.49</td><td>67.76</td><td>62.60</td><td>54.73</td></tr><tr><td>CLFTv2-Base</td><td>61.02</td><td>95.20</td><td>62.80</td><td>67.54</td><td>61.79</td><td>53.73</td></tr><tr><td>CLFTv2-Tiny</td><td>55.61</td><td>93.92</td><td>57.45</td><td>64.56</td><td>55.06</td><td>47.21</td></tr><tr><td>MaskFormer-Large</td><td>50.79</td><td>65.57</td><td>69.06</td><td>56.84</td><td>50.82</td><td>44.71</td></tr><tr><td>MaskFormer-Base</td><td>50.39</td><td>64.77</td><td>69.22</td><td>56.88</td><td>50.39</td><td>43.90</td></tr><tr><td>Mask2Former-Large</td><td>49.30</td><td>64.06</td><td>68.01</td><td>54.14</td><td>49.41</td><td>44.34</td></tr><tr><td>Mask2Former-Base</td><td>49.10</td><td>63.66</td><td>68.07</td><td>53.99</td><td>49.37</td><td>43.93</td></tr><tr><td>DeepLabV3+</td><td>48.15</td><td>91.01</td><td>50.08</td><td>61.88</td><td>46.19</td><td>36.40</td></tr><tr><td>MaskFormer-Tiny</td><td>42.71</td><td>56.70</td><td>62.92</td><td>53.17</td><td>41.92</td><td>33.05</td></tr><tr><td>Mask2Former-Tiny</td><td>41.85</td><td>55.99</td><td>61.73</td><td>51.54</td><td>40.79</td><td>33.21</td></tr></table>

TABLE VIII: Fusion Results on ISEAuto mIoU (%).
<table><tr><td>Method</td><td>mIoU</td><td>mRec</td><td>mPrec</td><td>Veh IoU</td><td>Hum IoU</td></tr><tr><td>MaskFormer-Large</td><td>75.58</td><td>85.23</td><td>86.77</td><td>81.15</td><td>70.01</td></tr><tr><td>Mask2Former-Large</td><td>75.37</td><td>85.28</td><td>86.42</td><td>81.06</td><td>69.67</td></tr><tr><td>MaskFormer-Base</td><td>75.21</td><td>84.66</td><td>86.84</td><td>81.02</td><td>69.39</td></tr><tr><td>Mask2Former-Base</td><td>74.61</td><td>85.19</td><td>85.47</td><td>80.77</td><td>68.46</td></tr><tr><td>CLFTv2-Large</td><td>73.30</td><td>94.29</td><td>76.59</td><td>79.14</td><td>67.45</td></tr><tr><td>CLFTv2-Base</td><td>73.23</td><td>94.03</td><td>76.69</td><td>79.18</td><td>67.27</td></tr><tr><td>Mask2Former-Tiny</td><td>70.13</td><td>80.80</td><td>83.74</td><td>77.48</td><td>62.78</td></tr><tr><td>MaskFormer-Tiny</td><td>69.96</td><td>81.09</td><td>83.23</td><td>76.71</td><td>63.20</td></tr><tr><td>CLFTv2-Tiny</td><td>69.94</td><td>94.24</td><td>72.95</td><td>76.49</td><td>63.39</td></tr><tr><td>CLFT-Hybrid</td><td>69.37</td><td>93.51</td><td>72.73</td><td>75.72</td><td>63.02</td></tr><tr><td>CLFT-Large</td><td>69.85</td><td>93.01</td><td>73.55</td><td>76.17</td><td>63.52</td></tr><tr><td>CLFT-Base</td><td>65.67</td><td>93.78</td><td>68.46</td><td>72.89</td><td>58.46</td></tr><tr><td>DeepLabV3+</td><td>56.97</td><td>95.53</td><td>58.35</td><td>66.60</td><td>47.33</td></tr></table>

## C. Ablation Studies

We ablate three factors: Swin window size, modality contribution, and fusion design.

1) Window Size Ablation: Table IX indicates a contextefficiency trade-off. For CLFTv2-Base, moving from window 24 to 16 reduces mIoU by 1.18% (ISEAuto), 6.73% (ZOD), and 4.01% (Waymo), while reducing FLOPs from 119.0G to 52.7G. The additional reduction from window 16 to 8 is smaller: 1.39% (ISEAuto), 1.03% (ZOD), and 0.51% (Waymo) for Base, and 0.82% (ISEAuto), 1.58% (ZOD), and 0.43% (Waymo) for Tiny. This is not a clean windowonly ablation: window-24 runs use 384×384 inputs, whereas window-16/8 runs use 256×256. Therefore, the observed gap reflects a joint window-size and resolution effect.

Table X shows that fusion consistently improves over RGB-only in all 32 settings $( \Delta _ { \mathrm { F \mathrm { - } R } } ~ > ~ 0 )$ . Waymo remains geometry-dominant, with LiDAR-only exceeding RGB-only in 5/6 transformer settings $( \Delta _ { \mathrm { R - L } } < 0 )$ , yet fusion still adds +4.76% to +11.30% over RGB-only. Averaging the Waymo rows reveals a large architecture gap in fusion gain: CLFT achieves a mean +10.52% over RGB-only, while CLFTv2 reaches +4.85% (about 2.2× smaller). In contrast, CLFTv2 gains on ISEAuto are only +0.34% to +1.21%, indicating limited marginal benefit from LiDAR when annotations are dense and the label space is simpler. Figure 4 illustrates these complementary failure modes qualitatively.

TABLE IX: Window-size ablation: fusion validation mIoU (%).
<table><tr><td>Dataset / Model</td><td>Window</td><td>Val mIoU</td><td>FLOPs (G)</td><td>Img. Size</td></tr><tr><td>ISEAuto / Base</td><td>24</td><td>73.23</td><td>119.0</td><td>384</td></tr><tr><td>ISEAuto / Base</td><td>16</td><td>72.05</td><td>52.7</td><td>256</td></tr><tr><td>ISEAuto / Base</td><td>8</td><td>70.66</td><td>52.2</td><td>256</td></tr><tr><td>ISEAuto / Tiny</td><td>16</td><td>69.94</td><td>30.9</td><td>256</td></tr><tr><td>ISEAuto / Tiny</td><td>8</td><td>69.12</td><td>30.8</td><td>256</td></tr><tr><td>ZOD / Base</td><td>24</td><td>52.45</td><td>119.0</td><td>384</td></tr><tr><td>ZOD / Base</td><td>16</td><td>45.72</td><td>52.7</td><td>256</td></tr><tr><td>ZOD / Base</td><td>8</td><td>44.69</td><td>52.2</td><td>256</td></tr><tr><td>ZOD / Tiny</td><td>16</td><td>43.44</td><td>30.9</td><td>256</td></tr><tr><td>ZOD / Tiny</td><td>8</td><td>41.86</td><td>30.8</td><td>256</td></tr><tr><td>Waymo / Base</td><td>24</td><td>61.02</td><td>119.0</td><td>384</td></tr><tr><td>Waymo / Base</td><td>16</td><td>57.01</td><td>52.7</td><td>256</td></tr><tr><td>Waymo / Base</td><td>8</td><td>56.50</td><td>52.2</td><td>256</td></tr><tr><td>Waymo / Tiny</td><td>16</td><td>55.61</td><td>30.9</td><td>256</td></tr><tr><td>Waymo / Tiny</td><td>8</td><td>55.18</td><td>30.8</td><td>256</td></tr></table>

TABLE X: Modality ablation (mIoU, %). $\Delta _ { \mathrm { F - R } }$ is fusion minus RGB-only; $\Delta _ { \mathrm { R - L } }$ is RGB-only minus LiDAR-only.
<table><tr><td>Dataset</td><td>Method</td><td>Fusion</td><td>RGB-Only</td><td>LiDAR-Only</td><td>∆F-R</td><td>∆R-L</td></tr><tr><td rowspan="13">ZOD</td><td>CLFTv2-Large</td><td>53.54</td><td>51.63</td><td>38.27</td><td>+1.91</td><td>+13.36</td></tr><tr><td>CLFTv2-Base</td><td>52.45</td><td>50.28</td><td>38.16</td><td>+2.17</td><td>+12.12</td></tr><tr><td>CLFTv2-Tiny</td><td>43.44</td><td>40.79</td><td>30.54</td><td>+2.65</td><td>+10.25</td></tr><tr><td>CLFT-Large</td><td>46.82</td><td>44.32</td><td>26.30</td><td>+2.50</td><td>+18.02</td></tr><tr><td>CLFT-Hybrid</td><td>45.79</td><td>42.74</td><td>31.03</td><td>+3.05</td><td>+11.71</td></tr><tr><td>CLFT-Base</td><td>44.64</td><td>41.95</td><td>23.72</td><td>+2.69</td><td>+18.23</td></tr><tr><td>MaskFormer-Large</td><td>52.83</td><td>51.92</td><td>38.11</td><td>+0.91</td><td>+13.81</td></tr><tr><td>MaskFormer-Base</td><td>52.34</td><td>51.39</td><td>37.66</td><td>+0.95</td><td>+13.73</td></tr><tr><td>MaskFormer-Tiny</td><td>42.90</td><td>40.72</td><td>29.84</td><td>+2.18</td><td>+10.88</td></tr><tr><td>Mask2Former-Large</td><td>52.52</td><td>51.28</td><td>38.45</td><td>+1.24</td><td>+12.83</td></tr><tr><td>Mask2Former-Base</td><td>52.07</td><td>51.04</td><td>37.66</td><td>+1.03</td><td>+13.38</td></tr><tr><td>Mask2Former-Tiny</td><td>43.28</td><td>40.59</td><td>31.36</td><td>+2.69</td><td>+9.23</td></tr><tr><td>DeepLabV3+</td><td>36.70</td><td>29.67</td><td>23.73</td><td>+6.03</td><td>+5.94</td></tr><tr><td rowspan="6">Waymo</td><td>CLFT-Large</td><td>68.26</td><td>56.97</td><td>60.38</td><td>+11.29</td><td>-3.41</td></tr><tr><td>CLFT-Base</td><td>66.32</td><td>55.02</td><td>58.26</td><td>+11.30</td><td>-3.24</td></tr><tr><td>CLFT-Hybrid</td><td>65.50</td><td>56.53</td><td>59.09</td><td>+8.97</td><td>-2.56</td></tr><tr><td>CLFTv2-Large</td><td>61.69</td><td>56.78</td><td>57.77</td><td>+4.91</td><td>-0.99</td></tr><tr><td>CLFTv2-Base</td><td>61.02</td><td>56.13</td><td>56.88</td><td>+4.89</td><td>-0.75</td></tr><tr><td>CLFTv2-Tiny</td><td>55.61</td><td>50.85</td><td>50.39</td><td>+4.76</td><td>+0.46</td></tr><tr><td rowspan="10">ISEAuto</td><td>CLFTv2-Large</td><td>73.30</td><td>72.96</td><td>63.92</td><td>+0.34</td><td>+9.04</td></tr><tr><td>CLFTv2-Base</td><td>73.23</td><td>72.45</td><td>63.60</td><td>+0.78</td><td>+8.85</td></tr><tr><td>CLFTv2-Tiny</td><td>69.94</td><td>68.73</td><td>59.11</td><td>+1.21</td><td>+9.62</td></tr><tr><td>CLFT-Large</td><td>69.85</td><td>67.94</td><td>50.15</td><td>+1.91</td><td>+17.79</td></tr><tr><td>CLFT-Hybrid</td><td>69.37</td><td>67.22</td><td>57.52</td><td>+2.15</td><td>+9.70</td></tr><tr><td>CLFT-Base</td><td>65.67</td><td>64.13</td><td>35.51</td><td>+1.54</td><td>+28.62</td></tr><tr><td>MaskFormer-Large</td><td>75.58</td><td>75.01</td><td>69.07</td><td>+0.57</td><td>+5.94</td></tr><tr><td>MaskFormer-Base</td><td>75.21</td><td>74.80</td><td>69.09</td><td>+0.41</td><td>+5.71</td></tr><tr><td>MaskFormer-Tiny</td><td>69.96</td><td>68.68</td><td>64.76</td><td>+1.28</td><td>+3.92</td></tr><tr><td>Mask2Former-Large</td><td>75.37</td><td>74.87</td><td>68.39</td><td>+0.50</td><td>+6.48</td></tr><tr><td>Mask2Former-Base</td><td>74.61</td><td>74.47</td><td>68.90</td><td>+0.14</td><td>+5.36</td></tr><tr><td>Mask2Former-Tiny</td><td>70.13</td><td>68.69</td><td>65.90</td><td>+1.44</td><td>+2.79</td></tr><tr><td>DeepLabV3+</td><td>56.97</td><td>51.92</td><td>47.46</td><td>+5.05</td><td>+4.46</td></tr></table>

TABLE XI: Progressive fusion ablation with CLFTv2-Tiny on ZOD.
<table><tr><td>Fusion Setting</td><td>mIoU (%) Time (ms)</td></tr><tr><td>Add (RGB + LiDAR)</td><td> $2 3 . 2 0 \pm 0 . 2 9$   $1 1 . 1 \pm 0 . 4$ </td></tr><tr><td>Average ((RGB + LiDAR)/2)</td><td> $2 2 . 8 3 \pm 0 . 3 6$   $1 0 . 8 \pm 0 . 4$ </td></tr><tr><td>ResConv on both streams</td><td> $2 3 . 0 9 \pm 0 . 3 6$   $1 3 . 3 \pm 0 . 5$ </td></tr><tr><td>ResConv + residual from  $M _ { i - 1 }$ </td><td> ${ \bf 3 5 . 9 1 \pm 0 . 4 7 }$   $1 3 . 2 \pm 0 . 4$ </td></tr><tr><td>Residual + α-gated variants (standard, LN, spatial, sigmoid)</td><td> $3 5 . 6 3 \pm 0 . 3 4$   $1 3 . 1 \pm 0 . 3$ </td></tr></table>

(a) RGB-Only  
![](images/83d8cccc88017350e2e3af6c8e278b9806c526d685c403d0a886e8603f3c25f9.jpg)

(b) LiDAR-Only  
![](images/f4be7b349257d10b58dacfb54b75e44b9e42fae69d1a25709919c3fe380c803c.jpg)

(c) Residual Fusion  
![](images/691f2c4e3595bf6e1b4573f047f2a66653e74b5d473349cbf700b7c72feef616.jpg)  
Fig. 4: Modality ablation on ZOD, illustrating failure modes of individual sensors. The scene shows a low-contrast white vehicle. (a) RGB-only fails to separate the sky and roof of vehicle. (b) LiDAR-only recovers shape geometry but hallucinate false classes. (c) Residual fusion correctly joins optical boundaries with sparse depth.

(a) Ground Truth  
![](images/dbed56e33ecfc5e2aa8e666b7964718d6220b4027808c8e9f2faca81bcd0887d.jpg)

(b) Simple Average Fusion  
![](images/ff20fa503abf58a013c4dbc3d72a6269c13a5f393d04eb2b0ab8dc8a3579348c.jpg)

(c) Residual Fusion  
![](images/c1e4bb61d0d7ea5ebc4f45a3aab6128a4a0296f04a32ef71d5bc93bd500d4953.jpg)  
Fig. 5: Visual comparison of fusion strategies using CLFTv2-Tiny on ZOD. (a) Ground truth mask. (b) Simple averaging of RGB and LiDAR creates poorly defined boundaries. (c) Residual fusion recovers boundaries and suppresses structural artifacts.

Table XI shows that simple fusion rules (add, average, ResConv-only) cluster around 23% mIoU, whereas adding residual propagation $( M _ { i - 1 } )$ lifts performance to 35.9% over simple averaging (22.8%). This gain is accompanied by a moderate latency increase from 10.8 ms to 13.2 ms (+22.2%), indicating a favourable accuracy-latency trade-off. For the gated group, we evaluated four α-gated variants: standard gated averaging, layer-normalized gated averaging, spatially weighted gated fusion, and sigmoid-constrained gated averaging. In our current setup, gating mechanism did not improve mIoU, suggesting that the residual pathway already captures most of the useful cross-modal interaction under these conditions. Figure 5 provides a visual example of the boundary recovery enabled by residual fusion.

## D. Transfer Learning and Cross-Dataset Convergence

We evaluate cross-dataset transfer learning to determine if representations learned on one dataset generalize to others and accelerate training. For each target dataset (ZOD, Waymo, ISEAuto) we train CLFTv2-Tiny both from ImageNetpretrained weights (baseline) and from weights fine-tuned on each of the two other datasets (transfer). Figure 6 reports validation mIoU every 10 epochs across all conditions. Transferinitialised models consistently reach higher mIoU in early epochs across all three target datasets. Class overlap mediates transfer quality: ISEAuto-to-ZOD transfer is weaker because

ISEAuto lacks the sign class present in ZOD. Initializing from a related dataset checkpoint accelerates convergence, which is especially beneficial for ZOD where learning from scratch is slow.

## E. Qualitative Analysis

Figure 7 compares representative predictions from CLFTv2- Large and CLFT-Hybrid across diverse ZOD and Waymo environments.

## F. Resilience and Efficiency Analysis

1) Safety-Critical Recall and Weather Resilience: Table XII shows a clear recall-precision trade-off. CLFTv2-Large has the highest Waymo pedestrian recall (96.1% Day Fair, 95.7% Night Rain), while CLFT-Large has the highest Waymo precision (72.1% Day Fair, 71.7% Night Rain). Table XIII shows dataset-dependent difficulty: for CLFTv2-Base/Large, the Day Fair→Night Rain mIoU drop is about 2.3–2.5% on Waymo, 8.8–9.4% on ZOD, and 15.9–17.1% on ISEAuto. On ZOD, CLFTv2-Large leads most weather splits, but Mask2Former-Base leads Night Rain (48.9% vs 44.5%).

2) Efficiency and Architectural Trade-offs: Table XIV shows that CLFTv2 offers a strong efficiency-accuracy balance. At Tiny scale, CLFTv2-Tiny runs at 22.0 ms and 187 MB, versus 48.4 ms and 314 MB for Mask2Former-Tiny (2.2× faster, about 40% lower memory), while remaining competitive on ZOD/Waymo in the main benchmark tables. At larger scale, CLFTv2-Large is also lighter than Mask2Former-Large (204.4G vs 265.3G FLOPs) and faster (81.8 ms vs 118.2 ms). DeepLabV3+ remains the fastest model overall, but with a clear accuracy gap relative to the best transformer-based fusion models.

ZOD – Transfer Learning Convergence  
![](images/450fea876c3866fd49c3a85d66d0571523720583614f21656a83f6c1b5d82238.jpg)  
(a) ZOD

Waymo – Transfer Learning Convergence  
![](images/ee5a39f69879135e0fb0f01c486440652ec94cffc02372b0f8545315103d35b7.jpg)  
(b) Waymo

ISEAuto – Transfer Learning Convergence  
![](images/0109612c39a6d5b4898b51fdab86ca9ceb726a1288aba89a79fdf10c5a548cec.jpg)  
(c) ISEAuto

Fig. 6: Transfer learning convergence for CLFTv2-Tiny on ZOD, Waymo, and ISEAuto (100 epochs total, sampled every 10). Initialisation from related dataset checkpoints systematically yields higher early-epoch mIoU compared to standard ImageNet pretraining.  
![](images/59576b3224e71abe31b52ec7eb2a80a18c5e344db7c68906cce364ec1f6c3a7f.jpg)  
Fig. 7: Qualitative comparison of CLFTv2-Large and CLFT-Hybrid models. Left: ZOD (Day Fair, Snow). Right: Waymo (Day Rain, Night Fair).

TABLE XII: Recall (%) and precision (%) for safety-critical classes under fusion. High recall minimizes missed objects (e.g., vulnerable road users); high precision limits false positives.
<table><tr><td rowspan="3">Method</td><td colspan="4">ZOD</td><td colspan="4">Waymo</td><td colspan="4">ISEAuto</td></tr><tr><td>Day Fair</td><td>Prec.</td><td colspan="2">Night Rain Rec.</td><td colspan="2">Day Fair</td><td colspan="2">Night Rain</td><td colspan="2">Day Fair</td><td colspan="2">Night Rain</td></tr><tr><td>Rec.</td><td></td><td></td><td>Prec.</td><td>Rec.</td><td>Prec.</td><td>Rec.</td><td>Prec.</td><td>Rec.</td><td>Prec.</td><td>Rec.</td><td>Prec.</td></tr><tr><td>CLFTv2-Large</td><td>83.5</td><td>51.4</td><td>82.5</td><td>52.1</td><td>96.1</td><td>64.5</td><td>95.7</td><td>65.4</td><td>98.6</td><td>75.9</td><td>90.9</td><td>60.1</td></tr><tr><td>CLFTv2-Base</td><td>84.4</td><td>49.7</td><td>84.7</td><td>49.2</td><td>95.9</td><td>63.8</td><td>94.9</td><td>65.6</td><td>98.1</td><td>78.8</td><td>90.6</td><td>61.7</td></tr><tr><td>CLFTv2-Tiny</td><td>78.1</td><td>39.5</td><td>83.2</td><td>42.4</td><td>94.9</td><td>57.3</td><td>86.8</td><td>55.8</td><td>97.6</td><td>72.3</td><td>90.7</td><td>57.0</td></tr><tr><td>CLFT-Base</td><td>59.2</td><td>47.1</td><td>24.5</td><td>63.3</td><td>93.0</td><td>68.9</td><td>88.6</td><td>68.0</td><td>93.0</td><td>68.5</td><td>86.5</td><td>56.2</td></tr><tr><td>CLFT-Hybrid</td><td>73.7</td><td>45.8</td><td>82.9</td><td>54.4</td><td>93.4</td><td>67.1</td><td>84.8</td><td>68.3</td><td>96.9</td><td>71.9</td><td>91.8</td><td>60.2</td></tr><tr><td>CLFT-Large</td><td>59.1</td><td>49.3</td><td>39.2</td><td>64.8</td><td>92.5</td><td>72.1</td><td>87.2</td><td>71.7</td><td>97.5</td><td>72.8</td><td>85.9</td><td>63.1</td></tr><tr><td>MaskFormer-Large</td><td>61.9</td><td>64.6</td><td>65.8</td><td>71.5</td><td>69.2</td><td>67.8</td><td>69.0</td><td>66.5</td><td>88.3</td><td>85.9</td><td>71.6</td><td>74.2</td></tr><tr><td>MaskFormer-Base</td><td>62.9</td><td>63.9</td><td>67.3</td><td>68.2</td><td>66.5</td><td>69.9</td><td>64.2</td><td>65.6</td><td>88.3</td><td>86.4</td><td>71.5</td><td>75.5</td></tr><tr><td>Mask2Former-Base</td><td>61.8</td><td>63.4</td><td>69.4</td><td>66.3</td><td>66.5</td><td>67.7</td><td>60.6</td><td>70.6</td><td>90.0</td><td>86.3</td><td>77.8</td><td>70.9</td></tr><tr><td>Mask2Former-Large</td><td>61.4</td><td>64.1</td><td>68.7</td><td>63.8</td><td>66.0</td><td>67.9</td><td>65.9</td><td>70.7</td><td>89.1</td><td>87.0</td><td>77.5</td><td>75.0</td></tr><tr><td>Mask2Former-Tiny</td><td>52.1</td><td>52.2</td><td>55.1</td><td>48.6</td><td>59.0</td><td>59.8</td><td>52.7</td><td>65.2</td><td>84.9</td><td>82.4</td><td>74.0</td><td>69.7</td></tr><tr><td>MaskFormer-Tiny</td><td>47.5</td><td>57.4</td><td>50.8</td><td>56.3</td><td>59.8</td><td>60.8</td><td>56.9</td><td>56.7</td><td>83.5</td><td>82.0</td><td>74.8</td><td>68.4</td></tr><tr><td>DeepLabV3+</td><td>75.2</td><td>28.6</td><td>74.5</td><td>29.3</td><td>92.5</td><td>49.0</td><td>77.3</td><td>45.1</td><td>97.9</td><td>61.4</td><td>94.1</td><td>42.1</td></tr></table>

TABLE XIII: Weather Resilience Analysis on ZOD, Waymo, and ISEAuto fusion by Condition (mIoU % and FW IoU %). ISEAuto mIoU is the average of Human and Vehicle IoU (two classes; no traffic sign). Waymo and ISEAuto have no snow split.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">Day Fair</td><td colspan="2">Day Rain</td><td colspan="2">Night Fair</td><td colspan="2">Night Rain</td><td colspan="2">Snow</td></tr><tr><td>mIoU</td><td>FW IoU</td><td>mIoU</td><td>FW IoU</td><td>mIoU</td><td>FW IoU</td><td>mIoU</td><td>FW IoU</td><td>mIoU</td><td>FW IoU</td></tr><tr><td rowspan="10">ZOD</td><td>CLFTv2-Large</td><td>53.3</td><td>61.8</td><td>55.8</td><td>66.7</td><td>52.6</td><td>56.6</td><td>44.5</td><td>44.6</td><td>53.7</td><td>68.4</td></tr><tr><td>Mask2Former-Large</td><td>52.8</td><td>61.5</td><td>52.3</td><td>63.0</td><td>51.3</td><td>58.2</td><td>46.2</td><td>46.7</td><td>50.7</td><td>62.9</td></tr><tr><td>MaskFormer-Large</td><td>52.5</td><td>60.6</td><td>53.4</td><td>63.5</td><td>52.6</td><td>58.1</td><td>45.8</td><td>44.5</td><td>53.2</td><td>65.8</td></tr><tr><td>MaskFormer-Base</td><td>52.5</td><td>60.8</td><td>53.9</td><td>65.8</td><td>50.8</td><td>56.7</td><td>45.7</td><td>44.6</td><td>52.4</td><td>66.4</td></tr><tr><td>Mask2Former-Base</td><td>52.4</td><td>61.2</td><td>53.5</td><td>64.4</td><td>50.4</td><td>56.3</td><td>48.9</td><td>49.8</td><td>52.6</td><td>65.1</td></tr><tr><td>CLFTv2-Base</td><td>52.1</td><td>60.5</td><td>53.1</td><td>64.5</td><td>51.4</td><td>56.8</td><td>42.7</td><td>43.1</td><td>51.1</td><td>65.2</td></tr><tr><td>CLFT-Large</td><td>46.2</td><td>55.6</td><td>46.5</td><td>59.7</td><td>41.9</td><td>53.3</td><td>35.4</td><td>37.0</td><td>48.2</td><td>58.7</td></tr><tr><td>CLFT-Hybrid</td><td>46.2</td><td>55.0</td><td>48.5</td><td>60.4</td><td>48.4</td><td>56.3</td><td>39.7</td><td>37.8</td><td>45.6</td><td>55.9</td></tr><tr><td>CLFT-Base</td><td>44.6</td><td>54.1</td><td>46.3</td><td>59.1</td><td>40.8</td><td>52.4</td><td>31.1</td><td>36.2</td><td>40.3</td><td>57.3</td></tr><tr><td>Mask2Former-Tiny</td><td>43.0 43.0</td><td>52.2 51.9</td><td>42.8 43.3</td><td>55.4 54.5</td><td>41.6 41.9</td><td>49.8</td><td>33.1 33.3</td><td>33.6 33.6</td><td>44.2 41.8</td><td>58.1</td></tr><tr><td>CLFTv2-Tiny</td><td>MaskFormer-Tiny</td><td></td><td></td><td></td><td></td><td></td><td>48.6</td><td></td><td></td><td></td><td>55.3</td></tr><tr><td></td><td>DeepLabV3+</td><td>42.7 33.8</td><td>52.1 43.5</td><td>44.3 34.8</td><td>55.6 47.7</td><td>43.8 31.7</td><td>52.5 40.4</td><td>34.1 26.3</td><td>33.8 27.8</td><td>42.3 32.1</td><td>58.0 49.3</td></tr><tr><td rowspan="10"></td><td></td><td>68.8</td><td>79.2</td><td>67.5</td><td>78.5</td><td>66.9</td><td>78.0</td><td>64.7</td><td>74.3</td><td></td><td></td></tr><tr><td>CLFT-Large CLFT-Base</td><td>66.7</td><td>78.2</td><td>65.5</td><td>77.5</td><td>65.2</td><td>77.3</td><td>62.9</td><td>72.9</td><td></td><td></td></tr><tr><td>CLFT-Hybrid</td><td>66.0</td><td>76.7</td><td>65.3</td><td>76.0</td><td>64.9</td><td>76.3</td><td>62.3</td><td>72.2</td><td></td><td></td></tr><tr><td>CLFTv2-Large</td><td>61.9</td><td>67.5</td><td>60.8</td><td>65.0</td><td>61.1</td><td>68.3</td><td>59.6</td><td>63.7</td><td></td><td></td></tr><tr><td>CLFTv2-Base</td><td>61.2</td><td>67.2</td><td>60.1</td><td>64.8</td><td>60.5</td><td>68.0</td><td>58.7</td><td>63.1</td><td></td><td></td></tr><tr><td>CLFTv2-Tiny</td><td>55.8</td><td>63.9</td><td>54.7</td><td>61.1</td><td>55.7</td><td>64.8</td><td>51.5</td><td>58.5</td><td></td><td></td></tr><tr><td>Mask2Former-Large</td><td>49.5</td><td>54.3</td><td>48.9</td><td>50.8</td><td>49.0</td><td>56.0</td><td>47.1</td><td>48.8</td><td></td><td></td></tr><tr><td>Mask2Former-Base</td><td>49.4</td><td>54.1</td><td>48.6</td><td>50.5</td><td>48.5</td><td>55.6</td><td>45.8</td><td>47.5</td><td></td><td></td></tr><tr><td>MaskFormer-Large</td><td>51.0</td><td>56.8</td><td>50.5</td><td>53.9</td><td>50.5</td><td>57.9</td><td>47.9</td><td>50.5</td><td></td><td></td></tr><tr><td>MaskFormer-Base</td><td>51.0</td><td>57.0</td><td>50.6</td><td>54.0</td><td>50.4</td><td>57.9</td><td>46.7</td><td>49.8</td><td></td><td></td></tr><tr><td>DeepLabV3+</td><td>48.5</td><td>60.6</td><td>46.9</td><td>57.6</td><td>47.7</td><td>61.3</td><td>43.6</td><td>54.2</td><td></td><td></td></tr><tr><td>Mask2Former-Tiny</td><td>42.1</td><td>50.8</td><td>41.6</td><td>47.8</td><td>42.0</td><td>52.3</td><td>39.4</td><td>44.6</td><td></td><td></td></tr><tr><td>MaskFormer-Tiny</td><td>43.0</td><td>52.0</td><td>42.3</td><td>48.9</td><td>42.8</td><td>53.1</td><td>38.8</td><td>45.0</td><td></td><td></td></tr><tr><td rowspan="9"></td><td>Mask2Former-Large</td><td>80.7</td><td>81.7</td><td>73.6</td><td>83.5</td><td>73.3</td><td>79.0</td><td>65.4</td><td>68.7</td><td></td><td></td></tr><tr><td>Mask2Former-Base</td><td>80.7</td><td>81.6</td><td>73.4</td><td>83.4</td><td>72.5</td><td>79.7</td><td>63.1</td><td>66.7</td><td></td><td></td></tr><tr><td>CLFTv2-Base</td><td>80.5</td><td>82.0</td><td>75.7</td><td>84.0</td><td>72.6</td><td>80.6</td><td>63.4</td><td>68.1</td><td></td><td></td></tr><tr><td>MaskFormer-Large</td><td>80.0</td><td>81.5</td><td>72.6</td><td>83.2</td><td>70.0</td><td>78.8</td><td>62.5</td><td>67.0</td><td></td><td></td></tr><tr><td>MaskFormer-Base</td><td>79.9</td><td>81.1</td><td>72.2</td><td>83.0</td><td>73.1</td><td>79.4</td><td>63.4</td><td>68.2</td><td></td><td></td></tr><tr><td>CLFTv2-Large</td><td>77.6</td><td>78.9</td><td>72.6</td><td>81.3</td><td>70.3</td><td>78.3</td><td>61.7</td><td>66.1</td><td></td><td></td></tr><tr><td>Mask2Former-Tiny</td><td>75.5</td><td>77.4</td><td>65.0</td><td>79.2</td><td>66.5</td><td>74.8</td><td>60.7</td><td>64.8</td><td></td><td></td></tr><tr><td>MaskFormer-Tiny</td><td>74.7</td><td>76.8</td><td>66.8</td><td>78.7</td><td>66.0</td><td>74.0</td><td>66.0</td><td>74.0</td><td></td><td></td></tr><tr><td>CLFTv2-Tiny CLFT-Large</td><td>74.2 73.5</td><td>75.8 74.5</td><td>68.6 68.0</td><td>79.0 77.5</td><td>66.0 65.0</td><td>74.8 74.3</td><td>60.4 61.2</td><td>66.2 64.7</td><td></td><td></td></tr></table>

TABLE XIV: A100 GPU inference efficiency. Latency is averaged over 100 runs.
<table><tr><td>Method</td><td>Params (M)</td><td>FLOPs (G)</td><td>Time (ms)</td><td>Mem (MB)</td></tr><tr><td>DeepLabV3+</td><td>118.7</td><td>44.7</td><td>19.1</td><td>461</td></tr><tr><td>CLFTv2-Tiny</td><td>43.1</td><td>30.9</td><td>22.0</td><td>187</td></tr><tr><td>CLFT-Base</td><td>115.5</td><td>185.3</td><td>25.4</td><td>466</td></tr><tr><td>MaskFormer-Tiny</td><td>64.2</td><td>17.3</td><td>25.8</td><td>281</td></tr><tr><td>CLFT-Hybrid</td><td>127.6</td><td>185.5</td><td>37.2</td><td>512</td></tr><tr><td>Mask2Former-Tiny</td><td>71.0</td><td>42.3</td><td>48.4</td><td>314</td></tr><tr><td>CLFTv2-Base</td><td>102.6</td><td>119.0</td><td>54.3</td><td>495</td></tr><tr><td>MaskFormer-Base</td><td>145.6</td><td>97.2</td><td>62.9</td><td>680</td></tr><tr><td>CLFT-Large</td><td>341.2</td><td>440.5</td><td>63.3</td><td>1328</td></tr><tr><td>CLFTv2-Large</td><td>211.4</td><td>204.4</td><td>81.8</td><td>907</td></tr><tr><td>Mask2Former-Base</td><td>152.4</td><td>153.0</td><td>90.7</td><td>719</td></tr><tr><td>MaskFormer-Large</td><td>316.8</td><td>209.5</td><td>91.9</td><td>1335</td></tr><tr><td>Mask2Former-Large</td><td>323.6</td><td>265.3</td><td>118.2</td><td>1374</td></tr></table>

## G. Limitations and Future Work

Our evaluation reveals several limitations of local-window multi-modal fusion. First, the observed accuracy gap between CLFTv2 and classical ViT-based architectures on the geometrically dense Waymo dataset indicates a limitation of local receptive fields when processing dense point clouds. While shifted-window attention yields high efficiency, its localized receptive field limits cross-modal integration under conditions of extreme point-cloud density. Future work should explore hybrid architectures that embed sparse, long-range token mixing within hierarchical backbones.

Second, our evaluation inherently incorporates the challenges of varied supervision quality. Models evaluated on the ZOD dataset were supervised via SAM-generated pseudolabels. Evaluating on high-quality, human-annotated datasets in the future will help separate the model’s architectural capabilities from its resilience to noisy pseudo-labels.

Finally, while the hardware efficiency metrics presented demonstrate substantial gains over query-based decoders, they are presently bound to a high-end compute profile (NVIDIA A100). Validating these lightweight architectures across lowerpower edge devices and micro-controllers represents a next step for real-time vehicular deployment.

## VI. CODE AVAILABILITY

To facilitate reproducibility and further research in multimodal perception, the full source code for CLFTv2, including the training pipeline and model definitions, is publicly available at: https://github.com/taltech-av/paper-tvt2026-clftv2. This repository also contains the data preprocessing and conversion scripts for all three training datasets. Training results can be found on https://app.visin.eu/projects/clftv2.

## VII. CONCLUSION

This work introduces CLFTv2, a hierarchical Swin-based camera–LiDAR fusion framework designed to overcome the computational bottlenecks of multi-modal perception for autonomous driving. CLFTv2 achieves a favorable accuracyefficiency trade-off compared to leading universal segmentation architectures, reducing latency and compute overhead while maintaining performance on safety-critical classes.

Our evaluation across three distinct autonomous driving datasets demonstrates that architectural complexity is not universally beneficial: under the sparse and noisy supervision typical of large-scale open datasets, lightweight residual fusion performs on par with computationally demanding querymatching decoders. Furthermore, we find that segmentation safety profiles are heavily architecture-dependent; the proposed CLFTv2 family prioritises high recall for vulnerable road users, while precision varies by dataset sensor characteristics.

These findings show the practical trade-offs between localwindow efficiency and global cross-modal alignment, guiding the design of scalable, real-time perception architectures.

## REFERENCES

[1] A. Valada, R. Mohan, and W. Burgard, “Self-supervised model adaptation for multimodal semantic segmentation,” International Journal of Computer Vision, vol. 128, no. 5, pp. 1239–1285, 2020.

[2] C. Xu, B. Wu, Z. Wang, W. Zhan, P. Vajda, K. Keutzer, and M. Tomizuka, “Squeezesegv3: Spatially-adaptive convolution for efficient point-cloud segmentation,” in ECCV, 2020.

[3] T. Tahves, J. Gu, M. Bellone, and R. Sell, “A novel vision transformer for camera-lidar fusion based traffic object segmentation,” in Proceedings of the 17th International Conference on Agents and Artificial Intelligence - Volume 2: ICAART, INSTICC. SciTePress, 2025, pp. 566–573.

[4] J. Gu, M. Bellone, T. Pivonka, and R. Sell, “Clft: Camera-lidar fusionˇ transformer for semantic segmentation in autonomous driving,” IEEE Transactions on Intelligent Vehicles, pp. 1–12, 2024.

[5] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in ICCV, 2021.

[6] W. Wang, E. Xie, X. Li, D.-P. Fan, K. Song, D. Liang, T. Lu, P. Luo, and L. Shao, “Pyramid vision transformer: A versatile backbone for dense prediction without convolutions,” in Proceedings ofthe IEEE/CVF international conference on computer vision, 2021, pp. 568–578.

[7] D. Xu, D. Anguelov, and A. Jain, “Pointfusion: Deep sensor fusion for 3d bounding box estimation,” in CVPR, 2018.

[8] S. Vora, A. H. Lang, B. Helou, and O. Beijbom, “Pointpainting: Sequential fusion for 3d object detection,” in CVPR, 2020.

[9] Z. Liu, H. Tang, A. Amini, X. Yang, H. Mao, D. Rus, and S. Han, “Bevfusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation,” 2024. [Online]. Available: https: //arxiv.org/abs/2205.13542

[10] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li, “Planning-oriented autonomous driving,” 2023. [Online]. Available: https://arxiv.org/abs/2212.10156

[11] X. Bai, Z. Hu, X. Zhu, Q. Huang, Y. Chen, H. Fu, and C.-L. Tai, “Transfusion: Robust lidar-camera fusion for 3d object detection with transformers,” in CVPR, 2022.

[12] Z. Liu, H. Hu, Y. Lin, Z. Yao, Z. Xie, Y. Wei, J. Ning, Y. Cao, Z. Zhang, L. Dong, F. Wei, and B. Guo, “Swin transformer v2: Scaling up capacity and resolution,” in CVPR, 2022.

[13] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, and M. Wang, “Swin-unet: Unet-like pure transformer for medical image segmentation,” in Proceedings of the European Conference on Computer Vision (ECCV) Workshops, 2022, pp. 205–218.

[14] Z. Li, Z. Chen, X. Liu, and J. Jiang, “Depthformer: Exploiting longrange correlation and local information for accurate monocular depth estimation,” Machine Intelligence Research, vol. 20, no. 6, pp. 837–854, 2023.

[15] B. Cheng, A. Schwing, and A. Kirillov, “Per-Pixel Classification is Not All You Need for Semantic Segmentation,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 17 864–17 875.

[16] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girdhar, “Masked-attention mask transformer for universal image segmentation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 1290–1299.

[17] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” 2017. [Online]. Available: https://arxiv.org/abs/1612.03144

[18] R. Ranftl, A. Bochkovskiy, and V. Koltun, “Vision transformers for dense prediction,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 12 179–12 188.

[19] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable DETR: Deformable transformers for end-to-end object detection,” in International Conference on Learning Representations, 2021.

[20] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[21] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chou, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine et al., “Scalability in perception for autonomous driving: Waymo open dataset,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 2446–2454.

[22] R. Sell, M. Malayjerdi, E. Malayjerdi, M. Bellone, and H. Pikner, “Autonomous vehicle for industry 5.0: Digital twin for system safety validation,” in Proceedings of the 11th International Conference on Vehicle Technology and Intelligent Transport Systems - VEHITS, INSTICC. SciTePress, 2025, pp. 660–667.

[23] J. Gu, M. Bellone, R. Sell, and A. Lind, “Object segmentation for autonomous driving using iseauto data,” Electronics, vol. 11, no. 7, 2022. [Online]. Available: https://www.mdpi.com/2079-9292/11/7/1119

[24] M. Alibeigi, W. Ljungbergh, A. Tonderski, G. Hess, A. Lilja, C. Lindstrom, D. Motorniuk, J. Fu, J. Widahl, and C. Petersson, “The zenseact¨ open dataset: A large-scale and diverse multimodal dataset for autonomous driving,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 20 178–20 188.

[25] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,´ “Segment anything,” in ICCV, 2023.

[26] H. Herrmann, T. Kaevand, and L. Anton, “Base: Taltech’s hpc infrastructure 2020–2024,” TalTech Data Repository, Mar. 2025.

[27] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” 2018. [Online]. Available: https://arxiv.org/abs/1802. 02611

[28] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR, 2015.