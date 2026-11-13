# EG360: Efficient and Generalizable 360<sup>◦</sup> Depth Estimation via Adaptive Panorama-Perspective Distillation and Spherical Cross-Level Feature Fusion

Laiyan Ding CUHKSZ

Chongxi Meng Tongji University

Rui Huang CUHKSZ

## Abstract

Generalized panorama depth estimation has witnessed great advances in recent years. These methods typically achieve robust zero-shot generalization performance using large pretrained models, but their high computational cost makes real-time inference on embeddedplatforms chal lenging. In contrast, previous lightweight panorama models have mostly been developed for single-domain evaluations and cannot generalize well due to domain shifts. To reduce the gap between these two types of models, we introduce EG360, an efficient and generalizable panorama depth estimation model. Specifically, building on a lightweight depth model, we design an Adaptive Panorama-Perspective Distillation (APPD) strategy guided by scene geometry, along with a Spherical Cross-Level Feature Fusion (SCFF) modulefor global semantic perception. As a result, with only 6.593M parameters after reparameterization, EG360 achieves strong zero-shot performance on real-world benchmarks and runs at 35 FPS on a Jetson Orin NX 16GB in MAXN mode, whereas a previous ViTsmall model runs at less than 2 FPS. Our code will also be made publicforfuture research.

## 1. Introduction

In recent years, there has been a wide range of work that addresses challenges in processing panoramic images for various computer vision tasks, including depth estimation [5, 17, 23], segmentation [55, 58, 60], and tracking [16, 25, 50]. Among these, monocular panorama depth estimation is an important task that infers the 3D geometry of the corresponding scene from a 360 degree image, which can be useful for other applications such as virtual reality [3, 44] and autonomous driving [30, 38].

In the domain of monocular panorama depth estimation, some generalizable methods [23, 46] that can perform robust zero-shot predictions have emerged. Though they can achieve good performance, they tend to be based on large foundation models [13, 28], which are hard to meet realtime requirements on mobile platforms, e.g., Jetson Orin NX, an embedded AI computing platform for robotics. As shown in Fig. 1, the DAP [23] model, which is built on top of a ViT-Large model [13, 34], obtains an average AbsRel of 0.135 on Matterport3D [7], Stanford2D3D [2] and Metropolis [26]. However, it can only run at 0.54 FPS on a Jetson Orin NX 16GB in MAXN mode. In contrast, previous lightweight models [17, 42, 62] focus on training and testing within the same domain and show limited crossdomain generalization performance. As shown in Fig. 1, UniFuse [17] achieves only an average AbsRel of 0.229 and runs at 11 FPS. Thus, current panorama depth estimation models fail to deliver depth estimation results with a plausible trade-off between quality and efficiency.

![](images/3694f248f4cd77d4a9ecf21419ef9d9c33fd4a100c630574132c4d08cddf108a.jpg)  
Figure 1. Accuracy–computation trade-off for panoramic depth estimation. Mean AbsRel is averaged equally over Matterport3D, Stanford2D3D, and Metropolis. Filled marker area is positively proportional to measured inference FPS on a Jetson Orin NX 16GB (MAXN mode, PyTorch eager, AMP FP16, batch size 1). EG360 uses reparameterization for inference. PaGeR ran into outof-memory (OOM) and is represented by a fixed-size hollow diamond.

We argue that to achieve both efficiency and generalization, exposing the model to more real-world data via distillation and allowing it to capture semantic information efficiently are necessary. Concretely, we propose an Adaptive Panorama-Perspective Distillation (APPD) strategy to exploit panorama and perspective teachers jointly. Whereas Depth Anywhere [45] and PanDA [6] rely on perspective and panoramic teachers, respectively, we combine their complementary supervision. Table 2 shows that DAP [23] performs better on Stanford2D3D, which is indoor, while MoGe2 [47] outperforms on Metropolis, which is outdoor. Their relative usefulness may also vary across views: a perspective crop dominated by a wall may provide limited cues for estimating absolute distance, whereas a panorama teacher can use the surrounding contexts. Motivated by this, APPD uses the proposed normalized plane-fitting residuals to dynamically adaptively weigh the two teacher losses, assigning more weight to the panorama teacher for more planar views. Additionally, to aggregate global contexts while accounting for spherical geometry, we introduce Spherical Cross-Level Feature Fusion (SCFF). SCFF uses crosslevel attention [40] between the two coarsest encoder feature maps, with the finer map querying the coarser one. Attention combines feature similarity with a spherical direction bias that reflects the spatial proximity on the sphere.

Combining these two proposed techniques, we present EG360, a lightweight monocular panorama depth estimation model that can perform robust zero-shot predictions. Training on abundant labeled data [36, 48, 59] together with unlabeled data [9, 61], EG360, comprising 6.593M parameters and costing 23.492 GFLOPs after structural reparameterization [12], can achieve an average AbsRel of 0.168 and run on Orin NX at 35 FPS. It not only runs faster than previous lightweight models but also narrows the gap between lightweight and foundation models.

In summary, our contributions are as follows:

• We propose Adaptive Panorama–Perspective Distillation (APPD), which uses normalized plane-fitting residuals to weigh panorama and perspective teacher supervision for unlabeled data adaptively.

• We introduce Spherical Cross-Level Feature Fusion (SCFF), which combines cross-level attention with a spherical direction bias to aggregate global context between the two coarsest feature maps.

• We develop EG360, a lightweight model with 6.593M parameters and 23.492 GFLOPs, and demonstrate its zeroshot performance on real-world benchmarks.

## 2. Related Works

## 2.1. In-Domain Panorama Depth Estimation

Depth estimation for panorama images presents unique challenges due to the equirectangular projection (ERP) and resulting distortions. Some works directly work on the ERP image and propose different techniques to tackle the distortion. OmniDepth [63] uses both rectangular and square CNN filters. ACDNet [62] combines convolution kernels with different dilations with adaptive channel-wise fusions. PanoFormer [33] and EGFormer [54] propose panoramaspecific transformer blocks to adapt widely used transformer [40] blocks to the panorama domain. Some other works use additional distortion-free formats for 360 images to mitigate the negative effects of distortions. BiFuse [42], BiFuse++ [43], and UniFuse [17] use cubemap (CP) images and fuse the features from CP and ERP images adaptively for better performance than using ERP or CP alone. HRDFuse [1] enhances depth prediction by collaboratively learning from ERP and tangent projections. In addition, S<sup>2</sup>Net [21] moves to the spherical surface generated from the ERP image with the HEALPix sampling [14]. Nevertheless, these methods are typically trained and evaluated within the same domain, limiting their generalizability.

## 2.2. Zero-Shot Panorama Depth Estimation

Different from in-domain methods, zero-shot methods aim to perform robust predictions on unseen data. Depth AnyWhere [45] distills the knowledge from large perspective models [52] to small panorama models via unlabeled panorama data, while PanDA [6] distills from large panorama models. DA360 [18] finetunes Depth Anything V2 [53] and equips the DPT [32] decoder with ERP padding. DA<sup>2</sup> [20] generates high-quality panoramic depth data from perspective data via perspective-to-panorama diffusion [37]. DAP [23] introduces a three-stage curation pipeline to generate reliable pseudo labels for unlabeled images. DepthMaster [46] and PaGeR [5] utilize a Depth Anything V3 [22] structure to process cubemap images and produce ERP depth images. However, these zero-shot methods with large backbones tend to be computationally expensive, making them unsuitable for deployment on mobile platforms. Thus, we aim to develop a lightweight model by distilling complementary knowledge from panoramic and perspective teachers adaptively and enabling efficient spherical global context modeling.

## 2.3. Zero-Shot Perspective Depth Estimation

Recent advances in monocular depth estimation from perspective images have benefited from improved architectures and increasingly diverse training data. MiDaS [31] enables joint learning from heterogeneous datasets using a robust, scale-shift-invariant loss. Depth Anything [52] and Depth Anything V2 [53] leverage large-scale unlabeled images to improve generalization. Beyond relative depth estimation, MoGe2 [47] recovers metric-scale 3D point maps from single images using a decoupled design that separates relative geometry and metric-scale learning. Despite their strong generalization capabilities, their computation costs make them unfriendly to mobile platforms. To improve deployment efficiency, FastDepth [49] combines a compact encoder-decoder architecture with network pruning for embedded inference. More recently, ZipDepth [39] enables efficient zero-shot depth estimation with large-scale knowledge distillation from a depth foundation model. However, these approaches are designed for perspective images and cannot be applied directly to panorama images.

<table><tr><td>Datasets</td><td>Indoor</td><td>Outdoor Samples</td></tr><tr><td colspan="3">Labeled Synthetic Datasets</td></tr><tr><td>Structured3D [59]</td><td>√</td><td>18,222</td></tr><tr><td>PanoCARLA [36]</td><td>√ √ √</td><td>112,212 432,381</td></tr><tr><td>TartanAirV2 [48] Labeled total</td><td></td><td>562,815</td></tr><tr><td colspan="3">Unlabeled Real-world Datasets</td></tr><tr><td>ZInD [9]</td><td>√</td><td>67,448</td></tr><tr><td>HoliCity [61]</td><td>√</td><td>6,253</td></tr><tr><td>Unlabeled total</td><td></td><td>73,701</td></tr></table>

Table 1. Labeled synthetic and unlabeled real-world datasets used.

## 3. Method

## 3.1. Overview and our Baseline

Given an equirectangular projection (ERP) image $I \in { \cal U }$ $\mathbb { R } ^ { H \times W \times 3 }$ , we estimate its metric depth map $\hat { D } \in$ $\mathbb { R } ^ { H \times W \times 1 }$ , where H and W denote the image height and width. The network predicts the inverse-depth map sˆ, and we recover metric depth as $\hat { D } = 1 / \hat { s }$

We first finetune the base variant of ZipDepth [39] as our baseline since it has shown great accuracy-efficiency tradeoffs in perspective depth estimation. Specifically, we initialize the model from the released official ZipDepth checkpoint and optimize metric inverse-depth objectives on labeled panorama data. Building on this baseline, we introduce APPD and SCFF to enhance its generalization to unseen data, resulting in our new model, EG360.

## 3.2. Training Data and Complementary Teachers

Labeled and unlabeled panorama data Similar to PanDA [6], we train our model on joint labeled synthetic data and unlabeled real-world data. Table 1 reports the actual samples used. The labeled set contains 562,815 RGBdepth pairs from Structured3D [59], PanoCARLA [36], and TartanAirV2 [48]. Note that Structured3D RGB images rendered from the same viewpoint under three lighting conditions share the same ground-truth depth map, and TartanAir panoramas are rectified to an upright orientation offline following DA360 [18]. The unlabeled set contains 73,701 real panoramas from ZInD [9] and HoliCity [61].

<table><tr><td rowspan="2">Teacher</td><td colspan="2">Stanford2D3D</td><td colspan="2">Metropolis</td></tr><tr><td>AbsRel↓</td><td> $\delta _ { 1 } \uparrow$ </td><td>AbsRel↓</td><td> $\delta _ { 1 } \uparrow$ </td></tr><tr><td>DAP [23]</td><td>0.071</td><td>0.965</td><td>0.185</td><td>0.733</td></tr><tr><td>MoGe2 [47]</td><td>0.150</td><td>0.774</td><td>0.127</td><td>0.862</td></tr></table>

Table 2. Comparison of panorama and perspective teachers. Metrics are averaged equally over six cube faces for Stanford2D3D and four horizontal faces for Metropolis, as its top and bottom faces lack valid ground-truth depth. Best results are bolded.

Pseudo labels from teachers DAP operates on the ERP image, while MoGe2 processes six $9 0 °$ cubemap faces and we use superscripts P and C for the DAP and MoGe2 teachers, respectively. Both teachers are frozen, and their predictions are cached offline. In particular, for each cubemap face $c \in \{ 1 , \ldots , 6 \}$ , MoGe2 predicts an axial depth map $z _ { c } ,$ and we convert it into radial depth map $D _ { c } ( u , v )$

$$
D _ { c } ( u , v ) = z _ { c } ( u , v ) \sqrt { 1 + \left( \frac { u - c _ { x } } { f } \right) ^ { 2 } + \left( \frac { v - c _ { y } } { f } \right) ^ { 2 } }\tag{1}
$$

Here, each cube face has a resolution of $R \times R$ pixels, and $( u , v )$ denotes the horizontal and vertical pixel coordinates. The principal point $( c _ { x } , c _ { y } )$ lies at the image center. The focal length $f = R / 2$ follows from the $9 0 ^ { \circ }$ field of view of each cube face. The teacher targets $t ^ { \mathrm { P } }$ and $t ^ { \mathrm { C } }$ are inverse radial depths, with $t _ { c } ^ { \mathrm { C } } = 1 / D _ { c }$ . Additionally, both teachers have a segmentation head that identifies the sky region in an outdoor image, and we set the distance of pixels in these regions to be 1000m. For labeled data, we also cap the distance to 1000m as pixels farther than 1000m are sky pixels.

Panorama and perspective teachers are complementary Different from previous perspective-teacher-only [45] or panorama-teacher-only [6] distillation, we design a joint panorama-perspective distillation mechanism to combine the power from both worlds. To this end, we first verify that they are complementary across indoor and outdoor scenarios. Concretely, we evaluate the performance of the panoramic teacher DAP [23] and the perspective teacher MoGe2 [47] on Stanford2D3D [2] and Metropolis [26], which are purely indoor and outdoor. For DAP, we input an ERP image, obtain its ERP depth, and convert it to CP depth. For MoGe2, we input CP images and get CP depth. Then evaluations are performed in the CP space in a facewise manner. As shown in Table 2, DAP performs better than MoGe2 on Stanford2D3D, whereas MoGe2 is more competitive on Metropolis. These observations motivate us to combine both panorama and perspective teachers for more effective distillation.

![](images/b07d0fcc902ad81416bc4b30a3a55f41fd081993b5fa577676d20a9bccad296a.jpg)  
Figure 2. Visualization of geometry-adaptive teacher weighting. Faces dominated by planar surfaces, such as walls or ceilings, receive higher panorama teacher weights, while views with greater geometric variation receive lower weights (blue or green).

## 3.3. Geometry-Guided Adaptive Teacher Weighting

As shown in Table 2, panorama and perspective teachers can be complementary, but the relative usefulness of the two teachers may vary across cube faces within the same panorama. A perspective view dominated by a wall may provide limited cues for estimating its absolute distance, while the panoramic teacher can exploit the surrounding scene. Thus, we use the predicted geometry of each cube face to weigh the two teachers’ supervision dynamically.

Normalized plane-fitting residual For each valid, nonsky pixel i on face c, we reconstruct MoGe2 3D point as $\mathbf { p } _ { c , i } = \pmb { \nu } _ { c , i } / t _ { c , i } ^ { \mathrm { C } }$ , where $t _ { c , i } ^ { \mathrm { C } }$ is the projected inverse radial depth and $\nu _ { c , i }$ is the unit viewing ray. Then, we fit a single plane Π to these points using PCA, minimizing the sum of squared perpendicular distances to the plane. Let $\mathcal { V } _ { c }$ denote the set of valid pixels on face $c .$ We compute the mean radial depth $\bar { D } _ { c }$ and the normalized plane-fitting residual $r _ { c }$ as:

$$
\begin{array} { l } { \displaystyle \bar { D } _ { c } = \frac { 1 } { | \mathcal { V } _ { c } | } \sum _ { i \in \mathcal { V } _ { c } } \| \mathbf { p } _ { c , i } \| _ { 2 } , } \\ { \displaystyle r _ { c } = \frac { 1 } { \bar { D } _ { c } } \sqrt { \frac { 1 } { | \mathcal { V } _ { c } | } \sum _ { i \in \mathcal { V } _ { c } } \mathrm { d i s t } ( \mathbf { p } _ { c , i } , \Pi _ { c } ) ^ { 2 } } , } \end{array}\tag{2}
$$

Here, $\bar { D } _ { c }$ is the mean distance from the camera to these points, and dist denotes the distance to the fitted plane. Intuitively, a smaller $r _ { c }$ indicates more planar geometry.

Teacher weights We assign complementary weights to the panoramic teacher DAP (P) and the perspective teacher MoGe2 (C):

$$
w _ { c } ^ { \mathrm { P } } = \frac { \tau } { r _ { c } + \tau } , \qquad w _ { c } ^ { \mathrm { C } } = \frac { r _ { c } } { r _ { c } + \tau } , \qquad \tau > 0 .\tag{3}
$$

Thus, more planar faces receive stronger panorama teacher supervision, while larger residuals increase the perspective teacher’s contribution. The parameter $\tau$ controls this balance, with equal weights at $r _ { c } = \tau$ . In practice, we set τ to be 0.3 experimentally. As shown in Fig. 2, our weighting is adaptive based on the view geometry.

![](images/9d7156df1319fdbf20450d4de52d643eab56ee920c0a74cd851c001a532ae172.jpg)  
Figure 3. Diagram of Spherical Cross-Level Feature Fusion. Cross-attention with a spherical direction bias is followed by a local feed-forward network with panorama padding and depthwise convolution. Reshaping is omitted for brevity.

Weighted L1 distillation We combine DAP supervision in ERP space and MoGe2 supervision in cubemap space. For one panorama, the distillation loss is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { d i s t i l l } } = \displaystyle \frac { 1 } { N _ { \mathrm { P } } } \sum _ { i = 1 } ^ { N _ { \mathrm { P } } } w _ { i } ^ { \mathrm { P } } \big | \hat { s } _ { i } - t _ { i } ^ { \mathrm { P } } \big | } \\ & { \quad \quad \quad + \displaystyle \frac { 1 } { | \mathcal { F } | } \sum _ { c \in \mathcal { F } } \frac { w _ { c } ^ { \mathrm { C } } } { N _ { c } } \sum _ { j = 1 } ^ { N _ { c } } \big | \hat { s } _ { c , j } - t _ { c , j } ^ { \mathrm { C } } \big | , } \end{array}\tag{4}
$$

Here, $\hat { s }$ and t denote student and teacher inverse depths. The value $\hat { s } _ { c , j }$ is obtained by projecting the student’s ERP inverse-depth prediction onto cube face c. $N _ { \mathrm { P } }$ is the number of valid pixels for the panorama teacher and $N _ { c }$ is the number of valid pixels for the perspective teacher on cube $c .$ The set $\mathcal { F }$ contains valid cube faces. Each ERP pixel receives the DAP weight of its corresponding cube face.

## 3.4. Spherical Cross-Level Feature Fusion

Previous works have shown that perceiving the global context is important for depth estimation [27, 57]. Also, our ZipDepth baseline already implements several modules to enhance global perception. However, due to the distortion and left-right discontinuity in ERP images, its global modeling is insufficient, so we design a Spherical Cross-Level Feature Fusion (SCFF) module to further enhance its contextual modeling capabilities. As shown in Fig. 3, it consists of a spherical cross-level attention module and a panorama local feed-forward network.

Cross-level attention Let $C _ { 3 }$ and $C _ { 4 }$ denote the outputs of the third and fourth encoder stages. We flatten their spatial dimensions into token sequences $X _ { 3 }$ and $X _ { 4 }$ , respectively. Queries, keys, and values are computed as $Q \ =$ $\mathrm { L N } ( X _ { 3 } ) W _ { Q } , K = \mathrm { L N } ( X _ { 4 } ) W _ { K }$ , and $V = \mathrm { L N } ( X _ { 4 } ) W _ { V }$ Here, LN denotes layer normalization [4], and $W _ { Q } , W _ { K }$ and $W _ { V }$ are learnable projection matrices. Bias terms are omitted for brevity. In this way, each fine-scale query can therefore aggregate information from the coarse feature.

Spherical direction bias Furthermore, we incorporate spherical geometry into attention to account for the spatial relationships represented by ERP images. For a feature token at normalized pixel-center coordinates $( x , y )$ , its longitude and latitude are $\begin{array} { r } { \lambda = 2 \pi ( x - \frac { 1 } { 2 } ) } \end{array}$ and $\phi = \pi ( { \textstyle { \frac { 1 } { 2 } } } - y )$ . Its corresponding unit viewing direction is

$$
\boldsymbol { \omega } = [ \cos \phi \cos \lambda , \sin \phi , \cos \phi \sin \lambda ] ^ { \mathsf { T } } .\tag{5}
$$

We compute these directions for feature grids and construct a distance matrix G, where $G _ { i j } ~ = ~ 1 - \omega _ { i } ^ { \intercal } \omega _ { j }$ measures the directional distance between query i and key $j .$ After splitting Q, K, and V into heads, each head computes (head indices are omitted for brevity)

$$
A = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \mathsf { T } } } { \sqrt { d _ { h } } } - \beta \mathbf { G } \right) ,\tag{6}
$$

Here, $d _ { h }$ is the query/key dimension of each head. Each head has its own positive bias strength $\beta = \mathrm { s o f t p l u s } ( b )$ with learnable parameter $b .$ This spherical direction bias favors nearby spherical directions, including neighbors across the ERP left-right seam. Let $Z$ denote the outputs AV from all heads. The residual update is $U = X _ { 3 } + Z W _ { O }$ , where the learnable projection $W _ { O }$ maps $Z$ back to the channel dimension of $X _ { 3 }$

Panoramic local feed-forward network. We insert a local panoramic feed-forward network after the attention:

$$
\widetilde { X } _ { 3 } = U + \mathrm { G E L U } \left( \mathrm { D W C o n v } _ { 3 \times 3 } ^ { \mathrm { E R P } } \left( \mathrm { L N } ( U ) W _ { 1 } \right) \right) W _ { 2 } .\tag{7}
$$

The matrices $W _ { 1 }$ and $W _ { 2 }$ are the learnable input and output projections of the feed-forward network, respectively. The $\mathrm { \Delta D W C o n v _ { 3 \times 3 } ^ { E R P } }$ is a depthwise convolution with panoramic padding [18]. Following the zero-initialization strategy of ControlNet [56], we initialize the output projections $W _ { O }$ and $W _ { 2 }$ to zero. In the end, we reshape ${ \widetilde { X } } _ { 3 }$ to refined ${ \bar { C } } _ { 3 } .$ which is passed together with $C _ { 4 }$ to the original decoder.

## 3.5. Overall loss function

Labeled data loss For labeled data, $D ^ { * }$ denotes the ground-truth metric depth map and $s ^ { * } = 1 / D ^ { * }$ its inversedepth map. Let M contain all valid pixels in the supervised batch. At pixel $p , \hat { s } _ { p }$ and $s _ { p } ^ { * }$ denote the predicted and ground-truth inverse depths, respectively. The supervised loss combines metric inverse-depth regression with a multiscale gradient term [18]:

$$
\mathcal { L } _ { \mathrm { s u p } } = \frac { 1 } { | \mathcal { M } | } \sum _ { p \in \mathcal { M } } | \hat { s } _ { p } - s _ { p } ^ { * } | + \lambda _ { \mathrm { g } } \mathcal { L } _ { \mathrm { g r a d } }\tag{8}
$$

The $\lambda _ { \mathrm { g } }$ controls the gradient loss weight and is set to 1.

Overall objective For training, the overall objective combines the supervised loss and the distillation loss:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { s u p } } + \mathcal { L } _ { \mathrm { d i s t i l l } } .
$$

## 4. Experiments

(9)

## 4.1. Experimental Setup

Data for training and evaluation We use the labeled synthetic and unlabeled real panoramas in Table 1 for training. We evaluate cross-dataset generalization on the Matterport3D [7], Stanford2D3D [2], and Metropolis [26] test sets, containing 2,014, 373, and 3,000 panoramas, respectively, in a zero-shot manner. The maximum evaluation depth is 10 meters for Matterport3D and Stanford2D3D, and 100 meters for Metropolis. We report the absolute relative error (AbsRel) and threshold accuracy $( \delta _ { 1 } )$ . For each image, let V denote the set of valid evaluation pixels, |V| its size, and $\hat { d } _ { p }$ and $d _ { p } ^ { * }$ the predicted and ground-truth depths at pixel $p ,$ respectively. AbsRel is the mean absolute depth error normalized by the ground-truth depth, AbsRel = $\begin{array} { r } { \frac { 1 } { | \mathcal { V } | } \sum _ { p \in \mathcal { V } } \frac { | \hat { d } _ { p } - d _ { p } ^ { * } | } { d _ { p } ^ { * } } } \end{array}$ . The threshold accuracy $\delta _ { 1 }$ is the fraction of pixels $p \in \mathcal V$ satisfying max $( \hat { d } _ { p } / d _ { p } ^ { * } , d _ { p } ^ { * } / \hat { d } _ { p } ) < 1 . 2 5$ Note that all evaluations are done without any alignment.

Implementation Details We initialize the base student from the released ZipDepth checkpoint and train each of our configurations independently for 100,000 iterations at 512 × 1024 ERP resolution using the PyTorch [29] framework on an RTX 2080 Ti. We use AdamW [24] with weight decay $1 0 ^ { - 3 }$ , gradient-norm clipping at 1.000, and AMP FP16. We also use a OneCycle [35] learning schedule with a peak learning rate of $1 0 ^ { - 4 }$ , a 1% warm-up fraction, and linear annealing. Unless stated otherwise, in each iteration we use a batch size of 8 for labeled data and 8 for unlabeled data when using distillation. We use the checkpoint from the last step to evaluate for fairness. Augmentation includes horizontal flips, color jitter, random yaw and pitch rotations, and polar RGB masking, following DA360 [18].

Comparison settings For UniFuse, BiFuse++, and ACD-Net, we retrain them using ImageNet-pretrained ResNet-18 encoders. For DA360-S, the ViT-Small version, we finetune its released checkpoint. These four models are retrained on our labeled data for fairness. For DAP, DepthMaster, and PaGeR, we directly use released weights. Efficiency is measured on Jetson Orin NX 16GB (MAXN, PyTorch eager, AMP FP16, batch size one, native input resolutions). Latency averages 300 forward passes after warm-up, excluding I/O and CPU preprocessing. Attention uses native PyTorch Scaled Dot-Product Attention (SDPA) with external xFormers [19] and flashAttn [10, 11] disabled for fairness and pytorch-native. Our model uses reparameterization [12]. DepthMaster and PaGeR did not complete this formal run due to OOM.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Params (M)</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="2">Matterport3D</td><td colspan="2">Stanford2D3D</td><td colspan="2">Metropolis</td></tr><tr><td>GFLOPs↓ Latency (ms)↓ AbsRel↓</td><td> $\delta _ { 1 }$ </td><td>↑AbsRel↓</td><td> $\delta _ { 1 }$ </td><td>↑AbsRel↓</td><td> $\delta _ { 1 }$  ↑</td></tr><tr><td colspan="10">Foundation models</td></tr><tr><td>DA360.  ${ \bf \nabla } \cdot { \bf S } ^ { \dagger }$  [18]</td><td>24.804</td><td>299.520</td><td>517.412</td><td>0.168</td><td>0.802</td><td>0.082</td><td>0.947</td><td></td><td>0.232 0.597</td></tr><tr><td>DAP [23]</td><td>365.050</td><td>2543.697</td><td>1868.322</td><td>0.137</td><td>0.840</td><td>0.072</td><td>0.963</td><td>0.195</td><td>0.734</td></tr><tr><td>DepthMaster [46]</td><td>385.191</td><td>11521.707</td><td>OOM</td><td>0.101</td><td>0.905</td><td>0.062</td><td>0.965</td><td>1.022</td><td>0.263</td></tr><tr><td>PaGeR [5]</td><td>1286.563</td><td>26200.188</td><td>00M</td><td>0.116</td><td>0.889</td><td>0.064</td><td>0.972</td><td>0.224</td><td>0.500</td></tr><tr><td colspan="10">Lightweight models</td></tr><tr><td>UniFuse† [17]</td><td>30.263</td><td>124.592</td><td>90.863</td><td>0.234 0.663</td><td></td><td></td><td>0.155 0.754</td><td>0.297</td><td>0.500</td></tr><tr><td>BiFuse++† [43]</td><td>32.272</td><td>106.598</td><td>71.284</td><td>0.295</td><td>0.555</td><td>0.2040.717</td><td></td><td>0.398</td><td>0.260</td></tr><tr><td>ACDNet† [62]</td><td>46.606</td><td>156.655</td><td>140.114</td><td>0.233</td><td>0.670</td><td>0.1460.795</td><td></td><td>0.301</td><td>0.487</td></tr><tr><td>EG360(ours)</td><td>6.593</td><td>23.492</td><td>28.509</td><td>0.173 0.784</td><td></td><td></td><td>0.094 0.923</td><td>0.237</td><td>0.632</td></tr></table>

Table 3. Comparison with existing method without ground-truth scale alignment. † denotes models retrained or finetuned with our recipe. Latency is measured on a Jetson Orin NX 16GB in MAXN mode, with AMP FP16 and batch size 1. EG360 uses structural reparameterization. OOM denotes out of memory. Bold denotes the best value in each column within each group.

## 4.2. Comparison with Existing Methods

Table 3 compares depth accuracy and inference cost, with latency measured on a Jetson Orin NX 16GB in MAXN mode. Compared with UniFuse, EG360 reduces AbsRel by 26.245%, 39.432%, and 20.297% on Matterport3D, Stanford2D3D, and Metropolis, respectively, and also improves $\delta _ { 1 }$ on all three datasets. With only 6.593M parameters and 23.492 GFLOPs, EG360 is also more efficient than previous lightweight models. Compared to DA360-S, EG360 achieves similar results on Matterport3D and Metropolis while running 18 times faster. DAP and PaGeR can achieve better results, but they require 2543.697 and 26200.188 GFLOPs of computation. As for DepthMaster, it achieves the lowest indoor AbsRel, but its high Metropolis error may be related to the absence of outdoor panoramas in training.

Qualitative comparisons Figure 4 shows qualitative results from indoor and outdoor scenarios. For example, our EG360 can obtain lower errors at the distorted door in the indoor scene and the far buildings in the outdoor scene than UniFuse. Additionally, EG360 can achieve similar results against DA360-S and DAP, as shown by the depth prediction and error maps.

## 4.3. Ablation Studies

Overall Abaltion Study Table 4 report the overall ablation study for APPD and SCFF. APPD alone reduces AbsRel from 0.199 to 0.169 on Matterport3D, from 0.126 to 0.101 on Stanford2D3D, and from 0.266 to 0.244 on Metropolis. SCFF alone also improves all six metrics over the baseline without using pseudo labels. Combining the two components yields EG360, which achieves the best overall results among test datasets. Especially, SCFF is more effective on Stanford2D3D, as it’s intended for global semantic enhancement, which could be more useful for indoor scenarios. In terms of efficiency, APPD leaves the inference architecture unchanged at 6.144M parameters and 21.196 GFLOPs while SCFF adds 0.449M parameters, 2.296 GFLOPs.

<table><tr><td></td><td></td><td></td><td colspan="2">Matterport3D</td><td colspan="2">Stanford2D3D</td><td colspan="2">Metropolis</td></tr><tr><td>Method</td><td>Params (M)</td><td>GFLOPs↓ AbsRel↓</td><td></td><td> $\delta _ { 1 }$ </td><td>↑ AbsRel↓</td><td></td><td>δ1 ↑ AbsRel↓</td><td> $\delta _ { 1 } \uparrow$ </td></tr><tr><td>Baseline</td><td>6.144</td><td>21.196</td><td>0.199</td><td>0.722</td><td>0.126</td><td>0.850</td><td>0.2660.578</td><td></td></tr><tr><td>+ APPD</td><td>6.144</td><td>21.196</td><td>0.169</td><td>0.781</td><td></td><td>0.101 0.903</td><td></td><td>0.2440.619</td></tr><tr><td>+ SCFF</td><td>6.593</td><td>23.492</td><td></td><td>0.195 0.733</td><td></td><td>0.113 0.876</td><td></td><td>0.263 0.582</td></tr><tr><td>EG360</td><td>6.593</td><td>23.492</td><td></td><td>0.173 0.784</td><td></td><td>0.094 0.923</td><td></td><td>0.2370.632</td></tr></table>

Table 4. Overall ablation of APPD and SCFF on three real-world test datasets. EG360 includes both components. Bold denotes the best value in each column.

Ablation on joint and single panorama-perspective distillation As discussed in Table 2, DAP and MoGe2 are complementary in terms of performance on different domains. Here, as listed in Table 5, simply averaging the losses from the two teachers yields more balanced results than using them alone. For example, on Matterport3D, simple averaging yields an AbsRel of 0.168 while using them alone leads to 0.176.

![](images/b88b8ae37e08179fe38493479976de8c00dae432384c11373aca51adbe8d1c19.jpg)  
Figure 4. Qualitative comparisons on Matterport3D (top) and Metropolis (bottom). The first column shows ground-truth depth and RGB images; the remaining columns show predictions above per-pixel AbsRel maps for UniFuse, EG360, DA360-S, and DAP. Error colors range from purple (low) to yellow (high). Pixels without ground truth depth are masked in the error map.

<table><tr><td></td><td>Matterport3D</td><td>Stanford2D3D</td><td>Metropolis</td><td></td></tr><tr><td>Distill Method</td><td>AbsRel↓ δ1↑</td><td>AbsRel↓  $\delta _ { 1 } \uparrow$ </td><td>AbsRel↓</td><td>δ1↑</td></tr><tr><td>No distill</td><td>0.199 0.722</td><td>0.126 0.850</td><td>0.266</td><td>0.578</td></tr><tr><td>DAP only</td><td>0.176 0.771</td><td>0.101 0.901</td><td>0.257</td><td>0.601</td></tr><tr><td>MoGe2 only</td><td>0.176 0.762</td><td>0.146 0.762</td><td>0.235</td><td>0.628</td></tr><tr><td>DAP + MoGe2 (Avg)</td><td>0.168 0.780</td><td>0.109 0.889</td><td></td><td>0.2380.630</td></tr></table>

Table 5. Teacher complementarity with our baseline student. Using both teachers leads to better results.

<table><tr><td></td><td>Matterport3D</td><td>Stanford2D3D</td><td colspan="2">Metropolis</td></tr><tr><td>Batch size</td><td>AbsRel↓  $\delta _ { 1 }$  ↑</td><td>AbsRel↓  $\delta _ { 1 }$ </td><td>↑ AbsRel↓</td><td> $\delta _ { 1 }$  ↑</td></tr><tr><td>1</td><td>0.184 0.758</td><td>0.115 0.875</td><td>0.240</td><td>0.630</td></tr><tr><td>2</td><td>0.175 0.769</td><td>0.114 0.872</td><td>0.242</td><td>0.613</td></tr><tr><td>4</td><td>0.173 0.778</td><td>0.110 0.883</td><td>0.244</td><td>0.618</td></tr><tr><td>8</td><td>0.168 0.780</td><td>0.109 0.889</td><td>0.238</td><td>0.630</td></tr><tr><td>16</td><td>0.166 0.782</td><td>0.108 0.890</td><td>0.239</td><td>0.628</td></tr></table>

Table 6. Effect of the batch size for unlabeled data under Avg distillation. The first column is the unlabeled batch size.

Ablation of batch size of unlabeled data In Table $^ { 6 , }$ we vary the unlabeled batch size from 1 to 16 under average distillation. Increasing it from 1 to 8 reduces Matterport3D AbsRel from 0.184 to 0.168 and Stanford2D3D AbsRel from 0.115 to 0.109. A batch size of 16 gives similar results against a batch size of 8 but would leads to more computation during training. Thus, we use 8 as the unlabeled batch size across our other experiments.

<table><tr><td></td><td>Matterport3D</td><td>Stanford2D3D</td><td>Metropolis</td></tr><tr><td>Distill Method AbsRel↓</td><td> $\delta _ { 1 } \uparrow$ </td><td>AbsRel↓  $\delta _ { 1 }$  ↑</td><td>AbsRel↓  $\delta _ { 1 }$  ←</td></tr><tr><td>Avg</td><td>0.168 0.780</td><td>0.109 0.889</td><td>0.238 0.630</td></tr><tr><td>Random</td><td>0.172 0.770</td><td>0.113 0.874</td><td>0.240 0.625</td></tr><tr><td>Manual</td><td>0.166 0.777</td><td>0.107 0.885</td><td>0.238 0.604</td></tr><tr><td>APPD</td><td>0.169 0.781</td><td>0.101 0.903</td><td>0.244 0.619</td></tr></table>

Table 7. Joint teacher strategies. Avg averages the two teacher losses. Random selects one teacher with equal probability per unlabeled batch. Manual assigns DAP to ZInD and MoGe2 to HoliCity. APPD is our geometry-adaptive loss weighting.

Ablation on how to combine teachers In Table 7, Avg averages both teacher losses at every iteration. Random [15] selects one teacher with equal probability for the entire unlabeled batch. Manual assigns DAP to ZInD and MoGe2 to HoliCity. APPD is our geometry-guided distillation, which reduces Stanford2D3D AbsRel by 7.411% relative to Avg and increases $\delta _ { 1 }$ from 0.889 to 0.903. It also attains the highest Matterport3D $\delta _ { 1 }$ . That is, APPD offers a useful accuracy trade-off across various domains.

Ablation on loss type for pseudo labels Table 8 compares three losses for pseudo labels while keeping both teachers fixed. SI normalizes each map by its dispersion as in DA360 [18], while LSS fits a scale by least squares before computing the L1 residual. L1 achieves the best result on five of the six metrics. We can see that with SI or LSS, the model would perform much worse on either indoor or outdoor datasets. This indicates that the training cannot recover the actual scale correctly, as these two losses are scale-invariant. Thus, we use L1 loss for unlabeled data.

<table><tr><td></td><td>Matterport3D</td><td>Stanford2D3D</td><td>Metropolis</td></tr><tr><td>Loss</td><td>AbsRel↓  $\delta _ { 1 }$  ↑</td><td>AbsRel↓  $\delta _ { 1 }$ </td><td>↑AbsRel↓  $\delta _ { 1 }$  ↑</td></tr><tr><td>L1</td><td>0.168 0.780</td><td>0.109 0.889</td><td>0.238 0.630</td></tr><tr><td>SI</td><td>0.228 0.680</td><td>0.155 0.805</td><td>0.246 0.603</td></tr><tr><td>LSS</td><td>0.176 0.748</td><td>0.111 0.901</td><td>0.466 0.095</td></tr></table>

Table 8. Pseudo-label loss type ablation under Avg teacher fusion.
<table><tr><td rowspan="2">Method</td><td>Matterport3D</td><td>Stanford2D3D</td><td>Metropolis</td></tr><tr><td>AbsRel↓  $\delta _ { 1 }$  ↑</td><td>AbsRel↓  $\delta _ { 1 }$  ↑</td><td>AbsRel↓  $\delta _ { 1 }$  ↑</td></tr><tr><td>UniFuse + APPD</td><td>0.234 0.663 0.197 0.745</td><td>0.155 0.754 0.119 0.867</td><td>0.297 0.500 0.250 0.603</td></tr><tr><td>BiFuse++ + APPD</td><td>0.295 0.555</td><td>0.204 0.717 0.158</td><td>0.398 0.260</td></tr><tr><td>ACDNet + APPD</td><td>0.211 0.718 0.233 0.670 0.190 0.753</td><td>0.760 0.146 0.795 0.114 0.873</td><td>0.270 0.546 0.301 0.487 0.253 0.600</td></tr></table>

Table 9. Applying APPD to other lightweight models.

![](images/b019e1f54638923fce959fad7e3c8e8e9d5d62b8cfcceaec0023c0bcfc0bc65e.jpg)  
(a) ERP RGB Image

![](images/df248b78d4d4236bc4ca54bd0a9cb086c148b388b1cb4628bd47b1ccea72bacb.jpg)  
(b) Our Baseline

![](images/86b56de754aedcc96e4bbf67acc615b01c798d0ac68d533e7b68ceb7abde6b27.jpg)  
(c) Our Baseline with APPD  
Figure 5. Qualitative results of applying APPD to our baseline model. The error is reduced in this real-world example.

Applying APPD to other architectures Table 9 reports results of applying APPD to UniFuse, BiFuse++, and ACD-Net. Clearly, all six reported metrics have been improved. For BiFuse++, its Matterport3D AbsRel decreases from 0.295 to 0.211, and its Metropolis $\delta _ { 1 }$ increases from 0.260 to 0.546. These results show that APPD can be applied to other models, either using ERP images alone [62] or using both the ERP and CP images [18].

Qualitative effect of APPD Figure 5 compares the results of applying APPD or not to our baseline. From the error maps, it is clear that APPD substantially reduces the high errors at the sofa and the door, which are challenging.

![](images/7c605aafb9cb108ef8d8490b54d7eaad8854e404e03f4d5d2b7b5b21f6ef9dad.jpg)  
(a) ERP RGB Image

![](images/d25911e82c3eea71cb31157b4e7072f85d62ea4127bf05bf13357371eceb523d.jpg)  
(b) Our Baseline

![](images/38233a49044568c5495edc2d9dd1346fd3f89793369c1ff09fa57df8d667a8b7.jpg)  
(c) Our Baseline with SCFF

Figure 6. Qualitative results of using SCFF. The error for the large shelf is lowered as SCFF allows long-range context modeling.
<table><tr><td>Configuration</td><td>Params (M)</td><td>GFLOPs↓ AbsRel↓</td><td> $\delta _ { 1 }$  ←</td></tr><tr><td>Baseline</td><td>6.144</td><td>21.196</td><td>0.126 0.850</td></tr><tr><td>Full</td><td>6.593</td><td>23.492</td><td>0.113 0.876</td></tr><tr><td>Full – D</td><td>6.593</td><td>23.492</td><td>0.125 0.864</td></tr><tr><td>Full – Cross</td><td>6.296</td><td>21.814</td><td>0.119 0.867</td></tr><tr><td>Full − 3 × 3 DW Conv</td><td>6.589</td><td>23.478</td><td>0.122 0.868</td></tr><tr><td> $\mathrm { F u l l - C r o s s + C 4 S e l f }$ </td><td>6.839</td><td>22.173</td><td>0.119 0.875</td></tr><tr><td> $\mathrm { F u l l - C r o s s + C 3 ~ S e l f }$ </td><td>6.494</td><td>26.914</td><td>0.118 0.874</td></tr></table>

Table 10. SCFF ablation on Stanford2D3D using labeled data only. Full is the complete SCFF module. − D removes the spherical direction bias, − Cross removes cross-level attention, and − 3 × 3 DW Conv removes the depthwise convolution in the panoramic FFN. Full − Cross + C4 Self/C3 Self replaces the cross attention with a self-attention block on C4/C3.

Ablation on components of SCFF Table 10 evaluates SCFF on Stanford2D3D under supervised training, without pseudo labels. Full reduces AbsRel from 0.126 to 0.113 and increases $\delta _ { 1 }$ from 0.850 to 0.876, adding 2.296 GFLOPs. Removing direction bias D, cross-level attention, or the panoramic 3 × 3 depthwise convolution degrades accuracy. Self-attention on C4 or C3 also yields lower accuracy than our cross-level attention.

Qualitative effect of using SCFF Figure 6 shows reduced errors over the highlighted shelf surfaces after adding SCFF. This example thus depicts the benefit of global feature fusion of the model with spherical geometry injected.

## 5. Conclusion

In this work, we presented EG360, an efficient and generalizable panorama depth estimation model built on top of our Adaptive Panorama–Perspective Distillation technique and Spherical Cross-Level Feature Fusion module. Experiments demonstrate that our model can achieve strong generalization ability while enabling real-time inference on a Jetson Orin NX 16GB in MAXN mode.

Limitations and future work. Although EG360 can achieve competitive results, we only used limited unlabeled data, and scaling the unlabeled data may further enhance performance [41, 51]. Additionally, as a single-frame model, it suffers from temporal incoherence. Extending

EG360 to panoramic video depth estimation [8, 36] via cross-frame fusions may reduce depth-flickering effects.

## References

[1] Hao Ai, Zidong Cao, Yan-Pei Cao, Ying Shan, and Lin Wang. Hrdfuse: Monocular 360 depth estimation by collaboratively learning holistic-with-regional depth distributions. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13273–13282. IEEE, 2023. 2

[2] Iro Armeni, Sasha Sax, Amir R Zamir, and Silvio Savarese. Joint 2d-3d-semantic data for indoor scene understanding. arXiv preprint arXiv:1702.01105, 2017. 1, 3, 5

[3] Shivam Asija, Edward Du, Nam Nguyen, Stefanie Zollmann, and Jonathan Ventura. 3d pano inpainting: Building a vr environment from a single input panorama. In 2024 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW), pages 1019–1020. IEEE, 2024. 1

[4] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016. 4

[5] Vukasin Bozic, Isidora Slavkovic, Dominik Narnhofer, Nando Metzger, Denis Rozumny, Konrad Schindler, and Nikolai Kalischek. Unified panoramic geometry estimation via multi-view foundation models. arXiv preprint arXiv:2605.26368, 2026. 1, 2, 6

[6] Zidong Cao, Jinjing Zhu, Weiming Zhang, Hao Ai, Haotian Bai, Hengshuang Zhao, and Lin Wang. Panda: Towards panoramic depth anything with unlabeled panoramas and mobius spatial augmentation. In¨ 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 982–992. IEEE, 2025. 2, 3

[7] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158, 2017. 1, 5

[8] Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, and Bingyi Kang. Video depth anything: Consistent depth estimation for super-long videos. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22831–22840. IEEE, 2025. 9

[9] Steve Cruz, Will Hutchcroft, Yuguang Li, Naji Khosravan, Ivaylo Boyadzhiev, and Sing Bing Kang. Zillow indoor dataset: Annotated floor plans with 360° panoramas and 3d room layouts. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2133–2143. IEEE, 2021. 2, 3

[10] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024. 5

[11] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re. FlashAttention: Fast and memory-efficient ´ exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 5

[12] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style

convnets great again. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13728–13737. IEEE, 2021. 2, 6

[13] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 1

[14] Krzysztof M Gorski, Eric Hivon, Anthony J Banday, Benjamin D Wandelt, Frode K Hansen, Mstvos Reinecke, and Matthia Bartelmann. Healpix: A framework for highresolution discretization and fast analysis of data distributed on the sphere. The Astrophysical Journal, 622(2):759–771, 2005. 2

[15] Xiankang He, Dongyan Guo, Hongji Li, Ruibo Li, Ying Cui, and Chi Zhang. Distill any depth: Distillation creates a stronger monocular depth estimator, 2025. 7

[16] Yuhang He, Wentao Yu, Jie Han, Xing Wei, Xiaopeng Hong, and Yihong Gong. Know your surroundings: Panoramic multi-object tracking by multimodality collaboration. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 2963–2974. IEEE, 2021. 1

[17] Hualie Jiang, Zhe Sheng, Siyu Zhu, Zilong Dong, and Rui Huang. Unifuse: Unidirectional fusion for 360 panorama depth estimation. IEEE Robotics and Automation Letters, 6 (2):1519–1526, 2021. 1, 2, 6

[18] Hualie Jiang, Ziyang Song, Zhiqiang Lou, Rui Xu, and Minglang Tan. Depth anything in 360<sup>◦</sup>: Towards scale invariance in the wild. arXiv preprint arXiv:2512.22819, 2025. 2, 3, 5, 6, 7, 8

[19] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. xformers: A modular and hackable transformer modelling library. https://github.com/ facebookresearch/xformers, 2022. 5

[20] Haodong Li, Wangguangdong Zheng, Jing He, Yuhao Liu, Xin Lin, Xin Yang, Ying-Cong Chen, and Chunchao Guo. DA<sup>2</sup>: Depth anything in any direction. arXiv preprint arXiv:2509.26618, 2025. 2

[21] Meng Li, Senbo Wang, Weihao Yuan, Weichao Shen, Zhe Sheng, and Zilong Dong. S<sup>2</sup>Net: Accurate Panorama Depth Estimation on Spherical Surface. IEEE Robotics and Automation Letters, 8(2):1053–1060, 2023. 2

[22] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 2

[23] Xin Lin, Meixi Song, Dizhe Zhang, Wenxuan Lu, Haodong Li, Bo Du, Ming-Hsuan Yang, Truong Nguyen, and Lu Qi. Depth any panoramas: A foundation model for panoramic depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26835–26844, 2026. 1, 2, 3, 6

[24] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5

[25] Kai Luo, Hao Shi, Sheng Wu, Fei Teng, Mengfei Duan, Chang Huang, Yuhang Wang, Kaiwei Wang, and Kailun

Yang. Omnidirectional multi-object tracking. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21959–21969. IEEE, 2025. 1

[26] Mapillary. Metropolis Dataset, 2024. Accessed: 2025-11- 01. 1, 3, 5

[27] Chao Ning and Hongping Gan. Trap attention: Monocular depth estimation with manual traps. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5033–5043. IEEE, 2023. 4

[28] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1

[29] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zach DeVito, Martin Raison,¨ Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library, 2019. 5

[30] Gregoire Payen de La Garanderie, Amir At-´ apour Abarghouei, and Toby P Breckon. Eliminating the blind spot: adapting 3d object detection and monocular depth estimation to 360 panoramic imagery. In European Conference on Computer Vision, pages 812–830. Springer, 2018. 1

[31] Rene Ranftl, Katrin Lasinger, David Hafner, Konrad´ Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3):1623–1637, 2020. 2

[32] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-´ sion transformers for dense prediction. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 12159–12168. IEEE, 2021. 2

[33] Zhijie Shen, Chunyu Lin, Kang Liao, Lang Nie, Zishuo Zheng, and Yao Zhao. Panoformer: panorama transformer for indoor 360<sup>◦</sup> depth estimation. In European Conference on Computer Vision, pages 195–211. Springer, 2022. 2

[34] Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 1

[35] Leslie N Smith and Nicholay Topin. Super-convergence: Very fast training of neural networks using large learning rates. In Artificial intelligence and machine learning for multi-domain operations applications, pages 369–386. SPIE, 2019. 5

[36] Chuanxin Song and Peixi Peng. Pvdepth: Panoramic video depth estimation via geometry-aware spatiotemporal adaptation. In Forty-third International Conference on Machine Learning. 2, 3, 9

[37] Tencent. HunyuanWorld 1.0: Generating immersive, explorable, and interactive 3D worlds from words or pixels. https : / / 3d . hunyuan . tencent . com / sceneTo3D, 2025. 2

[38] Fei Teng, Kai Luo, Sheng Wu, Siyu Li, Pujun Guo, Jiale Wei, Jiaming Zhang, Kunyu Peng, and Kailun Yang. Hallucinating 360: Panoramic street-view generation via local scenes diffusion and probabilistic prompting. In 2026 IEEE International Conference on Robotics and Automation (ICRA), pages 558–565. IEEE, 2026. 1

[39] Fabio Tosi, Luca Bartolomei, Matteo Poggi, and Stefano Mattoccia. Zipdepth: Bringing lightweight zero-shot monoc ular depth anywhere, on any device. In European Conference on Computer Vision, pages 668–690. Springer, 2026. 3

[40] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2

[41] Matthew Wallingford, Anand Bhattad, Aditya Kusupati, Vivek Ramanujan, Matt Deitke, Sham Kakade, Aniruddha Kembhavi, Roozbeh Mottaghi, Wei-Chiu Ma, and Ali Farhadi. From an image to a scene: Learning to imagine the world from a million 360 videos. Advances in Neural Information Processing Systems, 37:17743–17760, 2024. 8

[42] Fu-En Wang, Yu-Hsuan Yeh, Min Sun, Wei-Chen Chiu, and Yi-Hsuan Tsai. Bifuse: Monocular 360 depth estimation via bi-projection fusion. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 459–468. IEEE, 2020. 1, 2

[43] Fu-En Wang, Yu-Hsuan Yeh, Yi-Hsuan Tsai, Wei-Chen Chiu, and Min Sun. Bifuse++: Self-supervised and efficient bi-projection fusion for 360 depth estimation. IEEE transactions on pattern analysis and machine intelligence, 45(5): 5448–5460, 2022. 2, 6

[44] Guangcong Wang, Peng Wang, Zhaoxi Chen, Wenping Wang, Chen Change Loy, and Ziwei Liu. Perf: Panoramic neural radiance field from a single panorama. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(10): 6905–6918, 2024. 1

[45] Ning-Hsu Wang and Yu-Lun Liu. Depth anywhere: Enhancing 360 monocular depth estimation via perspective distillation and unlabeled data augmentation. Advances in Neural Information Processing Systems, 37:127739–127764, 2024. 2, 3

[46] Pengfei Wang, Shihao Wang, Liyi Chen, Zhiyuan Ma, Guowen Zhang, and Lei Zhang. Depthmaster: Unified monocular depth estimation for perspective and panoramic images. arXiv preprint arXiv:2606.12368, 2026. 1, 2, 6

[47] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. Advances in Neural Information Pro cessing Systems, 38:35928–35959, 2026. 2, 3

[48] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Se bastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020. 2, 3

[49] Diana Wofk, Fangchang Ma, Tien-Ju Yang, Sertac Karaman, and Vivienne Sze. Fastdepth: Fast monocular depth esti-

mation on embedded systems. In 2019 International Conference on Robotics and Automation (ICRA), pages 6101–6108. IEEE, 2019. 3

[50] Yinzhe Xu, Huajian Huang, Yingshu Chen, and Sai-Kit Yeung. 360vots: Visual object tracking and segmentation in omnidirectional videos. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(11):9785–9797, 2025. 1

[51] Feng Xue, Wu Chen, Mingshuai Zhao, Guofeng Zhong, Anlong Ming, Haozhe Wang, Dianqiao Lei, Zhaowen Lin, Haiyang Zhang, and Nicu Sebe. Depthart: Scaling foundation monocular depth to tiny models. arXiv preprint arXiv:2607.17099, 2026. 8

[52] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10371–10381. IEEE, 2024. 2

[53] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in neural information processing systems, 37:21875–21911, 2024. 2

[54] Ilwi Yun, Chanyong Shin, Hyunku Lee, Hyuk-Jae Lee, and Chae Eun Rhee. Egformer: Equirectangular geometrybiased transformer for 360 depth estimation. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 6078–6089. IEEE, 2023. 2

[55] Jiaming Zhang, Kailun Yang, Chaoxiang Ma, Simon Reiß, Kunyu Peng, and Rainer Stiefelhagen. Bending reality: Distortion-aware transformers for adapting to panoramic semantic segmentation. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16896–16906. IEEE, 2022. 1

[56] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3813–3824. IEEE, 2023. 5

[57] Ning Zhang, Francesco Nex, George Vosselman, and Norman Kerle. Lite-mono: A lightweight cnn and transformer architecture for self-supervised monocular depth estimation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18537–18546. IEEE, 2023. 4

[58] Weiming Zhang, Yexin Liu, Xu Zheng, and Lin Wang. Goodsam: Bridging domain and capacity gaps via segment anything model for distortion-aware panoramic semantic segmentation. arXiv preprint arXiv:2403.16370, 2024. 1

[59] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3d: A large photo-realistic dataset for structured 3d modeling. In European Conference on Computer Vision, pages 519–535. Springer, 2020. 2, 3

[60] Ding Zhong, Xu Zheng, Chenfei Liao, Yuanhuiyi Lyu, Jialei Chen, Shengyang Wu, Linfeng Zhang, and Xuming Hu. Omnisam: Omnidirectional segment anything model for uda in panoramic semantic segmentation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 23892–23901. IEEE, 2025. 1

[61] Yichao Zhou, Jingwei Huang, Xili Dai, Shichen Liu, Linjie Luo, Zhili Chen, and Yi Ma. Holicity: A city-scale data platform for learning holistic 3d structures. arXiv preprint arXiv:2008.03286, 2020. 2, 3

[62] Chuanqing Zhuang, Zhengda Lu, Yiqun Wang, Jun Xiao, and Ying Wang. Acdnet: Adaptively combined dilated con volution for monocular panorama depth estimation. In Pro ceedings of the AAAI conference on artificial intelligence, pages 3653–3661, 2022. 1, 2, 6, 8

[63] Nikolaos Zioulis, Antonis Karakottas, Dimitrios Zarpalas, and Petros Daras. Omnidepth: Dense depth estimation for indoors spherical panoramas. In European conference on computer vision, pages 453–471. Springer, 2018. 2