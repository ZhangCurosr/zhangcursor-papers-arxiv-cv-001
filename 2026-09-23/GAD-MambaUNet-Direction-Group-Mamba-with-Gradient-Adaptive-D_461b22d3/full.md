# GAD-MambaUNet: Direction-Group Mamba with Gradient-Adaptive DINOv3 Distillation for Lightweight Medical Image Segmentation

Fang Wang<sup>a,1</sup>, Huitao Li<sup>a,1</sup>, Wenhan Chao<sup>b</sup>, Zheng Zhuo<sup>a,∗</sup> and Xinxin Yang<sup>a</sup>

<sup>a</sup>College of Artificial Intelligence, Beijing Institute of Petrochemical Technology, Beijing, 102617, People’s Republic of China

<sup>b</sup>School of Computer Science and Engineering, Beihang University, Beijing, 100191, 100083, People’s Republic of China

## A R T I C L E I N F O

Keywords:   
medical image segmentation   
lightweight networks   
state-space models   
DINOv3   
knowledge distillation

## A BS T RA C T

Lightweight medical image segmentation demands both accurate boundary delineation and eficient inference, posing a fundamental challenge for models deployed in resource-constrained clinical environments. In this paper, we propose GAD-MambaUNet, an asymmetric local–global segmentation network that combines lightweight convolutional modeling with eficient state-space contextual aggregation. Our architecture employs lightweight convolutional blocks in shallow stages to perceive local features, and introduces Direction-Group Graph Selective Scan (DG-GSS) blocks in deeper stages for eficient long-range modeling. Within each DG-GSS block, direction–group feature responses are treated as graph nodes, and structured message passing is performed prior to multi-directional fusion, enabling cross-direction and cross-subspace information interaction. To further enhance semantic representation without increasing inference cost, we incorporate a frozen pretrained DINOv3 model as a semantic teacher during training only, transferring its semantic priors to the compact student network via Gradient-Adaptive Distillation. Extensive experiments on multiple public medical image segmentation benchmarks demonstrate that GAD-MambaUNet achieves superior segmentation accuracy with favorable inference eficiency.

## 1. Introduction

Accurate medical image segmentation is a cornerstone of computer-aided diagnosis, treatment planning, and quantitative disease assessment. Given a medical image, the objective is to assign a pixel-level label to each region of interest, with particular emphasis on preserving fine anatomical boundaries. This task is inherently challenging: lesions and organs often appear small, irregular, poorly contrasted, or visually similar to surrounding tissues. A robust segmentation model must therefore integrate fine-grained local evidence—essential for boundary delineation—with suficient global contextual information to resolve foreground– background ambiguities.

The U-Net architecture [1] established the dominant encoder–decoder paradigm, using a contracting path to extract hierarchical semantic features and symmetric skip connections to preserve spatial resolution for precise reconstruction. Subsequent CNN-based extensions have improved feature fusion, attention mechanisms, and multiscale decoding [2–4]. Despite these advances, convolutional networks remain inherently constrained by local receptive fields, which limits their ability to model long-range dependencies—a critical capability when boundary interpretation relies on global lesion shape, continuity cues, or contextual information.

Transformer-based architectures, such as TransUNet [4], address this limitation by introducing self-attention for global feature aggregation. This design substantially enlarges the receptive field and improves contextual reasoning. However, global attention and large-scale pretrained backbones come at a cost: increased parameter counts, higher memory consumption, and slower inference, which hinder deployment in resource-limited clinical or edgedevice settings. As medical AI shifts from centralized laboratory analysis to point-of-care and edge applications, segmentation accuracy alone no longer sufices. Model eficiency—measured by parameter count, FLOPs, and inference latency—has become equally important.

This has motivated a series of lightweight segmentation networks based on depth-wise convolution [5], compact channel designs [6], and eficient token mixing [7–9]. Among emerging techniques, visual state-space models, particularly Mamba [10] and VMamba [11], ofer a compelling alternative: they achieve global receptive fields with linear sequence complexity, avoiding the quadratic cost of selfattention. VM-UNet [12] further incorporates visual statespace blocks into a U-shaped segmentation backbone.

To further reduce computational cost, grouped Mamba [13] partitions feature channels into parallel groups and performs multi-directional selective scanning across each group independently. While this strategy efectively lowers FLOPs, it introduces a fundamental limitation: diferent direction– group branches are processed in isolation and merged only after scanning. This independent design restricts the exchange of complementary contextual cues across scanning directions and channel subspaces, potentially weakening the model’s capacity to capture holistic scene understanding. To address this limitation, we propose Direction-Group Graph Selective Scan (DG-GSS), which models direction–group feature responses as graph nodes and enables structured message passing before multi-directional fusion, thereby facilitating cross-direction and cross-subspace interaction.

![](images/2477a0c6c3ea59b849308d7b8d907e1972cbc4797aadbdaf59c6a4d25b222df9.jpg)

![](images/eabae390ee2772291de98cf5814680bf7793e8ccc1333cdaf874e09ae84b8149.jpg)  
Figure 1: Accuracy–eficiency comparison across PH<sup>2</sup>, ISIC2018, CVC-ClinicDB, and CVC-ColonDB. The vertical axis reports the macro-average Dice score over the four datasets, while the horizontal axes show student-side parameters and FLOPs. Compared with existing lightweight segmentation networks, GAD-MambaUNet achieves a more favorable Pareto balance between segmentation accuracy and deployment cost. The DINOv3 teacher is used only during training and is excluded from deployment cost.

Training lightweight segmentation models on limited medical datasets often leads to insuficient semantic abstraction. While larger vision foundation models, such as DINOv3 [14], provide rich semantic priors, they are too heavy for deployment. Feature distillation ofers a practical middle ground: a frozen teacher provides dense supervision during training, and is discarded at inference. However, a fixed distillation coeficient cannot adapt to the evolving optimization state of the student, potentially under- or over-regularizing the segmentation objective. To overcome this, we adapt Gradient-Adaptive Distillation (GAD), which dynamically adjusts the distillation strength based on the gradient contribution of the aligned decoder block, ensuring balanced semantic transfer without compromising the primary task.

Based on the above analysis, we propose GAD-MambaUNet, an asymmetric local–global lightweight segmentation network for medical image segmentation. Our architecture employs lightweight convolutional blocks in shallow stages to retain fine-grained boundary details, and introduces DG-GSS blocks in deeper stages for eficient long-range features. Within each DG-GSS block, direction–group feature responses are treated as graph nodes, and structured message passing is performed prior to multi-directional fusion, enabling cross-direction and cross-subspace information interaction. We further incorporate a frozen pretrained DINOv3 model as a semantic teacher during training only, transferring its semantic priors to the compact student network via Gradient-Adaptive Distillation.

In summary, the main contributions of this work can be categorized into four aspects:

• We propose DG-GSS, a novel graph-enhanced selective scan mechanism that models direction–group feature responses as graph nodes and performs structured message passing before multi-directional fusion, improving contextual interaction without increasing inference cost.

• We introduce Gradient-Adaptive Distillation, which dynamically regulates the distillation coeficient according to the gradient share of the aligned decoder block during training, and demonstrate its efectiveness in transferring DINOv3 semantic priors to a compact student without incurring inference overhead.

• We design GAD-MambaUNet, an asymmetric local– global lightweight segmentation network that combines convolutional local modeling in shallow stages with DG-GSS-based long-range modeling in deep stages, achieving a favorable accuracy–eficiency balance.

• We conduct extensive experiments on PH<sup>2</sup>, ISIC2018, CVC-ClinicDB, and CVC-ColonDB, showing that GAD-MambaUNet achieves a superior accuracy– eficiency balance in terms of Dice score, parameters, and FLOPs. As shown in Fig. 1, the proposed models obtain more favorable performance, while DINOv3 supervision improves accuracy without increasing inference cost.

## 2. Related Work

## 2.1. Medical Segmentation and Lightweight U-Shaped Networks

U-Net [1] and its encoder–decoder variants remain a primary template for medical image segmentation because skip connections combine deep semantic features with spatially precise reconstruction. This design is particularly suitable for medical segmentation, where the model must simultaneously capture high-level anatomical or lesion semantics and recover fine boundary details. Task-specific methods strengthen this template through mechanisms such as reverse attention in PraNet [2] and uncertainty-aware context attention in UACANet [3]. TransUNet [4] instead introduces a Transformer encoder to enlarge the receptive field through global self-attention. These approaches establish the value of contextual reasoning for resolving ambiguous foreground– background regions, but their attention modules or pretrained encoders can be costly when the deployment budget is limited.

Lightweight U-shaped networks reduce this cost through compact channel schedules and eficient local operators. UNeXt [9] combines convolution with shifted MLP blocks, CMUNeXt [6] uses large kernels and skip fusion, EGE-UNet [7] develops group-enhanced attention, and EM-CAD [8] employs eficient multi-scale convolutional decoding. MK-UNet [5] further uses parallel multi-kernel depthwise convolutions, channel shufle, and lightweight decoder attention to capture multi-scale boundary evidence with compact cost. These lightweight CNN-based designs are efective for preserving local details and improving deployment eficiency. However, they still mainly rely on local or convolution-dominated operators and may lack explicit long-range contextual modeling when target boundaries are incomplete, lesion regions are ambiguous, or distant anatomical structures share similar textures. This limitation motivates the use of more eficient global modeling mechanisms in lightweight segmentation networks.

## 2.2. Visual State-Space Models

State-space models ofer an eficient alternative to selfattention for long-range feature modeling. Mamba [10] uses input-dependent state transitions with linear sequence complexity, while VMamba [11] extends selective scanning to visual feature maps through multiple two-dimensional traversal directions. By replacing quadratic global attention with selective scanning, visual state-space models provide a favorable trade-of between global context modeling and computational eficiency. VM-UNet [12] incorporates visual state-space blocks into a U-shaped segmentation architecture, showing the potential of state-space modeling for medical image segmentation.

For lightweight medical segmentation, UltraLight VM-UNet [13] further reduces Vision Mamba cost by partitioning deep features into parallel channel groups and processing them with parallel Vision Mamba branches. This grouped design is attractive because global modeling is introduced only in a compact and computationally eficient form. However, in grouped multi-directional scans, direction–group branches are typically processed independently and merged only after selective scanning. Although this design reduces computation, it does not explicitly model the structural relationship between traversal directions and channel subspaces.

As a result, complementary directional cues and groupwise semantic responses cannot be exchanged before fusion.

Diferent traversal directions may capture diferent aspects of spatial context, while diferent channel groups may encode distinct but correlated semantic patterns. Processing them in isolation can therefore weaken the model’s ability to form a holistic contextual representation. This limitation motivates our Direction-Group Graph Selective Scan (DG-GSS), which introduces structured interaction among direction–group responses while preserving the eficiency of grouped selective scanning.

## 2.3. Foundation-Model Distillation for Lightweight Segmentation

Strong Transformer encoders and vision foundation models provide rich semantic representations for medical image segmentation. Swin Transformer [15] and its medical segmentation variant Swin-Unet [16] model hierarchical visual features, while DINOv3 [14] provides dense features through large-scale self-supervised pretraining. Such models can provide strong high-level semantic priors, which are useful for distinguishing ambiguous lesion regions from visually similar background tissues. However, directly deploying these large encoders conflicts with the parameter, memory, and latency budgets of lightweight systems.

Feature-level distillation ofers a practical alternative: a frozen teacher provides dense supervision only during training, allowing a compact student to inherit semantic context without the teacher’s deployment cost [17]. This strategy is especially suitable for lightweight segmentation, where the student must remain eficient during inference but can still benefit from stronger semantic guidance during optimization. Most distillation settings use a fixed distillation coeficient. However, a fixed coeficient cannot adapt to the dynamic state of optimization: as training progresses, the gradient contribution of the teacher-supervised deep student feature to the overall network can change, so the same coeficient may yield insuficient teacher supervision or impose an overly strong constraint on the primary task objective.

To address this dynamic balancing problem, RT-DETRv4 [18] proposes a gradient-aware distillation strategy for real-time detection. In its reported configuration, a frozen DINOv3- ViT-B teacher supervises deep detector features through a Deep Semantic Injector, and Gradient-guided Adaptive Modulation adjusts the semantic-transfer strength using gradient norm ratios. Inspired by this strategy, our work adapts DINOv3-based gradient-adaptive distillation to lightweight medical image segmentation. Specifically, teacher features are aligned with a deep decoder representation of the student, and the distillation strength is dynamically regulated according to the gradient share of the aligned decoder block. During inference, the teacher and projection head are discarded, so the semantic supervision improves the compact student without increasing deployment cost.

## 3. Methodology

We describe the proposed GAD-MambaUNet by first introducing its core building block and then integrating it into an asymmetric U-shaped segmentation architecture. As shown in Fig. 2, the left part presents the overall student network and the training-time DINOv3 teacher, while the right part details the proposed block-level design and the internal structure of Direction-Group Graph Selective Scan (DG-GSS). We first present DG-GSS, which enables structured interaction among scan directions and channel groups. Then, we describe how DG-GSS is embedded into the overall GAD-MambaUNet architecture. Finally, we introduce the training-time DINOv3 supervision with Gradient-Adaptive Distillation (GAD), where the teacher model is used only during training and removed during inference.

![](images/6a94d2a4845ea3091d3664da7bf02c450275a24d3726224b4ede431b96b71b74.jpg)  
Figure 2: Overall architecture of GAD-MambaUNet. High-resolution stages retain multi-kernel local modeling, whereas deep stages use grouped state-space blocks with DG-GSS. During training, a frozen DINOv3 teacher supervises the first decoder block and GAD regulates the distillation coeficient according to its gradient share. The teacher and projection head are removed at inference.

## 3.1. Direction-Group Graph Selective Scan

As illustrated in the lower-right part of Fig. 2, DG-GSS is designed to enhance grouped multi-directional selective scanning by introducing graph-based interaction among direction–group responses. DG-GSS regards each scandirection and channel-group response as a graph node and performs structured message passing before the final multidirectional fusion.

Let $\mathbf { X } \in \mathbb { R } ^ { B \times D _ { i } \times h \times w }$ denote the inner state-space feature of DG-GSS, where $D _ { i }$ is the inner channel dimension. We adapt four Cross2D traversal directions and partition the inner channels into � groups. With $d = D _ { i } / M$ and $L =$ ℎ�, cross scanning produces

$$
\begin{array} { r } { \mathbf { X } _ { \mathrm { s c a n } } \in \mathbb { R } ^ { B \times K \times M \times d \times L } , \qquad K = 4 . } \end{array}\tag{1}
$$

For each direction–group pair, the selective-scan parameters (Δ, �, �) are generated by independent grouped projections, and selective scanning is applied along the

corresponding sequence. The directional outputs are then realigned to the original spatial coordinate system:

$$
\mathbf { Y } \in \mathbb { R } ^ { B \times K \times M \times d \times h \times w } .\tag{2}
$$

Each tensor $\mathbf { Y } _ { k , m }$ corresponds to the response of scan direction � and channel group �. We regard every direction– group response as a graph node and obtain its node descriptor by global average pooling:

$$
\mathbf { r } _ { k , m } = \frac { 1 } { h w } \sum _ { u = 1 } ^ { h } \sum _ { v = 1 } ^ { w } \mathbf { Y } _ { k , m , : , u , v } .\tag{3}
$$

After layer normalization and linear projection, the node descriptor is transformed into $\mathbf { h } _ { k , m } .$

In our implementation, we adopt a factorized directiongroup graph, where two nodes are connected if they share the same scan direction or the same channel group, and selfconnections are included. For two connected nodes � and �, additive graph-attention logits [19] are computed as

$$
\begin{array} { r } { \boldsymbol { e } _ { i j } = \mathrm { L e a k y R e L U } \left( \mathbf { a } _ { s } ^ { \top } \mathbf { h } _ { i } + \mathbf { a } _ { t } ^ { \top } \mathbf { h } _ { j } \right) . } \end{array}\tag{4}
$$

The attention coeficients are obtained by masked softmax over the neighborhood:

$$
\alpha _ { i j } = \frac { \exp ( e _ { i j } ) } { \sum _ { j ^ { \prime } \in \mathcal { N } ( i ) } \exp ( e _ { i j ^ { \prime } } ) } , \qquad j \in \mathcal { N } ( i ) .\tag{5}
$$

Graph-based message passing is then performed on the full spatial feature maps:

$$
\widetilde { \mathbf { Y } } _ { i } = \mathbf { Y } _ { i } + \gamma \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } V ( \mathbf { Y } _ { j } ) ,\tag{6}
$$

where $V ( \cdot )$ is a shared $1 \times 1$ value projection and $\gamma$ is a learnable scaling parameter initialized to zero. This initialization makes DG-GSS start from the original grouped selective-scan behavior and progressively learn direction– group information exchange during training.

Finally, the enhanced group responses are concatenated along the channel dimension for each direction, and the four aligned directional feature maps are summed:

$$
\mathbf { Z } = \sum _ { k = 1 } ^ { K } \mathrm { C o n c a t } _ { m = 1 } ^ { M } \left( \widetilde { \mathbf { Y } } _ { k , m } \right) .\tag{7}
$$

Output normalization is then applied to obtain the DG-GSS output. In this way, DG-GSS preserves the eficiency of grouped multi-directional selective scanning while explicitly modeling the structural relationship between scan directions and channel groups.

## 3.2. GAD-MambaUNet Architecture

After defining DG-GSS, we embed it into a lightweight block and integrate the block into an asymmetric U-shaped segmentation network. As shown in the upper-right part of Fig. 2, the DG-GSS block consists of three residual components: a pre-scan local mixing stage, the proposed DG-GSS layer, and a post-scan local refinement stage. Given an input feature �, the block is formulated as

$$
\begin{array} { r l } & { \mathbf { u } _ { 1 } = \mathbf { x } + \mathrm { F F N } _ { 0 } ( \mathrm { D W } _ { 0 } ( \mathbf { x } ) ) , } \\ & { \mathbf { u } _ { 2 } = \mathbf { u } _ { 1 } + \mathrm { D G } \mathrm { - G S S } ( \mathbf { u } _ { 1 } ) , } \\ & { \mathbf { y } = \mathbf { u } _ { 2 } + \mathrm { F F N } _ { 1 } ( \mathrm { D W } _ { 1 } ( \mathbf { u } _ { 2 } ) ) , } \end{array}\tag{8}
$$

where DW(⋅) denotes depth-wise convolution for local mixing and FFN(⋅) denotes a lightweight feed-forward refinement module. The local operators enhance short-range texture and channel interactions, while DG-GSS provides efficient long-range context aggregation through direction– group graph interaction.

The overall GAD-MambaUNet follows a five-level encoder– decoder architecture, as shown in the left part of Fig. 2. Given an input image $\textbf { x } ~ \in ~ \mathbb { R } ^ { 3 \times H \times W }$ , the network predicts a binary segmentation logit map $\textbf { z } \in \mathbb { R } ^ { 1 \times H \overset { . } { \times } W }$ In the base configuration, the channel widths are set to [16, 32, 64, 96, 160]. To balance local boundary preservation and global context modeling, the first three encoder stages and the last three decoder stages use lightweight multikernel inverted residual (MKIR) blocks inherited from MK-UNet [5], while the two deepest encoder stages and the first two decoder stages use DG-GSS blocks. Resolution and channel changes are performed outside these blocks through depth-wise downsampling with point-wise projection or point-wise projection followed by bilinear upsampling, keeping the feature width fixed inside each block.

For decoder reconstruction, skip connections transfer high-resolution encoder features to the corresponding decoder stages. Before fusion, a grouped attention gate filters the skip feature according to the decoder context, reducing irrelevant background responses. Let � denote the upsampled decoder feature and � the encoder feature at the same resolution. The gated skip feature is computed as

$$
\alpha = \sigma \left( \psi \left( \phi ( W _ { g } { \bf g } + W _ { s } { \bf s } ) \right) \right) , \qquad { \hat { \bf s } } = \alpha \odot { \bf s } .\tag{9}
$$

where $W _ { g }$ and $W _ { s }$ are grouped convolutions, � is ReLU, and � maps the joint response to a spatial gate. The filtered skip feature �̂ is then fused with the decoder feature. The first decoder block operates at the deepest decoder level and provides the aligned student feature for the training-time DINOv3 supervision described in the next subsection.

## 3.3. Training-Time DINOv3 Supervision with GAD

Although DG-GSS improves contextual modeling inside the compact student network, training a lightweight segmentation model on limited medical datasets may still lead to insuficient semantic abstraction. Inspired by the DINOv3-based gradient-adaptive distillation strategy in RT-DETRv4 [18], we adapt training-time foundation-model supervision to lightweight medical image segmentation. A frozen pretrained DINOv3 model [14] is used as the semantic teacher only during training, and the teacher branch is removed during inference.

We adopt a capacity-matched teacher assignment strategy for diferent student scales. Specifically, DINOv3 ViT-T/16 is used for the Tiny, Small, and Base variants, while DI-NOv3 ViT-B/16 is used for the Medium and Large variants. For compact students, the smaller teacher provides moderate semantic guidance while avoiding an excessive teacher– student capacity gap and unnecessary training overhead. For larger students, the stronger ViT-B/16 teacher provides richer high-level semantic supervision, which can be better absorbed by higher-capacity decoder representations. Since the DINOv3 teacher is discarded during inference, this assignment afects only the training-time supervision strength and does not change the student-side deployment cost.

As described in the previous subsection, the student feature used for distillation is extracted after the first decoder block and before the first upsampling operation. This feature is semantically rich, spatially compact, and computationally inexpensive for feature alignment. Let ${ \bf { F } } _ { s }$ denote the projected student feature, and let $\mathbf { F } _ { t }$ denote the corresponding DINOv3 teacher feature. The student feature is projected to the teacher dimension by a 1 × 1 convolution followed by batch normalization when their channel dimensions are diferent.

After spatial alignment, both feature maps are flattened into � spatial tokens and $\ell _ { 2 }$ -normalized along the channel dimension. We use a cosine feature distillation loss:

$$
\mathcal { L } _ { \mathrm { d i s t } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left[ 1 - \frac { \mathbf { f } _ { s , n } ^ { \top } \mathbf { f } _ { t , n } } { \left\| \mathbf { f } _ { s , n } \right\| _ { 2 } \left\| \mathbf { f } _ { t , n } \right\| _ { 2 } + \epsilon } \right]\tag{10}
$$

where $\mathbf { f } _ { s , n }$ and $\mathbf { f } _ { t , n }$ denote the �-th student and teacher tokens, respectively, and � is a small constant for numerical stability.

The segmentation objective follows the structure-aware weighted BCE and weighted IoU loss [2]:

$$
\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { w b c e } } + \mathcal { L } _ { \mathrm { w i o u } }\tag{11}
$$

Table 1  
Quantitative comparison with representative lightweight medical image segmentation methods on $\mathsf { P H } ^ { 2 }$ , ISIC2018, CVC-ClinicDB, and CVC-ColonDB. Dice scores are reported in %. Baseline results are reproduced or cited from their reported settings when available. Params and FLOPs measure student-side inference complexity. For GAD-MambaUNet with DINOv3 supervision, the teacher model is used only during training and is excluded from deployment cost.
<table><tr><td>Method</td><td>Params(M)</td><td>FLOPs(G)</td><td>Throughput(/s)</td><td>PH²</td><td>ISIC2018</td><td>ClinicDB</td><td>ColonDB</td><td> $\mathsf { A v g } .$ </td></tr><tr><td>U-Net</td><td>34.53</td><td>65.53</td><td>92.74</td><td>93.68</td><td>86.67</td><td>91.43</td><td>83.95</td><td>88.93</td></tr><tr><td>PraNet</td><td>32.55</td><td>6.93</td><td>49.94</td><td>94.16</td><td>88.46</td><td>91.71</td><td>89.16</td><td>90.87</td></tr><tr><td>UACANet</td><td>69.16</td><td>31.51</td><td>33.72</td><td>94.37</td><td>88.72</td><td>93.29</td><td>89.76</td><td>91.53</td></tr><tr><td>TransUNet</td><td>105.32</td><td>38.52</td><td>50.91</td><td>94.63</td><td>89.04</td><td>93.18</td><td>89.97</td><td>91.70</td></tr><tr><td>UNeXt</td><td>1.47</td><td>0.57</td><td>134.33</td><td>93.60</td><td>87.78</td><td>90.20</td><td>83.84</td><td>88.86</td></tr><tr><td>CMUNeXt</td><td>0.418</td><td>1.09</td><td>136.68</td><td>94.02</td><td>87.51</td><td>92.82</td><td>83.85</td><td>89.55</td></tr><tr><td>EGE-UNet</td><td>0.054</td><td>0.072</td><td>78.70</td><td>93.69</td><td>86.95</td><td>84.76</td><td>76.03</td><td>85.36</td></tr><tr><td>UltraLight VM-UNet</td><td>0.050</td><td>0.060</td><td>76.38</td><td>93.82</td><td>87.85</td><td>87.11</td><td>80.06</td><td>87.21</td></tr><tr><td>MK-UNet-T</td><td>0.027</td><td>0.062</td><td>109.76</td><td>94.56</td><td>88.19</td><td>91.26</td><td>85.03</td><td>89.76</td></tr><tr><td>MK-UNet</td><td>0.316</td><td>0.314</td><td>107.23</td><td>94.39</td><td>88.74</td><td>93.48</td><td>90.01</td><td>91.66</td></tr><tr><td>MK-UNet-L</td><td>3.76</td><td>3.19</td><td>96.38</td><td>94.56</td><td>89.25</td><td>93.85</td><td>91.82</td><td>92.37</td></tr><tr><td>GAD-MambaUNet-T w/o DINOv3</td><td>0.076</td><td>0.054</td><td>100.14</td><td>94.43</td><td>88.27</td><td>92.47</td><td>83.20</td><td>89.59</td></tr><tr><td>GAD-MambaUNet-S w/o DINOv3</td><td>0.199</td><td>0.128</td><td>99.65</td><td>94.40</td><td>88.63</td><td>93.85</td><td>90.17</td><td>91.76</td></tr><tr><td>GAD-MambaUNet-B w/o DINOv3</td><td>0.493</td><td>0.322</td><td>97.97</td><td>94.90</td><td>88.91</td><td>94.04</td><td>91.24</td><td>92.27</td></tr><tr><td>GAD-MambaUNet-M w/o DINOv3 GAD-MambaUNet-L w/o DINOv3</td><td>1.839</td><td>0.955</td><td>92.25</td><td>94.71</td><td>89.13</td><td>94.07</td><td>91.35</td><td>92.31</td></tr><tr><td></td><td>5.871</td><td>3.325</td><td>90.86</td><td>94.88</td><td>89.34</td><td>94.70</td><td>91.37</td><td>92.57</td></tr><tr><td>GAD-MambaUNet-T w/ ViT-T</td><td>0.076</td><td>0.054</td><td>100.14</td><td>94.61</td><td>88.61</td><td>93.09</td><td>86.00</td><td>90.58</td></tr><tr><td>GAD-MambaUNet-S w/ ViT-T</td><td>0.199</td><td>0.128</td><td>99.65</td><td>94.73</td><td>88.94</td><td>94.16</td><td>90.76</td><td>92.15</td></tr><tr><td>GAD-MambaUNet-B w/ ViT-T</td><td>0.493</td><td>0.322</td><td>97.97</td><td>95.16</td><td>89.06</td><td>94.17</td><td>91.96</td><td>92.59</td></tr><tr><td>GAD-MambaUNet-M w/ ViT-B</td><td>1.839</td><td>0.955</td><td>92.25</td><td>95.21</td><td>89.11</td><td>94.52</td><td>93.08</td><td>92.98</td></tr><tr><td>GAD-MambaUNet-L w/ ViT-B</td><td>5.871</td><td>3.325</td><td>90.86</td><td>95.37</td><td>89.53</td><td>95.08</td><td>93.01</td><td>93.25</td></tr></table>

The overall training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { e } \mathcal { L } _ { \mathrm { d i s t } }\tag{12}
$$

where $\lambda _ { e }$ is the distillation coeficient at epoch �.

Using a fixed distillation coeficient assumes that the proper teacher supervision strength remains unchanged through out training. However, the optimization state of the student changes over time, and the gradient contribution of the teacher-supervised decoder block may become either too weak or too dominant. To balance semantic supervision and the primary segmentation objective, GAD dynamically updates $\lambda _ { e }$ according to the gradient share of the aligned decoder block. After back-propagation and before gradient clipping, we compute

$$
q _ { e } = 1 0 0 \times \frac { \sum _ { \theta _ { i } \in \Theta _ { a } } \left\| \nabla _ { \theta _ { i } } \mathcal { L } \right\| _ { 1 } } { \sum _ { \theta _ { j } \in \Theta } \left\| \nabla _ { \theta _ { j } } \mathcal { L } \right\| _ { 1 } + \epsilon }\tag{13}
$$

where $\Theta _ { a }$ denotes the parameters of the aligned decoder block and Θ denotes all trainable student parameters.

Given a target gradient-share interval $[ \rho - \delta , \rho + \delta ]$ , GAD adjusts the next-epoch distillation coeficient only when $q _ { e }$ falls outside this interval. Let $q ^ { * }$ be the nearest target boundary, and define $p _ { e } = q _ { e } / 1 0 0$ and $p ^ { * } = q ^ { * } / 1 0 0$ . The odds-ratio adjustment is computed as

$$
r _ { e } = \frac { p ^ { * } ( 1 - p _ { e } ) } { p _ { e } ( 1 - p ^ { * } ) + \epsilon }\tag{14}
$$

The distillation coeficient for the next epoch is updated by

$$
\lambda _ { e + 1 } = \mathrm { c l i p } \left( \lambda _ { e } \mathrm { c l i p } ( r _ { e } , r _ { \mathrm { m i n } } , r _ { \mathrm { m a x } } ) , \lambda _ { \mathrm { m i n } } , \lambda _ { \mathrm { m a x } } \right)\tag{15}
$$

where $r _ { \mathrm { m i n } }$ and $r _ { \mathrm { m a x } }$ bound the per-epoch adjustment, and $\lambda _ { \operatorname* { m i n } }$ and $\lambda _ { \operatorname* { m a x } }$ constrain the overall distillation strength.

Training begins with a segmentation-only burn-in stage, followed by a linear distillation warm-up. GAD is activated after warm-up to adaptively regulate the teacher contribution. During inference, the DINOv3 teacher, the feature projection head, and the GAD update are all discarded. Therefore, the proposed training-time supervision improves the compact student without increasing deployment cost.

## 4. Experiments and Results

## 4.1. Datasets and Metrics

We evaluate GAD-MambaUNet on four public binary medical image segmentation datasets covering two representative segmentation tasks. For skin lesion segmentation, we use $\mathrm { P H } ^ { 2 }$ [20], which contains 200 dermoscopic images with expert lesion annotations, and ISIC2018 [21], which contains 2,594 dermoscopic images with pixel-level lesion masks. For polyp segmentation, we use CVC-ClinicDB [22], which contains 612 colonoscopic polyp images, and CVC-ColonDB [23, 24], which contains 379 colonoscopic images collected from diferent video sequences. These datasets include diverse imaging conditions, lesion appearances, object scales, boundary ambiguities, and foreground–background contrasts, providing a comprehensive evaluation of the robustness of lightweight segmentation models.

We split each dataset into training, validation, and test subsets with a ratio of 80:10:10. The validation set is used for model selection, and the test set is used for final performance reporting. Dice score is used as the primary segmentation metric, and Intersection over Union (IoU) is also reported when applicable. To evaluate model eficiency, we report the number of trainable parameters and floating-point operations (FLOPs). Since the DINOv3 teacher is used only during training and is discarded during deployment, it does not introduce additional inference computation. Therefore, all eficiency metrics are computed using the student network alone.

Table 2  
Ablation study of DG-GSS. Dice scores are reported in $\% .$
<table><tr><td>Variant</td><td>Params(M)</td><td>FLOPs(G)</td><td>PH2</td><td>ISIC2018</td><td>ClinicDB</td><td>ColonDB</td></tr><tr><td>MKIR baseline (MK-UNet)</td><td>0.316</td><td>0.314</td><td>94.39</td><td>88.74</td><td>93.48</td><td>90.01</td></tr><tr><td>w/ GroupedSS2D</td><td>0.416</td><td>0.281</td><td>94.47</td><td>88.79</td><td>93.76</td><td>90.64</td></tr><tr><td>w/ DG-GSS</td><td>0.493</td><td>0.322</td><td>94.90</td><td>88.91</td><td>94.04</td><td>91.24</td></tr></table>

## 4.2. Implementation Protocol

All experiments are implemented in PyTorch and conducted under the same training and evaluation settings for fair comparison. Following the input-resolution settings adopted in MK-UNet, images from $\bar { \mathrm { P H } } ^ { 2 }$ and ISIC2018 are resized to 256×256, while images from CVC-ClinicDB and CVC-ColonDB are resized to 352 × 352. The validation set is used for checkpoint selection, and the corresponding test performance is reported.

For GAD-MambaUNet, we use AdamW as the optimizer with an initial learning rate of $5 \times 1 0 ^ { - 4 }$ , a weight decay of $1 0 ^ { - 4 }$ , and cosine learning-rate decay to $1 0 ^ { - 6 }$ . All models are trained for 400 epochs with a batch size of 8, gradient clipping at 0.5, and standard data augmentation. The segmentation objective is the sum of weighted binary cross-entropy and weighted IoU losses. Unless otherwise specified, each configuration is repeated over five independent runs, and the mean test performance is reported.

For training-time semantic supervision, we use frozen DINOv3 ViT-T/16 and ViT-B/16 models as semantic teachers in diferent experimental settings. The student feature after the first decoder block is aligned to the teacher feature through the projection head described in Section 3. DINOv3 distillation starts at epoch 10 and is linearly warmed up for 20 epochs. GAD is activated at epoch 30 and adaptively controls the distillation coeficient until epoch 360. During inference, the teacher branch, projection head, and GAD update are discarded, so the reported parameters and FLOPs are computed using only the student network.

## 4.3. Comparison with Reference Methods

Tab. 1 compares GAD-MambaUNet with representative medical image segmentation networks and lightweight baselines on four public datasets. Compared with heavy encoder– decoder and Transformer-based models, the proposed models achieve competitive or better accuracy with substantially lower complexity. As shown in Tab. 1, TransUNet requires 105.32M parameters and 38.52G FLOPs with an average Dice of 91.70%, whereas GAD-MambaUNet-B without DI-NOv3 achieves a higher average Dice of 92.27% using only 0.493M parameters and 0.322G FLOPs. This indicates that the proposed asymmetric local–global design can improve segmentation accuracy without relying on a heavy backbone.

Among lightweight methods, MK-UNet is a strong reference baseline because it achieves competitive results with compact multi-kernel convolutional blocks. Compared with MK-UNet, which obtains an average Dice of 91.66% with 0.316M parameters and 0.314G FLOPs, GAD-MambaUNet-B without DINOv3 improves the average Dice to 92.27% with comparable computational cost. This gain is mainly attributed to introducing DG-GSS blocks only at deep low-resolution stages, thereby enhancing contextual modeling while keeping the inference cost low. Specifically, GAD-MambaUNet-B obtains 94.90%, 88.91%, 94.04%, and 91.24% Dice on PH<sup>2</sup>, ISIC2018, CVC-ClinicDB, and CVC-ColonDB, respectively. Increasing the model scale further improves the ClinicDB result, with GAD-MambaUNet-Large achieving 94.70% on CVC-ClinicDB and 91.37% on CVC-ColonDB.

Training-time DINOv3 supervision consistently improves the student models without changing their student-side inference complexity. For example, GAD-MambaUNet-Base improves from 94.90% to 95.1% on $\mathrm { P H } ^ { 2 }$ , from 88.91% to 89.06% on ISIC2018, from 94.04% to 94.17% on CVC-ClinicDB, and from 91.24% to 91.96% on CVC-ColonDB. The improvement is more evident on the challenging CVC-ColonDB dataset, where stronger semantic guidance helps distinguish polyp regions from visually similar background tissues. The Medium variant with DINOv3 obtains the best ColonDB Dice of 93.08%, while the Large variant with DINOv3 achieves the best $\mathrm { P H } ^ { 2 }$ , ISIC2018, and CVC-ClinicDB results among the compared variants.

Overall, the results show that DG-GSS improves the accuracy–eficiency balance of the lightweight student, and DINOv3-GAD further enhances semantic representation during training without increasing deployment cost. This supports the motivation of combining deep direction–group state-space interaction with training-time foundation-model supervision for lightweight medical image segmentation.

## 4.4. Ablation Study of DG-GSS

To verify the efectiveness of the proposed Direction-Group Graph Selective Scan, we conduct an ablation study by comparing three representative variants, as shown in Tab. 2. The MKIR baseline [5] corresponds to the lightweight convolutional design of MK-UNet, which mainly relies on local multi-kernel feature extraction. The GroupedSS2D variant replaces part of the deep local blocks with grouped multi-directional selective scanning, while the complete DG-GSS variant further introduces direction–group graph interaction among scan-direction and channel-group responses.

Compared with the MKIR baseline, introducing GroupedS improves the Dice score from 94.39% to 94.47% on PH<sup>2</sup>, from 88.74% to 88.79% on ISIC2018, from 93.48% to 93.76% on CVC-ClinicDB, and from 90.01% to 90.64% on CVC-ColonDB. This indicates that replacing deep local convolutional blocks with grouped selective scanning can enhance long-range contextual modeling while maintaining low computational cost.

The complete DG-GSS variant further improves over GroupedSS2D on all four datasets, achieving 94.90%, 88.91%, 94.04%, and 91.24% Dice on PH<sup>2</sup>, ISIC2018, CVC-ClinicDB, and CVC-ColonDB, respectively. The improvement is especially clear on CVC-ColonDB, where DG-GSS improves the Dice score by 0.60% over GroupedSS2D and by 1.23% over the MKIR baseline. This suggests that graph-based interaction among direction–group responses helps capture complementary contextual cues across scan directions and channel subspaces.

In terms of eficiency, DG-GSS keeps the computation lightweight by applying state-space modeling only at lowresolution deep stages. Although the parameter count difers across variants, all models remain within the lightweight range, and DG-GSS achieves the best overall segmentation performance with only 0.493M parameters and 0.322G FLOPs. These results show that the proposed DG-GSS module improves the accuracy–eficiency balance of the student network rather than simply relying on a heavier architecture.

## 5. Conclusion

In this paper, we proposed GAD-MambaUNet, a lightweigh medical image segmentation network that combines eficient local modeling, direction–group state-space interaction, and training-time foundation-model supervision. To improve contextual modeling in compact segmentation networks, we introduced Direction-Group Graph Selective Scan (DG-GSS), which treated scan-direction and channel-group responses as graph nodes and enabled structured information exchange before multi-directional fusion. We further incorporated DINOv3-GAD supervision, where a frozen DINOv3 teacher provided semantic guidance during training, and Gradient-Adaptive Distillation dynamically regulated the distillation strength. GAD-MambaUNet achieves a favorable accuracy–eficiency balance compared with representative lightweight and general segmentation methods. Ablation studies further verify the efectiveness of DG-GSS and training-time DINOv3-GAD supervision. In future work, we will explore more flexible teacher–student alignment strategies and extend the proposed framework to more diverse medical segmentation scenarios, such as multi-class and multi-modal segmentation tasks.

## References

[1] O. Ronneberger, P. Fischer, T. Brox, U-Net: Convolutional networks for biomedical image segmentation, in: Medical Image Computing and Computer-Assisted Intervention, 2015, pp. 234–241.

[2] D.-P. Fan, G.-P. Ji, T. Zhou, G. Chen, H. Fu, J. Shen, L. Shao, PraNet: Parallel reverse attention network for polyp segmentation, in: Medical

Image Computing and Computer-Assisted Intervention, 2020, pp. 263–273.

[3] T. Kim, H. Lee, D. Kim, UACANet: Uncertainty augmented context attention for polyp segmentation, arXiv preprint arXiv:2107.02368 (2021).

[4] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L. Yuille, Y. Zhou, TransUNet: Transformers make strong encoders for medical image segmentation, arXiv preprint arXiv:2102.04306 (2021).

[5] M. M. Rahman, R. Marculescu, MK-UNet: Multi-kernel lightweight CNN for medical image segmentation, arXiv preprint arXiv:2509.18493 (2025).

[6] F. Tang, J. Ding, Q. Quan, L. Wang, C. Ning, S. K. Zhou, CMUNeXt: An eficient medical image segmentation network based on large kernel and skip fusion, in: IEEE International Symposium on Biomedical Imaging, 2024, pp. 1–5.

[7] J. Ruan, M. Xie, J. Gao, T. Liu, Y. Fu, EGE-UNet: An eficient group enhanced UNet for skin lesion segmentation, in: Medical Image Computing and Computer-Assisted Intervention, 2023, pp. 481–490.

[8] M. M. Rahman, M. Munir, R. Marculescu, EMCAD: Eficient multiscale convolutional attention decoding for medical image segmentation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 11769–11779.

[9] J. M. J. Valanarasu, V. M. Patel, UNeXt: MLP-based rapid medical image segmentation network, in: Medical Image Computing and Computer-Assisted Intervention, 2022, pp. 23–33.

[10] A. Gu, T. Dao, Mamba: Linear-time sequence modeling with selective state spaces, arXiv preprint arXiv:2312.00752 (2023).

[11] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, J. Jiao, Y. Liu, VMamba: Visual state space model, arXiv preprint arXiv:2401.10166 (2024).

[12] J. Ruan, S. Xiang, VM-UNet: Vision mamba UNet for medical image segmentation, arXiv preprint arXiv:2402.02491 (2024).

[13] R. Wu, Y. Liu, P. Liang, Q. Chang, Ultralight VM-UNet: Parallel vision mamba significantly reduces parameters for skin lesion segmentation, arXiv preprint arXiv:2403.20035 (2024).

[14] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jégou, P. Labatut, P. Bojanowski, DINOv3, arXiv preprint arXiv:2508.10104 (2025).

[15] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 10012–10022.

[16] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, M. Wang, Swin-unet: Unet-like pure transformer for medical image segmentation, in: European Conference on Computer Vision Workshops, 2022, pp. 205–218.

[17] G. Hinton, O. Vinyals, J. Dean, Distilling the knowledge in a neural network, in: NeurIPS Deep Learning and Representation Learning Workshop, 2015.

[18] Z. Liao, Y. Zhao, X. Shan, Y. Yan, C. Liu, L. Lu, X. Ji, J. Chen, RT-DETRv4: Painlessly furthering real-time object detection with vision foundation models, arXiv preprint arXiv:2510.25257 (2025).

[19] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio, Graph attention networks, in: International Conference on Learning Representations, 2018.

[20] T. Mendonça, P. M. Ferreira, J. S. Marques, A. R. Marcal, J. Rozeira, Ph 2-a dermoscopic image database for research and benchmarking, in: 2013 35th annual international conference of the IEEE engineering in medicine and biology society (EMBC), IEEE, 2013, pp. 5437– 5440.

[21] N. Codella, V. Rotemberg, P. Tschandl, M. E. Celebi, S. Dusza, D. Gutman, B. Helba, A. Kalloo, K. Liopyris, M. Marchetti, et al., Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration, arXiv preprint arXiv:1902.03368 (2019).

[22] J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, D. Gil, C. Rodríguez, F. Vilariño, WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians, Computerized Medical Imaging and Graphics 43 (2015) 99–111.

[23] N. Tajbakhsh, S. R. Gurudu, J. Liang, Automated polyp detection in colonoscopy videos using shape and context information, IEEE Transactions on Medical Imaging 35 (2016) 630–644.

[24] D. Vázquez, J. Bernal, F. J. Sánchez, G. Fernández-Esparrach, A. M. López, A. Romero, M. Drozdzal, A. Courville, A benchmark for endoluminal scene segmentation of colonoscopy images, Journal of Healthcare Engineering 2017 (2017).