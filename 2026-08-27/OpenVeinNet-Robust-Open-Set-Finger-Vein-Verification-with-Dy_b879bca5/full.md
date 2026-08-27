# OpenVeinNet: Robust Open-Set Finger Vein Verification with Dynamic Snake Convolution and Graph Learning

Sushrut Patwardhan and Raghavendra Ramachandra

Abstract—Finger vein verification is a promising biometric modality for secure authentication because vascular patterns are internal, difficult to observe externally, and relatively resistant to presentation attacks. However, reliable verification remains challenging in open-set settings, where test identities are unseen during training and non-enrolled probes must be rejected at inference. This paper presents OpenVeinNet, a finger vein verification framework designed for cross-dataset and open-set evaluation. The proposed mode combines Dynamic Snake Convolution with graph-based feature modelling. Dynamic Snake Convolution extracts local curvilinear and tubular vein structures using adaptive sampling, while the graph convolutional backbone models long-range topological relationships between vein regions. To improve the discriminative quality of the embedding space, we introduce a Centroid Angular Hybrid Loss, which jointly encourages intra-class compactness and inter-class angular separation for cosine-similarity-based verification. Experiments are conducted on five public finger vein datasets: FV-300, MMCBNU, FV-USM, PolyU, and VERA. The method is evaluated using leave-one-dataset-out training under both enrolment-based unknown-rejection and full-subject verification protocols, and is compared with handcrafted and recent deep learning-based baselines. The results show that OpenVeinNet achieves strong cross-dataset generalisation, consistently low equal error rates, and competitive true accept rates at fixed false accept rate operating points. Ablation studies further confirm the individual and combined contributions of adaptive tubular feature extraction, graph-based relational modelling, and the proposed loss function. These findings indicate that explicitly modelling local vein geometry, globa vascular relationships, and angularly compact embeddings is effective for open-set finger vein verification.

Index Terms—Biometrics, Finger vein, Deep learning, Graph Neural Network, Verification.

## 1 INTRODUCTION

Reliable user verification is essential for secure access control systems across a wide range of applications. Biometric authentication has become a widely adopted solution because it offers convenient and identity-linked access control. Conventional biometric traits such as fingerprints, facial features, irises, palmprints, and ears are commonly used; however, they may be vulnerable to spoofing, particularly when acquired in a non-intrusive manner. In contrast, finger vein patterns provide an attractive alternative because the vascular structures are located beneath the skin and are therefore difficult to observe externally or replicate. This intrinsic resistance to presentation attacks, together with high verification accuracy and relatively low deployment cost, has increased the use of finger vein recognition, particularly in banking and financial services.

Finger vein patterns are typically acquired using costeffective near-infrared (NIR) imaging systems, where the light source and camera are positioned above, alongside, or around the finger. When NIR light penetrates the finger, it is absorbed by haemoglobin in the veins, causing the vascular structures to appear darker than the surrounding tissue. Finger vein biometrics are also characterised by temporal stability, relatively low sensitivity to external environmental variations, and strong uniqueness across individuals. Furthermore, capturing images from both the dorsal and palmar sides of the finger can improve verification performance.

![](images/03ba4d00c72f56b5b635d37bd5eb842b0981f98bad9ad0e9ae2dc38c2aac3d88.jpg)  
Fig. 1: A brief taxonomy of finger vein verification techniques based on deep learning approaches.

Finger vein verification has been extensively investigated in the literature, resulting in a wide range of handcrafted and deep learning-based methods. Earlier studies primarily relied on image processing techniques to extract vein patterns for template-based matching. Representative handcrafted methods include Maximum Curvature Points (MCP) [35], Repeated Line Tracking (RLT) [36], Wide Line Detectors (WLD) [37], and Mean Curvature (MC), which enhance vascular structures and can achieve reliable verification under controlled acquisition conditions. Texture descriptors such as Local Binary Pattern (LBP) [38], Local Line Binary Pattern (LLBP) [39], Histogram of Oriented Gradients (HoG) [40], and Local Directional Pattern (LDP) have also been explored. In addition, dimensionality reduction methods, including Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA), have been used to improve computational efficiency and reduce feature redundancy. Although these handcrafted approaches remain effective in clean and well-controlled scenarios, they often struggle to generalise across changes in image quality, capture devices, and acquisition environments [41].

<table><tr><td rowspan=1 colspan=1>Authors (Year)</td><td rowspan=1 colspan=2>Technique Description</td></tr><tr><td rowspan=1 colspan=1>Huafeng Qin et al., [1] (2015)</td><td rowspan=1 colspan=1>3 Sequential convolution l</td><td rowspan=1 colspan=1>ayers and 2 dense layers</td></tr><tr><td rowspan=1 colspan=1>Itqan et al., [2] (2016)</td><td rowspan=1 colspan=1>3 Sequential convolution l</td><td rowspan=1 colspan=1>ayers and 1 dense layers</td></tr><tr><td rowspan=1 colspan=1>Syafeeza Radzi et al., [3] (2016)</td><td rowspan=1 colspan=1>2 Sequential convolution l</td><td rowspan=1 colspan=1>ayers</td></tr><tr><td rowspan=1 colspan=1>Huafeng Qin et al., [4] (2017)</td><td rowspan=1 colspan=1>4 Sequential convolution l</td><td rowspan=1 colspan=1>ayers with path-based training.</td></tr><tr><td rowspan=1 colspan=1>Hyung Gil Hong et al., [5] (2017)</td><td rowspan=1 colspan=1>12 Sequential convolution</td><td rowspan=1 colspan=1>layers and 3 dense layers</td></tr><tr><td rowspan=1 colspan=1>Rig Das et al., [6] (2018)</td><td rowspan=1 colspan=1>5 Sequential convolution l</td><td rowspan=1 colspan=1>ayers</td></tr><tr><td rowspan=1 colspan=1>Cihui Xie et al., [7] (2019)</td><td rowspan=1 colspan=2>Siamese network with 5 conventional layers trained on triplet loss function.</td></tr><tr><td rowspan=1 colspan=1>Jong Min Song et al., [8] (2019)</td><td rowspan=1 colspan=2>8 Sequential convolution layers trained using composite finger vein image gener-ated by converting the 1-channel input image to a 3-channel input image.</td></tr><tr><td rowspan=1 colspan=1>Su Tang et al., [9] (2019)</td><td rowspan=1 colspan=2>Siamese network with residual CNN architecture.</td></tr><tr><td rowspan=1 colspan=1>Borui Hou et al., [10] (2019)</td><td rowspan=1 colspan=2>Autoencoder architecture.</td></tr><tr><td rowspan=1 colspan=1>Wenming Yang et al., [11] (2019)</td><td rowspan=1 colspan=2>Using GAN for generating enhanced images and performing verification usingencoder features of GAN.</td></tr><tr><td rowspan=1 colspan=1>Junying Zeng et al., [12] (2020)</td><td rowspan=1 colspan=2>Deformable convolution in U-NET type architecture.</td></tr><tr><td rowspan=1 colspan=1>Ridvan Salih Kuzu et al., [13] (2020)</td><td rowspan=1 colspan=2>6 Sequential convolution layers and 2 dense layers and LSTM for classification.</td></tr><tr><td rowspan=1 colspan=1>Zhiang Hao et al., [14] (2020)</td><td rowspan=1 colspan=2>Multitask Learning based on Resnet-18 like model with SoftL1Loss and ArcFaceloss</td></tr><tr><td rowspan=1 colspan=1>Hengyi Ren et al., [15] (2021)</td><td rowspan=1 colspan=2>Squeeze and Excitation blocks embedded Resnet trained on encrypted finger veinimages.</td></tr><tr><td rowspan=1 colspan=1>Ridvan Salih Kuzu et al., [16] (2021)</td><td rowspan=1 colspan=2>Custom DenseNet 161 trained on additive angular penalty and large margincosine penalty loss function</td></tr><tr><td rowspan=1 colspan=1>Borui Hou et. al., [17] (2021)</td><td rowspan=1 colspan=2>Resnet-like architecture with ECA-Resnet blocks trained on arc-cosine loss andsoftmax loss together.</td></tr><tr><td rowspan=1 colspan=1>Zhenxiang Chen et. al., [18] (2021)</td><td rowspan=1 colspan=2>Resnet-18 architecture trained on arc-loss with image enhancement, andluminance-inversion data augmentation</td></tr><tr><td rowspan=1 colspan=1>Weili Yang et al., [19] (2022)</td><td rowspan=1 colspan=2>Multi-view finger vein with individual CNNs and view pooling.</td></tr><tr><td rowspan=1 colspan=1>Huafeng Qin et al., [20] (2022)</td><td rowspan=1 colspan=2>Based on Local Attention Transformer</td></tr><tr><td rowspan=1 colspan=1>Tingting Chai et al., [21] (2022)</td><td rowspan=1 colspan=2>5 Sequential convolution layers and 1 dense layers</td></tr><tr><td rowspan=1 colspan=1>Ismail et al., [22] (2022)</td><td rowspan=1 colspan=2>Deeply-fused 3 Sequential convolution layers and 2 dense layers.</td></tr><tr><td rowspan=1 colspan=1>Weiye Liu et al., [23] (2023)</td><td rowspan=1 colspan=2>Inception architecture with residual attention block.</td></tr><tr><td rowspan=1 colspan=1>Zhongxia Zhang et al., [24] (2023)</td><td rowspan=1 colspan=2>CNN with spatial and channel attention module.</td></tr><tr><td rowspan=1 colspan=1>Chunxin Fang et al., [25] (2023)</td><td rowspan=1 colspan=2>Siamese network with attention module.</td></tr><tr><td rowspan=1 colspan=1>Chih-Hsien Hsia et al. [26] (2023)</td><td rowspan=1 colspan=2>Lightweight Convolutional Neural Network with attention and Margin basedloss function.</td></tr><tr><td rowspan=1 colspan=1>Yiwei Huang et al. [27] (2023)</td><td rowspan=1 colspan=2>Axially enhanced local attention network (criss-cross kernel)</td></tr><tr><td rowspan=1 colspan=1>Huang, Junduan et al. [28] (2023)</td><td rowspan=1 colspan=2>Frequency spatial coupling network with resnet like architecture</td></tr><tr><td rowspan=1 colspan=1>Bin Wa et al., [29] (2023)</td><td rowspan=1 colspan=2>3 Sequential convolution layers with bilinear pooling and multiple attentionmodules.</td></tr><tr><td rowspan=1 colspan=1>Xiaoye Li et al., [30] (2023)</td><td rowspan=1 colspan=2>Vision Transformer with rigorous regularisation added in mlp.</td></tr><tr><td rowspan=1 colspan=1>Raghavendra Ramachandra et al., [31] (2024)</td><td rowspan=1 colspan=2>3 Sequential convolution layers followed by multi-head attention module con-nected in parallel with normal and enhanced finger vein images.</td></tr><tr><td rowspan=1 colspan=1>Huafeng Qin et al., [32] (2024)</td><td rowspan=1 colspan=2>Generates optimal network using Neural Architecture Search using Gated Recur-rent Units with self-attention.</td></tr><tr><td rowspan=1 colspan=1>Pengyang Zhao et. al. [33] (2024)</td><td rowspan=1 colspan=2>Transformer-based view finger vein recognition model which uses enhancedimages as attention mask</td></tr><tr><td rowspan=1 colspan=1>Enyan Li et. al. [34] (2025)</td><td rowspan=1 colspan=2>Transformer and convolution-based model processing local and global features.</td></tr><tr><td rowspan=1 colspan=1>Proposed Method (2026)</td><td rowspan=1 colspan=2>Dynamic Snake Convolution with Graph Convolution layers for featureextraction and novel Centroid-Angular Hybrid Loss.</td></tr></table>

TABLE 1: Summary of existing deep learning-based finger vein verification techniques.

In recent years, deep learning has been extensively explored for finger vein verification, yielding strong performance gains over traditional methods. A taxonomy of deep learning algorithms for finger vein verification is illustrated in Figure 1, and a summary of representative techniques is provided in Table 1. Early deep learning-based approaches typically employed serial convolutional neural network (CNN) architectures, consisting of convolutional layers followed by one or two fully connected layers. Such models, as used in [1], [2], [3], [4], [5], [6], [8], [13], [14], [15], [16], [18], [19], [21], [22], are generally lightweight and computationally efficient, but their generalisation can be limited when evaluated across unseen acquisition conditions or new subject populations.

Autoencoder-based networks have been used to learn high-level feature representations for finger vein verification [10], [11]. Deformable filters incorporated into Fully Convolutional Networks (FCNs) have also been proposed to better model spatial variations [12]. While these approaches improve local adaptability, they may require longer training and can still be sensitive to sensor-specific characteristics. Siamese networks have also been investigated for finger vein verification [7], [9], [25]. Their pairwise training formulation can help with class imbalance, but such models are computationally demanding to train and may become difficult to scale as the number of enrolled identities increases.

![](images/027cf5392a4d3fc38e2d8c199e4a7b2b7637e637b75c41f8d11df6972bf1babf.jpg)  
Fig. 2: Block diagram of the proposed OpenVeinNet framework for open-set finger vein verification. (a) The training phase incorporates a stem module with Dynamic Snake Convolution, followed by a multi-layer Graph Convolutional Network (GCN) and the proposed Centroid Angular Hybrid Loss for robust representation learning. (b) During inference, the system performs open-set verification by comparing probe samples against enrolled templates using the learned graphbased embeddings.

Attention-based architectures have attracted growing interest because they can model both local and global discriminative cues within a unified framework [17], [20], [23], [24], [25], [26], [27], [29], [30], [31], [32], [33]. However, their performance can degrade under cross-dataset evaluation when the training and testing data differ in sensor characteristics, image quality, and acquisition protocols. Vision Transformer (ViT)-based methods have also been explored [20], [30], but their effectiveness is often constrained by the limited scale of publicly available finger vein datasets. More recently, LGFIN [34] combines multi-scale local and global contextual features using a dynamic feature interaction network, showing the importance of jointly modelling complementary vein information.

Despite these advances, two challenges remain central for practical finger vein verification. First, many methods show reduced robustness under cross-dataset evaluation because the learned representations are affected by sensor and acquisition shifts. Second, open-set deployment requires not only matching enrolled identities but also rejecting nonenrolled probes at fixed operating thresholds. This distinction is important because a system that performs well when all test identities are enrolled may still fail when unknown identities are presented during inference [18].

To address these limitations, we propose OpenVeinNet, a finger vein verification framework designed for robust cross-dataset and open-set evaluation. The proposed model uses Dynamic Snake Convolution (DSConv) layers to capture the curvilinear and tubular structures of finger vein patterns. These features are then processed by Graph Convolutional Networks (GCNs), which model long-range topological relationships between vein regions. For verification, the extracted embeddings are compared using cosine similarity between enrolment and probe samples. To further improve open-set discrimination, we introduce a Centroid Angular Hybrid Loss that encourages intra-class compactness and inter-class angular separation in the learned embedding space.

We evaluate the proposed approach on five publicly available finger vein datasets, namely FV-300, FV-USM, MMCBNU, PolyU, and VERA, and compare it against handcrafted and deep learning-based state-of-the-art methods. The evaluation includes both enrollment-based unknownrejection and full-subject verification protocols, together with threshold-based operating points such as TAR@FAR. The results show that OpenVeinNet achieves overall generalisation and favourable verification performance across challenging cross-dataset settings. The main contributions of this work are as follows:

We introduce OpenVeinNet, a finger vein verification framework that integrates DSConv-based tubular feature extraction with graph-based relational modelling to capture both local vein geometry and longrange vascular connectivity.

We propose a Centroid Angular Hybrid Loss that improves discriminative representation learning by promoting intra-class compactness and inter-class angular separation, which is particularly beneficial for open-set verification.

We conduct extensive experiments on five public datasets, including FV-300, FV-USM, MMCBNU, PolyU, and VERA, and benchmark the proposed method against handcrafted and deep learningbased state-of-the-art methods.

• We evaluate the method under both enrollmentbased unknown-rejection and full-subject verification protocols, reporting AUC, EER, and thresholdbased TAR@FAR operating points to reflect practical biometric deployment conditions.

We provide ablation studies, computational-cost analysis, and Grad-CAM++-based interpretability analysis to justify the architectural design, loss formulation, and decision behaviour of the proposed model.

The complete source code, trained checkpoints, and scripts required to reproduce the reported results are available at: https://github.com/Blazkowiz47/ deep-vein-gcn

Beyond architectural design, the experiments yield three analytical observations relevant to finger vein verification. The component ablation shows that graph-based relational modelling is only effective when local node features are geometrically aligned with vein structures, revealing a prerequisite specific to tubular biometric patterns. The loss ablation shows that explicitly penalising intra-class scatter through centroid alignment provides a practical advantage over margin-only losses in cosine-similarity-based open-set verification.

The remainder of this paper is organised as follows. Section 2 presents the proposed method, including its architectural components and loss function. Section 3 reports the experimental protocols and quantitative comparison with state-of-the-art methods. Section 4 presents the ablation studies used to validate the main design choices. Section 5 discuss the computation cost and latency, Section 6 provides the interpretability analysis. Finally, Section 7 concludes the paper and discusses future research directions.

## 2 PROPOSED METHOD: OPENVEINNET FOR OPEN-SET FINGER VEIN VERIFICATION

Finger vein recognition relies on the distinctive curvilinear patterns of veins located beneath the skin for biometric verification. Although these patterns are unique to each finger and individual, reliable verification remains challenging due to variations in vein visibility, image quality, finger pose, rotation, and acquisition conditions. These challenges become more pronounced in open-set verification, where evaluation identities are unseen during training and nonenrolled probe samples must be rejected at inference time. To address these issues, we propose OpenVeinNet, a finger vein verification framework that combines a Dynamic Snake Convolution Graph Neural Network (DscGrapher) with a Centroid Angular Hybrid Loss. The DscGrapher is designed to model the tubular and branching structure of finger veins using Graph Convolutional Networks (GCNs) [42]. It first extracts vein-aware local features using Dynamic Snake Convolution (DSConv) [43], and then aggregates neighbouring regions into graph-based embeddings. The resulting compact feature representation is compared using cosine similarity for verification.

Figure 2 shows the block diagram of the proposed OpenVeinNet, and Table 2 provides the architectural details. Figure 2(a) illustrates the training phase, while Figure 2(b) shows the open-set verification phase. During inference, finger vein images are passed through OpenVeinNet to extract discriminative embeddings. These embeddings are then compared using cosine similarity between the probe and enrolled reference samples.

As shown in Figure 2(a), OpenVeinNet consists of two main components. The first is a stem module that extracts local tubular vein features using DSConv blocks. The second is a graph-based backbone composed of Grapher Blocks, which captures local and long-range relationships between vein regions. In addition to the architectural design, we introduce the Centroid Angular Hybrid Loss to encourage compact intra-class embeddings and improved interclass angular separation. The stem module also includes a learnable positional embedding to preserve spatial structure before graph-based feature aggregation. The final global representation is obtained using adaptive average pooling and is used for verification.

The following subsections describe each component of the proposed OpenVeinNet framework.

## 2.1 Stem: Dynamic Snake Convolution (DSConv) Blocks

Finger vein patterns form complex, non-repeating, branching structures. Standard convolutional layers operate on fixed and regular grids, which limits their ability to follow curved and irregular vascular patterns. To address this limitation, we employ Dynamic Snake Convolution (DSConv), which adjusts its sampling locations to better align with local vein contours. This adaptive sampling behaviour allows the network to extract features from faint, curved, or locally deformed vein structures more effectively than standard convolutions. Unlike general deformable and orientationadaptive convolutions, DSConv progressively constrains neighbouring sampling locations along a continuous curvilinear trajectory, providing a morphology-aligned inductive bias for following thin and tortuous finger-vein structures.

![](images/0526fab1cd59cf83cfdd53339d2ca8601dbe43bb3453fecd8b6a45561065d8d1.jpg)  
Fig. 3: DSBlock (In: input channels, Out: output channels, K: kernel size, x and y are the directions in which DSConv operates)

<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Layer</td><td rowspan=1 colspan=1>Output Dimensions</td></tr><tr><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>224,224,3</td></tr><tr><td rowspan=1 colspan=1>Stem</td><td rowspan=1 colspan=1>DSC module 1 (S=2, F=3)DSC module 2 (S=2, F=3)DSC module 3 (S=2, F=3)</td><td rowspan=1 colspan=1>(112,112,16)(56,56,32)(28,28,64)</td></tr><tr><td rowspan=1 colspan=1>Backbone</td><td rowspan=1 colspan=1>Stage I (K=18, S=2, F=3)Stage II (K=9, S=2, F=3)avg_pool</td><td rowspan=1 colspan=1>(14,14,128)(7,7,256)(1,1,256)</td></tr></table>

TABLE 2: OpenVeinNet architecture details indicating output dimension. K indicates the number of nearest neighbours, S indicates stride, and F indicates filter size.

Figure 3 illustrates the proposed DSConv Block. The block contains one standard convolution branch and two DSConv branches operating along the horizontal and vertical directions. The standard convolution preserves conventional local image evidence, while the DSConv branches adapt to direction-specific vein structures. The outputs of these branches are concatenated and passed through a final convolution to produce the block output. This design allows the stem to capture both regular local texture and deformable tubular features.

Figure 4 shows the complete stem architecture. Each DSConv Block is followed by batch normalisation and GELU activation. The GELU activation is omitted after the final DSConv Block to allow the extracted feature map to retain a wider range of responses, which is useful for preserving fine-grained vein information before graph-based processing.

Given a finger vein image $\pmb { X } \in \mathbb { R } ^ { B \times C _ { \mathrm { i n } } \times H \times W }$ , where B is the batch s $\mathrm { i z e } , \ C _ { \mathrm { i n } }$ is the number of channels, and $H ~ = ~ W ~ = ~ 2 2 4$ , each DSConv applies a deformable sampling strategy based on learned offset maps. $\mathrm { ~ A ~ 3 ~ } \times \mathrm { ~ 3 ~ }$ convolution predicts offsets $O \in \mathbb { R } ^ { B \times 2 K \times H \times W }$ , where K is the kernel size and $O = ( O _ { y } , O _ { x } )$ contains offsets for the vertical and horizontal directions:

$$
O = \operatorname { t a n h } ( \operatorname { G r o u p N o r m } ( \operatorname { C o n v } _ { 3 \times 3 } ( X ) ) )\tag{1}
$$

Let $\pmb { p } = ( p _ { y } , p _ { x } )$ denote a pixel position in the spatial grid, where $p _ { y } \in \{ 0 , 1 , \ldots , H - 1 \}$ and $p _ { x } \in \{ 0 , 1 , \ldots , W -$ 1}. The coordinate maps are generated as:

$$
Y _ { \mathrm { c o o r d } } , X _ { \mathrm { c o o r d } } = \mathrm { G e t C o o r d i n a t e M a p 2 D } ( \boldsymbol { O } , \boldsymbol { m } , \mathrm { e x t e n d \_ s c o p e } )
$$

where $m \in \{ 0 , 1 \}$ specifies deformation along the x-axis $( m = 0 )$ or y-axis $( m = 1 )$ , and extend scope = 1.0 controls the deformation range. The coordinates are normalised to $[ - 1 , 1 ]$ before interpolation.

The deformed feature map is sampled using bilinear interpolation:

$$
X _ { \mathrm { d e f o r m e d } } = \mathrm { G r i d S a m p l e } ( X , [ X _ { \mathrm { c o o r d } } , Y _ { \mathrm { c o o r d } } ] , \mathrm { m o d e } = \mathrm { " } ^ { \prime } \mathrm { b i l i n e a r " }\tag{3}
$$

A directional convolution is then applied to the deformed feature map:

$$
X _ { \mathrm { d i r } } = { \left\{ \begin{array} { l l } { \operatorname { C o n v } _ { ( K , 1 ) } ( X _ { \mathrm { d e f o r m e d } } ) , } & { { \mathrm { i f } } \ m = 0 } \\ { \operatorname { C o n v } _ { ( 1 , K ) } ( X _ { \mathrm { d e f o r m e d } } ) , } & { { \mathrm { i f } } \ m = 1 } \end{array} \right. }\tag{4}
$$

The output is normalised and activated:

$$
X _ { \mathrm { d i r } } = \mathrm { R e L U } ( \mathrm { G r o u p N o r m } ( X _ { \mathrm { d i r } } ) ) .\tag{5}
$$

The outputs from the standard convolution branch $\boldsymbol { X } _ { \mathrm { c o n v } } = \boldsymbol { \tilde { \mathrm { C o n v } } } _ { K \times K } ( \boldsymbol { X } )$ , the x-axis DSConv branch $X _ { x } =$ $\mathrm { D S C o n v } _ { x } ( X )$ , and the y-axis DSConv branch $\begin{array} { r l } { X _ { y } } & { { } = } \end{array}$ $\operatorname { D S C o n v } _ { y } ( X )$ are concatenated and fused by a final convolution:

$$
X _ { \mathrm { o u t } } = \operatorname { C o n v } _ { K \times K } ( \operatorname { C o n c a t } ( X _ { \mathrm { c o n v } } , X _ { x } , X _ { y } ) ) .\tag{6}
$$

![](images/167a8796bf6faef05eb1b4086aaf4da73486e11673eded7e9ad42ff096dadf85.jpg)  
Fig. 4: Stem architecture of OpenVeinNet, featuring a sequence of three Dynamic Snake Convolution (DSConv) layers with kernel sizes $K \in \{ 9 , 7 , 3 \}$ , stride $2 ,$ and GELU activation, processing an input image $\pm \nobreakspace \mathrm { ~ \mathbb ~ { ~ R ~ } ~ } ^ { B \times 3 \times 2 2 4 \times 2 2 4 }$ to produce a feature map $\dot { \bar { X } } _ { \mathrm { s t e m } } ~ \in ~ \mathbb { R } ^ { B \times 6 4 \times 2 8 \times 2 8 }$ for the backbone.

The adaptive offset learning in DSConv can be interpreted as aligning the sampling locations with vein-like structures:

$$
\operatorname* { m i n } _ { \pmb { O } } \sum _ { \mathrm { p i x e l s } } \| \pmb { X } ( \pmb { p } + \pmb { O } ) - \pmb { V } ( \pmb { p } ) \| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $V ( p )$ denotes an idealised vein response at location p. Although $V ( p )$ is not explicitly supervised, this formulation illustrates the intended behaviour of the learned offsets: to adapt the receptive field towards vein-aligned structures.

The final output of the stem is written as:

$$
\begin{array} { r } { X _ { \mathrm { s t e m } } = \mathrm { S t e m } ( \pmb { I } ) \in \mathbb { R } ^ { B \times C _ { \mathrm { o u t } } \times \frac { H } { 2 ^ { d } } \times \frac { W } { 2 ^ { d } } } , } \end{array}\tag{8}
$$

where $C _ { \mathrm { o u t } } = 6 4 , H = W = 2 2 4 _ { \cdot }$ , and $d = 3 ,$ , resulting in $X _ { \mathrm { s t e m } } \in \mathbb { R } ^ { B \times 6 4 \times 2 8 \times 2 8 }$

To preserve spatial information, a learnable positional embedding $\begin{array} { r } { P \in \dot { \mathbb { R } } ^ { 1 \times C _ { \mathrm { o u t } } \times \frac { H } { 2 ^ { d } } \times \frac { W } { 2 ^ { d } } } } \end{array}$ is added to the stem output:

$$
X _ { \mathrm { s t e m } } = \mathrm { S t e m } ( I ) + P .\tag{9}
$$

The positional embedding provides spatial context before graph construction, helping the backbone retain the topological arrangement of vein structures.

The backbone of OpenVeinNet is designed to model longrange connectivity between vein regions using Graph Convolutional Networks (GCNs). The stem output, augmented with positional embeddings, is processed by a sequence of Grapher Blocks. Each Grapher Block constructs a k-nearestneighbour (k-NN) graph over feature nodes and aggregates information from neighbouring nodes. This enables the model to capture non-local relationships that are difficult to represent using only standard convolutional operations.

![](images/383d37db3cfd26ed938bdc6dc7282df4acec89d2d9cfe8b667ff2f1cc4c5eaa5.jpg)  
Fig. 5: GrapherBlock (In: input channels, Out: output channels, K: kernel size, N: nearest neighbours taken into account by graph convolution.)

Given the input $X _ { \mathrm { s t e m } } ~ \in ~ \mathbb { R } ^ { B \times 6 4 \times 2 8 \times 2 8 }$ , each Grapher Block operates as follows:

Initial Projection: $\mathrm { ~ A ~ 1 ~ } \times \mathrm { ~ 1 ~ }$ convolution refines the input features, followed by batch normalisation:

$$
Y = \mathrm { B a t c h N o r m } ( \mathrm { C o n v } _ { 1 \times 1 } ( X _ { \mathrm { s t e m } } ) ) .\tag{10}
$$

• Dynamic Graph Construction: A k-NN graph is constructed based on feature similarity:

$$
\begin{array} { r } { \mathrm { E d g e I n d e x } = \mathrm { D e n s e D i l a t e d K n n G r a p h } ( Y , \ k , \ \mathrm { d i l a t i o n } } \\ { \mathrm { s t o c h a s t i c } , \ \epsilon ) . } \end{array}\tag{11}
$$

We use $k = 1 8$ in the first stage and $k = 9$ in the second stage.

• Graph Convolution: Features from neighbouring nodes are aggregated using an MRConv operation:

$$
X _ { \mathrm { o u t } } = \operatorname { M a x } ( \operatorname { B a s i c C o n v } ( \operatorname { C o n c a t } ( Y _ { i } , Y _ { j } - Y _ { i } ) ) ) ,\tag{12}
$$

where $\mathbf { Y } _ { i }$ and $Y _ { j }$ denote the features of node i and its neighbour $j ,$ , respectively.

Feed-Forward and Residual Connection: A feedforward network with two $1 \times 1$ convolutions refines the graph-aggregated features, and a residual connection is used to preserve the original feature information.

The graph convolution can be interpreted as aggregating the strongest relational evidence among the neighbouring nodes:

$$
\boldsymbol { X } _ { \mathrm { o u t } } \approx \operatorname* { m a x } _ { \boldsymbol { j } \in \mathcal { N } ( i ) } \ \mathrm { S i m } ( \boldsymbol { Y } _ { i } , \boldsymbol { Y } _ { j } ) ,\tag{13}
$$

![](images/10ecf25c7a9fb2c5f8b64b86c230d9fa3e43668e01dd01d0d17039316462950d.jpg)  
Fig. 6: Backbone structure of OpenVeinNet, with four sequential Grapher Blocks, a convolution layer with batch normalisation, six additional Grapher Blocks, another convolution layer with batch normalisation, and adaptive average pooling.

where $\mathcal { N } ( i )$ denotes the set of k nearest neighbours of node i.

As shown in Figure $^ { 6 , }$ the backbone has two graphprocessing stages followed by pooling. Block 0 takes 64 input channels and produces 128 output channels, using $k = 1 8$ neighbours and a $3 \times 3$ shrinker convolution with stride $2 ,$ reducing the spatial resolution to 14×14. This stage contains four internal Grapher units. Block 1 takes 128 input channels and produces 256 output channels, using $k \stackrel { = } { = } 9$ neighbours and a $3 \times 3$ shrinker convolution with stride $2 ,$ reducing the spatial resolution to $7 \times 7 .$ . This stage contains six internal Grapher units.

Finally, adaptive average pooling produces the embedding vector:

$$
E = \mathrm { A d a p t i v e A v g P o o l 2 d } ( \mathrm { G r a p h e r B l o c k s } ( X _ { \mathrm { { s t e m } } } ) ) \in \mathbb { R } ^ { B \times 2 5 6 } .
$$

The resulting embedding E is used for cosine-similaritybased verification. The use of k-NN graph construction enables the model to capture topological relationships between vein regions, while the pooled embedding provides a compact representation suitable for open-set verification.

## 2.3 Centroid Angular Hybrid Loss

The softmax loss is commonly used for classification tasks and is defined as:

$$
\mathcal { L } _ { 1 } = - \log \frac { e ^ { W _ { y _ { i } } ^ { T } x _ { i } + b _ { y _ { i } } } } { \sum _ { j = 1 } ^ { N } e ^ { W _ { j } ^ { T } x _ { i } + b _ { j } } } ,\tag{15}
$$

where $x _ { i } \in \mathbb { R } ^ { d }$ represents the embedding of the i-th sample with class label $\dot { y } _ { i } , W _ { j } \in \mathbb { R } ^ { d }$ denotes the j-th class weight vector from $W \in \mathbb { R } ^ { d \times N } , b _ { j }$ is the bias term, and N is the number of training classes.

Although softmax loss is effective for closed-set classification, it does not explicitly enforce compact intra-class embeddings or angularly separated inter-class representations.

This limitation is important in open-set verification, where the model must generalise beyond the identities seen during training and reject non-enrolled probes using a similarity threshold. To address this, we introduce the Centroid Angular Hybrid Loss $( \mathcal { L } _ { 2 } )$ , which augments the softmax objective with an angular term between the sample embedding and an independently learned class-direction reference.

The proposed loss is defined as:

$$
\mathcal { L } _ { 2 } = - \log \frac { e ^ { W _ { y _ { i } } ^ { T } x _ { i } + b _ { y _ { i } } + \beta \cdot \theta _ { y _ { i } } } } { \left( \sum _ { j = 1 } ^ { N } e ^ { W _ { j } ^ { T } x _ { i } + b _ { j } } \right) \cdot \left( \sum _ { j = 1 } ^ { N } e ^ { \theta _ { j } } \right) } .\tag{16}
$$

The angular component $\theta _ { j }$ is defined as:

$$
\theta _ { j } = \cos ^ { - 1 } \left( \frac { \tilde { W } _ { j } ^ { T } x _ { i } } { \| \tilde { W } _ { j } \| \cdot \| x _ { i } \| } \right) ,\tag{17}
$$

where $\tilde { W } \in \mathbb { R } ^ { d \times N }$ has the same shape as $W ,$ but is initialised independently. The term $\theta _ { j }$ captures the angular misalignment between the embedding $x _ { i }$ and the centroidlike direction $\tilde { W } _ { j }$

The balancing factor $\beta$ controls the contribution of the angular component. A small value of $\beta$ weakens angular supervision, while an overly large value can dominate the optimisation and reduce stability. The proposed formulation therefore encourages correct class prediction while also shaping the embedding space to be more compact within identities and more separated across identities. This is beneficial for open-set verification because the final decision is based on cosine similarity between enrolment and probe embeddings. In this formulation, the term centroid refers to a learnable class-direction prototype in the embedding space rather than an empirical mini-batch centroid.

In summary, the Centroid Angular Hybrid Loss complements the DSConv-GCN architecture by improving the discriminability of the learned feature space. DSConv extracts local curvilinear vein evidence, the graph backbone models long-range vascular relationships, and the proposed loss encourages embeddings that are better suited for thresholdbased verification of enrolled and non-enrolled identities.

## 3 EXPERIMENTS AND RESULTS

This section presents the experimental evaluation of the proposed OpenVeinNet and compares it with existing finger vein verification methods. Performance is reported using the Area Under the Curve (AUC), Equal Error Rate (EER), and threshold-dependent operating points, namely TAR@FAR= 0.1%, TAR@FAR= 1%, and TAR@FAR= 10%. EER is computed at the operating point where the False Match Rate (FMR) equals the False Non-Match Rate (FNMR). The proposed method is compared with handcrafted and deep learning-based finger vein verification methods, including MCP [41], RLT [41], WLD [41], multiple attention [29], deep fusion [22], self-attention [31], ECA-ResNet with arccosine loss [17], vision transformers [30], and the localglobal feature interaction network [34]. Training details for all baselines are provided in the supplementary material. The experiments are conducted on five publicly available finger vein datasets to evaluate both cross-dataset generalisation and open-set verification behaviour.

## 3.1 Datasets

The experiments utilise five publicly available finger vein datasets: FV-300 (A) [31], MMCBNU-6000 (B) [44], FV-USM (C) [45], PolyU (D) [46], and VERA (E) [47]. A summary of these datasets is provided in Table 3. A brief description of each dataset is given below.

FV-300 [31]: This dataset contains 300 distinct finger vein patterns collected from 75 subjects. Images were captured using a custom monochromatic CMOS camera with two lighting sources. For each subject, the index and middle fingers of both hands were imaged across multiple sessions, with intervals ranging from 1 to 4 days. Among the datasets used in this study, FV-300 provides the highest image quality. For our experiments, a total of 124 images were removed during preprocessing due to severe blur, poor illumination, or failed vein visibility, as these conditions prevent reliable extraction of vein structures.

MMCBNU-6000 [44]: Developed by Chonbuk National University, this dataset includes 600 distinct finger vein patterns from 100 individuals. Acquisition was performed using a camera equipped with an NIR-pass filter and an array of 850 nm infrared LEDs. For each subject, the index, middle, and ring fingers of both hands were captured ten times. The dataset includes participants of diverse nationalities, age groups, blood types, and genders.

FV-USM [45]: Collected at Universiti Sains Malaysia, this dataset comprises 492 distinct finger vein patterns from 123 subjects, including 83 males and 40 females aged between 20 and 52. Each subject contributed images of the index and middle fingers of both hands. Data were acquired in two sessions separated by a two-week interval.

PolyU [46]: This dataset contains 312 distinct finger vein patterns from 156 subjects, captured using a peg-free and unconstrained imaging setup. Data acquisition spanned an 11-month period. In this study, we use version 1 of the dataset, which consists of 218 unique finger vein patterns, with four images per finger.

VERA [47]: This dataset was acquired using an open finger vein sensor and contains 220 distinct finger vein patterns from 110 subjects. Each subject presented the index fingers of both hands, with two samples captured per finger at a 5-minute interval. The dataset includes 40 female and 70 male participants aged between 18 and 60, with a mean age of 33 years.

## 3.2 Evaluation Protocol

We evaluate OpenVeinNet using a leave-one-dataset-out strategy. In each experiment, the model is trained on four datasets and evaluated on the remaining held-out dataset. Therefore, all identities in the evaluation dataset are unseen during training. This setting allows us to assess both crossdataset generalisation and open-set verification behaviour under sensor, subject, and acquisition-domain shifts.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Classes</td><td rowspan=1 colspan=1>Images per identity</td><td rowspan=1 colspan=1>Total Images</td></tr><tr><td rowspan=1 colspan=1>FV-300 (A) [31]</td><td rowspan=1 colspan=1>300</td><td rowspan=1 colspan=1>92*</td><td rowspan=1 colspan=1>27,600*</td></tr><tr><td rowspan=1 colspan=1>MMCBNU (B) [44]</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6,000</td></tr><tr><td rowspan=1 colspan=1>FV-USM (C) [45]</td><td rowspan=1 colspan=1>492</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>5,904</td></tr><tr><td rowspan=1 colspan=1>PolyU (D) [46]</td><td rowspan=1 colspan=1>218</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>872</td></tr><tr><td rowspan=1 colspan=1>VERA (E) [47]</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>440</td></tr></table>

TABLE 3: Summary of the five public finger vein datasets used in this study. \*A total of 124 low-quality FV-300 images were removed due to severe blur, poor illumination, or failed vein visibility.\*
<table><tr><td rowspan=2 colspan=1>Dataset</td><td rowspan=1 colspan=2>Genuine Scores</td><td rowspan=1 colspan=2>Impostor Scores</td></tr><tr><td rowspan=1 colspan=1>Half</td><td rowspan=1 colspan=1>Full</td><td rowspan=1 colspan=1>Half</td><td rowspan=1 colspan=1>Full</td></tr><tr><td rowspan=1 colspan=1>FV-300 (A) [31]</td><td rowspan=1 colspan=1>598,279</td><td rowspan=1 colspan=1>1,198,364</td><td rowspan=1 colspan=1>181,709,824</td><td rowspan=1 colspan=1>362,208,956</td></tr><tr><td rowspan=1 colspan=1>MMCBNU (B) [44]</td><td rowspan=1 colspan=1>13,500</td><td rowspan=1 colspan=1>27,000</td><td rowspan=1 colspan=1>9,000,000</td><td rowspan=1 colspan=1>17,970,000</td></tr><tr><td rowspan=1 colspan=1>FV-USM (C) [45]</td><td rowspan=1 colspan=1>16,236</td><td rowspan=1 colspan=1>32,472</td><td rowspan=1 colspan=1>8,714,304</td><td rowspan=1 colspan=1>17,393,184</td></tr><tr><td rowspan=1 colspan=1>PolyU (D) [46]</td><td rowspan=1 colspan=1>1,562</td><td rowspan=1 colspan=1>3,107</td><td rowspan=1 colspan=1>405,765</td><td rowspan=1 colspan=1>807,794</td></tr><tr><td rowspan=1 colspan=1>VERA (E) [47]</td><td rowspan=1 colspan=1>110</td><td rowspan=1 colspan=1>220</td><td rowspan=1 colspan=1>48,400</td><td rowspan=1 colspan=1>96,360</td></tr></table>

TABLE 4: Number of genuine and impostor scores used for the half-subject and full-subject verification protocols.

We consider two complementary evaluation protocols.

Protocol 1: Enrollment-based unknown-rejection evaluation. The held-out evaluation dataset is deterministically split into two disjoint subject groups. The first 50% of subjects are treated as the enrolled set, while the remaining 50% are treated as non-enrolled subjects. Genuine scores are computed from all within-subject pairwise cosine similarities among the enrolled subjects only. Impostor scores are computed from all cross-subject pairwise cosine similarities between enrolled-subject samples and non-enrolled-subject samples. Threshold-dependent metrics are then obtained by sweeping the decision threshold over these score distributions.

Protocol 2: Full-subject verification evaluation. In the implemented full-subject protocol, all held-out subjects are included in evaluation and genuine/impostor scores are computed from all possible within-subject and cross-subject pairwise cosine similarities over the held-out set. Thus, the reported FAR/FRR, EER, AUC, and TAR values are derived from exhaustive pairwise score distributions rather than from a separate gallery-template matching stage.

For both protocols, genuine and impostor scores are computed using cosine similarity between the extracted embeddings. Genuine scores are generated by comparing samples belonging to the same finger identity, excluding selfcomparisons. Impostor scores are generated by comparing samples from different finger identities. Unless otherwise stated, all possible genuine and impostor pairs are used for evaluation without random pair sampling. Performance is reported using AUC, EER, and TAR at fixed FAR values. The threshold-based operating points, TAR@FAR= 0.1%, $\mathrm { T A R @ F A R = \ 1 \% }$ and $\mathrm { \hat { T } A R } @ \mathrm { F A } \mathrm { \bar { R } = \ 1 0 \% }$ , are included because practical biometric systems are deployed at fixed decision thresholds rather than only at the equal-error operating point.

Cross-validation and confidence intervals. To improve statistical reliability, we repeat each leave-onedataset-out experiment over multiple predefined statistical splits indexed by stat\_seed. For each stat\_seed, the corresponding dataset partition is loaded from data/<dataset>/<stat\_seed>, and all methods are evaluated on the same held-out dataset using the same score-generation protocol. The final results are reported as the mean and 95% confidence interval (CI) across the available matched runs for each protocol.

## 3.3 Results and Discussion

Tables 5 and 6 present the results under the enrollmentbased unknown-rejection and full-subject verification protocols, respectively. The results are reported across five leaveone-dataset-out settings. Overall, the proposed method achieves strong and stable performance across both protocols, demonstrating that the learned embeddings generalise well to unseen identities and different acquisition conditions.

Enrollment-based unknown-rejection performance. Table 5 reports the stricter open-set protocol, where only half of the subjects from the held-out dataset are enrolled and the remaining subjects are treated as unknown identities. Under this setting, the proposed method achieves the best or highly competitive performance across most evaluation splits. In the challenging ABDE → C setting, OpenVein-Net obtains an AUC of 97.72% and an EER of 7.48%, clearly outperforming the deep learning baselines. At the operating-point level, it achieves 53.52% TAR at FAR= 1% and 95.47% TAR at $\mathrm { F A R } = 1 0 \% ,$ showing that the learned embeddings support reliable acceptance of enrolled users while maintaining rejection capability for unknown probes.

A similar trend is observed for $A B C E  D ,$ where the proposed method achieves the best AUC (98.67%), lowest EER (5.34%), and the strongest TAR values across the reported operating points. These results are important because they directly address the unknown-rejection setting: the method is not only evaluated on unseen subjects but also tested for its ability to reject non-enrolled identities at fixed thresholds.

Full-subject verification performance. Table 6 reports the results when all subjects from the held-out dataset are enrolled. In this setting, the proposed method achieves the lowest EER in 4 of 5 evaluation splits and is competitive in the 5th one. For example, in the BCDE → A setting, OpenVeinNet achieves an EER of 3.17%, compared with 6.20% for VeinAttNet and substantially higher EER values for other deep learning baselines. In the more difficult $A B D E  { \dot { C } }$ and $A \bar { B C } E  D$ settings, the proposed method obtains EER values of 7.44% and 5.21%, respectively, while also achieving strong TAR values at low FAR thresholds. These results show that the embedding space learned by OpenVeinNet remains discriminative under substantial dataset shift.

Behaviour at strict operating points. From a deployment perspective, the low-FAR regime is particularly important because false accepts directly affect security. The proposed method generally achieves higher TAR values at FAR= 0.1% and FAR= 1% than the deep learning baselines, especially in the challenging $A B D E  C$ and $A B C E  \bar { D }$ protocols. Handcrafted methods such as MCP and WLD can perform strongly on clean and well-controlled data, particularly in $B C D \bar { E } \  \ A ,$ , but their performance degrades considerably under stronger domain shift. This suggests that handcrafted vessel-enhancement pipelines can be effective when the target data are clean, but their robustness is limited when image quality, sensor characteristics, and acquisition conditions vary.

<table><tr><td>Train → Test Dataset BCDE → A</td><td>Algorithm MCP [35]</td><td>AUC (%) 99.80</td><td>EER (%) 0.89</td><td>TAR@FAR= 0.1% (%) 40.19</td><td>TAR@FAR= 1% (%) 99.47</td><td>TAR@FAR= 10% (%) 100.00</td></tr><tr><td rowspan="9"></td><td>RLT [36]</td><td>99.78</td><td>0.95</td><td>30.00</td><td>99.24</td><td>100.00</td></tr><tr><td>WLD [37]</td><td>99.82</td><td>0.84</td><td>43.82</td><td>99.59</td><td>100.00</td></tr><tr><td>ArcVein [17]</td><td>89.39 (82.64–96.14)</td><td>19.49 (11.45–27.54)</td><td>6.67 (0.00–16.88)</td><td>22.81 (2.86–42.77)</td><td>61.22 (35.63–86.81)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LGFIN [34]</td><td>85.52 (83.95–87.08)</td><td>23.35 (22.54–24.15)</td><td>4.23 (0.00–8.51)</td><td>15.12 (6.50–23.74)</td><td>50.09 (41.24–58.95)</td></tr><tr><td>FV-ViT [30]</td><td>89.72 (88.31–91.13)</td><td>17.74 (16.16–19.33)</td><td>0.95 (0.00–2.12)</td><td>6.34 (3.56–9.12)</td><td>58.68 (51.23–66.13)</td></tr><tr><td>VeinAttNet [31]</td><td>97.83 (95.75–99.91)</td><td>6.56 (2.64–10.48)</td><td>21.55 (0.00–54.93)</td><td>50.87 (12.87–88.86)</td><td>96.75 (91.84–100.00)</td></tr><tr><td>Proposed Method</td><td>99.47 (99.19–99.75)</td><td>3.49 (2.47–4.51)</td><td>58.11 (42.09–74.13)</td><td>85.03 (76.76–93.30)</td><td>99.80 (99.53–100.00)</td></tr><tr><td>MCP [35]</td><td>92.90</td><td>12.31</td><td>0.04</td><td>3.29</td><td>77.65</td></tr><tr><td rowspan="9"></td><td>RLT [36]</td><td>90.31</td><td>13.76</td><td>0.00</td><td>0.04</td><td>57.20</td></tr><tr><td>WLD [37]</td><td>87.76</td><td>16.98</td><td>0.00</td><td>0.06</td><td>37.44</td></tr><tr><td>ArcVein [17]</td><td>91.71 (86.06–97.37)</td><td>15.61 (8.81–22.41)</td><td>0.99 (0.00–2.30)</td><td>18.34 (5.65–31.02)</td><td>72.04 (48.42–95.66)</td></tr><tr><td>LGFIN [34]</td><td>92.48 (91.94–93.02)</td><td>15.34 (14.66–16.02)</td><td>5.26 (1.47–9.06)</td><td>26.98 (25.83–28.13)</td><td>75.30 (72.60–77.99)</td></tr><tr><td>FV-ViT [30]</td><td>89.70 (87.28–92.12)</td><td>18.03 (15.65–20.41)</td><td>1.94 (0.82–3.06)</td><td>14.97 (7.27–22.68)</td><td>64.48 (54.60–74.36)</td></tr><tr><td>VeinAttNet [31]</td><td>99.06 (98.74–99.39)</td><td>4.35 (3.82–4.88)</td><td>39.09 (27.25–50.92)</td><td>78.50 (69.77–87.22)</td><td>98.79 (98.37–99.21)</td></tr><tr><td>Proposed Method</td><td>99.12 (98.86–99.38)</td><td>3.99 (3.29–4.70)</td><td>28.13 (19.96–36.30)</td><td>74.71 (67.50–81.91)</td><td>99.51 (99.18–99.84)</td></tr><tr><td>MCP [35]</td><td>87.01</td><td>18.93</td><td>0.01</td><td>0.55</td><td>35.11</td></tr><tr><td>RLT [36]</td><td>79.40</td><td>25.14</td><td>0.00</td><td>0.00</td><td>4.91</td></tr><tr><td rowspan="9"></td><td>WLD [37]</td><td>83.07</td><td>22.87</td><td>0.00</td><td>0.08</td><td>20.58</td></tr><tr><td>ArcVein [17]</td><td>87.51 (84.43–90.58)</td><td>20.22 (16.79–23.64)</td><td>2.02 (0.00–4.36)</td><td>7.84 (3.90–11.78)</td><td>52.75 (41.61–63.90)</td></tr><tr><td>LGFIN [34]</td><td>87.78 (86.39–89.17)</td><td>18.34 (14.20–22.47)</td><td>0.81 (0.00–2.13)</td><td>4.72 (0.00–10.76)</td><td>47.65 (15.14–80.17)</td></tr><tr><td>FV-ViT [30]</td><td>89.66 (88.43–90.89)</td><td>18.24 (17.19–19.28)</td><td>2.23 (0.95–3.51)</td><td>15.78 (8.26–23.30)</td><td>65.69 (61.22–70.17)</td></tr><tr><td>VeinAttNet [31]</td><td>86.79 (85.37–88.22)</td><td>22.20 (20.14–24.26)</td><td>8.01 (5.92–10.10)</td><td>15.23 (11.25–19.21)</td><td>49.21 (46.92–51.50)</td></tr><tr><td>Proposed Method</td><td>97.72 (97.28–98.17)</td><td>7.48 (6.70–8.25)</td><td>13.71 (3.69–23.72)</td><td>53.52 (47.25–59.79)</td><td>95.47 (93.72–97.21)</td></tr><tr><td>MCP [35]</td><td>95.77</td><td>9.90</td><td>3.83</td><td>21.88</td><td>90.35</td></tr><tr><td>RLT [36]</td><td>88.17</td><td>16.51</td><td>0.00</td><td>0.07</td><td>34.81</td></tr><tr><td>WLD [37]</td><td>89.72</td><td>15.60</td><td>0.00</td><td>0.86</td><td>48.95</td></tr><tr><td rowspan="9"></td><td>ArcVein [17]</td><td>83.78 (82.07–85.48)</td><td>24.18 (22.21–26.14)</td><td>1.43 (0.00–5.53)</td><td>5.44 (1.23–9.66)</td><td>46.83 (38.51–55.15)</td></tr><tr><td>LGFIN [34]</td><td>93.95 (92.23–95.66)</td><td>13.41 (11.36–15.45)</td><td>15.41 (10.38–20.43)</td><td>32.25 (18.67–45.83)</td><td>79.50 (71.57–87.43)</td></tr><tr><td>FV-ViT [30]</td><td>87.65 (85.36–89.93)</td><td>20.32 (18.21–22.42)</td><td>0.67 (0.00–1.73)</td><td>11.18 (2.88–19.48)</td><td>59.39 (52.00–66.78)</td></tr><tr><td>VeinAttNet [31]</td><td>97.90 (97.34–98.47)</td><td>7.72 (6.71–8.72)</td><td>24.96 (0.00–52.09)</td><td>67.50 (58.20–76.79)</td><td>94.53 (93.02–96.05)</td></tr><tr><td>Proposed Method</td><td>98.67 (98.32–99.02)</td><td>5.34 (4.40–6.29)</td><td>25.65 (11.73–39.56)</td><td>67.68 (60.38–74.98)</td><td>98.39 (97.81–98.97)</td></tr><tr><td>MCP [35]</td><td>72.21</td><td>33.19</td><td>0.00</td><td>0.04</td><td>11.14</td></tr><tr><td>RLT [36]</td><td>59.21</td><td>44.43</td><td>0.00</td><td>0.00</td><td>1.11</td></tr><tr><td>WLD [37]</td><td>69.67</td><td>35.45</td><td>0.00</td><td>0.02</td><td>8.51</td></tr><tr><td>ArcVein [17]</td><td>71.51 (69.25–73.77)</td><td>33.66 (31.29–36.04)</td><td>0.00 (0.00–0.00)</td><td>0.38 (0.05–0.71)</td><td>18.16 (15.53–20.79)</td></tr><tr><td rowspan="9"></td><td></td><td>75.88 (74.57–77.20)</td><td>32.29 (31.43–33.14)</td><td>0.00 (0.00–0.00)</td><td>9.23 (0.00–20.97)</td><td>32.59 (23.47–41.70)</td></tr><tr><td>LGFIN [34]</td><td></td><td></td><td></td><td>4.10 (0.00–8.95)</td><td>30.29 (23.60–36.98)</td></tr><tr><td>FV-ViT [30]</td><td>74.34 (72.58–76.09)</td><td>32.09 (28.61–35.57)</td><td>0.00 (0.00–0.00)</td><td></td><td></td></tr><tr><td>VeinAttNet [31]</td><td>89.73 (88.62–90.83)</td><td>18.19 (17.04–19.35)</td><td>0.00 (0.00–0.00)</td><td>13.49 (0.63–26.36)</td><td>67.34 (56.41–78.27)</td></tr><tr><td>Proposed Method</td><td>91.32 (87.69–94.95)</td><td>16.64 (13.57–19.71)</td><td>0.00 (0.00–0.00)</td><td>13.70 (0.00–31.29)</td><td>66.40 (45.03–87.76)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE 5: Half-subject threshold-based operating-point results.

One exception to the above trend is observed in the ACDE→B protocol at FAR=0.1%, where VeinAttNet achieves a higher TAR of 39.09% compared with 28.13% for the proposed method. This can be explained by the score distribution behaviour on MMCBNU, which has the highest gradient coherence score among all five datasets (0.5398, Table 7), reflecting strongly defined and structurally consistent vein patterns. Under such controlled acquisition conditions, VeinAttNet’s attention-based architecture produces genuine scores that are more tightly concentrated near the upper end of the cosine similarity range, which is advantageous specifically at the strict FAR=0.1% threshold. The graphbased embeddings of OpenVeinNet produce a smoother score distribution that is better calibrated across a wider range of operating points, as evidenced by the lower EER (3.99% vs. 4.35%) and competitive TAR at FAR=1% and FAR=10%. This trade-off between strict threshold performance and overall operating-point calibration is consistent with the dataset quality analysis in Section 3.4.

Challenging dataset analysis. The ABCD → E setting is the most challenging protocol because VERA contains only two images per finger identity/class and exhibits higher acquisition variability. All methods show reduced performance in this setting. Nevertheless, the proposed method achieves the best AUC and EER in both evaluation protocols. At FAR= 0.1%, the TAR is zero for all methods, indicating that extremely strict operating points are difficult to realise reliably for this low-sample dataset. This should be interpreted with care: the small number of genuine comparisons in VERA leads to a coarse score distribution, which limits score resolution at very low FAR thresholds. Hence, the zero TAR at FAR= 0.1% reflects the limited score resolution and strict operating threshold rather than a complete failure of the learned representation. Importantly, OpenVeinNet maintains competitive or best performance at FAR= 1% and FAR= 10%, demonstrating better relative robustness under severe domain shift.

Comparison with state-of-the-art methods. Compared with recent deep learning approaches such as ArcVein, LGFIN, FV-ViT, and VeinAttNet, the proposed method provides a stronger overall trade-off between AUC, EER, and threshold-based TAR. While VeinAttNet is competitive in some settings, such as ACDE → B, its performance is less stable across all held-out datasets. In contrast, OpenVeinNet achieves consistently low EER and strong TAR across both protocols, demonstrating more reliable generalisation.

Technical justification of performance gains. The improvement of OpenVeinNet can be attributed to the complementarity of its three main components. First, DSConv adaptively samples along curvilinear structures and therefore captures local tubular vein evidence more effectively than fixed-grid convolution. Second, the graph-based backbone models long-range relationships between vein regions, enabling the network to encode global vascular topology. Third, the Centroid Angular Hybrid Loss encourages compact intra-class embeddings and angularly separated interclass representations, which improves cosine-similaritybased verification under open-set conditions.

Overall, the experimental results show that OpenVein-Net improves cross-dataset generalisation and provides strong open-set verification capability. The consistent gains in EER and threshold-based TAR across several held-out datasets support the effectiveness of combining adaptive tubular feature extraction, graph-based relational modelling, and angularly constrained embedding learning.

## 3.4 Dataset Quality Analysis and Performance Interpretation

To better understand dataset-specific performance variations observed in Section 3.3, we analyse the intrinsic quality of the datasets using a set of handcrafted imagequality measures. The goal of this analysis is not to define a universal quality metric, but to provide consistent and interpretable indicators of structural clarity, continuity, and contrast across datasets.

For a grayscale image I defined over pixel domain Ω, we compute four complementary measures:

$$
Q _ { \mathrm { g r a d } } ( I ) = \frac { 1 } { | \Omega | } \sum _ { p \in \Omega } \sqrt { G _ { x } ( p ) ^ { 2 } + G _ { y } ( p ) ^ { 2 } } ,\tag{18}
$$

$$
Q _ { \mathrm { c o h } } ( I ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \left( \frac { \sqrt { ( j _ { 1 1 } ^ { ( b ) } - j _ { 2 2 } ^ { ( b ) } ) ^ { 2 } + 4 ( j _ { 1 2 } ^ { ( b ) } ) ^ { 2 } } } { j _ { 1 1 } ^ { ( b ) } + j _ { 2 2 } ^ { ( b ) } + \varepsilon } \right) ^ { 2 } ,\tag{19}
$$

$$
Q _ { \mathrm { l a p } } ( I ) = \mathrm { V a r } ( \Delta I ) ,\tag{20}
$$

$$
Q _ { \mathrm { c o n } } ( I ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \sigma _ { b } ,\tag{21}
$$

where $G _ { x }$ and $G _ { y }$ denote Sobel gradients, $( j _ { 1 1 } ^ { ( b ) } , j _ { 2 2 } ^ { ( b ) } , j _ { 1 2 } ^ { ( b ) } )$ are structure-tensor components computed over block $b , \Delta I$ represents the Laplacian response, and $\sigma _ { b }$ is the standard deviation within block b. In all cases, higher values indicate stronger structural visibility or contrast.

The results in Table 7 provide useful insights into the observed performance trends. As observed in Tables 5 and 6, handcrafted methods such as MCP and WLD outperform several deep learning baselines in the $B C D E  A$ setting. First, the $\bar { B C D E }  \bar { A }$ setting corresponds to evaluation on FV-300, which exhibits relatively high structural quality and, importantly, a significantly larger number of samples per subject compared to other datasets. Under such controlled and high-quality conditions, handcrafted methods based on vessel enhancement and template matching remain highly competitive, as the vein structures are already well-defined and less affected by noise or variability. Therefore, the strong performance of handcrafted approaches in this setting does not contradict the overall findings, but rather highlights that simpler methods can perform well when the data distribution is favourable.

In contrast, the performance drop observed in the $A B C D  E$ setting is consistent with both dataset characteristics and the quality analysis. VERA exhibits the lowest gradient-coherence among all datasets, indicating weaker structural continuity of vein patterns, and contains only two images per identity. This results in limited intra-class representation and a sparse score distribution during evaluation. Furthermore, the open-sensor acquisition setup introduces additional variability, increasing the domain shift relative to the training data.

As a result, the $A B C D \ \to \ E$ protocol represents a challenging low-quality, low-sample, and high-variability scenario. The reduced performance across all methods in this setting is therefore expected and reflects the inherent difficulty of the dataset rather than a limitation specific to the proposed approach.

Overall, this analysis clarifies that the observed performance variations are primarily driven by dataset characteristics, including image quality, sample density, and acquisition variability. These findings support a more precise interpretation of the experimental results, where the proposed method demonstrates strong generalisation across diverse conditions while remaining robust under challenging crossdataset scenarios.

Statistical significance: To validate that the observed improvements are not due to random variation, we performed paired statistical significance testing using a twosided sign-flip permutation test on full-subject EER across matched runs. The proposed method shows statistically significant improvements over all competing deep learning baselines after Holm correction $( p < 1 \bar { 0 } ^ { - 3 } )$ . Detailed results are provided in the supplementary material.

## 4 ABLATION STUDY

In this section, we present a consolidated ablation study to justify the main design choices of OpenVeinNet. The ablations analyse: (i) the kernel-size configuration of the DSConv stem, (ii) the number of Grapher Blocks in the backbone, (iii) the contribution of the proposed Centroid Angular Hybrid (CAH) loss compared with standard losses, (iv) its comparison against strong angular-margin losses, (v) the sensitivity of the balancing factor $\beta ,$ and (vi) the individual contributions of DSConv and graph-based modelling. Unless otherwise stated, the ablation experiments are conducted under the $A B D E  C$ protocol (Full-subject), where FV-USM is used as the held-out evaluation dataset. Lower EER indicates better verification performance.

## 4.1 Effect of DSConv Kernel Sizes

Table 8 shows the effect of different kernel-size configurations in the DSConv stem. Uniformly small kernels such as (3, 3, 3) and (5, 5, 5) are less effective, suggesting that very local receptive fields are insufficient to capture the extended curvilinear structure of finger veins. Uniformly large kernels improve the result, but do not provide the best performance. The best EER is obtained using the progressively decreasing configuration (9, 7, 3), which achieves an EER of 10.82%.

This trend is technically meaningful. Larger kernels in the shallow layers help capture broader vascular structures and long vein segments, while smaller kernels in deeper layers refine local discriminative details. Therefore, the (9, 7, 3) configuration provides a better balance between global vein continuity and fine local pattern representation.

<table><tr><td>Train → Test Dataset</td><td>Algorithm</td><td>AUC (%)</td><td>EER (%)</td><td>TAR@FAR= 0.1% (%)</td><td>TAR@FAR= 1% (%)</td><td>TAR@FAR= 10% (%)</td></tr><tr><td>BCDE → A</td><td>MCP [35] RLT [36]</td><td>99.80 99.78</td><td>0.89 0.95</td><td>40.19 30.00</td><td>99.47 99.24</td><td>100.00 100.00</td></tr><tr><td></td><td></td><td>99.82</td><td>0.84</td><td>43.82</td><td>99.59</td><td></td></tr><tr><td></td><td>WLD [37]</td><td></td><td></td><td></td><td></td><td>100.00</td></tr><tr><td></td><td>ArcVein [17]</td><td>89.77 (83.13–96.42)</td><td>19.38 (10.81–27.95)</td><td>6.63 (1.15–12.10)</td><td>24.80 (5.88–43.72)</td><td>63.72 (39.50–87.95)</td></tr><tr><td></td><td>LGFIN [34]</td><td>85.65 (84.32–86.99)</td><td>23.01 (22.30–23.72)</td><td>3.28 (1.06–5.50)</td><td>14.68 (6.02–23.35)</td><td>50.09 (42.60–57.59)</td></tr><tr><td></td><td>FV-ViT [30]</td><td>89.87 (88.51–91.23)</td><td>17.37 (15.84–18.89)</td><td>1.04 (0.00–2.11)</td><td>6.70 (3.57–9.84)</td><td>59.09 (51.73–66.46)</td></tr><tr><td></td><td>VeinAttNet [31]</td><td>98.10 (96.36–99.84)</td><td>6.20 (2.53–9.87)</td><td>20.70 (0.00–50.91)</td><td>56.05 (21.04–91.05)</td><td>97.46 (93.71–100.00)</td></tr><tr><td>ACDE → B</td><td>Proposed Method</td><td>99.53 (99.31–99.74)</td><td>3.17 (2.12–4.22)</td><td>53.52 (43.47–63.57)</td><td>87.33 (80.87–93.78)</td><td>99.86 (99.67–100.00)</td></tr><tr><td></td><td>MCP [35]</td><td>92.90</td><td>12.31</td><td>0.04</td><td>3.29</td><td>77.65</td></tr><tr><td></td><td>RLT [36]</td><td>90.31</td><td>13.76</td><td>0.00</td><td>0.04</td><td>57.20</td></tr><tr><td></td><td>WLD [37]</td><td>87.76</td><td>16.98</td><td>0.00</td><td>0.06</td><td>37.44</td></tr><tr><td></td><td>ArcVein [17]</td><td>91.74 (86.43–97.04)</td><td>15.47 (9.11–21.83)</td><td>1.49 (0.04–2.94)</td><td>16.02 (5.75–26.28)</td><td>72.07 (49.36–94.78)</td></tr><tr><td></td><td>LGFIN [34]</td><td>92.55 (91.90–93.21)</td><td>15.12 (14.27–15.97)</td><td>5.31 (3.33–7.30)</td><td>25.08 (22.97–27.20)</td><td>75.64 (72.46–78.82)</td></tr><tr><td></td><td>FV-ViT [30]</td><td>89.87 (87.55–92.20)</td><td>17.74 (15.41–20.08)</td><td>1.86 (0.77–2.95)</td><td>13.17 (7.55–18.79)</td><td>65.35 (55.79–74.92)</td></tr><tr><td></td><td>VeinAttNet [31]</td><td>99.21 (98.92–99.50)</td><td>4.07 (3.35–4.79)</td><td>46.07 (32.67–59.48)</td><td>82.77 (75.98–89.57)</td><td>98.89 (98.49–99.28)</td></tr><tr><td></td><td>Proposed Method</td><td>99.17 (98.88–99.45)</td><td>3.91 (3.02–4.80)</td><td>29.60 (27.53–31.66)</td><td>76.96 (69.52–84.41)</td><td>99.53 (99.12–99.93)</td></tr><tr><td>ABDE → C</td><td>MCP [35]</td><td>87.01</td><td>18.93</td><td>0.01</td><td>0.55</td><td>35.11</td></tr><tr><td></td><td>RLT [36]</td><td>79.40</td><td>25.14</td><td>0.00</td><td>0.00</td><td>4.91</td></tr><tr><td></td><td>WLD [37]</td><td>83.07</td><td>22.87</td><td>0.00</td><td>0.08</td><td>20.58</td></tr><tr><td></td><td>ArcVein [17]</td><td>88.09 (84.76–91.42)</td><td>19.74 (15.89–23.60)</td><td>1.55 (0.11–2.98)</td><td>8.65 (5.88–11.42)</td><td>55.14 (42.22–68.07)</td></tr><tr><td></td><td>LGFIN [34]</td><td>87.76 (86.73–88.80)</td><td>18.51 (14.31–22.72)</td><td>0.45 (0.00–1.27)</td><td>4.20 (0.00–9.20)</td><td>47.31 (14.97–79.66)</td></tr><tr><td></td><td>FV-ViT [30]</td><td>88.76 (87.44–90.08)</td><td>19.04 (17.80–20.28)</td><td>1.16 (0.00–2.42)</td><td>11.87 (6.76–16.97)</td><td>62.83 (58.00–67.65)</td></tr><tr><td></td><td>VeinAttNet [31]</td><td>86.93 (85.26–88.60)</td><td>22.02 (19.66–24.37)</td><td>6.07 (2.52–9.62)</td><td>15.45 (10.92–19.98)</td><td>50.11 (45.21–55.01)</td></tr><tr><td>ABCE → D</td><td>Proposed Method</td><td>97.78 (97.30–98.25)</td><td>7.45 (6.52–8.38)</td><td>16.82 (7.18–26.45)</td><td>55.77 (50.67–60.86)</td><td>95.39 (93.58–97.20)</td></tr><tr><td></td><td>MCP [35]</td><td>95.77</td><td>9.90</td><td>3.83</td><td>21.88</td><td>90.35</td></tr><tr><td></td><td>RLT [36]</td><td>88.17</td><td>16.51 15.60</td><td>0.00</td><td>0.07</td><td>34.81</td></tr><tr><td></td><td>WLD [37]</td><td>89.72</td><td></td><td>0.00</td><td>0.86</td><td>48.95</td></tr><tr><td></td><td>ArcVein [17]</td><td>84.16 (81.10–87.23)</td><td>23.85 (20.43–27.27)</td><td>0.56 (0.00–1.61)</td><td>6.57 (3.28–9.86)</td><td>49.40 (40.36–58.44)</td></tr><tr><td></td><td>LGFIN [34]</td><td>93.65 (91.84–95.46)</td><td>13.63 (11.45–15.82)</td><td>13.83 (10.69–16.97)</td><td>28.09 (19.74–36.44)</td><td>78.94 (71.26–86.61)</td></tr><tr><td></td><td>FV-ViT [30]</td><td>88.57 (86.33–90.80)</td><td>19.26 (17.28–21.24)</td><td>2.22 (0.00–4.91)</td><td>12.93 (5.81–20.06)</td><td>62.53 (55.11–69.96)</td></tr><tr><td></td><td>VeinAttNet [31]</td><td>97.54 (96.99–98.10)</td><td>8.17 (7.08–9.26)</td><td>17.29 (0.00–38.14)</td><td>57.93 (49.38–66.47)</td><td>93.71 (91.71–95.72)</td></tr><tr><td>ABCD → E</td><td>Proposed Method</td><td>98.80 (98.43–99.17)</td><td>5.20 (4.36–6.04)</td><td>29.97 (15.63–44.31)</td><td>72.23 (63.18–81.28)</td><td>98.35 (97.68–99.02)</td></tr><tr><td></td><td>MCP [35]</td><td>72.21</td><td>33.19</td><td>0.00</td><td>0.04</td><td>11.14</td></tr><tr><td></td><td>RLT [36]</td><td>59.21</td><td>44.43</td><td>0.00</td><td>0.00</td><td>1.11</td></tr><tr><td></td><td>WLD [37]</td><td>69.67</td><td>35.45</td><td>0.00</td><td>0.02</td><td>8.51</td></tr><tr><td></td><td>ArcVein [17]</td><td>69.52 (66.76–72.29)</td><td>35.56 (33.84–37.28)</td><td>0.00 (0.00–0.00)</td><td>1.11 (0.65–1.58)</td><td>17.31 (15.56–19.07)</td></tr><tr><td></td><td>LGFIN [34]</td><td>74.70 (73.03–76.37)</td><td>33.16 (31.68–34.63)</td><td>0.00 (0.00–0.00)</td><td>5.69 (0.00–14.49)</td><td>31.50 (23.59–39.41)</td></tr><tr><td></td><td>FV-ViT [30]</td><td>73.16 (70.55–75.76)</td><td>33.06 (29.81–36.32)</td><td>0.00 (0.00–0.00)</td><td>3.99 (0.02–7.96)</td><td>31.22 (26.66–35.78)</td></tr><tr><td></td><td>VeinAttNet [31]</td><td>88.43 (87.59–89.27)</td><td>19.90 (18.18–21.62)</td><td>0.00 (0.00–0.00)</td><td>18.27 (7.66–28.88)</td><td>61.01 (56.22–65.80)</td></tr><tr><td></td><td>Proposed Method</td><td>89.98 (86.88–93.07)</td><td>18.43 (15.70–21.16)</td><td>0.00 (0.00–0.00)</td><td>15.86 (2.85–28.86)</td><td>64.68 (51.47–77.88)</td></tr></table>

TABLE 6: Full-subject threshold-based operating-point results.

<table><tr><td>Dataset</td><td>Gradient</td><td>Grad. Coherence</td><td>Laplacian</td><td>Contrast</td><td>Images/class</td></tr><tr><td>FV-300 (A) [31]</td><td>0.0687</td><td>0.3675</td><td>0.0609</td><td>0.0311</td><td>89.57</td></tr><tr><td>MMCBNU (B) [44]</td><td>0.0781</td><td>0.5398</td><td>0.0026</td><td>0.0392</td><td>10.00</td></tr><tr><td>FV-USM (C) [45]</td><td>0.0312</td><td>0.2928</td><td>0.0009</td><td>0.0137</td><td>12.00</td></tr><tr><td>PolyU (D) [46]</td><td>0.0478</td><td>0.4605</td><td>0.0017</td><td>0.0249</td><td>5.84</td></tr><tr><td>VERA (E) [47]</td><td>0.0476</td><td>0.1429</td><td>0.0074</td><td>0.0196</td><td>2.00</td></tr></table>

TABLE 7: Mean fingervein image-quality indicators computed across datasets. The metrics quantify gradient strength, structural coherence, edge sharpness (Laplacian variance), and local contrast. Higher values indicate better visibility and continuity of vein structures. The number of images per subject is also reported to highlight dataset density, which impacts verification reliability.
<table><tr><td rowspan=1 colspan=1>Kernel set</td><td rowspan=1 colspan=1>EER (%)</td></tr><tr><td rowspan=7 colspan=1>(3, 3, 3)(5, 5, 5)(7,7,7)(9, 9, 9)(9, 7,5)(7, 5, 3)(9, 5, 3)(9, 7, 3)</td><td rowspan=1 colspan=1>12.335112.4453</td></tr><tr><td rowspan=1 colspan=1>11.8358</td></tr><tr><td rowspan=1 colspan=1>11.8376</td></tr><tr><td rowspan=1 colspan=1>11.0605</td></tr><tr><td rowspan=1 colspan=1>13.0697</td></tr><tr><td rowspan=1 colspan=1>11.2155</td></tr><tr><td rowspan=1 colspan=1>10.8235</td></tr></table>

TABLE 8: Ablation on the kernel sizes used sequentially in the DSConv Blocks of the stem.

## 4.2 Effect of the Number of Grapher Blocks

The number of Grapher Blocks controls the depth of graphbased message passing in the backbone. A shallow graph backbone may not propagate sufficient relational information between vein regions, while an overly deep graph backbone may introduce redundancy, over-smoothing, or unnecessary computational cost. To identify an effective configuration, we evaluate different combinations of Grapher Blocks in the first and second backbone stages, selected from {2, 4, 6, 8}.

![](images/1e8b92471b8c2e8d826f1bf5e2599724a9e13d6dcdaeabdb722e224b2eac5f53.jpg)  
Fig. 7: Ablation on the number of Grapher Blocks in the backbone. The Y-axis indicates the number of Grapher Blocks in the first stage, while the X-axis indicates the number of Grapher Blocks in the second stage.

As shown in Figure 7, the configuration with four Grapher Blocks in the first stage and six Grapher Blocks in the second stage achieves the best EER of 10.595%. This indicates that the model benefits from moderate graph depth: sufficient to capture long-range vein connectivity, but not so deep that feature representations become over-smoothed. This observation supports the final backbone design used in OpenVeinNet.

## 4.3 Comparison with Standard Loss Functions

TABLE 9: Ablation study on standard loss functions under the $A B D E  C$ protocol. Lower EER is better.

<table><tr><td>Loss Function</td><td>EER (%)</td></tr><tr><td>MSE</td><td>26.68</td></tr><tr><td>NLL</td><td>33.98</td></tr><tr><td>Proposed CAH Loss</td><td>7.44</td></tr></table>

Table 9 compares the proposed CAH loss with standard MSE and NLL losses. The proposed loss achieves a substantially lower EER than both alternatives, confirming that regression-style or likelihood-based objectives are not sufficient for learning highly discriminative finger-vein embeddings. In contrast, the proposed loss explicitly promotes class separability and compact identity-specific embeddings, which are essential for verification.

## 4.4 Sensitivity to Graph Neighbourhood Size k

Table 10 reports the effect of the neighbourhood size k used in the k-NN graph construction across the two backbone stages. Using the same value in both stages, whether $k { = } 9$ or $k { = } 1 8 ,$ gives weaker performance than the proposed asymmetric configuration of k=18 in Stage I and k=9 in Stage II. The symmetric k=18 configuration performs the worst among all variants, suggesting that using a large neighbourhood in the later stage introduces noise from less relevant nodes as spatial resolution decreases. The proposed decreasing schedule allows Stage I to aggregate broad vascular connectivity at higher resolution, while Stage II refines more localised relational evidence at lower resolution. This mirrors the progressively decreasing kernel configuration used in the DSConv stem and reflects a consistent design principle: broader context in early stages, finer discrimination in later stages.

## 4.5 Comparison with Strong Angular-Margin Losses

To further validate the proposed loss, Table 11 compares CAH against strong angular-margin losses, including ArcFace, MagFace, and AdaFace. These losses are widely used for discriminative biometric representation learning. The proposed CAH loss achieves the lowest mean EER of 7.44%, compared with 7.83% for ArcFace, 8.20% for Mag-Face, and 8.22% for AdaFace. The numerical differences are modest, reflecting the fact that all compared losses enforce angular separation to some degree. The distinction of the proposed loss lies in its explicit penalisation of intra-class scatter through the centroid alignment term, which complements the inter-class angular margin. In open-set verification, where decisions are made by thresholding cosine similarity against enrolled templates rather than through a fixed classifier, compact intra-class embeddings directly improve genuine score concentration and therefore TAR at strict operating thresholds. This is reflected in the TAR@FAR results in Tables 5 and 6, where the full model with CAH loss achieves competitive or higher TAR than the compared methods across most protocols.

TABLE 10: Sensitivity of the graph neighbourhood size k in the two backbone stages on the ABDE→C protocol. Lower EER is better.
<table><tr><td>Stage I k</td><td>Stage II k</td><td>EER (%)</td></tr><tr><td>9</td><td>9</td><td>10.37</td></tr><tr><td>9</td><td>18</td><td>9.42</td></tr><tr><td>18</td><td>18</td><td>10.48</td></tr><tr><td>18</td><td>9 (proposed)</td><td>7.44</td></tr></table>

<table><tr><td>Loss Function</td><td>Mean EER (%)</td></tr><tr><td>ArcFace [48] MagFace [49] AdaFace [50] Proposed CAH Loss</td><td>7.83 8.20 8.22 7.44</td></tr></table>

TABLE 11: Loss ablation against strong angular-margin losses on the $A B D E  C$ protocol. Lower EER is better.

## 4.6 Sensitivity to the Balancing Factor β

<table><tr><td> $\overline { { \beta } }$ </td><td>EER (%)</td></tr><tr><td>0.1 0.3 (default)</td><td>9.07 7.44</td></tr><tr><td>0.5</td><td>10.60</td></tr><tr><td>0.9</td><td>11.57</td></tr><tr><td></td><td></td></tr></table>

TABLE 12: Sensitivity of the proposed CAH loss to the balancing factor $\beta$ on the $A B D { \bar { E } } { \stackrel { . } { \to } } C$ protocol. Lower EER is better.

Table 12 reports the sensitivity of the proposed loss to the angular balancing factor $\beta .$ The best performance is achieved at $\beta \ : = \ : 0 . 3 ,$ , with an EER of 7.44%. When β is too small, such as 0.1, the angular component is underweighted and the model does not fully benefit from angular supervision. Conversely, when β is too large, such as 0.5 or 0.9, the angular term dominates the optimisation and degrades performance.

This confirms that the angular term is useful, but must be balanced carefully. The default value $\beta = 0 . 3$ provides the best trade-off between classification supervision and angular regularisation, supporting the choice used in the final model.

## 4.7 Component-Level Ablation of DSConv and Graph Modelling

<table><tr><td>Variant</td><td>Mean EER (%)</td></tr><tr><td>Standard stem + no graph</td><td>13.59</td></tr><tr><td>DSConv stem + no graph</td><td>10.96</td></tr><tr><td>Standard stem + graph backbone Full model (DSConv + graph backbone)</td><td>11.64 7.44</td></tr></table>

TABLE 13: Component ablation isolating the contributions of DSConv and graph modelling on the $A B D E \ \to \ C$ protocol. Lower EER is better.

Table 13 isolates the contribution of the DSConv stem and the graph-based backbone. The standard convolutional stem without graph modelling gives the weakest performance, with an EER of 13.59%. Replacing the standard stem with DSConv reduces the EER to 10.96%, showing that adaptive tubular feature extraction is beneficial for fingervein representation. Similarly, adding the graph backbone to a standard stem improves the EER to 11.64%, demonstrating that graph-based relational modelling also contributes positively.

The full model achieves an EER of 7.44%, compared with 13.59% for the standard baseline, 10.96% for DSConv alone, and 11.64% for the graph backbone alone. The combined improvement is larger than the sum of the individual gains from each component. This suggests that the two components interact synergistically: graph-based relational reasoning is most effective when node features are already geometrically aligned with vein structures, a condition that DSConv satisfies by design but standard convolution does not. The combination therefore produces gains that neither component achieves in isolation.

## 4.8 Overall Ablation Analysis

The ablation results provide strong evidence for the design of OpenVeinNet. The stem analysis shows that a progressively decreasing DSConv kernel configuration, (9, 7, 3), is most effective for capturing both broad and fine vein structures. The Grapher Block analysis confirms that a moderate graph depth, using four Grapher Blocks in the first stage and six in the second, provides the best balance between relational modelling and representation stability.

The loss ablations further show that the proposed CAH loss is a key contributor to the final performance. It substantially outperforms standard losses and also achieves the lowest EER among the evaluated angular-margin alternatives, including ArcFace, MagFace, and AdaFace. The $\beta$ sensitivity study confirms that the angular component is beneficial when properly balanced, with $\beta = 0 . 3$ giving the best result.

Finally, the component-level ablation directly addresses whether the architecture is merely a combination of existing modules. The results show that both DSConv and graph modelling independently improve performance, and their combination provides the strongest result. This demonstrates that OpenVeinNet benefits from a synergistic interaction between local tubular feature extraction, global graphbased relational modelling, and angularly constrained embedding learning.

Overall, these ablation studies confirm that each major component of the proposed framework contributes meaningfully to the final verification performance, thereby validating the architectural and loss-design choices of Open-VeinNet.

## 5 COMPUTATIONAL COST AND INFERENCE LA-TENCY

In addition to verification performance, practical deployment of biometric systems requires careful consideration of computational cost and inference latency. While the proposed OpenVeinNet is designed primarily for robust openset verification, it is equally important to quantify its efficiency characteristics and understand the associated tradeoffs relative to existing methods.

To this end, we benchmark all learned approaches under a unified experimental protocol. Specifically, we evaluate each method using its held-out FV-300 checkpoint, a batch size of 1, and repeated forward passes on both CPU and GPU platforms. We report the number of learnable parameters as a proxy for model size, FLOPs as an approximate measure of computational complexity, and mean singleimage latency and throughput. Handcrafted methods (MCP, RLT, WLD) are excluded due to the absence of a unified neural inference pipeline. For VeinAttNet, FLOPs are reported as $\mathrm { ~ n ~ } / \mathrm { a } ,$ , as its available implementation is in MATLAB and not directly compatible with the same profiling tools.

Table 14 summarises the computational characteristics of the evaluated methods. Several key observations can be drawn.

Efficiency-oriented models. Lightweight architectures such as FV-ViT exhibit the lowest computational footprint, with only 0.36M parameters and 0.03 GFLOPs, resulting in the fastest CPU latency (5.02 ms) and highest CPU throughput. Similarly, ArcVein achieves the fastest GPU latency (0.94 ms) and highest throughput, making it well suited for high-throughput scenarios. However, as demonstrated in Section 3.3, these efficiency gains come at the cost of reduced verification robustness, particularly under cross-dataset and open-set conditions.

Proposed method: accuracy-driven design. The proposed OpenVeinNet adopts a more expressive architecture, resulting in a moderate computational footprint of 4.76M parameters and 5.07 GFLOPs. This leads to higher latency (55.82 ms on CPU and 13.00 ms on GPU) compared to lightweight baselines. Nevertheless, the model remains computationally practical, especially in GPU-enabled or server-side deployment scenarios. Importantly, OpenVein-Net is still more compact than several CNN-based baselines (e.g., ArcVein and LGFIN) in terms of parameter count, while delivering substantially stronger verification performance.

Justification of computational cost. The increased computational cost is a direct consequence of the architectural components that enable improved verification performance. The DSConv layers introduce adaptive sampling mechanisms that align receptive fields with curvilinear vein structures, improving feature quality at the expense of additional computation. The Grapher Blocks further incorporate dynamic graph construction and message passing, allowing the model to capture long-range topological dependencies that cannot be effectively modelled by standard convolutions. While these operations are computationally more demanding, they contribute directly to the discriminative strength and generalisability of the learned embeddings.

In addition, the proposed Centroid Angular Hybrid Loss improves embedding separability in angular space, which is particularly beneficial for open-set verification. As shown in Section 3.3, this results in consistently lower EER and higher TAR, especially under strict low-FAR operating conditions that are critical for real-world biometric systems.

Accuracy-efficiency trade-off. Overall, the results highlight a clear and expected accuracy–efficiency trade-off. Methods such as FV-ViT and ArcVein achieve lower latency but exhibit weaker generalisation and reduced robustness under domain shift. In contrast, OpenVeinNet prioritises discriminative representation learning and open-set reliability, leading to improved verification performance at the cost of moderate computational overhead.

<table><tr><td>Method</td><td>Params (M)</td><td>FLOPs (G)</td><td>CPU Lat. (ms)</td><td>CPU Thr. (img/s)</td><td>GPU Lat. (ms)</td><td>GPU Thr. (img/s)</td></tr><tr><td>ArcVein [17]</td><td>7.83</td><td>2.45</td><td>7.09</td><td>141.10</td><td>0.94</td><td>1067.42</td></tr><tr><td>LGFIN [34]</td><td>6.79</td><td>7.51</td><td>45.48</td><td>21.99</td><td>3.84</td><td>260.42</td></tr><tr><td>FV-ViT [30]</td><td>0.36</td><td>0.03</td><td>5.02</td><td>199.37</td><td>1.74</td><td>574.79</td></tr><tr><td>VeinAttNet [31]</td><td>0.09</td><td>n/a</td><td>11.24</td><td>88.96</td><td>10.63</td><td>94.08</td></tr><tr><td>Proposed Method</td><td>4.76</td><td>5.07</td><td>55.82</td><td>17.92</td><td>13.00</td><td>76.94</td></tr></table>

TABLE 14: Runtime and computational-cost comparison on FV-300 using batch size 1. Lower latency is better; higher throughput is better.

From a deployment perspective, this trade-off is well justified. In security-critical applications, such as financial authentication and identity verification, robustness and low error rates are typically prioritised over minimal latency. At the same time, the proposed model remains sufficiently efficient for practical deployment, particularly when GPU acceleration is available.

In summary, OpenVeinNet achieves a favourable balance between computational cost and verification performance, offering strong cross-dataset and open-set robustness while maintaining practical inference efficiency for GPU-enabled deployment.

## 6 INTERPRETATION OF THE PROPOSED METHOD

Interpretability is important for understanding the decision behaviour of deep learning-based biometric systems, particularly in security-sensitive verification tasks. In this work, we employ Grad-CAM++ [51] to visualise the image regions that contribute most strongly to the proposed model’s embedding responses. This analysis helps verify whether Open-VeinNet focuses on meaningful vascular structures rather than irrelevant background or sensor-specific artefacts.

We analyse two verification cases: (i) genuine comparisons, where the enrolment and probe images belong to the same identity, and (ii) impostor comparisons, where the enrolment and probe images belong to different identities. For each case, we examine both high-similarity and lowsimilarity pairs. Grad-CAM++ activation maps are computed separately for the stem and backbone components to understand how local tubular feature extraction and graphbased feature aggregation contribute to the final representation.

Figures 8 and 9 show that the stem activations are strongly aligned with visible vein structures in the input images. This behaviour is expected because the DSConv stem is designed to follow curvilinear and tubular patterns. In genuine high-similarity cases, the activations are more concentrated around consistent vascular regions, indicating that the model relies on stable vein evidence for matching. In low-similarity or impostor cases, the responses become more diffuse or shift towards less consistent regions, reflecting weaker correspondence between the compared samples.

The backbone activations show a different behaviour. Rather than focusing only on local vein texture, the backbone highlights broader regions that appear to encode higher-level identity-specific information. This is consistent with the role of the Grapher Blocks, which aggregate relational information between neighbouring and non-local vein regions. Therefore, the interpretation results support the intended architectural design: the stem captures finegrained tubular vein morphology, while the graph-based backbone refines these features into a more discriminative identity representation.

Overall, the Grad-CAM++ analysis provides qualitative evidence that OpenVeinNet bases its verification decisions on meaningful vascular patterns and their higher-level structural relationships. This supports the reliability of the proposed architecture and helps explain its strong performance under cross-dataset and open-set verification settings.

## 7 CONCLUSION

Finger vein biometrics offer inherent resistance to spoofing and therefore represent a highly reliable modality for secure access control applications. In this work, we introduced OpenVeinNet, a novel framework for open-set finger vein verification that explicitly models both local vascular morphology and global structural relationships. The proposed architecture integrates a Dynamic Snake Convolution (DSConv)-based Stem for adaptive extraction of curvilinear vein patterns, with a graph-based Backbone that captures long-range dependencies through relational feature aggregation. In addition, the proposed Centroid Angular Hybrid (CAH) Loss enhances the discriminative capability of the embedding space by jointly enforcing intra-class compactness and inter-class angular separability, which is particularly beneficial in open-set scenarios. Extensive cross-dataset experiments conducted under both unknown-rejection and full-subject verification protocols demonstrate that Open-VeinNet consistently demonstrates strong generalisation compared with state-of-the-art methods across multiple benchmarks. The results show significant improvements in EER and TAR, especially under strict low-FAR operating conditions, confirming the robustness of the learned representations in challenging real-world scenarios. Furthermore, the computational analysis highlights a favourable accuracy–efficiency trade-off, where the moderate increase in computational cost is justified by substantial gains in verification performance and generalisation. The interpretability analysis using Grad-CAM++ further validates that the model focuses on meaningful vascular structures, reinforcing the reliability of its decision-making process. Overall, the proposed framework provides a principled and effective solution for open-set finger vein verification. Future work will explore lightweight variants of the architecture for edge deployment and investigate its applicability to presentation attack detection and multimodal biometric systems.

![](images/142f17e4a5748010074c1902d70b9105e4f43e3b2ea9f3dede2049749e23b48e.jpg)  
Fig. 8: Grad-CAM++ [51]-based interpretation of the proposed model for finger vein images from the same identity, representing a genuine comparison.

![](images/aa7a17a064f71e2ed766203c94c8e3d819bcfd9194baece17eb11b6838671a7c.jpg)  
Fig. 9: Grad-CAM++ [51]-based interpretation of the proposed model for finger vein images from different identities, representing an impostor comparison.

## REFERENCES

[1] H. Qin and M. El Yacoubi, “Finger-vein quality assessment by representation learning from binary images,” vol. 9489, 11 2015.

[2] F. Radzi, S.-I. Khalid, F. Gong, N. Mustafa, Y. C. Wong, and M. Mat ibrahim, “User identification system based on finger-vein patterns using convolutional neural network,” vol. 11, pp. 3316– 3319, 03 2016.

[3] F. Radzi, M. Khalil-Hani, and R. Bakhteri, “Finger-vein biometric identification using convolutional neural network,” TURKISH JOURNAL OF ELECTRICAL ENGINEERING & COMPUTER SCI-ENCES, vol. 24, pp. 1863–1878, 01 2016.

[4] H. Qin and M. A. El-Yacoubi, “Deep representation-based feature extraction and recovering for finger-vein verification,” IEEE Transactions on Information Forensics and Security, vol. 12, no. 8, pp. 1816– 1829, 2017.

[5] H. G. Hong, M. B. Lee, and K. R. Park, “Convolutional neural network-based finger-vein recognition using nir image

sensors,” Sensors, vol. 17, no. 6, 2017. [Online]. Available: https://www.mdpi.com/1424-8220/17/6/1297

[6] R. Das, E. Piciucco, E. Maiorana, and P. Campisi, “Convolutional neural network for finger-vein-based biometric identification,” IEEE Transactions on Information Forensics and Security, vol. 14, no. 2, pp. 360–373, 2019.

[7] C. Xie and A. Kumar, “Finger vein identification using convolutional neural network and supervised discrete hashing,” Pattern Recognition Letters, vol. 119, pp. 148–156, 2019, deep Learning for Pattern Recognition. [Online]. Available: https:// www.sciencedirect.com/science/article/pii/S0167865517304397

[8] J. M. Song, W. Kim, and K. R. Park, “Finger-vein recognition based on deep densenet using composite image,” IEEE Access, vol. 7, pp. 66 845–66 863, 2019.

[9] S. Tang, S. Zhou, W. Kang, Q. Wu, and F. Deng, “Finger vein verification using a siamese cnn,” IET Biometrics, vol. 8, no. 5, pp. 306–315, 2019. [Online]. Available: https://ietresearch. onlinelibrary.wiley.com/doi/abs/10.1049/iet-bmt.2018.5245

[10] B. Hou and R. Yan, “Convolutional autoencoder model for fingervein verification,” IEEE Transactions on Instrumentation and Measurement, vol. 69, no. 5, pp. 2067–2074, 2020.

[11] W. Yang, C. Hui, Z. Chen, J.-H. Xue, and Q. Liao, “Fv-gan: Finger vein representation using generative adversarial networks,” IEEE Transactions on Information Forensics and Security, vol. 14, no. 9, pp. 2512–2524, 2019.

[12] J. Zeng, F. Wang, J. Deng, C. Qin, Y. Zhai, J. Gan, and V. Piuri, “Finger vein verification algorithm based on fully convolutional neural network and conditional random field,” IEEE Access, vol. 8, pp. 65 402–65 419, 2020.

[13] R. S. Kuzu, E. Piciucco, E. Maiorana, and P. Campisi, “On-thefly finger-vein-based biometric recognition using deep neural networks,” IEEE Transactions on Information Forensics and Security, vol. 15, pp. 2641–2654, 2020.

[14] Z. Hao, P. Fang, and H. Yang, “Finger vein recognition based on multi-task learning,” in Proceedings of the 2020 5th International Conference on Mathematics and Artificial Intelligence, ser. ICMAI ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 133–140. [Online]. Available: https://doi.org/10.1145/3395260.3395277

[15] H. Ren, L. Sun, J. Guo, C. Han, and F. Wu, “Finger vein recognition system with template protection based on convolutional neural network,” Knowledge-Based Systems, vol. 227, p. 107159, 2021. [Online]. Available: https://www.sciencedirect. com/science/article/pii/S0950705121004226

[16] R. Salih Kuzu, E. Maiorana, and P. Campisi, “Loss functions for cnn-based biometric vein recognition,” in 2020 28th European Signal Processing Conference (EUSIPCO), 2021, pp. 750–754.

[17] B. Hou and R. Yan, “Arcvein-arccosine center loss for finger vein verification,” IEEE Transactions on Instrumentation and Measurement, vol. 70, pp. 1–11, 2021.

[18] Z. Chen, W. Yu, H. Bai, and Y. Li, “An arcloss-based and opensettest-oriented finger vein recognition system,” in Biometric Recognition, J. Feng, J. Zhang, M. Liu, and Y. Fang, Eds. Cham: Springer International Publishing, 2021, pp. 287–294.

[19] W. Yang, J. Huang, Z. Chen, J. Zhao, and W. Kang, “Multi-view finger vein recognition using attention-based mvcnn,” in Biometric Recognition, W. Deng, J. Feng, D. Huang, M. Kan, Z. Sun, F. Zheng, W. Wang, and Z. He, Eds. Cham: Springer Nature Switzerland, 2022, pp. 82–91.

[20] H. Qin, R. Hu, M. A. El-Yacoubi, Y. Li, and X. Gao, “Local attention transformer-based full-view finger-vein identification,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 6, pp. 2767–2782, 2023.

[21] T. Chai, J. Li, S. Prasad, Q. Lu, and Z. Zhang, “Shape-driven lightweight cnn for finger-vein biometrics,” Journal of Information Security and Applications, vol. 67, p. 103211, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/ S2214212622000886

[22] I. Boucherit, M. O. Zmirli, H. Hentabli, and B. A. Rosdi, “Finger vein identification using deeply-fused convolutional neural network,” Journal of King Saud University - Computer and Information Sciences, vol. 34, no. 3, pp. 646–656, 2022. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S1319157820303372

[23] W. Liu, H. Lu, Y. Wang, Y. Li, Z. Qu, and Y. Li, “Mmran: A novel model for finger vein recognition based on a residual attention mechanism,” Applied Intelligence, vol. 53, no. 3, pp. 3273–3290, Feb 2023. [Online]. Available: https://doi.org/10.1007/s10489-022-03645-7

[24] Z. Zhang and M. Wang, “Finger vein recognition based on lightweight convolutional attention model,” IET Image Processing, vol. 17, no. 6, pp. 1864–1873, 2023. [Online]. Available: https:// ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/ipr2.12761

[25] C. Fang, H. Ma, and J. Li, “A finger vein authentication method based on the lightweight siamese network with the self-attention mechanism,” Infrared Physics & Technology, vol. 128, p. 104483, 2023. [Online]. Available: https://www.sciencedirect. com/science/article/pii/S1350449522004649

[26] C.-H. Hsia, L.-Y. Ke, and S.-T. Chen, “Improved lightweight convolutional neural network for finger vein recognition system,” Bioengineering, vol. 10, no. 8, 2023. [Online]. Available: https://www.mdpi.com/2306-5354/10/8/919

[27] Y. Huang, H. Ma, and M. Wang, “Axially enhanced local attention network for finger vein recognition,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–10, 2023.

[28] J. Huang, A. Zheng, M. S. Shakeel, W. Yang, and W. Kang, “Fvfsnet: Frequency-spatial coupling network for finger vein authentication,” IEEE Transactions on Information Forensics and Security, vol. 18, pp. 1322–1334, 2023.

[29] B. Ma, K. Wang, and Y. Hu, “Finger vein recognition based on bilinear fusion of multiscale features,” Scientific Reports, vol. 13, no. 1, p. 249, Jan. 2023.

[30] X. Li and B.-B. Zhang, “Fv-vit: Vision transformer for finger vein recognition,” IEEE Access, vol. 11, pp. 75 451–75 461, 2023.

[31] R. Ramachandra and S. Venkatesh, “Fingervein verification using convolutional multi-head attention network,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), January 2024, pp. 6175–6184.

[32] H. Qin, C. Fan, S. Deng, Y. Li, M. A. El-Yacoubi, and G. Zhou, “Agnas: An attention gru-based neural architecture search for fingervein recognition,” IEEE Transactions on Information Forensics and Security, vol. 19, pp. 1699–1713, 2024.

[33] P. Zhao, Y. Song, S. Wang, J.-H. Xue, S. Zhao, Q. Liao, and W. Yang, “Vpcformer: A transformer-based multi-view finger vein recognition model and a new benchmark,” Pattern Recognition, vol. 148, p. 110170, 2024. [Online]. Available: https://www. sciencedirect.com/science/article/pii/S0031320323008671

[34] E. Li, L. Yang, K. Su, and H. Liu, “Local and global feature interaction network for partial finger vein recognition,” IEEE Signal Processing Letters, vol. 32, pp. 906–910, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:276369489

[35] N. Miura, A. Nagasaka, and T. Miyatake, “Extraction of fingervein patterns using maximum curvature points in image profiles,” IEICE Trans. Inf. Syst., vol. 90-D, pp. 1185–1194, 2007. [Online]. Available: https://api.semanticscholar.org/CorpusID:16384087

[36] N. Miura, A. Nagasaka, and T. Miyatake, “Feature extraction of finger-vein patterns based on repeated line tracking and its application to personal identification,” Machine Vision and Applications, vol. 15, pp. 194–203, 10 2004.

[37] B. Huang, Y. Dai, R. Li, D. Tang, and W. Li, “Finger-vein authentication based on wide line detector and pattern normalization,” in 2010 20th International Conference on Pattern Recognition, 2010, pp. 1269–1272.

[38] K. Wang, A. S. Khisa, X. Wu, and Q. Zhao, “Finger vein recognition using lbp variance with global matching,” in International Conference on Wavelet Analysis and Pattern Recognition, July 2012, pp. 196–201.

[39] B. A. Rosdi, C. W. Shing, and S. A. Suandi, “Finger vein recognition using local line binary pattern,” Sensors, vol. 11, no. 12, pp. 11 357–11 371, 2011.

[40] Y. Lu, S. Yoon, S. J. Xie, J. Yang, Z. Wang, and D. S. Park, “Finger vein recognition using histogram of competitive gabor responses,” in 22nd International Conference on Pattern Recognition, Aug 2014, pp. 1758–1763.

[41] R. Raghavendra, J. Surbiryala, and C. Busch, “Hand dorsal vein recognition: Sensor, algorithms and evaluation,” in 2015 IEEE International Conference on Imaging Systems and Techniques (IST), 2015, pp. 1–6.

[42] G. Li, M. Muller, A. Thabet, and B. Ghanem, “Deepgcns: Can¨ gcns go as deep as cnns?” in The IEEE International Conference on Computer Vision (ICCV), 2019.

[43] Y. Qi, Y. He, X. Qi, Y. Zhang, and G. Yang, “Dynamic snake convolution based on topological geometric constraints for tubular structure segmentation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 6070–6079.

[44] Y. Lu, S. Xie, S. Yoon, Z. Wang, and D. Park, “An available database for the research of finger vein recognition,” vol. 1, 12 2013, pp. 410–415.

[45] M. S. Mohd Asaari, S. A. Suandi, and B. A. Rosdi, “Fusion of band limited phase only correlation and width centroid contour distance for finger based biometrics,” Expert Systems with Applications, vol. 41, no. 7, pp. 3367– 3382, 2014. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S0957417413009536

[46] A. Kumar and Y. Zhou, “Human identification using finger images,” IEEE Transactions on Image Processing, vol. 21, no. 4, pp. 2228–2244, 2012.

[47] P. Tome, R. Raghavendra, C. Busch, S. Tirunagari, N. Poh, B. H. Shekar, D. Gragnaniello, C. Sansone, L. Verdoliva, and S. Marcel, “The 1st competition on counter measures to finger vein spoofing

attacks,” in 2015 International Conference on Biometrics (ICB), 2015, pp. 513–518.

[48] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “Arcface: Additive angular margin loss for deep face recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 4690–4699.

[49] Q. Meng, S. Zhao, Z. Huang, and F. Zhou, “Magface: A universal representation for face recognition and quality assessment,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 14 225–14 234.

[50] M. Kim, A. K. Jain, and X. Liu, “Adaface: Quality adaptive margin for face recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 18 750–18 759.

[51] A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, “Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks,” in 2018 IEEE Winter Conference on Applications of Computer Vision (WACV), 2018, pp. 839–847.

![](images/18d164ba7fcc5d9e4ab1dea8299f82dbb54b412ad67d4852545c2edb77db7af6.jpg)

Raghavendra Ramachandra received his Ph.D. in computer science and technology from the University of Mysore, India, and Tel´ ecom Sud-´ Paris, France, in 2010. He is currently a Full Professor at the Department of Information Security and Communication Technology, Norwegian University of Science and Technology (NTNU), Gjøvik, Norway, and Scientific Director of the SAFE Centre. His research interests include biometrics, presentation attack detection, morphing attack detection, multimodal fusion, deep learn-

ing, and image/video analysis. He has contributed to several EU, IARPA, and national research projects and holds multiple patents in biometric security. He is actively involved in ISO/IEC JTC1 SC37 biometric standardization and serves as an editor of ISO/IEC 24722 on multimodal biometrics. He is a Senior Member of the IEEE, has received several best paper awards, and currently serves as Vice President for Finances of the IEEE Biometrics Council.

![](images/2445d3ab83dd7c6897a8b8c5c52f0d349f5531d79a22e822b5d753177e4ee679.jpg)

Sushrut Patwardhan is a PhD student at the Norwegian University of Science and Technology (NTNU) and a data scientist at Mobai AS. His research focuses on deep learning applications in face biometrics, where he explores innovative solutions for identity verification and security. He is keen to investigate diverse deep learning architectures, constantly seeking to improve the accuracy, efficiency, and robustness of biometric systems.

## Supplementary Results OpenVeinNet: Robust Open-Set Finger Vein Verification with Dynamic Snake Convolution and Graph Learning

This supplementary material provides additional experimental evidence and implementation details for OpenVein-Net, a finger-vein verification framework based on dynamic snake convolution and graph-based representation learning. We first report paired statistical significance tests on full-subject verification results using seed-matched leaveone-dataset-out runs. Exact two-sided sign-flip permutation tests with Holm correction show that the equal error rate reductions obtained by OpenVeinNet are statistically significant against ArcVein, LGFIN, FV-ViT, and VeinAttNet. We then present intra-database open-set verification experiments on FV-300, FV-USM, and MMCBNU using subjectdisjoint identity partitions. OpenVeinNet achieves the lowest equal error rate on all three datasets, with values of 0.39%, 7.06%, and 4.07%, respectively, while maintaining strong true accept rates across practical false accept rate operating points. To support reproducibility, we document the training configurations, optimisation settings, input resolutions, losses, and augmentation strategies used for all learned baselines, together with the identity-level train– validation and held-out splits. Finally, t-SNE visualisations and detection error trade-off curves are used to examine the learned embedding space and verification behaviour. The results show compact and separable embeddings for higher quality datasets, increased overlap for low-sample and high-variability datasets, and favourable false match and false non-match trade-offs across evaluation protocols. Overall, the supplementary results support the statistical reliability, reproducibility, and open-set verification behaviour of OpenVeinNet.

## 8 STATISTICAL SIGNIFICANCE ANALYSIS

To complement the mean and 95% confidence interval reporting in the main paper, we perform formal pairwise statistical significance testing between the proposed method and competing approaches.

We conduct paired significance testing on the full-subject verification EER results using two-sided exact sign-flip permutation tests over matched leave-one-dataset-out runs. For each comparison, pairing is performed across seedmatched runs obtained from the five evaluation protocols. Since lower EER indicates better performance, a positive mean ∆EER corresponds to an average reduction in EER in favour of the proposed method. To account for multiple comparisons, Holm correction is applied to the resulting pvalues.

The results show (see Table 15) that all pairwise comparisons are statistically significant after Holm correction. This confirms that the observed improvements of OpenVeinNet over competing methods are not only reflected in mean performance, but are also consistent across matched evaluation runs.

## 9 INTRA-DATABASE OPEN-SET EVALUATION

## 9.1 Intra-Dataset Open-Set Evaluation

To complement the cross-dataset evaluation, we further conduct intra-database open-set experiments on FV-300, FV-USM, and MMCBNU. These datasets provide a sufficient number of samples per identity to support training and evaluation on identity-disjoint subsets within the same database. For each dataset and statistical seed, the first 80% of the sorted identities are used for model development, while the remaining 20% of identities are held out entirely and used only for open-set verification. During evaluation, embeddings are extracted for the held-out identities, and genuine and impostor scores are computed from withinidentity and cross-identity image pairs among these unseen subjects. PolyU and VERA are not included in this experiment because they contain only four and two samples per identity, respectively, which is insufficient for a reliable intra-database learning and held-out evaluation split under the proposed protocol.

Table 16 reports the intra-database open-set verification results. On FV-300, all strong methods perform well due to the high image quality and large number of samples per identity. Nevertheless, the proposed method achieves the lowest EER of 0.39% and the highest TAR at FAR= 0.1%, while matching the best TAR at FAR= 1% and FAR= 10%. This confirms that OpenVeinNet remains highly competitive even when the intra-database setting is favourable to several baseline methods.

On FV-USM, the advantage of the proposed method becomes more evident. OpenVeinNet achieves an AUC of 98.26% and an EER of 7.06%, substantially improving over the second-best LGFIN result of 11.25% EER. The proposed method also provides the highest TAR across all operating points, including 53.38% at FAR= 0.1% and 73.66% at FAR= 1%. These results indicate that the proposed representation is more robust when intra-class variability and acquisition differences are stronger.

On MMCBNU, the proposed method again achieves the best AUC and EER, with 99.31% AUC and 4.07% EER. LGFIN obtains a slightly higher TAR at the strictest FAR= 0.1%, but OpenVeinNet achieves the best TAR at FAR= 1% and FAR= 10%, showing a more favourable operating behaviour across practical thresholds. This suggests that the proposed combination of DSConv, graphbased modelling, and CAH loss produces embeddings that remain discriminative across both strict and relaxed operating points.

Overall, the intra-database results support the conclusions from the cross-dataset experiments. The proposed method performs strongly not only when evaluated across different datasets, but also when trained and tested within the same database under an open-set protocol. This additional experiment directly confirms that OpenVeinNet is effective for conventional intra-database open-set verification, while the cross-dataset experiments demonstrate its generalisation ability under stronger domain shift.

<table><tr><td>Comparison</td><td>n</td><td>Mean ∆EER (pp)</td><td>Holm-adjusted p</td><td>Wilcoxon p</td><td>Significant</td></tr><tr><td>Proposed vs ArcVein</td><td>20</td><td>15.17</td><td> $7 . 6 3 \times 1 0 ^ { - 6 }$ </td><td> $\overline { { 1 . 9 1 \times 1 0 ^ { - 6 } } }$ </td><td>Yes</td></tr><tr><td>Proposed vs LGFIN</td><td>20</td><td>13.05</td><td> $7 . 6 3 \times 1 0 ^ { - 6 }$ </td><td> $1 . 9 1 \times 1 0 ^ { - 6 }$ </td><td>Yes</td></tr><tr><td>Proposed vs FV-ViT</td><td>20</td><td>13.66</td><td> $7 . 6 3 \times 1 0 ^ { - 6 }$ </td><td> $1 . 9 1 \times 1 0 ^ { - 6 }$ </td><td>Yes</td></tr><tr><td>Proposed vs VeinAttNet</td><td>20</td><td>4.44</td><td> $9 . 5 4 \times 1 0 ^ { - 5 }$ </td><td> $1 . 6 8 \times 1 0 ^ { - 4 }$ </td><td>Yes</td></tr></table>

TABLE 15: Paired statistical significance testing on full-subject EER. A positive mean ∆EER indicates that the proposed method achieves lower EER than the comparator. Holm-adjusted p values are computed from the exact paired sign-flip permutation test, while the Wilcoxon column reports the paired two-sided signed-rank test.

TABLE 16: Intra-database open-set verification results with 95% confidence intervals over five statistical seeds. The experiments are reported on FV-300, FV-USM, and MMCBNU, where sufficient samples per identity are available for reliable intra-database training and evaluation.
<table><tr><td>Dataset</td><td>Algorithm</td><td>AUC (%)</td><td>EER (%)</td><td>TAR@FAR= 0.1% (%)</td><td>TAR@FAR= 1% (%)</td><td>TAR@FAR= 10% (%)</td></tr><tr><td rowspan="7">FV-300</td><td>MCP</td><td>99.96</td><td>0.42</td><td>95.32</td><td>100.00</td><td>100.00</td></tr><tr><td>RLT</td><td>99.92</td><td>0.74</td><td>78.14</td><td>99.70</td><td>100.00</td></tr><tr><td>ArcVein</td><td>96.36 (94.01–98.72)</td><td>11.03 (5.95–16.11)</td><td>54.26 (45.30–63.22)</td><td>67.25 (57.34–77.15)</td><td>89.13 (81.94–96.31)</td></tr><tr><td>LGFIN</td><td>99.99 (99.98–100.00)</td><td>0.43 (0.09–0.78)</td><td>98.42 (96.98–99.86)</td><td>99.77 (99.44–100.00)</td><td>100.00 (100.00–100.00)</td></tr><tr><td>FV-ViT</td><td>96.55 (95.04–98.06)</td><td>9.60 (7.59–11.61)</td><td>29.05 (7.24–50.86)</td><td>48.83 (28.26–69.40)</td><td>90.48 (85.25–95.70)</td></tr><tr><td>VeinAttNet</td><td>99.73 (99.63–99.83)</td><td>2.43 (1.81–3.05)</td><td>63.32 (47.63–79.01)</td><td>93.71 (90.81–96.60)</td><td>99.90 (99.86–99.94)</td></tr><tr><td>Proposed Method</td><td>99.99 (99.98–100.00)</td><td>0.39 (0.06–0.72)</td><td>98.66 (96.85–100.00)</td><td>99.77 (99.41–100.00)</td><td>100.00 (100.00–100.00)</td></tr><tr><td rowspan="7">FV-USM</td><td>MCP</td><td>86.64</td><td>18.98</td><td>0.01</td><td>0.62</td><td>30.20</td></tr><tr><td>RLT</td><td>79.69</td><td>24.76</td><td>0.00</td><td>0.01</td><td>5.35</td></tr><tr><td>ArcVein</td><td>65.05 (61.48–68.62)</td><td>38.01 (35.77–40.24)</td><td>0.69 (0.00–1.64)</td><td>2.69 (1.08–4.30)</td><td>18.54 (15.01–22.07)</td></tr><tr><td>LGFIN</td><td>95.60 (94.71–96.48)</td><td>11.25 (9.89–12.61)</td><td>28.91 (7.57–50.25)</td><td>45.42 (27.85–62.99)</td><td>87.17 (83.96–90.39)</td></tr><tr><td>FV-ViT</td><td>82.90 (79.16–86.65)</td><td>26.25 (23.52–28.98)</td><td>5.72 (0.00–14.46)</td><td>15.93 (3.59–28.26)</td><td>45.14 (33.70–56.58)</td></tr><tr><td>VeinAttNet Proposed Method</td><td>83.85 (77.20–90.51)</td><td>25.60 (20.08–31.11)</td><td>10.73 (3.40–18.05)</td><td>19.45 (6.84–32.06)</td><td>50.97 (34.37–67.56)</td></tr><tr><td></td><td>98.26 (97.48–99.04)</td><td>7.06 (5.40–8.72)</td><td>53.38 (41.30–65.46)</td><td>73.66 (67.37–79.94)</td><td>95.33 (92.58–98.08)</td></tr><tr><td rowspan="7">MMCBNU</td><td>MCP</td><td>93.04</td><td>12.24</td><td>0.04</td><td>3.84</td><td>77.91</td></tr><tr><td>RLT</td><td>90.44</td><td>13.76</td><td>0.00</td><td>0.08</td><td>59.24</td></tr><tr><td>ArcVein</td><td>78.51 (72.71–84.30)</td><td>29.70 (24.39–35.00)</td><td>4.79 (0.83–8.75)</td><td>9.20 (3.09–15.30)</td><td>30.04 (20.11–39.96)</td></tr><tr><td>LGFIN</td><td>99.19 (98.96–99.42)</td><td>4.33 (3.48–5.17)</td><td>52.17 (41.93–62.41)</td><td>84.43 (80.43–88.44)</td><td>98.50 (97.78–99.22)</td></tr><tr><td>FV-ViT VeinAttNet</td><td>92.20 (91.14–93.27)</td><td>16.25 (14.99–17.51)</td><td>21.14 (14.92–27.36)</td><td>40.06 (35.14–44.97)</td><td>74.98 (71.56–78.40)</td></tr><tr><td>Proposed Method</td><td>92.52 (87.99–97.04)</td><td>15.00 (9.46–20.54)</td><td>10.71 (0.21–21.20)</td><td>39.39 (21.50–57.28)</td><td>79.25 (65.82–92.68)</td></tr><tr><td></td><td>99.31 (99.13–99.50)</td><td>4.07 (3.43–4.71)</td><td>49.24 (30.59–67.90)</td><td>85.92 (82.15–89.69)</td><td>98.91 (98.45–99.37)</td></tr></table>

## 10 BASELINE TRAINING DETAILS AND EXPERI-MENTAL FAIRNESS

## 10.1 Overview

To ensure transparency and reproducibility, we provide detailed training configurations for all learned baselines used in the experiments. Unlike many prior works that report results directly from the literature, all deep learning baselines in this work were retrained by us under a unified evaluation protocol.

Specifically, ArcVein, LGFIN, FV-ViT, and the proposed method were implemented and trained in a common Py-Torch framework, while VeinAttNet was independently retrained in MATLAB using the published architecture within our codebase. Handcrafted methods (MCP, RLT, and WLD) do not involve learning and were executed as fixed pipelines.

## 11 SHARED EXPERIMENTAL PROTOCOL

All learned methods were trained and evaluated under the same leave-one-dataset-out protocol. For each experiment:

• Four datasets were used for training, and one dataset was held out for testing.

• The same statistical seeds were used across all methods to ensure matched runs.

• A common evaluation pipeline was used for computing AUC, EER, and TAR@FAR.

For PyTorch-based models, input images were converted from grayscale to three channels and processed using a shared data pipeline with the following augmentations:

• Random horizontal and vertical flips

Random autocontrast $( p = 0 . 0 5 )$

• Random rotations within $\pm 4 5 ^ { \circ }$

• Resizing to method-specific input resolution

VeinAttNet followed its original MATLAB-based augmentation strategy, including random reflections, translations (up to ±30 pixels), rotations, and scaling.

Importantly, no dataset-specific tuning was performed on the held-out test set. Each method used a fixed training configuration across all splits, except for the unavoidable change in the number of source classes.

## 11.1 Baseline Training Configurations

Table 17 summarises the training configurations for all learned methods.

## 11.2 Fairness of Comparison

The comparison is fair in the sense that all learned methods were:

• Retrained under the same cross-dataset protocol,

• Evaluated using identical metrics and code,

• Compared across matched statistical runs.

<table><tr><td>Method</td><td>Training setup</td><td>Optimizer</td><td>LR</td><td>Epochs</td><td>Batch</td><td>Input</td><td>Loss</td></tr><tr><td>MCP / RLT / WLD</td><td>Handcrafted pipelines</td><td>n/a</td><td> $\overline { { { \bf n } / { \bf a } } }$ </td><td>n/a</td><td>n/a</td><td>Native</td><td>No learning</td></tr><tr><td>ArcVein</td><td>Retrained (PyTorch)</td><td>SGD (Nesterov 0.9)</td><td> $1 \times 1 0 ^ { - 2 } ( \div 1 0 @ 3 0 )$ </td><td>100</td><td>64</td><td> $2 2 4 \times 2 2 4$ </td><td> $\mathrm { A r c - c o s i n e } + \mathrm { s o f t m a x } \left( \lambda = 0 . 0 1 \right)$ </td></tr><tr><td>LGFIN</td><td>Retrained (PyTorch)</td><td>AdamW</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>50</td><td>8</td><td> $1 4 7 \times 1 4 7$ </td><td>Cross-entropy</td></tr><tr><td>FV-ViT</td><td>Retrained (PyTorch)</td><td>AdamW</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>50</td><td>64</td><td> $2 2 4 \times 2 2 4$ </td><td>Cross-entropy</td></tr><tr><td>VeinAttNet</td><td>Retrained (MATLAB)</td><td>Adam</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>50</td><td>128</td><td> $2 2 4 \times 2 2 4$ </td><td>Softmax classification</td></tr><tr><td>Proposed</td><td>Retrained (PyTorch)</td><td>AdamW</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>150</td><td>128</td><td> $2 2 4 \times 2 2 4$ </td><td> $\mathrm { C A H } \left( \beta = 0 . 3 \right)$ </td></tr></table>

TABLE 17: Training configurations for the proposed method and all learned baselines.

At the same time, the comparison does not enforce identical hyperparameters across all methods. Each baseline retains its architecture-specific design choices, including input resolution, optimizer type, learning rate schedule, and loss formulation. For example, ArcVein is trained using its original SGD/Nesterov schedule rather than being forced into an Adam-based setup.

Enforcing identical hyperparameters across fundamentally different architectures would introduce a different form of bias. Therefore, we adopt a method-consistent but protocol-aligned evaluation strategy, which preserves the intended behaviour of each baseline while ensuring comparability across methods.

## 12 SUBJECT-DISJOINT TRAIN–VALIDATION SPLIT

To ensure clarity and reproducibility of the intra-database protocol, we report the subject-disjoint train–test statistics used for FV-300, FV-USM, and MMCBNU in Table 18. The table reports the identity-level development/test partition, together with the number of train and test images used within the development identities.

For each dataset and statistical seed, identities are first sorted and partitioned at the subject/identity level. The first 80% of identities are used for model development, and the remaining 20% of identities are held out entirely for intradatabase open-set testing. Thus, no subject or finger identity used for final open-set testing appears during model development.

Within the development identities, the dataset-provided train/test image folders are used for optimisation and validation/model selection, respectively. This image-level train/test split is not the source of open-set subject disjointness; subject disjointness is enforced by excluding the held-out 20% identities from all model-development steps. During the final intra-database evaluation, embeddings are extracted only for the held-out identities and genuine/impostor scores are computed from within-identity and cross-identity image pairs among those unseen subjects.

For FV-300, the reported counts are approximate because 124 low-quality images were removed before partitioning. Despite these variations, the same subject-disjoint development/test splitting strategy is applied consistently across all intra-database experiments and all learned methods.

## 13 T-SNE VISUALISATION AND EMBEDDING ANALYSIS

To further analyse the structure of the learned embedding space, we visualise the features extracted by OpenVein-Net using t-SNE across five datasets. For each dataset, we present three complementary views: (i) best 50 classes, (ii) worst 50 classes, and (iii) all classes. The best and worst classes are selected based on class-wise verification performance, allowing us to inspect embedding separability under favourable and challenging conditions.

<table><tr><td>Dataset</td><td>Total IDs</td><td>Train / Val. IDs</td><td>Train / Val. imgs</td><td>Held-out IDs</td></tr><tr><td>FV-300</td><td>300</td><td>240</td><td> $\overline { { 1 6 , 5 3 6 \mathrm { ~ / ~ } 4 , 9 5 4 } }$ </td><td>60</td></tr><tr><td>MMCBNU</td><td>600</td><td>480</td><td> $2 , \dot { 4 } 0 0 ~ \dot { / } ~ 2 , \dot { 4 } 0 0$ </td><td>120</td></tr><tr><td>FV-USM</td><td>492</td><td>393</td><td> $2 { , } 3 5 8 \ / \ 2 { , } 3 5 8$ </td><td>99</td></tr></table>

TABLE 18: Identity-level subject-disjoint train–test statistics for the intra-database open-set experiments. The first 80% of sorted identities are used for model development, and the remaining identities are held out entirely for open-set testing. The development image counts report the datasetprovided train/test folders used for optimisation and validation/model selection within the development identities.

FV-300 (see Fig. 10). The t-SNE visualisation for FV-300 shows well-defined and compact clusters for the best 50 classes, indicating strong intra-class compactness and clear inter-class separation. Even in the worst 50 classes, most identities remain reasonably separable, although slight overlaps appear in boundary regions. The all-class distribution exhibits a structured global arrangement, suggesting that the learned embeddings preserve identity-level discriminability across the dataset. This behaviour is consistent with the strong performance observed in the BCDE → A protocol, where FV-300 provides high-quality images and a large number of samples per identity.

FV-USM (see Fig. 14). For FV-USM, the best-class embeddings remain relatively compact but are more scattered compared to FV-300. The worst 50 classes show noticeable overlap and reduced cluster density, indicating increased intra-class variation and inter-class ambiguity. The all-class visualisation reveals a less structured global distribution, reflecting the moderate image quality and cross-session variability present in this dataset. Despite this, the presence of local grouping structures suggests that OpenVeinNet retains meaningful identity information under domain shift.

MMCBNU (see Fig. 18). The MMCBNU visualisation shows moderate cluster separability in the best 50 classes, while the worst classes exhibit increased dispersion and partial mixing. The all-class view reveals a relatively uniform distribution of embeddings, indicating that class boundaries are less sharply defined compared to FV-300. This behaviour can be attributed to variations in acquisition conditions and subject diversity. Nevertheless, identifiable cluster structures are still present, supporting the model’s ability to generalise across heterogeneous data.

PolyU (see Fig. 22). In the PolyU dataset, the best 50 classes show distinguishable clusters, though with reduced compactness compared to FV-300. The worst classes display significant overlap and irregular cluster shapes, indicating difficulty in separating certain identities. The all-class distribution appears more diffused, reflecting the unconstrained acquisition setup and long-term variability in the dataset. These observations are consistent with the moderate performance trends reported in the main experiments.

VERA (see Fig. 26). The VERA dataset presents the most challenging embedding structure. Even in the best 50 classes, clusters are sparse and less compact, while the worst classes show substantial overlap and poor separability. The all-class visualisation exhibits a highly scattered distribution with minimal cluster structure. This is expected due to the extremely limited number of samples per identity (only two images) and higher acquisition variability. The t-SNE results therefore provide strong qualitative support for the performance degradation observed in the ABCD → E protocol.

Overall analysis: Across all datasets, the t-SNE visualisations reveal a consistent pattern: datasets with higher image quality and more samples per identity (e.g., FV-300) produce well-separated and compact clusters, while lowsample and high-variability datasets (e.g., VERA) result in dispersed and overlapping embeddings. Importantly, even in challenging scenarios, the proposed OpenVeinNet maintains partial cluster structures, indicating that the learned representation remains discriminative.

These findings complement the quantitative results and dataset-quality analysis presented in the main paper. Together, they confirm that the performance variations across datasets are primarily driven by data characteristics rather than instability in the proposed model.

## 14 DET CURVE ANALYSIS

We further analyse the verification behaviour using DET curves under both half and full evaluation protocols (Fig. 18 and Fig. 19). Across all datasets, the proposed method consistently achieves curves closer to the origin, indicating a favourable trade-off between false match rate (FMR) and false non-match rate (FNMR).

On FV-300, all methods perform strongly due to high data quality; however, the proposed method maintains lower FNMR, particularly in the low-FMR regime, which is critical for secure applications. On FV-USM and MMCBNU, the performance gap becomes more pronounced, with OpenVeinNet showing consistently lower FNMR across most operating regions, reflecting improved robustness to intra-class variation and acquisition differences.

For PolyU, the DET curves exhibit greater dispersion across methods, yet the proposed approach retains a clear advantage, especially at moderate FMR values. VERA remains the most challenging dataset, where all methods degrade; nevertheless, OpenVeinNet maintains comparatively better performance and exhibits a more gradual degradation, indicating improved generalisation under low-sample and high-variability conditions.

The trends are consistent between half and full protocols, with the full protocol showing slight improvements due to increased enrolment data. Overall, the DET analysis confirms that OpenVeinNet provides stable and favourable operating characteristics across datasets, particularly in low-FMR regimes relevant to security-critical deployments.

![](images/bc5131f94d3bf8043dc729b585017499a46ba4e3434c49262a11ae462b8e0639.jpg)  
(a) Best 50 classes

![](images/81379d61430bab356611d1f26bf0d7d5f37312f03d38134f5026907517cc113c.jpg)  
(b) Worst 50 classes

![](images/fd8ee557c9594d168585ab4916a03169d321dd359e682c4fe4758d43953d42ba.jpg)  
(c) All Classes  
Fig. 10: FV-300 Database t-SNE visualisation of OpenVeinNet embeddings: (a) best 50 classes, (b) worst 50 classes, and (c) all classes.

![](images/e666c94822b38973919dc3b82d508dd9c139ccd7280a206a82b6ba3532681964.jpg)  
Fig. 11: Best 50 classes

![](images/60c11c99e12efe356250570c4930f9752cb4ce89bd797ad89c9eb00aeaeb6e37.jpg)  
Fig. 12: Worst 50 classes

![](images/63b26ddaffca5eea44e5662968d423a4191975baba090d779833bcb1ff98020a.jpg)  
Fig. 13: All Classes  
Fig. 14: FV-USM database t-SNE visualisation of OpenVeinNet embeddings: (a) best 50 classes, (b) worst 50 classes, and (c) all classes.

![](images/c45c692123955d7dc3dea3145700d7c6db798a1881c563107570f09ba2d0dfb2.jpg)  
Fig. 15: Best 50 classes

![](images/5671b5a422583f83d69cd2db47e1c13b710d394e46f8b61db1aeef6e4bd069b3.jpg)  
Fig. 16: Worst 50 classes

![](images/db58c0d442b065d4da5046729cd2ee4651b0b7883e3138690cccc06579e02c4b.jpg)  
Fig. 17: All Classes  
Fig. 18: MMCBNU database t-SNE visualisation of OpenVeinNet embeddings: (a) best 50 classes, (b) worst 50 classes, and (c) all classes.

![](images/4e47640af598b9ba4e2b407ad1a0e2bb20c886f133605f63699b3388cd25512a.jpg)  
Fig. 19: Best 50 classes

![](images/e6772dbaba32afeb776624f4564646fa900f753f48726e2acf6bd33f7e7a10da.jpg)  
Fig. 20: Worst 50 classes

![](images/041751c74f55adcc172a984641658a7dfdb937babd3496669dee72f897e4ba5b.jpg)  
Fig. 21: All Classes  
Fig. 22: PolyU database t-SNE visualisation of OpenVeinNet embeddings: (a) best 50 classes, (b) worst 50 classes, and (c) all classes.

![](images/9a129711613544706a49fb2d8a7d1bdcbb57172639d18cd98434e54d8fad6fea.jpg)  
Fig. 23: Best 50 classes

![](images/f8904a04d5a7e3a2afa9d44cd995474ff5927abc27e081a3053b09415f0f67e6.jpg)  
Fig. 24: Worst 50 classes

![](images/4af873e7afc3d10f8f35740e8629be7423dd1f9ad6a0245296b23516980cb208.jpg)  
Fig. 25: All Classes  
Fig. 26: VERA database t-SNE visualisation of OpenVeinNet embeddings: (a) best 50 classes, (b) worst 50 classes, and (c) all classes.

![](images/32c825f8674d4327f431b9036e0f3685106dc2aebcb87a055fb3d5cc7f64050c.jpg)  
(a) FV-300

![](images/bd8df656c0bb5cededd1112567260c4819e26b5a46e1fb9ad7ddbf0ed17f510d.jpg)  
(b) FV-USM

![](images/d78bcabffadf544989726f82fbbc42d803b7e8fe3bc7b58930f1d098f1d6d1b7.jpg)  
(c) MMCBNU

![](images/e4d1b939bacb3c4d8ed632bdfc1919c32689bdc26b492e3bfad0e4fdfe379072.jpg)  
(d) PolyU

![](images/be15b23398de4533af290a3bdf55a6e038cc3f1917fedc6ab5aa7411299ffae9.jpg)  
(e) VERA  
Fig. 27: DET curves for the half evaluation protocol across the five datasets: (a) FV-300, (b) FV-USM, (c) MMCBNU, (d) PolyU, and (e) VERA.

![](images/6d4d836f5c921d1af82221d995d468d8dc3c732db6a57727b19a49f3aadc631c.jpg)  
(a) FV-300

![](images/f24aaa8c02eb122e24a07f3fa231fde2fa21a5fc73bcfb3d6ee9a47441f560a6.jpg)  
(b) FV-USM

![](images/31b8f20217a0e731c429d9e6a99a77ce9d5a39b18fb82ba1af4aa2a8e393073a.jpg)  
(c) MMCBNU

![](images/19ccafe837872a5dbe5b54c9bfcd8a8c4b97144730c03c5317478e50e433e07e.jpg)  
(d) PolyU

![](images/d6928cd61db0edea9adab81f45a6b64a88c9b353cb58d42ddc120e7299c391d6.jpg)  
(e) VERA  
Fig. 28: DET curves for the full evaluation protocol across the five datasets: (a) FV-300, (b) FV-USM, (c) MMCBNU, (d) PolyU, and (e) VERA.