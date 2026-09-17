# Mask 2D-3D: Adaptive Dual-Masked Autoencoder Network for Image-to-Point Cloud Registration

Zhixin Cheng, Jiacheng Deng, Xiaotian Yin, Baoqun Yin, Richang Hong, Tianzhu Zhang

Abstract—Detection-free methods for image-to-point cloud registration are prone to erroneous correspondences caused by domain and modality discrepancies, limited sensitivity of feature extractors, and the presence of non-overlapping regions. The Masked Autoencoder (MAE) has shown strong performance in visual representation for images and point clouds. It may be helpful to apply this approach to image-to-point cloud registration—a task that requires unified feature extraction and accurate crossmodal correspondences. Standard MAE’s random masking may overlook key regions due to limited camera views, reducing registration effectiveness. To address this, we propose the Intermodal Dual-MAE Framework (ID-MAE) with a Similarity-based RL Masking Strategy (SRLM), which adaptively masks informative positions by leveraging cross-modal similarity and reinforcement learning, thus narrowing the modality gap. Our method enhances cross-modal representation learning by enforcing representation consistency during feature extraction, thereby enabling more reliable 2D–3D correspondence estimation. Experiments on RGB-D Scenes v2 and 7-Scenes benchmarks show that our method achieves state-of-the-art performance in image-to-point cloud registration.

Index Terms—Image-to-Point cloud Registration, Masked Autoencoder, Masking Strategy.

## I. INTRODUCTION

Mage-to-point cloud registration (I2P) aims to determine I the rigid transformation from the point cloud to the camera coordinate system, which involves the cross-modal matching of image and point cloud, followed by a pose estimator to compute rotation and translation matrices. Such registration is essential for tasks like 3D reconstruction [25], [26], [39], [40], [80], SLAM [22], [32]–[34], [74], and visual localization [29], [35], [36], [77], [78]. However, images are represented as regular, dense 2D grids, whereas point clouds are unordered, sparse, and irregular 3D points. The significant modality gap makes designing a model that effectively interacts between RGB and geometric modalities a challenging task.

Image-to-point cloud registration methods are typically classified into detect-then-match [3], [19], [42] and detectionfree approaches [41], [43]. Detect-then-match methods rely on independently detecting 2D and 3D keypoints and matching them via semantic features, but suffer from modality gaps and limited descriptor precision. Detection-free methods, such as 2D3D-MATR [1], use a coarse-to-fine pipeline to establish patch-level matches and refine them to dense correspondences, improving inlier ratios via contextual cues and multi-scale receptive fields. However, image feature extraction relies on texture information, whereas point cloud feature extraction depends on structural information. As a result, the gaps between modalities and the limitations of feature extractors continue to impede further accuracy improvements. Masked Autoencoders [51] have shown strong representation capability in both image and point cloud domains. However, despite some cross-modal attempts [52], [61], their potential for image-topoint cloud registration remains worth exploring. As a result, effectively leveraging MAE for cross-modal tasks remains challenging, since these tasks demand reconstructing features across different modalities while preserving precise correspondence alignment. Moreover, the commonly used random masking strategy may overlook critical regions or include irrelevant areas, limiting its ability to improve registration accuracy. Therefore, adapting both the MAE architecture and masking mechanism is essential to unlock its full potential for cross-modal alignment.

![](images/fda97e141f72142701ad325a6fc0c7302c5ca9fe06d4aecb7825a819c244fff7.jpg)  
Fig. 1. (a) Schematic of point cloud features aiding image feature reconstruction, where point cloud values denote attention-based correspondences. (b) SRLM Strategy Diagram: Due to the presence of overlapping regions (within the yellow box) and high-value registration areas (star-shaped regions) in the scene, the orange blocks in the random masking strategy indicate ineffective masks. Therefore, we use similarity-based initialization to select patches (gray blocks) in the overlapping regions. Subsequently, we employ RL-driven refinement to seek more informative masking regions (green blocks). SRLM reduces redundant masking and focuses on critical matching regions.

![](images/bd44dbbd1846429d6616609fdeccac68eb7078fa07533b1e4003128dcedd123e.jpg)  
Fig. 2. T-SNE visualization and MMD metrics (detailed in IV-A) showing improved alignment between image and point cloud features after training.

Based on the above discussions, we identify two key issues that need to be addressed to achieve accurate and robust image-to-point cloud registration. (1) How to design a unified framework that leverages MAE to enhancefeature extraction across different modalities. Although MAE has demonstrated remarkable improvements in feature extraction for singlemodal tasks, designing a unified cross-modal MAE paradigm remains challenging. If we can enhance the feature extraction capability and further leverage cross-modal guidance for local structure reconstruction, it will greatly improve the ability to extract high-quality features and alleviate domain discrepancies (see in Fig. 2), ultimately increasing the matching success rate. As shown in Fig. 1(a), the attention maps highlight the correspondences between point cloud and image features, with the point cloud features aiding in the reconstruction of the image features. (2) How to select masked regions to maximize the effectiveness of our design. Random masking may be suboptimal, as it can mask irrelevant areas due to the camera’s limited field of view, while missing key regions for registration. This not only increases computational cost but also degrades point cloud reconstruction, limiting the effectiveness of the MAE module. To address this, we propose a masking strategy focused on critical patches for registration in overlapping regions between the image and point cloud. As shown in Fig. 1(b), our method reduces redundant masked areas (orange) and selects more informative regions (green).

To address the above challenges, we propose the Adaptive Dual-Masked Autoencoder Network for Image-to-Point Cloud Registration (M23D), featuring two core components: the Intermodal Dual-MAE Framework (ID-MAE) and the Similarity-based RL Masking Strategy (SRLM). ID-MAE introduces a unified cross-modal MAE with a bidirectional design, where point cloud features assist image reconstruction. Partial masking is applied to both modalities, and a lowresolution point cloud constrains the MAE-reconstructed output, enhancing feature quality. Image reconstruction is guided by both point cloud features and original image supervision, aligning modalities and establishing correspondences. SRLM improves MAE’s effectiveness through a similarity-based initialization mechanism and RL-driven refinement. The masking strategy first targets overlapping regions by computing crossmodal similarity maps and selecting the top-k most relevant areas for masking. Then, we apply reinforcement learning [69]– [71] to make the selection of high-value regions differentiable, effectively focusing reconstruction on discriminative regions and improving registration accuracy.

Unlike transformer-based matching methods such as 2D3D-MATR, which enhance correspondence estimation by refining interactions among pre-extracted features, our approach addresses cross-modality discrepancies earlier. Specifically, the proposed Intermodal Dual-MAE explicitly constrains feature learning via cross-modal reconstruction, encouraging image and point cloud representations to encode consistent geometric and semantic cues from the outset. Recent interaction-based methods have made notable progress in addressing the crossmodality challenge of image-to-point-cloud registration. In particular, Flow-I2P introduces an innovative informationgeometric perspective, reformulating I2P registration as a manifold alignment problem and leveraging Beltrami flow to progressively refine cross-modal feature manifolds. This design effectively enhances feature interaction and improves generalization under limited training data. Our work is motivated by a complementary observation. Beyond refining feature interactions or manifold structures after feature extraction, the cross-modality gap can also be alleviated early in the representation learning process. Instead of explicitly modeling manifold evolution, we employ an intermodal dualmasked autoencoder. This method implicitly encodes crossmodal consistency through reconstruction, encouraging image and point cloud features to share aligned geometric semantics from the outset. In this sense, our approach focuses on learning more compatible representations before correspondence reasoning. It therefore complements existing interaction-based or manifold-alignment methods.

In summary, our work can be summarized as follows:

• We propose the Similarity Driven Dual-Masked Autoencoder Network (M23D), a unified architecture of the Intermodal Dual-MAE Framework (ID-MAE) with a novel Similarity-based RL Masking Strategy (SRLM) achieves strong accuracy and robustness. To our knowledge, this is the first MAE-based cross-modal design for image-topoint cloud registration.

• ID-MAE aligns modalities and establishes correspondences via a dual-MAE framework, while SDMS reduces modality gaps by selecting informative mask regions, improving efficiency and reconstruction quality.

• Extensive experiments and ablations on RGB-D Scenes v2 and 7-Scenes demonstrate the effectiveness of our approach, setting a new state-of-the-art in image-to-point cloud registration.

## II. RELATED WORK

In this section, we briefly overview related works on imageto-point cloud registration, including stereo image registration, point cloud registration, and inter-modality registration.

![](images/8794aebb5b588645978d83fee1500a550cdcd223025b520ad94fc9e09b77f5d7.jpg)  
Fig. 3. Overall pipeline of M23D. We extract image and point cloud features through the intermodal dual-MAE framework, which are then processed via attention mechanism. The point cloud and image features generate a score map using cosine similarity and maximum operations, enabling coarse-level matching and refining fine-level correspondences. Finally, PnP + RANSAC is applied to estimate the rigid transformation.

Stereo Image Registration. Detector-based methods have long dominated stereo image registration. Prior to deep learning, key points were detected using handcrafted techniques like SIFT [8] and ORB [9], which built 2D matches from local features. The advent of deep learning introduced neural networkbased detection, transforming the field. SuperGlue [11] was pioneering in using Transformers [23] for image registration, greatly enhancing local feature matching. However, the challenge of detecting repeatable interest points in non-salient areas has led to the rise of detector-free methods. Approaches like LoFTR [6] and Efficient LoFTR [7] use a coarse-to-fine pipeline with Transformers to efficiently estimate dense image matches through global receptive fields.

Point Cloud Registration. Point cloud registration has advanced from handcrafted descriptors like PPF [12], [13] and FPFH [14] to deep learning-based approaches. CoFiNet [16] introduced detector-free registration with a coarse-tofine strategy, and recent methods have replaced RANSAC [5] with deep robust estimators for better speed and accuracy. GeoTransformer [17] improves inlier ratios by integrating global information with the transformer and introduces a localto-global method for RANSAC-free registration.

Cross-modal Masked Autoencoder Methods. Existing 2D–3D studies have explored cross-modal representation learning and MAE frameworks from different perspectives. PiMAE [50] employs an interactive dual-branch MAE architecture, where image and point cloud tokens are reconstructed through complementary cross-modal masking based on geometric projection. Such a design enhances object-level semantic representations but relies on projection-based mask generation rather than adaptive selection of informative matching regions. I2P-MAE [49] transfers knowledge from large-scale 2D pre-trained models through semantic-guided point masking and hierarchical point reconstruction, where visible point tokens are selected according to 2D semantic priors. Although effective for generic 3D representation learning, its masking strategy is determined by fixed semantic guidance rather than pair-specific cross-modal correspondence. CrossNet [75] aligns image and point cloud feature spaces via cross-modal contrastive learning, emphasizing global feature consistency and discriminative capability without adopting MAE-based masked reconstruction. Inter-MAE [76] introduces image features to supervise point-cloud MAE pre-training through crossmodal contrastive learning while retaining conventional random masking for point reconstruction. Consequently, crossmodal interaction is mainly achieved through feature supervision instead of adaptive mask optimization. Joint-MAE [79] constructs a unified 2D–3D MAE by jointly encoding image and point cloud tokens and reconstructing both modalities with modality-specific decoders. It adopts random masking for both modalities and emphasizes generic multimodal pretraining, without explicitly considering overlap-aware masking or registration-oriented correspondence learning.

Overall, these methods demonstrate the effectiveness of MAE and cross-modal learning for 2D–3D representation modeling and provide valuable insights into cross-modal feature interaction. However, their architectures are primarily designed for generic representation learning or perception tasks, while their masking strategies are generally projectionbased, random, or guided by fixed semantic priors. None of them explicitly optimize mask generation according to cross-modal overlap, correspondence reliability, or the registration objective. In contrast, our method introduces a dual-MAE framework tailored for image-to-point cloud registration, where image and point cloud features are jointly optimized through cross-modal reconstruction. Furthermore, the proposed Similarity-based RL Masking Strategy (SRLM) first initializes masks using bidirectional cross-modal similarity and then refines them via reinforcement learning, enabling adaptive selection of informative regions that directly benefit crossmodal correspondence learning and registration accuracy.

Inter-modality Registration. Inter-modal registration presents greater challenges compared to intra-modal registration because of the significant domain discrepancies involved. Traditional approaches typically employ a detectthen-match strategy. For instance, 2D3D-Matchnet [19] uses SIFT [8] and ISS [18] to extract key points from images and point clouds, constructing patches around these key points. It then uses CNNs and PointNet [38] to extract features and build descriptors for matching. P2-Net [3] introduces a joint learning framework with a comprehensive reception mechanism, using a single forward pass to detect key locations and extract descriptors, enabling efficient matching through contrastive constraints. Unfortunately, the inefficiency of keypoint extraction in cross-modal scenarios has led to significant accuracy loss, prompting the emergence of detection-free methods. 2D3D-MATR [1] adopts a coarse-to-fine matching process, using a transformer-based approach to establish patch-level matches and then seeking fine-grained matches within them, followed by regressing rigid transformations using PNP+RANSAC [4], [5]. This detection-free method overcomes the challenge of obtaining repeatable key points and makes 2D-3D descriptors more consistent. Using transformers’ global receptive fields and multi-level methods significantly increases the inlier ratio of matches. B2-3Dnet [2] further enhance cross-modal correspondence learning by leveraging covariance-guided feature alignment to improve the robustness and consistency of descriptors. Based on this, CA-I2P [62] introduces channel adaptation and global optimal selection to better align cross-modal features and reduce redundant matches, achieving improved registration accuracy. Flow-I2P [10] improves image-to-point-cloud registration by using Beltrami flow for better manifold alignment and enhanced registration accuracy. Our method advances detection-free approaches by introducing a dual MAE framework (ID-MAE) combined with a Similarity-based RL Masking Strategy (SRLM), where the dual MAE framework aligns modalities and establishes correspondences, while SRLM selectively masks high-value regions through similarity-based initialization and RL-driven refinement, making it a state-of-the-art solution for image-to-point cloud registration.

![](images/582b04f13362d976791ab900eae029c9ad7a29c8ca5ddebab242e250589fe0c9.jpg)  
Fig. 4. Overall pipeline of ID-MAE. After selecting the mask regions through SRLM, the point cloud MAE module constructs a loss constraint by reconstructin the point cloud from its low-resolution version. The image MAE module incorporates point cloud features to assist in reconstruction and is supervised by th original features. After post-processing, $f _ { i }$ and $f _ { p }$ are output for subsequent matching.

## III. METHOD

## A. Overview

Given an image $I \in \mathbb { R } ^ { H \times W \times 3 }$ and a point cloud $P \in$ $\mathbb { R } ^ { N \times 3 }$ from the same scene, the objective of the registration between images and point clouds is to determine the rigid transformation [R, t] from the point cloud coordinate system to the camera coordinate system. Here, W and H are the image’s width and height, while N is the number of points. The transformation consists of a 3D rotation $R \in S O ( 3 )$ and a 3D translation vector $t \in \mathbb { R } ^ { 3 }$

Our method, M2-3D, depicted in Fig. 3, adopt a dual MAEenhanced detection-free registration paradigm, constructing the modality-aligned and corresponded image and point cloud features through the intermodal dual-MAE framework. After passing through a series of attention modules, similarity is calculated to generate a score map, enabling patch-level matches. High-resolution images and point cloud features are then used to refine dense correspondences from the patch matches. Finally, the PnP+RANSAC [4] algorithm effectively regresses the rigid transformation.

## B. Intermodal Dual-MAE Framework

We utilize ResNet [24] with FPN [27] and KPFCNN [28] to extract features from images and point clouds, respectively. The 2D features and 3D features are downsampled at the lowest resolution to obtain $f _ { i 1 } \in \mathbb { R } ^ { ( h \times w ) \times c }$ and $f _ { p 1 } \in \mathbb { R } ^ { n \times c }$ For the extracted features at the lowest resolution $f _ { i }$ and $f _ { p } ,$ positional encoding is applied to enhance the features. To prepare for the subsequent MAE, we do not adopt the conventional pretraining mode of MAE, as pretraining significantly increases computational overhead and is often constrained in certain scenarios. Instead, we aim to enhance the feature extraction capability of the network encoder through unsupervised MAE during normal training. This approach facilitates modality aggregation and establishes correspondences between images and point clouds. We can observe the overall pipeline of Intermodal Dual-MAE Framework (ID-MAE) from Fig. 4.

1) Dual-modal MAE Enhancement: Point MAE module adopts a self-supervised process, where the point cloud feature $f _ { p 1 }$ is masked based on the Similarity-based RL Masking Strategy (SRLM) (we will elaborate on this later), resulting in $f _ { p m } ~ \in ~ \mathbb { R } ^ { ( n - m _ { p } ) \times c }$ , where $m _ { p }$ denotes the number of masked points. For the encoder, we employ local self-attention instead of global self-attention, to construct a Transformerbased encoder. We fill the masked positions in $f _ { p m }$ with learnable mask tokens of dimension $\mathbb { R } ^ { 1 \times c }$ , completing the feature to obtain $f _ { p c } \in \mathbb { R } ^ { n \times c }$ with the exact dimensions as $f _ { p 1 }$ . After passing through the decoder, the mask tokens learn the corresponding point cloud features, completing the point cloud reconstruction process. We utilize standard point cloud cross-attention blocks in the point cloud decoder, along with self-attention blocks. The reconstructed point cloud features $f _ { p 1 } \in \mathbb { R } ^ { n \times c }$ are finally mapped to the point cloud $P _ { 2 } \in \mathbb { R } ^ { n \times 3 }$ while the features $f _ { p 2 }$ obtained from the encoder are decoded to $P _ { 1 } \in \mathbb { R } ^ { n \times 3 }$ . The two point clouds $P _ { 1 }$ and $P _ { 2 }$ are supervised by an $L _ { 2 }$ loss. Here, n denotes the number of points in the reconstructed point cloud, and m indexes the m-th point.

$$
\mathcal { L } _ { \mathrm { p m a e } } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \Vert P _ { 1 , m } - P _ { 2 , m } \Vert _ { 2 } ^ { 2 } ,\tag{1}
$$

after filtering out some excessively small point cloud patches, we output the point cloud feature $f _ { p }$ for subsequent matching. As illustrated in Fig. 4, we adopt a KPFCNN backbone that naturally produces a hierarchical multi-scale point pyramid from the input point cloud P. The point MAE operates on the coarsest-scale level of this pyramid as its reconstruction target, while finer-scale point features are preserved for subsequent matching and registration. Therefore, the effective downsampling ratio of the low-resolution point set is implicitly determined by the backbone’s subsampling configuration, and can be reproduced by reporting the actual point counts at different hierarchy levels. Our masking strategy further selects informative regions on the coarsest-level point tokens, producing masked subsets $P _ { m }$ for the training-only MAE branch. At the dataset level, a random subsampling is optionally applied only to cap the maximum number of raw input points for efficiency; this preprocessing neither defines the low-resolution point cloud used in ID-MAE nor affects the hierarchical structure. Importantly, the low-resolution point cloud is solely used as a reconstruction target during training and is removed at inference, thus it does not participate in similarity computation, masking selection, or pose estimation. This design naturally aligns with the coarse-to-fine registration paradigm, where global structural consistency is first captured at coarse resolution and progressively refined at finer scales.

For point clouds, we reconstruct them for supervision due to their sparsity, while for images, we directly supervise the features. And since the point cloud perspective contains more information, we aim to introduce cross-modal information to assist in the reconstruction of the image.

2) Cross-Modal Guided Reconstruction: Typically, MAE encoders benefit from learning a generalized encoder capable of capturing high-dimensional data representations of both images and point clouds. Due to the differences between the two modalities, a dedicated decoder is required to decode the high-level latent data across two different modalities. The encoder part of image MAE module is similar to that of the point cloud. The image feature $f _ { i 1 }$ undergoes SRLM, resulting in $f _ { i m } ~ \in ~ \mathbb { R } ^ { ( h \times w - \bar { m _ { i } } ) \times c }$ , where $m _ { i }$ denotes the number of masked patches.

We leverage point cloud features to restore the masked regions in the image, facilitating the identification of corresponding structures between the two modalities and enhancing semantic alignment. A learnable mask token of dimension $\mathbb { R } ^ { 1 \times c }$ is introduced and duplicated to fill the missing positions in the masked image features, yielding a complete feature map $f _ { i c } ~ \in ~ \mathbb { R } ^ { ( h \times w ) \times c }$ . The feature $f _ { i c }$ is used as the query (Q), while the point cloud representation $f _ { p 1 }$ serves as both the key (K) and value (V) in the cross-attention mechanism to reconstruct image features.

These components are jointly fed into the decoder to model inter-modal dependencies and produce the updated and restored feature $\bar { f _ { i r } } \in \mathbb { R } ^ { ( h \times w ) \times c }$ . This decoding process enhances alignment and improves reconstruction accuracy. The restored feature $f _ { i r }$ is obtained by aggregating relevant point cloud features through a weighted summation based on feature similarity, as defined below:

$$
f _ { i r } = \sum _ { b \in \mathcal { N } _ { f _ { p 1 } } ( a ) } \frac { e ^ { s _ { a b } } } { \sum _ { l \in \mathcal { N } _ { f _ { p 1 } } ( a ) } e ^ { s _ { a l } } } f _ { i c } ^ { b } ,\tag{2}
$$

where $\mathcal { N } _ { f _ { p 1 } } ( a )$ denotes the neighborhood of the a-th feature in $f _ { p 1 }$ associated with $f _ { i c }$ . The similarity $s _ { a b }$ between $f _ { p 1 } ^ { a }$ and $f _ { i c } ^ { b }$ is computed as:

$$
s _ { a b } = f _ { p 1 } ^ { a \phantom { \dagger } } W f _ { i c } ^ { b } ,\tag{3}
$$

where W is a learnable parameter matrix. This formula calculates the similarity between $f _ { i c }$ and $f _ { p 1 }$ features, applies a weighted summation over the neighboring image features $f _ { p 1 }$ and generates the reconstructed target feature $f _ { i r }$ . This process effectively fuses the point cloud and image features, providing the necessary support for subsequent decoding. The $f _ { i 1 }$ feature is passed through the encoder to obtain $f _ { i 2 } \in \bar { \mathbb { R } } ^ { ( h \times w ) \times c }$ , which is constrained during the reconstruction process. N represents the number of patches extracted from the image, corresponding to the total number of feature tokens in the first dimension of $f _ { i 2 }$ . The n-th patch refers to a specific patch among these N patches.

$$
\mathcal { L } _ { \mathrm { i m a e } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left( f _ { i 2 , n } - f _ { i r , n } \right) ^ { 2 } ,\tag{4}
$$

a lightweight three-stage CNN [64] is used to extract image features at three different scales, capturing multi-level spatial information. The resulting features form a feature pyramid denoted as $f _ { i } ,$ , which serves as input for subsequent process.

Discussion. Why do we choose to reconstruct image features using point cloud features rather than the other way around?

![](images/6bc3d10e003c74c3c5d88dda4abe4921e53a2340ca54a60f4416fb89baaab271.jpg)  
Fig. 5. Overall pipeline of SRLM, where similarity-based masking selection is used for prior initialization, followed by RL-driven refinement.

The reason lies in the fact that point clouds and images do not correspond exactly. Compared to images, point clouds provide a broader scene view and richer information. Thus, reconstructing point cloud features from images may suffer from information loss, degrading reconstruction quality and constraints. Moreover, restricting point cloud features to the image range requires ground truth R, t, which we aim to predict, making this approach unreasonable.

## C. Similarity-based RL Masking Strategy

In image-to-point-cloud registration, due to the existence of non-overlapping regions in the image and point cloud views, when image or point cloud patches fall within these areas, it can lead to completely incorrect matches, making these computations meaningless. Additionally, since the approach follows a coarse-to-fine paradigm, this can result in large misalignments during fine-grained matching, which is undesirable. Therefore, we aim to increase the likelihood of selecting regions within the overlapping areas through similarity-based selection. However, due to domain differences in image and point cloud features, areas with high similarity cannot be directly considered as critical matching regions. Since the task of selecting key regions is inherently nondifferentiable, we aim to address this challenge by employing a reinforcement learning strategy, allowing us to effectively solve the problem. Thus to improve cross-modal learning, we propose the Similarity-based RL Masking Strategy (SRLM), which masks regions of critical patches. SRLM comprises a Similarity-based Masking Selection (SMS) and RL-driven Refinement (RLR). The overall process of SDMS is presented in Fig. 5.

1) Similarity-based Masking Selection: The camera’s limited field of view may miss discriminative areas for registration. Expanding the masked regions increases computational costs and degrades point cloud reconstruction. Observing the matching process, we find that correspondences may rely on high similarity between image and point cloud features, which should be masked as initial state to enhance the model’s perception. We compute the similarity maps: the Image-to-Point Map (I2P Map) and the Point-to-Image Map (P2I Map). By setting similarity thresholds and performing topk selection, we determine the positions to mask for both images and point clouds, denoted as image mask and point mask, respectively. Using these guided positions, we apply Mask-Guided Selection, masking the corresponding areas to obtain $I _ { m }$ and $P _ { m }$ . Through the above design, we roughly determine the initial positions of the image-point cloud mask blocks, which will be used as the initial state input in the subsequent reinforcement learning process. For features $f _ { i m } \in$ $\mathbb { R } ^ { ( h \times \bar { w } - m _ { i } ) \times c }$ and $f _ { p m } \in \mathbb { R } ^ { ( n - \bar { m } _ { p } ) \times c }$ $m _ { i }$ and $m _ { p }$ represent the number of masked regions in the image and point cloud, respectively.

While similarity evaluation is indeed widely used in registration tasks, it is typically applied at the correspondence or matching stage. In contrast, our similarity-aware masking strategy leverages such cues during representation learning, guiding the MAE to focus on informative cross-modal regions before correspondence estimation. This shift in usage is crucial: rather than directly establishing correspondences, similarity is used to prioritize which regions should be preserved or reconstructed during masked modeling, enabling the MAE to learn more correspondence-aware representations. However, due to repetitive object textures in the scene and domain gaps in the initialization between images and point clouds, a similarity-only selection strategy may not be sufficient to fully unleash the potential of our bidirectional MAE architecture.

2) RL-driven Refinement: To enhance the accuracy of image-to-point cloud matching, we introduce a reinforcement learning (RL)-based strategy optimization to refine the selection of masked regions (as shown in Algorithm 1). Instead of relying on fixed or heuristic masking schemes, the proposed RL framework adaptively adjusts which regions to mask or preserve during training, enabling the model to focus on more informative and reliable cross-modal areas. The RL component is designed as an auxiliary optimization mechanism that operates alongside the MAE framework, rather than a standalone decision-making module. By observing the current similarity-aware state, the policy network dynamically refines the selection of masked regions, thereby reducing masking randomness and stabilizing MAE optimization. A task-aligned reward function derived from the matching loss guides policy updates, ensuring that the learned masking strategy consistently supports the main registration objective. Through this design, the RL-based masking strategy facilitates more reliable feature learning for image-to-point cloud matching while avoiding additional inference overhead, as it is only applied during training.

The initial state in the RL process is determined by the similarity obtained from SMS. The reward function plays a critical role in this process, as it guides the optimization of the policy. In our approach, the reward is computed based on $L _ { i }$ (described in Equation 11 in III-D), where the circle loss effectively constrains overall matching accuracy, making it an ideal candidate for the reward. Specifically, since better matching performance results in smaller values for $L _ { i } ,$ the reward increases as matching accuracy improves. Thus, the reward is defined as:

$$
R = \frac { 1 } { L _ { i } + \delta } ,\tag{5}
$$

where δ is a small constant to prevent division by zero.

The probabilities of selecting image and point cloud masks, denoted as $P r o b _ { i }$ and $P r o b _ { p }$ , are computed based on the output of the policy network for both regions. These probabilities

reflect the network’s confidence in selecting each region and are calculated as follows:

$$
\begin{array} { r } { P r o b _ { i } = \log ( \pi _ { \theta } ( a _ { \mathrm { i m a s k } } | s _ { \mathrm { i m a g e } } ) ) , } \end{array}\tag{6}
$$

$$
P r o b _ { p } = \log ( \pi _ { \theta } ( a _ { \mathrm { p m a s k } } | s _ { \mathrm { p o i n t } } ) ) ,\tag{7}
$$

where $\pi _ { \theta } { \big ( } a _ { \mathrm { i m a s k } } | s _ { \mathrm { i m a g e } } { \big ) }$ and $\pi _ { \theta } { \left( a _ { \mathrm { p m a s k } } \middle | s _ { \mathrm { p o i n t } } \right) }$ represent the probabilities output by the policy network for the image and point cloud mask selections, respectively.

The overall log-probability, logprob, is computed by averaging the log-probabilities of both modalities as follows:

$$
\log ^ { p r o b } = \frac { 1 } { 2 } \log \left[ 1 + \left( \frac { 1 } { h \cdot w } \sum _ { i = 1 } ^ { h \cdot w } P r o b _ { i } \right) \cdot \left( \frac { 1 } { n } \sum _ { j = 1 } ^ { n } P r o b _ { p } \right) \right] ,\tag{8}
$$

where $h \cdot w$ and n represent the number of elements in the image and point cloud regions, respectively. This averaging process ensures that both image and point cloud regions contribute equally to the overall decision-making process by combining their confidence levels.

In RL, the action corresponds to selecting the masked regions based on the current state. In our approach, actions involve determining which regions should be masked and which should remain visible. The action space consists of all possible combinations of masked regions, and the policy network learns to select the optimal regions by maximizing the reward. The network computes the log-probability of the selected action, reflecting its confidence in the chosen masked regions. Specifically, the gradient of the policy is estimated as:

$$
\nabla \theta J ( \theta ) = \mathbb { E } _ { t } \left[ \nabla \log \pi _ { \theta } ( a _ { t } | s _ { t } ) \cdot R _ { t } \right] ,\tag{9}
$$

where $\pi _ { \boldsymbol { \theta } } \big ( a _ { t } | \boldsymbol { s } _ { t } \big )$ represents the probability of selecting action $a _ { t }$ given state $s _ { t }$ , parameterized by θ. The log-probability $\nabla$ log $\pi _ { \boldsymbol { \theta } } \big ( a _ { t } | \boldsymbol { s } _ { t } \big )$ is used to compute the gradient for policy update, while $R _ { t }$ is the reward associated with the action taken at time $t ,$ guiding the policy towards optimal actions.

The reinforcement loss $L _ { t }$ is then computed as:

$$
L _ { t } = - l o g ^ { p r o b } \times R .\tag{10}
$$

This policy loss is added to the main loss, steering the optimization of the network parameters. By maximizing the reward and minimizing the total loss, the policy network gradually learns to select the best masked regions, ultimately improving the matching accuracy between the image and point cloud. This design helps stabilize the MAE training process by reducing randomness in mask selection and alleviating sensitivity to hyperparameters. The resulting adaptive masking strategy facilitates more reliable image-to-point cloud matching and improves robustness in practice.

Algorithm 1 RL-driven Refinement for Mask Region Selec  
tion   
Input: Image feature $\mathbf { F } _ { I } ,$ point-cloud feature $\mathbf { F } _ { P } ,$ , SMS, policy network $\pi _ { \theta } .$   
MAE network $\mathcal { M } .$ , matching network G, and constant $\delta .$   
Output: Image mask $a _ { \mathrm { i m a s k } }$ and point-cloud mask $a _ { \mathrm { p m a s k } }$   
Similarity-aware initialization:   
$( s _ { \mathrm { i m a g e } } , s _ { \mathrm { p o i n t } } ) \gets \mathrm { S M S } ( \mathbf { F } _ { I } , \mathbf { F } _ { P } ) .$   
Policy-based mask refinement:   
• Predict mask-selection distributions:   
$\pi _ { \theta } ( a _ { \mathrm { i m a s k } } \mid s _ { \mathrm { i m a g e } } ) , \qquad \pi _ { \theta } ( a _ { \mathrm { p m a s k } } \mid s _ { \mathrm { p o i n t } } ) .$   
• Sample masking actions:   
$a _ { \mathrm { i m a s k } } \sim \pi _ { \boldsymbol \theta } ( \cdot \mid s _ { \mathrm { i m a g e } } ) , \qquad a _ { \mathrm { p m a s k } } \sim \pi _ { \boldsymbol \theta } ( \cdot \mid s _ { \mathrm { p o i n t } } ) .$   
• Apply the masks:   
$\widetilde { \mathbf { F } } _ { I } \gets \mathrm { M a s k } ( \mathbf { F } _ { I } , a _ { \mathrm { i m a s k } } ) , \qquad \widetilde { \mathbf { F } } _ { P } \gets \mathrm { M a s k } ( \mathbf { F } _ { P } , a _ { \mathrm { p m a s k } } ) .$   
Reconstruction and matching:   
• Reconstruct the masked features:   
$( \mathbf { F } _ { I } ^ { r } , \mathbf { F } _ { P } ^ { r } )  \mathcal { M } ( \widetilde { \mathbf { F } } _ { I } , \widetilde { \mathbf { F } } _ { P } ) .$   
• Estimate correspondences and matching loss:   
$\begin{array} { r } { \mathcal { C }  \mathcal { G } ( \mathbf { F } _ { I } ^ { r } , \mathbf { F } _ { P } ^ { r } ) , \qquad L _ { i }  \mathrm { C i r c l e L o s s } ( \mathcal { C } ) . } \end{array}$   
Reward:   
$R  \frac { 1 } { L _ { i } + \delta } .$   
Policy probability:   
• Compute modal log-probabilities:   
$P r o b _ { i } \gets \log \pi _ { \theta } \left( a _ { \mathrm { i m a s k } } \ | \ s _ { \mathrm { i m a g e } } \right)$   
$P r o b _ { p } \gets \log \pi _ { \theta } \left( a _ { \mathrm { p m a s k } } \mid s _ { \mathrm { p o i n t } } \right)$   
• Aggregate the two modalities:   
$\log ^ { p r o b } \gets \frac { 1 } { 2 } \log \left[ 1 + \left( \frac { 1 } { h w } \sum _ { i = 1 } ^ { h w } P r o b _ { i } \right) \left( \frac { 1 } { n } \sum _ { j = 1 } ^ { n } P r o b _ { p } \right) \right] .$   
Policy optimization:   
• Compute the policy and total losses:   
$L _ { t } \gets - \log ^ { p r o b } R , \qquad L _ { \mathrm { t o t a l } } \gets L _ { \mathrm { m a i n } } + \lambda _ { t } L _ { t } .$   
• Update the policy and main-network parameters:   
$\theta  \theta - \eta _ { \theta } \nabla _ { \theta } L _ { t } , \qquad \phi  \phi - \eta _ { \phi } \nabla _ { \phi } L _ { \mathrm { t o t a l } } .$   
Return: $a _ { \mathrm { i m a s k } }$ and $a _ { \mathrm { p m a s k } } .$

Training stability discussion. Although the RL module is tightly coupled with the registration task, several design choices are adopted to ensure stable optimization. First, the RL policy is not activated from the beginning of training. As shown in our implementation, the network is first trained with MAE-based reconstruction losses during a warm-up phase (set for the first 20 epochs), allowing cross-modal representations to stabilize before policy learning is introduced. This significantly reduces the variance and sensitivity of subsequent policy updates. Second, the reward signal is dense and continuous, defined as the inverse of the main task loss at each iteration. Unlike sparse or delayed rewards, this design provides immediate and consistent feedback to the policy, effectively mitigating reward sparsity. Third, the policy loss is incorporated as a lightweight regularization term with a small weighting factor, ensuring that the primary optimization objective remains dominated by the registration loss. As a result, the RL component softly biases the masking strategy rather than aggressively steering the feature-learning process, reducing the risk of co-adaptation or training collapse. To further improve stability, the policy employs factorized Bernoulli sampling at the token level, and the log-probabilities of mask selections are averaged across multiple token-wise decisions. This design significantly reduces policy gradient variance compared to combinatorial masking actions. Finally, the policy state is initialized and constrained by similaritybased priors, limiting exploration to informative regions and further mitigating co-adaptation with the main network. As a result, the RL component serves as a lightweight refinement mechanism rather than a dominant optimization driver.

## D. Model Training & Inference

Our training consists of two stages: first, we train the point MAE module, followed by the image MAE module. As image feature reconstruction depends on point cloud features, we prioritize training the point MAE module to ensure highquality feature extraction. Our ID-MAE is employed solely during training and omitted at inference.

To obtain the total loss, first let us examine the loss functions for the coarse and fine matching networks. Both $\mathcal { L } _ { \mathrm { c o a r s e } }$ and ${ \mathcal L } _ { \mathrm { f i n e } }$ use the general circle loss [44], [45]. Given an anchor descriptor $d _ { i } ,$ , the descriptors of its positive and negative pairs are $\mathcal { D } _ { i } ^ { \dot { P } }$ and $\mathcal { D } _ { i } ^ { N }$ , respectively:

$$
\mathcal { L } _ { i } = \frac { 1 } { \gamma } \log \left[ 1 + \bigg ( \sum _ { d ^ { j } \in \mathcal { D } _ { i } ^ { P } } e ^ { \beta _ { p } ^ { i , j } ( d _ { i } ^ { j } - \Delta _ { p } ) } \bigg ) \cdot \bigg ( \sum _ { d ^ { k } \in \mathcal { D } _ { i } ^ { N } } e ^ { \beta _ { n } ^ { i , k } ( \Delta _ { n } - d _ { i } ^ { k } ) } \bigg ) \right] ,
$$

where $d _ { i } ^ { j }$ is the $L _ { 2 }$ feature distance, $\beta _ { p } ^ { i , j } = \gamma \lambda _ { p } ^ { i , j } ( d _ { i } ^ { j } - \Delta _ { p } )$ and $\beta _ { n } ^ { i , k } \doteq \gamma \lambda _ { n } ^ { i , k } ( \Delta _ { n } - d _ { i } ^ { k } )$ are the individual weights for the positive and negative pairs, with $\lambda _ { p } ^ { i , j }$ and $\lambda _ { n } ^ { i , k }$ as scaling factors [65].

Combining the discussions above, the total loss is composed of three key components: the MAE loss for image and point cloud $( { \mathcal { L } } _ { \mathrm { i m a e } } , { \mathcal { L } } _ { \mathrm { p m a e } } )$ , the aggregation loss $\mathcal { L } _ { t }$ from the SRLM module, and the matching loss $\mathcal { L } _ { d }$ (including $\mathcal { L } _ { \mathrm { c o a r s e } }$ and ${ \mathcal { L } } _ { \mathrm { f i n e } } )$ from the matching process. The total loss is computed as:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \gamma _ { 1 } \mathcal { L } _ { \mathrm { i m a e } } + \gamma _ { 2 } \mathcal { L } _ { \mathrm { p m a e } } + \gamma _ { 3 } \mathcal { L } _ { t } + \gamma _ { 4 } \mathcal { L } _ { d } ,\tag{12}
$$

where $\gamma _ { i }$ are hyperparameters balancing the contribution of different loss terms.

## IV. EXPERIMENTS

## A. Datasets and Implementation Details

Based on the 2D3D-MATR benchmark, we conducted extensive experiments and ablation studies on two challenging benchmarks: RGB-D Scenes v2 [20] and 7Scenes [21].

Dataset. RGB-D Scenes Dataset v2 consists of 14 scenes containing furniture (chair, coffee table, sofa, table) and a subset of the objects in the RGB-D Object Dataset (bowls, caps, cereal boxes, coffee mugs, and soda cans). For each scene, we create point cloud fragments from every 25 consecutive depth frames and sample one RGB image per 25 frames. We select image-point-cloud pairs with an overlap ratio of at least 30%. Scenes 1-8 are used for training, 9-10 for validation, and 11- 14 for testing, resulting in 1,748 training pairs, 236 validation pairs, and 497 testing pairs.

The 7-Scenes dataset is a collection of tracked RGB-D camera frames. All 7 indoor scenes were recorded from a handheld Kinect RGB-D camera at 640×480 resolution. We select image to point-cloud pairs from each scene with at least 50% overlap, adhering to the official sequence split for training, validation, and testing. This results in 4,048 training pairs, 1,011 validation pairs, and 2,304 testing pairs.

TABLE I  
EVALUATION RESULTS ON RGB-D SCENES V2. TEAL NUMBERS HIGHLIGHT THE BEST, THE SECOND BEST ARE BOLDFACED AND THE BASELINE ARE UNDERLINED.
<table><tr><td>Model</td><td>Scene.11</td><td>Scene.12</td><td>Scene.13</td><td>Scene.14</td><td>Mean</td></tr><tr><td>Mean depth (m)</td><td>1.74</td><td>1.66</td><td>1.18</td><td>1.39</td><td>1.49</td></tr><tr><td colspan="6">Inlier Ratio ↑</td></tr><tr><td>FCGF-2D3D [67]</td><td>6.8</td><td>8.5</td><td>11.8</td><td>5.4</td><td>8.1</td></tr><tr><td>P2-Net [3]</td><td>9.7</td><td>12.8</td><td>17.0</td><td>9.3</td><td>12.2</td></tr><tr><td>Predator-2D3D [68]</td><td>17.7</td><td>19.4</td><td>17.2</td><td>8.4</td><td>15.7</td></tr><tr><td>2D3D-MATR [1]</td><td>32.8</td><td>34.4</td><td>39.2</td><td>23.3</td><td>32.4</td></tr><tr><td>FreeReg [55]</td><td>36.6</td><td>34.5</td><td>34.2</td><td>18.2</td><td>30.9</td></tr><tr><td>B2-3Dnet [2]</td><td>36.4</td><td>32.7</td><td>43.8</td><td>27.4</td><td>35.1</td></tr><tr><td>CA-I2P [62]</td><td>38.6</td><td>40.6</td><td>38.9</td><td>24.0</td><td>35.5</td></tr><tr><td>Flow-I2P [10]</td><td>49.6</td><td>44.0</td><td>36.5</td><td>30.4</td><td>40.1</td></tr><tr><td>M23D (ours)</td><td>48.1</td><td>48.4</td><td>46.3</td><td>31.4</td><td>43.5</td></tr><tr><td colspan="6">Feature Matching Recall ↑</td></tr><tr><td>FCGF-2D3D [67] P2-Net [3]</td><td>11.1 48.6</td><td>30.4 65.7</td><td>51.5 82.5</td><td>15.5</td><td>27.1 59.6</td></tr><tr><td>Predator-2D3D [68]</td><td>86.1</td><td>89.2</td><td>63.9</td><td>41.6 24.3</td><td>65.9</td></tr><tr><td>2D3D-MATR [1]</td><td>98.6</td><td>98.0</td><td>88.7</td><td>77.9</td><td>90.8</td></tr><tr><td>FreeReg [55]</td><td>91.9</td><td>93.4</td><td>93.1</td><td>49.6</td><td>82.0</td></tr><tr><td>B2-3Dnet [2]</td><td>100.0</td><td>99.0</td><td>92.8</td><td>85.8</td><td>94.4</td></tr><tr><td>CA-I2P [62]</td><td>100.0</td><td>100.0</td><td>91.8</td><td>82.7</td><td>93.6</td></tr><tr><td></td><td></td><td></td><td>94.5</td><td></td><td>93.3</td></tr><tr><td>Flow-I2P [10]</td><td>100.0</td><td>100.0</td><td></td><td>78.7</td><td></td></tr><tr><td>M23D (ours)</td><td>100.0</td><td>100.0</td><td>92.8</td><td>83.6</td><td>94.1</td></tr><tr><td colspan="6">Registration Recall ↑</td></tr><tr><td>FCGF-2D3D [67]</td><td>26.5</td><td>41.2</td><td>37.1</td><td>16.8</td><td>30.4</td></tr><tr><td>P2-Net [3]</td><td>40.3</td><td>40.2</td><td>41.2</td><td>31.9</td><td>38.4</td></tr><tr><td>Predator-2D3D [68]</td><td>44.4</td><td>41.2</td><td>21.6</td><td>13.7</td><td>30.2</td></tr><tr><td>2D3D-MATR [1]</td><td>63.9</td><td>53.9</td><td>58.8</td><td>49.1</td><td>56.4</td></tr><tr><td>FreeReg+Kabsch [55]</td><td>38.7</td><td>51.6</td><td>30.7</td><td>15.5</td><td>34.1</td></tr><tr><td>FreeReg+PnP [55]</td><td>74.2</td><td>72.5</td><td>54.5</td><td>27.9</td><td>57.3</td></tr><tr><td>B2-3Dnet [2]</td><td>58.3</td><td>60.8</td><td>74.2</td><td>60.2</td><td>63.4</td></tr><tr><td>CA-I2P [62]</td><td>68.1</td><td>73.5</td><td>63.9</td><td>47.8</td><td>63.3</td></tr><tr><td>Flow-I2P [10]</td><td>90.0</td><td>65.9</td><td>54.8</td><td>63.0</td><td>68.4</td></tr><tr><td>M23D (ours)</td><td>88.9</td><td>76.5</td><td>83.5</td><td>63.3</td><td>78.0</td></tr></table>

The KITTI Odometry dataset contains 22 image and point cloud sequences, with 11 sequences providing ground-truth calibration. We use sequences 00–08 for training and 09– 10 for testing. To simulate mis-registration, a 2D translation within ±10 m and an unconstrained rotation around the upaxis are applied. Images are downsampled to $1 6 0 \times 5 1 2$ and point clouds to 40,960 points.

T-SNE & MMD. t-SNE [63] is a dimensionality reduction technique that projects high-dimensional data into 2D or 3D while preserving local structure, making it useful for visualizing feature distributions. MMD is a non-parametric metric that measures the difference between two distributions in a kernelbased feature space. Both are commonly used in tasks such as domain adaptation and cross-modal representation analysis.

Implementation Details. We use an NVIDIA Geforce RTX 3090 GPU for training. The entire pipeline is implemented using PyTorch. In the image decoder, the cross-attention module is applied once, while the self-attention module is applied three times. The threshold for the I2P map is set to 0.5, with $k _ { 1 } = 3 5$ in the top-k selection. Similarly, the threshold for the P2I map is set to 0.5, with $k _ { 2 } ~ = ~ 1 5$ in the top-k selection. For loss $\gamma _ { 1 } = \gamma _ { 2 } = \gamma _ { 3 } = \gamma _ { 4 } = 1$

Metrics. We evaluate our method using several key metrics: Inlier Ratio (IR), Feature Matching Recall (FMR), and Registration Recall (RR).

TABLE II  
EVALUATION RESULTS ON 7SCENES. TEAL NUMBERS HIGHLIGHT THE BEST, THE SECOND BEST ARE BOLDFACED AND THE BASELINE ARE UNDERLINED.
<table><tr><td>Model</td><td>Chs</td><td>Fr</td><td>Hds</td><td>Off</td><td>Pmp</td><td>Kit</td><td>Strs</td><td>Mean</td></tr><tr><td>Mean depth(m)</td><td>1.78</td><td>1.55</td><td>0.80</td><td>2.03</td><td>2.25</td><td>2.13</td><td>1.84</td><td>1.77</td></tr><tr><td colspan="9">Inlier Ratio ↑</td></tr><tr><td>FCGF-2D3D [67] P2-Net [3]</td><td>34.2</td><td>32.8</td><td>14.8</td><td>26</td><td>23.3</td><td>22.5</td><td>6.0</td><td>22.8</td></tr><tr><td>Predator-2D3D [68]</td><td>55.2</td><td>46.7</td><td>13.0</td><td>36.2</td><td>32.0</td><td>32.8</td><td>5.8</td><td>31.7 23.4</td></tr><tr><td>2D3D-MATR [1]</td><td>34.7</td><td>33.8</td><td>16.6</td><td>25.9</td><td>23.1</td><td>22.2</td><td>7.5</td><td>50.1</td></tr><tr><td>B2-3Dnet [2]</td><td>72.1</td><td>66.0</td><td>31.3 33.1</td><td>60.7</td><td>50.2</td><td>52.5</td><td>18.1</td><td>50.9</td></tr><tr><td>CA-I2P [62]</td><td>73.8</td><td>66.7</td><td>34.5</td><td>61.7 62.4</td><td>50.8</td><td>52.3</td><td>18.1</td><td>51.6</td></tr><tr><td>Flow-I2P [10]</td><td>73.6</td><td>66.4</td><td>37.1</td><td>62.0</td><td>52.1</td><td>52.8</td><td>19.1</td><td></td></tr><tr><td></td><td>76.6</td><td>64.7</td><td></td><td></td><td>52.3</td><td>52.8</td><td>18.5</td><td>52.0</td></tr><tr><td>M23D(ours)</td><td>75.0</td><td>68.3</td><td>37.7</td><td>65.5</td><td>53.1</td><td>55.2</td><td>18.7</td><td>53.4</td></tr><tr><td colspan="9">Feature Matching Recall ↑</td></tr><tr><td>FCGF-2D3D [67] P2-Net [3] Predator-2D3D [68]</td><td>99.7 100.0</td><td>98.2 99.3</td><td>69.9 58.9</td><td>97.1 99.1</td><td>83.0 87.2</td><td>87.7 92.2</td><td>16.2 16.2</td><td>78.8 79</td></tr><tr><td>2D3D-MATR [1]</td><td>91.3 100.0</td><td>95.1 99.6</td><td>76.6 98.6</td><td>88.6 100.0</td><td>79.2 92.4</td><td>80.6 95.9</td><td>31.1 58.2</td><td>77.5 92.1</td></tr><tr><td>B2-3Dnet [2]</td><td>100.0</td><td>100.0</td><td>98.6</td><td>100.0</td><td>92.7</td><td>95.6</td><td>64.9</td><td>93.1</td></tr><tr><td>CA-I2P [62]</td><td>100.0</td><td>100.0</td><td>98.6</td><td>100.0</td><td>92.0</td><td>95.5</td><td>60.8</td><td>92.4</td></tr><tr><td>Flow-I2P [10]</td><td>100.0</td><td>99.7</td><td>95.1</td><td>99.9</td><td>93.1</td><td>96.8</td><td>56.7</td><td>91.6</td></tr><tr><td>M23D(ours)</td><td>100.0</td><td></td><td>100.0</td><td>100.0</td><td>93.8</td><td>96.0</td><td>59.5</td><td>92.8</td></tr><tr><td></td><td></td><td>100.0</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Registration Recall ↑</td></tr><tr><td>FCGF-2D3D [67] P2-Net [3]</td><td>89.5 96.9</td><td>79.7 86.5</td><td>19.2 20.5</td><td>85.9 91.7</td><td>69.4 75.3</td><td>79.0 85.2</td><td>6.8 4.1</td><td>61.4 65.7</td></tr><tr><td>Predator-2D3D [68]</td><td>69.6</td><td>60.7</td><td>17.8</td><td>62.9</td><td>56.2</td><td>62.6</td><td>9.5</td><td>48.5</td></tr><tr><td>2D3D-MATR [1]</td><td>96.9</td><td>90.7</td><td>52.1</td><td>95.5</td><td>80.9</td><td>86.1</td><td>28.4</td><td>75.8</td></tr><tr><td>B2-3Dnet [2]</td><td>98.3</td><td>90.5</td><td>56.2</td><td>96.4</td><td>84.0</td><td>86.1</td><td>32.4</td><td>77.7</td></tr><tr><td>CA-I2P [62]</td><td>99.0</td><td>90.7</td><td>68.5</td><td>96.2</td><td>83.0</td><td>88.1</td><td>31.1</td><td>79.5</td></tr><tr><td>Flow-I2P [10]</td><td>98.8</td><td>90.0</td><td>58.4</td><td>93.9</td><td>82.1</td><td>88.6</td><td>37.6</td><td>78.4</td></tr><tr><td></td><td>97.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>M23D(ours)</td><td></td><td>95.6</td><td>67.1</td><td>98.7</td><td>84.4</td><td>89.0</td><td>35.1</td><td>81.3</td></tr></table>

Inlier Ratio (IR) quantifies the proportion of inliers among all putative pixel-point correspondences. A correspondence is deemed an inlier if its 3D distance is less than a threshold $\tau _ { 1 } = 5$ cm under the ground-truth transformation $\mathbf { T } _ { \mathcal { P }  \mathcal { T } } ^ { * } \dot { : }$

$$
\mathrm { I R } = \frac { 1 } { | C | } \sum _ { ( x _ { i } , y _ { i } ) \in C } [  \mathbf { T } _ { \mathcal { P  T } } ^ { * } ( x _ { i } ) - \mathbf { K } ^ { - 1 } ( y _ { i } )  _ { 2 } < \tau _ { 1 } ] .\tag{13}
$$

Here, [·] denotes the Iverson bracket, $x _ { i } \in \mathcal { P }$ , and $y _ { i } \in \mathcal { Q } \subseteq$ I are pixel coordinates. The function ${ \bf K } ^ { - 1 }$ projects a pixel to a 3D point based on its depth value.

Feature Matching Recall (FMR) represents the fraction of image-point-cloud pairs with an IR above a threshold $\tau _ { 2 } = 0 . 1$ It measures the likelihood of successful registration:

$$
\mathrm { F M R } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left[ \mathrm { I R } _ { i } > \tau _ { 2 } \right] ,\tag{14}
$$

where M is the total number of image-point-cloud pairs.

Registration Recall (RR) measures the fraction of imagepoint-cloud pairs that are correctly registered. A pair is correctly registered if the root mean square error (RMSE) between the ground-truth-transformed and predicted point clouds $\mathbf { T } _ { \mathcal { P } \to \mathcal { T } }$ is less than $\tau _ { 3 } = 0 . 1$ m:

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { | \mathcal { P } | } \sum _ { p _ { i } \in \mathcal { P } } \| \mathbf { T } _ { \mathcal { P }  \mathbb { Z } } ( p _ { i } ) - \mathbf { T } _ { \mathcal { P }  \mathbb { Z } } ^ { * } ( p _ { i } ) \| _ { 2 } ^ { 2 } } ,\tag{15}
$$

TABLE III  
EVALUATION RESULTS ON KITTI DATASET. TEAL NUMBERS HIGHLIGHT THE BEST.
<table><tr><td rowspan=1 colspan=4>Method                  Type</td><td rowspan=1 colspan=1>RTE(m) ↓</td><td rowspan=1 colspan=1>RRE (°) ↓</td></tr><tr><td rowspan=8 colspan=3>vpc + GeoTransformer [17]vpc + HunterDeepI2P(2D) [42]CorrI2P [41]VP2P [56]2D3D-MATR [1]RetrI2P [53]FreeReg [55]</td><td rowspan=2 colspan=1>Point-to-PointPoint-to-Point</td><td rowspan=1 colspan=1> $4 . 2 7 \pm 7 . 1 4$ </td><td rowspan=1 colspan=1> $8 . 6 7 \pm 8 . 5 5$ </td></tr><tr><td rowspan=1 colspan=1> $4 . 5 9 \pm 5 . 2 2$ </td><td rowspan=2 colspan=1> $6 . 2 3 \pm 5 . 1 7$  $9 . 1 4 \pm 8 . 0 2$ </td></tr><tr><td rowspan=1 colspan=1>Image-to-Point</td><td rowspan=1 colspan=1> $5 . 1 5 \pm 7 . 3 5$ </td></tr><tr><td rowspan=1 colspan=1>Image-to-Point</td><td rowspan=1 colspan=1> $4 . 2 4 \pm 7 . 2 6$ </td><td rowspan=1 colspan=1> $6 . 4 7 \pm 5 . 2 0$ </td></tr><tr><td rowspan=3 colspan=1>Image-to-PointImage-to-PointImage-to-Point</td><td rowspan=1 colspan=1> $2 . 0 5 \pm 3 . 2 3$ </td><td rowspan=1 colspan=1> $4 . 0 1 \pm 6 . 3 7$ </td></tr><tr><td rowspan=1 colspan=1> $1 . 8 6 \pm 3 . 7 9$ </td><td rowspan=1 colspan=1> $2 . 5 9 \pm 4 . 4 6$ </td></tr><tr><td rowspan=1 colspan=1> $1 . 6 1 \pm 2 . 3 9$ </td><td rowspan=1 colspan=1> $3 . 1 6 \pm 2 . 8 5$ </td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Image-to-Point</td><td rowspan=1 colspan=1> $1 . 7 8 \pm 1 . 7 6$ </td><td rowspan=1 colspan=1> $2 . 8 9 \pm 4 . 4 7$ </td></tr><tr><td rowspan=2 colspan=3>CFI2P [54]M23D (Ours)</td><td rowspan=1 colspan=1>Image-to-Point</td><td rowspan=1 colspan=1> $1 . 9 5 \pm 2 . 9 7$ </td><td rowspan=1 colspan=1> $2 . 6 3 \pm 3 . 1 9$ </td></tr><tr><td rowspan=1 colspan=1>Image-to-Point</td><td rowspan=1 colspan=1> $1 . 5 8 \pm 2 . 3 3$ </td><td rowspan=1 colspan=1> $2 . 3 4 \pm 3 . 2 9$ </td></tr></table>

$$
\mathrm { R R } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left[ \mathrm { R M S E } _ { i } < \tau _ { 3 } \right] .\tag{16}
$$

These metrics provide a comprehensive evaluation of the model’s capability to accurately match features and register image-point-cloud pairs, ensuring robust alignment and effective correspondence.

## B. Evaluations on Dataset

We compare our approach with 2D3D-MATR [1] and other baseline methods [2], [3], [10], [55], [62], [67], [68] on the RGB-D Scenes v2 dataset, as shown in Table I. Our method integrates the ID-MAE framework to enhance feature extraction and better align the image and point cloud modalities. The introduction of the SRLM strategy further refines the model by adaptively focusing on semantically meaningful regions while suppressing noise from irrelevant areas. These improvements lead to a substantial increase in performance across all evaluation metrics. Specifically, our method achieves a 11.1 percentage point gain in inlier ratio (from 32.4 to 43.5), a 3.3 percentage point improvement in feature matching recall (from 90.8 to 94.1), and a 21.6 percentage point increase in registration recall (from 56.4 to 78.0) compared to the baseline method. Notably, our method sets a new state-of-theart in registration recall, with a remarkable improvement of 7.8 percentage points, demonstrating its robustness in accurately estimating correspondences across different modalities.

Compared to RGB-D Scenes v2, 7-Scenes exhibit greater scale variations, yet our method still outperforms others, as shown in Table II. In the Inlier Ratio metric, we achieve 53.4%, surpassing 2D3D-MATR by 3.3 percentage points (50.1%). For Feature Matching Recall, our method reaches

TABLE IV  
ABLATION STUDIES ON 7SCENES. TEAL NUMBERS HIGHLIGHT THE BEST.
<table><tr><td rowspan=1 colspan=5>Model | Image-MAE | Point-MAE | Cross Modal | IR FMRRR</td></tr><tr><td rowspan=1 colspan=5>BL            一           一             |50.192.175.8</td></tr><tr><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=3>一            49.891.2 78.0</td></tr><tr><td rowspan=1 colspan=1>F21</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>V   1</td><td rowspan=1 colspan=1>|52.292.478.3</td></tr><tr><td rowspan=1 colspan=1>F31</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√   1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>|50.492.080.4</td></tr><tr><td rowspan=1 colspan=1>Full1</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1>53.492.881.3</td></tr></table>

![](images/4b4a3192d4900eb8df24c7b1b8da9428b0074239c1ffa89677b0c26b66b968ea.jpg)  
Fig. 6. The visualization results of M23D. To rigorously analyze the performance, we set the error threshold to a strict 50px.

92.8%, demonstrating robustness in correspondence establishment, outperforming 2D3D-MATR by 0.7 percentage points (92.1%). In Registration Recall, we achieve 81.3%, maintaining an advantage of 5.5 percentage points over 2D3D-MATR’s 75.8%. Notably, 2D3D-MATR shows improvements in the challenging Heads and Pumpkin scenes, with the Heads scene involving small errors due to texture-less areas, and the Pumpkin scene containing repetitive patterns. Our ID-MAE effectively addresses these challenges, ensuring superior performance across all scenes. Although B2-3Dnet achieves slightly better performance in FMR due to its targeted modeling of image features, our method offers a greater overall advantage when considering the comprehensive results.

To better evaluate our performance, we conducted experiments on the outdoor dataset KITTI to verify the robustness of our method. As shown in Table III, we evaluate our method, M23D, on the KITTI dataset and compare it with several state-of-the-art techniques. Our evaluation is based on two key metrics: Reprojection Error (RTE) and Rotation Error (RRE), both of which are essential for assessing the accuracy of 3D point cloud registration. Our method outperforms all other approaches in both RTE and RRE. Specifically, M23D achieves a significantly lower RTE of $1 . 5 8 \pm 2 . 3 3$ meters and a lower RRE of $2 . 3 4 \pm \ : 3 . 2 9$ degrees. This surpasses other image-to-point methods, such as RetrI2P (RTE: $1 . 6 1 \pm 2 . 3 9$ meters, RRE: $3 . 1 6 ~ \pm ~ 2 . 8 5$ degrees) and VP2P (RTE: 2.05 $\pm \nobreakspace 3 . 2 3 \nobreakspace$ meters, RRE: $4 . 0 1 ~ \pm ~ 6 . 3 7$ degrees). These results demonstrate that our method provides more accurate alignment with lower error margins compared to previous works. The results on the KITTI dataset confirm the robustness and superiority of M23D over existing methods, highlighting its potential for real-world applications that require high-precision registration.

TABLE V  
ABLATION STUDIES OF DIFFERENT MASKING STRATEGIES ON RGB-D SCENES V2. TEAL NUMBERS HIGHLIGHT THE BEST.
<table><tr><td></td><td>Method Random Similarity RL SRLM</td><td>GFLOPs</td><td>IR</td><td>FMR</td><td>RR</td></tr><tr><td>M1</td><td></td><td>388.9</td><td>32.5</td><td>91.0</td><td>56.4</td></tr><tr><td>M2</td><td></td><td>423.8</td><td>42.9</td><td>93.2</td><td>63.5</td></tr><tr><td>M3</td><td></td><td>457.2</td><td>37.1</td><td>92.9</td><td>67.2</td></tr><tr><td>M4</td><td></td><td>834.8</td><td>42.7</td><td>93.4</td><td>73.1</td></tr><tr><td>M5</td><td></td><td>535.1</td><td>43.5</td><td>94.1</td><td>78.0</td></tr></table>

## C. Ablation Studies

We perform ablation studies on the 7Scenes dataset to evaluate the contributions of different components in our model, as shown in Table IV. Compared to the baseline (BL), adding Image-MAE (F1) or Point-MAE (F2) slightly improves the performance. The Dual MAE (F3) leads to further improvement. After introducing Cross-Modal alignment (Full), the model achieves the best results, demonstrating the effectiveness of our full approach.

![](images/686e538cbd3fb4fcd4cf445796a07e2ed0bef93755617ec03fe4bdd58310af1e.jpg)

![](images/23b5d1c1ff14d4fae62bf91eb80f0e9008db17872af2b8a05f575e7a17e0c138.jpg)  
Fig. 7. Ablation studies on the hyperparameters.

![](images/3ae991da490d6f44918583c8462d8ebe6b2196ec3d12b7b7ed885773be995768.jpg)

![](images/c734f290244b8e3e104a2a4a2522de51985fc22f45e7bc75988c81dd8b559882.jpg)

![](images/481154063223fa35d4bdf553b1ebebede36e6d1d9f3f1fed10370bf6741156ff.jpg)  
Fig. 8. The visualization of the point cloud projection onto the image demonstrates that our method achieves precise rigid transformation, maintaining alignment without significant discrepancies.

We conduct ablation studies to evaluate the impact of different masking strategies on the RGB-D Scenes v2 dataset, as shown in Table V. Compared to the baseline, random masking (M2) improves performance; however, due to its lack of targeted selection, the gains remain limited. Similarity-based masking (M3) further improves the results with only a slight increase in computational cost. Although directly applying reinforcement learning (M4) achieves strong performance, it incurs a significant increase in computation. In contrast, our SRLM strategy (M5) achieves the highest performance while maintaining good efficiency, demonstrating the superiority of our design in balancing accuracy and computational cost. Our strategy is specifically designed to adapt to the MAE framework. Therefore, the effectiveness of the reinforcement learning (RL) component serves as evidence of the strong capability of our MAE-based design. Previous common masking strategies often limit the full potential of the MAE module, whereas our approach enables more effective utilization of its representation learning capacity. The inter-modal MAE serves as the core representation learning mechanism, enforcing cross-modal consistency through reconstruction and establishing a strong feature foundation for 2D–3D correspondence estimation. The RL module is introduced solely to optimize the mask selection process, addressing the non-differentiability of discrete masking and improving training efficiency, rather than replacing or weakening the role of MAE. Importantly, MAEbased supervision alone already brings consistent performance gains compared to the baseline. The RL strategy further refines the masking process by prioritizing informative regions, leading to additional improvements on top of the MAE framework. While similarity evaluation is widely used in registration tasks, it is typically applied at the correspondence or matching stage. In contrast, our similarity-aware masking strategy leverages these cues during the representation learning phase. By guiding the MAE to focus on informative cross-modal regions before correspondence estimation, we shift the usage of similarity. Rather than directly establishing correspondences, similarity is used to prioritize which regions should be preserved or reconstructed during the masked modeling process. This allows the MAE to learn more correspondence-aware representations early in the process, setting the stage for more effective matching and correspondence estimation later.

Fig. 7 reports ablation studies on the key hyperparameters in our framework, where the registration recall (RR) is evaluated under different parameter settings. As shown in Fig. 7(a)– (c), the performance consistently improves when increasing the corresponding hyperparameters from small values, and reaches a relatively stable region around the default setting $( { \bf e . g . } , \gamma _ { 1 } = \gamma _ { 2 } = \gamma _ { 3 } = 1 )$ . Further increasing these values leads to marginal performance fluctuations or slight degradation, indicating that overly strong weighting may introduce redundancy or reduce robustness. Fig. 7(d) exhibits a similar trend, where moderate values yield the best performance, while extreme settings result in noticeable drops. Overall, these results demonstrate that the proposed method is not overly sensitive to hyperparameter choices and achieves stable performance within a reasonable range, validating the robustness of our design and the practicality of the selected default parameters.

Table VI compares the efficiency of different detection-free image-to-point cloud registration methods in terms of inference speed, computational cost, memory usage, and model size. Our method achieves an inference speed of 7.684 FPS, which is comparable to 2D3D-MATR (7.852 FPS) and faster than Flow-I2P (6.061 FPS), indicating that no additional inference latency is introduced. Although our approach requires slightly higher computational cost (535.1 GFLOPs) than 2D3D-MATR, this overhead mainly arises from similarity computation and does not affect inference efficiency, since the proposed intermodal MAE and RL-based masking strategy are applied only during training. Moreover, our method maintains a comparable memory footprint and parameter count (28.6M), demonstrating that the improved registration performance is achieved without significantly increasing model complexity. Compared to our method, Flow-I2P exhibits better performance in terms of GFLOPs and memory usage, but lags slightly in FPS. Overall, the results suggest that our approach strikes a favorable balance between accuracy and efficiency, making it suitable for practical deployment.

TABLE VI  
EFFICIENCY COMPARISON OF DIFFERENT METHODS.
<table><tr><td>Model</td><td>2D3D-MATR</td><td>Flow-I2P</td><td>Ours</td></tr><tr><td>FPS ↑</td><td>7.852</td><td>6.061</td><td>7.684</td></tr><tr><td>GFLOPs↓</td><td>388.2</td><td>525.1</td><td>535.1</td></tr><tr><td>Memory (GB) ↓</td><td>5.637</td><td>2.844</td><td>5.423</td></tr><tr><td>Param (M) ↓</td><td>28.2</td><td>28.7</td><td>28.6</td></tr></table>

![](images/87a295b30c25036c9bb038a4045470c3b076fa297cb9d45df1a6fb39c5ef6481.jpg)  
Fig. 9. Visualization of image and point cloud corresponding regions selected by reinforcement learning. The selected regions are highlighted in yellow in the image and in blue in the point cloud.

## D. Visualization

We conduct extensive experiments to validate the effectiveness of our model and visualize M23D’s matching performance for a more intuitive understanding. As shown in Fig. 6, we select seven scene pairs from the 7Scenes and RGB-D Scenes v2 datasets. In each visualization, the top, middle, and bottom rows correspond to M23D, M23D without SRLM, and the baseline model, respectively. We consider matches with a projection distance below 50 pixels as correct (green lines), and others as incorrect (red lines), using this threshold for visualization. The results show that applying ID-MAE improves matching, with more green lines indicating better correspondence. However, random masking still leads to some mismatches (red lines). With SRLM, matching performance improves significantly, demonstrating the effectiveness of our approach in enhancing both accuracy and robustness for image-to-point cloud registration. In Fig. 8, we project the point cloud onto the image using the obtained rigid transformation. It can be seen that the structure aligns well, with no largescale misalignments. Especially in the Heads scene, which lacks sufficient texture information, the registration results are still maintained effectively.

TABLE VII  
GENERALIZATION ABILITY OF SRLM ACROSS DIFFERENT DATASETS.
<table><tr><td>Training Dataset</td><td>Testing Dataset</td><td>Mask Strategy IR</td><td></td><td>RR</td></tr><tr><td>7-Scenes</td><td>RGB-D Scenes v2</td><td>Random Mask</td><td>18.1</td><td>23.9</td></tr><tr><td>7-Scenes</td><td>RGB-D Scenes v2</td><td>SRLM</td><td>18.7</td><td>24.3</td></tr><tr><td>7-Scenes</td><td>ScanNet</td><td>Random Mask</td><td>15.9</td><td>24.1</td></tr><tr><td>7-Scenes</td><td>ScanNet</td><td>SRLM</td><td>16.2</td><td>28.7</td></tr></table>

The visualization in Fig. 9 shows the image and point cloud regions selected by the proposed reinforcement learning-based masking strategy during training. The highlighted yellow regions in the image and blue regions in the point cloud indicate the areas considered informative for cross-modal matching. It can be observed that these selected regions tend to concentrate on areas with high texture and prominent structures. This is likely because these regions can provide high-quality features, and applying the Masked Auto-Encoder (MAE) to these regions can significantly improve their representation learning, thereby enhancing the overall registration accuracy. The selective masking strategy focuses the model’s attention on the most informative parts of the input data, allowing it to learn robust cross-modal correspondences from these salient regions. By emphasizing the feature-rich areas, the model can better capture the intricate geometric and visual cues needed for accurate image-to-point cloud registration, leading to the improved performance demonstrated by the method. This approach is a key innovation of the proposed technique, as it enables efficient and effective learning of the crossmodal mapping without requiring the model to process the entire input indiscriminately. The targeted focus on the most discriminative regions contributes to the overall efficiency and accuracy of the registration pipeline. It can be observed that the selected regions in the image and the point cloud do not always exhibit a strict one-to-one correspondence. This behavior is expected, as the masking strategy operates at the feature and region level, aiming to preserve discriminative semantic and geometric regions rather than enforcing precise pixel-topoint alignment. Moreover, differences in viewpoint, sensing modality, and partial visibility naturally lead to asymmetric region selection between the image and point cloud domains.

These visualizations also reveal several limitations of the proposed method. First, similar to most learning-based imageto-point cloud registration approaches, in low-overlap scenarios (e.g., the bottom-right example in Fig. 9), reliable cross-modal cues are inherently scarce, and the selected regions tend to concentrate on limited areas, which weakens the overall discriminative capability of the masking strategy. This phenomenon stems from the intrinsic ambiguity of correspondence supervision under minimal overlap and remains a common challenge in the field. Second, the proposed similarity-aware masking strategy relies on an initial crossmodal similarity map to guide informative region selection. While this design improves training stability and efficiency, its effectiveness may degrade when the initial similarity estimation is severely affected by noise or large viewpoint variations. We view this dependency as a reasonable trade-off, and consider improving the robustness of initial similarity estimation as a promising direction for future work. Finally, regarding scalability to large-scale scenes, the current implementation follows standard detection-free registration pipelines and is primarily evaluated on indoor and moderate-scale outdoor datasets. Although the proposed MAE and RL-based masking strategy are applied only during training and introduce no additional inference overhead, extending the framework to larger-scale environments may require hierarchical processing or more memory-efficient similarity computation strategies.

To further evaluate the cross-dataset generalization ability of SRLM, we train the model on 7-Scenes and directly evaluate it on RGB-D Scenes v2 and ScanNet (see in Table VII). Compared with the random masking strategy, SRLM consistently improves both the Inlier Ratio (IR) and Registration Recall (RR) across the two target datasets. Specifically, on RGB-D Scenes v2, SRLM increases IR from 18.1% to 18.7% and RR from 23.9% to 24.3%. On ScanNet, SRLM improves IR from 15.9% to 16.2%, while achieving a more substantial RR improvement from 24.1% to 28.7%, corresponding to a gain of 4.6 percentage points. These consistent improvements indicate that SRLM does not merely overfit the scene distribution of the training dataset. Instead, similarity-based initialization helps suppress irrelevant and non-overlapping regions, while RLdriven refinement adaptively identifies registration-informative regions according to the task-aligned matching reward. Notably, the larger improvement in RR than in IR on ScanNet suggests that SRLM enhances not only the proportion of correct correspondences, but also their geometric reliability for pose estimation. Overall, these results demonstrate that SRLM learns a more transferable masking policy and improves registration robustness under considerable cross-dataset domain shifts.

Overall, these analyses demonstrate both the effectiveness and the current limitations of the proposed approach in crossmodal region selection and correspondence modeling. Future work will focus on more robust cross-modal cue modeling under low-overlap conditions, enhancing the robustness of initial similarity estimation, and developing hierarchical and efficient registration frameworks for large-scale scenes.

## V. CONCLUSION

We propose M23D, an Adaptive Dual-Masked Autoencoder Network for Image-to-Point Cloud Registration. Our method introduces a dual MAE to enhance feature extraction by aligning features and establishing correspondences. Additionally, we propose a similarity-based RL masking strategy that reduces unnecessary masking and focuses on key matching regions, improving registration accuracy and robustness. M23D achieves state-of-the-art performance on the RGB-D Scenes v2 and 7-Scenes datasets.

## ACKNOWLEDGMENT

This work was supported in part by the Fundamental Research Funds for the Central Universities (11040- 03742026007).

[1] M. Li, Z. Qin, Z. Gao, R. Yi, C. Zhu, Y. Guo, and K. Xu, “2d3dmatr: 2d-3d matching transformer for detection-free registration between images and point clouds,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14128–14138, 2023.

[2] Z. Cheng, J. Deng, X. Li, B. Yin, and T. Zhang, “Bridge 2D-3D: Uncertainty-aware Hierarchical Registration Network with Domain Alignment,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 3, pp. 2491–2499, 2025.

[3] B. Wang, C. Chen, Z. Cui, J. Qin, C. X. Lu, Z. Yu, P. Zhao, Z. Dong, F. Zhu, N. Trigoni, et al., “P2-net: Joint description and detection of local features for pixel and point matching,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16004–16013, 2021.

[4] V. Lepetit, F. Moreno-Noguer, and P. Fua, “EPnP: An Accurate O(n) Solution to the PnP Problem,” International Journal of Computer Vision, vol. 81, pp. 155–166, 2009.

[5] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981.

[6] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, “LoFTR: Detectorfree local feature matching with transformers,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8922–8931, 2021.

[7] Y. Wang, X. He, S. Peng, D. Tan, and X. Zhou, “Efficient LoFTR: Semidense local feature matching with sparse-like speed,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21666–21675, 2024.

[8] N. Inoue and K. Shinoda, “A fast and accurate video semantic-indexing system using fast MAP adaptation and GMM supervectors,” IEEE Transactions on Multimedia, vol. 14, no. 4, pp. 1196–1205, 2012.

[9] M. J. Reale, S. Canavan, L. Yin, et al., “A multi-gesture interaction system using a 3-D iris disk model for gaze estimation and an active appearance model for 3-D hand pointing,” IEEE Transactions on Multimedia, vol. 13, no. 3, pp. 474–486, 2011.

[10] P. An, Y. Yang, J. Yang, M. Peng, Q. Liu, and L. Nan, “Enhance Imageto-Point-Cloud Registration with Beltrami Flow,” International Journal of Computer Vision, vol. 133, no. 12, pp. 8589–8616, 2025.

[11] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, “Superglue: Learning feature matching with graph neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4938–4947, 2020.

[12] B. Drost, M. Ulrich, N. Navab, and S. Ilic, “Model Globally, Match Locally: Efficient and Robust 3D Object Recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 998–1005, 2010.

[13] Y. Liu, V. Kılıc¸, J. Guan, et al., “Audio–visual particle flow smc-phd filtering for multi-speaker tracking,” IEEE Transactions on Multimedia, vol. 22, no. 4, pp. 934–948, 2019.

[14] R. B. Rusu, N. Blodow, and M. Beetz, “Fast point feature histograms (FPFH) for 3D registration,” in 2009 IEEE International Conference on Robotics and Automation, pp. 3212–3217, 2009.

[15] X. Ding, Z. Chen, W. Lin, et al., “Towards 3D colored mesh saliency: Database and benchmarks,” IEEE Transactions on Multimedia, vol. 26, pp. 3580–3591, 2024.

[16] H. Yu, F. Li, M. Saleh, B. Busam, and S. Ilic, “Cofinet: Reliable coarseto-fine correspondences for robust pointcloud registration,” Advances in Neural Information Processing Systems, vol. 34, pp. 23872–23884, 2021.

[17] Z. Qin, H. Yu, C. Wang, Y. Guo, Y. Peng, S. Ilic, D. Hu, and K. Xu, “Geotransformer: Fast and robust point cloud registration with geometric transformer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 8, pp. 9806–9821, 2023.

[18] E. D. Sontag, “Comments on integral variants of ISS,” Systems & Control Letters, vol. 34, no. 1–2, pp. 93–100, 1998.

[19] M. Feng, S. Hu, M. H. Ang, and G. H. Lee, “2d3d-matchnet: Learning to match keypoints across 2D image and 3D point cloud,” in 2019 International Conference on Robotics and Automation (ICRA), pp. 4790– 4796, 2019.

[20] K. Lai, L. Bo, and D. Fox, “Unsupervised feature learning for 3D scene labeling,” in 2014 IEEE International Conference on Robotics and Automation (ICRA), pp. 3050–3057, 2014.

[21] B. Glocker, S. Izadi, J. Shotton, and A. Criminisi, “Real-time RGB-D camera relocalization,” in 2013 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pp. 173–179, 2013.

[22] S. Chen, Y. Chen, X. Yin, X. Liu, H. Lai, and T. Zhang, “PAF: Prototype Adaptive Fusion for Test-Time Adaptation of Vision-Language Models,” Proceedings of the 33rd ACM International Conference on Multimedia, pp. 3007–3016, 2025.

[23] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention Is All You Need,” Advances in Neural Information Processing Systems, vol. 30, pp. 5998–6008, 2017.

[24] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

[25] J. Deng, J. Lu, Z. Cheng, and W. Yang, “DiffCorr: Conditional Diffusion Model with Reliable Pseudo-Label Guidance for Unsupervised Point Cloud Shape Correspondence,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 3, pp. 2690–2698, 2025.

[26] X. Li, W. Yang, J. Deng, Z. Cheng, X. Zhou, and T. Zhang, “Implicit Correspondence Learning for Image-to-Point Cloud Registration,” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16922–16931, June 2025.

[27] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2117– 2125, 2017.

[28] H. Thomas, C. R. Qi, J.-E. Deschaud, B. Marcotegui, F. Goulette, and L. J. Guibas, “Kpconv: Flexible and deformable convolution for point clouds,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6411–6420, 2019.

[29] Z. Cheng, X. Yin, J. Deng, B. Liao, Y. Chen, X. Zhou, B. Yin, and T. Zhang, “Adaptive Agent Selection and Interaction Network for Imageto-Point Cloud Registration,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 5, pp. 3335–3343, 2026.

[30] J. Li, B. M. Chen, and G. H. Lee, “So-net: Self-organizing network for point cloud analysis,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9397–9406, 2018.

[31] Z. Qin, H. Yu, C. Wang, Y. Guo, Y. Peng, and K. Xu, “Geometric transformer for fast and robust point cloud registration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11143–11152, 2022.

[32] H. Durrant-Whyte and T. Bailey, “Simultaneous localization and mapping: part I,” IEEE Robotics & Automation Magazine, vol. 13, no. 2, pp. 99–110, 2006.

[33] Y. Jia, F. Cao, T. Wang, et al., “CAD-Mesher: A Convenient, Accurate, Dense Mesh-Based Mapping Module in SLAM for Dynamic Environments,” IEEE Transactions on Multimedia, vol. 28, pp. 1025–1036, 2026.

[34] H. Luo, Y. Gao, Y. Wu, et al., “Real-Time Dense Monocular SLAM With Online Adapted Depth Prediction Network,” IEEE Transactions on Multimedia, vol. 21, no. 2, pp. 470–483, 2019.

[35] A. Kendall, M. Grimes, and R. Cipolla, “PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,” in Proceedings of the IEEE International Conference on Computer Vision, pp. 2938–2946, 2015.

[36] B. Fan, Y. Yang, W. Feng, et al., “Seeing Through Darkness: Visual Localization at Night via Weakly Supervised Learning of Domain Invariant Features,” IEEE Transactions on Multimedia, vol. 25, pp. 1713–1726, 2023.

[37] Y. Ganin and V. Lempitsky, “Unsupervised domain adaptation by backpropagation,” in International Conference on Machine Learning, pp. 1180–1189, 2015.

[38] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3D classification and segmentation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 652–660, 2017.

[39] Y. Li, Q. Hao, J. Hu, et al., “3D3M: 3D Modulated Morphable Model for Monocular Face Reconstruction,” IEEE Transactions on Multimedia, vol. 25, pp. 6642–6652, 2023.

[40] F. Chu, Y. Cong, Y. Wang, et al., “DetailRecon: Focusing on Detailed Regions for Online Monocular 3D Reconstruction,” IEEE Transactions on Multimedia, vol. 27, pp. 3266–3278, 2025.

[41] S. Ren, Y. Zeng, J. Hou, and X. Chen, “CorrI2P: Deep Image-to-Point Cloud Registration via Dense Correspondence,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 3, pp. 1198– 1208, 2023.

[42] J. Li and G. H. Lee, “DeepI2P: Image-to-point cloud registration via deep classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15960–15969, 2021.

[43] S. Kang, Y. Liao, J. Li, F. Liang, Y. Li, X. Zou, F. Li, X. Chen, Z. Dong, and B. Yang, “CoFiI2P: Coarse-to-Fine Correspondences-Based Image to

Point Cloud Registration,” IEEE Robotics and Automation Letters, vol. 9, no. 11, pp. 10264–10271, 2024.

[44] Z. Qin, H. Yu, C. Wang, Y. Guo, Y. Peng, and K. Xu, “Geometric transformer for fast and robust point cloud registration,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11143–11152, 2022.

[45] Y. Sun, C. Cheng, Y. Zhang, C. Zhang, L. Zheng, Z. Wang, and Y. Wei, “Circle loss: A unified perspective of pair similarity optimization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6398–6407, 2020.

[46] Y. Pang, W. Wang, F. E. H. Tay, W. Liu, Y. Tian, and L. Yuan, “Masked autoencoders for point cloud self-supervised learning,” in European Conference on Computer Vision, pp. 604–621, 2022.

[47] R. Zhang, Z. Guo, P. Gao, R. Fang, B. Zhao, D. Wang, Y. Qiao, and H. Li, “Point-m2ae: Multi-scale masked autoencoders for hierarchical point cloud pre-training,” Advances in Neural Information Processing Systems, vol. 35, pp. 27061–27074, 2022.

[48] D. Li, H. Zhao, X. Yan, L. Zhao, and H. Cao, “Long-range attention classification for substation point cloud,” Neurocomputing, vol. 608, p. 128435, 2024.

[49] R. Zhang, L. Wang, Y. Qiao, P. Gao, and H. Li, “Learning 3D representations from 2D pre-trained models via image-to-point masked autoencoders,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21769–21780, 2023.

[50] A. Chen, K. Zhang, R. Zhang, Z. Wang, Y. Lu, Y. Guo, and S. Zhang, “Pimae: Point cloud and image interactive masked autoencoders for 3D object detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5291–5301, 2023.

[51] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked au-´ toencoders are scalable vision learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16000– 16009, 2022.

[52] G. Li, H. Zheng, D. Liu, C. Wang, B. Su, and C. Zheng, “Semmae: Semantic-guided masking for learning masked autoencoders,” Advances in Neural Information Processing Systems, vol. 35, pp. 14290–14302, 2022.

[53] L. Bie, S. Pan, S. Li, et al., “GraphI2P: Image-to-Point Cloud Registration with Exploring Pattern of Correspondence via Graph Learning,” in Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22161–22171, 2025.

[54] G. Yao, Y. Xuan, Y. Chen, and Y. Pan, “Quantity-Aware Coarse-to-Fine Correspondence for Image-to-Point Cloud Registration,” IEEE Sensors Journal, vol. 24, no. 20, pp. 33826–33837, 2024.

[55] H. Wang, Y. Liu, B. Wang, Y. Sun, Z. Dong, W. Wang, and B. Yang, “FreeReg: Image-to-Point Cloud Registration Leveraging Pretrained Diffusion Models and Monocular Depth Estimators,” in Proceedings of the International Conference on Learning Representations (ICLR), 2024.

[56] Y. Yue, H. Yuan, Q. Miao, X. Mao, R. Hamzaoui, and P. Eisert, “EdgeRegNet: Edge Feature-Based Multimodal Registration Network Between Images and LiDAR Point Clouds,” IEEE Transactions on Multimedia, vol. 27, pp. 9281–9292, 2025.

[57] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: Representing scenes as neural radiance fields for view synthesis,” Communications ofthe ACM, vol. 65, no. 1, pp. 99–106, 2021.

[58] R. Rubinstein, “The cross-entropy method for combinatorial and continuous optimization,” Methodology and Computing in Applied Probability, vol. 1, pp. 127–190, 1999.

[59] S. Ghanavati, D. Amyot, and A. Rifaut, “Legal goal-oriented requirement language (legal GRL) for modeling regulations,” in Proceedings of the 6th International Workshop on Modeling in Software Engineering, pp. 1–6, 2014.

[60] P. An, Y. Yang, J. Yang, M. Peng, Q. Liu, and L. Nan, “Enhance Imageto-Point-Cloud Registration with Beltrami Flow,” International Journal of Computer Vision, vol. 133, no. 12, pp. 8589–8616, 2025.

[61] Y. Chen, R. Sun, W. Li, et al., “Beyondmix: Leveraging structural priors and long-range dependencies for domain-invariant LiDAR segmentation,” in Advances in Neural Information Processing Systems, vol. 38, pp. 89462–89488, 2026.

[62] Z. Cheng, J. Deng, X. Li, X. Yin, B. Liao, B. Yin, W. Yang, and T. Zhang, “CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 27739–27749, 2025.

[63] L. van der Maaten and G. Hinton, “Visualizing Data Using t-SNE,” Journal of Machine Learning Research, vol. 9, pp. 2579–2605, 2008.

[64] X. Zhang, J. Zhou, W. Sun, and S. K. Jha, “A Lightweight CNN Based on Transfer Learning for COVID-19 Diagnosis,” Computers, Materials & Continua, vol. 72, no. 1, pp. 1123–1137, 2022.

[65] S. Huang, Z. Gojcic, M. Usvyatsov, A. Wieser, and K. Schindler, “Predator: Registration of 3D point clouds with low overlap,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4267–4276, 2021.

[66] A. Strehl, J. Ghosh, and R. Mooney, “Impact of similarity measures on web-page clustering,” in Workshop on Artificial Intelligence for Web Search (AAAI 2000), vol. 58, pp. 64, 2000.

[67] C. Choy, J. Park, and V. Koltun, “Fully convolutional geometric features,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 8958–8966, 2019.

[68] S. Huang, Z. Gojcic, M. Usvyatsov, A. Wieser, and K. Schindler, “Predator: Registration of 3D point clouds with low overlap,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4267–4276, 2021.

[69] Z. Wang, J. Wang, D. Zuo, Y. Ji, X. Xia, Y. Ma, J. Hao, M. Yuan, Y. Zhang, and F. Wu, “A Hierarchical Adaptive Multi-Task Reinforcement Learning Framework for Multiplier Circuit Design,” in Proceedings of the 41st International Conference on Machine Learning (ICML), vol. 235, pp. 51825–51853, 2024.

[70] Z. Wang, J. Wang, Q. Zhou, B. Li, and H. Li, “Sample-Efficient Reinforcement Learning via Conservative Model-Based Actor-Critic,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 8, pp. 8612–8620, June 2022.

[71] Z. Wang, T. Pan, Q. Zhou, and J. Wang, “Efficient Exploration in Resource-Restricted Reinforcement Learning,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 8, pp. 10279–10287, June 2023.

[72] Z. Qin, H. Yu, C. Wang, Y. Guo, Y. Peng, and K. Xu, “Geometric transformer for fast and robust point cloud registration,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11143–11152, 2022.

[73] J. Deng, C. Wang, J. Lu, et al., “SE-ORNet: Self-ensembling orientationaware network for unsupervised point cloud shape correspondence,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5364–5373, 2023.

[74] J. Deng, J. Lu, and T. Zhang, “Quantity-Quality Enhanced Self-Training Network for Weakly Supervised Point Cloud Semantic Segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 5, pp. 3580–3596, 2025.

[75] Y. Wu, J. Liu, M. Gong, P. Gong, X. Fan, A. K. Qin, Q. Miao, and W. Ma, “Self-Supervised Intra-Modal and Cross-Modal Contrastive Learning for Point Cloud Understanding,” IEEE Transactions on Multimedia, vol. 26, pp. 1626–1638, 2024.

[76] J. Liu, Y. Wu, M. Gong, Z. Liu, Q. Miao, and W. Ma, “Inter-Modal Masked Autoencoder for Self-Supervised Learning on Point Clouds,” IEEE Transactions on Multimedia, vol. 26, pp. 3897–3908, 2023.

[77] Z. Cheng, J. Deng, X. Li, L. Liu, X. Yin, B. Yin, and T. Zhang, “B2-3D++: Uncertainty-aware Hierarchical Registration Network with Domain Alignment,” Authorea Preprints, 2025.

[78] B. Liao, W. Zhai, Z. Wan, Z. Cheng, W. Yang, Y. Cao, T. Zhang, and Z.- J. Zha, “EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting,” Advances in Neural Information Processing Systems, vol. 38, 2025.

[79] Z. Guo, R. Zhang, L. Qiu, X. Li, and P.-A. Heng, “Joint-MAE: 2D– 3D Joint Masked Autoencoders for 3D Point Cloud Pre-training,” in Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (IJCAI), pp. 791–799, 2023.

[80] J. Liu, L. Kong, Y. Wu, M. Gong, H. Li, Q. Miao, W. Ma, and C. Qin, “Triple Point Masking,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 9, pp. 8466–8477, 2025.

Zhixin Cheng received his bachelor’s degree from the School of Electrical and Electronic Engineering, Huazhong University of Science and Technology, Wuhan, China, in 2020, and his Ph.D. degree from the School of Information Science and Technology, University of Science and Technology of China, Hefei, China, through a combined M.S.–Ph.D. program. He is currently a Specially Appointed Associate Professor with Hefei University of Technology, Hefei, China. His research interests include computer vision and machine learning, with a focus on

![](images/4c3a38621c447aabc1adb707be6479b4a43fb718de0e7f6442962bbbed183a68.jpg)  
3D scene registration, multimodal learning, and image enhancement.

![](images/76d96e445569bf79a0766dcd89abed1f78e0af67e353f0008a459eb5dc37229d.jpg)

Jiacheng Deng received the bachelor’s degree in Information Security from the University of Science and Technology of China in 2023. He is now pursuing a master degree in Control Science and Engineering at University of Science and Technology of China. His research interests include computer vision and deep learning, especially image-to-point cloud registration and pose estimation.

![](images/f1621fc37236825608ece9516883765332001703258087074a7a0a1244f72067.jpg)

Xiaotian Yin is currently pursuing a Ph.D. degree at the University of Science and Technology of China, Hefei, China. His research interests include computer vision and machine learning, with a focus on few-shot learning and multi-modal learning.

![](images/68a5daa26ec45eee00a08bc0e20ce48952e96d55b884653d0128c84eaa15def5.jpg)

Baoqun Yin received his bachelor’s degree in Mathematics from Sichuan University, Chengdu, China, in 1985, and his master’s degree in Applied Mathematics from the University of Science and Technology of China (USTC), Hefei, China, in 1993. He earned his Ph.D. degree in Pattern Recognition and Intelligent Systems from the Department of Automation, USTC, in 1998. He is currently a Professor in the Department of Automation at the University of Science and Technology of China. His research interests include stochastic systems, system optimization, and information networks, focusing on Markov decision processes, network optimization, and smart energy management.

![](images/01e9671494f2f20081219d733f9471c3f6c3716a3b25aea1f424605e5fa1d0a4.jpg)

Richang Hong (Senior Member, IEEE) received the Ph.D. degree from the University of Science and Technology of China, Hefei, China, in 2008. From September 2008 to December 2010, he was a Research Fellow with the School of Computing, National University of Singapore, Singapore. He is currently a Professor with the Hefei University of Technology, Hefei, China. He has coauthored more than 300 publications in the areas of his research interests, which include multimedia question answering, video content analysis, and pattern recognition.

Dr. Hong is a Member of the Association for Computing Machinery. He was the recipient of the Best Paper Award in the ACM Multimedia 2010.

![](images/d55a0f6153c103c79149eaccb3503e63080623e7b34e2f7a36943ef82b49bb75.jpg)

Tianzhu Zhang (M’11) received the bachelor’s degree in Communications and Information Technology from the Beijing Institute of Technology, Beijing, China, in 2006, and the Ph.D. degree in Pattern Recognition and Intelligent Systems from the Institute of Automation, Chinese Academy of Sciences, Beijing, China, in 2011. He is currently a Professor at the Department of Automation, School of Information Science, University of Science and Technology of China. His current research interests include computer vision and multimedia, with a focus on action recognition, object classification, object tracking, and social event analysis.