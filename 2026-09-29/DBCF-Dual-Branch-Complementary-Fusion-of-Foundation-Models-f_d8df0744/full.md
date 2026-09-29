# DBCF: Dual-Branch Complementary Fusion of Foundation Models for Generalized Deepfake Detection

Fengming Gu<sup>a,b</sup>, Mingjie He<sup>b,c</sup>, Zonghui Guo<sup>d</sup>, Jie Zhang<sup>b,c,∗</sup>, Shiguang Shan<sup>b,c</sup>

<sup>a</sup>School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences, Beijing, 100049, China

<sup>b</sup>State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences, Beijing, 100190, China

<sup>c</sup>University ofChinese Academy ofSciences, Beijing, 100049, China

<sup>d</sup>Faculty of Information Science and Engineering, Ocean University of China, Qingdao, 266404, China

## Abstract

As image generation and editing technologies have progressed substantially, facial forgeries pose significant challenges to privacy and public safety. Due to limited ability to capture forgery cues, existing small-scale forgery detection models often struggle to generalize across various domains and unseen manipulations. To address this limitation, researchers have turned to large-scale foundation models, which can provide richer representations and better generalization. Nevertheless, relying on a single foundation model alone remains insuficient for efective forgery detection. While models like CLIP ofer robust global semantic cues, they lack the capacity to capture detailed local facial features. In contrast, DINO excels at capturing local structural features of faces, but provides weaker global semantic context. To fully utilize the synergies among multiple foundation models, we propose a hierarchical multi-granular framework that integrates complementary pretrained representations. Specifically, a Global Context Branch (GCB) based on CLIP captures holistic semantic cues, while a Finegrained Cue Branch (FCB) built on DINOv3 captures localized structural irregularities. In addition, we design a feature fusion module that enables parameter-eficient adaptation of the frozen foundation backbones by adaptively extracting and integrating complementary features from the two models. By jointly leveraging global context

and fine-grained cues, our method learns more comprehensive forgery representations and achieves strong cross-manipulation performance. Extensive experiments on multiple benchmarks demonstrate the benefit of the proposed design, particularly under cross-dataset and cross-manipulation settings.

Keywords: Deepfake Detection, Pretrained Models, Domain Generalization

## 1. Introduction

The rapid advancement of artificial image generation technologies, particularly face manipulation and synthesis methods based on generative adversarial networks (GANs) and difusion models, has profoundly reshaped the landscape of digital media creation. While these techniques have enabled numerous beneficial applications, they have also drastically lowered the barrier for producing highly realistic facial forgeries. Such forgeries pose severe threats to personal privacy, social trust, and even public security, as they can be exploited for identity fraud, political misinformation, and malicious impersonation. Consequently, facial deepfake detection has emerged as an essential research task aimed at safeguarding the integrity and authenticity of visual content.

However, building robust and generalizable deepfake detectors remains highly challenging. The core dificulty lies in the diversity and rapid evolution of forgery techniques, which continuously introduce novel manipulation patterns. In response to such challenges, several prior methods[1, 2, 3] generate diverse manipulated samples that approximate potential unseen forgery patterns. Besides, some other methods[4, 5] employ disentangled representation learning which can suppress irrelevant factors such as identity, illumination, or background that may act as shortcuts rather than genuine forgery cues. To some extent, these methods can prevent the models from learning only limited, dataset-specific cues from known manipulations. However, their performance is still largely constrained by the scale of the training data and the distribution of the manipulation samples.

Recently, a promising complementary direction has emerged by leveraging largescale pretrained vision models for forgery detection. These foundation models encode rich visual priors from massive and diverse datasets. Among them, CLIP has attracted considerable attention because its contrastive image-text pretraining provides robust global semantic representations. Recent studies [6] demonstrate that using CLIP as the backbone for deepfake detection can enhance generalization across unseen manipulations by exploiting holistic global context.

Despite their strong generalization ability, directly applying pretrained vision models such as CLIP to facial forgery detection remains nontrivial. Owing to their representation characteristics, CLIP-based detectors often exhibit limited sensitivity to subtle, low-level forgery cues. This limitation primarily stems from the contrastive pretraining objective, which emphasizes global semantic alignment over localized visual inconsistencies. As noted in prior work [6], this strong global context bias, while beneficial for generalization, can be inefective for manipulations that introduce only minor local artifacts (e.g., NeuralTexture [7]) without altering overall semantics, resulting in suboptimal detection of fine-grained local cues.

To overcome this limitation, we seek to integrate complementary pretrained representations in a principled manner, thereby balancing holistic semantic understanding with sensitivity to subtle manipulation cues. We find that the self-supervised backbone DINOv3 [8] is particularly well suited for modeling local structural patterns and finegrained visual details. It has potential to complement CLIP, which primarily encodes global semantic context.

To this end, we propose a dual-branch multi-granular feature fusion framework designed to integrate two pretrained vision models, CLIP and DINOv3, while preserving their pretrained representations and enabling complementary feature fusion. Crucially, these two models are pretrained under fundamentally diferent learning paradigms, and their feature spaces are therefore not inherently aligned. In the absence of explicit alignment or regulation, naively fusing these representations or jointly finetuning them can erode the structured feature patterns induced by pretraining and compromise the efective use of pretrained priors. Specifically, DBCF keeps the pretrained backbones frozen and learns complementary interactions at the representation level, which helps preserve pretrained priors while reducing the risk of overfitting to dataset-specific artifacts. This design allows the model to exploit both global semantic consistency and fine-grained manipulation traces without directly collapsing the heterogeneous feature

spaces of CLIP and DINOv3.

Extensive experiments conducted on several widely used deepfake detection benchmarks demonstrate the efectiveness of our approach. In particular, it achieves strong performance in both cross-dataset and cross-manipulation evaluations when compared with prior approaches that use only frozen or fully fine-tuned pretrained backbones These results support our hypothesis that combining complementary pretrained backbones provides a principled and robust pathway toward generalizable deepfake detection.

In summary, the main contributions of this work are threefold:

1. We introduce a dual-branch, multi-granular framework that leverages the complementary strengths of CLIP and DINOv3. It incorporates a spatially aligned fusion mechanism to integrate global and local representations while preserving pretrained priors.

2. We provide an in-depth analysis of the inherent limitations in directly adapting pretrained visual models for deepfake detection, revealing a fundamental tension between preserving robust global semantic representations and enhancing sensitivity to fine-grained manipulation cues.

3. We conduct extensive evaluations on multiple datasets and forgery types, demonstrating that our approach achieves superior robustness and generalization performance compared with existing state-of-the-art methods.

## 2. Related Works

Face Forgery Generation. Deepfake refers to AI-generated forgeries produced by deep generative models, capable of synthesizing, modifying, or replacing humancentric visual or auditory content. Existing deepfake generation methods can be broadly categorized into three main types: face swapping, face reenactment, and entire face synthesis. Among them, face swapping [9] techniques are widely used to transfer the source identity onto a target individual’s appearance, motion, and scene context [10], producing videos where the target person convincingly appears as the source. In contrast, face reenactment focuses on transferring facial motion while preserving the target identity, allowing the target face to mimic expression, pose, or lip movements driven by various modalities such as images [11] and audio [12]. These approaches typically disentangle and manipulate identity-related and motion-related representations. Beyond manipulation-based pipelines, entire face synthesis methods generate photorealistic facial images from learned generative distributions. Representative GAN-based models, such as the StyleGAN family [13, 14], enable high-fidelity and controllable face generation, while transformer-assisted models like VQGAN [15] synthesize highresolution images via discrete tokens. Difusion-based methods, benefiting from advanced difusion architectures, further provide more realistic and controllable synthesis, as exemplified by Stable Difusion [16]. For clarity, entire face synthesis aligns more closely with generic image synthesis detection and thus falls outside the scope of the manipulation-based face forgery task addressed in this work.

Face Forgery Detection. Due to the rapid emergence of numerous forgery techniques, a large body of research has focused on improving the generalization capability of deepfake detectors against unseen forgery methods. Methods based on data augmentation enhance generalization by synthesizing diverse and transferable forgery patterns. Face X-ray [1] and SBI [2] augment samples at the image level, while LSDA [3] enriches forgery variations in the feature space. These techniques have been shown to be efective by exposing detectors to a broader distribution of manipulation artifacts. Another line of work improves domain generalization through feature disentanglement and real-face modeling. Disentanglement [4] eliminates the influence of irrelevant features, while real-face representation learning [17] enhances cross-manipulation robustness. Additionally, methods like facial landmarks [18] and mask-guided supervision [19] guide the model to focus on relevant features, while multi-definition crossdomain training [20] enhances robustness to low-quality or previously unseen deepfakes.

Complementary feature modeling has also been investigated in recent hybrid frameworks. For example, LGDF-Net [21] introduces local and global branches to capture localized artifacts and global facial texture/context through multi-scale and multi-level fusion, while Ding et al. [22] jointly exploit RGB and noise-map representations across multiple scales. These methods demonstrate the efectiveness of incorporating complementary cues through task-oriented architectural designs. Beyond architectural specialization for complementary cue extraction, the integration of complementary transferable representations from pretrained models represents another promising direction for generalized deepfake detection.

More recently, researchers have begun to explore the potential of large pretrained models for generalizable deepfake detection. For instance, CLIPping [23] and UniFD [24] demonstrate that it is feasible to equip CLIP with universal deepfake detection capabilities through tailored fine-tuning strategies. In addition, methods such as Ffaa [25] and X<sup>2</sup>-DFD [26] further investigate the efectiveness of vision–language models (VLMs) in this task. Overall, these studies demonstrate the strong transferability of large-scale pretrained representations for generalized deepfake detection, providing a foundation for exploring more efective ways to exploit pretrained visual priors in this task.

Vision Foundation models. Vision Foundation models (VFMs) have become a cornerstone of modern computer vision. After training on large-scale and diverse datasets, VFMs have acquired rich and transferable visual knowledge, yielding strong performance across a wide range of downstream tasks. Benefiting from the global contextual attention mechanism, Vision Transformer (ViT) [27] is adopted as the base architecture of most VFMs. The success of ViT-based VFMs can be attributed to two large-scale pretraining paradigms, i.e. Cross-Modal Supervised Learning and Visual Self-Supervised Learning. These two paradigms focus on distinct types of feature extraction. The first paradigm utilizes cross-modal (vision-text) contrastive learning for supervision. This approach, exemplified by the CLIP [28] family efectively extends the definition of supervised learning beyond hard classification labels. This cross-modal alignment grants these models strong global semantic consistency and impressive zero-shot transferability. Another critical paradigm focuses on Visual Self-Supervised Learning (SSL), where models generate their own supervisory signals exclusively from image data through pretext tasks. These methods can be broadly categorized into: those based on instance discrimination via contrastive techniques (e.g., MoCo [29]), those utilizing masked image modeling (e.g., MAE [30]), and those using non-contrastive clustering or distillation techniques (e.g., DINO [31]). These VFMs are trained to understand the image’s intrinsic structure without relying on external labels. Thanks to the novel Gram-Anchoring regularization, the latest DINOv3 [8] exhibits exceptional fidelity in modeling local structural patterns and fine-grained visual cues. This strong emphasis on internal structural consistency enables SSL-based visual models to adapt efectively to tasks such as object detection and semantic segmentation. While cross-modal supervised and self-supervised VFMs excel in diferent aspects of visual understanding, their complementary characteristics motivate a dual-branch approach that integrates global semantics and fine-grained local cues for improved generalization in forgery detection.

![](images/8d2dd7fbf82817c73fd6b9c1f1608dda882698fcd3ef6ef931975fc8d2520319.jpg)  
Figure 1: Overview of the proposed framework. A Global Context Branch (GCB) and a Fine-grained Cue Branch (FCB) operate in parallel and are connected via adaptive cross-feature interaction modules, enabling efective collaboration between global context and fine-grained forgery cues.

## 3. Method

## 3.1. Overview

As illustrated in Fig.1, we propose a dual-branch framework for deepfake detection that explicitly models the complementarity between global context and fine-grained forgery cues. The framework consists of a Global Context Branch (GCB), a Finegrained Cue Branch (FCB), and a set of adaptive cross-feature interaction modules that enable progressive and controlled information exchange across diferent representation levels. Fig.2 further presents the detailed structures of the three key components, including the Adaptive Feature Learner (AFL), the Cross-Feature Interaction (CFI) blocks, and the multi-scale decoder. To bridge the representational gap between the two branches, the AFL first extracts task-adaptive multi-scale spatial priors, and the CFI blocks then progressively refine these features by interacting with aligned global and fine-grained representations. Finally, the multi-scale decoder aggregates hierarchical features from both branches and produces the final real/fake prediction. Diferent from conventional hybrid frameworks that mainly fuse task-specific local/global features or texture/noise cues, DBCF performs spatially aligned hierarchical fusion of complementary foundation-model representations. By aligning and progressively interacting CLIP-based global semantic features with DINOv3-based fine-grained structural features, the proposed framework enables efective collaboration between heterogeneous pretrained feature spaces.

![](images/7763d004edca34f7ebe5a3e7cdb50ed1135004d228e989e04ffd5a9900949295.jpg)  
(a) Adaptive Feature Learner  
(b) Cross-Feature Interaction  
(c) Multi-scale Decoder  
Figure 2: Detailed structures of the core components in the proposed framework. (a) The Adaptive Feature Learner (AFL) extracts multi-scale spatial priors. (b) The Cross-Feature Interaction (CFI) block progressively exchanges complementary information between global context and fine-grained forgery cues. (c) The multi-scale decoder aggregates hierarchical interaction features for the final real/fake prediction.

## 3.2. Global Context Branch

A key challenge in deepfake detection is to achieve robust generalization across identities and manipulation methods while avoiding overfitting to localized, methodspecific artifacts. To address this issue, the Global Context Branch (GCB) is introduced to extract stable, high-level semantic representations associated with identity, pose, and expression. These cues serve as global semantic priors for forgery detection. Since fine-grained artifact cues are often fragile under distribution shifts, whereas global semantics are more invariant, we adopt a frozen CLIP visual backbone as the core of the GCB. CLIP is pretrained with a large-scale multimodal contrastive objective, which naturally encodes strong semantic alignment and invariance. Freezing the backbone preserves these pretrained inductive biases and mitigates semantic collapse during training.

Specifically, following common adapter designs in pretrained ViT frameworks [8, 32], given an input image x, the ViT backbone of CLIP produces a sequence of intermediate token representations $\{ H _ { g } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ , where $H _ { g } ^ { ( l ) } \in \mathbb { R } ^ { N _ { 1 } \times d _ { 1 } }$ denotes the output of the l-th transformer layer, with $N _ { 1 }$ being the number of tokens and $d _ { 1 }$ being the token embedding dimension of the GCB. To construct multi-granular global context, we select a subset of layers indexed by $\mathcal { L } _ { g } \subseteq \{ 1 , \ldots , L \}$ and define the global context features as

$$
{ \cal G } = \{ H _ { g } ^ { ( l ) } \mid l \in \mathcal { L } _ { g } \} .\tag{1}
$$

Following prior empirical studies on utilizing intermediate transformer representations for hierarchical feature construction [8], we select four representative layers from the CLIP visual encoder at diferent depths, i.e., $\mathcal { L } _ { g } = \{ 5 , 1 2 , 1 8 , 2 4 \}$ . The resulting multi-granular global features are subsequently paired with the FCB representations in the Cross-Feature Interaction module.

## 3.3. Fine-grained Cue Branch

While global semantic representations provide robustness, many deepfake artifacts manifest as localized and fine-grained inconsistencies, such as subtle texture distortions or boundary artifacts. To explicitly capture such cues, the Fine-grained Cue Branch (FCB) is designed to extract artifact-sensitive local representations that complement the global context.

Specifically, fine-grained forgery cues refer to localized manipulation traces that are often weak in global semantics but evident in local visual patterns. Typical examples include blending boundaries, texture/color/illumination mismatches, local blurring or detail degradation, and subtle structural distortions in facial components. Since these cues are spatially localized and closely related to patch-level structure and appearance, we employ DINOv3 to construct the FCB. Benefiting from its self-supervised visual pretraining, DINOv3 preserves rich local structural representations and has demonstrated strong performance on dense prediction tasks such as semantic segmentation, indicating its capability to model spatially detailed visual information. This property makes it well suited for complementing the CLIP branch, which mainly captures global semantic and contextual information. Moreover, DINOv3 supports variable input resolutions, enabling flexible spatial alignment with the Global Context Branch (GCB).

To facilitate token-wise interaction between the two branches, we adopt a resolutionadaptive preprocessing strategy that ensures spatial alignment. Specifically, we resize the input image x to a square resolution $S \times S$ with

$$
{ \cal S } = \sqrt { N _ { 1 } } \cdot P _ { \mathrm { F C B } } ,\tag{2}
$$

where $P _ { \mathrm { F C B } }$ denotes the patch size of the FCB backbone. With this choice, the token number produced by the FCB satisfies $N _ { 2 } = ( S / P _ { \mathrm { F C B } } ) ^ { 2 } = N _ { 1 }$ . The resized image is then fed into the FCB, producing a sequence of intermediate token representations $\{ H _ { f } ^ { ( l ) } \} _ { l = 1 } ^ { L }$ ， where $H _ { f } ^ { ( l ) } \in \mathbb { R } ^ { N _ { 2 } \times d _ { 2 } }$ and $d _ { 2 }$ denotes the feature dimension of the FCB.

The fine-grained features extracted by the FCB are formulated as:

$$
{ \cal F } = \{ H _ { f } ^ { ( l ) } \mid l \in \mathcal { L } _ { f } \} , \quad \mathcal { L } _ { f } = \{ 5 , 1 2 , 1 8 , 2 4 \} .\tag{3}
$$

These fine-grained features provide localized and artifact-sensitive cues that complement the global context for forgery detection.

## 3.4. Adaptive Feature Learner

While the Global Context Branch (GCB) and Fine-grained Cue Branch (FCB) provide complementary representations, the frozen features limit adaptability to datasetspecific manipulation patterns. To enhance adaptability to downstream tasks, following common practice in recent forgery detection frameworks [6, 8], we incorporate a parameter-eficient Adaptive Feature Learner (AFL) as an auxiliary component to provide task-adaptive cues.

As shown in Fig.2(a), the AFL is implemented as a trainable four-stage convolutional spatial prior module. It consists of a convolutional stem followed by three stride-2 convolutional stages, which progressively extract feature maps at 1/4, 1/8, 1/16, and

1/32 of the input resolution. Given an input image x, the AFL produces a hierarchy of multi-scale feature maps $C ^ { ( k ) } \in \mathbb { R } ^ { H _ { k } \times W _ { k } \times C _ { k } }$ , where the spatial resolution decreases and the semantic abstraction increases with k. These multi-scale features provide complementary spatial cues and serve as auxiliary representations for subsequent cross-feature interaction.

To enable unified processing with transformer-based features, each feature map is first projected by $\textbf { a } 1 \times 1$ convolution into a shared embedding space of dimension $d _ { c } = d _ { 1 } + d _ { 2 }$ . The three lower-resolution feature maps used for cross-feature interaction are augmented with a learnable level embedding $e ^ { ( k ) }$ and flattened into sequences of tokens:

$$
\tilde { C } ^ { ( k ) } = \mathrm { F l a t t e n } \big ( \mathrm { P r o j } _ { k } ( C ^ { ( k ) } ) \big ) + e ^ { ( k ) } \in \mathbb { R } ^ { N _ { k } \times d _ { c } } , \quad N _ { k } = H _ { k } W _ { k } , \quad k = 2 , 3 , 4 .\tag{4}
$$

For cross-feature interaction with the GCB and FCB, the sequences from scales $k = 2 , 3$ 4 are concatenated along the token dimension to form

$$
C _ { \mathrm { i n t } } ^ { ( 0 ) } = [ \tilde { C } ^ { ( 2 ) } ; \tilde { C } ^ { ( 3 ) } ; \tilde { C } ^ { ( 4 ) } ] \in \mathbb { R } ^ { N _ { \mathrm { i n t } } \times d _ { c } } , \qquad N _ { \mathrm { i n t } } = \sum _ { k = 2 } ^ { 4 } N _ { k } .\tag{5}
$$

The highest-resolution feature $C ^ { ( 1 ) }$ preserves its spatial structure and is reserved for the subsequent multi-scale decoding stage.

## 3.5. Cross-Feature Interaction

To integrate the multi-scale AFL features with spatially aligned global and local features, the Cross-Feature Interaction (CFI) module updates the interaction features via a cross-attention mechanism, as shown in Fig.2(b). Let $G _ { s } ~ \in ~ \mathbb { R } ^ { N _ { s } \times d _ { 1 } }$ and $F _ { s } \in \mathbb { R } ^ { N _ { s } \times d _ { 2 } }$ denote the global and local token representations at the s-th selected representation level, respectively, where $s = 1 , \ldots , 4$ corresponds to the selected transformer layers {5 12 18 24}. Since the input resolutions and patch sizes are chosen to yield spatially aligned token grids in the two branches, $G _ { s }$ and $F _ { s }$ are concatenated along the channel dimension.

Starting from the multi-scale AFL interaction features $C _ { \mathrm { i n t } } ^ { ( 0 ) }$ , each CFI module progressively updates them using the paired GCB–FCB features at the corresponding representation level:

$$
\begin{array} { r l } { C _ { \mathrm { i n t } } ^ { ( s ) } = C _ { \mathrm { i n t } } ^ { ( s - 1 ) } + \mathsf { M S D e f o r m A t t n } \big ( \mathrm { L a y e r N o r m } ( C _ { \mathrm { i n t } } ^ { ( s - 1 ) } ) , \qquad } & { } \\ { \mathrm { L a y e r N o r m } ( [ G _ { s } \mid F _ { s } ] ) \big ) , \qquad } & { s = 1 , \ldots , 4 . } \end{array}\tag{6}
$$

Here, ${ C } _ { \mathrm { i n t } } ^ { ( 0 ) }$ consists of the AFL features at $1 / 8 , 1 / 1 6 ,$ and $1 / 3 2$ input resolutions. The deformable attention operation allows these AFL features with diferent token lengths to interact with the spatially aligned global and local representations. In this way, the CFI modules progressively incorporate complementary global and local cues into the task-adaptive multi-scale features.

## 3.6. Multi-Scale Decoder and Training Objective

As shown in Fig. $. 2 ( \mathrm { c } )$ , the multi-scale decoder aggregates the refined hierarchical features through a top-down fusion strategy. Given the AFL and CFI outputs, we split the final CFI output token sequence $C _ { \mathrm { i n t } }$ into three groups and reshape them into spatial feature maps $C _ { 2 } , C _ { 3 }$ , and $C _ { 4 }$ at progressively lower resolutions. The highest-resolution feature map $C _ { 1 }$ is directly taken from the shallow AFL feature. When CFI features are used, the interaction outputs from GCB and FCB are partitioned and resized to match the corresponding AFL feature maps and added to them. After normalization, the preceding fusion process produces four spatial feature maps $\{ C _ { i } \} _ { i = 1 } ^ { 4 }$ at $1 / 4 , 1 / 8$ $1 / 1 6 ,$ , and $1 / 3 2$ input resolutions, respectively. The decoder progressively fuses these features to generate the final prediction.

Each feature map is first projected to a unified channel dimension $d _ { o }$ through a $1 \times 1$ convolution:

$$
P _ { i } = \mathrm { C o n v _ { 1 \times 1 } } ( C _ { i } ) , \quad i = 1 , . . . , 4 ,
$$

where $P _ { 1 }$ and $P _ { 4 }$ denote the highest and lowest resolution decoder features, respectively.

The decoder follows a top-down fusion strategy, where higher-level features are progressively upsampled and merged with lower-level ones. Specifically, the lowestresolution projected feature is retained as ${ \tilde { P } } _ { 4 } = P _ { 4 }$ , and the remaining levels are progressively fused as:

$$
\tilde { P } _ { 4 } = P _ { 4 } , \qquad \tilde { P } _ { i } = P _ { i } + \mathrm { U p } ( \tilde { P } _ { i + 1 } ) , \quad i = 3 , 2 , 1 .
$$

Each fused representation is then transformed into a scale-specific global descriptor:

$$
g _ { i } = { \bf G } \mathrm { A P } ( \tilde { P } _ { i } ) , i = 1 , 2 , 3 , 4 ,
$$

with Up(·) denoting bilinear interpolation and GAP(·) global average pooling.

The final image-level representation is obtained by concatenating all scale-specific descriptors and feeding it into a linear classifier to predict the two-class logits:

$$
{ \bf z } = \mathrm { F C } \big ( [ g _ { 1 } ; g _ { 2 } ; g _ { 3 } ; g _ { 4 } ] \big ) \in \mathbb { R } ^ { 2 } ,
$$

Finally, the model is optimized using the cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { c l s } } = - \log \frac { \exp ( z _ { y } ) } { \sum _ { c = 0 } ^ { 1 } \exp ( z _ { c } ) } ,
$$

where $y \in \{ 0 , 1 \}$ denotes the ground-truth label.

## 4. Experiments

## 4.1. Settings

## 4.1.1. Datasets

To comprehensively assess the efectiveness and generalization capability of the proposed method, we conduct experiments under both cross-dataset and cross-manipulation evaluation protocols. For cross-dataset evaluation, we follow the widely adopted setting in which the model is trained on one dataset and tested on several unseen datasets. Specifically, we use the $^ { \mathrm { c 2 3 } }$ compressed version of FaceForensics++ (FF++) [33] for training, which consists of 1,000 pristine videos and 4,000 manipulated videos generated by four manipulation techniques. Following the oficial split of FF++, only its training subset, consisting of 720 pristine videos and 2,880 manipulated videos, is used for training. No cross-validation or test-time adaptation is involved in the cross-dataset evaluation. The trained model is then evaluated on three challenging benchmarks: Celeb-DF v2 [34], DFDC [35], DFDCP [36], covering diverse real-world conditions and distribution shifts. Following the commonly adopted evaluation protocol for fair cross-dataset comparison, we evaluate our method on the oficial test split of each target benchmark. To further investigate robustness against unseen forgery types, we adopt

DF40 [37], a comprehensive dataset containing 40 manipulation techniques spanning a broad range of facial forgery categories, including face swapping, facial reenactment, and full-face synthesis.

## 4.1.2. Evaluation Metrics

We report both frame-level and video-level Area Under the ROC Curve (AUC) to accommodate diferent evaluation granularities. Following standard practice, videolevel AUC is computed by averaging the predicted probabilities over all sampled frames within each video. For extensive face anti-spoofing (FAS) experiment, we additionally employ the Half Total Error Rate (HTER) to further measure cross-domain generalization performance.

## 4.1.3. Implementation Details

We adopt CLIP-ViT-L/14-336 as the backbone for global feature extraction and DINOv3-ViT-L/16 to provide complementary fine-grained representations. AFL uses 64 base convolutional channels and projects the multi-scale features to a shared dimension of 2048. The decoder projection dimension is set to 512. During training, we sample 8 frames from each video. For the cross-dataset comparisons in Tables 1 and 2, we sample 32 frames from each test video to match the evaluation setting adopted by the reported baseline results. All frames are aligned using RetinaFace and resized to 224×224. Consistent with the spatial-alignment procedure described in Section 3, the preprocessed frames are subsequently resized to 336×336 for the CLIP-based GCB and 384×384 for the DINOv3-based FCB before being fed into the respective backbones. We use only image (patch) tokens for cross-branch alignment, excluding the CLS token. With patch sizes of 14 and 16, respectively, both branches produce a 24 × 24 grid containing 576 image tokens. Following other existing works, several image augmentations are introduced during training, including random Brightness Contrast, Image Compression, and the SBI-based augmentation strategy [2]. No additional data augmentation is applied during testing. All experiments are implemented within PyTorch framework and conducted on a single NVIDIA A100 GPU. All models are trained for 10 epochs. We use the Adam optimizer with a fixed learning rate of $2 \times 1 0 ^ { - 4 }$ for all trainable parameters. The batch size is set to 32 across all experiments to ensure fair comparison. For repeated-run experiments, we perform three independent runs using diferent fixed random seeds and report the mean AUC and the corresponding standard deviation across these runs.

Table 1: Cross-dataset comparison using frame-level AUC. The results reported in the table are taken from [38, 39], or directly obtained from the corresponding original papers.
<table><tr><td>Method</td><td>Venue</td><td>CDF</td><td>DFDC</td><td>DFDCP</td></tr><tr><td>Xception [33]</td><td>ICCV&#x27;19</td><td>73.65</td><td>70.77</td><td>73.74</td></tr><tr><td>FaceX-ray [1]</td><td>CVPR&#x27;20</td><td>67.86</td><td>63.26</td><td>69.42</td></tr><tr><td>RECCE [40]</td><td>CVPR’22</td><td>73.19</td><td>71.33</td><td>74.19</td></tr><tr><td>SBI [2]</td><td>CVPR&#x27;22</td><td>81.30</td><td>71.96</td><td>79.90</td></tr><tr><td>UIA-ViT [5]</td><td>ECCV&#x27;22</td><td>82.41</td><td>一</td><td>75.80</td></tr><tr><td>UCF [4]</td><td>ICCV&#x27;23</td><td>75.27</td><td>71.91</td><td>75.94</td></tr><tr><td>ProDet [41]</td><td>NIPS&#x27;24</td><td>84.48</td><td>72.40</td><td>81.16</td></tr><tr><td>LSDA[3]</td><td>CVPR&#x27;24</td><td>83.00</td><td>73.60</td><td>81.50</td></tr><tr><td>DeepFake-Adapter [6]</td><td>IJCV’25</td><td>71.74</td><td>72.66</td><td>一</td></tr><tr><td>FIA-USA [39]</td><td>NIPS’25</td><td>86.70</td><td>一</td><td>81.80</td></tr><tr><td>CRDA [42]</td><td>AAAI&#x27;26</td><td>85.36</td><td>74.29</td><td>79.73</td></tr><tr><td>DBCF (Ours)</td><td></td><td> $8 3 . 7 1 { \scriptstyle \pm 0 . 6 2 }$ </td><td> $7 8 . 0 2 { \pm } 0 . 2 2$ </td><td> $8 5 . 9 0 { \pm } 0 . 8 6 $ </td></tr><tr><td> $\mathrm { D B C F } \left( \mathrm { O u r s } \right) + \mathrm { S B I }$ </td><td>一 一</td><td> $8 3 . 2 5 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $\mathbf { 8 0 . 0 8 { \scriptstyle \pm 0 . 3 0 } }$ </td><td> $\mathbf { 8 8 . 6 6 { \pm 0 . 1 6 } }$ </td></tr></table>

## 4.2. Main Results

## 4.2.1. Cross-dataset evaluation

We first evaluate the generalization ability of our method by comparing frame-level Area Under the ROC Curve (AUC) across several unseen datasets, including Celeb-DF v2, DFDC, and DFDCP, which are completely excluded during training. Table 1 presents the cross-dataset results of our method alongside several representative baselines. Overall, our approach demonstrates strong generalization across all benchmarks. While some methods achieve slightly higher performance on CDF, DBCF attains the highest performance on DFDC and DFDCP, demonstrating strong overall generalization across the evaluated target datasets. The variation across target datasets can be attributed to their diferent degrees of distribution shift and manipulation diversity. Although CDF contains high-quality and visually realistic forged videos, its manipulation distribution is relatively concentrated, and thus transferable forgery patterns may remain comparatively consistent. In contrast, DFDC contains more diverse identities, backgrounds, acquisition conditions, and manipulation algorithms. Such real-world variations may obscure subtle local forgery traces, making DFDC more challenging, as reflected by the lower frame-level AUC of 78.02 compared with other datasets. We further investigate its compatibility with data-centric enhancement strategies. The results demonstrate that DBCF can be seamlessly combined with existing data augmentation methods, such as SBI [2], leading to performance improvements across multiple datasets. These findings suggest that DBCF ofers a solid foundation for cross-dataset deepfake detection and shows potential for further improvement when combined with complementary data augmentation techniques during training.

Table 2: Comparison with SOTA methods using the video-level AUC. Results marked with \* are obtained by using the authors’ released models or code, while the remaining results are directly taken from [38] or corresponding original papers.
<table><tr><td>Method</td><td>Venue</td><td>CDF</td><td>DFDC</td><td>DFDCP</td></tr><tr><td>FaceX-ray [1]</td><td>CVPR’20</td><td>一</td><td>一</td><td>71.1</td></tr><tr><td>FTCN [43]</td><td>ICCV’21</td><td>86.9</td><td>67.6</td><td>74.0</td></tr><tr><td>SBI [2]</td><td>CVPR’22</td><td>92.8</td><td>71.9</td><td>85.5</td></tr><tr><td>UIA-ViT* [5]</td><td>ECCV’22</td><td>82.4</td><td>75.0</td><td>75.8</td></tr><tr><td>CFM* [44]</td><td>TIFS’23</td><td>85.3</td><td>75.0</td><td>80.2</td></tr><tr><td>AltFreezing* [45]</td><td>CVPR&#x27;23</td><td>85.1</td><td>71.7</td><td>79.3</td></tr><tr><td>LSDA [3]</td><td>CVPR&#x27;24</td><td>91.1</td><td>77.0</td><td>81.2</td></tr><tr><td>NACO [46]</td><td>AJSE&#x27;25</td><td>89.5</td><td>76.7</td><td>-</td></tr><tr><td>SDR [47]</td><td>ICASSP’25</td><td>88.5</td><td>76.2</td><td>一</td></tr><tr><td>M2F2-Det* [48]</td><td>CVPR&#x27;25</td><td>83.2</td><td>76.4</td><td>71.2</td></tr><tr><td>FIA-USA [39]</td><td>NIPS&#x27;25</td><td>94.1</td><td>73.2</td><td>86.6</td></tr><tr><td>DBCF (Ours)</td><td></td><td>88.6±0.4</td><td>80.9±0.3</td><td>87.7±0.6</td></tr><tr><td>DBCF (Ours) + SBI</td><td>一 一</td><td>89.2±0.2</td><td> $\mathbf { 8 } 2 . 9 { \pm } \mathbf { 0 } . 2 $ </td><td>92.5±0.1</td></tr></table>

We also report video-level AUC in Table 2. The video-level results exhibit trends consistent with the frame-level evaluation, while showing improved stability across datasets. This indicates that the proposed method produces temporally consistent predictions across frames, and that video-level aggregation helps mitigate the impact of noisy frame-level predictions, particularly on challenging datasets such as DFDC.

Table 3: Cross-manipulation comparison on five representative face swapping forgery types in DF40 [37] using frame-level AUC (%)
<table><tr><td>Method</td><td>Venue</td><td>uniface</td><td>facedancer</td><td>fsgan</td><td>inswap</td><td>simswap</td><td>Avg.</td></tr><tr><td>RECCE [40]</td><td>CVPR&#x27;22</td><td>84.2</td><td>78.3</td><td>88.4</td><td>79.5</td><td>73.0</td><td>80.7</td></tr><tr><td>SBI [2]</td><td>CVPR’22</td><td>64.4</td><td>44.7</td><td>87.9</td><td>63.3</td><td>56.8</td><td>63.4</td></tr><tr><td>IID [49]</td><td>CVPR&#x27;23</td><td>79.5</td><td>79.0</td><td>86.4</td><td>74.4</td><td>64.0</td><td>76.7</td></tr><tr><td>UCF [4]</td><td>ICCV’23</td><td>78.7</td><td>80.0</td><td>88.1</td><td>76.8</td><td>64.9</td><td>77.7</td></tr><tr><td>LSDA [3]</td><td>CVPR&#x27;24</td><td>85.4</td><td>75.9</td><td>83.2</td><td>81.0</td><td>72.7</td><td>79.6</td></tr><tr><td>CDFA [50]</td><td>ECCV’24</td><td>76.5</td><td>75.4</td><td>84.8</td><td>72.0</td><td>76.1</td><td>77.0</td></tr><tr><td>ProDet [41]</td><td>NeurIPS&#x27;24</td><td>84.5</td><td>73.6</td><td>86.5</td><td>78.8</td><td>77.8</td><td>80.2</td></tr><tr><td>FIA-USA [39]</td><td>NIPS&#x27;25</td><td>91.8</td><td>83.0</td><td>86.3</td><td>87.4</td><td>91.0</td><td>87.8</td></tr><tr><td>DBCF (Ours)</td><td>一</td><td>97.7±0.3</td><td>90.0±0.3</td><td>95.4±0.2</td><td>92.3±0.8</td><td>93.4±1.3</td><td>93.8±0.4</td></tr><tr><td>DBCF (Ours) + SBI</td><td>一</td><td>97.9±0.5</td><td>91.1±0.5</td><td>95.3±0.4</td><td>93.1±1.9</td><td>94.8±0.4</td><td>94.4±0.4</td></tr></table>

## 4.2.2. Cross-manipulation evaluation

To further examine the robustness of our method against unseen manipulation types, we conduct a cross-manipulation evaluation on the DF40 benchmark [37], which contains five distinct generation pipelines: uniface, facedancer, fsgan, inswap, and simswap. As reported in Table 3, our method obtains higher AUC than the listed baselines on the five evaluated manipulation pipelines and achieves the highest average AUC in this comparison. Nevertheless, noticeable performance diferences can be observed across manipulation methods. In particular, facedancer and simswap are relatively more challenging than uniface and fsgan. This may be because modern high-fidelity face-swapping pipelines are designed to better preserve facial attributes and reduce visible blending inconsistencies, leaving weaker and more localized forgery traces for detection. In contrast, manipulation methods that leave more visible and straightforward forgery artifacts are comparatively easier to identify.

A comparative examination of Tables 1–3 reveals that many existing deepfake detection methods may achieve strong performance on certain datasets, but their efectiveness often fails to generalize to more diverse and complex forgery patterns in the DF40 cross-manipulation setting. This indicates that prior approaches might exploit datasetspecific or manipulation-dependent cues rather than capturing intrinsic forgery characteristics. In contrast, our method consistently achieves strong performance across all unseen types. Moreover, its improvements are particularly evident on relatively challenging manipulation methods, indicating improved robustness to high-fidelity forgeries with less conspicuous artifacts. We attribute this robustness to the use of more comprehensive and complementary representations, which allow the model to focus on fundamental forgery traits that remain stable despite significant shifts in manipulation strategies and generation mechanisms.

## 4.2.3. Ablation Study

Table 4 presents a comprehensive ablation study evaluating the individual and joint contributions of the Global Context Branch (GCB), the Fine-grained Cue Branch (FCB), and the Multi-granular feature aggregation module (MG). Using either GCB or FCB alone yields reasonable performance, indicating that both global semantic consistency and fine-grained artifacts are informative for forgery detection. However, directly concatenating the outputs of GCB and FCB (without MG) does not consistently improve performance and even degrades results on some datasets. This suggests that simple concatenation of heterogeneous features is insuficient for efective integration. The performance drop may stem from directly merging features with diferent distributions, which can lead to collapsed or suboptimal feature patterns. Introducing MG consistently improves performance by leveraging hierarchical representations instead of relying solely on final-layer tokens. When combined with both branches, our Adaptive Feature Learner and Cross-Feature Interaction mechanism further enhance performance, demonstrating that spatially aligned and interaction-aware fusion is critical for efectively integrating global and local cues. Enabling all components achieves the best results across datasets, validating the efectiveness of our multi-granular and structured fusion design for generalizable deepfake detection.

To further assess the efectiveness-eficiency trade-of of the proposed dual-branch framework, we compare four representative variants in Table 5. The average framelevel AUC is computed over the five evaluation settings reported in Table 4. Consistent with the component-wise observations above, DBCF achieves the best average performance among all evaluated variants, confirming that the gain provided by the dualbranch framework is substantial rather than marginal. In terms of eficiency, DBCF requires 56 ms per frame, corresponding to approximately 17.9 FPS, with 678M parameters and 544 GFLOPs. Although incorporating complementary foundation models introduces additional computational overhead, the resulting inference speed remains acceptable under frame-sampled deepfake detection. Therefore, the improved generalization capability represents a reasonable trade-of against the increased latency and computational cost. More specifically, compared with naive concatenation, DBCF improves the average frame-level AUC from 79.03% to 88.06%, while increasing inference latency from 33 ms to 56 ms per frame and computational cost from 367 to 544 GFLOPs. These results explicitly quantify the additional computational cost associated with the performance improvement of DBCF.

Table 4: Ablation study of diferent components on frame-level AUC across multiple datasets. GCB denotes the Global Context Branch for modeling global semantic consistency, FCB denotes the Fine-grained Cue Branch for capturing local manipulation artifacts, and MG denotes the multi-granular feature aggregation module for hierarchical feature fusion.
<table><tr><td colspan="3">Component Settings</td><td colspan="5">Frame-level AUC</td></tr><tr><td>GCB √</td><td>MG</td><td>FCB</td><td> $\mathrm { F F } { + } { + }$ </td><td> $_ \mathrm { C e l e b - D F }$ </td><td>DFDC</td><td>facedancer</td><td>inswap</td></tr><tr><td rowspan="4">√</td><td></td><td></td><td> $8 2 . 6 { \pm } 0 . 1 $ </td><td> $7 5 . 6 { \pm } 1 . 0 $ </td><td> $7 2 . 6 { \pm } 0 . 6 $ </td><td> $7 7 . 4 { \pm } 0 . 6 $ </td><td> $7 6 . 2 { \pm } 0 . 9 $ </td></tr><tr><td></td><td>√</td><td> $9 2 . 5 { \pm } 0 . 3 $ </td><td> $8 2 . 8 { \pm } 0 . 3 $ </td><td> $7 3 . 2 { \pm } 0 . 2 $ </td><td> $7 5 . 0 { \pm } 0 . 3 $ </td><td> $7 6 . 1 { \pm } 0 . 6 $ </td></tr><tr><td></td><td>√</td><td> $8 8 . 8 { \pm } 0 . 2 $ </td><td> $7 5 . 7 { \pm } 0 . 7 $ </td><td> $7 3 . 2 { \pm } 0 . 5 $ </td><td> $7 8 . 0 { \pm } 0 . 5 $ </td><td> $7 9 . 4 { \pm } 1 . 1 $ </td></tr><tr><td>√</td><td>√</td><td> $9 6 . 0 { \pm } 0 . 1 $ </td><td> $8 0 . 7 { \pm } 0 . 4 $ </td><td> $7 6 . 8 { \pm } 0 . 9 $ </td><td> $8 8 . 1 { \pm } 0 . 4 $ </td><td> $9 2 . 2 { \pm } 1 . 4 $ </td></tr><tr><td>√</td><td>√</td><td></td><td> $9 5 . 9 { \pm } 0 . 2 $ </td><td> $8 1 . 7 { \pm } 0 . 9 $ </td><td> $7 5 . 3 { \pm } 0 . 5 $ </td><td> $8 3 . 9 { \pm } 0 . 6 $ </td><td> $9 2 . 3 { \pm } 1 . 2 $ </td></tr><tr><td>√</td><td>√</td><td>√</td><td> ${ \bf 9 6 . 3 { \pm } 0 . 1 }$ </td><td> ${ \bf 8 3 . 7 \pm 0 . 6 }$ </td><td> $7 8 . 0 { \pm } 0 . 2 $ </td><td> ${ \bf 9 0 . 0 { \pm } 0 . 3 }$ </td><td> $\mathbf { 9 2 . 3 { \pm 0 . 8 } }$ </td></tr></table>

Deepfakes  
Table 5: Efectiveness-eficiency comparison of representative model variants. Latency and computational metrics are measured under the same inference setting and reported on a per-frame basis.
<table><tr><td>Variant</td><td>Avg. frame-level AUC↑</td><td>Latency (per frame)</td><td>Params</td><td>GFLOPs</td></tr><tr><td>GCB only</td><td>76.83</td><td>30 ms</td><td>330M</td><td>248</td></tr><tr><td>FCB only</td><td>79.92</td><td>31 ms</td><td>330M</td><td>249</td></tr><tr><td>GCB + FCB (Naive concat)</td><td>79.03</td><td>33 ms</td><td>607M</td><td>367</td></tr><tr><td>DBCF (Ours)</td><td>88.06</td><td>56 ms</td><td>678M</td><td>544</td></tr></table>

Face2Face  
NeuralTextures  
FaceSwap  
CDF  
DFDC  
![](images/4caadbbba4db884fad9e871466818e7bf37e0072da6a6081b88d6ba672902bf1.jpg)  
Figure 3: Qualitative visualization of attention maps from the GCB and FCB branches on diferent forgery datasets.

## 4.2.4. Visualization

To qualitatively examine the complementary behavior of GCB and FCB, we visualize the branch-wise averaged attention maps in Fig.3. The GCB maps show relatively dispersed responses over the face and nearby regions, while the FCB maps exhibit more spatially coherent and locally concentrated responses within facial regions. This difference suggests that GCB tends to capture broader semantic/contextual cues, whereas FCB is more sensitive to localized appearance patterns, which is consistent with their complementary roles in deepfake detection.

SBI UIA-ViT Altfreezing Ours  
![](images/572aa54104fda641c3807f00f41dc0282033ab83f33b8c735a3c2467624e440f.jpg)

![](images/465d5611b9fd32145858fce161e32c85790cf54040a5ebdcf6c8bebc03744c5a.jpg)

![](images/3397c25c0ea15b5f97dd3aaa1b204cc78103430c3ef721e8bf5039922f959567.jpg)

![](images/7ebacfc0921d9303c9c6dea56546ef7ddef6216f52a9210348f506f21b87095b.jpg)  
Figure 4: Frame-level AUC reduction ratio under diferent degradation levels and perturbation types. "Average" score represents the mean across all levels for each type of perturbation.

## 4.2.5. Robustness Analysis

Following AltFreezing [45], we evaluate the robustness of diferent methods under common image perturbations in real-world scenarios, including image compression, RGB shift, and contrast variation. We report the frame-level AUC reduction ratio, defined as $\mathrm { ( A U C _ { r a w } - A U C ) / A U C _ { r a w } }$ , where $\mathrm { \bf A U C } _ { \mathrm { r a w } }$ denotes the performance on clean (non-degraded) samples. A smaller value indicates stronger robustness. As shown in Fig. 4, our method consistently exhibits the lowest AUC reduction across most perturbations, highlighting its robustness. While image compression causes a moderate drop due to the loss of subtle texture cues, performance under RGB shift and contrast variation remains largely unafected.

## 4.3. Extensive Experiments on Face Anti-Spoofing (FAS)

We further evaluate the efectiveness of our framework on the face anti-spoofing task under cross-dataset settings. Experiments are conducted on four widely used benchmarks, and each source-target combination follows the standard Leave-One-Out protocol. As shown in Table 6, we compare our method with several state-of-the-art approaches, including NAS-FAS [52], SSAN-R [53], PatchNet [54], SA-FAS [55], and the more recent AG-FAS [51]. Our proposed DBCF achieves consistently strong performance across all transfer scenarios in terms of both HTER and AUC. Specifically, the dual-branch model (Ours dual) maintains HTERs below 5 for all domain-shift settings, with AUC up to 99.66, outperforming the compared methods in average metrics (Mean HTER = 3.71, Mean AUC = 99.36).

Table 6: Performance comparison of diferent face anti-spoofing methods under cross-dataset evaluation. All results for the compared methods are taken from [51]. The metrics reported include HTER and AUC for each source-target dataset combination, as well as the mean performance across all settings.
<table><tr><td rowspan="2">Method</td><td>O&amp;C&amp;I→M</td><td>O&amp;M&amp;I→C</td><td>O&amp;C&amp;M→I</td><td>C&amp;I&amp;M→O</td><td> $\mathbf { A v } \mathbf { g }$ </td><td>Avg</td></tr><tr><td>HTER↓/AUC↑</td><td>HTER↓/AUC↑</td><td>HTER↓/AUC↑</td><td>HTER↓/AUC↑</td><td>HTER↓</td><td>AUC↑</td></tr><tr><td>NAS-FAS</td><td>19.53 / 88.63</td><td>16.54 / 90.18</td><td>14.51 /93.84</td><td>13.80 / 93.43</td><td>16.10</td><td>91.52</td></tr><tr><td>SSAN-R</td><td>6.67 / 98.75</td><td>10.00 / 96.67</td><td>8.88 / 96.79</td><td>13.72 / 93.63</td><td>9.82</td><td>96.46</td></tr><tr><td>PatchNet</td><td>7.10/98.46</td><td>11.33 / 94.58</td><td>13.40 /95.67</td><td>11.82 /95.07</td><td>10.91</td><td>95.95</td></tr><tr><td>SA-FAS</td><td>5.95 / 96.55</td><td>8.78 / 95.37</td><td>6.58 / 97.54</td><td>10.00 / 96.23</td><td>7.83</td><td>96.42</td></tr><tr><td>AG-FAS</td><td>5.71/98.03</td><td>5.44 /98.55</td><td>6.71 / 98.23</td><td>9.43 / 96.62</td><td>6.82</td><td>97.86</td></tr><tr><td>Ours (GCB)</td><td>5.95 / 98.50</td><td>2.67 / 99.54</td><td>17.14/91.24</td><td>10.14 / 96.31</td><td>8.98</td><td>96.40</td></tr><tr><td>Ours (FCB)</td><td>4.52 / 98.54</td><td>1.22 / 99.96</td><td>3.71 / 99.59</td><td>5.99 / 98.68</td><td>3.86</td><td>99.19</td></tr><tr><td>Ours (Dual)</td><td>5.71 / 98.64</td><td>1.33 / 99.85</td><td>4.86 / 99.29</td><td>2.92 / 99.66</td><td>3.71</td><td>99.36</td></tr></table>

We also conduct ablation studies to investigate the contributions of the global and local feature branches. Using only the GCB branch achieves low HTER in certain settings (e.g., 2.67% on O&M&I → C) but sufers in others (17.14 on O&C&M → I), while the FCB branch alone improves overall HTER stability. The dual-branch design efectively combines the strengths of both branches, achieving robust performance across all cross-dataset scenarios. Together, these results support the utility of combining global and local representations for the evaluated face forgery and spoofing detection tasks.

## 5. Conclusion

In this work, we proposed a dual-branch framework, namely DBCF, for detecting facial forgeries and spoofing attempts. It captures both global and local manipulation cues from manipulated images, enabling the model to learn more comprehensive and discriminative representations for forgery detection. Motivated by the complementary strengths of pretrained visual models, DBCF leverages CLIP’s strong semantic-level generalization while incorporating DINO’s fine-grained local features, allowing the framework to benefit from both high-level transferable semantic knowledge and subtle local artifact perception. This combination is particularly important for facial forgery and spoofing detection, where both overall semantic consistency and fine-grained manipulation traces contribute to reliable prediction. To efectively fuse these complementary cues without disrupting the pretrained feature patterns, we introduce a multiscale spatial alignment mechanism that enables comprehensive learning of manipulation representations. By promoting more efective interaction and alignment across heterogeneous features at diferent spatial levels, the proposed mechanism improves the integration of global and local information while preserving the intrinsic strengths of each branch. Extensive experiments under cross-dataset and cross-manipulation settings show that our method achieves the best performance on most of the evaluated unseen datasets and manipulation types, demonstrating its generalization capability and robustness under distribution shifts. These results verify the efectiveness of the proposed design and suggest that combining heterogeneous pretrained representations is a promising direction for open-domain forgery detection. Overall, our framework provides a flexible foundation for open-domain detection tasks and ofers potential for future extensions to video-level and multi-modal detection scenarios.

## 6. Limitations

Despite its strong generalization performance, DBCF has several practical limitations. Its complementary representation relies on two large pretrained backbones, CLIP-ViT-L and DINOv3-ViT-L. Although both backbones are frozen during training, jointly running them increases model size, computational cost, inference latency, and GPU memory demand compared with single-backbone alternatives. Specifically, DBCF requires 56 ms for single-frame inference, corresponding to approximately 17.9 FPS, with a peak GPU memory consumption of approximately 6.1 GB under the same inference setting. While these costs are accompanied by the performance gains reported in our evaluations, they may still restrict deployment in strict real-time or resource-constrained scenarios. Moreover, the interpretability of DBCF remains limited because real-world forgery patterns can be highly complex and may not be cleanly attributed to either localized manipulation traces or global semantic inconsistencies. Consequently, although the two branches are designed to model complementary representations, the current framework does not explicitly identify how diferent cues interact in each detection decision. Future work will investigate eficient backbone replacement, model compression or distillation, and cue visualization or forgery localization mechanisms to improve deployment eficiency and provide more explicit evidence for detection decisions.

## CRediT authorship contribution statement

Fengming Gu: Formal analysis, Investigation, Methodology, Project administration, Software, Validation, Writing – original draft Writing, – review & editing. Mingjie He: Funding acquisition, Supervision, Validation, Writing – review & editing. Zonghui Guo: Funding acquisition, Supervision, Writing – review & editing. Jie Zhang: Funding acquisition, Project administration, Writing – original draft, Writing – review & editing. Shiguang Shan: Supervision, Writing – review & editing.

## Acknowledgments

This work is partially supported by the Strategic Priority Research Program of the Chinese Academy of Sciences (No. XDB0680202), the Beijing Nova Program (No. 20230484368), the National Natural Science Foundation of China (No. 62276249), the National Natural Science Foundation of China (No. 62306298), the TaiShan Scholars Youth Expert Program of Shandong Province (No. tsqn202507108), and the Youth Innovation Promotion Association of the Chinese Academy of Sciences.

## References

[1] L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen, B. Guo, Face x-ray for more general face forgery detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 5001–5010.

[2] K. Shiohara, T. Yamasaki, Detecting deepfakes with self-blended images, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 18720–18729.

[3] Z. Yan, Y. Luo, S. Lyu, Q. Liu, B. Wu, Transcending forgery specificity with latent space augmentation for generalizable deepfake detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8984–8994.

[4] Z. Yan, Y. Zhang, Y. Fan, B. Wu, Ucf: Uncovering common features for generalizable deepfake detection, in: Proceedings of the IEEE/CVF International Conference on Computer vision, 2023, pp. 22412–22423.

[5] W. Zhuang, Q. Chu, Z. Tan, Q. Liu, H. Yuan, C. Miao, Z. Luo, N. Yu, Uia-vit: Unsupervised inconsistency-aware method based on vision transformer for face forgery detection, in: European Conference on Computer Vision, Springer, 2022, pp. 391–407.

[6] R. Shao, T. Wu, L. Nie, Z. Liu, Deepfake-adapter: Dual-level adapter for deepfake detection, International Journal of Computer Vision 133 (6) (2025) 3613–3628.

[7] J. Thies, M. Zollhöfer, M. Nießner, Deferred neural rendering: Image synthesis using neural textures, Acm Transactions on Graphics (TOG) 38 (4) (2019) 1–12.

[8] O. Siméoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa, et al., Dinov3, arXiv preprint arXiv:2508.10104 (2025).

[9] R. Chen, X. Chen, B. Ni, Y. Ge, Simswap: An eficient framework for high fidelity face swapping, in: Proceedings of the 28th ACM International Conference on Multimedia, 2020, pp. 2003–2011.

[10] K. Liu, I. Perov, D. Gao, N. Chervoniy, W. Zhou, W. Zhang, Deepfacelab: Integrated, flexible and extensible face-swapping framework, Pattern Recognition 141 (2023) 109628.

[11] A. Siarohin, S. Lathuilière, S. Tulyakov, E. Ricci, N. Sebe, First order motion model for image animation, Advances in Neural Information Processing Systems 32 (2019).

[12] K. Prajwal, R. Mukhopadhyay, V. P. Namboodiri, C. Jawahar, A lip sync expert is all you need for speech to lip generation in the wild, in: Proceedings of the 28th ACM International Conference on Multimedia, 2020, pp. 484–492.

[13] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, T. Aila, Analyzing and improving the image quality of stylegan, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 8110–8119.

[14] T. Karras, M. Aittala, S. Laine, E. Härkönen, J. Hellsten, J. Lehtinen, T. Aila, Alias-free generative adversarial networks, Advances in Neural Information Processing Systems 34 (2021) 852–863.

[15] P. Esser, R. Rombach, B. Ommer, Taming transformers for high-resolution image synthesis, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 12873–12883.

[16] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, High-resolution image synthesis with latent difusion models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10684–10695.

[17] L. Shi, J. Zhang, Z. Ji, J. Bai, S. Shan, Real face foundation representation learning for generalized deepfake detection, Pattern Recognition 161 (2025) 111299.

[18] Q. Gao, B. Zhang, J. Wu, W. Luo, Z. Teng, J. Fan, Leveraging facial landmarks improves generalization ability for deepfake detection, Pattern Recognition 164 (2025) 111528.

[19] J. Li, Y. Hu, B. Liu, H. She, C.-T. Li, Deepfake detection with domain generalization and mask-guided supervision, Pattern Recognition 165 (2025) 111622.

[20] C. Zhao, C. Wang, Z. Song, G. Hu, L. Wang, D. Miao, Multi-definition deepfake detection via semantics reduction and cross-domain training, Pattern Recognition 163 (2025) 111469.

[21] M. Long, Z. Liu, L.-B. Zhang, F. Peng, Lgdf-net: Local and global feature-based dual-branch fusion networks for deepfake detection, IEEE Transactions on Circuits and Systems for Video Technology 35 (6) (2025) 5489–5500.

[22] Y. Ding, H. Zhai, Q. Ma, L. Zhang, L. Shao, F. Bu, Face forgery detection via multi-scale dual-modality mutual enhancement network, Computers, Materials & Continua 85 (1) (2025).

[23] S. A. Khan, D.-T. Dang-Nguyen, Clipping the deception: Adapting visionlanguage models for universal deepfake detection, in: Proceedings of the 2024 International Conference on Multimedia Retrieval, 2024, pp. 1006–1015.

[24] U. Ojha, Y. Li, Y. J. Lee, Towards universal fake image detectors that generalize across generative models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24480–24489.

[25] Z. Huang, B. Xia, Z. Lin, Z. Mou, W. Yang, J. Jia, Ffaa: Multimodal large language model based explainable open-world face forgery analysis assistant, arXiv preprint arXiv:2408.10072 (2024).

[26] Y. Chen, Z. Yan, G. Cheng, K. Zhao, S. Lyu, B. Wu, X<sup>2</sup>-DFD: A framework for explainable and extendable deepfake detection, in: The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[27] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations, 2021.

[28] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al., Learning transferable visual models from natural language supervision, in: International Conference on Machine Learning, PmLR, 2021, pp. 8748–8763.

[29] K. He, H. Fan, Y. Wu, S. Xie, R. Girshick, Momentum contrast for unsupervised visual representation learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 9729–9738.

[30] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, Masked autoencoders are scalable vision learners, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 16000–16009.

[31] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al., Dinov2: Learning robust visual features without supervision, Transactions on Machine Learning Research Journal (2024).

[32] Z. Chen, Y. Duan, W. Wang, J. He, T. Lu, J. Dai, Y. Qiao, Vision transformer adapter for dense predictions, in: The Eleventh International Conference on Learning Representations, 2023.

[33] A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Nießner, Faceforensics++: Learning to detect manipulated facial images, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1–11.

[34] Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu, Celeb-df: A large-scale challenging dataset for deepfake forensics, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 3207–3216.

[35] B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, C. C. Ferrer, The deepfake detection challenge (dfdc) dataset, arXiv preprint arXiv:2006.07397 (2020).

[36] B. Dolhansky, R. Howes, B. Pflaum, N. Baram, C. C. Ferrer, The deepfake detection challenge (dfdc) preview dataset, arXiv preprint arXiv:1910.08854 (2019).

[37] Z. Yan, T. Yao, S. Chen, Y. Zhao, X. Fu, J. Zhu, D. Luo, C. Wang, S. Ding, Y. Wu, et al., Df40: Toward next-generation deepfake detection, Advances in Neural Information Processing Systems 37 (2024) 29387–29434.

[38] Z. Yan, Y. Zhang, X. Yuan, S. Lyu, B. Wu, Deepfakebench: A comprehensive benchmark of deepfake detection, Advances in Neural Information Processing Systems 36 (2023) 4534–4565.

[39] L. Ma, Z. Yan, J. Xu, Y. Chen, Q. Guo, Z. Bi, Y. Liao, et al., From specificity to generality: Revisiting generalizable artifacts in detecting face deepfakes, Advances in Neural Information Processing Systems 38 (2026) 69306–69344.

[40] J. Cao, C. Ma, T. Yao, S. Chen, S. Ding, X. Yang, End-to-end reconstructionclassification learning for face forgery detection, in: Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 2022, pp. 4113–4122.

[41] J. Cheng, Z. Yan, Y. Zhang, Y. Luo, Z. Wang, C. Li, Can we leave deepfake data behind in training deepfake detector?, Advances in Neural Information Processing Systems 37 (2024) 21979–21998.

[42] Y. Chou, T. Yu, W. Huang, T. Dai, S.-T. Xia, et al., Improving deepfake detection with reinforcement learning-based adaptive data augmentation, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, 2026, pp. 3381–3389.

[43] Y. Zheng, J. Bao, D. Chen, M. Zeng, F. Wen, Exploring temporal coherence for more general video face forgery detection, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 15044–15054.

[44] A. Luo, C. Kong, J. Huang, Y. Hu, X. Kang, A. C. Kot, Beyond the prior forgery knowledge: Mining critical clues for general face forgery detection, IEEE Transactions on Information Forensics and Security 19 (2023) 1168–1182.

[45] Z. Wang, J. Bao, W. Zhou, W. Wang, H. Li, Altfreezing for more general video face forgery detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4129–4138.

[46] M. Alshehri, Deep fake video face recognition using supervised contrastive learning for scalability and interpretability, Arabian Journal for Science and Engineering 50 (15) (2025) 11779–11802.

[47] B. Chu, X. Xu, Y. Zhang, W. You, L. Zhou, Reduced spatial dependency for more general video-level deepfake detection, in: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[48] X. Guo, X. Song, Y. Zhang, X. Liu, X. Liu, Rethinking vision-language model in face forensics: Multi-modal interpretable forged face detector, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 105–116.

[49] B. Huang, Z. Wang, J. Yang, J. Ai, Q. Zou, Q. Wang, D. Ye, Implicit identity driven deepfake face swapping detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 4490–4499.

[50] Y. Lin, W. Song, B. Li, Y. Li, J. Ni, H. Chen, Q. Li, Fake it till you make it: Curricular dynamic forgery augmentations towards general deepfake detection, in: European Conference on Computer Vision, Springer, 2024, pp. 104–122.

[51] X. Long, J. Zhang, S. Shan, Generalized face liveness detection via de-fake face generator, IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).

[52] Z. Yu, J. Wan, Y. Qin, X. Li, S. Z. Li, G. Zhao, Nas-fas: Static-dynamic central diference network search for face anti-spoofing, IEEE Transactions on Pattern Analysis and Machine Intelligence 43 (9) (2020) 3005–3023.

[53] Z. Wang, Z. Wang, Z. Yu, W. Deng, J. Li, T. Gao, Z. Wang, Domain generalization via shufled style assembly for face anti-spoofing, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 4123–4133.

[54] C.-Y. Wang, Y.-D. Lu, S.-T. Yang, S.-H. Lai, Patchnet: A simple face antispoofing framework via fine-grained patch recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 20281–20290.

[55] Y. Sun, Y. Liu, X. Liu, Y. Li, W.-S. Chu, Rethinking domain generalization for face anti-spoofing: Separability and alignment, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 24563– 24574.