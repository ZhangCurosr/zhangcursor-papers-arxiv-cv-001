# AHMAD: Adaptive Hybrid Multi-task Vision Learning with Assisted Distillation

Mohammad Mahdi<sup>\*</sup> Nedyalko Prisadnikov Yuqian Fu Carmelo Scribano Danda Pani Paudel Luc Van Gool {firstname.lastname}@insait.ai

## Abstract

Generalist multitasking vision models aim to unify multiple vision tasks within a single framework, enabling more efficient and versatile learning. However, handling diverse vision tasks—spanning dense and sparse predictions—remains challenging due to their inherently varying output structures. In this paper, we propose AHMAD, a simple yet effectiveframeworkfor generalist multitask learning that integrates different key vision tasks: semantic segmentation, instance segmentation, depth estimation, keypoint detection, and object detection. Our approach incorporates these five tasks into a unified structure – a shared encoderdecoder with several lightweight task-specific projectors. Under the multitask learning paradigm, we observed a complementary performance gain, achieving a state-of-the-art PQ of 53.1 and an mIoU of 66.5 for COCO-val panoptic and semantic segmentation, respectively. Additionally, for top-down keypoint detection, which typically incurs high computational overhead due to multiple forward passes, we introduce a knowledge distillation-based method that enables a single forward pass over the entire image, greatly improving efficiency. Ultimately, our model delivers a lightweight yet effective generalist multitask learning framework, demonstrating strong performance across five vision tasks.

## 1. Introduction

Generalist models aim to solve multiple tasks within a single unified framework, eliminating the need for separate task-specific models. Given the inherently multitask nature, technically, building generalist models is closely aligned with multi-task learning, sharing the goal of optimizing a model to handle diverse tasks efficiently.

In the NLP domain, generalist multitask learning has achieved remarkable success, with models such as [1, 16], demonstrating strong performance across diverse language tasks. This success is largely attributed to the shared nature of text representations, where different language tasks can be formulated within a common sequence-to-sequence or auto-regressive framework. However, developing generalist multitask learning in vision remains is significantly more challenging due to the inherent diversity in vision tasks. Vision tasks span a broad range of dense and sparse predictions, each requiring distinct output structures and optimization objectives. For example, semantic segmentation operates at the pixel level, depth estimation requires continuous value regression, while object detection involves structured bounding box predictions. These fundamental differences make it difficult to design a generalist framework that seamlessly accommodates all vision tasks within a single model.

![](images/1764c0e89c8cf015f12379199def73f8ab1b30c7d0fe1d98bd60b4d73e7260a2.jpg)  
Figure 1. Our Proposed Training Framework. We leverage a powerful vision transformer encoder combined with a CNN-based decoder to generate a rich feature map, which contains representations useful for various tasks. Task-specific projectors are then applied to perform different tasks.

To tackle this, some studies focus on task output homogenization, where various task outputs are manually encoded into a shared format through hand-crafted methods. In this context, Painter [51] approaches many vision tasks as image inpainting problems, encoding task outputs in the RGB space. Pix2Seq-D [6] also leverages the Bit Diffusion model [5] to learn task-specific outputs by converting them into per-pixel representations across separate channels. Alternatively, other approaches add components to serialize both the input image and task outputs, treating the problem as a next-token prediction task. For instance, Unified-IO [29] employs a VQ-GAN [13] to serialize dense tasks and adds special tokens for sparse tasks. In this paradigm, the generalist model is trained on the frozen tokens, and the outputs need to be further decoded. However, both these approaches come with trade-offs, as they introduce complexity through demanding pre- and post-processing, along with the need for sophisticated training techniques, such as VQ-GAN for serialization.

In generalist multitasking, it is also essential to unify different vision tasks under a common problem-solving methodology. Among these tasks, person keypoint detection often follows a top-down strategy, where individuals are cropped from the image, and keypoints are then detected for each cropped person. This approach has become the standard due to the effectiveness of person detectors. However, it requires running the model multiple times, once for each individual in the image, which goes against the core idea of generalist multitasking, where the goal is to process the entire image and perform multiple tasks in a single forward pass. This multiple-run process is especially expensive in generalist vision models, which rely on large, computationally intensive encoder backbones. Despite this, no significant effort has been made to address this issue while still leveraging the strengths of the top-down approach.

We propose a simple yet effective training framework, as in Fig. 1, namely AHMAD, for generalist multitasking across different vision tasks: Panoptic Segmentation (Semantic Segmentation (SS) and Instance Segmentation (IS)), Depth Estimation (DE), Keypoint Detection (KD), and Object Detection (OD), demonstrating a complementary boost for the majority of tasks. Additionally, we introduce a novel method to reduce the forward passes required for top-down keypoint detection, achieving a single run for the entire image using knowledge distillation techniques.

Our contributions are as follows: 1) We present a simple multitask learning paradigm with no restrictions on task output shapes or the need for extra components, allowing the incorporation of each task’s ideal single-task approach within a unified multitask structure. 2) We propose a novel distillation method to perform top-down keypoint detection with just a single forward pass for the entire image. 3) We demonstrate strong performance across various vision tasks, achieving a PQ of 53.1 and an mIoU of 66.5 for COCO panoptic and semantic segmentation among generalist models, while also delivering competitive results in keypoint detection and object detection.

## 2. Related Work

Vision Transformers. Vision Transformers (ViTs) [21, 37] have emerged as powerful foundations for both multi-task and multimodal learning in computer vision [45]. Swin Transformer [28] introduces hierarchical feature by partitioning images into non-overlapping windows and computing self-attention within each. This enables efficient modeling of long-range dependencies through a shifting window mechanism. ViTs also excel in self-supervised learning [26, 44, 57], acting as strong backbones for various vision tasks. Methods like DINO [3] and DINOv2 [34] leverage unsupervised pretraining based on knowledge distillation on large unlabeled data, enabling them to learn transferable representations for different downstream tasks. Furthermore, ViTs are effective in multimodal learning [18], particularly in vision-language tasks [56] such as visual question answering [43], where combining visual and textual data enhances overall comprehension.

Segmentation and Object Detection. Transformerbased models excel in dense tasks like segmentation [31]. MaskFormer [7] frames segmentation as hierarchical mask generation and classification, while Mask2Former [8] refines it with multi-scale masked attention. OneFormer [19] extends this with task-conditioned training for handling diverse segmentation tasks. SAM [22] generates high-quality object masks from both sparse and dense prompts, using a MAE-pretrained Vision Transformer [17] as the encoder and a prompt encoder for boxes, points, and text (encoded using CLIP [36]), with support for dense mask prompts encoded by a CNN network. SAM2 [38] builds on this with a promptable visual segmentation framework, incorporating a memory bank and attention mechanism for improved accuracy. Transformers [47] are also widely used for open-vocabulary object detection [9, 12, 15, 32, 33, 55], leveraging autoregressive next-token prediction. Grounding DINO [27] detects objects from language inputs, with a Swin transformer for image encoding, BERT [10] for text, and deformable self-attention [52] for feature enhancement. It also employs language-guided query selection and a cross-modality decoder for bounding box prediction. Grounding DINO 1.5 [41] enhances this with Pro and Edge versions, offering a larger ViT-L backbone and cross-scale feature fusion for improved performance. Grounded SAM [42] merges Grounding DINO and SAM for both segmentation and detection, using BLIP [24, 25] and RAM [58] for vision-language understanding.

Multitask Generalist Models. Expanding beyond models that are strictly designed for segmentation and detection tasks, Painter [51] frames dense vision tasks as image inpainting, with outputs in the same RGB space as input images. The task prompt is encoded as a pair of images, and a Masked Vision Transformer is used for training. At inference, Painter offers flexible task adaptation through three types of prompts: random, searched, and learned. Unified-IO [29, 30] introduced an encoder-decoder architecture, using stacked transformer layers to represent all inputs and outputs as discrete tokens. It utilizes VQ-GAN [13] for image serialization and incorporates location tokens and sparse structures like bounding boxes as special tokens. However, the model has limitations in localization. Pix2Seq [4] takes an image and task prompts to generate discrete tokens corresponding to desired outputs, while Pix2Seq-D [6] handles semantic and instance segmentation with two channels per task. GiT [48] divides the image into subregions, generating sparse responses for each, which are then merged. DINO-X [40] extends the Grounding DINO encoder-decoder architecture for open-world understanding, supporting text, visual, and customized prompts. The DINO-X Pro version uses a pre-trained CLIP model to enhance performance across multimodal benchmarks, with visual prompts leveraging box and point formats for better segmentation and visual grounding tasks.

## 3. Method

Given a set of task IDs $T ~ = ~ \{ 1 , 2 , \ldots , k \}$ , each corresponding to a specific vision task, we collect datasets $M _ { j }$ where $j \in T$ , representing either individual datasets or a merged version of multiple datasets associated with task $j .$ We then establish a shared encoder backbone network $B _ { \theta }$ (referred to as the backbone for simplicity) along with a relatively lightweight decoder $D _ { \phi } .$ , forming a feature extraction pipeline. Additionally, we define k shallow, taskspecific projectors $H _ { \psi _ { i } } ^ { j }$ , each designed to align the output shape with the requirements of task j.

Our objective is to train an encoder-decoder network that learns a shared and robust feature map $F = D _ { \phi } ( B _ { \theta } ( \cdot ) )$ ensuring it captures all the essential information needed to perform various vision tasks. This feature map $F$ is then fed to task-specific projectors, whose main function is to adjust the output shape to match the requirements of each task, rather than extracting additional features from $F .$

Specifically, we iteratively select a random vision task ID $j \in T$ and sample a training data batch $( x , y ) \in M _ { j }$ , where x represents the input RGB data and y is the corresponding target. For simplicity, we define $F _ { x } = D _ { \phi } ( B _ { \theta } ( x ) )$ . In each iteration, the objective is to optimize the network by minimizing the task-specific loss function $\mathcal { L } _ { j } \left( H _ { \psi _ { j } } ^ { j } ( F _ { x } ) , y \right)$ Figure 2 illustrates the mechanism, including the network architecture and the training strategy.

Backbone. Our backbone is built upon ViT-L [11], utilizing weights obtained via the Dino-V2 [34] strategy, which leverages self-supervised learning. This reliance on large-scale pretrained weights enables the model to learn rich, generalizable feature representations that are not tied to any specific task. This is particularly well-suited for our design, where task-specific projectors specialize the learned features for different tasks. This enables the model to efficiently share common features while adapting to the unique requirements of each task.

Decoder. The decoder comprises four transposed convolutional layers whose goal is to upscale the patch level representations to pixel level representations. Although the decoder is significantly lighter than the backbone, it is considerably larger than the task-specific projectors. This is because the task projectors are not intended to focus on deeply extracting new features, but to simply project this representation to the required output format for the task.

## 3.1. Task-specific Projectors

Given an RGB input x of shape $H \times W \times 3 .$ , Each intermediate layer of the backbone outputs a sequence of tokens with shape $\left( { \frac { H } { s } } * { \frac { W } { s } } \right) \ \times$ dim, where s is the patch size of the ViT network and dim is the dimension of the tokens. To construct a latent image representation z, we concatenate tokens from four intermediate layers of the backbone. The upscaling decoder then processes this representation to generate a dense per-pixel feature map $F _ { x }$ of shape $H \times W \times O$ where the channel dimension O is set to 256 in our experiments. Finally, task-specific projectors utilize $F _ { x }$ to optimize their respective loss functions based on the given task.

For the first three tasks (SS, IS, and DE) involving pixelwise loss, we do not treat all pixels equally. Instead, we employ Edge Distance Sampling (EDS), which focuses training on pixels near object boundaries. For each pixel $( i , j )$ we calculate its Euclidean distance $d _ { i j }$ to the nearest boundary. The corresponding weight for that pixel is given by:

$$
w _ { i j } = w _ { \operatorname* { m i n } } + ( 1 - w _ { \operatorname* { m i n } } ) e ^ { - d _ { i j } ^ { 2 } / D ^ { 2 } } ,
$$

where $w _ { \mathrm { m i n } }$ represents the minimum weight assigned to pixels far from boundaries. In our experiments, we set $w _ { \mathrm { m i n } } =$ 0. The parameter $D$ controls how rapidly the weight diminishes with increasing distance from the boundary, with $D = 2 0$ in our experiments.

Edge Distance Sampling addresses two critical issues: (i) Unlabeled regions are effectively excluded, reducing ambiguity. (ii) Loss imbalance between small and large objects is mitigated by concentrating on instance boundaries.

SS Projector. This projector formulates the semantic segmentation task $( j = 1 )$ as per-pixel classification, following [34]. Specifically, the loss function is a per-pixel weighted cross-entropy loss, formulated as:

$$
\mathcal { L } _ { 1 } \left( H _ { \psi _ { 1 } } ^ { 1 } ( F _ { x } ) : = \hat { y } , y \right) = \sum _ { i = 1 } ^ { H } \sum _ { j = 1 } ^ { W } w _ { i j } \cdot C E \left( \hat { y } _ { i j } , y _ { i j } \right) ,
$$

where the cross-entropy loss $C E ( \cdot )$ is applied at each spatial location $( i , j )$ to refine segmentation accuracy. The $H _ { \psi _ { 1 } } ^ { 1 } ( F _ { x } )$ is a single $3 \times 3$ convolution layer from channel dimension O to the number of categories in the dataset.

IS Projector. We tackle class-agnostic instance segmentation $( j ~ = ~ 2 )$ by encoding instances via their center of mass. Each pixel predicts the coordinates $( u , v )$ of the corresponding instance centroid, while unlabeled pixels receive a void encoding. The loss depends on the accuracy of centroid predictions, with higher precision needed in regions where centroids are close together (e.g., near object boundaries), as small errors can lead to incorrect instance assignments.

![](images/e9e89eeceebce9758259547173e0c4640f3cda798bc9984a41ca7eb102c0774d.jpg)  
Figure 2. Method Overview. In each iteration, a random vision task—illustrated here with object detection—is selected. After patch tokenization, the backbone processes the tokens, and outputs from four intermediate layers are concatenated to form $z .$ This is then decoded to a shared feature map $F _ { x } = D _ { \phi } ( z )$ , which is further processed by the corresponding task-specific projector. For the first three tasks, EDS is applied during loss minimization (\*). The gradient-colored backbone highlights layer decay in $B _ { \theta } ,$ , where later layers receive stronger updates while earlier layers remain more stable, promoting better feature adaptation. Gray/dim arrows indicate inactive tasks, and for task projectors, input and outer arrows represent the forward and backward propagation, respectively.

To address this, we apply positional embeddings to map centroid coordinates into a higher-dimensional space. The u and v coordinates are first normalized to the range [−1, 1], then embedded as:

$$
\gamma ( p ) = \left[ \sin ( 2 ^ { k } \pi p ) , \cos ( 2 ^ { k } \pi p ) \right] _ { k = 0 } ^ { L - 1 } \in \mathbb { R } ^ { 2 L } ,
$$

where $L$ is the number of harmonics used, controlling the dimensionality of the embedding for coordinate $p .$

For a pixel $( i , j )$ , we define a predictor function $\mathbb { P } ( i , j )$ that returns a concatenated positional embedding $\gamma ( u ) \oplus$ $\gamma ( v )$ if pixel $( i , j )$ belong to an instance with centroid at $( u , v )$ , and returns a zero vector $Z \in \mathbb { R } ^ { 4 L }$ if the pixel is unlabeled. Thus, the target pixel output $y _ { i j }$ is:

$$
y _ { i j } = \mathbb { P } ( i , j ) \cdot ( \gamma ( u ) \oplus \gamma ( v ) ) + ( 1 - \mathbb { P } ( i , j ) ) \cdot Z
$$

The instance segmentation loss is then defined as:

$$
\mathcal { L } _ { 2 } \left( H _ { \psi _ { 2 } } ^ { 2 } ( F _ { x } ) : = \hat { y } , y \right) = \sum _ { i = 1 } ^ { H } \sum _ { j = 1 } ^ { W } w _ { i j } \cdot L ( \hat { y } _ { i j } , y _ { i j } ) ,
$$

where the pixel loss $L ( \hat { y } _ { i j } , y _ { i j } )$ is computed as the average euclidean distance between prediction and the target embeddings:

$$
{ \cal L } ( { \hat { y } } _ { i j } , y _ { i j } ) = { \frac { 1 } { 2 } } \sum _ { n = 1 } ^ { 2 } \left\| ( y _ { i j } - { \hat { y } } _ { i j } ) _ { [ 2 ( n - 1 ) L + 1 : 2 n L ] } \right\| ^ { 2 }
$$

In our experiments, $H _ { \psi _ { 2 } } ^ { 2 } ( F _ { x } )$ is a single $3 \times 3$ convolution layer from channel dimension O to $4 L$

During inference, the u and v coordinates are retrieved from $\hat { y } _ { i j }$ using the nearest neighbor search [35], matching predictions to positions in a discretized uv-grid space.

Panoptic Segmentation. We split panoptic segmentation into semantic and class-agnostic instance segmentation, solving them independently using the pipeline outlined in the SS and IS projectors sections.

DE Projector. For monocular depth estimation $( j = 3 )$ we minimize a weighted affine-invariant loss in disparity space. We define disparity d as the inverse depth, normalized between 0 and 1, with the depth range set from 0 to 10. The projection layer, $H _ { \psi _ { 4 } } ^ { 4 } ( F _ { x } )$ , consists of a single 3×3 convolution with a sigmoid activation. It maps the feature map from channel dimension O to a single-channel output and optimizes the following loss:

$$
\mathcal { L } _ { 3 } \left( H _ { \psi _ { 3 } } ^ { 3 } ( F _ { x } ) : = \hat { d } , d \right) = \sum _ { i = 1 } ^ { H } \sum _ { j = 1 } ^ { W } w _ { i j } \cdot \left| \hat { d } _ { i j } - d _ { i j } \right|
$$

OD Projector. For the object detection task $( j ~ = ~ 4 )$ we follow a Yolo-style [20, 39] approach but modify the architecture by replacing fully connected layers with convolutional layers for the final predictions. The last layer applies a sigmoid activation function to produce the output.

The image is divided into an $S \times S { \mathrm { g r i d } }$ , where each cell predicts B bounding boxes, each defined by $( a , b , w , h , p )$ Here, $a , b$ are the normalized center coordinates within the grid cell, w, h are normalized to the image, and p represents the predicted IoU with the ground truth.

From the B predicted boxes per cell, we select the one with the highest IoU. Each grid cell $( i , j )$ , also predicts a classification vector $\hat { c } _ { i j }$ of length $C ,$ where C is the number of object categories. The goal is for $\hat { c } _ { i j }$ to closely match the ground truth class vector $c _ { i j }$ , which is a one-hot vector when an object is present in the grid cell and a zero vector otherwise. Thus, $H _ { \psi _ { 4 } } ^ { 4 } ( F _ { x } )$ produces an output of shape $S \times$ $S \times ( 5 B + C )$

Specifically, the loss function consists of three main components: confidence loss $( { \mathcal { L } } _ { \mathrm { p } } ) _ { \mathrm { : } }$ , coordinate loss $( \mathcal { L } _ { \mathrm { a b w h } } )$ and classification loss $( { \mathcal { L } } _ { \mathrm { c } } ) { \mathrm { : } }$

$$
\mathcal { L } _ { \mathrm { p } } = \lambda _ { p _ { 1 } } \sum _ { i , j } \hat { p } _ { i j } ^ { 2 } \cdot \mathcal { k } _ { \mathrm { n o o b j } } + \lambda _ { p _ { 2 } } \sum _ { i , j } \left( \hat { p } _ { i j } - \mathrm { I } \hat { \bf o } \hat { \bf U } _ { i j } \right) ^ { 2 } \cdot \mathcal { k } _ { \mathrm { o b j } } ,
$$

$$
\begin{array} { c } { { \mathcal { L } _ { \mathrm { a b w h } } = \lambda _ { a } \displaystyle \sum _ { i , j } \left( \| \hat { a } _ { i j } - a _ { i j } \| ^ { 2 } + \| \hat { b } _ { i j } - b _ { i j } \| ^ { 2 } + \right. } } \\ { { \left. \qquad \| \hat { w } _ { i j } ^ { \frac 1 2 } - w _ { i j } ^ { \frac 1 2 } \| ^ { 2 } + \| \hat { h } _ { i j } ^ { \frac 1 2 } - h _ { i j } ^ { \frac 1 2 } \| ^ { 2 } \right) \cdot | \mathcal { k } _ { \mathrm { o b j } } , } } \\ { { \mathcal { L } _ { \mathrm { c } } = \lambda _ { c } \displaystyle \sum _ { i , j } \left( \hat { c } _ { i j } - c _ { i j } \right) ^ { 2 } \cdot | \mathcal { k } _ { \mathrm { o b j } } , } } \end{array}
$$

where $\nVdash _ { \mathrm { o b j } }$ is the boolean mask that selects the grid cells containing objects, $\nVdash _ { \mathrm { n o o b j } }$ is the boolean mask for grid cells without objects, and · performs tensor masking. The IoU<sup>ˆ</sup> represents the predicted IoU between the best predicted bounding box and the ground truth bounding box during training. For training, we set the weighting coefficients as $1 0 \lambda _ { p _ { 1 } } = 2 \lambda _ { p _ { 2 } } = \lambda _ { a } = 5 \lambda _ { c } ,$ , leading to the following loss:

$$
\mathcal { L } _ { 4 } \left( H _ { \psi _ { 4 } } ^ { 4 } ( F _ { x } ) , y \right) = \mathcal { L } _ { \mathrm { p } } + \mathcal { L } _ { \mathrm { a b w h } } + \mathcal { L } _ { \mathrm { c } }
$$

During inference, we utilize class-aware and DIoU-based Non-Maximum Suppression (NMS) to effectively eliminate duplicate bounding boxes.

KD Projector. We solve the 17-joint keypoint detection problem $( j ~ = ~ 5 )$ using a top-down approach, where cropped instances of individuals are processed independently. Instead of predicting separate heatmaps for each joint, we generate a single unified heatmap capturing all joints simultaneously, along with 17 classification channels trained exclusively on ground truth keypoints. We found this formulation significantly more efficient than the conventional approach of predicting one heatmap per joint. With all but one channel representing discrete class labels, this encoding is particularly well-suited for knowledge distillation, enabling the proposal of $\mathrm { K D } ^ { * }$ (3.2). The projector $H _ { \psi _ { 5 } } ^ { 5 } ( F _ { x } )$ is a single $3 \times 3$ convolution layer from channel dimension O to 18.

To optimize the model, we employ a hybrid loss function combining per-pixel mean squared error (MSE) loss for heatmap regression and cross-entropy loss for joint classification, applied only at ground truth locations:

$$
\mathcal { L } _ { 5 } \left( H _ { \psi _ { 5 } } ^ { 5 } ( F _ { x } ) , y \right) = \mathbf { M } \mathrm { S E } \left( \hat { h } t , h t ^ { \sigma } \right) + \sum _ { i , j } \mathbf { C } \mathbf { E } \left( \hat { c } _ { i j } , c _ { i j } \right) \cdot \boldsymbol { k } _ { \mathrm { g t } }
$$

The last channel of $H _ { \psi _ { 5 } } ^ { 5 } ( F _ { x } )$ $\hat { h } t ,$ , is the predicted heatmap of size $h s ,$ and $h t ^ { \dot { \sigma } }$ represents a Gaussian distribution palette centered at each keypoint location with standard deviation $\sigma .$ Additionally, cˆ, the first 17 channels of $H _ { \psi _ { 5 } } ^ { 5 } ( F _ { x } )$ , is the predicted class values. $c _ { i j }$ is the one-hot ground truth class vector at heatmap location $( i , j )$ , and ${ \mathbb { H } } _ { \mathrm { g t } }$ is a binary mask selecting the pixels that correspond to ground truth class labels.

During inference, we apply a class-aware argmax operation to the heatmap, selecting the highest-scoring location for each joint while filtering out low-confidence detections.

## 3.2. KD<sup>\*</sup>:Single-pass Top-down Keypoint Detection

In top-down keypoint detection, the entire network must run repeatedly for each instance in an image, leading to increased inference time and resource consumption—an issue exacerbated by the use of a large backbone. This contradicts the essence of efficient multitasking. To overcome this, we leverage knowledge distillation to enable the keypoint detection with just one forward pass for the entire image.

In our streamlined implementation of the keypoint detection task, we found that strong optimization performance could be achieved by directly using a part of the latent image representation ${ \it \Psi _ { 4 } } z = { \it \Psi _ { 4 } } B _ { \theta } ( x )$ , bypassing the decoder (indicated by the blue arrows in Figure 3). This approach generates a feature map suitable for cropped RGB images of individuals, but in a much more efficient manner, without being part of the multitasking pipeline. Building on this observation, we propose a knowledge distillation paradigm that aligns the feature map of cropped images with the corresponding regions of the full-image feature map. This technique allows us to reduce the forward passes from one per individual to a single pass for the entire image, essentially cropping the feature map instead of the RGB input.

![](images/67482ca04e286b3721bf34d99cfdab046cfccee456e5e600603e7016411125fe.jpg)  
Figure 3. $\mathbf { K D } ^ { * }$ Overview. Blue arrows indicate paths that are inactive during inference, while red arrows represent the active inference path. By enabling feature cropping, we reduce the number of forward passes from N to 1. In this process, $_ { 4 } z$ represents the latent image embedding obtained from the last of the four intermediate layers, $w ^ { \prime } = 4 w ,$ , and $h ^ { \prime } = 4 h$ . The green lines indicate contributions to the loss function, while the dashed red line represents the point at which we obtain the final output during the inference stage.

Let $x _ { G }$ be an image containing N individuals, and $x _ { C } ^ { i }$ be the cropped image corresponding to the i-th person. We define a cropping function $Q ( F ,$ , Box) that returns the cropped portion of the feature map $F$ based on the bounding box Box, interpolated to the same size as $F .$ . It is easy to see that $Q ( x _ { G } , \mathrm { B o x } _ { i } ) \ = \ x _ { C } ^ { i } .$ The goal is to minimize the disparity between student $S = Q ( F _ { x _ { G } } ^ { \prime } , \mathrm { B o x } _ { i } )$ and teacher $T = U _ { \gamma } { \big ( } _ { 4 } B _ { \theta } { \big ( } Q ( x _ { G } , \mathrm { B o x } _ { i } ) { \big ) } { \big ) }$ , as shown in Figure 3, where $U _ { \gamma }$ is a shallow upscaler. In the inference stage, we replace the N forward runs of $D _ { \phi } ( B _ { \theta } ( Q ( x _ { G } , \operatorname { B o x } _ { i } ) ) ) _ { i = 1 } ^ { N }$ with N cropping operations $Q ( D _ { \phi } ( B _ { \theta } ( x _ { G } ) )$ ), $\mathrm { B o x } _ { i } ) _ { i = 1 } ^ { N }$ , reducing the computation to a single forward run.

A challenge with our feature-cropping strategy in $Q ( F _ { x _ { G } } , \mathrm { B o x } _ { i } )$ is that small boxes result in a significant loss of spatial information in $F _ { x _ { G } } ,$ , leading to suboptimal features. To address this, we introduce a training-free upscaler T F U that increases the spatial resolution of $F _ { x _ { G } }$ by a factor of 4, enhancing its robustness to small boxes. The resulted feature map is denoted as $F _ { x _ { G } } ^ { \prime }$ . More specifically, TFU processes an $H \times W \times O$ feature map by dividing the O channels into $O / 1 6$ subchannels, each containing a $4 \times 4$ grid of values. The grid then replaces the original pixel at each spatial location, enhancing the spatial resolution by 4x while reducing the number of channels by 1/16x.

We use a hybrid loss to distill information from T to S:

$$
\begin{array} { r } { \mathcal { L } ^ { \ast } ( S , T ) = \underbrace { \mathcal { L } _ { 5 } \left( \hat { y } _ { t } , y \right) } _ { \mathcal { L } _ { 1 } ^ { \ast } } + \underbrace { \mathcal { L } _ { 5 } \left( \hat { y } _ { s } , y \right) } _ { \mathcal { L } _ { 2 } ^ { \ast } } + \mathcal { L } ^ { d } \left( S , T \right) , } \end{array}
$$

with the distillation loss function defined as:

$$
\begin{array} { r } {  { \mathcal { L } } ^ { d } \left( S , T \right) = \underbrace { \rho _ { 1 } \| S - T \| _ { 1 } } _ {  { \mathcal { L } } _ { 1 } ^ { d } } + \underbrace { \rho _ { 2 } \mathbf { K } \mathbf { L } ( \hat { y } _ { t } ^ { : 1 7 } \parallel \hat { y } _ { s } ^ { : 1 7 } ) } _ {  { \mathcal { L } } _ { 2 } ^ { d } } , } \end{array}
$$

where $\hat { y } _ { t } = H _ { \psi _ { 5 } } ^ { 5 } ( T )$ and $\hat { y } _ { s } = H ^ { \prime } { } _ { \psi _ { 5 } } ^ { 5 } ( S )$ . Note that $H _ { \psi _ { 5 } } ^ { 5 }$ and $\boldsymbol { H ^ { \prime } } _ { \psi _ { 5 } } ^ { 5 }$ have identical network architectures, but do not share weights. KL denotes the Kullback-Leibler divergence loss, and the superscript <sup>:17</sup> indicates the first 17 channels of the corresponding tensor. In our experiments, we set $\rho _ { 1 } =$ 1.0 and $\rho = 5 \times 1 0 ^ { - 5 }$

## 4. Experiments

## 4.1. Setup and Implementation

Three-Stage Training. We adopt a three-stage training strategy. In the first stage, we train only on the IS task, the most challenging one, for a substantial number of epochs using low-resolution images (280×280). In the second stage, we extend training to all tasks while maintaining the same low resolution and training for many epochs. Finally, in the third stage, we fine-tune the pretrained weights on high-resolution images (616×616) with significantly reduced augmentation and for considerably fewer epochs. The idea here is that most of the training occurs on computationally efficient low-resolution data, while fine-tuning on high-resolution images refines the model effectively without incurring excessive computational cost.

The three training stages run for 280, 150, and 60 epochs, respectively, totaling 5 days. Each stage uses a $5 \times 1 0 ^ { - 4 }$ learning rate, 0.6 decay, a $1 \times 1 0 ^ { - 6 }$ minimum learning rate, and 0.6 layer decay for the backbone. A warmup period of 10 epochs is applied. The effective batch sizes are 1024, 512, and 256 for the three stages, respectively. We use 4 intermediate layers of the backbone, specifically layers 4, 11, 17, and 23. We set $L = 4$ in IS task, as ablated in Table 3. In the object detection task, we use (B, S) of (2, 7) and (2, 10) in the second and third stages, respectively. For KD task, the heatmap size, hs, is set to $4 / 7 \ : ( \sigma = 1 0 )$ of the image resolution in the second and third stages. For tasks involving EDS, we include an additional total variation loss as a regularizer. Specifically for the IS task, we also incorporate generalized DICE loss (SupMat 1.2).

More Training Details. Training is conducted on ${ 8 \times \mathrm { A 1 0 0 ~ G P U s } }$ , with the ViT-Large backbone $B _ { \theta }$ and CNNbased upscaler decoder $D _ { \phi }$ containing 305M and 4M parameters, respectively. We target the COCO benchmark for training all five tasks. For depth estimation, we use pseudo-labels from DepthAnythingV2 [53, 54]. Experiments on $K D ^ { * }$ are conducted only at a low resolution of 280 within the previously stated training setup, using $U _ { \gamma }$ as a 1M-parameter upscaler, constructed by 3 transposed convolutional layers.

## 4.2. Main Results on COCO

We evaluate the performance of our multi-task learning pipeline using task-specific metrics on the COCO-val split. Specifically, for panoptic segmentation, we report Panoptic Quality (PQ), and for semantic segmentation, we use mean Intersection over Union (mIoU). For depth estimation, we report Root Mean Squared Error (RMSE), for object detection, we use mean Average Precision (mAP), and for keypoint detection, we provide OKS-based mAP. Table 1 compares these results with those from other specialized and multi-task generalist vision models, highlighting key performance trends.

Notably, our approach achieves strong performance in panoptic segmentation, surpassing other models with a PQ of 53.1 and an mIoU of 66.5. This represents a 22.3% boost compared to Painter and a 5.5% boost compared to Pix2Seq-D. As for semantic segmentation, we outperform LDM by a ∆mIoU of 6.4 and significantly surpass GiT with a ∆mIoU of 14.1. For depth estimation, we only compare the results of $\mathbf { A H M A D _ { \mathrm { s i n g l e } } }$ to $\mathrm { \ A H M A D _ { m u l t i } }$ (since other models use different ground truth labels for training, as noted by \* in Table 1), highlighting the complementary boost gained from joint training. We also achieve competitive results with an OKS-mAP of 68.12 for keypoint detection and an mAP of 44.6 for object detection, demonstrating the effectiveness of our multi-task framework in improving performance through joint training.

Figure 4 provides a visualization of our approach. Several examples with predictions on segmentation masks, depth estimations, object detections, and keypoint predictions, demonstrating how our model generalizes across different tasks within a unified framework.

As for our proposed KD<sup>\*</sup> method, we demonstrate how distilling information from the teacher feature map to the student feature map allows us to reduce the number of forward passes by a factor of 2.7x. This makes the entire multi-task paradigm unified, where the model processes the same input image for different tasks and generates the results in a single forward pass. Table 2 presents the statistics. The results from Table 4 further demonstrate that, with sufficient engineering effort, performance progressively improves, validating the efficacy of the approach. Additionally, there is potential for further enhancement as more engineering is applied.

Beyond the performance gains achieved through knowledge distillation, it is noteworthy that the bypassed path alone achieves strong metric scores. This highlights the suitability and effectiveness of the DINO-V2 backbone within our hybrid training framework. To better approximate real-world inference in top-down keypoint detection, we introduce 20% noise into the ground-truth bounding boxes when cropping input images.

![](images/d9905941fb3210ad7317a10d580dde553bf17a8dc7c373ce4b4d446ca1567f91.jpg)  
Figure 4. Qualitative Results. We show our results on five different tasks with diverse examples.

## 5. Ablation Study

For semantic segmentation, we investigated the impact of the EDS mechanism. In the case of instance segmentation, we also explored the effect of positional embeddings across different values of L, as summarized in Table 3.

For keypoint and object detection, we primarily conducted ablation studies on various hyperparameters, such as σ in keypoint detection and the number of grids S in object detection, along with different NMS versions for postprocessing. For example, we compared using DBSCAN and KMeans to obtain the final joints from candidates in keypoint detection, as well as IoU/dIoU-based NMS in object detection. In our post-processing analysis, we found that joint-aware argmax over heatmap values outperforms clustering-based methods, and dIoU-based NMS performs better than IoU-based NMS in object detection.

Finally, for the Knowledge Distillation approach, we primarily examined the effect of various loss functions and different training strategies, as detailed in Table 4.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Resolution</td><td rowspan="2">#Params</td><td rowspan="2">Backbone</td><td colspan="2">Segmentation</td><td colspan="2">Depth</td><td colspan="2">Keypoint</td></tr><tr><td>PQ↑ mIoU↑</td><td>RMSE↓</td><td> $\bf { m A P _ { 5 0 } }$ </td><td>Detection 个  $\mathbf { m A P _ { a l l } } \uparrow$ </td><td>OKS-mAP50 ↑</td><td> $\mathbf { O K S - m A P _ { a l l } } \uparrow$ </td></tr><tr><td colspan="8">Specialized Models</td></tr><tr><td>MaskFormer</td><td>640 × 640</td><td>212M</td><td>SWIN-L</td><td>52.7</td><td>1</td><td>X X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Mask2Former</td><td>640 × 640</td><td>216M</td><td>SWIN-L</td><td>57.8</td><td>-</td><td>X</td><td>X X</td><td>X</td><td>X</td></tr><tr><td>OneFormer</td><td> $6 4 0 \times 6 4 0$ </td><td>223M</td><td>DINAT-L</td><td>58.0</td><td>67.4</td><td>X</td><td>X X</td><td>X</td><td>X</td></tr><tr><td>Faster-RCNN [14]</td><td>600 × 1000</td><td>60M</td><td>ResNet-101</td><td>X</td><td>X</td><td>X</td><td>- 44.0</td><td>X</td><td>X</td></tr><tr><td>DETR [2]</td><td> $8 0 0 \times 8 0 0$ </td><td>60M</td><td>ResNet-101/DC5</td><td>45.1</td><td>-</td><td>X</td><td>64.7 44.9</td><td>X</td><td>X</td></tr><tr><td>HRNET [49]</td><td> $2 5 6 \times 1 9 2$ </td><td>28.5M</td><td>HRNET-V1-W32</td><td>X</td><td>X</td><td>X</td><td>X X</td><td>90.5</td><td>74.4</td></tr><tr><td colspan="10">Multi-Task Models</td></tr><tr><td>LDM [46]</td><td>512 × 512</td><td>850M</td><td>UNET</td><td>43.3</td><td>60.1</td><td> $0 . 0 7 5 ^ { * }$ </td><td>X X</td><td>X</td><td>X</td></tr><tr><td>Pix2Seq-D</td><td>1024 × 1024</td><td>94.5M</td><td>ResNet-50</td><td>50.3</td><td>-</td><td>X</td><td>X X</td><td>X</td><td>X</td></tr><tr><td>Pix2Seq</td><td>1024 × 1024</td><td>132M</td><td>ViT-B</td><td>X</td><td>×</td><td>×</td><td>- 46.5</td><td>-</td><td>64.8</td></tr><tr><td>Mask-RCNN [50]</td><td> $3 2 0 \times 3 2 0$ </td><td>43M</td><td>ResNet-101+FPN</td><td>X</td><td>X</td><td>X</td><td>67.8 45.0</td><td>87.3</td><td>66.5</td></tr><tr><td>UViM [23]</td><td> $1 2 8 0 \times 1 2 8 0$ </td><td>939M</td><td>ViT-L</td><td>45.8</td><td>-</td><td>-</td><td>X X</td><td>X</td><td>X</td></tr><tr><td>Painter</td><td>448 × 448</td><td>303.5M</td><td>ViT-L</td><td>43.4</td><td>-</td><td>-</td><td>X X</td><td>-</td><td>72.1</td></tr><tr><td>GiT†</td><td>11202|6722</td><td>756M</td><td>Multi-layer Transformer</td><td>X</td><td>52.4</td><td>X</td><td>71.0</td><td>52.9 X</td><td>X</td></tr><tr><td> $\mathrm { U n i f i e d - I O ^ { \dagger } \mathrm { _ { X L } } }$ </td><td>384 × 384</td><td>2.9B</td><td>ViT-L</td><td>X</td><td>56.5</td><td>0.385</td><td>X</td><td>X -</td><td>68.1</td></tr><tr><td> $\mathbf { A H M A D _ { \mathrm { s i n g l e } } }$ </td><td> $6 1 6 \times 6 1 6$ </td><td>309.5M</td><td>ViT-L</td><td>55.1</td><td>66.9</td><td>0.331</td><td>64.3</td><td>43.9 90.8</td><td>67.3</td></tr><tr><td> $\mathbf { A H M A D } _ { \mathrm { m u l t i } }$ </td><td> $6 1 6 \times 6 1 6$ </td><td>309.5M</td><td>ViT-L</td><td>53.1</td><td>66.5</td><td>0.310</td><td>62.9 44.6</td><td>92.2</td><td>68.2</td></tr></table>

Table 1. Comparison of multi-task models across various evaluation metrics on the COCO-val split. GiT performs object detection and instance segmentation at a resolution of 1120×1120, while semantic segmentation is conducted at 672×672. Additionally, Unified-IO is evaluated on the GRIT benchmark, which includes the COCO dataset. ”×” indicates that the model does not support the task, while ”-” means the task is supported, but no performance numbers for the COCO dataset are provided in the paper.

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1>#Projector Params</td><td rowspan=1 colspan=1> $\mathbf { O K S - m A P \uparrow }$ </td><td rowspan=1 colspan=1>Avg Run/Img</td><td rowspan=1 colspan=1>Decoder</td></tr><tr><td rowspan=1 colspan=1>Top-down KD</td><td rowspan=1 colspan=1> $C _ { 2 8 0 \times 2 8 0 }$ </td><td rowspan=1 colspan=1>440K</td><td rowspan=1 colspan=1>62.6</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>Bypassed KD</td><td rowspan=1 colspan=1> $C _ { 2 8 0 \times 2 8 0 }$ </td><td rowspan=1 colspan=1>2.5K</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>X</td></tr><tr><td rowspan=1 colspan=1>KnowDist KD</td><td rowspan=1 colspan=1> $G _ { 2 8 0 \times 2 8 0 }$ </td><td rowspan=1 colspan=1>2.5K</td><td rowspan=1 colspan=1>41.32</td><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>√</td></tr></table>

Table 2. A comparative analysis of the standard top-down Keypoint detection, the decoder-bypassed variant, and our proposed Knowledge Distillation approach. G denotes the global image input, while C represents the cropped image input. The last column shows a ✓ for methods that run the decoder, and × otherwise.

<table><tr><td colspan="3">Panoptic Segmentation</td></tr><tr><td>EDS</td><td> $\mathbf { \Pi } ( \mathbf { u } , \mathbf { v } )$  Encoding</td><td>PQ↑</td></tr><tr><td>X</td><td>2D Regression</td><td>38.01</td></tr><tr><td>×</td><td>PE L=4</td><td>44.86</td></tr><tr><td>√</td><td>2D Regression</td><td>41.61</td></tr><tr><td>√</td><td>Cross-Entropy</td><td>39.87</td></tr><tr><td>√</td><td>PE L=3</td><td>47.38</td></tr><tr><td>√</td><td> $\mathrm { P E L } { = } 4$ </td><td>48.61</td></tr><tr><td>V</td><td> $\mathrm { P E } \mathrm { L } { = } 5 $ </td><td>48.12</td></tr></table>

Table 3. Panoptic Segmentation Ablation. Effects of EDS and Positional Embedding (PE). In 2D Regression, we directly regressed (u, v). The ablation is performed on a ViT-Base model with a resolution of $4 2 0 \times 4 2 0$

<table><tr><td rowspan=1 colspan=1>Conf ID</td><td rowspan=1 colspan=1>Config</td><td rowspan=1 colspan=1>Loss Function</td><td rowspan=1 colspan=1> $\mathbf { O K S - m A P \uparrow }$ </td></tr><tr><td rowspan=1 colspan=1>CID-1</td><td rowspan=1 colspan=1>Stand-alone feature cropping</td><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=1>34.2</td></tr><tr><td rowspan=1 colspan=1>CID-2</td><td rowspan=1 colspan=1>CID-1, RGB concatenation</td><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=1>34.6</td></tr><tr><td rowspan=1 colspan=1>CID-3</td><td rowspan=1 colspan=1>Distillation, Shared Projector</td><td rowspan=1 colspan=1> $\mathcal { L } _ { 1 } ^ { * } + \mathcal { L } _ { 1 } ^ { d }$ </td><td rowspan=1 colspan=1>35.2</td></tr><tr><td rowspan=1 colspan=1>CID-4</td><td rowspan=1 colspan=1>CID-3</td><td rowspan=1 colspan=1> $\mathcal { L } _ { 1 } ^ { * } + \mathcal { L } _ { 1 } ^ { d } + \mathcal { L } _ { 2 } ^ { * }$ </td><td rowspan=1 colspan=1>35.6</td></tr><tr><td rowspan=1 colspan=1>CID-5</td><td rowspan=1 colspan=1>CID-4, Different Projectors</td><td rowspan=1 colspan=1> $\mathcal { L } _ { 1 } ^ { * } + \mathcal { L } _ { 1 } ^ { d } + \mathcal { L } _ { 2 } ^ { * }$ </td><td rowspan=1 colspan=1>36.2</td></tr><tr><td rowspan=1 colspan=1>CID-6</td><td rowspan=1 colspan=1>CID-5, TFU added</td><td rowspan=1 colspan=1> $\mathcal { L } _ { 1 } ^ { * } + \mathcal { L } _ { 1 } ^ { d } + \mathcal { L } _ { 2 } ^ { * }$ </td><td rowspan=1 colspan=1>39.8</td></tr><tr><td rowspan=1 colspan=1>CID-7</td><td rowspan=1 colspan=1>CID-6</td><td rowspan=1 colspan=1> $\mathcal { L } _ { 1 } ^ { * } + \mathcal { L } _ { 1 } ^ { d } + \mathcal { L } _ { 2 } ^ { * } + \mathcal { L } _ { 2 } ^ { d }$ </td><td rowspan=1 colspan=1>41.3</td></tr></table>

Table 4. $\mathbf { K D } ^ { * }$ Ablation. Comparison of different architectures and loss functions on OKS-mAP performance. For CID-1 and CID-2, no distillation is performed. In CID-2, the cropped image is directly concatenated with the feature map from the input. In CID-3, both the student and teacher utilize the same projector, $H _ { \psi _ { 5 } } ^ { 5 } .$ Conversely, in CID-5, the student and teacher employ different but architecturally identical projectors, $H _ { \psi _ { 5 } } ^ { 5 }$ and $\bar { H ^ { \prime } } _ { \bar { \psi } _ { 5 } } ^ { 5 }$ . Observe how our TFU function effectively enhances performance.

## 6. Conclusion

In this paper, we introduce AHMAD, a generalist multitasking framework that integrates five key vision tasks—semantic segmentation, instance segmentation, depth estimation, keypoint detection, and object detection—within a unified model. Our approach efficiently handles heterogeneous task outputs without complex serialization or pre/post-processing. To further improve efficiency, we propose a knowledge distillation strategy for keypoint detection, enabling a single forward pass instead of multiple passes required by traditional top-down methods. Experiments demonstrate strong performance across all tasks, with state-of-the-art results in panoptic and semantic segmentation. AHMAD offers a simple yet effective multitask framework, balancing efficiency and flexibility for realworld applications. Future work includes refining distillation techniques, incorporating higher-resolution training, and expanding to additional vision tasks.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1

[2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 8

[3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 2

[4] Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J Fleet, and Geoffrey E Hinton. A unified sequence interface for vision tasks. Advances in Neural Information Processing Systems, 35:31333–31346, 2022. 3

[5] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. arXiv preprint arXiv:2208.04202, 2022. 1

[6] Ting Chen, Lala Li, Saurabh Saxena, Geoffrey Hinton, and David J Fleet. A generalist framework for panoptic segmentation of images and videos. In Proceedings of the IEEE/CVF international conference on computer vision, pages 909– 919, 2023. 1, 3

[7] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. Advances in neural information processing systems, 34:17864–17875, 2021. 2

[8] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 2

[9] Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. Yolo-world: Real-time open-vocabulary object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16901–16911, 2024. 2

[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the

2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171– 4186, 2019. 2

[11] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3

[12] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary ob ject detection with vision-language model. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 14084–14093, 2022. 2

[13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Pro ceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 2

[14] Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 1440–1448, 2015. 8

[15] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921, 2021. 2

[16] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 1

[17] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable´ vision learners. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 2

[18] Yu Huang, Chenzhuang Du, Zihui Xue, Xuanyao Chen, Hang Zhao, and Longbo Huang. What makes multi-modal learning better than single (provably). Advances in Neural Information Processing Systems, 34:10944–10956, 2021. 2

[19] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2989–2998, 2023. 2

[20] Peiyuan Jiang, Daji Ergu, Fangyao Liu, Ying Cai, and Bo Ma. A review of yolo algorithm developments. Procedia computer science, 199:1066–1073, 2022. 5

[21] Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah. Transformers in vision: A survey. ACM computing surveys (CSUR), 54(10s):1–41, 2022. 2

[22] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international confer ence on computer vision, pages 4015–4026, 2023. 2

[23] Alexander Kolesnikov, Andre Susano Pinto, Lucas Beyer,´ Xiaohua Zhai, Jeremiah Harmsen, and Neil Houlsby. Uvim: A unified modeling approach for vision with learned guiding codes. Advances in Neural Information Processing Systems, 35:26295–26308, 2022. 8

[24] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022. 2

[25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730– 19742. PMLR, 2023. 2

[26] Shikun Liu, Edward Johns, and Andrew J Davison. Endto-end multi-task learning with attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1871–1880, 2019. 2

[27] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024. 2

[28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10012–10022, 2021. 2

[29] Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi. Unified-io: A unified model for vision, language, and multi-modal tasks. arXiv preprint arXiv:2206.08916, 2022. 2

[30] Jiasen Lu, Christopher Clark, Sangho Lee, Zichen Zhang, Savya Khosla, Ryan Marten, Derek Hoiem, and Aniruddha Kembhavi. Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26439–26455, 2024. 2

[31] Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, and Demetri Terzopoulos. Image segmentation using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(7):3523– 3542, 2021. 2

[32] Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, et al. Simple open-vocabulary object detection. In European conference on computer vision, pages 728–755. Springer, 2022. 2

[33] Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. Advances in Neural Information Processing Systems, 36:72983–73007, 2023. 2

[34] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 3

[35] Nedyalko Prisadnikov, Wouter Van Gansbeke, Danda Pan Paudel, and Luc Van Gool. A simple and generalist approach for panoptic segmentation. arXiv preprint arXiv:2408.16504, 2024. 4

[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2

[37] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-´ sion transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021. 2

[38] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland, Laura Gustafson, et al. Sam 2:¨ Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 2

[39] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779–788, 2016. 5

[40] Tianhe Ren, Yihao Chen, Qing Jiang, Zhaoyang Zeng, Yuda Xiong, Wenlong Liu, Zhengyu Ma, Junyi Shen, Yuan Gao, Xiaoke Jiang, et al. Dino-x: A unified vision model for open world object detection and understanding. arXiv preprint arXiv:2411.14347, 2024. 3

[41] Tianhe Ren, Qing Jiang, Shilong Liu, Zhaoyang Zeng, Wenlong Liu, Han Gao, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, Yihao Chen, et al. Grounding dino 1.5: Advance the” edge” of open-set object detection. arXiv preprint arXiv:2405.10300, 2024. 2

[42] Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv preprint arXiv:2401.14159, 2024. 2

[43] Sagar Soni, Akshay Dudhane, Hiyam Debary, Mustansar Fiaz, Muhammad Akhtar Munir, Muhammad Sohail Danish, Paolo Fraccaro, Campbell D Watson, Levente J Klein, Fahad Shahbaz Khan, et al. Earthdial: Turning multi-sensory earth observations to interactive dialogues. arXiv preprint arXiv:2412.15190, 2024. 2

[44] Trevor Standley, Amir Zamir, Dawn Chen, Leonidas Guibas, Jitendra Malik, and Silvio Savarese. Which tasks should be learned together in multi-task learning? In International conference on machine learning, pages 9120–9132. PMLR, 2020. 2

[45] Kim-Han Thung and Chong-Yaw Wee. A brief review on multi-task learning. Multimedia Tools and Applications, 77 (22):29705–29725, 2018. 2

[46] Wouter Van Gansbeke and Bert De Brabandere. A simple la tent diffusion approach for panoptic segmentation and mask

inpainting. In European Conference on Computer Vision, pages 78–97. Springer, 2024. 8

[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 2

[48] Haiyang Wang, Hao Tang, Li Jiang, Shaoshuai Shi, Muhammad Ferjad Naeem, Hongsheng Li, Bernt Schiele, and Liwei Wang. Git: Towards generalist vision transformer through universal language interface. 2024. 3

[49] Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng, Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, et al. Deep high-resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 43(10):3349– 3364, 2020. 8

[50] Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7794–7803, 2018. 8

[51] Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, and Tiejun Huang. Images speak in images: A generalist painter for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6830–6839, 2023. 1, 2

[52] Zhuofan Xia, Xuran Pan, Shiji Song, Li Erran Li, and Gao Huang. Vision transformer with deformable attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4794–4803, 2022. 2

[53] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024. 7

[54] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2025. 7

[55] Alireza Zareian, Kevin Dela Rosa, Derek Hao Hu, and Shih-Fu Chang. Open-vocabulary object detection using captions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14393–14402, 2021. 2

[56] Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2

[57] Yu Zhang and Qiang Yang. An overview of multi-task learning. National Science Review, 5(1):30–43, 2018. 2

[58] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1724–1732, 2024. 2