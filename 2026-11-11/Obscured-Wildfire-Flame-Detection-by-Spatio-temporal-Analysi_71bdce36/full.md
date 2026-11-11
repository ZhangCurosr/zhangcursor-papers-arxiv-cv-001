# Obscured Wildfire Flame Detection by Spatio-temporal Analysis of Smoke Patterns Using Frame-wise Transformers

1<sup>st</sup> Uma Meleti

2<sup>nd</sup> Abolfazl Razi

School Of Computing

3<sup>rd</sup> Fatemeh Afghah

School Of Computing

Clemson University)

Clemson University

Clemson, USA

Department of Electrical Engineering

Clemson, USA

umeleti@clemson.edu

arazi@clemson.edu

Clemson University

Clemson, USA

fafghah@clemson.edu

Abstract—In this paper, we propose a new paradigm in wildfire management for detecting both visible and obscured fire using Deep Learning (DL). The essence of our method is the joint spatial and temporal analysis of RGB videos, embracing the fact that temporal patterns of smoke motions in consecutive video frames can indicate obscured fire flames. This perspective contrasts the traditional approach of independently analyzing images and individual video frames. To this end, we propose a frame-wise transformer architecture that applies an attention mechanism between the consecutive frames while preserving spatial information within frames as opposed to regular attention mechanisms like the non-local block, which combines spatial and temporal information into a single vector. We also optimize the temporal pattern lengths under different spatial resolutions to achieve the highest performance while reducing the computational cost. To investigate the applicability of our method in aerial image processing, we applied it to a curated version of the FLAME2 dataset that includes dual-mode RGB/IR videos. The IR video feed is used to extract ground truth, while the model is trained only on RGB videos to maintain applicability to lowcost commercial drones (which are equipped only with regular cameras) and facilitate processing existing RGB-only datasets. Our proposed method achieves a Segmentation Foreground Dice score of 92.61%, an Object Detection Precision and Recall rate of 93.21% and 91.73%, and clip-level classification accuracy of 98.02%, respectively. We also investigated the efficacy and generalizability of our method through ablation study <sup>1</sup>.

Index Terms—Wildfire Monitoring, Obscured Fire Detection, Unmanned Aerial Vehicles, Temporal Video Analysis

## I. INTRODUCTION

Wildfires have become prevalent and destructive in many parts of the world. Regardless of the cause, wildfires can have severe consequences, including the loss of human lives, destruction of property, disruption of wildlife, food production, and crop supply chain as well as significant environmental damage. Once a wildfire starts, there are various ways to monitor and control these wildfires, including observation towers, direct human intervention, satellite imaging, and manned aircraft. Using drones is one of the most efficient ways of fire monitoring due to its low operation cost, customizable sensing and imaging, flexible operation, and ease of use in harsh environments by advanced flight control features and partial autonomy (e.g., safe auto-landing).

One of the most significant advantages of using drones in wildfire fighting is the ability of gathering real-time data about the fire’s behavior. Equipped with cameras and other sensors, drones can fly over the fire and capture relevant information. Drones can also be equipped with water tanks or fire extinguishers containing fire-retardant chemicals, such as carbon dioxide (CO2), potassium bicarbonate (KHCO3), or evaporating like bromochlorodifluoromethane (CF2ClBr) to be dropped on hot spots to create fire breaks. The efficient utilization of drones will significantly advance controlling these fires.

Targeting places where actual burning is happening, instead of blindly spraying fire retardant gases everywhere, will help set these fires off quickly and efficiently. Finding out these burning places in real-time is very difficult, especially when fire flames are obscured by thick smoke. An infrared camera can help find these hidden fires, but IR cameras are expensive when utilized on a large scale.

We devised a methodology that uses an RGB camera feed, analyzes the video frames sequentially, and detects the obscured fires using temporal features, such as smoke patterns. We show that such features can be indicative of the fire’s exact location and temporal behavior. We formulated the problem as a semantic segmentation of the obscured fire by analyzing the sequence of frames in the video.

We propose a deep learning architecture, ‘Frame-Wise Transformer,’ that utilizes a pre-trained CNN backbone to extract the features from video frames and passes through a convolutional transformer that aggregates the video features and sends them to a decoder, which will predict the segmentation map of the fire flames. We have implemented a convolutional transformer mimicking the NLP transformer, treating the video frame features as word embedding. Utilizing the popular non-local block [1] for attention mechanism did not yield reasonable results. Therefore, we re-formulated the attention mechanism to restrict attention only between frames while not mixing within the frame information. We modified the nonlocal block accordingly and achieved a significant performance gain. To balance the spatial resolution and temporal depth, we examined different frame rates while aggregating features at different resolutions. We observed that using more frames at lower resolutions provides higher performance while reducing the computation cost compared to using equal depth at all resolutions. The overall design can be extended to other videobased deep-learning tasks by making minor modifications to accommodate task-specific requirements.

To this end, we curated a dataset from the existing FLAME2 [2] dataset by selecting part of video frames from the original data where the drone is stationary, and there is high synchronization with the IR images to avoid misalignment errors. We also use the corresponding IR videos to extract ground truth for our task by performing a series of image-processing operations. We have visually verified the correspondence of RGB video with the processed IR video for frame-level synchronization.

The rest of the paper is organized as follows. We highlight the unique features of our method compared to previous methods in Section 2, then proceed by elaborating on the details of the generated dataset and its preparation method in Sections 3 and 4.1. We then elucidate the details of the proposed deep learning (DL) architecture in Section Section 4.2. The obtained results are investigated in Section 5, followed by concluding remarks in Section 6.

## II. RELATED WORK

The previous works on fire detection are mainly based on image-based techniques such as classification, object detection, and semantic segmentation of visible fire imagery. Wonjae Lee et al. proposed a wildfire detection system that classifies the presence of fire in images and evaluated the performance of AlexNet, GoogLeNet, and VGG, along with their modified variants [3]. Zhentian et al. have trained YOLOV3 for fire detection and reported a recognition rate of 91% [4]. An ensemble-based object detection method using YOLOV5 EfficientDet is proposed in [5] for detection and EfficientNet to capture global information about the fire. Their study showed a decrease in false positive rate by 51.3% on public datasets. Yo Zhao et al. have proposed a deep learning architecture, called Fire-Net by stacking convolutional and pooling layers for fast localization and segmentation of fire in aerial images with an accuracy of 98 % on standard ’UAV Fire’ dataset [6]. A similar method is proposed in [7] and [8] by using a deep learning architecture to classify frames into ”fire with smoke”, ”fire with no smoke”, and ”no fire” using FLAME datasets [2], [9]. However, most of these methods are limited to images, whereas real-time data for fire detection will be mostly in terms of video feed. Further, they ignore the fact that video feeds contain temporal patterns of smoke that can help locate the origin of smoke or equivalently the fire spots. Temporal analysis of such patterns, especially in the presence of clouds and other white patterns, can be extremely useful for precise fire detection. This concept is used as a key idea in our method to detect obscured fire positions.

Some works take a slightly different approach and analyze fire image patch by patch instead of one-shot analysis of the entire image. For instance, a CNN-based deep learning method is proposed in [10] which classifies the image first and then performs a patch-level analysis to offer more detailed information. They applied their method to video frames and performed patch-wise detection. They reported a 97 % detection accuracy on their own dataset. Still, this method does not consider the temporal relationships for fire classification since it treats the video frame as a sequence of independent still images. In a similar work, Gwangsu Kim et al [11]. proposed an algorithm that collects features of video frames using a pretrained VGG and stacks them together to be passed through a series of fully connected layers to classify the presence of fire in the video clips. However, this method which restricts fire classification only during a visible fire deemed inefficient in capturing obscured fire.

Anshuman et al. [12] have proposed SmokeyNet - a deep learning algorithm that offers stacking CNN, LSTM, and Vision Transformer to detect smoke in the video feed captured by stand-alone cameras. However, this method is not directly applicable to aerial imagery. Some other research works take advantage of Infra-Red (IR) cameras for more accurate fire positioning. For instance, Chi Yuan et al. proposed an algorithm that uses brightness and motion clues with histogram-based segmentation and optical flow to segment fire in IR images [13]. Another example is Norsuzial et al.’s work that offers an Image processing-based approach to convert IR images to YCbCr color and use a wavelet analyzer to detect fire [14]. Also, a DL architecture is proposed in [15] that analyzes dualfeed imagery captured by side-by-side RGB and IR cameras for precise fire positioning.

Although these methods yield high accuracy taking advantage of thermal information captured by IR cameras, they incur an extra monitoring cost for their reliance on pricey IR cameras. Also, they are not suitable for processing existing drone-based and satellite-based datasets that include only RGB imagery. In contrast to these works, our method uses only RGB videos for detecting both visible and obscured fire flames in an economical way.

## III. DATA PREPARATION

We investigated different fire datasets and found out that FLAME2 dataset [2] provides the most suitable testing condition because it has videos consisting of visible and obscured fire regions with smoke. Also, it offers a dual mode RGB/IR imagery that simplifies evaluating the proposed framework.

FLAME2 consists of seven pairs of lengthy RGB and IR video feeds taken by seven drones equipped with side-by-side RGB/IR cameras. The videos were taken from a prescribed fire region in Kaibab National forest in Northern Arizona. For experimentation purposes, we have carefully cropped the parts of the video where the camera is relatively stationary and there is a high alignment between the RGB and IR camera viewpoints, so IR stands as a legitimate ground truth. The selected video segments are split into clips of 20 consecutive frames to train our deep neural network, where each clip is considered a training sample. In total, we have used 839 video samples for training(594) and evaluating(255) our method. These video samples consist of a diverse set of positive samples consisting of visible and obscured fire regions shot at different locations and negative samples where fire is absent, which are included to make the model robust.

![](images/2c51d022ef00247ffe38ba4c7145c2b9e77f1e38254bb33148ebd63bb5b985cb.jpg)  
Fig. 1: Proposed Deep Learning architecture - Frame Wise Transformer.

## IV. METHOD

## A. Data Pre-Processing: IR to RGB Labels

The IR images consist of heat maps corresponding to the temperature of different regions on the image. The place where the fire is present generally has a higher temperature, and the pixel values in the heat map are closer to the maximum. We have extracted the ground truth for training data where the fire is present by processing the IR image with a series of handcrafted image processing methods. The sequence of operations performed on the IR image is presented below.

IR Image → Smooth Image (5 × 5 AvgPool) → Hard Thresholding → Dilation (5×5, 2 times) → Fill (flood fill) → Erosion (5 × 5, 1 time) → Remove small objects (200 pixel) → Ground Truth.

We initially smooth out the image and reduce the noise level using a low-pass filter, then use hard thresholding to select the regions of high-temperature values corresponding to fire. The resulting image is dilated to fill in the small spaces and make the fire boundary smooth; the spaces not filled in the previous operations are filled using flood fill, resulting in a complete blob of fire location. The image is eroded to compensate for the effect of dilation applied earlier. Finally, the small blobs that are likely to be representative of noise are removed. The extracted ground truth can been seen in Fig. 2

Note that we use IR images to identify ground truth fire locations when training the model, but in runtime (new monitoring tasks), we only use RGB images, so our method does not require expensive IR cameras on site. We treat each video clip with 20 frames as a sample and use the obtained groundtruth binary fire map from the last IR frame as our reference for the entire clip.

![](images/bc371bdc9c5dc1df5108ecd2a597ae88ffe463d0e2f650d94b8dd9c712e086c5.jpg)  
Fig. 2: Obtaining Ground truth fire locations from IR images.

## B. Deep Learning Architecture

The proposed deep Learning architecture ‘Frame-wise Transformer’ comprises three stages, as shown in Fig. 1. The first stage involves extracting features from the individual frames, which is done using a pre-trained encoder. Here, we used VGG-16 [16]. The next stage focuses on aggregating the sequential features of the video frames with a convolutional transformer; we have considered utilizing features at different stages of the encoder as followed in standard semantic segmentation architectures like U-net [17]. Furthermore, we have examined different frame rates at different resolutions for aggregation to obtain the most efficient architecture to exploit temporal relations. The later stage consists of a decoding network with a series of convolutional and upsampling blocks that combine information at different resolutions to obtain the segmentation map.

![](images/d7f41dac7c3e9f4216f9cd64e2bd40e84daa72f28d062d232ede3fb57d62d918.jpg)  
Fig. 3: Comparision of Non-local vs Frame-wise attention

## C. Convolutional Transformer

We have formulated a convolutional transformer inspired by the transformer [18] in Natural Language Processing by treating each frame of the video as a word and feature map from the encoder as word embedding and feature maps of a series of frames as a series of word embeddings. Multiple feature maps from the encoder can be equivalent to having multiple heads. These heads can be further scaled to increase the expressiveness of the features by multiplying the input channels with n heads while projecting the input into key, query, and value maps. Then, a proposed Frame-wise attention is applied to find the relations between the frames and finally fed to a convolutional block. The overall block diagram is shown in Fig. 1(b).

We have employed residual connections and normalization around each of the sublayers, and we have chosen batch normalization instead of layer normalization, which is usually preferred in CNNs. Positional embeddings were added to the extracted video features before the convolutional projection.

1) Convolutional Projection: The input(x<sub>i</sub>) from the encoder is projected into key $( x _ { k } ) .$ , query $( x _ { q } )$ , and value $( x _ { v } )$ using a conv3d [19], with a kernel size of 1 at the time axis, which is equivalent to convolving feature maps only in spatial dimension as we want to separate information between space and time, where the correlations between the frames are handled by attention in subsequent operations. The projection is shown in Eq. (1) to have convolution only in the spatial dimension of the features and time dimension will be handled by the subsequent attention operation.

$$
\begin{array} { r } { x _ { i }  \mathrm { c o n v 3 d } ( 1 \times k \times k , \mathrm { i n \_ c , o u t \_ c } ) _ { ( k , q , v ) }  x _ { k } , x _ { q } , x _ { v } , } \end{array}\tag{1}
$$

where k is the kernel size, in c is the number of input channels, and out $\underline { { { \mathrm { c } } } } = n _ { \ast }$ heads × in c is the output size.

2) SelfAttention: The scaled dot product provides attention to features across the frames. Initially, we tried the attention computation using the non-local block published in [1], which models space-time attention by flattening all the features into one single vector and performing the dot product and subsequent matrix multiplication, as shown in Eq. (2).

$$
\begin{array} { r l } & { \mathrm { N o n - l o c a l ~ A t t e n t i o n } ( Q , K , V ) = } \\ & { \quad \quad \quad \mathrm { s o f t m a x } \left( \frac { Q _ { ( C \times T H W ) } \cdot K _ { ( C \times T H W ) } ^ { T } } { \sqrt { d _ { k } } } \right) \cdot V _ { ( C \times T H W ) } } \end{array}\tag{2}
$$

where $C$ is the number of input channels, $T$ is the sequence length (number of consecutive frames), and $H , W$ is the dimension (height, width) of the feature space.

However, this method mixes up the information in space and time causing ambiguity in distinguishing information between space and time. In the NLP universe, one can imagine this operation as finding attention within the word embedding while finding attention with other words. Therefore, we have modified the computation by providing attention between the frames but not within the frame, which finds relations between the frames while separating within the frame information. The proposed attention mechanism and its distinction with nonlocal attention are presented in Fig. 3. The modified attention mechanism can be expressed as

$$
\begin{array} { r l } & { \mathrm { F r a m e - W i s e ~ A t t e n t i o n } ( Q , K , V ) = } \\ & { \mathrm { s o f t m a x } \left( \frac { Q _ { ( C \times T \times H W ) } \cdot K _ { ( C \times T \times H W ) } ^ { T } } { \sqrt { d _ { k } } } \right) \cdot V _ { ( C \times T \times H W ) } } \end{array}\tag{3}
$$

![](images/4ea7702aca7aed6538bf796a87d34745ec3e5b63ef2a04d297009ac7834b6e57.jpg)  
A. IR

![](images/f89182411d42aec31d134e6e620062cc5695846d9393956547c2d92a4edea38e.jpg)  
B. Ground Truth

C. Non-local Attn. (Low Res.)  
![](images/0d637593b92bcd066cf263fd88ffe9eb3e3e7fe4677edb38261d6cf9d0a917f2.jpg)  
D. Frame-Wise Attn. (High Res.)

![](images/9aa893beaec245bb22047c17279ac800f9a72983854a967e5a576c0b5e6edd31.jpg)  
E. Frame-Wise Attn. (Low Res.)  
Fig. 4: Top row is a few frames of sample video. Bottom row, A is the IR Image from where ground truth is extracted. The Green Regions correspond to Ground Truth and the Red are predictions.

3) No-feature Concatenation: Post the attention operation, unlike the NLP transformer, where the output of multiple heads is concatenated and averaged out, we have used a convolutional block to reduce the number of multi-headed selfattention feature maps to the original input number.

4) Convolutional Feed Forward: Similar to using a fully connected feed-forward network in the transformer, we have employed a convolutional feed-forward network, with a series of convolution, batch norm, and relu operations.

$$
\begin{array} { r l r } { \mathrm { c o n v \underline { { \Gamma } } f e e d \underline { { \ f } } f o r w a r d : } = \mathrm { c o n v 3 d } ( 1 \times k \times k , \mathrm { i n \underline { { \Gamma } } c , h i d d \underline { { \Gamma } } c } ) \to } & { } & \\ { \mathrm { R e L U } ( ) \to \mathrm { B a t c h N o r m } ( \mathrm { h i d d \underline { { \Gamma } } c } ) \to } & { } & \\ { \mathrm { c o n v 3 d } ( 1 \times k \times k , \mathrm { h i d d \underline { { \Gamma } } c , i n \_ c } ) \to \mathrm { R e L U } ( ) } & { ( 4 \pi ^ { 2 } \mathrm { i n \overline { { \Gamma } } } c , \mathrm { i n \overline { { \Gamma } } } c ) \to } \end{array}\tag{}
$$

where k is the kernel size, in c is the number of input channels, and $h i d d \_ c$ is the number of hidden channels

## D. Frame Rate Variation

We have developed a technique for varying frame rates based on principles of natural eyesight physics. Essentially, when objects are far away, they appear smaller, and the entire scene is visible, whereas when objects are closer, only parts of the scene are visible. Leveraging this concept, we have explored two theories. The first is to use all the frame features of low resolution from the encoder and lower the use of the number of frame features in the previous higher resolutions at a chosen frame rate. The second theory inverts the first theory by using more frames at higher resolution and fewer frames at lower resolution. We got the best results using the first theory, which aligns with our eyesight-based principle, with a significant margin in computational gain compared to the second theory. The results are discussed in Section 5.

## E. Loss Function

We have used Dice loss [20] to measure the alignment between the ground truth and the detected fire regions by the architecture to train the network. The Dice coefficient measures the alignment (similarity) between two corresponding segments by computing the overlap coefficient. The overlap coefficient ranges from 0 to 1, with 1 indicating a perfect match between the two segments. The Dice loss function is defined as one minus the Dice coefficient, with the objective of minimizing the loss during training. More specifically, we have

$$
\begin{array} { r } { D i c e L o s s = 1 - \frac { 2 \times \mathrm { I n t e r s e c t i o n } } { \mathrm { P r e d i c t e d } + \mathrm { G r o u n d ~ T r u t h } } } \\ { = 1 - \frac { 2 \sum _ { i = 1 } ^ { n } p _ { i } g _ { i } + \epsilon } { \sum _ { i = 1 } ^ { n } p _ { i } ^ { 2 } + \sum _ { i = 1 } ^ { n } g _ { i } ^ { 2 } + \epsilon } , } \end{array}\tag{5}
$$

where $p _ { i }$ and $g _ { i }$ are the predicted and ground truth segmentation masks, respectively, for the i-th pixel in the image. The summations are taken over all n pixels in the image. The ϵ term is a small constant added to the denominator to prevent division by zero.

## F. Inference

The inference in real-time, where videos are lengthy, is made by taking frames at a window size of 20 and sliding the window over the video at a stride of one as shown in Fig. 5.

## V. EXPERIMENTS

In this section, we present the simulation results using the FLAME2 dataset, which consists of RGB and IR images; we have trained with pre-processed videos as explained in the Data Preparation step. We have used 594 videos for training and 255 videos for testing. Out of these, 354 of training and 155 of testing are positive examples, consisting of visible and obscure fire regions, and the rest are negative videos with no fire locations. All the training and testing videos are independent and include no overlapping frames.

<table><tr><td rowspan="2">Threshold</td><td colspan="2">Speed</td><td colspan="3">F1 Score</td><td colspan="3">FG Dice</td><td>CL-Accuracy</td></tr><tr><td>Frame-Wise</td><td>Frame-Wise</td><td>Non-Local</td><td>Frame-Wise</td><td>Frame-Wise</td><td>Non-Local</td><td>Frame-Wise</td><td>Frame-Wise</td><td>Frame-Wise</td></tr><tr><td>0.1</td><td>High Res. 0.48 X</td><td>Low Res. 0.27 X</td><td>Low Res. 83.30</td><td>High Res. 90.37</td><td>Low Res. 91.50</td><td>Low Res. 90.18</td><td>High Res. 92.55</td><td>Low Res. 92.56</td><td>Low Res.</td></tr><tr><td>0.2</td><td>0.50 X</td><td>0.28 X</td><td>86.06</td><td>91.66</td><td>92.40</td><td>90.82</td><td>92.29</td><td>92.69</td><td>98.81 97.62</td></tr><tr><td>0.3</td><td>0.53 X</td><td>0.28 X</td><td>85.98</td><td>90.87</td><td>92.46</td><td>90.53</td><td></td><td>92.62</td><td></td></tr><tr><td>0.4</td><td>0.58 X</td><td></td><td>84.22</td><td>89.91</td><td>91.07</td><td>90.61</td><td>92.66 92.46</td><td>92.65</td><td>98.02</td></tr><tr><td></td><td>0.62 X</td><td>0.32 X 0.35 X</td><td>82.97</td><td>90.82</td><td>91.39</td><td>90.53</td><td>92.45</td><td>92.78</td><td>98.02</td></tr><tr><td>0.5 0.6</td><td>0.69 X</td><td>0.43 X</td><td>80.85</td><td>90.58</td><td>91.31</td><td>88.39</td><td>92.80</td><td>92.76</td><td>98.41</td></tr><tr><td>0.7</td><td>0.75 X</td><td>0.51 X</td><td>81.83</td><td>89.85</td><td>90.07</td><td>89.21</td><td>92.73</td><td>92.79</td><td>97.63</td></tr><tr><td>0.8</td><td>0.83 X</td><td>0.65 X</td><td>81.92</td><td>88.34</td><td>89.24</td><td>90.20</td><td>92.62</td><td>92.64</td><td>96.84 97.23</td></tr><tr><td>0.9</td><td>0.92 X</td><td>0.82 X</td><td>81.10</td><td>88.52</td><td>88.91</td><td>89.32</td><td>92.37</td><td>92.00</td><td>99.60</td></tr><tr><td>1.0</td><td>1.00 X</td><td>1.00 X</td><td>79.63</td><td>90.87</td><td>90.87</td><td>87.72</td><td>92.15</td><td>92.15</td><td>98.81</td></tr></table>

TABLE I: Comparision of Speed and Performance between taking more frames at high resolution (High Res.) Vs. More frames at low res (Low Res.). And Performance comparison between Non-local and Proposed Frame-Wise attention. All experiments are done using a 3x3 convolutional kernel size. X represents the number of flops performed per sample. CL-Accuracy - Clip Level classification accuracy.

![](images/3d3b106852d5bc4b35180a54d426bafffa68087dfef026e6518eaffb0efbead8.jpg)  
Fig. 5: Inference method: each output map results from sequential processing of 20 preceding input frames.

## A. Training

Our model is implemented in PyTorch lightning [17] using a single node Linux machine with 2 Tesla A-100 40 GB GPUs. The models were tuned for the best hyperparameters. We used a step-learning rate with an initial learning rate of 2e-4 with Adam Optimizer [21]. The models were trained for 150 epochs with early stopping and a batch size of 2.

## B. Evaluation Metrics

We are using a set of metrics to evaluate the quality of fire detection. We used the Fore Ground Dice score (presented in Eq. (5)) to assess the alignment quality between ground truth and the detected fire region. This assessment is particularly important when part of the fire is obscured by thick smoke to demonstrate how capable our model is in detecting such fire regions.

Another metric we used is blob-wise object detection F1 score Eq. (7), which is the harmonic mean of precision and recall Eq. (6) to ensure the correctness of our predictions; it is calculated by taking each blob in the ground truth and prediction, and if the intersection area is greater than 50%, considering it as a True Positive, else a False Positive.

<table><tr><td>Threshold</td><td>F1-Score Non-Local</td><td>F1-Score Frame-Wise</td></tr><tr><td>0.1</td><td>85.57</td><td>88.16</td></tr><tr><td>0.3</td><td>85.26</td><td>90.26</td></tr><tr><td>0.5</td><td>86.30</td><td>90.96</td></tr><tr><td>0.7</td><td>83.37</td><td>89.21</td></tr><tr><td>0.9</td><td>80.67</td><td>84.72</td></tr></table>

TABLE II: Non-local Vs. Frame-Wise using 1x1 kernel

$$
{ \mathrm { P r e c i s i o n } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F P } } } } , \quad { \mathrm { R e c a l l } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F N } } } }\tag{6}
$$

$$
\mathrm { F 1 ~ S c o r e } = { \frac { 2 \times \mathrm { P r e c i s i o n } \times \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } } }\tag{7}
$$

where TP is True Positive, FP is False Positive and FN is False Negative.

Also, we calculate clip-level classification accuracy to obtain clip-level labels, Fire and No-Fire. This is calculated by counting the number of fire spots in the video and comparing it with the prediction; if more than 50% of spots are predicted, we will classify the video as Fire; otherwise, No-Fire.

## C. Quantitative Results

We have experimented with our architecture from two perspectives. First, we investigate the architecture with the Non-Local attention method versus the proposed Frame-Wise attention method. Next, we study the impact of using various depths at different resolutions by comparing High Res. versus Low Res. at different thresholds.

1) Thresholds: The thresholds are the values that indicate the percentage of frames used in the subsequent resolution starting with a use of 100% frames. For example, in Low Res., if the value is $\alpha = 0 . 9 $ , at resolution (H x W)/16 which is the lowest, 100% of the frames will be used, and in the subsequent resolution (HxW)/8, 90% of the frames (the most recent ones) will be used and again in the next resolution 90% of the remaining frames in the previous resolution will be used $( \alpha ^ { 2 } = 0 . 9 \times 0 . 9 )$ and so on. Therefore, the number of frames at each layer follows exponential decay, layer by layer.

Likewise, for High Res. 100% of the frames will be used at the highest resolution, (HxW)/2, and the number of frames decreases, with the rate of $\alpha = 0 . 9 $ , for the lower resolutions.

2) Non-Local Vs. Frame-Wise Attention.: We have presented the F1 score and Fore Ground Dice coefficients for both methods in Table I. For the results in this table, we have used a 3x3 kernel for convolution in the conv-transformer. The proposed method has outperformed the non-local method. The best F1-Score using Frame-Wise at the threshold 0.3 has more than 7% margin than using Non-Local method The F1-Score graph in Fig. 6 clearly shows the difference.

![](images/109daa93fd63f6c2e895b8e1875f8149eb35d451e62823134c5b4497c9556088.jpg)  
Fig. 6: F1 Score Non-Local Vs Proposed Frame-Wise Method

![](images/f2c96eaac91886958e99e1cce73a2cb9b4d92080b31cc283b7c82f34d38336a7.jpg)  
Fig. 7: Precision - Non-Local Vs Proposed Frame-Wise Method

We have also experimented with a 1x1 kernel with few thresholds to make an exact match with the kernel size used in the non-local block original paper. Similar to the 3x3 kernel, the Frame-Wise method performed better with a margin of around 5 % at the 0.3 threshold.

3) High Res. Vs. Low-Res and Computational Gain: We also have displayed results of using more frames at low resolutions vs. using more frames at high resolution in Table I and Fig. 9. It can be seen that using more frames at low resolution leads to better results, which verifies the logic behind our justification in Section 4. Also, the best results were obtained at a threshold value of $\alpha = 0 . 3$ for the Frame-Wise attention, where the computational cost is reduced by 72 times compared to using all frames. From the speed column in Table I, we can infer that Low Res. takes fewer flops compared to High Res. at any threshold.

![](images/f0d74e5dd3db97d20f4439e3e9ff683e6250a9a3b011349b085c7dac14ef2249.jpg)  
Fig. 8: Recall - Non-Local Vs Proposed Frame-Wise Method

![](images/9d3b2414cbaa1c0a24881c99c584073193c09407d40a0e1a8d25e72ccc5059d2.jpg)  
Fig. 9: High Res. Vs Low Res.

## D. Qualitative Results

The qualitative results are displayed in Fig. 4. The top row shows a few frames of a sample input video. In the bottom row, A is the IR Image image of a sample frame, and B is the ground truth extracted from it and displayed over the image. Image C is the result using non-local attention, which fails in detecting many blobs. D and E are the results of the proposed frame-wise attention method for High Res. (more frames at higher resolution) and Low Res (more frames at lower resolution) approaches. The High Res. misses some blobs compared to the best output for Low Res.

## VI. CONCLUSION

This work is devoted to detecting obscured fire flames by analyzing video feeds captured by drones. All existing works, to the best of our knowledge, solve fire detection through image-based segmentation and object-detection methods. This neglects the fact that temporal analysis of video frames can recover motion patterns of smoke that can be indicative of obscured fire flames. Therefore, existing work fails to recover fire flames obscured by smoke patterns. To address this matter, we developed a novel architecture based on the frame-wise convolutional transformer that uses between-frame attention to capture relationships between frames to help locate obscured fire flames. We investigated the trade-off between feature resolution and temporal depth and observed that it is advantageous to use a higher depth for lower resolution, and lower depth for higher resolution. This reflects the fact that long-term motions can be captured in low-resolution feature spaces. Applying our model to FLAME2 datasets resulted in almost full success in clip-level fire classification while achieving about Dice score of 92.6%, and precision and recall scores of 93.2% and 91.73% in detecting fire flames. Our ablation study shows that our proposed attention mechanism outperformed the well-known non-local block with more than 7% improvement in F1 score.

We examined the interplay between spatial resolution and temporal depth by experimenting with various frame rates and resolutions. Our findings indicate that employing more frames at lower resolutions offers enhanced performance and computational efficiency compared to uniform depth allocation across resolutions. This versatile framework can be tailored for other video-based deep-learning tasks with slight modifications to meet specific task requirements.

To support our research, we curated a dataset from the FLAME2 dataset, carefully selecting video frames where the drone is stationary and synchronized with IR images to mitigate alignment discrepancies. Ground truth labels were generated using corresponding IR videos and image-processing operations. Through meticulous visual verification, we ensured frame-level synchronization between RGB and processed IR videos.

In conclusion, our methodology presents a robust solution for obscured fire detection using video analysis. The ’Frame-Wise Transformer’ architecture, informed by our tailored attention mechanisms and frame aggregation strategies, showcases its potential for video-based deep learning applications.

## REFERENCES

[1] X. Wang, R. B. Girshick, A. Gupta, and K. He, “Non-local neural networks,” CoRR, vol. abs/1711.07971, 2017.

[2] A. Shamsoshoara, F. Afghah, A. Razi, L. Zheng, P. Z. Fule, and´ E. Blasch, “Aerial imagery pile burn detection using deep learning: The flame dataset,” Computer Networks, vol. 193, p. 108001, 2021.

[3] W. Lee, S. Kim, Y.-T. Lee, H.-W. Lee, and M. Choi, “Deep neural networks for wild fire detection with unmanned aerial vehicle,” in 2017 IEEE International Conference on Consumer Electronics (ICCE), pp. 252–253, 2017.

[4] Z. Jiao, Y. Zhang, L. Mu, J. Xin, S. Jiao, H. Liu, and D. Liu, “A yolov3- based learning strategy for real-time uav-based forest fire detection,” in 2020 Chinese Control And Decision Conference (CCDC), pp. 4963– 4967, 2020.

[5] R. Xu, H. Lin, K. Lu, L. Cao, and Y. Liu, “A forest fire detection system based on ensemble learning,” Forests, vol. 12, no. 2, p. 217, 2021.

[6] Y. Zhao, J. Ma, X. Li, and J. Zhang, “Saliency detection and deep learning-based wildfire identification in uav imagery,” Sensors, vol. 18, no. 3, p. 712, 2018.

[7] L. Zhang, M. Wang, Y. Fu, and Y. Ding, “A forest fire recognition method using uav images based on transfer learning,” Forests, vol. 13, no. 7, p. 975, 2022.

[8] Y. Wang, C. Hua, W. Ding, and R. Wu, “Real-time detection of flame and smoke using an improved yolov4 network,” Signal, Image and Video Processing, vol. 16, no. 4, pp. 1109–1116, 2022.

[9] B. Hopkins, L. O’Neill, F. Afghah, A. Razi, E. Rowell, A. Watts, P. Fule, and J. Coen, “Flame 2: Fire detection and modeling: Aerial multi-spectral image dataset,” 2022.

[10] Q. Zhang, J. Xu, L. Xu, and H. Guo, “Deep convolutional neural networks for forest fire detection,” in 2016 International Forum on Management, Education and Information Technology Application, pp. 568–575, Atlantis Press, 2016.

[11] G. Kim, J. Kim, and S. Kim, “Fire detection using video images and temporal variations,” in 2019 International Conference on Artificial Intelligence in Information and Communication (ICAIIC), pp. 564–567, IEEE, 2019.

[12] A. Dewangan, Y. Pande, H.-W. Braun, F. Vernon, I. Perez, I. Altintas, G. W. Cottrell, and M. H. Nguyen, “Figlib & smokeynet: Dataset and deep learning model for real-time wildland fire smoke detection,” Remote Sensing, vol. 14, no. 4, p. 1007, 2022.

[13] C. Yuan, Z. Liu, and Y. Zhang, “Fire detection using infrared images for uav-based forest fire surveillance,” in 2017 International Conference on Unmanned Aircraft Systems (ICUAS), pp. 567–572, IEEE, 2017.

[14] N. Ya’acob, M. S. M. Najib, N. Tajudin, A. L. Yusof, and M. Kassim, “Image processing based forest fire detection using infrared camera,” in Journal of Physics: Conference Series, vol. 1768, p. 012014, IOP Publishing, 2021.

[15] X. Chen, B. Hopkins, H. Wang, L. O’Neill, F. Afghah, A. Razi, P. Fule, J. Coen, E. Rowell, and A. Watts, “Wildland fire detection and´ monitoring using a drone-collected rgb/ir image dataset,” IEEE Access, vol. 10, pp. 121301–121317, 2022.

[16] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” 2015.

[17] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” 2015.

[18] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” CoRR, vol. abs/1706.03762, 2017.

[19] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, “Learning spatiotemporal features with 3d convolutional networks,” in Proceedings of the IEEE international conference on computer vision, pp. 4489–4497, 2015.

[20] C. H. Sudre, W. Li, T. Vercauteren, S. Ourselin, and M. Jorge Cardoso, Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations, p. 240–248. Springer International Publishing, 2017.

[21] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.