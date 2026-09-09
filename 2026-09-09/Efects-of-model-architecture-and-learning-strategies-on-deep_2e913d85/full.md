# Efects of model architecture and learning strategies on deep learning-based recognition of activated sludge microscopic images and comparison with quantitative image analysis

Suguru Hakoshima<sup>1</sup>, Tomohiro Tobino\*<sup>1,2,3</sup>, and Fumiyuki Nakajima<sup>2</sup>

<sup>1</sup>Department of Urban Engineering, Graduate School of Engineering, The University of Tokyo, 7-3-1, Hongo, Bunkyo-ku, 113-8656, Tokyo, Japan

<sup>2</sup>Environmental Science Center, The University of Tokyo, 7-3-1, Hongo, Bunkyo-ku, 113-0033, Tokyo, Japan <sup>3</sup>Collaborative Research Institute for Innovative Microbiology, The University of Tokyo, 1-1-1, Yayoi, Bunkyo-ku, 113-8657, Tokyo, Japan

## Abstract

Microscopic image analysis has long been recognized as a promising approach for monitoring activated sludge. In recent years, deep learning-based image analysis has been increasingly adopted in this field because of its high performance. However, previous studies on microscopic image analysis of activated sludge have rarely explored transformer-based models or self-supervised foundation models and have instead relied on CNNs and supervised ImageNet pretraining. In addition, previous studies often downsampled image sizes, but the efects of downsampling have not been suficiently investigated, and the relationship between downsampling strategies and image analysis performance remains unclear. Furthermore, no study has quantitatively compared deep learning performance with quantitative image analysis (QIA), which was widely used before the emergence of deep learning. In this study, to examine how model architecture and learning strategies afect performance in microscopic image analysis of activated sludge and to quantitatively determine whether deep learning outperforms QIA, we prepared three types of activated sludge samples, classified their microscopic images, and evaluated classification accuracy. Our results showed that transformer-based architectures and alternative pretraining methods were efective in terms of classification accuracy. Our downsampling analysis showed that using overly small images reduced accuracy, but increasing image size beyond a certain point did not improve it further. In addition, the analysis indicated that, to achieve high classification accuracy, maintaining the field of view was a more efective downsampling strategy than maintaining resolution. Finally, our comparison between deep learning and QIA showed that deep learning outperformed QIA in terms of accuracy.

## 1 Introduction

Activated sludge is a biological wastewater treatment process widely used in municipal wastewater treatment plants (WWTPs). It plays a crucial role in preventing water pollution, protecting aquatic ecosystems, and supporting sustainable environmental management by removing organic matter and nutrients from wastewater. In recent years, image analysis-based artificial intelligence (AI) has achieved remarkable success [1–5]. In activated sludge systems, AI-driven image analysis has increasingly been explored for process condition monitoring and for the detection and prediction of operational problems. In particular, several studies [6–11] have applied deep learning techniques to microscopic images of activated sludge in order to improve process understanding and control.

Before the widespread application of deep learning to microscopic image analysis of activated sludge, quantitative image analysis (QIA) was the predominant analytical framework. [12]. In typical QIA-based approaches, microscopic images were first binarized, and floc structures were characterized using morphological parameters [13]. These morphological parameters were subsequently analyzed to investigate their relationships with various operational and performance indicators, including sludge settling performance [13–16], efluent water quality [17] and dewaterability [16]. Although QIA technologies provided powerful analytical tools, the binarization step was laborious and depended on semi-automated, operator-dependent procedures [6]. Consequently, convolutional neural networks (CNNs), a type of deep learning method, began to be applied to the microscopic image analysis of activated sludge to achieve automated image processing [6]. To date, CNN-based approaches in microscopic image analysis of activated sludge have been reported to enable the estimation of settling performance [7, 8] and MLSS [10], as well as the detection of protozoa [11, 18, 19].

However, previous studies that apply deep learning to the analysis of microscopic images of activated sludge still face several limitations. First, the architectural design and training strategies of deep learning models remain insuficiently investigated. Although several studies [6, 7] have compared diferent CNN architectures, transformer-based models [3] have not yet been systematically evaluated, despite their increasing prominence in recent years. Moreover, most prior work [6, 7, 10] has relied on supervised pre-training on ImageNet, whereas more recent strategies, such as self-supervised learning [5], have yet to be thoroughly applied. In addition, images are generally downsampled before model training because processing large images requires substantial memory [20]. However, few studies have investigated how image downsampling afects analytical performance, and the choice of resizing strategy has received little attention. Most studies resize images while maintaining the field of view, and no study has systematically compared this approach with downsampling that maintains the resolution. Comparisons between QIA and deep learning also remain limited. Previous study [6] has described QIA as a semi-automated approach with several limitations and have utilized CNNs as an alternative. Although we agree with this assessment, there has been little experimental evidence that deep learning outperforms QIA in the analysis of microscopic images of activated sludge, particularly in terms of analytical performance.

The objectives of this research were to examine, from the standpoint of analytical capabilities, how various training conditions, including model architecture, pre-training strategies, and microscopic image characteristics, influence deep learning performance, and to compare deep learning and QIA from the same perspective. We prepared three types of activated sludge samples and constructed a benchmark classification task in which microscopic images obtained from the samples were classified by sample type using deep learning and QIA. Here, we show that image size affects model performance, whereas model size has little efect. We also demonstrate that models pretrained using self-supervised learning and Vision Transformer (ViT) achieve suficient performance, that maintaining the field of view in microscopic images is more important than maintaining spatial resolution, and that deep learning outperforms QIA in terms of both analytical performance and the number of images required.

## 2 Materials and methods

## 2.1 Sample collection

Samples were collected from three process lines across two WWTPs in Japan. The collected samples were stored under refrigerated conditions until microscopic image acquisition. The detail of samples is shown in Table 1.

Table 1: Sample overview
<table><tr><td>Sample name</td><td>WWTPa</td><td>Treatment methodsb</td><td>Sampling datec</td><td>MLSS (mg/L)</td></tr><tr><td>Sample 1</td><td>A</td><td>AO</td><td>2024-Oct-9th</td><td>1,470</td></tr><tr><td>Sample 2</td><td>B</td><td>p-AO</td><td>2024-Nov-21st</td><td>1,300</td></tr><tr><td>Sample 3</td><td>B</td><td>Step A20</td><td>2024-Nov-21st</td><td>1,510</td></tr></table>

<sup>a</sup> All WWTPs operate under a combined sewer system  
<sup>b</sup> AO, Anaerobic-aerobic process; $p { \cdot } A O ,$ , pseudo anaerobic-aerobic process; Step  
A2O, step-feed anaerobic-anoxic-aerobic process  
<sup>c</sup> The images were acquired on November 28, 2024.

## 2.2 Benchmark classification task

In this study, we set up a benchmark classification task to systematically investigate how deep learning training conditions afect classification performance and to compare deep learning with QIA. In this task, microscopic images were classified by sample, and performance was assessed using classification accuracy (Eq. (1)) where TP, FP, TN, and FN represent true positives, false positives, true negatives, and false negatives, respectively. We first evaluated the efects of diferent model size, model architectures, pretraining methods, image sizes, and resizing strategies on the accuracy of deep learning. We then compared the accuracy achieved by deep learning with that achieved by QIA.

$$
\mathrm { A c c u r a c y } = { \frac { \mathrm { T P } + \mathrm { T N } } { \mathrm { T P } + \mathrm { T N } + \mathrm { F P } + \mathrm { F N } } }\tag{1}
$$

## 2.3 Deep learning

## 2.3.1 Image acquisition

A simple automated imaging system (Fig. 1) was constructed using a flow cell (Type 48, Starna), an inverted microscope (IX73, Evident), an automatic stage (B30102, CENTRAL MOTOR WHEEL CO., LTD), and a camera (NOA2000, Wraymer, 1824 px × 1216 px). A 10× objective lens was used. The imaging procedure was as follows: the sample was placed in a beaker, pumped into the flow cell mounted on the automatic stage using a peristaltic pump, allowed to settle for approximately 1 min, and imaged across multiple fields of view (25 fields of view, generally) by driving the automatic stage. In total, more than 300 images were acquired per sample.

![](images/0f6b3f80838d5c50d7ef0c00b8056ccc5282d6a293dc8e8c49bddcac8130fcf4.jpg)  
Fig. 1. Structre of a automated imaging system

## 2.3.2 Software and workstation

Python 3.11 was used as the programming language for the implementation of deep learning. The following packages were employed: PyTorch 2.7.1, torchvision 0.22.1, timm 1.0.19, and grad-cam 1.5.5. In addition, the oficial implementation of DINOv3 [5] provided by its authors was used. The prepared source code was executed on a workstation equipped with an AMD Ryzen Threadripper 7970X CPU (Advanced Micro Devices, Inc.) and an NVIDIA RTX 5090 GPU (NVIDIA Corporation). The virtual environment was constructed using Docker.

The obtained results were analyzed using R 4.4.2. The efsize package (version 0.8.1) was used to calculate efect sizes.

## 2.3.3 Models

In this study, we did not train models from scratch; instead, we retrained eight pretrained models (Table 2). To compare CNN and Transformer architectures, we employed ConvNeXt [4] and ViT [3, 21], respectively. ViT is a widely used Transformer architecture, whereas ConvNeXt is a modern CNN architecture designed using principles inspired by Vision Transformers, making it suitable for comparison with ViT [4]. We used the ViT implementation developed in a previous study [5]. This implementation supports adjustable input image sizes and incorporates register tokens [21], which were not included in the original ViT architecture [3]. We also compared diferent pretraining methods. Most previous studies on microscopic image analysis of activated sludge [6, 7, 10, 22] have used models pretrained through supervised learning on the ImageNet-1K [23] image classification dataset. In recent years, however, self-supervised learning has become an increasingly common approach to model pretraining [5, 24–26]. Therefore, we compared ConvNeXt models pretrained through supervised learning on ImageNet-1K with ConvNeXt models distilled [27] from the foundation model pretrained using self-supervised learning in the DINOv3 project [5]. Furthermore, to investigate the efect of model size, we employed multiple ConvNeXt and ViT variants with varying numbers of parameters.

Table 2: Eight deep learning models used in this study
<table><tr><td>Model name</td><td>Architecture</td><td>Pretraining method</td><td># Params (M)</td></tr><tr><td>ConvNeXt-Tiny_ImageNet</td><td rowspan="6">ConvNeXt [4] (CNN)</td><td>Supervised</td><td>29</td></tr><tr><td>ConvNeXt-Small_ImageNet</td><td>learning</td><td>50</td></tr><tr><td>ConvNeXt-Base_ImageNet</td><td>(ImageNet1k)</td><td>89</td></tr><tr><td>ConvNeXt-Tiny-DINOv3</td><td></td><td>29</td></tr><tr><td>ConvNeXt-Small_DINOv3</td><td>Self-supervised</td><td>50</td></tr><tr><td>ConvNeXt-Base_DINOv3</td><td>learning</td><td>89</td></tr><tr><td>ViT-Small_DINOv3</td><td>ViT [3, 21]</td><td>(DINOv3[5])</td><td>21</td></tr><tr><td>ViT-Base_DINOv3</td><td>(Transformer)</td><td></td><td>86</td></tr></table>

## 2.3.4 Training

The eight models were trained to classify our microscopic images by samples. Training was conducted under multiple conditions, resulting in a total of 48 trained models. For each sample, 201 – 207 images were used for training and 100 – 107 images were used for evaluation. Cross-entropy loss and the Adam optimizer were used in all 48 training runs. The learning-rate schedules are shown in Fig. S1.

To investigate the efects of training conditions on model performance, the models were trained under multiple conditions. Data augmentation was performed as follows. First, vertical and horizontal flips were independently applied with probabilities of 50%, followed by brightness adjustment. In accordance with a previous study [7], brightness was varied within a range of 20%. The augmented images were then resized using PyTorch’s RandomResizedCrop function. To evaluate the efect of image size, three input sizes were examined: $2 5 6 \times 2 5 6 , 6 7 2 \times 6 7 2$ , and 896 × 896 pixels. Following the normalization approach used by previous research [7], the images were normalized using channel-wise means of 0.485, 0.456, and 0.406 and standard deviations of 0.229, 0.224, and 0.225 for the red, green, and blue channels, respectively. For each image size, two training strategies were applied to investigate the efect of layer unfreezing. In the first strategy, only the fully connected layer (FC layer [28]) was unfrozen, whereas in the second strategy, all layers were unfrozen. Batch sizes of 8 and 4 were used for the former and latter strategies, respectively. Applying the two training strategies to each combination of eight models and three input image sizes resulted in a total of 48 trained models.

## 2.3.5 Evaluation

We evaluated the 48 models using two resizing strategies to determine whether maintaining the field of view or maintaining resolution resulted in higher classifcation accuracy. Under the former strategy, the central 1216 × 1216 pixel region of each test image was cropped and then resized to the input image size used for model training. The resized images were subsequently normalized using the same procedure as that applied during training. Under the latter strategy, the central region of each test image was cropped directly to the input image size used for model training. The cropped images were subsequently normalized using the same procedure as that applied during training. When examining the efects of factors other than the resizing strategy, we used the evaluation results obtained with the former strategy because maintaining the field of view is more commonly used and has also been adopted in previous study [20].

## 2.4 Quantitative image analysis

## 2.4.1 Image acquisition

Microscopic images for QIA were acquired using a diferent method from that used for deep learning because all flocs needed to be in focus for accurate calculation of morphological parameters. A 50 µL aliquot of the sample was placed onto a glass slide and covered with a 24 mm × 24 mm coverslip. The slide was examined using a microscope (SZX16, Olympus) at 10x magnification, equipped with a camera (NOA2000, Wraymer). From each slide, 50 images were captured.

## 2.4.2 Software

All image processing and morphological parameter calculations were performed in MATLAB 2024b. All other analyses were conducted in R 4.4.2, and machine learning models were implemented using the tidymodels 1.2.0, bonsai 0.4.0, and DALEX 2.4.3 packages.

## 2.4.3 Image binarization and calculation of morphological parameters

The binarization procedure is shown in Fig. 2. First, a background image was generated from the initial microscopic image using morphological opening and closing (Fig. 2, img background) [29]. This background image was subtracted from the initial image to produce a preprocessed image (Fig. 2, img dif). Next, the core and boundary regions of the flocs were binarized using diferent methods. For the boundary regions, eroded and dilated images were derived from the preprocessed image, and their diference was binarized using a fixed threshold (Fig. 2, img boundaries). For the core regions, adaptive histogram equalization (CLAHE) was applied to the preprocessed image to obtain an enhanced image (Fig. 2, img dif enhance), which was then binarized using a fixed threshold (Fig. 2, img core). Finally, the binarized images were combined and post-processed to produce the final binarized image (Fig. 2, img bin result).

![](images/5625745ff8ab0a4c20a736609099cb96966b9cd85d7ba41d2e2496bc8c731fda.jpg)  
Fig. 2. The binarization procedure for quantitative image analysis

After binarization, morphological parameters were calculated for each floc. The parameters included area (A), perimeter (P), form factor (Eq. (A.1)), convexity (Eq. (A.2)), solidity (Eq. (A.3)), aspect ratio (Eq. (A.4)), compactness (Eq. (A.5)), roundness (Eq. (A.6)), eccentricity [30], and extent [30]. In Eq. (A.2), $A _ { \mathrm { C o n v } }$ denotes the area of the convex hull, while in Eq. (A.3), $P _ { \mathrm { C o n v } }$ denotes its perimeter. These parameters were defined according to previous studies [13–15, 30] and were computed using the regionprops function in MATLAB.

## 2.4.4 Floc classification based on morphological parameters using machine learning

Flocs extracted from microscopic images were classified using LightGBM [31] based on morphological parameters. We used approximately 200 images per sample for training and 100 images per sample for evaluation. For each classification instance, n flocs $( 1 0 \leq n \leq 1 0 0 )$ , corresponding to 10n morphological parameter values, were randomly selected without replacement to account for variation in the number of flocs among images, and the selected flocs were used to classify the samples. For each value of $n ,$ 2000 classification instances per sample were generated for training, and

50 classification instances per sample were generated for evaluation. 20 % of the training data were used for validation, and training was stopped when the validation performance did not improve for 100 iterations. Each value of n was evaluated in three independent runs.

## 2.5 Frequency analysis of images

We conducted frequency analysis of the microscopic images to further investigate their image characteristics. Two microscopic images per sample were randomly selected from the deep learning training dataset, and their power spectra were calculated. Details of the implementation are provided in Text S1. In the implementation, Python 3.13.12 was used as the programming language. The following packages were employed: PyTorch 2.13.0 (CPU), torchvision 0.28.0, numpy 2.5.1, matplotlib 3.11.1, and opencv-python 5.0.0.93.

## 3 Results and discussion

## 3.1 Model architecture, model size, training strategy, image size

The performance of our 48 models are shown in Fig. 3. In Fig. 3a, the accuracy results are shown separately for each input image size and training strategy, with either all layers or only the FC layer unfrozen. Each panel shows results obtained using the ConvNeXt and ViT architectures, with multiple model sizes evaluated for each architecture. In all panels, the regression lines are nearly horizontal, with r ranging from -0.43 to 0.098 for all layers unfrozen and from -0.056 to 0.248 for the FC layer only unfrozen. These results indicate that the achieved accuracy was largely independent of both model architecture and model size (Fig. 3a). Meanwhile, the results show that the training strategy influenced model performance, with accuracy ranging from 0.98 to 1.00 when all layers were unfrozen and from 0.71 to 0.96 when only the FC layers were unfrozen (Fig. 3a). These results suggest that ViT are efective for microscopic image analysis of activated sludge, that increasing model size does not necessarily improve accuracy, and that unfreezing all layers yields better performance than unfreezing only the FC layer.

Our findings regarding model size and training strategy are consistent with previous studies [7, 28]. A previous study [7] that predicted sludge settleability from microscopic images reported better performance for ConvNeXt-Nano than for ConvNeXt-S. The lack of improvement in performance with increasing model size may be attributed to the limited amount of data available to adequately train larger models. A previous study [28] that detected sludge bulking from stained microscopic images reported higher performance when all layers were unfrozen than when only the FC layers were unfrozen. When only the FC layers are unfrozen, the feature extraction process remains fixed during training, whereas unfreezing all layers may allow the feature representations themselves to be further optimized. Because the models used in this study were pretrained on images distinct from microscopic images of activated sludge, such as those from ImageNet [23] or Instagram [5], training with all layers unfrozen may have enabled the models to learn feature representations more suitable for activated sludge images. In contrast, no previous studies have reported the efectiveness of ViT for microscopic image analysis of activated sludge, although more than five years have passed since its introduction [3] and several subsequent studies have applied deep

![](images/92bb224b44fa9fef786800b0b7f8e9a4da0c2eced644df59c80305631a009a94.jpg)

![](images/4341f0150881990e352d3b8fd4c59ff1167f9044d8e33a2bcee28fc197016f9f.jpg)  
Model\_Pretraining methods ConvNeXt-Base\_DINOv3 ConvNeXt-Base\_ImageNet ConvNeXt-Small\_DINOv3 ConvNeXt-Small\_ImageNet ConvNeXt-Tiny\_DINOv3 ConvNeXt-Tiny\_ImageNet ViT-Base\_DINOv3 ViT-Small\_DINOv3

Fig. 3. Accuracy of the 48 models evaluated in this study: (a) Separate panels are shown for each input image size and training strategy, with either all layers or only the FC layer unfrozen. The horizontal axis in each panel represents model size. The dashed lines indicate linear regression fits; (b) Separate panels are shown for the two training strategies, with either all layers or only the FC layer unfrozen. The horizontal axis in each panel represents the input image size.

learning to this field [7, 9, 10, 28]. There are two possible reasons why ViT achieved performance comparable to that of ConvNeXt in this study. First, we used pretrained models. In general, ViTs require more training data than CNNs [3], and high accuracy might not have been achieved if the models had been trained from scratch using only microscopic images of activated sludge. Second, as described in the Materials and Methods, we used ViTs equipped with register tokens [21], which were not included in the original ViT architecture [3]. According to the developers of register tokens [21], their introduction produces smoother feature maps by reducing artifacts. This efect may have facilitated efective transfer learning in the present study.

Fig. 3b shows the efect of image size. When all layers were unfrozen, accuracy remained high across all image sizes, with no clear efect of image size. In contrast, when only the FC layers were unfrozen, performance varied with image size. Specifically, accuracy difered significantly between image sizes of 256 and 672 px (paired t-test, $p = 0 . 0 0 0 1 3$ , Cohen’s $d = 1 . 8 )$ , whereas no significant diference was observed between 672 and 896 px (paired t-test, $p = 0 . 4 6$ , Cohen’s $d = - 0 . 2 3 )$ .

Because larger images contain more information, deep learning performance would be expected to improve with increasing image size. Consistent with this expectation, accuracy increased when the image size was increased from 256 to 672 px. This results suggests that a suficiently large image size is beneficial. Interestingly, however, increasing the image size further from 672 to 896 px did not result in a significant improvement in accuracy. Moreover, based on the efect size, performance tended to decrease rather than improve. These results suggest that, in deep learningbased microscopic image analysis of activated sludge, increasing image size improves accuracy up to a certain point, beyond which further increases provide little or no additional benefit. A previous study [22] used deep learning to classify microscopic images based on sludge settleability and compared several image sizes (224, 299, 448, and 512 px). The study showed that accuracy improved as image size increased, but that 448 px was suficient to achieve high accuracy. Their findings align with our results.

## 3.2 Pretraining methods

Fig. 4 shows the diference in accuracy acieved across pretraining methods. Although two pretraining methods were evaluated in this study, no significant diference in accuracy was observed between them (paired t-test, $p = 0 . 0 5 0 $ , Cohen’s $d = 0 . 2 4 )$ . This result suggests that pretraining with DINOv3 is efective for microscopic image analysis of activated sludge, as is supervised pretraining on ImageNet, which has been widely used in previous studies [6, 7, 10].

The increase in accuracy with increasing image size appeared to be somewhat more pronounced with DINOv3 pretraining than with ImageNet pretraining (Fig. 4). The original DINOv3 study [5] also reported improved performance with increasing image size, despite diferences in the tasks evaluated.

## 3.3 Field of view vs resolution

Fig. 5 compares the performance of our models when either the field of view or the resolution was maintained. We trained 48 models under each condition, for a total of 96 models. Maintaining the field of view resulted in higher accuracy than maintaining the resolution for most of the 48 model settings, except for several settings in which accuracy was already close to 1 (paired t-test, $p \ = \ 2 . 1 \times 1 0 ^ { - 8 }$ , Cohen’s $d = 0 . 7 0 )$ . These results suggest that, when preprocessing activated sludge microscopic images for deep learning-based analysis, maintaining the field of view is more important than maintaiing high image resolution.

![](images/4fc45e7ea1abeeec9437e58180fb53fc41a8818d6531d53346fe5e600e1b6dfe.jpg)  
Fig. 4. Diferences in accuracy achieved by our ConvNeXt models across pretraining methods.

![](images/7a6b88a2b8ce0d84fe912c58ba5fd596b341be5a1119b950635395d8edae9a96.jpg)  
Fig. 5. Comparison of accuracy between resizing strategies that maintain the field of view and resolution. Accuracy obtained from our 96 trained models is shown in separate panels for each image size and training strategy. Bar length represents accuracy. Blue bars indicate results obtained when the field of view was maintained, whereas yellow bars indicate results obtained when the resolution was maintained.

Several possible explanations can be considered for why maintaining the field of view was more efective than maintaining the resolution in improving accuracy. One possible explanation is that morphology played a more important role than texture in the benchmark classification. Maintaining resolution is likely advantageous for evaluating floc surface texture. However, reducing the field of view to maintain resolution makes it more dificult to capture the overall floc morphology. If this hypothesis is correct, applying activated sludge microscopic images to tasks where texture is more important than morphology (e.g., detection of protozoa and metazoa) could yield results opposite to those observed in this study. A second possible explanation is a mismatch between the field of view and floc size. Under the conditions of this study, maintaining the field of view yielded a side length of 747 µm, whereas reducing the field of view to maintain resolution at an image size of 256 px yielded a side length of only 158 µm. Although floc size distributions likely vary among the samples, Wilen and Balmer [32] reported that floc sizes followed a log-normal´ distribution ranging from 11.6 to 1128 µm. In addition, QIA in the present study detected flocs with an equivalent diameter of up to 721 µm (Fig. S2). Taken together, these observations imply that a field of view of 158 µm may have been too small. Another possible explanation is the reduction in information content. When the resolution is maintained while the field of view is reduced, both the representativeness and information content of the image decrease. In contrast, when the field of view is maintained while the resolution is reduced, the amount of information in the image may not decrease substantially. To examine this possibility, we conducted a frequency analysis, as shown in Fig. S3. We calculated the power spectra of our microscopic images and found that the image information was concentrated primarily in the low-frequency components. These results indicate that low-frequency components dominate microscopic images of activated sludge and that relatively little information is lost when the resolution is reduced. We assumed that recent improvements in image sensor quality and pixel count have exceeded the resolving capability of microscope objective lenses. Consequently, useful information in the images may be concentrated primarily in lower spatial frequencies, and the efect of reducing image resolution may therefore be limited, as in the case of empty magnification.

## 3.4 Comparison of deep learning and quantitative image analysis

We binarized the microscopic images acquired for QIA, extracted an average of 3.79 flocs per image, and calculated 10 morphological parameters for each extracted floc. For benchmark classification, the morphological parameters derived from n flocs were treated as a single instance. Fig. 6 shows the relationship between classification accuracy and the value of n. When more than 60 flocs were included in each instance, the classification accuracy exceeded 0.8, reaching a maximum of 0.873 at n = 90 (Fig. 6). However, from two perspectives, our results indicate that deep learning outperformed the QIA-based benchmark classification. First, deep learning achieved an accuracy above 0.9 (Fig. 3) and showed superior classification performance compared with the QIA-based approach. Second, deep learning was substantially more eficient in terms of the amount of image data required. Whereas deep learning treated a single image as one instance, the QIA-based approach required more than 60 flocs per instance to achieve an accuracy above 0.8. This requirement of more than 60 flocs corresponded to approximately 16 images.

## 3.5 Limitation and suggestion

## 3.5.1 Limitation of our research

This study has two limitations. The first limitation is the lack of discussion regarding interpretability. Throughout this study, we consistently used accuracy as the metric for evaluating the performance of deep learning and QIA. However, interpretability, which was not discussed in this study, is also an important consideration, particularly when comparing deep learning with QIA. In general, deep learning achieves outstanding performance, but it is often dificult to interpret how deep learning models process and evaluate images. Recently, several visualization methods have been developed [33–35] and applied to microscopic image analysis of activated sludge [7]. However, the resulting interpretations have generally been qualitative and subjective rather than quantitative and objective. In contrast, morphological parameters directly represent floc morphology and are therefore relatively easy to interpret. Indeed, in our previous study [36], we used QIA because our objective was to investigate the relationship between floc morphology and the microbial community. In the present study, our results indicate that deep learning outperformed the QIA-based benchmark classification, but this conclusion was based on accuracy and data requirements rather than interpretability.

![](images/e02d34a3c449c9777cf3c9de76314cfced510a78c2a825b28c68da84c7915e8e.jpg)  
Fig. 6. Relationship between the number of flocs per instance and classification accuracy in benchmark tasks using QIA-derived morphological parameters and machine learning. The error bars indicate the standard error.

The second limitation is the limited generalizability of our findings. In this study, we used a benchmark classification task in which three types of activated sludge samples were classified based on microscopic images. However, this task does not fully reflect practical applications, where the objective is often to predict sludge properties or process performance. Therefore, further evaluation is needed to determine whether the findings of this study can be generalized to such prediction tasks. We hypothesize that tasks in which global image features play a dominant role may show trends similar to those observed in this study, whereas tasks that depend primarily on local image features may yield diferent results. The classification task examined here was likely driven mainly by global features of activated sludge images. Similar trends may therefore be observed in tasks such as predicting sludge settleability, for which global image characteristics are also likely to be important. Indeed, the trends observed in our study for model size, image size, and training strategies were consistent with those reported in previous studies [7, 22, 28] (Fig. 3). In contrast, tasks such as identifying bacterial species based on floc texture may rely more strongly on local features, and the trends may therefore difer from those observed in this study.

## 3.5.2 Suggestion of our research

This study’s findings provide several useful insights for future research on microscopic image analysis of activated sludge. First, our results showed that increasing model size does not necessarily improve deep learning performance and that unfreezing all layers yields better performance than unfreezing only the FC layer. These trends have also been indicated in a previous study aimed at detecting sludge bulking [7, 28]. The similar trends observed in the present study, which addressed classification among diferent activated sludge samples, suggest that these findings may be broadly applicable to deep learning-based microscopic image analysis of activated sludge.

Second, we found that models pretrained using methods other than supervised learning on ImageNet achieved accuracy comparable to that of models pretrained using conventional supervised learning on ImageNet. We also found that ViTs performed comparably to CNNs. Previous studies on microscopic image analysis of activated sludge [6, 7, 10, 22] have generally relied on CNNs pretrained through supervised learning on ImageNet. In the broader field of deep learning-based image recognition, however, research has increasingly focused on developing general-purpose foundation models [37] using transformer architectures and self-supervised learning on large-scale datasets beyond ImageNet [5, 24, 25]. In light of these developments, we propose that environmental engineering research should also explore ViTs and alternative pretraining approaches to keep pace with advances in AI.

Third, across image size, resolution, and field of view, our results show that maintaining the field of view and using a suficiently large image size are important for deep learning-based microscopic image analysis of activated sludge. In contrast, maintaining the original resolution or using the maximum possible image size does not appear to be necessary. Reducing image size helps increase computational eficiency and supports model training when computational resources are limited. Therefore, these findings may provide practical guidance for future studies. An important diference between the present study and previous study [22] that reported similar trends is that we further investigated the possible reasons for these observations. Our frequency analysis showed that microscopic images of activated sludge contain predominantly low-spatial-frequency information. We suggest that this may be because the number of pixels provided by modern image sensors exceeds the resolving capability of the microscope objective lens. Although objective-lens resolving power is constrained by physical principles, image-sensor pixel counts are likely to continue increasing. Based on our findings, we therefore propose that further increases in image sensor pixel count alone are unlikely to substantially improve the performance of deep learning-based microscopic image analysis of activated sludge.

Finally, our study quantitatively showed that deep learning outperforms QIA in predictive performance. Previous studies [6, 7, 22, 28] have mainly justified using deep learning over QIA on qualitative grounds, such as the labor-intensive and time-consuming image preprocessing required for QIA. We agree that these are important advantages of deep learning. However, we believe a method’s ability to serve as a powerful image analysis tool matters more than whether it reduces labor or processing time. As discussed in the limitations, the present study did not evaluate model interpretability. Nevertheless, when high predictive performance is the primary objective, our results suggest preferring deep learning over QIA for microscopic image analysis of activated sludge.

## 4 Conclusion

This research examines how diferent training conditions afect deep learning performance in microscopic image analysis of activated sludge and compares deep learning with QIA. We prepared three types of activated sludge samples, classified their microscopic images, and evaluated classification accuracy. Our results showed that increasing model size did not necessarily improve performance and that transformer-based architectures and alternative pretraining methods were also efective for microscopic image analysis of activated sludge. We further examined how image size, field of view, and resolution afected accuracy. The results indicated that using images that were too small reduced performance, whereas increasing image size beyond a certain level did not lead to further improvement. In addition, maintaiing the field of view is more improtant than maintaining the resolution. We also found that the information in microscopic images of activated sludge was concentrated primarily in low-frequency components, which may explain these observations. This finding further suggests that continued increases in image sensor pixel counts alone may not substantially improve the performance of microscopic image analysis of activated sludge. Finally, we quantitatively showed that deep learning outperformed QIA in terms of accuracy. These results provide an additional rationale for applying deep learning to microscopic image analysis of activated sludge. Our findings ofer practical guidance for future studies.

## Data availability

Resarch data are available from the corresponding author upon request, with approval from the municipalities where the samples were collected.

## Acknowledgements

Part of this work was supported by JSPS KAKENHI Grant Number JP24KJ0636, JP25K22111, and JP26K24641.] Part of this work was supported by a collaborative research project by the University of Tokyo and KUBOTA Corporation. The authors thank Ms. Tomoko INOUE (technical staf) for assistance in experiments. The authors also acknowledge the Division of Creative Activity, Global Center in Engineering Education Institute for Innovation in International Engineering Education, Graduate School of Engineering, The University of Tokyo, for their support in the developments of research equipment.

Appendix A Definition of parameters

$$
\mathrm { F o r m ~ f a c t o r } = { \frac { 4 \pi A } { P ^ { 2 } } }\tag{A.1}
$$

$$
\mathrm { C o n v e x i t y } = { \frac { A } { A _ { \mathrm { C o n v } } } }\tag{A.2}
$$

$$
{ \mathrm { S o l i d i t y } } = { \frac { P _ { \mathrm { C o n v } } } { P } }\tag{A.3}
$$

$$
{ \mathrm { A s p e c t r a t i o } } = { \frac { \mathrm { M i n o r ~ a x i s ~ l e n g t h } } { \mathrm { M a j o r ~ a x i s ~ l e n g t h } } }\tag{A.4}
$$

$$
\mathrm { C o m p a c t n e s s } = { \frac { \sqrt { \frac { 4 A } { \pi } } } { \mathrm { M a j o r ~ a x i s ~ l e n g t h } } }\tag{A.5}
$$

$$
{ \mathrm { R o u n d n e s s } } = { \frac { 4 A } { \pi { \mathrm { M a j o r ~ a x i s ~ l e n g t h } } ^ { 2 } } }\tag{A.6}
$$

Appendix B Supplementary figures

![](images/a8bb2ae3294aa88b438bd491f44f1b7c6073f8b27831adc3ab237c7bfb6e26a0.jpg)

![](images/45645dd6f310da7382491184be13225831af96eb6ed0f2f979ad9ca35971c119.jpg)  
Fig. S1. Learning rate schedule over training epochs. We used diferent learning rates depending on whether all layers or only the FC layer were unfrozen.

![](images/da4ef9b30825852f8a888f99b286259e752a051dc5582b4d80b61009f595850d.jpg)  
Fig. S2. Floc size distribution.

![](images/28e5355175729065774247016bee98d13d0a79633ab8c6472e3dc737d0c03eb2.jpg)  
Fig. S3. Results of frequency analysis. The left, middle, and right panels show the original image, the image after center cropping and conversion to grayscale, and the power spectrum, respectively. The power spectrum is shifted so that the low-frequency components are located at the center.

## Appendix C Supplementary texts

Listing 1: The implementation of frequency analysis. The paths of the folder was erased.

```haskell
1 import os
2 import random
3 from numpy import z e r o s l i k e
4 import t o r c h
5 from t o r c h v i s i o n import t r a n s fo r m s as t r a n s fo r m s
6 import numpy as np
7 import m a t p l o t l i b . p y p l o t as p l t
8 import cv2
9
10 random . seed ( 1 2 3 )
11
12
13 c l a s s F o u r i e r T r a n s fo r m G r a y P o w e r :
14 d e f c a l l ( s e l f , img ) :
15 f = s e l f . F o u r i e r T r a n s f o r m ( img )
16 P o w e r S p e c t r u m = t o r c h . a b s ( f ) 2
17 r e t u r n PowerSpectrum
18
19 d e f F o u r i e r T r a n s fo r m ( s e l f , img ) :
20 f = t o r c h . f f t . f f t 2 ( img )
21 f s h i f t = t o r c h . f f t . f f t s h i f t ( f )
22 r e t u r n f s h i f t
23
24
25 def RGB2FLOAT( img ) :
26 r e t u r n np . f l o a t 6 4 ( img ) / 255
27
28
29 Power = F o u r i e r T r a n s f o r m G r a y P o w e r ( )
30 t r a n s f o r m p r e P r o c e s s i n g = t r a n s fo r m s . Compose (
31 [
32 t r a n s f o r m s . T o T e n s o r ( ) ,
33 t r a n s f o r m s . C e n t e r C r o p ( 1 2 1 6 ) ,
34 ]
35 )
36 t r a n s f o r m p o w e r = t r a n s f o r m s . Compose (
37 [
38 t r a n s f o r m p r e P r o c e s s i n g ,
39 Power ,
40 ]
```

41 )   
42   
43 d e f i m a g e r e a d ( p a t h ) :   
44 img = cv2 . i m r e a d ( p a t h )   
45 img = cv2 . c v t C o l o r ( img , cv2 . COLOR BGR2RGB)   
46 img rgb = RGB2FLOAT( img )   
47 img gray = cv2 . c v t C o l o r ( img , cv2 . COLOR RGB2GRAY)   
48 r e t u r n img rgb , i m g g r a y   
49   
50   
51 d e f i m a g e p r o c e s s i n g ( img rgb , i m g g r a y ) :   
52 i m g g r a y r e s i z e = t r a n s f o r m p r e P r o c e s s i n g ( i m g g r a y )   
53 i m g g r a y r e s i z e = i m g g r a y r e s i z e . t o ( ” c p u ” ) . d e t a c h ( ) . numpy ( ) .   
copy ( )   
54 i m g g r a y r e s i z e = np . s q u e e z e ( i m g g r a y r e s i z e , a x i s =0)   
55 i m g g r a y p o w e r = t r a n s f o r m p o w e r ( i m g g r a y )   
56 i m g g r a y p o w e r = i m g g r a y p o w e r . t o ( ” c p u ” ) . d e t a c h ( ) . numpy ( ) .   
copy ( )   
57 i m g g r a y p o w e r = np . s q u e e z e ( i m g g r a y p o w e r , a x i s =0)   
58   
59 f i g , a x e s = p l t . s u b p l o t s ( 1 , 3 , f i g s i z e = ( 1 5 , 5 ) )   
60 a x e s [ 0 ] . imshow ( i m g r g b )   
61 a x e s [ 1 ] . imshow ( i m g g r a y r e s i z e , cmap=” g r a y ” )   
62 im = a x e s [ 2 ] . imshow (   
63 i m g g r a y p o w e r ,   
64 cmap=” g r a y ” ,   
65 vmin =0 ,   
66 vmax=np . p e r c e n t i l e ( i m g g r a y p o w e r , 9 9 . 5 ) ,   
67 )   
68 f i g . c o l o r b a r ( im , a x= a x e s [ 2 ] )   
69   
70   
71 f o l d e r p a t h S 1 =   
72 f i l e s S 1 = o s . l i s t d i r ( f o l d e r p a t h S 1 )   
73 img S1 1 name = random . c h o i c e ( f i l e s S 1 )   
74 img S1 1 name = fo l d e r p a t h S 1 + ” / ” + img S1 1 name   
75 i m g S 1 1 r g b , i m g S 1 1 g r a y = i m a g e r e a d ( i m g S 1 1 n a m e )   
76 i m g S 1 2 n a m e = random . c h o i c e ( f i l e s S 1 )   
77 img S1 2 name = fo l d e r p a t h S 1 + ” / ” + img S1 2 name   
78 i m g S 1 2 r g b , i m g S 1 2 g r a y = i m a g e r e a d ( i m g S 1 2 n a m e )   
79   
80   
81 f o l d e r p a t h S 2 =   
82 f i l e s S 2 = o s . l i s t d i r ( f o l d e r p a t h S 2 )

```csv
83 img S2 1 name = random . c h o i c e ( f i l e s S 2 )
84 img S2 1 name = fo l d e r p a t h S 2 + ” / ” + img S2 1 name
85 img S2 1 rgb , i m g S 2 1 g r a y = i m a g e r e a d ( img S2 1 name )
86 img S2 2 name = random . c h o i c e ( f i l e s S 2 )
87 img S2 2 name = fo l d e r p a t h S 2 + ” / ” + img S2 2 name
88 i m g S 2 2 r g b , i m g S 2 2 g r a y = i m a g e r e a d ( i m g S 2 2 n a m e )
89
90 f o l d e r p a t h S 3 =
91 f i l e s S 3 = o s . l i s t d i r ( f o l d e r p a t h S 3 )
92 img S3 1 name = random . c h o i c e ( f i l e s S 3 )
93 img S3 1 name = f o l d e r p a t h S 3 + ” / ” + img S3 1 name
94 img S3 1 rgb , i m g S 3 1 g r a y = i m a g e r e a d ( img S3 1 name )
95 img S3 2 name = random . c h o i c e ( f i l e s S 3 )
96 img S3 2 name = f o l d e r p a t h S 3 + ” / ” + img S3 2 name
97 i m g S 3 2 r g b , i m g S 3 2 g r a y = i m a g e r e a d ( i m g S 3 2 n a m e )
98
99 i m a g e p r o c e s s i n g ( i m g S 1 1 r g b , i m g S 1 1 g r a y )
100 p l t . s a v e f i g ( ” i m g S 1 1 . p n g ” , d p i = 9 0 0 )
101 p l t . c l f ( )
102 i m a g e p r o c e s s i n g ( i m g S 1 2 r g b , i m g S 1 2 g r a y )
103 p l t . s a v e f i g ( ” i m g S 1 2 . p n g ” , d p i = 9 0 0 )
104 p l t . c l f ( )
105
106 i m a g e p r o c e s s i n g ( i m g S 2 1 r g b , i m g S 2 1 g r a y )
107 p l t . s a v e f i g ( ” i m g S 2 1 . p n g ” , d p i = 9 0 0 )
108 p l t . c l f ( )
109 i m a g e p r o c e s s i n g ( i m g S 2 2 r g b , i m g S 2 2 g r a y )
110 p l t . s a v e fi g ( ” img S2 2 . png ” , d p i =900)
111 p l t . c l f ( )
112
113 i m a g e p r o c e s s i n g ( i m g S 3 1 r g b , i m g S 3 1 g r a y )
114 p l t . s a v e f i g ( ” i m g S 3 1 . p n g ” , d p i = 9 0 0 )
115 p l t . c l f ( )
116 i m a g e p r o c e s s i n g ( i m g S 3 2 r g b , i m g S 3 2 g r a y )
117 p l t . s a v e f i g ( ” i m g S 3 2 . p n g ” , d p i = 9 0 0 )
118 p l t . c l f ( )
```

## References

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural networks. In Proceedings of the 25th International Conference on Neural Information Processing Systems - Volume 1, NIPS’12, pages 1097–1105, Red Hook, NY, USA. Curran Associates Inc., 2012. (Visited on 07/11/2021).

[2] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770– 778, 2016. (Visited on 07/19/2021).

[3] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929 [cs], 2021. arXiv: 2010.11929 [cs]. (Visited on 06/23/2021).

[4] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie. A ConvNet for the 2020s. arXiv:2201.03545 [cs], 2022. doi: 10.48550/arXiv.2201.03545. arXiv: 2201.03545 [cs]. (Visited on 08/29/2025).

[5] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose, V. Khalidov, M.´ Szafraniec, S. Yi, M. Ramamonjisoa, F. Massa, D. Haziza, L. Wehrstedt, J. Wang, T. Darcet, T. Moutakanni, L. Sentana, C. Roberts, A. Vedaldi, J. Tolan, J. Brandt, C. Couprie, J. Mairal, H. Jegou, P. Labatut, and P. Bojanowski. DINOv3.´ arXiv:2508.10104 [cs], 2025. doi: 10. 48550/arXiv.2508.10104. arXiv: 2508.10104 [cs]. (Visited on 09/20/2025).

[6] H. Satoh, Y. Kashimoto, N. Takahashi, and T. Tsujimura. Deep learning-based morphology classification of activated sludge flocs in wastewater treatment plants. Environmental Science: Water Research & Technology, 7(2):298–305, 2021. issn: 2053-1400, 2053-1419. doi: 10.1039/D0EW00908C. (Visited on 02/15/2021).

[7] S. Borzooei, L. Scabini, G. Miranda, S. Daneshgar, L. Deblieck, O. Bruno, P. De Langhe, B. De Baets, I. Nopens, and E. Torfs. Evaluation of activated sludge settling characteristics from microscopy images with deep convolutional neural networks and transfer learning. Journal of Water Process Engineering, 64:105692, 2024. issn: 2214-7144. doi: 10.1016/j.jwpe. 2024.105692. (Visited on 06/01/2025).

[8] U. Kaushalya, Y. Nakaya, S. Ishizaki, K. Sugino, R. Hirano, and H. Satoh. Quantification of morphological characteristics and filamentous bacteria in activated-sludge flocs through quantitative image-analysis techniques incorporating image-processing software and U-Net deep-learning framework. Journal of Water Process Engineering, 70:107053, 2025. issn: 2214-7144. doi: 10.1016/j.jwpe.2025.107053. (Visited on 03/31/2025).

[9] S. Bahr, P. Weigand, S. Gasparini, C. Treiber, F. Schulz, O. Wasenm ¨ uller, H. Suhr, and¨ P. Wiedemann. Introducing in situ activated sludge microscopy: A case study on reducing precipitant use. Journal of Water Process Engineering, 77:108513, 2025. issn: 2214-7144. doi: 10.1016/j.jwpe.2025.108513. (Visited on 01/26/2026).

[10] H. Li, Y. Tao, T. Xu, H. Wang, M. Yang, Y. Chen, and A. Wang. Real-time quantification of activated sludge concentration and viscosity through deep learning of microscopic images. Environmental Science and Ecotechnology, 24:100527, 2025. issn: 2666-4984. doi: 10.1016/j.ese.2025.100527. (Visited on 10/05/2025).

[11] S. Hakoshima, T. Tobino, and F. Nakajima. Development of a Comprehensive Detection and Autoclassification Model for Microfauna Species in Microscopy Images of Activated Sludge Using Deep Learning (in Japanese). Journal of Japan Society on Water Environment, 47(5):139–150, 2024. doi: 10.2965/jswe.47.139.

[12] D. P. Mesquita, A. L. Amaral, and E. C. Ferreira. Activated sludge characterization through microscopy: A review on quantitative image analysis and chemometric techniques. Analytica Chimica Acta, 802:14–28, 2013. issn: 0003-2670. doi: 10.1016/j.aca.2013.09.016. (Visited on 11/03/2020).

[13] K. Grijspeerdt and W. Verstraete. Image analysis to estimate the settleability and concentration of activated sludge. Water Research, 31(5):1126–1134, 1997. issn: 0043-1354. doi: 10.1016/S0043-1354(96)00350-8. (Visited on 06/23/2020).

[14] A. L. Amaral and E. C. Ferreira. Activated sludge monitoring of a wastewater treatment plant using image analysis and partial least squares regression. Analytica Chimica Acta. Papers Presented at the 9th International Conference on Chemometrics in Analytical Chemistry, 544(1):246–253, 2005. issn: 0003-2670. doi: 10.1016/j.aca.2004.12.061. (Visited on 06/20/2020).

[15] D. P. Mesquita, O. Dias, A. M. A. Dias, A. L. Amaral, and E. C. Ferreira. Correlation between sludge settling ability and image analysis information using partial least squares. Analytica Chimica Acta. Papers Presented at the 11th International Conference on Chemometrics in Analytical Chemistry, 642(1):94–101, 2009. issn: 0003-2670. doi: 10.1016/j.aca. 2009.03.023. (Visited on 06/19/2020).

[16] Y. Nakaya, J. Jia, and H. Satoh. Tracing morphological characteristics of activated sludge flocs by using a digital microscope and their efects on sludge dewatering and settling. Environmental Technology, 45(20):4042–4052, 2024. issn: 0959-3330. doi: 10.1080/09593330. 2023.2240026. (Visited on 01/12/2026).

[17] D. P. Mesquita, A. L. Amaral, and E. C. Ferreira. Estimation of efluent quality parameters from an activated sludge system using quantitative image analysis. Chemical Engineering Journal, 285:349–357, 2016. issn: 1385-8947. doi: 10 . 1016 / j . cej . 2015 . 09 . 110. (Visited on 06/19/2020).

[18] W. Lei, Z. Yikun, Z. Yu, L. Zihang, and L. Jie. Identification of Activated Sludge Microbial Based on Improving YOLOv8. IEEE Access, 12:152825–152838, 2024. issn: 2169-3536. doi: 10.1109/ACCESS.2024.3477967. (Visited on 08/05/2025).

[19] D. Liang, Y. Yao, M. Ye, Q. Luo, and J. Chu. Automatic visual detection of activated sludge microorganisms based on microscopic phase contrast image optimisation and deep learning. Journal of Microscopy, 298(1):58–73, 2025. issn: 1365-2818. doi: 10.1111/jmi.13385. (Visited on 08/05/2025).

[20] S. Al-Ani, H. Guo, S. Fyfe, Z. Long, S. Donnaz, and Y. Kim. Cross-resolution learning for scalable detection of filamentous bacteria in activated sludge. Journal of Environmental Chemical Engineering, 14(3):122948, 2026. issn: 2213-3437. doi: 10.1016/j.jece.2026. 122948. (Visited on 06/26/2026).

[21] T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski. Vision Transformers Need Registers. arXiv:2309.16588 [cs], 2023. doi: 10.48550/arXiv.2309.16588. arXiv: 2309.16588 [cs]. (Visited on 03/19/2024).

[22] J. Wang, X. Yang, W. Chen, Y. Zhao, S. Gong, D. Dong, J. Wang, and H. Ren. Prediction of Activated Sludge Sedimentation Performance Using Deep Transfer Learning. ACS ES&T Engineering, 4(6):1367–1377, 2024. doi: 10 . 1021 / acsestengg . 3c00631. (Visited on 07/01/2026).

[23] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

[24] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick. Momentum Contrast for Unsupervised Visual Representation Learning. arXiv:1911.05722 [cs], 2020. arXiv: 1911.05722 [cs]. (Visited on 11/01/2021).

[25] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A Simple Framework for Contrastive Learning of Visual Representations. arXiv:2002.05709 [cs, stat], 2020. arXiv: 2002.05709 [cs, stat]. (Visited on 10/04/2021).

[26] X. Chen and K. He. Exploring Simple Siamese Representation Learning. arXiv:2011.10566 [cs], 2020. arXiv: 2011.10566 [cs]. (Visited on 05/05/2022).

[27] G. Hinton, O. Vinyals, and J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531 [cs, stat], 2015. doi: 10.48550/arXiv.1503.02531. arXiv: 1503.02531 [cs, stat]. (Visited on 12/18/2022).

[28] Y. Gao, J. Li, B. Yang, D. Kang, J. Liu, Z. Ge, and L. Zhang. AI-Driven Early Warning of Sludge Bulking via Microscopic Image Analysis: Toward Sustainable Wastewater Treatment. ACS ES&T Water, 2025. doi: 10.1021/acsestwater.5c00953. (Visited on 01/26/2026).

[29] M. da Motta, M.-N. Pons, N. Roche, and H. Vivier. Characterisation of activated sludge by automated image analysis. Biochemical Engineering Journal, 9(3):165–173, 2001. issn: 1369-703X. doi: 10.1016/S1369-703X(01)00138-3. (Visited on 12/05/2020).

[30] M. B. Khan, H. Nisar, C. A. Ng, P. K. Lo, and V. V. Yap. Generalized classification modeling of activated sludge process based on microscopic image analysis. Environmental Technology, 39(1):24–34, 2018. issn: 0959-3330. doi: 10.1080/09593330.2017.1293166. (Visited on 07/03/2020).

[31] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu. LightGBM: A Highly Eficient Gradient Boosting Decision Tree. In Advances in Neural Information Processing Systems, volume 30, pages 3146–3154. Curran Associates, Inc., 2017. (Visited on 10/09/2025).

[32] B.-M. Wilen and P. Balm ´ er. The e ´ fect of dissolved oxygen concentration on the structure, size and size distribution of activated sludge flocs. Water Research, 33(2):391–400, 1999. issn: 00431354. doi: 10.1016/S0043-1354(98)00208-5. (Visited on 10/22/2020).

[33] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. In 2017 IEEE International Conference on Computer Vision (ICCV), pages 618–626, 2017. doi: 10.1109/ ICCV.2017.74. (Visited on 07/26/2024).

[34] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian. Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks. In 2018 IEEE Winter Conference on Applications ofComputer Vision (WACV), pages 839–847, 2018. doi: 10.1109/ WACV.2018.00097. arXiv: 1710.11063 [cs]. (Visited on 08/29/2025).

[35] M. B. Muhammad and M. Yeasin. Eigen-CAM: Class Activation Map using Principal Components. In 2020 International Joint Conference on Neural Networks (IJCNN), pages 1–7, 2020. doi: 10.1109/IJCNN48605.2020.9206626. arXiv: 2008.00299 [cs]. (Visited on 08/29/2025).

[36] S. Hakoshima, T. Tobino, and F. Nakajima. New insights into floc morphology and its associations with PAOs in activated sludge via quantitative image analysis. Journal of Water Process Engineering, 89:110307, 2026. doi: 10.1016/j.jwpe.2026.110307.

[37] C. Li, Z. Gan, Z. Yang, J. Yang, L. Li, L. Wang, and J. Gao. Multimodal Foundation Models: From Specialists to General-Purpose Assistants. arXiv:2309.10020 [cs], 2023. doi: 10. 48550/arXiv.2309.10020. arXiv: 2309.10020 [cs]. (Visited on 12/09/2023).