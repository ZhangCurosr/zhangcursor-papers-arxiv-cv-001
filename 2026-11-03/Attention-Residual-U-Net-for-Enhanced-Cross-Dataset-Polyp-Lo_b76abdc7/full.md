# Attention Residual U-Net for Enhanced Cross-Dataset Polyp Localization in Endoscopy

Romil Imtiaz

Department of Computer Science and Biomedical Informatics

University of Thessaly

Lamia, Greece

rimtiaz@uth.gr

Dimitris K. Iakovidis

Department of Computer Science and Biomedical Informatics

University of Thessaly

Lamia,Greece

diakovidis@uth.gr

Abstract— Precise localization of pathologies in endoscopy is a critical task in medical imaging, with deep learning (DL) frameworks now serving as the primary methodology for automated lesion detection. Today, generic object detection methods, such as YOLO, provide state-of-the-art performance in this context. However, their robustness often decreases under real-world clinical conditions due to variations across endoscopic devices, intestinal content, dataset bias, and the presence of small or low-contrast lesions. Motivated by these challenges, we propose a novel polyp detection framework based on a Attention Residual U-Net offering enhanced polyp localization. Instead of directly regressing bounding boxes from global feature representations, the proposed method learns lesion-focused spatial representations and converts them into bounding-box predictions during inference. Unlike conventional encoder–decoder localization frameworks that directly propagate all encoder activations into the decoding pathway, the proposed method selectively suppresses irrelevant background responses before feature reconstruction. This encourages the network to focus on lesion-relevant spatial evidence under heterogeneous imaging conditions. To achieve this, the framework integrates residual feature propagation, attentionguided encoder filtering, and multi-scale contextual learning. The proposed framework was evaluated on multiple public endoscopic datasets acquired using different devices. The comparative experimental results validate its effectiveness over the relevant state-of-the-art methods, with an improvement up to 22.0% in terms of the area under the receiving operating characteristic (AUC).

Keywords—Polyp detection, attention, residual learning, cross-dataset evaluation, deep Learning

## I. INTRODUCTION

Polyp localization in endoscopic imaging is an important problem in medical computer vision because colorectal polyps are considered precancerous lesions that may develop into colorectal cancer if not detected and removed at early stages. Accurate localization of abnormal tissue regions during endoscopic examination therefore plays a critical role in early diagnosis and clinical decision support. However, false negative detections may delay diagnosis and increase clinical risk, motivating the development of robust automated localization systems for challenging endoscopic imaging conditions.

Single stage detection models, such as You Only Look Once (YOLO), can be both efficient and accurate, e.g., YOLOv7 [1], YOLOv8, and YOLOv11 [2] models have been successfully applied to various object detection tasks in medical imaging, Autonomous driving, robotics, agriculture, surveillance, and manufacturing. However, these models are prone to errors in cases of noisiness, illumination variations, and complicated background in images. Endoscopic images usually have polyps of small sizes, unusual shapes, and color similar to background tissues. Thus, these objects are hard to identify with standard feature representations.

Recent polyp localization approaches have explored attention-guided feature refinement and multiscale contextual representation to improve lesion localization under challenging imaging conditions. Attention mechanisms can help suppress irrelevant background responses and emphasize lesion-related spatial regions [3], while multiscale representation learning improves the detection of polyps with \`varying sizes and [4]. However, despite the success of these models, the problem of distinguishing lesion from background is hard since lesions might appear anywhere within an image.

An important limitation in many existing object localization frameworks is the uncontrolled propagation of irrelevant feature activations during hierarchical feature extraction. In endoscopic imaging, background structures, such as mucosal folds, lumen boundaries, specular reflections, and illumination artifacts may generate strong activations that compete with lesion-related representations. When these irrelevant responses are directly propagated through encoderdecoder feature pathways, they may interfere with lesion localization and reduce robustness under heterogeneous imaging conditions. This limitation becomes particularly critical during cross-dataset evaluation, where variations in acquisition conditions and image appearance may further amplify unstable background activations.

Motivated by this challenge, this paper proposes a novel framework that explicitly suppresses irrelevant encoder responses before feature reconstruction to encourage lesionfocused spatial representation learning under challenging endoscopic conditions. It is based on DL architecture incorporating a detector that deals with the problem of irrelevant feature propagation and pays more attention to polyps. To reduce the impact of irrelevant background features and pay more attention to lesions, we developed a special attention guided module. Such approach allows us to filter out background information and obtain more relevant features for lesion detection. The proposed framework also incorporates residual feature learning to improve optimization stability and feature propagation, as well as multiscale contextual representation learning to better capture lesions with varying sizes and appearance. In addition to attention mechanism, the developed architecture uses residual learning and multiscale features extraction, which are particularly helpful in endoscopy detection tasks, since polyps are hard to detect due to various variations in their sizes, appearance, and conditions of acquisition. Furthermore, the proposed framework learns lesion-focused spatial representations that are transformed into bounding-box localization outputs during inference, enabling compatibility with detection-oriented evaluation protocols. A cross-dataset evaluation setting was used to assess and validate the advantages of the proposed framework.

The rest of this paper consists of four sections. Section II provides a review of relevant state-of-the-art methods, and section III describes the proposed framework and its methodological details. The results of the experimental evaluation are presented in section IV and the conclusions of this study are summarized in the last section.

## II. RELATED WORKS

Deep learning-based polyp detection has become a key area in computer-aided endoscopy, as it can support clinicians by automatically identifying suspicious lesion areas in colonoscopy frames. Existing detection approaches include two-stage detectors, single-stage detectors, transformer-based models, and region-aware encoder–decoder frameworks [5], [6]. Earlier object detection methods, such as Faster R-CNN introduced region proposal mechanisms for object localization [7], while single-stage detectors such as Single Shot MultiBox Detector (SSD) and YOLO improved detection efficiency by predicting object locations directly from feature maps [2], [8]. In endoscopic applications, YOLO-based models are widely used because of their speed and deployment simplicity, with YOLOv7 introducing a trainable bag-of-freebies strategy for real-time detection [2]. Recent YOLO-based frameworks, including YOLOv8 and YOLOv11, have also been used in medical detection tasks due to their practical deployment benefits [2]. For polyp detection specifically, Lalinia and Sahafi [9] applied YOLOv8 to colonoscopy images and demonstrated efficient localization performance on public polyp datasets.

Recent detection-oriented approaches have attempted to improve robust polyp localization through stronger multiscale feature fusion, contextual representation learning, and enhanced bounding-box regression strategies [10], [11]. These methods improve detection of polyps with varying sizes and appearances by enhancing feature aggregation and localization refinement mechanisms. Other approaches have incorporated contextual receptive field enhancement and refined multiscale pooling operations to improve contextual representation learning for gastrointestinal polyp localization [11]. Additional studies have further explored multiscale feature integration strategies to improve localization robustness under complex polyp appearance variations [12].

Despite these improvements, previous detection methods still show limitations in difficult endoscopic conditions. Existing approaches mainly improve detection heads, feature fusion, or multiscale representation, but they do not explicitly address the propagation of irrelevant background activations before localization. As a result, small or low-contrast polyps may be missed, and visually similar structures such as mucosal folds, specular reflections, bubbles, and lumen boundaries may contribute to false-positive detections. This limitation becomes more evident under cross-dataset evaluation, where changes in illumination, acquisition device, and tissue appearance can amplify unstable background responses. Motivated by these limitations, the proposed framework introduces attention-guided region-aware feature learning to suppress irrelevant encoder activations before localization.

Attention mechanisms have also been studied to reduce the influence of irrelevant background information. In endoscopic images, not all extracted features contribute meaningfully to detection. Lumen boundaries, tissue folds, and specular reflections can create strong activations that interfere with accurate lesion localization. Xie et al. proposed an attentional feature fusion strategy for gastric polyp detection, using attention to suppress background noise and emphasize useful target features [13]. Recent attention-based modifications to detection frameworks have shown that feature selectivity can be improved while maintaining computational efficiency in colorectal polyp detection [14]. These findings support the use of attention-guided feature refinement in proposed work, where the objective is to filter out irrelevant encoder features while preserving lesion-related information for detection.

Encoder–decoder frameworks have also been explored for polyp analysis because they can preserve spatial details while learning lesion-aware representations. Several studies have introduced attention-based refinement, multiscale contextual learning, and progressive region enhancement to improve polyp localization under complex colonoscopy conditions [15], [16], [17]. These works demonstrate the value of spatial feature refinement; however, detection-oriented localization remains challenging when irrelevant background activations are propagated before bounding-box generation. Despite successful results in various medical image analysis problems, direct implementation of U-Net architectures for the task of localization leads to certain limitations. First of all, conventional architectures of U-Nets [18] use the mechanism of direct skip connections, in which features from the encoder part are transferred into the decoder. Although such an approach helps to preserve spatial details, it also allows activations caused by non-relevant areas to pass through: specular reflections, lumen borders, folds of tissue, and noise. The problem arises since those areas can be confused with lesion regions during endoscopic imaging.

Secondly, the issue related to the receptive field of conventional convolutions limits the possibilities of using standard U-Nets in endoscopic localization. The problem is that standard convolutions use fixed-size spatial kernels; thus, there is no way to capture both local lesion texture and its surroundings in one network. Such a restriction becomes obvious when considering the diversity of the scale at which polyps are found in endoscopy. It means that small lesions require local texture representations while large lesions need broader contextual understanding to be localized.

Thirdy, optimization of deeper architectures tends to be problematic due to gradient vanishing. Deeper networks suffer from unstable learning and weak convergence and generalization performance. These restrictions encourage the use of residual learning, attention-based feature refinement, and multi-level contexts in the proposed architecture.

These limitations become even more critical under crossdataset evaluation settings, where variations in illumination conditions, acquisition devices, image quality, and tissue appearance may amplify irrelevant feature activations and reduce localization stability. When models rely heavily on dataset-specific background patterns, their performance may degrade significantly under unseen imaging conditions. The proposed framework attempts to address this limitation by suppressing irrelevant encoder activations before feature reconstruction and encouraging lesion-focused spatial representation learning. By reducing the influence of unstable background responses and integrating multiscale contextual representations, the proposed architecture is expected to improve robustness under heterogeneous endoscopic imaging conditions and cross-dataset domain variation.

Beyond attention, cross-dataset robustness has become an increasingly important issue. Models trained on one dataset often show reduced performance when evaluated on data from different institutions or acquisition environments. Ma et al. introduced the LDPolyp Video benchmark, a large-scale colonoscopy video dataset containing diverse polyp appearances, enabling evaluation under more realistic visual variability [19]. Recent studies on generalisability have also shown that model performance can differ considerably across datasets, highlighting the need for evaluation beyond a singlesource dataset [20]. Motivated by these findings, we evaluate the proposed framework under a source-to-target cross-dataset setting, where the model is trained on a source endoscopic dataset and tested on multiple unseen target datasets without fine-tuning, such as retraining the final layers, updating the full network weights, or adapting the model using samples from the target datasets before evaluation. Compared with direct YOLO-style detection, our method incorporates regionaware feature learning, attention-guided filtering, and multiscale contextual modeling to improve localization under noisy and heterogeneous imaging conditions, while still producing bounding-box outputs for detection evaluation.

## III. METHODOLOGY

This work proposes a detection-oriented localization framework for robust polyp detection in challenging endoscopic images. The architecture of the proposed framework is shown in Fig. 1. This network is composed of an encoder-decoder framework supplemented by residual feature learning, attention-based feature refining, and multiple context modeling mechanisms. Instead of regressing bounding boxes from global features in the manner typical for traditional object detection systems, the proposed model learns lesion-specific spatial activations and translates them to bounding box predictions during testing. In this way, we are able to retain fine details while suppressing any noise responses often seen in endoscopic images. Specifically, an encoder learns hierarchical visual features from the input image, while a decoder refines them into lesion-specific spatial activations utilizing attention-based skip connections.

To boost robustness in dealing with varying lesion sizes and appearances, a multi-scale bottleneck layer is inserted between the encoder and decoder modules. As a result, the final output of the system is a lesion confidence map from which bounding box predictions are generated via postprocessing. By combining selective feature propagation with multi-scale contextual learning, the proposed framework aims to improve localization reliability under noisy and heterogeneous imaging conditions.

## A. Residual Feature Learning

Aiming to enhance the robustness and feature flow through the optimization process, residual convolutional units are utilized in the design of the proposed architecture’s encoder, shown in Fig. 2. Rather than making the CNN learn a new transformation for every single layer, the residual learning concept enables the model to learn the residual mapping of the initial data:

$$
y = f ( x ) + x\tag{3}
$$

where � represents the input feature map and �(�) denotes the learned residual transformation. It helps improve gradient propagation and deep feature representation learning. With residual learning, there is a chance for efficient feature reuse by ensuring that low-level information is maintained through high-level feature refinement. In difficult endoscopic imagery, it is hard to learn the difference in texture between the lesion tissue and mucosa.

![](images/091d5e6a4e578e22f84b0d296fa96bbc0da5319371d499dd1838b4b07aa24906.jpg)  
Fig. 1. Overall architecture of the proposed detection-oriented Attention Residual U-Net with residual encoding, attention-guided feature refinement, and multi-scale contextual learning.

Without the aid of residual learning, more advanced feature extraction layers would likely have been affected by unstable optimization and poor convergence properties. This would compromise the capacity of the network to extract discriminatory lesion features, especially when dealing with imaging artifacts. With the inclusion of residual connections, the proposed model addresses this challenge.

![](images/0f47a776e94530df10f490a4ae2823857c533a6821f62c1660fb2f6134d6bfbd.jpg)  
Fig. 2. Attention gate mechanism used to suppress irrelevant encoder features before decoder feature fusion. The circled plus symbol denotes element-wise addition of transformed feature maps.

## B. Attention-Guided Feature Suppression

One of the key contributions of the proposed framework is the introduction of attention-guided feature suppression within the skip connection pathway. The attention mechanism used in the proposed architecture is illustrated in Fig. 3. Instead of directly forwarding all encoder features to the decoder, the proposed model selectively filters encoder activations before feature fusion.

Given encoder features � and decoder guidance features $^ { g , }$ the attention coefficient is computed as

$$
\alpha = \sigma \left( W _ { \psi } \left( \operatorname { R e L U } \left( W _ { x } x + W _ { g } g \right) \right) \right)\tag{4}
$$

where � denotes the encoder feature map transferred through the skip connection, � denotes the decoder guidance feature map, $W _ { x }$ and $W _ { g }$ are learnable linear transformations implemented using $1 \times 1$ convolutions, $W _ { \psi }$ is a learnable projection used to generate the attention response, ReLU(∙) is the nonlinear activation function, $\sigma ( \cdot )$ denotes the sigmoid function, and α represents the spatial attention coefficient. . The refined feature representation is then obtained as:

$$
x ^ { \prime } = \alpha \cdot x\tag{5}
$$

where $x ^ { \prime }$ denotes the attention-refined encoder feature map. This mechanism enables the decoder to guide the selection of relevant encoder information by suppressing activations that are unlikely to correspond to lesion regions. In endoscopic imagery, background structures such as folds, illumination artifacts, and specular highlights often produce strong feature responses that may negatively influence localization performance.

![](images/f4b5cc3bc3021684a285ec9041887fd68a4d36d3e3d2872e8cf8b8ec82b0f7fe.jpg)  
Fig. 3. Attention gate mechanism used to suppress irrelevant encoder features before decoder feature fusion.

Without attention-guided filtering, irrelevant activations are directly propagated into the decoder pathway, which can increase false positive predictions and reduce localization precision. By suppressing background interference before feature fusion, the proposed framework encourages the network to focus on lesion-relevant spatial patterns, improving robustness under visually complex imaging conditions.

## C. Multi-Scale Contextual Representation

To improve robustness against variations in lesion size and appearance, a multi-scale bottleneck module is introduced between the encoder and decoder stages, as shown in Fig. 4. The module applies parallel dilated convolutions using multiple receptive fields:

$$
F _ { r } = \mathsf { C o n } v _ { 3 \times 3 } ^ { ( r ) } ( x )\tag{6}
$$

where � represents the dilation rate and $F _ { r }$ denotes the feature map generated by the dilated convolution branch with dilation rate �. Thus $F _ { 1 } , F _ { 2 } ,$ and $F _ { 3 }$ represent feature maps produced using dilation rates 1, 2, and 3, respectively. The resulting feature maps are fused as:

$$
F = \phi ( [ F _ { 1 } , F _ { 2 } , F _ { 3 } ] )\tag{7}
$$

where [. ] denotes feature concatenation and $\phi$ represents feature fusion using convolutional projection.

This design allows the network to simultaneously capture fine local texture information and broader contextual representations. Small lesions benefit from local receptive fields, while larger lesions require wider contextual understanding. By integrating multiple receptive fields within the bottleneck representation, the proposed framework improves localization robustness across different lesion scales.

Without multi-scale contextual learning, the network may struggle to detect small lesions or incorrectly interpret surrounding anatomical structures. The proposed multi-scale module therefore improves the ability of the model to adapt to heterogeneous lesion appearance and spatial variability.

![](images/de4192c11f26d1f85520c232f5f7c1c6e44fac1ec44e7a1257212705e07098ef.jpg)  
Fig. 4. Multi-scale bottleneck module using parallel dilated convolutions for contextual feature extraction at multiple receptive fields. The circled plus symbol denotes feature-map fusion by addition before the final 1×1 convolution.

## D. Bounding Box Generation

Although the proposed framework learns spatial lesion representations, the final objective of the study remains detection-oriented localization. Therefore, the predicted confidence map is transformed into bounding-box predictions during inference. During inference, the predicted sigmoid probability map is resized to the original image resolution and threshold using a fixed value of 0.5 to generate binary lesion regions. Pixels satisfying $P ( i , j ) \geq 0 . { \bar { 5 } }$ are assigned to the lesion region, while remaining pixels are treated as background. The same threshold value was used for all validation, internal test, and external cross-dataset evaluations without dataset-specific tuning. he final bounding box is generated by identifying all positive pixels in the binary prediction map and computing the minimum axis-aligned rectangle enclosing the detected lesion region:

$$
B _ { k } = [ x _ { m i n } , y _ { m i n } , x _ { m a x } , y _ { m a x } ]\tag{8}
$$

where $x _ { m a x } , x _ { m i n }$ denote the minimum and maximum horizontal coordinates of positive pixels, while $y _ { m a x } , y _ { m i n }$ denote the corresponding vertical coordinates. If no positive pixels are detected, no bounding box is generated for that image. This post-processing stage creates a direct connection between region-aware feature learning and standard detection evaluation. By converting spatial predictions into bounding boxes, the proposed framework remains compatible with conventional detection metrics and enables direct comparison with YOLO-based detectors.

Without this conversion process, the framework would produce only spatial confidence responses, making comparison with bounding-box detection methods less consistent. The proposed bounding-box generation strategy therefore preserves the advantages of region-aware learning while maintaining compatibility with detection-oriented evaluation protocols.

## IV. RESULTS AND DISCUSSION

Sensitivity, specificity, area under the curve (AUC) [21], and F1 [22] score are metrics utilized for validating the proposed framework. The sensitivity [23] reflects the capability of the model for lesion detection, whereas the specificity [24] reflects the ability of the model to suppress false positives. In contrast, the F1 score shows the trade-off between the precision and recall rates. The AUC reflects the discriminative capability of the model. This metric is crucial for performing endoscopic lesion detection due to the more complicated conditions for lesions detection and high false positives.

To evaluate the polyp localization performance of the proposed framework, the LDPolyp [25] was used as the source dataset for model training and internal testing, while KUMC [26], EDD2020[27], and Kvasir-SEG[8] were used as external datasets for testing the trained system with different imaging conditions, lesion appearances, and data distributions. The details of the datasets are summarized in Table I. LDPolyp was selected as the source dataset because it provides a substantially larger number of annotated polyp frames compared with the external datasets, making it more suitable for training deep lesion localization models. None of the images of the external testing sets was used during training or model selection.

For comparison YOLOv7 and YOLOv11 were included, since they have been used by state-of-the-art polyp detection approaches (section II). This comparison enables to assess whether the proposed attention-guided and multi-scale localization framework provides benefits beyond standard YOLO-based detection pipelines.

Each image is associated with annotations represented in YOLO format, where each object is defined using the normalized center coordinates and dimensions of the bounding box. The annotation representation is defined as

$$
\boldsymbol { B } = ( c _ { x } , c _ { y } , w , h )\tag{1}
$$

where $( c _ { x } , c _ { y } )$ denote the center coordinates of the bounding box, while � ��� ℎ represent the normalized width and height of the lesion region.

All experiments were conducted using a frame-level partitioning strategy on the LDPolyp dataset. The proposed Attention Residual U-Net was trained using the Adam optimizer with an initial learning rate of $1 \times 1 { \bar { 0 } } ^ { - 4 }$ . The loss function combined binary cross-entropy and Dice loss. Input images were resized to 256×256 pixels, the batch size was set to 8, and training was performed for up to 80 epochs. Model selection was performed using the best validation performance, while early stopping monitored validation loss with a patience of 15 epochs. The learning rate was reduced by a factor of 0.5 when the validation loss did not improve for 5 epochs.

TABLE I. DATASETS USED IN THE ΕXPERIMENTS.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Usage</td><td rowspan=1 colspan=1>Purpose</td><td rowspan=1 colspan=1>PolypsImages</td><td rowspan=1 colspan=1>Normal</td></tr><tr><td rowspan=1 colspan=1>LDPolyp</td><td rowspan=1 colspan=1>ModelTraining/ReferenceTeststing</td><td rowspan=1 colspan=1>Modeltraining</td><td rowspan=1 colspan=1>33,884</td><td rowspan=1 colspan=1>6,382</td></tr><tr><td rowspan=1 colspan=1>KUMC</td><td rowspan=1 colspan=1>Externaltest set</td><td rowspan=1 colspan=1>Cross-datasetevaluation</td><td rowspan=1 colspan=1>103</td><td rowspan=1 colspan=1>30</td></tr><tr><td rowspan=1 colspan=1>Kvasir-SEG</td><td rowspan=1 colspan=1>Externaltest set</td><td rowspan=1 colspan=1>Cross-datasetevaluation</td><td rowspan=1 colspan=1>1000</td><td rowspan=1 colspan=1>35</td></tr><tr><td rowspan=1 colspan=1>EDD2020</td><td rowspan=1 colspan=1>Externaltest set</td><td rowspan=1 colspan=1>Cross-datasetevaluation</td><td rowspan=1 colspan=1>127</td><td rowspan=1 colspan=1>53</td></tr></table>

## A. ReferenceResults on LDPolyp Dataset

The focus of the experimental evaluation was on crossdataset assessments; however, for reference purposes we performed an additional 5-fold cross-validation experiment on non-overlapping subsets of the LDPolyp dataset. The dataset was partitioned at the frame level, with 20% of the samples used for validation and 10% for testing in each fold. The reported results correspond to the average performance obtained across all five folds. The proposed framework achieved the highest overall polyp detection performance among the compared methods, obtaining an AUC of 97.0% and an F1-score of 98.6%. In comparison, YOLOv7 achieved an AUC of 79.5% and YOLOv11 achieved an AUC of 78.4%. The proposed framework also produced higher specificity compared with both YOLO-based detectors, indicating improved suppression of irrelevant background activations.

The enhanced performance on the LDPolyp dataset suggests that the proposed architecture successfully learned discriminative lesion representations that generalize beyond the training samples. The attention-guided feature refinement mechanism appears to reduce the propagation of irrelevant encoder activations originating from specular reflections, tissue folds, and illumination artifacts. This selective feature filtering likely contributed to the reduction of false positive detections observed in the proposed framework.

In addition, the integration of residual feature learning improved optimization stability during training. The residual encoder blocks enabled more reliable gradient propagation across deeper layers, which improved feature representation learning under visually complex endoscopic conditions. The multi-scale contextual bottleneck further improved localization performance by allowing the network to simultaneously capture fine lesion textures and broader anatomical context.

TABLE II. RESULTS ON LDPOLYP DATASET.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Specificity</td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>F1-score</td></tr><tr><td rowspan=1 colspan=1>YOLOv7</td><td rowspan=1 colspan=1>91.9</td><td rowspan=1 colspan=1>67.0</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>91.5</td></tr><tr><td rowspan=1 colspan=1>YOLOv11</td><td rowspan=1 colspan=1>94.4</td><td rowspan=1 colspan=1>62.4</td><td rowspan=1 colspan=1>78.4</td><td rowspan=1 colspan=1>91.5</td></tr><tr><td rowspan=1 colspan=1>Proposed</td><td rowspan=1 colspan=1>98.5</td><td rowspan=1 colspan=1>95.4</td><td rowspan=1 colspan=1>97.0</td><td rowspan=1 colspan=1>98.6</td></tr></table>

## B. Cross-Dataset Evaluation on KUMC

Table III presents the cross-dataset evaluation results obtained on the KUMC dataset. All models experienced performance degradation compared with the LDPolyp evaluation results, which reflects the presence of domain variation between datasets. Differences in acquisition devices, illumination conditions, tissue appearance, and imaging protocols can substantially influence feature distribution and therefore reduce generalization performance.

TABLE III. POLYPS DETECTION (%) ON KUMC DATASET.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Specificity</td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>F1-score</td></tr><tr><td rowspan=1 colspan=1>YOLOv7</td><td rowspan=1 colspan=1>42.7</td><td rowspan=1 colspan=1>86.6</td><td rowspan=1 colspan=1>66.6</td><td rowspan=1 colspan=1>61.9</td></tr><tr><td rowspan=1 colspan=1>YOLOv11</td><td rowspan=1 colspan=1>46.6</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>68.0</td><td rowspan=1 colspan=1>59.1</td></tr><tr><td rowspan=1 colspan=1>Proposed</td><td rowspan=1 colspan=1>73.8</td><td rowspan=1 colspan=1>79.4</td><td rowspan=1 colspan=1>76.5</td><td rowspan=1 colspan=1>76.9</td></tr></table>

Despite this domain shift, the proposed framework maintained competitive localization performance and achieved stronger specificity compared with several baseline methods. The relatively high specificity suggests that the proposed attention-guided filtering mechanism improved the ability of the network to suppress irrelevant activations under unseen imaging conditions. In contrast, models relying primarily on direct bounding-box regression appeared more sensitive to background variations.

However, sensitivity values remained lower across all models on the KUMC dataset, indicating that lesion localization under severe domain variation remains challenging. This behavior may be associated with differences in lesion appearance and image contrast between datasets. Since the proposed framework learns region-aware spatial representations derived from box-supervised learning, the localization response may become more conservative under unfamiliar visual patterns.

## C. Cross-Dataset Evaluation on EDD2020

The quantitative results obtained on the EDD2020 dataset are summarized in Table IV. The proposed framework achieved the strongest overall balance between sensitivity, specificity, and AUC among the evaluated models. In particular, the proposed method achieved an AUC of 0.881, outperforming YOLOv7 and remaining competitive with YOLOv11.

The improved performance observed on EDD2020 suggests that the multi-scale contextual representation improved robustness against lesion appearance variation. Small lesions and low-contrast regions often require simultaneous modeling of local texture and broader contextual information. By integrating multiple receptive fields within the bottleneck representation, the proposed framework was able to better preserve lesion evidence under challenging imaging conditions.

TABLE IV. POLYPS DETECTION (%) ON EDD2020 DATASET.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Specificity</td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>F1-score</td></tr><tr><td rowspan=1 colspan=1>YOLOv7</td><td rowspan=1 colspan=1>78.0</td><td rowspan=1 colspan=1>76.6</td><td rowspan=1 colspan=1>77.3</td><td rowspan=1 colspan=1>85.0</td></tr><tr><td rowspan=1 colspan=1>YOLOv11</td><td rowspan=1 colspan=1>74.0</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>85.3</td><td rowspan=1 colspan=1>84.7</td></tr><tr><td rowspan=1 colspan=1>Proposed</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>96.6</td><td rowspan=1 colspan=1>88.1</td><td rowspan=1 colspan=1>88.2</td></tr></table>

## D. Cross-Dataset Evaluation on Kvasir-SEG

Table V presents the quantitative evaluation results obtained on the Kvasir dataset. The proposed framework achieved the strongest overall cross-dataset performance, obtaining the highest AUC and F1-score among the compared YOLO-based detection methods. In contrast, YOLOv7 demonstrated extremely weak specificity under this dataset shift, indicating unstable discrimination between lesion and background regions.

TABLE V. POLYPS DETECTION (%) ON KVASIR-SEG DATASET.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Specificity</td><td rowspan=1 colspan=1>AUC</td><td rowspan=1 colspan=1>F1-score</td></tr><tr><td rowspan=1 colspan=1>YOLOv7</td><td rowspan=1 colspan=1>95.7</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>52.1</td><td rowspan=1 colspan=1>85.0</td></tr><tr><td rowspan=1 colspan=1>YOLOv11</td><td rowspan=1 colspan=1>80.4</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>86.9</td><td rowspan=1 colspan=1>96.4</td></tr><tr><td rowspan=1 colspan=1>Proposed</td><td rowspan=1 colspan=1>96.7</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>95.0</td><td rowspan=1 colspan=1>98.2</td></tr></table>

The enhanced performance achieved by the proposed framework on Kvasir-SEG suggests that lesion-focused spatial learning improved robustness against dataset variability. Instead of relying solely on direct bounding-box regression, the proposed architecture first learns spatial lesion evidence and then converts these responses into localization outputs. This strategy appears to improve discrimination between true lesion structures and visually similar background patterns.

## Overall Qualitative Comparisons

Figure. 5 illustrates qualitative comparisons of activation responses and localization outputs obtained from YOLOv7, YOLOv11, and the proposed framework under challenging cross-dataset conditions. The first row shows an example from the KUMC dataset where all methods successfully localize the lesion region; however, the proposed framework produces a more compact and lesion-focused activation response with reduced activation in surrounding tissue regions. The second row corresponds to a challenging EDD2020 sample containing complex folds and illumination variations. In this case, YOLOv11 generates strong activations across large surrounding background regions rather than focusing only on the lesion area, whereas the proposed framework maintains more localized lesionfocused responses.

The third row illustrates a difficult example from the Kvasir dataset where YOLO-based detectors respond strongly to surrounding mucosal structures and broader anatomical regions, leading to less precise localization behavior and excessive background activation. In contrast, the proposed framework suppresses a substantial portion of these irrelevant responses and maintains stronger activation around lesion-related regions. The fourth row presents a lowcontrast LDPolyp example with challenging illumination artifacts. In this case, YOLOv7 and YOLOv11 produce diffuse activations extending beyond the lesion boundary, while the proposed framework preserves a more localized response near the lesion region despite the difficult imaging conditions.

These qualitative results support the quantitative findings reported in Tables II–V, particularly the improved specificity and AUC values observed for the proposed framework. The heatmap responses indicate that the proposed method is less influenced by irrelevant background structures and produces more lesion-centered activations. This behavior helps explain its more stable performance under cross-dataset evaluation, where imaging appearance and background patterns differ from the training data. Therefore, the qualitative analysis provides visual evidence for the role of attention-guided feature suppression in improving detection-oriented localization robustness.

## V. CONCLUSION

The proposed method in this work introduced a robust polyp localization framework based on the design of a detection-oriented localization mechanism. Specifically, the localization framework adopted residual learning, attentionguided suppression, and context-aware representations at multiple scales. According to experimental evaluation conducted using LDPolyp, KUMC, EDD2020, and Kvasir datasets, the proposed localization mechanism outperformed YOLOv7 and YOLOv11 with regard to localization accuracy and cross-dataset robustness especially in difficult image capturing scenarios characterized by poor contrast ratios, complex tissue structures, and variations between data domains. Additionally, the qualitative results illustrated the robust nature of the proposed method for lesion region localization even in situations where the standard methods could not correctly identify lesion regions.

![](images/69c816d9ef171597ecb688956df8e4fc6503c70adcb407bec09daa78b5c99461.jpg)

日

![](images/f4d6839ad92e3348413f0e7e4e26614a6889cee7146bb9bd75853d37da627407.jpg)

B

![](images/2258665970dfe037e4375b478b6ffbb870bb2d178818bc394c0fc5727495aca3.jpg)

![](images/77744492bf65f343bcf9aeaa157eb9f75dc21318c947f9f4ce2a678527ead621.jpg)

![](images/9f1e52c90471ef71e4403793255aee8c283d02e2c9833d259432e89ce38b6ac9.jpg)

![](images/31c4c0243168c04db76b11800c2cf9d7495c82c8be09bbed85c55aaa06a1c41b.jpg)

![](images/ad9bb387b7ed1cd13e63bddd07950f06f2ae64a4a20edc933f438cba99280aa7.jpg)

![](images/ab19fd6d379e98a303abfd93003b7cb3765064863bd4e15732c4504615f59815.jpg)

![](images/91fc6a5900d217cd01f11b86fabd68b7da87267a6341cd1838abc8bf4d3e0bea.jpg)

![](images/88e4fef92b0ac6fd46fcdf7d81db0f8c916830f1e4ff520e8491ed509b169cd8.jpg)  
(a)

![](images/52886ea8db337d9533aa91bed03f8e8efe3b58a81e6b9f2a29660a4fc945b69f.jpg)  
(b)

(c)  
![](images/ed12af34c4fa8b5a73505c48c7d3b6b7e2566da1fafb04b431d0a82e60e5843d.jpg)

![](images/25fef73211e2d83c8a83cd01ffde0f68822abc123a505d0d1005725dc5e17098.jpg)  
(d)  
Fig. 5. Qualitative comparison of activation responses and localization outputs for YOLOv7, YOLOv11, and the proposed framework under challenging cross-dataset conditions. rom top to bottom, the corresponding rows show representative samples from the KUMC, EDD2020, Kvasir-SEG, and LDPolyp datasets, respectively. (a) Ground truth lesion annotations. (b) YOLOv7 outputs. (c) YOLOv11 outputs. (d). Green boxes indicate ground truth lesion regions and red boxes indicate predicted localization outputs. The heatmaps illustrate the spatial activation behavior of each method during lesion localization.

## REFERENCES

[1] C.-Y. Wang, A. Bochkovskiy, and H.-Y. M. Liao, “YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 7464–7475.

[2] R. Sapkota and M. Karkee, “Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition,” Mar. 16, 2026, arXiv: arXiv:2510.09653. doi: 10.48550/arXiv.2510.09653.

[3] G. Papanastasiou, N. Dikaios, J. Huang, C. Wang, and G. Yang, “Is attention all you need in medical image analysis? A review,” IEEE J. Biomed. Health Inform., vol. 28, no. 3, pp. 1398–1411, 2023.

[4] P.-N. Bui, D.-T. Le, J. Bum, J. C. Han, V.-N. Pham, and H. Choo, “Multi-scale feature enhancement in multi-task learning for medical image analysis,” Artif. Intell. Med., p. 103338, 2025.

[5] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-End Object Detection with Transformers,” in Computer Vision – ECCV 2020, vol. 12346, A. Vedaldi, H. Bischof, T. Brox, and J.-M. Frahm, Eds., in Lecture Notes in Computer Science, vol. 12346. , Cham: Springer International Publishing, 2020, pp. 213–229. doi: 10.1007/978-3-030-58452-8\_13.

[6] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 779–788.

[7] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 6, pp. 1137–1149, 2016.

[8] W. Liu et al., “SSD: Single Shot MultiBox Detector,” in Computer Vision – ECCV 2016, vol. 9905, B. Leibe, J. Matas, N. Sebe, and M. Welling, Eds., in Lecture Notes in Computer Science, vol. 9905. , Cham: Springer International Publishing, 2016, pp. 21–37. doi: 10.1007/978-3-319-46448-0\_2.

[9] M. Lalinia and A. Sahafi, “Colorectal polyp detection in colonoscopy images using YOLO-V8 network,” Signal Image Video Process., vol. 18, no. 3, pp. 2047–2058, Apr. 2024, doi: 10.1007/s11760-023-02835-1.

[10] X. Yang et al., “YOLO-OB: An improved anchor-free real-time multiscale colon polyp detector in colonoscopy,” Biomed. Signal Process. Control, vol. 103, p. 107326, 2025.

[11] J. Wan, W. Zhu, B. Chen, L. Wang, K. Chang, and X. Meng, “CRH-YOLO for precise and efficient detection of gastrointestinal polyps,” Sci. Rep., vol. 14, no. 1, p. 30033, 2024.

[12] L. Wang, J. Wan, X. Meng, B. Chen, and W. Shao, “MCH-PAN: gastrointestinal polyp detection model integrating multi-scale feature information,” Sci. Rep., vol. 14, no. 1, p. 23382, 2024.

[13] Y. Xie, Y. Yu, M. Liao, and C. Sun, “Gastric polyp detection module based on improved attentional feature fusion,” Biomed. Eng. OnLine, vol. 22, no. 1, p. 72, Jul. 2023, doi: 10.1186/s12938-023- 01130-x.

[14] O. Zare et al., “LightAttn-YOLO-V8: an efficient colorectal polyp detection with lightweight attention mechanism based on YOLO,” Sci. Rep., 2026.

[15] D.-P. Fan et al., “PraNet: Parallel Reverse Attention Network for Polyp Segmentation,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2020, vol. 12266, A. L. Martel, P. Abolmaesumi, D. Stoyanov, D. Mateus, M. A. Zuluaga, S. K. Zhou, D. Racoceanu, and L. Joskowicz, Eds., in Lecture Notes in Computer Science, vol. 12266. , Cham: Springer International Publishing, 2020, pp. 263–273. doi: 10.1007/978-3-030-59725- 2\_26.

[16] X. Zhao, L. Zhang, and H. Lu, “Automatic Polyp Segmentation via Multi-scale Subtraction Network,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2021, vol. 12901, M. De Bruijne, P. C. Cattin, S. Cotin, N. Padoy, S. Speidel, Y. Zheng, and C. Essert, Eds., in Lecture Notes in Computer Science, vol. 12901. , Cham: Springer International Publishing, 2021, pp. 120–130. doi: 10.1007/978-3-030-87193-2\_12.

[17] N. K. Tomar et al., “Fanet: A feedback attention network for improved biomedical image segmentation,” IEEE Trans. Neural Netw. Learn. Syst., vol. 34, no. 11, pp. 9375–9388, 2022.

[18] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” Nov. 2015. doi: 10.1007/978-3-319-24574-4\_28.

[19] “LDPolypVideo Benchmark: A Large-scale Colonoscopy Video Dataset of Diverse Polyps,” MICCAI 2021 - Accepted Papers and Reviews.

[20] S. Ali et al., “Assessing generalisability of deep learning-based polyp detection and segmentation methods through a computer vision challenge,” Sci. Rep., vol. 14, no. 1, p. 2032, Jan. 2024, doi: 10.1038/s41598-024-52063-x.

[21] A. P. Bradley, “The use of the area under the ROC curve in the evaluation of machine learning algorithms,” Pattern Recognit., vol. 30, no. 7, pp. 1145–1159, 1997.

[22] D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,” BMC Genomics, vol. 21, no. 1, p. 6, Dec. 2020, doi: 10.1186/s12864-019-6413-7.

[23] F. Gamboa, T. Klein, A. Lagnoux, and L. Moreno, “Sensitivity analysis in general metric spaces,” Reliab. Eng. Syst. Saf., vol. 212, p. 107611, 2021.

[24] R. Trevethan, “Sensitivity, specificity, and predictive values: foundations, pliabilities, and pitfalls in research and practice,” Front. Public Health, vol. 5, p. 307, 2017.

[25] Y. Ma, X. Chen, K. Cheng, Y. Li, and B. Sun, “LDPolypVideo Benchmark: A Large-Scale Colonoscopy Video Dataset of Diverse Polyps,” Sep. 2021. doi: 10.1007/978-3-030-87240-3\_37.

[26] K. Li et al., “Colonoscopy polyp detection and classification: Dataset creation and comparative evaluations”, doi: 10.1371/journal.pone.0255809.

[27] S. Ali et al., “Endoscopy disease detection challenge 2020.” Jan. 15, 2020. doi: 10.21227/f8xg-wb80.