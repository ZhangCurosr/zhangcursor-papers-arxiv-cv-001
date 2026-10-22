# Early versus Late RGB–Infrared Fusion for Counter-UAV Detection: A Comparative Study

ALBERTO DE ZAN<sup>1</sup>, DENIS TAVARIS<sup>1</sup>, GIAN LUCA FORESTI<sup>1</sup>, and IVAN SCAGNETTO<sup>1</sup>

<sup>1</sup>Department of Mathematics, Computer Science, and Physics, University of Udine, Via delle Scienze 206, 33100 Udine, Italy Corresponding author: Ivan Scagnetto (e-mail: ivan.scagnetto@uniud.it).

Note: This arXiv version differs from the article published in IEEE Access, vol. 14, pp. 149702–149712, 2026, DOI: 10.1109/ACCESS.2026.3735919. It includes revisions requested by the reviewers during the final review round that could not be incorporated into the published version. The main changes include additional references, a more detailed description of the paired dataset, an additional fusion model (IFCNN) in the comparison, an extended discussion of model performance and an explanation of the distance threshold used for late fusion.

Abstract—This paper presents a novel strategy for UAV detection through visible (RGB) and infrared (IR) fusion, in order to improve accuracy and precision in harsh environments and weather conditions. In this research, two opposite approaches have been studied and implemented: the first concerns imagelevel (early) fusion, which merges two different spectra into a single image to which one detector is applied. On the other hand, the second approach, called decision-level (late) fusion, focuses on blending together the output bounding boxes of two different detection pipelines, aiming to create a common result that can be applied to both spectra simultaneously. Both strategies have been extensively tested in a dedicated environment, which included alternating low and high brightness scenes, different backgrounds (sky and ground), distances (5, 25 and 60 m), altitudes and zoom levels (1X, 2X and 4X). These tests allowed for the assessment of actual detection capabilities and system reliability in realworld applications. Furthermore, the two fusion strategies were evaluated and compared over a set of aligned images in order to highlight the better approach of the two. The results indicate that early fusion achieves better raw performance, since it incorporates the thermal information from the beginning of the training process. However, late fusion presents an innovative and useful detection workflow that can be applied in cases of sensor or frame misalignment. Both positive and negative aspects have also been pointed out and discussed, in order to give a more comprehensive overview. Finally, the paper outlines implementation considerations and practical recommendations based on operational priorities. Overall, the findings contribute to the research in RGB–IR fusion modalities, with the goal of enhancing UAV detection capabilities.

Index Terms—Unmanned Aerial Vehicles (UAVs), Object Detection, YOLO, Early Fusion, Late Fusion, Multimodality, Infrared Imaging.

## I. INTRODUCTION

U NMANNED Aerial Vehicles (UAVs) are becoming increasingly common in military and civilian applications. They can be used in surveillance, mapping and transporting materials. However, there are safety and security concerns which have to be taken into consideration, since modern UAVs might be able to operate in varied environments and evade traditional surveillance methods. In order to reduce these weaknesses, many advanced sensors are used in current scenarios. For example, visible light (RGB) cameras are widely adopted, as they offer excellent spatial resolutions and reduced costs [1]. However, they have problems in low light, adverse weather conditions [2] and crowded backgrounds. Concerning this particular aspect, the terrain texture may be difficult to separate from the drone’s surface, since UAVs are usually made from composite and camouflaged materials. On the other hand, infrared (IR) cameras work effectively in low light situations and night scenes. Nevertheless, they have reduced thermal and spatial resolution and are expensive. Therefore, this research focuses on fused solutions for UAV detection, aiming to combine the strengths of both sensors [3].

There are some studies in Computer Vision which have explored the capabilities of sensor fusion strategies in object detection systems [4]. However, the Counter-UAV setting still remains an underexplored research area for fusion. Most of the existing approaches require significant adjustments and fixes to the sensors before fusion can be applied. In this paper, the following innovative fusion pipelines for YOLO-based Counter-UAV detection have been adopted and evaluated:

• An image-level (early) fusion that composes synchronized and aligned RGB and IR inputs into a unified representation for a single detector.

• A decision-level (late) fusion that processes RGB and IR streams independently and merges their detections postinference.

This research presents two logical strategies and the framework designed and applied for each fusion type.

Finally, in order to present a complete evaluation, extensive performance analyses have been conducted, including quantitative metrics (such as average detection precision, recall and F1-score) and qualitative visualizations through image samples.

a) Synopsis.: This work presents two RGB–IR fusion architectures for drone-to-drone detection using YOLO: an image-level (early) fusion combining synchronized inputs into a single stream, and a decision-level (late) fusion merging outputs from separate detectors. Evaluated across varied conditions, the study compares accuracy, efficiency, and robustness, offering practical guidance for Counter-UAV deployment.

## II. RELATED WORK

## A. Detection Techniques

The detection of small unmanned aerial vehicles (UAVs) has attracted increasing attention due to their rapid proliferation and the associated security implications [5]. Early visionbased UAV detection methods relied primarily on classical image processing and handcrafted feature extraction techniques. Background subtraction, motion-based segmentation and orientation detection were commonly employed in static-camera surveillance scenarios, leveraging temporal consistency to isolate moving aerial targets [6]. Feature-based approaches utilizing Histogram of Oriented Gradients (HOG) [7], Haarlike features and Scale-Invariant Feature Transform (SIFT) [8] descriptors were also explored for detecting aerial objects against sky backgrounds. While computationally efficient, such methods often suffer from limited robustness under dynamic backgrounds, illumination changes and cluttered scenes. The emergence of deep learning has significantly advanced object detection performance [9]. Two-stage detectors, such as Faster R-CNN [10], achieve high detection accuracy through region proposal mechanisms followed by classification and regression refinement. In addition, they offer strong capabilities in feature extraction tasks. However, their computational complexity can limit real-time applicability in resource-constrained operational environments. In contrast, one-stage detectors, including Single Shot MultiBox Detector (SSD) [11] and the You Only Look Once (YOLO) [12] family, directly predict bounding boxes and class probabilities in a unified framework, enabling significantly higher inference speeds [13].

Moreover, models based on transformers, such as Vision Transformers (ViT) [14], [15], DEtection TRansformer (DETR) [16] and Spatio-Temporal Transformer [17], have increased overall robustness by capturing long-range dependencies within images. Furthermore, advanced multimodal deep learning models with hybrid CNN-transformer architectures enhance both feature extraction and decision making through data obtained from numerous sensors. Additionally, the integration of multi-sensor systems into embedded platforms has substantially increased their value in practical applications, including drone surveillance and search-and-rescue operations, demonstrating the effectiveness of these multisensor approaches.

Recent UAV vision research has also explored multimodal approaches beyond conventional object detection. Lu et al. [18] proposed MTCNet, a multimodal network for time-sensitive target concealment in UAV aerial imagery, while Tang et al. [19] introduced R2PLoc, a region-to-point framework for UAV visual geo-localization. These works further demonstrate the increasing adoption of multimodal and hierarchical visual representations in UAV-based perception systems. Classificationand-regression-based detectors, such as YOLO, have become particularly attractive for UAV detection due to their favorable trade-off between accuracy and latency and wide compatibility with many devices. Several studies have adapted YOLO architectures for small-object detection, aerial surveillance and drone monitoring tasks [20]. Modifications such as multiscale feature aggregation and anchor box optimization have been proposed to improve sensitivity to small aerial targets. Nevertheless, most existing UAV detection approaches rely solely on visible-spectrum imagery, which inherently limits performance under low-light, night-time or adverse weather conditions [21]. These limitations motivate the exploration of complementary sensing modalities to enhance robustness in real-world Counter-UAV applications, potentially involving other drones in the detection process [22].

## B. Fusion Techniques

Overall, RGB and infrared modalities exhibit complementary characteristics. Therefore, their integration represents a promising direction for enhancing UAV detection performance in operational Counter-UAV scenarios [23], [24]. Methods such as deep fusion networks and attention mechanisms have shown significant improvements in accuracy, particularly in scenarios such as low-visibility conditions and occlusions [25], but also in challenging cases such as the resemblance between drones and birds. Furthermore, several studies have shown interest in RGB and IR image fusion methods, with the goal of creating highly informative multi-modal images that integrate features from both spectra. These techniques employ two-scale image decomposition [26], contrast enhancement [27] and swin transformer [28]. However, they usually need aligned image pairs for achieving fusion, which are not always available due to the sensors’ physical limitations. For this reason, some fusion methods have been developed to also integrate image registration and alignment [29], [30]. However, recent works have explored approaches for fusing spatially misaligned yet temporally synchronized images [31]. Furthermore, multi-modal object detection has already been explored by Yue Cao et al. [32] and Fan Xu et al. [33]. More recently, Zhang et al. proposed CCSFuse [34], a collaborative compensation and selective fusion framework specifically designed for UAV-based RGB-IR object detection, further demonstrating the growing interest in feature-level fusion for multimodal aerial detection. Some studies also involve Bayesian fusion for enhancing perception [35]. Some studies also involve Bayesian fusion for enhancing perception [35]. Modern approaches also integrate additional data sources such as LiDAR and radar, in order to enhance depth perception and robustness, although these methods are mainly oriented toward pedestrian detection [36]. In the UAV domain, RGB-T fusion detectors such as SF-YOLO [37] have also been proposed for drone scenes. Nevertheless, despite these recent advances, publicly available UAV detection methods operating on multimodal fused data remain comparatively limited, particularly when considering practical constraints such as sensor misalignment and realtime deployment.

After this brief introduction, in the following subsections we delve deeper into the three most important fusion methods families: early-fusion, mid-fusion and late-fusion.

1) Early-Fusion (Pixel-Level): It combines the information of the two modalities at the pixel level. It is the most intuitive fusion approach, where IR and RGB images are concatenated to generate a 4-channel image, which is then fed into a regular object detection architecture. Several imagelevel fusion networks have been proposed in the literature, ranging from autoencoder-based methods such as DenseFuse [38] to illumination-aware architectures such as PIAFusion [39], IAIFNet [40] and FCDFusion [41]. These techniques rely on dedicated, relatively complex fusion architectures to better merge the images in the two spectra. However, as Wagner et al. show [42], pixel-level fusion was outperformed by deeper fusion variants in their pedestrian detection experiments, even though there are ways to improve it, such as the ones presented by Tianyi Zhao et al. [43]. Since aligned RGB-Thermal datasets are not available, the method developed by Lizhi Yang et al. [44] has also been evaluated, which uses a CycleGAN to create aligned synthetic RGB/IR data [45], [46]. In fact, it is important that the image pairs are perfectly aligned before performing early-fusion. In a more simplified way, this architecture can be described as a neural network composed of two different branches, dedicated respectively to RGB and IR image processing. In addition, this network presents a decision layer, which chooses how much importance to give to which of the two sub-networks.

In order to merge RGB and IR images, it is possible to use Python Imaging Library (PIL or Pillow) or dstack. The approach that these two techniques follow is based on the raw channel information overlapping. In fact, they take into consideration the bit depth of RGB images, which refers to the number of bits used to represent the color of a single pixel. It is usually 24 bits (8 bits for each color channel) plus other 8 bits for transparency (alpha channel). However, the transparency channel is not always present and it depends on the extension in which the image file is saved. In fact, if more than 3 channels are needed, the .png format needs to be considered, since .jpg supports a maximum of 3 channels. For IR images, only their thermal information (8 bits) should be considered, which will be added to the final fused image. Therefore, the resulting merged image will be of 32 bits.

2) Late-Fusion (Decision-Level): This fusion type merges detection results from separate models, improving robustness by leveraging complementary sensor characteristics. Furthermore, the aligned inputs (RGB and the respective IR counterpart) of the two modalities are fed separately into two unimodal object detection models to generate bounding boxes and the predicted boxes are fused using statistical methods, such as weighted averaging of the bounding box coordinates, which is based on the confidence scores from each modality. This method gives more importance to the bounding boxes with higher confidence scores, and this is a suitable option for the purpose of UAV detection.

This method works with both aligned and misaligned images. For this reason, it could be considered more resilient to sensor misalignment with respect to early fusion. However, when the same object appears with different orientations or in different positions across the two images, the resulting bounding boxes may be less precise in terms of localization. In practice, the bounding box tends to expand, and especially in the case of unaligned images, the object is still correctly identified while the box remains largely consistent across both views. As a result, the final bounding box may show slight variations in size and position.

In addition to the adopted strategy, there exist also ensemble methods such as probabilistic ensembling [35], in which a bounding box is considered valid if it is detected by both modalities.

3) Mid-Fusion (Feature-Level): This strategy, which falls between early and late fusion techniques [47], might be the best and most balanced option to choose when trying to develop a computer vision system that exploits sensor fusion. Essentially, this fusion integrates feature maps extracted separately from RGB and IR modalities before applying detection networks. More specifically, features are captured from each of the two spectra using specific convolutional pipelines in the first stages. Subsequently, feature maps are merged using methods like concatenation or sum, which prepare information to be used for the final layers of detection. Taking into account these aspects, the mid-fusion idea has the ability to improve the robustness of computer vision and object detection systems in harsh environments with low light [48], [49].

This type of fusion has also many advantages compared with the other two. First of all, images do not need to be merged into a single one as early-fusion does, but all their channels are kept in the original form, preserving modality-specific information. Nor are the final detection results combined using statistical methods. Therefore, this strategy appears to be more flexible with respect to the input data type and the final detection goals that we may have. In fact, it has already been studied and implemented in some feature alignment methods for RGB and IR images [50]. Recent UAV-specific work further supports this direction. Zhang et al. [34] proposed CCSFuse, which performs collaborative compensation and selective fusion for RGB-IR object detection, illustrating how modality-specific features can be selectively combined to improve multimodal UAV perception. However, despite these positive aspects, only the other two strategies have been implemented, as mid-fusion implies creating a custom detection architecture, markedly different from the state-of-the-art YOLO one used for the tests. In addition, early-fusion and late-fusion allowed us to obtain useful metrics for comparing the strategies in several scenarios, which were not possible with mid-fusion.

## III. LOGICAL PROPOSED ARCHITECTURES

The architecture proposed is based on the YOLO framework, since it represents the state-of-the-art in the real-time object detection field. Moreover, two separate fusion pipelines have been developed for enhancing detection performance and accuracy. Finally, a multimodal dataset [51] of real-world data on still and moving drones was built, for testing the validity of the following fusion methods. This approach has been preferred to the synthetic one because real acquired data better reflect operational conditions and lead to superior performance. In addition, the actual behaviour of a drone, that is, its ability to rotate, shift and change pose over the three axes (roll, pitch and yaw), has also been considered while collecting data. The paired dataset comprises 400 aligned images (300 for training and 100 for validation, following a 75/25 split) for each fusion strategy evaluated, starting from a base of aligned RGB-IR pairs.

## A. Early Fusion

The early fusion approach combines RGB and IR images at the pixel level to create a single multimodal input. This approach requires frames to be aligned. This process was performed by applying transformation steps to each image, such as interpolation, perspective distortion and rotation. After this, an overlap of the channels was performed, while removing the transparency level (also called ”A” in RGBA image format), which was present in .png images. However, since the focus of this research strictly concerns the analysis of fusion strategies, the reader is referred to the paper on autonomous UAV detection architecture development [52], in which all alignment steps are described.

Once spatial alignment between RGB and infrared frames was achieved, the fusion was performed through overlap, by stacking the image channels into a single array of 4 spectra. The RGB image was split into three channels: Red, Green and Blue. Each of them is represented by a two-dimensional intensity map, where every pixel contains intensity values for that particular channel. A visual interpretation is shown in Figure 1. On the other hand, the IR image was converted

![](images/1c327c972cce531e04278592c287bb767d914a3ca4ac4513420ae3c9cbc9e4f9.jpg)  
Fig. 1: From left to right: red, green and blue channel’s intensity maps.

into a grayscale single-channel image. These channels were then combined into a 4-channel image (RGBT), where the fourth channel corresponded to the thermal information. The operation result is shown in Figure 2.

For implementation simplicity, the Python Imaging Library (PIL) was employed, as it allows independent access to individual image channels. Initial experiments involved the creation of four-channel RGBT images obtained by stacking the red, green, blue, and thermal components into a single tensor. In addition to direct channel stacking, pretrained

![](images/02daaa45823d75dba8740746de41a183123709d02a892ac197c5d6791c6e1fd9.jpg)  
Fig. 2: Graphical representation of how layers are overlapped by Pillow library in order to get the resulting image on the right.

fusion models such as PIAFusion in Figure 3, FCDFusion in Figure 4 and DenseFuse in Figure 5 were evaluated to assess alternative image-level fusion strategies. These approaches generated fused outputs without requiring manual channel manipulation. However, Ultralytics introduced support

![](images/69f2a771adb745851f673cc2e21cdd86cfb7e8dc29449d9d013586fd6052fdc3.jpg)  
Fig. 3: PIAFusion

![](images/068b446fe2a496247afc741cb736fd02d8ddbe83b273c5b7a541a243e0de816c.jpg)  
Fig. 4: FCDFusion

![](images/5ee2b9c9b75e5791e9585c079d606cffce85a8e7539495d3495d9069e41436c1.jpg)  
Fig. 5: DenseFuse

![](images/d0ba4ef9ded85db24b74db0fe094f6c018eeedab0623f51882340e321b5b63f4.jpg)  
Fig. 6: IFCNN  
Fig. 7: Visible comparison between state-of-the-art fusion strategies, exploiting IAN framework and convolutional neural networks for multimodal image generation.

for multi-channel training only recently [53], subsequent to the experiments reported here. Hence, although OpenCV2 library, commonly used within YOLO pipelines, allows image loading with an arbitrary number of channels using the cv2.IMREAD\_UNCHANGED flag, in previous versions of Ultralytics, in order to use 4-channel inputs, extensive modifications to the YOLO architecture would be required, including adaptation of the model’s input layer and associated preprocessing routines. In order to preserve architectural consistency and ensure fair comparison with baseline experiments, modifications to the core framework and alteration to the network structure were avoided.

Instead, the characteristics of RGB channels were taken into consideration. RGB usually exhibits a high degree of spectral correlation, since most objects in the environment reflect light similarly across the visible spectrum. Both limitations and contributions of each channel have been summarized in Table I. Due to these highlights reported, a modified 3-channel tensor out to be redundant in several outdoor UAV scenes where red and green spectral content are often predominant. The blue channel was then replaced with the thermal component, mitigating some of the atmospheric noise. This resulted in an RGT image (Red, Green, Thermal), maintaining compatibility with YOLOv8 architecture while integrating complementary spectral information. Consequently, a 3-channel fusion representation was adopted. Therefore, the same channel-stacking principles used in the 4-channel implementation were applied to construct the 3-channel fused images. The methodological workflow is summarized in the dedicated flowchart in Figure 8. The process shown in Figure 9 can be summarized as

TABLE I: Summary of the characteristics of the RGB channels [54].
<table><tr><td>Channel</td><td>Main contribution</td><td>Limitations / Remarks</td></tr><tr><td>Green</td><td>Contributes the most to the perception of luminance and fine details.</td><td>Provides high structural in- formation.</td></tr><tr><td>Red</td><td>Crucial for discriminating between minerals and soils.</td><td>Complementary for material recognition.</td></tr><tr><td>Blue</td><td>Captures wavelengths in the 450–495 nm range, useful for identifying water bodies and atmospheric effects.</td><td>Lower contrast, limited ther- mal and low-light contribu- tions, sensitive to Rayleigh scattering, noisier in long- distance imagery.</td></tr></table>

was created by discarding the blue channel, which turned

![](images/a3220990b844778462ed281753a622fcfbff978696de3dba8b84860672c587d9.jpg)  
Fig. 8: Early Fusion framework.

a shift on the left of all the image channels, in order to include the thermal one inside the final image. Therefore, the resulting fused image consisted of red, green and thermal channels, forming an RGT representation. Some examples are shown in Figure 10. The distinction between a CIR image [55] and the proposed RGT one relies on the infrared type chosen. In fact, the camera sensor used for the tests operates in far-infrared (FIR) range, whereas CIR relies on near-infrared (NIR) imagery.

This design enabled YOLOv8s model training without architectural modification and allowed operation with the traditional YOLO detection pipeline while reducing potentially redundant spectral information.

Formally, each spectral channel is defined as a function over pixel coordinates (i, j):

$$
R ( i , j ) : \mathrm { R e d ~ i n t e n s i t y ~ a t ~ p i x e l } ( i , j )\tag{1}
$$

$$
G ( i , j ) : { \mathrm { ~ G r e e n ~ i n t e n s i t y ~ a t ~ p i x e l ~ } } ( i , j )\tag{2}
$$

$$
T ( i , j ) : \ { \mathrm { T h e r m a l ~ i n t e n s i t y ~ a t ~ p i x e l ~ } } ( i , j )\tag{3}
$$

![](images/558e8171c16f73ec9770bce0ded40a645a5e6e76e084428baf4af3fa2ede1ff3.jpg)  
Fig. 9: Channel shifting process. As it can be seen, there is a clear difference between CIR and RGT representations. Color-infrared imagery uses a portion of the electromagnetic spectrum known as near infrared, which ranges from 0.70µm to 1.0µm. RGT uses instead far infrared. In both cases, blue wavelengths are shifted out of the visible portion of the spectrum. However, only 3 channels are considered part of the RGT representation, without considering the non-visible part of the spectrum.

The fused image tensor is obtained by stacking these channels:

$$
I _ { R G T } ( i , j ) = \bigg [ R ( i , j ) \bigg ]\tag{4}
$$

Each pixel location $( i , j ) \in \mathbb { Z } ^ { 2 }$ is therefore mapped to a threedimensional spectral vector in $\mathbb { R } ^ { 3 }$

$$
I _ { R G T } : \mathbb { Z } ^ { 2 } \to \mathbb { R } ^ { 3 } , \quad ( i , j ) \mapsto ( R ( i , j ) , G ( i , j ) , T ( i , j ) ) .\tag{5}
$$

The complete fused image can thus be expressed as a tensor:

$$
I _ { R G T } \in \mathbb { R } ^ { H \times W \times 3 } ,\tag{6}
$$

where H and W denote image height and width, respectively. The same pipeline was also adopted for the other two channel combinations, respectively RBT and GBT, each obtained by omitting one of the three RGB channels. This enabled an ablation study on channel selection, providing a formal justification for the choice of RGT in the proposed testing scenario.

After generating and augmenting a substantial number of fused images, the YOLOv8s detector was trained using the standard (visible and infrared) and the fused datasets. All of them contain a total of 300 images for the training part and 100 for the validation one, corresponding to a split of 75% and 25% respectively. The results obtained after training and validation processes were compared against models trained exclusively on RGB and IR data.

These findings can be interpreted in light of the Data Fusion Principle<sup>1</sup>. Accordingly, a fusion-based system is expected to outperform its unimodal counterparts if complementary information is effectively exploited.

The comparative performance metrics obtained on unimodal and multimodal datasets are summarized in Table II. While the standard RGB configuration achieved higher precision, the overall detection performance did not uniformly surpass the fused models across all evaluation metrics. In fact, the superiority of the three proposed fusion strategies first emerges on mAP@50-95. The highest one is reached by RGT with 56.2%. A similar improvement can be observed on mAP@50, where the proposed fused approach achieves a result of 97.8%, compared to 89.8% of Infrared. These are indexes of good model performance, since mAP evaluates it over all confidence thresholds, rather than at a single operating point.

![](images/a6752380f993eb900b3aa5b1cf8574471bb88c661641342ab8655c7ab4c4f28e.jpg)  
Fig. 10: From the top: RGT, RBT and GBT fusion strategies. In all the cases only 3 of the 4 channels are considered for creating the fusions. For RGT the blue channel has been omitted in favour of the thermal one. A similar type of image is usually called CIR or false-color image [56]. The other types have only been considered for a technical comparison.

TABLE II: Performance comparison of YOLOv8s model (11.2M parameters, 28.6B FLOPs) over standard spectra (RGB, IR), literature fusions (FCD, PIA, Dense, IFCNN) and proposed early fusion approaches (RGT, RBT, GBT). The model has been trained for 300 epochs with imgsz=640 over the same dataset, only changing the color channels and the fusion strategies for the images inside it.
<table><tr><td>Dataset</td><td>Precision</td><td>Recall</td><td>mAP@50</td><td>mAP@50-95</td><td>F1-Score</td></tr><tr><td>RGT</td><td>95.9%</td><td>94.6%</td><td>97.8%</td><td>56.2%</td><td>95.2%</td></tr><tr><td>RBT</td><td>96.9%</td><td>93.8%</td><td>96.9%</td><td>55.6%</td><td>95.3%</td></tr><tr><td>GBT</td><td>93.2%</td><td>85%</td><td>92.4%</td><td>53.3%</td><td>88.9%</td></tr><tr><td>FCDFusion [41]</td><td>98.9%</td><td>91.8%</td><td>96.8%</td><td>56.9%</td><td>95.2%</td></tr><tr><td>PIAFusion [39]</td><td>96.2%</td><td>76.4%</td><td>84.6%</td><td>42.2%</td><td>85.2%</td></tr><tr><td>DenseFuse [38]</td><td>95.5%</td><td>84.7%</td><td>91.5%</td><td>50.3%</td><td>89.8%</td></tr><tr><td>IFCNN [57]</td><td>97.8%</td><td>90.7%</td><td>93.8%</td><td>50.4%</td><td>94.8%</td></tr><tr><td>IR</td><td>89%</td><td>80.8%</td><td>89.8%</td><td>51.5%</td><td>84.7%</td></tr><tr><td>RGB</td><td>97.9%</td><td>91.7%</td><td>97%</td><td>48.8%</td><td>94.7%</td></tr></table>

Another important point regards F1-Score, which penalizes models with unbalanced precision and recall values.

All these results highlight the improvements of RGT fusion in comparison to the other strategies reported. In fact, the metrics suggest that the proposed multimodal approach has gained the capability to enrich the scope and improve the stability of the model in object detection tasks. In addition, the comparison between fusion strategies enriches the study, highlighting the importance of a dedicated fusion approach with respect to the use-case scenario.

Another interesting point regards the results obtained from literature fusion strategies. All of them cannot be considered for real-time purposes, since they rely on dedicated fusion networks (illumination-aware or autoencoder-based) whose computational cost is unsuitable for real-time processing. Thus, even though the FCD approach gives promising results, it is not usable in real-time scenarios.

According to the results obtained, the RGT fusion strategy was also better than the other two presented (RBT and GBT). For this reason, it was selected for tests in real-world scenarios. Nevertheless, it should be pointed out that this is a preliminary evaluation step for the topic. It could surely be enriched and extended with a larger fused dataset and other object detection models, in order to carry on the research in the field.

A further consideration can be made by comparing the proposed RGT strategy with FCDFusion, which turned to be the two best models. Although FCDFusion outperforms RGT in precision and mAP@50-95, RGT achieves higher recall and mAP@50, while both methods obtain the same F1-Score of 95.2%, indicating comparable overall detection performance. Therefore, the superiority of RGT should not be interpreted exclusively in terms of individual detection metrics, but rather considering the relationship between performance and computational complexity. In fact, FCDFusion requires a dedicated fusion network to generate the fused representation, introducing an additional deep learning model into the processing pipeline. This results in additional computational requirements and increases the overall complexity of the system. Conversely, RGT performs the fusion directly at the frame level, without requiring a dedicated fusion network during inference. The multimodal information can therefore be combined in real time and directly provided to the object detector, which represents a significant advantage for real-world applications.

## B. Late Fusion

In the late fusion approach aligned RGB and IR images are processed separately by two object detection models. The resulting bounding boxes are then fused using a weighted average method. For each pair of boxes, the center coordinates are averaged based on their confidence scores, while the maximum width and height are kept to avoid box shrinking. A distance threshold δ is used to decide whether two boxes should be fused. In particular, if the centers of two boxes are within this range, they are fused, otherwise they are left as they are. More in detail, the Euclidean distance between the center points of two bounding boxes in the two spectra is computed. If this distance falls in the threshold set, weighted average is performed. Otherwise, the original box created by the model is preserved. This approach has been designed to avoid discarding detections that may only exist in one modality and to work in case of misaligned frames. In fact, if the centers of the two bounding boxes are near, the system fuses them. Alternatively, it preserves what each model has decided during detection.

In the decision-level fusion strategy, two spatially aligned images, one RGB and its corresponding infrared (IR) counterpart, are processed independently by their respective detection models. Each detector produces a set of bounding boxes with associated confidence scores. Fusion is subsequently performed at the bounding box level using a weighted averaging method. The weighted average approach computes the fused

![](images/abba048b30e32eefd6fc3a08b7b79ea939c9ca8f25d37587c0cf2ce203a1b9e8.jpg)  
Fig. 11: The bounding box is unique for both images. As it was seen in the experiments, it could work even with misaligned frames.

bounding box coordinates based on the confidence scores of the individual detections. This method can be applied to both perfectly aligned and moderately misaligned image pairs. However, if the same object appears with slight positional or orientational discrepancies across the two modalities, the resulting fused bounding box may exhibit reduced localization precision. In such cases, the fused bounding box may become enlarged, encompassing the object regions identified in both modalities within a single unified box.

The fusion procedure begins with the computation of the center coordinates of the new bounding box using confidenceweighted averaging. The width and height of the fused bounding box are then defined as the maximum of the corresponding dimensions from the RGB and IR detections. This design choice prevents the fused bounding box from shrinking excessively and ensures that the spatial extent of the detected object is fully preserved.

A dedicated fusion function was implemented to manage all detection scenarios. Since decision-level fusion requires detections from both modalities, situations may arise in which a UAV is detected by only one of the two models. In such cases, the bounding boxes generated by the detecting model are retained without modification. In order to regulate the correspondence between detections from the two modalities, a fixed distance threshold (δ = 50 pixels) was introduced. This threshold is defined as a geometric correspondence tolerance for the spatial displacement that may occur between RGB and IR detections. Since the two optical axes are physically separated and therefore non-coaxial, the same UAV may appear at different image coordinates in the two modalities due to parallax and residual registration errors. The threshold consequently represents the maximum positional discrepancy considered compatible with the hypothesis that two detections correspond to the same physical target. A fixed pixel-domain threshold was adopted because the proposed fusion operates directly on detector outputs at a fixed input resolution of 640 × 640 pixels. The value of δ is therefore treated as a predefined correspondence tolerance rather than as an optimized detection hyperparameter, and is kept constant throughout the evaluation. In particular, the use of a distance tolerance allows the proposed method to tolerate spatial discrepancies between the two modalities while preventing detections with larger spatial separation from being merged.

![](images/44da152cbcd5ad769c7667912ad15d10e41bfb90f42c499d579cf52842d27cb6.jpg)  
Fig. 12: Late Fusion logic.

For each pair of RGB and IR bounding boxes, the Euclidean distance between their center coordinates has also been computed. If this distance is below the threshold, weighted averaging is applied. Otherwise, the bounding boxes are preserved independently. This strategy enables the method to operate effectively under moderate sensor misalignment. When detections are spatially consistent across modalities, fusion is performed; when they differ significantly, each model’s decision is maintained.

Formally, consider two bounding boxes

$$
B _ { 1 } = ( x _ { 1 } , y _ { 1 } , w _ { 1 } , h _ { 1 } ) , \quad B _ { 2 } = ( x _ { 2 } , y _ { 2 } , w _ { 2 } , h _ { 2 } ) ,
$$

with associated confidence scores $s _ { 1 }$ and $s _ { 2 }$ . The center coordinates of the fused bounding box are computed as confidenceweighted averages:

$$
x _ { c } = { \frac { s _ { 1 } x _ { 1 } + s _ { 2 } x _ { 2 } } { s _ { 1 } + s _ { 2 } } }\tag{7}
$$

$$
y _ { c } = { \frac { s _ { 1 } y _ { 1 } + s _ { 2 } y _ { 2 } } { s _ { 1 } + s _ { 2 } } }\tag{8}
$$

The width w and height h of the fused bounding box are defined as the maximum between the values of the two original boxes. The resulting fused bounding box is therefore represented as:

$$
B _ { f } = ( x _ { c } , y _ { c } , w , h )\tag{9}
$$

To determine whether fusion should be applied, the Euclidean distance between the centers $C _ { 1 } = ( x _ { 1 } , y _ { 1 } )$ and $C _ { 2 } = ( x _ { 2 } , y _ { 2 } )$ is computed as:

$$
D ( C _ { 1 } , C _ { 2 } ) = \sqrt { ( x _ { 1 } - x _ { 2 } ) ^ { 2 } + ( y _ { 1 } - y _ { 2 } ) ^ { 2 } }\tag{10}
$$

Fusion is performed if:

$$
D ( C _ { 1 } , C _ { 2 } ) < \delta\tag{11}
$$

where δ is the empirical distance threshold. The final confidence score of the fused bounding box is the maximum of the two score values. If the distance condition is not satisfied, the original bounding boxes are preserved. This formulation ensures robustness to moderate spatial discrepancies between modalities while maintaining reliable detection outputs in the presence of partial or asymmetric detections.

## IV. EXPERIMENTAL TESTS

After computing the performance metrics of each model, the results gained from tests in real-world scenarios are analyzed and compared, both visually and empirically. The analysis will cover all the phases with both strategies: Early and Late Fusion.

## A. Acquisition Scenarios

After recording all data in the two modalities, some results are presented as frames of the collected videos, discussing both benefits and limitations of the two systems in all the conditions tested.

The metric considered in external tests is the confidence score calculated by the model during detection. This value indicates the probability of an unidentified flying object belonging to the UAV class and it has allowed us to perform an early qualitative review of the system in different environmental conditions. In the final part of the section, some quantitative results are also presented in a dedicated table, in which other metrics are considered and analyzed, in order to give a review of the two frameworks as complete as possible. Different

![](images/e375a1bc7012b984857624bdb76f326ef4eb7dd64e080a10c48d6c545b952604.jpg)  
Fig. 13: The outdoor testing field © Google Earth Pro. The takeoff field is highlighted in green. Inside it, in the red area, the master drone used for detection takes off. In the rest of the yellow scenario, a target drone flies at different altitudes, in order to test the capabilities of the detection models.

lighting scenarios were faced during the experiments. For safety reasons, a maximum distance of 60 meters and an altitude of 20 m were set for all the tests. Zooming in was also used in borderline cases, where the background was wooded and the drone was camouflaged among the leaves, thus difficult to detect. Confidence scores were increasing when zooming in and the drone was still detected with all degrees of zoom and at great distances. However, detection at 1X is sometimes intermittent, whereas 2X and 4X work reliably, making the system more stable. Furthermore, the score drops when the target flies in front of an obstacle. At those hours, temperatures were lower, which prevented disturbances from light and heat. On the other hand, the sky background greatly improved the confidence score, even with clouds. In fact, the cloud formation let light pass through and the environment was at least sufficiently illuminated, enhancing the contrast between the target UAV and the sky color.

Late-fusion technique was more challenging due to the management of two independent models. As can be seen, the stream was aligned during detection, in order to have both frames in the same plane with a horizontal sync. This method proves particularly useful in environments in which one of the two modalities is less reliable. In fact, the best of the two helps the other to perform detection, thanks to the weighted average strategy developed. As before, sky offers higher confidence scores, due to the better contrast between the UAV and the background. However, some small issues have been discovered with trees and terrain setup, as they generate clutter, a phenomenon that challenges the optical sensors. In addition, leaves make detection even more difficult due to the openings inside the trees, which let light in, reducing the possibility of detection. Moreover, since we recorded on summer days, the heat and brightness affected the infrared sensors, generating noise. Another aspect which should be taken into consideration regards UAV movements. If the drone is stationary or changes direction quickly, it can be lost. However, if the drone is in the air, it is much better identified, even at great distances.

TABLE III: Summary of the experimental evaluation conducted at different hours of the day, under different environmental conditions. The target drone was moving at different altitudes and distances from the master drone, while covering two main backgrounds: trees and sky. In the same way, three zoom levels have been tested.
<table><tr><td>Time</td><td>Fusion</td><td>Zoom</td><td>Background</td><td>Distance</td><td>Altitude</td></tr><tr><td>09:00</td><td>Early</td><td>1×,2×,4×</td><td>Trees</td><td>5, 25, 60 m</td><td>5, 10, 20 m</td></tr><tr><td>09:00</td><td>Early</td><td>1×,2×,4×</td><td>Sky</td><td>5,25,60 m</td><td>5,10,20 m</td></tr><tr><td>09:00</td><td>Late</td><td>1×,2×,4×</td><td>Trees</td><td>5,25,60 m</td><td>5, 10, 20 m</td></tr><tr><td>09:00</td><td>Late</td><td>1×,2×,4×</td><td>Sky</td><td>5,25,60 m</td><td>5, 10, 20 m</td></tr><tr><td>12:00</td><td>Early</td><td>1×,2×,4×</td><td>Trees</td><td>5, 25, 60 m</td><td>5, 10, 20 m</td></tr><tr><td>12:00</td><td>Early</td><td>1×,2×,4×</td><td>Sky</td><td>5,25,60 m</td><td>5, 10, 20 m</td></tr><tr><td>12:00</td><td>Late</td><td>1×,2×,4×</td><td>Trees</td><td>5,25,60 m</td><td>5, 10, 20 m</td></tr><tr><td>12:00</td><td>Late</td><td>1×,2×,4×</td><td>Sky</td><td>5,25,60 m</td><td>5, 10, 20 m</td></tr><tr><td>15:00</td><td>Early</td><td>1×,2×,4×</td><td>Trees</td><td>5,25,60 m</td><td>5, 10, 20 m</td></tr><tr><td>15:00</td><td>Early</td><td>1×,2×,4×</td><td>Sky</td><td>5, 25, 60 m</td><td>5,10,20 m</td></tr><tr><td>15:00</td><td>Late</td><td>1×,2×,4×</td><td>Trees</td><td>5,25,60 m</td><td>5, 10, 20 m</td></tr><tr><td>15:00</td><td>Late</td><td>1×,2×,4×</td><td>Sky</td><td>5,25, 60 m</td><td>5, 10, 20 m</td></tr></table>

Late-Fusion requires two detection models, increasing inference time and computational cost. Each model has 21.5 MB of weights and processes a 640×640 input, resulting in approximately twice the model and inference requirements of RGT.

![](images/e8d05b296d907d406b50892daa08dd42139e0c336643dc378e0980d7c46fd39d.jpg)  
Fig. 14: Ground and sky comparisons at different times and resolutions. From the top left corner of the image, it is possible to see the first line, regarding Early Fusion RGT process on the ground at different zoom levels (1X, 2X, 4X) in different times of day (9:00, 12:00, 15:00). Following it, there is a second line showing the same fusion techniques applied for detecting over a sky background. Finally, the third and fourth lines contain images of tests conducted in the same conditions presented before while using Late Fusion instead.

## V. RESULTS

This section presents a qualitative and quantitative comparison between the two proposed fusion approaches. Table IV summarizes the validation metrics computed from both modalities. In particular, Late-Fusion requires two independent

TABLE IV: Comparison of mean metrics between Late-Fusion (RGB-IR) and Early-Fusion (RGT) strategies (where mR is mean Recall and mF1 is mean F1).
<table><tr><td>Fusion Method</td><td>mAP@50</td><td>mR</td><td>mF1-score</td></tr><tr><td>Late-Fusion (RGB-IR)</td><td>72%</td><td>77%</td><td>73.7%</td></tr><tr><td>Early-Fusion (RGT)</td><td>75.7%</td><td>83%</td><td>77.8%</td></tr></table>

detection models to process the RGB and IR inputs, resulting in increased inference time and computational cost compared to the Early-Fusion approach. and processes a 640×640 input, resulting in approximately twice the model and inference requirements of RGT.

The evaluation process for both modalities follows a common series of steps. In particular, validation was conducted on a pool of N = 400 spatially aligned images, while inference was performed using a default Intersection-over-Union (IoU) threshold of 0.5. The image pool comprises several images acquired in the same scenarios to those considered in the experiments, involving different backgrounds, flight altitudes and lighting conditions. All captured images were aggregated into a single evaluation pool, as considering the individual scenarios separately would have resulted in datasets that were too small to support a meaningful comparison. In addition, this diversity of acquisition conditions increases the variability of the evaluation pool, providing a more representative assessment of the model’s ability to generalize across different operating conditions.

The experimental setup includes various scenarios involving different times of day, environmental conditions, and flight altitudes, whereas the final comparison is presented only as a set of aggregated results. It is recommended that the authors report the detection results of the early-fusion and decisionlevel-fusion methods separately under different experimental conditions. For example, the results could be grouped according to time period, background type, flight altitude, target distance, or zoom level.

For the Early-Fusion (RGT) approach, predictions generated by the detector for each image were compared with the corresponding ground-truth annotations using an overlap mechanism. This step allowed the computation of global True Positive, False Positive and False Negatives in the entire pool of images.

$$
T P = \sum _ { i = 1 } ^ { N } T P _ { i } \qquad F P = \sum _ { i = 1 } ^ { N } F P _ { i } \qquad F N = \sum _ { i = 1 } ^ { N } F N _ { i }\tag{12}
$$

These quantities were then used to compute average precision, recall and F1-score.

In contrast, the Late-Fusion approach performs inference independently over each RGB-IR image pair. Then, a set of fused bounding boxes is produced by the weighted average algorithm described in Equation 9. These fused detections are defined as:

$$
\begin{array} { r } { \hat { \mathcal { D } } _ { i } = \left\{ { B } _ { f } ^ { ( 1 ) } , { B } _ { f } ^ { ( 2 ) } , \dots , { B } _ { f } ^ { ( M _ { i } ) } \right\} , } \end{array}\tag{13}
$$

This set was then compared with the set of ground-truth annotations $\mathcal { G } _ { i }$

$$
\mathcal { G } _ { i } = \left\{ G _ { i } ^ { ( 1 ) } , G _ { i } ^ { ( 2 ) } , \dots , G _ { i } ^ { ( N _ { i } ) } \right\} ,\tag{14}
$$

Since the image pair is composed of geometrically aligned RGB and IR frames, they share a single annotation file. Therefore, the label file is unique across both spectra. This allows the comparison between fused boxes and ground-truth. From the results shown in Table IV, Early-Fusion turned out to be the better of the two, achieving the best overall performance. This point also reinforces the results obtained from previous trainings in Table II. Despite the final outcome, Late-Fusion strategy still provides competitive results, demonstrating the effectiveness of combining complementary information for aerial target detection.

Finally, some detection results showing the overlapping step performed by the validation process are visible in Figure 15.

![](images/8b94279c532d99bcd7660a376c8c20dfc8dbc00af3cf90eb000e819902c27ac1.jpg)

![](images/26f0a8ea457a2b0e9ede31cac9c42bfc5cc15c8206bc0ab01d5ff6ec5971eb59.jpg)

![](images/aaf62b161a1f9b3069e13ce95b33bed8706ec91ab3af249957ee2e58810d316c.jpg)

![](images/d89d4c8e09cb08a112590fb47318659d0e027dd44cc9403a5284cf8d0d9df701.jpg)  
Fig. 15: Comparison between blue ground truth boxes and models predicted boxes (red for Late-Fusion and green for Early-Fusion).

## VI. CONCLUSION AND FUTURE WORK

It has been quite difficult to identify the best strategy of the two fusion techniques. In fact, both early and late fusions gave promising outputs. However, they could be exploited in different contexts. For example, late fusion might be advantageous at night, when the RGB counterpart is completely lost, whereas IR offers better performance and could support it. On the other hand, early fusion is the best for general use, as it improves detection by adopting channel blending, considering the tests conducted and the results acquired. Despite this advantage, early fusion is harder to implement and run, due to the need to train a dedicated model over a fused dataset and its dependence on an accurate pre-alignment process.

Although the system reached good overall results, both mathematically and empirically, the need for more multimodal data for future research remains crucial. In particular, the proposed solution was specifically designed for drone-to-drone detection architectures. For this reason, it is necessary to integrate this work with more data, in order to adapt this strategy to different detection pipelines.

Some future improvements to the project will surely involve an upgrade to the drone platform, enabling the use of newer and more accurate sensors for detection. This is indeed linked to the exploitation of different and complex image types, such as multispectral or hyperspectral images. However, the use of other sensor data should also be taken into consideration, since radar platforms are widely used for developing sensor fusion paradigms in modern aviation.

## REFERENCES

[1] J. Zhao, J. Zhang, D. Li, and D. Wang, “Vision-based anti-uav detection and tracking,” IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 12, pp. 25 323–25 334, 2022.

[2] F. Samadzadegan, F. Dadrass Javan, F. Ashtari Mahini, and M. Gholamshahi, “Detection and recognition of drones based on a deep convolutional neural network using visible imagery,” Aerospace, vol. 9, no. 1, p. 31, 2022. [Online]. Available: https: //doi.org/10.3390/aerospace9010031

[3] O. Skorka and R. Ispasoiu, “Tradeoffs with rgb-ir image sensors,” IEEE Transactions on Electron Devices, vol. 69, no. 6, pp. 2915–2923, 2022.

[4] Y. Sun, Y. Meng, Q. Wang, M. Tang, T. Shen, and Q. Wang, “Visible and infrared image fusion for object detection: A survey,” in Proceedings of International Conference on Image, Vision and Intelligent Systems 2023 (ICIVIS 2023). Springer Nature Singapore, 2024, pp. 236–248.

[5] U. Seidaliyeva, L. Ilipbayeva, K. Taissariyeva, N. Smailov, and E. T. Matson, “Advances and challenges in drone detection and classification techniques: A state-of-the-art review,” Sensors, vol. 24, no. 1, 2024. [Online]. Available: https://www.mdpi.com/1424-8220/24/1/125

[6] S. Scholes, A. Ruget, G. Mora-Mart´ın, F. Zhu, I. Gyongy, and J. Leach, “Dronesense: The identification, segmentation, and orientation detection of drones via neural networks,” IEEE Access, vol. 10, pp. 38 154–38 164, 2022.

[7] O. Escobar D´ıaz, A. Sanchez L ´ opez, M. B. Bern ´ abe Loranca, and´ R. Gonzalez Vel ´ azquez, “Real time drone object tracking using his-´ tograms of oriented gradients and particle filters,” in 2016 Fifteenth Mexican International Conference on Artificial Intelligence (MICAI), Oct 2016, pp. 35–40.

[8] F. Liu, L. Yao, C. Zhang, T. Wu, X. Zhang, X. Jiang, and J. Zhou, “Boost uav-based object detection via scale-invariant feature disentanglement and adversarial learning,” arXiv preprint, vol. abs/2405.15465, pp. 1– 12, 2024.

[9] F. N. M. Zamri, T. S. Gunawan, M. Kartiwi, A. Pratondo, S. H. Yusoff, and Y. M. Mustafah, “Deep learning techniques for advanced drone detection systems: A comprehensive review of techniques, challenges and future directions,” Indonesian Journal of Electrical Engineering and Informatics, vol. 12, no. 4, pp. 931–944, 2024.

[10] S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time object detection with region proposal networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137– 1149, 2017.

[11] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, “Ssd: Single shot multibox detector,” in Computer Vision – ECCV 2016, B. Leibe, J. Matas, N. Sebe, and M. Welling, Eds. Cham: Springer International Publishing, 2016, pp. 21–37.

[12] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779–788.

[13] P. Hidayatullah, N. Syakrani, M. R. Sholahuddin, T. Gelar, and R. Tubagus, “Yolov8 to yolo11: A comprehensive architecture in-depth comparative review,” arXiv preprint, vol. abs/2501.13400, pp. 1–15, 2025.

[14] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu, Z. Yang, Y. Zhang, and D. Tao, “A survey on vision transformer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 87–110, 2023.

[15] S. V. Rebbapragada, P. Panda, and V. N. Balasubramanian, “C2fdrone: Coarse-to-fine drone-to-drone detection using vision transformer networks,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 6627–6633.

[16] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in Computer Vision – ECCV 2020, A. Vedaldi, H. Bischof, T. Brox, and J.-M. Frahm, Eds. Cham: Springer International Publishing, 2020, pp. 213–229.

[17] T. Sangam, I. R. Dave, W. Sultani, and M. Shah, “Transvisdrone: Spatiotemporal transformer for vision-based drone-to-drone detection in aerial videos,” in Proceedings of the 2023 IEEE International Conference on Robotics and Automation (ICRA 2023). London, UK: IEEE, 2023.

[18] R. Lu, Z. Zhuo, S. Wang, D. Zhang, B. Xi, X. Zheng, Z. Hou, and X. Yang, “MTCNet: Multimodal time-sensitive target fast concealment network for UAV aerial imagery via real-time mask generation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, p. 5643818, 2025.

[19] B. Tang, R. Lu, X. Yang, Y. Li, Y. Li, D. Zhang, and S. Chen, “R2PLoc: A region-to-point UAV visual geo-localization framework leveraging hierarchical semantic representation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, p. 5643818, 2025.

[20] K. AlDosari, A. Osman, O. Elharrouss, S. Almaadeed, and M. Z. Chaari, “Drone-type-set: Drone types detection benchmark for drone detection and tracking,” CoRR, vol. abs/2405.10398, 2024. [Online]. Available: https://dblp.org/rec/journals/corr/abs-2405-10398.html

[21] M. Bilal, S. Rehmat, N. Akhtar, and H. Gulzar, “Cost effective drone detection with deep neural networks for day and night surveillance,” in 2024 International Conference on Frontiers of Information Technology (FIT), 2024, pp. 1–6.

[22] M. W. Ashraf, W. Sultani, and M. Shah, “Dogfight: Detecting drones from drone videos,” in Proceedings of the IEEE/CVF Conference on

Computer Vision and Pattern Recognition (CVPR), 2021, pp. 7067– 7076.

[23] A. Sengupta, L. Cheng, and S. Cao, “Robust multiobject tracking using mmwave radar-camera sensor fusion,” IEEE Sensors Letters, vol. 6, no. 10, pp. 1–4, 2022.

[24] K. Yang, W. Xiang, Z. Chen, J. Zhang, and Y. Liu, “A review on infrared and visible image fusion algorithms based on neural networks,” Journal of Visual Communication and Image Representation, vol. 101, p. 104179, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1047320324001342

[25] Y. Wang, Y. Chen, J. Choi, and C.-C. J. Kuo, “Towards visible and thermal drone monitoring with convolutional neural networks,” APSIPA Transactions on Signal and Information Processing, vol. 8, p. e5, 2019.

[26] S. Li, X. Kang, and J. Hu, “Image fusion with guided filtering,” IEEE Transactions on Image Processing, vol. 22, no. 7, pp. 2864–2875, 2013.

[27] Y. Liu, Z. Wu, X. Han, Q. Sun, J. Zhao, and J. Liu, “Infrared and visible image fusion based on visual saliency map and image contrast enhancement,” Sensors, vol. 22, no. 17, p. 6390, 2022.

[28] J. Ma, L. Tang, F. Fan, J. Huang, X. Mei, and Y. Ma, “Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer,” IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 7, pp. 1200–1217, 2022.

[29] L. Tang, Y. Deng, Y. Ma, J. Huang, and J. Ma, “Superfusion: A versatile image registration and fusion network with semantic awareness,” IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 12, pp. 2121–2137, 2022.

[30] S. Ozer and A. P. Ndigande, “Visirnet: Deep image alignment for<sup>¨</sup> uav-taken visible and infrared image pairs,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–11, 2024.

[31] H. Hao, Y. Peng, Z. Ye, B. Han, W. Tang, W. Kang, X. Zhang, Q. Li, and W. Liu, “Tmrgbt-d2d: A temporal misaligned rgb-thermal dataset for drone-to-drone target detection,” Drones, vol. 9, no. 10, p. 694, 2025.

[32] Y. Cao, J. Bin, J. Hamari, E. Blasch, and Z. Liu, “Multimodal object detection by channel switching and spatial attention,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2023, pp. 403–411.

[33] F. Xu, Z. Lin, C. Chen, R. Yang, and H. Zhai, “Anti-drone detection in aerial multispectral images,” in IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium, 2024, pp. 9388–9391.

[34] T. Zhang, R. Lu, X. Yang, D. Zhang, Y. Li, X. Xie, and Y. Li, “CCSFuse: Collaborative compensation and selective fusion for UAV-based RGB-IR object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, p. 5400314, 2026.

[35] Y.-T. Chen, J. Shi, Z. Ye, C. Mertz, D. Ramanan, and S. Kong, “Multimodal object detection via probabilistic ensembling,” in Computer Vision – ECCV 2022, S. Avidan, G. Brostow, M. Cisse, G. M. Farinella,´ and T. Hassner, Eds. Cham: Springer Nature Switzerland, 2022, pp. 139–158.

[36] D. Guan, Y. Cao, J. Liang, Y. Cao, and M. Y. Yang, “Fusion of multispectral data through illumination-aware deep neural networks for pedestrian detection,” arXiv preprint, vol. abs/1802.09972, pp. 1–9, 2018.

[37] S. Zhang, T. Xie, Y. Wang, Y. Liu, Y. Dou, and S. Wang, “Sf-yolo: Rgb-t fusion object detection in uav scenes,” in 2023 8th International Conference on Image, Vision and Computing (ICIVC), 2023, pp. 51–59.

[38] H. Li and X.-J. Wu, “Densefuse: A fusion approach to infrared and visible images,” IEEE Transactions on Image Processing, vol. 28, no. 5, pp. 2614–2623, May 2019.

[39] L. Tang, J. Yuan, H. Zhang, X. Jiang, and J. Ma, “Piafusion: A progressive infrared and visible image fusion network based on illumination aware,” Information Fusion, vol. 83–84, pp. 79–92, 2022.

[40] Q. Yang, Y. Zhang, Z. Zhao, J. Zhang, and S. Zhang, “Iaifnet: An illumination-aware infrared and visible image fusion network,” IEEE Signal Processing Letters, vol. 31, pp. 1374–1378, 2024.

[41] H. Li and Y. Fu, “Fcdfusion: A fast, low color deviation method for fusing visible and infrared image pairs,” Computational Visual Media, vol. 11, no. 1, pp. 195–211, 2025.

[42] J. Wagner, V. Fischer, M. Herman, and S. Behnke, “Multispectral pedestrian detection using deep fusion convolutional neural networks,” in 24th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), vol. 587, 2016, pp. 509– 514.

[43] T. Zhao, M. Yuan, F. Jiang, N. Wang, and X. Wei, “Removal then selection: A coarse-to-fine fusion perspective for rgb-infrared object detection,” IEEE Transactions on Intelligent Transportation Systems, vol. 27, no. 2, pp. 2504–2519, 2026.

[44] L. Yang, R. Ma, and A. Zakhor, “Drone object detection using rgb/ir fusion,” in Electronic Imaging Symposium, Computational Imaging XX Conference, vol. 2022, 2022, pp. 1–6.

[45] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in 2017 IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2242– 2251.

[46] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1125– 1134.

[47] K. Gadzicki, R. Khamsehashari, and C. Zetzsche, “Early vs late fusion in multimodal convolutional neural networks,” in 2020 IEEE 23rd International Conference on Information Fusion (FUSION). IEEE, 2020, pp. 1–6.

[48] X. Liu, Z. Rozsypalek, and T. Krajn ´ ´ık, “Self-supervised learning for fusion of ir and rgb images in visual teach and repeat navigation,” in 2023 European Conference on Mobile Robots (ECMR). IEEE, 2023, pp. 1–7.

[49] T. Tian, J. Cai, Y. Xu, Z. Wu, Z. Wei, and J. Chanussot, “Rgb-infrared multi-modal remote sensing object detection using cnn and transformer based feature fusion,” in IGARSS 2023 – 2023 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2023, pp. 5728– 5731.

[50] G. Wang, T. Zhang, J. Cheng, S. Liu, Y. Yang, and Z. Hou, “Rgb-infrared cross-modality person re-identification via joint pixel and feature alignment,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2019, pp. 3622–3631.

[51] D. Tavaris, A. De Zan, A. Toma, G. L. Foresti, I. Scagnetto, and N. Martinel, “Multi-perspective rgb and infrared dataset for uav detection,” IEEE Access, vol. 13, pp. 168 792–168 803, 2025.

[52] A. De Zan, D. Tavaris, G. L. Foresti, I. Scagnetto, N. Martinel, and A. Toma, “Autonomous real-time drone-to-drone visual detection by onboard hardware platform,” Integrated Computer-Aided Engineering, vol. 33, no. 3, p. 10692509261452776, 2026.

[53] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics yolov8 (v8.3.112),” https://github.com/ultralytics/ultralytics, 2025, accessed: 2026-05-17.

[54] H.-C. Cheng, I. Ben-David, and S.-T. Wu, “Five-primary-color lcds,” Journal of Display Technology, vol. 6, no. 1, pp. 3–7, 2010.

[55] Minnesota Geospatial Information Office, “Color-infrared (cir) imagery,” https://www.mngeo.state.mn.us/chouse/airphoto/cir.html, accessed: 2026-05-17.

[56] U. G. S. (USGS), “What do the different colors in a color infrared (cir) aerial photograph represent?” accessed: 2026-05-05. [Online]. Available: https://www.usgs.gov/faqs/ what-do-different-colors-a-color-infrared-aerial-photograph-represent

[57] Y. Zhang, Y. Liu, P. Sun, H. Yan, X. Zhao, and L. Zhang, “Ifcnn: A general image fusion framework based on convolutional neural network,” Information Fusion, vol. 54, pp. 99–118, 2020.

![](images/6a631998d2ed38a403c9fb2d46f0afe308920a2d6b525bce918e19f74627e877.jpg)

Alberto De Zan received his Bachelor Degree (cum laude) in Computer Science at the University of Udine, Italy in 2023, on a thesis with title ”Drones Cluster Development for Air Surveillance and Defense”. He continued his studies with a Double Degree program in Artificial Intelligence & Cybersecurity at the University of Udine (Italy) and Klagenfurt (Austria). He graduated with maximum marks in 2025 on a thesis with title ”Unmanned Aerial Vehicles Detection adopting Multiple Sensor Fusion”. He is currently working as a Research

Fellow on some projects regarding Computer vision, Radar Platforms, Drone Detection, Open Source Intelligence and Quantum Computing in the Autonomous System Lab of the University of Udine.

![](images/8e244e51b8f22e2aa9de095f6020cf6964432ecc5307612919e6830bc02bf5ff.jpg)

Ivan Scagnetto received the Ph.D. in Computer Science in 2002 from the University of Udine. Since 2005, he has been an assistant professor at the same University. His research interests range from software certification by means of formal techniques based on Computer Aided Formal Reasoning to mobile computing, information retrieval with mobile devices, and IoT systems. He is author of several scientific papers with national and international peer review and he has been scientific reviewer of international journals and conferences. He worked in several research projects funded by the European Commission, by MUR (the Italian Ministry of University and Research), by the INTERREG programme, and by the PNRM (National Military Research Plan).

![](images/0b0a1aa92b6a4fb6c9a809c6aa311d58b04e571f463d4a27c8e689e65662079a.jpg)

Denis Tavaris is a Ph.D. student in Artificial Intelligence at the University of Udine, specializing in Computer Vision applied to robotics. His research focuses on the development of autonomous and cooperative multi-drone systems for environmental monitoring, integrating vision-based technologies and sensor fusion. Denis is involved in several research projects on autonomous systems, multi-domain robotics (aerial, surface, and underwater), and AI-driven environmental analysis. He has published papers in international journals and

conferences, with a focus on efficient and scalable solutions for autonomous monitoring. Denis has a strong background in data analysis, system design, and software development, with expertise in Python, Linux, and sensor integration.

![](images/60658c573a1614967dece42c98c885c349d8c8f3af8efb535f9e65dbf9dc3469.jpg)

Gian Luca Foresti received the Laurea degree (cum laude) in electronic engineering and the Ph.D. degree in computer science from the University of Genoa, Genoa, Italy, in 1990 and 1994, respectively. He is a Full Professor of computer science with the University of Udine, Udine, Italy, where he is the Deputy Director of the Department of Mathematics, Computer Science and Physics. Since 2006, he has been a Visiting Professor in artificial vision with the University of Klagenfurt, Klagenfurt, Austria. From 2000 to 2009, he was the appointed Italian

Member of the NATO RTO Information System Technology Panel. He has authored over 300 papers published in international journals and international conferences. He has co-edited several books in the field of multimedia and video surveillance. His current research interests include computer vision and image processing, multisensor data and information fusion, pattern recognition, and neural networks. Mr. Foresti was a recipient of the Best IEEE Vehicular Electronics Paper in 2002. He was the Finance Chair of the 11th IEEE Conference on Image Processing, the General Chair of the 16th International Conference on Image Analysis and Processing and the 8th IEEE International Conference on Advanced Video and Signal Based Surveillance. He has been the Guest Editor of a Special Issue of the Proceedings of the IEEE on “Video Communications, Processing and Understanding for Third Generation Surveillance Systems.” He is a member of IAPR and CVPL.