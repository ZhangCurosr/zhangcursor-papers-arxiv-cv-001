# Det-LIME: Detector-Aware, Multi-Instance Local Interpretable Model-Agnostic Explanations for Automated Marine Mammal Detection

Jiayi Zhou<sup>1</sup>, David W. Johnston<sup>2</sup>, and Brinnae Bent<sup>∗1</sup>

<sup>1</sup> Pratt School of Engineering, Duke University, Durham, North Carolina, United States

<sup>2</sup> Division of Marine Science and Conservation, Nicholas School of the Environment, Duke University, Durham, North Carolina, United States

## Abstract

Despite the rapid uptake of black-box object detectors in marine mammal research and monitoring, explainability techniques are rarely integrated into conservation workflows. Furthermore, most classification-oriented explainability tools are ill-suited to detection tasks involving imagery of social organisms or those with colonial life histories, as they ignore multiple detections within a scene and produce single-instance outputs that blur evidence across individuals. These methods also generate low-resolution, often biologically irrelevant visuals, limiting their utility for debugging, targeted data augmentation, and refined data collection.

We proposed Det-LIME, a detector-aware, multi-instance adaptation of Local Interpretable Model-Agnostic Explanations (LIME) that produced instance-specific, box-aligned explanations by combining per-detection weighting, a proximity kernel that emphasizes regions near each box, and Intersection-over-Union-based matching to track the same instance across perturbations. We evaluated Det-LIME on aerial drone imagery for harbor seal detection, with an additional seabird case study to assess generality, and compared it with vanilla LIME, Stabilized LIME, Deterministic LIME, and gradient-based attribution methods.

Using the Attribution Ratio and Max Saliency Hit Rate metrics, we showed that Det-LIME consistently improved multi-instance attribution. In practice, these higher-resolution, instance-aware explanations provide insight into model outputs and support post-processing, debugging, and actionable improvements in modeling and data collection or augmentation.

Keywords: Object detection; computer vision; explainable artificial intelligence; marine mammals; ecological monitoring

## 1 Introduction

Computer vision and artificial intelligence methods now underpin many ecological workflows: from tracking wildlife populations and assessing habitat change to informing management actions and protected-species policy (Gray et al., 2022; Růžička et al., 2023). Classification and object detection methods are commonly used within computer vision workflows to automatically identify, count, and locate species and landscape features in large image and video datasets, enabling scalable biodiversity monitoring and informed conservation decisions (Tabak et al., 2018; Choiński et al., 2021).

As detectors mature, accuracy is no longer the only limiting factor. Practitioners need evidence that predictions are grounded in animal morphology (not background shortcuts), clear views of failure modes, and defensible rationale when setting thresholds that carry operational and ethical costs (Gevaert, 2022; Buchelt et al., 2024). This is particularly true for what people refer to as "black-box" models like neural networks with opaque internal "reasoning". Although researchers who develop and train their own models may have access to the model architecture, this level of access is not always available in applied ecological workflows. Models may be developed by external collaborators, incorporated into third-party software, accessed through hosted services, or deployed in monitoring systems that provide predictions without exposing internal feature maps or gradients. Even when the model architecture is available, explanation methods that rely on internal representations generally require architecture-specific choices, including identifying appropriate feature layers and linking the prediction of interest to the corresponding activations and gradients. These requirements can vary across detector architectures and make the same explanation procedure more dificult to apply consistently across models. Despite these needs, explainability is rarely integrated into ecological research or conservation workflows.

Explainability helps provide insights into model predictions by revealing which input features and interna representations drive a model’s decision-making process. One such explainable AI (XAI) technique is Local Interpretable Model-Agnostic Explanations (LIME) (Ribeiro et al., 2016). LIME generates perturbed versions of the input image and observes how the black-box model’s predictions change. It then fits a simple linear model to these perturbations to approximate the local decision boundary, using the linear coeficients to estimate the contribution of each image region. This provides a locally interpretable explanation of the model’s prediction. In the context of marine mammal ecology, LIME is particularly well-suited because it is model agnostic, local, and produces spatially grounded, human-understandable overlays. Because it operates from model inputs and outputs, it does not require users to select internal feature layers or adapt the explanation procedure to specific internal representations. This provides a consistent way to examine predictions across detector architectures, including settings where internal model information is unavailable or dificult to use directly for explanation. These properties align with how ecologists audit individual predictions and allow visual verification that predictions are based on animal morphology rather than background features. LIME’s portability, locality, and communicability make it well-matched to ecological decision-making.

Classification models produce a single, image-level prediction, and explanation methods in this setting attribute evidence to that overall decision. In contrast, multi-object detection systems often produce multiple localized predictions within a single image, where each prediction is defined by a bounding box, a class label, and a confidence score. This distinction is important in ecological images, where multiple animals often appear together, crowded within a single frame. Conventional LIME approaches are designed for per-image classification, not instance-level detection in complex and often unpredictable ecological scenes. In practice, conventional LIME techniques explain a label’s image-level score rather than a detector’s per-box prediction; their perturbations are often coarse, engulfing multiple small animals and blurring evidence across neighboring boxes; they have no mechanism to remain focused on predicted objects as perturbations alter proposals; and they weight these perturbations globally, rather than centering on the queried box. The result is blocky heatmaps that attribute background textures more than morphology in predictions, ofering limited guidance for threshold setting, post-processing, or targeted data collection. Explanation methods for multi-object detection in ecological imagery must provide instance-level, spatially precise attributions that are stable for a given detection while accounting for nearby objects and complex background patterns.

To address these limitations, we introduce Det-LIME, a detector-aware, multi-instance adaptation of LIME designed for object detection in ecological imagery. Det-LIME produces instance-specific, box-aligned explanations by fitting a model that approximates the detector’s behavior near a target detection. The framework (i) handles multiple detections through instance weighting, (ii) emphasizes regions near each box with a proximity kernel, and (iii) tracks the same instance under perturbations using intersection over union (IoU)-based matching. The goal was actionable clarity: higher-resolution, per-animal evidence that aligns with how ecologists review detections and ultimately make policy-relevant decisions.

## 2 Related Work

## 2.1 Explainability in Ecology

Explainable AI (XAI) has shown substantial impact across healthcare and autonomous driving domains, where it reveals diagnostically relevant regions in medical images (Brima and Atemkeng, 2024) or justifies robust road detection (Mankodiya et al., 2022). In ecology, however, XAI remains comparatively underdeveloped (Saarela and Podgorelec, 2024). Recent reviews have discussed the need for explainability (Gevaert, 2022; Buchelt et al., 2024), especially in high-stakes conservation contexts; however, most ecological workflows, especially those involving computer vision tasks, employ “black box” models without any integrated explainability.

In classification and regression tasks, XAI methods provide insight into which features or regions drive model predictions. For instance, Local Interpretable Model-agnostic Explanations (LIME) has been used to provide localized insight in species distribution modeling (Ryo et al., 2021) and to highlight influential regions for bird image classification (Kumar and Kondaveeti, 2024), while SHapley Additive exPlanations (SHAP) has been applied to identify environmental drivers of vegetation health (Britton et al., 2024). These approaches help researchers understand feature importance at the image or data level.

For object detection, techniques such as Grad-CAM (Yamauchi and Ishikawa, 2022) and saliency maps (Petsiuk et al., 2021) have been adapted to reveal regions associated with individual detections. However, these gradient-based approaches typically require access to the model’s internal architecture or feature maps, limiting their generality. Model-agnostic methods like conventional LIME have been previously applied to object detection but generally explain only a single detected instance per image (Sejr et al., 2021). Detecting multiple small species in ecological images is particularly challenging and is often required for monitoring marine mammal species that exhibit social clustering in either aquatic or terrestrial habitats. Current methods cannot provide instance-level, spatially precise explanations across multiple detections without access to model internals, creating a need for a model-agnostic approach that supports interpretable and reliable ecological object detection for monitoring and conservation.

## 2.2 Local Interpretable Model-Agnostic Explanations (LIME)

Explainability methods such as LIME provide insight into a model’s behavior for a single prediction of a black-box model (Ribeiro et al., 2016). LIME generates perturbed versions of the input image and observes how the model’s predictions change across these perturbations. It then fits a simple interpretable model, typically linear regression, to approximate the local decision boundary and identify the image regions that most influence the prediction.

However, its reliance on random sampling can produce unstable explanations, prompting the development of variants such as Deterministic LIME (DLIME) and Stabilized LIME (S-LIME) that address these instabilities (Zafar and Khan, 2021; Zhou et al., 2021). DLIME uses a deterministic hierarchical approach to generate perturbations, while S-LIME aggregates the results of multiple runs to improve consistency. These adaptations make local explanations more reliable.

Despite these advances, LIME and its variants are designed for classification tasks, where a single prediction is produced for the entire image. Multi-object detection presents a diferent challenge, particularly in ecological applications, because the model produces multiple predictions, each corresponding to a distinct object. Traditional LIME cannot provide instance-level explanations in this setting.

We addressed this limitation by developing Det-LIME, a detector-aware adaptation of LIME that extends local explanations to multi-object detection. This approach preserves the model-agnostic strengths of LIME while providing instance-level, spatially specific explanations for ecological imagery.

## 3 Methods

## 3.1 Det-LIME: Enhanced Multi-Instance LIME for Object Detection

![](images/350c3d2de8d6bd554bc78d5ace0821b7e973081c5c406a07208c1cbad233911e.jpg)  
Figure 1: An illustration of the Det-LIME explanation process, applied to the output of a Faster R-CNN model detecting seals. This diagram illustrates our method, starting from an image with multiple detections (a) that was partitioned into superpixels (b). After weighting each detected instance (c), the system scored numerous perturbed samples (d) and used them to fit a model that approximated the detector’s behavior near a target detection (e). This yielded raw importance scores (f) for each superpixel, which were then spatially focused (g) to produce the final, box-aligned relevance map (h) that highlighted the key visual evidence for each detection.

## model-agnostic

We extended the Local Interpretable Model-agnostic Explanations (LIME) framework to multi-instance object detection in imagery generated during drone surveys of harbor seals in glacial fjords in Glacier Bay National Park and Preserve, (Aghakishiyeva et al., 2025), alongside a complementary case focused on drone surveys of colonial seabirds (Hayes et al., 2021) (details on both applications below). Our approach introduced three key enhancements: an improved segmentation and filtering pipeline, a flexible multi-instance weighting mechanism, and a refined spatial focusing strategy. These refinements generated explanations that were more detailed and better aligned with the objects detected by the model (Figure 1h). The resulting explanation maps highlighted image regions that most influenced the model’s detection decision, which could occur both within and around the predicted bounding box depending on the visual features and contextual cues used by the detector.

## 3.1.1 Segmentation and Filtering

The first enhancement refined image segmentation. Given an input image, we partitioned it into superpixels using an enhanced Simple Linear Iterative Clustering (SLIC) algorithm with additional hyperparameters for increased flexibility. Each superpixel satisfied constraints on compactness, minimum size, and connectivity,

and the superpixels collectively covered the image without overlap. An optional filtering step removed visually insignificant segments, such as regions that were uniformly dark. This reduced background noise and improved the fidelity of the final explanation.

## 3.1.2 Multi-Instance Weighting and LIME Adaptation

LIME is designed to explain a single prediction, while multi-object detection models typically return multiple detections of the same class within a single image. To bridge this gap, we extended LIME to a multi-instance setting by explicitly defining target instance selection, instance weighting, perturbation scoring, and linear explanation fitting.

Target Instance Identification and Weighting Given an input image, the object detection model produces a set of detections

$$
R = \{ ( b _ { j } , c _ { j } , p _ { j } ) \} , \quad j = 1 , \ldots , M ,
$$

where $b _ { j }$ denotes the �-th predicted bounding box, $c _ { j }$ the associated class label, $p _ { j }$ the detection confidence, and � the total number of detections. For a target class $c ^ { * }$ and confidence threshold �, we define the set of target instances as

$$
{ \cal { T } } = \{ b _ { j } \in R \mid c _ { j } = c ^ { * } , ~ p _ { j } \geq \theta \} .
$$

This set contains all suficiently confident detections belonging to the class of interest.

Each target instance $b _ { i } \in \mathcal { I }$ was assigned a weight $w _ { i }$ that determined its contribution to the explanation. We considered three weighting strategies:

$$
w _ { i } = \left\{ \begin{array} { l l } { \displaystyle \frac { p _ { i } } { \sum _ { k = 1 } ^ { K } p _ { k } } , } & { \mathrm { C o n f i d e n c e - b a s e d ~ w e i g h t i n g } } \\ { \displaystyle \frac { \mathrm { a r e a } ( b _ { i } ) } { \sum _ { k = 1 } ^ { K } \mathrm { a r e a } ( b _ { k } ) } , } & { \mathrm { A r e a - b a s e d ~ w e i g h t i n g } } \\ { \displaystyle \frac { 1 } { K } , } & { \mathrm { U n i f o r m ~ w e i g h t i n g } } \end{array} \right.
$$

Here, $K = | { \boldsymbol { \mathcal { I } } } |$ denotes the number of target instances. These strategies allowed the explanation to emphasize detections according to model confidence, spatial extent, or equal contribution.

Multi-Instance Perturbation and Scoring Following the LIME framework, we generated a set of perturbed images by masking diferent subsets of superpixels. Let $I _ { m } ^ { \prime }$ denote the �-th perturbed image. Applying the detector to $I _ { m } ^ { \prime }$ yielded a corresponding set of detections

$$
R _ { m } = \{ ( b _ { j } , c _ { j } , p _ { j } ) \} ,
$$

defined analogously to $R .$

For each target instance $b _ { i } \in \mathcal { I }$ , we computed an instance-level score $s _ { i } ^ { ( m ) }$ that measured how well the instance was preserved under the perturbation:

$$
s _ { i } ^ { ( m ) } = w _ { i } \cdot \operatorname* { m a x } _ { \begin{array} { c } { { b _ { j } \in R _ { m } } } \\ { { c _ { j } = c ^ { * } } } \\ { { \mathrm { I o U } ( b _ { i } , b _ { j } ) > \tau } } \end{array} } \Big ( \operatorname* { m i n } ( p _ { j } , 0 . 9 5 ) \cdot ( 0 . 7 + 0 . 3 \cdot \mathrm { I o U } ( b _ { i } , b _ { j } ) ) \Big ) .
$$

Here, $\mathrm { I o U } ( b _ { i } , b _ { j } )$ quantifies the spatial overlap between the original instance $b _ { i }$ and a detected bounding box $b _ { j }$ in the perturbed image:

$$
\mathrm { I o U } ( b _ { i } , b _ { j } ) = \frac { \mathsf { a r e a } ( b _ { i } \cap b _ { j } ) } { \mathsf { a r e a } ( b _ { i } \cup b _ { j } ) } .
$$

The overlap threshold � filtered out detections with minimal spatial correspondence to the original instance, ensuring that only perturbed detections that could be meaningfully matched back to the queried object contributed to the attribution score. For matched detections, confidence values above 0.95 were capped at 0.95 for scoring rather than removed from the analysis. Thus, for example, a perturbed detection with confidence 0.98 contributed a confidence value of 0.95 to the instance-level score. Detection confidence was capped at 0.95 to prevent a single highly confident prediction from dominating the multi-instance aggregation. This cap helped keep instance contributions comparable across detections within the same image and allowed the final attribution map to remain sensitive to spatial perturbations rather than being driven by the detector’s confidence scale. The overlap-dependent term assigned higher scores to detections that better aligned with the original instance. The confidence cap of 0.95 and the IoU weighting values of 0.7 and 0.3 were specific to our multi-instance extension and were not inherited from the original LIME formulation. These values were selected as reasonable default settings based on initial pilot runs and stability considerations.

The overall score for the perturbed image was obtained by aggregating contributions from all target instances:

$$
S ^ { ( m ) } = \sum _ { i \in \cal { I } } s _ { i } ^ { ( m ) } .
$$

This scalar score reflected the extent to which the set of target detections was preserved under the perturbation and served as the response variable for LIME.

LIME Linear Model Fitting Superpixel importance was estimated by fitting a weighted linear model following the standard LIME formulation:

$$
g ( z ) = \beta _ { 0 } + \sum _ { j = 1 } ^ { N } \beta _ { j } z _ { j } , \quad z _ { j } \in \{ 0 , 1 \} ,
$$

where � is the number of superpixels, $z _ { j } = 1$ indicates that superpixel $S _ { j }$ was present in the perturbed image, and $z _ { j } = 0$ indicates that it was masked. The coeficient $\beta _ { j }$ represents the contribution of superpixel $S _ { j }$ to preserving the aggregated multi-instance score $S ^ { ( m ) }$ , thereby capturing its importance across all target detections. A linear surrogate was used because each coeficient can be directly associated with an individual superpixel, allowing its contribution to the prediction to be mapped back to a specific image region. The surrogate approximates the detector only within the local set of perturbations and does not assume that the underlying detector itself is linear.

## 3.1.3 Spatial Focusing and Pixel-Level Explanation

We introduced a spatial focusing mechanism that translated superpixel-level LIME scores into a refined, pixel-level saliency map. For each pixel (�, �), two key factors were considered for each target instance $b _ { i } \in \mathcal { I }$

1. Superpixel overlap with the instance: Each superpixel $S _ { j }$ may cover part of a target instance. The fraction of the superpixel overlapping with $b _ { i }$ was computed as

$$
\mathrm { o v e r l a p } ( S _ { j } , b _ { i } ) = \frac { | S _ { j } \cap b _ { i } | } { | S _ { j } | } .
$$

This term ensured that pixels within superpixels that covered the object received higher importance.

2. Spatial proximity to the instance center: Pixels closer to the center of a bounding box were assumed to be more informative. We defined a distance-based weight

$$
\mathrm { d i s t a n c e \_ w e i g h t } ( x , y ; b _ { i } ) = \mathrm { e x p } \Bigg ( - \frac { d ( ( x , y ) , \mathrm { c e n t e r } ( b _ { i } ) ) } { r _ { i } / 3 } \Bigg ) ,
$$

where $d ( \cdot , \cdot )$ is the Euclidean distance and $r _ { i }$ is the diagonal length of $b _ { i }$ . This factor gradually decreased the importance of pixels farther from the object center, emphasizing central regions while still assigning partial credit to surrounding pixels.

The final pixel-level explanation combined these factors with the superpixel importance $\beta _ { j }$ from the LIME model and the instance weight $w _ { i }$ to form a unified multi-instance saliency map:

$$
E ( x , y ) = \operatorname* { m a x } _ { i \in { \mathcal { I } } } \operatorname* { m a x } _ { S _ { j } \notin B } \Big ( \beta _ { j } \cdot w _ { i } \cdot \mathrm { o v e r l a p } ( S _ { j } , b _ { i } ) \cdot \mathrm { d i s t a n c e } _ { - } \mathrm { w e i g h t } ( x , y ; b _ { i } ) \Big ) ,
$$

where � denotes excluded black or filtered segments. The inner maximization selected the superpixel most relevant for each instance, while the outer maximization ensured that each pixel reflected the contribution of the most influential instance among all targets. Finally, the explanation map � was normalized to [0, 1] and could be optionally smoothed with a Gaussian filter.

This approach allowed the explanation to accurately highlight pixels that contributed to detecting multiple instances, accounting for both the superpixel-level importance and the spatial structure of the objects.

## 3.2 Evaluation Protocol

We assessed the performance of four local explanation methods to evaluate the efectiveness of the proposed approach: LIME (Ribeiro et al., 2016), its stabilized variants DLIME (Zafar and Khan, 2021) and SLIME (Zhou et al., 2021), and the proposed Det-LIME. LIME served as the baseline, as Det-LIME was built directly on its framework. DLIME and SLIME were included because they address LIME’s instability issues by providing more consistent explanations. Unlike the other methods, which generate explanations only for the single highest-confidence object in an image, Det-LIME generated a comprehensive attribution map for all detected objects. For this analysis, Det-LIME was configured in “Uniform” mode, assigning equal importance to every detected object. Under this configuration, each detected instance contributed equally to the aggregated explanation, independent of its confidence score or bounding-box size, thereby establishing a standardized baseline for multi-instance aggregation. While alternative strategies, such as confidence-based or area-based weighting, can be employed to emphasize detection reliability or object scale, respectively, they introduce additional dependencies on detector-specific outputs. In this work, uniform weighting was adopted to maintain a consistent, assumption-light aggregation framework across diverse instances.

Performance was assessed using two commonly used metrics that measure how well attributions match ground-truth object locations. These metrics were used to assess how well each explanation aligned with the detected object region, rather than as standalone measures of detector quality. Because Det-LIME explains detector outputs, these metrics necessarily depend on the detector’s predicted bounding boxes, class labels, and confidence scores. To reduce the influence of low-confidence detections, Det-LIME explanations were evaluated for detections above a confidence threshold of 0.5. In addition, explanation methods were compared using the same trained detector outputs, datasets, and detection targets, thereby reducing variation attributable to diferent model predictions or evaluation settings. With the detector output held constant across methods, diferences in the resulting scores were more related to attribution behavior than to detection performance.

Let $A \ \in \ \mathbb { R } ^ { H \times W }$ be the attribution map for a detected instance, normalized to [0, 1], and let $B \subset$ $\{ 1 , \ldots , H \} \times \{ 1 , \ldots , W \}$ denote the ground-truth bounding box for that instance. The first metric, Attribution

Ratio (AR), measures the proportion of important pixels that fall inside the object region, a metric widely adopted in prior work (Zhou et al., 2015; Selvaraju et al., 2019). We thresholded the attribution map to keep only mid-to-high importance pixels:

$$
\mathcal { P } _ { \tau } = \{ ( i , j ) \mid A _ { i j } \geq \tau \} ,
$$

and computed

$$
\mathrm { A R } = \frac { \vert \mathcal { P } _ { \tau } \cap B \vert } { \vert \mathcal { P } _ { \tau } \vert } .
$$

We used a threshold of $\tau = 0 . 3$ to remove low-intensity pixels in the attribution map. This threshold provided a conservative cutof that improved the visual focus of the attention map by removing difuse, low-signal pixels while retaining high-importance structures associated with the detected objects. This ensured that the metric reflected the alignment between the high-intensity pixels and the object regions. The second metric, Max Saliency Hit Rate (MSHR), measures whether the single most important pixel lies within the object box, another commonly used evaluation metric (Zhang et al., 2018). Let

$$
( i ^ { * } , j ^ { * } ) = \arg \operatorname* { m a x } _ { ( i , j ) } A _ { i j } ,
$$

then the hit for one instance is

$$
\mathrm { M S H R } = \mathbb { I } \big ( ( i ^ { * } , j ^ { * } ) \in B \big ) ,
$$

where I is the indicator function. The final score was calculated as the average over all instances.

Together, AR and MSHR captured whether attribution was concentrated within the object region and whether the most salient pixel fell inside the ground-truth box. These metrics were applied across all instances in the held-out test dataset to evaluate the methods’ performance in a realistic, multi-object scenario. For Det-LIME, each of its multiple explanations was matched to its corresponding ground-truth object before calculating the metrics. In parallel, the single explanation from LIME, DLIME, and SLIME was evaluated against every ground-truth object present in the image. These results were then aggregated across the entire dataset for each of the four methods. The Mean Attribution Ratio was obtained by averaging the individual Attribution Ratios across all instances, while the Max Saliency Hit Rate was the overall percentage of instances where the most salient pixel fell within the ground-truth box.

Given the diference between Det-LIME’s multi-object output and the single-object output of other methods, we adopted a second evaluation strategy. Evaluating all objects together can disadvantage singleinstance methods in multi-object images, whereas Det-LIME naturally provides attribution maps for all detected objects. To isolate this efect, the specific object selected by LIME, DLIME, and SLIME in each image was first identified. A one-to-one subset was then created by pairing these instances with Det-LIME’s corresponding explanations. Recalculating the metrics on this filtered dataset produced the Adjusted Mean Attribution Ratio and Adjusted Max Saliency Hit Rate, providing a perspective that allowed a fair comparison across methods by controlling for the influence of multiple instances in the evaluation metrics.

For both evaluation strategies, we reported the total number of evaluated detection instances used to compute each metric. An evaluated instance refers to a predicted detection that could be matched to a corresponding ground-truth bounding box and for which an attribution map was included in the metric calculation. To ensure that the evaluation focused on interpretable explanations for valid object detections, unmatched or invalid detections were excluded before aggregating the instance-level metrics. For the all-instance evaluation, the reported number represents all valid matched detections across the held-out test images for each method. For the adjusted evaluation, it represents the one-to-one subset of instances selected by the single-instance LIME-based methods and matched to the corresponding Det-LIME explanations. This distinction is important because the all-instance evaluation reflects each method’s native explanation setting, whereas the adjusted evaluation controls for diferences between multi-instance and single-instance explanation outputs.

In addition to quantitative evaluation, we provided a visual comparison of Det-LIME with several existing explainability methods, including the gradient-based class activation mapping approach LayerCAM (Jiang et al., 2021), LIME (Ribeiro et al., 2016), DLIME (Zafar and Khan, 2021), and SLIME (Zhou et al., 2021). LayerCAM leverages internal convolutional layers to produce high-resolution attributions, enabling precise localization of salient features. In contrast, Det-LIME is model-agnostic and can be applied to any object detector, generating heatmaps that indicate which image regions contribute to each predicted bounding box. LIME, DLIME, and SLIME provide explanations for only a single instance within an image, whereas Det-LIME produces explanations for all detected objects in the image. These visual comparisons assessed aspects of explanation behavior that are not fully captured by spatial overlap metrics, including compactness, interpretability, and the ability to represent multiple detections within the same image. By visually comparing these methods, we ofered a qualitative view of how Det-LIME addressed the challenges of multi-object detection and provided a model-agnostic approach for interpretable explanations.

## 3.3 Det-LIME in Ecological Applications

We applied the above-described Det-LIME workflow to two diferent architectures trained on two object detection architectures trained on two datasets. We applied Det-LIME to a harbor seal detection task using Faster R-CNN (Ren et al., 2015) (data collection and labeling methods in Appendix A.1; modeling methods in Appendix B.1). As a broader example demonstrating the applicability of the technique beyond marine mammals and across detection models, we additionally evaluated Det-LIME on a penguin detection task using a YOLOv9 model (Wang et al., 2025), fine-tuned on a seabird dataset containing Southern Rockhopper Penguins among breeding black-browed albatrosses (Hayes et al., 2021) (modeling methods in Appendix B.2). Both examples focused on animals that are found in groups or colonies, and were representative of applications aimed at understanding animal density, abundance, and distributions in relation to environmental factors. Det-LIME was applied only to the held-out test set images in both cases. Code to apply Det-LIME is available here: https://osf.io/d456u/?view\_only=98eef0382ba745d9a1e7b89e2cd9b4a9

## 4 Results

## 4.1 Evaluation of Det-LIME

Our evaluation demonstrated that Det-LIME addressed the major limitations of vanilla and variant LIME approaches in ecological multi-instance detection. The number of evaluated detection instances was reported in Table 1 to clarify the denominator used for Attribution Ratio and Max Saliency Hit Rate, as both metrics were averaged over individual detections rather than over images. As shown in Table 1, Det-LIME substantially improved both the Attribution Ratio and Max Saliency Hit Rate across harbor seal and penguin detection tasks, with gains of more than 20-30 percentage points relative to the strongest baselines in several comparisons. These improvements supported Det-LIME’s ability to provide instance-specific explanations. By aligning perturbations with bounding boxes and performing IoU-based matching between detections in the original and perturbed images, Det-LIME limited attribution drift, preserved the correspondence between saliency and individual objects, and reduced spurious signals in backgrounds.

In comparison, baseline methods had lower Attribution Ratio and Max Saliency Hit Rate (Table 1), reflecting attributions that blurred across multiple objects, overemphasized background correlations, and lacked spatial precision, making it dificult to identify the features truly driving each detection. Visual comparisons in Figure 2 reinforced this quantitative evidence. While vanilla LIME and its stabilized variants produced blocky, cluttered maps that often highlighted ice, rock, or water textures, Det-LIME yielded sharper, more coherent attributions that corresponded more closely to annotated animal regions. This shift was especially pronounced in the penguin detection setting, where small, densely clustered animals in colonies posed a severe challenge for coarse perturbation methods; Det-LIME’s high Max Saliency Hit Rate demonstrated its robustness in precisely these contexts. Together, these results showed that Det-LIME adapted LIME’s perturbation framework to detection “in the wild,” producing cleaner, more actionable explanations that better supported threshold setting, post-processing, and targeted ecological data collection.

Table 1: Comparison of LIME Variants Across Datasets and Detection Models
<table><tr><td>Dataset and Detection LIME Variant Model</td><td></td><td>Mean Attribution Ratio</td><td>Max Hit Rate</td><td>Adjusted Mean Attribution Ratio</td><td>Adjusted Max Hit Rate</td></tr><tr><td rowspan="4">Harbor Seal Detection Faster R-CNN</td><td>LIME</td><td>16.23% (±17.85%)</td><td>17.91%</td><td>22.14% (±18.74%)</td><td>25.33%</td></tr><tr><td>SLIME</td><td>25.02% (±24.36%)</td><td>14.93%</td><td>36.32% (±24.37%)</td><td>22.67%</td></tr><tr><td>DLIME</td><td>24.17% (±24.09%)</td><td>31.34%</td><td>37.26% (±23.79%)</td><td>52.00%</td></tr><tr><td>Det-LIME</td><td>48.76% (±29.51%)</td><td>92.42%</td><td>52.34% (±29.65%)</td><td>96.00%</td></tr><tr><td rowspan="4">Penguin Detection YOLOv9</td><td>LIME</td><td>3.67% (±11.46%)</td><td>3.42%</td><td>29.55% (±20.44%)</td><td>27.96%</td></tr><tr><td>SLIME</td><td>4.23% (±12.94%)</td><td>3.46%</td><td>35.11% (±20.94%)</td><td>28.67%</td></tr><tr><td>DLIME</td><td>4.71% (±14.24%)</td><td>4.38%</td><td>40.15% (±21.41%)</td><td>36.20%</td></tr><tr><td>Det-LIME</td><td>39.25% (±28.19%)</td><td>82.20%</td><td>39.79% (±29.08%)</td><td>98.92%</td></tr></table>

Note. Values were computed over matched detection instances rather than images. Only valid matched detections with corresponding attribution maps were included in the instance-level evaluation. In the all-instance evaluation, the harbor seal dataset included 134 evaluated instances for LIME, SLIME, and DLIME, and 132 for Det-LIME; the penguin dataset included 2,720 evaluated instances for LIME, SLIME, and DLIME, and 2,450 for Det-LIME. The adjusted metrics were computed on a one-to-one matched subset selected by the single-instance methods, including 75 harbor seal instances and 279 penguin instances.

![](images/7b758161b84bc097ee66c9709c30008b6945d584bf7fac4a0491f61a35679f87.jpg)  
(a) Original image

![](images/c4fec4a5f7acbbf56169c926837092988e59aa2b960eb073996bd8228400e68b.jpg)  
(b) Object detections

![](images/8a6855a04f2066079b3d1dc380db625c08e7d962063494cdf1502736f5f355b1.jpg)  
(c) Det-LIME heatmap

![](images/ea4c31d9527d5cb08409ed2dea59e13368b8c17f504ffef1e158705673f25c55.jpg)  
(d) Det-LIME overlay

![](images/f00606ca4c08dd6a67e3321589602ce4f7110189d76ceb001915b1ce6c45c17c.jpg)  
(e) LayerCAM

![](images/01d8b65bc2fc06b324ac22f799ab1ad58f8f538a67bc0ed3ff8e6a93087fbde5.jpg)  
(f) Vanilla LIME

![](images/f89990f7507e6a0f33c5de85cbb2191760f5bb7692361f4af88882a5f9f69511.jpg)  
(g) SLIME

![](images/34b9dfb7f271d6eae4e33d979b2a3d555726880475aec01c348e273c76c9ae54.jpg)  
(h) DLIME  
Figure 2: Comparison of diferent LIME variants for the harbor seal multi-instance detection task using Faster R-CNN. (a) Original image, (b) object detections with bounding boxes, (c) Det-LIME explanation heatmap, (d) Det-LIME overlay, (e) LayerCAM, (f) vanilla LIME, (g) LIME variant SLIME, (h) LIME variant DLIME. In the attribution maps, a color spectrum represents relative importance, with warm colors (e.g., red, orange, yellow) indicating high contribution and cool colors (e.g., blue) indicating little or no contribution.

## 5 Discussion

![](images/8d267dd751c97e652a5fb9e255b243c7cc7417fe8b7387cc3534dea3dd9d8e23.jpg)  
(a) Original image

![](images/7624db1150960286b11cfdf4da6b23f464088ea57ea67c635b1684144db97dfe.jpg)  
(b) Object detections

![](images/0dabc208928ff3f9c0ff2d132887b37f96b66d99764c0cf3e104225e9dceebc6.jpg)  
(c) Det-LIME heatmap

![](images/f1522abea8ea3c272978b827e8ebe43d02c6e6de764d6878b376b6e268690ceb.jpg)  
(d) Det-LIME overlay

![](images/60df6a4ee224d93c34942cdbaa4a92b0cba461cb897a4b6f09e82540a37751af.jpg)  
(e) LayerCAM

![](images/85cc13cad6933d77ebb4a41918cb5bb432e8dda122b38dba138e3772ffc2409e.jpg)  
(f) Vanilla LIME

![](images/d068b278472f622c98e190cd807a5e4e2d5b4a21c7a25a8e5527407a1591615d.jpg)  
(g) SLIME

![](images/7e5f787238390ad161780681fc3a1b3c81261d18e3daab804b9535cb4e9400e7.jpg)  
(h) DLIME  
Figure 3: Visual explanations of a failure case where black ice was misidentified as a seal in the harbor seal multi-instance detection task using Faster R-CNN. (a) Original image, (b) object detections with bounding boxes, (c) Det-LIME explanation heatmap, (d) Det-LIME overlay, (e) LayerCAM, (f) vanilla LIME, (g) LIME variant SLIME, (h) LIME variant DLIME. In the attribution maps, a color spectrum represents relative importance, with warm colors (e.g., red, orange, yellow) indicating high contribution and cool colors (e.g., blue) indicating little or no contribution.

Applying Det-LIME to ecological imagery demonstrated how instance-aware explanations revealed both valid cues and recurrent sources of error (Figure 3 and Figure 4). For correctly identified seals, Det-LIME assigned higher attribution to superpixels overlapping the detected object or its immediate boundary, indicating that the detector relied on image regions associated with the target rather than background context alone. By contrast, vanilla LIME and other LIME variants tended to highlight coarse superpixels spanning both the animal and adjacent background, obscuring the distinction between object and context. This distinction underscored the added value of Det-LIME: while bounding boxes alone indicate where detections occur, attribution maps can help identify the visual evidence supporting those detections and assess whether the detector is relying on animal features, contextual cues, or both.

Failure cases further illustrated the diagnostic value of Det-LIME. In Figure 3, the region of black ice was incorrectly detected as a seal by Faster R-CNN. Although the false positive bounding box identified the location of the error, it did not reveal the visual evidence underlying the prediction. Det-LIME attribution maps showed concentrated importance over dark, high-contrast textures within the detected region, suggesting that the model relied on low-level visual similarities between seal bodies and surface cracks or melt patterns in the ice. This instance-level explanation provided evidence for why the false detection may have occurred by showing which image regions most influenced the detector’s output, rather than only indicating the location

![](images/df21751410c786d098f4ca6265520b7c600e49d5e01ae090239ca053de6c641a.jpg)  
(a) Original

![](images/9db4834122f6b961987f72721dde8d6f4d36e8339189753fbc3285b4aa607791.jpg)  
(b) Detection

![](images/1adcc9a1c6fae0af50477d40e0fc7234109a4156c2d70baa8889e78281c3239e.jpg)  
(c) Heatmap

![](images/2d90472989a56f91636cae4c06ee2da451655da75c78b6de082a3e7ccb7810a9.jpg)  
(d) Overlay

![](images/224e8740dafd16271220098f824dbcad2f66b6cfaac2cf172631c965928413d6.jpg)  
(e) Original

![](images/aaba1c2cd0f07c454c6d6fbb48dc366aecdab21d6aa19f7d3106498f70947364.jpg)  
(f) Detection

![](images/55003b4590f33fccb75f7b9287d08a37dbda43a854993149dd9d5f6381bbcc1a.jpg)  
(g) Heatmap

![](images/d50566e09a97d53bee4be47d61cdac5fa175e42da6ebdca30da8d2d969873f50.jpg)  
(h) Overlay

![](images/0aabd21aec6d48e8963033548154cb45dacc647f499b9e250fa321df90f73293.jpg)  
(i) Original

![](images/41978b0ed6de6390fbf99c6deb216441be09c2d024c55d06e47abe9372fe4f16.jpg)  
(j) Detection

![](images/c7b9efc38a93415ebcc837bf0e790c1be5ef43304eb1bf9b3c19092f0de1efe6.jpg)  
(k) Heatmap

![](images/a119d47a0b9c4d484ef737a4613ae6ea086996fbe72fa6a995eb0c542a4c8d3c.jpg)  
(l) Overlay

Figure 4: Visual explanations of penguin detection failures. Each column corresponds to one image case. The first column (a, e, i) shows the original images. The second column (b, f, j) illustrates false positive detections, with ground truth boxes in green, true positives in blue, and false positives in red. The third column (c, g, k) shows Det-LIME explanation heatmaps. The fourth column (d, h, l) shows Det-LIME overlays combining attribution maps with the original images. These visualizations illustrate how Det-LIME identifies image regions that the model focused on when making false positive predictions. Across cases, these errors were frequently located at boundaries, either at the image edge or at transitions between textures such as rock and grass.

of the false-positive bounding box. In contrast, LayerCAM produced difuse activations that extended beyond the detected instance, while vanilla LIME and SLIME provided limited insight into false-positive regions, which reduced their efectiveness in multi-instance detection settings. A similar pattern appeared in penguin detection failures (Figure 4), where Det-LIME explained false positives arising from co-occurring albatrosses and detections near image boundaries, including frame edges and transitions between habitats such as rock and grass. By conditioning explanations on individual detections, Det-LIME revealed the specific visual cues responsible for false positives and supported a more precise analysis of model failure modes than bounding boxes alone.

These examples also show that Det-LIME heatmaps should be interpreted as evidence of model behavior, rather than as direct maps of animal anatomy. In some cases, such as the leftmost seal in Figure 3, the strongest Det-LIME response was located on nearby ice or high-contrast background regions within or close to the detection box, rather than clearly on the seal body. This does not necessarily mean that the background region alone drives the detection. Convolutional neural networks build predictions by combining visual features across the image. Earlier layers typically respond to relatively simple features such as edges, contrast, and texture, while deeper layers combine these features into more complex representations of objects and their surrounding context. A seal detection may therefore depend on both features of the animal and how those features occur in relation to nearby ice, water, shadows, or other parts of the scene.

Det-LIME does not directly examine these internal representations. Instead, it perturbs diferent combinations of superpixels and measures how those changes afect a particular detection. High attribution to a background region therefore indicates that changing that region influences the detector’s prediction, potentially because the detector uses that information together with features of the animal. Because Det-LIME summarizes these efects as contributions from individual superpixels, relationships among regions are not shown directly in the heatmap.

For ecological applications, this distinction is important. A heatmap that is not centered on the animal does not necessarily indicate a failure of the explanation method. It may instead reveal that the model is relying on surrounding habitat or contextual information, potentially in combination with animal features, rather than on the animal itself alone. Det-LIME is therefore most useful as a diagnostic tool for identifying whether a detection is supported by animal features, surrounding habitat features, or a mixture of both.

LayerCAM provided a useful comparison for interpreting these patterns. In Figures 2 and 3, LayerCAM sometimes produced sharper relevance maps that were more closely aligned with visible animal structures than the corresponding Det-LIME maps. This visual clarity can be reassuring when the goal is to confirm that a detection is supported by features of the animal itself, such as body shape, edge structure, or local texture. LayerCAM can provide this type of localization because it uses intermediate feature activations and gradients from within the neural network, allowing it to highlight image regions that are strongly associated with the detected class.

Det-LIME provides a diferent and complementary view of model behavior. Rather than using internal feature activations, Det-LIME evaluates how changes to image regions afect the detector’s output. This makes it useful for identifying cases where a detection may depend not only on the animal, but also on nearby habitat features, image boundaries, or high-contrast background textures. For example, when Det-LIME assigns strong importance to ice, rock, or habitat edges near a detected animal, this suggests that the detector may be using contextual information that is correlated with the target species in the training data. In ecological applications, this distinction is important because a model can make a correct detection while still relying partly on environmental cues that may not generalize well to new locations, seasons, or image conditions.

This diference reflects a practical trade-of between the two approaches. LayerCAM requires access to the model architecture, internal feature maps, and gradients, which may not be available when researchers use third-party models or deployed detection systems that only return bounding boxes and confidence scores. LayerCAM is also sensitive to the choice of target layer. Lower layers may emphasize edges, textures, or local patterns, while deeper layers may capture broader semantic information related to the animal or its surrounding context. In our examples, the explanation quality varied depending on which layer was selected, indicating that layer choice can afect how clearly the resulting map supports ecological interpretation. Det-LIME is less visually fine-grained, but because it only requires the input image and detector outputs, it can be applied more broadly as a black-box diagnostic tool for evaluating whether detections are driven by animal features, contextual features, or a mixture of both. When model internals are available, LayerCAM and Det-LIME should be viewed as complementary rather than competing approaches. Used together, they provide a more diverse and comprehensive perspective on model behavior, helping researchers distinguish detections supported by visible animal structures from those influenced by habitat context or background cues.

The results of the present study illustrate that Det-LIME adapts the perturbation-based framework of LIME to ecological object detection in a robust manner, producing explanations that are box-conditioned and instance-stable. Perturbations were generated via superpixel masking, and detections were matched back to original instances using IoU-based correspondence across perturbed outputs. Instance contributions were weighted by detection confidence and spatial alignment, with only suficiently overlapping detections retained via an IoU threshold. This improved the stability of instance-level attribution across perturbations, reduced attribution drift, and suppressed weakly aligned background responses. We demonstrated instance fidelity using the Attribution Ratio and Max Saliency Hit Rate, and showed that Det-LIME outperformed vanilla LIME and LIME variants SLIME and DLIME for multi-instance detection tasks by keeping saliency concentrated within the queried detection and consistently anchoring peak attribution on the same instance.

A key contribution of Det-LIME lies in its diagnostic utility for multiple predictions within single images, a frequently encountered problem for researchers using “black box” AI approaches to study colonial or socially aggregating organisms. Explanations revealed that the seal detectors often misattributed dark background regions, such as black ice, leading to false positives when low-level visual similarity overwhelmed biological distinctiveness. In the penguin example, the detector misclassified albatrosses or produced false detections along habitat boundaries. These insights suggest concrete avenues for model improvement: targeted data augmentation with negative examples of ice and rock and refined training sets with greater habitat variability. Beyond improving models, attribution maps signal when outputs should be treated cautiously, ofering a safeguard against over-reliance on automated predictions in conservation workflows.

At the same time, several limitations warrant consideration. Det-LIME inherits LIME’s reliance on superpixels and perturbation kernels, which can introduce sensitivity to segmentation parameters and discretize importance into regions rather than addressing pixel-level fidelity. Our evaluation also measured localization fidelity against bounding boxes, which may understate explanation quality when salient regions align with subparts of an animal or spill beyond annotated boundaries. Residual labeling errors further complicate interpretation, particularly in ecological datasets where annotations are resource-intensive and often noisy.

In addition, the present study evaluated Det-LIME as a complete workflow rather than isolating the contribution of each individual component. The components of Det-LIME are sequentially connected, with each step relying on the output of the previous one, and the intermediate results are not independent attribution maps that can be evaluated in the same way as the final explanation. Consequently, the reported metrics assess the performance of the full explanation pipeline rather than that of individual components. A systematic ablation study would help quantify the contribution of each stage and remains an important direction for future work.

The proposed evaluation metrics, AR and MSHR, also have inherent limitations. Because object-detection explanations are generated with respect to model-predicted bounding boxes, class labels, and confidence scores, these metrics evaluate how well an explanation aligns with the detector’s outputs rather than providing a fully detector-independent measure of explanation quality. This limitation is not unique to Det-LIME, but reflects a broader challenge in explainable artificial intelligence (XAI). Previous studies have noted that interpretability lacks a universally accepted definition or evaluation standard, and that reliable ground-truth explanations are rarely available in real-world applications (Doshi-Velez and Kim, 2017). As a result, XAI methods are typically assessed using complementary proxy measures, such as localization, faithfulness, robustness, or human evaluation, each capturing a diferent aspect of explanation quality (Nauta et al., 2023). Future work could incorporate perturbation-based faithfulness evaluations, including deletion and insertion analyses, to examine whether regions highlighted by Det-LIME have a measurable efect on detector confidence or localization. Such analyses would complement the spatial overlap metrics presented in this study by providing additional evidence that the highlighted regions meaningfully influence model predictions. Finally, our study did not assess runtime eficiency, operator burden, or field usability, which are critical factors for deployment and should be explored in future work.

Despite these challenges, Det-LIME demonstrated how explanation methods can serve as actionable diagnostics for both ecological inference and model development. By revealing the visual evidence underpinning predictions, Det-LIME helped build justified confidence in correct detections while also highlighting recurrent weaknesses that can guide ecologists toward targeted data collection, model refinement, and field validation. More broadly, these results underscore the role of explainability as a bridge between computer vision models and ecological decision-making, ensuring that detectors are not only accurate but also evidence-based, clear about where they fail, and defensible in policy and field workflows.

## Note on references

In computer science and machine learning, many high-impact and foundational contributions are published in double-blind peer-reviewed conference and workshop proceedings rather than traditional journals. Accordingly, several key references cited in this work appear in conference proceedings that serve as primary archival venues within these fields.

## References

Günel Aghakishiyeva, Jiayi Zhou, Saagar Arya, Julian Dale, James David Poling, Holly R. Houliston, Jamie N. Womble, Gregory D. Larsen, David W. Johnston, and Brinnae Bent. Photorealistic inpainting for perturbationbased explanations in ecological monitoring, 2025. URL https://arxiv.org/abs/2510.03317.

Yusufu Brima and Mohamed Atemkeng. Saliency-driven explainable deep learning in medical imaging: Bridging visual explainability and statistical quantitative analysis. BioData Mining, 17(1):1–33, 2024. doi: 10.1186/s13040-024-00370-4. URL https://doi.org/10.1186/s13040-024-00370-4.

Annie Britton, Garrett Graham, and Molly Woloszyn. Exploring relationships between drought indices and ecological drought impacts using machine learning and explainable AI. J. Appl. Service Climatol., 2024 (5):1–12, 2024.

Alexander Buchelt, Alexander Adrowitzer, Peter Kieseberg, Christoph Gollob, Arne Nothdurft, Sebastian Eresheim, Sebastian Tschiatschek, Karl Stampfer, and Andreas Holzinger. Exploring artificial intelligence for applications of drones in forest ecology and management. Forest Ecology and Management, 551: 121530, 2024. ISSN 0378-1127. doi: 10.1016/j.foreco.2023.121530. URL https://www.sciencedir ect.com/science/article/pii/S0378112723007648.

Mateusz Choiński, Mateusz Rogowski, Piotr Tynecki, Dries P. J. Kuijper, Marcin Churski, and Jakub W. Bubnicki. A first step towards automated species recognition from camera trap images of mammals using ai in a european temperate forest. In Khalid Saeed and Jiří Dvorský, editors, Computer Information Systems and Industrial Management: 20th International Conference, CISIM 2021, Proceedings, volume 12883 of Lecture Notes in Computer Science, pages 299–310. Springer, Cham, 2021. ISBN 978-3-030-84339-7. doi: 10.1007/978-3-030-84340-3\_24. URL https://link.springer.com/chapter/10.1007/978-3-0 30-84340-3\_24.

Finale Doshi-Velez and Been Kim. Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608, 2017. URL https://arxiv.org/abs/1702.08608.

Caroline M. Gevaert. Explainable AI for earth observation: A review including societal and regulatory perspectives. International Journal ofApplied Earth Observation and Geoinformation, 112:102869, 2022. ISSN 1569-8432. doi: 10.1016/j.jag.2022.102869. URL https://www.sciencedirect.com/scienc e/article/pii/S1569843222000711.

Patrick C. Gray, Gregory D. Larsen, and David W. Johnston. Drones address an observational blind spot for biological oceanography. Frontiers in Ecology and the Environment, 20(7):375–376, 2022. doi: 10.1002/fee.2472. URL https://esajournals.onlinelibrary.wiley.com/doi/epdf/10.1002 /fee.2472?saml\_referrer.

Madeline C. Hayes, Patrick C. Gray, Guillermo Harris, Wade C. Sedgwick, Vivon D. Crawford, Natalie Chazal, Sarah Crofts, and David W. Johnston. Drones and deep learning produce accurate and eficient monitoring of large-scale seabird colonies. Ornithological Applications, 123(3):1–16, 2021. doi: 10.1093/ornithapp/duab022.

Peng-Tao Jiang, Chang-Bin Zhang, Qibin Hou, Ming-Ming Cheng, and Yunchao Wei. Layercam: Exploring hierarchical class activation maps for localization. IEEE Transactions on Image Processing, 30:5875–5888, 2021. doi: 10.1109/TIP.2021.3089943.

Samparthi V S Kumar and Hari Kondaveeti. Towards transparency in AI: Explainable bird species image classification for ecological research. Ecological Indicators, 169:112886, 12 2024. doi: 10.1016/j.ecolind. 2024.112886.

Harsh Mankodiya, Dhairya Jadav, Rajesh Gupta, Sudeep Tanwar, Wei-Chiang Hong, and Ravi Sharma. Od-xai: Explainable AI-based semantic object detection for autonomous vehicles. Applied Sciences, 12 (11):5310, 2022. doi: 10.3390/app12115310. URL https://doi.org/10.3390/app12115310.

Meike Nauta, Jan Trienes, Shreyasi Pathak, Elisa Nguyen, Michelle Peters, Yasmin Schmitt, Jörg Schlötterer, Maurice van Keulen, and Christin Seifert. From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable ai. ACM Computing Surveys, 55(13s):295:1–295:42, 2023. doi: 10.1145/3583558.

Vitali Petsiuk, Rajiv Jain, Varun Manjunatha, Vlad I. Morariu, Ashutosh Mehra, Vicente Ordonez, and Kate Saenko. Black-box explanation of object detectors via saliency maps. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11443–11452, 2021. ISBN 978-1-6654-4509-2. doi: 10.1109/CVPR46437.2021.01128.

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: towards real-time object detection with region proposal networks. In Proceedings of the 29th International Conference on Neural Information Processing Systems - Volume 1, NIPS’15, page 91–99, Cambridge, MA, USA, 2015. MIT Press.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "why should i trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, page 1135–1144, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450342322. doi: 10.1145/2939672.2939778. URL https://doi.org/10.1145/2939672.2939778.

Bryan C. Russell, Antonio Torralba, Kevin P. Murphy, and William T. Freeman. LabelMe: A database and web-based tool for image annotation. International Journal ofComputer Vision, 77(1-3):157–173, May 2008. doi: 10.1007/s11263-007-0090-8. URL https://doi.org/10.1007/s11263-007-0090-8.

Vít Růžička, Gonzalo Mateo-Garcia, Luis Gómez-Chova, Anna Vaughan, Luis Guanter, and Andrew Markham. Semantic segmentation of methane plumes with hyperspectral machine learning models. Scientific Reports, 13(1):19999, 2023. ISSN 2045-2322. doi: 10.1038/s41598-023-44918-6. URL https://www.nature.com/articles/s41598-023-44918-6.

Masahiro Ryo, Boyko Angelov, Stefano Mammola, Jonathan M. Kass, Blas M. Benito, and Florian Hartig. Explainable artificial intelligence enhances the ecological interpretability of black-box species distribution models. Ecography, 44(2):199–205, 2021. doi: 10.1111/ecog.05360. URL https: //doi.org/10.1111/ecog.05360.

Mirka Saarela and Vili Podgorelec. Recent applications of explainable AI (xai): A systematic literature review. Applied Sciences, 14(19):8884, 2024. doi: 10.3390/app14198884. URL https://doi.org/10 .3390/app14198884.

Jonas Herskind Sejr, Peter Schneider-Kamp, and Naeem Ayoub. Surrogate object detection explainer (sodex) with YOLOv4 and LIME. Machine Learning and Knowledge Extraction, 3(3):662–671, 2021. doi: 10.3390/make3030033. URL https://doi.org/10.3390/make3030033.

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. International Journal of Computer Vision, 128(2):336–359, October 2019. ISSN 1573-1405. doi: 10.1007/s11263-019-01228-7. URL http://dx.doi.org/10.1007/s11263-019-01228-7.

Michael A. Tabak, Mohammad S. Norouzzadeh, David W. Wolfson, Steven J. Sweeney, Kurt C. Vercauteren, Nathan P. Snow, Joseph M. Halseth, Paul A. Di Salvo, Jesse S. Lewis, Michael D. White, et al. Machine learning to classify animal species in camera trap images: Applications in ecology. Methods in Ecology and Evolution (Online), 10(4), 11 2018. ISSN ISSN 2041-210X. doi: 10.1111/2041-210x.13120. URL https://www.osti.gov/biblio/1614652.

Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. Yolov9: Learning what you want to learn using programmable gradient information. In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and Gül Varol, editors, Computer Vision – ECCV 2024, volume 15089 of Lecture Notes in Computer Science, pages 1–17. Springer, Cham, 2025. doi: 10.1007/978-3-031-72751-1\_1. URL https://doi.org/10.1007/978-3-031-72751-1\_1.

Jamie N. Womble, Jay M. Ver Hoef, Scott M. Gende, and Elizabeth A. Mathews. Calibrating and adjusting counts of harbor seals in a tidewater glacier fjord to estimate abundance and trends 1992–2017. Ecosphere, 11(4):e03111, 2020. doi: 10.1002/ecs2.3111. URL https://doi.org/10.1002/ecs2.3111.

Toshinori Yamauchi and Masayoshi Ishikawa. Spatial sensitive grad-cam: Visual explanations for object detection by incorporating spatial sensitivity. In 2022 IEEE International Conference on Image Processing (ICIP), pages 256–260, 2022. doi: 10.1109/ICIP46576.2022.9897350.

Muhammad Rehman Zafar and Naimul Mefraz Khan. Deterministic local interpretable model-agnostic explanations for stable explainability. Machine Learning and Knowledge Extraction, 3(3):525–541, 2021. doi: 10.3390/make3030027. URL https://doi.org/10.3390/make3030027.

Jianming Zhang, Sarah Adel Bargal, Zhe Lin, Jonathan Brandt, Xiaohui Shen, and Stan Sclarof. Top-down neural attention by excitation backprop. International Journal ofComputer Vision, 126(10):1084–1102, 2018. doi: 10.1007/s11263-017-1059-x. URL https://doi.org/10.1007/s11263-017-1059-x.

Bolei Zhou, Aditya Khosla, Àgata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2921–2929, 2015. URL https://api.semanticscholar.org/CorpusID:6789015.

Zhengze Zhou, Giles Hooker, and Fei Wang. S-lime: Stabilized-lime for model explanation. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, KDD ’21, page 2429–2438. ACM, August 2021. doi: 10.1145/3447548.3467274. URL http://dx.doi.org/10.1145/3447548.3 467274.

## A Data Collection and Labeling

## A.1 Harbor Seal Data

Drone surveys were conducted using a Wingtra One Gen II fixed-wing platform (Zurich, Switzerland) with vertical takeof and landing (VTOL) capabilities imaging with a Sony Alpha 6100 APS-C camera (Tokyo, Japan) with a Sony E 20 mmf/2.8 lens. Flight plans were created and carried out using WingtraPilot flight planning software. Flights operated at ∼ 60 − 85 m altitude and ∼ 9 − 22 m/s airspeed over regions that were historically sampled by occupied aircrafts. Drone operations were conducted under permit by NOAA and the NPS.

Flights in glacial fjords occurred along non-overlapping parallel transects oriented lengthwise through the glacial end of the fjord, with transects oriented perpendicular to the glacier terminus. These flight plans were like those of historic surveys that used occupied aircrafts (Womble et al., 2020), but were optimized to achieve high-density coverage of the inner regions of the fjords where seal densities are highest. Surveys from occupied aircrafts historically sampled the entire west arm of JHI along the same 12 transects year after year, maintaining a ∼ 100 − m bufer between photographs across transects and a ∼ 20 − m bufer between consecutive photographs along transects. Surveys from unoccupied aircrafts in 2023 and 2024 surveyed smaller gross extents with a series of nearly contiguous but not overlapping transects, maintaining a ∼ 5 − m bufer between photographs across transects and 65-70 % overlap between consecutive photographs along transects. Surveys from unoccupied aircrafts in glacial fjords consisted of 1-3 flights each using impromptu flight plans informed by the extent of floating ice habitat and drone performance in the prevailing weather conditions at the time of the survey. Surveys of terrestrial sites also consisted of parallel transects arranged in a high density to achieve a target of ∼ 601% overlap between photographs along transects and across transects.

The training dataset was visually reviewed and manually thinned to remove photographs that did not include at least one positive instance of a harbor seal on floating ice. This was done to mitigate the risk of negative bias in model training, which can occur with an overabundance of negative training data. The resulting dataset images were each subdivided into tiles of 640 × 640 pixels. Each tile was manually inspected and annotated to mark all harbor seal locations using LabelMe (Russell et al., 2008).

## B Modeling Method

## B.1 Harbor Seal Detection Task using Faster R-CNN

We employed a transfer learning approach for seal detection. Our model, a Faster R-CNN (Ren et al., 2015) with a ResNet-50 backbone and Feature Pyramid Network (FPN), was initialized with weights pre-trained on the COCO dataset. We then performed full fine-tuning, updating all layers of the network to adapt the model to our seal dataset.

To improve robustness, the training dataset was augmented with geometric transformations (horizontal and vertical flips, rotations up to 45<sup>◦</sup>, and random crops) and color adjustments (brightness, contrast, saturation, and hue). Validation images were processed only with normalization and tensor conversion for consistency, while the test set was completely held out and used only for final evaluation without any data augmentation.

The model was trained using stochastic gradient descent with gradient clipping, warmup learning rate scheduling, and early stopping. Weights & Biases was used for experiment tracking.

Hyperparameters were tuned over learning rate (0.001-0.01), momentum (0.85-0.95), and weight decay (0.0001-0.001) using a grid search. The chosen configuration used a batch size of 8, a learning rate of 0.0090, momentum of 0.874, and weight decay of 0.0001. Although training was set for up to 200 epochs, the best model was obtained at epoch 67, where validation performance peaked with a mean Average Precision (mAP, averaged over IoU thresholds from 0.5 to 0.95) of 0.61, mAP50 of 0.95, and mean Average Recall (mAR, recall averaged across up to 100 detections per image) of 0.69. The corresponding training and validation losses were 0.11 and 0.13, respectively. On the held-out test set of 76 images, the final model achieved an overall mAP of 0.65, with mAP50 of 0.98 and mAP75 of 0.78.

## B.2 Penguin Detection Task using YOLOv9

We trained an object detection model to identify penguins using the YOLOv9c architecture (Wang et al., 2025). The dataset contained 2,644 training images, 288 validation images, and 321 test images, all reformatted into the $\mathrm { Y O L O v 9 }$ format for consistency. To improve generalization and reduce overfitting, we applied a range of data augmentation techniques, including random rotation, translation, scaling, horizontal flips, and color space adjustments, along with mixup regularization. Training was carried out on 640-pixel images with a batch size of 16 for up to 200 epochs, with early stopping if validation performance did not improve for 30 epochs. We optimized the model with AdamW, using an initial learning rate of 0.01, momentum of 0.937, and weight decay of 0.0005. On the validation set, the model achieved robust performance, with a precision of 0.911, a recall of 0.923, an mAP50 of 0.956, and an mAP50–95 of 0.519. Evaluation on the held-out test set confirmed this robustness, yielding a precision of 0.912, a recall of 0.933, an mAP50 of 0.960, and an mAP50–95 of 0.528 across 2,720 instances. These results show that YOLOv9c can efectively detect penguins in challenging natural imagery and generalize well to unseen data.