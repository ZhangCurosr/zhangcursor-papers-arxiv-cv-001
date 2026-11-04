# Gradient-Guided Orientation Learning for Weakly Supervised Oriented Object Detection

Junyong Jang<sup>∗</sup> Juheon Park<sup>∗</sup> Minyoung Jeon Jaehyup Lee<sup>†</sup> Kyungpook National University, South Korea

bogwang6@knu.ac.kr, juheona@knu.ac.kr, mym0314@knu.ac.kr, jaehyuplee@knu.ac.kr

Rotated FCOS (ICCV’19)

(a) OBB IoU limitation

H2RBox-v2 (NeurIPS’23)

![](images/92d2b9d8e91e381e66fd8324e246547b3171936df2cc1797214b2d88add250ea.jpg)  
ABBSPO (CVPR’25)  
(b) Comparison with Existing Methods  
(c) Effect of ℒ<sub>grad</sub>

GradOrient

Figure 1. Qualitative comparison of detection results. (a) Similar OBB IoU values can correspond to substantially different angle errors. (b) GradOrient provides more accurate detections than existing methods in challenging scenes containing densely distributed or large elongated objects. (c) $\mathcal { L } _ { \mathrm { g r a d } }$ improves fine-grained angle estimation and oriented bounding-box alignment.

## Abstract

Weakly supervised oriented object detection (WS-OOD) aims to reduce the annotation cost of rotated bounding boxes by leveraging cheaper annotations such as horizontal boxes or points. However, these weak annotations provide no explicit orientation information, forcing existing methods to infer object angles through indirect geometric consistency or equivariant constraints. Such indirect supervision often becomes ambiguous and unstable under real-world appearance variations, including illumination changes, color shifts, shadows, and background clutter. In this paper, we present a simple yet effective gradient-guided orientation learning strategy for WS-OOD. Motivated by the observation that the input image itself contains useful orientation priors, our method extracts dominant gradientbased orientation cues from object regions and converts them into pseudo-angle labelsfor auxiliary supervision, enabling the detector to learn orientation-aware representa-

tions without requiring rotated bounding box annotations or extra learnable parameters. Furthermore, since AP can mask substantial angular errors when spatial IoU remains high, we introduce an orientation-aware evaluation metric that incorporates angular discrepancies into the standard AP protocol. Extensive experiments on weakly supervised oriented detection benchmarks demonstrate that the proposedpseudo-angle supervision improves both detection performance and orientation estimation quality, while the proposed metric provides a more faithful assessment of angle prediction than conventional AP alone.

## 1. Introduction

Oriented object detection (OOD) aims to localize objects with rotated bounding boxes (RBoxes), providing both object location and orientation information. This task is particularly important in remote sensing, where objects such as ships, aircraft, vehicles, bridges, and harbors frequently appear with arbitrary orientations. Compared with annotating horizontal bounding boxes (HBoxes), however, annotating RBoxes requires additional angle information and is considerably more labor-intensive [30]. This annotation burden has motivated weakly supervised oriented object detection (WS-OOD), which learns oriented detectors from cheaper annotations such as points or HBoxes. Despite their costeffectiveness, these weak annotations provide localization cues but do not explicitly specify object orientation, making accurate angle recovery inherently ambiguous.

To address the absence of direct angle labels, existing WS-OOD methods [9, 15, 19, 26, 29–31] mainly infer object orientations through transformation consistency, equivariant constraints, or geometric priors. These methods demonstrate that useful orientation cues can be obtained by enforcing prediction consistency across original, rotated, and flipped views. However, such cues remain indirect, since multiple orientations may satisfy the same weak supervision constraint. This ambiguity makes precise angle estimation difficult, especially under real-world appearance variations such as illumination changes, color shifts, shadows, occlusions, and background clutter. Therefore, a practical WS-OOD framework requires an additional orientation-aware cue that complements geometric consistency without increasing annotation cost.

In weakly supervised learning, where the provided labels are intentionally incomplete, the design of supervision plays an essential role. Nevertheless, one important source of orientation information has been relatively underexplored: the input image itself. Remote sensing objects often exhibit elongated structures, boundary edges, and local gradient patterns that are strongly correlated with their physical orientations. Unlike external labels or manually defined class priors, these visual cues are already available in every training image and can serve as object-specific orientation priors at no additional annotation cost. This observation raises a practical question: can we convert image-intrinsic orientation cues into explicit angle supervision for WS-OOD?

Motivated by this question, we propose GradOrient, a simple yet effective gradient-guided orientation learning strategy for weakly supervised oriented object detection. Inspired by histograms of oriented gradients (HOG) [3], GradOrient extracts dominant orientation cues from weakly annotated object regions and converts them into pseudo-angle labels for auxiliary supervision. Specifically, we estimate object-specific dominant directions using magnitude-weighted local gradient orientation histograms and assign confidence scores to suppress unreliable pseudoangle signals. Unlike purely geometric formulations, the proposed supervision directly encourages the detector to learn orientation-aware representations from image evidence. Since GradOrient uses only the input image and weak annotations, it does not require RBox labels, additional manual annotation, or extra learnable parameters. Although we validate our method in the HBox-to-RBox setting, the proposed strategy can be incorporated into various WS-OOD frameworks as a plug-in supervision signal.

In addition to training supervision, we identify an important limitation in the evaluation protocol of existing OOD and WS-OOD methods. Current studies predominantly rely on average precision (AP), which evaluates detection quality based on classification confidence and spatial overlap. While AP is widely used for object detection, it does not explicitly measure orientation accuracy. In many cases, predictions with substantially different angles can still achieve sufficiently high IoU values and be counted as true positives [5, 19, 27, 28]. As shown in Fig. 1(a), this issue becomes more severe for objects whose width and height are similar, where large angular errors may have only a limited effect on IoU. Since object orientation is critical for downstream remote sensing tasks such as maritime monitoring, aircraft pose interpretation, and infrastructure inspection, AP alone cannot fully reflect the quality of angle estimation.

To provide a more faithful assessment of orientation prediction quality, we introduce an orientation-aware evaluation metric that explicitly penalizes angular discrepancies between predicted and target boxes. The proposed metric preserves the conventional AP evaluation pipeline while incorporating an angular penalty into the matching criterion, enabling complementary analysis of both localization and orientation quality. We further evaluate existing OOD methods with the proposed metric, showing that conventional AP can mask substantial angle errors and that our gradientguided supervision improves orientation estimation beyond standard IoU-based evaluation. Our main contributions are summarized as follows:

• We propose GradOrient, a gradient-guided pseudoangle supervision strategy that improves WS-OOD without requiring RBox annotations.

• We introduce an image-intrinsic orientation prior that extracts dominant gradient directions from input data and can be plugged into existing WS-OOD frameworks.

• We present an orientation-aware evaluation metric that complements conventional AP by explicitly penalizing angular discrepancies in oriented detection.

## 2. Related Work

## 2.1. RBox-supervised Oriented Object Detection

Early oriented object detection (OOD) methods extended conventional object detectors to predict rotated bounding boxes (RBoxes), including Rotated RetinaNet [11], Rotated FCOS [18], RoI Transformer [4], Gliding Vertex [22], Oriented R-CNN [21], and Oriented RepPoints [10]. Subsequent studies have further improved orientation modeling and localization quality. R<sup>3</sup>Det [23] employs progressive regression with feature reconstruction, S<sup>2</sup>A-Net [6] aligns features according to object orientations, and ReDet [7] introduces rotation-equivariant representations. In addition, Gaussian-based formulations such as GWD [24], KLD [25], and KFIoU [28] alleviate the angle boundary discontinuity problem through alternative box representations and loss functions. Despite these advances, most OOD methods rely on fully annotated RBoxes, which are considerably more expensive to obtain than horizontal annotations. This annotation burden has motivated weakly supervised OOD.

## 2.2. Weakly Supervised Oriented Object Detection

Weakly supervised oriented object detection aims to reduce the annotation cost of RBoxes by learning from weaker supervision, such as point annotations or horizontal bounding boxes (HBoxes). Depending on the supervision type, existing methods can be broadly categorized into point-supervised and HBox-supervised approaches. More recently, Wholly-WOOD [32] provides a unified framework for both weakly and fully supervised OOD, while PWOOD [12] further explores a partially weakly supervised setting with mixed supervision.

Point-supervised Oriented Object Detection. Pointsupervised OOD [1, 8, 31, 33, 34] reduces annotation cost by replacing bounding boxes with single-point labels. Point2RBox [30] introduces an end-to-end framework that combines synthetic visual pattern knowledge with transformation self-supervision. PointOBB [15] generates pseudo RBoxes through scale-sensitive consistency learning and multiple instance learning, while PointOBB-v2 [16] further improves pseudo-label quality using class probability maps and geometric priors. These methods show that oriented detection can be learned from sparse annotations, but accurate angle estimation still depends heavily on indirectly generated pseudo labels and geometric assumptions.

HBox-supervised Oriented Object Detection. HBoxsupervised OOD [5, 14, 17, 35] has attracted increasing attention because HBoxes provide stronger localization cues than points while remaining much cheaper to annotate than RBoxes. H2RBox [26] learns object orientations through rotation consistency, and H2RBox-v2 [29] further exploits symmetry and flip consistency. EIE-Det [19] enhances equivariance-based learning through instance-level orientation constraints, while ABBSPO [9] leverages object symmetry as an additional cue for improving orientation estimation. Although these methods are effective, they mainly infer orientations through transformed-view consistency or predefined geometric priors. As a result, the supervision for absolute angle estimation remains indirect.

Position of Our Work. Unlike prior HBox-supervised methods [9, 19, 26, 29], our work exploits image-intrinsic gradient evidence as a practical orientation prior. We convert gradient orientation statistics from weakly annotated object regions into explicit pseudo-angle supervision, complementing existing geometric consistency constraints without requiring RBox annotations. Beyond training, we also introduce an orientation-aware evaluation metric that reveals angle errors overlooked by conventional AP. Together, our method provides both a simple supervision signal and a more faithful evaluation protocol for practical WS-OOD.

## 3. Method

## 3.1. Overall Pipeline

The overall pipeline of the proposed framework is illustrated in Fig. 2. We adopt H2RBox [26], H2RBox-v2 [29], and ABBSPO [9] as baseline architectures. The input image and its rotated and flipped views are processed through a backbone and a Feature Pyramid Network (FPN). A shared detection head then predicts class probabilities, centerness, and spatial parameters of the RBox.

On top of this WS-OOD framework, we introduce a gradient-guided orientation learning module that provides pseudo-angle supervision. The module estimates dominant orientation cues from image gradients within each annotated HBox and uses them as auxiliary targets for angle learning. This allows the detector to exploit image-intrinsic orientation priors without requiring RBox annotations.

## 3.2. Pseudo-Angle Generator

To obtain a gradient-guided pseudo-angle $\theta _ { \mathrm { g r a d } }$ for orientation learning, we use local image gradients within an inset region of each annotated HBox in the augmented training input. We first apply Gaussian smoothing to each inset HBox patch and compute pixel-level gradient responses. Then, we retain only the top-k% pixel-level gradient responses ranked by gradient magnitude within each inset HBox patch. The retained gradient orientations are converted into edge-tangent orientations and accumulated into a magnitude-weighted orientation histogram [3, 13].

For the i-th retained pixel, let $G _ { x } ( i )$ and $G _ { y } ( i )$ denote the horizontal and vertical image-gradient components, respectively. We define its gradient orientation $\alpha _ { i }$ and magnitude $m _ { i }$ as $\alpha _ { i } \ = \ \mathrm { a t a n 2 } ( G _ { y } ( i ) , G _ { x } ( i ) )$ and $\begin{array} { r l } { m _ { i } } & { { } = } \end{array}$ $\sqrt { G _ { x } ( i ) ^ { 2 } + G _ { y } ( i ) ^ { 2 } }$

Since $\alpha _ { i }$ corresponds to the edge-normal direction, we convert it into the edge-tangent orientation as:

$$
\phi _ { i } = \left( \alpha _ { i } + { \frac { \pi } { 2 } } \right) { \bmod { \pi } } .\tag{1}
$$

We then construct a magnitude-weighted orientation histogram using the retained pixels. Let R denote the set of retained pixels, B be the number of orientation bins, and $\Delta ~ = ~ \pi / B$ be the bin width. For each bin index $b \in \{ 0 , \ldots , B - 1 \}$ , we define the corresponding set of pixels as:

$$
\mathcal { R } _ { b } = \left\{ i \in \mathcal { R } \mid b \Delta \leq \phi _ { i } < ( b + 1 ) \Delta \right\} .\tag{2}
$$

![](images/7406ccbd6912055840e4acd88b0a23de9458daebe18d52363decf299a6016dac.jpg)  
Figure 2. Overall pipeline of GradOrient. A pseudo-angle generator estimates object-specific orientation cues from magnitude-weighted gradient orientation histograms within weakly annotated regions. The resulting pseudo angles are transformed according to rotated and flipped views and used for gradient-guided orientation supervision through threshold-based loss activation.

The value of the b-th bin is computed by accumulating the gradient magnitudes of pixels assigned to that bin:

$$
h _ { b } = \sum _ { i \in \mathscr { R } _ { b } } m _ { i } .\tag{3}
$$

We identify the dominant bin and define the center of its interval as the coarse center orientation:

$$
b ^ { * } = \arg \operatorname* { m a x } _ { b } h _ { b } , \qquad \theta _ { \mathrm { c } } = \left( b ^ { * } + \frac { 1 } { 2 } \right) \Delta .\tag{4}
$$

Using this coarse center as an anchor, we collect retained pixels whose edge-tangent orientations lie within one bin width of $\theta _ { \mathrm { c } }$

$$
\begin{array} { r } { \mathcal { N } = \left\{ i \in \mathcal { R } \vert \vert \mathcal { W } _ { \pi } \left( \phi _ { i } - \theta _ { \mathrm { c } } \right) \vert \leq \Delta \right\} , } \end{array}\tag{5}
$$

where $\mathscr { W } _ { \pi } ( \cdot )$ denotes a π-periodic wrapping function that maps an angle to the orientation range $[ - \pi / 2 , \pi / 2 )$ . The pseudo-angle is computed as a magnitude-weighted $\pi -$ periodic circular mean over the selected pixels:

$$
\begin{array} { c } { { S = \displaystyle \sum _ { i \in \mathcal { N } } m _ { i } \sin \left( 2 \phi _ { i } \right) , } } \\ { { { } } } \\ { { C = \displaystyle \sum _ { i \in \mathcal { N } } m _ { i } \cos \left( 2 \phi _ { i } \right) , } } \\ { { { } } } \\ { { \theta _ { \mathrm { g r a d } } = \displaystyle \mathcal { W } _ { \pi } \left[ \frac { 1 } { 2 } \tan 2 \left( S , C \right) \right] . } } \end{array}\tag{6}
$$

This formulation accounts for the π-periodicity of orientation, treating θ and $\theta + \pi$ as the same orientation.

We further estimate the reliability of the pseudo-angle from the concentration of the orientation histogram. The normalized histogram is given by:

$$
p _ { b } = \frac { h _ { b } } { \sum _ { j = 0 } ^ { B - 1 } h _ { j } } .\tag{7}
$$

The confidence score is defined as the complement of its normalized entropy:

$$
q _ { \mathrm { g r a d } } = 1 + \frac { \sum _ { b = 0 } ^ { B - 1 } p _ { b } \log p _ { b } } { \log B } .\tag{8}
$$

The resulting $\theta _ { \mathrm { g r a d } }$ and $q _ { \mathrm { g r a d } }$ are used as an object-specific orientation target and its confidence weight, respectively.

## 3.3. Gradient-Guided Orientation Supervision

Given the gradient-derived pseudo-angle $\theta _ { \mathrm { g r a d } } .$ , we transform it according to each training view so that the target is expressed in the same coordinate frame as the prediction. For $v \in \{ \mathrm { o r i } , \mathrm { r o t } , \mathrm { f l p } \}$ , the view-specific pseudo-angle target is defined as:

$$
\begin{array} { r } { \tilde { \theta } _ { \mathrm { g r a d } } ^ { v } = \left\{ \begin{array} { l l } { \theta _ { \mathrm { g r a d } } , } & { v = \mathrm { o r i } , } \\ { \mathcal { W } _ { \pi } \left( \theta _ { \mathrm { g r a d } } + R \right) , } & { v = \mathrm { r o t } , } \\ { \mathcal { W } _ { \pi } \left( - \theta _ { \mathrm { g r a d } } \right) , } & { v = \mathrm { f l p } , } \end{array} \right. } \end{array}\tag{9}
$$

where R denotes the rotation angle applied to the rotated view. The confidence score $q _ { \mathrm { g r a d } }$ is shared across the three view-specific targets.

For each prediction, we compute the π-periodic angular discrepancy between the predicted angle and the corresponding pseudo-angle target:

$$
\delta = \left| \mathcal { W } _ { \pi } \left( \theta _ { \mathrm { p r e d } } ^ { v } - \tilde { \theta } _ { \mathrm { g r a d } } ^ { v } \right) \right| , \quad \quad \delta \in \left[ 0 , \frac { \pi } { 2 } \right] .\tag{10}
$$

Since local gradients may indicate either object axis, a pseudo-angle shifted by $\pi / 2$ remains valid. We therefore relax the angular discrepancy as:

$$
d _ { \mathrm { g r a d } } = \operatorname* { m i n } \left( \delta , \frac { \pi } { 2 } - \delta \right) , \qquad d _ { \mathrm { g r a d } } \in \left[ 0 , \frac { \pi } { 4 } \right] .\tag{11}
$$

This relaxation treats orthogonal object axes as equivalent for gradient-based supervision, while box regression handles the width-height assignment.

We apply the gradient-guided loss only to predictions whose relaxed angular distance is within the threshold $\tau _ { \mathrm { g r a d } } .$ Let V denote the set of valid assigned predictions:

$$
\mathcal { V } = \left\{ n \Big | d _ { \mathrm { g r a d } } ^ { ( n ) } \leq \tau _ { \mathrm { g r a d } } \right\} .\tag{12}
$$

The gradient-guided loss is then defined as:

$$
\mathcal { L } _ { \mathrm { g r a d } } = \frac { 1 } { | \mathcal { P } | } \sum _ { n \in \mathcal { V } } q _ { \mathrm { g r a d } } ^ { ( n ) } \mathrm { s m o o t h } _ { L 1 } \left( d _ { \mathrm { g r a d } } ^ { ( n ) } \right) ,\tag{13}
$$

where P denotes the set of positive predictions. For each prediction, the pseudo-angle target and confidence score are obtained from its matched GT instance. The confidence score weights the reliability of the gradient-derived pseudoangle, and predictions outside the angular threshold are excluded from the gradient-guided supervision.

## 3.4. Loss Functions

Following multi-view HBox-supervised OOD methods [9, 29], we regularize angle predictions under rotation and vertical flipping. Let θ, $\theta _ { \mathrm { r o t } } .$ , and $\theta _ { \mathrm { f l p } }$ denote the predicted angles from the original, rotated, and flipped views, respectively. The rotation- and flip-consistency losses are defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r o t } } = \ell _ { s } ( \theta _ { \mathrm { r o t } } - \theta , R ) , \quad \mathcal { L } _ { \mathrm { f l p } } = \ell _ { s } ( \theta _ { \mathrm { f l p } } + \theta , 0 ) , } \end{array}\tag{14}
$$

where R is the rotation angle applied to the rotated view, and $\ell _ { s }$ denotes the Smooth L1-based snap loss [29].

The multi-view angle consistency loss is then given by:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a n g } } = \lambda _ { r } \mathcal { L } _ { \mathrm { r o t } } + \lambda _ { f } \mathcal { L } _ { \mathrm { f l p } } . } \end{array}\tag{15}
$$

$$
\mathcal { L } _ { \mathrm { a n g } }
$$

$$
\mathcal { L } _ { \mathrm { g r a d } }
$$

The baseline training objective is defined as:

$$
\mathcal { L } _ { \mathrm { b a s e } } = \lambda _ { \mathrm { a n g } } \mathcal { L } _ { \mathrm { a n g } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } + \lambda _ { \mathrm { c n } } \mathcal { L } _ { \mathrm { c n } } + \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } ,\tag{16}
$$

where $\mathcal { L } _ { \mathrm { r e g } } , \mathcal { L } _ { \mathrm { c n } }$ , and $\mathcal { L } _ { \mathrm { c l s } }$ denote the IoU-based regression loss [9], centerness loss, and focal loss-based classification loss [29], respectively. The overall objective is defined as

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { b a s e } } + \lambda _ { \mathrm { g r a d } } \mathcal { L } _ { \mathrm { g r a d } } .\tag{17}
$$

## 4. Orientation-aware Evaluation Metric

The standard average precision (AP) metric does not explicitly account for angular errors. An OBB prediction with a significantly misaligned angle can still be counted as a true positive (TP) as long as its spatial intersection over union (IoU) satisfies the threshold. To address this issue, we propose orientation-aware AP (OAP), which incorporates an angular penalty into the standard AP evaluation procedure.

![](images/25852499ffa85f03800f16fd0a2cd6001f6b07aaa5633d7a456bbf9a79281425.jpg)  
Figure 3. Comparison of angular penalty weights W. We adopt the square-root penalty, which imposes a stronger penalty for small angular errors than cosine, linear, and square penalties.

To measure the angular discrepancy consistently, we extract the predicted and ground-truth (GT) orientations, denoted by $\theta _ { \mathrm { p r e d } }$ and $\theta _ { \mathrm { g t } }$ , respectively, under the long-edge 90° (le90) representation. Since oriented boxes exhibit periodic angular representations, we compute the acute angular difference $\Delta \theta \in [ 0 , \pi / 2 ]$ as follows:

$$
\Delta \theta = \mathrm { m i n } ( | \theta _ { \mathrm { p r e d } } - \theta _ { \mathrm { g t } } | , \pi - | \theta _ { \mathrm { p r e d } } - \theta _ { \mathrm { g t } } | ) .\tag{18}
$$

We then define an angular penalty weight $W$ as:

$$
W = 1 - \sqrt { \frac { \Delta \theta } { \theta _ { \mathrm { { m a x } } } } } ,\tag{19}
$$

where $\theta _ { \mathrm { { m a x } } } = \pi / 2$ denotes the maximum possible angular deviation. The weight $W \in [ 0 , 1 ]$ represents the degree of orientation agreement between the predicted and GT boxes. When the two boxes have the same orientation, $\Delta \theta = 0$ and thus $W = 1$ , indicating that no angular penalty is applied. As the angular error increases, W decreases, and it becomes 0 when the angular deviation reaches $\theta _ { \mathrm { m a x } }$

Fig. 3 compares several choices for the angular penalty function, including cosine, linear, square, and square-root forms. We adopt the square-root penalty because it decreases more rapidly than the other alternatives for small angular errors. This behavior is desirable for orientationaware evaluation, since even a moderate angular mismatch can indicate inaccurate orientation estimation, especially when the spatial IoU remains high. Therefore, the proposed penalty enforces stricter orientation consistency while still preserving the standard IoU-based localization criterion.

Using this angular weight, we define the Orientation-Aware IoU (OIoU) as $W \times \mathrm { I o U }$ . During evaluation, a prediction is considered a TP only when its OIoU exceeds a predefined threshold (e.g., 0.5 for ${ \mathrm { O A P } } _ { 5 0 } )$ . The standard AP calculation is then applied using these orientation-aware matches, yielding the final OAP score. As a result, OAP preserves the conventional AP evaluation pipeline while penalizing predictions with inaccurate orientations.

<table><tr><td></td><td>|Methods</td><td>APL</td><td>APO</td><td>BF</td><td>BC</td><td>BR</td><td>CH</td><td></td><td>ESA</td><td>ETS</td><td>DAM</td><td>GF</td><td>GTF</td><td>HA</td><td>OP</td><td>SH</td><td>STA</td><td>STO</td><td>TC</td><td>TS</td><td>VE</td><td>WM|</td><td> $\mathrm { \ A P _ { 5 0 } }$ </td></tr><tr><td rowspan="6">Se</td><td>RetinaNet [11]</td><td>59.8</td><td>19.3</td><td>69.7</td><td>81.3</td><td>17.2</td><td>72.7</td><td>68.7</td><td></td><td>49.4</td><td>18.4</td><td>69.5</td><td>71.3</td><td>33.3</td><td>34.1</td><td>75.8</td><td>67.1</td><td>59.6</td><td>81.0</td><td>44.1</td><td>38.0</td><td>62.5</td><td>54.64</td></tr><tr><td>FCOS [18]</td><td>62.1</td><td>37.9</td><td>74.6</td><td>81.2</td><td>32.9</td><td>72.1</td><td>75.3</td><td></td><td>61.8</td><td>27.4</td><td>69.1</td><td>78.7</td><td>34.4</td><td>50.6</td><td>80.1</td><td>68.6</td><td>68.1</td><td>81.3</td><td>49.1</td><td>43.4</td><td>64.5</td><td>60.66</td></tr><tr><td>Oriented R-CNN [21]</td><td>63.0</td><td>36.7</td><td>71.9</td><td>81.6</td><td>41.1</td><td>72.6</td><td></td><td>77.8</td><td>65.5</td><td>24.8</td><td>72.9</td><td>82.1</td><td>40.9</td><td>56.5</td><td>81.2</td><td>73.4</td><td>62.4</td><td>81.5</td><td>53.3</td><td>43.3</td><td>65.6</td><td>62.41</td></tr><tr><td>GWD [24] (RetinaNet)</td><td>61.5</td><td>23.6</td><td></td><td>73.6 81.1</td><td></td><td>17.4</td><td>72.7</td><td>68.3</td><td>47.2</td><td>20.7</td><td>71.2</td><td>73.2</td><td>33.9</td><td>34.3</td><td>77.6</td><td>64.7</td><td>57.5</td><td>80.9</td><td>42.1</td><td>39.7</td><td>60.2</td><td>55.07</td></tr><tr><td>KLD [25] (RetinaNet)</td><td>57.8</td><td>22.6</td><td>71.5</td><td>81.2</td><td>16.9</td><td>72.7</td><td></td><td>68.9</td><td>52.1</td><td>20.6</td><td>73.5</td><td>71.0</td><td>33.7</td><td>33.2</td><td>77.1</td><td>68.9</td><td>59.9</td><td>80.9</td><td>43.9</td><td>39.1</td><td>60.9</td><td>55.32</td></tr><tr><td>KFIoU [28] (RetinaNet)</td><td>60.6</td><td>36.6</td><td>73.6</td><td>80.9</td><td>27.0</td><td>72.6</td><td></td><td>73.4</td><td>56.5</td><td>25.4</td><td>73.9</td><td>72.0</td><td>32.9</td><td>45.8</td><td>75.8</td><td>65.2</td><td>57.6</td><td>80.0</td><td>48.0</td><td>40.1</td><td>58.8</td><td>57.84</td></tr><tr><td rowspan="5">SH</td><td>H2RBox [26]</td><td>57.1</td><td>14.4</td><td>72.2</td><td>82.6</td><td>17.5</td><td>71.2</td><td></td><td>56.5</td><td>55.2</td><td>14.0</td><td>67.7</td><td>77.9</td><td>31.0</td><td>40.7</td><td>76.3</td><td>66.2</td><td>63.4</td><td>81.5</td><td>50.4</td><td>38.0</td><td>57.6</td><td>54.57</td></tr><tr><td>H2RBox-v2 [29]</td><td>55.5</td><td></td><td>17.8</td><td>76.9</td><td>80.5</td><td>27.7</td><td>72.2</td><td>63.0</td><td>58.6</td><td>24.4</td><td>73.9</td><td>80.3</td><td>33.9</td><td>47.2</td><td>77.4</td><td>58.7</td><td>60.9</td><td>81.4</td><td>48.1</td><td>41.1</td><td>53.9</td><td>56.67</td></tr><tr><td>EIE-Det*† [19]</td><td>65.2</td><td></td><td>29.4</td><td>68.5</td><td>83.1</td><td>26.6</td><td>72.2</td><td>72.0</td><td>62.9</td><td>23.7</td><td>67.6</td><td>77.1</td><td>32.5</td><td>45.5</td><td>79.8</td><td>60.2</td><td>67.1</td><td>81.3</td><td>51.9</td><td>42.7</td><td>63.5</td><td>58.64</td></tr><tr><td>ABBSPO† [9]</td><td>66.4</td><td></td><td>28.9</td><td>76.4</td><td>85.6</td><td>32.5</td><td>72.1</td><td>73.5</td><td>59.9</td><td>23.7</td><td>70.7</td><td>81.0</td><td>35.2</td><td>49.9</td><td>78.6</td><td>66.7</td><td>66.1</td><td>81.4</td><td>49.1</td><td>41.2</td><td>62.1</td><td>60.10</td></tr><tr><td>GradOrient (Ours)</td><td>67.8</td><td>37.0</td><td>76.1</td><td>87.1</td><td>32.0</td><td>72.1</td><td></td><td>73.1</td><td>64.1</td><td>26.5</td><td>72.0</td><td>81.4</td><td>35.9</td><td>51.9</td><td>79.4</td><td>67.1</td><td>67.0</td><td>81.5</td><td>53.1</td><td>43.1</td><td>62.3</td><td>61.50</td></tr></table>

Table 1. Results of each category on the DIOR-R [2] for RBox-supervised $( S _ { R } )$ and HBox-supervised $( S _ { H } )$ . The performance of the compared methods is reported from a previous study [9]. \* indicates our own implementation, and <sup>†</sup> denotes models trained in this work.

<table><tr><td></td><td>Methods</td><td>PL</td><td>BD</td><td>BR</td><td>GTF</td><td>SV</td><td>LV</td><td>SH</td><td>TC</td><td>BC</td><td>ST</td><td>SBF</td><td>RA</td><td>HA</td><td>SP</td><td>HC</td><td> $\mathrm { A P _ { 5 0 } }$ </td></tr><tr><td rowspan="7"> $\tilde { \varsigma }$ </td><td>RetinaNet [11]</td><td>87.5</td><td>75.1</td><td>39.9</td><td>59.6</td><td>66.3</td><td>66.3</td><td>78.2</td><td>90.5</td><td>55.0</td><td>62.7</td><td>47.1</td><td>63.6</td><td>59.4</td><td>55.1</td><td>43.0</td><td>63.3</td></tr><tr><td>FCOS [18]</td><td>88.8</td><td>74.0</td><td>46.8</td><td>59.1</td><td>70.1</td><td>81.4</td><td>87.7</td><td>90.7</td><td>67.7</td><td>68.3</td><td>60.2</td><td>66.1</td><td>64.9</td><td>58.7</td><td>44.0</td><td>68.6</td></tr><tr><td>Oriented R-CNN [21]</td><td>89.3</td><td>76.1</td><td>53.8</td><td>78.7</td><td>68.6</td><td>84.9</td><td>89.3</td><td>90.8</td><td>74.3</td><td>62.8</td><td>66.3</td><td>66.5</td><td>74.7</td><td>58.6</td><td>46.8</td><td>72.1</td></tr><tr><td>Oriented RepPoints [10]</td><td>89.7</td><td>80.1</td><td>50.5</td><td>74.4</td><td>75.0</td><td>82.0</td><td>88.7</td><td>90.4</td><td>64.0</td><td>70.0</td><td>45.7</td><td>60.6</td><td>73.6</td><td>60.4</td><td>42.8</td><td>69.86</td></tr><tr><td>GWD [24] (RetinaNet)</td><td>88.2</td><td>74.9</td><td>41.3</td><td>60.5</td><td>66.7</td><td>68.1</td><td>85.8</td><td>90.5</td><td>50.4</td><td>66.8</td><td>45.8</td><td>65.1</td><td>60.7</td><td>52.9</td><td>38.9</td><td>63.77</td></tr><tr><td>KLD [25] (RetinaNet)</td><td>88.4</td><td>75.8</td><td>41.4</td><td>60.0</td><td>66.1</td><td>68.8</td><td>84.7</td><td>90.6</td><td>56.8</td><td>60.4</td><td>50.4</td><td>70.1</td><td>60.0</td><td>50.5</td><td>45.7</td><td>64.65</td></tr><tr><td>KFIoU [28] (RetinaNet)</td><td>84.4</td><td>74.3</td><td>40.7</td><td>55.2</td><td>57.9</td><td>56.9</td><td>76.4</td><td>71.2</td><td>46.1</td><td>64.8</td><td>54.3</td><td>65.0</td><td>58.3</td><td>48.7</td><td>42.9</td><td>59.81</td></tr><tr><td rowspan="5"> $\breve { \sigma }$ </td><td>H2RBox [26]</td><td>89.5</td><td>73.1</td><td>37.3</td><td>55.1</td><td>70.7</td><td>76.4</td><td>85.4</td><td>90.3</td><td>66.5</td><td>67.3</td><td>59.6</td><td>64.9</td><td>60.6</td><td>57.9</td><td>36.5</td><td>66.07</td></tr><tr><td>H2RBox-v2 [29]</td><td>89.4</td><td>74.8</td><td>45.4</td><td>56.0</td><td>70.3</td><td>76.6</td><td>87.9</td><td>90.5</td><td>69.3</td><td>67.5</td><td>56.7</td><td>64.7</td><td>65.3</td><td>55.5</td><td>45.5</td><td>67.69</td></tr><tr><td> $\mathrm { E I E - D e t ^ { * } } \left[ 1 9 \right]$ </td><td>89.1</td><td>70.8</td><td>44.5</td><td>56.0</td><td>70.1</td><td>74.8</td><td>84.2</td><td>90.4</td><td>61.7</td><td>66.5</td><td>56.0</td><td>61.4</td><td>62.3</td><td>51.6</td><td>46.8</td><td>65.75</td></tr><tr><td>ABBSPO† [9]</td><td>89.1</td><td>75.7</td><td>47.3</td><td>58.1</td><td>69.9</td><td>77.8</td><td>88.1</td><td>90.5</td><td>69.3</td><td>67.8</td><td>67.1</td><td>64.3</td><td>71.8</td><td>55.3</td><td>44.9</td><td>69.13</td></tr><tr><td>GradOrient (Ours)</td><td>89.4</td><td>75.2</td><td>46.6</td><td>58.0</td><td>71.1</td><td>77.9</td><td>88.2</td><td>90.5</td><td>72.0</td><td>68.6</td><td>67.3</td><td>65.7</td><td>71.5</td><td>56.4</td><td>56.0</td><td>70.29</td></tr></table>

Table 2. Results of each category on the DOTA-v1.0 [20] for RBox-supervised $( S _ { R } )$ and HBox-supervised $( S _ { H } )$ . Results for the compared methods are taken from a previous study [9]. \* indicates our own implementation, and <sup>†</sup> denotes models trained in this work.

## 5. Experiments

## 5.1. Setup

Datasets. We utilize two prominent benchmarks for aerial object detection: DOTA-v1.0 [20] and DIOR-R [2]. DOTAv1.0 is a comprehensive dataset comprising 2,806 images with resolutions between 800 and 4,000 pixels, featuring 188,282 oriented bounding box annotations distributed across 15 classes. Similarly, DIOR-R extends the original DIOR dataset with oriented labels, offering 23,463 remote sensing images and 192,518 instances across 20 categories. Implementation Details. Our method is implemented based on the ABBSPO [9] framework, and all models are trained for 12 epochs under the same network setup. All experiments are conducted using an NVIDIA RTX 4090 GPU. For DOTA-v1.0, we train on the training set and evaluate on the validation set, as our angle-based metric requires access to ground-truth annotations. We use $\tau _ { \mathrm { g r a d } } = 6 ^ { \circ }$ for DIOR-R and 4<sup>◦</sup> for DOTA-v1.0. We fix the loss weights to $\lambda _ { r } = 1 . 0 .$ $\lambda _ { f } ~ = ~ 0 . 0 5 , ~ \lambda _ { \mathrm { a n g } } ~ = ~ 0 . 6 , ~ \lambda _ { \mathrm { r e g } } ~ = ~ \lambda _ { \mathrm { c n } } ~ = ~ \lambda _ { \mathrm { c l s } } ~ = ~ 1 . 0 ,$ , and $\lambda _ { \mathrm { g r a d } } = 1 0 . 0$ , with $\beta = 0 . 1$ for the Smooth $L _ { 1 }$ loss.

Metrics Used in Experiments. We report four evaluation metrics: $\mathrm { A P _ { 5 0 } , O A P _ { 5 0 } }$ , AngleGap, and AngleAcc. All angular quantities are computed under the le90 convention. $\mathrm { { A P } _ { 5 0 } }$ follows the standard detection protocol, where a prediction is considered a TP when its IoU with the matched GT box is at least 0.5. ${ \mathrm { O A P } } _ { 5 0 }$ follows the same AP calculation procedure but uses OIoU instead of IoU for matching, thereby accounting for both spatial overlap and orientation consistency. AngleGap measures the average angular difference between the predicted and GT boxes among TP pairs matched under $\mathrm { { A P } _ { 5 0 } }$ , where a lower value indicates more accurate orientation estimation. $\operatorname { A n g l e A c c } _ { k ^ { \circ } }$ denotes the percentage of $\mathrm { { A P } _ { 5 0 } }$ -matched TP pairs whose angular error is within $k ^ { \circ }$ , and we report AngleAcc from $1 ^ { \circ }$ to 5<sup>◦</sup>.

## 5.2. Results

Quantitative Results. Tabs. 1 and 2 compare GradOrient with existing RBox-supervised and HBox-supervised oriented object detectors under the conventional $\mathrm { { A P } _ { 5 0 } }$ metric. GradOrient achieves 61.50 $\mathrm { { A P } _ { 5 0 } }$ on DIOR-R and 70.29 $\mathrm { { A P } _ { 5 0 } }$ on DOTA-v1.0, outperforming the reproduced HBoxsupervised baseline ABBSPO<sup>†</sup> by 1.40 and 1.16 points, respectively. Tabs. 3 and 4 further report the orientationaware evaluation results. GradOrient improves ${ \mathrm { O A P } } _ { 5 0 }$ from 41.8 to 46.1 on DIOR-R and from 54.6 to 55.2 on DOTA, while reducing AngleGap from $5 . 0 1 9 ^ { \circ }$ to 4.159<sup>◦</sup> and from $3 . 6 7 6 ^ { \circ }$ to 3.348<sup>◦</sup>, respectively. These results indicate that the proposed gradient-guided pseudo-angle supervision improves both detection accuracy and orientation estimation.

<table><tr><td rowspan="3">Methods</td><td rowspan="3"></td><td colspan="10"> ${ \mathrm { O A P } } _ { 5 0 }$ </td><td rowspan="2">AngleAccθ</td><td colspan="4"></td><td rowspan="2"></td><td rowspan="2">AngleGap  $\overline { { 5 ^ { \circ } } }$ </td></tr><tr><td>|APL APO</td><td>BC</td><td></td><td>BR ESA</td><td>ETS</td><td>DAM</td><td>GTF</td><td>HA</td><td>OP</td><td>SH TC</td><td>TS</td><td>VE</td><td>|ALL| 1°</td><td>2°</td><td>3° 4°</td></tr><tr><td>RetinaNet [11]</td><td>22.4 6.3</td><td></td><td>79.5</td><td>10.3 54.7</td><td>41.2</td><td>7.0</td><td>69.2</td><td>23.9</td><td>28.3 68.1</td><td>80.7</td><td>34.3</td><td>30.4</td><td>39.7</td><td>39.84 60.15</td><td>71.36</td><td>78.54 83.29</td><td>5.263</td></tr><tr><td rowspan="6">Se</td><td>FCOS [18]</td><td>24.9</td><td>17.6</td><td>79.6 24.9</td><td>50.2</td><td>48.8</td><td>10.2</td><td>71.7</td><td>17.4 42.0</td><td>69.0</td><td>81.1</td><td>33.0</td><td>32.2</td><td>43.0</td><td>28.70 49.16</td><td>62.80</td><td>71.90 77.95</td><td>5.317</td></tr><tr><td>Oriented R-CNN [21]</td><td>40.6</td><td>21.0</td><td>81.2 34.3</td><td>65.3</td><td>59.3</td><td>17.5</td><td>80.6</td><td>32.9 49.0</td><td>80.7</td><td>81.4</td><td>44.9</td><td>41.1 52.1</td><td>43.67</td><td>64.98</td><td>77.20</td><td>84.68 89.33</td><td>3.262</td></tr><tr><td>GWD [24] (RetinaNet)</td><td>13.8</td><td>10.5 77.3</td><td>9.7</td><td>50.7</td><td>37.9</td><td>3.9</td><td>68.3 22.8</td><td>27.1</td><td>68.5</td><td>80.7</td><td>32.6 29.8</td><td>38.1</td><td>31.30</td><td>52.23 65.77</td><td>74.24</td><td>79.56</td><td>5.911</td></tr><tr><td>KLD [25] (RetinaNet)</td><td>18.2</td><td>11.3 77.6</td><td>12.1</td><td>52.6</td><td>39.7</td><td>7.4</td><td>69.0</td><td>25.4 30.2</td><td>68.9</td><td>80.7</td><td>32.9 30.9</td><td>39.8</td><td>33.16</td><td>55.11</td><td>68.27 76.41</td><td>81.47</td><td>5.567</td></tr><tr><td>KFIoU [28] (RetinaNet)</td><td>17.9</td><td>17.9 70.1</td><td>18.1</td><td>49.0</td><td>37.7</td><td>8.2</td><td>60.6</td><td>24.8 37.9</td><td>65.6</td><td>71.1</td><td>36.5</td><td>29.8 38.9</td><td>29.88</td><td>51.08</td><td>64.37</td><td>72.96 78.71</td><td>6.126</td></tr><tr><td></td><td></td><td>80.5</td><td>14.9</td><td>36.1</td><td>46.0</td><td>10.5</td><td>69.5</td><td>24.1 33.1</td><td>66.8</td><td>81.3</td><td>36.8 28.5</td><td>38.5</td><td>33.25</td><td>52.52</td><td>64.36 72.28</td><td>77.60</td><td></td></tr><tr><td rowspan="4">Sis</td><td>H2RBox [26] H2RBox-v2 [29]</td><td>9.7 22.6</td><td>0.9 8.0</td><td>80.5 26.2</td><td>34.2</td><td>43.8</td><td>13.9</td><td>74.0</td><td>26.0 43.1</td><td>66.7</td><td>81.3</td><td>33.5</td><td>33.4</td><td>42.0 29.75</td><td>51.74</td><td>66.63</td><td></td><td>5.387 4.624</td></tr><tr><td>EIE-Det* [19]</td><td>27.5</td><td>13.8</td><td>80.5 18.9</td><td>52.7</td><td>50.6</td><td>7.9</td><td>71.6</td><td>24.1 36.8</td><td>70.0</td><td>81.3</td><td>41.2</td><td>33.0</td><td>43.6 40.14</td><td>59.82</td><td>71.23</td><td>76.18 82.57 78.57 83.40</td><td>4.393</td></tr><tr><td>ABBSPO [9]</td><td>27.4</td><td>10.4</td><td>80.1 24.7</td><td>40.3</td><td>41.8</td><td>9.8</td><td>74.0</td><td>24.2 39.5</td><td>67.7</td><td>81.2</td><td>31.8</td><td>32.4 41.8</td><td>30.99</td><td>51.80</td><td>66.06 75.36</td><td>81.52</td><td>5.019</td></tr><tr><td>GradOrient (Ours)</td><td></td><td>20.2 80.6</td><td>25.5</td><td>51.6</td><td>50.8</td><td>14.4</td><td>74.5</td><td>26.6 44.0</td><td>69.3</td><td>81.4</td><td>39.5</td><td>33.5 46.1</td><td></td><td>35.70 59.43</td><td>74.15</td><td>82.91 88.14</td><td>4.159</td></tr><tr><td></td><td></td><td>33.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3. Orientation-aware results on the DIOR-R [2] for RBox-supervised $( S _ { R } )$ and HBox-supervised $( S _ { H } )$ . AngleAcc<sub>θ</sub> reports the percentage of $\mathrm { A P _ { 5 0 } }$ -matched TP pairs whose angular error is within each threshold. \* indicates our own implementation.

<table><tr><td rowspan="3">Methods RetinaNet [11]</td><td rowspan="3"></td><td colspan="10"> ${ \mathrm { O A P } } _ { 5 0 }$ </td><td colspan="5">AngleAccθ</td><td rowspan="3">AngleGap</td></tr><tr><td>PL BR</td><td></td><td>GTF</td><td>SV</td><td>LV</td><td>SH</td><td>TC BC</td><td>SBF</td><td>HA</td><td>SP</td><td>HC</td><td>ALL</td><td>1°</td><td>2° 3°</td><td> $4 ^ { \circ }$ </td><td>5°</td><td></td></tr><tr><td>61.0</td><td>24.8</td><td>48.8</td><td>59.5</td><td>60.7</td><td>73.4</td><td>90.3</td><td>46.9</td><td>41.8</td><td>38.8</td><td>20.8</td><td>7.0 47.8</td><td>32.14</td><td>55.93</td><td>72.02</td><td>81.86</td><td>87.80</td><td>4.106</td></tr><tr><td rowspan="8">Sn</td><td>FCOS [18]</td><td>63.1</td><td>26.8</td><td>42.8</td><td>60.6</td><td>75.7</td><td>77.1</td><td>90.3</td><td>54.4</td><td>39.8 38.7</td><td>23.5</td><td>16.0</td><td>50.7</td><td>26.32</td><td>47.64</td><td>63.00</td><td>73.80</td><td>81.13</td><td>4.349</td></tr><tr><td>Oriented R-CNN [21]</td><td>73.6</td><td>37.9</td><td>72.9</td><td>62.9</td><td>83.7</td><td>88.5</td><td>90.7</td><td>72.0</td><td>60.8</td><td>63.3</td><td>24.5 17.8</td><td>62.4</td><td>36.83</td><td>61.88</td><td>77.04</td><td>86.10</td><td>91.26</td><td>3.035</td></tr><tr><td>Oriented RepPoints [10]</td><td>74.2</td><td>31.9</td><td>64.5</td><td>66.2</td><td>78.7</td><td>87.3</td><td>90.0</td><td>53.1</td><td>41.0</td><td>52.0 25.8</td><td>9.1</td><td>56.1</td><td>33.80</td><td>58.84</td><td>74.66</td><td>84.40</td><td>90.03</td><td>3.304</td></tr><tr><td>GWD [24] (RetinaNet)</td><td>52.5</td><td>20.1</td><td>50.7</td><td>60.9</td><td>63.6</td><td>77.3</td><td>90.4</td><td>47.9</td><td>33.3</td><td>37.0</td><td>21.5 9.1</td><td>47.0</td><td>29.17</td><td>52.23</td><td>68.01</td><td>78.12</td><td>84.58</td><td>4.175</td></tr><tr><td>KLD [25] (RetinaNet)</td><td>49.9</td><td>17.8</td><td>52.3</td><td>58.2</td><td>61.5</td><td>76.0</td><td>90.2</td><td>45.5</td><td>35.5</td><td>32.4</td><td>20.3 8.7</td><td>45.7</td><td>27.92</td><td>49.82</td><td>65.44</td><td>75.67</td><td>82.22</td><td>4.629</td></tr><tr><td>KFIoU [28] (RetinaNet)</td><td>32.4</td><td>17.8</td><td>43.2</td><td>51.0</td><td>46.0</td><td>65.6</td><td>62.6</td><td>37.4</td><td>26.0</td><td>34.4</td><td>18.8 16.3</td><td>37.6</td><td>26.40</td><td>48.27</td><td>64.02</td><td>74.84</td><td>81.73</td><td>5.358</td></tr><tr><td></td><td>58.0</td><td>17.4</td><td>37.6</td><td>59.4</td><td>65.4</td><td>77.3</td><td>90.4</td><td>55.9</td><td>45.0</td><td>29.4 20.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>H2RBox [26] H2RBox-v2 [29]</td><td>63.4</td><td>29.7</td><td>45.3</td><td>60.7</td><td>73.5</td><td>78.5</td><td>90.3</td><td>65.5 49.8</td><td>41.8</td><td>23.6</td><td>11.4 16.2</td><td>47.3 53.2</td><td>28.22 31.66</td><td>49.86 55.52</td><td>65.41 71.32</td><td>76.19 81.60</td><td>83.17 87.73</td><td>4.360 3.752</td></tr><tr><td rowspan="5">SH</td><td>EIE-Det* [19]</td><td>58.4</td><td>23.1</td><td>46.6</td><td>61.2</td><td>66.3</td><td>76.9</td><td>90.3</td><td>55.5</td><td>35.7</td><td>30.4 20.6</td><td>24.7</td><td>49.1</td><td>27.10</td><td>49.37</td><td>65.88</td><td>76.86</td><td>84.10</td><td>4.415</td></tr><tr><td>ABBSPO [9]</td><td>64.0</td><td>28.9</td><td>45.0</td><td>61.4</td><td>74.3</td><td>79.0</td><td>90.4</td><td>66.8</td><td>51.3</td><td>46.6</td><td>23.9</td><td>23.9</td><td>54.6</td><td>31.50 55.68</td><td>71.97</td><td>81.85</td><td>87.99</td><td>3.676</td></tr><tr><td>GradOrient (Ours)</td><td>64.9</td><td>30.4</td><td>42.5</td><td>63.8</td><td>76.4</td><td>79.1</td><td>90.4</td><td>70.9</td><td>55.6</td><td>50.7</td><td>25.9 11.9</td><td>55.2</td><td>35.91</td><td>62.16</td><td>77.62</td><td>86.57</td><td>91.30</td><td>3.348</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 4. Orientation-aware results on the DOTA-v1.0 [20] for RBox-supervised $( S _ { R } )$ and HBox-supervised $( S _ { H } )$ . AngleAcc<sub>θ</sub> reports the percentage of $\mathrm { A P _ { 5 0 } }$ -matched TP pairs whose angular error is within each threshold. \* indicates our own implementation.

Rotated FCOS (ICCV’19)  
![](images/90ce402305248b015cc45eb6802bd8faf6946b67362c0c03e95a96b72da811ed.jpg)

H2RBox (ICLR’23)  
![](images/86251d82ec4cd02b1ac8713f2c63b020492a9dfcd0a63b0e9dbb6417e8469419.jpg)

H2RBox-v2 (NeurIPS’23)  
![](images/a66e55e6ec91584227114bec11488e162d66f9eaff8f6e3fa6c1171b7a47b7ad.jpg)

ABBSPO (CVPR’25)  
![](images/c18e830b0f6467426437861b1fa8082babd781f71a7a85934fdcfaf4a33fcd86.jpg)

GradOrient  
![](images/7e5166c106f8fc1d49007471cceb7bc33ef1975c845b70551dd30a6f9ccb7302.jpg)  
Figure 4. Visual comparison on DIOR-R and DOTA-v1.0. The top and bottom rows show results on DIOR-R and DOTA-v1.0, respectively.

Qualitative Results. Fig. 4 shows qualitative comparisons with existing oriented detectors. Compared with HBoxsupervised methods, GradOrient produces rotated boxes that are more consistent with object directions, especially for elongated objects and densely arranged instances. This demonstrates that image-gradient cues provide useful orientation information that complements HBox supervision.

Ablation Study. Tab. 5 analyzes the effects of the loss components, top-k, and $\lambda _ { \mathrm { g r a d } }$ Adding $\mathcal { L } _ { \mathrm { g r a d } }$ consistently improves ${ \mathrm { O A P } } _ { 5 0 }$ and reduces AngleGap on both DIOR-R and DOTA-v1.0. Notably, using $\mathcal { L } _ { \mathrm { g r a d } }$ without $\mathcal { L } _ { \mathrm { s p a } }$ achieves the best overall performance, suggesting that gradient-based pseudo-angle supervision provides a more direct and effective orientation cue than the symmetry-prior constraint, $\mathcal { L } _ { \mathrm { s p a } }$ used in ABBSPO [9]. This result supports our motivation that image-intrinsic gradient information can serve as a strong guidance signal for angle learning in WS-OOD. The results also show that GradOrient is relatively stable across different top-k and $\lambda _ { \mathrm { g r a d } }$ values. In addition, Tab. 6 shows that the method remains robust to $\tau _ { \mathrm { g r a d } } .$ , while the selected thresholds provide a favorable balance between detection accuracy and orientation-aware performance.

(a) Loss components
<table><tr><td rowspan="3"> $\mathcal { L } _ { \mathrm { s p a } } \ \mathcal { L } _ { \mathrm { g r a d } }$ </td><td rowspan="3"></td><td colspan="3">DIOR-R</td><td colspan="3">DOTA-v1.0</td></tr><tr><td> $\mathrm { A P _ { 5 0 } \ O A P _ { 5 0 } }$ </td><td></td><td>AngleGap</td><td></td><td> $\mathrm { A P _ { 5 0 } \ O A P _ { 5 0 } }$ </td><td>AngleGap</td></tr><tr><td rowspan="4">√ √</td><td></td><td>59.8</td><td>40.2</td><td>5.494</td><td>69.3</td><td>54.0</td><td>3.915</td></tr><tr><td></td><td>60.1</td><td>41.8</td><td>5.019</td><td>69.1</td><td>54.6</td><td>3.676</td></tr><tr><td>√</td><td>60.2</td><td>45.4</td><td>4.638</td><td>68.8</td><td>55.1</td><td>3.428</td></tr><tr><td>√</td><td>61.5</td><td>46.1</td><td>4.159</td><td>70.3</td><td>55.2</td><td>3.348</td></tr></table>

<table><tr><td rowspan="2">top-k (%)</td><td colspan="3">DIOR-R</td><td colspan="3">DOTA-v1.0</td></tr><tr><td> $\mathrm { A P _ { 5 0 } ~ O A P _ { 5 0 } ~ A n g l e G a p ~ A P _ { 5 0 } ~ O A P _ { 5 0 } ~ A n g l e G a p }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>75</td><td>60.4</td><td>44.5</td><td>4.549</td><td>68.6</td><td>54.7</td><td>3.347</td></tr><tr><td>50</td><td>59.2</td><td>43.7</td><td>4.611</td><td>69.4</td><td>54.7</td><td>3.341</td></tr><tr><td>25</td><td>61.5</td><td>46.1</td><td>4.159</td><td>70.3</td><td>55.2</td><td>3.348</td></tr><tr><td>10</td><td>60.3</td><td>45.3</td><td>4.563</td><td>68.7</td><td>55.0</td><td>3.315</td></tr><tr><td colspan="7">(b) Top-k ratio</td></tr></table>

<table><tr><td rowspan="3"> $\lambda _ { \mathrm { g r a d } }$ </td><td colspan="3">DIOR-R</td><td colspan="3">DOTA-v1.0</td></tr><tr><td colspan="6">AP50  ${ \mathrm { O A P } } _ { 5 0 }$  AngleGap  $\mathrm { A P _ { 5 0 } \ O A P _ { 5 0 } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td>54.3</td><td>3.769</td></tr><tr><td>0.1 1</td><td>60.4</td><td>42.9</td><td>5.239 4.562</td><td>68.7 69.5</td><td>55.3</td><td></td></tr><tr><td>10</td><td>61.4 61.5</td><td>44.1 46.1</td><td>4.159</td><td>70.3</td><td>55.2</td><td>3.526 3.348</td></tr><tr><td>20</td><td>60.9</td><td>45.7</td><td>4.150</td><td>68.3</td><td>53.9</td><td></td></tr><tr><td></td><td></td><td></td><td>(c) λ</td><td></td><td></td><td>3.352</td></tr></table>

Table 5. Ablation studies of GradOrient on DIOR-R and DOTA-v1.0. We evaluate the effects of components, top-k ratio, and $\lambda _ { \mathrm { g r a d } }$ in terms of $\mathrm { A P _ { 5 0 } , O A P _ { 5 0 } }$ , and AngleGap. Here, $\mathcal { L } _ { \mathrm { { s p a } } }$ denotes the symmetry-prior loss adopted from ABBSPO [9], while $\mathcal { L } _ { \mathrm { g r a d } }$ denotes our gradient-guided pseudo-angle loss. This indicates that gradient cues serve as a more direct image-intrinsic prior for angle learning.

<table><tr><td rowspan="2">(°)  $\tau _ { \mathrm { g r a d } }$ </td><td colspan="2">DIOR-R</td><td colspan="3">DOTA-v1.0</td></tr><tr><td> $\mathrm { { A P } _ { 5 0 } }$  OAP50 AngleGap AP50</td><td></td><td></td><td></td><td>OAP50 AngleGap</td></tr><tr><td>1</td><td>59.6</td><td>42.7</td><td>4.935</td><td>68.3</td><td>53.3 3.843</td></tr><tr><td>2</td><td>61.2</td><td>43.4</td><td>4.767 69.2</td><td>54.5</td><td>3.554</td></tr><tr><td>3</td><td>59.7</td><td>41.6</td><td>4.959 69.6</td><td>55.6</td><td>3.364</td></tr><tr><td>4</td><td>60.7</td><td>44.5</td><td>4.568</td><td>70.3 55.2</td><td>3.348</td></tr><tr><td>5</td><td>60.5</td><td>45.3</td><td>4.418</td><td>69.0 55.0</td><td>3.326</td></tr><tr><td>6</td><td>61.5</td><td>46.1</td><td>4.159</td><td>68.4 54.4</td><td>3.302</td></tr><tr><td>7</td><td>59.9</td><td>45.7</td><td>4.282</td><td>69.3 55.2</td><td>3.225</td></tr><tr><td>8</td><td>59.4</td><td>44.5</td><td>4.271</td><td>68.8 55.4</td><td>3.292</td></tr><tr><td>∞</td><td>58.0</td><td>44.2</td><td>4.293</td><td>67.0 52.9</td><td>3.507</td></tr></table>

Table 6. Sensitivity analysis of the hyperparameter $\tau _ { \mathrm { g r a d } }$ on the DIOR-R and DOTA-v1.0 datasets.
<table><tr><td rowspan="2">Framework</td><td rowspan="2">Setting</td><td colspan="3">DIOR-R</td><td colspan="3">DOTA-v1.0</td></tr><tr><td> $\mathrm { { A P } _ { 5 0 } }$ </td><td> ${ \mathrm { O A P } } _ { 5 0 }$ </td><td>AngleGap</td><td> $\mathrm { { A P } _ { 5 0 } }$ </td><td> ${ \mathrm { O A P } } _ { 5 0 }$ </td><td>AngleGap</td></tr><tr><td>EIE-Det [19]</td><td>Baseline</td><td>58.6</td><td>43.6</td><td>4.393</td><td>65.8</td><td>49.1</td><td>4.415</td></tr><tr><td rowspan="2">H2RBox [26]</td><td>w/  $\mathcal { L } _ { \mathrm { g r a d } }$ </td><td>60.0</td><td>45.7</td><td>4.226</td><td>67.3</td><td>49.6</td><td>4.242</td></tr><tr><td>Baseline</td><td>54.6</td><td>38.5</td><td>5.387</td><td>66.1</td><td>47.3</td><td>4.360</td></tr><tr><td rowspan="2">H2RBox-v2 [29]</td><td> $\mathrm { w } / \mathcal { L } _ { \mathrm { g r a d } }$ </td><td>57.5</td><td>41.6</td><td>5.007</td><td>67.1</td><td>50.4</td><td>3.932</td></tr><tr><td>Baseline</td><td>56.7</td><td>42.0</td><td>4.624</td><td>67.7</td><td>53.2</td><td>3.752</td></tr><tr><td rowspan="2">Rotated FCOS [18]</td><td> $\mathrm { w } / \mathcal { L } _ { \mathrm { g r a d } }$ </td><td>58.6</td><td>44.1</td><td>4.070</td><td>68.2</td><td>55.2</td><td>3.334</td></tr><tr><td>Baseline</td><td>60.7</td><td>43.0</td><td>5.317</td><td>68.6</td><td>50.7</td><td>4.349</td></tr><tr><td></td><td>w/  $\mathcal { L } _ { \mathrm { g r a d } }$ </td><td>60.7</td><td>44.8</td><td>5.152</td><td>67.5</td><td>51.6</td><td>3.959</td></tr></table>

Table 7. Generalizability of the gradient-guided orientation supervision on DIOR-R and DOTA-v1.0.

Generalizability analysis. Tab. 7 evaluates the generalizability of our gradient-guided orientation supervision across both HBox-supervised [19, 26, 29] and fully supervised frameworks [18]. We integrate our orientation supervisor into each framework and consistently observe improvements in ${ \mathrm { O A P } } _ { 5 0 }$ of 0.5–3.1 points, together with reductions in AngleGap of 3.1–12.0%. These results demonstrate the generalizability of our supervision across different supervision paradigms and framework designs.

Limitations of Pseudo-angle Estimation. Pseudo-angle estimation can become unreliable in atypical cases such as zebra-like objects, where strong internal textures produce gradients that are not aligned with the global object orientation. Similar misleading cues can also arise from cluttered backgrounds or intersecting boundaries. However, our gradient supervision is applied only when the pseudo-angle is consistent with the baseline orientation estimate through the proposed gate. Together with the original orientation supervision of the baseline, this design mitigates the influence of inaccurate pseudo-angles on orientation learning.

## 6. Conclusion

In this paper, we introduced GradOrient, a simple yet effective framework for HBox-supervised oriented object detection. Motivated by the observation that the image itself contains orientation cues, such as object boundaries and gradient patterns, GradOrient converts dominant gradient orientations within weakly annotated regions into pseudoangle supervision, providing a direct guidance signal for angle learning without requiring RBox annotations or extra learnable parameters. Experimental results show that this gradient-guided supervision consistently improves both detection accuracy and orientation estimation quality over existing HBox-supervised baselines.

Beyond training, we also highlight that accurate angle prediction should be explicitly considered in the evaluation of oriented object detection. Conventional IoU-based AP mainly reflects spatial overlap and can overlook substantial angular errors, especially when high IoU is preserved despite inaccurate orientation. To address this limitation, we proposed orientation-aware AP (OAP), which incorporates an angular penalty into the standard AP protocol and provides a complementary measure of orientation quality. We hope this metric encourages more careful analysis of angle prediction in future OOD research.

## References

[1] Guangming Cao, Xuehui Yu, Wenwen Yu, Xumeng Han, Xue Yang, Guorong Li, Jianbin Jiao, and Zhenjun Han. P2rbox: Point prompt oriented object detection with sam. arXiv preprint arXiv:2311.13128, 2023. 3

[2] Gong Cheng, Jiabao Wang, Ke Li, Xingxing Xie, Chunbo Lang, Yanqing Yao, and Junwei Han. Anchor-free oriented proposal generator for object detection. IEEE Transactions on Geoscience and Remote Sensing, 60:1–11, 2022. 6, 7

[3] Navneet Dalal and Bill Triggs. Histograms of oriented gradients for human detection. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), pages 886–893. Ieee, 2005. 2, 3

[4] Jian Ding, Nan Xue, Yang Long, Gui-Song Xia, and Qikai Lu. Learning roi transformer for oriented object detection in aerial images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2849– 2858, 2019. 2

[5] Chenlin Fu and Yingying Zhu. Bghr: Bridging the gap between hbox-supervised and rbox-supervised oriented object detection via adaptive fine-grained sample mining. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 3022–3030, 2025. 2, 3

[6] Jiaming Han, Jian Ding, Jie Li, and Gui-Song Xia. Align deep features for oriented object detection. IEEE transactions on geoscience and remote sensing, 60:1–11, 2021. 2

[7] Jiaming Han, Jian Ding, Nan Xue, and Gui-Song Xia. Redet: A rotation-equivariant detector for aerial object detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2786–2795, 2021. 3

[8] Shitian He, Huanxin Zou, Yingqian Wang, Boyang Li, Xu Cao, and Ning Jing. Learning remote sensing object detection with single point supervision. IEEE Transactions on Geoscience and Remote Sensing, 62:1–16, 2023. 3

[9] Woojin Lee, Hyugjae Chang, Jaeho Moon, Jaehyup Lee, and Munchurl Kim. ABBSPO: Adaptive bounding box scaling and symmetric prior based orientation prediction for detecting aerial image objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8848–8858, 2025. 2, 3, 5, 6, 7, 8

[10] Wentong Li, Yijie Chen, Kaixuan Hu, and Jianke Zhu. Oriented reppoints for aerial object detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 1829–1838, 2022. 2, 6, 7

[11] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In´ Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017. 2, 6, 7

[12] Mingxin Liu, Peiyuan Zhang, Yuan Liu, Wei Zhang, Yue Zhou, Ning Liao, Ziyang Gong, Junwei Luo, Zhirui Wang, Yi Yu, et al. Partial weakly-supervised oriented object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27644–27654, 2026. 3

[13] David G Lowe. Distinctive image features from scaleinvariant keypoints. International journal of computer vision, 60(2):91–110, 2004. 3

[14] Junyan Lu, Qinglei Hu, Ruifei Zhu, Yali Wei, and Tie Li. Afws: Angle-free weakly supervised rotating object detection for remote sensing images. IEEE Transactions on Geo science and Remote Sensing, 62:1–14, 2024. 3

[15] Junwei Luo, Xue Yang, Yi Yu, Qingyun Li, Junchi Yan, and Yansheng Li. Pointobb: Learning oriented object detection via single point supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16730–16740, 2024. 2, 3

[16] Botao Ren, Xue Yang, Yi Yu, Junwei Luo, and Zhidong Deng. Pointobb-v2: Towards simpler, faster, and stronger single point supervised oriented object detection. In International Conference on Learning Representations, pages 39843–39857, 2025. 3

[17] Yongqing Sun, Jie Ran, Feng Yang, Chenqiang Gao, Takayuki Kurozumi, Hideaki Kimata, and Ziqi Ye. Oriented object detection for remote sensing images based on weakly supervised learning. In 2021 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), pages 1– 6. IEEE, 2021. 3

[18] Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: Fully convolutional one-stage object detection. In Proceed ings of the IEEE/CVF international conference on computer vision, pages 9627–9636, 2019. 2, 6, 7, 8

[19] Linfei Wang, Yibing Zhan, Xu Lin, Baosheng Yu, Liang Ding, Jianqing Zhu, and Dapeng Tao. Explicit and implicit box equivariance learning for weakly-supervised rotated object detection. IEEE Transactions on Emerging Topics in Computational Intelligence, 9(1):509–521, 2025. 2, 3, 6, 7, 8

[20] Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, Serge Belongie, Jiebo Luo, Mihai Datcu, Marcello Pelillo, and Liangpei Zhang. Dota: A large-scale dataset for object detection in aerial images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3974–3983, 2018. 6, 7

[21] Xingxing Xie, Gong Cheng, Jiabao Wang, Xiwen Yao, and Junwei Han. Oriented r-cnn for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3520–3529, 2021. 2, 6, 7

[22] Yongchao Xu, Mingtao Fu, Qimeng Wang, Yukang Wang, Kai Chen, Gui-Song Xia, and Xiang Bai. Gliding vertex on the horizontal bounding box for multi-oriented object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(4):1452–1459, 2021. 2

[23] Xue Yang, Junchi Yan, Ziming Feng, and Tao He. R3det: Refined single-stage detector with feature refinement for ro tating object. In Proceedings of the AAAI conference on artificial intelligence, pages 3163–3171, 2021. 2

[24] Xue Yang, Junchi Yan, Qi Ming, Wentao Wang, Xiaopeng Zhang, and Qi Tian. Rethinking rotated object detection with gaussian wasserstein distance loss. In International conference on machine learning, pages 11830–11841. PMLR, 2021. 3, 6, 7

[25] Xue Yang, Xiaojiang Yang, Jirui Yang, Qi Ming, Wentao Wang, Qi Tian, and Junchi Yan. Learning high-precision bounding box for rotated object detection via kullback

leibler divergence. Advances in Neural Information Processing Systems, 34:18381–18394, 2021. 3, 6, 7

[26] Xue Yang, Gefan Zhang, Wentong Li, Yue Zhou, Xuehui Wang, and Junchi Yan. H2RBox: Horizontal box annotation is all you need for oriented object detection. In The Eleventh International Conference on Learning Representations, 2023. 2, 3, 6, 7, 8

[27] Xue Yang, Gefan Zhang, Xiaojiang Yang, Yue Zhou, Wentao Wang, Jin Tang, Tao He, and Junchi Yan. Detecting rotated objects as gaussian distributions and its 3-d generalization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4335–4354, 2023. 2

[28] Xue Yang, Yue Zhou, Gefan Zhang, Jirui Yang, Wentao Wang, Junchi Yan, XIAOPENG ZHANG, and Qi Tian. The KFIou loss for rotated object detection. In The Eleventh International Conference on Learning Representations, 2023. 2, 3, 6, 7

[29] Yi Yu, Xue Yang, Qingyun Li, Yue Zhou, Feipeng Da, and Junchi Yan. H2RBox-v2: Incorporating symmetry for boosting horizontal box supervised oriented object detection. Advances in Neural Information Processing Systems, 36:59137–59150, 2023. 2, 3, 5, 6, 7, 8

[30] Yi Yu, Xue Yang, Qingyun Li, Feipeng Da, Jifeng Dai, Yu Qiao, and Junchi Yan. Point2rbox: Combine knowledge from synthetic visual patterns for end-to-end oriented object detection with single point supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16783–16793, 2024. 2, 3

[31] Yi Yu, Botao Ren, Peiyuan Zhang, Mingxin Liu, Junwei Luo, Shaofeng Zhang, Feipeng Da, Junchi Yan, and Xue Yang. Point2rbox-v2: Rethinking point-supervised oriented object detection with spatial layout among instances. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19283–19293, 2025. 2, 3

[32] Yi Yu, Xue Yang, Yansheng Li, Zhenjun Han, Feipeng Da, and Junchi Yan. Wholly-wood: Wholly leveraging diversified-quality labels for weakly-supervised oriented object detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(6):4438–4454, 2025. 3

[33] Peiyuan Zhang, Junwei Luo, Xue Yang, Yi Yu, Qingyun Li, Yue Zhou, Xiaosong Jia, Xudong Lu, Jingdong Chen, Xiang Li, et al. Pointobb-v3: Expanding performance boundaries of single point-supervised oriented object detection. International Journal of Computer Vision, 133(9):6108–6128, 2025. 3

[34] Shun Zhang, Jihui Long, Yaohui Xu, and Shaohui Mei. Pmho: Point-supervised oriented object detection based on segmentation-driven proposal generation. IEEE Transactions on Geoscience and Remote Sensing, 62:1–18, 2024. 3

[35] Tianyu Zhu, Bryce Ferenczi, Pulak Purkait, Tom Drummond, Hamid Rezatofighi, and Anton Van Den Hengel. Knowledge combination to learn rotated detection without rotated annotation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15518–15527, 2023. 3