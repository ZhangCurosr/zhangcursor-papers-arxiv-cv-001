# Context-aware Bayesian Fusion for Multimodal Object Detection

Arunachalam Venkatachalam<sup>1</sup> Arunachalam.Venkatachalam@imec-int.com

<sup>1</sup> imec Michigan, USA

Mhafuzul Islam<sup>1</sup> <sup>6</sup>mhafuzul.islam@imec-int.com

Marwan Yusuf<sup>3</sup> 2marwan.yusuf@imec.be

<sup>2</sup> imec Leuven, Belgium

<sup>t</sup>David Van Hamme<sup>3</sup> david.vanhamme@imec.be

Ben Stoffelen<sup>2</sup> <sup>3</sup>ben.stoffelen@imec.be

<sup>3</sup> imec and Ghent University Ghent, Belgium

James Foresi<sup>1</sup> <sup>]</sup>Jim.Foresi@imec-int.com

## Abstract

Reliable detection of pedestrians and road users is essential for the safe operation of autonomous vehicles. However, this task becomes increasingly challenging under adverse environmental conditions, such as poor weather and low lighting—where the performance of state-of-the-art pre-trained object detectors and on-board sensors tends to degrade, which limit their effectiveness and reliability. Moreover, retraining models for each operating context imposes significant constraints due to high data requirements, computational resources, and time. Considering this, we propose a novel context-aware multimodal late fusion approach that integrates a lightweight gating network within our probabilistic calibration framework. The gating network extracts environmental context and dynamically modulates the detection outputs of pre-trained object detectors based on the inferred context, without retraining of the detectors. Our methodology is evaluated on a modified KITTI dataset, in which the camera images and lidar pointclouds are augmented for varying fog levels. Experimental results show that our approach yields a 14% relative improvement over non-adaptive late fusion baselines, achieving a consistent average precision of 56% across different fog levels.

## 1 Introduction

In autonomous systems and intelligent sensing applications, robust object detection under adverse environmental conditions remains a critical challenge. Multimodal sensor fusion has emerged as a promising approach to enhance detection reliability by leveraging complementary information from heterogeneous sensor modalities. Sensor fusion can be achieved using early, mid and late fusion approaches, and according to recent studies late fusion is more accurate and cost-effective [11] [25]. However, sensor performance can be highly dependent on environmental context, and sensor fusion approaches do not account for it in their pipelines. Some recent studies [19] [23] incorporate environmental context, but these deep neural network based methods require retraining to handle environmental conditions not present in the training data.

Therefore, we propose a novel multimodal fusion framework that integrates pretrained object detectors operating on RGB cameras, and lidar, while dynamically adapting to environmental context.

![](images/385f45f12ff0ea35cc7878437ce34e231c92e0cf303edd212ea8f01f1679d796.jpg)  
Figure 1: Overall system diagram for the adaptive multimodal sensor fusion.

Our approach begins by independently applying pretrained object detectors to each sensor modality as shown in Figure 1. These detectors provide confidence scores for each detected object, which are then calibrated using a probabilistic transformation or calibration layer that converts raw confidence values into log-likelihood ratios. This calibration compensates for biases inherent in pretrained models and unifies the outputs across modalities. To further refine the fusion process, we introduce a neural-network based adaptive layer. This layer ingests both the LLRs, and contextual information, such as lighting conditions, weather, and visibility, which are extracted from the RGB cameras to dynamically adjust the contribution from each sensor. By learning to emphasize or de-emphasize sensor inputs based on situational awareness, the system achieves greater robustness and flexibility. Finally, a fusion layer aggregates the adaptively weighted outputs from all sensors to produce a unified detection result. This architecture enables context-aware, confidence-calibrated sensor fusion, improving object detection accuracy and reliability across diverse operational scenarios, specifically for adverse weather conditions.

This paper makes the following key contributions:

1. We propose a context extraction network that captures fine-grained local context to better characterize varying sensing conditions.

2. We develop a mechanism to quantify how sensing modalities degrade with respect to this extracted context (e.g., lidar in fog).

3. We introduce a method to dynamically calculate how to compensate for this in probabilistic fusion, enabling more robust cross-modality performance.

## 2 Related Studies

We classify prior research in multimodal object detection on the basis of operating environment (clear vs adverse weather) and underlying estimation methodology (deterministic vs probabilistic). Specifically, we focus on (i) standard fusion architectures for clear conditions,

(ii) fusion architectures for adverse-weather, and (iii) fusion architectures with probabilistic methods for uncertainty estimation. Our work lies at the intersection of these threads, which we review below to position our contribution against the open gaps in the literature.

## 2.1 Standard Multimodal Sensor Fusion

Multimodal fusion has emerged as the dominant paradigm for safety-critical autonomous systems, leveraging the complementary strengths of cameras, lidars, and radars, with additional modalities such as gated NIR sensors being increasingly introduced [11] [25]. Early and mid-fusion methods such as MV3D [3], AVOD [13], F-PointNet [28], and transformerbased detectors (BEVFusion [16], CMT [29], FusionFormer [10]) achieve state-of-the-art accuracy on clear-weather benchmarks like KITTI and NuScenes by deeply coupling crossmodal features. However, this tight coupling makes them brittle to asymmetric sensor degradation. For instance, if one modality is corrupted, the joint feature space is compromised, often requiring the entire network to be retrained on data capturing the corrupted operating distribution. Conversely, late fusion approaches in [4] [22] [30] decouple the modalities at the decision level, making them modular, computationally light, and natively compatible with off-the-shelf pretrained detectors — though it comes at the cost of discarding valuable low-level cross-modal cues.

## 2.2 Probabilistic and Bayesian Late Fusion

Closest to our framework are probabilistic late-fusion methods that combine outputs of independent detectors under Bayesian hypothesis. [5] formulates a non-learned Bayesian fusion rule derived from a conditional-independence assumption across modalities, reporting a performance gain of over 13% in RGB-thermal pedestrian detection. [22] establishes a naive Bayes decision-level fusion framework for color and thermal detectors that works without retraining with graceful handling of sensor failure. [26] [27] develop a likelihood-ratio calibration layer that converts raw detector confidence scores into log-likelihood ratios (LLRs), which can be combined across sensors via Bayes’ rule. Our proposed framework builds directly upon this approach.

## 2.3 Adverse-Weather and Adaptive Fusion

Recent research has shifted focus toward multimodal perception in degraded environments such as fog, rain, snow, and low light conditions. [2] presents the Seeing Through Fog (STF) dataset and proposes an entropy-driven adaptive fusion that learns from clean data and generalizes to unseen weather. Another study, SAMFusion [19] extends this line with a transformer decoder that re-weights RGB, Lidar, gated-NIR, and radar features by distance and visibility, reporting +17.2 AP for distant pedestrians in fog. ContextualFusion [23] injects coarse domain knowledge about modality reliability (e.g., Lidar degrades in fog, camera struggles under low-light) into a 3D detector and demonstrates gains on the simulated CARLA-Adverse and naturalistic NuScenes datasets. FogFusion [31] uses depth-completion with fog-aware convolutions to densify lidar before camera fusion, L4DR [12] fuses Lidar with 4D radar for weather robustness, UAMF [15] introduces Bayesian uncertainty modeling with cross-modal attention conditioned on weather priors, and ACMD [8] performs adaptive cross-modal denoising guided by residual-noise estimation.

## 2.4 Research Gap

Existing adverse-weather and adaptive-fusion methods are powerful but require end-to-end retraining and rely on heuristic, scene-level weighting without probabilistic semantics. While existing Bayesian late-fusion methods preserve interpretability and reuse pretrained detectors, they assume context-invariant likelihoods and fail to adapt when sensing conditions shift. No prior work simultaneously offers: (i) Use of off-the-shelf pretrained single-sensor detectors with no fine-tuning, avoiding the high data and compute cost of retraining for new operating contexts, (ii) A context-dependent compensation factor derived directly from Bayes’ rule rather than using a heuristic gating mechanism, providing the fused score with a well-defined posterior-ratio interpretation, and (iii) A context adaptation mechanism that extracts context from the local image region surrounding each candidate detection rather than a single scene-level weather descriptor (such as a global weather label), allowing the model to handle within-scene variations (e.g., a partially occluded pedestrian, a near vs. far target in patchy fog). To our knowledge, this is the first multimodal pedestrian-detection framework that combines pretrained-detector reuse, first-principles probabilistic compensation, and fine-grained local context extraction.

<table><tr><td>Method Family</td><td>Pretrained Detectors? Context-Aware? Local Context?</td><td></td><td></td></tr><tr><td>Deep Mid-Fusion (e.g., BEVFusion, MV3D)</td><td>No (requires retraining)</td><td>No</td><td>No</td></tr><tr><td>Adverse-Weather Adaptive (e.g., STF, SAMFusion)</td><td>No (requires retraining)</td><td>Yes (scene-level)</td><td>No</td></tr><tr><td>Gating / Mixture-of-Experts (e.g., Mees, GRIF)</td><td>No (requires retraining)</td><td>Implicit</td><td>No</td></tr><tr><td>Bayesian Late Fusion (e.g., Chen, Shaikh)</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Proposed Method (Ours)</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

Table 1: A comparison of existing methods and our approach.

## 3 Method

In this section, we describe the theoretical formulation of the context-adaptive multimodal sensor fusion and implementation of our developed approach.

## 3.1 Bayesian Fusion

Pre-trained object detectors perform differently under different scenarios, e.g., daytime vs nighttime performance, up-close vs long-range detection. To address this variability, the study in [26] introduced a calibration layer for single-sensor object detection by formulating a probabilistic decision framework based on Bayesian inference. This calibration layer enables translation of raw detection scores from pre-trained off-the-shelf object detectors into conditional probabilities of presence/absence of objects. By incorporating scenario information within a small subset of the data, the calibration layer adjusts the detector’s output to better reflect the true likelihood of presence of an object under varying environmental conditions.

## 3.1.1 Theory

Let $H _ { 1 }$ denote the hypothesis that a pedestrian is present in a part of the scene, and $H _ { 0 }$ the complementary hypothesis that no pedestrian is present. A pre-trained object detector produces candidate detections represented by tuples $y _ { l } = \left( u _ { l } , s _ { l } , a _ { l } \right)$ where $u _ { l }$ is the location in sensor coordinates, $s _ { l }$ is the size of an object in sensor coordinates, and $a _ { l }$ is the confidence score in the range [0, 1]. These candidate detections can then be evaluated probabilistically against the hypotheses to assess the level of evidence for pedestrian presence they represent. With prior probability $P ( H )$ representing the initial belief about presence or absence of a road user and marginal likelihood $p ( y _ { l } )$ representing the overall probability of obtaining an observation/detection under all hypotheses, the posterior probability of presence given the observation is expressed in terms of likelihood ratios and prior distributions using Bayes’ rule as

$$
P ( H _ { 1 } | y _ { l } ) = \frac { p ( y _ { l } | H _ { 1 } ) P ( H _ { 1 } ) } { p ( y _ { l } ) }\tag{1}
$$

and the posterior probability of absence with the same expression for $H _ { 0 }$

A detection should be considered positive if the following condition is satisfied:

$$
P ( H _ { 1 } | y _ { l } ) > P ( H _ { 0 } | y _ { l } )\tag{2}
$$

From here onwards, we will focus only the detection confidence score $a _ { l }$ , omitting $u _ { l }$ and $s _ { l }$ since most object detectors are insensitive to location and trained to be insensitive to object size within a reasonable range. The decision boundary for candidate detections then becomes

$$
\frac { p ( a _ { l } | H _ { 1 } ) } { p ( a _ { l } | H _ { 0 } ) } \lesssim \frac { P ( H _ { 0 } ) } { P ( H _ { 1 } ) } .\tag{3}
$$

The functions $p ( a _ { l } | H _ { 1 } )$ and $p ( a _ { l } | H _ { 0 } )$ are called the likelihood functions and their ratio will be referred to as the likelihood ratio. Since the prior probabilities are independent from the input $a _ { l }$ (they can broadly be interpreted as representing a sensitivity level for the decision), all information necessary for the calibration layer is contained in the likelihood ratio.

In this study, we extend this calibration framework to support multimodal fusion. Calibrated decision scores from each modality are combined to produce a unified estimate of the object presence. Assuming conditionally independent detections in the different modalities with confidence scores $a _ { 1 }$ (i.e., lidar), $a _ { 2 }$ (i.e., camera), the joint posterior probabilities are given by the product of the individual likelihood functions with the priors:

$$
{ \frac { P ( H _ { 1 } | a _ { 1 } , a _ { 2 } ) } { P ( H _ { 0 } | a _ { 1 } , a _ { 2 } ) } } = { \frac { P ( H _ { 1 } ) } { P ( H _ { 0 } ) } } \prod _ { i = 1 } ^ { 2 } { \frac { p ( a _ { i } | H _ { 1 } ) } { p ( a _ { i } | H _ { 0 } ) } }\tag{4}
$$

and the optimal decision criterion becomes

$$
\prod _ { i = 1 } ^ { 2 } \frac { p ( a _ { i } | H _ { 1 } ) } { p ( a _ { i } | H _ { 0 } ) } \lessgtr \frac { P ( H _ { 0 } ) } { P ( H _ { 1 } ) } .\tag{5}
$$

## 3.1.2 Implementation

To calculate the likelihood functions, a small number of negative (no person present) and positive (person present) samples of the ground truth data are evaluated by the object detector. For each set, a histogram of the detection scores produced for these samples is made and normalized into a density function. The likelihood ratio for any confidence value produced by the detector is then approximated by the ratio of the density function value of the positive set for this confidence to the density function value of the negative set for this confidence. This likelihood ratio function is constructed for each of the sensor modalities using a publicly available pre-trained object detector for each individual sensor.

In our implementation the likelihood ratios are converted to their logarithm to simplify computation and to aid numerical stability when potentially combining very small with very large values. We will hereafter refer to Log-Likelihood Ratios (LLRs).

## 3.2 Context Extraction and Adaptation

## 3.2.1 Theory

It is observed that the distribution of confidence scores obtained can change significantly with respect to the sensing conditions, i.e., the scores depend on the observation context. The posterior probabilities in our decision also depend on these conditions and the derivation should be modified to model the conditional dependence of the likelihood functions on the observation context. In principle, the context dependent likelihood ratios of every sensor can be obtained by splitting a sufficiently varied dataset into different context categories but in practice this strongly reduces the number of samples in every context category, leading to reduced representativeness of the histograms, and additionally requires one to decide the split into relevant contexts up front. Instead, we propose a compensation strategy that applies a modifier to the generic likelihood ratios of each sensor. Let the local, immediate context be described by a variable c. The context-dependent decision criterion is then given by

$$
\frac { P ( H _ { 1 } | a _ { 1 } , a _ { 2 } , c ) } { P ( H _ { 0 } | a _ { 1 } , a _ { 2 } , c ) } = \frac { p ( a _ { 1 } , a _ { 2 } , c | H _ { 1 } ) P ( H _ { 1 } ) } { p ( a _ { 1 } , a _ { 2 } , c | H _ { 0 } ) P ( H _ { 0 } ) } \lesssim 1\tag{6}
$$

The conditional independence assumption becomes stronger once context is introduced, as it is assumed to fully capture the sensors dependency. Hence assuming conditionally independent sensors, the likelihood ratio can be decomposed as

$$
\frac { p ( a _ { 1 } , a _ { 2 } , c | H _ { 1 } ) } { p ( a _ { 1 } , a _ { 2 } , c | H _ { 0 } ) } = \frac { p ( a _ { 1 } , c | H _ { 1 } ) } { p ( a _ { 1 } , c | H _ { 0 } ) } \frac { p ( a _ { 2 } , c | H _ { 1 } ) } { p ( a _ { 2 } , c | H _ { 0 } ) } \frac { p ( c ) } { p ( c ) } .\tag{7}
$$

which in turn can be rewritten as

$$
\left[ \frac { p ( c | a _ { 1 } , H _ { 1 } ) } { p ( c | a _ { 1 } , H _ { 0 } ) } \frac { p ( a _ { 1 } | H _ { 1 } ) } { p ( a _ { 1 } | H _ { 0 } ) } \right] \left[ \frac { p ( c | a _ { 2 } , H _ { 1 } ) } { p ( c | a _ { 2 } , H _ { 0 } ) } \frac { p ( a _ { 2 } | H _ { 1 } ) } { p ( a _ { 2 } | H _ { 0 } ) } \right] .\tag{8}
$$

Comparing to equation 5, we see that every sensor’s likelihood ratio needs to be multiplied with a correction factor $p ( c | a , H _ { 1 } ) / p ( c | a , H _ { 0 } )$ . The principle of applying a multiplicative weight to each sensor’s assessment is similar in structure to a mixture-of-experts ensemble model, where various expert models (in this case the sensors’ detection algorithms) can have different levels of expertise for different types of input samples. Note that it is impractical to estimate the correction factors for all relevant contexts directly from labeled data because this suffers from the same histogram sparseness that prevents direct measurement of context-adaptive likelihood functions. Instead, we propose a two-tiered neural network structure to predict the factors (as shown in Fig 2):

1. a context extraction network that summarizes the local, immediate context pertaining to a detection in a continuous variable c,

2. a compensation network that predicts for any c the necessary correction factor for the not-context-specific likelihood function.

This structure can approximate context-dependent likelihood ratios without explicit distribution modeling or model retraining, and adjusts the relative importance of each sensor in the Bayesian fusion framework.

Note that the relationship between the weight factor and the context variable c can be highly non-linear and additionally depends on the likelihood ratio itself. The function of the weight can be seen as bringing the likelihood ratio closer to 1 (i.e equal probability of presence and absence) for contexts that reduce the reliability of the detector, and the required weight is thus different depending on whether the likelihood ratio is larger or smaller than one for the sample considered.

![](images/2db59e19f2089c7c5419e59fc1fd1e7b2ecb9a37ac2aa4032b54578b90b1f787.jpg)  
Figure 2: Overview of the context-aware probabilistic compensation network.

## 3.2.2 Implementation

Detection candidates across the sensor modalities are first matched and associated to produce candidate fused pairs. The region in the RGB image corresponding to each candidate pair is extracted from the RGB image and passed through the context extraction network which outputs a context value $c _ { l }$ that characterizes the current scene context associated with each detection. This context extraction network is based on a ResNet18 architecture with the final layer modified to produce a single scalar. Note that since the context is extracted from the RGB crop, it may include object appearance, distance, occlusion, or detector artifacts — not only environmental context. Secondly, a custom feedforward network takes in the context value produced by the context extraction network along with the sensor-specific LLR values computed under clear weather conditions. The input vector is defined as $[ c _ { l } , L L R _ { m } ]$ where m corresponds to the sensor modality (e.g. RGB, or lidar). To accommodate multiple sensor modalities, the architecture employs n independent feedforward networks—one per modality—each acting as a separate adaptation head. These heads produce modality-specific weights $[ r _ { 1 } , r _ { 2 } ]$ that recalibrate the corresponding LLRs to align with the current environmental context. Note that implemented in the log-domain, the adaptation factors become bias terms. The complete architecture is shown in Figure 3.

![](images/d146cbd5a6978118503e5cdcbb9dfc9ca7c2ebeeeed3314f3497e3ef6909d4de.jpg)  
Figure 3: Architecture of the implemented context-aware compensation network.

The context extraction network and the FFN compensation network are jointly trained, ensuring that the context variable $c _ { l }$ captures only context information that is necessary to compensate the LLRs, without containing redundant information that is already assessed by the individual sensors’ pre-trained detectors and thus represented in the original LLRs. To ensure that the compensation terms together with the original LLRs converge to fused LLRs, we use the loss function proposed in [18]. Note that in principle, the compensation can also be implemented as a single bias term r, but keeping separate terms for each sensor helps interpretability and aids convergence as the FFNs can be trained individually.

## 4 Experimental Setup

We evaluate the performance of the context-adaptive fusion framework for pedestrian detection in real-world scenarios using RGB image, and lidar pointcloud as input modalities. A key underlying assumption of this framework is that different modalities vary in how informative they are across contexts. By accordingly changing the relative impact each sensor has on the final decision, our method aims to reach better precision/recall tradeoffs across all conditions.

Initial detection candidates and confidence scores are obtained using widely adopted pre-trained single sensor object detectors: YOLOv8m model [20] pre-trained on the COCO dataset for camera; and KITTI pre-trained PointPillars [14] for lidar.

## 4.1 Dataset

We use the 3D object detection dataset from KITTI Vision Benchmark Suite [7] as it provides time-synchronized data from a RGB camera (Point Grey Flea 2) and a lidar (Velodyne HDL-64E ), accompanied by robust ground truth annotations and calibration matrices for each frame. The dataset consists of 7841 training images and 7518 test images.

We subdivide KITTI training samples into three splits: training (60%) with 4488 images, validation (20%) with 1496 images, and testing (20%) with 1497 images, as ground truth labels are required to evaluate performance.

## 4.2 Fog Augmentation

Due to the challenging task of reliably labeling real-world fog datasets for 3D object detection, synthetic fog was generated by augmenting the original KITTI images and point clouds as shown in Figure 4. To characterize the severity of the fog, we use the commonly used metric visibility level [17], which quantifies the density of fog based on how far objects can be seen. Real-world datasets typically lack explicit visibility annotations, making them unsuitable for controlled evaluation. Using visibility levels allows for a systematic analysis of model performance under varying fog conditions and helps identify the point at which perception begins to degrade. To simulate varying visibility conditions, we augment the RGB images and lidar point clouds with three levels of synthetic fog, corresponding to three visibility distances of 30m, 50m, and 75m.

To perform fog augmentation in point clouds, we employ the method by Hahner et al. [9]. For the image augmentation, a custom approach was tailored based on [6], [21], [1]. The method models fog as a population of droplets whose radii are sampled from a lognormal distribution, capturing the natural variability in particle sizes. For each droplet radius, wavelength-dependent extinction efficiencies are computed using Mie theory, which accounts for both scattering and absorption by spherical particles. These per-droplet extinction values are then multiplied by the droplet concentration to obtain the overall extinction coefficient of the fog medium. The depth maps corresponding to KITTI images were taken from [24]. This depth data is used to scale light attenuation based on the distance light travels through the fog. Using the computed extinction coefficients and pixel-wise distances, Beer–Lambert’s law is applied to determine the final pixel intensities. As a result, nearby objects appear less obscured, while distant objects are increasingly attenuated, producing realistic, depth-dependent fog effects.

![](images/514d0dba94bad174ad53fb9e820b39463218fe23db526c5876e975e248515308.jpg)

![](images/cf9af7233aa654f1039649f7f35b1ff533e594bb33368e1902433d5072415710.jpg)

![](images/704bacd7071ef2195b3d1033380651606858e366a02dff5a170eaaab05ad583f.jpg)

![](images/062dbdbf25d6a273bd1a9fba2261cd31302e6fa37adc1e6338731d2de22fa236.jpg)

![](images/ef86f3e6ff5e091a8740d4ef11e1d4f30d74e8587855f83c08b5c415d6786fc4.jpg)

![](images/e0f7694999ea67fc5af40f3415464e27021b293d28abb1e5d3cbd8fe1cfd404d.jpg)  
Figure 4: Sample images and pointclouds under varying visibility conditions. Top-left to bottom-right: clear, 75m, 50m, 30m.

## 4.3 Log-Likelihood Ratio construction

As described in section 3.1.2, inference is run on the KITTI clear weather training subset, comprising 4488 samples, using state-of-the-art pre-trained object detectors run at very high sensitivity. For each sensor, detection scores obtained after non-max suppression in empty areas (i.e., parts of the training images that do not contain ground truth objects) count towards the $H _ { 0 }$ histogram of that sensor, while scores obtained for detections that have an intersection-over-union (IoU) greater than 0.5 with a ground truth bounding box count towards the $H _ { 1 }$ histogram. Sample LLR construction for the camera detector is shown in Figure 5.

The histograms are converted into estimated probability density functions $p ( a | H )$ using Kernel Density Estimation (KDE), with Scott’s rule determining the (locally adaptive) bandwidth of the kernel used to smooth the histogram. The logarithm of $p ( a | H _ { 1 } ) / p ( a | H _ { 0 } )$ is then taken to obtain a single-sensor LLR for any value of confidence score a.

![](images/e823cfbf4e27f7f264c779a042c68116f6a9a1cd7104da28f42c183d20f81ca7.jpg)  
Figure 5: LLR construction for YOLO

## 4.4 Context Network Training

The context extraction and adaptation networks are jointly trained on a mixture of clear and foggy data as discussed in section 3.2.2. The training set comprises four variations of the original 4488 clear weather training samples—covering clear conditions and synthetic fog at visibility levels of 30m, 50m, and 75m for both images and point clouds, resulting in a total of 17952 samples. As mentioned earlier, model training is performed at the level of fused sensor detections, not individual frames. For each of the 4488 frames, detection outputs from YOLOv8 and PointPillars are associated to form fused detections. The resulting set includes independent class predictions, bounding boxes, confidence scores, and LLRs from each detector, along with fused bounding boxes, IoU, and is categorized into true positives and false positives by comparing to KITTI ground truth labels. The networks are trained with samples drawn from both true positive and false positive categories. Thus, the effective training set consists of $1 7 9 5 2 ^ { \ast } \mathrm { n }$ detection samples, where n is the average number of fused detections per frame.

The context network is trained using an end-to-end joint optimization approach. Based on the TP/FP distinction on the fused dataset, separate camera and lidar losses are computed using the loss function from [18] and summed to form a total loss. During the backward pass, gradients from the camera loss flow exclusively through the camera FFN, while gradients from the lidar loss flow exclusively through the lidar FFN. Both gradient pathways then converge at the shared ResNet-based context encoder, updating the weights across all components in a single, unified optimization step. For each training epoch, the model is validated using the validation subset consisting of clear and foggy dataset. The model is trained for 50 epochs using Stochastic Gradient Descent (SGD) as the optimizer, with a learning rate of $1 \times 1 0 ^ { - 6 }$ . The trained context network is then integrated into the overall fusion architecture, and the complete model is evaluated using the test set. Each of the 1497 samples in the test set is presented under four variations of clear and foggy conditions, similar to the training set, resulting in a total of 5988 test instances.

## 5 Results

This research focuses on robust pedestrian detection under foggy environmental conditions using multimodal sensor data. We evaluate pedestrian detection performance across three configurations:

C1 Individual object detectors operating independently.

C2 Fusion of independent object detectors using non-adaptive Bayesian fusion.

C3 Fusion of independent object detectors using context-adaptive Bayesian fusion.

This comparison highlights the impact of sensor fusion strategies from baseline performance of standalone detectors to improvements with non-adaptive fusion, and further gains achieved through context-aware adaptive fusion. Precision–Recall (PR) curves are used to quantify and visualize the comparative performance of the three candidate configurations. Area under the precision-recall curve is denoted as Average Precision (AP) and used as an indicator of general performance.

The study in [27] compares the pedestrian detection performance of state-of-the-art image based object detectors with and without the calibration layer. In this study, we focus on multimodal sensor fusion. First, we compare the performance of pedestrian detection between the individual detectors (i.e., C1) and non-adaptive Bayesian late fusion (i.e., C2)

![](images/b1533a042b5cb41746c9a658e662efdbfd169d4afad2b814ce3d14b0bfef1985.jpg)  
Figure 6: Performance of pre-trained individual detectors and non-adaptive sensor fusion under clear weather conditions

![](images/f0597cfd8d825555bc5716f0d7fcf8e4c81552010efb23768424394b2916ef43.jpg)  
(a) PointPillars

![](images/6c2ab1b7a62ce12c53c15b42b739fe7461a274392ff2264a3046066a1f2ef0d3.jpg)  
(b) YOLO  
Figure 7: Performance of pre-trained individual detectors on clear and foggy data in terms of AP

pipeline under clear weather conditions (Figure 6). This fusion approach realizes significantly higher average precision than any single detector (fused AP of 0.59 compared to 0.45 for the best single-sensor detector), validating Bayesian fusion as an effective method to exploit complementarity in heterogeneous perception systems in non-challenging conditions.

Figure 7 shows the impact of (simulated) fog on the detection performance of PointPillars and YOLO (pre-trained only on clear-weather data). Both experience significant degradation on foggy data, with PointPillars most affected due to reduced point cloud density and reflectivity. Consequently, non-adaptive fusion also suffers from these degradations of the RGB camera and lidar modalities (Figure 8).

In comparison, context-adaptive fusion with our proposed network (i.e., C3) maintains consistent performance. While non-adaptive fusion still improves performance compared to individual detectors across different fog levels, context-adaptive fusion demonstrates a significantly larger advantage. Our context-adaptive approach achieves an average precision (AP) of ∼ 0.56 across all fog densities, and offers a ∼ 14% relative improvement in average precision over non-adaptive fusion, showing the importance of context for sensor fusion (Table 2). The context-adaptive model was trained on both clear and multi-level fog conditions, enabling it to perform similar to non-adaptive fusion in clear weather and outperform it under foggy conditions. This highlights that a single context-aware model can effectively handle both clear and foggy scenarios.

Another key aspect is the role of the context extraction network (i.e., ResNet-18) in influencing modality-specific weights. Figure 9 illustrates how the context value varies with environmental conditions. For example, in the case of a cyclist, the context value increases as the detection task becomes more challenging. As fog density rises—from clear conditions to a lower visibility level of 30m (i.e., dense fog)—the context value grows accordingly, reflecting the network’s ability to capture scene difficulty.

![](images/9a1f9b73ea3471b7ad8d4aee2135c41168b352e9980b8d8b4ecc924c0481902c.jpg)  
(a) Clear

![](images/c99ec119d7aae1b415238655bab4624b3dd30bfcb7fcccd5fdc707a6b192dd3e.jpg)  
(b) 75m Fog

![](images/7f15d6728caee4bd19d1b635657560623ee792ae018654f692e26106039ddd45.jpg)  
(c) 50m Fog

![](images/0ccc8c13660e10c283c62b77d0dfd96e3972f56c8343387c273660f83674b734.jpg)  
(d) 30m Fog

![](images/56c3e7efa0dbc6d4003da28352c1387f648660488e4aa7b55fb3f9beef082f76.jpg)  
(e) Combined  
Figure 8: PR curves comparing the performance of individual detectors, non-adaptive fusion, and context-aware fusion across (a) clear (b) 75m fog (c) 50m fog (d) 30m fog (e) Combined (all fog levels and clear)

<table><tr><td></td><td>Context-Adaptive Fusion (AP)</td><td>Non-adaptive Fusion (AP)</td><td>Camera (AP)</td><td>Lidar (AP)</td></tr><tr><td>Clear</td><td>0.61</td><td>0.59</td><td>0.43</td><td>0.45</td></tr><tr><td>75m fog</td><td>0.57</td><td>0.52</td><td>0.41</td><td>0.33</td></tr><tr><td>50m fog</td><td>0.55</td><td>0.48</td><td>0.39</td><td>0.29</td></tr><tr><td>30m fog</td><td>0.50</td><td>0.43</td><td>0.37</td><td>0.20</td></tr><tr><td>Combined</td><td>0.56</td><td>0.49</td><td>0.40</td><td>0.31</td></tr></table>

Table 2: Comparison of Average Precision (AP) across different visibility conditions

![](images/818ac57ce7bdcd52b32b606591796920bcff663ce1493288e3a4500b42473603.jpg)  
Figure 9: Variation of context value under different conditions for a cyclist.

## 6 Conclusion

This research addresses the challenge of robust pedestrian detection under foggy conditions using multimodal sensor data. Our findings demonstrate that conventional non-adaptive sensor fusion fails to deliver significant improvements when individual detectors degrade under foggy conditions. To overcome this limitation, we introduced a context-adaptive Bayesian late fusion framework that extracts environmental context and dynamically adjusts modalityspecific weights to comply with diverse conditions. The model effectively generalizes across multiple visibility conditions with a limited dataset, learning features for varying visibility without requiring manual annotations of weather or visibility. By decoupling context modeling from the detection pipeline, our approach enables adaptation to different contexts and scenarios while avoiding the significant cost and difficulty of re-training detectors for different contexts. Experimental results confirm that context-adaptive fusion outperforms both individual detectors and non-adaptive fusion, providing an elegant solution to the challenge of strong variation of conditions in which real-world autonomous perception systems must operate. Future work will cover deeper analysis of the extracted context as well as testing real-world adverse weather scenarios.

## References

[1] A. Bernuth, G. Volk, and O. Bringmann. Simulating photo-realistic snow and fog on existing images for enhanced cnn training and evaluation. In IEEE Intelligent Transportation Systems Conference (ITSC), 2019.

[2] M. Bijelic, T. Gruber, F. Mannan, F. Kraus, W. Ritter, K. Dietmayer, and F. Heide. Seeing through fog without seeing fog: Deep multimodal sensor fusion in unseen adverse weather. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[3] X. Chen, H. Ma, J. Wan, B. Li, and T. Xia. Multi-view 3d object detection network for autonomous driving. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

[4] Y. Chen, J. Shi, C. Mertz, S. Kong, and D. Ramanan. Multimodal object detection via bayesian fusion. arXiv preprint arXiv:2104.02904, 3(6), 2021.

[5] Y. Chen, J. Shi, Z. Ye, C. Mertz, D. Ramanan, and S. Kong. Multimodal object detection via probabilistic ensembling. In European Conference on Computer Vision, pages 139–158, 2022.

[6] P. Duthon, M. Colomb, and F. Bernardin. Light transmission in fog: The influence of wavelength on the extinction coefficient. Applied Sciences, 2019.

[7] A. Geiger, P. Lenz, and R. Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.

[8] M. A. Ghaffar, K. Zhang, N. Pan, and L. Peng. Adaptive cross-modal denoising: Enhancing lidar-camera fusion perception in adverse circumstances. IEEE Sensors, 26 (2):408, 2026.

[9] M. Hahner, C. Sakaridis, D. Dai, and L. Van Gool. Fog simulation on real lidar point clouds for 3d object detection in adverse weather. In IEEE International Conference on Computer Vision (ICCV), 2021.

[10] C. Hu, H. Zheng, K. Li, J. Xu, W. Mao, M. Luo, L. Wang, M. Chen, Q. Peng, K. Liu, Y. Zhao, P. Hao, M. Liu, and K. Yu. Fusionformer: A multi-sensory fusion in bird’seye-view and temporal consistent transformer for 3d object detection. arXiv preprint arXiv:2309.05257, 2023.

[11] K. Huang, B. Shi, X. Li, X. Li, S. Huang, and Y. Li. Multimodal sensor fusion for auto driving perception: A survey. arXiv preprint arXiv:2202.02703, 2022.

[12] X. Huang, Z. Xu, H. Wu, J. Wang, Q. Xia, Y. Xia, J. Li, K. Gao, C. Wen, and C. Wang. L4dr: Lidar-4dradar fusion for weather-robust 3d object detection. In AAAI conference on artificial intelligence, volume 39, pages 3806–3814, 2025.

[13] J. Ku, M. Mozifian, J. Lee, A. Harakeh, and S. L. Waslander. Joint 3d proposal generation and object detection from view aggregation. In IEEE/RSJ international conference on intelligent robots and systems (IROS), pages 1–8, 2018.

[14] A.H. Lang, S. Vora, H. Caesae, L. Zhou, J. Yang, and O. Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[15] Z. Li and B. S. B. Singh. Autonomous driving in adverse weather: A multi-modal fusion framework with uncertainty-aware learning for robust obstacle detection. International Journal ofAdvanced Computer Science and Applications, 16(8), 2025.

[16] T. Liang, H. Xie, K. Yu, Z. Xia, Z. Lin, Y. Wang, T. Tang, B. Wang, and Z. Tang. Bevfusion: A simple and robust lidar-camera fusion framework. In Advances in Neural information processing systems, volume 35, pages 10421–10434, 2022.

[17] Organisation Météorologique Mondiale. Guide to instruments and methods of observation (wmo-no. 8). WMO, Geneva, 2018.

[18] G.V. Moustakides and K. Basioti. Training neural networks for likelihood/density ratio estimation. arXiv preprint arXiv:1911.00405, 2019.

[19] E. Palladin, R. Dietze, P. Narayanan, M. Bijelic, and F. Heide. SAMFusion:sensoradaptive multimodal fusion for 3d object detection in adverse weather. In European Conference on Computer Vision (ECCV), 2024.

[20] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, realtime object detection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[21] C. Sakaridis, D. Dai, and L. Van Gool. Semantic foggy scene understanding with synthetic data. International Journal ofComputer Vision, 2018.

[22] Z. A. Shaikh, D. Van Hamme, P. Veelaert, and W. Philips. Probabilistic fusion for pedestrian detection from thermal and colour images. IEEE Sensors, 2022.

[23] S. Sural, N. Sahu, and R.R. Rajkumar. ContextualFusion: Context-based multi-sensor fusion for 3D object detection in adverse operating conditions. In IEEE Intelligent Vehicles Symposium (IV), 2024.

[24] M. Tremblay, S.S. Halder, R. De Charette, and J.-F. Lalonde. Rain rendering for evaluating and improving robustness to bad weather. International Journal of Computer Vision, 2021.

[25] X. Wang, K. Li, and A. Chehri. Multi-sensor fusion technology for 3D object detection in autonomous driving: A review. IEEE Transactions on Intelligent Transportation Systems, 25(2):1148–1165, 2024.

[26] T. Willems, J. Aelterman, and D. Van Hamme. Bayesian optimisation of existing object detection methods for new contexts. IEEE Sensors, 2023.

[27] T. Willems, J. Aelterman, and D. Van Hamme. Context-compensated probabilistic pedestrian detection. In IEEE Intelligent Vehicles Symposium (IV), 2024.

[28] C. Wu, J. Pfrommer, J. Beyerer, K. Li, and B. Neubert. Object detection in 3d point clouds via local correlation-aware point embedding. In Joint 9th International Conference on Informatics, Electronics & Vision (ICIEV) and 4th International Conference on Imaging, Vision & Pattern Recognition (icIVPR), pages 1–8, 2020.

[29] J. Yan, Y. Liu, J. Sun, F. Jia, S. Li, T. Wang, and X. Zhang. Cross modal transformer: Towards fast and robust 3d object detection. In IEEE/CVF international conference on computer vision, pages 18268–18278, 2023.

[30] Q. Yang, Y. Zhao, and H. Cheng. Mmlf: multi-modal multi-class late fusion for object detection with uncertainty estimation. In CAA International Conference on Vehicular Control and Intelligence (CVCI), pages 1–6, 2025.

[31] B. Zhang, Y. Wang, C. Zhang, J. Jiang, X. Luo, X. Wang, Y. Zhang, Z. Liu, G. Shen, Y. Ye, and P. Jiang. Fogfusion: Robust 3d object detection based on camera-lidar fusion for autonomous driving in foggy weather conditions. Proceedings of the Institution of Mechanical Engineers, Part D: Journal of Automobile Engineering, 240(4):2312– 2322, 2026.