# Evaluating Depth-Augmented Visual Features for Cross-Dataset Lunar Traversability Estimation\*

Jan Baturo<sup>1,2</sup> and Bartosz Ptak<sup>2</sup>

Abstract— Autonomous mobility is essential for sustained lunar surface operations, as communication delays and lineof-sight occlusions limit continuous teleoperation. Although recent traversability estimation methods increasingly rely on deep learning, the domain gap between terrestrial analogue environments and real lunar terrain can limit their generalisation, and real lunar traversability labels are scarce, making fully supervised training difficult. To address these challenges, we build a weakly supervised framework trained on sparse pairwise annotations and study whether augmenting visual features with depth-derived geometric information: depth, surface normals, and combined geometric descriptors. We evaluate the models in both in-domain and cross-domain settings using two terrestrial analogue datasets and the real lunar Chang’E PCAM dataset. Contrary to expectations, we find that naive geometric augmentation yields no consistent improvement over a strong visual-only baseline in either setting, with effects small relative to seed-to-seed variance. Instead, the choice of training source emerges as the dominant factor in transfer quality: models trained on domain-relevant sources transfer substantially better to real lunar imagery than the geometric fusion strategy improves any single model. Our evaluation runs onboard a resource-constrained edge device and provides a controlled baseline for designing more expressive multimodal fusion architectures in future lunar perception systems.

## I. INTRODUCTION

Autonomous mobility is required for long-term lunar surface operations. Communication latency and frequent lineof-sight occlusions limit continuous teleoperation, so a rover must judge for itself whether the terrain ahead is safe to cross. Regardless of the rover’s mode of operation – whether it operates autonomously or is controlled remotely – terrain traversability estimation can increase movement speed and reduce hazardous situations. Recent traversability methods increasingly build on learned visual features, and self-supervised foundation models such as DINOv2 [1] produce dense descriptors that transfer across domains far better than encoders trained from scratch [2]. Lunar imagery, however, is strongly out-of-distribution: illumination is harsh and unfiltered, and surface albedo and regolith texture differ from terrestrial simulants and between analogue facilities. Even foundation features, therefore, retain a residual domain gap, and how well they transfer to traversability across lunar datasets is seldom measured [3]. Although the traversability investigated in this work has no physical reference, it provides semantic understanding of unstructured terrain, increasing the rover’s ability to distinguish safe from hazardous regions based on visual appearance alone.

Recent weakly supervised approaches estimate traversability by extracting dense per-pixel descriptors through FeatUp upsampling of DINO features and ranking terrain traversability using a lightweight MLP that operates on visual imagery alone. We hypothesise that augmenting these visual representations with explicit geometric information – namely depth, surface normals, and handcrafted geometric features – can further reduce the remaining cross-domain gap. Unlike photometric cues, which vary with illumination conditions and sensor characteristics, the physical determinants of traversability, such as slope, surface roughness, and the geometry of rocks and crater rims, remain largely invariant across domains (Figure 1). Building on this paradigm, we extend visual feature-based traversability estimation by integrating upsampled DINO descriptors with geometric channels. We train the resulting model within the weakly supervised W-RIZZ approach [4] and evaluate it using the Human Disagreement Rate (HDR).

Our specific contributions are:

• an adaptation of CHUNGUS (Continual Human Understanding for Navigational Guidance in Unstructured settings) [2] with geometry-augmented inputs – depth, surface normals, and handcrafted terrain descriptors;

Visual  
Surface Normals  
Traversability  
![](images/2205e659e03e83e5d4b7bb7a16ed2e0df8952dfc23a2ad7fdf92318ed7a74b37.jpg)  
Fig. 1: The same real lunar scene (Chang’E PCAM) shown as visual, surface normals, and predicted traversability. Photometric appearance shifts markedly across domains, while the geometry governing traversability is largely preserved, motivating depth-derived inputs for cross-dataset transfer.

• a cross-dataset evaluation protocol spanning two lunar analogue facilities (LunaPolaris [5] and our in-house OTRK K ˛akolewo) and real lunar imagery from the Chang’E-3/4 panoramic cameras [6], [7], reporting all train–test domain pairs;

• feasibility of the method working on an edge device.

## II. RELATED WORKS

The literature generally categorises traversability meth ods into 3 main scopes: geometry-based, vision-based and fusion-based. Geometry-based approaches mostly rely on using exteroceptive sensors such as LIDAR and depth cameras. Firstly applied techniques were computationally lightweight and straightforward to implement, however, they often lacked robustness and struggled to adapt to unstructured, rough terrains. Advancements in machine learning and deep neural networks encouraged researchers to estimate traversability from point clouds by applying machine learning techniques [8], [9], [10]. Nevertheless, relying exclusively on geospatial data presents significant disadvantages as it ignores semantic properties of the terrain – such as distinguishing between solid bedrock and highly compressible sand. Additionally, the high cost and computational demands of LiDAR-based algorithms led to the development of vision-based methods. These approaches used various architectures, e.g., AlexNet or PSPNet, and applied different training methods, such as contrastive, self-supervised, or supervised learning [11], [12], [13]. As vision-based methods relied solely on visual appearances, fusion-based methods were subsequently developed to incorporate multiple sensory modalities, creating more robust traversability representations. One of the methods developed using multimodal fusion and tested for space environments is Terrain Traversability Analysis using Geometry and Environmental Perception (TTA-GEP) for planetary rover path planning. The TTA-GEP system architecture operates on highly parallelised dual-stream processing, merging metric data with general object knowledge into one cohesive whole. The geometric stream uses stereovision to generate point clouds, which are processed into a 2.5D map that can be used to extract traditional geometric features such as slope or terrain roughness. Simultaneously, a powerful terrain classifier based on deep learning architecture processes images in order to semantically segment the environment. The goal is to accu rately classify every segment of the Martian or lunar landscape. Semantic labels applied in TTA-GEP research divide the environment into categories highly specific to planetary robotics: bearing soil, exposed bedrock, fine sand, loose rocks, crater edges and ridges, hazardous depressions, and sky background. The last part is the continuous evaluation function, which mathematically fuses and weighs on-the-fly the cost derived from local terrain geometry with the discrete material label derived from semantic segmentation [14].

## III. METHOD

Our traversability algorithm is based on the approach presented in the article “Do You Know the Way? Human-inthe-Loop Understanding for Fast Traversability Estimation in Mobile Robotics”[2]. The original method (called CHUN-GUS) combines a self-supervised visual backbone with a lightweight prediction head to estimate terrain traversability from sparse human annotations [4]. It uses a pre-trained DINO model to extract general visual features from RGB images. FeatUp upsamples these embeddings, then passes them to a lightweight multilayer perceptron (MLP) network that predicts terrain traversability. In our method, we concatenate geometric features directly to the DINOv2 embedding vector as raw values, yielding a (384 + d)-dimensional input to the MLP head. We choose this naive fusion strategy to preserve the frozen backbone’s feature space without retraining and to evaluate the contribution of geometric cues while minimising changes to the network architecture and their influence compared to the visual-only baseline. More sophisticated fusion mechanisms, such as cross-attention or learned gating, are left for future work.

We train the model using sparse pairwise annotations. Instead of assigning absolute values, a human annotator selects two points and specifies their relative traversability (less traversable, more traversable or equally traversable) – see labelling process details in Section IV-A.

The original framework included a Human-In-The-Loop (HITL) component that required additional annotations when the robot encountered unfamiliar terrain during online operation. Since we evaluate geometric features’ influence in offline behaviour, we removed the human-in-the-loop component to preserve a fair comparison. The model trains entirely offline, simplifying deployment on edge devices. Compared to the original approach, we also replaced FeatUp[15] with AnyUp[16] because it better preserves semantic consistency, aligns spatial features more sharply during upsampling, and is more compatible with modern deployment libraries.

## A. Geometric information

To assess the integration of deep visual features and geometric information, three depth-based representations were considered:

• Depth - DINO features are concatenated with a depth channel normalised to the [0,1] range.

• Normals - Added 3 channels (x, y, z) encoding local surface orientation, capturing slope and discontinuities.

• Geometric - In addition to the surface normals, three supplementary terrain descriptors were incorporated:

– Slope magnitude: $\sqrt { ( \partial _ { x } d ) ^ { 2 } + ( \partial _ { y } d ) ^ { 2 } }$ , which quantifies the local steepness. Here, d denotes the local depth value, while $\partial _ { x } d$ and $\partial _ { y } d$ are the partial derivatives (gradients) of the depth in the x and y directions, respectively.

– Local roughness: depth standard deviation in a $( 5 \times$ 5) neighbourhood (denoted as σ).

– Z-Score anomaly: $( z = ( d - \mu ) / \sigma )$ , used to identify local depth outliers, where z is the resulting anomaly score, d is the depth at the point of interest, µ is the mean depth of the surrounding local neighbourhood, and σ is the standard deviation of the depths within that same neighbourhood.

These additional geometric features are concatenated with the upsampled DINO feature embeddings for every pixel before passing them to MLP. This combined representation allows the network to leverage both semantic and geometric features.

## B. Implementation details

Each depth image was processed independently. We filled invalid measurements, defined as non-positive or non-finite values, using Navier–Stokes inpainting with a radius of three pixels. We used the valid measurements to determine the 1st and 99th percentile bounds, then clipped the depth values to this range and min–max normalised them to approximately [0, 1]. A bilateral filter was subsequently applied. Each geometry channel was independently scaled per image to [0, 1]. We applied no dataset-level mean or standard deviation normalisation to the depth or geometry channels. We normalised only the visual input using the ImageNet mean and standard deviation.

For the in-domain experiments, we divided each dataset into training, validation, and test subsets using a 70/15/15 split. For the cross-domain experiments, we divided the source dataset into 70% training and 30% validation subsets, while reserving the full target-domain dataset exclusively for testing. We based early stopping and best-checkpoint selection solely on validation data from the source domain. We used no samples or evaluation results from the crossdomain test set during training, early stopping, or model selection.

To train the prediction head, we used a similar strategy to Schreiber et al.[4], adopting $L _ { R i z z }$ loss and training on sparse annotations. We resize input images to 224×224 and process them with a pre-trained DINOv2 ViT-S/14 backbone to produce 384-dimensional feature embeddings. We used a fixed seed for reproducibility. Optimisation is performed with a learning rate of $( 3 \times 1 0 ^ { - 3 } )$ and zero weight decay. We decay the learning rate by a factor of 0.5 every 75 epochs. Training runs for up to 200 epochs, with early stopping enabled and a patience of 75 epochs based on the validation metric, HDR@0.25.

## IV. EXPERIMENTS

## A. Datasets

For cross-dataset lunar traversability estimation, three datasets are used, covering two lunar analogue test facilities and real lunar imagery from rover missions:

• LunaPolaris [5] was collected in the LunaLab [17] with a small rover. This indoor lunar analogue facility provides controlled regolith-like terrain under repeatable lighting conditions. RGB images were collected using a ZED2 stereo camera. Because the authors provide synchronised left and right images for each timestamp, along with calibration data, we use standard stereo vision techniques to compute depth maps. Figure 2 shows that the data used were taken under both diffuse and harsh lighting conditions.

• In-House dataset refers to our own analogue test facility at OTRK Kakolewo<sup>1</sup>, equipped with diverse rock formations and crater-like features designed to emulate visual characteristics of lunar surface morphology. We collected visual data and depth images using an Orbbec Femto Mega camera mounted on a mobile robot. The camera utilises an indirect Time-of-Flight approach, emitting continuously modulated light and computing depth from the phase shift between emitted and reflected signals, yielding dense depth maps; however, its performance degrades under strong ambient sunlight. We collected data across varied lighting conditions, including low light, full daylight, and low solar angles that produce strong shadows and lens flares. Figure 2 also shows that the data were gathered in both diffuse and harsh lighting conditions, similar to LunaPolaris.

• CE’3/CE’4. We use real lunar surface imagery captured by the panoramic cameras (PCAM) of the Chang’E-3 [6] and Chang’E-4 [7] missions, obtained via the Ground Research and Application System (GRAS) of the China Lunar Exploration Program [18]. The PCAM instruments aboard the Yutu and Yutu-2 rovers acquired ground-level colour images of the lunar surface in Mare Imbrium and the Von Kármán crater on the lunar farside, respectively. Although stereo image pairs are available for the panoramic cameras, they are not temporally aligned. Moreover, calibration data from the rover missions are not publicly available. Consequently, we had to make several assumptions for stereo reconstruction. First, we used the camera parameters reported in the datasheet [19] as an initial estimate. Second, to rectify unsynchronised image pairs, we applied a pointbased alignment approach. Specifically, we detected keypoints using SuperPoint [20] and matched them with LightGlue [21] to estimate a transformation between image pairs, without considering real-time constraints. Third, to address appearance differences, we used the S2M2 method [22], a global matching neural network, to compute disparity. Repeated size estimation experiments on the Chinese national flag visible on the lander yielded a consistent error of approximately 15%, which is acceptable given the absence of precise calibration.

For each dataset, manual annotations were prepared based on sparse pairwise comparisons following the W-RIZZ protocol [4]. For each image, an annotator labels one of the point pairs as more traversable, less traversable, or equally traversable. The process does not define levels of traversability; rather, it ranks them, defining a global relation between image pixels. We select labelling points arbitrarily, focusing on certain relations and uncommon cases. In addition, interimage pairwise comparisons are included, which improve the consistency of predictions across different scenes [2].

TABLE I: Summary of datasets used in this work.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Source</td><td rowspan=1 colspan=1>Depth</td><td rowspan=1 colspan=1>Images</td><td rowspan=1 colspan=1>Labels</td></tr><tr><td rowspan=1 colspan=1>LunaPolaris</td><td rowspan=1 colspan=1>Analogue</td><td rowspan=1 colspan=1>Stereovision</td><td rowspan=1 colspan=1>2404</td><td rowspan=1 colspan=1>3606</td></tr><tr><td rowspan=1 colspan=1>In-House</td><td rowspan=1 colspan=1>Analogue</td><td rowspan=1 colspan=1>Indirect ToF</td><td rowspan=1 colspan=1>2402</td><td rowspan=1 colspan=1>3603</td></tr><tr><td rowspan=1 colspan=1>CE&#x27;3/CE&#x27;4</td><td rowspan=1 colspan=1>Moon</td><td rowspan=1 colspan=1>Stereovision (estimated using neural methods)</td><td rowspan=1 colspan=1>566</td><td rowspan=1 colspan=1>849</td></tr></table>

![](images/9e3527e18b3a250b4a83965c39a4e71c36e96a0374d0ccfcca95a24a394997d1.jpg)  
Fig. 2: Examples from the two analogue datasets under different lighting conditions. Top: In-house dataset; bottom: LunaPolaris dataset..

A quantitative comparison of the datasets, along with their characteristics, is provided in Table I.

## B. Metrics

We adopt the Human Disagreement Rate (HDR) metric introduced in W-RIZZ [4], which measures the fraction of annotated point pairs for which the model’s predicted ordinal relation disagrees with the human label, using a threshold t on the predicted score difference. Unlike absolute measures like MSE, HDR evaluates the relative ranking of traversability between points, aligning with the weakly supervised pairwise annotation scheme and accommodating the intrinsic uncertainty in assigning traversability scores. We report HDR averaged over thresholds $t \in [ 0 . 1 0 , 0 . 5 0 ]$ as the main metric, along with the standard deviation, with HDR at t = 0.25 for direct comparison with prior work. To reduce the influence of randomness, we ran all experiments 10 times with different seeds and report the mean and standard deviation.

## C. Results

1) Quantitative: Table II reports full numeric results of HDR@0.25 and average HDR across all train–test domain pairs and input modalities, while Figure 3 summarises the change in HDR@0.25 relative to the visual-only baseline across all train/test combinations and geometric fusion modes.

In in-domain settings, geometric fusion has a small effect throughout. For In-house → In-house, all three modes remain within one standard deviation of the visual baseline, with no mode showing a statistically distinguishable improvement. For LunaPolaris → LunaPolaris, Depth and Normals yield modest gains (∆ HDR@0.25 of -0.005 and -0.006, respectively), while Geometric produces the largest improvement of the two in-domain cases (-0.006, HDR@0.25 0.040 vs 0.046 for Visual).

TABLE II: Results of traversability evaluation among datasets, averaged over 10 random seeds.
<table><tr><td></td><td>Train Data Test Data Mode</td><td></td><td></td><td>HDR@0.25 ↓ Avg. HDR ↓</td></tr><tr><td rowspan="3">In-house</td><td rowspan="3">In-house</td><td>Visual</td><td>0.030 ± 0.003 0.043 ± 0.003</td><td></td></tr><tr><td>Depth</td><td> $0 . 0 3 0 \pm 0 . 0 0 2 \ 0 . 0 4 2 \pm 0 . 0 0 2$ </td><td></td></tr><tr><td>Normals Geometric</td><td>0.031 ± 0.003 0.042 ± 0.002</td><td> $0 . 0 2 3 \pm 0 . 0 0 2 0 . 0 4 3 \pm 0 . 0 0 2$ </td></tr><tr><td rowspan="4">LunaPolaris LunaPolaris</td><td rowspan="4"></td><td>Visual</td><td> $0 . 0 4 6 \pm 0 . 0 0 7 0 . 0 5 6 \pm 0 . 0 0 7$ </td><td></td></tr><tr><td>Depth</td><td></td><td> $0 . 0 4 1 \pm 0 . 0 0 7 0 . 0 5 3 \pm 0 . 0 0 7$ </td></tr><tr><td>Normals</td><td></td><td> $0 . 0 4 0 \pm 0 . 0 0 5 0 . 0 5 1 \pm 0 . 0 0 5$ </td></tr><tr><td></td><td></td><td>Geometric 0.040 ± 0.003 0.052 ± 0.005</td></tr><tr><td rowspan="3">In-house</td><td rowspan="3">LunaPolaris</td><td>Visual</td><td>0.119 ± 0.013 0.145 ± 0.010</td><td></td></tr><tr><td>Depth</td><td></td><td>0.107 ± 0.016 0.135 ± 0.015</td></tr><tr><td>Normals</td><td>0.115 ± 0.009 0.138 ± 0.010</td><td>Geometric 0.114 ± 0.007 0.139 ± 0.007</td></tr><tr><td rowspan="3">LunaPolaris In-house</td><td rowspan="3"></td><td>Visual</td><td></td><td>0.075 ± 0.014 0.091 ± 0.012</td></tr><tr><td>Depth</td><td></td><td>0.077 ± 0.007 0.093 ± 0.005</td></tr><tr><td>Normals</td><td>0.079 ± 0.011 0.095 ± 0.011</td><td></td></tr><tr><td rowspan="4">In-house</td><td rowspan="4">CE&#x27;3/CE&#x27;4</td><td>Visual</td><td>Geometric 0.072 ± 0.007 0.089 ± 0.007</td><td></td></tr><tr><td>Depth</td><td> $0 . 1 5 3 \pm 0 . 0 0 6 0 . 1 6 7 \pm 0 . 0 0 8$ </td><td></td></tr><tr><td></td><td>0.150 ± 0.015 0.168 ± 0.015</td><td></td></tr><tr><td>Normals</td><td>0.153 ± 0.008 0.166 ± 0.007</td><td>Geometric 0.154 ± 0.011 0.168 ± 0.009</td></tr><tr><td rowspan="4">LunaPolaris CE’3/CE’4</td><td rowspan="4"></td><td>Visual</td><td> $0 . 1 0 2 \pm 0 . 0 1 2 0 . 1 1 8 \pm 0 . 0 1 1$ </td><td></td></tr><tr><td>Depth</td><td></td><td> $0 . 1 0 5 \pm 0 . 0 1 4 0 . 1 2 3 \pm 0 . 0 1 1$ </td></tr><tr><td>Normals</td><td></td><td> $0 . 1 0 3 \pm 0 . 0 1 7 0 . 1 2 0 \pm 0 . 0 1 6$ </td></tr><tr><td>Geometric</td><td> $0 . 0 9 8 \pm 0 . 0 1 0 0 . 1 1 8 \pm 0 . 0 0 9$ </td><td></td></tr><tr><td rowspan="4">LunaPolaris + In-house</td><td rowspan="4">CE’3/CE’4</td><td>Visual</td><td></td><td>0.107 ± 0.008 0.125 ± 0.006</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Depth</td><td>0.113 ± 0.009 0.128 ± 0.008</td><td></td></tr><tr><td>Normals</td><td>0.109 ± 0.012 0.129 ± 0.008</td><td>Geometric 0.124 ± 0.012 0.138 ± 0.009</td></tr></table>

In cross-domain transfer, the results are inconsistent across both settings and modalities. The clearest benefit appears for In-house → LunaPolaris, where Depth reduces HDR@0.25 from 0.119 to 0.107 (∆ = -0.012), the largest absolute improvement observed. For LunaPolaris → In-house, Geometric achieves the best result (0.072 vs. 0.075 for Visual, $\Delta \ : = \ : - 0 . 0 0 3 )$ , while Depth and Normals slightly degrade performance. Transfer to CE-3/4 shows no consistent pattern: for In-house → CE-3/4, all modes perform within the noise of Visual, and for LunaPolaris + In-house → CE-3/4, Geometric is the worst-performing mode (0.124 vs 0.107 for Visual, ∆ = +0.017), the only case where geometric fusion causes a clearly detrimental effect.

Overall, no modality consistently outperforms the visualonly baseline across the evaluated settings. Where geometric fusion helps, the improvements are small relative to the seed-to-seed variance, and in the one case where it clearly changes the result – LunaPolaris + In-house → CE-3/4 – the effect is detrimental. We therefore do not claim that geometric fusion improves traversability estimation under the naive concatenation strategy used here; rather, these results establish a controlled baseline against which more expressive multimodal architectures can be compared in future work.

We note that the apparent benefit of geometric augmentation is substantially smaller when averaged over multiple seeds than under a single training run: effects that appear sizeable in an individual run (exceeding 30% relative HDR reduction in some transfers) fall within seed-to-seed variance once averaged over 10 runs. This reinforces the importance of multi-seed evaluation when comparing closely-performing modalities.

![](images/7597bc6bd43c302ead65021f8963e13860854ea3e76a20e2ccdfcd9d32dd8778.jpg)  
Fig. 3: Change in HDR@0.25 relative to the visual-only baseline for each geometric fusion mode (Depth, Normals, Geometric). Values below zero indicate improvement over the visual baseline; values above zero indicate degradation.

2) Visual: Figure 4 provides qualitative traversability predictions for all train–test pairs. To further highlight the differences between prediction outputs, we provide an additional qualitative visualisation. Since the traversability score distributions differ across methods, we first discard the top and bottom 5% of prediction values for each method. We then compute the mean of the remaining values and use it as a threshold to separate traversable and non-traversable regions. We overlay the resulting contours on the original RGB image, with coloured regions indicating areas classified as non-traversable. Visual-only-based predictions are shown in green, while depth-based predictions are shown in purple. In in-domain settings (Figs. 4a–b), both Visual and Depth heatmaps produce spatially coherent predictions that align well with annotated boundaries: flat regolith receives high traversability scores while rocks and crater features are correctly penalised. The differences are particularly visible in cross-domain evaluations on terrestrial analogue datasets. In some cases, the Visual model fails to capture subtle variations between terrain types, leading to uncertain or inconsistent predictions. For example, it may classify craters as highly traversable or flat surfaces as poorly traversable (Figs. 4c–d). For cross-domain evaluation on real lunar data, the visualbased predictions are generally spatially plausible, but the depth-enhanced models produce more cohesive regions and show greater confidence in distinguishing traversable from non-traversable terrain (Figs. 4e–g).

3) Edge Deployment Evaluation: To assess edge deployment feasibility, we evaluated each model variant on an NVIDIA Jetson AGX Orin (JetPack 6.2.2, L4T 36.5.0)

TABLE III: Inference throughput for different models and execution modes.
<table><tr><td>Mode</td><td>Execution mode</td><td>Throughput [FPS] ↑</td></tr><tr><td rowspan="5">Visual</td><td>CPU</td><td> $0 . 2 1 5 \pm 0 . 0 0 1$ </td></tr><tr><td>CUDA</td><td> $1 7 . 0 9 \pm 1 . 0 2$ </td></tr><tr><td>TensorRT FP32</td><td> $2 8 . 4 9 \pm 2 . 5 3$ </td></tr><tr><td>TensorRT FP16</td><td> $2 8 . 9 3 \pm 2 . 3 2$ </td></tr><tr><td>CPU</td><td> $0 . 2 1 9 \pm 0 . 0 0 1$ </td></tr><tr><td rowspan="4">Depth</td><td>CUDA</td><td> $1 6 . 7 4 \pm 1 . 1 9$ </td></tr><tr><td>TensorRT FP32</td><td> $2 7 . 3 4 \pm 2 . 3 3$ </td></tr><tr><td>TensorRT FP16</td><td> $2 9 . 1 4 \pm 2 . 4 5$ </td></tr><tr><td></td><td></td></tr><tr><td rowspan="4">Normals</td><td>CPU</td><td> $0 . 2 1 1 \pm 0 . 0 0 1$ </td></tr><tr><td>CUDA</td><td> $1 4 . 0 9 \pm 0 . 6 5$ </td></tr><tr><td>TensorRT FP32</td><td> $1 9 . 3 5 \pm 1 . 2 1$ </td></tr><tr><td>TensorRT FP16</td><td> $2 0 . 2 3 \pm 2 . 1 7$ </td></tr><tr><td rowspan="4">Geometric</td><td>CPU</td><td> $0 . 2 1 4 \pm 0 . 0 0 1$ </td></tr><tr><td>CUDA</td><td> $1 3 . 7 3 \pm 0 . 5 8$ </td></tr><tr><td>TensorRT FP32</td><td> $1 9 . 1 0 \pm 1 . 3 8$ </td></tr><tr><td>TensorRT FP16</td><td> $1 8 . 6 5 \pm 1 . 9 9$ </td></tr></table>

using ONNX Runtime 1.22.0 with four execution configurations: CPU, CUDA, TensorRT FP32, and TensorRT FP16. Measured execution time included both model inference and modality-specific preprocessing (depth inpainting and geometric descriptor computation). Throughput was reported as mean ± standard deviation over 100 inferences, expressed in frames per second (FPS).

Table III presents the throughput obtained for four model variants and four execution modes. CPU-only execution achieved approximately 0.21–0.22 FPS and was unsuitable for real-time operation. CUDA acceleration increased throughput to 13.7–17.1 FPS, while TensorRT produced between 18.7 and 29.1 FPS. The visual and depth variants reached almost 29 FPS with TensorRT, demonstrating the feasibility of near-real-time edge deployment.

The differences between the TensorRT FP32 and FP16 configurations were small relative to the measured variability. FP16 achieved slightly higher mean throughput for the visual, depth, and normals variants, whereas FP32 was marginally faster for the Geometric variant. Therefore, reduced numerical precision did not consistently improve throughput, and you should also consider its effect on prediction quality when selecting the deployment configuration.

The visual and depth variants obtained similar throughput because they require only basic preprocessing. Normals and Geometric achieved lower throughput because they must compute surface normals and additional terrain descriptors before inference. The results therefore show a trade-off between the information provided by geometric representations and the operating frequency of the full perception pipeline.

## V. CONCLUSIONS

We presented a cross-dataset evaluation of geometryaugmented DINOv2 features for lunar traversability estimation. Our method extends the weakly supervised CHUN-GUS framework by concatenating depth-derived geometric channels – depth, surface normals, and handcrafted terrain descriptors – with AnyUp-upsampled DINOv2 embeddings before the traversability prediction head. Experiments across two analogue facilities and real Chang’E-3/4 lunar imagery show that, under this naive concatenation strategy, geometric augmentation has a negligible and inconsistent effect in both in-domain and cross-domain settings: where improvements occur, they are small relative to seed-to-seed variance, and in one cross-domain case geometric fusion is clearly detrimental. We therefore treat these results not as evidence that geometry improves traversability estimation, but as a controlled baseline that isolates the marginal contribution of geometric cues over a strong visual-only model and against which future work can compare more expressive multimodal fusion architectures. Independently of fusion mode, the choice of training source has a larger effect on transfer quality: LunaPolaris proves a more effective training base for real-lunar deployment than our In-House facility, consistent with the broader finding that domain relevance, rather than sensing modality, is the dominant factor in crossdomain traversability transfer.

![](images/c4a9827392a02c6e9435761624cd12269cb849a8dc423eed8a0cdcad4163e078.jpg)

![](images/b43675c4296d100c0be68671a8187fb4700ab850af188423c78b97dab11389c3.jpg)

![](images/986965be774e9593453357379023871c364feaa1f2b1fc0a27429d642f50ca5c.jpg)

![](images/bbf5a7fed70fb44fee911dc00dc8d2466e58e3e2e8c5c9ad0abbb05c59a70e57.jpg)

![](images/f7f4f0c4676d1e680dfad4becfc3d8b2259c64427eab510a1ad2e7c764c5324f.jpg)

![](images/0a83e9d7670c3759d8a754b4c91c36cf16062723b9197906db4b335e7b36ef5e.jpg)  
(a) In-domain: In-House → In-House

![](images/3de03db905261d89c5d18824717b4fb1fe1da0f55b3c629b37f720df87320b0a.jpg)

![](images/b13c3551f29000d6bb96c6c08c1fbaaf1d309978a6206f3141a827a66a587a89.jpg)  
(b) In-domain: LunaPolaris → LunaPolaris

![](images/28949678b9214d82b2bb10741b772e14566495dab5cb996ea5f7c51f8475fdf7.jpg)

![](images/83ebdd7909e0fd8909f6b0993504a1d76a9c7875e9248f320f6c10a1fd13b1a7.jpg)

![](images/4288a104e8ff78e4684fecf72a4bb55af914203f863ac6692c9c0f033e159c1f.jpg)

![](images/6e6ec5e308c86d616c210ae2b7683687ae5ad0b9b18589aac12e8a3f346acbf2.jpg)

![](images/328a1c78ffb02b7b89a74de81739723b39eab86899b3abe348d2ac24d5c999d3.jpg)

![](images/274665fdb222905876e17c3c2f08f7c5fa1e4b527efd1750cd3b92bf722129da.jpg)  
(c) Cross-domain: In-House → LunaPolaris

![](images/d7e057e39726ecea5f5114d4a4b96ae2a6dc5d181280605d380d37ad0e62f836.jpg)

![](images/d875862d805c3013c315baf83b2b440127bcd0c45805ff5b151ad1cc616bac94.jpg)  
(d) Cross-domain: LunaPolaris → In-House

![](images/d40c5d9f0e90f17a15fadbcc5e3563e768ce3b25c6391b1e99f49d8c3eb625e9.jpg)

![](images/eadf3d43645430ac876ace22b66d2317ca84347a31f1799d910a9935d33fa9ed.jpg)

![](images/76d7a8081e615ffc250ca00be4aa8806f815662bebe5f92d5f2cfdfa690b3d9c.jpg)

![](images/a5d0e0a3ea26d0abe189f419f3db8b4f8cf58a0f04e858593e0f1e51257af43a.jpg)

![](images/37869d685b0c8a6af6edba4ca7acd95b94bc9473d74a9ded69212eef2525461f.jpg)

![](images/e382dee496dda61c943017866abb3f1f26fb6d18d94f7e4d97e34839ac5a5d19.jpg)  
(e) Cross-domain: In-House → Chang’E-3/4

![](images/cbfdef6c4a99504f9f653c47077f1fd93f77228ecdfbbe9afa9cf9af040598b5.jpg)

![](images/185905e01260214a2bd509f796ff8040154f8aef58b05179af2cb4367b38a3df.jpg)  
(f) Cross-domain: LunaPolaris → Chang’E-3/4

![](images/ae1cc3c6bba44843404a2e957055b615e4e34e8e71b67bb255624c6611ce4c7c.jpg)

![](images/bd729e718e19bbcd077ebb3747249141cb663719de61cc919f8c453d9cdf2912.jpg)

![](images/82b2aeaa698619f2f1d786b039ea6cb38fab683800806a55353686630ca7f573.jpg)

![](images/c7d030eb33b26468d8bcd60bb874f26d811dbc56519b7c0613ee40bbf0614f14.jpg)  
(g) Combined source: In-House + LunaPolaris → Chang’E-3/4  
Fig. 4: Qualitative traversability predictions across all train-test domain pairs. Traversability heatmaps: warmer colours indicate lower scores. Overall traversability: green - visual is more sensitive, purple - Depth is more sensitive.

## A. Limitations

The study has several limitations; we highlight two key issues. First, the handcrafted pairwise annotations were produced by a single annotator, which may introduce subjective bias and limit the assessment of traversability to the selected image regions and their visual characteristics. Second, the depth data across the evaluated datasets were obtained using different sensing modalities, including conventional stereo vision, deep learning-enhanced stereo vision, and indirect time-of-flight. Consequently, the observed generalisation of depth-based representations may partially reflect not only geometric invariance but also appearance normalisation introduced by the respective depth estimation pipelines.

## B. Future work

Future work will extend the evaluation by examining model robustness under varying lighting conditions and systematically comparing alternative methods of incorporating geometric information into the model. We also plan to shift the focus of this research from algorithm-centric evaluation toward robot-centric validation. This includes integrating the proposed method with a path-planning algorithm and deploying it on a real rover in our in-house OTRK facility, where performance will be assessed using metrics such as trajectory length, obstacle avoidance, and power consumption. Another direction for future research is to investigate how visual traversability information can support teleoperation by analysing its impact on operator behaviour, navigation accuracy, and overall task performance.

## DATA ACCESS

The dataset (colour images, depth maps, camera parameters, and annotations) collected in our lunar facility is publicly available here: https://doi.org/10.5281/ zenodo.22871299.

## DISCLOSURE

The authors acknowledge the use of a large language model solely for editing and enhancing the manuscript’s grammar. No sections of the article, including the main technical content, figures, or images, were generated by the AI system. The use was limited to improving readability and clarity of language.

## ACKNOWLEDGMENT

Jan Baturo’s work was supported by the European Space Agency (ESA Spaceship Poland).

## REFERENCES

[1] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “DINOv2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[2] A. Schreiber and K. Driggs-Campbell, “Do you know the way? human-in-the-loop understanding for fast traversability estimation in mobile robotics,” IEEE Robotics and Automation Letters, 2025.

[3] B. Ptak, “From earth to the moon: A cross-domain study of weaklysupervised traversability estimation,” in IEEE ICRA’26 Workshop on Perceptual Challenges for Planetary Exploration, 2026.

[4] A. Schreiber, A. N. Sivakumar, P. Du, M. V. Gasparino, G. Chowdhary, and K. Driggs-Campbell, “W-RIZZ: A weakly-supervised framework for relative traversability estimation in mobile robotics,” IEEE Robotics and Automation Letters, vol. 9, no. 6, pp. 5623–5630, 2024.

[5] D. van der Meer, U. Wong, and M. A. OLIVARES MENDEZ, “LunaPolaris: A stereo camera, point cloud and imu dataset for future lunar exploration in polar regions,” in iSpaRo-International Conference on Space Robotics, 2026.

[6] Ground Research and Application System of China’s Lunar and Planetary Exploration Program, “Chang’E-3 Panoramic Cameras Dataset,” 2015, accessed via GRAS data portal. [Online]. Available: https://moon.bao.ac.cn

[7] ——, “Chang’E-4 Panoramic Cameras Dataset,” 2020, accessed via GRAS data portal. [Online]. Available: https://moon.bao.ac.cn

[8] J. Seo, T. Kim, S. Ahn, and K. Kwak, “METAVerse: Meta-learning traversability cost map for off-road navigation,” 2024. [Online]. Available: https://arxiv.org/abs/2307.13991

[9] H. Xue, H. Fu, R. Ren, J. Zhang, B. Liu, Y. Fan, and B. Dai, “LiDARbased drivable region detection for autonomous driving,” 10 2021.

[10] D. D. Fan, K. Otsu, Y. Kubo, A. Dixit, J. Burdick, and A.- A. Agha-Mohammadi, “STEP: Stochastic traversability evaluation and planning for risk-aware off-road navigation,” 2021. [Online]. Available: https://arxiv.org/abs/2103.02828

[11] S. Jung, J. Lee, X. Meng, B. Boots, and A. Lambert, “V-STRONG: Visual self-supervised traversability learning for off-road navigation,” 05 2024, pp. 1766–1773.

[12] T. Guan, Z. He, R. Song, D. Manocha, and L. Zhang, “TNS: Terrain traversability mapping and navigation system for autonomous excavators,” 2022. [Online]. Available: https://arxiv.org/abs/2109. 06250

[13] J. Seo, S. Sim, and I. Shim, “Learning off-road terrain traversability with self-supervisions only,” IEEE Robotics and Automation Letters, vol. 8, no. 8, p. 4617–4624, Aug. 2023. [Online]. Available: http://dx.doi.org/10.1109/LRA.2023.3284356

[14] L. Yang, C. Liang, X. He, and D. Zhao, “TTA-GEP: Terrain traversability analysis with geometry and environmental perception for the path planning of planetary rovers,” International Journal of Aerospace Engineering, vol. 2023, no. 1, p. 7147168, 2023. [Online]. Available: https://onlinelibrary.wiley.com/doi/abs/10.1155/ 2023/7147168

[15] S. Fu, M. Hamilton, L. Brandt, A. Feldman, Z. Zhang, and W. T. Freeman, “FeatUp: A model-agnostic framework for features at any resolution,” 2024. [Online]. Available: https://arxiv.org/abs/2403. 10516

[16] T. Wimmer, P. Truong, M.-J. Rakotosaona, M. Oechsle, F. Tombari, B. Schiele, and J. E. Lenssen, “AnyUp: Universal feature upsampling,” arXiv preprint arXiv:2510.12764, 2025.

[17] P. Ludivig, A. Calzada-Diaz, M. A. OLIVARES MENDEZ, H. VOOS, and J. Lamamy, “Building a piece of the moon: Construction of two indoor lunar analogue environments,” in 71st International Astronautical Congress (IAC)–The CyberSpace Edition, 2020.

[18] W. Zuo et al., “China’s lunar and planetary data system: Preserve and present reliable chang’e project and Tianwen-1 scientific data sets,” Space Science Reviews, 2021.

[19] J.-F. Yang, C.-L. Li, B. Xue, P. Ruan, W. Gao, W.-D. Qiao, D. Lu, X.-L. Ma, F. Li, Y.-H. He et al., “Panoramic camera on the yutu lunar rover of the chang’e-3 mission,” Research in Astronomy and Astrophysics, vol. 15, no. 11, pp. 1867–1880, 2015.

[20] D. DeTone, T. Malisiewicz, and A. Rabinovich, “SuperPoint: Selfsupervised interest point detection and description,” in Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 2018, pp. 224–236.

[21] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, “LightGlue: Local feature matching at light speed,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 17 627–17 638.

[22] J. Min, Y. Jeon, J. Kim, and M. Choi, “S2M2: Scalable stereo matching model for reliable depth estimation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 26 729–26 739.