# ARCOS: Zero-shot Boundary Localization for Corneal Layer Segmentation Across Optical Coherence Tomography Devices

Nuno Vivas Br´as<sup>1</sup>, Benjamin Memmi<sup>2</sup>, Ma¨elle Bouhassane<sup>1</sup>, Cristina Georgeon<sup>2</sup>, Vincent Borderie<sup>2</sup>, Karsten Plamann<sup>1</sup>, and Anatole Chessel<sup>1</sup>

<sup>1</sup> Laboratoire d’Optique et Biosciences, CNRS, INSERM, Ecole polytechnique, Institut <sup>´</sup> Polytechnique de Paris, Palaiseau, France

<sup>2</sup> GRC 32, Transplantation et Th´erapies Innovantes de la Corn´ee, Sorbonne Universit´e, Centre Hospitalier National d’Ophtalmologie des Quinze-Vingts, Paris, France

## Abstract

Accurate segmentation of corneal layers in optical coherence tomography (OCT) is essential for quantitative assessment of corneal morphology, including layer thickness and structural changes associated with disease or surgery. However, automatic segmentation remains challenging because corneal interfaces are thin, afected by speckle noise, and variable across acquisition devices. In this work, we propose ARCOS, a patch-based zero-shot boundary localization framework for corneal layer segmentation in clinical anterior-segment OCT images. Rather than performing conventional region classification, the method predicts boundary heatmaps for the main corneal interfaces from overlapping native-resolution patches. Patch-level predictions are stitched across the full B-scan and converted into boundary locations to obtain continuous, anatomically ordered layer segmentations. The network combines multi-scale feature fusion with a self-conditioned refinement module that uses intermediate boundary information to improve local heatmap predictions while preserving spatial detail. The method was evaluated on clinical OCT images acquired from multiple devices and compared with representative segmentation baselines using boundary localization and derived thickness metrics. The proposed method achieved an of-by-one boundary localization accuracy of 95.1% and a mean absolute boundary error of 0.514 pixels on the matched-device test set. In zero-shot cross-device evaluation, it maintained an average ofby-one accuracy of 84.3% and a mean absolute boundary error of 0.855 pixels across unseen acquisition devices, outperforming the baseline models. Thickness estimates derived from the predicted boundaries showed low error across corneal regions, supporting the method’s use for quantitative corneal OCT analysis.

Keywords: Optical coherence tomography, corneal segmentation, boundary localization, deep learning, zero-shot generalization, cross-device generalization.

## 1 Introduction

Anterior segment optical coherence tomography (AS-OCT) has become an important imaging modality in corneal ophthalmology, ofering non-invasive cross-sectional visualization of the cornea and its layered architecture. Based on low-coherence interferometry, AS-OCT provides micrometer-scale axial resolution and enables visualization of the major corneal layers, including the epithelium and stroma. In high-quality acquisitions, finer structures such as Bowman’s layer, Descemet’s membrane, and the endothelial interface may also be identified (see Fig. 1).

Beyond structural imaging, AS-OCT facilitates assessment of corneal morphology in several ocular conditions, including keratoconus, Fuchs endothelial corneal dystrophy (FECD), infectious keratitis, and post-surgical remodeling following procedures such as photorefractive keratectomy (PRK). Reliable delineation of corneal layers and their interfaces allows the extraction of clinically relevant measurements, including epithelial and stromal thickness maps, corneal geometry descriptors, and structural biomarkers that may support diagnosis, disease monitoring, and surgical planning [1, 2].

Accurate localization of corneal interfaces remains challenging, however. These boundaries can appear faint, noisy, or locally ambiguous due to pathology-related deformation, speckle noise, acquisition artifacts, and device diferences. Manual annotation is time-consuming, expertdependent, and dificult to perform consistently, especially in large or longitudinal datasets. Automated segmentation methods are therefore needed to improve reproducibility, reduce analysis time, and support the development of objective imaging biomarkers for clinical and research applications.

![](images/6e944afc9d8895123780f9a2baaa346d411e4e36b588315ff82a2d87ff0ce849.jpg)  
Figure 1: Representative clinical corneal OCT B-scan with an enlarged of view of the central corneal region. The principal cornea llayers are labelled to illustrate the anatomical interfaces targeted by the proposed boundary-localisation model.

Deep learning has increasingly been applied to corneal OCT layer and interface segmentation, building on convolutional and fully convolutional architectures for dense image prediction. U-Netbased encoder–decoder [3] models and related architectures have been used to delineate corneal layers and interfaces, with subsequent approaches introducing boundary-guided supervision, attention mechanisms, or multi-scale feature extraction to improve localization of thin or low-contrast structures [4–7].

While these approaches have demonstrated accurate corneal layer segmentation within their respective datasets and acquisition settings, their evaluation has predominantly focused on relatively consistent imaging conditions, often using the same scanner or acquisition protocol. Consequently, it remains unclear how well such models generalize to OCT systems that were entirely unseen during training, where diferences in image resolution, contrast, noise characteristics, and scanner-specific appearance can introduce substantial domain shifts. Broader clinical translation is further constrained by the limited availability of public datasets with detailed annotations of individual corneal layers, which complicates reproducible benchmarking and direct comparison between methods.

These challenges motivate segmentation methods that preserve local anatomical detail while maintaining coherent full-image boundary structure under diferent imaging conditions. In this work, we propose ARCOS, a patch-based framework for corneal boundary detection in AS-OCT images. The method processes overlapping full-resolution patches, predicts boundary heatmaps for corneal interfaces, and reconstructs full-resolution boundary estimates by combining patchlevel predictions across the entire B-scan. This design avoids resizing the entire image to a fixed input resolution and accommodates variations in image dimensions, pixel spacing, field of view,

and scanner-specific characteristics.

Rather than treating corneal layer segmentation as a conventional region classification problem, the proposed method defines the task as boundary localization. This formulation reflects the geometry of corneal OCT, where clinically relevant measurements depend on thin, continuous, and anatomically ordered interfaces rather than on large object-like regions.

The method is designed to improve interface localization, reduce dependence on user-defined parameter tuning, and support segmentation across diferent devices and pathological cases.

## 2 Materials and Methods

## 2.1 Devices and Datasets

The datasets used in this study included clinical acquisitions and public datasets obtained with five AS-OCT systems from diferent manufacturers. Clinical data were acquired using three SD-OCT systems: RTVue-XR Avanti (Optovue Inc., Fremont, CA, USA), Solix (Visionix/Optovue Inc., Fremont, CA, USA), and MS-39 (Costruzione Strumenti Oftalmici, Firenze, Italy). Two additional SS-OCT public datasets, MCOA [8], acquired using a TowardPi device (TowardPi Medical Technology Co., Ltd., Beijing, China), and AIDK [9], acquired using CASIA SS-1000 (Tomey Corporation, Nagoya, Japan), were included to further assess generalization to unseen acquisition technologies and image characteristics. Hereafter, these acquisition devices are referred to as Avanti, Solix, MS-39, TowardPi, and CASIA SS-1000.

The evaluated systems difered in OCT modality, resolution, field of view, and pixel spacing, as summarised in Table 1. These diferences afect both the physical sampling of corneal anatomy and the appearance of tissue interfaces, providing a heterogeneous setting for evaluating cross-device segmentation performance.

Table 1: Technical characteristics of the OCT systems used in this study. Optical resolution and field of view (FOV) are reported from device specifications. Pixel spacing was estimated from image dimensions and scan size.
<table><tr><td>Characteristic</td><td>Avanti</td><td>Solix</td><td>MS-39</td><td>TowardPi</td><td>CASIA SS-1000</td></tr><tr><td colspan="6">OCT technology</td></tr><tr><td>Type</td><td>SD-OCT</td><td>SD-OCT</td><td>SD-OCT</td><td>SS-OCT</td><td>SS-OCT</td></tr><tr><td colspan="6">Nominal optical resolution</td></tr><tr><td>Axial resolution (µm)</td><td>5.0</td><td>5.0</td><td>3.6</td><td>1.9</td><td>10.0</td></tr><tr><td>Lateral resolution (µm)</td><td>15.0</td><td>18.0</td><td>35.0</td><td>1.0</td><td>30.0</td></tr><tr><td colspan="6">Scan geometry and image sampling</td></tr><tr><td>FOV (mm)</td><td>8×3</td><td>8×3</td><td>16 × 8</td><td>24 × 15</td><td>16 × 6</td></tr><tr><td>Axial pixel spacing (µm/px)</td><td>4.31</td><td>3.91</td><td>3.60</td><td>4.10</td><td>6.00</td></tr><tr><td>Lateral pixel spacing (µm/px)</td><td>7.81</td><td>4.10</td><td>4.80</td><td>6.20</td><td>9.30</td></tr></table>

In addition to acquisition-related variability, the datasets also difered in clinical composition. The final evaluation set included healthy corneas, post-surgical PRK cases, Fuchs’ endothelial corneal dystrophy, and infectious keratitis, as reported in Table 2. Together, this provides a diverse evaluation setting under both device-related and biological variability.

Table 2: Summary of OCT datasets used in the study.
<table><tr><td>Device</td><td>Cohort</td><td>Subjects Scans Notes</td><td></td><td></td></tr><tr><td>SD-OCT</td><td></td><td></td><td></td><td></td></tr><tr><td>Avanti</td><td>Myopic PRK</td><td>41</td><td>169</td><td>65% pre-op, 35% post- op.</td></tr><tr><td>Avanti</td><td>Hyperopic PRK</td><td>11</td><td>28</td><td>75% pre-op, 25% post- op.</td></tr><tr><td>Avanti</td><td>Fuchs’ dystro- phy</td><td>36</td><td>83</td><td>FECD.</td></tr><tr><td>Solix</td><td>Healthy</td><td>3</td><td>20</td><td></td></tr><tr><td>MS-39</td><td>Healthy</td><td>11</td><td>20</td><td></td></tr><tr><td>SS-OCT</td><td></td><td></td><td></td><td></td></tr><tr><td>TowardPi</td><td>Healthy</td><td>17</td><td>20</td><td></td></tr><tr><td>CASIA 1000</td><td>SS- Keratitis</td><td>7</td><td>20</td><td></td></tr><tr><td>Total</td><td></td><td>126</td><td>360</td><td></td></tr></table>

## 2.2 Annotation Protocol

Certain thin corneal interfaces approach the axial resolution limits of standard clinical OCT systems, making reliable delineation of all individual layers dificult in some scans. The annotation protocol was defined according to visually and clinically distinguishable interfaces. Five boundary lines were manually delineated for each B-scan:

1. Boundary 1: Air–epithelium

2. Boundary 2: Epithelium–epithelial basal layer + Bowman’s layer

3. Boundary 3: Epithelial basal layer + Bowman’s layer–stroma

4. Boundary 4: Stroma–Descemet membrane + endothelium

5. Boundary 5: Descemet membrane + endothelium–anterior chamber

Annotations were performed by a trained operator following a standardized protocol to ensure consistent delineation across scans and acquisition systems. These annotations were used as the reference ground truth for model training and evaluation.

To reduce annotation time and preserve spatial consistency, a sparse point-based strategy was used. For each boundary, the operator placed about 30 to 50 control points along the visible interface. These points were then used to reconstruct the complete boundary profile. For regular corneal geometries, a sixth-degree polynomial fit was used. Piecewise cubic spline interpolation was used in cases with local curvature changes or irregular boundary morphology. All fitted boundaries were visually inspected and corrected manually when necessary to ensure anatomical plausibility.

Annotation consistency was checked using 20 images from the myopic PRK and Fuchs’ dystrophy cohorts. Three additional trained operators independently re-annotated these images using the same protocol. This analysis estimated intrinsic manual annotation variability and provided a reference level for automated segmentation performance.

As shown in Table 3, inter-operator disagreement was low, with a pooled mean absolute error of 1.06  0.56 pixels across all pairwise comparisons. Larger deviations were observed mainly near the lateral image regions, where signal quality decreases, and interface delineation becomes less reliable.

Table 3: Pairwise inter-operator variability across all annotated boundaries. Values are reported as mean absolute error (MAE) between boundary positions.
<table><tr><td>Operator 1 Operator 2</td><td></td><td></td><td>Mean Median</td><td>SD</td><td>P90</td><td>P95</td><td>n</td></tr><tr><td colspan="8">Reference Operator Comparisons</td></tr><tr><td>Reference</td><td>Operator A</td><td>0.89</td><td>0.76</td><td>0.53</td><td>1.52</td><td>2.04</td><td>100</td></tr><tr><td>Reference</td><td>Operator B</td><td>0.79</td><td>0.70</td><td>0.39</td><td>1.29</td><td>1.59</td><td>100</td></tr><tr><td>Reference</td><td>Operator C</td><td>1.17</td><td>1.06</td><td>0.55</td><td>1.95</td><td>2.29</td><td>100</td></tr><tr><td colspan="8">Inter-operator Comparisons</td></tr><tr><td>Operator A</td><td>Operator C</td><td>1.22</td><td>1.13</td><td>0.61</td><td>2.15</td><td>2.37</td><td>100</td></tr><tr><td>Operator A</td><td>Operator B</td><td>1.01</td><td>0.89</td><td>0.51</td><td>1.64</td><td>1.86</td><td>100</td></tr><tr><td>Operator B</td><td>Operator C</td><td>1.31</td><td>1.25</td><td>0.55</td><td>1.93</td><td>2.34</td><td>100</td></tr><tr><td>All pairs</td><td>pooled</td><td>1.06</td><td>0.94</td><td>0.56</td><td>1.82</td><td>2.18</td><td>600</td></tr></table>

## 2.3 Method Overview

The proposed framework, illustrated in Fig. 2, performs corneal boundary detection in AS-OCT B-scans by formulating the task as boundary localization rather than direct region classification. Given an input image, overlapping patches are extracted along the B-scan while preserving the native image resolution. Each patch is processed by a U-Net-based encoder–decoder network that predicts heatmaps for the five corneal interfaces. Patch-level predictions are then mapped back to their original image coordinates and fused using Gaussian-weighted stitching, which gives greater weight to the central region of each patch and attenuates edge-related artifacts. The reconstructed heatmaps are finally converted into continuous boundary profiles across the complete B-scan.

This design avoids resizing full B-scans to a fixed input size, reducing dependence on image dimensions, field of view, and scan geometry. Consequently, device-specific diferences primarily afect the number and placement of extracted patches rather than the network input representation. The heatmap formulation reduces the sparsity of one-pixel boundary labels and provides spatial tolerance for thin structures and annotation uncertainty, while overlapping patches and weighted stitching reduce border artifacts during reconstruction.

## 2.4 Patch Extraction

A lightweight boundary detection procedure was first applied to estimate the approximate anterior and posterior corneal interfaces. The first prominent peak was identified as the anterior boundary, while the last prominent peak was used to estimate the posterior corneal interface. The detected surfaces were used exclusively to guide patch positioning and ensure consistent coverage of the corneal tissue region, rather than to provide precise anatomical localization.

Patches of 384 384 pixels were extracted with a target lateral overlap of 50%; when exact overlap was not possible, the lateral stride was adjusted uniformly to ensure complete B-scan coverage. Because neighboring regions are vertically displaced by the natural curvature of the cornea, the vertical position of each patch was adapted according to the estimated surfaces to maintain full visibility of the corneal tissue region.

The selected patch size was chosen to preserve the complete corneal thickness while providing suficient lateral anatomical context for stable boundary estimation. For all OCT systems included in this study, a patch height of 384 pixels was suficient to fully contain the corneal regions within the extracted region. However, the model can be used with diferent patch sizes if necessary, as discussed later in section 4.3.

## 2.5 Network Architecture and Heatmap Prediction

ARCOS formulates corneal layer delineation as a boundary heatmap regression task. The network predicts five boundary response maps, one for each anatomical interface. Manual annotations are converted into dense supervision targets by placing a Gaussian profile in each lateral column, centred at the annotated axial position of the corresponding interface.

3) Boundary Prediction Network  
4) Patch-wise Boundaries  
![](images/e5cf866d5c2c3303933e7bc5280a48d5dd24fa0c679f42b05fc442458116101a.jpg)

![](images/7cf5ee35e904b2964b6fc3878a95d4586e4a4dae8b97c036d8ef7cfdbcf5d5ce.jpg)

![](images/315ebe105fd18fae64d7b981d03606332b2f3322e81f5c1d6d524efa93e6b4a6.jpg)  
Figure 2: Overview of the proposed segmentation framework. An input AS-OCT B-scan is decomposed into overlapping patches, processed by the segmentation network to estimate boundary heatmaps, and reconstructed into full B-scan corneal interface predictions by merging the patch-level outputs.

The network architecture builds on a U-Net encoder–decoder design with multi-scale feature fusion and decoder refinement. A ConvNeXt-Small backbone [10] is used as the encoder to extract hierarchical feature representations. The decoder progressively restores spatial resolution by combining features across scales. Before fusion, attention gates [11] are applied to the skip connections to suppress irrelevant activations and emphasize boundary-relevant information.

The decoder produces a fused multi-scale representation, denoted by F, from which an initial set of boundary heatmaps $\mathbf { H } _ { \mathrm { i n i t } }$ is predicted. These heatmaps are refined using a lightweight self-conditioned residual module based on feature-wise linear modulation (FiLM) [12]. A compact patch-level embedding is first obtained by global average pooling, followed by an MLP:

$$
\mathbf { r } = \phi _ { \mathrm { M L P } } ( \mathrm { G A P } ( \mathbf { F } ) ) .\tag{1}
$$

The refinement branch receives the concatenated representation ${ \bf Z } = [ { \bf F } , { \bf H } _ { \mathrm { i n i t } } ]$ , which is projected using a $1 \times 1$ convolution, instance normalization, and ReLU activation to produce features U.

![](images/5e3343d980c370075857ebf6baa63c336e68e19782850709ba30afa282b9a89e.jpg)  
Figure 3: The proposed network predicts boundary heatmaps. A ConvNeXt-Small encoder extracts hierarchical features. An attention-gated decoder combines these features to predict initial corneal interface heatmaps. A lightweight self-conditioned FiLM refinement module predicts residual heatmap corrections, using a patch-level embedding for conditioning.

FiLM parameters generated from r modulate these features as

$$
\widetilde { \mathbf { U } } _ { c , : ; } = \gamma _ { c } \mathbf { U } _ { c , : ; } + \beta _ { c } , \quad \gamma = W _ { \gamma } \mathbf { r } + b _ { \gamma } , \quad \beta = W _ { \beta } \mathbf { r } + b _ { \beta } .\tag{2}
$$

A final $1 \times 1$ convolution predicts a residual correction $\Delta \mathbf { H } .$ , giving

$$
\mathbf { H } _ { \mathrm { m a i n } } = \mathbf { H } _ { \mathrm { i n i t } } + \alpha \Delta \mathbf { H } ,\tag{3}
$$

where $\alpha = 0 . 4$ is fixed for all experiments. This residual formulation preserves the main prediction pathway while allowing patch-level appearance information to guide local heatmap refinement.

During inference, explicit boundary coordinates are obtained using a column-wise hard argmax applied independently to each predicted interface heatmap. This converts dense heatmap outputs into boundary profiles while avoiding the averaging efects associated with soft-coordinate extraction when responses are broad or locally uncertain.

An overview of the complete architecture is shown in Fig. 3.

## 2.6 Training Strategy

The model was trained using the Adaptive Wing (AW) loss [13], which is well suited to heatmap regression. Compared with conventional losses such as mean squared error, AW loss adaptively balances errors around high-confidence target regions and background areas, encouraging accurate localization of boundary responses without allowing easy background pixels to dominate the optimization.

To improve generalization across heterogeneous OCT imaging conditions, training used a multi-component augmentation strategy targeting three main sources of variability: acquisition geometry and corneal shape, image appearance and contrast, and OCT-specific frequency and intensity characteristics. These were addressed using spatial and structural transformations, photometric perturbations, and frequency- and intensity-domain modifications, respectively. Augmentations were applied using a curriculum schedule, in which the probability or strength of each transformation was increased progressively during training. This allowed the model to first learn stable boundary localization from moderately perturbed examples before being exposed to stronger appearance and acquisition variability. An overview of the augmentation workflow is shown in Fig. 4.

![](images/2c7b631074cf5b49ceb46d49fe80d225399ea74066b721b57e400b8639fb8861.jpg)  
Figure 4: Overview of the curriculum augmentation strategy used during training. Spatial, photometric, and OCT-specific frequency/intensity transformations are introduced progressively by increasing their probability or strength over training. This exposes the model to variability in corneal shape, acquisition geometry, contrast, speckle-like texture, and scanner-dependent image appearance while maintaining stable early optimisation.

In addition to standard geometric and photometric augmentations, two OCT-specific transformations were included. Inspired by Fourier-domain adaptation and Fourier-based domain generalization methods [14,15], Fourier amplitude perturbation was applied by decomposing each image into magnitude and phase components in the Fourier domain, adding Gaussian noise to the magnitude spectrum, and reconstructing the image using the original phase. This preserves the main spatial organization of the corneal structures while introducing variations in frequency content and texture. Unlike target-domain adaptation methods, the transformation does not require access to target-device images and is used only to increase the range of scanner-like appearance variations observed during training. As a second OCT-specific transformation, histogram warping was applied to smooth the intensity distribution, further modeling variations in contrast, brightness response, and scanner-specific image rendering.

Together, these augmentations expose the model to variations in shape, sampling, contrast, texture, and intensity distribution that may occur across OCT systems and acquisition protocols. This training strategy complements the patch-based heatmap formulation by improving robustness to appearance changes that are not fully captured by standard augmentations alone.

## 3 Experiments

## 3.1 Data Splits

Avanti was used as the training device, while all other OCT systems were reserved exclusively for inference-time evaluation. The Solix, MS-39, TowardPi, and CASIA SS-1000 datasets were not used for parameter selection and were used solely to assess method flexibility and cross-device generalization.

Within the Avanti device, the myopic PRK cohort and 90% of the Fuchs’ dystrophy cohort were used for training and validation, with an 80/20 split. The in-distribution Avanti test set consisted of the hyperopic PRK cohort and the remaining 10% of Fuchs’ dystrophy cases. This provided a held-out test set containing both post-surgical and pathological cases.

Patient-level separation was enforced in all splits: all scans from a given patient were assigned to the same training, validation, or test partition to avoid data leakage. The resulting efective split was approximately 70% training, 18% validation, and 12% testing.

## 3.2 Preprocessing and Standardization

Despite substantial diferences across OCT systems, all datasets were processed using a common standardization pipeline, without device-specific preprocessing or post-processing. Intensity preprocessing was limited to linear scaling to preserve the native appearance characteristics of each acquisition system.

For all datasets, 384 384-pixel patches were extracted directly from the original B-scans. Consequently, device diferences were reflected in the number of extracted patches per scan and in the lateral overlap between neighboring patches, which ranged from approximately 50% to 58%.

Several devices showed reduced boundary visibility away from the corneal apex, particularly for thinner internal interfaces, leading to greater annotation uncertainty outside the central region. To ensure consistent evaluation across systems, metrics were computed within a common central 6 mm region of interest (ROI), reducing the influence of peripheral signal degradation and annotation uncertainty. This efect was most pronounced in some SS-OCT scans, particularly from the CASIA SS-1000.

## 3.3 Baseline Models

The proposed framework was compared with representative baselines covering ablated, full-image, and previously proposed segmentation formulations.

• Basic Model: an ablated version without multi-scale feature fusion, FiLM-based refinement, or OCT-specific augmentations.

• Full Size (Original Size): the same architecture and training strategy as the proposed framework, but applied directly to full-width OCT B-scans instead of overlapping patches.

• CorneaNet-based model: a U-Net-based corneal OCT segmentation baseline reimplemented following CorneaNet [4], using region-based segmentation.

• ScLNet-based Model: a hybrid architecture following ScLNet [16], combining boundary prediction with region-based segmentation supervision, operating on resized full-image inputs.

All baselines were trained and evaluated using the same dataset splits, preprocessing pipeline, and evaluation metrics. For models producing region-based segmentation maps rather than explicit boundary heatmaps, boundary coordinates were extracted from the predicted masks as the first pixel of the corresponding layer in each image column.

## 3.4 Evaluation Metrics

Segmentation performance was evaluated with boundary-based metrics after patch stitching and full-image boundary reconstruction. Unless otherwise stated, metrics were first computed per image and then averaged across scans from each OCT system to obtain device-level results.

The primary metric was of-by-one accuracy, hereafter referred to as accuracy, defined as the proportion of predicted boundary locations within 1 pixel of the reference annotation. This tolerance accounts for small annotation uncertainty at weak or ambiguous interfaces. Mean absolute error (MAE) was used to quantify average boundary deviation in pixels. To assess geometric consistency and failure cases, Hausdorf distance (HD) and its 95th percentile variant (HD95) were also computed, with HD95 reducing the influence of isolated outliers.

Statistical comparisons were performed separately for each OCT device using image-level paired observations, thereby avoiding the masking of acquisition-specific efects. Significance was assessed using the paired Wilcoxon signed-rank test, selected due to the skewed distributions and localized outliers often observed in boundary error metrics. p-values were adjusted for multiple comparisons using the Benjamini–Hochberg false discovery rate correction, with adjusted $p < 0 . 0 5$ considered significant.

## 3.5 Implementation Details

The proposed framework and its ablated variants were trained using the AdamW optimiser with an initial learning rate of $1 0 ^ { - 3 }$ , a weight decay of $1 0 ^ { - 4 }$ , and a ReduceLROnPlateau learning-rate scheduler. Boundary heatmap targets were generated using Gaussian profiles with $\sigma = 2 . 5$ pixels. This value was selected empirically after evaluating alternative kernel widths, as it provided the best overall localisation performance. The curriculum augmentation schedule described earlier was applied during training, with augmentation strength increased linearly until reaching its maximum value at 60% of the training process.

Models were trained with a batch size of 12 for up to 150 epochs. The CorneaNet-based and ScLNet-based baselines were trained using the optimization settings reported in their original publications.

All experiments were implemented in $\mathrm { P y }$ Torch and performed on an NVIDIA Tesla V100 GPU.

## 4 Results and Discussion

## 4.1 Training device results

In-distribution performance is shown in Table 4, where the proposed model achieved the best overall localization performance, with an accuracy of 95.1%, an MAE of 0.514 pixels, and an HD95 of 1.22 pixels. The Basic Model was the closest competitor, showing only a small performance gap, as expected under matched acquisition conditions where robustness-oriented components are less likely to show their full benefit.

Table 4: Quantitative segmentation performance under in-distribution conditions. Metrics are reported as mean values over the test set. Best-performing results for each metric are highlighted in bold.
<table><tr><td>Model</td><td>Accuracy ↑</td><td>MAE</td><td>HD95</td><td>HD ↓</td></tr><tr><td>Proposed Model</td><td>95.1</td><td>0.51</td><td>1.22</td><td>3.81</td></tr><tr><td>Baseline Models</td><td></td><td></td><td></td><td></td></tr><tr><td>Basic Model</td><td>94.5</td><td>0.54</td><td>1.25</td><td>4.22</td></tr><tr><td>Full Size</td><td>84.8</td><td>0.80</td><td>1.75</td><td>3.19</td></tr><tr><td>CorneaNet</td><td>89.4</td><td>0.77</td><td>2.04</td><td>7.76</td></tr><tr><td>ScLNet</td><td>87.2</td><td>0.76</td><td>1.53</td><td>2.99</td></tr></table>

Figure 5 shows the lateral MAE distribution for each corneal boundary using the proposed model. Errors remained low centrally and increased moderately towards the periphery, where OCT signal quality and boundary visibility are often reduced. A small local increase was observed near the corneal apex for the most anterior interfaces, possibly reflecting stronger superficial signal and local ambiguity at the air–epithelium and epithelium–Bowman’s boundaries. Nevertheless, MAE remained below 1 pixel for all boundaries, indicating stable localization across the evaluated field of view.

![](images/6ff31f2dc24e12bef1de1e795130a2a201b998addc9eda06f2c6a41144d271a2.jpg)  
Figure 5: Lateral variation of mean absolute error (MAE) for each corneal boundary. Shaded regions indicate confidence intervals across the test set.

Representative outputs in challenging cases are shown in Fig. 6, including keratoconus, FECD, low-signal acquisitions, and central imaging artifacts. The predicted boundaries remained smooth and anatomically ordered despite local anatomical deformation and reduced boundary contrast. In particular, ARCOS correctly delineated boundaries in scans with central corneal artifacts, despite not being trained on images containing this artifact.

## 4.2 Cross-Device Generalization

Cross-device generalization was assessed by applying each model to OCT systems excluded from training. As shown in Table 5, the proposed model demonstrated the most consistent performance across devices, achieving the highest accuracy and lowest MAE on all five systems, as well as the lowest HD and HD95 on four of them. The largest gains were observed for MS-39, TowardPi, and CASIA SS-1000, where baseline models exhibited more pronounced degradation in localization accuracy and Hausdorf-based metrics.

Figure 7 shows relative thickness error across corneal regions and acquisition systems. As expected, the largest percentage deviations occurred in thinner layers, where small boundary errors have a greater proportional efect on thickness estimates. Thickness errors nevertheless remained controlled across devices, supporting quantitative pachymetric analysis under heterogeneous imaging conditions.

Representative qualitative results are shown in Fig. 8. Despite visible diferences in image appearance between devices, the predicted interfaces remain continuous and anatomically ordered in the selected examples. This qualitative behavior is consistent with the quantitative results and indicates that the proposed model preserves coherent corneal layer geometry across diverse acquisition conditions.

Overall, the combination of native-resolution patch processing and domain-oriented training improves stability under device shift, reducing the severe performance degradation observed in several baseline methods.

![](images/b3761a6e22752a5fc13ac94d42caba714981a19d89173156368a15b23dc03e6a.jpg)  
Figure 6: Representative challenging cases, including keratoconus, FECD, low-signal acquisition, and central imaging artifacts. Ground-truth annotations and predicted boundaries are shown with magnified regions highlighting structural deformation and reduced image quality.

## 4.3 Model Robustness

To evaluate the efect of patch-scale spatial context, ARCOS was additionally tested with diferent inference patch sizes across all acquisition devices.

When inference was performed with larger patches (512  512), performance remained close to the reference 384  384 setting (Table 6). This suggests that ARCOS is relatively stable under moderate increases in patch-scale spatial context. The result is consistent with the scale-related augmentations used during training, including cropping, zooming, and rescaling, which exposed the model to variations in efective field of view.

In contrast, performance decreased with smaller patches (256  256). This suggests that, while the model is robust to larger contextual windows, very small patches may remove relevant corneal structure needed for accurate localization.

## 4.4 Ablation

To assess the contribution of individual components, ablation experiments were performed across all evaluation devices. Table 7 compares the complete model with variants obtained by removing selected architectural and training components.

Table 5: Cross-device segmentation performance across OCT acquisition systems. Best values are highlighted in bold.
<table><tr><td>Model</td><td>Avanti</td><td>Solix</td><td>MS39</td><td>TowardPi</td><td>CASIA SS-1000</td></tr><tr><td colspan="6">Accuracy ↑</td></tr><tr><td>Proposed</td><td>95.1</td><td>94.8</td><td>86.1</td><td>75.9</td><td>80.5</td></tr><tr><td>Basic</td><td>94.5</td><td>93.7</td><td>68.3</td><td>67.9</td><td>76.4</td></tr><tr><td>Full Image</td><td>84.8</td><td>66.6</td><td>68.3</td><td>56.6</td><td>51.7</td></tr><tr><td>CorneaNet</td><td>89.4</td><td>87.6</td><td>53.7</td><td>62.5</td><td>68.9</td></tr><tr><td>ScLNet</td><td>87.2</td><td>83.1</td><td>76.6</td><td>37.7</td><td>79.1</td></tr><tr><td colspan="6">MAE↓</td></tr><tr><td>Proposed</td><td>0.51</td><td>0.67</td><td>0.79</td><td>1.01</td><td>0.95</td></tr><tr><td>Basic</td><td>0.54</td><td>0.61</td><td>1.35</td><td>1.23</td><td>1.04</td></tr><tr><td>Full Image</td><td>0.80</td><td>1.98</td><td>1.31</td><td>1.97</td><td>3.16</td></tr><tr><td>CorneaNet</td><td>0.77</td><td>0.91</td><td>33.1</td><td>3.66</td><td>2.57</td></tr><tr><td>ScLNet</td><td>0.76</td><td>5.25</td><td>1.11</td><td>6.41</td><td>0.97</td></tr><tr><td colspan="6">HD95↓</td></tr><tr><td>Proposed</td><td>1.22</td><td>1.24</td><td>1.88</td><td>2.05</td><td>2.11</td></tr><tr><td>Basic</td><td>1.25</td><td>1.32</td><td>2.58</td><td>2.32</td><td>2.20</td></tr><tr><td>Full Image</td><td>1.75</td><td>6.02</td><td>2.65</td><td>4.19</td><td>8.49</td></tr><tr><td>CorneaNet</td><td>2.04</td><td>2.75</td><td>5.51</td><td>4.42</td><td>8.17</td></tr><tr><td>ScLNet</td><td>1.53</td><td>18.4</td><td>2.30</td><td>24.4</td><td>1.95</td></tr><tr><td colspan="6">HD↓</td></tr><tr><td>Proposed</td><td>3.81</td><td>3.48</td><td>4.15</td><td>2.79</td><td>3.77</td></tr><tr><td>Basic</td><td>4.22</td><td>4.68</td><td>13.7</td><td>4.04</td><td>3.41</td></tr><tr><td>Full Image</td><td>3.19</td><td>31.4</td><td>6.55</td><td>29.8</td><td>20.4</td></tr><tr><td>CorneaNet</td><td>7.76</td><td>17.1</td><td>48.5</td><td>12.7</td><td>47.9</td></tr><tr><td>ScLNet</td><td>2.99</td><td>38.9</td><td>16.6</td><td>57.4</td><td>4.62</td></tr></table>

Table 6: Performance comparison between inference patch sizes across acquisition devices. The model was trained using a reference patch size of $3 8 4 \times 3 8 4$ . Values are reported separately for each device.
<table><tr><td rowspan="2">Device</td><td colspan="3">Accuracy ↑</td><td colspan="3">MAE↓</td><td colspan="3">HD95↓</td><td colspan="3">HD ↓</td></tr><tr><td>256</td><td>384</td><td>512</td><td>256</td><td>384</td><td>512</td><td>256</td><td>384</td><td>512</td><td>256</td><td>384</td><td>512</td></tr><tr><td>Avanti</td><td>94.9</td><td>95.1</td><td>94.8</td><td>0.53</td><td>0.51</td><td>0.53</td><td>33.35</td><td>1.22</td><td>1.29</td><td>49.2</td><td>3.81</td><td>1.79</td></tr><tr><td>Solix</td><td>93.9</td><td>94.8</td><td></td><td>94.4 0.96</td><td>0.67</td><td></td><td>0.58 4.36</td><td>1.24</td><td>1.27</td><td>31.15</td><td>3.48</td><td>2.51</td></tr><tr><td>MS-39</td><td>76.1</td><td>86.1</td><td></td><td>84.4 3.94</td><td>0.79</td><td></td><td>0.86 18.4</td><td>1.88</td><td>1.97</td><td>83.9</td><td>4.15</td><td>5.59</td></tr><tr><td>TowardPi</td><td>73.0</td><td>75.9</td><td></td><td>71.9 2.07</td><td>71.01</td><td></td><td>1.16 4.83</td><td>2.05</td><td>2.25</td><td>76.3</td><td>2.79</td><td>10.8</td></tr><tr><td>CASIA SS- 76.8 1000</td><td></td><td>80.5</td><td></td><td></td><td></td><td></td><td></td><td>81.1 1.05 0.95 0.91 2.34 2.11</td><td>2.01</td><td>3.17</td><td></td><td>3.77 2.89</td></tr><tr><td>Mean</td><td>82.9 86.5 85.3 1.71 0.79 0.81 6.66 1.70 1.76 48.7 3.60 4.72</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

The full model achieved the strongest overall performance, with the best balance between localization accuracy and geometric stability. The largest degradation was observed when Adaptive Wing heatmap supervision was replaced by mean squared error, increasing MAE from 0.79 to 1.48 pixels and markedly worsening HD95 and HD. This indicates that the heatmap loss is important for stable boundary localization.

Removing attention gates, FiLM-based refinement, or multi-scale fusion also reduced performance, with larger drops when components were removed jointly. Statistical testing confirmed that the full model significantly outperformed each ablated variant across the reported metrics after multiple-comparison correction, except for HD95 and HD when compared with the ConvNeXt-Tiny and no-attention ablations.

Overall, the ablation results indicate that the proposed performance gains arise from the combined efect of heatmap supervision, architectural refinement, multi-scale fusion, and OCTspecific augmentation, rather than from a single isolated component.

Cross-Machine Thickness Variation by Region  
![](images/23fc17ad9b1437d5529829408d15fc2df8e155f6863ef8f6c0ddd53226964a39.jpg)

Figure 7: Relative thickness error of ARCOS across corneal regions and OCT acquisition systems. Percentage errors are larger in thinner layers, where small localization diferences have a greater proportional efect on thickness estimation.  
![](images/e318ff8e3bbdbffd0c71b3e4d3ee1e11705684efc988ff90dbb31eaf23f0489c.jpg)  
Figure 8: Representative cross-device segmentation results on OCT images acquired with diferent systems. Ground-truth annotations and predicted boundaries are shown for each example, illustrating the preservation of continuous and anatomically ordered corneal interfaces under device-related domain shift.

## 4.5 Failure Cases

Despite the strong overall performance across devices and challenging imaging conditions, several recurrent failure modes were identified (Fig. 9).

The most common failure mode occurred in edge patches, where one or more anatomical interfaces left the visible image region before the patch boundary was reached. In these cases, the model occasionally continued the predicted interface beyond the region where the corresponding structure was physically visible.

Table 7: Ablation study across all evaluation devices. Values are reported as the mean across devices. Best-performing results for each metric are highlighted in bold.
<table><tr><td>Model Variant</td><td>Accuracy ↑</td><td>MAE↓</td><td>HD95</td><td>HD ↓</td></tr><tr><td>Proposed Model</td><td>86.5</td><td>0.79</td><td>1.70</td><td>3.60</td></tr><tr><td>Backbone</td><td></td><td></td><td></td><td></td></tr><tr><td>ConvNeXt Tiny</td><td>84.0</td><td>0.82</td><td>1.76</td><td>3.25</td></tr><tr><td>ResNet34</td><td>73.1</td><td>1.18</td><td>3.16</td><td>8.27</td></tr><tr><td>EfficientNet-B0</td><td>77.7</td><td>0.99</td><td>1.99</td><td>4.70</td></tr><tr><td>Loss Function</td><td></td><td></td><td></td><td></td></tr><tr><td>Heatmap MSE</td><td>76.8</td><td>1.48</td><td>3.63</td><td>24.3</td></tr><tr><td colspan="5">Attention and FiLM Modules</td></tr><tr><td>No Attention</td><td>83.7</td><td>0.84</td><td>1.77</td><td>3.55</td></tr><tr><td>No FiLM</td><td>84.2</td><td>0.83</td><td>1.82</td><td>3.81</td></tr><tr><td>No Attention + No FiLM</td><td>80.6</td><td>0.95</td><td>2.07</td><td>5.40</td></tr><tr><td colspan="3">Multi-scale Fusion</td><td></td><td></td></tr><tr><td>No Fusion</td><td>82.8</td><td>0.89</td><td>1.87</td><td>4.96</td></tr><tr><td>No FiLM + No Fusion</td><td>84.7</td><td>0.83</td><td>1.81</td><td>5.08</td></tr><tr><td>No Attention + No Fusion</td><td>84.4</td><td>0.84</td><td>1.81</td><td>5.24</td></tr><tr><td colspan="3">OCT-specific Augmentation</td><td></td><td></td></tr><tr><td>No OCT Augmentation</td><td>83.1</td><td>0.88</td><td>1.83</td><td>5.44</td></tr><tr><td>No Attention + No OCT</td><td>83.7</td><td>0.83</td><td>1.78</td><td>3.98</td></tr><tr><td>No Fusion + No OCT</td><td>77.3</td><td>1.08</td><td>2.13</td><td>6.41</td></tr><tr><td>No FiLM + No OCT</td><td>83.5</td><td>0.86</td><td>1.83</td><td>4.98</td></tr><tr><td colspan="3">Combined Ablations</td><td></td><td></td></tr><tr><td>No FiLM + No Fusion + No OCT</td><td>84.7</td><td>0.82</td><td>1.83</td><td>5.27</td></tr><tr><td>No Attention + No FiLM</td><td>79.4</td><td>1.0</td><td>1.92</td><td>5.24</td></tr><tr><td>+ No Fusion No Attention + No FiLM</td><td>79.8</td><td>0.96</td><td>1.96</td><td>6.09</td></tr><tr><td>+ No OCT No Attention + No</td><td>79.8</td><td>0.96</td><td>1.96</td><td>4.75</td></tr><tr><td>Fusion + No OCT No Attention + No FiLM</td><td>80.2</td><td>0.96</td><td>1.94</td><td>6.01</td></tr><tr><td>+ No Fusion + No OCT</td><td></td><td></td><td></td><td></td></tr></table>

![](images/9532527ee568bc83843287fc9992f954e7aff20e5810cb06de3c7e46ba124486.jpg)  
Figure 9: Representative failure cases of the proposed model. Examples include peripheral patches where anatomical interfaces leave the visible image region, and scans afected by strong central artifacts that reduce visibility of the posterior boundary.

The practical impact of this failure mode is limited for the main quantitative analyses in this study, since evaluation and thickness measurements were restricted to the central 6 mm region of interest. Nevertheless, this issue could be mitigated by anatomical post-processing, for example, by masking predictions after an interface reaches the upper or lower image boundary.

A second failure mode was observed in scans afected by strong central imaging artifacts, where posterior interface localization could become unreliable within the corrupted region. For downstream analyses, these regions could be excluded or interpolated from neighboring valid predictions when appropriate to reduce the influence of localized errors.

Overall, these failure modes were relatively uncommon, occurring in approximately 7% of test cases. They primarily occur in scan edges or reflect situations with strong imaging artifacts. These observations identify practical limitations of the current framework and suggest future improvements through uncertainty estimation, artifact detection, and anatomically constrained post-processing.

## 4.6 Running Time

Inference time was measured for the complete processing pipeline, including patch extraction, network prediction, heatmap reconstruction, and boundary stitching. The proposed method required, on average, 0.52 s per B-scan on an NVIDIA Quadro P2000 GPU and approximately 3.8 s per B-scan on an Intel Xeon These measurements indicate that full-scan segmentation can be performed within a few seconds on standard hardware, supporting practical use in clinical and research workflows.

## 5 Conclusion

This work presented a patch-based boundary heatmap framework for corneal layer detection in AS-OCT images, with a particular focus on zero-shot cross-device generalisation. By using native-resolution overlapping patches, the method preserves local anatomical detail, while boundary heatmap regression directly targets the ordered corneal interfaces. Across multiple OCT systems, the proposed method achieved accurate boundary localisation and demonstrated strong zero-shot cross-device generalisation, outperforming representative baseline architectures on unseen devices without device-specific fine-tuning. These results support boundary-focused, resolution-preserving segmentation strategies for robust corneal OCT analysis in clinical settings, particularly across heterogeneous imaging devices and patient populations.

Future work will focus on broader validation and adaptation to the full field of view of diferent OCT systems, including uncertainty-based masking of poorly visible inner boundaries and post-processing strategies to flag and mitigate failure cases.

## References

[1] Valentin Aranha dos Santos, Leopold Schmetterer, Martin Gr¨oschl, Gerhard Garhofer, Doreen Schmidl, Martin Kucera, Angelika Unterhuber, Jean-Pierre Hermand, and Ren´e M. Werkmeister. In vivo tear film thickness measurement and tear film dynamics visualization using spectral domain optical coherence tomography. Opt. Express, 23(16):21043–21063, Aug 2015.

[2] Ren´e M. Werkmeister, Sabina Sapeta, Doreen Schmidl, Gerhard Garh¨ofer, Gerald Schmidinger, Valentin Aranha dos Santos, Gerold C. Aschinger, Isabella Baumgartner, Niklas Pircher, Florian Schwarzhans, Anca Pantalon, Harminder Dua, and Leopold

Schmetterer. Ultrahigh-resolution oct imaging of the human cornea. Biomed. Opt. Express, 8(2):1221–1239, Feb 2017.

[3] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. 05 2015.

[4] Valentin Aranha dos Santos, Leopold Schmetterer, Hannes Stegmann, Martin Pfister, Alina Messner, Gerald Schmidinger, Gerhard Garhofer, and Ren´e M. Werkmeister. Corneanet: fast segmentation of cornea oct scans of healthy and keratoconic eyes using deep learning. Biomed. Opt. Express, 10(2):622–641, Feb 2019.

[5] Tejas Sudharshan Mathai, Kira Lathrop, and John Galeotti. Learning to Segment Corneal Tissue Interfaces in OCT Images. arXiv e-prints, page arXiv:1810.06612, October 2018.

[6] Lei Wang, Meixiao Shen, Qian Chang, Ce Shi, Yang Chen, Yuheng Zhou, Yanchun Zhang, Jiantao Pu, and Hao Chen. Automated delineation of corneal layers on oct images using a boundary-guided cnn. Pattern Recognition, 120:108158, 2021.

[7] Yoel F. Garcia-Marin, David Alonso-Caneiro, Damien Fisher, Stephen J. Vincent, and Michael J. Collins. Patch-based cnn for corneal segmentation of as-oct images: Efect of the number of classes and image quality upon performance. Computers in Biology and Medicine, 152:106342, 2023.

[8] Xinyu Ma, Jianxia Fang, Yaqi Wang, Zhichao Hu, Zhe Xu, Sha Zhu, Weijia Yan, Mengqi Chu, Jingwei Xu, Siting Sheng, Chujun Liu, Mingxuan Zhang, Ce Shi, Gangyong Jia, and Wen Xu. Mcoa: A comprehensive multimodal dataset for advancing deep learning in corneal opacity assessment. Scientific Data, 12, 05 2025.

[9] Yiming Sun, Nuliqiman Maimaiti, Peifang Xu, Peng Jin, Jingxuan Cai, Guiping Qian, Pengjie Chen, Mingyu Xu, Gangyong Jia, Qing Wu, and Juan Ye. An as-oct image dataset for deep learning-enabled segmentation and 3d reconstruction for keratitis. Scientific Data, 11, 06 2024.

[10] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11966–11976, 2022.

[11] Ozan Oktay, Jo Schlemper, Loic Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Hammerla, Bernhard Kainz, Ben Glocker, and Daniel Rueckert. Attention u-net: Learning where to look for the pancreas. 04 2018.

[12] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. Film: visual reasoning with a general conditioning layer. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications of Artificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI’18/IAAI’18/EAAI’18. AAAI Press, 2018.

[13] Xinyao Wang, Liefeng Bo, and Fuxin Li. Adaptive wing loss for robust face alignment via heatmap regression. CoRR, abs/1904.07399, 2019.

[14] Yanchao Yang and Stefano Soatto. Fda: Fourier domain adaptation for semantic segmentation. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4084–4094, 2020.

[15] Qinwei Xu, Ruipeng Zhang, Ya Zhang, Yanfeng Wang, and Qi Tian. A fourier-based framework for domain generalization. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14378–14387, 2021.

[16] Yang Cao, Xiang le Yu, Han Yao, Yue Jin, Kuangqing Lin, Ce Shi, Hongling Cheng, Zhiyang Lin, Jun Jiang, Hebei Gao, and Meixiao Shen. Sclnet: A cornea with scleral lens oct layers segmentation dataset and new multi-task model. Heliyon, 10(13):e33911, 2024.