# A Multi-Source Ultrasound Benchmark Revealing the Limits of Contemporary Self-Supervised Anomaly Detection Methods

Marco Riedenauer

University of Augsburg

marco.riedenauer@uni-a.de

Daniel Kienzle University of Augsburg daniel.kienzle@uni-a.de

Pratik Mayekar University of Augsburg pratik.mayekar@uni-a.de

Rainer Lienhart University of Augsburg rainer.lienhart@uni-a.de

Abstract—Self-supervised anomaly detection is a promising paradigm for medical ultrasound, as normal images are often easier to obtain than exhaustive annotations of all possible pathologies. However, most existing evaluations are limited to a single anatomy or task, making it unclear whether models learn a robust notion of normal ultrasound appearance or only a source-specific representation. We introduce the Selfsupervised Anomaly Detection in UltraSound Imaging (SADUSI) benchmark, a multi-source ultrasound dataset designed to train and evaluate anomaly detection methods across a broad range of anatomical regions, views, and acquisition protocols. The goal of SADUSI is to provide a diverse normal ultrasound distribution and a benchmark for visible structural anomalies that can be assessed from single images. We evaluate representative selfsupervised anomaly detection methods and find that current approaches struggle in this setting. In particular, reconstructionbased diffusion methods such as AnoDDPM and DeCo-Diff achieve pixel-level AUROC values of 0.56–0.72 and maximum F1 scores of 0.10–0.26, indicating limited separation of pathology from normal image regions. Feature-based PatchCore variants perform better, reaching pixel-level AUROC values of 0.76–0.83, but remain limited with maximum F1 scores of 0.14–0.40. These findings suggest that broad multi-source ultrasound anomaly detection remains an open challenge and that SADUSI can serve as a resource for developing methods that generalize beyond anatomy-specific settings.

Index Terms—medical ultrasound, anomaly detection, selfsupervised learning, dataset, benchmark

## I. INTRODUCTION

Anomaly detection aims to identify observations that deviate from an expected notion of normality [1]. Statistically, anomalies are rare or unlikely samples under the data distribution observed during training. In visual inspection, only a subset of such statistical deviations is domain-relevant: medical anomalies may correspond to visible pathologies such as tumors, nodules, or deposits, while industrial anomalies may correspond to material defects such as cracks, inclusions, or voids [2]–[4].

Medical ultrasound imaging is widely used because it is comparatively inexpensive, real-time capable, portable, and free of ionizing radiation. Automatic detection of suspicious structures in ultrasound images could support clinical workflows by highlighting potentially relevant findings and reducing repetitive screening effort. Fully supervised training, however, remains difficult because relevant pathologies are diverse and often rare, pixel-level annotations require clinical expertise, and ultrasound appearance varies strongly across anatomy, view, device, operator, and acquisition protocol. Self-supervised and unsupervised anomaly detection methods address part of this limitation by learning from visually normal data and identifying deviations at inference time [2], [5].

Most existing medical ultrasound anomaly detection settings are anatomy- or task-specific. Models trained on one region may learn organ- or view-specific appearance rather than a broader notion of normal ultrasound that transfers across fetal, thyroid, liver, breast, lung, cardiac, or abdominal imaging. Conversely, a broadly trained model must distinguish visible pathological structures from the large variability of normal anatomy.

This work focuses on the visual inspection setting. We do not aim to build a dataset that contains all possible pathologies, nor do we assume that every clinically relevant abnormality can be detected from a single ultrasound image. Functional or temporal abnormalities may require video, measurements, or broader clinical context and are therefore outside the scope of this single-image benchmark. Instead, our goal is to provide a broad multi-source ultrasound dataset that supports self-supervised learning of heterogeneous normal ultrasound appearance and enables evaluation on visible structural anomalies where suitable masks are available.

Despite the relevance of this setting, there is currently no large-scale public dataset specifically designed for training and evaluating self-supervised anomaly detection methods across multiple medical ultrasound domains. Existing anomaly detection benchmarks have advanced industrial visual inspection [3], but they do not capture anatomical variability, acquisitiondependent appearance, speckle patterns, and clinical label limitations of ultrasound imaging, while many medical anomaly detection studies rely on private or task-specific datasets. This gap makes it difficult to assess whether methods learn robust ultrasound representations or merely exploit dataset-specific appearance patterns.

To address this limitation, we introduce the SADUSI benchmark, a multi-source ultrasound dataset for training and evaluating self-supervised anomaly detection methods.<sup>1</sup> It combines public ultrasound sources from different anatomical regions, acquisition protocols, and image formats for visible structural anomaly detection in single frames. Our benchmark shows that current methods struggle in this broad setting: reconstructionbased diffusion methods show limited pathology localization, while feature-based methods such as PatchCore perform better but remain limited.

To summarize, the main contributions of this paper are:

• We curate SADUSI, a broad multi-source ultrasound dataset for self-supervised anomaly detection.

• We benchmark representative reconstruction, featurememory, and student-teacher anomaly detection methods.

• We show that source-domain variation can dominate pathology-related variation, limiting current methods in this setting.

## II. RELATED WORK

Public ultrasound datasets have enabled substantial progress in supervised medical image analysis, including fetal measurement and plane recognition [6], [7], thyroid and breast lesion analysis [8]–[10], echocardiography [11], [12], and lung ultrasound for COVID-19 analytics [13]. These datasets are valuable resources for segmentation, classification, measurement, or disease-specific prediction tasks, but they are usually released and evaluated in isolation. Consequently, they do not directly provide a unified benchmark for training self-supervised anomaly detection methods on heterogeneous normal ultrasound appearance and evaluating visible structural anomalies across multiple anatomical domains.

Deep learning in medical imaging often relies on taskspecific labels, architectures, and pretraining strategies. Selfsupervised learning has therefore become increasingly important because it can exploit large unlabeled or weakly labeled medical image collections [5]. For example, [14] study custom pretraining strategies for COVID detection and severity prediction in 3D CT data. Such work highlights the importance of domain-adapted representation learning in medical imaging, but it still addresses a supervised, disease-specific setting rather than broad anomaly detection from normal training data.

Anomaly detection has been studied through classical one-class, probabilistic, reconstruction-based, and deep representation-based formulations [1]. In medical imaging, reconstruction-based approaches such as GAN-based anomaly detection have been used to model normal anatomy and identify deviations [2]. In visual inspection, benchmarks such as MVTec AD have driven feature-memory and student-teacher methods, including PatchCore and EfficientAD [3], [15], [16]. More recently, diffusion-based methods such as AnoDDPM and DeCo-Diff have reformulated anomaly detection as reconstruction or correction of deviations from normality [17], [18]. However, these methods are commonly evaluated either on industrial data or on comparatively narrow medical settings.

SADUSI addresses this gap by combining diverse ultrasound sources into a multi-source benchmark for visible structural anomaly localization.

## III. DATASET

## A. Task Definition and Scope

SADUSI is designed as a multi-source dataset for training and evaluating single-image anomaly detection methods on medical ultrasound images. Its primary purpose is to expose self-supervised models to a broad normal ultrasound distribution. In this work, the term anomaly strictly refers to a visible structural anomaly, i.e., a pathological finding that can be identified in a single ultrasound image. This excludes abnormalities that require temporal, functional, or broader clinical context. For example, a cardiac rhythm disorder would be treated as normal if the corresponding single echocardiography frame does not show a visible structural abnormality.

Let $\mathcal { X } \subseteq \mathbb { R } ^ { h \times w \times c }$ denote the space of ultrasound images. For the structural anomaly detection setting, we represent the annotated dataset as

$$
\mathcal { D } = \{ ( x _ { i } , y _ { i } , m _ { i } , z _ { i } ) \} _ { i = 1 } ^ { N } ,\tag{1}
$$

where $s _ { i } = ( x _ { i } , y _ { i } , m _ { i } , z _ { i } )$ denotes one dataset entry, $x _ { i } \in$ $\mathcal { X }$ is an ultrasound image, $y _ { i } ~ \in ~ \{ 0 , 1 \}$ is an image-level binary structural label, $m _ { i }$ is an optional pixel-level mask, and $z _ { i }$ summarizes source metadata such as anatomy, view, acquisition protocol, and source dataset. The label $y _ { i } ~ = ~ 0$ denotes, according to the available source labels and metadata, an image that is considered visually normal, while $y _ { i } = 1$ denotes an image with a visible structural anomaly. Examples of structural anomalies include nodules, tumors, deposits, or other localized morphology changes that are visible in the image itself. The corresponding normal and anomalous subsets are defined as

$$
\mathcal { D } _ { n } = \{ s _ { i } \in \mathcal { D } \mid y _ { i } = 0 \} , \quad \mathcal { D } _ { a } = \{ s _ { i } \in \mathcal { D } \mid y _ { i } = 1 \} .\tag{2}
$$

If pixel-level annotations are available, the corresponding localization ground truth is given by

$$
\mathcal { G } _ { a } = \{ ( x _ { i } , m _ { i } ) \mid s _ { i } \in \mathcal { D } _ { a } , m _ { i } \neq \emptyset \} .\tag{3}
$$

For datasets without pixel-level annotations, $m _ { i }$ is undefined. Such datasets could support image-level anomaly detection based on $y _ { i } ,$ but they are not used for the segmentation benchmark in this work.

## B. Source Pool and Harmonization

SADUSI aggregates public ultrasound image datasets from multiple anatomical regions, acquisition protocols, and original data formats. Let $\mathcal { S } \mathit { \Pi } = \{ S _ { k } \} _ { k = 1 } ^ { \bar { K } }$ denote the set of source datasets, where each source contributes a subset $\mathcal { D } ^ { ( k ) }$ and the converted pool is $\begin{array} { r } { \mathcal { D } = \bigcup _ { k = 1 } ^ { K } \mathcal { D } ^ { ( k ) } } \end{array}$ . The dataset is designed to cover variation factors that affect ultrasound imaging in practice, including anatomy, view, acquisition protocol, device characteristics, operator-dependent appearance, resolution, field of view, and image quality. This heterogeneity is central to the intended self-supervised setting where a model trained on SADUSI should not only learn normality for one specific anatomy, but should be exposed to the diversity of normal ultrasound appearance across many regions and acquisition settings. By combining diverse sources, SADUSI aims to support the development and evaluation of anomaly detection methods that generalize beyond very narrow-domain anatomy-specific benchmarks.

All source datasets are converted into a unified imagebased representation. Raw source items are mapped to one or more 2D images: still-image datasets are imported directly, videos, 3D ultrasound sources, and cine clips are converted by extracting every fifth frame. This sampling strategy reduces the strong temporal redundancy of ultrasound sequences while preserving representative variations in probe motion, anatomy, and image appearance.

After conversion, each image is manually inspected and harmonized to reduce source-specific artifacts that are unrelated to the ultrasound content. In particular, black borders are cropped where possible, and visible device user interfaces, mouse cursors, measurement overlays, or other non-anatomical markings are removed if they appear in the image area. In Section $\operatorname { I I I - A } , x _ { i }$ refers to the harmonized image. This step is important because anomaly detection methods may otherwise learn shortcuts from dataset-specific layouts or annotations instead of modeling anatomical image content. During conversion, source-specific metadata and labels are preserved where available and linked to the corresponding extracted images.

## C. Filtered Normal Subset

The larger converted normal pool $\mathcal { D } _ { n }$ may contain implicit anomalies. Many public ultrasound datasets are released for tasks such as organ segmentation, plane classification, or measurement, and their documentation does not always state whether visible pathologies are absent. If such images are used as normal training data, a self-supervised model may learn pathological structures as part of the normal distribution and lose sensitivity to the masked pathologies used for evaluation. We therefore define a filtered normal subset $\mathcal { D } _ { n } ^ { \mathrm { f } } \subseteq \mathcal { D } _ { n }$ as a conservative alternative to the full converted normal pool. It only contains normal training sources for which the source labels, metadata, or dataset description explicitly support the absence of visible structural anomalies. Sources with explicitly labeled structural abnormalities remain part of the anomalous pool $\mathcal { D } _ { a }$ . No separate filtered anomalous subset is introduced. This filtering cannot guarantee that every normal image is clinically healthy, since visible pathologies may be missed during clinical assessment or annotation. Thus, normality is defined operationally from the available annotations and documentation. We report results for both, the full converted normal pool $\mathcal { D } _ { n }$ and the filtered normal subset $\mathcal { D } _ { n } ^ { \mathrm { f } }$ : the former reflects the amount of visually plausible normal ultrasound data available from public sources, while the latter is the conservative training pool used when normal status must be explicitly supported. The filtered normal subset is defined as $\mathcal { D } _ { n } ^ { \mathrm { f } } = \{ s _ { i } \in \mathcal { D } _ { n } \ |$ normal status is explicitly supported}. (4)

TABLE I SOURCE DATASETS INCLUDED IN SADUSI.
<table><tr><td>Source</td><td>Anatomy</td><td>Type</td><td> $\vert \mathcal { D } _ { n } \vert / \vert \mathcal { D } _ { n } ^ { \mathrm { f } } \vert / \vert \mathcal { D } _ { a } \vert$ </td><td>Use</td></tr><tr><td>AbdomenUS [19]</td><td>Abdominal</td><td>Images</td><td>616 / 616 / 0</td><td>Train</td></tr><tr><td>AUL [20]</td><td>Liver</td><td>Images</td><td>100 / 100 / 434</td><td>Train / Test</td></tr><tr><td>Breast-Lesions-USG [21], [22]</td><td>Breast</td><td>Images</td><td>4 / 4 / 0</td><td>Train</td></tr><tr><td>BUS-UCLM [10], [23]</td><td>Breast</td><td>Images</td><td>405 /  405 / 0</td><td>Train</td></tr><tr><td>BUSI [9], [24]</td><td>Breast</td><td>Images</td><td>131  /  131 / 512</td><td>Train / Test</td></tr><tr><td>COVID-19 [25]</td><td>Lung</td><td>Images</td><td>62 / 62 / 0</td><td>Train</td></tr><tr><td>EchoNetPediatric [26], [27]</td><td>Echocardiography</td><td>Videos</td><td>159,006 / 0 / 0</td><td>Train</td></tr><tr><td>FALLMUD [28]</td><td>Muscle</td><td>Images</td><td>813 / 0 / 0</td><td>Train</td></tr><tr><td>FASS [29]</td><td>Fetal abdominal</td><td>Images</td><td>1,577 /  1,577 / 0</td><td>Train</td></tr><tr><td>Fast-U-Net [30]</td><td>Fetal</td><td>Images</td><td>79 / / 0</td><td>Train</td></tr><tr><td>HC18 [6], [31]</td><td>Fetal head</td><td>Images</td><td>1,333 /  1,333 / 0</td><td>Train</td></tr><tr><td>NFLD [32]</td><td>Liver</td><td>Images</td><td>11 / 11 / 0</td><td>Train</td></tr><tr><td>SegThy [33]</td><td>Thyroid</td><td>3D volumes</td><td>1,962 /  1,962 /  0</td><td>Train</td></tr><tr><td>STMUS_NDA [34], [35]</td><td>Muscle</td><td>Images</td><td>6,749 / 0 / 0</td><td>Train</td></tr><tr><td>TG3K [36]</td><td>Thyroid</td><td>Images</td><td>3,585 / 0 / 0</td><td>Train</td></tr><tr><td>TN3K [8]</td><td>Thyroid</td><td>Images</td><td>0 / /  1,349</td><td>Test</td></tr><tr><td>UBPD [37]</td><td>Brachial plexus</td><td>Images</td><td>955 /  0 /  0</td><td>Train</td></tr><tr><td>Ultrasound-Fetus [38]</td><td>Fetal</td><td>Images</td><td>335 /  335 / 0</td><td>Train</td></tr><tr><td>Ultrasound-OpticNerveSheaths [39], [40]</td><td>Optic nerve</td><td>Images</td><td>464 / /  0</td><td>Train</td></tr><tr><td>Total</td><td>Multi-anatomy</td><td>Mixed</td><td>178,187 / 6,536 / 2,295</td><td></td></tr></table>

Table I documents the converted source pool and its contribution to the filtered normal subset.

The filtered normal subset combines fetal, thyroid and neck, liver, abdominal, breast, and lung ultrasound images. The anomalous evaluation pool currently includes visible structural findings from liver, breast, and thyroid ultrasound datasets, including focal liver masses, breast pathologies, and thyroid nodules. The current benchmark uses only anomalous datasets with segmentation masks and evaluates pixel-level anomaly localization. Datasets without masks are not used for the current evaluation.

## IV. BENCHMARK SETUP

We benchmark SADUSI on approaches for pixel-level anomaly segmentation. The benchmark evaluates whether methods trained on a heterogeneous set of ultrasound images considered visually normal can localize visible structural anomalies in held-out anomalous images. Our benchmark includes four representative self-supervised anomaly detection methods: DeCo-Diff [18], AnoDDPM [17], PatchCore [15], and EfficientAD [16]. DeCo-Diff and AnoDDPM represent diffusion-based reconstruction methods, PatchCore represents feature-embedding methods based on nearest-neighbor matching in a memory bank, and EfficientAD represents efficient student-teacher anomaly detection with an additional autoencoder component. For PatchCore, we evaluate two backbone variants: a WideResNet-50 feature extractor (PatchCore-WRN) based on Wide Residual Networks [41], and a DINOv3 ViT-B/16 feature extractor (PatchCore-DINOv3) [42]. All methods produce pixel-level anomaly maps and can therefore be evaluated on the segmentation masks available in ${ \mathcal { G } } _ { a }$ . A model is trained on

$$
\mathcal { D } _ { \mathrm { t r a i n } } \subseteq \mathcal { D } _ { n }\tag{5}
$$

and learns an anomaly scoring function $s _ { \theta } : \mathcal { X } \to \mathbb { R } ^ { h \times w }$ . At inference time, s<sub>θ</sub>(x) produces a pixel-level anomaly map for an input image x.

## A. Training Protocol

Each method is trained self-supervised on either the full normal pool $\mathcal { D } _ { n }$ or the filtered pool $\mathcal { D } _ { n } ^ { \mathrm { f } }$ , which contains only sources whose visually normal status is explicitly supported. Both variants use the same masked anomalous evaluation set ${ \mathcal { G } } _ { a }$ to isolate the effect of training-pool composition.

All runs use seed 1000 and no validation set for model selection. AnoDDPM, DeCo-Diff, and EfficientAD use 300k/11,341/70k steps with effective batch sizes 1/256/4, respectively. PatchCore performs one batch-8 feature pass and retains 10% of the extracted training patch features after coreset subsampling, while EfficientAD reserves 10% of its pool only to estimate anomaly-map normalization statistics. Since PatchCore-DINOv3 gives the strongest AUROC in the initial comparison, we additionally evaluate an augmented PatchCore-DINOv3 variant, denoted by “(A)”, to test whether mild acquisition-like variability in the support memory improves localization.

## B. Preprocessing

All images and masks are converted to a spatial size of 256× 256 pixels before training and inference. To avoid distorting anatomical structures, images are resized while preserving the original aspect ratio and padded to the target resolution. The same spatial transformation is applied to segmentation masks using nearest-neighbor interpolation.

Apart from this dataset-level harmonization, we keep the method-specific input preprocessing consistent with the respective original setups [15]–[18]. For RGB-based methods, grayscale ultrasound images are converted to three channels by duplicating the single intensity channel. Both PatchCore variants use ImageNet-normalized inputs for their pretrained WRN and DINOv3 feature extractors, and EfficientAD applies the corresponding normalization internally in its teacher, student, and autoencoder networks. The diffusion-based reconstruction methods use their original intensity scaling: DeCo-Diff operates on images normalized to [−1, 1] for the latent diffusion pipeline, while AnoDDPM is trained and evaluated in a single-channel setting. Except for the explicit PatchCore-DINOv3 augmentation experiment, no additional cropping, geometric, intensity, or synthetic augmentations are applied beyond dataset conversion and harmonization. For this variant, PatchCore stores each original training image plus one augmented support view generated by mild ultrasound-preserving transformations: affine perturbations $( \pm 7 ^ { \circ } , \pm 4 \%$ translation, 0.95–1.05 scale), horizontal flips, CLAHE, gamma changes in [0.85, 1.15], brightness/contrast changes up to 0.1, and multiplicative speckle noise with $\sigma = 0 . 0 3$

## C. Evaluation Metrics

The benchmark evaluates pixel-level anomaly localization. For each test image, a method predicts an anomaly map $\hat { m } _ { i } \in [ 0 , 1 ] ^ { H \times W }$ , which is compared against the corresponding binary ground-truth mask $m _ { i } .$ . We report pixel-level AUROC, average precision (AP), AUPRO, maximum F1 score $( F 1 _ { \operatorname* { m a x } } )$ precision, and recall. AUPRO measures the area under the per-region overlap curve and summarizes how well predicted anomaly regions overlap with connected ground-truth anomaly regions across thresholds.

TABLE II  
PIXEL-LEVEL ANOMALY LOCALIZATION RESULTS. $^ { \ast } ( \mathrm { A } ) ^ { \ast }$ DENOTES DATA AUGMENTATION.
<table><tr><td>Eval. set</td><td>Training</td><td>Method</td><td>AUROC</td><td>AP</td><td>AUPRO</td><td>Fl</td><td>Prec.</td><td>Rec.</td></tr><tr><td>TN3K</td><td>Filtered</td><td>AnoDDPM</td><td>0.59</td><td>0.14</td><td>0.17</td><td>0.23</td><td>0.14</td><td>0.82</td></tr><tr><td>TN3K</td><td>Filtered</td><td>DeCo-Diff</td><td>0.66</td><td>0.16</td><td>0.20</td><td>0.26</td><td>0.16</td><td>0.64</td></tr><tr><td>TN3K</td><td>Filtered</td><td>PatchCore-WRN</td><td>0.76</td><td>0.24</td><td>0.40</td><td>0.33</td><td>0.22</td><td>0.62</td></tr><tr><td>TN3K</td><td>Filtered</td><td>PatchCore-DINOv3</td><td>0.79</td><td>0.26</td><td>0.29</td><td>0.36</td><td>0.26</td><td>0.61</td></tr><tr><td>TN3K</td><td>Filtered</td><td>PatchCore-DINOv3 (A)</td><td>0.76</td><td>0.24</td><td>0.37</td><td>0.33</td><td>0.22</td><td>0.62</td></tr><tr><td>TN3K</td><td>Filtered</td><td>EfficientAD</td><td>0.50</td><td>0.11</td><td>0.07</td><td>0.20</td><td>0.11</td><td>0.94</td></tr><tr><td>TN3K</td><td>Full</td><td>AnoDDPM</td><td>0.61</td><td>0.15</td><td>0.19</td><td>0.24</td><td>0.14</td><td>0.84</td></tr><tr><td>TN3K</td><td>Full</td><td>DeCo-Diff</td><td>0.60</td><td>0.13</td><td>0.15</td><td>0.24</td><td>0.14</td><td>0.87</td></tr><tr><td>TN3K</td><td>Full</td><td>PatchCore-WRN</td><td>0.77</td><td>0.26</td><td>0.42</td><td>0.34</td><td>0.25</td><td>0.57</td></tr><tr><td>TN3K</td><td>Full</td><td>PatchCore-DINOv3</td><td>0.82</td><td>0.32</td><td>0.34</td><td>0.40</td><td>0.31</td><td>0.58</td></tr><tr><td>TN3K</td><td>Full</td><td>PatchCore-DINOv3 (A)</td><td>0.78</td><td>0.27</td><td>0.43</td><td>0.35</td><td>0.25</td><td>0.57</td></tr><tr><td>TN3K</td><td>Full</td><td>EfficientAD</td><td>0.53</td><td>0.12</td><td>0.14</td><td>0.21</td><td>0.12</td><td>0.88</td></tr><tr><td>BUSI</td><td>Filtered</td><td>AnoDDPM</td><td>0.62</td><td>0.11</td><td>0.23</td><td>0.18</td><td>0.11</td><td>0.42</td></tr><tr><td>BUSI</td><td>Filtered</td><td>DeCo-Diff</td><td>0.68</td><td>0.14</td><td>0.40</td><td>0.21</td><td>0.15</td><td>0.38</td></tr><tr><td>BUSI</td><td>Filtered</td><td>PatchCore-WRN</td><td>0.82</td><td>0.27</td><td>0.66</td><td>0.33</td><td>0.25</td><td>0.47</td></tr><tr><td>BUSI</td><td>Filtered</td><td>PatchCore-DINOv3</td><td>0.82</td><td>0.23</td><td>0.51</td><td>0.33</td><td>0.24</td><td>0.54</td></tr><tr><td>BUSI</td><td>Filtered</td><td>PatchCore-DINOv3 (A)</td><td>0.81</td><td>0.27</td><td>0.66</td><td>0.33</td><td>0.25</td><td>0.47</td></tr><tr><td>BUSI</td><td>Filtered</td><td>EfficientAD</td><td>0.50</td><td>0.08</td><td>0.30</td><td>0.15</td><td>0.08</td><td>0.92</td></tr><tr><td>BUSI</td><td>Full</td><td>AnoDDPM</td><td>0.63</td><td>0.11</td><td>0.24</td><td>0.18</td><td>0.11</td><td>0.55</td></tr><tr><td>BUSI</td><td>Full</td><td>DeCo-Diff</td><td>0.56</td><td>0.09</td><td>0.14</td><td>0.17</td><td>0.09</td><td>0.91</td></tr><tr><td>BUSI</td><td>Full</td><td>PatchCore-WRN</td><td>0.80</td><td>0.23</td><td>0.64</td><td>0.30</td><td>0.22</td><td>0.46</td></tr><tr><td>BUSI</td><td>Full</td><td>PatchCore-DINOv3</td><td>0.82</td><td>0.22</td><td>0.51</td><td>0.33</td><td>0.24</td><td>0.56</td></tr><tr><td>BUSI</td><td>Full</td><td>PatchCore-DINOv3 (A)</td><td>0.79</td><td>0.24</td><td>0.64</td><td>0.30</td><td>0.23</td><td>0.46</td></tr><tr><td>BUSI</td><td>Full</td><td>EfficientAD</td><td>0.55</td><td>0.09</td><td>0.32</td><td>0.16</td><td>0.09</td><td>0.78</td></tr><tr><td>AUL</td><td>Filtered</td><td>AnoDDPM</td><td>0.72</td><td>0.06</td><td>0.26</td><td>0.13</td><td>0.07</td><td>0.64</td></tr><tr><td>AUL</td><td>Filtered</td><td>DeCo-Diff</td><td>0.66</td><td>0.05</td><td>0.19</td><td>0.10</td><td>0.05</td><td>0.88</td></tr><tr><td>AUL</td><td>Filtered</td><td>PatchCore-WRN</td><td>0.79</td><td>0.07</td><td>0.41</td><td>0.16</td><td>0.09</td><td>0.82</td></tr><tr><td>AUL</td><td>Filtered</td><td>PatchCore-DINOv3</td><td>0.77</td><td>0.09</td><td>0.36</td><td>0.14</td><td>0.09</td><td>0.31</td></tr><tr><td>AUL</td><td>Filtered</td><td>PatchCore-DINOv3 (A)</td><td>0.79</td><td>0.08</td><td>0.39</td><td>0.16</td><td>0.09</td><td>0.82</td></tr><tr><td>AUL</td><td>Filtered</td><td>EfficientAD</td><td>0.34</td><td>0.03</td><td>0.07</td><td>0.07</td><td>0.04</td><td>1.00</td></tr><tr><td>AUL</td><td>Full</td><td>AnoDDPM</td><td>0.72</td><td>0.07</td><td>0.27</td><td>0.13</td><td>0.07</td><td>0.64</td></tr><tr><td>AUL</td><td>Full</td><td>DeCo-Diff</td><td>0.71</td><td>0.05</td><td>0.23</td><td>0.12</td><td>0.06</td><td>0.91</td></tr><tr><td>AUL</td><td>Full</td><td>PatchCore-WRN</td><td>0.80</td><td>0.08</td><td>0.42</td><td>0.16</td><td>0.09</td><td>0.78</td></tr><tr><td>AUL</td><td>Full</td><td>PatchCore-DINOv3</td><td>0.83</td><td>0.11</td><td>0.45</td><td>0.18</td><td>0.11</td><td>0.47</td></tr><tr><td>AUL</td><td>Full</td><td>PatchCore-DINOv3 (A)</td><td>0.81</td><td>0.08</td><td>0.45</td><td>0.17</td><td>0.09</td><td>0.77</td></tr><tr><td>AUL</td><td>Full</td><td>EfficientAD</td><td>0.49</td><td>0.03</td><td>0.10</td><td>0.08</td><td>0.04</td><td>0.85</td></tr></table>

Threshold-independent metrics are computed directly from the continuous anomaly maps. For threshold-dependent metrics such as precision, recall, and F1, we sweep thresholds over the anomaly maps and select the threshold $\tau ^ { * } =$ arg max<sub>τ</sub> $F 1 ( \tau )$ to compute their values. Thus, $F 1 _ { \mathrm { m a x } } ,$ precision, and recall measure the best achievable operating point on the evaluated segmentation masks, while AUROC, AP, and AUPRO summarize ranking quality without fixing a threshold.

## V. EXPERIMENTS AND RESULTS

In Table II, we report pixel-level anomaly localization results on three anomalous ultrasound subsets with segmentation masks for thyroid nodule, breast, and liver ultrasound evaluations.

Overall, the PatchCore variants achieve the strongest localization results, with best AUROC values of 0.82 on TN3K and BUSI and 0.83 on AUL, but the corresponding F1<sub>max</sub> values of 0.40, 0.33, and 0.18 show that pixel-level delineation remains difficult. PatchCore-WRN often yields competitive or higher AUPRO values, while augmented PatchCore-DINOv3 improves AUPRO in several settings, most clearly on TN3K and BUSI, but not the best AUROC or $F 1 _ { \mathrm { m a x } }$ values. AnoDDPM and DeCo-Diff generally yield lower AUROC and $F 1 _ { \mathrm { m a x } }$ values with high recall but low precision, and EfficientAD performs close to or below random ranking on several evaluations. Neither training pool is uniformly superior.

## A. Anomaly-Map Contrast Analysis

The pixel-level metrics in Table II summarize thresholded localization performance and ranking quality. To further inspect the produced anomaly maps, we measure whether each method assigns higher anomaly scores inside the annotated structural abnormality than in the remaining image background. For an anomalous image $x _ { i }$ with anomaly map $A _ { i }$ and binary ground-truth mask $M _ { i } .$ , we compute the anomalybackground contrast

$$
\Delta _ { i } = \frac { 1 } { \vert M _ { i } \vert } \sum _ { p \in M _ { i } } A _ { i } ( p ) - \frac { 1 } { \vert \Omega _ { i } \setminus M _ { i } \vert } \sum _ { p \in \Omega _ { i } \setminus M _ { i } } A _ { i } ( p ) ,\tag{6}
$$

where $\Omega _ { i }$ denotes the image domain. Positive $\Delta _ { i }$ -values indicate that the annotated pathology receives higher anomaly scores than the surrounding background. Because anomalymap scales differ across methods, we robustly normalize the contrast values separately for each model variant and anomalymap type. Let $\hat { q } _ { 0 . 0 2 5 } ^ { ( m ) }$ and $q _ { 0 . 9 7 5 } ^ { ( m ) }$ denote the corresponding quantiles over all contrast values of method m. We compute

$$
\widetilde { \Delta } _ { i } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \frac { \Delta _ { i } - q _ { 0 . 0 2 5 } ^ { ( m ) } } { q _ { 0 . 9 7 5 } ^ { ( m ) } - q _ { 0 . 0 2 5 } ^ { ( m ) } } \right) .\tag{7}
$$

Thus, the central 95% of each method’s contrast range defines the linear scaling interval, while values outside it are retained and clipped to 0 or 1.

Figure 1 compares these scaled contrast distributions by evaluation dataset and training pool. PatchCore-DINOv3 and PatchCore-WRN show relatively high scaled contrasts on BUSI, while PatchCore-DINOv3 also performs strongly on TN3K. On AUL, however, PatchCore-DINOv3 shifts toward the lower end of its method-specific range, whereas PatchCore-WRN shows stronger filtered-run contrast. AnoD-DPM changes little between the training pools, while DeCo-Diff trained on the full pool produces particularly narrow lowvalued distributions across all datasets. The effect of filtering is therefore model- and dataset-dependent rather than uniformly beneficial. Since each method is scaled independently, the figure supports comparison of relative distribution shapes and training-set effects, but not absolute raw contrast magnitudes between methods.

## B. Feature-Space Analysis

To better understand the limited localization performance, we analyze both normal training pools in a pretrained DINOv3 ViT-B/16 embedding space [42]. Images and ground-truth masks are transformed to $2 5 6 \times 2 5 6$ pixels using aspectratio preserving resizing and zero padding, with bilinear and nearest-neighbor interpolation, respectively. DINOv3 produces a feature tensor in $\bar { \mathbb { R } } ^ { 7 6 8 \times 1 6 \times 1 6 }$ . Only the patch tokens are retained, excluding class tokens. Each grid position is treated as one 768-dimensional patch embedding and normalized individually using $\ell _ { 2 }$ normalization. The corresponding mask is reduced to the same $1 6 \times 1 6$ grid by average pooling, yielding the fraction of pathological pixels within each patch region. A patch is labeled anomalous if this fraction exceeds 0.5 and normal otherwise. Consequently, normal patches may originate from normal images or regions outside the annotated pathology in anomalous images. Distances are computed as Euclidean distances between normalized patch embeddings.

For each training pool, we sample up to 500 balanced pairs per dataset pair for normal–normal and normal–anomalous comparisons, separated into within- and cross-dataset cases. We further query each anomalous patch exhaustively against all normal patches from other images using rank-1 nearestneighbor search and analyze its $k = 1 0$ neighborhood among all patches, excluding the query image to prevent spatially adjacent regions of the same pathology from dominating. This analysis is not used for training. It characterizes local dataset structure and whether filtering simplifies the task.

Figure 2 summarizes the patchwise feature-space results. Cross-dataset normal–normal distances remain larger than normal–anomalous distances: their means are 0.805 versus 0.770 for the filtered pool and 0.802 versus 0.761 for the full pool. The same pattern occurs within datasets, where normal– normal and normal–anomalous means are 0.738 and 0.710 for the filtered pool, and 0.728 and 0.717 for the full pool. For AUL, BUSI, and TN3K, the mean cross-image rank-1 distances to normal patches are 0.265/0.265, 0.367/0.365, and $0 . 4 1 4 / 0 . 4 0 6$ for the filtered/full pools, showing only marginally closer matches in the full pool. The corresponding mean normal-neighbor fractions for $k = 1 0$ are 0.709/0.722, 0.419/0.461, and 0.415/0.540. Thus, AUL anomalies mostly overlap normal neighborhoods, whereas BUSI and filtered TN3K more often neighbor anomalous patches across images. The larger cross-dataset distances indicate substantial source-domain structure. The normal-neighbor fractions differ between the training pools, most notably for TN3K. The distributions in Figure 3 have median annotated pathology areas of only 2.4%, 6.0%, and 7.5% for AUL, BUSI, and TN3K, respectively.

Thus, pathology-related neighborhoods exist for some anatomies, but visible pathology does not form a consistent out-of-distribution signal across datasets in the patch-token space. The small annotated regions further illustrate that models must identify localized pathology despite substantial normal anatomy and source variation.

## VI. DISCUSSION

Our results show that broad multi-source ultrasound anomaly detection remains challenging: PatchCore variants perform strongest, reaching AUROC values up to 0.82, 0.82, and 0.83 on TN3K, BUSI, and AUL, but their $F 1 _ { \mathrm { m a x } }$ values remain limited at 0.40, 0.33, and 0.18. Filtering sources with insufficiently documented pathology status improves dataset semantics but does not make the task uniformly easy.

The reconstruction-based diffusion methods DeCo-Diff and AnoDDPM show limited localization performance, possibly because the heterogeneous training distribution encourages a general ultrasound prior that reconstructs plausible anomalous structures together with normal anatomy.

PatchCore performs better than the reconstruction-based methods, suggesting that feature-space nearest-neighbor comparison may be more robust for this setting. The DINOv3 backbone gives the highest AUROC on TN3K and AUL, while the WRN backbone gives the strongest AUPRO on TN3K and BUSI. We therefore choose PatchCore-DINOv3 for the augmentation experiment because it combines strong ranking behavior with a pretrained patch-token representation that may benefit from additional acquisition-like variability in the memory bank. The augmented variant improves region overlap in several cases, but its lack of consistent AUROC or $F 1 _ { \mathrm { m a x } }$ gains suggests that mild appearance and geometric perturbations alone do not solve the separation of normal variation and pathology. Even with stronger pretrained visual features, absolute localization quality remains limited, indicating that normal ultrasound structures and pathological findings can share local texture, speckle statistics, and boundary appearance; this may also contribute to the weak EfficientAD results. The patchwise DINOv3 analysis supports this interpretation: BUSI and TN3K show lower normal-neighbor fractions than AUL, while larger cross-dataset distances indicate that feature geometry reflects both pathology and source domain without producing consistent pathology outliers.

![](images/12b3a0798d4cbab3bad0b5637a3b74a6b9a2e4f370d3268e948278468410d7d3.jpg)

![](images/6d7e56bf993669076f94f3369467b31cd4061bf9e40a330c3c3b007c4cb9d9bd.jpg)

![](images/0acec97023c463b4e1f037bece14b49135d305ba6fccaa4323847737c5cb04dd.jpg)

Fig. 1. Scaled pathology-background anomaly-map contrast by evaluation dataset, model, and training pool. Violin plots show the distributions and embedded box plots show their medians and interquartile ranges.  
![](images/a22f72a62f2a52324c1903968ad843f865bc8fa0ec4de6f4711a6a8cb976458b.jpg)

![](images/9266b4513546bb762b700ece9840e7f897df27da7ef5072fdfe6c15bd4c99971.jpg)

![](images/a4f5022405d48772c8f1e51c8cb0a697176aa0d3edd48837d8d873d72b32fff3.jpg)  
Fig. 2. Patchwise DINOv3 feature-space analysis. From left to right: balanced normal–normal and normal–anomalous patch distances; cross-image rank-1 distance from each anomalous patch to its nearest normal patch; and fraction of normal patches in its cross-image k = 10 neighborhood.

![](images/4a0f52fce1791d0dcae97b7f7056db70e3000045a971d0a98e936b1324020a97.jpg)  
Fig. 3. Distribution of the image area covered by annotated pathology in the three evaluation datasets.

SADUSI should therefore be understood not only as a dataset for training, but also as a stress test for the generalization ability of self-supervised anomaly detection methods in medical ultrasound. The goal is not to claim that all pathologies can be detected from single images, but to highlight the gap between current methods and the desired ability to model broad normal ultrasound appearance while still assigning high anomaly scores to visible structural findings. Important limitations remain, including incomplete coverage of anatomical regions, uneven source sizes, heterogeneous annotation quality, and the limited separability of normal tissue and visible pathology in the analyzed patch-feature space.

## VII. CONCLUSION

We introduced SADUSI, a multi-source ultrasound dataset for self-supervised anomaly detection across diverse anatomical regions and acquisition settings. Our experiments show that current methods struggle to separate visible structural anomalies from heterogeneous normal ultrasound appearance. Feature-memory methods outperform reconstructionbased diffusion methods, but even the strongest PatchCore variants remain limited in pixel-level localization. The patchwise feature-space analysis further suggests that visible pathology is often not cleanly separated from normal tissue and source-domain variation in pretrained representations. These findings motivate methods that learn broad ultrasound normality without losing sensitivity to localized structural abnormalities.

[1] L. Ruff, J. R. Kauffmann, R. A. Vandermeulen, G. Montavon, W. Samek, M. Kloft, T. G. Dietterich, and K.-R. Muller, “A unifying review of deep and shallow anomaly detection,” Proceedings of the IEEE, vol. 109, no. 5, 2021.

[2] T. Schlegl, P. Seebock, S. M. Waldstein, U. Schmidt-Erfurth, and¨ G. Langs, “Unsupervised anomaly detection with generative adversarial networks to guide marker discovery,” in Information Processing in Medical Imaging, 2017.

[3] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTec AD – a comprehensive real-world dataset for unsupervised anomaly detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[4] B. Wang, P. He, Y. Kang, J. Jia, X. Liu, and N. Li, “Ultrasonic testing of carbon fiber-reinforced polymer composites,” Journal of Sensors, vol. 2022, 2022.

[5] S.-C. Huang, A. Pareek, M. Jensen, M. P. Lungren, S. Yeung, and A. S. Chaudhari, “Self-supervised learning for medical image classification: A systematic review and implementation guidelines,” npj Digital Medicine, vol. 6, no. 1, 2023.

[6] T. L. A. van den Heuvel, D. de Bruijn, C. L. de Korte, and B. van Ginneken, “Automated measurement of fetal head circumference using 2d ultrasound images,” PLOS ONE, vol. 13, no. 8, 2018.

[7] X. P. Burgos-Artizzu, D. Coronado-Gutierrez, B. Valenzuela-Alcaraz, E. Bonet-Carne, E. Eixarch, F. Crispi, and E. Gratacos, “Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes,” Scientific Reports, vol. 10, no. 10200, 2020.

[8] H. Gong, J. Chen, G. Chen, H. Li, G. Li, and F. Chen, “Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules,” Computers in Biology and Medicine, vol. 155, 2023.

[9] W. Al-Dhabyani, M. Gomaa, H. Khaled, and A. Fahmy, “Dataset of breast ultrasound images,” Data in Brief, vol. 28, 2020.

[10] N. Vallez, G. Bueno, O. Deniz, M. A. Rienda, and C. Pastor, “Bus-uclm: Breast ultrasound lesion segmentation dataset,” Data in Brief, vol. 58, 2025.

[11] D. Ouyang, B. He, A. Ghorbani, N. Yuan, J. Ebinger, C. P. Langlotz, P. A. Heidenreich, R. A. Harrington, D. H. Liang, E. A. Ashley, and J. Y. Zou, “EchoNet-Dynamic: A large new cardiac motion video data resource for medical machine learning,” Nature, vol. 12, 2020.

[12] S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, F. Cervenansky, F. Espinosa, T. Espeland, E. A. R. Berg, P.-M. Jodoin, T. Grenier, C. Lartizien, J. Dhooge, L. Lovstakken, and O. Bernard, “Deep learning for segmentation using an open large-scale dataset in 2d echocardiography,” IEEE Transactions on Medical Imaging, vol. 38, no. 9, 2019.

[13] A. Ebadi, P. Xi, A. MacLean, A. Florea, S. Tremblay, S. Kohli, and A. Wong, “COVIDx-US: An open-access benchmark dataset of ultrasound imaging data for ai-driven covid-19 analytics,” Frontiers in Bioscience-Landmark, vol. 27, no. 7, 2022.

[14] D. Kienzle, J. Lorenz, R. Schon, K. Ludwig, and R. Lienhart, “COVID¨ detection and severity prediction with 3D-ConvNeXt and custom pretrainings,” 2022.

[15] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler, “Towards total recall in industrial anomaly detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[16] K. Batzner, L. Heckler, and R. Konig, “EfficientAD: Accurate visual¨ anomaly detection at millisecond-level latencies,” in IEEE/CVF Winter Conference on Applications of Computer Vision, 2024.

[17] J. Wyatt, A. Leach, S. M. Schmon, and C. G. Willcocks, “AnoDDPM: Anomaly detection with denoising diffusion probabilistic models using simplex noise,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2022.

[18] F. Beizaee, G. A. Lodygensky, C. Desrosiers, and J. Dolz, “Correcting deviations from normality: A reformulated diffusion model for multiclass unsupervised anomaly detection,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

[19] J. I. Orlando, “USSimSegm: Abdominal Ultrasound Image Segmentation Dataset,” 2021, accessed: 2026-04-29. [Online]. Available: https: //github.com/ignaciorlando/ussimsegm

[20] Annotated Ultrasound Liver Dataset Authors, “Annotated Ultrasound Liver Images,” 2022, zenodo; Accessed: 2026-04-29. [Online]. Available: https://zenodo.org/records/7272660

[21] A. Pawlowska, A. Cwierz-Pienkowska, A. Domalik, D. Jagus, P. Kasprzak, R. Matkowski, L. Fura, A. Nowicki, and N. Zolek, “Curated benchmark dataset for ultrasound-based breast lesion analysis,” Scientific Data, 2024.

[22] The Cancer Imaging Archive (TCIA), “Breast-Lesions-USG,” 2024, accessed: 2026-04-30. [Online]. Available: https://www. cancerimagingarchive.net/collection/breast-lesions-usg/

[23] N. Vallez, G. Bueno, O. Deniz, M. A. Rienda, and C. Pastor, “BUS-UCLM: Breast Ultrasound Lesion Segmentation Dataset,” 2024, mendeley Data, V1; Accessed: 2026-04-29. [Online]. Available: https://data.mendeley.com/datasets/7fvgj4jsp7/1

[24] BUSI Dataset Contributors, “Breast Ultrasound Images Dataset,” 2020, accessed: 2026-04-29. [Online]. Available: https://www.kaggle.com/ datasets/aryashah2k/breast-ultrasound-images-dataset

[25] A. Sedik, “COVID-19 Dataset,” 2020, mendeley Data, V1; Accessed: 2026-04-29. [Online]. Available: https://data.mendeley.com/datasets/ 6rs5mnvktk/1

[26] C. D. Reddy, L. Lopez, D. Ouyang, J. Y. Zou, and B. He, “Videobased deep learning for automated assessment of left ventricular ejection fraction in pediatric patients,” Journal of the American Society of Echocardiography, vol. 36, no. 5, 2023.

[27] Stanford University, AIMI, “EchoNet-Pediatric Dataset,” 2023, accessed: 2026-04-29. [Online]. Available: https://echonet.github.io/pediatric/

[28] R. Cunningham and N. Cronin, “FALLMUD: Fascicle Lower Leg Muscle Ultrasound Dataset,” 2020, accessed: 2026-04-29. [Online]. Available: https://kalisteo.cea.fr/index.php/fallmud/

[29] L. O. Santos, “Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images,” 2023, accessed: 2026-04-29. [Online]. Available: https://data.mendeley.com/datasets/4gcpm9dsc3/1

[30] V. Ashkani Chenarlogh, M. Ghelich Oghli, A. Shabanzadeh, N. Sirjani, A. Akhavan, I. Shiri, H. Arabi, M. Sanei Taheri, and M. K. Tarzamni, “Fast and accurate u-net model for fetal ultrasound image segmentation,” Ultrasonic imaging, vol. 44, no. 1, 2022.

[31] T. L. A. van den Heuvel, D. de Bruijn, C. L. de Korte, and B. van Ginneken, “Automated Measurement of Fetal Head Circumference Using 2D Ultrasound Images (HC18 Dataset),” 2018, accessed: 2026-04-29. [Online]. Available: https://hc18.grand-challenge.org/

[32] M. Hoque, G. Saha, F. Abdullah Al, T. A. Joy, N. Labib, M. J. Riyad, and T. T. Aurpa, “Dataset on NAFLD Severity Classification with Ultrasound Liver Image & Clinical Data,” 2026, accessed: 2026-04-29. [Online]. Available: https://data.mendeley.com/datasets/6rg4hk6728/3

[33] Chair for Computer Aided Medical Procedures (CAMP), Technical University of Munich, “SegTHy Dataset,” 2023, accessed: 2026-04- 29. [Online]. Available: https://www.cs.cit.tum.de/camp/publications/ segthy-dataset/

[34] F. Marzola, N. van Alfen, and S. Pillen, “Muscle ultrasound segmentation using deep learning,” Ultrasound in Medicine & Biology, vol. 47, no. 8, 2021.

[35] ——, “Transverse Musculoskeletal Ultrasound Images Dataset for CSA Segmentation,” 2021, accessed: 2026-04-29. [Online]. Available: https://data.mendeley.com/datasets/3jykz7wz8d/1

[36] H. Gong, J. Chen, G. Chen, H. Li, G. Li, and F. Chen, “Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules,” Computers in Biology and Medicine, vol. 155, 2023.

[37] Y. Ding, Q. Yang, Y. Wang, D. Chen, Z. Qin, and J. Zhang, “Mallesnet: A multi-object assistance based network for brachial plexus segmentation in ultrasound images,” Medical Image Analysis, vol. 80, 2022.

[38] A. Anitha, “Ultrasound Fetus Dataset,” 2024, accessed: 2026-04-29. [Online]. Available: https://data.mendeley.com/datasets/yrzzw9m6kk/2

[39] F. Marzola, P. Lochner, A. Naldi, R. Lemor, J. Stogbauer, and K. M. Meiburger, “Dataset for optic nerve characterization in transorbital ultrasound images,” Data in Brief, vol. 49, 2023.

[40] F. Marzola, “Dataset for Optic Nerve Characterization in Transorbital Ultrasound Images,” 2023, mendeley Data, V2; Accessed: 2026-04-29. [Online]. Available: https://data.mendeley.com/datasets/kw8gvp8m8x/2

[41] S. Zagoruyko and N. Komodakis, “Wide residual networks,” in British Machine Vision Conference, 2016. [Online]. Available: https: //arxiv.org/abs/1605.07146

[42] O. Simeoni´ et al., “DINOv3,” arXiv preprint arXiv:2508.10104, 2025.