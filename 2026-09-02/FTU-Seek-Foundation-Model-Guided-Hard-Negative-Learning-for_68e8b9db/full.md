# FTU-Seek: Foundation Model-Guided Hard-Negative Learning for Sparse Functional Tissue Unit Segmentation

Zonghao Liu<sup>1,2†</sup>, Lei Su<sup>3,4†</sup>, Jiguang Yu<sup>5†</sup>, Xuqing Geng<sup>3,4</sup>, Louis Shuo Wang<sup>6\*</sup>, Jianmin Wang<sup>2,7</sup>, Jingfeng Liu<sup>1,7,8\*</sup>

<sup>1</sup>Department of Hepatopancreatobiliary Surgery, Clinical Oncology School, Fujian Medical University, Fuzhou, 350014, China .

<sup>2</sup>Innovation Center for Cancer Research, Clinical Oncology School, Fujian Medical University, Fuzhou, 350014, China .

<sup>3</sup>Institute of Automation, Chinese Academy of Sciences, Beijing, 100190, China .

<sup>4</sup>University of Chinese Academy of Sciences, Beijing, 100190, China . <sup>5</sup>College of Engineering, Boston University, Boston, MA, 02215, USA . <sup>6</sup>Department of Mathematics, Northeastern University, Boston, MA, 02115, USA .

<sup>7</sup>Fujian Key Laboratory of Advanced Technology for Cancer Screening and Early Diagnosis, Fujian Cancer Hospital, Fuzhou, 350014, China . <sup>8</sup>Key Laboratory of Cancer Metabolism, National Health Commission of the People’s Republic of China, Fuzhou, 350014, China .

\*Corresponding author(s). E-mail(s): wang.s41@northeastern.edu; drjingfeng@126.com;

Contributing authors: liuzonghao42@gmail.com; sulei2023@ia.ac.cn;   
jyu678@bu.edu; gengxuqing2024@ia.ac.cn; wangjm8605@163.com; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Background/Objectives: Functional tissue units (FTUs), including tertiary lymphoid structures (TLSs), blood vessels, and glands, encode localized immune, vascular, and epithelial organization in histopathology. Accurate quantification of these structures is important for studying tissue architecture and disease-associated tissue organization. However, FTUs are frequently sparse,

heterogeneous, and surrounded by large amounts of morphologically similar background tissue, making automated segmentation in whole-slide images (WSIs) challenging. We therefore developed FTU-Seek, a pathology foundation model-guided framework that treats morphology-aware negative-patch selection as a key component of sparse FTU segmentation. Methods: FTU-Seek uses frozen multi-depth features from the UNI pathology foundation model to train a patch-level classifier that distinguishes FTU-containing from FTU-absent tissue. Target-absent patches are subsequently ranked according to their predicted target-containing probabilities, and the highest-scoring hard negatives are selected through a static TopK strategy to construct compact segmentation training sets. The framework was evaluated using five-fold cross-validation and internal test cohorts across TLS, blood-vessel, and gland segmentation tasks, with an additional independent 30-WSI held-out cohort for TLS. Positive-only, alltissue, random-negative, and matched random TopK sampling strategies served as comparators. Segmentation-derived phenotypes were further explored in external TCGA cohorts. Results: The patch-level classifiers achieved mean validation AUCs of 95.92%, 90.11%, and 98.16% for TLS, blood vessel, and gland classification, respectively. For TLS segmentation, the pre-specified Top1000 configuration retained 27.6% of the all-tissue training workload and achieved a slide-level Dice of 76.69 ± 11.89% on the independent 30-WSI held-out cohort. Compared with matched random Top1000 sampling, it improved Dice by 3.72 percentage points (95% CI, 2.05–5.38). Blood vessel and gland segmentation achieved performance approaching all-tissue training while reducing the retained training workload by approximately one-half and one-third, respectively. Compared with matched random sampling, classifier-guided hard-negative selection produced the greatest improvements for sparse and morphologically ambiguous FTUs. Exploratory TCGA analyses further showed associations of TLS phenotypes with overall survival, vascular phenotypes with overall survival and microvascular invasion, and glandular phenotypes with clinicopathological characteristics. Conclusions: FTU-Seek demonstrates that pathology foundation models can support sparse FTU segmentation not only through feature representation but also through morphology-aware construction of segmentation training sets. By prioritizing informative hard negatives, the framework reduces redundant segmentationtraining workload while maintaining competitive segmentation performance and supporting quantitative tissue phenotyping from routine histopathology.

Keywords: absorbing Markov chains; gambler’s ruin; increasing failure rate;   
remaining useful life; preventive intervention; remote patient monitoring.

MSC Classification: 60J10 , 60G40 , 60K10 , 90B25 , 15B48.

## 1 Introduction

Whole-slide images (WSIs) are increasingly used not only for diagnosis, grading, and treatment planning, but also for quantitative analysis of tissue architecture and the tumor microenvironment [1–3]. Beyond large histological compartments, WSIs contain spatially localized microscopic structures that reflect complementary biological processes. Glandular structures encode epithelial diferentiation and architectural organization [4–6], blood vessels reflect angiogenesis and vascular remodeling [7, 8], and tertiary lymphoid structures (TLSs) represent organized antitumor immune activity [9–12]. In this study, we use the term functional tissue units (FTUs) to describe these histologically identifiable multicellular structures whose morphology, abundance, and spatial organization are linked to epithelial, vascular, or immune functions. Accurate FTU segmentation may therefore provide a practical route for extracting interpretable image-derived phenotypes from routine histopathology slides [13–16].

A central challenge in FTU analysis is that these structures are frequently sparse, heterogeneous, and embedded within large amounts of target-absent tissue. As shown in Figure 1, the representative FTUs studied here difer substantially at the instance, patch, and pixel levels. TLSs and blood vessels occupy only a small fraction of tissue patches and total tissue area [17–20], whereas glandular structures are more abundant but still exhibit considerable architectural and morphological variability [21–23]. Their spatial distributions are also highly uneven across prostate cancer, liver cancer, and pancreatic cancer WSIs, as illustrated in Figure 2. Manual delineation by pathologists is possible but labor-intensive, particularly when FTUs are small, fragmented, boundary-ambiguous, or visually confounded by surrounding tissue [24–28]. These characteristics make FTU segmentation a sparse-structure learning problem rather than a conventional segmentation task involving large and spatially continuous regions.

Deep learning-based WSI segmentation has achieved strong performance for relatively large and spatially continuous compartments, such as tumor parenchyma, necrosis, and stroma [29–34]. Sparse FTU segmentation imposes a fundamentally different constraint. Target-containing patches often represent only a small minority of all tissue patches, whereas the substantially larger target-absent pool contains both redundant easy negatives and a limited subset of morphologically target-like regions that can trigger false-positive (FP) predictions [35, 36]. Training on all tissue patches is therefore ineficient because a large proportion of the negative samples contributes little additional information. Random negative sampling reduces the training workload, but may fail to retain the rare background regions that most closely resemble the target and are therefore most important for FP suppression. Accordingly, sparse FTU segmentation is not only a pixel-level prediction problem, but also a WSI-scale training-set construction problem: the model must be exposed to suficiently informative negative tissue without being dominated by redundant background patches.

TLS (pancreatic cancer)  
![](images/733289eb77e5afd5596644090acfd7a9ebc32955e21c299482e795074afcdfad.jpg)

![](images/67dc7573c3b1e99d510dca3637ba2e0864a9dc484b32edff778eda50f05da9b4.jpg)

![](images/ce5e51f5f42932fb09f2de6861733aa09a61a7ed68fddc4ab2d0da744609aa7e.jpg)  
Fig. 1 Quantitative assessment of spatial sparsity across representative functional tissue units (FTUs). Glands, blood vessels (BVs), and tertiary lymphoid structures (TLSs) were compared at the instance, patch, and pixel levels. Instance-level sparsity was assessed using FTU instance counts; the proportion of FTU-containing tissue patches measured patch-level sparsity; and pixel-level sparsity was quantified using the FTU pixel area ratio at a spatial resolution of 1 µm/pixel. This figure reports statistics calculated from the complete cohort by pooling the development and independent test sets.

![](images/f146e12d930c061bbf2ac68bd5194a2de40b6a93c99f8d27cfc3504198928fad.jpg)

![](images/f53308364941dfb4f9720f38755f96bc0927c489d1f4c2bb13fc8a1bc9f1b1f5.jpg)

![](images/b702481cf384ac562ca83076d53680c1ea3e624f1442641863e23bebd15e87c7.jpg)  
Fig. 2 Representative spatial distributions of annotated functional tissue unit (FTU) regions in whole-slide images. Annotated FTU regions are shown in cyan for glandular structures in prostate cancer, blood vessels in liver cancer, and tertiary lymphoid structures in pancreatic cancer.

Existing imbalance-handling strategies only partially address this challenge. Lossbased methods, including class-weighted objectives, Dice-based losses, and focal loss, increase the optimization contribution of under-represented or dificult pixels [37–39]. Sampling-based approaches rebalance the training distribution through oversampling, undersampling, or hard-example selection. Online hard example mining prioritizes high-loss samples produced by the current model and has been influential in computer vision [40]. In computational pathology, hard-negative mining has also been applied to histopathology classification and weakly supervised WSI analysis [41–43]. However, many previous methods have primarily been developed for image- or slide-level classification, or for dense prediction settings in which candidate samples can be evaluated within a manageable field of view. For gigapixel WSI segmentation, repeatedly screening the complete negative pool during pixel-level training can be computationally expensive. A remaining practical challenge is to identify, before segmentation training, a compact and reproducible subset of target-absent patches enriched for morphologically confusing hard negatives. This setting is distinct from active learning, which typically selects unlabeled samples for additional expert annotation and iteratively expands the labeled training set, and from online hard-example mining, which dynamically identifies high-loss or misclassified samples during model optimization [40]. In contrast, FTU-Seek operates on an already annotated WSI patch pool. It ranks FTU-absent patches before segmentation training and constructs a compact, fixed, and reproducible training set without requesting additional annotations.

Pathology foundation models (PFMs) provide a promising basis for this selection problem. Models such as CTransPath, UNI, Virchow, and Prov-GigaPath have been used to learn transferable histomorphological representations, while vision–language models such as CONCH incorporate semantic supervision from paired biomedical text [44–50]. These representations have demonstrated broad utility across cancer types and downstream prediction tasks [51]. Related studies have also explored pathology patch selection, representative region selection, and hard-example mining [41–43]. Accordingly, PFMs and patch-selection strategies have been investigated in related settings.

Here, we propose FTU-Seek, a PFM-guided hard-negative learning framework for eficient segmentation of sparse FTUs in WSIs. FTU-Seek uses frozen multi-depth PFM features to train a patch-level classifier, which ranks an already annotated FTUabsent patch pool according to target-like morphology. A static per-WSI TopK rule is then used to select the highest-scoring negative patches and construct a compact, workload-controlled segmentation training set. Thus, FTU-Seek integrates PFMbased morphological representation, target-specific hard-negative ranking, and static training-set construction for sparse FTU segmentation.

The main contributions of this study are summarized as follows:

• We develop a PFM-guided training-set construction strategy that integrates frozen multi-depth representations, target-specific ranking of annotated FTUabsent patches, and static per-WSI TopK selection to reduce redundant background sampling in sparse FTU segmentation.

• We systematically evaluate classifier-guided hard-negative selection across TLS, blood-vessel, and gland segmentation tasks with distinct sparsity and morphological profiles, using positive-only, all-tissue, random-negative, and matched random TopK strategies to assess the trade-of between segmentation performance and retained training workload.

• We further demonstrate the downstream utility of FTU-Seek through exploratory applications to external TCGA cohorts, showing that segmentation outputs can be transformed into quantitative phenotypes describing immune, vascular, and glandular tissue organization.

Because the selected negative pool is fixed before segmentation training, this construction reduces the number of patches that must be stored, loaded, and processed during the segmentation stage. The resulting benefit is therefore an expected reduction in training-data workload and data-transfer burden, rather than a directly measured reduction in wall-clock time, GPU-memory usage, or energy consumption.

## 2 Materials and Methods

## 2.1 Study Design and Datasets

To develop FTU-Seek for precise spatial segmentation, our methodology was primarily driven by highly granular, instance-level morphological analysis. We retrospectively constructed three task-specific H&E-stained WSI cohorts from Fujian Cancer Hospital (FCH). For each task, we purposely curated a representative subset of WSIs designed to encompass a wide spectrum of tissue architectures and staining profiles, thereby maximizing morphological heterogeneity. Through exhaustive, pixel-level manual delineation of H&E-stained whole-slide images, we extracted three datasets of FTUs, establishing annotations used to dichotomize all derived image patches into FTU-containing and FTU-absent categories. Specifically, the TLS dataset comprised 2079 FTU-containing patches and 41,094 FTU-absent patches from patients with pancreatic ductal adenocarcinoma (PAAD); the gland dataset included 11,670 FTU-containing patches and 20,128 FTU-absent patches from patients with prostate adenocarcinoma (PRAD); and the blood vessel dataset included 4005 FTU-containing patches and 22,048 FTU-absent patches from patients with intrahepatic cholangiocarcinoma. Each patient contributed one WSI. For each task, WSIs were randomized into a model development set and an independent test set at a 4:1 ratio before patch extraction. The resulting cohorts contained 10 patients and 10 TLS WSIs (8 for development and 2 for internal testing), 9 patients and 9 blood-vessel WSIs (7 for development and 2 for internal testing), and 10 patients and 10 gland WSIs (8 for development and 2 for internal testing). Data partitioning was performed at the patient/WSI level and was independent of patch-level sampling. Patch extraction and subsequent positive/negative patch selection were conducted within the predefined development or test cohort, without using patch-level information to determine the split. Given the limited number of internal-test WSIs, these results were interpreted cautiously and were complemented by the independent 30-WSI held-out TLS evaluation.

The development sets—which provided a morphologically dense corpus of training patches—underwent five-fold cross-validation to optimize model hyperparameters, while the independent test sets were completely held out for final segmentation performance evaluation. To further assess the robustness of the segmentation strategies beyond the original two-slide internal test set, we constructed an additional held-out cohort of 30 TLS whole-slide images from the same institution. These WSIs were not used for model training, model selection, hyperparameter tuning, TopK selection, or threshold optimization. The 30-slide cohort was evaluated only after the segmentation configurations had been pre-specified based on the development-set experiments. For each method, predictions from the five cross-validation models were combined as an ensemble, and a fixed segmentation threshold of 0.5 was used for all evaluations.

To investigate the downstream utility of FTU segmentation, diagnostic WSIs and matched clinicopathological data from public TCGA cohorts were downloaded from the NIH Genomic Data Commons Data Portal (https://portal.gdc.cancer.gov) [52].

One diagnostic WSI was included per patient. Three task-specific external analyses were performed. For TLS phenotyping, TCGA-READ, TCGA-ESCA, and TCGA-STAD included 146, 148, and 326 patients, with 26, 62, and 132 deaths, respectively. TLS phenotypes were quantified in TCGA-READ, TCGA-ESCA, and TCGA-STAD to assess immune-structure abundance, size composition, and spatial organization. For vascular phenotyping, TCGA-LIHC included 337 patients with 121 deaths for overall survival (OS) analysis and the microvascular invasion (MVI) analysis. Blood-vessel phenotypes were quantified in TCGA-LIHC. For glandular phenotyping, TCGA-PRAD included 401 patients, with 395, 372, and 342 patients having available pathological T stage, surgical margin, and biochemical recurrence (BCR) information, respectively. Glandular phenotypes were quantified and evaluated in relation to Gleason grade group and the available clinicopathological endpoints described above. TCGA cases were retained when a diagnostic WSI, matched clinical endpoint, and valid tissue-level segmentation output were available. Additionally, we excluded patients with concurrent other malignancies, those who underwent resection for tumor recurrence, and those with a follow-up duration of less than one month. These analyses were designed as exploratory downstream applications of FTU-Seek-derived segmentation outputs rather than as external validation of segmentation performance or definitive clinical prediction analyses.

All procedures and protocols in this study adhered to the principles of the Declaration of Helsinki. Ethical approval for the retrospective FCH cohorts was obtained from the Ethics Committee of Fujian Cancer Hospital on 14 August 2024 (No. SQ2026- 050). The requirement for informed consent was waived because the study used fully de-identified retrospective clinical data. Use of public datasets from TCGA was deemed exempt from ethics committee review because these data are de-identified and publicly available.

## 2.2 WSI Annotation and Preprocessing

All WSIs were acquired at 20× or 40× magnification. For pathological annotation, the boundaries of the target FTUs—namely TLSs, glands, and blood vessels—were manually delineated by three trained annotators using Aperio ImageScope v12.4.6 (Leica Biosystems, Wetzlar, Germany), with one annotator assigned to each FTU type. All annotations were subsequently reviewed and finalized by a senior pathologist (Y.K.) to ensure consistency with the predefined histopathological criteria described below. As annotations were not independently duplicated, inter-observer agreement was not assessed. Annotators were blinded to clinical outcome information during annotation.

• TLSs. TLSs were defined as dense aggregations of lymphocytes within nonlymphoid tissues, while loosely distributed inflammatory infiltrates lacking clear borders or cohesive architecture were excluded [53–55].

• Blood vessels. The annotation of vascular structures depended on the presence of an endothelial lining. Vascular contours were delineated along the outer wall, encompassing the endothelial lining and any associated structural layers (such as smooth muscle). To ensure a comprehensive vascular assessment, the annotations incorporated a diverse range of vessel types, spanning from arterioles with distinct smooth muscle layers to portal and hepatic veins, which frequently lack conspicuous smooth muscle tissue. To prevent the model from learning artifacts, tissue tearing spaces mimicking vascular lumens but lacking an endothelial lining were explicitly excluded [56–58].

• Glands. The boundaries of prostatic glands were precisely traced at the epithelial-stromal junction, excluding the surrounding fibromuscular stroma. These annotations encompassed both benign glands—marked by an intact bilayered architecture (inner secretory and outer basal cells) alongside intraluminal corpora amylacea—and malignant acini, characterized by basal cell loss, crowded or cribriform growth patterns, and enlarged hyperchromatic nuclei [59–61].

After annotation, foreground tissue regions were separated from the surrounding slide background using Otsu thresholding across all datasets [62]. To balance morphological detail with computational eficiency, tissue regions were processed at a spatial resolution of 1 µm/pixel, corresponding to approximately 10× magnification [35, 63]. Each foreground tissue region was subsequently divided into non-overlapping image patches of 256×256 pixels. Before model input, these patches were resized to 224×224 pixels.

The same non-overlapping patch coordinates were retained throughout patch-level classification, segmentation training, validation, and testing. No additional random cropping or overlapping patch generation was performed at later stages. The image transformations applied during model input consisted only of resizing the extracted patches to the model input size, conversion to tensors, and channel-wise normalization using ImageNet mean and standard-deviation values. No random flipping, rotation, color jittering, or other data augmentation was used for either the patch classifier or the segmentation model.

Patches located entirely in slide background regions were excluded from subsequent analysis. For patch-level classification, any patch overlapping the annotated target region was labeled as target-containing, whereas patches without target overlap were labeled as target-absent. This inclusive overlap criterion was selected because the auxiliary classifier was intended to identify all patches containing potential target morphology, including partial or boundary-crossing targets, for subsequent hard-negative ranking. Using a minimum-overlap threshold could discard small or peripheral target regions and increase the risk of assigning target-containing context to the negative pool. These patch-level labels were used only for classifier training and negative-patch ranking, while segmentation training and evaluation were performed using the original pixel-level annotation masks. Thus, non-target pixels within a target-containing patch remained background during segmentation learning. The same ImageNet mean and standard-deviation normalization was used for all classifier and segmentation inputs to reduce diferences in input scale across WSIs.

## 2.3 Model Development

As shown in Figure 3, FTU-Seek was designed as a two-stage hard-negative learning framework for sparse FTU segmentation in WSIs. The framework first uses a patchlevel classifier to identify FTU-absent tissue patches that are likely to be confused with FTU-containing patches.

The pathology foundation model used in FTU-Seek was UNI v1. Its encoder was kept frozen, and multi-depth UNI v1 features were used as inputs to a task-specific patch-level binary classifier. The classifier was trained to distinguish FTU-containing patches from FTU-absent patches using the development-set training WSIs within each cross-validation fold. After training, the classifier assigned each FTU-absent patch a probability of being FTU-containing. Within each WSI, FTU-absent patches were ranked according to this probability, and the highest-scoring TopK patches were selected as hard negatives. These selected hard negatives were combined with all FTUcontaining patches to form the segmentation training set. The resulting training set was then used to optimize the segmentation decoder and prediction head, while the UNI v1 encoder remained frozen.

These classifier-derived hard negatives are then used to construct static TopK segmentation training sets. The static strategy evaluates how the number of classifierderived hard negatives afects segmentation performance and directly compares FPfind TopK sampling with positive-only training, all-tissue training, and random TopK sampling under matched negative-sampling budgets.

![](images/318a5d72220dd1fe6a649fb7450ab3a1f6fa1c70e13b916a56625a3e26bfb946.jpg)  
Fig. 3 Overview of the proposed FTU-Seek framework. A patch-level classifier first ranks FTUabsent tissue patches according to their predicted FTU-containing probabilities, and classifier-derived TopK hard negatives are selected to construct compact segmentation training sets. The framework is evaluated by comparing positive-only training, all-tissue training, random TopK sampling, and classifier-guided FP-find TopK hard-negative sampling.

## 2.3.1 Patch-Level Training Set Definition

Stage I establishes a classifier-guided hard negative pool through patch-level FP screening. This stage consists of patch-level training set definition, classification network construction, and classifier-derived hard negative ranking. Let $\mathcal { P }$ denote the complete set of tissue patches extracted from the training WSIs. According to their overlap with the corresponding annotation masks, $\mathcal { P }$ is partitioned into a target-containing subset ${ \mathcal { P } } ^ { + }$ and a target-absent subset $\mathcal { P } ^ { - }$ :

$$
\mathcal P = \mathcal P ^ { + } \cup \mathcal P ^ { - } , \qquad \mathcal P ^ { + } \cap \mathcal P ^ { - } = \emptyset .
$$

A patch is assigned to $\mathcal { P } ^ { + }$ if it overlaps with at least one annotated target region; otherwise, it is assigned to $\mathcal { P } ^ { - }$ . These patch-level labels are used to train the classifier and to identify high-risk background patches for hard negative initialization.

## 2.3.2 Classification Network

The architecture of the patch-level classification network is illustrated in Figure 4. For each input patch, feature embeddings are extracted from K selected relative depths of the frozen UNI v1 encoder [46, 64]. Each depth-specific feature is independently processed by a dedicated feed-forward transformation head. The transformed features are then aggregated into a fused patch representation, which is finally mapped to a binary prediction indicating whether the patch is target-containing or target-absent.

![](images/c2524623e31d2a02ebb35c4f1436da5c1af3ae59884a70acfe04d32fe0e2623c.jpg)  
Fig. 4 Architecture of the patch-level classification network. Multi-dimensional feature embeddings are extracted from diferent relative depths of the frozen UNI v1 encoder. Each feature is processed by a depth-specific feed-forward transformation head, and the transformed representations are aggregated before binary patch-level classification.

Specifically, for an input patch $x _ { i } .$ , the frozen UNI v1 encoder extracts K feature embeddings, denoted as $\mathbf { f } _ { i } ^ { ( j ) } = E _ { \mathrm { U N I } } ^ { ( r _ { j } ) } ( x _ { i } )$ for $j = 1 , 2 , \dots , K$ . Here, $E _ { \mathrm { U N I } } ^ { ( r _ { j } ) } ( \cdot )$ denotes the encoder output at the selected relative depth $r _ { j }$ , and $\mathbf { f } _ { i } ^ { ( j ) } \in \mathbb { R } ^ { 1 0 2 4 }$

Each depth-specific feature is independently transformed by a feed-forward head:

$$
\mathbf h _ { i } ^ { ( j ) } = F _ { j } \left( \mathbf f _ { i } ^ { ( j ) } \right) , \qquad j = 1 , 2 , \ldots , K ,
$$

where $F _ { j } ( \cdot )$ denotes the feed-forward head for the $j \cdot$ th selected encoder depth. In this study, $K = 1 0$ , with relative encoder depths uniformly selected as $0 . 1 , 0 . 2 , \ldots , 1 . 0 $ In our implementation, each head consists of two fully connected layers with a ReLU activation, expanding the feature dimension from 1024 to 4096 and projecting it back to 1024.

The transformed representations from all selected depths are fused by mean aggregation and mapped to binary class probabilities:

$$
\hat { \mathbf { y } } _ { i } = \mathrm { S o f t m a x } \left( C \left( \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \mathbf { h } _ { i } ^ { ( j ) } \right) \right) , \qquad \hat { \mathbf { y } } _ { i } \in \mathbb { R } ^ { 2 } ,\tag{2.1}
$$

where $C ( \cdot )$ denotes the final linear classification layer. The two output dimensions correspond to the target-absent and target-containing classes, respectively. After training, the patch-level classifier is applied to the training patch set. For each patch $x _ { i } .$ the predicted patch label is defined as

$$
\hat { c } _ { i } = \arg \operatorname* { m a x } _ { c \in \{ 0 , 1 \} } \hat { y } _ { i , c } ,
$$

where $c \ = \ 0$ and $c \ = \ 1$ denote the target-absent and target-containing classes, respectively.

Among the target-absent patches, those assigned high target-containing probabilities were regarded as classifier-derived hard negatives. Let $s _ { i } \ = \ { \hat { y } } _ { i } ,$ denote the predicted target-containing probability for a negative patch $x _ { i } \in \mathcal P ^ { - }$ . In the reported experiments, hard negative selection was performed independently within each WSI using the per wsi scope, rather than on a globally pooled negative-patch set. The candidate records were ordered according to the stored within-WSI ranking, with the predicted target-containing probability used as a secondary sorting criterion. The first K eligible records were selected:

$$
\mathcal { H } _ { K } ^ { - } = \mathrm { T o p K } _ { x _ { i } \in \mathcal { P } ^ { - } } ( s _ { i } ) .
$$

Here, K denotes the nominal number of selected classifier-derived negative patches, and $K \in \{ 1 0 0 , 3 0 0 , 5 0 0 , 1 0 0 0 \}$ was evaluated using the same candidate values across all FTU categories. This common candidate set was chosen to enable controlled accuracy–workload comparisons under matched sampling budgets, rather than to define a prevalence-adaptive rule or a universally optimal budget. When a WSI contained fewer than K eligible negative patches, all available candidates were retained. Duplicate patch indices, missing coordinate entries, and patches identified as FTUcontaining during coordinate checking were excluded from the final negative-patch list. If both the stored ranking and the predicted probability were tied, the original order of the merged ranking records was retained. Therefore, the final number of selected negative patches was at most K and could vary across WSIs. For a crossvalidation fold with training-WSI set $\mathcal { W } _ { f }$ , the final negative-patch count was calculated as $\sum { } _ { w \in \mathcal { W } _ { f } }$ min $( K , N _ { \mathrm { e l i g i b l e } , w } )$ after the exclusion rules described above. Each crossvalidation fold used only the selected negative patches belonging to its corresponding training WSIs.

The corresponding static segmentation training set was defined as

$$
\mathcal { T } _ { \mathrm { T o p K } } = \mathcal { P } ^ { + } \cup \mathcal { H } _ { K } ^ { - } .
$$

This per-WSI design was used to prevent WSIs with larger tissue areas from dominating the selected negative-patch pool and to enable matched comparisons between classifier-guided and random sampling.

This setting isolates the contribution of classifier-derived hard negatives and allows direct comparison with positive-only training, all-tissue training, and random negative sampling.

## 2.3.3 Segmentation Network

The segmentation network used in Stage II is illustrated in Figure 5. It consists of a frozen UNI v1 encoder [46, 65] and a multi-scale feature fusion decoder inspired by dense prediction transformer designs [66]. For each input patch, multi-depth Transformer features extracted by UNI v1 are reshaped into spatial feature maps and progressively fused in the decoder. Starting from the deepest feature, the decoder gradually restores spatial resolution by integrating shallower Transformer features and a shallow convolutional representation of the original image, ultimately producing a dense two-class segmentation map.

![](images/da1909bf90b4c927aeda45233deb1654beb9fa736cc42cf653743f620b40c524.jpg)  
Fig. 5 Architecture of the segmentation network. Four Transformer-layer features are extracted from the frozen UNI v1 encoder at relative depths of $1 / 4 , 1 / 2 , 3 / 4 ,$ and $^ { 1 , }$ reshaped into spatial feature maps, and progressively fused with shallow image features by a multi-scale decoder to produce dense binary segmentation predictions.

Given an input patch $x _ { i } \in { \mathbb R } ^ { 3 \times H \times W }$ , the frozen UNI v1 encoder first partitions it into non-overlapping $h \times u$ image tokens. This results in a token grid of size $H _ { t } \times W _ { t } .$ where $H _ { t } = H / h$ and $W _ { t } = W / w$ . To provide multi-level representations for dense prediction, we extract token features from L selected relative encoder depths:

$$
T _ { i } ^ { ( j ) } = E _ { \mathrm { U N I } } ^ { ( r _ { j } ) } ( x _ { i } ) , \qquad j = 1 , 2 , \ldots , L ,
$$

where $E _ { \mathrm { U N I } } ^ { ( r _ { j } ) } ( \cdot )$ denotes the UNI v1 output at the selected relative depth $r _ { j }$ . In this study, $L = 4$ and $r _ { j } \in \{ 1 / 4 , 1 / 2 , 3 / 4 , 1 \}$ . Each feature sequence contains one class token and $H _ { t } \times W _ { t }$ patch tokens, i.e., $T _ { i } ^ { ( j ) } \in \mathbb { R } ^ { ( 1 + H _ { t } W _ { t } ) \times 1 0 2 4 }$

Before decoding, the class token is removed, and the remaining patch-token subset $T _ { i , \mathrm { p a t c h } } ^ { ( j ) }$ is reshaped into a two-dimensional feature map:

$$
Z _ { i } ^ { ( j ) } = \mathcal { R } \left( T _ { i , \mathrm { p a t c h } } ^ { \left( j \right) } \right) , \qquad Z _ { i } ^ { ( j ) } \in \mathbb { R } ^ { 1 0 2 4 \times H _ { t } \times W _ { t } } ,
$$

where $\mathcal { R } ( \cdot )$ denotes the token-to-feature-map reshaping operation. For simplicity, the four reshaped feature maps are denoted as $Z _ { 1 } , Z _ { 2 } , Z _ { 3 } , Z _ { 4 }$

The decoder adopts a progressive multi-scale fusion strategy. Starting from the deepest feature $Z _ { 4 }$ , it gradually restores spatial resolution by integrating shallower encoder features. Taking the fusion with $Z _ { 3 }$ as an example, $Z _ { 4 }$ is first upsampled to obtain $B _ { 4 }$ , while $Z _ { 3 }$ is transformed to the same spatial scale and channel dimension.

The two features are then concatenated and further refined:

$$
B _ { 3 } = U _ { 3 } \left( \mathrm { C o n c a t } \left( U _ { 4 } ( Z _ { 4 } ) , S _ { 3 } ( Z _ { 3 } ) \right) \right) ,
$$

where $U _ { 4 } ( \cdot )$ denotes the bottleneck upsampling block, $S _ { 3 } ( \cdot )$ denotes the scalealignment and channel-projection transformation for $Z _ { 3 } , U _ { 3 } ( \cdot )$ denotes the fusion-andupsampling block, and $\mathrm { C o n c a t } ( \cdot , \cdot )$ denotes channel-wise concatenation.

The same fusion procedure is applied to the remaining encoder features. Specifically, $B _ { 3 }$ is fused with the transformed $Z _ { 2 }$ to produce $B _ { 2 }$ , and $B _ { 2 }$ is further fused with the transformed $Z _ { 1 }$ to obtain $B _ { 1 }$ . In our implementation, the decoded feature resolution is progressively restored from the token-grid resolution $H _ { t } \times W _ { t }$ to the original input resolution $H \times W$

In parallel, the original input image is processed by a shallow convolutional branch $S _ { 0 } ( \cdot )$ to preserve local texture and boundary information. The final segmentation prediction is obtained by concatenating the decoded feature $B _ { 1 }$ with the shallow image feature $S _ { 0 } ( x _ { i } )$ and passing the fused representation through the segmentation head:

$$
\begin{array} { r l r } { \hat { Y } _ { i } = \operatorname { S o f t m a x } \left( H \left( \operatorname { C o n c a t } \left( B _ { 1 } , S _ { 0 } ( x _ { i } ) \right) \right) \right) , } & { } & { \hat { Y } _ { i } \in \mathbb { R } ^ { 2 \times H \times W } , } \end{array}
$$

where $H ( \cdot )$ denotes the final convolutional prediction head, and Conca $, ( \cdot , \cdot )$ denotes channel-wise concatenation. The two output channels correspond to the non-target and target classes, respectively.

During training, the UNI v1 encoder remains frozen, while the downstream decoding modules and segmentation head are optimized. This design leverages pretrained pathology representations while allowing the task-specific prediction layers to adapt to the segmentation targets. In the present experiments, the static TopK setting was used as the primary FTU-Seek configuration. This choice keeps the training protocol reproducible and isolates the efect of classifier-guided hard-negative selection from additional segmentation-stage re-mining. The candidate values $K \in$ {100, 300, 500, 1000} were predefined exploratory operating points selected to cover progressively broader hard-negative budgets, from highly compact training sets to more extensive negative coverage. They were not adopted from a specific prior study and were not intended to define a universally optimal or prevalence-adaptive budget.

## 2.3.4 Optimization Objectives and Training Protocol

The patch-level classifier was optimized using cross-entropy loss. Let $\mathbf { y } _ { i } = [ y _ { i , 0 } , y _ { i , 1 } ]$ denote the one-hot patch-level ground-truth label, and let $\hat { \mathbf { y } } _ { i } = [ \hat { y } _ { i , 0 } , \hat { y } _ { i , 1 } ]$ denote the predicted class probabilities obtained from Equation (2.1). The classification loss is defined as

$$
\mathcal { L } _ { \mathrm { c l s } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 0 } ^ { 1 } y _ { i , c } \log \hat { y } _ { i , c } ,
$$

where N is the number of training patches in a mini-batch, and $c = 0$ and $c = 1$ correspond to the target-absent and target-containing classes, respectively.

To address the foreground–background imbalance inherent to sparse target segmentation, Dice–Focal loss was adopted for model optimization [38, 67, 68]: $\mathcal { L } _ { \mathrm { s e g } } =$ $\mathcal { L } _ { \mathrm { D i c e } } + \mathcal { L } _ { \mathrm { F o c a l } } .$

Let $p _ { n } \in [ 0 , 1 ]$ denote the predicted target-class probability of the n-th pixel and $g _ { n } \in \{ 0 , 1 \}$ denote the corresponding ground-truth label. The Dice loss is defined as

$$
\mathcal { L } _ { \mathrm { { D i c e } } } = 1 - \frac { 2 \sum _ { n = 1 } ^ { M } p _ { n } g _ { n } + \epsilon } { \sum _ { n = 1 } ^ { M } p _ { n } + \sum _ { n = 1 } ^ { M } g _ { n } + \epsilon } ,
$$

where $M$ is the total number of pixels involved in the loss computation and ϵ is a small constant for numerical stability.

The focal loss is defined as

$$
\mathcal { L } _ { \mathrm { F o c a l } } = - \frac { 1 } { M } \sum _ { n = 1 } ^ { M } [ \alpha g _ { n } ( 1 - p _ { n } ) ^ { \gamma } \log p _ { n } + ( 1 - \alpha ) ( 1 - g _ { n } ) p _ { n } ^ { \gamma } \log ( 1 - p _ { n } ) ] ,
$$

where $\alpha$ and $\gamma$ denote the balancing and focusing parameters, respectively. Fivefold cross-validation was performed on the development set at the WSI level. In each fold, models were trained using the corresponding training split, and model selection was performed on the validation split. For classifier-guided hard negative strategies, the patch-level classifier was trained within the same fold to rank targetabsent patches, and the resulting TopK hard negatives were used to construct the segmentation training set.

The segmentation model was trained using a Dice–Focal loss and a ReduceLROn-Plateau learning rate scheduler. For each fold, the model with the best validation performance was retained and evaluated on the independent test set.

## 2.3.5 Comparative Patch Construction Strategies

To assess the contribution of classifier-guided hard negative selection, we compared the following patch construction strategies:

• Positive-only training. This strategy trains the segmentation model only with target-containing patches: $\mathcal { T } _ { \mathrm { p o s } } = \mathcal { P } ^ { + }$

• All-tissue training. This strategy uses all available tissue patches: $\mathcal { T } _ { \mathrm { a l l } } = \mathcal { P } ^ { + } \cup \mathcal { P } ^ { - }$

• Random negative sampling.

This strategy retains all target-containing patches and randomly selects a target-absent subset $\mathcal { R } ^ { - }$ from $\mathcal { P } ^ { - }$ . For the matched random-K experiments, eligible negative patches were sampled independently within each WSI without replacement using min $( K , N _ { \mathrm { e l i g i b l e } } )$ candidates, where $N _ { \mathrm { e l i g i b l e } }$ denotes the number of eligible negative patches available for that WSI. Thus, matched random-K sampling used the same per-WSI negative-patch budget as the corresponding FTU-Seek TopK configuration, but without classifier-based ranking. The corresponding segmentation training set was

$$
\mathcal { T } _ { \mathrm { r a n d } } = \mathcal { P } ^ { + } \cup \mathcal { R } ^ { - } .
$$

• Static FP-find TopK sampling. This strategy trains the model with all positive patches and the classifier-derived TopK hard negatives: $\mathcal { T } _ { \mathrm { T o p K } } = \mathcal { P } ^ { + } \cup \mathcal { H } _ { K } ^ { - }$ . This setting evaluates whether high-risk negative patches selected by the patch-level classifier are more informative than randomly selected negatives.

## 2.4 Outcomes

The performance of the deep learning framework was evaluated across two distinct dimensions: spatial segmentation accuracy and patch-level classification performance. To assess the spatial overlap and delineation accuracy of the segmentation network, the Dice similarity coeficient (DSC) was utilized as the primary metric [37, 69]. DSC quantifies the pixel-level agreement between the model’s predicted output masks and the manual ground truth annotations, providing a rigorous measure of morphological fidelity:

$$
\mathrm { D S C } = { \frac { 2 \mathrm { T P } } { 2 \mathrm { T P } + \mathrm { F P } + \mathrm { F N } } } .
$$

For the patch-level classification performance, a confusion matrix was constructed to tally true positives (TPs), true negatives (TNs), false positives (FPs), and false negatives (FNs). The principal metric for this phase was the area under the receiver operating characteristic curve (AUC), which provides a threshold-independent measure of discriminative ability. Additionally, several secondary metrics were derived from the confusion matrix to provide a comprehensive assessment. Sensitivity (recall) and specificity were calculated to evaluate the capacity to correctly identify positive and negative cases:

$$
\mathrm { S e n s i t i v i t y } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } , \qquad \mathrm { S p e c i f i c i t y } = { \frac { \mathrm { T N } } { \mathrm { T N } + \mathrm { F P } } } .
$$

Positive predictive value (PPV) and negative predictive value (NPV) assessed the reliability of the classifier predictions:

$$
\mathrm { P P V } = \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } , \qquad \mathrm { N P V } = \frac { \mathrm { T N } } { \mathrm { T N } + \mathrm { F N } } .
$$

Furthermore, overall accuracy and the F1-score—the harmonic mean of precision and sensitivity—were computed to measure holistic performance [70], particularly addressing potential class imbalance:

$$
\mathrm { A c c u r a c y } = { \frac { \mathrm { T P } + \mathrm { T N } } { \mathrm { T P } + \mathrm { T N } + \mathrm { F P } + \mathrm { F N } } } , \qquad \mathrm { F l - s c o r e } = 2 \times { \frac { \mathrm { P P V } \times \mathrm { S e n s i t i v i t y } } { \mathrm { P P V } + \mathrm { S e n s i t i v i t y } } } .
$$

For downstream analysis, segmentation outputs were converted into slide- or patient-level FTU phenotypes. TLS features included TLS area density (ratio of TLS area to total tissue area), TLS count density (number of TLS per unit area), size-stratified TLS densities (small/medium/large by area tertiles), hotspot TLS density, and nearest-neighbor clustering indices (observed mean nearest-neighbor distance divided by random expectation). Blood-vessel features included total and filtered

BV density, vessel count density, size-stratified vessel densities, and hotspot vascular density measures. Gland features included gland density, gland count density, sizestratified gland densities, and shape descriptors such as area, perimeter, circularity, solidity, and eccentricity. OS was used for TLS and BV survival analyses when survival information was available. In LIHC, the distributions of vascular characteristics were compared between MVI-positive and MVI-negative patients. For PRAD, glandular features were evaluated primarily against pathology-related endpoints and BCR status; BCR was analyzed as a binary endpoint because reliable BCR-specific censoring times were not available for non-recurrent cases. Additionally, we evaluated the diferences in gland features across patients with high versus low Gleason scores, diferent pT stages, and resection margin statuses (R0 vs. R1).

## 2.5 Implementation Details

All model training experiments were implemented in PyTorch and performed on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU with 97,887 MiB of memory. Although two GPUs were available on the workstation, each training run was executed on a single GPU using the specified CUDA device. To improve reproducibility, Python, NumPy, PyTorch, and CUDA random seeds were fixed, and deterministic cuDNN settings were enabled.

The patch-level classifier was trained in the FP-find classification pipeline. For each FTU task and cross-validation fold, frozen UNI v1 embeddings were extracted from ten relative encoder depths $( 0 . 1 , 0 . 2 , \ldots , 1 . 0 )$ . Each 1024-dimensional depth-specific feature was passed through a two-layer feed-forward transformation head with ReLU activation, expanded to 4096 dimensions and projected back to 1024 dimensions. The transformed multi-depth features were averaged and passed to a final linear classification layer to predict target-absent versus target-containing patch labels. The classifier was optimized using cross-entropy loss and AdamW with an initial learning rate of $5 \times 1 0 ^ { - 5 }$ , weight decay of $5 \times 1 0 ^ { - 6 }$ , batch size of 128, and a maximum of 500 epochs. A ReduceLROnPlateau scheduler was applied to the validation AUC, and early stopping was used with a patience of 15 epochs. For each fold, the checkpoint with the highest validation AUC was selected as the validation-best classifier. This classifier was then applied to FTU-absent training patches to compute the predicted FTU-containing probability. These probabilities were used to rank candidate hard negatives, and the top-ranked FTU-absent patches were exported for downstream static TopK segmentation training.

The segmentation model was trained within the FTU-Seek segmentation pipeline. For each FTU task, five-fold cross-validation was performed at the WSI level on the development set, and the independent test set was held out for final evaluation. The segmentation network used a frozen UNI v1 encoder and a trainable residual feature-fusion decoder. Four encoder depths $( 1 / 4 , 1 / 2 , 3 / 4 , \mathrm { ~ a n d } 1 )$ were used for dense prediction, and the corresponding token features were reshaped into spatial feature maps before decoder fusion. Input image patches were resized to 224 × 224 pixels and normalized before model input. For classifier-guided FTU-Seek sampling, the segmentation training set contained all FTU-containing patches and the classifierranked FTU-absent hard negatives selected by static TopK sampling within each WSI.

Matched random TopK experiments used the same number of negative patches per WSI but selected them randomly from the FTU-absent pool.

During segmentation training, only the decoder and prediction head were updated, whereas all UNI v1 encoder parameters remained frozen. The trainable parameters were optimized using AdamW with an initial learning rate of $5 \times 1 0 ^ { - 5 }$ , weight decay of $5 \times 1 0 ^ { - 6 }$ , batch size of 32, and a maximum of 50 epochs. Mixed-precision training was enabled when CUDA was available using automatic casting and gradient scaling. The segmentation objective combined multiclass Dice loss and focal loss. In the implementation, the focal term used class weights of 0.25 and 0.75 with a focusing parameter of $\gamma = 2 . 0$ , and the final loss was defined as $\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { D i c e } } + 2 0 \mathcal { L } _ { \mathrm { F o c a l } }$ . A ReduceL-ROnPlateau scheduler monitored the validation mean Dice coeficient and reduced the learning rate with a factor of 0.95, patience of 5 epochs, cooldown of 2 epochs, and minimum learning rate of $1 \times 1 0 ^ { - 7 }$ . Validation was performed after each training epoch, and early stopping was applied with a patience of 5 epochs. For each fold, the checkpoint with the highest validation mean Dice was retained and subsequently evaluated on the independent test set.

For reproducibility, the primary experiments were conducted using random seed 1, and repeated random-baseline experiments used seeds 1–5. The WSI-level fold assignments and train–validation–test split metadata were fixed before model training and were stored in the corresponding split files. Experiments were run in Python 3.11.15 using PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, NumPy 1.26.4, pandas 2.3.3, SciPy 1.12.0, h5py 3.15.0, timm 1.0.20, einops 0.8.1, Pillow 12.2.0, Matplotlib 3.10.7, and seaborn 0.13.2. The pathology foundation model was the publicly released UNI v1 model from the Mahmood Lab.

During inference, the foreground probability was obtained from the foreground class of the two-class softmax output. A fixed threshold of 0.5 was used to convert the probability map into a binary foreground mask, with pixels having foreground probability $\ge ~ 0 . 5$ classified as positive. Predictions outside the foreground tissue mask were set to zero. Patch-level predictions were mapped back to their original WSI coordinates for slide-level mask visualization and downstream tissue phenotyping. For Dice calculation, pixel-level true-positive, false-positive, and falsenegative counts were accumulated across all evaluated patches within each WSI. Thus, the reported Dice was not obtained by averaging patch-level Dice values. No connected-component filtering, hole filling, minimum-instance-size filtering, or other morphological post-processing was applied before Dice calculation.

## 2.6 Statistical Analysis

Segmentation and classification metrics on the development set and internal test set were summarized as mean ± standard deviation across five cross-validation folds. For the multi-seed experiments, the original run used seed 1, and four additional randombaseline runs using seeds 2–5 were subsequently performed. Random-baseline results were summarized across all five seeds (1–5) using the mean and standard deviation. These standard deviations quantify variability caused by random sampling and stochastic training across seeds. Because the original FP-find experiments were conducted using seed 1, the multi-seed analysis was interpreted primarily as an assessment of the stability of the random-sampling baselines rather than as a direct estimate of FP-find seed robustness.

For the 30-slide held-out TLS evaluation, Dice scores were calculated independently for each WSI and summarized as mean ± standard deviation across slides. For each paired comparison, the paired diference was calculated as the Dice score of FTU-Seek minus that of the comparator strategy. Two-sided 95% confidence intervals for the mean paired diferences were calculated using the Student’s t distribution with 29 degrees of freedom. Paired t-tests and Wilcoxon signed-rank tests were performed using the WSI as the pairing unit. The comparisons included matched random TopK strategies at K = 100, 300, 500, and 1000, as well as the representative Top1000 FTU-Seek strategy versus all-tissue and random-balanced training. All formal paired analyses were restricted to the independent 30-WSI held-out TLS cohort. The internal test cohort contained only two WSIs and was therefore retained for descriptive comparison rather than formal hypothesis testing. Predictions for the held-out cohort were generated using five-fold probability ensembles, in which foreground probabilities from the five-fold models were averaged before thresholding. The reported p-values were nominal two-sided values.

For segmentation, the Dice coeficient was reported on both validation folds and independent test sets, and training workload was calculated as the proportion of retained training patches relative to all-tissue training within each cohort.

For each evaluation WSI, pixel-level predictions and reference masks were processed patch by patch, while the numbers of true-positive, false-positive, and false-negative pixels were accumulated across all evaluated patches from that WSI. The WSI-level Dice coeficient was then calculated as

$$
\mathrm { D i c e } _ { \mathrm { W S I } } = \frac { 2 T P _ { \mathrm { W S I } } } { 2 T P _ { \mathrm { W S I } } + F P _ { \mathrm { W S I } } + F N _ { \mathrm { W S I } } } .
$$

Thus, the reported Dice was not obtained by averaging patch-level Dice coeficients and was not an instance-level metric. Because the extracted patches were non-overlapping, this accumulation corresponds to pixel-level evaluation over the complete WSI tissue region without double-counting overlapping patches.

For the original development and internal-test experiments, Dice values were summarized across the five cross-validation folds. For the held-out TLS analysis, one WSI-level Dice value was obtained for each of the 30 WSIs, and these WSI-level values were used for the paired comparisons and confidence interval calculations.

For downstream survival analyses, OS was defined as the interval from diagnosis or surgery to death from any cause, with surviving patients censored at last followup. A total of 60 TLS features were evaluated separately in each of TCGA-READ, TCGA-ESCA, and TCGA-STAD, while 36 vascular features were evaluated in TCGA-LIHC. For TLS analyses, pairwise Wilcoxon rank-sum tests were used for descriptive cross-cohort comparison of TLS morphology and spatial organization, and associations with OS were evaluated using Cox proportional hazards models with individual TLS features entered as continuous variables. Features were standardized to facilitate comparison across diferent measurement scales, and hazard ratios (HRs) with 95% confidence intervals (CIs) were reported per 1-standard-deviation (SD) increase.

To account for multiple feature-wise survival analyses, the Benjamini–Hochberg false discovery rate (FDR) procedure was applied separately within each TLS cohort across the 60 Cox regression tests performed in that cohort. Kaplan–Meier analyses were used as exploratory visualizations of selected TLS phenotypes. For exploratory cut-point analysis, candidate thresholds were searched under minimum group-size constraints to avoid extreme boundary splits, and the threshold with the smallest log-rank p value was retained for visualization.

For LIHC, vascular phenotypes were further examined in relation to OS and MVI. For survival analysis, selected features were dichotomized using the exploratory dataderived cut points described above and evaluated using Cox proportional hazards models, with HRs and 95% CIs reported for the high- versus low-feature groups. Because these cut points were outcome-derived, the corresponding survival analyses were considered exploratory and were not interpreted as defining validated prognostic thresholds. Multivariable Cox regression was used to assess whether the vascular phenotype remained associated with OS after adjustment for available clinical covariates. The proportional-hazards assumption was assessed using a feature-by-log-transformedtime interaction term, and multicollinearity among predictors was evaluated using variance inflation factors (VIFs). For MVI, all 36 vascular features were compared between patients with and without MVI using the Wilcoxon rank-sum test and summarized as median and interquartile range (IQR), with rank-biserial correlation reported as the efect size. The Benjamini–Hochberg FDR procedure was applied across the 36 feature-wise MVI comparisons within TCGA-LIHC. Features of interest were subsequently evaluated in multivariable logistic regression as continuous variables, with associations reported as adjusted odds ratios (ORs) and corresponding 95% CIs. Available clinical covariates were included without univariable statistical pre-screening, except for tumor stage, which was excluded because of its close relationship with vascular invasion and the potential for information overlap. No missing data were present among the 337 patients included in the analysis. Continuous vascular predictors were Z-standardized before regression, and associations were reported as adjusted ORs with 95% CIs per 1-SD increase. Multicollinearity was assessed using VIFs, and potential nonlinearity of the vascular phenotype–MVI association was examined using restricted cubic splines (RCSs). The multivariable regression analyses were used to assess adjusted associations rather than to develop clinical prediction models.

Similarly, for PRAD glandular analyses, 40 glandular features were evaluated according to Gleason score (≤7 vs. >7), pathological T stage (T2 vs. T3/T4), surgical margin status (R0 vs. R1), and BCR status (No vs. Yes). Continuous glandular features were compared between groups using the Wilcoxon test and summarized as median and interquartile range (IQR). Efect sizes were reported using rankbiserial correlation. The Benjamini–Hochberg FDR procedure was applied separately within each clinicopathological endpoint, corresponding to 40 feature-wise comparisons for Gleason score, 40 for pathological T stage, 40 for surgical margin status, and 40 for BCR status. All statistical tests were two-sided, and $\textit { p } < \ 0 . 0 5$ was considered nominally significant. For the exploratory TLS survival analyses and the vascular and glandular feature-comparison analyses, FDR-adjusted $p ~ < ~ 0 . 1 0$ was used as an exploratory multiple-testing threshold. The downstream analyses were designed as proof-of-concept evaluations of associations between segmentationderived phenotypes and clinicopathological characteristics rather than as confirmatory biomarker-validation analyses, and the resulting clinical associations were therefore considered hypothesis-generating.

## 3 Results

## 3.1 Dataset Characteristics

As shown in Table 1, the three FTU segmentation cohorts difered substantially in target abundance and spatial sparsity at the instance, patch, and pixel levels. At the instance level, the development sets contained 133 TLSs, 2508 blood vessels, and 3074 glands, corresponding to mean numbers of 16.62, 358.29, and 384.25 instances per WSI, respectively. A similar pattern was observed in the test sets, indicating that TLSs were considerably less frequent than blood vessels and glands.

The patch-level statistics further demonstrated pronounced class imbalance. In the TLS development set, only 1322 of 33,752 tissue patches contained annotated TLS regions, yielding an FTU-containing patch ratio of 3.92%. The corresponding ratio increased to 8.04% in the test set, with 757 positive patches among 9421 tissue patches. For BV segmentation, 3422 of 20,831 development patches and 583 of 5222 test patches contained target regions, corresponding to ratios of 16.43% and 11.16%, respectively. In contrast, glands were more densely distributed, with FTU-containing patch ratios of 36.86% in the development set and 35.92% in the test set.

This diference was also evident at the pixel level. TLS regions accounted for only 0.65% of the tissue area in the development set and 2.32% in the test set. BV pixels represented 2.46% and 1.67% of the tissue area, respectively, whereas gland regions occupied substantially larger proportions of 18.91% and 23.86%. Collectively, these statistics show that TLS segmentation represents the most spatially sparse and imbalanced task, BV segmentation exhibits moderate sparsity, and gland segmentation contains comparatively abundant target regions at both patch and pixel levels.

As summarized in Tables 2 and 3, external TCGA cohorts were used for exploratory downstream application across the three FTU segmentation tasks. For TLS, OS analyses were conducted in the TCGA-READ, TCGA-ESCA, and TCGA-STAD cohorts. For BV, vascular phenotypes in TCGA-LIHC were evaluated in relation to OS and MVI. For gland segmentation, the TCGA-PRAD cohort was evaluated against multiple clinicopathological endpoints, including Gleason score, pathological T stage, surgical margin status, and BCR status. These cohorts enabled an exploratory assessment of whether the automatically quantified FTU features were associated with patient prognosis and clinically relevant pathological characteristics across diferent cancer types.

## 3.2 Patch-Level Classifier Performance

Before segmentation training, we evaluated whether the patch-level classifier could distinguish FTU-containing patches from FTU-absent tissue patches and provide a reliable ranking score for hard-negative selection.

Table 1 Dataset characteristics and spatial sparsity of the three FTU segmentation cohorts at instance, patch, and pixel levels.
<table><tr><td>Characteristics</td><td>TLS</td><td>BV</td><td>Gland</td></tr><tr><td colspan="4">Development Set Instance level</td></tr><tr><td>Unique patients</td><td>8</td><td>7</td><td>8</td></tr><tr><td>WSIs</td><td>8</td><td>7</td><td>8</td></tr><tr><td>WSIs per patient</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Total FTU instances</td><td>133</td><td>2508</td><td>3074</td></tr><tr><td>Mean FTU instances per WSI</td><td>16.62</td><td>358.29</td><td>384.25</td></tr><tr><td>Median FTU instances per WSI</td><td>12.50</td><td>316.00</td><td>377.50</td></tr><tr><td>Patch level Total tissue patches</td><td>33,752</td><td>20,831</td><td>26,366</td></tr><tr><td>FTU-containing patches</td><td>1322</td><td>3422</td><td>9719</td></tr><tr><td>FTU-absent patches</td><td>32,430</td><td>17,409</td><td>16,647</td></tr><tr><td>FTU-containing patch ratio (%)</td><td>3.92</td><td>16.43</td><td>36.86</td></tr><tr><td>Pixel level</td><td></td><td></td><td></td></tr><tr><td>Tissue pixel area</td><td>145,723,284</td><td>67,053,366</td><td>80,370,006</td></tr><tr><td>FTU pixel area</td><td>942,758</td><td>1,646,695</td><td>15,197,324</td></tr><tr><td>FTU pixel area ratio (%)</td><td>0.65</td><td>2.46</td><td>18.91</td></tr><tr><td>Test Set</td><td></td><td></td><td></td></tr><tr><td>Instance level Unique patients</td><td>2</td><td>2</td><td>2</td></tr><tr><td>WSIs</td><td>2</td><td>2</td><td>2</td></tr><tr><td>WSIs per patient</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Total FTU instances</td><td>105</td><td>409</td><td>453</td></tr><tr><td>Mean FTU instances per WSI</td><td>52.50</td><td>204.50</td><td>226.50</td></tr><tr><td>Median FTU instances per WSI</td><td>52.50</td><td>204.50</td><td>226.50</td></tr><tr><td>Patch level</td><td></td><td></td><td></td></tr><tr><td>Total tissue patches</td><td>9421</td><td>5222</td><td>5432</td></tr><tr><td>FTU-containing patches</td><td>757</td><td>583</td><td>1951</td></tr><tr><td>FTU-absent patches</td><td>8664</td><td>4639</td><td>3481</td></tr><tr><td>FTU-containing patch ratio (%)</td><td>8.04</td><td>11.16</td><td>35.92</td></tr><tr><td>Pixel level</td><td></td><td></td><td></td></tr><tr><td>Tissue pixel area FTU pixel area</td><td>38,848,017 901,489</td><td>16,370,435</td><td>16,686,067</td></tr><tr><td></td><td></td><td>273,117</td><td>3,981,224</td></tr><tr><td>FTU pixel area ratio (%)</td><td>2.32</td><td>1.67</td><td>23.86</td></tr></table>

Abbreviations: FTU, functional tissue unit; TLS, tertiary lymphoid structure; BV, blood vessel; WSI, wholeslide image. Dataset statistics were calculated from the actual slides included in the train val test.json split files used for segmentation experiments. Development and test set statistics are reported separately in this table and were not pooled. “FTU-containing patches” denotes tissue patches overlapping annotated FTU regions, whereas “FTU-absent patches” denotes tissue patches without annotated FTU regions. Patch-level ratios were calculated as FTU-containing patches divided by total tissue patches. Pixel-level ratios were calculated as FTU pixel area divided by total tissue pixel area.

As summarized in Table 4 and visualized in Figure 6, the validation-best patch classifiers achieved consistently strong discrimination across the three FTU tasks, although their error profiles difered according to target prevalence and morphological complexity. The TLS classifier achieved a mean AUC of 95.92 ± 2.81%, with a sensitivity of 88.46 ± 4.74% and a specificity of 92.01 ± 6.46%. The corresponding confusion matrices showed true-positive rates ranging from 81.6% to 94.4% and truenegative rates ranging from 81.9% to 98.3%. Its high NPV of $9 9 . 4 8 \pm 0 . 3 0 \%$ indicates reliable exclusion of TLS-absent patches, whereas the lower and more variable PPV of $3 8 . 9 7 \pm 2 3 . 0 4 \%$ reflects the pronounced rarity of TLS-containing patches and the resulting sensitivity of precision to class prevalence.

Table 2 Baseline characteristics of the TCGA cohorts.
<table><tr><td></td><td>TCGA- STAD (n = 326)</td><td>TCGA- READ  $( n = 1 4 6 )$ </td><td>TCGA- ESCA  $( n = 1 4 8 )$ </td><td>TCGA- LIHC  $( n = 3 3 7 )$ </td><td>TCGA- PRAD (n = 401)</td></tr><tr><td>Sex</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Male</td><td>215 (65.95)</td><td>82 (56.16)</td><td>128 (86.49)</td><td>228 (67.66)</td><td>401 (100)</td></tr><tr><td>Female</td><td>111 (34.05)</td><td>64 (43.84)</td><td>20 (13.51)</td><td>109 (32.34)</td><td>0 (0)</td></tr><tr><td>Age</td><td>66 (57–72)</td><td>65 (57–72)</td><td>59 (53–69)</td><td>61 (51–68)</td><td>61 (56–66)</td></tr><tr><td>Stage</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>I</td><td>39 (11.96)</td><td>28 (19.18)</td><td>34 (22.97)</td><td>166 (49.26)</td><td>29 (7.23)</td></tr><tr><td>II</td><td>109 (33.44)</td><td>46 (31.51)</td><td>49 (33.11)</td><td>81 (24.04)</td><td>121 (30.17)</td></tr><tr><td>III</td><td>159 (48.77)</td><td>51 (35.62)</td><td>47 (31.76)</td><td>83 (24.63)</td><td>192 (47.88)</td></tr><tr><td>IV</td><td>19 (5.83)</td><td>21 (13.70)</td><td>18 (12.16)</td><td>7 (2.08)</td><td>59 (14.71)</td></tr><tr><td>Median Follow-up</td><td>26.68</td><td>25.03</td><td>20.01</td><td>28.88</td><td>34.83</td></tr><tr><td>(months, 95% CI) Status</td><td>(22.67–29.57)</td><td>(20.53–30.98)</td><td>(14.85–25.20)</td><td>(25.07–35.06)</td><td>(31.67–37.95</td></tr><tr><td>Alive</td><td>194 (59.51)</td><td>120 (82.19)</td><td>86 (58.11)</td><td>216 (64.09)</td><td>391 (97.51)</td></tr><tr><td>Dead</td><td>132 (40.49)</td><td>26 (17.81)</td><td>62 (41.89)</td><td>121 (35.91)</td><td>10 (2.49)</td></tr><tr><td>Survival Rate (%)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>93.19</td><td>76.87</td><td></td><td></td></tr><tr><td>1-year</td><td>79.52</td><td></td><td></td><td>84.92</td><td>99.73</td></tr><tr><td>3-year</td><td>50.22</td><td>82.86</td><td>40.57</td><td>63.73</td><td>98.26</td></tr><tr><td>5-year</td><td>37.24</td><td>49.60</td><td>19.13</td><td>49.67</td><td>97.61</td></tr></table>

Data are n (%) or median (IQR). TCGA, The Cancer Genome Atlas. Stage is based on the 8th edition of AJCC staging.

The BV classifier achieved a mean AUC of $9 0 . 1 1 \pm 3 . 1 1 \%$ , with a sensitivity of $8 3 . 4 9 \pm 5 . 8 2 \%$ and a specificity of $8 1 . 6 0 \pm 5 . 3 6 \%$ . Across the five folds, true-positive rates ranged from 75.0% to 89.1%, while true-negative rates ranged from 75.4% to 85.6%. The relatively high mean FPR of $1 8 . 4 0 \pm 5 . 3 6 \%$ , particularly in Folds 3 and 4, indicates that nonvascular tissue structures with vessel-like morphology were more frequently classified as positive. Nevertheless, the classifier maintained a high NPV of $9 6 . 0 1 \pm 1 . 3 9 \%$ , supporting its utility for excluding patches unlikely to contain vascular structures.

Among the three tasks, the gland classifier demonstrated the strongest and most stable performance, achieving a mean AUC of $9 8 . 1 6 \pm 0 . 6 1 \%$ , sensitivity of $9 2 . 4 9 \pm 1 . 1 2 \%$ , and specificity of $9 5 . 6 4 \pm 0 . 8 0 \%$ . The confusion matrices showed consistently high true-positive rates of 91.2–94.0% and true-negative rates of 94.6– 96.6%, together with a low mean FPR of 4.36 ± 0.80%. Its PPV and NPV reached $9 1 . 8 1 \pm 5 . 4 7 \%$ and $9 4 . 8 2 \pm 3 . 2 6 \%$ , respectively, indicating reliable identification of both gland-containing and gland-absent patches. Collectively, these results demonstrate that the patch classifiers provided efective initial screening for all three FTU segmentation tasks, with gland classification showing the greatest stability, TLS classification being primarily afected by extreme target sparsity, and BV classification exhibiting greater susceptibility to morphologically similar background regions.

Table 3 External TCGA cohorts used for exploratory downstream application.
<table><tr><td>FTU Task</td><td>TCGA Cohort</td><td>Analysis Endpoint</td><td>Matched Patients</td><td>Events/Groups</td></tr><tr><td>TLS</td><td>READ</td><td>Overall survival</td><td>146</td><td>26 deaths</td></tr><tr><td>TLS</td><td>ESCA</td><td>Overall survival</td><td>148</td><td>62 deaths</td></tr><tr><td>TLS</td><td>STAD</td><td>Overall survival</td><td>326</td><td>132 deaths</td></tr><tr><td>BV</td><td>LIHC</td><td>Overall survival</td><td>337</td><td>121 deaths</td></tr><tr><td>BV</td><td>LIHC</td><td>MVI</td><td>337</td><td>100 positive vs. 237 negative</td></tr><tr><td>Gland</td><td>PRAD</td><td>Gleason score</td><td>401</td><td>151 high vs. 250 low</td></tr><tr><td>Gland</td><td>PRAD</td><td>Pathological T stage</td><td>395</td><td>239 T3/T4 vs. 156 T2</td></tr><tr><td>Gland</td><td>PRAD</td><td>Surgical margin</td><td>372</td><td>115 R1 vs. 257 R0</td></tr><tr><td>Gland</td><td>PRAD</td><td>BCR status</td><td>342</td><td>301 non-recurrent</td></tr></table>

Abbreviations: BCR, biochemical recurrence; BV, blood vessel; ESCA, esophageal carcinoma; LIHC, liver hepatocellular carcinoma; FTU, functional tissue unit; PRAD, prostate adenocarcinoma; READ, rectum adenocarcinoma; STAD, stomach adenocarcinoma; TCGA, The Cancer Genome Atlas; TLS, tertiary lymphoid structure.

Table 4 Validation-best patch-level classifier performance across five-fold cross-validation. AUC is the primary metric. Values are percentages.
<table><tr><td>Cohort</td><td>Fold</td><td>AUC</td><td>Sensitivity</td><td>Specificity</td><td>FPR</td><td>NPV</td><td>PPV</td></tr><tr><td rowspan="6">TLS</td><td>0</td><td>92.02</td><td>88.04</td><td>81.94</td><td>18.06</td><td>99.72</td><td>8.52</td></tr><tr><td>1</td><td>95.70</td><td>87.47</td><td>91.63</td><td>8.37</td><td>99.28</td><td>35.79</td></tr><tr><td>2</td><td>97.97</td><td>90.77</td><td>97.05</td><td>2.95</td><td>99.37</td><td>67.14</td></tr><tr><td>3</td><td>99.19</td><td>94.44</td><td>98.27</td><td>1.73</td><td>99.87</td><td>55.43</td></tr><tr><td>4</td><td>94.70</td><td>81.56</td><td>91.16</td><td>8.84</td><td>99.16</td><td>27.98</td></tr><tr><td>Mean ± SD</td><td>95.92 ± 2.81</td><td>88.46 ± 4.74</td><td>92.01 ± 6.46</td><td>7.99 ± 6.46</td><td>99.48 ± 0.30</td><td>38.97 ± 23.04</td></tr><tr><td rowspan="6">BV</td><td>0</td><td>87.50</td><td>75.00</td><td>85.62</td><td>14.38</td><td>93.93</td><td>53.60</td></tr><tr><td>1</td><td>91.83</td><td>82.50</td><td>85.36</td><td>14.64</td><td>97.59</td><td>40.39</td></tr><tr><td>2</td><td>94.28</td><td>88.83</td><td>85.53</td><td>14.47</td><td>96.80</td><td>60.83</td></tr><tr><td>3</td><td>90.26</td><td>89.12</td><td>76.12</td><td>23.88</td><td>96.19</td><td>50.84</td></tr><tr><td>4</td><td>86.70</td><td>81.99</td><td>75.35</td><td>24.65</td><td>95.55</td><td>39.32</td></tr><tr><td>Mean ± SD</td><td>90.11 ± 3.11</td><td>83.49 ± 5.82</td><td>81.60 ± 5.36</td><td>18.40 ± 5.36</td><td>96.01 ± 1.39</td><td>49.00 ± 9.11</td></tr><tr><td rowspan="7">Gland</td><td>0</td><td>98.75</td><td>92.98</td><td>96.64</td><td>3.36</td><td>94.30</td><td>95.84</td></tr><tr><td>1</td><td>98.25</td><td>92.64</td><td>96.01</td><td>3.99</td><td>95.29</td><td>93.74</td></tr><tr><td>2</td><td>97.66</td><td>91.20</td><td>95.15</td><td>4.85</td><td>97.71</td><td>82.64</td></tr><tr><td>3</td><td>98.73</td><td>94.03</td><td>95.83</td><td>4.17</td><td>97.24</td><td>91.13</td></tr><tr><td>4</td><td>97.41</td><td>91.62</td><td>94.56</td><td>5.44</td><td>89.54</td><td>95.69</td></tr><tr><td>Mean ± SD</td><td>98.16 ± 0.61</td><td>92.49 ± 1.12</td><td>95.64 ± 0.80</td><td>4.36 ± 0.80</td><td>94.82 ± 3.26</td><td>91.81 ± 5.47</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Validation-best patch classifier confusion matrices

![](images/d7ce28167277d0dd565f8b61a7ec4e51150bf42bf7b411cc3c2219d0a59c6456.jpg)  
Fig. 6 Confusion matrices detailing the patch-level classification performance of FTUs across 5- fold cross-validation. The confusion matrices illustrate the validation-best prediction results for three independent FTU segmentation cohorts: tertiary lymphoid structure (TLS, top row), blood vessel (BV, middle row), and gland (bottom row). Columns correspond to the validation set results from Fold 0 through Fold 4, demonstrating the robust generalization of the deep learning framework. Within each matrix, the absolute counts of image patches are presented alongside the row-normalized proportions, detailing the true positive rate (sensitivity) and true negative rate (specificity). The color intensity of the heatmap corresponds to these row-normalized percentages. The overall area under the receiver operating characteristic curve (AUC) achieved for each specific fold and classification task is annotated above the respective subplot.

As shown in Figure 7, the predicted probability distributions revealed distinct task-specific patterns of class separability across the five validation folds. For TLS, FTU-absent patches were predominantly concentrated near a predicted probability of 0, whereas TLS-containing patches generally accumulated near 1.0. This pronounced bimodal separation was particularly evident in Folds 2 and 3, which achieved AUCs of 97.97% and 99.19%, respectively. Nevertheless, a subset of TLS-containing patches received low predicted probabilities in several folds, resulting in relatively low foldspecific decision thresholds ranging from below 0.01 to 0.11. This pattern is consistent with the extreme rarity and heterogeneous appearance of TLS-containing patches.

The BV classifiers exhibited substantially greater overlap between the positive and negative probability distributions. Although most BV-absent patches remained concentrated toward low probabilities, BV-containing patches were broadly distributed across the full probability range, particularly in Folds 2–4. The fold-specific thresholds consequently varied from 0.06 to 0.36. This less distinct separation agrees with the lower AUC and higher false-positive rates observed for BV classification, suggesting that vessel-containing patches were more dificult to distinguish from morphologically similar background tissue.

In contrast, the gland classifiers demonstrated the clearest and most consistent separation between classes. Across all folds, gland-absent patches were sharply concentrated near 0, while gland-containing patches formed prominent peaks near 1.0, with only limited overlap between the two distributions. This pattern was reflected in the consistently high fold-wise AUCs of 97.41–98.75%. Although the selected decision thresholds varied from 0.26 to 0.90, the strong separation of the two distributions resulted in stable sensitivity and specificity across folds. Overall, the probability distributions corroborate the quantitative results, demonstrating strong class separability for gland and TLS classification, while highlighting the greater ambiguity of the BV task.

![](images/8f8e46f299d1fbe1e5b879f9cea180ec6a127d6be71f7e5909ee1d94cea25c82.jpg)  
Fig. 7 Validation-best predicted probability distributions of the patch-level classifiers across fivefold cross-validation. Columns correspond to the TLS, BV, and gland cohorts, whereas rows represent Fold 0 through Fold 4. Each panel shows the normalized density distributions of the predicted FTUcontaining probabilities for annotated FTU-containing patches (orange; True 1) and FTU-absent patches (teal; True 0); n indicates the number of patches in each class. The black dashed vertical line indicates the fold-specific decision threshold, whose numerical value is provided within each panel. The validation AUC obtained from the best-performing checkpoint is reported in the corresponding panel title.

Representative hard-negative examples are shown in Figure 8. TLS hard negatives often contained dense inflammatory or stromal regions adjacent to adipose or gland-like spaces, BV hard negatives included lumen-like or collagen-rich structures, and gland hard negatives contained epithelial or stromal patterns resembling gland boundaries. These examples support the use of classifier-derived probability scores to prioritize FTU-absent patches that are visually similar to target-containing regions.

![](images/88fbf5247b75071066db84087dbef3305e4c113b5dd52d3f05b72563a3facba9.jpg)  
Fig. 8 Representative classifier-ranked FTU-absent patches across predicted probability gradients. For each cohort, two FTU-absent tissue patches are shown at each approximate predicted FTU-containing probability level. Patch borders indicate classification correctness according to the validation-best cutof of the corresponding fold: green denotes correctly classified negative patches (TN), whereas red denotes false-positive patches (FP) misclassified as FTU-containing. The examples illustrate how the patch-level classifier assigns high probabilities to morphologically confusing background regions, which are subsequently used as hard negatives for segmentation training.

## 3.3 Segmentation Performance and Training-Data Workload

Tables 5 to 7 summarize the segmentation accuracy and training workload of diferent patch construction strategies. Overall, positive-only training produced the weakest or near-weakest performance, indicating that FTU-absent tissue is necessary for suppressing false-positive predictions. All-tissue training provided the strongest reference performance in most settings but required the complete tissue-patch set. In contrast, classifier-guided FP-find TopK sampling retained only a fraction of the training data while preserving much of the independent-test performance.

For TLS segmentation, positive-only training used 4.4% of the all-tissue workload but achieved only 73.55 ± 4.61% independent-test Dice. Random-balanced training improved the test Dice to 82.25±2.11% with 8.8% workload, whereas all-tissue training reached 84.85 ± 1.60%. FTU-Seek TopK sampling matched or exceeded the all-tissue test performance at three of the four evaluated budgets while using substantially fewer training patches. FTU-Seek Top100 achieved $8 5 . 0 8 \pm 1 . 0 7 \%$ Dice using 6.3% of the patches, and FTU-Seek Top500 achieved the highest independent-test Dice of $8 5 . 1 4 \pm 1 . 6 1 \%$ using 15.8% of the workload. Compared with matched random TopK sampling, FTU-Seek improved the internal-test Dice by 5.22, 3.81, 3.48, and 1.20 percentage points at Top100, Top300, Top500, and Top1000, respectively. The corresponding paired slide-level comparisons on the independent held-out TLS cohort are reported separately in Table 9.

Table 5 TLS Dice performance (%) across hard-negative budgets and evaluation cohorts.
<table><tr><td rowspan="2"></td><td colspan="2">Development-Set Validation</td><td colspan="2">Internal Test</td><td colspan="2">Held-Out TLS Cohort</td></tr><tr><td>FTU-Seek</td><td>Random (5 Seeds)</td><td>FTU-Seek</td><td>Random (5 Seeds)</td><td>FTU-Seek</td><td>Random (5 Seeds)</td></tr><tr><td>100</td><td>68.75</td><td> $6 6 . 6 2 \pm 3 . 0 3$ </td><td>85.08</td><td> $7 9 . 8 6 \pm 3 . 4 5$ </td><td>73.99</td><td> $6 8 . 8 7 \pm 1 . 9 5$ </td></tr><tr><td>300</td><td>73.64</td><td> $6 8 . 8 2 \pm 1 . 6 3$ </td><td>84.69</td><td> $8 0 . 8 8 \pm 1 . 5 5$ </td><td>76.54</td><td> $6 9 . 9 2 \pm 2 . 7 4$ </td></tr><tr><td>500</td><td>72.63</td><td> $7 0 . 2 8 \pm 0 . 7 9$ </td><td>85.14</td><td> $8 1 . 6 6 \pm 1 . 2 4$ </td><td>75.19</td><td> $7 0 . 3 3 \pm 2 . 0 0$ </td></tr><tr><td>1000</td><td>76.91</td><td> $7 3 . 5 4 \pm 2 . 2 5$ </td><td>85.12</td><td> $8 3 . 9 2 \pm 0 . 5 4$ </td><td>76.69</td><td> $7 3 . 4 8 \pm 0 . 6 2$ </td></tr></table>

For each random seed, the model was trained using five-fold cross-validation. The fold-level performance estimates were first averaged within each seed to obtain a seed-level mean, and the final Random value was then calculated as the mean across five random seeds. Thus, the Random entries summarize perfor mance hierarchically across five folds and five seeds. For the compact budget comparison, FTU-Seek entries correspond to the original seed-1 experiments and are reported as fold-mean performance; the fold-level variability of the representative Top1000 configuration is reported in Table 8. For the held-out TLS cohort, each fold-specific model was evaluated on the same 30 held-out $\mathrm { W S I s } ,$ and the resulting performance estimates were summarized within folds, then across folds within each seed, and finally across the five random seeds. These values therefore represent aggregate performance summaries and are distinct from the WSIlevel paired analyses reported separately in Table 9.

Table 6 Comparison of representative training-set construction strategies for blood-vessel and gland segmentation. Values are Dice coeficients (%, mean ± SD) across five folds.
<table><tr><td>Task</td><td>Strategy</td><td>Workload (%)</td><td>Validation Dice</td><td>Internal-Test Dice</td></tr><tr><td>BV</td><td>Positive-only</td><td>16.4</td><td> $4 7 . 8 0 \pm 4 . 8 0$ </td><td> $3 2 . 1 4 \pm 3 . 9 8$ </td></tr><tr><td></td><td>Random-balanced (1:1)</td><td>32.8</td><td> $5 7 . 8 4 \pm 4 . 0 6$ </td><td> $4 8 . 5 3 \pm 8 . 1 6$ </td></tr><tr><td></td><td>All-tissue</td><td>100.0</td><td> $6 2 . 4 6 \pm 6 . 6 3$ </td><td> $6 0 . 3 8 \pm 9 . 7 6$ </td></tr><tr><td></td><td>FTU-Seek</td><td>50.0</td><td> $6 0 . 1 1 \pm 7 . 4 3$ </td><td> $5 7 . 6 8 \pm 6 . 8 8$ </td></tr><tr><td>Gland</td><td>Positive-only</td><td>36.9</td><td> $8 0 . 7 8 \pm 1 1 . 3 0$ </td><td> $8 0 . 0 1 \pm 2 . 4 6$ </td></tr><tr><td></td><td>Random-balanced (1:1)</td><td>73.78</td><td> $8 7 . 9 3 \pm 4 . 5 1$ </td><td> $8 5 . 3 9 \pm 0 . 5 7$ </td></tr><tr><td></td><td>All-tissue</td><td>100.0</td><td> $8 9 . 2 4 \pm 1 . 6 1$ </td><td> $8 6 . 5 7 \pm 0 . 6 9$ </td></tr><tr><td></td><td>FTU-Seek</td><td>67.2</td><td> $8 8 . 4 8 \pm 2 . 8 9$ </td><td> $8 6 . 3 3 \pm 0 . 2 7$ </td></tr></table>

Random-balanced training used an equal number of randomly selected negative patches and positive patches. FTU-Seek Top1000 was used as a representative operating point for cross-task comparison. The same candidate TopK values were evaluated across the FTU categories to characterize the accuracy– workload trade-of under matched sampling budgets; these values were exploratory operating points and were not selected using a prevalence-adaptive rule or intended to define a universally optimal budget. Nei ther the BV nor the gland cohort included a held-out cohort; therefore, only development-set validation and internal-test results are reported. Workload denotes the proportion of retained training patches relative to all-tissue training.

Table 7 Segmentation performance across hard-negative budgets for blood-vessel and gland segmentation. FTU-Seek values are reported as mean ± SD across five folds, whereas Random values are summarized as mean ± SD across five random seeds, with each seed evaluated using five-fold models.
<table><tr><td>Task</td><td>K</td><td>FTU-Seek Validation</td><td>Random Validation (5 Seeds)</td><td>FTU-Seek Internal</td><td>Random Internal (5 Seeds)</td></tr><tr><td>BV</td><td>100</td><td> $5 1 . 5 0 \pm 6 . 9 2$ </td><td> $5 3 . 2 5 \pm 4 . 6 6$ </td><td> $4 1 . 1 4 \pm 1 0 . 3 2$ </td><td> $3 7 . 1 6 \pm 3 . 8 5$ </td></tr><tr><td></td><td>300</td><td> $5 7 . 0 3 \pm 5 . 3 5$ </td><td> $5 8 . 6 3 \pm 3 . 2 9$ </td><td> $4 7 . 9 3 \pm 1 1 . 8 0$ </td><td> $4 7 . 2 3 \pm 6 . 7 8$ </td></tr><tr><td></td><td>500</td><td> $5 8 . 4 1 \pm 6 . 7 6$ </td><td> $5 8 . 4 5 \pm 2 . 9 7$ </td><td> $5 0 . 5 9 \pm 8 . 4 1$ </td><td> $4 7 . 6 3 \pm 1 1 . 6 2$ </td></tr><tr><td></td><td>1000</td><td> $6 0 . 1 1 \pm 7 . 4 3$ </td><td> $6 0 . 9 5 \pm 7 . 2 5$ </td><td> $5 7 . 6 8 \pm 6 . 8 8$ </td><td> $5 4 . 2 1 \pm 6 . 1 7$ </td></tr><tr><td>Gland</td><td>100</td><td> $8 4 . 8 1 \pm 9 . 3 7$ </td><td> $8 7 . 3 2 \pm 4 . 1 2$ </td><td> $8 3 . 0 1 \pm 2 . 4 4$ </td><td> $8 4 . 4 3 \pm 0 . 5 3$ </td></tr><tr><td></td><td>300</td><td> $8 7 . 7 8 \pm 3 . 3 8$ </td><td> $8 7 . 0 6 \pm 3 . 9 0$ </td><td> $8 5 . 1 3 \pm 0 . 3 8$ </td><td> $8 4 . 7 0 \pm 0 . 9 8$ </td></tr><tr><td></td><td>500</td><td> $8 7 . 4 6 \pm 5 . 2 7$ </td><td> $8 7 . 7 7 \pm 4 . 1 9$ </td><td> $8 5 . 9 0 \pm 0 . 9 1$ </td><td> $8 5 . 4 7 \pm 1 . 0 3$ </td></tr><tr><td></td><td>1000</td><td> $8 8 . 4 8 \pm 2 . 8 9$ </td><td> $8 8 . 5 9 \pm 2 . 3 6$ </td><td> $8 6 . 3 3 \pm 0 . 2 7$ </td><td> $8 6 . 4 9 \pm 0 . 8 2$ </td></tr></table>

Random denotes random selection of the same number of target-absent patches per WSI as the corresponding FTU-Seek TopK configuration. Neither the BV nor the gland cohort included a held-out cohort; therefore, only development-set validation and internal-test results are reported.

Table 8 Comparison of representative TLS training-set construction strategies.
<table><tr><td>Strategy</td><td>Workload (%)</td><td>Development-Set Validation</td><td>Internal Test</td><td>Held-Out TLS Cohort</td></tr><tr><td>Positive-only</td><td>4.4</td><td> $5 7 . 2 0 \pm 2 5 . 2 3$ </td><td> $7 3 . 5 5 \pm 4 . 6 1$ </td><td> $6 1 . 9 0 \pm 2 0 . 4 1$ </td></tr><tr><td>Random-balanced (1:1)</td><td>8.8</td><td> $7 1 . 5 0 \pm 1 2 . 5 5$ </td><td> $8 2 . 2 5 \pm 2 . 1 1$ </td><td> $7 0 . 8 3 \pm 1 6 . 2 3$ </td></tr><tr><td>All-tissue</td><td>100.0</td><td> $7 1 . 6 3 \pm 1 7 . 7 9$ </td><td> $8 4 . 8 5 \pm 1 . 6 0$ </td><td> $7 5 . 1 2 \pm 1 2 . 6 7$ </td></tr><tr><td>FTU-Seek</td><td>27.6</td><td> $7 6 . 9 1 \pm 1 0 . 1 4$ </td><td> $8 5 . 1 2 \pm 2 . 0 8$ </td><td> $7 6 . 6 9 \pm 1 1 . 8 9$ </td></tr></table>

Random-balanced training used all positive patches and an equal number of negative patches randomly sampled from the target-absent patch pool, thereby maintaining a 1:1 positive-to-negative patch ratio. For the representative-strategy comparison, FTU-Seek used Top1000 as a conservative operating point for subsequent downstream analyses. The diferent K values were explored to characterize the accuracy– workload trade-of, rather than to define a universally optimal budget. The internal test and held-out TLS cohort were not used for configuration selection. Workload denotes the proportion of retained training patches relative to all-tissue training.

BV segmentation showed a stronger dependence on negative-patch coverage. Positive-only training achieved only 32.14±3.98% test Dice, whereas all-tissue training achieved the best performance (60.38±9.76%). FTU-Seek performance increased as the hard-negative budget expanded, rising from 41.14±10.32% at Top100 to 57.68±6.88% at Top1000. The latter used 50.0% of the all-tissue workload and approached all-tissue performance while halving the retained training patches. FTU-Seek also outperformed matched random TopK sampling at all BV budgets, with test Dice gains of 3.98, 0.70, 2.96, and 3.47 percentage points from Top100 to Top1000.

Table 9 Paired slide-level comparisons between FTU-Seek and alternative training-set construction strategies on the held-out TLS cohort.
<table><tr><td>Comparison</td><td>FTU-Seek Dice</td><td>Comparator Dice</td><td>Mean Δ</td><td>95% CI</td></tr><tr><td>FTU-Seek Top100 vs. Random Top100</td><td> $7 3 . 9 9 \pm 1 3 . 8 7$ </td><td> $7 1 . 4 1 \pm 1 5 . 4 3$ </td><td>+2.58</td><td>[1.14, 4.02]</td></tr><tr><td>FTU-Seek Top300 vs. Random Top300</td><td> $7 6 . 5 4 \pm 1 3 . 0 3$ </td><td> $6 7 . 5 9 \pm 1 8 . 7 4$ </td><td>+8.96</td><td>[5.61, 12.31]</td></tr><tr><td>FTU-Seek Top500 vs. Random Top500</td><td> $7 5 . 1 9 \pm 1 3 . 5 4$ </td><td> $7 0 . 4 8 \pm 1 6 . 8 4$ </td><td>+4.72</td><td>[3.06, 6.38]</td></tr><tr><td>FTU-Seek Top1000 vs. Random Top1000</td><td> $7 6 . 6 9 \pm 1 1 . 8 9$ </td><td> $7 2 . 9 7 \pm 1 5 . 3 2$ </td><td>+3.72</td><td>[2.05, 5.38]</td></tr><tr><td>FTU-Seek Top1000 vs. All-tissue</td><td> $7 6 . 6 9 \pm 1 1 . 8 9$ </td><td> $7 5 . 1 2 \pm 1 2 . 6 7$ </td><td>+1.57</td><td>[0.86, 2.28]</td></tr><tr><td>FTU-Seek Top1000 vs. Random-balanced (1:1)</td><td> $7 6 . 6 9 \pm 1 1 . 8 9$ </td><td> $7 0 . 8 3 \pm 1 6 . 2 3$ </td><td>+5.86</td><td>[3.64, 8.07]</td></tr></table>

For the held-out TLS cohort, Table 5 reports aggregate performance summaries across cross-validation folds and random seeds, whereas the present table reports WSI-level paired analyses. Specifically, Dice was calculated separately for each of the 30 held-out WSIs, with each WSI serving as the statistical pairing unit. Therefore, the present analysis estimates within-slide performance diferences rather than reproducing the aggregate performance summaries reported in Table 5, and the corresponding mean values are not expected to be numerically identical.

For gland segmentation, all strategies achieved relatively high Dice scores, consistent with the greater abundance and more distinct morphology of glandular structures. All-tissue training remained the strongest baseline $( 8 6 . 5 7 \pm 0 . 6 9 \%$ test Dice), but reduced-workload strategies retained most of this performance. FTU-Seek Top300, Top500, and Top1000 achieved 85.13±0.38%, 85.90±0.91%, and 86.33±0.27%, respectively. FP-find Top1000 was within 0.24 percentage points of all-tissue training while reducing the workload to 67.2%. However, FP-find did not consistently outperform matched random TopK sampling across gland budgets, with random sampling performing comparably or better at several settings. These results indicate that FP-find ofers no clear overall advantage in gland segmentation, likely because gland-containing patches are relatively abundant and morphologically easier to distinguish.

As summarized in Tables 5 to 7, the value of hard-negative mining was taskdependent. TLS benefited most from classifier-guided selection at small workloads, BV required broader negative coverage to approach all-tissue performance, and gland segmentation was comparatively insensitive to the negative-sampling strategy. Overall, FTU-Seek provided the clearest advantage for sparse or morphologically ambiguous FTUs, where randomly selected background patches are less likely to capture dificult false-positive regions.

## 3.4 Multi-Seed Stability of Random-Sampling Baselines

We next examined whether the performance of the random-sampling baselines depended strongly on a single random draw. Across TLS, blood-vessel, and gland segmentation, each random-sampling configuration was repeated using four additional seeds, resulting in five seeds in total when the original seed was included. The repeated runs showed limited-to-moderate variability across random seeds, with the magnitude depending on the FTU category and negative-patch budget. For TLS segmentation, the random Top100, Top300, Top500, and Top1000 baselines achieved mean Dice values of 79.86 ± 3.45%, 80.88 ± 1.55%, 81.66 ± 1.24%, and 83.92 ± 0.54%, respectively, on the internal two-WSI test cohort. The corresponding FP-find results from the original seed-1 experiments were 85.08%, 84.69%, 85.14%, and 85.12%. Although these comparisons should be interpreted cautiously because the internal test cohort contained only two WSIs, the FP-find results remained above the random-baseline averages at all four budgets.

The benefit was less pronounced for the denser FTU tasks. For blood-vessel segmentation, the diferences between FP-find and the random-baseline averages were +3.98, +0.70, +2.96, and +3.47 percentage points from Top100 to Top1000, respectively. For gland segmentation, the corresponding diferences were −1.42, +0.43, +0.43, and −0.16 percentage points. These results are consistent with the taskdependent behavior observed in the main experiments: random negative sampling may be suficient when target-containing patches are relatively abundant and morphologically distinct, whereas classifier-guided selection is more advantageous for sparse or morphologically ambiguous targets.

## 3.5 Evaluation on an Independent Held-Out TLS Cohort

To determine whether the observed performance diferences were preserved beyond the original two-slide internal test set, we evaluated all 11 segmentation strategies on an independent held-out cohort of 30 TLS WSIs from the same institution. The cohort included 30 patients and 30 WSIs, with 1184 annotated TLS instances, 123,484 tissue patches, and a TLS pixel-area ratio of 1.68% (Table 10). This cohort was not used during training, model selection, TopK selection, or threshold tuning. The evaluated strategies included positive-only training, all-tissue training, random-negative sampling, random TopK sampling, and classifier-guided FP-find TopK sampling.

On the held-out TLS cohort, FTU-Seek achieved slide-level Dice values of 73.99%, 76.54%, 75.19%, and 76.69% at Top100, Top300, Top500, and Top1000, respectively. The corresponding matched random TopK strategies achieved 71.41%, 67.59%, 70.48%, and 72.97%, respectively. The largest paired improvement was observed at Top300, whereas Top1000 achieved the highest absolute FTU-Seek Dice.

Based on the development-set analysis, Top1000 was retained as a pre-specified representative operating point for comparisons with all-tissue and random-balanced training. The held-out cohort was not used for selecting the operating point or tuning the segmentation procedure. Formal paired comparisons across the 30 held-out WSIs are reported in Table 9.

## 3.6 Paired Slide-Level Analysis on the Held-Out TLS Cohort

To assess whether the observed benefit of FTU-Seek was preserved in an independent held-out cohort, we performed paired slide-level analyses on the held-out TLS cohort. The same 30 WSIs were evaluated using FTU-Seek and the corresponding comparator

Table 10 Characteristics and spatial sparsity of the independent held-out TLS cohort.
<table><tr><td>Characteristic</td><td>Value</td></tr><tr><td>Patients</td><td>30</td></tr><tr><td>WSIs</td><td>30</td></tr><tr><td>Total TLS instances</td><td>1184</td></tr><tr><td>Mean TLS instances per WSI</td><td>39.47</td></tr><tr><td>Median TLS instances per WSI</td><td>33.00</td></tr><tr><td>Total tissue patches</td><td>123,484</td></tr><tr><td>TLS-containing patches</td><td>9231</td></tr><tr><td>TLS-absent patches</td><td>114,253</td></tr><tr><td>TLS-containing patch ratio (%)</td><td>7.48</td></tr><tr><td>Tissue pixel area</td><td>525,755,637</td></tr><tr><td>TLS pixel area</td><td>8,822,896</td></tr><tr><td>TLS pixel area ratio (%)</td><td>1.68</td></tr></table>

The cohort contained one WSI per patient. Tissue patches were extracted at 1 µm/pixel with a nominal patch size of 256 × 256 pixels before resizing to the model input resolution. TLS-containing patches overlapped at least one annotated TLS region, whereas TLS-absent patches did not overlap an annotated TLS region. Pixel areas were calculated from the corresponding tissue and TLS annotation masks.

strategy. For each WSI, the paired Dice diference was calculated as

$$
\Delta _ { i } = \mathrm { D i c e } _ { \mathrm { F T U - S e e k } , i } - \mathrm { D i c e } _ { \mathrm { C o m p a r a t o r } , i } ,
$$

where a positive value indicates better performance of FTU-Seek. Each model prediction was generated using a five-fold probability ensemble, in which the foreground probabilities from the five fold models were averaged before thresholding. The mean paired diference, its two-sided 95% confidence interval, paired t-test, and Wilcoxon signed-rank test were calculated across the 30 WSIs. The internal test cohort contained only two WSIs and was therefore excluded from formal slide-level hypothesis testing and retained for descriptive comparison only. The paired comparisons included matched random TopK sampling at all four budgets, as well as the representative Top1000 comparisons with all-tissue and random-balanced training.

Across all evaluated hard-negative budgets, FTU-Seek achieved higher slide-level Dice than matched random TopK sampling on the held-out TLS cohort. The mean paired improvements ranged from 2.58 to 8.96 percentage points, and all corresponding 95% confidence intervals excluded zero. At the representative Top1000 operating point, FTU-Seek also showed positive paired diferences relative to all-tissue training $( \Delta = + 1 . 5 7$ percentage points) and random-balanced training $( \Delta = + 5 . 8 6$ percentage points). These results provide independent support for the generalization of FTU-Seek beyond the development data. The largest improvement over matched random sampling was observed at Top300, whereas Top1000 achieved the highest absolute FTU-Seek Dice.

## 3.7 Qualitative Analysis

Figure 9 presents representative whole-slide segmentation results for the TLS, gland, and BV cohorts, together with magnified regions for detailed comparison. Across the three tasks, FTU-Seek produced segmentation patterns that were generally more consistent with the ground-truth annotations than those obtained using random negative sampling. The diferences were most evident in regions containing sparse targets, morphologically ambiguous background structures, or complex boundaries.

For TLS segmentation, both methods identified the major TLS regions distributed across the tissue section. However, random negative sampling generated additional scattered predictions in background regions, particularly within the enlarged region, whereas FTU-Seek more efectively suppressed these isolated false-positive detections. At the same time, the principal TLS foci were retained, indicating that classifier-guided hard-negative selection improved background discrimination without substantially reducing sensitivity to sparse target regions.

![](images/07fcd161394a24b0e4442ef5229a7be0853f8a80bc39eb43e3cbf859d5e85eb3.jpg)  
Fig. 9 Visual comparison of segmentation performance across the three FTU cohorts. Representative results are shown for TLS, gland, and blood vessel segmentation using classifier-guided FTU-Seek and matched random negative sampling. Red boxes indicate the magnified regions used to highlight false-positive predictions and boundary discrepancies. GT, ground truth.

For gland segmentation, the overall spatial distribution of glandular tissue was captured by both approaches. Nevertheless, random negative sampling produced a visibly denser and more fragmented prediction pattern in the magnified region, including responses extending into non-glandular tissue. In comparison, FTU-Seek yielded a cleaner segmentation map with fewer spurious structures and a distribution more closely aligned with the annotated gland regions. These observations are consistent with its ability to prioritize morphologically confusing negative patches during training.

The distinction between the two strategies was also apparent in BV segmentation. Although both methods detected major vascular structures, random negative sampling produced several elongated or punctate false-positive regions in the background. FTU-Seek suppressed many of these vessel-like confounders while preserving the principal annotated vessels. This behavior is particularly important for BV segmentation, where tissue clefts, elongated stromal structures, and other linear components may resemble vascular profiles.

Overall, the qualitative results support the quantitative findings in Tables 5 to 7 and Figure 9. Compared with random negative sampling, FTU-Seek primarily improved segmentation by reducing false-positive predictions and refining target boundaries, rather than by simply increasing the total predicted area. The benefit was especially evident for sparse or morphologically ambiguous FTUs, demonstrating that classifier-guided hard-negative sampling provides more informative background examples for segmentation model training.

## 3.8 Proof-of-Concept Downstream Phenotypic Analyses Enabled by FTU-Seek Segmentation

To illustrate the utility of FTU-Seek beyond segmentation performance, we converted segmentation masks from external TCGA cohorts into quantitative FTU phenotypes and examined their relationships with selected biological and clinicopathological characteristics. These proof-of-concept analyses were exploratory and were intended to demonstrate that FTU-Seek outputs can support downstream phenotyping and hypothesis generation, rather than to develop or validate clinical prediction models.

For TLSs, segmentation-derived phenotypes were quantified in TCGA-READ, TCGA-ESCA, and TCGA-STAD. Matched survival analyses included 146 READ, 148 ESCA, and 326 STAD patients after merging available WSI-derived TLS features with clinical outcomes. A total of 60 TLS features captured complementary aspects of immune organization, including TLS burden, size composition, hotspot density, and nearest-neighbor clustering. Across cohorts, mean TLS density ranged from 0.0048 in READ to 0.0116 in STAD, indicating substantial inter-cohort heterogeneity in TLS abundance (Figure 10A). Furthermore, TLS features exhibited significant heterogeneity across the diferent cohorts (Figure 10A). Specifically, the median instance-level TLS area in the READ cohort was significantly smaller than that in the other cohorts $( p = 0 . 0 1 0 )$ . Conversely, TLSs in the STAD cohort demonstrated higher solidity (p $= 0 . 0 0 1 )$ . In morphological analysis, solidity is defined as the ratio of the actual TLS area to the area of its convex hull, essentially reflecting a more compact structure with smoother, less irregular boundaries. Regarding spatial distribution, there were no significant diferences across the three cohorts in the nearest neighbor (NN) cluster index. This spatial metric quantifies the degree of TLS clustering by comparing the observed average distance between adjacent TLSs against the expected distance under a completely random spatial distribution. Specifically, in READ, a 1-SD increase in TLS density was associated with a 51% reduction in the risk of death $( \mathrm { H R } = 0 . 4 9 , 9 5 \% \mathrm { C I } { } ;$ 0.26–0.92; nominal $p = 0 . 0 2 6 ;$ FDR-adjusted $p = 0 . 2 5 )$ , whereas higher NN cluster index values were associated with a 60% lower risk (HR = 0.40, 95% CI: 0.17–0.95; nominal $p = 0 . 0 3 9$ ; FDR-adjusted $p = 0 . 2 5 )$ . Neither association met the exploratory

FDR threshold. For illustrative exploratory visualization, Kaplan–Meier curves were additionally generated for selected TLS phenotypes using data-derived cut points. These included TLS density and NN cluster index in ${ \mathrm { R E A D } } ,$ median TLS solidity in ESCA, and mean TLS area in STAD (Figure 10B). The resulting group separations were used solely to illustrate potential survival stratification and were not interpreted as validated or multiplicity-adjusted prognostic associations.

For blood vessels, FTU-Seek-derived vascular phenotypes were quantified in TCGA-LIHC. The LIHC cohort included 337 patients and 121 death events. Among the 36 vascular phenotypes evaluated, higher BV count density showed a potential adverse association with OS (log-rank $p = 0 . 0 2 8 ;$ Figure 11A). In the multivariable Cox analysis, the high-BV-density group showed a higher mortality risk than the low-density group after adjustment for age, clinical stage, Child–Pugh class, and HBV status (adjusted HR = 1.70, 95% CI: 1.10–2.59; $p = 0 . 0 1 6 ;$ Figure 11B). The proportional-hazards diagnostic indicated evidence of a time-varying association, and the HR was therefore interpreted as an exploratory overall summary. No substantial multicollinearity was observed among the included predictors (all $\mathrm { V I F s } < 1 . 1 5 )$ Furthermore, we observed significant diferences in the distribution of computationally extracted vascular features—including BV area (nominal $p = 0 . 0 3 2 \mathrm { : }$ FDR-adjusted $p = 0 . 0 8 7 )$ , large BV area (nominal $p = 0 . 0 2 5$ ; FDR-adjusted $p = 0 . 0 8 7 )$ , hotspot top mean BV density (nominal $p = 0 . 0 1 7 ;$ FDR-adjusted $p = 0 . 0 8 7 )$ , and hotspot max BV density (nominal $p = 0 . 0 1 1 ;$ FDR-adjusted $p = 0 . 0 8 7 )$ —between patients with and without MVI (Figure 11C). Notably, in the multivariable logistic regression analysis, each 1-SD increase in hotspot top mean BV density—the mean BV density within the top decile of vascular hotspots—was associated with higher odds of MVI (adjusted OR = 1.32, 95% CI: 1.04–1.66; p = 0.020; Figure 11D). RCS analysis further supported an overall association between hotspot top mean BV density and MVI $( p _ { \mathrm { o v e r a l l } } = 0 . 0 0 1 )$ , with evidence of nonlinearity $( p _ { \mathrm { n o n l i n e a r i t y } } = 0 . 0 1 5 )$ and an overall trend toward higher odds of MVI at greater BV density. Overall, these analyses illustrate how FTU-Seekderived vascular phenotypes can be used to explore potential relationships between tumor-associated vascular organization, mortality risk, and MVI.

A  
![](images/6d43b04305c207f53c49416d0efc6c753f4d87eef72804a9426d6aa6a4126787.jpg)

![](images/01eb10b2345718ec073b2a47ac00f5a2a79f21a28d988cc40b034ce1d394b25f.jpg)

![](images/342deff14ecbc656bbc4c1d0a019df89e21b68f58a2400f5901ab3cfdf25ac7d.jpg)

![](images/b6f48d78009047a28a192aa063bf0613ee6e26d08d623f179ae2902f041bb09a.jpg)

B  
![](images/022725c9362a055cad0bf10726771e33f646f79e3eb83bd01ef9eae97c432fef.jpg)

![](images/1e1a1b6f2e9c99d4a9418a1f7855ab65c10e6dbbb845daf26098d6c9453977a5.jpg)

![](images/f6cea3588328e8be6ad614497ef28160285d01e1e173a8fb524018bb97def831.jpg)

![](images/ffd59f67a47fa986469c0957262d6020b460f140397785d3333a5e79592ab913.jpg)  
Fig. 10 Exploratory characterization of TLS phenotypes and survival associations in TCGA cohorts. (A) Comparison of multi-dimensional TLS features across TCGA-READ, TCGA-ESCA, and TCGA-STAD. Panels from left to right present TLS Density, Median TLS area, median TLS solidity, and NN (nearest neighbor) cluster index. $^ { * } \textit { p }  { i } 0 . 0 5 , ~ ^ { * * } \textit { p }  { i } 0 . 0 1$ , and \*\*\* p ¡ 0.001. (B) Kaplan–Meier survival analysis of key TLS features. Patients were stratified into high- and low-value groups based on exploratory data-derived cutof values. Represented are TLS density and NN cluster index in the TCGA-READ cohort, median TLS solidity in the TCGA-ESCA cohort, and Mean TLS area in the TCGA-STAD cohort.

FTU-Seek-derived 40 glandular architectural phenotypes were evaluated in TCGA-PRAD. The baseline clinical and pathological characteristics of the study cohort are summarized in Figure 12A. In terms of histological grading, the majority of patients presented with a Gleason score of $\leq 7 \ ( \mathrm { n } = 2 5 0 , 6 2 . 3 \% )$ , while 151 patients (37.7%) had a high-grade disease with a Gleason score of $\dot { \wr } , 7 .$ Regarding pathological staging, a higher proportion of patients were diagnosed with advanced $\mathrm { p T }$ stage $( \mathrm { T 3 } / \mathrm { T 4 } , \mathrm { n = 2 3 9 } , 6 0 . 5 \% )$ compared to localized stage $( \mathrm { T 2 } , \mathrm { n } = 1 5 6 , 3 9 . 5 \% )$ . Furthermore, at the end of the follow-up period, BCR was observed in 41 patients (12.0%), with the remaining 301 patients (88.0%) staying BCR-free (Figure 12A). FTU-Seekderived glandular architectural phenotypes were evaluated in TCGA-PRAD. In the BCR analysis, recurrent cases showed higher eccentricity and lower gland count density (nominal $p < 0 . 0 5$ ; FDR-adjusted $p < 0 . 1 ;$ Figure 12B). Consistent with these findings, compared with tumors with a Gleason score $\leq 7 .$ tumors with a Gleason score $> 7$ had lower gland count density (median 2.60 vs. 3.86 $\operatorname { g l a n d s } / \operatorname { m m } ^ { 2 } ;$ FDRadjusted $p < 0 . 0 0 1 )$ and higher median gland eccentricity (median 0.862 vs. 0.847; FDR-adjusted $p < 0 . 0 0 1$ ; Figure 12C). Similar trends were observed for local extension: pT3/T4 tumors showed lower gland area (median 0.103 vs. 0.175; FDR-adjusted $p ~ < ~ 0 . 0 0 1 )$ and higher gland eccentricity (median 0.821 vs. 0.808; FDR-adjusted $p \ < \ 0 . 0 0 1 )$ than $\mathrm { p T 2 }$ tumors (Figure 12D). Additionally, positive surgical margins were associated with lower large-gland count density and lower total gland density (both FDR-adjusted $p < 0 . 0 0 1$ ; Figure 12E). Collectively, these findings demonstrate that FTU-Seek-derived glandular phenotypes capture architectural alterations associated with adverse pathological characteristics and disease recurrence, supporting their utility for quantitative downstream tissue analysis.

A  
![](images/1d7046991725a2fdc7ed056d868e2caf862dd9b26b2de0c8fc3822d1bec201de.jpg)

B  
![](images/a5a51b4dec4d3ff6323f6faad0fdf3168a09bf36eed33025c7b531bc3e1e2a94.jpg)

C  
![](images/ace4ec69633757a6911b1c8fdbdb526e3b94b859d79ec9cd623e9e2b09ec945e.jpg)

![](images/70725ee4a0a2436b6ceaa71e13ccdc15a9abab42e1a596e75fc794c6974c936f.jpg)

![](images/d0dd6bbe9f753d1585bbbbb1cd599208397df2d882da3c71164a0d4131e6a6a9.jpg)

![](images/fc5a6bbd5e038e38a1c75fefc58ad6c81782ebf509d0f85ed721a9ddaa8dad2f.jpg)

D  
![](images/2af85f78b8eeabeb220ecc183677ddc6c876712cefa0151573f8670177782194.jpg)

![](images/b48fd81c5a9d6c82e62a95fad55b407bd5cb1c140ed4204ecfa260e065b8bf6f.jpg)  
Fig. 11 Exploratory associations of blood-vessel phenotypes with overall survival (OS) and microvascular invasion (MVI) in TCGA-LIHC. (A) Kaplan–Meier estimates of overall survival (OS) for patients stratified into BV-low (blue line) and BV-high (red line) groups based on an exploratory data-derived cutof value. Statistical significance was assessed using the log-rank test. (B) Forest plot of the multivariable Cox proportional hazards regression analysis for OS. The model incorporates clinical covariates including age, gender, AJCC stage, BV classification (BV-low vs. BV-high), Child– Pugh score, and Hepatitis B Virus (HBV) status. Hazard ratios (HR) and corresponding p-values are presented. (C) Boxplots displaying the distribution of quantitative computational BV features in patient groups without and with MVI. The central horizontal line represents the median, box edges denote the interquartile range (IQR). (D) Forest plot presenting the multivariable logistic regression analysis for MVI status. The model evaluates hotspot top mean BV density and adjusts for baseline clinical characteristics (age, gender, Child–Pugh score, and HBV status). Odds ratios (ORs), 95% confidence intervals (CIs), and p-values are provided for each variable. $^ { * } p < 0 . 0 5$

A  
![](images/ce866b9599d59be267cb8f0a7acefe824283964e2631e19d89e0d3c8730b43a0.jpg)  
B

![](images/37eeaeaad5946b9a2237b30b6f9a6d5d6610944d0711684c217cbf5be1483bc6.jpg)

![](images/ada9b7ddab1bc0b3773482122254ec3abf914fa53d31b65bc365cf5772413365.jpg)

![](images/a87afddc51ccd7b9a500f3aee82d99947719f5c529bf9b15f747d88a5da29986.jpg)

![](images/892e1130b140a7b9b7ca9337786202ea1f249bc854504b9b65593fbf97c72338.jpg)

![](images/dfc6a10e4dee6cfc2a2e616e5d6b78bf18e33243e9a060bccbc44d4516f0508e.jpg)

![](images/bb598606ecef6d297c395e021702c7ce714192366ed534f9b6b79a86929b31cc.jpg)

![](images/4a7d00abe749a2c64785a7554a2ff3be79d8623da0464ae91f0aa7404a7937dc.jpg)

![](images/a5bc0a3c3b0b6574fde24e6b63392a95d3fbb54d531de8d98ffbf2b0c50282a8.jpg)

C  
![](images/252a197053eb092b934044abae5b82d9931cd557e421847429de24ce9d496e8d.jpg)

![](images/9eb0a05465bd3701d843ec62a5a730874204f76351d26fdeacd7aca38f3de203.jpg)

D  
![](images/e2b03424cd319f9a6c273e7c5beecc37f8830dc87d78f5e15dafdccd256de112.jpg)

![](images/160378a9f7b9a6b52219f0832df4135800063842cf739b35d06de7acccd77413.jpg)

![](images/298cd54730898aa222cafdf8e378d715325c4246b4c270fec2b191f4e69b36ff.jpg)

![](images/24c7f251b1331c57cbf93a3e6a1e456086acddb76c52999d21abbcb99ef885fe.jpg)  
Fig. 12 Association of segmented gland features with key clinicopathological determinants in the TCGA-PRAD cohort. (A) Donut charts illustrating the proportion of patients within the TCGA-PRAD cohort stratified by key clinical and pathological categories: Gleason grade (≤7 vs. >7), pT stage (T2 vs. T3/T4), and biochemical recurrence (BCR) status (No BCR vs. BCR). Slices indicate the count and percentage for each respective subgroup. (B–E) Boxplots detailing the distribution of extracted gland features that exhibited statistically significant diferences between the respective clinical groups. (B) Distribution of gland features (mean gland eccentricity, median gland eccentricity, gland count density, gland area fraction, small-gland area fraction, and small-gland count density) compared between patients without and with BCR. (C) Comparison of gland count density and median gland eccentricity between Gleason grade $\leq 7$ and >7 groups. (D) Distribution of gland area fraction and mean gland eccentricity across $\mathrm { p T }$ stage categories (T2 vs. T3/T4). (E) Comparison of gland area fraction and large-gland count density categorized by resection margin status (R0 vs. R1). For all boxplots, the central horizontal line represents the median, the box edges span the interquartile range (IQR). <sup>∗</sup> nominal $p < 0 . 0 5 , \ ^ { * * }$ nominal $p < 0 . 0 1 , \ ^ { * * * }$ nominal $p < 0 . 0 0 1$

## 4 Discussion

In this study, we developed FTU-Seek , a pathology foundation model (PFM)-guided hard-negative learning framework for sparse functional tissue unit (FTU) segmentation in whole-slide images (WSIs). Unlike conventional approaches that primarily employ foundation models as feature extractors or segmentation backbones, FTU-Seek additionally exploits pretrained histomorphological representations to construct morphology-aware segmentation training sets [46, 47, 71]. A patch-level classifier first ranks FTU-absent tissue according to its predicted target-containing probability, after which a static TopK strategy selects the most target-like negative patches for segmentation training. This design addresses a key challenge in quantitative histopathology: biologically meaningful microscopic structures are often sparsely distributed throughout large tissue sections, whereas the majority of tissue patches contain redundant background and only a limited subset comprises morphologically confusing regions likely to generate false-positive predictions. Across TLS, gland, and blood-vessel segmentation, our results show that morphology-aware negative selection improves the trade-of between segmentation accuracy and retained training workload, with the clearest benefits observed for sparse or morphologically ambiguous FTUs. The resulting segmentation outputs can further be transformed into quantitative tissue phenotypes for exploratory downstream analysis.

The segmentation experiments highlight that the composition of the negative training set is more important than its absolute size. Training using only FTU-containing patches consistently resulted in inferior performance, confirming that representative negative tissue is indispensable for learning robust decision boundaries. Conversely, all-tissue training provides abundant negative information but requires processing a very large number of easy background patches that contribute little additional supervision. FTU-Seek addresses this imbalance by preferentially selecting classifierranked target-like negatives, thereby concentrating training on the background regions most likely to generate false-positive predictions. Although this concept shares the objective of hard-example mining, FTU-Seek difers fundamentally from conventional online hard-example mining because dificult negatives are identified before segmentation training using frozen pathology foundation model representations rather than repeatedly re-mined during optimization. Consequently, the proposed strategy is computationally reproducible, avoids repeated screening of the entire WSI patch pool, and enables direct comparison with alternative sampling strategies under identical negative-sampling budgets. The reduction in training workload should be distinguished from a reduction in peak GPU-memory usage. FTU-Seek decreases the total number of retained training patches and therefore improves training-data eficiency without requiring additional GPU memory or a larger computational platform. However, when the model architecture, batch size, input resolution, and hardware configuration are fixed, reducing the total number of training patches does not necessarily reduce the peak GPU-memory usage per iteration. Accordingly, the computational benefit of FTU-Seek should be interpreted as improved training-data eficiency under a fixed memory budget rather than as a proportional reduction in GPU-memory consumption. A complete assessment of computational eficiency would additionally require systematic measurements of wall-clock training time, GPU utilization, peak memory, and energy consumption.

The task-dependent benefit of classifier-guided hard-negative selection may be explained by diferences in target sparsity, class imbalance, and morphological ambiguity. When targets are sparse and imbalance is severe, the negative pool is dominated by easy background, making random sampling less likely to capture the small subset of target-like negatives most responsible for false-positive predictions. TLSs represented the sparsest segmentation task and exhibited the greatest eficiency gains. A relatively small number of classifier-selected negatives was suficient to recover or slightly exceed the performance of all-tissue training, suggesting that only a limited subset of inflammatory or stromal regions contributes substantially to false-positive TLS predictions. Blood-vessel segmentation demonstrated a diferent behaviour. Because vessel-like structures occur in diverse morphological contexts, including tissue clefts, empty lumina, collagen bundles, elongated stromal regions, and red blood cell-rich spaces, segmentation performance continued to improve as larger hard-negative pools were incorporated. Gland segmentation was comparatively insensitive to the sampling strategy because gland-containing patches were considerably more abundant, resulting in less severe class imbalance, and the distinction between positive and negative tissue was substantially clearer. Collectively, these findings indicate that sparse FTUs derive greater benefit from targeted hard-negative selection, whereas more abundant FTUs with clearer class separation show smaller gains; broader morphological ambiguity may additionally require greater negative coverage. This task-dependent sampling strategy also provides a practical means of reducing redundant training data. By removing easy-negative patches before segmentation training, FTU-Seek reduces the number of samples processed during optimization while keeping the segmentation architecture, batch size, input resolution, and hardware configuration unchanged.

The classifier probability distributions provide further insight into the mechanism underlying FTU-Seek. For TLSs, the vast majority of negative patches received probabilities close to zero, while only a small high-probability tail corresponded to localized target-like tissue. Such distributions naturally favour strict TopK selection because informative negatives are concentrated within a relatively small subset. Blood vessels exhibited substantially broader overlap between positive and negative probability distributions, reflecting widespread morphological ambiguity and explaining why larger negative budgets were beneficial. Gland classification showed the clearest overall separation, although intermediate-probability regions remained around gland boundaries, pseudo-glandular spaces, epithelial debris, and fibrotic cavities. These observations indicate that classifier probabilities capture biologically meaningful morphological ambiguity rather than merely reflecting classification confidence. Future developments may therefore combine probability-based ranking with diversity constraints, uncertainty estimation, or tissue-context clustering to obtain even more representative hard-negative sets.

The patch-level classifier is an important intermediate component, but its accuracy does not guarantee improved segmentation. Because all target-containing patches are retained for segmentation training, classifier errors on positive patches do not directly remove positive training examples. In contrast, errors within the target-absent pool alter the composition of the selected hard-negative set. If a morphologically confusing negative patch receives an underestimated target-containing probability, it may be omitted from TopK selection, leaving the corresponding false-positive pattern insuficiently represented during segmentation training. Conversely, overestimated probabilities may allocate TopK slots to less informative negatives. Although such selections may still increase the diversity of background examples, extensive misranking can reduce the eficiency of hard-negative sampling. Thus, the quality and calibration of the patch-level classifier afect segmentation indirectly through negativepool composition. Uncertainty-aware ranking or iterative negative re-mining may help mitigate this error propagation, but would introduce additional computational cost and potential variability.

An important component of FTU-Seek is the use of frozen pathology foundation model representations. Recent pathology foundation models such as UNI and Virchow have demonstrated remarkable transferability across computational pathology tasks by learning general histomorphological representations from large and heterogeneous pathology image collections [46, 47, 72]. In the present framework, these representations serve two complementary purposes. First, they provide robust patch-level features for identifying target-like negative tissue. Second, they supply transferable multi-scale representations for dense segmentation across structurally distinct FTUs. Freezing the encoder substantially reduces the number of trainable parameters while minimizing dependence on relatively small manually annotated datasets. Our results therefore suggest that pathology foundation models can contribute not only through transferable feature extraction but also through intelligent construction of segmentation training data, thereby extending their role beyond conventional transfer learning.

Beyond computational eficiency, FTU-Seek enables quantitative characterization of tissue architecture. TLSs, blood vessels, and glands represent complementary components of the tissue microenvironment that reflect adaptive immune organization, angiogenesis and vascular remodeling, and epithelial diferentiation, respectively. In this study, their segmentation masks were converted into quantitative spatial phenotypes for exploratory downstream analyses.

Our proof-of-concept downstream analyses further illustrate this potential. Quantitative TLS features spanning abundance, spatial organization, and morphology exhibited variation across tumor types. The observed diferences among READ, ESCA, and STAD are biologically plausible because TLS maturation, localization, cellular composition, and interactions with the surrounding tumor microenvironment differ substantially across tumor types [73–77]. Several TLS phenotypes also showed potential associations with OS when evaluated as continuous measures, supporting their utility for generating hypotheses regarding the relationship between immune architecture and patient outcome. Similarly, vascular phenotyping in hepatocellular carcinoma identified candidate features related to vessel density and localized vascular hotspots, with exploratory associations observed for OS and MVI. Finally, in prostate cancer, glandular phenotypes captured architectural alterations associated with adverse clinicopathological characteristics, including reduced gland density and increased eccentricity in more aggressive tumors. Collectively, these observations suggest that FTU-Seek-derived phenotypes retain information beyond the segmentation masks themselves and can serve as quantitative descriptors for subsequent biological and clinical investigation.

Several limitations should be acknowledged. First, although each development cohort contained extensive pixel-level annotations, the number of annotated WSIs remained relatively limited, and the internal test cohort contained only two WSIs per segmentation task. Accordingly, the internal-test results should be interpreted as descriptive evidence rather than definitive estimates of generalization. To provide a stronger independent assessment, we additionally evaluated the TLS segmentation strategies on 30 held-out WSIs from the same institution. However, this cohort does not constitute multi-center external validation. Larger multi-institutional studies will be necessary to evaluate the robustness of FTU-Seek across diverse patient populations, staining protocols, scanners, and tissue-processing procedures [78].

The additional held-out analysis showed that FTU-Seek achieved higher slide-level Dice than matched random TopK sampling across all four evaluated negative-patch budgets, with positive paired diferences and confidence intervals excluding zero. These results support the robustness of the method within the available same-institution data setting, while broader multi-institutional validation remains necessary.

Second, the present workload metric was defined as the proportion of retained training patches relative to all-tissue training rather than direct measurements of computational time, GPU utilization, memory consumption, or energy eficiency. Future studies should therefore include comprehensive computational benchmarking. Third, the proposed static TopK strategy primarily emphasizes false-positive suppression and does not explicitly incorporate false-negative regions, annotation uncertainty, or dificult target boundaries. Iterative re-mining, uncertainty-guided sampling, boundary-aware optimization, and combined false-positive/false-negative mining may further improve segmentation performance. Fourth, identical TopK candidate values were evaluated across all FTU categories, although our results indicate that the optimal hard-negative budget depends on target prevalence and morphological complexity.

The fixed per-WSI allocation also means that slides with diferent tissue areas do not contribute negative patches in proportion to their available tissue area. Proportional allocation and globally ranked negative-patch selection were not evaluated in the present study. These alternative allocation schemes, together with prevalence- or uncertainty-adaptive budgets, remain directions for future work.

Adaptive sampling strategies based on classifier uncertainty, probability distributions, or validation performance may therefore provide greater flexibility. Finally, FTU-Seek may have potential future applications in the analysis of stem cell-derived tissues, organoids, and regenerative medicine specimens; however, these settings were not investigated in the present study and would require dedicated validation using appropriately annotated datasets.

The downstream TCGA analyses should be interpreted as exploratory and hypothesis-generating. Because manually annotated reference masks were unavailable, the TCGA cohorts were used for exploratory downstream application rather than direct external validation of segmentation performance. These retrospective analyses were not designed for biomarker validation, and the data-derived survival cut points should not be considered validated prognostic thresholds. Accordingly, the findings primarily demonstrate the feasibility of deriving quantitative phenotypes from FTU-Seek segmentations. Future integration with spatial transcriptomics, proteomics, and molecular pathology may further clarify the biological basis of these phenotypes and identify reproducible architectural features across disease contexts.

Overall, FTU-Seek extends pathology foundation models beyond transferable feature representation by using pretrained histomorphological information to identify target-like negative tissue before segmentation training and construct compact, morphology-aware training sets. This pre-segmentation hard-negative selection reduces redundant background sampling and retained training workload while maintaining competitive segmentation performance, with the clearest practical benefit observed for sparse and imbalanced FTUs. These findings should be interpreted in light of the relatively small number of annotated WSIs, the lack of direct wall-clock and resource benchmarking, and the dependence of the static TopK strategy on classifierbased ranking and task-specific sampling budgets. Larger multi-institutional cohorts, systematic computational benchmarking, and adaptive or iterative negative-selection strategies will be needed to establish the robustness and practical clinical applicability of FTU-Seek.

## 5 Conclusions

Overall, FTU-Seek extends pathology foundation models beyond feature representation by using pretrained histomorphological information to identify target-like negative tissue before segmentation training and construct compact, morphology-aware training sets. This pre-segmentation hard-negative selection directly addresses the challenge of sparse FTU segmentation by reducing redundant background sampling and retained training workload while maintaining competitive segmentation performance, with particularly clear benefits for sparse and imbalanced FTUs. The present study is limited by the relatively small number of annotated WSIs and the absence of direct computational benchmarking, while the static TopK strategy remains dependent on classifier-based ranking and task-specific sampling requirements. Prospective validation in independent multi-institutional cohorts will therefore be required before the clinical applicability of FTU-Seek can be established, and more adaptive sampling strategies warrant further investigation.

## References

[1] Kumar, N., Gupta, R., Gupta, S.: Whole slide imaging (wsi) in pathology: current perspectives and future directions. Journal of digital imaging 33(4), 1034–1040 (2020)

[2] Bera, K., Schalper, K.A., Rimm, D.L., Velcheti, V., Madabhushi, A.: Artificial intelligence in digital pathology—new tools for diagnosis and precision oncology. Nature reviews Clinical oncology 16(11), 703–715 (2019)

[3] Bian, C., Wang, Y., Lu, Z., An, Y., Wang, H., Kong, L., Du, Y., Tian, J.: Immunoaizer: A deep learning-based computational framework to characterize cell distribution and gene mutation in tumor microenvironment. Cancers 13(7), 1659 (2021)

[4] Bulten, W., Pinckaers, H., Van Boven, H., Vink, R., De Bel, T., Van Ginneken, B., Laak, J., Kaa, C., Litjens, G.: Automated deep-learning system for gleason grading of prostate cancer using biopsies: a diagnostic study. The Lancet Oncology 21(2), 233–241 (2020)

[5] Str¨om, P., Kartasalo, K., Olsson, H., Solorzano, L., Delahunt, B., Berney, D.M., Bostwick, D.G., Evans, A.J., Grignon, D.J., Humphrey, P.A., et al.: Artificial intelligence for diagnosis and grading of prostate cancer in biopsies: a populationbased, diagnostic study. The lancet oncology 21(2), 222–232 (2020)

[6] Wang, L.S., Yu, J.: Analysis framework for stochastic predator–prey model with demographic noise. Results in Applied Mathematics 27, 100621 (2025)

[7] Lugano, R., Ramachandran, M., Dimberg, A.: Tumor angiogenesis: causes, consequences, challenges and opportunities. Cellular and molecular life sciences 77(9), 1745–1770 (2020)

[8] Seraphin, T.P., Mesropian, A., Zigutyt˙e, L., Brooks, J., Mauro, E., Gris-Oliver, <sup>ˇ</sup> A., Pinyol, R., Montironi, C., Balaseviciute, U., Piqu´e-Gili, M., et al.: Artificial intelligence predicts outcome-related molecular profiles and vascular invasion in hepatocellular carcinoma. JHEP Reports, 101592 (2025)

[9] Saut\`es-Fridman, C., Petitprez, F., Calderaro, J., Fridman, W.H.: Tertiary lymphoid structures in the era of cancer immunotherapy. Nature Reviews Cancer 19(6), 307–325 (2019)

[10] Cabrita, R., Lauss, M., Sanna, A., Donia, M., Skaarup Larsen, M., Mitra, S., Johansson, I., Phung, B., Harbst, K., Vallon-Christersson, J., et al.: Tertiary lymphoid structures improve immunotherapy and survival in melanoma. Nature 577(7791), 561–565 (2020)

[11] Wang, L.S., Yu, J., Li, S., Liu, Z.: Analysis and mean-field limit of a hybrid pdeabm modeling angiogenesis-regulated resistance evolution. Mathematics 13(17), 2898 (2025)

[12] Michot, A., Vanhersecke, L., Dinart, D., Bourdon, A., Azmani, R., Velasco, V., Bonomo, I., Toureille, M., Toulmonde, M., Perret, R.E., et al.: Beyond surgical margins: Fully mature tertiary lymphoid structures (fmtlss) are predictive biomarkers for local recurrence in primary soft-tissue sarcomas. Cancers 18(11), 1685 (2026)

[13] Diao, J.A., Wang, J.K., Chui, W.F., Mountain, V., Gullapally, S.C., Srinivasan,

R., Mitchell, R.N., Glass, B., Hofman, S., Rao, S.K., et al.: Human-interpretable image features derived from densely mapped cancer pathology slides predict diverse molecular phenotypes. Nature communications 12(1), 1613 (2021)

[14] Liang, Y., Wang, L.S., Yu, J., Liu, Z.: Global well-posedness and stability of nonlocal damage-structured lineage model with feedback and dediferentiation. Mathematics 13(22), 3583 (2025)

[15] Su, L., Li, R., Zhang, G., Liu, Z., Geng, X., Wang, H., Gai, W., Wu, H., Tian, J., Du, Y.: Gcunet: a graph neural network-based contextual learning network for tertiary lymphoid structure semantic segmentation in whole slide image. Visual Computing for Industry, Biomedicine, and Art 9(1), 15 (2026)

[16] Ye, Y., Yuan, J., Tang, J., Wan, P., Sun, L., Sheng, J., Zhang, D., Shao, W.: Foundation model based zero-shot tissue segmentation of pathological images via the mixture of local-to-global experts. IEEE Transactions on Image Processing (2026)

[17] Schumacher, T.N., Thommen, D.S.: Tertiary lymphoid structures in cancer. Science 375(6576), 9419 (2022)

[18] Yu, J., Wang, L.S., Liu, Z., Liu, J.: Pattern suppression and recovery under oneway versus two-way chemotactic coupling in hybrid partial diferential equation– ordinary diferential equation models. Transport Phenomena 1(1), 20260023 (2026)

[19] Xia, P., Chen, D., An, H., Lim, K.S., Yang, X.: Learnable prototype-guided multiple instance learning for detecting tertiary lymphoid structures in multi-cancer whole-slide pathological images. Medical Image Analysis 104, 103652 (2025)

[20] Hamidinekoo, A., Kelsey, A., Trahearn, N., Selfe, J., Shipley, J., Yuan, Y.: Automated quantification of blood microvessels in hematoxylin and eosin whole slide images. In: MICCAI Workshop on Computational Pathology, pp. 94–104 (2021). PMLR

[21] Sirinukunwattana, K., Pluim, J.P., Chen, H., Qi, X., Heng, P.-A., Guo, Y.B., Wang, L.Y., Matuszewski, B.J., Bruni, E., Sanchez, U., et al.: Gland segmentation in colon histology images: The glas challenge contest. Medical image analysis 35, 489–502 (2017)

[22] Yu, J., Wang, L.S., Liang, Y.: Size-selective threshold harvesting under nonlocal crowding and exogenous recruitment. International Journal of Diferential Equations 2026(1), 2523535 (2026)

[23] Chen, H., Qi, X., Yu, L., Dou, Q., Qin, J., Heng, P.-A.: Dcan: Deep contour-aware networks for object instance segmentation from histology images. Medical image analysis 36, 135–146 (2017)

[24] Niazi, M.K.K., Parwani, A.V., Gurcan, M.N.: Digital pathology and artificial intelligence. The lancet oncology 20(5), 253–261 (2019)

[25] Sevim, S., Hajar, C., Sonawane, S.: What’s new in digital and computational pathology 2026: advances in adoption, standards, ai technologies, and clinical integration. Journal of Pathology and Translational Medicine 60(3), 364–370 (2026)

[26] Wang, L.S., Yu, J.: Algebraic–spectral thresholds and discrete–continuous stability transfer in leslie–gower systems. Electronic Research Archive 34(1), 251–290 (2026)

[27] Al-Asi, H., Yilmaz, I., Reynolds, J., Agarwal, S., Nassar, A., Zubair, A., Horbinski, C., Dangott, B., Akkus, Z.: Pathology foundation models: Evolution, current landscape, challenges and opportunities from a technical and clinical perspective. Bioengineering 13(5), 577 (2026)

[28] Xia, R., Littlefield, N., Bao, R., Gu, Q.: Beyond algorithmic performance: translational gaps in implementing artificial intelligence for clinical digital pathology. Taylor & Francis (2026)

[29] Hosseini, M.S., Bejnordi, B.E., Trinh, V.Q.-H., Chan, L., Hasan, D., Li, X., Yang, S., Kim, T., Zhang, H., Wu, T., et al.: Computational pathology: a survey review and the way forward. Journal of Pathology Informatics 15, 100357 (2024)

[30] Al-Thelaya, K., Gilal, N.U., Alzubaidi, M., Majeed, F., Agus, M., Schneider, J., Househ, M.: Applications of discriminative and deep learning feature extraction methods for whole slide image analysis: A survey. Journal of Pathology Informatics 14, 100335 (2023)

[31] Kather, J.N., Krisam, J., Charoentong, P., Luedde, T., Herpel, E., Weis, C.-A., Gaiser, T., Marx, A., Valous, N.A., Ferber, D., et al.: Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. PLoS medicine 16(1), 1002730 (2019)

[32] Saltz, J., Gupta, R., Hou, L., Kurc, T., Singh, P., Nguyen, V., Samaras, D., Shroyer, K.R., Zhao, T., Batiste, R., et al.: Spatial organization and molecular correlation of tumor-infiltrating lymphocytes using deep learning on pathology images. Cell reports 23(1), 181–193 (2018)

[33] Wang, L.S., Yu, J., Liu, Z.: A damage-structured pde model of stem cell hierarchies: The dual role of dediferentiation in tissue homeostasis and aging. Plos one 21(2), 0335163 (2026)

[34] Campanella, G., Hanna, M.G., Geneslaw, L., Miraflor, A., Werneck Krauss Silva, V., Busam, K.J., Brogi, E., Reuter, V.E., Klimstra, D.S., Fuchs, T.J.: Clinicalgrade computational pathology using weakly supervised deep learning on whole

[35] Van Rijthoven, M., Balkenhol, M., Sili¸na, K., Van Der Laak, J., Ciompi, F.: Hooknet: Multi-resolution convolutional neural networks for semantic segmentation in histopathology whole-slide images. Medical image analysis 68, 101890 (2021)

[36] Lee, J., Cha, S., Kim, J., Kim, J.J., Kim, N., Jae Gal, S.G., Kim, J.H., Lee, J.H., Choi, Y.-D., Kang, S.-R., et al.: Ensemble deep learning model to predict lymphovascular invasion in gastric cancer. Cancers 16(2), 430 (2024)

[37] Milletari, F., Navab, N., Ahmadi, S.-A.: V-net: Fully convolutional neural networks for volumetric medical image segmentation. In: 2016 Fourth International Conference on 3D Vision (3DV), pp. 565–571 (2016). IEEE

[38] Lin, T.-Y., Goyal, P., Girshick, R., He, K., Doll´ar, P.: Focal loss for dense object detection. In: 2017 IEEE International Conference on Computer Vision (ICCV), pp. 2999–3007 (2017)

[39] Wang, L.S., Yu, J., Liang, Y., Zhang, J.: The breakdown of linear quasi-cycles: Demographic noise and absorbing boundaries in finite predator–prey systems. Electronic Research Archive 34(6), 4248–4289 (2026)

[40] Shrivastava, A., Gupta, A., Girshick, R.: Training region-based object detectors with online hard example mining. In: IEEE Conference on Computer Vision and Pattern Recognition, pp. 761–769 (2016)

[41] Li, M., Wu, L., Wiliem, A., Zhao, K., Zhang, T., Lovell, B.: Deep instance-level hard negative mining model for histopathology images. In: International Conference on Medical Image Computing and Computer-assisted Intervention, pp. 514–522 (2019). Springer

[42] Yu, J., Wang, L.S.: Beyond diagonal noise: A better predator-prey modeling framework with cross-covariance. Plos one 21(5), 0350127 (2026)

[43] Huang, W., Hu, X., Abousamra, S., Prasanna, P., Chen, C.: Hard negative sample mining for whole slide image classification. In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 144–154 (2024). Springer

[44] Wang, X., Yang, S., Zhang, J., Wang, M., Zhang, J., Yang, W., Huang, J., Han, X.: Transformer-based unsupervised contrastive learning for histopathological image classification. Medical image analysis 81, 102559 (2022)

[45] Yu, J., Wang, L.S., Liang, Y.: Rigorous analysis of a nonlocal transport–renewal system for physiologically structured populations. Mathematical Methods in the Applied Sciences (2026)

[46] Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F., Jaume, G., Song, A.H., Chen, B., Zhang, A., Shao, D., Shaban, M., et al.: Towards a general-purpose foundation model for computational pathology. Nature medicine 30(3), 850–862 (2024)

[47] Vorontsov, E., Bozkurt, A., Casson, A., Shaikovski, G., Zelechowski, M., Severson, K., Zimmermann, E., Hall, J., Tenenholtz, N., Fusi, N., et al.: A foundation model for clinical-grade computational pathology and rare cancers detection. Nature medicine 30(10), 2924–2935 (2024)

[48] Xu, H., Usuyama, N., Bagga, J., Zhang, S., Rao, R., Naumann, T., Wong, C., Gero, Z., Gonz´alez, J., Gu, Y., et al.: A whole-slide foundation model for digital pathology from real-world data. Nature 630(8015), 181–188 (2024)

[49] Lu, M.Y., Chen, B., Williamson, D.F., Chen, R.J., Liang, I., Ding, T., Jaume, G., Odintsov, I., Le, L.P., Gerber, G., et al.: A visual-language foundation model for computational pathology. Nature medicine 30(3), 863–874 (2024)

[50] Yu, J., Wang, L.S., Ban, S., Liang, Y.: From microscopic damage to macroscopic games: a dimensionality reduction of stem cell homeostasis. Transport Phenomena 1(2), 20260037 (2026)

[51] Campanella, G., Chen, S., Singh, M., Verma, R., Muehlstedt, S., Zeng, J., Stock, A., Croken, M., Veremis, B., Elmas, A., et al.: A clinical benchmark of public selfsupervised pathology foundation models. Nature Communications 16(1), 3640 (2025)

[52] Weinstein, J.N., Collisson, E.A., Mills, G.B., Shaw, K.R., Ozenberger, B.A., Ellrott, K., Shmulevich, I., Sander, C., Stuart, J.M.: The cancer genome atlas pan-cancer analysis project. Nature genetics 45(10), 1113–1120 (2013)

[53] Amisaki, M., Zebboudj, A., Yano, H., Zhang, S.L., Payne, G., Chandra, A.K., Yu, R., Guasp, P., Sethna, Z.M., Ohmoto, A., et al.: Il-33-activated ilc2s induce tertiary lymphoid structures in pancreatic cancer. Nature 638(8052), 1076–1084 (2025)

[54] Cai, J., Chen, X., Gu, L., Chen, J., Chu, N., Wang, L.S., Liang, Y., Yu, J.: Optimal harvesting for nonlinear size-structured populations with nonlocal environmental feedback. Mathematics 14(11), 2025 (2026)

[55] Ling, Y., Zhong, J., Weng, Z., Lin, G., Liu, C., Pan, C., Yang, H., Wei, X., Xie, X., Wei, X., et al.: The prognostic value and molecular properties of tertiary lymphoid structures in oesophageal squamous cell carcinoma. Clinical and translational medicine 12(10), 1074 (2022)

[56] Kumar, V., Abbas, A.K., Fausto, N., Aster, J.C.: Robbins and Cotran Pathologic Basis of Disease, Professional Edition E-book. Elsevier health sciences (2014)

[57] Yu, J., Wang, L.S., Liang, Y.: Age-structured harvesting models: A structural comparison of rate-control and efort-control optimality systems. International Journal of Mathematics and Mathematical Sciences 2026(1), 6212516 (2026)

[58] O’Dowd, G., Bell, S., Wright, S.: Wheater’s Functional Histology, E-Book: A Text and Colour Atlas. Elsevier Health Sciences (2023)

[59] Epstein, J.I., Egevad, L., Amin, M.B., Delahunt, B., Srigley, J.R., Humphrey, P.A., Committee, G., et al.: The 2014 international society of urological pathology (isup) consensus conference on gleason grading of prostatic carcinoma: definition of grading patterns and proposal for a new grading system. The American journal of surgical pathology 40(2), 244–252 (2016)

[60] Wang, L.S., Yu, J., Liang, Y., Zhang, J.: Elliptic criticality versus volterra memory in indirect chemotaxis cascades. Transport Phenomena 1(2), 20260061 (2026)

[61] Dasgupta, P.: Re: Artificial intelligence for diagnosis and gleason grading of prostate cancer: The panda challenge. European Urology 82(5), 571–571 (2022)

[62] Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M., Mahmood, F.: Data-eficient and weakly supervised computational pathology on whole-slide images. Nature biomedical engineering 5(6), 555–570 (2021)

[63] Yu, J., Wang, L.S., Gao, Y., Liang, Y.: Full-covariance chemical langevin predator–prey difusion with absorbing boundaries. Royal Society Open Science 13(8), 260260 (2026)

[64] Wang, Z., Wang, D., Yu, J.: Multi-strategy hybrid improved intelligent algorithm for solving uav-mtsp. Information Technology and Control 54(2), 413–438 (2025)

[65] Liu, Z., Su, L., Yu, J., Geng, X., Wang, L.S., Wang, J., Liu, J.: Ftu-seek: Foundation model-guided hard-negative learning for sparse functional tissue unit segmentation. Biomedicines 14(9), 1935 (2026)

[66] Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction. In: IEEE/CVF International Conference on Computer Vision, pp. 12179–12188 (2021)

[67] Sudre, C.H., Li, W., Vercauteren, T., Ourselin, S., Jorge Cardoso, M.: Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations. In: International Workshop on Deep Learning in Medical Image Analysis, pp. 240–248 (2017). Springer

[68] Liu, Z., Wang, L.S., Yu, J., Zhang, J., Martel, E., Li, S.: Bidirectional endothelial feedback drives turing-vascular patterning and drug-resistance niches: a hybrid pde-agent-based study. Bioengineering 12(10), 1097 (2025)

[69] Gao, Y., Li, L., Yu, J.: Rolling prediction model of closing price based on eemd data noise reduction and hgs-delm. In: 2022 International Conference on Data Analytics, Computing and Artificial Intelligence (ICDACAI), pp. 255–260 (2022). IEEE

[70] Ehteshami Bejnordi, B., Veta, M., Diest, P., Van Ginneken, B., Karssemeijer, N., Litjens, G., Van Der Laak, J.A., consortium, C., Hermsen, M., Manson, Q.F., et al.: Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. Jama 318(22), 2199–2210 (2017)

[71] Liang, Y., Wang, L.S., Yu, J., Zan, X.: Separation-like irregularity and sample size optimism in high-discrimination logistic prediction models. PloS one 21(8), 0342286 (2026)

[72] Liu, Z., Yu, J., Wang, L., Su, L., Liang, Y., Du, Y., Liu, J.: Computational oncology of chemotaxis-driven tumour–immune spatial patterning and stability. Bioengineering 13, 952 (2026)

[73] Cho, K.S., Liu, Y., Pei, G., Chen, J., Dai, Y., Liu, Y., Zhou, T., Bougouin, A., Serrano, A., Wani, K., et al.: Pan-cancer spatial atlas of tertiary lymphoid structures. Science 392(6801), 2742 (2026)

[74] Sili¸na, K., Soltermann, A., Attar, F.M., Casanova, R., Uckeley, Z.M., Thut, H., Wandres, M., Isajevs, S., Cheng, P., Curioni-Fontecedro, A., Foukas, P., Levesque, M.P., Moch, H., Lin¯e, A., Broek, M.: Germinal centers determine the prognostic relevance of tertiary lymphoid structures and are impaired by corticosteroids in lung squamous cell carcinoma. Cancer Res 78(5), 1308–1320 (2018)

[75] Yu, J., Wang, L., Liu, Z., et al.: Chemotactic feedback controls patterning in hybrid tumor–stroma model. Bulletin of Mathematical Biology 88, 169 (2026)

[76] Xu, W., Lu, J., Liu, W.-R., Anwaier, A., Wu, Y., Tian, X., Su, J.-Q., Qu, Y.- Y., Yang, J., Zhang, H., Ye, D.: Heterogeneity in tertiary lymphoid structures predicts distinct prognosis and immune microenvironment characterizations of clear cell renal cell carcinoma. J Immunother Cancer 11(12), 006667 (2023)

[77] Tang, Z., Bai, Y., Fang, Q., Yuan, Y., Zeng, Q., Chen, S., Xu, T., Chen, J., Tan, L., Wang, C., Li, Q., Lin, J., Yang, Z., Wu, X., Shi, G., Wang, J., Yin, C., Guo, J., Liu, S., Peng, S., Kuang, M.: Spatial transcriptomics reveals tryptophan metabolism restricting maturation of intratumoral tertiary lymphoid structures. Cancer Cell 43(6), 1025–1044 (2025)

[78] Mongan, J., Moy, L., Kahn Jr, C.E.: Checklist for artificial intelligence in medical imaging (CLAIM): a guide for authors and reviewers. Radiological Society of North America (2020)