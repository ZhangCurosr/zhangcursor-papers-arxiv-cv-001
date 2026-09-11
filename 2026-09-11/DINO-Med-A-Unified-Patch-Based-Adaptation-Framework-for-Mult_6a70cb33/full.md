# DINO-Med: A Unified Patch-Based Adaptation Framework for Multi-Modal Medical Image Analysis Applied to Liver Fibrosis Staging

Boya Wang<sup>1⋆</sup>, Ruizhe Li<sup>1,2</sup>, Chao Chen<sup>1</sup>, and Xin Chen<sup>1</sup>

<sup>1</sup> School of Computer Science, University of Nottingham, UK <sup>2</sup> Nottingham Biomedical Research Centre (BRC), School of Medicine, University of Nottingham, UK

Abstract. Adapting natural-image foundation models like DINOv3 to multi-modal medical imaging is challenging due to the significant domain gap between natural color images and multi-channel medical scans. We present a unified, patch-based framework that processes raw multimodal imaging through training-free registration, automated localization, and mask-filtered patch extraction. This architecture culminates in a hierarchical strategy that aggregates patch-level insights into subjectlevel diagnostics. Using liver fibrosis staging as a case study, we evaluate four patch-level feature representations: handcrafted Radiomics features, learned ResNet features, pre-trained foundation model SAM-Med2D features, and frozen DINOv3 features. To ensure a controlled comparison, all models utilize the same lightweight MLP head and are evaluated across both rigid and deformable registration settings. Our training protocol focuses on mild fibrosis (S1) and cirrhosis (S4) classes only, enabling a single classifier to address both substantial fibrosis detection and cirrhosis staging. Evaluated via 10 random train (90%)/ test (10%) splits on 360 subjects from the CARE 2025 Liver Track 4 cohort, our DINOv3- based framework significantly outperforms all baselines, achieving the best classification accuracy of 78.4% for S1 and 75.8% for S4.

Keywords: Multi-modality· DINOv3 · Patch-based · Classification · Subject-level aggregation · Liver fibrosis.

## 1 Introduction

In the clinical domain, the ability to extract precise, high-dimensional features from medical imaging is fundamental to accurate diagnosis and personalized treatment planning. Unlike natural images, medical data, such as multi-parametric MRI, contains subtle textural patterns and complex anatomical relationships that are often imperceptible to the human eye. Efective feature learning and extraction are therefore essential to transform these raw signals into discriminative biomarkers that can diferentiate disease stages with high sensitivity, forming the backbone of automated clinical decision support.

The recent emergence of large-scale, pre-trained foundation models has initiated a new era in medical image analysis. Driven by extensive datasets and self-supervised learning (SSL), these models act as robust, task-agnostic feature extractors. Segmentation-centric models such as Segment Anything Model (SAM) [13] and its medical domain-adapted variants, SAM-Med2D [6] and SAM-Med3D [26], have demonstrated exceptional zero-shot capabilities in mapping complex anatomical structures. Conversely, representation-centric models such as the DINO series (DINOv1 [5], DINOv2 [18], and the latest DINOv3 [20]) have set new standards in visual understanding. Trained on extensive natural image corpora, the DINO architecture generates highly discriminative dense representations that transfer exceptionally well to downstream tasks, often eliminating the need for exhaustive end-to-end fine-tuning [5, 18, 20].

Despite the remarkable capabilities of vision foundation models for feature extraction, a significant "domain gap" persists between natural color images and multi-modality medical imaging. This discrepancy in intensity distributions, spatial resolution, and dimensionality often limits the direct applicability of these models in clinical settings. Consequently, determining the optimal strategy to adapt pre-trained encoders and efectively fuse multi-modal representation features remains a critical and worthwhile challenge to explore.

In this work, we present a generic, unified patch-based adaptation framework designed to fully unlock the representational power of the DINOv3 foundation model for multi-modality medical imaging. Our approach does not merely propose a high-performing task-specific pipeline, but fundamentally investigates how natural-image foundation models can be efectively adapted to handle the complexities of multi-modality medical data for subject-level classification tasks.

To rigorously test the clinical relevance and robustness of our generic framework, we apply it to the challenging task of liver fibrosis staging. Early detection and accurate staging of liver fibrosis are crucial for efective clinical management and patient prognosis, as untreated fibrosis may deteriorate into cirrhosis or hepatocellular carcinoma. Although percutaneous liver biopsy currently serves as the diagnostic standard, its routine clinical translation is fundamentally limited by procedural invasiveness, susceptibility to sampling errors, and the risk of severe complications [2, 3]. To overcome these limitations, non-invasive techniques, specifically multi-parametric magnetic resonance images (MRIs), have rapidly become the primary option for assessing fibrotic disease [1, 16]. By applying our framework to this specific task, we demonstrate that an optimally adapted DINOv3 feature extractor can efectively address these clinical challenges, providing a robust, non-invasive alternative for accurate disease staging. The main contributions of this paper are summarized as follows:

1. A Unified Patch-Based Adaptation Framework: We propose a comprehensive, end-to-end diagnostic pipeline capable of processing raw, unaligned multi-modal medical images. The workflow integrates training-free registration [24], automated localization, and mask-guided patch extraction, culminating in a hierarchical classification strategy that aggregates patch-level insights into subject-level diagnostics. To ensure a rigorous, unbiased evaluation, all upstream preprocessing and downstream aggregation modules are strictly standardized across all experimental cohorts.

2. Frozen DINOv3 Multi-Modal Descriptors: We leverage the pre-trained DI-NOv3 backbone as a frozen feature extractor, creating a potent multi-modal descriptor through independent feature concatenation. This simple, finetuning-free approach establishes a robust performance baseline, demonstrating the model’s capacity for high-fidelity medical representation learning.

3. Clinical Benchmarking in Liver Fibrosis: Using liver fibrosis staging as a rigorous testbed based on the MICCAI CARE 2025 Liver Track 4 dataset [15], we evaluate our framework against radiomics, task-specific ResNet [25], medical domain fine-tuned SAM-Med2D baselines. Our results show that frozen DINOv3 consistently outperforms all baselines significantly across rigid and deformable registration settings.

## 2 Related Works

## 2.1 Radiomics and Deep Learning for Feature Extraction

In recent studies, deep learning and radiomics have played a central role in medical image analysis, improving diagnostic accuracy and clinical interpretability. Traditional radiomics approaches involve extracting high-dimensional features, such as shape, intensity, and texture features, from medical images to represent tissue characteristics [9, 19]. Deep learning architectures, in particular Convolutional Neural Networks (CNNs) [22, 14, 21], have shown great potential for automatically extracting hierarchical representations from MRIs [29]. To capture broader contextual dependencies, Vision Transformers (ViTs) have recently been adopted, leveraging self-attention mechanisms to produce superior feature representations [7, 17]. Current methodological trends emphasize the integration of multi-view learning and uncertainty modeling to improve robustness and interpretability simultaneously.

## 2.2 Pretrained Foundation Models in Medical Imaging

The emergence of large-scale, self-supervised visual foundation models has transformed representation learning in computer vision and is beginning to change the landscape of medical imaging analysis [12]. Foundation models have become increasingly important in medical image analysis due to their ability to produce transferable visual representations through large-scale pretraining, thereby reducing dependence on task-specific annotation. There are two main directions in recent works. One is adapting general-purpose vision backbones pre-trained on natural images. The DINO family [5, 18, 20] is the representative series that has shown strong transferability, high quality, and robust patch-level representations, and therefore has become a widely used backbone for downstream tasks. The other is directly developing for medical tasks by using large-scale clinical medical imaging datasets. In segmentation, SAM-Med2D [6] and SAM-Med3D [26] both adapt the Segment Anything Model [13] to medical imaging datasets. For multimodal learning, BiomedCLIP [31] has been trained on large-scale biomedical image-text datasets (PMC-15M) and can transfer performance strongly across diverse biomedical vision-language tasks. More recently, RadFM [27] has been extended to radiology. However, efective adaptation of natural-image foundation models to multi-modal medical imaging remains underexplored.

## 2.3 Liver Fibrosis Staging

Liver fibrosis staging has transitioned from invasive biopsies toward non-invasive multi-parametric MRI (mpMRI), which ofers superior soft-tissue contrast through complementary imaging sequences. Initial computational eforts primarily utilized radiomics, combining handcrafted features, such as shape, first-order intensity, and texture, with traditional classifiers or shallow neural networks [19]. However, these methods were constrained by their heavy reliance on precise segmentation and manual feature engineering, limiting their scalability and robustness. With the maturation of deep learning, convolutional neural networks (CNNs) became the standard for direct representation learning [30]. More recently, patch-based approaches applied to mpMRI $( \mathrm { e . g . }$ , Wang et al. [25]) have demonstrated strong performance on benchmarks like the MICCAI CARE2025 Liver Challenge [15]. Despite these gains, there remains a significant research gap in systematically exploring how state-of-the-art vision models, pretrained on natural images, can be efectively adapted to the application of multi-parametric MRI or multimodal medical imaging in general.

## 3 Methods

## 3.1 Overview

The proposed patch-based framework (Fig. 1) leverages DINOv3 representations for subject-level classification in multimodal medical imaging through a two-stage pipeline. In the preprocessing stage, raw unaligned 3D volumes are coregistered to a common space and cropped to a mask-defined Region of Interest (ROI), yielding standardized $K \times 2 2 4 \times 2 2 4$ axial slices. During feature representation and classification, a frozen DINOv3 encoder extracts $P \times D$ feature matrices per modality, where $P$ is the total number of patches and D is the embedding dimension. Then, a mask-guided selection module is applied to select S valid patches from the ROI. These features of valid patches are concatenated across the N modalities into a unified $S \times ( N \times D )$ matrix, processed by a lightweight Multi-layer Perceptron neural network (MLP) to generate patch-wise probability maps, and finally pooled by an aggregation module for the subject-level prediction. While described generally, this methodology is validated in Section 4 using liver T1, T2, and DWI MRI sequences for fibrosis staging.

![](images/b1370cff3110e8504628987d047619ce11e3cc9051c8d65e37947ffb36f3fc4a.jpg)  
Fig. 1: Overview of the proposed patch-based multimodal framework. The pipeline begins by co-registering multi-modality 3D volumes and cropped to a mask-defined ROI of $K \times 2 2 4 \times 2 2 4$ . Then, a frozen DINOv3 encoder extracts a $P \times D$ (number of patches × feature dimension) feature matrix per modality. S valid patches are selected from the ROI. Features from N modalities are concatenated into an $S \times ( N \times D )$ matrix, then passed through an MLP to generate patch-wise probability maps. Finally, an aggregation module pools these results for subject-level classification.

## 3.2 Preprocessing

To standardize multimodal 3D volumes for feature extraction, we first employ a registration model to establish spatial correspondence [24]. This alignment allows a reference mask to be propagated across all N modalities, ensuring uniform ROI application without modality-specific segmentations. Once aligned, a 3D bounding box isolates the target organ, expanded by 5 voxels in all directions to retain critical periorgan context. Finally, all cropped ROIs and masks are resized to 224×224 per axial slice. We preserve the original volume depth (K), resulting in a standardized input dimension of $K \times 2 2 4 \times 2 2 4$ for each modality.

## 3.3 Feature Extraction and Patch-level Classification

As a foundation model trained on massive natural image datasets, DINOv3 extracts high-quality features that transfer efectively to downstream tasks without fine-tuning [20]. However, adapting this architecture to medical imaging presents a challenge: DINOv3 requires 3-channel RGB input. Consequently, mapping multimodal MRI sequences into this format is a critical design choice, as the fusion strategy directly influences the quality of the transferred representations.

In our proposed scheme, for each modality, the 2D slice is duplicated across all three RGB channels to form a pseudo RGB image of shape 224×224×3 to feed into DINOv3. Each patch in each modality generates a feature vector with length D=384. These vectors of all modalities are then concatenated together to represent the feature of the patch.

As only the patches within the ROI should be considered for classification modeling, each 224 × 224 axial slice is partitioned into $1 4 \times 1 4 ~ ( P { = } 1 9 6 )$ nonoverlapping $1 6 \times 1 6$ patches. To filter out non-target anatomy and image borders included during bounding box expansion, we retain only patches with a maskto-patch area ratio exceeding 0.7. This threshold was selected to balance signal purity with data retention. It efectively minimizes non-organ noise while preserving critical boundary patches near the organ capsule that provide essential morphological information. This process results in $S$ valid patches, each represented by an $N \times D$ feature vector.

Subsequently, a lightweight MLP, consisting of two hidden layers (512, 128), each followed by ReLU and dropout (rate 0.2), is then trained on these features to perform patch-level binary classification. In this scheme, DINOv3 encodes each modality independently. Therefore, self-attention operates only within a modality, and cross-modality fusion is performed by the MLP head via feature concatenation.

## 3.4 Patch-to-Subject Aggregation

Following feature extraction, we aggregate the patch-level predictions to obtain the final subject-level diagnosis using liver fibrosis staging as a case-study.

In clinical practice, liver fibrosis is categorized into four progressive stages (S1–S4), ranging from mild fibrosis (S1) to cirrhosis (S4). Because diferentiating between the intermediate stages (S2 and S3) is notoriously dificult, even for experienced pathologists, direct four-class classification is highly challenging. Following the same evaluation protocol of CARE 2025 challenge [4], the problem is decomposed into binary sub-tasks: Substantial Fibrosis Detection (S1 vs. S2–S4) and Cirrhosis Detection (S4 vs. S1–S3).

The patch-level model is trained as a binary classifier using only the S1 (labeled 0) and S4 (labeled 1) classes. This design is driven by the hypothesis that fibrotic progression is frequently spatially heterogeneous, manifesting in localized regions rather than uniformly altering the entire liver parenchyma, particularly during the intermediate stages (S2, S3).

To determine a subject-level diagnosis, we evaluate all valid patches S within the liver ROI. Let $z _ { j } \in \{ 0 , 1 \}$ denote the binary prediction for the $j \cdot$ -th patch. The subject-level score s is then calculated as the proportion of patches identified as Stage 4-like across the entire volume:

$$
s = \frac { 1 } { S } \sum _ { j = 1 } ^ { S } z _ { j }\tag{1}
$$

To calculate the liver fibrosis score for each subject, we apply a fibrosis probability mapping that translates the raw patch-level proportion s into clinical likelihoods. Recognizing that the decision boundaries for mild fibrosis (S1) and

cirrhosis (S4) are distinct, we utilize two independent, piecewise linear functions anchored by thresholds $\tau _ { 1 2 }$ and $\tau _ { 3 4 }$

$$
\hat { y } _ { 1 } = \left\{ \begin{array} { l l } { 1 - \frac { 0 . 5 } { \tau _ { 1 2 } } s , } & { 0 \leq s \leq \tau _ { 1 2 } } \\ { 0 . 5 - \frac { 0 . 5 } { 1 - \tau _ { 1 2 } } ( s - \tau _ { 1 2 } ) , } & { \tau _ { 1 2 } < s \leq 1 } \end{array} \right.\tag{2}
$$

$$
\hat { y } _ { 4 } = \left\{ \begin{array} { l l } { \frac { 0 . 5 s } { \tau _ { 3 4 } } , } & { 0 \leq s \leq \tau _ { 3 4 } } \\ { \frac { 0 . 5 \left( s - \tau _ { 3 4 } \right) } { 1 - \tau _ { 3 4 } } + 0 . 5 , } & { \tau _ { 3 4 } < s \leq 1 } \end{array} \right.\tag{3}
$$

Both thresholds, $\tau _ { 1 2 }$ and $\tau _ { 3 4 }$ , are optimized on a validation set to account for clinical data distributions. Notably, while the patch-level model is trained exclusively on S1 and S4 images, the thresholds are optimized using the full range of validation data (S1, S2, S3, and S4). We strictly apply this identical scoring and mapping procedure across all evaluated configurations. Consequently, any observed variations in subject-level performance can be attributed exclusively to the eficacy of the chosen feature extraction and fusion mechanisms.

## 4 Experiments and Results

## 4.1 CARE-Liver 2025 Dataset

The CARE 2025 Liver Track 4 dataset [15, 28, 8] contains multi-parametric MRI scans from 610 patients, acquired across multiple clinical centers using three distinct scanners: the Philips Ingenia (3.0T), Siemens Skyra (3.0T), and Siemens Aera (1.5T). Each subject’s record includes a subset of sequences, including T1-weighted (T1), T2-weighted (T2), Difusion-Weighted Imaging (DWI), and four Gd-EOB-DTPA-enhanced dynamic phases (GED1–GED4). While other sequences vary by subject, the GED4 phase is available for every subject in the dataset. Because fibrosis stage labels were restricted to the training cohort, we focused our analysis on the 360 subjects with available ground truth. We utilized the non-contrast modalities (T1, T2, and DWI), which comprise 97 cases in Stage 1, 64 in Stage 2, 32 in Stage 3, and 167 in Stage 4.

To address the lack of pre-aligned data and the scarcity of liver segmentation masks (only 30 subjects were annotated), we implemented a preprocessing pipeline. First, all volumes were resampled to a uniform voxel size of 1mm×1mm×2.5mm and cropped/resized to a fixed dimension of 256×256×48. To generate the missing liver masks, we applied a semi-supervised segmentation method based on the BRBS framework [11] to the GED4 sequences. Finally, to ensure the GED4-derived masks could be used for other modalities, the T1, T2, and DWI sequences were spatially aligned to the GED4 coordinate space using the Search-MIND registration framework [24]. For any missing modalities, feature vectors of 1s are used for all methods evaluated in this paper.

## 4.2 Baseline Methods and Experimental Settings

To compare to our proposed method, three baseline methods are implemented. The evaluation protocol, evaluation metrics and parameter settings are also detailed in this section.

Radiomics-Based Classification: This baseline follows the classical radiomics paradigm, where expert-designed features are used in place of learned representations. Using PyRadiomics [23], we extract 70-dimensional texture features, including GLCM, GLRLM, GLSZM, and GLDM, from the S selected 16 × 16 patches across N modalities (T1, T2, and DWI). For each patch, the features from diferent modalities are concatenated and processed by the same lightweight MLP head described in Section 3.3. This allows us to generate patchwise probability maps and perform subject-level classification using handcrafted features as a direct comparison to the DINOv3-based approach.

Intensity-Based ResNet: As a second baseline, we employ the intensitybased ResNet approach proposed by Wang et al. [25]. In this setup, raw intensity values from the three modalities (T1, T2, and DWI) are stacked into a 16×16×3 tensor and fed into the network for end-to-end patch-level binary classification. We utilize ResNet-18 [10] as the backbone. Its architecture is specifically suited for the small 16 × 16 patch size, as deeper models risk reducing the feature map to a single pixel too early. To ensure a fair comparison with the DINOv3 pipeline, we replace the original ResNet-18 fully connected layer with the same lightweight MLP head described in Section 3.3. Additionally, all training hyperparameters, including the optimizer, learning rate, batch size, and epoch count, are matched exactly to those used for the MLP head in Section 3.3. Notably, this ResNetbased network has previously attained one of the best performances on the liver assessment task within the CARE2025 Challenge.

SAM-Based Classification: SAM-Med2D [6] is an additional baseline based on a large-scale pretrained model to compare with. This domain-specific variant of SAM[13] provides a robust, segmentation-driven baseline that has been extensively fine-tuned by utilizing comprehensive, multi-modal medical datasets. Although SAM-Med2D is purely engineered for the segmentation task, we adapt its underlying frozen image encoder to serve as a patch-level feature extractor. Due to the input requirements being diferent from the DINOv3 input size, we resize the input slices to 256 × 256. Each patch produces a feature with a length of 768. To maintain strict experimental parity with DINOv3, we apply the same single-modality replication and feature concatenation protocol to SAM-Med2D as detailed in Section 3.3.

Nested Evaluation Protocol: Given the limited sample size inherent to clinical cohorts, we adopt a nested cross-validation protocol to ensure unbiased and repeatable performance estimation.

The 360-subject cohort is stratified-sampled into a 90% development pool (324 subjects) and a 10% held-out test set (36 subjects). This partition is repeated 10 times with independent random seeds, yielding 10 independent experimental runs, each with its own test cohort. All subject-level metrics reported in Table 2 are the mean ± standard deviation computed across these 10 runs.

Table 1: Subject distribution across a single outer run’s splits (training / validation / held-out 10% test).
<table><tr><td>Dataset</td><td>S1</td><td>S2</td><td>S3</td><td>S4</td></tr><tr><td>Training</td><td>70</td><td>0</td><td>0</td><td>112</td></tr><tr><td>Validation</td><td>17</td><td>58</td><td>29</td><td>38</td></tr><tr><td>Test</td><td>10</td><td>6</td><td>3</td><td>17</td></tr></table>

Within each outer run, the 324 development subjects are further partitioned by 4-fold cross-validation into four train/validation splits (roughly 243/81 subjects each). Four patch-level S1/S4 classifiers are trained (one for each fold), and each validation fold is used exclusively to calibrate a pair of fibrosis-probability thresholds $\left( \tau _ { 1 2 } , \tau _ { 3 4 } \right)$ via the piecewise-linear mapping in Eq. (2)–(3). Due to the unbalanced numbers of S1 and S4, random duplication is used to boost the number of S1 class in the training set. Each of these four models is then tested on the 10% held-out test set in the outer run.

In total, the protocol trains $1 0 \times 4 = 4 0$ models, and each model is tested on the corresponding 10% held-out test set. Table 1 shows the quantitative subject distribution across all four fibrosis stages for a single representative outer run.

Evaluation Metrics: Following the CARE2025 Challenge evaluation protocols, two clinically critical binary subtasks are utilized to evaluate the classification performance for the liver fibrosis staging: Cirrhosis Detection (S1-S3 vs. S4) and Substantial Fibrosis Detection (S1 vs. S2-S4). The Area Under the Receiver Operating Characteristic curve (AUC) and Classification Accuracy (Acc) were computed. Statistical significance (Wilcoxon signed-rank test) is denoted by <sup>∗</sup> (p $< 0 . 0 5$ , DINO vs. Radiomics),† $\mathrm { ( p < 0 . 0 5 }$ , DINO vs. ResNet), and $: ( p < 0 . 0 5$ DINO vs. SAM-Med2D).

Parameter settings: For feature extraction, we utilized the oficial DINOv3 ViT-Small/16 checkpoint (dinov3-vits16-pretrain-lvd1689m), pretrained on the LVD-1689M dataset comprising 1.69 billion curated natural images [20]. The ViT-S/16 architecture consists of 12 transformer blocks with a 384-dimensional hidden feature space and a native 16×16 patch size, totaling approximately 21M parameters. The oficial SAM-Med2D checkpoint sam-med2d\_b was used as the SAM-Med2D baseline. For the downstream binary classification phase, models were trained for 30 epochs with a batch size of 256 using the Adam optimizer. The initial learning rate of $1 \times 1 0 ^ { - 4 }$ was managed by a step decay scheduler, which reduced the rate by a factor of 0.7 every 10 epochs.

## 4.3 Results

To evaluate how registration precision and model architecture influence subjectlevel classification, we systematically compared the radiomics, ResNet, SAM-Med2D and DINOv3 models across two registration settings (rigid and deformable) implemented via the Search-MIND framework [24]. Performance metrics (AUC and ACC) for the S1 vs. S4 binary classification task are summarized in Table 2.

Table 2: Subject-level classification performance under rigid and deformable registration across the S4 and S1 subsets. Statistical significance (Wilcoxon signedrank test) is denoted by $^ { * } \ ( p < 0 . 0 5$ , DINO vs. Radiomics), <sup>†</sup> $( p < 0 . 0 5$ , DINO vs. ResNet) and $^ \ddagger \ ( p < 0 . 0 5 ,$ DINO vs. SAM-Med2D)
<table><tr><td>Registration</td><td>Model</td><td>Mean AUC</td><td>Mean ACC</td><td>S4 AUC</td><td>S4 ACC</td><td>S1 AUC</td><td>S1 ACC</td></tr><tr><td rowspan="4"> $\mathrm { R i g i d }$ </td><td>Radiomics</td><td> $0 . 5 8 2 \pm 0 . 0 7 2$ </td><td> $0 . 6 4 4 \pm 0 . 0 2 2$ </td><td> $0 . 5 6 1 \pm 0 . 0 8 2$ </td><td> $0 . 5 4 7 \pm 0 . 0 3 3$ </td><td> $0 . 6 0 2 \pm 0 . 0 8 0$ </td><td> $0 . 7 4 0 \pm 0 . 0 3 2$ </td></tr><tr><td>ResNet</td><td> $0 . 7 5 3 \pm 0 . 0 4 4$ </td><td> $0 . 7 0 8 \pm 0 . 0 3 0$ </td><td> $0 . 7 3 8 \pm 0 . 0 4 5$ </td><td> $0 . 6 7 8 \pm 0 . 0 3 5$ </td><td> $0 . 7 6 8 \pm 0 . 0 7 2$ </td><td> $0 . 7 3 9 \pm 0 . 0 3 7$ </td></tr><tr><td>SAM-Med2D</td><td> $0 . 7 8 0 \pm 0 . 0 5 2$ </td><td> $0 . 7 2 8 \pm 0 . 0 3 8$ </td><td> $0 . 7 8 9 \pm 0 . 0 5 5$ </td><td> $0 . 7 1 9 \pm 0 . 0 4 4$ </td><td> $0 . 7 7 2 \pm 0 . 0 7 1$ </td><td> $0 . 7 3 8 \pm 0 . 0 4 5$ </td></tr><tr><td>DINO</td><td> $\mathbf { 0 . 8 4 3 \pm 0 . 0 4 1 }$ </td><td> $\mathbf { 0 . 7 6 6 \pm 0 . 0 4 0 ^ { * \dagger } }$ </td><td> $\mathbf { 0 . 8 4 8 \pm 0 . 0 5 6 }$ </td><td> $\mathbf { 0 . 7 5 8 \pm 0 . 0 5 7 ^ { \ast \dagger } }$ </td><td> $\mathbf { 0 . 8 3 8 \pm 0 . 0 6 3 }$ </td><td> $\mathbf { 0 . 7 7 4 \pm 0 . 0 5 4 }$ </td></tr><tr><td rowspan="4">Deformable</td><td>Radiomics</td><td> $0 . 5 2 7 \pm 0 . 0 6 1$ </td><td> $0 . 6 4 0 \pm 0 . 0 2 0$ </td><td> $0 . 5 1 0 \pm 0 . 0 5 8$ </td><td> $0 . 5 4 7 \pm 0 . 0 3 4$ </td><td> $0 . 5 4 3 \pm 0 . 0 9 1$ </td><td> $0 . 7 3 4 \pm 0 . 0 2 7$ </td></tr><tr><td>ResNet</td><td> $0 . 7 6 3 \pm 0 . 0 5 1$ </td><td> $0 . 7 2 2 \pm 0 . 0 2 8$ </td><td> $0 . 7 5 3 \pm 0 . 0 3 8$ </td><td> $0 . 6 8 3 \pm 0 . 0 3 3$ </td><td> $0 . 7 7 2 \pm 0 . 0 8 0$ </td><td> $0 . 7 6 1 \pm 0 . 0 3 4$ </td></tr><tr><td>SAM-Med2D</td><td> $0 . 7 8 5 \pm 0 . 0 4 9$ </td><td> $0 . 7 2 1 \pm 0 . 0 3 9$ </td><td> $0 . 7 9 8 \pm 0 . 0 5 8$ </td><td> $0 . 7 1 3 \pm 0 . 0 4 9$ </td><td> $0 . 7 7 2 \pm 0 . 0 5 9$ </td><td> $0 . 7 2 9 \pm 0 . 0 5 0$ </td></tr><tr><td>DINO</td><td> $\mathbf { 0 . 8 4 5 \pm 0 . 0 3 7 }$ </td><td> $\mathbf { 0 . 7 6 5 \pm 0 . 0 3 9 ^ { \ast \dagger \ddagger } }$ </td><td> $\mathbf { 0 . 8 4 5 \pm 0 . 0 5 6 }$ </td><td> $\mathbf { 0 . 7 4 6 \pm 0 . 0 6 4 ^ { * \dagger } }$ </td><td> $\mathbf { 0 . 8 4 5 \pm 0 . 0 5 1 }$ </td><td> $\mathbf { 0 . 7 8 4 \pm 0 . 0 5 3 ^ { \dagger } }$ </td></tr></table>

Table 2 shows the DINOv3 pipeline consistently outperforming all baselines. Under rigid registration, DINOv3 achieves a Mean AUC of 0.843 and Mean ACC of 0.766, significantly exceeding SAM-Med2D (0.780/0.728), ResNet (0.753/0.708), and Radiomics (0.582/0.644). Wilcoxon signed-rank tests confirm DINOv3’s Mean ACC is significantly superior to ResNet and Radiomics across both registration settings (marked ∗†).

Analysis of the specific stages reveals that the performance gap is most pronounced in the S4 task. DINOv3 achieves approximately 76% accuracy in both registration settings, outperforming SAM-Med2D by around 4%, ResNet by roughly 8% and Radiomics by around 20%. This suggests that the rich, pretrained representations of DINOv3 are particularly efective at capturing the macro-morphological changes characteristic of advanced cirrhosis. In contrast, while DINOv3 still maintains the highest scores in the S1 task (reaching 0.784 ACC with deformable registration), the statistical advantage over ResNet and Radiomics model is less significant, likely due to the subtle texture-based nature of early-stage fibrosis.

The impact of registration strategy (rigid vs. deformable) varied across methods. Radiomics performed better with rigid registration, likely because handcrafted features are sensitive to the pixel-level interpolations caused by deformable warping. Conversely, the ResNet approach favored deformable registration, as the model could efectively learn from the improved local alignment. Both SAM-Med2D and DINOv3 are pre-trained models, which yielded comparable outcomes regardless of the registration type, indicating greater immunity to local geometric variations.

Figure 2 visualizes the distribution of subject-level S4-predicted patch percentages across the four fibrosis stages, incorporating the optimal thresholds τ<sub>12</sub> and $\tau _ { 3 4 }$ from Table 3. This threshold gap serves as a proxy for feature quality: a larger margin indicates superior patch-level representation and class separability. DINOv3 achieves a substantial margin of 0.48 in both registration settings, significantly outperforming other methods. In contrast, radiomics margins fall

![](images/ca94890c87604e698a40884e3485b82d6dc158d8e690bb9b4ff72aa6b6b14e29.jpg)  
(a) Radiomics: Rigid

![](images/7efed25162ac2dfec6aea31fcc4fcd6e30afdec2f58badf9f65e639868b3485b.jpg)  
(b) Radiomics: Deformable

![](images/83e516b9113979e9bc5b5801ed74f2df91ec36ce717b083582a4b00e239308a9.jpg)  
(c) ResNet: Rigid

![](images/4ccd59a5ae1aba712897818279e83c4e5e57a42308d4a677c590fe2dbd3c78d8.jpg)  
(d) ResNet: Deformable

![](images/81e8f97e96257a6ce8e4ccaa476d669adf3d62212b69e93cd40845f68ca60a2b.jpg)  
(e) SAM-Med2D: Rigid

![](images/6a231d5e9e0e8cf90f4dec1cf5a6727cbc4647ee6ffd5c182e805896cd605064.jpg)  
(f) SAM-Med2D: Deformable

![](images/f2ca72148c04dd974cfee61469843d777771903bee9173ac4d9111ed445a2a76.jpg)  
(g) DINO: Rigid

![](images/46d8fa65ee168385fec8c7dbdd3560849e83673e113951fae3f27bce838c9722.jpg)  
(h) DINO: Deformable

Fig. 2: Qualitative comparison of rigid vs. deformable registration across four models. Box plots show the median, interquartile range, and subject-level values (orange dots) over 10 cross-validation rounds. Dashed lines denote thresholds for substantial fibrosis $\left( \tau _ { 1 2 } \right)$ and cirrhosis $( \tau _ { 3 4 } ) ;$ the vertical gap between them represents the model’s discriminative margin.

below 0.05. While ResNet improves on radiomics, its narrow gap suggests dificulty distinguishing intermediate stages (S2, S3) from extremes (S1, S4). SAM-Med2D provides wider margins than ResNet but remains less discriminative than DINOv3.

Table 3: Stage 1 and Stage 4 thresholds of patch-based classification under rigid and deformable registration
<table><tr><td>Registration</td><td>Model</td><td>T12</td><td>T34</td></tr><tr><td rowspan="4">Rigid</td><td>Radiomics</td><td> $0 . 4 7 1 \pm 0 . 0 1 7$ </td><td> $0 . 5 0 6 \pm 0 . 0 0 8$ </td></tr><tr><td>ResNet</td><td> $0 . 5 0 9 \pm 0 . 0 4 8$ </td><td> $0 . 6 2 4 \pm 0 . 0 3 5$ </td></tr><tr><td>SAM-Med2D</td><td> $0 . 3 7 4 \pm 0 . 0 5 2$ </td><td> $0 . 6 8 6 \pm 0 . 0 2 5$ </td></tr><tr><td>DINO</td><td> $0 . 3 5 5 \pm 0 . 0 3 9$ </td><td> $0 . 8 3 3 \pm 0 . 0 2 8$ </td></tr><tr><td rowspan="4">Deformable</td><td>Radiomics</td><td> $0 . 4 7 1 \pm 0 . 0 0 9$ </td><td> $0 . 5 0 2 \pm 0 . 0 0 7$ </td></tr><tr><td>ResNet</td><td> $0 . 5 1 0 \pm 0 . 0 3 7$ </td><td> $0 . 6 5 1 \pm 0 . 0 3 7$ </td></tr><tr><td>SAM-Med2D</td><td> $0 . 3 4 8 \pm 0 . 0 3 2$ </td><td> $0 . 7 1 7 \pm 0 . 0 5 0$ </td></tr><tr><td>DINO</td><td> $0 . 3 4 2 \pm 0 . 0 2 8$ </td><td> $0 . 8 2 0 \pm 0 . 0 2 9$ </td></tr></table>

## 5 Conclusions

This study introduces a unified, patch-based framework for liver fibrosis staging that successfully adapts the DINOv3 foundation model to multi-parametric MRI. By integrating training-free registration with hierarchical patch aggregation, our approach consistently outperformed radiomics, ResNet, and SAM-Med2D baselines, achieving peak accuracies of 78.4% (S1) and 75.8% (S4). We also observed that large foundation models (DINOv3 and SAM-Med2D) yielded comparable results across both rigid and deformable registration, indicating superior immunity to local geometric variations. Overall, this work demonstrates that frozen vision foundation models provide a powerful, scalable diagnostic baseline without the need for domain-specific fine-tuning. Future research will explore lightweight adapters (e.g., LoRA) to bridge the domain gap, extend the framework to cardiac and prostate multi-parametric MRI, and investigate early-fusion and late-fusion strategies to capture cross-sequence spatial dependencies. In addition, our objective is to address the classification of intermediate stages of fibrosis (S2 and S3), a clinically demanding task that typically requires expert consensus and is therefore often omitted from automated staging studies.

## References

1. Banerjee, R., Pavlides, M., Tunniclife, E.M., Piechnik, S.K., Sarania, N., Philips, R., Collier, J.D., Booth, J.C., Schneider, J.E., Wang, L.M., et al.: Multiparametric magnetic resonance for the non-invasive diagnosis of liver disease. Journal of hepatology 60(1), 69–77 (2014)

2. Bedossa, P., Dargère, D., Paradis, V.: Sampling variability of liver fibrosis in chronic hepatitis c. Hepatology 38(6), 1449–1457 (2003)

3. Bravo, A.A., Sheth, S.G., Chopra, S.: Liver biopsy. New England Journal of Medicine 344(7), 495–500 (2001)

4. CARE Challenge Organizers: CARE 2025 Track 4: Liver Fibrosis Quantification and Analysis. https://zmic.org.cn/care\_2025/track4/ (2025), accessed: 2026- 05-28

5. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

6. Cheng, J., Ye, J., Deng, Z., Chen, J., Li, T., Wang, H., Su, Y., Huang, Z., Chen, J., Jiang, L., et al.: Sam-med2d. arXiv preprint arXiv:2308.16184 (2023)

7. Dai, Y., Gao, Y., Liu, F.: Transmed: Transformers advance multi-modal medical image classification. Diagnostics 11(8), 1384 (2021)

8. Gao, Z., Liu, Y., Wu, F., Shi, N., Shi, Y., Zhuang, X.: A reliable and interpretable framework of multi-view learning for liver fibrosis staging. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 178–188 (2023)

9. Gillies, R.J., Kinahan, P.E., Hricak, H.: Radiomics: images are more than pictures, they are data. Radiology 278(2), 563–577 (2016)

10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

11. He, Y., Ge, R., Qi, X., Chen, Y., Wu, J., Coatrieux, J.L., Yang, G., Li, S.: Learning better registration to learn better few-shot medical image segmentation: Authenticity, diversity, and robustness. IEEE Transactions on Neural Networks and Learning Systems 35(2), 2588–2601 (2022)

12. He, Y., Huang, F., Jiang, X., Nie, Y., Wang, M., Wang, J., Chen, H.: Foundation model for advancing healthcare: challenges, opportunities and future directions. IEEE Reviews in Biomedical Engineering 18, 172–191 (2024)

13. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4015–4026 (2023)

14. Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems 25 (2012)

15. Liu, Y., Gao, Z., Shi, N., Wu, F., Shi, Y., Chen, Q., Zhuang, X.: Merit: Multiview evidential learning for reliable and interpretable liver fibrosis staging. Medical Image Analysis 102, 103507 (2025)

16. Loomba, R., Adams, L.A.: Advances in non-invasive assessment of hepatic fibrosis. Gut 69(7), 1343–1352 (2020)

17. Manzari, O.N., Ahmadabadi, H., Kashiani, H., Shokouhi, S.B., Ayatollahi, A.: Medvit: a robust vision transformer for generalized medical image classification. Computers in biology and medicine 157, 106791 (2023)

18. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

19. Park, H.J., Lee, S.S., Park, B., Yun, J., Sung, Y.S., Shim, W.H., Shin, Y.M., Kim, S.Y., Lee, S.J., Lee, M.G.: Radiomics analysis of gadoxetic acid–enhanced mri for staging liver fibrosis. Radiology 290(2), 380–387 (2019)

20. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al.: Dinov3. arXiv preprint arXiv:2508.10104 (2025)

21. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)

22. Targ, S., Almeida, D., Lyman, K.: Resnet in resnet: Generalizing residual architectures. arXiv preprint arXiv:1603.08029 (2016)

23. Van Griethuysen, J.J., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R.G., Fillion-Robin, J.C., Pieper, S., Aerts, H.J.: Computational radiomics system to decode the radiographic phenotype. Cancer research 77(21), e104–e107 (2017)

24. Wang, B., Li, R., Chen, C., Chen, X.: Search-mind: Training-free multi-modal medical image registration (2026), https://arxiv.org/abs/2604.09743

25. Wang, B., Li, R., Chen, C., Chen, X.: Semi-supervised liver segmentation and patch-based fibrosis staging with registration-aided multi-parametric mri. arXiv preprint arXiv:2602.09686 (2026)

26. Wang, H., Guo, S., Ye, J., Deng, Z., Cheng, J., Li, T., Chen, J., Su, Y., Huang, Z., Shen, Y., et al.: Sam-med3d: a vision foundation model for general-purpose segmentation on volumetric medical images. IEEE Transactions on Neural Networks and Learning Systems (2025)

27. Wu, C., Zhang, X., Zhang, Y., Hui, H., Wang, Y., Xie, W.: Towards generalist foundation model for radiology by leveraging web-scale 2d&3d medical data. Nature Communications 16(1), 7866 (2025)

28. Wu, F., Zhuang, X.: Minimizing estimated risks on unlabeled data: A new formulation for semi-supervised medical image segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(5), 6021–6036 (2023)

29. Yasaka, K., Akai, H., Abe, O., Kiryu, S.: Deep learning with convolutional neural network for diferentiation of liver masses at dynamic contrast-enhanced ct: a preliminary study. Radiology 286(3), 887–896 (2018)

30. Yasaka, K., Akai, H., Kunimatsu, A., Abe, O., Kiryu, S.: Liver fibrosis: deep convolutional neural network for staging by using gadoxetic acid–enhanced hepatobiliary phase mr images. Radiology 287(1), 146–155 (2018)

31. Zhang, S., Xu, Y., Usuyama, N., Xu, H., Bagga, J., Tinn, R., Preston, S., Rao, R., Wei, M., Valluri, N., et al.: Biomedclip: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915 (2023)