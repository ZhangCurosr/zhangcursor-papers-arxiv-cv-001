# Less Contouring, More Accuracy: Lesion-Guided ROI Deep Learning for Ovarian Ultrasound Classification

Mehran Ahmad<sup>1,2,3</sup>, Ali Abbasian Ardakani<sup>1</sup>, Afshin Mohammadi<sup>4</sup>, Alisa Mohebbi<sup>1</sup>, Gernot Kronreif<sup>2</sup>, Sepideh Hatamikia<sup>1,2,3\*</sup>

<sup>1</sup>Department of Medicine, Danube Private University (DPU), Krems, Austria.

<sup>2</sup>Austrian Centre for Medical Innovation and Technology (ACMIT), Wiener Neustadt, Austria.

<sup>3</sup>Department of Medical Physics and Biomedical Engineering, Medical University of Vienna, Vienna, Austria.

<sup>4</sup>Department of Radiology, Faculty of Medicine, Urmia University of Medical Science, Urmia, Iran.

\*Corresponding author(s). E-mail(s): sepideh.hatamikia@dp-uni.ac.at;

## Abstract

Objective: The classification and malignancy risk assessment of ovarian lesions using transvaginal ultrasound are challenging because of overlapping imaging features and operator-dependent interpretation. This study compared deep learning (DL) and machine learning (ML)-based radiomics strategies to identify an approach that maintains high diagnostic performance while minimizing annotation burden.

Methods: Two retrospective ovarian ultrasound datasets were used: the Multi-Modality Ovarian Tumor Ultrasound (MMOTU) dataset for eight-class classification and the Ovarian Ultrasound Dataset (OUD) for binary classification. Four strategies were evaluated under a unified experimental protocol: global image-based DL, lesion-guided region-of-interest (ROI)-based DL, lesion contourbased DL, and lesion contour-based radiomics with conventional ML classifiers. Four DL architectures, MaxViT-Tiny, Swin Transformer, EficientNet-B7, and ResNet18, were evaluated.For the radiomics strategy, quantitative features were extracted from mask-isolated lesion regions and classified using a radial basis

function support vector machine (SVM), k-nearest neighbors (KNN), and an artificial neural network (ANN). For the lower-sample OUD dataset, ANOVAbased feature selection was applied to reduce the initial 215-dimensional feature space before classification. Bootstrap resampling was used for robust performance evaluation. Results: The analysis included 507 patient cases across the MMOTU and OUD datasets. The lesion-guided ROI-based DL strategy achieved the strongest overall performance. MaxViT-Tiny achieved 93.10% accuracy with an AUC of 0.99 on MMOTU and 97.56% accuracy with an AUC of 0.99 on OUD. The lesion contourbased approach achieved comparable performance, with 91.81% accuracy and an AUC of 0.99 on MMOTU and 95.12% accuracy and an AUC of 0.99 on OUD, but required substantially greater annotation efort because of pixel-level contour delineation. Radiomics-based models showed competitive performance in the binary classification setting but were less efective for multiclass classification. Conclusion: Lesion-guided ROI-based DL provided the best balance between diagnostic performance and annotation burden for ovarian ultrasound classification. This strategy may ofer a more practical and scalable approach for AI-assisted ovarian lesion assessment and clinical workflow integration. Prospective multicentre validation is required to confirm its broader generalizability and clinical utility.

Keywords: Ovarian ultrasound, Deep learning, Machine learning, Lesion-guided ROI, Ovarian ultrasound classification, Radiomics, Artificial intelligence

## 1 Introduction

Transvaginal sonography (TVS) serves as the primary imaging modality for the evaluation of ovarian masses, providing real-time, high-resolution visualization of lesion morphology and internal composition [1–3]. In clinical practice, TVS enables risk stratification through standardized systems such as the International Ovarian Tumor Analysis (IOTA) models and the Ovarian-Adnexal Reporting and Data System (O-RADS), thereby supporting clinical management decisions [4, 5]. However, ultrasound interpretation remains highly operator-dependent, with substantial inter-observer variability arising from overlapping sonographic features between benign and malignant lesions and subtle distinctions among specific ovarian lesion etiologies [6, 7]. These limitations highlight the need for objective artificial intelligence (AI)-based adjunctive tools to improve diagnostic consistency and reduce subjectivity in radiological workflows.

Both DL and radiomics-based models applied to TVS have demonstrated considerable potential for assisting radiologists in ovarian lesion characterization and etiological classification. DL architectures can be integrated into clinical reporting workflows to provide rapid and objective probability estimates for malignancy and for specific diagnoses, such as serous cystadenoma, teratoma, or high-grade serous carcinoma. Such tools may help reduce inter-observer variability and support less experienced examiners in lesion triage, follow-up planning, and timely referral [8–13].

Radiomics-based approaches complement DL by providing quantitative and potentially interpretable imaging biomarkers derived from lesion texture, shape, and intensity characteristics. These features are typically extracted from segmented lesion regions and may be incorporated into existing O-RADS or IOTA-based workflows to refine risk stratification when visual characteristics overlap and to support etiological classification without requiring highly complex computational architectures [8, 14–16].

Recent studies have further demonstrated the potential of AI for ovarian ultrasound diagnosis in both binary and multiclass classification settings. For binary diagnosis, a deep convolutional neural network (DCNN) trained on pelvic ultrasound data achieved an area under the receiver operating characteristic curve (AUC) of 0.911 [17]. Feature-fusion and decision-fusion strategies applied to multimodal ultrasound achieved an AUC of 0.93 with a sensitivity of 92% [18]. An ensemble of ten pretrained convolutional neural networks (CNNs) achieved an accuracy of 92.15% and specificity of 92.92%, outperforming individual models [19]. A CNN combined with a denoising convolutional autoencoder, achieved 97.2% accuracy for normal-versus-tumor classification and 90.12% accuracy for malignancy discrimination [20]. DenseNet achieved a test accuracy of 97.5% for benign-versus-malignant classification [21], while a Swin Transformer demonstrated expert-level performance on TVS, achieving an AUC of 0.92 [22].

The field has subsequently progressed toward multiclass ovarian lesion classification, which is more challenging but potentially more clinically informative because it aims to distinguish specific ovarian conditions rather than only separate benign from malignant lesions. The MMOTU benchmark was introduced with baseline results in which EficientNetV2-M achieved a top-1 accuracy of 80.60% [23]. EficientOvaNet, a dual-branch EficientNet-B3 framework combining lesion ROI and global image features, achieved an accuracy of 91.9% [24]. However, performance across multiclass studies remains variable. A fine-tuned YOLOv8x reached only 70.6% mAP50 for eightclass classification [25], while ResNet18 achieved an accuracy of 76.2% and an F1-score of 78.2% on the three-class (OUD) dataset, comprising normal ovary, dominant follicle (DF), and polycystic ovary (PCO) images [26]. Collectively, these studies indicate that DL can achieve high performance in binary diagnosis, whereas multiclass recognition remains more sensitive to dataset size, class composition, lesion heterogeneity, and model design.

Despite these promising findings, several important gaps remain that directly afect the practical development and implementation of AI-assisted ovarian ultrasound analysis. Most existing studies have not systematically compared global image analysis with clinically practical lesion-focused annotation strategies, such as a rapidly defined rectangular ROI around the lesion versus detailed pixel-level contour delineation. Furthermore, previous studies [18–26] remain limited to a single-center dataset, and the assessment of a unified framework on diferent data from diferent centers and across diferent ovarian lesion types has not been explored. Moreover, in a busy center, the extra time needed for detailed contouring limits how easily these AI tools can be developed and adopted into routine practice. In routine clinical practice, ovarian lesion diagnosis relies predominantly on visual interpretation of ultrasound images, as illustrated in Fig. 1, which may be subjective, operator-dependent, and time-consuming.

![](images/fd72a83cef55d02097ad50891f5c92f0319f1790555a989f476826e2b3837ff0.jpg)  
Fig. 1 Clinical challenges in visual ovarian ultrasound interpretation and the role of AI-assisted diagnosis.

Therefore, the aim of this study was to compare four ultrasound-based AI strategies for ovarian lesion classification across binary and multiclass settings using two publicly available datasets. We performed a head-to-head evaluation of global image-based DL, lesion-guided ROI-based DL, lesion contour-based DL, and lesion contour-based radiomics combined with conventional ML classifiers. In particular, we investigated whether the less annotation-intensive lesion-guided ROI strategy could achieve diagnostic performance comparable to detailed contour-based approaches while substantially reducing annotation burden. Robust statistical evaluation with confidence intervals was additionally performed to assess the reliability of the observed performance diferences and to identify a practical strategy for both malignancy-related discrimination and classification of specific ovarian conditions.

## 2 Materials and methods

This study presents a comparative framework for ovarian ultrasound classification using two publicly available datasets with diferent levels of classification complexity. The MMOTU dataset [23] was evaluated as an eight-class classification problem. The OUD [26] was originally introduced as a three-class dataset comprising normal ovary, DF, and PCO images. In the present study, due to low amount of data in the normal class, we used only DF versus PCO classification task. Both source datasets are deidentified and open access; therefore, additional ethics committee approval was not required.

Four classification strategies were investigated, as illustrated in Fig. 2, 3: (A) global image-based DL, using the entire ultrasound image; (B) lesion-guided ROI-based DL, using a cropped region surrounding the lesion; (C) lesion contour-based DL, using only the radiologist-annotated lesion region, as illustrated in Fig. 2; and (D) lesion contourbased radiomics, in which quantitative features were extracted from the segmented lesion region, as illustrated in Fig. 3. The first three strategies were evaluated using four DL architectures: MaxViT-Tiny, EficientNet-B7, Swin Transformer, and ResNet-18. For the fourth strategy, radiomics and handcrafted imaging features were classified using three ML models (SVM, KNN, and ANN). All strategies used identical dataset splits, training protocols, and evaluation metrics to ensure a fair comparison.

## 2.1 Dataset description and partitioning

The MMOTU dataset [23] contains 1,469 2D ultrasound images acquired at Beijing Shijitan Hospital, Capital Medical University, from 247 patients using a Mindray Resona 8 ultrasound system. A total of 27 expert radiologists provided pixel-wise lesion masks and eight-class diagnostic labels based on pathological reports. The original MMOTU partition comprised 1,000 training images from 171 patients and 469 held-out images from 76 patients, with patient-level independence maintained between these two partitions. Because image-specific patient identifiers were not available within the 469-image held-out partition, this subset was further divided at the image level into validation (n = 237) and final test (n = 232) subsets. Therefore, patient-level independence between the validation and final test subsets could not be verified. Detailed class distributions are provided in Supplementary File S1.

The OUD [26] originally consists of 301 ultrasound examinations, with each examination corresponding to a unique patient, acquired at Fatemieh Hospital, Hamedan University of Medical Sciences, during 2023–2024. All ultrasound examinations were performed using a Philips Afiniti 50 ultrasound system by a single board-certified obstetrician-gynecologist with fellowship training in infertility. For the radiomics and lesion contour-based analyses, segmentation masks were manually generated for the DF and PCO classes. The normal-ovary class was excluded because the number of normal cases was limited. Consequently, 260 images from 260 patients were included in the binary DF-versus-PCO classification task. Detailed class distributions are provided in Supplementary File S1.

## 2.2 Proposed ovarian ultrasound classification strategies

Image preprocessing and training-time augmentation procedures applied across the investigated models are described in Supplementary File S2. Four classification strategies were implemented under a unified experimental protocol.

(A) Global image-based DL. In the first strategy, classification was performed using the complete ultrasound image without lesion masking or cropping. This strategy was designed to determine whether the DL models could learn discriminative ovarian lesion patterns directly from the full ultrasound frame. Both the lesion and adjacent tissues were retained, allowing the network to capture lesion-specific characteristics together with potentially informative surrounding anatomical context.

![](images/8055a4edb042e1aae8ece90fd341056eef19aaa3b313e0ff0e8d738da64d706c.jpg)  
Fig. 2 The three deep learning strategies are based on the global ultrasound image, lesion-guided ROI, and expert-annotated lesion contour.

(B) Lesion-guided ROI-based DL. In the second strategy, classification was performed using a lesion-guided ROI-based DL framework that focused the learning process on the lesion region rather than on the complete ultrasound image. A rectangular ROI surrounding the lesion was extracted while preserving the lesion and its immediate local context. The objective was to reduce the influence of irrelevant background structures while retaining clinically relevant contextual information surrounding the lesion, including the cyst wall, septa, papillary projections, and solid components commonly considered during ovarian lesion assessment using IOTA and O-RADS descriptors. This strategy was specifically designed to investigate whether lesion-guided localization could improve classification performance while reducing the annotation time and expertise required for precise pixel-level lesion contouring.

![](images/974d9100a54458da5077a9cd0b4a5b0f812f7e426bb48d263742f13614515d81.jpg)  
Fig. 3 The lesion contour-based radiomics strategy, including lesion masking, extraction of PyRadiomics and handcrafted ultrasound features, feature integration, and classification using SVM, KNN, and ANN.

(C) Lesion contour-based DL. In the third strategy, ovarian ultrasound classification was performed using a lesion contour-based DL framework that restricted the model input more specifically to the expert-annotated lesion region. Pixels outside the radiologist-annotated lesion contour were excluded, thereby reducing the contribution of surrounding tissues and background structures. This strategy was designed to encourage the models to learn predominantly from lesion-specific morphological and textural characteristics. It was evaluated to determine whether precise lesion isolation could provide additional classification benefits compared with the lesion-guided ROI strategy, which retained both the lesion and its immediate surrounding context.

(D) Lesion contour-based radiomics. In the fourth strategy, quantitative imaging features were extracted exclusively from the mask-isolated lesion region and subsequently classified using conventional machine learning algorithms. A total of 102 original PyRadiomics descriptors and 113 handcrafted ultrasound-specific features were initially extracted, resulting in a 215-dimensional feature vector. For MMOTU, all extracted features were retained for classification. For OUD, due to the limited sample size relative to the feature dimensionality, ANOVA-based feature selection was applied to rank features according to their discriminatory ability between DF and PCO classes.

Further methodological details regarding the DL models are provided in Supplementary File S3, while details of the ML models and the four classification strategies are provided in Supplementary File S4. Fig. 2 illustrates the schematic workflow for the three DL-based strategies, whereas Fig. 3 presents the schematic workflow for the lesion contour-based radiomics strategy.

## 2.3 Experimental protocol and model optimization

To ensure a fair comparison, all classification strategies were evaluated using consistent experimental settings within each dataset. Each dataset was split into training, validation, and test sets in a 70:15:15 ratio, and the same split was used across all strategies For the DL-based experiments, MaxViT-Tiny, Swin Transformer, EficientNet-B7, and ResNet-18 were trained using the same optimization settings.

ResNet-18 was additionally included as an architecture-matched baseline to facilitate comparison with the original OUD study [26], in which ResNet-18 was reported as the best-performing architecture on the original three-class classification task comprising normal ovary, DF, and PCO images. In the present study, ResNet-18 was independently retrained under the same experimental protocol used for the reformulated binary DF-versus-PCO subset.

Model optimization was performed using the AdamW optimizer with an initial learning rate of $3 \times 1 0 ^ { - 4 }$ and a weight decay of 0.05. A linear warm-up followed by cosine learning-rate decay was applied, with the first five epochs used for warm-up and the remaining epochs following the cosine decay schedule. All DL models were trained for a maximum of 100 epochs with a batch size of 16. Early stopping was applied based on validation loss with a patience of 15 epochs to reduce overfitting. Soft-target crossentropy loss was used in conjunction with MixUp and CutMix augmentation. For each experiment, the checkpoint achieving the highest validation accuracy was retained for final evaluation on the corresponding test set.

For the radiomics and handcrafted feature-based framework, multiple machine learning classifiers were evaluated, including SVM, ANN, and KNN, as discussed in Supplementary File S4.

All experiments were implemented in Python 3.11.9 using PyTorch 2.6.0, torchvision 0.21.0, CUDA 12.4, timm 1.0.25, NumPy 2.4.3, Pillow 12.1.1, scikit-learn 1.8.0, and Matplotlib 3.10.8. Experiments were conducted on Windows 11 using an NVIDIA RTX 4000 Ada Generation GPU.

## 2.4 Performance evaluation and statistical analysis

The performance of the proposed methods was evaluated on the corresponding test sets using top-1 accuracy, top-2 accuracy, specificity, sensitivity (recall), and AUC. For OUD radiomics, one-way ANOVA was used to rank the 215 candidate features by their ability to discriminate between DF and PCO classes. Feature selection and preprocessing were fitted exclusively on the training partition and then fixed for validation and test evaluation. Top-1 accuracy indicated model’s highest-probability prediction, whereas top-2 accuracy indicated whether the correct class was included among the two most probable predictions. Moreover, for each class, recall measured how many true samples of that class were correctly identified, and specificity measured how many samples from the remaining classes were correctly excluded. For the multiclass MMOTU dataset, AUC was computed using a one-vs-rest strategy, in which each class was compared against all remaining classes, with higher values indicating better performance.

The principal evaluation metrics were calculated as follows:

$$
\mathrm { A c c u r a c y } = { \frac { T P + T N } { T P + T N + F P + F N } }\tag{1}
$$

$$
\mathrm { S e n s i t i v i t y } = { \frac { T P } { T P + F N } }\tag{2}
$$

$$
{ \mathrm { S p e c i f i c i t y } } = { \frac { T N } { T N + F P } }\tag{3}
$$

where T P, T N, F P, and F N denote true positives, true negatives, false positives, and false negatives, respectively.

To quantify performance uncertainty, bootstrap resampling with 1,000 iterations was applied to the test-set predictions. In each iteration, samples were randomly drawn with replacement from the corresponding test set, and the evaluation metrics were recalculated. The resulting bootstrap distributions were used to estimate 95% confidence intervals using the percentile method.

## 3 Results

The final analysis included 1,729 ultrasound images from 507 patients across the MMOTU and OUD datasets. MMOTU contributed 1,469 images from 247 patients, whereas OUD contributed 260 images from 260 patients. Detailed class-wise distributions across the training, validation, and test sets are provided in Supplementary File S1.

## 3.1 Global image-based deep learning performance

When the complete ultrasound image was used without explicit lesion localization, performance difered according to dataset complexity as shown in Table 1. On the eight-class MMOTU dataset, top-1 accuracy across the evaluated architectures ranged from 71.1% to 81.6%, top-2 accuracy from 85.8% to 91.0%, sensitivity from 71.1% to 81.5%, and AUC from 0.93 to 0.95. MaxViT-Tiny achieved the highest top-1 accuracy of 81.6%.

On the binary OUD dataset, top-1 accuracy ranged from 85.4% to 95.1%, sensitivity from 85.4% to 95.1%, and AUC from 0.87 to 0.99. MaxViT-Tiny again achieved the highest top-1 accuracy, reaching 95.1%. However, the confidence intervals across architectures showed considerable overlap on the smaller OUD test set, limiting firm conclusions regarding diferences among individual models. Overall, global context proved informative but insuficient: the moderate multiclass accuracy suggests that background tissue and acquisition artifacts limit the separation of overlapping tumor morphologies, restricting reliability for detailed etiological assessment in routine practice.

Table 1 Performance of the global image-based deep learning models on the MMOTU and OUD datasets. Values in brackets represent 95% confidence intervals.
<table><tr><td>Data</td><td>Model</td><td>Top-1 Accuracy</td><td>Top-2 (%) Accuracy (%)</td><td>Specificity (%)</td><td>Sensitivity (%)</td><td>AUC</td></tr><tr><td rowspan="5">MMOTU MaxViT-Tiny</td><td></td><td>81.57 [76.72, 85.78]</td><td>90.09 [86.21, 93.53]</td><td>96.69 [90.80, 100.00]</td><td>81.57 [76.72, 85.78]</td><td>0.94 [0.92, 0.96]</td></tr><tr><td></td><td>78.45</td><td>91.00</td><td>96.22</td><td>78.45</td><td>0.95</td></tr><tr><td>Swin</td><td>[73.28, 83.62] 79.31</td><td>[86.21, 93.55] 89.66</td><td>[89.23, 100.00] 96.24</td><td>[73.28, 83.62] 79.31</td><td>[0.9386, 0.97] 0.95</td></tr><tr><td>EfficientNet-B7</td><td>[74.14, 84.48] 71.12</td><td>[85.34, 93.10] 85.78</td><td>[89.48, 100.00] 95.68</td><td>[74.14, 84.48] 71.12</td><td>[0.93, 0.97] 0.93</td></tr><tr><td>ResNet18</td><td>[65.09, 76.72]</td><td>[80.60, 90.09]</td><td>[88.43, 99.01]</td><td>[65.09, 76.72]</td><td>[0.91, 0.95]</td></tr><tr><td rowspan="5">OUD</td><td>MaxViT-Tiny</td><td>95.12 [87.80, 100.00]</td><td>100.00 [100.00, 100.00]</td><td>94.97 [86.80, 100.00]</td><td>95.12 [87.80, 100.00]</td><td>0.99 [0.96, 1.00]</td></tr><tr><td>EfficientNet-B7</td><td>90.24 [80.49, 97.56]</td><td>100.00 [100.00, 100.00]</td><td>87.54 [78.05, 97.56]</td><td>90.24 [80.49, 97.56]</td><td>0.99 [0.96, 1.00]</td></tr><tr><td>Swin</td><td>87.80 [78.05, 97.56]</td><td>100.00 [100.00, 100.00]</td><td>85.63 [73.17, 95.12]</td><td>87.80 [78.05, 97.56]</td><td>0.96 [0.89, 1.00]</td></tr><tr><td>ResNet18</td><td>85.37 [73.17, 95.12]</td><td>100.00 [100.00, 100.00]</td><td>81.30 [71.16, 93.23]</td><td>85.37 [73.17, 95.12]</td><td>0.87 [0.75, 0.97]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 3.2 Lesion-guided ROI-based deep learning performance

Lesion-guided ROI-centered crops with controlled perilesional padding produced stronger discrimination performance compared with the global image-based DL framework as shown in Table 2. On MMOTU, top-1 accuracy ranged from 85.7% to 93.1%, top-2 accuracy from 94.0% to 98.3%, and AUC from 0.98 to 0.99, with MaxViT-Tiny achieving the highest top-1 accuracy of 93.1%.

On OUD, top-1 accuracy ranged from 92.7% to 97.6%, top-2 accuracy was uniformly 100%, and AUC ranged from 0.97 to 0.99, with MaxViT-Tiny attaining the highest top-1 accuracy of 97.6%.

Clinically, this improvement over global image-based analysis demonstrates that directing model attention toward the lesion while preserving limited adjacent structures, such as the cyst wall, septa, papillary projections, and solid components, aligns closely with the lesion characteristics routinely assessed using IOTA and O-RADS descriptors. This lesion-focused strategy substantially reduced classification errors among visually similar entities and may support more confident and objective classification, particularly for less experienced operators.

## 3.3 Lesion contour-based deep learning performance

Strict contour-based isolation of the lesions yielded robust performance comparable to that of the lesion-guided ROI-based DL framework as shown in Table 3. On MMOTU, top-1 accuracy ranged from 85.8% to 91.8%, top-2 accuracy from 94.0% to 97.0%, and AUC from 0.98 to 0.99, with MaxViT-Tiny achieving the highest top-1 accuracy of 91.8%.

On OUD, top-1 accuracy ranged from 92.7% to 95.1%, top-2 accuracy reached 100%, and AUC ranged from 0.98 to 0.99. Although performance remained high, the absence of consistent gains over the simpler lesion-guided ROI-based approach indicates that complete removal of the perilesional context may discard subtle supportive features.

Table 2 Performance of the lesion-guided ROI-based deep learning models on the MMOTU and OUD datasets. Values in brackets represent 95% confidence intervals.
<table><tr><td>Data</td><td>Model</td><td>Top-1 Accuracy</td><td>Top-2 (%) Accuracy (%)</td><td>Specificity (%)</td><td>Sensitivity (%)</td><td>AUC</td></tr><tr><td rowspan="5">MMOTU MaxViT-Tiny</td><td></td><td>93.10 [89.66, 96.12]</td><td>96.98 [94.83, 99.14]</td><td>98.99 [94.66, 100.00]</td><td>93.10 [89.66, 96.12]</td><td>0.99 [0.98, 0.99]</td></tr><tr><td>EfficientNet-B7</td><td>88.79</td><td>96.55</td><td>98.91</td><td>88.79</td><td>0.99</td></tr><tr><td></td><td>[84.48, 92.67] 87.07</td><td>[93.97, 98.71] 98.28</td><td>[94.51, 100.00] 97.77</td><td>[84.48, 92.67] 87.07</td><td>[0.98, 0.99] 0.99</td></tr><tr><td>Swin</td><td>[82.76, 90.95] 85.78</td><td>[96.55, 99.57] 93.97</td><td>[92.86, 100.00] 98.49</td><td>[82.76, 90.95] 85.78</td><td>[0.98, 0.99] 0.97</td></tr><tr><td>ResNet18</td><td>[81.03, 90.10]</td><td>[90.95, 96.56]</td><td>[93.66, 100.00]</td><td>[81.03, 90.10]</td><td>[0.96, 0.98]</td></tr><tr><td rowspan="5">OUD</td><td>MaxViT-Tiny</td><td>97.56 [92.68, 100.00]</td><td>100.00 [100.00, 100.00]</td><td>98.09 [93.58, 100.00]</td><td>97.56 [92.68, 100.00]</td><td>0.99 [0.98, 1.00]</td></tr><tr><td>EfficientNet-B7</td><td>92.68 [82.93, 100.00]</td><td>100.00 [100.00, 100.00]</td><td>94.28 [90.81, 100.00]</td><td>92.68 [82.93, 100.00]</td><td>0.97</td></tr><tr><td></td><td>95.12</td><td>100.00</td><td>96.18</td><td>95.12</td><td>[0.92, 1.00] 0.97</td></tr><tr><td>Swin</td><td>[87.80, 100.00] 92.68</td><td>[100.00, 100.00] 100.00</td><td>[92.58, 100.00] 94.18</td><td>[87.80, 100.00] 92.68</td><td>[0.92, 1.00] 0.99</td></tr><tr><td>ResNet18</td><td>[82.93, 100.00]</td><td>[100.00, 100.00] [90.31, 100.00] [82.93, 100.00] [0.97, 1.00]</td><td></td><td></td><td></td></tr></table>

This finding has practical relevance because pixel-level lesion contouring is timeintensive and may be dificult to scale in high-volume clinical settings, whereas lesionguided ROI extraction provides comparable classification performance with a markedly lower annotation burden.

Table 3 Performance of the lesion contour-based deep learning models on the MMOTU and OUD datasets. Values in brackets represent 95% confidence intervals.
<table><tr><td>Dataset</td><td>Model</td><td>Top-1 Accuracy</td><td>Top-2 (%) Accuracy (%)</td><td>Specificity (%)</td><td>Sensitivity (%)</td><td>AUC</td></tr><tr><td rowspan="5">MMOTU MaxViT-Tiny</td><td></td><td>91.81</td><td>96.98 [94.83, 99.14]</td><td>98.97 [95.31, 100.00]</td><td>91.81 [88.36, 95.26]</td><td>0.99 [0.98, 0.99]</td></tr><tr><td></td><td>[88.36, 95.26] 91.38</td><td>95.69</td><td>99.00</td><td>91.38</td><td>0.99</td></tr><tr><td>EfficientNet-B7</td><td>[87.50, 94.40] 86.21</td><td>[92.67, 98.28] 96.98</td><td>[96.01, 100.00] 97.87</td><td>[87.50, 94.40] 86.21</td><td>[0.9855, 0.99] 0.99</td></tr><tr><td>Swin</td><td>[81.47, 90.52] 85.78</td><td>[94.83, 98.72] 93.97</td><td>[94.31, 100.00] 98.49</td><td>[81.47, 90.52] 85.78</td><td>[0.98, 0.99] 0.98</td></tr><tr><td>ResNet18</td><td>[81.03, 90.10]</td><td>[90.95, 96.56]</td><td>[95.10, 100.00]</td><td>[81.03, 90.10]</td><td>[0.97, 0.99]</td></tr><tr><td rowspan="4">OUD</td><td>MaxViT-Tiny</td><td>95.12 [87.80, 100.00]</td><td>100.00 [100.00, 100.00]</td><td>96.18 [89.10, 100.00]</td><td>95.12 [87.80, 100.00]</td><td>0.99 [0.98, 1.00]</td></tr><tr><td>EfficientNet-B7</td><td>95.12 [87.80, 100.00]</td><td>100.00 [100.00, 100.00]</td><td>96.18 [89.10, 100.00]</td><td>95.12 [87.80, 100.00]</td><td>0.99 [0.97, 1.00]</td></tr><tr><td>Swin</td><td>95.12 [87.80, 100.00]</td><td>100.00 [100.00, 100.00]</td><td>96.18 [88.10, 100.00]</td><td>95.12 [87.80, 100.00]</td><td>0.98 [0.93, 1.00]</td></tr><tr><td>ResNet18</td><td>92.68 [82.93, 100.00]</td><td>100.00 [100.00, 100.00]</td><td>94.24 [86.10, 100.00]</td><td>92.68 [82.93, 100.00]</td><td>0.99 [0.97, 1.00]</td></tr></table>

## 3.4 Lesion contour-based radiomics performance

Radiomics models produced moderate classification performance as shown in Table 4. On MMOTU, top-1 accuracy ranged from 71.55% to 74.2%, while top-2 accuracy ranged from 84.0% to 87.5%. Among the evaluated machine learning classifiers, the SVM with a radial basis function kernel achieved the highest top-1 accuracy of 74.2%.

In OUD, an ANOVA-based feature-selection sweep was performed across diferent numbers of retained features. Based on this analysis, k = 60 was selected as the final feature set, with SVM, KNN, and ANN each achieving 95.12% top-1 accuracy as shown in the supplementary file Fig. S1. Using these 60 selected features, top-2 accuracy reached 100%. These results highlight radiomics as an interpretable, low-resource, complementary approach that may be suitable for binary confirmation tasks or for integration into existing reporting systems. However, the performance gap compared with DL on the multiclass MMOTU task indicates that fixed quantitative descriptors alone may be insuficient to capture the complex and nonlinear imaging patterns required for reliable eight-class diferentiation among overlapping ovarian lesion types.

Table 4 Performance of the lesion contour-based radiomics classifiers on the MMOTU and OUD datasets. Values in brackets represent 95% confidence intervals.
<table><tr><td>Dataset</td><td></td><td>Top-1 Classifier Accuracy (%) Accuracy (%)</td><td>Top-2</td><td>Specificity (%)</td><td>Sensitivity (%)</td><td>AUC</td></tr><tr><td rowspan="3">MMOTU ANN</td><td></td><td>72.41 [66.81, 77.59]</td><td>84.05 [79.31, 88.36]</td><td>94.83 [93.51, 96.08]</td><td>72.41 [66.81, 77.59]</td><td>0.92 [0.90, 0.94]</td></tr><tr><td>KNN</td><td>71.55 [65.09, 76.72]</td><td>84.91 [80.17, 89.23]</td><td>94.32 [92.78, 95.60]</td><td>71.55 [65.09, 76.72]</td><td>0.91 [0.88, 0.93]</td></tr><tr><td>SVM-RBF</td><td>74.28 [67.67, 78.88]</td><td>87.50 [83.19, 91.38]</td><td>94.93 [93.80, 96.10]</td><td>74.28 [67.67, 78.88]</td><td>0.93 [0.91, 0.95]</td></tr><tr><td rowspan="3">OUD</td><td>ANN</td><td>95.12 [87.80, 100.00]</td><td>100.00 [100.00, 100.00]</td><td>96.18 [89.45, 100.00]</td><td>95.12 [87.80, 100.00]</td><td>0.98 [0.93, 1.00]</td></tr><tr><td>KNN</td><td>95.12 [87.80, 100.00]</td><td>100.00 [100.00, 100.00] [89.45, 100.00] [87.80, 100.00] [0.92, 1.00]</td><td>96.18</td><td>95.12</td><td>0.97</td></tr><tr><td>SVM-RBF</td><td>95.12 [87.80, 100.00]</td><td>100.00 [100.00, 100.00] [89.45, 100.00] [87.80, 100.00] [0.86, 1.00]</td><td>96.18</td><td>95.12</td><td>0.95</td></tr></table>

## 3.5 Overall comparison of classification strategies

Across all evaluated frameworks, the lesion-guided ROI-based and lesion contour-based DL strategies achieved the strongest overall performance. MaxViT-Tiny delivered the highest top-1 accuracies, reaching 93.1% on MMOTU and 97.6% on OUD, with corresponding AUC values exceeding 0.99, confirming excellent threshold-independent discrimination.

Compared with the global image-based framework, lesion-focused strategies improved classification performance, particularly on the multiclass MMOTU dataset. The lesion-guided ROI approach achieved the highest overall top-1 accuracy, while contour-based DL showed comparable performance but required more detailed annotation. As summarized in Fig. 4, lesion-guided ROI cropping provided the most practical balance between diagnostic performance and annotation burden.

## 4 Discussion

This study compared four ovarian ultrasound classification strategies across two open-access datasets representing multiclass and binary classification tasks. Overall, the lesion-guided ROI-based strategy provided the most favorable balance between classification performance and annotation burden. Because TVS assessment relies substantially on morphological characteristics within the lesion–tissue interface, a modest perilesional margin retains these cues while excluding distant background and acquisition artifacts.

Comparison of Top-1 Accuracy across Four Classification Strategies on MMOTU and OUD Datasets (A) MMOTU Dataset (8-class)  
![](images/7245bba3e96401dd67d3bfc119052e24a8b1919f29f14734d5fbc4060f2c531d.jpg)

![](images/1b38302ed663cc9fe95750d1c72a1c110c22dce1db153c7c8a8e1d7f184a1cf5.jpg)  
Fig. 4 Comparison of top-1 accuracies across all evaluated models and classification strategies.

This finding has potential clinical relevance for reducing operator dependency, which remains an important source of inter-observer variability in ultrasound interpretation. The gain from global to lesion-guided inputs indicates that explicit ROI guidance improves capture of textural and structural distinctions among lesions with overlapping appearances, which may yield more consistent outputs and support less experienced operators in triage, follow-up, and referral.

From a practical perspective, improved classification performance did not require highly detailed pixel-level lesion contouring. The lesion-guided ROI-based DL framework achieved the strongest overall performance using a simpler lesion-centered crop that preserved relevant tumor information and limited surrounding context while reducing annotation burden, as illustrated in Fig. 5. This makes lesion-guided ROI annotation more practical for large-scale dataset development and clinical workflows than precise contour annotation. Global image-based DL may be afected by irrelevant image regions, whereas radiomics ofers a resource-eficient and interpretable complement to DL. However, its lower multiclass performance suggests that handcrafted and quantitative features may not fully capture the complex patterns needed for fine-grained ovarian lesion classification.

Balancing Classification Accuracy and Annotation Burden Across Strategies  
![](images/d4d275be103c7f3aa50de360f2f142edf74ef018588b1b8b089c34d3bf0db2ea.jpg)  
Fig. 5 Efect of lesion localization strategy and annotation complexity on ovarian ultrasound classification performance.

Previous studies using the MMOTU and OUD datasets have demonstrated the potential of AI for ovarian ultrasound classification; however, systematic comparisons of diferent lesion representation strategies remain limited, as shown in Table 5. On MMOTU, [23] reached 80.60% top-1 accuracy, [24] reached 91.9% accuracy, and [25] reached 70.6% mAP50, whereas our lesion-guided MaxViT-Tiny achieved 93.10% accuracy on the same eight-class task.

The original OUD study [26] reported 76.2% accuracy using ResNet18 for threeclass classification. However, our study reformulated OUD as a binary DF-versus-PCO task because of the limited number of normal-ovary cases, so the two results are not directly comparable. To enable a fair within-study comparison, ResNet18 was retrained under the same binary setting, achieving 85.37% accuracy with global images and 92.68% with lesion-guided ROI inputs, corresponding to a 7.31% improvement with lesion guidance. Notably, contour-based ResNet18 achieved the same 92.68% accuracy, indicating that precise contouring did not provide an additional top-1 accuracy gain. Furthermore, lesion-guided MaxViT-Tiny achieved the highest accuracy of 97.56%, outperforming lesion-guided ResNet18 by 4.88%

Together, these comparisons suggest that the strategy used to define the learning region is an important determinant of classification performance. A simple lesionguided ROI achieved performance comparable to, and in some settings better than, precise contour-based isolation while requiring less detailed annotation. The systematic comparison of all four strategy under a consistent experimental protocol therefore highlights lesion representation as an important consideration in ovarian ultrasound AI development and supports lesion-guided ROI localization as a potentially scalable approach.

Table 5 Comparison of the proposed approach with previous studies on the MMOTU and OUD datasets.
<table><tr><td>Study (Year)</td><td>Dataset</td><td>Task</td><td>Method</td><td>Best Accuracy</td><td>AUC</td></tr><tr><td></td><td>MMOTU</td><td></td><td></td><td>70.6%</td><td></td></tr><tr><td>(2024) [25]</td><td>(public) MMOTU</td><td>8-class</td><td>YOLOv8x</td><td>(mAP50)</td><td></td></tr><tr><td>(2025) [23]</td><td>(public) MMOTU</td><td>8-class</td><td>EfficientNetV2-M (whole image)</td><td>80.60%</td><td></td></tr><tr><td>(2026) [24]</td><td>(public)</td><td>8-class</td><td>EfficientOvaNet (EfficientNet- B3, ROI + global; pixel masks)</td><td>91.9%</td><td>0.980</td></tr><tr><td>Proposed</td><td>MMOTU (public)</td><td>8-class</td><td>Lesion-guided ROI + MaxViT-Tiny (single model, bounding box)</td><td>93.10%</td><td>0.991</td></tr><tr><td>(2025) [26] (public)</td><td>OUD</td><td>3-class (normal/DF/PCO) ResNet18</td><td></td><td>76.2%</td><td></td></tr><tr><td>(2025) [26] (Binary)</td><td>OUD</td><td>2-class DF vs PCO</td><td>Global image ResNet18 architecture-matched baseline</td><td>85.37%</td><td>0.879</td></tr><tr><td>Proposed (Binary)</td><td>OUD</td><td>2-class DF vs PCO</td><td>Lesion-guided ROI ResNet18</td><td>92.68%</td><td>0.995</td></tr><tr><td>Proposed(Binary)</td><td>OUD</td><td>2-class DF vs PCO</td><td>Lesion contour ResNet18</td><td>92.68%</td><td>0.995</td></tr><tr><td>Proposed(public)</td><td>OUD</td><td>2-class DF vs PCO</td><td>Lesion-guided ROI + MaxViT-Tiny</td><td>97.56%</td><td>0.997</td></tr></table>

This study has several limitations. The analysis was based on public datasets and B-mode ultrasound images; therefore, further validation using heterogeneous clinical data, additional imaging modalities such as Doppler or contrast-enhanced ultrasound, and evaluation of computational requirements are needed before clinical implementation.

Future studies should investigate annotation-eficient approaches, including weakly supervised learning and automated lesion localization, as well as multimodal integration of ultrasound imaging with relevant clinical information. Prospective multicenter validation across diferent clinical settings, patient populations, and ultrasound systems will be essential to determine the generalizability and real-world applicability of the proposed framework.

## 5 Conclusion

This study suggests that a lesion-guided ROI DL strategy may ofer a practical balance between diagnostic performance and annotation feasibility for ovarian ultrasound classification. By directing attention to the lesion while retaining limited surrounding context, this approach could support more consistent interpretation with reduced clinician burden compared with global or strictly contour-based methods. Prospective multicenter validation is needed to confirm the performance of the proposed technique and its potential clinical utility.

Supplementary information. Supplementary material is available with the online version of this article.

## Declarations

## Funding

This research has been funded by FTI-Dissertationen grant with project number: FTI24-D-008.

## Competing interests

The authors declare no competing interests.

## Ethics approval

This study involved secondary analysis of publicly available, de-identified ovarian ultrasound datasets. No new participants were recruited, and no new patient data were collected for this study. Ethical approval and patient consent for the original data collection were reported by the respective dataset providers.

## Data availability

The datasets analyzed in this study are publicly available. The Multi-Modality Ovarian Tumor Ultrasound (MMOTU) dataset is described in [23], and the Ovarian Ultrasound Dataset (OUD) is described in [26].

## References

[1] Virarkar, M., Ortiz Cordero, R.G., Assumpcao, M.H., Guevara Tirado, O.A., Bhosale, P.: Figo 2025 gynecologic cancers: An ultrasound-focused imaging update. Ultrasound Q. 42(2) (2026)

[2] Lakany, M., Sharif, A., Alazzam, M., Howell, C., Mitchell, S., Pappa, C., Shibli, D., Story, L., Sayasneh, A.: Intraoperative ultrasound as a decision-making tool in modern gynecologic oncology. J. Pers. Med. 15(7) (2025)

[3] Lems, E., Koch, A.H., Delvaux, E., Leemans, J.C., Bongers, M.Y., Lok, C.A.R., Ramaekers, B.L., Geomini, P.: Diagnostic accuracy of ultrasound models for assessment of ovarian tumors: systematic review and meta-analysis. Ultrasound Obstet. Gynecol. 67(5), 590–603 (2026)

[4] Yang, Y., Wang, H., Su, N., Gao, L., Gu, Y., Cai, S., Dai, Q., Li, J., Jiang, Y.: Orads us versus iota simple rules in the diagnosis of benign and malignant adnexal masses: a prospective study. BMC Med. Imaging 25(1), 297 (2025)

[5] Almeida, G., Bort, M., Alc´azar, J.L.: Comparison of the diagnostic performance of ovarian adnexal reporting data system (o-rads) with iota simple rules and adnex model for classifying adnexal masses: A head-to-head meta-analysis. J. Clin. Ultrasound 53(7), 1574–1583 (2025)

[6] Deslandes, A., Avery, J.C., Chen, H.T., Leonardi, M., Knox, S., Lo, G., O’Hara, R., Condous, G., Hull, M.L.: Intra- and interobserver agreement of proposed objective transvaginal ultrasound image-quality scoring system for use in artificial intelligence algorithm development. Ultrasound Obstet. Gynecol. 65(3), 364–371 (2025)

[7] Wang, H., Wang, L., An, S., Ma, Q., Tu, Y., Shang, N., Pan, Y.: American college of radiology ovarian-adnexal reporting and data system ultrasound (orads): Diagnostic performance and inter-reviewer agreement for ovarian masses in children. Front. Pediatr. 11, 1091735 (2023)

[8] Zeng, S., Jia, H., Zhang, H., Feng, X., Dong, M., Lin, L., Wang, X., Yang, H.: Multimodal ultrasound-based radiomics and deep learning for diferential diagnosis of o-rads 4–5 adnexal masses. Cancer Imaging 25(1), 64 (2025)

[9] Wang, Y., Zhang, J., He, Y., Wang, X., Wu, X., Zhang, W., Gong, M., Gao, D., Liu, S., Liu, P., et al.: Ultrasound-based deep learning model as an assistant improves the diagnosis of ovarian tumors: a multicenter study. Insights Imaging 16(1), 221 (2025)

[10] Su, C., Miao, K., Zhang, L., Yu, X., Guo, Z., Li, D., Xu, M., Zhang, Q., Dong, X.: Multimodal deep learning based on ultrasound images and clinical data for better ovarian cancer diagnosis. J. Imaging Inform. Med. 39(2), 1168–1180 (2026)

[11] Dai, W.L., Wu, Y.N., Ling, Y.T., Zhao, J., Zhang, S., Gu, Z.W., Gong, L.P., Zhu, M.N., Dong, S., Xu, S.C., et al.: Development and validation of a deep learning pipeline to diagnose ovarian masses using ultrasound screening: a retrospective multicenter study. EClinicalMedicine 78, 102923 (2024)

[12] Shen, J., Fang, Y.H., Ding, J.Y., Jiao, R.S., Niu, Y.N., Huang, L., Jin, C., Chen, H.: Multitask deep learning models for ultrasound image analysis: identification of high-grade serous ovarian cancer and segmentation of tumor regions and intratumoral solid components. J. Ovarian Res. 19(1) (2026)

[13] Ji, X., Liu, C., Hu, J., Li, S., Wang, L., Cheng, X., Liu, C., Zhang, Y.: A multicenter deep learning framework integrating radiomics and vision transformers for comprehensive ovarian tumor analysis from ultrasound imaging. Eur. J. Med. Res. 31(1), 252 (2026)

[14] Moro, F., Ciancia, M., Sciuto, M., Baldassari, G., Tran, H.E., Carcagn\`ı, A., Fagotti, A., Testa, A.C.: Performance of radiomics analysis in ultrasound imaging for diferentiating benign from malignant adnexal masses: A systematic review and meta-analysis. Acta Obstet. Gynecol. Scand. 104(8), 1433–1442 (2025)

[15] Xie, W., Wang, Y., Du, Z., Chen, Y., Ke, X., Wu, T., Wang, Z., Tang, L.: A nomogram combining clinical features, o-rads us, and radiomics based on ultrasound imaging for diagnosing ovarian cancer. Sci. Rep. 15(1), 19279 (2025)

[16] Huang, Y., Peng, W., Yang, H., Liu, Z., Zhang, T., Jin, L., He, W., Du, M., Chen, Z.: Radiomics and radiogenomics in ovarian cancer: a review with a focus on ultrasound applications. Cancer Imaging 25(1), 130 (2025)

[17] Gao, Y., Zeng, S., Xu, X., Li, H., Yao, S., Song, K., Li, X., Chen, L., Tang, J., Xing, H., et al.: Deep learning-enabled pelvic ultrasound images for accurate diagnosis of ovarian cancer in china: a retrospective, multicentre, diagnostic study. Lancet Digit. Health 4(3), 179–187 (2022) https://doi.org/10.1016/ S2589-7500(21)00278-8

[18] Chen, H., Yang, B.W., Qian, L., Meng, Y.S., Bai, X.H., Hong, X.W., et al.: Deep learning prediction of ovarian malignancy at us compared with o-rads and expert assessment. Radiology 304(1), 106–113 (2022) https://doi.org/10.1148/ radiol.211367

[19] Hsu, S.T., Li, C.L., Hsu, J.D., Chen, S.N., Chen, C.F., Chan, T.S.: Automatic ovarian tumors recognition system based on ensemble convolutional neural network with ultrasound imaging. BMC Med. Inform. Decis. Mak. 22(1), 298 (2022) https://doi.org/10.1186/s12911-022-02047-6

[20] Jung, Y., Kim, T., Han, M.R., et al.: Ovarian tumor diagnosis using deep convolutional neural networks and a denoising convolutional autoencoder. Sci. Rep. 12, 17024 (2022) https://doi.org/10.1038/s41598-022-20653-2

[21] Xi, M., Zheng, R., Wang, M., Shi, X., Chen, C., Qian, J., Gu, X., Zhou, J.: Ultrasonographic diagnosis of ovarian tumors through the deep convolutional neural network. Ginekol. Pol. 95(3), 181–189 (2024) https://doi.org/10.5603/gpl.94956

[22] He, X., Bai, X.H., Chen, H., Feng, W.W.: Machine learning models in evaluating the malignancy risk of ovarian tumors: a comparative study. J. Ovarian Res. 17(1), 219 (2024) https://doi.org/10.1186/s13048-024-01544-8

[23] Zhao, Q., Lyu, S., Bai, W., Cai, L., Liu, B., Cheng, G., Wu, M., Sang, X., Yang, M., Chen, L.: MMOTU: A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation (2022). https: //doi.org/10.48550/arXiv.2207.06799

[24] Alsubai, S., Almadhor, A., Al Hejaili, A.: Integrative multi-stage deep learning framework for ovarian tumor ultrasound classification with explainability and confidence estimation. Front. Med. (Lausanne) 12, 1760167 (2026) https://doi. org/10.3389/fmed.2025.1760167

[25] Pham, T.L., Le, V.H.: Ovarian tumors detection and classification from ultrasound images based on yolov8. J. Adv. Inf. Technol. 15(2), 264–275 (2024) https://doi.org/10.12720/jait.15.2.264-275

[26] Borna, M.R., Saadat, H., Sepehri, M.M., Torkashvand, H., Torkashvand, L.,

Pilehvari, S.: Ai-powered diagnosis of ovarian conditions: insights from a newly introduced ultrasound dataset. Front. Physiol. 16, 1520898 (2025) https://doi. org/10.3389/fphys.2025.1520898