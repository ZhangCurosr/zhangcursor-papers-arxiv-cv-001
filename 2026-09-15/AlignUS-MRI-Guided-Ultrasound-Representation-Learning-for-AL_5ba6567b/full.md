# AlignUS: MRI-Guided Ultrasound Representation Learning for ALS Classification from Tongue Images

Khadijetou Abdel Ghader<sup>1</sup>, Emani Babe<sup>1</sup>, Lorenzo Pettinari<sup>2</sup>, Meya Haroune<sup>1</sup>, and Sidaty El Hadramy<sup>2</sup>

<sup>1</sup>Unité de Recherche en Systèmes Intelligents Avancés (URSIA), Institut Supérieur du

Numérique (SupNum), Nouakchott, Mauritanie

{21016, 21068, meya.haroune}@supnum.mr

<sup>2</sup>University of Basel, Basel, Switzerland

{lorenzo.pettinari, sidaty.elhadramy}@unibas.ch

Abstract. Amyotrophic lateral sclerosis (ALS) is a progressive neurodegenerative disease in which early assessment remains challenging, particularly in low-resource settings where MRI is often unavailable. Highresolution ultrasound (HRUS) of the tongue ofers a portable and low-cost alternative for evaluating bulbar involvement, but learning reliable diagnostic models is limited by small datasets and the dificulty of extracting robust representations from ultrasound alone. We propose AlignUS, a cross-modal knowledge distillation framework that transfers anatomical knowledge from MRI to a HRUS-based classifier while requiring only HRUS at inference time. The model combines classification loss, supervised contrastive learning, and feature-level distillation to align HRUS representations with MRI embeddings. AlignUS achieves a patient-level balanced accuracy of 0.958, macro-F1 of 0.963, and ROC-AUC of 0.990, aggregated across four patient-level cross-validation folds, with consistent improvements over HRUS baselines and cross-modal alternatives. These results demonstrate that MRI-derived supervision can substantially improve ultrasound-based ALS assessment while preserving low-cost, inference-time independence from MRI.

Keywords: Amyotrophic lateral sclerosis · Tongue ultrasound · Knowledge distillation · Supervised contrastive learning · Cross-modal learning

## 1 Introduction

Amyotrophic lateral sclerosis (ALS) is a progressive neurodegenerative disease characterized by the degeneration of both upper and lower motor neurons. As the disease progresses, patients gradually lose voluntary motor control, typically evolving from focal muscle weakness to widespread functional impairment, bul bar dysfunction, respiratory decline, and eventually respiratory failure [2,15,16]. Median survival after diagnosis is approximately two to five years [15,22], highlighting the importance of early and reliable diagnosis to enable timely clinical management, appropriate supportive care, and potential therapeutic interventions. Currently, ALS diagnosis remains primarily clinical and relies on a combination of neurological examination, electrophysiological assessment, and neuroimaging to exclude alternative conditions. Magnetic resonance imaging (MRI) is frequently used as part of the diagnostic workflow because it provides detailed structural information and enables the assessment of central nervous system abnormalities associated with motor neuron degeneration. However, ALS presents considerable diagnostic challenges due to its clinical heterogeneity, the overlap of early symptoms with other neurological disorders, and the absence of a single definitive biomarker [6,27]. Consequently, many patients experience prolonged diagnostic delays, often exceeding one year before reaching specialized care centers, with these delays particularly pronounced in low-resource healthcare settings where access to advanced imaging technologies such as MRI is limited by high acquisition costs and the need for specialized personnel [3].

The broader literature on AI-assisted neurodegenerative disease assessment ofers useful direction for addressing these limitations. Deep learning methods have achieved strong performance in MRI-based classification of motor neuron disease by learning complex patterns of neurodegeneration directly from imaging data [18,24,5], and related work has explored fusing structural MRI with difusion and functional imaging to capture complementary disease signatures [1]. Methods such as SF2Former [17], when trained on MRI, illustrate the capacity of these models to extract diagnostically relevant features from volumetric neuroimaging. Yet, these MRI-based AI methods still face MRI’s cost and infrastructure barriers, motivating the search for more accessible imaging alternatives.

High-resolution ultrasound (HRUS) of the tongue represents a promising candidate for this purpose. The tongue is particularly relevant because bulbar dysfunction is common in ALS and directly afects essential functions, including speech, swallowing, nutrition, and airway protection. Hensiek et al. [11] demonstrated that both 3T MRI and HRUS can capture structural changes in the tongue associated with bulbar impairment and disease progression, and related work has further shown that tongue ultrasound measurements can reveal structural alterations associated with bulbar involvement in ALS [23]. Compared with MRI, HRUS is portable, inexpensive, non-invasive, and suitable for repeated examinations, making it particularly attractive for monitoring disease progression and extending ALS assessment to settings where MRI access is limited. Despite these advantages, the use of tongue ultrasound for ALS assessment remains largely dependent on expert interpretation, limiting its scalability and reproducibility. AI-based methods could provide a means to support clinicians by enabling objective and consistent detection of disease-related ultrasound patterns. However, research on AI-based classification of ALS using tongue ultrasound remains limited, with existing studies constrained by the scarcity of annotated imaging datasets and the dificulty of collecting suficient patient data for robust model development.

Beyond dataset scarcity, developing reliable AI models from HRUS is further complicated by operator-dependent acquisition variability and class imbalance between patients and controls, limiting models’ ability to learn robust, generalizable representations and widening the gap between HRUS’s clinical potential and its automated use. To address these challenges, we propose AlignUS, a cross-modal knowledge distillation framework [12,8] that uses MRI-derived representations during training to improve ALS classification from tongue ultrasound, requiring only HRUS at inference. Building on prior cross-modal distillation work in medical imaging [9], AlignUS transfers complementary structural knowledge from MRI without depending on it at deployment, preserving the portability and low cost of ultrasound. The main contributions of this work are:

To the best of our knowledge, we introduce the first framework to predict ALS from tongue HRUS images using cross-modal knowledge distillation from MRI, without requiring MRI during inference.

– We design a framework adapted to low- and middle-income settings, relying solely on accessible, low-cost ultrasound imaging at deployment while benefiting from MRI-derived knowledge during training.

– We conduct experiments on the dataset proposed by [11], demonstrating state-of-the-art results for ALS assessment from tongue ultrasound images.

## 2 Method

We propose AlignUS, a teacher–student framework that leverages anatomical representations learned from MRI to improve HRUS-based diagnosis without requiring MRI at inference time. The central idea behind AlignUS is to use MRI as a modality during training: an MRI encoder first learns a robust representation of tongue anatomy and disease-related characteristics, which is subsequently transferred to an HRUS encoder through feature and prediction alignment. After training, only the HRUS encoder is retained, allowing the model to operate using ultrasound images alone. An overview of the proposed AlignUS framework is presented in Figure 1.

As shown in Figure 1, the framework consists of two stages. First, an MRI encoder is trained on labeled MRI data to learn a discriminative anatomical representation using classification and supervised contrastive objectives. The trained MRI encoder is then frozen and used as a teacher to guide the training of an HRUS student encoder through knowledge distillation. The student learns to approximate the MRI-informed feature space while relying only on HRUS images, enabling MRI-free inference. Figure 2 provides a detailed overview of each stage of AlignUS, and the following sections describe its components.

## 2.1 MRI Encoder

The first component of AlignUS is the MRI encoder, which provides the anatomical reference representation used to guide HRUS learning. We chose ResNet-18 as the backbone architecture for both MRI and HRUS encoders. This architecture provides a good balance between representation capacity and robustness for the available dataset size, while residual connections improve optimization stability during contrastive representation learning. For MRI input, the first convolutional layer was adapted to accept single-channel grayscale images, while the remaining residual blocks were unchanged. For each MRI volume, we selected the single 2D sagittal slice with the highest image variance, under the assumption that greater spatial variation corresponds to better visualization of the tongue structure. The selected MRI slice was intensity-normalized using percentile clipping followed by z-score standardization and was square-padded to preserve the original aspect ratio during training. The MRI encoder produces a 512-dimensional embedding via global average pooling. This embedding is then given as input to a linear classification head with dropout [25] for ALS/Control prediction and an MLP projection head that feeds a supervised contrastive objective. The contrastive formulation, following Khosla et al. [14] and the SimCLR framework [4], pulls embeddings from the same diagnostic class together while pushing those from opposing classes apart. Therefore, the MRI encoder is optimized using a combination of weighted cross-entropy and supervised contrastive loss: $\mathcal { L } _ { \mathrm { M R I } } = \lambda _ { \mathrm { C E } } \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { S u p C o n } } \mathcal { L } _ { \mathrm { S u p C o n } } .$

![](images/9bae85529c7b49622bbd66cfc594b16a293b52d1126b5e87b0d4668555b87b70.jpg)  
Fig. 1. Overview of the proposed AlignUS framework. It follows a teacher–student design where an MRI encoder is first trained to learn representations from MRI data. The learned MRI encoder is then frozen and used to guide the training of an HRUS encoder through supervised classification, supervised contrastive learning, and knowledge distillation. During inference, only the HRUS encoder and classifier are used.

## 2.2 HRUS Encoder

The second component of AlignUS is the HRUS student encoder, which learns to generate representations aligned with the MRI teacher while using ultrasound images as input. During this stage, the MRI encoder and classification head remain frozen, ensuring that the anatomical representation learned from MRI defines a stable target feature space. The HRUS encoder follows the same architecture as the MRI branch, using a single-channel ResNet-18 backbone with a 512-dimensional output after global average pooling. The architectural symmetry between the two modalities enables direct comparison between MRI and HRUS embeddings and allows the student representation to be aligned with the teacher without additional projection networks. To transfer the anatomical knowledge learned from MRI to HRUS, the student encoder is optimized using a multi-objective training strategy that combines supervised classification, supervised contrastive learning, and knowledge distillation. The overall HRUS training objective is defined as: ${ \mathcal { L } } _ { \mathrm { H R U S } } = \lambda _ { \mathrm { C E } } { \mathcal { L } } _ { \mathrm { C E } } + \lambda _ { \mathrm { S u p C o n } } { \mathcal { L } } _ { \mathrm { S u p C o n } } + \lambda _ { \mathrm { K L } } { \mathcal { L } } _ { \mathrm { K I } }$

![](images/7a8dbe70d87779be6b3b6021f81e1993df62820d97ba08ff5ddbadc2db624a2e.jpg)  
Fig. 2. Detailed architecture of the MRI and HRUS encoders, along with the loss functions used to train the AlignUS framework.

The cross-entropy term provides direct supervision for ALS/Control classification. However, relying solely on classification loss may result in poorly structured representations, particularly when training with limited and imbalanced datasets. Therefore, supervised contrastive learning is incorporated to regularize the embedding space by reducing intra-class variation and increasing separation between diagnostic categories. The knowledge distillation term enables the transfer of MRI-derived information to the HRUS encoder. Specifically, the KL-divergence loss [12,21,20] encourages the student’s predictions to match the soft output distribution of the frozen MRI teacher. This allows the student to benefit from the teacher’s learned confidence patterns and inter-class relationships, which are not captured by hard diagnostic labels alone. Through this strategy, AlignUS follows the principles of cross-modal distillation [9] and multimodal representation learning [1], where a modality available only during training guides the learning of a representation suitable for deployment in a single-modality setting.

## 3 Results

## 3.1 Dataset and implementation details

The dataset consisted of publicly available tongue imaging data from Schreiber et al. [23]. After excluding incomplete cases, the final cohort comprised 62 subjects (48 ALS, 14 Controls) and 513 HRUS tongue images. MRI data were used exclusively for training the teacher model, while HRUS images were used for the student model. All images underwent standard preprocessing, including resizing and intensity normalization, and were split at the patient level prior to frame-level sampling into training, validation, and test sets to avoid data leakage. To address class imbalance, the training set was balanced via oversampling of control subjects using augmented samples, with geometric and photometric augmentations applied only during training. All experiments were implemented in PyTorch and trained on an NVIDIA A100 GPU. The MRI encoder was optimized using AdamW with a learning rate of $3 \times 1 0 ^ { - 4 }$ , weight decay of $1 \times 1 0 ^ { - 4 }$ , batch size of 8, cosine annealing for up to 50 epochs, and early stopping based on validation balanced accuracy. The HRUS encoder used the same optimizer settings, with loss weights set to $\lambda _ { \mathrm { C E } } = 1 . 0 , \lambda _ { \mathrm { S u p C o n } } = 0 . 2$ and $\lambda _ { \mathrm { K L } } = 0 . 1 0$ , selected via validation grid search. Model selection and evaluation were performed using four-fold patient-level cross-validation, ensuring that all images from a given subject were assigned exclusively to either the training, validation, or test fold and never shared across splits. In each fold, approximately 75% of subjects were used for training, with the remaining subjects evenly partitioned into validation and held-out test sets; all the results reported in this work correspond to the aggregated performance across all four folds on the respective held-out test partitions.

## 3.2 Quantitative results

Table 1 shows that, at the class level, AlignUS achieves perfect recall for ALS (1.00), indicating that all ALS patients in the test cohort are correctly identified, alongside strong precision (0.94) and F1-score (0.97). Performance on the control class is also strong, with precision of 1.00, recall of 0.92, and F1-score of 0.95.

Table 1. Patient-level classification report on the held-out test set.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>F1-score</td><td>Nb of patients</td></tr><tr><td>Control</td><td>1.00</td><td>0.92</td><td>0.95</td><td>6 (46 Images)</td></tr><tr><td>ALS</td><td>0.94</td><td>1.00</td><td>0.97</td><td>8 (79 Images)</td></tr></table>

Table 2 presents comparison between MRI-based models, HRUS-only baselines, cross-modal learning approaches, and the proposed AlignUS on the same test set, with a column indicating whether each method is deployable using HRUS-only at inference time, which corresponds to the clinically relevant lowresource setting. MRI-based models, including ResNet-18, DenseNet-121 [13], EficientNet-B0 [26], and ViT-B/16 [7], are marked as not HRUS-deployable (×) since they require MRI inputs at inference, and while they achieve generally strong performance, they serve primarily as upper-bound references. In contrast, HRUS-only baselines are fully deployable (✓) but exhibit limited discriminative performance, with balanced accuracy ranging from 0.625 to 0.771 depending on architecture, highlighting the dificulty of learning directly from ultrasound data under limited supervision. Cross-modal methods, including vanilla knowledge distillation [12] and Fitnets [21], also retain HRUS-only inference capability and significantly improve performance (balanced accuracy 0.839), demonstrating that

Table 2. Comparison of MRI-based models, HRUS-only baselines, cross-modal methods, and AlignUS. The “Low-resource” column indicates HRUS-only deployability at inference time. The inference time of each model is also reported in ms.
<table><tr><td>Modality</td><td>Method</td><td>Low-resource Bal. Acc.</td><td></td><td>F1</td><td></td><td>AUC Time (ms)</td></tr><tr><td rowspan="4">MRI</td><td>ResNet-18 [10]</td><td>X</td><td>0.973</td><td>0.965</td><td>0.993</td><td>0.82</td></tr><tr><td>DenseNet-121 [13]</td><td>X</td><td>0.905</td><td>0.898</td><td>0.934</td><td>2.35</td></tr><tr><td>EfficientNet-B0 [26]</td><td>X</td><td>0.934</td><td>0.926</td><td>0.928</td><td>1.55</td></tr><tr><td>ViT-B/16 [7]</td><td>X</td><td>0.748</td><td>0.732</td><td>0.821</td><td>4.20</td></tr><tr><td rowspan="4">HRUS</td><td>ResNet-18 [10]</td><td>V</td><td>0.625</td><td>0.576</td><td>0.776</td><td>0.82</td></tr><tr><td>DenseNet-121 [13]</td><td>V</td><td>0.771</td><td>0.757</td><td>0.865</td><td>2.35</td></tr><tr><td>EfficientNet-B0 [26]</td><td>√</td><td>0.677</td><td>0.644</td><td>0.812</td><td>1.55</td></tr><tr><td>ViT-B/16 [7]</td><td>√</td><td>0.724</td><td>0.699</td><td>0.849</td><td>4.20</td></tr><tr><td rowspan="3">Cross-modal Fitnets [21]</td><td>Vanilla KD [12]</td><td>√</td><td>0.839</td><td>0.842</td><td>0.964</td><td>0.82</td></tr><tr><td></td><td>V</td><td>0.839</td><td></td><td>0.8400.960</td><td>0.82</td></tr><tr><td>ALIGNUS (Ours)</td><td>√</td><td>0.958</td><td></td><td>0.963 0.990</td><td>0.82</td></tr></table>

MRI-guided supervision provides meaningful representational benefits even when MRI is not available at test time. AlignUS, which is also HRUS-deployable (✓), achieves the best overall performance among all HRUS-only inference methods, with a balanced accuracy of 0.958, macro-F1 of 0.963, and ROC-AUC of 0.990. This result shows that AlignUS not only surpasses all direct HRUS baselines and existing cross-modal distillation techniques, but also closes a significant portion of the performance gap toward MRI-based models. These findings indicate that MRI-derived features can be transferred into HRUS representations during training, making AlignUS particularly suitable for deployment in low-resource clinical environments where only ultrasound imaging is available at inference time. Notably, AlignUS achieves this performance gain without incurring any additional inference cost relative to a standalone HRUS classifier, since the MRI teacher is discarded after training and the student retains the same ResNet-18 architecture, yielding an inference time of 0.82 ms comparable to the fastest HRUS baseline and substantially lower than ViT-B/16 (4.20 ms), underscoring AlignUS’s suitability for real-time clinical use.

## 3.3 Embedding Space Analysis

To further analyze the embedding space learned from HRUS representations, we performed a principal component analysis (PCA) projection [19] of the extracted embeddings from the held-out test set, as shown in Figure 3. Although some overlap between classes remains at the individual image level, the patientlevel aggregated embeddings exhibit a clearer separation between ALS and Control groups, indicating that aggregating multiple HRUS frames enhances representation stability and improves class-level discrimination.

![](images/ec591e8603146a512abe095529d22bb064ed1d1daef25b569138b3de25efe72e.jpg)  
Fig. 3. PCA visualization of HRUS embeddings on the held-out test set. Left: imagelevel embeddings; right: patient-level embeddings. red: ALS, blue: Control.

Table 3. Patient-level ablation study on the held-out test set.
<table><tr><td>Loss function</td><td>Bal. Acc.</td><td>F1</td><td>AUC</td></tr><tr><td>CE</td><td>0.792</td><td>0.831</td><td>0.875</td></tr><tr><td> $\mathrm { C E } + \mathrm { K L }$ </td><td>0.833</td><td>0.862</td><td>0.917</td></tr><tr><td> $\mathrm { C E } + \mathrm { S u p C o n }$ </td><td>0.875</td><td>0.899</td><td>0.938</td></tr><tr><td>CE + SupCon + KL (ALIGNUS)</td><td>0.958</td><td>0.963</td><td>0.990</td></tr></table>

## 3.4 Ablation Study

To evaluate the contribution of each training objective, we progressively removed components from the full AlignUS loss while keeping the preprocessing pipeline, patient-level split, and class-balancing strategy fixed. Results on the held-out test set are reported in Table 3. We show that performance improves consistently as additional training objectives are introduced.

## 4 Limitations

While AlignUS achieves strong performance for ALS classification from tongue HRUS, the main limitation of this study is the limited size of the available dataset. This constraint arises from the scarcity of publicly available datasets containing both HRUS and MRI acquisitions, which are required for cross-modal knowledge transfer. To mitigate this and reduce the risk of overestimating model performance, we performed four-fold cross-validation using strict patient-level splits, ensuring that images from the same subject were never shared between training and evaluation sets. Nevertheless, the current results represent validation within a single cohort, and further evaluation on larger, multi-center datasets with greater demographic and acquisition variability is required to assess the generalizability and robustness of AlignUS across diferent clinical environments.

## 5 Conclusion

In this work, we introduced AlignUS, a cross-modal knowledge distillation framework for ALS assessment from tongue HRUS, which leverages MRI-derived anatomical knowledge during training while requiring only ultrasound images at inference time. By transferring structural information from MRI into a portable ultrasound-based classifier, AlignUS enables accurate and accessible disease assessment while preserving the practical advantages of HRUS, including low cost, portability, and suitability for repeated examinations. The proposed framework achieves state-of-the-art performance for ALS classification from tongue ultrasound, demonstrating the potential of cross-modal learning to overcome the limitations of low-resource imaging modalities. Future work will focus on collecting larger and more diverse paired HRUS-MRI datasets and performing external validation across multiple clinical sites to further establish the reliability and clinical applicability of AlignUS.

## References

1. Akkus, C., Chu, L., Djakovic, V., Jauch-Walser, S., Koch, P., Loss, G., Marquardt, C., Moldovan, M., Sauter, N., Schneider, M., Schulte, R., Urbanczyk, K., Goschenhofer, J., Heumann, C., Hvingelby, R., Schalk, D., Aßenmacher, M.: Multimodal deep learning (2023)

2. Brown, R.H., Al-Chalabi, A.: Amyotrophic lateral sclerosis. N. Engl. J. Med. 377(2), 162–172 (Jul 2017)

3. Falcão de Campos, C., Gromicho, M., Uysal, H., Grosskreutz, J., Kuzma-Kozakiewicz, M., Oliveira Santos, M., Pinto, S., Petri, S., Swash, M., de Carvalho, M.: Trends in the diagnostic delay and pathway for amyotrophic lateral sclerosis patients across diferent countries. Front. Neurol. 13, 1064619 (2022)

4. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: Proceedings of the 37th International Conference on Machine Learning (ICML). pp. 1597–1607 (2020)

5. Cheplygina, V., de Bruijne, M., Pluim, J.P.W.: Not-so-supervised: a survey of semi-supervised, multi-instance, and transfer learning in medical image analysis. Medical Image Analysis 54, 280–296 (2019)

6. Chiò, A., Logroscino, G., Traynor, B.J., Collins, J., Simeone, J.C., Goldstein, L.A., White, L.A.: Global epidemiology of amyotrophic lateral sclerosis: a systematic review of the published literature. Neuroepidemiology 41(2), 118–130 (Jul 2013)

7. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.: An image is worth 16x16 words: transformers for image recognition at scale. In: Proceedings of the International Conference on Learning Representations (ICLR) (2021)

8. Gou, J., Yu, B., Maybank, S.J., Tao, D.: Knowledge distillation: a survey. International Journal of Computer Vision 129, 1789–1819 (2021)

9. Gupta, S., Hofman, J., Malik, J.: Cross modal distillation for supervision transfer. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2827–2836 (2016)

10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016)

11. Hensiek, N., Schreiber, F., Wimmer, T., Kaufmann, J., Machts, J., Fahlbusch, L., Garz, C., Vogt, S., Prudlo, J., Dengler, R., Petri, S., Nestor, P.J., Vielhaber, S., Schreiber, S.: Sonographic and 3T-MRI-based evaluation of the tongue in ALS. NeuroImage Clin. 26(102233), 102233 (Mar 2020)

12. Hinton, G., Vinyals, O., Dean, J.: Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 (2015)

13. Huang, G., Liu, Z., van der Maaten, L., Weinberger, K.Q.: Densely connected convolutional networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4700–4708 (2017)

14. Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D.: Supervised contrastive learning. In: Advances in Neural Information Processing Systems. vol. 33, pp. 18661–18673 (2020)

15. Kiernan, M.C., Vucic, S., Cheah, B.C., Turner, M.R., Eisen, A., Hardiman, O., Burrell, J.R., Zoing, M.C.: Amyotrophic lateral sclerosis. Lancet 377(9769), 942–955 (Mar 2011)

16. Kurosawa, K.: Amyotrophic lateral sclerosis. Ryoikibetsu Shokogun Shirizu 3(33), 178–180 (2001)

17. Kushol, R., Luk, C.C., Dey, A., Benatar, M., Briemberg, H., Dionne, A., Dupré, N., Frayne, R., Genge, A., Gibson, S., Graham, S.J., Korngut, L., Seres, P., Welsh, R.C., Wilman, A.H., Zinman, L., Kalra, S., Yang, Y.H.: SF2Former: Amyotrophic lateral sclerosis identification from multi-center MRI data using spatial and frequency fusion transformer. Comput. Med. Imaging Graph. 108(102279), 102279 (Sep 2023)

18. Litjens, G., Kooi, T., Bejnordi, B.E., et al.: A survey on deep learning in medical image analysis. Medical Image Analysis 42, 60–88 (2017)

19. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of Machine Learning Research 9, 2579–2605 (2008)

20. Park, W., Kim, D., Lu, Y., Cho, M.: Relational knowledge distillation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 3967–3976 (2019)

21. Romero, A., Ballas, N., Kahou, S.E., Chassang, A., Gatta, C., Bengio, Y.: Fitnets: hints for thin deep nets. In: Proceedings of the International Conference on Learning Representations (ICLR) (2015)

22. Sabatelli, M., Madia, F., Conte, A., Luigetti, M., Zollino, M., Mancuso, I., Lo Monaco, M., Lippi, G., Tonali, P.: Natural history of young-adult amyotrophic lateral sclerosis. Neurology 71(12), 876–881 (Sep 2008)

23. Schreiber, S., Hensiek, N., Wimmer, T., Kaufmann, J., Machts, J., Fahlbusch, L., Garz, C., Vogt, S., Prudlo, J., Dengler, R., Petri, S., Nestor, P.J., Vielhaber, S.: Sonographic and 3t-mri-based evaluation of the tongue in als (2020). https: //doi.org/10.17632/8d939wvc7r.1, mendeley Data, Version 1

24. Shen, D., Wu, G., Suk, H.I.: Deep learning in medical image analysis. Annual Review of Biomedical Engineering 19, 221–248 (2017)

25. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.: Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research 15, 1929–1958 (2014)

26. Tan, M., Le, Q.V.: Eficientnet: rethinking model scaling for convolutional neural networks. In: Proceedings of the 36th International Conference on Machine Learning (ICML). pp. 6105–6114 (2019)

27. Turner, M.R., Kiernan, M.C., Leigh, P.N., Talbot, K.: Biomarkers in amyotrophic lateral sclerosis. Lancet Neurol. 8(1), 94–109 (Jan 2009)