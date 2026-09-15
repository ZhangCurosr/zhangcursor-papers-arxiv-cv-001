# Deep Learning-based Intelligent Diagnosis of Congenital Uterine Anomalies in 3D Ultrasound

Yueyue Xu<sup>a,1</sup>, Yuhao Huang<sup>b,c,1</sup>, Jiaxiao Deng<sup>b</sup>, Yuanji Zhang<sup>b,d</sup>, Haoming Zhang<sup>b</sup>, Jiajia Qu<sup>a</sup>, Shiying Zheng<sup>a</sup>, Xiaomei Tang<sup>a</sup>, Haining Chen<sup>a</sup>, Chengcai Chen<sup>e</sup>, Yiyi Wu<sup>f</sup>, Xin Yang<sup>g</sup>, Dong Ni<sup>b,h,i</sup>, Hongyu Zheng<sup>a,∗</sup>

<sup>a</sup>The People’s Hospital of Guangxi Zhuang Autonomous   
Region, Nanning, Guangxi, China   
<sup>b</sup>Medical Ultrasound Image Computing (MUSIC) Lab, Shenzhen   
University, Shenzhen, Guangdong, China   
<sup>c</sup>Centre for Artificial Intelligence and Robotics, Hong Kong Institute of Science &   
Innovation, Chinese Academy of Sciences, Hong Kong, China   
<sup>d</sup>Shenzhen Luohu People’s Hospital (The Third Afiliated Hospital of Shenzhen   
University), Shenzhen, Guangdong, China   
<sup>e</sup>Afiliated Hospital of Youjiang Medical College for   
Nationalities, Youjiang, Guangxi, China   
<sup>f</sup>Guilin Maternal and Child Health Hospital, Guilin, Guangxi, China   
<sup>g</sup>School of Biomedical Engineering, Medical School, Shenzhen   
University, Shenzhen, Guangdong, China   
<sup>h</sup>School of Artificial Intelligence, Shenzhen University, Shenzhen, Guangdong, China   
<sup>i</sup>School of Biomedical Engineering and Informatics, Nanjing Medical   
University, Nanjing, Jiangsu, China

## Abstract

Objective: To develop an intelligent framework, termed CUA-Net, for the automated classification of congenital uterine anomalies (CUA) without requiring coronal plane reconstruction, and to evaluate its clinical applicability. Methods: CUA-Net was built on 3D ResNet-18, equipped with a dynamic data resampling strategy to mitigate the data imbalance issue and a hard sample mining technique to fully learn from the dificult cases by loss adjustment. We further proposed the self-supervised reconstruction to comprehensively explore the volumes and the online data augmentation to refine

the wrong predictions and enhance the model’s generalization. We compared the CUA-Net with diferent deep-learning methods and junior/senior sonographers in the testing set. The evaluation metrics included accuracy, precision, recall, F1-score, micro-AUC, and macro-AUC.

Results: The proposed CUA-Net exhibited satisfactory performance in both internal and external test sets. In the internal cohort, the model achieved accuracy of 93.88%, precision of 87.01%, recall of 95.92%, F1-score of 88.09%, and micro-AUC of 0.9982 and macro-AUC of 0.9997. In the external set, it maintained good performance with accuracy of 91.52%, precision of 83.27%, recall of 88.63%, F1-score of 81.49%, micro-AUC of 0.9945 and macro-AUC of 0.9990. Our CUA-Net outperformed the junior sonographers across all performance indicators and achieved performance comparable to that of the senior sonographers across most metrics.

Conclusion: The CUA-Net demonstrates favorable accuracy and generalizability in classifying common CUA categories, while showing preliminary potential for recognizing less prevalent anomalies. These capabilities may help optimize clinical workflows and support more standardized diagnosis.

Keywords: Congenital Uterine Anomalies, 3D Ultrasound, Deep Learning, Classification, Diagnostic Workflow

## 1. Introduction

Congenital Uterine Anomalies (CUA) refer to anatomical abnormalities caused by incomplete fusion or resorption of the paramesonephric (Müllerian) ducts during embryonic development [1]. CUA is one of the leading causes of female infertility, recurrent miscarriage, intrauterine growth restriction, preterm birth, and retained placenta. Studies have shown that its overall prevalence is estimated to be 5.5–6.7% in the general population, 7.3–8.0% among infertile women, and as high as 16.7% in those with recurrent spontaneous abortions [2, 3]. There are various types of CUAs, including septate uterus, arcuate uterus, uterus didelphys, unicornuate uterus, T-shaped uterus, and bicornuate uterus [4]. In clinical practice, the prevalence of different CUA subtypes is highly imbalanced. According to a systematic review by Chan et al., in unselected populations, the reported prevalence is 3.9% for arcuate uterus, 2.3% for septate uterus, and 0.4% for bicornuate uterus [5]. For T-shaped uterus, although reported prevalence varies because of inconsistent diagnostic criteria, a recent prospective study using the CUME criteria reported a prevalence of only 0.8% among fertile women [6, 7]. Since diferent types of CUA require distinct clinical interventions, and some forms can significantly improve fertility and pregnancy outcomes after surgical correction, accurate diagnosis and classification of uterine anomalies are of great importance for guiding clinical treatment decisions [8].

Ultrasound (US) serves as a primary auxiliary tool in the diagnosis and classification of CUA due to its distinct advantages over other modalities, e.g., hysterosalpingography, magnetic resonance imaging, hysteroscopy, and laparoscopy [9]. It is a non-invasive, radiation-free technique that ofers realtime imaging, making it particularly suitable for dynamic assessment and routine use in obstetrics and gynecology [10, 11]. US also allows observation of the endometrial cavity and the outer contour of the uterus through transverse and longitudinal sections. However, conventional 2D scanning cannot capture the coronal plane, which limits its ability to distinguish between diferent types of uterine anomalies. It highly relies on sonographers’ experience to output the subjective results based on longitudinal and transverse scanning (Fig. 1 (a)). Currently, a definitive diagnosis often requires further invasive procedures such as hysteroscopy or laparoscopy.

In contrast, 3D US enables visualization of the uterine coronal plane, providing a comprehensive view of the uterine structure and its spatial relationships, thereby supporting accurate diagnosis and classification of CUAs. However, 3D acquisitions of uterine coronal planes rely on high-quality images of the midsagittal plane, followed by a coronal sweep that allows for a good reconstruction of the uterine volume. Sonographers must be highly skilled with years of clinical experience to perform this technique. Accurate classification of uterine anomalies also typically involves detailed measurements on the obtained coronal plane (Fig. 1 (b)). Furthermore, constrained by the known disparities in incidence rates among subtypes, the diagnosis and identification of certain rare congenital uterine anomalies remain particularly challenging. These challenges may increase the complexity and time required for diagnosis, often leading to low consistency and accuracy, especially in complex anomaly cases.

In recent years, artificial intelligence and deep learning (DL) have made significant advances in the field of medical image analysis [12, 13, 14, 15], especially for intelligent US [16, 17, 18, 19, 20]. Researchers studied the automatic methods for uterine fibroid detection [21] and classification [22] in US images. Boneš et al. developed an automated system for the segmentation and alignment of uterine shapes from 3D US [23]. Yang et al. first proposed reinforcement learning-based methods to locate the uterine coronal plane in 3D US [24, 25]. Following the above framework, Zou et al. further enhanced the model flexibility and accuracy in detecting uterine planes [26]. The above methods are capable of iteratively extracting the uterine coronal plane and hold potential for enabling CUA diagnosis in subsequent steps [27]. However, accurate CUA analysis relies on measuring multiple parameters on the coronal plane, which requires additional annotations such as keypoints, measurement lines, or segmentations. Furthermore, accumulated errors of plane localization (>10<sup>◦</sup> spatial angular error in [26]) can significantly compromise the accuracy of downstream CUA assessment.

![](images/ea5843df4451df622271e27adec6bc881210647fd787fb0658090d5bbf0d3228.jpg)  
Figure 1: Comparison of diferent workflows for CUA classification.

Most recently, Dou et al. [28] introduced a difusion-based approach to handle the plane localization task in 3D uterine volumes. They also explored an uncertainty-driven strategy for binary classification of normal versus CUA cases, achieving an accuracy of 90.67% (Fig. 1 (c)). However, their approaches still rely on dynamic modeling of the plane localization process. At present, there is a lack of intelligent methods that directly analyze 3D data to achieve fine-grained and multi-class CUA classification.

In this work, we proposed a deep learning-based framework for automatic CUA classification in 3D US, named CUA-Net. For 3D uterine volumes, it can automatically distinguish between the normal class and six distinct CUA categories. Our contributions is as follows: First, we introduced a dynamic data resampling (DRS) strategy that adjusts sampling weights at both the class and image levels, thereby mitigating the data imbalance caused by disparities in the incidence of uterine anomalies. Second, we proposed hard sample mining (HSM) strategies to adjust the loss and make the model focus on learning hard cases. Third, we adopted a self-supervised reconstruction (SSR) method for model pretraining. This can better explore the limited uterine volumes and enhance the feature modeling ability of CUA-Net. Last, we developed an online data augmentation (ODA) strategy to refine the prediction during testing, further improving the model’s generalization ability. Our proposed CUA-Net was validated on a large uterine US dataset, demonstrating strong performance in CUA classification and the potential to transform the traditional clinical workflow (Fig. 1).

## 2. Materials and Method

## 2.1. Data source

Approved by the local IRB (No. KY-KJT-2023-1), we first collected 701 uterus volumes from a medical center between December 2022 and December 2024. All volumes were acquired using a Mindray Resona-9 US system with an intracavitary 3D probe.

Inclusion criteria contain: (1) Females aged 20-50 years; (2) Underwent transvaginal 3D US with acceptable image quality; (3) Diagnosed with normal uterine morphology or CUAs during hysteroscopy or laparoscopy; (4) Provided informed consent. Exclusion criteria include: (1) Presence of other reproductive system organic diseases (n=12); (2) Pregnancy/lactation (n=7); (3) Use of an intrauterine device (n=21); (4) Lack of clinical gold-standard hysteroscopy/laparoscopy supports (n=9).

After strict data inclusion and exclusion, our dataset for model development included 652 3D US volumes from 444 patients. The dataset comprised 336 normal uterus, 95 septate uterus, 139 arcuate uterus, 49 bicornuate uterus, 17 uterus didelphys, 11 T-shaped uterus, and 5 unicornuate uterus. The average volume size is 367×189×334, with an isotropic voxel spacing of 0.4mm. Specifically, we randomly split the dataset at patient level into a nearly 6:1:3 ratio for training (388), validation (68), and testing (196).

To enhance the generalizability and robustness of the model, we further incorporated a prospectively collected external test cohort from three hospitals. All data were acquired using GE Voluson E8 US systems between January 2025 and June 2026 and were temporally independent of the model development cohort (before December 2024). The detailed distribution of uterine types was as follows: 177 normal uterus, 108 septate uterus, 90 arcuate uterus, 36 unicornuate uterus, 31 uterus didelphys, 1 T-shaped uterus, and 5 bicornuate uterus. Dataset details can be found in Table 1.

![](images/afc468ecad856470241a23a346fcd76cb086e099c9873b67c747e80195a63c47.jpg)

Figure 2: Framework of our proposed CUA-Net.  
Table 1: Dataset details.
<table><tr><td></td><td>Train</td><td>Val</td><td>Internal Test</td><td>External Test</td></tr><tr><td>Normal uterus</td><td>201</td><td>34</td><td>101</td><td>177</td></tr><tr><td>Septate uterus</td><td>56</td><td>10</td><td>29</td><td>108</td></tr><tr><td>Arcuate uterus</td><td>83</td><td>14</td><td>42</td><td>90</td></tr><tr><td>Unicornuate uterus</td><td>29</td><td>5</td><td>15</td><td>36</td></tr><tr><td>Uterus didelphys</td><td>10</td><td>2</td><td>5</td><td>31</td></tr><tr><td>T-shaped uterus</td><td>6</td><td>2</td><td>3</td><td>1</td></tr><tr><td>Bicornuate uterus</td><td>3</td><td>1</td><td>1</td><td>5</td></tr></table>

## 2.2. Data pre-processing

All volumes were proportionally resized to 128×96×128 and padded with zeros. Online data augmentation during training includes: 1) random center scaling by 0.9–1.1×, 2) random rotation of $0 { - } 1 0 ^ { \circ }$ along the XYZ axes, 3) random translation of 10 voxels along the XYZ axes, and 4) random flipping along all axes. Additionally, all data is normalized by dividing by 255.

## 2.3. Model development

In this study, as shown in Fig. 2, we developed a deep learning-based framework (named CUA-Net) to extract the features from 3D US and directly output the CUA classification results. We used the common 3D ResNet-18 as the model backbone, with a fully connected layer and softmax function, to map the extracted features to the classification probability outputs. Crossentropy (CE) was used for training:

$$
\mathcal { L } _ { \mathrm { C E } } = - \sum _ { i = 1 } ^ { C } y _ { i } \log ( \hat { y } _ { i } ) ,\tag{1}
$$

where C is the number of classes, $y _ { i }$ is the one-hot GT label and $\hat { y } _ { i }$ is the predicted probability for i class.

Basic Dynamic Data Re-Sampling Strategy. Data resampling is an important strategy to address data imbalance. In our task, we first define the class-level base sampling weights inversely proportional to the original sample size of each class (Equ. 2), i.e., $1 / n _ { y _ { i } }$

$$
w _ { \mathrm { c l s } } ( y _ { i } ) = \frac { 1 / n _ { y _ { i } } } { \sum _ { k = 1 } ^ { C } ( 1 / n _ { y _ { i } } ) } .\tag{2}
$$

We subsequently introduce a dynamic technique to update the image-level probabilities (Equ. 3), driven by the loss function of individual case:

$$
w _ { \mathrm { h a r d } } ( x _ { i } ) = \frac { \log ( f ( x _ { i } ) , y _ { i } ) } { \sum _ { j = 1 } ^ { N } \log ( f ( x _ { j } ) , y _ { j } ) } .\tag{3}
$$

This sampling probability is dynamically adjusted every three iterations, ensuring that samples with higher loss are assigned greater sampling weights. Finally, the class- and image-level strategies are combined to form the final sampling probability (Equ. 4):

$$
P ( x _ { i } ) = \frac { w _ { \mathrm { c l s } } ( y _ { i } ) \cdot ( 1 + \beta \cdot w _ { \mathrm { h a r d } } ( x _ { i } ) ) } { \sum _ { j = 1 } ^ { N } ( w _ { \mathrm { c l s } } ( y _ { i } ) \cdot ( 1 + \beta \cdot w _ { \mathrm { h a r d } } ( x _ { i } ) ) ) } ,\tag{4}
$$

where $\beta$ is a hyperparameter to control the adjustment intensity.

Hard Sample Mining for Loss Adjustment. To fully mine the hard samples and improve the model robustness, we propose a novel hard sample mining strategy. Simply, sample dificulty can be ranked based on the loss function. However, directly selecting the highest-loss samples may introduce outliers (e.g., noisy data). Therefore, we employ a quantile filtering strategy, retaining high-loss samples while excluding extreme outliers.

$$
{ \mathcal { L } } = \{ { \mathcal { L } } _ { \mathrm { C E } } ( x _ { 1 } ) , { \mathcal { L } } _ { \mathrm { C E } } ( x _ { 2 } ) , . . . , { \mathcal { L } } _ { \mathrm { C E } } ( x _ { N } ) \} ,\tag{5}
$$

$$
Q _ { \tau } = \mathrm { Q u a n t i l e } ( \boldsymbol { \mathcal { L } } , \tau ) , \quad Q _ { \tau ^ { \prime } } = \mathrm { Q u a n t i l e } ( \boldsymbol { \mathcal { L } } , \tau ^ { \prime } ) ,\tag{6}
$$

$$
\mathcal { H } = \left\{ x _ { h } \vert Q _ { \tau } < \mathcal { L } _ { \mathrm { C E } } ( x _ { h } ) < Q _ { \tau ^ { \prime } } \right\} ,\tag{7}
$$

where $\tau$ and $\tau ^ { \prime }$ represents the 80% and 95% percentiles, representative. H denotes the hard sample set. Then, with the update loss weights (Equ. 8),the hard sample driven focal loss can be written as Equ. 9.

$$
w _ { \mathrm { h a r d } } ( x _ { i } ) = \left\{ \begin{array} { l l } { 1 + \beta , } & { x _ { i } \in \mathcal { H } } \\ { 1 , } & { x _ { i } \notin \mathcal { H } } \end{array} \right. ,\tag{8}
$$

$$
\mathcal { L } _ { \mathrm { h a r d - f o c a l } } = - w _ { \mathrm { h a r d } } ( x _ { i } ) ( 1 - p _ { t } ) ^ { \gamma } \log ( p _ { t } ) .\tag{9}
$$

Compared to the original focal loss, which amplifies the loss for all lowconfidence samples, our improvement better eliminates extreme outliers, reducing noise interference and enhancing learning eficiency.

Self-supervised Reconstruction for Pre-training. Due to the limited CUA data and annotations (classification tags only), we carefully constructed a self-supervised reconstruction method for model pre-training to comprehensively explore the key features in the volume pool. Inspired by Masked Autoencoder, we first randomly masked a portion of the input data and introduced an encoder-decoder architecture to predict the missing information based only on the visible context. This can significantly enhance the model’s feature representation capability by learning the connection between local and global features, especially in the data scarcity scenario.

Specifically, 3D ResNet-18 was taken as the encoder, while in the decoder part, 3D transposed convolutions were used. Besides, skip connections were also equipped to link the features of diferent levels, bringing more spatial information to improve the decoding process. Diferent from pixel-level masking, we used the patch-based strategy to reduce the learning dificulty. We split the volume into non-overlapping patches with size $8 \times 8 \times 8$ voxels, producing a total of 3072 patches $( 1 2 8 / 8 { \times } 9 6 / 8 { \times } 1 2 8 / 8 )$ . The masking rate was set to 75%. We used the common reconstruction loss for training, which calculates errors between predicted $( \hat { x } _ { i } )$ and original $( x _ { i } )$ images in the masked regions (with N numbers of voxels):

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( x _ { i } - \hat { x _ { i } } ) ^ { 2 } .\tag{10}
$$

Then, we can inherit the model weights from the above 3D ResNet-18 encoder for the following classification.

Online Data Augmentation Strategy. To enhance model generalization across test cases, we further introduce an online data augmentation strategy. Specifically, for each test sample $v _ { i } .$ , we randomly augmented it K times with the common strategies, including translating, flipping, and rotation, to form a batch input $( \mathcal { K } \times 1 \times 1 2 8 \times 9 6 \times 1 2 8 )$ . Then, the classifier will output the probability set with size $\boldsymbol { { \kappa } } \times \boldsymbol { { C } }$ , where $C = 7$ indicates the number of CUA types. Through the average operation at the class channel and after a sof tmax, an optimized probability $p _ { o }$ is obtained. Then, the predicted tag for the current testing uterus volume $v _ { i }$ is defined as $c ^ { * } ( v _ { i } ) = a r g m a x ( p _ { o } )$

## 2.4. Experimental setting

All experiments were implemented using Python (version 3.8.0) and Py-Torch (version 2.0.0) on an NVIDIA A40 GPU with 48GB of memory. The self-supervised reconstruction was trained using the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ for 30 epochs. For classification, we also used the Adam optimizer with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ . The learning rate follows a linear decay schedule, decreasing by 10% every 40 epochs. The total number of epochs is set to 100. Models achieving the best performance on the validation set were selected for final evaluation.

## 2.5. Statistical analysis

We evaluated the model using diferent metrics, including Accuracy, Precision, Recall, F1-score, micro-AUC, macro-AUC, etc. 95% CIs were evaluated by bootstrapping with 1,000 resamples. For statistical analysis, Accuracy was compared using the paired McNemar test, while diferences in other metrics were assessed using two-sided paired permutation tests with 1,000 permutations. Moreover, for comparison between CUA-Net and sonographers, we used Bonferroni correction to update the p-values and reduce false positive rate. The adjusted p-value was calculated as:

Table 2: Comparison of diferent models on the CUA classification task. <sup>∗</sup> indicates statistical significance compared with all other methods $\left( p < 0 . 0 5 \right)$
<table><tr><td>Methods</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1 (%)</td><td>micro-AUC</td><td>macro-AUC</td></tr><tr><td>2D ResNet-18</td><td>42.86 (36.20–50.00)</td><td>42.85 (26.24–47.17)</td><td>30.52 (22.35–37.95)</td><td>28.98 (17.90–35.95)</td><td>0.8280 (0.7872–0.8620)</td><td>0.9108 (0.8489–0.9532)</td></tr><tr><td>I3D</td><td>69.90 (63.51–76.02)</td><td>57.23 (51.24–62.72)</td><td>46.77 (41.16–50.81)</td><td>44.15 (35.89–50.61)</td><td>0.9519 (0.9332–0.9677)</td><td>0.9820 (0.9730–0.9905)</td></tr><tr><td>Video Swin transformer</td><td>75.00 (69.11–80.88)</td><td>52.86 (50.16–56.01)</td><td>56.42 (51.52–61.15)</td><td>47.83 (41.55–53.61)</td><td>0.9660 (0.9525–0.9769)</td><td>0.9965 (0.9930–0.9991)</td></tr><tr><td>3D ResNet-18 (Baseline)</td><td>72.45 (65.82–78.33)</td><td>55.17 (37.80–60.14)</td><td>49.11 (44.47–52.40)</td><td>40.95 (31.95–47.12)</td><td>0.9342 (0.9076–0.9599)</td><td>0.9566 (0.9296–0.9784)</td></tr><tr><td>3D MedicalNet</td><td>81.12 (76.02–86.22)</td><td>52.83 (46.82–59.12)</td><td>69.05 (54.73–70.73)</td><td>57.39 (50.53–63.46)</td><td>0.9172 (0.8796–0.9450)</td><td>0.9946 (0.9906–0.9975)</td></tr><tr><td>3D DenseNet</td><td>82.65 (76.77–87.76)</td><td>69.80 (52.06–74.99)</td><td>75.24 (58.18–77.74)</td><td>68.06 (48.82–73.08)</td><td>0.9841 (0.9759–0.9910)</td><td>0.9996 (0.9987–0.9999)</td></tr><tr><td>3D ViT</td><td>78.06 (71.93–83.16)</td><td>82.56 (51.91–86.95)</td><td>76.65 (59.90–79.19)</td><td>68.14 (48.20–73.47)</td><td>0.9796 (0.9674–0.9877)</td><td>0.9988 (0.9972–0.9998)</td></tr><tr><td>3D nnMamba</td><td>84.18 (78.30–89.04)</td><td>85.31 (69.39–87.33)</td><td>80.81 (53.95–86.85)</td><td>75.59 (56.34–81.05)</td><td>0.9869 (0.9800–0.9925)</td><td>0.9971 (0.9948–0.9987)</td></tr><tr><td>CUA-Net</td><td>93.88* (89.80–97.45)</td><td>87.01* (69.51–91.30)</td><td>95.92* (79.85–97.68)</td><td>88.09* (69.73–92.79)</td><td>0.9982* (0.9963–0.9995)</td><td>0.9997* (0.9990–1.0000)</td></tr></table>

$$
p _ { \mathrm { c o r } } = \mathrm { m i n } \left( N _ { \mathrm { s o n o } } * N _ { \mathrm { m e t r i c s } } * p _ { \mathrm { r a w } } , 1 \right) ,\tag{11}
$$

where $N _ { \mathrm { s o n o } } = 2$ and $N _ { \mathrm { { m e t r i c s } } } = 4$ denote the numbers of sonographers and evaluated performance metrics within each comparison group, respectively. Here, $p _ { \mathrm { c o r } }$ and $p _ { \mathrm { r a w } }$ denote the corrected and raw p-values. $p _ { \mathrm { c o r } } < 0 . 0 5$ indicates a statistically significant performance diference.

## 3. Results

To validate the efectiveness of the proposed method, we compared CUA-Net with 2D/3D ResNet-18 and several mainstream advanced video/3D models. As shown in Table 2, CUA-Net achieved the best performance across all evaluation metrics, e.g., accuracy of 93.88% (95% CI: 89.80–97.45), and F1- score of 88.09% (95% CI: 69.73–92.79), etc. Compared with the second-best model 3D nnMamba, it improved accuracy by 9.70% and macro-AUC by 0.0026. Compared with 2D ResNet-18, 3D ResNet-18 achieved significantly better accuracy (75.00% vs. 42.86%, $p < 0 . 0 5 )$ and macro-AUC (0.9566 vs. 0.9108, $p \ < \ 0 . 0 5 )$ , verifying the advantages of 3D modeling. Accordingly, considering both performance and model complexity, we adopted the relatively lightweight 3D ResNet-18, with 33.2M parameters, as the backbone for all subsequent experiments.

As shown in Table 3, the naive 3D ResNet-18 achieves limited performance (Accuracy 72.45%, F1: 40.95%), falling short of clinical intelligent CUA classification requirements. Adding the DRS and HSM can improve the model performance, validating their eficacy in handling the data imbalance and hard sample learning issues. However, because hard samples are selected globally between the 80th and 95th loss percentiles, optimization may become biased toward dificult cases from certain classes, temporarily afecting classwise probability ranking. Moreover, some samples within the highest-loss group may represent informative rare/atypical cases rather than true noise. Excluding them may weaken class-specific discrimination and consequently reduce macro-AUC, which weights all classes equally, despite improvements in accuracy and micro-AUC. This efect was subsequently alleviated by SSR and ODA, which enhanced feature learning and improved prediction stability. Specifically, adding SSR boosts all metrics, including the previously-dropped precision, macro-AUC, etc. This proves that the self-supervised technique assists the model in fully utilizing limited data and capturing classificationrelevant relationships between local patches and global volumes. Incorporating ODA also enhances CUA analysis, allowing the model to observe and analyze volumes from diferent perspectives through online learning, ultimately refining classification decisions via self-adjustment. Finally, CUA-Net achieves an accuracy of 93.88% and an F1-score of 88.09%, further validating the overall contributions of the proposed strategies.

Table 3: Ablation study.
<table><tr><td>Methods</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1 (%)</td><td>micro-AUC</td><td>macro-AUC</td></tr><tr><td>Baseline</td><td>72.45 (65.82-78.33)</td><td>55.17 (37.80-60.14)</td><td>49.11 (44.47-52.40)</td><td>40.95 (31.95-47.12)</td><td>0.9342 (0.9076-0.9599)</td><td>0.9566 (0.9296-0.9784)</td></tr><tr><td>Baseline+DRS</td><td>80.10 (74.49-84.96)</td><td>74.97 (57.31-78.83)</td><td>61.20 (53.60-71.25)</td><td>62.04 (50.17-70.97)</td><td>0.9814 (0.9719-0.9874)</td><td>0.9759 (0.9672-0.9825)</td></tr><tr><td>Baseline+DRS+HSM</td><td>86.22 (81.86-90.31)</td><td>59.52 (57.78-61.90)</td><td>62.35 (58.21-65.73)</td><td>56.21 (51.90-60.10)</td><td>0.9871 (0.9782-0.9922)</td><td>0.9110 (0.8983-0.9212)</td></tr><tr><td>Baseline+DRS+HSM+SSR</td><td>90.31 (86.47-94.39)</td><td>75.78 (59.91-80.67)</td><td>78.96 (64.06-82.32)</td><td>76.34 (60.99-80.56)</td><td>0.9898 (0.9821-0.9961)</td><td>0.9967 (0.9932-0.9995)</td></tr><tr><td>Baseline+DRS+HSM+SSR+ODA (CUA-Net)</td><td>93.88 (89.80-97.45)</td><td>87.01 (69.51-91.30)</td><td>95.92 (79.85-97.68)</td><td>88.09 (69.73-92.79)</td><td>0.9982 (0.9963-0.9995)</td><td>0.9997 (0.9990-1.0000)</td></tr></table>

As shown in Fig. 3 (a), the ROC curves with AUC values (same as in Table 1) of diferent methods further prove the strong overall performance of our method under various thresholds. In Fig. 4, we visualize the confusion matrix of diferent methods to show the class-level analysis. It further validates the superiority of our CUA-Net (see Fig. 4 (f)) on all classes, compared to other methods. We noticed that the Arcuate uterus (True label=2) often exhibits poorer performance than others. This may be due to its similar characteristics to other uterine types, which can confuse network learning.

![](images/f0736c49b08dc51785982c858a4507c1a0e4323a20eed8177619d14161da454b.jpg)

![](images/4f85555c15108a9c19328a5e3652a997be8c6c827d743940a5c4beec2722946c.jpg)

Figure 3: ROC curves with AUC values for (a) diferent methods and (b) junior and senior sonographers. The AUC values for sonographers are approximated by connecting several points, i.e., (0,0), (FPR, TPR), and (1,1), following [29].  
![](images/3028012f59b94e41493d414c200ed2191cfa009700ad7ccb4e90ffd9d71f9273.jpg)

![](images/dedf1fe368d3aa771ce23bb283bf078a1e17ac78d415dbf1c90271d845a9d08e.jpg)

![](images/4ffdb4730d360660b84e00b3bb1fa2c226cc6472d90cf417ab7ca96dd1e30ecf.jpg)

![](images/e844897ff785abdc197f473a3f664771031df27d9ee27ac47af21b82e62a3718.jpg)

![](images/705f7ee0b22ce1b5b150f1258788d802aede9dc5c882073a717c5a95b08c189a.jpg)

![](images/daf96091d8a8bec4e484650b1a43be676d496489ac1b0920ded7a5163fcccb67.jpg)  
Figure 4: Confusion matrix of diferent methods, including (a) 2D ResNet-18, (b) 3D ResNet-18, 3D ResNet-18 with (c) DRS, (d) DRS+HSM, (e) DRS+HSM+SSR, and (f) DRS+HSM+SSR+ODA. 0-6 correspond to normal uterus, septate uterus, arcuate uterus, uterus didelphys, unicornuate uterus, T-shaped uterus, and bicornuate uterus, respectively.

Table 4: Performance comparison between CUA-Net and sonographers.
<table><tr><td>Methods</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1 (%)</td></tr><tr><td>Junior 1</td><td>74.49 (67.86–81.12)</td><td>57.73 (47.12–71.61)</td><td>55.50 (43.88–69.64)</td><td>55.76 (44.34–67.79)</td></tr><tr><td>Junior 2</td><td>75.51 (68.88–81.63)</td><td>56.48 (43.78–68.98)</td><td>63.51 (42.34–71.43)</td><td>56.10 (42.39–67.76)</td></tr><tr><td>Senior 1</td><td>94.90 (91.84–97.96)</td><td>97.36 (94.83–99.01)</td><td>96.34 (93.35–98.42)</td><td>96.82 (94.02–98.65)</td></tr><tr><td>Senior 2</td><td>90.31 (86.22–94.39)</td><td>95.85 (76.87–97.45)</td><td>83.52 (72.79–93.58)</td><td>87.13 (74.57–94.81)</td></tr><tr><td>CUA-Net</td><td>93.88 (89.80–97.45)</td><td>87.01 (69.51–91.30)</td><td>95.92 (79.85–97.68)</td><td>88.09 (69.73–92.79)</td></tr></table>

We also compared the CUA-Net with two sonographer groups: one involving two junior sonographers (<5 years of experience) and one two senior sonographers (>15 years of experience). Within each group, a total of eight hypothesis tests were conducted, comprising two sonographer comparisons across four performance metrics (i.e., $p _ { c o r } = p _ { r a w } / 8 )$ . As shown in Table 4 and Fig. 5, CUA-Net achieved higher accuracy, precision, recall, and F1- score than both junior sonographers. After Bonferroni correction, all diferences remained statistically significant (corrected p-values< 0.05), indicating that CUA-Net consistently outperformed both junior sonographers across all evaluated metrics. For the senior group, no statistically significant diferences were observed between CUA-Net and the two senior sonographers in most comparisons after Bonferroni correction $( \mathrm { i . e . , 7 / 8 }$ metrics with corrected p-values> 0.05, only Precision for Senior 1 vs. CUA-Net with corrected pvalues=0.04). Overall, these results suggest that CUA-Net outperformed junior sonographers, and achieved competitive performance relative to the senior sonographers across most evaluated metrics.

Fig. 6 presents qualitative visualization results based on t-SNE and Class Activation Mapping (CAM). The t-SNE visualization reveals that our method achieves efective feature modeling, successfully learning discriminative representations for diferent categories. In addition, we applied the Grad-CAM technique on the testing set to examine the network’s attention regions across diferent cases. Specifically, Grad-CAM was first used to generate 3D heatmaps, and then, based on additional annotated coronal plane parameters, the corresponding 2D slices and their projected Grad-CAM responses were extracted. See the 10 pseudo-color maps in Fig. 6, the redder the color, the higher the level of attention from the model, and the black boxes highlight the regions that senior sonographers typically focus on. We also report quantitative attention-alignment analyses using the Dice similarity coeficient (Dice) and intersection over union (IoU), as shown in Table 5. It should be noted that CUA-Net is a classification model trained using only category labels, without any segmentation/detection annotations. Hence, the attention maps were generated in a weakly supervised manner. The quantitative Dice and IoU results show a moderate degree of overlap between the regions highlighted by CUA-Net and the diagnostic regions of interest identified by sonographers across diferent uterine categories. These findings indicate that CUA-Net can focus on clinically relevant anatomical regions to make decision, further supporting its efectiveness and interpretability.

![](images/fecf30139f2bc6b65de7a6821f0463903ada1fad5cd4bf26edf722e455d83b06.jpg)

![](images/b151fea2f68b937a251aaca4bff539e40552b4d88013623378d903b2cb4f5c08.jpg)

![](images/4cba7f8e6f68568b1b129f6d04a9f673459582455460008755444ea4b40e4d62.jpg)

![](images/5dfdfa3bbd240eca171acf4a33ccb30f441219d8eb0ff5605894efb89be56098.jpg)  
Figure 5: Performance comparison between CUA-Net and junior/senior sonographers. Radar charts illustrate (a) precision, (b) recall, and (c) F1-score across diferent categories (0-6: normal uterus, septate uterus, arcuate uterus, uterus didelphys, unicornuate uterus, T-shaped uterus and bicornuate uterus) for CUA-Net (blue), average juniors (red), and average seniors (green), respectively. (d) Bar chart presents the overall precision, recall, accuracy, and F1-score of juniors (Junior1/2), seniors (Senior1/2), and CUA-Net.

![](images/02a3d3a66f500bd4ac55a79ff5f7441b7a5c96ed0ccb992a99af4757b3d84cc0.jpg)  
Figure 6: t-SNE visualization with CAM results indicating the attention regions by our CUA-Net. The black bounding boxes show the key regions provided by senior sonographers, and the black arrows point out the detailed structures. The categories 0 to 6 correspond to normal uterus, septate uterus, arcuate uterus, uterus didelphys, unicornuate uterus, T-shaped uterus and bicornuate uterus, respectively.

To further verify the generalizability of the proposed CUA-Net, we evaluated its performance on an external test cohort. As shown in Table 6, we compared CUA-Net with eight competing methods. It achieved the best performance across all evaluation metrics, with an accuracy of 91.52%, an F1-score of 81.49%, etc (detailed performance refers to Fig. 7). These improvements were statistically significant compared with all competing methods $\left( p < 0 . 0 5 \right)$ . Compared with the average declines of 9.86, 6.51, 15.77, and 9.79 percentage points in accuracy, precision, recall, and F1-score and 0.0454 and 0.1350 in micro- and macro-AUC across the eight competing methods (from internal to external testing), CUA-Net showed markedly smaller drops of 2.36, 3.74, 7.29, and 6.60 percentage points and 0.0037 and 0.0007, respectively. Moreover, the class-wise confusion matrix demonstrates reliable generalization across common CUA categories, while also showing preliminary potential for recognizing rare anomalies such as unicornuate, bicornuate, and T-shaped uterus (Fig. 7). These results further support CUA-Net’s acceptable robustness and generalization to independent external data.

Table 5: Quantitative evaluation of attention alignment across diferent uterine categories. Values are reported as mean ± standard deviation.
<table><tr><td>Category</td><td>Dice</td><td>IoU</td></tr><tr><td>Normal uterus</td><td> $0 . 7 5 2 5 \pm 0 . 0 8 5 4$ </td><td> $0 . 6 8 7 9 \pm 0 . 1 3 2 6$ </td></tr><tr><td>Septate uterus</td><td> $0 . 7 2 1 3 \pm 0 . 0 9 7 6$ </td><td> $0 . 6 3 5 8 \pm 0 . 1 4 1 2$ </td></tr><tr><td>Arcuate uterus</td><td> $0 . 6 8 4 7 \pm 0 . 1 1 2 8$ </td><td> $0 . 5 8 7 5 \pm 0 . 1 5 3 4$ </td></tr><tr><td>Unicornuate uterus</td><td> $0 . 7 3 8 9 \pm 0 . 0 9 3 1$ </td><td> $0 . 6 5 4 2 \pm 0 . 1 3 7 5$ </td></tr><tr><td>Uterus didelphys</td><td> $0 . 7 2 1 4 \pm 0 . 1 0 1 3$ </td><td> $0 . 6 2 1 0 \pm 0 . 1 3 8 7$ </td></tr><tr><td>T-shaped uterus</td><td> $0 . 7 0 2 4 \pm 0 . 1 0 5 9$ </td><td> $0 . 6 0 8 7 \pm 0 . 1 4 8 6$ </td></tr><tr><td>Bicornuate uterus</td><td> $0 . 6 5 9 8 \pm 0 . 1 2 8 6$ </td><td> $0 . 5 8 4 5 \pm 0 . 1 5 1 1$ </td></tr></table>

Table 6: Performance comparison between CUA-Net and competing methods on the independent external test set. Best performances are highlighted in bold, and <sup>∗</sup> indicates statistical significance compared with all other methods $\left( p < 0 . 0 5 \right)$ . Red and blue values indicate increases and decreases, respectively, relative to the internal test set in Table 3.
<table><tr><td>Methods</td><td>Accuracy (%)</td><td>Precision (%)</td><td>Recall (%)</td><td>F1-score (%)</td><td>Micro-AUC</td><td>Macro-AUC</td></tr><tr><td>2D ResNet-18</td><td>39.962.90↓ (34.93–44.42)</td><td> $1 2 . 8 4 _ { 3 0 . 0 1 \downarrow }$  (5.39–20.37)</td><td> $1 5 . 2 1 _ { 1 5 . 3 1 \downarrow }$   $\begin{array} { c } { { ( 1 4 . 2 9 - 1 6 . 7 5 ) } } \\ { { 3 2 . 3 7 _ { 1 4 . 4 0 4 } } } \end{array}$ </td><td> $9 . 7 8 _ { 1 9 . 2 0 \downarrow }$  (7.83–12.40)</td><td>0.71530.1127↓ (0.6889–0.7418)</td><td>0.62190.28894 (0.5703–0.6614)</td></tr><tr><td>I3D</td><td>55.5814.32↓ (51.56–59.60)</td><td>51.275.96.↓</td><td>(28.98–35.50)</td><td>34.989.174 (30.44–38.30)</td><td>0.84920.1027↓ (0.8206–0.8724)</td><td>0.73250.2495↓ (0.6654–0.7868)</td></tr><tr><td>Video Swin Transformer</td><td>55.3419.66↓ (50.22–59.38)</td><td> $\begin{array} { c } { { ( 4 7 . 8 1 - 5 4 . 5 2 ) } } \\ { { 5 5 . 5 8 _ { 2 . 7 2 \uparrow } } } \end{array}$  (53.87–56.93)</td><td>41.3715.05↓ (38.56–44.37)</td><td>37.1410.69↓ (33.34–40.15)</td><td>0.88220.0838↓ (0.8641–0.9010)</td><td>0.86500.1315↓ (0.8142–0.9233)</td></tr><tr><td>3D ResNet-18 (Baseline)</td><td>68.533.92↓ (64.39–72.77)</td><td>51.094.08↓ (47.88–54.18)</td><td>41.127.99↓ (37.58–44.73)</td><td>42.441.49↑ (37.82–45.92)</td><td>0.93220.0020↓ (0.9150–0.9456)</td><td>0.88060.0760↓ (0.8584–0.9008)</td></tr><tr><td>3D MedicalNet</td><td>59.1521.97↓ (54.69–63.39)</td><td>60.978.14† (58.71–63.30)</td><td>36.7232.33↓ (33.50–40.26)</td><td>37.7019.69↓ (33.64–41.61)</td><td>0.90480.0124↓ (0.8848–0.9205)</td><td>0.83690.1577↓4 (0.8059–0.8644)</td></tr><tr><td>3D DenseNet</td><td>77.684.974 (73.88–81.59)</td><td>60.828.98↓ (51.58–69.13)</td><td>61.1314.11↓4 (56.60–67.20)</td><td>56.2411.82↓ (49.46–62.40)</td><td>0.96830.0158↓ (0.9581–0.9768)</td><td>0.94890.0507↓ (0.9017–0.9812)</td></tr><tr><td>3D ViT</td><td>75.003.06↓ (70.98–78.57)</td><td>74.338.23↓ (63.75–79.11)</td><td>60.4516.20↓ (51.38–67.73)</td><td> $6 3 . 4 7 _ { 4 . 6 7 \downarrow }$  (53.01–69.11)</td><td>0.96750.0121↓ (0.9569–0.9744)</td><td>0.92620.0726↓ (0.9184–0.9334)</td></tr><tr><td>3D nnMamba</td><td> $7 6 . 1 1 _ { 8 . 0 7 \downarrow }$  (71.65–79.91)</td><td>79.65.66↓4  $\begin{array} { c } { { ( 7 8 . 5 2 - 8 0 . 7 3 ) } } \\ { { 8 3 . 2 7 ^ { * } . } } \end{array}$ </td><td> $7 0 . 0 3 _ { 1 0 . 7 8 \downarrow }$ </td><td>71.024.57↓ (68.29–73.19)</td><td>0.96490.0220↓ (0.9540–0.9735)</td><td> $0 . 9 4 3 7 _ { 0 . 0 5 3 4 , }$  (0.9368–0.9507)</td></tr><tr><td>CUA-Net</td><td> ${ \bf 9 1 . 5 2 ^ { * } } _ { 2 . 3 6 \downarrow }$   $\mathbf { ( 8 9 . 0 6 - 9 4 . 2 0 ) }$ </td><td>(69.18–93.77)</td><td> $\begin{array} { c } { { ( 6 7 . 4 3 - 7 2 . 0 2 ) } } \\ { { 8 8 . 6 3 ^ { * } } _ { 7 . 2 9 \downarrow } } \end{array}$  (68.73–95.14)</td><td> $\mathbf { 8 1 . 4 9 ^ { * } } _ { 6 . 6 0 \downarrow }$  (67.71–91.26)</td><td> $\mathbf { 0 . 9 9 4 5 ^ { * } } _ { 0 . 0 0 3 7 \downarrow }$  (0.9917–0.9967)</td><td> $\mathbf { 0 . 9 9 9 0 ^ { * } } _ { 0 . 0 0 0 7 \downarrow }$  (0.9976–0.9998)</td></tr></table>

![](images/68a4d1019c82d0905346ee1742f5987f87150b091b07b829c8be95981df52f8e.jpg)

![](images/c807db7152af4aa3032395e6996a1347d04af2712c01a4beea73cb3e8a1595c5.jpg)  
Figure 7: (a) ROC curves with AUC values and (b) the confusion matrix of CUA-Net on the external test cohort. 0-6 denotes: normal uterus, septate uterus, arcuate uterus, unicornuate uterus, uterus didelphys, T-shaped uterus, bicornuate uterus, respectively.

## 4. Discussion

CUA is among the most common gynecological disorders, with a relatively high prevalence rate. It is closely associated with female infertility and may also lead to reproductive complications such as recurrent miscarriage, intrauterine growth restriction, preterm birth, and retained placenta. Therefore, accurate CUA classification is crucial for selecting appropriate surgical approaches and related treatments. US is a primary tool for diagnosing CUAs; however, conventional 2D US cannot capture the coronal plane, whereas 3D US visualizes the coronal plane, providing a comprehensive view of uterine structures to drive accurate diagnosis. Specifically, in traditional clinical workflow, sonographers must first locate and reconstruct the 2D standard coronal plane that meets diagnostic criteria within the high-dimensional 3D space, and then perform measurements on this plane to obtain an accurate CUA diagnosis. This process is highly dependent on the clinician’s experience, time-consuming, and negatively impacts diagnostic eficiency.

With the widespread adoption of AI, its integration into medical ultrasound image analysis has continued to advance. Several deep learning based methods have been introduced to achieve intelligent analysis in 3D uterine US. Most of them focus solely on detecting standard planes in 3D space [25, 26, 27], neglecting the vital CUA classification task. Two recent works explored classifying (1) normal/abnormal cases [28] and (2) normal/CUA samples [30] based on deep learning techniques. However, they still highly rely on the plane localization process, which may degrade the diagnosis performance due to the potential error propagation, e.g., a wrongly located plane will very likely lead to misclassification of CUA. Hence, designing a deep learning-based model that directly processes 3D uterine data and outputs CUA predictions in one step could improve overall accuracy by avoiding error accumulation. This could also reshape the clinical workflow and greatly enhance diagnostic efectiveness.

In this paper, we built the CUA-Net to achieve the above goal. Our proposed CUA-Net builds upon the baseline, i.e., 3D ResNet-18, and achieves significant performance improvements through the stepwise integration of data re-sampling (DRS), hard sample mining (HSM), self-supervised reconstruction (SSR), and online data augmentation (ODA) techniques. Due to their low incidence in the real world, complex CUAs such as T-shaped uterus and bicornuate uterus are underrepresented in our collected dataset. Our proposed strategies can alleviate the class imbalance issue, learn from the dificult cases, extract features from the limited volumes, and correct the wrong predictions during testing, to comprehensively improve the model performance on CUA classification. In conclusion, our CUA-Net achieved the best performance among diferent competitors on all metrics, including Accuracy (93.88%), Precision (87.01%), Recall (95.92%), F1 (88.09%), Micro-AUC (0.9982), and Macro-AUC (0.9997).

To assess clinical applicability, we compared CUA-Net with both junior and senior sonographers. CUA-Net outperformed junior sonographers across all metrics, demonstrating its potential to assist them in CUA classification. CUA-Net also demonstrated performance comparable to that of senior sonographers, with no statistically significant diferences observed in most metrics. Besides, CUA-Net only required about 0.1s for testing, significantly faster than juniors (61.8s) and seniors (40.6s). This demonstrates the model’s eficiency and efectiveness, enabling real-time clinical classification.

Regarding the limitations, CUA-Net may still misclassify certain CUA categories. For example, it showed relatively poor performance in identifying arcuate uterus (Fig. 4 (f)). This may be because the shallow fundal indentation of an arcuate uterus can resemble that of a small septate or Tshaped uterus on 3D US. These overlapping morphological characteristics make it dificult for the model to learn discriminative features. Moreover, without explicit morphological constraints during training, the model may rely excessively on local appearance cues, thereby reducing generalizability and causing misclassifications, such as confusion with unicornuate uterus in the external test set (Fig. 7 (b)). For the corresponding solutions, we believe that there are two feasible approaches. The first is to incorporate uterine morphology segmentation, as it could provide explicit anatomical guidance and encourage the model to focus on global uterine geometry. The second is to use controllable data synthesis to generate more diverse and balanced samples, thereby enriching morphological variations and potentially improving classification accuracy.

In future work, we will collect more underrepresented data (e.g., bicornuate uterus) to more comprehensively validate our method. Additionally, we will develop stronger fine-grained feature discrimination strategies to better distinguish CUAs with similar characteristics. Finally, we aim to promote broader multi-center collaboration and package the model as standalone software in clinical US systems to prospectively validate its clinical value and assist physicians in improving CUA diagnostic accuracy.

## Acknowledgments

This work was supported by the Guangxi Key Research and Development Program (No. AB23026042), the National Natural Science Foundation of China (No. 12326619), the Frontier Technology Development Program of Jiangsu Province (No. BF2024078), and the Guangxi Natural Science Foundation (No. 2025GXNSFAA069471).

## References

[1] P. Bortoletto, P. A. Romanski, S. M. Pfeifer, Müllerian anomalies: presentation, diagnosis, and counseling, Obstetrics & Gynecology 143 (3) (2024) 369–377.

[2] J. E. Dietrich, Diagnosis and management of mullerian anomalies across difering resource settings: Worldwide adaptations, Journal of Pediatric and Adolescent Gynecology 35 (5) (2022) 536–540.

[3] M. Abhinaya, B. U. Rani, V. Adilakshmi, N. B. Babu, A study of mullerian anomalies in pregnancy: Case series in a tertiary care centre., International Journal of Medicine & Public Health 14 (1) (2024).

[4] S. AM FERTIL, The american fertility society classification of adnexal adhesions, distal tubal occlusion secondary to tubal ligation, tubal pregnancies, mullerian anomalies and intrauterine adhesions, Fertil Steril 49 (1988) 944–955.

[5] Y. Y. Chan, K. Jayaprakasan, J. Zamora, J. G. Thornton, N. Raine-Fenning, A. Coomarasamy, The prevalence of congenital uterine anomalies in unselected and high-risk populations: a systematic review, Human reproduction update 17 (6) (2011) 761–771.

[6] M. d. A. Coelho Neto, A. Ludwin, F. Petraglia, W. d. P. Martins, Definition, prevalence, clinical relevance and treatment of t-shaped uterus: systematic review, Ultrasound in Obstetrics & Gynecology 57 (3) (2021) 366–377.

[7] A. Seyhan, S. Ertas, B. Urman, Prevalence of t-shaped uterus among fertile women based on eshre/esge and congenital uterine malformation by experts (cume) criteria, Reproductive BioMedicine Online 43 (3) (2021) 515–522.

[8] I. d. M. P. e Passos, R. L. Britto, Diagnosis and treatment of müllerian malformations, Taiwanese Journal of Obstetrics and Gynecology 59 (2) (2020) 183–188.

[9] A. Ludwin, S. M. Pfeifer, Reproductive surgery for müllerian anomalies: a review of progress in the last decade, Fertility and Sterility 112 (3) (2019) 408–416.

[10] Y. Huang, X. Yang, R. Li, J. Qian, X. Huang, W. Shi, H. Dou, C. Chen, Y. Zhang, H. Luo, et al., Searching collaborative agents for multi-plane localization in 3d ultrasound, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, 2020, pp. 553– 562.

[11] Y. Zhang, Y. Huang, H. Dou, X. Zhu, C. Ling, Z. Yang, L. Liang, J. Li, S. Liang, R. Li, et al., Artificial intelligence for detecting fetal orofacial clefts and advancing medical education, arXiv preprint arXiv:2603.06522 (2026).

[12] D. Shen, G. Wu, H.-I. Suk, Deep learning in medical image analysis, Annual review of biomedical engineering 19 (1) (2017) 221–248.

[13] Y. Huang, X. Yang, X. Huang, X. Zhou, H. Chi, H. Dou, X. Hu, J. Wang, X. Deng, D. Ni, Fourier test-time adaptation with multi-level consistency for robust classification, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2023, pp. 221–231.

[14] J. Wang, S. Wang, Y. Zhang, Deep learning on medical image analysis, CAAI Transactions on Intelligence Technology 10 (1) (2025) 1–35.

[15] X. Zhou, Y. Huang, H. Dou, S. Chen, A. Chang, J. Liu, W. Long, J. Zheng, E. Xu, J. Ren, et al., Ctrl-genaug: Controllable generative augmentation for medical sequence classification, International Journal of Computer Vision 134 (5) (2026) 216.

[16] Y. Huang, X. Yang, L. Liu, H. Zhou, A. Chang, X. Zhou, R. Chen, J. Yu, J. Chen, C. Chen, et al., Segment anything model for medical images?, Medical Image Analysis 92 (2024) 103061.

[17] Y. Huang, A. Chang, H. Dou, X. Tao, X. Zhou, Y. Cao, R. Huang, A. F. Frangi, L. Bao, X. Yang, et al., Flip learning: Weakly supervised erase to segment nodules in breast ultrasound, Medical Image Analysis 102 (2025) 103552.

[18] C. Chen, Y. Huang, X. Yang, X. Hu, Y. Zhang, T. Tan, W. Xue, D. Ni, Enhancing fetal ultrasound image quality assessment with multi-scale fusion and clustering-based optimization, Biomedical Signal Processing and Control 102 (2025) 107249.

[19] Z. Liu, Y. Huang, L. Liu, C. Zhang, H. Lin, T. Han, Z. Zhu, Y. Chen, R. Chen, D. Ni, et al., Mreg: A novel regression model with moe-based video feature mining for mitral regurgitation diagnosis, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025, pp. 383–392.

[20] H. Liang, Y. Zhang, X. Zhu, Y. Huang, X. Du, S. Liang, J. Xu, Y. Zhang, C. Sheng, Y. Liu, et al., Faa-net: Fetal abdominal anomaly diagnosis in prenatal ultrasound via llm-enhanced multi-instance learning, Medical Image Analysis (2026) 104201.

[21] T. Yang, L. Yuan, P. Li, P. Liu, Real-time automatic assisted detection of uterine fibroid in ultrasound images using a deep learning detector, Ultrasound in Medicine & Biology 49 (7) (2023) 1616–1626.

[22] K. Dilna, J. Anitha, A. Angelopoulou, E. Kapetanios, T. Chaussalet, D. J. Hemanth, Classification of uterine fibroids in ultrasound images using deep learning model, in: International Conference on Computational Science, Springer, 2022, pp. 50–56.

[23] E. Boneš, M. Gergolet, C. Bohak, Ž. Lesar, M. Marolt, Automatic segmentation and alignment of uterine shapes from 3d ultrasound data, Computers in biology and medicine 178 (2024) 108794.

[24] X. Yang, H. Dou, R. Huang, W. Xue, Y. Huang, J. Qian, Y. Zhang, H. Luo, H. Guo, T. Wang, et al., Agent with warm start and adaptive dynamic termination for plane localization in 3d ultrasound, IEEE Transactions on Medical Imaging 40 (7) (2021) 1950–1961.

[25] X. Yang, Y. Huang, R. Huang, H. Dou, R. Li, J. Qian, X. Huang, W. Shi, C. Chen, Y. Zhang, et al., Searching collaborative agents for multi-plane localization in 3d ultrasound, Medical Image Analysis 72 (2021) 102119.

[26] Y. Zou, H. Dou, Y. Huang, X. Yang, J. Qian, C. Zhen, X. Ji, N. Ravikumar, G. Chen, W. Huang, et al., Agent with tangent-based formulation and anatomical perception for standard plane localization in 3d ultrasound, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2022, pp. 300–309.

[27] Y. Huang, Y. Zou, H. Dou, X. Huang, X. Yang, D. Ni, Localizing standard plane in 3d fetal ultrasound, in: 3D Ultrasound, CRC Press, 2023, pp. 239–269.

[28] H. Dou, Y. Huang, Y. Huang, X. Yang, C. Zhen, Y. Zhang, Y. Xiong, W. Huang, D. Ni, Standard plane localization using denoising difusion model with multi-scale guidance, Computer Methods and Programs in Biomedicine (2025) 108619.

[29] W. Zhou, Y. Yang, C. Yu, J. Liu, X. Duan, Z. Weng, D. Chen, Q. Liang, Q. Fang, J. Zhou, et al., Ensembled deep learning model outperforms human experts in diagnosing biliary atresia from sonographic gallbladder images, Nature communications 12 (1) (2021) 1259.

[30] Y. Huang, Y. Xu, H. Dou, J. Deng, X. Yang, H. Zheng, D. Ni, Uncertainty-aware difusion and reinforcement learning for joint plane localization and anomaly diagnosis in 3d ultrasound, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025, pp. 650–660.