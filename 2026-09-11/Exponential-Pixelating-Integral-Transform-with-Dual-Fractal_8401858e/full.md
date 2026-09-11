# Exponential Pixelating Integral Transform with Dual Fractal Features for Enhanced Chest X-Ray Abnormality Detection

Naveenraj K<sup>a</sup>, Sri Ram Macharla<sup>b</sup>, Kanimozhi M<sup>c</sup>, Sudhakar M S <sup>c</sup>\*

a – The Construct, Barcelona, Spain

b – Involgixs Inc, USA

c - School of Electronics Engineering, Vellore Institute of Technology, Vellore, Tamilnadu, India

\* - Corresponding Author

## Abstract

The heightened prevalence of Respiratory Disorders (RD), particularly exacerbated by a significant upswing in fatalities due to the novel coronavirus, underscores the critical need for early detection and timely intervention. This imperative is paramount, possessing the potential to profoundly impact and safeguard numerous lives. Medically, chest radiography stands out as an essential and economically viable medical imaging approach for diagnosing and assessing the severity of diverse RD. However, their detection in Chest X-rays (CXR) is a cumbersome task even for well-trained radiologists owing to low contrast issues, overlapping of the tissue structures, subjective variability, and the presence of noise. To address these issues, a novel analytical model termed Exponential Pixelating Integral (EPI) is introduced for the automatic detection of infections in CXRs in this work. Initially, the presented EPI enhances the pixel intensities to overcome the low-contrast issues that are then polar-transformed followed by their representation using the locally invariant Mandelbrot and Julia fractal geometries for effective distinction of structural features. The collated features labeled Exponential Pixelating Integral with dually characterized Fractal features (EPIFF) are then classified by the non-parametric Multivariate Adaptive Regression Splines (MARS) to establish an ensemble model between each pair of classes for effective diagnosis of diverse Respiratory Disorders. Rigorous analysis of the proposed diagnostic framework on large medical benchmarked datasets showcases its superiority over its peers by registering a higher classification accuracy and F1 scores ranging from 98.46-99.45% and 96.53- 98.10% respectively, making it a precise and interpretable automated system for diagnosing diseases.

Keywords Exponential Pixelating Integral; Fractal Geometry; Multivariate Adaptive Regression Splines; Respiratory Disorders

## 1. Introduction

Globally Respiratory Disorders are the major cause of death, according to the World Health Organization. Especially, Edema, Effusion, Emphysema, COVID-19, and Pneumonia are the most common lung diseases found in both children and adults due to the infection caused by viruses, bacteria, and fungi [1]. Pulmonary edema occurs because of the accumulation of excess fluid in the alveoli of the lungs leading to shortness of breath and ending in death if untreated. Likewise, Effusion is caused by the buildup of excess fluid between the layers of the pleura outside the lungs and poses a setback in distinguishing from edema in CXRs. While the progressive lung disease emphysema leads to the gradual deterioration of lung tissue, particularly the alveoli, it culminates in the bursting of air sacs with a simultaneous reduction of lung surface area, obstructing breathing. In addition to these conditions, the infection of the lungs by bacteria, fungi, and viruses, such as pneumonia, results in inflammation and fluid accumulation. Likewise, COVID-19 induced by Syndrome Coronavirus 2 (SARS-CoV-2) virally infects the lungs or pneumonia affecting the upper respiratory tract. Apart from viral pneumonia, bacterial pneumonia caused by pathogens such as Streptococcus leads to lung inflammation[2].

According to recent studies, around 500 million are diagnosed with diverse pneumonia, and nearly 1.5 million die each year [4]. Also, the contagious nature of a few of these ailments requires prompt diagnosis and isolation, with varying treatment options[3]. Therefore, early identification and intervention will greatly improve the course of the illness, slowing progression, reducing symptoms, and decreasing the frequency of exacerbations [3,5,6]. Over the years, CXRs’ and Computed Tomography (CT) modalities have played an important role in acutely diagnosing such ailments by

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

assessing their severity through fluid presence, lung space, and other lesions. While the CXR is a fast imaging, less sensitive, and cost-effective modality than the CT, thereby, making it widely popular in the identification of common respiratory diseases, especially in low-resource hospitals[7,8].

However, CXRs are difficult to interpret because of anatomy ending in structural superimposition [9]. The natural pattern of branching blood arteries in the lung fields makes it difficult for even the most skilled radiologists to tell infiltrates apart from normal tissue, and minuscule nodules hinder lesion identification [10] and manual detection is time-consuming which are substantial variations in severity assessments. In addition, a lack of expertise contributes to a high rate of inaccurate assessments. These issues highlight the clinical significance and complexity of chest radiographs thereby, motivating research on automated diagnostic techniques to aid radiologists in image interpretation. Also, diagnostic efficiency, precision, and speed can be significantly enhanced by automating these processes on a large scale, and accessible to all.

As a result, more precise and automated interpretation methods for diagnosing RDs are desired and are broadly categorized as conventional handcrafted or Deep Learning (DL) models, depending on the choice of feature acquisition. Numerous investigations have demonstrated that convolution neural networks (CNN) are superior and have led to an increase in their utilization for chest diagnosis [3,11– 16]. In particular, the Deep CNN [17], utilized fractional-order derivative to extract the textural details of the ROC region from CXRs, which are then fed to a gray relational analysis for diagnosis of diverse RDs. Similarly, [18] determined the best convolution architecture using the learning-by-training mechanism that effectively detected pneumonia in CXRs. However, the accuracy and reliability of models depended heavily on the labels given by humans and on the choice of hyperparameters. On the other hand, Transfer learning successfully identified various RD types from CXRs [1,14,15,19,20]. Especially, the VGG-based architecture [21] exploited transfer learning for detecting pneumonia by localizing the affected area in CXRs. While this approach achieved higher precision, the need for an individual network for detection and localization amplifies its complexity. Alternatively, [22] fused handcrafted features extracted from the segmented lung region using the Info-MGAN network for effective disease classification. However, training the MGAN network is quite challenging and its performance was extremely sensitive to the number of labeled samples and used parameters. Perhaps, the unreliable and less interpretable nature of the above DL models restrain their extension to real-time diagnosis. To gain interpretability [23] realized the Multi-task deep-end-to-end COMiT-Net to diagnose infections from CXRs followed by segmentation of lesion regions. Likewise, [24] used an explainable artificial intelligence (XAI) model that localized the area with the greatest likelihood of categorized images using a trained CNN model. However, the inclusion of numerous models significantly increases the model's complexity while decreasing its accuracy.

Alternatively, [25] utilized the task-specific adversarial network to increase the domain-invariant features and an uncertainty-aware ensembling model for predicting unseen data. The dilation varying CNN (CovXNets) [13] extracted features across various resolutions and performed localization using the integrated gradient-based discriminative model for distinguishing the abnormal regions. The confidence-aware anomaly detection (CAAD) model [26], predicted abnormal regions in CXR using low confidence and high anomaly detection scores. Likewise, cascaded rotation-invariant augmented hand features [11] coupled with DL in the ensemble framework effectively discriminated lesions based on the rank of probability score. Nevertheless, the integration of numerous DL networks into the aforementioned model makes it extremely complicated. Alternatively, the semi-supervised open set domain adversarial network (SODA) [27] effectively aligned data distributions at both the domain and subspace levels, and extracted features that were closely related to pathology location, achieving improved classification performance. Alternately, the memristive crossbar CNN array [28] classified the tunable Q-wavelet features acquired from decomposed CXRs for better storage, and quicker processing. Similarly, the anatomy-aware (AA) DL model [8], underlined the anatomical information for severity determination of pneumonia by learning crucial features at the disease level employing lung involvement scores. Although these DL variants provided superior accuracy, their black-box nature and high reliability on known targets question their extension to random real-time environments comprising unexpected outcomes. Also, realizing them on a simple hardware framework is practically less feasible thereby, demanding high-performing GPUs insisting on several weeks for

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

model training. Also, an array of DL limitations refrain from their usage in medical imaging analysis including the lack of domain-specific knowledge and the position, density, and spread of fluid, and space varying significantly between cases, thereby, making it difficult to develop a one-size-fits-all model. In addition, their lack of transparency with interpretability is notoriously challenging when employed in medical diagnostics, where practitioners are uncertain of the arrived conclusion, which is a potential pitfall. Thereby, mandating well-defined and simple mathematical models in real-time diagnosis.

Analytical and machine learning models, on the other hand, are simple, easy to train, deploy, quick, and ensure reliability. Furthermore, ground-glass opacities derived from CXRs characterized textual information concerned with the parametric tuning of their patterns aiding accurate diagnosis [29] across diverse imaging modalities. The textual model [30], determined the RDs in CXR images using the Nearest Neighbour scheme using wavelet coefficients. The method performed moderate classification despite being simple. The fuzzy c means [31] clustered weighted pixels for pneumonia detection from frequency-transformed CXR images using the stationary wavelet transform. The pixelwise computations incurred more time than their peers. To annotate lesions on CXR images [32], utilized a Multiple-Instance utilizing Support Vector Machine (miSVM) tuned by occurrence probabilities for accurate label prediction, rather struggled in lesion localization. This uncertainty was addressed in [32], [33] including miSVM in an active learning framework to localize lesions using expert relabeling. Instead [34] integrated multiscale statistical shape characteristics with various textural features of distinct lung boundaries using multiresolution approaches for boosting stability with accuracy while the model was less noise resilient. Rather, [4] ensured uniformity by augmenting the lung-segmented CXR images using affine transformations followed by a pair-wise comparison to determine the pneumonia likelihood using the EMD (Earth Mover Distance). Similarly [35] ranked the weighted entropies of multiple-criteria decision models for COVID-19 offering enhanced classification accuracy. The merger of several models considerably increased the process complexity, making it unsuitable for real-time applications.

The Random Forest (RF) variant of [36] selected crucial features based on the permutation score produced by the Boruta technique for the assessment of COVID-19 risk. To reduce the complexity, the RF classifier generated an associative tree that was unsuitable for imbalanced datasets. While [37], coupled Shannon entropy with fractal dimensions for the detection of pneumonia in CXR. However, the method was unable to localize the infected area and its severity level based on the magnitude of the fractal dimension. An ensemble of machine learning classifiers [6] quantified pneumonia severity based on the selection of integrated handcraft features using Principal Component Analysis (PCA) with Recursive Feature Elimination (RFE) techniques. Similarly, [38] blended fractal dimension, radiomics, and superpixel-based histon to diagnose pneumonia in CXRs registered higher accuracy while, its inability to distinguish the different infiltration patterns and the selection of the masking process offered less usage. The automatic diagnostic [39] detected infections based on the significant score (Pvalue<0.0001) by determining the left lower lung field’s white pixels ratio on the segmented CXR. However, this hypothesis testing results were unreliable and completely relied on the sample size. To limit the influence of noisy labels, [40] weighted data distribution across multiple levels was performed followed by engagement of a variety of feature selection techniques based on the confidence scores that are then KNN classified as COVID-19 or not. Despite their usefulness, the aforementioned techniques demand complicated algorithms and are less effective than DL models when it comes to identifying lung infections. The pitfalls of both the black-box and white-box models necessitate the quest for suitable alternatives ensuring a trade-off between accuracy and complexity when extended to automatic real-time diagnosis offering reliability. As a result, a novel and computationally feasible automated model for the detection and classification of diverse RDs from CXRs using a pixel-wise approach is introduced by adopting the following strategies:

1. Primally the framework engages the novel Exponential Pixelating Integral (EPI) for the effective delineation of low-contrast regions with noise resilience

2. Feature representation deploying locally invariant fractal geometries namely the Mandelbrot and Julia sets for effective discerning of prominent textural features

3. Model classification using the ensembled Multivariate Adaptive Regression Splines (MARS) ensuring robustness with generalization

Detailed evaluations of the developed system on 100,000 CXRs obtained from various publicly available large datasets such as Kaggle, RSNA, and NIH are conducted. The observations reveal that EPI merged with fractal geometry improved classification performance, demonstrating its potential as a valuable means for automated analysis of CXRs in a clinical setting. Accordingly, the presented content is compartmentalized as follows: EPI formulation and application of fractal geometry for extracting the essential features coupled with the MARS classifier is dealt with in Section 2. An exhaustive evaluation of the diagnostic model is done in Section 3 followed by the simplicity analysis in Section 4. The conclusion in Section 5 outlines the proposal’s intention with its suitability for real-time diagnostics.

## 2. Methodology

Diagnosing and appropriately treating RDs based on CXRs poses a challenge due to the inherent similarities among these disorders and the superimposed nature of anatomical structures. Moreover, another key challenge in the analysis of CXR images is the small differential intensity values in grayscale images, which can limit the focus on smaller scales. To overcome this limitation, a new image transformation technique called Exponential Pixelating Integral (EPI) is proposed. EPI computes the exponential moving average over a set of pixels bound by an overlapping local 3x3 kernel for contrast enhancement by sidelining the subtle intensity variations observed between adjacent pixels, hence facilitating the extraction of more meaningful features for classification. Subsequently, the resulting EPI image is polar transformed to facilitate a convenient means of representing and manipulating complex numbers and symmetrical forms in comparison to alternative coordinate systems. The polar transformed image is inputted to the Mandelbrot and Julia sets to produce a fractal representation covering complex shapes thereby systematically capturing object roughness which is extremely essential in Feature characterization. Especially, fractal geometry holds significant utility in the realm of medical diagnosis, particularly in lesion detection. The detection of malignant cells is facilitated by the contrasting growth patterns exhibited by healthy human blood vessel cells, which normally adhere to an organized fractal arrangement, in contrast to the aberrant growth exhibited by malignant cells. The utilization of fractal analysis facilitates the differentiation between normal tissue structures and indicators of potential abnormalities. Later, a wide range of statistical characteristics are extracted from the EPIFF that are then fed to the ensemble Multivariate Adaptive Regression Splines (MARS) for pairwise comparison between diverse RDs favoring robustness with generalization. The aforementioned process is depicted in Fig. 1 including in-depth discussions of the aforementioned objective, dealt with in the appropriate subsections

![](images/aa112a0c15e85045da165d6666de0fb109dc2bca9cc68c222c7fc5466d905622.jpg)  
Fig.1 Process diagram of the proposed EPIFF-MARS diagnostic framework

## 2.1 Exponential Pixelating Integral (EPI) model

Detection of lesions on a CXR image becomes challenging when there is relatively low contrast between the lesion and the surrounding tissue. To tackle this issue, several methodologies have employed contrast enhancement strategies aimed at increasing either the global or local contrast, to improve the the visual clarity of the image [9]. Though contrast-enhancing techniques decrease the occurrence of misinterpretation, they add noise and introduce distortions at the finer level. As a result, developing a simple solution that successfully addresses this contrast issue without increasing the noise effect remains a challenge. To address these problems, an innovative Exponential Pixelating Integral model is introduced for successfully highlighting the overall intensity with noise suppression. The modeled EPI smoothens the image with simultaneous noise suppression thereby reducing its visibility. Owing to the overlapping localized EPI nature, the resultant image will undergo scaling and standardization thereby enhancing the lower-intensity difference regions in a wider sense. This highlights the lesions, which are efficiently utilized to reference pixel-related fractal sets.

Accordingly, to get the EPI image the process commences with resizing the input image � into $m \times n$ dimensions $( I \epsilon R ^ { m \times n } )$ to maintain uniformity between extracted features of CXRs in the dataset. The resultant image is then parted into � × � non-overlapping subregions to extract the localized features.

Let � be the kernel that represents subregions with the size $( r = 3 )$ of the given input image as defined in Eq.1

$$
K _ { r \times r } = \left[ p _ { i - 1 , j - 1 } p _ { i - 1 , j } p _ { i - 1 , j + 1 } p _ { i , j - 1 } p _ { i , j } p _ { i , j + 1 } p _ { i + 1 , j - 1 } p _ { i + 1 , j } p _ { i + 1 , j + 1 } \right]\tag{1}
$$

Where $p _ { i j } -$ pixel value of the $K _ { r \times r }$ at image �� $R ^ { m \times n }$ . Let $p _ { i - 1 , j - 1 }$ and $p _ { i - 1 , j }$ be the two consecutive pixels, then the intensity difference between them is denoted as $\Delta p$ and is defined in $\mathrm { E q } . 2$

$$
\Delta p = \left| p _ { i - 1 , j - 1 } - p _ { i - 1 , j } \right|\tag{2}
$$

Herein, the instant neighbor is written as $p _ { i - 1 , j } = p _ { i - 1 , j - 1 } \pm \ \Delta p$ . To determine the local variance from the extracted kernel elements, initially, mean (�) is calculated between $p _ { i - 1 , j - 1 }$ and $p _ { i - 1 , j }$ using Eq. 3.

$$
\begin{array} { r } { \mu = { \frac { p _ { i - 1 , j - 1 } + p _ { i - 1 , j } } { 2 } } } \end{array}\tag{3}
$$

Substituting $p _ { i - 1 , j }$ in Eq. 3 results in the approximation of � as expressed in Eq.4

$$
\begin{array} { r } { \mu = p _ { i - 1 , j - 1 } \pm \frac { \Delta p } { 2 } } \end{array}\tag{4}
$$

Consequently, the variance between two consecutive pixels $p _ { i - 1 , j - 1 }$ and $p _ { i - 1 , j }$ is evaluated using Eq.5

$$
\scriptstyle \sigma ^ { 2 } = { \frac { \left( p _ { i - 1 , j - 1 } - \mu \right) ^ { 2 } + \left( p _ { i - 1 , j } - \mu \right) ^ { 2 } } { 2 } }\tag{5}
$$

Equation 6 is obtained by substituting Eq.4 in Eq.5

$$
\begin{array} { r } { \sigma ^ { 2 } = \frac { \left( p _ { i - 1 , j - 1 } - p _ { i - 1 , j - 1 } \mp \frac { \Delta p } { 2 } \right) ^ { 2 } + \left( p _ { i - 1 , j - 1 } \pm \Delta p - p _ { i - 1 , j - 1 } \mp \frac { \Delta p } { 2 } \right) ^ { 2 } } { 2 } } \end{array}\tag{6}
$$

Simplifying Eq.6 produces the Variance of $p _ { i - 1 , j - 1 }$ and $p _ { i - 1 , j }$ in Eq.7

$$
\begin{array} { r } { \sigma ^ { 2 } = \frac { \left( \left( \frac { \mp \Delta p } { 2 } \right) ^ { 2 } + \left( \pm \Delta p \mp \frac { \Delta p } { 2 } \right) ^ { 2 } \right) } { 2 } = \frac { \Delta p ^ { 2 } } { 4 } } \end{array}\tag{7}
$$

Equation 7 represents the variance between the two consequent pixels of the normal kernel function over an image. The acquired variance is insufficient to capture fine subtle texture details in low-contrast CXR images which are crucial for diagnostic purposes. Hence, the process introduces the Exponential Pixelating Integral Transform (EPI) for enhancing the granular image details. To achieve that the process transforms the original image into the EPI equivalent by applying the kernel $\hat { k }$ which evaluates the exponential function of the cumulative sum of neighboring pixel intensities, allowing a greater degree of emphasis on minute-scale features and is described in Eq. 8

$$
\begin{array} { r l } & { \big [ e ^ { \mathcal { P } _ { i - 1 , j - 1 } } e ^ { \mathcal { P } _ { i - 1 , j - 1 } + \mathcal { P } _ { i - 1 , j } } e ^ { \mathcal { P } _ { i - 1 - 1 , j - 1 } + \mathcal { P } _ { i - 1 , j } + \mathcal { P } _ { i - 1 , j } + \mathcal { P } _ { i - 1 , j + 1 } } e ^ { \mathcal { P } _ { i , j - 1 } + \mathcal { P } _ { i , j } } e ^ { \mathcal { P } _ { i , j - 1 } + \mathcal { P } _ { i , j } + \mathcal { P } _ { i , j + 1 } } e ^ { \mathcal { P } _ { i + 1 , j - 1 } } e ^ { \mathcal { P } _ { i - 1 , j } + \mathcal { P } _ { i + 1 , j } } e ^ { \mathcal { P } _ { i - 1 - 1 } + \mathcal { P } _ { i + 1 , j } } e ^ { \mathcal { P } _ { i + 1 , j } } \big ] } \\ & { \big [ e ^ { \mathcal { P } _ { i - 1 , j - 1 } } e ^ { \mathcal { P } _ { i - 1 , j - 1 } + \mathcal { P } _ { i - 1 , j } } e ^ { \mathcal { P } _ { i - 1 , j - 1 } + \mathcal { P } _ { i - 1 , j } + \mathcal { P } _ { i - 1 , j } + \mathcal { P } _ { i + 1 , j } + \mathcal { P } _ { i + 1 , j + 1 } } \big ] } \end{array}\tag{8}
$$

In Eq. 8, the cumulative sum captures the accumulation of intensity changes within a local area creating a more prominent peak in the edge region. In addition, the exponential function amplifies intensity variations in input images, causing small changes to result in significant variations in the transformed images. Thus, the exponential of the cumulative sum effectively visualizes the localized intensity variations. Later, the mean $( \mu _ { e } )$ of EPI transformed image is determined between consecutive pixels $e ^ { p _ { i - 1 , j - 1 } }$ and $e ^ { p _ { i - 1 , j - 1 } + p _ { i - 1 , j } }$ present in the kernel is evaluated using Eq.9

$$
\begin{array} { r } { \mu _ { e } = \frac { e ^ { p _ { i - 1 , j - 1 } } + e ^ { p _ { i - 1 , j - 1 } + p _ { i - 1 , j } } } { 2 } } \end{array}\tag{9}
$$

Upon substituting $p _ { i - 1 , j } = p _ { i - 1 , j - 1 } \pm \ \Delta p$ in Eq.9 yields Eq.10

$$
\begin{array} { r } { \mu _ { e } = \frac { e ^ { p _ { i - 1 , j - 1 } } + e ^ { 2 p _ { i - 1 , j - 1 } \pm \Delta p } } { 2 } } \end{array}\tag{10}
$$

Finally, the mean determined in Eq.11 is obtained by simplifying Eq.10

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

$$
\begin{array} { r } { \mu _ { e } = = = \frac { e ^ { p _ { i - 1 , j - 1 } } + \left( e ^ { p _ { i - 1 , j - 1 } } \right) ^ { 2 } e ^ { \pm \Delta p } } { 2 } } \end{array}\tag{11}
$$

The mean in Eq.11 is subtracted from the pixels $e ^ { p _ { i - 1 , j - 1 } }$ and $e ^ { p _ { i - 1 , j - 1 } + p _ { i - 1 , j } }$ and squared to produce the variance as in Eq.12

$$
\begin{array} { r } { { \sigma _ { e } } ^ { 2 } = \frac { \left( e ^ { p _ { i - 1 , j - 1 } } - \mu _ { e } \right) ^ { 2 } + \left( e ^ { p _ { i - 1 , j - 1 } + p _ { i - 1 , j } } - \mu _ { e } \right) ^ { 2 } } { 2 } } \end{array}\tag{12}
$$

Substituting Eq. 11 in Eq. 12 results in Eq.13

$$
\begin{array} { r } { \sigma ^ { 2 } = \frac { \left( e ^ { p _ { i - 1 , j - 1 } } - \frac { e ^ { p _ { i - 1 , j - 1 } } } { 2 } \left( e ^ { p _ { i - 1 , j - 1 } } \right) ^ { 2 } , e ^ { \pm \Delta p } \right) ^ { 2 } + \left( e ^ { p _ { i - 1 , j - 1 } + p _ { i - 1 , j - 1 } \pm \Delta p } - \frac { e ^ { p _ { i - 1 , j - 1 } } } { 2 } \left( e ^ { p _ { i - 1 , j - 1 } } \right) ^ { 2 } , e ^ { \pm \Delta p } \right) ^ { 2 } } { 2 } } \end{array}\tag{13}
$$

Eq.13 is simplified to yield Eq.14

$$
\begin{array} { r } { \sigma ^ { 2 } = \frac { \left( e ^ { p _ { i - 1 , j - 1 } } \right) ^ { 2 } } { 4 } \cdot ( e ^ { p _ { i - 1 , j - 1 } } . e ^ { \pm \Delta p } - 1 ) ^ { 2 } } \end{array}\tag{14}
$$

The $e ^ { \pm \Delta p }$ term in Eq.14 enhances the variance even for a small intensity difference, thereby, intensifying EPI to focus more on minuscule features. To further contrast stretching, the process commences with the normalization of individual pixels in the EPI-transformed kernel $\hat { k } ( i , j )$ given in Eq. 8 using Eq.15

$$
\hat { g } ( i , j ) = e ^ { \sum _ { k = 1 } ^ { j } { \mathbf { \sigma } } _ { p _ { i , k } } }\tag{15}
$$

To remove the offset caused by illumination across the image and normalize the spread of intensity across the kernel, each pixel value is normalized by its row-wise mean and row-wise variance of the kernel (�) as in Eq. 16

$$
\begin{array} { r } { \underline { { p } } _ { i , j } = \gamma \times \hat { g } ( i , j ) ~ w h e r e ~ \gamma = \left( \frac { p _ { i , j } } { ( \varepsilon + \mu _ { i } ) * \left( \varepsilon + \sigma _ { i } ^ { 2 } \right) } \right) } \end{array}\tag{16}
$$

Where � a small constant is assumed to be 0.001 and is introduced to avoid division by zero thereby, ensuring numerical stability. This normalization removes the outliers and improves the classification accuracy by increasing the discrimination between samples. After normalizing, to mitigate variations in illumination and highlight subtle features for accurate diagnostic interpretation, the final transformed intensity values are enhanced using contrast stretching as expressed in Eq.17

$$
\hat { p } _ { i , j } = \{ \underline { { p } } _ { i , j } + \sigma ~ i f \underline { { p } } _ { i , j } \geq \mu ~ \underline { { p } } _ { i , j } - \sigma ~ i f \underline { { p } } _ { i , j } < \mu\tag{17}
$$

The normalized exponential cumulative sum followed by contrast stretching of each pixel in the kernel highlights the possible lesions and regions of interest, resulting in enhanced discrimination. Finally, the kernel function yielded upon applying the aforesaid modifications is given in Eq.18

$$
\begin{array} { r } { K = [ \hat { p } ( i - 1 , j - 1 ) \hat { p } ( i - 1 , j ) \hat { p } ( i - 1 , j + 1 ) \hat { p } ( i , j - 1 ) \hat { p } ( i , j ) \hat { p } ( i , j + 1 ) \hat { p } ( i + 1 , j - 1 ) \hat { p } ( i + 1 , j ) \hat { p } ( i + 1 , j )  } \\ {   \qquad \mathrm 1 , j + 1 ) ] } \end{array}
$$

A snapshot of the EPI-transformed images is shown in Fig. 2 wherein effective lesion distinctions are noticed by emphasizing the relevant regions.

![](images/4f8c84c3588e39c72a4162daf07d0a869e80b9c973013606a55618178e01b0bf.jpg)

![](images/9b52c83c9d289f6e8f7a78a64a01b78de116ef732e4f3d8be02b3d012748a00f.jpg)

![](images/d7941df8ebdd10f990f662825cad9351c7b06847c79fe59b9d07e4cfce7b8402.jpg)

![](images/d928497f88e57758378c44a4cf5ff8ecf72118ab497126f6aee6863d3817dfd0.jpg)

![](images/5780cef535836095fcf402c86824f684e5fd6f153f0ccae161d9aab9e9a2be0d.jpg)

![](images/38ff48474c21df89674b0dae827db125c273783c59004b713cf1f6518d5d5086.jpg)

![](images/9a2f8ea1d2b153ea63d5801400a6b89a7e2a592c101331c0007b935be7bef323.jpg)

![](images/75926fdc6c120c3217c6647921cf56492aac7fb926a5779b19786384543b5fe9.jpg)

![](images/1ab308a552090bedbe6b7b36fa902fbb9030cf6c53ac6b4fac3fb6526c001e1d.jpg)

![](images/5ecee82b18ebbad0087abbb74e02c9661dd05b1d5e2eb906825c90d6705241a7.jpg)

![](images/3625ce4b3aef610f0ea68167ae561cce08676bc10d5c8d35066330a29fe37e3e.jpg)

![](images/363c5dd3070791eec0801439230589d5a302ece30f31bb52d8e8cc553f49be5d.jpg)

![](images/0e3cb8cb5b66f7f1bac9343b71b3aaaef8e3891d281280e6d989f78c928f8e4b.jpg)  
(a)

![](images/63a7167aafe0cfd3366710e4228644c7a3e651f386d63039bb44744f7de7f6f9.jpg)  
(b)

![](images/6f8a46fe105a9c018454fc2ccf024cac26b21db9dd8ef293a4662cc5ee6c7766.jpg)  
(c)  
Fig 2. Resultant EPI transformation on the CXR images. (a) The original X-ray images, (b) Normalized EPI transformed images, (c) Contrast Stretched images.

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

From Fig. 2 it is understood that the EPI transform effectively highlights the subtle variations in pixel intensity, making it easier to identify and analyze specific features within an image. Upon attainment of the EPI transformed image additionally two more feature matrices are extracted for distinguishing the structural abnormalities by considering the local affinity with invariance towards affine variations and is elaborated below.

## 2.2 Fractal Representation

Fractal geometry is a fascinating subject with substantial applications in a variety of medical imageprocessing techniques for analyzing the sophisticated and diverse features present in a wide range of biological systems[41]. A prominent benefit of these sets is in their self-similarity, whereby the identical fundamental pattern is replicated at varying scales. This characteristic makes them extremely suitable for representing intricate and non-uniform patterns posing challenges. Accordingly, the Mandelbrot and Julia sets are constructed in this approach straightforwardly using analytical models[42]. This facilitates better interpretation of complex structures, leading to more accurate diagnosis as the regular fractal growth pattern exhibited commonly in human organ tissues aids in the detection of malignant cells that grow erratically. Therefore, engaging fractal analysis eases the structural distinction between normal and abnormal structures, enhancing diagnosis accuracy[43]. To determine the fractal sets, the EPI image is initially converted into polar coordinates owing to its simplicity in representation and manipulation of symmetrical shapes. In the polar coordinate system, each point on a plane is characterized by its distance from a reference point (�) and its angular displacement (�) from a reference direction. The utilization of the (�)and (�) parameters enhance the representation of texture variation by establishing a direct correlation between image features and the directionality of the texture [44], and one such outcome is demonstrated in Fig. 3 (a).

## 2.2.1 Mandelbrot Image Representation

The choice of Mandelbrot and Julia sets is attributed to identifying similar data patterns and significant attributes within images. In addition, their simplicity in generation yet powerful in representation, make them a popular tool in the field of fractal geometry. Mathematically, a Mandelbrot set represents a set of complex numbers as in Eq.19

$$
f ( z ) = z ^ { 2 } + c\tag{19}
$$

Where $z -$ is the independent variable and initially assumed to be zero, � is a complex number. The function is iterated a fixed number of times, and if the absolute value of the result remains less than a certain threshold, the point � is considered to be in the set.

To generate a mandelbrot set, the process first determines the standard deviation of the polar image along the rows $( \sigma _ { i } )$ and columns $\left( \sigma _ { j } \right)$ that are then composed as the real and imaginary components of the complex additive [45] as defined in Eq. 20

$$
c = \sigma _ { i } + i \sigma _ { j }\tag{20}
$$

Where, $\sigma _ { i }$ ��� $\sigma _ { j } -$ bounded in the range $i \in [ 0 , m ]$ ��� $j \in [ 0 , n ] , m$ , �-be the size of the polar image.

Finally, the Mandelbrot set is derived through the iterative process by substituting Eq.20 into Eq.19, resulting in Eq. 21

$$
f _ { c } ( z ) = | z _ { n + 1 } | = z _ { n } ^ { 2 } + \left( \sigma _ { i } + i \sigma _ { j } \right)\tag{21}
$$

$\left. z _ { n } \right.$ undergoes iterations (�) until surpassing a divergence threshold and is denoted as $| z _ { n + 1 } | \leq 2$ . If the absolute value of the result remains less than 2, then only the point � is considered to be in the set.

Accordingly, the mandelbrot set is constructed using Eq. 22

$$
{ \cal M } = \{ c \in C : f ^ { 0 } ( c ) \in i ; f ^ { t } ( c )  \infty a s t  \infty \}\tag{22}
$$

where $C - { \mathrm { s e t } }$ of complex numbers; $f ^ { t } ( c ) -$ iterates of the function. This configuration connects the Real and Imaginary axes, with the complex plane represented by black space. There are two sections to the structure shown in $\mathrm { F i g } . 3 ( \mathrm { b } )$ the main bulb with the main cardioid. These elements tend to compress, expand, rotate, or translate due to the nature of the standard deviation along both axes. The variance-Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine (CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

dependent Mandelbrot set generates definitive patterns subject to the class. The boundary, and area of spread clearly define the Region of Interest for feature extraction in the Mandelbrot set represented in Fig. 3(b).

## 2.2.2 Julia Set Image Representation

Technically, the Julia Set structure has strands of lines and infinitely many pieces. Such a structure is termed Fatou Dust [46]. In this set, the complex additive is a constant while the z is the point of interest. This set exhibits similarities to the Mandelbrot set, with the key distinction lying in their respective plotting methodologies. While the Mandelbrot set encompasses the entire range of possible values for c, the Julia set focuses on the visualization of a specific value of complex additive (�) and is defined in Eq.23

$$
f ( z ) = z ^ { 2 } + c\tag{23}
$$

The complex additive � is constant and bounded between [0,1]. While, the � value is dependent on the standard deviation of the polar transformed image, across row and column as the real and imaginary parts respectively as described in Eq.24

$$
z _ { n } = \sigma _ { i } + i \sigma _ { j }\tag{24}
$$

Finally, the Julia Set is derived through the substitution of Eq.24 into Eq.23, resulting in Eq.25

$$
f _ { c } ( z ) = | z _ { n + 1 } | = z _ { n } ^ { 2 } + c\tag{25}
$$

The initial value of � is assumed zero (denoted in complex form as $0 + 0 i )$ to avoid divergence to infinity. Similar to the mandelbrot set, the process iterates until it reaches the threshold value of $\vert z _ { n + 1 } \vert \leq 2$

Finally, as per the basin of attraction to infinity function, the Julia Set is constructed using Eq. 26

$$
J = \{ z \in { \cal C } : f ^ { n } ( z ) \to \infty a s n \to \infty \}\tag{26}
$$

where � is the set of complex numbers and $f ^ { n } ( z )$ are the iterates of the function.

The Julia set generates multiple clusters with varying intensities and definitive boundaries. Feature extraction concerning the cluster intensities and their variation are used to analyze the mapped source image depicted in Fig. 3(c). In comparison with other image transforms and representations, these fractal sets extract multiple features and are highly tunable enabling effective representation of minuscule image details.

![](images/478da8aee2f07a5fad2378db0f8530c2de7eff68a4e6155c1ed6d91474acb76b.jpg)  
(a)

![](images/118e8175a364f3bf0233566d19820381c4b0970240a4e212b3a79270342ef402.jpg)  
(b)

![](images/1a873d9ad4174f25b63b4f8074aca7c9f220780bb3c470420434fe765b19518e.jpg)  
(c)  
Fig.3 Image Representation (a) Polar Image (b) Mandelbrot set, and (c) Julia set Representation  
Consequently, to precisely classify various RDs, EPIFF extracted the multiple statistical features from the aforementioned image representation namely, the Polar transformed image, and both the Julia & Mandelbrot Image representation which are detailed with notation presented in Table 1. These features attributed to valuable information for different aspects of image analysis, including structure, intensity distribution, and statistical properties.

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

Table 1 Features extracted from Polar Transformation, Mandelbrot set, and Julia set Representation.
<table><tr><td rowspan="2">Features</td><td colspan="3">Symbols</td><td rowspan="2">Description</td></tr><tr><td>Polar</td><td>Mandelbro t</td><td>Julia</td></tr><tr><td>Binarized Sum</td><td> $p _ { b i n }$ </td><td> $m _ { b }$ </td><td> $j _ { b }$ </td><td>Provides a measure of the total intensity in a binary image</td></tr><tr><td>Centroid for x- axis</td><td> $p c _ { x }$ </td><td> $m c _ { x }$ </td><td> $j c _ { x }$ </td><td>Computes the Centroid:x of the corresponding image contour for localization</td></tr><tr><td>Centroid for y- axis</td><td> $p c _ { y }$ </td><td> $m c _ { y }$ </td><td> $j c _ { y }$ </td><td>Computes the Centroid:y of the corresponding image contour for localization</td></tr><tr><td>Euler Number</td><td> $p _ { e u l }$ </td><td> $m _ { e u l }$ </td><td> $j _ { e u l }$ </td><td>Calculate the Euler number of the corresponding image contour for shape characterization</td></tr><tr><td>Mean</td><td> $p _ { m e }$ </td><td> $m _ { m e }$ </td><td> $j _ { m e }$ </td><td>Average intensity of pixels in an image</td></tr><tr><td>Variance</td><td> $p _ { v a r }$ </td><td> $m _ { v a r }$ </td><td> $j _ { v a r }$ </td><td>Quantifies the spread or dispersion of pixel intensities to provide textural information</td></tr><tr><td>Dominant Pixel Frequency</td><td> $p _ { d f }$ </td><td> $m _ { d f }$ </td><td> $j _ { d f }$ </td><td>Highest occurring pixel intensity to diagnose predominant regions</td></tr><tr><td>Dominant Pixel</td><td> $p _ { d p }$ </td><td> $m _ { d p }$ </td><td> $j _ { d p }$ </td><td>Highest occurring pixel intensity to detect prominent textures</td></tr><tr><td>Median</td><td> $p _ { m e d }$ </td><td> $m _ { m e d }$ </td><td> $j _ { m e d }$ </td><td>The median of the pixel intensities</td></tr><tr><td>Median Frequency</td><td> $p _ { m f }$ </td><td> $m _ { m f }$ </td><td> $j _ { m f }$ </td><td>Frequency of Median of pixel intensities to measure the distribution of median pixels</td></tr><tr><td>Kurtosis</td><td> $p _ { k t }$ </td><td> $m _ { k t }$ </td><td> $j _ { k t }$ </td><td>Quantify the peakedness of distribution to characterize the shape of intensities</td></tr><tr><td>Skew</td><td> $p _ { s k }$ </td><td> $m _ { s k }$ </td><td> $j _ { s k }$ </td><td>Measure the asymmetry of the distribution reveals the distribution of pixel intensities</td></tr><tr><td>Standard Error of the Mean</td><td> $p _ { s e m }$ </td><td> $m _ { s e m }$ </td><td> $j _ { s e m }$ </td><td>SEM measures the average deviation of sample means from the true population mean.</td></tr><tr><td>Coefficient of Variation</td><td> $p _ { c v }$ </td><td> $m _ { c v }$ </td><td> $j _ { c v }$ </td><td>Computes the coefficient of variation, which is the ratio of the standard deviation to the mean of the data.</td></tr></table>

Further, the process extracted a few more essential features that are used to estimate the characteristics of the proposed EPI-transformed images in comparison with normal CXRs. These features are valuable for capturing subtle yet significant alterations in pixel values across the images aiding in diagnosing and understanding the disease's effect which are detailed in Table 2.

Table 2 Features extracted from EPI transformed images
<table><tr><td>Features</td><td>Symbol Description</td></tr></table>

<table><tr><td>Peak Signal- to-Noise Ratio</td><td>psnr</td><td>Quantifies the disease fidelity, while higher values indicate that the disease-induced changes are minimal, ensuring that relevant anatomical features are preserved.</td></tr><tr><td>Spearman Correlation</td><td>sp</td><td>Measures the degree of association. Also, identify the consistent changes in pixel values across the images due to disease-induced alterations. If a high correlation is observed, it suggests that certain features are consistently affected by the disease.</td></tr><tr><td>Mean squared error</td><td>mse</td><td>Estimate the extent of disease-related changes, aiding in identifying regions with significant alterations. Larger MSE values point to areas of pronounced differences, which could indicate disease presence with severity.</td></tr><tr><td>Structural Similarity</td><td>ssim</td><td>Quantifies image quality degradation. SSIM considers luminance, contrast, and structure, aligning with radiologists&#x27; interpretation of images. A lower SSIM suggests that disease-related structural changes impacting image similarity</td></tr><tr><td>Normalized MSE</td><td> $n _ { m s e }$ </td><td>Computes MSE at different scales to enhance the interpretability</td></tr><tr><td>Information of Case Normal</td><td>icn</td><td>Evaluates the predictive ability of the model when the case information is available</td></tr><tr><td>Information of Normal Case</td><td>inc</td><td>Computes mutual information of normal, given a Case, offering insights into the significance of the normal class.</td></tr><tr><td>Information Score</td><td> $i _ { s c }$ </td><td>Clustered mutual information between two images. It&#x27;s important to assess the degree to which disease alters relevant image features. A high Information Score could highlight regions of high diagnostic relevance.</td></tr><tr><td>Euclidean Distance</td><td>eu similarity</td><td>Computes Euclidean array between two arrays to measure the</td></tr><tr><td>Energy Distance</td><td> $e _ { d i s t }$ </td><td>Quantify the dissimilarity between images based on pixel values and distributional differences. They identify the global shifts caused by diseases. Higher distances indicate widespread alterations, offering insights into disease progression and impact.</td></tr></table>

Overall 52 essential features listed in Tables 1-2 are extracted from the EPI, Polar, and dual fractals accumulated to yield the Feature Vector (FV) of a given CXR. Though the feature length seems small, which is extracted from the EPI, polar transformed images followed by Mandelbrot and Julia set representation. EPI transformation enhances the low-contrast regions while polar conversion significantly highlights the curved surfaces. These transformations are further exploited by the inherent self-similarity nature of fractals to capture the structural distinction between normal and abnormal regions, resulting in compact and highly discriminating features that escalate the diagnostic accuracy with reduced complexity. Further, this concise set of features facilitates rapid prediction which is extremely crucial for real-time applications. Likewise, the FVs’ of dataset constituents are extracted and formulated into a feature database that is to be classified by considering the diverse criteria concerned with RDs. To cover the wider aspects of various RDs a multivariate regression model adapting to intricate data changes is adopted and dealt with in the below section.

## 2.3 Classification using MARS Ensemble Learning

Generalized Linear and Additive Models assume that coefficients associated with predictor variables remain constant across all predictor values. However, MARS discards assumptions and is nonparametric[47,48]. It uses a combination of basis functions, such as polynomials or splines, to approximate the relationship between the input and output variables. This makes it well suited for complex and highly varied data such as medical images, as it can adapt to the underlying patterns in the data without making strong assumptions about the shape of the relationship. Moreover, the MARS offers more interpretability than deep learning models by relatively weighing the different input features aiding prediction. Deep learning models that represent black boxes naturally make it difficult to understand the decided predictions and are more complex.

MARS model is mathematically defined in Eq. 27

$$
y = b _ { 0 } + b _ { 1 } x _ { 1 } + b _ { 2 } x _ { 2 } + \cdots + b _ { n } x _ { n } + \sum _ { i = 0 } ^ { n } \quad \alpha _ { i } h _ { i } ( z _ { i } )\tag{27}
$$

where � −output variable, �<sub>�</sub> −coefficient of linear terms, $x _ { i }$ −input variable, $\alpha _ { i }$ −coefficient of nonlinear terms, $h _ { i } ( z _ { i } ) -$ Basis functions.

The classification model is built in two phases by the MARS. In the forward phase, it systematically incorporates Basis Functions (BF) into its structure to minimize the mean squared residual error. This entails evaluating each variable in the dataset as a potential basis function until a negligible change in the residual error is observed. The BF takes the largest value from one of two options: 0 or the outcome of the environmental variable – knot value and is defined in Eq.(28)

$$
B F = m a x \ ( 0 , e n v \ v a r - \ k n o t )\tag{28}
$$

Where, ��� ��� −individual feature elements in FVs. By raising the value of the basis function, MARS produces a persistent estimate of the target function to an arbitrary order of derivatives. A distinct linear regression with its slope is constructed for each group while Knots link them automatically in an optimal manner. In contrast, the backward phase involves pruning non-participating basis functions that do not contribute to improving model performance. By eliminating irrelevant features through backward elimination, MARS reduces model complexity. Unlike traditional models that might use a generic set of features across all categories, MARS analyzes the nature of the input data and identifies the most critical features specific to each category. This streamlined model requires less computational power to train and employ, making it more efficient for handling large datasets. While feature selection improves scalability, it's crucial to consider the impact on accuracy. MARS addresses this through a rigorous pruning process wherein features having minimal impact on its performance are discarded based on the Generalized Cross-Validation and preserve the most essential features for accurate predictions.

Accordingly, to classify the diverse RDs, a pair-wise comparison of each disease (�<sub>�</sub>) against every other namely Edema, COVID-19, Effusion, Emphysema, and Pneumonia is performed. As the diseases were categorized into five, hence, the class-wise comparisons led to 5 thereby resulting in 25 (5x5) possible comparisons. Accordingly, 25 MARS models are trained on 80% of the data with each model specialized in differentiating a specific disease pair and identifying key features in the form of basis function and its coefficients during training. The remaining 20% of the data validates these models. When classifying a new data point, the ensemble learning mechanism comes into play wherein the remaining four models relevant to the disease (one for each comparison with other diseases) make predictions. These predictions are then combined by soft voting technique to determine the most likely disease. This approach leverages the strengths of multiple, specialized models, focusing on specific disease differentiation.

Accordingly, EPIFF-MARS formulates basis functions $( A _ { i j } \in R ^ { F \times k } )$ along with its coefficient matrix $( C _ { i j } \in R ^ { F \times \bar { k } } )$ for each category separately. where � −number of participant features $F ; k { \mathrm { ~ - t o t a l } }$ number of categories; �,� − denotes pairwise comparison between disease categories bounded between $[ 0 , k -$ 1];

For instance, to classify Edema, the model initiates training by tuning the basis function and corresponding coefficient matrix from the EPIFF features related to Edema with the remaining classes as, Edema vs. edema $( A _ { 0 , 0 } , C _ { 0 , 0 } ) ;$ Edema $\mathrm { V s } \mathrm { C O V I D - } 1 9 ( A _ { 0 , 1 } , C _ { 0 , 1 } ) ,$ ; Edema vs. effusion $( A _ { 0 , 2 } , C _ { 0 , 2 } ) .$ ; Edema

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

vs. emphysema $( A _ { 0 , 3 } , C _ { 0 , 3 } ) ;$ ; and Edema Vs Pneumonia $( A _ { 0 , 4 } , C _ { 0 , 4 } )$ resulting in five distinct $A _ { i j }$ with $C _ { i j }$ to build the MARS model related to Edema assuming $i \ = \ 0$ as in Eq. (29)

$$
D _ { 0 } ( E d e m a ) = \left[ C _ { 0 , 0 } ^ { T } \cdot A _ { 0 , 0 } + C _ { 0 , 1 } ^ { T } \cdot A _ { 0 , 1 } + C _ { 0 , 2 } ^ { T } \cdot A _ { 0 , 2 } + C _ { 0 , 3 } ^ { T } \cdot A _ { 0 , 3 } + C _ { 0 , 4 } ^ { T } \cdot A _ { 0 , 4 } \right]\tag{29}
$$

Eq. 29 is generalized as in Eq. 30 for training the MARS model.

$$
\begin{array} { r l } { D _ { i } = \sum _ { j = 0 } ^ { k - 1 } } & { { } \left( C _ { i j } \right) ^ { T } \cdot A _ { i j } \big | _ { i } } \end{array}\tag{30}
$$

$D _ { i }$ −represents Disease belonging to $i ^ { t h }$ category.

Similarly, upon adopting the aforesaid approach the overall MARS classification model is constructed in Eq. 31

$$
\begin{array} { r } { \left[ D _ { 0 } ( E d e m a ) D _ { 1 } ( C o v i d - 1 9 ) D _ { 2 } ( E f f u s i o n ) D _ { 3 } ( E m p h y s e m a ) D _ { 4 } ( P n e u m o n i a ) \right] = \left[ C _ { 0 ; 0 } ^ { T } \cdot A _ { 0 ; 0 } + } \\ { C _ { 0 ; 1 } ^ { T } \cdot A _ { 0 , 1 } + C _ { 0 ; 2 } ^ { T } \cdot A _ { 0 , 2 } + C _ { 0 ; 3 } ^ { T } \cdot A _ { 0 ; 3 } + C _ { 0 ; 4 } ^ { T } \cdot A _ { 0 ; 4 } C _ { 1 , 0 } ^ { T } \cdot A _ { 1 , 0 } + C _ { 1 ; 1 } ^ { T } \cdot A _ { 1 , 1 } + C _ { 1 ; 2 } ^ { T } \cdot A _ { 1 , 2 } + C _ { 1 ; 3 } ^ { T } \cdot A _ { 1 , 3 } + } \\ { C _ { 1 , 4 } ^ { T } \cdot A _ { 1 , 4 } C _ { 2 ; 0 } ^ { T } \cdot A _ { 2 , 0 } + C _ { 2 ; 1 } ^ { T } \cdot A _ { 2 , 1 } + C _ { 2 ; 2 } ^ { T } \cdot A _ { 2 , 2 } + C _ { 2 ; 3 } ^ { T } \cdot A _ { 2 , 3 } + C _ { 2 , 4 } ^ { T } \cdot A _ { 2 , 4 } C _ { 3 ; 0 } ^ { T } \cdot A _ { 3 , 0 } + C _ { 3 ; 1 } ^ { T } \cdot A _ { 3 , 1 } + C _ { 3 ; 2 } ^ { T } \cdot A _ { 3 , 2 } + } \\ { C _ { 3 ; 3 } ^ { T } \cdot A _ { 3 , 3 } + C _ { 3 ; 4 } ^ { T } \cdot A _ { 3 , 4 } C _ { 4 ; 0 } ^ { T } \cdot A _ { 4 , 0 } + C _ { 4 ; 1 } ^ { T } \cdot A _ { 4 , 1 } + C _ { 4 ; 2 } ^ { T } \cdot A _ { 4 , 2 } + C _ { 4 ; 3 } ^ { T } \cdot A _ { 4 , 3 } + C _ { 4 ; 4 } ^ { T } \cdot A _ { 4 ; 4 } \right] ( 3 ! ) } \end{array}
$$

This model is mathematically shortened in Eq. 32

$$
\begin{array} { r l r } & { \left[ D _ { 0 } \ D _ { 1 } \ D _ { 2 } \ D _ { 3 } \ D _ { 4 } \right] = \left[ \sum _ { j = 0 } ^ { k - 1 } } & { \left( C _ { i j } \right) ^ { T } \cdot A _ { i j } | _ { i = 0 } \ \sum _ { j = 0 } ^ { k - 1 } } & { \left( C _ { i j } \right) ^ { T } \cdot A _ { i j } | _ { i = 1 } \ \sum _ { j = 0 } ^ { k - 1 } \ } & { \left( C _ { i j } \right) ^ { T } \cdot } \\ & { A _ { i j } | _ { i = 2 } \ \sum _ { j = 0 } ^ { k - 1 } } & { \left( C _ { i j } \right) ^ { T } \cdot A _ { i j } | _ { i = 3 } \ \sum _ { j = 0 } ^ { k - 1 } \mathrm { \quad } \left( C _ { i j } \right) ^ { T } \cdot A _ { i j } | _ { i = 4 } \ \right] } & { ( 3 2 ) } \end{array}
$$

Based on the aforesaid construction, the pair-wise comparison of the MARS models for overall disease prediction with probability score is presented in Table 3.

Table 3. Pairwise MARS models for overall disease prediction
<table><tr><td>Disease</td><td>Edema</td><td>Covid</td><td>Effusion</td><td>Emphysem a</td><td>Pneumonia</td></tr><tr><td>Edema  $\left\{ > 0 . 5 \quad i f G _ { T } \right.$  = Edema  $< 0 . 5$  otherwise</td><td>1</td><td> $C _ { 0 , 1 } ^ { T } \cdot A _ { 0 , 1 }$ </td><td> $C _ { 0 , 2 } ^ { T } \cdot A _ { 0 , 2 }$ </td><td> $C _ { 0 , 3 } ^ { T } \cdot A _ { 0 , 3 }$ </td><td> $C _ { 0 , 4 } ^ { T } \cdot A _ { 0 , 4 }$ </td></tr><tr><td>COVID  $\left\{ > 0 . 5 \quad i f G _ { T } \right.$  = Covid  $< 0 . 5$  otherwise</td><td> $C _ { 1 , 0 } ^ { T } \cdot A _ { 1 , 0 }$ </td><td>1</td><td> $C _ { 1 , 2 } ^ { T } \cdot A _ { 1 , 2 }$ </td><td> $C _ { 1 , 3 } ^ { T } \cdot A _ { 1 , 3 }$ </td><td> $C _ { 1 , 4 } ^ { T } \cdot A _ { 1 , 4 }$ </td></tr><tr><td>Effusion  $\left\{ > 0 . 5 \quad i f G _ { T } \right.$  = Effusion  $< 0 . 5$  otherwise</td><td> $C _ { 2 , 0 } ^ { T } \cdot A _ { 2 , 0 }$ </td><td> $C _ { 2 , 1 } ^ { T } \cdot A _ { 2 , 1 }$ </td><td>1</td><td> $C _ { 2 , 3 } ^ { T } \cdot A _ { 2 , 3 }$ </td><td> $C _ { 2 , 4 } ^ { T } \cdot A _ { 2 , 4 }$ </td></tr><tr><td>Emphysema  $\left\{ > 0 . 5 \quad i f G _ { T } \right.$  = Emphysema  $< 0 . 5$  otherwise</td><td> $C _ { 3 , 0 } ^ { T } \cdot A _ { 3 , 0 }$ </td><td> $C _ { 3 , 1 } ^ { T } \cdot A _ { 3 , 1 }$ </td><td> $C _ { 3 , 2 } ^ { T } \cdot A _ { 3 , 2 }$ </td><td>1</td><td> $C _ { 3 , 4 } ^ { T } \cdot A _ { 3 , 4 }$ </td></tr><tr><td>Pneumonia</td><td> $C _ { 4 , 0 } ^ { T } \cdot A _ { 4 , 0 }$ </td><td> $C _ { 4 , 1 } ^ { T } \cdot A _ { 4 , 1 }$ </td><td> $C _ { 4 , 2 } ^ { T } \cdot A _ { 4 , 2 }$ </td><td> $C _ { 4 , 3 } ^ { T } \cdot A _ { 4 , 3 }$ </td><td>1</td></tr></table>

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

```markdown
{> 0.5 �� �<sub>�</sub>
= ���������
< 0.5 ��ℎ������
```

Finally, the positive probability of a disease $d _ { i }$ is determined by engaging the SoftMax function as defined in Eq.(33)

$$
\begin{array} { r } { P ( D _ { i } ) = \frac { e ^ { D _ { i } } } { \sum _ { j = 0 } ^ { k - 1 } \mathrm { ~ } e ^ { D _ { j } } } } \end{array}\tag{33}
$$

�(�<sub>�</sub>) produces a set of positive probabilities for each class associated with a given image by considering the maximum likelihood of occurrence to successfully predict the various RDs.

## 3. Performance Analysis

## 3.1 Datasets Description and Evaluation Metrics

To investigate the classification efficiency of the introduced EPIFF-MARS model, rigorous investigations were performed on Chest X-ray 14 and COVID-19 datasets. The Chest X-ray 14 dataset [49], is composed of 112,120 frontal-view CXRs acquired from 32,717 patients of the National Institutes of Health Clinical Center (NIH). The images are 8-bit gray-scale PNG files, sized 1024×1024 pixels, with an average of 3-4 images per patient due to follow-up scans. These images were initially associated with 8 thoracic pathologies and were later expanded to include 14 chest diseases[50]. The dataset includes metadata detailing pathologies, follow-up images, patient demographics, and radiography views. Among the 112,120 chest X-rays, 60,412 show no pathology, while 51,708 depict one or more pathologies. This dataset is valuable for chest radiography research and medical applications.

To demonstrate the EPIFF-MARS effectiveness in detecting COVID-19 abnormalities from CXRs, the experiment was conducted on the publically available COVID-19 Radiography dataset[51]. This dataset consists of 21,165 CXRs and their corresponding masks. Amongst them, 3,616 were imaged from COVID-19 patients with 6,012 showing non-COVID lung infections, 1,345 associated with pneumonia, and the remaining 10,912 were normal lung X-rays. The images are in PNG format with a dimension of 256 × 256 pixels. This dataset was created by the international collaboration of researchers from Qatar University, and the University of Dhaka, along with medical professionals from Pakistan and Malaysia.

EPIFF efficacy is assessed using classification accuracy (CA) and F1 score metrics. Furthermore, the receiver operating characteristic (ROC) analysis in terms of the True Positive Rate (TPR) and the False Positive Rate (FPR) is done to demonstrate the model’s superiority in the classification task. CA measures the accuracy of the predicted samples over the total number of samples as modeled in Eq. 34

$$
\begin{array} { r } { C A = \frac { T P + T N } { T P + F P + T N + F N } } \end{array}\tag{34}
$$

Where �� −True Positive; �� −Ture Negative; �� −False Positive; and �� −False Negative.

Likewise, F1-Score accounted in terms of Precision (����) and Recall (���) metrics. Herein, ���� expressed in Eq. 35 quantifies the proportionality of correctly predicted samples over the number of predicted positive samples

$$
\begin{array} { r } { P r e c = \frac { T P } { F P + T P } } \end{array}\tag{35}
$$

��� (Sensitivity / TPR) measures the proportionality of the correctly predicted samples over the actual number of positive samples and is given in Eq. 36

$$
\begin{array} { r } { R c l = \frac { T P } { F N + T P } } \end{array}\tag{36}
$$

Accordingly, the F1 score is presented in Eq. 37 and evaluates the model performance by successfully balancing the assessment across both positive and negative categories.

$$
\begin{array} { r } { F 1 s c o r e = 2 \times \frac { P r e c \times R c l } { P r e c \times R c l } } \end{array}\tag{37}
$$

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

The specificity investigates the model’s ability to correctly reject the healthy samples and is presented in Eq. 38

$$
\mathrm { S p e c i f i c i t y } = { \frac { T N } { T N + F P } }\tag{38}
$$

The FPR is (1-specificity) investigates the model’s reliability and accuracy by estimating the likelihood of incorrectly categorizing a negative sample as positive using Eq. 39

$$
\begin{array} { r } { F P R = \frac { F P } { T N + F P } } \end{array}\tag{39}
$$

To assess the degree of alignment between the model and the observed data, two additional metrics, namely the Mean Squared Error (MSE) and the $R ^ { 2 }$ scores are evaluated. The Mean squared error (MSE) is a common metric for evaluating the overall model accuracy by estimating the square difference between the actual and the model’s predicted values as in Eq. 40

$$
\begin{array} { r l } { M S E = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } } & { { } ( Y _ { i } - \widehat { Y } _ { i } ) ^ { 2 } } \end{array}\tag{40}
$$

Where � −number of samples; $Y _ { i }$ −observed values; $\hat { Y } _ { i }$ −predicted values.

Similarly, $R ^ { 2 }$ defined in Eq. 41 assesses the level of correlation between a model and the empirical data by estimating the difference between the squared sum residual with the total squared sum.

$$
\begin{array} { r l } { R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { n } } { \sum _ { i = 1 } ^ { n } } } & { { } ( Y _ { i } { - } \hat { Y } _ { i } ) ^ { 2 } } \end{array}\tag{41}
$$

Where $Y _ { i }$ −observed values; $\hat { Y } _ { i }$ −predicted values; and $\underline { { Y } }$ −mean value.

## 3.2 Results and Discussion

To demonstrate the EPIFF-MARS accuracy in RD classification, an experiment is conducted on the NIH Chest X-ray 14 dataset. Accordingly, the individual confusion matrices for each class are formulated to facilitate a comparison between the model's predictions and actual outcomes, providing a detailed breakdown of its performance in terms of TP, TN, FP, and FN. The structure of the confusion matrix aligns true classes as row elements and predicted classes as columns, delivering a comprehensive overview of the model's effectiveness and is given in Fig. 4.

![](images/a14669138ce0d724591b36768b0c58cb2a7b60908855f702cba5fab4dd5736ef.jpg)

![](images/5beb9c84e4e689f65d5640fcbcf7747056a3d22e3ab85077002c9ea8aafb8d54.jpg)

![](images/c98be4366e479a67ad9281e8e8ff078d28a5d274927c13209fdf7b9e58770f2b.jpg)

![](images/567a4053067b3ee1829a1a682bfeb6561c6374c3bb4f997a582ce9a29fa2b661.jpg)

![](images/674dd556b64480e338be230fbfc2f51d079ef5ddfc9f9df08aaa6ce1547eecac.jpg)  
Fig. 4 Confusion matrix of diverse RDs

The confusion matrix in Fig. 4 indicates high TP and TN rates which are greater than 98% and 97% respectively showcasing its potential to differentiate between healthy and diseased patients across multiple categories, including Emphysema, COVID-19, Edema, Pneumonia, and Effusion. Further, achieving higher accuracy with imbalanced data is challenging due to the bias towards the majority class leads to increased FP and FN for the minority class. However, despite the imbalance, the EPIFF-MARS model achieved low rates of FP and FN which are lesser than 1% and 3% respectively demonstrating the extremely minimal misclassifications of EPIFF-MARS. The impact of TP, TN, FP, and FN is furthered by categorical evaluation of Accuracy, precision, sensitivity (Recall), specificity, and F1 Score of EPIFF-MARS in Table 4.

Table 4. Qualitative Analysis of the EPIFF-MARS
<table><tr><td>Class Name</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>Specificity</td><td>F1 Score</td></tr><tr><td>Edema</td><td>0.9946</td><td>0.9935</td><td>0.9684</td><td>0.9989</td><td>0.9808</td></tr><tr><td>Covid - 19</td><td>0.9910</td><td>0.9856</td><td>0.9789</td><td>0.9945</td><td>0.9822</td></tr><tr><td>Effusion</td><td>0.9847</td><td>0.9779</td><td>0.9752</td><td>0.9893</td><td>0.9765</td></tr><tr><td>Emphysem a</td><td>0.9919</td><td>0.9784</td><td>0.9826</td><td>0.9943</td><td>0.9805</td></tr><tr><td>Pneumonia</td><td>0.9946</td><td>0.9810</td><td>0.9810</td><td>0.9968</td><td>0.9810</td></tr></table>

Upon examining Table 4, it is apparent that the EPIFF-MARS consistently exhibits superior outcomes, surpassing 97% in all metrics eventually demonstrating its effectiveness. Although, the 99% accuracy of EPIFF-MARS exhibits superiority perhaps misleads when dealing with imbalanced datasets, hence, the sensitivity(Recall) and specificity metrics are evaluated class-wise to register its reliability. The averaged percentage scores of 97.72 and 99.48 of sensitivity and specificity respectively in Table 4 are owed to the extremely low FP and FN rates as witnessed in Fig. 4. This achievement is due to the adaptation of a pairwise analysis of EPIFF-MARS for disease categorization which was particularly

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

beneficial for dealing with imbalanced data that automatically qualifies the highly essential features representing each category despite the availability of significantly fewer samples. By focusing on these qualified features, EPIFF-MARS effectively handles the data imbalance. This illustrates the strong differentiating ability of EPIFF-MARS between healthy and diseased patients and contributes to improved patient care by minimizing misdiagnoses and ensuring timely treatment. Furthermore, achieving the averaged F1 score greater than 98% is particularly challenging and requires an intricate balance between the Precision and Recall offered by EPIFF-MARS, highlighting its potency in classification. These achievements are attributed to the highly distinctive fractal features that are effectively classified by the pairwise MARS classifier, which focuses on differentiating between specific pairs of classes by learning more nuanced decision boundaries. This potentially leads to improved accuracy compared to "one-vs-all", especially for complex medical datasets with similar classes. Also, EPIFF-MARS merits in terms of computational efficiency, improved accuracy, and interpretability warrant its extension to real-time implementation.

In addition, the relative analysis of EPIFF-MARS with existing state-of-the-art schemes (SOTA) is presented in Table 5.

Table 5 Performance analysis of the proposed model with existing SOTA Schemes
<table><tr><td>Methods</td><td>AUC</td><td>Accurac y</td><td>Precisio n</td><td>Recall</td><td>Specificit y</td><td>F1-Score</td></tr><tr><td>EPIFF-MARS</td><td>0.9856</td><td>0.9912</td><td>0.9811</td><td>0.9764</td><td>0.9947</td><td>0.9787</td></tr><tr><td>Red Deer (2023)[52]</td><td></td><td>98.65</td><td>97.56</td><td>96.85</td><td>-</td><td>98.25</td></tr><tr><td>XG-Boost-Beta-T (2023)[53]</td><td></td><td>97.33</td><td>98.27</td><td>97.14</td><td>97.6</td><td>97.7</td></tr><tr><td>ACPL (2022)[54]</td><td>0.9436</td><td></td><td></td><td>0.7214</td><td></td><td>0.6223</td></tr><tr><td>HealthyGAN (2022) [55]</td><td>0.5600</td><td>–</td><td>0.5500</td><td>0.5500</td><td>0.5500</td><td>0.5500</td></tr><tr><td>SA-CNN (2021)[56]</td><td>-</td><td>96.67</td><td>96.69</td><td>96.67</td><td>一</td><td>96.67</td></tr><tr><td>Kai et al (2022) [57]</td><td>0.997</td><td>0.9796</td><td>0.9853</td><td>0.9738</td><td>0.9854</td><td>0.9795</td></tr><tr><td>SRC-MT (2021)[58]</td><td>0.9358</td><td></td><td>-</td><td>0.7147</td><td></td><td>0.6068</td></tr><tr><td>DenseNet121(2017)[59]</td><td>0.9928</td><td>0.929</td><td>0.8883</td><td>0.9793</td><td>0.8823</td><td>0.9307</td></tr><tr><td>ResNet18 (2016)[60]</td><td>0.9867</td><td>0.9509</td><td>0.9339</td><td>0.9686</td><td>0.9334</td><td>0.9504</td></tr><tr><td>ResNet50 (2016)[60]</td><td>0.9885</td><td>0.931</td><td>0.8951</td><td>0.9744</td><td>0.8903</td><td>0.9323</td></tr><tr><td>Inceptionv3 (2016)[61]</td><td>0.9907</td><td>0.9275</td><td>0.8866</td><td>0.9773</td><td>0.8805</td><td>0.9301</td></tr><tr><td>VGG16 (2015)[62]</td><td>0.9836</td><td>0.9274</td><td>0.9101</td><td>0.9439</td><td>0.9115</td><td>0.9264</td></tr><tr><td>VGG19 (2015) [62]</td><td>0.9878</td><td>0.9274</td><td>0.8883</td><td>0.9748</td><td>0.8834</td><td>0.929</td></tr><tr><td>AlexNet (2012) [63]</td><td>0.9791</td><td>0.9133</td><td>0.8827</td><td>0.9465</td><td>0.8826</td><td>0.9148</td></tr></table>

EPIFF achieved 98.56%, 99.12%, 98.11%, 97.64%, 99.47%, and 97.87% of AUC, accuracy, precision, recall, and F1 score respectively demonstrating its classification dominance over its peers. Specifically, it surpassed the recent DL schemes Red Deer[52], XG-Boost-Beta–T[53], ACPL[54], and HealthyGAN[55] thereby promising its effectiveness in acutely diagnosing abnormal samples. EPI’s distinctive quality

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

in effectively highlighting the low-contrast regions followed by localization using scale invariant fractals offers high class-wise structural discrimination with improved accuracy. However, EPIFF’s performance lags slightly behind Kai et al [57], whose accuracy completely relied on the training of the complex Siamese deep neural network, demanding extensive hyperparameter tuning that is empirically derived to achieve optimality and hence, consumes intensive time. Rather, the elegant EPIFF requires less training and no parameter tuning, offering swift classification in fewer computations than its peers, making this model a promising tool for real-world applications in medical diagnostics.

In addition to the qualitative analysis, the Receiver Operating Characteristic (ROC) metrics namely Specificity (TPR) and Sensitivity (FPR) are plotted along the y and x-axis respectively for quantitatively analyzing EPIFF performance in Fig. 5. Generally, the model’s ability to predict correct samples from the overall positive samples signifies its proximity to the top-left corner as shown in Fig. 5.

![](images/57fd995944910458643e90d14928649c014582af797bdc0da54d3d32e17b23da.jpg)  
Fig. 5 Proposed model’s Receiver Operating Characteristics curve

Fig. 5 illustrates the potential of EPIFF, emphasized by a substantial Area Under Curve (AUC). Notably, the AUC of ROC consistently exceeds 0.98 for all classes, indicating the superior classification ability of the EPIFF in distinguishing CXRs with and without abnormalities across all classes.

Consequently, Table 6 presents EPIFF-MARS performance with that of the recent competitors, assessed through the Area under the ROC Curves metric.

Table 6 Relative Analysis of Proposed Model’s AUC in the ROC
<table><tr><td>Methods</td><td>Edema</td><td>Effusion</td><td>Emphysema</td><td>Pneumonia</td></tr><tr><td>EPIFF-MARS</td><td>0.9836</td><td>0.9822</td><td>0.9884</td><td>0.9889</td></tr><tr><td>Hybrid CNN-T (2024)[64]</td><td>0.8979</td><td>0.8839</td><td>0.9227</td><td>0.7651</td></tr><tr><td>EEEA-Net-C2- KD(2024)[65]</td><td>0.9006</td><td>0.8762</td><td>0.9147</td><td>0.7582</td></tr><tr><td>SynthEnsemble (2023)[66]</td><td>0.9103</td><td>0.8897</td><td>0.9294</td><td>0.7764</td></tr><tr><td>Unichest (2023)[67]</td><td>0.893</td><td>0.861</td><td>0.958</td><td>0.933</td></tr><tr><td>FedKDF (2023) [68]</td><td>0.8382</td><td>0.8298</td><td>0.8534</td><td>0.6609</td></tr><tr><td>Nie et al (2023) [69]</td><td>0.9000</td><td>0.9100</td><td>0.9400</td><td>0.8200</td></tr><tr><td>BB-GCN (2023) [70]</td><td>0.9100</td><td>0.9220</td><td>0.8970</td><td>0.7680</td></tr><tr><td>SwinCheX (2022) [71]</td><td>0.8510</td><td>0.8270</td><td>0.9140</td><td>0.7310</td></tr><tr><td>LSAE (2022) [72]</td><td>0.8401</td><td>0.8214</td><td>0.8547</td><td>0.7088</td></tr><tr><td>ImageGCN (2022) [73]</td><td>0.8800</td><td>0.8700</td><td>0.9200</td><td>0.7200</td></tr><tr><td>Ouyang et al.(2021) [74]</td><td>0.9000</td><td>0.8800</td><td>0.9400</td><td>0.7300</td></tr><tr><td>Bose et al.(2021) [75]</td><td>0.9210</td><td>0.8310</td><td>0.8610</td><td>0.7620</td></tr><tr><td>Gundel et al. (2021) [76]</td><td>0.8920</td><td>0.8850</td><td>0.9250</td><td>0.7650</td></tr><tr><td>DGFN (2020) [77]</td><td>0.8925</td><td>0.8751</td><td>0.9357</td><td>0.7791</td></tr><tr><td>CheXGCN (2020) [78]</td><td>0.8500</td><td>0.8320</td><td>0.9440</td><td>0.7390</td></tr><tr><td>Baltruschat et al.(2019) [79]</td><td>0.8460</td><td>0.8220</td><td>0.8950</td><td>0.7140</td></tr><tr><td>DualCheXN (2019) [80]</td><td>0.8520</td><td>0.8310</td><td>0.9420</td><td>0.7270</td></tr><tr><td>ChestNet (2018) [81]</td><td>0.8327</td><td>0.8114</td><td>0.8222</td><td>0.6959</td></tr><tr><td>DNet D-161 (2018) [82]</td><td>0.8880</td><td>0.8640</td><td>0.8980</td><td>0.7150</td></tr><tr><td>AG-CNN (2018) [83]</td><td>0.9240</td><td>0.9030</td><td>0.9320</td><td>0.7740</td></tr><tr><td>CAN1 (2019) [84]</td><td>0.8460</td><td>0.8280</td><td>0.8920</td><td>0.7210</td></tr><tr><td>Wang et al. (2017) [49]</td><td>0.8350</td><td>0.7840</td><td>0.8150</td><td>0.6330</td></tr><tr><td>Yao et al. (2017) [85]</td><td>0.8820</td><td>0.8590</td><td>0.8290</td><td>0.7130</td></tr><tr><td>CheXNet (2017) [86]</td><td>0.8870</td><td>0.8630</td><td>0.9370</td><td>0.7680</td></tr><tr><td>DNet D-121 (2017) [86]</td><td>0.8920</td><td>0.8850</td><td>0.9250</td><td>0.7650</td></tr></table>

The EPIFF-MARS model demonstrates superior performance in Table 6, achieving a higher Area Under the ROC Curve (AUC) compared to existing X-ray-based lung disease detection models in which most existing models rely on deep learning architectures, demanding significant computational resources while EPIFF-MARS is relatively simple. The superior achievements registered by the EPIFF-MARS are attributed to the effectiveness of the individual processing modules contributing significantly to the overall accuracy. Specifically, EPIFF outperforms recent methods [64-70] in discriminating various types of RDs, achieving significant improvements of 7% in edema detection, 8% in effusion detection, 5% in emphysema detection, and 22% in pneumonia detection. This efficacy is due to the pairwise

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

analysis of the EPIFF-MARS approach that automatically identifies the most relevant features for each specific RDS category. Consequently, this leads to increased inter-class deviations, boosting classification accuracy while reducing the feature space.

Likewise, qualitative parameters estimated from CXRs for detecting COVID-19 are systematically compared with existing SOTA to illustrate the EPIFF efficacy in Table 7.

Table 7 Relative Analysis of Proposed Model in COVID-19 Detection
<table><tr><td>Methods</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>Specificity</td><td>F1-Score</td></tr><tr><td>EPIFF-MARS</td><td>0.9909</td><td>0.9785</td><td>0.9756</td><td>0.9945</td><td>0.9770</td></tr><tr><td>AIBF (2024)[87]</td><td>0.9583</td><td>0.9516</td><td>0.9621</td><td>1</td><td>0.976</td></tr><tr><td>ENet-B0-GAN (2023)[88]</td><td>0.841</td><td>0.86</td><td>0.86</td><td>0.946</td><td>0.856</td></tr><tr><td>Inception-v3 - GAN (2023)[88]</td><td>0.871</td><td>0.886</td><td>0.874</td><td>0.956</td><td>0.88</td></tr><tr><td>ResNet50 (2023) [28]</td><td>0.9567</td><td>0.9537</td><td>0.9594</td><td>0.9540</td><td>0.9566</td></tr><tr><td>AlexNet (2023) [28]</td><td>0.9362</td><td>0.9395</td><td>0.9334</td><td>0.9391</td><td>0.9364</td></tr><tr><td>ResNet-50 (2022)[89]</td><td>0.9614</td><td>0.963</td><td>0.96</td><td></td><td>0.963</td></tr><tr><td>MKSC (2021)[90]</td><td>0.9817</td><td>0.9813</td><td>0.9809</td><td>0.9825</td><td>0.9811</td></tr><tr><td>CBAM (2021) [91]</td><td>0.9633</td><td>0.9620</td><td>0.9530</td><td>0.9650</td><td>0.9650</td></tr><tr><td>ECA-Net (2021) [92]</td><td>0.9740</td><td>0.9761</td><td>0.9711</td><td>0.9770</td><td>0.9736</td></tr><tr><td>CheXnet (2020) [93]</td><td>0.9774</td><td>0.9661</td><td>0.9661</td><td>0.9661</td><td>0.9831</td></tr><tr><td>DenseNet201 (2020) [93]</td><td>0.9519</td><td>0.9506</td><td>0.959</td><td>0.9787</td><td>0.9504</td></tr><tr><td>VGG-Net (2020) [94]</td><td>0.9639</td><td>0.9632</td><td>0.9644</td><td>0.9640</td><td>0.9641</td></tr><tr><td>DarkCovidNet (2020)[95]</td><td>0.9632</td><td>0.9639</td><td>0.9621</td><td>0.9685</td><td>0.9630</td></tr><tr><td>SE-Net (2018) [96]</td><td>0.9693</td><td>0.9626</td><td>0.9734</td><td>0.9640</td><td>0.9680</td></tr><tr><td>ResNet18 (2016) [60]</td><td>0.9596</td><td>0.9610</td><td>0.9605</td><td>0.9624</td><td>0.9607</td></tr></table>

From Table 7, it is obvious that EPIFF-MARS outperformed other SOTA schemes in terms of the numerous ROC parameters. Specifically, its 99.10% accuracy signifies its dominance over its peers, whilst, it marginally falls behind the F1 scores of CheXnet [93]. Despite this decline, its superior consistency across the ROC spectrum is missing in this competitor. Likewise, MKSC [90] dominates EPIFF-MARS along the Precision and Recall dimensions thereby escalating the F1 score relatively. MKSC’s dominance is mainly due to the exhaustive training nature of multi-kernel attention networks demanding extensive computations. Overall, the comparisons in Table 7 reveal the intense competition offered by EPIFF-MARS at a minimal complexity making it more suitable for real-time scenarios than the trending DL counterparts.

In addition, to prove the scalable nature of EPIFF-MARS, an extensive investigation with varying feature lengths of the NIH Chest X-ray 14 dataset is performed. To begin with, EPIFF-MARS extracts the essential features from CXRs and is subjected to MARS for disease classification. During training, by varying the threshold (maximum number of feature dimensions), MARS automatically selects the optimal feature subset within that range for each disease category. The achieved accuracy and F1 scores are presented in Table 8.

Table 8. Analysis of EPIFF-MARS model’s Scalability
<table><tr><td>Disease</td><td>Feature Size</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>MaxThreshold Range</td><td rowspan=1 colspan=1>SelectedFeatureS</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>F1Sscore</td><td rowspan=1 colspan=1>Execution Time(ms)</td></tr><tr><td rowspan=3 colspan=1>Edema</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>0.9953</td><td rowspan=1 colspan=1>0.9883</td><td rowspan=1 colspan=1>30.71</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.9055</td><td rowspan=1 colspan=1>22.51</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>0.7943</td><td rowspan=1 colspan=1>20.96</td></tr><tr><td rowspan=3 colspan=1>Covid-19</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>0.996</td><td rowspan=1 colspan=1>0.9899</td><td rowspan=1 colspan=1>37.65</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.9346</td><td rowspan=1 colspan=1>32.93</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.8288</td><td rowspan=1 colspan=1>29.42</td></tr><tr><td rowspan=3 colspan=1>Effusion</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>0.992</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>30.64</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.9406</td><td rowspan=1 colspan=1>24.5</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>0.754</td><td rowspan=1 colspan=1>0.7773</td><td rowspan=1 colspan=1>22.79</td></tr><tr><td rowspan=3 colspan=1>Emphysema</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>0.9933</td><td rowspan=1 colspan=1>0.9832</td><td rowspan=1 colspan=1>39.05</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>0.915</td><td rowspan=1 colspan=1>0.9137</td><td rowspan=1 colspan=1>32.47</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>20.76</td></tr><tr><td rowspan=3 colspan=1>Pneumonia</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>0.9893</td><td rowspan=1 colspan=1>0.9732</td><td rowspan=1 colspan=1>31.17</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>0.965</td><td rowspan=1 colspan=1>0.9652</td><td rowspan=1 colspan=1>25.75</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.8293</td><td rowspan=1 colspan=1>21.78</td></tr></table>

Table 8 showcases the EPIFF-MARS efficiency in feature selection wherein an averaged accuracy over 75% with a 30% feature reduction, and 99% with a 40% reduction is achieved at a significantly lower processing time under 20-30 ms thereby enabling faster and more resource-efficient classification tasks. This is due to the MARS quality in optimally selecting a small number of features through an intelligent selection process by focusing on highly relevant features and discarding unnecessary information. This leads to two key benefits: 1) Reduced complexity: The model is faster to train and requires less computational power, making it efficient for real-world applications. 2) Enhanced accuracy: By selecting the most informative features, MARS avoids irrelevant data, ultimately achieving superior classification results compared to models that rely on a larger generic feature set. Thereby, the intended model effectively maintains a trade-off between accuracy and efficiency.

Further, to justify the Robustness of the EPIFF-MARS, the model conducted the test using the balanced subset dataset composed from the ChestX-ray NIH 14 and the Covid Image Repository. This dataset included 1500 images, with 300 images for each disease category and the ROC accomplishments are tabulated in Table 9.

Table 9 Performance of EPIFF-MARS on the balanced dataset
<table><tr><td rowspan=2 colspan=1>Disease</td><td rowspan=1 colspan=4>Confusion Matrix</td><td rowspan=2 colspan=1>Accuracy</td><td rowspan=2 colspan=1>Precision</td><td rowspan=2 colspan=1>Recall</td><td rowspan=2 colspan=1>Specificity</td><td rowspan=3 colspan=1>F1Score0.9883</td></tr><tr><td rowspan=1 colspan=1>TP</td><td rowspan=1 colspan=1>FP</td><td rowspan=1 colspan=1>FN</td><td rowspan=1 colspan=1>TN</td></tr><tr><td rowspan=1 colspan=1>Edema</td><td rowspan=1 colspan=1>296</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>1197</td><td rowspan=1 colspan=1>0.9953</td><td rowspan=1 colspan=1>0.9900</td><td rowspan=1 colspan=1>0.9867</td><td rowspan=1 colspan=1>0.9975</td></tr><tr><td rowspan=1 colspan=1>Covid</td><td rowspan=1 colspan=1>295</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1199</td><td rowspan=1 colspan=1>0.9960</td><td rowspan=1 colspan=1>0.9966</td><td rowspan=1 colspan=1>0.9833</td><td rowspan=1 colspan=1>0.9992</td><td rowspan=1 colspan=1>0.9899</td></tr><tr><td rowspan=1 colspan=1>Effusion</td><td rowspan=1 colspan=1>294</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1194</td><td rowspan=1 colspan=1>0.9920</td><td rowspan=1 colspan=1>0.9800</td><td rowspan=1 colspan=1>0.9800</td><td rowspan=1 colspan=1>0.995</td><td rowspan=1 colspan=1>0.9800</td></tr><tr><td rowspan=1 colspan=1>Emphysema</td><td rowspan=1 colspan=1>292</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1198</td><td rowspan=1 colspan=1>0.9933</td><td rowspan=1 colspan=1>0.9832</td><td rowspan=1 colspan=1>0.9733</td><td rowspan=1 colspan=1>0.9983</td><td rowspan=1 colspan=1>0.9832</td></tr><tr><td rowspan=1 colspan=1>Pneumonia</td><td rowspan=1 colspan=1>294</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>1196</td><td rowspan=1 colspan=1>0.9933</td><td rowspan=1 colspan=1>0.9833</td><td rowspan=1 colspan=1>0.9800</td><td rowspan=1 colspan=1>0.9967</td><td rowspan=1 colspan=1>0.9833</td></tr></table>

Table 9 showcases the EPIFF-MARS consistent performance on a balanced dataset which is owed to the individual processing steps. Especially, the EPI transformation significantly enhanced the intensity of low-contrast regions in chest X-rays (CXRs), highlighting crucial areas that are more difficult to capture by the existing schemes demanding pre-processing. As a result, EPIFF-MARS registered average accuracies with specificities over 99% and averaged F1 scores with Sensitivity greater than 98% demonstrating its potential in the classification of diverse RDs. This achievement is particularly noteworthy compared to imbalanced datasets, suggesting the model's robustness.

Further, To comprehensively validate the effectiveness of the EPIFF-MARS pipeline, the process conducted an ablation study utilizing a balanced dataset that consists of a subset of 200 samples per category from the ChestX-ray14 dataset. Accordingly, the process adopted a two-step approach. Firstly, the model's performance is assessed directly on the raw data for performing baseline comparisons. Subsequently, the process systematically applied each transformation (EPI, Polar, and Fractal set) in EPIFF-MARS one by one to the raw data. Across each stage, the model's performance is evaluated to understand the impact of each processing stage in the EPIFF-MARS. The stage-wise outcomes of the developed EPIFF-MARS are presented in Table 10.

Table 10 Ablation study of EPIFF-MARS vs. Raw data
<table><tr><td rowspan="2">Image Features</td><td rowspan="2">Primary Class</td><td colspan="4">Confusion Matrix</td><td rowspan="2">Accuracy</td><td rowspan="2">F1 Score</td><td rowspan="2">Sensitivit y</td><td rowspan="2">Specificit y</td></tr><tr><td>TP</td><td>FP</td><td>FN</td><td>T N</td></tr><tr><td rowspan="6">RAW</td><td>EDEMA</td><td>12 8</td><td>11 5</td><td>72</td><td>68 5</td><td>0.813</td><td>0.5779</td><td>0.64</td><td>0.8563</td></tr><tr><td>COVID</td><td>11 7</td><td>98</td><td>83</td><td>70 2</td><td>0.819</td><td>0.5639</td><td>0.585</td><td>0.8775</td></tr><tr><td>EFFUSION</td><td>10 3</td><td>13 3</td><td>97</td><td>66 7</td><td>0.777</td><td>0.4725</td><td>0.515</td><td>0.8338</td></tr><tr><td>EMPHYSEM A</td><td>95</td><td>17 7</td><td>10 5</td><td>62 3</td><td>0.718</td><td>0.4025</td><td>0.475</td><td>0.7788</td></tr><tr><td>PNEUMONI A</td><td>10</td><td>12 5</td><td>10</td><td>67</td><td>0.775</td><td>0.4706</td><td>0.5</td><td>0.8438</td></tr><tr><td>EDEMA</td><td>0 14</td><td>94</td><td>0 59</td><td>5 70 6</td><td>0.847</td><td>0.6483</td><td>0.705</td><td>0.8825</td></tr></table>

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

<table><tr><td rowspan="5"></td><td>COVID</td><td>13 7</td><td>90</td><td>71 63 0</td><td>0.847</td><td>0.6417</td><td>0.685</td><td></td><td>0.8875</td></tr><tr><td>EFFUSION</td><td>11 0</td><td>10 7</td><td>90</td><td>69 3</td><td>0.803</td><td>0.5276</td><td>0.55</td><td>0.8663</td></tr><tr><td>EMPHYSEM A</td><td>14 4</td><td>11 2</td><td>56</td><td>68 8</td><td>0.8286</td><td>0.6316</td><td>0.72</td><td>0.8564</td></tr><tr><td>PNEUMONI A</td><td>12 1</td><td>89</td><td>79</td><td>71 1</td><td>0.832</td><td>0.5902</td><td>0.605</td><td>0.8888</td></tr><tr><td>EDEMA</td><td>17</td><td>66</td><td>22</td><td>73 4</td><td>0.912</td><td>0.8018</td><td>0.89</td><td>0.9175</td></tr><tr><td rowspan="5">EPI + Polar</td><td>COVID</td><td>8 18</td><td>45</td><td>18</td><td>75</td><td>0.937</td><td>0.8525</td><td>0.91</td><td>0.9438</td></tr><tr><td>EFFUSION</td><td>2 14</td><td>41</td><td>58</td><td>5 75</td><td>0.901</td><td>0.7415</td><td>0.71</td><td>0.9488</td></tr><tr><td>EMPHYSEM</td><td>2 14</td><td>18</td><td>60</td><td>9 78</td><td>0.922</td><td>0.7821</td><td>0.7</td><td>0.9775</td></tr><tr><td>A PNEUMONI</td><td>0 15</td><td>11</td><td>47</td><td>2 78</td><td>0.942</td><td>0.8407</td><td>0.765</td><td></td></tr><tr><td>A</td><td>3</td><td></td><td></td><td>9</td><td></td><td></td><td></td><td>0.9863</td></tr><tr><td rowspan="5">Propose d EPIFF (EPI + Polar + Fractals)</td><td>EDEMA</td><td>19 7</td><td>2</td><td>3</td><td>79 8</td><td>0.995</td><td>0.9875</td><td>0.985</td><td>0.9975</td></tr><tr><td>COVID</td><td>19 6</td><td>1</td><td>4</td><td>79 9</td><td>0.995</td><td>0.9874</td><td>0.98</td><td>0.9988</td></tr><tr><td>EFFUSION</td><td>19 4</td><td>5</td><td>6</td><td>79 5</td><td>0.989</td><td>0.9724</td><td>0.97</td><td>0.9938</td></tr><tr><td>EMPHYSEM A</td><td>19 4</td><td>1</td><td>6</td><td>79 9</td><td>0.993</td><td>0.9823</td><td>0.97</td><td></td></tr><tr><td>PNEUMONI A</td><td>19 6</td><td>5</td><td>4</td><td>79 5</td><td>0.991</td><td>0.9776</td><td>0.98</td><td>0.9988 0.9938</td></tr></table>

The results in Table 10 effectively highlight the impact of individual transformations involved in the EPIFF-MARS model against Raw data. As evident in Table 10, the processing across each stage contributes to significant improvement in the overall performance of the EPIFF-MARS model. Features extracted from the raw data registered an average accuracy of 78% with an F1-score of 49%, indicating limited effectiveness in differentiating abnormalities. In contrast, the extracted EPI transformed features achieved an average percentage of accuracy of 83.2 and an F1 score of 60.1 due to its contrast stretching nature that effectively highlights structural variations in low-contrast regions. Following EPI transformation, the features from the polar domain showcase significant improvement in ROC metrics due to its radial nature supplementing curved localization. Later, the extracted fractal features from the transformations raise the EPIFF-MARS performance significantly with an improvement of 7% in accuracy, 21% in F1 score, 23% in sensitivity, and 4% in specificity over EPI+polar features. This achievement is due to the exploitation of self-similar fractal patterns that effectively capture the structural variations present in diverse disease categories on chest X-rays. Thus, the ablation study showcases the potency of individual transformations formulating the EPIFF-MARS model.

Further, to investigate EPIFF’s ability towards the noise, experiments are conducted involving the manual addition of various noise types namely Gaussian, Salt & Pepper, and Speckle to the CXRs, and their representations are illustrated in Fig. 6 with the averaged ROC outcomes stated in Table 11.

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

![](images/deea93a38225b161238e6b5247d7f104be2d0a64bfdb1142843d01adc26b8bd4.jpg)  
(a)

![](images/5e44a9374f8d537bf6f542ff3b93242563df7e6868ccf0321e6c2f60278dbb11.jpg)  
(b)

![](images/ba28e39fc457c8546be85f876a7018d83ee894b19becccf4af5eb90843df1a79.jpg)  
(c)

![](images/1419c47b74220e9e1e43c4c84712d628e26c5ae27297e88148a4a87b7582ad71.jpg)  
(d)

![](images/d289fbeb829289cd118849ad4f99632bd6fbad9814591c0c24141dc0da628a39.jpg)  
(e)

Fig. 6 Sample noise added images and its representations: (a) noisy input image;(b) EPI transformed image; (c) Polar transformed image; (d) Mandelbrot Set; (e) Julia Set.  
Table 11 Noise analysis of the EPIFF-MARS model
<table><tr><td>NOISE</td><td>Parameter S</td><td>AUC</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>Specificit y</td><td>F1-Score</td></tr><tr><td>w/o Noise</td><td></td><td>0.9857</td><td>0.9906</td><td>0.9949</td><td>0.9939</td><td>0.9821</td><td>0.9944</td></tr><tr><td>Gaussia n Noise</td><td>Variance (0.01, 0.1, 0.5)</td><td>0.9700</td><td>0.9789</td><td>0.9796</td><td>0.9600</td><td>0.9800</td><td>0.9697</td></tr><tr><td>Salt &amp; Pepper Noise</td><td>Density (0.01, 0.1, 0.5)</td><td>0.9601</td><td>0.9500</td><td>0.9388</td><td>0.9583</td><td>0.9423</td><td>0.9485</td></tr><tr><td>Speckle Noise</td><td>Variance (0.01, 0.1, 0.5)</td><td>0.9549</td><td>0.9412</td><td>0.9184</td><td>0.9574</td><td>0.9273</td><td>0.9375</td></tr></table>

It is evident from Table 11, that the EPIFF produced consistent output regardless of diverse kinds of noise added with varying ratios. This is achieved due to the inherent smoothening nature of the introduced EPI which successfully averages out noise, thereby, resulting in diminishing its visibility in the image.

Further, to investigate the model’s compatibility with the dataset, two additional metrics namely R2 score and MSE are estimated and presented in Table 12 indicating how well the model fits with datasets.

Table 12 Analysis of the Prediction capability of the proposed model
<table><tr><td>Class Name</td><td>R2 Score</td><td>MSE</td></tr><tr><td>Edema</td><td>0.9557</td><td>0.0054</td></tr><tr><td>Covid - 19</td><td>0.9390</td><td>0.0126</td></tr><tr><td>Effusion</td><td>0.9304</td><td>0.0153</td></tr><tr><td>Emphysem a</td><td>0.9506</td><td>0.0081</td></tr><tr><td>Pneumonia</td><td>0.9557</td><td>0.0054</td></tr></table>

MSE values less than 0.01 in Table 12, emphasize the prediction ability of EPIFF-MARS as lower MSE signifies the closer alignment of predictions with actual results. In contrast, the R2 score surpassed 93% showcasing the model’s generalization ability to fit with diverse datasets in terms of their prediction performance which is owed to the two-stage MARS process in formulating the classification model.

Furthermore, to illustrate the effectiveness of the EPIFF features, the experiments are performed using the extracted EPIFF features of CXR images coupled with diverse classification models, and the outcomes are tabulated in Table 13

Table 13 Relative Analysis of the proposed model with other SOTA classification schemes
<table><tr><td>Model Setting</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>EPIFF-MARS</td><td>0.9912</td><td>0.9811</td><td>0.9764</td><td>0.9787</td></tr><tr><td>XGBoost</td><td>0.8762</td><td>0.8810</td><td>0.8690</td><td>0.8750</td></tr><tr><td>RandomForest</td><td>0.9069</td><td>0.9480</td><td>0.8930</td><td>0.9197</td></tr><tr><td>LightGBM</td><td>0.9569</td><td>0.9480</td><td>0.9381</td><td>0.9430</td></tr><tr><td>Deep Neural Network with Adam Optimizer</td><td>0.9845</td><td>0.9783</td><td>0.9835</td><td>0.9809</td></tr><tr><td>SVM</td><td>0.9626</td><td>0.9410</td><td>0.9688</td><td>0.9547</td></tr><tr><td>LogisticRegression</td><td>0.8824</td><td>0.8840</td><td>0.8900</td><td>0.8870</td></tr><tr><td>DecisionTree</td><td>0.8320</td><td>0.8251</td><td>0.8144</td><td>0.8197</td></tr></table>

EPIFF coupled with MARS has surpassed its peers in Table 13 as witnessed in the registered performance scores. In comparison to a general linear regression or complex neural network, the MARS model is more flexible, computationally less expensive, and doesn’t demand complex architecture. Moreover, MARS’s well-maintained bias-variance trade-off with interpretability enables computing the predictors’ functionality with their overall weightage. Overall it is claimed that EPIFF-MARS is superior in diagnostic accuracy with reduced complexity, is well suitable for less-experienced clinicians, and reduces interpretation variability among physicians. Also, the introduced methodology ensures that it is more adaptable to various RDs based on datasets and clinical contexts.

## 4. Computational complexity

To assess the practical efficiency and applicability of the EPIFF-MARS, a thorough analysis of its computational complexity, considering both time and space, has been conducted. The overall time complexity of the model is assessed by accounting for the time required at each developmental step. The model is partitioned into three stages. In the initial stage, the EPI transform is applied to an image (�(� × �)) using a pixel-wise exponential function, incurring a time complexity of �(� × �) where � × � − represents the dimension of an image. Subsequently, in the second stage, the EPI is further transformed into fractal sets. The computational complexity for calculating Mandelbrot and Julia Sets is directly proportional to the number of iterations (�) needed for each pixel, resulting in a complexity of �(�(� × �)) from which features are extracted. Finally, the features are utilized in the MARS model for classification and its complexity is expressed as $O ( B \times ( N \times F ) + M )$ where, � − number of basis functions, � −number of samples in datasets, � − total number of features applied to the model, and the additional '+M' term signifies the complexity associated with post-processing. Hence, the time complexity of the entire model is expressed in Eq. 41

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

$$
\begin{array} { c } { { T _ { t o t a l } = O ( n \times m ) + O ( t ( n \times m ) ) + O ( B \times ( N \times F ) + B ) } } \\ { { { } } } \\ { { T _ { t o t a l } \approx O ( B \times ( N \times F ) + B ) } } \end{array}\tag{42}
$$

Further, to demonstrate the efficiency of the proposed model, the processing times (in seconds) are relatively compared with its peers and presented in Table 14.

Table 14. Relative analysis of time complexity
<table><tr><td>Models</td><td>Model Processing Time (S)</td></tr><tr><td>ResNet-50</td><td>1349</td></tr><tr><td>DenseNet-201</td><td>2399</td></tr><tr><td>VGG-16</td><td>811</td></tr><tr><td>DenseNet-169</td><td>2157</td></tr><tr><td>Inception-v3</td><td>1239</td></tr><tr><td>Simple CNN</td><td>405</td></tr><tr><td>VGG19</td><td>991</td></tr><tr><td>ResNet-101</td><td>789</td></tr><tr><td>EPIFF-MARS</td><td>57</td></tr></table>

From Table 14 it is understood that there is a substantial variation in processing times across models, and notably, EPIFF-MARS stands out with an exceptionally short processing time, showcasing its high simplicity. In contrast, the other models demonstrate different training times, influenced by their respective architectures and complexities. The remarkable efficiency of the EPI-MARS model is attributed to less intricate developmental stages, making it a noteworthy option for tasks requiring swift classification.

Space complexity is assessed by considering each step in the realization of EPIFF-MARS. Upon processing the input image $\left( I ( n \times m ) \right)$ by the EPI, yields a feature matrix of size whose complexity is $O ( n \times m )$ . Similarly, fractal sets generation demands $O ( n \times m )$ space, as the resulting fractal set is in matrix form with dimensions matching the input image size. Finally, the space complexity of the MARS model is contingent on the number of basis functions (�), in addition to the space required for storing the features of samples in the dataset ends in ${ \cal O } ( N \times F )$ . Consequently, the total space requirement for the MARS model is to $O ( B + ( N \times F )$ . Overall, the proposed model's space complexity is expressed in Eq. 42

$$
\begin{array} { r l } & { S _ { t o t a l } = O ( n \times m ) + O ( B + ( N \times F ) ) } \\ & { } \\ & { S _ { t o t a l } \approx O ( B + ( N \times F ) ) } \end{array}\tag{43}
$$

Thereby, the EPIFF-MARS is easy to implement on simple hardware structures and ensures low computational cost, making it suitable for use in resource-constrained settings.

## 5. Conclusion

In this paper, an elegant mathematical model for the automatic diagnosis of CXRs using a novel Exponential Pixelating Integral (EPI) is presented to classify various respiratory disorders for the benefit of both experienced medical professionals and novice physicians. The EPI transformation computes an exponential moving average over a set of pixels bound by an overlapping local 3x3 kernel. This technique effectively standardizes the images and helps to highlight all intensities, overcoming the limitations of differential intensity values in grayscale CXRs. The EPI is subsequently converted into Mandelbrot and Julia fractal representations via polar coordinates, enhancing the differentiation

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine   
(CIBM), Vol. 182, November 2024. Final version available at DOI:   
https://doi.org/10.1016/j.compbiomed.2024.109093

between normal tissue structures and abnormalities. The features from the three intermediaries are collated to construct an ensemble model of Multivariate Adaptive Regression Splines (MARS) for each pair of classes aiding Respiratory disorder classification. The rigorous analysis of the EPIFF-MARS on the large benchmark datasets demonstrated its consistency and superiority over its peers. Further, the noise analysis and Computational complexity revealed the robustness and amicability of EPIFF-MARS for diverse real-time scenarios. This study demonstrated a well-formulated design that can be used by the research community to develop AI solutions for diagnosing RDs. Although EPIFF-MARS effectively classified abnormality in CXRs, the acute identification of lesion positions needs further investigation for severity analysis thus motivating for deeper exploration of fractal-based textural features. Furthermore, optimizing EPIFF-MARS for clinical use requires extensive investigation by exploring model simplification and efficient algorithms to balance accuracy with computational efficiency. In addition, future efforts to broaden EPIFF-MARS to cover a wider range of RDs and explore its potential as a multimodal diagnostic tool, incorporating MRI, CT scans, and even ultrasound data alongside chest X-rays. Therefore, extending EPIFF-MARS to other imaging modalities will involve adapting it to the specific characteristics of each data type, potentially through data augmentation techniques or incorporating clinical settings into the feature extraction process.

## References

[1] Chouhan V, Singh SK, Khamparia A, Gupta D, Tiwari P, Moreira C, et al. A novel transfer learning based approach for pneumonia detection in chest X-ray images. Applied Sciences (Switzerland) 2020;10. https://doi.org/10.3390/app10020559.

[2] Wang G, Liu X, Shen J, Wang C, Li Z, Ye L, et al. A deep-learning pipeline for the diagnosis and discrimination of viral, non-viral and COVID-19 pneumonia from chest X-ray images. Nature Biomedical Engineering 2021;5:509–21. https://doi.org/10.1038/s41551-021-00704-1.

[3] Liu J, Qi J, Chen W, Nian Y. Multi-branch fusion auxiliary learning for the detection of pneumonia from chest X-ray images. Computers in Biology and Medicine 2022;147:105732. https://doi.org/10.1016/j.compbiomed.2022.105732.

[4] Khatri A, Jain R, Vashista H, Mittal N, Ranjan P, Janardhanan R. Pneumonia identification in chest Xray images using EMD. Trends in Communication, Cloud, and Big Data: Proceedings of 3rd National Conference on CCB, 2018, 2020, p. 87–98.

[5] Jacobi A, Chung M, Bernheim A, Eber C. Portable chest X-ray in coronavirus disease-19 (COVID-19): A pictorial review. Clinical Imaging 2020;64:35–42. https://doi.org/10.1016/j.clinimag.2020.04.001.

[6] Sayed SAF, Elkorany AM, Sayed Mohammad S. Applying Different Machine Learning Techniques for Prediction of COVID-19 Severity. IEEE Access 2021;9:135697–707. https://doi.org/10.1109/ACCESS.2021.3116067.

[7] Saaudi A, Mansoor R, Abed AK. Clustering and Visualizing of Chest X-ray Images for Covid-19 Detection. Proceedings of 2021 2nd Information Technology to Enhance E-Learning and Other Application Conference, IT-ELA 2021 2021:35–9. https://doi.org/10.1109/IT-ELA52201.2021.9773539.

[8] Siddiquee SM, Nizam NB, Shirin M, Bhuiyan MIH, Hasan T. COVID-19 Severity Prediction from Chest X-ray Images using an Anatomy-Aware Deep Learning Model 2022.

Qin C, Yao D, Shi Y, Song Z. Computer-aided detection in chest radiography based on artificial intelligence: A survey. BioMedical Engineering Online 2018;17:1–23. https://doi.org/10.1186/s12938- 018-0544-y.

[10] Iqbal T, Shaukat A, Akram MU, Mustansar Z, Khan A. Automatic Diagnosis of Pneumothorax from Chest Radiographs: A Systematic Literature Review. IEEE Access 2021;9:145817–39. https://doi.org/10.1109/ACCESS.2021.3122998.

[11] Saif AFM, Imtiaz T, Shahnaz C, Zhu WP, Ahmad MO. Exploiting cascaded ensemble of features for the detection of tuberculosis using chest radiographs. IEEE Access 2021;9:112388–99. https://doi.org/10.1109/ACCESS.2021.3102077.

[12] Li C, Zhang D, Du S, Tian Z. Deformation and Refined Features Based Lesion Detection on Chest X-Ray. IEEE Access 2020;8:14675–89. https://doi.org/10.1109/ACCESS.2020.2963926.

[13] Mahmud T, Rahman MA, Fattah SA. CovXNet: A multi-dilation convolutional neural network for automatic COVID-19 and other pneumonia detection from chest X-ray images with transferable multireceptive feature optimization. Computers in Biology and Medicine 2020;122:103869. https://doi.org/10.1016/j.compbiomed.2020.103869.

[14] Chowdhury MEH, Rahman T, Khandakar A, Mazhar R, Kadir MA, Mahbub Z Bin, et al. Can AI Help in Screening Viral and COVID-19 Pneumonia? IEEE Access 2020;8:132665–76. https://doi.org/10.1109/ACCESS.2020.3010287.

[15] Rahman T, Khandakar A, Qiblawey Y, Tahir A, Kiranyaz S, Abul Kashem S Bin, et al. Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images. Computers in Biology and Medicine 2021;132:104319. https://doi.org/10.1016/j.compbiomed.2021.104319.

[16] El-Kenawy ESM, Mirjalili S, Ibrahim A, Alrahmawy M, El-Said M, Zaki RM, et al. Advanced metaheuristics, convolutional neural networks, and feature selectors for efficient COVID-19 X-ray chest image classification. IEEE Access 2021;9:36019–37. https://doi.org/10.1109/ACCESS.2021.3061058.

[17] Wu JX, Chen PY, Li CM, Kuo YC, Pai NS, Lin CH. Multilayer Fractional-Order Machine Vision Classifier for Rapid Typical Lung Diseases Screening on Digital Chest X-Ray Images. IEEE Access 2020;8:105886–902. https://doi.org/10.1109/ACCESS.2020.3000186.

[18] Gupta A, Sheth P, Xie P. Neural architecture search for pneumonia diagnosis from chest X-rays. Scientific Reports 2022;12:1–12. https://doi.org/10.1038/s41598-022-15341-0.

[19] Dey N, Zhang YD, Rajinikanth V, Pugalenthi R, Raja NSM. Customized VGG19 Architecture for Pneumonia Detection in Chest X-Rays. Pattern Recognition Letters 2021;143:67–74. https://doi.org/10.1016/j.patrec.2020.12.010.

[20] Absar N, Mamur B, Mahmud A, Emran T Bin, Khandaker MU, Faruque MRI, et al. Development of a computer-aided tool for detection of COVID-19 pneumonia from CXR images using machine learning algorithm. Journal of Radiation Research and Applied Sciences 2022;15:32–43. https://doi.org/10.1016/j.jrras.2022.02.002.

[21] Brunese L, Mercaldo F, Reginelli A, Santone A. Explainable Deep Learning for Pulmonary Disease and Coronavirus COVID-19 Detection from X-rays. Computer Methods and Programs in Biomedicine 2020;196:105608. https://doi.org/10.1016/j.cmpb.2020.105608.

[22] Malik H, Anees T, Chaudhry MU, Gono R, Jasinski M, Leonowicz Z, et al. A Novel Fusion Model of Hand-Crafted Features With Deep Convolutional Neural Networks for Classification of Several Chest Diseases Using X-Ray Images. IEEE Access 2023;11:39243–68. https://doi.org/10.1109/ACCESS.2023.3267492.

[23] Malhotra A, Mittal S, Majumdar P, Chhabra S, Thakral K, Vatsa M, et al. Multi-task driven explainable diagnosis of COVID-19 using chest X-ray images. Pattern Recognition 2022;122:108243. https://doi.org/10.1016/j.patcog.2021.108243.

[24] Koyyada S prasad, Singh TP. An explainable artificial intelligence model for identifying local indicators and detecting lung disease from chest X-ray images. Healthcare Analytics 2023;4:100206. https://doi.org/10.1016/j.health.2023.100206.

[25] Luo L, Yu L, Chen H, Liu Q, Wang X, Xu J, et al. Deep Mining External Imperfect Data for Chest X-Ray Disease Screening. IEEE Transactions on Medical Imaging 2020;39:3583–94. https://doi.org/10.1109/TMI.2020.3000949.

[26] Zhang J, Xie Y, Pang G, Liao Z, Verjans J, Li W, et al. Viral Pneumonia Screening on Chest X-Rays Using Confidence-Aware Anomaly Detection. IEEE Transactions on Medical Imaging 2021;40:879–90. https://doi.org/10.1109/TMI.2020.3040950.

[27] Zhou J, Jing B, Wang Z, Xin H, Tong H. SODA: Detecting COVID-19 in Chest X-Rays With Semi-Supervised Open Set Domain Adaptation. IEEE/ACM Transactions on Computational Biology and Bioinformatics 2022;19:2605–12. https://doi.org/10.1109/TCBB.2021.3066331.

[28] Jyoti K, Sushma S, Yadav S, Kumar P, Pachori RB, Mukherjee S. Automatic diagnosis of COVID-19 with MCA-inspired TQWT-based classification of chest X-ray images. Computers in Biology and Medicine 2023;152:106331. https://doi.org/10.1016/j.compbiomed.2022.106331.

[29] Panetta K, Sanghavi F, Agaian S, Madan N. Automated Detection of COVID-19 Cases on Radiographs using Shape-Dependent Fibonacci-p Patterns. IEEE Journal of Biomedical and Health Informatics 2021;25:1852–63. https://doi.org/10.1109/JBHI.2021.3069798.

[30] Oliveira LLG, e Silva SA, Ribeiro LHV, de Oliveira RM, Coelho CJ, Andrade ALSS. Computer-aided diagnosis in chest radiography for detection of childhood pneumonia. International Journal of Medical Informatics 2008;77:555–64.

[31] Parveen NRS, Sathik MM. Detection of Pneumonia in chest X-ray images. Journal of X-Ray Science and Technology 2011;19:423–8. https://doi.org/10.3233/XST-2011-0304.

[32] Melendez J, Van Ginneken B, Maduskar P, Philipsen RHHM, Reither K, Breuninger M, et al. A novel multiple-instance learning-based approach to computer-aided detection of tuberculosis on chest Xrays. IEEE Transactions on Medical Imaging 2015;34:179–92. https://doi.org/10.1109/TMI.2014.2350539.

[33] Melendez J, Van Ginneken B, Maduskar P, Philipsen RHHM, Ayles H, Sánchez CI. On Combining Multiple-Instance Learning and Active Learning for Computer-Aided Detection of Tuberculosis. IEEE Transactions on Medical Imaging 2016;35:1013–24. https://doi.org/10.1109/TMI.2015.2505672.

[34] Li X, Luo S, Hu Q, Li J, Wang D, Chiong F. Automatic lung field segmentation in x-ray radiographs using statistical shape and appearance models. Journal of Medical Imaging and Health Informatics 2016;6:338–48. https://doi.org/10.1166/jmihi.2016.1714.

[35] Mohammed MA, Abdulkareem KH, Al-Waisy AS, Mostafa SA, Al-Fahdawi S, Dinar AM, et al. Benchmarking Methodology for Selection of Optimal COVID-19 Diagnostic Model Based on Entropy and TOPSIS Methods. IEEE Access 2020;8:99115–31. https://doi.org/10.1109/ACCESS.2020.2995597.

[36] Casiraghi E, Malchiodi D, Trucco G, Frasca M, Cappelletti L, Fontana T, et al. Explainable Machine Learning for Early Assessment of COVID-19 Risk Prediction in Emergency Departments. IEEE Access 2020;8:196299–325. https://doi.org/10.1109/ACCESS.2020.3034032.

[37] Namazi H, Kulish V V. COMPLEXITY-BASED CLASSIFICATION of the CORONAVIRUS DISEASE (COVID-19). Fractals 2020;28. https://doi.org/10.1142/S0218348X20501145.

[38] Ortiz-Toro C, Garcia-Pedrero A, Lillo-Saavedra M, Gonzalo-Martin C. Automatic detection of pneumonia in chest X-ray images using textural features. Computers in Biology and Medicine 2022;145:105466.

[39] Al-Zyoud W, Erekat D, Saraiji R. COVID-19 chest X-ray image analysis by threshold-based segmentation. Heliyon 2023;9:e14453. https://doi.org/10.1016/j.heliyon.2023.e14453.

[40] Ying X, Liu H, Huang R. COVID-19 chest X-ray image classification in the presence of noisy labels. Displays 2023;77:102370. https://doi.org/10.1016/j.displa.2023.102370.

[41] Iannaccone PM, Khokha M. Fractal geometry in biological systems: an analytical approach. CRC Press; 1996.

[42] Nayak SR, Mishra J. Analysis of medical images using fractal geometry. Research Anthology on Improving Medical Imaging Techniques for Analysis and Intervention, IGI Global; 2023, p. 1547–62.

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine (CIBM), Vol. 182, November 2024. Final version available at DOI: https://doi.org/10.1016/j.compbiomed.2024.109093

[43] Chen C-C, DaPonte JS, Fox MD. Fractal feature analysis and classification in medical imaging. IEEE Transactions on Medical Imaging 1989;8:133–42.

[44] Chen J, Luo Z, Zhang Z, Huang F, Ye Z, Takiguchi T, et al. Polar Transformation on Image Features for Orientation-Invariant Representations. IEEE Transactions on Multimedia 2019;21:300–13. https://doi.org/10.1109/TMM.2018.2856121.

[45] Marusina MY, Mochalina AP, Frolova EP, Satikov VI, Barchuk AA, Kuznetcov VI, et al. MRI image processing based on fractal analysis. Asian Pacific Journal of Cancer Prevention 2017;18:51–5. https://doi.org/10.22034/APJCP.2017.18.1.51.

[46] Abbas M, Iqbal H, De la Sen M. Generation of julia and mandelbrot sets via fixed points. Symmetry 2020;12:1–19. https://doi.org/10.3390/sym12010086.

[47] Raj N, Gharineiat Z. Evaluation of multivariate adaptive regression splines and artificial neural network for prediction of mean sea level trend around northern australian coastlines. Mathematics 2021;9. https://doi.org/10.3390/math9212696.

[48] Lu R, Duan T, Wang M, Liu H, Feng S, Gong X, et al. The application of multivariate adaptive regression splines in exploring the influencing factors and predicting the prevalence of hba1c improvement. Annals of Palliative Medicine 2021;10:1296–303. https://doi.org/10.21037/apm-19-406.

[49] Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, p. 2097–106.

[50] Summers R. Nih chest x-ray dataset of 14 common thorax disease categories. NIH Clinical Center: Bethesda, MD, USA 2019.

[51] Chowdhury MEH, Rahman T, Khandakar A, Mazhar R, Kadir MA, Mahbub Z Bin, et al. Can AI help in screening viral and COVID-19 pneumonia? Ieee Access 2020;8:132665–76.

[52] Sasikumar N, Senthilkumar M. Deep Convolutional Generative Adversarial Networks for Automated Segmentation and Detection of Lung Adenocarcinoma Using Red Deer Optimization Algorithm. Information Technology and Control 2023;52:680–92.

[53] Jennifer JS, Sharmila TS. A neutrosophic set approach on chest X-rays for automatic lung infection detection. Information Technology and Control 2023;52:37–52.

[54] Liu F, Tian Y, Chen Y, Liu Y, Belagiannis V, Carneiro G. ACPL: Anti-curriculum pseudo-labelling for semi-supervised medical image classification. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, p. 20697–706.

[55] Rahman Siddiquee MM, Shah J, Wu T, Chong C, Schwedt T, Li B. HealthyGAN: Learning from Unannotated Medical Images to Detect Anomalies Associated with Human Disease. International Workshop on Simulation and Synthesis in Medical Imaging, 2022, p. 43–54.

[56] Rehman N, Zia MS, Meraj T, Rauf HT, Damaševičius R, El-Sherbeeny AM, et al. A self-activated cnn approach for multi-class chest-related COVID-19 detection. Applied Sciences 2021;11:9023.

[57] Packhäuser K, Gündel S, Münster N, Syben C, Christlein V, Maier A. Deep learning-based patient reidentification is able to exploit the biometric nature of medical chest X-ray data. Scientific Reports 2022;12:14851.

[58] Liu Q, Yu L, Luo L, Dou Q, Heng PA. Semi-supervised medical image classification with relation-driven self-ensembling model. IEEE Transactions on Medical Imaging 2020;39:3429–40.

[59] Huang G, Liu Z, Van Der Maaten L, Weinberger KQ. Densely connected convolutional networks. Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, p. 4700–8.

[60] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, p. 770–8.

[61] Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the inception architecture for computer vision. Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, p. 2818– 26.

[62] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. ArXiv Preprint ArXiv:14091556 2014.

[63] Krizhevsky A, Sutskever I, Hinton GE. Imagenet classification with deep convolutional neural networks. Advances in Neural Information Processing Systems 2012;25.

[64] Singh S. Computer-Aided Diagnosis of Thoracic Diseases in Chest X-rays using hybrid CNN-Transformer Architecture. ArXiv Preprint ArXiv:240411843 2024.

[65] Termritthikun C, Umer A, Suwanwimolkul S, Xia F, Lee I. Explainable knowledge distillation for ondevice chest x-ray classification. IEEE/ACM Trans Comput Biol Bioinform 2023.

[66] Ashraf SMN, Mamun MA, Abdullah HM, Alam MGR. SynthEnsemble: A Fusion of CNN, Vision Transformer, and Hybrid Models for Multi-Label Chest X-Ray Classification. 2023 26th International Conference on Computer and Information Technology (ICCIT), 2023, p. 1–6.

[67] Dai T, Zhang R, Hong F, Yao J, Zhang Y, Wang Y. UniChest: Conquer-and-Divide Pre-training for Multi-Source Chest X-Ray Classification. IEEE Trans Med Imaging 2024.

[68] Li M, Yang G. Data-Free Distillation Improves Efficiency and Privacy in Federated Thorax Disease Analysis. 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology, 2023, p. 131–2.

[69] Nie W, Zhang C, Song D, Bai Y, Xie K, Liu A-A. Chest X-ray Image Classification: A Causal Perspective. International Conference on Medical Image Computing and Computer-Assisted Intervention, 2023, p. 25–35.

[70] Wang G, Wang P, Cong J, Liu K, Wei B. BB-GCN: A Bi-modal Bridged Graph Convolutional Network for Multi-label Chest X-Ray Recognition. ArXiv Preprint ArXiv:230211082 2023.

[71] Taslimi S, Taslimi S, Fathi N, Salehi M, Rohban MH. Swinchex: Multi-label classification on chest x-ray images with transformers. ArXiv Preprint ArXiv:220604246 2022.

[72] Zhou L, Bae J, Liu H, Singh G, Green J, Gupta A, et al. Lung swapping autoencoder: Learning a disentangled structure-texture representation of chest radiographs. ArXiv Preprint ArXiv:220107344 2022.

[73] Mao C, Yao L, Luo Y. Imagegcn: Multi-relational image graph convolutional networks for disease identification with chest x-rays. IEEE Transactions on Medical Imaging 2022;41:1990–2003.

[74] Ouyang X, Karanam S, Wu Z, Chen T, Huo J, Zhou XS, et al. Learning hierarchical attention for weaklysupervised chest X-ray abnormality localization and diagnosis. IEEE Transactions on Medical Imaging 2020;40:2698–710.

[75] Bose C, Basu A. Classification of COVID-19 from CXR Images in a 15-class Scenario: an Attempt to Avoid Bias in the System. ArXiv Preprint ArXiv:210912453 2021.

[76] Guendel S, Grbic S, Georgescu B, Liu S, Maier A, Comaniciu D. Learning to recognize abnormalities in chest x-rays with location-aware dense networks. Progress in Pattern Recognition, Image Analysis, Computer Vision, and Applications: 23rd Iberoamerican Congress, CIARP 2018, Madrid, Spain, November 19-22, 2018, Proceedings 23, 2019, p. 757–65.

[77] Gong X, Xia X, Zhu W, Zhang B, Doermann D, Zhuo L. Deformable gabor feature networks for biomedical image classification. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2021, p. 4004–12.

[78] Chen B, Li J, Lu G, Yu H, Zhang D. Label co-occurrence learning with graph convolutional networks for multi-label chest x-ray image classification. IEEE Journal of Biomedical and Health Informatics 2020;24:2292–302.

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine (CIBM), Vol. 182, November 2024. Final version available at DOI: https://doi.org/10.1016/j.compbiomed.2024.109093

[79] Baltruschat IM, Nickisch H, Grass M, Knopp T, Saalbach A. Comparison of deep learning approaches for multi-label chest X-ray classification. Scientific Reports 2019;9:6381.

[80] Chen B, Li J, Guo X, Lu G. DualCheXNet: dual asymmetric feature learning for thoracic disease classification in chest X-rays. Biomedical Signal Processing and Control 2019;53:101554.

[81] Wang H, Xia Y. Chestnet: A deep neural network for classification of thoracic diseases on chest radiography. ArXiv Preprint ArXiv:180703058 2018.

[82] Kumar P, Grewal M, Srivastava MM. Boosted cascaded convnets for multilabel classification of thoracic diseases in chest radiographs. Image Analysis and Recognition: 15th International Conference, ICIAR 2018, Póvoa de Varzim, Portugal, June 27--29, 2018, Proceedings 15, 2018, p. 546–52.

[83] Guan Q, Huang Y, Zhong Z, Zheng Z, Zheng L, Yang Y. Diagnose like a radiologist: Attention guided convolutional neural network for thorax disease classification. ArXiv Preprint ArXiv:180109927 2018.

[84] Ma C, Wang H, Hoi SCH. Multi-label thoracic disease image classification with cross-attention networks. Medical Image Computing and Computer Assisted Intervention--MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13--17, 2019, Proceedings, Part VI 22, 2019, p. 730– 8.

[85] Yan C, Yao J, Li R, Xu Z, Huang J. Weakly supervised deep learning for thoracic disease classification and localization on chest x-rays. Proceedings of the 2018 ACM international conference on bioinformatics, computational biology, and health informatics, 2018, p. 103–10.

[86] Rajpurkar P, Irvin J, Zhu K, Yang B, Mehta H, Duan T, et al. Chexnet: Radiologist-level pneumonia detection on chest x-rays with deep learning. ArXiv Preprint ArXiv:171105225 2017.

[87] Lella KK, Jagadeesh MS, Alphonse PJA. Artificial intelligence-based framework to identify the abnormalities in the COVID-19 disease and other common respiratory diseases from digital stethoscope data using deep CNN. Health Inf Sci Syst 2024;12:22.

[88] Fedoruk O, Klimaszewski K, Ogonowski A, Możdżonek R. Performance of GAN-based augmentation for deep learning COVID-19 image classification. AIP Conf Proc, vol. 3061, 2024.

[89] Lim MG, Lee HL. Diagnosis of COVID-19 based on Chest Radiography. ArXiv Preprint ArXiv:221213032 2022.

[90] Fan Y, Liu J, Yao R, Yuan X. COVID-19 Detection from X-ray Images using Multi-Kernel-Size Spatial-Channel Attention Network. Pattern Recognition 2021;119:108055. https://doi.org/10.1016/j.patcog.2021.108055.

[91] Woo S, Park J, Lee J-Y, Kweon IS. Cbam: Convolutional block attention module. Proceedings of the European conference on computer vision (ECCV), 2018, p. 3–19.

[92] Wang Q, Wu B, Zhu P, Li P, Zuo W, Hu Q. ECA-Net: Efficient channel attention for deep convolutional neural networks. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, p. 11534–42.

[93] Chowdhury MEH, Rahman T, Khandakar A, Mazhar R, Kadir MA, Mahbub Z Bin, et al. Can AI Help in Screening Viral and COVID-19 Pneumonia? IEEE Access 2020;8:132665–76. https://doi.org/10.1109/ACCESS.2020.3010287.

[94] Apostolopoulos ID, Mpesiana TA. Covid-19: automatic detection from x-ray images utilizing transfer learning with convolutional neural networks. Physical and Engineering Sciences in Medicine 2020;43:635–40.

[95] Ozturk T, Talo M, Yildirim EA, Baloglu UB, Yildirim O, Acharya UR. Automated detection of COVID-19 cases using deep neural networks with X-ray images. Computers in Biology and Medicine 2020;121:103792.

[96] Hu J, Shen L, Sun G. Squeeze-and-excitation networks. Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, p. 7132–41.

Preprint of the article published in the ELSEVIER journal Computers in Biology and Medicine (CIBM), Vol. 182, November 2024. Final version available at DOI: https://doi.org/10.1016/j.compbiomed.2024.109093