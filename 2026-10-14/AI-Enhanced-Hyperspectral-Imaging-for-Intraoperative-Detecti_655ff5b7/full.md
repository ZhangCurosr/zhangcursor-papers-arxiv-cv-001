# AI-Enhanced Hyperspectral Imaging for Intraoperative Detection of Low-Grade Glioma: A Clinical Study

Siri Willems<sup>a</sup>, Ishika Jain<sup>a,b</sup>, Jens De Winne<sup>a,c</sup>, Siri Luthman<sup>a</sup>, Hamed Zivari Adab<sup>a</sup>, Roeland Vandebriel<sup>a</sup>, Anna Alperovich<sup>e</sup>, Xiaohan Zhang<sup>f</sup>, Ben Stofelen<sup>a</sup>, Tom De Schepper<sup>a</sup>, Lien Solie<sup>d</sup>, Steven De Vleeschouwer<sup>d</sup>

<sup>a</sup>imec, Kapeldreef 75, Leuven, 3001, Belgium

<sup>b</sup>Department of Computer Science UAntwerp-imec, Sint-Pietersvliet 7, Antwerp, 2000, Belgium

<sup>c</sup>Department of Telecommunications and Information Processing, IPI-imec, Sint-Pietersnieuwestraat 41, Ghent, 9000, Belgium

<sup>d</sup>Research Group Experimental Neurosurgery and Neuroanatomy, Dept Neurosciences, KU Leuven, Leuven Brain Institute and Dept. Neurosurgery, UZ Leuven, Herestraat 49, Leuven, 3000, Belgium

<sup>e</sup>Carl Zeiss AG, Corporate Research & Technology, Carl-Zeiss-Str. 22, Oberkochen, 73447, Germany

<sup>f</sup>Carl Zeiss Meditec AG, Rudolf-Eber-Str. 11, Oberkochen, 73447, Germany

## Abstract

Current image-guided neuronavigation techniques, such as intraoperative MRI and ultrasound, are limited by high costs, limited sensitivity and operational complexity. There is a clear need for more eficient systems to enhance intra-operative tumor detection and improve patient outcomes for intrinsic brain tumor resection. Whereas previous intraoperative brain HSI studies have predominantly investigated high-grade gliomas, glioblastomas, or mixed-grade cohorts, this clinical study focuses specifically on low-grade gliomas (LGGs). LGGs strongly resemble normal brain under white light conditions on the operative microscope. This study characterizes spectral dif ferences between healthy and tumor tissue and provides a paired comparison of HSI- and RGB-based classification at matched tissue locations and under identical patient-level data splits. Tumor regions showed consistently lower reflectance, which may be associated with diferences in hemoglobin concentration and oxygenation. A range of classification models was evaluated. The 3D CNN, which integrates spectral and spatial features, achieved the highest median ROC-AUC of 74.5%, compared with 67.9% for the spectral-only XGBoost model and 64.0% for the RGB-based model. These results support

further investigation of HSI for low-grade glioma classification and quantify its performance relative to conventional RGB imaging in this cohort.

Keywords: Hyperspectral imaging, Deep learning, Brain tumor detection

## 1. Introduction

Brain and central nervous system (CNS) cancers are among the most fatal cancers worldwide, and their incidence is rising globally [1]. These include primary brain tumors, i.e. an irregular growth of malignant cells starting from intrinsic brain cells, and secondary tumors that metastasize to the brain from elsewhere in the body. Gliomas are a major group of primary brain tumors whose diagnosis and grading under the 2021 WHO classification depend on histological and molecular features and the presence or absence of an IDH mutation [2]. Diferent groups are low-grade gliomas (LGG, grade 1 and 2), high-grade gliomas (HGG, grade 3) and glioblastomas (GBM, grade 4). Surgical resection is the first step of the treatment, but achieving an optimal balance between complete tumor removal and preservation of healthy functional tissue remains challenging. Maximizing the extent of resection can improve patient survival, but resection near eloquent tissue may impair neurological function and postoperative quality of life [3, 4]. Distinguishing low-grade gliomas (LGGs) from healthy tissue intra-operatively is particularly dificult because LGGs can closely resemble healthy tissue, especially at the tumor periphery [5, 6].

Neurosurgery relies heavily on image-guided neuronavigation, but current methods have several limitations. Stereotactic neuronavigation uses preoperative magnetic resonance imaging (MRI) and a predefined surgical plan, and its accuracy decreases as brain shift occurs after craniotomy. Intraoperative MRI can visualize current surgical anatomy but is expensive, logistically complex, and time-consuming. Intraoperative ultrasound (US) is faster and less expensive but can be dificult to interpret and is afected by artifacts from bone and blood. Fluorescence imaging requires administration of a fluorophore and is primarily efective for high-grade gliomas, because most LGGs do not fluoresce. More eficient intraoperative guidance systems could therefore complement current techniques [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].

Hyperspectral imaging (HSI) combines conventional imaging with spectroscopy. Unlike RGB cameras, which capture three broad color bands, HSI systems acquire many distinct, contiguous bands across an instrument’s spectral range, from ultraviolet to short-wave infrared depending on the device [17, 18]. The resulting hypercube contains spatial and spectral information that enables detailed tissue analysis [19, 20]. Medical interest in HSI has increased because it can enhance fluorescent biomarkers [21] and provide rich spectral information without exogenous labels. HSI may support applications in gastroenterology, neuro-oncology, histopathology, and other medical fields [22, 23, 24, 25, 17].

Due to the limited human interpretability of hyperspectral data, interest in AI-based analysis of hyperspectral images has increased. The intraoperative neuro-oncology studies reviewed here predominantly address glioblastoma, high-grade glioma, or cohorts containing multiple tumor grades, often through pixel-wise classification of tumor and healthy brain tissue. Martin-Perez et al. compared snapshot and linescan HSI cameras using a pixel-wise random forest classifier [26]. Fabelo et al. proposed a hybrid framework combining a U-Net for parenchymal masking, a 2D CNN for spatial analysis and vessel identification, and a 1D CNN for tissue characterization [27]. Urbanos et al. evaluated a 3D CNN that jointly captures spectral and spatial features for pixel-wise glioblastoma classification and compared it with support-vector-machine and random-forest classifiers [28]. Puustinen et al. developed ML and DL methods for high-grade glioma detection [29]. Manni et al. combined PCA-based spectral dimensionality reduction, 3D convolutional layers for spatio-spectral learning, and 2D convolutional layers for spatial learning [30]. In contrast to this emphasis on high-grade, glioblastoma, or mixed-grade populations, the present study evaluates a clinical cohort focused on low-grade glioma.

In this paper, we present the results of a clinical study in which 18 patients were enrolled to investigate the diferentiation of low-grade glioma from healthy tissue using HSI. Of these patients, 17 were retained for model development after one patient was excluded because fluorescence imaging was used. Relative to the intraoperative brain HSI studies reviewed above, which predominantly investigated high-grade glioma, glioblastoma, or mixed-grade cohorts, the distinguishing focus of this study is a low-grade glioma cohort, which is the most challenging glioma entity in terms of intraoperative tumor detection. Preliminary results from five patients were published in a conference proceeding [31]. The study has two main aims:

• First, we compare this low-grade glioma dataset with the publicly available HELiCoiD dataset, which contains multiple tumor grades. We assess inter-patient and intra-patient spectral variability for diferent tissue types. HELiCoiD has previously been used to benchmark spectral imaging datasets [32]. This comparison contributes to the standardized assessment of spectral imaging for surgical guidance, particularly for low-grade glioma.

• Second, we examine the diference between HSI and conventional RGB imaging in greater depth through a paired model comparison. Multiple machine-learning and deep-learning methods are trained on the internal HSI data and compared with a model trained on RGB images acquired from the same tissue regions with a standard surgical microscope. All models use the same patient-level splits and evaluation procedure, allowing the contribution of HSI information to be assessed under matched conditions.

## 2. Data

Two datasets are used; their camera and acquisition parameters are summarized in Table 1. The first is a public dataset acquired during the European HypErspectraL Imaging Cancer Detection (HELiCoiD) project [5, 33], and the second is the internal InVivo-LGG dataset acquired during our clinical study [34].

## 2.1. InVivo-LGG dataset (internal, UZ Leuven)

## 2.1.1. Hardware setup and data acquisition

The setup was previously described by Vandebriel et al. [34] and is summarized here. An intraoperative imaging system was installed at UZ Leuven by attaching an imec Snapscan VNIR hyperspectral camera (imec, Leuven, Belgium) to the lateral co-observer of a ZEISS OPMI Pentero 900 neurosurgical microscope using a standard camera-port adapter (ZEISS camera adapter, $f = 1 0 5 ~ \mathrm { m m } )$ . The microscope’s optics and illumination channel were used for HSI acquisition. The microscope was operated at a focal distance of 250 mm and a zoom factor of 2.5 to match a typical standard-of-care field of view (Figure 1).

During each surgery, a predefined protocol was followed (Figure 1c). First, a spatial reference target printed on A4 paper was imaged with both the microscope’s integrated RGB camera and the attached HSI camera for subsequent spatial calibration. Second, a white-reference image was acquired by scanning a Zenith Polymer target (95% reflectance; SphereOptics, Germany) with the HSI camera before surgery. Images of the brain surface were acquired before, during, and after resection. To limit temporal changes, paired images were acquired sequentially: an RGB image was followed immediately by an HSI scan of the same region of interest [10, 35].

Table 1: Overview of dataset characteristics for both internal and public datasets used in the study.
<table><tr><td rowspan=1 colspan=1>Specifications</td><td rowspan=1 colspan=1>InVivo-LGG</td><td rowspan=1 colspan=1>HELiCoiD [5]</td></tr><tr><td rowspan=1 colspan=1>Tumor types</td><td rowspan=1 colspan=1>LGG</td><td rowspan=1 colspan=1>Mix of brain tumors</td></tr><tr><td rowspan=1 colspan=1>Number of patients</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>34</td></tr><tr><td rowspan=1 colspan=1>Number of hypercubes</td><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=1>61</td></tr><tr><td rowspan=1 colspan=1>Maximum spatial res-olution in pixels</td><td rowspan=1 colspan=1>1600 × 1600</td><td rowspan=1 colspan=1>741 × 1004</td></tr><tr><td rowspan=1 colspan=1>Spectral range</td><td rowspan=1 colspan=1>470-787 nm</td><td rowspan=1 colspan=1>400-1000 nm</td></tr><tr><td rowspan=1 colspan=1>Spectral resolution</td><td rowspan=1 colspan=1>10-15 nm</td><td rowspan=1 colspan=1>2-3 nm</td></tr><tr><td rowspan=1 colspan=1>Number of bands</td><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>826</td></tr><tr><td rowspan=1 colspan=1>HSI camera</td><td rowspan=1 colspan=1>imec Snapscan VNIR</td><td rowspan=1 colspan=1>Hyperspec VNIR A-Series (Headwall Pho-tonics Inc.)</td></tr><tr><td rowspan=1 colspan=1>Light source</td><td rowspan=1 colspan=1>Xenon</td><td rowspan=1 colspan=1>Quartz tungsten halogen</td></tr><tr><td rowspan=1 colspan=1>White reference</td><td rowspan=1 colspan=1>Zenith Polymer (95% re-flectance)</td><td rowspan=1 colspan=1>White tile (99% re-flectance)</td></tr><tr><td rowspan=1 colspan=1>Acquisition time</td><td rowspan=1 colspan=1>120 s</td><td rowspan=1 colspan=1>60 s</td></tr></table>

A clinical study (ID: NCT04859725 [33]) was conducted in which 18 patients were enrolled after providing written informed consent. The imaging protocol resulted in 75 intraoperative brain images. The head surgeon (S. DV), who performed all surgeries, annotated the RGB images directly after surgery using anatomical references from neuronavigational MRI, intraoperative US, and biopsies obtained at selected locations. The following classes were annotated: 1: "Healthy", 2: "Foreign object", 3: "Blood", 4: "Coagulation", 5: "HGG", 6: "LGG", 7: "Histology HGG", and 8: "Histology LGG". Unlabeled pixels were considered background. Patient 10 received fluorescence imaging and was excluded from model development and evaluation, leaving 17 patients in the modeling cohort. Patient 3 contained only healthy-tissue annotations and was retained for model training but excluded from patient-level binary classification evaluation, which requires both classes. Consequently, 16 patients contributed to the patient-level results.

![](images/b83a8bf208265a08028f3ea91e39a440a7643c73d7c19739d103a99a8029bbf4.jpg)  
Figure 1: (a) Clinical setup. (b) Reference target used to align the RGB and HSI cameras. (c) Image-acquisition protocol. (d) RGB–HSI calibration results.

## 2.1.2. Data preprocessing

As described by Vandebriel et al. [34], the spectral images were darkcorrected, white-balanced, and spectrally corrected. Dark correction and white balancing reduce sensor- and system-level noise. Spectral correction suppresses variation in the manufactured spectral bands by constructing idealized responses from linear combinations of the raw band responses. Bands with peak responses outside the microscope light source’s spectral range were discarded. The resulting data contain 104 corrected contiguous bands spanning 470–787 nm.

Annotations were created in the RGB frame of reference. To preserve the HSI data, the RGB images and corresponding delineations were registered to the HSI frame using a two-step procedure. Reference-target images were used for initial camera calibration with the scale-invariant feature transform in OpenCV. The resulting transformation was applied to the brain-surface RGB images and delineations. To compensate for possible camera movement during surgery, a second rigid registration was performed with SimpleElastix, allowing only small rotations and translations. All registrations were visually verified by overlaying the registered RGB images and delineations on the HSI images; examples are provided in Figure A.12. Misaligned cases were manually registered with the open-source ITK-SNAP software. Because illumination fallof and optical aberrations degrade image quality near the circular field-of-view boundary, all 104-band hypercubes were cropped to 800 × 800 pixels around the center. Figure 2(a) shows an RGB image, the mean HSI image across spectral bands, and the corresponding ground-truth labels.

## 2.1.3. Ground truth processing

The manual ground-truth annotations registered to the HSI images include small vascular structures, specular reflections, and edge noise. Additional processing was therefore applied to remove outliers. First, each HSI image was divided into spatially contiguous superpixels with similar spectral features. A circular mask with a diameter of 800 pixels was created, and simple linear iterative clustering (SLIC) was used to generate superpixels with a target area of 200 pixels [36].

To improve delineation quality and remove outliers, including specular reflections, vessels annotated as tissue, and noisy superpixels, variability within and between superpixels was evaluated using the following metrics:

First, only superpixels that contain histology confirmed ground truths for healthy and LGG tumor tissue are kept. If a superpixel contains a mix of diferent labels, the superpixel is removed.

Second, the average intensity $\mathrm { I } _ { S }$ within the superpixel is calculated as:

$$
I _ { s } = \frac { 1 } { n } \sum I _ { p }\tag{1}
$$

![](images/bc25618eec955fdbf50628c057c8f7c49df63c5a9aa6e129c9de2a52744b29e8.jpg)  
Figure 2: RGB image, histology-confirmed ground truth, initial superpixels, metric distri butions, filtered superpixels, and final filtered ground truth.

with $\mathrm { I } _ { p }$ being the intensity of each individual pixel, belonging to the superpixel. Superpixels with intensities outside the image-specific 5th–95th percentile range, which primarily represent dark vessels or bright specular reflections, were excluded (Figure 2b).

Third, Euclidean distance (ED) was used to quantify spectral-shape variability within each superpixel. Each pixel spectrum $\mathbf { f } _ { p }$ was first L2-normalized to remove multiplicative magnitude diferences:

$$
\tilde { \mathbf { f } } _ { p } = \frac { \mathbf { f } _ { p } } { \lVert \mathbf { f } _ { p } \rVert _ { 2 } } .\tag{2}
$$

The mean normalized spectrum of superpixel s was then calculated as

$$
\bar { \bf f } _ { s } = \frac 1 { n _ { s } } \sum _ { p = 1 } ^ { n _ { s } } \tilde { \bf f } _ { p } ,\tag{3}
$$

where $n _ { s }$ is the number of pixels in superpixel s. Finally, the superpixellevel ED was defined as the mean L2 distance between each normalized pixel spectrum and the mean normalized spectrum:

$$
d _ { s } = \frac { 1 } { n _ { s } } \sum _ { p = 1 } ^ { n _ { s } } \left. \tilde { \mathbf { f } } _ { p } - \bar { \mathbf { f } } _ { s } \right. _ { 2 } .\tag{4}
$$

Smaller $d _ { s }$ values indicate greater spectral-shape homogeneity within the superpixel.

Superpixels with a $d _ { s }$ value above the image-specific 50th percentile were excluded from the annotations for that image.

Last, the spectral angle mapper (SAM) describes the similarity between two spectral vectors as the angle between them in the spectral feature space. Smaller angles indicate greater spectral similarity. Because SAM depends on vector direction rather than magnitude, it is insensitive to multiplicative intensity diferences. For a superpixel $s ,$ the mean spectral angle between its mean spectrum $\bar { \mathbf { f } } _ { s }$ and the spectrum $\mathbf { f } _ { p }$ of each pixel $p$ is calculated in degrees as follows:

$$
S A M _ { s } = \frac { 1 8 0 } { \pi n _ { s } } \sum _ { p = 1 } ^ { n _ { s } } \operatorname { a r c c o s } \left( \frac { \bar { \mathbf { f } } _ { s } ^ { \mathsf { T } } \mathbf { f } _ { p } } { \left\| \bar { \mathbf { f } } _ { s } \right\| _ { 2 } \left\| \mathbf { f } _ { p } \right\| _ { 2 } } \right) ,\tag{5}
$$

where $n _ { s }$ is the number of pixels in superpixel s. All SAM values reported in this study are expressed in degrees. Superpixels with a $S A M _ { s }$ value above the 50th percentile of the corresponding image are excluded from the annotations, see Figure 2 (b) and (c). A full overview of the dataset, including the number of images per patient and the number of superpixels before and after filtering for each class, is provided in the supplementary material (Table B.3).

## 2.2. HELiCoiD dataset (public)

The public HELiCoiD setup difers from the system used to acquire the internal dataset [5, 33]. A Hyperspec VNIR A-Series pushbroom camera (Headwall Photonics Inc., Fitchburg, MA, USA) captured 826 spectral channels spanning 400–1000 nm with a spectral resolution of 2–3 nm. Table 1 summarizes the main setup characteristics. The public dataset was acquired during three clinical-study campaigns and contains 61 hyperspectral images from 34 patients: 10 with LGG, 3 with HGG, 15 with GBM, and 6 with other, including secondary, brain tumors. Images were processed using dark and white references as described in the original publication [5].

## 3. Method

## 3.1. Spectral characterization of glioma datasets

Spectral characterization methods are applied to investigate the extent and nature of spectral diferences between various types of tissue (healthy vs. tumor), data sets captured by diferent devices, and patient variations. The goal is to provide a comprehensive analysis of the spectral characteristics and the inherent variations both within and between these factors. This understanding can serve as a foundational step in developing a robust model for brain tumor classification, particularly for surgical resection.

## 3.1.1. Spectral characteristics of healthy and brain tumor tissue across datasets

Average spectra per tissue type are calculated first per patient and subsequently averaged over all patients in a dataset for both the internal and public dataset. The InVivo-LGG dataset consists of 104 spectral bands, while the HELiCoiD dataset contains 826 bands. SAM was calculated to quantify the diferences between the spectral signatures of healthy and tumor pixels for each dataset. For a simple and transparent equal-dimensional comparison, we selected the first 100 bands from the InVivo-LGG dataset and paired each band with the closest wavelength in the HELiCoiD dataset, resulting in 100 matched bands from each dataset before calculating SAM. This pragmatic matching procedure was chosen to keep the descriptive comparison straightforward; it does not constitute full harmonization of the devices’ spectral resolutions or response functions.

## 3.1.2. Inter-patient versus intra-patient variability

The following calculations were performed separately for each dataset and tissue class $t \in$ {healthy, tumor}. Let $\mathbf { x } _ { q t i } \in \mathbb { R } ^ { B }$ denote the B-band spectrum of pixel i from patient $q$ and tissue class $t ,$ and let $N _ { q t }$ be the number of pixels available for that patient and class. The patient-level mean spectrum was

$$
\pmb { \mu } _ { q t } = \frac { 1 } { N _ { q t } } \sum _ { i = 1 } ^ { N _ { q t } } \mathbf { x } _ { q t i }\tag{6}
$$

For two spectral vectors a and b, SAM in degrees was defined as

$$
\theta ( \mathbf { a } , \mathbf { b } ) = { \frac { 1 8 0 } { \pi } } \operatorname { a r c c o s } \left( { \frac { \mathbf { a } ^ { \mathsf { T } } \mathbf { b } } { \| \mathbf { a } \| _ { 2 } \| \mathbf { b } \| _ { 2 } } } \right) .\tag{7}
$$

Intra-patient variability for patient q and tissue class t was the mean angular distance between each pixel spectrum and the corresponding patient-level mean spectrum:

$$
\mathrm { I N T R A } _ { q t } = \frac { 1 } { N _ { q t } } \sum _ { i = 1 } ^ { N _ { q t } } \theta \bigl ( \mathbf { x } _ { q t i } , \pmb { \mu } _ { q t } \bigr ) .\tag{8}
$$

Thus, pixels were weighted equally within each patient. Dataset-level intrapatient distributions were formed from the resulting patient-level values, with each patient weighted equally (Figure 3, left). Let $\mathcal { P } _ { t }$ be the set of patients containing tissue class t and let $M _ { t } = | \mathcal { P } _ { t } |$ . Inter-patient variability for a unique pair of patients $q < r$ was

$$
D _ { q r t } = \theta \big ( \pmb { \mu } _ { q t } , \pmb { \mu } _ { r t } \big ) , \qquad q , r \in \mathscr { P } _ { t } .\tag{9}
$$

There were $C _ { t } = { \binom { M _ { t } } { 2 } }$ unique patient pairs for tissue class t. Their equally weighted mean defined the aggregate inter-patient variability:

$$
\mathrm { I N T E R } _ { t } = \frac { 1 } { C _ { t } } \sum _ { \stackrel { q , r \in \mathcal { P } _ { t } } { q < r } } D _ { q r t } .\tag{10}
$$

Patient-level mean spectra were compared without weighting them by $N _ { q t }$ and each unique patient pair contributed equally to $\mathrm { I N T E R } _ { t }$ (Figure 3, right). Higher values indicate greater spectral variation within or between patients, respectively. The wavelength-matching procedure described above was applied before cross-dataset comparisons.

![](images/2339cab55b5dd3ea18df63f863fd2cb3bbc2259212082e44ac3f8c6caf847d3f.jpg)  
Figure 3: Visual overview for calculating the intra-patient variability (left) and interpatient variability (right).

## 3.2. Classification methods

Diferent classification methods were developed and benchmarked to distinguish healthy tissue from low-grade glioma tissue. Each superpixel was represented by a surrounding $4 0 \times 4 0 \times 1 0 4$ crop, with values outside the superpixel masked to zero. First, a spectral classifier was developed to associate a spectral signature with healthy or tumor tissue. The normalized mean spectrum of each superpixel, comprising 104 features, was used to train an XGBoost classifier. This method uses spectral information but no additional spatial context.

Subsequently, the superpixel cube, consisting of a group of neighboring spectra, is processed as a whole, thus allowing for spectral analysis in a local spatial neighborhood. Methods such as 2D and 3D CNNs can capture variability of spectra along this cube and thus incorporate spatial information in addition to spectral information. An overview of the diferent methods is visualized in Figure 4.

The two-dimensional CNN using all bands (2D CNN–All channels) consists of 2D kernels learning spatial correlations along the full channel (spectral) dimension. The CNN contains 3 alternating blocks consisting of convolutional layers with a kernel size of (3x3), batch normalization, ReLU activation function and a max pooling layer with a kernel size of (2x2). At each iteration of this block, the number of feature maps is doubled. These 3 blocks are followed by two fully connected layers and a last classification layer with sigmoid activation function for final classification.

A second 2D CNN, similar to the first, is implemented with an additional spectral compression layer at the beginning. A convolutional layer with a kernel size of 1x1 is used to reduce the input to 3 or 12 channels before entering a similar 2D CNN as explained above. These models are referred to as 2D-CNN-ch3 and 2D-CNN-ch12.

Three-dimensional CNNs consist of 3D kernels learning a mix of spatial and spectral features in a local neighbourhood. The CNN contains 3 alternating blocks consisting of convolutional layers with a kernel size of (9x3x3), batch-normalization, ReLU activation function and a max pooling layer with a kernel size of (2x2x2). At each iteration of this block, the amount of feature maps is doubled. These 3 blocks are followed by two fully connected layers and a last classification layer with sigmoid activation function for final classification. This model is further in the text referred to as 3D CNN.

![](images/679ad5aed8ec91eb9ef62d3b1928802f33bb57549e19be73f177ec4c6b1ce952.jpg)  
(a) Neural-network building blocks

![](images/e365911e67e76e329f9e84e7c69152328dd985c2f0800bbd3a8a37eb986ba5d2.jpg)  
(b) Neural-network architectures  
Figure 4: Overview of (a) individual neural-network building blocks and (b) the evaluated neural-network architectures.

## 4. Experimental design

## 4.1. Study design

To develop and evaluate the ML/DL methods explained in the previous section, we used nested, patient-level cross-validation on the 17-patient modeling cohort after excluding patient 10. The 16 patients containing both healthy and LGG annotations were divided once into six outer folds at the patient level, such that all images and superpixels from a patient remained in the same fold. The same outer-fold assignments were used for every HSIand RGB-based model to enable paired modality comparisons. Each outer fold served once as the independent test set, resulting in six outer runs and ensuring that every two-class patient was evaluated exactly once. Patient 3, who contained only healthy-tissue annotations, was restricted to model development and was never included in an outer test fold. Figure 5 shows the patient and superpixel distribution across the six outer folds.

Within each outer run, the five non-test folds were used for five-fold inner cross-validation and hyperparameter optimization, including batch size, number of features, and learning rate. Hyperparameters were selected independently within each outer run; consequently, the selected configuration could difer between outer runs. For the selected configuration, the five innerfold models were applied to the held-out outer test fold, and their predicted probabilities were averaged. This cross-validated ensemble reduces dependence on any single inner split while preserving complete separation of the outer test patients from model and hyperparameter selection. The purpose of this design was not to identify one universally best architecture, but to compare how efectively models using RGB and HSI inputs extract information under identical patient splits and evaluation conditions.

## 4.2. Experiments

Several AI architectures were optimized and benchmarked using HSI data. To compare HSI- and RGB-based performance, a 2D CNN was also developed for RGB input and optimized using the same patient-level data splits as the HSI-based models.

## 4.3. Training details

All models were trained for 50 epochs using binary cross-entropy loss and the Adam optimizer. A learning-rate scheduler reduced the learning rate by a factor of 2 every 10 epochs. During each epoch, a balanced sampler selected 100 superpixels per class and patient for patients containing both labels, ensuring equal sampling of healthy and tumor tissue across patients. Patient 3 contributed only its available healthy-tissue superpixels.

Distribution of healthy and tumor tiles for each experimental run  
![](images/77a71aa66ad26ba709e68f9b8aa76f028c2fbffda10b5fe68ee815cc852d55d8.jpg)  
Figure 5: Overview of the patient and superpixel distribution across the six outer folds of the nested patient-level cross-validation design. Each fold is used once as the independent outer test set, while the remaining five folds are used for inner cross-validation and model optimization.

## 4.4. Evaluation

Model performance was evaluated using F1 score, balanced accuracy (BACC), accuracy (ACC), precision (PREC), recall (REC), and ROC-AUC. Metric definitions are provided in the supplementary material (Section Appendix C). ROC-AUC was computed from continuous predicted probabilities before thresholding. For F1, BACC, ACC, PREC, and REC, probabilities greater than or equal to 0.5 were classified as LGG, while probabilities below 0.5 were classified as healthy tissue. Metrics were calculated separately for each eligible test patient using all retained superpixels. For each metric, every HSI-based model was compared separately with 2DCNN-RGB using a paired Wilcoxon signed-rank test. The paired unit was the patient $( n = 1 6 )$ each pair comprised the metric value from an HSI-based model and the corresponding value from 2DCNN-RGB for the same patient. Because each patient appeared in exactly one held-out outer test fold, each patient contributed one paired observation to a given model–metric comparison. A diference was considered statistically significant when $p < 0 . 0 5$

## 5. Results

## 5.1. Spectral characterization

Figure 6 shows the mean reflectance spectra of healthy and tumor tissue across patients in both datasets. Although the datasets were acquired with diferent HSI systems and have diferent spectral resolutions and ranges, they share several spectral features. Peaks and troughs occur at similar wavelengths, including a peak near 505 nm, features near 560 nm, and a trough near 720 nm. These features may be related to the absorption characteristics of deoxyhemoglobin (Hb) and oxyhemoglobin $\left( \mathrm { H b O _ { 2 } } \right)$ . Near the isosbestic point at 584 nm, where Hb and $\mathrm { H b O _ { 2 } }$ have equal absorption, the healthyand tumor-tissue spectra are similar [37]. In both datasets, tumor tissue has lower reflectance than healthy tissue across the measured range. The HE-LiCoiD data also show a tumor-associated trough near 720 nm that is less prominent in healthy tissue. HELiCoiD LGG and GBM patients have SAM values of $4 . 9 ^ { \circ }$ and 4.2<sup>◦</sup>, respectively, indicating diferences between healthyand tumor-tissue signatures. The corresponding signatures in InVivo-LGG are more similar $( \mathrm { S A M } = 0 . 9 ^ { \circ } )$ , which may make classification more dificult.

For intra-patient variability (Figure 7), tumor tissue has higher SAM values than healthy tissue in all datasets, with a statistically significant difference reported only for InVivo-LGG. This finding indicates greater withinpatient spectral variability in tumor tissue. Mean inter-patient variability is higher in HELiCoiD than in InVivo-LGG for both tissue classes. Across datasets, tumor tissue has higher inter-patient SAM values than healthy tissue $\left( p < 0 . 0 5 \right)$ .

![](images/48f7c8e76481cfb31820df567260b13f0f7af786999b7cca75623f9405879764.jpg)

![](images/afad5f9904322f9456f7aa37d82c02ff94e757886ed21e6aa27e7732ec45f0b1.jpg)

![](images/ccdbab918536d459647471a1e4616ca7cc74662f4ccbfc63ea33be6b2e61bb56.jpg)  
Figure 6: Mean spectra of healthy (green) and tumor (blue) tissue in the internal InVivo LGG dataset and the public HELiCoiD dataset.

## 5.2. Classification results

Table 2 summarizes the performance of all models, calculated per patient over the superpixels from the test cohorts across the diferent runs. Among the HSI-based models, XGBoost achieved the highest median F1 score, accuracy, and recall; the 3D CNN achieved the highest median balanced accuracy and ROC-AUC; and 2DCNN-ch3 achieved the highest median precision. These numerical diferences between HSI-based model families were not statistically significant. Overall, the HSI-based models performed significantly better than the RGB-based model, demonstrating the added value of hyperspectral information for this classification task.

Table 2: Test-set classification metrics summarized per patient as the median $\pm \ : 9 5 \%$ confidence interval. Boldface indicates a statistically significant paired diference between that HSI-based model and 2DCNN-RGB for the corresponding metric. No statistically significant diferences were observed among the HSI-based model families.
<table><tr><td>Models</td><td>#patients</td><td>F1</td><td>ACC</td><td>b. ACC</td><td>AUC-ROC</td><td>PREC</td><td>RECALL</td></tr><tr><td>XGB</td><td>16</td><td> $\overline { { { \bf 0 . 6 8 \pm 0 . 0 9 } } }$ </td><td> $\overline { { { \bf 0 . 6 8 \pm 0 . 0 6 } } }$ </td><td> $\overline { { { \bf 0 . 6 8 \pm 0 . 0 6 } } }$ </td><td> $\overline { { { \bf 0 . 6 8 \pm 0 . 0 6 } } }$ </td><td> $\overline { { 0 . 6 8 \pm 0 . 1 3 } }$ </td><td> $\overline { { { \bf 0 . 7 4 \pm 0 . 0 8 } } }$ </td></tr><tr><td>2DCNN-All</td><td>16</td><td> $0 . 6 0 \pm 0 . 1 1$ </td><td> $0 . 6 4 \pm 0 . 0 7$ </td><td> $\mathbf { 0 . 6 7 \pm 0 . 0 6 }$ </td><td> ${ \bf 0 . 7 3 \pm 0 . 0 7 }$ </td><td> $0 . 6 7 \pm 0 . 1 5$ </td><td> $0 . 6 6 \pm 0 . 1 5$ </td></tr><tr><td>2DCNN-ch12</td><td>16</td><td> $0 . 5 8 \pm 0 . 1 3$ </td><td> $0 . 6 3 \pm 0 . 0 9$ </td><td> $\mathbf { 0 . 6 7 \pm 0 . 0 8 }$ </td><td> $\mathbf { 0 . 7 4 \ : \pm { \ : 0 . 0 9 } }$ </td><td> $0 . 6 8 \pm 0 . 1 5$ </td><td> $0 . 6 4 \pm 0 . 1 6$ </td></tr><tr><td>2DCNN-ch3</td><td>16</td><td> $0 . 5 9 \pm 0 . 1 1$ </td><td> $0 . 6 3 \pm 0 . 0 8$ </td><td> $0 . 6 7 \pm 0 . 0 8$ </td><td> $0 . 7 4 \pm 0 . 1 0$ </td><td> $0 . 6 9 \pm 0 . 1 3$ </td><td> $0 . 6 3 \pm 0 . 1 6$ </td></tr><tr><td>3DCNN</td><td>16</td><td> $0 . 6 4 \pm 0 . 1 0$ </td><td> $0 . 6 6 \pm 0 . 0 8$ </td><td> $\mathbf { 0 . 6 9 \ : \pm { \ : 0 . 0 7 } }$ </td><td> ${ \bf 0 . 7 5 \pm 0 . 0 8 }$ </td><td> $0 . 6 9 \pm 0 . 1 3$ </td><td> $0 . 7 0 \pm 0 . 1 4$ </td></tr><tr><td>2DCNN-RGB</td><td>16</td><td> $\overline { { 0 . 5 6 \pm 0 . 0 9 } }$ </td><td> $\overline { { 0 . 5 9 \pm 0 . 0 5 } }$ </td><td> $\overline { { 0 . 5 9 \pm 0 . 0 6 } }$ </td><td> $\overline { { 0 . 6 4 \pm 0 . 0 8 } }$ </td><td> $\overline { { 0 . 6 3 \pm 0 . 1 3 } }$ </td><td> $\overline { { 0 . 5 5 \pm 0 . 1 0 } }$ </td></tr></table>

![](images/97a603a5fb2464e94dcdb23233ad3db81cf0c1130b76166b26b7e39c4f88f442.jpg)

![](images/7c685c3c08e638d9ffd2f3fbdc65ed0f65ed23f9fd0973e9758e22fa97159c00.jpg)  
Figure 7: SAM values in degrees for (a) intra-patient variability and (b) inter-patient variability. In (a), each dot represents a patient, with green dots indicating healthy tissue and blue dots indicating tumor tissue; highlighted circles represent median values. In (b), each dot represents a pair of patients.

![](images/176fdb7747b0eee0b0244160474dc94cce7c1e7c38872a0b6bc0e41ede1c41c2.jpg)  
Figure 8: Scatterplot of patient-level F1 and AUC-ROC values obtained from the six outer test folds.

Figure 8 compares patient-level F1 and AUC-ROC scores for the RGB model on the x-axis and the XGB, 2DCNN-All and 3D CNN on the y-axis. The HSI-based models achieve higher performances for most of the patients. Predictions from the XGBboost and RGB model are shown in Figures 9, 10, and 11.

## 6. Discussion

This study extends the intraoperative brain HSI literature in two specific ways. First, it reports a clinical cohort focused on low-grade glioma, complementing prior studies that predominantly evaluated high-grade glioma, glioblastoma, or mixed-grade cohorts. Second, it examines the HSI–RGB diference through paired acquisitions, matched tissue locations, identical patient-level splits, and a common evaluation procedure. Spectral characterization of two datasets acquired with diferent clinical HSI systems also revealed similar features in the shapes of their reflectance curves despite dif ferences in spatial and spectral resolution.

Measured tissue reflectance is influenced by hemoglobin absorption, blood volume, and oxygenation in the visible and near-infrared range [37]. Accordingly, features associated with Hb and $\mathrm { H b O _ { 2 } }$ are present in both InVivo-LGG and HELiCoiD. LGG tissue shows lower reflectance than healthy brain tissue, particularly toward the near-infrared. Diferences in tissue water content, edema, and vascularity may contribute to this observation [38, 39, 40, 41], but these properties were not measured directly in this study. Microvascular proliferation is also more pronounced in high-grade gliomas [42, 43]. These biological explanations should therefore be considered hypotheses rather than demonstrated mechanisms.

Patient 18  
Patient 13  
![](images/b1ae6a0a1f777b59b5acc6724a667941109d91823b7afa30a14fff944c5f5813.jpg)  
Figure 9: Visual examples of patient 13

![](images/297847b7c479e7206361518957c1dc423dc4fc46b280daa0995f1d2cafc2993e.jpg)  
Figure 10: Visual examples of patient 18

![](images/3911b99af7c749e6fe69436ed55c163ffabeb07caeb1714a776016542c652ea7.jpg)  
Figure 11: Visual examples of patient 2

HELiCoiD LGG spectra show a trough near 720 nm that could be associated with deoxyhemoglobin absorption. Although this feature might be consistent with localized hypoxia [44, 45], no physiological measurements were available to validate that interpretation. Its absence from InVivo-LGG could instead reflect diferences in spectral resolution, instrument response, illumination, tissue geometry, preprocessing, clinical conditions or more rigid patient diagnosis according to the WHO 2021 classification.

Non-biological factors also influence reflectance spectra. Ambient conditions may vary during acquisition, and the larger HELiCoiD pushbroom system may be more sensitive to environmental fluctuations than the imec Snapscan VNIR system. Conversely, HELiCoiD has higher spectral resolution and more densely sampled signatures, whereas InVivo-LGG has broader spectral response functions. Diferences in spectral and spatial resolution, lenses, illumination, instrument response, acquisition protocol, and groundtruth procedures all complicate direct comparison between the datasets.

Reliable ground truth remains challenging in medical imaging. HELi-

CoiD used semi-automatic labeling: experts selected image locations with clear healthy-tissue, vessel, or tumor labels, after which pixels with similar spectral characteristics were assigned to those groups. This procedure increases separation between the resulting classes. In contrast, the present study broadly delineated regions using preoperative imaging and histological information from selected locations and then removed outliers, which may reduce interclass variation. Ideally, all imaging information would be accurately registered to the intraoperative frame of reference and validated against spatially matched histopathology, particularly at tumor margins.

Moreover, the study benchmarks several ML and DL model families. No statistically significant diferences were observed among the HSI-based model families, indicating that no single architecture can be considered uniformly superior in this cohort. Instead, the architectures ofer complementary benefits. XGBoost processes the complete mean spectrum of a superpixel and can model nonlinear interactions between spectrally distant wavelengths, but it does not use spatial context. In contrast, the 3D CNN jointly learns local spectral and spatial correlations within a superpixel cube. Under the paired evaluation design, the HSI-based models performed significantly better overall than the RGB-based model, supporting the added value of hyperspectral information for this low-grade glioma classification task while remaining specific to the selected tissue regions and cohort studied here. Previous studies have demonstrated the relevance of HSI for brain tumor detection, particularly for glioblastoma [5, 33, 30], while related work outside medicine has explored combined spectral and spatial modeling [46, 47, 48, 49, 50]. A transformer-based architecture could combine long-range spectral interactions with spatial context more directly; however, such models generally require substantially more training data than are available in the present cohort and should therefore be investigated in larger studies. Czempiel et al. [51] reconstructed spectral images from RGB images using deep learning and the public HeiPorSPECTRAL pig dataset [52]. This finding suggests that RGB images may contain information useful for spectral reconstruction. The present study evaluated direct RGB classification and did not investigate a two-stage RGB-to-HSI reconstruction and classification pipeline.

## 7. Limitations

This study has several limitations. A single surgeon provided the groundtruth delineations of tumor and healthy-tissue regions, informed by pathology at selected locations. No inter-rater or intra-rater agreement measurements were obtained. The classification task included only healthy and tumor tissue and excluded vessels and other background classes that would be important in clinical practice. Future studies should include these classes and explicitly evaluate specular reflections and small vessels.

An important limitation of the RGB–HSI comparison is that sample definition and filtering were based on the HSI data. SLIC superpixels were generated from the hyperspectral images, and intensity, ED, and SAM thresholds were subsequently used to retain spectrally homogeneous regions. This procedure removes ambiguous and spectrally heterogeneous observations before model evaluation and may therefore favor HSI-based models. Evaluating the RGB models at the same retained spatial locations ensures a paired comparison but does not eliminate this selection bias, because the inclusion criteria were still determined from HSI. Consequently, the observed performance difference between HSI and RGB should be interpreted in the context of this HSI-derived sample selection. Future work should quantify the efect through ablation analyses without spectral filtering or define evaluation regions using a modality-independent procedure.

The filtering procedure also excludes mixed-label superpixels at tissue boundaries, vessels, blood-contaminated regions, specular reflections, very dark regions, and spectrally heterogeneous tissue. These conditions are common in the intraoperative field, and mixed or heterogeneous regions may include tumor margins that are particularly relevant for surgical guidance. Therefore, the present results demonstrate classification performance for selected, spectrally homogeneous tissue regions and should not be interpreted as evidence of general intraoperative tumor detection or reliable margin delineation under all surgical conditions. Future studies should explicitly evaluate these excluded conditions and tumor-boundary regions.

In addition, filtering may afect the spectral characterization itself. Excluding observations outside the image-specific 5th–95th intensity percentiles and subsequently removing superpixels above the image-specific 50th percentiles for ED and SAM can alter estimated tissue reflectance distributions and reduce measured within-class variability. The supplementary material reports the total number of superpixels before and after filtering, but it does not quantify retention after each individual filtering step. Future analyses should report stepwise sample retention and repeat the spectral comparisons before and after filtering to assess the sensitivity of the observed healthy– tumor diferences to these choices.

Finally, the extent of spatial information used by the neural networks is limited because only superpixels are classified. Broadening the scope towards segmentation-based approaches that include both local and global spatial information in addition to spectral information is an important direction for future work.

## 8. Conclusion

This clinical study focuses on low-grade glioma, complementing intraoperative HSI studies centered on high-grade glioma, glioblastoma, or mixedgrade cohorts. It also provides a paired HSI–RGB comparison using matched tissue locations and identical patient-level evaluation splits. Despite diferences between the two HSI systems, both datasets show similar spectral features in brain tissue, and cross-dataset characterization indicates greater inter-patient spectral variability in tumor tissue than in healthy tissue. The 3D CNN, which evaluates spectral signatures within a local spatial context, achieved a median ROC-AUC of 74.5% and a median F1 score of 63.7%, compared with 64.0% and 55.6%, respectively, for the RGB-based model. These findings quantify the HSI–RGB performance diference for selected tissue regions in this low-grade glioma cohort.

## 9. Acknowledgments

This work has received funding from the Flemish Government under the “Onderzoeksprogramma Artificiele Intelligentie (AI) Vlaanderen” programme.

## References

[1] I. A. F. R. on Cander, Global Cancer Observatory: Cancer Today. World Fact Sheet. URL https://gco.iarc.fr/today/en

[2] D. N. Louis, A. Perry, P. Wesseling, D. J. Brat, I. A. Cree, D. Figarella-Branger, C. Hawkins, H. K. Ng, S. M. Pfister, G. Reifenberger, R. Sofietti, A. von Deimling, D. W. Ellison, The 2021 WHO Classification of Tumors of the Central Nervous System: a summary, Neuro-Oncology 23 (8) (2021) 1231–1251. doi:10.1093/neuonc/noab106.

[3] N. Sanai, M. S. Berger, Glioma extent of resection and its impact on patient outcome, Neurosurgery 62 (4) (2008) 753. doi:10.1227/01.neu.0000318159.21731.cf.

[4] W. Stummer, J.-C. Tonn, H. M. Mehdorn, U. Nestler, K. Franz, C. Goetz, A. Bink, U. Pichlmeier, Counterbalancing risks and gains from extended resections in malignant glioma surgery: a supplemental analysis from the randomized 5-aminolevulinic acid glioma resection studySection: Journal of Neurosurgery (Mar. 2011). doi:10.3171/2010.3.JNS097.

[5] R. Leon, H. Fabelo, S. Ortega, I. A. Cruz-Guerrero, D. U. Campos-Delgado, A. Szolna, J. F. Piñeiro, C. Espino, A. J. O’Shanahan, M. Hernandez, D. Carrera, S. Bisshopp, C. Sosa, F. J. Balea-Fernandez, J. Morera, B. Clavo, G. M. Callico, Hyperspectral imaging benchmark based on machine learning for intraoperative brain tumour detection, npj Precision Oncology 7 (1) (2023) 1–17. doi:10.1038/s41698-023-00475-9.

[6] A. S. Jakola, A. J. Skjulsvik, K. S. Myrmel, K. Sjåvik, G. Unsgård, S. H. Torp, K. Aaberg, T. Berg, H. Y. Dai, K. Johnsen, R. Kloster, O. Solheim, Surgical resection versus watchful waiting in low-grade gliomas, Annals of Oncology 28 (8) (2017) 1942–1948. doi:10.1093/annonc/mdx230.

[7] I. J. Gerard, M. Kersten-Oertel, K. Petrecca, D. Sirhan, J. A. Hall, D. L. Collins, Brain shift in neuronavigation of brain tumors: A review, Medical Image Analysis 35 (2017) 403–420. doi:10.1016/j.media.2016.08.007.

[8] N. Lakomkin, C. G. Hadjipanayis, Fluorescence-guided surgery for highgrade gliomas, Journal of Surgical Oncology 118 (2) (2018) 356–361. doi:10.1002/jso.25154.

[9] R. Sastry, W. L. Bi, S. Pieper, S. Frisken, T. Kapur, W. Wells, A. J. Golby, Applications of ultrasound in the resection of brain tumors, Journal of Neuroimaging: Oficial Journal of the American Society of Neuroimaging 27 (1) (2017) 5–15. doi:10.1111/jon.12382.

[10] N. Verburg, P. C. de Witt Hamer, State-of-the-art imaging for glioma surgery, Neurosurgical Review 44 (3) (2021) 1331–1343. doi:10.1007/s10143-020-01337-9.

[11] M. Zhang, X. Xiao, G. Gu, P. Zhang, W. Wu, Y. Wang, C. Pan, L. Wang, H. Li, Z. Wu, J. Zhang, L. Zhang, Role of neuronavigation in the surgical management of brainstem gliomas, Frontiers in Oncology 13 (May 2023). doi:10.3389/fonc.2023.1159230.

[12] M. Barberio, S. Benedicenti, M. Pizzicannella, E. Felli, T. Collins, B. Jansen-Winkeln, J. Marescaux, M. G. Viola, M. Diana, Intraoperative guidance using hyperspectral imaging: A review for surgeons, Diagnostics 11 (11) (2021) 2066, number: 11 Publisher: Multidisciplinary Digital Publishing Institute. doi:10.3390/diagnostics11112066.

[13] H.-A. Leroy, C. Delmaire, E. Le Rhun, E. Drumez, J.-P. Lejeune, N. Reyns, High-field intraoperative MRI and glioma surgery: results after the first 100 consecutive patients, Acta Neurochirurgica 161 (7) (2019) 1467–1474. doi:10.1007/s00701-019-03920-6.

[14] H. K. Bø, O. Solheim, K.-A. Kvistad, E. M. Berntsen, S. H. Torp, A. J. Skjulsvik, I. Reinertsen, D. H. Iversen, G. Unsgård, A. S. Jakola, Intraoperative 3D ultrasound–guided resection of difuse low-grade gliomas: radiological and clinical results, Journal of Neurosurgery 132 (2) (2020) 518–529. doi:10.3171/2018.10.JNS181290.

[15] W. Stummer, U. Pichlmeier, T. Meinel, O. D. Wiestler, F. Zanella, H.- J. Reulen, ALA-Glioma Study Group, Fluorescence-guided surgery with 5-aminolevulinic acid for resection of malignant glioma: a randomised controlled multicentre phase III trial, The Lancet. Oncology 7 (5) (2006) 392–401. doi:10.1016/S1470-2045(06)70665-9.

[16] A. K. Almekkawi, T. Y. El Ahmadieh, E. M. Wu, A. M. Abunimer, K. R. Abi-Aad, S. G. Aoun, A. R. Plitt, N. E. El Tecle, T. Patel, W. Stummer, B. R. Bendok, The Use of 5-Aminolevulinic Acid in Low-Grade Glioma Resection: A Systematic Review, Operative Neurosurgery 19 (1) (2020) 1–8. doi:10.1093/ons/opz336.

[17] S. Karim, A. Qadir, U. Farooq, M. Shakir, A. A. Laghari, Hyperspectral imaging: a review and trends towards medical imaging, Current Medical Imaging Reviews 19 (5) (2023) 417–427.

[18] IEEE Standards Association. URL https://standards.ieee.org/ieee/4001/7314/

[19] S. L. Jacques, Optical properties of biological tissues: a review, Physics in Medicine and Biology 58 (11) (2013) R37–61. doi:10.1088/0031- 9155/58/11/R37.

[20] M. Halicek, H. Fabelo, S. Ortega, G. M. Callico, B. Fei, In-Vivo and Ex-Vivo Tissue Analysis through Hyperspectral Imaging Techniques: Revealing the Invisible Features of Cancer, Cancers 11 (6) (2019) 756. doi:10.3390/cancers11060756.

[21] D. Black, D. Byrne, A. Walke, S. Liu, A. Di Ieva, S. Kaneko, W. Stummer, T. Salcudean, E. Suero Molina, Towards machine learning-based quantitative hyperspectral image guidance for brain tumor resection, Communications Medicine 4 (1) (2024) 1–11. doi:10.1038/s43856-024- 00562-3.

[22] G. Lu, B. Fei, Medical hyperspectral imaging: a review, Journal of Biomedical Optics 19 (1) (2014) 10901. doi:10.1117/1.JBO.19.1.010901.

[23] B. Fei, Hyperspectral imaging in medical applications, in: Data Handling in Science and Technology, Data Handling in Science and Technology, Elsevier Ltd, 2020, pp. 523–565. doi:10.1016/B978-0-444-63977- 6.00021-3.

[24] N. T. Clancy, G. Jones, L. Maier-Hein, D. S. Elson, D. Stoyanov, Surgical spectral imaging, Medical Image Analysis 63 (2020) 101699. doi:10.1016/j.media.2020.101699.

[25] J. Shapey, Y. Xie, E. Nabavi, R. Bradford, S. R. Saeed, S. Ourselin, T. Vercauteren, Intraoperative multispectral and hyperspectral label-free imaging: A systematic review of in vivo clinical studies, Journal of Biophotonics 12 (9) (2019) e201800455, \_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/jbio.201800455. doi:10.1002/jbio.201800455.

[26] A. Martín-Pérez, A. M. d. Ternero, A. Lagares, E. Juarez, C. Sanz, Spectral analysis comparison of pushbroom and snapshot hyperspectral cameras for in vivo brain tissues and chromophore identification, Journal of Biomedical Optics 29 (9) (2024) 093510. doi:10.1117/1.JBO.29.9.093510.

[27] H. Fabelo, H. Martin, O. Samuel, S. Maysam, S. Adam, J. F. Piñeiro, S. Coralia, O. Aruma J., S. Bisshopp, E. Carlos, M. Mariano, H. María,

C. David, M. Jesús, C. Gustavo M., S. Roberto, B. Fei, Deep learningbased framework for in-vivo identification of glioblastoma tumor using hyperspectral images of human brain. 19 (4) (2019) 920.

[28] G. Urbanos, A. Martín, G. Vázquez, M. Villanueva, M. Villa, L. Jimenez-Roldan, M. Chavarrías, A. Lagares, E. Juárez, C. Sanz, Supervised Machine Learning Methods and Hyperspectral Imaging Techniques Jointly Applied for Brain Cancer Classification, Sensors 21 (11) (2021) 3827. doi:10.3390/s21113827.

[29] S. Puustinen, H. Vrzáková, J. Hyttinen, T. Rauramaa, P. Fält, M. Hauta-Kasari, R. Bednarik, T. Koivisto, S. Rantala, M. von und zu Fraunberg, J. E. Jääskeläinen, A.-P. Elomaa, Hyperspectral imaging in brain tumor surgery — evidence of machine learning-based performance, World Neurosurgery 175 (2023) e614–e635. doi:10.1016/j.wneu.2023.03.149.

[30] F. Manni, F. van der Sommen, H. Fabelo, S. Zinger, C. Shan, E. Edström, A. Elmi-Terander, S. Ortega, G. Marrero Callicó, P. H. N. de With, Hyper spectral imaging for glioblastoma surgery: improving tumor ifentification using a deep spectral-spatial approach., Sensors 20 (23) (2020) 6955. doi:10.3390/s20236955.

[31] T. Giannantonio, A. Alperovich, P. Semeraro, M. Atzori, X. Zhang, C. Hauger, A. Freytag, S. Luthman, R. Vandebriel, M. Jayapala, L. Solie, S. d. Vleeschouwer, Intra-operative brain tumor detection with deep learning-optimized hyperspectral imaging, in: Optical Biopsy XXI: Toward Real-Time Spectroscopic Imaging and Diagnosis, Vol. 12373, SPIE, 2023, pp. 80–98. doi:10.1117/12.2646999.

[32] A. Martín-Pérez, B. Martinez-Vega, M. Villa, R. Leon, A. M. de Ternero, H. Fabelo, S. Ortega, E. Quevedo, G. M. Callico, E. Juarez, et al., Unifying heterogeneous hyperspectral databases for in-vivo human brain cancer classification: towards robust algorithm development, Computer Methods and Programs in Biomedicine Update (2025) 100183doi:https://doi.org/10.1016/j.cmpbup.2025.100183.

[33] H. Fabelo, S. Ortega, A. Szolna, D. Bulters, J. F. Piñeiro, S. Kabwama, A. J-O’Shanahan, H. Bulstrode, S. Bisshopp, B. R. Kiran, D. Ravi, R. Lazcano, D. Madroñal, C. Sosa, C. Espino, M. Marquez,

M. De La Luz Plaza, R. Camacho, D. Carrera, M. Hernández, G. M. Callicó, J. Morera Molina, B. Stanciulescu, G.-Z. Yang, R. Salvador, E. Juárez, C. Sanz, R. Sarmiento, In-vivo hyperspectral human brain image database for brain cancer detection., IEEE Access 7 (2019) 39098– 39116. doi:10.1109/ACCESS.2019.2904788.

[34] R. Vandebriel, S. Luthman, K. Vunckx, M. Jayapala, W. Charle, L. Solie, S. D. Vleeschouwer, T. Giannantonio, A. Alperovich, X. Zhang, Integrating hyperspectral imaging in an existing intra-operative environment for detection of intrinsic brain tumors, in: Advanced Biomedical and Clinical Diagnostic and Surgical Guidance Systems XXI, Vol. 12368, SPIE, 2023, pp. 65–74. doi:10.1117/12.2647690.

[35] Y. Zhang, X. Wu, L. He, C. Meng, S. Du, J. Bao, Y. Zheng, Applications of hyperspectral imaging in the detection and diagnosis of solid tumors, Translational Cancer Research 9 (2) (2020) 1265–1277. doi:10.21037/tcr.2019.12.53.

[36] R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua, S. Süsstrunk, Slic superpixels compared to state-of-the-art superpixel methods, IEEE Transactions on Pattern Analysis and Machine Intelligence 34 (11) (2012) 2274–2282. doi:10.1109/TPAMI.2012.120.

[37] Optical Absorption of Hemoglobin. URL https://omlc.org/spectra/hemoglobin/

[38] R. Bartha, J. Megyesi, C. Watling, Low-grade glioma: Correlation of short echo time 1h-mr spectroscopy with 23na mr imaging, American Journal of Neuroradiology 29 (3) (2008) 464–470. arXiv:https://www.ajnr.org/content/29/3/464.full.pdf, doi:10.3174/ajnr.A0854. URL https://www.ajnr.org/content/29/3/464

[39] M. Law, S. Yang, H. Wang, J. S. Babb, G. Johnson, S. Cha, E. A. Knopp, D. Zagzag, Glioma grading: Sensitivity, specificity, and predictive values of perfusion mr imaging and proton mr spectroscopic imaging compared with conventional mr imaging, American Journal of Neuroradiology 24 (10) (2003) 1989–1998. arXiv:https://www.ajnr.org/content/24/10/1989.full.pdf. URL https://www.ajnr.org/content/24/10/1989

[40] M. Law, S. Oh, G. Johnson, J. S. Babb, D. Zagzag, J. Golfinos, P. J. Kelly, Perfusion magnetic resonance imaging predicts patient outcome as an adjunct to histopathology: a second reference standard in the surgical and nonsurgical treatment of low-grade gliomas, Neurosurgery 58 (6) (2006) 1099–1107, discussion 1099–1107. doi:10.1227/01.NEU.0000215944.81730.18.

[41] M. Essig, T. B. Nguyen, M. S. Shiroishi, M. Saake, J. M. Provenzale, D. S. Enterline, N. Anzalone, A. Dörfler, À. Rovira, M. Wintermark, M. Law, Perfusion mri: the five most frequently asked clinical questions, AJR American Journal of Roentgenology 201 (3) (2013) W495–W510. doi:10.2214/AJR.12.9544.

[42] J. M. Provenzale, G. R. Wang, T. Brenner, J. R. Petrella, A. G. Sorensen, Comparison of permeability in high-grade and lowgrade brain tumors using dynamic susceptibility contrast mr imaging, American Journal of Roentgenology 178 (3) (2002) 711–716, pMID: 11856703. arXiv:https://doi.org/10.2214/ajr.178.3.1780711, doi:10.2214/ajr.178.3.1780711. URL https://doi.org/10.2214/ajr.178.3.1780711

[43] J. C. Wong, J. M. Provenzale, J. R. Petrella, Perfusion mr imaging of brain neoplasms, American Journal of Roentgenology 174 (4) (2000) 1147–1157, pMID: 10749268. arXiv:https://doi.org/10.2214/ajr.174.4.1741147, doi:10.2214/ajr.174.4.1741147. URL https://doi.org/10.2214/ajr.174.4.1741147

[44] S. M. Evans, K. D. Judy, I. Dunphy, W. T. Jenkins, W.-T. Hwang, P. T. Nelson, R. A. Lustig, K. Jenkins, D. P. Magarelli, S. M. Hahn, R. A. Collins, M. S. Grady, C. J. Koch, Hypoxia is important in the biology and aggression of human glial brain tumors, Clinical Cancer Research 10 (24) (2004) 8177–8184. doi:10.1158/1078-0432.CCR-04-1081. URL https://doi.org/10.1158/1078-0432.CCR-04-1081

[45] S. Xu, L. Tang, Z. Liu, C. Luo, Q. Cheng, Hypoxia-related lncrna correlates with prognosis and immune microenvironment in lowergrade glioma, Frontiers in Immunology Volume 12 - 2021 (2021). doi:10.3389/fimmu.2021.731048. URL https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.20

[46] S. O. Atik, Dual-stream spectral-spatial convolutional neural network for hyperspectral image classification and optimal band selection, Advances in Space Research 74 (5) (2024) 2025–2041. doi:10.1016/j.asr.2024.05.064.

[47] W. Dong, H. Wang, F. Wu, G. Shi, X. Li, Deep spatialspectral representation learning for hyperspectral image denoising, IEEE Transactions on Computational Imaging 5 (4) (2019) 635–648. doi:10.1109/TCI.2019.2911881.

[48] V. Kumar, R. S. Singh, M. Rambabu, Y. Dua, Deep learning for hyperspectral image classification: A survey, Computer Science Review 53 (2024) 100658. doi:10.1016/j.cosrev.2024.100658. URL https://www.sciencedirect.com/science/article/pii/S157401372400042X

[49] S. Li, W. Song, L. Fang, Y. Chen, P. Ghamisi, J. A. Benediktsson, Deep learning for hyperspectral image classification: an overview, IEEE Transactions on Geoscience and Remote Sensing 57 (9) (2019) 6690– 6709. doi:10.1109/TGRS.2019.2907932.

[50] Q. Shenming, L. Xiang, G. Zhihua, A new hyperspectral image classification method based on spatial-spectral features, Scientific Reports 12 (1) (2022) 1541. doi:10.1038/s41598-022-05422-5.

[51] T. Czempiel, A. Roddan, M. Leiloglou, Z. Hu, K. O’Neill, G. Anichini, D. Stoyanov, D. Elson, Rgb to hyperspectral: Spectral reconstruction for enhanced surgical imaging, Healthcare Technology Letters 11 (6) (2024) 307–317.

[52] A. Studier-Fischer, S. Seidlitz, J. Sellner, M. Bressan, B. Özdemir, L. Ayala, J. Odenthal, S. Knoedler, K.-F. Kowalewski, C. M. Haney, et al., Heiporspectral-the heidelberg porcine hyperspectral imaging dataset of 20 physiological organs, Scientific Data 10 (1) (2023) 414.

## Appendix A. Registration Examples

![](images/258736827a712a0c40304e2ede190ff93ef89e5c63ead944387afc897516f40b.jpg)  
Figure A.12: Examples of both good and bad registrations of RGB image and correspond ing contours with HSI. Bad registrations were manually registered afterwards.

## Appendix B. Data characteristics

Table B.3: Overview of all 18 enrolled patients, including the number of acquired images and superpixels before and after filtering. Patient 10 is shown for cohort transparency but was excluded from all model development and evaluation because fluorescence imaging was used. Patient 3 contained only healthy-tissue annotations and was used for training but excluded from patient-level binary classification evaluation.
<table><tr><td>Patient</td><td>Nr. of images</td><td>Initial Su- perpixels (Tumor)</td><td>Filtered Super- pixels (Tumor)</td><td>Initial Su- perpixels (Healthy)</td><td>Filtered Super- pixels (Healthy)</td></tr><tr><td>1</td><td>2</td><td>574</td><td>250</td><td>230</td><td>101</td></tr><tr><td>2</td><td>4</td><td>1316</td><td>564</td><td>361</td><td>162</td></tr><tr><td>3</td><td>2</td><td>0</td><td>0</td><td>588</td><td>274</td></tr><tr><td>4</td><td>4</td><td>467</td><td>212</td><td>1358</td><td>648</td></tr><tr><td>5</td><td>4</td><td>1321</td><td>441</td><td>3449</td><td>1732</td></tr><tr><td>6</td><td>4</td><td>626</td><td>264</td><td>4129</td><td>1842</td></tr><tr><td>7</td><td>5</td><td>3344</td><td>1456</td><td>4267</td><td>1923</td></tr><tr><td>8</td><td>4</td><td>4761</td><td>2033</td><td>2015</td><td>985</td></tr><tr><td>9</td><td>5</td><td>3390</td><td>1504</td><td>2258</td><td>996</td></tr><tr><td>10</td><td>2</td><td>607</td><td>329</td><td>2262</td><td>889</td></tr><tr><td>11</td><td>5</td><td>6828</td><td>3019</td><td>2433</td><td>1052</td></tr><tr><td>13</td><td>3</td><td>614</td><td>225</td><td>1713</td><td>832</td></tr><tr><td>14</td><td>7</td><td>10852</td><td>4876</td><td>1487</td><td>681</td></tr><tr><td>15</td><td>4</td><td>2446</td><td>1090</td><td>1146</td><td>486</td></tr><tr><td>18</td><td>5</td><td>3587</td><td>1699</td><td>2648</td><td>1186</td></tr><tr><td>19</td><td>4</td><td>3100</td><td>1296</td><td>2191</td><td>1113</td></tr><tr><td>20</td><td>6</td><td>5608</td><td>2462</td><td>3195</td><td>1526</td></tr><tr><td>21</td><td>5</td><td>7317</td><td>3147</td><td>1150</td><td>610</td></tr></table>

## Appendix C. Binary classification metrics

<table><tr><td></td><td></td><td colspan="2">Prediction</td></tr><tr><td></td><td></td><td>Positive</td><td>Negative</td></tr><tr><td>Ground truth</td><td>Positive Negative</td><td>True positive (TP) False Positive (FP)</td><td>False Negative (FN) True Negative (TN)</td></tr></table>

Table C.4: Binary confusion matrix.

Accuracy is the proportion of correct predictions among all predictions:

$$
\mathrm { A C C } = { \frac { T P + T N } { T P + T N + F P + F N } } .\tag{C.1}
$$

Precision is the proportion of positive predictions that are true positives:

$$
\mathrm { P R E C } = \frac { T P } { T P + F P } .\tag{C.2}
$$

Recall, also called sensitivity or the true-positive rate (TPR), is the proportion of actual positives that are correctly identified:

$$
\mathrm { R E C } = \mathrm { T P R } = { \frac { T P } { T P + F N } } .\tag{C.3}
$$

Specificity is the proportion of actual negatives that are correctly identi fied:

$$
{ \mathrm { S P E C } } = { \frac { T N } { T N + F P } } .\tag{C.4}
$$

The F1 score is the harmonic mean of precision and recall:

$$
F 1 = { \frac { 2 \mathrm { P R E C R E C } } { \mathrm { P R E C } + \mathrm { R E C } } } = { \frac { 2 T P } { 2 T P + F P + F N } } .\tag{C.5}
$$

The false-positive rate (FPR) is defined as

$$
\mathrm { F P R } = \frac { F P } { F P + T N } .\tag{C.6}
$$

The area under the receiver-operating-characteristic curve (ROC-AUC) is

$$
\mathrm { A U C } _ { \mathrm { R O C } } = \int _ { 0 } ^ { 1 } \mathrm { T P R } ( u ) d u ,\tag{C.7}
$$

where u denotes the false-positive rate as the decision threshold varies.

## Appendix D. Results cross-validations

The results of the hyperparameter optimization using a cross-validation for each run, is visualized in Figure D.13

![](images/81686a581d4c2205430cace525619f100f7a906f44e245f7990be3a5b6c105dd.jpg)

![](images/6f3b548ce3a333bc7c5cefc22732518da3eac7cdeb4f59ce59d4ae0022d78e00.jpg)

![](images/c286edab0b5c68f7d2972c30ec1e0fa84586c7ebe080b2efa75489922d23525b.jpg)

![](images/b52648bee02955341aa8305f59475dffa55616a42ab5dcaa1a5b3d0f1cf31f1f.jpg)

![](images/31c434259e71e4981e91190a2b55fc1ea4b91f718a1ceca6093ef1c06072ddc6.jpg)  
Figure D.13: Hyperparameter optimization results for the cross-validations for all runs. Each sample in the boxplot represents the cross-validation score for a run given the combination of hyperparameters