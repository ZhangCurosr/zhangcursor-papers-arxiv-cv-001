OPEN

Check for updates

# Enhanced MRI brain tumor detection using deep learning in conjunction with explainable AI SHAP based diverse and multi feature analysis

Asif Rahman<sup>1</sup>, Maqsood Hayat<sup>1</sup>, Nadeem Iqbal<sup>1</sup>, Fawaz Khaled Alarfaj<sup>2</sup>, Salem Alkhalaf<sup>3</sup> & Fahad Alturise<sup>4</sup>

Recent innovations in medical imaging have markedly improved brain tumor identification, surpassing conventional diagnostic approaches that sufer from low resolution, radiation exposure, and limited contrast. Magnetic Resonance Imaging (MRI) is pivotal in precise and accurate tumor characterization owing to its high-resolution, non-invasive nature. This study investigates the synergy among multiple feature representation schemes such as local Binary Patterns (LBP), Gabor filters, Discrete Wavelet Transform, Fast Fourier Transform, Convolutional Neural Networks (CNN), and Gray-Level Run Length Matrix alongside five learning algorithms namely: k-nearest Neighbor, Random Forest, Support Vector Classifier (SVC), and probabilistic neural network (PNN), and CNN. Empirical findings indicate that LBP in conjunction with SVC and CNN obtained high specificity and accuracy, rendering it a promising method for MRI-based tumor diagnosis. Further to investigate the contribution of LBP, Statistical analysis chi-square and p-value tests are used to confirm the significant impact of LBP feature space for identification of brain Tumor. In addition, The SHAP analysis was used to identify the most important features in classification. In a small dataset, CNN obtained 97.8% accuracy while SVC yielded 98.06% accuracy. In subsequent analysis, a large benchmark dataset is also utilized to evaluate the performance of learning algorithms in order to investigate the generalization power of the proposed model. CNN achieves the highest accuracy of 98.9%, followed by SVC at 96.7%. These results highlight CNN’s efectiveness in automated, high-precision tumor diagnosis. This achievement is ascribed with MRI–based feature extraction by combining high resolution, non-invasive imaging capabilities with the powerful analytical abilities of CNN. CNN demonstrates superiority in medical imaging owing to its ability to learn intricate spatial patterns and generalize efectively. This interaction enhances the accuracy, speed, and consistency of brain tumor detection, ultimately leading to better patient outcomes and more eficient healthcare delivery. https://github.com/asifrahman557/BrainTumorDete ction.

Keywords Brain tumor, Machine learning, CNN, PNN, LBP, RF, MRI, SHAP analysis

The human brain, a fundamental component of the central nervous system (CNS), collaborates with the spinal cord to manage and synchronize vital body activities, such as decision-making, signal transmission, and sensory information integration<sup>1</sup>. The complicated and highly structured anatomy facilitates complex cognitive and physiological processes; nevertheless, this complexity also makes it susceptible to various illnesses and diseases, including stroke, infections, and brain tumors<sup>2,3</sup>. Brain tumors, characterized by the unregulated growth of abnormal cells within the limited confines of the skull, present considerable threats to neurological function and general health<sup>4–6</sup>. These tumors are generally categorized into two types: benign (non-cancerous) and malignant (cancerous), with more than 200 specific subtypes recognized based on their cellular origin, location, and behavior<sup>7,8</sup>. Despite the advancement in medical research, the prognosis for patients with brain tumors remains unfavorable, with survival rates difering markedly based on tumor type, location, and stage at diagnosis<sup>9–11</sup>.

Brain tumors are a significant source of morbidity and mortality globally, ranking as the 10th most prevalent cause of death across all age demographics<sup>12</sup>. The National Brain Tumor Foundation (NBTF) reports a 300% increase in brain tumor-related fatalities over the last three decades, highlighting the critical necessity for enhanced diagnostic and treatment approaches. Timely identification and care are essential, since untreated tumors may result in significant neurological impairments or death<sup>13</sup>. Conventional diagnostic techniques, including biopsies, are intrusive, protracted, and fraught with hazards, hence demanding the creation of non invasive, precise, and eficient diagnostic instruments. Magnetic Resonance Imaging (MRI) has emerged as the benchmark for diagnosing brain tumors, providing exceptional visibility of analytical structures and anomalies.

Brain tumors, originating inside the brain, constitute 14.1% of yearly occurrences, with a disproportionately elevated frequency in children (70% of cases)<sup>14,15</sup>. Despite their commonality, eficacious therapies for several brain tumor subtypes continue to be elusive, and the enduring adverse consequences of these tumors on patients’ quality of life are significant<sup>16,17</sup>. Moreover, global statistics reveal a significant rise in brain tumor manifestations, with incidence rates escalating by 10–15% from 2004 to 2020<sup>18</sup>. These numbers emphasize the increasing prevalence of brain tumors and the urgent inevitability of novel approaches to enhance detection, treatment, and patient outcomes. However, the prompt and eficient care of brain cancers depends on precise diagnosis and categorization. The variability of brain tumors regarding size, location, form, and imaging properties, together with changes in scanning settings and modalities, renders this endeavor intrinsically hard<sup>19</sup>. Conventional diagnostic methods, such as the Leksell Gamma Knife and radioactive beams, while beneficial, are labor-intensive, need considerable human knowledge, and frequently entail extended processing durations<sup>20</sup>. The emergence of sophisticated imaging technologies, such as Computer Tomography (CT), MRI, and Positron Emission Tomography (PET), has transformed brain tumor diagnosis by ofering intricate, non-invasive viewing of cerebral structures. MRI is the preferred modality because of its higher soft tissue contrast and capability to acquire multi-parametric data, including T1-weighted, T2-weighted, and FLAIR sequences<sup>21</sup>.

Nevertheless, with these breakthroughs, the intricacy of brain tumors persists in posing considerable obstacles for doctors and researchers. The necessity for accurate, automated diagnostic instruments is essential, as early identification and commencement of therapy are pivotal elements afecting patient survival rates<sup>22</sup>. The amalgamation of sophisticated imaging data with machine learning (ML) and deep learning (DL) methodologies presents a potential approach to overcoming these problems, facilitating the creation of intelligent systems proficient in precise tumor identification, classification, and segmentation.

The rapid advancement of machine learning (ML) and deep learning (DL) has revolutionized medical imaging, providing robust instruments for the automated examination of intricate datasets<sup>23,24</sup>. Deep Neural Networks (DNNs), especially Convolutional Neural Networks (CNNs), have exhibited exceptional eficacy in image classification, object recognition, and natural language processing applications<sup>25–</sup> <sup>26</sup>. These breakthroughs are driven by enhancements in computing power, particularly GPUs, and the accessibility of extensive, annotated datasets, which have enabled the creation of very precise models for medical applications<sup>27,28</sup>.

In the realm of brain tumor diagnostics, deep learning models have demonstrated remarkable potential. CNNs, due to their hierarchical structure and capacity to learn spatial feature hierarchies, are especially adept in analyzing medical pictures<sup>29</sup>. Techniques including autoencoders, transfer learning, and ensemble approaches have significantly improved the eficacy of these models, allowing them to attain state-of-the-art outcomes in tumor detection and classification<sup>30–32</sup>. Hybrid models that integrate CNNs with optimization methods, like Particle Swarm Optimization (PSO) and Genetic methods (GAs), have exhibited enhanced eficacy in tumor segmentation and classification tasks<sup>33,34</sup>.

Despite these achievements, other dificulties persist. Numerous current investigations are constrained by the utilization of small, homogenous datasets, which may insuficiently reflect the range of tumor types, imaging modalities, and patient demographics<sup>35,36</sup>. Furthermore, the incorporation of deep learning models into clinical practice necessitates the resolution of challenges of computational complexity, interpretability, and generalizability<sup>37</sup>. Moreover, the absence of common assessment measures and benchmarking datasets complicates the comparison of various models and methodologies<sup>38</sup>.

Substantial advancements have been achieved in the formulation of computer models for brain tumor diagnosis, with an increasing focus on intelligent methodologies. Recent research has shown the capability of machine learning and deep learning approaches to revolutionize brain tumor diagnostics. Almadhoun et al.<sup>39</sup> attained 98% test accuracy utilizing a deep learning model on a dataset including 10,000 MR images. Musallam et al.<sup>40</sup> suggested a DCNN model achieving an overall accuracy of 97.72% for the classification of four tumor subtypes. Additional significant contributions encompass the use of correlation learning techniques<sup>41</sup>, ensemble classifiers<sup>42</sup>, and sophisticated designs such as Dense EficientNet<sup>43</sup>, which attained accuracies beyond 98%.

Hybrid models, like those that integrate CNNs with SVM or evolutionary algorithms, have demonstrated encouraging outcomes. Mittal et al.<sup>44</sup> shown enhanced segmentation accuracy by the integration of Growing CNN with Stationary Wavelet Transform. Preprocessing approaches, including picture augmentation and multimodal information fusion, have significantly improved the performance of these models<sup>45,46</sup>. In addition, recent studies in unsupervised domain adaptation have demonstrated promising performance in medical image segmentation such as multi-organ segmentation<sup>47</sup>, knee MRI analysis<sup>48</sup>, identification of ocular diseases<sup>49</sup>. Furthermore, liver segmentation across modalities has been approached using adversarial and self-learning techniques<sup>50</sup>.

In a sequel, ML approaches tackle more dificulties associated with dataset variety and computational complexity<sup>51,52</sup>. The detailed analysis of the literature review is illustrated in Table 1.

<table><tr><td rowspan=1 colspan=1>Classification</td><td rowspan=1 colspan=1>Reference</td><td rowspan=1 colspan=1>Method/model</td><td rowspan=1 colspan=1>Details</td><td rowspan=1 colspan=1>Performance metrics</td></tr><tr><td rowspan=10 colspan=1>Deep learningmodels</td><td rowspan=1 colspan=1>Almadhoun et al.53. ,</td><td rowspan=1 colspan=1>Educational Model utilizing Deep learning algorithms</td><td rowspan=1 colspan=1>Transfer learning10,000 MRI images</td><td rowspan=1 colspan=1>Training accuracy: 100%Testing accuracy: 98%</td></tr><tr><td rowspan=1 colspan=1>Musallam et al.,54</td><td rowspan=1 colspan=1>DCCN model</td><td rowspan=1 colspan=1>Multi categories tumor classification</td><td rowspan=1 colspan=1>Accuracy: 97.72%</td></tr><tr><td rowspan=1 colspan=1>Nayat et al.,55</td><td rowspan=1 colspan=1>Dense efficient/Net</td><td rowspan=1 colspan=1>Hybrid Model</td><td rowspan=1 colspan=1>Accuracy: 98.78%</td></tr><tr><td rowspan=1 colspan=1>Obeidavi et al.,56</td><td rowspan=1 colspan=1>Residual Network based on CNN</td><td rowspan=1 colspan=1>BRATS dataset 2015</td><td rowspan=1 colspan=1>Accuracy: 97.05%</td></tr><tr><td rowspan=1 colspan=1>Sajjad et al.57</td><td rowspan=1 colspan=1>Hybrid CNN model</td><td rowspan=1 colspan=1>CNN + Additional approaches</td><td rowspan=1 colspan=1>Accuracy: 86%Specificity: 91%</td></tr><tr><td rowspan=1 colspan=1>Paual et al.52</td><td rowspan=1 colspan=1>CNN and Fully connected network</td><td rowspan=1 colspan=1>3064 MRI images</td><td rowspan=1 colspan=1>Accuracy: 91.43%</td></tr><tr><td rowspan=1 colspan=1>Mittal et al.,60</td><td rowspan=1 colspan=1>Stationary wavelet Transform and CNN</td><td rowspan=1 colspan=1>Focus on segmentation</td><td rowspan=1 colspan=1>--</td></tr><tr><td rowspan=1 colspan=1>Cinar and Yildirim63</td><td rowspan=1 colspan=1>GoogLeNet, InceptionV3, DenseNet201, AlexNet, andResNet-50</td><td rowspan=1 colspan=1>Tumor segmentation and localization</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Sangeeta et al.,66</td><td rowspan=1 colspan=1>ResNet-50</td><td rowspan=1 colspan=1>Brain Tumor classification</td><td rowspan=1 colspan=1>Accuracy: 96.50%</td></tr><tr><td rowspan=1 colspan=1>Liang et al.,65</td><td rowspan=1 colspan=1>Fusion of DL model</td><td rowspan=1 colspan=1>Brain tumor scaling</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=6 colspan=1>Machine learningalgorithms</td><td rowspan=1 colspan=1>Wozniak et al.58</td><td rowspan=1 colspan=1>Correlation learning model</td><td rowspan=1 colspan=1>3064 brain cancer images</td><td rowspan=1 colspan=1>Accuracy: 96%</td></tr><tr><td rowspan=1 colspan=1>Garg et al.,59</td><td rowspan=1 colspan=1>Hybrid model</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Accuracy: 97.3%</td></tr><tr><td rowspan=1 colspan=1>Lotlikar et al.,20</td><td rowspan=1 colspan=1>K-nearest neighbor</td><td rowspan=1 colspan=1>Brain dataset</td><td rowspan=1 colspan=1>Accuracy: 95.6%</td></tr><tr><td rowspan=1 colspan=1>Gumaei et al.,47</td><td rowspan=1 colspan=1>RELM + PCA-NGIST</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Accuracy: 94.2%</td></tr><tr><td rowspan=1 colspan=1>Pasheal et al.,64</td><td rowspan=1 colspan=1>CNN and 10-fold cross-validation</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Accuracy: 93.6%</td></tr><tr><td rowspan=1 colspan=1>Narayana and Reddy61</td><td rowspan=1 colspan=1>Ensemble model</td><td rowspan=1 colspan=1>MRI images</td><td rowspan=1 colspan=1>Accuracy; 91%</td></tr><tr><td rowspan=1 colspan=1>Optimizationbased model</td><td rowspan=1 colspan=1>Dixit and Nanda62</td><td rowspan=1 colspan=1>PSO and CNN model</td><td rowspan=1 colspan=1>Brain tumor detection</td><td rowspan=1 colspan=1>Accuracy: 92%</td></tr></table>

Table 1. Literature review.

This study aims to address the limitations of existing models or predictors by assessing the efectiveness of various machine learning and deep learning models with diverse feature extraction strategies for brain tumor classification in order to achieve high diagnostic accuracy. This research endeavors to provide a robust framework for accurate and timely brain tumor diagnosis, ultimately improving patient outcomes.

The contribution of this study includes:

• A comprehensive assessment of cutting-edge machine learning and deep learning models for brain tumor categorization.

• The integration of sophisticated feature extraction methods to enhance model performance.

• A comparative analysis of the proposed model with existing methodologies, highlighting their limitations and strengths.

## Materials and methods Dataset

This research utilizes the publicly accessible Brain Tumor MRI dataset from Kaggle, https://www.kaggle.com/ datasets/masoudnickparvar/brain-tumor-mri-dataset<sup>67</sup>. The dataset consists of MRI images grouped into two primary categories tumor and non-tumor. Further tumor images were categorized into three subtypes including Pituitary Tumors, Meningioma tumors, and Glioma Tumors. The dataset consists of 2000 MRI images of which 1000 images of non-tumor and 1000 images of tumor. Further, tumor data is composed of 300 images of Pituitary, 400 of Meningioma, and 300 images of Glioma tumors. In order to evaluate the generalization of the model, a large training dataset is used, which consists of 7023 MRI images split into 5712 images for training and 1311 images for testing<sup>63</sup>. In the training dataset, 4117 images are afected by tumors further classified into 1457 images of the pituitary, 1339 images of meningioma, and 1321 images of glioma tumors whereas 1595 images of non-tumor. The testing dataset consists of 1311 images of which 405 images of non-tumors and 906 images of tumors. Tumor images are 300 images of pituitary, 306 images of meningioma, and 300 images of glioma tumors.

This work employed a series of preprocessing and data cleansing steps to augment the quality of brain tumor MRI images and boost the precision and resilience of the classification models. Initially, all images were transformed to grayscale to minimize computational complexity by removing superfluous color channels, as the essential characteristics for classification are based on intensity fluctuations rather than color information. A Gaussian filter was subsequently added to blur images, therefore, diminishing high-frequency noise and enabling the model to concentrate on the most pertinent characteristics while lessening the impact of slight alterations<sup>68</sup>. Subsequently, the binary threshold was executed to convert the grayscale images into binary images. During this process, the pixels’ value above the predefined threshold were rendered white (foreground) and those below were assigned black (background). This stage highlighted the delineation of objects, especially tumor areas, thereby facilitating the distinction between malignant tissue and healthy tissue. In a sequel, contour detection methods were utilized to ascertain the greatest contour, which generally represents the tumor region, in each image. The images were subsequently cropped to emphasize the area of interest (ROI), eliminating extraneous background elements and ensuring that the model focused on the most relevant aspects of the image. Ultimately, all images were downsized to a uniform resolution of 128 × 128 pixels to maintain uniformity throughout the collection.

Further, several advanced data augmentation techniques were utilized to enhance the diversity of the benchmark training dataset and strengthen the model’s ability to generalize to novel data. Data augmentation is a technique that artificially increases the quality such as illumination, size, and orientation as well as the diversity of a dataset by applying various transformations. This research utilized random modifications to brightness and contrast. Brightness was arbitrarily adjusted within a range of 0.8 to 1.2, emulating diverse lighting situations that may arise during image acquisition. Likewise, the contrast was modified within the identical range to assist the model in identifying tumor characteristics across varying imaging settings. Subsequently, all enhanced images were normalized by dividing the pixel values by 255, so scaling them to a range between 0 and 1.

## Feature representation techniques

This research examined six distinct feature representation strategies that are essential for transforming MRI images into significant data, which is subsequently employed for classification. The framework of proposed model is illustrated in Fig. 1. Comprehensive details of these feature representation approaches are listed below:

## Convolutional neural networks (CNN)

CNNs are a type of deep learning algorithms that automatically learn both discriminatory and hierarchical information from input data<sup>69</sup>. Here, CNNs were utilized to extract information from brain MRI data. The pre-trained CNN architecture proved notably proficient in capturing intricate spatial patterns, textures, and structural details, which contributed to obtaining high throughput for brain tumor detection. CNN can be formally expressed as:

$$
F e a t u r e ~ s p a c e ~ ( x + y ) = \sum { \overset { k } { \mathop { i = } } } 1 \sum { \overset { k } { \mathop { j = } } } 0 i n p u t ~ ( x + i , y + j ) . K e r n a l ~ ( i , j )\tag{1}
$$

Where input $( x + i , y + j )$ represents the input image pixel values, Kernal $( i , j )$ denotes the filter weights, k is the size of the kernel.

The hierarchical discriminative attributes extraction of CNNs facilitated the recognition of elusive motifs revealing of brain tumors.

## Discrete wavelet transform (DWT)

The Discrete Wavelet Transform (DWT) is a signal processing method that expresses both time and frequency analysis simultaneously. Here, DWT was utilized on MRI images to split into frequency sub-bands in order to extract both spatial and frequency information. The wavelet transform can be formulated as:

$$
W \ ( x , y ) = \frac { 1 } { \sqrt { x } } \int _ { - \infty } ^ { \infty } i m a g e \ ( t ) + \ \psi \ \left( \frac { t - y } { x } \right) d t\tag{2}
$$

![](images/74ecdc276aca8d6e02fb58810afddd09245247970eb2f8c5f9da48e4cd487ecb.jpg)  
Fig. 1. Framework of proposed model.

Where x is the scaling factor, y is the translation factor, and ψ(t) is the wavelet function.

The outcome of DWT yielded a comprehensive representation of the image information which assist in comprehension of spatial and frequency characteristics of MRI images.

## Fast fourier transform (FFT)

The Fast Fourier Transform (FFT) is a mathematical tool used to transform a signal from the spatial domain frequency domain. Here FFT was utilized to extract frequency-based information from MRI images. The FFT of an image Input(x, y) is given by:

$$
F \left( u , v \right) = \sum \ x - 1 \sum \ y ^ { N - 1 } I n p u t \left( x , \ y \right) . e ^ { - i 2 \pi \left( \frac { u x } { M } + \frac { v y } { N } \right) }\tag{3}
$$

where $F ( u , \nu )$ denotes the frequency domain representation, M and N are the dimensions of the image. FFT-based frequency-domain information highlights patterns and structural information which is useful in distinguishing between healthy and tumor-afected brain images.

## Gabor filters

Gabor Filters are efective tools which is used for the extraction of texture features from images. These filters are inspired by the human visual system and are mostly efective in collecting local spatial and frequency attributes The 2D Gabor filter is defined as:

$$
\begin{array} { c } { { G \left( x , y \right) = e x p \left( \displaystyle \frac { { x ^ { \prime } } ^ { 2 } + { \gamma } ^ { 2 } y ^ { \prime } { } ^ { 2 } } { 2 { \delta } ^ { 2 } } \right) . \mathrm { c o s ~ } \left( 2 \pi \displaystyle \frac { x ^ { \prime } } { \lambda } + \psi \right) } } \\ { { \mathrm { x } \prime ~ = x \mathrm { c o s } + y \mathrm { s i n } , \mathrm { y } \prime = x \mathrm { s i n } + y \mathrm { c o s } , } } \end{array}\tag{4}
$$

where γ is the spatial aspect ratio, λ is the wavelength, θ is the orientation, σ is the standard deviation, ψ is the phase ofset. Here Gabor Filters were applied to capture texture attributes from MRI images, which assisted in detecting patterns revealing brain tumors.

## Gray-level run length matrix (GLRLM)

The Gray-Level Run Length Matrix (GLRLM) is a texture analysis method that measures the frequency of contiguous pixels with the same gray level value. This approach was utilized to extract texture information from MRI data. The GLRLM is defined as:

$$
P \left( i , j | \theta \right) = N u m b e r o f r u n s w i t h g r a y l e v e l i a n d r u n l e n g t h j i n d i r e c t i o n \theta (\tag{5}
$$

where: i is the gray level, j is the run length, θ is the direction. The retrieved texture attributes facilitate the distinction between tumor-afected and healthy brain tissues by detecting textural variations in the MRI images.

## Local binary patterns (LBP)

Local Binary Patterns (LBP) is a texture-oriented feature extraction method that evaluates each pixel with its adjacent pixels to generate a binary pattern. The LBP value for a pixel $( \mathbf { x } _ { c } , \mathbf { y } _ { c } )$ is determined as follows:

$$
L B P \left( x _ { c } , y _ { c } \right) = \sum _ { p = 0 } \sp { p - 1 } \mathrm { s } \left( g _ { p } - g _ { c } \right) 2 \sp { p }\tag{6}
$$

where: $g c$ is the gray value of the center pixel, gp is the gray value of the adjacent pixel, P is the number of neighbors, s(x) is a threshold function defined as:

$$
\mathrm { ~ s ~ } ( x ) = \left\{ \begin{array} { l l } { 1 ~ i f ~ x ~ \ge ~ 0 } \\ { ~ 0 ~ x ~ < 0 } \end{array} \right.
$$

Here, LBP was applied to identify textural modification in MRI images, which assisted in recognizing tumorafected regions.

The utilized feature representation methods: CNN, DWT, FFT, Gabor Filters, GLRLM, and LBP are pivotal in transforming raw MRI images into significant data for classification. Each method played promising role in identifying distinct attributes, such as spatial patterns, frequency data, and texture properties, which help the learning algorithms to detect brain tumors accurately and efective.

## Evaluation metrics and performance measures

Here in this section, it is thoroughly defined how well the various feature representation techniques worked with diferent machine learning models for categorizing the MRI scan images. The objective is to know how these models with a combination of various feature representation techniques worked to distinguish the benign and malignant brain images. To accomplish this task, accuracy, F1-score, precision, sensitivity, and specificity were included in the set of defined evaluation criteria. Every metric was computed to evaluate the result of these classification techniques. These five basic evaluation metrics are defined as under which were used in series of publication<sup>70–74</sup>. In addition, five-fold stratified cross-validation test was adopted to ensure fair evaluation and address class imbalance. In five-fold cross validation test, data was split into five folds such that each fold preserved the same class distribution as the overall dataset. This method ensures that every tumor subtypes, including minority classes are proportionally represented in each training and validation test, this technique enhances the reliability of performance metrics.

Accuracy:

The overall correctness of a classifier can be evaluated by these metrics. It is considered the most efective aspect to be measured. The accuracy is actually when we divide the number of correct predictions by the total number of predictions. The formula for this is given as under.

$$
A c c u r a c y = { \frac { T r u e \ P o s i t i v e s \ + \ T r u e \ N e g a t i v e s } { F a l s e \ P o s i t i v e s \ + \ F a l s e \ N e g a t i v e s \ + \ T r u e \ P o s i t i v e s \ + \ T r u e \ N e g a t i v e s } }\tag{7}
$$

Sensitivity (recall or true positive rate):

The ratio of true positive cases and true positives plus false negatives is called the sensitivity of the model. It can be calculated using the following formula.

$$
S e n s i t i v i t y = { \frac { T r u e \ P o s i t i v e s } { F a l s e \ N e g a t i v e s + T r u e \ P o s i t i v e s | } }\tag{8}
$$

Precision:

The ratio between true positives and true positives plus false positives is known as the precision of the classifier. The formula of precision is given below.

$$
P r e c i s i o n = { \frac { T r u e \ P o s i t i v e s } { F a l s e \ P o s i t i v e s \ + \ T r u e \ P o s i t i v e s } }\tag{9}
$$

F1 score:

The F1 score is also a very important measure of the eficiency of a model. It can be calculated by using the precision and sensitivity as given.

$$
F 1 ~ S c o r e = \frac { 2 ~ * ~ P r e c i s i o n ~ * ~ S e n s i t i v i t y } { P r e c i s i o n ~ + ~ S e n s i t i v i t y }\tag{10}
$$

Specificity (true negative rate):

The specificity of a model determines how a model distinguishes between all the actual negatives and the calculated negatives. The formula is given as under.

$$
S p e c i f i c i t y = \frac { T r u e ~ N e g a t i v e s } { T r u e ~ N e g a t i v e s + F a l s e ~ P o s i t i v e s }\tag{11}
$$

• True Positives are those cases that are correctly predicted as positive.

• False Positives are those cases that are erroneously predicted as positive.

• True Negatives are those cases that are correctly predicted as negative.

• False Negatives are those cases that are erroneously predicted as negative.

## Result analysis and discussion

This section discusses comprehensively the performance assessment of various feature representation techniques (CNN, DWT, FFT, Gabor, GLRLM, and LBP) in conjunction with several machine learning and deep learning algorithms. Performance Analysis of Feature Extraction Techniques with ML Classifiers. Class weighting is implemented during training to rectify the class imbalance in the sizable dataset, giving the minority (nontumor) class larger loss penalties. Furthermore, we reported additional performance metrics that better reflect performance on imbalanced datasets, such as precision and F1-score, in order to ensure a fair evaluation.

## CNN feature space

CNN serves as both a feature extraction and classification technique in this work. To extract deep features from the MRI images, we used a pre-trained CNN model (such as imageNet in the manuscript) and frozen the convolutional layers during inference. For the final classification, these features were subsequently fed into diferent machine learning classifiers (such SVM, KNN, or PNN). The CNN was not fine-tuned or trained end-to-end with the classifier. This fixed-feature extraction approach was chosen to reduce training time and computational cost, and to evaluate the efectiveness of transfer learning in combination with classical classifiers.

In case of classification, a CNN is employed in an end-to-end fashion, where the network simultaneously learns feature representations and performs classification. The convolutional layers act as learnable feature extractors, and the resulting features are passed through fully connected layers to generate the final classification. All layers are trained jointly from scratch without using pre-trained weights.

The CNN-based image representation approach demonstrated remarkable performance across all learning algorithms are reported in Table 2. Among these algorithms, the SVC algorithm outperformed compared to other algorithms, and achieved 96.6% accuracy and 98% specificity, highlighting its efectiveness in identifying MRI-based tumors. So the success rates of SVC are balanced in the case of all used metrics. On the other hand, RF and CNN exhibited high precision and specificity ensuring reliable positive predictions. In contrast, the sensitivity of RF and CNN is unbalanced. So the models are biased toward one class. kNN obtained 81% accuracy and 80% F1-score, while sustaining success rates. While PNN performed moderately achieved 83% accuracy and 88% specificity. In this study, robust features such as spatial hierarchies in images are captured due to CNN capabilities and eradicating the requirements of manual feature engineering. In contrast, it is computationally expensive owing to its complex structure and also shows that it is not optimal for distance-based learning. In CNN the learned attributes are not interpretable, so it is sometimes dificult to make a decision. It is overfit in the case of a small dataset.

<table><tr><td rowspan=1 colspan=1>ML Models</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Specificity</td></tr><tr><td rowspan=1 colspan=1>kNN</td><td rowspan=1 colspan=1>81%</td><td rowspan=1 colspan=1>77%</td><td rowspan=1 colspan=1>83%</td><td rowspan=1 colspan=1>80%</td><td rowspan=1 colspan=1>85%</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>92%</td><td rowspan=1 colspan=1>85%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>92%</td><td rowspan=1 colspan=1>100%</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>96%</td><td rowspan=1 colspan=1>92%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>96%</td><td rowspan=1 colspan=1>100%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>96.6%</td><td rowspan=1 colspan=1>95%</td><td rowspan=1 colspan=1>96.7%</td><td rowspan=1 colspan=1>96.1%</td><td rowspan=1 colspan=1>98%</td></tr><tr><td rowspan=1 colspan=1>PNN</td><td rowspan=1 colspan=1>83%</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>87%</td><td rowspan=1 colspan=1>82%</td><td rowspan=1 colspan=1>88%</td></tr></table>

Table 2. Performance evaluation metrics of various learning algorithms using CNN feature space.

<table><tr><td rowspan=1 colspan=1>ML Models</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Specificity</td></tr><tr><td rowspan=1 colspan=1>kNN</td><td rowspan=1 colspan=1>86%</td><td rowspan=1 colspan=1>93%</td><td rowspan=1 colspan=1>81%</td><td rowspan=1 colspan=1>87%</td><td rowspan=1 colspan=1>79%</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>90%</td><td rowspan=1 colspan=1>81%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>89%</td><td rowspan=1 colspan=1>100%</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>95%</td><td rowspan=1 colspan=1>95%</td><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>95%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>95.9%</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>97.0%</td><td rowspan=1 colspan=1>95.4%</td><td rowspan=1 colspan=1>97.8%</td></tr><tr><td rowspan=1 colspan=1>PNN</td><td rowspan=1 colspan=1>93%</td><td rowspan=1 colspan=1>99%</td><td rowspan=1 colspan=1>89%</td><td rowspan=1 colspan=1>93%</td><td rowspan=1 colspan=1>87%</td></tr></table>

Table 3. Performance evaluation metrics of various used learning algorithms utilizing DWT feature space.

<table><tr><td rowspan=1 colspan=1>ML Models</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Specificity</td></tr><tr><td rowspan=1 colspan=1>kNN</td><td rowspan=1 colspan=1>76%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>68%</td><td rowspan=1 colspan=1>80%</td><td rowspan=1 colspan=1>55%</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>83%</td><td rowspan=1 colspan=1>91%</td><td rowspan=1 colspan=1>79%</td><td rowspan=1 colspan=1>84%</td><td rowspan=1 colspan=1>76%</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>93%</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>93%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>95.2%</td><td rowspan=1 colspan=1>93%</td><td rowspan=1 colspan=1>96.44%</td><td rowspan=1 colspan=1>95.6%</td><td rowspan=1 colspan=1>97.4%</td></tr><tr><td rowspan=1 colspan=1>PNN</td><td rowspan=1 colspan=1>75%</td><td rowspan=1 colspan=1>95%</td><td rowspan=1 colspan=1>68%</td><td rowspan=1 colspan=1>79%</td><td rowspan=1 colspan=1>56%</td></tr></table>

Table 4. Performance evaluation metrics of various used learning algorithms utilizing FFT feature space.

## Discrete wavelet transform (DWT) feature space

Discrete Wavelet Transform (DWT) yielded robust outcomes in which SVC still led with 95.9% accuracy and 97.8%% specificity ensuring reliable predictions are shown in Table 3. In a sequel, CNN and RF also exhibited strong classification power with 100% precision. The kNN model showed excellent sensitivity of 93% and a balanced F1-score, emphasizing its capability to recognize true positives efectively. In addition, PNN also performed well with 99% sensitivity and 93% accuracy. The significance of DWT is the multi-resolution analysis which extracts both frequency and spatial information. In contrast, it requires careful consideration in the case of wavelet functions and decomposition level selection.

## FFT feature space

Fast Fourier Transform (FFT) features obtained optimal success rates using SVC, which are 97.4% specificity and over 95.2% accuracy reported in Table 4. CNN maintained a high precision rate of 93%, while kNN demonstrated efectiveness in true positive detection with is sensitivity of 97%. The sensitivity of RF is high but low specificity which reveals imbalanced classification results. FFT is computationally eficient compared to DWT along with extracting global frequency information. In contrast, it lacks spatial information, which restrains its efectiveness for localized information. FFT is sensitive to noise and assumes the signal is periodic, which may not be always true in real-time applications.

## Filters Gabor feature space

The Gabor representation method demonstrated moderated results which are shown in Table 5. In this method, SVC continuously reflected exceptional performance with 93.6% accuracy and 95.4% specificity. CNN provided a commendable precision rate of 90%, while kNN and PNN achieved comparatively lesser accuracy at 74%, indicating that Gabor features may lack of discriminative power of alternative methods. The importance of the Gabor method is the extraction of texture and edge information. On the other hand, it is computationally expensive owing to the multi-scale and multi-orientation nature of Gabor filters. Gabor filter focuses on local information while ignoring global patterns in data.

<table><tr><td rowspan=1 colspan=1>ML Models</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Specificity</td></tr><tr><td rowspan=1 colspan=1>kNN</td><td rowspan=1 colspan=1>74%</td><td rowspan=1 colspan=1>67%</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>81%</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>81%</td><td rowspan=1 colspan=1>73%</td><td rowspan=1 colspan=1>87%</td><td rowspan=1 colspan=1>79%</td><td rowspan=1 colspan=1>88%</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>89%</td><td rowspan=1 colspan=1>88%</td><td rowspan=1 colspan=1>90%</td><td rowspan=1 colspan=1>89%</td><td rowspan=1 colspan=1>91%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>93.6%</td><td rowspan=1 colspan=1>91%</td><td rowspan=1 colspan=1>94.4%</td><td rowspan=1 colspan=1>94.2%</td><td rowspan=1 colspan=1>95.4%</td></tr><tr><td rowspan=1 colspan=1>PNN</td><td rowspan=1 colspan=1>74%</td><td rowspan=1 colspan=1>67%</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>81%</td></tr></table>

Table 5. Performance evaluation metrics of used learning algorithms utilizing Gabor feature space.

<table><tr><td rowspan=1 colspan=1>ML Models</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Specificity</td></tr><tr><td rowspan=1 colspan=1>kNN</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>84%</td><td rowspan=1 colspan=1>77%</td><td rowspan=1 colspan=1>86%</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>68%</td><td rowspan=1 colspan=1>85%</td><td rowspan=1 colspan=1>69%</td><td rowspan=1 colspan=1>88%</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>72%</td><td rowspan=1 colspan=1>82%</td><td rowspan=1 colspan=1>77%</td><td rowspan=1 colspan=1>84%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>94.8%</td><td rowspan=1 colspan=1>93%</td><td rowspan=1 colspan=1>93.9%</td><td rowspan=1 colspan=1>94.5%</td><td rowspan=1 colspan=1>96.6%</td></tr><tr><td rowspan=1 colspan=1>PNN</td><td rowspan=1 colspan=1>81%</td><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>89%</td><td rowspan=1 colspan=1>78%</td><td rowspan=1 colspan=1>91%</td></tr></table>

Table 6. Performance evaluation metrics for various used classifiers utilizing GLRLM feature space.

<table><tr><td rowspan=1 colspan=1>ML Models</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Specificity</td></tr><tr><td rowspan=1 colspan=1>kNN</td><td rowspan=1 colspan=1>95%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>93%</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>95%</td><td rowspan=1 colspan=1>98%</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>92%</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>97.8%</td><td rowspan=1 colspan=1>97.7%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>96%</td><td rowspan=1 colspan=1>97.9%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>98.06%</td><td rowspan=1 colspan=1>97.4%</td><td rowspan=1 colspan=1>97.44%</td><td rowspan=1 colspan=1>96.20%</td><td rowspan=1 colspan=1>98.7%</td></tr><tr><td rowspan=1 colspan=1>PNN</td><td rowspan=1 colspan=1>93%</td><td rowspan=1 colspan=1>97%</td><td rowspan=1 colspan=1>90%</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>89%</td></tr></table>

Table 7. Performance evaluation metrics for various classifiers utilizing LBP feature space.

<table><tr><td rowspan=1 colspan=1>Feature Space</td><td rowspan=1 colspan=1>χ² (Chi-Square) Statistic</td><td rowspan=1 colspan=1>P-value (α=0.05)</td><td rowspan=1 colspan=1>Conclusion</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>19.8</td><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=1>Significant</td></tr><tr><td rowspan=1 colspan=1>DWT</td><td rowspan=1 colspan=1>14.5</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>Significant</td></tr><tr><td rowspan=1 colspan=1>FFT</td><td rowspan=1 colspan=1>10.7</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>Not significant</td></tr><tr><td rowspan=1 colspan=1>GRLM</td><td rowspan=1 colspan=1>16.2</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>Not significant</td></tr><tr><td rowspan=1 colspan=1>Gabor filter</td><td rowspan=1 colspan=1>11.9</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>Not significant</td></tr><tr><td rowspan=1 colspan=1>LBP</td><td rowspan=1 colspan=1>19.3</td><td rowspan=1 colspan=1>0.003</td><td rowspan=1 colspan=1>Significant</td></tr></table>

Table 8. Chi-square and P-value of various feature representation schemes.

## GLRLM feature space

GLRLM-based information exhibited a robust correlation with SVC, obtaining an accuracy of 94.8% and 96.6% specificity are shown in Table 6. kNN yielded a balance between recall and precision, while CNN and RF successfully delivered correct positive predictions. In addition, PNN obtained 81% of accuracy which performed moderately. The performance variations indicate that GLRLM is appropriate for specific learning algorithms but not universally optimal. However, GLRLM representation methods extract texture information eficiently while its ability is limited to capture spatial relationships in images.

## LBP feature extraction

Local Binary Pattern (LBP) features show excellent outcomes which are shown in Table 7. SVC again obtained the highest success rate which is 98.06. The sensitivity and specificity of SVC are balanced. CNN yielded the second-highest accuracy of 97.8%, succeeded by RF at 95%. The balanced sensitivity, precision, and F1-score across learning algorithms validated the robustness of LBP features in brain tumor classification. LBP is efective in collecting local texture patterns and is also eficient and simple computationally.

In order to investigate the contribution of each feature representation scheme, statistical test chi-square and p-value are computed which is illustrated in Table  8. The results indicate that CNN, DWT, and LBP exhibit a significant impact (p < 0.05), thereby confirming their importance in the classification process. The results indicate that FFT, GRLM, and Gabor do not achieve statistical significance (p > 0.05), suggesting a minimal contribution to classification accuracy.

## Performance metrics of learning algorithms using large training dataset

The empirical results reveal that leaning algorithms have achieved the highest success rates on LBP feature representation methods. In order to further investigate and highlight the strength of LBP feature representation methods, the large dataset is expressed by the LBP method. In this dataset, the total number of MRI images are 7023 of which 5723 images are for training and 1311 images are for testing. The performance of learning algorithms on LBP feature space is shown in Table  9. The highest accuracy is achieved by CNN with 98.9% while SVC obtained 96.7% accuracy. The sensitivity and specificity of CNN are balanced. On the other hand the performance of kNN, RF, and PNN are comparatively the same. The performance of learning algorithms on LBP feature space is still advanced compared to other learning algorithms in the case of large training datasets.

The AI explainable SHAP analysis (Fig.  2) revealed that the most influential LBP features correspond to regions with high texture contrast and edge definitions. These align with known MRI characteristics of brain tumors. For instance, uniform texture patterns with high SHAP scores were often associated with meningioma, while features indicative of texture heterogeneity were linked to gliomas. This correspondence afirms the model’s clinical interpretability and the usefulness of the LBP-based features in capturing tumor characteristics that have medical significance. In Fig. 2 each row corresponds to a specific feature, where red points denote higher feature values and blue points indicate lower values. The SHAP values represented on the x-axis quantify the influence of features on model predictions. Positive SHAP values derive predictions toward the tumor, while negative values indicate healthy. Various feature subsets with diferent dimensions were evaluated, and the optimal selection of the highest-ranked features markedly improved the model’s prediction performance.

In addition, Accuracy of individual classes of tumors on both datasets of CNN and SVC are listed in Table 10.

In this work, comparison analysis was also carried out, focused on interpretability, deployment feasibility, and computing eficiency in order to substantiate the advantages of our proposed deep learning model over the transformer-based model. Our proposed model performs excellently compared to the Swin Transformer baseline in terms of computational cost, exhibiting about twice the inference performance on an NVIDIA RTX 3080 GPU, about 65% fewer floating-point operations (FLOPs), and 50% less memory utilization. These enhancements make our proposed model a strong contender for deployment in resource-constrained environments, such as rural healthcare settings where restricted hardware capacity and power availability often prohibit the implementation of more intricate transformer architectures. Additionally, our proposed model CNN-SHAP approach, uses feature attribution maps to produce interpretable visualizations in order to recognize the interpretability and generalizability trade-ofs. In medical deep learning, model transparency must guarantee diagnostic reliability without sacrificing clinical acceptability. This conflict has been highlighted in published research by emphasizing the necessity of explainability in high-stakes applications Doshi-Velez and Kim (2017) and Tjoa and Guan (2020). In order to make our method both eficient and useful for actual clinica applications, we strive for a balance between obtaining high performance on benchmark datasets and preserving transparency through SHAP-based post hoc explanations. The performance of the proposed model is evaluated using a stratified five-fold cross-validation test. The mean and standard deviation (mean ± SD) of each class are reported in Table 11. It demonstrates the consistency and reliability of the proposed model on both small and large datasets. In addition, 95% confidence intervals are also computed applying a Student’s t-distribution (df= 4 and sample size n = 5) for each class and shown in Table 12. The narrow confidence intervals indicate high model stability and consistent performance.

## Discussion

A comprehensive performance evaluation indicated significant diferences in the performance of various combinations of feature extraction methods and classifiers. The SVC exhibited consistently high accuracy and specificity, establishing it as a dependable model for the detection of brain tumors in MRI scans. This indicates that SVC demonstrates a high proficiency in diferentiating between tumor and non-tumor cases while maintaining a low rate of false positives. The RF classifier demonstrated high precision, reflecting its efectiveness in producing accurate positive predictions and reducing the occurrence of misclassifications. The CNN model demonstrated significant performance improvements, particularly in minimizing false positive instances (Relu, softmax, Adam), thereby increasing diagnostic reliability. In the meantime, the kNN algorithm demonstrated high sensitivity, successfully identifying true positive cases. This emphasizes its ability to accurately identify the presence of tumors; however, additional optimization may be necessary to minimize false positives.

LBP space has shown significant efectiveness in capturing texture-based features, leading to high classification accuracy. This highlights the significance of selecting a suitable feature representation method to improve tumor detection. Additionally, this study emphasizes the critical importance of feature extraction methods in enhancing diagnostic accuracy within medical imaging applications.

<table><tr><td rowspan=1 colspan=1>ML Models</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Sensitivity</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>F1 Score</td><td rowspan=1 colspan=1>Specificity</td></tr><tr><td rowspan=1 colspan=1>kNN</td><td rowspan=1 colspan=1>91%</td><td rowspan=1 colspan=1>90%</td><td rowspan=1 colspan=1>92%</td><td rowspan=1 colspan=1>90%</td><td rowspan=1 colspan=1>92%</td></tr><tr><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>91.6%</td><td rowspan=1 colspan=1>91.2%</td><td rowspan=1 colspan=1>92%</td><td rowspan=1 colspan=1>91%</td><td rowspan=1 colspan=1>92%</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>98.9%</td><td rowspan=1 colspan=1>98.6%</td><td rowspan=1 colspan=1>99%</td><td rowspan=1 colspan=1>98%</td><td rowspan=1 colspan=1>99.2%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>96.7%</td><td rowspan=1 colspan=1>95.4%</td><td rowspan=1 colspan=1>96.2%</td><td rowspan=1 colspan=1>96%</td><td rowspan=1 colspan=1>98.%</td></tr><tr><td rowspan=1 colspan=1>PNN</td><td rowspan=1 colspan=1>92%</td><td rowspan=1 colspan=1>94%</td><td rowspan=1 colspan=1>90%</td><td rowspan=1 colspan=1>92%</td><td rowspan=1 colspan=1>90%</td></tr></table>

Table 9. Performance evaluation metrics for various classifiers utilizing LBP feature space on large dataset.

![](images/7a541726b20ae1ed4d7de694fa73b4272dbb30930cd328737829c220c0744c6a.jpg)  
Fig. 2. SHAP interpretation of LBP features.

In the case of training dataset1, the prediction outcomes of SVC are reported the highest using all feature spaces. SVC obtained 98.06% accuracy on LBP feature space. The second highest success rates are obtained by CNN and are comparatively the same as SVC. CNN yielded 97.7% accuracy on LBP feature space. Further, evaluating the generalization and resilience of learning algorithms on the LBP feature representation method another large dataset training dataset2 is used. In the case of a large dataset expressed by the LBP feature representation method, CNN was identified as the highest-performing model, with a weighted accuracy of 98.9%, and SVC obtained 96.7% accuracy. In the case of the small dataset, the machine learning algorithm

<table><tr><td rowspan=1 colspan=1>ML models</td><td rowspan=1 colspan=3>Tumors classes</td><td rowspan=1 colspan=1>Non-tumors</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Pituitary</td><td rowspan=1 colspan=1>Meningioma</td><td rowspan=1 colspan=1>Glioma</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=5>Small dataset</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>97.0%</td><td rowspan=1 colspan=1>96.9%</td><td rowspan=1 colspan=1>97.4%</td><td rowspan=1 colspan=1>96%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>98.68%</td><td rowspan=1 colspan=1>97.6%</td><td rowspan=1 colspan=1>97.9%</td><td rowspan=1 colspan=1>96.20%</td></tr><tr><td rowspan=1 colspan=5>Large dataset</td></tr><tr><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>99.3%</td><td rowspan=1 colspan=1>98.4%</td><td rowspan=1 colspan=1>99%</td><td rowspan=1 colspan=1>99.2%</td></tr><tr><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>97.6%</td><td rowspan=1 colspan=1>96.4%</td><td rowspan=1 colspan=1>96.1%</td><td rowspan=1 colspan=1>98.%</td></tr></table>

Table 10. Accuracy of individual classes on both dataset of CNN and SVC.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Pituitary (%)</td><td rowspan=1 colspan=1>Meningioma (%)</td><td rowspan=1 colspan=1>Glioma (%)</td><td rowspan=1 colspan=1>Non-Tumor (%)</td></tr><tr><td rowspan=1 colspan=1>Small</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>97.0±0.45</td><td rowspan=1 colspan=1>96.9±0.51</td><td rowspan=1 colspan=1>97.4±0.37</td><td rowspan=1 colspan=1>96.0±0.59</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>98.68±0.32</td><td rowspan=1 colspan=1>97.6±0.44</td><td rowspan=1 colspan=1>97.9±0.41</td><td rowspan=1 colspan=1>96.20±0.50</td></tr><tr><td rowspan=1 colspan=1>Large</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>99.3±0.21</td><td rowspan=1 colspan=1>98.4±0.26</td><td rowspan=1 colspan=1>99.0±0.24</td><td rowspan=1 colspan=1>99.2±0.19</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>97.6±0.39</td><td rowspan=1 colspan=1>96.4±0.42</td><td rowspan=1 colspan=1> $9 6 . 1 \pm 0 . 4 7$ </td><td rowspan=1 colspan=1> $9 8 . 0 \pm 0 . 3 3 $ </td></tr></table>

Table 11. Accuracy (mean ± SD) of individual classes on both small and large datasets.

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Pituitary (%)</td><td rowspan=1 colspan=1>Meningioma (%)</td><td rowspan=1 colspan=1>Glioma (%)</td><td rowspan=1 colspan=1>Non-Tumor (%)</td></tr><tr><td rowspan=1 colspan=1>Small</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>96.44-97.56</td><td rowspan=1 colspan=1>96.27-97.53</td><td rowspan=1 colspan=1>96.94-97.86</td><td rowspan=1 colspan=1>95.27-96.73</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>98.28-99.08</td><td rowspan=1 colspan=1>97.05-98.15</td><td rowspan=1 colspan=1>97.39-98.41</td><td rowspan=1 colspan=1>95.58-96.82</td></tr><tr><td rowspan=1 colspan=1>Large</td><td rowspan=1 colspan=1>CNN</td><td rowspan=1 colspan=1>99.04-99.56</td><td rowspan=1 colspan=1>98.08-98.72</td><td rowspan=1 colspan=1>98.70-99.30</td><td rowspan=1 colspan=1>98.96-99.44</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>SVC</td><td rowspan=1 colspan=1>97.12-98.08</td><td rowspan=1 colspan=1>95.88-96.92</td><td rowspan=1 colspan=1>95.52–96.68</td><td rowspan=1 colspan=1>97.59-98.41</td></tr></table>

Table 12. 95% confidence intervals (CI) for accuracy.

SVC is well performed and the deep learning model CNN is very close in terms of accuracy. In contrast, the predictive outcomes of CNN in the case of large datasets are much better compared to SVC. The empirical results elaborate that as the number of images increased the performance of SVC reduced while the performance of CNN enhanced. So, machine learning algorithms are useful in the case of small datasets, in contrast, CNN performs well in the case of large datasets.

In conclusion, the results highlight the essential relationship between feature extraction methods and machine learning and deep learning classifiers in enhancing the accuracy of brain tumor detection. The findings demonstrate that LBP-based feature extraction yields strong feature representations. However, traditional methods such as DWT and Gabor filters also show competitive performance when utilized alongside efective classifiers like CNN, SVC, and RF. The insights presented have significant implications for enhancing the accuracy and reliability of automated diagnostic systems within the field of medical imaging.

CNN identifies spatial hierarchies of features, efectively capturing patterns including edges, textures, and intricate structures, thereby demonstrating high suitability for image-based tasks. The deep learning architecture enables the achievement of higher accuracy in large and complex datasets. The model demonstrates strong generalization capabilities across diverse image datasets through the acquisition of robust feature representations. The system efectively manages variations including rotation, scale, and intensity diferences by utilizing convolutional layers and pooling mechanisms. SVC is dependent on pre-extracted features, indicating that its performance is significantly influenced by the quality and relevance of these features. The system does not acquire spatial hierarchies; instead, it identifies optimal decision boundaries for classification based on the input features. It demonstrates robust performance when supplied with well-extracted and relevant features. It may encounter dificulties when processing high-dimensional and raw image data unless extensive preprocessing is applied. The system exhibits increased sensitivity to variations in image data due to its lack of inherent spatial feature learning, resulting in reduced robustness when addressing diverse and complex image patterns. It is computationally less expensive in comparison to deep learning models. The method demonstrates efective performance on smaller datasets and exhibits ease of training; however, it may not achieve the same level of proficiency as CNN in processing highly complex image data. However, the dependence on extracted features indicates that performance may be suboptimal when encountering unseen or complex variations in medical images.

SVC demonstrates high efectiveness in structured classification tasks that utilize well-engineered features. However, CNN exhibits superior performance in medical image analysis, attributed to its capability to learn intricate spatial features, achieve robust generalization, and uphold elevated accuracy and specificity. Considering the intricacies involved in MRI brain tumor detection, CNN is identified as a more efective option for automated and high-precision medical imaging applications.

<table><tr><td rowspan=1 colspan=1>Study</td><td rowspan=1 colspan=1>Methodology</td><td rowspan=1 colspan=1>Accuracy (%)</td><td rowspan=1 colspan=1>Sensitivity (%)</td><td rowspan=1 colspan=1>Precision (%)</td><td rowspan=1 colspan=1>F1-Score (%)</td><td rowspan=1 colspan=1>Specificity (%)</td></tr><tr><td rowspan=1 colspan=1>CNN-based model 75</td><td rowspan=1 colspan=1>CNN as a feature extractor &amp; classifier</td><td rowspan=1 colspan=1>96.00</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>91.14</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>Auto-encoder Network76</td><td rowspan=1 colspan=1>Auto-encoder + 2D CNN</td><td rowspan=1 colspan=1>95.63,</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>kNN77</td><td rowspan=1 colspan=1>kNN classifier</td><td rowspan=1 colspan=1>86.00</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>CNN + classic Architecture78</td><td rowspan=1 colspan=1>CNN hybrid model</td><td rowspan=1 colspan=1>96.00</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Gradient boosting79</td><td rowspan=1 colspan=1>Gradient Boosting</td><td rowspan=1 colspan=1>92.40</td><td rowspan=1 colspan=1>94.40</td><td rowspan=1 colspan=1>85</td><td rowspan=1 colspan=1>89.5</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Brain tumor Classification80</td><td rowspan=1 colspan=1>Proposed classification method</td><td rowspan=1 colspan=1>97.15</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1>97</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>BRATS 2018 Dataset81</td><td rowspan=1 colspan=1>Various approaches</td><td rowspan=1 colspan=1>94.60</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Proposed model</td><td rowspan=1 colspan=1>LBP feature extraction + CNN</td><td rowspan=1 colspan=1>98.9%</td><td rowspan=1 colspan=1>98.6%</td><td rowspan=1 colspan=1>99%</td><td rowspan=1 colspan=1>98%</td><td rowspan=1 colspan=1>99.2%</td></tr></table>

Table 13. Comparison of the proposed model with existing models.

## Comparative analysis of exiting models with propose model

This study distinguishes itself by employing multiple machine learning and deep learning algorithms in conjunction with diverse feature representation strategies, validating a comprehensive comparative analysis with existing research are reported in Table 13. The empirical results highlight the significance of our proposed approach in enhancing classification performance. For instance, a prior study<sup>75</sup> reported an accuracy of 93.3% using a CNN-based model. Similarly, an auto-encoder network and 2D CNN were proposed in<sup>76</sup>, attaining 95.63% and 96.47% accuracy, respectively. The kNN classifier in<sup>77</sup> achieved an accuracy of 86%. A novel hybrid approach combining CNN with classical architectures was proposed in<sup>78</sup>, yielding 96% accuracy along with 95% sensitivity and precision. Gradient Boosting, identified as the best-performing classifier in<sup>79</sup>, achieved an accuracy of 92.4%, an F1-score of 89.5%, 94.4% sensitivity, and 85% precision. Another brain tumor classification study<sup>80</sup> attained 97.15% accuracy with 97% precision, recall, and F1-score. To further validate the robustness of the model, a new model was introduced BRATS 2018 obtained 94.6%, accuracy, respectively<sup>81</sup>. In contrast, our proposed model achieved 98.9% accuracy, 98.6% sensitivity, 99.0% precision, and 99.2% specificity.

The proposed model exhibits enhanced performance across all evaluated metrics compared to existing models. The integration of LBP feature space along with CNN for classification likely contributes to its high accuracy, sensitivity, precision, F1-score, and specificity. The improved performance of the model highlights the efectiveness of merging texture-based feature extraction with the deep learning algorithm CNN for MRIbased brain tumor classification. Consequently, it presents itself as a viable option for brain tumor classification tasks, especially in medical applications where minimizing false positives and false negatives is critical. However, further, the proposed model is validated on larger and more diverse datasets in order to confirm its generalizability and robustness.

## Conclusion

This study investigates the efectiveness of various feature representation techniques, including LBP, Gabor filters, DWT, FFT, CNN, and GLRLM. It also examines their application with diferent machine learning classifiers, such as CNN, RF, SVC, PNN, and CNN.

The experimental findings indicate that LBP, in conjunction with SVC and CNN, attains elevated specificity and accuracy, positioning it as a viable method for MRI-based tumor diagnosis. The statistical analysis employing chi-square and p-value tests validates the significant influence of LBP feature representation on brain tumor identification. In the analysis of smaller datasets, the CNN demonstrated an accuracy of 97.8%, whereas the SVC achieved an accuracy of 98.06%. Subsequent assessment using an expanded benchmark dataset indicated that the CNN model achieved the highest performance, recording an accuracy of 98.9%, while the SVC mode followed with an accuracy of 96.7%. SVC demonstrates efectiveness in structured classification tasks that utilize well-engineered features. In contrast, CNN exhibits superior performance in medical imaging, attributable to its capacity to learn complex spatial features and generalize eficiently. In addition, The SHAP analysis was used to identify the most significant features in classification. The proposed model demonstrates superior performance compared to existing approaches across all critical metrics, such as accuracy, sensitivity, precision, F1-score, and specificity. The combination of texture-based feature extraction and deep learning, specifically CNN, improves the predictive capabilities of the model, rendering it particularly efective for the classification of brain tumors in MRI scans. This study highlights the necessity of employing advanced feature extraction and classification techniques to minimize false positives and false negatives in medical applications. In brain tumor classification challenges, this study shows that traditional texture-based feature extraction techniques can perform competitively when paired with shallow classifiers. These techniques are useful substitutes or supplements to deep learning because they ofer improved interpretability, lower computing overhead, and increased robustness across datasets, especially in clinical settings with low data and high interpretability.

Future research should concentrate on validating the proposed model using larger and more diverse datasets to enhance confirmation of its robustness and generalizability. Future research will expand the analysis to include transformer-based architectures and self-supervised models like DINO and CLIP, especially as larger annotated medical datasets become available, even though this study concentrates on classical and CNN-based models because of their efectiveness and suitability for smaller datasets.

## Data availability

The datasets used and/or analysed during the current study available from the corresponding author on reasonable request.

Received: 18 April 2025; Accepted: 4 August 2025

Published online: 11 August 2025

## References

1. Qureshi, S. A. et al. Intelligent ultralight deep learning model for multiclass brain tumor detection. Appl. Sci. 12, 3715 (2022).

2. Zahoor, M. M. et al. A new deep hybrid boosted and ensemble learning–based brain tumor analysis using MRI. Sensors 22, 2726 (2022).

3. Arabahmadi, M., Farahbakhsh, R. & Rezazadeh, J. Deep learning for smart healthcare: a survey on brain tumor detection from medical imaging. Sensors 22, 1960 (2022).

4. Prabukumar, M., Agilandeeswari, L. & Ganesan, K. An intelligent lung cancer diagnosis system using cuckoo search optimization and support vector machine classifier. J. Ambient Intell. Humaniz. Comput. 10, 267–293 (2019).

5. Tandel, G. S. et al. A review on a deep learning perspective in brain cancer classification. Cancers 11, 111 (2019).

6. Raut, G. et al. Deep learning approach for brain tumor detection and segmentation. In Proceedings of the IEEE International Conference on Convergence to Digital World – Quo Vadis (ICCDW) (IEEE, 2020).

7. DeAngelis, L. M. & Brain tumors. N. Engl. J. Med. 344, 114–123 (2001).

8. Borole, V. Y., Nimbhore, S. S. & Kawthekar, D. S. Image processing techniques for brain tumor detection: a review. Int. J. Emerg. Trends Technol. Comput. Sci. 4, 2 (2015).

9. Amin, J. et al. Big data analysis for brain tumor detection: deep convolutional neural networks. Future Gener Comput. Syst. 87, 290–297 (2018).

10. Iorgulescu, J. B. et al. Molecular biomarkerdefined brain tumors: epidemiology, validity, and completeness in the united States. NeuroOncol 24, 1989–2000 (2022).

11. Mabray, M. C., Barajas, R. F. & Cha, S. Modern brain tumor imaging. Brain Tumor Res. Treat. 3, 8–23 (2015).

12. Cha, S. Update on brain tumor imaging: from anatomy to physiology. Am. J. Neuroradiol. 27, 475–487 (2006).

13. ElDahshan, E. S. A. et al. Computeraided diagnosis of human brain tumor through MRI: a survey and a new algorithm. Expert Syst. Appl. 41, 5526–5545 (2014).

14. Meng, Y. et al. Exposure to lead increases the risk of meningioma and brain cancer: a metaanalysis. J. Trace Elem. Med. Biol. 60, 126474 (2020).

15. Tiwari, P. et al. CNN based multiclass brain tumor detection using medical imaging. Comput. Intell. Neurosci. 9394312 (2022).

16. AnayaIsaza, A. & MeraJiménez, L. Data augmentation and transfer learning for brain tumor detection in magnetic resonance imaging. IEEE Access. 10, 23217–23233 (2022).

17. Lotlikar, V. S., Satpute, N. & Gupta, A. Brain tumor detection using machine learning and deep learning: a review. Curr. Med. Imaging. 18, 604–622 (2022).

18. Xie, Y. et al. Convolutional neural network techniques for brain tumor classification (from 2015 to 2022): review, challenges, and future perspectives. Diagnostics 12, 1850 (2022).

19. Krizhevsky, A., Sutskever, I. & Hinton, G. E. Imagenet classification with deep convolutional neural networks. Adv. Neural Inf. Process. Syst. 25 (2012).

20. Amodei, D. et al. Deep Speech 2: Endtoend speech recognition in English and Mandarin. In Proceedings of the International Conference on Machine Learning (PMLR, 2016).

21. Fioretto, F. et al. A lagrangian dual framework for deep neural networks with constraints. ArXiv https://arxiv.org/abs/2001.09394 (2020).

22. Vinyals, O., Fortunato, M. & Jaitly, N. Pointer networks. Adv. Neural Inf. Process. Syst. 28 (2015).

23. Khalil, E. et al. Learning combinatorial optimization algorithms over graphs. Adv. Neural Inf. Process. Syst. 30 (2017).

24. Charles, O. O. et al. Machine learning predicts 1[(4fluorophenyl)methyl]indole-2,3dione as drug lead for peptide deformylase in Plasmodium falciparum. Medinformatics (2024).

25. Kool, W., Van Hoof, H. & Welling, M. Attention, learn to solve routing problems! arXiv (2018). https://arxiv.org/abs/1803.08475

26. Donti, P., Amos, B. & Kolter, J. Z. Taskbased endtoend model learning in stochastic optimization. Adv. Neural Inf. Process. Syst. 30 (2017).

27. Amos, B. & Kolter, J. Z. OptNet: Diferentiable optimization as a layer in neural networks. In Proceedings of the International Conference on Machine Learning (PMLR, 2017).

28. Wilder, B., Dilkina, B. & Tambe, M. Melding the datadecisions pipeline: decisionfocused learning for combinatorial optimization. In Proceedings of the AAAI Conference on Artificial Intelligence (2019).

29. Chattopadhyay, S. Decoding medical diagnosis with machine learning classifiers. Medinformatics https://doi.org/10.47852/bonvie wMEDIN42022583 (2024).

30. Shen, D., Wu, G. & Suk, H. I. Deep learning in medical image analysis. Annu. Rev. Biomed. Eng. 19, 221–248 (2017).

31. Litjens, G. et al. A survey on deep learning in medical image analysis. Med. Image Anal. 42, 60–88 (2017).

32. Suzuki, K. Overview of deep learning in medical imaging. Radiol. Phys. Technol. 10, 257–273 (2017).

33. Hijazi, S., Kumar, R. & Rowen, C. Using Convolutional Neural Networks for Image Recognition. Vol. 91 (Cadence Design Systems Inc., 2015).

34. O’Shea, K. & Nash, R. An introduction to convolutional neural networks. arXiv https://arxiv.org/abs/1511.08458 (2015).

35. Wong, K. K., Fortino, G. & Abbott, D. Deep learningbased cardiovascular image diagnosis: a promising challenge. Future Gener Comput. Syst. 110, 802–811 (2020).

36. Hua, K. L. et al. Computeraided classification of lung nodules on computed tomography images via deep learning technique. Onco Targets Ther. 2015, 2015–2022 (2015).

37. Işın, A., Direkoğlu, C. & Şah, M. Review of MRIbased brain tumor image segmentation using deep learning methods. Procedia Comput. Sci. 102, 317–324 (2016).

38. Sarker, P. et al. Segmentation and classification of lung tumor from 3D CT image using Kmeans clustering algorithm. In ICAEE ’17. (IEEE, (2017).

39. AbdEllah, M. K. et al. Twophase multimodel automatic brain tumour diagnosis system from magnetic resonance images using convolutional neural networks. EURASIP J. Image Video Process. 1–10 (2018).

40. Malathi, M. & Sinthia, P. Brain tumour segmentation using convolutional neural network with tensorflow. Asian Pac. J. Cancer Prev. 20, 2095–2099 (2019).

41. Nalbalwar, R. et al. Detection of brain tumor by using ANN. Image 2, 7 (2014).

42. Amin, J. et al. Detection of brain tumor based on features fusion and machine learning. J. Ambient Intell. Humaniz. Comput. (2018).

43. Usman, K. & Rajpoot, K. Brain tumor classification from multimodality MRI using wavelets and machine learning. Pattern Anal. Appl. 20, 871–881 (2017).

44. Afshar, P., Mohammadi, A. & Plataniotis, K. N. Brain tumor type classification via capsule networks. In Proceedings of the IEEE International Conference on Image Processing (IEEE, 2018).

45. Rehman, A. et al. A deep learningbased framework for automatic brain tumors classification using transfer learning. Circuits Syst. Signal. Process. 39, 757–775 (2020).

46. Gumaei, A. et al. A hybrid feature extraction method with regularized extreme learning machine for brain tumor classification. IEEE Access. 7, 36266–36273 (2019).

47. Hong, J., Zhang, Y. D. & Chen, W. Sourcefree unsupervised domain adaptation for crossmodality abdominal multiorgan segmentation. Knowl. Based Syst. 250, 109155 (2022).

48. Li, S., Zhao, S., Zhang, Y., Hong, J. & Chen, W. Sourcefree unsupervised adaptive segmentation for knee joint MRI. Biomed. Signal. Process. Control. 92, 106028 (2024).

49. Qi, Z. et al. MSLINet: retinal disease detection network based on multisegment localization and multiscale interaction. Front. Cell. Dev. Biol. 13, 1608325 (2025).

50. Hong, J., Yu, S. C. H. & Chen, W. Unsupervised domain adaptation for crossmodality liver segmentation via joint adversarial learning and selflearning. Appl. Soft Comput. 121, 108729 (2022).

51. Phaye, S. S. R. et al. Dense and diverse capsule networks: making the capsules learn better. ArXiv (2018). https://arxiv.org/abs/180 5.04001

52. Paul, J. S. et al. Deep learning for brain tumor classification. In Medical Imaging 2017: Biomedical Applications in Molecular, Structural, and Functional Imaging (SPIE, (2017).

53. Almadhoun, H. R. & AbuNaser, S. S. Detection of brain tumor using deep learning. (2022).

54. Musallam, A. S., Sherif, A. S. & Hussein, M. K. A new convolutional neural network architecture for automatic detection of brain tumors in magnetic resonance imaging images. IEEE Access. 10, 2775–2782 (2022).

55. Nayak, D. R. et al. Brain tumor classification using dense Eficientnet. Axioms 11, 34 (2022).

56. Obeidavi, M. R. & Maghooli, K. Tumor detection in brain MRI using residual convolutional neural networks. In Proceedings of the International Conference on Machine Vision Image Process (MVIP) (IEEE, 2022)

57. Sajid, S., Hussain, S. & Sarwar, A. Brain tumor detection and segmentation in MR images using deep learning. Arab. J. Sci. Eng. 44, 9249–9261 (2019).

58. Woźniak, M., Siłka, J. & Wieczorek, M. Deep neural network correlation learning mechanism for CT brain tumor detection. Neural Comput. Appl. (2021).

59. Amin, J. et al. Brain tumor detection and classification using machine learning: a comprehensive survey. Complex. Intell. Syst. (2021).

60. Mittal, M. et al. Deep learning based enhanced tumor segmentation approach for MR brain images. Appl. Soft Comput. 78, 346– 354 (2019).

61. Narayana, T. L. & Reddy, T. S. IEEE,. An eficient optimization technique to detect brain tumor from MRI images. In Proceedings of the International Conference on Smart System Invention Technology (ICSSIT) (2018).

62. Dixit, A. & Nanda, A. Brain MR image classification via PSO based segmentation. In Proceedings of the 12th International Conference on Contemporary Comput. (IC3) (IEEE, 2019).

63. Çinar, A. & Yildirim, M. Detection of tumors on brain MRI images using the hybrid convolutional neural network architecture. Med. Hypotheses. 139, 109684 (2020).

64. Pashaei, A., Sajedi, H. & Jazayeri, N. Brain tumor classification via convolutional neural network and extreme learning machines. In Proceedings of the 8th International Conference on Computer Knowledge Engineering (ICCKE) (IEEE, 2018).

65. Abdusalomov, A. B., Mukhiddinov, M. & Whangbo, T. K. Brain tumor detection based on deep learning approaches and magnetic resonance imaging. Cancers 15, 4172 (2023).

66. Sangeetha, R. et al. IEEE,. Automatic detection of brain tumor using deep learning algorithms. In Proceedings of the 4th International Conference on Electronics Communication Aerospace Technology (2020).

67. Ullah, N. et al. An efective approach to detect and identify brain tumors using transfer learning. Appl. Sci. 12, 5645 (2022).

68. Disci, R., Gurcan, F. & Soylu, A. Advanced brain tumor classification in MR images using transfer learning and pretrained deep CNN models. Cancers 17, 121 (2025).

69. Gelman, R. & Abramof, M. A pilot study of deep learningbased monocular depth estimation from fundus photographs. Medinformatics (2024).

70. Alam, W. et al. Unveiling the potential pattern representation of RNA 5methyluridine modification sites through a novel feature fusion model leveraging convolutional neural network and tetranucleotide composition. IEEE Access. 12, 10023–10035 (2024).

71. Gill, M., Ahmed, S., Kabir, M. & Hayat, M. A novel predictor for the analysis and prediction of enhancers and their strength via multiview features and deep forest. Information 14, 636 (2024).

72. Jan, A. et al. TargetAMP: computational prediction of antimicrobial peptides by coupling sequential information with evolutionary profile. Comput. Biol. Med. 151, 106311 (2022).

73. Hayat, M., Tahir, M. & Alarfaj, F. K. NLPBCHEns: NLPbased intelligent computational model for discrimination of malaria parasite. Comput. Biol. Med. 149, 105962 (2022).

74. Akbar, S. et al. cACPDeepGram: classification of anticancer peptides via deep neural network and skipgrambased word embedding model. Artif. Intell. Med. 131, 102349 (2022).

75. Mahmud, M. I., Mamun, M. & Abdelgawad, A. A deep analysis of brain tumor detection from MR images using deep learning networks. Algorithms 16, 176 (2023).

76. Saeedi, S. et al. MRIbased brain tumor detection using convolutional deep learning methods and chosen machine learning techniques. BMC Med. Inf. Decis. Mak. 23, 16 (2023).

77. Patil, S. & Kirange, D. Ensemble of deep learning models for brain tumor detection. Procedia Comput. Sci. 218, 2468–2479 (2023).

78. Woźniak, M., Siłka, J. & Wieczorek, M. Deep neural network correlation learning mechanism for CT brain tumor detection. Neural Comput. Appl. 35 (20), 14611–14626 (2023).

79. Ghosh, A. & Kole, A. A Comparative Study of Enhanced Machine Learning Algorithms for Brain Tumor Detection and Classification. Authorea Preprints (2023).

80. Celik, M. & Inik, O. Development of hybrid models based on deep learning and optimized machine learning algorithms for brain tumor multi-classification. Expert Syst. Appl. 238, 122159 (2024).

81. Sharif, M. I. et al. M3BTCNet: multi model brain tumor classification using metaheuristic deep neural network features optimization. Neural Comput. Appl. 36 (1), 95–110 (2024).

## Acknowledgements

The Researchers would like to thank the Deanship of Graduate Studies and Scientific Research at Qassim University for financial support (QU-APC-2025).

## Author contributions

Conceptualization: Maqsood Hayat, Fawaz Khaled Alarfaj, Salem Alkhalaf, Fahad AlturiseData curation: Maqsood Hayat, Salem AlkhalafFunding acquisition: Salem Alkhalaf, Fahad Alturise.Investigation: Asif Rahman, Maqsood Hayat, Fawaz Khaled Alarfaj, Salem Alkhalaf, Fahad AlturiseMethodology: Asif Rahman, Nadeem Iqbal, Maqsood Hayat, Fawaz Khaled Alarfaj, Salem Alkhalaf, Fahad Alturise.Project administration: Maqsood Hayat, Fawaz Khaled Alarfaj, Salem Alkhalaf, Fahad Alturise.Resources: Salem Alkhalaf.Software: Maqsood HayatSupervision: Maqsood Hayat, Salem AlkhalafValidation: Aisf Rahman, Nadeem Iqbal, Maqsood Hayat, Fawaz Khaled Alarfaj. Salem Alkhalaf, Fahad AlturiseVisualization: Asif Rahman, Maqsood Hayat, Fawaz Khaled Alarfaj Writing – original draft: Asif Rahman, Nadeem Iqbal, Maqsood Hayat, Fawaz Khaled Alarfaj. Salem Alkhalaf, Fahad Alturise Writing – review & editing: Asif Rahman, Maqsood Hayat, Fawaz Khaled Alarfaj. Salem Alkhalaf, Fahad Alturise.

## Declarations

## Competing interests

The authors declare no competing interests.

## Additional information

Correspondence and requests for materials should be addressed to M.H. or S.A.

Reprints and permissions information is available at www.nature.com/reprints.

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional afiliations.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2025