# Dimensionality Reduction for Hyperspectral Image Classification

Mohamed Cherifi

Ammar Mesloub

Mohammed Nabil El Korso

Laboratoire Traitement du Signal

Laboratoire Antennes et Dispositifs Micro-Ondes

Universite Paris-Saclay´

Ecole Militaire Polytechnique

Ecole Militaire Polytechnique

Saclay, France

Bordj-El-Bahri, Alger, 16111, Algerie´

Bordj-El-Bahri, Alger, 16111, Algerie´

Tayeb Touhami

Laboratoire Antennes et Dispositifs Micro-Ondes

Ecole Militaire Polytechnique

Bordj-El-Bahri, Alger, 16111, Algerie´

Abdennour Hacine Gharbi

Universite de Bordj-Bou-Ariridj´

Bordj-Bou-Ariridj, Algerie´

Abstract—This paper addresses the issue of supervised classification in the context of hyperspectral satellite images. It deals with two fundamental aspects: dimensionality reduction of data and the selection of appropriate supervised classification techniques.

Firstly, we delve into dimensionality reduction, a critical step in simplifying the management of hyperspectral data. The reduction aims to decrease complexity in terms of memory and computing time. We examine two commonly used methods: Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).

Subsequently, we explore the selection of the most suitable supervised classification algorithms for hyperspectral images. We compare the performance of three methods: K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Random Forest (RF) using real hyperspectral data. The results highlight that the combination of PCA and RF yields the highest overall accuracy and Kappa coefficient.

Index Terms—PCA (Principal Component Analysis), LDA (Linear Discriminant Analysis), KNN (k-Nearest Neighbors), RF (Random Forest), SVM (Support Vector Machine)

## I. INTRODUCTION

Remote Sensing is a technique introduced in the early 1960s for data analysis and interpretation [7]. It has revolutionized our ability to collect vast amounts of satellite data, providing extensive geographical coverage with high temporal frequency compared to other imaging methods. The interpretation of satellite images serves a multitude of purposes, ranging from environmental conservation and management, water resource research, and soil quality studies to post-natural disaster environmental assessments, meteorology simulations, land use and land cover analysis, disaster prevention, and the study of climatic changes[1].

Within the realm of remote sensing, hyperspectral remote sensors take center stage. These sensors, known for their high spectral resolution, are instrumental in monitoring the Earth’s surface. Hyperspectral images (HSI), characterized by their inclusion of more than three bands compared to conventional RGB images, are indispensable tools. They find application across diverse domains, including crop analysis, geological mapping, mineral exploration, defense research, urban investigation, military surveillance.

The interest of dimensionality reduction in the context of hyperspectral images lies in the efficient management of the information contained in these complex images. It not only reduces complexity in terms of memory and computing time but also enhances the performance of analysis and classification techniques. Furthermore, it helps eliminate unwanted noise or redundancies in hyperspectral data, thereby improving classification accuracy and the quality of image interpretation.

In the classification realm, methods are traditionally categorized as supervised or unsupervised. Unsupervised classification [10], often referred to as clustering, seeks to group data in the absence of sample sets. In contrast, supervised classification relies on analyst input and employs sample training sets to identify different classes. In most applications, supervised classification offers distinct advantages over unsupervised approaches. The MATLAB toolbox for supervised classification encompasses a wide array of classical classification algorithms and various classifying methods, all of which are evaluated in this paper, providing a comprehensive comparative analysis of their accuracy and suitability for remote sensing applications for hyperspectral image dataset [12] .

## II. DATASET

In this study, we utilize the Indian Pines (IP) hyperspectral image dataset[2], which has been widely employed in hyperspectral image analysis. The IP dataset was originally gathered using the AVIRIS sensor over the Indian Pines test site located in north-western Indiana. It comprises a wealth of information with the following key characteristics:

• Dimensions: 145 x 145 pixels

• Spectral Bands: 220 bands

• Classes: 16 distinct classes

This dataset serves as an essential resource for various hyperspectral image processing tasks, including classification, feature extraction, and dimensionality reduction. The richness of spectral information and the diversity of land cover classes make it a valuable benchmark for evaluating classification algorithms and conducting experiments in the field of remote sensing.

<table><tr><td>Class</td><td>Description</td><td>Samples</td></tr><tr><td>1</td><td>Alfalfa</td><td>46</td></tr><tr><td>2</td><td>Corn-notill</td><td>428</td></tr><tr><td>3</td><td>Corn-min</td><td>830</td></tr><tr><td>4</td><td>Corn</td><td>237</td></tr><tr><td>5</td><td>Grass/Pasture</td><td>483</td></tr><tr><td>6</td><td>Grass/Trees</td><td>730</td></tr><tr><td>7</td><td>Grass/pasture-mowed</td><td>28</td></tr><tr><td>8</td><td>Hay-windrowed</td><td>478</td></tr><tr><td>9</td><td>Oats</td><td>20</td></tr><tr><td>10</td><td>Soybeans-notill</td><td>972</td></tr><tr><td>11</td><td>Soybeans-min</td><td>2455</td></tr><tr><td>12</td><td>Soybean-clean</td><td>593</td></tr><tr><td>13</td><td>Wheat</td><td>205</td></tr><tr><td>14</td><td>Woods</td><td>1265</td></tr><tr><td>15</td><td>Bldg-Grass-Tree-Drives</td><td>386</td></tr><tr><td>16</td><td>Stone-steel towers</td><td>93</td></tr></table>

TABLE I: Class Descriptions and Sample Counts

## A. Class Reduction Approach

In this section, we will explain our approach to class reduction for hyperspectral image classification using the (IP) dataset. Our methodology is based on the physical characteristics of spectral reflections at different wavelengths. The original classes (16 in total) are consolidated into 6 classes that share similar spectral responses. This reduction simplifies the classification, making it more efficient and interpretable. Table 1 provides a brief description of each class in the original dataset, aiding in the understanding of the dataset’s content and potential applications. Additionally, as per your request, we have included Table 2, which shows the class descriptions and the sum of samples for the reduced classes.

## III. DIMENSIONALITY REDUCTION

Dimensionality Reduction addresses the challenges associated with analyzing multivariate data. When working with extensive datasets, it becomes necessary to reduce dimensionality. The primary objective of dimensionality reduction is to represent the data in a lower-dimensional space while preserving some of its essential properties. Equation 1 illustrates the reduction of a high-dimensional dataset n into a lowerdimensional space m.

$$
\mathbf { X } = { \left[ \begin{array} { l } { x _ { 1 } } \\ { x _ { 2 } } \\ { \vdots } \\ { x _ { n } } \end{array} \right] } \quad { \mathrm { r e d u c e ~ d i m e n s i o n a l i t y } } \quad \mathbf { Y } = { \left[ \begin{array} { l } { y _ { 1 } } \\ { y _ { 2 } } \\ { \vdots } \\ { y _ { m } } \end{array} \right] }\tag{1}
$$

where

$$
n > m
$$

Dimensionality reduction plays a crucial role at the intersection of various fields, including data mining, databases, statistics, pattern recognition, text mining, visualization, artificial intelligence, and optimization. It serves as an essential technique to manage and analyze complex data. Dimensionality reduction encompasses several approaches, including supervised methods like Linear Discriminant Analysis (LDA), Support Vector Machines (SVM), and Hopfield Neural Networks (HNN), as well as unsupervised techniques such as Principal Component Analysis (PCA), Singular Value Decomposition (SVD), and Independent Component Analysis (ICA). In this paper, we focus on unsupervised PCA and supervised LDA for dimensionality reduction in the context of image dataset[13].

## A. Principal Component Analysis

PCA stands as a widely recognized technique for dimensionality reduction, revered in a multitude of fields. Its primary objective is to diminish the dimensionality of a dataset while conserving the maximum possible variance [11]. The essence of PCA lies in its identification of orthogonal axes, termed principal components, that best encapsulate the dataset’s variance characteristics.

These principal components are ordered, with the first capturing the most substantial variance, followed by the second, and so on. By judiciously selecting a subset of these principal components [14], one can effectively represent the data in a lower-dimensional space.

a. For the covariance matrix:

$$
\pmb { \Sigma } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \mathbf { X } _ { i } - \bar { \mathbf { X } } ) ( \mathbf { X } _ { i } - \bar { \mathbf { X } } ) ^ { T }\tag{2}
$$

b. For the eigenvalue decomposition (eigenvalues and eigenvectors):

$$
\pmb { \Sigma } \mathbf { v } = \lambda \mathbf { v }\tag{3}
$$

• Where Σ is the covariance matrix.

$\mathbf { X } _ { i }$ represents individual data points.

• X<sup>¯</sup> is the mean (average) of the data points.

• v is an eigenvector.

• λ is the corresponding eigenvalue.

c. For projecting data onto the principal components:

$$
\mathbf { Y } = \mathbf { X } \mathbf { W }\tag{4}
$$

• where Y is the matrix of projected data.

• X is the original data matrix.

• W is the matrix of eigenvectors chosen as principal components.

PCA has demonstrated its efficacy in diverse domains, spanning image processing, face recognition, and data compression. It proves especially indispensable when confronting high-dimensional datasets, offering efficient means for data representation and visualization. Figure 2 illustrates the visualization of the first three principal components obtained from Principal Component Analysis (PCA). These principal components are essential for reducing the dimensionality of hyperspectral data and enable an efficient representation of the information contained within the image. Figure 1 demonstrates how these principal components capture significant data variance, thereby contributing to the understanding of essential features within the hyperspectral image.

<table><tr><td>Class</td><td>Description</td><td>Samples (Sum of Parent Classes)</td></tr><tr><td>1</td><td>Alfalfa</td><td>46</td></tr><tr><td>2</td><td>Corn-notill-Wheat</td><td> $4 2 8 + 8 3 0 + 2 3 7 + 2 0 5 = 1 7 0 0$ </td></tr><tr><td>3</td><td>Grass-Pasture</td><td> $4 8 3 + 7 3 0 + 2 8 = 1 2 4 1$ </td></tr><tr><td>4</td><td>Hay-windrowed</td><td> $4 7 8$ </td></tr><tr><td>5</td><td>Soybeans-notill-Oats</td><td> $9 7 2 + 2 4 5 5 + 5 9 3 + 2 0 = 4 0 2 0$ </td></tr><tr><td>6</td><td>Woods-Bldg-Grass-Stone</td><td> $1 2 6 5 + 3 8 6 + 9 3 = 1 7 4 4$ </td></tr></table>

TABLE II: Ground Truth Details for the Reduced Indian Pines (IP) Dataset

## B. Linear Discriminant Analysis (LDA)

In contrast to PCA, (LDA) is a dimensionality reduction technique tailored for supervised classification. LDA departs from unsupervised approaches by incorporating class labels in its quest to unearth a lower-dimensional space that maximizes the separation between distinct classes.

LDA’s core objective revolves around discovering linear combinations of features, aptly referred to as discriminants, that foster clear demarcation between class clusters, all while minimizing within-class variance [3]. As a result, LDA finds its niche in applications such as pattern recognition, face verification, and medical diagnosis.

The field of remote sensing has extensively explored LDA, leveraging its capabilities for tasks like land cover classification, hyperspectral image analysis, and target detection. Its unique ability to enhance class separability renders LDA an invaluable tool within the realm of remote sensing.

a. To calculate the between-class scatter matrix $( S _ { B } ) \colon$

$$
\mathbf { S _ { B } } = \sum _ { i = 1 } ^ { c } N _ { i } ( \mathbf { m } _ { i } - \mathbf { m } ) ( \mathbf { m } _ { i } - \mathbf { m } ) ^ { T }\tag{5}
$$

• where c is the number of classes.

• $N _ { i }$ is the number of samples in class i.

• m<sub>i</sub> is the mean of samples in class i.

• m is the overall mean of all data.

b. To calculate the within-class scatter matrix (S<sub>W</sub>):

$$
\mathbf { S } _ { \mathbf { W } } = \sum _ { i = 1 } ^ { c } \sum _ { j = 1 } ^ { N _ { i } } ( \mathbf { x } _ { i j } - \mathbf { m } _ { i } ) ( \mathbf { x } _ { i j } - \mathbf { m } _ { i } ) ^ { T }\tag{6}
$$

• where $\mathbf { x } _ { i j }$ is the $j \mathrm { - t h }$ sample in class i.

c. To calculate the generalized inverse of the within-class scatter matrix times the between-class scatter matrix $( S _ { W } ^ { - 1 } { \bf S _ { B } } ) \colon$

$$
\mathbf { S _ { W } } ^ { - 1 } \mathbf { S _ { B } }\tag{7}
$$

d. To obtain the eigenvectors $( \mathbf { v } _ { i } )$ and eigenvalues $( \lambda _ { i } )$ of $\mathbf { S _ { W } } ^ { - 1 } \mathbf { S _ { B } }$ :

$$
\mathbf { S _ { W } } ^ { - 1 } \mathbf { S _ { B } } \mathbf { v } _ { i } = \lambda _ { i } \mathbf { v } _ { i }\tag{8}
$$

• where $\mathbf { v } _ { i }$ is an eigenvector.

• $\lambda _ { i }$ is the corresponding eigenvalue.

e. To project the data onto the linear discriminant components (y):

$$
\mathbf { y } = \mathbf { X } \mathbf { V }\tag{9}
$$

• where y is the matrix of projected data.

• X is the original data matrix.

• V is the matrix of eigenvectors chosen as linear discriminant components.

The Figure 3 presents the visualization of the first three principal components obtained from Linear Discriminant Analysis (LDA). These principal components are crucial for reducing the dimensionality of hyperspectral data and aiding in the differentiation between various classes

## IV. METHODOLOGY

## A. K-Nearest Neighbors (KNN)

The KNN algorithm is a non-parametric method commonly used for classification, relying on the proximity of training examples in the feature space. The KNN classification process involves partitioning data into a test set and a training set. For each row in the test set, the K nearest training set instances are determined using the Euclidean distance metric, and the classification is decided through a majority vote. In cases where there is a tie for the Kth nearest neighbor, all tied candidates are included in the voting process. A noteworthy characteristic of KNN is its reliance on the entire training dataset during the testing phase, where decisions are made based on the entirety of the training data [6].

The mathematical expression of the (KNN) algorithm can be formulated as follows:

a. Let X be the training dataset with features $\mathbf { x } _ { i }$ and class labels $y _ { i }$ for $i = 1 , 2 , \dots , N$ , where N is the number of training examples.

b. Let x be a test example that we want to classify.

c. Calculate the Euclidean distance between x and each training example $\mathbf { x } _ { i }$ :

$$
\mathrm { d i s t a n c e } ( \mathbf { x } , \mathbf { x } _ { i } ) = \sqrt { \sum _ { j = 1 } ^ { d } ( x _ { j } - \mathbf { x } _ { i j } ) ^ { 2 } }\tag{10}
$$

\- Where d is the number of features.

d. Select the K training examples with the shortest distances to x.

e. Perform a majority vote among the class labels of these K neighbors.

\- The most frequent class among the K neighbors is assigned to the test example x as its predicted class. In case of a tie, all classes are considered.

The choice of the parameter K and the distance metric used (such as Euclidean distance) are important considerations when applying KNN to classification tasks.

![](images/1aba78fe079f6f2dfe77a19c44ed265856d5d593639726ce06048fddb7d9603f.jpg)  
(a) Band 23

![](images/0e5237ddc32614a0079a8c1557b7643dfd001fc93f7785c7b5789c9ed3cf9157.jpg)  
(b) Band 124

![](images/56646315325eb736d3ac06f53317e518fc921d18da8cb6278dc515d8824f1383.jpg)  
(c) Bande 95

Fig. 1: The visualization of the three randomly selected bands over 220.  
![](images/a116ae404f350186055fd68cab15f21a4f93ed8cf0103dfe61fc60aac30e89bc.jpg)  
(a) Band 1

![](images/c695f00496e57d7ecfce109cbef85a79742fcb960b1965602557245aad257848.jpg)  
(b) Band 2

![](images/fa9c07cdc2a3827d94654062e3e84b16d57facf822428c16c692676485f6198d.jpg)  
(c) Band 3

Fig. 2: Visualization of the selected bands after PCA.  
![](images/583846c21544eeb4fdeda50f622c6125f8ded244df975b46635843ecfae26193.jpg)  
(a) Band 1

![](images/87d52bac25cf675d8a2f383834a290ac1338d01f41d68ba235a7905318ef4aa1.jpg)  
(b) Band 2

![](images/4924227b367cb5a3bca9fa9ac8e01cb276ef396a8e56bf489855f4c3861c1a8d.jpg)  
(c) Band 3  
Fig. 3: Visualization of the selected bands after LDA.

## B. Support Vector Machines (SVM)

SVM is a parametric classification approach used to tackle classification challenges in datasets where the relationships between variables are not explicitly known.

SVM is rooted in statistical learning theory and was initially designed to classify linearly separable data with two classes. However, it has since been extended to handle nonlinear and multi-class datasets effectively. The core principle of SVM involves identifying the hyperplane that optimally discriminates between the two classe[5].

In our specific case, we employed the Gaussian kernel function, also known as the radial basis function (RBF) kernel. This choice of kernel allows SVM to effectively capture complex, nonlinear relationships within hyperspectral data.

The mathematical expression for the RBF kernel function is given by:

$$
K ( \boldsymbol { x } , \boldsymbol { x } ^ { \prime } ) = \exp { \left( - \frac { \| \boldsymbol { x } - \boldsymbol { x } ^ { \prime } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) }\tag{10}
$$

$K ( x , x ^ { \prime } )$ is the RBF kernel function.

• $x$ and $x ^ { \prime }$ are the input data points.

• σ is a parameter that controls the width of the kernel and influences the flexibility of the decision boundary.

To accommodate nonlinear data relationships, SVM employs kernel functions that map the data into a higherdimensional space[9]. The optimization process in SVM aims to maximize the margins between support vectors and establish an optimal decision function based on the hyperplane in the transformed feature space[6].

![](images/13f78f54b3b42957926fafd104ccc8c80cef467ed19595e80f93959f845b3fa2.jpg)  
Fig. 4: Illustration of Support Vector Machine

This choice of the RBF kernel and the optimization process in SVM enable it to perform effective classification even in cases with complex, nonlinear data relationships.

## C. Random Forest (RF)

We delve into the methodology of (RF) for hyperspectral image classification. RF is a powerful ensemble learning technique built upon the principle of employing decision trees as elementary classifiers and aggregating their results to create a robust collective learning model [4]. RF classifiers are renowned for their resilience against overfitting, ease of parameterization, and computational efficiency (Kavzoglu, 2017).

The primary objective of the RF classifier is to construct a multitude of decision trees using a bootstrapped sampling approach. During this process, the training dataset used for creating tree models within the decision forest is selected randomly from the original training dataset. Roughly two-thirds of the randomly sampled dataset are utilized for constructing the decision tree structure, while the remaining portion is reserved for validating the generated decision tree models. To classify an uncertain sample, the class label is determined using the majority voting principle, where each tree model in the decision forest contributes its prediction.

The mathematical expression of the RF algorithm can be described as follows:

Let N be the number of decision trees in the forest, D be the training dataset, and x be an uncertain sample to be classified.

For each decision tree $t _ { i }$ in the forest $i = 1 , 2 , \ldots , N ;$

a. Randomly select a bootstrapped dataset $D _ { i }$ from D with replacement.

b. Train $t _ { i }$ on $D _ { i }$

To classify x:

c. Aggregate predictions from all decision trees:

$$
\hat { y } _ { i } = t _ { i } ( x ) \quad \mathrm { f o r } \ i = 1 , 2 , \ldots , N\tag{11}
$$

d. Determine the final class label for x through majority voting:

$$
\hat { y } = \operatorname { a r g m a x } _ { y } \sum _ { i = 1 } ^ { N } I ( \hat { y } _ { i } = y )\tag{12}
$$

Where:

• $N$ is the number of decision trees in the forest.

• $D$ is the training dataset.

• $x$ is the uncertain sample to be classified.

$t _ { i }$ represents an individual decision tree.

$D _ { i }$ is the bootstrapped dataset for tree $t _ { i } .$

• $\hat { y } _ { i }$ is the prediction of tree $t _ { i }$ for x.

• $\hat { y }$ is the final class label for x.

• y represents class labels.

• I(·) is the indicator function.

## D. Model Parameter Tuning (K-Fold Cross-Validation)

In our study, we employed cross-validation techniques to meticulously tune the parameters of our hyperspectral image analysis models. This approach helps mitigate the risk of overfitting, especially in datasets with size constraints.

For the RF model, we focused on optimizing key parameters such as the number of decision trees in the forest, the maximum tree depth, and the minimum samples required to split a node. These parameters play a pivotal role in determining the model’s complexity and its susceptibility to overfitting [4]. We systematically explored a range of values for each parameter to identify optimal values that would yield superior classification performance.

In the case of the KNN model, we fine-tuned the number of neighbors (K) to consider during classification. The choice of K directly influences the model’s flexibility and sensitivity to noise. Therefore, we conducted experiments by adjusting K to find the value that offers the best generalization capacity.

For the SVM model with the Gaussian kernel function, we scrutinized two crucial parameters: the regularization coefficient (C) and the kernel width. The C parameter controls the tolerance to classification errors, while the kernel width affects the model’s flexibility. Through careful parameter tuning, we aimed to discover the optimal combination that would yield superior classification performance.

To assess and compare the performance of each model at each stage of the parameter tuning process, we employed Kfold cross-validation with K set to 5 . Performance measures such as the mean R-squared scores were used as evaluation criteria[8].

![](images/1df5316becb6e91aef3aec7760d77d78252338e4e37fe43ebfa3d992129d0f3c.jpg)  
Fig. 5: Classification after PCA Dimensionality Reduction with Various Classifiers

![](images/742e4f54e5ce2955fcea0da6eba277af0d9fced9dc545742d5492f1c3acfe698.jpg)  
Fig. 6: Classification after LDA Dimensionality Reduction with Various Classifiers

TABLE III: Classification Overall Accuracy and Kappa Values with Dimensionality Reduction (OA: Overall Accuracy, Ka: Kappa)
<table><tr><td rowspan=1 colspan=1>Reduction Method</td><td rowspan=1 colspan=1>Classifier</td><td rowspan=1 colspan=1>OA (%)</td><td rowspan=1 colspan=1>Ka (%)</td></tr><tr><td rowspan=1 colspan=1>PCA</td><td rowspan=1 colspan=1>KNNRFSVM</td><td rowspan=1 colspan=1>0.9410.9550.938</td><td rowspan=1 colspan=1>0.9040.9260.896</td></tr><tr><td rowspan=1 colspan=1>LDA</td><td rowspan=1 colspan=1>KNNRFSVM</td><td rowspan=1 colspan=1>0.9380.9450.926</td><td rowspan=1 colspan=1>0.8980.9380.905</td></tr></table>

TABLE IV: Classification overall accuracies and F-score values of datasets with PCA and LDA
<table><tr><td rowspan=2 colspan=1>LULC Classes</td><td rowspan=1 colspan=2>SVM</td><td rowspan=1 colspan=2>RF</td><td rowspan=1 colspan=2>KNN</td></tr><tr><td rowspan=1 colspan=1>PCA</td><td rowspan=1 colspan=1>LDA</td><td rowspan=1 colspan=1>PCA</td><td rowspan=1 colspan=1>LDA</td><td rowspan=1 colspan=1>PCA</td><td rowspan=1 colspan=1>LDA</td></tr><tr><td rowspan=1 colspan=1>Class 1</td><td rowspan=1 colspan=1>0.949</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.963</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.951</td><td rowspan=1 colspan=1>0.950</td></tr><tr><td rowspan=1 colspan=1>Class 2</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.899</td><td rowspan=1 colspan=1>0.944</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.913</td></tr><tr><td rowspan=1 colspan=1>Class 3</td><td rowspan=1 colspan=1>0.937</td><td rowspan=1 colspan=1>0.918</td><td rowspan=1 colspan=1>0.959</td><td rowspan=1 colspan=1>0.951</td><td rowspan=1 colspan=1>0.949</td><td rowspan=1 colspan=1>0.934</td></tr><tr><td rowspan=1 colspan=1>Class 4</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.929</td><td rowspan=1 colspan=1>0.960</td><td rowspan=1 colspan=1>0.943</td><td rowspan=1 colspan=1>0.946</td><td rowspan=1 colspan=1>0.945</td></tr><tr><td rowspan=1 colspan=1>Class 5</td><td rowspan=1 colspan=1>0.875</td><td rowspan=1 colspan=1>0.850</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.863</td><td rowspan=1 colspan=1>0.858</td></tr><tr><td rowspan=1 colspan=1>Class 6</td><td rowspan=1 colspan=1>0.901</td><td rowspan=1 colspan=1>0.848</td><td rowspan=1 colspan=1>0.952</td><td rowspan=1 colspan=1>0.962</td><td rowspan=1 colspan=1>0.914</td><td rowspan=1 colspan=1>0.941</td></tr><tr><td rowspan=1 colspan=1>OA (%)</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.955</td><td rowspan=1 colspan=1>0.945</td><td rowspan=1 colspan=1>0.941</td><td rowspan=1 colspan=1>0.938</td></tr></table>

1) Hyperparameters: For our parameter tuning process, we considered the following hyperparameters for each classification model:

KNN:

• Number of neighbors (K)

RF:

• Number of decision trees in the forest (numTrees)

• Minimum samples required to split a node (MinLeafSize)

SVM with Gaussian Kernel:

• Regularization coefficient (C)

These hyperparameters were systematically adjusted and optimized during the K-fold cross-validation process to enhance the classification performance of our models[8].

## V. APPLICATION,RESULTS AND DISCUSSION

In this section, we present the results of our analysis. We began by selecting the top 10 features for both PCA and LDA to reduce dimensionality. After applying k-fold crossvalidation, where k = 5, we determined the optimal hyperparameter settings for the three classifiers that maximized the mean accuracy. Specifically, for the K-Nearest Neighbors (KNN) classifier, the optimal value of K was found to be 1. In the case of the Random Forest classifier, we identified the best hyperparameters as numTrees = 150 and MinLeafSize = 1. For the Support Vector Machine (SVM) classifier, the optimal hyperparameter was C = 10.

Subsequently, with these tuned hyperparameters, we trained the three classifiers using 75% of the available data, corresponding to 15,769 pixels. The classifiers’ performance was evaluated on the test data, constituting 25% of the dataset and comprising 5,256 pixels. In this section, we present the results of our analysis, focusing on the overall accuracy and kappa values achieved by the three classifiers: KNN, RF, and SVM. We assess the performance of these classifiers under two dimensionality reduction techniques: PCA and LDA).

To evaluate the classifiers’ performance, we employed standard confusion matrices, which allowed us to calculate classification accuracies across different classes. The results for overall accuracy and kappa values are summarized in Table III.Moreover, F-score is computed from user and producer accuracies. The predicted overall accuracies and F-score values for all datasets, methods and classes as described in Table IV.

Whether it is LDA or PCA, the performance metrics, especially overall accuracy (OA) and the kappa coefficient (Ka), are promising for all three algorithms. We can observe that RF outperforms the other two with an OA of 95.5% and a Ka of 92.6%. It is closely followed by KNN and SVM, which are nearly equal. KNN achieves an OA of 94.1% and a Ka of 90.4%, while SVM shows an OA of 93.8% and a Ka of 89.6%.

## VI. CONCLUSION

In the realm of hyperspectral image analysis, the application of dimensionality reduction techniques has emerged as a powerful ally. This approach not only streamlines the processing of voluminous data but also mitigates the risk of computational errors. In this study, we explored the potential of LDA and PCA in enhancing the performance of various classifiers on hyperspectral data.

Our results revealed that both LDA and PCA have made impressive strides in optimizing the accuracy and efficacy of classifiers. Remarkably, PCA, with its relatively simpler method of computing principal eigenvectors, delivered outstanding outcomes, albeit with slight differences in the evaluated metrics compared to LDA.

It’s worth noting that our chosen classifiers are inherently non-linear. However, when dealing with linear classifiers, LDA emerges as the more fitting choice due to its mathematical underpinnings. The subtleties of these outcomes underscore the importance of selecting the right dimensionality reduction technique, depending on the nature of the problem and the classifier being employed.

Random Forest consistently achieves high classification accuracy with both PCA and LDA dimensionality reductions, making it a robust choice for hyperspectral image classification. K-Nearest Neighbors also performs well, especially in combination with PCA. Support Vector Machines, while effective, exhibit slightly lower performance. These results underscore the importance of the judicious choice of the classifier and the potential of PCA and LDA to enhance classification accuracy in hyperspectral data

In closing, our study underscores the indispensable role of dimensionality reduction in enhancing the efficiency and accuracy of multispectral image analysis. Whether it’s the streamlined simplicity of PCA or the mathematical rigor of LDA, these techniques equip us with invaluable tools to navigate the complexities of high-dimensional data. As we venture further into the realm of remote sensing and image analysis, it is clear that dimensionality reduction will continue to be a cornerstone for achieving robust, precise, and insightful results.

## REFERENCES

[1] Sunitha Abburu and Suresh Babu Golla. “Satellite image classification methods and techniques: A review”. In: Internationaljournal ofcomputer applications 119.8 (2015).

[2] Marion F Baumgardner, Larry L Biehl, and David A Landgrebe. “220 band aviris hyperspectral image data set: June 12, 1992 indian pine test site 3”. In: Purdue University Research Repository 10.7 (2015), p. 991.

[3] Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning. Vol. 4. 4. Springer, 2006.

[4] Leo Breiman. “Random forests”. In: Machine learning 45 (2001), pp. 5–32.

[5] L Hao and PL Lewin. “Partial discharge source discrimination using a support vector machine”. In: IEEE Transactions on Dielectrics and electrical Insulation 17.1 (2010), pp. 189–197.

[6] Trevor Hastie et al. The elements of statistical learning: data mining, inference, and prediction. Vol. 2. Springer, 2009.

[7] Jane Holland. “Bunce V.,(ed.) Axis, for GCSE and Standard Grade Geography. Volume 1, Number 1 (1994), 32 pp.; Volume 1, Numbers 2, 3 (1995), 32 pp. each; each magazine is accompanied by an eight-page set of teacher’s support material. Cheltenham: Stanley Thornes. Price£ 6.00 per annual magazine subscription;£ 9.99 per annual teacher’s notes subscription. ISSN 1354-800X.” In: Geological Magazine 132.6 (1995), pp. 746–747.

[8] Kichul Jung et al. “Evaluation of nitrate load estimations using neural networks and canonical correlation analysis with k-fold cross-validation”. In: Sustainability 12.1 (2020), p. 400.

[9] Taskin Kavzoglu, Furkan Bilucan, and Alihan Teke. “Comparison of support vector machines, random forest and decision tree methods for classification of sentinel-2A image using different band combinations”. In: 41st Asian Conference on Remote Sensing (ACRS 2020). Vol. 41. 2020, pp. 1–8.

[10] Paul Mather and Brandt Tso. Classification methods for remotely sensed data. CRC press, 2016.

[11] Yanwei Pang, Yuan Yuan, and Xuelong Li. “Effective feature extraction in high-dimensional space”. In: IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 38.6 (2008), pp. 1652–1656.

[12] Gareth Rees. Physical principles of remote sensing. Cambridge university press, 2013.

[13] Deshmukh Sachin et al. “Dimensionality reduction and classification through PCA and LDA”. In: International journal of computer Applications 122.17 (2015).

[14] Lindsay I Smith. “A tutorial on principal components analysis”. In: (2002).