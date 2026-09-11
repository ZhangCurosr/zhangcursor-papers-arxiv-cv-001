# Meta-Learning for Classifier Selection in Image Datasets: A Feature-Driven Framework for Accuracy Prediction

Zahra Nabizadeh Shahre Babak<sup>1</sup>, Farzaneh Koohestani<sup>2</sup>, Nader Karimi<sup>2</sup>, Shahram Shirani<sup>1</sup>, and Shadrokh Samavi<sup>3</sup>

<sup>1</sup>Department of Electrical and Computer Engineering, McMaster University, Hamilton L8S 4L8, Canada

<sup>2</sup>Department of Electrical and Computer Engineering, Isfahan University of Technology, Isfahan 8415683111, Iran

<sup>3</sup>Department of Computer Science, Seattle University, Seattle, WA 98122-1090, USA

## Abstract

No Free Lunch theorem implies that any performance gains achieved by a classifier on a particular image distribution are necessarily ofset by a loss of performance over the set of all possible problems; thus, no single model is universally optimal. Selecting the most suitable classifier for image datasets is a critical yet challenging task due to the intrinsic complexity and diversity of images. This paper proposes a meta-learning framework that leverages a comprehensive set of meta-features capturing dataset complexity to predict classifier performance without exhaustive training. By extracting and selecting features using methods such as autoencoders, pre-trained networks, and dimensionality reduction techniques, we train regression models to eficiently estimate classifier accuracies. Additionally, clustering techniques are employed to group classifiers with similar performance patterns, simplifying the recommendation process. The datasets used span a wide range of concepts including nature, animals, numbers, motorcycles, medical images, and human bodies to ensure broad generalization. Evaluated on 56 diverse image datasets, our approach achieves an average ranking prediction accuracy exceeding 86%, demonstrating its efectiveness in guiding model selection. This scalable and interpretable framework provides a practical solution to improve classification performance while reducing computational costs.

Keywords: Meta-learning, Classifier selection, Image dataset descriptor, Complexity, Meta-feature

## 1 Introduction

In recent years, machine learning algorithms have gained significant popularity and are now widely used across various applications. These algorithms are designed to learn from the distribution of input data to perform diverse tasks efectively. One key factor influencing the learning process is meta-learning (MtL), which involves learning how to learn. As a result, substantial research has been conducted in this area, leading to the development of various techniques aimed at improving the performance of Machine Learning (ML) algorithms.

In the context of ML, particularly with image datasets, a comprehensive understanding of the data is essential for successful classification. While numerous studies have focused on measuring dataset complexity, few specifically address the complexity of image datasets in classification tasks. Understanding dataset complexity is crucial for selecting the most suitable models, as it guides decisions related to preprocessing, feature extraction, and mode selection. Many decision-making challenges in machine learning can be formulated as ML problems, especially if a suitable dataset can be generated for the decision-making process itself. The following section discusses articles that explore the use of dataset complexity for decision-making purposes.

## 1.1 Complexity Measure

To measure the complexity of datasets, various perspectives can be considered. Each perspective highlights a specific property of the datasets that can be treated as meta-features. Research on meta-features often involves proposing new feature categories or introducing novel meta-features to enhance the learning process. For instance, in [1], the authors classified meta-features related to dataset characterization and classification complexity into five categories: feature-based, linearity, neighborhood, network, dimensionality, and class imbalance. In [2], Pimentela and Carvalho introduced a new set of meta-features that outperformed previous techniques in algorithm recommendation tasks. They extracted three sets of features—statistical, distance-based, and evaluation measures—from datasets and used them to recommend the best clustering algorithm. The paper [3] systematizes data characterization measures for classification datasets and introduces the Meta-Feature Extractor (MFE) to enhance reproducibility in meta-learning research.

There are various meta-features that describe datasets and can provide insights into their complexity. Some of these meta-features are applicable to image or 2D datasets, while others are specific to 1D datasets. These meta-features ofer diferent perspectives on the dataset’s characteristics. Below, we will explain these meta-features based on their respective approaches.

Some meta-features describe datasets using network-based approaches. By mapping each dataset into a graph or tree structure, these features capture dataset characteristics. In [4], the authors introduced a network metric called closeness centrality, which measures the relative distance of a node to all others based on shortest path. Another widely used feature, degree centrality [5], quantifies the number of connections a node has, indicating its importance. Betweenness centrality [6] measures a node’s significance by calculating the number of shortest paths passing through it. A related measure, edge betweenness centrality [7], evaluates the importance of edges based on shortest paths. PageRank [8] ranks nodes solely by the structure of their connections, while eigenvector centrality [9] considers both the number of connections and the significance of those connections. A spectral clustering-based metric partitions a graph into subgraphs by minimizing inter-cluster edge weights. This feature maps samples into a transformed space where similar images are clustered, followed by the computation of the Laplacian matrix and the Cumulative Spectral Gradient (CSG) using its eigenvalues and eigenvectors [10].

Several meta-features evaluate clustering output. The Silhouette Coeficient [11] assesses how well each data point fits within its assigned cluster relative to other clusters. The Calinski-Harabasz (CH) Index [12] measures clustering quality based on the ratio of between-cluster dispersion to within-cluster dispersion, where higher values indicate better clustering. The Davies-Bouldin (DB) Index [13] compares the average within-cluster distances to between-cluster distances, with lower values signifying better separation. Similarly, the Dunn Index (DI) [14] calculates the ratio of minimum inter-cluster distance to maximum intra-cluster distance. Another clustering-related feature, normalized relative entropy, also known as Kullback-Leibler divergence, quantifies dissimilarity between cluster probability distributions [2]. The Int Index [15] evaluates cluster compactness and separation based on intra- and inter-cluster distances.

Entropy-based features describe dataset complexity. Shannon entropy, GLCM entropy, and delentropy were used in [16] to rank image datasets based on complexity and correlate their rank with deep learning performance. Other information-theoretic features include Mutual Information (MI), which quantifies dependency between sample attributes and target labels, and the noisy feature ratio, which measures the proportion of irrelevant attributes [3].

Certain features are specific to image datasets. Histogram of Oriented Gradients (HOG) [17] captures edge directions and intensities, while Local Binary Patterns (LBP) [18] describe texture characteristics by encoding local intensity variations. Cho and Lee introduced two metrics in [19] to assess image dataset quality: Msep measures class separability, while Mvar quantifies in-class variability.

Decision tree-based meta-features assess dataset complexity by training a decision tree model on each dataset and extracting statistics, including the number of leaves, branches, nodes, and non-leaf nodes [3]. Additional features include the proportion of leaves per class, node-to-feature ratio, node-to-instance ratio, tree depth, probability of reaching a leaf randomly, and node importance scores.

In [20], a new metric for measuring data classification complexity, called C<sup>2</sup>M kNN, is proposed. This method allows for better prediction of classification performance compared to other features that describe synthetic and 1D real datasets. The paper [21] introduces the Distance-based Separability Index as an efective and model-independent measure of dataset separability, with potential applications in deep learning, data science, and AI interpretability. This metric is also evaluated on synthetic datasets and the CIFAR-10/100 dataset. In paper [22], two new metrics, Neighborhood Density (ND) and Decision Tree Progression (DTP), are proposed for assessing redundancy, complexity, and density in big data classification. These metrics help identify unnecessary data in large datasets, allowing for eficient preprocessing. The metrics, implemented in a Spark-based package, aid in reducing dataset size without significantly afecting accuracy, promoting smart data usage.

## 1.2 Decision-Making

Meta-features are widely used to guide dataset-level decisions such as classifier selection. For example, [23] leverages dataset complexity metrics for 1D classifier selection, while [24] demonstrates that dimensionality reduction on meta-features accelerates training without sacrificing predictive performance. For spam detection, [25] applies an Analytic Hierarchy Process to rank and recommend top-K classifiers across multiple metrics.

Meta-learning also optimizes visual tasks. Frameworks like MetaDelta [26] automate few-shot classifier selection, [27] introduces meta-learning initialization for weakly supervised segmentation, and SSM-SAM [28] integrates adaptive attention for medical imaging. As surveyed in [29], classifier eficacy depends directly on dataset characteristics. Motivated by this, our framework uses meta-learning to map dataset descriptors to empirical performance, enabling eficient, data-driven classifier selection.

However, automating decisions for image datasets presents distinct challenges. High-dimensional pixel data increases computational costs and overfitting risks. Extracting features demands either labor-intensive engineering or resource-intensive deep architectures that require massive labeled datasets. Real-world variability—such as noise, lighting shifts, and class imbalance—compounds these issues. Finally, the opaque nature of modern deep models turns model selection into a dificult trade-of between accuracy, latency, and interpretability, particularly in high-stakes domains like medical imaging.

Due to these challenges, limited research has focused on quantifying dataset complexity and classifier selection for image datasets. While we cannot address all challenges comprehensively, our work aims to mitigate several key issues through a meta-learning framework based on dataset complexity measures. Our approach: (1) generates a meta-dataset capturing relevant dataset characteristics, and (2) trains a meta-mode to predict classifier performance, thereby reducing—though not eliminating—the challenges in model selection. This partial solution focuses particularly on improving generalization through better understanding of dataset properties. For implementation, we categorize meta-features by their methodological perspectives and apply feature selection to identify the most impactful features for classifier prediction. The key contributions of this paper are as follows:

• Meta-Learning Framework for Image Classification: We introduce a meta-learning approach that predicts the best ML classifier for image datasets based on their inherent complexity measures.

• Generation of Meta-Dataset: We generate a meta-dataset that captures the characteristics and complexities of diferent image datasets. This meta-dataset is used to train a meta-model to predict classifier accuracy for new datasets.

• Enhanced Classifier Performance and Generalization: By leveraging the complexity measures and meta-learning approach, we improve the performance and generalization of machine learning systems, especially in image classification tasks.

• Categorization and Selection of Meta-Features: We categorize the various meta-features based on their methods and perspectives, ensuring a structured approach for the subsequent application of feature selection techniques.

• Application of Feature Selection Methods: We apply diferent feature selection techniques on the categorized meta-features to identify the most relevant ones for predicting classifier performance.

The remaining structure of the paper is as follows: In Section 2, the formulation of our work is presented. In Section 3, the details of the proposed framework are described. Section 4 provides an analysis of the results, and the final section concludes the paper.

## 2 Formulation of Decision-Making as a Machine Learning Problem

The origins of the algorithm selection problem trace back to a fundamental question in computational science: Why does a specific algorithm outperform others on certain problem instances? This inquiry, which lies at the core of optimization and machine learning research, was first formally modeled by John Rice in 1976. The significance of this framework is that it transitioned algorithm selection from a heuristic-based practice into a quantifiable scientific problem characterized by four fundamental components[30].

Building upon this foundational theory, the process of selecting an appropriate classifier for a given image dataset can be formulated as a meta-learning problem. In this context, the goal is to predict the optimal classifier by mapping the intrinsic characteristics of a dataset to the performance of various algorithms. This formulation typically involves three functional stages: feature extraction (characterizing the dataset), feature selection (identifying the most predictive metadata), and the training of a meta-model to automate the decision-making process.

## 2.1 Preprocessing

Given that images across diferent datasets often possess varying dimensions and aspect ratios, the initial step involves a preprocessing phase to ensure data uniformity. In this stage, all images are first resized to a fixed resolution to produce a standardized set of images, defined as:

$$
{ \cal D } ^ { \mathrm { r e s } } = \{ \tilde { I } _ { 1 } , \tilde { I } _ { 2 } , \ldots , \tilde { I } _ { N } \} .
$$

Subsequently, depending on the specific type of meta-features required, certain operations are performed directly on the resized images, while others necessitate the transformation of images into feature vectors. To this end, a vectorization and dimensionality reduction mapping is defined:

$$
\phi : \tilde { I } _ { i } \mapsto v _ { i } \in \mathbb { R } ^ { d }
$$

Consequently, the resulting set of feature vectors is obtained as:

$$
D ^ { \mathrm { v e c } } = \{ v _ { 1 } , v _ { 2 } , . . . , v _ { N } \} .
$$

## 2.2 Feature Extraction and Meta-Dataset Creation

In this stage, the set of meta-features is extracted from the preprocessed data $( D ^ { \mathrm { r e s } }$ or $D ^ { \mathrm { v e c } } )$ . These meta-features encompass metrics such as class separability, feature redundancy,

intrinsic dimensionality, and various statistical indices that describe the underlying structure and inherent complexity of the data:

$$
X _ { \mathrm { m e t a } } ( D ^ { \mathrm { r e s ~ o r ~ v e c } } ) = \{ f _ { 1 } , f _ { 2 } , \dots , f _ { n } \}
$$

where each $f _ { i }$ represents a meta-feature computed from either $D ^ { \mathrm { r e s } }$ or $D ^ { \mathrm { v e c } }$

## 2.3 Feature Selection

Once the meta-features are extracted, we apply feature selection methods to identify the most relevant features for predicting classifier performance. Feature selection helps reduce the dimensionality of the meta-dataset, removing irrelevant or redundant features, and retaining only those that significantly influence the model selection process. Common feature selection techniques include methods based on correlation, importance scores from regression models, and recursive elimination techniques. The goal is to retain a subset of meta-features $X _ { \mathrm { m e t a } } ^ { * } \subset$ $X _ { \mathrm { m e t a } }$ that are most predictive of classifier performance.

Let $X _ { \mathrm { m e t a } } ^ { \ast } ( D ^ { \mathrm { r e s ~ o r ~ v e c } } )$ represent the selected meta-features after applying the feature selection process:

$$
X _ { \mathrm { m e t a } } ^ { * } ( D ^ { \mathrm { r e s ~ o r ~ v e c } } ) = \{ f _ { 1 } ^ { * } , f _ { 2 } ^ { * } , \ldots , f _ { m } ^ { * } \}
$$

where $f _ { 1 } ^ { * } , f _ { 2 } ^ { * } , \ldots , f _ { m } ^ { * }$ are the selected meta-features after feature selection.

## 2.4 Training a Meta-Model

Once the meta-dataset is created and the feature selection step is completed, we train a meta-model $M _ { \mathrm { m e t a } }$ that learns the relationship between the selected meta-features $X _ { \mathrm { m e t a } } ^ { \ast } ( D ^ { \mathrm { r e s ~ o r ~ v e c } } )$ and the performance of various classifiers on the dataset $D .$ . The goal is to predict the accuracy $A _ { j }$ of classifier $j$ on the dataset $D ,$ given the selected meta-features $X _ { \mathrm { m e t a } } ^ { \ast } ( D ^ { \mathrm { r e s ~ o r ~ v e c } } )$ This transforms the problem into a regression problem where the meta-model is trained to predict classifier performance based on the dataset’s meta-features.

The meta-model $M _ { \mathrm { m e t a } }$ can be written as:

$$
A _ { j } = M _ { \mathrm { { m e t a } } } ( X _ { \mathrm { { m e t a } } } ^ { * } ( X _ { \mathrm { { m e t a } } } ( D ^ { \mathrm { { r e s ~ o r ~ v e c } } } ) ) )
$$

where $A _ { j }$ is the predicted accuracy of classifier $j$ for dataset D. $M _ { \mathrm { m e t a } }$ is the trained machine learning model (e.g., regression, decision tree, neural network) that predicts classifier performance.

## 2.5 Select Proper Classifier

Once the meta-model is trained, it can be used to predict the proper classifier for any new, unseen dataset $D ^ { \prime }$ . The decision-making process is now reduced to a machine learning task where input is the meta-features $X _ { \mathrm { m e t a } } ^ { * } ( D ^ { \prime } )$ of the new dataset $D ^ { \prime }$ and output is the predicted accuracy for each classifier $A _ { j }$ , and the classifier with the highest predicted accuracy is chosen as the best classifier for dataset $D ^ { \prime }$ . The decision rule can be expressed as:

$$
j ^ { * } = \arg \operatorname* { m a x } _ { j } A _ { j }
$$

where $j ^ { * }$ is the index of the classifier with the highest predicted accuracy $A _ { j }$

Thus, the classifier selection process is formalized as an optimization problem in which the goal is to maximize the predicted accuracy using the meta-model $M _ { \mathrm { m e t a } }$ . This turns the decision-making task into an end-to-end machine learning problem where both the model selection and accuracy prediction are based on dataset characteristics and the selected meta-features.

## 3 Proposed Method

Selecting an appropriate classifier for an image dataset is a crucial step in machine learning, as diferent classifiers exhibit varying levels of performance depending on the dataset characteristics. Image datasets often difer in terms of feature complexity, class distribution, resolution, and noise levels, making classifier selection a non-trivial task. Traditional approaches rely on empirical evaluation, where multiple classifiers are trained and tested on the dataset to determine the most suitable model. However, this process is computationally expensive, especially for large-scale image datasets.

To address this challenge, we propose a framework that leverages dataset-specific features from diferent view points to predict proper classifier based on its probability or accuracy without the need for extensive training. By extracting statistical, structural, and complexity-based features from image datasets, we build a predictive model that estimates the performance of various classifiers, facilitating eficient model selection. Our approach aims to solve the problem while primarily providing insights into the relationship between dataset characteristics and classifier performance, thereby contributing to a more interpretable and automated machine learning workflow.

The workflow of our approach is illustrated in Figure 1, encompassing several key contributions. Firstly, a diverse set of meta-features is extracted from each dataset, considering both image-based and vector-based characteristics. To transform images into one-dimensional representations, feature extraction techniques such as VGG19 [31] and autoencoder are employed. Additionally, dimensionality reduction methods like Principal Component Analysis (PCA) [32] and t-distributed Stochastic Neighbor Embedding (t-SNE) [33] are utilized to minimize computational costs while preserving essential information. Next, classifiers are trained using the extracted one-dimensional data, and their accuracy is recorded. Based on the extracted meta-features and corresponding classifier performance, we generate a new set of descriptors for each dataset. These descriptors serve as a predictive dataset, enabling the selection of the most suitable classifier without the need for an exhaustive search. Our approach consists of two main parts: generating a dataset to train the trainable component in the workflow, and selecting the appropriate classifier for each dataset. These two parts involve several components, many of which are shared between them. However, the accuracy calculation and feature selection components are used only in the first part. In the following subsections, we provide a detailed explanation of all the components involved in these two parts.

## 3.1 Resizing

Since we work with datasets that have diferent applications and concepts, the images come in various dimensions. This variability could lead to an unfair comparison, so to make the analysis more consistent, we resize all the images to a similar dimension.

## 3.2 Convert Image to Vector

Two widely used approaches exist for converting an image into a vector representation: (i) directly converting the image into one-dimensional data without changing the size of data and (ii) extracting learned features using pre-trained models.

The first approach involves flattening the image into a one-dimensional vector, which is simple and requires no special techniques. However, while easy to implement, flattening does not capture higher-level features or spatial relationships within the image. The second approach utilizes pre-trained neural network models such as autoencoders and deep classifiers like VGG19 to extract meaningful features. These models can be used in their pre-trained state or fine-tuned on our datasets. To evaluate their efectiveness, we explore both methods in our work.

Flattening is advantageous because it is simple, computationally eficient, and universally applicable to any dataset. Unlike feature extraction using neural networks, flattening does not require a training phase; it simply reshapes the image into a one-dimensional vector. This makes it a quick and easy method for converting images into numerical representations. However, a key drawback is that flattening discards spatial relationships and hierarchical structures within the image.

On the other hand, feature extraction using neural networks captures high-level patterns, textures, and semantic information that are crucial for classification. However, training deep networks from scratch is computationally expensive and requires large datasets to learn meaningful representations. This overhead can be mitigated by using pre-trained models, such as VGG19 or ResNet, which have already learned useful features from massive datasets like ImageNet. These models can directly extract robust and transferable features, significantly reducing the need for training.

Despite their advantages, pre-trained models have limitations—they may not generalize well to domain-specific datasets, such as medical images or satellite imagery, where the distribution of features difers significantly from natural images in ImageNet. For example, a model trained on everyday objects may struggle to recognize microscopic cell structures in histopathology images. To overcome this, transfer learning is applied, where a pre-trained model is fine-tuned on a smaller dataset specific to the target domain. This approach allows the model to adapt its learned representations to new tasks while leveraging prior knowledge, improving performance in specialized applications [34].

![](images/a136b5025707c8c028b59c719a15f7b01e9243faa5095506eba6572bcf638c48.jpg)  
Figure 1: The workflow of our approach.

## 3.3 Dimensionality Reduction

The use of dimensionality reduction comes after we convert the images into vectors. No matter the method used for this conversion, the resulting feature representations tend to be high-dimensional, which can slow down processing and increase memory usage. To handle this, we use PCA and t-SNE to reduce the dimensions, ensuring that the features are both more eficient computationally and still capture the key characteristics of the datasets.

PCA is a linear dimensionality reduction technique that transforms the original high-dimensional data into a lower-dimensional space by projecting it onto a set of orthogonal axes called principal components. These components capture the directions of maximum variance in the data, allowing for a more compact representation while preserving important statistical information. PCA is particularly efective when dealing with highly correlated features, as it removes redundancy and highlights dominant patterns. However, since PCA assumes linear relationships, it may not be optimal for datasets with complex, nonlinear structures [32].

Unlike PCA, t-SNE is a nonlinear dimensionality reduction technique that focuses on preserving the local structure of the data. It maps high-dimensional data into a lower-dimensional space while maintaining relationships between similar data points. However, it is computationally intensive and less suited for large-scale feature extraction compared to PCA. Additionally, since t-SNE is primarily used for visualization, it may not be ideal for downstream tasks that require consistent feature representations [33].

## 3.4 Accuracy Calculation

To create the dataset for our framework, we need an (input, output) pair for each sample. In earlier steps, the images in each dataset were converted into vector representations. Now, we define the target variable for each dataset sample. The goal of this work is to predict the classification accuracy of the most suitable classifier for a given dataset, helping us select the appropriate classifier. To achieve this, we evaluate multiple classifiers and calculate their respective accuracy scores for each dataset.

For each dataset, we evaluate the performance of 15 diferent classifiers, representing a broad range of machine learning models. These classifiers are trained using the vectorized features of the datasets. After training, the accuracy scores from each classifier are used as target values for our predictive model. In other words, each dataset is paired with a set of accuracy scores, one for each classifier. These scores are then used to train a regression model that predicts classifier performance on unseen datasets and ranks the classifiers accordingly. This ranking can also serve as input for training a separate classifier that selects the most appropriate model, providing an alternative to the regression approach.

## 3.5 Meta-Features

Describing image datasets accurately is critical due to their inherent complexity and diversity. Extracting a single descriptor that fully captures all dataset characteristics is challenging. To address this, we extract a diverse set of meta-features that reflect various aspects of dataset complexity, including image structure, class separability, distributional properties, and graph-based characteristics. These features provide valuable insights that facilitate more accurate classifier selection. In Figure 2, the whole meta-features and their categories are shown. Since there are lots of meta-features, they are organized into six main categories:

• Data Complexity Features: Capture intrinsic dataset properties such as intrinsic dimensionality, entropy measures, and class imbalance ratios.

• Class Separability and Overlap Features: Measure the degree of separability and overlap between classes to assess classification dificulty.

• Geometric and Topological Features: Characterize the geometric arrangement and topological structure of data points, including graph metrics like density and centrality.

• Model-Based Complexity Features: Derived from classifier models (e.g., decision trees) to quantify dataset complexity from a modeling perspective.

• Image and Low-Level Features: Include texture and spectral analysis such as Local Binary Patterns (LBP) and Histogram of Oriented Gradients (HOG) that capture image-specific characteristics.

• Dataset-Level Meta-Features: Describe overall dataset attributes, including sample size, number of classes, and attribute relationships.

This structured set of meta-features ensures a comprehensive representation of dataset properties, supporting more efective model selection.

By leveraging this extensive set of meta-features, we provide a comprehensive characterization of datasets, ensuring more accurate classifier selection and enhancing the generalization of our regression model.

## 3.6 Feature Selection

Classifier selection can be approached in two ways: (1) predicting the accuracy of each classifier using a regression model and choosing the best-performing one, or (2) directly predicting the optimal classifier label. Since both methods rely on training data, the quality and relevance of the data are critical. To improve model performance and reduce computational overhead, it is essential to use features that efectively reflect the complexity of the dataset. Given the significance of feature quality, various selection methods have been proposed. In this study, we evaluate three such approaches: (i) correlation analysis between features and classifier accuracy, (ii) game theory-based feature selection, and (iii) Recursive Feature Elimination with Cross-Validation (RFECV).

![](images/eea4009df3a3ad1cb5a3d13ad93162f5c99c10c4435d8497be2551418cd90385.jpg)  
Figure 2: The categories of used meta-features.

In the first approach, the correlation coeficient between each feature and classifier accuracy is calculated. We employ three correlation methods: Pearson, Spearman, and Kendall.

• Pearson correlation examines the linear relationship between two continuous variables.

• Spearman correlation measures the strength and direction of a relationship using ranked values instead of raw data, making it a non-parametric measure that does not assume a specific data distribution.

• Kendall correlation evaluates relationships by counting the number of concordant and discordant observation pairs. Two observations are concordant if both variables increase or decrease together and discordant if one increases while the other decreases.

Since each method captures diferent aspects of correlation, we compute all three and select the highest correlation value for each feature. The correlation coeficient lies within the range [-1, 1]. For feature selection, the minimum and maximum correlation values for each feature are first calculated. Features with an absolute correlation value exceeding 0.8 are then selected from this set, resulting in 38 among 58 features. A high correlation suggests these features have greater predictive power, as they more efectively capture variations in classifier accuracy.

The other two approaches, game theory-based feature selection and RFECV, are based on the trained model. In these two approaches, the value of each feature is calculated based on its efect on the training process, but from two diferent perspectives. For the game theory-based approach, the SHapley Additive exPlanations (SHAP) is used. SHAP explains the output of any machine learning model based on game-theory concepts, such as the Shapley value [35]. The RFECV approach, on the other hand, is a recursive feature elimination algorithm that iteratively removes the least significant features using cross-validation.

Both approaches require training a model first, after which feature importance scores are extracted. Since the goal is classifier accuracy prediction, we use regression models. The regression models are trained on the new dataset, where the extracted dataset features serve as inputs and each classifier’s accuracy is the target variable. After training, the importance of each feature is computed for every classifier separately. Since regression models can be linear, nonlinear, parametric, or non-parametric, we include a diverse set of models from each category, listed in Table 1. The three best models from this set is then selected to calculate the importance score.

After calculating the score for each feature in each classifier, the top features must be selected. In the first phase, 38 top features (equal to the number of features obtained from the correlation methods) are selected per classifier. Since there are 15 classifiers and three top regressors, the total number of selected features is 1620. These features are sorted in descending order based on their importance, and a weight is assigned to each feature according to its ranking, as follows:

• The top-ranked feature receives a weight of 1.

Table 1: List of regression models used for predicting dataset-specific classifier accuracy. Regression Models
<table><tr><td rowspan=1 colspan=3>CatBoost Regressor (CatBoost) [36]</td><td rowspan=1 colspan=1>Extreme Gradient Boosting (xgboost) [37]</td></tr><tr><td rowspan=1 colspan=2>Gradient Boosting Regressor (gbr) [38]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Random Forest Regressor (rf) [39]</td></tr><tr><td rowspan=1 colspan=3>Light Gradient Boosting Machine (LightGBM) [40]</td><td rowspan=1 colspan=1>Ridge Regression (ridge) [41]</td></tr><tr><td rowspan=1 colspan=3>Linear Regression (lr) [42]</td><td rowspan=1 colspan=1>Elastic Net (en) [43]</td></tr><tr><td rowspan=1 colspan=3>Lasso Regression (lasso) [44]</td><td rowspan=1 colspan=1>AdaBoost Regressor (ada) [45]</td></tr><tr><td rowspan=1 colspan=3>Huber Regressor (huber) [46]</td><td rowspan=1 colspan=1>Bayesian Ridge (BR) [47]</td></tr><tr><td rowspan=1 colspan=1>Decision Tree Regressor (DT) [48]</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>Orthogonal Matching Pursuit (omp) [49]</td></tr><tr><td rowspan=1 colspan=3>Passive Aggressive Regressor (par) [50]</td><td rowspan=1 colspan=1>K Neighbors Regressor (knn) [51]</td></tr><tr><td rowspan=1 colspan=3>Lasso Least Angle Regression (llar) [52]</td><td rowspan=1 colspan=1>Dummy Regressor (dummy) [53]</td></tr><tr><td rowspan=1 colspan=4>Extra Trees Regressor (et) [54]</td></tr></table>

• The least important feature receives a weight of $\frac { 1 } { 3 8 }$ .

• The remaining features are assigned weights in steps of $\frac { 1 } { 3 8 }$ .

Since a feature may appear across multiple classifiers, we compute the sum of its weights across all classifiers and assign it a final cumulative weight. Finally, the top 38 features with the highest total weights are selected.

## 3.7 Classifier Based Accuracy Prediction

To predict the accuracy of classifier models, two strategies can be employed: using a single model to estimate the accuracy of any given classifier based on its input features, or training a dedicated regression model for each classifier. Given that each classifier employs a distinct learning mechanism, the latter approach is adopted in this study. Various regression models, listed in Table 1, are trained using the selected features as input and the corresponding classifier accuracy as the target variable. Separate regressions are trained for each classifier, and the best-performing model is selected for final use. The performance of the regression models is evaluated using the MAE metric.

## 3.8 Select Best Model

Using the outputs from the previous step, the best-performing model can be identified. The classifier with the highest predicted accuracy is selected as the optimal choice. Additionally, a ranking of all classifiers based on their predicted accuracies can be generated as part of the output.

## 3.9 Select Best Group Model

Although each classifier has its own learning mechanism, they may produce similar results when applied to a given dataset. When classifiers yield comparable performance, they can be assigned the same rank for that dataset. To identify groups of classifiers exhibiting similar behavior, clustering is employed. The optimal number of clusters is determined using elbow method. Once the clusters are formed, classifiers are grouped accordingly, and instead of ranking individual classifiers as in the previous step, the ranks of the identified groups are used.

The output of the proposed framework includes the predicted accuracies of the classifiers, their individual rankings, and the rankings of classifier groups.

## 4 Experimental Results

The proposed framework consists of several components, each with diferent selection methods for its respective part. In this section, we provide detailed explanations of the implementation, the results for each part, and the analysis of those results.

## 4.1 Dataset

Our experimental setup utilizes 56 image datasets spanning multiple domains, such as medicine, nature, and industry. We randomly allocated 28% (n=16) of these datasets for testing, reserving the remaining 40 for training. This selection covers a broad spectrum of semantic concepts and structural characteristics (e.g., varying sample sizes and class counts) to facilitate a robust exploration of the problem space. For computational uniformity, all images were resized to a fixed resolution of 96 × 96 pixels.

## 4.2 Combination of Flatten and Dimensionality Reduction

To convert images into vectors and flatten them, we use a simple flattening method along with two neural network models. Since the image sizes in some datasets are large and vary, the simple flattening method is not suficient for our approach due to the time-consuming meta-feature extraction process. Therefore, two neural network models are employed in this work. Any classification or neural network model capable of extracting features from images could be used, but the models chosen here are VGG19 and AutoEncoder. For each model, two scenarios—pre-trained and transfer learning—can be applied. To evaluate both options, the pre-trained model is used for VGG19, while for AutoEncoder, the pre-trained model is applied first, followed by training for 100 epochs on each dataset. In this process, each image in the dataset is converted into a feature vector that captures the image’s properties.

The feature vectors extracted by these two models have dimensions of 100 and 288, which are still large for efective meta-feature extraction. Therefore, two dimensionality reduction techniques—PCA and t-SNE—are applied. For PCA, the dimensionality is reduced such that the reconstructed data retains 90% of the original variance. For t-SNE, the features are reduced to a fixed dimension of 10. As a result, we obtain four sets of vector representations for each image.

To compare these four sets, the correlation between the meta-features extracted from each set and the accuracy of the classifiers is calculated. Three methods—Pearson, Spearman, and Kendall—are used to calculate the correlation, and the number of features which have correlation greater than 0.8 is computed for each set. The results show that features extracted using an AutoEncoder with transfer learning, combined with PCA for dimensionality reduction, exhibit a stronger correlation with classifier performance. These results are presented in Table 2.

Table 2: Number of features with absolute value correlation greater than 0.8 for each feature set.
<table><tr><td rowspan=1 colspan=1>Features</td><td rowspan=1 colspan=1>AE_PCA</td><td rowspan=1 colspan=1>VGG19_PCA</td><td rowspan=1 colspan=1>AE_t-SNE</td><td rowspan=1 colspan=1>VGG19_t-SNE</td></tr><tr><td rowspan=1 colspan=1>Number</td><td rowspan=1 colspan=1>184</td><td rowspan=1 colspan=1>154</td><td rowspan=1 colspan=1>115</td><td rowspan=1 colspan=1>127</td></tr></table>

## 4.3 Compare Feature Selection

Three diferent feature selection methods are employed in this study. For the training-based methods, a set of regression models is first trained for each classifier, after which the top three performing regressors are selected per classifier. The variation in selected regression models across classifiers supports the decision to use separate training processes for each. The three selected regression models for each classifier are summarized in Table 3. To

Table 3: Selected Regression Models for Each Classifier
<table><tr><td>Classifier</td><td>Selected Regression Models</td></tr><tr><td>svm</td><td>et, gbr, rf</td></tr><tr><td>knn</td><td>et, gbr, rf</td></tr><tr><td>mpl3</td><td>br, gbr, rf</td></tr><tr><td>mpl5</td><td>et, rf, gbr</td></tr><tr><td>rf</td><td>ada, lightgbm, gbr</td></tr><tr><td>dt</td><td>et, rf, gbr</td></tr><tr><td>nb</td><td>et, rf, gbr</td></tr><tr><td>et</td><td>dt, gbr, lightgbm</td></tr><tr><td>gbc</td><td>et,gbr,rf</td></tr><tr><td>qda</td><td>et, gbr, lightgbm</td></tr><tr><td>lda</td><td></td></tr><tr><td>lr</td><td>et, gbr, lightgbm</td></tr><tr><td>rc</td><td>et, lightgbm,gbr et, gbr, lightgbm</td></tr><tr><td>abc</td><td>et, gbr, lightgbm</td></tr><tr><td>dummy</td><td>et,dt,gbr</td></tr></table>

evaluate the impact of these methods, the results are compared in this section. Since the correlation method directly uses the classifier’s accuracy, the features selected by the last two methods (RFECV and SHAP) are compared with those selected by the correlation method. Table 4 presents a comparison of the selected features across the three methods. In the second column, the blue rows highlight the features that are common between correlation and RFECV. In the third column, the yellow rows highlight the features that are common between correlation and SHAP. The results indicate:

• 26 features are shared between correlation Analysis and RFECV.

• 21 features are shared between correlation Analysis and SHAP.

• 12 features are common across all three methods.

The significant overlap among the selected features indicates that the regression models tend to prioritize features that efectively capture trends in classifier accuracy. Interestingly, image-specific features such as texture and edge-related metrics were not among the top features selected based on correlation analysis. However, texture features were included among those selected by RFECV, and both texture and edge features were identified as important using SHAP analysis.

## 4.4 Classifier-Based Accuracy Prediction

For training our regression models, the 38 features that are selected by the correlation method are used as input, and the accuracy of each classifier is used as the target variable. The 15 classifiers used in this work are Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Multilayer Perceptron with 3 Layers (MLP3), Multilayer Perceptron with 5 Layers (MLP5), Random Forest (RF), Decision Tree (DT), Naive Bayes (NB), Extra Trees (ET), Gradient Boosting (GB), Quadratic Discriminant Analysis (QDA), Linear Discriminant Analysis (LDA), Logistic Regression (LR), Ridge, AdaBoost (AB), and Dummy.

Each classifier brings a distinct perspective to the training process. LR models class probabilities using a logistic function, while Ridge incorporates an $L _ { 2 }$ penalty to reduce overfitting. SVM identifies the optimal separating hyperplane in high-dimensional settings, and LDA projects features along axes that maximize class separation. In contrast, QDA models each class as an independent Gaussian distribution, while NB applies Bayes’ theorem under a conditional feature independence assumption. Instance- and rule-based methods include KNN, which assigns labels via local majority voting, and DT, which recursively partitions feature space for interpretability. Tree ensembles expand on this: RF aggregates diverse trees to mitigate variance, ET introduces randomized split thresholds to limit overfitting, GB sequentially minimizes residual errors for high predictive accuracy, and AB iteratively upweights dificult instances. Neural representations are captured by MLP3 and MLP5 to model patterns across three and five layers, respectively. Finally, Dummy serves as a heuristic baseline to benchmark empirical performance.

Since the classifiers ofer diferent perspectives, regression models in Table 1 are trained separately for each classifier. For each classifier, the best regression model are selected. Prediction errors are computed on the test set, with the corresponding MSE and MAPE metrics reported in Table 5.

## 4.5 Select Best Model

Based on the predicted accuracies, a ranking of classifiers can be generated for each dataset, with the highest rank corresponding to the best-performing model. The ranking accuracy achieved using all features is 50%, while using correlation-based selected features yields 40% accuracy. During the analysis, we observed that some classifiers exhibit similar or closely related behavior on certain datasets. Consequently, we propose a method for selecting the best-performing group of classifiers.

Table 4: Comparison of features selected with three feature selection methods.
<table><tr><td>Correlation</td><td>Pycaret</td><td>SHAP</td></tr><tr><td>nre</td><td>nodes_per_inst</td><td>attr_conc</td></tr><tr><td>pb</td><td>sil</td><td>n2</td></tr><tr><td>sil</td><td>n1</td><td>closeness_centrality</td></tr><tr><td>class_conc</td><td>n3</td><td>class_ent</td></tr><tr><td>class_ent</td><td>n4</td><td>f3</td></tr><tr><td>eq-num_attr</td><td>pb</td><td>nre</td></tr><tr><td>joint_ent</td><td>vdb</td><td>degree_centrality</td></tr><tr><td>leaves</td><td>density</td><td>ch</td></tr><tr><td>leaves_branch</td><td>int</td><td>class_conc</td></tr><tr><td>leaves_corrob</td><td>f3</td><td>mut_inf</td></tr><tr><td>leaves_homo</td><td>12</td><td>leaves-per_class</td></tr><tr><td>leaves-per_class</td><td>lsc</td><td>joint_ent</td></tr><tr><td>nodes</td><td>13</td><td>n_class</td></tr><tr><td>nodes-per_attr</td><td>f1v</td><td>hog</td></tr><tr><td>nodes-per_inst</td><td>f4</td><td>sil</td></tr><tr><td>nodes_per_level</td><td>n2</td><td>lbp</td></tr><tr><td>nodes_repeated</td><td>hubs</td><td>eq-num_attr</td></tr><tr><td>tree_depth</td><td>joint_ent</td><td>density</td></tr><tr><td>tree_imbalance</td><td>nodes_repeated</td><td>attr_ent</td></tr><tr><td>tree_shape vdb</td><td>nodes-per_level</td><td>ns_ratio</td></tr><tr><td></td><td>nodes_per_attr</td><td>csg</td></tr><tr><td>f3 f4</td><td>leaves_branch</td><td>edge_betweenness_centrality</td></tr><tr><td></td><td>tree_depth</td><td>glcm</td></tr><tr><td>n1</td><td>msep</td><td>eigenvector_centrality</td></tr><tr><td>n2</td><td>class_ent</td><td>msep</td></tr><tr><td>n3</td><td>nre</td><td>vdu</td></tr><tr><td>n4</td><td>tree_shape</td><td>f4</td></tr><tr><td>t1</td><td>ch</td><td>mvar</td></tr><tr><td>lsc</td><td>lbp</td><td>clscoef</td></tr><tr><td>density</td><td>attr_conc</td><td>bp</td></tr><tr><td>hubs</td><td>vdu</td><td>leaves_homo</td></tr><tr><td>n_class</td><td>closeness_centrality</td><td>hubs</td></tr><tr><td>csg</td><td>class_conc</td><td>betweenness_centrality</td></tr><tr><td>degree_centrality</td><td>f2</td><td>shannon</td></tr><tr><td>eigenvector_centrality</td><td>clustering</td><td>f1</td></tr><tr><td>closeness_centrality</td><td>11</td><td>clustering</td></tr><tr><td>betweenness_centrality</td><td>var_importance.mean</td><td>pagerank</td></tr><tr><td>msep</td><td>nodes</td><td>int</td></tr></table>

Table 5: Regressor Performance Metrics (MAE, MSE, MAPE) for each Classifier
<table><tr><td>Classifier</td><td>Regressor</td><td>MAE</td><td>MSE</td><td>MAPE</td></tr><tr><td>svm</td><td>et</td><td>0.0372</td><td>0.0025</td><td>0.0711</td></tr><tr><td>knn</td><td>et</td><td>0.0241</td><td>0.0011</td><td>0.0409</td></tr><tr><td>mlp3</td><td>et</td><td>0.0630</td><td>0.0077</td><td>0.1995</td></tr><tr><td>mlp5</td><td>et</td><td>0.1331</td><td>0.0321</td><td>0.4662</td></tr><tr><td>rf</td><td>ada</td><td>0.0002</td><td>0.0000</td><td>0.0002</td></tr><tr><td>dt</td><td>et</td><td>0.0087</td><td>0.0003</td><td>0.0097</td></tr><tr><td>nb</td><td>et</td><td>0.0508</td><td>0.0038</td><td>0.1220</td></tr><tr><td>et</td><td>dt</td><td>0.0001</td><td>0.0000</td><td>0.0001</td></tr><tr><td>gbc</td><td>et</td><td>0.0431</td><td>0.0036</td><td>0.0613</td></tr><tr><td>qda</td><td>gbr</td><td>0.0669</td><td>0.0092</td><td>0.1378</td></tr><tr><td>lda</td><td>et</td><td>0.0464</td><td>0.0033</td><td>0.1111</td></tr><tr><td>lr</td><td>et</td><td>0.0521</td><td>0.0041</td><td>0.1776</td></tr><tr><td>rc</td><td>et</td><td>0.0421</td><td>0.0028</td><td>0.1132</td></tr><tr><td>abc</td><td>et</td><td>0.0501</td><td>0.0042</td><td>0.1580</td></tr><tr><td>dummy</td><td>dt</td><td>0.0460</td><td>0.0068</td><td>0.2085</td></tr></table>

## 4.6 Select Best Group Model

In this section, we propose an alternative approach to ranking classifiers for a given dataset by combining clustering techniques with regression-based accuracy predictions. Instead of directly ranking classifiers by predicted accuracy, we cluster them based on their performance on the training data and use these clusters to refine the ranking process. This method addresses the observation that some classifiers demonstrate similar performance levels, where small diferences in accuracy do not justify separate rankings in the selection process.

## Clustering Classifiers Based on Performance

To cluster the classifiers, we use the actual accuracies of 15 classifiers on the training datasets. The K-means clustering algorithm is applied to group the classifiers based on their predicted accuracy values. The key parameter in clustering is the number of clusters, which is determined using the elbow method. Using this approach, four clusters are identified. The output of the elbow is shown in Figure 3. The final four clusters are as follows:

• Cluster 1: RF, DT, ET, GB

• Cluster 2: SVM, KNN, MLP3, MLP5, QDA

![](images/92b23d7e4b9bd69ec804addbf195e9df967870b6ad510be435ccd74874fa5ab7.jpg)  
Figure 3: Elbow output.

• Cluster 3: NB, LDA, LR, RC, ABC

• Cluster 4: Dummy

The rationale behind clustering is based on the similarity in performance among certain classifiers. By comparing their predicted accuracies, we observe that the diferences between classifiers within the same cluster are often minimal and do not significantly impact their relative efectiveness on a given dataset. Clustering simplifies the ranking process by grouping classifiers with comparable predictive capabilities, reducing the complexity of distinguishing between marginally diferent performers.

To evaluate the efectiveness of this approach, we follow these steps: First, we compute the true ranking of classifiers on the test datasets based on their actual classification accuracy. Next, we predict the accuracy of each classifier on the test datasets using the regression models selected for each classifier (as shown in Table 3) and establish a predicted ranking based on these values. The predicted ranking represents the order in which classifiers are recommended for a given dataset.

Clustering the classifiers before ranking prediction ofers several advantages. First, it reduces the framework’s sensitivity to small, often statistically insignificant accuracy diferences, focusing on broader performance trends instead. Second, it improves interpretability by grouping classifiers with similar strengths, making it easier for practitioners to identify which families of methods are most suitable for a given dataset. For instance, Cluster 1 (RF, DT, ET, GB) represents tree-based ensemble methods known for their robustness and generalization, while Cluster 2 (SVM, KNN, QDA, MLP3, MLP5) includes methods sensitive to data geometry and separability.

By applying this approach to the test datasets, we assess whether the predicted ranking, informed by both clustering and regression, captures the true order of classifier performance.

The results of this evaluation, including the accuracy of the cluster-based ranking predictions, will be presented in the next section once the clustering-specific metrics are fully analyzed. This hybrid approach, combining clustering and regression, aims to balance precision and practicality in classifier selection, enhancing the framework’s utility for diverse image classification tasks.

## 4.7 Results

In this section, we present the results of the proposed clustering and regression-based framework for predicting the ranking of 15 classifiers on test datasets. The framework leverages K-means clustering to group classifiers into four performance-based clusters and employs regression models to predict classifier accuracies, which are then used to establish a predicted ranking.

The prediction accuracies for each of the 15 ranking levels are summarized in Table 6. The framework achieves a mean prediction accuracy of 86.15% across all ranks, demonstrating good performance in identifying the relative efectiveness of classifiers for a given dataset.

Table 6: Prediction Accuracy and Cumulative Average for Classifier Rankings Across 15 Levels
<table><tr><td>Rank</td><td>Prediction Accuracy (%)</td><td>Cumulative Average (%)</td></tr><tr><td>1</td><td>95.3</td><td>95.30</td></tr><tr><td>2</td><td>100.0</td><td>97.65</td></tr><tr><td>3</td><td>98.4</td><td>97.90</td></tr><tr><td>4</td><td>81.3</td><td>93.75</td></tr><tr><td>5</td><td>75.0</td><td>90.00</td></tr><tr><td>6</td><td>81.3</td><td>88.55</td></tr><tr><td>7</td><td>90.6</td><td>88.84</td></tr><tr><td>8</td><td>78.1</td><td>87.50</td></tr><tr><td>9</td><td>78.1</td><td>86.46</td></tr><tr><td>10</td><td>71.9</td><td>85.00</td></tr><tr><td>11</td><td>89.1</td><td>85.37</td></tr><tr><td>12</td><td>87.5</td><td>85.55</td></tr><tr><td>13</td><td>85.9</td><td>85.58</td></tr><tr><td>14</td><td>89.1</td><td>85.83</td></tr><tr><td>15</td><td>90.6</td><td>86.15</td></tr></table>

We also evaluate the impact of diferent feature extraction methods on the framework’s performance. Four methods are compared: pca ae, pca vgg19, tsne ae, and tsne vgg19. Figure 4 shows the number of matches and mismatches for each method. Among these, feature extraction using pca ae—which combines an Autoencoder with PCA compression—yields the best performance in the proposed framework. Both correlation and accuracy metrics confirm the superior results of pca ae.

This outcome indicates that multiple factors influence the results. First, the number of features extracted from the pre-trained models afects performance. Second, applying transfer learning alongside pre-training contributes to improved results. Third, the dimensionality reduction approach, tailored specifically to each dataset, also plays a critical role. For the Autoencoder, the first two factors are considered, while for PCA, instead of using a fixed dimension, we select the number of components based on a fixed reconstruction accuracy, resulting in a variable number of features for each dataset. The proposed clustering

![](images/70c1e5546359f47e5080ffcdfa07361be54a43a6ccda95942d1999430a4a2e7a.jpg)  
Figure 4: Number of matched and mismatched predictions for each feature extraction method (pca ae, pca vgg19, tsne ae, tsne vgg19). Blue bars represent correct predictions (matches), while red bars indicate incorrect predictions (mismatches).

and regression-based framework demonstrates strong performance in predicting the rankings of 15 classifiers across various test datasets. The framework combines K-means clustering to group classifiers by performance similarity and regression models to estimate individual classifier accuracies, which are then used to derive predicted rankings. Key observations from Table 6:

High accuracy at top ranks: The prediction accuracy for the highest ranks (1st to 3rd) is exceptionally high, reaching above 95%, with rank 2 achieving a perfect 100% accuracy. This indicates that the framework is very efective at correctly identifying the top-performing classifiers for a dataset, which is often the most critical in practical applications.

Gradual decline in mid-rank accuracy: From ranks 4 to 10, prediction accuracy shows a moderate decline, fluctuating between approximately 72% and 90%. This suggests that while the framework remains fairly accurate at identifying mid-tier classifier rankings, some ambiguity or overlap exists among classifiers ranked in this middle range, possibly due to closer performance levels.

Stable accuracy for lower ranks: Interestingly, the accuracy for ranks 11 to 15 remains relatively stable and consistently high (around 85% to 91%). This implies that the framework is also reliable in distinguishing the lower-performing classifiers, helping to avoid recommending less efective models.

## 4.8 Comparison

To compare our work, we need to find articles that provide relevant features or datasets that can be used for comparison. A similar study that utilizes comparable features is [23]. In that work, complexity features are extracted from nearly 140 1D datasets, and the accuracy of four classifiers is calculated for each dataset. The goal of that study was to recommend the best classifier for each dataset based on these features and the classifiers’ accuracies.

However, since our datasets are image-based and the features we use difer from those in [23], it is important to compare the results and assess the impact of our proposed features. To this end, we applied the 22 features from [23] to our dataset and evaluated their performance. Using these features, we achieved a ranking accuracy of 85.41% on the test set. For a fair comparison, we also selected the first 22 features out of our 38 features, matching the same feature count. In this case, the ranking accuracy improved to 87.60%. These findings indicate that our features, which are selected from diferent view point for image-based datasets, provide better performance than those used in [23]. Moreover, the improvement observed when using 22 features instead of all 38 suggests that a higher correlation threshold for feature selection may help retain only the most relevant features and further enhance performance.

Instead of using the 38 selected features, we also conducted experiments with only the 12 features, which are common to the three selected feature methods. The results show that using these features reduces the complexity of feature extraction by nearly 70%, the accuracy decreases 5.32% (from 86.15% to 81.56%).

Among the 38 selected features, some describe the generated tree after training a decision tree model. Since DT is one of our classifiers, we remove these features (13 features) to avoid any influence on our decisions. As a result, the accuracy decreases slightly from 86.15% to 85.0%.

## 5 Comparison with Related Work

Several recent studies have examined the challenges of image classification under diverse conditions, including industrial environments, noisy data, and cross-dataset variability. While these works provide valuable insights into classifier behavior, their objectives and methodological focus difer substantially from our proposed meta-learning framework for classifier recommendation.

The work of [55] investigates image classification in industrial inspection settings, where datasets are typically domain-specific and collected under controlled acquisition conditions. Their primary objective is to optimize classification performance within a fixed industrial application domain. The study focuses on improving predictive accuracy in a specialized context rather than addressing cross-domain generalization or automated model selection.

Similarly, [56] examine classifier robustness under noisy image conditions. Their analysis evaluates how classification performance degrades as noise levels increase, providing insights into model reliability in adverse or corrupted data scenarios. The emphasis is placed on robustness evaluation through controlled perturbations, ofering a deeper understanding of classifier stability under noise.

The study by [57] highlights empirical performance variation of classifiers across multiple datasets, reinforcing the implications of the No Free Lunch theorem. Their findings demonstrate that no single classifier consistently outperforms others across all image distributions. The work primarily provides an empirical analysis of cross-dataset variability and underscores the dependency of classifier performance on dataset characteristics.

In a diferent domain, [58] investigate the efect of data complexity on classifier performance in the context of software defect prediction, relying on tabular software engineering datasets rather than image data. Their study analyzes how complexity measures influence classification outcomes and shows that classifier efectiveness strongly depends on dataset properties. This further supports the idea that no universally optimal classifier exists across all problems.

In contrast to these studies, our work proposes a unified and scalable meta-learning framework specifically designed for heterogeneous image datasets. Rather than optimizing performance within a single domain, analyzing robustness under specific perturbations, or solely conducting empirical performance comparisons, we aim to predict classifier performance directly from dataset characteristics. By extracting comprehensive image-based meta-features using autoencoders, pre-trained networks, and dimensionality reduction techniques, we train regression models to estimate classifier accuracies without exhaustive retraining. Furthermore, we incorporate clustering methods to group classifiers with similar performance patterns, simplifying the recommendation process.

## 6 Conclusion

In this paper, we proposed a meta-learning framework designed to predict the performance of various classifiers on image datasets by leveraging an extensive set of dataset complexity measures (meta-features). By extracting, selecting, and utilizing these meta-features, our approach enables eficient and accurate prediction of classifier accuracies without the need for exhaustive training and evaluation of all candidate models.

The framework employs feature extraction techniques such as autoencoders and pre-trained deep networks, combined with dimensionality reduction methods like PCA and t-SNE, to represent image datasets efectively. Through rigorous feature selection and the training of dedicated regression models for each classifier, the system identifies the most relevant features that influence classifier performance. Furthermore, clustering classifiers based on performance similarities simplifies decision-making by grouping models with comparable accuracies, improving interpretability and robustness.

Experimental results on a diverse set of 56 image datasets demonstrate that our method achieves a high average accuracy in predicting classifier rankings, notably excelling in identifying top-performing models. The approach also highlights the importance of using image-specific features and tailored meta-features to improve prediction performance.

Overall, this work contributes to the field by providing a scalable, interpretable, and efective method for classifier recommendation in image classification tasks. Future research may extend this framework to other types of data and explore more advanced meta-features and meta-models to further enhance prediction accuracy and decision-making capabilities.

## References

[1] A. C. Lorena, L. P. Garcia, J. Lehmann, M. C. Souto, T. K. Ho, How complex is your classification problem? a survey on measuring classification complexity, ACM Computing Surveys (CSUR) 52 (5) (2019) 1–34.

[2] B. A. Pimentel, A. C. De Carvalho, A new data characterization for selecting clustering algorithms using meta-learning, Information Sciences 477 (2019) 203–219.

[3] A. Rivolli, L. Garcia, C. Soares, J. Vanschoren, A. de Carvalho, Characterizing classification datasets: a study of meta-features for metalearning (1808).

[4] C. Perez, R. Germon, Graph creation and analysis for linking actors: Application to social data, in: Automating open source intelligence, Elsevier, 2016, pp. 103–129.

[5] J. Bang-Jensen, G. Z. Gutin, Digraphs: theory, algorithms and applications, Springer Science & Business Media, 2008.

[6] L. C. Freeman, A set of measures of centrality based on betweenness, Sociometry (1977) 35–41.

[7] U. Brandes, On variants of shortest-path betweenness centrality and their generic computation, Social networks 30 (2) (2008) 136–145.

[8] L. Page, S. Brin, R. Motwani, T. Winograd, The pagerank citation ranking: Bringing order to the web., Tech. rep., Stanford infolab (1999).

[9] M. E. Newman, The mathematics of networks, The new palgrave encyclopedia of economics 2 (2008) (2008) 1–12.

[10] F. Branchaud-Charron, A. Achkar, P.-M. Jodoin, Spectral metric for dataset complexity assessment, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 3215–3224.

[11] P. J. Rousseeuw, Silhouettes: a graphical aid to the interpretation and validation of cluster analysis, Journal of computational and applied mathematics 20 (1987) 53–65.

[12] T. Cali´nski, J. Harabasz, A dendrite method for cluster analysis, Communications in Statistics-theory and Methods 3 (1) (1974) 1–27.

[13] D. L. Davies, D. W. Bouldin, A cluster separation measure, IEEE transactions on pattern analysis and machine intelligence (2) (1979) 224–227.

[14] J. C. Dunn, Well-separated clusters and optimal fuzzy partitions, Journal of cybernetics 4 (1) (1974) 95–104.

[15] J. C. Bezdek, N. R. Pal, Some new indexes of cluster validity, IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) 28 (3) (1998) 301–315.

[16] A. A. Rahane, A. Subramanian, Measures of complexity for large scale image datasets, in: 2020 international conference on artificial intelligence in information and communication (ICAIIC), IEEE, 2020, pp. 282–287.

[17] N. Dalal, B. Triggs, Histograms of oriented gradients for human detection, in: 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), Vol. 1, Ieee, 2005, pp. 886–893.

[18] D. Huang, C. Shan, M. Ardabilian, Y. Wang, L. Chen, Local binary patterns and its application to facial image analysis: a survey, IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews) 41 (6) (2011) 765–781.

[19] H. Cho, S. Lee, Data quality measures and eficient evaluation algorithms for large-scale high-dimensional data, Applied Sciences 11 (2) (2021) 472.

[20] Q. Leng, L. Zhou, X. Meng, A complexity measure for data classification based on knn with dynamic optimal k-value finding, in: Fuzzy Systems and Data Mining IX, IOS Press, 2023, pp. 739–746.

[21] S. Guan, M. Loew, H. Ko, Data separability for neural network classifiers and the development of a separability index, arXiv preprint arXiv:2005.13120 (2020).

[22] J. Maillo, I. Triguero, F. Herrera, Redundancy and complexity metrics for big data classification: Towards smart data, IEEE Access 8 (2020) 87918–87928.

[23] L. P. Garcia, A. C. Lorena, M. C. de Souto, T. K. Ho, Classifier recommendation using data complexity measures, in: 2018 24th International Conference on Pattern Recognition (ICPR), IEEE, 2018, pp. 874–879.

[24] G. T. Pereira, M. R. d. Santos, A. C. P. d. L. F. de Carvalho, Evaluating meta-feature selection for the algorithm recommendation problem, arXiv preprint arXiv:2106.03954 (2021).

[25] S. Mekouar, Classifiers selection based on analytic hierarchy process and similarity score for spam identification, Applied Soft Computing 113 (2021) 108022.

[26] Y. Chen, C. Guan, Z. Wei, X. Wang, W. Zhu, Metadelta: A meta-learning system for few-shot image classification, in: AAAI Workshop on Meta-Learning and MetaDL Challenge, PMLR, 2021, pp. 17–28.

[27] S. M. Hendryx, A. B. Leach, P. D. Hein, C. T. Morrison, Meta-learning initializations for image segmentation, arXiv preprint arXiv:1912.06290 (2019).

[28] T. Leng, Y. Zhang, K. Han, X. Xie, Self-sampling meta sam: enhancing few-shot medical image segmentation with meta-learning, in: Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2024, pp. 7925–7935.

[29] S. Luo, Y. Li, P. Gao, Y. Wang, S. Serikawa, Meta-seg: A survey of meta-learning for image segmentation, Pattern Recognition 126 (2022) 108586.

[30] J. R. Rice, The algorithm selection problem, Advances in Computers 15 (1976) 65–118.

[31] K. Simonyan, Very deep convolutional networks for large-scale image recognition, arXiv preprint arXiv:1409.1556 (2014).

[32] I. T. Jollife, J. Cadima, Principal component analysis: a review and recent developments, Philosophical transactions of the royal society A: Mathematical, Physical and Engineering Sciences 374 (2065) (2016) 20150202.

[33] L. Van der Maaten, G. Hinton, Visualizing data using t-sne., Journal of machine learning research 9 (11) (2008).

[34] K. Weiss, T. M. Khoshgoftaar, D. Wang, A survey of transfer learning, Journal of Big data 3 (2016) 1–40.

[35] S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, R. Garnett (Eds.), Advances in Neural Information Processing Systems 30, Curran Associates, Inc., 2017, pp. 4765–4774.

[36] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, A. Gulin, Catboost: unbiased boosting with categorical features, Advances in neural information processing systems 31 (2018).

[37] T. Chen, C. Guestrin, Xgboost: A scalable tree boosting system, Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining (2016) 785–794.

[38] R. Zemel, T. Pitassi, A gradient-based boosting algorithm for regression problems, Advances in neural information processing systems 13 (2000).

[39] L. Breiman, Random forests, Machine learning 45 (2001) 5–32.

[40] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, Lightgbm: A highly eficient gradient boosting decision tree, Advances in neural information processing systems 30 (2017).

[41] A. E. Hoerl, R. W. Kennard, Ridge regression: Biased estimation for nonorthogonal problems, Technometrics 42 (2000) 80–86.

[42] G. A. F. Seber, A. J. Lee, Linear regression analysis, Vol. 330, John Wiley and Sons, 2003.

[43] H. Zou, T. Hastie, Regularization and variable selection via the elastic net, Journal of the royal statistical society: series B (statistical methodology) 67 (2005) 301–320.

[44] R. Tibshirani, Regression shrinkage and selection via the lasso, Journal of the Royal Statistical Society: Series B (Methodological) 58 (1996) 267–288.

[45] G. Ridgeway, D. Madigan, T. S. Richardson, Boosting methodology for regression problems, Seventh International Workshop on Artificial Intelligence and Statistics (1999).

[46] Q. Sun, W.-X. Zhou, J. Fan, Adaptive huber regression, Journal of the American Statistical Association 115 (2020) 254–265.

[47] A. E. G. Kalatzis, C. F. Bassetto, C. R. Azzoni, Multicollinearity and financial constraint in investment decisions: A bayesian ridge regression, Proceedings of the International Conference on Applied Economics (2008) 437–447.

[48] M. Xu, P. Watanachaturaporn, P. K. Varshney, M. K. Arora, Decision tree regression for soft classification of remote sensing data, Remote Sensing of Environment 97 (2005) 322–336.

[49] Y. C. Pati, R. Rezaiifar, P. S. Krishnaprasad, Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition, Proceedings of 27th Asilomar conference on signals, systems and computers (1993) 40–44.

[50] K. Crammer, O. Dekel, J. Keshet, S. Shalev-Shwartz, Y. Singer, Online passive aggressive algorithms (2006).

[51] O. Kramer, Unsupervised k-nearest neighbor regression, arXiv preprint arXiv:1107.3600 (2011).

[52] C. Fraley, T. Hesterberg, Least angle regression and lasso for large datasets, Statistical Analysis and Data Mining: The ASA Data Science Journal 1 (2009) 251–259.

[53] D. B. Suits, Use of dummy variables in regression equations, Journal of the American Statistical Association 52 (1957) 548–551.

[54] P. Geurts, D. Ernst, L. Wehenkel, Extremely randomized trees, Machine learning 63 (2006) 3–42.

[55] L. F. M. Sepulveda, A. C. Silva, J. O. B. Diniz, Meta-learning applied to the selection of the classification methods in industrial images, in: Anais do 14° Simp´osio Brasileiro de Automa¸c˜ao Inteligente (SBAI 2019), Ouro Preto, Brazil, 2019.

[56] J. de Hoog, A. Anwar, P. Hellinckx, S. Mercelis, Selection of image classifiers for noisy images through meta-learning, in: Proc. of the 7th International Conference on Machine Vision and Applications (ICMVA 2024), 2024, pp. 92–99.

[57] R. L. Theriault, R. E. Ellis, Data complexity measures can predict classification accuracy, SSRN Preprint (2025).

[58] J. Eberlein, D. Rodriguez, R. Harrison, The efect of data complexity on classifier performance, Empirical Software EngineeringOnline First (2024).