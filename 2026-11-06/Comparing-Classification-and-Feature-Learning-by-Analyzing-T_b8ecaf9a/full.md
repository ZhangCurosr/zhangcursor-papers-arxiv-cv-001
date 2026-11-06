# Comparing Classification and Feature Learning by Analyzing Their Loss Functions in Normalized Spaces with Diferent Scale Factors

Qiufu Li<sup>1,2,4</sup>, Huibin Xiao<sup>1,2,4</sup>, Linlin Shen<sup>1,2,4\*</sup>, Weijia Fan<sup>3</sup>, Xi Jia<sup>5</sup>, Yongdong Huang<sup>6</sup>

<sup>1</sup>School of Artificial Intelligence, Shenzhen University, Shenzhen, 518060, China.

<sup>2</sup>National Engineering Laboratory for Big Data System Computing Technology, Shenzhen University, Shenzhen, 518060, China.

<sup>3</sup>College of Computer Science and Software Engineering, Shenzhen University, Shenzhen, 518060, China.

<sup>4</sup>Guangdong Provincial Key Laboratory of Intelligent Information Processing, Shenzhen University, Shenzhen, 518060, China.

<sup>5</sup>Big Data Institute, University of Oxford, Oxford, OX3 7LF, United Kingdom.

<sup>6</sup>Institute of Image Understanding Research, North Minzu University, Yinchuan, 750021, China.

\*Corresponding author(s). E-mail(s): llshen@szu.edu.cn;

## Abstract

Classification and feature learning are two closely interrelated fundamental tasks. Feature learning in supervised settings is typically achieved by training classification models, under the common assumption that higher classification accuracy implies better feature quality. In this paper, we first point out that classification accuracy reflects the sample-wise local properties of a model on a dataset, whereas feature learning aims to learn features with good dataset-level global properties, such as intra-class compactness and inter-class separability. From the first principle of feature learning, we redesign the binary cross-entropy (BCE) loss, a counterpart of the cross-entropy (CE) loss used in classification. Then, in normalized feature space, we analyze the minima of the BCE and CE losses and show that both can lead to neural collapse, where sample features from diferent classes become maximally compact and maximally separated. More importantly,

by investigating the convergence behaviors of the two losses as the scale factor γ varies in the normalized space, we reveal how classification accuracy and feature quality change diferently across spaces of diferent sizes. Under a fixed training strategy, a very large γ achieves perfect classification accuracy but slows convergence toward the ideal feature geometry, whereas a very small γ makes classification highly sensitive to small perturbations. A moderate γ is therefore necessary for simultaneously achieving reliable classification and well-structured features. Extensive experiments validate these findings and demonstrate that BCE loss exhibits advantages in long-tail and open-set recognitions, as it better aligns with the global requirements of feature learning.

Keywords: Classification, Feature Learning, Cross-entropy, Binary Cross-entropy, Scale Factor

## 1 Introduction

In machine learning, classification and feature learning are fundamental and closely intertwined tasks in supervised scenarios, a connection that can be traced back to Fisher’s linear discriminant analysis (FISHER, 1936), which achieves efective classification by learning a linear projection that makes the projected feature vectors as compact as possible within classes and as separable as possible between classes. In modern deep learning, classification tasks commonly take the cross-entropy (CE) loss to train deep feature extraction models and linear classifiers, and they could learn desirable features, making the trained models widely applied in downstream fields that require features with well properties, such as face recognition (W. Liu, Wen, et al., 2023), object detection and tracking (Jain et al., 2024), image segmentation (Azad et al., 2024), and image-text retrieval (Vasu et al., 2024), etc. For feature learning, while feature properties required in diferent scenarios are diferent, intra-class compactness and inter-class separability are most attractive in many tasks. Intuitively, the higher classification accuracy usually implies the better feature properties, and conversely, the better feature properties suggest the higher accuracy.

It should be noted that the classification accuracy of a model is calculated by checking whether each sample is correctly classified, reflecting the local properties of the model on a dataset. In contrast, the intra-class compactness and inter-class separability are the global properties of the model on the whole dataset. As Fig. 1 shows, correct classification only requires that a sample’s feature vector be relatively closer to its corresponding class center/classifier vector than to other class centers; whereas feature learning demands that all samples’ feature vectors be as close as possible to their own class centers while simultaneously remaining as far as possible from other class centers. Therefore, high classification accuracy alone is insuficient to guarantee good feature representations. That is to say, there is an essential diference between the classification and feature learning. However, there has not been a indepth analysis of this diference, and in supervised scenarios, the feature learning is usually implemented through training classification models without considering its global nature.

![](images/2cb68d2f6df1396b4f9d2ae49ab6caa7105a1dd031e48d9ba45c7abfeb941d8e.jpg)  
Fig. 1: The intuitive diference between classification and feature learning. Classification (left) requires that the feature vector of each sample be closer to its own class center/classifier vector than to those of other classes, and classification accuracy reflects the model’s local properties at each sample. In contrast, intra-class compactness and inter-class separability of sample features are the model’s global properties at the dataset level, and feature learning (right) drives the feature vectors of all samples to be as close as possible to their respective class centers while simultaneously being far from other class centers

In unbounded Euclidean space, measuring intra-class compactness and inter-class separability of features using distance-based metrics cannot guarantee a fair comparison across diferent models. A more common approach is to use cosine similarity in a normalized space to measure the compactness and separability. When training models in practice, normalizing features can improve the training stability, accelerate the convergence, and enhance the models’ generalization, which is a frequent regularization strategy in various fields, especially in face recognition (F. Wang et al., 2017). Feature normalization is generally achieved by dividing by the $L _ { \mathrm { { 2 } } } \mathrm { { - n o r m } }$ of the feature vectors, which scales the feature space into a hypersphere of unit radius. To better adapt to downstream tasks, when training models in normalized space, the normalized features are typically multiplied by a scale factor γ before being input into the training loss. From a classification perspective, it is generally believed that a large γ sharpens the probability distribution of features belonging to the true class (Zheng & Yang, 2024), enhancing the model’s confidence in its classification decisions, while a small one smooths the probability distribution, suppressing noise efects. While the above empirical observations exist, there is currently a lack of in-depth theoretical understanding of the scale factor or a convincing explanation of how it afects the feature learning and classification in normalized spaces.

In deep learning, although loss functions such as contrastive loss and triplet loss are commonly used for feature learning, they operate only at the local level of individual sample pairs or triplets and do not account for the global nature of intra-class compactness and inter-class separability. Besides that, the CE loss for training classification models can also yield well features with desirable properties, and there is theoretical evidence (Yaras et al., 2022) that when the CE reaches its minimum, the sample feature and classifier vectors exhibit neural collapse (NC), at which point the feature compactness and discriminability are maximized uniformly across diferent classes. However, the CE loss is designed for classification and does not address the global nature of feature learning; therefore, it is not always eficient in more challenging scenarios such as long-tailed recognition (LTR) and open-set recognition (OSR).

In this paper, in the normalized spaces, we first redesign binary cross-entropy (BCE) from the “first principles” of feature learning. We then compare feature learning and classification under diferent scale factors in normalized spaces by analyzing the minima and convergence behavior of BCE and comparing them with those of CE, which is commonly used for classification. The main contributions are summarized as follows.

• For the first time, we in-depth reveal the distinction between classification and feature learning. Classification is depend on sample-level relative ordering of decision scores of each sample, while feature learning expect dataset-level global properties, such as intra-class compactness and inter-class separability. With the basic requirements of feature learning, we derive a loss function which turns out to be the binary cross-entropy (BCE) loss used for multi-class tasks. Meanwhile, we prove that when BCE reaches its minima, it can uniformly maximize the compactness and separability of features across diferent classes. Although the CE used for classification also exhibits this property, the advantage of BCE lies in its classifier biases, which help it achieve this desirable property more rapidly.

• We make an in-depth analysis of the convergence behavior of the CE and BCE as the scale factor γ changes, to further reveal the diferences between classification and feature learning in normalized spaces of diferent sizes. As γ increases, the convergence rates of the two losses may decay exponentially. Consequently, in a large normalized space with a very large γ, achieving high classification performance becomes relatively easy, whereas learning features with desirable properties becomes dificult. Conversely, as γ approaches zero, the convergence rates decrease linearly. Thus, in a very small space with γ close to 0, the setting is unfavorable for feature learning and even more so for classification.

• We conduct extensive experiments using deep models, including ResNet and ViT. Using the fixed training strategy, when the scale factor $\gamma = 6 4$ is very large, all the models achieve 100% classification accuracy on the training set, but the intra-class compactness and inter-class separability of features they extract are comparatively poor. When $\gamma = 0 . 1$ is very small, the models’ feature properties are unsatisfactory but superior to those at $\gamma = 6 4$ , while their classification performs very poor. When γ is at a moderate level, the models’ classification and feature properties can both reach their optimal points. In addition, the superior performance of BCE in more challenging scenarios that require higher-quality features, such as LTR and OSR, further demonstrates the practical relevance of the above analysis.

In our previous work (Q. Li et al., 2025) published in ICML2025, we have investigated the optimization dynamics of models in Euclidean space trained with BCE and CE losses integrated with regularization terms, demonstrating for the first time that the BCE can induce neural collapse. It further showed that the BCE loss explicitly enhances intra-class compactness and inter-class separability of features through decision score constraint and classifier bias.

Building on the ICML work (Q. Li et al., 2025), this paper addresses a diferent and broader question: whether classification and feature learning are equivalent in normalized spaces of diferent sizes. To this end, we reformulate the BCE loss based on the global requirements of feature learning, analyze the minima of the normalized BCE and CE losses, and further investigate how the scale factor γ, which controls the size of the normalized space, afects their convergence behavior. Our new theoretical analysis, study of the scale factor, comparison between local classification and global feature learning, and long-tailed/open-set recognition evaluations substantially extend the previous work.

## 2 Related Works

Classification and feature learning. The success of deep models such as ResNet (He et al., 2016) and ViT (Dosovitskiy et al., 2021a) in classification has continually driven the development of deep learning. The deep classification models have been consistently developed, including DenseNet (Huang et al., 2017), MobileNet (Howard et al., 2017), Swin Transformer (Z. Liu et al., 2021), and ConvNeXt (Z. Liu et al., 2022), etc., all of them consist of deep feature extraction models with CNN or Transformer architectures along with linear classifiers. In supervised scenarios, employing the CE loss and training these models on large amounts of labeled data can yield good classification alongside robust feature extraction capabilities, making them widely applicable in fields such as computer vision, natural language processing, and multi-modal large models.

For the tasks that require better feature properties, such as face recognition and long-tailed recognition, normalizing sample features and the classifier vectors can efectively enhance the performance of deep models. NormFace (F. Wang et al., 2017) and SphereFace (W. Liu et al., 2017) are among the earliest face recognition works, and advanced methods such as CosFace (H. Wang et al., 2018), ArcFace (Deng et al., 2019), TopoFR (Dan et al., 2024), and GFace (Zhao et al., 2025) continue to be developed in the normalized space. On imbalanced datasets, normalization can suppress the scale diferences between the classifier vectors, thereby enhancing classification accuracy (J. Liu et al., 2020). In current, various re-balancing strategies have been designed based on the normalized CE and are widely applied in long-tailed recognition (LTR) tasks (Z. Chen et al., 2026; Han, 2023; J. Li et al., 2023; Liang et al., 2024).

Scale factor and temperature. In the normalized space, a scale factor γ is typically used to adjust the size of feature space (F. Wang et al., 2017), while it is temperature coeficient τ when it is applied to the denominator in contrastive learning (He et al., 2020). An inappropriate scale factor can significantly degrade the model performance, yet there is no clear and reasonable explanation for its efects. A commonly circulated notion is that large scale factor sharpens the probability distribution of the models’ outputs, increasing the confidence in the classification decisions, while the small one smooths the probability distribution, helping to mitigate the efects of noise and other influences. Then, in the classification losses, the scale factor is usually greater than 1, with a typical value of $\gamma = 6 4$ in face recognition (Deng et al., 2019; Zhou et al., 2023). The temperature used in the contrastive learning plays a similar role as that of the scale factor, and it on the denominator is usually set to a value less than 1. In this paper, for the first time, we reveal the role of the scale factor in the deep feature learning, which also helps people understand the temperature coeficient.

Minima of loss function. When training a classification model and the loss reaches a minimum, it maximizes intra-class compactness and inter-class separability of sample features, and features from diferent classes form an equiangular tight frame (ETF); this phenomenon is known as neural collapse (NC), first discovered in Papyan et al. (2020). The NC studies on CE loss have been extended to various scenarios, including class imbalance (Dang et al., 2024; Fang et al., 2021; Gao et al., 2024), outof-distribution data (Z. Chen et al., 2024), and fixed classifiers (Kim & Kim, 2024), as well as other CE variants such as focal loss, label smoothing loss.

As previously stated, we have compared BCE and CE from perspective of NC in Euclidean space (Q. Li et al., 2025). For the case in the normalized space, it has proved that the global minima of CE loss are all NC solutions (Lu & Steinerberger, 2022; Yaras et al., 2022). It should be noted that existing theoretical studies on NC primarily address whether a given loss function can induce the optimal feature geometry at its global minimum. However, merely establishing the inevitability of NC does not explain why a model can achieve high classification performance before learning well-structured feature representations, why a model with optimal classification performance does not necessarily attain optimal performance on feature-contrastive tasks, or how the corresponding loss functions converge during the model training. Therefore, this paper shifts the focus from “whether the loss function can reach $\mathrm { N C } ^ { \mathfrak { s } }$ to “explaining how classification and feature learning approach their respective desired conditions across feature spaces of diferent sizes, by analyzing the convergence behaviors of CE and BCE.” The normalized feature space provides a suitable framework for this analysis, while the scale factor γ directly controls the size of the space and afects the convergence behaviors of the CE and BCE losses.

## 3 Preliminaries

Let $\mathcal { D } = \bigcup _ { k = 1 } ^ { K } \big \{ X _ { i } ^ { ( k ) } \big \} _ { i = 1 } ^ { n _ { k } }$ be a sample set from K classes, where $X _ { i } ^ { ( k ) }$ is the i-th sample of the k-th class and $n _ { k }$ is number of samples in the class. In feature learning, for $\forall X \in { \mathcal { D } }$ , a feature extractor maps the sample into its feature vector $\textbf { \em h } =$ $\mathcal { M } ( \pmb { X } ) \in \mathbb { R } ^ { d }$ , where d denotes feature dimension.

For classification, a classifier converts the feature vector into a decision vector $z \ \in \ \mathbb { R } ^ { K } , \ { \mathrm { i . e . } } , \ z \ = \ { \mathcal { C } } { \big ( } { \mathcal { M } } ( X ) { \big ) } \ = \ { \mathcal { C } } ( h )$ . In deep learning, the classifier is commonly represented by a fully connection layer containing a weight matrix $W =$ $[ { \pmb w } _ { 1 } , { \pmb w } _ { 2 } , \cdots , { \pmb w } _ { K } ] ^ { \top } \in \mathbb { R } ^ { K \times d }$ and a bias vector $\pmb { b } = [ b _ { 1 } , b _ { 2 } , \cdots , b _ { K } ] ^ { \top } \in \mathbb { R } ^ { K }$ . In normalized space, both sample features and classifier vectors are normalized to $\| h \| _ { 2 } = 1$ and $\| \pmb { w } _ { j } \| _ { 2 } = 1$ , for $\forall j \in [ K ]$ . Then, a scale factor $\gamma > 0$ is multiplied on the decision vector to adjust the size of feature space, $\mathrm { i . e . , } z = \gamma W h - b$

In the following, we will re-derive the cross-entropy (CE) loss and binary crossentropy (BCE) loss functions used in multiclass tasks, and show that they are fundamentally suited to classification and feature learning, respectively.

## 3.1 CE for Classification

In a classification task, for $\forall X \in { \mathcal { D } }$ with feature vector $\pmb { h } = \mathcal { M } ( \pmb { X } )$ , its predicted class label <sup>ˆ</sup>k is decided by the vector $z = [ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } - b _ { j } ] _ { j = 1 } ^ { K }$ , and $\begin{array} { r } { \hat { k } = \arg \operatorname* { m a x } _ { j } \{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } - b _ { j } \} _ { j = 1 } ^ { K } } \end{array}$ For any sample $X ^ { ( k ) }$ from class $k .$ , in its decision vector $z ^ { ( k ) } = \gamma W h ^ { ( k ) } - b ,$ we refer to the k-th component $\gamma { \pmb w } _ { k } ^ { \top } { \pmb h } ^ { ( k ) } - b _ { k }$ as its positive decision score and the others $\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } ^ { ( k ) } - b _ { j } \} _ { j \neq k }$ as negative decision scores. Then, for correct classification, it only requires that the sample satisfy a relative ordering where its positive decision score exceeds its negative ones,

$$
\gamma \pmb { w } _ { k } ^ { \top } \pmb { h } ^ { ( k ) } - b _ { k } > \operatorname* { m a x } \{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } ^ { ( k ) } - b _ { j } \} _ { j \neq k } .\tag{1}
$$

Classification accuracy  is calculated by checking whether each sample is correctly classified via Eq. (1), equal to ratio of number of correctly classified samples to sample number of dataset ,

$$
\mathcal { A } ( \mathcal { M } , \mathcal { C } ) = \frac { | \mathcal { D } ( \mathcal { M } , \mathcal { C } ) | } { | \mathcal { D } | } \times 1 0 0 \% ,\tag{2}
$$

where

$$
{ \mathcal { D } } ( { \mathcal { M } } , { \mathcal { C } } ) = \bigcup _ { k = 1 } ^ { K } \left\{ { \boldsymbol { X } } ^ { ( k ) } : k = \arg \operatorname* { m a x } _ { { \boldsymbol { \ell } } } \{ \gamma w _ { { \boldsymbol { \ell } } } ^ { \top } h ^ { ( k ) } - b _ { { \boldsymbol { \ell } } } \} , \ { \boldsymbol { X } } ^ { ( k ) } \in { \mathcal { D } } _ { k } , \ h ^ { ( k ) } = { \mathcal { M } } ( { \boldsymbol { X } } ^ { ( k ) } ) \right\}\tag{3}
$$

consisting of the samples correctly classified by  and  in . Obviously, the classification accuracy  reflect a local property of the feature extraction model  at each sample on .

During training of a classification model, the decision vector $z ^ { ( k ) }$ of sample $X ^ { ( k ) }$ is converted via Softmax into predicted probabilities, $\begin{array} { r } { \left\{ \hat { p } _ { j } = \frac { \exp \left( \gamma \pmb { w } _ { j } \pmb { h } ^ { ( k ) } - b _ { j } \right) } { \sum _ { \ell = 1 } ^ { K } \exp \left( \gamma \pmb { w } _ { \ell } \pmb { h } ^ { ( k ) } - b _ { \ell } \right) } \right\} _ { j = 1 } ^ { K } , } \end{array}$ of the sample belonging to each class. For $X ^ { ( k ) }$ , its true probabilities are $p _ { k } \ = \ 1$ and $p _ { j } = 0$ for k and j with $j \neq k ,$ . Then, cross-entropy (CE) loss is calculated and minimized to drive the model training,

$$
\mathcal { L } _ { \mathrm { c e } } ( \boldsymbol { X } ^ { ( k ) } ) = - \sum _ { j = 1 } ^ { K } p _ { j } \log ( \hat { p } _ { j } ) = - \log \Bigg ( \frac { \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h ^ { ( k ) } - b _ { k } } } { \sum _ { \ell = 1 } ^ { K } \mathrm { e } ^ { \gamma w _ { \ell } ^ { \top } h ^ { ( k ) } - b _ { \ell } } } \Bigg ) .\tag{4}
$$

The CE loss is one of the most widely used loss function in deep learning.

## 3.2 BCE for Feature Learning

In supervised settings, the sample features can be learned in the training of classification models using the CE loss, while this loss does not take into account the sample-independent global constrains required for inter-class compactness and inter-class separability of all sample features on dataset.

In the normalized space, to precisely measure the properties of sample features $\textstyle \bigcup _ { k = 1 } ^ { K } \left\{ h _ { i } ^ { ( k ) } \right\} _ { i = 1 } ^ { n _ { k } }$ , we define the metrics of intra-class compactness $\mathcal { E } _ { \mathrm { c o m } }$ and inter-class separability $\mathcal { E } _ { \mathrm { s e p } }$ as

$$
\mathcal { E } _ { \mathrm { c o m } } = \frac { 1 } { 2 } \bigg [ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bigg ( \frac { 1 } { n _ { k } ^ { 2 } } \sum _ { i = 1 } ^ { n _ { k } } \sum _ { i ^ { \prime } = 1 } ^ { n _ { k } } \big \langle h _ { i } ^ { ( k ) } , h _ { i ^ { \prime } } ^ { ( k ) } \big \rangle \bigg ) + 1 \bigg ] ,\tag{5}
$$

$$
\mathcal { E } _ { \mathrm { s e p } } = \frac { 1 } { 2 } \Bigg [ 1 - \frac { 1 } { K ( K - 1 ) } \sum _ { k = 1 } ^ { K } \sum _ { \boldsymbol { k } ^ { \prime } = 1 \atop k ^ { \prime } \ne k } ^ { K } \left( \frac { 1 } { n _ { k } } \frac { 1 } { n _ { k ^ { \prime } } } \sum _ { i = 1 } ^ { n _ { k } } \sum _ { i ^ { \prime } = 1 } ^ { n _ { k ^ { \prime } } } \left. \boldsymbol { h } _ { i } ^ { ( k ) } , \boldsymbol { h } _ { i ^ { \prime } } ^ { ( k ^ { \prime } ) } \right. \right) \Bigg ] .\tag{6}
$$

Eqs. (5) and $( 6 )$ compute the compactness and separability using the feature similarities of all same-class sample pairs and all diferent-class sample pairs, respectively, which is entirely reasonable. However, if all these feature similarities are taken into account in the loss function for feature learning, the computational cost increases significantly. One can observe that, if classifier vectors $\{ w _ { k } \} _ { k = 1 } ^ { \bar { K } }$ are used as anchors, then the distribution of the unbiased decision scores $\cup _ { k = 1 } ^ { K } \cup _ { i = 1 } ^ { n _ { k } } \{ \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } \} _ { j = 1 } ^ { K }$ for all samples could reflect the compactness and separability of features. Moreover, since the number of classes K is usually smaller than the number of samples, $\begin{array} { r } { N = \sum _ { k = 1 } ^ { K } n _ { k } } \end{array}$ ， the number of decision scores is substantially lower than that of sample pairs, so using these scores could improve the computational eficiency. Therefore, we re-express the two properties using the decision scores to design a loss function for feature learning.

For the sample features $\{ \boldsymbol { h } _ { i } ^ { ( k ) } = \mathcal { M } ( \boldsymbol { X } _ { i } ^ { ( k ) } ) \} _ { i = 1 } ^ { n _ { k } }$ of class $k ,$ taking the k-th classifier vector ${ \pmb w } _ { k }$ as an anchor, the high intra-class compactness requires that all the features should lie close to ${ \pmb w } _ { k }$ . To explicitly measure this closeness, we choose the smallest possible unified threshold $t _ { k } ^ { \prime }$ such that the maximum distance from the feature vectors to the anchor is no greater than this threshold, i.e., max $\{ \| \pmb { w } _ { k } - \pmb { h } _ { i } ^ { ( k ) } \| _ { 2 } ^ { 2 } \} _ { i = 1 } ^ { n _ { k } } < t _ { k } ^ { \prime }$ . As $\| \pmb { w } - \pmb { h } \| _ { 2 } ^ { 2 } = 2 - 2 \pmb { w } ^ { \top } \pmb { h }$ in the normalized space, the above compactness condition is equivalent to defining the largest possible unified threshold $t _ { k } = 1 - t _ { k } ^ { \prime } / 2$ to impose a uniform lower bound on cosine similarities between each sample feature and the anchor,

$$
\operatorname* { m i n } \{ w _ { k } ^ { \top } h _ { i } ^ { ( k ) } \} _ { i = 1 } ^ { n _ { k } } > t _ { k } .\tag{7}
$$

Similarly, to explicitly measure the separability between the classes k and $j$ for $\forall j \ne k$ we take the j-th classifier vector ${ \pmb w } _ { j }$ as an anchor, and choose the largest possible unified threshold $t _ { j } ^ { \prime }$ such that the minimum distance from the feature vectors of class k to ${ \pmb w } _ { j }$ is no smaller than this threshold, i.e., min $\{ \| \pmb { w } _ { j } - \pmb { h } _ { i } ^ { ( k ) } \| _ { 2 } ^ { 2 } \} > t _ { j } ^ { \prime }$ . This separability condition is equivalent to defining the smallest possible unified threshold $t _ { j } = 1 - t _ { j } ^ { \prime } / 2$ to impose a uniform upper bound on similarities between each feature vector of class k and the anchor ${ \pmb w } _ { j }$

$$
\operatorname* { m a x } \{ \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } \} _ { i = 1 } ^ { n _ { k } } < t _ { j } , \quad \forall j \ne k .\tag{8}
$$

According to the above analysis, for class k, the desirable feature properties require that all positive unbiased decision scores $\{ \gamma \pmb { w } _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } \} _ { i = 1 } ^ { n _ { k } }$ uniformly exceed as large a threshold as possible, while all negative ones $\textstyle \bigcup _ { j = 1 \atop i \neq k } ^ { K } \{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } \} _ { i = 1 } ^ { n _ { k } }$ uniformly remain below as small a threshold as possible, which are sample-independent on dataset $\mathcal { D } _ { \ell }$

In the model training, set $b _ { k } ~ = ~ \gamma t _ { k }$ for $\forall k \in [ K ]$ . For $\forall X ^ { ( k ) }$ from class $k ,$ applying Sigmoid on its positive decision score $\gamma \tilde { \mathbf { { w } _ { k } } } \bar { h ^ { ( k ) } } - b _ { k }$ , one can compute the predicted probability that it satisfies the compactness condition in $\mathrm { E q . } \ ( 7 )$ $\hat { p } _ { k k } = \sigma ( \gamma w _ { k } ^ { \top } h ^ { ( k ) } - b _ { k } )$ ; similarly, applying Sigmoid on its negative decision scores $\{ \gamma { \pmb w } _ { j } { \pmb h } ^ { ( k ) } - \tilde { b } _ { j } \} _ { j \neq k } ^ { K }$ , one can compute the predicted probabilities that it satisfies the separability condition in Eq. (8), $\left\{ \hat { p } _ { k j } = 1 - \sigma ( \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } ^ { ( k ) } - b _ { j } ) : j \neq k \right\}$ . Clearly, for any reasonable threshold, the perfect feature vector always satisfies both the compactness and separability conditions; therefore, for the ground-truth labels used in feature learning, we set $p _ { k j } = 1$ with $\forall j \in [ K ]$ for the sample $X ^ { ( k ) }$ from class k. Then, for feature learning, one can calculate $K$ ordinary binary cross-entropy (BCE) losses for any sample $\smash { \boldsymbol { X } ^ { ( k ) } }$ and minimizes them together,

$$
\mathcal { L } _ { \mathrm { b c e } } ( X ^ { ( k ) } ) = - \sum _ { j = 1 } ^ { K } p _ { k j } \log \hat { p } _ { k j } = \log \big ( 1 + \mathrm { e } ^ { - \gamma w _ { k } ^ { \top } h ^ { ( k ) } + b _ { k } } \big ) + \sum _ { j = 1 \atop j = k } ^ { K } \log \big ( 1 + \mathrm { e } ^ { \gamma w _ { j } ^ { \top } h ^ { ( k ) } - b _ { j } } \big ) .\tag{9}
$$

This BCE loss can also be derived by decomposing a multi-class classification into multiple binary classification tasks, each of which checks whether a sample $X ^ { ( k ) }$ belongs to class $j$ for $\forall j \ \in \ [ K ]$ , which has been widely used in multi-label classification (Kobayashi, 2023). Here, we reformulate it based on the global constraints of feature learning, thereby revealing, for the first time, its connection to the global properties of sample features across the entire dataset.

## 4 Main Theoretical Results

The above derivations of CE and BCE highlight the fundamental diference between classification and feature learning. As classification performance reflects the model’s local behavior at each sample, the CE loss merely requires that, for each sample, the positive decision score be relatively higher than the negative ones. In contrast, feature learning requires the model to exhibit globally favorable properties across the entire dataset, and the classifier bias parameters in the BCE loss impose a global and absolute constraint on the feature vectors of all samples.

In this section, we will further reveal the diferences between the two tasks by analyzing the minima of CE and BCE, as well as their convergence rates to these minima in normalized spaces of diferent sizes.

## 4.1 Minima of Normalized BCE and CE

Following Z. Zhu et al. (2021) and Yaras et al. (2022), we simplify the analysis by using unconstrained feature model (UFM) on balanced dataset. Specifically, we take the sample features $\textstyle \bigcup _ { k = 1 } ^ { K } \{ h _ { i } ^ { ( k ) } \} _ { i = 1 } ^ { n _ { k } } ,$ classifier vectors $\{ \boldsymbol { w } _ { j } \} _ { j = 1 } ^ { K }$ , and classifier biases $\{ b _ { j } \} _ { j = 1 } ^ { K }$ as free variables, without considering the parameters within the feature extraction model , and assume that $n = n _ { k }$ , for $\forall k \in [ K ]$ . Let

$$
f _ { \mu } ( W , H , b ) = \frac { 1 } { n K } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \mathcal { L } _ { \mu } ( X _ { i } ^ { ( k ) } ) ,\tag{10}
$$

where $\mu \in \{ \mathrm { c e } , \mathrm { b c e } \}$ indicating the normalized CE and BCE losses in Eqs. (4) and (9), and $\pmb { H } = \left[ \pmb { H } _ { 1 } , \pmb { H } _ { 2 } , \cdot \cdot \cdot , \pmb { H } _ { K } \right] \in \mathbb { R } ^ { d \times ( n K ) }$ with $H _ { k } = \left[ \pmb { h } _ { 1 } ^ { ( k ) } , \pmb { h } _ { 2 } ^ { ( k ) } , \cdots , \pmb { h } _ { n } ^ { ( k ) } \right] \in \mathbb { R } ^ { d \times n }$ Then, we achieve the following theorems.

Theorem 1 Suppose d $\geq K - 1 , i . e . ,$ the feature dimension is greater than the number of classes. The loss function $f _ { \mathrm { c e } } ( W , H , b )$ defined using $\mathcal { L } _ { \mathrm { c e } }$ in Eq. (4) satisfies

$$
f _ { \mathrm { c e } } ( W , H , b ) \geq \log \Big ( 1 + ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma K } { K - 1 } } \Big ) ,\tag{11}
$$

and the equality is attained if and only if W, H and b satisfy

$$
{ \pmb w } _ { k } = { \pmb h } _ { i } ^ { ( k ) } , \forall k \in [ K ] , i \in [ n ] ,\tag{12}
$$

$$
\pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = - \frac { 1 } { K - 1 } , \forall j \neq k \in [ K ] ,\tag{13}
$$

$$
a n d b _ { k } = b ^ { \star } \in \mathbb { R } , \forall k \in [ K ] ,\tag{14}
$$

where $b ^ { \star } \in \mathbb { R }$ could be any number.

Proof The lower bound and its equality conditions are established in Theorem C.9 in Appendix. □

Theorem 2 Suppose $d \geq K - 1$ . The loss function $f _ { \mathrm { b c e } } ( W , H , b )$ defined using $\mathcal { L } _ { \mathrm { b c e } }$ in $E q .$ (9) satisfies

$$
\begin{array} { r } { f _ { \mathrm { b c e } } ( W , H , b ) \geq ~ - ~ 2 \gamma - ( K - 2 ) b ^ { \star } + \log \left( 1 + \mathrm { e } ^ { \gamma - b ^ { \star } } \right) + ( K - 1 ) \log \left( 1 + \mathrm { e } ^ { b ^ { \star } + \frac { \gamma } { K - 1 } } \right) } \end{array}
$$

$$
= \log \big ( 1 + \mathrm { e } ^ { - \gamma + b ^ { \star } } \big ) + ( K - 1 ) \log \Big ( 1 + \mathrm { e } ^ { - \frac { \gamma } { K - 1 } - b ^ { \star } } \Big ) ,\tag{15}
$$

and the equality holds if and only if W, H and b satisfy Eqs. (12, 13) and (14), where

$$
b ^ { \star } = - \frac { \gamma } { K - 1 } - \log 2 + \log \left[ ( K - 2 ) + \sqrt { ( K - 2 ) ^ { 2 } + 4 ( K - 1 ) } \mathrm { e } ^ { \gamma + \frac { \gamma } { K - 1 } } \right] .\tag{16}
$$

Proof The proof establishes a lower bound for the BCE loss and characterizes its equality conditions. The details can be found in Theorem C.4 in Appendix. □

Theorems 1 and 2 show that CE and BCE can respectively attain their constantvalued lower bounds over the entire dataset; hence the constant values are their minima. Furthermore, when the CE and BCE reach their minima, the sample features and classifier vectors satisfy the conditions of NC:

• (collapse) The features of samples within the same class converge to the class center. From Eq. (12), for $\forall k \in [ K ]$ , the features $\{ h _ { i } ^ { ( k ) } \} _ { i = 1 } ^ { n }$ converge to its class center, $\begin{array} { r } { \bar { \pmb { h } } _ { k } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \pmb { h } } _ { i } ^ { ( k ) } } \end{array}$

• (alignment) The class center of each class is parallel to its corresponding classifier vector, difering only by a constant coeficient. From Eq. (12), $\bar { h } _ { k } = w _ { k }$ , for $\forall k \in [ K ]$

• (equiangular) The angle between any two class center vectors is equal and maximized, forming an equiangular tight frame (ETF). From Eqs. (12, 13), $\begin{array} { r } { \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = \pmb { \bar { h } } _ { k } ^ { \top } \pmb { \bar { h } } _ { j } = - \frac { 1 } { K - 1 } , \forall j \neq \pmb { \bar { k } } \in [ K ] } \end{array}$ , which indicates $\{ \pmb { w } _ { k } \} _ { k = 1 } ^ { K }$ and $\{ \bar { h } _ { k } \} _ { k = 1 } ^ { K }$ form two perfectly overlapping ETFs.

NC indicates that, when reaching the minima of the CE or BCE, the compactness of any class is maximized with $\mathcal { E } _ { \mathrm { c o m } } = 1$ , and the separability of any two classes is also maximized with $\begin{array} { r } { \mathcal { E } _ { \mathrm { s e p } } = \frac { 1 } { 2 } + \frac { 1 } { 2 ( K - 1 ) } . } \end{array}$

The classifier biases. At their minimum points, the normalized CE and BCE share the same geometric structures of sample feature and classifier vectors. Their diference lies in the classifier biases. In designing the loss functions, the classifier biases of the normalized CE are merely ofsets of the decision scores, whereas in the normalized BCE the classifier biases are derived from the thresholds of the decision scores.

When the losses reach their minimum points, the classifier bias vector $\pmb { b } \in \mathbb { R } ^ { K }$ become multiples of the all-one vector $\mathbf { 1 } _ { K } \in \mathbb { R } ^ { K }$ . For the normalized CE loss, the multiple factor $b ^ { \star } \in \mathbb { R }$ can be any number. As long as $b _ { k } = b _ { j } , \forall k \neq j ,$ they do not afect the classification results, and they can be canceled out in the Softmax of the CE loss. Therefore, in the CE used for classification tasks, the K classifier biases can take any identical value when the loss achieving minimum value. This analysis also implies that, when training models with CE, the classifier biases cannot meaningfully contribute to the unbiased decision scores that reflect the feature properties. In previous works (Lu & Steinerberger, 2022; Yaras et al., 2022), the authors ignore the classifier biases in the CE when analyzing its minimum value.

In contrast, for the normalized BCE, the final value of classifier biases, $b ^ { \star }$ , is figured out via Eq. (16), indicating the BCE loss has only one minimum point in terms of the classifier biases. On the contrary, the unique converged value of $b ^ { \star }$ indicates that, during BCE minimization, the classifier biases play a substantive role by efectively contributing to the update of decision scores that reflect the feature properties, thereby accelerating loss convergence and improving training eficiency. Furthermore, with a fixed class number $K$ , the curve of $b ^ { \star }$ in terms of $\gamma$ defined in Eq. (16) has an

asymptote, i.e.,

$$
b ^ { \star } \to \frac { 1 } { 2 } \log ( K - 1 ) + \frac { K - 2 } { K - 1 } \frac { \gamma } { 2 } ,\tag{17}
$$

indicating that the final bias $b ^ { \star }$ increases almost linearly as $\gamma$ approaches positive infinity.

## 4.2 Convergence of Normalized CE and BCE

When the classifier biases are equal, i.e., $b _ { k } = b _ { j } , \forall k , j \in [ K ]$ , the unbiased decision scores $\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } \} _ { j , k , i }$ not only determine each sample’s classification but also reflect the feature properties. According to Theorems 1 and $^ { 2 , }$ at the minimum points of normalized CE or BCE, the positive and negative unbiased decision scores will converge two fixed values related to the radius $\gamma$ of normalized space,

$$
\gamma { \pmb w } _ { k } ^ { \top } { \pmb h } _ { i } ^ { ( k ) }  \gamma , \forall k \in [ K ] , \forall i \in [ n ] ,\tag{18}
$$

$$
\gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) }  - \frac { \gamma } { K - 1 } , \forall k \neq j , \forall i \in [ n ] .\tag{19}
$$

Clearly, as $\gamma$ increases, both the positive and negative decision scores at the minima increase linearly. Then, we can analyze the convergence of the losses by analyzing the convergence behavior of the unbiased decision scores, which will help to compare classification and feature learning in the normalized spaces with diferent scale factors. Before these analysis, we first define two critical conditions (CC) for the unbiased decision scores.

Definition 3 CC I states that for every sample in dataset ${ \mathcal { D } } ,$ its positive unbiased decision score exceeds the negative ones, i.e., for $\forall k \in [ K ] , \ i \in [ n ]$ ，

$$
\gamma w _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } > \operatorname* { m a x } \big \{ \gamma w _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } : j \in [ K ] , \ j \neq k \big \} .\tag{20}
$$

Definition 4 CC II states that for all the samples of D, the positive unbiased decision scores are greater than zero and the negative ones are less than zero, i.e., for $\forall k \in [ K ] , \ i \in [ n ]$ ，

$$
\gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } > 0 > \operatorname* { m a x } \big \{ \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } : j \in [ K ] , \ j \neq k \big \} .\tag{21}
$$

When the classifier biases are equal, CC I is equivalent to all the samples being correctly classified, while CC II not only implies that the classification accuracy is 100%, but also that their positive and negative unbiased decision scores are uniformly bounded by the threshold $t = 0$ , indicating better intra-class compactness and interclass separability of sample features.

Theorem 5 (1) When training models using the normalized CE loss defined in $E q . \ ( 4 )$ , as γ approaches zero, its linear decrease leads to a linear decay in the convergence rate $o f$ unbiased decision scores.

(2) Once the CC I is satisfied, as γ linearly approaches positive infinity, the convergence rate of unbiased decision scores decay exponentially.

Proof The result follows from the gradient analysis and asymptotic estimates given in Theorem C.16 in Appendix. □

Theorem 6 (1) When training models using the BCE loss defined in Eq. (9), as γ approaches zero, its linear decrease leads to a linear decay in the convergence rate of unbiased decision scores.

(2) In contrary, once the CC II is satisfied, as γ linearly approaches positive infinity, the convergence rate of unbiased decision scores decay exponentially.

Proof The result follows from the gradient analysis and asymptotic estimates given in Theorem C.15 in Appendix. □

Using the unbiased decision scores, Theorems 5 and 6 reveal the convergence rates of the CE and BCE in normalized spaces of varying sizes: as the space radius γ decreases linearly toward zero, the convergence rates of CE and BCE decay linearly; whereas once CCs I and II are satisfied, respectively, their convergence rates decay exponentially. Meanwhile, based on these two theorems, one can also analyze the differences between classification and feature learning across normalized spaces of varying sizes.

(1) When the radius of normalized space is very small, for example $0 < \gamma < 1$ , it is not suitable for feature learning, and even less suitable for classification. As γ decreases linearly, the theoretical margin between the positive and negative decision scores of each sample also decreases linearly. In this case, any small perturbation during training or even testing can easily change their relative magnitudes, thereby significantly afecting the classification results; however, this does not necessarily mean that the learned feature properties are also extremely poor. Nevertheless, since the convergence rate of the loss function decreases linearly as γ decreases, the number of training epochs required for the feature properties to reach their theoretical maximum increases linearly. Therefore, under the same training strategy, the smaller the value of γ, the poorer the learned feature properties will be.

(2) When the radius of normalized space is very large, it is beneficial for classification but detrimental to feature learning. When γ is very large, the theoretical margin between the positive and negative decision scores of each sample also becomes very large. With training, it is then easy to satisfy the relative ordering required for correct classification, and the larger γ is, the easier this becomes. However, as γ increases linearly, the convergence rate of the loss function decays exponentially, which means that the number of training epochs required for the feature properties to reach their theoretical optimum grows exponentially. Therefore, under the same training strategy, increasing the spatial size does not afect classification accuracy, but it significantly degrades the quality of the learned features.

(3) The model’s classification performance is misaligned with the properties of the sample features. With a fixed training strategy, the classification accuracy of the trained model increases monotonically with the radius γ of the normalized space, while the final feature qualities including $\mathcal { E } _ { \mathrm { c o m } }$ and $\mathcal { E } _ { \mathrm { s e p } }$ first increase and then decrease as γ grows. Therefore, a model’s classification accuracy cannot be used to reliably indicate its performance on downstream tasks that require feature comparison, such as face recognition, image retrieval, person re-identification, and object tracking, etc.

(4) BCE outperforms CE in larger normalized spaces. Theorems 5 and 6 stem from the fact that, once the decision scores of the samples satisfy CCs I and II, respectively, their contributions to the convergence rates of CE and BCE decay exponentially. Therefore, in a larger space, once a sample is correctly classified and thus satisfies CC I, its contribution to the convergence of CE becomes very small. At this point, however, its contribution to the convergence of BCE may still remain relatively large, and only when its decision score further satisfies CC II will its contribution to the convergence of BCE also become very small. This analysis implies that, in larger spaces, BCE can converge faster than CE. Consequently, using the same training strategy, BCE can train the model into a more favorable state than CE, thereby achieving better classification accuracy and feature properties.

Through the above analysis, we highlight the diferences between classification and feature learning and depict the variation in classification performance and feature properties with space size; meanwhile, we demonstrate that BCE performs better than CE in larger spaces. In the following, we will empirically validate these conclusions.

## 5 Experiments

To validate the conclusions about classification and feature learning as well as the diferences between the CE and BCE, we trained CNN (He et al., 2016) and Transformer (Dosovitskiy et al., 2021b) on MNIST (Lecun et al., 1998), CIFAR10, and CIFAR100 (Krizhevsky, 2009).

## 5.1 Minima of the normalized CE and BCE

We first verify that the normalized CE and BCE losses can lead to the NC geometry predicted by Theorems 1 and 2. We trained ResNet18, ResNet50, DenseNet121, and ViT on MNIST, CIFAR10, and CIFAR100 using SGD and AdamW optimizers when the scale factor $\gamma = 8$ . We applied the metrics of $\mathcal { N C } _ { 1 } , \mathcal { N C } _ { 2 }$ , and $\mathcal { N C } _ { 3 }$ defined in Z. Zhu et al. (2021) and W. Liu, Yu, et al. (2023) to measure the evolution of NC during the training. The smaller values indicate the stronger NC properties.

Fig. 2 shows the evolution of the three NC metrics on the training data of CIFAR10. For diferent models and optimizers, both the CE and BCE steadily reduce the NC metrics during the training, which verifies that the two normalized losses can lead to NC when approaching their global minima. The results are consistent with our theorems, where samples from the same class collapse to their class center, class centers align with their corresponding classifier vectors, and the diferent class centers form an ETF structure.

![](images/fa5de17d9d8b7dad241c583e489b6b0857879c38a47498ee1168420b268f3308.jpg)

![](images/795f55cfe7e8bf3eed40495ee163d05877a5f6c9a2cb284f7c2013669e8ed35b.jpg)  
(a) N C<sub>1</sub>, $\ N C _ { 2 } ,$ and $\mathcal { N } \mathcal { C } _ { 3 }$ of ResNet18 on CIFAR10

![](images/b0d9544969c14c7e0606044540fe0da4f50bbb008478ea26c8218c3229f1d6ab.jpg)

![](images/6be46e3e47d42b61e5dbbbb7dc09415331140cf066c7a183a9bd23c0f215742e.jpg)

![](images/4e49274fd5b399285e8bab99c4db8c40c471c66ed4e08df1f9769d7caf48f806.jpg)  
(b) $\small { \mathcal { N C } } _ { 1 } , { \mathcal { N C } } _ { 2 } ,$ , and $\mathcal { N } \mathcal { C } _ { 3 }$ of ResNet50 on CIFAR10

![](images/e2286e53755345bac032a26c919fc0d3e7e1652dac15d624df6b692d2d278da8.jpg)

![](images/e493578884539f2a56b155ce395c58d97bdcadb72c7fa879637c64d63153f9d4.jpg)

![](images/a6377ac8350f55ed55262e1ce3f7d53291ec03535b03d629614c6d85ff90a32b.jpg)

![](images/7e53499291b8e0cb748320da8d990f3a97038fdd978a4ccda1581ef5bc4f3ec0.jpg)  
(c) NC , NC , and $\mathcal { N } \mathcal { C } _ { 3 }$ of DenseNet121 on CIFAR10

![](images/72e205286524b3e55e95d47bbe35feaec01c0547469d1defb741d4d649f09ceb.jpg)

![](images/68bc4ed8a870acfd0b597310210cab1a58e250021990c7b8865fea84b5ecece6.jpg)

![](images/20eb7213412a1cbac2ff68337925ee8fce5b9b295ac719ee7857db3b284d8fa7.jpg)  
(d) ${ \mathcal { N } } { \mathcal { C } } _ { 1 }$ ${ \mathcal { N C } } _ { 2 } ,$ and $\mathcal { N } \mathcal { C } _ { 3 }$ of ViT on CIFAR10  
Fig. 2: The evolution of the three NC metrics in the training of ResNet18, ResNet50, DenseNet121, and ViT on the training data of CIFAR10 driven by the normalized CE and BCE losses with $\gamma = 8$ using SGD and AdamW, respectively

Although the sample features deduced by the CE and BCE share the same geometric structure at their respective minima, the BCE typically reaches this structure faster during the early training. As Figure 2 shows, within the first 50 epochs, the NC metrics for the BCE are generally lower than those for the CE, suggesting that the BCE more eficiently pushes the feature distribution toward the global NC structure, rather than only improving sample-level classification. This advantage of BCE comes from its explicit constraints on the positive and negative decision scores of samples. Compared with CE, which mainly relies on the relative ordering of the decision scores for each sample, BCE uses the classifier biases as the thresholds to uniformly distinguish the positive and negative decision scores across the entire dataset.

Similar results can be observed on MNIST and CIFAR100, as reported in Sec. B.2 in Appendix. Overall, these experiments support the minima analysis in the normalized spaces. Both the CE and BCE can induce NC, while the BCE tends to reach the desired compact and separable feature geometry more rapidly.

## 5.2 Convergence of the normalized CE and BCE

Theorems 5 and 6 describe the convergence behaviors of the CE and BCE. To directly verify these two theorems, we set the NC structure as the stopping criterion, and train ResNet18 on MNIST with various scale factor γ. In these experiments, the stopping criterion is set as

$$
\begin{array} { r } { \lvert \boldsymbol { w } _ { k } ^ { \top } \boldsymbol { h } _ { i } ^ { ( k ) } - 1 \rvert \leq t , \forall k \in [ K ] , \forall i \in [ n ] , } \end{array}\tag{22}
$$

$$
\mathrm { a n d } \quad \bigg | \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } + \frac { 1 } { K - 1 } \bigg | \leq t , \forall j \neq k \in [ K ] ,\tag{23}
$$

where for $\gamma < 5 , t = 0 . 0 0 1$ , and for $\gamma \ge 5 , t = 0 . 0 0 3 5$ . Moreover, only SGD optimizer was applied and the learning rate was fixed at 0.01. Table 1 shows the epoch numbers required by the models trained with the CE and BCE losses when they satisfying the stopping criterion for $\gamma > 1$

For the cases with $\gamma ~ < ~ 0 . 5$ and $\gamma > 5$ , we took six and five data points to respectively fit the dependence of epoch number on γ using linear, quadratic, and cubic polynomial functions, as well as exponential and logarithmic functions, while the remaining data points were used to evaluate the prediction quality of the fitted functions. For $\gamma > 5$ , Table 2 shows the fitting results, together with the metrics of $R ^ { 2 }$ and RMSE on the fitting and prediction data. It can be observed that for the CE, although the exponential function exhibits weaker performance metrics than the quadratic and cubic polynomial functions on the fitting data, it better reflects the varying trend of epoch number on the prediction data. For the BCE, the exponential function achieves the best fitting results among the five types of functions, both on the fitting and prediction data. These results indicate that when $\gamma$ is relatively large, as it increases, the number of epochs required for the CE and BCE convergence grows exponentially, meaning their convergence rates decay exponentially and aligning with Theorems 5 and 6.

Moreover, in the two fitted exponential functions, the coeficient (0.52) of BCE is less than that (0.87) of CE, indicating that the BCE converges faster than the CE in the large feature spaces with larger scale factor γ. The results for the case with $\gamma < 0 . 5$ are presented in Sec. B.7 of Appendix, which also align with the theorems.

## 5.3 Classification accuracy vs. feature property

For a model, its classification and feature learning perform diferently in the normalized space with diferent scale factor γ, as we analyzed in Sec. 4.2. To verify this diference, using the CE and BCE with SGD and AdamW, we trained 56 ResNet18s and 56 ViTs on CIFAR10 with 200 epochs by setting γ to 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32, and 64, respectively. The detailed experimental settings are shown in Table B.1 in Appendix. Fig. 3 shows the distributions of the final unbiased decision scores and classifier biases on the training data, while the red, blue, and green solid lines represent the theoretical values of the final positive/negative unbiased decision scores and classifier biases for diferent γ.

Table 1: The number of training epochs required by ResNet18 to reach the stopping criterion of Eqs. (22) and (23) on MNIST trained by the normalized CE and BCE with diferent $\gamma > 1 . \ ^ { \ast } \ast ^ { \ast }$ indicating that the data is used for fitting the functions of epoch number on γ. When $\gamma \geq 1 0$ , training ResNet18 with the CE loss requires more than 4, 000 epochs to converge, so the experiment was not performed
<table><tr><td>γ</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6*</td><td>6.5</td><td>7*</td><td>7.5</td><td>8*</td><td>8.5</td><td>9*</td><td>9.5*</td><td>10*</td><td>11</td><td>12*</td><td>13</td><td>14*</td><td>15</td><td> $1 6 ^ { * }$ </td></tr><tr><td>CE</td><td>28</td><td>31</td><td>64</td><td>31</td><td>77</td><td>145</td><td>180</td><td>308</td><td>435</td><td>741</td><td>1448</td><td>2083</td><td>1</td><td>一</td><td>1</td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>BCE</td><td>24</td><td>22</td><td>22</td><td>14</td><td>15</td><td></td><td>21</td><td>一</td><td>37</td><td>1</td><td>53</td><td>一</td><td>83</td><td>137</td><td>243</td><td>411</td><td>737</td><td></td><td>1165 2076</td></tr></table>

Table 2: Fitted functions and their evaluation results for the epoch number used by the CE and BCE with $\gamma > 5$
<table><tr><td rowspan="2">Loss</td><td rowspan="2">Fitted function</td><td colspan="2">Fitting data</td><td colspan="2">Predicted data</td></tr><tr><td> $R ^ { 2 } \uparrow$ </td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td><td>RMSE↓</td></tr><tr><td rowspan="5">CE</td><td>568.39γ - 3645.68</td><td>0.8556</td><td>299.08</td><td>-0.5945</td><td>317.58</td></tr><tr><td> $2 3 6 . 0 1 \gamma ^ { 2 } - 3 0 9 8 . 3 9 \gamma + 1 0 2 0 5 . 4 9$ </td><td>0.9939</td><td>61.59</td><td>0.7463</td><td>126.69</td></tr><tr><td> $4 0 . 6 5 \gamma ^ { 3 } - 7 0 6 . 2 8 \gamma ^ { 2 } + 4 0 7 6 . 8 0 \gamma - 7 7 2 9 . 4 6$ </td><td>0.9969</td><td>43.85</td><td>0.9038</td><td>77.99</td></tr><tr><td> $0 . 5 4 \exp ( 0 . 8 7 \gamma ) - 6 1 . 8 3$ </td><td>0.9932</td><td>64.98</td><td>0.9253</td><td>68.72</td></tr><tr><td> $4 . 2 2 \log ( 0 . 3 2 \gamma ) - 3 . 0 1$ </td><td>0.8094</td><td>343.40</td><td>-0.9326</td><td>349.63</td></tr><tr><td rowspan="5">BCE</td><td> $\overline { { 2 3 6 . 6 0 \gamma - 2 2 0 4 . 0 0 } }$ </td><td>0.7714</td><td>364.33</td><td>0.5566</td><td>849.72</td></tr><tr><td> $5 2 . 1 4 \gamma ^ { 2 } - 1 0 1 4 . 8 3 \gamma + 4 8 8 7 . 4 3$ </td><td>0.9812</td><td>104.54</td><td>0.8140</td><td>550.31</td></tr><tr><td> $7 . 6 1 \gamma ^ { 3 } - 2 2 1 . 9 8 \gamma ^ { 2 } + 2 1 7 1 . 1 1 \gamma - 7 0 2 7 . 8 7$ </td><td>0.9996</td><td>15.55</td><td>0.8837</td><td>435.17</td></tr><tr><td> $0 . 4 9 \exp ( 0 . 5 2 \gamma ) - 3 . 7 9$ </td><td>0.9999</td><td>7.56</td><td>0.9128</td><td>376.77</td></tr><tr><td> $2 . 5 8 \log ( 0 . 2 5 \gamma ) - 2 . 1 1$ </td><td>0.6871</td><td>426.22</td><td>0.4633</td><td>934.77</td></tr></table>

As Fig. 3 shows, when $\gamma \le 2$ , although the mean of all positive/negative unbiased decision scores eventually aligns with their theoretical final values, their variance is relative large, particularly for the negative unbiased scores of ViT, which indicates the losses has not fully converged to their minima with 200 epochs. When $3 \leq \gamma \leq 8 .$ all the final unbiased decision scores fall along their theoretical extreme values on the corresponding real lines, as well as the final classifier biases for the BCE, which indicates that the losses has basically converged at this stage. As γ continues to increase to 12 or more, the unbiased decision scores of the trained models gradually deviate from their theoretical extremum, and they are distributed over an increasingly larger areas, particularly for models trained with the CE, with an even broader distribution, which indicates that the losses have totally not converged.

![](images/f00c2ab95002ba17f0d12e24a92088321d38220e91aa62b7d1fffe3551240b25.jpg)

![](images/874983774706cc00b8675f566d69883b65187427ddb269009f475d13d36b948a.jpg)

![](images/a141474c0fab3855b4b4464acfb9c5b5d0933a6945894bc24b0f38fe0d30821d.jpg)

![](images/06a603fb5f4626bd64b9c0516ad8233150746080b6cbe2ae4029d4babfa36dc2.jpg)

(a) ResNet18 trained on CIFAR10  
![](images/bdb0a2d7c972ba475bde762d1733eb7678834f74ad06acdf38655afe0912be57.jpg)

![](images/5665526556c7547791911bf2c1c01eb78b060b85250cfbbbbd00c509923e3a7b.jpg)

![](images/11d7c239303f60278ba4b2982a286b5f67c6b615c02b4a0b05b9ad1852d5d947.jpg)

![](images/fc105f90f99a2df6cef549e88b9df847486900515e7bd51cd2056cf1c3bd18ac.jpg)  
(b) ViT trained on CIFAR10  
Fig. 3: The distribution of the final classifier biases and the positive/negative unbiased decision scores for 56 ResNet18s (top, without the final ReLU activation) and 56 ViTs (bottom) trained on CIFAR10, with various scale factors $\gamma .$ The maximum number of training epochs is fixed at 200. The initial mean $\begin{array} { r } { ( \bar { b } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } b _ { j } ) } \end{array}$ of the biases is 0 for ResNet18 and 5 for ViT, and the initial variance is both 0. The solid line represents the theoretical extremum when achieving NC

Since the substantial role of classifier biases in the BCE during the model training, when $\gamma \leq 8 ,$ they eventually converges to their theoretical extremum. In contrast, the final biases of the CE are almost entirely dependent on their initial value, i.e. 0 for ResNet18 and 5 for ViT in the experiments.

Table 3 presents part of results for ResNet18 trained by CE and BCE with small and large γ on the training and test data of CIFAR10, and more results can be found in Sec. B.5 in Appendix. The expressions of accuracy without the bias, $\ b { A } ^ { * }$ , are presented in Sec. A.2 in Appendix. When $\gamma = 0 . 1$ or 0.5, both the accuracy and the unbiased one $\ b { A } ^ { * }$ are very low, but they difer clearly, indicating the slight variance in the classifier biases significantly afect the classification. However, with $\gamma = 0 . 1$ or 0.5, the final feature properties are not so bad, as the compactness $\mathcal { E } _ { \mathrm { c o m } }$ and separability $\mathcal { E } _ { \mathrm { s e p } }$ are high, although they have not yet reached their maximum. In contrast, when $\gamma = 3 2$ or 64, the models’ accuracies are very high reaching 100% on the training data, while the intra-class compactness and inter-class separability of their features are comparatively poor and worse than that learned with small γ. As Table 3 shows, a similar phenomenon is also presented in the evaluation results on the test dataset.

The above results are entirely consistent with the analytical conclusions in this paper for classification and feature learning: with a fixed training strategy in practice, classification performs well in a relatively large feature space, whereas in a very small feature space it is more susceptible to interference from random factors, leading to a marked reduction in the final accuracy. In contrast, feature learning that pursues intraclass compactness and inter-class separability requires an appropriately smaller feature space. In summary, a model’s classification accuracy cannot reflect its performance in downstream tasks that require feature comparison.

Table 3: The classification accuracy (%) and feature properties of ResNet18 (without the final ReLU) with diferent γ on CIFAR10. The accuracies of  and $\ b { A } ^ { * }$ are calculated using the decision scores $\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } ^ { ( k ) } - b _ { j } \} _ { j = 1 } ^ { K }$ and the unbiased ones $\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } ^ { ( k ) } \} _ { j = 1 } ^ { K }$ , respectively. The higher $\mathcal { E } _ { \mathrm { c o m } }$ and $\mathcal { E } _ { \mathrm { s e p } }$ respectively indicate the better compactness and separability of features  
(a) Results on the training data of CIFAR10 — CE with SGD.
<table><tr><td colspan="8">J</td></tr><tr><td>γ  $\boldsymbol { A }$ </td><td>0.1 26.24</td><td>0.5 54.28</td><td>1 79.50</td><td> $\overline { { 2 } }$  99.97</td><td>16 100.0</td><td>32 100.0</td><td>64 100.0</td></tr><tr><td> $A ^ { * }$ </td><td>27.27</td><td>57.00</td><td>80.01</td><td>99.97</td><td>100.0</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>100.0</td><td>100.0</td></tr><tr><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td>0.965</td><td>0.995</td><td>0.998</td><td>0.999</td><td>0.862</td><td>0.789</td><td>0.766</td></tr><tr><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>0.526</td><td>0.554</td><td>0.555</td><td>0.555</td><td>0.394</td><td>0.288</td><td>0.221</td></tr><tr><td colspan="8">(a) Results on the training data of CIFAR10 一 BCE with SGD.</td></tr><tr><td> $\gamma$ </td><td>0.1</td><td>0.5</td><td>1</td><td>2</td><td>16</td><td>32</td><td>64</td></tr><tr><td> $\boldsymbol { A }$ </td><td>25.47</td><td>52.07</td><td>76.73</td><td>93.79</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td> $A ^ { * }$ </td><td>27.77</td><td>54.75</td><td>77.76</td><td>93.78</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td>0.965</td><td>0.995</td><td>0.997</td><td>0.999</td><td>0.973</td><td>0.899</td><td>0.842</td></tr><tr><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>0.526</td><td>0.554</td><td>0.555</td><td>0.553</td><td>0.410</td><td>0.354</td><td>0.313</td></tr><tr><td colspan="8">(b) Results on the test data of CIFAR10 CE with SGD.</td></tr><tr><td>γ  $\boldsymbol { A }$ </td><td>0.1 25.81</td><td>0.5 44.18</td><td>1</td><td>2 78.95</td><td>16 78.91</td><td>32 77.84</td><td>64 77.46</td></tr><tr><td> $A ^ { * }$ </td><td>25.62</td><td>46.94</td><td>61.54 61.60</td><td>78.94</td><td>78.91</td><td>77.79</td><td>77.44</td></tr><tr><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td>0.889</td><td>0.829</td><td>0.824</td><td>0.821</td><td>0.725</td><td>0.691</td><td>0.685</td></tr><tr><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>0.515</td><td>0.535</td><td>0.535</td><td>0.535</td><td>0.361</td><td>0.266</td><td>0.204</td></tr><tr><td colspan="8">(b) Results on the test data of CIFAR10 — BCE with SGD.</td></tr><tr><td> $\gamma$   $\boldsymbol { A }$ </td><td>0.1 24.98</td><td>0.5 44.12</td><td>1 58.41</td><td>2 76.76</td><td>16 82.14</td><td>32 81.69</td><td>64 82.54</td></tr><tr><td> $A ^ { * }$ </td><td>26.55</td><td>45.67</td><td>59.71</td><td>76.75</td><td>82.12</td><td>81.72</td><td>82.56</td></tr><tr><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td>0.889</td><td>0.824</td><td>0.825</td><td>0.837</td><td>0.807</td><td>0.749</td><td>0.726</td></tr><tr><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>0.514</td><td>0.535</td><td>0.536</td><td>0.534</td><td>0.368</td><td>0.318</td><td>0.293</td></tr></table>

## 5.4 Performance of BCE and CE on LTR and OSR

The applications of the BCE in the face recognition have been explored by Wen et al. (2022) and Zhou et al. (2023), and we here explore its advantages over the CE on other more challenging tasks such as long-tailed recognition (LTR) and open-set recognition (OSR). Based on practical experience, we set $\gamma = 3 2$ in these two tasks.

LTR. On the imbalanced datasets, CIFAT10-LT and $\mathrm { C I F A R 1 0 0 – L T }$ , when the imbalance factors (IF) of the training sets are 10, 50, and 100, we trained ResNet32 using the normalized CE and BCE with SGD, respectively. Table 4 shows the classification results on their balanced test sets. The BCE consistently achieves better LTR results on the six pairs of models. We believe that, compared to the CE, which couples the K decision scores of each sample into one Softmax, the BCE decouples them using K Sigmoids, mitigating the imbalance efects caused by the imbalanced datasets and improving the LTR performance.

Table 4: LTR results (%) on CIFAR10- LT and CIFAR100-LT
<table><tr><td rowspan="2">IF</td><td colspan="3">CIFAR10</td><td colspan="3">CIFAR100</td></tr><tr><td>10</td><td>50</td><td>100</td><td>10</td><td>50</td><td>100</td></tr><tr><td>CE</td><td>93.68</td><td>87.80</td><td>83.37</td><td>69.30</td><td>55.15</td><td>49.47</td></tr><tr><td>BCE</td><td>93.96</td><td>88.75</td><td>84.47</td><td>69.49</td><td>58.53</td><td>52.15</td></tr></table>

Table 5: The OSR results under diferent setups
<table><tr><td rowspan="2"></td><td colspan="2">MNIST</td><td colspan="2">SVHN</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR+50</td></tr><tr><td>CE</td><td>BCE</td><td>CE</td><td>BCE</td><td>CE</td><td>BCE</td><td>CE</td><td>BCE</td></tr><tr><td>AUROC</td><td>98.6 99.2</td><td></td><td>94.3</td><td>95.1</td><td>84.2</td><td>85.8</td><td>87.8</td><td>90.2</td></tr><tr><td>OSCR</td><td>98.5</td><td>99.0</td><td>92.4</td><td>93.4</td><td>81.9</td><td>83.7</td><td>85.7</td><td>88.5</td></tr></table>

OSR. In the open-set experiments, we evaluate the performances of the CE and BCE on the MNIST, SVHN, CIFAR10 and CIFAR+50 datasets, and the models and training details follow APRL (G. Chen et al., 2022). Table 5 presents the OSR results in terms of the area under the receiver operating characteristic (AUROC) and the open set classification rate (OSCR). On these OSR experiments, the BCE consistently achieves higher performance across the two metrics. We believe that in the BCE, the uniform and explicit constraints of the classifier biases on the positive/negative decision scores are beneficial for learning a clear decision boundary for sample features, which in turn aids the models in discriminating unknown classes and enhances the performance in the OSR.

## 6 Discussion, Limitation, and Future Work

Weight decay factor. In Euclidean space without normalization on sample features or classifier vectors, $L _ { 2 }$ regularization with weight decay factor λ are typically added on the CE and BCE losses, which can also constrain the features into a bounded space. Intuitively, a small λ results in a large feature space, while a large λ leads to a small one. In Q. Li et al. (2025), we have compared the CE and BCE in Euclidean space but did not reveal the diference of classification and feature learning. We conjecture that, similar to the scale factor $\gamma$ in normalized space, when $\lambda \geq 0$ is very small, it is dificult for the losses to reach their minima, resulting in poor feature properties; when λ is very large, the small theoretical gap between the positive and negative decision scores will harm the classification. However, in Euclidean space, it is not easy to rigorously analyze the losses’ convergence behavior.

Temperature coeficient. In contrastive learning (CL), a temperature $\tau$ is applied on the denominator of the feature cosine similarity. In CL, the Softmax-based losses are typically applied, similar to the CE loss in classification. When the CL losses reach their minima, the scaled feature similarity $^ { \underline { { 1 } } } \langle h , h _ { * } \rangle$ of any positive or negative sample pair converge to fixed values, according to NC studies in Graf et al. (2021) and Koromilas et al. (2024). We here conjecture that the impact of the temperature τ on the convergence rate of Softmax-based losses parallels that of the scale factor γ on the convergence rate of CE loss in classification, i.e., too large or too small τ would result in slow convergence of the CL losses and thereby poor feature properties.

Minima of losses within more challenging scenarios. In more challenging scenarios, such as low-dimensional spaces where the feature dimension is smaller than the number of classes $( \mathrm { i . e . , } \ d < K )$ and in class-imbalanced settings, theoretically analyzing the minima and convergence of the losses becomes more dificult. Currently, we are investigating the minima of the BCE loss on imbalanced datasets.

Feature learning in other scenarios. In this paper, within the normalized, supervised, closed-set, and high-dimensional feature spaces, we analyze the convergence behavior of the CE and BCE, to compare classification and feature learning. When the setting changes, the convergence behavior of the losses may also change, thereby altering the conclusions regarding the two tasks. However, we conjecture that in both supervised and self-supervised contrastive learning (CL), as scale factor γ linearly increases, the convergence rate of the existing Softmax-based losses decays exponentially, then, too large feature spaces are less favorable for the CL.

## 7 Conclusions

This paper conducts an in-depth comparison for classification and feature learning in normalized space, by theoretically analyzing the minima and convergence of CE and BCE losses. While the CE has been widely used for classification, the BCE is specially designed from the first principle of feature learning for the first time. Although both the CE and BCE can lead to neural collapse at their global minima, the former constrains the relative values of the positive and negative decision scores for samples through probabilistic constraints, whereas the latter imposes uniform thresholds on the absolute values of the positive and negative decision scores for all samples. By analyzing the convergence behavior of the CE and BCE, we find that the scale factor γ, i.e., the size of the normalized space, can significantly afect classification and feature learning of a model trained by fixed strategy. A larger γ allows the trained model to achieve higher classification accuracy even without well-structured feature geometry, while a small γ makes classification unstable, yet may yield samples with higher intraclass compactness and inter-class separability in feature space. Experiments on CNNs and ViTs support our conclusions. Furthermore, as the BCE could obtain better features for samples, it outperforms the CE in more challenging tasks such as long-tailed and open-set recognitions.

## Appendix A Neural Collapse and Feature Properties

## A.1 Neural Collapse

Neural collapse (NC) was first found in Papyan et al. (2020) that occurs in the terminal training phase of the classification model, manifesting as elegant geometric structures in the sample features and classifier vectors. With NC, the geometric structure of sample features and classifier vectors manifests as follows:

• For any class k, all the sample features $\left\{ h _ { i } ^ { \left( k \right) } \right\} _ { i = 1 } ^ { n _ { k } }$ of this class converge to the feature center $\begin{array} { r } { \bar { \boldsymbol { h } } _ { k } \ = \ \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } \boldsymbol { h } _ { i } ^ { ( k ) } } \end{array}$ , i.e., $h _ { i } ^ { ( k ) } \ = \ h _ { j } ^ { ( k ) }$ , for $\forall i , j \ \in \ [ n _ { k } ] \ =$ $\{ 1 , 2 , \cdots , n _ { k } \}$ ;

• The feature center vectors of any two classes have the same angular distance and equal magnitudes, i.e., $\| \bar { \boldsymbol h } _ { k } \| = \| \bar { \boldsymbol h } _ { \ell } \|$ and $\begin{array} { r } { \frac { \langle \bar { h } _ { k } , \bar { h } _ { \ell } \rangle } { \| \bar { h } _ { k } \| \| \bar { h } _ { \ell } \| } = \frac { \langle \bar { h } _ { k ^ { \prime } } , \bar { h } _ { \ell ^ { \prime } } \rangle } { \| \bar { h } _ { k ^ { \prime } } \| \| \bar { h } _ { \ell ^ { \prime } } \| } } \end{array}$ , for $\forall k \neq \ell , k ^ { \prime } \neq$ $\ell ^ { \prime } \in [ K ]$ , forming an equiangular tight frame (ETF);

• The feature center vector of any class is parallel to its corresponding classifier vector with a fixed constant between them, i.e., $\exists \alpha > 0$ , such that $\bar { \pmb { h } } _ { k } = \alpha \pmb { w } _ { k }$ , for $\forall k \in [ K ]$

In Z. Zhu et al. (2021) and Zhou et al. (2022), the authors defined metrics of $\mathcal { N C } _ { 1 }$ $\mathcal { N C } _ { 2 } , \mathcal { N C } _ { 3 }$ to evaluate the NC properties.

$$
\displaystyle N \mathcal { C } _ { 1 } : = \frac { 1 } { K } \operatorname { t r a c e } \left( \Sigma _ { W } \Sigma _ { B } ^ { \dag } \right) ,\tag{A.1}
$$

$$
\mathcal { N C } _ { 2 } : = \bigg \| \frac { \tilde { W } \tilde { W } ^ { \top } } { \| \tilde { W } \tilde { W } ^ { \top } \| _ { F } } - \frac { 1 } { \sqrt { K - 1 } } ( I _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } ) \bigg \| _ { F } ,\tag{A.2}
$$

$$
\mathcal { N C } _ { 3 } : = \bigg \| \frac { W \tilde { H } } { \| W \tilde { H } \| _ { F } } - \frac { 1 } { \sqrt { K - 1 } } \big ( I _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \big ) \bigg \| _ { F } ,\tag{A.3}
$$

where

$$
\Sigma _ { W } = \frac { 1 } { \sum _ { k } n _ { k } } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { k } } \left( \pmb { h } _ { i } ^ { ( k ) } - \bar { \pmb { h } } _ { k } \right) \left( \pmb { h } _ { i } ^ { ( k ) } - \bar { \pmb { h } } _ { k } \right) ^ { \top } ,
$$

$$
\Sigma _ { B } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left( \bar { \pmb { h } } _ { k } - \bar { \pmb { h } } \right) \left( \bar { \pmb { h } } _ { k } - \bar { \pmb { h } } \right) ^ { \top } ,
$$

$$
\tilde { \pmb { W } } = [ \pmb { w } _ { 1 } - \pmb { w } , \ \pmb { w } _ { 2 } - \pmb { w } , \ \cdots , \ \pmb { w } _ { K } - \pmb { \bar { w } } ] ^ { T } \in \mathbb { R } ^ { K \times d } ,
$$

$$
\tilde { { \pmb H } } = [ \bar { { \pmb h } } _ { 1 } - \bar { { \pmb h } } , \ \bar { { \pmb h } } _ { 2 } - \bar { { \pmb h } } , \ \cdots , \ \bar { { \pmb h } } _ { K } - \bar { { \pmb h } } ] \in \mathbb { R } ^ { d \times K } ,
$$

and $\begin{array} { r } { \bar { w } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } w _ { k } , \bar { h } _ { k } = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } h _ { i } ^ { ( k ) } , \bar { h } = \frac { 1 } { \sum _ { \iota } n _ { k } } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { k } } h _ { i } ^ { ( k ) } } \end{array}$ . In this paper, we take $\mathcal { N C } _ { 1 } , \mathcal { N C } _ { 2 } .$ and $\mathcal { N C } _ { 3 }$ to compare the evolution of NC of the CE and BCE in the normalized feature space.

## A.2 Classification

In our experiments, we apply various metrics to comprehensively compare the performance of the BCE and CE losses in the normalized feature space, i.e., classification accuracy ${ \mathcal { A } } ,$ unbiased classification accuracy $\ b { A } ^ { * }$ , features compactness $\mathcal { E } _ { \mathrm { { c o m } } } .$ and features separability $\mathcal { E } _ { \mathrm { s e p } }$ . These metrics will be maximized when reaching NC.

For a classification task, suppose a dataset

$$
{ \mathcal { D } } = \bigcup _ { k = 1 } ^ { K } { \mathcal { D } } _ { k } = \bigcup _ { k = 1 } ^ { K } \big \{ X _ { i } ^ { ( k ) } \big \} _ { i = 1 } ^ { n _ { k } }\tag{A.4}
$$

from K categories, where $X _ { i } ^ { ( k ) }$ denotes the ith sample from the class $k .$ For any sample $X _ { i } ^ { ( k ) }$ , a model $\mathcal { M }$ converts it into its feature $\pmb { h } _ { i } ^ { ( k ) } = \mathcal { M } ( \pmb { X } _ { i } ^ { ( k ) } ) \in \mathbb { R } ^ { d }$ , where d is dimension of the feature space. To classify the sample, a linear, full connection classifier $\mathcal { C } = \left\{ \left( \boldsymbol { w } _ { k } , b _ { k } \right) \right\} _ { k = 1 } ^ { K }$ transform the feature into K decision scores $\left\{ \gamma { \pmb w } _ { j } ^ { \top } { \pmb h } _ { i } ^ { ( k ) } - b _ { j } \right\} _ { j = 1 } ^ { K } ,$ where $\{ \pmb { w } _ { k } \} _ { k = 1 } ^ { K }$ are classifier vectors and $\{ b _ { k } \} _ { k = 1 } ^ { K }$ are classifier biases. Then, the sample is classified into class $\hat { k } .$

$$
\hat { k } = \arg \operatorname* { m a x } _ { j } \left\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { j } \right\} _ { j = 1 } ^ { K } ,\tag{A.5}
$$

and correct classification for the sample is achieved when $k = \hat { k }$ , which is equivalent to

$$
\gamma \pmb { w } _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { k } = \operatorname* { m a x } \left\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { j } \right\} _ { j = 1 } ^ { K } .\tag{A.6}
$$

Classification accuracy. The commonly used classification accuracy is defined as

$$
\mathcal { A } ( \mathcal { M } , \mathcal { C } ) = \frac { | \mathcal { D } ( \mathcal { M } , \mathcal { C } ) | } { | \mathcal { D } | } \times 1 0 0 \% ,\tag{A.7}
$$

where

$$
\mathcal { D } ( \mathcal { M } , \mathcal { C } ) = \bigcup _ { k = 1 } ^ { K } \Big \{ \mathbf { X } ^ { ( k ) } : k = \arg \operatorname* { m a x } _ { \ell } \{ \gamma w _ { \ell } ^ { \top } h ^ { ( k ) } - b _ { \ell } \} , \mathbf { X } ^ { ( k ) } \in \mathcal { D } _ { k } , { h } ^ { ( k ) } = \mathcal { M } ( \mathbf { X } ^ { ( k ) } ) \Big \} ,\tag{A.8}
$$

consisting of all the samples correctly classified by and in $\mathcal { D }$

Unbiased classification accuracy. Similarly, the unbiased classification accuracy $\ b { A } ^ { * }$ is computed without using the classifier biases $\{ b _ { k } \} _ { k = 1 } ^ { K } , \mathrm { i . e . }$

$$
\mathcal { A } ^ { * } ( \mathcal { M } , \mathcal { C } ) = \frac { | \mathcal { D } ^ { * } ( \mathcal { M } , \mathcal { C } ) | } { | \mathcal { D } | } \times 1 0 0 \% ,\tag{A.9}
$$

where

$$
\mathcal { D } ^ { * } ( \mathcal { M } , \mathcal { C } ) = \bigcup _ { k = 1 } ^ { K } \Big \{ \pmb { X } ^ { ( k ) } : k = \arg \operatorname* { m a x } _ { \ell } \{ \gamma w _ { \ell } ^ { \top } h ^ { ( k ) } \} , \pmb { X } ^ { ( k ) } \in \mathcal { D } _ { k } , \pmb { h } ^ { ( k ) } = \mathcal { M } ( \pmb { X } ^ { ( k ) } ) \Big \} .\tag{A.10}
$$

The diference between the classification accuracy and unbiased one $\ b { A } ^ { * }$ reflects the afect of the variance of the biases $\{ b _ { j } \} _ { j = 1 } ^ { K }$ to the classification.

## A.3 Feature Properties

There is a close and intricate relationship between the sample classification and their feature properties. To precisely measure the properties of sample features, we define their intra-class compactness $\mathcal { E } _ { \mathrm { c o m } }$ and inter-class separability $\mathcal { E } _ { \mathrm { s e p } }$ 2

$$
\mathcal { E } _ { \mathrm { c o m } } = \frac { 1 } { 2 } \bigg [ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bigg ( \frac { 1 } { n _ { k } ^ { 2 } } \sum _ { i = 1 } ^ { n _ { k } } \sum _ { i ^ { \prime } = 1 } ^ { n _ { k } } \big \langle h _ { i } ^ { ( k ) } , h _ { i ^ { \prime } } ^ { ( k ) } \big \rangle \bigg ) + 1 \bigg ] ,\tag{A.11}
$$

$$
\mathcal E _ { \mathrm { s e p } } = \frac { 1 } { 2 } \Bigg [ 1 - \frac { 1 } { K ( K - 1 ) } \sum _ { k = 1 } ^ { K } \sum _ { \boldsymbol { k } ^ { \prime } = 1 \atop k ^ { \prime } \ne k } ^ { K } \left( \frac { 1 } { n _ { k } } \frac { 1 } { n _ { k ^ { \prime } } } \sum _ { i = 1 } ^ { n _ { k } } \sum _ { i ^ { \prime } = 1 } ^ { n _ { k ^ { \prime } } } \left. \boldsymbol h _ { i } ^ { ( k ) } , \boldsymbol h _ { i ^ { \prime } } ^ { ( k ^ { \prime } ) } \right. \right) \Bigg ] .\tag{A.12}
$$

where $\begin{array} { r } { \bar { \pmb { h } } = \frac { 1 } { | \mathcal { D } | } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n _ { k } } \pmb { h } _ { i } ^ { ( k ) } } \end{array}$ is the global feature center.

Due to the properties of neural collapse, when approaching the minima of the CE or BCE, the intra-class compactness $\mathcal { E } _ { \mathrm { c o m } }$ might be higher than $\begin{array} { r } { \frac { 1 } { 2 } - \frac { 1 } { 2 ( K - 1 ) } } \end{array}$ , and the inter-class separability $\mathcal { E } _ { \mathrm { s e p } }$ might be lower than $\begin{array} { r } { \frac { 1 } { 2 } + \frac { 1 } { 2 ( K - 1 ) } } \end{array}$ , for the model and classifier trained on the dataset .

## Appendix B Experiments

## B.1 Experimental Setting Details

In Sec. 5, we conducted extensive experiments with multiple models across various datasets to validate our theoretical findings and analyses. We present additional experimental details in Table B.1.

Neural collapse on balanced classification. By default, to validate the phenomenon of neural collapse and the convergence of models, we train ResNet18, ResNet50 (He et al., 2016), and DenseNet121 on the MNIST (Lecun et al., 1998), CIFAR10, and CIFAR100 (Krizhevsky, 2009) by using settings-1 and 2, and we train ViT (Dosovitskiy et al., 2021b) from scratch on the CIFAR10 dataset using settings-3 and 4, without applying additional regularization or data augmentation techniques during the training.

To assess classification performance on balanced datasets, we train ResNet18 and ResNet50 on the CIFAR10, CIFAR100, and Tiny-ImageNet (Wu et al., 2017) datasets using settings-5 and 6, employing the commonly used data augmentation techniques for a fair comparison. When training ViT models on the CIFAR10 and CIFAR100 datasets with the AdamW optimizer, we follow the weight decay setting of Chhabra (2024), and conduct a learning rate search based on setting-5 and 6. For training ViT on the Tiny-ImageNet dataset, we follow the network architecture and hyperparameter settings of Lee et al. (2021). The classification results of the models trained using CE and BCE losses are shown in Table B.9.

Table B.1: Detailed experimental settings
<table><tr><td rowspan="2"></td><td colspan="5">Neural Collapse</td><td colspan="2">Classification</td></tr><tr><td></td><td>setting-1</td><td>setting-2</td><td>setting-3</td><td>setting-4</td><td>setting-5</td><td>setting-6</td></tr><tr><td rowspan="6">Hpr-pter</td><td>epochs</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td><td>200</td></tr><tr><td>optimizer</td><td>SGD</td><td>AdamW</td><td>SGD</td><td>AdamW</td><td>SGD</td><td>AdamW</td></tr><tr><td>batch size</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>learning rate</td><td>0.01</td><td>0.001</td><td>0.03</td><td>0.0003</td><td>0.1</td><td>0.005</td></tr><tr><td>learning rate decay</td><td>step</td><td>cosine</td><td>step</td><td>cosine</td><td>step</td><td>cosine</td></tr><tr><td>weight decay λ</td><td>x</td><td>x</td><td>x</td><td>x</td><td>0.0001</td><td>0.05</td></tr><tr><td rowspan="9">Data Au.</td><td>warmup epochs</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>random cropping</td><td>x</td><td>x</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>horizontal flipping</td><td>x</td><td>x</td><td>x</td><td>x</td><td>0.5</td><td>0.5</td></tr><tr><td>random rotation</td><td>x</td><td>x</td><td>x</td><td>x</td><td>15</td><td>15</td></tr><tr><td>label smoothing</td><td>x</td><td>x</td><td>x</td><td>x</td><td>0.1</td><td>0.1</td></tr><tr><td>mixup alpha</td><td>x</td><td>x</td><td>x</td><td>x</td><td>0.8</td><td>0.8</td></tr><tr><td>cutmix alpha mixup prob.</td><td>x x</td><td>x x</td><td>x x</td><td>x x</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.8</td><td>0.8</td></tr><tr><td>normalization</td><td></td><td></td><td></td><td></td><td>mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]</td><td></td></tr></table>

Long-tailed recognition. The long-tailed datasets, CIFAR10-LT and CIFAR100- LT , are produced by sampling the training samples of the original datasets, using an exponential decay imbalance mode across classes, following the work of Cao et al. (2019). For each of them, we produced three variants of long-tailed datasets by using three diferent imbalance factors (IF), 10, 50, and 100. The IF is defined as the number of training samples in the largest class divided by the smallest. The experimental setting details for long-tailed recognition can be found in the work of Alshammari et al. (2022).

Open-set recognition. We sample from original datasets to generate the corresponding known and unknown classes, a simple summary is provided: (1) For MNIST, SVHN (Netzer et al., 2011) and CIFAR10, six known classes and four unknown classes are randomly sampled. (2) For the CIFAR+50 experiments, four classes are sampled from CIFAR10 for training and 50 nonoverlapping classes are used as unknown classes, which are sampled from the CIFAR100 dataset. We use the area under the receiver operating characteristic (AUROC) curve (Neal et al., 2018) and open set classification Rate (OSCR) (G. Chen et al., 2022) as evaluation metrics in the experiments. The experimental setting details for open-set recognition can be found in the work of G. Chen et al. (2022).

![](images/7312ed8979bdb19dd741a8c38e0ab1035e8c55aef5c2bb0fac2d9ae4ba55558a.jpg)

![](images/3bf3b203b4bdf1edf741031e696b56884ce19360c5dc3db585f8f718ea1b7c10.jpg)  
(a) ${ \mathcal { N } } { \mathcal { C } } _ { 1 }$ ${ \mathcal { N C } } _ { 2 } ,$ and $\mathcal { N } \mathcal { C } _ { 3 }$ of ResNet18 on MNIST

![](images/8bf6f54c2aa98d93fb80833ff0623f5ab4aa0d88893639291697315dc7b2aea8.jpg)

![](images/ff26c4223f0bec1deedc9fa13ddecd72aa0cc20e7ad928a1d6fcaabc859cad31.jpg)

![](images/a5b1a76394f0da87a6b141f88bb17aea7fcbe13505524512ca3aef5941af6b33.jpg)

![](images/7ddd285b505f63353ec5a841e6a01dc6e56a2bd945ce7101ab382ca9a1687e34.jpg)  
(b) N C<sub>1</sub>, ${ \mathcal { N C } } _ { 2 } ,$ and $\mathcal { N } \mathcal { C } _ { 3 }$ of ResNet50 on MNIST

![](images/ceea7c26039322b008b921ec5bc2b47156d34d04c3de61a5f4c3dc97d1c7342f.jpg)

![](images/93d1a44bbe038b0e13a3c5cf341d6bcfd5fe8d8a6c5278e7a64e45647d00bc28.jpg)

![](images/e0b307605c2bb9fab88e7cee344385d13192265c2dabec5e943d3ac45dd8b741.jpg)  
(c) $\small { \mathcal { N C } } _ { 1 } , { \mathcal { N C } } _ { 2 } ,$ and $\mathcal { N } \mathcal { C } _ { 3 }$ of DenseNet121 on MNIST  
Fig. B.1: The evolution of the three NC metrics in the training of ResNet18 (top), ResNet50 (middle) and DenseNet121 (bottom) on MNIST with CE and BCE losses using SGD and AdamW, respectively

## B.2 Neural Collapse of BCE and CE Losses

This section presents more results for the neural collapse (NC) that support the conclusions drawn in the paper. These results are computed on the training data from the the datasets. As we have mentioned before, we take $\mathcal { N C } _ { 1 } , \mathcal { N C } _ { 2 } ,$ and $\mathcal { N C } _ { 3 }$ presented in Z. Zhu et al. (2021) and Zhou et al. (2022) to evaluate the evolution of NC for the CE and BCE.

When $\gamma = 8 ,$ Fig. B.1 - B.4 show the evolution in the training of ResNet18, ResNet50, DenseNet121, and ViT on the training datasets of MNIST, CIFAR10, and CIFAR100 with CE and BCE losses. In the training on MNIST and CIFAR10 dataset, the three NC metrics of both CE and BCE losses approach zero at the terminal phase of training, indicating that both have approached the neural collapse. Additionally, it is obvious that BCE decreases faster and converges to a higher degree than CE in the first 50 epochs. However, during the training on CIFAR100, which is a more challenging dataset than MNIST and CIFAR10, the NC metrics of models trained with the SGD optimizer, especially $\mathcal { N C } _ { 2 }$ and $\mathcal { N C } _ { 3 }$ , do not approach zero, whereas those of models trained with AdamW approach zero. One can still observe that in most cases, the NC metrics of the BCE loss decreases faster and converges better than the CE loss.

![](images/dfdbe9942c3f968996f677d9322d2649e344a7248ffe785d1e07cc037556c875.jpg)

![](images/42ed6ea566a79e6032c650b83d6393042e3932d0e06c258fbaad9ad96aabc743.jpg)  
(a) N C<sub>1</sub>, N C<sub>2</sub>, and N C<sub>3</sub> of ResNet18 on CIFAR10

![](images/196e53146a7329c16be023b713f8bf3b26c1f94608b83b6b4ac987d1968ff87b.jpg)

![](images/545dd88eed587ce8a789ddd767d94deaca36c0f0c8d62a92a6184992bcae0e2d.jpg)

![](images/e3253694046086f74f6eecd7e050e05450a47d796dd9ddfdd49680f64646ed04.jpg)

![](images/1d4aa02eb73fbc1c310b7e107181ea586b69b6d94e7e21af4e6b0b740b48c996.jpg)  
(b) N C<sub>1</sub>, N C<sub>2</sub>, and N C<sub>3</sub> of ResNet50 on CIFAR10

![](images/721e2efc06caf9b5b66a4e7cca99da579da9091cf79765fb5ba6302b82819e3f.jpg)

![](images/f7596d7318beadceca3ff46cc696613553954cac3ba3706bff0259c5ec71047b.jpg)

![](images/bb4e3c18ce6ac8b8aa04ce6949caaf27a01a2cf5b038d5835fe867529f5f3396.jpg)  
(c) NC , NC , and NC of DenseNet121 on CIFAR10  
Fig. B.2: The evolution of the three NC metrics in the training of ResNet18 (top), ResNet50 (middle) and DenseNet121 (bottom) on CIFAR10 with CE and BCE losses using SGD and AdamW, respectively

## B.3 Impact of Classifier Biases and Scale Factor

The mean of initial classifier biases b. In Sec. 4.1, we point out that classifier biases difers substantially between the BCE and CE losses; specifically, at the global minimizer, the final bias of the BCE loss is uniquely determined, whereas in the CE loss there are infinitely many solutions. To validate this finding, we present the distribution of the final classifier biases and positive/negative unbiased decision scores for ResNet18 trained on MNIST, and ViT trained from scratch on CIFAR10 using diferent optimizer, with varying mean of initial classifier biases.

Fig. B.5 and B.6 show that the final classifier biases of CE-trained models are determined by their initial values, while the biases of BCE-trained models always converge to the same value, consistent with our theoretical analysis in the paper.

It is notable to point out a diference between CNNs and Transformers. For the CNN models such as ResNet and DenseNet, the output sample features are processed by the ReLU activation function, which makes that each component of the generated features is a nonnegative number and results in the sum of all sample features not being equal to zero (unless all the features are zero vectors, which is meaningless), conflicting with Eqs. (C.56) and (C.126) required for proving Theorems. Therefore, the perfect results for NC cannot be achieved based on the ResNet and DenseNet, as demonstrated by the obvious diference between the converged positive unbiased

![](images/e846ef035b2bdc24cebb0faf348d1d89aac6d3d1dfe81128e8ab774283422cf9.jpg)

![](images/5b216a1f6885fad4cd24b30c411653178392dcd4e4a2914403932c0419b16590.jpg)  
(a) $\mathcal { N C } _ { 1 } , \mathcal { N C } _ { 2 }$ , and $\mathcal { N } \mathcal { C } _ { 3 }$ of ResNet18 on CIFAR100

![](images/cee4d33183c90a3a5b2797a4ffbc00c6afed76f0a113c2f48386d6c4509e3cd6.jpg)

![](images/21f7e51305355774f84dcf297083e17c28f2feede4232251be2e45095463197d.jpg)

![](images/67e0983e8a4c3a27ab712cfe2eef3b178962b0e957adb63ef0d9ba678583061f.jpg)

![](images/411bb9673225966221f5adc8726255f094e4a38af960bfba974dec0d5ee2959d.jpg)  
(b) $\mathcal { N C } _ { 1 } , \mathcal { N C } _ { 2 } ,$ and $\mathcal { N } \mathcal { C } _ { 3 }$ of ResNet50 on CIFAR100

![](images/ff6cc7cbc092be1d7dba34622b5f0b9fe4b546f187af57689a2403a76d1a171e.jpg)

![](images/86d27020a1c3d77014a0904cb45e26eb7fbcdf35c5ea3c62c9f4a33010033e3f.jpg)

![](images/1ce838e6498501d4c11eb5534ce3b26e9757312940a8374127d4fc970848cc7d.jpg)  
(c) N C<sub>1</sub>, $\mathcal { N C } _ { 2 }$ , and $\mathcal { N } \mathcal { C } _ { 3 }$ of DenseNet121 on CIFAR100  
Fig. B.3: The evolution of the three NC metrics in the training of ResNet18 (top), ResNet50 (middle) and DenseNet121 (bottom) on CIFAR100 with CE and BCE losses using SGD and AdamW, respectively

![](images/be9ce7798a7daae6a43cbbcbda42f08ff468963ba19c0edb21ecad97a21b33c3.jpg)

![](images/d04ffbb9800981f4ecf19e13ee43edbc16ba4ec81179717ce28f24948ef26347.jpg)

![](images/174983a1f2cd932be2f3c52736c190d97e9f424454ea34d518fd6ff9c7b6909b.jpg)  
(a) ${ \mathcal { N } } { \mathcal { C } } _ { 1 }$ $\mathcal { N } \mathcal { C } _ { 2 }$ , and NC of ViT on CIFAR10

![](images/3d79339792fb7982ff840ff5c2b250c9f5d3f90135a5dbe129b1aca0f41e9faa.jpg)

![](images/fc8c5f2ac966a5e82e4aec58524530ac98b6f7b1b9ea8bd277fd69c564a44561.jpg)

![](images/94c6f9c0a0615079f22dd00a3b6e817efbe5a204ed50a20cb844773e7db1b8f9.jpg)  
(b) ${ \mathcal { N } } { \mathcal { C } } _ { 1 }$ ${ \mathcal { N C } } _ { 2 } ,$ and $\mathcal { N } \mathcal { C } _ { 3 }$ of ViT on CIFAR100

Fig. B.4: The evolution of the three NC metrics in the training of ViT on CIFAR10 (top) and CIFAR100 (bottom) with CE and BCE losses using SGD and AdamW, respectively

![](images/8329025ed949d94e86b6fc60d04f5a2d0234ab02b33e0f59748cfceb7fb7df29.jpg)

![](images/3a48375dbe9655cc9f643434fefe44aa271fa8368f319b62a1c1491554be72cd.jpg)

![](images/76c9b71b484f41be15db2badeecf438f694e6ef1369e292d7e11b8e0ac958dec.jpg)

![](images/354e73be9d4fcd21153b33d6a46eaf5add7e2f1b6752a598a636686d1104d4dd.jpg)  
Fig. B.5: The distribution of the final classifier biases and positive/negative unbiased decision scores for ResNet18 trained on MNIST with varying mean of initial classifier biases, while the variance of initial classifier biases is 0 and $\gamma = 8$ . The final classifier biases of CE-trained models are determined by their initial values, while that of BCEtrained models always converge to the same value

![](images/a2c8bd97ce090c6acd1b0d3e77e84205179d5ad65dd132e25608c51695d68aa1.jpg)  
Mean of Initialized Bias b

![](images/92f4edfe67066f1b8e09dc5c26f0f9bf80b405ffbe6df42b6541ea9a9c26ff45.jpg)

![](images/6016c03e4e252692807bddbf6425f074015f6c6e43539f39a7d96eb7f23cb9f3.jpg)  
Mean of Initialized Bias b

![](images/13af9bc6b6bb8b039f78e0312bc3cac8faf0ef28d96ca4aa70abf2822192e4ab.jpg)  
Mean of Initialized Bias b  
Fig. B.6: The distribution of the final classifier biases and positive/negative unbiased decision scores for ViT trained on CIFAR10 with varying mean of initial classifier biases, while the variance of initial classifier biases is 0 and $\gamma = 8$ . The final classifier biases of CE-trained models are determined by their initial values, while that of BCEtrained models always converge to the same value

decision scores and the theoretical value $\gamma = 8$ , regardless of whether the model is trained using CE or BCE. This phenomenon has been found and discussed in Z. Zhu et al. (2021). In contrast, the features output by ViT are not processed by ReLU or any other activation functions, which aligns more closely with the unconstrained feature model (Yaras et al., 2022; Z. Zhu et al., 2021). When training ViT with the CE or BCE, the converged positive and negative unbiased decision scores can converge to $\gamma = 8 \mathrm { a n d } - \frac { \gamma } { K - 1 } \overline { { { = } } } - \frac { 8 } { 9 }$ , respectively.

The variance of initial classifier biases b. It is well known that the variance of the parameters at model initialization determines whether gradient signals propagate stably in deep neural networks, thereby afecting convergence speed, training stability, and final generalization performance. Based on this consensus, to more comprehensively validate our theoretical findings, we conduct experiments with a fixed mean of initial classifier biases of 0, when the variances of initial classifier biases are set to 0, 0.1, 0.5, 1, 2, 3, 4, 5, 6, respectively, to further compare the BCE and CE losses.

Figures B.7 and B.8 visually show the distributions of the final unbiased decision scores and classifier biases for ResNet18 trained on MNIST and that for ViT trained from scratch on CIFAR10. For the diferent network architectures, optimizers, and scale factors, the BCE generally achieves better convergence compared to the CE, especially when $\gamma = 8 .$ . When $\gamma = 1 6 , 3 2$ or 64, the large normalized feature space makes it dificult for the models trained with either CE or BCE to converge, but the BCE-trained models usually yield better convergence.

![](images/4e5173aef9125b2424e697ea4cd92ce7ef471a98e6c4d4b52b44b79a47bbe154.jpg)

![](images/90ac72d365d9c52cc0b8b883a63f8249a52e225f16384b2ffa02c01ac3e42de8.jpg)  
(a) ResNet18 on MNIST, γ = 8

![](images/596a8708ace1b2325577018e04821b4038ea35554a26c7b44f0ba9ddf3c41769.jpg)

![](images/e1949be28d9bf83205ea3a1d82544b4b5db116a75b3ee513d8970a61d3aef8c6.jpg)

![](images/7e3121866444b5bba6d2a9736b7b5df87a944845484f20f2904a36e144f03f60.jpg)

![](images/e8b39676c01d4aa10cd274a46d29f26d4f5874de333dbdcc54bc3a5a6b876f96.jpg)

![](images/54e8358a9f9724ff061596137f154db6be0cdccc6cc72f0466a6802bc9d492a7.jpg)  
(b) ResNet18 on MNIST, γ = 16

![](images/84d4d421a7ba15e3463d6a939a2d3cea2db3e44ce63ac208ec34950c6d01221d.jpg)

![](images/349b660959e945a08b97ba9fda7bf393991fb649773d36ba911e6c5e9edb704d.jpg)

![](images/a2d4072baf9fd540d64f4329f3c8f04c039ad2991a7bb22e44afdf29c92ee8b4.jpg)

![](images/5fcc0ac6cb81b4f8d793713e8fba29e8a84283180ea51a13b99743a4cbc0767e.jpg)  
(c) ResNet18 on MNIST, γ = 32

![](images/7bb60341d7bb508737a0c4a314e005befd430a6ae32ba89c267dfe67b12fbb9e.jpg)

![](images/1dcfe3d2644bffb8a3bb55a8c6ad6166be874f3297e4bf7d859486bb3d744364.jpg)

![](images/676568353aa791a99f7de6859a366f3705b50b17527ed23644201e924511fad2.jpg)

![](images/1fddfd885ac2817fef07def5b2de9959e6abb79d2ec8560e8faed4a6c3e8596c.jpg)  
(d) ResNet18 on MNIST, γ = 64

![](images/bb8d23d8d73a66cc3a0135e12a76519fadd1974091297a397dd0312b8c34f1c3.jpg)  
Fig. B.7: The distribution of the final classifier biases and positive/negative unbiased decision scores of ResNet18 trained on MNIST with varying initial variance of the classifier biases. The diferent scale factors, $\gamma = 8 ~ ( \mathrm { 1 - s t ~ r o w } ) , ~ \gamma = 1 6$ (2-nd row), $\gamma = 3 2$ (3-rd row), and γ = 64 (4-th row), also significantly impact the convergence of decision scores and the final classifier biases

![](images/4611fd6edb89827dfafe42e2dbd08f597b561c696a2c3b49d2a85f8880808a24.jpg)

![](images/7ab6986d4c9c9864a316cde54de5ea86f20a837974a9ef40b68aefd663b33a03.jpg)

![](images/d6ee639b3dedb93a91b69301bedd7e11d9ce35f1605dc44c5a47bebbbfef43fe.jpg)  
(a) ViT on CIFAR10, γ = 8

![](images/611c3594004b9ffffb37e9d312418f8fbe5706506167aad75cf8266e9f7ddfc4.jpg)

![](images/adb9adc78aa366833c083eb84f8e6092740734d6636ae77605b1a09c20637f60.jpg)

![](images/4dde4737cc18d6e42e159549f45207db0e861c054f473914486bd332c273225b.jpg)

![](images/a6d114b0ba5d16e6ec760e3a41a2a20d7f407a5885f083afa7d03200853ed9fb.jpg)

![](images/056a9e61e2221261c605c7169654177798048ecdeb8ab2a072a326de964b372e.jpg)

(b) ViT on CIFAR10, γ = 16  
![](images/2405b5a9c5045254d7047938a5544c9244f17dd62d0392fe9b6b6841208dbd4c.jpg)

![](images/e0a304aee5a21ea28a760872ec8641162bb20adf631eee6d39784afab1e9e8db.jpg)

![](images/db9e499e3904ae5b11b69ff504169072b58d77afd29f35b5697036baef5f7b66.jpg)

![](images/35fd5670329fd474f4431dfc9f930181f431038cadc1348444838cd946373661.jpg)  
(c) ViT on CIFAR10, γ = 32  
Fig. B.8: The distribution of the final classifier bias and unbiased positive/negative decision scores of ViT trained on CIFAR10 with varying initial variance of the classifier bias. The diferent scale factors, $\gamma = 8$ (1-st row), $\gamma = 1 6$ (2-nd row), and $\gamma = 3 2$ (3-rd row), also significantly impact the convergence of decision scores and the final classifier bias

## B.4 Impact of Scale Factor to Model Training

The scale factor $\gamma .$ . To illustrate the efect of the scale factor for model training in the normalized feature space, we trained ResNet18 and ViT from scratch on the MNIST and CIFAR10 when γ varies from 0.1 to 64.

Fig. B.9 - B.12 visually show the distributions of the final decision scores and classifier biases for ResNet18 and ViT trained from scratch on the MNIST and CIFAR10 datasets. One can observe that, when $\gamma \le 8$ is small, the positive and negative unbiased decision scores, as well as the final classifier biases, of the models trained with BCE and CE losses closely align with the theoretical curves. However, as $\gamma$ increases, they increasingly deviate from their theoretical values. Specifically, when γ is reaching 32 or even 64, the distributions of the model’s positive/negative unbiased decision scores span large ranges and do not converge.

Furthermore, one can find that, in the CE-trained models, the converged classifier biases largely depend on their initial values, providing no correlation with the final positive/negative unbiased decision scores. In contrast, with appropriately small γ, regardless of the initial value of the biases, the final classifier biases of the BCE-trained models converge to the theoretical values.

## B.5 Numerical Results of ResNet18 and ViT trained on CIFAR10

In this section, Tables B.5 and B.6 provide the classification results and feature properties of ResNet18 and ViT trained from scratch on the MNIST and CIFAR10 datasets, when the scale factor $\gamma$ varies from 0.1 to 64.

It is evident that when $\gamma$ takes a very small value, it can significantly afect the model’s classification performance, as the distance between the positive and negative decision scores becomes small. In this situation, even slight disturbance in the model training or small variance in the classifier biases can substantially impact the classification accuracy. However, based on the results of $\mathcal { E } _ { \mathrm { c o m } }$ and $\mathcal { E } _ { \mathrm { s e p } }$ , we conclude that poor classification accuracy does not imply that the model has learned unsatisfactory sample features.

Moreover, as analyzed in Sec. 4.1, the classifier biases of the BCE loss play a crucial role in the model training, explicitly constraining the model’s positive and negative decision scores, which helps the model to learn features with better intra-class compactness and inter-class separability. One can find that, when $\gamma > 5$ , the BCEtrained models usually exhibit higher $\mathcal { E } _ { \mathrm { c o m } }$ and $\mathcal { E } _ { \mathrm { s e p } }$ than the CE-trained models, which provides strong support for our theoretical conclusions. However, when $\gamma \leq 1$ , due to the small normalized feature space, the minor perturbations during the training could significantly afect the feature learning, overshadowing the role of classifier biases in the BCE-trained models, which prevents them from obtaining good feature properties.

Furthermore, Tables B.2 and B.3 present the numerical results of the final positive/negative unbiased decision scores and classifier biases, for the CE- and BCEtrained ResNet18 and ViT, with $\gamma$ varying from 0.1 to 64. When $\gamma$ is very small, the gap between the converged unbiased positive and negative decision scores is also very small. When $\gamma \leq 0 . 5$ , despite the ResNet18 trained on CIFAR10 being very close to

![](images/7a6a290f3f0f234d000523be38da037ef84daf33c0255d4e4ed5175d54c723bf.jpg)

![](images/695629114ba94ed8f1647cd8e091ad619bf9a1d5b2697f40b0409c0a8e9941f5.jpg)

![](images/3f2b40323ccae9979435e020cea493420cd3fb288de6a327c286f297bb8a071e.jpg)  
(a) ResNet18 on MNIST, Mean(b) = 0, and Var(b) = 0

![](images/3e9fcffed64410bbc67a711203d0f70a7ad08e5846c8426527529382ae32e630.jpg)

![](images/50c004975a401c51503aab307353e283c3d0e8fb450e873ddb9b500483ebb6be.jpg)

![](images/f210947a3dea0150541e54165aeee0e8eb88b44c650e8a37153b48ac5e88644d.jpg)

![](images/e799779c10086fad00cb8d54294fd913322012a3506617301d566e615cc39eda.jpg)

![](images/c8286726e5ab0c17bd171017731796d24b79bc0eaae19cadd27bd7f0af0d8e44.jpg)  
(b) ResNet18 on MNIST, Mean(b) = 10, and Var(b) = 0

![](images/bc5da6fab643fa719317719c2532d221c7215c3b7db7ac7d8b361c7fe0e0a6d4.jpg)

![](images/3a8a7eb3c2fa3df4423b2c2f8e4f6c15c7d5f4bde29d151c8fea7d80c449d727.jpg)

![](images/dd53c2db4994b34372ce523801390f0fe13a72ad107efe207029d9d4a2d72533.jpg)  
(c) ResNet18 on MNIST, Mean(b) = 20, and Var(b) = 0

![](images/b71faca1785370e8f5cece9ef009826e329df2094469500671f83cb61010a1c8.jpg)

![](images/1598c25f0e4ef089e458d2ff0c0aba7582965737bbdbc9fe0f70c551c6503cac.jpg)

![](images/b1faa6a057476c3872f69ea2b3bafb3b1b734d1eb990d9c1c94a98cdc127ccd4.jpg)

![](images/539a4895a4d4c757fb388837eff7007caa5d44d6621ce0241ee45624f785cc47.jpg)

![](images/36bfc8fa20a4f3c10b13b171674ddc13900b45639ac8d2b8ba3d918142193f22.jpg)  
(d) ResNet18 on MNIST, Mean(b) = 30, and Var(b) = 0

Fig. B.9: The distribution of the final classifier biases and positive/negative unbiased decision scores of ResNet18 trained on MNIST by using various scale factors γ. The variance of initial classifier biases is set to 0, and their initial mean takes diferent values, 0, 10, 20, and 30, respectively

Scaling Factor γ  
Scaling Factor γ  
![](images/946a7b16ef825222fb3bef3c05dbc22b44731f6a5f15ea39d74776b413cce639.jpg)

![](images/fb36d2e629e5a80554610de3fa182f8827baec384eec488a52e7f4e9261a688d.jpg)

![](images/3f87c9d1453437427dc578a84d6946bed8b283ad22df388376dac91826b797a8.jpg)  
(a) ResNet18 on MNIST, Mean(b) = 5, and Var(b) = 0

![](images/80824458b6de45f5a9cc481a1bf35205261e32056035b33a0e477a2d6a9accde.jpg)

![](images/fda5bb6011b904ca9c490f432a2ee99d4a75c06ecb11bdc268a5010ef1049e19.jpg)

![](images/ecf1e05e9908489d3b9dfe438101aacee45af0c0a8d23a574d9db61b8edbb363.jpg)

![](images/67822a659704852cf4b3c864167bbeefeb2bddd5880a717137ed1160afa8c169.jpg)

![](images/f44c627c2be7d65a736581f475ec2a77120999fe22dee6283cddb6d3e458ca23.jpg)  
(b) ResNet18 on MNIST, Var(b) = 0, while the mean (Mean(b)) of initial classifier biases set to the theoretical value of the BCE loss (defined by Eq. (16))

Fig. B.10: The distribution of the final classifier biases and positive/negative unbiased decision scores of ResNet18 trained on MNIST by using various scale factors γ. The variance of the initial classifier biases is set to 0  
![](images/0fd778e8084e850ce866d8483ba68173993143d1a120564be1e0e11f2c8105a5.jpg)

![](images/13ad559701604df6c9e137b256e6500cbb9f4cb1d75e7ae0e9504154f8977279.jpg)

![](images/f38fec726817d316c18e25d2d6e660def97f76f7d3f309a49ad2ac39f64e15e9.jpg)

![](images/da1205873a3bb6470f9496fc947e0f8864ea5f21c700c58696d78faba99ca1ef.jpg)  
Fig. B.11: The distribution of the final classifier biases and positive/negative unbiased decision scores of ResNet18 (without the final ReLU) trained on CIFAR10 by using various scale factors γ. The mean and variance of the initial classifier biases is both set to 0

neural collapse, as indicated by the relatively small variance of its positive and negative unbiased decision scores in Table B.2, which are essentially converged, even a slight variance in the classifier bias can still significantly afect its classification accuracy, as indicated by the significantly gap between the classification accuracy and the unbiased one $\ b { A } ^ { * }$ in the Table B.5. When $\gamma \leq 2$ , due to the disturbances during the model training, the ViT trained on CIFAR10 has not fully converged, which is reflected in the relatively large variance of its positive and negative unbiased decision scores. In fact, as shown in Figure B.12b, when $\gamma \leq 2 ,$ , the distribution areas of the positive and negative unbiased decision scores of the ViT show significant overlapping, resulting in both low classification accuracy  and the unbiased one $\ b { A } ^ { * }$ for the ViT in Table B.6.

![](images/6464f2d14782e2d43b09449c6d908bc3f8bd0dcef060e4a59cd0e3ab40192877.jpg)

![](images/f53e5fd389a5d40e3008aae8fbb906f0f23480c8bdfaf937f1389e3eaf55f132.jpg)

![](images/4788169af390b8a93285ab643704bcc8fc20e7987e9e79009abcfc4dfa0da7ae.jpg)  
(a) ViT on CIFAR10, Mean(b) = 0, and Var(b) = 0

![](images/5b3aaab91899eb9d0b8e756d771da74c9823f50d0ac63d5f42e54b3ba2617bfa.jpg)

![](images/643468ed49b6f2e5e094889f97a186495c57d75c16ddec391794abf1009e3c28.jpg)

![](images/2d56b10288141916d02c2af3dda20cd5e79f7d6bac96383021d2e5ccfbbb6301.jpg)

![](images/468698360e11373e9b316990e09403ff359eb2394fe98e4c5be98e5fb8606b19.jpg)

![](images/f47b543fb47987f055b235de6080114de89d6f17445aa7802537c1533875d884.jpg)  
(b) ViT on CIFAR10, Mean(b) = 5, and Var(b) = 0

![](images/5fea8ff77e3124fb484b8b59228b39a9fb5aef1e01948045ae75390ca7af886a.jpg)

![](images/f0c2d3b13913e570dd6f19ec36a239f901cd086a970ca3af836e320b1ff3a7d9.jpg)

![](images/440e52ea8388450a5b43ebfafd0052335cbf89eb4834c29a9a7466d73c9d2fcd.jpg)

![](images/2055885afc907ddeac00e6483de559162241e03d332edf0304bbc9f8f1f06062.jpg)  
(c) ViT on CIFAR10, Mean(b) = 10, and Var(b) = 0  
Fig. B.12: The distribution of the final classifier biases and positive/negative unbiased decision scores of ViT trained on CIFAR10 by using various scale factors γ. The variance of the initial classifier biases is set to 0

Conversely, a larger γ expands the normalized feature space, leading to a large gap between the theoretical values of the converged positive and negative unbiased decision scores, which ensures robust classification; thus, the small perturbations such as the variance of the classifier biases cannot significantly afect the classification results. However, when $\gamma$ is too large, the positive and negative decision scores struggle to converge, as evidenced by the large variance of the final unbiased decision scores, indicating that the sample features have poor intra-class compactness and inter-class separability.

Table B.2: The final positive/negative unbiased decision scores and classifier biases of ResNet18 (without the final ReLU) with various γ on training data of CIFAR10 (mean standard deviation). The symbols $\mathbf { \boldsymbol { s } } ^ { ( k k ) }$ and $\pmb { s } ^ { ( j k ) }$ represent the positive and negative unbiased decision scores, respectively. The smaller standard deviation implies that the scores and biases tend to converge to similar values after training. The Mean(b) = 0, and $\mathrm { V a r } ( b ) = 0$  
(a) ResNet18 on CIFAR10 with SGD optimizer
<table><tr><td rowspan="2">γ</td><td colspan="3">CE with SGD</td><td colspan="3">BCE with SGD</td></tr><tr><td> $s ^ { ( k k ) }$ </td><td> $\mathbf { \boldsymbol { s } } ^ { ( j k ) }$ </td><td>b</td><td> $s ^ { ( k k ) }$ </td><td> $\mathbf { \boldsymbol { s } } ^ { ( j k ) }$ </td><td>b</td></tr><tr><td>0.1</td><td> $0 . 1 0 \pm 0 . 0 2$ </td><td> $- 0 . 0 1 \pm 0 . 1 0$ </td><td> $0 . 0 0 \pm 0 . 0 2$ </td><td> $0 . 1 0 \pm 0 . 0 2$ </td><td> $- 0 . 0 1 \pm 0 . 1 0$ </td><td> $2 . 2 0 \pm 0 . 0 2$ </td></tr><tr><td>0.5</td><td> $0 . 5 0 \pm 0 . 0 3$ </td><td> $- 0 . 0 5 \pm 0 . 2 6$ </td><td> $0 . 0 0 \pm 0 . 0 2$ </td><td> $0 . 5 0 \pm 0 . 0 3$ </td><td> $- 0 . 0 5 \pm 0 . 2 6$ </td><td> $2 . 2 4 \pm 0 . 0 2$ </td></tr><tr><td>1</td><td> $1 . 0 0 \pm 0 . 0 4$ </td><td> $- 0 . 1 1 \pm 0 . 3 1$ </td><td> $0 . 0 0 \pm 0 . 0 4$ </td><td> $1 . 0 0 \pm 0 . 0 5$ </td><td> $- 0 . 1 1 \pm 0 . 3 4$ </td><td> $2 . 2 9 \pm 0 . 0 3$ </td></tr><tr><td>2</td><td> $2 . 0 0 \pm 0 . 0 4$ </td><td> $- 0 . 2 2 \pm 0 . 0 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 0 0 \pm 0 . 0 5$ </td><td> $- 0 . 2 2 \pm 0 . 3 4$ </td><td> $2 . 4 6 \pm 0 . 1 4$ </td></tr><tr><td></td><td> $3 . 0 0 \pm 0 . 0 2$ </td><td> $- 0 . 3 3 \pm 0 . 0 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 0 0 \pm 0 . 0 6$ </td><td> $- 0 . 3 3 \pm 0 . 0 3$ </td><td> $2 . 6 8 \pm 0 . 0 0$ </td></tr><tr><td>34</td><td> $3 . 9 9 \pm 0 . 0 1$ </td><td> $- 0 . 4 4 \pm 0 . 0 5$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $4 . 0 0 \pm 0 . 0 5$ </td><td> $- 0 . 4 4 \pm 0 . 0 3$ </td><td> $3 . 0 2 \pm 0 . 0 0$ </td></tr><tr><td>5</td><td> $4 . 9 7 \pm 0 . 0 1$ </td><td> $- 0 . 5 5 \pm 0 . 1 1$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $4 . 9 9 \pm 0 . 0 0$ </td><td> $- 0 . 5 5 \pm 0 . 0 3$ </td><td> $3 . 4 0 \pm 0 . 0 0$ </td></tr><tr><td>6</td><td> $5 . 9 1 \pm 0 . 0 3$ </td><td> $- 0 . 6 6 \pm 0 . 2 3$ </td><td> $0 . 0 0 \pm 0 . 0 2$ </td><td> $5 . 9 9 \pm 0 . 0 0$ </td><td> $- 0 . 6 7 \pm 0 . 0 5$ </td><td> $3 . 8 1 \pm 0 . 0 0$ </td></tr><tr><td>7</td><td> $6 . 7 8 \pm 0 . 0 7$ </td><td> $- 0 . 7 5 \pm 0 . 4 1$ </td><td> $0 . 0 0 \pm 0 . 0 4$ </td><td> $6 . 9 8 \pm 0 . 0 1$ </td><td> $- 0 . 7 8 \pm 0 . 0 7$ </td><td> $4 . 2 3 \pm 0 . 0 0$ </td></tr><tr><td>8</td><td> $7 . 5 5 \pm 0 . 1 3$ </td><td> $- 0 . 8 4 \pm 0 . 6 0$ </td><td> $0 . 0 0 \pm 0 . 0 3$ </td><td> $7 . 9 6 \pm 0 . 0 1$ </td><td> $- 0 . 8 9 \pm 0 . 1 0$ </td><td> $4 . 6 5 \pm 0 . 0 1$ </td></tr><tr><td>12</td><td> $9 . 8 6 \pm 0 . 5 3$ </td><td> $- 1 . 0 5 \pm 1 . 3 0$ </td><td> $0 . 0 0 \pm 0 . 0 2$ </td><td> $7 . 7 6 \pm 0 . 1 0$ </td><td> $- 3 . 1 5 \pm 0 . 2 3$ </td><td> $3 . 0 0 \pm 0 . 0 1$ </td></tr><tr><td>16</td><td> $1 1 . 6 5 \pm 0 . 9 8$ </td><td> $- 1 . 1 4 \pm 1 . 9 7$ </td><td> $0 . 0 0 \pm 0 . 0 1$ </td><td> $7 . 0 1 \pm 0 . 2 2$ </td><td> $- 5 . 5 9 \pm 0 . 4 3$ </td><td> $1 . 3 5 \pm 0 . 0 3$ </td></tr><tr><td>32</td><td> $1 7 . 4 4 \pm 2 . 3 2$ </td><td> $- 1 . 0 2 \pm 4 . 0 8$ </td><td> $0 . 0 0 \pm 0 . 0 1$ </td><td> $9 . 7 9 \pm 1 . 3 9$ </td><td> $- 1 1 . 7 0 \pm 2 . 0 9$ </td><td> $0 . 2 0 \pm 0 . 0 4$ </td></tr><tr><td>64</td><td> $2 4 . 5 6 \pm 4 . 3 7$ </td><td> $- 2 . 1 0 \pm 6 . 8 6$ </td><td> $0 . 0 0 \pm 0 . 0 2$ </td><td> $1 3 . 8 7 \pm 2 . 5 5$ </td><td> $- 2 0 . 1 3 \pm 4 . 5 5$ </td><td> $0 . 0 2 \pm 0 . 1 6$ </td></tr></table>

(b) ResNet18 on CIFAR10 with AdamW optimizer
<table><tr><td colspan="7">(b) ResNet18 on CIFAR10 with Adam W optimizer</td></tr><tr><td rowspan="2">γ</td><td colspan="3">CE with AdamW</td><td colspan="3">BCE with AdamW</td></tr><tr><td> $s ^ { ( k k ) }$ </td><td> $\mathbf { \boldsymbol { s } } ^ { ( j k ) }$ </td><td>b</td><td> $s ^ { ( k k ) }$ </td><td> $\mathbf { \boldsymbol { s } } ^ { ( j k ) }$ </td><td>b</td></tr><tr><td>0.1</td><td> $0 . 1 0 \pm 0 . 0 2$ </td><td> $- 0 . 0 1 \pm 0 . 1 0$ </td><td> $0 . 0 0 \pm 0 . 0 2$ </td><td> $0 . 1 0 \pm 0 . 0 2$ </td><td> $- 0 . 0 1 \pm 0 . 1 0$ </td><td> $2 . 2 1 \pm 0 . 0 2$ </td></tr><tr><td>0.5</td><td> $0 . 5 0 \pm 0 . 0 0$ </td><td> $- 0 . 0 6 \pm 0 . 1 5$ </td><td> $0 . 0 0 \pm 0 . 0 3$ </td><td> $0 . 5 0 \pm 0 . 0 0$ </td><td> $- 0 . 0 6 \pm 0 . 1 8$ </td><td> $2 . 2 2 \pm 0 . 0 1$ </td></tr><tr><td>1</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 1 1 \pm 0 . 1 8$ </td><td> $0 . 0 0 \pm 0 . 0 3$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 1 1 \pm 0 . 2 5$ </td><td> $2 . 2 7 \pm 0 . 0 4$ </td></tr><tr><td>2</td><td> $2 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 2 2 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 2 2 \pm 0 . 0 0$ </td><td> $2 . 4 1 \pm 0 . 0 0$ </td></tr><tr><td></td><td> $3 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 3 3 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $3 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 3 3 \pm 0 . 0 0$ </td><td> $2 . 6 8 \pm 0 . 0 0$ </td></tr><tr><td></td><td> $4 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 4 4 \pm 0 . 0 1$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $4 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 4 4 \pm 0 . 0 1$ </td><td> $3 . 0 2 \pm 0 . 0 0$ </td></tr><tr><td>345</td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 5 6 \pm 0 . 0 2$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 5 6 \pm 0 . 0 1$ </td><td> $3 . 4 0 \pm 0 . 0 0$ </td></tr><tr><td>6</td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 6 7 \pm 0 . 0 3$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $6 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 6 7 \pm 0 . 0 2$ </td><td> $3 . 8 1 \pm 0 . 0 0$ </td></tr><tr><td>7</td><td> $7 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 7 8 \pm 0 . 0 3$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $7 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 7 8 \pm 0 . 0 2$ </td><td> $4 . 2 4 \pm 0 . 0 0$ </td></tr><tr><td>8</td><td> $7 . 9 9 \pm 0 . 0 0$ </td><td> $- 0 . 8 9 \pm 0 . 0 5$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $8 . 0 0 \pm 0 . 0 0$ </td><td> $- 0 . 8 9 \pm 0 . 0 3$ </td><td> $4 . 6 7 \pm 0 . 0 0$ </td></tr><tr><td>12</td><td> $1 1 . 9 0 \pm 0 . 0 8$ </td><td> $- 1 . 3 2 \pm 0 . 4 5$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $1 1 . 9 9 \pm 0 . 0 0$ </td><td> $- 1 . 3 3 \pm 0 . 0 5$ </td><td> $6 . 4 4 \pm 0 . 0 0$ </td></tr><tr><td>16</td><td> $1 5 . 2 6 \pm 0 . 3 7$ </td><td> $- 1 . 6 8 \pm 1 . 5 2$ </td><td> $0 . 0 1 \pm 0 . 0 2$ </td><td> $1 4 . 3 5 \pm 0 . 3 2$ </td><td> $- 2 . 4 8 \pm 0 . 3 7$ </td><td> $6 . 6 6 \pm 0 . 4 0$ </td></tr><tr><td>32</td><td> $2 6 . 0 9 \pm 1 . 8 4$ </td><td> $- 1 . 8 5 \pm 4 . 6 4$ </td><td> $0 . 0 0 \pm 0 . 0 2$ </td><td> $1 4 . 1 4 \pm 0 . 2 1$ </td><td> $- 1 1 . 4 0 \pm 0 . 3 3$ </td><td> $2 . 0 9 \pm 0 . 1 4$ </td></tr><tr><td>64</td><td> $4 0 . 9 8 \pm 5 . 5 9$ </td><td> $2 . 4 0 \pm 7 . 9 8$ </td><td> $0 . 0 0 \pm 0 . 0 6$ </td><td> $1 6 . 8 5 \pm 1 . 2 4$ </td><td> $- 2 4 . 0 1 \pm 3 . 8 8$ </td><td> $0 . 1 7 \pm 0 . 0 4$ </td></tr></table>

## B.6 Numerical Results of ResNet18 on Test Set of CIFAR10

This paper focuses on the training set, providing an in-depth analysis of the diferences between classification and feature learning. We believe that their diference can generalize to the test set, although there is still a lack of solid theoretical proof. Table B.4 ofers the classification accuracy and feature properties of trained ResNet18 on the test set of CIFAR10, while the results on training set have been shown in Table B.5.

Table B.3: The final positive/negative unbiased decision scores and classifier biases of ViT with varying γ on training data of CIFAR10 (mean  standard deviation). The symbols $\mathbf { \boldsymbol { s } } ^ { ( k k ) }$ and $\pmb { s } ^ { ( j k ) }$ represent the positive and negative unbiased decision scores, respectively. The smaller standard deviation implies that the scores and biases tend to converge to similar values. The $\mathrm { M e a n } ( \boldsymbol { b } ) = 5$ , and $\mathrm { V a r } ( b ) = 0$
<table><tr><td rowspan="2">γ</td><td colspan="3">CE with SGD</td><td colspan="3">BCE with SGD</td></tr><tr><td> $s ^ { ( k k ) }$ </td><td> $\boldsymbol { s } ^ { ( j k ) }$ </td><td>b</td><td> $s ^ { ( k k ) }$ </td><td> $\mathbf { \boldsymbol { s } } ^ { ( j k ) }$ </td><td>b</td></tr><tr><td>0.1</td><td> $0 . 0 8 \pm 0 . 0 6$ </td><td> $- 0 . 0 1 \pm 0 . 1 0$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $0 . 0 8 \pm 0 . 0 6$ </td><td> $- 0 . 0 1 \pm 0 . 1 0$ </td><td> $2 . 2 0 \pm 0 . 0 2$ </td></tr><tr><td>0.5</td><td> $0 . 4 4 \pm 0 . 1 5$ </td><td> $- 0 . 0 5 \pm 0 . 3 4$ </td><td> $5 . 0 0 \pm 0 . 0 2$ </td><td> $0 . 4 4 \pm 0 . 1 6$ </td><td> $- 0 . 0 5 \pm 0 . 3 6$ </td><td> $2 . 2 5 \pm 0 . 0 1$ </td></tr><tr><td>1</td><td> $0 . 9 6 \pm 0 . 1 9$ </td><td> $- 0 . 1 1 \pm 0 . 4 0$ </td><td> $5 . 0 0 \pm 0 . 0 2$ </td><td> $0 . 9 5 \pm 0 . 2 2$ </td><td> $- 0 . 1 0 \pm 0 . 5 2$ </td><td> $2 . 3 5 \pm 0 . 0 5$ </td></tr><tr><td>2</td><td>1.96 ± 0.24</td><td>−0.21 ± 0.35</td><td>5.00 ± 0.08</td><td>1.95 ± 0.26</td><td>−0.21 ± 0.48</td><td>2.48 ± 0.12</td></tr><tr><td>3</td><td> $2 . 9 7 \pm 0 . 2 2$ </td><td> $- 0 . 3 3 \pm 0 . 1 0$ </td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 9 5 \pm 0 . 3 3$ </td><td> $- 0 . 3 3 \pm 0 . 1 3$ </td><td> $2 . 6 6 \pm 0 . 0 0$ </td></tr><tr><td>4</td><td> $3 . 9 7 \pm 0 . 1 5$ </td><td> $- 0 . 4 4 \pm 0 . 1 1$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $3 . 9 6 \pm 0 . 3 2$ </td><td> $- 0 . 4 4 \pm 0 . 1 3$ </td><td> $2 . 9 9 \pm 0 . 0 0$ </td></tr><tr><td>5</td><td> $4 . 9 4 \pm 0 . 0 8$ </td><td> $- 0 . 5 5 \pm 0 . 1 6$ </td><td> $5 . 0 0 \pm 0 . 0 2$ </td><td> $4 . 9 6 \pm 0 . 2 7$ </td><td> $- 0 . 5 5 \pm 0 . 1 3$ </td><td> $3 . 3 5 \pm 0 . 0 1$ </td></tr><tr><td>6</td><td> $5 . 8 8 \pm 0 . 0 7$ </td><td> $- 0 . 6 5 \pm 0 . 2 6$ </td><td> $5 . 0 0 \pm 0 . 0 4$ </td><td> $5 . 9 6 \pm 0 . 2 0$ </td><td> $- 0 . 6 6 \pm 0 . 1 5$ </td><td> $3 . 7 4 \pm 0 . 0 1$ </td></tr><tr><td>7</td><td> $6 . 7 7 \pm 0 . 1 1$ </td><td> $- 0 . 7 5 \pm 0 . 3 9$ </td><td> $5 . 0 0 \pm 0 . 0 7$ </td><td> $6 . 9 4 \pm 0 . 1 5$ </td><td> $- 0 . 7 7 \pm 0 . 1 7$ </td><td> $4 . 1 4 \pm 0 . 0 1$ </td></tr><tr><td>8</td><td> $7 . 6 0 \pm 0 . 1 7$ </td><td> $- 0 . 8 4 \pm 0 . 5 4$ </td><td> $5 . 0 0 \pm 0 . 1 1$ </td><td> $7 . 9 2 \pm 0 . 1 2$ </td><td> $- 0 . 8 8 \pm 0 . 2 3$ </td><td> $4 . 5 5 \pm 0 . 0 2$ </td></tr><tr><td>12</td><td> $1 0 . 5 1 \pm 0 . 5 3$ </td><td> $- 1 . 1 7 \pm 1 . 2 9$ </td><td> $5 . 0 0 \pm 0 . 2 2$ </td><td> $1 1 . 1 8 \pm 0 . 1 7$ </td><td> $- 1 . 5 0 \pm 0 . 6 3$ </td><td> $5 . 7 0 \pm 0 . 1 4$ </td></tr><tr><td>16</td><td> $1 2 . 8 5 \pm 0 . 9 1$ </td><td> $- 1 . 4 5 \pm 2 . 1 8$ </td><td> $5 . 0 0 \pm 0 . 2 2$ </td><td> $1 2 . 3 1 \pm 0 . 3 2$ </td><td> $- 2 . 9 6 \pm 0 . 9 7$ </td><td> $5 . 6 1 \pm 0 . 1 7$ </td></tr><tr><td>32</td><td> $1 8 . 3 8 \pm 2 . 3 8$ </td><td> $- 1 . 6 4 \pm 5 . 0 3$ </td><td> $5 . 0 0 \pm 0 . 1 1$ </td><td> $1 4 . 5 1 \pm 1 . 0 3$ </td><td> $- 8 . 0 8 \pm 2 . 1 8$ </td><td> $5 . 1 1 \pm 0 . 1 0$ </td></tr><tr><td>64</td><td> $2 1 . 5 6 \pm 4 . 9 9$ </td><td> $- 3 . 3 3 \pm 7 . 8 5$ </td><td> $5 . 0 0 \pm 0 . 1 3$ </td><td> $1 4 . 0 1 \pm 1 . 3 2$ </td><td> $. 1 0 . 0 9 \pm 3 . 2 9$ </td><td> $5 . 1 4 \pm 0 . 5 8$ </td></tr></table>

(b) ViT on CIFAR10 with AdamW optimizer
<table><tr><td rowspan="2">γ</td><td colspan="3">CE with AdamW</td><td colspan="3">BCE with AdamW</td></tr><tr><td> $s ^ { ( k k ) }$ </td><td> $\boldsymbol { s } ^ { ( j k ) }$ </td><td>b</td><td> $s ^ { ( k k ) }$ </td><td> $\mathbf { \boldsymbol { s } } ^ { ( j k ) }$ </td><td>b</td></tr><tr><td> $_ { 0 . 1 }$ </td><td> $0 . 0 8 \pm 0 . 0 6$ </td><td> $- 0 . 0 1 \pm 0 . 1 0$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $0 . 0 7 \pm 0 . 0 7$ </td><td> $- 0 . 0 0 \pm 0 . 1 0$ </td><td> $2 . 2 0 \pm 0 . 0 1$ </td></tr><tr><td> $_ { 0 . 5 }$ </td><td> $0 . 4 8 \pm 0 . 1 0$ </td><td>−0.05 ± 0.22</td><td>5.00 ± 0.01</td><td> $0 . 3 6 \pm 0 . 2 7$ </td><td> $- 0 . 0 3 \pm 0 . 4 3$ </td><td> $2 . 2 8 \pm 0 . 0 6$ </td></tr><tr><td>1</td><td> $0 . 9 8 \pm 0 . 1 4$ </td><td> $- 0 . 1 1 \pm 0 . 3 1$ </td><td> $5 . 0 0 \pm 0 . 0 4$ </td><td> $0 . 7 2 \pm 0 . 4 1$ </td><td> $- 0 . 0 5 \pm 0 . 6 2$ </td><td> $2 . 3 8 \pm 0 . 1 1$ </td></tr><tr><td>2</td><td> $1 . 9 8 \pm 0 . 1 9$ </td><td> $- 0 . 2 2 \pm 0 . 0 7$ </td><td> $5 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 8 0 \pm 0 . 5 5$ </td><td> $- 0 . 1 9 \pm 0 . 5 9$ </td><td> $2 . 5 2 \pm 0 . 1 4$ </td></tr><tr><td>3</td><td> $2 . 9 9 \pm 0 . 1 5$ </td><td> $- 0 . 3 3 \pm 0 . 0 6$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $2 . 9 6 \pm 0 . 3 4$ </td><td> $- 0 . 3 3 \pm 0 . 1 2$ </td><td> $2 . 6 7 \pm 0 . 0 0$ </td></tr><tr><td>4</td><td> $3 . 9 9 \pm 0 . 0 7$ </td><td> $- 0 . 4 4 \pm 0 . 0 6$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $3 . 9 7 \pm 0 . 3 1$ </td><td> $- 0 . 4 4 \pm 0 . 1 1$ </td><td> $3 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>5</td><td> $4 . 9 8 \pm 0 . 0 4$ </td><td> $- 0 . 5 5 \pm 0 . 0 5$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $4 . 9 8 \pm 0 . 2 4$ </td><td> $- 0 . 5 5 \pm 0 . 0 9$ </td><td> $3 . 3 7 \pm 0 . 0 0$ </td></tr><tr><td>6</td><td> $5 . 9 8 \pm 0 . 0 4$ </td><td> $- 0 . 6 6 \pm 0 . 0 6$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $5 . 9 8 \pm 0 . 1 8$ </td><td> $- 0 . 6 6 \pm 0 . 0 8$ </td><td> $3 . 7 6 \pm 0 . 0 0$ </td></tr><tr><td>7</td><td> $6 . 9 6 \pm 0 . 0 2$ </td><td> $- 0 . 7 7 \pm 0 . 1 0$ </td><td> $5 . 0 0 \pm 0 . 0 2$ </td><td> $6 . 9 8 \pm 0 . 1 1$ </td><td> $- 0 . 7 8 \pm 0 . 0 8$ </td><td> $4 . 1 4 \pm 0 . 0 2$ </td></tr><tr><td>8</td><td> $7 . 9 2 \pm 0 . 0 4$ </td><td> $- 0 . 8 8 \pm 0 . 2 0$ </td><td> $5 . 0 0 \pm 0 . 0 3$ </td><td> $7 . 9 8 \pm 0 . 0 5$ </td><td> $- 0 . 8 9 \pm 0 . 0 6$ </td><td> $4 . 5 3 \pm 0 . 0 1$ </td></tr><tr><td>12</td><td> $1 1 . 4 4 \pm 0 . 2 6$ </td><td> $\cdot 1 . 2 6 \pm 0 . 8 4$ </td><td> $5 . 0 0 \pm 0 . 0 3$ </td><td> $1 1 . 8 5 \pm 0 . 0 4$ </td><td> $- 1 . 3 6 \pm 0 . 1 4$ </td><td> $5 . 8 2 \pm 0 . 0 9$ </td></tr><tr><td>16</td><td> $1 4 . 4 7 \pm 0 . 5 8$ </td><td> $- 1 . 5 7 \pm 1 . 3 7$ </td><td> $5 . 0 0 \pm 0 . 0 2$ </td><td> $1 2 . 6 8 \pm 0 . 1 3$ </td><td> $- 3 . 1 6 \pm 0 . 2 9$ </td><td> $5 . 3 2 \pm 0 . 0 6$ </td></tr><tr><td>32</td><td> $2 3 . 3 9 \pm 2 . 2 7$ </td><td> $- 2 . 6 7 \pm 3 . 9 9$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $1 6 . 2 9 \pm 0 . 6 0$ </td><td> $- 9 . 3 4 \pm 1 . 5 1$ </td><td> $5 . 0 2 \pm 0 . 0 1$ </td></tr><tr><td>64</td><td> $3 2 . 2 5 \pm 4 . 9 5$ </td><td> $- 4 . 4 3 \pm 8 . 4 3$ </td><td> $5 . 0 0 \pm 0 . 0 1$ </td><td> $2 0 . 4 5 \pm 1 . 6 0$ </td><td> ${ \cdot 1 8 . 1 8 \pm 3 . 3 8 }$ </td><td> $5 . 0 1 \pm 0 . 0 1$ </td></tr></table>

From the table, one can observe that when γ is small, such as 0.1 or 0.5, the models’ classification is poor, but the intra-class compactness and inter-class separability of the features are higher than 0.8 and 0.5, respectively. Conversely, when γ is large, such as 32 or 64, the models achieve higher classification accuracy, but the feature properties significantly degrade. These results demonstrate that the diferences between the tasks of classification and deep feature learning, as analyzed on the training set, also generalize to the test set.

Table B.4: The classification accuracy and feature properties for ResNet18 on the test set of CIFAR10  
(a) CE with SGD — smaller scales
<table><tr><td> $\gamma$ </td><td>0.1</td><td>0.5</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td> $\boldsymbol { A }$ </td><td>25.81</td><td>44.18</td><td>61.54</td><td>78.95</td><td>80.24</td><td>80.69</td><td>80.16</td></tr><tr><td> $\ b { A } ^ { * }$ </td><td>25.62</td><td>46.94</td><td>61.60</td><td>78.94</td><td>80.24</td><td>80.68</td><td>80.16</td></tr><tr><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td>0.8887</td><td>0.8290</td><td>0.8240</td><td>0.8209</td><td>0.8293</td><td>0.8318</td><td>0.8232</td></tr><tr><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>0.5153</td><td>0.5347</td><td>0.5348</td><td>0.5348</td><td>0.5358</td><td>0.5363</td><td>0.5350</td></tr><tr><td colspan="8">(b) CE with SGD — larger scales</td></tr><tr><td> $\gamma$ </td><td>6</td><td>7</td><td>8</td><td>12</td><td>16</td><td>32</td><td>64</td></tr><tr><td> $\boldsymbol { A }$ </td><td>80.07</td><td>79.49</td><td>79.76</td><td>78.72</td><td>78.91</td><td>77.84</td><td>77.46</td></tr><tr><td> $\ b { A } ^ { * }$ </td><td>80.08</td><td>79.49</td><td>79.76</td><td>78.71</td><td>78.91</td><td>77.79</td><td>77.44</td></tr><tr><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td>0.8136</td><td>0.7972</td><td>0.7823</td><td>0.7428</td><td>0.7251</td><td>0.6908</td><td>0.6849</td></tr><tr><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>0.5336</td><td>0.5273</td><td>0.5114</td><td>0.4159</td><td>0.3606</td><td>0.2658</td><td>0.2044</td></tr><tr><td colspan="8">(c) BCE with SGD — smaller scales</td></tr><tr><td> $\gamma$ </td><td>0.1</td><td>0.5</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td> $\boldsymbol { A }$ </td><td>24.98</td><td>44.12</td><td>58.41</td><td>76.76</td><td>81.13</td><td>82.34</td><td>82.70</td></tr><tr><td> $A ^ { * }$ </td><td>26.55</td><td>45.67</td><td>59.71</td><td>76.75</td><td>81.13</td><td>82.34</td><td>82.70</td></tr><tr><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td>0.8891</td><td>0.8244</td><td>0.8254</td><td>0.8370</td><td>0.8324</td><td>0.8430</td><td>0.8464</td></tr><tr><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>0.5142</td><td>0.5349</td><td>0.5355</td><td>0.5343</td><td>0.5364</td><td>0.5375</td><td>0.5380</td></tr><tr><td colspan="8">(d) BCE with SGD — larger scales</td></tr><tr><td> $\gamma$ </td><td>6</td><td>7</td><td>8</td><td>12</td><td>16</td><td>32</td><td>64</td></tr><tr><td> $\boldsymbol { A }$ </td><td>82.54</td><td>82.62</td><td>81.83</td><td>81.63</td><td>82.14</td><td>81.69</td><td>82.54</td></tr><tr><td> $\ b { A } ^ { * }$ </td><td>82.55</td><td>82.62</td><td>81.82</td><td>81.63</td><td>82.12</td><td>81.72</td><td>82.56</td></tr><tr><td> $\scriptstyle { \mathcal { E } } _ { \mathrm { c o m } }$ </td><td>0.8455</td><td>0.8457</td><td>0.8355</td><td>0.8226</td><td>0.8073</td><td>0.7491</td><td>0.7261</td></tr><tr><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>0.5379</td><td>0.5378</td><td>0.5362</td><td>0.4239</td><td>0.3680</td><td>0.3177</td><td>0.2934</td></tr></table>

Table B.5: The classification accuracy (%) and feature properties of ResNet18 (without the final ReLU) with various γ on training data of CIFAR10. The accuracies of and $\ b { A } ^ { * }$ are calculated using $\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } ^ { ( k ) } - b _ { j } \}$ and $\{ \gamma w _ { j } ^ { \top } h ^ { ( k ) } \}$ , respectively. The higher $\mathcal { E } _ { \mathrm { c o m } }$ and $\mathcal { E } _ { \mathrm { s e p } }$ respectively indicate the smaller intra-class variability and the bigger inter-class separability. The Mean(b) = 0, and $\mathrm { V a r } ( b ) = 0$  
(a) Results with SGD
<table><tr><td rowspan="2"> $\gamma$ </td><td colspan="4">CE with SGD</td><td colspan="4">BCE with SGD</td></tr><tr><td> $\mathcal { A }$ </td><td> $\ b { A } ^ { * }$ </td><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td>A</td><td>A*</td><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td></tr><tr><td>0.1</td><td>26.24</td><td>27.27</td><td>0.965</td><td>0.526</td><td>25.47</td><td>27.77</td><td>0.965</td><td>0.526</td></tr><tr><td>0.5</td><td>54.28</td><td>57.00</td><td>0.995</td><td>0.554</td><td>52.07</td><td>54.75</td><td>0.995</td><td>0.554</td></tr><tr><td>1</td><td>79.50</td><td>80.01</td><td>0.998</td><td>0.555</td><td>76.73</td><td>77.76</td><td>0.997</td><td>0.555</td></tr><tr><td>2</td><td>99.97</td><td>99.97</td><td>0.999</td><td>0.555</td><td>93.79</td><td>93.78</td><td>0.999</td><td>0.553</td></tr><tr><td>3</td><td>100.0</td><td>100.0</td><td>0.999</td><td>0.555</td><td>99.97</td><td>99.97</td><td>0.999</td><td>0.555</td></tr><tr><td>4</td><td>100.0</td><td>100.0</td><td>0.998</td><td>0.555</td><td>99.99</td><td>99.99</td><td>0.999</td><td>0.555</td></tr><tr><td>5</td><td>100.0</td><td>100.0</td><td>0.995</td><td>0.555</td><td>100.0</td><td>100.0</td><td>0.999</td><td>0.555</td></tr><tr><td>6</td><td>100.0</td><td>100.0</td><td>0.987</td><td>0.553</td><td>100.0</td><td>100.0</td><td>0.999</td><td>0.555</td></tr><tr><td>7</td><td>100.0</td><td>100.0</td><td>0.975</td><td>0.547</td><td>100.0</td><td>100.0</td><td>0.998</td><td>0.555</td></tr><tr><td>8</td><td>100.0</td><td>100.0</td><td>0.960</td><td>0.535</td><td>100.0</td><td>100.0</td><td>0.997</td><td>0.554</td></tr><tr><td>12</td><td>100.0</td><td>100.0</td><td>0.905</td><td>0.454</td><td>100.0</td><td>100.0</td><td>0.988</td><td>0.459</td></tr><tr><td>16</td><td>100.0</td><td>100.0</td><td>0.862</td><td>0.394</td><td>100.0</td><td>100.0</td><td>0.973</td><td>0.410</td></tr><tr><td>32</td><td>100.0</td><td>100.0</td><td>0.789</td><td>0.288</td><td>100.0</td><td>100.0</td><td>0.899</td><td>0.354</td></tr><tr><td>64</td><td>100.0</td><td>100.0</td><td>0.766</td><td>0.221</td><td>100.0</td><td>100.0</td><td>0.842</td><td>0.313</td></tr></table>

(b) Results with AdamW
<table><tr><td rowspan="2"> $\gamma$ </td><td colspan="4">CE with AdamW</td><td colspan="4">BCE with AdamW</td></tr><tr><td> $\mathcal { A }$ </td><td> $\ b { A } ^ { * }$ </td><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td> $\boldsymbol { A }$ </td><td> $\ b { A } ^ { * }$ </td><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td></tr><tr><td>0.1</td><td>19.96</td><td>43.64</td><td>0.986</td><td>0.532</td><td>19.93</td><td>42.66</td><td>0.980</td><td>0.531</td></tr><tr><td> $_ { 0 . 5 }$ </td><td>70.00</td><td>84.59</td><td>1.0</td><td>0.555</td><td>60.00</td><td>74.56</td><td>1.0</td><td>0.555</td></tr><tr><td>1</td><td>90.00</td><td>94.17</td><td>1.0</td><td>0.555</td><td>80.00</td><td>82.80</td><td>1.0</td><td>0.555</td></tr><tr><td>2</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td></tr><tr><td>3</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td></tr><tr><td>4</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td></tr><tr><td>5</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td></tr><tr><td>6</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td></tr><tr><td>7</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td></tr><tr><td>8</td><td>100.0</td><td>100.0</td><td>0.999</td><td>0.555</td><td>100.0</td><td>100.0</td><td>1.0</td><td>0.555</td></tr><tr><td>12</td><td>100.0</td><td>100.0</td><td>0.992</td><td>0.555</td><td>100.0</td><td>100.0</td><td>0.999</td><td>0.555</td></tr><tr><td>16</td><td>100.0</td><td>100.0</td><td>0.957</td><td>0.549</td><td>100.0</td><td>100.0</td><td>0.998</td><td>0.527</td></tr><tr><td>32</td><td>100.0</td><td>100.0</td><td>0.883</td><td>0.478</td><td>100.0</td><td>100.0</td><td>0.993</td><td>0.403</td></tr><tr><td>64</td><td>100.0</td><td>100.0</td><td>0.809</td><td>0.339</td><td>100.0</td><td>100.0</td><td>0.928</td><td>0.371</td></tr></table>

Table B.6: The classification accuracy (%) and feature properties of ViT with various γ on training data of CIFAR10. The accuracies of and $\ b { A } ^ { * }$ are calculated using the decision score $\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } ^ { ( k ) } - b _ { j } \}$ and $\{ \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } ^ { ( k ) } \}$ , respectively. The higher $\mathcal { E } _ { \mathrm { c o m } }$ and $\mathcal { E } _ { \mathrm { s e p } }$ respectively indicate the smaller intra-class variability and the bigger inter-class separability. The $\mathrm { M e a n } ( \boldsymbol { b } ) = 5$ , and $\mathrm { V a r } ( b ) = 0$  
(a) Results with SGD
<table><tr><td rowspan="2"> $\gamma$ </td><td colspan="4">CE with SGD</td><td colspan="4">BCE with SGD</td></tr><tr><td> $\boldsymbol { A }$ </td><td> $A ^ { * }$ </td><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td> $\boldsymbol { A }$ </td><td> $A ^ { * }$ </td><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td></tr><tr><td>0.1</td><td>27.19</td><td>27.20</td><td>0.826</td><td>0.523</td><td>26.42</td><td>26.45</td><td>0.823</td><td>0.524</td></tr><tr><td>0.5</td><td>45.63</td><td>45.65</td><td>0.892</td><td>0.542</td><td>41.51</td><td>41.54</td><td>0.886</td><td>0.542</td></tr><tr><td>1</td><td>69.31</td><td>69.32</td><td>0.962</td><td>0.551</td><td>57.02</td><td>57.00</td><td>0.954</td><td>0.549</td></tr><tr><td>2</td><td>92.14</td><td>92.14</td><td>0.982</td><td>0.552</td><td>90.57</td><td>90.57</td><td>0.977</td><td>0.551</td></tr><tr><td>3</td><td>99.52</td><td>99.52</td><td>0.990</td><td>0.554</td><td>98.94</td><td>98.94</td><td>0.985</td><td>0.554</td></tr><tr><td>4</td><td>99.87</td><td>99.87</td><td>0.992</td><td>0.555</td><td>99.43</td><td>99.43</td><td>0.990</td><td>0.554</td></tr><tr><td>5</td><td>99.98</td><td>99.98</td><td>0.988</td><td>0.554</td><td>99.75</td><td>99.75</td><td>0.992</td><td>0.555</td></tr><tr><td>6</td><td>100.0</td><td>100.0</td><td>0.981</td><td>0.553</td><td>99.90</td><td>99.90</td><td>0.993</td><td>0.555</td></tr><tr><td>7</td><td>100.0</td><td>100.0</td><td>0.969</td><td>0.551</td><td>99.96</td><td>99.96</td><td>0.992</td><td>0.555</td></tr><tr><td>8</td><td>100.0</td><td>100.0</td><td>0.954</td><td>0.549</td><td>99.98</td><td>99.98</td><td>0.990</td><td>0.554</td></tr><tr><td>12</td><td>100.0</td><td>100.0</td><td>0.890</td><td>0.542</td><td>100.0</td><td>100.0</td><td>0.972</td><td>0.541</td></tr><tr><td>16</td><td>100.0</td><td>100.0</td><td>0.837</td><td>0.536</td><td>100.0</td><td>100.0</td><td>0.950</td><td>0.501</td></tr><tr><td>32</td><td>100.0</td><td>100.0</td><td>0.722</td><td>0.513</td><td>100.0</td><td>100.0</td><td>0.882</td><td>0.470</td></tr><tr><td>64</td><td>100.0</td><td>100.0</td><td>0.716</td><td>0.474</td><td>100.0</td><td>100.0</td><td>0.896</td><td>0.447</td></tr></table>

(b) Results with AdamW
<table><tr><td rowspan="2"> $\gamma$ </td><td colspan="4">CE with AdamW</td><td colspan="4">BCE with AdamW</td></tr><tr><td> $\boldsymbol { A }$ </td><td> $A ^ { * }$ </td><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td><td> $\boldsymbol { A }$ </td><td> $A ^ { * }$ </td><td> $\varepsilon _ { \mathrm { c o m } }$ </td><td> $\mathcal { E } _ { \mathrm { s e p } }$ </td></tr><tr><td>0.1</td><td>23.45</td><td>23.47</td><td>0.791</td><td>0.519</td><td>20.57</td><td>20.58</td><td>0.760</td><td>0.513</td></tr><tr><td>0.5</td><td>65.21</td><td>65.24</td><td>0.961</td><td>0.551</td><td>24.47</td><td>24.35</td><td>0.769</td><td>0.519</td></tr><tr><td>1</td><td>79.29</td><td>79.28</td><td>0.983</td><td>0.553</td><td>35.28</td><td>35.25</td><td>0.762</td><td>0.513</td></tr><tr><td>2</td><td>99.23</td><td>99.23</td><td>0.991</td><td>0.555</td><td>76.80</td><td>76.82</td><td>0.905</td><td>0.542</td></tr><tr><td>3</td><td>99.79</td><td>99.79</td><td>0.997</td><td>0.555</td><td>98.93</td><td>98.93</td><td>0.988</td><td>0.554</td></tr><tr><td>4</td><td>99.98</td><td>99.98</td><td>0.997</td><td>0.555</td><td>99.48</td><td>99.48</td><td>0.993</td><td>0.555</td></tr><tr><td>5</td><td>100.0</td><td>100.0</td><td>0.997</td><td>0.555</td><td>99.80</td><td>99.80</td><td>0.997</td><td>0.555</td></tr><tr><td>6</td><td>100.0</td><td>100.0</td><td>0.997</td><td>0.555</td><td>99.92</td><td>99.92</td><td>0.997</td><td>0.555</td></tr><tr><td>7</td><td>100.0</td><td>100.0</td><td>0.995</td><td>0.555</td><td>99.98</td><td>99.98</td><td>0.997</td><td>0.555</td></tr><tr><td>8</td><td>100.0</td><td>100.0</td><td>0.991</td><td>0.555</td><td>100.0</td><td>100.0</td><td>0.997</td><td>0.555</td></tr><tr><td>12</td><td>100.0</td><td>100.0</td><td>0.959</td><td>0.550</td><td>100.0</td><td>100.0</td><td>0.995</td><td>0.553</td></tr><tr><td>16</td><td>100.0</td><td>100.0</td><td>0.920</td><td>0.544</td><td>100.0</td><td>100.0</td><td>0.989</td><td>0.500</td></tr><tr><td>32</td><td>100.0</td><td>100.0</td><td>0.795</td><td>0.520</td><td>100.0</td><td>100.0</td><td>0.940</td><td>0.422</td></tr><tr><td>64</td><td>100.0</td><td>100.0</td><td>0.728</td><td>0.369</td><td>100.0</td><td>100.0</td><td>0.876</td><td>0.311</td></tr></table>

## B.7 Experimental verification for Theorems 5 and 6

In Sec. 4.2, we theoretically analyze how the convergence rate of unbiased positive and negative decision scores changes as γ linearly increases or decreases, which reflect the convergence behavior of the CE and BCE losses. Specifically, (1) when $\gamma$ decreases linearly toward 0, the convergence rate of the losses decays linearly, meaning that the number of training epochs required to reach their minima increases linearly; (2) when CC I or II is satisfied, as $\gamma$ increases linearly, the convergence rate of the losses decays exponentially, meaning that the number of training epochs required to reach their minima increases exponentially.

To directly verify the above conclusions, we set the NC structure at the minimum point as the stopping criterion, and train ResNet18 on MNIST with varying gamma. In these experiments, the stopping criterion is set as

$$
\begin{array} { r l } & { \left| w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - 1 \right| \leq t , \quad \forall k \in [ K ] , i \in [ n ] , } \\ & { \left| w _ { j } ^ { \top } h _ { i } ^ { ( k ) } + \cfrac { 1 } { K - 1 } \right| \leq t , \quad \forall j \neq k \in [ K ] , i \in [ n ] . } \end{array}\tag{B.1}
$$

When $\gamma < 5$ , we set $t = 0 . 0 0 1$ ; when $\gamma \geq 5 ,$ , we set $t = 0 . 0 0 3 5$ . Moreover, we use SGD optimizer and fix the learning rate at 0.01. Table 1 and Figure B.13 show the epoch numbers required by CE and BCE losses when they satisfying the criterion.

For cases with $\gamma < 0 . 5$ and $\gamma > 5$ , we took six and five data points to respectively fit the dependence of epoch number on the scale factor $\gamma$ using linear, quadratic, and cubic polynomial functions, as well as exponential and logarithmic functions in terms of $\textstyle { \frac { 1 } { \gamma } } \ ( \operatorname { f o r } \gamma < 0 . 5 )$ and $\gamma \left( \mathrm { f o r } \ \gamma > 1 \right)$ . The remaining data points were used to evaluate the predication quality of the fitted functions, and the evaluate metrics are coeficient of determination $R ^ { 2 }$ and root mean squared error (RMSE)

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { \sum _ { i = 1 } ^ { n } ( y _ { i } - \bar { y } ) ^ { 2 } } ,\tag{B.2}
$$

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } ,\tag{B.3}
$$

where $\{ y _ { i } \} _ { i = 1 } ^ { n }$ are the observed true data points, $\{ \hat { y } _ { i } \} _ { i = 1 } ^ { n }$ are the predicted values by the fitted function, and $\bar { y }$ denotes the mean of the true values. Tables B.7 and B.8 show the fitting results, together with the metrics of $R ^ { 2 }$ and RMSE on the fitting and prediction data points.

It can be seen that when $\gamma < 0 . 5 ,$ a linear function in $1 / \gamma \ ( \mathrm { i . e . }$ , an inverse proportional function) already provides a very good fitting to the number of epochs required by CE and BCE loss to reach their minima. In particular, their predictions achieve very high $R ^ { 2 }$ and low RMSE. Although for BCE a quadratic polynomial in $1 / \gamma$ yields a better overall fitting, by examining its coeficients together with the $R ^ { 2 }$ and

![](images/0ca4ae60fbe5dc0e9be567c1e683167179a31559ec25e16482dca3b5207e1ed3.jpg)

![](images/b2262f7289f3d6e4b3d3c0ea667a5c025a8b3a5fe5c1b38f755492f9abaa5b5a.jpg)  
(a) fitted functions for the CE loss

![](images/cd9421171adc30150d707feb207c58773be17fbeb7cfaa63a5e00ac2660151e9.jpg)

![](images/6744268a53cf23afe8023e2038851f957c52ee5b19dc94bb858a1dccbd124039.jpg)  
(b) fitted functions for the BCE loss  
Fig. B.13: The fitted functions describing the variation of epoch number with respect to γ for CE (top) and BCE (bottom) losses. The epoch numbers in the figure indicate the training duration required to achieve minimum values (i.e., the criterion in Eq. (B.1)) of the two losses for training ResNet18 on MNIST, with diferent scale factors γ. The blue dots represent that the data was used for function fitting, the red dots represent the predicted values, and the green triangles represent the observed data in experiments

RMSE on both the fitted and predicted data, one can clearly observe that the inverseproportional term is the main factor determining how the required epoch number varies with γ.

For $\gamma > 5 ,$ , it can be observed that linear and logarithmic functions do not conform to the trend of epoch number varying with $\gamma ;$ consequently, their fitting results are poor. Compared to quadratic and cubic polynomials, the exponential function achieved superior fitting results, particularly on the prediction data. This is consistent with the conclusion in the paper that the convergence rate of the losses exhibits exponential decay as γ approaches infinity. Furthermore, it is noteworthy that as the polynomial order increases, the corresponding fitting results also improve progressively. Given that exponential functions can be expanded through Taylor series, the phenomenon is reasonable.

Table B.7: Fitted functions and their evaluation results for the epoch number used by the CE and BCE with $\gamma < 0 . 5$
<table><tr><td rowspan="2">Loss</td><td rowspan="2">Fitted function</td><td colspan="2">Fitting data</td><td colspan="2">Predicted data</td></tr><tr><td> $R ^ { 2 } \uparrow$ </td><td> $\mathrm { R M S E } \downarrow$ </td><td> $R ^ { 2 } \uparrow$ </td><td>RMSE↓</td></tr><tr><td rowspan="4">CE</td><td> $\overline { { { 2 7 . 1 4 } / { \gamma + 4 4 . 4 1 } } }$ </td><td>0.9987</td><td>32.93</td><td>0.9908</td><td>52.08</td></tr><tr><td> $- 0 . 0 6 / \gamma ^ { 2 } + 3 3 . 0 5 / \gamma - 7 . 5 2$ </td><td>0.9999</td><td>7.92</td><td>0.9897</td><td>55.16</td></tr><tr><td> $- 0 . 0 0 2 / \gamma ^ { 3 } + 0 . 1 6 / \gamma ^ { 2 } + 2 . 8 3 / \gamma + 1 . 2 0$ </td><td>1.0000</td><td>3.39</td><td>0.9638</td><td>103.11</td></tr><tr><td> $4 0 0 2 . 2 1 \exp ( - 4 5 . 0 8 \gamma ) + 1 8 5 . 1 1$   $- 7 1 0 . 6 9 \log ( 5 . 6 3 \gamma ) + 1 7 5 . 7 5$ </td><td>0.9943 0.8213</td><td>69.52</td><td>0.8388</td><td>217.72 422.27</td></tr><tr><td rowspan="4">BCE</td><td> $\overline { { 2 6 . 6 8 / \gamma + 6 1 . 5 0 } }$ </td><td>0.9973</td><td>389.56 47.11</td><td>0.3936 0.9930</td><td>47.16</td></tr><tr><td> $\cdot 0 . 0 8 / \gamma ^ { 2 } + 3 5 . 2 2 / \gamma - 1 3 . 4 3$ </td><td>0.9999</td><td>9.62</td><td>0.9937</td><td>44.55</td></tr><tr><td> $\cdot 0 . 0 0 1 / \gamma ^ { 3 } + 0 . 0 7 / \gamma ^ { 2 } + 3 . 1 7 / \gamma + 1 . 0 7$ </td><td>0.9999</td><td>8.02</td><td>0.9796</td><td>80.37</td></tr><tr><td> $3 8 4 9 . 4 6 \exp ( - 4 2 . 6 8 \gamma ) + 1 8 5 . 5 0$   $- 7 0 4 . 0 7 \log ( 5 . 3 3 \gamma ) + 1 4 7 . 9 5$ </td><td>0.9936 0.8326</td><td>72.78 370.97</td><td>0.8842 0.4346</td><td>191.46</td></tr></table>

Table B.8: Fitted functions and their evaluation results for the epoch number used by the CE and BCE with $\gamma > 5$
<table><tr><td rowspan="2">Loss</td><td rowspan="2">Fitted function</td><td colspan="2">Fitting data</td><td colspan="2">Predicted data</td></tr><tr><td> $R ^ { 2 } \uparrow$ </td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td><td>RMSE↓</td></tr><tr><td rowspan="4">CE</td><td> $\overline { { 5 6 8 . 3 9 \gamma - 3 6 4 5 . 6 8 } }$ </td><td>0.8556</td><td>299.08</td><td>-0.5945</td><td>317.58</td></tr><tr><td> $2 3 6 . 0 1 \gamma ^ { 2 } - 3 0 9 8 . 3 9 \gamma + 1 0 2 0 5 . 4 9$ </td><td>0.9939</td><td>61.59</td><td>0.7463</td><td>126.69</td></tr><tr><td> $4 0 . 6 5 \gamma ^ { 3 }$   $7 0 6 . 2 8 \gamma ^ { 2 } + 4 0 7 6 . 8 0 \gamma$  -7729.46</td><td>0.9969</td><td>43.85</td><td>0.9038</td><td>77.99</td></tr><tr><td> $0 . 5 4 \exp ( 0 . 8 7 \gamma ) - 6 1 . 8 3$   $4 . 2 2 \log ( 0 . 3 2 \gamma ) - 3 . 0 1$ </td><td>0.9932</td><td>64.98</td><td>0.9253</td><td>68.72</td></tr><tr><td rowspan="5">BCE</td><td rowspan="5"> $\overline { { 2 3 6 . 6 0 \gamma - 2 2 0 4 . 0 0 } }$   $5 2 . 1 4 \gamma ^ { 2 } - 1 0 1 4 . 8 3 \gamma + 4 8 8 7 . 4 3$   $7 . 6 1 \gamma ^ { 3 } - 2 2 1 . 9 8 \gamma ^ { 2 } + 2 1 7 1 . 1 1 \gamma - 7 0 2 7 . 8 7$ </td><td>0.8094</td><td>343.40</td><td>-0.9326</td><td>349.63</td></tr><tr><td>0.7714</td><td>364.33</td><td>0.5566</td><td>849.72</td></tr><tr><td>0.9812</td><td>104.54</td><td>0.8140</td><td>550.31</td></tr><tr><td>0.9996</td><td>15.55</td><td>0.8837</td><td>435.17</td></tr><tr><td>0.9999 0.6871</td><td>7.56 426.22</td><td>0.9128 0.4633</td><td>376.77 934.77</td></tr></table>

## B.8 Balanced Classification

To further show the potential of BCE loss, we provide the classification performance on balanced dataset here. Table B.9 shows the classification results of the models trained on the balanced datasets, CIFAR10, CIFAR100, and Tiny-ImageNet dataset (Wu et al., 2017). It can be observed that, the classification performance of models trained with BCE loss is comparable to or even surpassing that of models trained with CE loss.

When training ViT on the CIFAR10, CIFAR100, and Tiny-ImageNet datasets, its classification is significantly worse than that of the ResNets, which is intuitive and reasonable. As noted by H. Zhu et al. (2023) and Zhang et al. (2022), due to the influence of factors such as model architecture and inductive biases, the ViT trained on small datasets is hugely diferent from ViT trained on large datasets, which may be the reason why the performance drops a lot on these small datasets. Setting aside the inherent diferences between these models, we still find that training with the

Table B.9: The classification accuracy (%) on the test datasets of CIFAR10, CIFAR100 and Tiny-ImageNet. We conducted learning rate optimization for each experimental combination while keeping other hyperparameters fixed, and the $\gamma = 3 2$
<table><tr><td rowspan="2">M</td><td rowspan="2">Loss</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td><td colspan="2">Tiny-ImageNet</td></tr><tr><td>SGD</td><td>AdamW</td><td>SGD</td><td>AdamW</td><td>SGD</td><td>AdamW</td></tr><tr><td rowspan="2">ResNet18</td><td>CE</td><td>95.90</td><td>96.24</td><td>78.39</td><td>78.05</td><td>68.10</td><td>67.13</td></tr><tr><td>BCE</td><td>95.97</td><td>95.96</td><td>78.46</td><td>78.38</td><td>69.10</td><td>69.12</td></tr><tr><td rowspan="2">ResNet50</td><td>CE</td><td>96.30</td><td>96.51</td><td>80.89</td><td>78.06</td><td>71.65</td><td>69.49</td></tr><tr><td>BCE</td><td>96.40</td><td>96.24</td><td>80.87</td><td>79.89</td><td>72.95</td><td>70.82</td></tr><tr><td rowspan="2">ViT</td><td>CE</td><td>79.51</td><td>87.00</td><td>64.82</td><td>63.19</td><td>50.42</td><td>54.83</td></tr><tr><td>BCE</td><td>82.00</td><td>87.57</td><td>64.79</td><td>64.91</td><td>51.15</td><td>53.91</td></tr></table>

BCE loss can achieve performance on par with, or even surpassing, that of the CE loss. The BCE loss helps the model learn deep features with better compactness and separability, which is usually and intuitively beneficial for classification.

## B.9 Open-set Recognition Performance

In Sec. 5, we present the open-set recognition (OSR) performance of models trained by BCE and CE losses. To further analyze their diferences and the impact of classifier biases in practice, we conducted additional OSR experiments with and without the classifier biases, respectively setting the mean of initial classifier biases as 0, 5, and 10 when the classifier biases were included in the losses. Table B.10 presents the OSR results.

It can be observed that when the losses taking the classifier biases, regardless of their initial mean (0, 5, or 10), the BCE-trained models consistently outperforms the CE-trained ones in the OSR experiments. In contrast, without the biases, the OSR of BCE-trained models experiences a dramatic decline (over 8.92% degradation), compared with the CE-trained models, especially on MNIST and SVHN, which demonstrate again the crucial role of classifier biases in the BCE loss.

## Appendix C Proof of Theorems

In the normalized feature space, Theorems 1 and 2 have concluded the neural collapse of the CE and BCE losses on balanced datasets. In this section, we present their detail proofs.

## C.1 Basics

Before providing the proofs, we first formally present the definition of canonical Ksimplex ETF in d-dimension space, and two lemmas required by the proofs.

Table B.10: The results of open-set recognition. Results are averaged among five randomized trials and the scale factor $\gamma = 3 2$
<table><tr><td rowspan="2">Bias Mean</td><td rowspan="2">Metrics</td><td colspan="2">MNIST</td><td colspan="2">SVHN</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR+50</td></tr><tr><td>CE</td><td>BCE</td><td>CE</td><td>BCE</td><td>CE</td><td>BCE</td><td>CE</td><td>BCE</td></tr><tr><td rowspan="2"></td><td>AUROC</td><td>96.74</td><td>90.22</td><td>94.44</td><td>66.60</td><td>74.34</td><td>68.14</td><td>78.34</td><td>62.17</td></tr><tr><td>OSCR</td><td>96.59</td><td>90.04</td><td>92.56</td><td>71.00</td><td>60.10</td><td>67.00</td><td>61.22</td><td>63.87</td></tr><tr><td rowspan="2">0</td><td>AURÓĆ</td><td>98.63</td><td>99.29</td><td>94.42</td><td>95.21</td><td>83.86</td><td>85.70</td><td>87.82</td><td>90.06</td></tr><tr><td>OSCR</td><td>98.50</td><td>99.09</td><td>92.53</td><td>93.49</td><td>81.54</td><td>83.50</td><td>85.81</td><td>88.43</td></tr><tr><td rowspan="2">5</td><td>AURÓC</td><td>98.58</td><td>99.15</td><td>94.30</td><td>95.07</td><td>84.17</td><td>85.78</td><td>87.76</td><td>90.15</td></tr><tr><td>OSCR</td><td>98.46</td><td>98.96</td><td>92.42</td><td>93.37</td><td>81.87</td><td>83.66</td><td>85.71</td><td>88.51</td></tr><tr><td rowspan="2">10</td><td>AURÓC</td><td>98.66</td><td>99.21</td><td>94.49</td><td>95.07</td><td>84.02</td><td>85.51</td><td>87.55</td><td>89.72</td></tr><tr><td>OSCR</td><td>98.53</td><td>99.04</td><td>92.64</td><td>93.37</td><td>81.72</td><td>83.35</td><td>85.62</td><td>88.08</td></tr></table>

Definition C.1 (K-simplex ETF) A K-simplex ETF is a collection of vectors in $\mathbb { R } ^ { d }$ specified by the columns of matrix

$$
M = \sqrt { \frac { K } { K - 1 } } P \left( \pmb { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \right) ,\tag{C.1}
$$

where $\pmb { I } _ { K } \in \mathbb { R } ^ { K \times K }$ is the identity matrix and ${ \bf 1 } _ { K } \in \mathbb { R } ^ { K }$ is the ones vector, and $\textbf { \textit { P } } \in$ $\mathbb { R } ^ { d \times K } ( d \geq K )$ is a partial-orthogonal matrix such that $P ^ { \top } P = I _ { K }$

Note that the matrix M satisfies:

$$
\boldsymbol { M } ^ { \top } \boldsymbol { M } = \frac { K } { K - 1 } \left( \pmb { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \right) .\tag{C.2}
$$

Lemma C.2 (Young’s Inequality) Let $p , q$ be positive numbers satisfying $\begin{array} { r } { \frac { 1 } { p } + \frac { 1 } { q } = 1 } \end{array}$ . Then for any $a , b \in \mathbb { R }$ , we have

$$
| a b | \leq \frac { | a | ^ { p } } { p } + \frac { | b | ^ { q } } { q } ,\tag{C.3}
$$

where the equality holds if and only $i f | a | ^ { p } = | b | ^ { q }$ . The case for $p = q = 2$ is just the AM-GM inequality which is $| a b | \leq \frac { 1 } { 2 } ( a ^ { 2 } + b ^ { 2 } )$ , where the equality holds if and only $i f \left| a \right| = \left| b \right|$

Lemma C.3 (Jensen’s Inequality) Let f be a convex function on an interval I. Then

$$
f ( \lambda _ { 1 } x _ { 1 } + \cdot \cdot \cdot + \lambda _ { n } x _ { n } ) \leq \lambda _ { 1 } f ( x _ { 1 } ) + \cdot \cdot \cdot + \lambda _ { n } f ( x _ { n } ) ,\tag{C.4}
$$

where

$$
x _ { 1 } , x _ { 2 } , \cdots , x _ { n } \in I , \quad \lambda _ { 1 } , \lambda _ { 2 } , \cdots , \lambda _ { n } \in [ 0 , 1 ] .\tag{C.5}
$$

When f is strictly convex, let

$$
I _ { 1 } = \{ k : \lambda _ { k } \in ( 0 , 1 ] \} , \ a n d \ I _ { 2 } = \{ k : \lambda _ { k } = 0 \} .\tag{C.6}
$$

The equality holds if and only if all $x _ { k } , k \in I _ { 1 }$ are equal. In practice, if the function f is concave like exp(x), log(x), or log(1 + e<sup>x</sup>), Jensen’s Inequality can be written as

$$
\exp \left( \frac { \sum _ { i = 1 } ^ { n } x _ { i } } { n } \right) \leq \frac { \sum _ { i = 1 } ^ { n } \exp ( x _ { i } ) } { n } ,\tag{C.7}
$$

$$
\log \left( \frac { \sum _ { i = 1 } ^ { n } x _ { i } } { n } \right) \geq \frac { \sum _ { i = 1 } ^ { n } \log ( x _ { i } ) } { n } ,\tag{C.8}
$$

$$
\log \left( 1 + \exp \left( \frac { \sum _ { i = 1 } ^ { n } x _ { i } } { n } \right) \right) \geq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log \left( 1 + \exp ( x _ { i } ) \right) .\tag{C.9}
$$

## C.2 Proof of Theorem 2

Theorem C.4 Assume that the feature dimension d is larger than the number of classes $K _ { i }$ $i . e . , d \ge K { - } 1$ , and $\gamma > 0$ . In the normalized feature space, any global minimizer $( W ^ { \star } , H ^ { \star } , b ^ { \star } )$ of

$$
f _ { \mathrm { b c e } } ( W , H , b ) \triangleq \frac { 1 } { n K } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \mathcal { L } _ { \mathrm { b c e } } ( \gamma W h _ { i } ^ { ( k ) } - b )\tag{C.10}
$$

with $\| \pmb { w } _ { j } \| _ { 2 } = \| \pmb { h } _ { i } ^ { ( k ) } \| _ { 2 } = 1 f o r \forall j , k \in [ K ] , i \in [ n ] , o b e y s$

$$
\pmb { w } _ { k } ^ { \star } = \pmb { h } _ { i } ^ { ( k ) \star } , \quad \forall k \in [ K ] , \ i \in [ n ] ,\tag{C.11}
$$

$$
\tilde { \pmb { h } } _ { i } ^ { \star } : = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \pmb { h } _ { i } ^ { ( k ) \star } = \mathbf { 0 } , \quad \forall i \in [ n ] ,\tag{C.12}
$$

$$
\boldsymbol { W } ^ { \star } \boldsymbol { W } ^ { \star \top } = \frac { K } { K - 1 } \left( \boldsymbol { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \right) ,\tag{C.13}
$$

$$
\boldsymbol { b } ^ { \star } = \boldsymbol { b } ^ { \star } \mathbf { 1 } ,\tag{C.14}
$$

where

$$
b ^ { \star } = \log \frac { ( K - 2 ) \mathrm { e } ^ { - \frac { \gamma } { \kappa - 1 } } + \sqrt { ( K - 2 ) ^ { 2 } \mathrm { e } ^ { - \frac { 2 \gamma } { \kappa - 1 } } + 4 ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma } { \kappa - 1 } + \gamma } } } { 2 } .\tag{C.15}
$$

Proof For the function $f _ { \mathrm { b c e } } ( W , H , b )$ , according to Lemma C.6,

$$
f _ { \mathrm { b c e } } ( W , { \pmb H } , { \pmb b } ) \geq - 2 \gamma - ( K - 2 ) { \pmb b } ^ { \star } + \log \big ( 1 + \mathrm e ^ { \gamma - { \pmb b } ^ { \star } } \big ) + ( K - 1 ) \log \Big ( 1 + \mathrm e ^ { { \pmb b } ^ { \star } + { \frac { \gamma } { K - 1 } } } \Big ) .\tag{C.16}
$$

According to Lemma C.7, the inequality (C.16) achieves its equality when Eqs. (C.11), (C.12), (C.13), (C.14), and (C.15) hold. For any fixed $\gamma > 0$ and K, the right side of the inequality (C.16) is a constant. Therefore, when $f _ { \mathrm { b c e } } ( W , H , b )$ reaching this constant, it gets its minimum. □

Lemma C.5 For any ${ h _ { i } ^ { ( k ) } }$ with $c _ { 1 } , c _ { 2 } > 0$ and $\gamma > 0$ , the normalized BCE loss is lower bounded by

$$
\mathcal { L } _ { \mathrm { b c e } } \big ( \gamma W h _ { i } ^ { ( k ) } - b \big ) \geq \frac { 1 } { 1 + c _ { 1 } } \Big ( - \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } + b _ { k } \Big ) + \frac { 1 } { 1 + c _ { 2 } } \sum _ { \stackrel { j = 1 } { j \neq k } } ^ { K } \Big ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } \Big ) + C ,\tag{C.17}
$$

where

$$
C = \log ( 1 + c _ { 1 } ) - \frac { c _ { 1 } } { 1 + c _ { 1 } } \log ( c _ { 1 } ) + ( K - 1 ) \left[ \log ( 1 + c _ { 2 } ) - \frac { c _ { 2 } } { 1 + c _ { 2 } } \log ( c _ { 2 } ) \right] .\tag{C.18}
$$

The inequality becomes an equality if and only if

$$
\begin{array} { r } { c _ { 1 } = \exp \Big ( \gamma \pmb { w } _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { k } \Big ) , \quad \forall k \in [ K ] , } \end{array}\tag{C.19}
$$

$$
c _ { 2 } = \exp \Big ( b _ { j } - \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } \Big ) , \quad \forall j \neq k \in [ K ] .\tag{C.20}
$$

Proof By the concavities of log(x), and according to Lemma $\mathrm { C . 3 } { } _ { ; }$ the normalized BCE loss can be lower bounded by

(C.21)

$$
\begin{array} { r l } & { \quad \int _ { \mathrm { S } _ { \tau } } \langle x , y , N _ { t } ^ { ( 1 ) } , \hat { y } , y \rangle } \\ & { = | w _ { 0 } \rangle \Big ( 1 + \mathrm { e } ^ { \mathrm { i } \tau } \langle y , N _ { t } ^ { ( 1 ) } , \hat { y } , \hat { w } , \hat { y } , \hat { y } \rangle - \mathrm { i } \frac { w _ { 0 } ^ { \prime } } { \sqrt { 2 } } \Big ) \times \mathrm { e } ^ { \mathrm { i } \tau } \Big ( 1 - \mathrm { e } ^ { \mathrm { i } \tau } \langle y , N _ { t } ^ { ( 1 ) } , \hat { y } , \hat { b } , \hat { y } \rangle \Big ) } \\ & { \quad - | w _ { 0 } \rangle \Big ( 1 - \mathrm { e } ^ { \mathrm { i } \tau } \langle x , y , N _ { t } ^ { ( 1 ) } , \hat { y } , \hat { y } , \hat { y } , \hat { y } \rangle + \mathrm { i } \frac { \mathrm { e } ^ { \mathrm { i } \tau } \langle x , y , N _ { t } ^ { ( 1 ) } , \hat { y } , \hat { y } \rangle } { \sqrt { 2 } } \Big ) } \\ & { \quad \quad - | w _ { 0 } \rangle \Big ( 1 - \mathrm { e } ^ { \mathrm { i } \tau } \langle x , \hat { y } , \hat { x } , \hat { x } , \hat { y } , \hat { y } , \hat { y } \rangle + \mathrm { i } \frac { \mathrm { e } ^ { \mathrm { i } \tau } \langle x , y , N _ { t } ^ { ( 1 ) } , \hat { y } , \hat { y } \rangle } { \sqrt { 2 } } \Big ) } \\ &  \quad \quad \quad + \frac { \mathrm { e } ^ { \mathrm { i } \tau } \sqrt { 2 } } { 2 } \mathrm { e } ^ { \mathrm { i } \tau } \Big ( \frac { 1 + \mathrm { e } ^ { \mathrm { i } \tau } } { 1 + \mathrm { e } ^ { \mathrm { i } \tau } \sqrt { 2 } } + \frac  1 - \mathrm { e } ^ { \mathrm { i } \tau } \langle x , y , N _ { t } ^ { ( 1 ) } , \hat { y } , \hat { y } , \hat  y  \end{array}\tag{C.22}
$$

(C.23)

(C.24)

The inequality is derived from the concavity of log(x), which achieves the equality if and only if

$$
\frac { 1 + c _ { 1 } } { c _ { 1 } } = \left( 1 + c _ { 1 } \right) \exp \left( - \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } + b _ { k } \right) \mathrm { ~ o r ~ } c _ { 1 } = 0 \mathrm { ~ o r ~ } c _ { 1 } = + \infty , \mathrm { ~ a n d ~ }\tag{C.25}
$$

$$
\frac { 1 + c _ { 2 } } { c _ { 2 } } = ( 1 + c _ { 2 } ) \exp \left( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } \right) \mathrm { o r } c _ { 2 } = 0 \mathrm { o r } c _ { 2 } = + \infty , \quad \forall j \neq k \in [ K ] .\tag{C.26}
$$

However, these equalities are trivial when $c _ { 1 } = 0 ~ \mathrm { o r } ~ c _ { 1 } = + \infty$ and $c _ { 2 } = 0$ or $c _ { 2 } = + \infty$ Therefore, if and only if

$$
c _ { 1 } = \exp \left( \gamma \pmb { w } _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { k } \right) ,\tag{C.27}
$$

$$
c _ { 2 } = \exp \left( b _ { j } - \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } \right) , \quad \forall j \neq k \in [ K ] ,\tag{C.28}
$$

Eq. (C.17) becomes an equality.

Lemma C.6 For the function $f _ { \mathrm { b c e } } ( W , H , b )$ defined in $E q .$ (C.10),

$$
f _ { \mathrm { b c e } } ( W , { \pmb H } , { \pmb b } ) \geq - 2 \gamma - ( K - 2 ) { \pmb b } ^ { \star } + \log \big ( 1 + \mathrm { e } ^ { \gamma - { \pmb b } ^ { \star } } \big ) + ( K - 1 ) \log \Big ( 1 + \mathrm { e } ^ { { \pmb b } ^ { \star } + { \pmb \frac { \gamma } { K - 1 } } } \Big ) .\tag{C.29}
$$

Proof According to Lemma C.5, Eq. (C.17) holds for any $c _ { 1 } , c _ { 2 } \ > \ 0$ and any ${ h _ { i } ^ { ( k ) } }$ with $k \in [ K ] , i \in [ n ]$ . We take the same $c _ { 1 }$ and $c _ { 2 }$ for all ${ h } _ { i } ^ { ( k ) }$ with $\forall i \in [ n ] , \forall k \in [ K ]$ , then

$$
f _ { \mathrm { b c e } } ( W , H , b ) = \frac { 1 } { n K } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \mathcal { L } _ { \mathrm { b c e } } \left( \gamma W h _ { i } ^ { ( k ) } - b \right)\tag{C.30}
$$

(C.31)

$$
\begin{array} { r l } &  \begin{array} { r l } & { \frac { 1 } { 2 \omega _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial y _ { \mathrm { S } } ^ { 2 } } - \gamma \frac { 1 } { 2 \omega _ { \mathrm { S } } ^ { 2 } } \Delta x _ { \mathrm { S } } ^ { 2 } = \omega ^ { 2 } \Delta x _ { \mathrm { S } } ^ { 2 } + \omega ^ { 2 } - \frac { 1 } { 1 6 \omega _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial y _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 3 } } \omega ^ { 3 } \omega ^ { 3 } \omega ^ { 3 } \omega ^ { 3 } \omega ^ { 3 } } \\ & { - \frac { 1 } { 2 \omega _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 2 } } \left( \gamma \frac { 1 } { 2 \omega _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 2 } } - \gamma \frac { 1 } { 2 \omega _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 3 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 3 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 3 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 3 } } \right) \omega ^ { 3 } \omega ^ { 4 } } \\ &  - \frac { 1 } { 2 \omega _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 2 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 3 } } \frac { \partial } { \partial x _ { \mathrm { S } } ^ { 3 } } \end{array} \end{array}\tag{C.32}
$$

(C.33)

(C.34)

(C.35)

where in the last equality, $\begin{array} { r } { \tilde { \pmb { h } } _ { i } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \pmb { h } _ { i } ^ { ( j ) } } \end{array}$

For the function $\hat { f } _ { \mathrm { b c e } } ( W , H , b )$ , its global minimum must occur at its stationary points that satisfies the first-order necessary condition,

$$
0 = \frac { \partial \hat { f } _ { \mathrm { b c e } } } { \partial b _ { k } } = - \frac { 1 } { K } \bigg ( \frac { K - 1 } { 1 + c _ { 2 } } - \frac { 1 } { 1 + c _ { 1 } } \bigg ) , \quad k \in [ K ] .\tag{C.36}
$$

Therefore,

$$
\hat { f } _ { \mathrm { b c e } } ( { \cal W } , { \cal H } , b ) \geq \frac { \gamma } { n } \frac { 1 } { 1 + c _ { 2 } } \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { K } w _ { k } ^ { \top } \left( \tilde { h } _ { i } - h _ { i } ^ { ( k ) } \right) + C .\tag{C.37}
$$

Furthermore, according to the AM-GM inequality (see Lemma C.2), we have

$$
u ^ { \top } v \geq - \Big ( \frac { c } { 2 } \| u \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c } \| v \| _ { 2 } ^ { 2 } \Big ) , \quad \forall u , v \in \mathbb { R } ^ { d } , \ \forall c > 0 ,\tag{C.38}
$$

which becomes an equality when $\mathbf { \boldsymbol { v } } = - c \mathbf { \boldsymbol { u } }$ . Then,

$$
\begin{array} { c } { { \displaystyle f _ { \mathrm { b c e } } ( { \boldsymbol W } , { \boldsymbol H } , { \boldsymbol b } ) \geq \hat { f } _ { \mathrm { b c e } } ( { \boldsymbol W } , { \boldsymbol H } , { \boldsymbol b } ) \geq \frac { \gamma } { n } \frac { 1 } { 1 + c _ { 2 } } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { K } w _ { k } ^ { \top } \left( \tilde { h } _ { i } - h _ { i } ^ { ( k ) } \right) + C } } \\ { { \geq - \frac { \gamma } { n } \frac { 1 } { 1 + c _ { 2 } } \left( \frac { c _ { 3 } } { 2 } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { K } \| { \boldsymbol w } _ { k } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c _ { 3 } } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { K } \| \tilde { h } _ { i } - h _ { i } ^ { ( k ) } \| _ { 2 } ^ { 2 } \right) + C } } \end{array}\tag{C.39}
$$

$$
= - \frac { \gamma } { n } \frac { 1 } { 1 + c _ { 2 } } \bigg [ \frac { c _ { 3 } } { 2 } \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { K } \| { \pmb w } _ { k } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c _ { 3 } } \sum _ { i = 1 } ^ { n } \bigg ( \sum _ { k = 1 } ^ { K } \| { \pmb h } _ { i } ^ { ( k ) } \| _ { 2 } ^ { 2 } - K \| \tilde { \pmb h } _ { i } \| _ { 2 } ^ { 2 } \bigg ) \bigg ] + C\tag{C.40}
$$

$$
= - \frac { \gamma } { n } \frac { 1 } { 1 + c _ { 2 } } \bigg ( \frac { c _ { 3 } } { 2 } n K + \frac { 1 } { 2 c _ { 3 } } n K \bigg ) + \frac { \gamma } { n } \frac { 1 } { 1 + c _ { 2 } } \frac { 1 } { 2 c _ { 3 } } \sum _ { i = 1 } ^ { n } K \| \tilde { h } _ { i } \| _ { 2 } ^ { 2 } + C\tag{C.41}
$$

$$
\geq - \frac { \gamma } { n } \frac { 1 } { 1 + c _ { 2 } } \bigg ( \frac { c _ { 3 } } { 2 } n K + \frac { 1 } { 2 c _ { 3 } } n K \bigg ) + C .\tag{C.42}
$$

Inequality (C.39) becomes an equality if and only if

$$
c _ { 3 } { w } _ { k } = h _ { i } ^ { ( k ) } - \tilde { h } _ { i } , \forall k \in [ K ] , \quad i \in [ n ] , \ \gamma > 0 , \ \forall c _ { 3 } > 0 ,\tag{C.43}
$$

and, the inequality (C.42) becomes an equality if and only if

$$
\tilde { \pmb { h } } _ { i } = \mathbf { 0 } , \quad \forall i \in [ n ] .\tag{C.44}
$$

Eqs. (C.43) and (C.44) imply

$$
c _ { 3 } { \pmb w } _ { k } = { \pmb h } _ { i } ^ { ( k ) } \Rightarrow \| c _ { 3 } { \pmb w } _ { k } \| _ { 2 } ^ { 2 } = \| { \pmb h } _ { i } ^ { ( k ) } \| _ { 2 } ^ { 2 } \Rightarrow c _ { 3 } ^ { 2 } = 1 \Rightarrow c _ { 3 } = 1 .\tag{C.45}
$$

Combining Eqs. (C.42) and (C.45), we get

$$
f _ { \mathrm { b c e } } ( W , H , b ) \geq - \frac { \gamma } { n } \frac { 1 } { 1 + c _ { 2 } } \bigg ( \frac { c _ { 3 } } { 2 } n K + \frac { 1 } { 2 c _ { 3 } } n K \bigg ) + C = - \frac { \gamma K } { 1 + c _ { 2 } } + C ,\tag{C.46}
$$

which becomes an equality if and only if Eqs. (C.19), (C.20), (C.43) and (C.44) hold.

Furthermore, according to Lemma C.7, when f<sub>bce</sub>(W, H, b) achieves its lower bound, one can get $b _ { 1 } = b _ { 2 } = \cdot \cdot \cdot = b _ { k } = b ^ { \star }$ , and

$$
c _ { 1 } = \exp \left( \gamma - b ^ { \star } \right) \quad \mathrm { a n d } \quad c _ { 2 } = \exp \left( b ^ { \star } + \frac { \gamma } { K - 1 } \right) .\tag{C.47}
$$

Then, combining Eqs. (C.24), (C.36), (C.46) and (C.47), we have

$$
f _ { \mathrm { b c e } } ( W , H , b ) \geq - { \frac { \gamma K } { 1 + c _ { 2 } } } + C
$$

$$
= - \frac { \gamma K } { 1 + c _ { 2 } } + \log \left( 1 + c _ { 1 } \right) - \frac { c _ { 1 } } { 1 + c _ { 1 } } \log ( c _ { 1 } ) + ( K - 1 ) \biggl [ \log ( 1 + c _ { 2 } ) - \frac { c _ { 2 } } { 1 + c _ { 2 } } \log ( c _ { 2 } ) \biggr ] ,\tag{C.48}
$$

$$
= - 2 \gamma - ( K - 2 ) b ^ { \star } + \log \big ( 1 + \mathrm { e } ^ { \gamma - b ^ { \star } } \big ) + ( K - 1 ) \log \Big ( 1 + \mathrm { e } ^ { b ^ { \star } + \frac { \gamma } { K - 1 } } \Big ) ,\tag{C.49}
$$

which is desired.

Lemma C.7 Under the same assumption of Lemma $C . 6 ,$ the lower bound in Eq. (C.29) is achieved if and only if

$$
{ \pmb w } _ { k } = { \pmb h } _ { i } ^ { ( k ) } , \quad \forall k \in [ K ] , \ i \in [ n ] ,\tag{C.50}
$$

$$
\tilde { \pmb { h } } _ { i } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \pmb { h } _ { i } ^ { ( k ) } = \mathbf { 0 } , \quad \forall i \in [ n ] ,\tag{C.51}
$$

$$
\boldsymbol { W } \boldsymbol { W } ^ { \intercal } = \frac { K } { K - 1 } \left( \pmb { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \intercal } \right) ,\tag{C.52}
$$

$$
\boldsymbol { b } ^ { \star } = \boldsymbol { b } ^ { \star } \mathbf { 1 } ,\tag{C.53}
$$

$$
c _ { 1 } = \exp \left( \gamma - b ^ { \star } \right) \quad a n d \quad c _ { 2 } = \exp \left( b ^ { \star } + \frac { \gamma } { K - 1 } \right) ,\tag{C.54}
$$

where $b ^ { \star }$ satisfies

$$
b ^ { \star } = \log \frac { ( K - 2 ) \mathrm { e } ^ { - \frac { \gamma } { \kappa - 1 } } + \sqrt { ( K - 2 ) ^ { 2 } \mathrm { e } ^ { - \frac { 2 \gamma } { \kappa - 1 } } + 4 ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma } { \kappa - 1 } + \gamma } } } { 2 } .\tag{C.55}
$$

Proof According to the proof of Lemma C.6, when the loss function $f _ { \mathrm { b c e } }$ achieves its lower bound if and only if $E q s .$ . (C.19), (C.20), (C.43) and (C.44) hold. From Eqs. (C.43) and (C.44),

$$
{ \pmb w } _ { k } = { \pmb h } _ { i } ^ { ( k ) } , \quad \forall k \in [ K ] , \ i \in [ n ] ,\tag{C.56}
$$

$$
\quad { \mathrm { a n d } } \qquad \sum _ { k = 1 } ^ { K } w _ { k } = \sum _ { k = 1 } ^ { K } h _ { i } ^ { ( k ) } = \mathbf { 0 } .\tag{C.57}
$$

Then, we have

$$
c _ { 1 } = \exp \left( \gamma { \pmb w } _ { k } ^ { \top } { \pmb h } _ { i } ^ { ( k ) } - b _ { k } \right) = \exp \left( \gamma \| { \pmb w } _ { k } \| _ { 2 } ^ { 2 } - b _ { k } \right) , \quad \forall k \in [ K ] ,\tag{C.58}
$$

$$
c _ { 2 } = \exp \left( b _ { j } - \gamma { \pmb w } _ { j } ^ { \top } { \pmb h } _ { i } ^ { ( k ) } \right) = \exp \left( b _ { j } - \gamma { \pmb w } _ { k } ^ { \top } { \pmb w } _ { j } \right) , \quad \forall j \neq k \in [ K ] .\tag{C.59}
$$

Since the $c _ { 1 } , c _ { 2 } > 0$ are the same for diferent $j \neq k \in [ K ]$ , we have

$$
\gamma \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } - b _ { k } = \gamma \| \pmb { w } _ { j } \| _ { 2 } ^ { 2 } - b _ { j } , \quad \forall k , j \in [ K ] ,\tag{C.60}
$$

$$
\gamma w _ { k } ^ { \top } w _ { j } - b _ { j } = \gamma w _ { k } ^ { \top } w _ { \ell } - b _ { \ell } , \quad \forall j \neq \ell \in [ K ] , \ \forall k \in [ K ] .\tag{C.61}
$$

In the normalized feature space, $E q .$ (C.60) implies

$$
\begin{array} { r } { \gamma \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } - b _ { k } = \gamma \| \pmb { w } _ { j } \| _ { 2 } ^ { 2 } - b _ { j } \Rightarrow \gamma - b _ { k } = \gamma - b _ { j } \Leftrightarrow b _ { k } = b _ { j } , \quad \forall k , j \in [ K ] , } \end{array}\tag{C.62}
$$

Therefore, one can write ${ \pmb b } = { \pmb b } ^ { \star } { \bf 1 }$ for some $b \in \mathbb { R }$ . Then, from Eq. (C.61),

$$
\gamma w _ { k } ^ { \top } w _ { j } = \gamma w _ { k } ^ { \top } w _ { \ell } \Rightarrow w _ { k } ^ { \top } w _ { j } = w _ { k } ^ { \top } w _ { \ell } , \quad \forall j \neq \ell \in [ K ] , \quad \forall k \in [ K ] ,\tag{C.63}
$$

and combining with Eq. (C.57),

$$
\begin{array} { r l r } & { } & { 1 = \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } = - \displaystyle \sum _ { j = 1 \atop j \not \in k } ^ { K } \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = - \left( K - 1 \right) \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } } \\ & { } & \\ & { } & { \Rightarrow \pmb { w } _ { k } ^ { \top } \pmb { w } _ { j } = - \frac { 1 } { K - 1 } , \quad \forall j \not = k \in [ K ] . } \end{array}\tag{C.64}
$$

Therefore,

$$
\boldsymbol { W } \boldsymbol { W } ^ { \intercal } = \frac { K } { K - 1 } \left( \pmb { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \intercal } \right) .\tag{C.65}
$$

Plugging Eqs. (C.62), (C.64) into Eqs. (C.58), (C.59),

$$
c _ { 1 } = \exp ( \gamma - b ^ { \star } ) \quad \mathrm { a n d } \quad c _ { 2 } = \exp \bigg ( b ^ { \star } + \frac { \gamma } { K - 1 } \bigg ) .\tag{C.66}
$$

When $f _ { \mathrm { b c e } }$ defined in $E q .$ (C.10) achieves its minimum value, it theoretically satisfies

$$
0 = \frac { \partial f _ { \mathrm { b c e } } } { \partial b _ { k } } = \frac { 1 } { n K } \left( n - \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \Big ( 1 + \exp \left( - \gamma w _ { k } ^ { \top } h _ { i } ^ { ( j ) } + b _ { k } \right) \Big ) ^ { - 1 } \right) , \quad \forall k \in [ K ] ,\tag{C.67}
$$

thus, combining Eqs. (C.62) and (C.64),

$$
n - \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \left( 1 + \exp \left( - \gamma \pmb { w } _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( j ) } + b _ { k } \right) \right) ^ { - 1 }
$$

$$
= n - \sum _ { i = 1 } ^ { n } \left( { \frac { 1 } { 1 + \exp ( - \gamma + b ) } } + \sum _ { j \neq k } ^ { K } { \frac { 1 } { 1 + \exp ( { \frac { \gamma } { K - 1 } } + b ) } } \right)\tag{C.68}
$$

$$
= n - n \left( { \frac { 1 } { 1 + \exp ( - \gamma + b ) } } + { \frac { K - 1 } { 1 + \exp ( { \frac { \gamma } { K - 1 } } + b ) } } \right) = 0 ,\tag{C.69}
$$

then,

$$
b ^ { \star } = \log \frac { ( K - 2 ) \mathrm { e } ^ { - \frac { \gamma } { \kappa - 1 } } + \sqrt { ( K - 2 ) ^ { 2 } \mathrm { e } ^ { - \frac { 2 \gamma } { \kappa - 1 } } + 4 ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma } { \kappa - 1 } + \gamma } } } { 2 } ,\tag{C.70}
$$

which satisfies $\operatorname { E q . }$ (C.36).

Lemma C.8 When the class number $K > 2$ , the scale factor $\gamma > 0$ , and

$$
\log { ( 2 K - 3 ) } ( 1 - \frac { 1 } { K } ) < \gamma ,\tag{C.71}
$$

the final critical bias $\boldsymbol { b } ^ { \star }$ could separate the all positive unbiased decision scores

$$
\left\{ \gamma \pmb { w } _ { k } ^ { \star \top } \pmb { h } _ { i } ^ { ( k ) \star } : ~ k \in [ K ] , ~ i \in [ n ] \right\} ,\tag{C.72}
$$

and the all negative ones

$$
\left\{ \gamma w _ { j } ^ { \star \top } h _ { i } ^ { ( k ) \star } : ~ k , j \in [ K ] , ~ i \in [ n ] , ~ k \neq j \right\} .\tag{C.73}
$$

Proof According to Lemma $\mathrm { C } . 7 ,$ when the function $f _ { \mathrm { b c e } }$ achieves the lower bound, we have

$$
\gamma { \pmb w } _ { k } ^ { \star \top } { \pmb h } _ { i } ^ { ( k ) \star } = \gamma , \quad \forall k \in [ K ] , \ i \in [ n ] ,\tag{C.74}
$$

$$
{ \gamma } { \pmb w } _ { j } ^ { \star \top } { \pmb h } _ { i } ^ { ( k ) \star } = - \frac { \gamma } { K - 1 } , \quad \forall j \neq k \in [ K ] , \ i \in [ n ] ,\tag{C.75}
$$

Let $\begin{array} { r } { b _ { \mathrm { p o s } } = \gamma , b _ { \mathrm { n e g } } = - \frac { \gamma } { K - 1 } } \end{array}$ . Then the critical $b ^ { \star }$ separating the all positive and negative decision scores means

$$
b _ { \mathrm { n e g } } = - \frac { \gamma } { K - 1 } < b ^ { \star } < \gamma = b _ { \mathrm { p o s } } .\tag{C.76}
$$

Due to

$$
\begin{array} { r l r } { \quad } & { - \frac { \gamma } { K - 1 } < b ^ { \star } \Leftrightarrow 2 \mathrm { ~ e } ^ { - \frac { \gamma } { K - 1 } } < \left( K - 2 \right) \mathrm { e } ^ { - \frac { \gamma } { K - 1 } } + \sqrt { \left( K - 2 \right) ^ { 2 } } \mathrm { e } ^ { - \frac { 2 \gamma } { K - 1 } } + 4 \left( K - 1 \right) \mathrm { e } ^ { - \frac { \gamma } { K - 1 } + \gamma } } & \\ { \quad } & { \quad } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad } & { \quad } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad } & { \quad } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad } & { \quad } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad ( - 1 ) \mathrm { e } ^ { - \frac { \gamma } { K - 1 } + \gamma } } & \\ { \quad } & { \quad } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad } & { \quad } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad } &  \end{array}\tag{7}
$$

$$
\begin{array} { r l } { \quad b ^ { \star } < \gamma \Leftrightarrow ( K - 2 ) \mathrm { e } ^ { - \frac { \gamma } { K - 1 } } + \sqrt { ( K - 2 ) ^ { 2 } } \mathrm { e } ^ { - \frac { 2 \gamma } { K - 1 } } + 4 ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma } { K - 1 } + \gamma } < 2 \mathrm { e } ^ { \gamma } } & { } \\ { \quad \Leftrightarrow ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma } { K - 1 } } < \mathrm { e } ^ { \gamma } - ( K - 2 ) \mathrm { e } ^ { - \frac { \gamma } { K - 1 } } } & { } \\ { \quad \Leftrightarrow \log \left( 2 K - 3 \right) ( 1 - \frac { 1 } { K } ) < \gamma } & { \Leftarrow \quad \log \left( 2 K - 3 \right) < \gamma } \\ { \quad \Leftrightarrow K < \frac { \mathrm { e } ^ { \gamma } + 3 } { 2 } , } \end{array}\tag{C.78}
$$

which completes the proof.

## C.3 Proof of Theorem 1

Theorem C.9 Assume that the feature dimension d is larger than the number of classes $K , \ i . e . , \ d \geq K - 1$ , and the scale factor $\gamma > 0$ . In the normalized feature space, any global minimizer $( W ^ { \star } , H ^ { \star } , b ^ { \star } )$ of

$$
\operatorname* { m i n } _ { W , H , b } f _ { \mathrm { c e } } ( W , H , b ) : = \frac { 1 } { n K } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \mathcal { L } _ { \mathrm { c e } } ( \gamma W h _ { i } ^ { ( k ) } - b )\tag{C.79}
$$

with $\| \pmb { w } _ { j } \| = \| \pmb { h } _ { i } ^ { ( k ) } \| = 1$ , obeys the following

$$
\pmb { w } _ { k } ^ { \star } = \pmb { h } _ { i } ^ { ( k ) \star } , \quad \forall k \in [ K ] , i \in [ n ] ,\tag{C.80}
$$

$$
\tilde { \pmb { h } } _ { i } ^ { \star } : = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \pmb { h } _ { i } ^ { ( k ) \star } = \mathbf { 0 } , \quad \forall i \in [ n ] ,\tag{C.81}
$$

$$
\boldsymbol { W } ^ { \star } \boldsymbol { W } ^ { \star \top } = \frac { K } { K - 1 } \left( \boldsymbol { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \top } \right) ,\tag{C.82}
$$

$$
\begin{array} { r } { \boldsymbol { b } ^ { \star } = b \mathbf { 1 } , \quad \forall b \in \mathbb { R } . } \end{array}\tag{C.83}
$$

Proof According to Lemma C.11, for any fixed $\gamma > 0 .$ , we have

$$
f _ { \mathrm { c e } } ( W , H , b ) \geq \log \Big ( 1 + ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma K } { K - 1 } } \Big ) .\tag{C.84}
$$

According to Lemma C.12, the inequality (C.84) achieves its equality when Eqs. (C.80, C.81, C.82, C.83) hold, which finishes the proof. □

Lemma C.10 For any ${ h _ { i } ^ { ( k ) } }$ with $c _ { 1 } ^ { \prime } > 0$ and $\gamma > 0$ , the normalized CE loss is lower bounded by

$$
\mathcal { L } _ { \mathrm { c e } } ( \gamma W h _ { i } ^ { ( k ) } - b ) \geq \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } + C ^ { \prime } ,\tag{C.85}
$$

where

$$
C ^ { \prime } = \frac { c _ { 1 } ^ { \prime } } { 1 + c _ { 1 } ^ { \prime } } \log \left( \frac { 1 + c _ { 1 } ^ { \prime } } { c _ { 1 } ^ { \prime } } \right) + \frac { 1 } { 1 + c _ { 1 } ^ { \prime } } \log \left[ \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) \right] .\tag{C.86}
$$

The inequality becomes an equality when

$$
c _ { 1 } ^ { \prime } = \left[ ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) \right] ^ { - 1 } ,\tag{C.87}
$$

and

$$
\gamma { \pmb w } _ { j } ^ { \top } { \pmb h } _ { i } ^ { ( k ) } - b _ { j } = \gamma { \pmb w } _ { \ell } ^ { \top } { \pmb h } _ { i } ^ { ( k ) } - b _ { \ell } , \ \forall j , \ell \neq k \in [ K ] .\tag{C.88}
$$

Proof According to Jensen’s inequality in Lemma C.3, the normalized CE loss can be lower bounded,

$$
\mathcal { L } _ { \mathrm { c e } } ( \gamma W h _ { i } ^ { ( k ) } - b ) = \log \left( 1 + \sum _ { \stackrel { j = 1 } { j \neq k } } ^ { K } \frac { \exp \left( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } \right) } { \exp \left( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } \right) } \right)\tag{C.89}
$$

$$
\begin{array} { r l } & { = \log \left( 1 + ( K - 1 ) \displaystyle \sum _ { j = 1 } ^ { K } \frac { \exp \left( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } \right) } { ( K - 1 ) \exp \left( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } \right) } \right) } \\ & { \geq \log \left( 1 + ( K - 1 ) \exp \left( \displaystyle \sum _ { j = 1 } ^ { K } \frac { \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } - \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } + b _ { k } } { K - 1 } \right) \right) } \end{array}\tag{C.90}
$$

(C.91)

$$
= \log \left( 1 + ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) \right) ,\tag{C.92}
$$

which achieves the equality if and only $\mathrm { i f } \gamma \pmb { w } _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { j } = \gamma \pmb { w } _ { \ell } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { \ell }$ for all $j , \ell \neq k \in [ K ]$ Second, by the concavity of the function l $\log ( 1 + \mathrm { e } ^ { x } )$ , for any $c _ { 1 } ^ { \prime } > 0$ , we get

$$
\geq \log \left( 1 + ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) \right)\tag{C.93}
$$

$$
= \log \left( \frac { c _ { 1 } ^ { \prime } } { 1 + c _ { 1 } ^ { \prime } } \frac { 1 + c _ { 1 } ^ { \prime } } { c _ { 1 } ^ { \prime } } + \frac { 1 + c _ { 1 } ^ { \prime } } { 1 + c _ { 1 } ^ { \prime } } \left( K - 1 \right) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) \right)\tag{C.94}
$$

$$
\begin{array} { r l } & { \geq \frac { c _ { 1 } ^ { \prime } } { 1 + c _ { 1 } ^ { \prime } } \log \left( \frac { 1 + c _ { 1 } ^ { \prime } } { c _ { 1 } ^ { \prime } } \right) } \\ & { \quad + \frac { 1 } { 1 + c _ { 1 } ^ { \prime } } \log \left( \left( 1 + c _ { 1 } ^ { \prime } \right) ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) \right) } \end{array}\tag{C.95}
$$

$$
\begin{array} { r l } & { = \cfrac { 1 } { 1 + c _ { 1 } ^ { \prime } } \cfrac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } } \\ & { \quad + \underbrace { \frac { c _ { 1 } ^ { \prime } } { 1 + c _ { 1 } ^ { \prime } } \log \left( \frac { 1 + c _ { 1 } ^ { \prime } } { c _ { 1 } ^ { \prime } } \right) + \frac { 1 } { 1 + c _ { 1 } ^ { \prime } } \log \left[ \left( 1 + c _ { 1 } \right) ( K - 1 ) \right] } _ { C ^ { \prime } } . } \end{array}\tag{C.96}
$$

The last inequality achieves its equality if and only if

$$
\frac { 1 + c _ { 1 } ^ { \prime } } { c _ { 1 } ^ { \prime } } = \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) ,\tag{C.97}
$$

$$
\mathrm { o r } c _ { 1 } ^ { \prime } = 0 \mathrm { o r } c _ { 1 } ^ { \prime } = + \infty .\tag{C.98}
$$

Actually, when $c _ { 1 } ^ { \prime } = 0$ or $c _ { 1 } ^ { \prime } = + \infty$ , the equality is trivial. Therefore, we have

$$
c _ { 1 } ^ { \prime } = \left[ ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) \right] ^ { - 1 } ,\tag{C.99}
$$

as desired.

Lemma C.11 For the function $f _ { \mathrm { c e } } ( W , H , b )$ defined in $E q .$ . (C.79),

$$
f _ { \mathrm { c e } } ( W , H , b ) \geq \log \Big ( 1 + ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma K } { K - 1 } } \Big ) .\tag{C.100}
$$

Proof According to Lemma C.10, Eq. (C.85) holds for any $c _ { 1 } ^ { \prime } > 0$ and any ${ h } _ { i } ^ { ( k ) }$ with $k \in$ $[ K ] , \ i \in [ n ]$ . We take the same $c _ { 1 } ^ { \prime }$ for all ${ h _ { i } ^ { ( k ) } }$ , then we have the following lower bound for the function $f _ { \mathrm { c e } }$ as

$$
\big ( 1 + c _ { 1 } ^ { \prime } \big ) ( K - 1 ) [ f _ { \mathrm { c e } } ( W , H , b ) - C ]\tag{C.101}
$$

$$
= \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) \left[ \frac { 1 } { n K } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \mathcal { L } _ { \mathrm { c e } } ( \gamma W h _ { i } ^ { ( k ) } - b ) - C \right]\tag{C.102}
$$

$$
\geq \frac { 1 } { n K } \sum _ { k = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \left[ \sum _ { j = 1 } ^ { K } \left( \gamma w _ { j } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { j } \right) - K \left( \gamma w _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { k } \right) \right]\tag{C.103}
$$

$$
= \frac { 1 } { n K } \sum _ { i = 1 } ^ { n } \left[ \left( \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { K } \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - K \sum _ { k = 1 } ^ { K } \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } \right) + \underbrace { \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { K } \left( b _ { k } - b _ { j } \right) } _ { 0 } \right]\tag{C.104}
$$

$$
= \frac { 1 } { n K } \sum _ { i = 1 } ^ { n } \left( \sum _ { k = 1 } ^ { K } \sum _ { j = 1 } ^ { K } \gamma \pmb { w } _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( j ) } - K \sum _ { k = 1 } ^ { K } \gamma \pmb { w } _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } \right)\tag{C.105}
$$

$$
\begin{array} { r l } & { = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { K } \left[ \gamma w _ { k } ^ { \top } \left( \frac { 1 } { K } \sum _ { j = 1 } ^ { K } h _ { i } ^ { ( j ) } - h _ { i } ^ { ( k ) } \right) \right] } \\ & { = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { K } \gamma w _ { k } ^ { \top } \left( \tilde { h } _ { i } - h _ { i } ^ { ( k ) } \right) , } \end{array}\tag{C.106}
$$

(C.107)

where $\begin{array} { r } { \tilde { \pmb { h } } _ { i } = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \pmb { h } _ { i } ^ { ( j ) } } \end{array}$

Then, according to the AM-GM inequality (see Lemma C.2), we have

$$
{ \boldsymbol { u } } ^ { \top } { \boldsymbol { v } } \leq \frac { c } { 2 } \| { \boldsymbol { u } } \| _ { 2 } ^ { 2 } + \frac { 1 } { 2 c } \| { \boldsymbol { v } } \| _ { 2 } ^ { 2 } , \ \forall { \boldsymbol { u } } , { \boldsymbol { v } } \in \mathbb { R } ^ { d } , \ \forall c > 0 ,\tag{C.108}
$$

which becomes an equality when ${ \pmb v } = c { \pmb u }$ . Then, based on Eq. (C.107), we get

$$
\begin{array} { r l } & { \quad \left( 1 + c _ { 1 } ^ { ( 1 ) } \left( K - 1 \right) \left[ f _ { \infty } ( W , H , b ) - C ^ { \prime } \right] \right. } \\ & { \geq \frac { 1 } { n } \displaystyle \sum _ { i = 1 } ^ { n } \displaystyle \sum _ { k = 1 } ^ { K } \gamma w _ { k } ^ { \top } \left( \tilde { h } _ { i } - h _ { i } ^ { ( k ) } \right) } \\ & { \geq - \frac { c _ { 2 } ^ { \prime } } { 2 } \displaystyle \sum _ { k = 1 } ^ { K } \| \gamma w _ { k } \| ^ { 2 } - \frac { 1 } { 2 c _ { 2 } ^ { \prime } n } \displaystyle \sum _ { i = 1 } ^ { n } \sum _ { k = 1 } ^ { K } \| \tilde { h } _ { i } - h _ { i } ^ { ( k ) } \| _ { 2 } ^ { 2 } } \\ & { = - \frac { c _ { 2 } ^ { \prime } } { 2 } \displaystyle \sum _ { k = 1 } ^ { K } \gamma ^ { 2 } \| w _ { k } \| ^ { 2 } - \frac { 1 } { 2 c _ { 2 } ^ { \prime } n } \displaystyle \sum _ { i = 1 } ^ { n } \left[ \displaystyle \sum _ { k = 1 } ^ { K } \| h _ { i } ^ { ( k ) } \| _ { 2 } ^ { 2 } - K \| \tilde { h } _ { i } \| _ { 2 } ^ { 2 } \right] } \\ & { = - \frac { c _ { 2 } ^ { \prime } } { 2 } \gamma ^ { 2 } K - \frac { 1 } { 2 c _ { 2 } ^ { \prime } n } \left( n K - K \displaystyle \sum _ { i = 1 } ^ { K } \| \tilde { h } _ { i } \| _ { 2 } ^ { 2 } \right) , } \end{array}\tag{C.109}
$$

(C.110)

(C.111)

where the second inequality becomes an equality if and only if

$$
c _ { 2 } ^ { \prime } \gamma { w _ { k } } = { h } _ { i } ^ { ( k ) } - \tilde { h } _ { i } , \forall k \in [ K ] , i \in [ n ] , \gamma > 0 .\tag{C.112}
$$

Therefore,

$$
f _ { \mathrm { c e } } ( W , H , b ) \geq - \frac { c _ { 2 } ^ { \prime } \gamma ^ { 2 } K } { 2 \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } - \frac { K } { 2 c _ { 2 } ^ { \prime } n \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } \left( n - \sum _ { i = 1 } ^ { n } \| \tilde { h } _ { i } \| _ { 2 } ^ { 2 } \right) + C ^ { \prime }\tag{C.113}
$$

$$
\ge - \frac { c _ { 2 } ^ { \prime } \gamma ^ { 2 } K } { 2 \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } - \frac { K } { 2 c _ { 2 } ^ { \prime } \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } + C ^ { \prime }\tag{C.114}
$$

$$
= - \ : \frac { K } { 2 \ : \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } \ : \left( c _ { 2 } ^ { \prime } \gamma ^ { 2 } + \frac { 1 } { c _ { 2 } ^ { \prime } } \right) + C ^ { \prime } ,\tag{C.115}
$$

where the second inequality becomes an equality if and only if

$$
\tilde { \pmb { h } } _ { i } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \pmb { h } _ { i } ^ { ( j ) } = \mathbf { 0 } , \forall i \in [ n ] .\tag{C.116}
$$

Based on Eqs. (C.112) and (C.116), we get

$$
c _ { 2 } ^ { \prime } \gamma w _ { k } = h _ { i } ^ { ( k ) } \Rightarrow \| c _ { 2 } ^ { \prime } \gamma w _ { k } \| _ { 2 } ^ { 2 } = \| h _ { i } ^ { ( k ) } \| _ { 2 } ^ { 2 } \Rightarrow ( c _ { 2 } ^ { \prime } \gamma ) ^ { 2 } = 1 \Rightarrow c _ { 2 } ^ { \prime } = \frac { 1 } { \gamma } .\tag{C.117}
$$

Plugging $E q .$ (C.117) into (C.115), we have

$$
f _ { \mathrm { c e } } ( W , H , b ) \geq - \frac { K ( c _ { 2 } ^ { \prime } \gamma ^ { 2 } + \frac { 1 } { c _ { 2 } ^ { \prime } } ) } { 2 \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } + C ^ { \prime } = - \frac { \gamma K } { \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } + C ^ { \prime } .\tag{C.118}
$$

Furthermore, according to Lemma C.12, when the objective function $f _ { \mathrm { c e } }$ achieves its lower bound, we have $\begin{array} { r } { \frac { \partial f _ { \mathrm { c e } } } { \partial b _ { k } } \equiv 0 } \end{array}$ , and $c _ { 1 } ^ { \prime }$ satisfies

$$
{ c _ { 1 } ^ { \prime } } = \left[ \left( K - 1 \right) \exp \left( - \frac { \gamma K } { K - 1 } \right) \right] ^ { - 1 } .\tag{C.119}
$$

Then, we have

$$
f _ { \mathrm { c e } } ( W , \pmb { H } , \pmb { b } ) \geq - \frac { \gamma K } { \left( 1 + c _ { 1 } ^ { \prime } \right) \left( K - 1 \right) } + C ^ { \prime } = \log \Big ( 1 + ( K - 1 ) \mathrm { e } ^ { - \frac { \gamma K } { K - 1 } } \Big ) ,\tag{C.120}
$$

as suggested in $E q .$ (C.100).

Lemma C.12 Under the same assumption of Lemma C.11, the lower bound in Eq. (C.100) is achieved for any critical point $( W , H , b )$ of Eq. (C.79) if and only if the following hold

$$
{ \pmb w } _ { k } = { \pmb h } _ { i } ^ { ( k ) } , \quad \forall k \in [ K ] , \ i \in [ n ] ,\tag{C.121}
$$

$$
\tilde { \pmb { h } } _ { i } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \pmb { h } _ { i } ^ { ( k ) } = \mathbf { 0 } , \quad \forall i \in [ n ] ,\tag{C.122}
$$

$$
\boldsymbol { W } \boldsymbol { W } ^ { \intercal } = \frac { K } { K - 1 } \left( \pmb { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \intercal } \right) ,\tag{C.123}
$$

$$
\begin{array} { r } { \boldsymbol { b } ^ { \star } = \boldsymbol { b } ^ { \star } \mathbf { 1 } , \quad \forall \boldsymbol { b } ^ { \star } \in \mathbb { R } , } \end{array}\tag{C.124}
$$

$$
{ c _ { 1 } ^ { \prime } } = \left[ \left( K - 1 \right) \exp \left( - \frac { \gamma K } { K - 1 } \right) \right] ^ { - 1 } .\tag{C.125}
$$

Proof With the proof of Lemma C.11, to achieve the lower bound, we need at least $E q s$ (C.112) and (C.116) to hold, $i . e ,$

$$
\tilde { \boldsymbol { h } } _ { i } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \boldsymbol { h } _ { i } ^ { ( k ) } = \mathbf { 0 } , \ \forall i \in [ n ] \quad \mathrm { a n d } \quad \boldsymbol { w } _ { k } = \boldsymbol { h } _ { i } ^ { ( k ) } , \ \forall k \in [ K ] , \ i \in [ n ] ,\tag{C.126}
$$

which further implies that

$$
\sum _ { k = 1 } ^ { K } w _ { k } = \sum _ { k = 1 } ^ { K } h _ { i } ^ { ( k ) } = 0 .\tag{C.127}
$$

Then,

$$
\sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) = \gamma \left( \sum _ { j = 1 } ^ { K } w _ { j } ^ { \top } \right) h _ { i } ^ { ( k ) } - \sum _ { j = 1 } ^ { K } b _ { j } = 0 - \sum _ { j = 1 } ^ { K } b _ { j } = - K \bar { b } ,\tag{C.128}
$$

where $\begin{array} { r } { \bar { b } = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } b _ { j } } \end{array}$ , and

$$
K ( \gamma \pmb { w } _ { k } ^ { \top } \pmb { h } _ { i } ^ { ( k ) } - b _ { k } ) = K \left( \gamma \| \pmb { w } _ { k } \| _ { 2 } ^ { 2 } - b _ { k } \right) = K \gamma - K b _ { k } .\tag{C.129}
$$

Combining Eqs. (C.87), (C.88), (C.128), and (C.129), we have

$$
\begin{array} { r l } & { c _ { 1 } ^ { \prime } = \left[ ( K - 1 ) \exp \left( \frac { \sum _ { j = 1 } ^ { K } ( \gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } ) - K ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( k ) } - b _ { k } ) } { K - 1 } \right) \right] ^ { - 1 } } \\ & { \quad = \left[ ( K - 1 ) \exp \left( \frac { - K \bar { b } - ( K \gamma - K b _ { k } ) } { K - 1 } \right) \right] ^ { - 1 } } \\ & { \quad = \left[ ( K - 1 ) \exp \left( \frac { K } { K - 1 } \left( b _ { k } - \bar { b } - \gamma \right) \right) \right] ^ { - 1 } . } \end{array}\tag{C.130}
$$

Since the $c _ { 1 } ^ { \prime } > 0$ can be arbitrary number, we choose the same $c _ { 1 } ^ { \prime }$ for all $k \in [ K ]$ . Then, we have

$$
\begin{array} { r } { b _ { k } - \bar { b } - \gamma = b _ { j } - \bar { b } - \gamma \Leftrightarrow b _ { k } = b _ { j } , \forall j \neq k \in [ K ] . } \end{array}\tag{C.131}
$$

Therefore, we can write $\mathbf { \delta } \mathbf { b } = b \mathbf { 1 } _ { K }$ . Moreover, plugging Eqs. (C.126) and (C.131) into $E q .$ (C.129), we get

$$
\gamma w _ { j } ^ { \top } h _ { i } ^ { ( k ) } - b _ { j } = \gamma w _ { \ell } ^ { \top } h _ { i } ^ { ( k ) } - b _ { \ell } \Leftrightarrow w _ { j } ^ { \top } h _ { i } ^ { ( k ) } = w _ { \ell } ^ { \top } h _ { i } ^ { ( k ) } \Leftrightarrow w _ { j } ^ { \top } w _ { k } = w _ { \ell } ^ { \top } w _ { k } .\tag{C.132}
$$

Then, combining with $E q .$ (C.127), we have

$$
1 = \| { \pmb w } _ { k } \| _ { 2 } ^ { 2 } = - \sum _ { j = 1 \atop j \neq k } ^ { K } { \pmb w } _ { k } ^ { \top } { \pmb w } _ { j } = - \left( K - 1 \right) { \pmb w } _ { k } ^ { \top } { \pmb w } _ { j } \Rightarrow { \pmb w } _ { k } ^ { \top } { \pmb w } _ { j } = - \frac { 1 } { K - 1 } , \forall j \neq k \in [ K ] .\tag{C.133}
$$

Therefore,

$$
\boldsymbol { W } \boldsymbol { W } ^ { \intercal } = \frac { K } { K - 1 } \left( \pmb { I } _ { K } - \frac { 1 } { K } \mathbf { 1 } _ { K } \mathbf { 1 } _ { K } ^ { \intercal } \right) .\tag{C.134}
$$

Finally, plugging the results in $E q .$ (C.131) into (C.130), we have

$$
c _ { 1 } ^ { \prime } = \left[ \left( K - 1 \right) \exp \left( \frac { K } { K - 1 } \left( b _ { k } - \bar { b } - \gamma \right) \right) \right] ^ { - 1 } = \left[ \left( K - 1 \right) \exp \left( - \frac { \gamma K } { K - 1 } \right) \right] ^ { - 1 } .
$$

When $f _ { \mathrm { c e } }$ defined in $E q .$ (C.79) achieves its lower bound, it theoretically satisfies

$$
\frac { \partial f _ { \mathrm { c e } } } { \partial b _ { k } } = \frac { 1 } { n K } \left( n - \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \frac { \exp ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( j ) } - b _ { k } ) } { \sum _ { \ell = 1 } ^ { K } \exp ( \gamma w _ { \ell } ^ { \top } h _ { i } ^ { ( j ) } - b _ { \ell } ) } \right) = 0 , \ \forall k , \ell \in [ K ] .\tag{C.135}
$$

However, combining with Eqs. (C.131) and (C.133), we have

$$
\begin{array} { r l } & { \quad n - \displaystyle \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \frac { \exp ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( j ) } - b _ { k } ) } { \sum _ { \ell = 1 } ^ { K } \exp ( \gamma w _ { \ell } ^ { \top } h _ { i } ^ { ( j ) } - b _ { \ell } ) } } \\ & { = n - \displaystyle \sum _ { j = 1 } ^ { K } \sum _ { i = 1 } ^ { n } \frac { \exp ( \gamma w _ { k } ^ { \top } h _ { i } ^ { ( j ) } - b _ { k } ) } { \sum _ { \ell \neq j } ^ { K } \exp ( - \frac { \gamma } { K - 1 } - b _ { \ell } ) + \exp ( \gamma - b _ { j } ) } } \\ & { = n - \displaystyle \sum _ { i = 1 } ^ { n } \left( \frac { \sum _ { j \neq k } ^ { K } \exp ( - \frac { \gamma } { K - 1 } - b ) + \exp ( \gamma - b ) } { ( K - 1 ) \exp ( - \frac { \gamma } { K - 1 } - b ) + \exp ( \gamma - b ) } \right) = n - n \equiv 0 , } \end{array}\tag{C.136}
$$

(C.137)

which means when the function $f _ { \mathrm { c e } }$ achieves its lower bound, the classifier biases $\boldsymbol { b } ^ { \star }$ only needs to satisfies $\pmb { b } ^ { \star } = b ^ { \star } \mathbf { 1 }$ . In other words, in terms of the classifier biases, the function $f _ { \mathrm { c e } }$ has infinitely many minima. □

## C.4 Proof of Theorems 5 and 6

Lemma C.13 (1) For a polynomial $P ( \gamma )$ and Sigmoid function σ $\begin{array} { r } { ( s \gamma - b ) = \frac { 1 } { 1 + \exp ( - s \gamma + b ) } } \end{array}$ with $s , b \in \mathbb { R }$ , as γ approaches positive infinity, their product exhibits exponential decay, converging to zero,

$$
\operatorname* { l i m } _ { \gamma \to + \infty } P ( \gamma ) \sigma ( s \gamma - b ) = \operatorname* { l i m } _ { \gamma \to + \infty } P ( \gamma ) \frac { 1 } { 1 + \exp ( - s \gamma + b ) } = 0 , \quad \forall s < 0 , \ b \in \mathbb { R } ,\tag{C.138}
$$

$$
\operatorname* { l i m } _ { \gamma \to + \infty } P ( \gamma ) \big ( 1 - \sigma ( s \gamma - b ) \big ) = \operatorname* { l i m } _ { \gamma \to + \infty } P ( \gamma ) \frac { \exp ( - s \gamma + b ) } { 1 + \exp ( - s \gamma + b ) } = 0 , \quad \forall s > 0 , \ b \in \mathbb { R } .\tag{C.139}
$$

$\frac { ( \mathcal { Q } ) \ F o r \ a \ p o } { 1 + \exp ( - s \gamma + b ) }$ lynomial $P ( \gamma )$ without constant term and Sigmoid function $\sigma ( s \gamma \mathrm { ~ - ~ } b ) \ =$ with $s , b \in \mathbb { R }$ , their product approaches 0 as γ approaches zero, i.e.,

$$
\operatorname* { l i m } _ { \gamma \to 0 } P ( \gamma ) \sigma ( s \gamma - b ) = \operatorname* { l i m } _ { \gamma \to 0 } P ( \gamma ) \frac { 1 } { 1 + \exp ( - s \gamma + b ) } = 0 , \quad \forall s , b \in \mathbb { R } ,\tag{C.140}
$$

$$
\operatorname* { l i m } _ { \gamma \to 0 } P ( \gamma ) \big ( 1 - \sigma ( s \gamma - b ) \big ) = \operatorname* { l i m } _ { \gamma \to 0 } P ( \gamma ) \frac { \exp ( - s \gamma + b ) } { 1 + \exp ( - s \gamma + b ) } = 0 , \forall s , b \in \mathbb { R } .\tag{C.141}
$$

Proof (1) Suppose that $P ( \gamma )$ is an p-degree polynomial in terms of $\gamma ,$ with the leading coeficient being $a _ { p }$ . Then by applying L’hospital’s rule p times, we have

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { \gamma \to + \infty } P ( \gamma ) \big ( 1 - \sigma \big ( s \gamma - b \big ) \big ) = \operatorname* { l i m } _ { \gamma \to + \infty } P ( \gamma ) \frac { \exp \left( - s \gamma + b \right) } { 1 + \exp \left( - s \gamma + b \right) } } \\ { = \displaystyle \operatorname* { l i m } _ { \gamma \to + \infty } \frac { P ( \gamma ) } { 1 + \exp \left( s \gamma - b \right) } = \operatorname* { l i m } _ { \gamma \to + \infty } \frac { \mathrm { d } P ( \gamma ) / \mathrm { d } \gamma } { s \exp \left( s \gamma - b \right) } = \cdots } \\ { = \displaystyle \operatorname* { l i m } _ { \gamma \to + \infty } \frac { \mathrm { d } ^ { p } P ( \gamma ) / \mathrm { d } \gamma ^ { p } } { s ^ { p } \exp \left( s \gamma - b \right) } = \operatorname* { l i m } _ { \gamma \to + \infty } \frac { p ! a _ { p } } { s ^ { p } \exp \left( s \gamma - b \right) } = 0 , \quad \forall s > 0 . } \end{array}\tag{C.142}
$$

When $s < 0$ , one can similarly prove Eq. (C.138).

(2) As $P ( \gamma )$ is without constant term, lim $_ { 1 \gamma \to 0 } P ( \gamma ) = 0$ . Therefore,

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { \gamma  0 } P ( \gamma ) \sigma ( s \gamma - b ) = \operatorname* { l i m } _ { \gamma  0 } P ( \gamma ) \operatorname* { l i m } _ { \gamma  0 } \sigma ( s \gamma - b ) = 0 \cdot \operatorname* { l i m } _ { \gamma  0 } \frac { 1 } { 1 + \exp ( - s \gamma + b ) } = 0 \cdot \frac { 1 } { 1 + \mathrm { e } ^ { b } } = 0 , \ } \\ { \displaystyle \operatorname* { l i m } _ { \gamma  0 } P ( \gamma ) \big ( 1 - \sigma ( s \gamma - b ) \big ) = \operatorname* { l i m } _ { \gamma  0 } P ( \gamma ) \operatorname* { l i m } _ { \gamma  0 } \big ( 1 - \sigma ( s \gamma - b ) \big ) } \\ { = 0 \cdot \operatorname* { l i m } _ { \gamma  0 } \frac { \exp \big ( - s \gamma + b \big ) } { 1 + \exp ( - s \gamma + b ) } = 0 \cdot \frac { \mathrm { e } ^ { b } } { 1 + \mathrm { e } ^ { b } } = 0 , \ } \end{array}
$$

which are desired.

Lemma C.14 Let $\{ s _ { j } \} _ { j = 1 } ^ { K } \subset \mathbb { R }$ and $\{ b _ { j } \} _ { j = 1 } ^ { K } \subset \mathbb { R }$ be two sets of real numbers and

$$
s _ { k } = \operatorname* { m a x } \{ s _ { j } \} _ { j = 1 } ^ { K } > \operatorname* { m a x } \{ s _ { j } \} _ { { \stackrel { j = 1 } { j \neq k } } } ^ { K } .\tag{C.143}
$$

(1) For a polynomial $P ( \gamma )$ and Softmax function Softmax $\begin{array} { r } { \ _ { \ell } ( \gamma ) \ = \ \frac { \exp ( s _ { \ell } \gamma - b _ { \ell } ) } { \sum _ { i = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) } } \end{array}$ , as γ approaches positive infinity, their product exhibits exponential decay, converging to zero,

$$
\operatorname* { l i m } _ { \gamma \to + \infty } P ( \gamma ) \mathtt { S o f t m a x } _ { \ell } ( \gamma ) = \operatorname* { l i m } _ { \gamma \to + \infty } P ( \gamma ) \frac { \exp ( s _ { \ell } \gamma - b _ { \ell } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) } = 0 , \qquad \forall \ell \neq k ,\tag{C.144}
$$

$$
\operatorname* { l i m } _ { \gamma \to + \infty } \pmb { P } ( \gamma ) \big ( 1 - \mathrm { S o f t m a x } _ { k } ( \gamma ) \big ) = \operatorname* { l i m } _ { \gamma \to + \infty } \pmb { P } ( \gamma ) \frac { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) } = 0 .\tag{C.145}
$$

(2) For a polynomial $P ( \gamma )$ without constant term and Softmax $\begin{array} { r } { ( \gamma ) = \frac { \exp ( s _ { \ell } \gamma - b _ { \ell } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) } } \end{array}$ , their product approaches 0 as γ approaches 0, and

$$
\operatorname* { l i m } _ { \gamma \to 0 } P ( \gamma ) \mathrm { S o f t m a x } _ { \ell } ( \gamma ) = \operatorname* { l i m } _ { \gamma \to 0 } P ( \gamma ) \frac { \exp ( s _ { \ell } \gamma - b _ { \ell } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) } = 0 , \qquad \forall j \neq k ,\tag{C.146}
$$

$$
\operatorname* { l i m } _ { \gamma \to 0 } P ( \gamma ) \big ( 1 - \mathtt { S o f t m a x } _ { k } ( \gamma ) \big ) = \operatorname* { l i m } _ { \gamma \to 0 } P ( \gamma ) \frac { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) } = 0 .\tag{C.147}
$$

Proof (1) Suppose $\pmb { P } ( \gamma ) = a _ { p } \gamma ^ { p } + a _ { p - 1 } \gamma ^ { p - 1 } + \cdot \cdot \cdot + a _ { 1 } \gamma + a _ { 0 }$ . Then,

$$
0 \leq \left| P ( \gamma ) \mathrm { { S o f t m a x } } _ { \ell } ( \gamma ) \right| \leq \sum _ { i = 1 } ^ { p } | a _ { i } | \gamma ^ { i } \frac { \exp ( s _ { \ell } \gamma - b _ { \ell } ) } { \sum _ { j = 1 } ^ { K } \exp ( s _ { j } \gamma - b _ { j } ) }
$$

$$
\leq \sum _ { i = 1 } ^ { p } | a _ { i } | \gamma ^ { i } { \frac { \exp ( s _ { \ell } \gamma - b _ { \ell } ) } { \exp ( s _ { k } \gamma - b _ { k } ) } } = \sum _ { i = 1 } ^ { p } | a _ { i } | \gamma ^ { i } { \frac { 1 } { \mathrm { e } ^ { ( s _ { k } - s _ { \ell } ) \gamma - ( b _ { k } - b _ { \ell } ) } } } .\tag{C.148}
$$

As $s _ { k } - s _ { \ell } > 0$ for $\ell \neq k ,$ the right side approaches zero when γ approaches positive infinity, according to Lemma $\mathrm { C . 1 3 ( 1 ) }$ ). Therefore, lim $_ { \gamma \to 0 } P ( \gamma )$ Softma $\mathfrak { z } ( \gamma ) = 0$

By applying the arithmetic operations of limits,

$$
\operatorname* { l i m } _ { \gamma  + \infty } P ( \gamma ) \big ( 1 - \mathtt { S o f t m a x } _ { k } ( \gamma ) \big ) = \sum _ { \stackrel { \ell = 1 } { \ell \neq k } } ^ { K } \operatorname* { l i m } _ { \gamma  + \infty } P ( \gamma ) \frac { \exp \big ( s _ { \ell } \gamma - b _ { \ell } \big ) } { \sum _ { j = 1 } ^ { K } \exp \big ( s _ { j } \gamma - b _ { j } \big ) } = \sum _ { \stackrel { \ell = 1 } { \ell \neq k } } ^ { K } 0 = 0 .\tag{C.149}
$$

(2) As $P ( \gamma )$ is without constant term, lim $_ { 1 \gamma \to 0 } P ( \gamma ) = 0$ . Therefore,

$$
0 \leq \operatorname* { l i m } _ { \gamma \to 0 } \left| P ( \gamma ) \mathrm { S o f t m a x } _ { \ell } ( \gamma ) \right| \leq \operatorname* { l i m } _ { \gamma \to 0 } \left| P ( \gamma ) \right| \operatorname* { l i m } _ { \gamma \to 0 } \left| \mathrm { S o f t m a x } _ { \ell } ( \gamma ) \right| \leq 0 \cdot 1 = 0 \quad \mathrm { a n d }
$$

$$
0 \leq \operatorname* { l i m } _ { \gamma \to 0 } \left| P ( \gamma ) \big ( 1 - \mathtt { S o f t m a x } _ { \ell } ( \gamma ) \big ) \right| \leq \operatorname* { l i m } _ { \gamma \to 0 } \left| P ( \gamma ) \right| \operatorname* { l i m } _ { \gamma \to 0 } \left| 1 - \mathtt { S o f t m a x } _ { \ell } ( \gamma ) \right| \leq 0 \cdot 1 = 0 ,
$$

which lead to the conclusions.

Theorem C.15 When training the model using the BCE loss $f _ { b c e } ,$ as γ approaches zero, the linear decrease in $\gamma$ leads to a linear decay in the convergence rate of unbiased decision scores. In contrary, once the critical condition II is satisfied, as $\gamma$ linearly approaches positive infinity, the convergence rate of unbiased decision scores decay exponentially.

Proof In the model training, for any unbiased decision score $s _ { i } ^ { ( k j ) } = \gamma w _ { k } ^ { \top } h _ { i } ^ { ( j ) }$ , its update step is determined by that of the classifier vector $\mathbf { \boldsymbol { w } } _ { k }$ and the feature $h _ { i } ^ { ( j ) }$ . For an unconstrained feature model (UFM), $\{ w _ { k } \} _ { k }$ and $\{ h _ { i } ^ { ( j ) } \} _ { i , j }$ are independent variables. Therefore, with an iteration, they are updated as

$$
\hat { w } _ { k } = w _ { k } - \eta \frac { \partial f _ { \mu } } { \partial w _ { k } } \quad \mathrm { a n d } \quad \hat { h } _ { i } ^ { ( j ) } = h _ { i } ^ { ( j ) } - \eta \frac { \partial f _ { \mu } } { \partial h _ { i } ^ { ( j ) } } ,\tag{C.150}
$$

where η is the learning rate and $\mu \in \{ \mathrm { c e } , \mathrm { b c e } \}$ . Then, the unbiased decision score is updated as

$$
\hat { s } _ { i } ^ { ( k j ) } = \gamma \hat { w } _ { k } ^ { \top } \hat { h } _ { i } ^ { ( j ) } = \gamma \bigg ( w _ { k } - \eta \frac { \partial f _ { \mu } } { \partial w _ { k } } \bigg ) ^ { \top } \bigg ( h _ { i } ^ { ( j ) } - \eta \frac { \partial f _ { \mu } } { \partial h _ { i } ^ { ( j ) } } \bigg )\tag{C.151}
$$

$$
= \gamma w _ { k } ^ { \top } h _ { i } ^ { ( j ) } - \gamma \eta w _ { k } ^ { \top } \frac { \partial f _ { \mu } } { \partial h _ { i } ^ { ( j ) } } - \gamma \eta \Big ( \frac { \partial f _ { \mu } } { \partial w _ { k } } \Big ) ^ { \top } h _ { i } ^ { ( j ) } + \gamma \eta ^ { 2 } \Big ( \frac { \partial f _ { \mu } } { \partial w _ { k } } \Big ) ^ { \top } \frac { \partial f _ { \mu } } { \partial h _ { i } ^ { ( j ) } } ,\tag{C.152}
$$

and its update step is

$$
\triangle ( s _ { i } ^ { ( k j ) } ) = s _ { i } ^ { ( k j ) } - \hat { s } _ { i } ^ { ( k j ) } = - \gamma \eta w _ { k } ^ { \top } \frac { \partial f _ { \mu } } { \partial h _ { i } ^ { ( j ) } } - \gamma \eta \Big ( \frac { \partial f _ { \mu } } { \partial w _ { k } } \Big ) ^ { \top } h _ { i } ^ { ( j ) } + \gamma \eta ^ { 2 } \Big ( \frac { \partial f _ { \mu } } { \partial w _ { k } } \Big ) ^ { \top } \frac { \partial f _ { \mu } } { \partial h _ { i } ^ { ( j ) } } .\tag{C.153}
$$

For the BCE loss,

$$
\frac { \partial f _ { \mathrm { b c e } } } { \partial h _ { i } ^ { ( j ) } } = \frac { \gamma } { n K } \Bigg ( - \frac { \mathrm { e } ^ { - \gamma w _ { j } ^ { \top } h _ { i } ^ { ( j ) } + b _ { j } } } { 1 + \mathrm { e } ^ { - \gamma w _ { j } ^ { \top } h _ { i } ^ { ( j ) } + b _ { j } } } w _ { j } + \sum _ { \ell = 1 } ^ { K } \frac { \mathrm { e } ^ { \gamma w _ { \ell } ^ { \top } h _ { i } ^ { ( j ) } - b _ { \ell } } } { 1 + \mathrm { e } ^ { \gamma w _ { \ell } ^ { \top } h _ { i } ^ { ( j ) } - b _ { \ell } } } w _ { \ell } \Bigg ) ,\tag{C.154}
$$

$$
\frac { \partial f _ { \mathrm { b c e } } } { \partial w _ { k } } = \frac { \gamma } { n K } \sum _ { i ^ { \prime } = 1 } ^ { n } \Bigg ( - \frac { \mathrm { e } ^ { - \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( k ) } + b _ { k } } } { 1 + \mathrm { e } ^ { - \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( k ) } + b _ { k } } } h _ { i ^ { \prime } } ^ { ( k ) } + \sum _ { \ell ^ { \prime } = 1 } ^ { K } \frac { \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } - b _ { k } } } { 1 + \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } - b _ { k } } } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } \Bigg ) ,\tag{C.155}
$$

and

$$
\begin{array} { r l } { \Delta Q _ { 1 } ^ { ( 1 ) } = - \frac { 2 ^ { 2 } \Delta _ { 1 } ^ { 2 } } { 2 } \Bigg ( \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } } & { = - \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } } \\ & { - \frac { 2 ^ { 2 } \Delta _ { 1 } ^ { 2 } } { 2 } \sin ^ { 2 } \theta + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } } \\ & { - \frac { 2 ^ { 2 } \Delta _ { 1 } ^ { 2 } } { 2 } \sin ^ { 2 } \theta + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } \Bigg ) ^ { 2 } \Bigg \{ \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } } \\ & { \quad + \frac { 2 ^ { 2 } \Delta _ { 1 } ^ { 2 } } { 2 } \Bigg ( \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } \Bigg ) ^ { 2 } \Bigg \{ \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } } \\ &  \quad + \frac { 2 ^ { 2 } \Delta _ { 1 } ^ { 2 } } { 2 } \Bigg ( \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } + \frac { \sin ^ { 2 } \theta } { \sin ^ { 2 } \theta } \Bigg ) ^ { 2 } \Bigg \{ \frac  \sin ^ { 2 } \ \end{array}
$$

$$
\begin{array} { r l } & { - \displaystyle \sum _ { i ^ { \prime } = 1 } ^ { n } \sum _ { \ell ^ { \prime } = 1 } ^ { K } \frac { \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } } - b _ { k } } { 1 + \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } - b _ { k } } } \frac { \mathrm { e } ^ { - \gamma w _ { j } ^ { \top } h _ { i } ^ { ( j ) } + b _ { j } } } { 1 + \mathrm { e } ^ { - \gamma w _ { j } ^ { \top } h _ { i } ^ { ( j ) } + b _ { j } } } w _ { j } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } } \\ & { + \displaystyle \sum _ { i ^ { \prime } = 1 } ^ { n } \sum _ { \ell ^ { \prime } = 1 } ^ { K } \sum _ { \ell = 1 } ^ { K } \frac { \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } } - b _ { k } } { 1 + \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } - b _ { k } } } \frac { \mathrm { e } ^ { \gamma w _ { \ell } ^ { \top } h _ { i } ^ { ( j ) } - b _ { \ell } } } { 1 + \mathrm { e } ^ { \gamma w _ { \ell } ^ { \top } h _ { i } ^ { ( j ) } - b _ { \ell } } } w _ { \ell } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } \bigg ) . } \end{array}\tag{C.157}
$$

Then,

$$
\begin{array} { r l } { \delta \leq | | \delta ( \hat { x } _ { 1 } ^ { 0 } ) | | \leq \frac { \sum _ { j = 1 } ^ { n } ( \frac { \sigma - \sigma ^ { 2 } } { \sigma } ) ^ { k _ { 1 } ( j ) } ( 1 - \sigma ^ { 2 } + \sigma ^ { 2 } + \sigma ^ { 2 } ) } { \operatorname { i n } \big ( 1 + \sigma ^ { 2 } + \sigma ^ { 2 } + \sigma ^ { 2 } + \sigma ^ { 2 } \big ) } + \frac { K } { \frac { \sigma } { \sigma } } \frac { \sigma ^ { 2 } \sigma ^ { 2 } K ^ { 2 } ( \sigma ^ { 2 } - \sigma ^ { 2 } ) } { 1 + \sqrt { \sigma ^ { 2 } K ^ { 2 } ( \sigma ^ { 2 } - \sigma ^ { 2 } ) } } } \\ & { \quad + \frac { K ^ { 2 } \eta } { \sigma \kappa } \frac { \sigma } { \sigma + \sigma ^ { 2 } + \sigma ^ { 2 } + \sigma ^ { 2 } + \sigma ^ { 2 } + \sigma ^ { 2 } } \frac { K } { \frac { \sigma } { \sigma } \sigma ^ { 2 } K ^ { 2 } ( \sigma ^ { 2 } - \sigma ^ { 2 } ) K ^ { 2 } ( \sigma ^ { 2 } - \sigma ^ { 2 } + \sigma ^ { 2 } ) } \frac { K } { \sqrt { \sigma ^ { 2 } K ^ { 2 } ( \sigma ^ { 2 } K ^ { 2 } - K ) } } } \\ & { \quad + \frac { K ^ { 2 } \eta } { \sigma \kappa } \frac { \sigma } { \sigma + \sigma } \frac { \sigma ^ { 2 } K ^ { 2 } ( \sigma ^ { 2 } - K ^ { 2 } ) K ^ { 2 } ( \sigma ^ { 2 } K ^ { 2 } - K ) } { 1 + \sigma ^ { 2 } + \sigma ^ { 2 } + \sigma ^ { 2 } K ^ { 2 } ( K ^ { 2 } - K ) } \frac { K } { \sqrt { \sigma ^ { 2 } K ^ { 2 } ( \sigma ^ { 2 } K ^ { 2 } - K ) } } } \\ &  \quad + \frac { K ^ { 2 } \eta ^ { 2 } } { \sigma ^ { 2 } K ^ { 2 } } ( \frac { K } { \sigma - 1 } + \frac  \sigma ^  2  \end{array}\tag{C.158}
$$

Each term in the right side of Eq. (C.158) is product of a polynomial without constant term and Sigmoid functions in terms of γ. When the critical condition II holds, i.e.,

$$
\operatorname* { m i n } \bigcup _ { k = 1 } ^ { K } \big \{ { w _ { k } ^ { \top } h _ { i } ^ { ( k ) } } \big \} _ { i = 1 } ^ { n } > 0 > \operatorname* { m a x } \bigcup _ { k = 1 } ^ { K } \bigcup _ { j = 1 \atop j \neq k } ^ { K } \big \{ { w _ { k } ^ { \top } h _ { i } ^ { ( j ) } } \big \} _ { i = 1 } ^ { n } ,\tag{C.159}
$$

every term in Eq. (C.158) satisfies the requirements of Lemma C.13(1); then, as γ increases, $\triangle ( s _ { i } ^ { ( k j ) } )$ decays exponentially and converges to 0.

When γ approaches zero, each term in the right side of Eq. (C.158) satisfies the requirements of Lemma C.13(2). Then, as $\gamma$ decreases linearly toward 0, every term in the right side of Eq. (C.158) decays to 0 at a quadratic or cubic rate; consequently, $\triangle ( s _ { i } ^ { ( k j ) } )$ decays quadratically to 0. Consider that the optimal values of unbiased decision scores are $\gamma \mathrm { o r } - \frac { \gamma } { K - 1 } .$ , which are linearly decay as $\gamma$ decreases. Therefore, the convergence rate of the unbiased decision scores decay linearly when γ linearly decreases to zero. □

Theorem C.16 When training the model using the CE loss $f _ { c e _ { i } }$ , as $\gamma$ approaches zero, the linear decrease in $\gamma$ leads to a linear decay in the convergence rate of unbiased decision scores. Once the critical condition I is satisfied, as $\gamma$ linearly approaches positive infinity, the convergence rate of unbiased decision scores decay exponentially.

Proof For the CE loss,

$$
\frac { \partial f _ { \mathrm { c e } } } { \partial h _ { i } ^ { ( j ) } } = \frac { \gamma } { n K } \bigg [ \bigg ( \frac { \mathrm { e } ^ { \gamma w _ { j } ^ { \top } h _ { i } ^ { ( j ) } - b _ { j } } } { \sum _ { m = 1 } ^ { K } \mathrm { e } ^ { \gamma w _ { m } ^ { \top } h _ { i } ^ { ( j ) } - b _ { m } } } - 1 \bigg ) w _ { j } + \sum _ { \ell = 1 } ^ { K } \frac { \mathrm { e } ^ { \gamma w _ { \ell } ^ { \top } h _ { i } ^ { ( j ) } - b _ { \ell } } } { \sum _ { m = 1 } ^ { K } \mathrm { e } ^ { \gamma w _ { m } ^ { \top } h _ { i } ^ { ( j ) } - b _ { m } } } w _ { \ell } \bigg ] ,\tag{C.160}
$$

$$
\frac { \partial f _ { \mathrm { c e } } } { \partial w _ { k } } = \frac { \gamma } { n K } \sum _ { i ^ { \prime } = 1 } ^ { n } \bigg [ \bigg ( \frac { \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( k ) } - b _ { k } } } { \sum _ { m = 1 } ^ { K } \mathrm { e } ^ { \gamma w _ { m } ^ { \top } h _ { i ^ { \prime } } ^ { ( k ) } - b _ { m } } } - 1 \bigg ) h _ { i ^ { \prime } } ^ { ( k ) } + \sum _ { \ell ^ { \prime } = 1 } ^ { K } \frac { \mathrm { e } ^ { \gamma w _ { k } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } - b _ { k } } } { \sum _ { m = 1 } ^ { K } \mathrm { e } ^ { \gamma w _ { m } ^ { \top } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } - b _ { m } } } h _ { i ^ { \prime } } ^ { ( \ell ^ { \prime } ) } \bigg ] .\tag{C.161}
$$

The critical condition I is

$$
w _ { k } h _ { i } ^ { ( k ) } > \operatorname* { m a x } \{ w _ { j } h _ { i } ^ { ( k ) } : j = 1 , 2 , \cdots , K , \mathrm { a n d } j \neq k \} , \quad \forall k , i .\tag{C.162}
$$

Similar to the proof of Theorem C.15, one can easily get the conclusions with help of Lemma C.14. □

Acknowledgements. The authors acknowledge the support provided by the Intelligent Computing Center of Shenzhen University.

Funding. This work was supported by the National Natural Science Foundation of China (Grant Nos. 62576217 and 62576216), Guangdong Provincial Key Laboratory (Grant No. 2023B1212060076), and the Natural Science Foundation of Ningxia Province (Grant No. 2025AAC020002).

Data availability. The datasets used in this study are publicly available. CIFAR-10 and CIFAR-100 can be obtained from https://cave.cs.toronto.edu/kriz/cifar-10 -python.tar.gz and https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz, respectively, MNIST from https://storage.googleapis.com/cvdf-datasets/mnist/, SVHN from http://ufldl.stanford.edu/housenumbers/, and Tiny ImageNet from http:// cs231n.stanford.edu/tiny-imagenet-200.zip. The corresponding dataset references are Krizhevsky (2009), Lecun et al. (1998), Netzer et al. (2011), and Wu et al. (2017), respectively.

## References

Alshammari, S., Wang, Y.-X., Ramanan, D., Kong, S. (2022). Long-tailed recognition via weight balancing. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 6897–6907).

Azad, R., Aghdam, E.K., Rauland, A., Jia, Y., Avval, A.H., Bozorgpour, A., . . . Merhof, D. (2024). Medical image segmentation review: The success of U-Net. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46 (12), 10076–10095, https://doi.org/10.1109/TPAMI.2024.3435571

Cao, K., Wei, C., Gaidon, A., Arechiga, N., Ma, T. (2019). Learning imbalanced datasets with label-distribution-aware margin loss. Advances in neural information processing systems (Vol. 32). (Available at https://proceedings.neurips.cc/ paper files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf)

Chen, G., Peng, P., Wang, X., Tian, Y. (2022). Adversarial reciprocal points learning for open set recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44 (11), 8065–8081, https://doi.org/10.1109/TPAMI.2021 .3106743

Chen, Z., Xu, C., Hu, T., Wang, L., Chen, J., Li, J. (2026). Decompose-compose feature augmentation for imbalanced crack recognition in industrial scenarios. IEEE Transactions on Automation Science and Engineering, 23, 3329–3339, https://doi.org/10.1109/TASE.2025.3535764

Chen, Z., Zhang, M., Cui, S., Li, H., Niu, G., Gong, M., . . . Zhang, K. (2024). Neural collapse inspired feature alignment for out-of-distribution generalization. The thirty-eighth annual conference on neural information processing systems. Retrieved from https://openreview.net/forum?id=wQpNG9JnPK

Chhabra, S. (2024). Pytorch-scratch-vision-transformer-vit. https://github.com/s -chh/PyTorch-Scratch-Vision-Transformer-ViT. GitHub. (Accessed: 2025-09- 01)

Dan, J., Liu, Y., Deng, J., Xie, H., Li, S., Sun, B., Luo, S. (2024). TopoFR: A closer look at topology alignment on face recognition. The thirty-eighth annual conference on neural information processing systems. Retrieved from https://openreview.net/forum?id=KVAx5tys2p

Dang, H., Huu, T.T., Nguyen, T.M., Ho, N. (2024). Neural collapse for crossentropy class-imbalanced learning with unconstrained reLU features model. Forty-first international conference on machine learning. Retrieved from https://openreview.net/forum?id=YBetKvUlF7

Deng, J., Guo, J., Xue, N., Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 4690–4699).

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., . . . Houlsby, N. (2021a). An image is worth 16x16 words: Transformers for image recognition at scale. International conference on learning representations. Retrieved from https://openreview.net/forum?id=YicbFdNTTy

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., . . . Houlsby, N. (2021b). An image is worth 16x16 words: Transformers for image recognition at scale. International conference on learning representations. Retrieved from https://openreview.net/forum?id=YicbFdNTTy

Fang, C., He, H., Long, Q., Su, W.J. (2021). Exploring deep neural networks via layer-peeled model: Minority collapse in imbalanced

training. Proceedings of the National Academy of Sciences, 118(43), e2103091118, https://doi.org/10.1073/pnas.2103091118 Retrieved from https://www.pnas.org/doi/abs/10.1073/pnas.2103091118

FISHER, R.A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179–188, https://doi .org/https://doi.org/10.1111/j.1469-1809.1936.tb02137.x Retrieved from https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-1809.1936.tb02137.x

Gao, J., Zhao, H., Guo, D.D., Zha, H. (2024, 21–27 Jul). Distribution alignment optimization through neural collapse for long-tailed classification. Proceedings of the 41st international conference on machine learning (Vol. 235, pp. 14969–14987). PMLR. Retrieved from https://proceedings.mlr.press/v235/gao24s.html

Graf, F., Hofer, C., Niethammer, M., Kwitt, R. (2021). Dissecting supervised contrastive learning. International conference on machine learning (pp. 3821– 3830).

Han, B. (2023). Wrapped cauchy distributed angular softmax for long-tailed visual recognition. International conference on machine learning (pp. 12368–12388).

He, K., Fan, H., Wu, Y., Xie, S., Girshick, R. (2020). Momentum contrast for unsupervised visual representation learning. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9729–9738).

He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the ieee conference on computer vision and pattern recognition (cvpr) (pp. 770–778). Las Vegas, USA: IEEE.

Howard, A., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., . . . Adam, H. (2017). Mobilenets: Eficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861. Retrieved from https://arxiv.org/abs/1704.04861

Huang, G., Liu, Z., van der Maaten, L., Weinberger, K.Q. (2017). Densely connected convolutional networks. Proceedings of the ieee conference on computer vision and pattern recognition (cvpr) (pp. 4700–4708). Honolulu, HI, USA: IEEE.

Jain, D.K., Zhao, X., Gan, C., Shukla, P.K., Jain, A., Sharma, S. (2024). Fusion-driven deep feature network for enhanced object detection and tracking in video surveillance systems. Information Fusion, 109, 102429, https://doi.org/https://doi.org/10.1016/j.infus.2024.102429 Retrieved from https://www.sciencedirect.com/science/article/pii/S1566253524002070

Kim, H., & Kim, K. (2024). Fixed non-negative orthogonal classifier: Inducing zero-mean neural collapse with feature dimension separation. The twelfth international conference on learning representations. Retrieved from https://openreview.net/forum?id=F4bmOrmUwc

Kobayashi, T. (2023). Two-way multi-label loss. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 7476–7485).

Koromilas, P., Bouritsas, G., Giannakopoulos, T., Nicolaou, M., Panagakis, Y. (2024). Bridging mini-batch and asymptotic analysis in contrastive learning: From infonce to kernel-based losses. International conference on machine learning (pp. 25276–25301).

Krizhevsky, A. (2009). Learning multiple layers of features from tiny images (Tech. Rep.). University of Toronto. Retrieved from https://www.cs.toronto.edu/ kriz/learning-features-2009-TR.pdf

Lecun, Y., Bottou, L., Bengio, Y., Hafner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86 (11), 2278–2324, https:// doi.org/10.1109/5.726791

Lee, S.H., Lee, S., Song, B.C. (2021). Vision transformer for small-size datasets. arXiv preprint arXiv:2112.13492. Retrieved from https://arxiv.org/abs/2112.13492

Li, J., Meng, Z., Shi, D., Song, R., Diao, X., Wang, J., Xu, H. (2023). Fcc: Feature clusters compression for long-tailed visual recognition. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 24080–24089).

Li, Q., Xiao, H., Shen, L. (2025). BCE vs. CE in deep feature learning. Forty-second international conference on machine learning. Retrieved from https://openreview.net/forum?id=aoLFIUlyPE

Liang, R., Zhang, S., Zhang, W., Zhang, G., Tang, J. (2024). Nonlocal hybrid network for long-tailed image classification. ACM Transactions on Multimedia Computing, Communications and Applications, 20(4), 1–22, https://doi.org/10.1145/ 3630256

Liu, J., Sun, Y., Han, C., Dou, Z., Li, W. (2020). Deep representation learning on long-tailed data: A learnable embedding augmentation perspective. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 2970–2979).

Liu, W., Wen, Y., Raj, B., Singh, R., Weller, A. (2023). Sphereface revived: Unifying hyperspherical face recognition. IEEE Transactions on Pattern Analysis and

Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., Song, L. (2017). SphereFace: Deep hypersphere embedding for face recognition. Proceedings of the ieee conference on computer vision and pattern recognition (pp. 212–220).

Liu, W., Yu, L., Weller, A., Sch¨olkopf, B. (2023). Generalizing and decoupling neural collapse via hyperspherical uniformity gap. The eleventh international conference on learning representations. Retrieved from https://openreview.net/forum?id=inU2quhGdNU

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., . . . Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012–10022).

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., Xie, S. (2022). A convnet for the 2020s. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 11976–11986).

Lu, J., & Steinerberger, S. (2022). Neural collapse under cross-entropy loss. Applied and Computational Harmonic Analysis, 59, 224–241,

Neal, L., Olson, M., Fern, X., Wong, W.-K., Li, F. (2018). Open set learning with counterfactual images. Proceedings of the european conference on computer vision (eccv) (pp. 613–628).

Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., Ng, A.Y. (2011). Reading digits in natural images with unsupervised feature learning. Nips workshop on deep learning and unsupervised feature learning.

Papyan, V., Han, X.Y., Donoho, D.L. (2020). Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40), 24652–24663, https://doi.org/10.1073/pnas.2015509117 Retrieved from https://www.pnas.org/doi/abs/10.1073/pnas.2015509117

Vasu, P.K.A., Pouransari, H., Faghri, F., Vemulapalli, R., Tuzel, O. (2024, June). MobileCLIP: Fast image-text models through multi-modal reinforced training. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (cvpr) (pp. 15963–15974).

Wang, F., Xiang, X., Cheng, J., Yuille, A.L. (2017). NormFace: L<sub>2</sub> hypersphere embedding for face verification. Proceedings of the 25th acm international conference on multimedia (pp. 1041–1049).

Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., . . . Liu, W. (2018). Cos-Face: Large margin cosine loss for deep face recognition. Proceedings of the ieee conference on computer vision and pattern recognition (pp. 5265–5274).

Wen, Y., Liu, W., Weller, A., Raj, B., Singh, R. (2022). SphereFace2: Binary classification is all you need for deep face recognition. International conference on learning representations.

Wu, J., Zhang, Q., Xu, G. (2017). Tiny ImageNet challenge (Tech. Rep.). Stanford University. Retrieved from https://cs231n.stanford.edu/reports/2017/pdfs/930.pdf

Yaras, C., Wang, P., Zhu, Z., Balzano, L., Qu, Q. (2022). Neural collapse with normalized features: A geometric analysis over the riemannian manifold. S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, & A. Oh (Eds.), Advances in neural information processing systems (pp. 11547–11560). New Orleans: Curran Associates, Inc.

Zhang, H., Duan, J., Xue, M., Song, J., Sun, L., Song, M. (2022). Bootstrapping vits: Towards liberating vision transformers from pre-training. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 8944– 8953).

Zhao, W., Zhu, X., Shi, H., Zhang, X.-Y., Zhao, G., Lei, Z. (2025). Global crossentropy loss for deep face recognition. IEEE Transactions on Image Processing, 34 , 1672–1685, https://doi.org/10.1109/TIP.2025.3546481

Zheng, K., & Yang, E.-H. (2024). Knowledge distillation based on transformed teacher matching. The twelfth international conference on learning representations. Retrieved from https://openreview.net/forum?id=MJ3K7uDGGl

Zhou, J., Jia, X., Li, Q., Shen, L., Duan, J. (2023). UniFace: Unified cross-entropy loss for deep face recognition. Proceedings ofthe IEEE/CVF international conference on computer vision (pp. 20730–20739).

Zhou, J., You, C., Li, X., Liu, K., Liu, S., Qu, Q., Zhu, Z. (2022). Are all losses created equal: A neural collapse perspective. S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, & A. Oh (Eds.), Advances in neural information processing systems (pp. 31697–31710). New Orleans: Curran Associates, Inc.

Zhu, H., Chen, B., Yang, C. (2023). Understanding why ViT trains badly on small datasets: An intuitive perspective. arXiv preprint arXiv:2302.03751. Retrieved from https://arxiv.org/abs/2302.03751

Zhu, Z., Ding, T., Zhou, J., Li, X., You, C., Sulam, J., Qu, Q. (2021). A geometric analysis of neural collapse with unconstrained features. M. Ranzato, A. Beygelzimer,

Y. Dauphin, P. Liang, & J.W. Vaughan (Eds.), Advances in neural information processing systems (pp. 29820–29834). Curran Associates, Inc.