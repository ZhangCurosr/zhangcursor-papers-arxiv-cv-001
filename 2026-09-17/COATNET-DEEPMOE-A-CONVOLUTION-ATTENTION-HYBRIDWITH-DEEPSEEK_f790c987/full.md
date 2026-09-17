# COATNET-DEEPMOE: A CONVOLUTION–ATTENTION HYBRIDWITH DEEPSEEK MIXTURE-OF-EXPERTS FORPARAMETER-EFFICIENT TOMATO DISEASE CLASSIFICATION

## A PREPRINT

Md Nadim Mahamood Department of Computer Science and Engineering Begum Rokeya University Rangpur, Bangladesh nadim.cse.brur@gmail.com

Md Shafi Ud Doula Department of Information and Communication Technologies Asian Institute of Technology Pathum Thani, Thailand shafi.cse.brur@gmail.com

Md Arif Shahriar   
Department of Computer Science Texas State University San Marcos, Texas, USA   
arif.shahriar@txstate.edu Kamrul Hasan   
Department of Computer Science Texas State University San Marcos, Texas, USA   
kamrul.hasan@txstate.edu

## ABSTRACT

The world population is growing rapidly, and technology is improving in parallel. Meeting the huge demand for food for these 7 billion people not only depends on increasing food production but also on reducing food loss. Crop losses due to disease affect both the food supply and the financial and economic stability of a country. Tomatoes are among the top food-producing crops globally, and a significant portion of this production is lost due to disease. People have used Machine Learning techniques for feature extraction and early diagnosis of tomato diseases, and nowadays, Deep Learning-based models are widely used for disease recognition. However, most existing models are highly parameter-intensive, which increases the time required for training and inference. As a result, while lightweight models are more suitable for user-friendly applications, they often show a reduction in performance. To balance performance and model size, we propose CoAtNet-DeepMoE, a Convolution–Attention hybrid architecture for rich feature extraction, further enhanced with a DeepSeek Mixture of Experts to substantially reduce the number of parameters without sacrificing accuracy. We evaluate our model on both balanced and imbalanced datasets from Kaggle and PlantVillage, demonstrating robustness and achieving 99.80% accuracy, 99.80% precision, 99.80% recall, and 99.80% F1-score on Kaggle, and 99.83% accuracy, 99.85% precision, 99.76% recall, and 99.80% F1-score on PlantVillage, representing state-of-the-art performance with only 2.47M parameters. The source code will be available at https://github.com/nadimbrur/CoAt-MoE.

Keywords DeepSeek MoE · Parameter optimization · Deep Learning · Tomato disease

## 1 Introduction

The continuous growth of the global population has heightened the demand for food, and the Office of the Director of National Intelligence projects that the world population will reach 9.2 billion by 2040.<sup>1</sup> Furthermore, the Food and Agriculture Organization of the United Nations (FAO) estimates that global food production must increase by approximately 70% by 2050 to meet future needs.<sup>2</sup> However, recent evidence shows that 2.3–2.9 billion people are unable to afford a healthy diet because their incomes are insufficient relative to the rising cost of nutritious foods [1], highlighting a widening gap between food availability and economic access. To satisfy rising food demand, agricultural technologies are advancing rapidly; nevertheless, plant diseases and pests remain major obstacles. For example, the FAO reports that pests cause 20–40% of global crop losses annually, imposing an estimated economic burden of about USD 290 billion.<sup>3</sup>

Tomatoes are one of the most regularly planted crops around for nutritional value and financial worth [2]. Globally, it is the leading vegetable crop which production exceeds 180 million tons annually [3], contributing nearly one-sixth of total vegetable production [4–6]. However, the farmer reported that tomato losses due to diseases were 64.71% [7], and losses from virus diseases should be around 2 to 5% annually [8]. In addition, 80% to 90% of the disease on the plant appears on its leaves [9]. Therefore, it is crucial to monitor plant leaves for signs of disease to control its spread through early recognition.

Tomato leaf diseases are caused by viruses, fungi, and bacteria, and they show several visual symptoms, including wilting, lesions, and leaf browning [2]. Traditional laboratory-based diagnostic techniques such as Western blotting, enzyme-linked immunosorbent assay, and microarrays have been used to detect these pathogens [10]. However, in practical agricultural settings, disease detection still largely depends on manual visual inspection by experts, a time-consuming, subjective, and error-prone process [2, 11, 12]. In addition, monitoring large fields is tedious for farmers and often requires specific training, experience in recognizing disease symptoms [13], and broad knowledge of multiple plant diseases. Furthermore, manual inspections lack scientific consistency because farmers’ skills and backgrounds differ, making the process less reliable [14].

Traditional machine learning (ML) and modern deep learning (DL) techniques have been extensively applied in tomato plant disease analysis. In the early years, traditional machine learning focused on recognizing plant diseases using image processing and feature-based data analysis [15]. For example, Geetha et al. [16] developed a plant leaf disease detection system that preprocesses images, extracts key features using Histogram of Oriented Gradients and Gray Level Co-occurrence Matrix, and classifies diseases with the K-Nearest Neighbors (KNN) algorithm. Similarly, Basavaiah et al. [17] proposed a method for detecting tomato leaf diseases by fusing multiple features, Color Histograms, Hu Invariant Moments, Haralick texture features, and Local Binary Patterns to improve classification accuracy. Gadade et al. [18] applied median filtering, extracted shape, color, and texture features, and evaluated multiple classifiers, Support Vector Machine, KNN, Naive Bayes, Decision Trees, and Linear Discriminant Analysis for both disease type detection and severity assessment to guide treatment. Most traditional machine learning-based models rely on manually extracted features, and their performance is heavily dependent on feature quality [19, 20], leading to less accurate results compared to modern approaches [21]. Early models also required high-resolution images with controlled backgrounds; currently, research focuses on handling images captured under complex, real-world conditions [22].

Deep learning has achieved significant success in both computer vision and natural language processing applications, including agriculture (e.g., rice disease detection [23]), due to its ability to model complex patterns in challenging environments. Similarly, deep learning is increasingly applied in tomato leaf disease analysis.

For instance, Wu et al. [24] used a ResNet50 model with augmentation techniques to improve classification performance, although the model’s computational complexity was high. Hong et al. [25] employed transfer learning to reduce training data requirements, time, and computational costs. Nine types of tomato leaves, including healthy leaves, were classified using five deep network architectures: ResNet50, Xception, MobileNet, ShuffleNet, and DenseNet121- Xception. DenseNet121-Xception achieved the highest accuracy of 97.10% but had the largest number of parameters, while ShuffleNet achieved 83.68% accuracy with far fewer parameters. Recently, concepts from natural language processing, such as transformer attention [26], have been applied in computer vision tasks, including tomato leaf disease classification. For example, Chelladurai et al. [27] segmented tomato leaf images using U-Net, extracted features with VGG-16, and classified diseases using a transductive Long Short-Term Memory (LSTM) network with attention, achieving very high accuracy under controlled conditions. However, these deep learning models typically involve millions of parameters, requiring substantial training time and computational resources, which makes deployment on low-end devices challenging. Reducing the number of parameters can also degrade performance, making it difficult to maintain high accuracy with lightweight models.

To maintain efficiency in both model size and performance, we propose CoAtNet-DeepMoE, a model that integrates convolution and attention mechanisms to capture rich features. It also employs the DeepSeek Mixture-of-Experts approach to reduce the number of parameters without compromising performance. The contributions of this study are summarized as follows:

• Efficient Architecture: Designed a model that reduces parameters from 26.67 million to 2.47 million (an ≈ 90.73% reduction) by implementing the latest DeepSeek Mixture-of-Experts (MoE), replacing a traditional multi-layer perceptron (MLP), without sacrificing accuracy.

• State-of-the-Art Performance: Achieved 99.80% accuracy, 99.80% precision, 99.80% recall, and 99.80% F1-score on the Kaggle dataset (balanced) and 99.83% accuracy, 99.85% precision, 99.76% recall, and 99.80% F1-score on the PlantVillage dataset (imbalanced), demonstrating robustness across both balanced and imbalanced datasets.

• Ablation Study: Performed an ablation study comparing well-known convolutional and attention-based models, including CoAtNet-Base [28], ResNet50 [29], and ViT-Tiny [30], to assess inference time, parameter efficiency, and overall performance.

## 2 Related Work

## 2.1 Machine learning-based

Machine learning-based approaches for tomato leaf disease detection typically rely on handcrafted or color-based feature extraction methods, which are then fed into traditional classifiers. These models depend entirely on manually designed feature descriptors. For instance, Gadade et al. [31] proposed a segmentation-based system where infected regions were segmented and analyzed using color, texture, and shape features for classification and severity measurement. Among the various combinations of feature extraction methods and classifiers evaluated, the HOG + SVM model achieved the highest performance, yet the accuracy remained relatively low at 48.77% across 45 combinations on 3000 PlantVillage images. Similarly, Joshi et al. [32] introduced a GA-KNN framework, where morphological, statistical, and textural features of tomato leaves were extracted, and a Genetic Algorithm selected the optimal subset of features. This approach enabled KNN to achieve a significantly higher accuracy of 94.3% using only 13 features on the PlantVillage dataset. In addition, Khan et al. [33] developed a system using Gray Level Co-occurrence Matrix (GLCM) and Scale-Invariant Feature Transform (SIFT) features with a quadratic SVM classifier, achieving 92.3% accuracy with ten-fold cross-validation on a 2700-image, nine-class dataset, demonstrating reliable multiclass classification.

Recent studies increasingly combine deep learning feature extraction with traditional machine learning classifiers. Imam et al. [34] proposed a hybrid MobileNet–SVM approach, where features extracted from MobileNet were classified by an SVM, achieving 99.37% overall accuracy across nine disease classes, outperforming previous methods. Similarly, Terziouglu et al. [35] evaluated 21 deep learning models for tomato disease detection on a 6414-image dataset. The top-performing combination used EfficientNet-b0 features with Chi-Square feature selection and a Fine KNN classifier, achieving 92.0% accuracy, highlighting the effectiveness of deep feature extraction combined with traditional classifiers for robust disease identification.

## 2.2 Deep learning-based

Deep learning approaches are increasingly used in tomato leaf disease detection for automatically learning hierarchical image features. Major directions include CNNs, attention-based models, and hybrid approaches, which outperform traditional machine learning in accuracy and robustness.

CNN [36] focuses on extracting spatial features from images using various filters to identify shapes, textures, and other patterns for classification. It is prominently used in tomato leaf disease detection. For example, Assaduzzaman et al. [37] developed XSE-TomatoNet, an enhanced EfficientNetB0 with Squeeze-and-Excitation (SE) blocks and multi-scale feature fusion, achieving 98.83% test accuracy, outperforming MobileNet and VGG19. Since it is based on EfficientNetB0 ( 6.0 million parameters), the total parameters are at least 6 million, with additional parameters from SE blocks and multi-scale fusion. Another study [38] employed YOLOv8s for tomato leaf disease detection on the PlantVillage dataset, attaining the highest mAP of 92.5% and a fast inference speed of 121.5 FPS, outperforming YOLOv5 and Faster R-CNN in both accuracy and real-time detection. The estimated number of parameters for YOLOv8s is approximately 11.0 million. In addition, Prabhasha et al. [39] used a transfer learning-based approach with pre-trained models (CNN, AlexNet, ResNet, InceptionV3, VGG-16) for tomato leaf disease detection, with VGG-16 achieving the highest accuracy of 93.7%, demonstrating efficient classification and reduced training time through transfer learning.

Hybrid models that combine CNN and attention mechanisms are increasingly popular for tomato leaf disease detection. For instance, Zhao et al. [40] proposed a deep CNN with residual blocks and attention modules, achieving 96.81% accuracy. Similarly, Karthik et al. [41] employed two architectures, residual CNN and residual CNN with attention, on the PlantVillage dataset, achieving 98% validation accuracy using 5-fold cross-validation. Additionally, Sunil et al.

[42] proposed a Multilevel Feature Fusion Network (MFFN) with ResNet50 and an Adaptive Attention Mechanism (channel, spatial, and pixel attention), achieving 99.88% training and validation accuracy and 99.83% external test accuracy, along with a pesticide recommendation module based on detected diseases.

Machine learning–based classification suffers from a lack of generalizability and depends heavily on handcrafted features [23]. In contrast, deep learning achieves highly prominent performance in CNNs and attention-based models, but these come with a huge number of parameters, leading to high costs, long training times, large GPU power requirements, increased energy consumption, and a higher carbon footprint. Reducing parameters, however, can hinder performance. To address this, we propose CoAtNet-DeepMoE, which utilizes convolution and attention to capture fine-grained features and incorporates an advanced version of Mixture-of-Experts, DeepSeek-MoE, developed by DeepSeek, that reduces the number of parameters by splitting the large network into smaller subparts with a shared space for all. This implementation reduces the number of parameters from 26.67 million to 2.47 million, achieving approximately 90.73% reduction, without any loss in classification accuracy.

![](images/45e0863a07506cbab00d25666bd2f0f4da5451b5697c3b2221299eb062890e2e.jpg)  
Figure 1: Overview of the proposed CoAtNet-DeepMoE architecture. The model integrates convolutional layers for local feature extraction, MbConv and SE blocks for channel-wise enhancement, and attention layers for global feature modeling. A parameter-efficient DeepSeek Mixture-of-Experts replaces the traditional MLP, where the router selects the top-k experts (2 in this case) along with a shared expert for feature processing. The selected experts specialize in capturing disease-specific patterns, enabling the model to focus on the most relevant features for each input. The shared expert, on the other hand, preserves common information across all classes (e.g., common characteristics), ensuring stable and generalizable representations. This selective expert activation enhances feature representation while greatly reducing computational cost and overall parameter count (26.67 M to 2.47 M, ≈ 90.73%).

## 2.3 Methods

The proposed CoAtNet-DeepMoE model combines the strengths of convolutional networks, attention mechanisms present in CoAtNet [28], and Mixture-of-Experts (MoE) [43] to achieve an efficient and high-performance architecture for tomato leaf disease classification. The methodology is composed of several key stages that progressively transform the input image into a compact, discriminative representation suitable for final classification.

## 2.3.1 Stem Block

The stem block is the first stage of the proposed model, responsible for extracting low-level spatial features from the input image. It consists of two consecutive 3 × 3 convolution layers with strides 2 and 1, respectively, along with a Layer Normalization applied between them. This block captures essential early information such as edges, textures, and color transitions, while also reducing the spatial resolution to prepare the image for deeper feature extraction.

Given an input image $X \in \mathbb { R } ^ { 2 2 4 \times 2 2 4 \times 3 }$ , the stem block operation can be formulated as:

$$
X _ { s } = \operatorname { C o n v } _ { 3 \times 3 } \big ( \operatorname { N o r m } ( \operatorname { C o n v } _ { 3 \times 3 } ( X ) ) \big ) ,\tag{1}
$$

where the first $\mathrm { { C o n v } _ { 3 \times 3 } ( \cdot ) }$ projects the feature map to $\mathbb { R } ^ { 1 1 2 \times 1 1 2 \times 3 2 }$ , and the second $\mathrm { { C o n v } _ { 3 \times 3 } ( \cdot ) }$ further projects it to $\mathbb { R } ^ { 1 1 2 \times 1 1 2 \times 6 4 }$ . This stage efficiently reduces spatial dimensions while preserving important visual information needed for subsequent MbConv and Transformer layers.

## 2.3.2 MbConv Block

The MbConv block [44] consists of depthwise convolution [45] to capture spatial interactions. In equation form, it can be written as:

$$
X _ { \mathrm { M b C o n v } } = \underbrace { \mathrm { C o n v _ { 1 } \times 1 } \big ( \mathrm { A v g P o o l _ { 2 \times 2 } } ( X _ { s } ) \big ) } _ { \mathrm { S i i p ~ c o n n e c t i o n } } + \underbrace { \mathrm { C o n v _ { 1 \times 1 } } \Big ( \mathrm { S E } \big ( \mathrm { D e p C o n v _ { 3 \times 3 } } \big ( \mathrm { C o n v _ { 1 \times 1 } } ( \mathrm { A v g P o o l _ { 2 \times 2 } } ( X _ { s } ) ) \big ) \big ) \Big ) } _ { \mathrm { M a i n ~ p a h } }\tag{2}
$$

This block has two paths: the skip connection performs spatial downsampling by a factor of 2 followed by $\mathrm { ~ a ~ } 1 \mathrm { ~ } \times$ 1 convolution to project features into the desired dimension. The main path consists of downsampling, a $1 \times 1$ convolution that expands the channel dimension by 4×, a depthwise convolution $( D e p t h C o n v _ { 3 \times 3 } ( \cdot ) )$ to capture spatial interactions, and a Squeeze-and-Excitation (SE) module, which adaptively recalibrates channel-wise feature responses by emphasizing important channels and suppressing less useful ones without affecting the feature dimensions. Finally, another $1 \times 1$ convolution projects the features to match the skip connection dimensions.

After the first MbConv block, the output feature has dimensions $\mathbb { R } ^ { 5 6 \times 5 6 \times 9 6 }$ , and after the second block, the output feature has dimensions $\mathbb { R } ^ { 2 8 \times 2 8 \times 1 9 2 }$

## 2.3.3 Transformer2d Block

Relative Attention: CoAtNet [28] employs a variant of the transformer attention mechanism known as relative attention [46], which is input-independent and eliminates both parameter sharing across layers and the need for a bucketing mechanism. The first portion of Transformer2d consists of two parallel paths: the skip connection, where the input is downsampled by a factor of 2 and passed through a $1 \times 1$ convolution to expand the number of channels; and the main path, where the downsampled input is processed by the relative attention module. This module captures global receptive fields and models long-range spatial dependencies across the leaf surface, such as vein patterns, texture irregularities, and spot distributions, which are essential for accurate disease identification. It is outlined as:

$$
X _ { \mathrm { a t t e n i o n } } = \underbrace { \mathrm { R e l \_ a t t e n t i o n } \big ( \mathrm { A v g { P o o l } _ { 2 \times 2 } \big ( X _ { \mathrm { M b C o n v } } \big ) \big ) } } _ { \mathrm { M a i n \ p a t h } } + \underbrace { \mathrm { C o n v _ { 1 \times 1 } \big ( A v g { P o o l } _ { 2 \times 2 } \big ( X _ { \mathrm { M b C o n v } } \big ) \big ) } } _ { \mathrm { S k i p \ c o n n e c t i o n } }
$$

DeepSeek MoE: We replace the traditional multi-layer perceptron (MLP) with DeepSeek MoE [47] for parameter optimization. DeepSeek MoE is an improved version of the Mixture-of-Experts (MoE) [43] framework, in which the network is divided into several smaller subnetworks called experts. Unlike standard MLPs, which have a large number of parameters, MoE reduces the parameter count by activating only a subset of experts for each input, determined dynamically by a gating router.

DeepSeek MoE further introduces a shared expert, which provides information common to all inputs. While the top-k experts are dynamically selected for each input via the gating function, the shared expert is always activated, ensuring essential information flows across all input types. For the input $X _ { \mathrm { a t t e n t i o n } } .$ , the operation can be formulated as:

$$
X _ { \mathrm { D e e p S e e k M o E } } = S ( x ) + \sum _ { i \in \mathrm { I o p K } ( G ( x ) , 2 ) } g _ { i } \cdot E _ { i } ( x )\tag{3}
$$

where $E _ { i } ( \cdot )$ denotes the output of the i-th expert, $S ( \cdot )$ is the output of the shared expert, $G ( \cdot )$ represents the gating function producing scores $g _ { i }$ , and $\mathrm { T o p K } ( G ( x ) , 2 )$ indicates the indices of the top-2 selected experts. Each expert first reduces the input channels by a factor of four and then projects them back to the original dimension using $1 \times 1$ convolutions.

The output of DeepSeek MoE is then combined with the original input via a residual connection:

$$
X _ { \mathrm { M L P } } = \underbrace { X _ { \mathrm { D e e p S e e k M o E } } } _ { \mathrm { M a i n p a t h } } + \underbrace { X _ { \mathrm { a t t e n t i o n } } } _ { \mathrm { S k i p c o n n e c t i o n } }\tag{4}
$$

Finally, the model includes two Transformer2D stages. The first outputs a feature map of size $\mathbb { R } ^ { 1 4 \times 1 4 \times 3 8 4 }$ , and the second outputs a feature map of size $\mathbb { R } ^ { 7 \times 7 \times 7 6 8 }$ , which we denote as $X _ { \mathrm { M L P } }$ before passing it to the classifier head.

## 2.3.4 Classifier head

The classifier head consists of a Global Average Pooling (GAP) layer that converts the 2D feature map into a 1D feature vector. This vector is then passed through a fully connected (FC) layer to project it into the desired number of output classes. Finally, a softmax activation is applied to compute the class probabilities. Mathematically, this can be expressed as:

$$
X _ { \mathrm { c l a s s } } = \mathrm { F C } \big ( \mathrm { G A P } ( X _ { \mathrm { M L P } } ) \big )\tag{5}
$$

where $\mathrm { G A P ( \cdot ) }$ denotes the Global Average Pooling operation, which converts the feature map $X _ { \mathrm { M L P } } \in \mathbb { R } ^ { 7 \times 7 \times 7 6 8 }$ into a 1D feature vector of size 768, and $\operatorname { F C } ( \cdot )$ projects this vector into the number of output classes, here 10. The use of DeepSeek MoE reduces the total number of parameters from 26,674,252 to 2,473,316 (≈90.73% reduction) without any loss in accuracy. The overall architecture is illustrated in Fig. 1, and as shown in Table 1, the block-wise parameter comparison highlights that CoAt-DeepMoE substantially reduces the number of parameters in both transformer stages, while the stem and classifier head remain unchanged.

Table 1: Block-wise parameter comparison between CoAt-Base and CoAt-DeepMoE.
<table><tr><td>Module</td><td>CoAt-Base</td><td>CoAt-DeepMoE</td></tr><tr><td>Stem</td><td>19,360</td><td>19,360</td></tr><tr><td>MbConv-0</td><td>236,768</td><td>83,648</td></tr><tr><td>MbConv-1</td><td>1,410,720</td><td>208,416</td></tr><tr><td>Transformer2D-2</td><td>12,145,374</td><td>437,174</td></tr><tr><td>Transformer2D-3</td><td>12,852,804</td><td>1,715,492</td></tr><tr><td>Classifier Head</td><td>9,226</td><td>9,226</td></tr><tr><td>Total</td><td>26,674,252</td><td>2,473,316</td></tr></table>

## 2.3.5 Dataset

The tomato disease dataset used in this study is constructed by combining images from the Kaggle repository<sup>4</sup> and the PlantVillage benchmark dataset<sup>5</sup>, resulting in a total of 29,160 images across ten classes. While the Kaggle dataset maintains a perfectly uniform distribution with 1,100 samples per class, the PlantVillage dataset is significantly imbalanced. In particular, Yellow Leaf Curl Virus (YLCV) is the dominant class, contributing 5,357 images (0.30), whereas classes such as Mosaic Virus (MV), with 373 images (0.02), and Leaf Mold (LM), with 952 images (0.05), are notably underrepresented. When combined, the overall dataset still exhibits class imbalance, with YLCV remaining the largest class (6,457 images; 0.22) and MV becoming the least represented class (1,473 images; 0.05). Such an imbalance can negatively affect model learning by biasing predictions toward frequent classes while reducing sensitivity to rare diseases. To mitigate this issue, we apply targeted data augmentation, such as rotations, flips, and color jittering, to improve class uniformity and enhance the model’s ability to generalize across all disease categories. The distribution of the Kaggle dataset, the PlantVillage dataset, and the combined dataset is shown in Table 2. Also, an image example for each class of the dataset is shown in Figure 2.

## 2.3.6 Training details

The proposed CoAtNet-DeepMoE model was trained for 300 epochs with a batch size of 32, using input images resized to $2 2 4 \times 2 2 4 \times 3 .$ . To stabilize training and improve robustness, a Cosine Annealing learning rate scheduler [48] was employed. The model was trained using Cross-Entropy loss with label smoothing [49] set to 0.1 to reduce overconfidence in predictions. Label smoothing works by assigning a slightly lower probability to the correct class and distributing the remaining probability across the other classes. For instance, with a smoothing value of 0.1, the correct class receives a probability of 0.9, while the remaining 0.1 is distributed among the other classes. This technique help improve generalization and reduces overfitting. Additional training details, including optimizer, weight decay, early stopping, and hardware specifications, are summarized in Table 3.

Table 2: Corrected class distribution comparison between Kaggle and PlantVillage tomato datasets (instances and ratios).
<table><tr><td>Dataset</td><td>BS</td><td>EB</td><td>LB</td><td>LM</td><td>SLS</td><td>SE</td><td>TS</td><td>YLCV</td><td>MV</td><td>Healthy</td><td>Total</td></tr><tr><td>Kaggle</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>1100 (0.10)</td><td>11000 (1.00)</td></tr><tr><td>PlantVillage</td><td>2127 (0.12)</td><td>1000 (0.06)</td><td>1909 (0.11)</td><td>952 (0.05)</td><td>1771 (0.10)</td><td>1676 (0.09)</td><td>1404 (0.08)</td><td>5357 (0.30)</td><td>373 (0.02)</td><td>1591 (0.09)</td><td>18160 (1.00)</td></tr><tr><td>Total</td><td>3227 (0.11)</td><td>2100 (0.07)</td><td>3009 (0.10)</td><td>2052 (0.07)</td><td>2871 (0.09)</td><td>2776 (0.10)</td><td>2504 (0.09)</td><td>6457 (0.22)</td><td>1473 (0.05)</td><td>2691 (0.09)</td><td>29160 (1.00)</td></tr></table>

Bacterial spot: BS; Leaf Mold: LM; Early blight: EB; Late blight: LB; Septoria leaf spot: SLS; Spider mites: SE; Target Spot: TS; Yellow Leaf Curl Virus: YLCV; Mosaic virus: MV

![](images/5845f6917db08832f466a88dee9c9415358b333807e41dcfc1314ed66ef1fb4e.jpg)  
(a) Bacterial spot

![](images/9a85fe001aab8c9e57673c5f3d04c243e03efd99b0e430b36e8e8e55161aa3a0.jpg)  
(b) Early blight

![](images/e8e8592b0b7fe2fe266ec6485b2dde339bd08d0c82a1d284f80a9c8dcff00217.jpg)  
(c) Healthy

![](images/420b764a41b3696a77dc00b7da547046e44386248a018091cf9d5581d7e8736d.jpg)  
(d) Late blight

![](images/ce427b3d84b0a77a47085f817787ac106c915d9e18eb34abce1107dadfa46797.jpg)  
(e) Leaf Mold

![](images/0d90753ccf1da3dfe8ceef11c237c90020f31188b943c7a17baf5becbb8aa41f.jpg)  
(f) Mosaic virus

![](images/cf65ca92189c884e182b4355277b0b35a1e0feebce7f32d10d227917c70eaa23.jpg)  
(g) Septoria leaf spot

![](images/1ff64e9b50362e8618aca8ffb7bc1dfa33bd1a8e76277f2dc3f075dde1bbaa19.jpg)  
(h) Two-spotted spider

![](images/c5905d3e514ef87eeed97309c66f87a8a7331e12625dbdffdb684176c4c71d67.jpg)  
(i) Target spot

![](images/f4cdff3cefb9d17e3c7c7dd366fb4d062fa756d0d687a37cb3e5b953fbc2b810.jpg)  
(j) Yellow leaf curl virus  
Figure 2: Examples of representative images from each class in the tomato leaf disease datasets, illustrating the visua diversity and key characteristics.

Table 3: Training configuration details used for model training.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Learning Rate (LR)</td><td> $\overline { { 1 \times 1 0 ^ { - 5 } } }$ </td></tr><tr><td>Loss Function</td><td>CE (label_smoothing = 0.1)</td></tr><tr><td>Optimizer</td><td>AdamW (weight_  $\mathrm { \underline { { d e c a y } } = 1 \times 1 0 ^ { - 4 } ) }$ </td></tr><tr><td>LR Scheduler</td><td>CosineAnnealingLR  $( T _ { \mathrm { m a x } } = 3 0 0 )$ </td></tr><tr><td>Stopping Patience</td><td>50 epochs</td></tr><tr><td>Batch Size</td><td>32</td></tr><tr><td>Input Image Size</td><td>(224, 224, 3)</td></tr><tr><td>Activation Function</td><td>Softmax</td></tr><tr><td>Hardware</td><td>NVIDIA GeForce RTX 3090 GPU</td></tr></table>

## 2.3.7 Performance evaluation metrics

We consider four evaluation metrics, along with the number of parameters of the model, to validate the performance of the proposed approach. These metrics include Accuracy, Precision, Recall, and F1-score. Accuracy represents the ratio of correctly classified instances to the total number of predictions. Precision measures the proportion of correctly identified positive samples among all predicted positive samples. For example, when computing the precision of the healthy class, we treat healthy as the positive class and the remaining classes as negative; thus, precision indicates how many predicted healthy instances are actually healthy. Recall, in contrast, measures how many true positive samples are correctly identified among all actual positive samples. Following the same example, the recall of the healthy class reflects how many truly healthy instances are correctly classified as healthy. The F1-score is then computed as the harmonic mean of precision and recall, providing a balanced assessment of model performance.

Table 4: Macro-averaged metrics used to evaluate the proposed CoAtNet-DeepMoE Metric Expression
<table><tr><td colspan="2">Metric Expression</td></tr><tr><td rowspan="2">Accuracy</td><td> $\textstyle \sum _ { i = 1 } ^ { K } \mathrm { T P } _ { i }$ </td></tr><tr><td> $\overline { { \sum _ { i = 1 } ^ { K } ( \mathrm { T P } _ { i } + \mathrm { F P } _ { i } + \mathrm { F N } _ { i } ) } }$ </td></tr><tr><td>Recall</td><td> $\frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { \mathrm { T P } _ { i } } { \mathrm { T P } _ { i } + \mathrm { F N } _ { i } }$ </td></tr><tr><td>Precision</td><td> $\frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { \mathrm { T P } _ { i } } { \mathrm { T P } _ { i } + \mathrm { F P } _ { i } }$ </td></tr><tr><td>F1-score</td><td> $\frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { 2 \mathrm { T P } _ { i } } { 2 \mathrm { T P } _ { i } + \mathrm { F P } _ { i } + \mathrm { F N } _ { i } }$ </td></tr></table>

Since one of our datasets is imbalanced, we report the macro-averaged Precision, Recall, and F1-score to ensure equal contribution of each class to the final evaluation. Table 4 presents the mathematical formulations of these metrics.

## 3 Experiment & Results

## 3.1 Comparison with SOTA

## 3.1.1 Performance on the Kaggle dataset and compare with the SOTA models:

Table 5: Performance comparison of state-of-the-art (SOTA) models on the Kaggle tomato disease dataset. $\mathbf { \ddot { \theta } M } ^ { \prime \prime }$ denotes the number of trainable parameters in millions. Accuracy, Precision, Recall, and F1-score are reported in percentage (%) form. Bold values indicate the best performance among the compared models.
<table><tr><td>SL</td><td>Reference</td><td>Parameters</td><td>Model Name</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>1</td><td>[38]</td><td>11.00 M</td><td>YOLOv8s</td><td>93.2</td><td>90.3</td><td>一</td><td>一</td></tr><tr><td>2</td><td>[10]</td><td>32.29 M</td><td>DenseNet</td><td>95.4</td><td>95.7</td><td>95.4</td><td>95.4</td></tr><tr><td>3</td><td>[24]</td><td>23.50 M</td><td>ResNet50</td><td>92.4</td><td>一</td><td>一</td><td>一</td></tr><tr><td>4</td><td>[24]</td><td>23.56 M</td><td>SDE-ResNet50</td><td>98.0</td><td>98.0</td><td>一</td><td>98.0</td></tr><tr><td>5</td><td>[24]</td><td>54.32 M</td><td>Inception-ResNet</td><td>98.2</td><td>98.23</td><td>一</td><td>98.2</td></tr><tr><td>6</td><td>[50]</td><td>200.00 M</td><td>AlexNet + ResNet50 + VGG16 + QSSVM</td><td>98.3</td><td></td><td></td><td></td></tr><tr><td>7</td><td>[51]</td><td>143.30 M</td><td>VGG16 + NASNet</td><td>98.7</td><td>97.9</td><td>98.6</td><td>98.6</td></tr><tr><td>8</td><td>Our</td><td>2.47 M</td><td>CoAtNet-DeepMoE</td><td>99.80</td><td>99.80</td><td>99.80</td><td>99.80</td></tr></table>

Our proposed CoAtNet-DeepMoE model was evaluated on the Kaggle dataset and compared with several SOTA models, as presented in Table 5. The comparison includes the number of parameters, Accuracy, Precision, Recall, and F1-score.

![](images/650fbc5b0fe74896b54885e99e5cbc5a6ad43e18e9b2037856a1fb36d2826c4e.jpg)

(a) Kaggle  
![](images/2a41dac2457e9f566fe92e324980835bf04dc831b95d9a136806d110f02a4fea.jpg)  
(b) PlantVillage  
Figure 3: 2D comparison of SOTA models on (a) the Kaggle dataset and (b) the PlantVillage dataset. Each plot presents model accuracy against parameter count (in millions), illustrating how different architectures balance predictive performance and model complexity.

Model parameter counts were taken directly from the corresponding papers; when unavailable, we used the smallest official model variant and computed the parameters ourselves. The dataset was split according to the partition provided by the original authors, and the confusion matrix of CoAtNet-DeepMoE for all classes is shown in Figure 4a. As observed, CoAtNet-DeepMoE achieves the best performance across all metrics. The closest model in terms of Accuracy is VGG16+NASNet [51], which reaches 98.7% accuracy; however, it requires approximately 58 times more parameters than CoAtNet-DeepMoE (143.30M vs. 2.47M).

On the other hand, [38] is the closest in terms of parameter count, with 11.00M parameters, approximately 4.5 times more than our model, yet its performance is 6.6% lower in Accuracy and 9.5% lower in Precision. These results demonstrate the strong feature-extraction capability of CoAtNet-DeepMoE while significantly reducing the number of parameters through the integration of DeepSeek-MoE. Figure 3a further illustrates how CoAtNet-DeepMoE surpasses existing models in both Accuracy and model size.

![](images/4089b3a5ee9d4f9ec85951bd05d43f84d4fd6058a18a0047818d426090c406d8.jpg)  
(a) Kaggle

![](images/c51421aa7d57b00ba6efbfc68a542fcc216e8b5d3238e365eeec5655d51d3639.jpg)  
(b) PlantVillage  
Figure 4: Normalized confusion matrices of the proposed CoAtNet-DeepMoE model on (a) the Kaggle tomato leaf disease dataset and (b) the PlantVillage tomato dataset.

## 3.1.2 Performance on the PlantVillage dataset and compare with the SOTA models:

Our proposed CoAtNet-DeepMoE model was further evaluated on the PlantVillage dataset and compared with several SOTA models, as shown in Table 6. Similar to the earlier evaluation, the comparison includes the number of parameters, Accuracy, Precision, Recall, and F1-score. The parameter counts of the baseline models were followed as reported in the Kaggle implementations. For fairness, we used the same train–test split provided by [52], as their split configuration is widely adopted and yields the most consistent results. The confusion matrix of CoAtNet-DeepMoE for all classes is presented in Figure 4b.

Table 6: Performance comparison of state-of-the-art (SOTA) models on the PlantVillage dataset. “M” denotes the number of parameters in millions. Accuracy, Precision, Recall, and F1-score are reported in percentage (%) form. Bold values indicate the best performance among the compared models.
<table><tr><td>SL</td><td>Reference</td><td>Parameters</td><td>Model Name</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>1</td><td>[53]</td><td>138.36 M</td><td>VGG16</td><td>96.46</td><td>一</td><td>一</td><td>一</td></tr><tr><td>2</td><td>[53]</td><td>3.50 M</td><td>MobileNetV2</td><td>95.76</td><td>一</td><td>一</td><td>一</td></tr><tr><td>3</td><td>[53]</td><td>27.53 M</td><td>Swin Transformer</td><td>99.10</td><td>一</td><td>一</td><td>一</td></tr><tr><td>4</td><td>[53]</td><td>8.06 M</td><td>Vision Transformer</td><td>99.45</td><td>一</td><td>一</td><td>一</td></tr><tr><td>5</td><td>[42]</td><td>25.60 M</td><td>ResNet50</td><td>99.80</td><td>一</td><td>一</td><td>一</td></tr><tr><td>6</td><td>[52]</td><td>5.70 M</td><td>ViT</td><td>96.50</td><td>93.90</td><td>96.70</td><td>94.20</td></tr><tr><td>7</td><td>Our</td><td>2.47 M</td><td>CoAtNet-DeepMoE</td><td>99.83</td><td>99.85</td><td>99.76</td><td>99.80</td></tr></table>

As observed, CoAtNet-DeepMoE again achieves the best performance across all metrics. While Swin Transformer [53] and ResNet50 [42] report strong accuracies of 99.1% and 99.8%, respectively, they still fall short of the 99.83% achieved by CoAtNet-DeepMoE. Additionally, their parameter sizes are approximately 11 and 10 times larger than ours, respectively. Furthermore, although the ViT model in [52] is lightweight with 5.70M parameters, it yields noticeably lower Accuracy, Precision, Recall, and F1-score, falling behind by 3.3%, 6.0%, 3.1%, and 5.6%, respectively.

In contrast, CoAtNet-DeepMoE attains 99.83% accuracy, 99.85% precision, 99.76% recall, and 99.80% F1-score with only 2.47M parameters. These results highlight the superior efficiency and strong feature-extraction capability of CoAtNet-DeepMoE, driven by its parameter-efficient mixture-of-experts design. Figure 3b further illustrates its advantage in both accuracy and model size compared to other approaches.

## 3.2 Ablation Study

In addition to comparisons with state-of-the-art models, we conducted several ablation studies to validate the robustness and parameter efficiency of the proposed CoAtNet-DeepMoE. To demonstrate robustness, we considered an imbalanced dataset, specifically the PlantVillage dataset, and performed experiments to evaluate the model’s ability to generalize and maintain high performance despite class imbalance.

## 3.2.1 Cross-Dataset Evaluation

To assess the generalization capability of the trained models, we conducted cross-dataset evaluations. Specifically, models trained on the PlantVillage dataset were tested on the Kaggle tomato leaf disease dataset, and conversely, models trained on the Kaggle dataset were tested on the PlantVillage dataset. Both datasets consist of ten classes of tomato leaf diseases. The resulting normalized confusion matrices for these cross-dataset experiments are presented in Figure 5. These matrices provide a clear visualization of the model’s ability to generalize across datasets with differing image characteristics, including variations in lighting, background, and acquisition conditions. They also indicate which classes are accurately predicted and highlight the classes that are more prone to misclassification under unseen dataset distributions.

![](images/71242c2d7da5804d9e66c185d2d068e7ee439198e8d338ce8a845134ae108ed1.jpg)  
(a) Confusion matrix of the Kaggle dataset evaluated on the model trained with the PlantVillage dataset.

![](images/6add2ad7ed6b0579564c126fec03920f9a206a718899bdcb6a85120fa682c6c0.jpg)  
(b) Confusion matrix of the PlantVillage dataset evaluated on the model trained with the Kaggle dataset.  
Figure 5: Normalized confusion matrices for cross-dataset evaluation.

If we observe Table 7, which presents class-wise accuracy, precision, recall, and F1-score for cross-dataset evaluation, we notice that the BS, SE, and TS classes exhibit slightly lower performance compared to the others, with a few instances misclassified in both datasets. The remaining classes achieve near-perfect scores in one dataset but may have minor errors in the other. Overall, these experiments demonstrate that CoAtNet-DeepMoE maintains high classification performance and effectively generalizes to unseen data, even when trained on imbalanced datasets.

## 3.2.2 Inference Cost Analysis

We evaluated CoAtNet-DeepMoE alongside CoAtNet-Base, ResNet50, and ViT-Tiny under identical hardware and measurement conditions. As shown in Table 8, CoAtNet-DeepMoE requires the fewest parameters (2.47M) and the lowest computational cost (1.34 GFLOPs), being 2.2 to 10.8 times smaller and 1.6 to 6.3 times less compute-intensive than the other models. In terms of raw per-image latency, ViT-Tiny is marginally the fastest on average (3.78 ms vs. 4.04 ms for CoAtNet-DeepMoE); however, CoAtNet-DeepMoE achieves the lowest tail latency, with the best p95 (4.19

Table 7: Class-wise performance of the proposed model under cross-dataset evaluation. The model is trained on one dataset and evaluated on the other. Accuracy, Precision, Recall, and F1-score are reported as decimal values. The Average row represents the macro-average across all ten classes.
<table><tr><td colspan="5">PlantVillage → Kaggle</td><td colspan="4">Kaggle → PlantVillage</td></tr><tr><td>Class</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-score</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>BS</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9994</td><td>1.0000</td><td>0.9951</td><td>0.9975</td></tr><tr><td>EB</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9997</td><td>1.0000</td><td>0.9953</td><td>0.9977</td></tr><tr><td>LB</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.9994</td><td>0.9949</td><td>1.0000</td><td>0.9974</td></tr><tr><td>LM</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>SLS</td><td>0.9990</td><td>0.9901</td><td>1.0000</td><td>0.9950</td><td>0.9997</td><td>1.0000</td><td>0.9971</td><td>0.9986</td></tr><tr><td>SE</td><td>0.9990</td><td>0.9901</td><td>1.0000</td><td>0.9950</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>TS</td><td>0.9990</td><td>1.0000</td><td>0.9900</td><td>0.9950</td><td>0.9992</td><td>0.9885</td><td>1.0000</td><td>0.9942</td></tr><tr><td>YLCV</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>MV</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Healthy</td><td>0.9990</td><td>1.0000</td><td>0.9900</td><td>0.9950</td><td>0.9997</td><td>1.0000</td><td>0.9970</td><td>0.9985</td></tr><tr><td>Average</td><td>0.9996</td><td>0.9980</td><td>0.9980</td><td>0.9980</td><td>0.9997</td><td>0.9983</td><td>0.9985</td><td>0.9984</td></tr></table>

ms) and p99 (4.24 ms) values among all four models, indicating more consistent inference behavior across repeated runs. CoAtNet-Base is the slowest and most compute-heavy model overall, at 6.79 ms/img and 8.42 GFLOPs.  
![](images/3634675c6bf2a3836f90289f93f151c22c6e632aa378db9b99defaaf962cfb21.jpg)  
Figure 6: Cost–latency trade-off among CoAtNet-Base, ResNet50, ViT-Tiny, and the proposed CoAtNet-DeepMoE. The x-axis shows computational cost (GFLOPs), the y-axis shows inference latency (ms/img), and bubble size represents the number of parameters (M). The dashed line marks the Pareto frontier of non-dominated models.

Figure 6 visualizes this trade-off, plotting GFLOPs against inference latency with bubble size encoding parameter count. CoAtNet-DeepMoE and ViT-Tiny lie on the Pareto frontier, jointly dominating CoAtNet-Base and ResNet50 in both cost and latency, while CoAtNet-DeepMoE additionally achieves this with less than half the parameters of

Table 8: Cost analysis of different models in terms of parameter count (M), computational complexity (GFLOPs), and inference latency (ms/img), measured on the same hardware. p95/p99 denote the 95th/99th-percentile per-image latency over repeated runs. All metrics follow a lower-is-better (↓) convention, except FPS, which is higher-is-better (↑). Bold values indicate the most efficient performance among the compared models.
<table><tr><td>Model</td><td>Params (M)↓</td><td>GFLOPs↓</td><td>Time/img (ms)↓</td><td>p95 (ms).↓</td><td>p99 (ms).↓</td><td>FPS↑</td></tr><tr><td>CoAtNet-Base</td><td>26.67</td><td>8.42</td><td>6.79</td><td>6.90</td><td>6.98</td><td>147.3</td></tr><tr><td>ResNet50</td><td>23.53</td><td>8.26</td><td>4.47</td><td>4.65</td><td>4.70</td><td>223.6</td></tr><tr><td>ViT-Tiny</td><td>5.53</td><td>2.15</td><td>3.78</td><td>4.93</td><td>5.02</td><td>264.8</td></tr><tr><td>CoAtNet-DeepMoE</td><td>2.47</td><td>1.34</td><td>4.04</td><td>4.19</td><td>4.24</td><td>247.6</td></tr></table>

ViT-Tiny. Overall, CoAtNet-DeepMoE delivers the best parameter and computational efficiency among the compared architectures, while remaining competitive with the fastest model (ViT-Tiny) in latency and throughput, underscoring its suitability for low-resource and real-time applications.

## 4 Discussion

The experimental results across both the Kaggle and PlantVillage tomato disease datasets demonstrate that the proposed CoAtNet-DeepMoE model achieves state-of-the-art performance (Table 5 and Table 6) while maintaining exceptionally low computational cost. Compared to traditional convolutional and residual networks (e.g., ResNet50) and recent transformer-based architectures (e.g., ViT-Tiny, CoAtNet-Base), CoAtNet-DeepMoE consistently yields higher accuracy, precision, recall, and F1-score, despite having significantly fewer parameters.

We believe that the DeepSeek-MoE mechanism can have a substantial impact on vision applications for agriculture. In leaf disease classification, for example, certain feature patterns, such as healthy leaf texture or background characteristics, are common across classes and can be efficiently captured by the shared expert space, which acts as a repository for universally relevant information. At the same time, disease-specific variations (e.g., lesion shape, color distortion, or infection boundary) can be processed by the top-k experts selected dynamically by the routing function based on the input. This selective routing enables each expert to specialize in identifying distinct disease-related features while avoiding unnecessary computation. Furthermore, DeepSeek-MoE inherits the core advantages of traditional mixture-of-experts architectures, where a large model is decomposed into several smaller subnetworks (experts). The addition of a shared expert enhances this design, enabling extensive parameter reduction without compromising classification performance. Overall, this structure allows high representational capacity with low computational cost, making it particularly well-suited for agricultural vision tasks.

## This performance gain can be attributed to two key design principles:

First, the hybrid convolution-attention structure of CoAtNet [28] provides stronger spatial modeling than pure CNNs and better local feature extraction than pure transformers. Specifically, the CoAtNet model initially applies convolutional layers to extract local features, followed by MbConv blocks for channel-wise feature refinement, SE blocks for enhanced feature recalibration, and attention layers to capture global dependencies. This combination allows the model to simultaneously extract both local and global features, which is critical for accurate disease classification. In contrast, other popular models such as MobileNetV4, EfficientNet-V2-S, and ConvNeXt-V2 rely solely on convolutional operations and do not incorporate attention mechanisms, limiting their ability to model long-range dependencies.

Second, the parameter-efficient DeepSeek-MoE mechanism selectively routes features through specialized experts, enabling richer and more diverse representations while drastically reducing computational overhead compared to traditional mixture-of-experts models. As observed in the ablation study, the compared models require 2.2 to 10.8 times more parameters than CoAtNet-DeepMoE, and in the broader SOTA comparison several models require substantially more parameters yet still achieve lower performance, demonstrating that a higher parameter count does not necessarily translate to better accuracy. This efficient design allows CoAtNet-DeepMoE to maintain high representational capacity with minimal computational cost, making it particularly effective for real-time and resource-constrained applications.

## 5 Conclusion

This study presents CoAtNet-DeepMoE, a parameter-efficient hybrid convolution-attention model with a DeepSeek mixture-of-experts design for tomato leaf disease classification. Evaluated on two benchmark datasets, it achieves SOTA performance while using only 2.47M parameters, the lowest GFLOPs, and the fastest tail latency, outperforming several state-of-the-art models. The confusion matrices (Fig. 4a and Fig. 4b) confirm near-perfect classification across all disease categories, and the model demonstrates strong generalizability under different imaging conditions. With its small footprint and low latency, CoAtNet-DeepMoE is well-suited for real-time and edge-based agricultural applications. Future work may focus on field-level deployment, more diverse imagery, and integration with early-warning decisionsupport systems, further enhancing its practical impact.

## References

[1] Jonas Stehl, Lutz Depenbusch, and Sebastian Vollmer. Global poverty and the cost of a healthy diet. Food Policy, 132:102849, 2025.

[2] Jatin Sharma, Asma A Al-Huqail, Ahmad Almogren, Hardik Doshi, B Jayaprakash, B Bharathi, Ateeq Ur Rehman, and Seada Hussen. Deep learning based ensemble model for accurate tomato leaf disease classification by leveraging resnet50 and mobilenetv2 architectures. Scientific Reports, 15(1):13904, 2025.

[3] Aritra Das, Fahad Pathan, Jamin Rahman Jim, Md Mohsin Kabir, and MF Mridha. Deep learning-based classification, detection, and segmentation of tomato leaf diseases: A state-of-the-art review. Artificial Intelligence in Agriculture, 2025.

[4] Flávia Barbosa Abreu, Fabiano Ricardo Brunele Caliman, Adilson Castro Antonio, Vinood B Patel, et al. Tomatoes: origin, cultivation techniques and germplasm resources. In Tomatoes and tomato products, pages 3–25. CRC Press, 2008.

[5] Luqmon Azeez, Segun A Adebisi, Abdulrasaq O Oyedeji, Rasheed O Adetoro, and Kazeem O Tijani. Bioactive compounds’ contents, drying kinetics and mathematical modelling of tomato slices influenced by drying temperatures and time. Journal ofthe Saudi Society ofAgricultural Sciences, 18(2):120–126, 2019.

[6] Mohammed Mustafa, Ruth W Mwangi, Zita Szalai, Noémi Kappel, and László Csambalik. Sustainable responses to open field tomato (solanum lycopersicum l.) stress impacts. Journal ofAgriculture and Food Research, page 101825, 2025.

[7] Tintswalo Molelekoa, Edwin M Karoney, Nazareth Siyoum, Jarishma K Gokul, and Lise Korsten. Quality and quantity losses of tomatoes grown by small-scale farmers under different production systems. Horticulturae, 11 (8):884, 2025.

[8] Mario Sánchez Sánchez, Emmanuel Aispuro Hernández, Eber A Quintana-Obregón, Irasema Vargas Arispuro, and Miguel Ángel Martínez Téllez. Estimating tomato production losses due to plant viruses, a look at the past and new challenges. Comunicata Scientiae, 15:71, 2024.

[9] SW Zhang, YJ Shang, and L Wang. Plant disease recognition based on plant leaf image. 2015.

[10] Intan Nurma Yulita, Naufal Ariful Amri, and Akik Hidayat. Mobile application for tomato plant leaf disease detection using a dense convolutional network architecture. Computation, 11(2):20, 2023.

[11] Chuanqi Xie, Yongni Shao, Xiaoli Li, and Yong He. Detection of early blight and late blight diseases on tomato leaves using hyperspectral imaging. Scientific reports, 5(1):1–11, 2015.

[12] NJK Madufor, WJ Perold, and UL Opara. Detection of plant diseases using biosensors: a review. In VII International Conference on Managing Quality in Chains (MQUIC2017) and II International Symposium on Ornamentals in 1201, pages 83–90, 2017.

[13] Dominique Blancard. Tomato diseases. Academic Press Cambridge, MA, USA:, 2012.

[14] Naresh K Trivedi, Vinay Gautam, Abhineet Anand, Hani Moaiteq Aljahdali, Santos Gracia Villar, Divya Anand, Nitin Goyal, and Seifedine Kadry. Early detection and classification of tomato leaf disease using high-performance deep neural network. Sensors, 21(23):7987, 2021.

[15] Mukesh Kumar Choudhary and Saroj Hiranwal. Feature selection algorithms for plant leaf classification: A survey. In Proceedings ofInternational Conference on Communication and Computational Technologies: ICCCT-2019, pages 657–669. Springer, 2020.

[16] G Geetha, S Samundeswari, G Saranya, K Meenakshi, and M Nithya. Plant leaf disease classification and detection system using machine learning. In Journal of Physics: Conference Series, volume 1712, page 012012. IOP Publishing, 2020.

[17] Jagadeesh Basavaiah and Audre Arlene Anthony. Tomato leaf disease classification using multiple feature extraction techniques. Wireless Personal Communications, 115(1):633–651, 2020.

[18] Haridas D Gadade and DK Kirange. Machine learning based identification of tomato leaf diseases at various stages of development. In 2021 5th International Conference on Computing Methodologies and Communication (ICCMC), pages 814–819. IEEE, 2021.

[19] Ritesh Maurya, Lucky Rajput, and Satyajit Mohapatra. Rai-net: Tomato plant disease classification using residual-attention-inception network. IEEE Access, 2025.

[20] Andreas Kamilaris and Francesc X Prenafeta-Boldú. Deep learning in agriculture: A survey. Computers and electronics in agriculture, 147:70–90, 2018.

[21] Seyed Mohamad Javidan, Yiannis Ampatzidis, Ahmad Banakar, Keyvan Asefpour Vakilian, and Kamran Rahnama. Tomato fungal disease diagnosis using few-shot learning based on deep feature extraction and cosine similarity. AgriEngineering, 6(4):4233–4247, 2024.

[22] Kamaldeep Joshi, Sahil Hooda, Archana Sharma, Humira Sonah, Rupesh Deshmukh, Narendra Tuteja, Sarvajeet Singh Gill, and Ritu Gill. Precision diagnosis of tomato diseases for sustainable agriculture through deep learning approach with hybrid data augmentation. Current Plant Biology, 41:100437, 2025.

[23] Md Zasim Uddin, Md Nadim Mahamood, Ausrukona Ray, Md Ileas Pramanik, Fady Alnajjar, and Md Atiqur Rahman Ahad. E2etca: End-to-end training of cnn and attention ensembles for rice disease diagnosis1. Journal of Integrative Agriculture, 2024.

[24] Jinyi Wu, Jiaxin Dong, Shengwei Chen, Jie Wang, and Xin He. Tomato leaf disease classification based on feature enhancement and sde-resnet50. 2024.

[25] Huiqun Hong, Jinfa Lin, and Fenghua Huang. Tomato disease detection and classification by deep learning. In 2020 International Conference on Big Data, Artificial Intelligence and Internet ofThings Engineering (ICBAIE), pages 25–29. IEEE, 2020.

[26] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[27] Aarthi Chelladurai, DP Manoj Kumar, SS Askar, and Mohamed Abouhawwash. Classification of tomato leaf disease using transductive long short-term memory with an attention mechanism. Frontiers in Plant Science, 15: 1467811, 2025.

[28] Zihang Dai, Hanxiao Liu, Quoc V Le, and Mingxing Tan. Coatnet: Marrying convolution and attention for all data sizes. Advances in neural information processing systems, 34:3965–3977, 2021.

[29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016.

[30] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

[31] Haridas D Gadade and DK Kirange. Tomato leaf disease diagnosis and severity measurement. In 2020 fourth world conference on smart trends in systems, security and sustainability (WorldS4), pages 318–323. IEEE, 2020.

[32] Bibek Joshi, Sandhya Bansal, and Kavita Gupta. Precision classification of tomato leaf diseases using genetic algorithm and knn-based feature optimization. In International Conference on Mobile Radio Communications & 5G Networks, pages 509–521. Springer, 2024.

[33] Rashid Khan, Nasir Ud Din, Asim Zaman, and Bingding Huang. Automated tomato leaf disease detection using image processing: An svm-based approach with glcm and sift features. Journal ofEngineering, 2024(1):9918296, 2024.

[34] Md Hasan Imam, Nazmun Nahar, Ronok Bhowmik, Shudeb Babu Sen Omit, Tanjim Mahmud, Mohammad Shahadat Hossain, and Karl Andersson. A transfer learning-based framework: Mobilenet-svm for efficient tomato leaf disease classification. In 2024 6th International Conference on Electrical Engineering and Information & Communication Technology (ICEEICT), pages 693–698. IEEE, 2024.

[35] Hakan Terzioglu, Adem Gölcük, Adnan Mohammad Anwer Shakarji, and Mateen Yilmaz Al-Bayati. Comparative˘ analysis of deep learning-based feature extraction and traditional classification approaches for tomato disease detection. Agronomy, 15(7):1509, 2025.

[36] Keiron O’shea and Ryan Nash. An introduction to convolutional neural networks. arXiv preprint arXiv:1511.08458, 2015.

[37] Md Assaduzzaman, Prayma Bishshash, Md Asraful Sharker Nirob, Ahmed Al Marouf, Jon G Rokne, and Reda Alhajj. Xse-tomatonet: An explainable ai based tomato leaf disease classification method using efficientnetb0 with squeeze-and-excitation blocks and multi-scale feature fusion. MethodsX, 14:103159, 2025.

[38] Akram Abdullah, Gehad Abdullah Amran, SM Ahanaf Tahmid, Amerah Alabrah, Ali A AL-Bakhrani, and Abdulaziz Ali. A deep-learning-based model for the detection of diseased tomato leaves. Agronomy, 14(7):1593, 2024.

[39] SMUN Prabhasha, WPDJN Wijesinghe, BND Jayasinghe, and J Tuvensha. Tomato leaf disease classification using deep learning techniques. 2025.

[40] Shengyi Zhao, Yun Peng, Jizhan Liu, and Shuo Wu. Tomato leaf disease diagnosis based on improved convolution neural network by attention module. Agriculture, 11(7):651, 2021.

[41] Ramamurthy Karthik, M Hariharan, R Menaka, et al. Attention embedded residual cnn for disease detection in tomato leaves. Applied Soft Computing, 86:105933, 2020.

[42] CK Sunil, CD Jaidhar, et al. Tomato plant disease classification using multilevel feature fusion with adaptive channel spatial and pixel attention mechanism. Expert Systems with Applications, 228:120381, 2023.

[43] Siyuan Mu and Sen Lin. A comprehensive survey of mixture-of-experts: Algorithms, theory, and applications. arXiv preprint arXiv:2503.07137, 2025.

[44] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 4510–4520, 2018.

[45] Sifre Laurent. Rigid-motion scattering for image classification. Ph. D. thesis section, 6(2), 2014.

[46] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

[47] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

[48] PyTorch. torch.optim.lr\_scheduler.CosineAnnealingLR. https://pytorch.org/docs/stable/generated/ torch.optim.lr\_scheduler.CosineAnnealingLR.html, n.d. Accessed: 2025-11-25.

[49] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.

[50] Emine Cengil and Ahmet Çınar. Hybrid convolutional neural network based classification of bacterial, viral, and fungal diseases on tomato leaf images. Concurrency and Computation: Practice and Experience, 34(4):e6617, 2022.

[51] Senthilkumar AM, PRAVEEN JOE IR, Shravan Venkatraman, Pavan Kumar S, et al. Improved tomato leaf disease classification through adaptive ensemble models with exponential moving average fusion and enhanced weighted gradient optimization. Frontiers in Plant Science, 15:1382416, 2024.

[52] Divas Karimanzira. Context-aware tomato leaf disease detection using deep learning in an operational framework. Electronics, 14(4):661, 2025.

[53] Zhichao Chen, Guoqiang Wang, Tao Lv, and Xu Zhang. Using a hybrid convolutional neural network with a transformer model for tomato leaf disease detection. Agronomy, 14(4):673, 2024.