# A Lightweight Attention-Based CNN with Saliency-Map Interpretability for Skin Cancer Detection

Sanjana Islam Kasfia   
Dept. of Computer Science and Engineering International Standard University Dhaka, Bangladesh   
Email: sanjanaislamkasfia2648@gmail.com   
AKM Bahalul Haque   
Faculty of Science & Engineering   
Abo Akademi University   
Turku, Finland   
Email: akmbahalul.haque@abo.fi

Md. Ismiel Hossen Abir Dept. of Public Health and Life Sciences Canadian University of Bangladesh Dhaka, Bangladesh Email: ismielabir286@gmail.com

Abir Mahmud Shahariar Dept. of Computer Science and Engineering International Standard University Dhaka, Bangladesh Email: abirmahmudshahariar@gmail.com

Abstract—Skin cancer is among the most prevalent and lifethreatening forms of cancer worldwide. Early and accurate diagnosis significantly improves patient survival rates. However, manual dermoscopic assessment remains challenging due to visual similarity. This study proposes an Explainable AI-driven framework for automated binary classification of skin lesions using a lightweight attention-based convolutional neural network (CNN-Attention). The model integrates channel attention to focus on informative regions of dermoscopic images while maintaining low computational complexity. A publicly available dataset comprising 2,637 training and 660 test images was used, with balanced class weights. The proposed model achieved 87% accuracy with 1.95 million parameters and 7.45 MB storage, matching the best baseline (MobileNet) while using approximately half the parameters and storage. ROC-AUC and PR-AUC scores reached 0.9475 and 0.9430, respectively. Additionally, saliency maps were incorporated as an explainable AI technique to visualize regions influencing model decisions. Comparative analysis with VGG16, VGG19, ResNet101, and MobileNet demonstrates the proposed model’s effectiveness in balancing classification results, computational efficiency, and interpretability.

Index Terms—Skin cancer, CNN, attention mechanism, explainable AI, saliency maps.

## I. INTRODUCTION

Skin cancer is one of the most widespread cancer types globally [1]. It refers to various malignant tumors that develop from abnormal skin cell growth [2]. The uncontrolled growth of DNA structures within damaged skin cells leads to this condition [3]. The primary types are basal cell carcinoma, squamous cell carcinoma, and melanoma [4]. Melanoma is the most dangerous form [3]. Most cases result from exposure to ultraviolet radiation from sunlight [5]. It spreads quickly and causes high mortality rates [6]. Early identification of malignant skin lesions is critical [6]. Doctors must promptly distinguish harmful lesions from harmless ones [3]. This early recognition is essential for successful treatment. It directly improves patient survival and recovery prospects. Dermoscopic imaging serves as a valuable non-invasive tool [7]. It provides magnified and illuminated views of lesion structures [8]. These structures are often invisible to the bare eyes [1]. However, diagnosis remains challenging. Different lesion types appear visually similar [1]. Lesions vary in shape, color, texture, and borders [6]. This variation makes reliable diagnosis difficult.

Traditional assessment methods face significant limitations. This has increased focus on computer-aided diagnostic systems for automated skin lesion analysis [9]. These systems perform automated skin lesion analysis. Standard diagnostic methods involve visual examination, dermoscopy, biopsy, and clinical assessment procedures [3]. These methods present difficulties. These methods require significant time and expert personnel [10]. Earlier computer-aided systems relied on manual feature engineering [11]. These included color, texture, structure, and morphological characteristics [12]. Traditional machine-learning classifiers evaluated these features [12]. However, significant limitations existed. Results depended on the quality and relevance of feature selection. Dermoscopic images from actual patients posed many challenges [13]. These challenges included inadequate contrast, hair presence, reflections, shadows, and assorted artifacts [13]. As deep learning has advanced, convolutional neural networks (CNNs) have become powerful tools for analyzing skin lesions [12]. These networks automatically learn hierarchical features from image data without manual descriptor engineering [2]. Transfer learning with CNNs has proven beneficial for skin lesion classification [9]. Pretrained architectures can effectively classify skin lesions [4]. This is especially valuable when medical image datasets are small. Even with all this progress, there are still difficulties in achieving reliable and effective automated skin lesion classification. Skin lesion datasets often suffer from class imbalance [3]. Different lesion types share major similarities [7]. Lesions in the same category show significant variation [7]. As a result, Classification becomes inherently difficult. Additionally, image quality and acquisition conditions can introduce artifacts and variations [8]. These interfere with feature extraction and prediction [8]. Traditional deep CNN models use many parameters and demand high computational resources. Balancing accuracy with model size and efficiency becomes difficult. Recent studies have examined attention mechanisms and feature enhancement techniques. These help deep networks prioritize informative regions. They suppress less relevant information. Some researchers have looked into lightweight designs to reduce computational complexity without sacrificing important discriminative properties. These findings advocate the need for models that accurately classify skin lesions while focusing on salient features and reducing computational demands.

To address these issues, this study presents an attentionbased CNN framework for the automated classification of skin lesions. We also used transfer learning models for comparative analysis. To improve model transparency, we integrated Explainable Artificial Intelligence (XAI) with saliency maps that display which image regions influenced the model’s classification decisions. Additionally, we designed a lightweight CNN-Attention architecture that reduces computational demands while preserving adequate classification performance. We conducted a comparative analysis between our proposed model and the pretrained architecture. The contributions of this study are listed below: i. We developed an attention-based convolutional neural network framework that applies transfer learning for benign and malignant skin lesion categorization.

ii. We incorporated an attention mechanism to direct focus toward meaningful regions in dermoscopic images and created a resource-efficient architecture to lower computational requirements.

iii. We applied explainable AI, specifically saliency maps, which visualize the regions that influenced model predictions. iv. We did a comparative analysis of the proposed model with the pretrained architecture and measured their performance using standard evaluation metrics.

This research is structured across multiple sections, such as the literature review in section 2, methodology in section 3. Section 4 displays the experimental results. Section 5 includes discussion and comparative analysis. Section 6 concludes and then references.

## II. LITERATURE REVIEW

Recent studies have shown prominent use of several techniques for skin lesion identification. Researchers have used different feature extraction techniques. They have explored various learning architectures. This literature review section summarizes current research approaches.

![](images/99f5191d957707710aa59f44dcb928d9fb991f2ea9f8de854b7ea922096b56eb.jpg)  
Fig. 1. Workflow Diagram

## A. Traditional and Feature-Based Methods

Salomi et al. developed skin cancer detection system that combines convolutional neural networks, ABCD rule-based feature extraction, and K-means clustering to automatically classify lesions [14]. Additionally, they deployed an Android application to ensure timely and widely available skin cancer detection services [14]. Hatem et al. constructed a skin lesion classification system using MATLAB and the K-nearest neighbor (KNN) algorithm to classify lesions as normal or malignant [15]. His solution confirmed that KNN is a practical approach for accurate and efficient skin lesion classification [15].

## B. Deep Learning and Transfer Learning

Shetty et al. used ML models alongside a custom CNN on the HAM10000 dataset [2]. Their custom CNN achieved 95.18% accuracy [2]. The authors propose further advancement through the application of CNN fine-tuning, the use of pretrained models, and the deployment of real-time segmentation and classification methods [2]. Bozkurt et al. utilized a pretrained Inception-ResNet-v2 model with affine data augmentation [3]. Their work addressed skin lesion classification [3]. Results showed improvement from 83.59% to 95.09% accuracy on the augmented dataset [3]. Their study primarily focused on increasing dataset size and improving classification accuracy [3]. A hybrid LSTM-CNN framework was developed by Abohashish et al. for binary classification [16]. The model processed image patches to recognize spatial structures and extract local features from the HAM10000 dataset [16]. Shah et al. analyzed the use of ANN and CNN models for automating skin cancer detection [6]. The review confirmed that both approaches were capable of identifying skin lesions early and with operational capability [6]. Afza et al. introduced a hierarchical framework that integrates image fusion, superpixel-based lesion segmentation, ResNet-50 model, and feature optimization for skin cancer classification [5].

## C. Advanced and Efficient Deep Learning Approaches

Asif et al. developed SKINC-NET, an efficient deep learning model for classifying seven skin lesion types using the HAM10000 dataset [10]. The model achieved 98.54% accuracy while requiring less computational power and maintaining fewer trainable parameters than several competing transferlearning models [10]. Ali et al. presented an FCEDN-SpaSAbased segmentation method with an adaptive CNN component [17]. This system distinguishes normal lesions from malignant skin lesions [17]. The approach reduces energy usage, storage requirements, memory access operations, and training time [17]. Halawani et al. introduced a hybrid EViT-DenseNet169 model [1]. It merges global features obtained from Vision Transformer attention mechanisms with local features extracted from DenseNet169 [1]. The model classifies seven categories of skin lesions on the ISIC 2018 dataset [1]. Kumar et al. introduced an enhanced Falcon Finch deep convolutional neural network that uses optimization-based parameter adjustment for identifying skin cancer [8]. Their study focused on improving accuracy and reducing computational cost [8]. Reka et al. investigated quantum ML approaches for skin disease classification using the HAM10000 dataset [18]. The research incorporated a quantum convolutional neural network and a Quantum Support Vector Classifier, which were then compared against conventional pretrained models [18]. Kumar et al. developed SCC-Net, a deep CNN model intended for nine-class skin cancer classification on the ISIC-2019 dataset [11]. The model incorporates small kernels and dropout regularization to improve computational efficiency and reduce overfitting [11].

## III. METHODOLOGY

This section describes the overall method of CNN with Attention model for skin cancer classification. Figure 1 reflects the overall workflow of this study.

## A. Data Analysis and Preparation

In this study, we used a public dataset for binary classification of skin lesions [19]. It consists of dermoscopic images where the train data contains 2,637 images, and the test set contains 660 images. The train data comprises 1,440 benign cases and 1,197 images for malignant cases. Here, the dataset exhibits class imbalance, which was addressed during training through class weighting. All images are RGB and were resized to 224 × 224 pixels. Figure 2 shows sample images from both classes.

![](images/7c4dcff1dffdad9e2c20eac5361b89549c62c4da23d73e95ba2c2811601c2eed.jpg)  
Fig. 2. Sample Images

The train set of data was divided into two part, such as train and validation subsets. For this task, we used stratified split method where 85% for training and 15% for validation. To solve the imbalance issue, we used the class-weight method that were computed using the scikit-learn function. The class weights were approximately 0.915 for the benign class and 1.102 for the malignant class.

## B. Proposed CNN-Attention Model

We build a custom lightweight CNN-Attention model. The architecture is organized into four convolutional blocks, attention, and then fully connected layers. Figure 3 presents the proposed model.

The first block contains two 3x3 conv2d layers with 32 filters each. Also, included batchnormalization and for activation purpose we used ReLU. Then, the max-pooling layer contains a 2x2 kernel size to reduce the dimensions, and dropout of 0.2 is applied for regularization. After that, in the second block, we used 64 filters and followed the same pattern, except it increases the dropout rate to 0.25. Similarly, in the third block 128 filters used three convolutional layers, batch normalization, max-pooling, and a dropout rate of 0.3. The fourth block uses 256 filters with two convolutional layers, batch normalization, max-pooling, and dropout of 0.35. All convolutional layers use L2 regularization.

After the convolutional blocks, the channel attention mechanism inspired by the Squeeze-and-Excitation (SE) block is used [20]. The feature map from the last block used global average pooling. Then, we used two fully connected layers with 256 and 512 units, respectively, both using ReLU activation. Then, final dense layer with a sigmoid activation was used. After that, a second global average pooling layer and a classifier head consisting of three dense layers with 512, 256, and 128 units. Each dense layer included batchnormalization and dropout. In the output layer, used a dense layer with softmax activation.

![](images/a21da81db9a561105cf4987eba80c956858c433bd1b95ce0a8f680325a710cfe.jpg)  
Fig. 3. Proposed CNN-Attention Model Architecture

## C. Transfer Learning

To provide a comparative evaluation, four pretrained convolutional neural networks were adapted for the binary classification task: VGG16 [21], VGG19 [21], ResNet101 [22], and MobileNet [23]. In each case, the pretrained backbone was loaded with ImageNet weights, and the feature extractor. Then, a global average pooling layer was used with two dense layers 512 and 256 units. Also, batch normalization and dropout of 0.5 and 0.3 were included, respectively. The final output layer was a dense layer with softmax activation.

## D. Explainable AI and Evaluation Metrics

All models were compiled using the Adam optimizer and initial learning rate of 0.001. We also used sparse categorical cross-entropy for loss function. The proposed CNN-Attention model was trained using 100 epochs, 32 batch size, early stopping (patience 20, restoring best weights), a reduced learning rate on plateau callback (factor 0.2, patience 7), model checkpointing based on validation accuracy, and a custom callback that halted training when validation accuracy reached 99.5%. The baseline transfer learning models (VGG16, VGG19, ResNet101, and MobileNet) were trained with the same optimizer and loss, but with early stopping patience of 15 and reduced learning rate patience of 5. For training purposes, we used Kaggle notebook environment with GPU.

For model transparency, an enhanced saliency map was used as an explainable AI technique. Saliency maps were generated by computing the gradient of the predicted class score with respect to the input image to identify influential regions [24]. For evaluation, we used classification report, confusion matrix, ROC AUC, and PR AUC.

## IV. EXPERIMENTAL RESULTS

Table I presents the classification report. For the benign class, precision is 0.91, recall is 0.85, and F1-score is 0.88. For the malignant class, precision is 0.83, recall is 0.89, and F1-score is 0.86. Also, this report shows the accuracy score is 0.87. Both macro and weighted averages are also 0.87.

TABLE I  
CLASSIFICATION REPORT FOR BENIGN VS MALIGNANT DETECTION
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>Benign</td><td>0.91</td><td>0.85</td><td>0.88</td></tr><tr><td>Malignant</td><td>0.83</td><td>0.89</td><td>0.86</td></tr><tr><td>Accuracy</td><td></td><td>0.87</td><td></td></tr></table>

Figure 4 presents the confusion matrix of the CNN-Attention model. It reflects that model correctly predicted 307 benign cases and 268 malignant cases. However, the model shows some misclassifications, with 53 false positives where it predicts benign as malignant. On the other hand, 32 false negative that means malignant identify as benign. Overall, the model shows correct predictions in most cases with an accuracy of 0.87.

Figure 5 presents the ROC curves for the CNN-Attention model on both classes. Both the benign and malignant curves have an AUC score of 0.9475. In this case, the model can correctly distinguish benign from malignant cases.

Table II shows the Precision-Recall AUC (PR-AUC) scores for both classes. The benign class has a score of 0.9563, while the malignant class has a score of 0.9296. The average PR-AUC score is 0.9430.

![](images/25b201b4442ad1bc0e3b0765cc7d667e6be9b6545cbac4ee041633257167f127.jpg)  
Fig. 4. Confusion Matrix

![](images/41f84db51750aa225e1f539404e8ad81aaa26bcceda588dc1553ee21101999c2.jpg)  
Fig. 5. ROC

TABLE II  
PRECISION-RECALL AUC SCORES FOR BENIGN VS MALIGNANT DETECTION
<table><tr><td>Class</td><td>PR-AUC Score</td></tr><tr><td>Benign</td><td>0.9563</td></tr><tr><td>Malignant</td><td>0.9296</td></tr><tr><td>Average</td><td>0.9430</td></tr></table>

Figures 6 and 7 show the saliency maps of both benign and malignant classes. The left image is truly benign in Figure 6 and malignant in Figure 7.

The right image shows the saliency map. Both images show that the model correctly predicts them as benign or malignant, with confidence scores of 98.9% and 71.3%, respectively. The bright yellow and red spots in the saliency maps show the exact regions the model focused on to make a decision for a specific class. In both cases, the model focuses on the actual lesion area rather than the surrounding skin, confirming that its decisions are based on the correct features.

![](images/4c001009f61435f31cc6607b1f5fdf67f2d6c329d7e8db48a6eb18e24b338726.jpg)  
Fig. 6. Saliency Map for Benign Class

![](images/4eaf8a9351a9c199c4b1a98077a10c021370dd09858c6176681a594d93511b6b.jpg)  
Fig. 7. Saliency Map for Malignant Class

## V. DISCUSSION AND COMPARATIVE ANALYSIS

This section presents discussion and comparative analysis of the study. According to Table III, VGG16 and VGG19 both achieved 84% accuracy, whereas ResNet101 reached 83% accuracy. MobileNet achieved superior results with 87% accuracy, the highest among the baseline models. The proposed CNN-Attention model matched this performance level. It also achieved 87% accuracy. With respect to model complexity, VGG16 and VGG19 contained 15.11 million and 20.42 million parameters, respectively. ResNet101 possessed the largest parameter count at 43.84 million. MobileNet operated with minimal overhead. It required only 3.89 million parameters and consumed 14.84 MB of storage space. The proposed CNN-Attention model involved only 1.95 million parameters and 7.45 MB of storage. The CNN-Attention model provided competitive detection performance. It showed substantially lower parameter usage and storage requirements. The reduced complexity improves its applicability for efficient skin cancer detection.

TABLE III  
COMPARISON OF MODEL PERFORMANCE AND COMPLEXITY
<table><tr><td>Model</td><td>Accuracy</td><td>Parameters (M)</td><td>Size (MB)</td></tr><tr><td>VGG16</td><td>0.84</td><td>15.11</td><td>57.64</td></tr><tr><td>VGG19</td><td>0.84</td><td>20.42</td><td>77.90</td></tr><tr><td>ResNet101</td><td>0.83</td><td>43.84</td><td>167.25</td></tr><tr><td>MobileNet</td><td>0.87</td><td>3.89</td><td>14.84</td></tr><tr><td>CNN-Attention</td><td>0.87</td><td>1.95</td><td>7.45</td></tr></table>

## VI. CONCLUSION

In this study, we built a lightweight custom model to detect skin cancer. We classified skin images as benign and malignant. A lightweight CNN-Attention model was developed. It combines transfer learning and an attention mechanism. The mechanism focuses on important regions of skin lesions. The proposed model achieved 87% accuracy. Precision values were 0.91 for benign and 0.83 for malignant. The average PR-AUC score was 0.9430. The model used only 1.95 million parameters. It required 7.45 MB of storage. This made it smaller than baseline models. It achieved the same accuracy as MobileNet. We integrated Explainable AI using saliency maps. These maps show image regions that influenced classification decisions. Some previous studies reported higher accuracy. However, they did not include Explainable AI. This study merges three important aspects. These are classification performance, lower model complexity, and prediction interpretability. The framework integrates all three. Future studies will use a custom dataset and more advanced techniques. This will improve classification performance. It will further support practical skin cancer detection.

## ACKNOWLEDGEMENT

During the preparation of this work, the authors used Grammarly Premium, Gemini, and ChatGPT 5.0 to improve sentence structure, fix grammatical mistakes, enhance writing, and maintain coherence throughout the manuscript. After using these, the authors reviewed and edited the content as needed.

## REFERENCES

[1] H. T. Halawani, E. M. Senan, Y. Asiri, I. Abunadi, A. M. Mashraqi, and E. A. Alshari, “Enhanced early skin cancer detection through fusion of vision transformer and CNN features using hybrid attention of EViT-Dens169,” Sci. Rep., vol. 15, p. 34776, Oct. 2025.

[2] B. Shetty, R. Fernandes, A. P. Rodrigues, R. Chengoden, S. Bhattacharya, and K. Lakshmanna, “Skin lesion classification of dermoscopic images using machine learning and convolutional neural network,” Sci. Rep., vol. 12, p. 18134, Oct. 2022.

[3] F. Bozkurt, “Skin lesion classification on dermatoscopic images using effective data augmentation and pre-trained deep learning approach,” Multimed. Tools Appl., Nov. 2022.

[4] M. M. Musthafa, M. T R, V. K. V, and S. Guluwadi, “Enhanced skin cancer diagnosis using optimized CNN architecture and checkpoints for automated dermatological lesion classification,” BMC Med. Imaging, vol. 24, p. 201, Aug. 2024.

[5] F. Afza, M. Sharif, M. Mittal, M. A. Khan, and D. Jude Hemanth, “A hierarchical three-step superpixels and deep learning framework for skin lesion classification,” Methods, vol. 202, pp. 88–102, June 2022.

[6] A. Shah, M. Shah, A. Pandya, R. Sushra, R. Sushra, M. Mehta, K. Patel, and K. Patel, “A comprehensive study on skin cancer detection using artificial neural network (ANN) and convolutional neural network (CNN),” Clin. eHealth, vol. 6, pp. 76–84, Dec. 2023.

[7] G. Yue, P. Wei, T. Zhou, Q. Jiang, W. Yan, and T. Wang, “Toward multicenter skin lesion classification using deep neural network with adaptively weighted balance loss,” IEEE Trans. Med. Imaging, vol. 42, pp. 119–131, Jan. 2023.

[8] A. Kumar, M. Kumar, V. P. Bhardwaj, S. Kumar, and S. Selvarajan, “A novel skin cancer detection model using modified finch deep CNN classifier model,” Sci. Rep., vol. 14, p. 11235, May 2024.

[9] K. Rezaee and H. G. Zadeh, “Self-attention transformer unit-based deep learning framework for skin lesions classification in smart healthcare,” Discov. Appl. Sci., vol. 6, Jan. 2024.

[10] S. Asif, Qurrat-ul-Ain, S. U. R. Khan, K. Amjad, and M. Awais, “SKINC-NET: an efficient lightweight deep learning model for multiclass skin lesion classification in dermoscopic images,” Multimed. Tools Appl., vol. 84, pp. 12531–12557, June 2024.

[11] V. A. Kumar, C. Chandana, G. Supraja, D. Haripriya, and E. Ravalika, “SCCNet: Skin cancer detection and multi-class classification using deep CNN model with estimated disease probabilities,” SN Comput. Sci., vol. 5, July 2024.

[12] R. Sulthana, V. Chamola, Z. Hussain, F. Albalwy, and A. Hussain, “A novel end-to-end deep convolutional neural network based skin lesion classification framework,” Expert Syst. Appl., vol. 246, p. 123056, July 2024.

[13] S. A. Amiri, M. Nasrolahzadeh, Z. Mohammadpoory, and A. H. Z. Kordkheili, “Skin lesion classification via ensemble method on deep learning,” Multimed. Tools Appl., July 2024.

[14] M. Salomi, G. Daram, and S. S. Harshitha, “Early skin cancer detection using CNN-ABCD rule based feature extraction classification and kmeans clustering algorithm through android mobile application,” in 2024 Second International Conference on Emerging Trends in Information Technology and Engineering (ICETITE), pp. 1–5, IEEE, Feb. 2024.

[15] M. Q. Hatem, “Skin lesion classification system using a k-nearest neighbor algorithm,” Vis. Comput. Ind. Biomed. Art, vol. 5, p. 7, Mar. 2022.

[16] S. M. M. Abohashish, H. H. Amin, and E. I. Elsedimy, “Enhanced melanoma and non-melanoma skin cancer classification using a hybrid LSTM-CNN model,” Sci. Rep., vol. 15, p. 24994, July 2025.

[17] R. Ali, A. Manikandan, R. Lei, and J. Xu, “A novel SpaSA based hyperparameter optimized FCEDN with adaptive CNN classification for skin cancer detection,” Sci. Rep., vol. 14, p. 9336, Apr. 2024.

[18] S. S. Reka, H. L. Karthikeyan, A. J. Shakil, P. Venugopal, and M. Muniraj, “Exploring quantum machine learning for enhanced skin lesion classification: A comparative study of implementation methods,” IEEE Access, vol. 12, pp. 104568–104584, 2024.

[19] C. Fanconi, “Skin cancer: Malignant vs. benign.” Kaggle, 2017. Accessed: 2026-09-02.

[20] J. Hu, L. Shen, S. Albanie, G. Sun, and E. Wu, “Squeeze-and-Excitation networks,” Sept. 2017.

[21] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” Sept. 2014.

[22] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778, IEEE, June 2016.

[23] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “MobileNets: Efficient convolutional neural networks for mobile vision applications,” Apr. 2017.

[24] K. Simonyan, A. Vedaldi, and A. Zisserman, “Deep inside convolutional networks: Visualising image classification models and saliency maps,” Dec. 2013.