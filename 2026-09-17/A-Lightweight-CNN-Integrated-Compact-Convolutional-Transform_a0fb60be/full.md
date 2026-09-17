# A Lightweight CNN Integrated Compact Convolutional Transformer for Multi-Scale Feature Learning and reducing computational complexity for breast cancer mammography image detection and classification

Md Taimur Ahad (Corresponding Author) Department of Management   
North South University, Dhaka, Bangladesh Email: Taimur.ahad@northsouth.edu

Ainuddin Ahmed

Department of Management

North South University, Dhaka, Bangladesh

Email: ainuddin.ahmed.251@northsouth.edu

# A Lightweight CNN Integrated Compact Convolutional Transformer for Multi-Scale Feature Learning and reducing computational complexity for breast cancer mammography image detection and classification

## Abstract

This study proposes a novel Convolutional Neural Network integrated Compact Convolutional Transformer (CNN–CCT) model for automated breast cancer classification using mammographic imaging. The model integrates CNNs in CCT to extract both local texture features and global features from mammographic images. The CNN-extracted features are then reshaped into compact patch tokens using a CCT tokenizer, followed by the addition of positional embeddings to preserve spatial structure. A lightweight transformer encoder consisting of multi-head self-attention layers is employed to model long-range dependencies within the token sequence. Finally, a classification head with global average pooling and a dense SoftMax layer produces the final prediction for benign and malignant classes. The model was tested on 3 sets of breast cancer mammography. With only 250,435 parameters, the model consistently achieves strong performance across 5-fold cross-validation experiments on 2- class, 3-class, and 5-class mammographic datasets. The model achieved 99%–100% accuracy across 3 datasets, indicating robust generalization. Furthermore, Explainable AI (XAI) is integrated into the model to explain the breast cancer classification process to enhance clinical trust. The results indicate that the proposed framework is well-suited for computer-aided diagnosis systems, particularly in resource-constrained clinical environments. This study highlights the potential of combining convolutional and transformer-based architectures for improved medical image analysis. The novelty of the proposed CNN-integrated CCT overcomes the limitation of CNN’s gradient degradation in the last layers by integrating convolutional tokenization with transformer-based learning. Lighter than ViT, which is effective in capturing long-range dependencies, the model has also proven efficient in breast cancer classification by capturing long-range dependencies among breast tissue regions.

Keywords: Breast cancer detection, Convolutional Neural Network, CNN, Vision transformer, ViT, CAD, Machine Learning

## 1. Introduction

Breast cancer is one of the foremost global health concerns, as breast cancer significantly contributes to mortality across both industrialized and developing nations. According to the World Health Organization (WHO), approximately one in every ten newly diagnosed cancers worldwide is breast cancer, making it the most frequently diagnosed cancer among women. Recent reports from the International Agency for Research on Cancer (IARC) estimate that about 2.3 million women were diagnosed with breast cancer in 2022, resulting in nearly 670,000 deaths globally. In this context, early and accurate breast cancer detection is critical for a better treatment plan.

Breast cancer often presents subtle abnormalities such as lumps, architectural distortions, or tissue heterogeneity. These symptoms are identified through medical imaging, including mammography. Mammographic images play a vital role in breast cancer screening, specifically in dense breast tissues. The traditional breast cancer detection process involves manually inspecting features such as texture, shape, and intensity. Even mammographic interpretation depends on a doctor or pathologist's manual inspection. While these approaches provided initial improvements, the inspection is often affected by noise, low contrast, and imaging artifacts, which lead to diagnostic variability, human error, and high false-positive rates. Consequently, there is a growing demand for computer-aided diagnosis (CAD) systems that help clinicians improve diagnostic accuracy and consistency.

The advent of deep learning (DL) has significantly enhanced CAD systems by enabling the extraction of features from breast cancer mammography images, thereby improving the precision of cancer detection and classification. Among DL models, Convolutional Neural Networks (CNNs) in particular have demonstrated strong performance in breast cancer classification tasks. However, CNN-based models often struggle to capture long-range contextual dependencies and require substantial amounts of labeled data and computational resources. To fill this gap, transformer-based architectures have emerged as powerful alternatives, thanks to their ability to model global dependencies via self-attention mechanisms.

Among such transformer-based architectures, the Compact Convolutional Transformer (CCT) is a hybrid vision model that combines convolutional neural networks with transformer-based self-attention. It uses a convolutional tokenizer at the input stage to extract local spatial features and convert images into token sequences for transformer encoders (Hassani et al., 2021). This design helps the model learn both local patterns and global dependencies more evenly. CCT performs well in small- to medium-scale data settings where full-scale Vision Transformers often struggle. It also reduces the need for very large datasets by adding useful CNN-style inductive bias. Overall, it is more data-efficient and easier to train than standard Vision Transformers in many practical cases.

However, CCT still has some limitations. Transformer models require large amounts of data and are computationally expensive. The self-attention mechanism remains computationally intensive, especially as input sequences grow longer, leading to higher memory usage and slower inference. Even with convolutional tokenization, it can still be less efficient than lightweight CNN models in resource-constrained environments (Hosseini et al., 2025). The architecture also requires careful tuning of convolutional depth, kernel size, and pooling strategy, making model design less straightforward. Some studies also note that hybrid models may lose flexibility compared to pure transformer designs in certain structured vision tasks (EmergentMind, 2026).

To address these limitations, we propose a lightweight CNN-integrated CCT that efficiently learns both local and global features from mammographic images for breast cancer classification. Breast mammography analysis requires a simultaneous understanding of finegrained tissue variations and broader anatomical structures. This design is motivated by the nature of breast imaging, in which clinically relevant cues appear at multiple scales, ranging from subtle micro-level patterns, such as microcalcifications, to large structural abnormalities, such as masses and asymmetries. A single DL model might be insufficient to represent both types of features effectively, and therefore, a CNN-integrated CCT is adopted.

In the case of CCT, ablation studies are especially important as the model performance depends on convolutional tokenization, transformer encoder layers, and sequence pooling. Since CCT is designed to balance CNN-style local feature extraction with transformer-based global attention, it is unclear which component is responsible for the gains in accuracy or efficiency. Moreover, this study integrates a lightweight CNN into CCT; therefore, ablation studies help isolate the effects of the framework's components. Without ablation analysis, it is difficult to justify the claim that the hybrid design is truly necessary or optimal, rather than merely empirically effective. In this study, we conducted ablation studies to understand how each component of the CNN integrates CCT to improve performance, efficiency, and generalization in breast cancer detection and classification.

In this study, we have integrated XAI into the lightweight CNN-integrated CCT model. CCT is often treated as a black-box model, as it lacks explainability. This makes it difficult to fully interpret how convolutional tokenization and attention jointly contribute to predictions in CCT. Recent studies on vision transformers also highlight that explanation methods remain limited (Fantozzi & Naldi, 2024; Xie et al., 2023). Existing research on ViTs shows that even attentionbased explanations can be unreliable or incomplete for understanding model decisions (Xie et al., 2023).

The contribution of the study is the development of a CNN-integrated CCT framework that combines local feature extraction with global context learning for mammographic breast cancer classification. CNN models are effective in extracting local patterns but may struggle to capture relationships between distant regions. However, ViT is ineffective in identifying small local abnormalities due to early patch-based image division. In mammography, important signs such as microcalcifications, irregular margins, and architectural distortion are often small but related to surrounding tissue structures. Therefore, the proposed model uses convolutional tokenization to preserve meaningful local features before transformer processing. Instead of directly splitting images into fixed patches, the model learns informative feature tokens from convolutional representations, thereby better retaining subtle lesion patterns. The use of stochastic depth and a compact transformer design further improves model stability and reduces overfitting, making the framework suitable for medical image

## 2. Literature Review

Early deep learning approaches relied primarily on standalone CNN architectures for binary or multi-class classification of breast cancer images. For example, Simonyan et al. (2024) demonstrated the effectiveness of CNNs for histopathological classification, achieving competitive accuracy using standard architectures trained on publicly available datasets. Similarly, transfer learning approaches using pre-trained models such as ResNet, VGG, EfficientNet, and MobileNet have shown strong performance in breast cancer detection tasks (Elzaghmouri et al., 2025). These models benefit from learned representations from large-scale datasets such as ImageNet, allowing them to perform well even with limited medical imaging data. However, their performance is often constrained by dataset imbalance and domain shift between natural and medical images.

A major research trend involves enhancing CNNs with hybrid deep learning architectures to improve feature representation and classification accuracy. Kaddes et al. (2025) proposed a CNN–LSTM hybrid model for breast cancer classification, in which the CNN extracts spatial features and the LSTM captures sequential dependencies within the feature maps. The model achieved up to 99.90% accuracy, demonstrating the benefit of integrating temporal modeling with CNN-based spatial learning. Similarly, Brahmareddy and Selvan (2025) introduced TransBreastNet, a CNN–Transformer hybrid model capable of simultaneously classifying breast cancer subtypes and analyzing temporal lesion progression. The integration of transformers allows the model to capture long-range dependencies that CNNs typically struggle with. Sreelekshmi et al. (2024) also proposed a SwinCNN architecture that combines Swin Transformers with CNN layers for improved histopathological grading, achieving strong performance across multiple datasets, including BACH and BreakHis. These studies indicate that hybrid architectures consistently outperform traditional CNN models by combining local feature extraction with global contextual learning.

Another significant direction in the literature is feature fusion using multiple CNN architectures. Chakravarthy et al. (2024) proposed a multi-class classification system using hybrid feature fusion from VGG16, VGG19, ResNet50, and DenseNet121. This approach improves robustness by combining complementary feature representations from different networks. The proposed fusion-based model achieved accuracies exceeding 97% across multiple datasets, including MIAS, CBIS-DDSM, and INbreast. This demonstrates that ensemble CNN approaches can reduce model bias and improve generalization.

Several studies integrate traditional feature extraction techniques with CNN-based models to enhance performance and interpretability. Gül (2025) proposed a hybrid model that combines Local Binary Patterns (LBP) with a CNN for histopathological breast cancer diagnosis. The LBP method enhances texture representation, while CNN provides deep feature learning. Mannarsamy et al. (2025) introduced SIFT-BCD, which integrates the Scale-Invariant Feature Transform (SIFT) with CNN features and fuzzy decision-tree classifiers. The model achieved up to 99.20% accuracy, showing that hybrid classical-deep learning approaches can significantly improve performance in medical image classification. These approaches are particularly useful in histopathological imaging, where texture and structural variations play a critical role in diagnosis.

Recent advancements focus on improving CNN architectures through multi-scale processing, optimization algorithms, and dimensionality reduction techniques. Bohra et al. (2026) proposed a wavelet-CNN fusion architecture that leverages multi-resolution wavelet transforms to extract fine-grained texture features, achieving 99.34% accuracy on the BreakHis dataset. Similarly, Liu et al. (2024) introduced a kernel-based CNN with principal component feature fusion (CNN-PCFF), which reduces dimensionality while preserving discriminative features.

Alzahrani et al. (2025) enhanced CNN performance for thermography-based breast cancer detection by using Particle Swarm Optimization (PSO) to tune hyperparameters, achieving 98.8% accuracy. These studies demonstrate that optimization and multi-scale feature extraction significantly enhance CNN performance in medical imaging tasks.

Recent research has shifted toward integrating CNNs with Transformer architectures to capture both local and global features. Sreelekshmi et al. (2024) proposed SwinCNN, a hybrid model combining Swin Transformers with CNN layers for breast cancer grading. The model achieved strong performance across multiple datasets, demonstrating the advantage of hierarchical attention mechanisms. Abimouloud et al. (2024) introduced Vision Transformer-based CNN models for histopathological classification, demonstrating improved accuracy over standalone CNNs. Similarly, Katayama et al. (2024) highlighted the transition from CNNs to Vision Transformers in breast pathology, emphasizing improved global feature learning. Nayak (2024) proposed RDTNet, a residual deformable attention-based Transformer network achieving up to 99% accuracy across multiple magnification levels. This model demonstrates how attention-based architectures can significantly enhance lesion feature extraction. More advanced models such as

BreasTransNeXt (Acikgoz et al., 2026) integrate CNN inductive bias with transformer attention and GAN-based augmentation, achieving high precision and recall (>97%).

These studies confirm that hybrid CNN–Transformer models outperform traditional CNNs in capturing both fine-grained and global contextual features.

Transformers have gained significant attention in medical imaging due to their ability to model long-range dependencies. Goceri (2025) proposed a CNN-Transformer hybrid system with stain normalization for whole-slide image analysis, improving robustness in histopathological classification. Vanitha et al. (2024) introduced attention-based feature fusion using external attention transformers to enhance feature representation in breast cancer images. Abd Elaziz et al. (2024) proposed CrossViT, combined with the Growth Optimizer algorithm, for feature selection in IoMT environments, achieving improved classification accuracy. Furthermore, EAT (External Attention Transformer) models have demonstrated high efficiency, achieving up to 99% accuracy while maintaining low computational complexity. These studies highlight that transformer-based models are particularly effective in capturing global dependencies but often require hybridization to overcome data limitations.

Recent studies have also explored multimodal learning approaches that combine imaging and non-imaging data. Brahmareddy and Selvan (2025) proposed TransBreastNet, a CNN– Transformer hybrid system capable ofsimultaneously performing subtype classification and lesion progression analysis. The model integrates spatial, temporal, and clinical features, achieving over 95% accuracy. Similarly, research from the MICCAI workshop (2024) demonstrated multimodal fusion of imaging and textual data (radiology reports), showing that late fusion techniques improve classification performance across multiple architectures.

Table 1: Research matrix
<table><tr><td colspan="1" rowspan="1">Author(Year)</td><td colspan="1" rowspan="1">Method</td><td colspan="1" rowspan="1">Dataset</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1">KeyContribution</td><td colspan="1" rowspan="1">Gap</td></tr><tr><td colspan="1" rowspan="1">Kaddes et al.(2025)</td><td colspan="1" rowspan="1">CNN-LSTM</td><td colspan="1" rowspan="1">Kaggledatasets</td><td colspan="1" rowspan="1">99.90%</td><td colspan="1" rowspan="1">Spatial +temporallearning</td><td colspan="1" rowspan="1">Limitedmodalitydiversity</td></tr><tr><td colspan="1" rowspan="1">Sreelekshmi etal. (2024).</td><td colspan="1" rowspan="1">SwinCNN</td><td colspan="1" rowspan="1">BACH,BreakHis, IDC</td><td colspan="1" rowspan="1">~98%</td><td colspan="1" rowspan="1">Transformer+ CNNhybrid</td><td colspan="1" rowspan="1">Highcomplexity</td></tr><tr><td colspan="1" rowspan="1">Chakravarthyet al. (2024).</td><td colspan="1" rowspan="1">CNN featurefusion</td><td colspan="1" rowspan="1">MIAS, CBIS-DDSM,INbreast</td><td colspan="1" rowspan="1">~98%</td><td colspan="1" rowspan="1">EnsembleCNNs</td><td colspan="1" rowspan="1">Computationalcost</td></tr><tr><td colspan="1" rowspan="1">Elzaghmouri etal. (2025).</td><td colspan="1" rowspan="1">Transfer learningCNNs</td><td colspan="1" rowspan="1">Multipledatasets</td><td colspan="1" rowspan="1">~97-98%</td><td colspan="1" rowspan="1">ComparativeCNN study</td><td colspan="1" rowspan="1">Limitedinnovationbeyondbaseline</td></tr><tr><td colspan="1" rowspan="1">Brahmareddyet al. (2025).</td><td colspan="1" rowspan="1">CNN-Transformer</td><td colspan="1" rowspan="1">Mammogramdataset</td><td colspan="1" rowspan="1">95%+</td><td colspan="1" rowspan="1">Multi-tasklearning</td><td colspan="1" rowspan="1">Needs real-worldvalidation</td></tr><tr><td colspan="1" rowspan="1">Gül (2025)</td><td colspan="1" rowspan="1">LBP + CNN</td><td colspan="1" rowspan="1">Histopathology</td><td colspan="1" rowspan="1">~98%</td><td colspan="1" rowspan="1">Texture-basedenhancement</td><td colspan="1" rowspan="1">Limitedscalability</td></tr><tr><td colspan="1" rowspan="1">Alzahrani etal. (2025).</td><td colspan="1" rowspan="1">CNN + PSO</td><td colspan="1" rowspan="1">Thermography</td><td colspan="1" rowspan="1">98.80%</td><td colspan="1" rowspan="1">OptimizedCNN tuning</td><td colspan="1" rowspan="1">Domain-specific</td></tr><tr><td colspan="1" rowspan="1">Simonyan etal. (2024).</td><td colspan="1" rowspan="1">CNN baseline</td><td colspan="1" rowspan="1">Histopathology</td><td colspan="1" rowspan="1">~9092%</td><td colspan="1" rowspan="1">BaselineCNNevaluation</td><td colspan="1" rowspan="1">Loweraccuracy vshybrids</td></tr><tr><td colspan="1" rowspan="1">Fontes et al.(2025).</td><td colspan="1" rowspan="1">3D CNN</td><td colspan="1" rowspan="1">MRI dataset</td><td colspan="1" rowspan="1">AUC0.961</td><td colspan="1" rowspan="1">Volumetricanalysis</td><td colspan="1" rowspan="1">Limiteddataset size</td></tr><tr><td colspan="1" rowspan="1">Bohra et al.(2026).</td><td colspan="1" rowspan="1">Wavelet-CNN</td><td colspan="1" rowspan="1">BreakHis</td><td colspan="1" rowspan="1">99.34%</td><td colspan="1" rowspan="1">Multi-scalefeatures</td><td colspan="1" rowspan="1">Highcomputationalcost</td></tr><tr><td colspan="1" rowspan="1">Nayak(2024)</td><td colspan="1" rowspan="1">RDTNet(Transformer)</td><td colspan="1" rowspan="1">Histopathology</td><td colspan="1" rowspan="1">~99%</td><td colspan="1" rowspan="1">Deformableattentiontransformer</td><td colspan="1" rowspan="1">Highcomplexity</td></tr><tr><td colspan="1" rowspan="1">Sreelekshmi etal. (2024).</td><td colspan="1" rowspan="1">SwinCNN</td><td colspan="1" rowspan="1">BACH,BreakHis</td><td colspan="1" rowspan="1">~98%</td><td colspan="1" rowspan="1">SwinTransformer+ CNN</td><td colspan="1" rowspan="1">Resourceheavy</td></tr><tr><td colspan="1" rowspan="1">Abimouloud etal. (2024).</td><td colspan="1" rowspan="1">ViT-CNNhybrid</td><td colspan="1" rowspan="1">BreakHis</td><td colspan="1" rowspan="1">~98%</td><td colspan="1" rowspan="1">Hybrid ViT-CNN</td><td colspan="1" rowspan="1">Data intensive</td></tr><tr><td colspan="1" rowspan="1">Feng et al.(2024).</td><td colspan="1" rowspan="1">HTBE-Net</td><td colspan="1" rowspan="1">Ultrasound</td><td colspan="1" rowspan="1">High</td><td colspan="1" rowspan="1">Segmentationmodel</td><td colspan="1" rowspan="1">Limitedgeneralization</td></tr><tr><td colspan="1" rowspan="1">Goceri(2025)</td><td colspan="1" rowspan="1">CNN            +Transformer</td><td colspan="1" rowspan="1">Whole-slideimages</td><td colspan="1" rowspan="1">~97-98%</td><td colspan="1" rowspan="1">Stainnormalization十     hybridmodel</td><td colspan="1" rowspan="1">Complexity</td></tr><tr><td colspan="1" rowspan="1">Aldawsari etal. (2026).</td><td colspan="1" rowspan="1">Swin + FusionNet</td><td colspan="1" rowspan="1">Mammogram</td><td colspan="1" rowspan="1">~98-99%</td><td colspan="1" rowspan="1">Multi-scalefusion</td><td colspan="1" rowspan="1">High computecost</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">Vanitha et al.(2024).</td><td colspan="1" rowspan="1">Attention Fusion</td><td colspan="1" rowspan="1">Histopathology</td><td colspan="1" rowspan="1">~97-98%</td><td colspan="1" rowspan="1">Externalattentiontransformer</td><td colspan="1" rowspan="1">Datasetdependency</td></tr><tr><td colspan="1" rowspan="1">Katayama etal. (2024).</td><td colspan="1" rowspan="1">CNN   → ViTreview</td><td colspan="1" rowspan="1">Pathology</td><td colspan="1" rowspan="1">N/A</td><td colspan="1" rowspan="1">Review of AItransition</td><td colspan="1" rowspan="1">Theoretical</td></tr><tr><td colspan="1" rowspan="1">Acikgoz   etal. (2026).</td><td colspan="1" rowspan="1">BreasTransNeXt</td><td colspan="1" rowspan="1">Mammogram</td><td colspan="1" rowspan="1">~97%</td><td colspan="1" rowspan="1">Hybrid attentionsystem</td><td colspan="1" rowspan="1">Needsdeploymenttesting</td></tr><tr><td colspan="1" rowspan="1">Abd Elaziz etal. (2024).</td><td colspan="1" rowspan="1">CrossViT       十optimizer</td><td colspan="1" rowspan="1">Multi-dataset</td><td colspan="1" rowspan="1">High</td><td colspan="1" rowspan="1">Featureselection    十ViT</td><td colspan="1" rowspan="1">Complexity</td></tr><tr><td colspan="1" rowspan="1">Hussain et al.(2024)</td><td colspan="1" rowspan="1">MultimodalCNN + ViT</td><td colspan="1" rowspan="1">Imaging + text</td><td colspan="1" rowspan="1">~95%</td><td colspan="1" rowspan="1">Multimodalfusion</td><td colspan="1" rowspan="1">Integrationchallenges</td></tr></table>

## 3. Research methodology

The research methodology (see Figure 1) describes the hardware and software environments, the datasets used in the study, image preprocessing, the CNN-integrated CCT model, and the training process.

![](images/abbc1173e1cd7f06c07b8422eb65e027cb82ea2b17295d9a0e2a4c17971fc409.jpg)  
Figure 1: Research methodology adopted in this research

## 3.1 Dataset description

To ensure performance consistency and generalizability, the CNN-Integrated CCT model was trained and tested on both balanced and imbalanced datasets with varying numbers of breast cancer modalities (See Figure 2). Furthermore, it was essential to analyze how the model performs in detecting and classifying unseen breast cancer images after training on the training data. These strategies will provide a better understanding of the CNN-Integrated CCT model construction strategy, especially across classes that are not evenly distributed.

Dataset A consists of 7,632 mammogram images categorized into two classes: 2,520 benign and 5,112 malignant (Huang & Lin, 2020). The mammography images were originally collected from the Breast Center at the Centro Hospitalar de S. Joao (CHSJ) in Porto. Dataset B consists of 3 classes: Malign, Normal, and Benign. The link to the dataset is https://www.kaggle.com/datasets/emiliovenegas1/mammography-dataset-from-inbreast-miasand-ddsm

Dataset C is MammoNet32k, a comprehensive, standardized mammography dataset comprising 32,191 high-quality mammographic images from 7,079 unique patients across 7 major public datasets. All images converted to 16-bit grayscale PNG at 1024×1024 resolution. The dataset combines data from the US, Europe, China, and the Middle East. Five classes were included: CBIS, DSM-5, DMID, Inbreast, and Kau BCDM. The source of the dataset is https://www.kaggle.com/datasets/theosmithdevey/mammonet20k.

In this study, 70% of the images were used to train the CNN-Integrated CCT, 20% for validation, and 10% for testing on unseen images. To prevent data leakage, the photos were split into train, validation, and test directories. As the authors [53] suggested, the CNN - Integrated CCT model verification stage should include evaluating the final model on a dataset unseen during training. The study also indicated that ‘test set’ and ‘validation set’ should not be considered the same.

![](images/cbc2e5c632335508ac9e45071aca1536733dbd85e86fb79d240637a8ca399bfa.jpg)

![](images/c30403bbfbf87d43958d3cdfebd71faae274c450f1381dbfa6b2c452a5b36fa0.jpg)

![](images/f9ab197bb9638b5fcc1ca078fe5f4e7ec86a5db0706c46e2d83521c55d0b267e.jpg)  
(a) Binary Classification (Benign vs. Malignant)

![](images/694d57832acad635bc63f01135f899cb52d4628325d5360409df29f248482a5e.jpg)

![](images/63362121984bb6e07bdb6ae37a54e9e1ac6a5e6a5fb1e6ca64c1645173cbf98a.jpg)

![](images/31acb1a655141ae5585e7f676b7b751da5ecfd2224a4cce07fce80753f816792.jpg)

![](images/0392a4a480ec275b3bf9db053af94c328a8ff7032baa0e9ef5373b7557826d89.jpg)  
(b) Three-Class Classification (Benign vs. Malignant vs. Normal)

![](images/84e4f9de9ff5e35f10fa863644880e42895d239cb521760bf77b66dee4629769.jpg)

![](images/731ec5e699ea4bb27c8da8f1c4023021b67a7e366cf27beb08b6dcc14fcce5b0.jpg)

![](images/6da58a917bbdb9f01ebc03b57fe913336c040801d11f7aa27985162ab021dcfa.jpg)

![](images/2134c22465de7528909ea1c71862116d169009b602ddede4f4ec90c7f53beade.jpg)  
(c) Five-Class Classification (Cbis Ddsm, Dmid, Inbreast, Kau Bemd, Mini Mias)  
Figure 2: Data distribution in 3 datasets

![](images/3ab1cb0670956db1bc3b3de4ab236cee8e8ab806cf87ad244b88318ae9087ff0.jpg)

## 3.2 Hardware Specification

The experiments were conducted via the Precision 7680 Workstation. The workstation is a 13th-generation Intel® Core™ i9-13950HX vPro with Windows 11 Pro, an NVIDIA® RTX™ 3500 Ada Generation GPU, 32 GB DDR5 RAM, and a 1 TB SSD. Python (version 3.9) was

chosen as the programming language because it supports TensorFlow-GPU, SHAP, and LIME generation.

## 3.3 Image preprocessing

To enhance the visibility of imaging features and reduce noise, a multistep preprocessing pipeline was applied. Contrast Limited Adaptive Histogram Equalization (CLAHE) was employed to improve local contrast in dense tissue regions, followed by Gaussian blurring to suppress high-frequency noise, defined as:

$$
G ( x , y ) = 1 2 \pi \sigma 2 \exp { - x 2 + y 2 2 \sigma 2 \left( 1 \right) }
$$

Where σ represents the standard deviation of the Gaussian distribution, subsequently, bilateral filtering was applied to preserve edge structures critical for mass detection, expressed as:

$$
I f i l t e r e d ( x ) = 1 W p \sum x i \in \Omega I ( x i ) \cdot f r ( | I ( x i ) - I ( x ) | ) \cdot f s ( | x i - x | ) ( 2 )
$$

where fr denotes the range kernel, fs the spatial kernel, and $\mathrm { W p }$ the normalization factor. Nonlocal means denoising was used to enhance smoothness further while preserving important features. Finally, unsharp masking was performed to improve edge clarity, formulated as:

$$
I s h a r p = I o r i g i n a l + \alpha \cdot ( I o r i g i n a l - G b l u r )
$$

with α =1.5 controlling sharpness. (3)

## 3.4 Image augmentation

In this study, we employed extensive data augmentation techniques, including rotation, flipping, and contrast modulation, to generate a richer and more diverse dataset for Mammography-based breast cancer classification (see Figure 3). Image augmentation has proven to be a highly effective and practical approach for enhancing the robustness of models with limited ground-truth datasets [54]. In this study, we applied rotations, scaling, flipping, contrast adjustments, and noise addition on the training datasets. In the context of breast cancer detection, these techniques enable models to better account for variability in cancer characteristics, including size, shape, and location. For example, rotation (±20°) allows the model to handle cancers at different orientations, while scaling $( \pm 1 0 \% )$ enables it to handle variations in cancer size. Additionally, random flipping helps the model generalize across different image orientations, ensuring it can identify cancers in both left- and right-hemisphere scans. Contrast adjustment (±15%) enables the model to adapt to differences in image quality, while Gaussian noise (variance of 0.05) improves robustness against imaging artifacts often found in medical scans.

![](images/f925ebecd06cc4b89ec1fb304e0fa1ca68dd800feb8d39aee555fe760898d958.jpg)  
Figure 3. Samples of augmentation

## 3.5 Description of proposed CNN-Integrated CCT Model

The proposed model combines convolutional feature extraction with transformer-based global attention. The CNN-integrated CCT model can be expressed as a sequential transformation: the mammogram image is first encoded by a convolutional feature extractor, then transformed into token representations, followed by global contextual modeling with a transformer encoder, and finally mapped to diagnostic classes via a classification head. The CNN block extracts local breast tissue patterns. The CCT block converts these patterns into tokens and learns global relationships between mammographic regions. The complete model flow is:

����� → ���������������� → ������������������� → ������������ → ������������������� → ������������������ → �������������������� → �����������������

A more complete expression is:

$$
\hat { \Psi } = S o f t m a x \left( D e n s e \left( G A P \left( T r a n s f o r m e r \left( T o k e n i z e r \left( C N N \big ( A ( X ) \big ) \right) + P \right) \right) \right) \right)
$$

where X is the input mammogram image. $A ( X ) .$ is the augmented image. ���(. )extracts local feature maps. ���������(. )converts the feature maps into token embeddings. �����������(. )learns global contextual relationships. �������(. )produces the final diagnostic probability.

## Input Layer

The input mammogram image can be represented as $X \in R ^ { ( H \times W \times 3 ) }$ . Here, H is the image height, W is the image width, and 3 represents the RGB channels. Although mammograms are usually grayscale, they are converted to RGB format to match the model's input requirements. This provides a uniform input format for training.

## Data Augmentation Layer

This layer is important for mammography classification because breast position, compression, and acquisition angle may vary across patients. The augmentation process can be expressed as $X _ { a u g } = A ( X )$ In the implementation, $X ^ { \prime } = X / 2 5 5$ and $X _ { a u g } =$ ��������������������(�′). Here $X ^ { \prime }$ is the rescaled image and $X _ { a u g }$ is the augmented image. Rescaling pixel values from 0–255 to 0–1. This improves numerical stability during training.

## CNN Feature Extraction Block

After data augmentation, the mammogram is passed through the CNN feature extractor. This CNN block is placed before the CCT tokenizer. It extracts local and fine-grained features from the mammogram. The CNN block follows this structure:

$$
\begin{array} { c c } { { C o n v 2 D ( 4 8 )  M a x P o o l i n g  C o n v 2 D ( 1 2 8 )  M a x P o o l i n g  C o n v 2 D ( 1 9 2 ) } } \\ { { } } \\ { {  C o n v 2 D ( 1 9 2 )  C o n v 2 D ( 1 2 8 )  M a x P o o l i n g } } \end{array}
$$

The CNN feature extraction process can be written as $F _ { C \mathrm { N N } } = C N N \left( X _ { a u g } \right)$ . Here $F _ { C N N }$ is the final convolutional feature map. A convolutional layer can be expressed as:

$$
F _ { l } = R e L U \big ( W _ { l } * F _ { ( l - 1 ) } + b _ { l } \big )
$$

where $F _ { l }$ is the output feature map of layer l. $W _ { l }$ is the convolution kernel. $b _ { l }$ is the bias term.   
The symbol \* represents convolution. ReLU is the activation function.

The first convolutional layer uses 48 filters. It captures low-level features such as edges, curves, small texture changes, and intensity variations. The second convolutional layer uses 128 filters. It learns more detailed tissue patterns, including density variations and abnormal local textures. The next two convolutional layers use 192 filters. These layers learn deeper and more complex features. They help detect irregular lesion margins, mass boundaries, and structural distortion. The final convolutional layer uses 128 filters. It refines the feature representation before passing it to the CCT tokenizer. Max pooling is applied after selected convolutional layers. It can be written as:

$$
F _ { p o o l } = M a x P o o l ( F _ { l } )
$$

Max pooling reduces spatial size while preserving the strongest responses. It also reduces computational cost. This is useful in mammography because suspicious patterns can appear in slightly different positions. The CNN block is crucial for breast cancer screening mammography. Many cancer-related signs are small and local. These include microcalcification-like bright spots, tissue density changes, mass edges, spiculated margins, and local architectural distortion. The CNN block helps the model capture these subtle visual features before transformer processing.

## CCT Tokenizer

After CNN feature extraction, the feature map is passed into the CCT tokenizer. The tokenizer converts the CNN feature map into a sequence of tokens. The tokenizer operation can be written as:

$$
F _ { t o k } = P o o l \left( R e L U \big ( C o n v ( F _ { C N N } ) \big ) \right)
$$

where $F _ { t o k }$ is the tokenized convolutional feature map. In the model, the tokenizer uses convolutional layers followed by max-pooling. This is different from a standard Vision Transformer, which directly divides an image into fixed patches. The convolutional tokenizer is better suited to mammography because it preserves local spatial information before tokenization. This is important because mammographic abnormalities are often small and subtle. If the image is divided into large fixed patches too early, fine details may be weakened. The convolutional tokenizer helps retain local lesion-related features.

## Projection to Token Embedding Space

The tokenizer's output is projected into a fixed embedding dimension using a dense layer. This step prepares the features for transformer processing. The projection can be written as:

$$
E = D e n s e ( F _ { t } o k )
$$

where E is the projected feature representation, then the projected feature map is reshaped into a sequence:

$T = R e s h a p e ( E )$ or, $T = [ t _ { 1 } , t _ { 2 } , t _ { 3 } , \dots , t _ { N } ]$ where T is the token sequence. $t _ { i }$ is one token. N is the total number of tokens. Each token represents a local mammographic region after CNN and tokenizer-based feature extraction.

## Positional Embedding

The transformer does not naturally understand token positions. Therefore, positional embedding is added to the token sequence. This can be written as:

$Z _ { 0 } = T + P$ where $Z _ { 0 }$ is the position-aware token sequence. T is the token embedding. P is the learnable positional embedding. For each token:

$z _ { i } = t _ { i } + p _ { i } { \mathrm { w h e r e } } t _ { i }$ is the token embedding and $p _ { i }$ is its positional embedding.

This step is important in mammography because the location of abnormal tissue matters. Breast cancer diagnosis depends not only on texture but also on lesion position, tissue distortion, and relationships among surrounding regions.

## Transformer Encoder Block

The position-aware tokens are passed through transformer encoder layers. Each transformer block contains layer normalization, multi-head self-attention, residual connection, stochastic depth, and an MLP block.

The input to a transformer layer can be written as:

$Z _ { l } = T r a n s f o r m e r L a y e r \bigl ( Z _ { ( l - 1 ) } \bigr )$ where $Z _ { l }$ is the output of the transformer layer l.

Layer normalization is first applied:

$Z _ { n } o r m = L a y e r N o r m \bigl ( Z _ { ( { l - 1 } ) } \bigr ) \mathrm { L a y e r }$ normalization stabilizes training and improves convergence.

## Multi-Head Self-Attention

The transformer uses multi-head self-attention to learn relationships between different mammographic regions. The attention mechanism is written as:

��������� $( Q , K , V ) = S o f t m a x \big ( Q K ^ { T } / \sqrt { d _ { k } } \big )$ �where Q is the query matrix, and K is the key matrix. V is the value matrix. $d _ { k }$ is the key dimension. Multi-head self-attention can be written as:

����(�) = ������������������(�, �)This allows each token to attend to all other tokens.   
It helps the model learn global relationships across the mammogram.

This is useful because malignant signs may not be completely local. A suspicious region may need to be interpreted together with surrounding tissue density, asymmetry, and architectural distortion. Self-attention allows the model to capture these long-range dependencies.

## Residual Connection and Stochastic Depth

After self-attention, the output is added back to the input using a residual connection. Stochastic depth is applied for regularization. This can be written as:

$$
Z _ { a } t t = Z _ { ( l - 1 ) } + S D \left( M H S A \left( L a y e r N o r m { \left( Z _ { ( l - 1 ) } \right) } \right) \right)
$$

where SD(.) represents stochastic depth. Stochastic depth randomly drops residual branches during training. It can be written as:

$$
S D ( x ) = x / p , w i t h p r o b a b i l i t y p
$$

$$
S D ( x ) = 0 , w i t h p r o b a b i l i t y 1 - p
$$

where p is the survival probability. In the model, the stochastic depth rate increases gradually across transformer layers. This improves regularization and reduces overfitting. It is useful for mammography datasets because medical image datasets are often limited in size.

## MLP Block with GELU Activation

After attention, the model applies another layer normalization followed by an MLP block. This layer

can be written as $Z _ { m } l p = M L P \bigl ( L a y e r N o r m ( Z _ { a } t t ) \bigr )$ . The MLP operation is:

$$
M L P ( x ) = D e n s e { \Big ( } G E L U { \big ( } D e n s e ( x ) { \big ) } { \Big ) } .
$$

In the model, the MLP hidden units are:

$$
[ 2 \times p r o j e c t i o n _ { d } i m , p r o j e c t i o n _ { d } i m ]
$$

The final output of the transformer block is:

$$
Z _ { l } = Z _ { a } t t + S D \left( M L P \Big ( L a y e r N o r m ( Z _ { a } t t ) \Big ) \right)
$$

The MLP block learns nonlinear relationships among token features. This helps the model distinguish between benign and malignant tissue patterns that may appear visually similar.

## Final Layer Normalization

After all transformer layers, layer normalization is applied again. The layer normalization is expressed as $Z _ { f } i n a l = L a y e r N o r m ( Z _ { L } )$ . here $Z _ { L }$ is the output of the last transformer layer. This produces a stable final token representation before pooling.

## Global Average Pooling

The model applies global average pooling over all tokens. Global average pooling allows all token features to contribute to the final prediction. The model does not use a separate class token. This makes the architecture simpler and compact. If g is the image-level feature vector, �is the number of tokens, $z _ { i }$ is the representation of a token �, $g = G A P { \left( Z _ { f } i n a l \right) }$ or $g =$ $( 1 / N ) \Sigma z _ { i }$

## Softmax Classification Layer

The pooled feature vector is passed into a dense classification layer. The final prediction is produced using softmax activation. The layer can be written as $\hat { \mathbf { y } } = S o f t m a x ( W _ { c } g +$ $b _ { c } )$ where $\hat { \mathbf { y } }$ is the predicted probability distribution. $W _ { c }$ is the classifier weight matrix. $b _ { c }$ is the classifier bias. The predicted $C l a s s = a r g m a x ( { \hat { y } } )$ . For binary classification, the classes may be benign and malignant. For multi-class classification, the model predicts the probability of each diagnostic category.

## Loss Function

The model is trained using categorical cross-entropy loss. This loss function measures the difference between the true label and the predicted probability distribution. The loss function can be expressed as $L = - \it { \Sigma y } _ { i } l o g ( \hat { y } _ { i } )$ . here $y _ { i } \mathrm { i s }$ the true class label and ŷ<sub>�</sub>is the predicted probability for the class �.

## Optimizer

AdamW, with adaptive learning and weight decay regularization, was used as the optimizer in the model. The update rule can be written as:

$\theta _ { ( t + 1 ) } = \theta _ { t } - \eta m _ { t } / \bigl ( \sqrt { v _ { t } } + \varepsilon \bigr ) - \lambda \theta _ { t }$ where $\theta _ { t }$ represents model parameters. η is the learning rate. $m _ { t }$ is the first moment estimate. $v _ { t }$ This is the second moment estimate. λ is the weight decay coefficient. Weight decay helps reduce overfitting. Gradient clipping is also used to improve training stability.

![](images/1648ee41b28743bd6cd077a7d7d0587c33cd72165c40908d6a87cf3304f2af91.jpg)  
Figure 4: Layer-wise visualization of the CNN-integrated CCT

e algorithm of the proposed Input: Image dataset X, Labels Y Output: Final predicted labels Ŷ and evaluation metrics 1: Initialize dataset X and Y 2: Set image size = 64×64×3, number of classes = C 3: Split the dataset into training (90%) and testing (10%) 4: Apply Stratified K-Fold Cross Validation (K = 5) 5: for each fold k = 1 to K do 6: Initialize CNN-CCT model 7: 8: Apply data augmentation (rescaling, flipping) 9: CNN Feature Extraction: 10: Conv2D(48) → ReLU → MaxPool(2×2) 11: Conv2D(128) → ReLU → MaxPool(2×2) 12: Conv2D(192) → ReLU 13: Conv2D(192) → ReLU 14: Conv2D(128) → ReLU → MaxPool(2×2) 15: CCT Tokenization: 16: Extract convolutional tokens 17: Project features to d-dimensional embeddings 18: Reshape feature maps into a token sequence T 19: Positional Encoding: 20: T ← T + P 21: Transformer Encoder (L = 2 layers): 22: Apply Layer Normalization 23: Multi-Head Self Attention 24: Apply Stochastic Depth 25: Residual Connection 26: Apply Layer Normalization 27: MLP Block 28: Apply Stochastic Depth 29: Residual Connection 30: Global Representation: 31: Apply Global Average Pooling 32: Classification: 33: Dense layer + Softmax activation 34: Train model using AdamW optimizer 35: Optimize using categorical cross-entropy loss 36: Apply EarlyStopping and ReduceLROnPlateau 37: Evaluate fold performance: 38: Compute accuracy, loss, ROC curve, Generate confusion matrix, Generate classification report 41: Store fold-wise metrics: 42: Accuracy curve, loss curve, Validation ROC, validation metrics 44: end for 45: Aggregate all folds: 46: Compute average classification report, averaged confusion matrix, mean ROC curve, average validation accuracy curve, validation loss curve 51: Test Phase Evaluation: 52: Evaluate model on unseen test set, Compute test accuracy and loss, test classification report, test confusion matrix 56: Generate test ROC curve, Compute test accuracy and loss curves 58: Explainable AI Analysis: 59: Apply Grad-CAM for spatial activation mapping 60: Extract attention maps from the transformer encoder 61: Apply LIME for local explanations 62: Apply SHAP for global and local feature attribution 63: Return final predictions, fold-wise metrics, test metrics, and XAI visualizations END

## 3.6 Training Process of CNN-Integrated CCT Model

The training process for the proposed CNN-Integrated CCT begins with the mammography dataset, which is organized into class-wise directories. Each image is assigned a categorical label based on its folder index, where the label belongs to a finite set of diagnostic classes defined as: $y _ { i } \in { 0 , 1 , \ldots , C - 1 }$

Here, C represents the total number of breast cancer classes in the dataset. The images were converted to RGB format and resized to a fixed resolution to ensure uniform input representation across the model. This preprocessing step can be expressed as $I \in R ^ { ( 6 4 \times 6 4 \times 3 ) }$

To ensure numerical stability during training and reduce variability across different imaging devices, pixel intensities are normalized to the range [0, 1]. This normalization is defined as: $I ^ { \prime } = I / 2 5 5$

After preprocessing, the dataset is split into training, validation, and testing subsets. A stratified sampling strategy is applied to preserve the original class distribution across all subsets. This ensures that the probability distribution of classes remains approximately consistent, which can be expressed as: $P _ { t } r a i n ( y ) \approx P _ { t } e s t ( y )$

To improve the robustness and generalization ability of the model, K-fold cross-validation is applied on the training data. In this setting, the dataset is divided into K equal folds, typically $\mathrm { K } = 5$ . For each iteration, the model is trained on $\mathrm { K } - 1$ folds and validated on the remaining fold, which is mathematically represented as:

$$
D _ { t } r a i n ^ { ( k ) } \cup D _ { v } a l ^ { ( k ) } = D _ { t } r a i n
$$

This strategy ensures stable performance estimates, reduces bias due to data imbalance, and improves generalization to unseen samples.

To further enhance generalization and reduce overfitting, data augmentation is applied in real time during training. Each input image undergoes stochastic transformations such as horizontal flipping and intensity rescaling. This augmentation process is defined as:

$$
I _ { a } u g = A ( I )
$$

where A(·) denotes the augmentation function that simulates variations in breast positioning, imaging angle, and scanner noise, thereby improving model robustness to real-world variations.

The training pipeline follows a hierarchical learning strategy beginning with convolutional feature extraction. In this stage, the augmented input image is passed through a convolutional network to extract local visual patterns such as edges, textures, and micro-calcifications. This process is defined as:

$$
F _ { C } N N = f _ { c } o n v ( I _ { a } u g )
$$

The extracted convolutional feature maps are then converted into a sequential representation using the CCT tokenizer. This transformation flattens and projects spatial features into a token sequence suitable for transformer processing. It is expressed as:

$$
T = F l a t t e n { \big ( } W _ { p } F _ { C } N N { \big ) }
$$

where $W _ { p }$ is a learnable projection matrix that maps convolutional features into token embeddings. The resulting token sequence is then enhanced with positional information and passed through a transformer encoder, which learns global dependencies across breast regions. This stage is represented as:

$$
Z = T r a n s f o r m e r ( T + P )
$$

where P denotes positional embeddings that preserve spatial awareness, enabling the model to capture anatomical structure, symmetry between breasts, and long-range dependencies across tissue regions.

After global feature modeling, the transformer output is aggregated using global average pooling, and the final classification is performed using a softmax classifier. The prediction function is defined as:

$$
\hat { \mathbf { y } } = S o f t m a x \bigl ( W \cdot G A P ( Z ) \bigr )
$$

The training process is optimized with the AdamW optimizer, which updates model parameters via adaptive moment estimation with weight decay. The update rule is given by:

$$
\theta _ { t + 1 } = \theta _ { t } - \eta \cdot \left( m _ { t } / \big ( \sqrt { v _ { t } + \varepsilon } \big ) \right) - \lambda \theta _ { t }
$$

where η is the learning rate, λ is the weight decay coefficient, and m\_t, v\_t are the first and second moment estimates, respectively. The model is trained using categorical cross-entropy loss, which measures the divergence between predicted probabilities and true class labels. The loss function is defined as:

$$
\begin{array} { r } { L = - \sum _ { i = 1 } ^ { C } y _ { i } l o g ( \hat { \bf y } _ { i } ) } \end{array}
$$

During training, performance is continuously monitored on the validation set, and the model achieving the lowest validation loss is selected as the best checkpoint. This selection criterion is defined as:

$$
B e s t M o d e l = a r g m i n ( L _ { v } a l )
$$

This complete training strategy ensures that the model learns robust, generalizable, and clinically meaningful representations for mammography classification by effectively combining local convolutional feature extraction with global transformer-based reasoning. Final Training Hyperparameters

Table 2: Hyperparameters of the CNN-integrated CCT model
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Image Size</td><td rowspan=1 colspan=1>64×64× 3</td></tr><tr><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>250</td></tr><tr><td rowspan=1 colspan=1>Batch Size</td><td rowspan=1 colspan=1>32</td></tr><tr><td rowspan=1 colspan=1>Learning Rate</td><td rowspan=1 colspan=1>1 × 10−-4</td></tr><tr><td rowspan=1 colspan=1>Weight Decay</td><td rowspan=1 colspan=1>1 × 10−4</td></tr><tr><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>AdamW (fallback: Adam)</td></tr><tr><td rowspan=1 colspan=1>Loss Function</td><td rowspan=1 colspan=1>Categorical Cross-Entropy</td></tr><tr><td rowspan=1 colspan=1>K-Folds</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>CNN Conv Layers</td><td rowspan=1 colspan=1>2 (base configuration)</td></tr><tr><td rowspan=1 colspan=1>Transformer Layers</td><td rowspan=1 colspan=1>2</td></tr><tr><td rowspan=1 colspan=1>Projection Dimension</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>Attention Heads</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Stochastic Depth Rate</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>Positional Embedding</td><td rowspan=1 colspan=1>Enabled</td></tr></table>

The proposed CCT model is effective for mammography-based breast cancer classification because it combines three useful properties:

First, the convolutional tokenizer captures local mammographic features. These local patterns are important because suspicious findings may appear as small bright microcalcification clusters, irregular mass margins, or localized density changes. The convolution operation learns these local patterns through small receptive fields.

Second, the transformer encoder captures global contextual relationships. This is important because mammography interpretation often depends on the relationship between a suspicious region and the surrounding breast structure. For example, architectural distortion is a global pattern of tissue deformation rather than only a single local texture. Self-attention allows each token to compare itself with all other regions. $z _ { i }  \{ z _ { 1 } , z _ { 2 } , \dots , z _ { N } \}$ . Therefore, the model learns both.����� ������ − ����� ����������� ������ ������ − ����� �������

Third, stochastic depth and dropout improve generalization. This is important in medical imaging because annotated datasets are often limited, and class imbalance is common. Regularization helps reduce overfitting and improves robustness when the model is tested on unseen mammograms.

The proposed model employs a Compact Convolutional Transformer architecture for breast cancer mammography classification. Given an input mammogram, data augmentation is first applied to improve robustness against image-level variations. The augmented image is then processed by a convolutional tokenizer consisting of repeated convolution, ReLU activation, and max-pooling operations. This tokenizer extracts local mammographic features and converts the image into a compact sequence of visual tokens. Compared with direct patch tokenization, convolutional tokenization introduces an inductive bias that preserves local spatial information, which is important for detecting subtle mammographic structures such as microcalcifications, irregular lesion boundaries, and local density variations.

The resulting token sequence is projected into a fixed-dimensional embedding space and combined with learnable positional embeddings. The encoded sequence is then processed using multiple transformer encoder blocks. Each encoder block contains layer normalization, multihead self-attention, residual connections, stochastic depth, and an MLP module with GELU activation. The self-attention mechanism models long-range dependencies among mammographic regions, allowing the network to learn both local lesion-specific patterns and global breast-structure relationships. After the final transformer layer, global average pooling aggregates the token representations into a single image-level descriptor. Finally, a dense softmax classifier estimates the probability distribution over the target breast cancer classes.

## 4. Result of the proposed model

The performance of the models is evaluated using the following metrics:

$$
A c c u r a c y = ( ( T P + T N ) ) / ( ( T P + F P + F N + T N ) )
$$

$$
P r e c i s i o n = T P / ( T P + F P )
$$

$$
R e c a l l = T P / ( T P + F N )
$$

$$
F 1 s c o r e = 2 \times ( P r e c i s i o n \times R e c a l l ) / ( P r e c i s i o n + R e c a l l )
$$

$$
S p e c i f i c i t y = T N / ( T N + F P )
$$

Furthermore, the accuracy curve, data loss curve, confusion matrix, and XAI were used to visualize the models' performance.

## 4.1 Performance of 5-folds of the CNN-Integrated CCT Model

![](images/f4722d52947e9e763e9cd2080fc08a1cf442b5ca5ab335193b143aa01b9ef00f.jpg)  
The CNN-CCT model consistently achieves high accuracy across all five folds. For 2-class classification, accuracy ranges between 99%-100%, for 3-class classification between 99%-100%, and for 5-class classification between 99%-99%, demonstrating strong stability and generalization across different classification scenarios.

Figure 5: 5 Fold Cross-Validation Performance of CNN-integrated CTT Model

Table 3: Classification report of the 5-fold of the model
<table><tr><td colspan="1" rowspan="1">FPd</td><td colspan="1" rowspan="1">Casname</td><td colspan="1" rowspan="1">Precion</td><td colspan="1" rowspan="1">Real</td><td colspan="1" rowspan="1">F1-ccre</td><td colspan="1" rowspan="1">Supprt</td><td colspan="1" rowspan="1">CIassname</td><td colspan="1" rowspan="1">Precion</td><td colspan="1" rowspan="1">Recal</td><td colspan="1" rowspan="1">F1-c0re</td><td colspan="1" rowspan="1">Support</td><td colspan="1" rowspan="1">Classname</td><td colspan="1" rowspan="1">Preccion</td><td colspan="1" rowspan="1">Recal</td><td colspan="1" rowspan="1">FI1-ccre</td><td colspan="1" rowspan="1">Suppport</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">454</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1956</td><td colspan="1" rowspan="1">CBIS DSM</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">514</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">920</td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">2468</td><td colspan="1" rowspan="1">Dmid</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">183</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Normal</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">365</td><td colspan="1" rowspan="1">Inbreast</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">97%</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">148</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Kau Bcmd</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">429</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Mini Mias</td><td colspan="1" rowspan="1">96%</td><td colspan="1" rowspan="1">95%</td><td colspan="1" rowspan="1">96%</td><td colspan="1" rowspan="1">58</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4789</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4789</td><td colspan="1" rowspan="1">Macro Avg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4789</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">454</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1956</td><td colspan="1" rowspan="1">CBIS DSM</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">514</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">920</td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">2468</td><td colspan="1" rowspan="1">Dmid</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">184</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Normal</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">364</td><td colspan="1" rowspan="1">Inbreast</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">147</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Kau Bcmd</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">429</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Mini Mias</td><td colspan="1" rowspan="1">97%</td><td colspan="1" rowspan="1">97%</td><td colspan="1" rowspan="1">97%</td><td colspan="1" rowspan="1">58</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">Macro Avg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">454</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1956</td><td colspan="1" rowspan="1">CBIS DSM</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">515</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">920</td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">2468</td><td colspan="1" rowspan="1">Dmid</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">184</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Normal</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">364</td><td colspan="1" rowspan="1">Inbreast</td><td colspan="1" rowspan="1">97%</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">97%</td><td colspan="1" rowspan="1">147</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Kau Bcmd</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">97%</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">428</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Mini Mias</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">58</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">Macro Avg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1374</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">453</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1956</td><td colspan="1" rowspan="1">CBIS DSM</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">514</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">920</td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">2467</td><td colspan="1" rowspan="1">Dmid</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">184</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Normal</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">365</td><td colspan="1" rowspan="1">Inbreast</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">148</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Kau Bcmd</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">428</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Mini Mias</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">95%</td><td colspan="1" rowspan="1">96%</td><td colspan="1" rowspan="1">58</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1373</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1373</td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">Macro Avg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1373</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">453</td><td colspan="1" rowspan="1">Benign</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">1955</td><td colspan="1" rowspan="1">CBIS DSM</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">514</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">920</td><td colspan="1" rowspan="1">Malignant</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">2468</td><td colspan="1" rowspan="1">Dmid</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">184</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Normal</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">365</td><td colspan="1" rowspan="1">Inbreast</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">96%</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">148</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Kau Bcmd</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">428</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Mini Mias</td><td colspan="1" rowspan="1">93%</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">96%</td><td colspan="1" rowspan="1">58</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1373</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">Accuracy</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1373</td><td colspan="1" rowspan="1">MacroAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">Macro Avg</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">98%</td><td colspan="1" rowspan="1">1332</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1373</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">100%</td><td colspan="1" rowspan="1">4788</td><td colspan="1" rowspan="1">WeightedAvg</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">99%</td><td colspan="1" rowspan="1">1332</td></tr></table>

The CNN-Integrated CCT model achieved consistently high performance across all five folds (see Table 3). For the 2-class. 3-class and 5-class breast cancer mammography accuracy remained between 99% and 100% in all folds. Slightly lower scores were observed for smaller classes, such as Mini-MIAS, where precision dropped to 93% in Fold 5, and INbreast, where recall ranged from 96% to 99%.

![](images/fc9cc2b9f7f79ef6a76e8a2ea9bbd547d536744c99232bad78b7511d59ca1502.jpg)  
Figure 6: 5-Fold Confusion Matrix of 2-class. 3-class and 5-class Breast Mammography Classification

The confusion matrix in Figure 5 corresponds to the classification reports in Table 3. In the 2- class breast cancer classification, the model correctly classified around 468–472 benign and 918–920 malignant cases per fold, with only 3–8 total errors. This indicates the model effectively learned the features of benign and malignant cancer. In the 3-class breast cancer detection and classification task, the model correctly predicted about 2451–2460 malignant cases per fold, and the class ‘Normal’ was mostly correctly classified, around 361–364 samples. In the 5-class case, minor confusion occurs in smaller dataset groups such as CBIS-DDSM, INbreast, and KAUBCMD.

![](images/e69b596793e514ad870e485fe6665e50ca78529eaff86ea9b0cc876073983a7d.jpg)  
Figure 7: 5-Fold ROC-AUC Analysis of 2-class, 3-class, and 5-class Breast Mammography Classification

The ROC curves show strong and stable performance of the classification model across different classes (see Figure 7). In the 2-class breast cancer classification, both benign and malignant classes achieved an AUC of 1.00 across all folds, indicating excellent separation between cancerous and non-cancerous cases. The 3-class model also maintained AUC values of 1.00 for ‘Benign’, ‘Malignant’, and ‘Normal’ classes, showing that adding the normal category did not reduce classification performance. In the more challenging 5-class datasetlevel classification, the model still performed strongly, with AUC values mostly between 0.98 and 0.99, while Mini-MIAS showed the lowest but still high AUC of about 0.96. Overall, the curves remain close to the upper-left corner and far above the random baseline of 0.50, supporting the robustness and generalization ability of the lightweight CNN embedding model.

![](images/8474dfde4bc3cc7347574754e328fd96b5e82e6f1d9f3addbc0a53a5fa0c9115.jpg)  
Figure 8: 5-Fold Training and Validation Accuracy Curves of CNN-Integrated CCT for Breast Mammography Classification

The training accuracy curve rises sharply from about 0.70–0.85 at the initial epochs and reaches nearly 0.98–1.00 in most folds (see Figure 8). Validation accuracy follows a similar pattern, stabilizing around 0.97–0.99, especially for Fold 4 and Fold 5. The narrow gap between training and validation curves suggests that the model learns effectively without strong overfitting.

![](images/57df43df28c813e52f2e594922eee510841245377615fd4243a46bee89ec5052.jpg)  
Figure 9: 5-Fold Training and Validation Loss Curves of CNN-Integrated CCT

Table 9 presents the training and validation loss. The loss curves show stable convergence across all five folds, with both training and validation losses decreasing rapidly in the early epochs. Initial loss values start around 0.65–0.80 for training and gradually decline to approximately 0.10–0.15, while validation loss stabilizes at 0.13–0.17. A few temporary fluctuations in validation are visible, particularly in Fold 3, Fold 4, and Fold 5, but they settle quickly without sustained divergence.

![](images/ae305f7675f4156f38c4290c6d72620d185a32db6c2061b23e33bdcdca7fc7e1.jpg)

![](images/3f06cf112c125da83d2eb9091922a14ce0fa53fab88e04e7d63d213133a172ba.jpg)

<table><tr><td colspan="6">5-Fold Average Confusion Matrix (%)</td></tr><tr><td>cbis_ddsm</td><td>99.6%</td><td>0.0%</td><td>0.0%</td><td>0.1%</td><td>0.3%</td></tr><tr><td>dmid</td><td>0.0%</td><td>100.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td>rue inbreast</td><td>0.0%</td><td>0.0%</td><td>97.8%</td><td>2.2%</td><td>0.0%</td></tr><tr><td>kau_bcmd</td><td>0.5%</td><td>0.0%</td><td>0.6%</td><td>98.7%</td><td>0.1%</td></tr><tr><td rowspan="2">mini_mias</td><td>3.1%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>96.9%</td></tr><tr><td>cbis_ddsm dmid</td><td>Predicted</td><td>inbreast kau_bcmd mini_mias</td><td></td><td></td></tr></table>

Figure 10: 5-Fold Average Confusion Matrices for Lightweight CNN-Based Breast Mammography Classification

The averaged confusion matrices are presented in Figure 10. In the 2-class classification, the model correctly identified 99.1% of the Benign class and 99.8% of the Malignant class. For the 3-class cancer detection and classification, the correct prediction rates of 99.5% for benign, 99.5% for malignant, and 99.3% for normal. The 5-class task was more challenging but still robust, with class-wise accuracies ranging from 96.9% to 100%; the lowest value was observed for Mini-MIAS, likely due to its smaller sample size.

3 Class  
![](images/a4b8ad96b9cadf70e049641575901829be4ebe2022b9bb623d8d50826ceb5491.jpg)

![](images/4b15fdbf777da614ca3e358ee82e4cbbc7985e948a359b0f686f851f3b3b5014.jpg)

![](images/c79520e1cd1ec0d49fa3c1ba9fec16248ccfb153613d525efaa8efe12c25f5cc.jpg)

![](images/b40868819b7990d72e4a6834991964e59f1766f74eaa65e9bc7784828c56fdab.jpg)

![](images/fada54ed4bb3abaeea6b0f0ee44aebc13d1e53bcf975d3f5c0b7e120ca91858b.jpg)

![](images/cd1d93bb45f4dbe496436c6c30df9e043261a503a337a1feae94cd15575e5d35.jpg)  
Figure 11: Average Validation Accuracy and Loss Curves for 2-Class, 3-Class, and 5-Class

The proposed CNN-integrated CCT model exhibits consistent performance for 2-, 3-, and 5- class breast cancer classification, with an average validation accuracy of 0.99 (±0.01) and a low, stable validation loss (see Figure 11). In the 5-class breast cancer classification, the model shows only a slight drop in accuracy due to increased class imbalance; accuracy remains above 0.98. The learning curves indicate rapid convergence in the early epochs and very stable behavior thereafter. The important aspect of the model is that there are no signs of overfitting, as the validation loss closely tracks the accuracy trends. Even in the case of 5-class classification, the model maintains high accuracy (>98%), indicating strong feature extraction capability and robustness.

## 4.2 Performance of unseen data detection and classification

Table 4: Performance of the Proposed Model on Unseen Test Data for Multi-Class Breast Cancer Classification
<table><tr><td rowspan=1 colspan=1>ClassificationTask</td><td rowspan=1 colspan=1>Test Accuracy (Mean ±SD)</td><td rowspan=1 colspan=1>Test Loss (Mean ±SD)</td><td rowspan=1 colspan=1>95% CI(Accuracy)</td></tr><tr><td rowspan=1 colspan=1>2-Class</td><td rowspan=1 colspan=1> $0 . 9 9 5 5 \pm 0 . 0 0 2 4$ </td><td rowspan=1 colspan=1> $0 . 1 2 8 7 \pm 0 . 0 0 5 6$ </td><td rowspan=1 colspan=1>[0.993, 0.998]</td></tr><tr><td rowspan=1 colspan=1>3-Class</td><td rowspan=1 colspan=1> $0 . 9 9 4 9 \pm 0 . 0 0 0 8$ </td><td rowspan=1 colspan=1> $0 . 1 8 4 4 \pm 0 . 0 0 3 0$ </td><td rowspan=1 colspan=1>[0.994, 0.996]</td></tr><tr><td rowspan=1 colspan=1>5-Class</td><td rowspan=1 colspan=1> $0 . 9 8 7 0 \pm 0 . 0 0 1 6$ </td><td rowspan=1 colspan=1> $0 . 2 6 4 0 \pm 0 . 0 0 4 9$ </td><td rowspan=1 colspan=1>[0.985, 0.989]</td></tr></table>

The CNN-integrated CCT model performs very strongly on unseen test data, maintaining consistently high accuracy above 98% (see Table 4). The best performance is observed in the 2-class and 3-class tasks; a slight drop in performance appears in the 5-class case due to increased class complexity. The model's performance is further confirmed using a confusion matrix, ROC curve, and training and validation loss curves.

Analysis (Short):

![](images/61ddb56330adce4a3bc22240820e764bcd98641545662b788729acd0c3d31796.jpg)

![](images/4dacf4b4c309b04a7d1911408c2a150d4d6f7ec0b38ae3d52d686b1f660bee11.jpg)

![](images/5dab128ffdad233b54bea42be4d573d8b712351acbe2ef53770d23ce357e46c3.jpg)  
• 2-Class: Correct predictions = 249.8 + 510.8 = 760.6 out of 764.0 → Accuracy = 99.55%. Misclassifications = 2.2 + 1.2 = 3.4.  
• 3-Class: Correct predictions = 1079.6 + 1366.8 + 201.0 = 2647.4 out of 2661.8 → Accuracy = 99.46%. Misclassifications = 13.6.  
• 5-Class: Correct predictions = 285.0 + 103.0 + 80.8 + 231.6 + 31.0 = 731.4 out of 737.8 → Accuracy = 99.13%. Misclassifications = 6.4.  
• Most errors are between Kau Bcmd and Inbreast (4.0) in 5-class.  
• Overall, the model shows very high accuracy and very few misclassifications across all settings.

Figure 12: Comprehensive Confusion matrix of CNN-integrated CCT Performance (2-Class, 3-Class, and 5-Class Evaluation)

![](images/684d64957acdfefc01a637a0c78a3bd5b523f31fe9dbec117d14501baecd9aae.jpg)

![](images/31e198bbece05ea16a394847398b38a319548a7e72260c7147aeb262be2cd950.jpg)

![](images/e1dc1db4233fdd88d726d75d332bccdbb917e0c7f3b2d8974ee4539ad5e71ba1.jpg)  
Figure 13: Comprehensive ROC Curve Analysis for Multi-Class Classification Performance (2-Class, 3-Class, and 5-Class Evaluation)

Figure 12 presents the confusion matrices for the test datasets of 2-class, 3-class, and 5-class breast cancer classification. The ROC curves also support the results from the Confusion Matrix. The ROC curve analysis suggests that in the 2-class and 3-class classification, the model achieved an AUC of 1.0000 (see Figure 12). For the 5-class classification task, the AUC values remain exceptionally high, ranging from 0.9994 to 1.0000, indicating near-perfect classification even with increased class complexity. Across all experiments, ROC curves remain tightly aligned toward the top-left corner of the plot, indicating very high true positive rates with minimal false positives.

![](images/c1333028c9a4285e0ef4cf877561e37daa27ddd29f3e9269002c48bb7ec4d645.jpg)  
Figure 14: Comprehensive K-Fold Evaluation of Test Accuracy and Loss Across

Furthermore, the K-fold averaged test performance indicates that, in the 2-class model, test accuracy rises sharply in the initial epochs and reaches approximately 0.99 by around 25 epochs (see Figure 14). However, the test loss decreases steadily and stabilizes near 0.12, indicating fast learning and efficient optimization. For the 3-class scenario, the model shows slightly slower but consistent convergence, with accuracy stabilizing near 0.99 by approximately 30 epochs and test loss reducing to about 0.19, reflecting increased task complexity while still maintaining strong predictive performance. In the 5-class classification task, convergence remains stable, though comparatively gradual, with accuracy settling between 0.98 and 0.99 and test loss stabilizing around 0.27, indicating the expected impact of higher class complexity on model learning. Across all three experimental setups, the shaded variance bands remain narrow, confirming low inter-fold variability and robust generalization capability. Importantly, there is no clear evidence of overfitting, as both the accuracy and loss curves show synchronized, stable convergence.

## 4.3 Computational cost analysis of the CNN-integrated CCT Model

Table 5 summarizes the computational cost of the proposed CNN-integrated CCT model. The model contains only 0.2504 million parameters and requires less than 1 MB of storage, showing its lightweight design. The inference performance was also efficient. The model required approximately 1.4884 ms to process one image, with an increase in inference memory of only 0.9180 MB. The reported convolution-only computational cost was 0.00210944 GFLOPs, showing that the convolutional part of the model required very low computational effort. Since this value represents only the convolutional operations, it should be reported as conv-only GFLOPs rather than complete model GFLOPs.

Table 5. Computational efficiency profile of the proposed CNN-integrated CCT model
<table><tr><td colspan="1" rowspan="1">Metric</td><td colspan="1" rowspan="1">Value</td><td colspan="1" rowspan="1">Interpretation</td></tr><tr><td colspan="1" rowspan="1">Total parameters</td><td colspan="1" rowspan="1">250,435</td><td colspan="1" rowspan="1">Indicates a lightweight architecture with few learnable weights.</td></tr><tr><td colspan="1" rowspan="1">Parameters</td><td colspan="1" rowspan="1">0.2504 M</td><td colspan="1" rowspan="1">The model contains only about one-quarter million parameters, making it compact.</td></tr><tr><td colspan="1" rowspan="1">Model size</td><td colspan="1" rowspan="1">0.9553 MB</td><td colspan="1" rowspan="1">The trained model requires less than 1 MB of storage, supporting deployment inresource-limited settings.</td></tr><tr><td colspan="1" rowspan="1">Inference latency</td><td colspan="1" rowspan="1">1.4884 ms/image</td><td colspan="1" rowspan="1">The model produces predictions rapidly for each input image.</td></tr><tr><td colspan="1" rowspan="1">FLOPs</td><td colspan="1" rowspan="1">0.00210944 G</td><td colspan="1" rowspan="1">Indicates low computational cost for the convolutional component.</td></tr><tr><td colspan="1" rowspan="1">GFLOPs</td><td colspan="1" rowspan="1">0.00210944</td><td colspan="1" rowspan="1">Confirms that the model requires very limited floating-point operations.</td></tr><tr><td colspan="1" rowspan="1">Inference memory delta</td><td colspan="1" rowspan="1">0.9180 MB</td><td colspan="1" rowspan="1">Shows low additional memory use during inference.</td></tr></table>

## 4.4 Statistical analysis of the result of 3-class breast mammography

The descriptive training statistics suggest that the mean training accuracy was 0.9488 and the median was 0.9905 (see Table 6). This indicates that the model achieved high accuracy during training. The mean training loss was 0.2628, while the median loss was 0.1952, showing that the loss gradually decreased as the model learned more discriminative features.

Table 6. Descriptive statistics of training accuracy and loss
<table><tr><td rowspan=1 colspan=1>Statistical Measure</td><td rowspan=1 colspan=1>Training Accuracy</td><td rowspan=1 colspan=1>Training Loss</td></tr><tr><td rowspan=1 colspan=1>Mean</td><td rowspan=1 colspan=1>0.9488</td><td rowspan=1 colspan=1>0.2628</td></tr><tr><td rowspan=1 colspan=1>Standard deviation</td><td rowspan=1 colspan=1>0.0691</td><td rowspan=1 colspan=1>0.1120</td></tr><tr><td rowspan=1 colspan=1>Median</td><td rowspan=1 colspan=1>0.9905</td><td rowspan=1 colspan=1>0.1952</td></tr><tr><td rowspan=1 colspan=1>Interquartile range</td><td rowspan=1 colspan=1>0.0857</td><td rowspan=1 colspan=1>0.1576</td></tr><tr><td rowspan=1 colspan=1>Range</td><td rowspan=1 colspan=1>0.4207</td><td rowspan=1 colspan=1>0.6277</td></tr><tr><td rowspan=1 colspan=1>95% confidence interval</td><td rowspan=1 colspan=1>0.9348–0.9628</td><td rowspan=1 colspan=1>0.2401-0.2855</td></tr></table>

The normality tests showed that the epoch-wise accuracy and loss distributions were not normally distributed; therefore, the Wilcoxon test provided additional non-parametric support for the statistical consistency of the training results.

The normality tests showed that the epoch-wise accuracy and loss distributions were not normally distributed; therefore, the Wilcoxon test provided additional non-parametric support for the statistical consistency of the training results.

Table 7. Statistical testing summary of training behavior
<table><tr><td rowspan=1 colspan=1>Test / Measure</td><td rowspan=1 colspan=1>Training Accuracy T</td><td rowspan=1 colspan=1>raining Loss</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>t-statistic</td><td rowspan=1 colspan=1>134.5178           2</td><td rowspan=1 colspan=1>2.9966</td><td rowspan=1 colspan=1>Both metrics showed statistically strong deviation from the tested reference level.</td></tr><tr><td rowspan=1 colspan=1>p-value</td><td rowspan=1 colspan=1> $\overline { { 3 . 8 6 \times 1 0 ^ { - 1 1 1 } } }$ </td><td rowspan=1 colspan=1> $\sqrt { 7 . 8 9 \times 1 0 ^ { - 4 1 } }$ </td><td rowspan=1 colspan=1>Very small p-values indicate statistically significant training trends.</td></tr><tr><td rowspan=1 colspan=1>Effect size</td><td rowspan=1 colspan=1>13.7292             2</td><td rowspan=1 colspan=1>.3471</td><td rowspan=1 colspan=1>Accuracy showed a very large effect size, and loss showed a strong effect as well.</td></tr><tr><td rowspan=1 colspan=1>Bias-corrected    1effect size</td><td rowspan=1 colspan=1>3.6205            2</td><td rowspan=1 colspan=1>.3285</td><td rowspan=1 colspan=1>Confirms the strength of the observed training behavior after correction.</td></tr><tr><td rowspan=1 colspan=1>Normality statistic</td><td rowspan=1 colspan=1>0.7322              0</td><td rowspan=1 colspan=1>.7718</td><td rowspan=1 colspan=1>Values suggest that the epoch-wise distributions were not normally distributed.</td></tr><tr><td rowspan=1 colspan=1>Normality p-value</td><td rowspan=1 colspan=1> $\sqrt { 5 . 2 0 \times 1 0 ^ { - 1 2 } }$ </td><td rowspan=1 colspan=1> $5 . 7 5 \times 1 0 ^ { - 1 1 }$ </td><td rowspan=1 colspan=1>The low p-values indicate significant deviation from normality.</td></tr><tr><td rowspan=1 colspan=1>Wilcoxon statistic0</td><td rowspan=1 colspan=1>.0000              0</td><td rowspan=1 colspan=1>.0000</td><td rowspan=1 colspan=1>Non-parametric testing supports the statistical consistency of the training trend.</td></tr><tr><td rowspan=1 colspan=1>Wilcoxon p-value</td><td rowspan=1 colspan=1> $\overline { { . 2 2 \times 1 0 ^ { - 1 7 } } }$ </td><td rowspan=1 colspan=1> $\sqrt { 1 . 2 2 \times 1 0 ^ { - 1 7 } }$ </td><td rowspan=1 colspan=1>The non-parametric results were statistically significant.</td></tr></table>

Training a 3-class breast mammography model suggests a strong negative relationship between accuracy and loss. The p-values of the Pearson correlation coefficient (-0.9966) and the Spearman correlation coefficient (-0.9981) confirm that, during training, accuracy increased and loss decreased. This suggests a consistent, expected learning pattern for CNN-integrated CCT from breast mammography.

Table 8. Relationship between training accuracy and loss
<table><tr><td rowspan=1 colspan=1>Correlation Type</td><td rowspan=1 colspan=1>Correlation Coefficient      p</td><td rowspan=1 colspan=1>-value</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>Pearson correlation</td><td rowspan=1 colspan=1>-0.9966</td><td rowspan=1 colspan=1> $\sqrt { 5 . 3 0 \times 1 0 ^ { - 1 0 5 } }$ </td><td rowspan=1 colspan=1>Shows a very strong inverse linear relationship betweenlaccuracy and loss.</td></tr><tr><td rowspan=1 colspan=1>Spearman correlation</td><td rowspan=1 colspan=1>-0.9981                          $</td><td rowspan=1 colspan=1>\phantom { + } \overline { { 5 . 7 0 \times 1 0 ^ { - 1 1 7 } } }$ </td><td rowspan=1 colspan=1>Confirms a very strong monotonic inverse relationship</td></tr></table>

Table 9. Ablation Study of the Proposed CNN–integrated CCT Model
<table><tr><td colspan="1" rowspan="1">Experiment</td><td colspan="1" rowspan="1">TrainAcc.</td><td colspan="1" rowspan="1">EvalAcc.</td><td colspan="1" rowspan="1">TestAcc.</td><td colspan="1" rowspan="1">TrainLoss</td><td colspan="1" rowspan="1">EvalLoss</td><td colspan="1" rowspan="1">Test MLoss</td><td colspan="1" rowspan="1">acro BAUC</td><td colspan="1" rowspan="1">estEvalAcc.</td><td colspan="1" rowspan="1">BestEvalLoss</td><td colspan="1" rowspan="1">Epochs</td><td colspan="1" rowspan="1">Aug</td><td colspan="1" rowspan="1">CNNStem</td><td colspan="1" rowspan="1">Pos.Emb.</td><td colspan="1" rowspan="1">Stoch. LDepth</td><td colspan="1" rowspan="1">ayers</td><td colspan="1" rowspan="1">DimH</td><td colspan="1" rowspan="1">eads [L</td><td colspan="1" rowspan="1">abelSmooth</td></tr><tr><td colspan="1" rowspan="1">BaselineCNN-CCT</td><td colspan="1" rowspan="1">99.98%</td><td colspan="1" rowspan="1">98.78%</td><td colspan="1" rowspan="1">98.65%</td><td colspan="1" rowspan="1">0.2250</td><td colspan="1" rowspan="1">.2570</td><td colspan="1" rowspan="1">.261</td><td colspan="1" rowspan="1">0.9989</td><td colspan="1" rowspan="1">98.99%0</td><td colspan="1" rowspan="1">.2574</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.05</td></tr><tr><td colspan="1" rowspan="1">No StochastiDepth</td><td colspan="1" rowspan="1">c99.98</td><td colspan="1" rowspan="1">%99.32%</td><td colspan="1" rowspan="1">99.05%0</td><td colspan="1" rowspan="1">.2240</td><td colspan="1" rowspan="1">.2450</td><td colspan="1" rowspan="1">.261</td><td colspan="1" rowspan="1">0.9999|</td><td colspan="1" rowspan="1">99.32%0</td><td colspan="1" rowspan="1">.245</td><td colspan="1" rowspan="1">67</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">X</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.05</td></tr><tr><td colspan="1" rowspan="1">ShallowTransformer(1 Layer)</td><td colspan="1" rowspan="1">99.98%</td><td colspan="1" rowspan="1">99.26%</td><td colspan="1" rowspan="1">98.92%</td><td colspan="1" rowspan="1">0.2260</td><td colspan="1" rowspan="1">.2440</td><td colspan="1" rowspan="1">.2530</td><td colspan="1" rowspan="1">.9999|</td><td colspan="1" rowspan="1">99.26%|</td><td colspan="1" rowspan="1">0.2444</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.05</td></tr><tr><td colspan="1" rowspan="1">FewerAttentionHeads (2)</td><td colspan="1" rowspan="1">99.96%</td><td colspan="1" rowspan="1">99.12%</td><td colspan="1" rowspan="1">98.79%</td><td colspan="1" rowspan="1">0.2260</td><td colspan="1" rowspan="1">.2530</td><td colspan="1" rowspan="1">.255</td><td colspan="1" rowspan="1">0.9999|</td><td colspan="1" rowspan="1">99.12%0</td><td colspan="1" rowspan="1">.253</td><td colspan="1" rowspan="1">38</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">0.05</td></tr><tr><td colspan="1" rowspan="1">No PositionEmbedding</td><td colspan="1" rowspan="1">al99.9</td><td colspan="1" rowspan="1">8%98.78</td><td colspan="1" rowspan="1">%98.65</td><td colspan="1" rowspan="1">%0.225</td><td colspan="1" rowspan="1">0.250</td><td colspan="1" rowspan="1">0.2540</td><td colspan="1" rowspan="1">.9997</td><td colspan="1" rowspan="1">98.99%0</td><td colspan="1" rowspan="1">.25063</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">X</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.05</td></tr><tr><td colspan="1" rowspan="1">No LabelSmoothing</td><td colspan="1" rowspan="1">99.98%9</td><td colspan="1" rowspan="1">8.71%9</td><td colspan="1" rowspan="1">8.65%0.</td><td colspan="1" rowspan="1">00130</td><td colspan="1" rowspan="1">.0540</td><td colspan="1" rowspan="1">.042</td><td colspan="1" rowspan="1">0.9999</td><td colspan="1" rowspan="1">98.78%0</td><td colspan="1" rowspan="1">.0543</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0</td></tr><tr><td colspan="1" rowspan="1">No CNNStem</td><td colspan="1" rowspan="1">99.03%</td><td colspan="1" rowspan="1">98.51%</td><td colspan="1" rowspan="1">98.38%</td><td colspan="1" rowspan="1">0.251</td><td colspan="1" rowspan="1">0.2610</td><td colspan="1" rowspan="1">.262</td><td colspan="1" rowspan="1">0.99969</td><td colspan="1" rowspan="1">8.58%|0</td><td colspan="1" rowspan="1">.261</td><td colspan="1" rowspan="1">77</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">X</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.05</td></tr><tr><td colspan="1" rowspan="1">No DataAugmentation</td><td colspan="1" rowspan="1">99.98%</td><td colspan="1" rowspan="1">98.78%</td><td colspan="1" rowspan="1">98.25%</td><td colspan="1" rowspan="1">0.226</td><td colspan="1" rowspan="1">0.2640</td><td colspan="1" rowspan="1">.2740</td><td colspan="1" rowspan="1">.9995</td><td colspan="1" rowspan="1">98.91%0</td><td colspan="1" rowspan="1">.264</td><td colspan="1" rowspan="1">26</td><td colspan="1" rowspan="1">X</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">V</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">64</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">0.05</td></tr></table>

To systematically evaluate the contribution of each architectural component in the proposed CNN–CCT framework, an extensive ablation study was conducted, as summarized in Table 9. The results demonstrate that the full model achieves the most balanced performance across accuracy, loss stability, and Macro AUC, confirming the effectiveness of the hybrid CNN– Transformer design.

The removal of stochastic depth leads to a slight improvement in evaluation accuracy; however, this is accompanied by less stable convergence behavior across epochs, indicating reduced regularization effectiveness. Similarly, reducing the transformer depth to a single layer maintains competitive accuracy but slightly limits the model’s ability to capture deep global contextual dependencies. This confirms that even shallow transformer structures can perform well when supported by strong CNN feature extraction, but additional depth enhances representational richness.

Reducing the number of attention heads results in a minor decline in test performance, highlighting the importance of multi-head self-attention in capturing diverse feature subspaces within mammographic representations. In contrast, removing the CNN stem produces a more noticeable drop in performance, confirming that convolutional inductive bias is essential for extracting local texture patterns such as lesion boundaries and intensity variations.

The absence of positional embeddings also negatively impacts performance, demonstrating that spatial encoding remains critical even in compact token-based architectures. Furthermore, removing data augmentation leads to the most significant degradation in generalization, as reflected in increased test loss and reduced accuracy, underscoring its role in mitigating dataset bias and improving robustness.

Finally, disabling label smoothing yields highly confident but less well-calibrated predictions, as evidenced by extremely low training loss and reduced generalization performance. Overall, the ablation study confirms that each component contributes synergistically to the final performance and that integrating CNN feature extraction with transformer-based global reasoning yields a robust and clinically reliable classification framework.

XAI integration analysis

![](images/2772c423c0a70d6e3af66f00ac242d287ec6ec0d288df3f299ae7a21232e3462.jpg)  
Figure 15. Grad-CAM visualizations with class probability scores

The Grad-CAM visualizations provide qualitative evidence of the model’s decision-making process (see Figure 15). The class probability scores demonstrate that the proposed CNNintegrated CCT model produced confident predictions for Benign, Malignant, and Normal samples. The highlighted activation regions were mostly located within tissue-containing regions, suggesting that the model relied on relevant visual patterns rather than background information. The Grad-CAM heatmaps show that the model focused mainly on visible tissue regions rather than empty background areas. For Malignant cases, the highlighted regions were generally concentrated in dense, irregular tissue patterns. For Benign and Normal cases, the activation regions appeared more localized and less aggressive in distribution. These visual results suggest that the model learned meaningful image features and relied on diagnostically relevant regions for its predictions.

![](images/60d6030152dbf7db1eaee51977110c9819812b17db00c14472d338a9675987d8.jpg)  
Figure 16: Comparative explainability visualization of the proposed CNN-integrated CCT model using fused Grad-CAM–attention maps, Grad-CAM, attention maps, and LIME

Figure 16 shows the explainability results of the proposed CNN-integrated CCT model for representative breast image samples. The visualization results indicate that the model mainly focused on informative tissue regions rather than background areas. For example, for a Malignant sample, the highlighted regions were mostly located around dense and irregular tissue patterns. The highlighted regions indicate image areas that contributed most strongly to the model prediction. The results show that the proposed model focused on relevant breast tissue regions, supporting the transparency and reliability of the classification decision.

![](images/a4287ccbb4fe441f63019d7c9ccbc45a1cadc37443f02095b4ced8bc93c78a83.jpg)  
Figure 17: Explainable artificial intelligence visualization of the proposed model

Figure 17 combines Grad-CAM, attention map, LIME, and SHAP explanations. The color scale ranges from low to high contribution, with brighter regions indicating a stronger influence on the model prediction. The visual explanations show that the proposed model focused on relevant breast tissue regions for Benign and Malignant classification, supporting the interpretability and trustworthiness of the prediction process.

## 5. Discussion

Significant studies applied DL methods to detect and classify breast cancer. Our proposed CNN-integrated CCT consistently achieves strong performance across 5-fold cross-validation experiments on 2-class, 3-class, and 5-class mammographic datasets. The model achieved 99%–100% accuracy across 3 datasets, indicating robust generalization. Moreover, performance stability across folds suggests that the hybrid architecture effectively mitigates overfitting despite dataset imbalance and inter-dataset variability. However, slight performance degradation was observed on smaller, more challenging datasets such as Mini-MIAS and INbreast, where recall values dropped to 93–97% across certain folds. This indicates that class scarcity and domain-specific imaging variability continue to influence feature separability, particularly under low-sample conditions.

Breast Cancer Classification Performance Comparison (Proposed CNN-CCT vs. State-of-the-Art Methods)

![](images/91f6a05aae3c15695b3de177fb792f4ddbc56d7644fa1e6a14134a41bb763a57.jpg)  
Key Takeaway: The proposed CNN-CCT model achieves 98.65% test accuracy, demonstrating competitive performance compared to recent state-of-the-art methods while maintaining a more compact and efficient architecture, suitable for real-world clinical deployment.

Figure 18: Breast Cancer Classification Performance Comparison

Comparing the recent state-of-the-art studies, our model demonstrates competitive or superior performance. For instance, CNN–LSTM architectures (Kaddes et al., 2025) achieve extremely high accuracy (\~99.9%) but are limited by modality constraints and lack generalization across diverse datasets. Similarly, Swin CNN-based hybrid transformers (Sreeleklekshmi et al., 2024) and ViT-CNN architectures (Abimouloud et al., 2024) achieve strong accuracy (\~98%), but at the cost of high computational complexity and increased training overhead. In contrast, the proposed CNN–integrated CCT model achieves comparable or improved accuracy while maintaining a more compact and efficient transformer design.

Traditional CNN-based approaches, such as Simonyan et al. (2024) and ensemble fusion models (Chakravarthy et al., 2024), generally achieve lower or comparable accuracy (90–98%), primarily due to limited global contextual modeling. Likewise, wavelet-CNN and PSOoptimized CNN models (Bohra et al., 2026; Alzahrani et al., 2025) improve feature extraction but incur additional computational cost and do not fully address long-range dependency modeling.

The superior performance of the proposed framework can be attributed to three key design advantages. First, the CNN backbone efficiently captures fine-grained spatial texture patterns such as microcalcifications and mass boundaries. Second, the CCT tokenizer reduces computational redundancy by converting feature maps into compact token representations. Third, the transformer encoder enhances global contextual reasoning through multi-head selfattention, enabling better discrimination between visually similar classes.

Despite these strengths, certain limitations persist. Performance variability in smaller datasets indicates that the model still depends on sufficient training diversity. Additionally, although transformer layers are computationally more efficient than standard Vision Transformers, their inclusion still introduces non-trivial inference overhead compared to pure CNN models.

Overall, the results confirm that integrating convolutional inductive bias with transformerbased global representation learning yields a balanced and effective solution for medical image classification. The CNN–integrated CCT framework demonstrates strong potential for clinical decision support systems, particularly in breast cancer screening applications where both accuracy and robustness are critical.

Prior studies emphasized the need for DL on edge devices for real-time detection and monitoring of breast cancers. Figure 18 compares the parameter counts of SOTA CNNs and CNN-integrated CCT, demonstrating that CNN-integrated CCT is a suitable model for edge devices. This is because the model will require fewer computational resources than the CNN. The XAI implementation is not limited to generating heatmaps using LIME, SHAP, and Grad-CAM. Instead, our XAI implementation is much more extensive than only generating heatmaps.

## 6. Limitations, Future, and Conclusions

Despite the promising performance of the proposed CNN-integrated CCT model for breast mammography image classification, several limitations should be acknowledged. The main limitation of this study is that it utilized a secondary dataset to evaluate the performance of the CNN-integrated CCT model in breast cancer detection and classification. However, we would like to point out that primary data collection is difficult in the area where this research was conducted. Second, this research considered only mammographic images of breast cancer; however, the model can be tested on Magnetic Resonance Imaging (MRI), Computed Tomography (CT), Ultrasound (US), X-ray Imaging (XR), Plasmid Bluescript (PBS), and Microscopy images. Future studies, therefore, should conduct extensive multi-center validation using the model across heterogeneous datasets.

There is room for improvement in the model as well. Since the model training process starts with input augmentation of mammography images, the quality and diversity of the input mammographic dataset are important. Any bias in the input image may propagate through subsequent CCT tokenizer and transformer encoder stages, potentially limiting generalization to unseen clinical trial settings. In short, variations in mammographic devices, acquisition protocols, and operator-dependent artifacts may reduce robustness when deployed in realworld hospital environments. Although the CNN backbone reduces the spatial complexity of the input mammography image before tokenization, the transformer encoder may introduce additional computational overhead due to multi-head self-attention and stochastic depth regularization. This increases inference latency compared to purely convolutional architectures, potentially restricting deployment in resource-constrained or real-time diagnostic systems. Future studies should optimize the transformer encoder by using lightweight attention mechanisms, such as linear attention, sparse attention, or token pruning. The CCT tokenizer and positional embedding module assume a fixed patch-based representation of CNN feature maps. This fixed tokenization strategy may result in suboptimal representation of fine-grained tumor boundaries, particularly when lesion morphology is highly irregular or low-contrast. The model may be sensitive to hyperparameter configurations, including the embedding dimension (D), the number of transformer layers (L), the patch size, and the dropout rate. Suboptimal tuning may lead to instability in convergence or degraded classification performance.

The proposed CNN-integrated CCT framework effectively combines local mammographic feature extraction with global contextual learning, enabling the detection of subtle abnormalities while maintaining efficient transformer-based representation learning. The study

demonstrates the potential of convolutional tokenization and compact transformer design for developing accurate and robust computer-aided breast cancer classification systems.

In the model. The CNN layer captures hierarchical local features, while the CCT tokenizer transforms these feature maps into compact patch embeddings, which are critical for accurate breast lesion characterization in mammographic imaging. Positional embeddings are used to preserve spatial information, and a lightweight transformer encoder is employed to model longrange contextual dependencies via multi-head self-attention. Finally, a classification head produces the diagnostic prediction for benign and malignant cases.

Experimental results indicate that the hybrid CNN-integrated CCT model achieves strong classification performance for 2-, 3-, and 5-class breast lesions. With only 250,435 parameters, the model is a suitable model for real-time clinical deployment, further enhancing its applicability in computer-aided diagnosis systems. The model was integrated with LIME, SHAP, and Grad-CAM to deepen our understanding of how it detects and classifies input images.

Finally, since many scholars advocate integrating AI for practicality, we present a prototype that combines the CNN-integrated CCT model, a Streamlit-based web interface, and an Android mobile application. The prototype is expected to support ongoing efforts in the biomedical field to make CAD more practical, interpretable, and widely usable. Furthermore, while the core focus is on breast cancer classification, the broader framework has potential applications in other areas of disease detection, including agricultural disease management, where similar challenges around accessibility and decision support persist.

## References

1. Kaddes, M., Ayid, Y. M., Elshewey, A. M., & Fouad, Y. (2025). Breast cancer classification based on a hybrid CNN with an LSTM model. Scientific Reports, 15(1), 4409.

2. Sreelekshmi, V., Pavithran, K., & Nair, J. J. (2024). Swincnn: An integrated swin transformer and CNN for improved breast cancer grade classification. IEEE Access, 12, 68697- 68710

3. Chakravarthy, S., Bharanidharan, N., Khan, S. B., Kumar, V. V., Mahesh, T. R., Almusharraf, A., & Albalawi, E. (2024). Multi-class breast cancer classification using CNN features hybridization. International Journal of Computational Intelligence Systems, 17(1), 191.

4. Elzaghmouri, B., Elwasil, O., Elaiwat, S., Al-Khateeb, A., AbdelRahman, S. M., Osman, A. A. F., ... & Doumi, A. B. (2025). Comprehensive evaluation of transfer-CNN based models for breast cancer detection. Journal of Information Systems Engineering and Management, 10(18s), 1-15.

5. Brahmareddy, A., & Selvan, M. P. (2025). TransBreastNet a CNN transformer hybrid deep learning framework for breast cancer subtype classification and temporal lesion progression analysis. Scientific Reports, 15(1), 35106.

Gül, M. (2025). A novel local binary patterns-based approach and proposed CNN model to diagnose breast cancer by analyzing histopathology images. IEEE Access.

Alzahrani, R. M., Sikkandar, M. Y., Begum, S. S., Babetat, A. F. S., Alhashim, M., Alduraywish, A., ... & Ng, E. Y. (2025). Early breast cancer detection via infrared thermography using a CNN enhanced with particle swarm optimization. Scientific Reports, 15(1), 25290.

Simonyan, E. O., Badejo, J. A., & Weijin, J. S. (2024). Histopathological breast cancer classification using CNN. Materials Today: Proceedings, 105, 268-275.

Fontes, J. P. P., Raimundo, J. N. C., Magalhães, L. G. M., & Lopez, M. A. G. (2025). Accurate phenotyping of luminal A breast cancer in magnetic resonance imaging: A new 3D CNN approach. Computers in biology and medicine, 189, 109903.

Liu, W., Liang, S., & Qin, X. (2024). A novel embedded kernel cnn-pcff algorithm for breast cancer pathological image classification. Scientific Reports, 14(1), 23758.

Mannarsamy, V., Mahalingam, P., Kalivarathan, T., Amutha, K., Paulraj, R. K., & Ramasamy, S. (2025). Sift-BCD: SIFT-CNN integrated machine learning-based breast cancer detection. Biomedical Signal Processing and Control, 106, 107686.

Bohra, M., Singh, K. U., Kumar, I., & Shah, M. A. (2026). Wavelet-CNN feature fusion architecture for robust breast cancer classification in histopathological imaging. International Journal of Computational Intelligence Systems, 19(1), 136.

Nayak, D. R. (2024). RDTNet: A residual deformable attention based transformer network for breast cancer classification. Expert Systems with Applications, 249, 123569.

Sreelekshmi, V., Pavithran, K., & Nair, J. J. (2024). Swincnn: An integrated swin transformer and cnn for improved breast cancer grade classification. IEEE Access, 12, 68697-68710.

Abimouloud, M. L., Bensid, K., Elleuch, M., Ammar, M. B., & Kherallah, M. (2024). Vision transformer based convolutional neural network for breast cancer histopathological images classification. Multimedia Tools and Applications, 83(39), 86833-86868.

Feng, J., Dong, X., Liu, X., & Zheng, X. (2024). HTBE-Net: A hybrid transformer network based on boundary enhancement for breast ultrasound image segmentation. Displays, 84, 102753.

Goceri, E. (2025). A convolution and transformer-based method with effective stain normalization for breast cancer detection from whole slide images. Biomedical Signal Processing and Control, 110, 108138.

Aldawsari, M. A., Aldosari, S. J., Ismail, A., & Emam, M. M. (2026). A deep learning framework for breast cancer diagnosis using Swin Transformer and Dual-Attention Multi-scale Fusion Network. Scientific Reports.

Vanitha, K., Manimaran, A., Chokkanathan, K., Anitha, K., Mahesh, T. R., Kumar, V. V., & Vivekananda, G. N. (2024). Attention-based feature fusion with external attention transformers for breast cancer histopathology analysis. IEEE Access, 12, 126296-126312.

Katayama, A., Aoki, Y., Watanabe, Y., Horiguchi, J., Rakha, E. A., & Oyama, T. (2024). Current status and prospects of artificial intelligence in breast cancer pathology: convolutional neural networks to prospective Vision Transformers. International Journal of Clinical Oncology, 29(11), 1648-1668.

Acikgoz, H., Aytekin, A., & Gezici, S. (2026). BreasTransNeXt: An Enhanced Multi-Module Vision Transformer For Early Breast Cancer Diagnosis. Journal of Imaging Informatics in Medicine, 1-20.

Abd Elaziz, M., Dahou, A., Aseeri, A. O., Ewees, A. A., Al-Qaness, M. A., & Ibrahim, R. A. (2024). Cross vision transformer with enhanced Growth Optimizer for breast cancer detection in IoMT environment. Computational Biology and Chemistry, 111, 108110.

Hussain, S., Ali, M., Naseem, U., Bosques Palomo, B. A., Monsivais Molina, M. A., Garza Abdala, J. A., ... & Tamez Pena, J. G. (2024, October). Performance evaluation of deep learning and transformer models using multimodal data for breast cancer classification. In MICCAI Workshop on Cancer Prevention through Early Detection (pp. 59-69). Cham: Springer Nature Switzerland.

Hassani, A., Walton, S., Shah, N., Abuduweili, A., Li, J., & Shi, H. (2021). Escaping the big data paradigm with compact transformers. arXiv. https://arxiv.org/abs/2104.05704

EmergentMind. (2026). Overview of compact convolutional transformer architectures. https://emergentmind.com

Hosseini, M., et al. (2025). Discussion on efficient attention mechanisms in hybrid vision transformers. Machine Learning Systems Review.

Fantozzi, P., & Naldi, M. (2024). The explainability of transformers: Current status and directions. Computers, 13(4), 92. https://doi.org/10.3390/computers13040092

Xie, W., Li, X.-H., Cao, C. C., & Zhang, N. L. (2023). ViT-CX: Causal explanation of vision transformers. Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI-23). https://doi.org/10.24963/ijcai.2023/174