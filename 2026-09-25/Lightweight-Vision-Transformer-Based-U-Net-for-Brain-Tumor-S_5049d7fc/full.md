# Lightweight Vision Transformer-Based U-Net for Brain Tumor Segmentation from MRI

1<sup>st</sup> Sheekar Banerjee

Department of CSE IUBAT

Dhaka, Bangladesh sheekar.cse@iubat.edu

2<sup>nd</sup> Md. Srabon Chowdhury

Department of CSE

IUBAT

Dhaka, Bangladesh chowdhurysrabon2002@gmail.com

3<sup>rd</sup> Md. Mahbub Hasan Akash

Department of CSE IUBAT

Dhaka, Bangladesh mahbub.akashp@gmail.com

4<sup>th</sup> Ishtiak Al Mamoon   
Department of CSE   
IUBAT   
Dhaka, Bangladesh   
ishtiak.cse@iubat.edu

Abstract—Accurate brain tumor segmentation from Magnetic Resonance Imaging (MRI) is essential for diagnosis, treatment planning, and surgical guidance. Although Convolutional Neural Networks (CNNs), particularly U-Net, have achieved significant success in medical image segmentation, they often struggle to capture the long-range spatial dependencies required to model tumors with irregular shapes and complex boundaries. This paper proposes a lightweight Vision Transformer U-Net (ViT-UNet) that combines the hierarchical feature extraction capability of U-Net with the global context modeling of Vision Transformers. The proposed architecture incorporates a compact ViT bottleneck within a U-Net encoder-decoder framework, enabling effective learning of both local and global features while maintaining computational efficiency with only 2.6 million trainable parameters. The model was evaluated on the TCGA-LGG MRI Segmentation dataset, achieving a mean Intersection over Union (IoU) of 0.8100 and a Dice score of 0.8446, outperforming the baseline U-Net by 3.75% and 3.15%, respectively. Extensive quantitative and qualitative analyses, including confusion matrix evaluation, precision–recall curves, per-image performance distribution, and tumor-size dependency analysis, demonstrate the effectiveness and robustness of the proposed method for brain tumor segmentation.

Index Terms—Brain Tumor Segmentation, Vision Transformer, U-Net, Medical Image Analysis, Deep Learning, MRI.

## I. INTRODUCTION

Brain tumors are among the most life-threatening neurological diseases, and accurate segmentation from Magnetic Resonance Imaging (MRI) is important for diagnosis and treatment planning [1]. However, manual tumor delineation is time-consuming and subjective, motivating the development of automated segmentation methods [2]. U-Net and its variants have achieved strong performance in medical image segmentation through encoder-decoder architectures with skip connections [3]. However, CNNs mainly capture local spatial features and may have limited ability to model long-range dependencies, particularly for tumors with irregular shapes and ambiguous boundaries [4]. Vision Transformers (ViTs) address this limitation through self-attention and global contextual modeling [5]. However, existing hybrid CNN-Transformer architectures, such as TransUNet, Swin-UNet, UNETR, and VT-UNet, can introduce considerable computational and parameter overhead [6], [7]. To address this limitation, we propose a lightweight ViT-UNet that combines CNN-based local feature extraction with global contextual modeling through a compact

Vision Transformer bottleneck. Unlike architectures that use Transformer modules extensively throughout the network, our approach applies the Transformer only at the bottleneck, using two self-attention blocks with four heads. This design maintains a compact model size of approximately 2.6 million parameters while providing an effective balance between segmentation performance and computational efficiency. The main contributions of this work are:

1) A compact hybrid ViT-UNet architecture that integrates a shallow Vision Transformer exclusively at the U-Net bottleneck to capture global contextual dependencies while retaining convolution-based hierarchical feature extraction.

2) A lightweight Vision Transformer bottleneck consisting of two self-attention blocks with four attention heads, designed to introduce global context with limited additional computational and parameter overhead.

3) A hybrid segmentation framework combining the proposed Transformer bottleneck with the U-Net encoderdecoder and hybrid loss formulation, enabling complementary modeling of local boundaries and global tumor context.

4) Comprehensive evaluation using quantitative metrics, qualitative segmentation analysis, ablation experiments, and comparison with the baseline U-Net and existing hybrid segmentation approaches.

5) The proposed model achieves a mean Intersection over Union (IoU) of 0.8100 and a Dice score of 0.8446 on the TCGA-LGG MRI segmentation dataset.

The remainder of this paper is organized as follows. Section II reviews related work; Section III presents the proposed methodology; Section IV describes the experimental results; Section V discusses the findings and limitations; and Section VI concludes the paper.

## II. RELATED WORK

This section reviews representative studies on brain tumor segmentation using CNNs, Vision Transformers, and hybrid architectures, with emphasis on the trade-off between segmentation performance and computational complexity.

## A. U-Net and CNN-based Segmentation

U-Net, introduced by Ronneberger et al. [3], uses an encoder-decoder architecture with skip connections and has become a widely used framework for medical image segmentation. Several variants, including 3D U-Net [8], Attention U-Net [9], and nnU-Net [10], have further improved segmentation performance. However, CNN-based models primarily rely on local convolutional operations and may have difficulty modeling long-range spatial dependencies, particularly for tumors with irregular shapes and ambiguous boundaries.

## B. Vision Transformers in Medical Imaging

Vision Transformers (ViTs) use self-attention to model longrange dependencies and global contextual information [5], [11]. Hybrid architectures such as TransUNet [6], Swin-UNet [7], and VT-UNet [12] combine Transformer-based global modeling with U-Net-style feature extraction and decoding. Although these approaches provide effective contextual representation, their relatively large Transformer components can increase the number of parameters and computational requirements. In contrast, the proposed ViT-UNet uses a compact Vision Transformer only at the U-Net bottleneck rather than employing Transformer modules throughout the network. This design retains CNN-based local feature extraction while introducing global contextual modeling at a reduced spatial resolution. The proposed bottleneck uses two self-attention blocks with four attention heads and contributes to a total model size of approximately 2.6 million parameters.

## C. Brain Tumor Segmentation

The BraTS challenges have established widely used benchmarks for brain tumor segmentation [13], [14]. In this work, the TCGA-LGG dataset is used to evaluate binary tumor segmentation for lower-grade glioma cases. The objective is to develop a lightweight segmentation framework that can capture both local tumor features and global contextual information while maintaining a compact model size.

## III. METHODOLOGY

To enhance brain tumor segmentation, the proposed method is based on the ViT-UNet, which incorporates a lightweight Vision Transformer (ViT) bottleneck in the U-Net. This section outlines the preparation of the dataset, the network architecture, the loss function, the training procedure and the evaluation metrics.

## A. Dataset and Preprocessing

We use the TCGA-LGG MRI Segmentation dataset which contains MRI scans and corresponding binary tumor masks from 110 patients. Each of them is in TIF format consisting of slices in all three directions of (axial, coronal, sagittal). and all available slices were used for training.

Figure 1 presents representative MRI slices and their corresponding ground truth tumor masks from the LGG dataset.

Brain MRI Scans with Tumor Segmentation Masks  
![](images/655dc139a89ca0fa89cf04c4ff549b32867f341ac0c72ac6d2c9df47c636f765.jpg)  
Fig. 1. Example MRI slices with corresponding ground truth tumor masks from the LGG dataset.

## Preprocessing Pipeline:

• MRI images are resized to 224 × 224 pixels and normalized to the range [0,1].

• Data augmentation includes random horizontal and vertical flips with probabilities of 0.5 and 0.3.

• The dataset is split into training (80%), validation (10%), and test (10%) sets using stratified sampling.

## B. Model Architecture

The proposed ViT-UNet integrates a U-Net encoder-decoder with a Vision Transformer (ViT) bottleneck, as shown in Fig. 2.

Encoder:The hierarchical features are extracted using maxpooling, whereas feature channel (32–256) is increased in four DoubleConv blocks with batch normalization and ReLU.

ViT Bottleneck:To obtain global contextual information from the deepest feature representation, a lightweight Vision Transformer consisting of two blocks of self-attention with 4 heads is used.

Decoder: A number of transposed convolutions and skip connections are used to focus on recovering the original spatial resolution, then followed by $\textbf { a } 1 \times 1$ convolution to produce the binary segmentation mask. The proposed network is composed of about 2.6 million trainable parameters keeping the complexity and accuracy in a balance.

## C. Loss Function

To optimize segmentation performance, we use a hybrid loss that combines Binary Cross-Entropy (BCE) and Dice loss:

$$
\mathcal { L } = \alpha \mathcal { L } _ { B C E } + ( 1 - \alpha ) \mathcal { L } _ { D i c e } ,
$$

where $\alpha \ = \ 0 . 5$ . BCE improves pixel-wise classification, while Dice loss maximizes the overlap between predicted and ground-truth masks, effectively addressing class imbalance. The Dice loss is defined as

$$
\mathcal { L } _ { D i c e } = 1 - \frac { 2 | P \cap G | + \epsilon } { | P | + | G | + \epsilon } ,
$$

where P and G denote the predicted and ground-truth masks, respectively, and $\epsilon = 1 0 ^ { - 6 }$ ensures numerical stability.

![](images/44a26e0c1b514696bdb3202128c80ef037827672f2efec11418531ef79f4dae8.jpg)  
Fig. 2. Proposed ViT-UNet architecture. The encoder extracts hierarchical features, the ViT bottleneck captures global context, and the decoder reconstructs the segmentation map using skip connections.

## D. Training Details

The proposed model was trained for 15 epochs using the hyperparameters listed in Table I. The AdamW optimizer was

TABLE I  
TRAINING HYPERPARAMETERS
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight Decay</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Batch Size</td><td>8</td></tr><tr><td>Scheduler</td><td>Cosine Annealing  $( T _ { \mathrm { m a x } } = 1 5 )$ </td></tr><tr><td>Loss Function</td><td>BCE + Dice (α = 0.5)</td></tr><tr><td>Early Stopping</td><td>Validation IoU</td></tr></table>

used to optimize the model, and learning rate was scheduled using cosine annealer with k=22. To benefit memory efficiency during training, gradient accumulation and automatic mixed precision (AMP) were used. All experiments were implemented in PyTorch and executed on a 12GB NVIDIA GPU with CUDA support.

## E. Evaluation Metrics

The model was evaluated segmentation performance using standard metrics:

• Intersection over Union (IoU): $\begin{array} { r } { \mathrm { I o U } = \frac { | P \cap G | } { | P \cup G | } } \end{array}$ , computed with a threshold of 0.5 on the predicted probabilities.

• Dice Similarity Coefficient (DSC): $\begin{array} { r } { \mathrm { D S C } = \frac { 2 | P \cap G | } { | P | + | G | } } \end{array}$

• Precision: Precision $= { \frac { T P } { T P + F P } }$

• Recall: Recall $\begin{array} { r } { { \bf \Pi } = \frac { T P } { T P + F N } . } \end{array}$

• Specificity: $\begin{array} { r } { { \mathrm { S p e c i f i c i t y } } = \frac { T N } { T N + F P } . } \end{array}$

The metrics are able to measure the segmentation quality comprehensively, by considering segmentation overlap (IoU, Dice) as well as classification accuracy (precision, recall, specificity).

## IV. EXPERIMENTAL RESULTS

In this section, we evaluate our proposed ViT-UNet on the TCGA-LGG dataset through experiments. To evaluate performance in terms of segmentation accuracy, robustness, and computational efficiency, quantitative, qualitative, and ablation analyses are performed.

## A. Quantitative Evaluation

The model was evaluated the model on the held-out test set (10% of total data). The test set performance is summarized in Table II. The Table II gives the details of segmentation performance of the proposed model on the test data. The model shows a great performance in terms of IoU, Dice, precision, recall and specificity highlighting accurate and reliable brain tumour segmentation. The variation in IoU and Dice is due to the different tumor sizes and shapes observed. Figure 4 is an overall visualization of performance measurement.

TEST SET PERFORMANCE METRICS  
TABLE II
<table><tr><td>Metric</td><td>Mean</td><td>Std Dev</td></tr><tr><td>IoU</td><td>0.8100</td><td>0.3142</td></tr><tr><td>Dice</td><td>0.8446</td><td>0.2949</td></tr><tr><td>Precision</td><td>0.8051</td><td></td></tr><tr><td>Recall</td><td>0.8260</td><td></td></tr><tr><td>Specificity</td><td>0.9973</td><td></td></tr><tr><td>Best Validation IoU</td><td>0.6527</td><td></td></tr></table>

![](images/2eeef9a4cf618076d4c5b0356a9a44cf140a2f476f561e961cb75725e4a2c6e9.jpg)  
Fig. 3. Performance analysis summary showing various metrics and their distributions.

## B. Comparison with Baseline

The proposed ViT-UNet is compared with a standard U-Net [3] trained under identical settings. Table III shows that the

TABLE III  
COMPARISON OF THE PROPOSED METHOD WITH REPRESENTATIVE CNN-AND TRANSFORMER-BASED SEGMENTATION MODELS.
<table><tr><td>Method</td><td>IoU</td><td>Dice</td><td>Parameters</td></tr><tr><td>U-Net [3]</td><td>0.7807</td><td>0.8188</td><td>2.4M</td></tr><tr><td>TransUNet [6]</td><td>0.7770</td><td>0.8210</td><td>105M</td></tr><tr><td>Swin-UNet [7]</td><td>0.7900</td><td>0.8270</td><td>27.1M</td></tr><tr><td>VT-UNet [14]</td><td>0.8010</td><td>0.8360</td><td>25.8M</td></tr><tr><td>EF-VPT-Net [15]</td><td>0.8070</td><td>0.8420</td><td>18.5M</td></tr><tr><td>Proposed ViT-UNet</td><td>0.8100</td><td>0.8446</td><td>2.6M</td></tr></table>

proposed ViT-UNet achieves a 3.75% higher IoU and a 3.15% higher Dice score than the baseline U-Net, with only a 0.2M increase in parameters. This illustrates that the lightweight ViT bottleneck effectively improves segmentation performance.

## C. Confusion Matrix Analysis

The pixel-wise confusion matrix on the test set is presented with the corresponding heatmap shown in Fig. 4.

the proposed model achieves high specificity (99.73%) and sensitivity (82.6%), correctly classifying the majority of tumor and background pixels while maintaining a low false-positive rate. Figure 4 provides the corresponding visual representation.

![](images/b822c8678695c0a5be576d0762a8c9993cbb95ac38776e16c187d2c28e7d4c61.jpg)  
Fig. 4. Pixel-wise confusion matrix heatmap of the proposed model.

## D. Precision–Recall Analysis

Figure 5 shows the precision–recall (PR) curve of the proposed model. This model’s discriminative power is about 0.89 by AUC. It shows high accuracy at a wide recall level and is very effective in detecting tumors and has a low value of false positive.

![](images/21991eb44db4e7dfb1a546d3fcdc8fed99b88a56d1eb60171bd7e42bcfd98b75.jpg)  
Fig. 5. Precision–Recall curve of the proposed model.

## E. Tumor Size vs. Performance

Figure 6 shows the correlation between tumor size and IoU. This is seen as a moderate positive relationship $( r = 0 . 3 7 )$ : In general, segmentation accuracy increases with larger size of the tumours. However, the proposed model can detect many small tumors with satisfactory IoU, indicating the capacity to detect tumor with different size.

## F. Distribution Analysis

Figure 7 Examples of the results are shown in per-image IoU and Dice scores. Results show consistently good segmentation performance, with median IoU (Dice) scores of 0.74 (0.99). Although the accuracy is high in most images, a few more difficult images show lower scores, which may be attributed

![](images/b12a2d5c2c69338396265a30654cc3f16d07f17ac26c41fb9daf5c8a90638ec2.jpg)  
Fig. 6. IoU versus tumor area with a linear trend line.

to the differences in the size, shape and contrast of the images of tumors.  
![](images/3a6fa88b05592eb04b86241796f9ce9328197290b254477006e47158f960caec.jpg)

![](images/35bd8427cf766c90ccb666b607714a18ff260958061c0c977c3ed43112833a61.jpg)  
Fig. 7. Distribution of per-image IoU and Dice scores using box and violin plots.

## G. Qualitative Results

Figure 8 Shows representative segmentation results. The proposed model is able to capture boundaries of tumors with high accuracy compared with the ground truth with irregular boundaries, with only minor discrepancies in the boundaries in challenging cases. Additionally, we showcase the best and worst predicted cases in Figure 9 to highlight the model’s strengths and failure modes.

## H. ABLATION STUDY

To assess the impact of each of the parts of the proposed ViT-UNet architecture, an ablation study is performed on the LGG dataset, and the results are shown in Table IV. We successively examine the Vision Transformer (ViT) bottleneck and the hybrid loss function.

## I. Training Dynamics

Figure 10 shows the training process of the proposed model. The training and validation losses show a clear steady decline whereas IoU and Dice metrics show gradual improvement which illustrates the smooth convergence without overtraining. The cosine annealing learning rate schedule is also used which further helps to optimize the model during the training process.

![](images/a990b257cec25c430598039f204db0838ada35e1ee52191de4ac66e8d77ba58c.jpg)  
Fig. 8. Qualitative segmentation results: original image, ground truth, prediction, overlay.

![](images/a20810c60405c1026d8574ed8cfe605dcbc46153d4b1604bf7b834fe200c33b2.jpg)  
Fig. 9. Best and worst segmentation predictions based on IoU score, illustrating the range of performance.

## V. DISCUSSION

Evaluation of the proposed architecture, interpretation of the experiment results and suggestions for future research and limitations of this study are addressed.

## A. Performance Interpretation

The proposed ViT-UNet achieved a mean IoU of 0.8100 and a Dice score of 0.8446, demonstrating promising brain tumor segmentation performance. The model achieved high specificity (99.73%) and a sensitivity of 82.6%, indicating effective tumor detection while maintaining a low false-positive rate. Further investigation will focus on multimodal MRI data and improved learning strategies to enhance segmentation performance and generalizability.

ABLATION STUDY OF THE PROPOSED VIT-UNET ARCHITECTURE ON THE LGG DATASET.
<table><tr><td>Model Variant</td><td>ViT</td><td>Hybrid Loss</td><td>IoU</td><td>Dice</td></tr><tr><td>U-Net Baseline</td><td>×</td><td>×</td><td>0.7807</td><td>0.8188</td></tr><tr><td>U-Net + Hybrid Loss</td><td>×</td><td>√</td><td>0.7924</td><td>0.8296</td></tr><tr><td>U-Net + ViT Bottleneck</td><td>√</td><td>X</td><td>0.8031</td><td>0.8372</td></tr><tr><td>Proposed ViT-UNet</td><td>√</td><td>√</td><td>0.8100</td><td>0.8446</td></tr></table>

![](images/34248ffd9f9b677f25ce413f7eaefedd27c00958223765907d969adfb0a9a71b.jpg)  
Fig. 10. Training and validation loss, IoU, and Dice over the training epochs.

## B. Limitations and Future Work

Despite the promising performance of the proposed model, this study has several limitations. The model was evaluated on a relatively small LGG dataset using single-modality 2D MRI images, which may limit its generalizability across different datasets, MRI modalities, and acquisition protocols. Future work will focus on evaluation using larger and more diverse datasets, multimodal and 3D segmentation, explainable AI, and real-time optimization to further improve the model’s robustness and practical applicability.

## VI. CONCLUSION

This paper presented a lightweight brain tumor segmentation model based on a Vision Transformer (ViT) bottleneck incorporated into a U-Net architecture. The proposed model combines CNN-based local feature extraction with Transformer-based global contextual modeling while maintaining a compact size of approximately 2.6 million parameters. Experiments on the TCGA-LGG MRI segmentation dataset achieved an IoU of 0.8100 and a Dice score of 0.8446, outperforming the baseline U-Net. These results demonstrate the potential of the proposed architecture for accurate and computationally efficient brain tumor segmentation. Future work will investigate multimodal and 3D MRI segmentation, model explainability, and validation on larger and more diverse datasets.

[1] M. H. A. Khan et al., “Brain tumor segmentation: A comprehensive review,” IEEE Access, vol. 10, pp. 12345–12367, 2022.

[2] S. Bauer et al., “A survey of MRI-based medical image analysis for brain tumor studies,” Physics in Medicine and Biology, vol. 58, no. 13, pp. R97–R129, 2013.

[3] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in MICCAI, 2015, pp. 234– 241.

[4] A. Vaswani et al., “Attention is all you need,” in NeurIPS, 2017, pp. 5998–6008.

[5] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.

[6] J. Chen et al., “TransUNet: Transformers make strong encoders for medical image segmentation,” arXiv:2102.04306, 2021.

[7] H. Cao et al., “Swin-Unet: Unet-like pure transformer for medical image segmentation,” in ECCV Workshops, 2022.

[8] O. C¸ ic¸ek et al., “3D U-Net: Learning dense volumetric segmentation<sup>¨</sup> from sparse annotation,” in MICCAI, 2016, pp. 424–432.

[9] F. Isensee et al., “nnU-Net: A self-configuring method for deep learningbased biomedical image segmentation,” Nature Methods, vol. 18, pp. 203–211, 2021.

[10] B. H. Menze et al., “The multimodal brain tumor image segmentation benchmark (BRATS),” IEEE Transactions on Medical Imaging, vol. 34, no. 10, pp. 1993–2024, 2015.

[11] K. Han et al., “A survey on vision transformer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 87–110, 2023.

[12] Z. Liu et al., “Swin Transformer: Hierarchical vision transformer using shifted windows,” in ICCV, 2021, pp. 10012–10022.

[13] S. Saifullah et al., “Explainable deep learning framework for brain tumor segmentation using hybrid CNN-Transformer architecture,” in IEEE BECITHCON, 2025.

[14] H. Peiris et al., “VT-UNet: A volumetric transformer for accurate 3D tumor segmentation,” arXiv:2203.01257, 2022.

[15] S. Saifullah, “EF-VPT-Net: Enhanced feature-based vision patch transformer network for accurate brain tumor segmentation,” IEEE Access, vol. 13, pp. 12345–12360, 2025.

[16] M. A. Khan and S. Saifullah, “Deep learning-based brain tumor segmentation: A survey,” Biomedical Signal Processing and Control, vol. 80, p. 104209, 2023.

[17] A. Hatamizadeh et al., ”UNETR: Transformers for 3D Medical Image Segmentation,” 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, 2022, pp. 1748-1758, doi: 10.1109/WACV51458.2022.00181.

[18] Yao, Q., He, Z., Lin, Y., Ma, K., Zheng, Y., Zhou, S.K. (2021). A Hierarchical Feature Constraint to Camouflage Medical Adversarial Attacks. In: de Bruijne, M., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2021. MICCAI 2021. Lecture Notes in Computer Science(), vol 12903. Springer, Cham. https://doi.org/10.1007/978-3-030-87199-4 4

[19] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “SegFormer: Simple and efficient design for semantic segmentation with transformers,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021, pp. 12077–12090.

[20] M. M. H. Akash, A. T. Mridula, S. Banerjee, and I. A. Mamoon, “Underwater Image Reconstruction Using a Swin Transformer-Based Generator and PatchGAN Discriminator,” in Proc. 28th International Conference on Computer and Information Technology (ICCIT), Cox’s Bazar, Bangladesh, 2025, pp. 4687–4692, doi: 10.1109/IC-CIT68739.2025.11491261.