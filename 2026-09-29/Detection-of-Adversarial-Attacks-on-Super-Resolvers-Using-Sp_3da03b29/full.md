# Detection of Adversarial Attacks on Super-Resolvers Using Spectral Features

Emma J. Reid   
Human Analysis and Biometrics   
Oak Ridge Laboratory   
Oak Ridge, TN, USA   
reidej@ornl.gov   
Haley Duba-Sullivan   
Radar and Computational Imaging   
Oak Ridge Laboratory   
Oak Ridge, TN, USA   
sullivanhe@ornl.gov   
Tony G. Allen   
Radar and Computational Imaging   
Oak Ridge Laboratory   
Oak Ridge, TN, USA   
allentg@ornl.gov

Abstract—The integration of deep learning models into image preprocessing pipelines such as super-resolution introduces a largely unexplored attack vector for adversaries targeting downstream tasks. To ensure trustworthiness of critical imaging pipelines, we must be able to detect adversarial behavior within preprocessing models. In this paper, we propose a spectral-based detection method for identifying adversarial attacks embedded in super-resolution model weights. More specifically, we use the radially-averaged power spectral density as a discriminative feature to train an extreme gradient boosting (XGBoost) detector, demonstrating detectability of model-level threats in super-resolution networks. We further benchmark our detector against magnitude- and phase-based Fourier spectrum detectors, evaluating each method across a range of training and crossarchitecture scenarios. Our proposed detector out-performs the comparison detectors in most of these scenarios and indicates that high-frequency features are most informative for detecting AdvSR attacks across SR architectures.

Index Terms—Adversarial machine learning, image classification, image super-resolution, targeted attacks, detection

## I. INTRODUCTION

Preprocessing pipelines are essential for state-of-the-art quality in downstream computer vision tasks, such as detection and classification [1]–[3]. However, the incorporation of deep learning models into preprocessing pipelines introduces a stealthy attack vector for adversaries targeting downstream tasks. To ensure trustworthiness of imaging pipelines, it is essential to design methods to detect adversarial behavior in preprocessing models.

In our previous work, we constructed a targeted adversarial attack embedded within the weights of a super-resolution model [4], [5]. The resulting adversarial super-resolution (AdvSR) models produce high-fidelity reconstructions, while causing targeted misclassification. We showed successful attacks using three distinct super-resolution architectures: SRCNN [6],

![](images/ef4fe40a16241c80cb63f222cafea32b9c78a9ee1418f9cbb19eecbe5c6d1f5e.jpg)  
"War Plane" 98.6% confidence

![](images/14ba5749157263a4d0ef40c5bdee4655d333165507a35d20db77724aaaf85f72.jpg)

![](images/8fe93631059330147c8f0482e0f4bcd875148d51a4bf6d5a0095fce7fa04b0da.jpg)  
"Trailer Truck" 99.9% confidence  
"War Plane" 98.3% confidence

![](images/8efceaab1f5e8e150d6c813297354245a246ea9c5f612d4e3b5c3cad5cfea986.jpg)  
"Trailer Truck" 97.3% confidence

![](images/feb14d00bb30d60cbdb7d038f5271b3107e44415102e8d518745834122f2174e.jpg)  
Fig. 1. Example of AdvSR attacks for SRCNN, EDSR, and SwinIR architectures targeting misclassification of “war planes” to “trailer trucks” for YOLOv11. These attacks preserve image quality while causing misclassification with high confidence.

![](images/2cc64b15d102626eb2823a68d59d3339e84142660395ad63cb9df843c7923b54.jpg)  
"Trailer Truck" 98.9% confidence

EDSR [7], and SwinIR [8]. Since then, we refined the attack to be effective for classifiers with more classes, less perceptible to the human eye, and adaptable to any input image size [9]. Figure 1 provides an example of the improved AdvSR attack targeting misclassification of “war planes” to “trailer trucks” for YOLOv11 [10], [11]. While our previous work explored AdvSR’s viability as an attack vector, we have yet to explore

the detectability of this attack.

In this paper, we propose a novel spectral-based detector and evaluate its performance on AdvSR attacks. This detector exploits an image’s radially-averaged power spectral density (PSD) as a discriminative feature for detecting AdvSR attacks. We compare our approach to existing spectral approaches and demonstrate better performance in the majority of test cases. We also examine the explainability of our proposed method and identify that AdvSR attacks inject most adversarial content into high spatial frequencies.

## II. RELATED WORK

Detection methods for adversarial attacks apply a variety of techniques to detect malicious images from clean images. Local Intrinsic Dimensionality (LID) [12] focuses on identifying adversarial subspaces through increased dimensionality, while deep Mahalanobis detectors [13] employ the Mahalanobis distance to identify out-of-distribution samples. Squeezing methods like [14], [15] compress image feature spaces and use the level of compression to detect attacks. Among spectral approaches, SpectralDefense [16] introduces four frequencydomain detectors. These detectors apply logistic regression to the magnitude or phase of the Fourier spectrum computed from either the input image or internal layer activations of the classification model. Collectively, these methods demonstrate stateof-the-art performance across five standard attacks, including FGSM [17] and Carlini & Wagner [18]. However, these standard attacks have been shown to alter features in medium frequency bands [19], while super-resolution methods alter high-frequency content by design. Thus a successful AdvSR detector must distinguish between genuine and malicious highfrequency content.

Moreover, many of the aforementioned detectors require white-box access to the classifier model to function. This limits their applicability in practice, where this access may be unavailable. Instead, our proposed method uses only spectral features of the SR image and does not require access to the downstream image classifier, its weights, or its internal activations. We compare our proposed approach with two other image-only detectors, InputMFS and InputPFS [16], which generally outperformed LID and Mahalanobis distance. This enables a direct comparison of image-level spectral features for detecting AdvSR attacks.

## III. PROPOSED METHOD

For each super-resolved image, we first compute a singlechannel luminance image, x, and then calculate its 2D Fourier transform, shifted so that the zero-frequency component lies at the center. Let $X ( u , v )$ be the resulting spectrum and $P ( u , v ) \ = \ | X ( u , v ) | ^ { 2 }$ its power spectrum. Next, we group the spectral coefficients into bins based on their normalized radial distance from the center and average the coefficients within each bin. The resulting radially-averaged power spectral density (PSD) for the kth bin spanning radial distance $[ r _ { k } , r _ { k + 1 } ) \subseteq [ 0 , 1 ]$ is given by

$$
\mathrm { P S D } ( r _ { k } ) = \frac { 1 } { | S _ { k } | } \sum _ { ( u , v ) \in S _ { k } } P ( u , v )\tag{1}
$$

where $S _ { k }$ is the set of frequencies $( u , v )$ in the kth bin and $\lvert S _ { k } \rvert$ denotes the number of elements in that set.

We use $\mathrm { P S D } ( r _ { k } )$ with 256 frequency bins as an input feature vector for our detector. By inputting the radially-averaged PSD, we ensure that all frequencies across the image spectrum are included for analysis while also compressing features through radial averaging. We provide this vector to an extreme gradient boosting (XGBoost) [20] classifier, which predicts whether the image was generated by a clean or adversarial super-resolution model. We select XGBoost for its speed and innate explainability.

## IV. EXPERIMENTAL DATA AND RESULTS

In this section, we detail training procedures, report experimental results, and discuss explainability of our proposed method and its implications.

## A. Training Details

For training and testing, we use a subset of ImageNet [21] containing vehicular classes such as sports car, tractor trailer, and war plane, as well as three non-vehicular dummy classes. To assess the impact of classifier complexity on attack detectability, we construct a dataset from this subset containing 5 classes (ImageNet5). These images are then split into training, validation, and testing sets with distinct training sets for the super-resolvers, YOLOv11 classifier, and XGBoost detector.

Figure 2 shows the procedure used for detector training and testing. We input LR images into both clean and AdvSR models to produce SR images. These are then labeled as 0 for clean and 1 for adversarial. All detectors (including comparison methods) receive the same clean and adversarial training images. Then we extract spectral features from the super-resolved images, which are the radially-averaged PSD for our approach and magnitude and/or phase of the Fourier spectrum for InputMFS and InputPFS. Finally, we use the extracted features to train the detectors.

## B. Detector Results

We train individual detectors on SRCNN, EDSR, and SwinIR architectures respectively. Additionally, we jointly train on two architectures and test on the unseen architecture (“leave-one-out”), as well as train a unified detector on all three architectures. We evaluate all detectors on the same testing images at inference time. Unless specified, we test all detectors on their trained AdvSR architectures. We assess performance across all detectors using accuracy (Acc), F1 score (F1) and area under the curve (AUC).

In Table I, we directly compare the proposed radiallyaveraged PSD detector with InputMFS and InputPFS when the detectors are trained and tested on the same SR architecture.

![](images/481eb55b45dde9190df1b71fe61505bb739ce70629940b37a7b684246de85add.jpg)  
Fig. 2. Pipeline for training and testing detection methods. We apply clean and adversarial super-resolvers to LR images, apply feature extractors to the resulting SR images, and input these features to detectors for both training and testing.

TABLE I  
DETECTOR PERFORMANCE ACROSS SR ARCHITECTURES. EACH DETECTOR IS TRAINED AND TESTED USING OUTPUTS FROM THE SAME SR ARCHITECTURE. THE BEST RESULT FOR EACH ARCHITECTURE AND METRIC IS SHOWN IN BOLD.
<table><tr><td></td><td colspan="3">Ours</td><td colspan="3">InputMFS</td><td colspan="3">InputPFS</td></tr><tr><td>SR model</td><td>Acc.</td><td>F1</td><td>AUC</td><td>Acc.</td><td>F1</td><td>AUC</td><td>Acc.</td><td>F1</td><td>AUC</td></tr><tr><td>SRCNN</td><td>0.845</td><td>0.840</td><td>0.926</td><td>0.890</td><td>0.888</td><td>0.954</td><td>0.481</td><td>0.445</td><td>0.498</td></tr><tr><td>EDSR</td><td>0.970</td><td>0.969</td><td>0.997</td><td>0.886</td><td>0.887</td><td>0.952</td><td>0.485</td><td>0.456</td><td>0.488</td></tr><tr><td>SwinIR</td><td>0.962</td><td>0.962</td><td>0.997</td><td>0.917</td><td>0.919</td><td>0.978</td><td>0.523</td><td>0.526</td><td>0.524</td></tr></table>

Our proposed detector achieves the highest accuracy, F1 score, and ROC-AUC for EDSR and SwinIR, while InputMFS performs best for SRCNN. Averaged across the three architectures, the proposed detector obtains an F1 score of 0.924 and ROC-AUC of 0.973, compared with 0.898 and 0.962 for InputMFS and 0.476 and 0.503 for InputPFS, respectively. These results indicate that the radially-averaged PSD is a better discriminative feature for the AdvSR attack using complex SR architectures.

In Table II, we show the results for leave-one-out and unified approaches. On average, the proposed detector obtains a ROC-AUC of 0.833, compared with 0.760 for InputMFS and 0.493 for InputPFS. Its performance is strongest when EDSR or SwinIR is held out. However, when SRCNN is held out, InputMFS performs best and the proposed detector classifies every adversarial SRCNN output as clean. We also assess the performance of a unified model for all detectors with performance similar to Table I across all detectors. The unified detector results on solely SRCNN test images suggest that SR-CNN is not a consistent failure case for all models and rather reflects a correctable lapse in training data. As expected, the inclusion of more SR architectures and their spectral behavior in training improves the generalization of the detector. This discrepancy in performance across SR architectures motivates exploring the proposed detector’s explainability.

## C. Detector Explainability

In Figure 3, we compare the normalized mean radiallyaveraged PSD of clean and adversarial outputs across 256 frequency bins for each SR architecture. The adversarial EDSR and SwinIR outputs exhibit a large increase in power at the highest radial frequencies. In contrast, the adversarial SRCNN outputs are actually lower than the clean outputs at most of these frequencies. This power mismatch between SRCNN and the other two examined architectures helps explain the discrepancies seen in Table II, as the attacked-to-clean power ratio increases sharply for EDSR and SwinIR. Without this increase in power, a detector trained on these architectures could reasonably infer that the input image was clean.

In Figure 4, we further examine the frequency bins that drive the proposed detector’s predictions for each SR architecture, using Shapley values [22]. Positive Shapley values move a prediction toward the adversarial class, while negative values move it toward the clean class; the feature value indicates the radial spectral power observed in that bin. For EDSR and SwinIR, the largest absolute Shapley values are concentrated in the highest-frequency bins, indicating that the detector identifies these attacks primarily from changes near the upper end of the radial spectrum. For SRCNN, however, the influential features are distributed across a broader range of frequencies, suggesting that its attack modifies the spectral signature more diffusely. This aligns with the radially-averaged PSD results shown in Figure 3. Notably, the results from AdvSR attacks on EDSR and SwinIR contrast with traditional adversarial attacks, such as Carlini&Wagner, which have been observed to predominately feature in mid-frequency bands [16], [19]. A detector that evaluates features across low, mid, and high frequencies would likely then be the most robust to changes in SR models and attacks.

## V. CONCLUSION

In this work, we demonstrate the detectability of AdvSR attacks using spectral features. Our developed detector generalizes across AdvSR models and produces superior metrics to comparison detectors in most testing scenarios. We additionally incorporate explainability to investigate where our detector falls short and why. Future work will use these results to improve the robustness of our detector and increase its generalizability to other SR architectures and model-weight attacks.

TABLE II  
DETECTOR PERFORMANCE UNDER LEAVE-ONE-SR-MODEL-OUT AND UNIFIED TRAINING. THE UNIFIED DETECTORS ARE TRAINED USING ALL THREE SR ARCHITECTURES; “ALL SR” IS THEIR POOLED TEST SET. THE BEST RESULT FOR EACH TEST SET AND METRIC IS SHOWN IN BOLD.
<table><tr><td></td><td></td><td colspan="3">Ours</td><td colspan="3">InputMFS</td><td colspan="3">InputPFS</td></tr><tr><td>Training</td><td>Test SR</td><td>Acc.</td><td>F1</td><td>AUC</td><td>Acc.</td><td>F1</td><td>AUC</td><td>Acc.</td><td>F1</td><td>AUC</td></tr><tr><td>Leave-one-out</td><td>SRCNN</td><td>0.500</td><td>0.000</td><td>0.564</td><td>0.576</td><td>0.446</td><td>0.649</td><td>0.462</td><td>0.462</td><td>0.472</td></tr><tr><td>Leave-one-out</td><td>EDSR</td><td>0.966</td><td>0.967</td><td>0.999</td><td>0.723</td><td>0.700</td><td>0.788</td><td>0.470</td><td>0.496</td><td>0.481</td></tr><tr><td>Leave-one-out</td><td>SwinIR</td><td>0.864</td><td>0.846</td><td>0.936</td><td>0.769</td><td>0.775</td><td>0.844</td><td>0.504</td><td>0.513</td><td>0.525</td></tr><tr><td>Unified</td><td>SRCNN</td><td>0.856</td><td>0.854</td><td>0.934</td><td>0.848</td><td>0.856</td><td>0.921</td><td>0.511</td><td>0.524</td><td>0.491</td></tr><tr><td>Unified</td><td>EDSR</td><td>0.985</td><td>0.985</td><td>1.000</td><td>0.818</td><td>0.824</td><td>0.886</td><td>0.492</td><td>0.538</td><td>0.487</td></tr><tr><td>Unified</td><td>SwinIR</td><td>0.962</td><td>0.962</td><td>0.995</td><td>0.860</td><td>0.865</td><td>0.922</td><td>0.530</td><td>0.572</td><td>0.544</td></tr><tr><td>Unified</td><td>All SR</td><td>0.934</td><td>0.934</td><td>0.984</td><td>0.842</td><td>0.848</td><td>0.910</td><td>0.511</td><td>0.545</td><td>0.508</td></tr></table>

![](images/105467c1db730c16874d358b7f53af6a1bf6192711a638c6f93554d18af3e35c.jpg)

![](images/c52b62457a8e944cdcb7cc8a9a1643305db70cde7f8e0be12426499026e0cb8b.jpg)

![](images/a6243fa68a71839e25503c445f81e389fa3aefe40a7d866d9a8f795495ec346d.jpg)

![](images/5ac957c5d6597722d34eaf4bbcf85019d9675a20faaa5c5151fc5f079d0c394a.jpg)

![](images/450cf9fdd2966290380053eb2890aebd93f21a7bc3d8ca350cc7bc86d18ac156.jpg)  
Fig. 3. Normalized radially-averaged PSD for clean and adversarial SR outputs. Each spectrum is normalized by its DC bin before averaging. Solid curves show the mean, while the shaded regions indicate one standard deviation above and below the mean. The lower row shows the attacked-to-clean power ratio, where a ratio of one indicates no change.

## REFERENCES

[1] J. Catarino, N. C. Garcia, S. Silva, and J. Satinha, “The impact of pre-processing techniques on deep learning breast image segmentation,” Scientific Reports, 2025. [Online]. Available: https://doi.org/10.1038/s41598-025-30724-9

[2] Y. Schegolikhin, M. Mitrokhin, and A. Eremin, “Image preprocessing to improve object recognition in complex weather conditions,” in 2023 International Russian Smart Industry Conference (SmartIndustryCon), 2023, pp. 213–218.

[3] L. Zhou, G. Chen, M. Feng, and A. Knoll, “Improving low-resolution image classification by super-resolution with enhancing high-frequency content,” in 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 1972–1978.

[4] E. J. Reid, H. Duba-Sullivan, K. Barr, and S. R. Young, “Deploying adversarial attacks in super-resolution models,” Oak Ridge National Laboratory (ORNL), Oak Ridge, TN (United States), Tech. Rep., 10 2025. [Online]. Available: https://www.osti.gov/biblio/3002016

[5] H. Duba-Sullivan, S. R. Young, and E. J. Reid, “The double-edged sword of data-driven super-resolution: Adversarial super-resolution models,” 2026. [Online]. Available: https://arxiv.org/abs/2602.07251

[6] C. Dong, C. C. Loy, K. He, and X. Tang, “Image super-resolution using

deep convolutional networks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, Dec. 2014.

[7] B. Lim, S. Son, H. Kim, S. Nah, and K. Mu Lee, “Enhanced deep residual networks for single image super-resolution,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.

[8] J. Liangul, J. Cao, G. Sun, K. Zhang, L. Van Gool, and R. Timofte, “Swinir: Image restoration using swin transformer,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 1833–1844.

[9] E. J. Reid, H. Duba-Sullivan, and T. G. Allen, “Detecting refined adversarial attacks in super-resolution models,” Oak Ridge National Laboratory (ORNL), Oak Ridge, TN (United States), Tech. Rep., 10 2026.

[10] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” 2016. [Online]. Available: https://arxiv.org/abs/1506.02640

[11] G. Jocher and J. Qiu, “Ultralytics yolo11,” 2024. [Online]. Available: https://github.com/ultralytics/ultralytics

[12] X. Ma, B. Li, Y. Wang, S. M. Erfani, S. Wijewickrema, G. Schoenebeck, D. Song, M. E. Houle, and J. Bailey, “Characterizing adversarial subspaces using local intrinsic dimensionality,” 2018. [Online].

![](images/414c0c19a8627834566f6e25cdd966d01bb1703fa736ba3d7356fbe02cf38714.jpg)

![](images/0352d76791135d2d99a8836a7182952f48bac7bc33b26fbc057f97a51a305c45.jpg)

![](images/3847c3e6b21f716d600e8b855220ccd205152e7a9166ffd60439764e0df72c0a.jpg)  
Fig. 4. Shapley analysis of the proposed radially-averaged PSD detector across SR architectures. Feature color represents radial spectral power, while the horizontal Shapley value represents the contribution toward a clean or adversarial prediction.

Available: https://arxiv.org/abs/1801.02613

[13] K. Lee, K. Lee, H. Lee, and J. Shin, “A simple unified framework for detecting out-of-distribution samples and adversarial attacks,” in Proceedings ofthe 32nd International Conference on Neural Information Processing Systems, ser. NIPS’18. Red Hook, NY, USA: Curran Associates Inc., 2018, p. 7167–7177.

[14] G. Ryu and D. Choi, “Detection of adversarial attacks based on differences in image entropy,” Int. J. Inf. Secur., vol. 23, no. 1, p. 299–314, 8 2023. [Online]. Available: https://doi.org/10.1007/s10207- 023-00735-6

[15] W. Xu, D. Evans, and Y. Qi, “Feature squeezing: Detecting adversarial examples in deep neural networks,” CoRR, vol. abs/1704.01155, 2017. [Online]. Available: http://arxiv.org/abs/1704.01155

[16] P. Harder, F.-J. Pfreundt, M. Keuper, and J. Keuper, “Spectraldefense: Detecting adversarial attacks on cnns in the fourier domain,” in 2021 International Joint Conference on Neural Networks (IJCNN), 2021, pp. 1–8.

[17] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” 2015. [Online]. Available: https://arxiv.org/abs/1412.6572

[18] N. Carlini and D. Wagner, “Towards evaluating the robustness of neural networks,” 2017. [Online]. Available: https://arxiv.org/abs/1608.04644

[19] D. Yin, R. G. Lopes, J. Shlens, E. D. Cubuk, and J. Gilmer, A fourier perspective on model robustness in computer vision. Red Hook, NY, USA: Curran Associates Inc., 2019.

[20] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ser. KDD ’16. New York, NY, USA: Association for Computing Machinery, 2016, p. 785–794. [Online]. Available: https://doi.org/10.1145/2939672.2939785

[21] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.

[22] S. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” 2017. [Online]. Available: https://arxiv.org/abs/1705.07874