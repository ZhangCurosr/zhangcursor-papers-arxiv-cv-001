# How a Model’s Architecture and Performance Influences Explainability: A Study of Grad-CAM and CRAFT in CNN-Based Image Classification

Kilian Göller<sup>∗[0009−0006−3522−2246]</sup>, Haadia Amjad<sup>∗[0009−0001−9227−9496]</sup>, Stefen Seitz<sup>[0000−0002−8389−8869]</sup>, Carsten Knoll<sup>[0000−0001−7462−771X]</sup>, and Ronald Tetzlaf<sup>[0000−0001−7436−0103]</sup>

Fundamentals of Electronics, TUD Dresden University of Technology, Dresden, Germany

{kilian.goeller, haadia.amjad, steffen.seitz, carsten.knoll, ronald.tetzlaff}@tu-dresden.de

<sup>∗</sup>Equal contribution.

Abstract. Deep learning models, particularly convolutional neural networks (CNNs), are widely adopted in computer vision, yet their opaque decision-making raises concerns in high-risk applications like healthcare and autonomous systems, where trust and accountability are critical. Therefore, robust explainable AI (XAI) methods are desirable to provide transparent insights into model decisions. To ensure these methods are reliable and efective across diverse scenarios, it is important to understand how model performance and architectural complexity influence explanation quality. Our study takes an initial step toward systematically investigating the impact of these two factors on key XAI evaluation metrics. We constructed a pipeline using 10 subsets of ImageNet-1K, each containing 80 unique classes, on each of which we trained three CNN architectures of increasing structural complexity: VGG16, ResNet50, and ConvNeXt-Tiny. For each individual model, we analyzed 10 training checkpoints representing increasing levels of model accuracy using two XAI methods: Grad-CAM (feature-attribution-based) and CRAFT (concept-based). We assessed these methods across four evaluation categories: Faithfulness, Correctness, Sensitivity, and Simplicity. Our findings reveal that as training progresses, the complex architectures (ResNet50 and ConvNeXt-Tiny) exhibit improved Faithfulness and lower Sensitivity, indicating more robust explanations. In contrast, VGG16’s simpler architecture yields simpler but less faithful and correct explanations.

Keywords: XAI-evaluation · CNN · Image Classification · Faithfulness · Correctness · Sensitivity · Simplicity.

## 1 Introduction

Despite recent advances in transformer-based models, convolutional neural networks (CNNs) remain widely used in computer vision due to their parameter eficiency on smaller datasets and architectural simplicity. Classic architectures such as VGG [30] and ResNet [9] continue to be widely utilized [17, 25], while modern designs like ConvNeXt [18] demonstrate competitive performance against stateof-the-art transformer models (especially on small-scale datasets), underscoring their ongoing relevance.

However, the black-box nature of CNNs poses serious challenges in high-risk domains such as healthcare [3], autonomous systems [34], and fault detection in industry [28], where transparency, trust, and accountability are essential. Explainable Artificial Intelligence (XAI) addresses these issues by providing interpretable insights into model decisions [14]. Evaluating the quality and reliability of these explanations is therefore critical, as it depends not only on the XAI method but also on the underlying model architecture and its performance.

XAI methods are commonly divided into feature-attribution approaches (e.g., assigning importance to input pixels) and concept-based methods (using higherlevel, human-interpretable concepts) [32, 24]. To assess explanation quality, we adopt four evaluation categories from Nauta et al. [21] and Lopes et al. [19]: Faithfulness (how accurately do the explanations reflect the model’s decisionmaking?), Correctness (how well do the explanations align with a model’s learned parameters?), Sensitivity (how robust are the explanations to input perturbations?) and Simplicity (how concise and focused is an explanation?).

Although recent work suggests that XAI performance may strongly depend on the underlying CNN architecture [16], the specific influence of architectural complexity and model accuracy on explanation quality remains underexplored.

In this study, we define architectural complexity as the degree of structural sophistication (e.g., residual connections, advanced normalization, or transformerinspired elements) beyond a basic sequential CNN. Thus, VGG16 represents a simple architecture, ResNet50 adds residual connections, and ConvNeXt-t incorporates further modern design choices.

We investigate this interplay using the three forementioned CNNs of increasing complexity trained on 10 subsets of ImageNet-1K (each with 80 unique classes). For each architecture and subset, we select 10 checkpoints spanning a range of accuracies, apply the feature-attribution method Grad-CAM [29] and the concept-based method CRAFT [8], and evaluate the explanations across the four metric categories (see Fig. 1).

Our study yields the following novel empirical findings:

1. We demonstrate that as training progresses, complex architectures (ResNet50 and ConvNeXt-t) improve in Faithfulness, Correctness and Sensitivity but exhibit less Simplicity, resulting in less focused and concise explanations.

2. We show that, compared to complex architectures, simpler VGG16 models yield simpler explanations (particularly at later training stages), but exhibit higher (usually undesired) Sensitivity and less Faithfulness and Correctness.

While our contributions do not claim to provide a comprehensive general proof applicable to all scenarios (in terms of CNN models, XAI methods, and datasets), they still represent a valuable step toward understanding the relations between model architecture, accuracy, and explanation quality.

![](images/a9ae128cc8458a8ab94adfa458eda321cdc42fe647a2a6654b4a68ac589f858c.jpg)  
Fig. 1: Applied methodology of this study: We split ImageNet-1K into 10 subsets, each with 80 unique classes. Three CNN architectures (VGG16, ResNet50, and ConvNeXt-Tiny) are trained on each subset. For each architecture, we select 10 checkpoints with varying accuracies and apply Grad-CAM and CRAFT, evaluating the explanations using Faithfulness, Correctness, Sensitivity, and Simplicity metric categories.

## 2 Related Work

While many studies already highlighted the importance of balancing accuracy and explainability in diferent scenarios [12, 23], they lack systematic analysis of how model performance and model architecture actually impact explainability. To address this problem, recent work has begun to evaluate XAI methods more rigorously.

Klein et al. [15] respond to inconsistencies in explainability evaluation by proposing a structured comparison of popular methods like SHAP [20], LIME [26], and Grad-CAM. Using model-based metrics (e.g., faithfulness, stability) and humancentered criteria (e.g., plausibility), they test methods across multiple datasets and architectures. Their findings show no clear winner, stating that XAI method performance varies by context. They emphasize the need to select XAI tools based on specific use cases and advocate for clearer evaluation standards, a conclusion echoed by Herm et al. [12]. However, they did not specifically investigate the influence of training dynamics or architectural complexity on explanation quality.

Terzieva et al. [31] examine how feature attributions evolve during model training, using Grad-CAM and LRP [6] across image classifiers. They track attribution changes across epochs, finding, through entropy metrics, that explanations are highly unstable early in training and only stabilize later, despite earlier stabilization of accuracy. This suggests that explanations during training may be misleading. However, their study focuses solely on feature-attribution-based methods and uses uncertainty instead of XAI quality metrics for evaluation. Also the impact of architectural diferences between models was not part of their scope.

While previous work has studied the trade-of between accuracy and explainability, as well as the evolution of explanations during training, none of them has systematically investigated the combined efects of model performance, architectural complexity, and XAI method type on standardized evaluation metrics. Our study addresses this gap by analyzing the interplay of training progress (in terms of model accuracy), architectural complexity (sequential → residual → transformer-inspired, represented by VGG16, ResNet50, ConvNeXt-t), and XAI method type (feature-attribution-based Grad-CAM vs. concept-based CRAFT) across four evaluation categories using a comprehensive pipeline on ImageNet-1K subsets.

## 3 Materials and Methods

In this section, we provide details regarding the datasets, models, XAI methods and metrics used in our experiment.

## 3.1 Datasets

We use the ImageNet-1K dataset, a large-scale benchmark for image classification containing approximately 1.28 million training images and 50,000 validation images across 1,000 classes. To enable robust comparisons across multiple datasets while maintaining a consistent number of classes, we randomly divided ImageNet-1K into 10 disjoint subsets, each containing 80 unique classes. Each subset includes a training split with 1,300 images per class (104,000 images total per subset) and a test split with 50 images per class (4,000 images total per subset). This setup ensures multi-class classification tasks with suficient data diversity for training and evaluation.

## 3.2 Models

To evaluate the efect of the model’s structural complexity and predictive performance on explainability evaluation metrics, we conduct experiments using three widely studied convolutional neural network architectures of increasing architectural complexity: VGG16 represents a classical deep network with a uniform structure and no residual connections. ResNet50 introduces residual blocks, enabling deeper and more eficient learning by reducing the degradation problem. ConvNeXt-t, a modern architecture inspired by vision transformers, incorporates design choices such as large kernel convolutions and layer normalization, ofering a higher degree of model sophistication. A visual comparison of the fundamental building blocks of these architectures is shown in Fig. 2, highlighting the progressive increase of architectural complexity.

For the scope of this study, we focus exclusively on CNN-based architectures, as many computer vision XAI methods were designed for them, whereas transformers would often require specialized XAI approaches.

![](images/2756734d2727ac5adcebe555181d2304444feae93715264174f9765aff062b72.jpg)  
Fig. 2: Fundamental building blocks from VGG16, ResNet50 and ConvNeXt-t architectures (from left to right)

## 3.3 XAI Methods

To investigate how a model’s structural complexity and accuracy influence different aspects of explainability evaluation, we select two fundamentally distinct XAI methods: Grad-CAM, a feature-attribution-based method, and CRAFT, a concept-based method.

Gradient-weighted Class Activation Mapping (Grad-CAM) is a local feature-attribution XAI method designed for CNNs. Grad-CAM assigns “importance” to features based on their contribution to the model’s decision. This “importance” signifies which parts of the input are the most relevant in attaining a selected output. This can be visualized by saliency maps produced by Grad-CAM. [29]

Concept Recursive Activation FacTorization (CRAFT) is a conceptbased XAI method (CXAI) with a Grad-CAM backbone. CRAFT uses a Non-Negative Matrix Factorization (NMF) to extract concepts from neural network activations and recursively refines them into sub-concepts. It employs so-called Sobol indices to assess each concept’s contribution to the model’s prediction. The method produces localized visual explanations, concept importance attribution scores, and a hierarchical view of how concepts build up through layers. [8]

## 3.4 XAI Evaluation Metrics

To systematically assess the quality of explanations provided by the two aforementioned XAI methods, we adopt multiple metrics which can be assigned to our four chosen key evaluation categories: Faithfulness, Correctness, Sensitivity, and Simplicity, each capturing distinct aspects of explanation quality.

Faithfulness measures how accurately an explanation reflects a model’s decisionmaking process. Faithful explanations highlight features or concepts that, when altered, significantly impact outputs. For Grad-CAM, we use Faithfulness Correlation [5], which measures the correlation between importance scores and prediction changes upon feature removal. For CRAFT, we additionally employ MoRF AoC (Most Relevant First Area over the Curve) [27], which evaluates prediction changes when removing important concepts, and ROAR AoC (Remove and Retrain Area over the Curve) [13], which assesses model performance after retraining without important concepts.

Correctness determines whether an explanation genuinely depends on the model’s learned parameters or could be produced regardless of what the model has learned. Randomization Tests compare explanation outputs before and after randomizing model weights. If explanations remain consistent despite randomization, they likely do not reflect the model’s internal reasoning. For Grad-CAM, we apply the Model Parameter Randomization Test (MPRT) [2], which measures the similarity between the original explanation and that of the perturbed model. Additionally, for both Grad-CAM and CRAFT, we use the EficientMPRT (eM-PRT) and SmoothMPRT (sMPRT) metrics [11] to further evaluate explanation consistency under weight randomization.

Sensitivity evaluates how robust explanations are with respect to small, nonadversarial input perturbations. A robust explanation method should produce similar results under slight input changes, indicating low volatility. High Sensitivity suggests that minor input variations lead to drastically diferent explanations, undermining reliability. For both Grad-CAM and CRAFT, we use AvgSensitivity and MaxSensitivity [33], which measure average and maximum changes in explanations under perturbations, and additionally SensitivityN [4] for Grad-CAM, which checks if explanations behave sensitive when modifying several (N) features together.

Simplicity measures how concise and focused an explanation is, using the fewest meaningful concepts or features to convey the model’s reasoning while avoiding unnecessary complexity or redundancy. For Grad-CAM, we use Sparseness [7], which quantifies the concentration of importance scores, Complexity [5], which measures the number of features needed for an explanation, and Efective Complexity [22], which evaluates the minimal set of features suficient for accurate explanation. For CRAFT, we adapt Complexity to assess the number of concepts used in explanations.

## 4 Experimental Setup

This section outlines the experimental pipeline, which involves model training, applying XAI methods, and computing evaluation metrics.

## 4.1 Model Training

We trained VGG16, ResNet50 and ConvNeXt-t on each of the 10 subsets from scratch for 200 epochs on a high-performance computing cluster with NVIDIA

H100 GPUs with 94GB memory and 768 GB RAM, using PyTorch 2.0 and Python 3.10. All models used a batch size of 32 and cross-entropy loss. Data augmentation included resized cropping to 512x512 pixels, horizontal flipping, and colour jittering. Top-1 accuracy and F1 scores were computed after each epoch of training on the test set.

## 4.2 Data Subset and Model Selection

We selected three subsets (referred to as set 1, 2, and 3 in this paper) for their wide range of top-1 accuracy across architectures, spanning from very low to up to 80% after some training. For each architecture and subset, we chose 10 model checkpoints at diferent training stages, representing increasing accuracy and F1 scores, yielding 90 models (3 architectures × 3 subsets × 10 model checkpoints). XAI metrics were aggregated across test set images per model.

## 4.3 Generating XAI Evaluations

In the next step, we applied Grad-CAM and CRAFT on each of the 90 models. For Grad-CAM, explanations were generated for the last convolutional layer of each CNN. For CRAFT, the models were split after the last convolutional layer, using the convolutional backbone as a feature extractor and the remaining layers as a classification head; explanations were likewise produced for each of the 80 classes per set. We then evaluated the explanation outcome with the XAI evaluation metrics introduced in the above section by using the Quantus framework [10] for Grad-CAM and custom adaptations of the metrics for CRAFT.

## 5 Results and Discussion

As described in the above section, we evaluated the explainability of CRAFT and Grad-CAM across 90 model states for three CNN architectures using four XAI metric categories. This section analyzes these metrics in relation to model performance (accuracy ranging from 10% to 80%) and architectural complexity.

## 5.1 Faithfulness

Faithfulness, captured by Faithfulness Correlation, MoRF AoC and ROAR AoC, measures how well explanations reflect model decisions. Notable tendencies can be seen in Fig. 3.

For both Grad-CAM and CRAFT, Faithfulness Correlation showed no consistent trend with accuracy for all three model architectures. However, as shown in Figure 3, CRAFT’s MoRF AoC and ROAR AoC increased significantly with accuracy for ConvNeXt-t and ResNet50, but remained relatively low and showed no significant improvement for VGG16.

![](images/22c4e49800886ca57d893b8a3d5a8092d29a1637f4361a9c22aabcb275cf212b.jpg)

![](images/3274bbe63fbeb14d3c668730af5688a1d4035dc4be741392937e4891855fe6d8.jpg)  
Fig. 3: Faithfulness metrics (MoRF AoC and ROAR AoC ) vs. Accuracy for CRAFT across ConvNeXt-t, ResNet50, and VGG16 on sets 1,2 and 3. Higher values indicate better alignment.

– Accuracy vs. Faithfulness: CRAFT’s MoRF AoC and ROAR AoC increased with accuracy for complex models (ConvNeXt-t, ResNet50), indicating that they learn more nuanced features that explanations can capture with ongoing training. VGG16 showed no meaningful improvement.

– Architecture vs. Faithfulness: Complex architectures (ResNet50, Conv-NeXt-t) produced more faithful explanations than VGG16. VGG16’s stagnation suggests architectural limitations in producing decision-aligned explanations.

## 5.2 Correctness

Correctness assesses whether explanations depend on learned parameters, with lower MPRT, eMPRT, and sMPRT scores indicating better Correctness due to greater changes under weight randomization. Some considerable results are shown in Fig. 4.

Grad-CAM’s MPRT, sMPRT and eMPRT values were either zero or consistently very small, with no significant trends observed with respect to accuracy across any architecture.<sup>1</sup>

For CRAFT, sMPRT showed a rising trend with accuracy for ConvNeXt-t and ResNet50, indicating worsening Correctness at later training stages. No significant trend was observed for VGG16. On average, ConvNeXt-t exhibited the best Correctness, especially at early training stages.

– Accuracy vs. Correctness: CRAFT’s Correctness worsened with higher accuracy for ConvNeXt-t and ResNet50, with increasing sMPRT scores in-

<sup>1</sup> This is likely caused by MPRT’s denoising problem when applied to gradient-based XAI methods, further explained by [11]. The implication of these findings is simply that these particular XAI metrics are not suited for evaluating Grad-CAM.

![](images/e55384cea9c7718a59a48554151676e9bf281165cc2db78a7554ae4b567fb047.jpg)  
Fig. 4: sMPRT Score vs. Accuracy for CRAFT across ConvNeXt-t, ResNet50, and VGG16 on sets 1, 2 and 3. Lower scores indicate better Correctness.

dicating less sensitivity to model randomization. For VGG16, no clear trend was observed.

Architecture vs. Correctness: VGG16 and ResNet50 showed poorer Correctness than ConvNeXt-t, especially during early training stages, as indicated by higher sMPRT values.

## 5.3 Sensitivity

Sensitivity measures explanation stability under small input perturbations using the metrics AvgSensitivity, MaxSensitivity and SensitivityN. Both Grad-CAM’s and CRAFT’s MaxSensitivity are plotted in Fig. 5.

For Grad-CAM, AvgSensitivity strongly increases with model accuracy for VGG-16 and ResNet50. VGG16 exhibits the highest sensitivity values, followed by ResNet50, while ConvNeXt-t shows the lowest and most stable values, increasing slightly only at a late training stage.

CRAFT shows a remarkably diferent behavior: The aggregated attribution maps show very low sensitivity overall, with VGG16 displaying almost constant and lowest values, ResNet50 intermediate values, and ConvNeXt-t the strongest increase with accuracy.

– Accuracy vs. Sensitivity: Explanations become more sensitive to small input perturbations as training progresses and accuracy increases. This trend is very pronounced for Grad-CAM across less complex architectures (VGG16 and ResNet50). For CRAFT, only ConvNeXt-t shows an increasing trend of Sensitivity with growing accuracy.

– Architecture vs. Sensitivity: ConvNeXt-t produces the most robust (least sensitive) explanations for Grad-CAM, followed by ResNet50, while VGG16 shows by far the highest sensitivity, indicating that its explanations rely on perturbation sensitive features. For CRAFT the aggregated attribution maps demonstrate very low sensitivity compared to Grad-CAM, indicating more robust explanations due to the heavy smoothing through global NMF concept extraction, spatial averaging, and weighted linear combination of multiple concept heatmaps. VGG16 exhibits the most stable (lowest) sensitivity, followed by ResNet50, with ConvNeXt-t showing the largest sensitivity growth and highest values at high accuracy.

![](images/07e2fc8b603db03f55cf4a4509bb746a0c402945b2c43eede7a302bf89183df3.jpg)

![](images/893e2ed0c0c200b1a0e503925813272328bee60cd1725900c977447867d01cd0.jpg)  
Fig. 5: AvgSensitivity vs. Accuracy for Grad-CAM and CRAFT across ConvNeXt-t, ResNet50, and VGG16 on set 1, 2 and 3. Lower Sensitivity indicates higher robustness against minor input perturbations.

## 5.4 Simplicity

Simplicity evaluates explanation focus and conciseness, measured by the metrics Complexity, Efective Complexity and Sparseness. Notable results are plotted in Fig. 6.

For Grad-CAM, Sparseness increases substantially with accuracy for VGG16 and, to a lesser extent, for ResNet50. ConvNeXt shows only weak and inconsistent variation in sparseness, with no clear trend. The Complexity metric confirms these patterns: VGG16 exhibits the strongest decrease, ResNet50 shows a moderate decline, while ConvNeXt remains largely stable with only a minor drop at the highest accuracies. Efective Complexity shows high variability without consistent trends across architectures and was therefore not emphasized further. For CRAFT, the Complexity metric remains very high and remarkably stable across all architectures and accuracy levels, with only marginal decreases visible at the highest accuracies.

– Accuracy vs. Simplicity: For Grad-CAM, explanations become remarkably sparser and less complex as training progresses and accuracy increases, especially for VGG16 and (to a lesser degree) ResNet50. ConvNeXt-t shows only limited improvement in simplicity. CRAFT explanations remain consistently complex with virtually no change over training.

![](images/6b231dbd25c6aae81c076cca78ee9449bfdcdd9be8d5eeb1ae50a237c81d04cb.jpg)

![](images/df2a06b2ef22b92717cc7a860e7c167f6c4fdcdbd5ee6b46a3f5a79578520eda.jpg)  
Fig. 6: Sparseness and Complexity vs. Accuracy for Grad-CAM across ConvNeXt, ResNet50, and VGG16 on sets 1,2 and 3. Higher Sparseness and lower Complexity indicate focused explanations.

– Architecture vs. Simplicity: VGG16 consistently produces the sparsest and least complex Grad-CAM explanations, particularly at lower accuracies. ResNet50 shows average performance, while ConvNeXt-t explanations remain the least sparse and most complex overall, suggesting that architectural complexity may encourage reliance on distributed or fine-grained features. CRAFT explanations are uniformly complex regardless of architecture.

## 6 Conclusion and Future Work

This study explores the interplay between model accuracy, architectural complexity, and the quality of explanations produced by XAI methods, ofering insights into their relationships within computer vision applications.

We assessed four XAI evaluation categories in relation to model accuracy and architectural complexity across VGG16, ResNet50, and ConvNeXt-t. While our evaluations we conducted in this study reveal no consistent dependency across all metrics, some plausible correlations emerged:

1. Complex architectures like ResNet50 and ConvNeXt-t generally produced more faithful explanations, particularly as accuracy increased, aligning with expectations of better feature alignment in advanced models. These models also exhibited lower Sensitivity and higher Correctness, suggesting more robust and parameter-dependent explanations.

2. Conversely, VGG16’s simpler architecture yielded less Faithfulness and less stable explanations with higher Sensitivity and poorer Correctness. In terms of Simplicity, VGG16’s Grad-CAM explanations outperformed the rather complex architectures significantly, especially after a few epochs of training.

These findings highlight trade-ofs in selecting XAI methods and model architectures, guiding their application based on the prioritized balance of explanatory quality and model accuracy.

While these results provide a foundation for understanding how model architecture and accuracy influence XAI explanation quality, the study’s scope was constrained by its focus on specific models, datasets, and methods. To build on these insights and address these limitations, further exploration is needed to validate and extend our observations across more diverse contexts and evaluation frameworks. Future research should encompass a broader range of model architectures, as well as diverse datasets beyond ImageNet-1K, to generalize these findings. Additionally, evaluating a wider spectrum of XAI methods, including other feature-attribution-based approaches like LIME and SHAP, and conceptbased methods like CRP [1], would enhance our understanding of explanation quality across varied model contexts. Incorporating more XAI evaluation metrics could further validate and extend our observations, enhancing the applicability of XAI in high-risk domains.

Acknowledgments. This study was partially supported by DFG (grant number TE257/37-1), by the Konrad Zuse School, SECAI (BMBF Project Nr. 57616814) and by the DZSF as part of the XRAISE project. The authors gratefully acknowledge the computing time made available to them by ZIH at TU Dresden.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Achtibat, R., Dreyer, M., Eisenbraun, I., Bosse, S., Wiegand, T., Samek, W., Lapuschkin, S.: From attribution maps to human-understandable explanations through concept relevance propagation. Nature Machine Intelligence 5(9), 1006– 1019 (2023)

2. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B.: Sanity checks for saliency maps. In: Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 31. Curran Associates, Inc. (2018)

3. Agarwal, D., Berbís, M.Á., Luna, A., Lipari, V., Ballester, J.B., de la Torre-Díez, I.: Automated medical diagnosis of alzheimer´ s disease using an eficient net convolutional neural network. Journal of Medical Systems 47(1), 57 (2023)

4. Ancona, M., Ceolini, E., Öztireli, C., Gross, M.: Towards better understanding of gradient-based attribution methods for deep neural networks. In: International Conference on Learning Representations (2018)

5. Bhatt, U., Weller, A., Moura, J.M.F.: Evaluating and aggregating feature-based model explanations. In: Bessiere, C. (ed.) Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20. pp. 3016–3022. International Joint Conferences on Artificial Intelligence Organization (7 2020)

6. Binder, A., Montavon, G., Lapuschkin, S., Müller, K.R., Samek, W.: Layer-wise relevance propagation for neural networks with local renormalization layers. In: International Conference on Artificial Neural Networks. pp. 63–71. Springer International Publishing (2016)

7. Chalasani, P., Chen, J., Chowdhury, A.R., Wu, X., Jha, S.: Concise explanations of neural networks using adversarial training. In: III, H.D., Singh, A. (eds.) Proceedings of the 37th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 119, pp. 1383–1391. PMLR (13–18 Jul 2020)

8. Fel, T., Picard, A., Béthune, L., Boissin, T., Vigouroux, D., Colin, J., Cadène, R., Serre, T.: Craft: Concept recursive activation factorization for explainability. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2711–2721 (06 2023)

9. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016)

10. Hedström, A., Weber, L., Krakowczyk, D., Bareeva, D., Motzkus, F., Samek, W., Lapuschkin, S., Höhne, M.M.C.: Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond. Journal of Machine Learning Research 24(34), 1–11 (2023)

11. Hedström, A., Weber, L., Lapuschkin, S., Höhne, M.: Sanity checks revisited: An exploration to repair the model parameter randomisation test. In: XAI in Action: Past, Present, and Future Applications (2023)

12. Herm, L.V., Heinrich, K., Wanner, J., Janiesch, C.: Stop ordering machine learning algorithms by their explainability! a user-centered investigation of performance and explainability. International Journal of Information Management 69, 102538 (2023)

13. Hooker, S., Erhan, D., Kindermans, P.J., Kim, B.: A benchmark for interpretability methods in deep neural networks. In: Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 32. Curran Associates, Inc. (2019)

14. Kalasampath, K., Spoorthi, K.N., Sajeev, S., Kuppa, S.S., Ajay, K., Maruthamuthu, A.: A literature review on applications of explainable artificial intelligence (xai). IEEE Access 13, 41111–41140 (2025)

15. Klein, L., Lüth, C., Schlegel, U., Bungert, T., El-Assady, M., Jäger, P.: Navigating the maze of explainable ai: A systematic approach to evaluating methods and metrics. Advances in Neural Information Processing Systems 37, 67106–67146 (2024)

16. Knoll, C., Ullrich, J., Manjooran, T., Göller, K., Amjad, H., Tetzlaf, R., Seitz, S.: Xaiev - a framework for the evaluation of xai-algorithms for image classification. In: Proceedings of The 3rd World XAI Conference. Springer (2025)

17. Liu, Z., Peng, J., Guo, X., Chen, S., Liu, L.: Breast cancer classification method based on improved vgg16 using mammography images. Journal of Radiation Research and Applied Sciences 17(2), 100885 (2024)

18. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 11976–11986 (2022)

19. Lopes, P., Silva, E., Braga, C., Oliveira, T., Rosado, L.: Xai systems evaluation: A review of human and computer-centred methods. Applied Sciences 12(19) (2022)

20. Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017)

21. Nauta, M., Trienes, J., Pathak, S., Nguyen, E., Peters, M., Schmitt, Y., Schlötterer, J., van Keulen, M., Seifert, C.: From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable ai. ACM Comput. Surv. 55(13s) (Jul 2023)

22. Nguyen, A., Martínez, M.R.: On quantitative aspects of model interpretability (2020), https://arxiv.org/abs/2007.07584

23. Petkovic, D.: It is not “accuracy vs. explainability”—we need both for trustworthy ai systems. IEEE Transactions on Technology and Society 4(1), 46–53 (2023)

24. Poeta, E., Ciravegna, G., Pastor, E., Cerquitelli, T., Baralis, E.: Concept-based explainable artificial intelligence: A survey (2023), https://arxiv.org/abs/2312.12936

25. Rath, A., Mishra, B.S.P., Bagal, D.K.: Resnet50-based deep learning model for accurate brain tumor detection in mri scans. Next Research 2(1), 100104 (2025)

26. Ribeiro, M.T., Singh, S., Guestrin, C.: "why should i trust you?": Explaining the predictions of any classifier. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. p. 1135–1144. KDD ’16, Association for Computing Machinery (2016)

27. Samek, W., Binder, A., Montavon, G., Lapuschkin, S., Müller, K.R.: Evaluating the visualization of what a deep neural network has learned. IEEE Transactions on Neural Networks and Learning Systems 28(11), 2660–2673 (2017)

28. Seitz, S., Götz, T., Lindenberg, C., Tetzlaf, R., Schlegel, S.: Towards generalizable classification of partial discharges in gas-insulated hvdc systems using neural networks: Protrusions and particles. IEEE Transactions on Power Delivery 39(3), 1491–1499 (2024)

29. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Gradcam: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV) (10 2017)

30. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. In: Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015). pp. 1–14. Computational and Biological Learning Society (2015)

31. Terzieva, E., Muschalik, M., Hofman, P., Hüllermeier, E.: Identifying trends in feature attributions during training of neural networks. In: Meo, R., Silvestri, F. (eds.) Uncertainty meets Explainability Workshop, ECML-PKDD 2023. pp. 360– 366. Springer Nature Switzerland (2025)

32. Wang, Y., Zhang, T., Guo, X., Shen, Z.: Gradient based feature attribution in explainable ai: A technical review (2024), https://arxiv.org/abs/2403.10415

33. Yeh, C.K., Hsieh, C.Y., Suggala, A., Inouye, D.I., Ravikumar, P.K.: On the (in)fidelity and sensitivity of explanations. In: Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 32. Curran Associates, Inc. (2019)

34. Zhao, J., Wu, Y., Deng, R., Xu, S., Gao, J., Burke, A.: A survey of autonomous driving from a deep learning perspective. ACM Comput. Surv. 57(10) (May 2025)