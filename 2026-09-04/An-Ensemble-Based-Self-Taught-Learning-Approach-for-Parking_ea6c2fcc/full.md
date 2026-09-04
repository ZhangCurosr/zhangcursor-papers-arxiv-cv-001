# An Ensemble-Based Self-Taught Learning Approach for Parking Space Classification Under Limited Data

Lucas de Oliveira Cunha<sup>1</sup>, Joelton Deonei Gotz<sup>1</sup>, Paulo Lisboa de Almeida<sup>2</sup>, Andre Gustavo Hochuli<sup>3</sup>

Abstract— Parking spot classification is a fundamental task in intelligent transportation systems, yet most deep learning approaches rely on large amounts of annotated data and exhibit limited generalization across heterogeneous environments. To address these limitations, we investigate a self-taught learning framework based on unsupervised representation learning with convolutional autoencoders. The proposed approach learns transferable visual representations from unlabeled data and reuses the learned encoders as fixed feature extractors for supervised classification with limited annotated samples in the target domain. To further enhance robustness and mitigate architectural bias, an ensemble of heterogeneous autoencoders is employed, with independent classifier heads and prediction fusion at inference time. Experiments conducted on the PKLot and CNRPark benchmarks under cross-dataset evaluation protocols show that the proposed ensemble-based strategy substantially reduces annotation requirements while improving robustness under significant domain shifts, achieving accuracies between 93% and 96% in data-constrained scenarios.

Index Terms—self-taught learning, parking lot monitoring, parking spot classification.

## I. INTRODUCTION

The expansion of urban environments has increased the demand for intelligent transportation systems capable of efficiently monitoring parking infrastructure. Parking lot analysis is essential to this context, supporting tasks such as occupancy estimation and availability forecasting [1].

Recent advances in computer vision and deep learning have improved performance in parking-related tasks [2], [3]. However, most existing approaches rely on large, scenariospecific annotated datasets and exhibit limited robustness when deployed across heterogeneous parking environments characterized by environmental variations and diverse scene layouts [4], often resulting in significant performance degradation under unseen conditions [5], [6].

A major challenge in parking lot analysis is the high cost of manual annotation of data from the target parking lot [7] (e.g., collect and annotate as occupied or empty thousands of images from the target), often required by highly accurate systems [2], [4], [8]. Such annotation is time-consuming and often impractical at scale, motivating learning paradigms that reduce reliance on labeled data while improving robustness to domain shifts.

In this vein, Self-Taught Learning (STL) has emerged as an effective paradigm to mitigate the reliance on large-scale annotated datasets [9], [10]. Unlike fully supervised methods, STL learns generic representations from unlabeled data and transfers them to target tasks with limited labeled samples.

In this work, we apply self-taught learning to parking spot classification and, as a contribution, introduce an ensemble of heterogeneous convolutional autoencoders to learn complementary unsupervised representations, improving robustness and generalization under cross-dataset parking scenarios.

To guide our investigation, we formulate the following research questions (RQs):

• (RQ1) To what extent can self-taught learning reduce the number of labeled target samples required to achieve competitive performance in parking lot classification tasks?

• (RQ2) How robust are STL-based representations to domain shifts across different parking lots in terms of model generalization?

• (RQ3) How does the proposed approach compare with fully supervised state-of-the-art methods?

To address these questions, we conduct a comprehensive experimental evaluation on the PKLot [8] and CNRPark [4] benchmark datasets using a well-established cross-dataset evaluation protocol. The proposed self-taught learning framework is systematically assessed by varying the unlabeled source domain and the number of labeled target samples. This analysis evaluates robustness under domain shifts and demonstrates the feasibility of learning transferable parking representations with limited labeled target data.

## II. STATE OF THE ART

Numerous strategies for parking lot classification have been explored in prior work, as comprehensively reviewed in [2]. However, most methods are evaluated under closeddomain assumptions, with only a limited number explicitly addressing domain adaptation, particularly under scarce labeled target data. To ensure objective comparison and reproducibility, our discussion is restricted to approaches evaluated on publicly available datasets. A graphical timeline is shown in Figure 1.

One of the earliest works on parking lot occupancy classification was proposed by Almeida et al. [8], who introduced the PKLot dataset, consisting of approximately 700,000 manually annotated images captured from two parking facilities and three fixed camera viewpoints. Their approach combined hand-crafted texture descriptors with Support Vector Machine (SVM) classifiers, achieving accuracies above 99% when extensive labeled data from the target parking lot were available for training. However, the study also highlighted limited cross-domain generalization, with performance dropping to around 90% when models were applied to unseen parking lots without explicit domain adaptation.

Using a cross-view evaluation protocol without domain adaptation, Amato et al. [4] reported an average accuracy of 88.5% on the CNRPark-EXT dataset, which contains approximately 160,000 labeled images acquired from nine camera viewpoints within a single parking lot. The dataset exhibits substantial visual variability due to viewpoint changes, illumination variations, shadow effects, and partial occlusions from surrounding structures.

In a related study, Nurullayev and Lee [11] proposed a deep learning framework based on dilated convolutions and pixel-skipping strategies to enlarge the effective receptive field, achieving an average accuracy of 96.4% under a crossview evaluation protocol without explicit domain adaptation.

Recently, Almeida et al. [2] reported that state-of-theart approaches reach approximately 92% accuracy when no target-domain samples are used. Further, Hochuli et al. [5] showed that a global model trained on PKLot and CNRPark-EXT generalizes to unseen parking lots with an average accuracy of 92.8%, further highlighting the challenges of crossdomain generalization in parking occupancy classification.

With respect to data annotation, state-of-the-art methods typically rely on rotated rectangles or polygonal annotations to precisely delimit parking spots, which are costly to obtain. Hochuli et al. [7] showed that simpler bounding-box annotations can achieve competitive performance by exploiting contextual information, reducing annotation effort, and reaching up to 97% accuracy using 1,000 labeled samples.

Alves et al. [6] developed a strong general teacher models trained on thousands of labeled samples, which can be distilled into lightweight students. By distilling these students with pseudo-labeled samples collected from the target domain, using images from the first seven days of deployment, the authors achieved an average accuracy of 96.6%.

More recently, Santos et al. [12] explored the use of generative adversarial networks (GANs) to synthesize targetdomain samples from a limited set of labeled examples, achieving approximately 97% accuracy using 256 annotated target samples. However, this performance depends on a complex and computationally expensive training pipeline.

As discussed so far, most state-of-the-art approaches rely on large annotated source datasets to train supervised models that are later adapted to target domains with limited labeled data. Self-taught learning (STL) addresses this limitation by learning generic representations from unlabeled data using unsupervised models, typically autoencoders, and reusing them as fixed feature extractors for task-specific classifiers. While STL has shown strong generalization capabilities in limited-data settings, such as facial expression recognition [9], its effectiveness for parking lot classification remains unexplored.

![](images/51fb93ad9d5a69fb25cdf2eec2e837b9f4f9a3913963152a8999be32836dc049.jpg)  
Fig. 1: Timeline of works related to the parking space classification problem focusing on cross-dataset approaches.

## III. PROPOSED METHOD

To reduce data acquisition and labeling costs while improving generalization under cross-dataset parking scenarios, this work proposes an ensemble-based self-taught learning (STL) framework for parking lot applications. The key contribution is the use of an ensemble of heterogeneous convolutional autoencoders to learn transferable visual representations in a fully unsupervised manner. Architectural diversity across autoencoders enables the learning of complementary latent features, mitigating architectural bias and improving robustness to domain shifts, as evidenced in Delazeri et al. [9]. These representations are subsequently reused to train lightweight classifiers with limited labeled target data, enabling scalable and efficient deployment.

Figure 2 illustrates the proposed framework. In the representation learning stage (Step 1), the ensemble of convolutional autoencoders is trained independently on unlabeled source datasets. In the task learning stage (Step 2), encoder weights are frozen and reused as fixed feature extractors, while lightweight classifiers are trained on top of the latent representations using a limited number of labeled target samples. During inference, predictions from all encoderclassifier pairs are combined through an ensemble fusion rule that aggregates posterior probabilities to produce the final decision.

To the best of our knowledge, this approach has not yet been explored in the context of parking lot analysis. Additionally, we analyze the impact of the source domain, varying levels of similarity to the target domain, the effect of reducing annotated target samples, and the generalization capability of the proposed method.

## A. Representation Learning

In the representation learning stage, multiple convolutional autoencoders (CAEs) are independently trained to capture diverse and complementary visual representations from unlabeled data. Each CAE follows an encoder–decoder design with progressive spatial downsampling in the encoder and a symmetric decoder for reconstruction.

![](images/551d806a75af9e9a9f68d832908076fedd5f24ba4b1ec7bb560a2b8ba911d342.jpg)  
Fig. 2: Proposed method: an ensemble of CAEs learns unsupervised representations from unlabeled source-domain data (Step 1), which are used to train lightweight target-domain classifiers with limited labels (Step 2). Final predictions are obtained by fusing classifier posterior probabilities.

We employ a stochastic generator that instantiates convolutional architectures (encoder) with heterogeneous depth, downsampling strategies, and latent dimensionality. The generation procedure is detailed in Algorithm 1, while the architectural hyperparameters follow the guidelines proposed in Delazeri et al. [9]. The number of generated CAEs and the corresponding training protocol are described in Section IV.

Algorithm 1 Stochastic Convolutional Autoencoder Gener  
ator   
1: Sample number of encoder blocks $L \sim \mathcal { U } \{ 2 , 5 \}$   
2: Initialize pooling counter $p \gets 0$   
3: for i = 1 to L do   
4: Sample filters f<sub>i</sub> ∈ {8, 16, 32, 64, 128}   
5: Add Conv2D(3 × 3, f ) + ReLU   
6: if U(0, 1) < .5 and $p < 4$ then   
7: Add MaxPooling2D, p ← p + 1   
8: end if   
9: end for   
10: Sample latent dimension d ∼ U[128, 2048]   
11: Project encoder output to latent vector $\mathbf { z } \in \mathbb { R } ^ { d }$   
12: Construct decoder symmetrically to encoder   
13: for each encoder block in reverse order do   
14: Add Conv2DTransp (stride=2 if pooled else 1)   
15: end for   
16: Add final Conv2D(3 × 3) with sigmoid activation

For each instance, the number of convolutional blocks is sampled from the interval {2, 5}. Each block consists of a 3×3 Conv2D layer with ReLU activation, where the number of filters is randomly selected from {8, 16, 32, 64, 128}. After each convolutional layer, a MaxPooling2D operation is inserted with probability 0.5. To prevent excessive spatial collapse for 64 × 64 inputs, the total number of pooling operations is explicitly constrained to a maximum of four. The convolutional sequence output is mapped to a latent representation of dimension d, where d is randomly sampled

from the range {128, 2048}.

The decoder is constructed symmetrically to the encoder. The latent vector is first expanded via a fully connected layer and reshaped to match the final encoder feature-map dimensions. A sequence of Conv2DTranspose layers then progressively reconstructs the input, using stride $2 \times 2$ for stages corresponding to encoder blocks that applied max pooling, and stride $1 \times 1$ otherwise. The final reconstruction layer employs a $3 \times 3$ convolution with sigmoid activation to match normalized pixel intensities.

## B. Task Learning

In the task learning phase, the encoder learned during representation learning (Section III-A) is reused as a fixed convolutional feature extractor, with all parameters frozen to preserve the learned representations.

The classifier is constructed by stacking a lightweight fully connected head on top of the frozen encoder, as illustrated in the Figure 2 (Step 2). A dropout layer with a rate of 0.2 is first applied to the latent features to improve regularization. This is followed by two fully connected layers with 256 and 128 neurons, respectively, both using ReLU activation function. The final layer has two output units followed by Softmax, producing the posterior class probabilities.

## C. Training and Inference

Each convolutional autoencoder (CAE) in the ensemble is trained independently during the unsupervised representation learning stage. Given an unlabeled source dataset $\mathcal { D } _ { s } ~ =$ $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N _ { s } }$ , each CAE $f _ { \theta _ { k } }$ , with $k = 1 , \ldots , K$ , is optimized by minimizing the mean squared reconstruction error:

$$
\mathcal { L } _ { \mathrm { r e c } } ^ { ( k ) } = \frac { 1 } { \left| \mathcal { D } _ { s } \right| } \sum _ { { \bf x } \in \mathcal { D } _ { s } } \left\| { \bf x } - f _ { \theta _ { k } } ( { \bf x } ) \right\| _ { 2 } ^ { 2 } ,
$$

where $f _ { \theta _ { k } } ( { \bf x } ) = D _ { \theta _ { k } } ( E _ { \theta _ { k } } ( { \bf x } ) )$ , and $E _ { \theta _ { k } }$ and $D _ { \theta _ { k } }$ denote the encoder and decoder of the k-th CAE, respectively. Each CAE is trained using the Adam optimizer, without parameter sharing or joint optimization, promoting the learning of different latent representations across the ensemble.

Given a labeled target dataset $\mathcal D _ { t } ~ = ~ \{ ( \mathbf x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { N _ { t } }$ , a lightweight classifier $g _ { \phi _ { k } }$ is trained on top of the extracted latent features $\mathbf { z } _ { j } ^ { ( k ) } = E _ { \theta _ { k } } ( \mathbf { x } _ { j } )$ . The classifier head is optimized using the Adam optimizer by minimizing the categorical cross-entropy loss.

At inference time, the ensemble produces a set of predictions $\{ \hat { y } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ , one for each CAE–classifier pair. These predictions are aggregated using a majority voting rule

$$
\hat { y } = \arg \operatorname* { m a x } _ { c \in \{ 1 , \ldots , C \} } \sum _ { k = 1 } ^ { K } \mathbb { I } \Big ( \hat { y } ^ { ( k ) } = c \Big ) ,
$$

yielding the final class decision.

## D. Datasets

In this work, we employ two widely used benchmark datasets for vision-based parking occupancy classification: (i) PKLot [8] and (ii) CNRPark-EXT [4]. Both datasets reflect real-world deployment conditions by incorporating multiple parking facilities and substantial environmental variability.

PKLot. The PKLot dataset [8] was acquired over a period of roughly three months using a fixed temporal sampling rate of five minutes between consecutive captures. It contains 12,417 images and approximately 700,000 manually labeled parking spaces, organized into three independent parking lot scenarios: UFPR04, UFPR05, and PUCPR. Image examples of the dataset are shown in Figure 3.

![](images/3951916cfbf81d74b68dc565252b9db66bfdcb93ffff202d7ca0fb6cd6460768.jpg)  
(a) UFPR04

![](images/fe6e5d81a2dbc6f109c92019fdf148d3803cf4458143d6fd14f98c32a8dfbbb1.jpg)  
(b) UFPR05

![](images/6a9c9b6fb9a15ba7c97e4bf0d90df1b8e37a449a9bdf2b354058c8e440f431e9.jpg)  
(c) PUCPR  
Fig. 3: The three deployment scenarios from PKLot.

CNRPark-EXT. Introduced by Amato et al. [4], the CNRPark-EXT dataset includes approximately 160,000 labeled parking spaces captured from nine static camera viewpoints observing a single parking lot. The dataset poses several challenges, such as severe illumination artifacts, environmental noise from raindrops on the camera lens, and frequent partial occlusions from trees and lamp posts. Image examples are presented in Figure 4.

![](images/19c3b4d8a4bfc61e40bd3d58ba20f5fc9749ca5620fa1d4f78737e4477c8ab0c.jpg)  
(a) Camera 1

![](images/0458b3eaee6123130bff91ff3acc8683cd31b044434e59170080ac32bc5cad02.jpg)  
(b) Camera 4

![](images/1075961a1a1bea0ec2aedebd7ea513bd5cf0201e0d55e60b27a5d49c2602a8d3.jpg)  
(c) Camera 9  
Fig. 4: Three out of nine scenarios from the CNRPark-EXT.

Table I briefly summarizes the properties of the datasets. For a thorough description of the datasets and their applications, refer to [2], [4], [8].

TABLE I: Summary parking lots datasets used in this work
<table><tr><td>Dataset</td><td>Spots</td><td>Days</td><td>Camera Angles</td><td>Park. Lots</td></tr><tr><td>PKLot</td><td>695,851</td><td>99</td><td>3</td><td>2</td></tr><tr><td>CNRPark-EXT</td><td>157,549</td><td>23</td><td>9</td><td>1</td></tr></table>

## IV. EXPERIMENTS

In this section, we evaluate the proposed approach by adopting a widely used cross-dataset evaluation protocol [2], [4], [5], [7], [11], [12] to ensure fair comparison with state-of-the-art methods. Using the datasets described in Section III-D, the cross-dataset protocol enforces a strict separation between datasets used for unsupervised representation learning and those used for target-domain task-specific adaptation. This design prevents data bias by ensuring that samples from scenarios observed during representation learning are not reused during target-domain optimization. Each target camera is treated as an independent deployment scenario, yielding 12 target scenarios under heterogeneous conditions. All models are trained following the procedures and hyperparameters defined in Section III-C.

## A. Unsupervised Representations Learning (STEP 1)

For unsupervised representation learning, ten convolutional autoencoders (K = 10) were generated using the stochastic architecture generator described in Section III-A. Each autoencoder $( f _ { \theta _ { k } } ^ { * } )$ was trained independently for 50 epochs, and the model achieving the lowest validation MSE was selected. Training was conducted using 1,000 randomly sampled images and 64 validation images per representation dataset, ensuring balanced exposure to representation-domain variability. As representation learning is performed in an unsupervised manner, this subset size is sufficient to capture dominant structural patterns without introducing datasetspecific bias.

Figure 5 illustrates reconstruction results on representation-domain scenarios. Although some geometric distortions are present, the primary goal is to expose the representation learning process to diverse shapes and textures [13]. This diversity is particularly relevant, as the downstream task focuses on occupancy-related visual cues rather than object identity (e.g., vehicle models).

![](images/091ff7945d9cbc2975af89e1246a8c0b95c2d1db9f76f7de810664bb83ed3d83.jpg)  
Fig. 5: Qualitative comparison between real samples (top) and reconstructed ones (bottom).

## B. Supervised Task Learning (STEP 2)

In the supervised task learning stage, each selected encoder $( E _ { \theta _ { k } } )$ was reused as a fixed feature extractor and paired with an independent classifier head $g _ { \phi _ { k } }$ , as discussed in Section III-B. All encoder parameters were frozen. Each classifier $( g _ { \phi _ { k } } ^ { * } )$ was trained independently for 10 epochs, and the model achieving the lowest validation loss was selected.

Figure 6 illustrates representative parking spaces highlighting the domain shifts considered in our evaluation.

![](images/d77e3b77beb1188204653e6284217d99ed0dbdae02ef06a3c11232a90fbd2f60.jpg)  
Fig. 6: Examples individual parking spots across the camera views in the PKLot and CNRPark-EXT datasets.

To address RQ1, which investigates the impact of the number of annotated target samples (R) on effective task adaptation, we conducted experiments using $R \in$ {64, 128, 256, 512, 1024} labeled samples from the target domain. These samples were used exclusively in the supervised task learning stage. Samples were collected chronologically, starting from the first day of acquisition for each target camera, and all samples from these days were excluded from the test set to prevent data leakage. The values of R are selected based on prior studies [7], [12], which show that competitive parking occupancy classification performance can be achieved with approximately 1,000 annotated target samples, with marginal gains beyond this point.

Table II reports classification accuracy across target scenarios. Accuracy consistently improves with additional annotated samples, with several scenarios achieving strong performance under limited supervision (64–128 samples). In most cases, gains saturate beyond 256 samples, corroborating prior observations [12]. More challenging camera views show greater sensitivity to annotation size, while overall results remain stable, underscoring the robustness and transferability of the proposed approach across heterogeneous target conditions.

TABLE II: Classification accuracy (%) achieved by taskspecific classifiers across different target scenarios as a function of the number of annotated target samples
<table><tr><td rowspan="2">Representation Domain</td><td rowspan="2">Target Domain</td><td colspan="5">Annotated Data</td></tr><tr><td>64</td><td>128</td><td>256</td><td>512</td><td>1024</td></tr><tr><td rowspan="3">CNR</td><td>PUC</td><td>95.1</td><td>97.4</td><td>98.2</td><td>98.4</td><td>98.7</td></tr><tr><td>UFPR04</td><td>94.0</td><td>96.3</td><td>96.5</td><td>97.0</td><td>97.4</td></tr><tr><td>UFPR05</td><td>97.0</td><td>97.6</td><td>98.1</td><td>98.4</td><td>98.3</td></tr><tr><td rowspan="9">PKLOT</td><td>CNR-1</td><td>90.7</td><td>92.6</td><td>93.7</td><td>94.3</td><td>95.8</td></tr><tr><td>CNR-2</td><td>97.3</td><td>97.1</td><td>97.4</td><td>97.6</td><td>98.4</td></tr><tr><td>CNR-3</td><td>95.8</td><td>94.7</td><td>96.3</td><td>97.0</td><td>97.7</td></tr><tr><td>CNR-4</td><td>94.7</td><td>94.5</td><td>95.1</td><td>95.0</td><td>95.4</td></tr><tr><td>CNR-5</td><td>86.2</td><td>89.7</td><td>92.3</td><td>93.5</td><td>94.7</td></tr><tr><td>CNR-6</td><td>92.2</td><td>92.5</td><td>93.7</td><td>93.4</td><td>95.1</td></tr><tr><td>CNR-7</td><td>92.3</td><td>91.8</td><td>91.1</td><td>92.0</td><td>92.9</td></tr><tr><td>CNR-8</td><td>92.9</td><td>95.0</td><td>95.3</td><td>96.0</td><td>96.6</td></tr><tr><td>CNR-9</td><td>91.2</td><td>91.0</td><td>92.5</td><td>94.3</td><td>95.8</td></tr><tr><td colspan="2">Average</td><td>93.3</td><td>94.2</td><td>95.0</td><td>95.6</td><td>96.4</td></tr></table>

An important aspect is the ability to generalize to unseen environments (RQ2). Accordingly, task-specific classifiers trained on the target domain using 1,024 annotated samples, the amount yielding the best performance, are evaluated across the remaining scenarios. The results are presented in Table III.

The proposed ensemble-based self-taught learning framework demonstrates feasible generalization to domain shifts, without additional weight optimization. High accuracy is consistently achieved across target scenarios, particularly among cameras within the same dataset, indicating effective transfer of learned representations. Performance degradation is more evident when transferring between datasets with markedly different acquisition conditions, such as CNRPark-EXT and PKLot, reflecting severe shifts. Nevertheless, competitive accuracy is maintained in most cross-domain settings, confirming the robustness and transferability of the learned representations and validating the proposed approach in realistic deployment scenarios (RQ2).

One may question the choice of using 10 autoencoders. Figure 7 shows that cross domain generalization performance, averaged across all dataset scenarios, increases with ensemble size but quickly saturates, reaching a plateau around 10 models. Beyond this point, improvements are negligible, indicating that 10 autoencoders effectively approximate the upper performance bound while maintaining a favorable cost benefit trade off.

![](images/db2043a1b12d1e781d334083f48fb46e91dcc6334bc6a40e43564cfb4523d1ad.jpg)  
Fig. 7: Ensemble size vs. accuracy (%) averaged over all dataset scenarios (red: CNR, blue: PKLot). Solid lines with circular markers (•) denote ensemble performance; dashed lines (–) denote single-model baselines (⋄).

## C. State-of-Art Comparison

Table IV presents a comparison between the proposed framework and state-of-the-art methods (RQ3), focusing on cross-dataset evaluation, reliance on target-domain supervision, and annotation requirements. As can be observed, even with only 64 annotated samples, the proposed method maintains an average accuracy of about 93%, outperforming several approaches.

Overall, the results show that the proposed ensemblebased CAE framework achieves competitive performance with substantially reduced annotation requirements. While most state-of-the-art methods rely on large annotated datasets to learn visual representations, our approach learns representations in an unsupervised manner and requires only limited supervision for task adaptation. With 1,024 annotated target samples, it achieves an average accuracy of approximately 96%, within two percentage points of the GAN-based method of Santos et al. [12], while requiring only the training of the classifier head during deployment.

Compared to Alves et al. [6], which proposed a recent approach with competitive results, our method does not require several days (e.g., 7 days) of target data collection, and does not require the training/fine-tunning of the entire classifier during deployment (only the classification head). However, it has the drawback that, although it requires only a small number of samples, these must be manually labeled, whereas Alves et al. rely on pseudo-labels from the target parking lot.

TABLE III: Cross-scenario performance (accuracy %). The first column denotes the classifier domain and the remaining columns correspond to the target cross-domains.
<table><tr><td rowspan="2">Task-Specific Classifier</td><td colspan="10">Cross Domains (without weight optimization)</td></tr><tr><td>CNR-1</td><td>CNR-2</td><td>CNR-3</td><td>CNR-4</td><td>CNR-5</td><td>CNR-6</td><td>CNR-7</td><td>CNR-8</td><td>CNR-9</td><td>PUC</td><td>UFPR04 UFPR05</td></tr><tr><td>CNR-1</td><td>95.8</td><td>98.1</td><td>96.5</td><td>96.1</td><td>92.0</td><td>95.7</td><td>90.9</td><td>95.7 93.7</td><td>89.2</td><td>76.9</td><td>76.6</td></tr><tr><td>CNR-2</td><td>90.5</td><td>98.4</td><td>95.2</td><td>96.3</td><td>91.4</td><td>95.8 90.3</td><td>96.2</td><td>93.8</td><td>91.2</td><td>85.4</td><td>84.1</td></tr><tr><td>CNR-3</td><td>85.9</td><td>98.0</td><td>97.7</td><td>96.0</td><td>89.0</td><td>95.1</td><td>87.3 95.9</td><td>91.5</td><td>92.3</td><td>83.0</td><td>87.4</td></tr><tr><td>CNR-4</td><td>86.3</td><td>97.2</td><td>94.1</td><td>95.4</td><td>90.8</td><td>95.3</td><td>87.7</td><td>96.5 92.7</td><td>93.7</td><td>88.3</td><td>87.4</td></tr><tr><td>CNR-5</td><td>88.5</td><td>98.3</td><td>96.3</td><td>97.2</td><td>94.7</td><td>96.1</td><td>90.7</td><td>96.8 93.8</td><td>96.6</td><td>88.4</td><td>85.4</td></tr><tr><td>CNR-6</td><td>83.1</td><td>96.5</td><td>93.0</td><td>94.5</td><td>90.1</td><td>95.1</td><td>86.0</td><td>95.5 90.1</td><td>93.8</td><td>85.5</td><td>84.2</td></tr><tr><td>CNR-7</td><td>91.1</td><td>98.1</td><td>97.7</td><td>97.6</td><td>95.4</td><td>97.2</td><td>92.9</td><td>97.3 95.4</td><td>93.3</td><td>90.1</td><td>79.4</td></tr><tr><td>CNR-8</td><td>83.0</td><td>93.9</td><td>94.6</td><td>93.7</td><td>89.5</td><td>95.3</td><td>87.3</td><td>96.6 92.0</td><td>94.8</td><td>84.7</td><td>85.7</td></tr><tr><td>CNR-9</td><td>91.1</td><td>98.0</td><td>95.4</td><td>97.3</td><td>94.0</td><td>97.0</td><td>91.0</td><td>97.6 95.8</td><td>91.8</td><td>90.1</td><td>88.2</td></tr><tr><td>PUC</td><td>91.4</td><td>95.0</td><td>92.0</td><td>95.4</td><td>92.4</td><td>94.4</td><td>90.2</td><td>93.4</td><td>91.4 98.7</td><td>91.5</td><td>81.3</td></tr><tr><td>UFPR04</td><td>89.0</td><td>94.2</td><td>92.8</td><td>92.6</td><td>88.5</td><td>93.6</td><td>90.6</td><td>91.8 88.5</td><td>93.2</td><td>97.4</td><td>84.5</td></tr><tr><td>UFPR05</td><td>71.7</td><td>83.0</td><td>85.1</td><td>84.9</td><td>74.9</td><td>87.6</td><td>76.9</td><td>85.8</td><td>77.4 82.6</td><td>87.4</td><td>98.3</td></tr></table>

TABLE IV: State-of-the-Art Comparison. In <sup>1</sup> the target samples used are unlabeled.
<table><tr><td>Author</td><td>Approach</td><td>Limited Target Annot. Data</td><td>Target Training</td><td>Accuracy</td></tr><tr><td>Almeida et al. [8]</td><td>LBP + SVM</td><td></td><td>No</td><td>~87%</td></tr><tr><td>Amato et al. [4]</td><td>Custom CNN</td><td></td><td>No</td><td>~88%</td></tr><tr><td>Nurullayev et al. [11]</td><td>Custom CNN</td><td></td><td>No</td><td>~96%</td></tr><tr><td>Almeida et al. [2]</td><td>Survey</td><td></td><td></td><td>~92%</td></tr><tr><td>Hochuli et al. [5]</td><td>MobileNetV3</td><td></td><td>No</td><td>~92%</td></tr><tr><td>Alves et al. [6]</td><td>MobileNetV3 and Custom CNN</td><td></td><td>Yes1</td><td>~97%</td></tr><tr><td>Santos et al. [12]</td><td>GANs and MobileNetV3</td><td>1024 samples</td><td>Yes</td><td>~98%</td></tr><tr><td>Ours</td><td>Ensemble CAE-ANN</td><td>64 samples 1024 samples</td><td>Yes</td><td>~93% ~96%</td></tr></table>

## V. CONCLUSION

This work introduced an ensemble-based self-taught learning framework for parking spot occupancy classification that combines unsupervised representation learning with efficient task adaptation. By training heterogeneous convolutional autoencoders on unlabeled data and reusing their encoders as fixed feature extractors, the proposed approach reduces annotation requirements while promoting robustness under domain shifts.

Experiments on the PKLot and CNRPark-EXT benchmarks under cross-dataset protocols demonstrate competitive performance across twelve target scenarios, even with limited supervision. High accuracy is achieved with as few as 64 annotated samples, and performance saturates with modest annotation effort. Cross-scenario evaluations further confirm strong generalization without additional weight optimization. Overall, the proposed framework offers an effective and scalable alternative to fully supervised methods for realworld parking monitoring applications.

## ACKNOWLEDGMENT

The authors acknowledge the financial support of the Coordenac¸ao de Aperfeic¸oamento de Pessoal de N ˜ ´ıvel Superior (CAPES), Financiadora de Estudos e Projetos (FINEP), and the Fundac¸ao Arauc˜ aria, in partnership with the Secre-´ taria da Ciencia, Tecnologia e Ensino Superior do Estado doˆ Parana (SETI-PR), under Grant No. 653/2025.´

## REFERENCES

[1] S. Bhattacharya, S. R. K. Somayaji, T. R. Gadekallu et al., “A review on deep learning for future smart cities,” Internet Technology Letters, vol. 5, no. 1, p. e187, 2022.

[2] P. Almeida, J. Alves, R. Parpinelli, and J. Barddal, “A systematic review on computer vision-based parking lot management applied on public datasets,” Expert Systems With Applications, vol. 198, p. 116731, 2022.

[3] P. R. L. de Almeida, J. H. Alves, L. S. Oliveira, A. G. Hochuli, and others., “Vehicle occurrence-based parking space detection,” in 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC), 2023, pp. 1524–1529.

[4] G. Amato, F. Carrara, F. Falchi, C. Gennaro, C. Meghini, and C. Vairo, “Deep learning for decentralized parking lot occupancy detection,” Expert Systems with Applications, vol. 72, pp. 327–334, 2017.

[5] A. G. Hochuli, J. P. Barddal et al., “Deep single models vs. ensembles: Insights for a fast deployment of parking monitoring systems,” in 2023 ICMLA, 2023, pp. 1379–1384.

[6] P. L. Alves, A. Hochuli, L. E. de Oliveira, and P. L. de Almeida, “Optimizing parking space classification: Distilling ensembles into lightweight classifiers,” in 2024 International Conference on Machine Learning and Applications (ICMLA), 2024, pp. 1016–1020.

[7] A. G. Hochuli, A. S. Britto, P. R. Almeida, Alves et al., “Evaluation of different annotation strategies for deployment of parking spaces classification systems,” in 2022 IJCNN. IEEE, 2022, pp. 1–8.

[8] P. R. Almeida, L. S. Oliveira, A. S. Britto Jr, E. J. Silva Jr, and A. L. Koerich, “Pklot–a robust dataset for parking lot classification,” Expert Systems with Applications, vol. 42, no. 11, pp. 4937–4949, 2015.

[9] B. R. Delazeri, A. G. Hochuli, J. P. Barddal, A. L. Koerich, and A. d. S. Britto, “Representation ensemble learning applied to facial expression recognition,” Neural Computing and Applications, vol. 37, no. 1, pp 417–438, Jan 2025

[10] E. Germani, E. Fromont, and C. Maumet, “On the benefits of selftaught learning for brain decoding,” GigaScience, vol. 12, p. giad029, 2023.

[11] S. Nurullayev and S.-W. Lee, “Generalized parking occupancy analysis based on dilated convolutional neural network,” Sensors, vol. 19, no. 2, p. 277, 2019.

[12] A. M. F. dos Santos, P. R. L. de Almeida, J. P. Barddal, and A. G. Hochuli, “A generative domain adaptation scheme for swift deployment of parking monitoring systems,” in Intelligent Systems. Cham: Springer Nature Switzerland, 2026, pp. 34–49.

[13] N. Baker, H. Lu, G. Erlikhman, and P. J. Kellman, “Deep convolutional networks do not classify based on global object shape,” PLoS computational biology, vol. 14, no. 12, p. e1006613, 2018.