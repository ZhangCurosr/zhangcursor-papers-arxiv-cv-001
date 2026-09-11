# A Multi-View and Confusion-Guided Ensemble Framework for Robust Synthetic Image Attribution ∗

Zuomin Qu

State Key Laboratory of HVDC,

China Southern Power Grid Electric Power Research Institute, China quzuomin.@csg.cn

## Abstract

Synthetic image attribution (SIA) has become increasingly important with the rapid advancement of text-to-image generation models. However, accurately identifying the source model of a generated image remains challenging due to the growing similarity among modern difusion-based generators and the presence of diverse post-processing operations. In this report, we present a multi-view and confusion-guided ensemble framework for the Synthetic Image Attribution Challenge of the DLMMDD Workshop at ICANN 2026. Our approach integrates multiple complementary architectures, including FFT-ConvNeXt, DI-NOv2, CLIP, and Xception, to capture diverse attribution cues from frequency, semantic, and forensic perspectives. To improve robustness against unknown degradations and image manipulations, extensive data augmentation strategies are employed during training, simulating realistic post-processing operations such as compression, resizing, grayscale conversion, and blur. Furthermore, we analyze the confusion patterns of the ensemble model and observe severe ambiguity between Stable Difusion 3 and Stable Difusion 3.5. To address this issue, we introduce a dedicated binary expert classifier that is selectively activated under low-confidence conditions. We additionally apply class-adaptive confidence calibration to improve the discrimination of challenging classes such as Tencent Hunyuan. The proposed framework achieved 99.53% on the public leaderboard and 99.20% on the private leaderboard. The source code and implementation details are publicly available at https://github.com/ZOMIN28/SIA.

## 1 Introduction

Recent advances in large-scale text-to-image generative models have significantly improved the realism and diversity of synthetic images. Models such as Stable Difusion, PixArt, Playground, and Tencent Hunyuan are now capable of producing highly photorealistic face images that are increasingly dificult to distinguish from real content. As a result, synthetic image attribution (SIA), which aims to identify the source generative model of a synthetic image, has emerged as an important research problem in digital media forensics and AI security [6, 10].

Despite recent progress [1, 10, 9], robust attribution remains challenging for several reasons. First, modern difusion-based generators often share similar architectures, training strategies, and sampling mechanisms, resulting in highly overlapping visual characteristics and generation fingerprints. Second, practical scenarios frequently involve various post-processing operations, including compression, resizing, grayscale conversion, and enhancement, which can substantially weaken source-specific forensic traces. These factors make it dificult for a single model or feature representation to consistently achieve reliable attribution performance across all source classes.

To address these challenges, we propose a multi-view and confusion-guided ensemble framework for the Deep Learning and Mathematical Methods for Deepfake Detection (DLMMDD) Workshop Synthetic Image Attribution Challenge [5]. Our method combines multiple complementary architectures, including FFT-ConvNeXt [3], DINOv2 [7], CLIP [8], and Xception [2], to jointly capture frequency-domain artifacts, semantic representations, and forensic patterns. In addition, we employ extensive data augmentation strategies during training to improve robustness against the unknown post-processing operations introduced in the test set.

Beyond model ensembling, we further investigate the confusion behavior of the attribution system through confusion matrix analysis. We observe that certain source pairs, particularly Stable Difusion 3 and Stable Difusion 3.5, exhibit significantly higher mutual confusion due to their architectural and generative similarity. Motivated by this observation, we introduce a confusion-guided expert refinement strategy based on a dedicated binary classifier that is selectively activated when the ensemble model produces uncertain predictions between these two classes. We additionally apply class-adaptive confidence calibration to improve the discrimination capability for challenging classes such as Tencent Hunyuan.

Our final solution achieved 99.53% on the public leaderboard and 99.20% on the private leaderboard.

## 2 Methodology

As shown in Fig. 1, our framework follows a multi-view ensemble paradigm designed for robust synthetic image attribution under diverse post-processing conditions. The overall system combines multiple complementary architectures that capture attribution signals from diferent perspectives, including frequency-domain artifacts, semantic representations, and forensic patterns. In addition, we introduce a confusion-guided expert refinement strategy to specifically address highly ambiguous source pairs observed during validation.

## 2.1 FFT-ConvNeXt Architecture

Among the ensemble components, we design a customized FFT-ConvNeXt model to jointly exploit spatial-domain and frequency-domain attribution cues. The motivation stems from the observation that synthetic image generators often leave subtle frequency artifacts that are dificult to capture using standard RGB representations alone, especially after image postprocessing.

Given an input image $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times 3 }$ , we first apply a two-dimensional Fast Fourier Transform (FFT) to obtain its frequency representation:

$$
F ( x ) = { \mathcal { F } } ( x ) .\tag{1}
$$

We then decompose the transformed spectrum into magnitude and phase components:

$$
M = \log ( 1 + | F ( x ) | ) , \quad P = \angle F ( x ) ,\tag{2}
$$

where logarithmic scaling is applied to stabilize the magnitude distribution.

To suppress low-frequency image content and emphasize generator-specific high-frequency traces, a high-pass mask is further applied around the spectrum center. The filtered magnitude and phase maps are concatenated to form a six-channel frequency representation:

$$
X _ { \mathrm { f f t } } = \operatorname { C o n c a t } ( M , P ) .\tag{3}
$$

The FFT branch consists of several convolutional layers with Batch Normalization and ReLU activation for frequency feature extraction. In parallel, the RGB image is processed by a

![](images/143aa9e3da91672fccce46c963e77c18d9218dfcf2b82e9df37382b796ffe65c.jpg)  
Figure 1: Overview of the proposed multi-view and confusion-guided ensemble framework. During training, synthetic face images undergo robust data augmentation and are fed into four complementary models, FFT-ConvNeXt, DINOv2, CLIP and Xception, to learn diverse attribution features. At inference, the models generate logits that are combined via a weighted ensemble. A confusion-guided binary expert classifier selectively refines predictions, while classadaptive confidence calibration improves predictions for challenging classes such as Tencent Hunyuan.

ConvNeXt-Base backbone pretrained on ImageNet. The final RGB and frequency features are concatenated and passed through a multilayer classifier:

$$
f _ { \mathrm { f u s i o n } } = \mathrm { C o n c a t } ( f _ { \mathrm { r g b } } , f _ { \mathrm { f f t } } ) .\tag{4}
$$

The final prediction logits are obtained as:

$$
z = \phi ( f _ { \mathrm { f u s i o n } } ) ,\tag{5}
$$

where ϕ(·) denotes the classification head composed of fully connected layers with dropout regularization.

The proposed FFT-ConvNeXt architecture enables the model to jointly leverage semantic image structures and frequency-domain forensic traces, improving robustness against postprocessing degradations.

## 2.2 Multi-View Ensemble Strategy

Synthetic image attribution is inherently challenging because diferent generative models may share similar visual characteristics while difering only in subtle generation fingerprints. To improve attribution robustness and diversity, we adopt a multi-view ensemble strategy composed of four complementary architectures:

• FFT-ConvNeXt: captures frequency-domain artifacts and high-frequency forensic traces introduced by image generation pipelines.

• DINOv2: provides strong self-supervised visual representations and captures global structural characteristics of generated images.

• CLIP: introduces semantic-level image representations with strong robustness to image variations and distribution shifts.

• Xception: serves as a forensic-oriented backbone capable of modeling subtle manipulation patterns and local image inconsistencies . “‘

These models provide complementary attribution perspectives rather than relying on a single representation space. In particular, frequency-aware models are sensitive to generator fingerprints, while semantic and forensic backbones improve robustness under unknown image transformations.

During inference, the outputs of all models are combined at the logits level. Let $z _ { i } \in \mathbb { R } ^ { C }$ denote the logits predicted by the i-th model for $C = 1 0$ source classes. The final ensemble prediction is computed as:

$$
z _ { \mathrm { e n s e m b l e } } = \sum _ { i = 1 } ^ { N } w _ { i } z _ { i } ,\tag{6}
$$

where N denotes the number of models and $w _ { i }$ represents the ensemble weight assigned to the i-th model.

The final predicted label is obtained by:

$$
\hat { y } = \arg \operatorname* { m a x } ( z _ { \mathrm { e n s e m b l e } } ) .\tag{7}
$$

In addition, K-fold training and inference are adopted to further improve model stability and generalization performance. Predictions from diferent folds are averaged during inference to reduce variance and improve robustness.

## 2.3 Robust Data Augmentation

The challenge test set contains multiple unknown post-processing operations, including JPEG compression, WEBP compression, resizing, grayscale conversion, blur, rotation, and superresolution enhancement. To improve robustness against such transformations, we employ extensive data augmentation strategies during training.

Our augmentation pipeline includes:

• random resizing and restoration;

• random resized cropping;

• center cropping;

• horizontal flipping;

• Gaussian blur;

• brightness and contrast adjustment;

• random grayscale conversion;

• small-angle rotation;

• random JPEG compression simulation.

These augmentations are designed to simulate realistic image degradations and distribution shifts introduced by the hidden test-time post-processing pipeline. In particular, JPEG simulation and grayscale conversion were found to substantially improve robustness against compressed and low-color-information samples. In addition, center cropping is randomly applied during training. Since the challenge focuses on synthetic face images, the most discriminative facial regions are typically concentrated near the image center. Random center cropping therefore helps the model focus on stable facial generation patterns while improving robustness to spatial perturbations and resizing operations. To simulate compression artifacts eficiently during training, we employ diferentiable JPEG compression based on DifJPEG<sup>1</sup>. This strategy enables eficient and realistic compression augmentation without introducing significant computational overhead.

## 2.4 Confusion-Guided Expert Refinement

To better understand the failure modes of the ensemble system, we analyze the confusion matrix on the validation set. As illustrated in Fig. 2, we observe severe confusion between Stable Difusion 3 and Stable Difusion 3.5. This behavior is expected because both models belong to the same difusion family and share highly similar generation mechanisms and visual fingerprints.

![](images/2044502ae2b3921166df52c6c431e94e3673e563205275a26ccbe3e69000d612.jpg)  
Figure 2: Normalized confusion matrix of the ensemble model on the validation set. Each row represents the predicted class (Top-1), and each column indicates the true confusing class (Top-2). Darker colors indicate higher confusion. Notably, Stable Difusion 3 (SD3) and Stable Difusion 3.5 (SD3.5) exhibit significant mutual confusion, motivating the design of a dedicated binary expert classifier. Tencent Hunyuan also shows partial confusion with multiple classes, which is addressed via class-adaptive confidence calibration.

Motivated by this observation, we introduce a dedicated binary expert classifier based on the FFT-ConvNeXt architecture to specifically distinguish between these two classes. During inference, the expert model is selectively activated when the confidence diference between the two classes falls below a predefined threshold:

$$
| p _ { \mathrm { S D 3 } } - p _ { \mathrm { S D 3 . 5 } } | < \tau ,\tag{8}
$$

where τ denotes the confidence threshold.

Table 1: Training details and hyperparameter settings for diferent ensemble models.
<table><tr><td>Model</td><td>Pretrained Weights</td><td>Trainable Components</td><td>Learning Rate</td><td>Optimizer</td></tr><tr><td>FFT-ConvNeXt</td><td>ImageNet pretrained ConvNeXt</td><td>FFT branch, classifier, full backbone</td><td>FFT/classifier: 3 × 10−4Backbone: 1 × 10−5</td><td>AdamW [4] (wd = 10−4)</td></tr><tr><td>CLIP</td><td>OpenAI pretrained CLIP</td><td>Classification head + last 3 vision encoder blocks</td><td>Head: 1 × 10−3 Encoder blocks: 1 × 10−5</td><td>AdamW [4] (wd = 10−4)</td></tr><tr><td>DINOv2</td><td>Self-supervised pretrained DINOv2</td><td>Classification head + last 3 transformer blocks</td><td>Head: 1 × 10−3Transformer blocks: 1 × 10−5</td><td>AdamW [4] (wd = 10−4)</td></tr><tr><td>Xception</td><td>ImageNet pretrained weights</td><td>Classification head + full backbone</td><td>1 × 10−4</td><td>AdamW [4] (wd = 10−4)</td></tr></table>

Under this condition, the final prediction between the two candidate classes is refined using the binary expert classifier. This confusion-guided routing strategy improves discrimination performance for highly ambiguous source pairs without afecting the overall ensemble behavior.

We additionally observe that Tencent Hunyuan exhibits partial confusion with multiple classes. To alleviate this issue, we apply class-adaptive confidence calibration by slightly increasing the ensemble confidence associated with the Tencent Hunyuan category during inference. This strategy improves the absorption of uncertain samples into the correct class region and stabilizes final predictions.

## 2.5 Training Details and Model Hyperparameters

All models were initialized from publicly available pretrained weights to improve convergence and generalization performance. During training, we adopted partial fine-tuning strategies for transformer-based backbones to preserve pretrained visual representations while adapting the models to the attribution task.

We employed the AdamW optimizer with a weight decay of $1 \times 1 0 ^ { - 4 }$ for all experiments. Diferent learning rates were assigned to backbone and classification layers to stabilize optimization. The detailed training configurations are summarized in Table 1.

For transformer-based architectures such as CLIP and DINOv2, only the final several encoder blocks were fine-tuned, while earlier layers remained frozen. This strategy improves training stability and reduces overfitting on the relatively limited attribution dataset. In contrast, FFT-ConvNeXt and Xception were trained with broader backbone adaptation to better capture low-level forensic artifacts and frequency-domain generation traces.

## 3 Experimental Setup

## 3.1 Dataset

Experiments were conducted on the oficial dataset released for the DLMMDD Workshop Synthetic Image Attribution Challenge [5]. The dataset consists of synthetic face images generated by 10 open-source text-to-image models, including AuraFlow, Freepik, Lumina, Photon, PixArt(σ), Playground v2.5, Stable Difusion 3, Stable Difusion 3.5, Stable Difusion XL-Turbo, and Tencent Hunyuan.

The dataset was constructed using prompts derived from the WILD dataset, where each prompt was used to generate images from all source models under controlled conditions. The complete dataset contains 10,000 synthetic images with balanced class distribution, where each source contributes 1,000 images. The provided training set contains 7,000 labeled images, while the test set contains 3,000 unlabeled images with hidden ground-truth annotations.

To increase the challenge dificulty and evaluate attribution robustness, the test images were subjected to multiple undisclosed post-processing operations, including image compression, resizing, cropping, blur, grayscale conversion, and super-resolution enhancement. These transformations substantially weaken source-specific generation traces and increase inter-class ambiguity.

Table 2: Approximate training cost of diferent ensemble models.
<table><tr><td>Model</td><td>Training FLOPs (GFLOPs)</td><td>Training Time (s/epoch)</td></tr><tr><td>FFT-ConvNeXt</td><td>45.50</td><td>140</td></tr><tr><td>DINOv2</td><td>43.94</td><td>99</td></tr><tr><td>CLIP</td><td>33.73</td><td>94</td></tr><tr><td>Xception</td><td>9.15</td><td>103</td></tr></table>

## 3.2 Hardware Environment

All experiments were conducted on a server running Ubuntu 18.04.6 LTS, equipped with an Intel Core i9-10940X CPU and a single NVIDIA RTX 3090 Ti GPU with 24 GB memory.

## 3.3 Training Configuration

Unless otherwise specified, all models were trained using a batch size of 16 for a maximum of 50 epochs with an early stopping strategy of 10 patience epochs. Diferent input resolutions were adopted according to the characteristics of each backbone architecture. Specifically, FFT-ConvNeXt was trained using input images of size 256 × 256, DINOv2 and CLIP used 224 × 224 inputs, while Xception was trained with a larger input resolution of 288 × 288 to better capture fine-grained forensic artifacts. For the Confusion-Guided Expert classifier, we adopted FFT-ConvNeXt as the backbone and set the input resolution to 512 × 512, allowing the model to capture finer forensic fingerprints that distinguish highly similar generators (i.e., Stable Difusion 3 vs Stable Difusion 3.5).

K-fold cross-validation was employed during training to improve generalization performance and reduce prediction variance. During inference, predictions from diferent folds were averaged to produce the final model outputs. In the experiments, We set K = 5. This strategy efectively reduces variance and improves robustness, leading to more stable and reliable attribution results.

## 3.4 Training Cost

Table 2 summarizes the approximate computational cost and training time of the four ensemble models. The reported FLOPs correspond to the estimated training computation per image, including both forward and backward propagation.

FFT-ConvNeXt exhibits the highest computational complexity due to the additional frequencydomain processing branch and feature fusion operations. Although Xception has significantly lower FLOPs, its training time remains relatively high because of implementation overhead and high-resolution forensic feature extraction. Overall, the multi-view ensemble achieves a favorable balance between attribution performance and computational eficiency.

## 4 Results

## 4.1 Inference Configuration

During inference, we adopt a weighted logits-level ensemble strategy to combine predictions from the four base models, including FFT-ConvNeXt, DINOv2, CLIP, and Xception. The corresponding ensemble weights are empirically set as [0.2, 0.4, 0.3, 0.1], reflecting the relative contribution of each model to the final attribution performance.

In particular, DINOv2 and CLIP are assigned higher weights due to their stronger generalization capability under distribution shifts and post-processing perturbations, while FFT-ConvNeXt and Xception contribute complementary frequency-domain and forensic-local cues.

Table 3: Public leaderboard performance of diferent models and ensemble strategies.
<table><tr><td>Method</td><td>Public Score</td></tr><tr><td>FFT-ConvNeXt</td><td>0.969333</td></tr><tr><td>DINOv2 CLIP</td><td>0.976000 0.955333</td></tr><tr><td>Xception</td><td>0.958000</td></tr><tr><td>Ensemble</td><td>0.987333</td></tr><tr><td>Ensemble + K-fold</td><td>0.995333</td></tr></table>

Table 4: Ablation study of the SD3/SD3.5 expert classifier.
<table><tr><td>Setting</td><td>Public Score</td></tr><tr><td>Without expert classifier</td><td>0.992666</td></tr><tr><td>With expert classifier</td><td>0.995333</td></tr></table>

Additionally, for the Confusion-Guided Expert classifier targeting SD3 and SD3.5, we set a trigger threshold of 0.5: the expert classifier is activated if the top-1 and top-2 predicted probabilities correspond to SD3 and SD3.5 and their diference is within 0.5. For Tencent Hunyuan, the class-adaptive confidence calibration is applied with a threshold of 0.2 to refine predictions for this partially ambiguous class. The calibration threshold was selected using the validation set and fixed before test-time inference.

## 4.2 Public Leaderboard Performance

Table 3 reports the public leaderboard performance of individual models and the proposed ensemble framework. Among the single-model approaches, DINOv2 achieved the best standalone performance, demonstrating the efectiveness of self-supervised visual representations for synthetic image attribution. FFT-ConvNeXt also achieved strong results, indicating that frequency-domain forensic cues provide highly discriminative attribution information.

By combining multiple complementary models through logits-level ensemble learning, the attribution performance was further improved. In addition, K-fold aggregation significantly enhanced prediction stability and generalization capability, resulting in the best public leaderboard score of 0.995333. On the hidden private leaderboard, the proposed system achieved 99.20%, indicating that the observed performance generalizes beyond the public evaluation subset.

## 4.3 Ablation Study

We further evaluate the efectiveness of the proposed confusion-guided refinement strategies through ablation experiments.

## 4.3.1 Efectiveness of the SD3/SD3.5 Expert Classifier

As discussed in Section 3, Stable Difusion 3 and Stable Difusion 3.5 exhibit significant mutual confusion due to their highly similar generation characteristics. To address this issue, we introduce a dedicated binary expert classifier that is selectively activated under low-confidence conditions.

Table 4 shows that the expert classifier improves the public leaderboard score from 0.992666 to 0.995333, demonstrating the efectiveness of confusion-guided refinement for highly ambiguous source pairs.

Table 5: Ablation study of Tencent Hunyuan confidence calibration.
<table><tr><td>Setting</td><td>Public Score</td></tr><tr><td>Without calibration</td><td>0.993333</td></tr><tr><td>With calibration</td><td>0.995333</td></tr></table>

## 4.3.2 Efectiveness of Tencent Hunyuan Confidence Calibration

We additionally evaluate the proposed class-adaptive confidence calibration strategy for Tencent Hunyuan. As observed in the confusion analysis, Tencent Hunyuan exhibits partial ambiguity with several other source classes.

As shown in Table 5, applying confidence calibration improves the final public leaderboard score from 0.993333 to 0.995333, indicating that class-adaptive refinement helps stabilize predictions for dificult classes.

## 5 Reproducibility Statement

All results reported in this report can be fully reproduced using the provided code repository. The reproduction procedure is as follows:

1. Train the four baseline classification models (FFT-ConvNeXt, DINOv2, CLIP, and Xception) with K-fold cross-validation by running:

python train.py

The trained model weights are saved under the checkpoints/ directory.

2. Train the Confusion-Guided Expert classifier by running:

python train\_bin.py

The trained expert model weights are also stored in checkpoints/.

3. Obtain the K-fold ensemble logits on the test set by running:

python infer.py

The resulting logits are saved in the logits result/ directory.

4. Perform the final weighted ensemble, activate the Confusion-Guided Expert classifier when applicable, and apply class-adaptive confidence calibration for Tencent Hunyuan by running:

python softvote.py

The final test set predictions are saved in the results/ directory.

This stepwise procedure, combined with the provided configuration files and data preprocessing routines, ensures full reproducibility of our results on the Synthetic Image Attribution Challenge dataset.

## 6 Conclusion

In this report, we presented a multi-view and confusion-guided ensemble framework for robust synthetic image attribution. By leveraging complementary models—FFT-ConvNeXt, DINOv2, CLIP, and Xception—at the logits level, combined with targeted confusion-guided refinement strategies, our approach efectively addresses inter-class ambiguities, particularly between highly similar generators. All code and training protocols are publicly released to ensure reproducibility, facilitating further research in synthetic image forensics and attribution.

## References

[1] Lucy Chai, David Bau, Ser-Nam Lim, and Phillip Isola. What makes fake images detectable? understanding properties that generalize. In European conference on computer vision, pages 103–120. Springer, 2020.

[2] Fran¸cois Chollet. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1251–1258, 2017.

[3] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986, 2022.

[4] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[5] Andrea Montibeller, Barbara Corradini, Pietro Bongini, Sara Mandelli, and Simone Bonechi. Dlmmdd workshop: Synthetic image attribution. https://kaggle.com/ competitions/dlmmdd-workshop-synthetic-source-attribution-challenge, 2026. Kaggle Competition.

[6] Tingshu Mou, Zhipeng Wei, Chao Gong, Jingjing Chen, and Xingjun Ma. Imageattributionbench: How far are we from generalizable attribution? arXiv preprint arXiv:2605.12967, 2026.

[7] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[8] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[9] Shiyu Wu, Shuyan Li, Jing Li, Jing Liu, and Yequan Wang. Omnidfa: A unified framework for open set synthesis image detection and few-shot attribution. arXiv preprint arXiv:2509.25682, 2025.

[10] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22412–22423, 2023.