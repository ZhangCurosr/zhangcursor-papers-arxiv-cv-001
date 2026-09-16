# FAHCD-Net: Frequency-Adaptive Heatmap-Conditional Diffusion Networks for Robust Facial Landmark Detection

Jun Wan, Jiwei Hu, Shengkai Hu, and Qilu Zhu

Abstract—Facial Landmark Detection(FLD) is a crucial task in various applications and has achieved significant advancements in recent years. However, current FLD methods still struggle under challenging conditions, where facial structural variations, information loss, and noise interference severely compromise the integrity and accuracy of learned facial features. To address these issues, we propose Frequency-Adaptive Heatmap-Conditional Diffusion Network (FAHCD-Net), which integrates a Frequency-Adaptive Heatmap-Conditional Diffusion (FAHCD) model with a Smoothness Regularization (SR) loss in a cascaded framework. Specifically, the FAHCD model incorporates a Hierarchical Frequency Adaptation (HFA) module designed to suppress redundant high-frequency noise through multi-layer frequency decomposition and adaptive reconstruction, thereby preserving essential facial structures. Additionally, the SR loss is proposed to further mitigate the interference of high-frequency noise and enhance the smoothness of the generated landmark heatmaps. By cascading the FAHCD model with the SR loss, FAHCD-Net effectively leverages both statistical and frequencybased distribution characteristics of the data to progressively generate more accurate landmark heatmaps from noisy inputs. Extensive experiments on popular benchmarks demonstrate the effectiveness and robustness of the proposed method, achieving state-of-the-art performance in FLD tasks under challenging scenarios. The source code is available at https://github.com/ HJWKryptonite/FAHCD-Net.

Index Terms—Facial Landmark Detection, Diffusion Probabilistic Models, Heatmap regression

## I. INTRODUCTION

ing, involving the localization of specific landmarks in facial images (such as eye corners, nose tip, lips, and facial contour). This task is fundamental to various practical applications, including face recognition [1], [2], face synthesis [3], [4], [5], expression analysis [6], [7], and 3D face reconstruction [8].

With the development of convolutional neural network (CNN)[9], [10], [11], [12], [13], [14], FLD has made significant progress. Early methods for FLD were mostly based on coordinate regression approaches [15], [16], [17], [18], [19], which aimed to learn the mapping from CNN features to landmark coordinates through fully connected layers. Recent works predominantly adopt the heatmap regression method [20], [21], [22], [23], [24], predicting intermediate heatmaps for each landmark and decoding the landmark coordinates from them. The current state-of-the-art FLD has achieved very impressive results.

![](images/1db953f8a787d0509434ebd32dcff0872ee32690189f45fef075ec0cdaee019e.jpg)  
Fig. 1. Composition of the Power Spectral Density (PSD) map. The area within the red circle indicates the main frequency information. The yellow region shows the high-frequency components (i.e., faster-changing parts of the image, such as edges, textures, and details) of the image, while the blue region represents the low-frequency components (i.e., slower-changing parts of the image, usually including large-scale backgrounds and smoother areas, such as cheeks and forehead).

However, FLD remains challenging in complex scenes where facial expressions, poses, illumination, blur, and occlusions vary drastically. These challengings can be categorized into three aspects: (1) Facial structure variations: Facial expressions and large pose variations can cause significant shifts in landmark positions and alterations in geometric relationships, increasing the complexity of landmark distributions. Current models often struggle to handle these variations effectively. (2) Information loss: Occlusion and extreme poses (e.g., profile views) render some landmark information directly invisible, causing the loss of information. (3) Noise interference: Blur and illuminations introduce substantial noise that can mess up the model’s predictions.

Recently, Diffusion Probabilistic Models (DPMs) and scorebased generative models [25], [26], [27] have achieved impressive results in image generation and processing, surpassing traditional GANs [28]. Furthermore, the inherent characteristics of diffusion models enable them to perform effectively in addressing the three types of challenging scenarios mentioned above: (1) Diffusion models learn the statistical distribution of data rather than relying solely on simple geometric features or predefined geometric constraints, which allows them to naturally adapt to variations in facial expressions and poses. (2) The generative nature and multi-step recursive prediction mechanism of diffusion models enable them to capture the dynamic patterns of landmarks’ diffusion even in cases of missing features. (3) The denoising process can extract useful information from the noise and iteratively recover the data, as the model is originally trained using different levels of noise.

![](images/538432d0f385f80df18969e866a349738fa74e3af46c79b52960256b44cf3d32.jpg)  
GT  
FAHCD-Net w/o HFA & SR  
FAHCD-Net  
Fig. 2. Visualization of Power Spectral Density (PSD) maps (the first row), High Frequency Component (HFC) maps (the second row) and landmark heatmaps (the last row). Red circles highlight the main frequency information, while red boxes mark the shifted landmarks and their corrections by our proposed FAHCD-Net. Observations reveal that heatmaps generated solely by the diffusion model (FAHCD model w/o HFA module) contain some highfrequency noise (i.e., red box in (e)), causing shifted landmarks (i.e., red box in (h)). In contrast, heatmaps generated by the FAHCD-Net effectively suppress excessive high-frequency noise (i.e., red box in (f)), resulting in a frequency distribution closer to the ground-truth (i.e., red box in (c)). Consequently, the landmark position is corrected. (as shown in (i)).

Building on these strengths, we introduce diffusion models and propose the Frequency-Adaptive Heatmap-Conditional Diffusion Network (FAHCD-Net) for more robust FLD. As illustrated in Fig. 1, facial images exhibit distinct power spectral density (PSD) distributions, where high-frequency components capture edges and fine textures while low-frequency components correspond to smooth regions such as cheeks and forehead. The FAHCD-Net progressively generates accurate facial landmark heatmaps from noise while constraining the frequency distribution and capturing complex structural details. Moreover, to reduce the impact of redundant high-frequency noise on the quality of generated landmark heatmaps, we also design a Frequency-Adaptive Heatmap-Conditional Diffusion (FAHCD) model and a Smoothness regularization (SR) loss. The former introduces a Hierarchical Frequency Adaptation (HFA) module to perform frequency decomposition and reconstruction, adaptively adjusting the proportion of different frequency components, and the latter imposes smoothness and continuity constraints on the generated heatmap by penalizing dramatically changed parts of landmark heatmaps. As shown in Fig. 2, the diffusion model without our proposed modules generates heatmaps with excessive high-frequency noise, leading to shifted landmark predictions. In contrast, FAHCD-Net effectively suppresses such noise and corrects the landmark positions. Therefore, by cascading the FAHCD model and SR loss, the proposed FAHCD-Net outperforms state-of-the-art FLD methods and its main contributions are as follows:

1) Building on the importance of frequency information in modeling facial structures and leveraging the advantages of diffusion models in learning statistical distribution features and noise robustness, we propose FAHCD model to cope with FLD in challenging scenarios.

2) A well-designed SR loss is proposed to suppress redundant high-frequency noise and constrain the generated landmark heatmap to ensure its smoothness and continuity, thereby achieving accurate FLD, especially for occluded, illuminated and blurred faces.

3) A novel framework called FAHCD-Net is developed to seamlessly integrate the FAHCD model and SR loss in a cascaded manner to handle FLD in challenging scenarios. Experimental results show that FAHCD-Net outperforms the state-of-the-art methods on multiple challenging datasets (such as 300W, COFW, WFLW, and AFLW).

## II. RELATED WORK

In this section, we will introduce the related work on facial landmark detection and diffusion models.

## A. Facial Landmark Detection

In the early stages, FLD is primarily based on statistical model methods, such as Active Appearance Models (AAM) [29], Active Shape Models (ASM) [30], and Constrained Local Models (CLM) [31]. Recently, with the development of CNNs[32], [33], [34], [35], [36], deep learning methods have achieved significant success. Deep learning-based facial landmark detection mainly follows two mainstream approaches: coordinate regression methods [19], [37], [38], [39], [40] and heatmap regression methods [41], [42], [32], [43].

Coordinate Regression-Based Methods. These methods primarily use fully connected layers to learn the mapping between facial features and landmark coordinates. For example, Sun et al. [15] are the first to apply CNNs to facial landmark detection, proposing a deep convolutional neural network based on cascaded regression, which achieves more accurate landmark detection results. Later, to further enhance performance, MDM [17] and RAR [18] employ cascaded recurrent refinements to sequentially fine-tune landmark estimation. Li et al. [44] propose a new topology-adaptive deep graph to strengthen the results. SLPT [45] and RePFormer [19] employ attention mechanisms to learn an adaptive inherent relationship. SCE-MAE [37] addresses the limitations of selfsupervised learning in facial landmark detection tasks by combining the pretraining advantages of MAE with a selective optimization strategy for corresponding relationships. Liang et al. [38] combine conditional facial deformation with landmark detection to learn a highly generalized landmark detector through an alternating optimization approach, significantly improving detection performance on stylized faces and unseen data. Although coordinate regression methods are simple and efficient, they often struggle to capture complex spatial relationships and local details, particularly when handling large pose variations and partial occlusions.

Heatmap Regression-Based Methods. These methods output an intermediate heatmap for each landmark and use the Argmax function to consider the point with the highest intensity as the best output. For example, Kowalski et al. [20] introduce a novel cascaded deep neural network called DAN (Deep Alignment Network), which stands out from earlier cascaded networks by utilizing the whole image as input at every stage instead of focusing on specific image regions. SBR [23] utilizes the registration of synthetic images to provide supervision signals for training. By analyzing the main shortcomings of different loss functions, AWingLoss [24] is proposed to address the imbalance between foreground and background pixels. HRNet [46] connects and exchanges information by merging multi-scale image features from multiple branches, thereby generating more effective heatmaps. More recently, PIPNet [42] proposed performing heatmap and offset predictions concurrently on low-resolution feature maps, significantly improving inference efficiency while maintaining competitive accuracy. LDEQ [47] utilizes cascaded computation based on DEQ and achieve state-of-the-art performance. STARLoss [48] is proposed to cope with the semantic ambiguity problem in order to improve the FLD accuracy. PoseGuidedSW [41] innovatively integrates diffusion model features, a self-training mechanism, a pose-guided proxy task, and a two-stage clustering approach, providing an efficient and robust solution for unsupervised landmark detection tasks.

However, heatmap regression methods rely on the quality of the heatmaps, where low quality may lead to blurred details and inaccurate localization. In contrast, diffusion models generate high-quality heatmaps through a step-by-step denoising process, enabling them to recover landmark details more precisely.

## B. Diffusion Models

Diffusion Probabilistic Models (DPMs) are a type of generative model based on a Markov chain. These models can transform noise sampled from a simple distribution (e.g., Gaussian distribution) into target data sampled under a complex distribution. Ho et al. systematically explain the denoising diffusion probabilistic model (DDPM) [25], demonstrating the outstanding performance in image generation, surpassing GANs [28] in many tasks. Nichol and Dhariwal address the low sampling efficiency issue of diffusion models by adjusting the diffusion process, further improving the quality of image generation. Latent Diffusion Models (LDM) [26] significantly reduces computational resource requirements by running the diffusion process in latent space, making them especially effective in high-resolution image generation tasks. DDNM [49] formulates a sophisticated identity equation that seamlessly integrates conditions into the reverse process of the diffusion model without requiring additional training, demonstrating excellent performance in linear image restoration tasks. Conditional Diffusion Models (CDM)[50] enhance control over the generative process by introducing conditional variables. The EDM[51] framework decomposes the design of complex diffusion models into key components: the diffusion process, model architecture, optimization objectives, and sampling procedures. This modular approach provides researchers with a clearer understanding of these models and facilitates their improvement. DPM-solver[52] further accelerates sampling by employing methods like calculating the exact ODE solutions and designing higher-order solvers. Currently, extensive research and applications are being explored in areas such as controllable image generation (ControlNet[53]), image editing (DreamBooth[54]), image inpainting (SmartBrush[55]), and style transfer (StyleDrop[56]). Given the powerful image generation capabilities of the diffusion model in the tasks mentioned above, we introduce it to enhance landmark heatmap generation, aiming to tackle the challenges posed by FLD.

## III. METHODOLOGY

In this section, we introduce our proposed FAHCD-Net, as depicted in Fig. 3. We start by outlining the preliminaries of DDPM in Section III.A, followed by a comprehensive overview of FAHCD-Net in Section III.B. Detailed explanations of the FAHCD model and the SR loss are provided in Sections III.C and III.D, respectively.

## A. Preliminary

In the diffusion model, given an initial data $( \mathrm { e . g . }$ , image) distribution $x _ { 0 } \sim q ( x )$ , Gaussian noise is gradually added to the distribution. Given a predefined variance schedule $\{ \beta _ { t } \} _ { t = 1 } ^ { T }$ noise is added at each step to the previous data $x _ { t - 1 }$ as follows:

$$
q ( x _ { t } | x _ { t - 1 } ) = N ( x _ { t } ; \sqrt { 1 - \beta _ { t } } x _ { t - 1 } , \beta _ { t } I )\tag{1}
$$

Then, with the reparameterization trick, the noise distribution at any time-step t can be computed as $q ( x _ { t } | x _ { 0 } )$ :

$$
q ( x _ { t } | x _ { 0 } ) = \mathcal { N } ( \sqrt { \overline { { \alpha _ { t } } } } x _ { 0 } , ( 1 - \overline { { \alpha } } _ { t } ) I ) , \quad \overline { { \alpha } } _ { t } = \prod _ { i = 1 } ^ { t } ( 1 - \beta _ { i } )\tag{2}
$$

Based on this distribution, $x _ { t }$ at a specific time-step t can be computed directly from $x _ { 0 }$ as follows:

$$
x _ { t } = \sqrt { \bar { \alpha } _ { t } } x _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { t }\tag{3}
$$

where $\epsilon _ { t }$ follows a standard Gaussian distribution $\mathcal { N } ( 0 , I )$ , and $\epsilon _ { t } \in \mathbb { R } ^ { 3 \times h \times w }$

The reverse process $q ( x _ { t - 1 } | x _ { t } )$ in the diffusion model is designed to progressively reconstruct data from noise, leveraging a deep neural network to model the state transition from $x _ { t }$ to $x _ { t - 1 }$ . The distribution $p _ { \theta } ( x _ { t - 1 } | x _ { t } )$ , representing this transition, can be expressed as follows:

$$
p _ { \theta } ( x _ { t - 1 } | x _ { t } ) = \mathcal { N } ( x _ { t - 1 } ; \mu _ { \theta } ( x _ { t } , t ) , \Sigma _ { \theta } ( x _ { t } , t ) )\tag{4}
$$

where $\Sigma _ { \theta } ( x _ { t } , t )$ is usually a predefined constant related to the variance schedule [25], and $\mu _ { \theta } ( x _ { t } , t )$ is typically parameterized by a denoising network $\epsilon _ { \theta } ( x _ { t } , t )$ according to:

$$
\mu _ { \theta } ( x _ { t } , t ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( x _ { t } - \frac { 1 - \alpha _ { t } } { \sqrt { 1 - \overline { { \alpha } } _ { t } } } \epsilon _ { \theta } ( x _ { t } , t ) \right)\tag{5}
$$

Generally, the diffusion model is supervised by the $\mathcal { L } _ { \mathrm { s i m p l e } }$ loss function with respect to $\theta \colon$

$$
\mathcal { L } _ { \mathrm { s i m p l e } } = \sum _ { t = 1 } ^ { T } \left[ \| \epsilon _ { \theta } ( x _ { t } , t ) - \epsilon _ { t } \| _ { 2 } ^ { 2 } \right]\tag{6}
$$

![](images/4d2f2dd3e2be2aced3adbd84ad7df8a4745a5bd29459ee950cd6fc7f10e43c7c.jpg)  
Fig. 3. The overview of our proposed FAHCD-Net. (a) Training Process, (b) FAHCD model, (c) HFA module, and (d) Sampling Process. It consists of FAHCD model and SR loss. The FAHCD model integrates HFA modules to perform frequency decomposition and reconstruction, adaptively adjusting the contribution of different frequency components. SR loss aims to apply smoothness and continuity constraints to the generated heatmaps. Then, by integrating the FAHCD model and SR loss in a cascading manner, the proposed FAHCD-Net can achieve outstanding performance.

## B. FAHCD-Net Overview

The denoising process of FAHCD-Net takes facial image features x as inputs and uses conditional landmark heatmap h<sup>ˆ</sup> as conditional information, which can be defined as follows:

$$
p _ { \theta } ( h _ { 0 : T } | x , \hat { h } ) = p ( h _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( h _ { t - 1 } | h _ { t } , x , \hat { h } )\tag{7}
$$

where $h _ { t }$ denotes the noisy landmark heatmap correspond to time-step t, and $h _ { 0 }$ represents the ground-truth landmark heatmap.

The conditional distribution $p _ { \theta } ( h _ { t - 1 } | h _ { t } , x , \hat { h } )$ is given by:

$$
p _ { \theta } ( h _ { t - 1 } | h _ { t } , x , \hat { h } ) = \left\{ \begin{array} { l l } { \mathcal { N } ( \mu _ { \theta } ( h _ { 1 } , 1 , x , \hat { h } ) , 0 ) , } & { \mathrm { i f ~ } t = 1 } \\ { q ( h _ { t - 1 } | h _ { t } , \mu _ { \theta } ( h _ { t } , t , x , \hat { h } ) ) , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{8}
$$

Following the DDIM [57], we employ a deterministic generation process:

$$
\mu _ { \theta } ( h _ { t } , t , x , \hat { h } ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( h _ { t } - \sqrt { 1 - \alpha _ { t } } \epsilon _ { \theta } ( h _ { t } , t , x , \hat { h } ) \right)\tag{9}
$$

where $\epsilon _ { \theta }$ denotes the proposed FAHCD model. Note that FAHCD-Net does not predict the noise directly but instead predicts the intermediate heatmaps. And the heatmap prediction formula can be derived as:

$$
\tilde { h } _ { 0 } = \frac { h _ { t } - \sqrt { 1 - \bar { \alpha _ { t } } } \epsilon _ { t } } { \sqrt { \bar { \alpha _ { t } } } }\tag{10}
$$

where $\tilde { h } _ { 0 }$ denotes the generated landmark heatmap.

In the experiments, we found that using MSE or AWingLoss [24], [58], [59] to optimize the FAHCD-Net led to local mutations and redundant high-frequency noise in the heatmap. To tackle this issue, we further propose SR loss to guarantee the smoothness and continuity of the generated heatmaps. Accordingly, the complete loss function is defined as follows:

$$
\mathcal { L } = \mathcal { L } _ { a w i n g } + \lambda \mathcal { L } _ { s r }\tag{11}
$$

where λ is used to balance these two losses.

Inspired by [60], we cascade multiple FAHCD-Nets across three stages to gradually enhance the effectiveness of generated landmark heatmap, which can be formulated as follows:

$$
\tilde { h } _ { 0 } ^ { s } = \mathrm { F A H C D - N e t } ^ { s } ( h _ { t } , t , x , \hat { h } ) , s \in \{ 0 , 1 , 2 \}\tag{12}
$$

where s denotes the index of the current stage, with $s \in$ {0, 1, 2}. When $s = 0 , \hat { h } ^ { s }$ represents the conditional landmark heatmap generated by current FLD models (e.g., SHN [21]). For $s > 0 , \hat { h } ^ { s }$ corresponds to the generated landmark heatmap from the previous stage (i.e., $\bar { h ^ { s } } ~ = ~ \tilde { h } _ { 0 } ^ { s - 1 } )$ . Therefore, by cascading the FAHCD-Net, richer conditional facial priors can be leveraged to generate more accurate landmark heatmaps, thus achieving precise landmark detection.

Next, we will introduce the proposed FAHCD model and SR loss in detail.

## C. FAHCD model

Currently, most DPMs [25], [26], [50] are based on the U-net framework. When applied to FLD tasks, it is necessary to generate landmark heatmaps step-by-step by learning the pixel-level difference from Gaussian noise to the target heatmap. However, during this generation process, the frequency information of images undergoes significant changes(as shown in Fig.4), which profoundly degrades the quality of the generated heatmaps for landmark detection.

![](images/a897457eeca2394e69cf00994495689423daa9221646d50a5ee6ee1e64f85ad3.jpg)  
Fig. 4. Comparison of heatmaps and their corresponding Power Spectral Density (PSD) maps. The first, second and third rows show landmark heatmap and PSD maps generated by ground-truth, FAHCD-Net w/o HFA module and SR loss, and FAHCD-Net, respectively. From the results, it can be observed that the heatmap generated by using FAHCD-Net w/o HFA module and SR loss produces a significant amount of additional high-frequency noise compared to the ground-truth. These noises appear as strong high-frequency components in the PSD map, affecting the precise localization of landmarks. In contrast, FAHCD-Net effectively suppresses unnecessary high-frequency noise by combining HFA module and SR loss, thereby improving the quality of the heatmap and the accuracy of landmark detection.

In image processing, different frequency components typically reflect specific characteristics of an image: highfrequency signals represent rapidly changing features such as edges and textures, while low-frequency signals correspond to more gradually varying features, such as smooth regions. For facial images, high-frequency signals are primarily concentrated in key facial feature areas (e.g., eyes, nose, mouth) and facial contours, whereas low-frequency signals are more likely to appear in smooth facial regions and the background.

Building on this, we further analyzed the frequency components of the ground-truth heatmap $h _ { 0 }$ and the generated heatmap $\tilde { h } _ { 0 } .$ , revealing significant differences between the two (as shown in Fig. 4). Specifically, low-frequency information remains stable, but high-frequency information increases in the generated heatmap compared to the ground-truth heatmap. This suggests that while diffusion models preserve the overall structure of the image, they also introduce additional highfrequency information during the generation process, which may cause the following problems: (1) Peak blurring: The clarity of the Gaussian distribution is disrupted, reducing localization accuracy. (2) Noise overfitting: The model may overfit the high-frequency noise in the training data, leading to diminished generalization capability. (3) Noise in smooth areas: Even in the smooth regions of the heatmap, the presence of noise can interfere with the accuracy of landmark localization. Irrelevant noise in these areas may mislead the model into producing incorrect activations in non-target regions. Therefore, an ideal landmark heatmap should exhibit a clear Gaussian peak at key positions and remain smooth in other regions. To achieve this, we propose the FAHCD model. FAHCD model is actually an hourglass unit, which is a multi-scale feature framework. So we introduce Hierarchical Frequency Adaptive (HFA) module at each scale of the FAHCD model.

In HFA module, the multi-scale features undergo frequency decomposition and reconstruction to better align with the ground-truth heatmap’s frequency distribution. For frequency decomposition, the HFA module uses multi-scale Gaussian blur (i.e., Gaussian kernels with different sizes and standard deviations) as low-pass filters to achieve hierarchical frequency processing and feature filtering. For frequency reconstruction, we introduce a trainable parameter W to combine decomposed frequency components. The design of the HFA module is illustrated in Fig. 3(c).

Assume that ${ \mathcal { K } } _ { d \times d } ^ { \sigma }$ represents a 2D Gaussian kernel with a kernel size of d and a standard deviation of $\sigma .$ . Hence, a Gaussian blur process can be formulated as:

$$
\mathcal { F } _ { d } ^ { \sigma } = \mathcal { K } _ { d \times d } ^ { \sigma } * \mathcal { F } _ { i n }\tag{13}
$$

where ∗ denotes the convolution operation, and $\begin{array} { r l } { d } & { { } \in } \end{array}$ $\{ 3 , 5 , 7 , \ldots \} . \mathcal { F } _ { i n }$ signifies the input features and $\mathcal { F } _ { d } ^ { \sigma }$ represents the output after Gaussian blurring.

Three different Gaussian kernels with varying sizes and standard deviations are applied to filter varying frequency components (i.e., high-frequency component $\mathcal { F } _ { \vert }$ , midfrequency component $\mathcal { F } _ { \dagger }$ and low-frequency component $\mathcal { F } _ { \sharp } )$ They will be concatenated with $\mathcal { F } _ { i n }$ and then fused to obtain the filtered feature $\mathcal { F } _ { o u t }$

$$
[ \mathcal { F } _ { i n } , \mathcal { F } _ { | } , \mathcal { F } _ { \dagger } , \mathcal { F } _ { \dagger } ] = [ \mathcal { F } _ { i n } , \mathcal { F } _ { i n } - \mathcal { F } _ { 3 } ^ { \sigma _ { 1 } } , \mathcal { F } _ { 3 } ^ { \sigma _ { 1 } } - \mathcal { F } _ { 5 } ^ { \sigma _ { 2 } } , \mathcal { F } _ { 5 } ^ { \sigma _ { 2 } } - \mathcal { F } _ { 7 } ^ { \sigma _ { 3 } } ]\tag{14}
$$

$$
\mathcal { F } _ { o u t } = W [ \mathcal { F } _ { i n } , \mathcal { F } _ { | } , \mathcal { F } _ { \dagger } , \mathcal { F } _ { \ddagger } ] ^ { T }\tag{15}
$$

where $W = [ w _ { 1 } , w _ { 2 } , w _ { 3 } , w _ { 4 } ]$ is a trainable parameter that can be optimized during the training process.

Since the FAHCD is based on the U-net framework, which is the encoder-decoder structure, $\mathcal { F } _ { o u t }$ will be concatenated with the corresponding original encoder output into the Decoder. Therefore, the reconstruction frequency information is introduced to eliminate redundant noise, thus modeling more effective facial structure and obtaining more accurate landmark detection. The complete training process is presented in Alg. 1.

## D. Smoothness Regularization Loss

In addition to proposing the FAHCD model to incorporate frequency information for modeling facial structure and generating landmark heatmaps, we also introduce the SR loss to ensure the smoothness and continuity of the generated heatmaps.

SR loss can avoid excessive high-frequency noise interference with either Total Variation (TV) regularization or Heatmap Gradient (HG) regularization. The former is typically used to remove noise while preserving the overall structure of the image. By penalizing large changes in the output heatmap, it suppresses the generation of high-frequency noise while retaining the main edge information. The form of TV regularization is as follows:

$$
\mathcal { L } _ { T V } ( \tilde { h } _ { 0 } ) = \sum _ { i , j } \left( ( \tilde { h } _ { 0 } ^ { i + 1 , j } - \tilde { h } _ { 0 } ^ { i , j } ) ^ { 2 } + ( \tilde { h } _ { 0 } ^ { i , j + 1 } - \tilde { h } _ { 0 } ^ { i , j } ) ^ { 2 } \right)\tag{16}
$$

Algorithm 1 FAHCD-Net Training Process   
Require: Max Diffusion Step T, Image Set $q ( x )$ , ground-truth   
heatmap $h ,$ conditional heatmap h<sup>ˆ</sup> and hyper-parameter   
$s = 0 . 0 0 8$   
1: for each input sample $x _ { 0 } \in q ( x _ { 0 } )$ do   
2: Sample $\epsilon \sim \mathcal { N } ( 0 , I )$   
3: Sample $t \sim \mathcal { U } ( \{ 1 , \cdots , T \} )$   
4: $\begin{array} { r } { \beta _ { t } = 1 - \frac { c o s ( ( { \dot { t } } / T + s ) / ( 1 + \dot { s } ) \cdot \pi / 2 ) } { c o s ( s \cdot \pi / 2 ) } } \end{array}$   
5: $\alpha _ { t } = 1 - \beta _ { t }$   
6: $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 0 } ^ { t } \alpha _ { s } } \end{array}$   
7: Calculate $h _ { t } = \sqrt { \bar { \alpha } _ { t } } h + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon$   
8: Predict noise: $\epsilon _ { \theta } ( h _ { t } , t , x , \tilde { h } )$   
9: Predict output heatmap $\tilde { h } _ { 0 } { : }$   
$\begin{array} { r } { \mu _ { \theta } ( h _ { t } , t , x , \hat { h } ) = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( h _ { t } - \sqrt { 1 - \alpha _ { t } } \epsilon _ { \theta } ( h _ { t } , t , x , \hat { h } ) \right) } \end{array}$   
$\begin{array} { r } { \tilde { h } _ { 0 } = \frac { h _ { t } - \sqrt { 1 - \bar { \alpha _ { t } } } \epsilon _ { t } } { \sqrt { \bar { \alpha } _ { s } } } } \end{array}$   
10: Calculate the heatmap loss term $\mathcal { L } _ { a w i n g }$   
11: Calculate the Smoothness Regularization term $\mathcal { L } _ { s r }$   
12: Update with $\mathcal { L } = \mathcal { L } _ { a w i n g } + \lambda \mathcal { L } _ { s r }$   
13: Until convergence   
14: end for

![](images/1515b827a230e72ade68a90643d1465228dab94e0953356f7aea3be1facbdb19.jpg)  
Fig. 5. The experimental results of our proposed FAHCD-Net are shown in the figure. These examples demonstrate the performance of the model in various scenarios, illustrating its effectiveness in handling complex tasks. The first row is images in facial expressions and large poses. The second row is in heavy occlusion. The third row is in illumination, and the last row is blur images.

where i, j represent the pixel indexes.

HG regularization directly adds constraints on the heatmap gradients, penalizing excessive gradient changes, which can be formulated as:

$$
\mathcal { L } _ { H G } ( \tilde { h } _ { 0 } ) = \sum _ { i , j } \left( ( \frac { \partial \tilde { h } _ { 0 } } { \partial x } ) ^ { 2 } + ( \frac { \partial \tilde { h } _ { 0 } } { \partial y } ) ^ { 2 } \right)\tag{17}
$$

By computing the squared derivatives of the heatmap $\tilde { h } _ { 0 }$ in both the horizontal and vertical directions, HG regularization suppresses large variations, ensuring the smoothness of the generated landmark heatmap. Note that our proposed SR loss can be either $\mathcal { L } _ { T V }$ or ${ \mathcal { L } } _ { H G } ,$ , and we have reported corresponding experimental results in Section IV.

## IV. EXPERIMENTS

In this section, we conduct experiments on our FAHCD-Net across multiple datasets to validate its effectiveness. Fig. 5

presents qualitative results across four challenging scenarios: large facial expressions and poses, heavy occlusion, illumination variation, and blur. The details are given as follows.

## A. Datasets and Experimental Settings

Datasets. We evaluate the proposed methods on public datasets including 300W [61], COFW [62], WFLW [63] and AFLW [64].

300W(68 landmarks): is a combination of LFPW, HELEN, AFW, and IBUG, with each face annotated with 68 landmarks. Typically, the training set consists of all images from AFW, as well as the training images from LFPW and HELEN, resulting in a total of 3,148 images. The test set includes all images from IBUG and the test images from LFPW and HELEN, amounting to 689 images. The LFPW and HELEN test images are designated as the Common Subset, whereas the IBUG images constitute the Challenging Subset.

COFW(29 landmarks): primarily focuses on images with occlusions. Typically, the training set consists of 845 faces from the LFPW training set and an additional 500 heavily occluded faces. The test set includes 507 images with significant occlusion. Evaluations are conducted on 29 landmarks.

WFLW(98 landmarks): consists of 10,000 facial images, with 7,500 allocated for training and 2,500 for testing, each annotated with 98 landmarks. Beyond landmark annotations, it provides extensive attribute information, including occlusion, pose, makeup, illumination, blur, and expression.

AFLW (19 landmarks): This dataset consists of 25,993 high-quality facial images, each annotated with 19 landmarks. It includes a diverse range of scenarios, featuring both indoor and outdoor scenes. Moreover, it covers challenging cases such as side profiles and various rotation angles, making it ideal for evaluating robust facial landmark detection methods.

Evaluation metric. Normalized Mean Error (NME) is a commonly used metric for evaluating landmark detection performance, which is defined as:

$$
\mathrm { N M E } ( P , \hat { P } ) = \frac { 1 } { N _ { P } } \sum _ { i = 1 } ^ { N _ { P } } \frac { \| p _ { i } - \hat { p } _ { i } \| _ { 2 } } { d }\tag{18}
$$

where P and $\hat { P }$ represent the ground-truth and predicted coordinates, respectively, while $p _ { i }$ and $\hat { p } _ { i }$ correspond to the coordinates of the i-th landmark in the ground-truth and prediction. $N _ { p }$ represents the total number of landmarks, and d serves as the reference distance to normalize the absolute errors. Commonly, the inter-ocular distance (distance between the outer corners of the eyes) or the inter-pupil distance (distance between the centers of the pupils) is used as the normalization factor.

Failure Rate (FR) is another important metric for evaluating landmark detection performance, which is defined as:

$$
\mathrm { F R } ( P , \hat { P } ) = \frac { 1 } { N _ { P } } \sum _ { i = 1 } ^ { N _ { P } } { \mathcal { H } \left( \frac { \lVert p _ { i } - \hat { p } _ { i } \rVert _ { 2 } } { d } > \tau \right) }\tag{19}
$$

where the threshold $\tau$ defines the maximum normalized error allowed for a successful prediction, and $\nVdash ( \cdot )$ is an indicator function that returns 1 if the condition is true and 0 otherwise.

FR calculates the proportion of landmarks whose normalized error exceeds the threshold, providing insight into the robustness of the detection method in handling difficult cases.

Implementation Details. The proposed FAHCD-Net is trained on an NVIDIA GeForce RTX 4090 GPU. For data preprocessing, all input images are uniformly cropped and resized to a spatial resolution of $2 5 6 \times 2 5 6$ pixels for both training and evaluation.

To enhance the robustness of the model, various image augmentation strategies are applied to the training set samples: (1) random clockwise or counterclockwise rotation between 0° and $1 8 ^ { \circ } { \mathrm { : } }$ ; (2) random cropping of 0-5%; (3) random grayscale conversion for 20% of the samples; (4) random blurring for 30% of the samples; (5) random occlusion for 40% of the samples; (6) random horizontal flipping for 50% of the samples.

The training process utilizes the Adam optimizer, starting with an initial learning rate of $1 \times 1 0 ^ { - 3 }$ . This rate is decayed by a factor of 0.9 at the 80th, 150th, and 200th epochs. The model is trained over 300 epochs with a batch size of 12.

In terms of hyperparameter settings, ω for $\mathcal { L } _ { h e a t m a p }$ is set to 14, θ to 0.5, ϵ to 1, and α to 2.1. The weight for the smoothness regularization term is set to $\lambda = 3 \times 1 0 ^ { - 7 }$ . When processing the dataset, a Gaussian kernel $( \sigma = 3 )$ is empirically used to generate the landmark heatmaps corresponding to the images.

## B. Evaluations under Normal Circumstances

Under normal circumstances, we primarily conducts comparative experiments on the 300W Common Subset, 300W Fullset and AFLW-frontal subset. These subsets consist mostly of facial images captured in favorable conditions, with clear images and minimal pose variations, illuminations, occlusion, or noise interference.

Tab. I presents the comparative experimental results between the proposed model and state-of-the-art methods. It can be observed that our proposed FAHCD-Net can achieve an $\mathrm { N M E } _ { \mathrm { i o } }$ of 2.51 on the 300W Common Subset, outperforming other methods [65], [66], [45], [48], [38], [67]. On the 300W Fullset, it can attaine an $\mathrm { N M E } _ { \mathrm { i o } }$ of 2.99, demonstrating commendable performance. Furthermore, on the AFLWfrontal subset, FAHCD-Net achieved an $\mathrm { N M E _ { d i a g } }$ of 1.17 (Tab.IV), outperforming other methods [68], [69], [63], [46], [70], [21]. These results demonstrate that the FAHCD-Net model achieves a relatively advanced level of performance under normal circumstances. This can be attributed to the characteristics of the diffusion model underlying FAHCD-Net. Under normal circumstances, the diffusion model fully leverages its ability to learn data distributions, progressively generating high-quality heatmaps and effectively FLD.

## C. Evaluation of Robustness against Occlusion

Under normal conditions, most models perform well. However, when facial images are heavily occluded, many existing methods struggle to maintain their effectiveness. To assess the performance of our FAHCD-Net under these conditions, we conducted experiments on datasets that contain significant occlusions, such as the COFW dataset, the 300W Challenging Subset, and the WFLW dataset.

TABLE I  
COMPARISONS WITH STATE-OF-THE-ART METHODS ON THE 300WDATASET. THE ERROR (NME) IS NORMALIZED BY THE INTER-OCULARDISTANCE. ◦ AND ⋄ DENOTE HEATMAP REGRESSION AND COORDINATEREGRESSION METHODS, RESPECTIVELY.
<table><tr><td>Method</td><td>Common</td><td>Challenging</td><td>Full</td></tr><tr><td> LAB (CVPR18) [63]</td><td>2.98</td><td>5.19</td><td>3.49</td></tr><tr><td> AWing (ICCV19) [71]</td><td>2.72</td><td>4.52</td><td>3.07</td></tr><tr><td> LUVI (CVPR20) [69]</td><td>2.76</td><td>5.16</td><td>3.23</td></tr><tr><td> ADNet (ICCV21)[65]</td><td>2.53</td><td>4.58</td><td>2.93</td></tr><tr><td>STAR (CVPR23)[48] ODN (CVPR19)[72]</td><td>2.52 3.56</td><td>4.32</td><td>2.87</td></tr><tr><td> DAG (ECCV20)[73]</td><td>2.62</td><td>6.67 4.77</td><td>4.17 3.04</td></tr><tr><td>◇ LGSA (TMM21)[74]</td><td>2.92</td><td>5.16</td><td>3.36</td></tr><tr><td>◇ PIPNet (IJCV21) [75]</td><td>2.78</td><td>4.89</td><td></td></tr><tr><td>◇ SLPT (CVPR22)[45]</td><td>2.75</td><td>4.90</td><td>3.19</td></tr><tr><td>◇ GlomFace (CVPR22)[76]</td><td>2.79</td><td>4.87</td><td>3.17</td></tr><tr><td>◇ DTLD (CVPR23) [77]</td><td>2.59</td><td></td><td>3.20</td></tr><tr><td>◇ ATF (TMM23) [78]</td><td>2.75</td><td>4.50</td><td>2.96</td></tr><tr><td>◇EfficientFAN (TNNLS23)[79]</td><td>2.98</td><td>4.89</td><td>3.17</td></tr><tr><td>◇ PicasoNet (TNNLS23)[80]</td><td>3.03</td><td>5.21</td><td>3.42</td></tr><tr><td></td><td>3.97</td><td>5.81</td><td>3.58</td></tr><tr><td> Lite-HRNet (ICIP23)[81]</td><td></td><td>6.89</td><td>4.54</td></tr><tr><td>◇ Liang et al. (CVPR24)[38] FAHCD-Net (ours)</td><td>2.68 2.51</td><td>4.86 4.49</td><td>3.10 2.99</td></tr></table>

TABLE II

COMPARISONS WITH STATE-OF-THE-ART METHODS ON THE COFWDATASET. THE ERROR (NME) IS NORMALIZED BY THE INTER-PUPILDISTANCE. ◦ AND ⋄ DENOTE HEATMAP REGRESSION AND COORDINATEREGRESSION METHODS, RESPECTIVELY.
<table><tr><td>Method</td><td> $\overline { { { \bf N } { \bf M } { \bf E } _ { \mathrm { i p } } } }$ </td><td> $\overline { { \mathbf { F } \mathbf { R } _ { \mathrm { i p } } ^ { \mathrm { 1 0 } } } }$ </td></tr><tr><td> LAB (CVPR18)[63]</td><td>5.58</td><td>2.76</td></tr><tr><td> DCFE (ECCV18)[82]</td><td>5.27</td><td>7.29</td></tr><tr><td>AWing (ICCV19)[71]</td><td>4.94</td><td>0.99</td></tr><tr><td>o ADNet (ICCV21)[65]</td><td>4.68</td><td>0.59</td></tr><tr><td>STAR (CVPR23)[48]</td><td>4.62</td><td>0.79</td></tr><tr><td>DSAT (PR24)[68]</td><td>4.74</td><td>0.00</td></tr><tr><td> ODN (CVPR19)[72]</td><td>5.30</td><td>1.78</td></tr><tr><td> MMDN (TNNLS22)[83]</td><td>5.01</td><td>1.78</td></tr><tr><td>◇ GlomFace (CVPR22)[76]</td><td>4.37</td><td>1.56</td></tr><tr><td>◇ SLPT (CVPR22) [45]</td><td>4.79</td><td>1.18</td></tr><tr><td>◇ DSLPT-R50 (TPAMI23)[84]</td><td>4.81</td><td>1.18</td></tr><tr><td>FAHCD-Net (ours)</td><td>4.65</td><td>0.20</td></tr></table>

On COFW dataset, FAHCD-Net achieves an ${ \bf N M E } _ { \mathrm { i p } }$ of 4.65 (Tab. II), outperforming most other methods [68], [48], [45], [65], [24], [82], [63], [86]. On the 300W Challenging Subset, FAHCD-Net attained an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.49, as shown in Tab. I, also surpassing state-of-the-art methods [67], [38], [48], [45], [66], [65], [24], [69]. Furthermore, as presented in Tab. III, FAHCD-Net exhibited exceptional performance on the Make-Up Subset and Occlusion Subset of the WFLW dataset. These outstanding performances can be attributed to FAHCD-Net’s strong robustness to noise and its exceptional handling of frequency information. Specifically, FAHCD-Net enhances noise resistance through the generative process of the diffusion model while effectively extracting structural information from low-frequency signals using the HFA module. These attributes collectively enable FAHCD-Net to excel in occlusion scenarios.

TABLE III  
COMPARISONS WITH STATE-OF-THE-ART METHODS ON THE WFLW DATASET. NME IS NORMALIZED BY THE INTER-OCULAR DISTANCE. ◦ AND ⋄ DENOTE HEATMAP REGRESSION AND COORDINATE REGRESSION METHODS, RESPECTIVELY.
<table><tr><td>Method</td><td>Testset</td><td>Pose Subset</td><td>Expression Subset</td><td>Illumination Subset</td><td>Make-Up Subset</td><td>Occlusion Subset</td><td>Blur Subset</td></tr><tr><td>LAB (CVPR18)[63]</td><td>5.27</td><td>10.24</td><td>5.51</td><td>5.23</td><td>5.15</td><td>6.79</td><td>6.32</td></tr><tr><td>Wing (CVPR18)[71]</td><td>4.99</td><td>8.43</td><td>5.21</td><td>4.88</td><td>5.26</td><td>6.21</td><td>5.81</td></tr><tr><td>HRNet (TPAMI20)[46]</td><td>4.60</td><td>7.86</td><td>4.78</td><td>4.57</td><td>4.26</td><td>5.42</td><td>5.36</td></tr><tr><td>AWing (ICCV19)[24]</td><td>4.36</td><td>7.38</td><td>4.58</td><td>4.32</td><td>4.27</td><td>5.19</td><td>4.96</td></tr><tr><td>LUVLi (CVPR20)[69]</td><td>4.37</td><td>7.56</td><td>4.77</td><td>4.30</td><td>4.33</td><td>5.29</td><td>4.94</td></tr><tr><td>ADNet (ICCV21)[65]</td><td>4.14</td><td>6.96</td><td>4.38</td><td>4.09</td><td>4.05</td><td>5.06</td><td>4.79</td></tr><tr><td>SLPT (CVPR22)[45]</td><td>4.12</td><td>6.99</td><td>4.37</td><td>4.02</td><td>4.03</td><td>5.01</td><td>4.79</td></tr><tr><td>LDEQ (CVPR23)[47]</td><td>3.92</td><td>6.86</td><td>3.94</td><td>4.17</td><td>3.75</td><td>4.77</td><td>4.59</td></tr><tr><td>FRA (CVPR24)[67]</td><td>4.11</td><td></td><td></td><td></td><td></td><td>-</td><td></td></tr><tr><td>◇DAG (ECCV20)[73]</td><td>4.21</td><td>7.36</td><td>4.49</td><td>4.12</td><td>4.05</td><td>4.98</td><td>4.82</td></tr><tr><td>◇PIPNet (IJCV21)[75]</td><td>4.31</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>◇ MMDN (TNNLS22)[83]</td><td>4.87</td><td>7.71</td><td>4.79</td><td>4.61</td><td>4.72</td><td>6.17</td><td>5.72</td></tr><tr><td>◇GlomFace (CVPR22)[76]</td><td>4.81</td><td>8.71</td><td></td><td></td><td></td><td>5.14</td><td></td></tr><tr><td>◇DTLD (CVPR22)[77]</td><td>4.08</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>◇ ATF (TMM23)[78]</td><td>4.50</td><td>7.54</td><td>4.65</td><td>4.45</td><td>4.20</td><td>5.30</td><td>5.19</td></tr><tr><td>◇EfficientFAN (TNNLS23)[79]</td><td>4.54</td><td>8.20</td><td>4.87</td><td>4.39</td><td>4.54</td><td>5.42</td><td>5.04</td></tr><tr><td>◇PicasoNet (TNNLS23)[80]</td><td>4.82</td><td>8.61</td><td>5.14</td><td>4.73</td><td>4.68</td><td>5.91</td><td>5.56</td></tr><tr><td>◇Lite-HRNet (ICIP23)[81]</td><td>5.58</td><td>9.79</td><td>6.13</td><td>5.44</td><td>5.87</td><td>6.57</td><td>6.05</td></tr><tr><td>◇ FAHCD-Net (ours)</td><td>4.02</td><td>7.09</td><td>4.59</td><td>4.25</td><td>4.10</td><td>5.18</td><td>4.70</td></tr></table>

TABLE IV

COMPARISONS WITH STATE-OF-THE-ART METHODS ON THE AFLWDATASET. ◦ AND ⋄ DENOTE HEATMAP REGRESSION AND COORDINATEREGRESSION METHODS, RESPECTIVELY.
<table><tr><td>Method</td><td> $\mathbf { N M E } _ { \mathrm { d i a g } } \ ( \mathrm { f u l l } )$ </td><td> $\mathbf { N M E } _ { \mathrm { d i a g } }$  (frontal)</td><td> $\mathbf { N M E } _ { \mathrm { b o x } } \ \mathrm { ( f u l l ) }$ </td></tr><tr><td>SHN (CVPR17)[21]</td><td>2.46</td><td>1.92</td><td>3.67</td></tr><tr><td>SAN (CVPR18)[70]</td><td>1.91</td><td>1.85</td><td>4.04</td></tr><tr><td>LAB (CVPR18)[63]</td><td>1.85</td><td>1.62</td><td></td></tr><tr><td>HRNet (TPAMI20)[46]</td><td>1.57</td><td>2.46</td><td></td></tr><tr><td>Wing (CVPR17)[71]</td><td></td><td></td><td>3.56</td></tr><tr><td>LUVLi (CVPR20)[69]</td><td>1.39</td><td>1.19</td><td>2.28</td></tr><tr><td>DSAT (PR24)[68]</td><td>1.35</td><td>1.17</td><td></td></tr><tr><td>◇DTLD+ (CVPR22)[85]</td><td>1.37</td><td></td><td></td></tr><tr><td>◇ DSLPT (TPAMI23)[84]</td><td>1.38</td><td></td><td></td></tr><tr><td>◇ FAHCD-Net (ours)</td><td>1.30</td><td>1.17</td><td>2.08</td></tr></table>

D. Evaluation of Robustness against Large Poses and Expressions

Facial images with large poses and expressions present significant challenges in facial landmark detection tasks. To evaluate the proposed model’s detection capability under these circumstances, experiments are conducted on the AFLW-full dataset, WFLW dataset and 300W Challenging Subset.

On the AFLW-full dataset, our proposed FAHCD-Net can achieve an $\mathrm { N M E _ { d i a g } }$ of 1.30 and an $\mathrm { N M E } _ { \mathrm { b o x } }$ of 2.08, as shown in Tab. IV, outperforming other methods [68], [85], [69], [71], [87], [88]. On the 300W Challenging Subset, as shown in Tab. I, FAHCD-Net can attain an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.49, demonstrating great performance. Additionally, on the WFLW dataset’s Pose Subset and Expression Subset, as presented in Tab. III, FAHCD-Net can also exhibite exceptional results. These results indicate that FAHCD-Net, leveraging the capabilities of the diffusion model to learn statistical properties and simulate the initial data distribution, effectively adapts to variations in facial poses. Meanwhile, the HFA module further utilizes frequency information to capture global structural characteristics from low-frequency information. This enables FAHCD-Net to maintain accurate FLD in scenarios involving pose and expression variations.

TABLE V  
COMPARISON OF DIFFERENT CONDITIONAL HEATMAPS ON THE 300W CHALLENGING SUBSET. NME IS NORMALIZED BY THE INTER-OCULAR DISTANCE.
<table><tr><td>Method</td><td>Stage I</td><td>Stage II</td><td>Stage III</td></tr><tr><td>Mean Shape</td><td>4.81</td><td>4.73</td><td>4.69</td></tr><tr><td>FAHCD-Net (ours)</td><td>4.48</td><td>4.33</td><td>4.29</td></tr></table>

## E. Evaluation of Robustness against Illumination and Blur

This part focuses on facial images with varying illumination and blur. To evaluate the proposed model’s detection capabilities under these conditions, experiments are conducted on WFLW dataset and the 300W Challenging Subset. On the 300W Challenging Subset, as shown in Tab. I, FAHCD-Net can achieve an $\mathrm { N M E } _ { \mathrm { i o } }$ of 4.29, demonstrating leading performance. Additionally, on the WFLW dataset’s Illumination Subset and Blur Subset, as presented in Tab. III, FAHCD-Net also exhibites outstanding results. This improvement performance can be attributed to the HFA module and SR loss effectively suppressing high-frequency noise present in the data and generated by the diffusion model. By aligning the frequency distribution of the generated heatmaps with the ground-truth, the module ensures the generation of clean and accurate landmark heatmaps.

## F. Self Evaluation

1) Mean shape as Conditional Heatmap: To verify whether the proposed FAHCD-Net relies heavily on the quality of the conditional heatmap, we replace the SHN-generated heatmaps [21] with a mean shape prior, which is obtained by averaging a large number of facial landmark annotations.

Tab. V reports the results on the 300W Challenging subset. Replacing the learned heatmaps with the mean shape leads to a noticeable performance drop. However, the degradation remains moderate. This indicates that the proposed framework does not strictly rely on high-quality heatmap initialization. These results can be attributed to the fact that, although the mean shape lacks instance-specific structural details, it still provides a coarse yet stable spatial prior, which can guide the model toward reasonable landmark localization. Meanwhile, this alternative significantly simplifies the training pipeline and removes the reliance on an additional heatmap generator, making it more practical in real-world scenarios where such prior models are unavailable.

![](images/be1df647e147ad51d7194b54b5e50b109c876e3530687e44cd654474aaa8049c.jpg)  
Fig. 6. The process of frequency domain visualization and reconstruction using the Fourier Transform.

2) Frequency Component Analysis: In this section, we evaluate the role of frequency decomposition and reconstruction in improving the model’s detection accuracy. We aim to validate the impact of these operations from the following perspectives.

Frequency Domain Visualization. To better understand the influence of frequency components on model performance, we first apply a Fourier Transform to both the input and output heatmaps, converting them from the spatial domain to the frequency domain. The Fourier Transform reveals the distribution of the image across different frequencies, helping us distinguish between low-frequency smooth areas and highfrequency detailed regions in the image. The Fourier Transform formula is given by:

$$
F _ { h } ( u , v ) = \iint h ( x , y ) e ^ { - i 2 \pi ( u x + v y ) } d x d y\tag{20}
$$

where $h ( x , y )$ is the heatmap in the spatial domain, $F _ { h } ( u , v )$ is the heatmap in the frequency domain, and u and v are the coordinates in the frequency domain. We then calculate the Power Spectral Density (PSD) to gain further insight into the distribution of frequency components. For better visualization, we apply a logarithmic transformation $\log ( 1 + | F _ { h } ( u , v ) | ^ { 2 } )$ to the PSD values.

Fig.6(b) presents the frequency domain visualization results after applying the Fourier Transform. Generally, the central part of the image corresponds to low-frequency components, reflecting the smooth regions of the image (such as facial contours, skin, and large structures). The edges of the image correspond to high-frequency components, highlighting fine details (e.g., the contours of the eyes, mouth, and subtle facial textures).

To investigate the influence of different frequency components on image details, we employ low-pass and highpass filters on the frequency-domain representations of images obtained through Fourier transformation. The effects are then analyzed by applying an inverse Fourier Transform to reconstruct the filtered images.

As shown in Fig.6(e) and (f), the low-pass filter removes high-frequency information, such as fine details and edges, retaining primarily smooth, low-frequency components. This results in images with diminished sharpness and a lack of intricate details. Conversely, the high-pass filter suppresses low-frequency components while preserving high-frequency details, such as textures and edges, thereby enhancing the prominence of fine structures and sharp transitions in the image. This comparative analysis demonstrates the distinct roles of low- and high-frequency components in defining image features.

These operations help us analyze the influence of frequency components on the heatmap generation process. They also validate the critical role that frequency domain information plays in enhancing the accuracy of landmark detection. By identifying the effects of different frequency bands on the heatmap, we gain insights into how specific frequency components contribute to model performance in facial landmark detection.

Frequency Correlation Analysis. To further evaluate the relationship between frequency components and facial landmark localization accuracy, we quantify the energy of the frequency components and analyze its correlation with the model’s performance. We use frequency energy as a metric to quantify the frequency components, which is calculated as follows:

$$
E = \sum _ { u , v } | F _ { h } ( u , v ) | ^ { 2 }\tag{21}
$$

where E represents the energy of the frequency components. In the experiment, to distinguish between high-frequency and low-frequency energy, we categorize the frequency components based on their position in the frequency spectrum. Specifically, we calculate the distance $D ( u , v )$ of the frequency component (u, v) from the center of the spectrum:

$$
D ( u , v ) = \sqrt { u ^ { 2 } + v ^ { 2 } }\tag{22}
$$

Next, using a predefined distance threshold $D _ { \mathrm { t h r e s h o l d } } .$ , we classify the frequency components into high-frequency and low-frequency components. In this experiment, the threshold is set to 20% of the spectrum size. Based on this criterion, we divide the spectrum into high-frequency and low-frequency regions, and compute the corresponding energies for each:

$$
{ \cal E } _ { \mathrm { l o w } } = \sum _ { D ( u , v ) \leq D _ { \mathrm { t h r e s h o l d } } } | F _ { h } ( u , v ) | ^ { 2 }\tag{23}
$$

$$
{ \cal E } _ { \mathrm { h i g h } } = \sum _ { D ( u , v ) > D _ { \mathrm { t h r e s h o l d } } } | F _ { h } ( u , v ) | ^ { 2 }\tag{24}
$$

After calculating the low-frequency energy $E _ { \mathrm { l o w } }$ and highfrequency energy $E _ { \mathrm { h i g h } } ,$ , we compare these values with the localization error in the facial landmark detection task. The experimental results of frequency decomposition, visualization, and energy calculation are shown in Fig.7. From the figure, it can be observed that when using the diffusion model-based neural network alone (FAHCD model w/o HFA module), the generated images exhibit noticeable high-frequency noise, which interferes with the accurate localization of landmarks, leading to a decrease in accuracy. However, after introducing the HFA module and the SR loss function, the high-frequency noise is effectively suppressed.

The HFA module adaptively adjusts the balance of different frequency levels, significantly reducing unnecessary high-frequency components, while the SR loss further ensures that the generated landmark heatmaps are smoother and more continuous. After these optimizations, the model successfully improves landmark localization accuracy while managing high-frequency noise, indicating that controlling frequency components plays a crucial role in enhancing model performance.

![](images/cf7b408e7e16f72afb9525906d31db1333b969edd2440c9361af9301bf14507c.jpg)  
Fig. 7. Results of frequency component analysis. In each group of images, the first, second and third columns display the landmark heatmap, high-frequency component map and PSD (Power Spectral Density) map, respectively. And the fourth column contains the numerical values, which represent the energy of the high-frequency components and their corresponding ${ \mathrm { N M E } } _ { \mathrm { i o } }$ results. From the analysis, it is evident that, in the absence of the HFA module and SR loss, the model generates excessive high-frequency information, leading to inaccuracies in landmark detection. In contrast, FAHCD-Net effectively suppresses redundant high-frequency noise, resulting in more accurate FLD.

TABLE VI  
EFFECT OF MULTI-DATASET TRAINING ON THE 300W CHALLENGING SUBSET. NME IS NORMALIZED BY THE INTER-OCULAR DISTANCE.
<table><tr><td>Idx</td><td>300W</td><td>+COFW</td><td>+WFLW+AFLW</td><td> $\overline { { { \bf N M E } _ { \mathrm { { i o } } } } }$ </td></tr><tr><td>(1)</td><td>√</td><td></td><td></td><td>4.76</td></tr><tr><td>(2)</td><td>√</td><td>√</td><td></td><td>4.62</td></tr><tr><td>(3)</td><td>√</td><td>√</td><td>√</td><td>4.49</td></tr></table>

3) Different Smooth Regularization Item: To validate the effectiveness of different smooth regularization items, we employ TV regularization and HG regularization as $\mathcal { L } _ { s r }$ respectively on 300W dataset. The experimental result are shown in Tab. VII.

Experimental results demonstrate that TV regularization significantly outperforms HG regularization. This is likely because, when generating facial landmark heatmaps using diffusion models, TV regularization more effectively balances noise suppression and detail preservation, maintaining sharp edges and local peaks in key areas. In contrast, HG regularization, due to the incorporation of higher-order gradient information, tends to over-smooth, thereby blurring crucial details.

## G. Ablation Study

In this section, we perform ablation experiment to analyze the contribution of each component to the performance of the FAHCD-Net. The experiment is conducted on the 300W dataset, and evaluation is performed using the Inter-Occular setting.

The experimental results, shown in Tab. VI, demonstrate that the introduction of the HFA module and SR loss significantly enhances the model’s robustness and prediction accuracy. By comparing the experimental outcomes under different configurations, we are able to clearly identify the impact of each design component on the performance of proposed FAHCD-Net model.

TABLE VII  
EFFECT OF DIFFERENT REGULARIZATION TERMS ON THE 300W DATASET. NME IS NORMALIZED BY THE INTER-OCULAR DISTANCE.
<table><tr><td>Reg. Item</td><td>Common</td><td>Challenging</td><td>Full</td></tr><tr><td>HGR</td><td>2.59</td><td>4.35</td><td>2.93</td></tr><tr><td>TVR</td><td>2.50</td><td>4.29</td><td>2.85</td></tr></table>

## V. CONCLUSION

Robust and accurate facial landmark detection (FLD) in complex scenarios remains a significant challenge due to structural variations, information loss, and noise interference. In this paper, we propose an innovative FAHCD-Net to address these issues by integrating the FAHCD model and SR loss in a cascaded manner. In each FAHCD model, the HFA module dynamically adjusts the frequency components to better align with the ground-truth, effectively suppressing overrecovered details and avoiding unnecessary high-frequency noise. Additionally, the SR loss is designed to mitigate redundant high-frequency noise while enforcing smoothness and continuity in the generated heatmaps, thereby ensuring accurate FLD. Experimental results on challenging datasets demonstrate that FAHCD-Net exhibits strong robustness and accuracy in complex FLD tasks. Furthermore, our approach shows that adaptive frequency information helps retain facial structure and suppress redundant noise, ultimately improving accuracy. In the future, we aim to develop a universal method for landmark detection across diverse scenarios by combining frequency information with scenario-specific labels.

## REFERENCES

[1] J. C. L. Chai, T.-S. Ng, C.-Y. Low, J. Park, and A. Teoh, “Recognizability embedding enhancement for very low-resolution face recognition and quality estimation,” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9957–9967, 2023.

[2] S. Hu, H. Qi, J. Wan, J. Huang, L. Zhang, H. Sun, and D. Tao, “Protoformer: Unified facial landmark detection by prototype transformer,” IEEE Transactions on Multimedia, vol. 28, pp. 6189–6200, 2026.

[3] J. Bao, D. Chen, F. Wen, H. Li, and G. Hua, “Towards open-set identity preserving face synthesis,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6713–6722, 2018.

[4] S. Ren, K. Jin, S. Hu, B. Song, H. Sun, W. Min, Y. Liu, and J. Wan, “Freqfld: Towards all-in-one facial landmark detection via frequency modulation,” Neural Networks, p. 109604, 2026.

[5] J. Zhou, W. Pedrycz, C. Gao, Z. Lai, J. Wan, and Z. Ming, “Robust jointly sparse fuzzy clustering with neighborhood structure preservation,” IEEE Transactions on Fuzzy Systems, vol. 30, no. 4, pp. 1073– 1087, 2021.

[6] V. Bettadapura, “Face expression recognition and analysis: The state of the art,” ArXiv, vol. abs/1203.6722, 2012.

[7] X. Hou, X. Zhang, H. Liang, L. Shen, Z. Lai, and J. Wan, “Guidedstyle: Attribute knowledge guided style manipulation for semantic face editing,” Neural Networks, vol. 145, pp. 209–220, 2022.

[8] P. Garrido, M. Zollhofer, D. Casas, L. Valgaerts, K. Varanasi, P. P ¨ erez,´ and C. Theobalt, “Reconstruction of personalized 3d face rigs from monocular video,” ACM Transactions on Graphics (TOG), vol. 35, pp. 1 – 15, 2016.

[9] H. Sun, B. Li, Z. Dan, W. Hu, B. Du, W. Yang, and J. Wan, “Multi-level feature interaction and efficient non-local information enhanced channel attention for image dehazing,” Neural Networks, vol. 163, pp. 10–27, 2023.

[10] Q. Zhu, Z. Lu, J. Zhu, N. Chen, S. Yin, and S. Fong, “Alphaseek: Trajectory-level self-iterative factor mining framework for multi-source financial data,” in International Conference on Intelligent Computing. Springer, 2026, pp. 265–276.

[11] X. Chen, H. Li, J. Dong, J. Pan, X. Li, X. He, N. Chen, S. Li, F. Liu, H. Lv et al., “Lovif 2026 challenge on real-world all-in-one image restoration: Methods and results,” arXiv preprint arXiv:2604.19445, 2026.

[12] J. Ma, S. Hu, X. Zhang, J. Wan, J. Huang, L. Zhang, and S. Khan, “Evoir: towards all-in-one image restoration via evolutionary frequency modulation,” arXiv preprint arXiv:2512.05104, 2025.

[13] S. Hu, J. Ma, X. Zhang, Y. Jing, L. Zhang, and J. Wan, “Clusir: Towards cluster-guided all-in-one image restoration,” arXiv preprint arXiv:2512.10948, 2025.

[14] S. Hu, J. Shao, J. Ma, X. Zhang, K. Wu, Q. Zhu, B. Song, and J. Wan, “Spikerestormer: Towards energy-efficient all-in-one image restoration via unified event reasoning,” arXiv preprint arXiv:2608.02290, 2026.

[15] Y. Sun, X. Wang, and X. Tang, “Deep convolutional network cascade for facial point detection,” 2013 IEEE Conference on Computer Vision and Pattern Recognition, pp. 3476–3483, 2013.

[16] E. Zhou, H. Fan, Z. Cao, Y. Jiang, and Q. Yin, “Extensive facial landmark localization with coarse-to-fine convolutional network cascade,” 2013 IEEE International Conference on Computer Vision Workshops, pp. 386–391, 2013.

[17] G. Trigeorgis, P. Snape, M. A. Nicolaou, E. Antonakos, and S. Zafeiriou, “Mnemonic descent method: A recurrent process applied for end-to-end face alignment,” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4177–4187, 2016.

[18] S. Xiao, J. Feng, J. Xing, H. Lai, S. Yan, and A. A. Kassim, “Robust facial landmark detection via recurrent attentive-refinement networks,” in European Conference on Computer Vision, 2016.

[19] J. Li, H. yang Jin, S. Liao, L. Shao, and P.-A. Heng, “Repformer: Refinement pyramid transformer for robust facial landmark detection,” ArXiv, vol. abs/2207.03917, 2022.

[20] M. Kowalski, J. Naruniec, and T. Trzcinski, “Deep alignment network: A convolutional neural network for robust face alignment,” 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 2034–2043, 2017.

[21] J. Yang, Q. Liu, and K. Zhang, “Stacked hourglass network for robust facial landmark localisation,” 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 2025–2033, 2017.

[22] A. Dapogny, K. Bailly, and M. Cord, “Decafa: Deep convolutional cascade for face alignment in the wild,” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6892–6900, 2019.

[23] X. Dong, S.-I. Yu, X. Weng, S.-E. Wei, Y. Yang, and Y. Sheikh, “Supervision-by-registration: An unsupervised approach to improve the precision of facial landmark detectors,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 360–368, 2018.

[24] X. Wang, L. Bo, and F. Li, “Adaptive wing loss for robust face alignment via heatmap regression,” 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6970–6980, 2019.

[25] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” ArXiv, vol. abs/2006.11239, 2020.

[26] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10 674–10 685, 2021.

[27] A. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. McGrew, I. Sutskever, and M. Chen, “Glide: Towards photorealistic image generation and editing with text-guided diffusion models,” in International Conference on Machine Learning, 2021.

[28] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. C. Courville, and Y. Bengio, “Generative adversarial nets,” in Neural Information Processing Systems, 2014.

[29] T. Cootes, G. Edwards, and C. Taylor, “Active appearance models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 23, no. 6, pp. 681–685, 2001.

[30] T. Cootes, C. Taylor, D. Cooper, and J. Graham, “Active shape modelstheir training and application,” Computer Vision and Image Understanding, vol. 61, no. 1, pp. 38–59, 1995.

[31] D. Cristinacce and T. Cootes, “Feature detection and tracking with constrained local models,” in British Machine Vision Conference, 2006.

[32] J. Wan, J. Liu, J. Zhou, Z. Lai, L. Shen, H. Sun, P. Xiong, and W. Min, “Precise facial landmark detection by reference heatmap transformer,” IEEE Transactions on Image Processing, vol. 32, pp. 1966–1977, 2023.

[33] C. Gao, J. Zhou, D. Miao, X. Yue, and J. Wan, “Granular-conditionalentropy-based attribute reduction for partially labeled data with proxy labels,” Information Sciences, vol. 580, pp. 111–128, 2021.

[34] H. Sun, Z. Luo, D. Ren, W. Hu, B. Du, W. Yang, J. Wan, and L. Zhang, “Partial siamese with multiscale bi-codec networks for remote sensing image haze removal,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–16, 2023.

[35] H. Sun, Z. Luo, D. Ren, B. Du, L. Chang, and J. Wan, “Unsupervised multi-branch network with high-frequency enhancement for image dehazing,” Pattern Recognition, vol. 156, p. 110763, 2024.

[36] J. Zhou, W. Pedrycz, X. Yue, C. Gao, Z. Lai, and J. Wan, “Projected fuzzy c-means clustering with locality preservation,” Pattern Recognition, vol. 113, p. 107748, 2021.

[37] K. Yin, V. R. Rao, R. Jiang, X. Liu, P. Aarabi, and D. B. Lindell, “Scemae: Selective correspondence enhancement with masked autoencoder for self-supervised landmark estimation,” 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1313–1322, 2024.

[38] J. Liang, H. Liu, H. Xu, and D. Luo, “Generalizable face landmarking guided by conditional face warping,” 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2425–2435, 2024.

[39] J. Wan, Z. Lai, J. Liu, J. Zhou, and C. Gao, “Robust face alignment by multi-order high-precision hourglass network,” arXiv preprint arXiv:2010.08722, 2020.

[40] J. Wan, X. Xiong, N. Chen, Z. Lai, J. Zhou, and W. Min, “Fgtbt: Frequency-guided task-balancing transformer for unified facial landmark detection,” Information Sciences, p. 123130, 2026.

[41] S. Tourani, A. Alwheibi, A. Mahmood, and M. H. Khan, “Poseguided self-training with two-stage clustering for unsupervised landmark discovery,” 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23 041–23 051, 2024.

[42] H. Jin, S. Liao, and L. Shao, “Pixel-in-pixel net: Towards efficient facial landmark detection in the wild,” International Journal of Computer Vision, vol. 129, pp. 3174 – 3194, 2020.

[43] J. Wan, Y. Yao, Z. Lai, J. Zhou, X. Hou, and W. Min, “Supervisionby-hallucination-and-transfer: A weakly-supervised approach for robust and precise facial landmark detection,” arXiv preprint arXiv:2601.12919, 2026.

[44] W. Li, Y. Lu, K. Zheng, H. Liao, C. Lin, J. Luo, C.-T. Cheng, J. Xiao, L. Lu, C.-F. Kuo, and S. Miao, “Structured landmark detection via topology-adapting deep graph learning,” in European Conference on Computer Vision, 2020.

[45] J. Xia, W. Qu, W.-F. Huang, J. Zhang, X. Wang, and M. Xu, “Sparse local patch transformer for robust face alignment and landmarks inherent relation learning,” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4042–4051, 2022.

[46] J. Wang, K. Sun, T. Cheng, B. Jiang, C. Deng, Y. Zhao, D. Liu, Y. Mu, M. Tan, X. Wang, W. Liu, and B. Xiao, “Deep high-resolution representation learning for visual recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, pp. 3349–3364, 2019.

[47] P. Micaelli, A. Vahdat, H. Yin, J. Kautz, and P. Molchanov, “Recurrence without recurrence: Stable video landmark detection with deep equilibrium models,” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22 814–22 825, 2023.

[48] Z. Zhou, H. Li, H. Liu, N. na Wang, G. Yu, and R. Ji, “Star loss: Reducing semantic ambiguity in facial landmark detection,” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15 475–15 484, 2023.

[49] Y. Wang, J. Yu, and J. Zhang, “Zero-shot image restoration using denoising diffusion null-space model,” ArXiv, vol. abs/2212.00490, 2022.

[50] A. Niu, K. Zhang, T. X. Pham, J. Sun, Y. Zhu, I.-S. Kweon, and Y. Zhang, “Cdpmsr: Conditional diffusion probabilistic models for single image super-resolution,” 2023 IEEE International Conference on Image Processing (ICIP), pp. 615–619, 2023.

[51] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” ArXiv, vol. abs/2206.00364, 2022.

[52] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, “Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps,” ArXiv, vol. abs/2206.00927, 2022.

[53] L. Zhang, A. Rao, and M. Agrawala, “Adding conditional control to textto-image diffusion models,” 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3813–3824, 2023.

[54] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, “Dreambooth: Fine tuning text-to-image diffusion models for subjectdriven generation,” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22 500–22 510, 2022.

[55] S. Xie, Z. Zhang, Z. Lin, T. Hinz, and K. Zhang, “Smartbrush: Text and shape guided object inpainting with diffusion model,” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22 428–22 437, 2022.

[56] K. Sohn, N. Ruiz, K. Lee, D. C. Chin, I. Blok, H. Chang, J. Barber, L. Jiang, G. Entis, Y. Li, Y. Hao, I. Essa, M. Rubinstein, and D. Krishnan, “Styledrop: Text-to-image generation in any style,” ArXiv, vol. abs/2306.00983, 2023.

[57] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” ArXiv, vol. abs/2010.02502, 2020.

[58] S. Ren, Q. Li, H. Sun, X. Wu, B. Song, and J. Wan, “Frequency-aware gradient correction for unified facial landmark detection,” 2026.

[59] J. Wan, Y. Yao, J. Huang, X. Ding, L. Zhang, Y. Gao, and D. Tao, “Universal facial landmark detection by landmark-clustering relationreasoning transformer,” International Journal of Computer Vision, vol. 134, no. 6, p. 310, 2026.

[60] J. Ho, C. Saharia, W. Chan, D. J. Fleet, M. Norouzi, and T. Salimans, “Cascaded diffusion models for high fidelity image generation,” J. Mach. Learn. Res., vol. 23, pp. 47:1–47:33, 2021.

[61] C. Sagonas, E. Antonakos, G. Tzimiropoulos, S. Zafeiriou, and M. Pantic, “300 faces in-the-wild challenge: database and results,” Image and Vision Computing, pp. 3–18, 2016.

[62] X. P. Burgos-Artizzu, P. Perona, and P. Dollar, “Robust face landmark´ estimation under occlusion,” 2013 IEEE International Conference on Computer Vision, pp. 1513–1520, 2013.

[63] W. Wu, C. Qian, S. Yang, Q. Wang, Y. Cai, and Q. Zhou, “Look at boundary: A boundary-aware face alignment algorithm,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2129–2138, 2018.

[64] S. Zhu, C. Li, C. C. Loy, and X. Tang, “Unconstrained face alignment via cascaded compositional learning,” 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3409–3417, 2016.

[65] Y. Huang, H. Yang, C. Li, J. Kim, and F. Wei, “Adnet: Leveraging error-bias towards normal direction in face alignment,” 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3060–3070, 2021.

[66] C. Zhu, X. Wan, S. Xie, X. Li, and Y. Gu, “Occlusion-robust face alignment using a viewpoint-invariant hierarchical network architecture,” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11 102–11 111, 2022.

[67] Z. Gao and I. Patras, “Self-supervised facial representation learning with facial region awareness,” 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2081–2092, 2024.

[68] J. Wan, H. Liu, Y. Wu, Z. Lai, W. Min, and J. Liu, “Precise facial landmark detection by dynamic semantic aggregation transformer,” Pattern Recognit., vol. 156, p. 110827, 2024.

[69] A. Kumar, T. K. Marks, W. Mou, Y. Wang, M. Jones, A. Cherian, T. Koike-Akino, X. Liu, and C. Feng, “Luvli face alignment: Estimating landmarks’ location, uncertainty, and visibility likelihood,” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8233–8243, 2020.

[70] X. Dong, Y. Yan, W. Ouyang, and Y. Yang, “Style aggregated network for facial landmark detection,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 379–388, 2018.

[71] Z. Feng, J. Kittler, M. Awais, P. Huber, and X. Wu, “Wing loss for robust facial landmark localisation with convolutional neural networks,” 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2235–2245, 2017.

[72] H. Ding, P. Zhou, and R. Chellappa, “Occlusion-adaptive deep network for robust facial expression recognition,” 2020.

[73] W. Li, Y. Lu, K. Zheng, H. Liao, C. Lin, J. Luo, C.-T. Cheng, J. Xiao, L. Lu, C.-F. Kuo et al., “Structured landmark detection via topologyadapting deep graph learning,” in European Conference on Computer Vision. Springer, 2020, pp. 266–283.

[74] P. Gao, K. Lu, J. Xue, L. Shao, and J. Lyu, “A coarse-to-fine facial landmark detection method based on self-attention mechanism,” IEEE Transactions on Multimedia, vol. 23, pp. 926–938, 2021.

[75] H. Jin, S. Liao, and L. Shao, “Pixel-in-pixel net: Towards efficient facial landmark detection in the wild,” International Journal of Computer Vision, Sep 2021.

[76] C. Zhu, X. Wan, S. Xie, X. Li, and Y. Gu, “Occlusion-robust face alignment using a viewpoint-invariant hierarchical network architecture,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11 112–11 121.

[77] H. Li, Z. Guo, S.-M. Rhee, S. Han, and J.-J. Han, “Towards accurate facial landmark detection via cascaded transformers,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 4176–4185.

[78] X. Lan, Q. Hu, and J. Cheng, “Atf: An alternating training framework for weakly supervised face alignment,” IEEE Transactions on Multimedia, vol. 25, pp. 1798–1809, 2023.

[79] P. Gao, K. Lu, J. Xue, J. Lyu, and L. Shao, “A facial landmark detection method based on deep knowledge transfer,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 3, pp. 1342– 1353, 2023.

[80] T. Wen, Z. Ding, Y. Yao, Y. Wang, and X. Qian, “Picassonet: Searching adaptive architecture for efficient facial landmark localization,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 12, pp. 10 516–10 527, 2023.

[81] S. Kato, K. Hotta, Y. Hatakeyama, and Y. Konishi, “Lite-hrnet plus: Fast and accurate facial landmark detection,” in 2023 IEEE International Conference on Image Processing (ICIP), 2023, pp. 1500–1504.

[82] R. Valle, J. M. Buenaposada, A. Valdes, and L. Baumela, “A deeply-´ initialized coarse-to-fine ensemble of regression trees for face alignment,” in European Conference on Computer Vision, 2018.

[83] J. Wan, Z. Lai, J. Li, J. Zhou, and C. Gao, “Robust facial landmark detection by multiorder multiconstraint deep networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 5, pp. 2181–2194, 2022.

[84] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, “Hierarchical text-conditional image generation with clip latents,” ArXiv, vol. abs/2204.06125, 2022.

[85] H. Li, Z. Guo, S.-M. Rhee, S. J. Han, and J.-J. Han, “Towards accurate facial landmark detection via cascaded transformers,” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4166–4175, 2022.

[86] J. Wan, H. Xi, Y. Yao, H. Sun, Z. Lai, and J. Zhou, “Interpretable facial landmark detection by multi-expert collaborative uncertainty-aware deep networks,” Neural Networks, p. 108195, 2025.

[87] T. Liu, J. Li, J. Wu, B. Du, Y. Zhan, D. Tao, and J. Wan, “Facial expression recognition with heatmap neighbor contrastive learning,” IEEE Transactions on Multimedia, vol. 27, pp. 4795–4807, 2025.

[88] J. Wan, H. Liu, Y. Wu, Z. Lai, W. Min, and J. Liu, “Precise facial landmark detection by dynamic semantic aggregation transformer,” Pattern Recognition, vol. 156, p. 110827, 2024.