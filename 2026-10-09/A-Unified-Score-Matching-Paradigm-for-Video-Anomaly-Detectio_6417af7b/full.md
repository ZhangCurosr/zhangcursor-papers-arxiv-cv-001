# A Unified Score Matching Paradigm for Video Anomaly Detection and Anticipation

Congqi Cao , Member, IEEE, Zhenhe Liang , Hanwen Zhang , Yifan Zhao , Qinyi Lv , Lingtong Min , Member, IEEE, and Yanning Zhang , Fellow, IEEE

Abstract—Video anomaly detection (VAD) is a fundamental and safety-critical task in computer vision. Recent generative approaches detect anomalies from a distributional perspective, but remain limited by local anomaly modes. Meanwhile, video anomaly anticipation (VAA), as a proactive extension beyond post-hoc detection, introduces additional challenges. In particular, the contrastive inference paradigm in VAD, which relies on ground-truth frames, is not applicable to VAA, hindering its development. To address these challenges, we propose a unified score-driven framework, termed Uni-DSM, based on denoising score matching (DSM), which models anomaly patterns through likelihood estimation and score functions over the learned data distribution. Within this unified framework, we adopt a shared noise-conditioned score transformer backbone with scene-dependent embeddings and motion-aware weighting for distribution-level modeling. Instead of introducing separate architectures, Uni-DSM unifies VAD and VAA through different inference and supervision paradigms built upon the same scorebased formulation. For VAD, we instantiate an autoregressive denoising score matching (ADSM) mechanism, which progressively accumulates anomalous evidence via autoregressive denoising, enabling enhanced perception of local modes beyond visual cues. For VAA, we extend the same architecture by incorporating a lightweight auxiliary decoder and a novel self-distilled denoising score matching (SDSM) mechanism. By constructing supervision from output discrepancies instead of relying on unavailable future ground truth, our method achieves efficient training suitable or early anomaly anticipation. Extensive experiments on multiple benchmark datasets demonstrate state-of-the-art performance in both VAD and VAA while maintaining high efficiency, establishing a unified and scalable pipeline from anomaly detection to anticipation.

Index Terms—Scene-dependent anomaly, video anomaly detection and anticipation, unified framework, denoising score matching.

## I. INTRODUCTION

anticipation (VAA) are core tasks in intelligent surveillance. Compared to VAD, which detects anomalies after occurrence, VAA forecasts near-future risks to enable proactive response, which is particularly critical in high-risk scenarios.

![](images/1d24deb0753d6f24eb843690a40e87e9dc467640ee5761d3d2dca7675886f523.jpg)  
Fig. 1. An illustration of the local modes of anomalies. We explore two regions in which the anomalies are residing. The likelihood-based method is highly sensitive to regions of anomalies with inherently low probability likelihood while blind to anomalies located in local modes near the learned distribution.

Due to the rarity and unpredictability of anomalous events, existing approaches generally adopt a semi-supervised paradigm trained solely on normal data [1], [2], [3], [4]. In this setting, reconstruction-based methods [5], [6], [7], [8], [9], [10], [11] and prediction-based methods [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22] have emerged as two dominant paradigms. Their core assumption is that anomalous samples are more difficult to accurately reconstruct and predict than normal ones [8], [14]. However, these methods primarily rely on low-level visual contrastive errors, which makes them prone to misclassifying samples with significant appearance variations but normal semantics as anomalies.

In recent years, advances in generative models [23], [24], [25], [26], [27], [28] have offered new perspectives for video anomaly modeling: anomalous events inherently lie in lowprobability density regions [29] of the normal data distribution and can be characterized via likelihoods or their gradients. Although prior studies have explored leveraging likelihood information for auxiliary visual pretraining [30], [31] or density estimation [32], [33], [34], they have not fully exploited its potential for direct anomaly modeling. However, directly relying on log-likelihoods or their gradients (i.e., the Stein score [35]) for anomaly detection remains challenging. As shown in Fig. 1, the Stein score tends to vanish at extrema and saddle points of the distribution, making anomalies within local modes difficult to distinguish effectively [33]. More fundamentally, we attribute this limitation to several gaps between likelihood-based measures and visual perception, including a scene gap that is insensitive to semantic context [12], [13], a motion gap due to insufficient modeling of dynamic patterns, and an appearance gap stemming from limited capability to capture fine-grained visual variations.

![](images/af52d1a21e748f97fa7ce404db2547489515a26cf430015f981c45bd7f0321aa.jpg)  
Fig. 2. An illustration of three video anomaly anticipation frameworks, where GT denotes ground truth. Compared with semantic memory bank and bidirectional frame prediction frameworks, our self-distilled DSM (SDSM) within Uni-DSM adopts a lightweight unidirectional main-auxiliary architecture, enabling efficient video anomaly anticipation with lower GPU memory consumption and faster inference speed.

Meanwhile, as a proactive anticipation task, VAA poses even greater challenges. The primary difficulty lies in the lack of future ground truth, which renders contrastive-objectivebased VAD inference paradigms unsuitable. As shown in Fig. 2, prior work addresses this using semantic memory banks [36] or forward-backward prediction frameworks [9], [12]. While bidirectional architectures better capture spatiotemporal context and achieve strong generalization, surpassing semantic-based methods, they significantly increase computational complexity. This limits their applicability in real-world deployment.

To address these challenges, as shown in Fig. 3, we propose a unified score-driven framework, termed Uni-DSM, based on denoising score matching (DSM), which models video anomaly detection and anticipation from a distributional perspective. In this framework, anomalies are characterized as low-likelihood regions under the learned normal distribution, and score functions are leveraged to quantify such distributional deviations. Unlike conventional methods relying on pixel-level contrastive errors, our approach directly operates on the intrinsic data distribution. Building upon this formulation, we formulate a unified score matching paradigm that supports both detection and anticipation through task-aligned inference and supervision mechanisms. Within the Uni-DSM framework, we adopt a shared noise-conditioned score transformer (NCST) as the backbone to approximate the Stein score of noisy data under multiple noise levels. The NCST embeds denoising score matching into a diffusion transformer, enabling effective modeling of complex spatiotemporal dependencies. On top of this shared formulation, we further introduce a scene-dependent and motion-aware score by incorporating scene-conditioned embeddings and motion weights derived from key-frame discrepancies, which strengthens temporal consistency and improves robustness under varying contexts.

For the VAD task, we instantiate an autoregressive denoising score matching (ADSM) mechanism. Specifically, we develop an autoregressive denoising strategy during inference that progressively injects intensifying noise and accumulates anomalous context across steps. Meanwhile, discrepancies between denoised outputs and original inputs are integrated to enhance appearance perception and suppress local modes, leading to more reliable anomaly discrimination.

For the VAA task, we extend the same score-driven formulation and propose a self-distilled denoising score matching (SDSM) mechanism. Instead of relying on unavailable future ground-truth frames, SDSM constructs supervision from output discrepancies under a lightweight main-auxiliary architecture with unidirectional temporal inference. This design maintains architectural consistency with ADSM while introducing a self-distillation paradigm tailored for anticipation, enabling efficient training and practical predictive anomaly warning with reduced computational overhead.

In summary, we present the first unified framework that introduces denoising score matching into both video anomaly detection and anticipation under a consistent score-driven formulation. The proposed Uni-DSM framework integrates a shared noise-conditioned score transformer, a scene-dependent and motion-aware score function, and two task-specific mechanisms (e.g., autoregressive DSM for detection and self-distilled DSM for anticipation). Extensive experimental results demonstrate that the proposed methods consistently outperform stateof-the-art approaches on multiple benchmark datasets, including Avenue [37], ShanghaiTech [38], and NWPU Campus [12]. Our contributions are as follows:

• A unified score-driven framework, Uni-DSM, based on denoising score matching (DSM), representing the first formulation that models anomalies as low-likelihood patterns under the learned normal distribution for both video anomaly detection and anticipation.

• A shared score-based modeling strategy with a noise-conditioned score transformer, integrating scenedependent embeddings and motion-aware weighting to mitigate scene, motion, and appearance gaps and align likelihood estimation with structured visual dynamics.

• An autoregressive DSM (ADSM) mechanism for VAD, enhancing sensitivity to local anomalies by progressively injecting Gaussian noise and aggregating discrepancies between denoised and original data via the score function.

• A self-distilled DSM (SDSM) mechanism for VAA, extending the architecture with a lightweight auxiliary decoder to enable ground-truth-free anticipation via selfdistillation and unidirectional temporal inference.

• Extensive experiments on multiple benchmarks demonstrate state-of-the-art performance in both detection and anticipation, confirming the effectiveness and generalization of the unified framework.

This paper is an extension of our previous work [39]. We extend the preliminary version in the following aspects: (1) We generalize the framework from video anomaly detection to video anomaly anticipation under the same unified denoising score matching formulation, establishing a coherent pipeline for both tasks. (2) We introduce a self-distilled DSM (SDSM) mechanism to address the absence of future groundtruth frames by leveraging output discrepancy. (3) We design a lightweight main-auxiliary architecture with unidirectional temporal inference, which replaces bidirectional paradigms and significantly reduces model complexity. (4) We conduct more comprehensive experiments with thorough analyses, demonstrating consistent improvements in both performance and efficiency, especially in predictive anomaly warning scenarios.

![](images/260529b42affa4b9e78a20dbe2747f573440b8b8ccef9698f89c20e1d56f2a94.jpg)  
Fig. 3. An overview of the Uni-DSM framework, a unified score-driven architecture based on denoising score matching (DSM). The red flow represents autoregressive denoising score matching (ADSM) for video anomaly detection, while the purple flow denotes self-distilled score matching (SDSM) for video anomaly anticipation.

## II. RELATED WORK

## A. Video Anomaly Detection Methods

Mainstream video anomaly detection (VAD) methods can be broadly categorized into semi-supervised [40], weakly supervised [41], [42], and fully supervised [43] approaches, depending on the availability of anomaly data and labels during training. This paper focuses on semi-supervised methods, which primarily fall into three paradigms including distancebased methods [44], [45], [46], [47], [48], probability-based methods [13], [17], [18], [49], [50], and reconstruction-based methods [5], [6], [7], [8], [9], [10], [11], [12], [14], [15], [16], [19], [20], [21], [22], [51], [52], [53], [54]. Reconstructionbased methods include both spatial frame reconstruction and temporal feature prediction. These methods train deep networks via pretext tasks such as reconstructing frames with autoencoders [5], [6], [7], [8], [11], predicting future frames or features [12], [13], [14], [15], [16], [17], [18], [20], [22], [51], [52], [53], learning dictionaries [45], [46], [47], [55], and solving masked jigsaw puzzles [10], [19], [21]. The strong mode coverage capability of generative models [30], [31], [32], [34] has further inspired a novel class of likelihood-based VAD methods. These approaches build generative models to learn the distribution of normal data, either to enhance visual pretraining objectives [30], [31] or to assist density estimation models [32], [34]. However, despite their effectiveness, directly adopting likelihood-based metrics still suffers from the issue of invisible anomalies residing in local modes, making them difficult to distinguish.

In recent years, diffusion models [23] with Langevin-based score matching [24], [25] have shown strong performance. The score function is defined as the gradient of log density with respect to the input and can be viewed as a vector field pointing to higher data density, making it suitable for anomaly detection [13], [18], [34]. Cao et al. [13] build a scene conditioned latent autoencoder for anomaly detection and anticipation. Rodrigues et al. [18] approximate scores at multiple noise levels to capture multi-scale gradients. Micorek et al. [34] show that using video features improves log density estimation but reduces nonlinearity. However, these methods are limited by the local modes of the score function, and relying solely on likelihood or score magnitude is insufficient for robust anomaly discrimination. More importantly, they lack explicit modeling of structured visual dynamics, including scene context and motion patterns, which leads to a misalignment between distribution-level estimation and visual perception.

To address these challenges, we propose an autoregressive denoising score matching (ADSM) mechanism within Uni-DSM. It models temporal dependencies via autoregressive denoising with accumulated abnormal context, while jointly leveraging visual appearance and likelihood for complementary enhancement. We adopt a noise-conditioned score transformer (NCST) as the backbone. Compared with standard DSM, which is insensitive to local-mode anomalies, Uni-DSM introduces scene-dependent and motion-aware constraints into score estimation, producing a more robust anomaly indicator that better separates normal and anomalous events.

## B. Video Anomaly Anticipation Methods

Compared to the well-established video anomaly detection task, video anomaly anticipation (VAA) is significantly more challenging, as it requires forecasting events that have not occurred, and remains at an early stage. As shown in Fig. 2, existing approaches primarily focus on how to construct a reference for predictive inference in the absence of future ground truth, leading to two main technical paradigms.

![](images/eaeb66af322e862da4de62cfa5ed8c2572262645ee6c0d29486b40e38045b730.jpg)  
Fig. 4. Schematic illustration of the video anomaly detection (VAD) and video anomaly anticipation (VAA) tasks.

Semantic memory bank methods replace missing groundtruth references with dynamically updated feature banks. For instance, EPAP-Net [36] learns the semantic distribution of future frames from normal videos and maintains a semantic pool using a fill-and-update strategy, while employing a channel-selective shifting encoder to capture motion dynamics. Although this approach avoids the complexity of pixel-level anticipation, it suffers from weak correlation between stored semantic features and evolving anomaly patterns, and its reliance on a fixed semantic representation limits generalization across diverse scenarios.

Bidirectional frame prediction frameworks adopt an end-toend data-driven strategy, achieving strong spatiotemporal modeling and outperforming semantic memory bank methods on standard benchmarks. A representative example is FBAE [12], which jointly trains a forward network for future prediction and a backward network for past reconstruction. During inference, the backward network takes forward-predicted frames as input, and the resulting forward-backward reconstruction error serves as the anomaly signal. Despite their effectiveness, such designs introduce additional computational overhead during both training and inference due to the involvement of the backward network. Moreover, the forward-backward interaction and reliance on reconstruction accuracy may pose challenges for maintaining stable performance in long-term anticipation.

To further improve efficiency and better accommodate predictive anomaly anticipation in real-world scenarios, we extend Uni-DSM to video anomaly anticipation by introducing a lightweight main-auxiliary decoder and a self-distilled denoising score matching (SDSM) mechanism. This design maintains architectural consistency with ADSM while enabling groundtruth-free supervision for efficient and accurate predictive anomaly anticipation.

## III. TASK DEFINITION

Video anomaly detection (VAD) aims to determine whether an anomalous event is occurring at the current time, while video anomaly anticipation (VAA) predicts whether an anomaly will occur within a future time window. Compared to VAD, VAA enables proactive early warning without requiring precise prediction of the exact anomaly onset. Fig. 4 illustrates both tasks, where $f _ { t }$ denotes the frame at time $t , S ( \cdot )$ represents the anomaly score, and α is the anticipation horizon.

## A. Video Anomaly Detection

The VAD task focuses on the immediate judgment of the current state. Given a historical observation sequence $x _ { i n } = \{ f _ { t - n } , \ldots , f _ { t - 1 } \}$ , VAD aims to estimate the anomaly score $S ( t )$ for the current frame $f _ { t } .$ . Let $\mathcal { G } _ { 0 } = \{ g _ { t } \} _ { t = 1 } ^ { T }$ denote the frame-level ground-truth labels, where $g _ { t } \in \{ 0 , 1 \}$ . A normal frame corresponds to a low anomaly score, while an anomalous frame yields a high score. Due to its reactive nature, VAD can only raise alarms after anomalies have occurred.

## B. Video Anomaly Anticipation

Given the same input $x _ { i n }$ , VAA predicts whether an anomaly will occur within the future window $[ t + 1 , t + \alpha ]$ . The model outputs a score $S ( t { + } 1 : t { + } \alpha )$ indicating the probability of at least one anomaly within this interval. Accordingly, the anticipation label sequence is defined as:

$$
\mathcal G _ { \alpha } = \{ \mathrm { m a x } \left( \{ g _ { t + i } \} _ { i = 1 } ^ { \alpha } \right) \} _ { t = 1 } ^ { T - \alpha } .\tag{1}
$$

Unlike VAD, which evaluates the current state, VAA focuses on future risk, leading to different optimization objectives under the same input. Moreover, the absence of future groundtruth frames makes conventional contrastive or predictionbased paradigms inapplicable, motivating the need for alternative supervision strategies.

## IV. A UNIFIED SCORE-DRIVEN FRAMEWORK BASED ON DENOISING SCORE MATCHING

## A. Framework Overview

As illustrated in Fig. 3, we propose Uni-DSM, a unified probability-driven framework based on denoising score matching (DSM). Built upon a shared noise-conditioned score transformer (NCST), the framework models anomaly patterns through likelihood estimation and score functions over the learned data distribution. To bridge the gaps across scene, motion, and appearance, we incorporate a scene-dependent and motion-aware score function that embeds scene conditions and assigns motion weights based on key-frame discrepancies. Uni-DSM further includes two task-specific variants under this unified formulation. The autoregressive DSM (ADSM) targets detection and addresses anomalies residing in local modes through autoregressive noise injection. The self-distilled DSM (SDSM) targets anticipation and enables ground-truth-free supervision with a lightweight auxiliary branch. Together, this design forms a coherent pipeline that supports both detection and anticipation within a single architecture.

## B. Denoising Score Matching

Let p(x) denote the probability density of normal video data. Its Stein score is defined as the gradient of the logdensity: $\nabla _ { x } \log p ( x )$ . Following [56], we inject i.i.d. Gaussian noise into the clean data x to obtain perturbed samples x˜, and train a score network $s _ { \theta } ( \widetilde { x } )$ using score matching [57]. Through this process, we aim to approximate the score function of the corrupted data distribution $q _ { \sigma } ( { \tilde { x } } ) \ =$ $\textstyle \int q _ { \sigma } ( { \tilde { x } } | x ) p ( x ) d x$ , where σ controls the noise magnitude. It has been theoretically established that this training objective is equivalent to matching the score of a non-parametric Parzen density estimator constructed from the data:

![](images/1b4be4cd3e5bacd2cbdcca9a0df91342887d3e12eac26f31c20dea80f150ddb6.jpg)  
Fig. 5. Overview of the proposed noise-conditional score transformer (NCST) architecture used in Fig. 3.

$$
\frac 1 2 \mathbb { E } _ { q _ { \sigma } ( \tilde { x } | x ) p ( x ) } \left[ \| s _ { \theta } ( \tilde { x } ) - \nabla _ { \tilde { x } } \log q _ { \sigma } ( \tilde { x } | x ) \| ^ { 2 } \right]\tag{2}
$$

This approximation can be further extended to generative models [24], [25], which progressively perturb data with increasing Gaussian noise levels and train a noiseconditioned score network $s _ { \theta } ( x , \sigma )$ to jointly estimate the score functions for all noise levels (i.e., for any $\sigma \in \{ \sigma _ { i } \} _ { i = 1 } ^ { L }$ $s _ { \theta } ( x , \sigma ) \approx \nabla _ { x } \log q _ { \sigma } ( x ) )$ . If the noise distribution is chosen as $\mathcal { N } ( \tilde { x } | x , \sigma ^ { 2 } I )$ , the score function simplifies to

$$
\nabla _ { \tilde { x } } \log q _ { \sigma } ( \tilde { x } | x ) = - \frac { \tilde { x } - x } { \sigma ^ { 2 } }\tag{3}
$$

which can be interpreted as a vector field pointing toward the denoising direction [24], [56]. Consequently, the training objective becomes:

$$
\frac { 1 } { L } \sum _ { i = 1 } ^ { L } \lambda ( \sigma _ { i } ) \cdot \frac { 1 } { 2 } \mathbb { E } _ { q _ { \sigma _ { i } } ( \tilde { x } | x ) p ( x ) } \left\| s _ { \theta } ( \tilde { x } , \sigma _ { i } ) + \frac { \tilde { x } - x } { \sigma _ { i } ^ { 2 } } \right\| ^ { 2 }\tag{4}
$$

where $\lambda ( \sigma _ { i } ) > 0$ is a weighting function. Following [24], we set $\lambda ( \sigma _ { i } ) = \sigma _ { i } ^ { 2 }$

## C. Noise-Conditional Score Transformer

To implement denoising score matching for both video anomaly detection and anticipation, we require a model that not only captures the complex spatiotemporal dynamics of video but also conditions on auxiliary variables such as noise level. Attention-based architectures [58] provide an intuitive solution for modeling long-range dependencies in video sequences. Moreover, the diffusion transformer (DiT) [59] has demonstrated the feasibility of integrating denoising into attention mechanisms. Motivated by this, we propose a novel noiseconditional score transformer (NCST), illustrated in Fig. 5, which is trained by adapting the DSM objective to a blockwise formulation.

Given an input video sequence $x _ { t } = \{ f _ { t - n } , \ldots , f _ { t - 1 } \}$ and a geometric noise schedule $\{ \sigma _ { i } \} _ { i = 1 } ^ { L }$ satisfying $\sigma _ { L } / \sigma _ { L - 1 } =$ $\dots { } = \ \sigma _ { 2 } / \sigma _ { 1 } > 1$ , we perturb $x _ { t }$ with Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \sigma _ { i } ^ { 2 } I )$ at level $\sigma _ { i }$ to obtain $\tilde { x } _ { t }$ , where $q _ { \sigma _ { i } } ( \tilde { x } | x ) =$ $\mathcal { N } ( \tilde { x } | x , \sigma _ { i } ^ { 2 } I )$ . The n perturbed frames are then uniformly partitioned into $N$ non-overlapping image patches of size $d \times d \times c ,$ yielding token sequences $\{ \tilde { P } _ { j } \} _ { j = 1 } ^ { N } \in \mathbb { R } ^ { d \times d \times c }$ and $\{ P _ { j } \} _ { j = 1 } ^ { N } \in \mathbb { R } ^ { d \times d \times c }$ corresponding to $\tilde { x } _ { t }$ and $x _ { t }$ , respectively. Here, c denotes the channel dimension and d is a predefined hyperparameter. For an input frame of spatial resolution $H \times W$ , the total number of patches is $N = n \cdot ( H / d ) \cdot ( W / d )$

![](images/2093681cc137432597b23fa10807fde20ee996c116d9278b2acdc33bb8754e64.jpg)  
Fig. 6. Detailed architecture of the NCST block utilized within the noiseconditional score transformer. MLP, FFN, and MHA stand for the multi-layer perception layer, the feed-forward neural network, and the multi-head attention layer, respectively.

Our noise-conditional score transformer $s _ { \theta } ( P , \sigma )$ aims to estimate the score for each noisy patch via the block-wise DSM objective:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { N C S T } } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \frac { \sigma _ { i } ^ { 2 } } { 2 } \sum _ { j = 1 } ^ { N } \mathbb { E } _ { q _ { \sigma _ { i } } ( \tilde { P _ { j } } | x ) p ( x ) } } } \\ { { \displaystyle \left\| s _ { \theta } ( \tilde { P _ { j } } , \sigma _ { i } ) + \frac { \tilde { P _ { j } } - P _ { j } } { \sigma _ { i } ^ { 2 } } \right\| ^ { 2 } } } \end{array}\tag{5}
$$

The NCST is instantiated within a diffusion-transformer (DiT) framework [59], [60], which has recently demonstrated strong scalability and representation capacity in generative modeling. We condition the model on the diffusion time step i (which is mapped to $\sigma _ { i }$ via an MLP embedding) to control the noise level. Leveraging the attention-based architecture, continuous supervision across multi-scale block-wise denoising enables the learning of a stable vector field, thereby enhancing the discriminability of anomalies.

## D. Scene-Conditioned Embedding

Scene-dependent anomalies [4] have attracted increasing attention as a crucial characteristic of real-world video anomalies. Since the normality or abnormality of an event is inherently contingent on its specific environment, incorporating scene-aware context significantly improves detection performance [12], [13], [61].

For each video sequence, we assign a scene class label $y$ based on the scene (i.e., the camera ID) to which it belongs. We then combine the scene label $y$ with the diffusion time step i to form the overall condition z. To incorporate this condition into the transformer backbone, we apply the scalable adaptive layer normalization (S-AdaLN) mechanism [60] to embed and transmit the overall condition z. This approach is demonstrated to propagate the condition more effectively throughout the model.

As shown in Fig. 6, once the conditioning vector z is obtained, it is fed into an MLP to produce a series of scaling and shifting coefficients $\gamma _ { z } ^ { 1 } , \beta _ { z } ^ { 1 } , \alpha _ { z } ^ { 1 } , \gamma _ { z } ^ { 2 } , \beta _ { z } ^ { 2 }$ , and $\alpha _ { z } ^ { 2 } .$ These coefficients are applied to regulate the hidden features inside each transformer block through a two-step adaptive modulation process:

$$
h ^ { \prime } = h + \alpha _ { z } ^ { 1 } \cdot \mathbf { M H A } \big ( \gamma _ { z } ^ { 1 } \cdot \mathbf { L a y e r N o r m } ( h ) + \beta _ { z } ^ { 1 } \big )\tag{6a}
$$

$$
h = h ^ { \prime } + \alpha _ { z } ^ { 2 } \cdot \mathrm { F F N } \big ( \gamma _ { z } ^ { 2 } \cdot \mathrm { L a y e r N o r m } ( h ^ { \prime } ) + \beta _ { z } ^ { 2 } \big )\tag{6b}
$$

where h and $h ^ { \prime }$ are hidden outputs within the transformer blocks. MHA and FFN denote the multi-head attention layer [58] and the feed-forward neural network, respectively. By performing block-level adaptive modulation, this conditioning strategy transmits the condition information to each transformer block, contributing to superior performance and faster model convergence.

## E. Key-Frame Motion Weighting

Unlike the Fisher score $\nabla _ { \theta }$ log p<sub>θ</sub>(x) in classical statistics, which depends on the model parameters θ, the score function in our framework is defined with respect to the input sequence $x _ { t }$ . This data-dependent formulation allows the score to be adaptively adjusted according to the characteristics of the observed sequence. In practical surveillance scenarios, the dynamic foreground and static background of video frames are not balanced [4]. As a result, directly estimated scores tend to be biased toward background statistics. To alleviate this issue, we explicitly inject motion information into the denoising score matching process.

Following insights from video coding principles [62], we estimate motion by computing the absolute difference between the first and last frames of the input sequence $\begin{array} { r l } { x _ { t } } & { { } = } \end{array}$ $\{ f _ { t - n } , \ldots , f _ { t - 1 } \}$ . This yields a motion representation:

$$
g _ { t } = | f _ { t - 1 } - f _ { t - n } |\tag{7}
$$

The resulting motion map is divided into $N$ non-overlapping patches $\{ G _ { j } \bar  \} _ { j = 1 } ^ { N } \in \mathbb { R } ^ { d \times \bar { d } \times c }$ . For each patch, a motion saliency score $v _ { j }$ is computed by first applying a spatial maximum within each channel and then averaging across channels:

$$
v _ { j } = \frac { 1 } { c } \sum _ { l = 1 } ^ { c } \operatorname* { m a x } \left( G _ { j , l } \right)\tag{8}
$$

![](images/56c989e687e01bc9d93d024d4ae2ba32354512f4f1f7215056af7defb12d0e75.jpg)  
Fig. 7. An overview of the proposed autoregressive denoising score matching process.

where $G _ { j , l } \in \mathbb { R } ^ { d \times d }$ denotes the l-th channel of the j-th patch. These saliency values are subsequently normalized to produce patch-level weights:

$$
\omega _ { j } = \frac { v _ { j } } { \sum _ { k = 1 } ^ { N } v _ { k } }\tag{9}
$$

The weighted denoising score matching objective is then expressed as:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { { D S M } } } = \displaystyle \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \frac { \sigma _ { i } ^ { 2 } } { 2 } \sum _ { j = 1 } ^ { N } \omega _ { j } \mathbb { E } _ { q _ { \sigma _ { i } } ( \tilde { P } _ { j } | x ) p ( x ) } } \\ & { \qquad \quad \left\| s _ { \theta } ( \tilde { P } _ { j } , \sigma _ { i } ) + \frac { \tilde { P } _ { j } - P _ { j } } { \sigma _ { i } ^ { 2 } } \right\| ^ { 2 } } \end{array}\tag{10}
$$

Through this key frame motion weighting mechanism, the denoising score matching procedure assigns greater emphasis to dynamic foreground and reduces interference from static background areas, thereby encapsulating more motion information.

F. Autoregressive Denoising Mechanism for Anomaly Detection

To amplify responses to local-mode anomalies while preserving low scores for normal data, we propose an autoregressive DSM (ADSM) mechanism across multiple noise levels. As illustrated in Fig. 7, for each perturbed sample $\tilde { x } _ { t }$ obtained by adding noise $\sigma _ { i }$ to the original sequence, the denoised output is computed as $\hat { x } _ { t } ~ = ~ \tilde { x } _ { t } + \sigma _ { i } ^ { 2 } s _ { \theta } ( \tilde { x } _ { t } , \sigma _ { i } )$ . Crucially, we improve the mechanism by autoregressively applying intensifying noise to the denoised data instead of the original data, and estimate the score at each step. This autoregressive refinement accumulates the abnormal context over iterations, yielding a more effective score function.

Furthermore, to bridge the appearance gap, we incorporate visual fidelity via PSNR between the original and denoised frames. Specifically, for each denoising step at noise level $\sigma _ { i } .$ we define the per-noise-level anomaly score as:

$$
\mathrm { s c o r e } _ { i } ( t ) = \frac { \| s _ { \theta } ( \tilde { x } _ { t } , \sigma _ { i } ) \| } { P S N R ( \hat { x } _ { t } , x _ { t } ) }\tag{11}
$$

where the denominator acts as a visual consistency regularizer. When likelihood and appearance agree (e.g., normal data), the score remains low. In contrast, when they disagree (e.g., anomalous motion with plausible appearance), the denominator suppresses false positives while the numerator compensates for missed detections.

G. Self-Distilled Denoising Mechanism for Anomaly Anticipation

To enable ground-truth-free supervision for video anomaly anticipation (VAA), we propose a novel self-distilled DSM (SDSM) mechanism. Unlike conventional knowledge distillation, SDSM leverages the consistency of score estimates under the learned normal distribution. Specifically, since both the main and auxiliary branches are trained exclusively on normal data, they produce highly consistent denoised outputs for normal test samples. In contrast, anomalous inputs tend to induce noticeable divergence between their outputs. Consequently, the discrepancy between the two predictions provides a reliable proxy for anomaly probability, circumventing the absence of future ground truth in VAA.

As illustrated in Fig. 3, SDSM is built upon the noiseconditioned score transformer (NCST). It comprises a shared backbone with two branches, including a main branch and an auxiliary branch. The auxiliary branch diverges after the first transformer block of the main branch and adds only one extra transformer layer.

Specifically, let $x _ { \mathrm { i n } } ^ { t } = \{ f _ { t - n } , \ldots , f _ { t - 1 } \}$ denote the input video clip at time $t ,$ and $x _ { \mathrm { t a r } } ^ { t } = \{ f _ { t + 1 } , \dotsc , f _ { t + n } \}$ denote the target video clip at time t. Let $\{ \sigma _ { i } \} _ { i = 1 } ^ { L }$ be a geometric noise schedule satisfying $\sigma _ { L } / \sigma _ { L - 1 } = \cdot \cdot \cdot = \sigma _ { 2 } / \sigma _ { 1 } > 1$ . At noise level $\sigma _ { i } ,$ the input and target video sequences are perturbed with Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \sigma _ { i } ^ { 2 } I )$ to obtain $\tilde { x } _ { \mathrm { i n } } ^ { t }$ and $\tilde { x } _ { \mathrm { t a r } } ^ { t } ,$ where $q _ { \sigma _ { i } } ( \tilde { x } | x ) = \mathcal N ( \tilde { x } | x , \sigma _ { i } ^ { 2 } I )$ . We partition the n perturbed frames from the input sequence $\tilde { x } _ { \mathrm { i n } } ^ { t }$ and the n perturbed frames from the target sequence $\tilde { x } _ { \mathrm { t a r } } ^ { t }$ using the uniform frame patch embedding method, embedding each individual frame into tokens. Consequently, we obtain tokens $\{ \tilde { P } _ { i n } ^ { j } \} _ { j = 1 } ^ { N } \in \mathbb { R } ^ { d \times d \times c } ,$ $\{ P _ { i n } ^ { j } \} _ { j = 1 } ^ { N } \in \mathbb { R } ^ { d \times d \times c } , \{ \tilde { { \bf p } } _ { t a r } ^ { j } \} _ { j = 1 } ^ { N } \in \mathbb { R } ^ { d \times d \times c }$ , and $\{ \mathbf { p } _ { t a r } ^ { j } \} _ { j = 1 } ^ { N } \in$ $\mathbb { R } ^ { d \times d \times c }$ from $\tilde { x } _ { \mathrm { i n } } ^ { t } , \ x _ { \mathrm { i n } } ^ { t } , \ \tilde { x } _ { \mathrm { t a r } } ^ { t } ,$ and $x _ { \mathrm { t a r } } ^ { t } .$ , respectively. Here, N denotes the number of non-overlapping patches of size d×d×c per sequence, where c is the number of input channels and d is a hyperparameter that determines the relationship between N and $n , \mathrm { A s }$ described in Section IV-B, the noise-conditional score transformer model $s _ { \theta } ( \cdot , \sigma )$ is trained via denoising score matching to estimate the score function $\nabla _ { \tilde { x } } \log q _ { \sigma } ( \tilde { x } )$ for noisy patches. This is achieved by minimizing the following weighted score-matching loss over L noise levels and $N$ patches per sequence:

$$
\begin{array} { r l } & { \mathcal { L } _ { \operatorname* { m i n } } = \displaystyle \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \frac { \sigma _ { i } ^ { 2 } } { 2 } \sum _ { j = 1 } ^ { N } \omega _ { j } \mathbb { E } _ { q _ { \sigma _ { i } } ( \tilde { P } _ { i n } ^ { j } | x ) p ( x ) } } \\ & { \qquad \quad \left\| s _ { \theta } ( \tilde { P } _ { i n } ^ { j } , \sigma _ { i } ) + \frac { \tilde { \mathbf { p } } _ { t a r } ^ { j } - \mathbf { p } _ { t a r } ^ { j } } { \sigma _ { i } ^ { 2 } } \right\| ^ { 2 } } \end{array}\tag{12}
$$

To enhance motion sensitivity, we reuse the key-frame motion weighting scheme from Section IV-E, which remains applicable here because anomaly cues in both detection and anticipation tasks are strongly linked to motion variations. For each input sequence $x _ { \mathrm { i n } } ,$ the motion saliency weights $\omega _ { j }$ are computed according to $\mathrm { E q . ~ } ( 7 ) \mathrm { , ~ E q . ~ } ( 8 )$ , and Eq. (9). Reweighting each patch-level term with $\omega _ { j }$ emphasizes dynamic regions while suppressing static background, improving motion sensitivity and anomaly discriminability.

Training proceeds in two stages. In the first stage, the main decoder is trained using the motion-aware objective defined in Eq. (12). In the second stage, the shared backbone (i.e., the encoder and main decoder) is frozen, and the auxiliary decoder is trained via self-distillation to reconstruct the denoised target patches $\begin{array} { r } { \hat { \mathbf { p } } _ { t a r } ^ { j } = \tilde { \mathbf { p } } _ { t a r } ^ { j } + \sigma _ { i } ^ { 2 } s _ { \theta } \big ( \tilde { \mathbf { p } } _ { t a r } ^ { j } , \sigma _ { i } \big ) } \end{array}$ generated by the main decoder. Let $\{ \bar { \mathbf { p } } _ { t a r } ^ { j } \} _ { j = 1 } ^ { N }$ denote the output of the auxiliary decoder. The self-distillation loss is formulated as:

$$
\mathcal { L } _ { \mathrm { S D } } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \left\{ \sum _ { j = 1 } ^ { N } \omega _ { j } \cdot \left. \bar { \mathbf { p } } _ { t a r } ^ { j } - \hat { \mathbf { p } } _ { t a r } ^ { j } \right. _ { 2 } ^ { 2 } \right\}\tag{13}
$$

where $\bar { \mathbf { p } } _ { t a r } ^ { j }$ and $\hat { \mathbf { p } } _ { t a r } ^ { j }$ represent the patches denoised by the auxiliary and main branches, respectively. The motion weight $\omega _ { j }$ is reused from Eq. (9).

As an integral component of the unified Uni-DSM framework illustrated in Fig. 3, we employ the shared noiseconditioned score transformer (NCST) backbone to drive the VAA task. Leveraging its attention-based mechanisms, this architecture effectively constructs a stable vector field through consecutive supervision of the multiscale patch-wise denoising process, thereby making anomalies clearly distinguishable in video action anticipation (VAA) tasks.

## H. Score-Based Anomaly Inference

In both video anomaly detection (VAD) and video anomaly anticipation (VAA), anomaly inference is fundamentally grounded in the learned data density. Since our model estimates the score function $s _ { \theta } ( x , \sigma ) \ = \ \nabla _ { x } \log p ( x )$ , anomaly evidence can be directly derived from the magnitude of the log-density gradient:

$$
\| s _ { \theta } ( x ) \| = \| \nabla _ { x } \log p ( x ) \| = \left\| \frac { \nabla _ { x } p ( x ) } { p ( x ) } \right\|\tag{14}
$$

Intuitively, samples that lie in high-density regions corresponding to normal events tend to produce small score magnitudes, whereas samples in low-density regions corresponding to anomalous events yield stronger responses.

In VAD, the entire video sequence is observable, enabling direct score evaluation on each frame. Given a test video, we divide it into η segments of $T$ frames each. For segment t, we compute $L$ scores $\{ \mathrm { s c o r e } _ { i } ( t ) \} _ { i = 1 } ^ { L }$ across noise levels and take the maximum per level to enhance separability [13], [63]:

$$
\mathrm { s c o r e } _ { i } ( t ) = \operatorname* { m a x } _ { 1 \leq \tau \leq T } \mathrm { s c o r e } _ { i } ( \tau )\tag{15}
$$

Each score $\mathbf { \nabla } _ { i } ( t )$ is then normalized to $[ 0 , 1 ]$ to obtain the calibrated anomaly probability [64]:

$$
\begin{array} { l l } { S _ { i } ( t ) = \displaystyle \mathrm { n o r m a l i z e } \big ( \mathrm { s c o r e } _ { i } ( t ) \big ) } \\ { \displaystyle = \frac { \mathrm { s c o r e } _ { i } ( t ) - \operatorname* { m i n } _ { 1 \leq \tau \leq \eta T } \mathrm { s c o r e } _ { i } ( \tau ) } { \displaystyle \operatorname* { m a x } _ { 1 \leq \tau \leq \eta T } \mathrm { s c o r e } _ { i } ( \tau ) - \displaystyle \operatorname* { m i n } _ { 1 \leq \tau \leq \eta T } \mathrm { s c o r e } _ { i } ( \tau ) } } \end{array}\tag{16}
$$

Finally, we weigh and average these anomaly scores $S _ { i } ( t )$ to obtain an ultimate anomaly indicator, which is strong enough for VAD. This yields a robust, multi-scale anomaly indicator suitable for VAD, as summarized in Algorithm 1.

Algorithm 1 Autoregressive Denoising Score Matching for   
VAD   
Require: Number of noise levels $\overline { { L ; } }$   
1: ϵ ← sample $\mathcal { N } ( 0 , I ) ;$   
2: $\sigma \sim$ sample log-uniform $( \sigma _ { 1 } , \sigma _ { L } ) ;$   
3: Input video sequence $x _ { t } = \{ f _ { t - n } , \ldots , f _ { t - 1 } \}$   
4: Score network $s _ { \theta }$   
Ensure: Anomaly score $S ( t )$   
5: $x _ { t } = \{ f _ { t - n } , . . . , f _ { t - 1 } \}$   
6: $\hat { x } _ { t } \gets x _ { t }$   
7: for $i = 1$ to L do   
8: $\tilde { x } _ { t } \gets \hat { x } _ { t } + \sigma _ { i } \epsilon$   
9: $\hat { x } _ { t } \gets \tilde { x } _ { t } + \sigma _ { i } ^ { 2 } s _ { \theta } ( \tilde { x } _ { t } , \sigma _ { i } )$   
10: scor $\mathsf { z } _ { i } ( t ) \gets \| s _ { \theta } ( \tilde { x } _ { t } , \sigma _ { i } ) \| / \mathrm { P S N R } ( \hat { x } _ { t } , x _ { t } )$   
11: end for   
12: $S ( t ) \gets$ normalize $\left( \{ \mathrm { s c o r e } _ { i } ( t ) \} _ { i = 1 } ^ { L } \right)$   
13: return $S ( t )$

Unlike VAD, VAA operates under partial observability where future frames are unavailable at inference time. Consequently, direct comparison with future ground truth cannot be directly applied. Nevertheless, the score function remains well-defined over the learned distribution. Therefore, we retain the score norm normalized by visual fidelity as the primary anomaly metric.

However, an additional challenge arises in anticipation because the score function inherently approaches zero near local density modes. In detection, such cases can often be compensated by reconstruction discrepancy, whereas in anticipation, no future ground truth is available to provide this corrective signal. As a result, anomalies residing in local modes may remain undetected. To address this limitation while preserving the probabilistic foundation of inference, we leverage the SDSM mechanism. For each perturbed input $\tilde { x } _ { \mathrm { i n } } ^ { t }$ and target $\tilde { x } _ { \mathrm { t a r } } ^ { t }$ at noise level $\sigma _ { i } ,$ the main decoder generates the denoised target $\hat { x } _ { \mathrm { t a r } } ^ { t } = \tilde { x } _ { \mathrm { t a r } } ^ { t } + \sigma _ { i } ^ { 2 } s _ { \theta } ( \tilde { x } _ { \mathrm { i n } } ^ { t } , \sigma _ { i } )$ , while the auxiliary decoder reconstructs $\bar { x } _ { t } ^ { \mathrm { t a r } }$ from $\hat { x } _ { \mathrm { t a r } } ^ { t }$ . The discrepancy between their outputs serves as a surrogate for anomaly probability. Crucially, in each denoising step, we use the PSNR between $\hat { x } _ { \mathrm { t a r } } ^ { t }$ and $\bar { x } _ { t } ^ { \mathrm { t a r } }$ as the denominator of the score norm, yielding the anomaly score:

$$
\mathrm { s c o r e } _ { i } ( t ) = \frac { \| s _ { \theta } ( \tilde { x } _ { \mathrm { i n } } ^ { t } , \sigma _ { i } ) \| } { P S N R ( \bar { x } _ { t } ^ { \mathrm { t a r } } , \hat { x } _ { t a r } ^ { t } ) }\tag{17}
$$

When the likelihood and appearance cues in the numerator and denominator are consistent, the model yields highly confident and reliable predictions for both normal and abnormal events. In contrast, when they diverge, the regions where the numerator misjudges are suppressed by the denominator, while the parts overlooked by the denominator are compensated by the numerator, resulting in a more robust and balanced score with the appearance gap filled.

For the VAA task, let α denote the anticipation horizon. We treat score<sub>i</sub> $( t + k )$ as the anomaly score for the k-th future frame $( k \in [ 1 , \alpha ] )$ . The predicted anomaly score for the window $[ t + 1 , t + \alpha ]$ is then defined as the maximum over the α frames:

Algorithm 2 Self-Distillation Denoising Score Matching for   
VAA   
Require: Number of noise levels $\overline { { L ; } }$   
1: ϵ ← sample ${ \mathcal { N } } ( 0 , I ) ;$   
2: σ ∼ sample log-uniform $( \sigma _ { 1 } , \sigma _ { L } ) ;$   
3: Input video sequence $x _ { \mathrm { i n . } } ^ { t } = \{ f _ { t - n } , \ldots , f _ { t - 1 } \} ;$   
4: Target video sequence $x _ { \mathrm { t a r } } ^ { t } = \{ f _ { t + 1 } , \ldots , f _ { t + n } \}$   
5: Main score network $s _ { \theta } ;$   
6: Sub score network $s _ { \theta } ^ { \prime }$   
Ensure: Anomaly score $S _ { i } ( t + 1 : t + \alpha )$   
7: $x _ { \mathrm { i n } } ^ { t } = \{ f _ { t - n } , . . . , f _ { t - 1 } \}$   
8: $x _ { \mathrm { t a r } } ^ { t } = \{ f _ { t + 1 } , \dotsc , f _ { t + n } \}$   
9: for $i = 1$ to L do   
10: $\tilde { x } _ { \mathrm { i n } } ^ { t } \gets x _ { \mathrm { i n } } ^ { t } + \sigma _ { i } \epsilon$   
11: $\hat { x } _ { t a r } ^ { t }  \tilde { x } _ { \mathrm { t a r } } ^ { t } + \sigma _ { i } ^ { 2 } s _ { \theta } ( \tilde { x } _ { \mathrm { i n } } ^ { t } , \sigma _ { i } )$   
12: $\bar { x } _ { \mathrm { t a r } } ^ { t }  s _ { \theta } ^ { \prime } ( \tilde { x } _ { \mathrm { i n } } ^ { t } , \sigma _ { i } )$   
13: sc $\mathbf { \sigma } ^ { \mathrm { { s r e } } _ { i } ( t ) } \gets \lVert s _ { \theta } ( \tilde { x } _ { t } ^ { \mathrm { { i n } } } , \sigma _ { i } ) \rVert / \mathrm { P S N R } ( \bar { x } _ { t } ^ { \mathrm { t a r } } , \hat { x } _ { t a r } ^ { t } )$   
14: end for   
15: score $i ( t + 1 : t + \alpha ) \gets \operatorname* { m a x } ( \mathrm { s c o r e } _ { i } ( t + k ) )$   
16: $S _ { i } ( t + 1 : t + \alpha ) \gets$ normalize(score $( t + 1 : t + \alpha ) )$   
17: return $S _ { i } ( t + 1 : t + \alpha )$

$$
\operatorname { s c o r e } _ { i } ( t + 1 : t + \alpha ) = \operatorname* { m a x } _ { 1 \leq k \leq \alpha } \operatorname { s c o r e } _ { i } ( t + k )\tag{18}
$$

Each score<sub>i</sub> $( t + 1 : t + \alpha )$ is normalized to [0, 1] over the temporal window $\tau \in [ t + 1 , t + \alpha ]$ to obtain the calibrated anomaly probability: Each $\mathrm { s c o r e } _ { i } ( t )$ is then normalized to [0, 1] to obtain the calibrated anomaly probability [64]:

$$
\begin{array} { r } { S _ { i } ( t + 1 : t + \alpha ) = \mathrm { n o r m a l i z e } \big ( \mathrm { s c o r e } _ { i } ( t + 1 : t + \alpha ) \big ) } \\ { \qquad \mathrm { s c o r e } _ { i } ( t + 1 : t + \alpha ) - \underset { t + 1 \leq \tau \leq t + \alpha } { \operatorname* { m i n } } \mathrm { s c o r e } _ { i } ( \tau ) } \\ { \qquad = \frac { \displaystyle \operatorname* { m a x } _ { t \leq \tau \leq t + \alpha } \mathrm { s c o r e } _ { i } ( \tau ) - \underset { t + 1 \leq \tau \leq t + \alpha } { \operatorname* { m i n } } \mathrm { s c o r e } _ { i } ( \tau ) } { \displaystyle \operatorname* { m a x } _ { t + 1 \leq \tau \leq t + \alpha } \mathrm { s c o r e } _ { i } ( \tau ) } . } \end{array}\tag{19}
$$

Finally, the overall anomaly probability for the anticipation window is obtained by computing a weighted average of the normalized scores across all noise levels. The entire process is detailed in Algorithm 2.

## V. EXPERIMENTS

## A. Experimental Setup

Datasets. To comprehensively evaluate our method, we utilize the CUHK Avenue dataset [37], the ShanghaiTech dataset [38] and the NWPU Campus dataset [12]. The CUHK Avenue dataset (hereafter referred to as the Avenue dataset) [37] is a widely used benchmark containing 16 training videos and 21 test videos, totaling 35,240 frames, with each video lasting approximately two minutes. Typical anomalies include running on sidewalks, throwing bags, and children jumping rope. The ShanghaiTech dataset [38] consists of 330 training videos and 107 test videos captured across 13 distinct scenes, with anomalies such as fighting, robbery, and brawling. The NWPU Campus dataset (hereafter referred to as the NWPU dataset) [12] is currently the largest semisupervised video anomaly detection benchmark, comprising

305 training videos and 242 test videos with 1,466,073 frames in total. It covers 43 different scenes with anomalies including jaywalking and illegal U-turns. Notably, it is the first dataset that explicitly considers scene-dependent anomalies and supports video anomaly anticipation (VAA).

Evaluation Metric. We adopt the Area Under the ROC Curve (AUC) metric for both VAD and VAA. In VAD, we evaluate performance using both macro-average and microaverage AUC [65], [66], [67], [68]. Macro-AUC computes the AUC for each test video individually and then averages across all videos, while Micro-AUC concatenates all frames from all test videos before computing a single AUC. For VAA, we concatenate all frames in the dataset and compute frame-level AUC.

Implementation details. Input frames $x _ { t }$ are cropped to $1 6 0 \times 1 6 0$ pixel regions centered on objects detected by the pre-trained ByteTrack [69], which is implemented in MM-Tracking [70]. This cropping strategy enables the model to focus on salient targets while implicitly capturing contextual scene information. If a frame contains multiple targets, each is processed independently. For VAD, our model $s _ { \theta }$ takes an 8-frame input sequence, partitioned into $N = 8 \times 1 0 0$ non-overlapping patches $\{ \mathbf { P } _ { j } \} _ { j = 1 } ^ { N } \in \mathbb { R } ^ { d \times d \times c }$ with patch size $d = 1 6$ . The model is trained for 100 epochs with a batch size of 20 using the Adamax optimizer [71]. The initial learning rate is set to $1 0 ^ { - 4 }$ and gradually decays following the scheme of cosine annealing. During training, the noise scale $\sigma _ { i }$ for each batch element is sampled from a log-uniform distribution over $[ \sigma _ { 1 } , \sigma _ { L } ] = [ 0 . 0 0 1 , 1 . 0 ]$ . During inference, we use $L = 2 0$ uniformly spaced noise levels within this interval. Both scene category embeddings and diffusion time embeddings have dimensions of 768 and are concatenated for model learning and inference. For VAA, the input sequence length is reduced to 5 frames $( N \ = \ 5 \times \ 1 0 0$ patches, $d \ = \ 1 6 )$ , and the model is trained for 50 epochs, while all other settings remain unchanged. All experiments are conducted on four NVIDIA GeForce RTX 4090 GPUs using the PyTorch [72] framework.

## B. VAD Performance Benchmarking

Table I presents the VAD performance comparison between our method and existing approaches on the Avenue [37], ShanghaiTech [38], and NWPU Campus [12] datasets. The results are reported using both the micro and macro scores. We also list the input modalities of different methods, where unmarked entries denote single-modality inputs, “⋄” and “+” indicate multimodal inputs incorporating optical flow and skeleton data, respectively. We prefer the macro score, especially on multi-scene datasets like ShanghaiTech [38] and NWPU [12], as ambiguous scene boundaries prevent micro score from comprehensively and fairly judging the performance. On all three datasets, each containing more than 10 anomaly categories, our method outperforms all competitors in terms of macro score. Notably, we achieve a significant improvement of up to 2.4% on the ShanghaiTech dataset [38], demonstrating effectiveness of our method. Moreover, on the NWPU dataset [12], our method attains the best performance on both metrics, validating the scalability and stability of our method. Compared with multimodal-input methods, our approach, using only raw video clips, achieves performance that is not only comparable but often superior. Against methods employing the same single-modality input, our method consistently achieves state-of-the-art results across all three datasets, obtaining the highest scores in both micro and macro evaluations. This outstanding performance confirms that our Uni-DSM can effectively detect anomalies in complex and challenging video anomaly detection tasks. It is worth noting that since the score is computed based on random noise, it inherently possesses a certain degree of uncertainty. However, the qualitative analysis in Section V-G verifies the validity of the denoising score matching mechanism. By fusing conditional embeddings with visual features, our method maximally controls this uncertainty. Furthermore, extensive testings reveal minimal performance fluctuations (<1% across multiple runs), confirming the stability and reliability of our reported results.

TABLE I  
COMPARISON OF DIFFERENT METHODS ON THE AVENUE, SHANGHAITECH, AND NWPU DATASETS IN AUCS (%) METRIC. THE BEST RESULT ON EACH DATASET IS SHOWN IN BOLD. WE ALSO LIST THE INPUTS OF THE METHODS, IN WHICH METHOD WITHOUT MARKING DENOTES SINGLE MODAL INPUT.
<table><tr><td rowspan="2">Method</td><td colspan="2">Avenue</td><td colspan="2">ShanghaiTech</td><td colspan="2">NWPU</td></tr><tr><td>Micro</td><td>Macro</td><td>Micro</td><td>Macro</td><td>Micro</td><td>Macro</td></tr><tr><td> $\overline { { \mathcal { O } _ { \mathrm { F F P } } \ : [ 1 4 ] } }$ </td><td>84.9</td><td></td><td>72.8</td><td></td><td></td><td></td></tr><tr><td>AMMC-Net [15]</td><td>86.6</td><td></td><td>73.7</td><td></td><td>64.5</td><td></td></tr><tr><td>HF2-VAD [16]</td><td>91.1</td><td>93.5</td><td>76.2</td><td>83.1</td><td>63.7</td><td></td></tr><tr><td>VABD [17]</td><td>86.6</td><td></td><td>78.2</td><td></td><td></td><td></td></tr><tr><td>+MTP [18]</td><td>82.9</td><td></td><td>76.0</td><td>一</td><td>一</td><td></td></tr><tr><td>+STG-NF [32]</td><td></td><td></td><td>85.9</td><td></td><td></td><td></td></tr><tr><td>+MoCoDAD [30]</td><td>89.0</td><td></td><td>77.9</td><td>1</td><td></td><td></td></tr><tr><td>MemAE [5]</td><td>84.9</td><td></td><td>72.8</td><td></td><td>61.9</td><td></td></tr><tr><td>CAE [11]</td><td>87.4</td><td>90.4</td><td>78.7</td><td>84.9</td><td></td><td></td></tr><tr><td>MNAD [20]</td><td>88.5</td><td></td><td>70.5</td><td></td><td>62.5</td><td></td></tr><tr><td>OG-Net [7]</td><td></td><td></td><td></td><td></td><td>62.5</td><td></td></tr><tr><td>MPN [22]</td><td>89.5</td><td></td><td>73.8</td><td></td><td>64.4</td><td></td></tr><tr><td>SSMTL [19]</td><td>91.5</td><td>91.9</td><td>82.4</td><td>89.3</td><td></td><td></td></tr><tr><td>LLSH [45]</td><td>87.4</td><td>88.6</td><td>77.6</td><td>85.9</td><td>62.2</td><td></td></tr><tr><td>FBAE [12]</td><td>86.8</td><td></td><td>79.2</td><td>1</td><td>68.2</td><td></td></tr><tr><td>FPDM [31]</td><td>90.1</td><td></td><td>78.6</td><td></td><td></td><td></td></tr><tr><td>Two-stream [9]</td><td>90.8</td><td>93.0</td><td>83.7</td><td>90.8</td><td></td><td></td></tr><tr><td>SDMAE [21]</td><td>91.3</td><td>90.9</td><td>79.1</td><td>84.7</td><td></td><td></td></tr><tr><td>SSAE [13]</td><td>90.2</td><td>91.5</td><td>80.5</td><td>88.9</td><td>75.6</td><td>76.9</td></tr><tr><td>Uni-DSM (ours)</td><td>91.6</td><td>94.2</td><td>84.5</td><td>93.2</td><td>76.9</td><td>78.1</td></tr></table>

TABLE II

ANTICIPATION RESULTS OF DIFFERENT METHODS ON THE NWPU DATASET
<table><tr><td>α</td><td>0.5s</td><td>1s</td><td>1.5s</td><td> $\overline { { 2 \mathrm { s } } }$ </td><td>2.5s</td><td> $\overline { { 3 \mathrm { s } } }$ </td></tr><tr><td>Chance</td><td>50.0</td><td>50.0</td><td>50.0</td><td>50.0</td><td>50.0</td><td>50.0</td></tr><tr><td>Human</td><td></td><td></td><td></td><td></td><td></td><td>90.4</td></tr><tr><td>FBAE [12]</td><td>65.8</td><td>65.3</td><td>64.9</td><td>64.6</td><td>64.2</td><td>64.0</td></tr><tr><td>SSAE [13]</td><td>69.5</td><td>69.3</td><td>68.7</td><td>68.2</td><td>67.1</td><td>66.9</td></tr><tr><td>Uni-DSM (ours)</td><td>71.3</td><td>71.1</td><td>70.5</td><td>70.2</td><td>69.5</td><td>69.1</td></tr></table>

## C. VAA Performance Benchmarking

To evaluate the effectiveness of our self-distilled DSM (SDSM) mechanism on the VAA task, we follow the standard video anomaly anticipation protocol on the NWPU Campus dataset [12], which requires predicting anomalies that occur within a specific future time window. Comparative experiments are conducted across different anticipation horizons. Tables II present the performance of the SDSM on the NWPU dataset [12], with a focus on comparing different methods in terms of anticipation accuracy. We also report the results of stochastic anticipations (“Chance”) and human beings (“Human”). Four volunteers not involved in the construction of the dataset participate in the evaluation of anomaly anticipation. Since humans cannot perceive time precisely, the volunteers only anticipate whether an anomalous event will occur in 3 seconds or not. The result of “Human” is the average performance of all the volunteers. For algorithms, there is still much room for improvement, indicating that the VAA task is very challenging. Notebly, SDSM consistently outperforms both the random baseline and existing learning-based approaches across all anticipation horizons from 0.5s to 3s. Compared with the forward-backward frame prediction framework FBAE [12], SDSM achieves a clear improvement of around 5%, while maintaining stable performance as the anticipation horizon increases. Furthermore, compared with the stronger bidirectional modeling approach SSAE [13], SDSM still yields consistent gains across all horizons, with an average improvement of 2.1%. Although performance naturally declines as the anticipation horizon extends, SDSM consistently outperforms all baselines and remains computationally efficient, with detailed efficiency analysis provided in Section V-F.

## D. Study on Latent Space

Latent-space generative models are widely used to remove redundancy from raw inputs and improve efficiency in synthesizing high-quality images and videos [60], [73], [74], [75]. However, VAD differs from typical generative tasks in that it is highly sensitive to the underlying data distribution, implying limited redundancy in the input. Consequently, projecting data into a compact latent space may discard subtle but critical cues for anomaly detection. These observations highlight the importance of directly modeling raw video signals. In practice, improvements in VAD performance often stem from stronger feature representations rather than more sophisticated likelihood modeling schemes. This motivates us to reconsider whether performing denoising score matching in the latent space of pre-trained generative models is necessary for VAD.

We adopt the variational autoencoder used in Stable Diffusion [73], pre-trained on the COCO dataset [76], to encode clean input sequences into latent features. Our ADSM, instantiated with the NCST shown in Fig. 3, is then applied within this latent representation space. Quantitative results in Table III, evaluated on the ShanghaiTech [38] and NWPU [12] benchmarks using both micro and macro metrics, show that operating in latent space yields inferior performance compared to modeling original data directly. This supports our hypothesis that the inconsistent distribution between the features extracted by the pre-trained model and the VAD dataset exacerbates the unstable performance of the latent version. Although representation alignment techniques [77] could alleviate this issue, directly implementing our ADSM on the original data space is more intuitive with the advancement of computing resources and model capabilities.

TABLE III  
AUCS (%) OF OUR MODEL IN LATENT SPACE AND ORIGINAL DATA SPACE ON SHANGHAITECH AND NWPU DATASETS.
<table><tr><td rowspan="2">Method</td><td colspan="2">ShanghaiTech</td><td colspan="2">NWPU</td></tr><tr><td>Micro</td><td>Macro</td><td>Micro</td><td>Macro</td></tr><tr><td>L-ADSM</td><td>77.6</td><td>87.5</td><td>68.1</td><td>73.2</td></tr><tr><td>Uni-DSM</td><td>84.5</td><td>93.2</td><td>76.9</td><td>78.1</td></tr></table>

TABLE IV

AUCS (%) OF DIFFERENT LIKELIHOOD-BASED METHODS IN THE ORIGINAL DATA SPACE ON SHANGHAITECH AND NWPU DATASETS.
<table><tr><td rowspan="2">Method</td><td colspan="2">ShanghaiTech</td><td colspan="2">NWPU</td></tr><tr><td>Micro</td><td>Macro</td><td>Micro</td><td>Macro</td></tr><tr><td>MSMA [33]</td><td>81.3</td><td>90.4</td><td>72.3</td><td>75.1</td></tr><tr><td>MULDE [34]</td><td>82.6</td><td>90.4</td><td>73.1</td><td>74.5</td></tr><tr><td>Uni-DSM</td><td>84.5</td><td>93.2</td><td>76.9</td><td>78.1</td></tr></table>

To further investigate this issue, we examine two likelihoodbased baselines closely related to our framework, including a latent-space method [34] and an approach originally proposed for image anomaly detection [33]. For fair comparison, both methods are re-implemented under consistent experimental settings and evaluated on the ShanghaiTech and NWPU benchmarks. Results in Table IV show that our method consistently outperforms these baselines, demonstrating that applying denoising score matching in the original data space is more effective. The original data space enables us to take both the efficiency of likelihood estimation and the characteristic of video anomalies into account, further supporting our motivation.

## E. Ablation Studies

To further verify the effectiveness of each specific design of Uni-DSM, we conduct ablation experiments to compare Uni-DSM with its several model variants. Table V reports the detection performance of model variants on the ShanghaiTech dataset [38]. Components are progressively added to the noise-conditional score transformer, including denoising score matching (DSM), autoregressive denoising score matching (ADSM), scene-conditioned embedding (Scene), key-frame motion weighting (Motion), and visual difference aggregation (Appearance). As shown in Table V, the autoregressive mechanism significantly improves DSM for visual difference aggregation in VAD, yielding an average gain of 4.1% in both micro and macro scores. Although ShanghaiTech does not explicitly emphasize scene-dependent anomalies, the macro score still increases by 2.2%, suggesting that ADSM captures scene-aware information more effectively. Introducing motion consistency further improves performance by 2.9%, indicating that ADSM effectively associates spatiotemporal information. Aggregating the discrepancy between denoised and original data proves to be the most effective strategy for constructing a discriminative anomaly metric, improving micro and macro scores by 5.7% and 5.8%, respectively. When all components are combined, the micro and macro scores further increase by

TABLE V  
AUCS(%) OF OUR MODEL VARIANTS ON THE SHANGHAITECH DATASET FOR THE VAD TASK.
<table><tr><td rowspan="2"></td><td rowspan="2">DSM</td><td rowspan="2">ADSM</td><td rowspan="2">Scene</td><td rowspan="2">Motion</td><td rowspan="2">Appearance</td><td colspan="2">ShanghaiTech</td></tr><tr><td>Micro</td><td>Macro</td></tr><tr><td>1</td><td>√</td><td></td><td></td><td></td><td></td><td>73.9</td><td>78.5</td></tr><tr><td>2</td><td>√</td><td>√</td><td></td><td></td><td></td><td>76.4</td><td>84.1</td></tr><tr><td>3</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>76.9</td><td>86.3</td></tr><tr><td>4</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td>78.6</td><td>87.7</td></tr><tr><td>5</td><td>√</td><td>√</td><td>L</td><td></td><td>√</td><td>82.1</td><td>89.9</td></tr><tr><td>6</td><td>√</td><td>√</td><td></td><td>√</td><td>√</td><td>84.5</td><td>93.2</td></tr></table>

TABLE VI

AUCS(%) OF OUR MODEL VARIANTS ON THE NWPU DATASET FOR THE VAA TASK.
<table><tr><td></td><td>DSM</td><td>SDSM</td><td>Scene</td><td>Motion</td><td>NWPU datasets</td></tr><tr><td>1</td><td>√</td><td></td><td></td><td></td><td>67.5</td></tr><tr><td>2</td><td>√</td><td>√</td><td></td><td></td><td>68.1</td></tr><tr><td>3</td><td>√</td><td>√</td><td>√</td><td></td><td>68.6</td></tr><tr><td>4</td><td>√</td><td>√</td><td></td><td>√</td><td>69.3</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>70.5</td></tr></table>

2.4% and 3.3%, respectively, demonstrating strong complementarity among the modules of ADSM.

Table VI reports the anticipation performance of model variants on the NWPU dataset [12]. Components are progressively added to the noise-conditional score transformer, including denoising score matching (DSM), self-distilled denoising score matching (SDSM), scene-conditioned embedding (Scene), and key-frame motion weighting (Motion) to validate their contributions. Each module improves performance. The DSM baseline achieves 67.5% accuracy. Adding SDSM increases accuracy to 68.1%, indicating that self-distillation optimizes denoising and enhances anticipation. Scene-conditioned embedding raises performance to 68.6%, highlighting the auxiliary role of scene context. Key-frame motion weighting further increases accuracy to 69.3%, emphasizing critical motion cues for discriminative temporal features. When all modules are integrated, the model achieves 70.5%, consistently outperforming both individual modules and their partial combinations. This demonstrates strong complementarity among the modules, resulting in a more robust and accurate anomaly anticipation mechanism.

In summary, the ablation studies systematically validate the effectiveness of each proposed component. Their incremental integration consistently yields substantial performance gains, and the full combination achieves the best results, validating the effectiveness of our model architecture.

## F. Computational Resource Analysis

To evaluate the computational cost of the model, we conducted a comparative analysis in both detection and anticipation tasks. For the VAD task, Uni-DSM operates directly on raw video clips without additional feature extractors. It uses a generative model only for score prediction rather than high fidelity generation, simplifying denoising and reducing computational cost. The runtime is mainly determined by the size of the noise conditional score transformer. In this setup, Uni-DSM uses an NCST with 130 million parameters and processes 8 frames in under 20 milliseconds, achieving 50 FPS. With the added computation of ByteTrack, the speed remains around 35 FPS, meeting typical real time surveillance requirements of about 30 FPS. We also compare inference speed and micro score with five representative methods including $\mathrm { H F ^ { 2 }  – V A D }$ [16], VABD [17], LLSH [45] , FPDM [31] , and SSAE [13] on the ShanghaiTech dataset. As shown in Table VII, Uni-DSM achieves the best performance of 84.5% with competitive speed. Although VABD achieves the highest speed, it relies on pre-extracted optical flow, and its runtime would drop significantly if the extraction cost is considered. Our method captures characteristics of video anomalies while maintaining stable speed without additional input modalities.

TABLE VII  
COMPARISON OF COMPUTATIONAL RESOURCE CONSUMPTION DURING VAD INFERENCE ON THE SHANGHAITECH DATASET.
<table><tr><td>Model</td><td>GPU Memory (G)</td><td>Running Speed (s)</td><td>Frame Rate (FPS)</td><td>Performance (%)</td></tr><tr><td>HF2-VAD [16]</td><td></td><td>0.0667</td><td>15.0</td><td>76.2</td></tr><tr><td>VABD [17]</td><td>一 一</td><td>0.0147</td><td>68.0</td><td>78.2</td></tr><tr><td>LLSH [45]</td><td>一</td><td>0.0392</td><td>25.5</td><td>77.6</td></tr><tr><td>FPDM [31]</td><td></td><td>0.1282</td><td>7.8</td><td>78.6</td></tr><tr><td>SSAE [13]</td><td>24</td><td>0.1096</td><td>10.1</td><td>80.5</td></tr><tr><td>Uni-DSM (ours)</td><td>11</td><td>0.0194</td><td>51.5</td><td>84.5</td></tr></table>

TABLE VIII

COMPARISON OF COMPUTATIONAL RESOURCE CONSUMPTION DURING VAA INFERENCE ON THE NWPU DATASET
<table><tr><td>Model</td><td>GPU Memory (G)</td><td>Running Speed (s)</td><td>Frame Rate (FPS)</td><td>Performance (%)</td></tr><tr><td>EPAP-Net [36]</td><td>一</td><td>0.0219</td><td>45.7</td><td>1</td></tr><tr><td>FBAE [12]</td><td>21</td><td>0.0274</td><td>36.5</td><td>64.9</td></tr><tr><td>SSAE [13]</td><td>24</td><td>0.1096</td><td>10.1</td><td>68.7</td></tr><tr><td>SDSM (ours)</td><td>16</td><td>0.0163</td><td>61.3</td><td>70.5</td></tr></table>

For the VAA task, Uni-DSM demonstrates significant advantages in computational efficiency and resource utilization, as detailed in Table VIII. Specifically, its GPU memory consumption is only 16 GB, substantially lower than the 21 GB of FBAE [12] and the 24 GB of SSAE [13], reflecting effective control over parameter count and computational complexity. Simultaneously, Uni-DSM achieves the fastest inference speed of 0.0163 seconds per frame, corresponding to 61.3 FPS, which markedly outperforms competing methods. This indicates that Uni-DSM achieves both strong anticipation performance and high computational efficiency, making it suitable for real-world deployment under stringent real-time constraints.

## G. Qualitative Analysis

To provide intuitive insights into the detection and anticipation capabilities of Uni-DSM, we present qualitative visualizations of anomaly score dynamics. For the VAD task, we visualize anomaly score curves of ADSM on the NWPU dataset [12] for qualitative analysis. We select three clips to cover scene-dependent cyclist in D013 03, motion anomaly climber in D001 03, and appearance anomaly dog in

![](images/cce926f23bf8c1be0efaecf2689d9b737c5e7edc82946efb545c3232a447ed6c.jpg)  
Fig. 8. Visualization results on the NWPU dataset for the VAD task. The red boxes indicate that an anomaly is occurring, while the green boxes denote that the anomaly has disappeared.

![](images/8581ccd497b2a08676928c76e25b32da63381abfaf2040a4a0d236af3b9b7a8d.jpg)  
Fig. 9. Visualization results on the NWPU dataset for the VAA task. The orange boxes indicate that an anomaly is about to occur, the red boxes denote that an anomaly is currently happening, and the yellow boxes signify that the anomaly is disappearing.

D003 05. As shown in Fig. 8, the cyclist that is abnormal on the plaza but normal on roads is detected promptly without false alarms. For the anomalous motion of the climber and the anomalous appearance of the dog, anomaly scores exhibit a sharp increase upon the occurrence of abnormal events and return to flat when the abnormal events disappear. These results show that the method captures the characteristics of video anomalies in terms of scene, motion, and appearance.

For the VAA task, we visualize anticipated anomaly score curves of SDSM for qualitative evaluation on three representative clips including wall climbing in D014 01, lake disturbance in D122 05, and theft in D149 05. As shown in Fig. 9, the curves characterize the evolution of predicted anomaly likelihood within a given anticipation horizon. Notably, the anticipated curves are smoother than VAD curves, as SDSM performs anticipation over a continuous temporal window rather than discrete frames, resulting in reduced fluctuations. The score initially remains at a low baseline, then rises rapidly before anomaly onset and peaks around that point, stays elevated during the predicted interval, and gradually decreases as the scene returns to normal. This result highlights SDSM’s capability for proactive anomaly anticipation. Rather than passively reacting to observed events, the score evolves coherently with the anticipated emergence, progression, and resolution of anomalies. The early rise before visible onset reflects sensitivity to subtle precursory cues, while the sustained high values indicate continuous probabilistic forecasting over the anomaly duration. Meanwhile, consistently low scores on normal segments demonstrate effective suppression of false positives, and sharp transitions further reveal responsiveness to abnormal patterns.

## VI. CONCLUSION

In this paper, we present Uni-DSM, a unified score-driven framework grounded in denoising score matching that jointly addresses video anomaly detection and anticipation from a distributional perspective. Central to our approach is a shared noise-conditioned score transformer (NCST) backbone, enhanced by scene-dependent embeddings and motion-aware weights to ensure temporal consistency and robustness. For detection, we instantiate an autoregressive DSM (ADSM) mechanism that progressively accumulates anomalous context and integrates discrepancies between denoised outputs and original inputs to suppress local modes and enhance appearance perception. For anticipation, we propose a self-distilled DSM (SDSM) mechanism that employs a lightweight mainauxiliary architecture for unidirectional inference, constructing supervision from output discrepancies without relying on future ground truth. Extensive experiments demonstrate state-ofthe-art performance across both tasks with improved inference speed and reduced computational overhead. Ultimately, our work establishes a scalable, unified paradigm that effectively integrates scene, motion, and appearance cues into a comprehensive score-driven framework.

## REFERENCES

[1] V. Chandola, A. Banerjee, and V. Kumar, “Anomaly detection: A survey,” ACM computing surveys (CSUR), vol. 41, no. 3, pp. 1–58, 2009.

[2] A. Adam, E. Rivlin, I. Shimshoni, and D. Reinitz, “Robust real-time unusual event detection using multiple fixed-location monitors,” IEEE transactions on pattern analysis and machine intelligence, vol. 30, no. 3, pp. 555–560, 2008.

[3] V. Saligrama, J. Konrad, and P.-M. Jodoin, “Video anomaly identification,” IEEE Signal Processing Magazine, vol. 27, no. 5, pp. 18–33, 2010.

[4] B. Ramachandra, M. J. Jones, and R. R. Vatsavai, “A survey of singlescene video anomaly detection,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 5, pp. 2293–2312, 2020.

[5] D. Gong, L. Liu, V. Le, B. Saha, M. R. Mansour, S. Venkatesh, and A. v. d. Hengel, “Memorizing normality to detect anomaly: Memoryaugmented deep autoencoder for unsupervised anomaly detection,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 1705–1714.

[6] T.-N. Nguyen and J. Meunier, “Anomaly detection in video sequence with appearance-motion correspondence,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 1273– 1283.

[7] M. Z. Zaheer, J.-h. Lee, M. Astrid, and S.-I. Lee, “Old is gold: Redefining the adversarially learned one-class classifier training paradigm,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 14 183–14 193.

[8] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury, and L. S. Davis, “Learning temporal regularity in video sequences,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 733–742.

[9] C. Cao, Y. Lu, and Y. Zhang, “Context recovery and knowledge retrieval: A novel two-stream framework for video anomaly detection,” IEEE Transactions on Image Processing, 2024.

[10] G. Wang, Y. Wang, J. Qin, D. Zhang, X. Bao, and D. Huang, “Video anomaly detection by solving decoupled spatio-temporal jigsaw puzzles,” in European Conference on Computer Vision. Springer, 2022, pp. 494–511.

[11] R. T. Ionescu, F. S. Khan, M.-I. Georgescu, and L. Shao, “Object-centric auto-encoders and dummy anomalies for abnormal event detection in video,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 7842–7851.

[12] C. Cao, Y. Lu, P. Wang, and Y. Zhang, “A new comprehensive benchmark for semi-supervised video anomaly detection and anticipation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 20 392–20 401.

[13] C. Cao, H. Zhang, Y. Lu, P. Wang, and Y. Zhang, “Scene-dependent prediction in latent space for video anomaly detection and anticipation,” IEEE transactions on pattern analysis and machine intelligence, 2024.

[14] W. Liu, W. Luo, D. Lian, and S. Gao, “Future frame prediction for anomaly detection–a new baseline,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 6536– 6545.

[15] R. Cai, H. Zhang, W. Liu, S. Gao, and Z. Hao, “Appearance-motion memory consistency network for video anomaly detection,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 2, 2021, pp. 938–946.

[16] Z. Liu, Y. Nie, C. Long, Q. Zhang, and G. Li, “A hybrid video anomaly detection framework via memory-augmented flow reconstruction and flow-guided frame prediction,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 13 588–13 597.

[17] J. Li, Q. Huang, Y. Du, X. Zhen, S. Chen, and L. Shao, “Variational abnormal behavior detection with motion consistency,” IEEE Transactions on Image Processing, vol. 31, pp. 275–286, 2021.

[18] R. Rodrigues, N. Bhargava, R. Velmurugan, and S. Chaudhuri, “Multitimescale trajectory prediction for abnormal human activity detection,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2020, pp. 2626–2634.

[19] M.-I. Georgescu, A. Barbalau, R. T. Ionescu, F. S. Khan, M. Popescu, and M. Shah, “Anomaly detection in video via self-supervised and multitask learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12 742–12 752.

[20] H. Park, J. Noh, and B. Ham, “Learning memory-guided normality for anomaly detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 14 372–14 381.

[21] N.-C. Ristea, F.-A. Croitoru, R. T. Ionescu, M. Popescu, F. S. Khan, M. Shah et al., “Self-distilled masked auto-encoders are efficient video anomaly detectors,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 15 984–15 995.

[22] H. Lv, C. Chen, Z. Cui, C. Xu, Y. Li, and J. Yang, “Learning normal dynamics in videos with meta prototype network,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 15 425–15 434.

[23] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840– 6851, 2020.

[24] Y. Song and S. Ermon, “Generative modeling by estimating gradients of the data distribution,” Advances in neural information processing systems, vol. 32, 2019.

[25] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” arXiv preprint arXiv:2011.13456, 2020.

[26] D. Kingma, T. Salimans, B. Poole, and J. Ho, “Variational diffusion models,” Advances in neural information processing systems, vol. 34, pp. 21 696–21 707, 2021.

[27] P. Dhariwal and A. Nichol, “Diffusion models beat gans on image synthesis,” Advances in neural information processing systems, vol. 34, pp. 8780–8794, 2021.

[28] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, “Image super-resolution via iterative refinement,” IEEE transactions on pattern analysis and machine intelligence, vol. 45, no. 4, pp. 4713–4726, 2022.

[29] B. Zong, Q. Song, M. R. Min, W. Cheng, C. Lumezanu, D. Cho, and H. Chen, “Deep autoencoding gaussian mixture model for unsupervised anomaly detection,” in International conference on learning representations, 2018.

[30] A. Flaborea, L. Collorone, G. M. D. Di Melendugno, S. D’Arrigo, B. Prenkaj, and F. Galasso, “Multimodal motion conditioned diffusion model for skeleton-based video anomaly detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 10 318–10 329.

[31] C. Yan, S. Zhang, Y. Liu, G. Pang, and W. Wang, “Feature prediction diffusion model for video anomaly detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 5527–5537.

[32] O. Hirschorn and S. Avidan, “Normalizing flows for human pose anomaly detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 13 545–13 554.

[33] A. Mahmood, J. Oliva, and M. Styner, “Multiscale score matching for out-of-distribution detection,” arXiv preprint arXiv:2010.13132, 2020.

[34] J. Micorek, H. Possegger, D. Narnhofer, H. Bischof, and M. Kozinski, “Mulde: Multiscale log-density estimation via denoising score matching for video anomaly detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 18 868– 18 877.

[35] Q. Liu, J. Lee, and M. Jordan, “A kernelized stein discrepancy for goodness-of-fit tests,” in International conference on machine learning. PMLR, 2016, pp. 276–284.

[36] J. Leng, M. Tan, X. Gao, W. Lu, and Z. Xu, “Anomaly warning: Learning and memorizing future semantic patterns for unsupervised exante potential anomaly prediction,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 6746–6754.

[37] C. Lu, J. Shi, and J. Jia, “Abnormal event detection at 150 fps in matlab,” in Proceedings of the IEEE international conference on computer vision, 2013, pp. 2720–2727.

[38] W. Luo, W. Liu, and S. Gao, “A revisit of sparse coding based anomaly detection in stacked rnn framework,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 341–349.

[39] H. Zhang, C. Cao, Q. Lv, L. Min, and Y. Zhang, “Autoregressive denoising score matching is a good video anomaly detector,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 12 057–12 067.

[40] W. Sultani, C. Chen, and M. Shah, “Real-world anomaly detection in surveillance videos,” in Proceedings ofthe IEEE conference on computer vision and pattern recognition, 2018, pp. 6479–6488.

[41] J.-X. Zhong, N. Li, W. Kong, S. Liu, T. H. Li, and G. Li, “Graph convolutional label noise cleaner: Train a plug-and-play action classifier for anomaly detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 1237–1246.

[42] Y. Zhu and S. Newsam, “Motion-aware feature for improved video anomaly detection,” arXiv preprint arXiv:1907.10211, 2019.

[43] S. Zhou, W. Shen, D. Zeng, M. Fang, Y. Wei, and Z. Zhang, “Spatial– temporal convolutional neural networks for anomaly detection and localization in crowded scenes,” Signal Processing: Image Communication, vol. 47, pp. 358–368, 2016.

[44] S. Smeureanu, R. T. Ionescu, M. Popescu, and B. Alexe, “Deep appearance features for abnormal behavior detection in video,” in International conference on image analysis and processing. Springer, 2017, pp. 779– 789.

[45] Y. Lu, C. Cao, Y. Zhang, and Y. Zhang, “Learnable locality-sensitive hashing for video anomaly detection,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 2, pp. 963–976, 2022.

[46] Y. Chang, Z. Tu, W. Xie, and J. Yuan, “Clustering driven deep autoencoder for video anomaly detection,” in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XV 16. Springer, 2020, pp. 329–345.

[47] P. Wu, J. Liu, and F. Shen, “A deep one-class neural network for anomalous event detection in complex scenes,” IEEE transactions on neural networks and learning systems, vol. 31, no. 7, pp. 2609–2622, 2019.

[48] D. Xu, E. Ricci, Y. Yan, J. Song, and N. Sebe, “Learning deep representations of appearance and motion for anomalous event detection,” arXiv preprint arXiv:1510.01553, 2015.

[49] W. Li, V. Mahadevan, and N. Vasconcelos, “Anomaly detection and localization in crowded scenes,” IEEE transactions on pattern analysis and machine intelligence, vol. 36, no. 1, pp. 18–32, 2013.

[50] R. Hinami, T. Mei, and S. Satoh, “Joint detection and recounting of abnormal events by learning deep generic knowledge,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 3619– 3627.

[51] Z. Fang, J. Liang, J. T. Zhou, Y. Xiao, and F. Yang, “Anomaly detection with bidirectional consistency in videos,” IEEE transactions on neural networks and learning systems, vol. 33, no. 3, pp. 1079–1092, 2020.

[52] W. Luo, W. Liu, D. Lian, and S. Gao, “Future frame prediction network for video anomaly detection,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 11, pp. 7505–7520, 2021.

[53] H. Song, C. Sun, X. Wu, M. Chen, and Y. Jia, “Learning normal patterns via adversarial attention-based autoencoder for abnormal event detection in videos,” IEEE Transactions on Multimedia, vol. 22, no. 8, pp. 2138– 2148, 2019.

[54] G. Yu, S. Wang, Z. Cai, E. Zhu, C. Xu, J. Yin, and M. Kloft, “Cloze test helps: Effective video anomaly detection via learning to complete video events,” in Proceedings of the 28th ACM international conference on multimedia, 2020, pp. 583–591.

[55] Y. Jiang, X. Yan, G.-P. Ji, K. Fu, M. Sun, H. Xiong, D.-P. Fan, and F. S. Khan, “Effectiveness assessment of recent large vision-language models,” Visual Intelligence, vol. 2, no. 1, p. 17, 2024.

[56] P. Vincent, “A connection between score matching and denoising autoencoders,” Neural computation, vol. 23, no. 7, pp. 1661–1674, 2011.

[57] A. Hyvarinen and P. Dayan, “Estimation of non-normalized statistical¨ models by score matching.” Journal of Machine Learning Research, vol. 6, no. 4, 2005.

[58] A. Vaswani, “Attention is all you need,” Advances in Neural Information Processing Systems, 2017.

[59] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4195–4205.

[60] X. Ma, Y. Wang, G. Jia, X. Chen, Z. Liu, Y.-F. Li, C. Chen, and Y. Qiao, “Latte: Latent diffusion transformer for video generation,” arXiv preprint arXiv:2401.03048, 2024.

[61] S. Sun and X. Gong, “Hierarchical semantic contrast for scene-aware video anomaly detection,” in Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, 2023, pp. 22 846–22 856.

[62] G. Liu and J. Zhao, “Key frame extraction from mpeg video stream,” in 2010 Third International Symposium on Information Processing. IEEE, 2010, pp. 423–427.

[63] Z. Yang, J. Liu, Z. Wu, P. Wu, and X. Liu, “Video event restoration based on keyframes for video anomaly detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 14 592–14 601.

[64] M. Mathieu, C. Couprie, and Y. LeCun, “Deep multi-scale video prediction beyond mean square error,” arXiv preprint arXiv:1511.05440, 2015.

[65] A. Acsintoae, A. Florescu, M.-I. Georgescu, T. Mare, P. Sumedrea, R. T. Ionescu, F. S. Khan, and M. Shah, “Ubnormal: New benchmark for supervised open-set video anomaly detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 20 143–20 153.

[66] A. Barbalau, R. T. Ionescu, M.-I. Georgescu, J. Dueholm, B. Ramachandra, K. Nasrollahi, F. S. Khan, T. B. Moeslund, and M. Shah, “Ssmtl++: Revisiting self-supervised multi-task learning for video anomaly detection,” Computer Vision and Image Understanding, vol. 229, p. 103656, 2023.

[67] M. I. Georgescu, R. T. Ionescu, F. S. Khan, M. Popescu, and M. Shah, “A background-agnostic framework with adversarial training for abnormal event detection in video,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 9, pp. 4505–4523, 2021.

[68] T. Reiss and Y. Hoshen, “Attribute-based representations for accurate and interpretable video anomaly detection,” arXiv preprint arXiv:2212.00789, 2022.

[69] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, “Bytetrack: Multi-object tracking by associating every detection box,” in European conference on computer vision. Springer, 2022, pp. 1–21.

[70] M. Contributors, “Mmtracking: Openmmlab video perception toolbox and benchmark,” in https://github.com/open-mmlab/mmtracking, 2020. [Online]. Available: https://github.com/open-mmlab/mmtracking

[71] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.

[72] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in Neural Information Processing Systems, vol. 32, 2019.

[73] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “Highresolution image synthesis with latent diffusion models,” in Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[74] F. Bao, S. Nie, K. Xue, Y. Cao, C. Li, H. Su, and J. Zhu, “All are worth words: A vit backbone for diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 22 669–22 679.

[75] F. Bao, S. Nie, K. Xue, C. Li, S. Pu, Y. Wang, G. Yue, Y. Cao, H. Su, and J. Zhu, “One transformer fits all distributions in multi-modal diffusion at scale,” in International Conference on Machine Learning. PMLR, 2023, pp. 1692–1717.

[76] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft coco: Common objects in´ context,” in Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer, 2014, pp. 740–755.

[77] S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie, “Representation alignment for generation: Training diffusion transformers is easier than you think,” arXiv preprint arXiv:2410.06940, 2024.

![](images/88c28597785c0164d470778f81b27f57b9366967c327c4007de15dd397699e93.jpg)

Congqi Cao (Member, IEEE) received the BE degree in information and communication from Zhejiang University, Hangzhou, China, in 2013, and the PhD degree in pattern recognition and intelligent systems from the Institute of Automation, Chinese Academy of Sciences, in 2018. Then, she joined the School of Computer Science, Northwestern Polytechnical University, Xi’an, China, where she is currently an associate professor. Her research interests include computer vision, pattern recognition, and relative applications, especially action recognition/anticipation, video anomaly detection/anticipation, and few-shot learning

![](images/125e5b48af1948fbabd2461a8a2e74fac11775361ba8d150b8dd14590da358a6.jpg)

ing.

Zhenhe Liang received the BE degree in data science and big data technology from Shandong University, Jinan, China, in 2024. He is currently working toward the ME degree in computer technology with the National Engineering Laboratory for Integrated Aero-Space-Ground-Ocean Big Data Application Technology, School of Computer Science, Northwestern Polytechnical University, Xi’an, China. His current research interests include computer vision and deep learning, especially video anomaly detection/anticipation and weather forecast-

![](images/b6f9d2eeca3b25c0b4ac5adc85827fb279cbebb46376a20be825cb5b94a09f9e.jpg)

Hanwen Zhang received the BE degree in computer science and technology from Northwestern Polytechnical University, Xi’an, China, in 2023, and the MS degree in computer science and technology from the School of Computer Science, Northwestern Polytechnical University, Xi’an, China, in 2026. His current research interests include computer vision and deep learning, especially video anomaly detection/anticipation.

![](images/66c888c3867c65ac22b7b42188522555e67d966117ae70f03af6f0339f8028f8.jpg)

Yifan Zhao received the BE degree in computer science and technology from Northwestern Polytechnical University (NWPU), Xi’an, China, in 2025. He is currently working toward the ME degree in computer technology with the National Engineering Laboratory for Integrated Aero-Space-Ground-Ocean Big Data Application Technology, School of Computer Science, Northwestern Polytechnical University, Xi’an, China. His main research interests include video anomaly detection.

![](images/d8d014e16b7d7bed31f5dfe9aa3e4184041748bcada9272186b805a9ac57d70c.jpg)

Qinyi Lv received the BS and PhD degrees from Zhejiang University, Hangzhou, China, in 2013 and 2018, respectively. He is an associate professor with Northwestern Polytechnical University. His recent research interests include intelligent sensing, intelligent systems, radar detection, and inverse scattering problems.

![](images/cc9fb63ece5790b3d8becaca9f4e6a8515fa280704e54bea9ca58dacd4a8ec2a.jpg)  
Lingtong Min (Member, IEEE) received the BS degree from Northeastern University, Shenyang, China, in 2012, and the PhD degree from Zhejiang University, Hangzhou, China in 2019. He is a professor with Northwestern Polytechnical University. His main research interests are computer vision, pattern recognition, and remote sensing image understanding.

![](images/a9a7d1a05f64e64e851295f1afb516e2a94b26d5cdc38dc4e9317b870733b840.jpg)

Yanning Zhang (Fellow, IEEE) received the B.S. degree from Dalian University of Technology, Dalian, China, in 1988, the M.S. degree from the School of Electronic Engineering, Northwestern Polytechnical University, Xi’an, China, in 1993, and the Ph.D. degree from the School of Marine Engineering, Northwestern Polytechnical University, in 1996. She is currently a Professor with the School of Computer Science, Northwestern Polytechnical University. She is also a Cheung Kong Professor with the Ministry of Education, China. She has authored more than 200 articles. Her current research interests include remote sensing image analysis, computer vision, and pattern recognition. She is an Associate Editor of IEEE Transactions on Geoscience and Remote Sensing.