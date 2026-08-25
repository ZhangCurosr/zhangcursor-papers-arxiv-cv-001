# GuidedFlow: An Attention-Guided Framework for Anomaly Detection in Additive Manufacturing

Sosmita Paul, Krishna Roy New Mexico Institute of Mining and Technology Socorro, NM 87801

## Abstract

Additive Manufacturing (AM) plays a vital role in the ongoing industrial revolution. However, quality control remains crucial and challenging due to printing defects or potential cyber-physical intrusions. Image or video-based anomaly detection is a key effort towards addressing these challenges. Various approaches have been explored in this domain, including reconstruction-based, embedding-based, and flow-based methods. Though normalizing flow-based methods address some of the core challenges of unforeseen defects and generalization while maintaining detection performance, existing approaches struggle with tiny/stringing defects common in 3D printing. In a small-data setting, this poses a limitation in generalization. To address these limitations, we propose GuidedFlow, a novel attention-guided normalizing flow model for anomaly detection and localization. GuidedFlow employs a pre-trained ResNet model, fine-tuned on the domain dataset. An attention-guided spatial and temporal flow framework models the dynamics across multiple scales and frames. A Spatio-Temporal Attention Network (SAN) enables the flow model to prioritize relevant contextual cues from input frames. We evaluate GuidedFlow on our AM3D-AD dataset, consisting of benign and anomalous real 3D printed object images and videos. We also conduct a comparative study using the MVTec-AD industrial image anomaly detection dataset. Experimental results demonstrate that GuidedFlow outperforms most ofthe state-of-the-art models with enhanced detection accuracy and AUROC.

## 1. Introduction

In modern industrial environments, particularly in additive manufacturing (AM), ensuring the reliability and safety of printed components remains crucial and challenging due to printing defects or potential cyber-physical intrusions [10]. Even minor surface flaws or internal structural defects can cause part failure, increased production costs, or safety risks. Consequently, reliable anomaly detection and quality assurance mechanisms are critical to the safe deployment of AM technologies. In this domain, video/imagebased anomaly detection offers the advantage of capturing spatio-temporal patterns of defects, making it well-suited for monitoring dynamic processes during 3D printing.

Several video/image anomaly detection and localization approaches have been explored for industrial and manufacturing applications. Reconstruction-based methods [2, 17, 19, 28, 29, 33], such as autoencoders, generative adversarial networks (GANs) and diffusion models, learn to reconstruct benign samples and flag images with a high reconstruction error as anomalous. However, they often miss subtle or localized defects, especially when biased towards dominant regular patterns. Embedding-based methods [4, 23] aim to learn compact latent representations, modeling normal feature distributions and detecting deviations through feature similarity. While these approaches improve generalization, they may struggle in detecting context-dependent anomalies. Recently, flow-based methods [9, 14, 24, 32], particularly those based on normalizing flows, have attracted attention for their ability to estimate exact likelihoods and retain invertibility. These methods are well-suited for detecting unforeseen anomalies, as they model the full data distribution rather than just its lower-dimensional embedding. Despite recent advancements of existing flow-based approaches, they often fall short in detecting irregular anomalies, such as stringing or tiny layer inconsistency defects that are especially common in AM processes. These problems get worse in small-data settings, where learning feature distributions is difficult. Furthermore, most existing methods treat images or frames independently and do not incorporate temporal dynamics, which is often critical for accurate localization and early detection.

To overcome these limitations, we propose GuidedFlow, a novel attention-guided normalizing flow framework for anomaly detection in additive manufacturing. GuidedFlow integrates a ResNet-50 model with a spatio-temporal attention network to extract relevant contextual features. GuidedFlow introduces a spatio-temporal attention guided flow model, enabling it to handle context-dependent anomalies. The multi-level flow blocks model appearance and structure across resolutions. By learning where and when to focus in the visual stream, GuidedFlow enhances sensitivity to small and localized defects. We evaluated Guided-Flow on our AM3D-AD dataset containing both benign and anomalous 3D-printed object images and videos. Additionally, we benchmark our model against state-of-the-art methods on the publicly available MVTec-AD [3] dataset, due to the limited availability of additive manufacturing datasets. The model demonstrates robust performance in both anomaly detection and localization with improved generalization compared to most of the baseline methods.

We summarize our contributions as follows: (i) we propose and develop GuidedFlow, a novel conditioned-flow framework for image and video anomaly detection and localization integrating attention-guided spatial and temporal flow; (ii) we introduce a spatio-temporal attention network (SAN) using multi-head self-attention to enhance sensitivity to spatial and contextual anomalies; (iii) we propose a real-world additive manufacturing anomaly detection dataset, AM3D-AD, containing 3D-printed object images and videos; and (iv) we perform a comprehensive evaluation of GuidedFlow on both the AM3D-AD and MVTec-AD datasets.

## 2. Related Work

In this section, we review the key approaches for anomaly detection in AM, highlighting their advantages and shortcomings.

## 2.1. Unsupervised Deep Learning Methods

Unsupervised deep learning methods, particularly Autoencoders and Generative Adversarial Networks (GAN) based models, have gained popularity in addressing the challenge of limited anomalous data samples. In this domain, reconstruction-based strategies have shown effectiveness. For instance, MemAE, a memory augmented autoencoder, was used to detect anomalies in visual data by learning benign patterns and identifying deviations during reconstruction [8]. Similarly, encoder-decoder architectures trained exclusively on defect-free data have been effective in highlighting unexpected patterns through reconstruction loss [27], though they often struggle with borderline or ambiguous cases.

Several unsupervised efforts combined traditional image features, such as oriented gradient histograms, with physics-based rendering to create open-source, layer-wise monitoring systems [22]. Such systems may be less adaptive to unseen defect variations. In [15], video data from wire arc AM systems was processed using unsupervised deep models for real-time anomaly detection, demonstrating scalability to high-throughput environments but requiring substantial computational resources.

Although existing studies emphasized the growing importance of deep unsupervised techniques with high adaptability and precision for anomaly detection in industrial settings [6, 21], generalization remains a challenge, particularly under noisy or dynamically changing manufacturing conditions.

## 2.2. Transformer-Based and Contrastive Methods

Vision Transformers (ViT) introduced a new direction for anomaly detection by modeling long-range dependencies through self-attention mechanisms [7]. VT-ADL, a transformer-based framework, was developed to detect and locate anomalies in industrial images using the MVTec dataset [20]. Another work, EVAL, an explainable video anomaly localization framework, extended this by integrating explainable visual reasoning into a video anomaly localization pipeline, improving transparency in AM monitoring [25]. To enhance robustness across varying AM processes, GeneralAD applied attention to distorted features, offering cross-domain adaptability [26].

Contrastive learning approaches have also gained attention for anomaly detection. CLEP adopted contrastive pretraining for vulnerability detection, demonstrating the broader utility of discriminative representations [5]. In the context of AM, One-for-All utilized masked cross-class contrastive learning for unsupervised anomaly detection, highlighting scalability across diverse defect types [30]. DualAD further advanced this by employing a dual-branch design to separately model normal and anomalous behavior, yielding better detection granularity [12]. Recently, a multi-scale framework was proposed that captures both coarse and fine-grained spatio-temporal features, boosting performance in complex AM scenarios [35]. Despite their elevated anomaly detection performance, these methods demand substantial training data and computation, which limits real-time applicability in AM environments.

## 2.3. Normalizing Flow and Diffusion Methods

Flow-based models are increasingly being applied in anomaly detection due to their ability to model exact data likelihoods through invertible transformations. SPADE [23], introduced spatially-weighted normalizing flows over pretrained feature embeddings, allowing effective localization of anomalies in industrial images. Another work, PatchSVDD-Flow [31], embedded flow-based priors into support vector data descriptions, resulting in more compact latent distributions for benign data. Fast-Flow [32] used 2D normalization flows for efficient unsupervised detection and localization of anomalies in visual data. In the temporal domain, Lu et al. [16] proposed a future frame prediction model that employs normalizing flows over temporal embeddings, demonstrating its applicability in video-based surveillance anomaly detection. Recently, flow-based models have been extended toward spatio-temporal domains using hierarchical and multiscale architectures. PyramidFlow Matching [13] implements a coarse-to-fine flow matching strategy across Laplacian pyramid levels to model fine-grained spatial and motion cues. This is a diffusion-based approach utilizing a unified Diffusion Transformer (DiT) for video generation modeling. Diffusion-based methods[13, 17, 29] offer strong generative power for reconstructing normal data in anomaly detection. Although diffusion-based models are effective in capturing complex structures, they are computationally intensive and may blur fine details critical to accurate stringing defect localization.

Flow-based models may still encounter limitations in detecting tiny printing defects, such as stringing in 3D printing, causing a generalization challenge. Likelihood-based scoring can be misled by high-frequency background patterns, and pretrained features may lack semantic alignment with task-specific distributions. To address these challenges, we propose GuidedFlow, an attention-guided, hierarchical flow framework that integrates spatio-temporal guidance into the flow levels.

## 3. Methodology

The proposed GuidedFlow system architecture is shown in Figure 1. We introduce the GuidedFlow model, a deep framework for high-resolution video anomaly detection and localization using an attention guided spatio-temporal flow mechanism. It models the distribution of multi-scale feature pyramids extracted from frames rather than raw pixels. The model is trained on benign data using a likelihood objective. GuidedFlow processes $2 2 4 \times 2 2 4 .$ , 3-channel video clips and identifies 3D printing defects, followed by defect localization. The approach uses a pretrained feature extractor, spatio-temporal attention mechanisms, attention-guided multi-scale flow modeling for precise localization and detection of spatio-temporal anomalies.

## 3.1. ResNet Feature Extraction

A pretrained ResNet is used to extract video features. Given a video frame $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ , each frame is first passed through a ResNet-50 backbone to extract hierarchical features $X _ { t } ~ = ~ E ( x _ { t } )$ from the first 4 ResNet layers named $f _ { 1 } , f _ { 2 } , f _ { 3 } , f _ { 4 }$ with channel depths of 256, 512, 1024, and 2048 respectively. During training, layers $f _ { 1 } , \ f _ { 2 }$ were frozen to preserve general low-level feature representations and layers $f _ { 3 } , \ f _ { 4 }$ are trained. The extracted features $X _ { t }$ along with the original frame $x _ { t }$ and feature map $F _ { m }$ are combined to generate a scene embedding $S _ { e } .$ . The ResNet feature map $F _ { m }$ captures spatial information across varying resolutions. Extracted features $X _ { t }$ and the scene embeddings $S _ { e }$ are fed to the spatio-temporal attention network to the next stage.

## 3.2. SAN Spatio-temporal Attention Encoding

To capture spatial and temporal dependencies across video frames and prioritize potential mapping of the anomaly, a spatio-temporal attention module is applied on the ResNet features. We used a multi-head self-attention mechanism, where the input $X _ { t }$ is transformed into query $Q ,$ key $K ,$ and value $V$ vectors via linear projections, with the output computed as: $\alpha _ { N } = \mathrm { A t t e n t i o n } ( Q , K , V )$ . The attention mechanism is defined as:

$$
{ \mathrm { A t t e n t i o n } } ( Q , K , V ) = { \mathrm { s o f t m a x } } \left( { \frac { Q K ^ { T } } { \sqrt { d _ { k } } } } \right) V ,\tag{1}
$$

Where $( d _ { k } )$ denotes the dimension of the key vector. The network implements multiple attention heads (H heads), each processing a subset of $X _ { t }$ independently. The outputs of these heads $\alpha _ { N }$ and scene embeddings $S _ { e }$ from the previous stage are concatenated and linearly transformed to produce a unified attention output vector $a _ { 1 }$ to $a _ { N }$ for N frames:

$$
a _ { N } = \operatorname { c a t } ( S _ { e } , \operatorname { h e a d } _ { 1 } , \dots , \operatorname { h e a d } _ { H } ) W _ { o }\tag{2}
$$

where $W _ { o }$ is an output projection matrix, and H is the number of heads. $a _ { N }$ reflects the spatial and temporal dependencies and enhances the ability to focus on relevant regions across frame sequences in the GuidedFlow mechanism. Thus, the SAN module generates the spatio-temporal context embedding for each frame, $a = C _ { \phi } ( X _ { t } , S _ { e } )$ , where $C _ { \phi } ( \cdot )$ is learnable and produces conditioning attention a.

## 3.3. Guided Spatio-Temporal Flow Processing

The ResNet features $X _ { t }$ are passed to a Guided Spatio-Temporal Flow (GSTF) framework. The GSTF module integrates spatial and temporal flow components conditioned to SAN attention to process video features $X _ { t }$ through a hierarchical, invertible transformation. GSTF leverages a Laplacian pyramid decomposition and attention coupling to model spatial dependencies within frames and temporal dependencies across frames. First, we build a multi-scale Laplacian pyramid $\mathcal { P } ( X _ { t } ) = \{ l p _ { 0 } , l p _ { 1 } , . . . , l p _ { L } \}$ from the backbone features and apply a conditional invertible transformation to obtain latent variables:

$$
\mathbf { z } = F _ { \theta } ( \mathbf { l } ; a ) .\tag{3}
$$

The mapping $F _ { \theta } ( \cdot ; a )$ is invertible with respect to l for any fixed $^ { a , }$ enabling exact likelihood evaluation. The spatial and temporal flow components are explained in the next sections.

![](images/1fc5f0b4587b17b4c88a7eafbfac0a5de3b098d8d2399e6661fac95a600564a6.jpg)  
Figure 1. GuidedFlow Overview. The model is composed of four parts: ResNet Feature Extractor, Spatio-temporal Attention Network, Guided Spatiot-Temporal Flow, and Anomaly Localization Map.

Spatial Flow: The spatial flow is implemented to transform feature representations across multiple scales and spatial dynamics. The flow mechanism starts with the construction of a Laplacian pyramid from the features $X _ { t }$ . The Laplacian Pyramid decomposes the input features $X _ { t }$ into L levels, where each level $l _ { i }$ represents a scale-specific representation,

$$
l _ { i } ^ { \prime } = f _ { \mathrm { s p a t i a l } } ( l _ { i } , a ) ,\tag{4}
$$

where $f _ { \mathrm { s p a t i a l } }$ are coupling across frames. The spatial flow applies invertible transformations to each of the pyramid levels. The spatial flow thus transforms the pyramid levels $l _ { i }$ to $l _ { i } ^ { \prime }$ across scales, guided by context coupling a, with invertibility ensured by the coupling and convolution operations.

Temporal Flow: The temporal flow extends the spatial transformations across the temporal dimension $T ,$ modeling dependencies between frames $i = 0 \mathrm { t o } i = T$ . The input to the temporal flow is the spatially transformed pyramid levels $l _ { i } ^ { \prime } \ \in \ \mathbb { R } ^ { B \times C \times T \times H _ { l } \times \dot { W _ { l } } }$ for each level l. The temporal flow is modeled as a sequence of invertible coupling operations across $l _ { i } ^ { \prime } ,$ spatio-temporal attention vectors a:

$$
l _ { i } ^ { \prime \prime } = f _ { \mathrm { t e m p o r a l } } ( l _ { i } ^ { \prime } , a ) ,\tag{5}
$$

Where $f _ { \mathrm { t e m p o r a l } }$ represents the temporal flow function, implemented through the invertible coupling.

Each GSTF layer uses affine coupling. Splitting an input $u = [ u _ { 1 } , u _ { 2 } ]$ , we define

$$
\begin{array} { r } { v _ { 1 } = u _ { 1 } , \qquad v _ { 2 } = u _ { 2 } \odot \exp ( s ( u _ { 1 } ; a ) ) + t ( u _ { 1 } ; a ) , } \end{array}\tag{6}
$$

where $( s ( \cdot ; a ) , t ( \cdot ; a ) )$ are produced by a conditioning network. We implement conditioning via a hypernetwork modulation:

$$
[ s , t ] = g ( u _ { 1 } ) \oplus h ( a ) ,\tag{7}
$$

where $g ( \cdot )$ is the standard coupling subnetwork and $h ( \cdot )$ maps SAN context a to additive (or scale/shift) modulation of coupling parameters. The log-determinant is computed as

$$
\log \left| \operatorname * { d e t } { \frac { \partial v } { \partial u } } \right| = \sum s ( u _ { 1 } ; a ) .\tag{8}
$$

The spatial flow transforms pyramid levels using scalespecific invertible coupling and convolution, while the temporal flow extends this across frames using 3D coupling guided by spatio-temporal attention $a .$ The invertibility of the GSTF flow mechanism makes GuidedFlow computationally efficient for fast likelihood based anomaly scoring and localization.

## 3.4. Anomaly Scoring and Localization

For anomaly detection, we calculate anomaly score $S ( x _ { t } )$ from $\mathbf { l } = \mathcal { P } ( E ( x _ { t } ) )$ and $a = C _ { \phi } ( E ( x _ { t } ) )$ ) by evaluating the negative log-likelihood:

$$
S ( x ) = - \log p ( 1 \mid a ) .\tag{9}
$$

We compute per-location contributions from the latent energy and log-determinant terms, aggregate across pyramid levels, and upsample to the input resolution:

$$
M ( x _ { t } ) = \sum _ { j = 0 } ^ { L - 1 } \uparrow _ { H , W } \bigg ( \sum _ { c } \frac { 1 } { 2 } ( z _ { c u v } ^ { j } ) ^ { 2 } - d _ { u v } ^ { j } \bigg ) ,\tag{10}
$$

where $d _ { u v } ^ { j }$ denotes the spatially-resolved log-determinant contribution induced by the coupling layers at level j.

## 4. Experiment

## 4.1. Experimental Settings

## 4.1.1. Dataset Details

To evaluate the effectiveness of the GuidedFlow framework, we conduct experiments on a domain dataset AM3D-AD (ours) and a subset of the baseline MVTec-AD [3] dataset.

AM3D-AD Dataset contains both images and videos from five categories: Bolt, Gear, Nut, Block, and Cube. All the objects are printed with NylonX Carbon Fiber. The dataset consists of 537 images (254 benign and 283 anomalous) and 129 videos of the printed object, each 25 seconds long on average, supporting temporal analysis. To create anomalous samples with 3D printing defects, the Gcode was intentionally sabotaged by inserting redundant or noisy motion commands, including high-frequency or random G0 (non-extruding move) and G1 (extruding move) instructions. These modifications cause surface-level distortions by introducing non-uniform paths or jagged trajectories. The attacks were performed in a controlled environment with Creality K1 Max and Ender-3 3D printers[1]. Further details regarding the anomalous data collection procedure can be found in [11]. All the images and videos are manually labeled. An overview of the AM3D dataset is shown in Figure 2a.

MVTec-AD Dataset comprises 5,354 high-resolution images across 15 distinct categories, with a training set of 3,629 anomaly-free images and a test set of 1,725 images containing normal and anomalous samples. An illustration is shown in Figure 2b.

Evaluation Metrics We evaluate our method using the Area Under the Receiver Operating Characteristic Curve (AUROC) at the pixel P-AUROC and image level I-AUROC for image anomaly detection and localization. Video-level AUROC is reported as V-AUROC. We also report the Area Under the Per-Region Overlap (AUPRO) curve and the Average Precision (AP). AUROC scores are reported in percentage (%).

Implementation Details GuidedFlow is implemented using Pytorch 2.5 in Python 3.10. Our implementation uses ResNet-50. The spatio-temporal attention network was implemented with a multi-head self-attention encoder with H = 8 attention heads, chosen according to the hyperparameter performance analysis discussed later. The guidedflow mechanism is implemented with a spatial flow and a temporal flow containing 6 flow levels 4 temporal flow stacks.

Anomalous  
Benign  
![](images/d56230d129e2386868a43549d33f7d999af7e093941816ffc08cb61f25577eb8.jpg)

![](images/4279317d49949cb37964cb8185ef9d1df61ef38a28fca59c7ee9719384a10569.jpg)  
(a) AM3D-AD Dataset  
(b) MVTec-AD Dataset  
Figure 2. AM3D-AD and MVTec-AD dataset samples. The first column shows benign samples, while the second column shows anomalous samples in each dataset.

## 4.1.2. Training and Evaluation Setup

Model training and comprehensive evaluation are performed in multiple parameter settings, including an ablation study.

Training Details: In the data pre-processing phase, all images from AM3D-AD and MVTec-AD datasets are resized to 224x224. In training, we assume a standard normal prior to latent variables z:

$$
p ( \mathbf { z } ) = \prod _ { j = 0 } ^ { L - 1 } \mathcal { N } ( z ^ { j } ; 0 , I ) .\tag{11}
$$

The conditional log-likelihood of the pyramid representation is

$$
\log p ( \boldsymbol { 1 } \mid a ) = \sum _ { j = 0 } ^ { L - 1 } \left( \log p ( z ^ { j } ) + \log \left| \operatorname* { d e t } \frac { \partial F _ { \theta } ^ { j } } { \partial l ^ { j } } \right| \right) .\tag{12}
$$

We train the flow model using benign samples only by minimizing the negative log-likelihood:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { N L L } } = - \mathbb { E } _ { { x } _ { t } \sim \mathcal { D } _ { \mathrm { b e n i g n } } } \Big [ \log p \big ( \mathcal { P } ( E ( x _ { t } ) ) \mid C _ { \phi } ( E ( x _ { t } ) ) \big ) \Big ] . } \end{array}\tag{13}
$$

During training, the backbone $E ( \cdot )$ is trained, while SAN parameters ϕ and flow parameters θ are optimized jointly.

(b) MVTec-AD Anomaly Visualization

The complete framework is trained on benign samples of both datasets in two phases: Phase I) Backbone training: Top layers of the ResNet or 1 × 1 Convolution feature extractor trained with an Adam optimizer, maintaining a learning rate of $1 \times 1 0 ^ { - 4 }$ , batch size of 8, and training for 20 epochs. Phase-II) Conditioned Spatio-Temporal Flow Learning: The SAN and GSTF flow model is trained with an invertible normalizing flow model, maintaining a learning rate of $1 \times 1 0 ^ { - 4 }$ , a batch size of 8, and training for 100 epochs.

Evaluation Details For both datasets, we split the benign set into 80% for training and 20% for testing. All anomalous samples are used only at test time. All model training and evaluation are performed on a NVIDIA A100 GPU server with 40 GB of memory.

## 4.2. Anomaly Detection Performance

We calculate AUROC at the image and pixel levels presented with I-AUROC and P-AUROC, respectively, to evaluate the performance of GuidedFlow without temporal-flow (T-Flow) and temporal-attention (T-Attn) anomaly detection. It achieves I-AUROC and P-AUROC scores, ranging from 95.5 to 99.3 and 94.1 to 98.9, respectively, with AUPRO between 0.88 and 0.96 and I-AP ranging from 0.93 to 0.98 on the two datasets. GuidedFlow I-AUROC outperforms for the majority classes with Bolt in the AM3D dataset and MetalNut in the MVTec dataset, achieving the highest scores of 98.7 and 99.3, respectively. GuidedFlow effectively generalizes across the 10 classes, with consistent AUROC scores. The results are presented in Table 1.

We performed a comparative study of GuidedFlow against three state-of-the-art (SOTA) anomaly detection models, including DiffusionAD [34], INP-Former [18] and PyramidFlow [14] as shown in Table 1. All baseline models are trained with the respective AM3D-AD and MVTec-AD datasets for fair comparison. GuidedFlow attains an average I-AUROC of 97.4, an average P-AUROC of 96.8, AUPRO 0.93, and I-AP 0.95, outperforming most of the SOTA models DiffusionAD(96.8, 95.8, 0.86, 0.87), INP-Former (95.6, 95.2, 0.90, 0.92), and PyramidFlow (96.1, 95.3, 0.91, 0.94). The inclusion of guided attention and spatial flow enabled the model to distinguish manufacturing defects. GuidedFlow showed consistent performance across multiple categories of defects, especially in visually similar cases where baseline models struggled. However, GuidedFlow shows a lower P-AUROC for some specific classes, such as Cube (94.2) and Block (94.1) in the AM3D-AD dataset, and Toothbrush (95.8) and Tile (96.4) in the MVTec-AD dataset, potentially posing questions about the generalizability of the model. This indicates that GuidedFlow may struggle with anomalies that exhibit minimal pixel-wise deviation or complex textures like in the case of Cube (AM3D) and Toothbrush (MVTec), causing model to emphasize global texture over subtle structural defects. A visualization of the heatmaps of the AM3D-AD and MVTec-AD sample is presented in Figure 3.

![](images/07df73668deb39f775d2a4466e31a73cb2d96a2ae6c8752bff5c3bd3b0df6df9.jpg)  
Figure 3. Visualization of GuidedFlow anomaly localization results on AM3D-AD and MVTec-AD datasets. The top row is input images, the middle row ground truths, and the bottom row GuidedFlow localization as heatmaps, where higher intensity values indicate a higher anomaly likelihood

To evaluate video anomaly detection and localization performance, we experiment with GuidedFlow, including temporal attention (T-Attention) and temporal flow (T-Flow) on the AM3D-AD video dataset. The video-level detection results are reported in Table 2. We observe that GuidedFlow significantly improves V-AUROC compared to image-level I-AUROC for several categories. In particular, the model achieves 98.9, 97.2, and 98.6 V-AUROC for Gear, Cube, and Block, respectively. These improvements can be attributed to the temporal attention mechanism of SAN that leverages contextual information from four sequential frames (see the hyperparameter analysis in Section 4.4). GuidedFlow also achieves improved AUPRO scores (0.93 for Gear and 0.92 for Cube) compared to image-based AUPRO, while maintaining stable AP across all categories. These results demonstrate that GuidedFlow effectively captures temporal consistency and accurately detects and localizes anomalies across diverse object geometries and defect patterns, indicating strong generalization across AM3D categories.

Table 1. Comparison of image anomaly detection methods on AM3D and MVTec datasets using I-AUROC, P-AUROC, AUPRO, and I-AP.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="4">AM3D Dataset</td><td colspan="5">MVTec Dataset</td><td rowspan="2">Avg</td></tr><tr><td>Bolt</td><td>Nut</td><td>Gear</td><td>Cube Block</td><td>MetalNut</td><td>Pill</td><td>Toothbrush</td><td>Tile</td><td>Wood</td></tr><tr><td rowspan="4">DiffusionAD</td><td>I-AUROC</td><td>97.8</td><td>97.3</td><td>96.9</td><td>94.9</td><td>95.9</td><td>99.1</td><td>98.3</td><td>95.7</td><td>96.2 96.0</td><td>96.8</td></tr><tr><td>P-AUROC</td><td>98.1</td><td>96.9</td><td>95.5</td><td>93.5 92.6</td><td>97.8</td><td>97.3</td><td>94.1</td><td>96.1</td><td>95.9</td><td>95.8</td></tr><tr><td>AUPRO</td><td>0.92</td><td>0.86</td><td>0.84</td><td>0.79</td><td>0.85</td><td>0.87 0.86</td><td>0.82</td><td>0.91</td><td>0.88</td><td>0.86</td></tr><tr><td>I-AP</td><td>0.92</td><td>0.87</td><td>0.91</td><td>0.84 0.87</td><td>0.90</td><td>0.85</td><td>0.82</td><td>0.84</td><td>0.86</td><td>0.87</td></tr><tr><td rowspan="4">INP-Former</td><td>I-AUROC</td><td>96.5</td><td>96.4</td><td>97.2</td><td>94.0</td><td>94.6</td><td>97.5 96.3</td><td>94.1</td><td>94.0</td><td>95.4</td><td>95.6</td></tr><tr><td>P-AUROC</td><td>97.2</td><td>96.6</td><td>96.5</td><td>93.8 93.5</td><td>96.8</td><td>96.1</td><td>93.8</td><td>93.5</td><td>93.8</td><td>95.2</td></tr><tr><td>AUPRO</td><td>0.93</td><td>0.92</td><td>0.86</td><td>0.83</td><td>0.91</td><td>0.92 0.94</td><td>0.85</td><td></td><td>0.92 0.91</td><td>0.90</td></tr><tr><td>I-AP</td><td>0.95</td><td>0.94</td><td>0.91</td><td>0.89</td><td>0.92</td><td>0.91 0.94</td><td>0.89</td><td>0.88</td><td>0.92</td><td>0.92</td></tr><tr><td rowspan="4">PyramidFlow</td><td>I-AUROC</td><td>97.5</td><td>96.4</td><td>95.7</td><td>94.9 95.8</td><td>98.1</td><td>97.4</td><td>93.9</td><td>95.7</td><td>95.3</td><td>96.1</td></tr><tr><td>P-AUROC</td><td>96.2</td><td>95.6</td><td>94.7</td><td>93.8 94.0</td><td>97.0</td><td>95.8</td><td>94.9</td><td>95.2</td><td>95.3</td><td>95.3</td></tr><tr><td>AUPRO</td><td>0.94</td><td>0.94</td><td>0.89</td><td>0.86</td><td>0.92</td><td>0.91 0.92</td><td>0.87</td><td>0.91</td><td>0.95</td><td>0.91</td></tr><tr><td>I-AP</td><td>0.97</td><td>0.95</td><td>0.94</td><td>0.92</td><td>0.93</td><td>0.95 0.96</td><td>0.89</td><td>0.94</td><td>0.93</td><td>0.94</td></tr><tr><td rowspan="4">GuidedFlow</td><td>I-AUROC</td><td>98.7</td><td>98.2</td><td>97.9</td><td>95.5</td><td>96.0</td><td>99.3 98.9</td><td>95.5</td><td>97.1</td><td>96.7</td><td>97.4</td></tr><tr><td>P-AUROC</td><td>98.9</td><td>97.8</td><td>96.7</td><td>94.2 94.1</td><td>98.8</td><td>98.1</td><td>95.8</td><td>96.4</td><td>96.8</td><td>96.8</td></tr><tr><td>AUPRO</td><td>0.96</td><td>0.93</td><td>0.91</td><td>0.88 0.94</td><td>0.93</td><td>0.94</td><td>0.89</td><td>0.96</td><td>0.96</td><td>0.93</td></tr><tr><td>I-AP</td><td>0.97</td><td>0.97</td><td>0.96</td><td>0.94 0.95</td><td>0.98</td><td>0.98</td><td>0.93</td><td>0.93</td><td>0.95</td><td>0.95</td></tr></table>

Table 2. GuidedFlow video anomaly detection performance on AM3D dataset via video-level AUROC (V-AUROC), P-AUROC, AUPRO, and AP.
<table><tr><td rowspan="2">Metric</td><td colspan="5">AM3D Dataset</td></tr><tr><td>Bolt</td><td>Nut</td><td>Gear</td><td>Cube</td><td>Block</td></tr><tr><td>V-AUROC</td><td>98.9</td><td>98.8</td><td>98.9</td><td>97.2</td><td>98.6</td></tr><tr><td>P-AUROC</td><td>99.1</td><td>98.7</td><td>98.2</td><td>96.4</td><td>96.1</td></tr><tr><td>AUPRO</td><td>0.94</td><td>0.93</td><td>0.93</td><td>0.92</td><td>0.95</td></tr><tr><td>AP</td><td>0.97</td><td>0.96</td><td>0.97</td><td>0.94</td><td>0.94</td></tr></table>

Table 3. Ablation of GuidedFlow architectural components for AM3D-AD video dataset. ”S-Flow”, ”T-Flow” refer to the spatial and temporal flow respectively and ”ST-Flow” refer to the spatio-temporal flow components. Similarly ”S-Attention or S-Attn”, ”T-Attention or T-Attn” refer to the spatial and temporal attention respectively, and ”ST-Attention or ST-Attn” refer to the spatio-temporal attention.
<table><tr><td>Model Variants</td><td>V-AUROC</td><td>P-AUROC</td></tr><tr><td>1 × 1Conv + ST-Flow</td><td>91.5</td><td>89.5</td></tr><tr><td>TunedResNet + ST-Flow</td><td>92.1</td><td>91.0</td></tr><tr><td>ResNet + S-Attn + ST-Flow</td><td>97.4</td><td>95.2</td></tr><tr><td>ResNet + T-Attn + ST-Flow</td><td>96.1</td><td>94.8</td></tr><tr><td>ResNet + ST-Attn + S-Flow</td><td>95.3</td><td>96.1</td></tr><tr><td>ResNet + ST-Attn + T-Flow</td><td>94.6</td><td>94.8</td></tr><tr><td>GuidedFlow (ST-Attn + ST-Flow)</td><td>98.7</td><td>98.9</td></tr></table>

## 4.2.1. Qualitative Evaluation

Fig. 3 visualizes qualitative localization results, showing the input frame, ground-truth, and the predicted anomaly heatmap produced by GuidedFlow. In the heatmap, higher intensity indicates a higher likelihood of anomalies.

## 4.3. Ablation Study

To evaluate the significance of different architectural components of GuidedFlow, we performed an ablation study on the AM3D-AD video dataset presented in Table 3. We started with a simple 1×1 convolution connected to flow blocks without attention module SAN and observed a modest performance with an V-AUROC of 91.5 and a P-AUROC of 89.5. Replacing the shallow encoder with AM3D data-tuned ResNet-50 improves both video and pixel-level performance, suggesting that stronger semantic features help the flow model better capture benign patterns.

Effectiveness of Spatial and Temporal Attention We introduce spatial attention on top of the ResNet feature extraction (S-Attention + ST-Flow) that demonstrates a significant boost in both V-AUROC and P-AUROC to 97.4 and 95.2 respectively. Spatial guidance clearly adds to the learning of contextual representations for anomaly localization. Temporal attention (T-Attention + ST-Flow) also improves performance over the models without attention coupling (1×1Conv+ST-Flow or TunedResNet+ST-Flow). Moreover, combining both spatial and temporal attention in the GuidedFlow architecture leads to the highest performance, with a V-AUROC of 98.7 and P-AUROC of 98.9. These results indicate that spatial and temporal cues are complementary to each other. Spatial attention enhances feature focus across spatial hierarchies within a frame, while temporal attention reinforces consistency in sequential frame context.

Effectiveness of Spatial and Temporal Flow To assess the significance of spatial and temporal flow, we designed two variants of GuidedFlow, ST-Attention + S-Flow and ST-Attention + T-Flow, varying the type of normalizing flow being applied and trained with the AM3D-AD video dataset. The ablation results for these two variants in Table 3 shows complementary strengths of spatial and temporal flow mechanisms. Both ST-Attention + S-Flow and

ST-Attention + T-Flow achieve higher V-AUROC of 95.3 and 94.6, indicating the effectiveness of spatial and temporal flow in capturing spatial and sequence-level deviations during video anomaly detection. Multi-resolution spatial and multi-frame temporal encoding improve the model’s sensitivity to localized fine/stringing defects. While temporal flow improves overall detection performance, spatial flow contributes more directly to accurate and fine-grained anomaly localization. In GuidedFlow the spatial flow obtains strong detection performance, and the temporal flow further improves it by incorporating inter-frame relational context.

## 4.4. Parameter Performance

We evaluate GuidedFlow’s performance using different hyperparameter settings with the number of flow levels L, number of temporal frames T, number of attention heads H, and number of ResNet Tuned Layers. A detailed hyperparameter vs. AUROC analysis is shown in Figure 4. We calculate AUROC for different combinations of L, T, H, and Tuned Layers.

We observe that the number of flow levels $L = 6 ,$ , the temporal frames $T = 3 ,$ and the attention heads $H = 8$ produce the highest AUROC with four tuned ResNet layers.

![](images/1b33b263227e24d07e5bb6dc85acd181880ccc42372da3f5ecc91ed76827fe59.jpg)  
Figure 4. GuidedFlow hyperparameter performance. AUROC values for the hyperparameter settings are shown as percentages (%).

## 5. Limitations and Future Work

While GuidedFlow demonstrates strong overall performance, we observe reduced effectiveness for certain classes, such as Block and Cube in AM3D-AD, and Toothbrush and Tile in MVTec-AD, where anomalies show minimal pixelwise deviation or highly repetitive textures. In addition, evaluation is currently limited to a small number of publicly available additive manufacturing datasets. Future work will expand experiments to more domain datasets, including diverse materials and printing conditions, and explore improved domain-adaptive feature learning to enhance robustness and generalizability.

## 6. Conclusion

In this work, we introduced GuidedFlow, an attentionguided spatio-temporal normalizing flow model designed for detecting and localizing anomalies in both images and videos in additive manufacturing. The model uses hierarchical features from a pretrained ResNet-50 with spatiotemporal attention conditioned flow processing, enabling analysis across multiple resolutions and frames. We evaluated the model on our AM3D-AD dataset, which includes real images and videos of 3D-printed objects, and the MVTec-AD baseline anomaly detection dataset. Experimental results demonstrate consistent performance in both image-based and video-based anomaly detection tasks.

## Acknowledgements

This material is based upon work supported by the National Science Foundation under Award No. 2417062. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## References

[1] Creality 3d printers. https://www.creality.com/, 2025. Accessed: 2025-01-08. 5

[2] Paul Bergmann, Sindy Lowe, Michael Fauser, David Sattleg-¨ ger, and Carsten Steger. Improving unsupervised defect seg mentation by applying structural similarity to autoencoders. arXiv preprint arXiv:1807.02011, 2018. 1

[3] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9592–9600, 2019. 2, 5

[4] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Uninformed students: Student-teacher anomaly detection with discriminative latent embeddings. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4183–4192, 2020. 1

[5] Jie Chen, Liangmin Wang, Huijuan Zhu, and Victor S Sheng. Clep: A novel contrastive learning method for evolution ary reentrancy vulnerability detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 67–74, 2025. 2

[6] Sourabh Deshpande, Vysakh Venugopal, Manish Kumar, and Sam Anand. Deep learning-based image segmentation for defect detection in additive manufacturing: an overview. The International Journal ofAdvanced Manufacturing Tech nology, 134(5):2081–2105, 2024. 2

[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Trans-

formers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2

[8] Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1705–1714, 2019. 2

[9] Denis Gudovskiy, Shun Ishizaka, and Kazuki Kozuka. Cflow-ad: Real-time unsupervised anomaly detection with localization via conditional normalizing flows. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 98–107, 2022. 1

[10] Abid Haleem and Mohd Javaid. Additive manufacturing applications in industry 4.0: a review. Journal of Industrial Integration and Management, 4(04):1930001, 2019. 1

[11] Md Mahbub Hasan, Marcus Sternhagen, and Krishna Chandra Roy. Engineering attack vectors and detecting anomalies in additive manufacturing. arXiv preprint arXiv:2601.00384, 2026. 5

[12] Shiwen He, Yuehan Chen, Liangpeng Wang, Wei Huang, Rong Xu, and Yurong Qian. Dualad: exploring coupled dual-branch networks for multi-class unsupervised anomaly detection. Electronics, 14(3):594, 2025. 2

[13] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024. 3

[14] Jiarui Lei, Xiaobo Hu, Yue Wang, and Dong Liu. Pyramidflow: High-resolution defect contrastive localization using pyramid normalizing flow. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 14143–14152, 2023. 1, 6

[15] Runsheng Li, Hui Ma, Rui Wang, Hao Song, Xiangman Zhou, Lu Wang, Haiou Zhang, Kui Zeng, and Chunyang Xia. Application of unsupervised learning methods based on video data for real-time anomaly detection in wire arc additive manufacturing. Journal of Manufacturing Processes, 143:37–55, 2025. 2

[16] Wen Liu, Weixin Luo, Dongze Lian, and Shenghua Gao. Future frame prediction for anomaly detection–a new baseline. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6536–6545, 2018. 2

[17] Victor Livernoche, Vineet Jain, Yashar Hezaveh, and Siamak Ravanbakhsh. On diffusion modeling for anomaly detection. arXiv preprint arXiv:2305.18593, 2023. 1, 3

[18] Wei Luo, Yunkang Cao, Haiming Yao, Xiaotian Zhang, Jianan Lou, Yuqi Cheng, Weiming Shen, and Wenyong Yu. Exploring intrinsic normal prototypes within a single image for universal anomaly detection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9974–9983, 2025. 6

[19] Takashi Matsubara, Kazuki Sato, Kenta Hama, Ryosuke Tachibana, and Kuniaki Uehara. Deep generative model using unregularized score for anomaly detection with hetero-

geneous complexity. IEEE Transactions on Cybernetics, 52 (6):5161–5173, 2020. 1

[20] Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, and Gian Luca Foresti. Vt-adl: A vision transformer network for image anomaly detection and localiza tion. In 2021 IEEE 30th International Symposium on Industrial Electronics (ISIE), pages 01–06. IEEE, 2021. 2

[21] Guansong Pang, Chunhua Shen, Longbing Cao, and Anton Van Den Hengel. Deep learning for anomaly detection: A review. ACM computing surveys (CSUR), 54(2):1–38, 2021. 2

[22] Aliaksei Petsiuk and Joshua M Pearce. Towards smart monitored am: Open source in-situ layer-wise 3d printing im age anomaly detection using histograms of oriented gradients and a physics-based rendering engine. Additive Manu facturing, 52:102690, 2022. 2

[23] Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Scholkopf, Thomas Brox, and Peter Gehler. Towards to-¨ tal recall in industrial anomaly detection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 14318–14328, 2022. 1, 2

[24] Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, and Bastian Wandt. Fully convolutional cross-scale-flows for imagebased defect detection. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 1088–1097, 2022. 1

[25] Ashish Singh, Michael J Jones, and Erik G Learned-Miller. Eval: Explainable video anomaly localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18717–18726, 2023. 2

[26] Luc PJ Strater, Mohammadreza Salehi, Efstratios Gavves,¨ Cees GM Snoek, and Yuki M Asano. Generalad: Anomaly detection across domains by attending to distorted features. In European Conference on Computer Vision, pages 448– 465. Springer, 2024. 2

[27] Yingshui Tan, Baihong Jin, Alexander Nettekoven, Yuxin Chen, Yisong Yue, Ufuk Topcu, and Alberto Sangiovanni-Vincentelli. An encoder-decoder based approach for anomaly detection with application in additive manufacturing. In 2019 18th IEEE international conference on ma chine learning and applications (ICMLA), pages 1008–1015. IEEE, 2019. 2

[28] Shashanka Venkataramanan, Kuan-Chuan Peng, Rajat Vikram Singh, and Abhijit Mahalanobis. Attention guided anomaly localization in images. In European Conference on Computer Vision, pages 485–503. Springer, 2020. 1

[29] Haohao Xu, Shuchang Xu, and Wenzhen Yang. Unsupervised industrial anomaly detection with diffusion models. Journal ofVisual Communication and Image Representation, 97:103983, 2023. 1, 3

[30] Xincheng Yao, Chongyang Zhang, Ruoqi Li, Jun Sun, and Zhenyu Liu. One-for-all: Proposal masked cross-class anomaly detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 4792–4800, 2023. 2

[31] Jihun Yi and Sungroh Yoon. Patch svdd: Patch-level svdd for anomaly detection and segmentation. In Proceedings of the Asian conference on computer vision, 2020. 2

[32] Jiawei Yu, Ye Zheng, Xiang Wang, Wei Li, Yushuang Wu, Rui Zhao, and Liwei Wu. Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows. arXiv preprint arXiv:2111.07677, 2021. 1, 2

[33] Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj. Draem-ˇ a discriminatively trained reconstruction embedding for surface anomaly detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8330– 8339, 2021. 1

[34] Hui Zhang, Zheng Wang, Dan Zeng, Zuxuan Wu, and Yu-Gang Jiang. Diffusionad: Norm-guided one-step denoising diffusion for anomaly detection. IEEE transactions on pattern analysis and machine intelligence, 2025. 6

[35] Menghao Zhang, Jingyu Wang, Qi Qi, Haifeng Sun, Zirui Zhuang, Pengfei Ren, Ruilong Ma, and Jianxin Liao. Multiscale video anomaly detection by multi-grained spatiotemporal representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17385–17394, 2024. 2