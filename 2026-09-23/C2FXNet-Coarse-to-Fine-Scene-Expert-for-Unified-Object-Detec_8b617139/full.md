# C2FXNet: Coarse-to-Fine Scene Expert for Unified Object Detection across Adverse Weather

Tianle Fang<sup>∗</sup>   
School of Computer Science and   
Information Security   
Guilin University of Electronic   
Technology   
Guilin, Guangxi, China   
polarisftl123@gmail.com   
Zhenbing Liu<sup>∗</sup>   
School of Artificial Intelligence   
Guilin University of Electronic   
Technology   
Guilin, Guangxi, China   
zbliu@guet.edu.cn   
Bolun Li   
School of Computer Science and   
Information Security   
Guilin University of Electronic   
Technology   
Guilin, Guangxi, China   
blli@mails.guet.edu.cn   
Chong Yin   
School of Computer Science and   
Technology   
Hainan University   
Haikou, Hainan, China   
chongyin@comp.hkbu.edu.hk   
Haoxiang Lu<sup>†</sup>   
School of Computer Science and   
Information Security   
Guilin University of Electronic   
Technology   
Guilin, Guangxi, China   
hxlu1005@guet.edu.cn

## Abstract

Object detection in adverse weather remains challenging because severe degradations weaken visual quality and disrupt semantic feature representations across diverse scenes. Existing methods usually rely on condition-specific designs, which limits their ability to generalize within a unified detector. In this paper, we propose a Coarse-to-Fine Scene Expert Network (C2FXNet) that achieves unified detection through hierarchical scene guidance. Specifically, C2FXNet introduces a dual-level guidance mechanism consisting of a Multi-step Reasoning Router (MRR), which performs GRUbased recurrent scene reasoning over compressed multi-scale visual cues and frozen coarse scene prototypes, and a Fine Scene Refinement (FSR) module, which uses image-specific semantic cues to modulate high-level features for local variation handling. Furthermore, a Scene-aware Mixture-of-Experts (SMoE) dynam ically combines scene-specific experts under the joint guidance of MRR and FSR. By coupling coarse scene reasoning with finegrained semantic refinement, C2FXNet enables robust multi-scene detection without scene-specific training. Extensive experiments on RTTS, ExDark, and our newly constructed Adverse Weather Dataset (AWD) demonstrate that C2FXNet consistently outperforms state-of-the-art methods across foggy, dark, and clear conditions, reaching 63.70%, 71.14%, and 54.19% mAP on RTTS, Ex-Dark, and AWD, respectively. The source code will be released at https://github.com/PolarisFTL/C2FXNet.

CCS Concepts • Computing methodologies → Object detection.

## Keywords

Object detection in adverse weather, Unified multi-scene detection, Scene-aware Mixture-of-Experts, Vision language models.

Tianle Fang, Zhenbing Liu, Chong Yin, Bolun Li, and Haoxiang Lu. 2026. C2FXNet: Coarse-to-Fine Scene Expert for Unified Object Detection across Adverse Weather. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3767308.3834979

## 1 Introduction

Object detection is a fundamental computer vision task, with critical applications in autonomous driving [2, 40] and outdoor surveillance [29]. However, real-world deployment often encounters adverse weather conditions (fog, rain, etc.), which significantly degrade detection performance due to reduced visibility, color distortion, and feature ambiguity. Developing robust detection systems that adapt to adverse conditions remains challenging.

Existing approaches to scene-specific detection mainly follow three paradigms. Image enhancement methods restore degraded inputs before detection but may introduce artifacts and disrupt end-toend optimization [1, 23, 48]. Multi-task frameworks jointly optimize restoration and detection, improving robustness at the cost of careful loss balancing [11, 17, 20]. Domain adaptation methods transfer detectors from clear to adverse weather but often remain specialized to particular conditions [13, 24, 25]. Fundamentally, all three paradigms treat diferent weather conditions as isolated problems rather than related tasks sharing common visual semantics. This prevents knowledge transfer across conditions and limits generalization to unseen weather variations. Our key insight is that adverse weather conditions, despite their diverse low-level appearances, share high-level semantic attributes, such as reduced visibility and contrast, that can be described through language. Vision-language models such as CLIP [31] provide transferable semantic priors for modeling these conditions as related variations rather than isolated domains. However, their global alignment lacks the hierarchical understanding and region-level adaptation required for weather adaptive detection. We therefore employ two-level textual prompts: coarse scene prompts (foggy, dark, or clear) guide recurrent scene reasoning and expert routing, while fine prompts encoding visibil ity, illumination, and object context refine regional features. This design enables hierarchical weather adaptation within a unified detection framework.

![](images/70c4fca683f60a85ab4cdaabc18ab2581f940ba9502d2b417ec128789b75abc0.jpg)

(a) Existing methods: Separate models for each scene.  
![](images/37ddf68c98736a8070f719ce0dfd03dad0e443741843a0ea985c7e1cfa7d2130.jpg)  
(b) Our C2FXNet: Single unified model via coarse-to-fine text guidance.  
Figure 1: (a) Existing methods: Separate models for each scene. (b) Our C2FXNet: Single unified model via coarse-to-fine text guidance.

Building on this insight, we introduce a coarse-to-fine scene expert network (C2FXNet) for unified object detection across adverse weather. The framework progressively refines scene understanding from coarse global reasoning to fine-grained local adaptation, culminating in dynamic expert fusion. Specifically, C2FXNet contains two complementary modules aligned with the two-level prompt hierarchy: a Multi-step Reasoning Router (MRR) that compresses multi-scale visual cues and iteratively updates a latent routing state with GRU units before matching it to frozen coarse scene prototypes, and a Fine Scene Refinement (FSR) module that uses image-specific semantic prompts to refine high-level features for local variations. On top of them, a Scene-aware Mixture-of-Experts (SMoE) dynamically combines scene-specific experts under joint coarse-to-fine guidance, enabling unified detection without manual scene classification. Our contributions are summarized as follows:

• We propose C2FXNet, a coarse-to-fine scene expert network for unified object detection under adverse weather, which progressively refines scene understanding from coarse scene reasoning to fine-grained local adaptation.

• We develop a Multi-step Reasoning Router (MRR) that performs GRU-based recurrent scene reasoning over compressed multi-scale descriptors and frozen coarse scene prototypes, producing sparse and interpretable routing weights with load-balancing regularization.

• We introduce Fine Scene Refinement (FSR) to handle intracondition variations by using image-specific semantic prompts and normalization-based channel-wise modulation with learnable afine parameters.

• We devise a Scene-aware Mixture-of-Experts (SMoE) that adaptively fuses scene-specific experts under the joint guidance of MRR and FSR. Extensive experiments on RTTS, Ex-Dark, and AWD validate the efectiveness and robustness of the proposed framework.

## 2 Related work

## 2.1 Object Detection in Adverse Weather.

Object detection is a core task in computer vision, which requires both classification and localization. Despite rapid progress, mainstream detectors such as the YOLO series [28, 37, 38] sufer performance degradation under adverse weather conditions due to image quality deterioration. Existing solutions can be categorized into three main approaches. Image enhancement methods [1, 23, 48] restore degraded inputs before detection, but often lack end-toend design and incur high computational cost. Multi-task learning frameworks [7, 11, 17] jointly optimize detection and auxiliary tasks (dehazing or low-light enhancement), improving robustness but requiring complex architectures and loss balancing. Domain adaptation techniques [13, 24, 46] leverage unsupervised transfer to adapt models to adverse weather, but these methods lead to color distortion of the image and detection inconsistency. While these methods mitigate some adverse efects, challenges remain in feature degradation and limited generalization across scenes.

## 2.2 Vision Language Models.

Vision–language models (VLMs) integrate visual and textual modalities into unified representations. Early architectures such as ViL-BERT [27] and VisualBERT [18] pioneered multimodal pretraining via transformer-based fusion. More recently, CLIP [31] and its regional extensions [4, 34, 45] scaled contrastive image–text learning and enhanced region–text alignment for downstream recognition and detection tasks. In parallel, large-scale open-vocabulary and LLM-based detectors [9, 19, 42] further bridge visual and linguistic reasoning, demonstrating the potential of multimodal supervision for robust localization and generalization. MLLM [44] leverages large language models (LLMs) for high-level reasoning and localization, alleviating the linguistic limitations of conventional detectors. Building on these advances, our method leverages text descriptions generated by an LLM to provide global-to-local textual guidance for the Multi-step Reasoning Router and Fine Scene Refinement modules under adverse weather conditions.

## 2.3 Dynamic Routing Networks.

Dynamic routing enhances neural adaptability by conditionally activating submodules based on input features. Early work, such as Capsule Networks [32], introduced routing-by-agreement to model part–whole relationships, but their limited depth constrained scalability. Later Mixture-of-Experts (MoE) models [33] employed learnable gating to selectively activate expert branches, improving eficiency and scalability. These ideas were extended in largescale systems, including GShard [15], Switch Transformer [8], and DeepSeek-MoE [6], emphasizing routing eficiency and load balance. Recent studies further refine expert specialization for vision and large-scale networks [30, 43, 47]. Building upon these advances, we propose a scene-aware mixture-of-experts that adaptively routes features to scene-specific experts under adverse weather, enhancing detection robustness in diverse conditions.

![](images/48f0cabf40dcbf6a8343a955344bc5ae071347f4382b5547ab588c5240185165.jpg)  
Figure 2: Overview of proposed C2FXNet. It consists of three key modules: the Multi-step Reasoning Router (MRR) compresses multi-scale visual cues and performs GRU-based recurrent scene reasoning under coarse text-guided prototypes to generate interpretable routing weights, the Fine Scene Refinement (FSR) refines visual features using fine-grained textual cues, and the Scene-aware Mixture-of-Experts (SMoE) dynamically fuses fog, dark, and clear experts under the joint guidance of MRR and FSR, enabling robust and scene-consistent detection. Among them, and $\beta$ denote two learnable linear layers.

## 3 Method

Overview. As illustrated in Figure 2, we propose a coarse-to-fine scene expert network that progressively refines scene understanding from coarse scene reasoning to local variation modeling. Given an image � $\sharp \mathbb { R } ^ { 3 \times H \times W }$ , the feature encoder backbone extracts multiscale features $\{ \mathcal { F } _ { 1 } , \mathcal { F } _ { 2 } , \mathcal { F } _ { 3 } \}$ , which are then processed by two complementary guidance modules. The Multi-step Reasoning Router (MRR) compresses the multi-scale features into a compact routing sequence and recurrently updates a latent routing state through GRU-based reasoning before matching the final state to frozen coarse scene prototypes (fog, dark, clear) to obtain sparse routing weights $\tilde { p } \in \mathbb { R } ^ { 3 }$ . In parallel, the Fine Scene Refinement (FSR) module uses image-specific semantic prompts to modulate the high level feature and produce a refinement feature � that captures local scene-dependent variations. The Scene-aware Mixture-of-Experts (SMoE) then dynamically fuses scene-specific expert outputs using both $\tilde { p }$ and � to produce semantically aware representations $\tilde { f } _ { \mathrm { o u t } } .$ These refined features are fed into a standard PAFPN [22] neck and YOLO [38] detection head for final prediction.

The key innovation lies in coupling GRU-based coarse scene reasoning in MRR with text-guided fine feature modulation in FSR, and using their outputs to steer expert fusion in SMoE. This coarseto-fine interaction enables robust and interpretable all-weather detection by dynamically adjusting expert contributions according to both global scene semantics and local visual variations.

## 3.1 Multi-step Reasoning Router

The proposed Multi-step Reasoning Router (MRR) performs sceneaware expert selection by combining frozen language priors with recurrent visual reasoning. Unlike conventional routers that infer routing weights from a single visual descriptor, MRR organizes compressed multi-scale evidence into a short reasoning sequence and progressively refines the routing decision, which leads to more reliable scene assignment under adverse conditions.

For each coarse scene category $s \in$ {fog, dark, clear}, we define a small set of coarse textual prompts $\mathcal { T } _ { s }$ . A frozen text encoder $\varphi$ maps each prompt $t \in \mathcal { T } _ { s }$ to a text embedding

$$
\begin{array} { r } { e _ { t } = \varphi ( t ) , \qquad e _ { t } \in \mathbb { R } ^ { d _ { t } } , } \end{array}\tag{1}
$$

which is projected into the routing space by a lightweight projector $P : \mathbb { R } ^ { d _ { t } }  \mathbb { R } ^ { d }$ . The projected embeddings are averaged within each scene category and $\ell _ { 2 }$ -normalized to form a prototype $k _ { s } \in \mathbb { R } ^ { d }$ . The resulting prototype matrix is

$$
\begin{array} { r } { W = [ k _ { \mathrm { f o g } } , k _ { \mathrm { d a r k } } , k _ { \mathrm { c l e a r } } ] ^ { \intercal } \in \mathbb { R } ^ { 3 \times d } . } \end{array}\tag{2}
$$

Given multi-scale backbone features $\{ \mathcal { F } _ { 1 } , \mathcal { F } _ { 2 } , \mathcal { F } _ { 3 } \}$ , the Feature Compression Module first produces an ordered set of compact routing descriptors

$$
\{ c _ { n } \} _ { n = 1 } ^ { N } = \psi ( { \mathcal { F } } _ { 1 } , { \mathcal { F } } _ { 2 } , { \mathcal { F } } _ { 3 } ) , \qquad c _ { n } \in \mathbb { R } ^ { d _ { h } } ,\tag{3}
$$

![](images/7fa4179984a84c3c99d8f93dae675906e0da5e4b20a3a01173c61bc15f43472e.jpg)  
Figure 3: Feature-level interpretability of FSR in clear, dark, and foggy scenes. FSR refines noisy SMoE activations into more coherent, object-centric, and weather-consistent feature responses.

where $\psi ( \cdot )$ denotes multi-scale compression and feature re-organization, and � is the number of reasoning steps. Starting from a learnable initial state $h _ { 0 } ~ \in ~ \mathbb { R } ^ { d _ { h } }$ , MRR sequentially aggregates the routing descriptors through a GRU:

$$
h _ { n } = \mathrm { G R U } ( c _ { n } , h _ { n - 1 } ) , \qquad n = 1 , \ldots , N .\tag{4}
$$

The final reasoning state is projected into the routing space:

$$
z = \Pi ( h _ { N } ) , \qquad z \in \mathbb { R } ^ { d } .\tag{5}
$$

MRR then computes scene afinities by matching the normalized routing feature with the fixed scene prototypes:

$$
\dot { p } = \mathrm { s o f t m a x } \left( \frac { \hat { z } W ^ { \top } } { \tau } \right) , \qquad \hat { z } = \frac { z } { \| z \| _ { 2 } } ,\tag{6}
$$

where � is a temperature parameter. To encourage sparse expert activation, we retain only the Top-� entries and renormalize the routing weights:

$$
\tilde { p } = \frac { p \odot m } { \sum _ { s } p _ { s } m _ { s } + \epsilon } ,\tag{7}
$$

where $m \in \{ 0 , 1 \} ^ { 3 }$ is the Top-� mask. The sparse routing weights $\tilde { p }$ are finally used to select and fuse the scene experts. By accumulating evidence across multiple compressed cues before prototype matching, MRR yields a more stable and interpretable routing process than one-shot similarity-based assignment.

## 3.2 Fine Scene Refinement

While MRR provides coarse scene-level routing, it cannot directly enhance the adaptability of each expert to subtle visual variations. To address this limitation, the Fine Scene Refinement (FSR) mod ule introduces fine-grained textual conditioning on the high-level backbone feature F , enabling the model to focus on task-relevant regions and better align features under adverse conditions.

For each image, we utilize Qwen2.5-VL [3] to generate a set of Fine-Text Prompts $\mathcal { P } = \{ t _ { i } \} _ { i = 1 } ^ { N }$ that describe the scene and the objects in it (“an automobile in heavy fog”, “a person sitting on a boat in daylight”). The detailed text generation procedure is provided in the Appendix (Section A.5). The frozen text encoder � maps each prompt $\mathcal { P } _ { i }$ to an embedding $e _ { i } = \varphi ( \mathcal { P } _ { i } )$ ), and the same lightweight projector � used in MRR is applied to obtain projected vectors $z _ { i } = P ( e _ { i } ) \in \mathbb { R } ^ { d }$ . We then average the set $\{ z _ { i } \} _ { i = 1 } ^ { N }$ and apply $\ell _ { 2 }$ normalization to obtain a single per-image textual representation $t \in \mathbb { R } ^ { d }$ , which serves as the fine-text prompt.

Given the feature map $\scriptstyle x = { \mathcal { F } } _ { 1 }$ and the textual vector $t \in \mathbb { R } ^ { d } ;$ , FSR injects fine-grained semantic information through channel-wise refinement. We first apply Group Normalization to �, and then use two learnable linear layers $\gamma ( \cdot ) , \bar { \beta } ( \cdot ) : \mathbb { R } ^ { d } \to \mathbb { R } ^ { C }$ to generate scaling and shifting coeficients from �:

$$
\gamma ( t ) = W _ { \gamma } t + b _ { \gamma } , \qquad \beta ( t ) = W _ { \beta } t + b _ { \beta } ,\tag{8}
$$

where $W _ { \gamma } , W _ { \beta } \in \mathbb { R } ^ { C \times d }$ and $b _ { \gamma } , b _ { \beta } \in \mathbb { R } ^ { C }$ are learnable parameters. The resulting coeficients are broadcast to the spatial dimensions and applied to the normalized feature map:

$$
u = \mathrm { G N } ( x ) \cdot { \big ( } 1 + \gamma ( t ) { \big ) } + \beta ( t ) ,\tag{9}
$$

where $\gamma ( t ) , \beta ( t ) \in \mathbb { R } ^ { C }$ are implicitly reshaped to $\mathbb { R } ^ { B \times C \times H \times W }$ for element-wise refinement.

This refinement is applied exclusively within the residual enhancement paths of each expert, leaving the backbone and MRR routing computations unchanged. Such a decoupled design stabilizes training: MRR decides which experts to activate at the coarse scene level, whereas FSR refines how each expert adapts to local scene variations.

To analyze the role of FSR, Fig. 3 visualizes SMoE output features with and without FSR across clear, dark, and foggy scenes. Without FSR, the activations are noisy and fragmented, showing background interference in clear scenes, illumination bias in dark scenes, and texture flattening in foggy scenes. With FSR, the responses become more concentrated on foreground objects, with clearer boundaries and less background interference. These results show that FSR complements MRR: MRR selects coarse scene experts, whereas FSR injects image-specific semantic cues to improve local discriminability after global routing.

## 3.3 Scene-aware Mixture-of-Experts

The Scene-aware Mixture-of-Experts (SMoE) module serves as the final adaptive stage of C2FXNet, receiving joint guidance from MRR and FSR. Specifically, it leverages the sparse routing weights �˜ from MRR (Eq. (7)) and the scene-aware refinement feature � from FSR (Eq. (9)) as complementary signals. Guided by these inputs, SMoE dynamically combines multiple scene-specific experts to adapt expert contributions across diverse environmental conditions.

Each expert branch specializes in one coarse scene domain (fog, dark, or clear) and operates on the refined high-level feature $u \in$ $\mathbb { R } ^ { B \times C \times H \times W }$ . Within each branch, a domain-specific enhancement operator is used to capture the characteristics of its corresponding environment. For foggy conditions, we employ a Feature-Aware Frequency Enhancement (FAFE) operator to restore high-frequency

![](images/245b3a7a8480e6968b6465de8d54d4a316c35c0cd91237117ef0d1b1e03e0a0b.jpg)  
Figure 4: t-SNE visualization of feature embeddings. C2FXNet produces more compact and better-separated clusters than the baseline for fog, dark, and clear scenes.

details and contrast:

$$
\mathrm { F A F E } ( u ) = \mathrm { L e a k y R e L U } \big ( ( u ^ { \prime } \odot u ) - u ^ { \prime } + 1 \big ) ,\tag{10}
$$

where ${ \boldsymbol { u } } ^ { \prime } { = } \mathbf { \boldsymbol { C } } \mathbf { o n v } _ { 1 \times 1 } ( { \boldsymbol { u } } )$ is a linear projection and ⊙ denotes elementwise multiplication. For low-light scenes, the dark expert applies a Low-Light Frequency Restoration (LLFR) operator to rebalance illumination:

$$
\mathrm { L L F R } ( u ) = \mathrm { L e a k y R e L U } ( ( \alpha | u ^ { \prime } + u | + \epsilon ) ^ { \rho } ) ,\tag{11}
$$

where $\alpha , \rho$ and � are learnable or fixed hyperparameters controlling the enhancement strength. The clear expert uses two standard convolutional layers to provide neutral enhancement in well-lit conditions. Let $\boldsymbol { v } _ { s } ^ { ( g ) } \in \mathbb { R } ^ { \bar { \boldsymbol { B } } \times \boldsymbol { C } \times \boldsymbol { H } \times \boldsymbol { W } }$ denote the output of scene expert $s \in \{ \mathrm { f o g } \colon$ , dark, clear} in the �-th expert group when processing �. Given the routing weights $\tilde { p } = [ \tilde { p } _ { \mathrm { f o g } }$ , <sub>�</sub>˜<sub>dark</sub>, $\tilde { p } _ { \mathrm { c l e a r } } ]$ from MRR, the fused output of the �-th group is

$$
\tilde { f } ^ { ( g ) } = \sum _ { s } \tilde { p } _ { s } v _ { s } ^ { ( g ) } , \qquad \tilde { f } _ { \mathrm { o u t } } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \tilde { f } ^ { ( g ) } ,\tag{12}
$$

where � is the total number of expert groups. In practice, � con trols the model capacity, while the same sparse routing weights $\tilde { p } ,$ inferred by MRR through recurrent scene reasoning, are shared across groups. The fused feature $\tilde { f } _ { \mathrm { o u t } }$ is both scene-consistent and context-aware, providing a unified representation for robust object detection under adverse weather and lighting conditions.

As visualized by the t-SNE embeddings in Fig. 4, the baseline features exhibit severe overlap across fog/dark/clear domains, indi cating weak scene discriminability. In contrast, C2FXNet produces clearly separated clusters with improved intra-domain compactness, validating that SMoE yields a more scene-consistent representation. By coupling the coarse recurrent reasoning of MRR with the finegrained semantic modulation of FSR, SMoE bridges global semantic understanding and local feature adaptation. The fused feature $\tilde { f } _ { \mathrm { o u t } }$ is therefore both scene-consistent and context-aware, providing a unified representation for robust object detection under adverse weather and lighting conditions. Finally, C2FXNet is trained end-toend with a joint loss that combines detection, scene classification, and a load-balancing loss on the MRR:

$$
\begin{array} { r } { \mathcal { L } _ { t o t a l } = \mathcal { L } _ { d e t } + \lambda _ { 1 } \mathcal { L } _ { s } + \lambda _ { 2 } \mathcal { L } _ { a u x } , } \end{array}\tag{13}
$$

where $\mathcal { L } _ { d e t }$ is the standard YOLO detection loss [38], and $\mathcal { L } _ { s }$ is a cross-entropy loss supervising the scene prediction branch. The

Table 1: Statistics of datasets used for training and evaluation.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Scene</td><td rowspan="2">#Img</td><td colspan="5">#Bounding Boxes / Class</td></tr><tr><td>Pers.</td><td>Bic.</td><td>Car</td><td>Moto.</td><td>Bus</td></tr><tr><td>AWD (Train)</td><td>Fog/Dark/Clear</td><td>41,685</td><td>157,707</td><td>4,014</td><td>26,064</td><td>4,875</td><td>3,489</td></tr><tr><td>AWD (Test)</td><td>Fog/Dark/Clear</td><td>8,805</td><td>33,012</td><td>948</td><td>5,796</td><td>1,113</td><td>855</td></tr><tr><td>RTTS</td><td>Fog</td><td>4,322</td><td>7,950</td><td>534</td><td>18,415</td><td>862</td><td>1,838</td></tr><tr><td>ExDark</td><td>Dark</td><td>7,363</td><td>7,460</td><td>1,120</td><td>2,927</td><td>1,072</td><td>706</td></tr></table>

MRR load-balancing loss $\mathcal { L } _ { a u x }$ is computed from the routing weights over � experts:

$$
\mathcal { L } _ { a u x } = R \sum _ { r = 1 } ^ { R } \bar { r } _ { r } ^ { 2 } , \qquad \bar { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r _ { i } ,\tag{14}
$$

where $r _ { i } \in \mathbb { R } ^ { 3 }$ denotes the MRR routing weights for the �-th image.   
This term encourages balanced expert utilization within the MRR.

## 4 Experiments

Datasets and Evaluation Metrics. We evaluate our method on three datasets: Adverse Weather Dataset (AWD), RTTS [16], and ExDark [26]. AWD is a synthetic dataset we constructed using CycleGAN [49] applied to MS COCO [21] to simulate foggy and low-light conditions for both training and evaluation. RTTS and ExDark contain real foggy and low-light images, respectively, while Rain and Snow are synthetic datasets created by applying physical degradation models to VOC images. Table 1 summarizes the statistics and label distributions of each dataset.

To ensure fair comparison, all datasets share five common object categories. We use Parameters (Params), Floating Point Operations (FLOPs), and mean Average Precision (mAP) for evaluation.

Implementation Details. The Stochastic Gradient Descent (SGD) is applied as the optimizer with an initial learning rate of 0.01. The momentum is set to 0.937, and weight decay is set to 5e-4. The batch size is set to 16. We utilize NVIDIA GeForce RTX3090 GPU to train the model for 100 epochs. For hyperparameter settings, we empirically set the temperature coeficient �=1 in MRR, Top-�=1 for sparse expert selection, and $\epsilon { = } 1 0 ^ { - 9 }$ for normalization stability. In the LLFR operator, �=0.5, $\rho { = } 1 . 2 ,$ , and $\epsilon { = } 1 0 ^ { - 6 }$ are adopted for illumination restoration. For the overall loss, the weighting coeficients are $\lambda _ { 1 } { = } 0 . 1$ and $\lambda _ { 2 } { = } 0 . 0 5$ to balance scene classification and routing regularization. To obtain fine-grained textual descriptions for instance-level refinement, we employ Qwen2.5-VL [3] deployed via the Ollama framework, which generates multiple object-centric textual descriptions for each image. Additional experimental settings are provided in the Appendix (Section A.8).

## 4.1 Comparisons with State-of-the-Art Methods

We compare C2FXNet with state-of-the-art detectors, including TogetherNet [41], IA-YOLO [24], GDIP-YOLO [13], and recent YOLO variants from YOLOv9s to YOLOv13s [12, 14, 35, 36, 39] and EEnvA-Mamba [5]. All methods are evaluated on RTTS, ExDark, and AWD, covering foggy, low-light, and clear-weather conditions.

4.1.1 Quantitative Results on All-Weather Benchmarks. Table 2 presents comprehensive comparisons across three benchmarks.

Table 2: Comparison of C2FXNet with previous detectors on RTTS, ExDark, and AWD datasets. C2FXNet attains optimal parameter eficiency while achieving the highest detection accuracy. The best and second-best results are boldfaced and underlined.
<table><tr><td>Method</td><td>Dataset</td><td>Publication</td><td>Params ↓</td><td>FLOPs ↓</td><td>Bicycle</td><td>Bus</td><td>Car</td><td>Motorbike</td><td>Person</td><td>mAP (%) ↑</td></tr><tr><td>TogetherNet [41]</td><td rowspan="10">RTTS</td><td>CGF&#x27;22 AAAI&#x27;22</td><td>15.7M</td><td>26.1G</td><td>49.50</td><td>27.06</td><td>70.86</td><td>46.69</td><td>81.66</td><td>55.15</td></tr><tr><td>IA-YOLO [24]</td><td></td><td>58.9M</td><td>65.6G</td><td>39.06</td><td>17.87</td><td>41.29</td><td>30.63</td><td>58.76</td><td>37.52</td></tr><tr><td>GDIP-YOLO [13]</td><td>ICRA&#x27;23</td><td>68.0M</td><td>105.2G</td><td>45.89</td><td>11.58</td><td>49.50</td><td>32.96</td><td>71.98</td><td>42.38</td></tr><tr><td>YOLOv9s [39]</td><td>ECCV&#x27;24</td><td>7.1M</td><td>26.4G</td><td>40.73</td><td>23.41</td><td>61.13</td><td>33.17</td><td>79.76</td><td>47.64</td></tr><tr><td>YOLOv10s [36]</td><td>NeurIPS&#x27;24</td><td>7.6M</td><td>24.5G</td><td>42.22</td><td>23.65</td><td>60.42</td><td>33.60</td><td>78.72</td><td>47.72</td></tr><tr><td>YOLOv11s [12]</td><td></td><td>9.0M</td><td>21.5G</td><td>53.64</td><td>37.68</td><td>72.56</td><td>53.23</td><td>82.87</td><td>60.00</td></tr><tr><td>YOLOv12s [35]</td><td>NeurIPS&#x27;25</td><td>8.6M</td><td>19.3G</td><td>58.30</td><td>31.01</td><td>69.67</td><td>52.64</td><td>81.43</td><td>58.61</td></tr><tr><td>YOLOv13s [14]</td><td>Arxiv&#x27;25</td><td>8.6M</td><td>21.7G</td><td>54.75</td><td>39.15</td><td>72.80</td><td>52.61</td><td>83.77</td><td>60.62</td></tr><tr><td>EEnvA-Mamba [5]</td><td>PR&#x27;26</td><td>6.9M</td><td>15.6G</td><td>53.80</td><td>71.60</td><td>77.60</td><td>47.00</td><td>53.40</td><td>60.70</td></tr><tr><td>C2FXNet (Ours)</td><td></td><td>6.7M</td><td>22.0G</td><td>57.90</td><td>41.90</td><td>76.53</td><td>57.09</td><td>85.08</td><td>63.70</td></tr><tr><td>Zero-DCE [10]</td><td rowspan="7">ExDark</td><td>CVPR&#x27;20 AAAI&#x27;22</td><td>12.1M</td><td>13.9G</td><td>50.21</td><td>61.21</td><td>46.00</td><td>37.35</td><td>49.72</td><td>48.90</td></tr><tr><td>IA-YOLO [24]</td><td></td><td>58.9M</td><td>65.6G</td><td>68.47</td><td>72.56</td><td>59.96</td><td>53.77</td><td>73.31</td><td>65.61</td></tr><tr><td>GDIP-YOLO [13]</td><td>ICRA&#x27;23</td><td>68.0M</td><td>105.2G</td><td>53.31</td><td>62.79</td><td>42.72</td><td>39.09</td><td>49.72</td><td>49.53</td></tr><tr><td>YOLOv9s [39]</td><td>ECCV&#x27;24</td><td>7.1M</td><td>26.4G</td><td>51.35</td><td>66.61</td><td>46.97</td><td>43.67</td><td>60.56</td><td>53.83</td></tr><tr><td>YOLOv10s [36]</td><td>NeurIPS&#x27;24</td><td>7.6M</td><td>24.5G</td><td>53.47</td><td>61.69</td><td>45.65</td><td>46.52</td><td>59.22</td><td>53.31</td></tr><tr><td>YOLOv11s [12]</td><td></td><td>9.0M</td><td>21.5G</td><td>64.36</td><td>80.87</td><td>62.82</td><td>60.80</td><td>70.19</td><td>67.81</td></tr><tr><td>YOLOv12s [35]</td><td>NeurIPS&#x27;25</td><td>8.6M</td><td>19.3G</td><td>64.62</td><td>76.31</td><td>60.32</td><td>61.75</td><td>68.42</td><td>66.23</td></tr><tr><td>YOLOv13s [14]</td><td></td><td>Arxiv&#x27;25</td><td>8.6M</td><td>21.7G</td><td>65.53</td><td>77.92</td><td>59.40</td><td>60.15</td><td>71.28</td><td>66.86</td></tr><tr><td>C2FXNet (Ours)</td><td></td><td></td><td>6.7M</td><td>22.0G</td><td>68.65</td><td>81.16</td><td>66.59</td><td>63.24</td><td>76.08</td><td>71.14</td></tr><tr><td>YOLOv9s [39]</td><td rowspan="5">AWD</td><td>ECCV&#x27;24 NeurIPS&#x27;24</td><td>7.1M</td><td>26.4G</td><td>22.21</td><td>54.20</td><td>38.27</td><td>41.20</td><td>60.28</td><td>43.23</td></tr><tr><td>YOLOv10s [36]</td><td></td><td>7.6M</td><td>24.5G</td><td>22.80</td><td>53.99</td><td>37.31</td><td>43.49</td><td>59.53</td><td>43.42</td></tr><tr><td>YOLOv11s [12]</td><td></td><td>9.0M</td><td>21.5G</td><td>33.72</td><td>65.32</td><td>49.15</td><td>54.06</td><td>67.08</td><td>53.87</td></tr><tr><td>YOLOv12s [35]</td><td>NeurIPS&#x27;25</td><td>8.6M</td><td>19.3G</td><td>34.51</td><td>64.23</td><td>46.87</td><td>56.30</td><td>66.93</td><td>53.77</td></tr><tr><td>YOLOv13s [14]</td><td>Arxiv&#x27;25</td><td>8.6M</td><td>21.7G</td><td>33.82</td><td>62.63</td><td>45.70</td><td>53.55</td><td>63.17</td><td>51.17</td></tr><tr><td>C2FXNet (Ours)</td><td></td><td></td><td>6.7M</td><td>22.0G</td><td>34.73</td><td>64.25</td><td>48.02</td><td>54.45</td><td>69.50</td><td>54.19</td></tr></table>

![](images/f82a3bd4644e89be1df2841c21e8100fcd1c8c00b7ca27896ba6fde6a70af44e.jpg)  
Figure 5: Visual comparison on the RTTS dataset under foggy conditions. C2FXNet detects more objects with clearer boundaries and higher confidence, especially for small or distant targets that other methods miss.

On RTTS, C2FXNet achieves 63.70% mAP, surpassing YOLOv13s by 3.08% with notable gains in Bus (+2.75%), Car (+3.73%), Motorbike (+4.48%), and Person (+1.31%). These improvements indicate that our semantic-guided feature refinement efectively preserves discriminative object structures under fog-induced degradation, especially for categories sensitive to blurred boundaries and lowcontrast appearance.

On ExDark, our method reaches 71.14% mAP, outperforming YOLOv11s by 3.33% and Zero-DCE by 22.24%. The substantial mar gin over enhancement-based Zero-DCE highlights the advantage of performing feature-space adaptation with semantic priors, which avoids pixel-level artifacts while strengthening missing scene cues under severe low-light conditions. Meanwhile, compared with recent lightweight YOLO variants, C2FXNet consistently improves all categories and achieves the best overall accuracy, demonstrating the efectiveness of the proposed coarse-to-fine expert routing strategy for low-light perception.

On AWD, C2FXNet achieves 54.19% mAP, exceeding YOLOv11s and YOLOv12s by 0.32% and 0.42%, respectively. Although YOLOv11s performs slightly better on Bus and Car, our model achieves higher AP on Bicycle and Person and yields the best overall mAP. This suggests that C2FXNet favors more balanced cross-scene generalization rather than over-optimizing a few dominant categories. In particular, the consistent gains on challenging classes such as Motorbike and Person across RTTS and ExDark further verify the robustness of the proposed recurrent scene-aware expert routing mechanism under diverse adverse weather conditions.

Scene-wise Adaptation Analysis on AWD. To validate the scene-adaptive capability of our text-guided framework, we compare C2FXNet with the baseline (YOLOv7-T [38]) across diferent weather conditions on the AWD dataset. Table 3 presents the per-scene breakdowns. In foggy scenes, C2FXNet achieves 53.28% mAP (+1.19%), with notable gains in Bus (+2.24%) and Motorbike (+2.36%). This improvement indicates that MRR can accumulate reliable coarse scene evidence before dispatching features to the fog expert, while FSR further strengthens object-aware responses under haze-induced contrast degradation. In dark scenes, C2FXNet reaches 51.30% mAP (+1.26%), with the largest improvement on Motorbike (+3.79%). This gain validates the complementarity between MRR and FSR: the former stabilizes expert selection under severe illumination ambiguity, whereas the latter selectively ampli fies object-relevant channels from semantic descriptions such as “a motorcycle with dim headlights”. Under clear weather, C2FXNet attains 59.22% mAP (+0.77%), with strong gains in Bus (+3.10%) and Motorbike (+2.26%). These improvements show that the proposed semantic priors remain useful even in non-degraded scenes, where coarse scene reasoning and fine textual cues help resolve spatial ambiguities in crowded layouts.

![](images/299410a67c5b915ebf57f4fa4c60804aed53b6881a9ef455796ca4bc2d16fb35.jpg)  
Figure 6: Visual comparison on the ExDark dataset under low-light conditions. C2FXNet achieves more accurate and confident detections, successfully identifying objects in dark and overexposed areas.

![](images/e9642010382f039ed930bcebf8fd5678cc42ae692f4ab9ef4ffe2de223156b90.jpg)  
Figure 7: Visual comparison on the AWD dataset under dark, foggy, and clear conditions. C2FXNet maintains consistent detection quality and higher confidence across scenes, highlighting its strong cross-weather adaptability.

4.1.2 Qualitative Results on All-Weather Scenes. To further substan tiate the quantitative, we present qualitative comparisons between C2FXNet and the representative approach on the RTTS, ExDark, and AWD datasets, covering diverse weather conditions.

Foggy Scenes (RTTS). As shown in Figure 5, C2FXNet yields denser and more accurate detections in heavy fog. IA-YOLO, GDIP YOLO, and recent YOLO variants often miss distant pedestrians or partially occluded vehicles because fog severely suppresses edges and contrast. With MRR progressively reasoning over compressed multi-scale cues before routing features toward fog experts, together with FAFE-based enhancement, our model better restores highfrequency structures and keeps small or distant objects separable from the hazy background.

Table 3: Comparison of C2FXNet and the baseline on the AWD dataset under fog, dark, and clear scenes. C2FXNet yields consistent gains across most categories, highlighting the efectiveness of scene-aware adaptation.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Scene</td><td rowspan=1 colspan=1>Bicycle</td><td rowspan=1 colspan=1>Bus</td><td rowspan=1 colspan=1>Car</td><td rowspan=1 colspan=1>Motorbike</td><td rowspan=1 colspan=1>Person</td><td rowspan=1 colspan=1>mAP (%) ↑</td></tr><tr><td rowspan=1 colspan=1>BaselineC2FXNet (Ours)</td><td rowspan=1 colspan=1>Fog</td><td rowspan=1 colspan=1>33.0734.46</td><td rowspan=1 colspan=1>61.3463.58</td><td rowspan=1 colspan=1>45.7745.01</td><td rowspan=1 colspan=1>54.5856.94</td><td rowspan=1 colspan=1>65.6766.42</td><td rowspan=1 colspan=1>52.0953.28</td></tr><tr><td rowspan=1 colspan=1>BaselineC2FXNet (Ours)</td><td rowspan=1 colspan=1>Dark</td><td rowspan=1 colspan=1>30.4831.66</td><td rowspan=1 colspan=1>62.6263.91</td><td rowspan=1 colspan=1>44.6144.13</td><td rowspan=1 colspan=1>49.3853.17</td><td rowspan=1 colspan=1>63.1163.61</td><td rowspan=1 colspan=1>50.0451.30</td></tr><tr><td rowspan=1 colspan=1>BaselineC2FXNet (Ours)</td><td rowspan=1 colspan=1>Clear</td><td rowspan=1 colspan=1>41.7942.61</td><td rowspan=1 colspan=1>65.4768.57</td><td rowspan=1 colspan=1>54.6354.58</td><td rowspan=1 colspan=1>57.8860.14</td><td rowspan=1 colspan=1>72.4873.77</td><td rowspan=1 colspan=1>58.4559.22</td></tr></table>

Low-Light Scenes (ExDark). Figure 6 presents detection results on nighttime scenes with extreme illumination variations. Enhancement-based approaches like Zero-DCE tend to introduce ringing artifacts and noise, which translate into false positives and inaccurate box boundaries, while the YOLO series frequently misses small or distant objects that are barely visible against dark backgrounds. By incorporating the FSR module, C2FXNet injects fine-text prompt (“vehicle in dark street” or “person near bright sign”) into the high-level features and applies the LLFR operator to selectively amplify informative channels while suppressing illumi nation noise. As a result, our model produces more stable confidence scores, reduces background clutter, and yields more precise localizations in underexposed regions than both enhancement-based and purely convolutional methods.

Multi-Scenes (AWD). As shown in Figure 7. In clear weather, C2FXNet and strong YOLO variants such as YOLOv12s generate similarly accurate boxes on large, well-illuminated objects. Under adverse conditions, however, the advantage of our design becomes more pronounced: in dark and foggy scenes with mixed degradations and spatially varying visibility, only C2FXNet can stably detect most objects across all categories, while YOLOv9s–YOLOv11s produce many false negatives and fragmented boxes, especially for small or partially occluded targets. This robustness stems from the joint efect of MRR, FSR, and SMoE: MRR recurrently accumulates coarse scene evidence before routing, the fine-text prompts adapt feature refinement to the current image, and the expert fusion mechanism balances their contributions across scene-specific branches.

Table 4: Ablation for C2FXNet components on all-weather benchmarks. We adopt two frozen text encoders: all-MiniLM-L6-v2 and ViT-B-32. “� × SMoE” denotes the number of expert groups, each containing three sub-experts corresponding to the above scenes, while multiple groups are fused by diferent aggregation strategies (mean, sum, concat). “<sup>✓</sup>” and “✗” denote with and without each component, respectively.
<table><tr><td rowspan="2">Ablation</td><td rowspan="2">MRR</td><td rowspan="2">FSR</td><td rowspan="2">Text Encoder</td><td rowspan="2">SMoE</td><td rowspan="2">Aggregation</td><td colspan="3">mAP (%) ↑</td></tr><tr><td>RTTS</td><td>ExDark</td><td>AWD</td></tr><tr><td>(a)</td><td>x</td><td>x</td><td>x</td><td>X</td><td></td><td>61.99</td><td>69.16</td><td>53.53</td></tr><tr><td>(b)</td><td>√</td><td>x</td><td>x</td><td>1×</td><td></td><td>58.91-3.08</td><td> $6 9 . 7 4 \substack { + 0 . 5 8 }$ </td><td>48.61-4.92</td></tr><tr><td>(c)</td><td>x</td><td>√</td><td>x</td><td>1X</td><td></td><td> $6 0 . 0 5 _ { - 1 . 9 4 }$ </td><td> $6 9 . 5 7 _ { + 0 . 4 1 }$ </td><td> $4 8 . 8 0 _ { - 4 . 7 3 }$ </td></tr><tr><td>(d)</td><td>√</td><td>x</td><td>all-MiniLM-L6-v2</td><td>1×</td><td></td><td>61.38-0.61</td><td> $6 9 . 9 5 _ { + 0 . 7 9 }$ </td><td> $5 2 . 0 9 _ { - 1 . 4 4 }$ </td></tr><tr><td>(e)</td><td>√</td><td>√</td><td>ViT-B-32</td><td>1×</td><td></td><td>59.72-2.27</td><td> $7 1 . 0 3 _ { + 1 . 8 7 }$ </td><td> $5 0 . 1 6 _ { - 3 . 3 7 }$ </td></tr><tr><td>(f)</td><td>√</td><td>√</td><td> $a l l - M i n i L M - L 6 { - } \nu 2$ </td><td>2×</td><td>Mean</td><td> $5 9 . 2 1 _ { - 2 . 7 8 }$ </td><td> $7 0 . 1 8 _ { + 1 . 0 2 }$ </td><td>50.58-2.95</td></tr><tr><td>(g)</td><td>√</td><td>√</td><td> $a l l - M i n i L M - L 6 { - } \nu 2$ </td><td>2×</td><td>Sum</td><td> $5 8 . 8 8 _ { - 3 . 1 1 }$ </td><td> $7 1 . 2 9 _ { + 2 . 1 3 }$ </td><td>49.83-3.70</td></tr><tr><td>(h)</td><td>√</td><td>√</td><td> $a l l - M i n i L M - L 6 { - } \nu 2$ </td><td>2×</td><td>Concat</td><td> $6 1 . 1 2 _ { - 0 . 8 7 }$ </td><td> $\mathbf { 7 1 . 9 8 _ { + 2 . 8 2 } }$ </td><td>49.91-3.62</td></tr><tr><td>(i)</td><td>√</td><td>√</td><td> $a l l - M i n i L M - L 6 { - } \nu 2$ </td><td>1×</td><td></td><td> $\mathbf { 6 3 . 7 0 _ { + 1 . 7 1 } }$ </td><td> $7 1 . 1 4 \substack { + 1 . 9 8 }$ </td><td> $\mathbf { 5 4 . 1 9 _ { + 0 . 6 6 } }$ </td></tr></table>

![](images/929d3f8f570436d712784dc61e14c906866d89f38c3281a83b955ea68096f1a5.jpg)  
Figure 8: Visualization of detection results produced by C2FXNet on the VOC-Rain and VOC-Snow datasets.

Table 5: Quantitative evaluation on VOC-Rain and VOC-Snow, comparing the baseline model with our C2FXNet across five object categories and the overall mAP.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Bicycle</td><td rowspan=1 colspan=1>Bus</td><td rowspan=1 colspan=1>Car</td><td rowspan=1 colspan=1>Motorbike</td><td rowspan=1 colspan=1>Person</td><td rowspan=1 colspan=1>mAP (%) ↑</td></tr><tr><td rowspan=1 colspan=1>BaselineC2FXNet (Ours)</td><td rowspan=1 colspan=1>Rain</td><td rowspan=1 colspan=1>40.9437.65</td><td rowspan=1 colspan=1>67.5774.96</td><td rowspan=1 colspan=1>50.1148.82</td><td rowspan=1 colspan=1>46.7246.40</td><td rowspan=1 colspan=1>65.2668.22</td><td rowspan=1 colspan=1>54.1255.21</td></tr><tr><td rowspan=1 colspan=1>BaselineC2FXNet (Ours)</td><td rowspan=1 colspan=1>Snow</td><td rowspan=1 colspan=1>31.3328.28</td><td rowspan=1 colspan=1>59.2761.42</td><td rowspan=1 colspan=1>30.1833.50</td><td rowspan=1 colspan=1>30.8130.90</td><td rowspan=1 colspan=1>52.3255.85</td><td rowspan=1 colspan=1>40.7841.99</td></tr></table>

## 4.2 Ablation Studies

As shown in Table 4, our ablation study shows that C2FXNet benefits from the collaboration of MRR, FSR, the frozen text encoder, and SMoE, rather than any single module. MRR performs coarse recurrent scene reasoning, while FSR provides semantically aligned local modulation, jointly enabling reliable expert routing and adaptive fusion.

From rows (b)–(d), enabling only partial routing-related components yields limited gains and often degrades performance. For example, row (b) drops by 3.08% on RTTS and 4.92% on AWD compared with the baseline, indicating that expert fusion alone cannot provide stable routing without suficient semantic guidance. After introducing the frozen text encoder in row (d), performance becomes notably more stable, especially on AWD, which is reduced to only 1.44% below the baseline, validating the benefit of coarse lan guage priors for scene discrimination. Rows (e)–(h) further evaluate the efects of FSR, text encoder choice, and multiple expert groups. With FSR enabled, the model consistently improves on ExDark, e.g., achieving gains of +1.87% in row (e) and +2.82% in row (h), which demonstrates the advantage of semantic alignment under low-light conditions. However, increasing the number of expert groups or changing the aggregation strategy leads to mixed results suggesting that simply enlarging expert capacity does not necessarily improve cross-domain robustness.

When all components are jointly activated in row (i), C2FXNet achieves the best overall trade-of and delivers consistent improvements across all benchmarks, with gains of+1.71% on RTTS, +1.98% on ExDark, and +0.66% on AWD over the baseline. This verifies that the full model forms an efective cooperative framework that unifies recurrent scene reasoning, text-aware local modulation, and adaptive expert fusion for robust all-weather object detection.

Generalization of rain and snow weather. To further evaluate the robustness and cross-weather generalization, we evaluate C2FXNet on two challenging adverse-weather benchmarks: VOC-Rain and VOC-Snow. Both datasets are constructed by synthesizing rain and snow degradations on PASCAL VOC images, producing realistic conditions with heavy raindrops, motion streaks, snowflakes, haze-like artifacts, and reduced visibility. As shown quantitatively in Table 5 and qualitatively in Fig. 8. The model demonstrates strong robustness under various synthetic rain and snow degradations, successfully detecting objects despite heavy streaks, dense particles, and visibility loss.

## 5 Conclusion

In this paper, we present C2FXNet, a coarse-to-fine scene expert network for unified object detection across adverse weather. By integrating the Multi-step Reasoning Router, Fine Scene Refinement, and Scene-aware Mixture-of-Experts, our model establishes a coherent text-guided adaptation process that spans coarse recurrent scene reasoning, fine-grained feature modulation, and dynamic expert fusion for robust all-weather detection. Extensive experiments further demonstrate consistently superior accuracy and cross-domain generalization compared with existing methods. However, C2FXNet is trained in a fully supervised manner with limited categories and weather types. Future work will extend it toward open-world all-weather detection that adapts to unseen conditions and novel object categories.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grants 62462017, 82502451, 82272075, and 62502130, in part by the Natural Science Foundation of Guangxi Province under Grant 2025GXNSFBA069390, in part by the Guangxi Key Research and Development Program under Grant AB2401008, and in part by the Hainan Provincial Natural Science Foundation under Grant 826QN0577.

## References

[1] Fatmah AlHindaassi, Mohammed Talha Alam, and Fakhri Karray. 2025. ADAM-Dehaze: Adaptive Density-Aware Multi-Stage Dehazing for Improved Object Detection in Foggy Conditions. In 2025 IEEE International Conference on Systems, Man, and Cybernetics. 4370–4376.

[2] Kishore Kumar Anguchamy and Venketesh Palanisamy. 2025. Real-time ob ject detection using improvised YOLOv4 and feature mapping technique for autonomous driving. Expert Systems with Applications 280 (2025), 127452.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2.5-VL Technica Report. arXiv:2502.13923 (2025).

[4] Yun-Hao Cao, Kaixiang Ji, Ziyuan Huang, et al. 2024. Towards Better Vision Inspired Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13537–13547.

[5] Yonglin Chen, Binzhi Fan, Nan Liu, Yalong Yang, and Jinhui Tang. 2026. EEnvA Mamba: Efective and Environtology-aware Adaptive Mamba for Road Object Detection in Adverse Weather Scenes. Pattern Recognition 175 (2026), 113127.

[6] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. 2024. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 1280–1297.

[7] Tianle Fang, Zhenbing Liu, Yutao Tang, Yingxin Huang, Haoxiang Lu, and Chuangtao Zheng. 2025. RDFNet: Real-time Object Detection Framework for Foggy Scenes. In 2025 IEEE International Conference on Multimedia and Expo. IEEE, 1–6.

[8] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and eficient sparsity. Journal ofMachine Learning Research 23, 120 (2022), 1–39.

[9] Shenghao Fu, Qize Yang, Qijie Mo, Junkai Yan, Xihan Wei, Jingke Meng, Xiaohua Xie, and Wei-Shi Zheng. 2025. Llmdet: Learning strong open-vocabulary object detectors under the supervision of large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14987–14997.

[10] Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. 2020. Zero-reference deep curve estimation for lowlight image enhancement. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1780–1789.

[11] Shih-Chia Huang, Trung-Hieu Le, and Da-Wei Jaw. 2020. DSNet: Joint semantic learning for object detection in inclement weather conditions. IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 8 (2020), 2623–2633.

[12] Glenn Jocher and Jing Qiu. 2024. Ultralytics YOLO11. https://github.com/ ultralytics/ultralytics

[13] Sanket Kalwar, Dhruv Patel, Aakash Aanegola, Krishna Reddy Konda, Sourav Garg, and K Madhava Krishna. 2023. GDIP: gated diferentiable image processing for object detection in adverse conditions. In Proceedings ofthe IEEE International Conference on Robotics and Automation. 7083–7089.

[14] Mengqi Lei, Siqi Li, Yihong Wu, Han Hu, You Zhou, Xinhu Zheng, Guiguang Ding, Shaoyi Du, Zongze Wu, and Yue Gao. 2025. YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception. arXiv:2506.17733 (2025).

[15] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2021. Gshard: Scaling giant models with conditional computation and automatic sharding. In Proceedings ofthe International Conference on Learning Representations.

[16] Boyi Li, Wenqi Ren, Dengpan Fu, Dacheng Tao, Dan Feng, Wenjun Zeng, and Zhangyang Wang. 2018. Benchmarking single-image dehazing and beyond. IEEE Transactions on Image Processing 28, 1 (2018), 492–505.

[17] Chengyang Li, Heng Zhou, Yang Liu, Caidong Yang, Yongqiang Xie, Zhongbo Li, and Liping Zhu. 2023. Detection-friendly dehazing: Object detection in real-world hazy scenes. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 7 (2023), 8284–8295.

[18] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. 2019. VisualBERT: A Simple and Performant Baseline for Vision and Language. arXiv:1908.03557.

[19] Yanqi Li, Jianwei Niu, and Tao Ren. 2025. Benefit From Seen: Enhancing Open-Vocabulary Object Detection by Bridging Visual and Textual Co-Occurrence

Knowledge. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 22110–22119.

[20] Chen Liang, Shaobing Gao, Liangtian He, and Yiguang Liu. 2025. Biological Vision Inspired Context-awareness Network for Various Non-generic Object Detection. IEEE Transactions on Circuits and Systems for Video Technology 35, 7 (2025), 6726–6739.

[21] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Proceedings ofthe European Conference on Computer Vision. 740–755.

[22] Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. 2018. Path aggregation network for instance segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8759–8768.

[23] Weifeng Liu, Jian Pang, Bingfeng Zhang, Jin Wang, Baodi Liu, and Dapeng Tao. 2025. See Degraded Objects: A Physics-Guided Approach for Object Detection in Adverse Environments. IEEE Transactions on Image Processing 34 (2025), 2198–2212.

[24] Wenyu Liu, Gaofeng Ren, Runsheng Yu, Shi Guo, Jianke Zhu, and Lei Zhang. 2022. Image-adaptive YOLO for object detection in adverse weather conditions. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 36. 1792–1800.

[25] Zhenbing Liu, Tianle Fang, Haoxiang Lu, Weidong Zhang, and Rushi Lan. 2025. MASFNet: Multiscale Adaptive Sampling Fusion Network for Object Detection in Adverse Weather. IEEE Transactions on Geoscience and Remote Sensing 63 (2025), 1–15.

[26] Yuen Peng Loh and Chee Seng Chan. 2019. Getting to know low-light images with the exclusively dark dataset. Computer Vision and Image Understanding 178 (2019), 30–42.

[27] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. 2019. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks. In Advances in Neural Information Processing Systems, Vol. 32.

[28] Wei Miao, Jiangrong Shen, Qi Xu, Timo Hamalainen, Yi Xu, and Fengyu Cong. 2025. SpikingYOLOX: Improved YOLOX Object Detection with Fast Fourier Convolution and Spiking Neural Networks. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 1465–1473.

[29] Aref Miri Rekavandi, Lian Xu, Farid Boussaid, Abd-Krim Seghouane, Stephen Hoefs, and Mohammed Bennamoun. 2025. A Guide to Image- and Video-Based Small Object Detection Using Deep Learning: Case Study of Maritime Surveil lance. IEEE Transactions on Intelligent Transportation Systems 26, 3 (2025), 2851– 2879.

[30] James Oldfield, Markos Georgopoulos, Grigorios Chrysos, Christos Tzelepis, Yannis Panagakis, Mihalis Nicolaou, Jiankang Deng, and Ioannis Patras. 2024. Multilinear mixture of experts: Scalable expert specialization through factorization. Advances in Neural Information Processing Systems 37 (2024), 53022–53063.

[31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamila Mishkin, et al. 2021. Learning Transferable Visual Models from Natural Language Supervision. In Proceedings of the International Conference on Machine Learning. 8748–8763.

[32] Sara Sabour, Nicholas Frosst, and Geofrey E Hinton. 2017. Dynamic routing between capsules. In Advances in Neural Information Processing Systems, Vol. 30.

[33] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv:1701.06538 (2017).

[34] Zeyi Sun, Ye Fang, Tong Wu, Pan Zhang, Yuhang Zang, Shu Kong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. 2024. Alpha-clip: A clip model focusing on wherever you want. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13019–13029.

[35] Yunjie Tian, Qixiang Ye, and David Doermann. 2025. Yolov12: Attention-centric real-time object detectors. Advances in neural information processing systems 38 (2025), 78433–78457.

[36] Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, et al. 2024. Yolov10: Real-time end-to-end object detection. In Advances in Neural Information Processing Systems, Vol. 37. 107984–108011.

[37] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. 2021. Scaledyolov4: Scaling cross stage partial network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13029–13038.

[38] Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. 2023. YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7464–7475.

[39] Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao. 2024. Yolov9: Learning what you want to learn using programmable gradient information. In Proceedings ofthe European Conference on Computer Vision. 1–21.

[40] Jian Wang, Fan Li, and Lijun He. 2025. A Unified Framework for Adversarial Patch Attacks Against Visual 3D Object Detection in Autonomous Driving. IEEE Transactions on Circuits and Systemsfor Video Technology 35, 5 (2025), 4949–4962.

[41] Yongzhen Wang, Xuefeng Yan, Kaiwen Zhang, Lina Gong, Haoran Xie, Fu Lee Wang, and Mingqiang Wei. 2022. TogetherNet: Bridging Image Restoration and Object Detection Together via Dynamic Enhancement Learning. Computer

Graphics Forum 41, 7 (2022), 465–476.

[42] Xing Xi, Yangyang Huang, Ronghua Luo, and Yu Qiu. 2025. OW-OVD: Unified Open World and Open Vocabulary Object Detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 25454–25464.

[43] Yuqi Yang, Peng-Tao Jiang, Qibin Hou, Hao Zhang, Jinwei Chen, and Bo Li. 2024. Multi-task dense prediction via mixture of low-rank experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 27927–27937.

[44] Heng Yin, Yuqiang Ren, Ke Yan, Shouhong Ding, and Yongtao Hao. 2025. ROD-MLLM: Towards More Reliable Object Detection in Multimodal Large Language Models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14358–14368.

[45] Heng Zhang, Qiuyu Zhao, Linyu Zheng, Hao Zeng, Zhiwei Ge, Tianhao Li, and Sulong Xu. 2024. Exploring region-word alignment in built-in detector for open-vocabulary object detection. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16975–16984.

[46] Rui Zhao, Huibin Yan, and Shuoyao Wang. 2024. Revisiting Domain-Adaptive Object Detection in Adverse Weather by the Generation and Composition of High-Quality Pseudo-labels. In Proceedings of the European Conference on Computer Vision. 270–287.

[47] Shizhen Zhao, Jiahui Liu, Xin Wen, Haoru Tan, and Xiaojuan Qi. 2025. Equipping Vision Foundation Model with Mixture of Experts for Out-of-Distribution Detection. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 1751–1761.

[48] Fujin Zhong, Wenxin Shen, Hong Yu, Guoyin Wang, and Jun Hu. 2024. Dehazing & Reasoning YOLO: Prior knowledge-guided network for object detection in foggy weather. Pattern Recognition 156 (2024), 110756.

[49] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. 2017. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 2223–2232.