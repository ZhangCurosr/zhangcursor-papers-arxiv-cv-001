# Asymmetric Cross-Modal Fine-Grained Visual Categorization: ACF-Net and the BirdPro Benchmark

Bohan Deng<sup>1</sup>, Shuo Ye<sup>1\*</sup>, and Zitong Yu<sup>1</sup>

Great Bay University

bohandeng102647@gmail.com

shuoye.ke@gmail.com

zitong.yu@ieee.org

Corresponding author: Shuo Ye

Abstract. Audio-visual cross-modal Fine-Grained Visual Categorization (FGVC) aims to identify fine-grained categories by jointly leveraging visual and auditory information. However, FGVC under asymmetric cross-modal scenarios has received limited attention, where paired video and audio are not strictly synchronized and may not even correspond to the same individual or moment. Such weak and ambiguous cross-modal correspondence poses substantial challenges to efective representation learning and modality alignment. To address these issues, we propose ACF-Net, a novel optical flow-guided framework for asymmetric audio-visual fine-grained learning. ACF-Net consists of two key modules: Optical Flow-Guided Motion (OFGM) and Asymmetric Cross-Modal Adaptive Fusion (ACAF). OFGM captures motion-sensitive visual cues and suppresses irrelevant background interference, thereby enhancing discriminative dynamic representations in videos. ACAF estimates modality reliability under weakly matched audio-video pairs and performs uncertainty-aware adaptive fusion to improve category-level recognition robustness. To support research on asymmetric cross-modal FGVC, we further construct BirdPro, a new bird-oriented audio-visual benchmark, since existing datasets often lack large-scale category-level audio-video associations under non-strict temporal and instance correspondence. BirdPro contains 1,919 audio recordings and 11,965 videos covering 194 bird species. Extensive experiments show that ACF-Net achieves the best results compared with representative baseline methods, outperforming the strongest baselines by 2.97% and 1.92% in the fused and mismatched settings, respectively.

The code repository is available at wxdbh/BirdPro-and-Fine-grained-Recognition.

Keywords: Fine-grained audio-visual analysis · Cross-modal alignment · Multimodal fusion · Optical flow guidance.

## 1 Introduction

Fine-grained Visual Categorization (FGVC) aims to distinguish highly similar subcategories or instances within the same semantic superclass [24,32]. Its core objective is not merely to determine whether an object belongs to a general category, but to identify the specific fine-grained type of the object [34]. As a result, fine-grained visual analysis relies heavily on weak discriminative cues, such as internal local structures, subtle textures, geometric proportions, and part composition relationships, to achieve precise diferentiation [8,33]. Related studies have been widely applied in various domains, including biological species identification, intelligent manufacturing, surveillance and security, cross-modal content retrieval, and medical image–assisted diagnosis [18,19], playing an important role in improving perceptual accuracy under complex environments [35].

In real-world FGVC, subtle visual cues are easily degraded by pose variation, occlusion, background clutter, illumination changes, or limited image quality. Similar dificulties have also been observed in robust recognition and camouflaged/salient object perception, where category confusion, background camouflage, underwater degradation, and temporal instability motivate stronger discriminative and alignment mechanisms [31,12,29,21]. Although video-based FGVC exploits appearance and motion, it is still limited by visual observability: discriminative parts may be occluded, key behaviors may be absent, and visually similar species can remain ambiguous. Additional modalities can provide complementary cues, but cross-modal heterogeneity in representation, information density, and noise distribution makes alignment and discriminative modeling challenging [10,30]. Existing cross-modal FGVC methods often establish correspondences via vision-language adaptation, prompt learning, adapter tuning, or textual guidance, from category-level matching to attribute-, part-, or region-level alignment [10]. However, they usually assume strict correspondence across modalities, such as the same instance, temporal segment, or synchronized context [24,6,8], as shown in Fig. 1. This assumption is often violated in realworld audio-visual data, where audio and visual signals may share category-level semantics but difer in instance, time, or context. For example, bird calls can indicate species identity even when the observed bird is silent or the sound comes from another instance. Therefore, cross-modal FGVC should exploit semantically related yet non-strictly aligned cues from asymmetric modalities. Inspired by expert cognition, where bird experts associate calls with appearance and behavior, we argue that learning such asymmetric semantic correlations is essential for robust cross-modal FGVC.

Therefore, we propose ACF-Net, an optical-flow-guided framework for asymmetric audio-visual cross-modal FGVC, to address the weak correspondence and heterogeneous characteristics between audio and visual modalities under asymmetric settings [24,6,8]. ACF-Net consists of two key modules: Optical Flow-Guided Motion (OFGM) and Asymmetric Cross-Modal Adaptive Fusion (ACAF). Specifically, OFGM enhances motion-sensitive visual representations by emphasizing category-relevant dynamic patterns while suppressing static background distractions. ACAF further performs uncertainty-aware adaptive fusion by estimating the reliability of each modality, allowing the model to rely more on the more informative modality under weakly associated audio-video pairs [10,30]. This design improves the robustness of category-level recognition in asymmetric settings. Furthermore, we construct BirdPro, a new bird-oriented asymmetric cross-modal dataset, providing a new benchmark for asymmetric audio-visual FGVC. The main contributions are summarized as follows:

![](images/41c0bda6c8b91d65bc9558fd03320cac5f10fa459a9eb134a4c30d3672b44557.jpg)  
Fig. 1: Motivation of asymmetric audio-visual FGVC. (a) Unimodal video FGVC [13] sufers from fragile visual cues under pose variation, occlusion, or background clutter. (b) Synchronized audio-visual FGVC [10] leverages complementary acoustic cues under strict temporal and instance-level correspondence. (c) Asymmetric audio-visual FGVC only assumes category-level semantic consistency, where audio and video may difer in instance, time, or context.

– We revisit cross-modal FGVC under asymmetric audio-visual scenarios and show that semantically correlated discriminative cues can still be jointly learned even without strict instance-, time-, or context-level correspondence.

– We propose an optical-flow-guided asymmetric audio-visual FGVC framework, which enhances motion-sensitive visual cues and adaptively aligns heterogeneous audio-visual representations for robust semantic fusion.

We construct BirdPro, a bird-oriented asymmetric audio-visual FGVC dataset, and conduct extensive experiments to demonstrate that the proposed method achieves state-of-the-art performance.

## 2 Related Work

Fine-grained Visual Categorization (FGVC). FGVC aims to distinguish visually similar categories by capturing subtle discriminative cues [17]. Early studies mainly relied on part-based modeling and region-level analysis, while recent convolutional and transformer-based methods improve fine-grained representation learning through multi-scale modeling, self-attention, and structured feature aggregation [28,13,7]. Related dense prediction and robust recognition studies further show that suppressing distractors and preserving weak target cues are crucial under category confusion, camouflage, and complex spatiotemporal backgrounds [31,12,29,21]. In addition, class-incremental learning methods explore predictive prompting, embedding distillation, and task-oriented generation to preserve discriminative knowledge when new categories arrive [14,15]. Beyond image-based recognition, video-based FGVC further exploits temporal dynamics, and auditory fine-grained recognition shows that sound can provide complementary semantic cues for tasks such as bird species recognition and bioacoustic monitoring. However, most existing FGVC methods are developed under unimodal settings or assume reliable cross-modal correspondence. In realworld audio-visual scenarios, visual and auditory observations can be asymmetric, asynchronous, or partially missing, making it necessary to exploit complementary cues while suppressing noisy or weakly related modal information.

Cross-modal Alignment. Cross-modal alignment seeks to bridge heterogeneous modalities by learning semantically consistent representations [10,30]. Representative methods include shared embedding learning, contrastive learning, cross-modal attention, and collaborative encoding, which have shown efectiveness in cross-modal retrieval, multimodal classification, and vision-language understanding [12]. For FGVC, however, alignment is more challenging because recognition often depends on sparse local cues, fine-grained attributes, and weakly salient patterns rather than global semantics alone. Recent studies therefore explore attribute- or token-level alignment to improve fine-grained discrimination [30]. Nevertheless, most alignment methods still assume that informative cues from diferent modalities are synchronously observed and semantically consistent. This assumption is often violated in asymmetric audio-visual FGVC, where modalities may provide unequal, incomplete, or weakly correlated evidence [24,6,8,25]. Thus, the key challenge is not only reducing modality gaps, but also determining which cross-modal information is reliable and beneficial for fine-grained recognition [11].

Auxiliary Information-Guided Learning. Auxiliary information-guided learning improves representation learning by introducing additional priors or side information to guide the primary modality. Unlike conventional multimodal learning, auxiliary information is often used to constrain feature extraction or representation optimization, rather than directly serving as an independent prediction source. Typical guidance signals include semantic cues, such as textual descriptions or attribute labels [30]; structural or motion cues, such as pose, segmentation masks, and optical flow [2]; and weakly coupled cross-modal cues that provide auxiliary constraints without requiring strict alignment [4]. Although these methods have shown promising results, many of them still rely on reliable correspondence between auxiliary information and the target modality [3]. In asymmetric audio-visual FGVC, such correspondence can be weak, noisy, or temporally inconsistent. Therefore, how to exploit latent cross-modal correlations as efective guidance without enforcing strict correspondence remains an important problem.

## 3 Proposed Method

The framework of ACF-Net is illustrated in Fig. 2. Given an audio-visual bird sample, we denote the video clip and audio signal as $\mathcal { T } = \{ I _ { t } \} _ { t = 1 } ^ { T }$ and A, respectively, with a shared category label $y \in \{ 1 , \ldots , K \}$ . The two modalities are only category-aligned and may come from diferent temporal segments, recording conditions, or instances of the same bird species. Let $E _ { v } ( \cdot )$ and $E _ { a } ( \cdot )$ denote the visual and audio encoders, respectively [5,30,10]. ACF-Net first applies OFGM to inject optical-flow-guided motion cues into RGB frames, producing a motionaware visual representation $\tilde { \mathbf { z } } _ { v } .$ In parallel, the audio encoder extracts ${ \bf z } _ { a }$ from A. Based on these representations, ACAF estimates modality reliability from prediction uncertainty and adaptively fuses visual and audio logits for final classification.

![](images/b014bbc0107e15e1125b0070cd26eff9cbe7b07b4a4383290a4757a5e50cb016.jpg)  
Fig. 2: Framework of ACF-Net. OFGM enhances RGB frames with optical-flowguided motion cues, while ACAF adaptively fuses multimodal features for final classification.

## 3.1 Optical Flow-Guided Motion (OFGM)

In asymmetric audio-visual fine-grained scenarios, visual clips provide dense temporal observations but often lack explicitly structured motion cues. Fine-grained bird recognition may rely on subtle dynamics, such as posture changes, beak movements, and wing flapping, which can be overwhelmed by static appearance or background variations. To address this issue, OFGM introduces optical flow as a motion prior to emphasize motion-sensitive visual regions [26]. Given an input video clip $\mathcal { T } = \{ I _ { t } \} _ { t = 1 } ^ { T }$ , we first compute optical flow maps between consecutive frames:

$$
F _ { t } = \mathrm { S t r e a m F l o w } ( I _ { t } , I _ { t + 1 } ) , \quad t = 1 , \ldots , T - 1 .\tag{1}
$$

For each spatial position $( x , y )$ , the optical flow vector is written as:

$$
F _ { t } ( x , y ) = \big ( u _ { t } ( x , y ) , v _ { t } ( x , y ) \big ) ,\tag{2}
$$

where $\boldsymbol { u } _ { t } ( \boldsymbol { x } , \boldsymbol { y } )$ and $v _ { t } ( x , y )$ denote horizontal and vertical displacement components, respectively. The motion intensity is computed by the flow magnitude:

$$
S _ { t } ( x , y ) = \sqrt { u _ { t } ( x , y ) ^ { 2 } + v _ { t } ( x , y ) ^ { 2 } } .\tag{3}
$$

We then normalize the magnitude map to obtain a motion-aware spatial mask:

$$
M _ { t } = \operatorname { N o r m } ( S _ { t } ) ,\tag{4}
$$

where Norm(·) rescales the flow magnitude into [0, 1]. The motion-enhanced RGB frame is generated by:

$$
\tilde { I } _ { t } = I _ { t } \odot ( \alpha + ( 1 - \alpha ) M _ { t } ) ,\tag{5}
$$

where ⊙ denotes element-wise multiplication and α controls the minimum retained appearance intensity. When $M _ { t } ( x , y )$ is large, the corresponding visual region is preserved or emphasized; when $M _ { t } ( x , y )$ is small, the region is weakened but not completely removed. This design preserves appearance information while injecting motion-sensitive guidance.

The motion-enhanced video clip is denoted as:

$$
\tilde { \mathcal { Z } } = \{ \tilde { I } _ { t } \} _ { t = 1 } ^ { T } .\tag{6}
$$

For the last frame, we reuse the preceding motion mask, i.e., $M _ { T } = M _ { T - 1 }$ . The visual representation is then extracted by the visual encoder:

$$
\tilde { \mathbf { z } } _ { v } = E _ { v } ( \tilde { \mathcal { T } } ) .\tag{7}
$$

Unlike direct RGB-flow feature concatenation, this mask-based formulation uses optical flow only as a spatial prior, avoiding an additional motion encoder while guiding the visual encoder toward fine-grained foreground dynamics.

## 3.2 Asymmetric Cross-Modal Adaptive Fusion (ACAF)

Fixed-weight fusion is often suboptimal for asymmetric audio-visual fine-grained recognition [24,6,8], since modality reliability varies across samples. Audio signals may contain clear species-specific vocal patterns or be noisy and weakly related, while visual observations may sufer from occlusion, background clutter, or limited motion. Therefore, ACAF performs sample-adaptive fusion according to modality uncertainty.

Given the motion-enhanced visual representation $\tilde { \mathbf { z } } _ { v }$ and the audio representation ${ \bf z } _ { a } .$ , we first obtain modality-specific logits:

$$
\begin{array} { r } { \mathbf o _ { v } = C _ { v } ( \tilde { \mathbf z } _ { v } ) , \quad \mathbf o _ { a } = C _ { a } ( \mathbf z _ { a } ) , } \end{array}\tag{8}
$$

where $C _ { v } ( \cdot )$ and $C _ { a } ( \cdot )$ denote the visual and audio classifiers, respectively.

We estimate modality uncertainty from the prediction distributions:

$$
\mathbf { p } _ { v } = \mathrm { S o f t m a x } ( \mathbf { o } _ { v } ) , \quad \mathbf { p } _ { a } = \mathrm { S o f t m a x } ( \mathbf { o } _ { a } ) .\tag{9}
$$

The entropy-based uncertainty of modality $m \in \{ v , a \}$ is computed as:

$$
u _ { m } = - \sum _ { k = 1 } ^ { K } p _ { m } ^ { k } \log p _ { m } ^ { k } .\tag{10}
$$

A lower entropy indicates a more confident and reliable modality, while a higher entropy indicates a more uncertain prediction.

We convert uncertainty into reliability scores:

$$
\rho _ { v } = \exp ( - u _ { v } / \tau ) , \quad \rho _ { a } = \beta \exp ( - u _ { a } / \tau ) ,\tag{11}
$$

where τ is a temperature parameter and $\beta > 0$ is a learnable audio scaling factor. The adaptive fusion weights are obtained by normalization:

$$
w _ { v } = \frac { \rho _ { v } } { \rho _ { v } + \rho _ { a } } , \quad w _ { a } = \frac { \rho _ { a } } { \rho _ { v } + \rho _ { a } } .\tag{12}
$$

The fused logits are then computed as:

$$
\mathbf { o } _ { v a } = w _ { v } \mathbf { o } _ { v } + w _ { a } \mathbf { o } _ { a } .\tag{13}
$$

This uncertainty-aware design allows ACAF to assign higher weights to the more reliable modality for each sample. The training objective combines fused and modality-specific classification losses:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { f u s e d } } + \lambda _ { v } \mathcal { L } _ { v } + \lambda _ { a } \mathcal { L } _ { a } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r o t o } } ^ { a } + \lambda _ { d } \mathcal { L } _ { \mathrm { d e c } } . } \end{array}\tag{14}
$$

Here, $\mathcal { L } _ { \mathrm { f u s e d } } , \mathcal { L } _ { v }$ , and $\mathcal { L } _ { a }$ are cross-entropy losses for fused, visual, and audio predictions, respectively. $\mathcal { L } _ { \mathrm { p r o t o } } ^ { a }$ denotes the audio prototype regularization loss, and ${ \mathcal { L } } _ { \mathrm { d e c } }$ denotes the concept decoupling regularization. The coeficients $\lambda _ { v } , \lambda _ { a } , \lambda _ { p } ,$ and $\lambda _ { d }$ control the contributions of the visual classification, audio classification, audio prototype, and decoupling losses, respectively.

$$
\mathcal { L } _ { \mathrm { p r o t o } } ^ { a } = \mathrm { C E } \left( \frac { \mathrm { N o r m } ( \mathbf { z } _ { a } ) \mathrm { N o r m } ( \mathbf { P } _ { a } ) ^ { \top } } { \tau _ { p } } , y \right) ,\tag{15}
$$

where $\tau _ { p }$ is the prototype temperature and $\mathbf { P } _ { a } \in \mathbb { R } ^ { K \times d }$ denotes the audio prototype matrix.

$$
\mathbf { P } _ { a } ^ { ( k ) } \gets \mu \mathbf { P } _ { a } ^ { ( k ) } + ( 1 - \mu ) \operatorname { N o r m } \left( \frac { 1 } { | \mathcal { B } _ { k } | } \sum _ { i \in \mathcal { B } _ { k } } \mathbf { z } _ { a , i } \right) ,\tag{16}
$$

where $\mathbf { P } _ { a } ^ { ( k ) }$ is the audio prototype of class k, $\mu$ is the EMA momentum, and $\boldsymbol { B } _ { k }$ denotes the set of samples belonging to class k in the current mini-batch.

$$
\mathcal { L } _ { \mathrm { d e c } } ^ { m } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \frac { 1 } { Q _ { m } ( Q _ { m } - 1 ) } \sum _ { \stackrel { r , s = 1 } { r \ne s } } ^ { Q _ { m } } \left( \hat { \mathbf { c } } _ { i , r } ^ { m \top } \hat { \mathbf { c } } _ { i , s } ^ { m } \right) ^ { 2 } , \quad m \in \{ v , a \} .\tag{17}
$$

Here, B denotes the mini-batch size, $Q _ { m }$ is the number of concept queries for modality m, and $\hat { \mathbf { c } } _ { i , r } ^ { m }$ denotes the L2-normalized feature of the r-th concept query for the i-th sample in modality m.

$$
{ \mathcal { L } } _ { \mathrm { d e c } } = { \mathcal { L } } _ { \mathrm { d e c } } ^ { v } + { \mathcal { L } } _ { \mathrm { d e c } } ^ { a } .\tag{18}
$$

## 4 Experiments

## 4.1 BirdPro Dataset

We adopt the species-level semantic annotations provided by the CUB dataset [27]   
as a unified label space, and use the corresponding scientific names as query key  
words to systematically retrieve videos and audio data from publicly available   
online resources. Based on this process, we construct a multi-modal fine-grained   
bird dataset, ensuring semantic consistency and comparability across diferent   
modalities. The dataset statistics are shown in Table 1.   
Table 1: Statistics of Bird-During data collection, candidate videos are   
Pro. “M” denotes the modal-first manually screened to ensure that the tar  
get species is visually identifiable and occu- ity type.   
pies a dominant region in a suficient number M Classes Train Test   
of frames. From the selected videos, represen-Video 194 10,054 1,911   
tative key frames are extracted to cover di-Audio 194 1,531 388   
verse viewpoints, poses, and environmental con-Pair 194 6,022 470   
ditions, thereby preserving fine-grained visual

details that are critical for species discrimination. In parallel, audio samples corresponding to the same species are collected from independent sources, with preference given to segments featuring low background noise and clear acoustic structures. The resulting dataset provides a more challenging and realistic benchmark for studying cross-modal FGVC and robust representation learning under complex environmental conditions.

## 4.2 Implementation Details

Our framework adopts a two-stream audio-visual architecture. For the visual branch, we use ViT-B/16 with an input resolution of 448, and each video is represented by 6 sampled frames. For the audio branch, we follow the Image-Bind [10]-style audio representation and use log-mel spectrogram inputs with a target length of 204. To enhance motion-sensitive visual representation, Stream-Flow [26] is used to generate compact 28×28 optical flow maps as motion priors, which suppress background-dominant motion while preserving eficient storage and computation. For multimodal fusion, the feature dimension is set to 512, and the dropout rate is set to 0.1. During fine-tuning, the last 6 blocks of the audio encoder and the last 2 layers of the visual encoder are unfrozen, while the remaining parameters are kept fixed to preserve pretrained representations. Unless otherwise specified, the same configuration is used for all ablation and comparison experiments. We evaluate the proposed method under both unimodal and multimodal settings.

To further evaluate robustness under asymmetric multimodal correspondence, we conduct an audio-visual mismatch experiment. Specifically, we apply a class-level derangement to the training split, where audio samples from class c are reassigned to another class $\pi ( c )$ with $\pi ( c ) \neq c ,$ while the visual samples remain unchanged. The validation split is kept clean. This setting tests whether

![](images/2fc76e8298daa41884eba689890b7c6d2095d3e0380af330c67f98d1d69cf543.jpg)  
Fig. 3: Pairwise sensitivity analysis of loss weights.

Table 2: Sensitivity analysis of key hyperparameters.  
(a) Frames and audio clips
<table><tr><td rowspan="2">Frames</td><td rowspan="2">Clips</td><td rowspan="2">10</td><td rowspan="2">20</td><td rowspan="2">30</td></tr><tr><td></td></tr><tr><td>2</td><td></td><td>83.40</td><td>83.19</td><td>84.25</td></tr><tr><td>4</td><td></td><td>85.75</td><td>84.68</td><td>86.17</td></tr><tr><td>6</td><td></td><td>85.75</td><td>85.53</td><td>87.23</td></tr><tr><td>8</td><td></td><td>86.81</td><td>84.47</td><td>86.38</td></tr></table>

(b) Impact of query numbers (v<sub>N</sub>, a<sub>N</sub>)
<table><tr><td>aN</td><td rowspan="3">2</td><td rowspan="3">4</td><td rowspan="3">6</td><td rowspan="3"></td><td rowspan="3">8</td></tr><tr><td>vN</td></tr><tr><td>3</td></tr><tr><td></td><td>87.02</td><td>84.04 84.68</td><td></td><td>84.68</td><td>87.02</td></tr><tr><td>5</td><td>85.75</td><td></td><td></td><td>86.38</td><td>84.89</td></tr><tr><td>7 9</td><td></td><td>84.68</td><td>85.11 85.11</td><td>85.11 85.11</td><td>84.68 86.60</td></tr><tr><td></td><td></td><td>84.89</td><td></td><td></td><td></td></tr></table>

each fusion strategy can tolerate corrupted audio-visual correspondence during training and still perform reliable audio-visual recognition during evaluation.

## 4.3 Model Configuration

Loss-weight sensitivity. As shown in Fig. 3, a moderate weight for the audio prototype loss generally improves performance, suggesting that prototype regularization enhances the discriminability of audio representations. For the decoupling loss, overly large weights tend to degrade accuracy, implying that excessive concept separation may weaken useful shared information. The best performance is achieved with a balanced combination of prototype and decoupling losses, demonstrating that these regularization terms are complementary when properly weighted.

Temporal sampling and concept queries. We further investigate the efects of the number of sampled visual frames, audio clips, and modality-specific concept queries. As shown in Table 2a, using too few visual frames, e.g., 2 frames, leads to inferior performance, since limited visual observations are insuficient to capture fine-grained motion and pose variations. Increasing the number of frames generally improves the recognition accuracy. However, further increasing the visual frames does not consistently bring additional gains, suggesting that excessive frames may introduce redundant or less discriminative information. Therefore, we adopt 6 visual frames and 30 audio clips as the default temporal sampling configuration. For the number of visual and audio queries, the results in Table 2 show that simply increasing the number of queries does not always

Table 3: Comparison results. <sup>∗</sup> denotes our implemented baseline variant based on the cited method. The best results are highlighted in bold.
<table><tr><td>Method</td><td>M</td><td>Audio</td><td>Video</td><td>Fusion</td><td>Mismatch</td></tr><tr><td>FG-CLIP [30]</td><td> $\operatorname { V } _ { - \mathrm { o n l y } }$ </td><td></td><td>70.30</td><td></td><td>一</td></tr><tr><td>ACF (ours)</td><td> $\mathrm { \Delta V - o n i \mathrm { \bar { y } } }$ </td><td></td><td>71.06</td><td></td><td></td></tr><tr><td> $\mathrm { I m a g e B i n d \ [ 1 0 ] }$   $\mathrm { A C F ~ ( o u r s ) }$ </td><td> $_ { \mathrm { A - o n l y } }$ </td><td>47.42 57.45</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { C o C ^ { * } \ [ 1 6 ] }$ </td><td> $_ { \mathrm { A - o n l y } }$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { M D L ^ { * } } ^ { * } [ 2 \dot { 3 } ]$ </td><td> $\mathrm { ~ A ~ } + \mathrm { ~ V ~ }$ </td><td>45.74 47.73</td><td>71.06</td><td>80.43</td><td>70.21</td></tr><tr><td> $\mathrm { W e C ^ { * } \ \tilde { \ n b } \tilde { \ l } }$ </td><td> $\mathrm { ~ A ~ } + \mathrm { ~ V ~ }$ </td><td></td><td>71.92</td><td>73.40</td><td>70.43</td></tr><tr><td></td><td> $\mathrm { ~ A ~ } + \mathrm { ~ V ~ }$ </td><td>42.34</td><td>71.06</td><td>82.34</td><td>69.79</td></tr><tr><td> $\mathrm { U A F ^ { * } } \ \mathrm { \bar { [ 2 5 ] } }$ </td><td> $\mathrm { ~ A ~ } \overset { \cdot } { + } \mathrm { ~ V ~ }$ </td><td>47.87</td><td>71.91</td><td>83.19</td><td>72.13</td></tr><tr><td> $\mathrm { U D F ^ { * } \left. 2 5 \right. }$ </td><td> $\mathrm { ~ A ~ } \dot { + } \mathrm { ~ V ~ }$ </td><td>49.36</td><td>70.21</td><td>84.26</td><td>72.34</td></tr><tr><td> $\mathrm { D M I ^ { * } \partial [ 2 0 ] }$ </td><td> $\mathrm { ~ A ~ } + \mathrm { ~ V ~ }$ </td><td>41.70</td><td>70.64 71.49</td><td>75.32 79.15</td><td>70.00</td></tr><tr><td> $\mathrm { C o S ^ { * } \ \left\lceil 2 5 \right\rceil ^ { . } }$   $\mathrm { G M U ^ { * } \ [ 1 ] }$ </td><td> $\mathrm { ~ A ~ } + \mathrm { ~ V ~ }$ </td><td>42.77 49.15</td><td>71.91</td><td>75.11</td><td>72.34</td></tr><tr><td> $\mathrm { C B P ^ { * } \ [ \dot { 9 } ] }$ </td><td> $\mathrm { ~ A ~ } + \mathrm { ~ V ~ }$ </td><td>42.98</td><td>72.34</td><td>77.66</td><td>70.64</td></tr><tr><td></td><td> $\mathrm { ~ A ~ } + \mathrm { ~ V ~ }$ </td><td></td><td>72.77</td><td>87.23</td><td>52.55</td></tr><tr><td> $\mathrm { A C F - N e t } \ \mathrm { ( o u r s ) }$ </td><td> $\mathrm { ~ A ~ } + \mathrm { ~ V ~ }$ </td><td>58.30</td><td></td><td></td><td>74.26</td></tr></table>

improve performance. A small number of visual queries can already capture discriminative fine-grained visual concepts, while excessive queries may introduce redundant concepts and increase optimization dificulty. For the audio branch, diferent query numbers lead to performance fluctuations, indicating that audio concept modeling is sensitive to the balance between representation capacity and noise suppression. Based on these observations, we choose a moderate query configuration that achieves competitive performance while avoiding unnecessary redundancy.

## 4.4 Comparison Results

Table 3 reports the comparison results across diferent modality settings.

Unimodal results. ACF-Net improves the video-only accuracy from 70.30% to 71.06% compared with FG-CLIP [30], and improves the audio-only accuracy from 47.42% to 57.45% compared with ImageBind [10]. These results indicate that the proposed framework enhances both visual and audio representations, with a more substantial gain on the weaker audio modality.

Multimodal results. Under this setting, direct fusion variants such as feature concatenation, cross-modal attention, gated fusion, and bilinear fusion do not consistently improve performance. This suggests that symmetric fusion can introduce unreliable modality interactions in asymmetric audio-visual FGVC. In contrast, uncertainty-aware methods achieve stronger results, confirming the importance of estimating modality reliability. ACF achieves the best fusion accuracy of 87.23%, outperforming the strongest baseline by 2.97%, which demonstrates its ability to exploit complementary audio-visual information while reducing the influence of unreliable audio cues.

Mismatch robustness. To further evaluate robustness under unreliable crossmodal correspondence, we conduct an audio-visual mismatch experiment. Specifically, audio samples are reassigned to semantically unrelated visual classes during training, while the validation split remains clean. This setting simulates asymmetric multimodal scenarios where the audio modality may be weak, noisy, or incorrectly paired with the visual modality. As can be seen, conventional fusion strategies sufer from clear performance degradation under mismatched audiovisual supervision. In particular, Compact Bilinear Pooling drops to 52.55%, indicating that strong multiplicative interaction is sensitive to corrupted crossmodal correspondence. Uncertainty-aware methods are more robust because they reduce the influence of unreliable modality information. ACF obtains the best accuracy of 74.26%, outperforming the strongest baseline by 1.92%, demonstrating its robustness under asymmetric multimodal conditions.

## 4.5 Ablation Study

To evaluate the efectiveness of the proposed method, we conduct comprehensive ablation studies on our benchmark dataset. Baseline. We first report the performance of a unimodal visual recognition framework as the baseline, which serves as a reference for assessing the contribution of each proposed component under the asymmetric cross-modal fine-grained setting. OFGM. We then introduce the setting w/ OFGM to evaluate the overall performance gain brought by the Optical Flow-Guided Fine-grained Motion Modeling module. ACAF. Next, we adopt the setting w/ ACAF to assess the impact of the Asymmetric Cross-Modal Adaptive Fusion module. Full model. Finally, we combine both modules in the full model to examine whether OFGM and ACAF provide complementary benefits. The experimental results are summarized in Table 4.

Efect of OFGM. As can be seen, the baseline model achieves 82.98% fusion accuracy, while introducing OFGM improves the video accuracy from 71.06% to 71.92%. This result indicates that optical-flow-guided motion cues can en-

Table 4: Ablation study results.
<table><tr><td>Method</td><td>Audio</td><td>Video</td><td>Fusion</td></tr><tr><td>Baseline</td><td>45.96</td><td>71.06</td><td>82.98</td></tr><tr><td>w/ OFGM</td><td>46.38</td><td>71.92</td><td>81.64</td></tr><tr><td>w/ ACAF</td><td>57.66</td><td>72.77</td><td>84.89</td></tr><tr><td>Ours</td><td>58.30</td><td>72.77</td><td>87.23</td></tr></table>

hance the visual branch by capturing fine-grained dynamic patterns that are dificult to identify from RGB frames alone. However, the fusion accuracy slightly decreases when only OFGM is introduced, suggesting that improving the visual representation alone does not necessarily lead to better multimodal fusion under asymmetric audio-visual conditions.

Efect of ACAF. When ACAF is adopted, the audio accuracy is significantly improved from 45.96% to 57.66%, and the fusion accuracy increases from 82.98% to 84.89%. This demonstrates that ACAF efectively alleviates the imbalance between the stronger visual modality and the weaker audio modality by adaptively estimating modality reliability.

Full model. Compared with the baseline, the full model achieves the best performance, with 58.30% audio accuracy, 72.77% video accuracy, and 87.23% fusion accuracy. These results verify that OFGM and ACAF provide complementary benefits: OFGM strengthens fine-grained motion-sensitive visual representation, while ACAF enables robust asymmetric cross-modal fusion. Their combination leads to the most efective multimodal representation and achieves the best overall recognition performance.

## 4.6 Visualization

To further analyze the efect of the proposed modules, we provide qualitative visualizations from both feature-space and spatial perspectives.

t-SNE visualization [22]. We visualize the feature distributions of baseline and ACF-Net representations for the video and audio modalities. As shown in

![](images/980439d57710623ec64638e9307eda39e1944945339726fc95081288152f31a3.jpg)  
Fig. 4: t-SNE visualization of raw and ACF-enhanced video and audio features. ACF improves the intra-class compactness of audio features while preserving the discriminative structure of video features.

Figure 4, the video modality exhibits clearer discriminative structures than the audio modality, indicating that visual inputs provide strong appearance-based cues for FGVC. After applying ACF-Net, video features form more compact and better-separated clusters, suggesting that the proposed framework further enhances visual discriminability by refining motion-aware representations. In contrast, raw audio features are more scattered and highly entangled across categories, reflecting the dificulty of learning discriminative representations from noisy and variable acoustic signals. Compared with raw audio features, ACF-Net enhanced audio features show improved intra-class compactness and a more coherent feature distribution.

Attention visualization. We further visualize the attention responses of the proposed method in Figure 5 to examine whether the model focuses on semantically meaningful image regions. The visualization shows that the proposed framework highlights the main object regions, especially the head, neck, and body, while suppressing irrelevant background areas. The corresponding patch-level attention maps also exhibit stronger responses around semantically informative patches. This suggests that our method guides the model toward discriminative visual regions and provides more interpretable evidence for the learned visual representation. These qualitative results are consistent with the quantitative improvements reported in previous sections.

## 5 Conclusion

In this paper, we investigate cross-modal FGVC under asymmetric audio-visual scenarios, where strict instance- or time-level correspondence is often unavailable. We propose ACF-Net, an optical flow-guided framework that enhances motion-sensitive visual cues and performs reliability-aware adaptive fusion under weakly matched audio-video pairs. We also construct the BirdPro dataset, a bird-oriented benchmark for asymmetric audio-visual FGVC. Extensive experiments demonstrate the efectiveness and robustness of ACF-Net, showing its potential for fine-grained recognition in realistic non-ideal cross-modal settings.

![](images/6ad0297f6c2aa0a1f74fec5247d3a39d13dc4a8ca1dda69c2e6fffa058d6e1d7.jpg)

![](images/19b60edae986c60cd65840d768c7dd82f64d7f101675aa9e951fe7115ffef150.jpg)  
Fig. 5: Attention visualization of the proposed framework. For each example, we show the input frame, attention overlay, and patch-level attention map.

## References

1. Arevalo, J., Solorio, T., Montes-y G´omez, M., Gonz´alez, F.A.: Gated multimodal units for information fusion. arXiv preprint arXiv:1702.01992 (2017)

2. Chen, S., Chen, S., Hong, Z., Shao, Y., You, X.: Dynamic semantic complementary network for zero-shot learning. IEEE Transactions on Emerging Topics in Computational Intelligence (2025)

3. Chen, S., Chen, S., Ye, S., Wang, Y., You, X.: Toward disentangled and controllable deep metric learning with human-like concept decomposition. IEEE Transactions on Neural Networks and Learning Systems (2025)

4. Chen, S., Fu, D., Chen, S., Ye, S., Hou, W., You, X.: Causal visual-semantic correlation for zero-shot learning. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 4246–4255 (2024)

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

6. Fan, Y., Xu, W., Wang, H., Wang, J., Guo, S.: Pmr: Prototypical modal rebalance for multimodal learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20029–20038 (2023)

7. Fu, J., Zheng, H., Mei, T.: Look closer to see better: Recurrent attention convolutional neural network for fine-grained image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4438–4446 (2017)

8. Fu, J., Gao, J., Bao, B.K., Xu, C.: Multimodal imbalance-aware gradient modulation for weakly-supervised audio-visual video parsing. IEEE Transactions on Circuits and Systems for Video Technology 34(6), 4843–4856 (2023)

9. Fukui, A., Park, D.H., Yang, D., Rohrbach, A., Darrell, T., Rohrbach, M.: Multimodal compact bilinear pooling for visual question answering and visual grounding. In: Proceedings of the 2016 conference on empirical methods in natural language processing. pp. 457–468 (2016)

10. Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: Imagebind: One embedding space to bind them all. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 15180– 15190 (2023)

11. Hao, C., Yu, Z., Liu, X., Wang, Y., Xie, W., Shi, J., Yue, H., Yang, J.: Distributionspecific learning for joint salient and camouflaged object detection. Available at SSRN 5089840

12. Hao, C., Yu, Z., Liu, X., Xu, J., Yue, H., Yang, J.: A simple yet efective network based on vision transformer for camouflaged object and salient object detection. IEEE Transactions on Image Processing 34, 608–622 (2025). https://doi.org/ 10.1109/TIP.2025.3528347

13. He, J., Chen, J.N., Liu, S., Kortylewski, A., Yang, C., Bai, Y., Wang, C.: Transfg: A transformer architecture for fine-grained recognition. In: Proceedings of the AAAI conference on artificial intelligence. vol. 36, pp. 852–860 (2022)

14. Huang, L., Li, X., Zhao, J., An, Z., Yang, C., Diao, B., Wang, F., Zeng, Y., Hao, Z., Xu, Y.: Preprompt: Predictive prompting for class incremental learning. In: Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) (2026)

15. Huang, L., Zeng, Y., Yang, C., An, Z., Diao, B., Xu, Y.: etag: Class-incremental learning via embedding distillation and task-oriented generation. In: Proceedings of the AAAI Conference on Artificial Intelligence (AAAI). vol. 38, pp. 12591–12599 (2024)

16. Kittler, J., Hatef, M., Duin, R.P., Matas, J.: On combining classifiers. IEEE transactions on pattern analysis and machine intelligence 20(3), 226–239 (1998)

17. Krause, J., Jin, H., Yang, J., Fei-Fei, L.: Fine-grained recognition without part annotations. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5546–5555 (2015)

18. Lin, X., Xiang, Y., Wang, Z., Cheng, K.T., Yan, Z., Yu, L.: Samct: Segment any ct allowing labor-free task-indicator prompts. IEEE Transactions on Medical Imaging 44(3), 1386–1399 (2024)

19. Lin, X., Yu, L., Cheng, K.T., Yan, Z.: Batformer: Towards boundary-aware lightweight transformer for eficient medical image segmentation. IEEE journal of biomedical and health informatics 27(7), 3501–3512 (2023)

20. Liu, S., Quan, W., Wang, C., Liu, Y., Liu, B., Yan, D.M.: Dense modality interaction network for audio-visual event localization. IEEE Transactions on Multimedia 25, 2734–2748 (2022)

21. Liu, Y., Ye, S., Hao, C., Yu, Z.: YUV20K: A complexity-driven benchmark and trajectory-aware alignment model for video camouflaged object detection. arXiv preprint arXiv:2604.09985 (2026)

22. Van der Maaten, L., Hinton, G.: Visualizing data using t-sne. Journal of machine learning research 9(11) (2008)

23. Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., Ng, A.Y., et al.: Multimodal deep learning. In: Icml. vol. 11, pp. 689–696 (2011)

24. Peng, X., Wei, Y., Deng, A., Wang, D., Hu, D.: Balanced multimodal learning via on-the-fly gradient modulation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8238–8247 (2022)

25. Subedar, M., Krishnan, R., Meyer, P.L., Tickoo, O., Huang, J.: Uncertainty-aware audiovisual activity recognition using deep bayesian variational inference. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 6301– 6310 (2019)

26. Sun, S., Liu, J., Li, H., Liu, G., Li, T.H., Gao, W.: Streamflow: streamlined multiframe optical flow estimation for video sequences. Advances in neural information processing systems 37, 9205–9228 (2024)

27. Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.: The caltech-ucsd birds-200-2011 dataset (2011)

28. Wei, X.S., Song, Y.Z., Mac Aodha, O., Wu, J., Peng, Y., Tang, J., Yang, J., Belongie, S.: Fine-grained image analysis with deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence 44(12), 8927–8948 (2021)

29. Wu, W., Ye, S., Liu, Y., He, J., Wang, Z., Yu, Z.: High-resolution underwater camouflaged object detection: GBU-UCOD dataset and topology-aware and frequencydecoupled networks. Pattern Recognition Letters (2026). https://doi.org/10. 1016/j.patrec.2026.07.007, in press

30. Xie, C., Wang, B., Kong, F., Li, J., Liang, D., Zhang, G., Leng, D., Yin, Y.: Fgclip: Fine-grained visual and textual alignment. arXiv preprint arXiv:2505.05071 (2025)

31. Ye, S., Chen, L., Li, Q., Zhang, J., Chen, C., Xia, S.: IKA<sup>2</sup>: Internal knowledge adaptive activation for robust recognition in complex scenarios. Machine Intelligence Research 23(2), 429–443 (2026). https://doi.org/10.1007/ s11633-025-1618-5

32. Ye, S., Chen, S., Wang, R., Wu, T., Khan, S., Khan, F.S., Shao, L.: Concept drift and long-tailed distribution in fine-grained visual categorization: Benchmark and method. IEEE Transactions on Pattern Analysis and Machine Intelligence pp. 1–17 (2026). https://doi.org/10.1109/TPAMI.2026.3674763

33. Ye, S., Peng, Q., Sun, W., Xu, J., Wang, Y., You, X., Cheung, Y.M.: Discriminative suprasphere embedding for fine-grained visual categorization. IEEE Transactions on Neural Networks and Learning Systems 35(4), 5092–5102 (2024). https://doi. org/10.1109/TNNLS.2022.3202534

34. Ye, S., Wang, Y., Peng, Q., You, X., Chen, C.P.: The image data and backbone in weakly supervised fine-grained visual categorization: A revisit and further thinking. IEEE Transactions on Circuits and Systems for Video Technology (2023)

35. Zhu, Y., Lyu, Y., Yu, Z., Shao, R., Zhou, K., Nie, L.: Emosym: A symbiotic framework for unified emotional understanding and generation via latent reasoning. In: Proceedings of the 33nd ACM International Conference on Multimedia (2025)