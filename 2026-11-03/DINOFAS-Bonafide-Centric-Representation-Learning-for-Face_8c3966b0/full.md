# DINOFAS: Bonafide-Centric Representation Learning for Face

Anti-Spoofing

Shakeel Muhammad Ibrahim, Muhammad Sohail Ibrahim, Muhammad Shahid Jabbar, and Shujaat Khan

Abstract—Face anti-spoofing (FAS) models often learn attackand dataset-specific cues that degrade performance under unseen acquisition conditions and presentation attacks. We present DINOFAS, a DINOv3-based framework that formulates FAS as bonafide-centric representation learning. The proposed framework combines parameter-efficient DINOv3 adaptation using low-rank adaptation (LoRA), an attention style-content disentanglement (ASCD) module for reducing domain-specific appearance variations, and a dual-branch architecture that separates prototype-based liveness learning from attack-semantic supervision. The model is jointly optimized using a bonafidecentric prototype objective, an attack-semantic classification loss applied to detached features, and orthogonality regularization. DINOFAS achieves an average HTER of 9.09% under the standard OCIM leave-one-domain-out protocol, where training is performed on three source datasets and evaluation is conducted on the held-out unseen domain. We further evaluate syntheticto-real generalization by training exclusively on SynthASpoof and testing on four unseen real-world datasets, where DINOFAS achieves the lowest average HTER of 21.41% and the highest average AUC of 86.14% among the compared methods. Evaluations on OULU-NPU and SiW-Mv2 additionally demonstrate robustness to variations in acquisition conditions, devices, and unseen presentation attacks.

Index Terms—Face anti-spoofing, Domain generalization, Bonafide-centric representation learning, Vision foundation models, Style-content disentanglement.

## I. INTRODUCTION

F <sup>ACE</sup> <sup>recognition</sup> <sup>(FR)</sup> <sup>systems</sup> <sup>have</sup> <sup>become</sup> <sup>an</sup> <sup>integral</sup>component of modern biometric authentication, enabling component of modern biometric authentication, enabling secure access to mobile devices, financial services, border control, and identity verification. Their remarkable progress over the past decade has largely been driven by advances in deep learning and large-scale training datasets [1]. Despite these advances, FR systems remain vulnerable to presentation attacks (PAs), in which an adversary attempts to impersonate a legitimate user using printed photographs, replayed videos, or three-dimensional masks [2], [3]. Successful presentation attacks can compromise the security of biometric systems, leading to identity theft, unauthorized access, and financial fraud. Consequently, face anti-spoofing (FAS) has become an indispensable component of practical biometric systems.

Deep learning-based FAS methods achieve strong performance when training and testing data originate from the same distribution. However, their performance degrades considerably under cross-dataset evaluation due to variations in acquisition devices, illumination, image quality, presentation attack instruments, and environmental conditions [2], [4]. Consequently, many methods learn dataset-specific spoof artifacts rather than intrinsic liveness characteristics, limiting their ability to generalize to unseen attacks. Moreover, collecting diverse FAS datasets remains expensive and constrained by privacy concerns, making robust representation learning increasingly important [3], [5].

Recent advances in foundation models have opened new opportunities for FAS by providing transferable visual representations learned from large-scale image collections. Recent studies have demonstrated the potential of foundation models and vision-language models for improving FAS robustness, while also highlighting the challenge of adapting these representations to domain-specific liveness cues [6]–[8].

Despite their strong representational capability, existing foundation-model-based approaches primarily optimize binary classification objectives, causing the learned representation to become entangled with attack-specific characteristics and dataset-dependent cues. Furthermore, treating all spoof samples as a single negative class overlooks the substantial intraclass variability of presentation attacks, whereas bonafide faces generally exhibit a more stable distribution across domains.

To address these limitations, we propose DINOFAS, a DINOv3-based framework that formulates FAS as a bonafidecentric representation learning problem (Fig. 1). The proposed framework combines parameter-efficient adaptation of DINOv3 using LoRA, an attention atyle-content disentanglement (ASCD) module for suppressing domain-specific appearance variations, and a dual-branch architecture comprising a prototype-based binary branch and a detached attack-semantic branch. Together, these components encourage the backbone to learn robust liveness representations while preventing attackspecific supervision from contaminating the shared feature space.

The main contributions of this work are summarized as follows:

• A DINOv3-based face anti-spoofing framework, termed DINOFAS, that formulates FAS as a bonafide-centric representation learning problem.

• An attention style-content disentanglement (ASCD) mod ule that suppresses domain-specific appearance variations through gated style-content decomposition and labelaware style borrowing.

![](images/778f11ccde3e36cf43554dfc3b7d4e612fc33ad5c3f64c6f4b0bec6eac13d340.jpg)  
Fig. 1. Conceptual overview of DINOFAS. The proposed framework replaces conventional binary classification with bonafide-centric representation learning, where a prototype-based liveness branch and a detached attack-semantic branch jointly learn robust and complementary representations.

• A dual-branch learning framework comprising a prototype-based binary branch and a detached attacksemantic branch, enabling complementary representation learning while preventing attack-category supervision from contaminating the shared backbone representation.

• Comprehensive experimental evaluation on multiple public FAS benchmarks, demonstrating the effectiveness of the proposed framework under cross-dataset, syntheticto-real, and protocol-based evaluation settings.

## II. RELATED WORKS

## A. Cross-Domain Face Anti-Spoofing

Face anti-spoofing (FAS) has evolved from handcrafted feature descriptors based on texture, frequency analysis, and motion cues [9]–[13] to deep learning methods based on convolutional neural networks (CNNs) [14]–[16] and, more recently, vision transformers [6], [7]. Although these methods achieve strong performance under intra-dataset evaluation [17]–[20], their performance degrades considerably under cross-dataset evaluation due to variations in acquisition devices, illumination, image quality, environments, and presentation attack instruments [15], [21]–[23].

To address this problem, existing methods primarily adopt domain adaptation (DA) or domain generalization (DG). DA methods align source and target distributions through feature alignment and adversarial learning [24], [25], whereas DG methods employ adversarial optimization and meta-learning to improve generalization to unseen domains [21], [26]– [28]. More recently, the work in [29] proposed dual-teacher knowledge distillation with domain alignment (DTDA), combining domain adversarial perturbations with perceptual and generative teacher models to transfer facial priors and reduce domain discrepancies. Although these approaches improve cross-dataset performance, they often require multiple source domains, additional teacher networks, or specialized optimization strategies [15], [29], [30].

Despite these advances, most existing FAS methods continue to formulate spoof detection as a binary classification problem, resulting in representations that become entangled with attack-specific artifacts and dataset-dependent cues, ultimately limiting their generalization to unseen presentation attacks and acquisition conditions.

## B. Foundation Models for FAS

Foundation models (FMs) pretrained on large-scale image collections have demonstrated strong transferability across diverse computer vision tasks through self-supervised and vision-language pretraining [31]–[33]. Representative vision foundation models include SAM [31], DINOv2 [34], CLIP [32], and ALIGN [35]. To efficiently adapt these large models to downstream tasks, parameter-efficient fine-tuning methods such as AdaptFormer [36], ViT-Adapter [37], and Low-Rank Adaptation (LoRA) [38] have been proposed.

The application of foundation models to FAS remains relatively recent. FoundPAD [6] demonstrated that adapting CLIP with LoRA significantly improves cross-domain FAS performance, while [39] explored the zero-shot capabilities of multiple foundation models for detecting unseen presentation attacks. Although these studies highlight the potential of foundation models for FAS, they primarily formulate spoof detection as a conventional binary classification problem without explicitly disentangling liveness information from attackspecific semantics.

## C. Domain-Invariant Representation Learning

Learning domain-invariant representations is fundamental to improving the robustness of FAS under unseen acquisition conditions and presentation attack instruments. Existing methods primarily achieve this through DA and DG, employing techniques such as feature alignment, adversarial optimization, and meta-learning to reduce domain discrepancies [15], [21], [24]–[28], [40]. Although these approaches improve cross-domain performance, they often require multiple source domains, access to target-domain samples, or computationally expensive optimization strategies [28], [30]. Methods such as DTDA [29] address domain discrepancies through adversarial perturbations and auxiliary knowledge transfer. In contrast, DINOFAS emphasizes bonafide-centric prototype learning, attention-based style-content disentanglement, and gradient-decoupled attack supervision without requiring separate teacher networks.

![](images/9f313db8f79ca154a07e0bd8b4ec9a842c9005afb6a3d396ea1de7a4d417650f.jpg)

![](images/cf0ffccbe1ae931c21c677c79382b19c545096fb6dd3eb017997e2b395d5035a.jpg)  
Fig. 2. Modified DINOv3 transformer with LoRA adaptation and the proposed attention style-content disentanglement (ASCD) module. The lower panel illustrates the ASCD pipeline, comprising gated style-content decomposition, label-aware style borrowing, and adaptive feature recombination.

## III. METHODOLOGY

The proposed framework models live faces as a compact reference manifold and treats spoof samples as deviations from this manifold. The framework comprises three main components, a DINOv3 vision transformer [41] adapted with low-rank adaptation (LoRA) [42], an attention style-content disentanglement (ASCD) module for suppressing domainspecific appearance variations, and a dual-branch learning strategy that decouples bonafide-centric liveness learning from attack-semantic supervision. Given an input face image, the backbone extracts global and spatial representations that are processed by two task-specific branches: a prototype-based binary branch for liveness estimation and a detached attack branch for spoof-category modeling. This design encourages the backbone to learn robust liveness representations while preventing attack-specific gradients from contaminating the shared feature space.

## A. DINOv3 Backbone with Low-Rank Adaptation

Let $x \in \mathbb { R } ^ { 3 \times H \times W }$ denote an input RGB face image. A pretrained DINOv3 vision transformer is used as the feature extractor, producing a global class-token representation $z _ { \mathrm { c l s } }$ and a sequence of spatial patch-token representations $Z _ { \mathrm { p a t c h } }$

$$
z _ { \mathrm { c l s } } , Z _ { \mathrm { p a t c h } } = f _ { \theta } ( x )\tag{1}
$$

where $z _ { \mathrm { c l s } } \in \mathbb { R } ^ { D }$ and $Z _ { \mathrm { p a t c h } } \in \mathbb { R } ^ { N \times D }$

To preserve the pretrained visual knowledge, most DI-NOv3 parameters remain frozen during training. Task-specific adaptation is performed using Low-Rank Adaptation (LoRA), where the attention projection matrix is updated as $\begin{array} { r l } { W ^ { \prime } = } \end{array}$ $W + B A$ , with A and B denoting trainable low-rank matrices. Layer normalization parameters are also optimized to allow mild distributional adaptation. As illustrated in Fig. 2, LoRA adapters are inserted into the transformer attention blocks, while the proposed ASCD module is integrated into the feature extraction pipeline to improve robustness against domainspecific appearance variations.

## B. Attention Style-Content Disentanglement

Face anti-spoofing is highly sensitive to domain-specific appearance variations arising from differences in imaging devices, illumination, compression, and acquisition conditions. To improve robustness, the proposed ASCD module, illustrated in Fig. 2, decomposes transformer token representations into style and content components, performs label-aware style borrowing, and adaptively recombines the resulting features.

Given a token sequence T, a lightweight gating network predicts a soft attention mask

$$
G = \sigma \big ( \phi ( T + \bar { T } _ { \mathrm { p a t c h } } ) \big )\tag{2}
$$

where $\bar { T } _ { \mathrm { p a t c h } }$ denotes the mean patch-token context. The token representation is decomposed into style and content components as $T _ { \mathrm { s t y l e } } = G \odot T$ and $T _ { \mathrm { c o n t e n t } } = ( 1 - G ) \odot T$ To preserve global semantic information, style statistics are computed only from the patch tokens, resulting in the patchwise mean $\mu$ and standard deviation $\sigma _ { s }$

The ASCD module performs attention style borrowing across samples within the mini-batch. During training, style exchange is restricted to samples from the same class to avoid unrealistic mixing between live and spoof distributions. Let $\left( \mu _ { t } , \sigma _ { t } \right)$ denote the style statistics borrowed through the attention mechanism. The source and borrowed statistics are adaptively fused as

$$
\mu _ { \mathrm { m i x } } = \lambda \mu + ( 1 - \lambda ) \mu _ { t } , \qquad \sigma _ { \mathrm { m i x } } = \lambda \sigma _ { s } + ( 1 - \lambda ) \sigma _ { t }\tag{3}
$$

where λ is a learnable mixing coefficient. The mixed style representation is reconstructed by adaptive normalization,

$$
T _ { \mathrm { s t y l e } } ^ { \prime } = \frac { T _ { \mathrm { s t y l e } } - \mu } { \sigma _ { s } } \odot \sigma _ { \mathrm { m i x } } + \mu _ { \mathrm { m i x } } ,\tag{4}
$$

and recombined with the preserved content representation,

$$
T ^ { \prime } { = } T _ { \mathrm { s t y l e } } ^ { \prime } { + } T _ { \mathrm { c o n t e n t } }\tag{5}
$$

This regularization encourages the backbone to suppress domain-specific style variations while preserving livenessrelated content.

## C. Spatial Projection for Localized Attack Evidence

Presentation attacks often exhibit localized artifacts such as mask boundaries, print texture, screen reflections, and partial occlusions. Relying solely on the global class token may therefore discard useful spatial evidence. To incorporate both global and local information, each branch employs a lightweight spatial projection network (SPN), illustrated in Fig. 3.

The patch tokens obtained from the DINOv3 backbone are reshaped into a two-dimensional spatial grid and processed by the SPN using a $3 \times 3$ depthwise convolution, GroupNorm (up to 32 groups), GELU, spatial dropout $( p = 0 . 2 )$ , a $1 \times 1$ pointwise convolution, and $1 \times 1$ adaptive average pooling. The resulting D-dimensional local feature is concatenated with the D-dimensional class token and projected from 2D back to $D$ using a fully-connected layer followed by LayerNorm, ReLU, and dropout $( p = 0 . 1 )$ ,

$$
z = \psi ( [ z _ { \mathrm { c l s } } ; \mathrm { P o o l } ( \mathrm { S P N } ( Z _ { \mathrm { p a t c h } } ) ) ] )\tag{6}
$$

where $\psi ( \cdot )$ denotes the final projection layer. The resulting feature combines global facial semantics with localized spoof evidence and serves as the input to both the bonafide-centric binary branch and the detached attack-semantic branch.

## D. Bonafide-Centric Binary Branch

As illustrated in the upper part of Fig. 3, the SPN output is forwarded to a bonafide-centric binary head that models the live-face distribution using a learnable prototype. Instead of employing a conventional linear classifier, the binary score is computed as the Euclidean distance between the normalized feature representation $z _ { \mathrm { b i n } }$ and the normalized liveness prototype c:

$$
d ( x ) = \left. { \frac { z _ { \mathrm { b i n } } } { \| z _ { \mathrm { b i n } } \| _ { 2 } } } - { \frac { c } { \| c \| _ { 2 } } } \right. _ { 2 }\tag{7}
$$

where c denotes the learnable liveness prototype.

Live samples are encouraged to remain close to the prototype, while spoof samples are pushed farther away, effectively treating FAS as a bonafide-centric anomaly detection problem. During inference, the prototype distance serves as the final spoof score, where larger distances indicate a higher likelihood of a presentation attack.

## E. Detached Attack-Semantic Branch

As illustrated in the lower part of Fig. 3, the detached attack-semantic branch learns fine-grained spoof categories from detached backbone features. The shared DINOv3 representations are first detached from the computational graph and then processed by the SPN:

$$
z _ { \mathrm { a t k } } = \psi _ { \mathrm { a t k } } ( \mathrm { s t o p g r a d } ( z _ { \mathrm { c l s } } ) , \mathrm { s t o p g r a d } ( Z _ { \mathrm { p a t c h } } ) )\tag{8}
$$

where the stop-gradient operation prevents the attackclassification objective from updating the shared backbone. The resulting feature is passed to an MLP classifier that predicts the predefined spoof categories:

$$
\hat { y } _ { \mathrm { a t k } } = h _ { \mathrm { a t k } } ( z _ { \mathrm { a t k } } )\tag{9}
$$

The attack-classification loss is computed only for spoof samples, while live samples are excluded since they do not belong to any spoof category. This enables the attack branch to learn discriminative attack semantics without contaminating the bonafide-centric representation used for liveness estimation.

## F. Training Objective

The two branches are jointly optimized using three complementary objectives: a bonafide-centric prototype loss, an attack-semantic classification loss applied to detached features, and an orthogonality loss that encourages the two branches to learn complementary representations. Let $y _ { i } \in \{ 0 , 1 \}$ denote the binary label of the i-th sample (0: live, 1: attack), and let $d _ { i }$ denote the prototype distance predicted by the binary branch.

The binary branch employs a margin-based SoftPlus objective that pulls live samples towards the bonafide prototype while pushing spoof samples away:

$$
\begin{array} { l }  { \displaystyle { \mathcal { L } } _ { \mathrm { b i n } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left[ \left( 1 - y _ { i } \right) \mathrm { s o f t p l u s } \left( \frac { d _ { i } - m _ { \mathrm { l i v e } } } { \tau } \right) \right. } \\ { { \displaystyle \qquad + y _ { i } \alpha _ { \mathrm { s p o o f ~ s o f t p l u s } } \left. \left( \frac { m _ { \mathrm { s p o o f } } - d _ { i } } { \tau } \right) \right] } } \end{array}\tag{10}
$$

where B is the batch size, $m _ { \mathrm { l i v e } } ~ = ~ 0 . 5 , ~ m _ { \mathrm { s p o o f } } ~ = ~ 1 . 6 ,$ $\alpha _ { \mathrm { s p o o f } } = 5 . 0$ , and $\tau = 0 . 1$ . Compared with a hard-margin formulation, the SoftPlus objective provides smoother optimization around the decision boundary.

The attack branch is supervised using cross-entropy computed only on spoof samples, i.e., $i \in \mathcal { M } _ { \mathrm { a t k } } = \{ i \mid a _ { i } > 0 \}$ where $a _ { i }$ denotes the attack label $\mathbf { \Phi } ( a _ { i } \ \mathbf { \Phi } = \ \mathbf { \Phi } 0$ for live and $a _ { i } ~ > ~ 0$ for the predefined spoof categories available in the training dataset, e.g., print, replay, and mask attacks). Since the classifier predicts only spoof categories, attack labels are shifted by one before computing the loss:

![](images/61c048bf4d739f6353a1a3945a5e01a70c1e0927b4ba27500ef0351072de99fb.jpg)  
Fig. 3. Task-specific branches of the proposed framework. The upper branch performs bonafide-centric prototype learning for binary liveness estimation, while the lower branch learns attack semantics through a detached classifier to prevent attack-specific supervision from contaminating the shared representation.

$$
\mathcal { L } _ { \mathrm { a t k } } = \frac { 1 } { \left| \mathcal { M } _ { \mathrm { a t k } } \right| } \sum _ { i \in \mathcal { M } _ { \mathrm { a t k } } } \mathrm { C E } \left( \hat { y } _ { \mathrm { a t k } , i } , a _ { i } - 1 \right)\tag{11}
$$

Label smoothing with a factor of 0.1 is applied, and the loss is set to zero when a mini-batch contains no spoof samples.

To encourage the two branches to encode complementary information, an orthogonality loss minimizes the absolute cosine similarity between the normalized binary and attack representations:

$$
\mathcal { L } _ { \mathrm { o r t h } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \left| \left. \frac { z _ { \mathrm { b i n } , i } } { \| z _ { \mathrm { b i n } , i } \| _ { 2 } } , \frac { z _ { \mathrm { a t k } , i } } { \| z _ { \mathrm { a t k } , i } \| _ { 2 } } \right. \right|\tag{12}
$$

The overall training objective is

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { b i n } } { \mathcal { L } } _ { \mathrm { b i n } } + \lambda _ { \mathrm { a t k } } { \mathcal { L } } _ { \mathrm { a t k } } + \lambda _ { \mathrm { o r t h } } { \mathcal { L } } _ { \mathrm { o r t h } }\tag{13}
$$

where $\lambda _ { \mathrm { b i n } } , \lambda _ { \mathrm { a t k } }$ , and $\lambda _ { \mathrm { o r t h } }$ are the loss weights.

## G. Inference

At inference, only the binary branch is required, the prototype distance $S ( x ) \ = \ d ( x )$ is used as the spoof score and thresholded to obtain the live/attack decision. ASCD style borrowing in the DINOFAS backbone is disabled during inference and the module acts as an identity mapping module.

## IV. RESULTS

## A. Datasets and Evaluation Protocols

The proposed method is evaluated on multiple publicly available FAS datasets, including SynthASpoof [43], Oulu-NPU (O) [4], CASIA-FASD (C) [48], Idiap Replay-Attack (I) [9], MSU-MFSD (M) [49], and SiW-Mv2 [50]. Crossdataset generalization is first assessed under a challenging synthetic-to-real protocol, where the model is trained exclusively on the SynthASpoof dataset and evaluated on the M, C, I, and O datasets. The proposed framework is further evaluated on the standard OCIM leave-one-out benchmark, where configurations such as OCI→M denote training on Oulu-NPU, CASIA-FASD, and Idiap Replay-Attack while testing on MSU-MFSD; OMI→C, OCM→I, and ICM→O follow the same protocol. Finally, robustness to specific sources of domain variation is evaluated using the standard OULU-NPU and SiW-Mv2 protocols, which assess generalization across unseen environmental conditions, presentation attack instruments, acquisition devices, and their combined variations.

## B. Implementation Details

The proposed framework is implemented in PyTorch and trained using the AdamW optimizer for a maximum of 120 epochs with early stopping (patience = 30) and a batch size of 300. LoRA adapters and task-specific heads are optimized using learning rates of $5 \times 1 0 ^ { - 6 }$ and $5 \times 1 0 ^ { - 4 }$ , respectively, with a weight decay of 0.05. Mixed-precision training, gradient clipping (1.0), and weighted random sampling are employed throughout training to improve optimization stability and mitigate class imbalance. Input images are resized to 224×224 and augmented using random horizontal flipping, color jittering, Gaussian blur, JPEG compression, and Gaussian noise. During evaluation, the decision threshold is determined independently for each evaluation protocol by maximizing Youden’s index on the corresponding validation split and is subsequently used to compute HTER, ACER, APCER, and BPCER. Unless otherwise stated, the loss weights are set to $\lambda _ { \mathrm { b i n } } ~ = ~ 1 . 0 .$ $\lambda _ { \mathrm { a t t a c k } } = 0 . 5 ,$ , and $\lambda _ { \mathrm { o r t h } } = 0 . 2$

TABLE I  
SINGLE-SOURCE CROSS-DATASET EVALUATION WHEN TRAINING ON THE SYNTHASPOOF DATASET. LOWER HTER AND HIGHER AUC INDICATE BETTER PERFORMANCE. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="2">M</td><td colspan="2">C</td><td colspan="2">I</td><td colspan="2">0</td><td colspan="2">Average</td></tr><tr><td>HTER(%)↓</td><td>AUC(%)↑</td><td>HTER(%)↓</td><td>AUC(%)↑</td><td>HTER(%)↓</td><td>AUC(%)↑</td><td>HTER(%)↓</td><td>AUC(%)↑</td><td>HTER(%)↓</td><td>AUC(%)↑</td></tr><tr><td>ResNet [43]</td><td>25.48</td><td>79.54</td><td>39.22</td><td>62.00</td><td>8.90</td><td>96.96</td><td>34.23</td><td>71.48</td><td>26.96</td><td>77.50</td></tr><tr><td>PixBiS [43]</td><td>38.33</td><td>63.87</td><td>38.44</td><td>64.79</td><td>7.50</td><td>96.88</td><td>35.77</td><td>63.50</td><td>30.74</td><td>72.26</td></tr><tr><td>ViT-SIDE B [44]</td><td>36.67</td><td>69.78</td><td>33.33</td><td>75.21</td><td>9.80</td><td>96.67</td><td>13.26</td><td>94.04</td><td>23.27</td><td>83.93</td></tr><tr><td>SynFace Co-Former A [44]</td><td>18.57</td><td>90.76</td><td>41.11</td><td>64.49</td><td>16.30</td><td>92.31</td><td>21.67</td><td>86.44</td><td>24.41</td><td>83.50</td></tr><tr><td>SynFace Co-Former B [44]</td><td>16.67</td><td>91.61</td><td>40.00</td><td>63.05</td><td>18.80</td><td>88.20</td><td>25.35</td><td>82.02</td><td>25.21</td><td>81.22</td></tr><tr><td>CoDe-Lc [44]</td><td>37.14</td><td>71.45</td><td>37.11</td><td>69.08</td><td>12.10</td><td>95.31</td><td>37.58</td><td>66.30</td><td>30.98</td><td>75.54</td></tr><tr><td>CoDe-Lh [44]</td><td>39.05</td><td>70.58</td><td>39.33</td><td>63.70</td><td>13.90</td><td>93.84</td><td>38.11</td><td>68.33</td><td>32.60</td><td>74.11</td></tr><tr><td>OrthPADNet [44]</td><td>20.95</td><td>87.59</td><td>39.78</td><td>67.32</td><td>23.70</td><td>79.55</td><td>34.92</td><td>71.69</td><td>29.84</td><td>76.54</td></tr><tr><td>idvcVT [44]</td><td>45.71</td><td>64.58</td><td>56.44</td><td>41.82</td><td>23.10</td><td>85.08</td><td>51.09</td><td>49.68</td><td>44.09</td><td>60.29</td></tr><tr><td>hdafVPAD [44]</td><td>65.71</td><td>31.95</td><td>71.33</td><td>21.95</td><td>47.80</td><td>52.10</td><td>37.89</td><td>66.49</td><td>55.68</td><td>43.12</td></tr><tr><td>FoundPAD (ViT-B) [6] FoundPAD (ViT-L) [6]</td><td>47.14 45.71</td><td>66.18 69.76</td><td>27.33 9.89</td><td>83.03 96.03</td><td>16.15 6.40</td><td>90.79</td><td>33.12</td><td>73.56</td><td>30.94 23.51</td><td>78.39</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>98.58</td><td>32.05</td><td>75.69</td><td></td><td>85.01</td></tr><tr><td>DINOFAS (ViT-B) (Proposed)</td><td>25.83</td><td>83.06</td><td>31.30</td><td>75.54</td><td>9.12</td><td>96.57</td><td>19.40</td><td>89.39</td><td>21.41</td><td>86.14</td></tr></table>

TABLE II

CROSS-DOMAIN FACE PRESENTATION ATTACK DETECTION RESULTS ON THE OCIM BENCHMARK. LOWER HTER AND HIGHER AUC INDICATE BETTER PERFORMANCE. THE BEST RESULTS ARE SHOWN IN BOLD, WHILE THE SECOND-BEST RESULTS ARE UNDERLINED.
<table><tr><td rowspan="2">Method</td><td colspan="2">OCI→M</td><td colspan="2">OMI→C</td><td colspan="2">OCM→I</td><td colspan="2">ICM→0</td><td rowspan="2">Avg. HTER(%)↓</td><td rowspan="2">Avg. AUC(%)↑</td></tr><tr><td>HTER↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td></tr><tr><td>NAS-FAS [20]</td><td>19.53</td><td>88.63</td><td>16.54</td><td>90.18</td><td>14.51</td><td>93.84</td><td>13.80</td><td>93.43</td><td>16.10</td><td>91.52</td></tr><tr><td>LMFD [17]</td><td>10.48</td><td>94.55</td><td>12.50</td><td>94.17</td><td>18.49</td><td>84.72</td><td>12.41</td><td>94.95</td><td>13.47</td><td>92.10</td></tr><tr><td>MADDG [26]</td><td>17.69</td><td>88.06</td><td>24.50</td><td>84.51</td><td>22.19</td><td>84.99</td><td>27.89</td><td>80.02</td><td>23.07</td><td>84.40</td></tr><tr><td>RFM [21]</td><td>17.30</td><td>90.48</td><td>13.89</td><td>93.98</td><td>20.27</td><td>88.16</td><td>16.45</td><td>91.16</td><td>16.98</td><td>90.95</td></tr><tr><td>SSDG-R [27]</td><td>7.38</td><td>97.17</td><td>10.44</td><td>95.94</td><td>11.71</td><td>96.59</td><td>15.61</td><td>91.54</td><td>11.29</td><td>95.31</td></tr><tr><td>D2AM [28]</td><td>12.70</td><td>95.66</td><td>20.98</td><td>85.58</td><td>15.43</td><td>91.22</td><td>15.27</td><td>90.87</td><td>16.10</td><td>90.83</td></tr><tr><td>ViT [45]</td><td>4.75</td><td>98.59</td><td>15.70</td><td>92.76</td><td>17.68</td><td>86.66</td><td>16.46</td><td>90.37</td><td>13.65</td><td>92.10</td></tr><tr><td>TransFAS [22]</td><td>7.08</td><td>96.69</td><td>9.81</td><td>96.13</td><td>10.12</td><td>95.53</td><td>15.52</td><td>91.10</td><td>10.63</td><td>94.86</td></tr><tr><td>CIFAS [30]</td><td>5.95</td><td>96.32</td><td>10.66</td><td>95.30</td><td>8.50</td><td>97.24</td><td>13.17</td><td>93.44</td><td>9.57</td><td>95.58</td></tr><tr><td>SSAN-M [46]</td><td>10.42</td><td>94.76</td><td>16.47</td><td>90.81</td><td>14.00</td><td>96.58</td><td>19.51</td><td>88.17</td><td>15.10</td><td>92.08</td></tr><tr><td>GDA [47]</td><td>9.20</td><td>98.00</td><td>12.20</td><td>93.00</td><td>10.00</td><td>96.00</td><td>14.40</td><td>92.60</td><td>11.45</td><td>94.90</td></tr><tr><td>CF-PAD [15]</td><td>8.11</td><td>96.43</td><td>11.78</td><td>95.64</td><td>16.50</td><td>91.50</td><td>9.87</td><td>95.13</td><td>11.57</td><td>94.68</td></tr><tr><td>DTDA [29]</td><td>5.71</td><td>98.03</td><td>6.67</td><td>97.27</td><td>13.12</td><td>92.24</td><td>13.13</td><td>94.24</td><td>9.66</td><td>95.45</td></tr><tr><td>FoundPAD (ViT-B) [6]</td><td>20.95</td><td>89.88</td><td>4.89</td><td>98.08</td><td>10.45</td><td>95.80</td><td>6.19</td><td>98.31</td><td>10.62</td><td>95.52</td></tr><tr><td>FoundPAD (ViT-L) [6]</td><td>16.90</td><td>93.18</td><td>6.00</td><td>98.72</td><td>9.90</td><td>96.07</td><td>5.87</td><td>98.41</td><td>9.67</td><td>96.60</td></tr><tr><td>DINOFAS (ViT-B) (Proposed)</td><td>3.75</td><td>98.39</td><td>7.59</td><td>95.99</td><td>17.25</td><td>86.05</td><td>7.77</td><td>97.16</td><td>9.09</td><td>94.40</td></tr></table>

## C. Comparison with State-of-the-Art Methods

a) Synthetic-to-Real Generalization: Herein, we train DINOFAS exclusively on the SynthASpoof dataset and evaluate it on four unseen FAS benchmarks (M, C, I, and O). The results are summarized in Table I. DINOFAS achieves the lowest average HTER of 21.41% and the highest average AUC of 86.14%, outperforming previous synthetic-databased FAS methods, including FoundPAD [6]. Although the proposed framework does not obtain the best performance on every individual target dataset, it consistently maintains competitive results across all evaluation benchmarks, leading to the best overall cross-dataset performance. These results demonstrate that the proposed bonafide-centric representation learning strategy transfers effectively from synthetic training data to previously unseen real-world domains.

b) Cross-Dataset Evaluation: Table II presents the results on the standard OCIM leave-one-out benchmark. DINO-FAS achieves the lowest average HTER of 9.09%, outperforming CIFAS [30] (9.57%), DTDA [29] (9.66%), and FoundPAD (ViT-L) [6] (9.67%). The proposed method achieves the lowest HTER on OCI→M (3.75%) and competitive performance on OMI→C (7.59%) and ICM→O (7.77%). Compared to DTDA,

DINOFAS improves HTER on OCI→M and ICM→O, but performs worse on OMI→C and OCM→I. The latter also contributes to the lower average AUC of DINOFAS (94.40%) compared with DTDA (95.45%). These results indicate that the proposed representation learning strategy improves aggregate HTER without consistently outperforming competing methods on every target domain.

Figure 4 visualizes the learned representations across the four OCIM protocols. Bonafide and attack samples remain largely separated across unseen target domains, while the increased overlap for OCM→I is consistent with its comparatively lower quantitative performance.

## D. Further Generalization Analysis

To further assess the robustness of the proposed framework under different sources of domain variation, Table III reports the results on the four standard evaluation protocols of the OULU-NPU dataset. These protocols evaluate generalization across unseen environmental conditions (Protocol 1), presentation attack instruments (Protocol 2), acquisition devices (Protocol 3), and their combined variations (Protocol 4).

![](images/c70a0fba41338d8e4558473f46ef75ed3a81c6d9e5168c66b1ce920165a49d4a.jpg)  
Fig. 4. t-SNE visualization of DINOFAS representations under the four OCIM cross-domain protocols.  
TABLE III

COMPARISON OF FACE PRESENTATION ATTACK DETECTION PERFORMANCE ON THE OULU-NPU DATASET UNDER THE FOUR STANDARD EVALUATION PROTOCOLS. LOWER ACER, APCER, AND BPCER INDICATE BETTER PERFORMANCE. THE BEST AND SECOND-BEST RESULTS FOR EACH PROTOCOL ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.

<table><tr><td>P</td><td>Method</td><td>ACER</td><td>APCER</td><td>BPCER</td></tr><tr><td rowspan="2">1</td><td>LMFD-PAD [17] PatchSwap [14]</td><td>1.50 0.60</td><td>1.40 0.40</td><td>1.60 0.80</td></tr><tr><td>MAX (DINO, CLIP) [39] DINOFAS (Proposed)</td><td>2.64 1.67</td><td>2.50</td><td>2.77</td></tr><tr><td rowspan="4">2</td><td>LMFD-PAD [17]</td><td>2.00</td><td>1.67 3.10</td><td>1.67 0.80</td></tr><tr><td>PatchSwap [14]</td><td>1.80</td><td>2.50</td><td>1.10</td></tr><tr><td>MAX (DÍNO, CLIP) [39]</td><td>2.13</td><td>2.04</td><td>2.22</td></tr><tr><td>DINOFAS (Proposed)</td><td>1.87</td><td>2.08</td><td>1.67</td></tr><tr><td>3</td><td>LMFD-PAD [17] PatchSwap [14] MAX (DINO, CLIP) [39] DINOFAS (Proposed)</td><td>3.40±3.10 3.30±4.90 2.61±0.66 0.17±0.20</td><td>3.50±3.20 1.40±1.30 2.63±0.67 0.34±0.41</td><td>3.40±3.10 5.30±10.00 2.59±0.65 0.00±0.00</td></tr><tr><td>4</td><td>LMFD-PAD [17] PatchSwap [14] MAX (DÍNO, CLIP) [39] DINOFAS (Proposed)</td><td>3.30±3.10 3.80±6.30 3.42±0.85 2.92±2.33</td><td>2.50±4.10 2.50±8.30 3.06±1.12 5.00±5.24</td><td>3.30±3.10 5.00±4.20 3.78±0.69 0.83±2.04</td></tr></table>

On the OULU-NPU benchmark, DINOFAS demonstrates competitive performance across all four evaluation protocols. Under Protocols 1 and 2, which evaluate robustness to unseen environmental conditions and presentation attack instruments, respectively, the proposed framework achieves performance comparable to recent state-of-the-art methods while maintaining low ACER, APCER, and BPCER. The most significant improvement is observed on Protocol 3, which evaluates camera interoperability using the leave-one-cameraout protocol. Here, DINOFAS substantially outperforms the state-of-the-art methods, reducing the ACER from the previous best of 2.61±0.66% obtained by MAX (DINO, CLIP) [39] to 0.17 ± 0.20%, while also achieving the lowest APCER and a zero BPCER. Under the most challenging Protocol 4, where environmental conditions, presentation attack instruments, and acquisition devices vary simultaneously, DINOFAS achieves the lowest ACER (2.92±2.33%) and BPCER (0.83±2.04%), demonstrating strong robustness under multiple sources of domain variation despite a comparatively higher APCER.

TABLE IV  
HTER (%) ON THE SIW-MV2 LEAVE-ONE-OUT PROTOCOL. LOWER VALUES INDICATE BETTER PERFORMANCE. THE BEST AND SECOND RESULT FOR EACH ATTACK TYPE ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td>Attack</td><td>MD-FAS [50]</td><td>Meta [51]</td><td>DINO (ViT-B-14) [39]</td><td>DINOFAS (Proposed)</td></tr><tr><td>FunE.</td><td>29.50</td><td>14.80</td><td>8.52</td><td>19.62</td></tr><tr><td>PEye</td><td>2.70</td><td></td><td>0.19</td><td>0.38</td></tr><tr><td>PMouth</td><td>1.10</td><td>10.30</td><td>0.19</td><td>0.19</td></tr><tr><td>PaperG.</td><td>11.90</td><td>15.20</td><td>0.19</td><td>5.44</td></tr><tr><td>Ob.</td><td>1.30</td><td>45.30</td><td>18.16</td><td>7.47</td></tr><tr><td>Impers.</td><td>24.50</td><td>12.90</td><td>0.39</td><td>0.38</td></tr><tr><td>Cosmetic</td><td>10.90</td><td>12.10</td><td>18.88</td><td>6.32</td></tr><tr><td>HalfM.</td><td>8.00</td><td>11.70</td><td>1.47</td><td>0.76</td></tr><tr><td>Silicone</td><td>9.20</td><td>13.60</td><td>0.58</td><td>0.38</td></tr><tr><td>TransM.</td><td>0.00</td><td>24.70</td><td>1.41</td><td>3.25</td></tr><tr><td>Paper</td><td>0.60</td><td>3.20</td><td>0.00</td><td>0.00</td></tr><tr><td>Mann.</td><td>4.00</td><td>9.50</td><td>0.19</td><td>0.38</td></tr><tr><td>Replay</td><td>17.90</td><td>3.90</td><td>7.05</td><td>5.93</td></tr><tr><td>Print</td><td>9.60</td><td>1.20</td><td>13.99</td><td>9.88</td></tr><tr><td>Avg.±Std.</td><td>9.40 ± 8.80</td><td>13.70 ± 10.80</td><td>5.09 ± 7.05</td><td>4.31 ± 5.50</td></tr></table>

To further evaluate generalization to unseen presentation attack instruments, Table IV reports the results on the SiW-Mv2 leave-one-out protocol. DINOFAS achieves the best performance on the Impers., Cosmetic, Half Mask, and Silicone protocols, while matching the best HTER on PMouth and Paper. It further attains the lowest average HTER of 4.31%, outperforming all competing methods. Performance remains comparatively weaker on FunE., TransM., and Print, indicating that these attack types remain more challenging due to their larger appearance variability.

The OULU-NPU and SiW-Mv2 evaluations collectively demonstrate that DINOFAS maintains robust performance under complementary sources of distribution shift, including acquisition conditions, camera sensors, and previously unseen presentation attack types.

## E. Ablation Studies

1) Effect of the Backbone and LoRA Adaptation: An ablation study comparing different DINO backbone variants on the OCIM cross-dataset benchmark is presented in Table V. For both DINOv2 and DINOv3, parameter-efficient adaptation using LoRA (rank r = 8, α = 16, dropout = 0.05) substantially improves performance over the corresponding frozen backbones, reducing the average HTER from 22.63% to 13.10% for DINOv2 and from 24.62% to 10.85% for DINOv3. Although the frozen DINOv2 backbone exhibits slightly better generalization than the frozen DINOv3 backbone, DINOv3 consistently achieves superior performance after adaptation, indicating its stronger capacity for learning task-specific liveness representations. We further investigate the influence of DINOv3 model scale by evaluating DINOFAS with ViT-S/16, ViT-B/16, and ViT-L/16 backbones. Among the three variants, ViT-B/16 achieves the best average HTER (9.09%), outperforming both the smaller ViT-S/16 and the larger ViT-L/16 models. This suggests that the base-sized backbone provides the most favorable balance between representation capacity and adaptation for the proposed bonafide-centric learning framework.

TABLE V  
EFFECT OF FOUNDATION-MODEL BACKBONE AND LORA ADAPTATION ON THE OCIM CROSS-DATASET BENCHMARK. THE DINOV2/DINOV3 BASELINE VARIANTS USE A CONVENTIONAL LINEAR BINARY HEAD WITH BCE. WHEREAS DINOFAS ROWS USE THE COMPLETE PROPOSED FRAMEWORK. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="2">OCI→M</td><td colspan="2">OMI→C</td><td colspan="2">OCM→I</td><td colspan="2">ICM→0</td><td rowspan="2"> $\mathbf { A v g } .$  HTER↓</td><td rowspan="2">Avg. AUC ↑</td></tr><tr><td>HTER ↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td><td>HTER ↓</td><td>AUC ↑</td><td>HTER↓</td><td>AUC ↑</td></tr><tr><td>DINOv2 (ViT-B/16) (Frozen)</td><td>22.50</td><td>83.22</td><td>20.55</td><td>87.55</td><td>16.75</td><td>86.95</td><td>30.72</td><td>74.83</td><td>22.63</td><td>83.14</td></tr><tr><td>DINOv2 (ViT-B/16) + LoRA</td><td>12.50</td><td>95.77</td><td>11.85</td><td>95.26</td><td>13.37</td><td>91.40</td><td>14.68</td><td>91.44</td><td>13.10</td><td>93.47</td></tr><tr><td>DINOv3 (ViT-B/16) (Frozen)</td><td>23.75</td><td>81.02</td><td>23.14</td><td>83.62</td><td>25.62</td><td>80.38</td><td>25.97</td><td>80.53</td><td>24.62</td><td>81.39</td></tr><tr><td>DINOv3 (ViT-B/16) + LoRA</td><td>13.33</td><td>94.22</td><td>11.85</td><td>95.15</td><td>8.87</td><td>94.71</td><td>9.34</td><td>95.27</td><td>10.85</td><td>94.84</td></tr><tr><td>DINOFAS (ViT-S/16)</td><td>17.08</td><td>90.41</td><td>12.40</td><td>94.74</td><td>20.06</td><td>83.03</td><td>15.52</td><td>91.72</td><td>16.26</td><td>89.97</td></tr><tr><td>DINOFAS (ViT-B/16) (ours)</td><td>3.75</td><td>98.39</td><td>7.59</td><td>95.99</td><td>17.25</td><td>86.05</td><td>7.77</td><td>97.16</td><td>9.09</td><td>94.40</td></tr><tr><td>DINOFAS (ViT-L/16)</td><td>9.16</td><td>97.31</td><td>7.77</td><td>97.58</td><td>16.00</td><td>89.48</td><td>12.84</td><td>93.82</td><td>11.44</td><td>94.55</td></tr></table>

TABLE VI  
ABLATION STUDY OF THE PROPOSED LEARNING STRATEGY ON THE OCIM CROSS-DATASET BENCHMARK. ASCD IS ENABLED IN ALL CONFIGURATIONS, WHILE $L _ { \mathrm { b i n } } ,$ L<sub>attack</sub>, $L _ { \mathrm { o r t h } }$ , AND THE STOP-GRADIENT MECHANISM ARE VARIED TO EVALUATE THEIR CONTRIBUTIONS.
<table><tr><td>Cfg.</td><td> $\underline { { L _ { \mathrm { b i n } } } }$ </td><td>Lattack</td><td> $\overline { { L _ { \mathrm { o r t h } } } }$ </td><td>StopGrad</td><td> ${ \overline { { \mathrm { A v g . } } } }$  HTER  $\overline { { ( \% ) \downarrow } }$ </td></tr><tr><td>A</td><td>√</td><td>×</td><td>×</td><td>一</td><td>14.16</td></tr><tr><td>B</td><td>√</td><td>√</td><td>×</td><td>×</td><td>13.95</td></tr><tr><td>C</td><td>√</td><td>X</td><td>√</td><td>√</td><td>14.03</td></tr><tr><td>D</td><td>√</td><td>√</td><td>×</td><td>√</td><td>13.45</td></tr><tr><td>E</td><td>√</td><td>√</td><td>√</td><td>X</td><td>13.30</td></tr><tr><td>Proposed</td><td>√</td><td>√</td><td>√</td><td>√</td><td>9.09</td></tr></table>

2) Effect of the Proposed Learning Strategy: The contribution of the proposed learning strategy on the OCIM cross-dataset benchmark is evaluated in Table VI. Configuration A, which optimizes only the bonafide-centric prototype loss $( L _ { \mathrm { b i n } } )$ , establishes a baseline with an average HTER of 14.16%. Configuration B incorporates the attack-semantic classification loss $( L _ { \mathrm { a t t a c k } } )$ without stop-gradient, resulting in a modest improvement over the baseline. Enabling the stop-gradient mechanism in configuration D further reduces the average HTER to 13.45%, indicating that isolating attack supervision from the shared backbone improves generalization . Introducing the orthogonality loss alone, as expressed by configuration C, provides only a marginal gain over the baseline, whereas combining all three objectives in configuration E, without stop-gradient achieves an average HTER of 13.30%. The complete DINOFAS framework, which jointly optimizes $L _ { \mathrm { b i n } } , L _ { \mathrm { a t t a c k } }$ , and $L _ { \mathrm { o r t h } }$ with stop-gradient enabled, achieves the best performance with an average HTER of 9.09%. These results show that, on top of the fixed ASCD-enabled backbone, the attack-semantic objective and orthogonality regularization are most effective when attack supervision is decoupled from the shared backbone through the stop-gradient mechanism.

3) Effect of ASCD Design Choices: The effectiveness of the proposed ASCD module is evaluated through three complementary ablation experiments on the OCIM cross-dataset benchmark, as summarized in Table VII. First, replacing the proposed class-aware style borrowing with class-agnostic style mixing degrades the average HTER from 9.09% to 13.36%, demonstrating the importance of restricting style exchange within the same liveness class. Second, removing the gating mechanism increases the average HTER to 12.14%, indicating that adaptive style-content decomposition is more effective than uniform feature mixing. Finally, the placement of ASCD is investigated by inserting the modules at different transformer stages. Applying ASCD to the early transformer blocks (Blocks 1, 2) performs similarly to removing the module altogether, while inserting it into the late transformer blocks (Blocks 10, 11) also degrades performance. In contrast, placing ASCD in the middle semantic blocks (Blocks 4 and 6) achieves the best performance, suggesting that intermediate transformer representations provide the most suitable balance between semantic abstraction and domain-specific style information for style-content disentanglement.

TABLE VII  
ABLATION STUDY OF THE PROPOSED ASCD MODULE ON THE OCIM CROSS-DATASET BENCHMARK. LOWER HTER AND HIGHER AUC INDICATE BETTER PERFORMANCE.
<table><tr><td>Component</td><td>Configuration</td><td>Avg. HTER (%)↓</td><td>Avg. AUC (%)↑</td></tr><tr><td rowspan="2">Style Borrowing</td><td>Class-agnostic</td><td>13.36</td><td>92.00</td></tr><tr><td>Class-aware (Proposed)</td><td>9.09</td><td>94.40</td></tr><tr><td rowspan="2">Gating Mechanism</td><td>Without gating</td><td>12.14</td><td>93.52</td></tr><tr><td>With gating (Proposed)</td><td>9.09</td><td>94.40</td></tr><tr><td rowspan="4">ASCD Placement</td><td>None</td><td>13.32</td><td>91.87</td></tr><tr><td>Early (Blocks 1,2)</td><td>13.67</td><td>91.84</td></tr><tr><td>Middle (Blocks 4,6–Proposed)</td><td>9.09</td><td>94.40</td></tr><tr><td>Late (Blocks 10,11)</td><td>14.02</td><td>92.35</td></tr></table>

![](images/50ddfb686dc0de4873380b599196090fc710d0ade5a6fe167d39656fff6faf8f.jpg)  
Fig. 5. Sensitivity to $\lambda _ { \mathrm { o r t h } }$ on OCIM.

4) Sensitivity to the Orthogonality Loss Weight: The influence of the orthogonality loss weight $\left( \lambda _ { \mathrm { o r t h } } \right)$ is evaluated on the OCIM cross-dataset benchmark by varying its value while keeping all other training settings fixed. As shown in Fig. 5, $\lambda _ { \mathrm { o r t h } } ~ = ~ 0 . 2$ consistently provides the best overall trade-off, achieving the lowest average HTER of 9.09% while also achieving the best or near-best performance across the individual evaluation protocols. Increasing the orthogonality weight beyond this point degrades performance, particularly on the CASIA-FASD and OULU-NPU datasets, indicating that excessive feature decorrelation can interfere with the primary liveness and attack-semantic learning objectives.

## V. CONCLUSION

This paper presented DINOFAS, a DINOv3-based framework for face anti-spoofing that combines bonafide-centric representation learning with parameter-efficient foundation model adaptation. The proposed framework integrates attention stylecontent disentanglement, detached attack-semantic supervision, and a joint optimization strategy based on bonafidecentric prototype, attack-semantic, and orthogonality objectives to learn robust and complementary liveness representations while reducing reliance on dataset-specific spoof characteristics. Experimental evaluations on multiple public FAS benchmarks demonstrate strong cross-dataset generalization, achieving the best average HTER among the compared methods on both the SynthASpoof-to-real transfer setting and the standard OCIM benchmark, while maintaining robust performance on the OULU-NPU and SiW-Mv2 generalization protocols.

## REFERENCES

[1] A. George, C. Ecabert, H. O. Shahreza, K. Kotwal, and S. Marcel, “Edgeface: Efficient face recognition model for edge devices,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 6, no. 2, pp. 158–168, 2024.

[2] F. Abdullakutty, E. Elyan, P. Johnston, and A. Ali-Gombe, “Deep transfer learning on the aggregated dataset for face presentation attack detection,” Cognitive computation, vol. 14, no. 6, pp. 2223–2233, 2022.

[3] S. Chen, T. Yao, K. Zhang, Y. Chen, K. Sun, S. Ding, J. Li, F. Huang, and R. Ji, “A dual-stream framework for 3d mask face presentation attack detection,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 834–841.

[4] Z. Boulkenafet, J. Komulainen, L. Li, X. Feng, and A. Hadid, “Oulu-npu: A mobile face presentation attack database with real-world variations,” in 2017 12th IEEE international conference on automaticface & gesture recognition (FG 2017). IEEE, 2017, pp. 612–618.

[5] F. Boutros, N. Damer, F. Kirchbuchner, and A. Kuijper, “Elasticface: Elastic margin loss for deep face recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 1578–1587.

[6] G. Ozgur, E. Caldeira, T. Chettaoui, F. Boutros, R. Ramachandra, and N. Damer, “Foundpad: Foundation models reloaded for face presentation attack detection,” in Proceedings of the Winter Conference on Applications of Computer Vision, 2025, pp. 745–755.

[7] A. George and S. Marcel, “On the effectiveness of vision transformers for zero-shot face anti-spoofing,” in 2021 IEEE international joint conference on biometrics (IJCB). IEEE, 2021, pp. 1–8.

[8] A. Komaty, H. O. Shahreza, A. George, and S. Marcel, “Exploring chatgpt for face presentation attack detection in zero and few-shot in-context learning,” in Proceedings of the Winter Conference on Applications of Computer Vision, 2025, pp. 834–843.

[9] I. Chingovska, A. Anjos, and S. Marcel, “On the effectiveness of local binary patterns in face anti-spoofing,” in 2012 BIOSIG-proceedings of the international conference of biometrics special interest group (BIOSIG). IEEE, 2012, pp. 1–7.

[10] J. Galbally, S. Marcel, and J. Fierrez, “Biometric antispoofing methods: A survey in face recognition,” Ieee Access, vol. 2, pp. 1530–1552, 2014.

[11] M. R. Hasan, S. Hasan Mahmud, and X. Y. Li, “Face anti-spoofing using texture-based techniques and filtering methods,” in Journal of Physics: Conference Series, vol. 1229, no. 1. IOP Publishing, 2019, p. 012044.

[12] C. Wang, B. Yu, and J. Zhou, “A learnable gradient operator for face presentation attack detection,” Pattern Recognition, vol. 135, p. 109146, 2023.

[13] L. J. Gonzalez-Soler, M. Gomez-Barrero, and C. Busch, “On the´ generalisation capabilities of fisher vector-based face presentation attack detection,” IET Biometrics, vol. 10, no. 5, pp. 480–496, 2021.

[14] M. Fang, H. Ali, A. Kuijper, and N. Damer, “Patchswap: Boosting the generalizability of face presentation attack detection by identityaware patch swapping,” in 2022 IEEE International Joint Conference on Biometrics (IJCB). IEEE, 2022, pp. 1–10.

[15] M. Fang and N. Damer, “Face presentation attack detection by excavating causal clues and adapting embedding statistics,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp. 6269–6279.

[16] D. Deb and A. K. Jain, “Look locally infer globally: A generalizable face anti-spoofing approach,” IEEE Transactions on Information Forensics and Security, vol. 16, pp. 1143–1157, 2020.

[17] M. Fang, N. Damer, F. Kirchbuchner, and A. Kuijper, “Learnable multilevel frequency decomposition and hierarchical attention mechanism for generalized face presentation attack detection,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2022, pp. 3722–3731.

[18] Y. Liu, A. Jourabloo, and X. Liu, “Learning deep models for face antispoofing: Binary or auxiliary supervision,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 389– 398.

[19] K. Srivatsan, M. Naseer, and K. Nandakumar, “Flip: Cross-domain face anti-spoofing with language guidance,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 19 685–19 696.

[20] Z. Yu, J. Wan, Y. Qin, X. Li, S. Z. Li, and G. Zhao, “Nas-fas: Staticdynamic central difference network search for face anti-spoofing,” IEEE transactions on pattern analysis and machine intelligence, vol. 43, no. 9, pp. 3005–3023, 2020.

[21] R. Shao, X. Lan, and P. C. Yuen, “Regularized fine-grained meta face anti-spoofing,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 07, 2020, pp. 11 974–11 981.

[22] Z. Wang, Q. Wang, W. Deng, and G. Guo, “Face anti-spoofing using transformers with relation-aware mechanism,” IEEE Transactions on Biometrics, Behavior, and Identity Science, vol. 4, no. 3, pp. 439–450, 2022.

[23] W. Yan, Y. Zeng, and H. Hu, “Domain adversarial disentanglement network with cross-domain synthesis for generalized face anti-spoofing,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 10, pp. 7033–7046, 2022.

[24] H. Li, W. Li, H. Cao, S. Wang, F. Huang, and A. C. Kot, “Unsupervised domain adaptation for face anti-spoofing,” IEEE Transactions on Information Forensics and Security, vol. 13, no. 7, pp. 1794–1809, 2018.

[25] G. Wang, H. Han, S. Shan, and X. Chen, “Improving cross-database face presentation attack detection via adversarial domain adaptation,” in 2019 International Conference on Biometrics (ICB). IEEE, 2019, pp. 1–8.

[26] R. Shao, X. Lan, J. Li, and P. C. Yuen, “Multi-adversarial discriminative deep domain generalization for face presentation attack detection,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 10 023–10 031.

[27] Y. Jia, J. Zhang, S. Shan, and X. Chen, “Single-side domain generalization for face anti-spoofing,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 8484–8493.

[28] Z. Chen, T. Yao, K. Sheng, S. Ding, Y. Tai, J. Li, F. Huang, and X. Jin, “Generalizable representation learning for mixture domain face anti-spoofing,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 2, 2021, pp. 1132–1139.

[29] Z. Kong, W. Zhang, T. Wang, K. Zhang, Y. Li, X. Tang, and W. Luo, “Dual teacher knowledge distillation with domain alignment for face anti-spoofing,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 12, pp. 13 177–13 189, 2024.

[30] Y. Liu, Y. Chen, W. Dai, C. Li, J. Zou, and H. Xiong, “Causal intervention for generalizable face anti-spoofing,” in 2022 IEEE International Conference on Multimedia and Expo (ICME). IEEE, 2022, pp. 01–06.

[31] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[32] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[33] M. Awais, M. Naseer, S. Khan, R. M. Anwer, H. Cholakkal, M. Shah, M.-H. Yang, and F. S. Khan, “Foundation models defining a new era in

vision: a survey and outlook,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 4, pp. 2245–2264, 2025.

[34] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[35] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in International conference on machine learning. PMLR, 2021, pp. 4904–4916.

[36] S. Chen, C. Ge, Z. Tong, J. Wang, Y. Song, J. Wang, and P. Luo, “Adaptformer: Adapting vision transformers for scalable visual recognition,” Advances in Neural Information Processing Systems, vol. 35, pp. 16 664–16 678, 2022.

[37] Z. Chen, Y. Duan, W. Wang, J. He, T. Lu, J. Dai, and Y. Qiao, “Vision transformer adapter for dense predictions,” arXiv preprint arXiv:2205.08534, 2022.

[38] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” Iclr, vol. 1, no. 2, p. 3, 2022.

[39] L. J. Gonzalez-Soler, J. E. Tapia, and C. Busch, “Are foundation models all you need for zero-shot face presentation attack detection?” in 2025 IEEE 19th International Conference on Automatic Face and Gesture Recognition (FG). IEEE, 2025, pp. 1–10.

[40] K. Wang, G. Zhang, H. Yue, A. Liu, G. Zhang, H. Feng, J. Han, E. Ding, and J. Wang, “Multi-domain incremental learning for face presentation attack detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 6, 2024, pp. 5499–5507.

[41] O. Simeoni, H. V. Vo, M. Seitzer, F. Baldassarre, M. Oquab, C. Jose,´ V. Khalidov, M. Szafraniec, S. Yi, M. Ramamonjisoa et al., “Dinov3,” arXiv preprint arXiv:2508.10104, 2025.

[42] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.

[43] M. Fang, M. Huber, and N. Damer, “Synthaspoof: Developing face presentation attack detection based on privacy-friendly synthetic data,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2023, pp. 1061–1070.

[44] M. Fang, M. Huber, J. Fierrez, R. Ramachandra, N. Damer, A. Alkhaddour, M. Kasantcev, V. Pryadchenko, Z. Yang, H. Huangfu et al., “Synfacepad 2023: Competition on face presentation attack detection based on privacy-aware synthetic training data,” in 2023 IEEE International Joint Conference on Biometrics (IJCB). IEEE, 2023, pp. 1–11.

[45] H.-P. Huang, D. Sun, Y. Liu, W.-S. Chu, T. Xiao, J. Yuan, H. Adam, and M.-H. Yang, “Adaptive transformers for robust few-shot crossdomain face anti-spoofing,” in European conference on computer vision. Springer, 2022, pp. 37–54.

[46] Z. Wang, Z. Wang, Z. Yu, W. Deng, J. Li, T. Gao, and Z. Wang, “Domain generalization via shuffled style assembly for face anti-spoofing,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022, pp. 4113–4123.

[47] Q. Zhou, K.-Y. Zhang, T. Yao, R. Yi, K. Sheng, S. Ding, and L. Ma, “Generative domain adaptation for face anti-spoofing,” in European conference on computer vision. Springer, 2022, pp. 335–356.

[48] Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, and S. Z. Li, “A face antispoofing database with diverse attacks,” in 2012 5th IAPR international conference on Biometrics (ICB). IEEE, 2012, pp. 26–31.

[49] D. Wen, H. Han, and A. K. Jain, “Face spoof detection with image distortion analysis,” IEEE Transactions on Information Forensics and Security, vol. 10, no. 4, pp. 746–761, 2015.

[50] X. Guo, Y. Liu, A. Jain, and X. Liu, “Multi-domain learning for updating face anti-spoofing models,” in European conference on computer vision. Springer, 2022, pp. 230–249.

[51] U. Muhammad, J. Laaksonen, D. Romaissa Beddiar, and M. Oussalah, “Domain generalization via ensemble stacking for face presentation attack detection,” International Journal of Computer Vision, vol. 132, no. 12, pp. 5759–5782, 2024.

![](images/e98a3f4e5a33ec8cbc9d4cfd72de8e7c3f093c4f772b6b06769bb83abce42da2.jpg)

Shakeel Muhammad Ibrahim received his BS degree in Computer Science from the University of Karachi, Karachi, Pakistan, in 2019, and his MS degree in Computer Engineering from Hallym University, Chuncheon-si, Republic of Korea, in 2025. From 2020 to 2023, he worked as an ML Engineer at Softech Worldwide, focusing on computer vision problems, including object detection, semantic segmentation, and face recognition. During his master’s studies, he served as a Graduate Research Assistant at the AI Accelerator Computing Lab at Hallym

University. He is currently working as an AI Researcher at Geomexsoft Co., Ltd., in Chuncheon-si, Republic of Korea. His research interests include computer vision and deep learning, with a focus on face recognition, image reconstruction, face presentation attack detection, and aerial segmentation.

![](images/cbc8411451779e28a7a09389411dbe7a5d0e5d352e4c6248938fc3479d334651.jpg)

Muhammad Sohail Ibrahim earned his B.E. degree in Electronic Engineering from Iqra University, Pakistan, in 2012, followed by an M.E. degree in Telecommunications from N.E.D. University of Engineering & Technology, Pakistan, in 2016, and his Ph.D. degree from Chosun University, Republic of Korea, in 2024. From 2013 to 2019, he served as a lecturer in the Faculty of Engineering, Science, and Technology at Iqra University, Karachi, Pakistan. During 2013-2014, he also worked as a research assistant with the Embedded Systems Research Group

at the Karachi Institute of Economics and Technology, Pakistan. During 2024- 2025, he conducted his postdoctoral research at Microfluidics and Nano-BioSystems (MNBS) lab. at Kumoh National Institute of Technology, Gumi-si, Republic of Korea. He is currently pursuing his postdoctoral research at The Interdisciplinary Research Center for Intelligent Secure Systems (IRC-ISS) at King Fahd University of Petroleum & Minerals, Dhahran, Saudi Arabia. His research interests span deep learning, computer vision, bioinformatics, and medical image processing. Muhammad Sohail Ibrahim was honored with the 2020 Highly Cited Review Paper award from Applied Energy, Elsevier, for his paper titled ”Machine Learning Driven Smart Electric Systems: Current Trends and New Perspectives”.

![](images/621b693d8d237be65fd4b5d94b2111164574dcf1c48f0578b4d8fc86a8502acc.jpg)

Muhammad Shahid Jabbar earned his Ph.D. in Computer Engineering from Sungkyunkwan University, Suwon, South Korea, in 2022. He was a Researcher at the H-Lab and the Media System Lab at SKKU. His doctoral research focused on developing modern deep learning solutions using multimodal data for convergence applications involving multisensory interaction and cross-modular associations, with particular emphasis on accessibility for visually impaired users. He is currently a Post-Doctoral Fellow at the SDAIA–KFUPM Joint Research Center

for Artificial Intelligence, King Fahd University of Petroleum & Minerals (KFUPM), Dhahran, KSA. Prior to joining KFUPM, he worked as Senior AI Researcher at Radisen Co. Ltd., Seoul, South Korea, where he contributed to AI-powered digital radiography solutions for medical image analysis. His research interests include computer vision, diffusion models, multimodal learning, and inverse problems, with a focus on biomedical and humancentered artificial intelligence.

![](images/fb2c16cb5b3b1f22ee95c11333adcdca76b56bf601f3969691c6c3c7ee81d557.jpg)

Shujaat Khan is an Assistant Professor in the Department of Computer Engineering and a Fellow at the Saudi Data and AI Authority (SDAIA) and King Fahd University of Petroleum & Minerals (KFUPM) under the SDAIA-KFUPM Joint Research Center for Artificial Intelligence (JRC-AI) at KFUPM, Dhahran, KSA. Prior to joining KFUPM, he was a Senior AI Scientist at Digital Technology & Innovation, Siemens Medical Solutions USA, Inc. He earned his Ph.D. from the Department of Bio and Brain Engineering at the Korea Advanced

Institute of Science and Technology (KAIST), Daejeon, South Korea, in 2022. He was a researcher with Synergistic Bioinformatics (SynBi) and the Bio Imaging, Signal Processing Learning (BISPL) Labs at KAIST. His research interests include machine learning, optimization, inverse problems, and signal processing, with a focus on biomedical and bioinformatics applications.