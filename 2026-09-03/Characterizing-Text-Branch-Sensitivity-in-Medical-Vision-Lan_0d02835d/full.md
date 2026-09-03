# Characterizing Text Branch Sensitivity in Medical Vision-Language Segmentation via Evidence Decoupling

Ziquan Liu , Zhewei Zhu , and Xuyang Shi<sup>⋆</sup>

School of Information and Control Engineering, Southwest University of Science and Technology, Mianyang 621010, China

{ziquanliu, zhuzw}@mails.swust.edu.cn, xuyangshi@swust.edu.cn

Abstract. Pretrained vision-language models (VLMs) have shown promis ing performance in medical image segmentation by incorporating clinical text. However, it remains unclear how much textual information actually contributes to pixel-level predictions. In this work, we systematically investigate the role of text in multimodal medical image segmentation. We first analyze several commonly used fusion strategies and find that segmentation performance is largely insensitive to the choice of fusion module. To further understand modality interactions, we propose an Evidence Decoupling Decoder (EDD) based on evidential deep learning and deep supervision. EDD serves as an internal representation analysis tool that decomposes image evidence and text-modulated evidence throughout the decoding process while maintaining competitive segmentation performance. Experimental results show that the sensitivity to text perturbation varies substantially across datasets. On BUSI and BTMRI, removing text causes catastrophic performance drops, indicating strong model reliance on textual input. On ISIC and Kvasir-SEG, text exerts relatively marginal influence. We further find that text afects predictions mainly through global semantic modulation rather than independent spatial localization, and that the specific semantic components driving text sensitivity difer across datasets. These findings provide a deeper understanding of modality interaction in multimodal medical image segmentation and ofer practical insights for future model design.

Keywords: Medical image segmentation · Vision-language model · Evidential deep learning · Deep Supervision · Evidence Decoupling Decoder

## 1 Introduction

Pretrained vision-language models (VLMs) have advanced multimodal medical image segmentation by incorporating clinical text into vision networks [22,16]. However, existing work implicitly assumes text is a beneficial auxiliary modality and focuses on designing increasingly complex fusion architectures [33,23,30,24].

While recent studies have identified modality imbalance and collapse as structural properties of VLM training [2,17,35], these operate at the representation or gradient level and do not characterize per-pixel modality interaction in medical segmentation. A more basic question remains understudied: how much does the text branch actually contribute to pixel-level decisions?

To study this, we evaluate multimodal medical segmentation across four imaging modalities [1,5,13,7,6], five pretrained medical VLMs [26,32,36,8,14] and four cross-modal fusion operators [25,31,11,27]. Our experiments reveal that four topologically distinct fusion operators yield very similar segmentation results, suggesting that changing the fusion micro-structure has little impact on multimodal information use, as visual features dominate under current pretrained representations.

To examine internal modality interactions, we introduce evidential deep learning (EDL) [29] and propose a lightweight Evidence Decoupling Decoder (EDD). EDL reformulates deterministic probability outputs as non-negative evidence under a Dirichlet distribution, allowing us to characterize the learned modality representations. Inspired by deep supervision [18], we design separate vision and text evidence heads at each upsampling layer of the decoder. These heads extract image evidence $\mathbf { e } _ { I }$ from visual-only features and text-modulated evidence e<sub>T</sub>, driven by a unified EDL joint loss. Experiments confirm that this decoder preserves segmentation accuracy close to the standard decoder across all VLM and dataset combinations, providing an interpretability tool that reveals how the model internally represents modality interaction. In summary, this paper makes three contributions:

We show that four topologically distinct fusion operators yield nearly identical segmentation performance across five VLMs and four medical imaging datasets, indicating that fusion design has limited impact under current pretrained representations.

– We propose an Evidence Decoupling Decoder (EDD) that combines evidential deep learning with deep supervision, achieving per-layer decomposition of image and text-modulated evidence without sacrificing accuracy.

– We reveal through perturbation analysis that text sensitivity is strongly dataset dependent. On BTMRI and BUSI, text corruption causes severe degradation, while the impact of text is relatively small on ISIC and Kvasir-SEG. Text-modulated evidence is globally distributed, and diferent text components drive sensitivity across datasets.

## 2 Related Work

## 2.1 Text-Guided Medical Image Segmentation

Vision-language models (VLMs) increasingly leverage clinical text as auxiliary guidance for medical image segmentation. Following foundational architectures such as ViLT [15], text-guided segmentation frameworks typically adopt either early fusion of linguistic and visual features within the encoder [33,22], or textconditioned attention and contrastive alignment in the decoder [30,23,24]. Med-CLIPSeg [16] further introduces probabilistic adaptation for data-eficient multimodal segmentation. While these methods advance architecture design, they uniformly treat text as an inherently beneficial modality, without quantitatively investigating how much the text branch actually contributes to pixel-level decisions.

## 2.2 Modality Interaction and Imbalance in Vision-Language Models

Recent work questions whether multimodal learning always yields balanced contributions. Studies characterize the modality gap in contrastive VLMs, where image and text embeddings form distinct clusters in the joint space, and link it to mismatched training pairs, temperature, and information imbalance between modalities [35,28]. At the fusion level, modality collapse has been attributed to noisy cross-modal feature entanglement and unaligned inter-modal gradients that prevent balanced convergence [2,17]. In medical segmentation, TMCA [20] uses contrastive alignment to narrow image-text pattern gaps. While these works establish modality dominance as a structural property of VLM training, they operate at the representation or gradient level and do not ofer per-pixel decomposition of modality-specific spatial contributions.

## 2.3 Evidential Deep Learning for Medical Image Analysis

Evidential deep learning (EDL) [29] casts classification as evidence under a Dirichlet distribution, enabling joint prediction and uncertainty quantification. Recent EDL-based medical segmentation methods span semi-supervised learning [10], brain tumor segmentation [19], and progressive uncertainty guidance [34]. Building on Dempster–Shafer theory, Fidon et al. [9] and Huang et al. [12] fuse imaging modalities (e.g., PET/CT) with reliability weights, while DEviS [38] and SURE [21] improve uncertainty interpretability and calibration. However, existing multimodal EDL frameworks focus on multiple imaging modalities rather than vision-language pairs, and none decompose decisions into per-modality, per-layer evidence streams. Our evidence decoupling decoder fills this gap.

## 3 Methodology

## 3.1 Problem Definition and Overall Framework

Given an input image $\mathbf { I } \in \mathbb { R } ^ { H \times W \times 3 }$ and its text description T, we aim to predict a binary segmentation mask $\hat { \mathbf { Y } } \in \{ 0 , 1 \} ^ { H \times W }$ . As shown in Fig. 1, the overall framework consists of an image encoder $f _ { I }$ , a text encoder $f _ { T }$ , and a segmentation decoder D:

$$
\hat { \mathbf { Y } } = D \left( f _ { I } ( \mathbf { I } ) , \ f _ { T } ( \mathbf { T } ) \right)\tag{1}
$$

To enable fine-grained analysis of modality-specific representations, we propose an Evidence Decoupling Decoder (EDD) that separately models image and text evidence streams throughout the decoding process.

![](images/c6e030e47fe52bedbbc3448889e6366c321d28019de59680b50cc21c23f291f6.jpg)  
Fig. 1. Our baseline framework feeds image samples and corresponding text into the image encoder and text encoder, respectively. The resulting features are then processed step by step through a U-shaped segmentation decoder for segmentation. The lower half of the figure provides a schematic of the four fusion operators.

Table 1. Architecture and pretraining details of evaluated encoders.
<table><tr><td>Model</td><td>Vision Backbone Text Backbone Pretraining Data</td><td></td><td></td><td>Scale</td><td>Modality Coverage</td></tr><tr><td>CLIP [26]</td><td>|ViT-B/16</td><td>Transformer</td><td>WIT</td><td>~400M</td><td>Natural images</td></tr><tr><td>MedCLIP [32]</td><td>Swin-Tiny</td><td>BioClinicalBERT MIMIC-CXR</td><td></td><td>~200K</td><td>Chest X-ray only</td></tr><tr><td>BioMedCLIP [36]</td><td>ViT-B/16</td><td>PubMedBERT</td><td>PMC-15M</td><td>~15M</td><td>Biomedical images</td></tr><tr><td>PubMedCLIP [8]</td><td>ViT-B/32</td><td>Transformer</td><td>PubMed+ROCO</td><td></td><td>300K-400K Multimodal medical</td></tr><tr><td>UniMed-CLIP [14][ViT-B/16</td><td></td><td>BiomedBERT</td><td>Mixed medical corpora ~5.3M</td><td></td><td>All medical modalities</td></tr></table>

## 3.2 Baseline Segmentation Network with Multimodal Features

To cover diferent vision-language pretraining paradigms, we evaluate five representative pretrained medical VLMs as image-text feature extractors. Their configurations are listed in Table 1.

The standard decoder uses a U-shaped hierarchical structure that gradually recovers spatial resolution through skip connections. Let l index decoder levels from the shallowest (l = 0) to the deepest $( l = L - 1 )$ . At level l, we upsample the deeper-level feature and fuse it with the encoder skip feature $\mathbf { F } ^ { ( l ) }$ to obtain an intermediate visual feature $\mathbf { u } ^ { ( l ) }$

$$
\mathbf { u } ^ { ( l ) } = \mathrm { U p } \left( \mathbf { x } ^ { ( l + 1 ) } \right) + \mathbf { F } ^ { ( l ) } ,\tag{2}
$$

where $\mathrm { U p } ( \cdot )$ is the upsampling operation. After fusing image features with skip connections, the fusion module injects text embedding $\mathbf { t } \in \mathbb { R } ^ { D }$ into the visual

features $\mathbf { u } ^ { ( l ) }$ . A residual convolutional block refines the result to produce the current level output $\mathbf { x } ^ { ( l ) }$ . The shallowest layer output $\mathbf { x } ^ { ( 0 ) }$ passes through a simple segmentation head to produce the segmentation result.

To study the efect of diferent cross-modal interaction mechanisms, we integrate four typical fusion operators into the standard decoder [25,31,11,27], as shown in the lower part of Fig. 1.

## 3.3 Evidence Decoupling Decoder

Design Motivation and Modeling Considerations. A standard softmaxbased decoder outputs deterministic class probabilities and cannot separate modality-specific representations. We adopt the EDL framework [29], which models outputs as non-negative evidence under a Dirichlet distribution, enabling explicit decomposition of modality-level evidence. Inspired by deep supervision [18], we aim to capture the per-layer evolution of evidence from both branches. The evidence decoupling decoder is guided by two design principles:

Evidence-space projectability. Intermediate features at each decoder level are semantically rich enough to be projected into non-negative evidence via a lightweight convolution and Softplus activation.

Hierarchical supervision consistency. Per-level evidence maps share the same semantic target as the final output, difering only in spatial resolution, so all levels can use the same supervision signal.

Guided by these principles, we design an evidence decoupling branch, dividing the evidence at each layer into pure image evidence, text-modulated evidence, and fused evidence.

Evidence Decoupling Branch Design. To avoid contaminating raw visual representations, the evidence decoupling branch runs after each skip connection but before the fusion module, where the visual feature $\mathbf { u } ^ { ( l ) }$ is not yet textmodulated and thus provides a suitable base for evidence extraction.

Image evidence stream. The image evidence head consists of a 1 × 1 convolution and a Softplus activation. It directly extracts a non-negative image evidence map $\mathbf { e } _ { I } ^ { ( l ) } \in \mathbb { R } _ { + } ^ { \tilde { K } \times H _ { l } \times W _ { l } }$ from the raw visual feature, where $K = 2$ is the number of classes and $H _ { l } , W _ { l }$ denote the spatial resolution at decoder level l:

$$
\mathbf { e } _ { I } ^ { ( l ) } = \mathrm { S o f t p l u s } \left( \mathrm { C o n v } _ { 1 \times 1 } \left( \mathbf { u } ^ { ( l ) } \right) \right)\tag{3}
$$

This stream depends only on visual features. It reflects how well the vision encoder can independently discriminate the segmentation target.

Text evidence stream. Unlike image features, the text embedding $\mathbf { t } \in \mathbb { R } ^ { D }$ is a global semantic vector. It lacks spatial structure and per-pixel correspondence, so it cannot directly produce a spatially resolved evidence map. To bridge this gap, we model text evidence as a global channel-level modulation on image spatial features, where the text provides class-wise semantic intensity coeficients and the image provides spatial response bases. Their product spatializes the text semantics.

First, an MLP maps the text embedding $\mathbf { t } \in \mathbb { R } ^ { D }$ into K dimension semantic intensity coeficients:

$$
\mathbf { e } _ { T } ^ { \mathrm { c h a n } } = \mathrm { S o f t p l u s } \left( \mathrm { M L P } ( \mathbf { t } ) \right) \in \mathbb { R } _ { + } ^ { K }\tag{4}
$$

At the same time, we extract spatial evidence bases $\mathbf { s } ^ { ( l ) } \in \mathbb { R } _ { + } ^ { K \times H _ { l } \times W _ { l } }$ from the current-level visual feature:

$$
\mathbf { s } ^ { ( l ) } = \mathrm { S o f t p l u s } \left( \mathrm { C o n v } _ { 1 \times 1 } \left( \mathbf { u } ^ { ( l ) } \right) \right)\tag{5}
$$

Note that although $\mathbf { s } ^ { ( l ) }$ and $\mathbf { e } _ { I } ^ { ( l ) }$ both use a $1 \times 1$ convolution plus Softplus on the same input $\mathbf { u } ^ { ( l ) }$ , they have independent parameters and distinct semantic roles. $\mathbf { e } _ { I } ^ { ( l ) }$ learns to provide direct evidence for the segmentation target from the image modality. $\mathbf { s } ^ { ( l ) }$ learns spatial response bases that are suitable for modulation by text semantics.

The text-modulated evidence ${ \bf e } _ { T } ^ { ( l ) }$ is then computed by channel-wise broadcast multiplication:

$$
\mathbf { e } _ { T } ^ { ( l ) } = \mathbf { e } _ { T } ^ { \mathrm { { c h a n } } } \odot \mathbf { s } ^ { ( l ) } \in \mathbb { R } _ { + } ^ { K \times H _ { l } \times W _ { l } }\tag{6}
$$

This design lets text evidence inherit visual spatial patterns in structure while being globally regulated by text semantic coeficients in intensity.

It is important to clarify that ${ \bf e } _ { T } ^ { ( l ) }$ does not represent a purely text-driven evidence map that would exist independently of vision. Text evidence is, by construction, text-modulated visual evidence. Its spatial structure is inherited from $\mathbf { s } ^ { ( l ) }$ , which derives from visual features $\mathbf { u } ^ { ( l ) }$ . The decomposition primarily characterizes how the trained decoder represents modality interaction, rather than measuring an intrinsic modality-independent text contribution. This architectural choice is intentional. A global text vector has no native spatial resolution, so spatialization through visual features is necessary. However, this choice imposes an interpretive constraint. The resulting evidence maps reflect the model’s learned encoding of text semantics in visual coordinates, not an independent text-to-pixel pathway.

Evidence fusion. The two decoupled evidence streams are combined additively to form the total fused evidence $\mathbf { \bar { e } } _ { \mathrm { a d d } } ^ { ( l ) }$ at the current level:

$$
{ \bf e } _ { \mathrm { a d d } } ^ { ( l ) } = { \bf e } _ { I } ^ { ( l ) } + { \bf e } _ { T } ^ { ( l ) }\tag{7}
$$

This additive combination requires no learned parameters, making the decomposition structurally transparent by construction. We emphasize that this decomposition reflects the model’s learned evidence representation rather than ground-truth modality attribution.

## 3.4 Loss Function and Inference Quantification

Baseline Network Loss. For the standard segmentation network without EDL branches, we use the combination of Dice loss and cross-entropy (CE) loss:

$$
\mathcal { L } _ { \mathrm { s e g } } = \mathcal { L } _ { \mathrm { D i c e } } + \lambda \mathcal { L } _ { \mathrm { C E } } ,\tag{8}
$$

where λ is the loss balance parameter.

Evidence Network Joint Loss. With EDL, segmentation is reformulated as an evidence regression process under the Dirichlet distribution. Given an evidence vector e, the corresponding Dirichlet concentration parameters are ${ \pmb { \alpha } } = { \bf e } + 1$ , and the total evidence is $\textstyle S = \sum _ { k = 1 } ^ { K } \alpha _ { k }$ . The single-layer EDL loss $\mathcal { L } _ { \mathrm { E D L } } ( \mathbf { e } , Y )$ consists of a fidelity term and a KL regularization term.

Fidelity term. For the true class $y ,$ we minimize the expected squared error under the Dirichlet distribution:

$$
{ \mathcal { L } } _ { \mathrm { e r r } } ( \mathbf { e } , Y ) = \sum _ { k = 1 } ^ { K } \left( y _ { k } - { \frac { \alpha _ { k } } { S } } \right) ^ { 2 } + { \frac { \alpha _ { k } ( S - \alpha _ { k } ) } { S ^ { 2 } ( S + 1 ) } }\tag{9}
$$

The first term measures the deviation between the expected probability and the ground-truth label. The second term accounts for prediction variance.

KL regularization term. For non-true classes, we use KL divergence to push their evidence toward zero. This prevents the model from assigning high confidence to all classes:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { K L } } ( \mathbf { e } , Y ) = \mathrm { K L } \left[ \mathrm { D i r } ( \mathbf { p } \mid \tilde { \alpha } ) \parallel \mathrm { D i r } ( \mathbf { p } \mid \mathbf { 1 } ) \right] , } \end{array}\tag{10}
$$

where $\tilde { \alpha } = Y + ( 1 - Y ) \odot \alpha$ is the Dirichlet parameter after removing the evidence of the true class.

The single-layer EDL loss is then:

$$
\mathcal { L } _ { \mathrm { E D L } } ( \mathbf { e } , Y ) = \mathcal { L } _ { \mathrm { e r r } } ( \mathbf { e } , Y ) + \lambda _ { t } \cdot \mathcal { L } _ { \mathrm { K L } } ( \mathbf { e } , Y ) ,\tag{11}
$$

where $\lambda _ { t }$ is an annealing coeficient that gradually increases during training to avoid excessive regularization early on.

Multi-level joint loss. The overall training loss is a weighted sum of the EDL loss at the final segmentation head and at all intermediate levels $( L = 4$ decoder levels):

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { E D L } } \left( \mathbf { e } ^ { \mathrm { f i n a l } } , Y \right) + \sum _ { l = 0 } ^ { L - 1 } w _ { l } \cdot \mathcal { L } _ { \mathrm { E D L } } \left( \mathbf { e } _ { \mathrm { a d d } } ^ { ( l ) } , Y \right)\tag{12}
$$

where the weights $w _ { l }$ decrease with increasing layer depth.

At inference time, the model outputs expected class probabilities $\hat { p } _ { k } = \alpha _ { k } / S$ Additionally, the epistemic uncertainty at each pixel can be quantified from the total evidence:

$$
u = \frac { K } { S } = \frac { K } { \sum _ { j = 1 } ^ { K } \alpha _ { j } }\tag{13}
$$

When a pixel has low total evidence $S ,$ the model lacks suficient support for its decision, corresponding to high epistemic uncertainty.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate on four medical imaging datasets with paired text descriptions, including BUSI [1], BTMRI [5], Kvasir-SEG [13], and ISIC [7,6]. The image-text pairs for all four datasets are obtained from the MedCLIPSeg [16] repository, which provides curated text descriptions aligned with each sample. To ensure fair comparison, the dataset splits also follow those provided by MedCLIPSeg, and all experiments adopt the same training strategy and optimization hyperparameters. Both the image encoder and text encoder are not frozen throughout training. All experiments use images resized to 224 × 224, batch size 32, AdamW optimizer with learning rate $2 \times 1 0 ^ { - 4 }$ and weight decay $1 \times 1 0 ^ { - 4 }$ , trained and tested on an NVIDIA V100 32G GPU. Segmentation performance is measured by Dice coeficient and IoU.

## 4.2 Baseline Performance Analysis

Impact of Diferent VLM Encoders. We first evaluate the baseline segmentation performance of five pretrained VLM encoders across four datasets, as shown in Table 2. The Gating fusion operator is used as the default in all experiments in this section.

Substantial performance gaps exist among VLM encoders. For example, on BTMRI, MedCLIP achieves 91.61% Dice, while PubMedCLIP reaches only 78.40% and BioMedCLIP attains 87.63%. The best-performing encoder also depends on the dataset, as UniMed-CLIP leads on BUSI and ISIC while MedCLIP excels on BTMRI and Kvasir-SEG. No single VLM encoder consistently dominates all datasets.

Fusion Module Sensitivity Analysis. To systematically evaluate the impact of cross-modal fusion mechanisms, we compare four fusion operators (FiLM [25], cross-attention [31], channel gating [11], concatenation [27]) across all VLM encoders.

As shown in Table 3, fusion modules cause only marginal performance variation, with the maximum fluctuation $\varDelta \mathrm { { _ { m a x } } }$ of 2.75% for BioMedCLIP. The gaps between diferent VLMs, such as the roughly 4% diference between CLIP and

Table 2. Segmentation performance of diferent VLM encoders across four datasets. Best results are in bold.
<table><tr><td rowspan="2">Model</td><td colspan="2">BTMRI</td><td colspan="2">BUSI</td><td colspan="2">Kvasir-SEG</td><td colspan="2">ISIC</td></tr><tr><td>Dice</td><td>IoU</td><td>Dice</td><td>IoU</td><td>Dice</td><td>IoU</td><td>Dice</td><td>IoU</td></tr><tr><td>UNet [27]</td><td>86.34</td><td>79.88</td><td>68.12</td><td>60.73</td><td>80.87</td><td>71.64</td><td>88.39</td><td>82.41</td></tr><tr><td>UNet++ [37]</td><td>85.76</td><td>77.71</td><td>67.22</td><td>59.48</td><td>81.75</td><td>73.30</td><td>88.47</td><td>82.84</td></tr><tr><td>DeepLabv3 [4]</td><td>85.25</td><td>78.63</td><td>65.81</td><td>58.23</td><td>82.03</td><td>72.58</td><td>81.94</td><td>70.79</td></tr><tr><td>TransUNet [3]</td><td>71.47</td><td>60.36</td><td>62.80</td><td>54.74</td><td>60.26</td><td>50.08</td><td>85.18</td><td>77.63</td></tr><tr><td>CLIP [26]</td><td>87.85</td><td>80.07</td><td>81.22</td><td>68.88</td><td>85.60</td><td>74.85</td><td>92.37</td><td>85.85</td></tr><tr><td>MedCLIP [32]</td><td></td><td>91.61 85.42</td><td>82.43</td><td>70.68</td><td></td><td>91.32 84.06</td><td>93.71 88.18</td><td></td></tr><tr><td></td><td>78.40</td><td>68.46</td><td>81.12</td><td>68.58</td><td>76.93</td><td>362.53</td><td></td><td></td></tr><tr><td>PubMedCLIP [8]</td><td>87.63</td><td>79.69</td><td>83.93 72.89</td><td></td><td>88.74</td><td>79.81</td><td>92.77</td><td>91.3584.11</td></tr><tr><td>BioMedCLIP [36]</td><td></td><td></td><td>82.99</td><td>971.45</td><td>89.32</td><td>80.73</td><td></td><td>86.55</td></tr><tr><td>UniMed-CLIP [14]</td><td>88.84</td><td>81.32</td><td></td><td></td><td></td><td></td><td>93.60</td><td>87.98</td></tr></table>

Table 3. Average Dice scores (%) across four datasets for diferent fusion modules, along with the maximum fluctuation range $\varDelta \mathrm { { _ { m a x } } }$ (%).
<table><tr><td>Model</td><td>|FiLM Cross-Attention Gating Concat</td><td></td><td> $\Delta _ { \mathrm { m a x } }$ </td></tr><tr><td>CLIP</td><td>85.45 86.01</td><td>86.76 85.57</td><td>1.31</td></tr><tr><td>MedCLIP</td><td>89.81 89.56</td><td>89.77 89.29</td><td>0.52</td></tr><tr><td>PubMedCLIP</td><td>82.02 82.26</td><td>81.95 82.47</td><td>0.52</td></tr><tr><td>BioMedCLIP</td><td>86.42 85.52</td><td>88.27 85.53</td><td>2.75</td></tr><tr><td>UniMed-CLIP</td><td>88.54 88.91</td><td>88.69 89.26</td><td>0.72</td></tr></table>

MedCLIP on BTMRI, far exceed the diferences across fusion strategies. This suggests that segmentation quality depends primarily on visual feature quality rather than fusion topology, and that merely optimizing the fusion module cannot fundamentally alter multimodal information utilization.

## 4.3 Evidence Decoupling Decoder Validation

We verify that the proposed Evidence Decoupling Decoder (EDD) preserves segmentation performance while enabling evidence decomposition. Table 4 compares the standard VLM decoder against EDD. For nearly all VLM-dataset pairs, the EDD segmentation performance remains close to that of the standard decoder, with Dice and IoU deviations generally modest, although a few cases exhibit larger changes. In most cases, the EDD variant yields gains over the standard decoder, for example a 2.08% Dice improvement for CLIP on BUSI and a 6.32% gain for PubMedCLIP on BTMRI. A few pairs show slight degradations, such as BioMedCLIP on BUSI dropping by 2.15% and UniMed-CLIP on BUSI by 1.82%, which are consistent with the added EDL regularization and remain within the expected variance for medical image segmentation. The occasional performance improvements may be due to the efect of the auxiliary deep supervision signal, which can improve and optimize the baseline decoder when it is underfitting. EDD’s primary purpose remains evidence decomposition for interpretability rather than performance enhancement. Overall, these results demonstrate that the evidence decoupling branch can be introduced without systematically disrupting the original segmentation optimization.

Table 4. Segmentation performance comparison between the standard VLM decoder and the evidence decoupling decoder (Dice / IoU, %).
<table><tr><td rowspan="2">Model</td><td>BTMRI</td><td>BUSI</td><td rowspan="2">|Kvasir-SEG|</td><td rowspan="2"></td><td rowspan="2">ISIC</td></tr><tr><td>Dice IoU</td><td>Dice IoU</td></tr><tr><td>CLIP</td><td>|87.85 80.07</td><td>|81.22 68.88</td><td>Dice 85.60</td><td>IoU 74.85</td><td>Dice IoU |92.37 85.85</td></tr><tr><td>CLIP + EDD MedCLIP</td><td>87.50 79.52 |91.61 85.42</td><td>83.30 72.05 |82.43 70.68</td><td>|91.32</td><td>86.4676.22 84.06</td><td>92.17 85.50 |93.71 88.18</td></tr><tr><td>MedCLIP + EDD PubMedCLIP</td><td>91.61 85.41 |78.40 68.46</td><td>84.34 78.07 |81.12 68.58</td><td>76.93</td><td>90.46 82.61 62.53</td><td>93.12 87.17 |91.35 84.11</td></tr><tr><td>PubMedCLIP + EDD</td><td>84.72 75.83</td><td>79.82 67.16</td><td>82.06 69.60</td><td></td><td>92.05 85.29</td></tr><tr><td>BioMedCLIP</td><td>|87.63 79.69</td><td>|83.93 72.89</td><td>|88.74 79.81</td><td></td><td>|92.77 86.55</td></tr><tr><td>BioMedCLIP + EDD</td><td>88.55 81.01</td><td>81.78 69.62</td><td>87.80</td><td>78.26</td><td>92.55 86.19</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UniMed-CLIP</td><td>88.84 81.32</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>|82.99 71.45</td><td></td><td>|89.32 80.73</td><td>|93.60 87.98</td></tr><tr><td>UniMed-CLIP + EDD</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>89.46 82.24</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>|81.17 68.91</td><td></td><td>90.4482.60</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>93.39 87.62</td></tr></table>

## 4.4 Evidence Decoupling Visualization Analysis

Fig. 2 shows the evidence maps $\mathbf { e } _ { I } , \ \mathbf { e } _ { T } ,$ , and $\mathbf { e } _ { \mathrm { a d d } }$ from the final decoder layer of CLIP and MedCLIP. Image evidence e<sub>I</sub> concentrates on lesion regions and provides precise spatial localization. Text evidence ${ \bf e } _ { T }$ generates no independent spatial responses but is superimposed on image features as a smooth global modulator, so it cannot autonomously correct spatial deviations in image features. Notably, text evidence can activate textures missing from image evidence. On the BTMRI dataset, MedCLIP’s ${ \bf e } _ { T }$ clearly responds to fine-grained structures that $\mathbf { e } _ { I }$ does not capture, which indicates that text semantics can complement local patterns missing from the visual representation. The additive evidence $\mathbf { e } _ { \mathrm { a d d } }$ remains nearly identical to $\mathbf { e } _ { I } ,$ confirming that image evidence dominates the fused output. Comparing the two models, MedCLIP’s Swin-Tiny backbone yields finer-grained evidence, while for CLIP, text evidence visibly enhances weak visual responses. These visualizations illustrate EDD as a qualitative tool that reveals how image and text interact diferently across VLM encoders and datasets.

![](images/1833b01cc847e7c88e8b109c43445113e7accb7e82ba65db06c907fb30e53aaa.jpg)  
Fig. 2. Decoupled evidence maps from the last decoder layer of CLIP and MedCLIP.

Table 5. Impact of text perturbation on MedCLIP and CLIP segmentation performance (Dice / IoU, %).
<table><tr><td rowspan="2">Model</td><td rowspan="2">Text Condition</td><td>BTMRI</td><td>BUSI</td><td colspan="2">Kvasir-SEG</td><td colspan="2">ISIC</td></tr><tr><td>Dice IoU</td><td>Dice</td><td>IoU</td><td>Dice</td><td>IoU Dice</td><td>IoU</td></tr><tr><td rowspan="4"></td><td>Original</td><td>|91.61 85.42</td><td>|82.43 70.68</td><td>|91.32</td><td>84.06</td><td></td><td>|93.71 88.18</td></tr><tr><td>w/o Text</td><td>40.76 40.58</td><td>54.5046.68</td><td></td><td>78.8669.47</td><td></td><td>89.76 80.16</td></tr><tr><td>MedCLIP Random</td><td>60.05 54.95</td><td>56.06 47.71</td><td></td><td>88.9382.37</td><td></td><td>91.81 85.52</td></tr><tr><td>Missing Location</td><td>90.88 85.62</td><td>71.58 63.47</td><td></td><td>89.60 83.56</td><td></td><td>92.51 86.63</td></tr><tr><td rowspan="5">CLIP</td><td>Contradictory</td><td>14.7312.26</td><td>59.71 51.06</td><td></td><td>88.6882.08</td><td></td><td>91.87 81.93</td></tr><tr><td>Original</td><td>|87.85 80.07</td><td>|81.22 68.88</td><td></td><td>|85.6074.85</td><td></td><td>|92.37 85.85</td></tr><tr><td>w/o Text</td><td>54.25 50.69</td><td>41.6033.83 42.77 35.08</td><td></td><td>64.7854.65</td><td></td><td>49.37 39.32</td></tr><tr><td>Random</td><td>59.09 54.90</td><td>56.10 47.87</td><td></td><td>69.5259.78 75.6366.08</td><td></td><td>85.43 77.01</td></tr><tr><td>Missing Location Contradictory</td><td>82.84 76.21 23.09 15.01</td><td>49.01 39.79</td><td></td><td>58.7646.89</td><td></td><td>90.6483.89 82.08 72.61</td></tr></table>

## 4.5 Text Perturbation Experiments

Visual analysis alone is insuficient to quantify text branch impact on segmentation performance. This section designs a text perturbation experiment using five text conditions during inference, applied to already-trained multimodal segmentation models.

1. Original Text refers to the original, correct text description.

2. w/o Text indicates complete removal of text input.

3. Random Text means a randomly matched, unrelated text description from another sample.

4. Missing Location removes lesion location information from the text.

5. Contradictory Text supplies internally inconsistent descriptions that can confuse the model.

These text perturbation conditions follow the design of MedCLIPSeg [16], where the Missing Location condition removes spatial cues (e.g., lesion position descriptors) from the prompt, and the Contradictory condition supplies internally inconsistent descriptions.

![](images/bbfac3413f168eafdb2ad651dee074246ca26c279d1a70b498a1f183a56f9088.jpg)  
Fig. 3. Box plots of the Dice metric for five models tested on four datasets with text perturbations.

As observed in Table 5, removing text entirely leads to a drastic drop in performance. For example, MedCLIP on BTMRI falls from 91.61% to 40.76% Dice, confirming that text supplies complementary constraints beyond those available from visual features alone. Conversely, supplying contradictory text causes further degradation, with MedCLIP’s Dice on BTMRI falling to 14.73%. However, the sensitivity to text perturbation is strongly dataset-dependent. ISIC demonstrates marked robustness, with MedCLIP retaining 89.76% Dice even without text. In contrast, BTMRI and BUSI exhibit much higher sensitivity to the quality of the accompanying text. These findings support a diferentiated view of text sensitivity rather than a uniform text contribution. On BTMRI and BUSI, text perturbation causes severe performance degradation, indicating strong model reliance on textual input. On ISIC, MedCLIP remains robust without text (89.76% Dice), whereas CLIP experiences a large drop (49.37%), demonstrating that text sensitivity is also model-dependent. On Kvasir-SEG, both models show moderate degradation, with MedCLIP falling to 78.86% and CLIP to 64.78%.

Furthermore, the perturbation patterns reveal that diferent text components play distinct roles across datasets. On BTMRI, removing location descriptors barely afects MedCLIP performance, with Dice moving from 91.61% to 90.88%, yet removing all text causes a catastrophic drop to 40.76%. This suggests that text sensitivity on BTMRI is driven by category-level semantic information and lesion existence cues, rather than by spatial location. On BUSI, missing location descriptors lead to a larger decline, with MedCLIP Dice falling from 82.43% to 71.58%, indicating that spatial location information is more important for ultrasound images. These results show that text sensitivity depends on both the dataset and the specific semantic component. The text components that drive sensitivity in each modality require further investigation.

Fig. 3 further visualizes the distribution of Dice scores across the four datasets under diferent text perturbations. The box plots show that on ISIC and Kvasir-SEG, the Dice values of all models remain relatively stable across text conditions, indicating weak dependence on text guidance. In contrast, on BUSI and BTMRI, performance fluctuates dramatically when text is perturbed, confirming that these datasets exhibit much higher sensitivity to textual input. This stark contrast highlights that the degree of text sensitivity varies considerably across diferent medical imaging modalities and lesion characteristics, aligning with the quantitative findings in Table 5.

## 5 Conclusion

In this paper, we investigated the role of text in multimodal medical image segmentation. Systematic experiments across five VLMs and four datasets revealed that segmentation performance is largely insensitive to fusion module choice, suggesting visual representations dominate current multimodal models. We proposed an Evidence Decoupling Decoder (EDD) that decomposes image and text-modulated evidence while maintaining competitive performance. Perturbation experiments showed that text sensitivity is strongly dataset dependent. On BTMRI and BUSI, removing text caused severe degradation, while on ISIC and Kvasir-SEG text exerted relatively marginal influence. Notably, deleting location descriptors barely afected BTMRI performance but removing all text caused catastrophic failure, indicating category-level semantics rather than spatial cues primarily drive sensitivity on this dataset. We emphasize that perturbation results reflect model sensitivity rather than intrinsic text contribution, and that EDD characterizes learned representations rather than ground-truth attribution. Future work will pursue more rigorous modality attribution methods and adaptive multimodal frameworks.

## References

1. Al-Dhabyani, W., Gomaa, M., Khaled, H., Fahmy, A.: Dataset of breast ultrasound images. Data in Brief 28, 104863 (2020)

2. Chaudhuri, A., Dutta, A., Bui, T., Georgescu, S.: A closer look at multimodal representation collapse. In: Proceedings of the 42nd International Conference on Machine Learning (ICML). pp. 7555–7577. PMLR (2025)

3. Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A.L., Zhou, Y.: Transunet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306 (2021)

4. Chen, L.C., Papandreou, G., Schrof, F., Adam, H.: Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587 (2017)

5. Cheng, J.: Brain tumor dataset (2017). https://doi.org/10.6084/m9.figshare. 1512427.v8

6. Codella, N., Rotemberg, V., Tschandl, P., Celebi, M.E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., Kittler, H., Halpern, A.: Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1902.03368 (2019)

7. Codella, N.C.F., Gutman, D., Celebi, M.E., Helba, B., Marchetti, M.A., Dusza, S.W., Kalloo, A., Liopyris, K., Mishra, N., Kittler, H., Halpern, A.: Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic). In: Proceedings of the IEEE 15th International Symposium on Biomedical Imaging (ISBI). pp. 168–172 (2018)

8. Eslami, S., Meinel, C., de Melo, G.: PubMedCLIP: How much does CLIP benefit visual question answering in the medical domain? In: Findings of the Association for Computational Linguistics: EACL 2023. pp. 1181–1193 (2023). https://doi. org/10.18653/v1/2023.findings-eacl.88

9. Fidon, L., Aertsen, M., Kofler, F., Bink, A., David, A.L., Deprest, T., Emam, D., Gufens, F., Jakab, A., Kasprian, G., Kienast, P., Melbourne, A., Menze, B., Mufti, N., Pogledic, I., Prayer, D., Stuempflen, M., Van Elslander, E., Ourselin, S., Deprest, J., Vercauteren, T.: A dempster-shafer approach to trustworthy ai with application to fetal brain mri segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(5), 3784–3795 (2024). https://doi.org/10. 1109/TPAMI.2023.3346330

10. He, Y., Li, L.: Uncertainty-aware evidential fusion-based learning for semisupervised medical image segmentation. arXiv preprint arXiv:2404.06177 (2024)

11. Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7132–7141 (2018). https://doi.org/10.1109/CVPR.2018.00745

12. Huang, L., Ruan, S., Decazes, P., Denoeux, T.: Deep evidential fusion with uncertainty quantification and contextual discounting for multimodal medical image segmentation. arXiv preprint arXiv:2309.05919 (2023)

13. Jha, D., Smedsrud, P.H., Riegler, M.A., Halvorsen, P., de Lange, T., Johansen, D., Johansen, H.D.: Kvasir-SEG: A segmented polyp dataset. In: Proceedings of the 26th International Conference on Multimedia Modeling (MMM). pp. 451–462 (2020)

14. Khattak, M.U., Kunhimon, S.K., Naseer, M., Khan, S., Khan, F.S.: UniMed-CLIP: Towards a unified image-text pretraining paradigm for diverse medical imaging modalities. arXiv preprint arXiv:2412.10372 (2024)

15. Kim, W., Son, B., Kim, I.: ViLT: Vision-and-language transformer without convolution or region supervision. In: Proceedings of the 38th International Conference on Machine Learning (ICML). vol. 139, pp. 5583–5594 (2021)

16. Koleilat, T., Asgariandehkordi, H., Manzari, O.N., Barile, B., Xiao, Y., Rivaz, H.: MedCLIPSeg: Probabilistic vision-language adaptation for data-eficient and generalizable medical image segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1406–1417 (2026)

17. Kwon, J., Kim, M., Lee, E., Choi, J., Kim, Y.: See-saw modality balance: See gradient, and sew impaired vision-language balance to mitigate dominant modality bias. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 4364–4378 (2025)

18. Lee, C.Y., Xie, S., Gallagher, P., Zhang, Z., Tu, Z.: Deeply-Supervised Nets. In: Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS). vol. 38, pp. 562–570 (2015)

19. Li, H., Nan, Y., Del Ser, J., Yang, G.: Region-based evidential deep learning to quantify uncertainty and improve robustness of brain tumor segmentation. arXiv preprint arXiv:2208.06038 (2022)

20. Li, M., Meng, M., Ye, S., Fulham, M., Bi, L., Kim, J.: Language-guided medical image segmentation with target-informed multi-level contrastive alignments. arXiv preprint arXiv:2412.13533 (2024)

21. Li, Y., Sui, A., Wu, F., Zhuang, X.: Uncertainty-supervised interpretable and robust evidential segmentation. In: Proceedings of the International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) (2025)

22. Li, Z., Li, Y., Li, Q., Wang, P., Guo, D., Lu, L., Jin, D., Zhang, Y., Hong, Q.: LViT: Language meets vision transformer in medical image segmentation. IEEE Transactions on Medical Imaging 43(1), 96–107 (2024). https://doi.org/10.1109/TMI. 2023.3291719

23. Liu, B., Lu, D., Wei, D., Wu, X., Wang, Y., Zhang, Y., Zheng, Y.: Improving medical vision-language contrastive pretraining with semantics-aware triage. IEEE Transactions on Medical Imaging 42(12), 3579–3589 (2023). https://doi.org/10. 1109/TMI.2023.3294980

24. Liu, J., Zhou, H.Y., Li, C., Huang, W., Yang, H., Liang, Y., Wang, S.: MLIP: Medical language-image pre-training with masked local representation learning. arXiv preprint arXiv:2401.01591 (2024)

25. Perez, E., Strub, F., de Vries, H., Dumoulin, V., Courville, A.: FiLM: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI Conference on Artificial Intelligence. pp. 3942–3951 (2018)

26. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the 38th International Conference on Machine Learning. vol. 139, pp. 8748–8763. PMLR (2021)

27. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: Proceedings of the International Conference on Medical Image Computing and Computer-assisted Intervention (MICCAI). pp. 234–241 (2015)

28. Schrodi, S., Hofmann, D.T., Argus, M., Fischer, V., Brox, T.: Two efects, one trigger: On the modality gap, object bias, and information imbalance in contrastive vision-language models. arXiv preprint arXiv:2404.07983 (2024)

29. Sensoy, M., Kaplan, L., Kandemir, M.: Evidential deep learning to quantify classification uncertainty. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 31 (2018)

30. Tomar, N.K., Jha, D., Bagci, U., Ali, S.: TGANet: Text-guided attention for improved polyp segmentation. In: Proceedings of the International Conference on Medical Image Computing and Computer-assisted Intervention (MICCAI). pp. 151–160 (2022)

31. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 30 (2017)

32. Wang, Z., Wu, Z., Agarwal, D., Sun, J.: MedCLIP: Contrastive learning from unpaired medical images and text. In: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 3876–3887 (2022). https://doi.org/10.18653/v1/2022.emnlp-main.256

33. Yang, Z., Wang, J., Tang, Y., Chen, K., Zhao, H., Torr, P.H.: LAVT: Languageaware vision transformer for referring image segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18134–18144 (2022). https://doi.org/10.1109/CVPR52688.2022.01762

34. Yang, Z., Ma, Y., Chen, L.: Progressive uncertainty-guided evidential U-KAN for trustworthy medical image segmentation. arXiv preprint arXiv:2510.08949 (2025)

35. Yaras, C., Chen, S., Wang, P., Qu, Q.: Explaining and mitigating the modality gap in contrastive multimodal learning. arXiv preprint arXiv:2412.07909 (2024)

36. Zhang, S., Xu, Y., Usuyama, N., Xu, H., Bagga, J., Tinn, R., Preston, S., Rao, R., Wei, M., Valluri, N., Wong, C., Tupini, A., Wang, Y., Mazzola, M., Shukla, S., Liden, L., Gao, J., Crabtree, A., Piening, B., Bifulco, C., Lungren, M.P., Naumann, T., Wang, S., Poon, H.: BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915 (2023)

37. Zhou, Z., Rahman Siddiquee, M.M., Tajbakhsh, N., Liang, J.: UNet++: A nested u-net architecture for medical image segmentation. In: Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support. pp. 3–11 (2018)

38. Zou, K., Chen, Y., Huang, L., Zhou, N., Yuan, X., Shen, X., Wang, M., Goh, R.S.M., Liu, Y., Tham, Y.C., Fu, H.: Toward reliable medical image segmentation by modeling evidential calibrated uncertainty. IEEE Transactions on Cybernetics 55(12), 5975–5988 (2025). https://doi.org/10.1109/TCYB.2025.3604432