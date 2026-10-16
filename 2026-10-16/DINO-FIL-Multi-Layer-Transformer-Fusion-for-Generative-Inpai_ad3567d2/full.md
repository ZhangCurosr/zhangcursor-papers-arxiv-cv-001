# DINO-FIL: Multi-Layer Transformer Fusion for Generative Inpainting Forgery Localization

Simon Freitter<sup>1</sup> , David Fischinger<sup>1</sup> , Silvia Poletti<sup>1</sup> , Marcel Hasenbalg<sup>1</sup> , and Martin Boyer<sup>1</sup>

AIT Austrian Institute of Technology GmbH, Giefinggasse 4, 1210 Vienna, Austria simon@freitter.com

Abstract. We propose DINO-FIL, a transformer-based framework for pixel-level localization of generative image forgeries. The method leverages multi-layer representations from a pretrained DINOv3 ViT-H+/16 encoder combined with a lightweight fusion decoder to capture both semantic structure and fine-grained texture inconsistencies introduced by image editing. By aggregating intermediate transformer features across depths, the model improves sensitivity to subtle, spatially localized artifacts that are often missed by standard single-layer approaches. We evaluate DINO-FIL in a cross-dataset setting, training on B-Free and testing on Beyond the Brush, CocoGlide, TGIF, and SAGI-D. The model achieves strong generalization, outperforming existing baselines on average and on four of five evaluated datasets, while maintaining a high true negative rate on pristine images.

Keywords: Inpainting Localization · Image Forensics · DINOv3

## 1 Introduction

Generative image editing has rapidly improved in visual quality and accessibility. Text-guided and mask-guided inpainting systems can now remove, replace, or regenerate selected image regions while preserving global scene coherence, making manipulated content increasingly dificult to recognize from visual inspection alone [5]. Unlike conventional manipulations such as splicing or copy-move editing, generative inpainting can synthesize new content inside a user-specified mask and blend it into the surrounding context. This is particularly challenging in fully regenerated settings, where the edited region is synthesized jointly with its local visual context rather than inserted as an externally pasted region. As a result, traditional image forgery localization methods may not observe clear low-level discontinuities [9,10]. Reliable localization of inpainted regions is therefore an important task for image forensics, media verification, and downstream trustworthiness analysis.

In this work, we address dense image inpainting localization. Given an RGB image, the goal is to predict a pixel-level binary mask that separates inpainted regions from authentic image content. Unlike image-level synthetic detection, dense inpainting localization targets a finer-grained, pixel-level task: identifying edited regions by detecting subtle statistical inconsistencies and visual artifacts, such as boundary discontinuities between authentic and synthesized content. In this setting, “edited” does not necessarily mean that the background pixels remain unchanged. Depending on the editing workflow, the background may either stay pixel-identical to the original image or be regenerated together with the modified region. In the latter case, pixel-level changes can occur across the entire image while preserving the background’s semantic content, potentially destroying the local forensic traces needed for inpainting localization, even though image-level synthetic detection may still succeed [9].

Recent inpainting-focused benchmarks such as TGIF [10], Beyond the Brush (BtB) [1], SAGI-D [5], CocoGlide [6], and B-Free [7] reflect this shift toward realistic local edits.

Pretrained vision backbones have become central to image forensics. Tru-For [6] leverages a pretrained vision transformer architecture to process RGB images together with learned noise-sensitive traces for general forgery localization. In parallel, self-supervised ViT backbones such as DINO [3] provide transferable dense representations that can be adapted to localize subtle generative traces. DinoLizer [2] demonstrates the value of DINOv2 [12] patch embeddings for generative inpainting localization and reports strong cross-dataset performance.

However, existing DINO-based localization methods typically rely on finallayer features [2, 14]. Forensic methods primarily rely on low-level artifacts such as compression and sensor noise [10], while modern vision foundation models encode higher-level semantic structure in image representations, motivating the use of multi-level features. Multi-level fusion is a standard principle in dense prediction [4,8,13]; for ViTs, intermediate hidden states provide a natural hierarchy of local, structural, and semantic information.

We propose DINO-FIL, a dense Forensic Inpainting Localization framework built on a pretrained DINOv3 ViT-H+/16 encoder and a lightweight multilayer fusion decoder. DINO-FIL extracts hidden states from several backbone depths, projects them to a common channel dimension, enriches them with lightweight multi-scale context modules inspired by atrous spatial pyramid pooling (ASPP) [4], and fuses them in a top-down decoder inspired by feature pyramid networks (FPN) [8]. Since all selected DINOv3 hidden states share the same patch-grid resolution, DINO-FIL treats transformer depth, rather than spatial scale, as the fusion hierarchy. This enables the model to combine local appearance cues, texture- and pattern-sensitive intermediate representations, and higher-level semantic context. We further study partial encoder fine-tuning with a lower learning rate for unfrozen DINOv3 layers than for the newly initialized decoder.

In summary, our contributions are threefold:

– We introduce DINO-FIL, a DINOv3-based dense inpainting localization model that fuses intermediate transformer features from multiple backbone depths rather than relying only on the final layer.

– We systematically study encoder adaptation and show that partial finetuning of later DINOv3 layers improves inpainting localization while full encoder fine-tuning provides limited additional benefit.

– We provide a cross-dataset evaluation of our proposed approach against strong forensic baselines and show that high-resolution inference substantially improves inpainting localization, while maintaining low false-positive rates on pristine images.

## 2 Related Work

## 2.1 Pre-trained Vision Transformers in Image Forensics

The adaptation of pre-trained Vision Transformers (ViTs) has emerged as a dominant paradigm for discovering localized anomalies in visual media. A prominent milestone in this space is TruFor [6], which utilizes a pre-trained SegFormer backbone architecture coupled with a learned noise-print stream to map localized forgery traces. While TruFor remains a heavily cited baseline, it is engineered for general image manipulation detection and localization (IMDL). Consequently, its training regimes and evaluation benchmarks encompass a wide variety of traditional manipulations (such as splicing, copy-move, and removal) rather than dedicating focus to the specific artifacts of generative AI inpainting.

Parallel to traditional end-to-end architectures, an emerging paradigm in dense forensic tasks explores leveraging the robust representation spaces of massive vision foundation models. DinoLizer [2] establishes that self-supervised representations from DINOv2 contain rich, localized artifacts useful for dense binary forgery segmentation. To handle arbitrary image dimensions, overlapping image crops are cut out of the target image and all crops are processed by the DINO backbone sequentially. A 1×1 convolutional layer is then used to infer localized logit maps directly from the patch embeddings, which are subsequently aggregated to construct the final heatmap.

Yu et al. [14] introduce a baseline that applies Low-Rank Adaptation (LoRA) to a frozen DINOv3 backbone paired with a lightweight convolutional decoder. Similar to TruFor, Yu et al.’s framework targets broad image forensics across diverse manipulation types rather than focusing exclusively on generative AI inpainting.

## 2.2 Architectural Distinctions in DINO-FIL

Our proposed framework, DINO-FIL, fundamentally departs from these existing DINO-based approaches in two key architectural and task-specific dimensions:

– Feature Extraction Strategy: While both DinoLizer and Yu et al. rely exclusively on extracting features from the final layer of the DINO backbone, DINO-FIL systematically taps into the internal depth hierarchy of a highcapacity ViT-H+ encoder – extracting intermediate hidden states from layers 8, 16, 24, and 32 to capture both low-level texture boundaries and high-level semantic inconsistencies.

Fine-Tuning & Learning Rate Paradigm: While both DinoLizer and Yu et al. evaluate full-parameter fine-tuning of the DINO encoder, their implementations default to training the entire backbone, utilizing a single global learning rate across both the encoder and decoder networks. In contrast, DINO-FIL adopts a more targeted optimization strategy. We investigate both full-parameter fine-tuning and fine-tuning restricted to the later transformer layers, and introduce discriminative learning rates that decouple the adaptation pace of the pretrained backbone from that of the newly initialized decoder. This helps preserve the integrity of the backbone’s foundational features while allowing the decoder to specialize for the localization task.

## 3 Methodology

## 3.1 Model Architecture

DINO-FIL is designed as a dense binary segmentation model for localizing inpainted image regions. As illustrated in Figure 1, given a batch of RGB images $\mathbf { \Phi } _ { x _ { 0 } } ^ { \bullet } \in \mathbb { R } ^ { B \times 3 \times \breve { H } \times W }$ , the images are resized to a square input resolution $S \times S ,$ normalized, and passed through the network. The model outputs a single-channel logit map per image, which is converted with a sigmoid function into an inpainting probability mask $\hat { M } \in [ 0 , 1 ] ^ { B \times 1 \times S \times S }$

The encoder is a pretrained DINOv3 ViT- $\mathrm { \cdot H + } / 1 6$ backbone with patch size $P = 1 6$ . Rather than using only the final transformer output, DINO-FIL extracts intermediate hidden states from layers 8,16,24, and 32. These layers sample the 32-layer backbone at uniform depth intervals. This design is intended to expose the decoder to representations from diferent transformer depths: earlier hidden states can retain more local appearance and spatial detail, intermediate states can provide texture- and pattern-sensitive cues, and deeper states can contribute more semantic context useful for inpainting localization. The degree to which the DINOv3 backbone is fine-tuned is evaluated experimentally by unfreezing diferent numbers of final transformer layers, as described in Section 4.3.

For an input resolution $S \times S$ , the spatial patch grid has resolution $q \times q$ where

$$
q = { \frac { S } { P } } .\tag{1}
$$

Each selected DINO hidden state contains $N = q ^ { 2 }$ spatial patch tokens, followed by five non-spatial tokens: one CLS token and four register tokens. The nonspatial tokens are discarded before reshaping. Thus, for each selected layer $\ell \in$ , the spatial tokens are converted into a patch-grid feature map

$$
f _ { \ell } \in \mathbb { R } ^ { B \times C _ { b } \times q \times q } , \qquad \ell \in \{ 8 , 1 6 , 2 4 , 3 2 \} .\tag{2}
$$

where $C _ { b } = 1 2 8 0$ is the DINO backbone channel dimension.

Each selected feature map is then passed through an independent projection adapter consisting of a 1 \times 1 convolution, group normalization, and GELU activation. The adapter maps the backbone channel dimension from $C _ { b } = 1 2 8 0$ to the decoder channel dimension $C _ { d } = 1 0 2 4$ . This reduces downstream computational cost while preserving the shared $q \times q$ patch-grid resolution.

![](images/d8b62efb91b790e74443d5abf85680deed26265b612928b9d33cf1bdae62c957.jpg)  
Fig. 1: Overview of DINO-FIL. The batch of resized images is encoded with DINOv3, and hidden states from layers $\mathrm { 8 , }$ , 16, 24, and 32 are converted into spatial patch maps $f _ { \ell }$ by reshaping and discarding one CLS token and four register tokens. The projection adapter maps $f _ { \ell }$ to $p _ { \ell } \in \mathbb { R } ^ { B \times C _ { d } \times q \times q }$ . The context block produces $g _ { \ell } \in \mathbb { R } ^ { B \times C _ { c } }$ ×q×q using lightweight multi-scale depthwise-separable convolutions. Before fusion, $g _ { \ell }$ is optionally regularized with spatial dropout, yielding $\tilde { g } _ { \ell }$ during training and the identity mapping otherwise. The resulting $\tilde { g } _ { \ell }$ maps are fused by a horizontal depth-wise topdown decoder on the fixed $q \times q$ patch grid. A 1×1 prediction head produces patch-level logits, which are upsampled and passed through a sigmoid to obtain the final inpainting probability mask. Here, $q = S / P$ $P = 1 6$ ， $C _ { b } = 1 2 8 0$ , and $C _ { d } = 1 0 2 4$

To enrich the transformer features with local multi-scale spatial context, each projected feature map is processed by a lightweight ASPP-inspired context block. Let $p _ { \ell }$ denote the projected feature from layer $\ell .$ The context block applies three parallel depthwise separable $3 \times 3$ convolution branches with dilation rates 1, 2, and 4. The branch outputs are summed and fused with a $1 \times 1$ convolution:

$$
g _ { \ell } = \psi _ { \ell } \Big ( D _ { \ell } ^ { ( 1 ) } ( p _ { \ell } ) + D _ { \ell } ^ { ( 2 ) } ( p _ { \ell } ) + D _ { \ell } ^ { ( 4 ) } ( p _ { \ell } ) \Big ) .\tag{3}
$$

where $D _ { \ell } ^ { ( d ) }$ denotes a depthwise separable convolution with dilation $d ,$ and $\psi _ { \ell }$ denotes the $1 \times 1$ fusion convolution. This context module is useful for inpainting localization because manipulation traces may appear at diferent spatial scales, ranging from fine boundary discontinuities and local texture artifacts to broader semantic or blending inconsistencies.

The resulting context-enhanced features are optionally regularized with spatial dropout before fusion:

$$
\begin{array} { r } { \tilde { g } _ { \ell } = \left\{ \begin{array} { l l } { \mathrm { D r o p o u t 2 d } ( g _ { \ell } ; p ) , } & { p > 0 \mathrm { ~ d u r i n g ~ t r a i n i n g } , } \\ { g _ { \ell } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{4}
$$

where $p$ is the decoder dropout rate. The regularized features are then merged by an FPN-inspired depth-wise top-down fusion decoder. Unlike a conventional Feature Pyramid Network, the selected DINO hidden states do not form a spatial pyramid: all selected layers share the same $q \times q$ patch-grid resolution, so no resizing is required between fusion stages. Instead, the decoder uses transformer depth as the fusion hierarchy, merging the selected hidden states from deepest to shallowest rather than across spatial scales.

Let $\phi _ { \ell }$ denote the learned post-fusion $3 \times 3$ convolution for layer \el . Starting from the deepest selected feature, the decoder progressively adds shallower features and refines each merged representation:

$$
\begin{array} { r c l } { { } } & { { } } & { { m _ { 3 2 } = \phi _ { 3 2 } ( \tilde { g } _ { 3 2 } ) , } } \\ { { } } & { { } } & { { m _ { 2 4 } = \phi _ { 2 4 } ( \tilde { g } _ { 2 4 } + m _ { 3 2 } ) , } } \\ { { } } & { { } } & { { m _ { 1 6 } = \phi _ { 1 6 } ( \tilde { g } _ { 1 6 } + m _ { 2 4 } ) , } } \\ { { } } & { { } } & { { m _ { 8 } = \phi _ { 8 } ( \tilde { g } _ { 8 } + m _ { 1 6 } ) . } } \end{array}\tag{5}
$$

This preserves the top-down fusion principle of FPN while adapting it to a transformer backbone with equal-resolution intermediate states.

Finally, a $1 \times 1$ prediction head maps $m _ { 8 }$ to a single-channel patch-level logit map

$$
z \in \mathbb { R } ^ { B \times 1 \times q \times q } .\tag{6}
$$

The logits are bilinearly upsampled to the resized input resolution $S \times S$ and passed through a sigmoid function:

$$
\hat { M } = \sigma ( \mathrm { U p s a m p l e } ( z ) ) , \qquad \hat { M } \in [ 0 , 1 ] ^ { B \times 1 \times S \times S } .\tag{7}
$$

The predicted mask is supervised with the binary inpainting ground-truth mask using the segmentation loss described in Section 4.2.

## 4 Experimental Setup

## 4.1 Datasets

We evaluate DINO-FIL on a cross-dataset image inpainting localization benchmark. To make the results comparable to DinoLizer, which uses the same datasetlevel training setup, DINO-FIL is trained on B-Free and evaluated on Beyond the Brush, CocoGlide, TGIF, SAGI-D, and B-Free. The reorganized B-Free split contains 206,065 training images, 25,755 validation images, and 25,765 test images. However, for evaluation we follow the benchmark protocol and test on 10,000 B-Free test images. The B-Free test split is included to measure in-domain performance, while the remaining datasets are used to assess cross-dataset generalization. In total, DINO-FIL is tested on 28,835 images, comprising 10,000 B-Free, 3,896 Beyond the Brush, 512 CocoGlide, 4,249 TGIF, and 10,178 SAGI-D test images.

## 4.2 Training Protocol

DINO-FIL is trained as a binary segmentation model for dense inpainting localization. The network predicts a single-channel logit map, which is supervised with the binary ground-truth manipulation mask. The training objective combines binary cross-entropy with logits and Dice loss:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { D i c e } } . } \end{array}\tag{8}
$$

This objective encourages both accurate pixel-wise classification and spatial overlap between the predicted and ground-truth manipulation masks.

Unless otherwise stated, all models are trained using an input resolution of $5 1 2 \times 5 1 2$ . Images are converted to RGB, resized to the target resolution, transformed into tensors, and normalized with ImageNet statistics. During training, we apply random horizontal flipping, color jitter, and B-Free-style image perturbations, including blur, JPEG recompression, down/up-scaling, and mild noise. At test time, only resizing, tensor conversion, and ImageNet normalization are applied.

The decoder is optimized with a learning rate of $1 \times 1 0 ^ { - 4 }$ , while unfrozen encoder layers are optimized with a lower learning rate of $5 \times 1 0 ^ { - 6 }$ . The batch size is set to 12, and the decoder dropout rate is set to 0.05. For all experiments, predicted probability maps are thresholded at 0.5 to obtain binary localization masks.

## 4.3 Encoder Fine-Tuning Ablation

DinoLizer reports that keeping the DINO backbone frozen can improve localization performance. To investigate whether this observation also holds for the proposed DINO-FIL architecture, we conduct an encoder fine-tuning ablation. Specifically, we train variants in which diferent numbers of final DINOv3 transformer layers are unfrozen while the remaining encoder layers stay frozen.

We evaluate six settings: 0, 6, 12, 16, 24, and 32 unfrozen transformer layers. The setting with 0 unfrozen layers corresponds to a fully frozen DINOv3 encoder, while the setting with 32 unfrozen layers fine-tunes the complete transformer backbone. All unfrozen encoder layers use the same encoder learning rate of $5 \times$ $1 0 ^ { - 6 }$ . This experiment tests whether adapting the pretrained DINOv3 backbone features, either partially or end-to-end, improves inpainting localization when used with the proposed decoder. Due to the large number of B-Free training images, all models in this experiment are trained for two epochs.

## 4.4 Test Resolution Evaluation

Although DINO-FIL is trained at $5 1 2 \times 5 1 2$ , we evaluate the trained models at both $5 1 2 \times 5 1 2$ and $1 0 2 4 \times 1 0 2 4$ test resolutions. This experiment measures whether higher-resolution inference improves localization by preserving finer manipulation traces, boundary artifacts, and texture inconsistencies that may be suppressed at lower resolution.

The comparison is particularly relevant because the benchmark datasets differ in their native image resolutions. B-Free consists of fixed $5 1 2 \times 5 1 2$ images, whereas other datasets contain images with more diverse resolutions and aspect ratios. Consequently, evaluating only at $5 1 2 \times 5 1 2$ may underestimate performance on datasets where additional high-frequency evidence is available in the original image.

## 4.5 Training Resolution Evaluation

To further study the role of image resolution, we conduct a separate trainingresolution experiment using SAGI-D as the training dataset. SAGI-D is suitable for this analysis because it contains images with a broad range of resolutions, including images smaller than 512 px and images larger than $1 0 2 4 \ \mathrm { p x }$ . This contrasts with B-Free and CocoGlide, whose images have fixed resolutions of $5 1 2 \times 5 1 2$ and $2 5 6 \times 2 5 6$ , respectively.

In this experiment, we compare models trained at $5 1 2 \times 5 1 2$ and $1 0 2 4 \times 1 0 2 4$ The $( 5 1 2 \times 5 1 2 )$ -trained model is evaluated at both $5 1 2 \times 5 1 2$ and $1 0 2 4 \times 1 0 2 4 .$ while the $( 1 0 2 4 \times 1 0 2 4 )$ -trained model is evaluated at 1024 × 1024. B-Free and CocoGlide are excluded from this experiment because their fixed low resolutions do not provide additional image information at $1 0 2 4 \times 1 0 2 4 ;$ upsampling would merely duplicate existing pixels rather than introducing meaningful highresolution evidence. Due to the large number of SAGI-D training images, all models in this experiment are trained for two epochs.

## 4.6 Evaluation Metrics

To enable direct comparison with DinoLizer, we follow the same pixel-level evaluation protocol for manipulated images and report Intersection over Union (IoU) and F1 score. The predicted probability map is binarized using a fixed threshold of 0.5. Let $T P , F P ,$ and F N denote the number of true-positive, false-positive, and false-negative pixels, respectively. IoU is computed as shown in Eq. (9).

$$
{ \mathrm { I o U } } = { \frac { T P } { T P + F P + F N } } .\tag{9}
$$

Following DinoLizer, F1 is computed from precision and recall with a small stabilization constant $\epsilon = 1 0 ^ { - 8 }$

$$
\mathrm { P r e c i s i o n } = { \frac { T P } { T P + F P + \epsilon } } , \qquad \mathrm { R e c a l l } = { \frac { T P } { T P + F N + \epsilon } } .\tag{10}
$$

$$
\mathrm { F 1 } = 2 \cdot \frac { \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } + \epsilon } .\tag{11}
$$

Both IoU and F1 are computed independently for each manipulated image and then macro-averaged over all manipulated test images. Thus, each image contributes equally to the final score, independent of the size of the manipulated region. Since IoU and F1 are averaged separately after being computed per image, the final mean scores do not necessarily preserve the one-to-one monotonic relationship that holds for a single fixed set of TP, FP, and FN values. Consequently, it is possible for one model to obtain a slightly lower mean IoU but a higher mean F1 than another model.

In addition to manipulated-image localization, we evaluate false positives on pristine images. For this analysis, we use original, non-manipulated images from the SAGI-D test split, consisting of COCO, Open Images, and RAISE images. We report the pixel-level false-positive rate and the image-level true-negative rate. An image is counted as a true negative if no pixel exceeds the decision threshold of 0.5.

## 4.7 Comparison with Existing Detectors

We compare DINO-FIL against DinoLizer and TruFor, the two strongest-performing methods among the six detectors compared in the DinoLizer benchmark. For both baselines, we use the results reported in the DinoLizer paper. The reported baseline results are not based on externally downsampling all test images to a fixed resolution before inference: DinoLizer avoids downsampling through sliding-window inference, and TruFor processes arbitrary-size inputs and produces localization maps at input resolution. Since DinoLizer reports separate results for SAGI-SP and SAGI-FR, whereas we evaluate SAGI-D as a unified dataset, we aggregate the reported SAGI-SP and SAGI-FR scores into a single SAGI-D score using an image-count-weighted average.<sup>1</sup>

## 5 Results

## 5.1 Encoder Fine-Tuning and Test Resolution

Table 1 reports the localization performance of DINO-FIL when diferent numbers of final DINOv3 transformer layers are unfrozen and the model is evaluated at diferent test image resolutions. The results show that adapting the encoder yields higher test performance for the proposed architecture. With a fully frozen encoder, DINO-FIL achieves an average IoU/F1 of 0.51/0.58 at a test resolution of 512 × 512 and 0.58/0.67 at 1024 × 1024. At 1024 × 1024, unfreezing later transformer blocks consistently improves performance up to 24 unfrozen layers, although the gains become smaller as more layers are unfrozen. At 1024 × 1024 resolution, the model reaches its peak average IoU/F1 with 24 unfrozen layers. Fully unfreezing all 32 layers does not further improve the average result and slightly decreases the average F1 before rounding, from 0.774 to 0.771. This suggests that partial fine-tuning of the upper DINOv3 layers is suficient, while full encoder fine-tuning provides diminishing returns.

Table 1: Comparison of DINO-FIL variants under diferent numbers of unfrozen encoder layers and test image resolutions (training resolution is fixed at 512px), evaluated using average pixel-level IoU and F1 score on image forgery localization. The best results are shown in bold, and the second-best results are underlined.
<table><tr><td rowspan="2">Test image resolution</td><td rowspan="2">Unfrozen layers</td><td colspan="2">BtB</td><td colspan="2">CocoGlide</td><td colspan="2">TGIF</td><td colspan="2">SAGI-D</td><td colspan="2">B-Free</td><td colspan="2">Average</td></tr><tr><td>IoU</td><td>F1</td><td>IoU</td><td>F1</td><td>IoU</td><td>F1</td><td>IoU</td><td>F1</td><td>IoU</td><td>F1</td><td>IoU</td><td>F1</td></tr><tr><td rowspan="6">512px</td><td>0</td><td>0.23</td><td>0.31</td><td>0.75</td><td>0.82</td><td>0.27</td><td>0.33</td><td>0.47</td><td>0.55</td><td>0.82</td><td>0.89</td><td>0.51</td><td>0.58</td></tr><tr><td>6</td><td>0.33</td><td>0.41</td><td>0.77</td><td>0.85</td><td>0.38</td><td>0.45</td><td>0.52</td><td>0.61</td><td>0.86</td><td>0.92</td><td>0.57</td><td>0.65</td></tr><tr><td>12</td><td>0.36</td><td>0.44</td><td>0.81</td><td>0.87</td><td>0.38</td><td>0.44</td><td>0.58</td><td>0.66</td><td>0.89</td><td>0.94</td><td>0.60</td><td>0.67</td></tr><tr><td>16</td><td>0.36</td><td>0.43</td><td>0.83</td><td>0.88</td><td>0.38</td><td>0.43</td><td>0.59</td><td>0.67</td><td>0.90</td><td>0.94</td><td>0.61</td><td>0.67</td></tr><tr><td>24</td><td>0.32</td><td>0.38</td><td>0.84</td><td>0.89</td><td>0.33</td><td>0.37</td><td>0.57</td><td>0.64</td><td>0.91</td><td>0.95</td><td>0.60</td><td>0.65</td></tr><tr><td>32</td><td>0.33</td><td>0.40</td><td>0.87</td><td>0.92</td><td>0.36</td><td>0.40</td><td>0.58</td><td>0.64</td><td>0.91</td><td>0.95</td><td>0.61</td><td>0.66</td></tr><tr><td rowspan="6">1024px</td><td>0</td><td>0.38</td><td>0.50</td><td>0.75</td><td>0.83</td><td>0.42</td><td>0.51</td><td>0.53</td><td>0.64</td><td>0.80</td><td>0.88</td><td>0.58</td><td>0.67</td></tr><tr><td>6</td><td>0.47</td><td>0.57</td><td>0.77</td><td>0.85</td><td>0.47</td><td>0.56</td><td>0.57</td><td>0.66</td><td>0.83</td><td>0.90</td><td>0.62</td><td>0.71</td></tr><tr><td>12</td><td>0.54</td><td>0.64</td><td>0.84</td><td>0.90</td><td>0.52</td><td>0.59</td><td>0.63</td><td>0.71</td><td>0.88</td><td>0.93</td><td>0.68</td><td>0.75</td></tr><tr><td>16</td><td>0.56</td><td>0.65</td><td>0.85</td><td>0.91</td><td>0.54</td><td>0.61</td><td>0.66</td><td>0.73</td><td>0.89</td><td>0.94</td><td>0.70</td><td>0.77</td></tr><tr><td>24</td><td>0.59</td><td>0.68</td><td>0.87</td><td>0.92</td><td>0.57</td><td>0.63</td><td>0.66</td><td>0.73</td><td>0.87</td><td>0.92</td><td>0.71</td><td>0.77</td></tr><tr><td>32</td><td>0.57</td><td>0.66</td><td>0.88</td><td>0.93</td><td>0.58</td><td>0.64</td><td>0.65</td><td>0.72</td><td>0.88</td><td>0.92</td><td>0.71</td><td>0.77</td></tr></table>

A second clear trend is the efect of test resolution. Evaluating at 1024×1024 substantially improves performance on the cross-dataset test sets. Comparing the best-performing configurations at each resolution, the average IoU/F1 increases from 0.61/0.67 at 512×512 to 0.71/0.77 at 1024×1024. This improvement is most pronounced on datasets with larger or more diverse native image resolutions: Beyond the Brush increases from 0.36/0.44 to 0.59/0.68, TGIF from 0.38/0.45 to 0.58/0.64, and SAGI-D from 0.59/0.67 to 0.66/0.73. These results suggest that higher-resolution inference can preserve manipulation traces, boundary inconsistencies, and local texture artifacts that may be suppressed when images are resized to 512 × 512, making 1024 × 1024 evaluation better aligned with the available image evidence for higher-resolution datasets.

CocoGlide also shows a small improvement, increasing from 0.87/0.92 to 0.88/0.93. However, this gain should be interpreted cautiously because CocoGlide images have a fixed resolution of 256 × 256 pixels. Evaluating these images at 1024 × 1024 therefore does not introduce additional high-frequency information, but only upsamples the existing pixels. The small observed increase may consequently be due to interpolation efects or random variation rather than a genuine resolution benefit.

The only exception is B-Free, where 512 × 512 evaluation performs slightly better than 1024 × 1024 evaluation. This is expected because B-Free consists of fixed $5 1 2 \times 5 1 2$ images and is also the training dataset for this experiment. As with CocoGlide, testing B-Free at 1024 × 1024 does not introduce additional image information, but instead upsamples the same image content.

Table 2: Comparison of DINO-FIL (ours) with DinoLizer and TruFor, the two strongest-performing methods reported in the DinoLizer benchmark [2]. Performance is reported using average pixel-level IoU and F1 score for image forgery localization. The SAGI-D results for DinoLizer and TruFor are computed as image-count-weighted averages of the reported SAGI-SP and SAGI-FR results. The best results are shown in bold, and the second-best results are underlined.
<table><tr><td>Model</td><td>BtB IoU F1</td><td>CocoGlide IoU F1</td><td>TGIF IoU F1</td><td>SAGI-D IoU</td><td>F1</td><td>B-Free F1</td><td>Average IoU</td><td>F1</td></tr><tr><td>DINO-FIL (ours)</td><td>0.59 0.68</td><td>0.87 0.92</td><td></td><td>0.57 0.63</td><td>0.66 0.73</td><td>IoU 0.87 0.92</td><td></td><td>0.71 0.77</td></tr><tr><td>DinoLizer</td><td>0.42 0.56</td><td>0.57 0.69</td><td></td><td>0.390.49</td><td>0.38 0.49</td><td>0.70</td><td>0.81</td><td>0.49 0.61</td></tr><tr><td>TruFor</td><td>0.22 0.30</td><td>0.29 0.36</td><td>0.70 0.80</td><td></td><td>0.35 0.41</td><td>0.22</td><td>0.30</td><td>0.360.43</td></tr></table>

Overall, the best average performance is obtained by the model with 24 unfrozen layers evaluated at 1024×1024, achieving an average IoU/F1 of 0.71/0.77, with the fully fine-tuned model with 32 unfrozen layers obtaining nearly identical average performance. This indicates that partial fine-tuning of the final DINOv3 layers is suficient for strong localization performance, while full end-to-end finetuning does not provide a clear additional benefit.

## 5.2 Comparison with Existing Detectors

Table 2 compares DINO-FIL with DinoLizer and TruFor. For this comparison, we use the strongest-performing DINO-FIL variant from Table 1, which unfreezes the final 24 DINOv3 transformer layers and is evaluated at 1024×1024 resolution. DINO-FIL achieves the best average performance across the benchmark, with an average IoU/F1 of 0.71/0.77. This is substantially higher than DinoLizer, which obtains 0.49/0.61, and TruFor, which obtains 0.36/0.43. DINO-FIL also achieves the best performance on four of the five evaluated datasets: Beyond the Brush, CocoGlide, SAGI-D, and B-Free.

The largest gains over DinoLizer are observed on CocoGlide and SAGI-D. On CocoGlide, DINO-FIL improves IoU from 0.57 to 0.87 and F1 from 0.69 to 0.92. On SAGI-D, DINO-FIL improves IoU from 0.38 to 0.66 and F1 from 0.49 to 0.73. These results indicate that the proposed multi-layer DINOv3 feature fusion decoder generalizes well across inpainting datasets and benefits from highresolution inference.

TruFor achieves the strongest result on TGIF, with an IoU/F1 of 0.70/0.80, while DINO-FIL obtains 0.57/0.63. This is the only dataset on which DINO-FIL does not obtain the best result. Nevertheless, DINO-FIL remains the secondbest method on TGIF and outperforms TruFor by a large margin on all other datasets.

The comparison should be interpreted as a benchmark-level comparison rather than a fully controlled retraining study. The DinoLizer and TruFor scores are taken from the DinoLizer benchmark. TruFor, however, was not trained on TGIF, as stated in the TruFor paper. Despite this limitation, the results show that DINO-FIL provides strong cross-dataset localization performance relative to the best-performing methods reported in the DinoLizer benchmark.

Table 3: False-positive evaluation of DINO-FIL on original, non-manipulated images. We report the pixel-level false-positive rate and the image-level true-negative rate. The average is computed over all images across the three datasets.
<table><tr><td>Dataset</td><td>Number of Images</td><td>Pixel-level FP Rate</td><td>Image-level TN Rate</td></tr><tr><td>COCO</td><td>2,922</td><td>0.0019</td><td>0.9767</td></tr><tr><td>Open Images</td><td>5,579</td><td>0.0040</td><td>0.9684</td></tr><tr><td>RAISE</td><td>580</td><td>0.0021</td><td>0.9810</td></tr><tr><td>Average</td><td>9,081</td><td>0.0032</td><td>0.9719</td></tr></table>

## 5.3 False Positives on Pristine Images

Table 3 evaluates DINO-FIL on pristine, non-manipulated images from the SAGI-D test split. For this evaluation, we use the strongest-performing DINO-FIL variant from Table 1, which unfreezes the final 24 DINOv3 transformer layers and is evaluated at 1024 × 1024 resolution. Across 9,081 original images from COCO, Open Images, and RAISE, the model obtains a pixel-level falsepositive rate of 0.0032 and an image-level true-negative rate of 0.9719 at the fixed threshold of 0.5.

DINO-FIL achieves high true-negative rates across all three pristine-image subsets, with 97.67% on COCO, 96.84% on Open Images, and 98.10% on RAISE. These results suggest that DINO-FIL is not only efective at localizing manipulated regions but also relatively conservative on pristine images. However, a low false-positive rate does not by itself guarantee that every predicted region in a manipulated image is spatially correct. Rather, it indicates that, under the chosen threshold, predicted inpainted regions are uncommon on pristine-image inputs.

## 5.4 Efect of Training Resolution

Motivated by the superior performance observed when evaluating DINO-FIL at 1024 × 1024, Table 4 studies whether this benefit also transfers to training at higher resolution. All models in this experiment are based on the strongestperforming encoder configuration identified in Table 1, i.e., DINO-FIL with the final 24 DINOv3 transformer layers unfrozen.

This experiment is independent from the B-Free training experiment because the models are trained on SAGI-D instead. Therefore, the absolute numbers should not be compared directly with the B-Free-trained models in Table 1.

The results show that high-resolution inference also improves performance for DINO-FIL trained on SAGI-D at $5 1 2 \times 5 1 2$ . For the model trained at this resolution, average IoU/F1 increases from 0.51/0.56 when evaluated at 512×512 to 0.71/0.78 when evaluated at 1024 × 1024.

Table 4: Comparison of DINO-FIL variants trained at diferent resolutions (512px and 1024px) and evaluated at corresponding test resolutions for image forgery localization. Performance is reported in terms of average pixel-level IoU and F1 score across datasets. The best results are shown in bold, and the second-best results are underlined.
<table><tr><td rowspan="2">Train Resolution</td><td rowspan="2">Test Resolution</td><td colspan="2">Beyond the Brush</td><td colspan="2">TGIF</td><td colspan="2">SAGI-D</td><td colspan="2">Average</td></tr><tr><td>IoU</td><td>F1</td><td>IoU</td><td>F1</td><td>IoU</td><td>F1</td><td>IoU</td><td>F1</td></tr><tr><td rowspan="2">512px</td><td>512px</td><td>0.37</td><td>0.44</td><td>0.41</td><td>0.44</td><td>0.76</td><td>0.81</td><td>0.51</td><td>0.56</td></tr><tr><td>1024px</td><td>0.68</td><td>0.76</td><td>0.62</td><td>0.69</td><td>0.83</td><td>0.88</td><td>0.71</td><td>0.78</td></tr><tr><td>1024px</td><td>1024px</td><td>0.64</td><td>0.71</td><td>0.55</td><td>0.62</td><td>0.84</td><td>0.88</td><td>0.68</td><td>0.74</td></tr></table>

However, training directly at 1024 × 1024 does not further improve average performance in this experiment. The model trained and tested at 1024 × 1024 achieves an average IoU/F1 of 0.68/0.74, which is below the 512 × 512-trained model evaluated at 1024 × 1024. The only dataset where 1024 × 1024 training slightly improves IoU is SAGI-D, where IoU increases from 0.83 to 0.84 while F1 remains unchanged at 0.88.

Several factors may explain this result. First, most SAGI-D images are not substantially larger than 1024 px: only about 21% have a maximum side length of at least 1024 px, while the majority fall between 512 px and 1024 px. Consequently, training at 1024 × 1024 may not consistently expose the model to genuinely higher-resolution evidence.

Second, training at 512×512 may act as a form of scale regularization. When combined with 1024 × 1024 inference, the model may still exploit high-frequency cues in the test images while retaining stronger cross-dataset generalization from the lower-resolution training setting.

These findings suggest that high-resolution inference is beneficial for DINO-FIL, whereas high-resolution training does not provide an additional benefit under the fixed-resolution setup evaluated here. This indicates that the relationship between training resolution, test resolution, and dataset image statistics requires further investigation.

## 6 Conclusion

This paper presented DINO-FIL, a dense inpainting localization model that combines intermediate DINOv3 representations with a lightweight multi-layer feature fusion decoder. Instead of relying only on the final transformer output, DINO-FIL exploits hidden states from multiple backbone depths and fuses them through projection, context aggregation, and top-down refinement. This design allows the model to combine local appearance cues, texture-sensitive intermediate representations, and deeper semantic information for pixel-level inpainting localization.

The experimental results show that DINO-FIL achieves strong cross-dataset generalization when trained on B-Free and evaluated on five inpainting localization benchmarks. Partial fine-tuning of the DINOv3 encoder improves performance over a fully frozen backbone, with the best average result obtained by unfreezing the final 24 transformer layers and evaluating at 1024 × 1024 resolution. Compared with DinoLizer and TruFor, DINO-FIL achieves the highest average IoU and F1 score and obtains the best performance on four of the five evaluated datasets. The false-positive analysis on pristine SAGI-D images further shows that the model remains conservative on non-manipulated inputs, achieving a low pixel-level false-positive rate and a high image-level true-negative rate.

The resolution experiments indicate that high-resolution inference is particularly important for inpainting localization. Evaluating at 1024 × 1024 substantially improves performance on datasets with larger or more diverse native image resolutions, suggesting that fine manipulation traces, boundary artifacts, and texture inconsistencies can be suppressed by lower-resolution resizing. However, training directly at 1024 × 1024 does not provide a consistent additional benefit in the fixed-resolution setup evaluated here. Instead, it yields slightly lower average performance than training at 512 × 512 and evaluating at 1024 × 1024. This suggests that the relationship between training resolution, test resolution, and dataset image statistics is more nuanced than simply increasing both resolutions.

## 6.1 Future Work

The observation that high-resolution inference is beneficial, whereas high-resolution training does not provide an additional benefit under the fixed-resolution setup evaluated here, suggests that the interaction between training resolution, test resolution, and dataset image statistics requires further investigation.

Future work should therefore investigate adaptive-resolution training and evaluation strategies. One possible approach is to group images according to their native resolution and process them at higher target resolutions only when suficient high-resolution evidence is available. Since the DINOv3 backbone uses a patch size of 16, these target resolutions should be chosen as multiples of 16 to avoid implicit cropping of the input image as described in the DINOv3 model card [11].

Another promising direction is parameter-eficient adaptation of the backbone using Low-Rank Adaptation (LoRA). Compared to full or partial finetuning of transformer layers, LoRA introduces trainable low-rank updates while keeping the pretrained weights frozen, potentially improving eficiency and reducing overfitting. Yu et al. [14] explored LoRA-style adaptation for DINO-based forensic backbones, suggesting that it may improve performance compared to full fine-tuning. Integrating LoRA with the proposed multi-layer feature fusion architecture is therefore a promising direction for enhancing the DINO’s representational capacity.

## Acknowledgements

This work was supported by the GADMO 2.0 project, co-funded by the European Union’s Digital Europe Programme (DIGITAL) and the Austrian Research Promotion Agency (FFG), under grant agreement No. 101083573.

## References

1. Bertazzini, G., Albisani, C., Baracchi, D., Shullani, D., Piva, A.: Beyond the brush: Fully-automated crafting of realistic inpainted images. In: 2024 IEEE International Workshop on Information Forensics and Security (WIFS). pp. 1–6. IEEE (2024)

2. Butora, J., Itier, V., Boulanger, J., Bas, P., et al.: Dinolizer: Learning from the best for generative inpainting localization. arXiv preprint arXiv:2511.20722 (2025)

3. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

4. Chen, L.C., Papandreou, G., Schrof, F., Adam, H.: Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587 (2017)

5. Giakoumoglou, P., Karageorgiou, D., Papadopoulos, S., Petrantonakis, P.C.: Sagi: Semantically aligned and uncertainty guided ai image inpainting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16090–16101 (2025)

6. Guillaro, F., Cozzolino, D., Sud, A., Dufour, N., Verdoliva, L.: Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 20606–20615 (2023)

7. Guillaro, F., Zingarini, G., Usman, B., Sud, A., Cozzolino, D., Verdoliva, L.: A biasfree training paradigm for more general ai-generated image detection. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 18685–18694 (2025)

8. Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2117–2125 (2017)

9. Mareen, H., Karageorgiou, D., Giakoumoglou, P., Lambert, P., Papadopoulos, S., Van Wallendael, G.: Tgif2: extended text-guided inpainting forgery dataset and benchmark. Journal on Information Security 2026(1), 13 (2026)

10. Mareen, H., Karageorgiou, D., Van Wallendael, G., Lambert, P., Papadopoulos, S.: Tgif: Text-guided inpainting forgery dataset. In: 2024 IEEE International Workshop on Information Forensics and Security (WIFS). pp. 1–6. IEEE (2024)

11. Meta AI: DINOv3 model card. https://github.com/facebookresearch/dinov3/ blob/main/MODEL\_CARD.md (2025), accessed: 2026-06-25

12. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

13. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

14. Yu, J., Feng, Q., Wang, Z., Ma, X.: Dinov3 beats specialized detectors: A simple foundation model baseline for image forensics. arXiv preprint arXiv:2604.16083 (2026)