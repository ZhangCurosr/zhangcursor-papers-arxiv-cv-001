# BruNet: A Cross-Domain Transfer Framework for Bruise Segmentation

Qiming Wang   
Cardiff University   
United Kingdom   
WangQ79@cardiff.ac.uk

Richard J. Motley Cardiff University United Kingdom Richard.Motley@wales.nhs.uk

Xianfang Sun Cardiff University United Kingdom SunX2@cardiff.ac.uk

Ebube E. Obi Cardiff University United Kingdom ObiE@cardiff.ac.uk

Paul L. Rosin   
Cardiff University   
United Kingdom   
RosinPL@cardiff.ac.uk

## Abstract

Segmenting bruises is a challenging task in medical imaging due to limited data and annotations, diffuse boundaries, and highly variable appearance. In this work, we propose BruNet, a segmentation framework that combines a ViT-based visual encoder (a self-supervised DINOv3 or a pretrained LingBot-Vision backbone) with a SAM-based mask decoder. BruNet is trained on the HAM10000 skin lesion dataset and evaluated on a separate bruise dataset without additional fine-tuning. Although a small number ofprior studies have explored machine learning and computer vision for bruise analysis, existing work has primarily focused on detection, classification, or colour analysis rather than pixel-level localisation. To the best of our knowledge, this is the first study to address automatic bruise segmentation. Our results show that BruNet outperforms CNN-based models, state-of-the-art segmentation models, ChatGPT-4o/5-assisted SAM2 zero-shot baselines, and the medical-oriented MedSAM model, demonstrating strong cross-domain generalisation to bruise segmentation.

## 1. Introduction

Ever since the development of convolutional neural networks (CNNs) and Vision Transformers (ViTs) [8], they have replaced traditional statistical and machine learning (ML) methods and become one of the mainstream methods for medical image segmentation tasks. Architectures like U-Net [26] and UNETR [12] demonstrated strong performance on such tasks. However, they often lack generalisability and often require a large amount of annotated high-quality data, which is usually expensive in real-world scenarios, especially when dealing with data-scarce targets such as bruises. Foundation models such as CLIP [24] and SAM [17] offer zero-shot capabilities through semantic alignment and prompt-based segmentation, yet CLIP lacks localisation, and both SAM and CLIP often fail in clinical cases.

Bruise analysis is a representative yet underexplored challenge for medical imaging. Bruises, also referred to as ‘Contusions’ or ‘Ecchymoses’ in clinical contexts, usually exhibit substantial variability in appearance across individuals and over time, with diffuse, poorly standardised boundary definitions that complicate accurate delineation. Moreover, there is a lack of publicly available annotated datasets for bruises, partly because such injuries are often self-managed and rarely clinically documented, particularly in settings such as contact sports where they occur frequently [11]. However, bruising is frequently encountered in clinical practice and may reflect underlying disorders of haemostasis (a process to prevent and stop bleeding), vascular integrity, or systemic disease [31]. In addition, bruises play a critical role in forensic and legal contexts, where accurate documentation and assessment can provide important evidence in cases of physical violence [21, 37]. With the assistance of computer vision and machine learning, analysing such non-standardised tasks can significantly reduce bias compared to a manual process; the related research remains extremely limited, with existing studies focusing primarily on age classification [32] or detection using bounding boxes [1], rather than accurate localisation or segmentation.

In this work, we propose a segmentation framework that integrates a ViT-based backbone, and a SAM decoder to enable cross-domain generalisation. The model is trained on the HAM10000 skin lesion dataset and evaluated on a separate bruise dataset. This method is evaluated against CNNbased baselines, state-of-the-art (SOTA) vision transformer (ViT) segmentation models, and prompt-driven SAM2 variants, including its medical-oriented model MedSAM with MedGemma as prompter [20, 27]. Results show the segmentation robustness under domain shift, highlighting the potential of cross-domain foundation models for dermatological and forensic imaging tasks.

This work also presents the first deep learning framework for bruise segmentation, along with a curated bruise dataset with expert annotations and a novel dual-region protocol for modelling boundary uncertainty.

Our contributions are summarised as follows:

• We introduce a dual-region annotation protocol for bruise segmentation that separates high-confidence bruise core regions from uncertain boundary regions.

• We curate and evaluate on an 86-image bruise dataset with expert-reviewed dual-region annotations.

• We propose BruNet, a cross-domain segmentation framework trained only on HAM10000 skin lesion masks and evaluated without bruise-specific fine-tuning.

• We compare BruNet against CNN, ViT, promptbased SAM2, and medical-oriented baselines under an uncertainty-aware evaluation protocol.

## 2. Related Works

## 2.1. Vision Transformer and Its Medical Uses

With the development of deep neural networks (DNNs), they are playing an increasingly vital role in the medical imaging field. Early in this trend, convolutional neural networks like U-Net [26] for segmentation, and ResNet [13] for image classification played a dominant role in medical image analysis. However, with the recent development of transformers [36], Vision Transformers (ViTs) [8] with their self-attention mechanisms as backbones, in combination with CNNs [12], are starting to outperform traditional CNN models. Despite their success, most Transformerbased methods still rely heavily on supervised training with large amounts of data, limiting their applicability in datascarce domains such as rare diseases.

Self-supervised pretraining addresses this limitation: DI-NOv3 [28] produces frozen dense representations that transfer to segmentation without labelled data, while LingBot-Vision [10] additionally supervises spatial structure through masked boundary modelling, in which the student reconstructs the semantics and geometry of teacher-identified boundary patches. Both are benchmarked on natural images with well-defined object boundaries, and whether their advantages persist for diffuse, low-contrast targets such as bruises remains untested.

## 2.2. Segment Anything Model

The Segment Anything Model (SAM) [17] is a promptable, zero-shot segmentation framework that generalises across domains without task-specific training. SAM supports various inputs as prompts, such as bounding boxes, points, and masks. While SAM demonstrates impressive zero-shot performance on natural images, it underperforms in medical contexts due to differences in features between common objects and disease boundaries. Thus, models like Med-SAM [20] were developed by fine-tuning SAM on a largescale (over 1.5M), annotated medical segmentation dataset. These models improve boundary fidelity and prompt alignment, yet still inherit SAM’s underlying assumptions about objectness and edge contrast, which are less reliable in the context of diffuse boundaries. More recently, SAM 3 extends SAM with open-vocabulary text and exemplar prompts, allowing it to detect and segment objects specified by semantic concepts. This improves its suitability for zero-shot segmentation of previously unseen targets without requiring manually defined spatial prompts [5].

## 2.3. Computer Vision for Dermatologic Analysis

Although there is rarely any research done on bruise analysis for computer vision, there are many well-studied works in a similar field. Besides skin lesions, deep learning has been applied to wound classification to support wound assessment and treatment planning [2]. Skin burn level classification used to be a challenging task for computers, but recent work has achieved high accuracy in identifying burn severity from skin burn images [30].

However, these conditions are not fully equivalent to bruises. Many skin lesions, wounds, and burns manifest as surface-visible abnormalities, often with clearer texture, structural disruption, or boundary cues. Bruises, in contrast, are typically caused by damage to blood vessels beneath the skin, producing subsurface colour changes that may be diffuse, low-contrast, and gradually blended into surrounding tissue. This makes bruise boundaries less well-defined and more difficult to annotate consistently. Therefore, although skin lesion, wound, and burn datasets provide useful visual priors for skin-region analysis, bruise segmentation presents a distinct challenge due to its weaker surface structure and higher boundary ambiguity.

Despite the advances in many dermatology analysis tasks, the success of deep learning methods is heavily dependent on the availability of large, high-quality annotated datasets. Such datasets are more widely available for skin lesions, wounds, and burns [9, 19]. For bruises, data is particularly difficult to obtain. This is largely due to the fact that bruises are often self-managed and under-reported, and in some cases associated with sensitive contexts such as domestic violence, where ethical and privacy constraints limit data sharing and annotation [25].

## 2.4. Computer Vision for Bruise Analysis

Bruises remain a challenging target for machine learning due to their diffuse appearance, inter-subject variability, and sensitivity to lighting, skin tone, as well as skin thickness. Existing work on bruise analysis can be broadly categorised into detection/localisation and feature analysis (e.g. age estimation), while traditional visual assessment by humans remains subjective and prone to bias.

Early studies have explored the use of deep learning for related forensic tasks, including domestic violence classification [22] and injury recognition [4]. More targeted work by Aminfar et al. [1] adapted lightweight models for bruise detection and examined bias across skin tone classifications [7], highlighting disparities in performance. In parallel, Tirado and Mauricio [32] demonstrated the feasibility of bruise age estimation using CNNs, achieving high accuracy on a large dataset.

Despite these advances, existing approaches focus primarily on detection or classification, with limited attention to precise pixel-level segmentation. This highlights a significant gap in applying modern computer vision methods to bruise localisation.

## 3. Methods

In this paper, we propose a structure-aware segmentation framework that integrates a pre-trained vision transformer backbone and the SAM mask decoder to enable promptable, semantically guided segmentation. The model is trained on a pixel-level labelled skin lesion dataset (HAM10000) and evaluated on a separate bruise dataset to assess Zero-Shot Transfer and generalisability under annotation ambiguity, see Fig. 1.

## 3.1. Overview

Our architecture consists of three main components: (1) a visual backbone — LingBot-Vision ViT-B/16 (maskedboundary pretraining, 512×512 input) or DINOv3 ViT-B/16 (self-supervised, 224×224 input) — as the visual encoder [10, 28], (2) a convolutional upsampling adapter, and (3) the SAM [17] (supervised) prompt encoder and mask decoder.

The visual encoder extracts patch-level features from the input image. Because ViT tokens form a coarse spatial grid, the two backbones emit embeddings at very different resolutions: with a patch size of 16, LingBot-Vision produces a $3 2 \times 3 2$ token grid at its native 512 × 512 input, while DINOv3 produces a 14 × 14 grid at 224 × 224. Reshaping either grid directly into a dense embedding yields a low-resolution feature map that produces masks with visible blocking artefacts, and passes a low-resolution embedding to SAM. We therefore insert a lightweight convolutional upsampling adapter that reshapes the tokens back to their spatial grid, projects them to 256 channels, and upsamples the grid to 56 × 56, followed by two 3 × 3 convolution blocks with group normalisation (8 groups) and GELU activation. The adapter both matches the embedding resolution expected by the SAM mask decoder and recovers fine boundary detail. Crucially, it also acts as a resolutionagnostic interface: both backbones are mapped to a common 56 × 56 × 256 embedding, so the same SAM decoder is used unchanged across variants. The final segmentation mask is produced by SAM from these projected visual tokens.

## 3.2. Training Strategy

## 3.2.1. Training Dataset

To train the model, we adopt the HAM10000 dataset [33], with an expert-annotated binary mask set [34], while reserving the 86 annotated bruise images exclusively for evaluation. Due to the scarcity of pixel-level annotated bruise data, we formulate the problem as a Zero-Shot Transfer task, reflecting realistic clinical scenarios where labelled data for rare conditions are limited.

The HAM10000 dataset is one of the largest publicly available dermatology datasets, containing 10,015 high-resolution annotated images, whereas most alternative datasets provide only image-level labels. Although not bruise-specific, skin lesions share key structural characteristics with bruises, including irregular shapes, diffuse boundaries, colour variation, and heterogeneous textures.

By training on diverse lesion morphologies, the model learns boundary-aware and structure-sensitive segmentation priors that are not tied to a specific semantic class. These representations can transfer to bruise segmentation under domain shift, enabling generalisation despite the absence of bruise-specific training data.

For data augmentation, we noticed that due to the use of dermoscopes while collecting images for HAM10000, the illumination condition changes the appearance of lesions; to allow the model to learn features which are invariant to illumination and contrast, we applied a simple Retinex method [18] as a data augmentation method, specifically, Multi-scale Retinex with Chromaticity Preservation (MSRCP) [3].

## 3.2.2. Training Details

We trained two variants of our model using LingBot-Vision ViT-B/16 [10] or DINOv3 [28] as visual backbones with all parameters unfrozen, combined with the SAM ViT-B mask decoder adapted via LoRA with rank r = 8 and scaling factor α = 8, which applies to its attention, MLP, projection, and linear layers.

Both variants were optimised using AdamW with a weight decay of $1 \times 1 0 ^ { - 4 }$ and bfloat16 mixed precision. The DINOv3 model was trained for 150 epochs using a learning rate of $2 \times 1 0 ^ { - 5 }$ and a batch size of 64. Its input images were resized to $2 5 6 \times 2 5 6$ and centrally cropped to $2 2 4 \times 2 2 4$ The LingBot images were resized to 576 × 576 and centrally cropped to $5 1 2 \times 5 1 2$ . LingBot was trained for 50 epochs, with a learning rate of $3 \times 1 0 ^ { - 6 }$ , and a batch size of 8 with gradient accumulation over eight steps produced an effective batch size of 64.

![](images/09b6adc790c539d5afad5c5ab688353ccfcf2ec045c7c749e07cca3d1be99676.jpg)  
Figure 1. Overview of the bruise segmentation architecture. LingBot-Vision or DINOv3 encodes the input into a 32 × 32 or 14 × 14 patch-token grid, respectively. The convolutional upsampling adapter reshapes the tokens, projects them to 256 channels, upsamples them to $5 6 \times 5 6 ,$ , and applies two $3 \times 3$ Conv–GN(8)–GELU refinement blocks. The resulting embedding is processed by the SAM ViT-B mask decoder, with optional internal LoRA, before the mask logits are bilinearly upsampled and thresholded to produce the final bruise mask.

The total loss function $\mathcal { L } _ { \mathrm { t o t a l } }$ combined two components:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { D i c e } } + \mathcal { L } _ { \mathrm { B C E } }\tag{1}
$$

The Dice loss encourages overlap between the predicted probability map P and the binary ground truth mask G:

$$
\mathcal { L } _ { \mathrm { { D i c e } } } = 1 - \frac { 2 \sum _ { i = 1 } ^ { N } P _ { i } G _ { i } } { \sum _ { i = 1 } ^ { N } P _ { i } + \sum _ { i = 1 } ^ { N } G _ { i } + \epsilon } .\tag{2}
$$

For BCE, we compute probabilities via a sigmoid, $P =$ $\sigma ( z )$ , and apply BCE:

$$
\mathcal { L } _ { \mathrm { B C E } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ G _ { i } \log ( P _ { i } ) + ( 1 - G _ { i } ) \log ( 1 - P _ { i } ) \right] .\tag{3}
$$

All the above training processes were performed on an NVIDIA GeForce 5070Ti GPU and an Intel i7-12700 CPU with Ubuntu 24.04 LTS via Python 3.11 and PyTorch 2.10.

## 4. Experiments

## 4.1. Baselines and Comparisons

Due to the little to no prior research in this field, we adopted classic CNNs and state-of-the-art segmentation models commonly used in medical imaging and generalpurpose vision as baselines for comparison:

• U-Net (CNN baseline): a 2D U-Net baseline [26] for binary bruise segmentation, trained also on HAM10000 and evaluated on our bruise dataset, with four encoderdecoder stages ([64,128,256,512] channels), batch normalisation, ReLU, skip connections, transposedconvolution upsampling, and a single-channel output head. Images were resized to 256 and centre-cropped to 224 with ImageNet normalisation; training used AdamW (lr=1e-4, wd=1e-4, batch size 4, up to 50 epochs, AMP) and a combined Dice + BCE loss.

• OneFormer (ViT baseline): a OneFormer model [16] with a Swin-L backbone, initialised from the publicly available COCO-pretrained checkpoint and subsequently trained for binary lesion segmentation on HAM10000. Training used 640 × 640 images, AdamW with a learning rate of $1 \times 1 0 ^ { - 5 }$ and weight decay of $1 \times 1 0 ^ { - 2 }$ , a batch size of 1, and 30 epochs. The checkpoint with the highest validation IoU was evaluated on our bruise dataset. Binary masks were obtained from the two-class semantic predictions using a threshold of 0.5, without post-processing.

• VLM Prompt-based SAM baselines: SAM and its variants have been recognised as one of the best segmentation models in many applications as well as the medical field, even without training. To evaluate the effectiveness of our proposed methods, we evaluated several zero-shot large Vision Language Model prompt-based combinations using GPT-4o, GPT-5, and MedGemma as prompters, paired with either SAM2 or MedSAM as the segmentation model. Specifically, we tested GPT-4o + SAM2, GPT-5 + SAM2, GPT-4o + MedSAM, MedGemma + MedSAM, and MedGemma + SAM2 [15, 20, 27, 29].

• SAM3 (zero-shot baseline): The latest model from the SAM family [5] was evaluated directly on our bruise dataset using ’skin bruise’ as its text prompt, without bruise-specific training or fine-tuning. SAM3 uses openvocabulary text prompts to detect and segment objects specified by semantic concepts, allowing bruise regions to be segmented without manually defined spatial prompts.

## 4.2. Evaluation Setup

While BruNet was trained solely on skin lesion images, using bruise images was necessary for evaluation. Hence, a ground-truth annotated dataset was created, using images searched from various publicly available sources on the internet such as Roboflow and Google Images.

Due to the diffuse and ambiguous nature of bruise boundaries, defining a single precise ground-truth contour is often clinically unrealistic. The segmentation literature has long recognised this limitation and developed evaluation protocols that operate on multiple ground-truth annotations per image rather than a single reference, most notably the Probabilistic Rand Index (PRI) [35] and its normalised variant, which measure agreement between a prediction and a set of human segmentations through pairwise pixel relations. In medical imaging, methods such as STAPLE [38] complement this view by estimating a probabilistic consensus across expert annotations. In an ideal protocol, multiple medical experts would independently annotate each bruise image, enabling pixel-wise agreement maps to be constructed and PRI-style evaluation to be applied: pixels on which all annotators agreed would form high-confidence bruise regions, while partially overlapping regions would reflect boundary uncertainty.

However, the data-collection assumptions underlying these protocols are difficult to satisfy in our setting. PRI and related metrics were often adopted on natural-image benchmarks such as BSDS [23], where each image could be annotated at a low cost by several non-expert observers; bruise images, in contrast, require time-consuming annotation by clinically trained experts, making it impractical to obtain a large number of independent annotations per image.

To approximate uncertainty-aware annotation under these constraints, each image was first annotated by one medical expert, after which a lightweight segmentation model was trained to generate candidate masks at multiple probability thresholds, producing a set of more conservative or more permissive boundary estimates. A second medical expert then reviewed these candidate masks and selected two representative regions: (1) a high-confidence inner region corresponding to the most reliable bruise region, and (2) a broader outer region representing highly possible but uncertain bruise boundaries. This process enabled the construction of a dual-region evaluation protocol that approximates inter-observer uncertainty while remaining feasible under limited annotation resources. An example final dualregion annotation is shown in Fig. 2(a).

We note that this protocol is a pragmatic surrogate for the multi-annotator standard underlying PRI [35] and STAPLEstyle [38] evaluation rather than a substitute for it: the candidate masks reflect model behaviour conditioned on a single expert annotation rather than independent inter-observer variability, and a proper multi-expert annotation study remains an important direction for our future work.

## 4.3. Metrics

We report two primary standard metrics: 1/ The Dice score, which is widely used in medical image segmentation for clinical relevance, 2/ Intersection-over-Union (IoU), for a more straightforward segmentation performance against other baselines. Both metrics are defined below, where P is the predicted mask, G is the ground truth mask (inside of the inner region), both of which are treated as binary sets:

$$
\operatorname { D i c e } ( P , G ) = { \frac { 2 | P \cap G | } { | P | + | G | } }\tag{4}
$$

$$
\operatorname { I o U } ( P , G ) = { \frac { | P \cap G | } { | P \cup G | } }\tag{5}
$$

For the secondary metrics, we define true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN) at the pixel level. Precision, Recall, and Accuracy are then computed as:

$$
{ \mathrm { P r e c i s i o n } } = { \frac { T P } { T P + F P } } ,\tag{6}
$$

$$
{ \mathrm { R e c a l l } } = { \frac { T P } { T P + F N } } ,\tag{7}
$$

$$
{ \mathrm { A c c u r a c y } } = { \frac { T P + T N } { T P + T N + F P + F N } } .\tag{8}
$$

For Table 1, each metric is calculated independently for each image and then averaged across the 86-image evaluation set. For statistical testing, however, we use per-image Dice and IoU scores, since paired tests require one paired observation per image.

In the dual-region evaluation, pixels in the highconfidence inner region are treated as positive ground truth, while pixels between the inner and outer regions are excluded from all metric calculations. This prevents uncertain boundary pixels from being classified as either false positives or false negatives.

## 4.4. Results

Table 1 reports the mean results across the 86 test images. Both BruNet variants achieved higher Dice and IoU scores than every evaluated baseline. BruNet-LingBot obtained the highest Dice (0.8674), IoU (0.7897), and accuracy (0.9556), whereas BruNet-DINOv3 achieved the highest recall (0.8491) and the second-highest Dice (0.8386) and IoU (0.7592). Moreover, neither variant was trained on bruise images, yet both substantially outperformed the evaluated baselines. These results demonstrate the effectiveness of adapting pre-trained visual representations and the SAM mask decoder for cross-domain transfer from skin lesion segmentation to bruise segmentation.

Table 1. Segmentation performance on the 86-image bruise test set. Each metric was calculated for every image and then averaged across all images. BruNet is compared with U-Net, OneFormer, and zero-shot or prompt-based segmentation baselines.
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>Dice</td><td>IoU</td><td>Accuracy</td></tr><tr><td>U-Net</td><td>0.4619</td><td>0.7515</td><td>0.5198</td><td>0.4092</td><td>0.7900</td></tr><tr><td>OneFormer</td><td>0.7063</td><td>0.8085</td><td>0.7097</td><td>0.6352</td><td>0.9039</td></tr><tr><td> $\mathrm { G P T } { \cdot } 5 + \mathrm { S A M } 2$ </td><td>0.9467</td><td>0.2924</td><td>0.4116</td><td>0.2922</td><td>0.8496</td></tr><tr><td>GPT-4o + SAM2</td><td>0.8902</td><td>0.1497</td><td>0.2299</td><td>0.1494</td><td>0.8178</td></tr><tr><td>GPT-4o + MedSAM</td><td>0.8344</td><td>0.1943</td><td>0.2995</td><td>0.1916</td><td>0.8236</td></tr><tr><td>MedGemma + MedSAM</td><td>0.6025</td><td>0.5771</td><td>0.5157</td><td>0.3911</td><td>0.8472</td></tr><tr><td> $\mathbf { M e d G e m m a } + \mathbf { S A M } 2$ </td><td>0.6575</td><td>0.2846</td><td>0.3491</td><td>0.2632</td><td>0.8479</td></tr><tr><td>SAM3 (zero-shot)</td><td>0.9372</td><td>0.5888</td><td>0.7053</td><td>0.5875</td><td>0.9208</td></tr><tr><td>BruNet-DINOv3 (Ours)</td><td>0.8691</td><td>0.8491</td><td>0.8386</td><td>0.7592</td><td>0.9441</td></tr><tr><td>BruNet-LingBot (Ours)</td><td>0.9464</td><td>0.8227</td><td>0.8674</td><td>0.7897</td><td>0.9556</td></tr></table>

Among the conventional segmentation baselines, One-Former achieved the strongest Dice (0.7097) and IoU (0.6352), supported by a high recall of 0.8085. However, its lower precision of 0.7063 indicates a greater tendency to include non-bruise regions. This behaviour is visible in Fig. 2(h), where the prediction extends beyond the annotated outer boundary. U-Net achieved a recall of 0.7515 but a precision of only 0.4619, resulting in a Dice score of 0.5198 and an IoU of 0.4092. Its prediction in Fig. 2(d) similarly extends outside the outer boundary, illustrating its tendency towards over-segmentation.

The LLM-prompted SAM2 and MedSAM pipelines generally produced conservative or incomplete masks. GPT-5 + SAM2 achieved the highest precision among all evaluated models (0.9467), but its low recall (0.2924) limited its Dice and IoU to 0.4116 and 0.2922, respectively. GPT-4o + SAM2 was even more conservative, with a recall of 0.1497, a Dice score of 0.2299, and an IoU of 0.1494. As illustrated in Fig. 2(b) and (c), these predictions capture only limited portions of the bruise. Among the five LLMprompted pipelines, MedGemma + MedSAM performed best, obtaining a Dice score of 0.5157 and an IoU of 0.3911. Nevertheless, its performance remained below OneFormer, SAM3, and both BruNet variants. These results suggest that bounding-box localisation followed by SAM-style segmentation often fails to recover the complete extent of diffuse bruises. However, the current evaluation does not isolate whether this limitation originates from bounding-box localisation, prompting, or mask generation.

Native text-prompted SAM3 achieved high precision (0.9372) and the highest accuracy among the baselines (0.9208). Its lower recall of 0.5888 reduced its Dice and IoU to 0.7053 and 0.5875, respectively. Its Dice score was close to that of OneFormer, although its IoU was lower. SAM3 produced a strong segmentation for the example in Fig. 2(i), but it returned no retained bruise instance for five of the 86 test images. Therefore, the selected visualisation does not represent its performance on every image.

In comparison, the BruNet predictions in Fig. 2(j) and (k) form spatially coherent bruise regions that closely follow the annotated boundaries. BruNet-LingBot provides the strongest overall overlap and precision–recall balance, while BruNet-DINOv3 achieves slightly greater bruise coverage, as reflected by its higher recall. The strong performance of both variants supports the use of pretrained visua representations adapted with the SAM mask decoder for segmenting bruises characterised by weak contrast, diffuse colour changes, and uncertain boundaries. Furthermore, our best model, BruNet-LingBot, requires only 400 MiB of VRAM and takes at most 30 ms per image on our experimental setup, while one of the better-performing baselines, SAM 3, requires over 4000 MiB of VRAM and more than 230 ms per image. This demonstrates BruNet-LingBot’s substantially higher computational efficiency and its greater suitability for real-world applications.

## 4.4.1. Statistical Analysis

To assess whether the observed performance differences were statistically reliable, we applied two-sided paired Wilcoxon signed-rank tests [6] to the per-image Dice and IoU scores from the 86-image evaluation set. Pairing was used because every model was evaluated on the same images. Table 1 reports the arithmetic mean of these per-image scores, whereas the statistical tests operate on matched image-level differences between models.

For each model, we estimated a 95% confidence interval for the mean Dice and IoU using 10,000 nonparametric percentile-bootstrap resamples of the images. BruNet-DINOv3 was retained as the reference model for the paired comparisons. For each comparison, $\Delta _ { \mathrm { m e d } }$ denotes the median of the 86 paired differences, calculated as the BruNet-DINOv3 score minus the comparison-model score. Its 95% confidence interval was obtained by paired bootstrap resampling. Zero differences were excluded from the signed-rank calculation using the Wilcox convention. Holm–Bonferron correction [14] was applied separately to the nine Dice comparisons and the nine IoU comparisons. Statistical significance was assessed at $\alpha = 0 . 0 5$ . A fixed random seed of 20260824 was used for reproducibility.

![](images/d8dcc41f6c14169c841f6d6ecedeba2acf629428fd842e3a4f4468ef0264ef21.jpg)  
(j)  
(k)  
Figure 2. Visualised comparison of bruise segmentation results. (a) Ground-truth annotation; (b) GPT-4o + SAM2; (c) $\mathrm { G P T - } 5 \ +$ SAM2; (d) U-Net; (e) GPT-4o + MedSAM; (f) MedGemma + MedSAM; (g) MedGemma + SAM2; (h) OneFormer; (i) SAM3 zero-shot; (j) BruNet-LingBot (Ours); and (k) BruNet-DINOv3 (Ours). In each prediction panel, the translucent red region represents the predicted bruise mask, while the green and blue contours indicate the inner positive boundary and outer uncertainty boundary, respectively. Pixels between the two ground-truth boundaries were ignored during evaluation and metrics calculation.

As shown in Table 2, BruNet-LingBot achieved the highest mean Dice and IoU, followed by BruNet-DINOv3. Table 3 shows that BruNet-DINOv3 outperformed all eight non-BruNet baselines for both Dice and IoU after Holm– Bonferroni correction. This includes the strongest non-BruNet baseline, OneFormer, for which the adjusted $p \textmd { - }$ values were 0.0044 for Dice and 0.0059 for IoU. The comparisons with SAM3 and all remaining baselines had adjusted p-values below 0.001.

The differences between BruNet-DINOv3 and BruNet-

LingBot were not statistically significant for either Dice $( p _ { \mathrm { a d j } } ~ = ~ 0 . 4 0 9 6 )$ or IoU $( p _ { \mathrm { a d j } } ~ = ~ 0 . 5 4 8 0 )$ Their median paired differences were close to zero, and both bootstrap intervals included zero. These results indicate that the two BruNet backbone variants provide comparable imagelevel Dice and IoU, although BruNet-LingBot achieved the higher arithmetic mean for both metrics.

The bootstrap interval for $\Delta _ { \mathrm { m e d } }$ and the Wilcoxon test summarise different properties of the paired differences: the former estimates the median difference, whereas the latter uses the signed ranks of all non-zero differences. Consequently, a median-difference interval can include zero even when the corrected Wilcoxon test is significant, as observed for the OneFormer comparison.

## 5. Ablation Study

To evaluate the contribution of the SAM mask decoder and the intermediate feature upsampling in BruNet, we conducted two ablation experiments using the LingBot-Vision ViT-B/16 backbone. We first replaced the SAM ViT-B mask decoder with a lightweight convolutional segmentation head. We then removed the intermediate bilinear upsampling to $5 6 \times 5 6$ and passed the native $3 2 \times 3 2$ LingBot feature grid directly to the SAM decoder. All variants were trained using the same data, optimisation settings, and Dice + BCE loss, and evaluated on the same 86-image bruise dataset using the dual-region protocol.

The lightweight segmentation head projects the patch features to 256 channels, followed by ${ \mathrm { ~ a ~ 3 ~ } } \times { \mathrm { ~ 3 ~ } }$ convolution to 128 channels with GroupNorm and GELU. Four additional $3 \times 3$ convolutional blocks and two bilinear upsampling stages are then applied before a final 1×1 convolution produces the segmentation mask. This replaces the SAM prompt encoder and mask decoder with a conventional convolutional decoder.

As shown in Table 4, replacing the SAM decoder reduced Dice from 0.867 to 0.848 and IoU from 0.790 to 0.764. The convolutional head attained the highest recall of the three variants (0.881 against 0.823), but this reflects over-segmentation rather than better delineation, as its precision fell from 0.946 to 0.866. Removing the intermediate feature upsampling reduced Dice further to 0.839 and IoU to 0.752. Although this variant retained high precision (0.939), visual inspection showed more fragmented masks that tended to retain only high-confidence core regions, reducing recall to 0.789. This supports the role of intermediate $5 6 \times 5 6$ upsampling in preserving spatial continuity and recovering complete bruise regions.

## 6. Conclusion

Bruise segmentation remains challenging due to diffuse appearance and limited annotated data. We proposed

Table 2. Bootstrap 95% confidence intervals for the mean per-image Dice and IoU scores across the 86-image evaluation set.
<table><tr><td>Model</td><td>Dice mean (95% CI)</td><td>IoU mean (95% CI)</td></tr><tr><td>U-Net</td><td>0.5198 (0.4572–0.5841)</td><td>0.4092 (0.3500–0.4719)</td></tr><tr><td>OneFormer</td><td>0.7097 (0.6398–0.7776)</td><td>0.6352 (0.5646–0.7058)</td></tr><tr><td> $\mathrm { G P T } { \cdot } 5 + \mathrm { S A M } 2$ </td><td>0.4116 (0.3601–0.4657)</td><td>0.2922 (0.2479–0.3396)</td></tr><tr><td> $\mathrm { G P T } { \cdot } 4 0 + \mathrm { S A M } 2$ </td><td>0.2299 (0.1856–0.2755)</td><td>0.1494 (0.1150–0.1860)</td></tr><tr><td> $\mathrm { G P T } { \cdot } 4 0 + \mathrm { M e d S A M }$ </td><td>0.2995 (0.2606–0.3390)</td><td>0.1916 (0.1612–0.2234)</td></tr><tr><td> $\mathrm { M e d G e m m a + M e d S A M }$ </td><td>0.5157 (0.4583–0.5718)</td><td>0.3911 (0.3403–0.4428)</td></tr><tr><td> $\mathbf { M e d G e m m a } + \mathbf { S A M } 2$ </td><td>0.3491 (0.2812–0.4174)</td><td>0.2632 (0.2071–0.3204)</td></tr><tr><td>SAM3 (zero-shot)</td><td>0.7053 (0.6527–0.7539)</td><td>0.5875 (0.5374–0.6369)</td></tr><tr><td> $\overline { { \mathrm { B r u N e t - D I N O v } 3 } }$ </td><td>0.8386 (0.7962–0.8766)</td><td>0.7592 (0.7100–0.8058)</td></tr><tr><td>BruNet-LingBot</td><td>0.8674 (0.8348–0.8963)</td><td>0.7897 (0.7488–0.8278)</td></tr></table>

Table 3. Paired comparisons between BruNet-DINOv3 and each other model on the 86-image evaluation set. $\Delta _ { \mathrm { m e d } }$ is the median paired difference (BruNet-DINOv3 minus the comparison model), reported with a paired-bootstrap 95% confidence interval. The two-sided Wilcoxon signed-rank p-values were Holm–Bonferroni corrected separately across the nine comparisons for each metric. <sup>∗</sup> denotes significance at $\alpha = 0 . 0 5$ after correction.

(a) Dice
<table><tr><td>Comparison model</td><td> $\overline { { \Delta _ { \mathrm { m e d } } ( 9 5 \% { \bf C } { \bf I } ) } }$ </td><td> $p _ { \mathrm { a d j } }$ </td></tr><tr><td>U-Net</td><td>+0.2676 (+0.2093, +0.3198)</td><td> $< 0 . 0 0 1 ^ { * }$ </td></tr><tr><td>OneFormer</td><td> $+ 0 . 0 1 2 5 \ : ( - 0 . 0 0 5 5 , + 0 . 0 6 2 7 )$ </td><td> $0 . 0 0 4 4 ^ { * }$ </td></tr><tr><td> $\mathrm { G P T } { \cdot } 5 + \mathrm { S A M } 2$ </td><td>+0.4564 (+0.3504, +0.4957)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathrm { G P T } \mathrm { - } 4 0 + \mathrm { S A M } 2$ </td><td>+0.6682 (+0.6037, +0.7487)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathrm { G P T } { \cdot } 4 0 + \mathrm { M e d S A M }$ </td><td>+0.5774 (+0.5170, +0.6225)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathbf { M e d G e m m a } + \mathbf { M e d S A M }$ </td><td>+0.2705 (+0.2373, +0.3575)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathbf { M e d G e m m a } + \mathbf { S A M } 2$ </td><td>+0.5120 (+0.3586, +0.6213)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td>SAM3 (zero-shot)</td><td> $+ 0 . 1 1 2 8 \ : ( + 0 . 0 9 7 5 , + 0 . 1 3 2 7 )$ </td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td>BruNet-LingBot</td><td> $- 0 . 0 0 1 0 \left( - 0 . 0 1 3 2 , + 0 . 0 1 2 6 \right)$ </td><td>0.4096</td></tr></table>

(b) IoU
<table><tr><td>Comparison model</td><td> $\overline { { \Delta _ { \mathrm { m e d } } ( 9 5 \% { \bf C } { \bf I } ) } }$ </td><td> $p _ { \mathrm { a d j } }$ </td></tr><tr><td>U-Net</td><td>+0.3167 (+0.2561, +0.3733)</td><td> $\overline { { < 0 . 0 0 1 ^ { * } } }$ </td></tr><tr><td>OneFormer</td><td> $+ 0 . 0 2 2 0 \ ( - 0 . 0 1 1 0 , + 0 . 0 9 1 3 )$ </td><td> $0 . 0 0 5 9 ^ { \ast }$ </td></tr><tr><td> $\mathrm { G P T } { \cdot } 5 + \mathrm { S A M } 2$ </td><td>+0.4847 (+0.4049, +0.5689)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathrm { G P T } \mathrm { - } 4 0 + \mathrm { S A M } 2$ </td><td>+0.6738 (+0.5851, +0.7704)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathrm { G P T } { \cdot } 4 0 + \mathrm { M e d S A M }$ </td><td>+0.6425 (+0.5433, +0.6945)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathbf { M e d G e m m a } + \mathbf { M e d S A M }$ </td><td>+0.3792 (+0.3038, +0.4493)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathbf { M e d G e m m a } + \mathbf { S A M } 2$ </td><td>+0.5011 (+0.4220, +0.6115)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathbf { S A M } 3 \left( \mathbf { z e r o - s h o t } \right)$ </td><td>+0.1719 (+0.1441, +0.1966)</td><td> $< 0 . 0 0 1 ^ { \ast }$ </td></tr><tr><td> $\mathbf { B r u N e t - L i n g B o t }$ </td><td>-0.0019 (-0.0219, +0.0217)</td><td>0.5480</td></tr></table>

Table 4. Ablation study on the 86-image bruise dataset, evaluating the SAM mask decoder and intermediate ViT-to-SAM feature upsampling.
<table><tr><td>Model Variant</td><td>Precision</td><td>Recall</td><td>Dice</td><td>IoU</td><td>Accuracy</td></tr><tr><td>LingBot + Seg. Head</td><td>0.866</td><td>0.881</td><td>0.848</td><td>0.764</td><td>0.950</td></tr><tr><td>LingBot w/o intermediate upsample</td><td>0.939</td><td>0.789</td><td>0.839</td><td>0.752</td><td>0.947</td></tr><tr><td>BruNet-LingBot</td><td>0.946</td><td>0.823</td><td>0.867</td><td>0.790</td><td>0.956</td></tr></table>

BruNet, a zero-shot transfer segmentation framework that combines a ViT-based visual encoder with a SAM mask decoder. When trained on the HAM10000 skin lesion dataset, BruNet demonstrates strong generalisation to outof-distribution bruise images without any task-specific finetuning.

To the best of our knowledge, this is the first work to address bruise segmentation with the assistance of computer vision and machine learning. Our dual-region evaluation set highlights the ambiguity of bruise boundaries, and BruNet consistently outperforms CNN baselines and prompt-based SAM variants across Dice and IoU metrics. The results suggest that combining structural reasoning (via ViTs) improves robustness in visually ambiguous, underannotated clinical targets. These findings support future research in forensic and dermatological imaging, where annotation scarcity and domain shift are major challenges.

## 6.1. Future Work

## 6.1.1. Integration of foundation models

With the rapid advancement of deep learning, foundation models such as PanDerm [39] have demonstrated strong performance across a wide range of skin lesion analysis tasks. Future work could explore replacing the current ViT backbone with such domain-specific foundation models, enabling improved feature representations and more effective instruction of the SAM decoder, particularly under few-shot learning settings.

## 6.1.2. More modalities

While bruise segmentation serves as an important first step towards automated bruise analysis, clinically relevant assessment extends beyond localisation. Future research should investigate the integration of additional predictive tasks, such as estimating bruise age and severity, which could provide more comprehensive support for medical and forensic decision-making.

## References

[1] Kiyarash Aminfar, Katherine Scafide, Janusz Wojtusiak, and David Lattanzi. From structure health monitoring to forensics: adapting computer vision to support victims of violence. In Health Monitoring of Structural and Biological Systems XVIII, page 93, Long Beach, United States, 2024. SPIE. 1, 3

[2] D M Anisuzzaman, Yash Patel, Behrouz Rostami, Jeffrey Niezgoda, Sandeep Gopalakrishnan, and Zeyun Yu. Multimodal wound classification using wound image and location by deep neural network. Sci. Rep., 12(1):20057, 2022. 2

[3] Kobus Barnard and Brian Funt. Investigations into multiscale retinex. Proc. Colour Imaging in Multimedia’98, pages 9–17, 1998. 3

[4] Ahmad Faiz Bin Nor’azam and Yoshihiro Mitani. Arm injury classification on a small custom dataset using cnns and augmentation. In 2023 International Conference on Computer Graphics and Image Processing (CGIP), pages 29–33, 2023. 3

[5] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. Sam 3: Segment anything with concepts. In International Conference on Learning Representations, pages 138846–138923, 2026. 2, 4

[6] William Jay Conover. Practical Nonparametric Statistics. John Wiley & Sons, 1999. 6

[7] Dharmi Desai, Amin Nayebi, Mehrdad Ghyabi, David Lattanzi, Katherine Scafide, and Janusz Wojtusiak. The effect of skin tone classification on bias in bruise detection. Research Square, 2025. 3

[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16×16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020. 1, 2

[9] Ahmed Elsarta, Habiba Fathalla, Marina Nasser, Sara Elwatany, Rawan Fekry, Mohamed Ebaid, Yomna Mahmoud, Ayman Anwar, and Amira Gaber. Integrating multi-source data for skin burn classification using deep learning. Comput. Biol. Med., 195(110556):110556, 2025. 2

[10] Zelin Fu, Bin Tan, Changjiang Sun, Shaohui Liu, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, and Nan Xue. Vision pretraining for dense spatial perception. arXiv preprint arXiv:2607.05247, 2026. 2, 3

[11] Tudor Vladimir Gurau, Gabriela Gurau, Carmina Liana Musat, Doina Carina Voinescu, Lucretia Anghel, Gelu Onose, Constantin Munteanu, Ilie Onu, and Daniel Andrei

Iordan. Epidemiology of injuries in professional and amateur football men (Part II). J. Clin. Med., 12(19):6293, 2023. 1

[12] Ali Hatamizadeh, Yucheng Tang, Vishwesh Nath, Dong Yang, Andriy Myronenko, Bennett Landman, Holger R Roth, and Daguang Xu. UNETR: Transformers for 3D medical image segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 574–584, 2022. 1, 2

[13] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceed ings ofthe IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 2

[14] Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, pages 65–70, 1979. 6

[15] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 4

[16] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. OneFormer: One Transformer to Rule Universal Image Segmentation. 2023. 4

[17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 1, 2, 3

[18] Edwin H Land. The retinex. In Ciba Foundation Symposium Colour Vision: Physiology and Experimental Psychology, pages 217–227. Wiley Online Library, 1965. 3

[19] Leo Lebrat, Rodrigo Santa Cruz, Remi Chierchia, Yu-´ lia Arzhaeva, Mohammad Ali Armin, Joshua Goldsmith, Jeremy Oorloff, Prithvi Reddy, Chuong Nguyen, Lars Petersson, et al. Syn3DWound: A synthetic dataset for 3d wound bed analysis. arXivpreprint arXiv:2311.15836, 2023. 2

[20] Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. Segment anything in medical images. Nature Communications, 15(1):654, 2024. 2, 4

[21] Lidia Maggioni, Emanuela Maderna, Maria Carlotta Gorio, Annalisa Cappella, Salvatore Andreola, Gaetano Bulfamante, and Cristina Cattaneo. The frequently dismissed importance of properly sampling skin bruises. Legal Medicine, 50:101867, 2021. 1

[22] Puspita Majumdar, Saheb Chhabra, Richa Singh, and Mayank Vatsa. On detecting domestic abuse via faces. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018. 3

[23] D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001, pages 416–423 vol.2, 2001. 5

[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,

Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. arXiv [cs.CV], 2021. 1

[25] Audrey Raut, Mary Clyde Pierce, Kim Kaczor, Doug Lorenz, Gina Bertocci, Karen Bertocci, and Kirsten Simonton. Single bruise characteristics associated with abusive vs accidental injury. Pediatrics, 155(3):e2024067932, 2025. 2

[26] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Lecture Notes in Computer Science, pages 234– 241. Springer International Publishing, Cham, 2015. 1, 2, 4

[27] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, C´ıan Hughes, Charles Lau, et al. MedGemma technical report. arXiv preprint arXiv:2507.05201, 2025. 2, 4

[28] Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 2, 3

[29] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025. 4

[30] Sayma Alam Suha and Tahsina Farah Sanam. A deep convolutional neural network-based approach for detecting burn severity from skin burn images. Machine Learning with Applications, 9:100371, 2022. 2

[31] R E Taylor and P M Blatt. Clinical evaluation of the patient with bruising and bleeding. J. Am. Acad. Dermatol., 4(3): 348–368, 1981. 1

[32] Jhonatan Tirado and David Mauricio. Bruise dating using deep learning. J. Forensic Sci., 66(1):336–346, 2021. 1, 3

[33] Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data, 5(1):180161, 2018. 3

[34] Philipp Tschandl, Christoph Rinner, Zoe Apalla, Giuseppe Argenziano, Noel Codella, Allan Halpern, Monika Janda, Aimilios Lallas, Caterina Longo, Josep Malvehy, et al. Human–computer collaboration for skin cancer recognition. Nature medicine, 26(8):1229–1234, 2020. 3

[35] Ranjith Unnikrishnan, Caroline Pantofaru, and Martial Hebert. Toward objective evaluation of image segmentation algorithms. IEEE Transactions on Pattern Analysis and Machine Intelligence, 29(6):929–944, 2007. 5

[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2017. 2

[37] A Vora and M Makris. An approach to investigation of easy bruising. Archives of Disease in Childhood, 84(6):488–491, 2001. 1

[38] S.K. Warfield, K.H. Zou, and W.M. Wells. Simultaneous truth and performance level estimation (STAPLE): an algorithm for the validation of image segmentation. IEEE Trans actions on Medical Imaging, 23(7):903–921, 2004. 5

[39] Siyuan Yan, Zhen Yu, Clare Primiero, Cristina Vico-Alonso, Zhonghua Wang, Litao Yang, Philipp Tschandl, Ming Hu, Lie Ju, Gin Tan, Vincent Tang, Aik Beng Ng, David Powell, Paul Bonnington, Simon See, Elisabetta Magnaterra, Peter Ferguson, Jennifer Nguyen, Pascale Guitera, Jose Banuls, Monika Janda, Victoria Mar, Harald Kittler, H Peter Soyer, and Zongyuan Ge. A multimodal vision foundation model for clinical dermatology. arXiv [cs.CV], 2024. 8