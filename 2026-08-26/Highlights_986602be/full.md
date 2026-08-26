Graphical Abstract

Example-based Robust Abnormality Detection with Minimal Annotations using Exemplar Med-DETR

Sheethal Bhat, Bogdan Georgescu, Awais Mansoor, Mathias Zinnen, Pranjal Sahu, Florin C. Ghesu, Sasa Grbic, Andreas Maier

![](images/c9d77b2d7954297771c3e8bec5700e8683bee3ebdaca65fc260cfd30ce040933.jpg)  
Figure 1: Overview of the pretraining and finetuning stages of Exemplar Med-DETR (EM-DETR). The model is pretrained on a set of base classes and finetuned on a smaller set of the novel class of interest. The Exemplar generation module $\mathcal { G }$ extracts the patches belonging to Region-of-Interest (ROI) and computes a unique exemplar or prototype embedding per class. These embeddings are stored in a memory bank and are used during inference (dotted arrows) for improved detection. The decoder learns to associate the derived exemplar embeddings with the characteristics of the class in an image, along with the text, thus allowing minimal-shot detection. This approach lays the foundation for a continuous learning framework with minimal annotations when new findings are introduced.

## Highlights

Example-based Robust Abnormality Detection with Minimal Annotations using Exemplar Med-DETR

Sheethal Bhat, Bogdan Georgescu, Awais Mansoor, Mathias Zinnen, Pranjal Sahu, Florin C. Ghesu, Sasa Grbic, Andreas Maier

• Scaling Exemplar Med-DETR with Minimal Annotations: Demonstrates how exemplar-based detection can efectively scale to large chest radiography datasets with minimal annotation efort, reducing reliance on extensive labeled data for new findings.

• Adaptable Framework for Emerging Findings: Proposes a structure that could be extended to accommodate new disease classes with limited retraining, ofering a foundation for future continuous learning that avoids catastrophic forgetting on previously learned classes.

• Includes evaluations on both proprietary and public CXR datasets with ablation studies, demonstrating robustness and scalability under minimal-annotation and out-of-distribution settings.

# Example-based Robust Abnormality Detection with Minimal Annotations using Exemplar Med-DETR

Sheethal Bhat<sup>a,b</sup>, Bogdan Georgescu<sup>c</sup>, Awais Mansoor<sup>c</sup>, Mathias Zinnen<sup>a</sup>, Pranjal Sahu<sup>c</sup>, Florin C. Ghesu<sup>b</sup>, Sasa Grbic<sup>c</sup>, Andreas Maier<sup>a</sup>

<sup>a</sup>Pattern Recognition Lab, Friedrich-Alexander-Universität, Erlangen, 91058, Germany <sup>b</sup>Digital Technology and Innovation, Siemens Healthineers, Erlangen, , Germany <sup>c</sup>Digital Technology and Innovation, Siemens Medical Solutions, Princeton, NJ, 08540, U.S.A

## Abstract

Reducing annotation requirements remains a key challenge in developing robust medical object detectors. To address this, Vision-Language (VL) object detection methods leverage grounding text information to enable powerful zero-shot and few-shot object detectors in the natural image domain [1, 2, 3, 4]. However, transferring these methods to the medical domain is challenging due to the absence of comparable quality and quantity of the grounding data. Regardless, significant contextual and non-imaging information exists in medical images that remains underutilized. Few-shot learning (FSL) techniques partially address this limitation but struggle to generalize to unseen medical findings and require extensive retraining when new findings are introduced [5, 6]. To overcome these challenges, we extend our prior EM-DETR framework [7] and introduce a scalable FS detection approach designed for eficient abnormality detection in Chest X-Ray (CXR) images under minimal supervision. The proposed architecture incorporates exemplar-based feature generation and domain-aware contrastive optimiza tion, enabling efective adaptation to novel disease findings without exhaustive retraining. Our method achieves near state-of-the-art (SOTA) detection performance using less than 10% of the annotated data, demonstrating its potential for practical, annotation-eficient clinical deployment across both proprietary and public CXR datasets.

Keywords: Multi-modal DETR, Abnormality Detection, Minimal

## 1. Introduction

Deep learning has achieved remarkable progress in visual understanding tasks, yet these advances are heavily dependent on large-scale annotated datasets [8, 9]. In the medical imaging domain, obtaining such annotations remains a persistent bottleneck, as each label requires expert interpretation that is time-consuming, and often varies across institutions and modalities [7, 10, 11]. Reducing the annotation efort while maintaining high diagnostic performance has therefore been a central research objective for Computer-Aided Diagnostic (CAD) systems [12, 13].

In pursuit of this goal, various strategies have been proposed in the literature aimed at reducing the dependence on manual expert annotations. These approaches range from weakly-supervised and semi-supervised learning methods that exploit incomplete labels [14, 15, 16], as well as active learning methods that selectively query the most informative samples[17], and few-shot learning (FSL) methods that aim to generalize from only a handful of labeled examples per class [5, 6]. Nevertheless, these methods have not yet demonstrated competitive spatial precision at scale in medical imaging tasks [? ].

In contrast to these annotation-eficient approaches, recent multi-modal Vision-language (VL) detection methods [1, 2, 4] demonstrate zero-shot, open-vocabulary detection capabilities over a wide variety of target classes in computer vision datasets [8, 9]. These results are achieved in part through strong pretraining and the use of a large number of semi-automatically generated “grounding” captions. However, transferring these methods to the medical domain remains dificult: radiology reports rarely provide explicit fine-grained grounding descriptions, and similar large-scale grounded medical datasets are not available. As a result, integrating annotations with descriptive text and spatial location, despite its potential to reduce the burden of annotations, remains largely underexplored in medical imaging.

In Chest X-Ray (CXR) research, current state-of-the-art (SOTA) systems rely extensively on both increased number of annotations per finding and post-processing techniques such as handcrafted Non-Maximal Suppression (NMS) [18] methods to achieve high diagnostic accuracy. Nonetheless, in the current age of automated report production [19, 20, 21], there is an increasing need for CAD systems that can detect findings with minimal annotations. This need is particularly evident in CXR datasets, which often contain long-tailed label distributions in which many findings are represented by only a small number of samples [10, 22]. Thus, it is beneficial to develop a CAD system that learns from base class(es) where training data are inexpensive to acquire, available in abundance; and therefore can achieve optimal performance with minimal annotations for novel or rare findings.

These requirements motivated our example-based abnormality detection method, called Exemplar Med-DETR (EM-DETR) [7], which reduces the dependence on dense annotations and avoids post-processing. Although inspired by VL detection architectures, EM-DETR operates on learnt visual exemplars, making it suitable for domains without grounded radiology captions. In our previous work, we demonstrated that EM-DETR achieves SOTA performance in multiple medical domains, such as the detection of mass and calcification in mammography, masses in CXR images, and stenoses in angiography images [7]. In this study, we evaluate the performance of EM-DETR by simulating a minimal-shot scenario with a set of novel or rare classes that are held out during base class training. This allows us to assess the ability of EM-DETR to detect clinically important but sparsely annotated findings under realistic annotation constraints.

With the recent success of transformer-based multi-modal detection pipelines, our proposed approach is based on the principles of grounding DINO [1], which extends GLIP [2] and DETR [23] to tightly fuse language and vision modalities through a feature enhancer and cross-modality decoder. To incorporate prototypical visual cues, we introduce an Exemplar Generation Module G, which serves as an additional input to the decoder (see Fig. 3). The module derives an exemplar or prototype class-specific embedding for each class. This embedding is included in the cross-attention pipeline such that the decoder detects regions that not only match the text description but also the visual features of that class. In other words, to the existing multimodal framework, we add class-conditional knowledge (few-shot exemplars or prototypes) into the detection process. This encourages decision boundaries that reflect semantic distances between findings rather than relying solely on their correlation with text descriptions. Theoretically, this enables improved generalization to unseen domains or novel classes, supports few-shot learning through stable prototype anchors, and reduces overfitting to frequently occurring classes.

In addition to architectural modifications, we also adapt the model to the unique characteristics of the CXR data, which difer substantially from natural image datasets. As CXR images are acquired under standardized protocols, they exhibit strong spatial alignment [24], with several abnormalities consistently appearing in specific anatomical regions. Furthermore, unlike natural images where class features are typically distinct (e.g., car vs. tree), CXR findings often exhibit high intra-class variability due to disease manifestation [25], and the overlapping features of diferent abnormalities. Accurate diagnosis also depends on contextual, often non-imaging, information [26], increasing the dificulty of purely appearance-based discrimination. Given these domain-specific challenges and the reliance of EM-DETR on contrasting classes, we find that explicit selection of negative samples plays a critical role in enhancing localization accuracy—even in minimal-annotation regimes.

Our main contributions are summarized as follows:

• Scaling EM-DETR with minimal annotations: Demonstrates how exemplar-based detection can scale efectively in chest radiographs using only a small number of labeled samples, reducing annotation burden for clinical deployment.

• Adaptable framework for new findings: Proposes an architecture designed to facilitate addition of new disease classes with limited retraining, forming a basis for future work in continuous learning.

• Domain-aware iterative negative sampling: Introduces a domainspecific background selection strategy for iterative contrastive learning, improving training stability and detection accuracy in multi-class, longtailed scenarios.

• Evaluation on CXR datasets: Validated across proprietary and public datasets with ablation studies, including an out-of-distribution test, demonstrating robustness under minimal-annotation scenarios.

## 2. Related Works

The field of abnormality detection in radiology has been extensively researched in the domain of Artificial Intelligence (AI), with numerous SOTA techniques available that include open-source [27? , 28] and commercial options [26, 29, 30, 31, 32, 33]. Approaches to reduce manual annotations for

AI training is one of the most active research areas in medical AI at the moment [25, 34, 35, 36]. These strategies can be broadly categorized into weaklyor semi-supervised learning, multi-modal learning, and few-shot learning approaches, spanning tasks in classification, detection, and segmentation across both computer vision and medical domains. In the following sections, we review the current research landscape in these domains, highlighting recent advances and methodological trends.

## 2.1. Weakly or semi-supervised methods

Weakly or semi-supervised frameworks leverage unlabelled or weakly labelled data through pseudo-labels obtained by various methods such as clustering, consistency regularization or contextual label generation. Recent studies by Ju et al. [37] address universal semi-supervised learning for classification by employing a variational auto-encoder (VAE). The VAE is pretrained to understand unseen samples based on the assumption that these samples follow a diferent distribution. The authors propose a unified framework using a "dual-path outlier estimation scheme". While elegant, this approach is strongly dependent on the Gaussian distribution assumption of the underlying data. Alternatively, He et al. [38] demonstrate openset semi-supervised classification through a multi-binary discriminator and a joint outlier filter mechanism, learning prototypes that help identify distinct classes.

In the field of segmentation, Du et al. [39] propose using a geometric prior along with contrastive loss to improve 3D segmentation with only bounding box annotations and Kuang et al. [40] propose multi-class segmentation through “feature decomposition’. Likewise, Li et al. [41] present a weakly supervised segmentation approach for histopathology images using multi-instance learning (MIL) with self-attention, efectively aggregating global contextual information through deep supervision. Alternatively, the authors of INSIGHT [42] draw on general vision strategies, combining a detection module with a heatmap aggregator for classification and segmentation at the same time. For a broader overview, Jiao et al. [43] provide a comprehensive survey of semi-supervised learning techniques for medical image segmentation.

Beyond the medical domain, semi-supervised and weakly supervised methods have also been extensively explored in general computer vision tasks. Weak supervision in computer vision typically uses box, point annotations [44] and image-level labels, while annotation types in medical imaging employ points, scribbles, and gaze [45, 46]. Moreover, in the medical domain, there is a greater influence of domain shift, spatial resolution, and class imbalance compared to the general image domain. Recent studies in the latter domain include STAC [47] and object detection via teacher-student training [15, 48], which are used to produce pseudo-labels for unlabeled images. On the other hand, for holistic scene understanding, weakly and semi-supervised panoptic segmentation methods have emerged as eficient alternatives to full supervision. Works such as Li et al. [49], Li et al. [50], and Lee et al. [51] integrate semantic and instance cues through proposal refinement, region-to-pixel consistency, and cross-task distillation, respectively, to jointly segment “thing” and “stuf” categories with limited labels. Nevertheless, such progress does not directly translate to medical detection in CXRs, where the challenge lies in localizing subtle findings rather than performing generic classification or pixel-level segmentation. In the following section, we investigate multi-modal approaches that facilitate DL tasks with minimal annotations.

## 2.2. Vision-Language (VL) methods

Recent multimodal methods, especially vision–language frameworks, im prove model performance by integrating contextual knowledge from accompanying text, efectively using language as a weak supervisory signal. Early studies, including UNITER [52] and OSCAR [53], focus on developing universal embeddings or features by pretraining with image-caption datasets in the natural image domain. UNITER [52] presents four pretraining tasks, including masked language/region modeling, VL matching, and optimal-transport word–region alignment, aimed at achieving detailed text and image region alignment. OSCAR [53] similarly uses object tags as “anchor” tokens to align image regions with words, pretraining on 6.5M image–text pairs and achieving SOTA on VQA, retrieval, and grounding benchmarks [54, 55, 56, 57, 58]. These methods illustrate the core paradigm of concatenating region features (from object detectors) with text tokens and applying cross-modal self-attention. In this context, the text acts as a distinctive encoding key, enabling the transformer encoder-decoder mechanism to efectively recognize image data from the provided textual input.

Likewise, recent methods like GLIP[2] and OWL-ViT[4] develop openvocabulary detectors that utilize contrastive image-text training with vision transformers, enabling text-conditioned training on arbitrary categories. The most recent VL model Grounding DINO [1] further improves upon GLIP[2] by combining the powerful DINO backbone with grounded language-based object detection, leading to further enhancements in zero-shot object detection performance. This approach incorporates enhancements to the De tection Transformer Encoder-Decoder (DETR), including the use of a deformable MS-attention encoder along with robust pretraining utilizing captions and descriptions from over 1.8 million images. This setup facilitates open-set detection for wide variety of unseen classes. On the other hand, BLIP [59] focuses on data: It bootstraps noisy web images by using a captioner to generate high-quality captions and a filter to remove noise, enabling unified pretraining for both understanding and generation. BLIP achieves strong gains in retrieval, captioning, and VQA by leveraging this synthetic web-scale supervision.

Beyond alignment and detection, large multi-modal methods now handle open-ended and few-shot tasks. Flamingo [60] is a few-shot VLM that interleaves pretrained vision and language modules via a “perceiver” resampler, later extended to Med-Flamingo [36]. The model processes mixed text and images to address open-ended tasks through prompts. With only a handful of examples, Med-Flamingo shows SOTA few-shot performance on medical VQA benchmarks, outperforming methods that were fine-tuned on much larger datasets. Analogously, Kosmos-2 [61] extends these ideas to a generalist multi-modal LLM. Trained from scratch on web-scale interleaved image–text data, Kosmos-2 [61] supports zero-/few-shot learning, instruction following, and even OCR-free language tasks by converting visual inputs into “language” tokens. It shows strong performance on various VL benchmarks such as captioning and VQA. In contrast, MedOptNet [35] proposes a metalearning framework utilizing high-performance convex optimization methods for few-shot classification.

Despite impressive results on few-shot classification and reporting tasks, the medical domain lacks large-scale grounding text paired with bounding box annotations, limiting the applicability of these methods for few-shot detection. VL methods can reduce annotation needs, but only when textual cues provide meaningful guidance. Methods such as Grounding DINO [1] under-perform on unseen classes in few-shot evaluations, and Flamingo [60] struggles to localize specific findings. Language-based supervision is further limited in the medical domain due to scarce pretraining data, a constrained and non-standardized vocabulary, and frequent overlap of terms across different findings [62, 63, 17].

## 2.3. Few-shot learning methods

Few-shot detection enables models to adapt to new classes with minimal samples, leveraging pretraining on base classes. Bulat et al. [5] introduce FS-DETR, a few-shot detection transformer that uses visual prompts to detect novel classes. By generating visual templates and pseudo-class embeddings with a strong image backbone, FS-DETR outperforms zero-shot and even some fine-tuning methods on benchmarks like PASCALVOC [9] and MSCOCO [8]. In natural images, ROIs (e.g., “cat” or “dog”) yield discriminative embeddings from ImageNet-pretrained CNNs [64], enabling efective template matching. In contrast, CXR ROIs such as “pleural efusion” or “lesion” produce overlapping embeddings [65] (see Fig. A.12, Appendix A), influenced by anatomical structures and lacking isolated class examples. As a result, the decoder must distinguish pathological from normal anatomy using both global and local context, motivating the need for a robust template generator that produces highly discriminative, class-specific features.

Meanwhile Zhang et al. present Meta-DETR [6], that bypasses region proposals and uses an inter-class correlational meta-learning strategy to improve few-shot object detection, mainly achieved by capturing the correlation between classes. These methods highlight the trend toward encoder-decoder architectures that treat all V+L tasks as generative language modeling. While both FS-DETR [5] and Meta-DETR [6] extend DETR for few-shot detection using meta-learning concepts, FS-DETR [5] primarily focuses on feature re-adaptation and query re-weighting, whereas Meta-DETR introduces a meta-learning module that learns transferable parameters for task-level adaptation. Although efective on computer vision datasets, these methods do not translate as strongly across all CXR medical findings.

Complementing these studies, several recent works have advanced fewshot image classification through diverse representation learning strategies. Zheng et al. [66] proposed Unsupervised Few-Shot Image Classification via One-vs-All Contrastive Learning, extending few-shot learning to unsupervised settings by leveraging contrastive objectives. Building on these foundations, Semantic-Guided Generalization Enhancement (SGE) [67] integrates semantic priors to guide generalization, addressing the semantic shift across domains in few-shot learning. Similarly, Wu et al. [68] developed invariant and consistent unsupervised representation learning for few-shot visual recognition, emphasizing stability under feature transformations. Together, these studies illustrate a coherent research trajectory focused on enhancing feature robustness, semantic consistency, and generalization in low-data regimes, forming a solid foundation for subsequent multimodal or domainadaptive extensions such as MERGE [69] and DARA [70].

Converse to previous work focussing on classification, Wang et al. presented a prototype-based feature mapping network designed to address the adaptation of the few-shot domain in medical segmentation in PFMNet [71]. However, these approaches are primarily designed for natural image domains within the few-shot learning paradigm, often focus solely on classification or segmentation, and do not readily translate to robust few-shot detection of multiple findings in CXR medical imaging. Furthermore, open-source implementations are limited, hindering replication and extension. Building on these observations, we next detail the dataset preparation and the proposed EM-DETR adaptation for minimal-shot CXR detection, before presenting the experimental results.

## 3. Data preparation

We first examine our hypothesis on a smaller anatomy dataset to verify the feasibility of few-shot detection using EM-DETR. This approach is logical, as anatomy detection on CXRs is less challenging compared to the detection of complex findings. Subsequently, we proceed to assess EM-DETR using minimal annotations on larger standard datasets and intricate findings.

Towards this end, two separate datasets (A and B) are derived from a larger private dataset that was acquired from multiple sites in the United States and Europe. The study includes frontal and lateral chest exams from adult cohorts (>18 years). Two general radiologists independently reviewed and annotated all CXRs with the presence and location of 16 findings. A third general radiologist served as an independent tiebreaker for discrepant findings. The following subsections detail the private dataset specifics, and then describe the public dataset used for further evaluation.

## 3.1. Proprietary CXR Anatomy dataset

We use a dataset A of anatomy annotations, comprising 100 full-resolution anonymized DICOM images. The dataset includes 80 frontal images for training and 20 for validation. The splits are obtained patient-wise to avoid patient overlap. The images contain segmentation masks for 7 of the major organs such as the lungs, clavicle, heart, aorta, and ribs, as shown in Fig. 2.

The 7 segmentation masks are first converted into bounding box coordinates. These annotations are further separated into left and right organ classes, resulting in 10 classes. Table. 2 denotes the final number of classes used to train EM-DETR. Moreover, the ribs and spine annotations are also divided into 3 separate classes: upper, middle, and lower organ. The demarcations are obtained using a third of the original height of the bounding box. To simulate a few-shot condition, only five training images include annotations of the heart region.

![](images/21461086af9c9cf42c6b718a3ba801c3c4dd406ec99295eb5bb620ec110d6ae3.jpg)  
Figure 2: Example image from the anatomy dataset, where we only use the frontal views and discard the lateral views.

## 3.2. Proprietary CXR Abnormality dataset

A second dataset B of 18,681 full-resolution anonymized DICOM CXR images is used to evaluate the few-shot performance of EM-DETR on 3 diferent CXR findings. The dataset includes frontal-view images of patients, annotated with bounding boxes indicating the presence of 16 findings provided by an expert panel of radiologists. We use a split of 3 validation datasets of 331, 412 and 412 images for pleural efusion, pneumothorax (ptx) and lesion detection. The validation datasets have binary labels {1,0} indicating the existence or absence of a finding. We ensure that there is no overlap between the patients in the training and validation splits and Table. 1 presents the data split with the number of positive samples in brackets.

Pleural efusion is a relatively easy finding for the network to detect while pneumothorax and lesion are more challenging. Together, they cover the spectrum of dificulty for detection in CXRs [72, 16]. To simulate a minimalshot scenario, a pretraining dataset called $\boldsymbol { B } _ { P }$ is created with all findings excluding pleural efusion, pneumothorax, and lesion. The finetuning dataset $\boldsymbol { B } _ { F }$ is then composed of a smaller number of images with the finding of interest and a suficient number of images without the finding, which are taken as negative samples. For eg. a smaller subset of 200 images with pneumothorax and 800 images without pneumothorax is taken as the finetuning dataset for pneumothorax. This accounts for 7% of the entire training dataset, where the samples are selected at random. This training scheme simulates a minimalshot scenario, in which we have a powerful pretrained model, and introduce a new class with minimal annotations.

<table><tr><td rowspan="2">Finding</td><td colspan="2">Images (Positive samples)</td></tr><tr><td>Training</td><td>Test</td></tr><tr><td>Pleural effusion</td><td>18,681(4037)</td><td>331(74)</td></tr><tr><td>Pneumothorax</td><td>18,681(1868)</td><td>412(146)</td></tr><tr><td>Lesion</td><td>18,681(5654)</td><td>412(109)</td></tr></table>

Table 1: Number of images per finding in the training and test sets. The number in the brackets indicate the number of positive samples for each finding.

## 3.3. Public CXR Abnormality dataset

Additionally, we test the finetuned EM-DETR model on the public VinDR-CXR test dataset [10] without any further finetuning. The VinDR-CXR [10] testing dataset comprises 3000 images and features 22 local labels alongside 6 global labels. We test the classification and FROC performance for pneumothorax due to it being a dificult task with only 18 positive samples in this test dataset.

## 4. Method

Inspired by FS-DETR [5], we adapt our previously introduced EM-DETR [7] for minimal-shot detection. The proposed framework consists of two main components: exemplar-based finetuning and iterative contrastive training. In the following subsections, we first describe how per-finding exemplars are generated to enhance detection, and then explain the iterative training scheme.

EM-DETR builds on multi-modal DETR [23, 1], performing “languageguided query selection” via cross-attention between image and text embeddings. To guide detection, we learn class-specific exemplars computed from visual features corresponding to each class’s spatial location, bootstrapping template generation similarly to FS-DETR. A moving average of these exemplars are interleaved with the text embeddings and passed to the Detection Transformer Encoder-Decoder pipeline. Attending to these exemplars enables the detection heads to perform prototype-based searches, facilitating straightforward expansion to novel classes. Fig. 3 illustrates the EM-DETR pretraining stage, including exemplar generation and the additional losses that regularize the generated exemplars.

![](images/aeb4d9f6b1bd65c994a328e3b3bbd71128ef4650f65b889e7ce3293ee73c8e1d.jpg)  
Figure 3: Overview of EM-DETR pretraining. Visual features $\mathbf { X } _ { k }$ & positional embeddings $\mathbf { P } _ { k }$ are extracted from the frozen image backbone based on the $k ^ { t h }$ class location (red). F uses learnable class-specific embeddings $\mathbf { c t } _ { k }$ & $\mathbf { c e } _ { k }$ to calculate $\mathbf q _ { k }$ . A moving average of $\mathbf q _ { k }$ results in the prototype embedding $\mathbf { e } _ { k } .$ . Interleaving the text embedding $\mathbf { t } _ { k }$ with $\mathbf { e } _ { k }$ enables text-and feature-based detection for each base class. $\mathbf { e } _ { k }$ is used in additional $\mathcal { L } _ { c o n t r a s t }$ & $\mathcal { L } _ { f e a t }$ losses to improve the decoder search operations.

To further improve learning, we adopt an iterative contrastive strategy. In each stage, hard negatives are mined either by generating background annotations from normal images or by leveraging false positives, refining class prototypes and improving overall detection performance.

## 4.1. Exemplar Generation G:

The Swin transformer [73] image backbone is a multi-scale, shifted window transformer that produces a set of J patch embeddings $\mathbf { X } ^ { \prime } = [ \mathbf { x _ { 1 } } , . . . \mathbf { x } _ { J } ] ^ { T }$ for an input image, where each $\mathbf { x } _ { j } \in \mathbb { R } ^ { d }$ . A similar dimension set of positional encodings is also generated, denoted as $\mathbf { P } ^ { \prime } = [ \mathbf { p _ { 1 } } , . . . \mathbf { p } _ { J } ] ^ { T }$ . Let $\mathbf { X } _ { k } \subset \mathbf { X } ^ { \prime }$ and $\mathbf { P } _ { k } \subset \mathbf { P } ^ { \prime }$ represent the set of M tokens that fall within the ROI derived from the bounding box ground truth annotations of the class k (red).

In CXRs, certain findings consistently manifest in predictable patterns and are frequently localized to specific areas, such as in the cases of “pleural efusion” and “cardiomegaly”. Conversely, other findings exhibit significant variability in both presentation and anatomical location. To account for this, $\mathbf { P } _ { k }$ is scaled by a learnable class-specific parameter $a _ { k } \in \mathbb { R }$ and added to ${ \bf X } _ { k }$ This produces $\mathbf { H } _ { k }$ , mathematically shown as

$$
\mathbf { H } _ { k } = \mathbf { X } _ { k } + \mathbf { P } _ { k } \times a _ { k } .\tag{1}
$$

$a _ { k }$ modulates the impact of the positional embeddings, thereby influencing the decoder search operation. Additional learnable class-wise token $\mathbf { c t } _ { k }$ and positional c embeddings are then concatenated with $\mathbf { H } _ { k }$ . The simple attention pooling transformer $\mathcal { F }$ processes the concatenated result to learn a single normalized embedding using self-attention mechanism [74]. The use of $\mathbf { c t } _ { k }$ and $\mathbf { c t } _ { k }$ within $\mathcal { F }$ provides additional class-specific information that helps to build a unique example feature. Concurrently, $\mathbf { c t } _ { k }$ and $\mathbf { c e } _ { k }$ are also added to the output of $\mathcal { F }$ to produce $\mathbf { q } _ { k } \in \mathbb { R } ^ { d }$ encapsulating the textural and spatial features corresponding to class k. The addition of $\mathbf { c t } _ { k }$ and $\mathbf { c e } _ { k }$ is analogous to the positional embeddings added to the text embeddings in the feature enhancer [1, 6]. Mathematically, $\mathbf { q } _ { k }$ is given as

$$
\mathbf { q } _ { k } = \mathcal { F } ( c o n c a t ( \mathbf { c } \mathbf { t } _ { k } , \mathbf { c e } _ { k } , \mathbf { H } _ { k } ) ) + \mathbf { c } \mathbf { t } _ { k } + \mathbf { c e } _ { k } ,\tag{2}
$$

where $k \in \{ 1 , \ldots , N \}$ and N is the total number of classes. A per-class moving average is calculated over L samples of $\mathbf { q } _ { k } ,$ , thereby generating the prototype embedding $\mathbf { e } _ { k } \in \mathbb { R } ^ { d }$ . This prevents the decoder from pursuing rapidly changing features during each training iteration. $\mathbf { e } _ { k }$ is stored in a Memory bank as a prototype feature embedding that helps prevent catastrophic forgetting. Eventually, model inference relies on the memory bank of saved feature representations, or exemplars ${ \bf e } _ { k } ,$ since ROI regions are not available for $\mathbf { e } _ { k }$ generation in test images.

In parallel, the frozen text encoder processes text prompts and produces embeddings $\mathbf { t } _ { k }$ which are interleaved with the corresponding $\mathbf { e } _ { k }$ and passed downstream to the encoder-decoder pipeline [23, 1]. The text prompt inputs are literal class names (e.g. “pleural efusion”, “background”). Fig. 4 illustrates the interleaving and encoding operations in greater detail. Attention masks ensure that self-attention is applied within the tokens for each class at a sub-sentence level similar to [1].

Thereafter, the decoder predicts the location of the $k ^ { t h }$ class by processing the cross-attention encodings of the text and representative visual features, $\mathbf { t } _ { k }$ and $\mathbf { e } _ { k }$ with the input image embeddings, $\mathbf { X } ^ { \prime }$ and $\mathbf { P ^ { \prime } }$ . This is done through a contrastive search that looks for highest cosine similarity between the many hypothesis boxes and the text or feature embeddings.

![](images/35625afed7e57eeb4b6428816e285209be0a7cd9cc72ffdc04e749d3ce5c26a6.jpg)  
Figure 4: Representation of the per-class interleaving of the text and feature embeddings. The tokens generated from the encoder undergo self-attention at a sub-sentence level through additional attention masks [1].

Additional Losses $\mathcal { L } \dot { z }$ . In addition to the original DETR loss terms $\mathcal { L } _ { \mathrm { b b o x } } .$ $\mathcal { L } _ { \mathrm { { I o U } } }$ , and $\mathcal { L } _ { \mathrm { c l a s s i f y } }$ in [1] we introduce two additional loss functions to improve the robustness of EM-DETR. A cosine similarity contrastive feature loss $\mathcal { L } _ { c o n t r a s t } .$ , is applied on $E = [ \mathbf { e } _ { 1 } . . . \mathbf { e } _ { N } ] ^ { T }$ of all $1 \leq k \leq N$ classes. $\mathcal { L } _ { c o n t r a s t }$ ensures that all class prototype embeddings remain orthogonal to each other in the latent space, promoting distinct and separable representations [75]. A $L _ { 2 }$ feature loss, $\mathcal { L } _ { f e a t } ,$ is applied between $\mathbf { e } _ { k }$ and the decoder’s top proposal ${ \bf d } _ { k }$ for class k ensuring that the model predicts a consistent latent representation for each class [5, 6]. This approach helps maintain class-specific embeddings over time while stabilizing the training process. Although previous studies[5, 6] indicate that $\mathcal { L } _ { f e a t }$ does not significantly afect precision results in general vision studies, it is empirically observed to improve our Free Response Operating Characteristic (FROC) results [7].

## 4.2. Iterative training strategy:

The trained decoder learns class-specific features but struggles to distinguish normal anatomy from disease traits in abnormality detection. Therefore, we propose a multi-stage iterative learning approach, for efective minimalshot abnormality detection. Fig. 5 illustrates the iterative training schematic. Stage I involves pretraining the proposed model with all base class annotations dataset $\boldsymbol { B } _ { P }$ , while subsequent finetuning stages refine the weights through a binary per-class background-versus-foreground detection task using only the minimal positive samples.

![](images/faf00c86cc2326bda6627c25e2667afe21f314ef9db1370468ce55e253445de6.jpg)  
Figure 5: Schematic of iterative training strategy, where ∗ denotes generated annotations.

Fig. 6 illustrates the EM-DETR workflow for finetuning and inference. In our framework, finetuning is not restricted to a single stage but is designed as a modular, multi-stage strategy that incrementally refines the model for a given novel class. For example, Stage II initiates this process, and additional stages can be appended to further improve performance by leveraging feedback from prior stages, such as false positive suppression. This design allows continued adaptation without retraining from scratch.

In Stage II training, the pretrained model is trained on dataset $B _ { F }$ . The pretrained memory bank-I of the base classes, can either be discarded to create a binary class memory bank-II or extended by incorporating novel class exemplars, depending on the model design. Since the network’s classification layer aligns text prompts with hypothesis regions, freezing the decoder doesn’t significantly impact performance. However, freezing the encoder in Stage II impairs performance, as the novel class features require crossattention encoding. F stays trainable to ensure efective attention pooling for the novel classes.

In an additional Stage III, we evaluate the Stage II model on the training dataset and denote the False Positive (FP) regions as background classes to further refine the network. This helps the model refine its ability to separate visually ambiguous areas from actual pathological findings. During inference, the memory bank is used exclusively, as there are no ROIs available.

Background generation:. The efectiveness of the contrastive detection model depends on the contrastive learning scheme between classes. In medical applications such as CXR, two main challenges arise: diferentiating anatomical features from abnormalities and distinguishing closely related class features. To address this, we reformulate the multi-class problem into a binary detection task by negative mining the background class from images that do not contain the finding.

Moreover, CXR images have an advantage over natural images, as they are typically well-aligned. Although the location of each finding may vary, the findings tend to appear in specific anatomical regions. Using these characteristics, we select backgrounds based on the prior distribution of findings within our minimal dataset. This approach improves model accuracy by enabling it to distinguish between normal and diseased anatomy. As an example, Fig. 8 highlights the selection of the foreground and background bounding box regions based on the prior location of pleural efusion and a device.

![](images/badf8aa020af298e6ce5b89c36d91b08a71265204fbfeb277041dbd5585a327d.jpg)  
Figure 6: Overview of the finetuning and inference process. The memory bank accumulated during Stage I is either discarded or the novel class embeddings are added to create a new memory bank II. The inference path is shown in dotted arrow line.

Additionally, stage III uses multiple false positive locations as background annotations to further refine the model, thereby enhancing learning and performance metrics. Fig. 7 illustrates the efectiveness of exemplar feature extraction and class separation between the foreground-background classes for pneumothorax and pleural efusion through a TSNE [76] analysis of the embeddings stored in the memory bank.

## 5. Experiments

## 5.1. Experimental details

The experiments were run on a single node with four 40 GB A100 GPUs, using a learning rate of 0.0008 with a linear scheduler under the MMDetection [77] framework. An average of 5 runs is recorded. The image and text backbones were frozen. F is designed as a simple 2-head, 4-layer transformer.

![](images/4fd1aa12bbff2ecb1b5fb6b81f40fb542adf2d55adbebb6ff4e67893290d82f9.jpg)

![](images/94a826c34627d0f8e7c525372f7a4c31b0f47552b9343d666ca2bb9a24570705.jpg)

Figure 7: 2D t-SNE analysis of pneumothorax (left) and pleural efusion (right) prototypes contrasted with their background samples.  
![](images/eb78a53eb2f12fab724fc21ab757d3d2440289fc2479a24de7d917da9516d07e.jpg)  
Background – select based on location of foreground  
Figure 8: Diagram to show how background areas are selected based on the finding of interest.

The moving average is computed over L = 200 exemplars. The mean average precision at 50% IoU $( m A P _ { 5 0 } )$ is reported.

Background selection. The image is divided into 4 quadrants, and two background annotations are selected per quadrant, based on the approximate location of the foreground class. We randomly add a 50 pixel top/bottom, left/right perturbation to the selected location along with a 20% variation in size of the bounding box based on the average size of the foreground class. At the final stage III, we further finetune the model with the top 8 misclassified regions (FP) in training images set as background examples.

## 5.2. Evaluation metrics

We evaluate all models using both classification and localization metrics. For image level classification, the highest probability per image per class is saved to calculate the image level area under the ROC curve (AUC) score. The $m A P _ { 5 0 }$ metric is used to verify the localization result at higher Intersection-over-Union levels of 50%. The default $m A P _ { 5 0 }$ is calculated using the standard coco metric settings provided by the MMDetection [77] framework. As a practical metric, we also evaluate the distance-based Freeresponse Receiver Operating Characteristic (FROC) response of the model [78, 79, 16, 26, 18, 80, 81]. The sensitivity is computed on the basis of the distance between the center of the ground truth and the predicted box [82].

<table><tr><td>Category</td><td>Baseline G.DINO [1]</td><td>EM-DETR [7]</td></tr><tr><td>Clavicle</td><td>0.94</td><td>0.95</td></tr><tr><td>Aorta</td><td>1.0</td><td>0.99</td></tr><tr><td>Heart</td><td>0.23</td><td>0.92</td></tr><tr><td>Lungs</td><td>1.0</td><td>1.0</td></tr><tr><td>Upper ribs</td><td>1.0</td><td>1.0</td></tr><tr><td>Middle ribs</td><td>1.0</td><td>0.97</td></tr><tr><td>Lower ribs</td><td>1.0</td><td>0.96</td></tr><tr><td>Upper spine</td><td>1.0</td><td>1.0</td></tr><tr><td>Middle spine</td><td>1.0</td><td>1.0</td></tr><tr><td>Lower spine</td><td>1.0</td><td>1.0</td></tr><tr><td>Average</td><td>0.82</td><td>0.98</td></tr></table>

Table 2: Few shot detection of the heart class with only 5 training images for heart and 80 training images for other classes. The results are depicted in average m $A P _ { 5 0 }$ . The inclusion of the Exemplar Generation module $\mathcal { G }$ results in a decrease in detection performance for classes that overlap with the heart region due to overlapping region boundaries.

## 6. Experimental Results

## 6.1. Comparison with SOTA

Table. 2 indicates the few-shot detection results for the “heart” class on the anatomy dataset A. The first column presents per-class $m A P _ { 5 0 }$ for baseline grounding DINO [1], and the second column for EM-DETR. With just 5 heart-annotated images, EM-DETR boosts the $m A P _ { 5 0 }$ from 23% to 92%. However, a 3-4% decrease in the mA $P _ { 5 0 }$ for the middle and lower ribs is observed due to overlapping annotated regions with heart, thus leading to an increase in the cosine similarity and consequent decrease in the classification accuracy. An ablation study in Appendix A.13 indicates that the $m A P _ { 5 0 }$ for the heart reaches 99% with only 15 images.

<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td rowspan="2">Data %</td><td colspan="2">Sensitivity</td><td rowspan="2">AUC</td><td rowspan="2"> $m A P _ { 5 0 }$ </td></tr><tr><td>0.25 FP</td><td>0.5 FP</td></tr><tr><td rowspan="6">tx</td><td>Baseline G.DINO [1]</td><td>7%</td><td> $0 . 5 1 8 \pm \gamma$ </td><td>0.634 ± γ</td><td> $0 . 8 3 7 \pm \epsilon$ </td><td> $0 . 2 5 8 \pm 0 . 0 5$ </td></tr><tr><td>RPS [18]</td><td>7%</td><td>0.518</td><td>0.642</td><td>0.877</td><td></td></tr><tr><td>EM-DETR [7]</td><td>7%</td><td> $\mathbf { 0 . 8 8 5 \pm } \epsilon$ </td><td> ${ \bf 0 . 9 3 2 \pm } \epsilon$ </td><td> $\mathbf { 0 . 9 5 7 \pm 0 }$ </td><td> ${ \bf 0 . 4 2 7 \pm 0 . 0 2 }$ </td></tr><tr><td>Baseline G.DINO [1]</td><td>100%</td><td> $\begin{array} { c } { 0 . 8 4 3 \pm \gamma } \\ { 0 . 9 7 0 } \end{array}$ </td><td></td><td> $0 . 9 6 0 \pm \epsilon$ </td><td> $0 . 5 2 4 \pm 0 . 0 3$ </td></tr><tr><td>RPS [18]</td><td>100%</td><td></td><td> $\begin{array} { c } { 0 . 9 2 5 \pm \gamma } \\ { 0 . 9 7 0 } \end{array}$ </td><td>0.983</td><td></td></tr><tr><td>EM-DETR [7]</td><td>100%</td><td> $0 . 9 1 5 \pm \epsilon$ </td><td> $0 . 9 5 6 \pm \epsilon$ </td><td>0.980</td><td> $\mathbf { 0 . 5 6 4 \pm 0 . 0 1 }$ </td></tr><tr><td rowspan="6">P.i on</td><td>Baseline G.DINO [1]</td><td>7%</td><td> $\begin{array} { c } { 0 . 9 0 2 \pm \gamma } \\ { 0 . 6 7 0 } \end{array}$ </td><td> $\begin{array} { c } { 0 . 9 6 0 \pm \gamma } \\ { 0 . 8 1 9 } \end{array}$ </td><td> $\begin{array} { c } { 0 . 9 7 5 \pm \epsilon } \\ { 0 . 8 9 4 } \end{array}$ </td><td> $0 . 3 1 4 \pm 0 . 0 3$ </td></tr><tr><td>RPS [18]</td><td>7%</td><td></td><td></td><td></td><td></td></tr><tr><td>EM-DETR [7]</td><td>7%</td><td> ${ \bf 0 . 9 8 7 \pm } \epsilon$ </td><td> ${ \bf 0 . 9 8 7 \pm } \epsilon$ </td><td> ${ \bf 0 . 9 8 6 \pm } \epsilon$ </td><td> ${ \bf 0 . 3 5 2 \pm 0 . 0 2 }$ </td></tr><tr><td>Baseline G.DINO [1]</td><td>100%</td><td> $0 . 8 7 8 \pm \gamma$ </td><td> $0 . 8 9 9 \pm \gamma$ </td><td> $0 . 9 8 9 \pm \epsilon$ </td><td> $0 . 3 8 9 \pm 0 . 0 2$ </td></tr><tr><td>RPS [18]</td><td>100%</td><td>0.997</td><td>0.997</td><td>0.993</td><td></td></tr><tr><td>EM-DETR [7]</td><td>100%</td><td> $0 . 9 8 7 \pm \epsilon$ </td><td> $0 . 9 8 7 \pm \epsilon$ </td><td> ${ \bf 0 . 9 9 3 \pm } \epsilon$ </td><td> $\mathbf { 0 . 4 2 2 \pm 0 . 0 3 }$ </td></tr><tr><td rowspan="6">esiont</td><td>Baseline G.DINO [1]</td><td>7%</td><td> $0 . 3 0 9 \pm \gamma$ </td><td></td><td></td><td> $0 . 1 5 7 \pm 0 . 0 2$ </td></tr><tr><td>RPS [18]</td><td>7%</td><td>0.420</td><td> $\begin{array} { c } { 0 . 3 6 8 \pm \gamma } \\ { 0 . 4 8 7 } \end{array}$ </td><td> $\begin{array} { c } { 0 . 7 3 0 \pm \epsilon } \\ { 0 . 7 8 9 } \end{array}$ </td><td></td></tr><tr><td>EM-DETR [7]</td><td>7%</td><td> ${ \bf 0 . 5 4 1 \pm } \epsilon$ </td><td> $\mathbf { 0 . 6 5 0 \pm } \epsilon$ </td><td> $\mathbf { 0 . 8 6 5 \pm } \epsilon$ </td><td> ${ \bf 0 . 3 3 8 \pm 0 . 0 1 }$ </td></tr><tr><td>Baseline G.DINO [1]</td><td>100%</td><td></td><td></td><td></td><td> ${ \bf 0 . 5 2 7 \pm 0 . 0 2 }$ </td></tr><tr><td>RPS [18]</td><td>100%</td><td> $\begin{array} { c } { 0 . 4 5 8 \pm \gamma } \\ { 0 . 8 0 0 } \end{array}$ </td><td> $\begin{array} { c } { 0 . 6 7 8 \pm \gamma } \\ { 0 . 8 9 0 } \end{array}$ </td><td> $\mathbf { 0 . 9 0 1 } \pm \epsilon$ </td><td></td></tr><tr><td>EM-DETR [7]</td><td>100%</td><td> $0 . 5 9 6 \pm \epsilon$ </td><td> $0 . 6 9 1 \pm \epsilon$ </td><td> $0 . 8 9 9 \pm \epsilon$ </td><td> $0 . 5 1 4 \pm 0 . 0 2$ </td></tr></table>

Table 3: Comparison of Sensitivity and AUC Scores for Diferent Findings at Various False Positive Rates and Dataset Sizes. All experiments conducted with a Swin-B image encoder backbone. In the case of Lesion† we evaluate the highest scale of the multi-scale feature from the Swin-B backbone. For readability we denote the standard deviation as $\epsilon \leq 0 . 0 1$ and $\gamma \leq 0 . 1$

Following this, EM-DETR is tested on a task involving detecting three specific abnormalities or findings. Table. 2 presents the detection results when using only 7% of the training data for pleural efusion, pneumothorax, and lesions. The sensitivites at 0.25 FP and 0.5 FP are noted from the FROC curves along with the classification AUC scores and localization mA $P _ { 5 0 }$ . An average 10–30% point sensitivity improvement and 2–13% point AUC increase is observed for all three findings compared to the baseline Grounding DINO model [1] when using only 7% of annotations. We also discern 4-16% point improvement in $m A P _ { 5 0 }$ with 7% annotations and a corresponding 3-4% point improvement from the baseline when 100% of annotations are used. Our results are further compared with our proprietary Reference

Product Solution (RPS), demonstrating comparable performance at maximum annotations and superior performance with minimal annotations in all three cases.

We observe that EM-DETR achieves near SOTA sensitivity and classification results for minimal-shot pleural efusion detection, unlike pneumothorax and lesions. This is attributed to the closely clustered latent space distribution and the consistent positioning of pleural efusion within CXR images as seen in Fig. 7. Analysis of the failure cases in Fig. 11 shows that pleural efusion is typically and correctly detected at the lung bases even when bounding box overlap is low.

Conversely, pneumothorax exhibits much higher variation in appearance and location, which is reflected in the ablation study (Section 6.2). Increasing the number of pneumothorax annotations from 200 to 400 substantially improves both classification and localization, approaching SOTA performance (Table 6). This underscores the importance of sample diversity for certain pathologies and suggests that uniform random sampling may be insuficient for findings with heterogeneous presentation. Fig. 9 shows the resulting FROC curves for pleural efusion (left) and pneumothorax (right) under <10% training data.

![](images/c60feb55b036c147388274d875eb28fdc65376eef9b9a03e6f19c2b40cfb43bc.jpg)  
(a) FROC curve for pleural efusion.

![](images/b9d822706af4ec0298341c5fac5c748915a6d83da5ee9691a2f8e61b47f1cb4b.jpg)  
(b) FROC curve for pneumothorax  
Figure 9: Example FROC curves from EM-DETR for a pair of findings, displaying the average and standard deviation when finetuned on less than 10% of the training data.

As an additional experiment for lesion detection, we evaluated the pretrained model using only the highest-resolution scale of the multi-scale features. Despite this simplification, the model still outperformed the SOTA FROC results. Given that lesions are complex abnormalities, careful selection of background samples is necessary, as evidenced by the performance gap with 100% annotations.

<table><tr><td rowspan="2">Method (Dataset size)</td><td colspan="2">Sensitivity</td><td rowspan="2">AUC</td></tr><tr><td>0.25FP</td><td>0.5FP</td></tr><tr><td>EM-DETR (7%) [7]</td><td> $\mathbf { 0 . 7 2 2 \pm 0 . 0 2 ^ { \ast } }$ </td><td> $\mathbf { 0 . 8 3 3 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 0 9 \pm 0 . 0 1 ^ { \ast } }$ </td></tr><tr><td>Grounding DINO (7%) [1]</td><td> $0 . 5 5 6 \pm 0 . 0 6$ </td><td> $0 . 7 7 7 \pm 0 . 1 3$ </td><td> $0 . 8 5 3 \pm 0 . 0 2$ </td></tr><tr><td>RPS (7%) [18]</td><td>0.278</td><td>0.333</td><td>0.682</td></tr><tr><td>Deformable DETR (7%) [23]</td><td>0.271</td><td>0.542</td><td>0.524</td></tr><tr><td>Deformable DETR (100%) [23]</td><td>0.944</td><td>0.944</td><td>0.995</td></tr><tr><td>RPS (100%) [18]</td><td>0.889</td><td>0.889</td><td>0.967</td></tr><tr><td>EM-DETR (100%) [7]</td><td> $0 . 8 8 9 \pm 0 . 0 2 ^ { \ast }$ </td><td> $\mathbf { 0 . 9 4 4 \pm 0 . 0 3 ^ { \ast } }$ </td><td> $0 . 9 6 5 \pm 0 . 0 1 ^ { \ast }$ </td></tr><tr><td>Grounding DINO (100%) [1]</td><td> $0 . 8 3 3 \pm 0 . 0 4$ </td><td> $0 . 8 3 3 \pm 0 . 0 3$ </td><td> $0 . 8 9 7 \pm 0 . 0 2$ </td></tr><tr><td>RetinaNet (100%) [83]</td><td></td><td></td><td>0.865</td></tr><tr><td>Meta-DETR† (50%) [6]</td><td></td><td></td><td>0.857</td></tr><tr><td>RTMDet (100%) [84]</td><td></td><td></td><td>0.806</td></tr><tr><td>Deformable DETR† (100%) [23]</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>0.729</td></tr><tr><td>Faster RCNN (100%) [85]</td><td></td><td></td><td>0.496</td></tr></table>

Table 4: Comparison of pneumothorax performance with various SOTA models. † indicates within-distribution tests where the model was trained and evaluated on VinDR-CXR dataset. \* indicates statistical significance with $\mathrm { p < 0 . 0 5 }$ when compared to the baseline. Standard deviation provided where available.

We further evaluate our finetuned EM-DETR model on the VinDR-CXR [10] dataset and compare its performance with an in-house RPS solution. The models are finetuned only on the $\boldsymbol { B } _ { \mathcal { P } }$ and $\boldsymbol { B _ { \mathcal { F } } }$ datasets and not further finetuned on VinDR-CXR, allowing us to assess their out-of-distribution (OOD) performance. As shown in Table 4, EM-DETR outperforms the SOTA baseline when trained with fewer annotations and achieves higher sensitivity at 0.25, 0.5 false positives (FP) per image when using 100% of the available annotations. These results highlight the eficiency and robustness of EM-DETR in detecting CXR abnormalities even on OOD datasets with minimal supervision. For additional context, we also report comparisons with other detection methods. These results serve only as reference points, since our primary evaluation focuses on minimal-shot generalization. Additionally, Deformable DETR is evaluated as a within-distribution and out-of-distribution case. The model outperforms the RPS when trained on the internal dataset B as it has more pneumothorax training samples (1868) than the VinDR-CXR dataset (96). However, under the minimal-shot configuration, EM-DETR achieves the best performance. Meta-DETR, designed for a much smaller k-shot setting, is trained with only 50% of VinDR-CXR pneumothorax annotations, as using more samples leads to overfitting and lower performance.

<table><tr><td rowspan="2">Configuration</td><td colspan="2">Sensitivity</td><td rowspan="2">AUC</td><td rowspan="2"> $m A P _ { 5 0 }$ </td></tr><tr><td>0.25FP</td><td>0.5FP</td></tr><tr><td>Baseline</td><td> $0 . 5 1 8 \pm 0 . 0 1$ </td><td> $0 . 6 3 4 \pm 0 . 0 1$ </td><td> $0 . 8 3 7 \pm 0 . 0 1$ </td><td> $0 . 2 5 8 \pm 0 . 1 1$ </td></tr><tr><td> $\mathrm { B a s e l i n e + b k g n d } ( B )$ </td><td> $0 . 6 1 1 \pm 0 . 0 1$ </td><td> $0 . 7 5 6 \pm 0 . 0 1$ </td><td> $0 . 9 0 2 \pm 0 . 0 1$ </td><td> $0 . 3 3 1 \pm 0 . 0 1$ </td></tr><tr><td> ${ \mathrm { B a s e l i n e } } + { \mathrm { B } } + { \mathcal { G } }$ </td><td> $0 . 7 2 8 \pm 0 . 0 2$ </td><td> $0 . 8 0 6 \pm 0 . 0 2$ </td><td> $0 . 9 3 9 \pm 0 . 0 0$ </td><td> $0 . 3 7 9 \pm 0 . 0 3$ </td></tr><tr><td> ${ \mathrm { B a s e l i n e } } + B + { \mathcal { G } } + { \mathrm { f r o z e n ~ } } F$ </td><td> $0 . 7 6 6 \pm 0 . 0 1$ </td><td> $0 . 8 7 8 \pm 0 . 0 2$ </td><td> $0 . 9 3 7 \pm 0 . 0 0$ </td><td> $0 . 3 4 6 \pm 0 . 0 2$ </td></tr><tr><td>EM-DETR</td><td> $\mathbf { 0 . 8 8 5 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 9 3 2 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 5 7 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 4 2 7 \pm 0 . 0 2 }$ </td></tr></table>

Table 5: Impact of each module on classification and detection performance for pneumothorax. Experiments are conducted using the Swin-Base backbone within the MMDetection framework, with default data augmentations. The first row shows the baseline performance. The second row adds B, indicating inclusion of background annotations in the minimal-shot training file. The third row adds G, denoting the exemplar feature extraction module. In the fourth row, frozen $F$ indicates that the attention pooling module is frozen after convergence to further refine the decoder. The last row is the full EM-DETR pipeline finetuned after pretraining on base classes.

## 6.2. Ablation Studies

In this section, we conduct a few ablation studies to understand the potential, limitations, and impact of the components of EM-DETR.

Table. 5 lists the impact of each of the proposed modules, where the first row denotes the baseline grounding DINO [1] performance when the model is trained with 2 or more classes. The second row shows the efect of including background annotations and training the model on a background vs. foreground task. This is followed by the impact of our proposed feature extraction module ${ \mathcal { G } } .$ In the next experiment, we freeze the attention pooling network $F$ after convergence, to further refine the decoder for an additional 10 epochs. This experiment is conducted to showcase that training both the decoder and feature extractor simultaneously is recognized for its efect on performance[6]. In this scenario, although $m A P _ { 5 0 }$ decreases, the FROC sensitivities improve. Freezing the encoder has a similar impact on performance as freezing F.In the last experiment, we use the iterative training strategy, where we pretrain with $B _ { P }$ and finetune with $B _ { F }$ . (Stage I and Stage II training).

<table><tr><td rowspan="3">Training data</td><td colspan="2">Sensitivity</td><td rowspan="3">AUC</td><td rowspan="3"> $m A P _ { 5 0 }$ </td></tr><tr><td>0.25FP</td><td>0.5FP</td></tr><tr><td> $5 0 \ \mathrm { p t x } + \mathrm { b l g n d }$ </td><td> $0 . 6 7 9 \pm 0 . 0 1$ </td><td> $0 . 7 7 3 \pm 0 . 0 1$ </td><td>0.914</td><td> $0 . 2 6 2 \pm 0 . 2 4$ </td></tr><tr><td> $5 0 \ \mathrm { p t x } + \mathrm { F P } \ \mathrm { a s \ b k g n d }$ </td><td> $0 . 7 5 3 \pm 0 . 0 1$ </td><td> $0 . 8 5 6 \pm 0 . 0 1$ </td><td>0.920</td><td> $0 . 3 1 5 \pm 0 . 0 2$ </td></tr><tr><td> $1 0 0 \ \mathrm { p t x } + \mathrm { b k g n d }$ </td><td> $0 . 8 2 3 \pm 0 . 0 1$ </td><td> $0 . 8 9 3 \pm 0 . 0 1$ </td><td>0.951</td><td> $0 . 3 4 0 \pm 0 . 0 1$ </td></tr><tr><td> $2 0 0 \ \mathrm { p t x } + \mathrm { b k g n d }$ </td><td> $0 . 8 8 5 \pm 0 . 0 1$ </td><td> $0 . 9 3 2 \pm 0 . 0 1$ </td><td>0.957</td><td> $0 . 4 2 7 \pm 0 . 0 2$ </td></tr><tr><td> $4 0 0 \ \mathrm { p t x } + \mathrm { b k g n d }$ </td><td> $0 . 8 9 7 \pm 0 . 0 1$ </td><td> $0 . 9 3 4 \pm 0 . 0 0$ </td><td>0.964</td><td> $0 . 5 0 6 \pm 0 . 0 2$ </td></tr><tr><td> $1 0 0 \% \ \mathrm { p t x , n o \ b l g n d }$ </td><td> $0 . 9 0 4 \pm 0 . 0 1$ </td><td> $0 . 9 2 5 \pm 0 . 0 1$ </td><td>0.977</td><td> $0 . 5 1 2 \pm 0 . 0 3$ </td></tr><tr><td> $1 0 0 \% \ \mathrm { p t x + b k g n d }$ </td><td> $\mathbf { 0 . 9 1 5 \pm 0 . 0 1 }$ </td><td> $\mathbf { 0 . 9 5 6 \pm 0 . 0 1 }$ </td><td>0.980</td><td> $\mathbf { 0 . 5 6 4 \pm 0 . 0 1 }$ </td></tr></table>

Table 6: Comparison of pneumothorax performance with diferent number of positive annotations. The negative annotations are kept equal to the positive sample set in each experiment. The second row shows the impact of further retraining the model with False Positives as background annotations in a second stage of finetuning.

Table. 6 demonstrates the impact of diferent number of pneumothorax annotations. The results show a trend of improvement with more annotations, achieving near SOTA RPS levels with just 14% of annotations. Additionally in the case of 50 positive annotations, we improve results with the Stage III training phase as described in Section 4.2. This refinemenet yields a 7-8% point improvement in sensitivity and an improvement of 5.3% point in $m A P _ { 5 0 }$ . The final two rows illustrate the improved performance of training EM-DETR with and without background samples, underscoring the significance of high-quality contrasting samples for efective learning.

## 6.3. Qualitative results

Fig. 10 highlights a few examples of pleural efusion (top row) and pneumothorax detections (bottom row). In some cases, the detected regions are more tightly localized than the ground truth, which often encompasses broader areas or multiple adjacent instances of findings. This mismatch leads to an underestimation of performance in quantitative metrics such as mAP. Since DETR generates multiple hypothesis regions that overlap the target ROI and uses contrastive cosine similarity to match, several overlapping boxes may be valid positives for the same finding. However, these additional detections are counted as false positives by the evaluation framework, thereby reducing the mAP score as well.

Fig. 11 illustrates failure cases for pleural efusion detection. In most examples, the model correctly localizes the costophrenic angles and attributes the findings to the appropriate lung. However, a few images show instances where true pleural efusion regions are misclassified as background. Despite these occasional errors, the overall high AUC score suggests that the model accurately identifies abnormal regions in the majority of cases.

![](images/9182f5ac4bc036671f67694cff86b43cc0b3f73f192d82c7d1803b7309c24320.jpg)  
Figure 10: Top row illustrates examples of top 2 predictions for pleural efusion while bottom row illustrates examples of top 2 predictions for pneumothorax. The groundtruth is in yellow and predictions in green. We observe detected regions are more tightly localized than ground truth, which may span multiple nearby findings of same class, leading to an artificial reduction in mAP.

## 7. Discussion and Conclusion

EM-DETR achieves near state-of-the-art results on chest radiographs using fewer than 10% of available annotations, demonstrating a scalable approach for handling long-tailed class distributions with limited annotations. By leveraging few-shot learning to create exemplar or prototype embeddings, our model efectively diferentiates between medical findings and anatomical structures, addressing challenges unique to medical imaging. Moreover, the framework is structured to accommodate the addition of new disease classes with limited retraining, laying the groundwork for future studies in continuous learning. The method also incorporates domain-aware negative sampling, which empirically improves detection of rare classes in our few-shot setting. Our previous conference work demonstrated the generalizability of EM-DETR across multiple imaging domains, including mammography and stenosis detection. In this submission, we focus specifically on the minimalannotation aspect and its efectiveness for chest radiographs. Evaluation across multiple thoracic findings on both proprietary and public datasets demonstrates the robustness and practical utility of the approach.

![](images/4d5d248e44f0507c3f927d1d118eba6c6c9654b248b212719843f43ae3f5982e.jpg)  
Figure 11: Example images of incorrect detections of pleural efusion on the test dataset. Yellow boxes mark the groundtruth annotation, while blue indicates the predicted pleural efusion with the probability score. The background regions are marked in orange and reflect the training distribution of background annotations.

While EM-DETR demonstrates strong performance under minimal-annotation settings, there are a few limitations to discuss. The current implementation of EM-DETR is memory-intensive, as it maintains a separate list of patch embeddings and interleaves text and feature embeddings in a naive manner. Moreover, the text encoder functions primarily as a pseudo class identifier to leverage Grounding DINO pretraining, rather than contributing meaningful linguistic grounding. Future work will explore more memoryeficient integration strategies, and also extending text tokens with layerspecific feature embeddings from the multi-scale Swin backbone, enabling finer-grained feature matching and reducing ambiguity between related findings. We plan to investigate the implementation of various background prototypes to further distinguish anatomical priors. The method is also sensitive to annotation noise, as incomplete labeling of co-occurring findings can negatively afect the contrastive loss and lead to performance degradation. Addressing these aspects through improved annotation completeness and architectural refinement will be key to further enhancing EM-DETR’s scalability and robustness.

Together, these insights underscore both the promise and current boundaries of EM-DETR in practical deployment. In conclusion, despite the current limitations, which are primarily related to implementation and memory eficiency, extensive pretraining ofers the potential to further boost detection capabilities, positioning EM-DETR as a promising step toward scalable and eficient diagnostic methods in healthcare.

## 8. Declaration

During the preparation of this work the author(s) used AI tools in order to improve grammar and readability. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article.

## 9. Acknowledgment

This study is funded by Siemens-Healthineers.

## Appendix A.

Fig. A.12 illustrates the 2D TSNE [76] analysis of image backbone embeddings for the regions of interest across each class, when processed using the robust TorchXRayVision model[65].

Fig. A.13 presents an ablation study with varying number of few-shot references images for the heart in the anatomy dataset.

![](images/ab74e2f1687c004fa7eb68214fbeb544af69ca87efe3f9c87b0ca0c727229ee9.jpg)

Figure A.12: 2D TSNE analysis of feature embeddings per class when input to a torchxray vision backbone  
![](images/0ef12049c17bf5ae8fb7d2013e23037f27ea40a72d3fae76628004e88736b117.jpg)  
Figure A.13: mAP@50 performance for the Heart class across diferent few-shot settings.

## References

[1] S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, J. Zhu, L. Zhang, Grounding dino: Marrying dino with grounded pre-training for open-set object detection, in: A. Leonardis,

E. Ricci, S. Roth, O. Russakovsky, T. Sattler, G. Varol (Eds.), Computer Vision – ECCV 2024, Springer Nature Switzerland, Cham, 2025, pp. 38– 55.

[2] L. H. Li, P. Zhang, H. Zhang, J. Yang, C. Li, Y. Zhong, L. Wang, L. Yuan, L. Zhang, J.-N. Hwang, et al., Grounded language-image pretraining, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10965–10975.

[3] A. Zareian, K. D. Rosa, D. H. Hu, S.-F. Chang, Open-vocabulary object detection using captions, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 14393–14402.

[4] M. Minderer, A. A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen, et al., Simple open-vocabulary object detection with vision transformers. arxiv abs/2205.06230 (2022) (2022).

[5] A. Bulat, R. Guerrero, B. Martinez, G. Tzimiropoulos, Fs-detr: Fewshot detection transformer with prompting and without re-training, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 11793–11802.

[6] G. Zhang, Z. Luo, K. Cui, S. Lu, E. P. Xing, Meta-detr: Image-level few-shot detection with inter-class correlation exploitation, IEEE transactions on pattern analysis and machine intelligence 45 (11) (2022) 12832–12843.

[7] S. Bhat, B. Georgescu, A. B. Panambur, M. Zinnen, T.-T. Nguyen, A. Mansoor, K. K. Elbarbary, S. Bayer, F.-C. Ghesu, S. Grbic, A. Maier, Exemplar med-detr: Toward generalized and robust lesion detection in mammogram images and beyond, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2025.

[8] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, C. L. Zitnick, Microsoft coco: Common objects in context, in: Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, Springer, 2014, pp. 740–755.

[9] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, A. Zisserman, The pascal visual object classes (voc) challenge, International journal of computer vision 88 (2010) 303–338.

[10] H. Q. Nguyen, K. Lam, L. T. Le, H. H. Pham, D. Q. Tran, D. B. Nguyen, D. D. Le, C. M. Pham, H. T. Tong, D. H. Dinh, et al., Vindr-cxr: An open dataset of chest x-rays with radiologist’s annotations, Scientific Data 9 (1) (2022) 429.

[11] M. Aljabri, M. AlAmir, M. AlGhamdi, M. Abdel-Mottaleb, F. Collado-Mesa, Towards a better understanding of annotation tools for medical imaging: a survey, Multimedia tools and applications 81 (18) (2022) 25877–25911.

[12] A. K. Pujitha, J. Sivaswamy, Solution to overcome the sparsity issue of annotated data in medical domain, CAAI Transactions on Intelligence Technology 3 (3) (2018) 153–160.

[13] P. Monkam, S. Jin, W. Lu, Annotation cost minimization for ultrasound image segmentation using cross-domain transfer learning, IEEE Journal of Biomedical and Health Informatics 27 (4) (2023) 2015–2025.

[14] S. Jang, H. Song, Y. J. Shin, J. Kim, J. Kim, K. W. Lee, S. S. Lee, W. Lee, S. Lee, K. H. Lee, Deep learning–based automatic detection algorithm for reducing overlooked lung cancers on chest radiographs, Radiology 296 (3) (2020) 652–661.

[15] H. Choi, Z. Chen, X. Shi, T.-K. Kim, Semi-supervised object detection with object-wise contrastive learning and regression uncertainty, arXiv preprint arXiv:2212.02747 (2022).

[16] S. Bhat, A. Mansoor, B. Georgescu, M. Zinnen, P. Sahu, A. B. Panambur, F. C. Ghesu, S. Grbic, A. Maier, Patch-clip-contrastive health record-image joint training with patch embedding loss, ResearchSquare Preprint (2025).

[17] E. Pachetti, S. Colantonio, A systematic review of few-shot learning in medical imaging, Artificial intelligence in medicine 156 (2024) 102949.

[18] F. C. Ghesu, B. Georgescu, A. Mansoor, Y. Yoo, D. Neumann, P. Patel, R. S. Vishwanath, J. M. Balter, Y. Cao, S. Grbic, et al., Contrastive

self-supervised learning from 100 million medical images with optional supervision, Journal of Medical Imaging 9 (6) (2022) 064503–064503.

[19] P. Sloan, P. Clatworthy, E. Simpson, M. Mirmehdi, Automated radiology report generation: A review of recent advances, IEEE Reviews in Biomedical Engineering 18 (2024) 368–387.

[20] F. Yu, M. Endo, R. Krishnan, I. Pan, A. Tsai, E. P. Reis, E. K. U. N. Fonseca, H. M. H. Lee, Z. S. H. Abad, A. Y. Ng, et al., Evaluating progress in automatic chest x-ray radiology report generation, Patterns 4 (9) (2023).

[21] G. Liu, T.-M. H. Hsu, M. McDermott, W. Boag, W.-H. Weng, P. Szolovits, M. Ghassemi, Clinically accurate chest x-ray report generation, in: Machine Learning for Healthcare Conference, PMLR, 2019, pp. 249–269.

[22] M. Lin, G. Holste, S. Wang, Y. Zhou, Y. Wei, I. Banerjee, P. Chen, T. Dai, Y. Du, N. C. Dvornek, et al., Cxr-lt 2024: A miccai challenge on long-tailed, multi-label, and zero-shot disease classification from chest x-ray, arXiv preprint arXiv:2506.07984 (2025).

[23] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko, End-to-end object detection with transformers (2020). arXiv:2005. 12872. URL https://arxiv.org/abs/2005.12872

[24] A. Tafti, D. W. Byerly, X-ray radiographic patient positioning, Stat-Pearls (2020).

[25] A. Galan-Cuenca, A. J. Gallego, M. Saval-Calvo, A. Pertusa, Few-shot learning for covid-19 chest x-ray classification with imbalanced data: an inter vs. intra domain study, Pattern Analysis and Applications 27 (3) (2024) 69.

[26] J. Lian, J. Liu, S. Zhang, K. Gao, X. Liu, D. Zhang, Y. Yu, A structureaware relation network for thoracic diseases detection and segmentation, IEEE Transactions on Medical Imaging 40 (8) (2021) 2042–2052. doi: 10.1109/TMI.2021.3070847.

[27] S. K. Zhou, H. Greenspan, C. Davatzikos, J. S. Duncan, B. Van Ginneken, A. Madabhushi, J. L. Prince, D. Rueckert, R. M. Summers, A review of deep learning in medical imaging: Imaging traits, technology trends, case studies with progress highlights, and future promises, Proceedings of the IEEE 109 (5) (2021) 820–838. doi:10.1109/JPROC. 2021.3054390.

[28] C. Lin, Y. Zheng, X. Xiao, J. Lin, et al., Cxr-refinedet: Single-shot refinement neural network for chest x-ray radiograph based on multiple lesions detection, Journal of Healthcare Engineering 2022 (2022).

[29] E. Schwab, A. Gooßen, H. Deshpande, A. Saalbach, Localization of critical findings in chest x-ray without local annotations using multi-instance learning, in: 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI), 2020, pp. 1879–1882. doi:10.1109/ISBI45749.2020. 9098551.

[30] J. G. Nam, M. Kim, J. Park, E. J. Hwang, J. H. Lee, J. H. Hong, J. M. Goo, C. M. Park, Development and validation of a deep learning algorithm detecting 10 common abnormalities on chest radiographs, European Respiratory Journal 57 (5) (2021).

[31] J. Chiramal, P. Putha, T. Manoj, B. Reddy, P. Warier, Developing a novel deep learning based cavity detection algorithm for tuberculosis screening on chest xrays, in: AMERICAN JOURNAL OF TROPICAL MEDICINE AND HYGIENE, Vol. 99, AMER SOC TROP MED & HYGIENE 8000 WESTPARK DR, STE 130, MCLEAN, VA 22101 USA, 2018, pp. 642–642.

[32] P. Putha, M. Tadepalli, B. Reddy, T. Raj, J. A. Chiramal, S. Govil, N. Sinha, M. KS, S. Reddivari, A. Jagirdar, et al., Can artificial intelligence reliably report chest x-rays?: Radiologist validation of an algorithm trained on 2.3 million x-rays, arXiv preprint arXiv:1807.07455 (2018).

[33] J. Gipson, V. Tang, J. Seah, H. Kavnoudias, A. Zia, R. Lee, B. Mitra, W. Clements, Diagnostic accuracy of a commercially available deeplearning algorithm in supine chest radiographs following trauma, The British Journal of Radiology 95 (1134) (2022) 20210979.

[34] Z. Zheng, Y. Zhu, H. Wu, L. Lv, G. Yu, S. Niu, Cross-domain fewshot chest x-ray recognition, in: 2024 5th International Conference on Computers and Artificial Intelligence Technology (CAIT), IEEE, 2024, pp. 224–229.

[35] L. Lu, X. Cui, Z. Tan, Y. Wu, Medoptnet: Meta-learning framework for few-shot medical image classification, IEEE/ACM Transactions on Computational Biology and Bioinformatics 21 (4) (2023) 725–736.

[36] M. Moor, Q. Huang, S. Wu, M. Yasunaga, Y. Dalmia, J. Leskovec, C. Zakka, E. P. Reis, P. Rajpurkar, Med-flamingo: a multimodal medical few-shot learner, in: Machine Learning for Health (ML4H), PMLR, 2023, pp. 353–367.

[37] L. Ju, Y. Wu, W. Feng, Z. Yu, L. Wang, Z. Zhu, Z. Ge, Universal semisupervised learning for medical image classification, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2024, pp. 355–365.

[38] A. He, T. Li, Y. Zhao, J. Zhao, H. Fu, Open-set semi-supervised medical image classification with learnable prototypes and outlier filter, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2024, pp. 492–501.

[39] J. Liao, Y. Xu, H. Du, Q. DONG, Weakly-supervised 3d medical image segmentation using geometric prior and contrastive similarity, uS Patent App. 18/150,228 (Jul. 11 2024).

[40] Z. Kuang, Z. Yan, L. Yu, Weakly supervised learning for multi-class medical image segmentation via feature decomposition, Computers in Biology and Medicine 171 (2024) 108228.

[41] K. Li, Z. Qian, Y. Han, E. I.-C. Chang, B. Wei, M. Lai, J. Liao, Y. Fan, Y. Xu, Weakly supervised histopathology image segmentation with selfattention, Medical Image Analysis 86 (2023) 102791.

[42] W. Zhang, J. Chen, C. Kanan, Insight: Explainable weakly-supervised medical image analysis, arXiv preprint arXiv:2412.02012 (2024).

[43] R. Jiao, Y. Zhang, L. Ding, B. Xue, J. Zhang, R. Cai, C. Jin, Learning with limited annotations: a survey on deep semi-supervised learning for

medical image segmentation, Computers in Biology and Medicine 169 (2024) 107840.

[44] S. Zhang, Z. Yu, L. Liu, X. Wang, A. Zhou, K. Chen, Group r-cnn for weakly semi-supervised object detection with points, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 9417–9426.

[45] S. Wang, Z. Zhao, Z. Shen, B. Wang, Q. Wang, D. Shen, Improving selfsupervised medical image pre-training by early alignment with human eye gaze information, IEEE Transactions on Medical Imaging (2025).

[46] J. Chen, H. Duan, X. Zhang, B. Gao, V. Grau, J. Han, From gaze to insight: bridging human visual attention and vision language model explanation for weakly-supervised medical image segmentation, IEEE Transactions on Medical Imaging (2025).

[47] K. Sohn, Z. Zhang, C.-L. Li, H. Zhang, C.-Y. Lee, T. Pfister, A simple semi-supervised learning framework for object detection, arXiv preprint arXiv:2005.04757 (2020).

[48] M. Xu, Z. Zhang, H. Hu, J. Wang, L. Wang, F. Wei, X. Bai, Z. Liu, End-to-end semi-supervised object detection with soft teacher, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 3060–3069.

[49] Z. Li, Z. Zeng, Y. Liang, J.-G. Yu, Complete instances mining for weakly supervised instance segmentation, arXiv preprint arXiv:2402.07633 (2024).

[50] Q. Li, A. Arnab, P. H. Torr, Weakly- and semi-supervised panoptic segmentation, in: Proceedings of the European Conference on Computer Vision (ECCV), 2018.

[51] J.-H. Lee, C. Kim, S. Sull, Weakly supervised segmentation of small buildings with point labels, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 7406–7415.

[52] Y.-C. Chen, L. Li, L. Yu, A. El Kholy, F. Ahmed, Z. Gan, Y. Cheng, J. Liu, Uniter: Learning universal image-text representations (2019).

[53] N. Nguyen, J. Bi, A. Vosoughi, Y. Tian, P. Fazli, C. Xu, Oscar: Object state captioning and state change representation, arXiv preprint arXiv:2402.17128 (2024).

[54] Z. Li, X. Wu, H. Du, F. Liu, H. Nghiem, G. Shi, A survey of state of the art large vision language models: Benchmark evaluations and challenges, in: Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops, 2025, pp. 1587–1606.

[55] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, M. Rohrbach, Towards vqa models that can read, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8317–8326.

[56] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al., Mmbench: Is your multi-modal model an all-around player?, in: European conference on computer vision, Springer, 2024, pp. 216–233.

[57] E. Reiter, A structured review of the validity of bleu, Computational Linguistics 44 (3) (2018) 393–401.

[58] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, L. Wang, Mm-vet: Evaluating large multimodal models for integrated capabilities, arXiv preprint arXiv:2308.02490 (2023).

[59] J. Li, D. Li, C. Xiong, S. Hoi, Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation, in: International conference on machine learning, PMLR, 2022, pp. 12888– 12900.

[60] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al., Flamingo: a visual language model for few-shot learning, Advances in neural information processing systems 35 (2022) 23716–23736.

[61] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, F. Wei, Kosmos-2: Grounding multimodal large language models to the world, arXiv preprint arXiv:2306.14824 (2023).

[62] R. Windsor, A. Jamaludin, T. Kadir, A. Zisserman, Vision-language modelling for radiological imaging and reports in the low data regime, in: I. Oguz, J. Noble, X. Li, M. Styner, C. Baumgartner, M. Rusu, T. Heinmann, D. Kontos, B. Landman, B. Dawant (Eds.), Medical Imaging with Deep Learning, Vol. 227 of Proceedings of Machine Learning Research, PMLR, 2024, pp. 53–73. URL https://proceedings.mlr.press/v227/windsor24a.html

[63] F. Shakeri, Y. Huang, J. Silva-Rodriguez, H. Bahig, A. Tang, J. Dolz, I. Ben Ayed, Few-shot Adaptation of Medical Vision-Language Models , in: proceedings of Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, Vol. LNCS 15012, Springer Nature Switzerland, 2024.

[64] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, Imagenet: A large-scale hierarchical image database, in: 2009 IEEE conference on computer vision and pattern recognition, Ieee, 2009, pp. 248–255.

[65] J. P. Cohen, J. D. Viviano, P. Bertin, P. Morrison, P. Torabian, M. Guarrera, M. P. Lungren, A. Chaudhari, R. Brooks, M. Hashir, H. Bertrand, TorchXRayVision: A library of chest X-ray datasets and models, in: Medical Imaging with Deep Learning, 2022. URL https://github.com/mlmed/torchxrayvision

[66] Z. Zheng, X. Feng, H. Yu, X. Li, M. Gao, Unsupervised few-shot image classification via one-vs-all contrastive learning, Applied Intelligence 53 (7) (2023) 7833–7847.

[67] Z. Zheng, Y. Zhu, H. Wu, L. Lv, S. Niu, G. Yu, Sge: Semanticguided generalization enhancement for few-shot learning, Knowledge-Based Systems (2025) 113761.

[68] H. Wu, Y. Zhao, J. Li, Invariant and consistent: Unsupervised representation learning for few-shot visual recognition, Neurocomputing 520 (2023) 1–14.

[69] Z. Zheng, H. Wu, L. Lv, D. Bardou, S. Niu, G. Yu, Merge: multimodalenhanced representation and guided ensemble for pneumonia recognition in chest x-ray images: Z. zheng et al., The Journal of Supercomputing 81 (8) (2025) 907.

[70] H. Wu, Z. Zheng, L. Lv, C. Zhang, D. Bardou, S. Niu, G. Yu, Dara: distribution-aware representation alignment for semi-supervised domain adaptation in image classification, The Journal of Supercomputing 81 (2) (2025) 376.

[71] R. Wang, G. Zheng, Pfmnet: Prototype-based feature mapping network for few-shot domain adaptation in medical image segmentation, Computerized Medical Imaging and Graphics 116 (2024) 102406.

[72] S. Bhat, A. Mansoor, B. Georgescu, A. B. Panambur, F. C. Ghesu, S. Islam, K. Packhäuser, D. Rodríguez-Salas, S. Grbic, A. Maier, Aucreshaping: improved sensitivity at high-specificity, Scientific Reports 13 (1) (2023) 21097.

[73] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, B. Guo, Swin transformer: Hierarchical vision transformer using shifted windows, in: Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 10012–10022.

[74] D. Marin, J.-H. R. Chang, A. Ranjan, A. Prabhu, M. Rastegari, O. Tuzel, Token pooling in vision transformers for image classification, in: Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2023, pp. 12–21.

[75] F. Yang, K. Wu, S. Zhang, G. Jiang, Y. Liu, F. Zheng, W. Zhang, C. Wang, L. Zeng, Class-aware contrastive semi-supervised learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 14421–14430.

[76] L. Van der Maaten, G. Hinton, Visualizing data using t-sne., Journal of machine learning research 9 (11) (2008).

[77] K. Chen, J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu, et al., Mmdetection: Open mmlab detection toolbox and benchmark, arXiv preprint arXiv:1906.07155 (2019).

[78] H. Miller, The froc curve: A representation of the observer’s performance for the method of free response, The Journal of the Acoustical Society of America 46 (6B) (1969) 1473–1476.

[79] H. H. Pham, H. Q. Nguyen, H. T. Nguyen, L. T. Le, L. Khanh, An accurate and explainable deep learning system improves interobserver agreement in the interpretation of chest radiograph, IEEE Access 10 (2022) 104512–104531.

[80] L. Lu, S. Miao, L. Ye, Fracture detection and localization in chest x-rays using semi-supervised learning with dynamic sharpening, in: ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 1271–1275. doi: 10.1109/ICASSP43922.2022.9746079.

[81] Q. Zhang, X. Kong, Design of automatic lung nodule detection system based on multi-scene deep learning framework, IEEE Access 8 (2020) 90380–90389. doi:10.1109/ACCESS.2020.2993872.

[82] L. Maier-Hein, A. Reinke, P. Godau, M. D. Tizabi, F. Buettner, E. Christodoulou, B. Glocker, F. Isensee, J. Kleesiek, M. Kozubek, et al., Metrics reloaded: recommendations for image analysis validation, Nature methods 21 (2) (2024) 195–212.

[83] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, Focal loss for dense object detection, in: Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[84] C. Lyu, W. Zhang, H. Huang, Y. Zhou, Y. Wang, Y. Liu, S. Zhang, K. Chen, Rtmdet: An empirical study of designing real-time object detectors, arXiv preprint arXiv:2212.07784 (2022).

[85] R. Girshick, Fast r-cnn, in: Proceedings of the IEEE international conference on computer vision, 2015, pp. 1440–1448.