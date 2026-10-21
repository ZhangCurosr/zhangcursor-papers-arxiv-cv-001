# Assessing Representation Semantics Under Language Pretraining in Fine-Grained Classification

Md Raisul Kibria1\* and Sébastien Lafond¹

1Faculty of Science and Engineering, Åbo Akademi University, Turku, 20500, Finland .

\*Corresponding author(s). E-mail(s): raisul.kibria@abo.fi;

## Abstract

Understanding how supervision signals shape semantic visual representations remains challenging particularly in fine-grained, low-data settings. Standard evaluations of vision models rely on downstream accuracy, which often fails to capture structural differences in learned representations. We propose a compact explainability-driven framework for comparing encoders trained with classificationonly and caption-generation supervision on CUB-200. Using a controlled two-stage protocol in which both encoders are ultimately optimized for the same classification task, we enable a direct comparison of the resulting representations. Saliency maps are generated using attention rollout and gradient-weighted rollout and evaluated with five metrics covering faithfulness (positive and negative perturbation) and localization (keypoint emphasis, background attribution, and Earth Mover's Distance). Across random seeds and explanation methods, caption supervision yields representations that are more sensitive to the removal of salient regions, better aligned with annotated object parts, and less reliant on background regions, whereas classification supervision produces representations that are slightly more robust to the removal of low-saliency regions. An ablation study on the Oxford-IIIT Pet dataset shows consistent trends across key metrics. These results demonstrate that the proposed framework makes supervision-induced representational differences observable and quantifiable beyond downstream accuracy alone. More broadly, it provides a systematic basis for assessing learned representations and comparing alternative training and transfer-learning strategies.

Keywords: Vision-Language Supervision, Representation Explainability, Object-Centric Visual Representations, Multitask Representation Transfer

## 1 Introduction

Cross-modal representation learning has become central to modern machine learning, particularly in vision-language modeling. A core motivation is that natural language supervision can provide semantically rich signals beyond categorical labels. Contrastive pretraining approaches such as in CLIP [Radford et al., 2021] demonstrate that aligning image and text embeddings at scale yields transferable visual representations. Hybrid objectives further combine contrastive alignment with generative modeling, as in CoCa [Yu et al., 2022]. Other work studies modality imbalance [Zhang et al., 2024, Huang et al., 2025] or integrates contrastive and discriminative objectives within unified architectures [Wei et al., 2023]. Despite these advances, it remains challenging to quantify whether different supervision modalities induce systematically different visual representation structures.

Among common visual objectives, classification and image captioning provide distinct forms of supervision. Supervision signals act as inductive biases that shape the structure of learned representations. While classification labels provide discriminative constraints, caption supervision encodes structured semantic descriptions that may encourage different forms of feature organization. While large-scale captioning models such as GIT [Wang et al., 2022], BLIP [Li et al. 2022a], BLIP-2 [Li et al., 2023], LLaVA [Liu et al., 2023], and Kosmos-2 [Peng et al., 2023] illustrate the scalability of transformer-based multimodal architectures [Vaswani et al., 2017, Dosovitskiy et al., 2020], representational differences are often obscured by scale [Cornia et al., 2022 Elguendouze et al., 2023] and evaluated primarily through downstream accuracy [Sariyildiz et al. 2020, Hu et al., 2022]. However, in small-data finegrained settings, where uncertainty is higher and inductive biases play a stronger role, the choice of supervision modality may materially influence how models allocate attention to object parts and contextual cues.

This raises a fundamental question: does caption-based supervision induce measurably different visual representation structure compared to classification-based supervision, even when downstream predictive performance is similar? Answering this requires quantitative metrics and controlled comparisons that isolate supervision modality while holding architecture, optimization, and computational budget fixed.

Analyzing such differences requires systematic evaluation beyond accuracy. Attribution methods, including Integrated Gradients [Sundararajan et al., 2017], SHAP [Lundberg and Lee, 2017], LIME [Ribeiro et al., 2016], Grad-CAM [Selvaraju et al., 2017], and attention rollout [Abnar and Zuidema, 2020], provide tools for probing model behavior. However, explanation quality is sensitive to the evaluation criterion. Recent work emphasizes the importance of multi-axis evaluation assessing explainability across diverse objective dimensions to capture the complementary aspects of explanations [Kazmierczak et al., 2025, Kibria et al., 2025]. Such multi-criteria approaches are particularly valuable when comparing supervision regimes, where differences often emerge in representation alignment rather than task performance.

In this work, we investigate and quantify whether caption-based supervision transfers semantic structure which remains measurable after subsequent fine-grained classification tuning in a low-data setting. Using the CUB-200 dataset, which provides part-level annotations, we adopt a controlled two-stage protocol. In stage 1, two variants of an identical Vision Transformer (ViT) encoder [Dosovitskiy et al., 2020] are trained under different supervision: (i) classification via an attached linear head, and (ii) captioning using a frozen language decoder. In stage 2, both encoders are fine-tuned for fine-grained classification under matched optimization budgets. This design isolates the effect of supervision modality while controlling for architecture and compute.

To compare representation properties beyond predictive performance, we evaluate attribution behavior using attention rollout and gradientaugmented attention methods. We further introduce a structured evaluation protocol integrating five explainability metrics across two categories: localization, measured by keypoint emphasis, background fraction, and distributional alignment via Earth Mover's Distance (EMD); and faithfulness, measured by perturbation AUC under positive and negative masking. Leveraging CUB's part annotations enables objective quantification of semantic alignment. An overview of the proposed framework is presented in Figure 1. We further assess robustness through two ablations: a cross-dataset evaluation on a second fine-grained benchmark and an encoder-scale ablation using a larger ViT from the same architectural family.

Empirically, we find that caption-based pretraining produces consistent and reproducible differences in attribution structure, including stronger correspondence to annotated keypoints, improved geometric alignment with keypoint density maps, and reduced attribution to background regions. These trends persist across attribution methods, datasets, and encoder scales, suggesting that language supervision can shape visual representations in ways that are not reflected by classification accuracy alone. Although captionbased pretraining does not improve downstream accuracy under fixed computational budgets, the explainability-based analysis reveals a more general implication: supervision modality can affect the semantic organization of learned representations independently of predictive risk. Our results therefore position multi-criteria attribution evaluation as a useful tool for quantifying representation transfer beyond task performance.

![](images/4cd78e25612f749123cb565b29f9dae321b79f6856a9fa105368385dcf939660.jpg)  
Fig. 1: Overview of the proposed two-stage training and evaluation framework. (a) Stage 1 performs task-specific base tuning via either classification or captioning objectives. (b) Stage 2 applies controlled downstream finetuning for fine-grained classification using identical optimization budgets. (c) Explainability evaluation is conducted using multi-criteria metrics including perturbation-based faithfulness and localization measures.

## 2 Related Work

## Language supervision in ine-grained recognition.

Language supervision has been used to enrich visual representations in fine-grained settings such as CUB-200. Hendricks et al. [2016] connected caption generation with grounded visual explanations, while Reed et al. [2016] learned joint image-text embeddings that improved retrieval and zero-shot classification. Similarly, He and Peng [2017] proposed a two stream model, separately trained for vision and language supervision for learning robust semantic representations. These works demonstrate that textual supervision can shape transferable features.

More recent multimodal representation learning studies including large-scale vision-language pretraining and contrastive objectives further show the potential of cross-modal transfers such as induction of semantically dense and localizationaware representations [Radford et al., 2021, Desai and Johnson, 2021, Li et al., 2022b, El Banani et al., 2023]. However, most studies evaluate such representations through downstream transfer performance, large-scale pretraining effectiveness, or task accuracy improvements, leaving open the question of how language supervision alters semantic alignment under controlled training budgets in small-data regimes.

## Supervision modality and interpretability.

Language-based supervision has also been studied in reasoning and multitask settings. The VCR benchmark [Zellers et al., 2019] highlights the role of language in grounding visual reasoning. Parelli et al. [2023] distill reasoning supervision into visual models and report improvements in taskaligned faithfulness measures. Similarly, Najjar et al. [2025] show that auxiliary tasks can improve interpretability when strongly correlated with the primary objective. In contrast, we consider caption supervision applied only during pretraining and removed at evaluation time, enabling analysis of cross-modal transfer effects without explicit auxiliary objectives during classification.

Quantitative evaluation of representation alignment.

For transformer-based models, attention-driven attribution methods such as attention rollout [Abnar and Zuidema, 2020] and gradientbased localization [Selvaraju et al., 2017, Chefer et al., 2021] provide tools to probe learned representations. However, explainability is often evaluated using a single metric or qualitative inspection. Recent work argues that systematic evaluation of explainability should involve multi-dimensional analysis to account for their multi-faceted and subjective nature [Nauta et al., 2023, Kibria et al., 2025].

Overall, prior work establishes that language supervision can improve transfer performance. Our contribution is to examine, under matched architectures and compute, whether caption-based pretraining induces measurable differences in representation alignment in low-data fine-grained settings.

## 3 Methods

This section describes the dataset, training protocols, explanation methods, and the proposed explainability evaluation framework.

## 3.1 Dataset

We conduct experiments on the CUB-200-2011 dataset, which contains 11,788 images across 200 bird species and is widely used for fine-grained visual classification. In addition to class labels, the dataset provides object bounding boxes and 15 annotated part keypoints per image. These keypoints correspond to semantically meaningful bird parts (e.g., beak, wing, tail) and are commonly used to evaluate part-level localization. The dataset also includes binary attribute annotations and segmentation masks. Fine-grained visually descriptive captions without including the species or the background have been constructed in prior work for the dataset [Reed et al., 2016] and are used in this study for caption supervision. Further details regarding dataset annotations are provided in Appendix A.

## 3.2 Models

To evaluate the effect of supervision modality, we compare two training variants that differ only in the objective used during the first stage (Stage 1) of pretraining. Both variants share the same vision backbone: a Vision Transformer (ViT-B/16) initialized with ImageNet pretraining.

We note that ImageNet contains certain bird categories that may overlap semantically with CUB-200 classes. While exact image overlap is not present, this semantic proximity may influence representation transfer. However, this factor is not expected to confound our comparison because ImageNet initialization is held fixed across all experimental conditions, therefore, is used as a controlled common starting point to ensure stable optimization in the low-data regime; in contrast, training from scratch leads to substantially reduced performance and unstable convergence.

## Classification-tuned variant.

In the first variant, the encoder is trained with a standard classification objective by attaching a multi-layer perceptron (MLP) head to the [CLS] token representation.

## Caption-tuned variant.

In the second variant, caption supervision is introduced by attaching a pretrained GPT-2 decoder [Radford et al., 2019] to the encoder through crossattention layers. During Stage 1 training, most of the decoder parameters are frozen except the cross-attention weights, and the vision encoder is updated. This design isolates the effect of textual supervision on visual representations while preventing adaptation within the language model. Since the decoder is used only during Stage 1 training and primarily in the forward pass, with limited parameter updates, the additional computational overhead remains moderate.

In the second stage (Stage 2), both encoder variants are fine-tuned for downstream finegrained classification under identical optimization settings. In this stage, a classification MLP head is trained on top of the encoder ([CLS] token representation), and no caption loss is applied. The overall two-stage architecture is illustrated in Figure 1 (a-b).

## 3.3 Explanation Algorithms

To analyze representation differences beyond predictive accuracy, we employ attribution methods tailored to transformer-based architectures.

## Attention Rollout.

We apply attention rollout [Abnar and Zuidema 2020], which aggregates attention weights across layers to estimate token-level influence. This method relies solely on forward-pass attention maps and produces class-agnostic attribution scores. Given that each image in CUB-200 contains a single labeled object, class-agnostic saliency remains informative for assessing spatial focus.

## Gradient-weighted Attention Rollout.

We additionally employ the transformer attribution method inspired from the work of Chefer et al. [2021], which combines attention weights with gradients of the target logit with respect to attention scores. This approach introduces class-specific information into the attribution and is conceptually related to gradient-based localization methods such as Grad-CAM [Selvaraju et al., 2017]. By incorporating gradient signals, the resulting saliency maps reflect the contribution of image regions to the predicted class.

The detailed mathematical formulation of both adapted algorithms is provided in Appendix B.

Using both methods allows us to distinguish between general spatial attention patterns and class-conditional feature attribution, which is essential for evaluating representational differences induced by supervision modality.

## 3.4 Explanation Evaluation

Evaluating explanations requires multiple complementary criteria, as no single metric fully captures explanatory quality. We therefore assess explanations along two objective dimensions: faithfulness and localization. Faithfulness quantifies the extent to which saliency values reflect the true contribution of input regions to the model's prediction, while localization evaluates the spatial correspondence between saliency and annotated semantic structure. All metrics are computed identically across supervision variants.

## 3.4.1 Faithfulness: Positive and Negative Perturbation

We adopt a perturbation-based protocol following Chefer et al. [2021], in which features are progressively masked according to their saliency ranking. The resulting change in prediction confidence is measured as the reflection of explanation faithfulness.

Let $\begin{array} { r c l } { S } & { \in } & { [ 0 , 1 ] ^ { H \times W } } \end{array}$ denote the normalized saliency map for an input image I. Since both attribution methods operate at patch level (ViT-$\mathrm { B } / 1 6$ , 14×14 grid), masking is performed over patches rather than individual pixels. Let $\tau \in$ $\{ 0 , 5 , 1 0 , \ldots , 1 0 0 \}$ denote percentile thresholds over saliency values. We define two perturbation schemes:

## Positive perturbation (most-salient first).

Regions with saliency above threshold $\tau$ are masked, as in Equation 1.

$$
I _ { \tau } ^ { + } ( x , y ) = \left\{ \begin{array} { l l } { \mu , } & { S ( x , y ) \geq \tau } \\ { I ( x , y ) , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{1}
$$

Negative perturbation (least-salient first).

Regions with saliency below threshold $\tau$ are masked, as in Equation 2.

$$
I _ { \tau } ^ { - } ( x , y ) = { \left\{ \begin{array} { l l } { I ( x , y ) , } & { S ( x , y ) \geq \tau } \\ { \mu , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{2}
$$

Here, $\mu$ denotes the per-channel median pixel value computed over the training set. This baseline mitigates out-of-distribution artifacts that can arise from zero-masking.

For each $\tau ,$ we compute the predicted probability of the target class. Plotting confidence as a function of masking percentage yields perturbation curves. We summarize these curves using area under the curve (AUC) using $A U C _ { I ^ { + } }$ and $A U C _ { I ^ { - } }$ , respectively for positive and negative perturbations. A faithful explanation yields a rapid decrease in confidence under positive perturbation resulting in a lower $A U C _ { I ^ { + } }$ value whereas under negative perturbation, confidence should remain comparatively stable resulting in a higher $A U C _ { I }$ – value. The resulting $A U C _ { I } .$ + and $A U C _ { I }$ provide quantitative measures of sensitivity and robustness to feature removal.

A visual example of the progressive masking in both positive and negative directions is presented in Appendix C.

## 3.4.2 Localization

Using the segmentation masks and keypoint annotations available in the dataset, we measure localization at complementary scales: objectlevel (fraction of background attribution), partlevel (keypoint emphasis), and spatial-distribution alignment via EMD.

## Background Fraction (Object-level localization).

Let B denote the set of background pixels derived from the segmentation mask. The background fraction measures the proportion of total saliency assigned to background regions as in Equation 3.

$$
F _ { b g } = \frac { \sum _ { p \in B } S ( p ) } { \sum _ { p \in \mathrm { i m a g e } } S ( p ) } .\tag{3}
$$

Lower values indicate stronger saliency within object regions and reduced reliance on background context.

## Keypoint Emphasis Ratio (Part-level localization).

Fine-grained recognition often depends on localized discriminative parts, such as the beak, wing, or tail. To quantify how strongly a model emphasizes these regions, we introduce a new metric: the Keypoint Emphasis Ratio. Keypoint Emphasis Ratio measures the relative saliency assigned to annotated part locations compared to the remaining foreground region. Formally, let $\{ C _ { k } \} _ { k = 1 } ^ { K }$ denote circular neighborhoods centered at the K annotated keypoints. The radius R is scaled with object size to ensure scale-adaptive neighborhoods calculated using Equation 4.

$$
R = \operatorname* { m a x } \left( 1 , \lfloor \alpha \cdot \kappa \rfloor \right) ,\tag{4}
$$

where $\alpha \in ( 0 , 1 ]$ reflects normalized object area and κ $; > 0$ controls spatial extent.

The average saliency within keypoint neighborhoods is

$$
\mathrm { K P _ { s a l } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { | C _ { k } | } \sum _ { p \in C _ { k } } S ( p ) .
$$

Let M denote the binary object mask. The mean saliency over object regions excluding keypoint neighborhoods is

$$
{ \mathrm { O b j } } _ { \mathrm { o t h e r } . { \mathrm { s a l } } } = { \frac { 1 } { | M \setminus \cup _ { k = 1 } ^ { K } C _ { k } | } } \sum _ { p \in M \setminus \cup _ { k = 1 } ^ { K } C _ { k } } S ( p ) .
$$

We define the Keypoint Emphasis Ratio as in Equation 5.

$$
R _ { \mathrm { k p } } = { \frac { \mathrm { K P _ { s a l } } } { \mathrm { O b j _ { o t h e r \it { s a l } } } + \epsilon } } , \qquad \epsilon = 1 0 ^ { - 8 } .\tag{5}
$$

Values greater than one indicate preferential concentration of saliency around annotated discriminative parts relative to other object regions. A demonstration for example inputs on how the metric is calculated is presented in Appendix D.

## Earth Mover's Distance (Spatial alignment).

While $F _ { \mathrm { b g } }$ and $R _ { \mathrm { k p } }$ quantify saliency concentration within foreground and part regions, they do not account for spatial proximity. To measure geometric alignment between attribution maps and annotated discriminative parts, we compute the EMD between the attribution map and a Gaussianized keypoint density map.

Given a set of annotated keypoints $( x _ { k } , y _ { k } ) _ { k = 1 } ^ { K } ,$ we first construct a Gaussianized keypoint density map. Each keypoint contributes a Gaussian kernel centered at its annotated location, as in Equation 6.

$$
G ( \boldsymbol { p } ) = \sum _ { k = 1 } ^ { K } \exp \left( - \frac { | \boldsymbol { p } - ( \boldsymbol { x } _ { k } , \boldsymbol { y } _ { k } ) | _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{6}
$$

where σ controls the spatial extent of each keypoint neighborhood. The resulting density map is multiplied by the object segmentation mask to suppress background regions. Appendix E illustrates an example Gaussianized keypoint density map.

The spatial alignment is then quantified from the attribution map $S$ and keypoint density map $G ,$ as in Equation $^ { 7 . }$

$$
D _ { \mathrm { E M D } } = \mathrm { E M D } ( \hat { S } , \hat { G } ) ,\tag{7}
$$

where $\hat { S }$ and $\hat { G }$ denote the normalized attribution and keypoint distributions, respectively. Lower values of $D _ { \mathrm { E M D } }$ indicate stronger agreement between the spatial distribution of saliency and annotated discriminative parts. Unlike ratiobased metrics, EMD is geometry-aware and assigns smaller penalties to nearby localization errors than to distant ones, providing a complementary measure of spatial localization quality.

## 4 Experiments

This section details the training protocol, experimental setup, and associated hyperparameters.

## 4.1 Training Setup

Experiments were conducted on a single cluster node equipped with four NVIDIA V100 GPUs. Image captioning models were trained using the Huggingface API, enabling seamless multi-GPU training, while classification models were trained using PyTorch.

We use the CUB-200 dataset with 11,788 images, split into training, validation, and test subsets with proportions 50.9%, 24.7%, and 24.4%, respectively. For caption supervision, we use the Reed et al. [2016] caption extension, which provides ten natural-language descriptions per image, resulting in ten image-caption pairs for each image while preserving the same imagelevel splits used for classification. Captions are tokenized with the pretrained GPT-2 tokenizer, and images are processed using the standard ViT preprocessing pipeline [Dosovitskiy et al., 2020].

## 4.2 Training Protocol

## 4.2.1 Stage 1: Base Encoder Tuning

During the first stage of training, two encoder variants are produced. Hyperparameters were chosen based on informed intuition and resulting experimentation without aiming for exhaustive optimization, as the optimal values are not crucial for the primary objectives of this study.

![](images/cd89eb987c1fb234dceafc7a15e1d719c7e8c08e6b19fb05f8609af596f1d634.jpg)  
(a) Training and Validation Loss

![](images/a67498427b820966bff5c16160dc6a5f367974241e35ae8aa695327a41105b82.jpg)  
(b) Test Accuracy  
Fig. 2: Stage 1 training progress for the classification-tuned encoder. (a) training and validation loss; (b) test accuracy each averaged over three seeds with CI.

## Classification-tuned encoder.

The ImageNet-pretrained ViT is fine-tuned on CUB-200 for five iterations with a smaller learning rate for the encoder $( 5 \times 1 0 ^ { - 5 } )$ and a larger rate for the classification head $\left( 1 \times 1 0 ^ { - 3 } \right)$ , enabling gradual adaptation of the backbone. Figure 2 shows the evolution of training and validation loss, and test accuracy averaged over three seeds with 95% confidence intervals (CI). The final iteration achieves a mean test accuracy of $0 . 8 4 5 4 \pm 0 . 0 1 4 8$ , with the highest mean accuracy observed at iteration 3.

## Caption-tuned encoder.

The image captioning pipeline is trained using a teacher-forcing approach. Here, only the encoder and the cross-attention weights of the decoder are fine-tuned, while the remaining decoder parameters remain frozen. Similar to the first variant, a smaller learning rate of $3 \times 1 0 ^ { - 5 }$ is employed to train the encoder with a cross-entropy loss on the token outputs. Due to the limited diversity of captions in the dataset and the partial fine-tuning of the pipeline, the resulting captions exhibit only moderate quality, as illustrated in Figure 3. Nevertheless, for the purposes of this study, the injection of external knowledge is prioritized over maximizing captioning accuracy.

![](images/b2c59f31cf4de4ec79de168707944a56d8e56680d1db85d8e3446642e9f50e15.jpg)  
Fig. 3: Qualitative captioning results under teacher forcing. Captions are color-coded based on maximum token-level F1 score against any of the 10 ground-truth captions (green: good match, orange: partial, red: poor).

## 4.2.2 Stage 2: Downstream Classification Fine-Tuning

For each encoder variant, a classification head is added and fine-tuned for five iterations. A small learning rate $( 1 \times 1 0 ^ { - 5 } )$ is used for the encoder, while the classification head uses a larger rate $( 1 \times 1 0 ^ { - 3 } )$ , primarily, to guide the captiontuned encoder towards label alignment. Training summaries for both variants are shown in Figure 4.

For the classification-tuned encoder, the training loss decreases gradually while validation loss remains stable (Figure 4a). Test accuracy rises steadily, with minor fluctuations due to the large head learning rate (Figure 4b). The caption-tuned encoder initially exhibits higher loss due to misalignment but stabilizes quickly, achieving comparable final accuracy. Final Stage 2 test accuracies (mean ± 95% CI over five iterations) are summarized in Appendix F. Classification-tuned encoders achieve 0.8566 ± 0.0114, whereas caption-tuned encoders reach $0 . 8 4 7 3 \ \pm \ 0 . 0 0 6 2$ , slightly lower under identical training budgets.

## 5 Results

We analyze the impact of the two training regimes beyond classification accuracy using explainability-oriented metrics computed over the test set across three random seeds. To assess whether supervision modality induces measurable differences in visual concept utilization, we report both quantitative analysis and qualitative trends.

![](images/5eee8938b9e875d1764ffbe5979618c1d8f9819a0dade02cc3899de8c547f42f.jpg)  
(a) Classification encoder: Training and Validation Loss

![](images/5df2c10bcc8c421d62451ed4e5e95ad0313c01396d1d9e433fd3cbc6dcd4abf7.jpg)

![](images/6099289695be83b71b2f298d66e47a2c0069f64cb6c4db4027b675286849746d.jpg)  
(c) Caption encoder: Training and Validation Loss  
(b) Classification encoder: Test Accuracy

![](images/62f738588366d7e85a0ba74f606694980862cd78fabb61eb0ce7d305bd7b5762.jpg)  
(d) Caption encoder: Test Accuracy  
Fig. 4: Second-stage training progress over iterations averaged over three seeds with CI. Top row: classification-tuned encoder. Bottom row: caption-tuned encoder. Left: loss curves. Right: test accuracy with Stage 1 baseline.

## 5.1 Perturbation-based Evaluation of Explanations

We first evaluate explanation faithfulness using perturbation-based analysis. Following prior work, image regions are progressively masked according to saliency rankings while tracking the model's predicted probability for the ground-truth class. We consider positive and negative perturbations.

Figure 5 shows the mean normalized prediction probability as a function of masking ratio for attention rollout and gradient-weighted attention explanations across classification- and captiontuned encoders. The curves reflect explanation faithfulness in terms of sensitivity (positive perturbation) and robustness (negative perturbation).

![](images/70805835c857980ec33cbe9f1c68644106f03df51ffb709594d38fd49c949be2.jpg)  
(a) Perturbation Curves for Attention Rollout

![](images/00e856731a642a929b5f2c9da19b9c375a100d14563c91d70f66c682e3a6335f.jpg)  
(b) Perturbation Curves for Gradient-weighted Rollout

Fig. 5: Positive and negative perturbation curves under progressive masking based on saliency maps from two explanation methods. Blue curves correspond to the classification-tuned encoder and red curves to the caption-tuned encoder. Solid lines indicate positive perturbation (descending saliency masking), and dashed lines indicate negative perturbation (ascending saliency masking). Curves are normalized by the unmasked prediction probability.

Under negative perturbation, both encoders exhibit nearly similar stability across masking levels, indicating regions identified as minimally important can be removed without substantial degradation in predictive confidence. This suggests comparable robustness in discarding lowsaliency regions, with the classification-tuned encoder showing a slight advantage.

In contrast, under positive perturbation, the caption-tuned encoder shows a consistently steeper decline in confidence as salient regions are progressively removed. This pattern appears across both attention-based and gradientweighted explanations and persists throughout masking levels. The sharper degradation indicates that the caption-tuned model concentrates predictive mass on regions identified as salient, consistent with improved faithfulness.

Overall, these results suggest that caption supervision does not primarily increase reliance on contextual cues. Instead, it promotes stronger dependence on a smaller set of semantically meaningful regions, such that removing them results in a pronounced performance drop. While this increased sensitivity reflects improved alignment between saliency and model reliance, it does not imply enhanced robustness to distribution shift or adversarial perturbations, which remain open questions.

## 5.2 Localization-based Evaluation of Explanations

We next evaluate spatial alignment using three localization metrics: the Keypoint Emphasis Ratio $( R _ { k p } )$ , the background fraction $\left( F _ { b g } \right)$ and EMD $\left( D _ { E M D } \right)$ . These metrics compare saliency maps against ground-truth keypoint annotations and segmentation masks from complementary perspectives. For each seed, scores are averaged over the test set, yielding one aggregate value per training regime.

Higher values for the Keypoint Emphasis Ratio $R _ { k p }$ indicate stronger alignment with discriminative object parts. The background fraction $F _ { b g }$ quantifies the proportion of saliency assigned to background pixels; lower values correspond to more object-centric explanations. Given the finegrained nature of the dataset, reduced background attribution is a meaningful indicator of improved localization. $D _ { E M D }$ captures the spatial alignment between the attribution distribution and Gaussianized keypoint density maps. Thus, lower $D _ { E M D }$ values indicate better localization and closer correspondence between model attributions and keypoint density.

Figure 6 presents violin plots of $R _ { k p } , \ F _ { b g }$ and $D _ { E M D }$ across three seeds for both training regimes and explanation methods.

Across both explanation methods, caption tuning consistently yields higher $R _ { k p }$ values with reduced inter-seed variability compared to classification tuning. This indicates stronger and more stable alignment with discriminative object parts.

![](images/afbe092b845dbd1379b25db42025e0e3d7d328a4fd11f392538593376399594a.jpg)

![](images/3346e33a05c056addf4861038ccc188c1358cfce910bcec618f114b7c59b3cf0.jpg)

![](images/ef7cfd9432276985d65f8c010c3ebc23b803c704574f52138ee66431318038f0.jpg)  
(a) Distribution of Localization Metrics for Attention Rollout

![](images/9259c9b302d7d354afef49fd456e121a2cf60881817385c50b4337e802f1f6b2.jpg)

![](images/8ce05b8ea660dcd399ae464c10cd0e4cd59fdb7ca7f3ba055229af35c0a97512.jpg)

![](images/3310f2318843739cd5e767253ae1c9c6e8a06b386a51e066807ecd3b9f1e3df5.jpg)  
(b) Distribution of Localization Metrics for Gradient-weighted Rollout  
Fig. 6: Violin plots of localization metrics across three random seeds for classification- and captiontuned encoders. Results are shown for (a) attention rollout and (b) gradient-weighted rollout. Blue violins correspond to the classification-tuned encoder and red violins to the caption-tuned encoder. Points denote seed-wise averages and the central horizontal line indicates the median.

This improvement is also maintained across different radius values (see Appendix G). Similarly, caption-tuned encoders produce uniformly lower $F _ { b g }$ values, reflecting reduced saliency allocation to background regions.

While the improvement in $D _ { E M D }$ is smaller than for the ratio-based metrics, caption-tuned models consistently obtain lower values across both attribution methods and all seeds, indicating closer alignment between the saliency and target spatial distributions.

Collectively, these results indicate that caption supervision systematically biases representations toward foreground-centric and semantically grounded explanations. The consistency of these effects across two attribution methods and multiple seeds suggests that the observed improvements arise from the training objective rather than artifacts of a particular saliency algorithm.

## 5.3 Statistical Analysis of the Metrics

All five metrics were found to be non-normally distributed using the Shapiro-Wilk test. We therefore apply the Wilcoxon signed-rank test to compare paired per-image metric values between classification- and caption-tuned encoders. Effect sizes are reported as paired Cohen's d. For each seed, metrics were computed over the full test set; pooled paired samples across three seeds yield $n =$ 8,646 comparisons per metric. Statistical results for both explanation methods are summarized in Table 1.

All metrics show statistically significant paired differences $( p \ll 0 . 0 0 1 )$ . For perturbation-based faithfulness, the results show a consistent tradeoff. Caption tuning consistently yields lower values across both saliency methods for $A U C _ { I ^ { + } }$ , with small-to-moderate effect sizes $( d = 0 . 3 8 8 , 0 . 4 1 0 )$ 2 reflecting stronger confidence degradation when salient regions are removed. In contrast, classification tuning achieves higher $A U C _ { I }$ - , indicating greater stability when low-saliency regions are removed. Thus, caption tuning produces stronger sensitivity to removal of salient evidence, while classification tuning remains more robust to removal of low-ranked regions.

Table 1: Comparison between Classification- and Captioning-tuned encoder training regimens using Wilcoxon signed-rank tests $( n = 8 6 4 6$ paired samples per metric). Effect sizes reported as paired Cohen's d. For the five metrics, expected directions are: $A U C _ { I ^ { + } }$ ↓; $A U C _ { I }$ - ↑; $F _ { b g }$ ↓; $R _ { k p }$ ↑; $D _ { E M D }$ ↓.
<table><tr><td>Saliency Method</td><td>Metric</td><td>Regimen</td><td>Mean</td><td>Std</td><td>95% CI</td><td>p-value</td><td>Cohen&#x27;s d</td></tr><tr><td rowspan="6">Grad-Weighted</td><td> $A U C _ { I ^ { + } }$ </td><td>Classification Captioning</td><td>0.1762 0.1328</td><td>0.1430 0.1074</td><td>[0.1733, 0.1793] [0.1305, 0.1351]</td><td> $5 . 3 3 \times 1 0 ^ { - 3 0 7 }$ </td><td>0.388</td></tr><tr><td> $A U C _ { I ^ { - } }$ </td><td>Classification Captioning</td><td>0.7716 0.7048</td><td>0.1844 0.2032</td><td>[0.7675, 0.7754] [0.7004, 0.7091]</td><td> $< 1 0 ^ { - 3 0 8 }$ </td><td>0.405</td></tr><tr><td> $F _ { b g }$ </td><td>Classification Captioning</td><td>0.6854 0.6144</td><td>0.1072 0.1115</td><td>[0.6831, 0.6877] [0.6120, 0.6168]</td><td> $< 1 0 ^ { - 3 0 8 }$ </td><td>0.741</td></tr><tr><td> $R _ { k p }$ </td><td>Classification Captioning</td><td>1.1339 1.2246</td><td>0.3318 0.4299</td><td>[1.1271, 1.1411] [1.2156, 1.2338]</td><td> $1 . 3 2 \times { { 1 0 } ^ { - 9 6 } }$ </td><td>-0.227</td></tr><tr><td> $D _ { E M D }$ </td><td>Classification Captioning</td><td>0.1483 0.1443</td><td>0.0304 0.0329</td><td>[0.1476, 0.1489] [0.1436, 0.1449]</td><td> $2 . 3 8 \times 1 0 ^ { - 3 9 }$ </td><td>0.1286</td></tr><tr><td rowspan="6">Attention Rollout</td><td> $A U C _ { I ^ { + } }$ </td><td>Classification Captioning</td><td>0.1199 0.0898</td><td>0.1167 0.0806</td><td>[0.1174, 0.1224] [0.0880, 0.0915]</td><td> $< 1 0 ^ { - 3 0 8 }$ </td><td>0.410</td></tr><tr><td> $A U C _ { I ^ { - } }$ </td><td>Classification Captioning</td><td>0.7822 0.7135</td><td>0.1767 0.1987</td><td>[0.7782, 0.7859] [0.7092, 0.7176]</td><td> $< 1 0 ^ { - 3 0 8 }$ </td><td>0.434</td></tr><tr><td> $F _ { b g }$ </td><td>Classification Captioning</td><td>0.6122 0.5644</td><td>0.1184 0.1225</td><td>[0.6096, 0.6147] [0.5617, 0.5670]</td><td> $< 1 0 ^ { - 3 0 8 }$ </td><td>1.768</td></tr><tr><td> $R _ { k p }$ </td><td>Classification Captioning</td><td>1.1015 1.1127</td><td>0.1925 0.1896</td><td>[1.0976, 1.1056] [1.1089, 1.1168]</td><td> $2 . 5 6 \times 1 0 ^ { - 1 4 5 }$ </td><td>-0.220</td></tr><tr><td> $D _ { E M D }$ </td><td>Classification Captioning</td><td>0.1419 0.1328</td><td>0.0249 0.0247</td><td>[0.1414, 0.1424] [0.1323, 0.1333]</td><td> $< 1 0 ^ { - 3 0 8 }$ </td><td>0.8882</td></tr></table>

For localization, caption tuning produces lower $F _ { b g }$ with moderate-to-large effect sizes, indicating reduced background attribution. Gains in $R _ { k p }$ for caption-tuning are smaller (small effect sizes) but consistent across the saliency methods, indicating stronger alignment with annotated keypoints. The same trend is reflected by $D _ { E M D }$ , where caption tuning improves geometric alignment with keypoint locations, with a small effect for Grad-Weighted Rollout and a larger effect for Attention Rollout.

Together, the statistical results confirm systematic differences in representation structure induced by supervision modality that are not captured by classification accuracy alone.

## 5.4 Qualitative Comparison

To complement quantitative analysis, we present representative test images with predicted labels and overlaid saliency maps for both training regimes and explanation methods in Figure 7.

Across examples, gradient-weighted rollout exhibits higher-frequency artifacts, particularly in background regions, likely due to the sensitivity of gradients and upsampling from patch to pixel resolution. Attention rollout produces smoother, more spatially coherent maps.

Comparing training regimes, differences are subtle but consistent with quantitative trends. Caption-tuned models show reduced background activation and more spatially concentrated attributions, particularly under gradient-weighted rollout. This pattern appears for both correct and incorrect predictions, suggesting a structural shift in how evidence is distributed rather than a byproduct of accuracy differences.

![](images/e3b5ba5584898a5410999aa8441d775bcca466e429c91e1dccd74d9c6ac3db5c.jpg)  
Fig. 7: Saliency map visualizations for classification- and caption-tuned models. Each row corresponds to one test image. Column 1 shows the original image with the true label. Columns 2–5 show saliency overlays: classification-tuned and caption-tuned encoders both under attention rollout and gradient-weighted rollout. Predicted labels are shown above each heatmap (green = correct, red = incorrect). A shared colorbar indicates saliency intensity.

While qualitative maps support the statistical findings, they should be interpreted cautiously given the non-causal nature of saliency methods. Conclusions rely primarily on systematic quantitative evaluation.

## 6 Ablation Studies

We conduct two ablation studies to examine whether the trends observed on CUB-200 are robust to changes in dataset and model capacity. First, we evaluate the same protocol on the Oxford-IIIT Pet dataset, which provides a second fine-grained recognition setting with foreground segmentation masks. Second, we repeat the CUB-200 experiment using a larger ViT-L/16 encoder to test whether the attribution trends persist beyond a single encoder scale.

## 6.1 Results on the Oxford-IIIT Pet Dataset

The Oxford-IIIT Pet dataset contains 37 finegrained pet categories, comprising 12 cat breeds and 25 dog breeds, with 7,349 images in total. In contrast to CUB-200, the dataset does not provide part-level keypoint annotations, but includes pixel-level segmentation masks separating the foreground pet from the background. We therefore evaluate background reliance using the background saliency fraction, $F _ { b g } .$ together with the perturbation-based faithfulness metrics. For the captioning objective, we use an augmented version of the dataset in which image captions describing the foreground pet and surrounding scene are generated using BLIP-2 [Li et al., 2023]. We use an 80–10–10 train-validation-test split of the original dataset.

We apply the same two-stage protocol as in the main CUB-200 experiments: task-specific encoder tuning with either classification or captioning supervision, followed by controlled downstream classification fine-tuning under matched optimization budgets. We reduced the second-stage finetuning schedule relative to CUB-200 (2 epochs instead of 5), as the smaller dataset reached stable validation performance with fewer updates, however, comparison between supervision modalities remains controlled. Table 2 reports results over three random seeds using Wilcoxon signed-rank tests on paired test examples.

The Oxford-IIIT Pet ablation shows the same qualitative pattern as the main CUB-200 analysis. Caption tuning consistently reduces $A U C _ { I ^ { + } }$ across both saliency methods, indicating sharper degradation when the most salient regions are removed. It also reduces $F _ { b g }$ , suggesting lower background attribution and stronger concentration on the foreground object. These effects are statistically significant for both explanation methods, with small-to-moderate effects for perturbation faithfulness and moderate-to-large effects for background reliance.

As in CUB-200, classification-tuned models retain higher $A U C _ { I ^ { - } }$ , indicating weaker sensitivity to removal of least salient regions. While this effect remains statistically significant, its magnitude is comparatively smaller.

Overall, despite differences in dataset structure and annotation granularity, the qualitative trends are preserved. This consistency supports the conclusion that supervision modality induces systematic and reproducible differences in representation structure, rather than dataset-specific artifacts.

## 6.2 Encoder-Scale Ablation with ViT-L/16

To assess whether the observed trends are specific to the ViT-B/16 backbone, we conduct an encoder-scale ablation using an ImageNetpretrained ViT-L/16. This preserves the same architectural family and attribution pipeline while evaluating the framework under a higher-capacity encoder. The two-stage training protocol and matched downstream tuning setup are kept consistent with the main experiments, with the optimization budget adjusted for the larger model.

Table 3 shows that the ViT-L/16 results preserve the same directional trends observed in the main experiments. Across both Grad-Weighted Rollout and Attention Rollout, caption tuning improves the preferred direction for $A U C _ { I ^ { + } }$ $F _ { b g } , R _ { k p }$ and $D _ { E M D }$ , while classification tuning remains stronger for $A U C _ { I }$ -. Although the magnitude of the differences is generally smaller than with ${ \mathrm { V i T - B } } / { 1 6 }$ particularly for Attention Rollout, the consistency across metrics and attribution methods supports the robustness of the observed supervision-dependent attribution patterns.

## 7 Discussion and Conclusion

In this work, we present a controlled empirical framework for analyzing multitask transferability from image captioning to fine-grained classification through explainability-based quantification of learned visual representations. Using CUB-200 as the primary benchmark, we compare classification- and caption-tuned Vision Transformer encoders under a two-stage protocol: task-specific base tuning followed by shared downstream classification finetuning under matched optimization budgets. This design enables attribution-level comparison of supervision regimes while controlling for architecture, downstream task, and compute.

The main contribution of this study is a multi-criteria evaluation framework that links representation transfer to measurable explanation properties. Rather than relying only on classification accuracy, we combine faithfulness-based metrics, through positive and negative perturbation $\mathrm { A U C } .$ with localization-based metrics, including keypoint emphasis, background attribution, and spatial-distribution alignment. This provides a structured way to quantify how different supervision modalities organize visual evidence after transfer.

Across the main CUB-200 experiments, caption-based pretraining produces consistent advantages on most attribution-based metrics. Caption-tuned encoders show stronger correspondence to annotated keypoints, improved geometric alignment with keypoint density maps, reduced attribution to background regions, and greater sensitivity to removal of highly salient regions. Although these improvements do not translate into higher downstream classification accuracy under the fixed training budget, they reveal systematic differences in representation structure that predictive performance alone does not capture. This suggests that language supervision can shape how visual evidence is organized, even when the final task risk remains comparable. In contrast, classification tuning produces slightly greater stability under the removal of low-saliency regions.

Table 2: Ablation on Oxford-IIIT Pet comparing classification- and caption-tuned encoders. Each cell reports mean ± standard deviation over paired test samples. $\Delta$ denotes Captioning - Classification. Arrows indicate the preferred direction.
<table><tr><td>Saliency Method</td><td>Metric</td><td>Classification  $( \mathrm { B e s t \ a c c u r a c y } \colon$   $9 3 . 8 3 \% \pm 0 . 0 0 9 8 )$ </td><td>Captioning  $( \mathrm { B e s t \ a c c u r a c y } \colon$   $9 2 . 3 7 \% \pm 0 . 0 0 7 1 )$ </td><td>∆</td><td>p</td><td> $d$ </td></tr><tr><td rowspan="3">Grad-Weighted Attention</td><td> $A U C _ { I ^ { + } } \downarrow$ </td><td> $0 . 3 7 4 7 \pm 0 . 2 0 9 2$ </td><td> $\mathbf { 0 . 3 1 1 8 \pm 0 . 1 8 2 9 }$ </td><td>-0.0630</td><td> $2 . 2 8 \times 1 0 ^ { - 9 9 }$ </td><td>0.436</td></tr><tr><td> $A U C _ { I ^ { - } } \uparrow$ </td><td> $\mathbf { 0 . 7 7 4 4 \pm 0 . 1 5 7 0 }$ </td><td> $0 . 7 5 7 1 \pm 0 . 1 6 0 1$ </td><td>-0.0173</td><td> $1 . 8 3 \times 1 0 ^ { - 3 1 }$ </td><td>0.124</td></tr><tr><td> $F _ { b g } \downarrow$ </td><td> $0 . 5 2 7 8 \pm 0 . 1 6 0 0$ </td><td> $\mathbf { 0 . 4 7 9 3 \pm 0 . 1 5 1 5 }$ </td><td>-0.0485</td><td> $1 . 1 6 \times 1 0 ^ { - 9 3 }$ </td><td>0.559</td></tr><tr><td rowspan="3">Attention Rollout</td><td> $A U C _ { I ^ { + } } \downarrow$ </td><td> $0 . 3 2 4 9 \pm 0 . 2 1 8 4$ </td><td> $\mathbf { 0 . 2 7 8 5 \pm 0 . 1 8 3 6 }$ </td><td>-0.0464</td><td> $6 . 6 5 \times 1 0 ^ { - 7 1 }$ </td><td>0.383</td></tr><tr><td> $A U C _ { I }$  -↑</td><td> $\mathbf { 0 . 7 9 7 6 \pm 0 . 1 4 9 1 }$ </td><td> $0 . 7 7 2 1 \pm 0 . 1 5 2 0$ </td><td>-0.0255</td><td> $1 . 0 4 \times 1 0 ^ { - 8 1 }$ </td><td>0.208</td></tr><tr><td> $F _ { b g } \downarrow$ </td><td> $0 . 4 2 1 7 \pm 0 . 1 4 7 1$ </td><td> $\mathbf { 0 . 3 7 6 4 \pm 0 . 1 4 7 3 }$ </td><td>-0.0453</td><td> $< 1 0 ^ { - 3 0 8 }$ </td><td>0.986</td></tr></table>

Table 3: Encoder-scale ablation on CUB-200 using ${ \mathrm { V i T - L } } / 1 6 .$ Each cell reports mean ± standard deviation over $n = 2 8 8 2$ paired test samples. $\Delta$ denotes Captioning - Classification. Arrows indicate the preferred direction.
<table><tr><td>Saliency Method</td><td>Metric</td><td>Classification (Best accuracy: 87.05%)</td><td>Captioning (Best accuracy: 84.52%)</td><td>∆</td><td>p</td><td>d</td></tr><tr><td rowspan="5">Grad-Weighted Rollout</td><td> $A U C _ { I ^ { + } } \downarrow$ </td><td> $0 . 1 4 3 0 \pm 0 . 1 2 2 2$ </td><td> $\mathbf { 0 . 1 2 7 2 \pm 0 . 0 9 4 4 }$ </td><td>-0.0158</td><td> $2 . 8 9 \times 1 0 ^ { - 1 1 }$ </td><td>0.175</td></tr><tr><td> $A U C _ { I }$  -↑</td><td> $\mathbf { 0 . 7 6 8 8 \pm 0 . 1 9 1 9 }$ </td><td> $0 . 7 1 2 4 \pm 0 . 2 1 4 9$ </td><td>-0.0563</td><td> $1 . 6 4 \times 1 0 ^ { - 1 4 4 }$ </td><td>0.313</td></tr><tr><td> $F _ { b g } \downarrow$ </td><td> $0 . 6 5 2 8 \pm 0 . 0 9 2 5$ </td><td> $\mathbf { 0 . 5 9 4 2 \pm 0 . 1 0 8 0 }$ </td><td>-0.0585</td><td> $8 . 1 9 \times { 1 0 } ^ { - 2 1 2 }$ </td><td>0.674</td></tr><tr><td> $R _ { k p } \uparrow$ </td><td> $1 . 2 0 3 5 \pm 0 . 3 4 2 0$ </td><td> $\mathbf { 1 . 2 7 0 0 \pm 0 . 4 1 0 1 }$ </td><td>0.0665</td><td> $9 . 5 2 \times 1 0 ^ { - 2 1 }$ </td><td>-0.187</td></tr><tr><td> $D _ { E M D } \downarrow$ </td><td> $0 . 1 3 5 8 \pm 0 . 0 2 9 3$ </td><td> $\mathbf { 0 . 1 2 8 6 \pm 0 . 0 2 9 3 }$ </td><td>-0.0072</td><td> $1 . 9 5 \times { 1 0 } ^ { - 3 3 }$ </td><td>0.224</td></tr><tr><td rowspan="5">Attention Rollout</td><td> $A U C _ { I ^ { + } } \downarrow$ </td><td> $0 . 1 8 1 3 \pm 0 . 1 4 1 1$ </td><td> $\mathbf { 0 . 1 5 6 7 \pm 0 . 1 2 5 8 }$ </td><td>-0.0246</td><td> $1 . 4 0 \times { { 1 0 } ^ { - 4 5 } }$ </td><td>0.254</td></tr><tr><td> $A U C _ { I ^ { - } } \uparrow$ </td><td> $\mathbf { 0 . 7 5 1 2 \pm 0 . 1 9 9 7 }$ </td><td> $0 . 6 9 8 3 \pm 0 . 2 1 9 6$ </td><td>-0.0530</td><td> $2 . 4 8 \times 1 0 ^ { - 1 1 3 }$ </td><td>0.292</td></tr><tr><td> $F _ { b g } \downarrow$ </td><td> $0 . 6 8 6 7 \pm 0 . 1 2 7 2$ </td><td> $\mathbf { 0 . 6 8 0 3 \pm 0 . 1 2 7 1 }$ </td><td>-0.0064</td><td> $6 . 2 6 \times 1 0 ^ { - 6 7 }$ </td><td>0.296</td></tr><tr><td> $R _ { k p } \uparrow$ </td><td> $1 . 0 4 6 4 \pm 0 . 1 5 0 9$ </td><td> $\mathbf { 1 . 0 6 8 6 \pm 0 . 1 8 4 9 }$ </td><td>0.0222</td><td> $1 . 5 1 \times 1 0 ^ { - 5 1 }$ </td><td>-0.301</td></tr><tr><td> $D _ { E M D } \downarrow$ </td><td> $0 . 1 6 5 3 \pm 0 . 0 2 8 1$ </td><td> $\mathbf { 0 . 1 6 2 8 \pm 0 . 0 2 8 0 }$ </td><td>-0.0026</td><td> $3 . 2 3 \times { 1 0 } ^ { - 1 0 4 }$ </td><td>0.390</td></tr></table>

The ablation studies further support the utility of the framework. On Oxford-IIIT Pet, where only foreground segmentation is available, the same protocol identifies consistent supervisiondependent trends using background attribution and perturbation metrics. The ViT-L/16 encoderscale ablation preserves the directional patterns observed with ViT-B/16, indicating that the analysis is not tied to a single encoder capacity. Together, these experiments show that the proposed framework can be applied across datasets, annotation granularities, and model scales to quantify how training objectives affect learned representations.

Overall, the findings highlight the value of explainability-based evaluation for studying representation transfer beyond accuracy. Captionbased pretraining is not superior in downstream classification accuracy, but it induces more semantically structured attribution behavior across most evaluated criteria. This distinction is important for settings where the organization of evidence, localization of task-relevant regions, and reduced background reliance matter alongside predictive performance. Future work can extend this framework to larger-scale multimodal objectives, including contrastive vision-language learning, additional fine-grained and domain-specific datasets to further characterize how supervision modality shapes visual representations.

## Data availability

All datasets used in this study are publicly available. The CUB-200-2011 dataset is available from its official distribution. The natural language captions for CUB-200-2011 are those provided by Reed et al. [2016]. The Oxford-IIIT Pet dataset is publicly available, and the enriched captions used in this work are obtained from the Visual Layer Oxford-IIIT Pet VL Enriched dataset available at https://huggingface.co/datasets/visual-layer/ oxford-iiit-pet-vl-enriched/. No new datasets were generated during this study.

## Author contribution

M.R.K.: Conceptualization, Methodology, Investigation, Software, Formal analysis, Visualization, Writing—original draft. S.L.: Conceptualization, Methodology, Supervision, Writing—review and editing. All authors read and approved the final manuscript.

## Declarations

The authors declare no conflict of interest.

## Funding

The authors received no financial support for the research, and/or authorship of this article.

## Code Availability

The source code, trained models, and scripts required to reproduce the experiments will be made publicly available upon publication.

## Ethics Approval

Not applicable.

## References

Samira Abnar and Willem Zuidema. Quantifying attention flow in transformers. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 4190–4197, 2020.

Hila Chefer, Shir Gur, and Lior Wolf. Generic attention-model explainability for interpreting

bi-modal and encoder-decoder transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 397–406, 2021.

Marcella Cornia, Lorenzo Baraldi, and Rita Cucchiara. Explaining transformer-based image captioning models: An empirical analysis. AI Communications, 35(2):111–129, 2022.

Karan Desai and Justin Johnson. Virtex: Learning visual representations from textual annotations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11162–11173, 2021.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Mohamed El Banani, Karan Desai, and Justin Johnson. Learning visual representations via language-guided sampling. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 19208–19220, 2023.

Sofiane Elguendouze, Adel Hafiane, Marcilio CP de Souto, and Anais Halftermeyer. Explainability in image captioning based on the latent space. Neurocomputing, 546:126319, 2023.

Xiangteng He and Yuxin Peng. Fine-grained image classification via combining vision and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5994–6002, 2017.

Lisa Anne Hendricks, Zeynep Akata, Marcus Rohrbach, Jeff Donahue, Bernt Schiele, and Trevor Darrell. Generating visual explanations. In European conference on computer vision, pages 3–19. Springer, 2016.

Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, and Lijuan Wang. Scaling up vision-language pre-training for image captioning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 17980–17989, 2022.

Chengxiang Huang, Yake Wei, Zequn Yang, and Di Hu. Adaptive unimodal regulation for balanced multimodal information acquisition. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 25854- 25863, 2025.

Rémi Kazmierczak, Eloïse Berthier, Goran Frehse, and Gianni Franchi. Explainability and vision foundation models: A survey. Information Fusion, 122:103184, 2025.

Md Raisul Kibria, Sébastien Lafond, and Janan Arslan. Decoding the multimodal maze: A systematic review on the adoption of explainability in multimodal attention-based models. arXiv preprint arXiv:2508.04427, 2025.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022a.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023.

Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded languageimage pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10965–10975, 2022b.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. volume 30. 2017.

Hiba Najjar, Bushra Alshbib, and Andreas Dengel. Can multitask learning enhance model explainability? In DAGM German Conference on Pattern Recognition, pages 600–614. Springer, 2025.

Meike Nauta, Jan Trienes, Shreyasi Pathak, Elisa Nguyen, Michelle Peters, Yasmin Schmitt, Jörg Schlötterer, Maurice Van Keulen, and Christin Seifert. From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable ai. ACM Computing Surveys, 55(13s):1–42, 2023.

Maria Parelli, Dimitrios Mallis, Markos Diomataris, and Vassilis Pitsikalis. Interpretable visual question answering via reasoning

supervision. In 2023 IEEE International Conference on Image Processing (ICIP), pages 2525–2529. IEEE, 2023.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Scott Reed, Zeynep Akata, Honglak Lee, and Bernt Schiele. Learning deep representations of fine-grained visual descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 49–58, 2016.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " why should i trust you?" explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 1135–1144, 2016.

Mert Bulent Sariyildiz, Julien Perez, and Diane Larlus. Learning visual representations with caption annotations. In European conference on computer vision, pages 153–170. Springer, 2020.

Ramprasaath R Selvaraju, Michael Cogswell Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradientbased localization. In Proceedings of the IEEE international conference on computer vision, pages 618–626, 2017.

Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In International conference on machine learning, pages 3319–3328. PMLR, 2017.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100, 2022.

Yixuan Wei, Yue Cao, Zheng Zhang, Houwen Peng, Zhuliang Yao, Zhenda Xie, Han Hu, and Baining Guo. iclip: Bridging image classification and contrastive language-image pretraining for visual recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2776–2786, 2023.

Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are imagetext foundation models. arXiv preprint arXiv:2205.01917, 2022.

Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6720–6731, 2019.

Xiaohui Zhang, Jaehong Yoon, Mohit Bansal, and Huaxiu Yao. Multimodal representation learning by alternating unimodal adaptation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 27456–27466, 2024.

## Appendix A Dataset Description

An example of the dataset annotations is shown in Figure A1. The dataset provides 15 annotated object parts (keypoints): back, beak, belly, breast, crown, forehead, left eye, left leg, left wing, nape, right eye, right leg, right wing, tail, and throat.

## Appendix B Explanation Algorithm Formulations

Here, we present the formulation of the adaptation of the explanation algorithms applied in this work.

## Attention Rollout.

Let $\bar { A } ^ { ( l ) } \in \mathbb { R } ^ { N \times N }$ denote the attention matrix of layer l, obtained by aggregating attention across heads. We incorporate residual connections and normalize row-wise as in Equation B1.

$$
\tilde { A } ^ { ( l ) } = \mathrm { N o r m } \Big ( \bar { A } ^ { ( l ) } + I \Big ) ,\tag{B1}
$$

where Norm(·) denotes row normalization.

Attention is propagated through the network as in Equation B2.

$$
R = \prod _ { l = 1 } ^ { L } \tilde { A } ^ { ( l ) } , \qquad { \bf r } = R _ { 0 , : } .\tag{B2}
$$

## Gradient-Weighted Attention Rollout.

Let $A ^ { ( l , h ) }$ denote the attention matrix of head h in layer l, and let $G ^ { ( l , h ) }$ be the gradient of the target output y with respect to that attention matrix. We define a scalar importance weight per head as $\alpha _ { h } ^ { ( l ) }$ and construct the gradient-weighted layer attention as in Equation B3.

![](images/5b88f0cd62da2d318f768615b58b86b6f122a0155b1335a21f3a9867dbd31d56.jpg)  
(a) Original input image (left) and the same image with the object mask overlay (right), along with one example caption (top).

![](images/709530a25ed096da38ef5407cdb3968283799eaf65e7587f873ea374749ffe77.jpg)  
(b) Annotated part names with their corresponding keypoint locations for the example image.  
Fig. A1: Illustration of CUB-200-2011 dataset annotations, including object masks and part locations.

$$
\hat { A } ^ { ( l ) } = \sum _ { h } \alpha _ { h } ^ { ( l ) } A ^ { ( l , h ) } .\tag{B3}
$$

where,

$$
\begin{array} { c } { { G ^ { ( l , h ) } = \displaystyle \frac { \partial y } { \partial A ^ { ( l , h ) } } } } \\ { { \alpha _ { h } ^ { ( l ) } = \mathrm { M e a n } \Big ( G ^ { ( l , h ) } \Big ) } } \end{array}
$$

The normalized layer attention is then calculated using Equation B4.

$$
\tilde { A } ^ { ( l ) } = \mathrm { N o r m } \Big ( \hat { A } ^ { ( l ) } + I \Big ) ,\tag{B4}
$$

and rollout is computed identically as in Equation B2.

## Appendix C Positive and Negative Perturbation Demonstration

An example of how an attribution map derived for a sample test input from the attention rollout method is used to measure positive and negative perturbation progressively is presented in Figure C2.

## Appendix D Keypoint Emphasis Ratio Demonstration

Figure D3 illustrates how the Keypoint Emphasis Ratio is measured for two representative test inputs across the two training regimes and both attribution methods.

## Appendix E Earth Mover's Distance Demonstration

Example of how keypoints are used to create a Gaussianized density maps is presented in Figure E4.

![](images/8320d7f4369712132403354076585bfa1c2f05ccc65cee64262432480b7ac32f.jpg)  
(a) Saliency map (attention rollout).

![](images/247d8784cf89273f6d7854ef35b5ba0b14468f9e3424e233c0190ed8bc94ab8b.jpg)  
(b) Positive perturbation (mostsalient regions removed first).

![](images/2544b8808383880706a994eddfd88103aecfd5f7855c305257ab8a9afe1fb1e1.jpg)  
(c) Negative perturbation (leastsalient regions removed first).  
Fig. C2: Saliency-based perturbation protocol. Positive perturbation removes the most-salient regions first, producing a rapid drop in model confidence if the explanation is faithful. Negative perturbation removes the least-salient regions first, preserving task-relevant evidence and resulting in comparatively stable confidence.

![](images/0c4e4115ca11ce1bdb997e24582ef842109517c22e344d87e0faceb471182330.jpg)  
Fig. D3: Illustration of the Keypoint Emphasis Ratio $( R _ { k p } )$ , which quantifies the relative concentration of saliency around annotated object keypoints. Each row shows one example. Column 1 presents the original image with its ground-truth label. Columns 2–5 display saliency overlays for classification- and captiontuned encoders under attention rollout and gradient-weighted rollout. Green circles denote annotated keypoints with a scale-adaptive radius (spatial extent 0.25) used to compute $R _ { k p } .$ The white contour defines the object boundary. Red indicates high attribution intensity and blue indicates low attribution intensity.

## Appendix F Two Stage Training Summary

Table F1 summarizes the training behavior of the two supervision regimes by reporting test-set classification accuracy on CUB-200 across different training iterations. This comparison provides an overview of how predictive performance evolves during training for the classification-based and caption-based settings under the same evaluation protocol.

## Appendix G Ablating Radius for Keypoint Emphasis Calculation

We report the Keypoint Emphasis Ratio, $R _ { k p } ,$ across spatial extension ratios $\kappa \_ { \mathbf { \Omega } } \in$ {0.15,0.25, 0.35, 0.50}. Results for a single random seed, comparing both training regimes and saliency methods, are shown in Table G2.

Across all spatial extensions, the captioning regime consistently achieves higher $R _ { k p }$ values, indicating stronger alignment with semantically meaningful keypoints. This trend holds for both small and large neighborhoods around annotations and across saliency methods, reinforcing the main localization findings. Gradient-weighted rollout yields overall higher $R _ { k p }$ scores compared to attention rollout, though the relative advantage of caption supervision remains consistent within each method.

![](images/0ecbc085ef808174fd1df8390f784067f903e6935c3b566184358e1182ae9ef3.jpg)  
Fig. E4: Example Gaussianized keypoint density map generated from annotated object parts. Individual keypoints are represented by Gaussian kernels and masked using the object segmentation.

Table F1: Two-stage training results. Stage 2 follows Stage 1 (5 Iterations).
<table><tr><td>Stage</td><td>Training Regimen</td><td>Iteration</td><td>Test Accuracy  $\mathbf { \overline { { ( m e a n \pm 9 5 \% } } }$  CI)</td></tr><tr><td rowspan="6">Stage 1</td><td rowspan="6">V1: Classification-base training (used for Stage 2A)</td><td>0</td><td> $\overline { { 0 . 0 0 4 9 \pm 0 . 0 0 8 2 } }$ </td></tr><tr><td>1</td><td> $0 . 7 5 4 7 \pm 0 . 0 2 8 6$ </td></tr><tr><td>2</td><td> $0 . 8 3 3 9 \pm 0 . 0 0 6 5$ </td></tr><tr><td>3</td><td> $0 . 8 4 6 3 \pm 0 . 0 2 3 2$ </td></tr><tr><td>4</td><td> $0 . 8 4 6 2 \pm 0 . 0 1 9 4$ </td></tr><tr><td>5</td><td> $\mathbf { 0 . 8 4 5 4 \ : \pm { \ : 0 . 0 1 4 8 } }$ </td></tr><tr><td>V2: Captioning-base training (used for Stage 2B)</td><td>Post training</td><td> $0 . 0 3 6 7 \pm 0 . 0 2 5 4$ </td></tr><tr><td rowspan="6">Stage 2</td><td rowspan="6">A: Final tuning of classification-tuned encoder (from Stage 1 V1)</td><td>1</td><td>0.8537 ± 0.0212</td></tr><tr><td>2</td><td> $0 . 8 5 6 1 \pm 0 . 0 1 1 6$ </td></tr><tr><td>3</td><td> $0 . 8 5 7 0 \pm 0 . 0 0 8 2$ </td></tr><tr><td>4</td><td> $0 . 8 5 4 2 \pm 0 . 0 0 7 8$ </td></tr><tr><td>5</td><td> $\mathbf { 0 . 8 5 6 6 \pm 0 . 0 1 1 4 }$ </td></tr><tr><td>1</td><td> $0 . 8 1 3 1 \pm 0 . 0 1 5 9$ </td></tr><tr><td rowspan="4">B: Final tuning of caption-tuned encoder (from Stage 1 V2)</td><td>2</td><td> $0 . 8 3 8 4 \pm 0 . 0 0 4 7$ </td></tr><tr><td>3</td><td> $0 . 8 4 1 4 \pm 0 . 0 0 7 5$ </td></tr><tr><td>4</td><td> $0 . 8 4 3 5 \pm 0 . 0 0 7 7$ </td></tr><tr><td>5</td><td> $\mathbf { 0 . 8 4 7 3 \ : \pm { \ : 0 . 0 0 6 2 } }$ </td></tr></table>

Table G2: Radius Ablation across Saliency Algorithms and Training Regimes. Mean Keypoint Emphasis Ratio, $R _ { k p }$ , reported for four scale factors.
<table><tr><td>Saliency Algorithm</td><td>Training Regime</td><td>0.15</td><td>0.25  $\mathbf { ( d e f a u l t ) }$ </td><td>0.35</td><td>0.50</td></tr><tr><td rowspan="2">Attention Rollout</td><td>Captioning</td><td>1.083620</td><td>1.114433</td><td>1.153625</td><td>1.225190</td></tr><tr><td>Classification</td><td>1.070167</td><td>1.098750</td><td>1.134251</td><td>1.199433</td></tr><tr><td rowspan="2">Grad-Weighted Rollout</td><td>Captioning</td><td>1.173308</td><td>1.235900</td><td>1.315662</td><td>1.463961</td></tr><tr><td>Classification</td><td>1.109976</td><td>1.154779</td><td>1.215932</td><td>1.337401</td></tr></table>