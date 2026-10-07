# Comprehensive Evaluation and Fine-Tuning of Foundational Cell Nuclei Segmentation Models in Renal Pathology

Ruijie Wu<sup>a†</sup>, Junlin Guo<sup>a†</sup>, Ruining Deng<sup>b</sup>, Yu Wang<sup>c</sup>, Shilin Zhao<sup>c</sup>, Haichun Yang<sup>d</sup>, and Yuankai Huo<sup>a,d,e</sup>

<sup>a</sup>Department of Electrical and Computer Engineering, Vanderbilt University, Nashville, TN 37235, USA

<sup>b</sup>Department of Radiology, Weill Cornell Medicine, New York, NY 10021, USA <sup>c</sup>Department of Biostatistics, Vanderbilt University Medical Center, Nashville, TN 37235, USA <sup>d</sup>Department of Pathology, Microbiology and Immunology, Vanderbilt University Medical Center, Nashville, TN 37235, USA

<sup>e</sup>Department of Computer Science, Vanderbilt University, Nashville, TN 37235, USA <sup>†</sup>These authors contribute equally to this work.

## ABSTRACT

Accurate nuclei instance segmentation is essential for quantitative renal pathology, yet general-purpose AI foundation models often struggle with renal tissue due to low contrast, densely packed nuclei, complex morphology, and strong background staining. Consequently, domain-specific fine-tuning is often necessary to bridge the gap between general-purpose segmentation capability and reliable performance in focused pathology applications. In our prior SPIE Medical Imaging studies, we demonstrated a substantial performance gap when applying multiple general-purpose cell segmentation foundation models to renal pathology. In our other study, we further showed that fine-tuning with well-segmented “Easy” cases and model-failure “Hard” cases could substantially improve segmentation performance. However, an important piece remained unresolved: whether annotating the much larger and more time-consuming set of intermediate-dificulty cases provides additional value beyond Easy and Hard examples. In this study, we complete this evaluation by introducing a “Medium”-dificulty cohort and systematically assessing its contribution to model adaptation. Specifically, we extended our human-in-the-loop framework by combining 5,901 foundation-model-generated pseudo-labels from well-segmented cases (Easy), 860 newly expert-annotated unresolved challenging cases (Medium), and 198 expert-annotated cases that were jointly identified as failures by the foundation models in our prior study (Hard). We additionally incorporated recently developed general-purpose cell segmentation models to evaluate the performance of current state-of-the-art ap proaches. Together, these annotations spanning multiple levels of segmentation dificulty enabled a systematic evaluation of seven single-source and mixed-source fine-tuning strategies across nine cell segmentation mode configurations. The segmentation results demonstrate that annotations across diferent levels of segmentation dificulty can efectively support domain adaptation, while also revealing that the optimal composition of finetuning data remains model dependent. The code and resources used in this study are publicly available at https://github.com/hrlblab/AFM\_kidney\_cells/tree/main/stage4\_paper.

Keywords: cell nuclei instance segmentation, renal pathology, foundation models, human-in-the-loop, finetuning, domain adaptation

## 1. INTRODUCTION

Renal pathology presents a particularly demanding setting for cell nuclei instance segmentation. Although general-purpose cell segmentation models have improved considerably, their performance remains inconsistent in kidney-specific regions that difer from the images used for pretraining, and architectural advances alone do not guarantee that dificult kidney nuclei will be adequately segmented. Samples with low contrast, unusual morphology, dense nuclear distributions, or strong background staining may remain unresolved even after applying newer models.<sup>1–3</sup> These limitations motivate domain-specific adaptation of general-purpose segmentation models using kidney-specific supervision.

Our previous studies<sup>1–3</sup> progressively identified this domain gap and explored two complementary directions for addressing it: human-in-the-loop data enrichment and advances in general-purpose cell segmentation models. In our prior study (2024),<sup>1</sup> Cellpose,<sup>4</sup> StarDist,<sup>5</sup> and CellViT<sup>6</sup> were evaluated on approximately 8,800 kidney pathology image patches; even the best-performing model, CellViT, adequately segmented 63.6% of images, leaving approximately 37% insuficiently resolved. In 2025, a human-in-the-loop data-enrichment strategy using scalable Easy pseudo-labels and 198 expert-annotated Hard cases improved all three models on kidney pathology instance segmentation.<sup>2</sup> More recently, newly released models converted 1,301 of 2,091 previously challenging patches to Good-quality predictions, while 782 remained Medium and eight remained Bad; across the full evaluation set, 1,079 Medium-quality patches remained and formed the candidate pool for annotation.<sup>3</sup> These remaining Medium cases represented a substantially larger unresolved cohort than the 198 previously annotated Hard cases. Unlike scalable Easy pseudo-labels and the relatively small set of consensus Hard failures, incorporating these Medium cases would require substantial additional expert annotation, while the benefit of this annotation efort remained unclear. Therefore, a key question arises: Does expert annotation of the remaining Medium-rated cases provide additional value beyond existing Easy pseudo-labels and Hard expert annotations, and is this benefit consistent across diferent segmentation models?

Motivated by this remaining gap, we extend our previous human-in-the-loop framework by introducing expertannotated Medium cases, enabling a systematic evaluation across Easy, Medium, and Hard annotation sources. We use 5,901 foundation-model-generated pseudo-labels from well-segmented cases as Easy data and 198 prior expert-annotated consensus failure cases as Hard data.<sup>2</sup> Pathology experts selected and annotated 860 challenging patches from a pool of 1,079 previously unresolved Medium-quality image patches (Fig. 1A). Together, these annotation sources were used to evaluate whether incorporating Medium annotations provides additional value beyond the previously established Easy + Hard fine-tuning strategy for kidney nuclei instance segmentation.<sup>2,</sup> <sup>3</sup> We fine-tuned and evaluated nine cell segmentation model configurations released from 2022 to 2026, including Cellpose 3.0, Cellpose-SAM, Cellpose-DINO, StarDist, CellViT (HIPT and SAM), CellViT++ (Virchow), LSP-DETR, and PathoSAM<sup>5–13</sup> (Fig. 1B). Each model was fine-tuned using seven single-source and mixed-source strategies across Easy, Medium, and Hard annotation sources, resulting in 63 fine-tuning experiments (Fig. 1C). The overall framework for this study is shown in Fig. 1.

The main contributions of our study are as follows:

• Expert annotation of previously unresolved Medium patches: We extend human-in-the-loop enrichment beyond Easy pseudo-labels and Hard failure cases by selecting and annotating Medium-quality patches that remained insuficiently segmented after successive model evaluations.

• Import recent cell nuclei segmentation models released in 2025–2026: We systematically evaluate nine cell segmentation model configurations under the same annotation enrichment framework and show that the optimal composition of Easy, Medium, and Hard annotations varies across models.

• Fine-tuning with diferent combinations of Easy, Medium and Hard datasets: We systematically evaluate Easy pseudo-labels, Medium annotations, and Hard expert annotations, both separately and in diferent combinations, to determine whether Medium annotations provide additional benefits when combined with large-scale pseudo-labels, challenging expert-labeled data, or both.

## 2. METHODS

## 2.1 Comprehensive Cell Nuclei Segmentation Model Benchmark

In this study, we chose nine cell segmentation model configurations to fine-tune, spanning models released from 2022 to 2026, as illustrated in Fig. 1B and summarized in Table 1. The models can be categorized into three groups. The first group consists of Cellpose 3.0, StarDist, and CellViT, which were fine-tuned in our previous study and were re-fine-tuned here using the new dataset.<sup>2</sup> The second group, CellViT++ [Virchow] and Cellpose-SAM, had previously been evaluated only for inference.<sup>3</sup> The third group, PathoSAM, LSP-DETR, and Cellpose-DINO, comprises models that had not been evaluated or fine-tuned in our prior kidney studies.<sup>9,</sup> <sup>12,</sup> <sup>13</sup>

![](images/b9f659166793a067ffbaaaf60d6fde390b74dc5f612c0444693141f4febb1bc6.jpg)  
Figure 1: Overall framework. (A) Expert annotation of previously unresolved Medium-quality kidney pathology patches to enrich the existing Easy and Hard fine-tuning data. (B) Expanded evaluation of cell nuclei segmentation models released from 2022 to 2026, including both previously evaluated and newly incorporated models. (C) Fine-tuning with seven single-source and mixed-source combinations of Easy, Medium, and Hard datasets to evaluate model-specific adaptation performance.

1. Cellpose-based models: Cellpose 3.0, Cellpose-SAM, and Cellpose-DINO follow the Cellpose flow-based instance segmentation framework. They predict horizontal and vertical pixel flows together with a cellprobability map and reconstruct individual objects by grouping pixels that converge to the same location. Cellpose-SAM and Cellpose-DINO replace the convolutional encoder with pretrained SAM and DINOv3 ViT encoders, respectively, while retaining the Cellpose prediction and instance-reconstruction framework.<sup>4,</sup> <sup>7–10</sup> For Cellpose 3.0, we followed our prior preprocessing pipeline by converting RGB histopathology patches into grayscale, DAPI-like nuclear inputs for both fine-tuning and inference.<sup>2</sup>

2. StarDist: StarDist represents each nucleus as a star-convex polygon defined by radial distances from an interior pixel to the object boundary. The network densely predicts object probabilities and radial distances along predefined directions. The resulting polygon candidates are filtered using non-maximum suppression to obtain the final nucleus instances.<sup>5</sup>

3. CellViT-based models: CellViT combines a pretrained ViT encoder with a U-Net-like decoder for nucleus instance segmentation. It predicts a binary nuclei map, horizontal and vertical distance maps, and a nuclei-type map, followed by HoVer-Net-style post-processing to separate touching nuclei.<sup>6,</sup> <sup>14</sup> The HIPT and SAM variants primarily difer in their pretrained encoders.<sup>15,</sup> <sup>16</sup> CellViT++ retains the CellViT segmentation framework while incorporating foundation-model encoders, including the pathology-specific Virchow encoder, and nucleus-level embeddings for downstream cell classification.<sup>11,</sup> <sup>17</sup>

4. LSP-DETR: LSP-DETR formulates nucleus instance segmentation as a direct set-prediction problem. Each nucleus is represented by a center point and radial distances defining a star-convex boundary. A DETR-style decoder predicts nucleus locations and shapes, allowing instance masks to be generated directly without conventional watershed-based post-processing.<sup>12,</sup> <sup>18</sup>

5. PathoSAM: PathoSAM adapts SAM to histopathology images for both interactive and automatic instance segmentation. Its automatic instance-segmentation decoder predicts foreground probabilities, distances to the closest object centers, and distances to the closest object boundaries. These prediction maps are then processed using seeded watershed to separate adjacent nuclei and generate individual instances.<sup>13,</sup> <sup>16</sup>

Table 1: Cell Nuclei Segmentation Models in Pathology (2022-2026)
<table><tr><td>Models</td><td>Release Date</td><td>Backbone</td><td>Code</td></tr><tr><td>StarDist  $\mathrm { ( H i s t o . ) ^ { 5 } }$ </td><td>2022 Mar</td><td> $\mathrm { U - N e t ^ { 1 9 } }$ </td><td>URL</td></tr><tr><td>CellViT  $[ \mathrm { H I P T - 2 5 6 } ] ^ { 6 }$ </td><td>2023 Oct</td><td> $\mathrm { H I P T ~ V i T ^ { 1 5 } }$ </td><td>URL</td></tr><tr><td>CellViT  $[ \mathrm { S A M - H } ] ^ { 6 }$ </td><td>2023 Oct</td><td> $\mathrm { S A M \ V i T ^ { 1 6 } }$ </td><td>URL</td></tr><tr><td> $\mathrm { C e l l p o s e 3 . 0 ^ { 7 } }$ </td><td>2024 Feb</td><td> $\mathrm { U - N e t ^ { 1 9 } }$ </td><td>URL</td></tr><tr><td> $\mathrm { C e l l { \bar { V } i T + + } \Gamma [ \bar { V } i r c h o w ] ^ { 1 1 } }$ </td><td>2025 Jan</td><td> $\mathrm { V i r c h o w ~ V i T ^ { 1 7 } }$ </td><td>URL</td></tr><tr><td> $\mathrm { P a t h o S A M ^ { 1 3 } }$ </td><td>2025 Feb</td><td> $\mathrm { S A M \ V i T ^ { 1 6 } }$ </td><td>URL</td></tr><tr><td> $\mathrm { C e l l p o s e { - } S A M ^ { 8 } }$ </td><td>2025 Apr</td><td> $\mathrm { S A M \ V i T ^ { 1 6 } }$ </td><td>URL</td></tr><tr><td> $\mathrm { L S P  – D E T R ^ { 1 2 } }$ </td><td>2026 Jan</td><td> $\mathrm { S w i n V 2 – T ^ { 2 0 } }$ </td><td>URL</td></tr><tr><td> $\mathrm { C e l l p o s e { - } D I N O ^ { 9 } }$ </td><td>2026 Jun</td><td> $\mathrm { { D I N O v 3 } \ V i T ^ { 1 0 } }$ </td><td>URL</td></tr></table>

## 2.2 Human-in-the-Loop Annotation and Multi-source Data Enrichment

The previous evaluations categorized foundation-model predictions as Good, Medium, or Bad according to the proportion of identifiable nuclei captured in each patch. Predictions capturing at least 90% of nuclei were rated Good, those capturing approximately 50%–90% were rated Medium, and those capturing fewer than 50% were rated Bad.<sup>1–3</sup> Representative examples of the three prediction-rating categories are shown in Fig. 2.

![](images/12de8c0032af1dee0fef39f002451f6d6d778408ec0585f54a45d1b49dd11cab.jpg)  
Figure 2: Representative examples of prediction-rating categories. Foundation-model predictions were categorized as Good, Medium, or Bad according to the proportion of identifiable nuclei captured in each image patch. Green contours denote nucleus instances; the top row shows model predictions, whereas the bottom row shows the corresponding ground-truth annotations.

As shown in Fig. 3A, pathology experts reviewed the 1,079 unresolved Medium-rated patches and selected 860 of the most challenging cases for manual instance annotation. These samples represented regions that remained insuficiently segmented after the evaluation of newly released models. The human-in-the-loop annotation process was performed from the original image patches rather than by correcting foundation-model predictions.

As shown in Fig. 3B, the fine-tuning data were organized into three annotation sources according to segmentation dificulty. Easy samples consisted of 5,901 Good-rated image patches with foundation-model-generated pseudo-labels from our prior study. Hard samples consisted of 198 consensus failure cases that were rated Bad across the previously evaluated foundation models and subsequently annotated by pathology experts. Medium samples consisted of the 860 newly selected unresolved challenging patches described above, which were manually annotated from the original images. Together, these three sources provided scalable pseudo-labels, newly annotated unresolved cases, and expert-annotated consensus failures for subsequent fine-tuning experiments.

![](images/0ac48dd33dd6462fc19748dfaf5ba6f58e8cee3d18c7f663ad06bb326f36f285.jpg)  
Figure 3: Human-in-the-loop annotation and construction of multi-source fine-tuning data. (A) Pathology experts screened unresolved Medium-quality kidney patches, selected challenging samples, and manually annotated nucleus instances from the original images. (B) The final fine-tuning data consisted of three annotation sources: prior foundation-model-generated pseudo-labels from Easy cases, prior expert-annotated consensus failure cases (Hard), and newly human-in-the-loop annotated unresolved cases (Medium).

## 2.3 Weighted Oversampling for Annotation Source Rebalancing

For the four mixed-source datasets—Easy + Medium, Easy + Hard, Medium + Hard, and Easy + Medium + Hard—source-wise weighted oversampling was applied to reduce imbalance among annotation sources, following the rebalancing strategy used in our prior work:<sup>2</sup>

$$
w _ { c } = \frac { N } { \gamma n _ { c } + ( 1 - \gamma ) N } ,
$$

where N denotes the total number of training samples, n<sub>c</sub> is the number of samples from annotation source c, and γ controls the strength of source rebalancing. This weighting scheme increased the sampling probability of underrepresented annotation sources and reduced the dominance of the majority source during fine-tuning, thereby increasing model exposure to challenging Medium and Hard patches.

## 3. DATA AND EXPERIMENTS

Implementations: The model fine-tuning and inference pipeline was developed using Python 3.12.13 and PyTorch 2.12.1, with CUDA 12.8 enabling GPU acceleration. All experiments were performed on an NVIDIA RTX 5090 GPU with 32 GB of memory. The AdamW optimizer was used for all fine-tuning experiments.

## 3.1 Dataset

## 3.1.1 Composition of Fine-tuning Data

Our previous performance evaluation identified challenging samples that remained unresolved by the cell foundation models.<sup>3</sup> Challenging patches were selected in the human-in-the-loop annotation process for fine-tuning cell segmentation models. All identifiable nuclei were annotated as a single foreground class, without nuclear subtype classification. Nuclei intersecting the image boundary were retained, whereas ambiguous objects whose nuclear identity or boundary could not be reliably determined were resolved during expert review. The fina annotations were stored as instance-label masks with a unique integer identifier for each nucleus.

As summarized in Table 2, three annotation sources were used to construct the fine-tuning datasets: 5,901 Easy pseudo-labels corresponding to the 50% Easy subset used in our prior study, 860 newly annotated unresolved challenging Medium-quality patches, and 198 prior expert-annotated Hard cases identified as foundation-model failures.<sup>2,</sup> <sup>3</sup>

Table 2: Summary of the annotation sources used for model fine-tuning.
<table><tr><td>Dataset</td><td>Count</td><td>Annotation Source</td></tr><tr><td>Easy</td><td>5,901</td><td>Foundation-model-generated pseudo-labels</td></tr><tr><td>Medium</td><td>860</td><td>Unresolved Medium-quality patches annotated by experts</td></tr><tr><td>Hard</td><td>198</td><td>Prior foundation-model failure patches annotated by experts</td></tr></table>

## 3.1.2 Test Dataset

In this study, all experiments were evaluated using the same independent set of 185 expert-annotated kidney pathology patches established in our prior study.<sup>2</sup> The test set was excluded from pseudo-label generation, Medium and Hard data enrichment, validation, and fine-tuning.

## 3.2 Fine-Tuning Strategies

As illustrated in Fig. 1C and summarized in Table 3, seven fine-tuning strategies were used: Easy, Medium, Hard, Easy + Medium, Easy + Hard, Medium + Hard, and Easy + Medium + Hard.

The nine cell segmentation model configurations were fine-tuned under all seven strategies, resulting in 63 fine-tuning experiments. As shown in Table 3, γ was set to 0.85 for Easy + Medium, Easy + Hard, and Easy + Medium + Hard, and to 0.55 for Medium + Hard. These settings reduced dataset imbalance and increased the sampling frequency of the underrepresented Medium and Hard samples. The strategy-specific γ values were determined before model fine-tuning based on the degree of dataset imbalance and were applied consistently across all model configurations.

Table 3: Fine-tuning data composition and expected sampling ratios for the seven fine-tuning strategies
<table><tr><td>Fine-tuning Strategies</td><td>Samples</td><td>Original Ratio (%)</td><td>Expected Ratio (%)</td></tr><tr><td>Easy</td><td>5,901</td><td>100.0</td><td>100.0</td></tr><tr><td>Medium</td><td>860</td><td>100.0</td><td>100.0</td></tr><tr><td>Hard</td><td>198</td><td>100.0</td><td>100.0</td></tr><tr><td>Easy + Medium</td><td>6,761</td><td>87.3 / 12.7</td><td>66.5 / 33.5 (γ = 0.85)</td></tr><tr><td>Easy + Hard</td><td>6,099</td><td>96.8 / 3.2</td><td>84.5 / 15.5 (γ = 0.85)</td></tr><tr><td>Medium + Hard</td><td>1,058</td><td>81.3 / 18.7</td><td>72.8 / 27.2 (γ = 0.55)</td></tr><tr><td>Easy + Medium + Hard</td><td>6,959</td><td>84.8 / 12.4 / 2.8</td><td>60.0 / 29.9 / 10.1 (γ = 0.85)</td></tr></table>

## 3.3 Evaluation Metrics

The performance of all model configurations and their fine-tuned variants was evaluated using the F1 score.<sup>21</sup> Predicted and reference instances were matched using an IoU threshold of 0.5. The F1 score was computed for each test patch and then averaged across all test patches. It was calculated as:

$$
\mathrm { F 1 } = \frac { \mathrm { 2 T P } } { \mathrm { 2 T P } + \mathrm { F P } + \mathrm { F N } } ,
$$

where TP, FP, and FN denote true-positive, false-positive, and false-negative nuclei instances, respectively. The F1 score was used as the primary evaluation metric because it jointly accounts for false-positive and falsenegative predictions.

## 4. RESULTS

4.1 Efect of Annotation Strategies Across Model Configurations

![](images/ef442ba056ad36cc073696b047332072294d2813bee6ccd4e1b9b816c1fcdf3f.jpg)

![](images/d79173912c18fa9735aaf3069b41cacf8577db22af98b321f6d8b95b168b5b82.jpg)  
(c) Other Segmentation Models

![](images/80794beb063b2d602734b76eca566c6de52997e80f1fd63a6d257df1fc675521.jpg)  
Figure 4: Comparison of F1 scores across seven fine-tuning strategies. The nine nuclei instancesegmentation model configurations are grouped into (a) the Cellpose family, (b) the CellViT family, and (c) other segmentation models. For each model, performance is shown for the baseline and seven fine-tuning strategies: Easy (E), Medium (M), Hard (H), Easy + Hard (E+H), Easy + Medium (E+M), Medium + Hard (M+H) and Easy + Medium + Hard (E+M+H).

In Figure 4, we compare the F1 scores obtained using the seven annotation strategies across the Cellpose family, CellViT family, and other segmentation models. We found that no single fine-tuning strategy was optimal for all nine model configurations.

Medium-only fine-tuning produced the best F1 score for four models: StarDist, Cellpose-SAM, CellViT [HIPT-256], and CellViT [SAM-H]. We obtained the best results for CellViT++ [Virchow] and PathoSAM by combining Medium and Hard annotations. Cellpose 3.0 achieved its highest F1 score using the complete Easy plus Medium plus Hard dataset. In contrast, Hard-only fine-tuning was optimal for Cellpose-DINO and LSP-DETR.

Overall, seven of the nine best-performing strategies included the newly annotated Medium data. Although Easy + Hard fine-tuning improved performance over the corresponding baseline for eight of the nine mode configurations, it was not the optimal strategy for any model once Medium-containing strategies were included in the comparison. These results indicate that the newly annotated Medium cases provided complementary kidney-specific supervision beyond the previously available Easy and Hard data, although the optimal data composition remained model dependent.

## 4.2 Overall Fine-Tuning Performance

We evaluated all baselines and fine-tuned models on the common 185-patch hold-out test set. Table 4 summarizes the best fine-tuning result for each model configuration. We found that fine-tuning improved the best achievable F1 score of all nine model configurations relative to their corresponding baselines. Baseline F1 scores ranged from 0.6748 for Cellpose 3.0 to 0.8339 for LSP-DETR, whereas the best fine-tuned F1 scores ranged from 0.7378 to 0.8725.

Among all configurations, we obtained the highest overall F1 score of 0.8725 with LSP-DETR fine-tuned on the Hard annotations, representing an improvement of 0.0386 over its baseline. Cellpose-SAM achieved the second-highest F1 score of 0.8657 when fine-tuned on the Medium annotations, followed by Cellpose-DINO, which reached 0.8543 using the Hard annotations. Among the CellViT-based configurations, CellViT [SAM-H] achieved the highest F1 score of 0.8423, followed by CellViT++ [Virchow] at 0.8338 and CellViT [HIPT-256] at 0.8299.

We observed substantial diferences in the magnitude of improvement across models. StarDist showed the largest absolute gain, increasing from 0.7380 to 0.8332 with Medium-only fine-tuning, corresponding to an improvement of 0.0952. Cellpose-SAM showed the second-largest gain of 0.0841. Cellpose 3.0 and Cellpose-DINO improved by 0.0630 and 0.0625, respectively. We also observed smaller but consistent gains for CellViT [HIPT-256], LSP-DETR, CellViT++ [Virchow], CellViT [SAM-H], and PathoSAM. These results show that the efect of human-in-the-loop data enrichment varied according to the model configuration and its baseline.

Table 4: Best fine-tuning strategy and corresponding F1 performance for each model.
<table><tr><td>Cell Segmentation Models</td><td>Baseline Performance</td><td>Fine-tuned Performance</td><td>Fine-tuning Strategy</td></tr><tr><td>StarDist</td><td>0.7380</td><td>0.8332</td><td>Medium</td></tr><tr><td>Cellpose 3.0</td><td>0.6748</td><td>0.7378</td><td>Easy + Medium + Hard</td></tr><tr><td>Cellpose-SAM</td><td>0.7816</td><td>0.8657</td><td>Medium</td></tr><tr><td>CellViT (HIPT-256)</td><td>0.7838</td><td>0.8299</td><td>Medium</td></tr><tr><td>CellViT++ (Virchow)</td><td>0.7995</td><td>0.8338</td><td>Medium + Hard</td></tr><tr><td>CellViT (SAM-H)</td><td>0.8108</td><td>0.8423</td><td>Medium</td></tr><tr><td>Cellpose-DINO</td><td>0.7918</td><td>0.8543</td><td>Hard</td></tr><tr><td>PathoSAM</td><td>0.8247</td><td>0.8386</td><td>Medium + Hard</td></tr><tr><td>LSP-DETR</td><td>0.8339</td><td>0.8725</td><td>Hard</td></tr></table>

Note: Bold values indicate the highest F1 score, and underlined values indicate the second-highest F1 score within each performance column.

## 4.3 Qualitative Results

In Figure 5, we present representative test patches comparing the baselines with models fine-tuned using their optimal annotation strategies. We selected examples from StarDist, Cellpose-SAM, and LSP-DETR to illustrate improvements in missed-nucleus recovery and instance separation.

For StarDist, we found that Medium-only fine-tuning recovered multiple nuclei that were missed by baseline, particularly in the highlighted low-contrast and densely populated regions. In the Cellpose-SAM example, we observed improved separation of closely located nuclei together with the recovery of previously missed instances. For LSP-DETR, Hard-only fine-tuning improved the performance of cell nuclei separation while recovering additional nuclei in morphologically complex regions.

These qualitative observations were consistent with the quantitative F1 improvements. We found that the fine-tuned models more accurately captured nuclei in regions with weak contrast, dense nuclear distributions, and complex renal structures. However, diferences between the predictions and ground truth remained in some regions, particularly for closely overlapping nuclei and nuclei with poorly defined boundaries.

![](images/9d6a966e1ea81d1e28c614a064b04a361ab777a24165b47ba22a272d6279dd79.jpg)  
Figure 5: Qualitative comparison of baseline and fine-tuned models. Green contours denote nucle instances. Dashed boxes highlight recovered false negatives (FN) and improved instance separation.

## 5. DISCUSSION

In this study, we observed substantial variation in the optimal fine-tuning strategies across model configurations, suggesting that the efectiveness of diferent annotation sources is model dependent. Medium annotations were included in the best-performing strategies for seven of the nine model configurations, indicating that expert annotation of previously unresolved cases provides useful supervision beyond large-scale Easy pseudo-labels and a small set of Hard consensus failures. Although Easy + Hard fine-tuning improved over the baseline for eight of the nine models, it was not optimal for any model once Medium-containing strategies were considered, suggesting that the newly annotated Medium cases provided complementary supervision beyond the data used in our prior human-in-the-loop study. In particular, Medium-only fine-tuning achieved the best performance for StarDist, Cellpose-SAM, and both CellViT configurations, whereas Medium + Hard was optimal for CellViT++ [Virchow] and PathoSAM. However, Medium data were not universally required: Cellpose-DINO and LSP-DETR achieved their best performance using Hard-only fine-tuning, while Cellpose 3.0 benefited most from the complete Easy + Medium + Hard dataset. These diferences suggest that the value of each annotation source may depend on model-specific characteristics, including pretrained representations, segmentation formulations, and baseline alignment with renal pathology. Within the Cellpose family, this behavior may be related to the combination of Cellpose 3.0’s conventional CNN encoder and the grayscale, DAPI-like preprocessing used in our pipeline, which may reduce discriminative information in faintly stained and low-contrast Medium and Hard cases. In contrast, Cellpose-SAM and Cellpose-DINO leverage large-scale pretrained ViT encoders and may therefore be more robust to these challenging appearances.

The frequent benefit of Medium annotations also suggests that samples remaining unresolved after successive foundation-model evaluations contain complementary information that is not fully represented by either wellsegmented Easy cases or extreme Hard failures. However, because no single annotation composition consistently performed best across all models, a fixed fine-tuning dataset may not be optimal for adapting diverse segmentation architectures. Several limitations should also be considered. First, this study focuses exclusively on rena pathology, and the efectiveness of the proposed annotation strategies may difer across organs, staining protocols, or imaging domains. Second, evaluation was performed on a fixed hold-out set of 185 expert-annotated kidney pathology patches. In addition, the source-rebalancing parameters were predefined for each mixed-data strategy and applied consistently across models rather than optimized separately for individual architectures. Finally, each model–strategy configuration was evaluated using a single fine-tuning run, and variability arising from random initialization or stochastic optimization was not explicitly quantified. Future work could investigate model-specific or adaptive data-selection strategies that automatically identify the most informative annotation sources and determine how annotation composition should be adjusted for diferent segmentation models.

## 6. CONCLUSION

In this study, we evaluated seven single-source and mixed-source fine-tuning strategies using Easy pseudo-labels, newly annotated Medium cases, and expert-annotated Hard consensus failures across nine cell nuclei segmentation model configurations. For every configuration, at least one fine-tuning strategy improved F1 over its corresponding baseline on the independent kidney nuclei test set. LSP-DETR achieved the highest F1 score of 0.8725 with Hard-only fine-tuning, while StarDist showed the largest improvement, increasing from 0.7380 to 0.8332 with Medium-only fine-tuning. Medium data were included in seven of the nine optimal strategies, demonstrating the value of targeted annotation of previously unresolved cases. However, no single data composition was optimal across all models. These findings support human-in-the-loop annotation enrichment across diferent segmentation-dificulty levels as an efective approach for adapting general-purpose nuclei segmentation models to renal pathology.

## REFERENCES

[1] Guo, J., Lu, S., Cui, C., Deng, R., Yao, T., Tao, Z., Lin, Y., Lionts, M., Liu, Q., Xiong, J., Wang, Y., Zhao, S., Chang, C., Wilkes, M., Yin, M., Yang, H., and Huo, Y., “Assessment of cell nuclei AI foundation models in kidney pathology,” in [Medical Imaging 2025: Image Perception, Observer Performance, and Technology Assessment], 13409, 76–82, SPIE (2025).

[2] Guo, J., Lu, S., Cui, C., Deng, R., Yao, T., Tao, Z., Lin, Y., Lionts, M., Liu, Q., Xiong, J., Wang, Y., Zhao, S., Chang, C., Wilkes, M., Fogo, A., Yin, M., Yang, H., and Huo, Y., “Evaluating cell AI foundation models in kidney pathology with human-in-the-loop enrichment,” Communications Medicine 5(1), 495 (2025).

[3] Wang, R., Guo, J., Lu, S., Deng, R., Lu, Z., Zhu, Y., Yang, Y., Qu, C., Wang, Y., Zhao, S., Chang, C., Wilkes, M., Yin, M., Yang, H., and Huo, Y., “Evaluating new AI cell foundation models on challenging kidney pathology cases unaddressed by previous foundation models,” in [Medical Imaging 2026: Image Perception, Observer Performance, and Technology Assessment], 13932, 139320A, SPIE (2026).

[4] Stringer, C., Wang, T., Michaelos, M., and Pachitariu, M., “Cellpose: A generalist algorithm for cellular segmentation,” Nature Methods 18, 100–106 (2021).

[5] Weigert, M. and Schmidt, U., “Nuclei instance segmentation and classification in histopathology images with StarDist,” in [2022 IEEE International Symposium on Biomedical Imaging Challenges (ISBIC)], 1–4, IEEE (2022).

[6] H¨orst, F., Rempe, M., Heine, L., Seibold, C., Keyl, J., Baldini, G., Ugurel, S., Siveke, J., Gr¨unwald, B., Egger, J., and Kleesiek, J., “CellViT: Vision transformers for precise cell segmentation and classification,” Medical Image Analysis 94, 103143 (2024).

[7] Stringer, C. and Pachitariu, M., “Cellpose3: One-click image restoration for improved cellular segmentation,” Nature Methods 22(3), 592–599 (2025).

[8] Pachitariu, M., Rariden, M., and Stringer, C., “Cellpose-SAM: Superhuman generalization for cellular segmentation,” bioRxiv , 2025.04.28.651001 (2025).

[9] MouseLand, “Models: Cellpose-DINO.” Cellpose Documentation, https://cellpose.readthedocs.io/ en/latest/models.html (2026). Accessed August 2, 2026.

[10] Sim´eoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., J´egou, H., Labatut, P., and Bojanowski, P., “DINOv3,” Transactions on Machine Learning Research (2026).

[11] H¨orst, F., Rempe, M., Becker, H., Heine, L., Keyl, J., and Kleesiek, J., “CellViT++: Energy-eficient and adaptive cell segmentation and classification using foundation models,” Computer Methods and Programs in Biomedicine 277, 109206 (2026).

[12] Pek´ar, M., Musil, V., Nenutil, R., Holub, P., and Br´azdil, T., “LSP-DETR: Eficient and scalable nuclei segmentation in whole slide images,” arXiv preprint arXiv:2601.03163 (2026).

[13] Griebel, T., Archit, A., and Pape, C., “Segment anything for histopathology,” in [Proceedings of the 8th International Conference on Medical Imaging with Deep Learning], Proceedings of Machine Learning Research 301, 515–546, PMLR (2026).

[14] Graham, S., Vu, Q. D., Raza, S. E. A., Azam, A., Tsang, Y. W., Kwak, J. T., and Rajpoot, N., “HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images,” Medical Image Analysis 58, 101563 (2019).

[15] Chen, R. J., Chen, C., Li, Y., Chen, T. Y., Trister, A. D., Krishnan, R. G., and Mahmood, F., “Scaling vision transformers to gigapixel images via hierarchical self-supervised learning,” in [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition], 16144–16155 (2022).

[16] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Doll´ar, P., and Girshick, R., “Segment anything,” in [Proceedings of the IEEE/CVF International Conference on Computer Vision], 4015–4026 (2023).

[17] Vorontsov, E., Bozkurt, A., Casson, A., Shaikovski, G., Zelechowski, M., Severson, K., Zimmermann, E., Hall, J., Tenenholtz, N., Fusi, N., Yang, E., Mathieu, P., van Eck, A., Lee, D., Viret, J., Robert, E., Wang, Y. K., Kunz, J. D., Lee, M. C. H., Bernhard, J. H., Godrich, R. A., Oakley, G., Millar, E., Hanna, M., Wen, H., Retamero, J. A., Moye, W. A., Yousfi, R., Kanan, C., Klimstra, D. S., Rothrock, B., Liu, S., and Fuchs, T. J., “A foundation model for clinical-grade computational pathology and rare cancers detection,” Nature Medicine 30(10), 2924–2935 (2024).

[18] Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., and Zagoruyko, S., “End-to-end object detection with transformers,” in [Computer Vision – ECCV 2020], 213–229, Springer (2020).

[19] Ronneberger, O., Fischer, P., and Brox, T., “U-Net: Convolutional networks for biomedical image segmentation,” in [Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015], 234–241, Springer (2015).

[20] Liu, Z., Hu, H., Lin, Y., Yao, Z., Xie, Z., Wei, Y., Ning, J., Cao, Y., Zhang, Z., Dong, L., Wei, F., and Guo, B., “Swin Transformer V2: Scaling up capacity and resolution,” in [Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition], 12009–12019 (2022).

[21] Sokolova, M. and Lapalme, G., “A systematic analysis of performance measures for classification tasks,” Information Processing and Management 45(4), 427–437 (2009).