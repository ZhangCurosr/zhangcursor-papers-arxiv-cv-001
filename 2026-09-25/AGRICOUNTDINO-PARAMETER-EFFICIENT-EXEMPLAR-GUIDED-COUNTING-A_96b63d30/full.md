# AGRICOUNTDINO: PARAMETER-EFFICIENT EXEMPLAR-GUIDED COUNTING AND LOCALIZATION IN AGRICULTURE

Shengjie Guo<sup>1,3</sup>, Xin Li<sup>2</sup>, Borjana Arsova<sup>3</sup>, Hanno Scharr<sup>4</sup>, Silvio Salvi<sup>1∗</sup>

<sup>1</sup>Department of Agricultural and Food Sciences, University of Bologna, Bologna, Italy

<sup>2</sup>Department of Architecture, Built Environment and Construction Engineering, Politecnico di Milano, Milan, Italy

<sup>3</sup>Institute of Bio- and Geosciences, Plant Sciences, Forschungszentrum Jülich, Jülich, Germany

<sup>4</sup>Institute for Advanced Simulation, Data Analytics and Machine Learning, Forschungszentrum Jülich, Jülich, Germany

## ABSTRACT

Accurate counting and localization of plants and their organs support phenotyping and yield estimation, yet target appearance, scale, and density vary widely across species and imaging conditions. Exemplar boxes specify the target without categoryspecific retraining, and point predictions identify the individual instances contributing to the count. We introduce AgriCountDINO, a parameter-efficient exemplar-guided framework for joint counting and localization. It conditions frozen multiscale DINOv3 features on exemplar appearance and size, then progressively decodes them into target points. Missed-object recovery extends supervision to targets overlooked by initial matching, and exemplar-adaptive point NMS filters duplicate predictions according to exemplar scale. With 8.4M trainable parameters, approximately one-tenth of TasselNetV4’s, AgriCountDINO achieves a three-shot MAE of 11.92 on the TPC-268 benchmark, reducing counting error by 9.7% while providing individual target locations. Trained only on TPC-268, it achieves a zero-shot MAE of 14.25 on unseen generic object categories in FSC-147, improving upon the best compared zero-shot method by 6.0% without target-domain training or fine-tuning.

Index Terms— Plant Agnostic Counting, Exemplar-guided Counting, Point Localization, Computer Vision

## 1. INTRODUCTION

Accurate counting and localization of plants and their organs are fundamental to agricultural image analysis. Count estimates support yield estimation and phenotyping, while instance locations enable spatial analysis and inspection of individual predictions. Agricultural targets vary substantially across species, organs, growth stages, scales, and imaging conditions, making it difficult to extend category-specific models to diverse targets [1]. This diversity motivates exemplar-guided approaches, in which a few visual examples specify the objects of interest without requiring a separate model for each target category.

Recent counting methods can return individual target locations through point predictions, detections, or segmentation masks [2, 3, 4, 5]. In agriculture, TasselNet introduced local count regression for maize tassels [6], while TasselNetV4 combined local counting with exemplar guidance to address variation across scenes, scales, and species [1]. TPC-268 provides exemplar boxes and point annotations for hundreds of fine-grained plant categories [7], allowing counting and localization to be evaluated together across diverse agricultural targets.

TasselNetV4’s local-count predictions, however, do not explicitly identify individual targets. Exemplar-guided counting depends on relating the supplied examples to corresponding regions in the image [8]. To also localize targets, these correspondences must retain the spatial detail needed to distinguish nearby instances as plant appearance and size vary. Self-supervised visual encoders provide transferable representations [9, 10]; DINOv3, in particular, offers dense features that provide a basis for learning such correspondences [11]. This motivates AgriCountDINO<sup>1</sup>, an exemplar-guided framework for joint plant counting and localization.

AgriCountDINO uses a frozen DINOv3-pretrained encoder [11, 12] to extract multiscale image features. Exemplar appearance and size condition these features, which are fused from coarse to fine and decoded into instance points. Training updates only the exemplarconditioning, fusion, and prediction modules. Missed-object recovery supervises annotated targets overlooked by initial peak matching. At inference, the exemplar scale determines which nearby point predictions are suppressed as duplicates. The retained points specify individual target locations, and their number gives the count. Counting and point-localization results on diverse plant categories, together with zero-shot transfer to unseen generic objects, suggest that the model captures transferable visual and semantic correspondences between exemplars and targets.

Our contributions are summarized as follows:

• We introduce AgriCountDINO, an exemplar-guided model that counts and localizes diverse plants and plant organs with only 8.4M trainable parameters.

• We develop multiscale point prediction guided by exemplar appearance and size, with missed-object recovery during training and exemplar-adaptive duplicate suppression at inference.

• AgriCountDINO achieves a three-shot validation MAE of 11.92 on TPC-268, 9.7% below TasselNetV4, and a zeroshot MAE of 14.25 on FSC-147, 6.0% below the strongest compared zero-shot method.

## 2. METHOD

Given an image I and a set of exemplar boxes E, AgriCountDINO predicts target locations P<sup>ˆ</sup> and reports |P| <sup>ˆ</sup> as the count. As shown in Fig. 1, exemplar information is introduced at multiple levels of a frozen encoder before the conditioned features are aggregated for dense point prediction.

![](images/154d1b77ff59ae4e69882529e8d67e28fe610b77f13f54b8063fa0d6a122fc57.jpg)  
Fig. 1. Overview of AgriCountDINO. Exemplar appearance and size tokens condition frozen multiscale features through scale-specific query modules (SQMs). Progressive coarse-to-fine fusion produces a high-resolution representation for point scoring and offset prediction. Feature dimensions are shown for a 1024 × 1024 input.

![](images/d10e0db24b617cc65884c08b58912d5019f4bd9fd8c6613a6f8a3b31b544dd92.jpg)

![](images/42e467ececa98fc4145e358a7abca520eb4e5b418bead162973fc6739a06e5d6.jpg)  
Fig. 2. Missed-object recovery and exemplar-adaptive point NMS. (a) During training, unmatched targets are assigned to unique unused queries in their local neighborhoods. (b) During inference, nearby candidate points are suppressed using a radius determined by the exemplar scale.

## 2.1. Multiscale Exemplar Conditioning

A frozen DINOv3-pretrained encoder [11, 12] extracts four feature maps. For a 1024 × 1024 input, $C _ { 1 } , C _ { 2 } , C _ { 3 } , C _ { 4 }$ have sizes $2 5 6 ^ { 2 } \times$ $1 2 \hat { 8 } , 1 2 8 ^ { 2 } \times 2 5 6 , 6 4 ^ { 2 } \times 5 1 2$ , and $3 2 ^ { 2 } \times 1 0 2 4 .$ , respectively. A separate $1 \times 1$ convolution projects each map to 256 channels, producing $\{ F _ { l } \} _ { l = 1 } ^ { 4 }$

At each scale, RoIAlign [13] extracts an appearance token from each of the K exemplar boxes, while a shared lightweight MLP embeds each box’s normalized width and height into a size token. The K appearance tokens and K size tokens form $E _ { l } \in \mathbb { R } ^ { 2 K \times 2 5 6 }$ . Inspired by previous work [14], a scale-specific query module (SQM) uses tokens from $F _ { l }$ as queries and $E _ { l }$ as keys and values. Crossattention relates image locations to the exemplars’ appearance and size cues, and single-scale deformable attention [15] refines the resulting features, yielding $\{ Q _ { l } \} _ { l = 1 } ^ { 4 }$

## 2.2. Progressive Dense Point Decoding

The conditioned maps are fused from coarse to fine by upsampling the fused feature from the previous level and adding it to the next finer map. Let $\mathcal { R } _ { l }$ denote $2 \times$ bilinear upsampling followed by a $3 \times 3$ convolution and GELU. The decoder computes

$$
\begin{array} { r l } & { Z _ { 4 } = Q _ { 4 } , } \\ & { Z _ { l } = Q _ { l } + \mathcal { R } _ { l } ( Z _ { l + 1 } ) , \quad l = 3 , 2 , 1 , } \\ & { Q _ { f } = \mathcal { R } _ { f } ( Z _ { 1 } ) . } \end{array}\tag{1}
$$

Here, $Z _ { l }$ is the fused feature at level l. For a 1024 × 1024 input, the final upsampling produces $Q _ { f }$ with 256 channels at a resolution of 512 × 512.

Each position i in $Q _ { f }$ defines a feature vector $q _ { i }$ and a reference point $r _ { i }$ at the center of its grid cell in normalized image coordinates. Score and coordinate heads predict

$$
s _ { i } = h _ { s } ( q _ { i } ) , \qquad \hat { p } _ { i } = \mathrm { c l i p } ( r _ { i } + 2 \sigma ( h _ { p } ( q _ { i } ) ) - 1 , 0 , 1 ) .\tag{2}
$$

Here, $h _ { s }$ produces a target score, while $2 \sigma ( h _ { p } ( q _ { i } ) ) - 1$ gives a twodimensional displacement; clip bounds the resulting point to the image. A score peak may lie outside the grid cell containing its target center, particularly in crowded regions. Allowing the predicted point to cross cell boundaries lets that query localize the target center.

## 2.3. Localization-Aware Learning and Inference

The dense decoder provides a scored point at every position in $Q _ { f }$ To establish unambiguous supervision, $3 \times 3$ local score maxima above the image-wise median are matched one-to-one with annotated points. The matching favors confident predictions close to target centers. For a selected candidate k and annotation $g _ { j }$ , its cost is

$$
\begin{array} { r } { \mathcal { C } _ { k j } = \lambda _ { p } \| \hat { p } _ { k } - g _ { j } \| _ { 1 } - \lambda _ { s } \sigma ( s _ { k } ) . } \end{array}\tag{3}
$$

Here, $\hat { p } _ { k }$ and $g _ { j }$ are normalized image coordinates, $\sigma$ is the sigmoid function, and $\lambda _ { p }$ and $\lambda _ { s }$ balance localization and confidence. Matched pairs are retained only if their Euclidean distance in image pixels falls within an exemplar-dependent tolerance.

In crowded regions, closely spaced instances can produce overlapping responses on the score map, leaving fewer distinct peaks than annotated targets. Peak-based matching may therefore leave some targets without a positive query. Missed-object recovery (Fig. 2(a)) extends supervision to these targets by assigning them unused dense queries. Candidate reference positions $r _ { i }$ are searched within a $5 \times 5$ grid neighborhood of each unmatched annotation $g _ { j } .$ , followed by $\textbf { a 7 } \times \mathbf { 7 }$ search for any remaining targets. A joint one-to-one assignment minimizes the total reference-to-annotation $\ell _ { 1 }$ distance while reserving queries accepted by the initial matching. This gives nearby targets distinct training queries.

The initial matching and missed-object recovery define the positive queries for training. Accepted matches and recovered queries receive positive score labels; the remaining selected peaks receive negative labels. Their supervision is organized as

$$
\begin{array} { r l } & { \mathcal { L } = \mathcal { L } _ { \mathrm { s c o r e } } + \lambda _ { \mathrm { l o c } } \mathcal { L } _ { \mathrm { l o c } } } \\ & { ~ + \rho _ { t } \left( \lambda _ { \mathrm { x y } } \mathcal { L } _ { \mathrm { r e c } } ^ { \mathrm { x y } } + \lambda _ { \mathrm { p e a k } } \mathcal { L } _ { \mathrm { r e c } } ^ { \mathrm { p e a k } } \right) + \alpha ( \mathcal { E } ) \mathcal { L } _ { \mathrm { a u x } } . } \end{array}\tag{4}
$$

The masked squared-error term $\mathcal { L } _ { \mathrm { s c o r e } }$ supervises selected peaks and recovered queries. The $\ell _ { 1 }$ term $\mathcal { L } _ { \mathrm { l o c } }$ trains point coordinates on accepted initial matches. For recovered queries, $\mathcal { L } _ { \mathrm { r e c } } ^ { \mathrm { x y } }$ penalizes coordinate error, while $\mathcal { L } _ { \mathrm { r e c } } ^ { \mathrm { p e a k } }$ encourages each query to score above nearby unassigned queries in its $3 \times 3$ neighborhood. Accepted and recovered positives are excluded from this comparison, and both recovery terms place greater weight on closely spaced targets relative to the exemplar scale. Recovered queries enter $\mathcal { L } _ { \mathrm { s c o r e } }$ from the first epoch; $\rho _ { t }$ gradually introduces only their coordinate and peak losses over the first eight epochs. The auxiliary loss $\mathcal { L } _ { \mathrm { a u x } }$ supervises predictions from $Q _ { 1 }$ and is weighted by $\alpha ( \mathcal { E } )$ for small targets.

At inference, confident local score maxima yield candidate points, but several peaks may regress to the same target. Exemplaradaptive point NMS (Fig. 2(b)) processes candidates in descending score order and suppresses points near each retained prediction. Its radius is based on $\begin{array} { r } { \dot { \frac { 1 } { 2 } } \operatorname* { m i n } ( \bar { w } , \bar { h } ) } \end{array}$ , where w¯ and h<sup>¯</sup> are the mean exemplar width and height in image pixels. The retained points form $\begin{array} { r } { \hat { \mathcal P } _ { : } } \end{array} \hat { \mathcal P } _ { : }$ whose cardinality gives the count.

## 3. EXPERIMENTS

## 3.1. Experimental Setup

## 3.2. Experimental Setup

Implementation details. AgriCountDINO uses a frozen DINOv3- pretrained ConvNeXt-B encoder [11, 12]. Images are resized to $1 0 2 4 \times 1 0 2 4$ , with up to three exemplar boxes used during training. The trainable parameters are optimized for 100 epochs with AdamW [18], a learning rate and weight decay of $1 0 ^ { - 4 }$ , an effective batch size of 32, and BF16 mixed precision. The matching weights are $\lambda _ { p } = 1$ and $\lambda _ { s } = 2 ;$ the loss weights are $\lambda _ { \mathrm { l o c } } = 1 , \lambda _ { \mathrm { x y } } = 0 . 5 ,$ and $\lambda _ { \mathrm { p e a k } } = 0 . 0 5$ . Recovery coordinate and peak losses are gradually introduced over the first eight epochs. The auxiliary weight is 0.3 when the smaller mean exemplar dimension is below 25 pixels, and zero otherwise.

Datasets and metrics. AgriCountDINO is trained on the 7,000 images in the official TPC-268 training split and evaluated in one-shot and three-shot settings. MAE, RMSE, and $R ^ { 2 }$ measure image-level counting accuracy. To assess the spatial distribution of predicted counts, GAME-L [19] sums absolute count errors over $4 ^ { L }$ image regions; we report levels $L = 1 , 2 , 3$ . This exposes regional errors that can cancel in the image-level count. Point-level precision and

![](images/7776319eae6c7e90b419b3d8be3a773c85fe8f9090872d174d6ca17011e2a378.jpg)  
Fig. 3. Qualitative comparison with TasselNetV4. Exemplar boxes are shown in the ground truth, and dots denote target locations. The top three rows are from TPC-268; the bottom two show zero-shot transfer to FSC-147.

C<sup>2</sup>recall assess individual target localization. Zero-shot transfer is eval-Tuated on the generic-object dataset FSC-147 without target-domain training or fine-tuning.

## 3.3. Counting Accuracy and Spatial Localization

With 8.40M trainable parameters, approximately one tenth of TasselNetV4’s, AgriCountDINO achieves the lowest TPC-268 validation MAE and RMSE in both exemplar settings (Table 1). In the 4three-shot setting, its MAE of 11.92 and RMSE of 28.55 reduce the S<sup>C</sup>corresponding errors of TasselNetV4 by 9.7% and 35.0%. With one exemplar, it obtains a validation MAE of 13.16. On the test split, the three-shot and one-shot MAEs are 21.76 and 22.03, compared with 22.95 and 22.20 for TasselNetV4.

To examine whether accurate counts are accompanied by accurate target placement, we also evaluate predicted point localization. In the three-shot setting, AgriCountDINO reduces GAME3 from 33.33 for TasselNetV4 to 25.37, a 23.9% reduction at the finest reported grid level (Table 2). Its predicted points achieve 0.719 precision and 0.729 recall. The examples in Fig. 3 show points aligned with individual targets, including closely spaced instances where TasselNetV4 produces broader counting responses.

Remaining errors on the validation and test sets may arise from both difficult target configurations and annotation quality. Dense or visually ambiguous instances can challenge point prediction, while Fig. 4 shows examples of apparent discrepancies between visible targets and point annotations. A prediction on a plausible but unannotated target may therefore increase the measured error.

Table 1. Comparison with representative state-of-the-art methods on TPC-268. Best and second-best results under each shot setting are shown in bold and underline, respectively.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Shot</td><td colspan="3">Validation</td><td colspan="3">Test</td><td rowspan="2">Train. Params.↓</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td><td>MAE↓</td><td>RMSE↓</td><td> $R ^ { 2 } \uparrow$ </td></tr><tr><td>LOCA (ICCV’23)[16]</td><td>3</td><td>17.26</td><td>53.19</td><td>0.75</td><td>17.51</td><td>38.37</td><td>0.78</td><td>11.37M</td></tr><tr><td>CACViT (AAAI&#x27;24)[17]</td><td>3</td><td>16.63</td><td>42.49</td><td>0.82</td><td>22.04</td><td>41.79</td><td>0.73</td><td>99.77M</td></tr><tr><td>TasselNetV4 (ISPRS J.&#x27;26)[1]</td><td>3</td><td>13.20</td><td>43.93</td><td>0.83</td><td>22.95</td><td>51.36</td><td>0.67</td><td>84.67M</td></tr><tr><td>AgriCountDINO (Ours)</td><td>3</td><td>11.92</td><td>28.55</td><td>0.93</td><td>21.76</td><td>49.58</td><td>0.74</td><td>8.40M</td></tr><tr><td>LOCA (ICCV’23)[16]</td><td>1</td><td>17.19±0.31</td><td> $4 8 . 1 4 \pm 2 . 1 9$ </td><td> $0 . 8 0 { \pm } 0 . 0 2$ </td><td>21.47±0.29</td><td>42.36±0.72</td><td> $\mathbf { 0 . 7 3 \pm 0 . 0 1 }$ </td><td>11.37M</td></tr><tr><td>CACViT (AAAI&#x27;24)[17]</td><td>1</td><td>17.96±0.16</td><td> $4 3 . 3 8 { \pm } 0 . 4 7$ </td><td> $0 . 8 3 { \pm } 0 . 0 0$ </td><td> $2 2 . 0 6 { \pm } 0 . 1 1$ </td><td>42.97±0.81</td><td> $\underline { { 0 . 7 1 \pm 0 . 0 1 } }$ </td><td>99.77M</td></tr><tr><td>TasselNetV4 (ISPRS J.&#x27;26)[1]</td><td>1</td><td> $1 3 . 4 9 { \pm } 0 . 0 2 $ </td><td> $4 1 . 3 0 { \pm } 0 . 4 6 $ </td><td> $0 . 8 5 { \pm } 0 . 0 0 $ </td><td> $2 2 . 2 0 { \pm } 0 . 1 1$ </td><td>48.70±0.26</td><td> $0 . 6 7 { \pm } 0 . 0 0$ </td><td>84.67M</td></tr><tr><td>AgriCountDINO (Ours)</td><td>1</td><td>13.16±0.07</td><td> $\mathbf { 3 3 . 1 6 { \pm } 0 . 6 6 }$ </td><td> $\mathbf { 0 . 9 0 { \overset { . } { \bot } } 0 . 0 0 }$ </td><td> $\underline { { 2 2 . 0 3 \pm 0 . 1 8 } }$ </td><td>50.08±0.46</td><td> $0 . 7 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>8.40M</td></tr></table>

Table 2. Spatial counting and localization on TPC-268 validation. G1–G3 denote GAME1–GAME3.
<table><tr><td>Method</td><td>Shot</td><td>G1↓</td><td>G2↓</td><td>G3↓</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td>TasselNetV4 [1]</td><td>3</td><td>17.71</td><td>22.75</td><td>33.33</td><td>一</td><td>一</td></tr><tr><td>TasselNetV4 [1]</td><td>1</td><td>19.14</td><td>23.86</td><td>33.93</td><td>一</td><td>一</td></tr><tr><td>AgriCountDINO</td><td>3</td><td>14.92</td><td>18.64</td><td>25.37</td><td>.719</td><td>.729</td></tr><tr><td>AgriCountDINO</td><td>1</td><td>15.90</td><td>19.72</td><td>27.17</td><td>.713</td><td>.709</td></tr></table>

Table 3. Three-shot in-domain and zero-shot TPC-268→FSC-147 results.
<table><tr><td rowspan="2">Method</td><td colspan="2">FSC-147→FSC-147</td><td colspan="2">TPC-268→FSC-147</td></tr><tr><td>MAE↓</td><td>RMSE↓</td><td>MAE↓</td><td>RMSE↓</td></tr><tr><td>LOCA [16]</td><td>10.79</td><td>56.97</td><td>15.16</td><td>109.15</td></tr><tr><td>CACViT [17]</td><td>9.13</td><td>48.96</td><td>17.88</td><td>82.57</td></tr><tr><td>TasselNetV4 [1]</td><td>16.16</td><td>110.56</td><td>19.17</td><td>116.24</td></tr><tr><td>AgriCountDINO</td><td>11.62</td><td>85.07</td><td>14.25</td><td>106.88</td></tr></table>

## 3.4. Zero-Shot Cross-Dataset Transfer

Trained only on TPC-268, AgriCountDINO transfers directly to unseen generic object categories in FSC-147 without target-domain training or fine-tuning. Table 3 reports a zero-shot MAE of 14.25, improving on LOCA’s 15.16 by 6.0%. The FSC-147 examples in Fig. 3 further show that the model localizes individual targets in these unseen categories. These results suggest that the model captures transferable semantic correspondences between exemplars and target objects, enabling counting and localization beyond the agricultural categories seen during training.

## 3.5. Ablation Study

The ablations in Table 4 reveal distinct roles for the three components. Replacing DINOv3 with supervised ConvNeXt-B causes the largest increase in MAE (11.92 to 17.64), suggesting that the DI-NOv3 representation provides a stronger basis for relating exemplars to diverse plant targets. Removing scale-wise conditioning worsens both MAE and GAME3, indicating that exemplar guidance across the feature hierarchy contributes to accurate global and regional counts. Without missed-object recovery, point recall falls from 0.729 to 0.692 and MAE increases, consistent with more targets being omitted after initial peak matching. F1 remains unchanged, so the clearest contribution of recovery is improved instance coverage.

Table 4. Ablation study on the TPC-268 validation set under the 3- shot setting.
<table><tr><td>Configuration</td><td>MAE↓</td><td>G3↓</td><td>F1↑</td><td>Recall↑</td></tr><tr><td>Supervised ConvNeXt-B prior</td><td>17.64</td><td>32.19</td><td>0.668</td><td>0.646</td></tr><tr><td>w/o scale-wise exemplar conditioning</td><td>15.12</td><td>31.78</td><td>0.669</td><td>0.632</td></tr><tr><td>w/o missed-object recovery</td><td>12.71</td><td>26.27</td><td>0.723</td><td>0.692</td></tr><tr><td>Full model</td><td>11.92</td><td>25.37</td><td>0.723</td><td>0.729</td></tr></table>

![](images/f8c8783a0c029c0a40b90d3cac65cffdaa91b95b25c95a0b71d50eb11c3f93ad.jpg)  
Fig. 4. Possible annotation discrepancies in TPC-268. Top: annotations and exemplar boxes; bottom: AgriCountDINO predictions. Red boxes highlight apparent discrepancies

## 4. CONCLUSIONS

We introduced AgriCountDINO, a parameter-efficient model for exemplar-guided counting and localization of plants and plant organs. It conditions frozen multiscale DINOv3 features on exemplar appearance and size, then predicts individual target points whose number gives the count. With 8.4M trainable parameters, Agri-CountDINO achieves lower validation counting and spatial errors than TasselNetV4 on TPC-268. It also obtains the lowest zero-shot MAE among the compared methods on FSC-147 without targetdomain fine-tuning. These results demonstrate AgriCountDINO’s ability to combine accurate counting, instance localization, and transfer beyond the agricultural categories used for training.

## 5. ACKNOWLEDGMENT

This work was co-funded by the European Union through the FutureData4EU project (Grant Agreement No. 101126733). The views expressed are those of the authors and do not necessarily reflect those of the European Union or REA. We acknowledge ISCRA for awarding access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking and hosted by CINECA (Italy).

## 6. REFERENCES

[1] Xiaonan Hu, Xuebing Li, Jinyu Xu, Abdulkadir Duran Adan, Letian Zhou, Xuhui Zhu, Yanan Li, Wei Guo, Shouyang Liu, Wenzhong Liu, and Hao Lu, “Tasselnetv4: A vision foundation model for cross-scene, cross-scale, and cross-species plant counting,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 231, pp. 745–760, 2026.

[2] Qingyu Song, Changan Wang, Zhengkai Jiang, Yabiao Wang, Ying Tai, Chengjie Wang, Jilin Li, Feiyue Huang, and Yang Wu, “Rethinking counting and localization in crowds: A purely point-based framework,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 3365–3374.

[3] Thanh Nguyen, Chau Pham, Khoi Nguyen, and Minh Hoai, “Few-shot object counting and detection,” in Computer Vision – ECCV 2022, Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, Eds., Cham, 2022, pp. 348–365, Springer Nature Switzerland.

[4] Jer Pelhan, Alan Lukežic, Vitjan Zavrtanik, and Matej Kristan, “Dave - a detect-and-verify paradigm for low-shot counting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2024, pp. 23293– 23302.

[5] Jer Pelhan, Alan Lukežic, Vitjan Zavrtanik, and Matej Kris-ˇ tan, “A novel unified architecture for low-shot counting by detection and segmentation,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds. 2024, vol. 37, pp. 66260–66282, Curran Associates, Inc.

[6] Hao Lu, Zhiguo Cao, Yang Xiao, Bohan Zhuang, and Chunhua Shen, “TasselNet: Counting maize tassels in the wild via local counts regression network,” Plant Methods, vol. 13, no. 1, pp. 79, 2017.

[7] Jinyu Xu, Tianqi Hu, Xiaonan Hu, Letian Zhou, Songliang Cao, Meng Zhang, and Hao Lu, “Plant taxonomy meets plant counting: A fine-grained, taxonomic dataset for counting hundreds of plant species,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 167–177.

[8] Min Shi, Hao Lu, Chen Feng, Chengxin Liu, and Zhiguo Cao, “Represent, compare, and learn: A similarity-aware framework for class-agnostic counting,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 9529–9538.

[9] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023, pp. 15619–15629.

[10] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin, “Emerging properties in self-supervised vision transformers,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 9650–9660.

[11] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Theo Moutakanni, Leo Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski, “Dinov3,” arXiv preprint, 2025.

[12] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie, “A convnet for the 2020s,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 11976–11986.

[13] Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick, “Mask R-CNN,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2961–2969.

[14] Jer Pelhan, Alan Lukežic, and Matej Kristan, “Generalized-ˇ scale object counting with gradual query aggregation,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, 2026, vol. 40, pp. 8314–8321.

[15] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai, “Deformable DETR: Deformable transformers for end-to-end object detection,” in International Conference on Learning Representations (ICLR), 2021.

[16] Nikola Ðukic, Alan Lukeži´ c, Vitjan Zavrtanik, and Matej Kris-ˇ tan, “A low-shot object counting network with iterative prototype adaptation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18872– 18881.

[17] Zhicheng Wang, Liwen Xiao, Zhiguo Cao, and Hao Lu, “Vision transformer off-the-shelf: A surprising baseline for few-shot class-agnostic counting,” arXiv preprint arXiv:2305.04440, 2023.

[18] Ilya Loshchilov and Frank Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.

[19] Ricardo Guerrero-Gómez-Olmedo, Beatriz Torre-Jiménez, Roberto López-Sastre, Saturnino Maldonado-Bascón, and Daniel Oñoro-Rubio, “Extremely overlapping vehicle counting,” in Pattern Recognition and Image Analysis (IbPRIA). 2015, pp. 423–431, Springer.