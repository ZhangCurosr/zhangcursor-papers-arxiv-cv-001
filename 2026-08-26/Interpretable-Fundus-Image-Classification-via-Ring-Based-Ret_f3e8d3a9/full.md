# Interpretable Fundus Image Classification via Ring-Based Retinal Vasculature Features

Xiaoyan Li<sup>1</sup>, Shixin Xu<sup>3</sup>, Arvind Gupta<sup>1</sup>, Huaxiong Huang<sup>1,2,3,4\*</sup>

<sup>1\*</sup>Computer Science, University of Toronto, 27 King’s College Cir., Toronto, M5S 1A1, Ontario, Canada.

<sup>2</sup>Mathematics and Statistics, York University, 4700 Keele St., Toronto, M3J 1P3, Ontario, Canada.

<sup>3</sup>Zu Chongzhi Center, Duke Kunshan University, No.8 Duke Ave., Kunshan, 215300, Jiangsu, China.

<sup>4</sup>Mathematics, Analytics, and Data Science Lab, Fields Institute for Research in Mathematical Sciences, 222 College Street, Toronto, M5T 3J1, Ontario, Canada.

\*Corresponding author(s). E-mail(s): hhuang@yorku.ca; Contributing authors: xiaoy.li@mail.utoronto.ca; shixin.xu@dukekunshan.edu.cn; arvind.gupta@utoronto.ca;

## Abstract

Retinal fundus photography is widely used for screening and monitoring ocular diseases, but many modern classification pipelines rely on deep latent representations and provide limited interpretability. This study develops an interpretable fundus image classification framework based on a ring-structured representation of the retinal vasculature centered on the optic disc. The method quantifies vessel geometry, color appearance, oxygenation-related vascular appearance, and vessel–background entropy within concentric retinal regions. These physiologically motivated descriptors are derived from vessel masks, image intensities, and optical-density measurements and aggregated across rings to capture spatial variation in vascular properties. Using only quantitative vascular descriptors, the proposed method achieved strong classification performance across three public fundus datasets. On HRF, it achieved 91.1% accuracy using automatically generated vessel masks, matching RETFound, a vision transformer pretrained on large-scale retinal fundus image data, under the same evaluation setting. Additional analyses suggest that pretrained image models are sensitive to acquisition-related spatial cues, including fundus scale and retinal position

within the field of view, as well as broader non-vessel image characteristics. This framework may support interpretable disease classification, quantitative retinal phenotyping, and retinal biomarker discovery without requiring large task-specific training datasets.

Keywords: Diabetic retinopathy, glaucoma, fundus photography, retinal vessel analysis, Optical density, vascular geometry

## 1 Introduction

Diabetic retinopathy (DR) and glaucoma are among the leading causes of vision impairment and blindness worldwide [1–4]. DR is a microvascular complication of diabetes and a major cause of vision loss in working-age adults, whereas glaucoma is a chronic optic neuropathy and a leading cause of irreversible blindness, particularly in aging populations. Identifying early retinal manifestations of these diseases is therefore critical for timely intervention and the prevention of vision loss.

Fundus photography represents one of the most widely used imaging modalities in ophthalmology and is routinely employed for screening, diagnosis, and longitudinal monitoring. Ophthalmologists visually assess color fundus images for disease-related signs, relying largely on qualitative judgment of vascular abnormalities (e.g., caliber changes, tortuosity), optic disc appearance, and the presence and distribution of lesions. Although fundus images contain rich structural, color, and textural information, qualitative grading may introduce subjectivity and variability. In this context, computer-aided diagnosis (CAD) systems can provide complementary quantitative assessments that are more reproducible across patients and imaging conditions.

Deep learning has driven substantial progress in automated fundus image analysis, achieving strong performance in DR and glaucoma classification [5–10]. Vision transformer (ViT)–based architectures [6, 8] and hybrid CNN–ViT models [7, 9] have shown competitive performance relative to conventional convolutional approaches, particularly when supported by large-scale pretraining. However, despite their strong predictive performance, deep models typically rely on latent representations that are dificult to interpret in clinically meaningful terms. Their development may also require substantial training data and computational resources, and their predictions can be influenced by acquisition-related or other image characteristics that are not directly aligned with the retinal abnormalities used in clinical assessment.

From a clinical point of view, vascular alterations are central to the pathophysiology of both DR and glaucoma and are readily observable in color fundus images. DR is associated with microvascular remodeling, venous dilation, oxygenation-related appearance changes, and structural alterations in arterioles and venules. In glaucoma, peripapillary vascular attenuation and optic disc-centered changes have been documented [11–13]. These observations motivate the use of vascular descriptors as a clinically meaningful representation for fundus image classification. Moreover, because vascular morphology and appearance (including color- and oxygenation-related contrast) vary with retinal eccentricity (i.e., with distance from the optic disc), capturing these properties within concentric annular regions centered at the optic disc provides a structured, spatially organized characterization of retinal vasculature. Accordingly, there is a need for classification frameworks that retain strong discriminative power while providing a more transparent link between model predictions and clinically meaningful retinal features.

In this work, we introduce an interpretable ring-based vascular feature representation for fundus image classification. The proposed approach explicitly quantifies vascular geometry, color characteristics, oxygenation-related appearance, and vessel– background texture contrast within concentric annular regions centered on the optic disc. These measurements produce a compact set of physiologically motivated descriptors that capture spatial variations in vascular structure across retinal eccentricity. All features are computed from vessel masks and optical density measurements, after which ring-wise feature vectors are concatenated and used as input to a shallow classifier for three-class classification among healthy, DR, and glaucomatous fundus images.

Prior studies have investigated retinal vascular descriptors and oxygenation-related measurements, but these sources of information have often been examined separately. Our framework integrates oxygenation-related appearance with structural vascular descriptors within a unified ring-wise representation for interpretable classification across multiple retinal disease categories. Rather than focusing solely on absolute classification accuracy, we evaluate the discriminative information captured by explicit vascular descriptors relative to learned deep image representations. Accordingly, we compare the proposed method with pretrained image-based models, including RET-Found [14], a ViT–based retinal foundation model pretrained on approximately 1.6 million retinal images and subsequently adapted to downstream classification tasks.

Furthermore, we examine how acquisition-related spatial characteristics, including field-of-view (FOV) size and retinal position within the image, influence the performance of pretrained image models. Although these characteristics may contribute to classification performance, they are not directly aligned with the retinal abnormalities commonly used in clinical assessment. In contrast, the proposed ring-based representation is constructed from physiologically meaningful vascular measurements, providing a more transparent link between model predictions and retinal characteristics.

The main contributions of this work are summarized as follows:

• We propose an interpretable optic-disc-centered ring representation for fundus image classification that integrates vessel geometry, red–green color characteristics, oxygenation-related appearance, and vessel–background entropy across retinal eccentricity.

• We demonstrate across multiple public datasets that these compact, physiologically motivated descriptors provide strong classification performance and enable direct interpretation of feature and annular-region contributions, ofering a transparent complement to pretrained deep image representations.

• We assess the sensitivity of pretrained image models to FOV size and retinal position within the image through controlled spatial standardization experiments. The observed performance changes highlight the potential influence of acquisition-related spatial characteristics when interpreting deep-model classification performance.

## 2 Results

## 2.1 Benchmarking Against RETFound and Other Pretrained Image-Based Baselines

Table 1 compares the proposed ring-based representation with RETFound [14] and three ImageNet-pretrained image-based baselines: ResNet-50 [15], ViT-B/16 [16], and ConvNeXt-B [17]. For the proposed method, results are reported using vessel masks predicted by RIP-AV [18] and, where available, expert vessel masks. Lightweight finetuning improved RETFound performance on HRF but did not yield consistent gains for the remaining datasets or pretrained backbones. We therefore report the fine-tuned RETFound result for HRF and frozen-feature results for the other evaluation settings.

On HRF, the proposed method achieved an accuracy of 1.000 using expert vessel masks. When predicted vessel masks were used, its accuracy decreased to 0.911, matching RETFound in accuracy and remaining higher than the other ImageNetpretrained baselines. The diference between the ground-truth- and predicted-mask results indicates that vessel-segmentation quality can influence downstream classification, particularly for descriptors that depend on accurate delineation of fine vascular structures.

A progressive evaluation using ground-truth vessel masks further showed that HRF classification performance generally improved as additional peripheral annuli were included, particularly for the combined and geometry-based feature sets. This result suggests that peripheral vascular structure provides complementary information beyond the immediate peripapillary region. Detailed results are provided in Supplementary Section S14, Progressive Ring-Wise Evaluation.

On SUSTech-SYSU, all methods performed strongly for binary DR-versus-healthy classification. The proposed representation achieved an accuracy of 0.896, while the pretrained image models achieved higher accuracies, with ConvNeXt-B and ViT-B/16 slightly outperforming RETFound. This suggests that, in this dataset, full-image models may have captured informative global cues beyond the explicit vascular measurements emphasized by the proposed method.

Performance was lower across methods on FIVES. Using expert vessel masks, the proposed method achieved an accuracy of 0.766, comparable to ConvNeXt-B at 0.769 and ViT-B/16 at 0.753, but approximately four percentage points below RET-Found at 0.806. Its accuracy decreased to 0.724 when predicted vessel masks were used. The lower performance may partly reflect the greater image-quality variability in FIVES, including blur, distortion, and reduced contrast, which can obscure fine vascular structures and reduce the reliability of segmentation-dependent descriptors. Per-class sensitivity, specificity, F1-score, and one-vs-rest ROC-AUC for each dataset and method are reported in Supplementary Table S7.

Table 1 Classification performance of the proposed method and pretrained image-based baselines.
<table><tr><td>Dataset</td><td>Method</td><td>Acc.</td><td>Macro-F1</td><td>ROC-AUC</td><td>95% CI</td></tr><tr><td rowspan="5">HRF</td><td>Proposed (GT) Proposed (Pred.)</td><td>1.000</td><td>1.000</td><td>1.000</td><td>[0.921, 1.000]</td></tr><tr><td></td><td>0.911</td><td>0.910</td><td>0.964</td><td>[0.793, 0.965]</td></tr><tr><td>RETFound</td><td>0.911</td><td>0.911</td><td>0.966</td><td>[0.793, 0.965]</td></tr><tr><td>ViT-B/16</td><td>0.800</td><td>0.800</td><td>0.920</td><td>[0.662, 0.891]</td></tr><tr><td>ConvNeXt-B</td><td>0.867</td><td>0.868</td><td>0.958</td><td>[0.738, 0.937]</td></tr><tr><td rowspan="5"></td><td>ResNet-50</td><td>0.778</td><td>0.780</td><td>0.901</td><td>[0.637, 0.875]</td></tr><tr><td>Proposed (GT)</td><td>0.766</td><td>0.740</td><td>0.884</td><td>[0.721, 0.806]</td></tr><tr><td>Proposed (Pred.)</td><td>0.724</td><td>0.700</td><td>0.843</td><td>[0.677, 0.766]</td></tr><tr><td>RETFound</td><td>0.806</td><td>0.788</td><td>0.931</td><td>[0.763, 0.842]</td></tr><tr><td>ViT-B/16 ConvNeXt-B</td><td>0.753</td><td>0.741</td><td>0.879</td><td>[0.708, 0.794]</td></tr><tr><td rowspan="5"></td><td>ResNet-50</td><td>0.769 0.724</td><td>0.755 0.710</td><td>0.892 0.848</td><td>[0.724, 0.809] [0.678, 0.767]</td></tr><tr><td>Proposed (Pred.)</td><td>0.896</td><td></td><td></td><td></td></tr><tr><td>RETFound</td><td>0.946</td><td>0.888 0.946</td><td>0.948 0.983</td><td>[0.875, 0.913]</td></tr><tr><td>ViT-B/16</td><td>0.953</td><td>0.949</td><td>0.977</td><td>[0.930, 0.958]</td></tr><tr><td>ConvNeXt-B</td><td>0.964</td><td>0.961</td><td></td><td>[0.938, 0.964]</td></tr><tr><td></td><td>ResNet-50</td><td>0.944</td><td>0.940</td><td>0.989 0.977</td><td>[0.950, 0.974] [0.928, 0.956]</td></tr></table>

Note: HRF and FIVES were evaluated using leave-one-out cross-validation, whereas SYSU was evaluated using stratified five-fold cross-validation. Accuracy and its 95% Wilson confidence interval were computed from aggregated out-of-fold predictions. Macro-F1 and ROC-AUC were also computed from the aggregated out-of-fold predictions. GT, groundtruth vessel masks; Pred., predicted vessel masks; SYSU, SUSTech-SYSU. For RETFound, the fine-tuned result is reported on HRF, whereas frozen-feature results are reported on FIVES and SYSU. On FIVES, all pretrained image-based baselines were evaluated using the FOV-standardized images described in Section 2.2.

## 2.2 Analysis of FIVES-Specific Acquisition-Related Spatial Characteristics

We investigated whether acquisition-related spatial characteristics in FIVES could influence the performance of pretrained image-based classifiers. Specifically, we examined FOV size, FOV center, and optic-disc location across disease groups. This analysis focused on FIVES because it showed the most pronounced group-wise spatial diferences among the datasets studied, together with noticeable changes in classification performance after FOV standardization. The corresponding spatial diferences were less pronounced in HRF and SUSTech-SYSU and did not produce comparably large changes in classification accuracy.

As shown in Fig. 1, the disease groups exhibit distinct FOV-related spatial characteristics. Healthy images generally have larger inscribed fundus radii, concentrated around 990–996 pixels, whereas glaucoma and DR images span a broader range of approximately 975–985 pixels (Fig. 1(a)). Healthy images also show a more compact distribution of FOV centers, while glaucoma and DR images exhibit greater spatial variability (Fig. 1(b)). By contrast, the distributions of optic-disc centers largely overlap across groups, with no clear class-dependent separation (Fig. 1(c)). These observations suggest that the group-wise FOV diferences are more likely related to image acquisition or preprocessing than to systematic anatomical variation.

To reduce the influence of these spatial characteristics, we standardized the FOV by retaining the common retinal region shared across all images. This procedure primarily afected the outer FOV boundary while preserving most of the valid retinal field. Nevertheless, classification performance decreased for all pretrained image-based models. RETFound showed the largest reduction in accuracy, from 0.927 on the original images to 0.806 after FOV standardization. Accuracy also decreased from 0.829 to 0.769 for ConvNeXt-B, from 0.819 to 0.753 for ViT-B/16, and from 0.798 to 0.724 for ResNet-50. These performance decreases suggest that the pretrained models were sensitive to FOV geometry, image boundaries, or related acquisition characteristics. Although this experiment does not isolate the contribution of each spatial cue, it demonstrates that acquisition-related spatial variation can materially afect image-based classification performance.

![](images/11d4c9aba2097b4a51c23fce95de5bd7d5ac8cb240413b62d8b88f9bd545c490.jpg)  
(a)

![](images/67c3e315b304a00f0e3b76e4b72655a10cbe5ee6a230e2910b6b38da5a7a66ec.jpg)  
(b)

![](images/063a4c41b8525e0affbc5d22dba925172390dc5e2f65b750affeaf9f370ff2ce.jpg)  
(c)  
Fig. 1 Acquisition-related spatial characteristics in the FIVES dataset. (a) Distribution of the inscribed fundus FOV radius. (b) Spatial distribution of FOV centers. (c) Spatial distribution of optic-disc centers.

## 2.3 Sensitivity of RETFound to Non-Vessel Background Appearance

We further examined whether pretrained RETFound was sensitive to non-vessel image information beyond the vascular structures emphasized by the proposed representation. Using FOV-standardized FIVES images, we evaluated two complementary controls: broad perturbations that standardized or suppressed non-vessel color, contrast, and brightness, and a lesion-aware perturbation that selectively inpainted automatically detected lesions in non-vessel regions. The vessel, near-vessel, and optic-disc regions were preserved in both analyses.

Broad background perturbations substantially reduced classification performance relative to the FOV-standardized baseline accuracy of 0.806. Accuracy decreased to 0.725 after background color standardization, 0.763 after contrast reduction, and 0.760 after brightness standardization. Joint standardization of color, contrast, and brightness reduced accuracy to 0.682, which was similar to the result obtained by replacing the valid non-vessel background with a shared constant value. These reductions indicate that RETFound was sensitive to broad variation in non-vessel appearance. However, because these regions may contain both clinically meaningful pathology and acquisition-related variation, the perturbations should not be interpreted as isolating acquisition bias alone.

To examine the contribution of localized lesion-related information more directly, non-vessel lesions were automatically detected using a pretrained fundus-lesion segmentation method [19] and inpainted using LaMa [20]. This lesion-aware perturbation produced only a small change in overall accuracy, from 0.806 to 0.803, and in DR recall, from 0.838 to 0.831. The limited change suggests that suppressing the automatically detected lesion regions alone did not substantially alter RETFound predictions in this evaluation. Nevertheless, it does not establish that RETFound was independent of lesion-related information, because lesions may have been incompletely detected and residual contextual or textural information may have remained after inpainting.

Overall, the larger efects of broad background perturbations than lesion-aware inpainting suggest that RETFound was sensitive to a mixture of non-vessel appearance characteristics rather than only to the localized lesions targeted by the inpainting procedure. These characteristics may include clinically relevant retinal abnormalities, local texture, illumination, reflections, pigmentation-related variation, and acquisitiondependent appearance. Detailed perturbation procedures and complete performance results are provided in Supplementary Section S8 Controlled Non-Vessel Background Perturbations and Lesion-Aware Inpainting.

## 2.4 Qualitative Comparison with RETFound Explanations

Although pretrained RETFound achieves strong classification performance, its learned image-level representations are less directly tied to predefined vascular measurements than the proposed ring-wise descriptors. The proposed method provides complementary interpretability by operating on explicit vascular features extracted from concentric annular regions. In this subsection, we qualitatively compare explanation maps derived from RETFound with those produced by our ring-based approach: RET-Found provides pixel- or patch-level attribution maps, whereas the proposed method provides ring-level vascular evidence based on measurable retinal descriptors.

## Ring-Based Evidence Heatmap

Let $\mathbf { x } _ { \mathrm { s t d } }$ denote the standardized feature vector (zero mean and unit variance per feature), and let ${ \bf w } _ { c }$ and $b _ { c }$ denote the learned weights and bias for class c. The model produces class logits $\ell _ { c } = \mathbf { w } _ { c } ^ { \top } \mathbf { x } _ { \mathrm { s t d } } + b _ { c }$ , yielding the predicted class $c ^ { * }$ and the runnerup class c˜ (the class with the second-largest logit). Each feature $j$ contributes a signed amount to the logit diference between $c ^ { * }$ and c˜. Specifically, we define the feature-level evidence as $\delta _ { j } = ( w _ { c ^ { * } , j } - w _ { \tilde { c } , j } ) x _ { \mathrm { s t d } , j }$ , where $\delta _ { j } > 0$ indicates support for the predicted class over the runner-up and $\delta _ { j } < 0$ indicates support for the runner-up. Since the features are naturally grouped by annular rings, for a given ring mask A we define the ring-wise evidence score as $\textstyle \psi ( A ) = \sum _ { j \in { \mathcal { I } } ( A ) } \delta _ { j }$ , where $\mathcal { I } ( A )$ denotes the set of features extracted from ring A. The final heatmap assigns the ring-level score $\psi ( A )$ to vessel pixels within ring A. Fig. 2 shows representative examples of this visualization. Rings with larger positive $\psi ( A )$ appear more prominently, indicating stronger support for the predicted class over the runner-up.

## RETFound Explanation Map

Because transformer attention weights characterize internal token interactions rather than directly providing class-specific attribution to the final classifier output, we use Integrated Gradients (IG) [21] to explain the classification decision through the complete encoder–classifier pipeline. IG is computed for the logit margin between the predicted and runner-up classes by integrating its gradients over inputs linearly interpolated between a reference image corresponding to the ImageNet channel-wise mean color and the actual input image. The resulting signed pixel-wise attributions are converted to a non-negative support map by retaining positive values and summing across the RGB channels. To reduce noise, we apply SmoothGrad-IG [22] by averaging raw signed attributions from 12 noisy realizations of the input. Finally, the support values are averaged within each $1 6 \times 1 6$ patch to form a 14 × 14 patch-level map (igpatch), which is upsampled to $2 2 4 \times 2 2 4$ , masked to the valid retinal field of view, and normalized within this region.

## Comparison

Fig. 2 compares the RETFound patch-level IG map (igpatch) with the proposed ring-based explanation for a representative healthy example. In this example the RETFound attribution map assigns positive attribution to broad image regions that are not closely aligned with the retinal vasculature, including the darker macular region around the fovea. This qualitative pattern is consistent with the controlled background-appearance experiments in Supplementary Section S8, where perturbing non-vessel background appearance reduced RETFound performance. These results suggest that RETFound is sensitive to non-vascular image cues under the tested setting, although the attribution map itself should be interpreted as qualitative evidence.

![](images/723befdb1bee397c59d09f6e2f05677f571816762168c13dede9d8a7f55cedaa.jpg)  
(a) Original healthy fundus image.

![](images/f070658e8b2d36cefbbb8a6bdb50d3e5e19992fd0a6cc0beff83912f3de174ec.jpg)  
(b) RETFound explanation map using patch-level Integrated Gradients.

![](images/c90eeb0d19175abdb85e00e0446fb67f276e6229a5c68e92883cb7015c00af21.jpg)  
(c) Proposed ring-based evidence heatmap.

![](images/343ae16db3f5fc3c1173d7dc78b2a9de177782a2fb92b6f795f8da1c26cdb792.jpg)  
(d) Top ring-wise feature evidence.  
Fig. 2 Qualitative and quantitative comparison of explanation maps for a representative healthy fundus image. Both methods correctly classify the image as healthy. The RETFound map shows patch-level positive Integrated Gradients attributions for the predicted-versus-runner-up logit margin. Warmer colors indicate stronger positive contributions to the model’s preference for the predicted class over the runner-up class. The proposed heatmap shows signed evidence for the predicted class relative to the runner-up class (∆ = Predicted − Runner-up), aggregated by ring and visualized on the retinal vasculature. Panel (d) summarizes the top ring-wise feature contributions supporting the predicted class over the runner-up class.

In contrast, the proposed explanation is explicitly grounded in annular vascular regions and ring-wise vascular descriptors. The heatmap identifies which annular regions provide positive evidence for the predicted class relative to the runner-up class, while the feature-evidence panel summarizes the top contributing descriptors. This provides a quantitative link between the model decision and measurable retinal vascular phenotypes, such as vessel curvature, regional vessel area fraction, caliber-related descriptors, branching density, and vascular connectivity. The feature-level attribution context analysis in Supplementary Section S11, Feature-Level Reference Context for Ring-Wise Evidence, further compares the raw values of the top evidence features with predicted-class and runner-up-class reference distributions. This provides dataset-level context for how these model-based vascular descriptors difer between the predicted and competing classes. These features should therefore be interpreted as model-based vascular evidence rather than standalone clinical diagnostic signs. Systematic clinician evaluation of explanation quality remains an important direction for future work.

![](images/705575b70a88df2b9e0894bd4456d5da8bfba235a0be281ae69adb2c3906dbe0.jpg)  
Fig. 3 Representative image-level examples linking selected ring-wise vascular descriptors to clinically relevant retinal vasculature abnormalities. Cyan indicates the vessel mask, and the yellow band indicates the highlighted optic-disc-centered annulus. Each row compares a disease example with a healthy reference image for the same feature and annular region. The examples show reduced vessel area fraction in glaucoma, reduced artery–vein caliber ratio in diabetic retinopathy, and increased caliber variability in diabetic retinopathy. Feature values are compared with the healthy reference distribution for the same ring and descriptor.

To further connect the extracted ring-wise descriptors with clinically relevant retinal vascular changes, we provide representative image-level examples in Fig. 3. Each example shows a healthy reference image and a disease image with the vessel mask overlaid and the corresponding optic-disc-centered annulus highlighted. The selected feature value from the disease image is compared with the healthy reference distribution for the same descriptor and annular region. The examples include reduced vessel area fraction in glaucoma, reduced artery–vein caliber ratio in diabetic retinopathy, and increased caliber variability in diabetic retinopathy. These visualizations show how localized vascular deviations from healthy reference patterns can be inspected directly within the proposed ring-wise framework. The example-level interpretations are broadly consistent with prior retinal vascular studies summarized in Supplementary Section S16, Related Work, including reports of reduced vessel density in glaucoma and altered vessel caliber or caliber heterogeneity in diabetic retinopathy.

Table 2 Comparison of the proposed ring-based method using vessel masks obtained from three segmentation algorithms: FR-UNet [23], SA-UNetv2 [24], and RIP-AV [18]. Results are reported as mean accuracy, macro $\mathrm { F 1 , }$ , and 95% confidence intervals.
<table><tr><td rowspan="2">Segment. Method</td><td colspan="3">HRF</td><td colspan="3">FIVES</td><td colspan="3">SUSTech-SYSU</td></tr><tr><td> $\operatorname { A c c } .$ </td><td>F1</td><td>95%CI</td><td> $\operatorname { A c c } .$ </td><td>F1</td><td>95%CI</td><td> $\operatorname { A c c } .$ </td><td>F1</td><td>95% CI</td></tr><tr><td>FR-UNet</td><td>0.844</td><td>0.844</td><td>[0.712, 0.923]</td><td>0.685</td><td>0.653</td><td>[0.637, 0.730]</td><td>0.896</td><td>0.888</td><td>[0.876, 0.914]</td></tr><tr><td>SA-UNetv2</td><td>0.800</td><td>0.800</td><td>[0.662, 0.891]</td><td>0.722</td><td>0.695</td><td>[0.675, 0.764]</td><td>0.893</td><td>0.884</td><td>[0.872, 0.910]</td></tr><tr><td>RIP-AV</td><td>0.911</td><td>0.909</td><td>[0.793, 0.965]</td><td>0.724</td><td>0.700</td><td>[0.677, 0.766]</td><td>0.896</td><td>0.888</td><td>[0.875, 0.913]</td></tr></table>

## 2.5 Comparison Across Diferent Segmentation Algorithms

To assess the sensitivity of the proposed ring-based classification framework to the choice of vessel segmentation algorithm, we compared three segmentation methods: RIP-AV [18], FR-UNet [23], and SA-UNetv2 [24]. The detailed segmentation implementation is described in Supplementary Section S13 Feature-Family Importance Analysis. Table 2 summarizes the classification results obtained using vessel masks generated by the three segmentation algorithms. RIP-AV achieved the highest accuracy and macro F1 on HRF, while on FIVES it performed similarly to SA-UNetv2 and better than FR-UNet. All three algorithms yielded similar results on SUSTech-SYSU.

Supplementary Figures S23–S26 show the feature-importance rankings for feature groups aggregated across all annular rings under the three automatic segmentation settings and, where available, the corresponding ground-truth vessel mask settings. Across the segmentation-based settings, appearance-related feature groups, including vessel color, $\mathrm { S O _ { 2 } }$ proxies, and vessel–background entropy, were generally more important than geometric feature groups. In contrast, on the high-quality HRF dataset with ground-truth vessel masks, the top-ranked features included a mixture of geometric and appearance-related features. On the lower-quality FIVES dataset, however, the highest-ranked features remained predominantly appearance-related even when ground-truth masks were used. This pattern suggests that, when image quality is lower or vessel delineation is less reliable, appearance-related cues may remain more stable than fine-grained geometric measurements, even with expert vessel annotations. Overall, these findings suggest that the performance of the proposed framework depends in part on vessel-mask quality and segmentation characteristics, underscoring the importance of accurate vascular delineation for downstream ring-based classification. They also suggest that improved segmentation of fine vasculature may further enhance downstream classification performance.

## 2.6 Ablation Study

Using vessel masks predicted by RIP-AV on HRF, the complete feature representation achieved an accuracy of 0.911. Removing the $\mathrm { S O } _ { 2 } .$ -proxy features produced the largest performance decrease, reducing accuracy to 0.778. Removing the vessel–background entropy features reduced accuracy to 0.800, whereas removing either the geometry or red–green color features reduced accuracy to 0.844.

This pattern difers from the standalone feature-family analysis in Supplementary Fig. S27, in which the $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ features achieved lower individual classification performance than the geometry and red–green color features. The two findings are nevertheless consistent because they measure diferent aspects of feature utility. Standalone evaluation measures the discriminative value of a feature family in isolation, whereas ablation analysis measures its incremental contribution when combined with the remaining feature families. The results therefore indicate that the $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ features provide complementary information that is not fully captured by the other feature families. Additional $\mathrm { S O _ { 2 ^ { - } P r o x y } }$ ablations across datasets are reported in Supplementary Section S3.4.

## 2.7 Cross-Dataset Validation

Cross-dataset validation is challenging because retinal datasets may difer in imaging device, color response, field of view, resolution, preprocessing, and class composition. Such domain shifts can reduce external performance even when a model performs well within its source dataset. To evaluate external generalization, we trained the models on FIVES and evaluated them directly on HRF for three-class classification. No HRF images were used for model fitting, feature standardization, or model selection. For the proposed method, ground-truth vessel masks were used in both datasets to assess feature transfer independently of vessel-segmentation errors. Annular regions were constructed using a width of $0 . 5 \times \mathrm { D D }$ and eight rings.

To reduce diferences in spatial scale, each image was resized so that the optic disc had the same diameter in pixels while preserving the original FOV layout. Because RGB-derived descriptors may be sensitive to color-domain shift, we evaluated two configurations of the proposed representation. The first excluded both red–green color and $\mathrm { S O _ { 2 } }$ -proxy features. The second retained the same structural descriptors and additionally included a within-image normalized $\mathrm { S O _ { 2 ^ { - } P r o x y } }$ descriptor. For each image, the median $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ value was first computed within each ring. The resulting ring-wise values were then standardized by subtracting their within-image mean and dividing by their within-image standard deviation. This normalization reduces image-level ofsets while preserving relative variation across retinal eccentricity.

Table 3 summarizes the $\mathrm { F I V E S - t o - H R F }$ external validation results. The pretrained image-based baselines showed limited transfer, with accuracies ranging from 0.333 to 0.356. This finding should be interpreted within the present experimental setting and does not imply that pretrained retinal or natural-image encoders are intrinsically unsuitable for cross-domain retinal analysis. Rather, it indicates that frozen whole-image embeddings combined with a shallow classifier did not transfer efectively between these two datasets, potentially because of diferences in image acquisition, color response, FOV geometry, resolution, preprocessing, or dataset composition.

The proposed ring-based representation showed stronger external performance. After excluding red–green color and $\mathrm { S O } _ { 2 } .$ -proxy features, it achieved an accuracy of 0.733, a macro-F1 of 0.731, and a macro $\mathrm { R O C - A U C }$ of 0.844. Adding the withinimage normalized $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ descriptor increased accuracy to 0.778 and macro-F1 to

0.780, with a macro ROC-AUC of 0.849. These results suggest that disc-normalized ring-wise vascular descriptors captured structural information that transferred more efectively in this specific FIVES-to-HRF setting, while relative ring-wise $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ patterns provided additional complementary information.

The results also support a cautious interpretation of RGB-derived appearance descriptors. As shown in Supplementary Section S3.2, Dataset-Dependent Distribution of RGB Color and $S O _ { 2 } { - } P r o x y$ Features, raw red–green color and absolute $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ descriptors exhibit clear dataset-level shifts between FIVES and HRF. Raw color descriptors were therefore excluded from the cross-dataset configurations, and the $\mathrm { S O } _ { 2 } .$ -proxy was included only after within-image normalization. Although this normalization reduces image-level ofsets, the resulting descriptor remains derived from RGB intensities and may still be afected by camera response, white balance, pigmentation, and preprocessing. Future work should investigate calibrated color harmonization, domain adaptation, fine-tuning of pretrained encoders, and validation across larger multi-device datasets.

Table 3 FIVES-to-HRF external validation. Models were trained on FIVES and evaluated directly on HRF. Ground-truth vessel masks were used for the proposed method in both datasets to assess feature transfer independently of vessel-segmentation errors.
<table><tr><td>Model</td><td>Acc.</td><td>Macro-F1</td><td>ROC-AUC</td><td>95% CI  $( \mathbf { A c c . } )$ </td></tr><tr><td>ResNet-50</td><td>0.356</td><td>0.248</td><td>0.637</td><td>[0.232, 0.502]</td></tr><tr><td>ViT-B/16</td><td>0.333</td><td>0.167</td><td>0.694</td><td>[0.214, 0.479]</td></tr><tr><td>ConvNeXt-B</td><td>0.356</td><td>0.211</td><td>0.596</td><td>[0.232, 0.502]</td></tr><tr><td>RETFound</td><td>0.333</td><td>0.167</td><td>0.524</td><td>[0.214, 0.479]</td></tr><tr><td>Proposed without RGB color and SO₂-proxy features</td><td>0.733</td><td>0.731</td><td>0.844</td><td>[0.590, 0.840]</td></tr><tr><td>Proposed with normalized  $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$  descriptor</td><td>0.778</td><td>0.780</td><td>0.849</td><td>[0.637, 0.875]</td></tr></table>

## 3 Discussion

This study introduces an interpretable ring-based vascular representation for fundus image classification. The framework organizes vessel geometry, vessel color, oxygenation-related appearance, and vessel–background entropy within anatomically defined annular regions centered on the optic disc. Across HRF, FIVES, and SUSTech-SYSU, these descriptors provided useful classification information while linking model predictions to measurable retinal vascular characteristics. The results also clarify how segmentation quality, image appearance, and dataset shift afect the representation.

Segmentation quality afected downstream classification diferently across datasets. On HRF, accuracy ranged from 0.800 to 0.911 with automatic masks, whereas accuracy reached 1.000 with expert masks. By contrast, the three segmentation algorithms produced similar results on SUSTech-SYSU. Because the datasets difer in sample size, image characteristics, disease composition, and classification setting, the smaller performance variation observed on SUSTech-SYSU cannot be attributed to sample size alone. Nevertheless, the HRF results indicate that more accurate vascular delineation can improve downstream ring-based measurements and classification.

Ablation analysis further showed that the feature families provide complementary rather than interchangeable information. Removing the SO -proxy features produced the largest decrease in HRF accuracy, from 0.911 to 0.778, despite their weaker standalone performance than that of the geometry feature group. Standalone analysis evaluates a feature family in isolation, whereas ablation evaluates the information lost when that family is removed from the complete representation. The results therefore suggest that the SO -proxy descriptors capture variation not fully represented by geometry, vessel color, or vessel–background entropy. Removing vessel–background entropy also reduced performance, indicating that perivascular intensity and contrast diferences contribute additional information. These findings support a multifamily representation.

Cross-dataset validation from FIVES to HRF provided preliminary evidence that anatomically normalized ring-wise descriptors may retain useful information under dataset shift. The frozen whole-image embedding baselines showed limited direct transfer, whereas the proposed representation achieved higher accuracy in this cross-dataset evaluation after raw color features were excluded. Adding a within-image-normalized SO -proxy descriptor improved accuracy from 0.733 to 0.778. This suggests that relative ring-wise attenuation patterns may be more transferable than absolute RGB-derived measurements when image-level color ofsets are reduced. Optic-disc based spatial normalization may also contribute by expressing vascular measurements relative to an anatomical reference.

However, these cross-dataset findings should be interpreted cautiously. The limited transferability of the frozen pretrained embeddings does not imply that pretrained image models are intrinsically unsuitable for cross-dataset retinal image analysis. Their performance may be improved through multi-device training or alternative classification heads. Similarly, the higher accuracy of the ring-based representation in this evaluation does not establish general superiority across institutions or imaging devices. Rather, anatomical normalization and selective exclusion or normalization of color-sensitive features may help reduce certain forms of dataset shift.

Field-of-view standardization and non-vessel-background perturbation experiments suggested that whole-image models may use information not directly related to retinal vascular pathology. Changes in RETFound performance after field-of-view and background standardization do not imply that the model is clinically unreliable. Rather, under the evaluated FIVES setting, predictions were partly associated with dataset-specific spatial or appearance characteristics, possibly because acquisition properties were correlated with class labels. Explicit vascular descriptors provide greater transparency because the measurements used by the classifier can be inspected directly, although robustness across acquisition settings requires further validation.

Several limitations should be acknowledged. First, the framework depends on reliable vessel segmentation, optic-disc localization, and stable color and intensity measurements. This dependence is particularly important for images afected by uneven illumination, blur, media opacity, compression artifacts, low vascular contrast, or severe pathology. Image-enhancement and contrast-normalization methods may improve vessel visibility, but they may also alter the color and optical-density relationships used by the appearance-based descriptors. Their efects should therefore be evaluated at both the segmentation and feature-extraction stages.

Second, the evaluation was based on publicly available datasets containing primarily healthy, diabetic retinopathy, and glaucoma images. These datasets do not provide harmonized information about camera systems, acquisition protocols, screening environments, ethnicity, fundus pigmentation, or relevant clinical covariates. Moreover, the FIVES analysis used images selected according to the available image-quality labels. The present results therefore do not fully establish robustness across institutions, populations, imaging devices, or routine screening conditions. Validation using larger multi-institutional datasets with broader demographic and acquisition metadata, additional retinal diseases, and prospective clinical cohorts will be necessary.

Third, the $\mathrm { S O } _ { 2 } .$ -proxy descriptors are not calibrated measurements of retinal oxygen saturation. They are relative appearance features derived from RGB fundus images, whose broadband channels are afected by camera spectral sensitivity, illumination spectra, pigmentation, compression, and proprietary image-processing pipelines. These descriptors should therefore be interpreted as oxygenation-related optical-density proxies rather than direct physiological measurements. Their complementary classification value is encouraging, but physiological interpretation will require validation using better-characterized spectral imaging systems or calibrated multispectral and dual-wavelength measurements.

Finally, the current implementation was designed for reproducible analysis rather than real-time deployment. Most of the computational cost arises from CPU-based ring-wise feature extraction, whereas logistic-regression training and prediction are lightweight. Future work will focus on accelerating and integrating the processing pipeline through vectorized and parallel feature computation, GPU acceleration, and automated quality control. Further directions include uncertainty-aware vessel segmentation, identification of low-confidence rings, device-aware color harmonization, broader external validation, and integration of explicit vascular descriptors with deep image embeddings.

Overall, this study demonstrates that anatomically organized ring-wise vascular descriptors can provide an interpretable and complementary representation for retinal fundus image classification. By making the underlying geometric and appearancerelated measurements directly accessible for inspection, the framework supports more transparent retinal-image analysis than classification outputs alone. Although broader validation across institutions, imaging devices, populations, and clinical settings is required, the results support the potential of ring-structured vascular measurements for retinal phenotype characterization, disease classification, and future investigations of associations with clinical outcomes.

## 4 Method

## 4.1 Datasets

We evaluated the proposed method on three publicly available retinal fundus image datasets with difering image characteristics, disease categories, and availability of expert vessel annotations to assess its robustness across dataset settings.

The High-Resolution Fundus (HRF) dataset [25, 26] contains $4 5$ color fundus images, including 15 healthy, 15 diabetic retinopathy (DR), and 15 glaucoma images. Each image has a resolution of $3 5 0 4 \times 2 3 3 6$ pixels and is accompanied by an expert-annotated vessel mask and a FOV mask.

The FIVES dataset [27] contains 800 fundus images with a resolution of 2048×2048 pixels, manual pixel-level vessel annotations, and image-quality labels for illumination or color distortion, blur, and low contrast. Based on these labels, we selected highquality images from three classes: healthy $( n = 1 6 8 )$ ), glaucoma $( n = 8 3 )$ , and DR $( n = 1 3 0 )$

The SUSTech-SYSU dataset [28] contains high-resolution color fundus images with a resolution of $2 8 8 0 \times 2 1 3 6$ pixels and was collected for DR analysis. We used 1,016 images, including 385 DR and 631 healthy images, to evaluate binary DR-versushealthy classification.

## 4.2 Framework Overview

The proposed framework consists of ring-based retinal vascular feature extraction followed by disease classification. Starting from binary vessel masks obtained from expert annotations or existing segmentation methods, we extract geometry, red–green color, and optical-density-derived $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ features from the segmented vasculature. Vessel–background entropy features additionally incorporate information from nearby perivascular pixels.

To preserve spatial variation across retinal eccentricity, the FOV is partitioned into concentric annular rings centered on the optic disc. Because the major retinal vessels emerge from the optic nerve head and branch outward across the retina [29], the optic disc provides an anatomically interpretable reference for organizing vascular measurements. Disc-centered regions are also commonly used for vessel-caliber and peripapillary vascular assessment [30, 31]. Unless otherwise specified, each ring has a radial width of half an optic disc diameter (0.5×DD). Using DD as the radial unit provides an image-relative anatomical scale that reduces dependence on absolute image resolution and scale. This setting balances spatial specificity with suficient vessel support for stable feature estimation. Classification performance remained stable across alternative ring widths, as detailed in Supplementary Section S1.

Within each ring, we compute four complementary feature families: vessel geometry, OD-derived $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ appearance, red–green vessel color, and vessel–background entropy. The ring-wise descriptors are concatenated and standardized to account for diferences in scale across feature types. The resulting feature vector is classified using logistic regression. Together, the anatomically organized representation and linear classifier permit direct examination of feature-level and annular-region contributions to individual predictions. Fig. 4 summarizes the processing pipeline.

![](images/aeb85d8e39eea64d55d0ab29f0137f0c2c60d509fd7f5bbcf9e51a332b52c5f2.jpg)  
Fig. 4 Overview of the proposed ring-based classification framework. Concentric annular regions are defined around the optic disc, and vessel geometry, red–green color, OD-derived $\mathrm { S O } _ { 2 } .$ -proxy, and vessel–background entropy features are extracted within each ring. The resulting descriptors are concatenated, standardized, and classified using logistic regression.

## 4.3 Ring-Based Features

The ring-based representation captures complementary aspects of vascular geometry, oxygenation-related appearance, red–green color, and vessel–background texture across retinal eccentricity. Pixel- and segment-level measurements within each ring are primarily summarized using the median and interquartile range (IQR), where $\mathrm { I Q R } = Q _ { 3 } - Q .$ <sub>1</sub> and $Q _ { 1 }$ and $Q _ { 3 }$ denote the 25th and 75th percentiles, respectively. The median represents the typical within-ring value, whereas the IQR quantifies spatial heterogeneity and is relatively robust to skewed distributions, segmentation noise, spurious vessel fragments, and focal atypical regions. Artery–vein contrast features are expressed as ratios of the corresponding median arterial and venous measurements. Inherently aggregated quantities, including vessel densities, counts, lengths, and graph statistics, are retained in their natural ring-level form. This spatial organization preserves regional variation between the peripapillary and more peripheral retinal vasculature. Supplementary Table S4 summarizes the extracted features, their ring-wise statistics, and the rationale for each choice.

## 4.3.1 Geometry-Based Vessel Features

## Vessel Density and Branching

Vessel area fraction and vessel length density are established descriptors of retinal vascular structure [32–34]. In this study, we compute these descriptors within each annular region and additionally quantify branch-point density. Given an annular ring mask A, a binary vessel mask V, its one-pixel-wide skeleton S, and a branch-point mask J, vessel area fraction (VAF), vessel length density (VLD), and branch-point

density (BPD) are defined as

$$
{ \mathrm { V A F } } = { \frac { | A \cap V | } { | A | } } , \qquad { \mathrm { V L D } } = { \frac { | A \cap S | } { | A | } } , \qquad { \mathrm { B P D } } = { \frac { | A \cap J | } { | A | } } ,\tag{1}
$$

where | · | denotes the number of pixels and ∩ denotes set intersection. The skeleton S is obtained from V through morphological thinning to a one-pixel-wide centerline.

Because some datasets provide only a combined vessel mask without artery–vein labels, branch points are extracted directly from the complete skeletonized vascular network, and artery–vein crossings are not removed. Accordingly, BPD represents the junction density of the projected vascular network rather than a count of anatomically verified bifurcations. The branch-point mask J is constructed by identifying skeleton pixels with more than two 8-connected neighbors (i.e., more than two neighbouring skeleton pixels among the eight pixels sharing an edge or corner with the current pixel), followed by removal of connected branch-point components containing fewer than three pixels to suppress spurious detections. A representative example is shown in Fig. 5(a).

## Vessel Caliber and Related Thickness Features

Vessel caliber. Retinal vessel caliber is commonly estimated from segmented vessel boundaries or centerline-to-boundary distances [11, 33, 35]. In this study, for each skeleton pixel $p \in S _ { A }$ , where $S _ { A } = A \cap S$ denotes the set of vessel-centerline pixels within annular region A, the local vessel diameter is estimated from the Euclidean distance transform (EDT) of the binary vessel mask:

$$
d ( p ) = 2 \operatorname { E D T } ( p ) ,\tag{2}
$$

where $\operatorname { E D T } ( p )$ is the Euclidean distance, in pixels, from $p$ to the nearest vessel boundary. Multiplying this distance by two provides an approximation of the local vessel diameter under the assumption that the skeleton pixel lies near the vessel centerline. Within each ring, vessel-caliber variability is summarized by the IQR of the local diameter values $d ( p )$ . A representative visualization of this estimation is shown in Fig. 5(b).

Artery–vein ratio. The artery–vein ratio is an established caliber-based descriptor of the relative widths of retinal arteries and veins [11, 33, 36]. When artery and vein labels are available, the same local diameter estimates are used to calculate arteryand vein-specific median calibers. Let $S _ { A } ^ { \mathrm { a r t } }$ and $S _ { A } ^ { \mathrm { v e i n } }$ denote the arterial and venous skeleton pixels within ring A, respectively. The artery–vein ratio (AVR) is defined as

$$
\mathrm { A V R } = \frac { \mathrm { m e d i a n } ( d ( p ) \mid p \in S _ { A } ^ { \mathrm { a r t } } ) } { \mathrm { m e d i a n } \big ( d ( p ) \mid p \in S _ { A } ^ { \mathrm { v e i n } } \big ) } .\tag{3}
$$

This ratio characterizes the typical arterial caliber relative to the typical venous caliber within the same annular region. Median diameters are used to reduce sensitivity to local segmentation errors and atypical vessel segments.

Thin-vessel length. When artery–vein labels are unavailable, we define two discnormalized descriptors of relative vessel thickness: thin-vessel length and the thick-tothin diameter ratio. Thin skeleton pixels are defined as

$$
S _ { A } ^ { \mathrm { t h i n } } = \left\{ p \in S _ { A } : d ( p ) < \tau \right\} , \qquad \tau = \alpha D D ,\tag{4}
$$

where DD is the optic-disc diameter in pixels and α is a fixed proportional threshold. The thin-vessel length is then calculated as

$$
L _ { \mathrm { t h i n } } = \left| S _ { A } ^ { \mathrm { t h i n } } \right| ,\tag{5}
$$

where |·| denotes set cardinality. Because the skeleton is one pixel wide, $L _ { \mathrm { t h i n } }$ represents the total thin-vessel centerline length within the ring in pixel units.

We set $\alpha = 0 . 0 1 8$ , corresponding to a diameter threshold of approximately 1.8% of the optic-disc diameter. This value was selected empirically from the observed vessel-caliber distribution so that the threshold lies near the upper boundary of the small-caliber range. Scaling the threshold by $D D$ reduces its dependence on absolute image resolution and fundus scale. The resulting feature provides a ring-wise surrogate measure of small-caliber vessel content.

Thick-to-thin diameter ratio. The complementary set of thick-vessel skeleton pixels is defined as

$$
S _ { A } ^ { \mathrm { t h i c k } } = S _ { A } \setminus S _ { A } ^ { \mathrm { t h i n } } .\tag{6}
$$

The thick-to-thin diameter ratio is then calculated as

$$
D _ { \mathrm { r a t i o } } = { \frac { \mathrm { m e d i a n } { \big ( } d ( p ) \mid p \in S _ { A } ^ { \mathrm { t h i c k } } { \big ) } } { \mathrm { m e d i a n } { \big ( } d ( p ) \mid p \in S _ { A } ^ { \mathrm { t h i n } } { \big ) } } } .\tag{7}
$$

This feature quantifies the contrast between larger- and smaller-caliber vessels within the same ring. Together, $L _ { \mathrm { t h i n } }$ and $D _ { \mathrm { r a t i o } }$ provide artery–vein-label-free measures of small-vessel content and within-ring caliber heterogeneity. They should not be interpreted as direct substitutes for the anatomically defined AVR because the thick and thin groups do not necessarily correspond exactly to veins and arteries.

## Vessel Tortuosity and Curvature

Vessel tortuosity. Vessel tortuosity is quantified using the arc-to-chord ratio, a commonly used centerline-based measure of the deviation of a vessel segment from a straight path [35, 37, 38]. For a segment represented by an ordered sequence of centerline points $\mathbf { p } _ { i } = ( x _ { i } , y _ { i } ) , i = 1 , \ldots , N$ , the cumulative arc length is defined as

$$
\ell _ { 1 } = 0 , \qquad \ell _ { i } = \sum _ { k = 1 } ^ { i - 1 } \left\| \mathbf { p } _ { k + 1 } - \mathbf { p } _ { k } \right\| , \quad i = 2 , \ldots , N ,\tag{8}
$$

where $\| \cdot \|$ denotes the Euclidean norm. The total arc length is $L = \ell _ { N }$ , and the chord length is $C = \| \mathbf { p } _ { N } - \mathbf { p } _ { 1 } \|$ . For segments with $C > 0$ , tortuosity is defined as

$$
T = { \frac { L } { C } } \geq 1 .\tag{9}
$$

A value of $T = 1$ corresponds to a straight segment, whereas larger values indicate greater deviation from the direct path between the segment endpoints. Within each annular region A, variability in segment tortuosity is summarized by the IQR.

Curvature. Whereas the arc-to-chord ratio characterizes the overall shape of a vessel segment, curvature quantifies local bending along the centerline [35, 38]. In this study, the tangent orientation at an interior point $\mathbf { p } _ { i }$ is approximated using a centered diference:

$$
\theta _ { i } = \operatorname { a t a n 2 } ( y _ { i + 1 } - y _ { i - 1 } , x _ { i + 1 } - x _ { i - 1 } ) .\tag{10}
$$

The magnitude of the local angular change per unit arc length is then estimated as

$$
\kappa _ { \theta } ( \ell _ { i } ) = \left| \frac { \Delta \theta _ { i } } { \ell _ { i + 1 } - \ell _ { i - 1 } } \right| , \qquad \Delta \theta _ { i } = \mathrm { w r a p } ( \theta _ { i + 1 } - \theta _ { i - 1 } ) ,\tag{11}
$$

where wrap(·) maps angular diferences to $( - \pi , \pi ]$ , preventing artificial discontinuities when the orientation crosses the $\pm \pi$ boundary. Because arc length is measured in pixels, $\kappa _ { \theta }$ is expressed in inverse-pixel units. Within each ring, typical local vessel bending is summarized by the median curvature.

## Vascular Connectivity Features

Within each annular region A, the vessel skeleton is represented as an undirected graph $G _ { A } = ( N _ { A } , E _ { A } )$ , where nodes correspond to skeleton pixels and edges connect pairs of 8-neighboring skeleton pixels. From this graph, we extract two connectivity descriptors: the vascular connection count, defined as the graph edge count $\left| E _ { A } \right|$ , and the mean vascular connectivity,

$$
\mathrm { V C } _ { \mathrm { m e a n } } = { \frac { 1 } { | N _ { A } | } } \sum _ { n \in N _ { A } } \deg ( n ) = { \frac { 2 | E _ { A } | } { | N _ { A } | } } ,\tag{12}
$$

where $N _ { A }$ and $E _ { A }$ denote the node and edge sets of the ring-restricted graph, respectively, and deg(n) is the number of graph edges incident to node n. The vascular connection count reflects the total number of pixel-level skeleton connections within the ring and therefore depends partly on the amount of vasculature present. Mean vascular connectivity normalizes this quantity by graph size and characterizes the average local connectedness of the skeleton.

These graph-based descriptors complement branch-point density (BPD). Whereas BPD measures the frequency of detected junctions relative to ring area, vascular connection count and mean vascular connectivity also describe the organization of neighboring skeleton pixels and may be sensitive to vessel continuity and fragmentation.

## 4.3.2 OD-derived $\mathbf { S O _ { 2 } } – \mathbf { P }$ roxy Features

Prior studies have reported altered retinal oxygenation patterns in DR and glaucoma, including increased venous oxygen saturation in DR and reduced arteriovenous oxygen diference in glaucoma [39, 40]. Motivated by these findings, we include $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ descriptors as one component of our ring-structured vascular representation.

## Optical Density

Our formulation is based on the dual-wavelength retinal oximetry principle, which combines a near-isosbestic green wavelength with an oxygen-sensitive red wavelength [41–43]. Under a Beer–Lambert approximation, the optical density (OD) at wavelength $\lambda$ is

$$
O D ( \lambda ) = - \ln \left( \frac { I _ { \mathrm { i n } } ( \lambda ) } { I _ { \mathrm { o u t } } ( \lambda ) } \right) \approx \varepsilon ( \lambda ) h l ,\tag{13}
$$

where $I _ { \mathrm { o u t } } ( \lambda )$ and $I _ { \mathrm { i n } } ( \lambda )$ denote the outside- and inside-vessel intensities, respectively, $\varepsilon ( \lambda )$ is an efective haemoglobin extinction coeficient, h is an efective haemoglobin concentration term, and l is the efective optical path length. To reduce the influence of pigmentation and background reflectance, we use a local vessel-free reference estimated at the same spatial location as each vessel pixel.

In retinal vessels, haemoglobin is the dominant chromophore, and the OD spectrum can be approximated as the sum of oxy- and deoxyhaemoglobin contributions:

$$
O D ( \lambda ) \approx \varepsilon _ { \mathrm { H b O } _ { 2 } } ( \lambda ) h _ { \mathrm { H b O } _ { 2 } } + \varepsilon _ { \mathrm { H b } } ( \lambda ) h _ { \mathrm { H b } } ,\tag{14}
$$

where $h _ { \mathrm { { H b O _ { 2 } } } }$ and $h _ { \mathrm { H b } }$ are efective nonnegative coeficients that absorb the approximately wavelength-invariant path-length factor for a fixed vessel geometry. Green wavelengths near an isosbestic point are only weakly dependent on oxygen saturation, whereas red wavelengths provide stronger oxygen-sensitive contrast, motivating our RGB-based approximation.

## RGB Optical Density Estimation and SO<sub>2</sub>-Proxy Inversion

Because RGB cameras record band-integrated rather than monochromatic intensities, we model each channel $c ~ \in ~ \{ R , G , B \}$ over nominal wavelength supports $\Lambda _ { B } = [ 4 4 3 , 4 8 8 ]$ nm, $\Lambda _ { G } = [ 5 3 5 , 5 5 5 ]$ nm, and $\Lambda _ { R } = [ 6 0 0 , 6 2 0 ]$ nm, following prior retinal oximetry practice [44–46], with each support discretized at 1-nm intervals. The observed intensity in channel c at location $p$ is

$$
\bar { I } _ { c } ( p ) = \sum _ { \lambda \in \Lambda _ { c } } \Phi _ { c } ( \lambda ) I ( \lambda , p ) ,\tag{15}
$$

where $\Phi _ { c } ( \lambda )$ denotes the efective spectral sensitivity of channel c and $I ( \lambda , p )$ is the reflected-light spectral signal reaching the sensor. Applying (15) to the vessel-free

reference and vessel signal gives

$$
\begin{array} { l } { { \displaystyle \bar { I } _ { \mathrm { o u t } , c } ( p ) \triangleq \sum _ { \lambda \in \Lambda _ { c } } \Phi _ { c } ( \lambda ) I _ { \mathrm { o u t } } ( \lambda , p ) , } } \\ { { \displaystyle \bar { I } _ { \mathrm { i n } , c } ( p ) \triangleq \sum _ { \lambda \in \Lambda _ { c } } \Phi _ { c } ( \lambda ) I _ { \mathrm { i n } } ( \lambda , p ) . } } \end{array}\tag{16}
$$

Under the Beer–Lambert approximation (13), the channel-wise OD can be written as

$$
O D _ { c } ( p ) \approx - \ln \left( \frac { \bar { I } _ { \mathrm { i n } , c } ( p ) } { \bar { I } _ { \mathrm { o u t } , c } ( p ) } \right) \approx \bar { \varepsilon } _ { c } h ( p ) l ( p ) ,\tag{17}
$$

where $\bar { \varepsilon } _ { c }$ is an efective channel-averaged extinction coeficient. Because $\bar { I } _ { \mathrm { i n } , c } ( p )$ and $\bar { I } _ { \mathrm { o u t } , c } ( p )$ are measured under the same local illumination and camera settings, their ratio is less sensitive to multiplicative changes in brightness and smooth shading, although this may be degraded by nonlinear image processing, clipping, or wavelengthdependent system efects. This expression is a compact lumped approximation of haemoglobin attenuation over the band $\Lambda _ { c } .$ To make the channel-wise band averaging explicit, we introduce the corresponding local normalized within-band weighting, $\begin{array} { r } { \bar { W _ { c } } ( \lambda ; p ) = \frac { \Phi _ { c } ( \lambda ) I _ { \mathrm { o u t } } ( \lambda , p ) } { \sum _ { \lambda ^ { \prime } \in \Lambda _ { c } } \Phi _ { c } ( \lambda ^ { \prime } ) I _ { \mathrm { o u t } } ( \lambda ^ { \prime } , p ) } } \end{array}$ . Assuming that, after local referencing, the normalized spectral shape of $I _ { \mathrm { o u t } } ( \lambda , p )$ varies only weakly across vessel locations within each channel band, we approximate $W _ { c } ( \lambda ; p )$ by a channel-dependent weighting $W _ { c } ( \lambda )$ This yields $\begin{array} { r } { \bar { \varepsilon } _ { c } \triangleq \sum _ { \lambda \in \Lambda _ { c } } W _ { c } ( \lambda ) \varepsilon ( \lambda ) } \end{array}$ ), where $W _ { c } ( \lambda )$ represents the normalized efective within-band weighting for channel c.

The compact form in (17) serves only as an intuitive lumped approximation. The final OD-derived $\mathrm { S O _ { 2 ^ { - } P r o x y } }$ inversion instead uses the species-resolved version of the same band-averaged model. Under the same within-band weighting,

$$
O D _ { c } ( p ) \approx E _ { c , \mathrm { H b O _ { 2 } } } h _ { \mathrm { H b O _ { 2 } } } ( p ) + E _ { c , \mathrm { H b } } h _ { \mathrm { H b } } ( p ) ,\tag{18}
$$

where, for haemoglobin species $s \in \{ \mathrm { H b O } _ { 2 } , \mathrm { H b } \}$ ,

$$
E _ { c , s } = \sum _ { \lambda \in \Lambda _ { c } } \varepsilon _ { s } ( \lambda ) W _ { c } ( \lambda ) .\tag{19}
$$

Here, $\bar { \varepsilon } _ { c }$ denotes the lumped channel-averaged extinction coeficient in the simplified model, whereas $E _ { c , s }$ denotes the species-resolved channel-averaged extinction coeficient used in the final inversion. Stacking the three RGB channels yields the matrix form

$$
O D ( p ) \approx E \mathbf { h } ( p ) ,\tag{20}
$$

where

$$
\begin{array} { r l } & { O D ( p ) = \big [ O D _ { B } ( p ) , O D _ { G } ( p ) , O D _ { R } ( p ) \big ] ^ { \top } , } \\ & { ~ \mathbf { h } ( p ) = \big [ h _ { \mathrm { H b O _ { 2 } } } ( p ) , h _ { \mathrm { H b } } ( p ) \big ] ^ { \top } . } \end{array}
$$

A detailed derivation of the RGB band-integrated OD model is provided in Supplementary Section S2 Derivation of the RGB Optical-Density Approximation.

Efective within-band weighting estimation. Because camera spectral responses are typically unavailable for public RGB fundus datasets, the efective within-band weighting $W _ { c } ( \lambda )$ is approximated by a Gaussian-shaped band-pass profile:

$$
\begin{array} { c } { { W _ { c } ( \lambda ) = w _ { c } \exp \left( - \displaystyle \frac { ( \lambda - \lambda _ { c } ) ^ { 2 } } { 2 \sigma _ { c } ^ { 2 } } \right) , } } \\ { { w _ { c } ^ { - 1 } = \displaystyle \sum _ { \lambda ^ { \prime } \in \Lambda _ { c } } \exp \left( - \displaystyle \frac { ( \lambda ^ { \prime } - \lambda _ { c } ) ^ { 2 } } { 2 \sigma _ { c } ^ { 2 } } \right) . } } \end{array}\tag{21}
$$

where $\lambda _ { c }$ is the channel center wavelength and $\sigma _ { c }$ controls the bandwidth, with $\mathrm { F W H M } _ { c } = 2 \sqrt { 2 \ln { 2 } } \sigma _ { c }$

In our experiments, we set $\lambda _ { R } = 6 1 0 \mathrm { n m } , \lambda _ { G } = 5 4 5 \mathrm { n m } ,$ and $\lambda _ { B } = 4 6 5 \mathrm { n m }$ . Unless otherwise stated, we use FWHM $_ B \ = \ 4 5$ nm, $\mathrm { F W H M } _ { G } ~ = ~ 2 0 $ nm, and $\mathrm { F W H M } _ { R } ~ { = } ~$ 20 nm. Additional perturbation experiments on the red $\lvert / \mathrm { g }$ reen center wavelengths and channel bandwidths produced similar $\mathrm { S O _ { 2 ^ { - } p r o x y } }$ trends and class separation, suggesting robustness to moderate passband variation.

Vessel-free background reference estimation.

To estimate the vessel-free reference $\bar { I } _ { \mathrm { o u t } , c } ( p )$ at vessel locations, we construct a smooth background map separately for each RGB channel. The vessel mask is first dilated to define an exclusion region around the vasculature. This region is removed and filled using nearby non-vessel pixels with OpenCV’s Navier–Stokes PDE-based inpainting method. Large-scale Gaussian smoothing is then applied to the inpainted image to obtain a smoothly varying vessel-free background estimate. The reference value at each vessel pixel is obtained by sampling this background map at the same spatial location. Further implementation details and illustrative examples are provided in Supplementary Section S4, Construction of the Vessel-Free Background Reference, and Supplementary Figs. S9 and S10.

RGB extinction coeficients. Using Prahl’s tabulated haemoglobin spectra [47], we compute the channel-averaged extinction coeficients for oxyhaemoglobin and deoxyhaemoglobin according to (19). Stacking these coeficients yields the extinction matrix

$$
E = \left[ \begin{array} { l l } { E _ { B , \mathrm { H b O _ { 2 } } } } & { E _ { B , \mathrm { H b } } } \\ { E _ { G , \mathrm { H b O _ { 2 } } } } & { E _ { G , \mathrm { H b } } } \\ { E _ { R , \mathrm { H b O _ { 2 } } } } & { E _ { R , \mathrm { H b } } } \end{array} \right] \in \mathbb { R } ^ { 3 \times 2 } .\tag{22}
$$

In this RGB approximation, the green channel provides a near-isosbestic reference, the red channel provides oxygen-sensitive contrast, and the blue channel contributes additional short-wavelength information.

OD-derived $\mathbf { S } \mathbf { O } _ { 2 ^ { - } } \mathbf { p r o x y }$ estimation. We estimate the efective haemoglobin coeficients using ridge-regularized least squares:

$$
\hat { \mathbf { h } } ( p ) = \left( E ^ { \top } E + \gamma _ { \mathrm { r e g } } \mathbf { I } _ { 2 } \right) ^ { - 1 } E ^ { \top } O D ( p ) ,\tag{23}
$$

where $\mathbf { I } _ { 2 }$ is the $2 \times 2$ identity matrix and $\gamma _ { \mathrm { r e g } } ~ > ~ 0$ is the ridge-regularization parameter. A non-negativity constraint is enforced through element-wise clipping,

$\hat { \mathbf { h } } ( p ) \gets \mathrm { m a x } ( \hat { \mathbf { h } } ( p ) , 0 )$ . The local OD-derived $\mathrm { S O } _ { 2 } .$ -proxy is then defined as

$$
\mathrm { S O } _ { 2 , \mathrm { p r o x y } } ( p ) = \frac { \hat { h } _ { \mathrm { H b O } _ { 2 } } ( p ) } { \hat { h } _ { \mathrm { H b O } _ { 2 } } ( p ) + \hat { h } _ { \mathrm { H b } } ( p ) } .\tag{24}
$$

Because this RGB-based formulation relies on approximate passbands and does not explicitly model scattering or camera-specific image processing, the resulting values are not calibrated physiological oxygen-saturation measurements. We therefore interpret $\mathrm { S O _ { 2 , p r o x y } } ( p )$ as an oxygenation-related appearance descriptor. For each annular ring $A ,$ , the proxy is evaluated on vessel pixels in $A \cap V$ , and its within-ring variability is summarized by the IQR. When artery and vein masks are available, artery–vein contrast is represented by the ratio of the median arterial and venous $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ values within the ring. A representative proxy map is shown in $\mathrm { F i g . \ 5 ( c ) }$

Because the proxy is estimated from RGB intensities rather than calibrated multispectral measurements, we additionally evaluated its sensitivity to controlled global-brightness and channel-wise color-balance perturbations. The perturbation procedures and corresponding results are provided in Supplementary Section S3.3.

## 4.3.3 Red–Green Color Features

For vessel pixels $p \in A \cap V$ , we compute two empirical red–green color descriptors from the original RGB image. The red-to-green ratio, $I _ { R } ( p ) / I _ { G } ( p )$ , is summarized by its within-ring IQR to characterize variability in vascular color. The normalized red–green diference,

$$
\frac { I _ { R } ( p ) - I _ { G } ( p ) } { I _ { R } ( p ) + I _ { G } ( p ) } ,
$$

is summarized by its within-ring median to represent the typical chromatic contrast while reducing sensitivity to outliers. Here, $I _ { R } ( p )$ and $I _ { G } ( \boldsymbol { p } )$ denote the red- and greenchannel intensities at pixel $p ,$ respectively. Unlike the OD-derived $\mathrm { S O _ { 2 ^ { - } P r o x y } }$ features, these empirical descriptors are calculated directly from the original image intensities without background normalization or logarithmic transformation.

## 4.3.4 Vessel–Background Entropy Features

## Fundus Image Preprocessing

To obtain reliable vessel–background texture measurements, entropy features are computed from a photometrically normalized green-channel image, denoted by $I _ { G } ^ { \mathrm { n o r m } }$ . For each RGB channel, a smooth illumination field is estimated using large-scale Gaussian filtering. Ratio-based correction is then performed by dividing the original channel image pixel-wise by its corresponding illumination field, followed by linear rescaling. This procedure reduces low-frequency illumination variation and vignetting while preserving local retinal contrast.

The corrected RGB image is converted to CIE-Lab space, and contrast-limited adaptive histogram equalization (CLAHE) is applied to the luminance channel $L ^ { * }$ After conversion back to RGB, mild unsharp blending is applied to the corrected green channel to obtain $I _ { G } ^ { \mathrm { n o r m } }$ . The green channel is used because it generally provides strong vessel-to-background contrast in color fundus images. The complete preprocessing procedure and parameter settings are provided in Supplementary Section S5, Photometric Normalization and Vessel-Enhancing Preprocessing.

![](images/4c89dbb8357842295fc68fe5deadd36e3b243b6eb315ac5d9613f1478870f6ac.jpg)  
(a) Branch-point extraction.

![](images/03d4d7e527495be5438b1d613ff9ce191c2d2bb7a20fcd195bfe511a5be26510.jpg)  
(b) Vessel-diameter estimation.

![](images/923640b88c27b5906f3f862851f1cc3ab565d9054ab9c9b21bd6f531b8a5eeaa.jpg)  
(c) OD-derived $\mathrm { S O _ { 2 ^ { - } P r o x y } }$ map.  
Fig. 5 Representative ring-wise vascular feature extraction. (a) Vessel branch points extracted from the skeletonized vasculature; yellow dots indicate skeleton pixels with more than two 8-connected neighbors. (b) Local vessel-diameter estimation along representative centerline segments. Red and blue indicate arterial and venous segments, respectively; orange indicates uncertain or unknown vessel labels; transverse markers indicate sampled local diameters. (c) Estimated OD-derived SO<sub>2</sub>-proxy values along the retinal vasculature of a representative healthy image.

## Image Inputs for Diferent Feature Groups

This photometric-normalization pathway is used only for vessel–background entropy measurements. Red–green color features and OD-derived $\mathrm { S O _ { 2 } \mathrm { - p r o x y } }$ features are computed from the original RGB image to preserve the vessel-to-background intensity relationships required by these measurements. Geometry-based features depend only on the vessel masks and skeletons and are therefore independent of image normalization.

## Local Entropy and Ring-Wise Statistics

Local texture complexity is quantified using a Shannon entropy map computed from $I _ { G } ^ { \mathrm { n o r m } }$ . For each pixel $p ,$ entropy is calculated over a disk-shaped neighborhood $\textstyle { \mathcal { N } } _ { r } ( p )$

with radius $r = 5$ pixels:

$$
H ( p ) = - \sum _ { k : \pi _ { k } ( p ) > 0 } \pi _ { k } ( p ) \log _ { 2 } \bigl ( \pi _ { k } ( p ) \bigr ) ,\tag{25}
$$

where $\pi _ { k } ( p )$ denotes the empirical probability of gray level k within $\textstyle { \mathcal { N } } _ { r } ( p )$ . In implementation, $I _ { G } ^ { \mathrm { n o r m } }$ is clipped to $[ 0 , 1 ]$ , rescaled to [0, 255], and converted to an 8-bit image with gray levels $k \in \{ 0 , \ldots , 2 5 5 \}$ . Because a base-2 logarithm is used, entropy is reported in bits.

Within each annular region A, entropy is summarized over two spatial supports: vessel centerline pixels and a near-vessel background band. Let S denote the vessel skeleton. The near-vessel band $B _ { \mathrm { n e a r } }$ contains non-vessel pixels whose distance to the nearest vessel pixel lies between 2 and 6 pixels. The lower distance bound reduces direct vessel-boundary contamination, while the upper bound retains a localized perivascular background region.

The vessel-centerline and near-vessel background entropy features are defined as

$$
H _ { \mathrm { S } } ( A ) = \mathrm { m e d i a n } \left\{ H ( p ) : p \in A \cap S \right\} ,\tag{26}
$$

$$
H _ { \mathrm { B G } } ( A ) = \operatorname { m e d i a n } \left\{ H ( p ) : p \in A \cap B _ { \mathrm { n e a r } } \right\} .\tag{27}
$$

Here, $H _ { \mathrm { S } }$ characterizes local texture complexity along vessel centerlines, whereas $H _ { \mathrm { B G } }$ characterizes texture complexity in the immediately adjacent retinal background. These quantities are included as two separate classifier features; their diference is used only for descriptive interpretation.

A descriptive analysis of the HRF dataset showed diferent vessel–background entropy patterns across classes. Healthy images exhibited a moderate separation between vessel-centerline and near-vessel background entropy, with class-level median values of $H _ { \mathrm { S } } = 3 . 8 4 9$ and $H _ { \mathrm { B G } } = 3 . 7 1 6 .$ corresponding to $\Delta H = 0 . 1 3 3$ . The separation was smaller for glaucoma $( H _ { \mathrm { S } } = 3 . 7 1 7$ $H _ { \mathrm { B G } } = 3 . 6 6 6$ , and $\Delta H = 0 . 0 5 1 )$ and was nearly absent for DR $( H _ { \mathrm { S } } = 3 . 8 7 1$ ， $H _ { \mathrm { B G } } = 3 . 8 7 0$ , and $\Delta H = 0 . 0 0 1 )$ .

The reduced entropy separation in DR may reflect greater heterogeneity in the perivascular background associated with abnormalities such as microaneurysms, hemorrhages, and exudates. The smaller separation in glaucoma may reflect subtler vascular or perivascular appearance changes; however, this interpretation is less direct because glaucoma is primarily associated with alterations in the optic nerve head, retinal nerve fiber layer, and retinal vasculature rather than lesion-driven background abnormalities. These observations are descriptive, because local entropy is nonspecific and may also be afected by image quality and acquisition characteristics.

Representative ring-wise feature distributions across disease groups are shown in Supplementary Figs. S16 and S17, including geometry-based vessel features, vesselappearance features, and vessel–background entropy features.

## 4.4 Implementation and Evaluation Protocol

For the proposed method, vessel masks were generated using RIP-AV [18]. For HRF and FIVES, which provide expert vessel annotations, we additionally evaluated the proposed features using ground-truth vessel masks to quantify the influence of vessel-segmentation quality. Ring-wise vascular descriptors were concatenated to form one feature vector per image. Within each cross-validation fold, the standardization parameters were estimated from the training data and then applied to the held-out data.

The standardized feature vectors were classified using elastic-net-regularized logistic regression. The classifier configuration was kept consistent across datasets, using the saga solver, an elastic-net penalty, an $l _ { 1 }$ ratio of 0.10, a maximum of 3000 iterations, and class-balanced weights to reduce the influence of unequal class sizes. Only the regularization strength was dataset-specific, with $C = 1 . 0$ for HRF and SUSTech-SYSU and C = 0.1 for FIVES.

We compared the proposed ring-based representation with RETFound [14], a retinal foundation model pretrained using a masked-autoencoder objective on approximately 1.6 million unlabeled retinal images. We used the publicly available color-fundus-photography checkpoint. To broaden the comparison, we additionally evaluated three ImageNet-pretrained image-classification backbones: ResNet-50 [15], ViT-B/16 [16], and ConvNeXt-B [17]. For the frozen pretrained models, fundus images were resized to 224×224 pixels and normalized using the ImageNet channel-wise mean and standard deviation. The final image-level representation produced by each encoder was extracted and classified using the same elastic-net logistic-regression framework as the proposed features.

Lightweight fine-tuning was also evaluated within the training portion of each fold. For RETFound and ViT-B/16, the final two transformer blocks and the classification head were made trainable. For the convolutional backbones, the final two backbone stages and the classification head were made trainable.

Leave-one-out cross-validation was used for HRF and FIVES, whereas stratified five-fold cross-validation was used for SUSTech-SYSU. Accuracy, macro-F1, ROC-AUC, and per-class performance measures were calculated from the aggregated outof-fold predictions.

Runtime profiling was performed using the high-resolution HRF images. Vessel segmentation and pretrained-encoder inference were conducted on a GPU, whereas ring-wise feature extraction, feature standardization, and logistic-regression prediction were conducted on a CPU. Detailed stage-wise runtimes and hardware specifications are provided in Supplementary Section S17, Training and Inference Runtime Comparison.

## 4.5 Use of Generative Artificial Intelligence

During the preparation of this manuscript, the authors used ChatGPT (OpenAI) to assist with improving the language, clarity, and organization of the manuscript. All AI-assisted text was critically reviewed, verified, and revised by the authors. The authors retained full responsibility for the scientific conception and development of the proposed framework, study design, conduct of the experiments, data analysis, interpretation of the results, conclusions, and final manuscript.

## Data Availability

The retinal fundus image datasets analysed in this study are publicly available from their original sources: the High-Resolution Fundus (HRF) dataset [25, 26], the FIVES dataset [27], and the SUSTech-SYSU dataset [28]. No new image datasets were generated during the current study.

## References

[1] Grauslund, J., Green, A. & Sjølie, A. K. Prevalence and 25 year incidence of proliferative retinopathy among Danish type 1 diabetic patients. Diabetologia 52, 1829–1835 (2009).

[2] Klein, R., Knudtson, M. D., Lee, K. E., Gangnon, R. & Klein, B. E. K. The Wisconsin Epidemiologic Study of Diabetic Retinopathy: XXII the twenty-fiveyear progression of retinopathy in persons with type 1 diabetes. Ophthalmology 115, 1859–1868 (2008).

[3] Morizane, Y. et al. Incidence and causes of visual impairment in Japan: the first nation-wide complete enumeration survey of newly certified visually impaired individuals. Jpn. J. Ophthalmol. 63, 26–33 (2019).

[4] Quigley, H. A. & Broman, A. T. The number of people with glaucoma worldwide in 2010 and 2020. Br. J. Ophthalmol. 90, 262–267 (2006).

[5] Akhtar, S., Aftab, S., Ali, O. et al. A deep learning based model for diabetic retinopathy grading. Sci. Rep. 15, 3763 (2025). URL https://doi.org/10.1038/ s41598-025-87171-9.

[6] Wang, D., Lian, J. & Jiao, W. Multi-label classification of retinal disease via a novel vision transformer model. Front. Neurosci. 17, 1290803 (2024).

[7] Xu, H., Shao, X., Fang, D. & Huang, F. A hybrid neural network approach for classifying diabetic retinopathy subtypes. Front. Med. (Lausanne) 10, 1293019 (2024).

[8] Bhoopalan, R., Sekar, P., Nagaprasad, N., Mamo, T. R. & Krishnaraj, R. Task optimized vision transformer for diabetic retinopathy detection and classification in resource constrained early diagnosis settings. Sci. Rep. 15, 39047 (2025).

[9] Vijayalakshmi, S., Manoharan, J. S., Nivetha, B. et al. Multi-task deep learning framework combining CNN, vision transformers and PSO for accurate diabetic retinopathy diagnosis and lesion localization. Sci. Rep. 15, 35076 (2025). URL https://doi.org/10.1038/s41598-025-18742-z.

[10] Chaurasia, A. K., Liu, G. S., Greatbatch, C. J. et al. A generalised computer vision model for improved glaucoma screening using fundus images. Eye 39,

[11] Niemeijer, M. et al. Automated measurement of the arteriolar-to-venular width ratio in digital color fundus photographs. IEEE Trans. Med. Imaging 30, 1941– 1950 (2011).

[12] Cheung, N., Mitchell, P. & Wong, T. Y. Diabetic retinopathy. Lancet 376, 124–136 (2010).

[13] Weinreb, R. N., Aung, T. & Medeiros, F. A. The pathophysiology and treatment of glaucoma: A review. JAMA 311, 1901–1911 (2014).

[14] Zhou, Y. et al. A foundation model for generalizable disease detection from retinal images. Nature 622, 156–163 (2023).

[15] He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 770–778 (2016).

[16] Dosovitskiy, A. et al. An image is worth 16x16 words: Transformers for image recognition at scale. International Conference on Learning Representations (2021).

[17] Liu, Z. et al. A ConvNet for the 2020s. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11966–11976 (2022).

[18] Dai, W. et al. RIP-AV: Joint representative instance pre-training with context aware network for retinal artery/vein segmentation. Medical Image Computing and Computer Assisted Intervention – MICCAI 2024, Vol. 15001 of Lecture Notes in Computer Science, 764–774 (Springer Nature Switzerland, 2024).

[19] Pla, C. Fundus lesions toolkit. https://github.com/ClementPla/ fundus-lesions-toolkit (2023). Accessed: 2026-07-02.

[20] Suvorov, R. et al. Resolution-robust large mask inpainting with fourier convolutions. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2149–2159 (2022).

[21] Sundararajan, M., Taly, A. & Yan, Q. Axiomatic attribution for deep networks. Proceedings of the 34th International Conference on Machine Learning (ICML), Vol. 70 of Proceedings of Machine Learning Research, 3319–3328 (PMLR, 2017).

[22] Smilkov, D., Thorat, N., Kim, B., Vi´egas, F. & Wattenberg, M. SmoothGrad: removing noise by adding noise (2017). arXiv:1706.03825.

[23] Liu, W. et al. Full-resolution network and dual-threshold iteration for retinal vessel and coronary angiograph segmentation. IEEE J. Biomed. Health Inform. 26, 4623–4634 (2022).

[24] Guo, C., Christensen, A. N., Dahl, A. B., Yi, Y. & Hannemose, M. R. SA-UNetv2: Rethinking spatial attention U-Net for retinal vessel segmentation. 2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI) (2026).

[25] Budai, A., Bock, R., Maier, A., Hornegger, J. & Michelson, G. Robust vessel segmentation in fundus images. Int. J. Biomed. Imaging 2013, 154860 (2013).

[26] High-Resolution Fundus (HRF) image database. https://www5.cs.fau.de/ research/data/fundus-images/. Accessed: 2026-01-12.

[27] Jin, K. et al. FIVES: A fundus image dataset for artificial intelligence based vessel segmentation. Sci. Data 9, 475 (2022).

[28] Lin, L. et al. The SUSTech-SYSU dataset for automated exudate detection and diabetic retinopathy grading. Sci. Data 7, 409 (2020).

[29] Kocabiyik, N., Yalcin, B. & Ozan, H. The morphometric analysis of the central retinal artery. Ophthalmic Physiol. Opt. 25, 375–378 (2005).

[30] Sherry, L. M. et al. Reliability of computer-assisted retinal vessel measurement in a population. Clin. Exp. Ophthalmol. 30, 179–182 (2002).

[31] Yarmohammadi, A. et al. Optical coherence tomography angiography vessel density in healthy, glaucoma suspect, and glaucoma eyes. Invest. Ophthalmol. Vis. Sci. 57, OCT451–OCT459 (2016).

[32] Habib, M. S., Al-Diri, B., Hunter, A. & Steel, D. H. W. The association between retinal vascular geometry changes and diabetic retinopathy and their role in prediction of progression – an exploratory study. BMC Ophthalmol. 14, 89 (2014).

[33] Dashtbozorg, B. et al. Infrastructure for retinal image analysis. Proceedings of the Ophthalmic Medical Image Analysis International Workshop (OMIA), Vol. 3, 105–112 (2016).

[34] Andreini, P. & Bonechi, S. Towards a comprehensive characterization of arteries and veins in retinal imaging. Comput. Biol. Med. 195, 110516 (2025).

[35] Asl, M. E., Koohbanani, N. A., Frangi, A. F. & Gooya, A. Tracking and diameter estimation of retinal vessels using Gaussian process and Radon transform. J. Med. Imaging 4, 034006 (2017).

[36] Sun, C., Wang, J. J., Mackey, D. A. & Wong, T. Y. Retinal vascular caliber: systemic, environmental, and genetic associations. Surv. Ophthalmol. 54, 74–95 (2009).

[37] Owen, C. G. et al. Measuring retinal vessel tortuosity in 10-year-old children: validation of the Computer-Assisted Image Analysis of the Retina (CAIAR) program. Invest. Ophthalmol. Vis. Sci. 50, 2004–2010 (2009).

[38] Huang, F. et al. Vascular biomarkers for diabetes and diabetic retinopathy screening in Computational Retinal Image Analysis: Tools, Applications and Perspectives (Trucco, E., MacGillivray, T. & Xu, Y., eds) Ch. 16, 319–352 (Academic Press, 2019).

[39] Geirsdottir, A. et al. Retinal vessel oxygen saturation in healthy individuals. Invest. Ophthalmol. Vis. Sci. 53, 5433–5442 (2012).

[40] Olafsdottir, O. B., Hardarson, S. H., Gottfredsdottir, M. S., Harris, A. & Stef´ansson, E. Retinal oximetry in primary open-angle glaucoma. Invest. Ophthalmol. Vis. Sci. 52, 6409–6413 (2011).

[41] Hammer, M., Vilser, W., Riemer, T. & Schweitzer, D. Retinal vessel oximetrycalibration, compensation for vessel diameter and fundus pigmentation, and reproducibility. J. Biomed. Opt. 13, 054015 (2008).

[42] Beach, J. M., Schwenzer, K. J., Srinivas, S., Kim, D. & Tiedeman, J. S. Oximetry of retinal vessels by dual-wavelength imaging: Calibration and influence of pigmentation. J. Appl. Physiol. 86, 748–758 (1999).

[43] Hirsch, K., Cubbidge, R. P. & Heitmar, R. Dual wavelength retinal vessel oximetry – influence of fundus pigmentation. Eye 37, 2246–2251 (2023).

[44] Bartczak, P. Spectrally Tunable Light Sources for Implementing Computationally Designed Illuminations. Ph.D. thesis, University of Eastern Finland (2016).

[45] Cameron, M. & Kumar, L. The depths of cast shadow. Remote Sens. 11, 1806 (2019).

[46] Koll´ath, Z., Cool, A., Jechow, A., Koll´ath, K. et al. Introducing the dark sky unit for multi-spectral measurement of the night sky quality with commercial digital cameras. J. Quant. Spectrosc. Radiat. Transf. 253, 107162 (2020).

[47] Prahl, S. A. Optical absorption of hemoglobin. https://omlc.org/spectra/ hemoglobin/ (1999).

## Acknowledgements

This work was supported in part by an NSERC Discovery Grant (H.H.) and the Fields Institute. The authors thank Faisal Habib and other members of the Fields MDAS Lab for helpful and stimulating discussions.

## Funding

This study was funded by the Natural Sciences and Engineering Research Council of Canada (NSERC).

## Author Contributions

X.L. contributed to conceptualization, methodology, software development, investigation, validation, formal analysis, visualization, and writing the original draft. S.X. contributed to conceptualization, methodology, and reviewing and editing the manuscript. A.G. contributed to supervision, project administration, and reviewing and editing the manuscript. All authors read and approved the final manuscript. H.H. contributed to conceptualization, methodology, supervision, project administration, and reviewing and editing the manuscript. All authors read and approved the final manuscript.

## Competing Interests

The authors declare no competing interests.