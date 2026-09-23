# Interpretable AI plus Handheld, Portable Retinal Photographs: A Low-Cost Glaucoma Screening Solution for West Africa

Charis Y. N. Chiang<sup>1,2,3</sup>, Tarela Sarimiye<sup>4</sup>, Adeyinka Ashaye<sup>4</sup>, Martin Buist<sup>2</sup>, Michael A. Hauser<sup>5</sup>, Olusola Olawoye<sup>4</sup> <sup>\*</sup>, Michaël J.A. Girard<sup>1,3,6,7,8</sup> <sup>\*</sup>

1. Department of Ophthalmology, Emory University, Atlanta, Georgia, United States

2. Department of Biomedical Engineering, National University of Singapore, Singapore

3. Singapore Eye Research Institute, Singapore National Eye Centre, Singapore

4. Department of Ophthalmology, College of Medicine, University of Ibadan, Ibadan, Nigeria

5. Duke University School of Medicine, Durham, North Carolina, United States

6. Duke-NUS Graduate Medical School-Singapore, Singapore

7. Department of Biomedical Engineering, Georgia Institute of Technology, Atlanta, Georgia, United States

8. Emory Empathetic AI for Health Institute, Emory University, Atlanta, Georgia

\* Both authors contributed equally and share senior and corresponding authorship.

Keywords: glaucoma, artificial intelligence, deep learning, optic nerve head, macula, portable devices, handheld devices, community screening, fundus photography

Word count: 3674 (Manuscript Text)

256 (Abstract)

Tables: 2 + 2 (supplementary material)

Figures: 5 + 1 (supplementary material)

Commercial relationship: MJAG is the co-founder of the start-up company Abyss Processing Pte Ltd.

Ethics statement: This study was approved by the Institutional Ethics Review Board of the University of Ibadan/University College Hospital, Ibadan, Nigeria (UI/EC/24/0027), and all study procedures adhered to the tenets of the Declaration of Helsinki. Written informed consent was obtained from each subject.

Support: (1) the BrightFocus Foundation grant P383001572 (sub-award from Duke University) [MAH, OO]; (2) the Emory Eye Center (Emory University School of Medicine, Start-up funds, MJAG & CRE) [MJAG]; (3) a Challenge Grant from Research to Prevent Blindness, Inc. to the Department of Ophthalmology at Emory University [CYNC, MJAG]; (4) the NIH grant P30EY06360 to the Atlanta Vision Research Community [MJAG]; (5) the National Eye Institute of the National Institutes of Health under award numbers R01EY037299 and R01EY037245 [MJAG]; (6) the NMRC-LCG grant ‘TAckling & Reducing Glaucoma Blindness with Emerging Technologies (TARGET)’, award ID: MOH-OFLCG21jun-0003 [CYNC, MB, MJAG]; and (7) the NIH grant U01AG097746 [OO].

Corresponding Authors: Dr Michaël J.A. Girard Ophthalmic Engineering & Innovation Laboratory (OEIL), Emory Eye Center, Emory School of Medicine, Emory Clinic Building B, 1365B Clifton Road, NE, Atlanta GA 30322 mgirard@ophthalmic.engineering https://www.ophthalmic.engineering

Dr Olusola Olawoye   
College of Medicine, University of Ibadan, Ibadan, Nigeria, +2348023890063   
solaolawoye@yahoo.com

## ABSTRACT

Purpose. To develop and evaluate an interpretable artificial intelligence (AI) framework for glaucoma screening from low-cost portable, handheld retinal fundus photographs in a West African population and to compare its performance with clinical tabletop fundus imaging. Methods. We used data from a community-based study of 681 participants (1,362 eyes) in Nigeria, comprising 414 glaucoma, 478 glaucoma suspect, and 470 non-glaucoma eyes. Fundus photographs were acquired using the low-cost handheld, portable Volk Viva retinal camera and the Canon CR-2-AF tabletop camera. We fine-tuned component models separately to each device to perform vessel segmentation, cup and disc boundary segmentation, and feature extraction to detect optic nerve head features. A final classification model combined these components to classify scans as glaucoma, glaucoma suspect or non-glaucoma. Feature-weight analysis and Gradient-weighted Class Activation Mapping were used for interpretation. Results. The models performed well on both Volk Viva and Canon CR-2-AF images: Vessel segmentation: 0.98 Dice Coefficient (DC) (Volk) and 0.94 DC (Canon); Cup and disc segmentation: 0.95 DC (Volk) and 0.96 DC (Canon); Optic nerve head feature detection: area under the receiver operating characteristic curve (AUCs) of 0.83±0.03 (Volk) and 0.87±0.04 (Canon); Classification model: AUCs of 0.85±0.01 (Volk) and 0.93±0.01 (Canon). Reports for each image, present model decision confidence scores and decision-rationale visualizations to support clinical interpretation.

Conclusions. Volk Viva results were reasonably comparable to Canon CR-2-AF in the component models and not far behind in classification. This shows that interpretable AI combined with lowcost, portable imaging may enhance community-level glaucoma screening, especially in settings with limited specialist access and resources.

## INTRODUCTION

Glaucoma is the world’s leading cause of permanent blindness [1], [2], characterized by

distinct damage to the optic nerve [3], [4]. Early detection is crucial to slow disease progression

and prevent irreversible vision loss [5]. This is particularly important in West African populations

which bear a disproportionately high but understudied glaucoma burden [6], [7], [8]. However,

limited access to eye care remains a major barrier to diagnosis. There is approximately one

ophthalmologist per million population across West Africa, many without glaucoma-specific

training, and very few glaucoma subspecialists [6], [7], [8]. Even in Nigeria, one of the better-

resourced ophthalmic environments in West Africa, clinical need still far outstrips capacity. In

Nigeria, glaucoma accounts for over 16% of blindness and affects around 5% of adults over 40

years old; however, only 12.5% of Nigerian ophthalmologists reported glaucoma subspeciality

practice, and only 2.3% practice as strict glaucoma specialists [8].

In high-resource settings, glaucoma evaluation typically involves multiple tests. These

include intraocular pressure measurement, visual field testing, optical coherence tomography,

and fundus photography [9], [10]. Despite all this, glaucoma remains underdiagnosed even in

resource-rich settings [11]. The high-cost clinical imaging systems, limited specialist availability,

and limited clinical access amplify these barriers to glaucoma diagnosis in West Africa [7].

Therefore, scalable, low-cost screening approaches are urgently needed.

In recent years, portable fundus cameras have become increasingly affordable and

portable. However, their utility in ophthalmology has been studied primarily in urban and clinical

settings with few studies evaluating their use for glaucoma screening in low-resource community

settings [12]. Studies conducted in Australian, American, and Nepalese clinical settings have

reported mixed findings regarding the agreement between ophthalmologists’ assessments of

fundus photographs acquired using portable and conventional tabletop cameras [13], [14], [15],

[16]. A separate hospital-based study in China found poor agreement (κ=0.32) between glaucoma

predictions generated by the same deep learning algorithm from portable and tabletop fundus

photographs [17]. In Nigeria, Garba et al. evaluated several portable tools in 524 eyes (5.92%

glaucoma prevalence), including a Remidio portable fundus camera with integrated AI-based

glaucoma detection, tonometry, visual-field testing, and visual acuity testing [18]. The portable

fundus camera performed best, achieving an AUC of 0.91 for binary glaucoma classification [18].

However, this was a hospital-based study limited to binary glaucoma detection and there remains

limited evidence for AI-based glaucoma screening from a single low-cost portable fundus

photograph in West African community populations, particularly for distinguishing non-

glaucoma, glaucoma suspect, and glaucoma in the absence of available glaucoma specialists.

Additionally, although many glaucoma AI systems increasingly do incorporate explainability

techniques such as class activation maps and cup and disc segmentation, relatively few provide

a structured set of clinically interpretable intermediate findings that can be independently

reviewed by the clinician alongside the final prediction [19], [20].

In this study, we developed an interpretable deep learning pipeline to screen for glaucoma from a color fundus photograph acquired using a low-cost portable, handheld camera. The pipeline automatically identified the optic cup, optic disc, retinal vessels, and twelve optic nerve head (ONH) features, and then integrated these outputs using a downstream AI classifier to categorize each eye as non-glaucomatous, glaucoma suspect, or glaucomatous. Finally, we applied explainability techniques to generate a report summarizing the image-derived findings and model rationale to support clinical interpretation of the results. This approach aims to support scalable glaucoma screening in West African populations where specialist access and clinical imaging resources are limited.

## METHODS

## Patient Recruitment

A total of 681 subjects were included in this community-based study conducted in Nigeria. The cohort comprised 414 eyes with glaucoma, 478 glaucoma suspect eyes, and 470 nonglaucoma eyes. All subjects provided written informed consent. The study adhered to the tenets of the Declaration of Helsinki and was approved by the institutional Ethics review board of the University of Ibadan/University College Hospital Ibadan Nigeria (UI/EC/24/0027).

Diagnosis was performed by Nigerian clinicians with subspecialty training in glaucoma using anterior and posterior segment examination, dilated ONH assessment from slit lamp binocular fundus examination using a 78D lens, tabletop fundus photographs (Canon CR-2-AF), gonioscopy, visual field testing and optical coherence tomography where possible.

Glaucoma was defined as the presence of typical glaucomatous optic neuropathy, characterized by optic disc cupping, pallor, neuroretinal rim thinning, or retinal nerve fiber layer loss on optic disc examination and/or optical coherence tomography, with corresponding definite visual field loss. Although intraocular pressure was measured in all participants, it was not required for the diagnosis of glaucoma. Diagnostic classification followed the first two levels of evidence from the International Society of Geographical and Epidemiological Ophthalmology criteria. The first level required optic disc abnormality, defined as vertical cup-to-disc ratio (vCDR) ≥0.7, and/ or vCDR asymmetry ≥0.2, or focal neuroretinal rim narrowing to ≤0.1 cup-to-disc ratio (CDR) between 11–1 o’clock or 5–7 o’clock, together with a definite visual field defect consistent with glaucoma. The second level allowed a diagnosis of glaucoma in eyes with severe optic disc damage, defined as vCDR ≥0.9, when reliable visual field testing could not be performed due to poor vision.

Glaucoma suspects were defined as eyes with clinical findings suggestive of glaucoma that did not meet criteria for definite glaucoma. This group included optic disc suspects, visual field suspects, and ocular hypertension. Optic disc suspects had a glaucoma-appearing ONH, characterized by suspicious neuroretinal rim thinning, increased CDR, or inter-eye CDR asymmetry, without corresponding visual field loss. Visual field suspects had visual field defects suspicious for glaucoma without glaucomatous optic neuropathy. Ocular hypertension was defined as intraocular pressure >22 mmHg without glaucomatous optic neuropathy or visual field defects consistent with glaucoma.

Non-glaucomatous subjects were defined as eyes with intraocular pressure <22 mmHg and no evidence of glaucomatous optic neuropathy or other clinical evidence of glaucoma. Specifically, controls had no neuroretinal rim loss, and no definite visual field loss consistent with glaucomatous optic nerve damage.

## Fundus Photography

Participants were recruited from community outreaches within Oyo State and imaged out of clinic, in the community, using a low-cost, handheld, portable Volk Viva retinal camera following pupil dilation to improve image clarity. The operators acquired between one and four photographs from each eye during the same imaging session using the Volk Viva camera. Where multiple Volk Viva photographs were acquired from the same eye, each photograph was retained as a separate image for image-level analyses. The same participants were then referred to the eye clinic of the University College Hospital Ibadan where they were imaged using a Canon CR-2- AF tabletop retinal camera and underwent other clinical tests. The respective cameras and sample photographs from each are shown in Figure 1.

Separate device-specific datasets were used for development and evaluation of the component models. The Volk Viva analysis cohort comprised 1,224 eyes from 630 participants after exclusion of corrupted images, while the Canon CR-2-AF cohort comprised 1,291 eyes from 665 participants as shown in Table 1. Component models were fine-tuned separately for each device to perform retinal vessel segmentation, cup and disc segmentation, and detection of ONH features. Images considered ungradable by CYNC because of poor image quality or inadequate visualization of the optic disc were excluded from analyses requiring these features. For development and comparison of the final diagnostic classifiers, the dataset was further restricted to eyes with gradable imaging available from both devices. This paired classification cohort comprised 1,091 eyes from 589 participants. The same paired cohort was used for both devicespecific classifiers to permit direct comparison of their performance.

## Optic Nerve Head Feature Detector

Previously, our group had manually reviewed a set of 566 fundus photographs, including photographs from publicly available datasets and a separate West African dataset [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]. In each photograph, we determined whether the image quality was sufficient for grading. Images with insufficient quality for reliable grading, including poor focus, poor illumination, media opacity, or inadequate optic disc visibility, were classified as ungradable. We also determined whether each of the twelve fundus-derived ONH features were present or absent. These were, as depicted in Figure 2: tigroid fundus, saucerization, vessel bayonetting, visible lamina cribosa pores, notching, nasal peripapillary atrophy, temporal peripapillary atrophy, tortuous vessels, disc hemorrhage, bleeding, veinous occlusions, arterial occlusions. A ResNet18 model was then trained to automatically perform the grading. This pretrained model was used as a starting point for the models trained in this study.

In this study, we manually reviewed and graded images from both the Volk Viva and Canon CR-2-AF datasets. Five-fold cross-validation was performed on 90% of the dataset, and a 10% hold-out set was used for testing. Identical subject-level partitions were used across the two imaging datasets to enable direct comparison between devices. All images from both eyes of the same participant were assigned to the same partition, ensuring no subject-level leakage across the training, validation, and test sets. Images were resized to 224 × 224 pixels and the previously trained model was finetuned to each dataset separately.

## Cup and Disc Segmentation Model

In previous work, cup and disc boundaries were manually segmented in 60 images from African American patients at Emory Hospital and 60 images from a separate dataset collected from West Africans. These annotations were used to train an initial UNET++ segmentation model to automatically delineate the optic cup and disc.

A custom graphical user interface was developed to manually annotate the optic cup and optic disc on fundus photographs. This was used to manually delineate the cup and disc in 60 Volk Viva and 60 Canon CR-2-AF fundus images. Our previously pretrained UNET++ model was then fine-tuned separately to the Volk Viva and Canon CR-2-AF fundus images.

All images were resized to 320 × 480 pixels before training. The model was trained using a Jaccard-index-based loss function averaged across segmentation classes. To reduce overfitting and improve generalizability, data augmentation was applied to the training set using the Albumentations Python library. Augmentations included horizontal flipping, random rotation and translation, additive Gaussian noise, and random changes in saturation and image intensity.

The segmentation dataset was split into training and test sets, with balanced representation of glaucomatous, suspect and non-glaucomatous eyes. Model performance was evaluated on held-out test images using the Dice coefficient, calculated by comparing predicted cup and disc masks with the corresponding manual annotations.

The cup and disc mask predictions were used to take vCDR and horizontal CDR (hCDR) measurements from each image.

## Vessel Segmentation Model

The vessels in the same 60 Volk Viva and 60 Canon CR-2-AF photos were manually annotated for training of a vessel segmentation model.

We pretrained another UNET++ segmentation model on 113 publicly available manually annotated fundus photographs from the Digital Retinal Images for Vessel Extraction dataset, Structured Analysis of the Retina dataset, Child Heart and Health Study in England database 1 dataset, and High-Resolution Fundus dataset [31], [32], [33], [34]. This pretrained UNET++ model was then separately fine-tuned to the manually annotated Volk Viva and Canon CR-2-AF photographs in a similar manner as the cup and disc segmentation models. The resulting vessel masks were used as an additional interpretable input for downstream glaucoma classification, capturing vascular patterns and vessel trajectories around the ONH.

## Classification Model

Glaucoma classification was performed as a three-class task distinguishing non-glaucoma, glaucoma suspects, and glaucoma eyes. Outputs from the component models were integrated into a downstream classification model to categorize each eye as non-glaucomatous, glaucoma suspect, or glaucomatous. Inputs included the original fundus image, cup and disc segmentation outputs, vessel segmentation masks, ONH feature predictions, and the participant’s age. The classifier was trained and evaluated separately for Volk Viva and Canon CR-2-AF images to compare performance of low-cost portable imaging with the gold standard of tabletop retinal photography.

The classification model was a late-fusion neural network comprising an ImageNetpretrained ResNet18 backbone to process the raw colored fundus photograph, a convolutional branch for the cup-and-disc and vessel masks, and a tabular branch for all other variables. Fivefold cross-validation was performed on 90% of the dataset, and a 10% hold-out set was used for testing using the same splits as the feature-extraction model previously. Data augmentation was applied to the training set including horizontal flips, slight rotations, and Gaussian noise.

Model performance was evaluated using accuracy, class-balanced accuracy, class-specific precision and recall, and class-average one-vs-others area under the receiver operating characteristic curve (AUC).

A full overview of the automated pipeline is depicted in Figure 3.

## Interpretable Report

For each image, the final pipeline generated an interpretable report summarizing the model outputs. This report included image quality, cup and disc segmentation, vessel segmentation, estimated cup-to-disc ratio, detected ONH features, predicted diagnostic class, and model confidence. Three complementary explainability methods were applied to better understand the model decision and generate clinically interpretable summaries to support review by eye care providers. One, ablation experiments were conducted by removing each input type, namely the photograph, masks, and features, in turn from the full model and measuring the resulting decline in AUC compared to the full classification model [35]; two, gradientweighted class activation map (Grad-CAM) to identify the image regions contributing most strongly to the final classification [36]; and three, calculating Shapley Additive exPlanation (SHAP)

values to rank tabular feature importance [37]. For the individual report, the absolute SHAP values were normalized across tabular features to express each feature’s relative contribution as a percentage, such that the contributions summed to 100%.

## RESULTS

Of the 681 participants enrolled, Canon CR-2-AF photographs were acquired from 1,291 eyes of 665 participants, with one Canon photograph available per eye. A total of 2,777 Volk Viva photographs were acquired from 1,246 eyes of 642 participants. Forty-five Volk Viva images were corrupted on export, leaving 2,732 images from 1,224 eyes of 630 participants in the Volk Viva analysis cohort. Image-quality assessment identified 233 ungradable Volk Viva images and 113 ungradable Canon CR-2-AF images. Restricting the final diagnostic classification analysis to eyes with gradable imaging available from both devices resulted in a paired cohort of 1,091 eyes from 589 participants, comprising 287 glaucoma, 410 glaucoma suspect, and 394 non-glaucoma eyes as shown in Table 1.

The individual component models achieved excellent agreement with manual annotations for both retinal vessel and optic cup and disc segmentation (Table 2). For retinal vessel segmentation, Dice coefficients were 0.98 for Volk Viva images and 0.94 for Canon CR-2- AF images. Cup and disc segmentation similarly demonstrated high accuracy, with Dice coefficients of 0.95 and 0.96 for Volk Viva and Canon CR-2-AF images, respectively. Examples of the automated segmentations are shown in Figure 3. The feature extraction models achieved $0 . 8 3 \pm 0 . 0 3 \mathsf { A U C }$ and $0 . 8 7 \pm 0 . 0 4 \mathsf { A U C }$ for the Volk Viva and Canon CR-2-AF models, respectively.

Performance of the multimodal classification model differed between imaging devices, with the Canon CR-2-AF dataset achieving a macro-AUC of $0 . 9 3 \pm 0 . 0 1$ compared with $0 . 8 5 \pm 0 . 0 1$ for the portable Volk Viva dataset (Table 2). Receiver operating characteristic curves and confusion matrices are presented in Figure 4.

Ablation analysis revealed that the images and tabular features produced the largest reduction in classification performance for the Volk Viva classification model, while removal of segmentation masks contributed to the largest decrease in performance for the Canon CR-2-AF model (Supplementary Table 1). Across both datasets, SHAP analysis revealed that temporal peripapillary atrophy and CDR measurements were among the most influential in the tabular features, while tortuous vessels were also influential in the Canon CR-2-AF classification model (Supplementary Table 2). Grad-CAM analysis also revealed that the model generally focused on the disc region of each photo to make its decision (Supplementary Figure 1).

The model and explainability outputs were compiled to generate a report from each scan and a sample from a glaucomatous eye is shown in Figure 5.

## DISCUSSION

In this study, we developed an AI pipeline to screen for glaucoma specifically in a West African population. This was done using a single colored fundus photograph from a low-cost portable retinal camera. Rather than a black-box AI model, the pipeline was designed to parallel elements of clinical optic nerve assessment by first performing vessel segmentation, cup and disc segmentation and ONH feature detection before integrating these outputs with the fundus image in a downstream classifier to classify eyes into non-glaucoma, glaucoma suspect, and glaucoma. The pipeline also generated a clinician-friendly report. In this way, the screening aims to be transparent and accessible for clinicians working in a resource-limited setting where access to glaucoma specialists and advanced diagnostic equipment is scarce.

Overall, the proposed framework demonstrated good performance across each model component with comparable performance in both datasets. The vessel segmentation and cupand-disc boundary delineation achieved strong agreement with manual annotations while the ONH feature detector accurately identified most of the visible features on images from both devices. These models performed similarly or even better than other AI applications in fundus photo datasets from other demographics [38], [39], [40], [41]. These results suggest that the modular approach, modelled after a clinician’s mode of interpreting the same scans is feasible and can provide reliable intermediate outputs which clinicians can verify.

The performance achieved using the Canon CR-2-AF tabletop camera was only slightly better in classification performance than the low-cost Volk Viva portable retinal. Given the substantial difference in cost, portability, and ease of deployment between the two systems, this finding has important implications for expanding access to glaucoma screening in resourcelimited settings. Consequently, affordable portable imaging devices can feasibly support community-based glaucoma screening and lighten the clinical workload

A key strength of this work is its transparency and interpretability. Although many deep learning systems have demonstrated high accuracy for glaucoma diagnosis and screening, most operate as black-box models that provide little insight into how a prediction is made [42], [43].

This lack of transparency may reduce clinician confidence and hinder implementation in routine clinical practice. Our framework addresses this limitation by generating an interpretable report comprising retinal vessel segmentation, cup and disc segmentation, and ONH feature detection alongside the final classification. This is particularly valuable in resource-limited settings where glaucoma specialists are scarce, and clinicians may benefit from rapid, structured interpretation of fundus photographs. Importantly, the intermediate outputs are themselves clinically meaningful, allowing clinicians to independently assess the ONH and draw their own conclusions without relying solely on the model's final classification. Furthermore, immediate classification of an image as ungradable would provide real-time feedback to the camera operator, allowing them to determine whether the image is suitable for report generation or whether an additional image should be acquired while still in the field.

The modality contribution results for the Volk Viva classification model indicate that the relative importance of image and tabular features was greater than the vessel and cup and disc masks, suggesting that though the segmentation masks did supplement the imaging data in leading to the model’s classification decision, they did not dominate the decision. In comparison, the Canon CR-2-AF model showed substantially greater reliance on the masks. This may reflect the richer anatomical information available in the higher-quality Canon CR-2-AF images, including clearer cup and disc boundaries and a more extensive visible vascular network. Consequently, they may have contained more diagnostically informative structural information, whereas the sparser Volk Viva masks may have provided less information beyond that already captured by the RGB image and derived tabular features. However, it should be noted that because ablation measures the incremental contribution of each input in the presence of the remaining inputs, the

small performance drop after removing Volk masks should not be interpreted as indicating that the masks contained no diagnostically relevant information. Amongst tabular features, SHAP analysis indicated that temporal peripapillary atrophy, and vertical and horizontal CDR were the most influential predictors. The Grad-CAM outputs also suggest that across both cameras and across diagnosis classes, the model focus was centred on the disc. Overall, these results indicate that structural optic disc measurements were important features for the multimodal classifier.

Several limitations should be discussed. First, the proposed pipeline consists of multiple models, meaning that errors from upstream tasks, such as image quality assessment, segmentation, or feature detection, may propagate to downstream classification performance and affect performance. Although each individual component demonstrated strong performance, diagnostic accuracy remains dependent on the reliability of each component model.

Second, the Canon CR-2-AF photographs were used as part of the clinical assessment that established the ground truth diagnosis. As such, there may be a methodological advantage for the Canon CR-2-AF dataset in comparison with the Volk Viva dataset. This may have contributed to the observed difference in classification performance between the two imaging devices.

Third, this was a single-cohort study conducted in a community-based population in Nigeria. External validation in independent cohorts of other West African cohorts is necessary before widespread deployment can be recommended.

Fourth, in many cases, the Canon CR-2-AF photos were sharper and thus contained more visible features than the Volk Viva photos. Although the Volk Viva camera achieved high performance for vessel segmentation, cup and disc segmentation, and ONH feature detection, these tasks were evaluated only on structures that were clearly visible in the images. Fine anatomical features, such as lamina cribrosa pores and subtle optic disc saucerization, and smaller vessels, were more readily visualized on Canon CR-2-AF images than the portable images. Features that could not be reliably observed were therefore excluded from the ground-truth annotations and were not evaluated by the detection model.

Finally, several ONH features such as notching, arterial and venous occlusion, and vessel bayonetting were uncommon or absent within the study population. This limited the amount of training data available for those individual feature detectors. Future studies involving larger and more diverse datasets may improve the detection of these relatively rare findings and further enhance the interpretability of the proposed framework.

In conclusion, we developed an interpretable deep learning framework capable of screening for glaucoma from a single portable fundus photograph while simultaneously providing clinically meaningful intermediate outputs, including cup and disc segmentation, vessel segmentation, and detection of ONH features. The comparable performance achieved using a low-cost portable camera demonstrates the potential for scalable glaucoma screening in underserved regions. With further external validation, this approach may facilitate earlier detection of glaucoma while providing clinicians with transparent, explainable decision support that is well suited for community-based eye care.

## ACKNOWLEDGEMENTS

We acknowledge support from (1) the BrightFocus Foundation grant P383001572 (sub-award from Duke University) [MAH, OO]; (2) the Emory Eye Center (Emory University School of Medicine, Start-up funds, MJAG & CRE) [MJAG]; (3) a Challenge Grant from Research to Prevent Blindness, Inc. to the Department of Ophthalmology at Emory University [CYNC, MJAG]; (4) the NIH grant P30EY06360 to the Atlanta Vision Research Community [MJAG]; (5) the National Eye Institute of the National Institutes of Health under award numbers R01EY037299 and R01EY037245 [MJAG]; (6) the NMRC-LCG grant ‘TAckling & Reducing Glaucoma Blindness with Emerging Technologies (TARGET)’, award ID: MOH-OFLCG21jun-0003 [CYNC, MB, MJAG]; and (7) the NIH grant U01AG097746 [OO].

## REFERENCES

[1] A. Giangiacomo and A. L. Coleman, ‘The Epidemiology of Glaucoma’, in Glaucoma, F. Grehn and R. Stamper, Eds, in Essentials in Ophthalmology. , Berlin, Heidelberg: Springer, 2009, pp. 13–21. doi: 10.1007/978-3-540-69475-5\_2.

[2] L. Racette, M. R. Wilson, L. M. Zangwill, R. N. Weinreb, and P. A. Sample, ‘Primary openangle glaucoma in blacks: a review’, Surv. Ophthalmol., vol. 48, no. 3, pp. 295–313, Jun. 2003, doi: 10.1016/s0039-6257(03)00028-6.

[3] P. J. Foster, R. Buhrmann, H. A. Quigley, and G. J. Johnson, ‘The definition and classification of glaucoma in prevalence surveys’, Br. J. Ophthalmol., vol. 86, no. 2, pp. 238–242, Feb. 2002.

[4] M. Almasieh, A. M. Wilson, B. Morquette, J. L. Cueva Vargas, and A. Di Polo, ‘The molecular basis of retinal ganglion cell death in glaucoma’, Prog. Retin. Eye Res., vol. 31, no. 2, pp. 152–181, Mar. 2012, doi: 10.1016/j.preteyeres.2011.11.002.

[5] T. Qi, H. Liu, L. Frühn, K. Löw, C. Cursiefen, and V. Prokosch, ‘Understanding Glaucoma: Why it Remains a Leading Cause of Blindness Worldwide’, Klin. Monatsbl. Augenheilkd., vol. 242, no. 7, pp. 712–717, Jul. 2025, doi: 10.1055/a-2617-1575.

[6] D. L. Budenz et al., ‘Prevalence of Glaucoma in an Urban West African Population: The Tema Eye Survey’, JAMA Ophthalmol., vol. 131, no. 5, pp. 651–658, May 2013, doi: 10.1001/jamaophthalmol.2013.1686.

[7] P. R. Egbert, ‘Glaucoma in West Africa: a neglected problem’, Br. J. Ophthalmol., vol. 86, no. 2, pp. 131–132, Feb. 2002, doi: 10.1136/bjo.86.2.131.

[8] S. N. Onwubiko, N. N. Udeh, O. Nkwegu, D. O. Ukwu, and N. Z. Nwachukwu, ‘Glaucoma care in Nigeria: Is the current practice poised to tackle this emerging sight-threatening disease?’, Int. Ophthalmol., vol. 39, no. 10, pp. 2385–2390, Oct. 2019, doi: 10.1007/s10792-019-01078-9.

[9] P. Harasymowycz, A. Kamdeu Fansi, and D. Papamatheakis, ‘Screening for primary openangle glaucoma in the developed world: are we there yet?’, Can. J. Ophthalmol. J. Can. Ophtalmol., vol. 40, no. 4, pp. 477–486, Aug. 2005, doi: 10.1016/s0008-4182(05)80010-9.

[10] G. Mowatt et al., ‘Screening Tests for Detecting Open-Angle Glaucoma: Systematic Review and Meta-analysis’, Invest. Ophthalmol. Vis. Sci., vol. 49, no. 12, pp. 5373–5385, Dec. 2008, doi: 10.1167/iovs.07-1501.

[11] C. Jan, M. He, A. Vingrys, Z. Zhu, and R. S. Stafford, ‘Diagnosing glaucoma in primary eye care and the role of Artificial Intelligence applications for reducing the prevalence of undetected glaucoma in Australia’, Eye, vol. 38, no. 11, pp. 2003–2013, Aug. 2024, doi: 10.1038/s41433-024-03026-z.

[12] F. Garba et al., ‘Portable devices for the diagnosis of glaucoma: a scoping review’, BMJ Open, vol. 15, no. 10, Oct. 2025, doi: 10.1136/bmjopen-2025-105681.

[13] K. Yogesan et al., ‘Evaluation of a portable fundus camera for use in the teleophthalmologic diagnosis of glaucoma’, J. Glaucoma, vol. 8, no. 5, pp. 297–301, Oct. 1999.

[14] M. Waisbourd et al., ‘Evaluation of Nonmydriatic Hand-held Optic Disc Photography Grading in the Philadelphia Glaucoma Detection and Treatment Project’, J. Glaucoma, vol. 25, no. 5, pp. e520-525, May 2016, doi: 10.1097/IJG.0000000000000382.

[15] S. E. Miller et al., ‘Glaucoma Screening in Nepal: Cup-to-Disc Estimate With Standard Mydriatic Fundus Camera Compared to Portable Nonmydriatic Camera’, Am. J. Ophthalmol., vol. 182, pp. 99–106, Oct. 2017, doi: 10.1016/j.ajo.2017.07.010.

[16] S. Das et al., ‘Feasibility and clinical utility of handheld fundus cameras for retinal imaging’, Eye, vol. 37, no. 2, pp. 274–279, Feb. 2023, doi: 10.1038/s41433-021-01926-y.

[17] S. He et al., ‘Cross-camera Performance of Deep Learning Algorithms to Diagnose Common Ophthalmic Diseases: A Comparative Study Highlighting Feasibility to Portable Fundus Camera Use’, Curr. Eye Res., vol. 48, no. 9, pp. 857–863, Sep. 2023, doi: 10.1080/02713683.2023.2215984.

[18] F. Garba et al., ‘Comparison of portable devices with standard glaucoma diagnostic testing for the detection of glaucoma for the purposes of glaucoma case finding in low-and middle- income countries’, Eye, vol. 40, no. 7, pp. 1057–1066, May 2026, doi: 10.1038/s41433-026-04297-4.

[19] D. P. Rao et al., ‘Evaluation of an offline, artificial intelligence system for referable glaucoma screening using a smartphone-based fundus camera: a prospective study’, Eye, vol. 38, no. 6, pp. 1104–1111, Apr. 2024, doi: 10.1038/s41433-023-02826-z.

[20] S. Senthil et al., ‘Evaluating real-world performance of an automated offline glaucoma AI on a smartphone fundus camera across glaucoma severity stages’, PLOS One, vol. 20, no. 6, p. e0324883, Jun. 2025, doi: 10.1371/journal.pone.0324883.

[21] L.-P. Cen et al., ‘Automatic detection of 39 fundus diseases and conditions in retinal photographs using deep neural networks’, Nat. Commun., vol. 12, no. 1, p. 4828, Aug. 2021, doi: 10.1038/s41467-021-25138-w.

[22] R. Venkatesh et al., ‘Dissecting the clinical and pathophysiological complexity of fundus tessellation’, Surv. Ophthalmol., vol. 71, no. 2, pp. 382–392, Mar. 2026, doi: 10.1016/j.survophthal.2025.09.001.

[23] Y. N. Yan et al., ‘Long-term Progression and Risk Factors of Fundus Tessellation in the Beijing Eye Study’, Sci. Rep., vol. 8, no. 1, p. 10625, Jul. 2018, doi: 10.1038/s41598-018- 29009-1.

[24] ‘fundus-dataset.zip’. figshare, Nov. 11, 2021. doi: 10.6084/m9.figshare.16986166.v1.

[25] R. Kiefer, M. Abid, J. Steen, M. R. Ardali, and E. Amjadian, ‘A Catalog of Public Glaucoma Datasets for Machine Learning Applications: A detailed description and analysis of public glaucoma datasets available to machine learning engineers tackling glaucoma-related problems using retinal fundus images and OCT images.’, in Proceedings of the 2023 7th International Conference on Information System and Data Mining, in ICISDM ’23. New York, NY, USA: Association for Computing Machinery, Oct. 2023, pp. 24–31. doi: 10.1145/3603765.3603779.

[26] P. Kaur, ‘Bajwa Hospital (Multi Eye Disease Dataset)’. Mendeley Data, Jul. 26, 2022. doi: 10.17632/rgwpd4m785.3.

[27] ‘Fundus-AVSeg: fundus image dataset for AI-based artery-vein segmentation’. figshare, Dec. 02, 2024. doi: 10.6084/m9.figshare.27938034.v2.

[28] M. R. Rashid, S. Sharmin, T. Khatun, M. Z. Hasan, and M. Shorif Uddin, ‘Eye Disease Image Dataset’. Mendeley Data, Apr. 02, 2024. doi: 10.17632/s9bfhswzjb.1.

[29] A. Jain, ‘Glaucoma Fundus Imaging Datasets’. Kaggle. Accessed: Aug. 13, 2026. [Online]. Available: https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets

[30] O. Kovalyk, J. Morales-Sánchez, R. Verdú-Monedero, I. Sellés-Navarro, A. Palazón-Cabanes, and J.-L. Sancho-Gómez, ‘PAPILA: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment’, Sci. Data, vol. 9, no. 1, p. 291, Jun. 2022, doi: 10.1038/s41597-022-01388-1.

[31] J. Staal, M. D. Abràmoff, M. Niemeijer, M. A. Viergever, and B. van Ginneken, ‘Ridge-based vessel segmentation in color images of the retina’, IEEE Trans. Med. Imaging, vol. 23, no. 4, pp. 501–509, Apr. 2004, doi: 10.1109/TMI.2004.825627.

[32] A. D. Hoover, V. Kouznetsova, and M. Goldbaum, ‘Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response’, IEEE Trans. Med. Imaging, vol. 19, no. 3, pp. 203–210, Mar. 2000, doi: 10.1109/42.845178.

[33] M. M. Fraz et al., ‘An Ensemble Classification-Based Approach Applied to Retinal Blood Vessel Segmentation’, IEEE Trans. Biomed. Eng., vol. 59, no. 9, pp. 2538–2548, Sep. 2012, doi: 10.1109/TBME.2012.2205687.

[34] A. Budai, R. Bock, A. Maier, J. Hornegger, and G. Michelson, ‘Robust Vessel Segmentation in Fundus Images’, Int. J. Biomed. Imaging, vol. 2013, no. 1, p. 154860, 2013, doi: 10.1155/2013/154860.

[35] M. A. Alrasheedi, A. S. M. A. Luhayb, and A. A. R. Alharbi, ‘Deep learning framework for early diagnosis of lung cancer using multi- modal medical imaging’, AIMS Math., vol. 10, no. 12, pp. 29815–29852, Dec. 2025, doi: 10.3934/math.20251310.

[36] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, ‘Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization’, in 2017 IEEE International Conference on Computer Vision (ICCV), Oct. 2017, pp. 618–626. doi: 10.1109/ICCV.2017.74.

[37] S. M. Lundberg and S.-I. Lee, ‘A Unified Approach to Interpreting Model Predictions’, in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2017. Accessed: Sep. 14, 2026. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2017/hash/8a20a8621978632d76c43df d28b67767-Abstract.html

[38] Q. Qin and Y. Chen, ‘A review of retinal vessel segmentation for fundus image analysis’, Eng. Appl. Artif. Intell., vol. 128, p. 107454, Feb. 2024, doi: 10.1016/j.engappai.2023.107454.

[39] V. G. Edupuganti, A. Chawla, and A. Kale, ‘Automatic Optic Disk and Cup Segmentation of Fundus Images Using Deep Learning’, in 2018 25th IEEE International Conference on Image Processing (ICIP), Oct. 2018, pp. 2227–2231. doi: 10.1109/ICIP.2018.8451753.

[40] Y. Bazi, M. M. Al Rahhal, H. Elgibreen, and M. Zuair, ‘Vision transformers for segmentation of disc and cup in retinal fundus images’, Biomed. Signal Process. Control, vol. 91, p. 105915, May 2024, doi: 10.1016/j.bspc.2023.105915.

[41] P. Sharma et al., ‘A hybrid multi model artificial intelligence approach for glaucoma screening using fundus images’, NPJ Digit. Med., vol. 8, p. 130, Feb. 2025, doi: 10.1038/s41746-025-01473-w.

[42] A. R. Ran et al., ‘Deep learning in glaucoma with optical coherence tomography: a review’, Eye, vol. 35, no. 1, pp. 188–201, Jan. 2021, doi: 10.1038/s41433-020-01191-5.

[43] Y. G. Liang, L. Fan, A. Teixeira-Pinto, G. Liew, and A. J. R. White, ‘A systematic review of AI for predicting glaucoma progression: challenges and recommendations towards clinical implementation’, Npj Digit. Med., vol. 9, no. 1, p. 140, Jan. 2026, doi: 10.1038/s41746-025- 02321-7.

Table 1. Cohort Demographics
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Volk Viva Dataset</td><td rowspan=1 colspan=1>Canon CR-2-AFDataset</td><td rowspan=1 colspan=1>Paired Cohort forClassification Model</td></tr><tr><td rowspan=1 colspan=1>No. subjects</td><td rowspan=1 colspan=1>630</td><td rowspan=1 colspan=1>665</td><td rowspan=1 colspan=1>589</td></tr><tr><td rowspan=1 colspan=1>Age</td><td rowspan=1 colspan=1> $5 2 . 3 \pm 1 1 . 7$ </td><td rowspan=1 colspan=1> $5 2 . 6 \pm 1 2 . 0$ </td><td rowspan=1 colspan=1> $5 1 . 4 \pm 1 1 . 3$ </td></tr><tr><td rowspan=1 colspan=1>Sex</td><td rowspan=1 colspan=1>394 female; 236 male</td><td rowspan=1 colspan=1>413 female; 252 male</td><td rowspan=1 colspan=1>378 female; 211 male</td></tr><tr><td rowspan=1 colspan=1>No. Eyes</td><td rowspan=1 colspan=1>1224</td><td rowspan=1 colspan=1>1291</td><td rowspan=1 colspan=1>1,091</td></tr><tr><td rowspan=1 colspan=1>Glaucoma Eyes</td><td rowspan=1 colspan=1>354</td><td rowspan=1 colspan=1>379</td><td rowspan=1 colspan=1>287</td></tr><tr><td rowspan=1 colspan=1>GlaucomaSuspect Eyes</td><td rowspan=1 colspan=1>440</td><td rowspan=1 colspan=1>466</td><td rowspan=1 colspan=1>410</td></tr><tr><td rowspan=1 colspan=1>Non-GlaucomaEyes</td><td rowspan=1 colspan=1>430</td><td rowspan=1 colspan=1>446</td><td rowspan=1 colspan=1>394</td></tr></table>

Table 2. Performance Metrics from each Model
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Volk Viva</td><td rowspan=1 colspan=1>Canon CR-2-AF</td></tr><tr><td rowspan=1 colspan=1>Vessel Segmentation</td><td rowspan=1 colspan=1>0.98 Dice Coefficient</td><td rowspan=1 colspan=1>0.94 Dice Coefficient</td></tr><tr><td rowspan=1 colspan=1>Cup/ Disc Segmentation</td><td rowspan=1 colspan=1>0.95 Dice Coefficient</td><td rowspan=1 colspan=1>0.96 Dice Coefficient</td></tr><tr><td rowspan=1 colspan=1>Feature Extraction</td><td rowspan=1 colspan=1> $0 . 8 3 \pm 0 . 0 3 \mathsf { A U C }$ </td><td rowspan=1 colspan=1> $0 . 8 7 \pm 0 . 0 4 \mathsf { A U C }$ </td></tr><tr><td rowspan=1 colspan=1>Classification</td><td rowspan=1 colspan=1> $0 . 8 5 \pm 0 . 0 1 \mathsf { A U C }$ </td><td rowspan=1 colspan=1> $0 . 9 3 \pm 0 . 0 1 \mathsf { A U C }$ </td></tr></table>

![](images/6f95844d8d6ca7fd923314d6886a840ce71f942672e384b514c002aaa882e9ce.jpg)

B  
![](images/34294c057fa1e2b05c48f946c7099f1223699d2d603708f2d3f0fc78ad56252e.jpg)  
C

![](images/8406a2a80d6fb4df9647188edfcfbdae6e5ba243492ad40309022dd920881fee.jpg)

![](images/197db387c2d6b81a9de552ab98f3609838ce94e6d3ea030fec6fd4b099c73411.jpg)  
Figure 1A. Depiction of the Canon CR-2-AF tabletop camera; B. Fundus photo of a glaucomatous eye taken with Canon CR-2-AF tabletop camera. C. Depiction of the Volk Viva portable camera; D. Fundus photo of the same eye taken with Volk Viva portable camera.

![](images/0346269541dee632e2650c45e1cdc413c668c3024f2bd56a6287f5827054b597.jpg)  
Figure 2. Examples of each of the eleven ONH features that were detected in the fundus photographs. Nasal and temporal peripapillary atrophy were assessed separately in photographs. Arterial occlusion was not detected in any photograph in the dataset.

A. Image Analysis and Feature Extraction  
![](images/435af67258b5808540ebaca715ef4409325f4d041ae47b66a2343b3f00a30bf6.jpg)  
Figure 3. Overview of AI Pipeline. A. A single fundus photograph is processed by three component models: One, a UNET++ model to segment the vessels; two, a UNET++ model to segment the optic cup and disc, from which the vertical and horizontal cup-to-disc ratios (vCDR and hCDR) are derived; and three, a CNN-based feature detector to identify clinically relevant optic nerve head features and flag ungradable images for exclusion. B. The original fundus photograph, vessel and cup and disc masks, vCDR, hCDR, detected features, and participant’s age are subsequently integrated into a model which classifies the eye as non-glaucoma, glaucoma suspect, or glaucoma, and generates a report to support clinical assessment.

![](images/c26e60cf545252e3ae3054e807d3b82c7cd273923ef6ac348864ad0c961f8876.jpg)

B  
![](images/6c72b47a06c8fa9e7dc5cad33d7a1ffc74e514d096c6b48465032b672f1f3cc8.jpg)

C  
![](images/555a934518d5df4d1c98178817ff4f08acc12c805a55ae80ae4efdae7fb9a414.jpg)  
Figure 4A. Receiver Operating Characteristic Curve for both classification models. B. Volk Viva confusion matrix from mean of confusion matrix across the five folds. C. Canon CR-2-AF confusion matrix from mean of confusion matrix across the five folds.

Fundus Glaucoma Report Subject ID: Glaucoma123 Eye: OD

Device: Volk Viva Handheld

Fundus Photo  
![](images/89329211753c39d6ab745b2096669275df0a4b7a460dd316e19fad1255df5ea0.jpg)

![](images/aac52ecb541558917ec2f8dc8ee1c67e3d0beb3382c83460e5606b6b3dd73c44.jpg)  
Vessel Contours

![](images/b2e7f6c9ff6815b75087a2a80e955d074dbd1d847a3c69191d40e26efc83517a.jpg)

![](images/413b28e4c43a92caac0a7b9e30a18a4414d53a4aaa5e5165e14886c19b8cbd95.jpg)

<table><tr><td rowspan=1 colspan=1>Classification</td><td rowspan=1 colspan=1>Probability</td></tr><tr><td rowspan=1 colspan=1>Glaucoma</td><td rowspan=1 colspan=1>88%</td></tr><tr><td rowspan=1 colspan=1>Glaucoma Suspect</td><td rowspan=1 colspan=1>10%</td></tr><tr><td rowspan=1 colspan=1>Non-Glaucoma</td><td rowspan=1 colspan=1>2%</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Feature</td><td rowspan=1 colspan=1>FeatureValue</td><td rowspan=1 colspan=1>% Contribution toPrediction</td></tr><tr><td rowspan=1 colspan=1>Vertical C/D</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>41.8</td></tr><tr><td rowspan=1 colspan=1>Age</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>23.9</td></tr><tr><td rowspan=1 colspan=1>Horizontal C/D</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>18.4</td></tr><tr><td rowspan=1 colspan=1>Visible Lamina CribosaPores</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>5.1</td></tr><tr><td rowspan=1 colspan=1>Peripapillary Atrophy (T)</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>4.5</td></tr><tr><td rowspan=1 colspan=1>Saucerization</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>3.2</td></tr><tr><td rowspan=1 colspan=1>Tigroid Fundus</td><td rowspan=1 colspan=1>Present</td><td rowspan=1 colspan=1>1.4</td></tr><tr><td rowspan=1 colspan=1>Vessel Bayonetting</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>0.7</td></tr><tr><td rowspan=1 colspan=1>Peripapillary Atrophy (N)</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>0.6</td></tr><tr><td rowspan=1 colspan=1>Notching</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>Bleeding</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>Tortuous Vessels</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>Disc Hemorrhage</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>0.1</td></tr><tr><td rowspan=1 colspan=1>Vein Occlusion</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>Arterial Occlusion</td><td rowspan=1 colspan=1>Absent</td><td rowspan=1 colspan=1>0.0</td></tr></table>

Figure 5. An example report from a glaucomatous eye presents the original fundus photograph, the model focus area identifying the regions contributing most to the classification decision, it also overlays the segmented vessel contours on the photograph and outlines the disc and cup boundary identified. Below is a table presenting the class-specific confidence scores indicating the probability that the photograph should be classified as each class. Finally, the bottom of the report displays a table with the feature value extracted from each image by the feature detection model, and for each feature, how much it contributed to the prediction, relative to the other features.

## SUPPLEMENTARY MATERIAL

## Supplementary Table 1. Ablation Test of Inputs

<table><tr><td rowspan=1 colspan=1>Input removed</td><td rowspan=1 colspan=1>Canon CR-2-AF % drop in macro-AUC</td><td rowspan=1 colspan=1>Volk Viva % drop in macro-AUC</td></tr><tr><td rowspan=1 colspan=1>RGB image</td><td rowspan=1 colspan=1>0.3%</td><td rowspan=1 colspan=1>7.2%</td></tr><tr><td rowspan=1 colspan=1>Masks</td><td rowspan=1 colspan=1>19.5%</td><td rowspan=1 colspan=1>0.3%</td></tr><tr><td rowspan=1 colspan=1>Tabular features</td><td rowspan=1 colspan=1>4.7%</td><td rowspan=1 colspan=1>8.5%</td></tr></table>

Supplementary Table 2. Shapley Additive exPlanation (SHAP) scores of tabular features
<table><tr><td colspan="1" rowspan="1">Feature</td><td colspan="1" rowspan="1">Volk Viva (mean ± SD)</td><td colspan="1" rowspan="1">Canon CR-2-AF (mean ± SD)</td></tr><tr><td colspan="1" rowspan="1">Temporal peripapillary atrophy</td><td colspan="1" rowspan="1"> $0 . 3 1 5 \pm 0 . 0 2 5$ </td><td colspan="1" rowspan="1"> $0 . 2 1 1 \pm 0 . 0 7 5$ </td></tr><tr><td colspan="1" rowspan="1">Vertical Cup-to-Disc Ratio</td><td colspan="1" rowspan="1"> $0 . 0 9 8 \pm 0 . 0 4 6$ </td><td colspan="1" rowspan="1"> $0 . 0 6 5 \pm 0 . 0 2 8$ </td></tr><tr><td colspan="1" rowspan="1">Horizontal Cup-to-Disc Ratio</td><td colspan="1" rowspan="1"> $0 . 0 8 8 \pm 0 . 0 2 5$ </td><td colspan="1" rowspan="1"> $0 . 0 8 1 \pm 0 . 0 2 1$ </td></tr><tr><td colspan="1" rowspan="1">Age</td><td colspan="1" rowspan="1"> $0 . 0 6 2 \pm 0 . 0 2 2$ </td><td colspan="1" rowspan="1"> $0 . 0 7 0 \pm 0 . 0 1 9$ </td></tr><tr><td colspan="1" rowspan="1">Visible Lamina Cribosa Pores</td><td colspan="1" rowspan="1"> $0 . 0 5 9 \pm 0 . 0 1 3$ </td><td colspan="1" rowspan="1"> $0 . 0 1 5 \pm 0 . 0 0 5$ </td></tr><tr><td colspan="1" rowspan="1">Saucerization</td><td colspan="1" rowspan="1"> $0 . 0 2 7 \pm 0 . 0 0 3$ </td><td colspan="1" rowspan="1"> $0 . 0 2 0 \pm 0 . 0 0 7$ </td></tr><tr><td colspan="1" rowspan="1">Tigroid Fundus</td><td colspan="1" rowspan="1"> $0 . 0 2 0 \pm 0 . 0 0 8$ </td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td colspan="1" rowspan="1">Tortuous Vessels</td><td colspan="1" rowspan="1"> $0 . 0 1 1 \pm 0 . 0 0 4$ </td><td colspan="1" rowspan="1"> $0 . 1 3 4 \pm 0 . 0 1 6$ </td></tr><tr><td colspan="1" rowspan="1">Nasal Peripapillary Atrophy</td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td colspan="1" rowspan="1"> $0 . 0 1 1 \pm 0 . 0 0 7$ </td></tr><tr><td colspan="1" rowspan="1">Vessel Bayonetting</td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td colspan="1" rowspan="1">Disc Haemorrhage</td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td colspan="1" rowspan="1">Notching</td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td colspan="1" rowspan="1">Arterial occlusion</td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td colspan="1" rowspan="1">Bleeding</td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td colspan="1" rowspan="1">Veinous occlusion</td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td colspan="1" rowspan="1"> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

![](images/e077c7ff8593683037c7c4b7852216e5c918801f86738d1bcb6091ecee9d8f1e.jpg)  
Supplementary Figure 1. Grad-CAM outputs for each image from each camera and diagnosis class revealed that model focus was primarily on the disc and cup region.