# FORESEE-MS: TOWARD PROSPECTIVE SPATIAL FORECASTING OF FUTURE MULTIPLE SCLEROSIS LESIONS FROM BASELINE FLAIR MRI

Maryam Akhavan Aghdam, Sudip Mittal, Noorbakhsh Amiri Golilarz, Shahram Rahimi

Department of Computer Science, The University of Alabama, Tuscaloosa, AL, USA

## ABSTRACT

Traditional multiple sclerosis (MS) image analysis has primarily focused on retrospectively detecting and segmenting lesions. This study investigates whether baseline FLAIR MRI can identify regions at risk for future lesion formation beyond population-level spatial patterns. We introduce FORESEE-MS, an evaluation protocol that uses a foldspecific population-prior atlas as a baseline and assesses predictions using pooled voxel-level discrimination (AU-ROC and AUPRC) and within-patient lesion localization (Recall@Top-K%), with patient-level bootstrap confidence intervals. We evaluated five deep learning models (3D U-Net, Attn+ResBlock+VAE, ViT, and two SwinUNETR variants) using 5-fold cross-validation on 39 subjects from the MSSEG-2 dataset. The atlas achieved the highest pooled voxel-level AUROC and AUPRC, with no deep learning model reliably outperforming it on either metric. However, the 3D U-Net and Attn+ResBlock+VAE achieved higher mean voxel-level within-patient Recall@Top-K% than the atlas at all evaluated thresholds, although most paired confidence intervals for the differences included zero. These findings suggest that baseline FLAIR may contain patientspecific predictive information, but in this small cohort, it remains uncertain whether this information improves lesion forecasting beyond a population-prior atlas.

Index Terms— Multiple sclerosis, lesion forecasting, deep learning, FLAIR MRI, population-prior atlas

## 1. INTRODUCTION

Multiple sclerosis (MS) is a chronic demyelinating disease of the central nervous system, characterized by focal white matter lesions on magnetic resonance imaging (MRI) [1]. Prior MS image-analysis work has largely addressed retrospective lesion segmentation [2, 3]: given one or more scans, identify pathology that is already present. This leaves open a different question: given only a patient’s baseline scan, can we anticipate where new pathology is likely to emerge?

Prior evidence suggests this may be possible. Normalappearing white matter (NAWM) that later becomes a lesion has been reported to show elevated T2 signal and reduced magnetization transfer ratio before lesion formation [4], and fluid-attenuated inversion recovery (FLAIR) is closely related to T2-weighted contrast. We therefore ask: can baseline

FLAIR MRI identify regions at risk offuture lesionformation, beyond population-level spatial patterns?

Existing work on future lesion activity mostly operates at the patient or region level. Baseline MRI has been used to predict future new and enlarging T2 lesion counts [5], and a treatment-aware diffusion model generates voxel-level future lesion masks from MRI and treatment information [6]. A recent study extracted radiomic features from prelesional tissue in prior scans and trained classifiers to distinguish them from those of sampled control normal-appearing white matter (NAWM) [7]. In contrast, we evaluate dense voxel-wise risk maps across the entire brain using baseline FLAIR MRI and derived spatial information, without incorporating treatment or clinical data.

We make two primary contributions. First, we introduce FORESEE-MS, an evaluation protocol for forecasting future MS lesion locations from baseline MRI. We construct a fold-specific population-prior atlas as a baseline and evaluate predictions using two complementary approaches: pooled voxel-level discrimination, measured by the area under the receiver operating characteristic curve (AUROC) and the area under the precision-recall curve (AUPRC), and withinpatient lesion localization, measured by Recall@Top-K%, with patient-level bootstrap confidence intervals (CIs). Second, we evaluate five deep learning models, including a 3D U-Net, an Attn+ResBlock+VAE, a vision transformer (ViT), and two SwinUNETR variants, using this protocol on the MSSEG-2 dataset [2].

## 2. METHOD

Problem setting. Fig. 1 presents an overview of the FORESEE-MS workflow. For each subject, the forecasting methods produce voxel-wise risk maps for new-lesion formation at follow-up. Each deep learning model receives two input channels derived solely from the baseline scan: intensitynormalized FLAIR and a Euclidean distance-to-brain-boundary map. Follow-up new-lesion masks serve as training targets and evaluation ground truth, while training-fold masks also contribute to the population-prior atlas. We used held-out follow-up scans and lesion masks only for evaluation, not for training, model selection, or risk-map generation.

Population-prior baseline. To obtain a population-level baseline, we built a voxel-wise atlas of new-lesion frequency.

![](images/559744ac0dc99fa76cd7e37a4a77e413500b16d56108c9e16fd5d20e173c5861.jpg)  
Fig. 1: Overview of the FORESEE-MS workflow. (a) Training-subject follow-up new-lesion masks are used to construct the fold-specific population-prior atlas and supervise the deep learning models. (b) At inference, the models receive only baseline FLAIR and the distance-to-brain-boundary map. (c,d) The atlas and each deep learning model independently produce voxelwise risk maps for future lesion formation. (e) Risk maps are evaluated using pooled voxel-level AUROC and AUPRC and within-patient Recall@Top-K%. (f) Held-out follow-up lesion masks are used only for evaluation.

We registered each training subject’s FLAIR to the ICBM152 2009a template [8] using affine and SyN registration [9] (ANTsPy), and applied the same transform to that subject’s new-lesion mask. Each template voxel was assigned the fraction of training subjects with a new lesion there (among subjects whose brain mask covers it), followed by Gaussian smoothing (σ = 2 mm). We rebuilt the atlas for each crossvalidation fold (Section 3.1) from the training subjects of that fold only. We then warped it into each held-out subject’s native space using the inverse of that subject’s template registration. This approach ensured that held-out subjects contributed only their anatomical information, not their lesion data.

Models. We compared five deep learning models; the loss and training details are in Section 3.1. (1) A shallow 3D U-Net [10] (∼1.4M parameters). (2) An attention- and residual-enhanced 3D U-Net (Attn+ResBlock+VAE) incorporating attention gates [11], pre-activation residual blocks, and a VAE-based regularization branch that reconstructs the input FLAIR [12] (∼2.4M parameters). (3) A lightweight 3D ViT [13] with a patch-embedding layer, a 6-layer transformer encoder, and a convolutional upsampling decoder producing voxel-wise predictions (∼3.5M parameters). Models (1)–(3) were trained from scratch. (4–5) SwinUNETR [14] (∼62M parameters), initialized from BrainSegFounder [15], a brain MRI foundation model pretrained with self-supervision on UK Biobank brain MRI and then fine-tuned for brain tumor segmentation. We tested SwinUNETR with two transfer strategies: in the frozen variant, we keep the Swin transformer encoder’s pretrained weights fixed and train the remaining layers; in the fine-tuned variant, we train all layers on our data. For both variants, we re-initialized three input-facing layers (the patch-embedding projection and two convolutions in the first encoder block) to accommodate our two-channel input, and the output layer for our single-channel output.

## 3. EXPERIMENTAL RESULTS AND DISCUSSION 3.1. Experimental Setup

Data and preprocessing. We used the MSSEG-2 dataset [2], which includes 40 training subjects with baseline and followup FLAIR images and consensus new-lesion masks. The 60-subject test partition lacks public ground-truth labels; therefore, we used only the 40 labeled training subjects. We preprocessed the volumes using the official MSSEG-2 pipeline [2], which performs brain extraction and bias-field correction, and then applied z-score normalization within the brain mask. During quality control, we excluded one subject whose lesion mask was largely outside the brain mask (27.6% overlap compared to 98–100% for all others), resulting in 39 subjects. New lesions comprise approximately 0.035% of brain voxels (58,292 of 1.65 × 10<sup>8</sup>).

Training. We used 5-fold subject-level cross-validation and trained models on $6 4 ^ { 3 }$ patches. For lesion-positive subjects, approximately half of the patches were centered on new-lesion voxels with random offsets. Data augmentation included random axis flips and additive Gaussian noise on the FLAIR channel. Whole-volume inference used overlapping sliding windows (stride 32), with predictions averaged across overlapping regions. All models minimized a segmentation loss, $\mathcal { L } _ { \mathrm { s e g } } ~ = ~ \mathcal { L } _ { \mathrm { B C E } } + \mathcal { L } _ { \mathrm { T v e r s k y } }$ , with a positive-class weight of 50 for binary cross-entropy (BCE) and Tversky parameters $\alpha ~ = ~ 0 . 3$ and $\beta ~ = ~ 0 . 7 ~ [ 1 6 ]$ . Additionally, Attn+ResBlock+VAE used the combined loss $\mathcal { L } = \mathcal { L } _ { \mathrm { s e g } } + 0 . 1 ( \mathcal { L } _ { \mathrm { M S E } } + 0 . 1 \mathcal { L } _ { \mathrm { K L } } )$ , where MSE measures input FLAIR reconstruction error and KL regularizes the VAE latent representation.

The U-Net, Attn+ResBlock+VAE, and ViT were trained for 120 epochs (batch size 4), while both SwinUNETR variants were trained for 60 epochs (batch size 2). We used the Adam optimizer (weight decay $1 0 ^ { - 5 } )$ for the U-Net and Attn+ResBlock+VAE, and AdamW (weight decay $1 0 ^ { - 4 } )$ for ViT and SwinUNETR. Initial learning rates were $3 \times 1 0 ^ { - 4 }$ for the U-Net, Attn+ResBlock+VAE, and ViT; $1 0 ^ { - 4 }$ for frozen SwinUNETR; and $2 \times 1 0 ^ { - 5 }$ for fine-tuned SwinUNETR.

Evaluation metrics. We evaluated predicted risk maps against follow-up new-lesion masks using pooled voxel-level AUROC and AUPRC across all out-of-fold brain voxels to assess discrimination between future lesion and non-lesion voxels. We assessed within-patient lesion localization using Recall@Top-K% $( K \in \{ 1 , 5 , 1 0 , 2 0 \} )$ , which measures the fraction of future lesion voxels or lesion objects captured within each subject’s top-K% highest-risk brain voxels. We averaged voxel-level recall across 28 lesion-positive subjects, whereas lesion-level recall was computed across 147 new-lesion objects, counting an object as detected if any of its voxels overlapped the selected region. Atlas ties were handled using exact expected recall under uniform random tie-breaking. Because baseline lesion masks were unavailable, we considered all brain voxels eligible for selection, including those containing pre-existing lesions.

We estimated 95% CIs using patient-level bootstrap resampling, with 1,000 resamples for pooled AUROC and AUPRC and 10,000 for Recall@Top-K%. All voxels and lesion objects from each subject were resampled together. Paired comparisons used the same resamples across methods. For each model and metric M, we computed $\Delta \ =$ $M _ { \mathrm { m o d e l } } - M _ { \mathrm { a t l a s } }$ and reported the mean bootstrap difference with a 95% percentile CI. Negative differences favored the atlas. The 40 recall comparisons and 10 pooled AUROC and AUPRC comparisons were not adjusted for multiple testing; thus, intervals excluding zero were interpreted as nominal, exploratory findings.

## 3.2. Quantitative Results

Pooled voxel-level discrimination. Table 1 summarizes pooled AUROC and AUPRC. The population-prior atlas achieved the highest AUROC (0.747) and AUPRC (1.24 × $1 0 ^ { - 3 } )$ , while all five deep learning models had lower AU-ROC point estimates (0.643–0.682). All paired AUROC CIs included zero. For the U-Net, the mean paired bootstrap difference (∆) was −0.043 (95% CI [−0.237, +0.120]) for AUROC and −0.0004 (95% CI [−0.0015, +0.0005]) for

AUPRC. Two nominal paired AUPRC comparisons favored the atlas over the lightweight 3D ViT $~ ( \Delta ~ = ~ - 0 . 0 0 0 6 2 ,$ 95% $\mathrm { C I } \left[ - 0 . 0 0 1 6 3 , - 0 . 0 0 0 0 2 \right] )$ and fine-tuned SwinUNETR $( \Delta ~ = ~ - 0 . 0 0 0 7 1$ , 95% CI $\left[ - 0 . 0 0 1 6 9 , \ - 0 . 0 0 0 0 1 \right] )$ . These results suggest that population-level spatial patterns provide useful information for forecasting future lesion locations, while baseline FLAIR-based deep learning models did not demonstrate a reliable improvement in pooled discrimination over the population-prior atlas.

Table 1: Pooled voxel-level AUROC/AUPRC from 5-fold cross-validation across 39 subjects, with 95% CIs estimated from 1,000 patient-level bootstrap resamples. Best values in each column are shown in bold.
<table><tr><td>Method</td><td>AUROC [95% CI]</td><td> $\mathsf { A U P R C } ( \times 1 0 ^ { - 3 } )$  [95% CI]</td></tr><tr><td>Attn+ResBlock+VAE</td><td>0.643 [0.465, 0.867]</td><td>0.82 [0.28, 2.54]</td></tr><tr><td>3D U-Net</td><td>0.680 [0.503, 0.866]</td><td>0.78 [0.30, 2.01]</td></tr><tr><td>SwinUNETR (frozen)</td><td>0.682 [0.645, 0.755]</td><td>0.54 [0.25, 0.98]</td></tr><tr><td>SwinUNETR (fine-tune)</td><td>0.677 [0.619, 0.759]</td><td>0.54 [0.24, 0.98]</td></tr><tr><td>Lightweight 3D ViT</td><td>0.681 [0.548, 0.814]</td><td>0.61 [0.21, 1.33]</td></tr><tr><td>Population atlas</td><td>0.747 [0.682, 0.777]</td><td>1.24 [0.31, 2.63]</td></tr></table>

Within-patient localization. Fig. 2 shows Recall@Top-K% across the 28 lesion-positive subjects. At $K \ = \ 2 0 \%$ mean voxel-level recall was 0.671 for Attn+ResBlock+VAE and 0.660 for the U-Net, compared with 0.572 for the atlas; the ViT and both SwinUNETR variants reached 0.481–0.529. Both the U-Net and Attn+ResBlock+VAE also achieved higher mean recall than the atlas at K = 1%, 5%, and 10%. However, uncertainty remained substantial: among the 40 paired recall comparisons, six 95% CIs excluded zero (four favoring the U-Net or Attn+ResBlock+VAE over the atlas and two favoring the atlas over the ViT), while the remaining 34 included zero. These results suggest that baseline FLAIR may contain patient-specific information useful for localizing future lesions, but it remains uncertain whether this improves localization beyond the population-prior atlas in this small MSSEG-2 cohort.

## 3.3. Qualitative Results

Fig. 3 illustrates three localization outcomes on the slice containing the most new-lesion voxels for each subject: a hit (057), a partial hit (021), and a miss (094). The risk maps indicate relative lesion risk rather than calibrated probabilities. In the hit case, the deep learning models assigned high risk to the new lesions or nearby regions; in the partial-hit case, they highlighted only one of two lesions; and in the miss case, they failed to localize the lesion on the displayed slice.

## 3.4. Discussion

No deep learning model reliably outperformed the populationprior atlas on pooled voxel-level AUROC or AUPRC. The atlas achieved the highest AUROC (0.747 versus 0.643–0.682 for the models), suggesting that population-level spatial patterns provide a strong baseline for forecasting future MS lesion locations. However, the 3D U-Net and Attn+ResBlock+VAE achieved higher mean voxel-level within-patient Recall@Top-K% than the atlas at all evaluated thresholds $( K \in \{ 1 , 5 , 1 0 , 2 0 \} )$ At K = 20%, recall was 0.660 for the U-Net and 0.671 for Attn+ResBlock+VAE, compared with 0.572 for the atlas. Most paired CIs included zero. These results do not establish a reliable improvement in patient-specific lesion localization over the atlas. The ViT and SwinUNETR variants showed no consistent improvement over the atlas.

![](images/5036d7698d72b004036b0d622397d5c3a41afaa109212001980f05f33093e3bb.jpg)

(b) Lesion-level Recall@Top-K%  
![](images/df5fe324f3a154feebdf307e9894fe2654c2e061f559435a3acebc3ef813c066.jpg)  
Fig. 2: Recall@Top-K% across methods: (a) voxel-level recall averaged over 28 lesion-positive subjects and (b) lesionlevel recall pooled over 147 lesions. Error bars indicate 95% patient-level bootstrap CIs for each method.

The higher mean voxel-level within-patient recall of the 3D U-Net and Attn+ResBlock+VAE may partly reflect preexisting lesions visible on baseline FLAIR, which the atlas does not explicitly model, rather than subtle changes in normal-appearing tissue. The small study cohort (39 subjects from MSSEG-2, including 28 with new lesions) limits statistical precision and contributes to wide CIs. In addition, the atlas was constructed using approximately 31 subjects per fold and could benefit from a larger reference cohort. These findings suggest that baseline FLAIR may contain patientspecific predictive information, but whether this information improves lesion forecasting beyond the population-prior atlas remains uncertain in this small cohort.

## 4. CONCLUSION

We introduce FORESEE-MS, an evaluation protocol to examine whether baseline FLAIR MRI can forecast future MS lesion locations beyond population-level spatial patterns. We evaluated two complementary aspects of forecasting performance: pooled voxel-level discrimination (AUROC and

![](images/2ec9ab7de490a6cab2072681a18702025d4dd797b7bc44ce64b146b749ba2da6.jpg)

(a) Subject 057: Hit (successful localization)  
![](images/1f6b089032fd8ff85703c355fdc2a7356558c389e893a9f0b76408431769ca60.jpg)

(b) Subject 021: Partial hit (partial localization)  
![](images/67e3ce25217cc0ab281f1938de93d9b97f55a20a73a680254cc9f311ea7e5d2d.jpg)  
(c) Subject 094: Miss (unsuccessful localization)  
Fig. 3: Qualitative comparison of predicted risk maps for three MSSEG-2 subjects. (a) Subject 057 (hit): deep learning models assign high risk around a ring-shaped FLAIR hyperintensity adjacent to new lesions, while the atlas produces smaller periventricular hotspots. (b) Subject 021 (partial hit): most methods highlight the smaller lesion but miss the larger lesion above it. (c) Subject 094 (miss): no method places a hotspot directly on the lesion in the displayed slice, although some models assign high risk to nearby regions. In each map, blue to red indicates low to high relative risk within the brain.

AUPRC) and within-patient lesion localization (Recall@Top-K%). The population-prior atlas achieved the highest pooled voxel-level AUROC and AUPRC, with no deep learning model reliably outperforming it on either metric. However, the 3D U-Net and Attn+ResBlock+VAE achieved higher mean voxel-level within-patient recall than the atlas at all evaluated thresholds $( K ~ \in ~ \{ 1 , 5 , 1 0 , 2 0 \} )$ ), although most paired confidence intervals for the differences included zero. The ViT and SwinUNETR variants showed no consistent improvement over the atlas. These findings suggest that baseline FLAIR may contain patient-specific predictive information, but whether this information improves lesion forecasting beyond the population-prior atlas remains uncertain given the small MSSEG-2 cohort. Future work will investigate baseline lesion masks and atlas-informed models using larger multi-site cohorts.

## 5. COMPLIANCE WITH ETHICAL STANDARDS

This study used retrospective, de-identified MRI data from the MSSEG-2 dataset, derived from the OFSEP cohort and distributed through the Shanoir platform under a Data Use Agreement. No additional ethical approval was required for this secondary analysis of de-identified data.

## 6. REFERENCES

[1] Daniel S. Reich, Claudia F. Lucchinetti, and Peter A. Calabresi, “Multiple sclerosis,” New England Journal ofMedicine, vol. 378, pp. 169–180, 2018.

[2] Olivier Commowick, Fred´ eric Cervenansky, Franc¸ois´ Cotton, and Michel Dojat, “MSSEG-2 challenge proceedings: Multiple sclerosis new lesions segmentation challenge using a data management and processing infrastructure,” in MICCAI 2021 Challenge, 2021.

[3] Aaron Carass, Snehashis Roy, Amod Jog, Jennifer L. Cuzzocreo, Elizabeth Magrath, Adrian Gherman, Julia Button, James Nguyen, Ferran Prados, Carole H. Sudre, et al., “Longitudinal multiple sclerosis lesion segmentation: Resource and challenge,” NeuroImage, vol. 148, pp. 77–102, 2017.

[4] C. Elliott, P. Momayyezsiahkal, D. L. Arnold, et al., “Abnormalities in normal-appearing white matter from which multiple sclerosis lesions arise,” Brain Communications, vol. 3, no. 3, pp. fcab176, 2021.

[5] Joshua Durso-Finley, Jean-Pierre R. Falet, Brennan Nichyporuk, Douglas L. Arnold, and Tal Arbel, “Personalized prediction of future lesion activity and treatment effect in multiple sclerosis from baseline MRI,” arXiv preprint arXiv:2204.01702, 2022.

[6] Gian Mario Favero, Ge Ya Luo, Nima Fathi, Justin Szeto, Douglas L. Arnold, Brennan Nichyporuk, Chris Pal, and Tal Arbel, “Spatio-temporal conditional diffusion models for forecasting future multiple sclerosis lesion masks conditioned on treatments,” arXiv preprint arXiv:2508.07006, 2025.

[7] Brendan S. Kelly, Prateek Mathur, Gerard McGuinness, Henry Dillon, Edward H. Lee, Kristen W. Yeom, Aonghus Lawlor, and Ronan P. Killeen, “A radiomic “warning sign” of progression on brain MRI in individuals with MS,” American Journal of Neuroradiology, vol. 45, no. 2, pp. 236–243, 2024.

[8] Vladimir Fonov, Alan C. Evans, Kelly Botteron, C. Robert Almli, Robert C. McKinstry, D. Louis Collins, and Brain Development Cooperative Group, “Unbiased

average age-appropriate atlases for pediatric studies,” NeuroImage, vol. 54, no. 1, 2011.

[9] Brian B. Avants, Charles L. Epstein, Murray Grossman, and James C. Gee, “Symmetric diffeomorphic image registration with cross-correlation: Evaluating automated labeling of elderly and neurodegenerative brain,” Medical Image Analysis, vol. 12, no. 1, pp. 26–41, 2008.

[10] Ozg<sup>¨</sup> un C¸ ic¸ek, Ahmed Abdulkadir, Soeren S. Lienkamp,¨ Thomas Brox, and Olaf Ronneberger, “3D U-Net: Learning dense volumetric segmentation from sparse annotation,” in Medical Image Computing and Computer-Assisted Intervention (MICCAI). 2016, vol. 9901 of LNCS, pp. 424–432, Springer.

[11] Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y. Hammerla, Bernhard Kainz, Ben Glocker, and Daniel Rueckert, “Attention U-Net: Learning where to look for the pancreas,” in Medical Imaging with Deep Learning (MIDL), 2018, arXiv:1804.03999.

[12] Andriy Myronenko, “3D MRI brain tumor segmentation using autoencoder regularization,” in MICCAI BrainLes Workshop, 2018, arXiv:1810.11654.

[13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in International Conference on Learning Representations (ICLR), 2021.

[14] Yucheng Tang, Dong Yang, Wenqi Li, Holger R. Roth, Bennett Landman, Daguang Xu, Vishwesh Nath, and Ali Hatamizadeh, “Self-supervised pre-training of Swin transformers for 3D medical image analysis,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 20730–20740.

[15] Joseph Cox, Peng Liu, Skylar E. Stolte, Yunchao Yang, Kang Liu, Kyle B. See, Huiwen Ju, and Ruogu Fang, “BrainSegFounder: Towards 3D foundation models for neuroimage segmentation,” Medical Image Analysis, vol. 97, pp. 103301, 2024.

[16] Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, and Ali Gholipour, “Tversky loss function for image segmentation using 3D fully convolutional deep networks,” in Machine Learning in Medical Imaging (MLMI). 2017, Springer, arXiv:1706.05721.