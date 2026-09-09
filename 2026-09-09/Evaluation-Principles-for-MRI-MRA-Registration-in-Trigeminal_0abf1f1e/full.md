# Evaluation Principles for MRI–MRA Registration in Trigeminal Neuralgia: An ROI-Centered Neurovascular Benchmark

Xupeng Zhang<sup>1</sup>, Xihang Wang<sup>2</sup>, Michael Xie<sup>2</sup>, Haoyuan Liang<sup>3</sup>, Hau Ern Lien<sup>3</sup>, Oishika Das<sup>2</sup>, James Feghali<sup>2</sup>, Risheng Xu<sup>2</sup>, and Peirong Liu<sup>1,∗</sup>

<sup>1</sup>Department of Electrical and Computer Engineering and Data Science and AI Institute, Johns Hopkins University, Baltimore, MD 21218, USA <sup>2</sup>Department of Neurosurgery, Johns Hopkins University School of Medicine, Baltimore, MD 21287, USA <sup>3</sup>Department of Biomedical Engineering, Johns Hopkins University, Baltimore, MD 21287, USA <sup>∗</sup>Corresponding author: Peirong Liu (email: peirong@jhu.edu)

Abstract. Preoperative evaluation of trigeminal neuralgia (TN) often requires joint interpretation of structural MRI, which depicts the trigeminal nerve and surrounding cisternal anatomy, and time-of-flight MRA, which highlights vascular structures. Although MRI–MRA fusion is clinically attractive for visualizing neurovascular compression, this task is poorly captured by conventional whole-brain registration evaluation because the clinically relevant target is a small trigeminal ROI, vessel annotations are partial and clinically focused, local TOF-MRA contrast is variable, and field-of-view mismatch can limit deformable alignment. We formulate TN MRI–MRA fusion as an ROI-centered neurovascular registration-evaluation problem and construct a benchmark from 149 patients with clinician-annotated bilateral trigeminal ROIs. Six representative registration pipelines were evaluated using local image-based metrics, segmentation-derived vessel-localization metrics, prediction-volume analysis, and contrast- and FOV-stratified comparisons. Conventional evaluation summaries were often misleading: local image similarity, vessel-background separability, and downstream vessel localization did not co-rank methods; one-sided vessel distances were strongly affected by predicted vessel extent under partial annotations; and local MRA contrast determined when vessel-separability metrics were informative. Deformable refinement provided only a small, FOV-dependent benefit over affine alignment, while reader review showed that locally favorable vessel distances could coexist with globally implausible registrations. These findings indicate that TN MRI–MRA registration should be evaluated as a local vessel-aware, contrast-sensitive, and FOV-aware visualization task rather than as generic multimodal brain registration. Our code is publicly available at https://github.com/jhuldr/TN-Reg-Benchmark.

Keywords: Trigeminal neuralgia, MRI-MRA registration, neurovascular compression, vessel segmentation.

## 1. Introduction

Trigeminal neuralgia (TN) is a debilitating facial pain disorder commonly caused by neurovascular compression (NVC), in which a blood vessel contacts or compresses the trigeminal nerve along its cisternal segment. Preoperative evaluation relies on skull-base imaging that visualizes both the trigeminal nerve and surrounding vasculature. Structural MRI, particularly CISS-type imaging, depicts the trigeminal nerve and adjacent cisternal anatomy, whereas time-of-flight magnetic resonance angiography (TOF-MRA) highlights vascular structures. In current clinical workflows, these modalities are often reviewed side-by-side rather than spatially integrated, requiring clinicians to synthesize complementary anatomical information during interpretation. Accurate MRI-MRA co-registration could therefore support multimodal visualization of nerve-vessel relationships. However, TN MRI-MRA fusion is not a generic wholebrain registration problem: the clinically relevant endpoint is local neurovascular interpretability within a small trigeminal region of interest (ROI), and conventional registration summaries may not reflect whether the fused images are useful for evaluating NVC.

## 1.1. MRI–MRA Fusion as a Local Neurovascular Task

Structural MRI provides high-resolution visualization of the trigeminal nerve and surrounding cisternal anatomy, including vessel-related signal voids near the nerve [1]. This supports visual identification of NVC and segmentation-based assessment of NVC morphology and pain-outcome modeling [2, 3]. However, structural MRI alone has limited nerve–vessel contrast, incomplete vascular conspicuity, and limited vessel-type information. TOF-MRA complementarily emphasizes vascular structures and can help distinguish vessels from surrounding soft tissue. Because TOF-MRA preferentially depicts arterial flow, TOF-visible vessels may also provide indirect information about vascular identity, although TOF visibility alone is not definitive.

The goal of TN’s MRI–MRA fusion is therefore not simply whole-brain alignment, but spatial integration of vascular MRA with structural MRI near the trigeminal nerve. A registration that appears acceptable globally may still fail within the trigeminal ROI, while a favorable local vessel metric may be misleading if global alignment is implausible. This motivates ROI-centered evaluation focused on local neurovascular interpretability rather than whole-volume similarity alone.

## 1.2. Clinical Use Case

We target preoperative evaluation for microvascular decompression (MVD), which treats TN by separating contacting vessel from the trigeminal nerve. Accurate visualization of the nerve and candidate offending vessels is clinically important as NVC morphology and vessel type are associated with postoperative pain outcomes, including contact location, contact severity or surface area, and arterial versus venous compression [3–6]. Spatially accurate MRI–MRA fusion may therefore support morphologic assessment of nerve–vessel contact within the trigeminal ROI. The immediate technical goal is not to determine vessel type or predict outcome, but to produce and evaluate anatomically plausible multimodal visualization.

## 1.3. Why Conventional Registration Evaluation Is Insufficient

Although MRI and TOF-MRA play complementary roles in TN evaluation, MRI–MRA co-registration remains underexplored in TN-specific applications. Prior TN studies often review modalities separately or demonstrate fusion qualitatively, while most registration benchmarks target generic brain alignment rather than clinician-defined trigeminal ROIs. TN MRI–MRA fusion challenges conventional evaluation in four ways: the endpoint is a small local ROI rather than the full brain; vessel annotations are partial and clinically focused, making onesided distances sensitive to predicted vessel volume; local TOF-MRA conspicuity varies, limiting vessel-separability metrics in low-contrast ROIs; and MRI–MRA FOV mismatch can limit deformable refinement, increasing the importance of robust affine initialization and quality control.

## 1.4. Related Work

Registration methodology. Classical deformable registration relies on similarity-driven optimization with explicit transformation models. ANTs [7–9] remains a widely used standard for affine and diffeomorphic registration. FireANTs [10] accel erates diffeomorphic matching on GPU, and ConvexAdam [11] combines discrete and continuous optimization but assumes approximate affine alignment. Learning-based methods such as EasyReg [12] and SynthMorph [13] aim to improve robustness and inference speed. Downstream vessel extraction can provide task-specific localization readouts; here, we use VesselFM [14] as the segmentation component. These methods are typically evaluated on whole-brain alignment tasks with adequate anatomical overlap, and their reliability for local TN visualization under FOV mismatch, variable MRA conspicuity, and partial vessel annotations remains unclear.

Clinical imaging for TN and NVC. Preoperative MRI and MRA are central to TN evaluation, but standardized objective NVC grading remains challenging, and NVC identification alone is not a complete surrogate for symptoms or recurrence [15]. NVC severity, compression location, nerve atrophy, and vessel type have been associated with clinical presentation or MVD outcomes [4–6, 16–20]. Prior studies have demonstrated fusion of high-resolution structural MRI with TOF-MRA for posterior-fossa neurovascular visualization and surgical planning [21–29], including ultra-high-field multimodal MRI in secondary TN [30]. These studies motivate multimodal visualization, but do not define how registration should be evaluated when the endpoint is local vessel interpretability in a clinician-defined trigeminal ROI.

Table 1. Cohort flow from initial paired scans to final analysis subsets.
<table><tr><td>Analysis step</td><td>Patients</td><td>ROIs</td><td>Excluded</td></tr><tr><td>Initial paired scans</td><td>159</td><td></td><td></td></tr><tr><td>Preprocessed pairs</td><td>153</td><td></td><td>6 (preprocessing)</td></tr><tr><td>Final analysis cohort</td><td>149</td><td>298</td><td>4 (MRA unpaired)</td></tr><tr><td>Registration outputs</td><td>149</td><td>298</td><td>0</td></tr><tr><td>Contrast-stratified</td><td>149</td><td>286</td><td>12 (empty SyN ROI)</td></tr><tr><td>FOV paired</td><td>149</td><td>223</td><td>75 (empty/out-of-FOV)</td></tr></table>

## 1.5. Contributions

This work makes four contributions. First, we formulate TN MRI-MRA fusion as an ROI-centered neurovascular registration-evaluation problem rather than a generic wholebrain alignment task. Second, we assemble a clinical cohort of 149 TN patients with clinician-annotated bilateral trigeminal ROIs and link whole-brain registration outputs to local vessel-focused evaluation. Third, we evaluate six representative registration pipelines using complementary image-based metrics and segmentation-derived downstream localization metrics, explicitly separating local vessel separability, downstream vessel extraction, and global registration plausibility. Fourth, through predicted-volume, contrast-stratified, FOV-stratified, and reader-based analyses, we identify practical confounds missed by conventional evaluation and define evaluation principles for future TN-specific MRI-MRA registration.

## 2. Materials and Methods

## 2.1. Clinical Cohort

We retrospectively studied a single-institution cohort of 149 TN patients (298 bilateral trigeminal ROIs) who underwent diagnostic structural MRI and TOF-MRA between October 2019 and December 2022 and subsequently underwent MVD. Inclusion required paired structural MRI and TOF-MRA from the same preoperative imaging encounter and clinician-provided trigeminal annotations. The final cohort had a mean age of 56.6 ± 14.0 years (85 female / 64 male; 88 right- / 61 left-sided presentation; 7 patients [4.7%] with multiple sclerosis). The cohort flow from 159 initial paired scans to 149 analyzable patients, and the per-analysis subset counts, are summarized in Table 1. The study was approved by the Johns Hopkins Medicine Institutional Review Board (IRB00338945); informed consent was waived given the retrospective design.

## 2.2. Imaging Data and ROI Annotations

Structural MRI used a 3D CISS-type sequence (∼0.6mm isotropic), and TOF-MRA used a 3D multi-slab sequence

Overview of the MRI-MRA Registration and ROl-Centered Validation Pipeline

![](images/bcbdf825338af41e09554950b162614083c29787baa04610f7d734bc047867fe.jpg)  
Fig. 1. Overview of the MRI–MRA registration and ROI-centered evaluation pipeline. MRI provides the anatomical reference and clinician-annotated trigeminal ROI; MRA is registered into MRI space. Local ROIs are extracted from the warped MRA for segmentation-based vessel validation. Cyan: clinician-labeled vessel segment; red: predicted vessel; green: overlap

(∼0.26mm in-plane, ∼0.5mm slice). The cohort was predominantly Siemens 3T: 98.0% Siemens; structural MRI included 138/149 scans at 3T and 11/149 at 1.5T; TOF-MRA included 137/149 scans at 3T and 12/149 at 1.5T.

Local annotations were available within 48 × 48 × 48 ROI crops centered on the cisternal trigeminal nerve, sampled on a 0.47mm isotropic grid. Each ROI contained per-voxel labels for the trigeminal nerve and vessel structures, and all patients had both ipsilateral and contralateral ROIs annotated. Annotations were drawn on structural MRI alone, independent of MRA and registration outputs, using a two-pass workflow consisting of initial segmentation followed by independent review. Vessel labels marked only the segment considered clinically relevant to the trigeminal nerve, not an exhaustive vascular tree. This partial-annotation property is central to metric interpretation and is analyzed quantitatively in Section 3.4.

## 2.3. Preprocessing and Registration Methods

All images were converted from DICOM to NIfTI and resampled to RAS orientation. Brain masks were estimated to suppress non-brain background and support automatic cropping. MRI served as the fixed image and MRA as the moving image; each method resampled MRA into MRI space.

We evaluated six registration methods: ANTs Affine [8, 9], ANTs SyN [7], ConvexAdam [11], FireANTs [10], EasyReg [12], and SynthMorph [13]. In figures, they are abbreviated as Affine, SyN, CvxAd, Fire, Easy, and Synth. Table 2 summarizes their categories and initialization protocols.

Pre-alignment protocol. ANTs SyN, EasyReg, and Synth-Morph include affine alignment within their workflows. ConvexAdam and FireANTs were supplied with the ANTs Affine output as a unified external pre-alignment, because they were not treated here as standalone cross-modality affine pipelines for MRI–MRA. Per-method configuration details, including similarity metrics, multi-resolution schedules, descriptors, normalization, software versions, and hardware, are provided in the released code repository.

## 2.4. ROI-Centered Segmentation-Based Evaluation

After whole-brain registration, the MRA was linked to the annotated trigeminal ROI using physical-space centroids and a fixed local sampling grid (Fig. 1). Metrics were computed within the matched 48<sup>3</sup> ROI at 0.47 mm isotropic resolution.

For segmentation-based vessel evaluation, we applied VesselFM [14] (v1.0, dyn\_unet\_base, no retraining) to each warped MRA volume in MRI space, then cropped the predicted whole-brain vessel mask to the matched trigeminal ROI. The same inference configuration was used for all warped MRA outputs, so cross-method differences reflect changes induced by registration rather than changes in the segmentation pipeline. Vessel-proximity metrics required at least one predicted vessel voxel in the ROI; observations with empty vessel predictions were excluded only from those metrics. Because predicted vessel extent varied substantially across methods, we tracked predicted vessel volume as an auxiliary quantity and analyzed its effect on distance metrics in Section 3.4. Full preprocessing, inference, binarization, and postprocessing details are provided in the released code repository.

Table 2. Summary of the registration methods evaluated in this study.
<table><tr><td>Method</td><td>Category</td><td>Initialization</td><td>GPU</td><td>Notes</td></tr><tr><td>ANTs Affine</td><td>Affine</td><td>ANTs affine</td><td>No</td><td>Global affine baseline.</td></tr><tr><td>ANTs SyN</td><td>Classical deformable</td><td>Internal (ANTs affine + SyN)</td><td>No</td><td>Diffeomorphic refinement after affine.</td></tr><tr><td>ConvexAdam</td><td>Optimization-based</td><td>External ANTs affine</td><td>Yes</td><td>Requires affine pre-alignment.</td></tr><tr><td>FireANTs</td><td>Optimization-based</td><td>External (ANTs affine + FireANTs)</td><td>Yes</td><td>GPU-accelerated; not standalone affine.</td></tr><tr><td>EasyReg</td><td>Learning-based</td><td>Internal</td><td>No</td><td>Segmentation-guided contrast-agnostic.</td></tr><tr><td>SynthMorph</td><td>Learning-based</td><td>Internal</td><td>No</td><td>Synthetic-training contrast-invariant.</td></tr></table>

## 2.5. Stratified Analyses

Contrast stratification. For each ROI, we computed a local vessel-to-background intensity contrast score on warped MRA,

$$
\mathrm { c o n t r a s t } = \bar { I } _ { \mathrm { M R A } } ( V _ { \mathrm { a n n } } ) / \bar { I } _ { \mathrm { M R A } } ( B ) ,\tag{1}
$$

$V _ { \mathrm { a n n } }$ and B denote the clinician-labeled vessel voxels and the background voxels (labeled neither vessel nor trigeminal nerve), respectively, within ROI. The contrast score was computed on the ANTs SyN warped MRA and used as a canonical per-ROI reference. ROIs were stratified into Low (contrast ≤ 1.1, N = 181, 63%), Mid (1.1 < contrast ≤ 1.3, N = 66, 23%), and High (contrast > 1.3, N = 39, 14%) tiers; 12 of 298 ROIs were excluded as ANTs SyN deformable warp placed the TNcentered sampling grid outside the warped MRA coverage.

FOV stratification. MRI-MRA FOV compatibility was characterized using a per-case worst-axis coverage ratio,

$$
r _ { \mathrm { m i n } } = \operatorname* { m i n } _ { a \in \{ x , y , z \} } \frac { \mathrm { F O V } _ { a } ^ { \mathrm { M R I } } } { \mathrm { F O V } _ { a } ^ { \mathrm { M R A } } } , \quad \mathrm { F O V } _ { a } = \mathrm { s h a p e } _ { a } \times \mathrm { s p a c i n g } _ { a } .\tag{2}
$$

Cases with $r _ { \operatorname* { m i n } } \leq 0 . 7 6$ (the cohort’s 20th percentile) were classified as Bad FOV, and the remainder as Good FOV. The Affine-versus-SyN paired analysis included 177 Good-FOV and 46 Bad-FOV ROIs, after excluding 75 ROIs with empty vessel predictions or out-of-coverage warped regions. For descriptive cross-tabulation, we defined image-level high-AUC as $\mathrm { A U C } _ { \mathrm { S y N } } > 0 . 6 0$ and downstream miss as $d _ { \mathrm { a n n \to P r e d } } > 2 \mathrm { m m }$ Volume-distance coupling was assessed by Spearman correlation, and paired Affine-versus-SyN comparisons used the Wilcoxon signed-rank test. Confidence intervals were estimated by patient-level bootstrap resampling with 1000 resamples to account for bilateral ROIs.

## 2.6. Blinded Reader Study

To relate ROI-centered metrics to clinical interpretability and to probe whether low local TOF-MRA contrast may reflect venous-offender biology, we conducted a blinded reader study on 100 stratified trigeminal ROIs. The sample included 50 Lowcontrast ROIs, 20 Mid-contrast ROIs with downstream vessellocalization misses, 20 High-contrast hit controls, and 10 harder Mid/High hit controls. ROIs were randomized as R001–R100, and the manifest linking review ID to patient identity, side, and sampling stratum was withheld during review.

Two attending neurosurgeons assessed each ROI jointly in consensus using the structural MRI, original MRA, ANTs-SyN warped MRA, and the single-side ROI label. For each ROI, they recorded clinical evaluability, compressive vessel type, and confidence. An ROI was considered evaluable if the ANTs-SyN warped MRA was anatomically consistent with the structural MRI and usable for assessing the neurovascular relationship; globally implausible overlays were rated non-evaluable. For evaluable ROIs, vessel type was recorded as artery, vein, mixed, or no definite vessel. Because TOF-MRA preferentially depicts arterial flow and may not show slow venous flow, vessel type was assigned by tracing candidate vessels across adjacent slices and assessing continuity with named posterior-fossa vessels, rather than by TOF signal alone. Arterial candidates were assessed for continuity with the superior cerebellar or anterior inferior cerebellar artery, whereas vessels lacking arterial continuity and following a typical petrosal venous course were classified as venous. SynthMorph outputs were additionally tracked as a method-specific registration-quality comparison in a separate field, so SynthMorph-specific failures did not affect the primary ANTs-SyN evaluability labels.

## 3. Experimental Results

## 3.1. Evaluation Metrics

We evaluated registration outputs within the clinician-defined $4 8 ^ { 3 }$ trigeminal ROI using complementary metrics that capture different aspects of the TN MRI–MRA fusion task. Let $V _ { \mathrm { a n n } }$ denote the clinician-labeled vessel voxels within the ROI, $V _ { \mathrm { P r e d } }$ the segmentation-predicted vessel voxels, and B the background voxels labeled as neither vessel nor trigeminal nerve. The annotation $V _ { \mathrm { a n n } }$ is a clinically focused partial vessel segment rather than an exhaustive vascular tree, which is central to the interpretation of the distance metrics below. Let $I _ { \mathrm { M R A } } ( \nu )$ denote the warped MRA intensity at voxel v.

Image-based metrics. Vessel AUC treats the registered MRA intensity as a score for separating clinician-labeled vessel voxels from background voxels within the ROI:

$$
\mathrm { A U C } = P \big ( I _ { \mathrm { M R A } } ( \nu _ { + } ) > I _ { \mathrm { M R A } } ( \nu _ { - } ) \big ) , \quad \nu _ { + } \in V _ { \mathrm { a n n } } , \nu _ { - } \in B .\tag{3}
$$

A value of 0.5 corresponds to chance-level separability. ROI NMI measures local multimodal intensity correspondence between the fixed MRI and warped MRA:

$$
\mathrm { N M I } ( X , Y ) = ( H ( X ) + H ( Y ) ) / H ( X , Y ) ,\tag{4}
$$

where X and Y denote the intensity distributions of the fixed MRI and warped MRA within the ROI, respectively.

Vessel-proximity metrics. To evaluate downstream vessel localization, we computed mean nearest-neighbor distances between the annotation and post-registration vessel prediction:

$$
d _ { \mathrm { a n n } \to \mathrm { P r e d } } = 1 / | V _ { \mathrm { a n n } } | \sum _ { x \in V _ { \mathrm { a n n } } } \operatorname* { m i n } _ { y \in V _ { \mathrm { P r e d } } } \| x - y \| _ { 2 } ,\tag{5}
$$

$$
d _ { \mathrm { P r e d \to a n n } } = 1 / | V _ { \mathrm { P r e d } } | \sum _ { y \in V _ { \mathrm { P r e d } } } \operatorname* { m i n } _ { x \in V _ { \mathrm { a n n } } } \| x - y \| _ { 2 } ,\tag{6}
$$

$$
d _ { \mathrm { s y m } } = \textstyle { \frac { 1 } { 2 } } ( d _ { \mathrm { a n n \to P r e d } } + d _ { \mathrm { P r e d \to a n n } } ) .\tag{7}
$$

Because the annotation is partial, $d _ { \mathrm { a n n \to P r e d } }$ measures whether the clinically annotated segment is covered by the predicted vessel tree, whereas $d _ { \mathrm { P r e d } \to \mathrm { a n n } }$ penalizes predicted vessels that extend beyond the annotated segment. We therefore report both directional distances, their symmetric average, and the predicted vessel volume $| V _ { \mathrm { P r e d } } |$

## 3.2. Cohort and Analysis Subsets

The final cohort included 149 TN patients and 298 bilateral trigeminal ROIs. All registration methods produced wholebrain warped MRA outputs for all patients. Image-based metrics were computed on all 298 ROIs. Contrast-stratified analysis used 286 ROIs with valid ANTs SyN-based contrast measurements; 12 ROIs were excluded because the SyN-warped sampling grid fell outside the warped MRA coverage. Vesselproximity metrics were computed on per-method subsets with non-empty post-registration vessel predictions, ranging from 243 to 277 ROIs depending on the registration method. The FOV-stratified Affine-versus-SyN paired analysis included 223 ROIs, consisting of 177 Good-FOV and 46 Bad-FOV ROIs, after excluding cases with empty predicted vessel masks or out-of-coverage warped regions.

## 3.3. Conventional Metrics Reveal Discordant Rankings

We first compared the six registration methods using local image-based and downstream vessel-proximity summaries. For vessel-background intensity discrimination (Fig. 2A), EasyReg and SynthMorph achieved the highest median Vessel AUC values (0.578 and 0.581), then ANTs SyN (0.557), while ANTs Affine, ConvexAdam, and FireANTs were closer to chance. ROI NMI ranked methods differently (Fig. 2B): FireANTs was highest (1.072), followed by SynthMorph (1.056) and ANTs SyN (1.051), with ANTs Affine and ConvexAdam lowest. Thus, local vessel separability and local multimodal intensity correspondence captured different properties of the registered images rather than a single shared notion of registration quality.

Downstream vessel-proximity metrics showed an additional discordance (Fig. 3). Under $d _ { \mathrm { a n n } \to \mathrm { P r e d } } ,$ SynthMorph and Fire-ANTs appeared strongest, whereas under $d _ { \mathrm { P r e d } \to \mathrm { a n n } }$ , the ordering reversed and these methods produced the largest distances back to the annotation. This asymmetry coincided with large differences in predicted vessel volume, with EasyReg and SynthMorph producing substantially larger predicted vessel masks than others. Symmetric distance $d _ { \mathrm { s y m } }$ reduced but did not eliminate this directional dependence. Numerical values are reported in Table 3. These results show that image similarity, vessel separability, and one-sided vessel localization are not interchangeable measures of TN MRI-MRA registration quality.

![](images/557d202ede60042dd95e6020818ee50870df08515329751e39c101d6f206843f.jpg)

![](images/28f98f8477d997f0dd4ee02f64f2fa6b4d065180cd5bb1a58651054e1f795d33.jpg)  
Fig. 2. Local image-based metrics across registration methods. (A) Vessel AUC; dashed line indicates chance level (0.5). (B) ROI NMI. The two metrics do not co-rank methods, indicating that local vessel separability and local multimodal intensity correspondence capture different properties of the registered images. Numerical values are reported in Table 3.

![](images/aafd8a476d68e7b3583f76f22a6b3640f699809d0ddf3e2061b0e3d7506873a6.jpg)

![](images/c46840b8892f51f41fce95d5bba9e08b00434f8251e79c135caf67d57afca56a.jpg)

![](images/baff07e94ddbca7f0dde4aadfc821cff6d8716d7e70b1a935333c8d191068653.jpg)

![](images/889102ed2f0a4575bf619beaabf40045abba5d6aa86127837c5c808766fe07f1.jpg)  
Fig. 3. Vessel-proximity and prediction-volume metrics. (A) $d _ { \mathrm { a n n \to P r e d } } .$ (B) $d _ { \mathrm { P r e d } \to \mathrm { a n n } } .$ (C) $d _ { \mathrm { s y m } }$ . (D) Predicted vessel voxels $| V _ { \mathrm { P r e d } } |$ on a log scale. Directional distances rank methods differently, and these differences are coupled to predicted vessel volume. Numerical values are reported in Table 3.

## 3.4. Predicted Volume Confounds One-Sided Distance Metrics

To test the volume-confound suggested by the overall metrics, we examined predicted vessel volume against directional distance metrics across all method-ROI observations (Fig. 4). $d _ { \mathrm { a n n } \to \mathrm { P r e d } }$ showed a strong negative association with predicted volume (Spearman $\rho = - 0 . 5 3 , 9 5 \% \mathrm { C I } \mathrm { [ - 0 . 5 9 , - 0 . 4 7 ] ) }$ , consistent with the intuition that larger predicted vessel trees are more likely to cover the annotated segment. $d _ { \mathrm { P r e d } \to \mathrm { a n n } }$ showed only a negligible association $( \rho = + 0 . 0 8 $ , 95% CI: $[ + 0 . 0 1 , + 0 . 1 5 ] )$ , demonstrating that enlarging the prediction does not, on average, bring predicted voxels closer to the annotated vessel. $d _ { \mathrm { s y m } }$ showed an intermediate association $( \rho = - 0 . 3 2$ , 95% CI: $[ - 0 . 3 8 , - 0 . 2 5 ] )$ . These results indicate that, under partial clinical annotations and cross-method differences in prediction extent, one-sided vessel-proximity metrics can systematically favor methods that produce more voluminous vessel trees, independent of local geometric accuracy. Symmetric distance is less directionally biased than either one-sided distance, but it remains a segmentation-derived downstream localization metric rather than an independent registration-error measure. We therefore report it jointly with the two one-sided distances and prediction-volume statistics.

Table 3. Local image-based and vessel-proximity metrics across registration methods. Image-based metrics are computed on 298 ROIs. Vessel-proximity metrics are computed on per-method subsets with non-empty post-registration vessel predictions (N in the last column). Distances are in mm; volumes are in voxels. Values are median [Q1, Q3]. See Figs. 2 and 3 for distributions.
<table><tr><td rowspan="2">Method</td><td colspan="2">Image-based</td><td colspan="5">Vessel-proximity (segmentation-derived)</td></tr><tr><td>Vessel AUC</td><td>ROI NMI</td><td> $d _ { \mathrm { a n n \to P r e d } } ( \mathbf { m m } )$ </td><td> $d _ { \mathrm { P r e d } \to \mathrm { a n n } } \left( \mathbf { m m } \right)$ </td><td> $d _ { \mathrm { s y m } } \left( \mathbf { m m } \right)$ </td><td> $\left| V _ { \mathrm { P r e d } } \right| \left( \mathbf { v o x e l s } \right)$ </td><td>N</td></tr><tr><td>ANTs Affine</td><td>0.526 [0.432, 0.667]</td><td>1.009 [1.005, 1.021]</td><td>5.38 [3.23, 9.12]</td><td>7.22 [5.17, 9.35]</td><td>6.26 [4.71, 8.67]</td><td>1148 [388, 2227]</td><td>243</td></tr><tr><td>ANTs SyN</td><td>0.557 [0.476, 0.665]</td><td>1.051 [1.010, 1.078]</td><td>4.18 [1.54, 7.24]</td><td>8.60 [6.25, 10.58]</td><td>5.97 [4.67, 8.56]</td><td>1493 [803, 2643]</td><td>250</td></tr><tr><td>ConvexAdam</td><td>0.522 [0.396, 0.640]</td><td>1.009 [1.006, 1.012]</td><td>4.90 [3.10, 7.86]</td><td>6.16 [4.55, 8.46]</td><td>5.67 [4.09, 7.84]</td><td>1218 [624, 2039]</td><td>246</td></tr><tr><td>FireANTs</td><td>0.506 [0.399, 0.611]</td><td>1.072 [1.053, 1.084]</td><td>3.40 [1.29, 6.81]</td><td>9.61 [7.89, 11.64]</td><td>6.46 [4.98, 8.77]</td><td>1928 [1082, 2774]</td><td>277</td></tr><tr><td>EasyReg</td><td>0.578 [0.481, 0.724]</td><td>1.022 [1.015, 1.039]</td><td>4.36 [2.06, 7.30]</td><td>7.93 [6.12, 9.81]</td><td>6.10 [4.53, 8.19]</td><td>2849 [1529, 5318]</td><td>267</td></tr><tr><td>SynthMorph</td><td>0.581 [0.493, 0.698]</td><td>1.056 [1.037, 1.071]</td><td>2.88 [1.30, 5.45]</td><td>9.49 [8.00, 11.02]</td><td>6.23 [5.03, 7.62]</td><td>2876 [1868, 4000]</td><td>277</td></tr></table>

![](images/c51ce3f15eee228673dc46e51c0a6e558a6dd1b00c619266632382896f0368c4.jpg)

![](images/ee4126311b1433958b8db0160efd6680e848c22c23a01f75f0407a4c39e40417.jpg)

![](images/bf0d72dd37e36a36d80e02c7520b83276baa60219cf98af8fa2a5d7f879fefd6.jpg)  
Fig. 4. Coupling between predicted vessel volume and directional vessel-distance metrics. Each panel shows the 2D density of per-(method ROI) observations; red line is a linear regression on log-transformed volume, and ρ is the Spearman correlation. Annotation-to-prediction distance is strongly volume-dependent, whereas prediction-to-annotation distance is not.

## 3.5. Visibility Does Not Guarantee Downstream Localization

Fig. 5 illustrates three qualitative operating regimes that explain why TN MRI–MRA fusion requires separate evaluation axes. In a high-contrast, Good-FOV case (Fig. 5A), local vessel signal is conspicuous and segmentation-based prediction localizes the clinician-annotated vessel segment, representing the favorable regime in which image-level visibility and downstream localization agree. In a Mid-tier case (Fig. 5B), local vesselbackground separability remains measurable under ANTs SyN, yet the segmentation-based prediction is non-empty but spatially displaced from the annotation. This demonstrates that image-level vessel visibility does not necessarily imply successful model-based vessel localization. In a Low-contrast case with Good-FOV coverage (Fig. 5C), vessel-background separability is near chance and downstream localization fails. These examples motivate evaluating TN MRI-MRA registration along separate whole-brain, image-level, and downstream

localization axes.

## 3.6. Local Contrast Determines When Vessel AUC Is Informative

We next asked how Vessel AUC behaves across MRA contrast regimes. Fig. 6A shows that 181 ROIs (63%) were Low contrast, 66 (23%) Mid contrast, and 39 (14%) High contrast. Vessel AUC showed clear method separation in the Mid and High tiers; in the High tier, ANTs SyN achieved the highest median AUC (approximately 0.76). In the Low tier, Vessel AUC converged toward chance across methods, reflecting limited discriminative signal when local vessel-background contrast was weak. Because both the contrast score and Vessel AUC are derived from local MRA intensities sampled at clinicianlabeled vessel and background voxels, this stratification should be interpreted as an assessment of when Vessel AUC is informative, not as a causal analysis of acquisition quality. Thus, method separation was most interpretable in the Mid and High tiers.

Importantly, the Mid tier was not simply a failure regime. Within the SyN-defined Mid tier $( N = 6 6 )$ , ANTs SyN achieved a median Vessel AUC of 0.600 [IQR: 0.532-0.661], with 33/66 ROIs (50.0%, patient-level bootstrap 95% CI: 38.2-62.1%) exceeding $\mathrm { A U C } = 0 . 6 0$ . However, measurable image-level vessel separability did not guarantee downstream segmentationbased localization. Among 65 Mid tier ROIs with valid SyN vessel predictions, 36/65 ROIs (55.4%, 95% Wilson CI:

Fixed MRI Original MRA  
Affine  
SyN  
CvxAd  
Fire  
Easy  
Synth  
![](images/6f961acfcbf45de76f138f9415029dade2f1fe91d534789ea72700d2ce75a465.jpg)  
Fig. 5. Qualitative operating regimes for TN MRI-MRA registration across all six methods. Each panel (A-C) shows the fixed MRI, original MRA, and warped MRA outputs from ANTs Affine, ANTs SyN, ConvexAdam, FireANTs, EasyReg, and SynthMorph. Rows within each panel show the whole-brain slice (yellow box marks the trigeminal ROI), the matched 48 × 48 × 48 trigeminal ROI intensity, and the ROI overlay (cyan: clinician annotation; red: post-registration vessel prediction). (A) High-contrast, Good-FOV success case (Representative Case A, ipsilateral ROI): under ANTs SyN, $\mathrm { A U C } _ { \mathrm { S y N } } = 0 . 8 8$ and $d _ { \mathrm { a n n \to P r e d } } = 0$ .64mm. Most methods localize the annotated vessel segment, illustrating the favorable operating regime in which local MRA conspicuity and downstream vessel localization agree. (B) Mid-tier image-level success with downstream vessel-localization miss (Representative Case B, ipsilateral ROI): despite Good-FOV coverage and measurable image-level separability under ANTs SyN $( \mathrm { A U C } _ { \mathrm { S y N } } = 0 . 7 0 )$ , segmentation-based predictions are non-empty across methods but spatially displaced from the annotated segment $( d _ { \mathrm { a n n }  \mathrm { P r e d } } = 6 .$ 53mm under SyN), illustrating the gap between image-level vessel separability and model-level vessel extraction. (C) Low-contrast image-level failure (Representative Case C, contralateral ROI): although the whole-brain registrations place the trigeminal ROI in comparable anatomical locations, local vessel-background separability is near chance under ANTs SyN $( \mathrm { A U C } _ { \mathrm { S y N } } = 0 . 5 0$ contrast = 0.89) and the predicted vessel mask does not localize the annotated segment $( d _ { \mathrm { a n n \to P r e d } } = 6 . 8 9 \mathrm { m m } )$ . All distances are computed in the full 3D ROI, not only on the displayed axial slice. Together, these examples illustrate that whole-brain visual alignment, image-level vessel conspicuity, and downstream model-based vessel localization are related but non-interchangeable evaluation axes, and that this dissociation is not specific to any single registration method.

![](images/3f299b970f7bd829e5daeca6294ec80ea3ca681ff5c09746a173250e65aa0fef.jpg)

![](images/7d7504ec474479fb483e07f5a7bece8c0702bb0564ac81fc4fa0133634c23661.jpg)  
Fig. 6. Contrast-stratified analysis. (A) Distribution of ROIs across Low, Mid, and High contrast tiers. (B) Vessel AUC by method within each tier. The dashed line indicates chance-level vessel-background separability.

![](images/a4df712c0a9e2a5dd885c47d761e255df798945506b4df55fe10c344a5a62bad.jpg)

![](images/04b235e2bc9af2b4e3bb917ff764254444592c89a9aea23bd1c7d190c7848711.jpg)  
Fig. 7. Effect of FOV mismatch on deformable refinement. (A) Paired Affine→SyN trajectories; method medians are shown as filled squares. (B) Distribution of paired deltas $( \Delta = \mathrm { S y N - A f f u n e } )$ by FOV group; vertical dashed lines mark group medians.

43.3-66.8%) missed the annotation under a 2-mm criterion $( d _ { \mathrm { a n n \to P r e d } } > 2 \mathrm { m m } )$ . Even among ROIs with $\mathrm { A U C } _ { \mathrm { S y N } } > 0 . 6 0$ 16/33 (48.5%) missed the annotation; conversely, 16/36 Midtier miss cases (44.4%) still exceeded $\mathrm { A U C } = 0 . 6 0$ . Within the Mid tier, Vessel AUC and $d _ { \mathrm { a n n \to P r e d } }$ were not significantly associated (Spearman $\rho = - 0 . 1 2 , p = 0 . 3 5 )$ . These results separate image-level vessel separability from post-registration model-level vessel extraction.

## 3.7. FOV Mismatch Limits the Benefit of Deformable Refinement

Finally, we examined whether deformable refinement remained beneficial under MRI–MRA FOV mismatch. The paired Affineversus-SyN analysis included 223 ROIs: 177 Good-FOV and 46 Bad-FOV ROIs. In the Good-FOV subset, SyN produced a small but directionally consistent reduction in $d _ { \mathrm { a n n } \to \mathrm { P r e d } }$ relative to affine alignment (median paired $\Delta = - 0$ .18mm, Wilcoxon $p < 0 . 0 0 1$ ; patient-level bootstrap 95% CI for the median: $[ - 0 . 5 5 , + 0 . 0 1 ] \mathrm { m m } )$ . In the Bad-FOV subset, this benefit was not observed (median paired $\Delta = - 0 . 0 7 \mathrm { m m }$ , Wilcoxon $p = 0 . 3 2$ ; bootstrap 95% CI: $[ - 3 . 6 3 , + 0 . 6 5 ] \mathrm { m m } )$ , and individual trajectories moved in both directions (Fig. 7).

These results indicate that deformable refinement provides at most a small and FOV-dependent improvement over affine alignment in this clinical setting. The wide Bad-FOV interval reflects limited sample size and high case-to-case variability, suggesting attenuation rather than reversal of the deformableregistration benefit. Robust affine initialization remains essential, and deformable outputs should be interpreted in light of shared anatomical coverage rather than assumed to improve local neurovascular alignment uniformly.

## 4. Discussion

This retrospective ROI-centered benchmark shows that TN MRI-MRA registration should be treated as a task-specific local neurovascular visualization problem rather than a generic whole-brain registration problem.

Metric design under partial clinical annotations. No single metric captures registration quality comprehensively. Vessel AUC and ROI NMI rank methods differently; $d _ { \mathrm { a n n } }$ →Pred and $d _ { \mathrm { P r e d } \to \mathrm { a n n } }$ rank methods in opposite directions; and predicted vessel volume differs by more than an order of magnitude across methods. The volume-distance coupling analysis $( \rho = - 0 . 5 3$ versus $\rho = + 0 . 0 8 )$ makes clear that one-sided distance metrics can systematically favor methods with larger vessel predictions under partial clinical annotations. TN-focused benchmarks should therefore report Vessel AUC, ROI NMI, and symmetric vessel distance $d _ { \mathrm { s y m } }$ jointly with $| V _ { \mathrm { P r e d } } |$ , rather than ranking methods on a single one-sided criterion.

Image-level visibility versus model-level extraction. The Mid-tier analysis shows a gap between local image conspicuity and downstream vessel extraction. These ROIs were not uniformly poor-quality: under ANTs SyN, the median Vessel AUC was 0.600. Nevertheless, downstream localization remained unreliable. Even among Mid-tier ROIs with $\mathrm { A U C } _ { \mathrm { S y N } } > 0 . 6 0 .$ 48.5% missed the clinician-annotated vessel segment under the 2-mm criterion, and Vessel AUC was not significantly associated with $d _ { \mathrm { a n n } \to \mathrm { P r e d } }$ $( \rho = - 0 . 1 2$ $p = 0 . 3 5 ;$ ; Fig. 5B). Failures were driven by non-empty predictions that did not spatially match the annotated segment, rather than by empty masks. Thus, local vessel-background separability on the registered MRA is not interchangeable with successful downstream vessel extraction. Because vessel segmentation is applied after registration, segmentation-derived distances reflect the combined registration–segmentation pipeline rather than registration error alone. A future segment-once-then-warp analysis, in which vessels are extracted from the original MRA and then warped by each registration field, could further separate registration geometry from segmentation-model sensitivity.

Contrast-stratified evaluation as a methodological necessity. Vessel AUC measures local vessel-background separability in the warped MRA (Eq. (3)) and cannot distinguish methods when local contrast is intrinsically weak. In this cohort, the Low-contrast tier comprised 63% of ROIs and drove Vessel AUC toward chance for all methods, whereas clear method separation appeared in the Mid and High tiers. Thus, contrast stratification should not be interpreted causally as contrast being more important than method choice; rather, it identifies when Vessel AUC is informative. Reporting Vessel AUC without contrast stratification can mask genuine method differences in clinical cohorts dominated by low-contrast ROIs.

Robust affine alignment is a prerequisite. FOV-stratified paired comparisons showed that deformable refinement provided a statistically significant but small benefit over affine alignment in Good-FOV ROIs $( p < 0 . 0 0 1 )$ , while this benefit was attenuated in Bad-FOV ROIs $( p = 0 . 3 2 )$ . The Good-

Table 4. Qualitative method profile across the evaluated registration methods. Numerical values are reported in Table 3.
<table><tr><td>Method</td><td>Observed strength</td><td>Main trade-off</td></tr><tr><td>ANTs SyN</td><td>Most balanced across all metrics</td><td>Higher CPU runtime; batch-parallelizable</td></tr><tr><td>ConvexAdam</td><td>Lowest symmetric distance  $d _ { \mathrm { s y m } }$ </td><td>Requires affine pre-alignment</td></tr><tr><td>SynthMorph</td><td>Best one-sided dann→Pred; high Vessel</td><td>Large prediction volume inflates one-sided</td></tr><tr><td>EasyReg</td><td>AUC High Vessel AUC; fast</td><td>coverage Large, variable</td></tr><tr><td>FireANTs</td><td>learning-based inference Highest ROI NMI (local</td><td>prediction volume Highest reverse distance</td></tr><tr><td>ANTs Affine</td><td>intensity match) Reliable baseline under FOV mismatch</td><td> $d _ { \mathrm { P r e d } \to \mathrm { a n n } }$  No deformable refinement</td></tr></table>

Segmentation-derived distances are downstream metrics, not direct registration error.

FOV bootstrap CI nearly touched zero, indicating substantial ROI-level heterogeneity and modest average improvement. In clinical data with partial modality overlap, robust affine initialization should therefore be treated as a prerequisite, and methods that assume strong pre-alignment, such as Convex-Adam, require an explicit affine stage.

Blinded reader study and the venous-offender hypothesis. The Low-contrast tier may reflect limited TOF-MRA visibility, including possible enrichment for venous offenders since TOF-MRA preferentially depicts fast arterial flow. In the blinded reader study of 100 stratified ROIs, 66 were clinically evaluable and 34 were not; among evaluable ROIs, readers identified 42 arterial, 9 venous, 14 mixed arterial-venous, and 1 with no definite vessel. Thus, 23/66 evaluable ROIs (34.8%) had some venous component, supporting the hypothesis that part of the Low-contrast regime may reflect venous-offender biology rather than generic acquisition failure. However, TOF-MRA does not reliably depict venous anatomy, and this reader-based vessel typing is hypothesis-generating rather than a substitute for complementary venous imaging or intraoperative confirmation.

Clinical evaluability exposes a metric confound. Reader review also showed that favorable one-sided ROI distances can coexist with clinically unusable whole-brain registration. Across 100 reviewed ROIs, SynthMorph produced globally shifted or failed registrations in 71 cases; among the 66 ANTs-SyN-evaluable ROIs, 48 still showed SynthMorph whole-brain misalignment. As shown in Fig. 8, predicted vessels could fall close to the annotation within the small trigeminal ROI despite global distortion, yielding deceptively low $d _ { \mathrm { a n n } \to \mathrm { P r e d } }$ values. This explains why SynthMorph had the lowest median $d _ { \mathrm { a n n } \to \mathrm { P r e d } }$ in Table 3 despite frequent visual registration failure, and reinforces that segmentation-derived one-sided distances must be interpreted jointly with whole-brain registration quality.

Balanced performance of ANTs SyN. Across the six methods, ANTs SyN showed the most balanced profile (Table 4).

![](images/2cb17e7fca8a17af11fcb45763888ded9eec1ba0afc5f7ff06dd886e93f3cc93.jpg)  
Fig. 8. Whole-brain registration failures can dissociate from one-sided ROI distance metrics. Each row shows the fixed structural MRI, ANTs-SyN warped MRA, and SynthMorph warped MRA at the trigeminal ROI. (A) Gross misalignment rated non-evaluable. (B) SynthMorph global distortion with preserved local ROI proximity, while ANTs SyN remains anatomically plausible. Since $d _ { \mathrm { a n n \to P r e d } }$ is computed only within the ROI, globally failed but locally proximate predictions can yield deceptively low one-sided distances.

It was not the lowest performer on any metric, achieved the highest median Vessel AUC in the High-contrast tier (∼0.76), and produced moderate predicted vessel volumes $( | V _ { \mathrm { P r e d } } |$ median 1493). In contrast, SynthMorph and EasyReg achieved lower $d _ { \mathrm { a n n } \to \mathrm { P r e d } }$ and higher Vessel AUC but produced much larger predicted vessel trees, inflating $d _ { \mathrm { P r e d } \to \mathrm { a n n } }$ Convex-Adam had the lowest $d _ { \mathrm { s y m } }$ but required external affine prealignment, while FireANTs showed high $d _ { \mathrm { P r e d } \to \mathrm { a n n } }$ despite favorable $d _ { \mathrm { a n n } \to \mathrm { P r e d } }$ , again reflecting prediction-extent effects. ANTs SyN was slower, requiring approximately 20 minutes per MRI–MRA pair, but cases were independent and batchparallelizable. These results support ANTs SyN as a reasonable baseline for local overlay generation when contrast and FOV coverage are adequate, while faster or more detection-sensitive alternatives may be preferable for other task-specific priorities.

Clinical implications. MRI–MRA co-registration may improve TN surgical evaluation by integrating structural nerve detail with vascular trajectories near the trigeminal nerve, a relationship directly relevant to MVD planning and prognosis. Prior work has demonstrated the clinical value of fused highresolution MRI, TOF-MRA, and 3D visualization for posteriorfossa neurovascular assessment [21–29]. However, the registration accuracy needed for local trigeminal ROI interpretation has not been systematically quantified.

Our ROI-centered benchmark addresses this gap by evaluating registration behavior directly at the trigeminal nerve in a 149-patient MVD cohort, using local image-based and vessellocalization metrics. When local MRA contrast is adequate and MRI–MRA FOV coverage is comparable, ANTs SyN provides a reasonable offline baseline for local overlay generation. MRA integration may also strengthen MRI-only neurovascular segmentation by adding vascular conspicuity to structural nerve information, potentially improving quantitative assessment of NVC morphology and prognosis [2, 3].

Because TOF-MRA is more sensitive to arterial than venous flow, MRI–MRA fusion may also help generate hypotheses about compressive vessel identity. Vessels visible on both structural MRI and TOF-MRA are more suggestive of arterial flow, whereas neurovascular contact visible on MRI but faint or absent on TOF-MRA may raise suspicion for low-flow venous structures. This distinction is clinically relevant because venous compression is associated with less favorable MVD outcomes than arterial compression [6]. However, TOF-MRA does not reliably depict venous anatomy, so vessel-type interpretation from this pipeline remains non-definitive without complementary venous imaging or intraoperative confirmation.

Learning-based methods such as SynthMorph and EasyReg remain useful when speed, throughput, or vessel-detection sensitivity is prioritized, but their larger predicted vessel volumes require joint interpretation with extent-sensitive distance metrics. In cases with poor local contrast, the current registration–segmentation pipeline is unlikely to provide reliable vessel localization without improved acquisition or contrast-aware extraction. In cases with substantial FOV mismatch, deformable refinement should not be assumed to improve local alignment; affine alignment should remain an essential reference, and deformable outputs should undergo local quality control.

## 4.1. Limitations

This study has several limitations. It is retrospective and singlecenter; although it includes multiple scanners, vendors, and field strengths, it remains predominantly Siemens 3T, and external validation on more heterogeneous TN cohorts would further assess generalizability. The annotations contain clinically relevant partial vessel segments rather than exhaustive vessel masks, matching the surgical-review workflow but complicating geometric evaluation because predicted vessel trees may extend beyond the labeled segment; one-sided distances should therefore be interpreted jointly with predicted vessel volume. Vessel segmentation was performed after registration, so the resulting masks may be affected by interpolation, deformation, resampling, and method-specific warped-image appearance; segmentation-derived distances therefore reflect the registration–segmentation pipeline rather than registration error alone. The statistical analyses are exploratory and do not include mixed-effects modeling or correction for multiple stratified analyses, and annotation and reader-review variability were not fully quantified because both used consensus workflows. Future work should incorporate intraoperative offending-vessel confirmation, formal NVC grading, MVD outcome modeling, and external validation to better relate ROI-centered MRI–MRA registration to operative findings and patient outcomes.

## 5. Conclusion

We presented an ROI-centered benchmark for MRI–MRA registration in preoperative TN neurovascular visualization.

In 149 patients and 298 clinician-reviewed ROIs, we evaluated six registration pipelines using local image-based metrics and segmentation-derived vessel-localization measures. Our findings show that TN MRI–MRA fusion cannot be judged by generic whole-brain registration criteria: image-level vessel separability and downstream vessel localization are noninterchangeable, one-sided vessel distances are confounded by predicted vessel extent under partial annotations, Vessel AUC is informative mainly when local MRA contrast is sufficient, and deformable refinement provides only a small, FOV-dependent benefit over affine alignment. These results support evaluating TN MRI–MRA registration as a local, vessel-aware, contrastsensitive, and FOV-aware visualization task. Joint reporting of image-based and segmentation-derived metrics, prediction volume, contrast regime, FOV compatibility, and registration quality control provides a practical foundation for future ROIaware neurovascular visualization methods.

## References

[1] Risheng Xu, Sumil K Nair, Divyaansh Raj, Joshua Materi, Raymond J So, Sachin K Gujar, Judy Huang, Ari M Blitz, Michael Lim, Haris I Sair, et al. The role of preoperative magnetic resonance imaging in assessing neurovascular compression before microvascular decompression in trigeminal neuralgia. World Neurosurgery, 168:e216–e222, 2022. doi: 10.1016/j.wneu.2022.09.092.

[2] Kyra M Halbert-Elliott, Michael E Xie, Bryan Dong, Oishika Das, Xihang Wang, Christopher M Jackson, Michael Lim, Judy Huang, Vivek S Yedavalli, Chetan Bettegowda, and Risheng Xu. Deep learning-based segmentation of the trigeminal nerve and surrounding vasculature in trigeminal neuralgia. Journal of Neurosurgery, 143(1):83–91, 2025. doi: 10.3171/2024.10. JNS241060.

[3] Xihang Wang, Kyra M Halbert-Elliott, Michael E Xie, Oishika Das, Kathleen R Ran, Bryan Dong, M Abdulrahim, Christopher M Jackson, Michael Lim, Judy Huang, Vivek S Yedavalli, Chetan Bettegowda, and Risheng Xu. Machine learning-based quantification of neurovascular compression for correlation with trigeminal neuralgia pain outcomes. Pain, 2026. doi: 10.1097/j.pain.0000000000003946. Epub ahead of print.

[4] Akshitkumar M Mistry, Kara J Niesner, Wesley B Lake, Jonathan A Forbes, Chevis N Shannon, Robert A Kasl, Peter E Konrad, and Joseph S Neimat. Neurovascular compression at the root entry zone correlates with trigeminal neuralgia and early microvascular decompression outcome. World Neurosurgery, 95:208–213, 2016. doi: 10.1016/j.wneu.2016.08.040.

[5] Richard Loayza, Johan Wikström, Anna Grabowska, Robert Semnic, Hans Ericson, and Sami Abu Hamdeh. Outcome after microvascular decompression for trigeminal neuralgia in a single center—relation to sex and severity of neurovascular conflict. Acta Neurochirurgica, 165(7):1955–1962, 2023. doi: 10.1007/ s00701-023-05642-2.

[6] Sumil K Nair, Michael E Xie, Kathleen Ran, Anita Kalluri, Collin Kilgore, Judy Huang, Michael Lim, Chetan Bettegowda, and Risheng Xu. Outcomes after microvascular decompression for sole arterial versus venous compression in trigeminal

neuralgia. World Neurosurgery, 173:e542–e547, 2023. doi: 10.1016/j.wneu.2023.02.090.

[7] Brian B Avants, Charles L Epstein, Murray Grossman, and James C Gee. Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Medical Image Analysis, 12(1):26–41, 2008.

[8] Brian B Avants, Nicholas J Tustison, Gang Song, Philip A Cook, Arno Klein, and James C Gee. A reproducible evaluation of ANTs similarity metric performance in brain image registration. NeuroImage, 54(3):2033–2044, 2011.

[9] Nicholas J Tustison, Philip A Cook, Andrew J Holbrook, Hans J Johnson, John Muschelli, Gabriel A Devenyi, Jeffrey T Duda, Sandhitsu R Das, Nicholas C Cullen, Daniel L Gillen, et al. The ANTsX ecosystem for quantitative biological and medical imaging. Scientific Reports, 11(1):9068, 2021.

[10] Rohit Jena, Pratik Chaudhari, and James C Gee. FireANTs: Adaptive Riemannian optimization for multi-scale diffeomorphic matching. arXiv preprint arXiv:2404.01249, 2024.

[11] Hanna Siebert, Christoph Großbröhmer, Lasse Hansen, and Mattias P Heinrich. ConvexAdam: Self-configuring dualoptimization-based 3d multitask medical image registration. IEEE Transactions on Medical Imaging, 44(2):738–748, 2025.

[12] Juan Eugenio Iglesias. A ready-to-use machine learning tool for symmetric multi-modality registration of brain MRI. Scientific Reports, 13(1):6657, 2023.

[13] Malte Hoffmann, Benjamin Billot, Douglas N Greve, Juan Eugenio Iglesias, Bruce Fischl, and Adrian V Dalca. SynthMorph: Learning contrast-invariant registration without acquired images. IEEE Transactions on Medical Imaging, 41(3):543–558, 2022.

[14] Bastian Wittmann, Yannick Wattenberg, Tamaz Amiranashvili, Suprosanna Shit, and Bjoern Menze. vesselFM: A foundation model for universal 3d blood vessel segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20874–20884, 2025.

[15] Albert Lee, Shirley McCartney, Cole Burbidge, Ahmed M Raslan, and Kim J Burchiel. Trigeminal neuralgia occurs and recurs in the absence of neurovascular compression. Journal of Neurosurgery, 120(5):1048–1054, 2014. doi: 10.3171/2014.1. JNS131410.

[16] Marion A Hughes, Ronak H Jani, Saeed Fakhran, Yue-Fang Chang, Barton F Branstetter, Parthasarathy D Thirumala, and Raymond F Sekula. Significance of degree of neurovascular compression in surgery for trigeminal neuralgia. Journal of Neurosurgery, 133(2):411–416, 2019. doi: 10.3171/2019.3. JNS183174.

[17] Jacob Worm, Tone Bruvik Heinskou, Per Rochat, Jacob Bertram Springborg, Emil Andonov Smilkov, Lars Bendtsen, Henrik Winther Schytz, and Stine Maarbjerg. Five-year prospective outcomes of medical management and microvascular decompression in trigeminal neuralgia. Journal ofNeurology, 272(10): 701, 2025. doi: 10.1007/s00415-025-13447-9.

[18] Giovanni Antonini, Antonella Di Pasquale, Giorgio Cruccu, Andrea Truini, Stefania Morino, Giorgia Saltelli, Andrea Romano, Guido Trasimeni, Nicola Vanacore, and Alessandro Bozzao. Magnetic resonance imaging contribution for diagnosing symptomatic neurovascular contact in classical trigeminal neuralgia:

a blinded case-control study and meta-analysis. Pain, 155(8): 1464–1471, 2014. doi: 10.1016/j.pain.2014.04.020.

[19] Paulo Roberto Lacerda Leal, Charlotte Barbier, Marc Hermier, Miguel Angelo Souza, Gerardo Cristino-Filho, and Marc Sindou. Atrophic changes in the trigeminal nerves of patients with trigeminal neuralgia due to neurovascular compression and their association with the severity of compression and clinical outcomes. Journal ofNeurosurgery, 120(6):1484–1495, 2014. doi: 10.3171/2014.2.JNS131288.

[20] Jian Cheng, Jinli Meng, Wenke Liu, Heng Zhang, Xuhui Hui, and Ding Lei. Nerve atrophy in trigeminal neuralgia due to neurovascular compression and its association with surgical outcomes after microvascular decompression. Acta Neurochirurgica, 159(9):1699–1705, 2017. doi: 10.1007/ s00701-017-3250-9.

[21] Jonathan Miller, Feridun Acar, Bronwyn Hamilton, and Kim J Burchiel. Preoperative visualization of neurovascular anatomy in trigeminal neuralgia. Journal ofNeurosurgery, 108(3):477–482, 2008. doi: 10.3171/JNS/2008/108/3/0477.

[22] Jorge Docampo, Nadia Gonzalez, Alexandra Muñoz, Fernando Bravo, Daniel Sarroca, and Carlos Morales. Neurovascular study of the trigeminal nerve at 3T MRI. The Neuroradiology Journal, 28(1):28–35, 2015. doi: 10.15274/NRJ-2014-10116.

[23] Francesca Granata, Sergio Lucio Vinci, Marcello Longo, Gianmarco Bernava, Maria Caffo, Mariano Cutugno, Rosa Morabito, Ignazio Salamone, Francesco Tomasello, and Concetta Alafaci. Advanced virtual magnetic resonance imaging (MRI) techniques in neurovascular conflict: bidimensional image fusion and virtual cisternography. La Radiologia Medica, 118(6):1045–1054, 2013. doi: 10.1007/s11547-013-0928-9.

[24] Parviz Dolati, Alexandra Golby, Daniel Eichberg, Mohamad Abolfotoh, Ian F Dunn, Srinivasan Mukundan, Mohamed M Hulou, and Ossama Al-Mefty. Pre-operative image-based segmentation of the cranial nerves and blood vessels in microvascular decompression: Can we prevent unnecessary explorations? Clinical Neurology and Neurosurgery, 139:159–165, 2015. doi: 10.1016/j.clineuro.2015.10.006.

[25] S Yao, J Zhang, Y Zhao, Y Hou, X Xu, Z Zhang, Ron Kikinis, and X Chen. Multimodal image-based virtual reality presurgical simulation and evaluation for trigeminal neuralgia and hemifacial spasm. World Neurosurgery, 113:e499–e507, 2018. doi: 10. 1016/j.wneu.2018.02.069.

[26] Omar A Gamaleldin, Mohamed M Donia, Noha A Elsebaie, Abdelkhalek Abd-Elkhalek Abdelrazek, Tarek Rayan, and Mohamed H Khalifa. Role of fused three-dimensional time-offlight magnetic resonance angiography and 3-dimensional T2- weighted imaging sequences in neurovascular compression. World Neurosurgery, 133:e180–e186, 2020. doi: 10.1016/j. wneu.2019.08.190.

[27] Hong Duc Pham, Thu Ha Dang, Trung Kien Duong, Trung Thanh Dinh, Van Giang Bui, Tuan Vu Nguyen, and Quang Huy Huynh. Predictability of fused 3D-T2-SPACE and 3D-TOF-MRA images in identifying conflict in trigeminal neuralgia. Journal of Pain Research, 14:3421–3428, 2021. doi: 10.2147/JPR.S331054.

[28] Peter Hastreiter, Barbara Bischoff, Rudolf Fahlbusch, Arnd Doerfler, Michael Buchfelder, and Ramin Naraghi. Data fusion and 3d visualization for optimized representation of neurovascular

relationships in the posterior fossa. Acta Neurochirurgica, 164 (8):2141–2151, 2022. doi: 10.1007/s00701-021-05099-1.

[29] Yu Huang, Ying Huang, Chaoyong Xiao, Qingling Huang, and Xue Chai. Preoperative evaluation of neurovascular relationship in primary trigeminal neuralgia (PTN) by magnetic resonance virtual endoscopy (MRVE) combined with 3D-FIESTA-c and 3D-TOF-MRA. Journal ofPain Research, 17:2561–2570, 2024. doi: 10.2147/JPR.S465956.

[30] Annie E Arrighi-Allisan, Bradley N Delman, John W Rutland, Amy Yao, Judy Alper, Kuang-Han Huang, Priti Balchandani, and Raj K Shrivastava. Neuroanatomical determinants of secondary trigeminal neuralgia: application of 7T ultra-high-field multimodal magnetic resonance imaging. World Neurosurgery, 137:e34–e42, 2020. doi: 10.1016/j.wneu.2019.11.130.

[31] Junyu Chen et al. MIR: A medical image registration toolbox. https://github.com/junyuchen245/MIR, 2024. GitHub repository.

## Supplementary Material

This Supplementary Material provides additional methodological details for the MRI–MRA registration benchmark described in the main manuscript. Sections S1–S3 describe cohort selection, imaging acquisition, and ROI annotation; Sections S4–S5 describe registration settings, computing resources, and VesselFM inference; and Sections S6–S7 describe contrast stratification, descriptive thresholds, and statistical analysis. Supplementary Table S1 summarizes the imaging acquisition parameters. References to numbered sections and figures refer to the main manuscript unless otherwise indicated.

## S1. Cohort Selection and Demographics

Eligibility. We retrospectively studied a single-institution TN cohort consisting of patients who underwent microvascular decompression (MVD) between January 2020 and December 2022; diagnostic imaging dates ranged from October 2019 to December 2022. Inclusion required paired structural MRI and TOF-MRA from the same preoperative diagnostic imaging encounter and clinician-provided trigeminal annotations.

Cohort flow. From an initial 159 paired MRI-MRA scans, 6 cases were excluded during preprocessing because of DICOM read errors, autocropping artifacts, or related technical failures, yielding 153 successfully preprocessed pairs. A further 4 cases were excluded because trigeminal annotations existed but the corresponding MRA volume was missing or could not be reliably paired, resulting in a final analysis cohort of 149 patients (mean age 56.6 ± 14.0 years, range 20.1-82.4; 85 female / 64 male; 88 right-sided / 61 left-sided symptomatic presentation, no bilateral; 7 patients [4.7%] with multiple sclerosis as a secondary cause of TN).

Analysis cohort. All final-cohort patients subsequently underwent MVD, and all imaging preceded surgery. All patients had bilateral ROIs annotated, yielding 298 trigeminal ROIs in total.

Ethics approval. The study was approved by the Johns Hopkins Medicine Institutional Review Board (IRB00338945). Informed patient consent was waived given the retrospective nature of the study

## S2. Imaging Acquisition Parameters

Sequences and scanners. Structural MRI (CISS-type) was used to visualize the trigeminal nerve and surrounding soft tissue, and TOF-MRA was used to depict vascular structures. In brief, 98.0% of scans were acquired on Siemens scanners (predominantly Skyra) and the remainder on GE Healthcare; 92.6% of MRI and 91.9% of MRA scans were acquired at 3T, with the remainder at 1.5 T. Structural MRI used a 3D CISStype sequence (median TR/TE 5.45/2.42 ms; median in-plane spacing 0.597mm; median slice thickness 0.60mm), and TOF-MRA used a 3D multi-slab TOF sequence (median TR/TE 22.00/3.78 ms; median in-plane spacing 0.260mm; median slice thickness 0.50mm).

Acquisition timing and reconstructions. In the final analysis cohort, MRI acquisition dates ranged from January 2020 to December 2022 and MRA acquisition dates from October 2019 to December 2022. A small subset of structural MRI scans had DICOM slice thickness greater than 0.6mm, reflecting non-CISS reconstructions; however, ROI annotation and all analyses were performed on the native high-resolution CISS series. Detailed per-modality parameters are reported in Supplementary Table S1.

## S3. ROI Annotation Protocol

Annotation workflow. ROI centroids were identified using ITK-SNAP, and per-voxel labels were drawn manually on cropped structural MRI volumes using the Napari Python toolkit. The final annotations used for this benchmark were defined on 48 × 48 × 48 ROI crops centered on the cisternal trigeminal nerve, which is the unit of analysis used through out the study. Each volume received a first-pass segmentation followed by independent second-pass review and editing before finalization. Annotations were drawn on structural MRI alone, using vessel-related signal voids and local neurovascular anatomy visible on MRI, without overlay on MRA or any registration output. The annotation protocol was therefore independent of the registration methods evaluated here. The labels are intended to mark clinically relevant candidate vessel segments rather than to provide an exhaustive vascular tree; this partial-annotation property is central to the metric design described in Section 3.1. The two-pass workflow produced consensus rather than parallel annotations, which limits direct quantification of inter-annotator variability.

Operational definition of the clinically relevant vessel segment. Vessel labels were drawn to capture all vessels in contact with or in close proximity to the trigeminal nerve, with an emphasis on the contact region. Distinctly visible non-contacting vessels reasonably close to the nerve were also segmented; very peripheral or low-signal vessels (e.g., short tracks visible only across 3-4 contiguous slices) were not labeled. Because the protocol focuses on the contact region, vessel coverage may be incomplete near the periphery of the 48×48×48 ROI for some cases. This is the operational source of the partial-annotation property described next.

Implications for vessel-proximity metrics. An important property of this protocol is that vessel labels mark only the vessel segment considered clinically relevant to the trigemina nerve within the ROI, rather than every vessel voxel that could in principle be labeled on MRA. In contrast, VesselFM and other foundation vessel segmentation models produce a full vascular trace that typically extends beyond the annotated segment. This mismatch is intrinsic to the clinical annotation workflow and is a critical caveat for all vessel-proximity metrics reported in the main text; we revisit it quantitatively in Section 3.4.

## S4. Registration Settings and Computing Resources

Registration settings. ANTs Affine and SyN used a multiresolution schedule with Mattes mutual information for rigid/affine stages and cross-correlation for SyN refinement. ConvexAdam used the ConvexAdam\_MIND\_brain\_default configuration with MIND-SSC descriptors and 99.5thpercentile intensity normalization. FireANTs was run with moments-based center-of-mass/rigid alignment followed by multi-resolution greedy deformable registration. EasyReg and SynthMorph were run with FreeSurfer default configurations. No external brain or registration masks were supplied to any method.

Table S1. Imaging acquisition parameters per modality (N = 149 patients). Continuous parameters are summarized as median [Q1, Q3] (range). Categorical parameters are summarized as count (%). All parameters are derived from DICOM headers; in-plane and slice spacing are confirmed from NIfTI volume headers used for registration.
<table><tr><td>Parameter</td><td>Structural MRI (CISS-type)</td><td>TOF-MRA</td></tr><tr><td>Vendor and scanner</td></tr><tr><td>Manufacturer (Siemens / GE) 146 (98.0%) / 3 (2.0%)</td><td>146 (98.0%) / 3 (2.0%)</td></tr><tr><td>Most common model</td><td>Skyra: 115 (77.2%); Verio: 12 (8.1%);</td><td>Skyra: 112 (75.2%); Verio: 13 (8.7%);</td></tr><tr><td></td><td>Aera: 9 (6.0%); MAGNETOM Vida: 5 (3.4%); other: 8 (5.4%)</td><td>Aera: 9 (6.0%); MAGNETOM Vida: 7 (4.7%); other: 8 (5.4%)</td></tr><tr><td>Field strength (3.0 T / 1.5 T)</td><td>138 (92.6%) / 11 (7.4%)</td><td>137 (91.9%) / 12 (8.1%)</td></tr><tr><td>Sequence parameters (DICOM)</td></tr><tr><td>Predominant series description POST/PRE CISS SAG MPR (and recon-</td><td>TOF_3D_multi-slab (130/149); MRA</td></tr><tr><td>structions thereof)</td><td>cow (7); other configurations (12)</td></tr><tr><td>Repetition time TR (ms) 5.45 [5.43, 5.46] (5.00-7.84)</td><td>22.00 [22.00, 22.00] (19.00-25.00)</td></tr><tr><td>Echo time TE (ms)</td><td>3.78 [3.78, 3.78] (3.40-7.15)</td></tr><tr><td>Slice thickness, DICOM (mm) 0.60 [0.60, 0.60] (0.59-1.00)</td><td>2.42 [2.41, 2.43] (2.04-3.66)</td></tr><tr><td></td><td>0.50 [0.50, 0.60] (0.40-1.20)</td></tr><tr><td>Volume geometry (NIfTI)</td><td></td></tr><tr><td>In-plane spacing (mm) 0.597 [0.597, 0.597] (0.372-0.880)</td><td>0.260 [0.260, 0.288] (0.260-0.482)</td></tr><tr><td>Slice spacing (mm) 0.594 [0.594, 0.594] (0.150-0.859)</td><td>0.500 [0.500, 0.600] (0.400-0.800)</td></tr><tr><td>Median matrix shape</td><td>254× 256× 164 646× 768×165</td></tr><tr><td></td><td></td></tr><tr><td>Acquisition timing</td><td></td></tr><tr><td></td><td></td></tr></table>

Software and hardware. All experiments were run under Ubuntu 24.04 with Python 3.10. The registration methods used the following versions: ANTs 2.6.2, FireANTs 1.0.0, Convex-Adam (MIND-SSC implementation, accessed through the publicly released Medical Image Registration (MIR) toolbox [31]), and EasyReg/SynthMorph from FreeSurfer 8.1.0. Learningbased and GPU-accelerated methods used PyTorch 2.10.0 with CUDA 12.8. The compute server (ldr01 at our institution) had 2 AMD EPYC 9354 CPUs (128 logical threads), 1.5 TB RAM, and 8 NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs (96 GB each, driver 580.126). Cases are independent and were processed in parallel across CPU cores or GPUs as appropriate; the effective per-case wall-clock time depended on the batch parallelization rather than on the single-case algorithmic cost. ANTs SyN was the runtime-dominant step, on the order of ∼20 min CPU per MRI-MRA pair in our deployment, but offline batch processing across the 149-patient cohort completed within a small number of days.

Memory-related fallback. All six methods produced warped MRA outputs for all 149 patients (298 ROIs). Four patients had high-resolution MRI reconstructions with in-plane voxel spacing 0.15-0.17mm (versus the cohort-typical 0.59mm

CISS), yielding preprocessed volumes that exceeded single-GPU memory for ConvexAdam; these cases were processed using an identical CPU-based ConvexAdam pipeline and completed successfully.

## S5. VesselFM Inference and Postprocessing

Model and application. We applied VesselFM [14] (v1.0, https://github.com/bwittmann/vesselFM) to each warped MRA volume in MRI space to produce a whole-brain vessel segmentation, which was then cropped to the matched trigeminal ROI and compared against the clinician annotation. We used the publicly released model (dyn\_unet\_base) without retraining or fine-tuning, and applied the same inference configuration to every method’s warped MRA.

Inference and postprocessing. Each warped MRA was preprocessed by 1st-99th-percentile intensity normalization to [0, 1] with clipping, then run through sliding-window inference with a 128<sup>3</sup> patch size, 0.5 patch overlap, and constant-mode merging. No test-time augmentation was used. The predicted probability map was binarized at threshold 0.5, after which the repository’s default postprocessing removed connected components smaller than 500 voxels (face-edge-vertex connectivity); no additional filtering was applied.

ROI-based evaluation. The resulting whole-brain vessel mask was resampled into the same trigeminal 48<sup>3</sup> ROI sampling grid used for image-based metrics. Vessel-proximity metrics required at least one predicted vessel voxel in the ROI for both compared methods; ROI-method observations with empty predicted vessel mask were excluded only from those metrics, while contrast and Vessel AUC, which depend only on the warped MRA intensity, were retained.

## S6. Contrast Stratification and Exclusions

Cross-method consistency. Across the six registration methods, per-ROI contrast values shared a similar overall scale (pairwise Pearson 0.41-0.74 on the 278 ROIs with valid contrast measurements under all six methods) but the per-ROI rank ordering of contrast was only weakly preserved between methods (pairwise Spearman 0.11-0.51, median 0.24).

Reference definition and exclusions. Using the SyN contrast as the canonical reference therefore reflects an analysis choice rather than a method-invariant ROI property, and the permethod Vessel AUC distributions in Section 3.6 are reported within the same SyN-defined tiers to ensure comparability. Of 298 total ROIs, 286 yielded valid contrast measurements; 12 ROIs were excluded because the ANTs SyN deformable warp placed the TN-centered sampling grid outside the warped MRA coverage.

## S7. Descriptive Thresholds and Statistical Analysis

Descriptive thresholds. For cross-tabulation of image-level versus downstream outcomes within the Mid-contrast tier (Section 3.6), we additionally classified ROIs by two descriptive thresholds. An ROI was classified as image-level high-AUC if its Vessel AUC under ANTs SyN exceeded 0.60, chosen as a pragmatic cutoff above the 0.5 chance level rather than an optimized decision threshold. A non-empty postregistration vessel prediction was classified as a downstream miss $\mathrm { i f } d _ { \mathrm { a n n \mathrm { \to } P r e d } } > 2$ 2mm; this threshold corresponds to roughly four voxels on the 0.47mm ROI grid and was used solely for descriptive cross-tabulation, not for primary method ranking.

FOV stratification threshold. The 20th-percentile FOV cutoff $( r _ { \operatorname* { m i n } } \le 0 . 7 6 )$ was used to define an interpretable severemismatch subgroup; the primary conclusion is the attenuation pattern of deformable benefit rather than the specific numerical cutoff, and this threshold-based stratification should be interpreted as descriptive.

Statistical analysis. Distributions are summarized as medians with interquartile ranges. Volume-distance coupling was assessed using Spearman rank correlation. Paired Affine-versus-SyN comparisons were assessed using the Wilcoxon signedrank test. Because each patient contributes bilateral ROIs that are not independent (same scanner, same anatomy), we report patient-level bootstrap 95% confidence intervals (1000 resamples) for all key descriptive and inferential statistics. Each bootstrap iteration resampled the 149 patients with replacement; resampled patients always contributed both ipsilateral and contralateral ROIs together, naturally preserving the withinpatient correlation. Because the analyses are exploratory and the same data are used for multiple stratifications, we do not adjust for multiple comparisons; reported p-values and CIs should be interpreted with this in mind.