# FleXray: Universal Clinical X-ray Segmentation

Victor Ion Butoi <sup>1</sup>, Vivek Gopalakrishnan <sup>1</sup>, John V. Guttag <sup>1</sup>, Adrian V. Dalca <sup>1,2,3</sup>, Neel Dey <sup>1,2,3</sup>

1. MIT CSAIL 2. Massachusetts General Hospital 3. Harvard Medical School

Abstract. X-ray is medicine’s most widely used imaging modality, yet remains among its least quantitative. Unlike volumetric modalities like CT or MRI, X-ray collapses 3D anatomy into a 2D projection, causing structures to overlap and anatomical boundaries to be ambiguous, even to experts. As a result, labeling X-ray databases for training general-purpose segmentation systems is impractical, leaving morphometric and functional X-ray analysis confined to narrow anatomical regions and applications. To this end, we present FleXray, a generalist model for anatomical segmentation across the entire body in clinical X-rays. Instead of curating large, manually annotated X-ray datasets, we build a scalable, physics-based generative X-ray data engine. Using existing 3D whole-body CT segmentation datasets and generative image-editing models, we simulate fully-annotated 2D X-rays with diverse appearances, physiological properties, and imaging geometries. Trained on these simulations, FleXray accurately segments 60 anatomical structures across unseen research datasets and in-the-wild X-rays. We further show that FleXray makes X-rays directly amenable to quantitative analysis, enabling automated measurements for disease grading, robust navigation dur ing X-ray-guided interventions, and data-efficient learning of pathological targets. We release the model, code, a full-body X-ray segmentation dataset, and a local, easy-to-use browser-based tool at https://flexray.csail.mit.edu.

Modern segmentation tools automatically outline dozens of anatomical structures in CT and MRI volumes [1– 6], enabling treatment planning [7–9], population-scale studies [10, 11], and routine extraction of quantitative biomarkers [12]. However, despite their widespread clinical use and greater abundance (Figure 1a), X-rays lack comparable tools for quantitative anatomical analyses.

The main bottleneck is not a shortage of X-rays, but rather the difficulty of annotating them. Because X-rays collapse 3D anatomy onto a 2D detector, organs overlap and obscure boundaries that are easily separated in volumetric modalities. Creating dense pixel-level annotation of every organ for training X-ray segmentation systems is therefore difficult even for experts [13, 14]. These challenges are compounded by the unique heterogeneity of X-ray: anatomical regions, patient positions, fields of view, and acquisition setups vary substantially across medical contexts [15, 16], while patient- and procedurespecific factors like bone density, contrast agents, and implanted devices add further diversity and difficulty [17– 19]. As a result, existing datasets typically annotate only a handful of organs (e.g., the heart and lungs) under restricted acquisition settings and in specific populations [20, 21] (Figure 1b), leaving insufficient training data for reliable, multi-organ anatomical segmentation across the breadth of real-world X-rays.

Overcoming this limitation would create substantial biomedical and scientific opportunities. X-rays are acquired routinely and at enormous scale, yet much of their anatomical information is still assessed only through visual interpretation or coarse manual measurements. Instead, a generalist X-ray segmentor could make radiography quantitative by converting each X-ray into a structured anatomical map, enabling automatic and consistent extraction of anatomical measurements, imaging biomarkers, and longitudinal changes. Further, retrospective archives and biobanks could be mined to determine population-level associations between anatomy, disease, treatment, and outcomes using a modality that is already routinely acquired worldwide.

We address this bottleneck with a scalable, physicsbased generative data engine and use it to train FleXray, a generalist model that automatically segments 60 anatomical structures in clinical X-rays. Instead of relying on painstaking manual annotation, our data engine uses several densely labeled 3D CT datasets to simulate X-rays (digitally reconstructed radiographs or DRRs [22, 23]) together with dense anatomical labels (Figure 1c,d), providing broad whole-body coverage across age groups. Because the data engine is fully controllable, we also randomize pose, projection geometry, magnification, and organ density to sample annotated training examples across a wide range of acquisition conditions (Figure 2a).

However, dense synthetic supervision alone does not guarantee generalization to real radiographs due to order-of-magnitude differences in resolution between CT and X-ray, content differences such as inscribed text, and the challenges of modeling higher-order X-ray image formation terms [25]. We therefore use generative image-editing models [26] to introduce clinical appearance factors that are difficult to model analytically, including scanner artifacts, text overlays, scatter, and other higher-order effects (Figure 2b). Further, because b. Publicly available X-ray segmentation labels provide sparse coverage.

a. Total annual scan volume by imaging modality.  
![](images/2d039f4cd4b764c7cc3023e605f16c65ca24b20ae898a397d1d092820c6131b7.jpg)

![](images/cdcc00d19981032b87d796aca141c72ea2bb026ad4ce93081bbdf968ffcfe5be.jpg)

c. Generating synthetic labeled X-ray from CT Scans.  
![](images/326c1cced7072f49f1f4d08aa66d95e1f124761be8797cd0888193854104b3ce.jpg)

![](images/3a9fcb5bd3af8d3ccbc36babb4305870bca8cf4f203c8a310d1c0955b46c3225.jpg)

d. Examples of rendered synthetic X-rays and ground truth segmentation labels.  
![](images/523a1f433f4d05fa0ea86a674efa0296760ac7fab5b46daae70d66d3129b2a45.jpg)  
Figure 1. Labeled 3D CTs can provide dense 2D X-ray supervision beyond the coverage of public X-ray segmentation datasets. a, Global annual X-ray and CT examination volumes reported by the United Nations Scientific Committee on the Effects of Atomic Radiation [24], with X-rays subdivided by examination type. b, Despite their abundance, the labels in X-ray segmentation datasets unevenly distributed across the body and focused on the chest. Colors summarize label availability across 12 public datasets containing 667,380 labeled X-rays (Appendix K.3). c, We generate labeled synthetic X-rays by doing rayprojection through the CT volumes and their labels from different viewing angles. d, A single labeled CT therefore generates multiple annotated training views; examples show synthetic radiographs across the body (top) and their overlapping anatomical labels (bottom).

whole-body CT collections often omit some regions that are commonly imaged with X-rays (e.g., the hands and feet), we supplement our CT training sources with real X-ray datasets to close coverage gaps across common X-ray use cases. More broadly, we treat dataset construction as an iterative coverage problem: we identify and close gaps in the training distribution by including targeted data sources, custom X-ray-specific augmentations, and carefully tuned data mixtures. Together with our synthetic labeled X-ray generation pipeline, this combined dataset spans greater anatomical, appearance, and geometric diversity than any existing collection.

Trained using this generative data engine, FleXray accurately segments anatomy across held-out research datasets and in-the-wild X-rays, spanning regions, populations, imaging setups, and medical conditions. Beyond segmentation accuracy, we demonstrate how making X-rays quantitative enables automated measurements for disease grading, robust surgical X-ray guided navigation, and data-efficient learning of previously unseen pathological targets. We release FleXray’s trained models and data engine as open-source software, together with our generatively enhanced X-ray dataset (https :// huggingface.co/datasets/VictorButoi/flexray-data) and a one-click, browser-based tool for segmenting any X-ray at https://flexray.csail.mit.edu.

## Results

FleXray learns full-body X-ray segmentation from a generative data engine.

Scaling anatomical coverage across the body. Labeled Xray simulation from CT datasets provides dense anatomical supervision beyond the limited coverage of public Xray annotations (Figure 1b). Prior work has rendered labeled radiographs from segmented CT volumes [22, 25, 29–32], but generally targets specific regions, particularly the chest [22, 25, 29, 32], standard frontal/lateral projections [30], or interactive workflows [31]. Our goal is instead to scale simulation across the body, patient populations, and continuously varying acquisition parameters and geometries.

We aggregate six CT datasets comprising 2,159 subjects, including 1,597 densely annotated whole-body CTs from the MOOSE dataset [3]. Because “whole-body” CT scans generally cover only neck-to-knee anatomy, use standardized patient poses, and rarely include pediatric subjects, we add five targeted CT datasets to fill gaps in anatomical and population coverage [33–37]. However, peripheral anatomy remains sparsely imaged in public CT databases. Therefore, we incorporate four real X-ray datasets totaling 764 samples: public hand and foot annotations [38, 39], plus manual forearm and upper-arm annotations that we collect from MURA [40].

Combining these sources requires reconciling differences in anatomical coverage and label definitions between source datasets, such as “spine” in one dataset versus “L1 vertebra” in another. We therefore develop an automatic harmonization pipeline that hierarchically maps dataset-specific ontologies into a common 60-structure label space (Figure 2c; Methods A.4). We restrict this vocabulary to anatomy meaningfully represented in radiographs and exclude structures generally invisible on X-ray, such as the brain and skeletal muscle.

Sampling anatomy across X-ray acquisition conditions. We train on DRRs sampled on-the-fly during training from multiple CT sources (Figure 1c,d; Methods A.1 and Table 2). A physics-based analytical X-ray renderer [23] generates these images from the CT volumes and specified scanner intrinsics and extrinsics. At each training iteration, we sample an anatomical target, viewpoint, field of view, magnification, and projection geometry. We also randomize per-organ attenuation to simulate organ density variations independently of acquisition geometry (Figure 2a; Methods A.1). Because several structures can contribute to the same detector pixel, we project every 3D anatomical mask independently. The resulting targets preserve the dense, overlapping nature of X-ray anatomy rather than imposing a mutually exclusive segmentation map (Figure 1c,d; Methods A.1).

Closing the synthetic-to-real appearance gap. Analytical DRRs capture anatomy, geometry, and attenuation, but do not reproduce the full appearance distribution of clinical X-rays. Resolution differences, text overlays and higher-order effects such as dosage, detector response, and scatter remain difficult to model efficiently [25]. We therefore apply a pretrained language-conditioned generative image editor [26] to DRRs sampled from the MOOSE dataset to add coverage for higher-order clinical effects (Figure 2b; Methods A.2). The generative editor requires no task-specific fine-tuning and is instructed to introduce clinical appearance while preserving projected anatomy. While generative models have previously been used to synthesize chest X-rays [41–43], ours is the first whole-body, label-preserving generative simulation-toreal pipeline for constructing a reusable densely labeled X-ray dataset. Further, because generative editing can occasionally alter anatomy and introduce image-label disagreement, we develop an automatic quality-control pipeline that rejects simulations with gross deviations, removing approximately 4% of enhanced DRRs (Figure 8 and Methods A.2).

The final data engine combines three complementary data sources: online analytic projections for anatomical and geometric diversity, quality-controlled generatively-edited DRRs for realistic appearance, and targeted real X-rays for regions poorly represented in CT (Table 1). The utility of the online/offline mixture and the contribution of generative editing are examined separately in Table 8 and Figure 4c. We release the resulting dataset, including the quality-controlled enhanced DRRs and our MURA annotations, as part of our data release (Appendix H).

CT Datasets Labels  
X-ray Datasets Labels  
a. Modeling per-subject density variability.  
![](images/567d725bbd5037149c53980e64ea159abbe74af08c543b46a97735db6c9fdf41.jpg)  
b. Improving digitally reconstructed radiograph (DRR) realism with an off-the-shelf generative model.  
Generatively-edited DRRs

![](images/0e444df50c7077f770eb105943e80d11ebe62e017f5bba885b1331e71505362c.jpg)  
c. Converting multi-modal, multi-label datasets into the same label space.

![](images/f611301a151c8d31d6246bb7b4e5e9f73f00ca485b9bf8c24a4c3ed4947dd628.jpg)  
Figure 2. Our generative data engine broadens radiographic appearance and harmonizes inter-dataset label formats. a, Clinical X-ray appearance varies with factors including bone pathology [27] and contrast agents [28] (left). We use per-structure attenuation changes to introduce label-specific density randomization during training (right). b, An off-the-shelf generative image editor transforms raw digitally reconstructed radiographs (DRRs; top) into realistic X-rays with clinical textures and acquisition artifacts (bottom). c, Because our source datasets use different label definitions, we map their annotations (top) into a common 60-structure vocabulary (bottom), allowing CT and real X-ray supervision to be freely combined.

Training on incomplete and overlapping annotations. We train FleXray on our data engine’s outputs. Combining heterogeneous datasets increases coverage but means that each source annotates only a subset of the 60- structure vocabulary. Treating unannotated structures in a source as absent would therefore mislead training. Instead, we route each source through a loss matched to its annotation coverage, ignoring labels whose status is undetermined. Where absence is anatomically certain, we include impossible structures as explicit negatives; for example, hand structures can safely be supervised as absent in foot X-rays. This dataset-routed objective allows complementary sources to be learned jointly without assuming that unannotated anatomy is absent (Methods B.2). Further, X-ray segmentation is intrinsically multi-label at the pixel level due to organ overlap. We therefore derive supervision directly from the projection geometry, mark every organ intersected by a ray as positive, and predict an independent probability map for each structure (Figure 1c,d; Methods B.1 and B.2).

The final FleXray system is an ensemble of five independently trained networks exposed to different proportions of generatively enhanced data, with each network using 16-sample test-time augmentation (Methods B.1 and C.3). Ensembling and test-time-augmentation aim to broaden the data distribution at training and inference, respectively, both seeking to improve robustness on in-thewild X-rays (Figure 9). For the ablations and finetuning experiments that follow, we use a single model—FleXray (single)—trained with equal proportions of standard and generatively enhanced X-rays. Together, this pipeline converts heterogeneous CT and X-ray datasets into a harmonized source of dense supervision spanning anatomy, populations, projection geometries, and clinical appearance, enabling the training of full-body X-ray segmentation networks that require no user prompts or acquisition metadata at inference.

## FleXray robustly segments real X-rays across medical contexts.

Quantitative segmentation evaluation. We evaluate FleXray on eight held-out X-ray datasets spanning the limbs, pelvis, spine, and chest, spanning multiple fieldsof-view and populations (Figure 3; Table 4). Where possible, we compare against existing X-ray segmentation frameworks that aim to generalize to the target dataset, including PAXray [29], TotalSegmentator2D [30], and FluoroSAM [31]. PAXray and TotalSegmentator2D are evaluated only on supported anatomical regions, whereas FluoroSAM is evaluated throughout but receives a simulated ground-truth-derived text prompt and positive click for each target. As in-domain supervised upper bounds, we also train a separate nnU-Net on the training split of each evaluation dataset (Methods C.4). Predictions are post-processed to match dataset-specific annotation conventions (Methods C.1) and evaluated using Dice score (Methods C.2).

Across all eight datasets, FleXray achieves an equaldataset mean Dice of 0.875 and significantly outperforms every supported baseline on seven datasets (Figure 3, left; all $p { < } 0 . 0 0 1$ on six datasets). On VinDr-Rib, its performance is not significantly different from PAXray (p=0.146), a baseline specifically developed for chest

X-rays. Qualitatively, FleXray remains anatomically coherent despite large differences in field of view, projection angle, image polarity, acquisition modality, and patient population (Figure 3, right).

Although FleXray never trains on any of these evaluation datasets, its performance is comparable to that of supervised nnU-Nets trained for that specific dataset. FleXray trails these in-domain networks by only 0.039 Dice on average (0.875 versus 0.914), and on DeepFluoro it performs better than the nnU-Net trained directly on the dataset (0.915 versus 0.866). This is due to Deep-Fluoro containing little training data and spanning a wide range of acquisition angles, illustrating the advantages of learning from a broad synthetic acquisition distribution.

However, while held-out, academic datasets still represent relatively curated imaging conditions. We therefore also apply FleXray to “in-the-wild” radiographs collected from Radiopaedia (Figure 4a). These examples span frontal, lateral, and oblique views; unilateral and bilateral examinations; adult and pediatric anatomy; and large differences in field of view. Across these heterogeneous cases, FleXray continues to produce qualitatively coherent anatomical segmentations, providing evidence that its robustness extends beyond benchmark datasets.

Robustness to acquisition angle. Clinical radiographs are frequently acquired at oblique angles for certain examinations, whereas most X-ray segmentation datasets and models focus entirely on standard frontal and lateral views [44]. We isolate this factor using DeepFluoro, which provides both labels and ground-truth acquisition angles for each X-ray. Across the 213 frames from its two held-out test subjects, we measure per-structure Dice as a function of geodesic distance from the frontal view (Figure 4b). Overall, FleXray achieves a mean Dice of 0.915 [0.912, 0.919], with 5<sup>◦</sup>-bin averages ranging from 0.925 [0.921, 0.928] at 5–10<sup>◦</sup> to 0.888 [0.847, 0.923] at 35–40<sup>◦</sup>. Femurs, hips, and sacrum remain within 0.05 Dice of their frontal-view performance across the full pose range. Only the lumbar spine, which becomes increasingly small and foreshortened in oblique views, shows a larger decline, from 0.840 [0.829, 0.850] within 15<sup>◦</sup> of frontal (n=136) to 0.751 [0.690, 0.805] beyond 25<sup>◦</sup> (n=27). In comparison, the applicable baselines (FluoroSAM and TotalSegmentator2D) are both less accurate and less stable, reaching mean Dice scores of 0.711 [0.701, 0.722] and 0.371 [0.357, 0.386], respectively. Thus, FleXray remains robust well beyond the standard acquisition angles represented in most X-ray datasets.

Ablating generative enhancement. To isolate whether generative X-ray enhancement provides useful supervision in addition to our augmentation pipeline, we remove it as an augmentation. While the equal-dataset average Dice is unchanged (0.870 enhanced versus 0.870 raw; n=1,458), we find substantial benefits in regions that are challenging to render faithfully in rendered DRRs. For example, as the lower ribs on VinDr-Rib are obfuscated by abdominal organs in standard DRRs (Figure 4c), ribs $5 \mathrm { - } 1 0$ all improve with generative enhancement (+3.9 [+2.2, +5.8] pooled, $p { < } 0 . 0 0 1 )$ , with gains increasing from +2.5 Dice points at rib $5 ~ \mathrm { t o } ~ + 7 . 4$ at rib 10. These results suggest that generative editing contributes useful appearance variation beyond conventional augmentation, with substantial benefit to specific anatomical structures.

a. Quantitative Performance (Dice score)  
b. Qualitative Prediction Samples  
![](images/965a7bea47d12a345616d49b2a91cb978cb4867d2ecfd8df366e27a5088fa2db.jpg)  
Figure 3. FleXray generalizes across eight held-out X-ray datasets. Each row shows a dataset entirely withheld from FleXray training; N denotes the number of test images. a, Mean label-averaged Dice across test images for FleXray, PAXray, TotalSegmentator2D, and FluoroSAM, with 95% confidence intervals. FluoroSAM receives a ground-truth-derived text prompt and positive click for each target. Dataset-specific nnU-Nets trained on the corresponding training splits provide in-domain supervised references (dashed lines and shaded confidence intervals). N/A indicates that a method does not predict the labels required for evaluation on that dataset. Full Dice results are in Table 6. b, Example radiographs, ground-truth labels, and prediction overlays thresholded at 0.5. The FleXray column is outlined in purple.

a.  
Generalization to in-the-wild radiographs  
![](images/f682ea405c4785da681ad45c52a64a5e3c841c421d2258aa063b4c726f2c81b9.jpg)

b.  
Segmentation across different acquisition views  
![](images/867154f00e0894675a7a574e18c9fd40c379dd0b52326fc049ea42a7b52dd928.jpg)

c.  
Generative enhancement improves challenging targets  
![](images/ae865af9b8c8753da991573f32d8f9d4373a99019e56c2c7dae60f427a8e5463.jpg)

Attenuation randomization improves robustness to density variation  
d.  
![](images/424405c8361c93007b29b018c1b2fb433e9d74f3a908a3874e1ba0249ada682c.jpg)

![](images/1c225543fd89b0fa5621a12a8420ce05c39d3b2911b4334db6833ca08430fb02.jpg)  
Figure 4. FleXray generalizes across acquisition conditions. a, Ensemble predictions on 18 unlabeled Radiopaedia radiographs (attributions in Appendix K.4). $\mathbf { b } ,$ Mean per-structure Dice versus geodesic distance from the frontal view on DeepFluoro (n=213 frames from two held-out subjects), grouped into $5 ^ { \circ }$ bins. Shading shows confidence intervals obtained by resampling label–frame pairs within each bin; pairs are included only when the ground-truth label occupies at least 0.1% of the image. $\bullet ,$ Per-rib Dice on VinDr-Rib $\scriptstyle ( n = 3 7$ images per rib) for FleXray (single) trained with generatively edited or unedited DRRs, showing improvements for the lower ribs. d, Example lung segmentations (left) and per-image Dice (right) on DarwinCVD19, comparing FleXray (single) with and without attenuation randomization against TotalSegmentator2D and PAXray. Results are stratified into healthy (n=241) and pneumonia (n=711) cases; brackets report p values between the two FleXray variants.

Ablating attenuation randomization. We now test whether per-label attenuation randomization (Methods $\mathsf { A } . 1 )$ improves robustness. Averaged across datasets, the effect is modest as equal-dataset mean Dice increases only from 0.863 to 0.870 with randomization. However, in pathological datasets, such as DarwinCVD19, we find that lung Dice increases from 0.907 to 0.922 $_ { ( n = 9 5 2 }$ $p { < } 0 . 0 0 1 )$ Stratifying by diagnosis (Figure 4d) shows that this gain is negligible in healthy lungs (0.945 to $0 . 9 4 6 , ~ n { = } 2 4 1 )$ but concentrated in pneumonia cases where the lungs are dense and Dice rises from 0.894 to 0.914 (n=711, $p { < } 0 . 0 0 1$ ; Welch t-test between groups, $p { < } 0 . 0 0 1 )$ . Correspondingly, the number of pneumonia images below 0.85 Dice falls from 122 to 50. Therefore, as intentionally designed, attenuation randomization improves robustness when pathology substantially alters radiographic density.

## FleXray unlocks diverse downstream clinical and scientific workflows.

We next evaluate whether FleXray is useful beyond segmentation itself. We consider three complementary modes of reuse: directly converting its anatomical masks into clinical measurements, using them as semantic priors within downstream workflows, and adapting it for new anatomical and pathological targets.

FleXray predictions quantify scoliosis severity within inter-rater variability. Scoliosis severity is routinely quantified from X-rays using the Cobb angle, a standard measure of spinal curvature [47, 48]. However, manual measurements vary by approximately $\mathtt { 3 - 5 ^ { \circ } }$ within observers and $6 \mathrm { - } 7 ^ { \circ }$ between observers [49, 50]. We therefore ask whether FleXray can recover this measurement directly from its vertebral segmentations on the Accurate Automated Spinal Curvature Estimation (AASCE) MICCAI 2019 challenge dataset [51], which provides reference Cobb-angle annotations on anteriorposterior spinal X-rays. Following clinical guidelines [50], we stratify curves as low $( \leq 2 0 ^ { \circ } )$ , moderate $( 2 0 { - } 4 0 ^ { \circ } )$ or severe $( > 4 0 ^ { \circ } )$ . Predictions with fewer than three usable vertebrae are reported separately as failures and the segmentation to Cobb angle conversion is detailed in Figure 5a and Methods D.1.

On AASCE, FleXray produces a measurable curve with no failures for all 218 images and achieves the lowest overall error, with a mean absolute error (MAE) of $5 . 1 3 ^ { \circ }$ [4.55, 5.74] (Figure 5b). TotalSegmentator2D fails on 1 image and reaches $6 . 3 8 ^ { \circ }$ [5.51, 7.32] MAE on the remainder, compared with 24.94<sup>◦</sup> [22.06, 27.91] for PAXray, which fails on 7 images, and 31.82<sup>◦</sup> [28.19, 35.58] for FluoroSAM, which fails on 42. FleXray also attains the lowest MAE in every severity stratum $( 2 . 0 8 ^ { \circ } , 4 . 1 2 ^ { \circ }$ , and $7 . 5 0 ^ { \circ }$ for low, moderate, and severe curves), significantly so in every comparison $( p { \le } 0 . 0 2 4 )$ , except against TotalSegmentator2D for severe curves $( 7 . 8 7 ^ { \circ } ; p { = } 0 . 7 0 )$

Importantly, FleXray’s error lies below the $6 \mathrm { - } 7 ^ { \circ }$ variability between expert observers. For moderate curves, which are particularly relevant to clinical planning [52], its $4 . 1 2 ^ { \circ } \mathsf { M A E }$ also falls within the $\mathtt { 3 - 5 ^ { \circ } }$ variability of repeated measurements by a single expert. Overall, 61% [55, 67] of estimates fall within $5 ^ { \circ }$ of the reference and 88% [83, 92] within $1 0 ^ { \circ }$ . Thus, general-purpose anatomical segmentations derived from FleXray can be used to power application-specific quantitative measurements and grading workflows.

FleXray segmentations improve the robustness of 2D/3D registration. Many image-guided procedures use preoperative 3D imaging for planning but rely on live 2D Xray fluoroscopy during intervention for navigation [53–58]. Aligning these images through 2D/3D registration can recover 3D anatomical context during navigation [59]. Currently, conventional methods optimize image similarity between a real intraoperative X-ray and a DRR rendered from a candidate 3D pose from the preoperative CT [45, 46, 60–64]. However, many of the same challenges with poor soft-tissue contrast and overlapping anatomy make this registration objective highly non-convex, leaving iterative registration highly susceptible to local minima without proper initialization [23, 65, 66].

We therefore use FleXray’s anatomical predictions as auxiliary semantic supervision for registration. Specifically, we add a bidirectional Chamfer loss [67, 68] that aligns distance transforms of structures segmented in the real X-ray with projections of the corresponding structures from the preoperative CT (Methods D.2). We incorporate this loss into xvr, an image-based iterative X-ray to volume registration pipeline [46]. Unlike image similarity, whose useful gradient can disappear when the DRR and X-ray have little overlap, the segmentation-based Chamfer term remains informative from substantially worse initial poses (Methods D.2).

We evaluate on all 213 held-out DeepFluoro X-rays with ground-truth C-arm poses starting from two possible initialization strategies (Figure 5c). Performance is measured using mean target registration error (mTRE) (Methods C.2) and gross failure rate (GFR), defined as the fraction of registrations with mTRE greater than 10 mm. From a manually selected frontal pose, a common clinical initialization [45], the median starting error is 324.5 mm with 100% GFR. Image-based optimization alone remains trapped at a median mTRE of 335.3 mm (86.4% GFR). However, adding the FleXray-enabled Chamfer loss reduces median mTRE to 1.2 mm (p<0.001) and GFR to 12.7% (Figure 5d).

a. Cobb angle estimation from FleXray masks  
![](images/52005826096d403017fd8a1510f093eb7d3393be3763f50402d0ae49ab056f34.jpg)

b.  
Major Cobb angle error by severity  
![](images/5af8ebb73399b46e0c9a1b768a538d14b7aeb49cc118b4095a146777e76ae131.jpg)

c.  
2D/3D registration examples  
![](images/26278d0beca38a3bfbb5f514189b5b9f244bccb2b975ee7b443709a3eb285335.jpg)

d. Registration error across pose initializations  
![](images/3608302b67d582a148c6c60b0747cdcac011b59aff465d11d3c18a4cee206876.jpg)  
Figure 5. FleXray enables downstream clinical tasks. a, From an AASCE X-ray, FleXray predicts vertebra masks, from which centerline-derived end-plate orientations yield the major Cobb angle. b, Absolute major Cobb-angle error by reference severity, over the images on which each method yields a measurable curve. Counts at the top give the number of unmeasurable images (failed/total), which are excluded from that method’s distribution. Mean errors are shown above the whiskers. c, Visualizations of the C-arm poses estimated by xvr and xvr with our FleXray-derived Chamfer loss term compared to the ground truth C-arm poses from the test split of the DeepFluoro dataset with two pose initializations: a manual frontal initial pose (top; [45]) and a foundation model initialization (bottom; [46]). Poses with mTRE >10 mm are shown in red. d, Mean target registration error (mTRE) for each X-ray for each initialization strategy and registration method. The dashed line represents the 10 mm success threshold. Boxes show the median, IQR, and Tukey whiskers at 1.5×IQR; all n=213 X-rays are shown as points. Gray trajectories denote the same sample evaluated by different methods.

To measure potential benefits over the state-of-theart, we next initialize optimization using a foundation model trained to predict C-arm pose directly from an X-ray [46], yielding a median initial mTRE of 76.6 mm (100% GFR). Image-based refinement already reduces GFR to 7.0%, but leaves eight catastrophic failures with errors between 93 and 1,628 mm. Adding the Chamfer term recovers all eight cases with mTRE >50 mm and brings each below 10 mm, reducing overall GFR to 3.3% and worst-case mTRE to 21 mm. This additional semantic loss does not sacrifice accuracy on cases that already register successfully: among the 198 frames solved by both methods, median mTRE remains 1.01 versus 1.04 mm (Figure 5d). Thus, FleXray automatically injects useful semantic information into existing 2D/3D registration methods, widening the capture range of conventional optimization while suppressing catastrophic failures from learned initializations.

FleXray adapts to unseen segmentation tasks with limited annotations. Finally, we ask whether FleXray provides a useful initialization for targets that were never part of its pretraining vocabulary. We study adaptation to fine-grained pelvic anatomy in pediatric hip dysplasia (MTDDH [70]) and bone-tumor detection (BTXRD [71]). Because accurate anatomical delineation supports hipdysplasia measurements [72], we evaluate MTDDH using Dice. For BTXRD, where the primary focus is detecting individual tumors, we use detection average precision at an intersection-over-union of 0.5 (AP50; Methods D.3). We replace the output head of FleXray (single) with a randomly initialized task-specific segmentation head and fine-tune the full network on 1%, 5%, 10%, 50%, or 100% of each training split. We compare against nnU-Net [69] trained from random initialization (Methods D.3), representing a well tuned segmentation baseline.

The benefit of initialization from FleXray is largest when labels are scarce (Figure 6). On MTDDH, using only 1% of the training set (6 patients), finetuned FleXray reaches a mean Dice of 0.896 [0.892, 0.900] versus 0.867 [0.852, 0.879] for nnU-Net, with the largest improvement on the ilium (0.948 [0.946, 0.951] versus 0.907 [0.893, 0.920]). As supervision increases, the gap asymptotically narrows, a finding consistent with the few-shot segmentation literature [73]. The harder BTXRD task shows a larger and more persistent advantage. With 10% of the training set (∼130 cases), finetuned FleXray reaches 23% AP50 versus 7% for nnU-Net, a difference of 16 percentage points [11, 22]. With only 1% (13 cases), performance is 5% versus 0.4%, and nnU-Net does not match FleXray’s 5%-budget AP50 until it receives 50% of the training set (Figure 6d).

## Discussion

Scaling generalist X-ray segmentation beyond manual annotations. Across all held-out segmentation datasets, FleXray matched or exceeded prior methods without requiring label-set-specific models (as in TotalSegmentator2D [30]) or per-structure prompts (FluoroSAM [31]), with no method being a consistent second place. FleXray’s performance also approached that of dataset-specific custom networks trained directly on the evaluation domains. These results suggest that successful synthetic training depends less on reproducing any single notion of realism than on achieving broad coverage in training of the anatomical, geometric, and appearance variation encountered at deployment. For example, physics-based projection alone transfers poorly to real radiographs, while appearance randomization accounts for most of the gain (Appendix E.2.3). Generative

X-ray enhancement provides additional benefit where analytic rendering is least faithful (Figure 4c), whereas real X-ray datasets of the limbs (constituting only 4% of training iterations) close gaps in the coverage offered by public CT datasets (Appendix E.2.4).

Making X-rays a quantitative modality. The broader value of FleXray is not just the segmentation masks themselves, but the quantitative workflows they make possible. For example, its vertebral predictions disease grading measurements to within inter-rater variability. In 2D/3D registration, the same anatomical predictions provide semantic gradients where image similarity becomes uninformative, removing all catastrophic optimization failures (mTRE >50 mm) without degrading successful registrations. For unseen segmentation targets, FleXray provides a robust initialization that is most useful when labels are scarce. These examples underscore a broader opportunity: one anatomy can be extracted reliably from routine radiographs, X-rays can support analyses that have traditionally been confined to time consuming or ionizing volumetric modalities such as MRI or CT, respectively. Further, retrospective archives and biobanks can now be mined for anatomical biomarkers, longitudinal changes, and associations between anatomy, disease, treatment, and outcomes without requiring dedicated prospective imaging or task-specific annotation for every new analysis or region of interest.

A reusable resource for full-body X-ray analysis. We release a first-of-its-kind enhanced DRR dataset and accompanying training resources (Appendix H). The collection varies anatomy, projection geometry, and appearance across the body and further includes qualitycontrolled generatively-enhanced DRRs, our manual forearm and humerus annotations for MURA, and seven re-distributable real X-ray sources in the format used by FleXray. We also fix the image-level splits and exclusions for every evaluation dataset, allowing future methods to be compared on identical held-out samples. Because our results indicate that performance depends strongly on the training distribution, we expect this resource to lower the barrier to the development of pan-anatomy X-ray models beyond the architecture studied here.

Relation to prior work. Prior DRR-based segmentation methods use a single projection model, either parallel [29] or cone-beam [31, 32], and, to our knowledge, train on a finite set of DRRs rendered before training. FleXray combines offline-generated DRRs with online rendering that varies projection models and acquisition parameters during training. Our attenuation randomization also differs from AnyCXR [32], which shares scaling factors across soft-tissue structures and independently perturbs only individual vertebrae and

FleXray Dice 0.961   
nnU-Net Dice 0.823

a. Overall MTDDH Dice b.  
![](images/23e99437f7747857e1cb989e158f20ee16df04b2dac73197c62cd7b5d2d5ed86.jpg)

![](images/eb780035da1b9f74f7d1ebfcd405e97d27ed57c75933ba5df7c4a385ef8522fa.jpg)

![](images/e9d9df9ae81c954560b65513e1e360bb663f5d0a0da5e873744630f2644626e5.jpg)  
Percentage of Total Training Data Available

MTDDH Dice by anatomical region  
![](images/57a2894ad270c796767ec5b5136a3ed556ad1b124582d0368a858935c4610b6f.jpg)

![](images/af21acc29143f5f44d2e34097d5e247bae955bac40d8f9d77d26941c6045eebe.jpg)

c.  
![](images/a42bfff41495c7893f1912f843bec15e37723c0bb70ce351f7b9ac65e377446c.jpg)

![](images/b6465bbc1914b524cecf5896bf9e4c0ec0bed833712d1ae53e9b2561a3a91705.jpg)

MTDDH segmentation differences at 1% training data  
![](images/4d82dfe3e71df14f8639976bb69da2042b3e715fc561691d0ada80f62defe7e5.jpg)

![](images/329042842bdf188695382d250f89a0f46d97ef85e768364a00a29648f1826331.jpg)

![](images/7958e924abd1d8b3136a7265343dab0b616045dbe1da8a8cbe49850d96aea65c.jpg)  
FleXray Dice 0.753nnU-Net Dice 0.574

## d. Overall BTXRD detection AP50 e.

![](images/929f652083b4275fc7dc7b6d8c0cd54d657e8b6dfc7cf17bb84080845f947ab9.jpg)

![](images/7c34f6d3168cf9f79a14e1ab0a51ba711a9bb20616ee72bb3c1d0dde1aa3fade.jpg)

BTXRD detection AP50 by lesion size  
![](images/ad0b4819a724328a7181682c677b9dc269b7a5b4560248ba479312855ccede66.jpg)

![](images/432ea3581eabca0f21374a0dab13f376a6bcd261239c3e9399acce24819f5e69.jpg)  
Percentage of Total Training Data Available  
Q4 · Largest boxes median area 4.36% · n = 92

![](images/3032b98f883a985e1ddc98e324f77f4160189e83e9bd86de8a31e3a4977fce56.jpg)

f.  
![](images/72225e4f81e3fbc7c597f24e315b3834e167ba2e302488d848008268b628ea8a.jpg)  
FleXray: 1/1 found · 0 FP nnU-Net: 0/1 found · 0 FP

![](images/d32cefd9399dc7900960c970f0ce929f38c6c0a602af54989e17dc66b0f1bc1a.jpg)

BTXRD detection example across training budgets  
![](images/982aba5f1ea13fe6f07e569de769888a1b36c4ce59e0b132f249d038658c39df.jpg)

![](images/be17af6857a925ca968fc9a40117b7d11e528d0e8cea1e70982e631c6dc6ab33.jpg)  
Figure 6. FleXray provides a data-efficient initialization for new anatomical and pathological targets. We fine-tune FleXray (single) and train a well tuned baseline (nnU-Net [69]) from random initialization using 1%, 5%, 10%, 50%, or 100% of each training split. a, Mean Dice across four MTDDH structures (n=135 test images) as a function of the fraction of 632 labeled training patients. b, Corresponding per-structure Dice. c, MTDDH segmentation differences at 1% training data for one test radiograph. Gray shows shared predictions, purple and teal show foreground predicted only by FleXray and nnU-Net, respectively, and dashed yellow contours show ground truth. Dice scores use the full image. d, Detection average precision at an intersection-over-union of 0.5 (AP50) on BTXRD, using fractions of 1,305 training images. The test set contains 354 tumors in 281 tumor-positive radiographs and no normal cases. e, BTXRD AP50 stratified by ground-truth bounding-box area. Quartile boundaries are defined on the training split, yielding 74, 105, 83, and 92 test tumors in successive size groups. f, Detections across annotation budgets for one test tumor from the third size quartile. Solid purple and teal boxes show fine-tuned FleXray and nnU-Net predictions on a shared crop; dashed yellow boxes show ground truth.

![](images/36e63ea3e1aedbab10c943005b3d11cdd161a166bab69c9f84d8402b5dfc5f1d.jpg)  
FleXray: 1/1 found · 0 FP nnU-Net: 1/1 found · 0 FP

ribs. We instead draw an independent multiplier for every labeled structure, increasing the diversity of relative contrast. Whereas prior generative enhancement uses image translation trained on unpaired real radiographs of the target anatomy [25], we apply a pretrained, general-purpose image editor [26] with no X-ray-specific training. Beyond these methodological differences, prior evaluations on real X-rays have been confined to one or two body regions, chiefly the chest [29, 31, 32] and pelvis [25]. To our knowledge, FleXray is the first automatic model evaluated and performant on real X-rays across the body, spanning eight held-out datasets of the limbs, pelvis, spine, and chest.

Limitations and future work. As X-rays were hard to annotate before the development of our model, our evaluation is limited by a low number of evaluation datasets with available ground truth. For example, none of our quantitative experiments include X-ray datasets with substantial clinical text overlays, despite their prevalence in real X-rays. Further, the data engine inherits biases from its CT sources, which image predominantly adults in supine poses with extremities out of the field-of-view. Targeted real X-rays partly compensate for these gaps, but FleXray can still fail on unseen acquisition geometries. Importantly, pathologies, implants, and surgical hardware are also not explicitly synthesized beyond those that are already present in the source CTs, leading to failures on severe trauma and other strongly out-ofdistribution cases (Figure 12). These limitations suggest a direct path forward: remaining coverage gaps can be addressed with targeted real or synthesized examples, while our finetuning experiments show that FleXray can adapt to new pathological targets using relatively few labeled radiographs.

Conclusion. FleXray demonstrates that broad X-ray segmentation can be learned largely from synthetic supervision when the training distribution is designed to cover the variation encountered in real radiographs. More importantly, the resulting predictions make routine X-rays amenable to quantitative measurement, downstream optimization, and data-efficient learning. Together with the released models, data engine, and dataset, this creates a foundation for the quantitative analysis of X-rays in clinical research, analogous to the advances enabled by general-purpose segmentation tools in CT and MRI [1, 6].

## References

[1] Jakob Wasserthal, Hanns-Christian Breit, Manfred T Meyer, Maurice Pradella, Daniel Hinck, Alexander W Sauter, Tobias Heye, Daniel T Boll, Joshy Cyriac, Shan Yang, et al. Totalsegmentator: robust segmentation of

104 anatomic structures in ct images. Radiology: Artificial Intelligence, 5(5):e230024, 2023.

[2] Tugba Akinci D’Antonoli, Lucas K Berger, Ashraya K Indrakanti, Nathan Vishwanathan, Jakob Weiss, Matthias Jung, Zeynep Berkarda, Alexander Rau, Marco Reisert, Thomas Küstner, et al. Totalsegmentator mri: robust sequence-independent segmentation of multiple anatomic structures in mri. Radiology, 314(2):e241613, 2025.

[3] Daria Ferrara, Manuel Pires, Sebastian Gutschmayer, Josef Yu, Yasser G Abdelhafez, Elisabetta Abenavoli, Ramsey D Badawi, Abhijit J Chaudhari, Moon S Chen Jr, Simon R Cherry, et al. Sharing a whole-/total-body [18f] fdg-pet/ct dataset with ct-derived segmentations: an enhance. pet initiative. Scientific Data, 2026.

[4] Marnee J McKay, Kenneth A Weber, Evert O Wesselink, Zachary A Smith, Rebecca Abbott, David B Anderson, Claire E Ashton-James, John Atyeo, Aaron J Beach, Joshua Burns, et al. Musclemap: an open-source, community-supported consortium for whole-body quantitative mri of muscle. Journal of imaging, 10(11):262, 2024.

[5] Daniel C Mann, Michael W Rutherford, Phillip Farmer, Joshua M Eichhorn, Fathima Fijula Palot Manzil, and Christopher P Wardell. Evaluating skellytour for automated skeleton segmentation from whole-body ct images. Radiology: Artificial Intelligence, 7(2):e240050, 2025.

[6] Benjamin Billot, Douglas N Greve, Oula Puonti, Axel Thielscher, Koen Van Leemput, Bruce Fischl, Adrian V Dalca, Juan Eugenio Iglesias, et al. Synthseg: Segmentation of brain mri scans of any contrast and resolution without retraining. Medical image analysis, 86:102789, 2023.

[7] Jean-Emmanuel Bibault and Paul Giraud. Deep learning for automated segmentation in radiotherapy: a narrative review. British Journal of Radiology, 97(1153):13–20, 2024.

[8] Sharif Elguindi, Michael J Zelefsky, Jue Jiang, Harini Veeraraghavan, Joseph O Deasy, Margie A Hunt, and Neelam Tyagi. Deep learning-based auto-segmentation of targets and organs-at-risk for magnetic resonance imaging only planning of prostate radiotherapy. Physics and imaging in radiation oncology, 12:80–86, 2019.

[9] Eric Pei Ping Pang, Hong Qi Tan, Fuqiang Wang, Jarkko Niemelä, Gregory Bolard, Susan Ramadan, Timo Kiljunen, Marta Capala, Steven Petit, Jan Seppälä, et al. Multicentre evaluation of deep learning ct autosegmentation of the head and neck region for radiotherapy. NPJ Digital Medicine, 8(1):312, 2025.

[10] Ijeamaka Anyene Fumagalli, Sidney T Le, Peter D Peng, Patricia Kipnis, Vincent X Liu, Bette Caan, Vincent Chow, Mirza Faisal Beg, Karteek Popuri, and Elizabeth M Cespedes Feliciano. Automated ct analysis of body composition as a frailty biomarker in abdominal surgery. JAMA surgery, 159(7):766–774, 2024.

[11] Vamsi Krishna Thiriveedhi, Deepa Krishnaswamy, David Clunie, Steve Pieper, Ron Kikinis, and Andrey Fedorov. Cloud-based large-scale curation of medical imaging data using ai segmentation. Research Square, pages rs–3, 2024.

[12] Bruce Fischl. Freesurfer. Neuroimage, 62(2):774–781, 2012.

[13] Constantin Marc Seibold, Simon Reiß, Jens Kleesiek, and Rainer Stiefelhagen. Reference-guided pseudolabel generation for medical semantic segmentation. In Proceedings of the AAAI conference on artificial intelligence, volume 36, pages 2171–2179, 2022.

[14] Matthias Lenga, Tobias Klinder, Christian Bürger, Jens von Berg, Astrid Franz, and Cristian Lorenz. Deep learning based rib centerline extraction and labeling. In International Workshop on Computational Methods and Clinical Applications in Musculoskeletal Imaging, pages 99–113. Springer, 2018.

[15] Barry Kelly. The chest radiograph. The Ulster medical journal, 81(3):143, 2012.

[16] Amin Tafti and Doug W Byerly. X-ray image acquisition. 2020.

[17] Michael P Federle, Tracy A Jaffe, Peter L Davis, Mahmoud M Al-Hawary, and Marc S Levine. Contrast media for fluoroscopic examinations of the gi and gu tracts: current challenges and recommendations. Abdominal Radiology, 42(1):90–100, 2017.

[18] Sara Guerri, Daniele Mercatelli, Maria Pilar Aparisi Gómez, Alessandro Napoli, Giuseppe Battista, Giuseppe Guglielmi, and Alberto Bazzocchi. Quantitative imaging techniques for the assessment of osteoporosis and sarcopenia. Quantitative imaging in medicine and surgery, 8 (1):60, 2018.

[19] Rishi P Mathew, Timothy Alexander, Vimal Patel, and Gavin Low. Chest radiographs of cardiac devices (part 1): Cardiovascular implantable electronic devices, cardiac valve prostheses and amplatzer occluder devices. SA journal of radiology, 23(1):1–13, 2019.

[20] Nicolás Gaggion, Candelaria Mosquera, Lucas Mansilla, Julia Mariel Saidman, Martina Aineseder, Diego H Milone, and Enzo Ferrante. Chexmask: a large-scale dataset of anatomical segmentation masks for multi-center chest xray images. Scientific Data, 11(1):511, 2024.

[21] Viacheslav Danilov, A Proutski, A Kirpich, DE Litmanovich, and Y Gankin. Chest x-ray dataset for lung segmentation. Mendeley Data, 10, 2022.

[22] Mathias Unberath, Jan-Nico Zaech, Sing Chun Lee, Bastian Bier, Javad Fotouhi, Mehran Armand, and Nassir Navab. Deepdrr–a catalyst for machine learning in fluoroscopy-guided procedures. In International conference on medical image computing and computer-assisted intervention, pages 98–106. Springer, 2018.

[23] Vivek Gopalakrishnan and Polina Golland. Fast autodifferentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging. In Workshop on Clinical Image-Based Procedures, pages 1–11. Springer, 2022.

[24] United Nations Scientific Committee on the Effects of Atomic Radiation. Sources, effects and risks of ionizing radiation: Unscear 2020/2021 report to the general assembly, with scientific annexes. volume i, scientific annex a: Evaluation of medical exposure to ionizing radiation, 2022.

[25] Cong Gao, Benjamin D Killeen, Yicheng Hu, Robert B Grupp, Russell H Taylor, Mehran Armand, and Mathias Unberath. Synthetic data accelerates the development of generalizable learning-based algorithms for x-ray image analysis. Nature Machine Intelligence, 5(3):294–308, 2023.

[26] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

[27] Ryoungwoo Jang, Jae Ho Choi, Namkug Kim, Jae Suk Chang, Pil Whan Yoon, and Chul-Ho Kim. Prediction of osteoporosis from simple hip radiography using deep learning algorithm. Scientific reports, 11(1):19997, 2021.

[28] Hooman Yarmohammadi, Esben Vogelius, Mariana Meyers, and Nami Azar. Vicarious urinary excretion of iodinated contrast in a crohn’s patient. Journal of Radiology Case Reports, 4(2):5, 2010.

[29] Constantin Seibold, Alexander Jaus, Matthias A Fink, Moon Kim, Simon Reiß, Ken Herrmann, Jens Kleesiek, and Rainer Stiefelhagen. Accurate fine-grained segmentation of human anatomy in radiographs via volumetric pseudo-labeling. arXiv preprint arXiv:2306.03934, 2023.

[30] Ahmed Alshenoudy, Bertram Sabrowsky-Hirsch, Stefan Thumfart, and Michael Giretzlehner. Leveraging synthetic data for whole-body segmentation in x-ray images. In Annual Conference on Medical Image Understanding and Analysis, pages 145–158. Springer, 2025.

[31] Benjamin D Killeen, Liam J Wang, Blanca Iñígo, Han Zhang, Mehran Armand, Russell H Taylor, Greg Osgood, and Mathias Unberath. Fluorosam: A languagepromptable foundation model for flexible x-ray image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 248–258. Springer, 2025.

[32] Zifei Dong, Wenjie Wu, Jinkui Hao, Tianqi Chen, Ziqiao Weng, and Bo Zhou. Anycxr: Human anatomy segmentation of chest x-ray at any acquisition position using multistage domain randomized synthetic data with imperfect annotations and conditional joint annotation regularization learning. arXiv preprint arXiv:2512.17263, 2025.

[33] Gašper Podobnik, Bulat Ibragimov, Elias Tappeiner, Chanwoong Lee, Jin Sung Kim, Zacharia Mesbah, Romain Modzelewski, Yihao Ma, Fan Yang, Mikołaj Rudecki, et al. Han-seg: the head and neck organ-at-risk ct and

mr segmentation challenge. Radiotherapy and Oncology, 198:110410, 2024.

[34] Li Cheng. Automatically transform CT datasets into DRRs. https://www.kaggle.com/datasets/syxlicheng/ automatically-transform-ct-datasets-into-drrs, 2021.

[35] Hui Ming Lin, Errol Colak, Tyler Richards, Felipe C Kitamura, Luciano M Prevedello, Jason Talbott, Robyn L Ball, Ekim Gumeler, Kristen W Yeom, Mohammad Hamghalam, et al. The rsna cervical spine fracture ct dataset. Radiology: Artificial Intelligence, 5(5):e230034, 2023.

[36] Petr Jordan, Philip M Adamson, Vrunda Bhattbhatt, Surabhi Beriwal, Sangyu Shen, Oskar Radermecker, Supratik Bose, Linda S Strain, Michael Offe, David Fraley, et al. Pediatric chest-abdomen-pelvis and abdomenpelvis ct images with expert organ contours. Medical physics, 49(5):3523–3528, 2022.

[37] Jun Wang, Yan Wu, Yongxing Zhang, Xiying Ding, and Qing Zhang. 3D models of elbow joints along with corresponding CT data from Chinese individuals. figshare, 2026. URL https : / / doi . org / 10 . 6084 / m9 . figshare . 28245599.v2. Dataset, Version 2.

[38] zanajn. "hand bones" dataset. https : / / universe . roboflow . com / zanajn / -hand-bones-mdjkr-k4rcb, oct 2025. URL https : / / universe . roboflow . com / zanajn / -hand-bones-mdjkr-k4rcb. visited on 2026-06-24.

[39] monchbot1. foot\_op dataset. https://universe.roboflow. com/monchbot1/foot\_op, jan 2026. URL https://universe. roboflow.com/monchbot1/foot\_op. visited on 2026-06-24.

[40] Pranav Rajpurkar, Jeremy Irvin, Aarti Bagul, Daisy Ding, Tony Duan, Hershel Mehta, Brandon Yang, Kaylie Zhu, Dillon Laird, Robyn L Ball, et al. Mura: Large dataset for abnormality detection in musculoskeletal radiographs. arXiv preprint arXiv:1712.06957, 2017.

[41] Shobhita Sundaram and Neha Hulkund. Gan-based data augmentation for chest x-ray classification. arXiv preprint arXiv:2107.02970, 2021.

[42] Pierre Chambon, Christian Bluethgen, Jean-Benoit Delbrouck, Rogier Van der Sluijs, Małgorzata Połacin, Juan Manuel Zambrano Chaves, Tanishq Mathew Abraham, Shivanshu Purohit, Curtis P Langlotz, and Akshay Chaudhari. Roentgen: vision-language foundation model for chest x-ray generation. arXiv preprint arXiv:2211.12737, 2022.

[43] Fabio De Sousa Ribeiro, Emma AM Stanley, Charles Jones, Tian Xia, Dominic C Marshall, Laurent Renard Triché, Christopher V Cosgriff, Panagiotis Dimitrakopoulos, Sotirios A Tsaftaris, and Ben Glocker. Scaling generative foundation models for chest radiography with rectified flow transformers. arXiv preprint arXiv:2606.19460, 2026.

[44] Ramandeep Singh, Mannudeep K Kalra, Chayanin Nitiwarangkul, John A Patti, Fatemeh Homayounieh, Atul Padole, Pooja Rao, Preetham Putha, Victorine V Muse,

Amita Sharma, et al. Deep learning in chest radiography: detection of findings and presence of change. PloS one, 13(10):e0204155, 2018.

[45] Robert B Grupp, Mathias Unberath, Cong Gao, Rachel A Hegeman, Ryan J Murphy, Clayton P Alexander, Yoshito Otake, Benjamin A McArthur, Mehran Armand, and Russell H Taylor. Automatic annotation of hip anatomy in fluoroscopy for robust and efficient 2d/3d registration. International journal of computer assisted radiology and surgery, 15:759–769, 2020.

[46] Vivek Gopalakrishnan, David-Dimitris Chlorogiannis, Andrew Abumoussa, Anna M Larson, Nazim Haouchine, Darren B Orbach, Sarah Frisken, Neel Dey, and Polina Golland. Rapid patient-specific neural networks for x-ray to volume registration. Nature, 2026.

[47] S Langensiepen, O Semler, R Sobottke, O Fricke, J Franklin, E Schönau, and P Eysel. Measuring procedures to determine the cobb angle in idiopathic scoliosis: a systematic review. European Spine Journal, 22 (11):2360–2371, 2013.

[48] John Cobb. Outline for the study of scoliosis. Instructional course lecture, 1948.

[49] RAYMOND T Morrissy, GS Goldsmith, EC Hall, D Kehl, and GH Cowie. Measurement of the cobb angle on radiographs of patients who have scoliosis. evaluation of intrinsic error. JBJS, 72(3):320–327, 1990.

[50] Stefano Negrini, Sabrina Donzelli, Angelo Gabriele Aulisa, Dariusz Czaprowski, Sanja Schreiber, Jean Claude De Mauroy, Helmut Diers, Theodoros B Grivas, Patrick Knott, Tomasz Kotwicki, et al. 2016 sosort guidelines: orthopaedic and rehabilitation treatment of idiopathic scoliosis during growth. Scoliosis and spinal disorders, 13(1):3, 2018.

[51] Liansheng Wang, Cong Xie, Yi Lin, Hong-Yu Zhou, Kailin Chen, Dalong Cheng, Florian Dubost, Benjamin Collery, Bidur Khanal, Bishesh Khanal, et al. Evaluation and comparison of accurate automated spinal curvature estimation algorithms with spinal anterior-posterior x-ray images: The aasce2019 challenge. Medical image analysis, 72:102115, 2021.

[52] B Stephens Richards, Robert M Bernstein, Charles R D’Amato, and George H Thompson. Standardization of criteria for adolescent idiopathic scoliosis brace studies: Srs committee on bracing and nonoperative management. Spine, 30(18):2068–2075, 2005.

[53] Andrew Abumoussa, Vivek Gopalakrishnan, Benjamin Succop, Michael Galgano, Sivakumar Jaikumar, Yueh Z Lee, and Deb A Bhowmick. Machine learning for automated and real-time two-dimensional to threedimensional registration of the spine using a single radiograph. Neurosurgical Focus, 54(6):E16, 2023.

[54] Roshan Ramakrishna Naik, Anitha Hoblidar, Shyamasunder N Bhat, Nishanth Ampar, and Raghuraj Kundangar. A hybrid 3d-2d image registration framework for pedicle

screw trajectory registration between intraoperative X-ray image and preoperative CT image. Journal of Imaging, 8 (7):185, 2022.

[55] Coert T Metz, Michiel Schaap, Stefan Klein, Lisan A Neefjes, Ermanno Capuano, Carl Schultz, Robert Jan Van Geuns, Patrick W Serruys, Theo Van Walsum, and Wiro J Niessen. Patient specific 4d coronary models from ecg-gated cta data for intra-operative dynamic alignment of cta with X-ray images. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 369–376. Springer, 2009.

[56] Martin Wagner, Sebastian Schafer, Charles Strother, and Charles Mistretta. 4D interventional device reconstruction from biplane fluoroscopy. Medical Physics, 43(3):1324– 1334, 2016.

[57] Elizabeth Huynh, Ahmed Hosny, Christian Guthier, Danielle S Bitterman, Steven F Petit, Daphne A Haas-Kogan, Benjamin Kann, Hugo JWL Aerts, and Raymond H Mak. Artificial intelligence in radiation oncology. Nature Reviews Clinical Oncology, 17(12):771–781, 2020.

[58] Yoonho Kim, Emily Genevriere, Pablo Harker, Jaehun Choe, Marcin Balicki, Robert W Regenhardt, Justin E Vranic, Adam A Dmytriw, Aman B Patel, and Xuanhe Zhao. Telerobotic neurovascular interventions with magnetic manipulation. Science Robotics, 7(65):eabg9907, 2022.

[59] Mathias Unberath, Cong Gao, Yicheng Hu, Max Judish, Russell H Taylor, Mehran Armand, and Robert Grupp. The impact of machine learning on 2D/3D registration for image-guided interventions: A systematic review and perspective. Frontiers in Robotics and AI, 8:716007, 2021.

[60] Graeme P Penney, Jürgen Weese, John A Little, Paul Desmedt, Derek LG Hill, et al. A comparison of similarity measures for use in 2D/3D medical image registration. IEEE Transactions on Medical Imaging, 17(4):586–595, 1998.

[61] Dotan Knaan and Leo Joskowicz. Effective intensitybased 2D/3D rigid registration between fluoroscopic X-ray and CT. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 351–358. Springer, 2003.

[62] Robert B Grupp, Rachel A Hegeman, Ryan J Murphy, Clayton P Alexander, Yoshito Otake, Benjamin A McArthur, Mehran Armand, and Russell H Taylor. Pose estimation of periacetabular osteotomy fragments with intraoperative X-ray navigation. IEEE Transactions on Biomedical Engineering, 67(2):441–452, 2019.

[63] Bastian Bier, Florian Goldmann, Jan-Nico Zaech, Javad Fotouhi, Rachel Hegeman, Robert Grupp, Mehran Armand, Greg Osgood, Nassir Navab, Andreas Maier, et al. Learning to detect anatomical landmarks of the pelvis in x-rays from arbitrary views. International Journal of Computer Assisted Radiology and Surgery, 14:1463– 1473, 2019.

[64] Pragyan Shrestha, Chun Xie, Yuichi Yoshii, and Itaru Kitahara. Rayemb: Arbitrary landmark detection in X-ray images using ray embedding subspace. In Proceedings of the Asian Conference on Computer Vision, pages 665– 681, 2024.

[65] Wenhao Gu, Cong Gao, Robert Grupp, Javad Fotouhi, and Mathias Unberath. Extended capture range of rigid 2D/3D registration by estimating Riemannian pose gradients. In International Workshop on Machine Learning in Medical Imaging, pages 281–291. Springer, 2020.

[66] Cong Gao, Anqi Feng, Xingtong Liu, Russell H Taylor, Mehran Armand, and Mathias Unberath. A fully differentiable framework for 2D/3D registration and the projective spatial transformers. IEEE Transactions on Medical Imaging, 2023.

[67] RC Bolles HG Barrow JM Tenenbaum, RC Bolles, HC Wolf, et al. Parametric correspondence and chamfer matching: Two new techniques for image matching. In Proceedings of the 5th international joint conference on Artificial intelligence, pages 659–663, 1977.

[68] Gunilla Borgefors. Distance transformations in digital images. Computer vision, graphics, and image processing, 34(3):344–371, 1986.

[69] Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnu-net: a selfconfiguring method for deep learning-based biomedical image segmentation. Nature methods, 18(2):203–211, 2021.

[70] Guoqiang Qi, Xiongfei Jiao, Jing Li, Chaojin Qin, Xinxin Li, Zhexian Sun, Yonggen Zhao, Renjie Jiang, Zhu Zhu, Guoqiang Zhao, and Gang Yu. A dataset for quality evaluation of pelvic x-ray and diagnosis of developmental dysplasia of the hip. Scientific Data, 12:865, 2025. doi: 10.1038/s41597-025-05146-x.

[71] Shunhan Yao, Yuanxiang Huang, Xiaoyu Wang, Yiwen Zhang, Ian Costa Paixao, Zhikang Wang, Charla Lu Chai, Hongtao Wang, Dinggui Lu, Geoffrey I Webb, et al. A radiograph dataset for the classification, localization, and segmentation of primary bone tumors. Scientific data, 12 (1):88, 2025.

[72] D Tönnis. Normal values of the hip joint for the evaluation of x-rays in children and adults. Clinical orthopaedics and related research, (119):39–47, 1976.

[73] Neel Dey, Benjamin Billot, Hallee E. Wong, Clinton Wang, Mengwei Ren, Ellen Grant, Adrian V Dalca, and Polina Golland. Learning general-purpose biomedical volume representations using randomized synthesis. In The Thirteenth International Conference on Learning Representations, 2025. URL https : // openreview. net / forum?id=xOmC5LiVuN.

[74] Robert L Siddon. Fast calculation of the exact radiological path for a three-dimensional ct array. Medical physics, 12 (2):252–255, 1985.

[75] Christophe Leys, Christophe Ley, Olivier Klein, Philippe Bernard, and Laurent Licata. Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. Journal of experimental social psychology, 49(4):764–766, 2013.

[76] Peter J Rousseeuw and Mia Hubert. Robust statistics for outlier detection. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 1(1):73–79, 2011.

[77] Lalith Kumar Shiyam Sundar, Josef Yu, Otto Muzik, Oana C Kulterer, Barbara Fueger, Daria Kifjak, Thomas Nakuz, Hyung Min Shin, Annika Katharina Sima, Daniela Kitzmantl, et al. Fully automated, semantic segmentation of whole-body 18f-fdg pet/ct images based on datacentric artificial intelligence. Journal of Nuclear Medicine, 63(12):1941–1948, 2022.

[78] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.

[79] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[80] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017.

[81] Agrim Gupta, Piotr Dollar, and Ross Girshick. Lvis: A dataset for large vocabulary instance segmentation. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5351–5359. IEEE, 2019.

[82] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens Van Der Maaten. Exploring the limits of weakly supervised pretraining. In European conference on computer vision, pages 185– 201. Springer, 2018.

[83] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565– 571. Ieee, 2016.

[84] V7 Labs and CloudFactory. COVID-19 chest xray dataset. https : / / darwin . v7labs . com / v7-labs / covid-19-chest-x-ray-dataset, 2020. URL https://darwin. v7labs.com/v7-labs/covid-19-chest-x-ray-dataset. visited on 2026-07-18.

[85] ionspace. elbow\_lat dataset. https : / / universe . roboflow.com/ionspace/elbow\_lat-lnn0s-pmycd, jul 2026. URL https : / / universe . roboflow. com / ionspace / elbow\_ lat-lnn0s-pmycd. visited on 2026-07-18.

[86] Daniel Gut. X-ray images of the hip joints. Mendeley Data, 1, 2021.

[87] OrthopedicStitching. Bone identifier dataset. https : / / universe . roboflow . com / orthopedicstitching / bone-identifier-1rey5, jan 2026. URL https://universe. roboflow. com / orthopedicstitching / bone-identifier-1rey5. visited on 2026-06-24.

[88] Songxiao Yang, Haolin Wang, Yao Fu, Ye Tian, Tamotsu Kamishima, Masayuki Ikebe, Yafei Ou, and Masatoshi Okutomi. Ram-w600: A multi-task wrist dataset and benchmark for rheumatoid arthritis. arXiv preprint arXiv:2507.05193, 2025.

[89] monchbot1. Thoracoabdominal dataset. Roboflow Universe dataset export, March 2026. URL https://universe. roboflow.com/monchbot1/thoracoabdominal. Dataset of 78 images with annotations exported in COCO format on March 18, 2026. The original Roboflow Universe page is no longer publicly available.

[90] Hoang C Nguyen, Tung T Le, Hieu H Pham, and Ha Q Nguyen. Vindr-ribcxr: A benchmark dataset for automatic segmentation and labeling of individual ribs on chest xrays. arXiv preprint arXiv:2107.01327, 2021.

[91] Lee R. Dice. Measures of the amount of ecologic association between species. Ecology, 26(3):297–302, 1945. doi: 10.2307/1932409.

[92] Daniel P. Huttenlocher, Gregory A. Klanderman, and William J. Rucklidge. Comparing images using the hausdorff distance. IEEE Transactions on Pattern Analysis and Machine Intelligence, 15(9):850–863, 1993. doi: 10.1109/34.232073.

[93] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[94] Divya Shanmugam, Davis Blalock, Guha Balakrishnan, and John Guttag. When and why test-time augmentation works. arXiv preprint arXiv:2011.11156, 1(3):4, 2020.

[95] Divya Shanmugam, Davis Blalock, Guha Balakrishnan, and John Guttag. Better aggregation in test-time augmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1214–1223, 2021.

[96] Yaling Pan, Qiaoran Chen, Tongtong Chen, Hanqi Wang, Xiaolei Zhu, Zhihui Fang, and Yong Lu. Evaluation of a computer-aided method for measuring the cobb angle on chest x-rays. European Spine Journal, 28(12):3035– 3043, 2019.

[97] Michel C Delfour and J-P Zolésio. Shapes and geometries: metrics, analysis, differential calculus, and optimization. SIAM, 2011.

[98] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos, 2024. URL https://arxiv. org/abs/2408.00714, 3, 2024.

[99] Sergios Gatidis, Tobias Hepp, Marcel Früh, Christian La Fougère, Konstantin Nikolaou, Christina Pfannenberg, Bernhard Schölkopf, Thomas Küstner, Clemens Cyran, and Daniel Rubin. A whole-body fdg-pet/ct dataset with manually annotated tumor lesions. Scientific Data, 9(1): 601, 2022.

[100] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

[101] Jose Javier Gonzalez Ortiz. The thunderpack data format. url<https://github.com/JJGO/thunderpack/>, 2023.

[102] Charles R Harris, K Jarrod Millman, Stéfan J Van Der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J Smith, et al. Array programming with numpy. nature, 585(7825):357–362, 2020.

[103] Edgar Riba, Dmytro Mishkin, Daniel Ponsa, Ethan Rublee, and Gary Bradski. Kornia: an open source differentiable computer vision library for pytorch. In 2020 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 3663–3672. IEEE, 2020.

[104] CVAT.ai Corporation. Computer vision annotation tool (CVAT), 2023. URL https://github.com/cvat-ai/cvat.

Frontal Lung Projection

## Methods

A Training data engine 18   
A.1 DRRs generated online from CT scans . 18   
A.2 Diffusion-enhanced synthetic X-rays 18   
A.3 Annotated real X-rays 20   
A.4 Label harmonization 20   
A.5 Data augmentation 20   
A.6 Preprocessing 21   
B Training 22   
B.1 Network architecture 22   
B.2 Optimization 22   
C Evaluation 23   
C.1 Protocol post-processing 24   
C.2 Metrics 24   
C.3 Test-time augmentation 25   
C.4 Baselines 25   
D Downstream tasks 26   
D.1 Cobb-angle estimation 26   
D.2 2D/3D registration . 26   
D.3 Data-efficient finetuning 27

## A Training data engine

We train FleXray using a mixture of CT and X-ray data sampled from three data sources:

1. DRRs rendered online from CT scans (Methods A.1).

2. Diffusion-enhanced synthetic X-rays (Methods A.2).

3. Annotated real X-rays (Methods A.3).

We sample each dataset according to the mixture weights in Table 1.

## A.1 DRRs generated online from CT scans

Randomly sampling DRRs and their labels. We render DRRs by integrating CT attenuation along sourceto-detector rays [22, 74], following established conventions [46]. For each view, we randomly sample the source-to-detector distance SDD, detector spacing ∆, C-arm rotational parameters $\theta = [ \alpha , \beta , \gamma ]$ C-arm translational parameters $\tau ~ = ~ [ x , y , z ]$ and projection geometry (cone-beam or orthographic) from dataset-specific ranges (Table 2). Because y and SDD are drawn independently, we sort each pair so that $y \le \mathrm { S D D }$ , and choose the per-dataset SDD ranges to prevent the detector plane from intersecting with the body. For a CT volume, we place the view isocenter c either at the volume’s isocenter or at the centroid of a randomly sampled foreground label. As CT volumes in the MOOSE dataset span a larger FOV, we place the isocenter at the center of a randomly sampled foreground structure (Table 2).

![](images/1f4f760cd18fa0a4aff1ad0e1eb02e496495a9faa96c02183193697ddc791927.jpg)  
Figure 7. Lung label projection threshold. At a threshold of 0, every pixel whose ray intersects any lung tissue is marked, including retro-cardiac and sub-diaphragmatic lung. Raising the threshold to 0.1 restricts the mask to the radiographically visible lung fields.

Segmentation masks are analytically rendered in a similar fashion to DRRs. Each projected label accumulates the attenuation contributed by casting rays passing through voxels containing that label, so thresholding the projection above 0 produces a binary 2D segmentation of that structure. The one exception to thresholding at 0 is lungs, where we use a higher threshold of 0.1 to restrict the lung mask to regions that are more aligned with existing X-ray lung masks (Figure 7).

Label-based attenuation randomization. Similar to recent work in chest X-ray segmentation using CT labels [32], we use the existing segmentation supervision to simulate differences in attenuation between organs. We scale the attenuation of every voxel in each 3D foreground label $\ell \in \{ 1 , \ldots , L \}$ by a multiplier drawn independently per label from a truncated log-normal distribution:

$$
m _ { \ell } \sim \mathrm { T r u n c L o g N o r m a l } \left( \eta , \sigma _ { m } ; \left[ 0 . 1 , 4 . 0 \right] \right) ,\tag{1}
$$

where $\eta = 1 . 0$ is the mode of the underlying log-normal $( \mathsf { i } . \mathsf { e } . , \mu = \ln \eta + \sigma _ { m } ^ { 2 } )$ and $\sigma _ { m } = 1 . 0$ is its log-space standard deviation; draws outside [0.1, 4.0] are rejected and redrawn. Attenuation randomization is applied with probability 0.5 per CT batch. We sample the per-label multipliers once per batch and shared by them amongst all rendered views.

## A.2 Diffusion-enhanced synthetic X-rays

DRRs can differ substantially in appearance from clinical X-rays [25]. To reduce this domain shift, we further generate a dataset of enhanced DRRs by applying a pretrained text-conditioned image editing diffusion model. We apply flux.2-klein-9b [26] (Flux) to DRRs sampled from the whole-body MOOSE CT dataset, while retaining the DRRs’ projected segmentation labels. This generative image editing is done offline so as to not bottleneck segmentation network training. This data source is constructed in three stages: rendering, enhancement, and filtering.

Table 1. Training data sources and mixture proportions. Proportion is the probability that a batch is drawn from each data source at each training step (proportions sum to 100%). The online MOOSE and enhanced-DRR proportions are paired to sum to 75%, with the remaining sources contributing 25%. Counts are CT volumes for CT sources and radiographs for X-ray sources; enhanced-DRR counts refer to source CT subjects. All splits are subject-disjoint. CT-derived sources reserve no test split because all evaluation is performed on the held-out real X-ray datasets (Table 4). Enhanced DRRs is a dataset of DRRs pre-rendered from MOOSE and refined with the diffusion model flux.2-klein-9b [26].
<table><tr><td>Dataset</td><td>Proportion (%)</td><td>Source</td><td>Train / Val / Test</td><td>Supervised labels</td></tr><tr><td>MOOSE [3]</td><td>{0, 25, 37.5, 50, 75}</td><td>CT</td><td>1,437 / 160 / 0</td><td>Whole-body skeleton (skull, long bones, ribs 1–12, verte- brae C1-L5, hips, sacrum) and thoracoabdominal organs (lungs, heart, liver, spleen, kidneys)</td></tr><tr><td>HaN-Seg [33]</td><td>5.0</td><td>CT</td><td>37/5/0</td><td>MOOSE label space</td></tr><tr><td>Shoulder-CT [34]</td><td>5.0</td><td>CT</td><td>16/2/0</td><td>MOOSE label space</td></tr><tr><td>RSNAFrac [35]</td><td>4.0</td><td>CT</td><td>78/9/0</td><td>Skull, cervical and upper thoracic (T1–T7) vertebrae</td></tr><tr><td>PedsCT [36]</td><td>4.0</td><td>CT</td><td>323 /36/ 0</td><td>Kidneys, liver, spleen, lungs, heart</td></tr><tr><td>ElbowCT [37]</td><td>3.0</td><td>CT</td><td>49/7/0</td><td>Humeri, ulnae</td></tr><tr><td>HandBones [38]</td><td>1.0</td><td>X-ray</td><td>65/15/ 13</td><td>Carpals, phalanges, metacarpals, radii, ulnae</td></tr><tr><td>FootBones [39]</td><td>1.0</td><td>X-ray</td><td>400 /86 / 85</td><td>Metatarsals, toes</td></tr><tr><td>MURA Forearm [40]</td><td>1.0</td><td>X-ray</td><td>35/8/7</td><td>Humeri, radii, ulnae</td></tr><tr><td>MURA Humerus [40]</td><td>1.0</td><td>X-ray</td><td>35/8/7</td><td>Humeri, radii, ulnae</td></tr><tr><td>Enhanced DRRs</td><td>{0, 25, 37.5, 50, 75}</td><td>Diffusion-refined DRR</td><td>1,437 / 160 / 0</td><td>MOOSE label space</td></tr></table>

Fixed-pose rendering. We render 90 DRRs from each MOOSE CT volume from a fixed set of C-arm poses, producing image-segmentation pairs with high heterogeneity in anatomical FOV. Following the camera parameterization of Table 2, the grid is the Cartesian product of translations $y \in \{ 6 5 0 , 9 0 0 , 1 1 5 0 \}$ mm and $z \in \{ - 3 0 0 , - 2 0 0$ −100, 0, 100, 200} mm with rotations $\alpha \in \{ 0 , \pm 4 5 , \pm 9 0 \} ^ { \circ }$ holding $x = 0$ mm and $\beta = \gamma = 0 ^ { \circ } , \mathsf { f o r } 3 { \times } 6 { \times } 5 = 9 0$ poses per volume. The raw render stream uses an SDD of 1250, detector spacing $0 . 6 \times 0 . 6$ , output size 1024×1024, the volume center as the isocenter, and fixed (default) attenuation. During training, each enhanced DRR draw selects a subject and then one of its poses uniformly at random.

Diffusion model-based enhancement. We use Flux to edit each DRR with 4 denoising steps, bfloat16, guidance scale 1.0, and the following anatomy-preserving prompt:

“Make this synthetic X-ray (generated by a projection through a CT) look like a real X-ray. The generated image should preserve ALL anatomical structure, no shape changes at all. Add a small amount of clinical text to the image typical of X-rays but do not change the position or location of the anatomy at all. Also, add some realistic X-ray textures to the image. The output must be well registered with the input.”

We found that the final sentence in the prompt was crucial for preserving alignment between the images and their labels.

Hallucination filtering. Despite explicit instructions in the prompt, Flux can occasionally alter projected anatomy (particularly for non-standard views), introducing disagreement between the enhanced image and inherited segmentation labels. We apply the following strategies to filter out these failures:

Foreground filtering. For an enhanced DRR and its corresponding raw DRR, we set to zero enhanced pixels whose corresponding raw DRR intensity is <0.05. This suppresses anatomy synthesized outside of the projected body.

Paired difference filtering. A U-Net $\phi ,$ trained exclusively on raw MOOSE-derived DRRs (no real X-rays, enhanced DRRs, or other datasets), is applied to each pre-rendered DRR $x _ { d r r }$ and enhanced DRR $x _ { f l u x } ,$ both sharing the same projected ground truth $y ,$ yielding predictions $\phi ( x _ { d r r } ) ~ = ~ \hat { y } _ { d r r }$ and $\phi ( x _ { f l u x } ) ~ = ~ \hat { y } _ { f l u x } .$ We compute $s _ { d r r } ~ = ~ \mathsf { S o f t D i c e } ( y , \hat { y } _ { d r r } )$ and $s _ { f l u x } ~ = ~ \mathsf { S o f t D i c e } ( y , \hat { y } _ { f l u x } )$ (empty labels ignored and background included) and consider the paired difference $\Delta = s _ { d r r } - s _ { f l u x }$

We frame filtering as a per-sample, one-sided outlier rule. Within the group of enhanced DRRs that share the same pose $p ,$ benign enhancement draws $\Delta$ from a null distribution assumed symmetric about its median. We set an upper-tail cutoff by mirroring the empirical 2nd percentile about the median, a robust nonparametric fence in the spirit of median-based outlier rejection [75, 76]. A sample is kept if $\Delta \leq \ell _ { p } .$ , where

Table 2. Online DRR rendering settings per training CT source. Shared defaults apply to all datasets. Each dataset specifies its own source-to-detector distance (SDD), detector pixel spacing $\Delta ,$ projection mode, in-plane rotation $\gamma ,$ and translation ranges. Camera rotations are ZXY Euler angles: α rotates about the patient’s long axis, β is the out-of-plane elevation, and $\gamma$ is the in-plane detector roll. Projection-mode values are sampling probabilities, drawn once per batch of rendered views. All values are sampled uniformly from the listed ranges.
<table><tr><td>Shared defaults Setting Value</td><td colspan="5"></td></tr><tr><td>Rotation range</td><td colspan="6"> $\alpha \in [ - 1 8 0 ^ { \circ } , 1 8 0 ^ { \circ } ] , \beta \in [ - 3 0 ^ { \circ } , 3 0 ^ { \circ } ] , \gamma = 0 ^ { \circ }$  unless specified below</td></tr><tr><td>Detector 256×256 pixels</td><td colspan="6"></td></tr><tr><td>Dataset-specific geometry Dataset</td><td>Isocenter</td><td>SDD (mm)</td><td>∆ (mm/pixel)</td><td>Projection mode</td><td>γ (deg)</td><td>Translation (mm) x ∈ [−100, 100],</td></tr><tr><td>MOOSE</td><td>random label</td><td>[1150, 1250]</td><td>[1.25, 2.25]</td><td>cone 0.9 / orthographic 0.1</td><td>0</td><td>y ∈ [500, 900], z∈[−100,100] x ∈ [−50, 50],</td></tr><tr><td>HaN-Seg</td><td>volume center</td><td>[1150, 1250]</td><td>[1.0, 2.0]</td><td>cone only</td><td>[-45, 45]</td><td>y ∈ [500, 800], z∈[-100,100] x ∈ [−50, 50],</td></tr><tr><td>Shoulder-CT</td><td>volume center</td><td>[950, 1050]</td><td>[1.9, 2.0]</td><td>cone only</td><td>[-45,45]</td><td>y ∈ [400, 600], z∈[-100,100] x ∈ [−50, 50],</td></tr><tr><td>RSNAFrac</td><td>volume center</td><td>[950, 1050]</td><td>[1.0, 2.0]</td><td>cone only</td><td>[-45, 45]</td><td>y ∈ [500, 800], z∈[−100,100] x ∈ [−50, 50],</td></tr><tr><td>PedsCT</td><td>volume center</td><td>[1050, 1150]</td><td>[1.25, 2.0]</td><td>cone 0.9 / orthographic 0.1</td><td>0</td><td>y ∈ [700, 800], z∈[-50,50]</td></tr><tr><td>ElbowCT</td><td>volume center</td><td>[1050, 1150]</td><td>[1.25, 2.0]</td><td>cone only</td><td>[-45, 45]</td><td>x = 0, y ∈ [500, 800], z∈[-50,50]</td></tr></table>

$$
\ell _ { p } = 2 \mathrm { m e d i a n } ( \{ \Delta \} _ { p } ) - q _ { 0 . 0 2 } ( \{ \Delta \} _ { p } ) ,\tag{2}
$$

with median $( \{ \Delta \} _ { p } )$ the median difference within the group and $q _ { 0 . 0 2 } ( \{ \Delta \} _ { p } )$ its 2nd percentile. The cutoff targets the upper 2% of a symmetric reference distribution (Figure 8a).

## A.3 Annotated real X-rays

The real X-ray datasets supplement CT-derived supervision with peripheral bone anatomy that is poorly represented in available CT datasets: HandBones, which covers diverse hand segmentations; FootBones, which has annotations for toes and metatarsals; and the MURA forearm and humerus subsets, which we annotated ourselves (Appendix K.2). Adding these real X-rays significantly improved performance on held-out wrist and elbow X-ray datasets (Appendix E.2.4).

## A.4 Label harmonization

FleXray segments 60 structures: the skull, sternum, sacrum, and hip bones; the scapulae, clavicles, humeri, radii, ulnae, carpals, metacarpals, and hand phalanges; the femurs, patellae, tibiae, fibulae, tarsals, metatarsals, and toes; each rib pair individually (ribs 1–12); each vertebra individually (C1 – C7, T1 – T12, L1 – L5); and the lungs, heart, liver, kidneys, and spleen. Bilateral structures share one channel (e.g., the left and right femur are both “femurs”).

For each dataset, we define a mapping from its native labels to this label protocol. The mapping merges distinctions finer than ours (e.g., left and right rib labels into single per-rib channels, heart substructures into heart) and sends structures our protocol leaves out to background, dropping them from supervision.

## A.5 Data augmentation

To improve robustness, we apply a diverse set of augmentations during training (Table 3), and introduce several X-ray-specific transforms that improve generalization to real images (Appendix E.2.3). Label Zoom randomly selects one present foreground structure, computes its centroid, samples a zoom factor z, crops a window of size $( H / z , W / z )$ centered on that centroid, and resizes the crop back to standard image size. Letter Drop stamps a rasterized ‘L’ or $\mathsf { \Pi } ^ { \mathsf { \prime } } \mathsf { R } ^ { \mathsf { \prime } }$ (chosen with equal probability) at a uniformly random position, with a height sampled uniformly between 3% and 10% of the image height and a brightness sampled from [0.8, 1.0]. Aspect Crop chooses to either perform a horizontal or vertical crop, samples a center-crop of the alternate dimension l so that the image is either (H, l) or (l, W), and zero-pads to maintain the image’s original size (H, W).

a. Distributions of differences in performance of (raw dice - flux dice) across pose groups.  
![](images/8e8a2a2afd207bd0bc08f439e20ff5303ed42102a46aa295d8cccf5eba1ac1d8.jpg)

![](images/db0789e2b8f2d0b93497e446aaf4629d3715bfb4a3f4586bd0586a0a26a1f9ba.jpg)

![](images/3ad5c95a309038d2c1a6e671b21450c4c5d5224a3b19c2dc4bca17bf19cd160f.jpg)

b. Worst case hallucinations pre-filter.  
![](images/3c8e670155b03c9f22068560504a4b41e3fba896e140f20487f28bd6c2923c19.jpg)

c. Worst case hallucinations post-filter.  
![](images/c0627500f75974ded9d0795a2127bee7c0d286e48ef13c1ff4eb01d6f4f9b8f2.jpg)  
Figure 8. Filtering Flux-induced hallucinations. a, Distributions of the per-sample SoftDice difference ∆ for three example pose groups: all samples (gray), the removed high-∆ tail (red), the group median (solid line), and the mirrored upper cutoff $\ell _ { p }$ (dashed line). b, Worst-case hallucinations before filtering: on non-standard views, Flux occasionally replaces the rendered anatomy with a frontal-chest-like image, so the inherited labels (yellow overlay) no longer match the image. c, Worst-case surviving samples after filtering, with structural hallucination substantially reduced.

## A.6 Preprocessing

For CT datasets, we divided each volume into overlapping 512×512×256 crops to reduce disk-to-RAM transfer costs during data-loading for online DRR rendering. For each crop, we padded with HU intensities corresponding to air if the crop z < 256 and limited crop-overlap to 50%. Each crop stores the CT, label map, affine matrix, and foreground centroids used to sample randomized DRR views during training (Table 2). For MOOSE, HaN-Seg, and Shoulder-CT, we ran the MOOSE-Z automated CT segmentation pipeline [77] to segment the volumes. For RSNAFrac, we ran TotalSegmentator [1] to add a skull label. For PedsCT, we took the raw RTSTRUCT contours and converted them into voxel masks. Finally, for ElbowCT, we took the original label STL surface meshes and transformed them into the CT coordinate frame, sampled and voxelized, and filled them into solid masks.

For X-rays, we square-pad and resize all imagemask pairs to 256×256 pixels using area interpolation for images and nearest-neighbor interpolation for masks. We split CT-derived data in fixed 90/10 train/validation splits with no test set. X-ray datasets use fixed 70/15/15 train/validation/test splits (held-out X-ray datasets use train for nnUnet training only), with three exceptions listed in Table 4: RAM-W600 keeps its released partition, DeepFluoro is split by specimen (two of its six specimens per partition), and AASCE, on which no model is trained, is divided into validation and test partitions only. For all images (X-ray or DRR), we clip each to its per-image 0.5th and 99.5th intensity percentiles, and then min– max rescale each image to [0, 1]. We keep all splits subject-disjoint and fixed across experiments.

Table 3. Training augmentation presets. Augmentations are grouped by whether they update both image and mask or image only. The Stream column indicates whether an operation is applied to rendered DRRs, real X-rays, or both. Rotations and shear are in degrees, kernel size, sigma, and crop sizes in pixels, translate is a fraction of image size, and noise std and intensity factors are on the [0, 1] intensity scale. Elastic alpha and sigma are fixed per-axis magnitudes rather than sampled ranges. CLAHE and gamma are mutually exclusive per sample, with the listed marginal probabilities.
<table><tr><td>Operation</td><td>Parameters</td><td>Values</td><td>Stream</td><td>Prob.</td></tr><tr><td>Image and mask</td><td></td><td></td><td></td><td></td></tr><tr><td>Horizontal flip</td><td>一</td><td>一</td><td>both</td><td>0.5</td></tr><tr><td>Label zoom</td><td>zoom range</td><td>[1.1, 4.0]</td><td>X-ray</td><td>0.5</td></tr><tr><td>Affine</td><td>rotation translate scale shear</td><td>[-90,90] [-0.25, 0.25] [0.8, 1.2] [-16,16]</td><td>both</td><td>0.25</td></tr><tr><td>Elastic deformation</td><td>kernel size alpha (x, y) sigma</td><td>127 (12, 20) 64</td><td>both</td><td>0.1</td></tr><tr><td>Aspect crop</td><td>height width</td><td>[64, 224] [64, 224]</td><td>both</td><td>0.1</td></tr><tr><td>Image only</td><td></td><td></td><td></td><td></td></tr><tr><td>Letter drop</td><td>height brightness</td><td>[0.03, 0.10]× H [0.8, 1.0]</td><td>both</td><td>0.25</td></tr><tr><td>Invert</td><td>一</td><td>一</td><td>both</td><td>0.5</td></tr><tr><td>CLAHE</td><td>clip limit grid</td><td>[1.0, 2.0] 8×8</td><td>both</td><td>0.1</td></tr><tr><td>Gamma</td><td>gamma</td><td>[0.9, 1.1]</td><td>both</td><td>0.25</td></tr><tr><td>Contrast</td><td>gain contrast</td><td>[0.9, 1.1] [0.7, 1.3]</td><td>both</td><td>0.25</td></tr><tr><td>Plasma</td><td>roughness</td><td>[0.3, 0.6]</td><td>both</td><td>0.1</td></tr><tr><td>brightness Sharpness</td><td>intensity</td><td>[0.05, 0.2]</td><td></td><td></td></tr><tr><td></td><td>sharpness</td><td>[0.7, 1.3]</td><td>both</td><td>0.5</td></tr><tr><td>Gaussian noise</td><td>std</td><td>0.01</td><td>both</td><td>0.25</td></tr></table>

## B Training

## B.1 Network architecture

FleXray is an ensemble of five networks (members) that share one architecture and differ only in their training-data mixture. Each member is a seven-level 2D convolutional U-Net [78], with its encoder and decoder using the same feature widths at matched resolutions, 64, 128, 256, 512, 512, 720, and 1024. Each encoder block is composed of three 3×3 convolutions, each followed by a LeakyReLU activation and instance normalization. A residual connection is used around the convolutional layer, using an identity mapping when the input and output widths match, and a 1×1 convolution with normalization to match feature widths otherwise. A final 1×1 convolution maps the decoder features to per-channel logits with per-channel sigmoid activations used to map the logits to independent, overlapping probability maps. We also considered using a pretrained segmentation foundation model for fine-tuning with our data recipe, but did not observe a clear accuracy advantage (Appendix E.2.2).

MOOSE-derived data fills a fixed 75% of the sampling budget in every member; the supplemental CT and real X-ray sources fill the remaining 25% at the weights of Table 1. The members differ only in how this 75% is divided between DRRs rendered online during training and the offline enhanced DRRs: the offline enhanced DRRs take 0, 1/3, 1/2, 2/3, or all of the MOOSE budget, with online rendering taking the rest. We found that combining online and offline rendering yields higher mean Dice than using either alone (Appendix E.2.1).

## B.2 Optimization

FleXray trains for 2,000 epochs with 500 mixed-dataset loader iterations per epoch and a batch size of 16 images (either DRRs or X-rays). AdamW [79] is used for optimization with an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , weight decay 0.01 on all parameters, EMA [80] with $\alpha = 0 . 9 9 9 9$ and a cosine learning rate scheduler stepped per epoch over the 2,000 epochs with $\eta _ { m i n } = 0 . 0$ , without warm-up or gradient clipping. Inference uses the EMA weights at the final epoch of training. Importantly, we don’t perform any checkpoint selection based on the results of the heldout X-ray datasets.

Crop sampling weights. To produce a DRR batch, we first sample from a CT dataset’s crops with replacement under tempered inverse label-frequency weights [81, 82]. Each foreground structure c contributes $( N / n _ { c } ) ^ { \tau }$ , where N is the source’s crop count, $n _ { c }$ is the number of crops containing $c ,$ and $\tau \ : = \ : 0 . 5$ tempers the distribution. A crop’s weight is the maximum of its structures’ contributions. CT crops without foreground labels, under the FleXray training label-protocol, are assigned 0 weight.

Training objective. Our training objective is an equally weighted sum of the Soft-Dice loss [83] and binary crossentropy (BCE). For FleXray f and input image $x , f ( x ) =$ $z _ { b c i }$ , target $y _ { b c i }$ , and probability $p _ { b c i } ~ = ~ \sigma ( z _ { b c i } )$ , the Soft-Dice term uses $\epsilon = \dot { 1 } 0 ^ { - 7 }$ and the squared-denominator form

$$
\mathcal { L } _ { \mathrm { D i c e } } = 1 - \frac { 2 \sum _ { i } p _ { b c i } y _ { b c i } + \epsilon } { \sum _ { i } p _ { b c i } ^ { 2 } + \sum _ { i } y _ { b c i } ^ { 2 } + \epsilon }\tag{3}
$$

Soft-Dice is restricted to foreground channels present in the image, whereas BCE supervises every channel unless a source is partially labeled, as described below.

Partial labeling loss. The six data sources (RSNAFrac, PedsCT, FootBones, ElbowCT, MURA forearm, and MURA humerus) annotate only a subset of the structures visible in their images. For these, BCE is computed only on output channels that contain annotated pixels. Channels with no annotation are excluded rather than supervised as background, so structures that are present but unlabeled receive no negative signal.

Table 4. Held-out evaluation data sources. Counts are X-rays retained after the quality-control exclusions described in the text. Images from the same patient remain in the same split. Annotated labels are pixel-level segmentation masks, except for AASCE, whose annotations are Cobb angles derived from 68 vertebral corner landmarks (T1–L5). Mask type describes the masks used fo scoring after protocol post-processing (Methods C.1): structures may occupy the same pixel (overlapping) or partition the image (exclusive). Visible labels are the foreground structures exempted dataset-wide from aFPR (Equation (4)).
<table><tr><td>Dataset</td><td>Description</td><td>Train</td><td>Val</td><td>Test</td><td>Mask type</td><td>Annotated labels</td><td>Visible labels</td></tr><tr><td colspan="8">Primary segmentation evaluation DarwinCVD19 [84] Chest X-ray</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>kidneys, liver, lungs, ribs 1–12, scapulae, skull, spleen, sternum, vertebrae C1-L5</td></tr><tr><td>DeepFluoro [45]</td><td>Fluoroscopic X-ray</td><td>70</td><td>79</td><td>213</td><td>Overlapping</td><td>Femurs, hips, lumbar spine, sacrum</td><td>Femurs, hips, lumbar spine, sacrum, vertebrae L1-L5</td></tr><tr><td>ElbowLat [85] HipRay [86]</td><td>Lateral elbow X-ray</td><td>419</td><td>91</td><td>91</td><td>Overlapping</td><td>Humeri, radii, ulnae Femurs, hips</td><td>Humeri, radii, ulnae Femurs, hips, sacrum</td></tr><tr><td>LowerLimbs [87]</td><td>Pelvic X-ray Pelvis and lower extremity</td><td>97 39</td><td>21 9</td><td>21 8</td><td>Exclusive Overlapping</td><td>Femurs, fibulae, tibiae</td><td>Femurs, fibulae, hips,</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>metatarsals, patellae, sacrum, tarsals, tibiae, toes, vertebrae L1-L5</td></tr><tr><td>RAM-W600 [88]</td><td>Hand/wrist X-ray</td><td>425</td><td>69</td><td>124</td><td>Overlapping</td><td>Carpals, metacarpals, radii, ulnae</td><td>Carpals, metacarpals, phaianges, radii, ulnae Clavicles, femurs, heart, hips,</td></tr><tr><td></td><td>Pediatric X-ray</td><td></td><td></td><td></td><td></td><td></td><td>humeri, kidneys, liver, lungs, radii, ribs 1–12, sacrum, scapulae, skull, spleen, sternum, thoracolumbar spine, ulnae, vertebrae C1–L5 Clavicles, heart, humeri,</td></tr><tr><td></td><td>Chest X-ray</td><td></td><td></td><td></td><td></td><td></td><td>kidneys, liver, lungs, ribs 1–12, scapulae, skull, spleen, sternum, vertebrae C1-L5</td></tr><tr><td colspan="8">Downstream tasks AASCE [51]</td></tr><tr><td></td><td>Spinal AP X-ray</td><td>0</td><td>262</td><td>218</td><td></td><td>Cobb angles Bone tumors</td><td>一</td></tr><tr><td>BTXRD [71] MTDDH [70]</td><td>Musculoskeletal X-ray</td><td>1,305</td><td>281</td><td>281</td><td>Exclusive</td><td></td><td>一</td></tr><tr><td></td><td>Pediatric pelvic X-ray</td><td>632</td><td>138</td><td>135</td><td>Overlapping</td><td>Femoral head, ilium, ischium, pubis</td><td></td></tr></table>

Targeted negative supervision. Excluding empty channels also discards negative supervision on structures that are known to be absent. The byproduct is that the model hallucinates extra classes in ambiguous views. For five of the six sources we reinstate BCE on channels that are anatomically impossible in the image yet easily confused with the labeled structures: FootBones supervises the hand and forearm channels (phalanges, metacarpals, carpals, ulnae, and radii) as empty; ElbowCT, MURA forearm, and MURA humerus supervise the lower-limb channels (femurs, patellae, tibiae, and fibulae) as empty; and RSNAFrac supervises the humeri channel as empty.

## C Evaluation

X-ray segmentation evaluation is challenging due to the scarcity of diverse segmentation datasets available online. We made a best-effort attempt to gather a diverse selection of eight X-ray segmentation datasets that cover a broad range of target anatomy and acquisition settings (Table 4). These collectively cover the trunk and both limbs, spanning lungs (DarwinCVD19), ribs (VinDr-Rib), spine (PedsTorso, DeepFluoro), pelvis and hip (HipRay, DeepFluoro), femur, tibia, and fibula (LowerLimbs), upper limbs (ElbowLat), and the carpals, metacarpals, and forearm bones of the hand and wrist (RAM-W600). They span standard and oblique radiography and intraoperative fluoroscopy (DeepFluoro), frontal and lateral acquisitions (ElbowLat), and both adult and pediatric anatomy (PedsTorso).

In five of the eight sets, the evaluation masks retain spatial overlap, so one pixel can carry several labels (Mask type in Table 4). FleXray and the generalist baselines each emit a separate mask per structure, and the dataset-specific nnU-Nets are trained in region-based mode on these five datasets (Methods C.4), so no method is restricted to a mutually exclusive partition when scoring them.

All of the main reported results, and the ablations, are scored on the test partitions, which were held out during model development. The validation partitions were used only for development decisions, such as choosing the FleXray (single) network for the ablations and finetuning (Table 8). The pretrained FleXray members and every ablation model are evaluated at their final training epoch without checkpoint selection (Methods B.2). All evaluation annotations were sourced from publicly released datasets and annotation projects.

Quality control and splits. The per-image split assignment of every source, together with each excluded image and its reason, is released with the dataset (https: //huggingface.co/datasets/VictorButoi/flexray-data). For sources we cannot redistribute, the lists are keyed by the original filenames. Before assigning splits, we removed exact-duplicate images, which would otherwise place the same radiograph in several partitions, and images whose annotation contained no foreground after mapping into our label protocol. These affected DarwinCVD19 (33 duplicates, 108 empty masks, and 16 annotations whose image is absent from the public download) and ElbowLat (11 empty masks). We additionally excluded a small number of individual images whose annotations we judged unreliable on manual review: one in HipRay, five in Lower-Limbs, four DeepFluoro frames with questionable groundtruth poses reported upstream, one AASCE radiograph whose landmarks fall outside the image, and three MT-DDH images with a corrupt file or an out-of-bounds polygon. No other evaluation dataset had exclusions. All criteria were fixed before any model was scored and apply identically to FleXray and every baseline.

## C.1 Protocol post-processing

In order to compare our predictions, and those of our baselines, to the held-out X-ray ground-truth labels, we apply minimal deterministic post-processing to all predictions to align them with our evaluation sets. Each adjustment is applied identically to every method. All rules were derived from each dataset’s annotation documentation and inspection of its training-split annotations, fixed before scoring the evaluation partitions, and were not tuned on their performance.

For quantitative evaluation on lung datasets (DarwinCVD19 and PedsTorso), we set lung probabilities to zero where there is predicted liver or spleen to restrict the predictions to visible contours of the lungs. For DeepFluoro lumbar-spine and PedsTorso thoracolumbar-spine, we take the pixelwise maximum in probability space over the corresponding vertebra channels (L1–L5, and T1– T12 together with L1–L5, respectively). For PedsTorso and HipRay, we collapse both the prediction and the reference into a single mutually exclusive overlay under a fixed per-dataset priority order (thoracolumbar spine over lungs for PedsTorso, femurs over hips for HipRay).

## C.2 Metrics

Segmentation quality. We evaluate segmentation overlap quality with both the Dice similarity coefficient [91], which measures the spatial overlap between a ground-truth mask and a predicted mask, and 95thpercentile Hausdorff distance (HD95) [92], which measures boundary error in pixels. For every boundary pixel of one mask we take the Euclidean distance to the nearest boundary pixel of the other, and HD95 is the larger of the two directed 95th percentiles; boundaries are each mask minus its one-pixel erosion. A structure present in the ground truth but absent from the thresholded prediction is excluded from that image’s HD95 average. Dice and HD95 are computed per structure and averaged per image. A structure whose ground truth occupies less than 0.1% of the image area is dropped from that image, so that vanishingly small structures do not contribute to its score.

Anatomical false-positive rate (aFPR). To quantify hallucinations in FleXray predictions, we introduce the anatomical false-positive rate (aFPR): the average number of predicted foreground classes per image that are anatomically incompatible with the X-ray dataset. For an image x from dataset $d ,$ let ${ \mathcal { P } } ( x )$ be the classes whose predicted probability exceeds 0.5 anywhere in the image, $\mathcal { G } ( x )$ the classes in its ground truth, and $\mathcal { T } _ { d }$ the classes visible in the radiographs of dataset $d ,$ annotated or not (visible labels in Table 4). Then

$$
\mathrm { a F P R } ( x ) = \left| \mathcal { P } ( x ) \setminus ( \mathcal { G } ( x ) \cup \mathcal { T } _ { d } ) \right| .\tag{4}
$$

Mean target registration error (mTRE). We evaluate 2D/3D registration accuracy with the mean target registration error. Given a set of 3D fiducial landmarks $\{ \mathbf { t } _ { k } \} _ { k = 1 } ^ { K }$ defined in the CT coordinate frame (K = 14 per Deep-Fluoro specimen), a recovered camera pose $\hat { p } ,$ and the ground-truth pose $p ^ { * }$

$$
\mathrm { m T R E } ( \hat { p } , p ^ { * } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \bigl \| T _ { \hat { p } } ( \mathbf { t } _ { k } ) - T _ { p ^ { * } } ( \mathbf { t } _ { k } ) \bigr \| _ { 2 } ,\tag{5}
$$

where $T _ { p }$ maps a point into the camera coordinate frame of pose $p .$ mTRE is reported in millimeters; lower is better.

Detection average precision (AP50). For detection targets we score whether each annotated lesion is localized rather than how well it is delineated. Candidate detections are the 4-connected components of at least four pixels in which the predicted lesion probability exceeds 0.5, each represented by its enclosing box and scored by its mean lesion probability. Candidates are matched one-to-one to ground-truth boxes in decreasing order of confidence, a match requiring an intersection-over-union of at least 0.5, with at most 100 candidates per image, using the reference COCO implementation [93]. AP50 is the area under the 101-point interpolated precision–recall curve. Higher is better.

Table 5. Test-time augmentation preset. Each augmented view draws the operations below independently, in the order listed, with the per-draw probability in Prob. CLAHE and gamma are mutually exclusive per draw.
<table><tr><td>Operation</td><td>Parameters</td><td>Values</td><td>Prob.</td></tr><tr><td>Horizontal flip</td><td>一</td><td>一</td><td>0.5</td></tr><tr><td>Invert</td><td>一</td><td>一</td><td>0.5</td></tr><tr><td>CLAHE</td><td>clip limit grid</td><td>[1.0, 2.0] 8×8</td><td>0.1</td></tr><tr><td>Gamma</td><td>gamma gain</td><td>[0.9, 1.1] [0.9, 1.1]</td><td>0.25</td></tr><tr><td>Contrast</td><td>contrast</td><td>[0.7, 1.3]</td><td>0.25</td></tr><tr><td>Sharpness</td><td>sharpness</td><td>[0.7, 1.3]</td><td>0.5</td></tr><tr><td>Gaussian noise</td><td>std</td><td>0.01</td><td>0.25</td></tr></table>

## C.3 Test-time augmentation

To improve robustness and reduce false-positive predictions (Figure 9), we wrap FleXray in a test-time augmentation (TTA) module that aggregates predictions over several stochastically perturbed views of each image [94, 95]. The first pass uses the unaugmented image and the remaining passes are independent draws from the TTA preset of Table 5. We ensure that every ensemble member receives the same set of K augmented views:

$$
\hat { y } _ { e n s } = \frac { 1 } { 5 } \sum _ { i = 1 } ^ { 5 } \left[ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } T _ { k } ^ { - 1 } ( f _ { i } ( T _ { k } ( x ) ) ) \right] ,\tag{6}
$$

where $T _ { k }$ is the $k ^ { t h }$ augmentation view and $T _ { k } ^ { - 1 }$ is its necessary inverse operation (identity except for views with left/right flips).

The number of TTA samples, K = 16, was fixed on the validation partitions of Table 4 before the final models were trained. Figure 9 reports this frozen setting on the test partitions, sweeping K only to show how Dice, HD95, and aFPR vary around it. Ensembling and TTA change Dice and HD95 very little. From the FleXray (single) model without TTA to FleXray, equal-dataset mean Dice rises from 0.866 to 0.875 and HD95 falls from 11.8 to 10.8 pixels. Both substantially reduce hallucinated structures. For the FleXray (single) model, 16 TTA samples lower aFPR (Equation (4)) from 3.64 to 2.35 classes per image. Ensembling the five members without TTA lowers it to 1.43. Combining the two yields 1.18, roughly one third of the single-pass rate.

![](images/257d3a656d81e635059a51c985c2307619ba7ac2767e9ab4c3e137989bbda6ff.jpg)

![](images/76febeb30b396de902a60b04484bb763a486bf36ac8f996ae79155de57af5c9f.jpg)

![](images/5bbea742b015966e024710e28ce85fa1fa8230b71aa7c62222c4ab0d00a06164.jpg)  
FleXray single model mean (95% CI) FleXray 5-member ensemble mean (95% CI)  
Figure 9. TTA sample sweep. Equal-dataset mean a, Dice, b, HD95, and c, aFPR versus the number of TTA samples for FleXray (single) and FleXray, computed over the eight held-out real-radiograph segmentation datasets in Table 4. Each line is the per-method mean over four inference seeds (40–43) with 95% confidence intervals over seeds. Higher Dice and lower HD95 and aFPR are better.

## C.4 Baselines

We compare against general-purpose X-ray segmentation methods that, like FleXray, did not train on the heldout real X-ray evaluation sets. As a reference, we also trained a set of dataset-specific nnU-Net models on each of the X-ray evaluation segmentation datasets. We evaluated only labels represented by a method’s released output vocabulary. For our baselines, we applied the TTA scheme that they considered as part of their method (no TTA if they did not use it). Table 7 summarizes their computational footprints, and Table 6 reports the full perdataset comparison (both in Appendix E.1).

PAXray [29]. We used the authors’ ResNet50-UNet pretrained checkpoint which predicts 159 output channels. Each X-ray was replicated to three channels, normalized with ImageNet statistics, and resized to 512×512 to match their training.

TotalSegmentator2D [30]. We used the released ensemble of models for cardiac, muscle, organ, rib, and vertebral groups, yielding 117 native foreground channels. Images were processed with the nnU-Net plans and preprocessing stored with the released models. For Deep-Fluoro only, we inverted each image’s intensities within its per-image dynamic range before applying the TotalSegmentator2D preprocessing. Inference uses nnU-Net’s default flip-mirroring test-time augmentation.

FluoroSAM [31]. FluoroSAM differs from other baselines and FleXray as it is an interactive-segmentation method. We used the released Swin-L checkpoint at its 448×448 training resolution. Prompt names were derived from the foreground labels in our shared protocol. For each scored non-bilateral label, we supplied its text prompt and one positive point at the center of each connected ground-truth component. For bilateral structures, we issued separate left- and right-specific text prompts with one positive ground-truth-derived point per available side and combined the two score maps by pixelwise maximum.

Efficiency measurement. We time per-image model execution at batch size 1 in FP32 on an NVIDIA V100 at each method’s native input resolution: 256×256 for FleXray and for each of the five TotalSegmentator2D anatomy-group models, 512×512 for PAXray, and $4 4 8 \times 4 4 8$ for FluoroSAM. Input preparation and datasetspecific postprocessing are outside the timer. Ensemble timings include prediction averaging, and TTA timings additionally include augmentation, normalization, inverse alignment, and averaging across views. We select one test image from each of the eight primary segmentation evaluation datasets in Table 4. For each image, latency is the mean of 10 CUDA-synchronized runs after 5 warmup runs, and memory is the peak allocation during a postwarmup run. We report mean latency across the eight images with an image-level bootstrap confidence interval. FluoroSAM’s timing covers one image-encoder pass followed by 62 text-prompt encoder and mask-decoder passes, covering the 60 protocol structures plus lumbar spine and thoracolumbar spine, without point prompts.

Dataset-specific networks. We trained independent nnU-Net [69] networks for each evaluation dataset in Table 4. We use nnU-Net’s default configuration and 1,000-epoch schedule, without dataset-specific hyperparameter tuning or fold ensembling. Where annotated structures overlap (Mask type in Table 4), we train nnU-Net in its region-based mode. Mutually exclusive datasets use the standard softmax configuration. Inference uses nnU-Net’s default flip-mirroring test-time augmentation.

## D Downstream tasks

## D.1 Cobb-angle estimation

Splits and development. AASCE is partitioned by patient into a validation split (262 images; 39, 107, and 116 low, moderate, and severe curves) and a test split (218 images; 33, 100, and 85).

Procedure. We apply the same procedure with identical settings to FleXray and every baseline. We threshold the prediction channels for T1–T12 and L1–L5, the 17 vertebrae annotated in AASCE, at 0.5, assigning overlapping pixels to the more superior vertebra. For each vertebra, we retain the largest connected component and discard it if its area is below 0.1% of the image. We order the surviving vertebrae superior to inferior by mask centroid. Images with fewer than three surviving vertebrae are excluded from that method’s error statistics and counted toward its failure rate.

Cobb-angle prediction. Following Pan et al. [96], we approximate vertebral end-plate orientations from the spinal centerline connecting consecutive mask centroids. For each interior vertebra, the end-plate direction is the axial bisector of the normals to its two adjacent centerline segments, treating the normals as undirected lines. The first and last end-plate directions are horizontal. We represent these directions by unit vectors pointing toward image right and report their largest pairwise angular separation as the predicted major Cobb angle, following the challenge’s reference procedure [51]. The reference major angle is the maximum of the three provided AASCE angles.

## D.2 2D/3D registration

We use the semantic information estimated by FleXray to inject additional supervision into 2D/3D registration protocols via a bidirectional Chamfer distance [67, 68]. We build upon xvr [46], which registers a preoperative CT to an intraoperative X-ray x by minimizing an image similarity loss ${ \mathcal { L } } _ { \mathrm { i m g } }$ between a DRR $I _ { p }$ rendered from camera pose $p$ and x with respect to $p .$ We focus on DeepFluoro as a representative study.

Per-structure Chamfer loss. Let S be the set of structures in the pelvis (i.e., hips, lumbar vertebrae, and sacrum). For each structure $s \in \mathcal { S }$ , let $M _ { s } \subseteq { \mathcal { D } }$ be its FleXray-predicted mask, where $\mathcal { D }$ denotes the set of detector pixels and $u \in \mathcal { D } \mathtt { a }$ pixel.

Forward term. We pre-compute the 2D Euclidean distance transform $D _ { s } ( u )$ of $M _ { s }$ , representing the distance from u to the nearest pixel of $M _ { s }$ in mm. Next, as the preoperative CT is labeled, the renderer partitions the DRR into per-structure channels $I _ { p } ^ { s } ( u )$ . For structures with nonzero projected mass, the forward term integrates the unit-normalized projected mass of s over all detector pixels against $D _ { s }$ :

$$
\ell _ { s } ^ { \mathrm { f w d } } ( p ) = \frac { \sum _ { u \in \mathcal { D } } I _ { p } ^ { s } ( u ) D _ { s } ( u ) } { \sum _ { u \in \mathcal { D } } I _ { p } ^ { s } ( u ) } .\tag{7}
$$

Reverse term. We pre-compute the 3D Euclidean distance field $\Phi _ { s } ( \cdot )$ from every voxel to s in mm. For every pixel in the predicted mask $u \in M _ { s }$ , we sample points $\dot { \mathbf { v } } _ { u } \in \mathbb { R } ^ { K \times 3 }$ along the camera ray and reduce:

$$
\ell _ { s } ^ { \mathrm { r e v } } ( p ) = \frac { 1 } { \left| M _ { s } \right| } \sum _ { u \in M _ { s } } \bigl < \alpha _ { u } , \Phi _ { s } ( { \mathbf v } _ { u } ) \bigr > ,\tag{8}
$$

where $\alpha _ { u } = \mathrm { s o f t m a x } \big ( - \Phi _ { s } ( \mathbf { v } _ { u } ) / \tau \big )$ with temperature $\tau =$ 2 mm and $K = 3 0 0$ . The reverse term penalizes poses for which the FleXray mask’s rays are far from the 3D structure.

Why distance transforms. Both terms owe their robustness to a property of Euclidean distance transforms. The distance fields $D _ { s }$ and $\Phi _ { s }$ have unit-magnitude spatial gradients almost everywhere outside their respective structures, regardless of distance, satisfying the Eikonal equation, $\lVert \nabla \Phi _ { s } \rVert = 1 \ [ 9 7 ]$ . The reverse term uses these gradients through its softmax-weighted aggregation, and the forward term weights displaced projected mass linearly in its distance to the predicted mask. An image similarity objective, by contrast, carries no signal once the DRR and X-ray no longer overlap, so the Chamfer terms provide informative gradients even for very poor initial pose estimates.

Optimization. Each term is averaged over the structures $S ^ { + } \subseteq S$ that have a non-empty predicted segmentation mask, with weights $\omega _ { s }$ proportional to the area of $M _ { s }$ at the coarsest resolution and normalized to sum to one, so that large, well-observed structures dominate:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { f w d } } ( p ) = \displaystyle \sum _ { s \in \mathcal { S } ^ { + } } \omega _ { s } \ell _ { s } ^ { \mathrm { f w d } } ( p ) , } \\ & { \mathcal { L } _ { \mathrm { r e v } } ( p ) = \displaystyle \sum _ { s \in \mathcal { S } ^ { + } } \omega _ { s } \ell _ { s } ^ { \mathrm { r e v } } ( p ) . } \end{array}\tag{9}
$$

The final registration objective is

$$
\begin{array} { r } { \mathcal { L } ( x , p ) = \mathcal { L } _ { \mathrm { i m g } } ( x , I _ { p } ) + w \left( \mathcal { L } _ { \mathrm { f w d } } ( p ) + \mathcal { L } _ { \mathrm { r e v } } ( p ) \right) . } \end{array}\tag{10}
$$

Gradient-based optimization follows the multistage protocol of xvr [46], with the Chamfer weights w following a per-stage schedule tuned for each initialization. From the coarse initialization, where image similarity is uninformative, the weight is $w = 1$ at the two coarse stages and $w =$ 0.05 at the finest, so the coarse stages pull the pose into the mask-consistent basin while the final fit is determined by image similarity. From the foundation-model initialization, we use a weak schedule of $w = 0 . 0 5 , 0 . 0 5$ , and 0.01 to provide moderate supervision while safeguarding against divergence.

FleXray predicted masks. We apply the default FleXray configuration to each frame after the same cropping and log-linearization used by the registration, and threshold predictions at 0.5. Segmentation masks are predicted by FleXray once per frame and are shared across every initialization and method.

Evaluation protocol. DeepFluoro [45] provides groundtruth camera poses and 3D fiducial landmarks for six cadaver specimens. We evaluate on the 213 frames of the two held-out test specimens (Table 4). Each frame is registered from two initializations: a coarse, clinically practical pose constructed from anatomical landmarks (Method 1 in Grupp et al. [45]), and the pose predicted by the whole-body foundation pose-regression model of xvr [46], which is not trained on DeepFluoro. Because a single radiograph does not disambiguate the anterior– posterior orientation, the foundation initialization is run in both orientations and the result with the lower final objective (Equation (10)) is kept for each method, so the two methods can start from different orientations of the same frame and their initial errors differ slightly (76.6 versus 76.1 mm median).

From each initialization, we run xvr with and without the Chamfer terms and score the recovered pose by the mean target registration error over the specimen’s $K = 1 4$ anatomical landmarks (bilateral pelvic landmarks defined in the CT frame by Grupp et al. [45]; mTRE, Methods C.2), calling a registration successful below 10 mm. Image-similarity registration takes 3–5 seconds per frame. The Chamfer terms raise this to about 9 seconds from the foundation initialization and 28 seconds from the coarse initialization, whose strong weights keep the optimizer from stopping early.

## D.3 Data-efficient finetuning

All fine-tuned networks are initialized from the final checkpoint of FleXray (single). We replace the final $1 \times 1$ output convolution with a randomly initialized head sized to the target protocol, and fine-tune end-to-end.

Optimization. Each run trains for 250 epochs of 50 iterations at a batch size of 16 and 256×256 resolution. We use AdamW with a peak learning rate of $3 \times 1 0 ^ { - 4 }$ , weight decay $3 \times 1 0 ^ { - 5 }$ , a 5-epoch linear warm-up from 0.1× the peak, and cosine decay thereafter, in 32-bit precision and without EMA. We use the same objective as FleXray pretraining (Methods B.2). For every run we keep the checkpoint with the highest mean foreground Dice on the validation split and report on the test split. To ensure a fair comparison, during training we apply the default 2D augmentation pipeline of nnU-Net $\boldsymbol { \mathsf { v } } \boldsymbol { 2 }$ [69] (rotation, scaling, Gaussian noise and blur, brightness, contrast, simulated low resolution, and gamma), restricting rotation to $\pm 1 5 ^ { \circ }$ and using the same left–right flip as TTA.

Training subsets. For few-shot configurations, we draw subsets from the training split of each dataset (Table 4). Within a repeat, every smaller subset is a prefix of the larger ones. Budgets below 100% use three repeats drawn with different seeds. FleXray and the from-scratch nnU-Net comparators (Methods C.4) read the identical subset manifests, validate on the same validation split, and are scored on the same test split.

MTDDH. The head has five channels: background, ilium, pubis, ischium, and femoral head. The femoral head overlaps the acetabular bones, so MTDDH is an overlapping dataset (Table 4) and its nnU-Net comparator is trained in region-based mode on the same four foreground labels. The dataset’s native femur annotation is mapped to background for both FleXray and nnU-Net because FleXray already segments the femur. Subsets are selected at the patient level by a seeded random permutation, at 1%, 5%, 10%, 50%, and 100% of the training split (6, 32, 63, 316, and 632 patients).

BTXRD. The head has two channels, background and tumor, and all images carry a tumor. Subsets cover 1%, 5%, 10%, 50%, and 100% of the training split (13, 65, 130–131, 653, and 1,305 images). Because BTXRD spans three centers and nine diagnosis categories with skewed frequencies, subsets are stratified rather than drawn uniformly. Cases are added in an order that preserves the full split’s center and diagnosis proportions at every budget.

BTXRD is scored by detection AP50 (Methods C.2), the endpoint reported by the dataset’s authors [71]. Ground truth is the expert’s bounding rectangle for each of the 354 tumors in the test split, transformed with the same padding and resizing as the image. Both FleXray and nnU-Net are scored on softmax probabilities averaged over the image and its horizontal reflection. AP50 is computed per network and averaged over training repeats. Tumor-size quartiles are fixed once on the training split by box area as a fraction of the padded image; following the COCO convention, unmatched candidates outside a quartile’s size range are ignored rather than counted as false positives. Because every BTXRD radiograph contains a tumor, precision is measured on tumor-positive images only and does not estimate specificity on normal radiographs.

## E Additional results

## E.1 Full quantitative quality comparison against baselines

Table 6 gives the per-dataset results behind Figure 3. On the wrist, elbow, and lower-limb datasets, FluoroSAM is the only released generalist with sufficient label coverage, and FleXray exceeds it by 0.42–0.71 Dice even though FluoroSAM receives a ground-truth-derived text prompt and click for every structure (Methods C.4). On the hip and on fluoroscopy, TotalSegmentator2D covers the labels but reaches only 0.478 and 0.371, the latter after per-image intensity inversion (0.026 if using the native polarity for DeepFluoro). The margins are smallest on adult chest radiographs, the domain both chest generalists target. On VinDr-Rib, PAXray is numerically higher but not significantly different from FleXray (0.751 vs. 0.737; n=37 images; $\textstyle p { = } 0 . 1 4 6 )$ . On PedsTorso, FleXray significantly exceeds TotalSegmentator2D (0.831 vs. $0 . 7 1 9 ; n = 1 2 ; p = 0 . 0 3 1 )$ On DarwinCVD19, FleXray leads TotalSegmentator2D by 0.018 Dice (0.924 vs. $0 . 9 0 6 ; n { = } 9 5 2 ; p { < } 0 . 0 0 1 )$

The in-domain nnU-Nets exceed FleXray on seven datasets by 0.03–0.09 Dice. The exception is DeepFluoro, where the nnU-Net trained on the 70 frames of two subjects reaches 0.866 on the two held-out subjects, against 0.915 for FleXray $( n { = } 2 1 3 ; p { < } 0 . 0 0 1 )$

The four FleXray configurations separate the contributions of ensembling and test-time augmentation. Averaging the five members raises the equal-dataset mean Dice from 0.866 to 0.874; 16-sample TTA adds a further 0.004 to FleXray (single) and 0.001 to the ensemble. These gains come at the cost of additional computation (Table 7). FleXray (single) is of the same order as PAXray in size and latency (101.4M vs. 73.4M parameters; 14.7 vs. 15.5 ms per image), TotalSegmentator2D accumulates 167.5M parameters and 23.8 ms across its five anatomy-group models, and FluoroSAM (268.7M parameters) needs 623.7 ms for 62 text prompts, a cost that grows with the number of structures queried. The released ensemble uses five times as many parameters (507.0M) and takes 74.2 ms per image; TTA increases its latency to 1.24 s, while peaking at 2.36 GB of GPU memory. FleXray (single) without TTA therefore retains 99% of the released configuration’s accuracy at approximately 1/84 of its latency.

## E.2 Ablations

We trained several different versions of the FleXray (single) network to evaluate different parts of our training pipeline. The ablations reuse the FleXray training recipe (Methods B) except for the components under study and the SAM2-specific input resolutions and optimization settings described in Appendix E.2.2. All ablation models are evaluated on the test partitions of the held-out real-X-ray suite (Table 4) with the same 16-sample test-time augmentation.

## E.2.1 Mixing online and offline rendering performs best

FleXray draws its MOOSE-derived supervision from two streams: DRRs rendered online during training (Methods A.1) and the offline enhanced DRRs dataset (Methods A.2). We trained five networks that hold the combined budget at 75% (all other streams at the weights of Table 1) while sweeping the online/offline split: 75/0, 50/25, 37.5/37.5, 25/50, and 0/75.

The three mixed splits performed about the same (equal-dataset mean Dice 0.870–0.874; Table 8), and both extremes were worse. Online rendering alone (75/0) reached 0.858 and dropped to 0.705 on LowerLimbs: online DRRs are rendered from $5 1 2 \times 5 1 2 \times 2 5 6$ CT crops (Methods A.6) and therefore cannot simulate the far, whole-limb views in that dataset. Offline rendering alone (0/75) reached an equal-dataset mean Dice of 0.852 and was lowest on four of the eight datasets, consistent with overfitting to the fixed offline renders. Because of this, we chose the balanced 37.5/37.5 split as the representative FleXray (single) from its validation-partition score.

Table 6. Per-dataset Dice comparison against baselines. Per-example mean label Dice, shown as mean [95% CI] (Appendix K.1), on the eight held-out real X-ray test sets (n images each). FleXray (ours) is the released five-member ensemble with 16-sample test-time augmentation (Methods B.1 and Table 5); the other FleXray rows remove TTA, the ensemble, or both. Dashes mark datasets outside a method’s label coverage. Bold marks FleXray (ours) and baselines not significantly different from it. Paired comparisons include only FleXray (ours) versus each supported generalist baseline, with Holm correction within each dataset. nnU-Net is greyed as an in-domain reference trained on each dataset’s own annotations. Mean weights the eight datasets equally and is shown only for methods covering all of them.
<table><tr><td>Method</td><td>RAM-W600 LowerLimbs ElbowLat (n = 124)</td><td>(n = 8)</td><td> $( n = 9 1 )$ </td><td>HipRay  $( n = 2 1 )$ </td><td> $( n = 2 1 3 )$ </td><td>DeepFluoro PedsTorso (n = 12)</td><td>VinDr-Rib (n = 37)</td><td>DarwinCVD19 Mean (n = 952)</td><td>Dice</td></tr><tr><td>PAXray</td><td></td><td></td><td></td><td></td><td></td><td>0.570 [0.499,0.645]</td><td>0.751 [0.733,0.769]</td><td>0.888 [0.884, 0.892]</td><td></td></tr><tr><td>TotalSeg2D</td><td>一</td><td>一</td><td>一</td><td>0.478 [0.423,0.531]</td><td>0.371 [0.357,0.386]</td><td>0.719 [0.603, 0.808]</td><td>0.695 [0.672,0.715]</td><td>0.906 [0.903,0.908]</td><td>一</td></tr><tr><td>FluoroSAM</td><td>0.239 [0.230, 0.249]</td><td>0.416 [0.300, 0.520]</td><td>0.182 [0.161,0.202] [0.277,0.353]</td><td>0.313</td><td>0.711 [0.701,0.722]</td><td>0.308 [0.220,0.385]</td><td>0.091 [0.080,0.103]</td><td>0.527 [0.519,0.534]</td><td>0.348</td></tr><tr><td>FleXray (single)</td><td>0.945 [0.944, 0.947]</td><td>0.831 [0.717,0.920]</td><td>0.854 [0.836, 0.871] [0.934,0.941]</td><td>0.937</td><td>0.904 [0.898, 0.908]</td><td>0.811 [0.780,0.839]</td><td>0.724 [0.705, 0.742]</td><td>0.918 [0.915,0.921]</td><td>0.866</td></tr><tr><td>FleXray (single) + 16 TTA</td><td>0.946 [0.944, 0.947]</td><td>0.835 [0.723,0.920]</td><td>0.864 [0.847, 0.879] [0.932,0.940]</td><td>0.936</td><td>0.910 [0.905, 0.914]</td><td>0.817 [0.789, 0.843]</td><td>0.726 [0.704, 0.745]</td><td>0.922 [0.920,0.925]</td><td>0.870</td></tr><tr><td>FleXray (no TTA)</td><td>0.948 [0.947,0.950]</td><td>0.836 [0.714,0.929]</td><td>0.871 [0.855, 0.883] [0.934,0.942]</td><td>0.938</td><td>0.913 [0.909,0.917]</td><td>0.830 [0.805, 0.853]</td><td>0.736 [0.718,0.753]</td><td>0.923 [0.920,0.926]</td><td>0.874</td></tr><tr><td>FleXray (ours)</td><td>0.949 [0.947, 0.950]</td><td>0.834 [0.715, 0.931]</td><td>0.871 [0.855, 0.884] [0.934,0.942]</td><td>0.938</td><td>0.915 [0.912, 0.919]</td><td>0.831 [0.806,0.855] [0.716, 0.756]</td><td>0.737</td><td>0.924 [0.921,0.927]</td><td>0.875</td></tr><tr><td>nnU-Net (in-domain)</td><td>0.988 [0.987, 0.988]</td><td>0.928 [0.885, 0.959]</td><td>0.921 [0.907, 0.931] [0.979, 0.985]</td><td>0.982</td><td>0.866 [0.856,0.875]</td><td>0.864 [0.823,0.898]</td><td>0.800 [0.772,0.825]</td><td>0.961 [0.959,0.962]</td><td>0.914</td></tr></table>

## E.2.2 Finetuned SAM2 encoders show no significant advantage over a U-Net

To test whether the visual prior of a natural-image foundation model improves transfer to X-rays, we fully finetune SAM2 [98] image encoders in place of the FleXray U-Net. We consider all four SAM2 Hiera variants, Tiny, Small, Base+, and Large (26.9M, 34.0M, 68.7M, and 212.2M parameters), each paired with a linear segmentation head over the 60-channel label protocol. SAM2 and FleXray (single) share the complete data engine, label harmonization, augmentation pipeline, training objective, and evaluation pipeline, including the same 16-sample test-time augmentation.

The SAM2 backbones are finetuned end-to-end at a reduced learning rate $( 3 \times 1 0 ^ { - 5 }$ , versus $3 \times 1 0 ^ { - 4 }$ for the segmentation head). SAM2 uses AdamW weight decay of 0.05 for both the backbone and head, compared with 0.01 for FleXray (single). Tiny, Small, and Base+ use 896×896 inputs and 2,000 epochs; Hiera Large uses

$1 0 2 4 \times 1 0 2 4$ inputs and 1,000 epochs, compared with 256×256 inputs and 2,000 epochs for FleXray (single). In preliminary sweeps we trained every Hiera scale at backbone learning rates of $3 \times 1 0 ^ { - 4 }$ (uniform with the segmentation head), $3 \times 1 0 ^ { - 5 }$ , and $1 0 ^ { - 5 }$ , holding the head learning rate fixed. The 10× reduced backbone rate of $3 \times 1 0 ^ { - 5 }$ performed best, with a ranking that was stable across encoder scales, so all final runs finetune the full, unfrozen encoder at this rate.

Accuracy rose monotonically with encoder scale. The equal-dataset mean Dice was 0.812 [0.753, 0.857] for Hiera Tiny, 0.838 [0.781, 0.885] for Small, 0.855 [0.805, 0.896] for Base+, and 0.876 [0.829, 0.916] for Large, against 0.870 [0.817, 0.915] for FleXray (single) (n=1,458 test images across the eight held-out datasets; Figure 10). On the same images, Tiny and Small trailed FleXray (single) by 0.058 [0.023, 0.098] and 0.031 [0.008, 0.056] Dice, whereas Base+ trailed it by 0.014 [−0.005, 0.034] and Large exceeded it by 0.006 [−0.009, 0.023], intervals that both include zero (paired hierarchical bootstrap of the equal-dataset mean difference). Given our data engine, we find no statistically detectable accuracy advantage for the finetuned SAM2 encoders over a from-scratch U-Net in these experiments (Methods B.1).

## E.2.3 Image-space augmentation drives transfer to real X-rays

Transfer from rendered to real X-rays depends on appearance robustness that the model can only acquire through augmentation. Our data-engine injects variation at two stages: at render time, through per-label attenuation randomization, and in image space, through the augmentation preset of Table 3. To attribute the transfer gain between these stages, we trained four regimes that add components cumulatively: (i) a base regime with neither; (ii) attenuation randomization alone; (iii) attenuation plus intensity inversion, which we isolate because radiographs are displayed in both polarities, making inversion the single largest appearance shift in the preset; and (iv) attenuation plus the complete preset, which is FleXray (single). All other training choices are unchanged.

![](images/25e747daa6dc8ef7d9749009e7a3a3bb7ed87c964ddc8bad8053161abebfe06a.jpg)  
Figure 10. Per-dataset Dice of finetuned SAM2 encoders and FleXray (single). Per-image Dice on the test partitions of the eight held-out real-radiograph datasets for fully finetuned SAM2 Hiera encoders with a linear segmentation head and FleXray (single). Points are images; boxes show median and quartiles, whiskers 1.5× the inter-quartile range; numbers give per-dataset means. Right, one point per dataset (n=8); numbers give the equal-dataset mean Dice.

The equal-dataset mean Dice rose from 0.353 for the base regime to 0.438 with attenuation randomization alone, to 0.650 with intensity inversion added, and to 0.870 with the complete preset (n=1,458 test images across the eight held-out datasets; Figure 11). Perdataset paired comparisons localize the three additions; every difference named below is significant. Attenuation randomization alone improved only the adult chest radiographs and the pelvic fluoroscopy (+0.15 Dice on DarwinCVD19 and +0.07 on DeepFluoro) and was indistinguishable from the base regime on the other six datasets. Adding inversion improved seven of the eight datasets (all but the eight-image LowerLimbs), most on the hip and pelvic views (+0.46 on HipRay and +0.39 on DeepFluoro). Only the complete preset recovered the wrist and elbow datasets, which stayed below 0.3 Dice under every partial regime (+0.90 on RAM-W600 and +0.57 on ElbowLat over the inversion regime), whereas on the rib and pediatric datasets the inversion regime already matched it. Attenuation randomization therefore contributes modestly in isolation, whereas the imagespace augmentations account for most of the transfer, and no single image-space operation suffices: inversion recovers only part of the full preset’s gain, and the remainder is concentrated on the peripheral anatomy. We retain the complete preset in the released model.

![](images/52152e77d501a7e3f73c9dff87910e9d0a0aece1b65a1774bf8b2d65ed0384e5.jpg)  
Figure 11. Cumulative augmentation ablation. Equal-dataset mean Dice on the test partitions of the eight held-out datasets (n=1,458 images) as attenuation randomization, intensity inversion, and the complete augmentation preset are progressively added. Base trains with neither attenuation randomization nor image-space augmentation. Error bars are hierarchical 95% confidence intervals over datasets and then images.

## E.2.4 Real X-rays provide training coverage for peripheral radiographs

FleXray draws supervision from three families of sources: the MOOSE-derived streams (online and offline renders), five supplemental CT datasets, and four manually annotated real X-ray sources (Table 1). We train three mixtures that divide their MOOSE-derived sampling budget equally between online rendering and offline enhanced DRRs: (i) the complete mixture (75%/21%/4% MOOSEderived/supplemental CT/real X-ray); (ii) the mixture without real X-rays (79%/21%/0%); and (iii) the mixture without supplemental CT (96%/0%/4%). When a source family is removed, its sampling share is returned equally to the two MOOSE-derived streams, keeping the number of optimization steps fixed. We quantify each source family’s contribution by comparing the complete mixture with the corresponding omission. An additional reference is trained on online MOOSE renders alone.

Table 7. Computational efficiency of FleXray and the generalist baselines. Parameter count, peak GPU memory, and per-image latency of model execution at batch size 1 in FP32 on an NVIDIA V100 at each method’s native input resolution (Methods C.4). Latency is the mean over eight test images, one per primary segmentation evaluation dataset, with a 95% confidence interval (Appendix K.1). TotalSeg2D sums its five sequential anatomy-group models; FluoroSAM uses 62 text prompts without point prompts, and its latency grows with the number prompted. Ensemble and TTA timings include prediction averaging, with TTA timings also including augmentation and inverse alignment. FleXray rows are as in Table 6.
<table><tr><td>Method</td><td>(M)</td><td>Params Peak GPU mem. Latency (ms) (GB)</td><td>mean [95% CI]</td></tr><tr><td>PAXray</td><td>73.4</td><td>0.80</td><td>15.5 [15.4, 15.8]</td></tr><tr><td>TotalSeg2D (5 subnetworks)</td><td>167.5</td><td>0.89</td><td>23.8 [23.5, 24.1]</td></tr><tr><td>FluoroSAM</td><td>268.7</td><td>1.64</td><td>623.7 [615.1, 632.3]</td></tr><tr><td>FleXray (single)</td><td>101.4</td><td>0.62</td><td>14.7 [14.7,14.8]</td></tr><tr><td>FleXray (single) + 16 TTA</td><td>101.4</td><td>0.68</td><td>288.4 [287.5, 289.4]</td></tr><tr><td>FleXray (no TTA)</td><td>507.0</td><td>2.31</td><td>74.2 [74.1,74.3]</td></tr><tr><td>FleXray (ours)</td><td>507.0</td><td>2.36</td><td>1242.3 [1241.3,1243.1]</td></tr></table>

We focus on source effects that are significant (Appendix K.1) and at least 0.01 Dice. Under this criterion, the real X-ray sources (535 annotated training radiographs, 4% of the sampling budget) improve two datasets: RAM-W600, whose wrists are otherwise essentially unsegmented (0.004 to 0.946, n=124), and ElbowLat (+0.325 Dice, n=91; both p<0.001; Table 9). Including real X-rays raises the equal-dataset mean Dice from 0.712 to 0.870. The supplemental CT datasets (503 annotated training subjects, 21% of the budget) provide an additional gain on ElbowLat (+0.024 Dice, n=91, p<0.001). No other dataset meets both criteria in either comparison. These comparisons show the importance of real X-ray annotations for the wrist and elbow datasets, with supplemental CT providing a smaller additional benefit on the elbow. The held-out suite does not evaluate the skull and shoulder anatomy targeted by some supplemental CT sources, so this ablation does not measure their contribution to those regions.

## F Failure cases

Figure 12 illustrates limitations of FleXray on six radiographs outside the evaluation suite, including oblique and lateral views of peripheral anatomy, a heavily collimated and overexposed shoulder, pediatric anatomy, and surgical hardware. These examples highlight gaps in the acquisition conditions and anatomical variation represented by the training sources, motivating additional targeted training data.

## G Code availability

Code is available at https : / / github . com / VictorButoi / FleXray. Trained model weights are available at https: //huggingface.co/VictorButoi/flexray-base. The release is intended for research use. Our enhanced-DRRs are distributed as a precomputed dataset in the same repository (Appendix H).

## H Data availability

We used the following publicly available CT datasets (Table 1):

• ElbowCT (https://figshare.com/articles/dataset/3D models\_of\_elbow\_joints\_along\_with\_corresponding CT\_data\_from\_Chinese\_individuals/28245599)

• HaN-Seg (https : / / han-seg2023 . grand-challenge. org/)

• MOOSE (https://registry.opendata.aws/enhancepet-1-6k/)

• PedsCT (https://www.cancerimagingarchive. net/ collection/pediatric-ct-seg/)

• RSNAFrac (https://www.kaggle.com/competitions/ rsna-2022-cervical-spine-fracture-detection/)

• Shoulder-CT (https : / / www.kaggle.com / datasets / syxlicheng / automatically - transform - ct - datasets - into-drrs)

We used the following publicly available X-ray datasets (Tables 1 and 4):

• AASCE (https://aasce19.github.io/)

• BTXRD (https : / / doi . org / 10 . 6084 / m9 . figshare . 27865398)

• DarwinCVD19 (https://darwin.v7labs.com/v7-labs/ covid-19-chest-x-ray-dataset)

• DeepFluoro (https : / / huggingface . co / datasets / eigenvivek/xvr-data)

Table 8. Online rendering fraction sweep. MOOSE-derived data accounts for 75% of total training steps (Table 1). The first column gives the fraction of this budget assigned to online DRRs, with the remaining fraction assigned to offline enhanced DRRs. We report per-dataset mean Dice for the five networks and their ensemble (the released FleXray) on the test partitions of the eight held-out real X-ray datasets (n is the number of test images). Each cell reports the per-example mean Dice with its 95% confidence interval (Appendix K.1). Mean Dice averages the per-dataset means with equal weight. All rows use 16-sample TTA.
<table><tr><td>Online rendering fraction</td><td>RAM-W600 (n = 124)</td><td>LowerLimbs  $( n = 8 )$ </td><td>ElbowLat  $( n = 9 1 )$ </td><td>HipRay  $( n \dot { = } 2 \dot { 1 } )$ </td><td>DeepFluoro  $( n \dot { = } 2 1 3 )$ </td><td>PedsTorso  $( n = 1 2 )$ </td><td>VinDr-Rib  $( n = 3 7 )$ </td><td>DarwinCVD19  $( n = 9 5 2 )$ </td><td>Mean Dice</td></tr><tr><td>1</td><td>0.947 [0.946,0.948]</td><td>0.705 [0.398,0.924]</td><td>0.875 [0.860, 0.887]</td><td>0.946 [0.942, 0.950]</td><td>0.911 [0.907, 0.915]</td><td>0.824 [0.795,0.852]</td><td>0.724 [0.699,0.747]</td><td>0.928 [0.924,0.931]</td><td>0.858</td></tr><tr><td>2/3</td><td>0.947 [0.945,0.948]</td><td>0.852 [0.755,0.923]</td><td>0.871 [0.855, 0.883]</td><td>0.936 [0.933, 0.940]</td><td>0.915 [0.912, 0.919]</td><td>0.826 [0.803, 0.848]</td><td>0.727 [0.703,0.747]</td><td>0.919 [0.916, 0.922]</td><td>0.874</td></tr><tr><td>1/2</td><td>0.946 [0.944,0.947]</td><td>0.835 [0.723,0.920]</td><td>0.864 [0.847,0.879]</td><td>0.936 [0.932, 0.940]</td><td>0.910 [0.905, 0.914]</td><td>0.817 [0.789,0.843]</td><td>0.726 [0.704,0.745]</td><td>0.922 [0.920,0.925]</td><td>0.870</td></tr><tr><td>1/3</td><td>0.946 [0.944,0.948]</td><td>0.846 [0.742,0.924]</td><td>0.865 [0.848, 0.878]</td><td>0.934 [0.929, 0.938]</td><td>0.910 [0.906, 0.914]</td><td>0.832 [0.812,0.853]</td><td>0.710 [0.689,0.730]</td><td>0.920 [0.917,0.923]</td><td>0.870</td></tr><tr><td>0</td><td>0.947 [0.945, 0.949]</td><td>0.765 [0.619,0.892]</td><td>0.834 [0.806, 0.857]</td><td>0.927 [0.924,0.931]</td><td>0.893 [0.889, 0.898]</td><td>0.818 [0.790,0.844]</td><td>0.722 [0.700,0.740]</td><td>0.913 [0.910,0.916]</td><td>0.852</td></tr><tr><td>Ensemble</td><td>0.949 [0.947,0.950]</td><td>0.834 [0.715,0.931]</td><td>0.871 [0.855, 0.884]</td><td>0.938 [0.934, 0.942]</td><td>0.915 [0.912, 0.919]</td><td>0.831 [0.806,0.855]</td><td>0.737 [0.716,0.756]</td><td>0.924 [0.921,0.927]</td><td>0.875</td></tr></table>

![](images/0a6a4d69d93158c0ae876a885d58eaee0b033298a928c700641700ae0bd2d19f.jpg)  
Figure 12. Illustration of failure cases. Predictions of the released ensemble with test-time augmentation on six radiographs outside the evaluation suite. a, Oblique finger. b, Lateral wrists. c, Heavily collimated, overexposed shoulder. d, Lateral ankle o an 11-year-old. e, Forearm with plate and screws. f, Lower leg of a 4-year-old. Panels a, c, and e are from the MURA validation split and are not among the MURA images annotated for training; b, d, and f are from Radiopaedia. Predictions use the released FleXray ensemble with test-time augmentation.

• ElbowLat (https://universe.roboflow.com/ionspace/ elbow\_lat-lnn0s-pmycd)

• FootBones (https://universe.roboflow.com/monchbot1/ foot\_op)

• HandBones (https : / / universe . roboflow . com / boneage-x90qt/-hand-bones-mdjkr)

• HipRay (https : / / data . mendeley . com / datasets / zm6bxzhmfz/1)

• LowerLimbs (https : / / universe . roboflow . com / orthopedicstitching/bone-identifier-1rey5)

• MTDDH (https : / / doi . org / 10 . 57760 / sciencedb . 24372)

• MURA (https://stanfordmlgroup.github.io/competitions/ mura/)

• RAM-W600 (https : / / huggingface . co / datasets / TokyoTechMagicYang/RAM-W600)

• PedsTorso (https://universe.roboflow.com/monchbot1/ thoracoabdominal)

• VinDr-Rib (https://vindr.ai/ribcxr)

Versions of the X-ray datasets whose licenses permit redistribution (BTXRD, ElbowLat, FootBones, Hand-Bones, HipRay, LowerLimbs, and MTDDH; all CC BY 4.0), repackaged with dataset-native labels in the format that FleXray ingests directly, are available at https://huggingface.co/datasets/VictorButoi/flexray-data, together with the per-image split assignments and qualitycontrol exclusions of every X-ray source. The same repository holds our manual forearm and humerus annotations for MURA (masks and an image manifest keyed to MURA file paths, without the source images, in accordance with the MURA Research Use Agreement). The remaining sources are not redistributed; for these we provide the label specifications used to map them into the protocol. Our 138,063 quality-controlled enhanced DRRs are synthetic and released with this work under CC BY-NC 4.0, inherited from the AutoPET-derived subset of the MOOSE source CTs [99], in the FluXray/ folder of the same repository.

Table 9. Training-source ablation. Per-example mean label Dice, shown as mean [95% CI] (Appendix K.1), for the four training-source mixtures on the test sets of RAM-W600 and ElbowLat (n images each), the two datasets showing significant source effects of at least 0.01 Dice (Appendix E.2.4). Source effects compare the complete mixture with each version omitting a source family; these three mixtures divide their MOOSEderived budget equally between online and enhanced offline DRRs. Mean Dice weights all eight held-out datasets equally. Bold marks the highest mean and mixtures not significantly different from it on that dataset.  
initial draft of the 2D-3D registration experiment. N.D. conceived the initial idea, identified the training and evaluation datasets, and provided technical and application feedback throughout the project. All authors contributing to editing and providing technical insight and feedback on drafts.
<table><tr><td>Mixture</td><td>RAM-W600 (n = 124)</td><td>ElbowLat (n = 91)</td><td>Mean Dice</td></tr><tr><td>Online MOOSE only</td><td>0.004 [0.002,0.006]</td><td>0.435 [0.384,0.485]</td><td>0.697</td></tr><tr><td>Without real X-rays</td><td>0.004 [0.003, 0.005]</td><td>0.541 [0.483, 0.597]</td><td>0.712</td></tr><tr><td>Without supplemental CT</td><td>0.948 [0.946,0.949]</td><td>0.842 [0.820,0.860]</td><td>0.868</td></tr><tr><td>Complete mixture (flagship)</td><td>0.946 [0.944, 0.948]</td><td>0.865 [0.849,0.879]</td><td>0.870</td></tr></table>

## I Acknowledgments

This work was supported by the National Science Foundation Graduate Research Fellowship Program, the MIT Health and Life Sciences Collaborative (HEALS), the MIT CSAIL METEOR Fellowship. Quanta Computer Inc., and NIH grants R01 EB033773, NIBIB 5T32EB001680-19, and S10 OD038222. Compute was also provided by the Massachusetts Life Sciences Center (MLSC).

## J Author Contributions

V.I.B. developed the code, curated and preprocessed data, trained models, conducted experiments, created figures, and led manuscript development. V.G. developed the rendering toolkit central to the work, provided training and evaluation datasets, and executed and wrote the

## K Appendix

## K.1 Statistics and reproducibility

Uncertainty. Unless stated otherwise, results are means with 95% percentile-bootstrap confidence intervals from 10,000 resamples. Per-dataset estimates resample images; equal-dataset summaries resample datasets and then images. Task-specific analyses resample inference seeds (TTA), cases (BTXRD), or matched label–frame observations (view-angle analyses). MTDDH repeats are averaged within images before resampling. Registration errors are summarized by medians.

Comparisons. Paired t-tests assess mean differences in Dice and Cobb-angle error between methods evaluated on the same images; Cobb comparisons require measurable predictions from both methods. Welch’s t-test compares attenuation-related improvements between independent healthy and pneumonia groups. Registration errors are compared with Wilcoxon signed-rank tests, using ranks to limit the influence of extreme error magnitudes. Success and failure rates are reported descriptively. All tests are two-sided at $\alpha = 0 . 0 5$ . Reported p values are Holm-corrected across baseline comparisons or ablation pairs within datasets, within Cobb-angle severity strata, and across diagnostic conditions in the attenuation analysis.

## K.2 Implementation details

FleXray is trained using PyTorch [100] on NVIDIA H200 GPU nodes; each ensemble member trains on a single H200 for approximately 60–70 hours (roughly 320 GPUhours for the ensemble). We use PyTorch automatic mixed precision (AMP) with bfloat16 autocasting for eligible forward-pass operations; model parameters and optimizer updates remain in FP32. We store our datasets in individual LMDB databases using thunderpack [101] in numpy [102] arrays. We store images and volumes as float16 arrays and segmentation labels as uint8 arrays. We generate projections from CTs using nanoDRR [23] (https://github.com/eigenvivek/nanodrr). Standard data augmentations were implemented with the Kornia library [103]. The MURA forearm and humerus subsets were annotated in CVAT [104] using SAM2 [98] as an interactive point-prompted segmentation tool, followed by manual polygon correction of every mask.

## K.3 Anatomical coverage of public X-ray segmentation data

Figure 1b summarizes the anatomical coverage of the public X-ray segmentation datasets we located, listed in Table 10. Each dataset’s native labels are mapped into the 60-structure protocol as in Methods A.4, and an image counts toward a structure when it carries a mask of that structure, summed over the dataset’s partitions. For display, the protocol is collapsed to 25 regions: the individual vertebrae form the spine, the individual ribs form the ribs, and the clavicles are omitted. The distal radius and ulna visible in hand and wrist radiographs are not counted as coverage of those bones. The skeleton is the surface of one MOOSE subject’s label map, and each region is tinted by the logarithm of its image count on a shared color scale.

Table 10. Public X-ray segmentation datasets behind Figure 1b. Regions are the collapsed labels of Appendix K.3 that each dataset annotates. Labeled X-rays is the number of images carrying a mask of at least one listed region, summed over the dataset’s partitions. The radii and ulnae of the hand and wrist datasets are not counted because those radiographs show only the distal ends of the bones.
<table><tr><td>Dataset</td><td>Regions Labeled X-rays</td></tr><tr><td>CheXmask [20]</td><td>Heart, lungs 657,566</td></tr><tr><td>DarwinCVD19 [84]</td><td>Lungs 6,347</td></tr><tr><td>MendeleyCXR [21] Lungs</td><td>704</td></tr><tr><td>RAM-W600 [88]</td><td>Carpals, metacarpals 618</td></tr><tr><td>ElbowLat [85]</td><td>Humeri, radii, ulnae 601</td></tr><tr><td>FootBones [39]</td><td>Metatarsals, toes 571</td></tr><tr><td>DeepFluoro [45]</td><td>Femurs, hips, sacrum, spine 362</td></tr><tr><td>VinDr-Rib [90]</td><td>Ribs 245</td></tr><tr><td>HipRay [86]</td><td>Femurs, hips 139</td></tr><tr><td>HandBones [38]</td><td>Carpals, metacarpals, phalanges 93</td></tr><tr><td>PedsTorso [89]</td><td>Lungs, spine 78</td></tr><tr><td>LowerLimbs [87]</td><td>Femurs, fibulae, tibiae 56</td></tr></table>

## K.4 Radiopaedia image attributions

The 18 radiographs shown in Figure 4a and the three shown in Figure 12b,d,f are drawn from Radiopaedia.org and are used under the Creative Commons BY-NC-SA 3.0 license, in accordance with the Radiopaedia image use and attribution guidelines (https://radiopaedia.org/ articles/using-and-attributing-images-from-radiopaedia-1). Panels are numbered in reading order: band 1 is the upper pair of source and overlay rows and band 2 the lower pair, with columns counted left to right. Each source X-ray and its segmentation overlay share one attribution. Panel 1 (band 1, column 1): Case courtesy of Bahman

Rasuli, Radiopaedia.org, rID: 76252.

Source: Radiopaedia image 52385720.

Panel 2 (band 1, column 2): Case courtesy of Andrew Murphy, Radiopaedia.org, rID: 48335. Source: Radiopaedia image 25366595.

Panel 3 (band 1, column 3): Case courtesy of Frank Gaillard, Radiopaedia.org, rID: 37967. Source: Radiopaedia image 14074481.

Panel 4 (band 1, column 4): Case courtesy of Tan (Vivian) Hooi Hooi, Radiopaedia.org, rID: 200442. Source: Radiopaedia case 200442.

Panel 5 (band 1, column 5): Case courtesy of Tudor Hughes, Radiopaedia.org, rID: 214066. Source: Radiopaedia image 71451814.

Panel 6 (band 1, column 6): Case courtesy of Sigmund Stuppner, Radiopaedia.org, rID: 45298. Source: Radiopaedia image 22567772.

Panel 7 (band 1, column 7): Case courtesy of Liam Pugh, Radiopaedia.org, rID: 58094. Source: Radiopaedia image 35566320.

Panel 8 (band 1, column 8): Case courtesy of Tudor Hughes, Radiopaedia.org, rID: 222397. Source: Radiopaedia image 72933366.

Panel 9 (band 1, column 9): Case courtesy of Stefan Tigges, Radiopaedia.org, rID: 195402. Source: Radiopaedia image 67285607.

Panel 10 (band 2, column 1): Case courtesy of Andrew Dixon, Radiopaedia.org, rID: 31533. Source: Radiopaedia image 8689771.

Panel 11 (band 2, column 2): Case courtesy of Yaïr Glick, Radiopaedia.org, rID: 89657. Source: Radiopaedia case 89657.

Panel 12 (band 2, column 3): Case courtesy of Ian Bickle, Radiopaedia.org, rID: 46399. Source: Radiopaedia image 23887924.

Panel 13 (band 2, column 4): Case courtesy of Amanda Er, Radiopaedia.org, rID: 85561. Source: Radiopaedia image 54151097.

Panel 14 (band 2, column 5): Case courtesy of Tudor Hughes, Radiopaedia.org, rID: 220698. Source: Radiopaedia image 72665895.

Panel 15 (band 2, column 6): Case courtesy of Andrew Murphy, Radiopaedia.org, rID: 48335. Source: Radiopaedia image 25366594.

Panel 16 (band 2, column 7): Case courtesy of Stefan Tigges, Radiopaedia.org, rID: 195402. Source: Radiopaedia image 67285609.

Panel 17 (band 2, column 8): Case courtesy of Ian Bickle, Radiopaedia.org, rID: 46578. Source: Radiopaedia image 23961548.

Panel 18 (band 2, column 9): Case courtesy of Andrew Murphy, Radiopaedia.org, rID: 48227. Source: Radiopaedia image 25293680.

Figure 12b: Case courtesy of Andrew Murphy, Radiopaedia.org, rID: 48227. Source: Radiopaedia image 25293683.

Figure 12d: Case courtesy of Tudor Hughes, Radiopaedia.org, rID: 220012. Source: Radiopaedia image 72575068.

Figure 12f: Case courtesy of Andrew Kirby, Radiopaedia.org, rID: 238301. Source: Radiopaedia image 75212780.