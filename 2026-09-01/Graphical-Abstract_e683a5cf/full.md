## Graphical Abstract

AI-enabled Low-Cost 3D Maize Ear Morphometry Platform at Breeding Scale

Therin Young, Elijah Rodriguez, Lisa Cofey, Talukder Zaki Jubery, Adarsh Krishnamurthy, Patrick Schnable, Baskar Ganapathysubramanian

![](images/741b0874c37a96c71ec65cfd3e4844fd594637455fa44960927f2abc903ace53.jpg)

## Highlights

## AI-enabled Low-Cost 3D Maize Ear Morphometry Platform at Breeding Scale

Therin Young, Elijah Rodriguez, Lisa Cofey, Talukder Zaki Jubery, Adarsh Krishnamurthy, Patrick Schnable, Baskar Ganapathysubramanian

• Low-cost stationary-camera NeRF pipeline gives 3-D maize ear traits from video.

• Ten-seed COLMAP with intrinsic averaging enforces 100% frame registration.

• A known-diameter holder scales the mesh; quality control flags failed fits.

• One command runs frames, COLMAP, NeRF, scaling and trait output to CSV.

• Length and volume match calipers and displacement across 250 maize ears.

# AI-enabled Low-Cost 3D Maize Ear Morphometry Platform at Breeding Scale

Therin Young<sup>a,1</sup>, Elijah Rodriguez<sup>b,1</sup>, Lisa Cofey<sup>c</sup>, Talukder Zaki Jubery<sup>b</sup>, Adarsh Krishnamurthy<sup>a,b</sup>, Patrick Schnable<sup>c,∗</sup>, Baskar Ganapathysubramanian<sup>a,b,∗</sup>

<sup>a</sup>Department of Mechanical Engineering, Iowa State University, Ames, 50011, Iowa, USA <sup>b</sup>Translational AI Research and Education Center, Iowa State University, Ames, 50011, Iowa, USA <sup>c</sup>Department of Agronomy, Iowa State University, Ames, 50011, Iowa, USA

## Abstract

Maize ear geometry (length, width, curvature, and volume) is closely tied to yield and grain-filling outcomes, but existing high-throughput phenotyping pipelines remain constrained by the cost, labor, and specialized hardware they require. We developed and validated a low-cost pipeline that reconstructs a watertight 3-D mesh of a maize ear from a single 20-second video captured with a consumer-grade DSLR on a motorized turntable under uniform LED illumination. Camera poses from a multi-seed COLMAP procedure initialize a Neural Radiance Field (NeRF), and a cylindrical holder of known diameter, visible in every frame, provides automatic metric scaling with downstream geometric quality control. Applied to 300 ears spanning a diverse maize inbred panel, 250 (83.3 %) passed automated processing and quality control. Skeleton length agreed with manual caliper measurements across all 250 ears (R<sup>2</sup> = 0 964, RMSE = 4 68 mm), and convex-hull volume agreed with water-displacement volume on a 15-ear subset spanning the full size range (R<sup>2</sup> = 0 982, RMSE = 5 26 mL). Residual length error grew with ear curvature, whereas bounding-box height, which records the same straight-line chord as calipers, showed no such trend; the discrepancy therefore originates in the measurement definition, since calipers record the chord while skeleton length traces the geodesic arc. The capture hardware costs approximately 607 USD, and operator involvement fell from roughly five minutes to one minute per ear, with all downstream processing running unattended. The platform provides a founda tion for breeding-scale 3-D ear phenotyping.

Keywords: 3-D phenotyping, maize ear, neural radiance fields (NeRF), COLMAP, high-throughput phenotyping

## 1. Introduction

High-resolution characterization of maize ear traits such as total kernel number, row architecture, section-wise diameter profile, and overall shape is central to genetic analyses and yield prediction. However, the manual processes required to capture these traits are labor-intensive, subjective, and dificult to scale up for larger phenotyping studies that introduce additional constraints such as lighting, weather, and labor availability. Maize ear phenotyping is a technical challenge for high-throughput phenotyping research studies. Furthermore, the variation in maize ear geometries requires robust phenotyping pipelines capable of capturing these geometric and physical complexities.

Traditional two-dimensional (2-D) studies have produced low-cost, high-precision estimations of ear and cob length that are highly correlated with manual hand measurements [1, 2]. Open-source projects like EarCV leveraged fundamental computer vision principles to extract ear length and width, in addition to more complex traits like taper, curvature, tip fill, and color in diverse inbred datasets to assess each trait’s relationship with yield [3]. The opensource project EARBOX adds to the literature by also accurately capturing ear diameter along the length of the ear to increase throughput and more accurately estimate spatial organization metrics from rotating ears [4]. However, pure 2-D methods cannot recover back-side kernels, true volumes, or complete 3-D spatial distributions and sufer from the usual limitations that are inherent of in-field studies, such as environmental occlusions, single-view imaging, and limited robustness to mixed textures and complex geometries. A more recent study by Fan et al. features a DIY maize ear-imaging platform with a deep learning-based end-to-end phenotypic data extraction pipeline [5]. Ear length, diameter, volume, and weight were calculated along with other grain-based traits which included kernel number (KN), kernel row number (KRN), kernel number per row (KPR), kernel thickness, kernel width, and thousand kernel weight. The imaging platform used in the study was very similar to that used in this study; a key distinction being that maize ear videos were not used for 3-D reconstructions. Rather, the videos were converted into projected 2-D images that represent a flattened or unrolled view of the maize ear which were then used as input features for training a CNN. Fan et al. demonstrate that video-based platforms can support both ear-level and kernel-level trait extraction, simultaneously. Kernel-level measurements are not yet directly extracted from the 3-D NeRF mesh in the current pipeline; the trait set reported here is therefore ear-level rather than kernel-level. A companion study [6] extends this same reconstruction platform to eleven ear- and kernel-level traits (including kernel count, kernel row number, kernels per row, per-kernel surface area, volume, packing geometry, hue, and aspect ratio) via cylindrical unwrapping of the point cloud and zero-shot kernel instance segmentation. Three-dimensional (3-D) high-throughput phenotyping for maize ears can help address the limitations of 2-D studies by providing access to the full surface geometry of the ear, a prerequisite for volumetric traits and complete cross-sectional profiles that single-view imaging cannot capture. Volume, geodesic arc length, and cross-sectional eccentricity profiles have no counterpart in a single projection, so for these traits the relevant question is access rather than relative accuracy. The bottleneck lies in accurately recovering the ear surface and, for finer-grained analyses, segmenting structural components such as kernels, rows, and ear boundaries.

Classical multi-view stereo (MVS) approaches such as COLMAP [7, 8] and Metashape yield detailed surfaces but require well-controlled views, careful backgrounds, and expert tuning, and are slow; for commercial packages, licensing cost and operator efort limit routine use at breeding scale [9]. Turntable rigs built from consumer hardware have already brought metric 3-D reconstruction of individual fruits within reach of a modest budget [10]. Active 3-D sensors (structured light, laser) provide precise geometry, but are costly, calibration-heavy, and sensitive to surface reflectance and lighting [11, 9]. Ear-focused 3-D pipelines underline the phenotypic value of true 3-D while still relying on resource-intensive stages that hinder scaling [12].

Neural Radiance Fields (NeRF) reconstruct continuous volumetric scenes from images [13]. Modern implementations such as Instant-NGP and Nerfstudio’s nerfacto accelerate training using multi-resolution hash encodings and practical training recipes [14, 15]. A stationary-camera variant (rotating object, fixed camera) simplifies capture and supports more delicate or bulky optics. Recent agricultural work reports high-fidelity plant geometry from short videos with COLMAP-based poses, both in the field [16] and against commercial MVS baselines [17]. Learning-based radiance/geometry fields such as 3D Gaussian Splatting provide an additional path for fast, photorealistic scene models in phenotyping contexts [18].

As remote sensing technologies such as LIDAR (active sensing) and photogrammetry (passive sensing) have become more available and eficient over the past decade, several high-throughput 3-D phenotyping studies have emerged and advanced the field. Many of these studies focus on micro-level traits of the maize ear, such as kernel row count and kernels per row [19]. Zhu et al. used a Laplace-based skeleton extraction algorithm to generate a 3-D point cloud skeleton followed by several sub-skeleton refinements and localized constraints to accurately segment maize ear components and accurately measure ear height, length, diameter, and the ratio of plant height to ear height [20]. The $R ^ { 2 }$ values for automated vs manual hand measurements were 0.97, 0.78, 0.73, and 0.96, respectively. Sun et al. used structured light reconstructions for KPR and KRN detections achieved through the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm with a directional erosion strategy to improve robustness for adherent kernels [19]. Sun et al.’s study resembles our study in how the 3-D reconstructions were achieved passively. However, their use of structured light resulted in a higher-resolution point cloud, which made KPR and KRN detections accessible. The use of NeRF in our study coupled with COLMAP currently supports structurally accurate reconstructions at the macroscopic level; direct 3-D kernel-level measurements from the NeRF mesh have not yet been achieved at this resolution. Related work from our own group has separately shown that zero-shot instance segmentation with the Segment Anything Model [21] can recover kernel counts directly from 2-D ear images without task-specific training [22]. The ear-level trait set reported in the present study is a deliberately preliminary scope: a companion study [6] extends the same reconstruction platform to direct 3-D kernel-level extraction, cylindrically unwrapping the calibrated point cloud and applying zero-shot instance segmentation to jointly recover eleven ear- and kernel-level traits.

Table 1: Comparative analysis of automated maize ear phenotyping methods. Abbreviations marked <sup>a</sup> are outputs of the cited method only; all others are defined in Table 3. Cost is a qualitative tier (\$ = consumer-grade hardware; \$\$ = specialist hardware; \$\$\$ = specialist/industrial-grade or resource-intensive hardware) inferred from each study’s reported equipment, not a verified price.
<table><tr><td>Method</td><td>Modality</td><td>Hardware</td><td>Cost</td><td>Outputs</td></tr><tr><td>Makanza et al. [1], Miller et al. [2]</td><td>2-D image</td><td>Consumer</td><td>$</td><td>Ear lengtha, cob lengtha</td></tr><tr><td>Gonzalez et al. [3] (EarCV)</td><td>2-D image</td><td>Consumer</td><td>$</td><td>Ear lengthª, ear widthª, Tapera, Curvatureª, Tip Fillª, Colora</td></tr><tr><td>Oury et al. [4] (EARBOX)</td><td>2-D rotating image</td><td>Consumer</td><td>$</td><td>Ear lengthª, diameter profileª</td></tr><tr><td>Fan et al. [5]</td><td>360° video (2-D unrolled)</td><td>Consumer</td><td>$</td><td>Ear lengtha, diametera, volumeª, weighta, KNa, KRNa, KPRª, kernel thicknessª, kernel widthª, TKWa</td></tr><tr><td>Yang et al. [12]</td><td>3-D rotational reconstruction</td><td>Specialist, resource- intensive</td><td>$$$</td><td>Ear 3-D geometry</td></tr><tr><td>Zhu et al. [20]</td><td>3-D point cloud (Laplace skeleton)</td><td>Specialist</td><td>$$</td><td>Ear heightª, ear length, ear diameter, plant-height ratioª</td></tr><tr><td>Sun et al. [19]</td><td>Structured light</td><td>Specialist</td><td>$$$</td><td>KRNa, KPRª</td></tr><tr><td>Ours</td><td>Stationary-camera 360°video (NeRF)</td><td>Consumer</td><td>$</td><td>Skeleton Length, Max Width, Chord/Arc Ratio, Taper-to-Tip, Curvature Score, Mean Eccentricity, Convex-Hull Volume, Surface Areab</td></tr></table>

<sup>a</sup>Output of cited method only; see respective reference for definition.  
<sup>b</sup>Ear-level outputs of this platform paper; a companion study [6] extends this  
point cloud to eleven ear- and kernel-level traits (kernel count, row number, kernels per row,  
per-kernel surface area, volume, packing, hue, and aspect ratio).

Table 1 summarizes existing automated maize ear phenotyping methods and situates the present work among them. Relative to 2-D pipelines, the present pipeline recovers true 3-D surface geometry rather than a single-view projection; relative to classical MVS and active 3-D sensors, it uses only consumer-grade hardware and requires no specialist calibration equipment; and relative to prior 3-D ear-phenotyping studies, it is the first to combine a stationary-camera NeRF reconstruction with fully automated metric scaling and quality control at breeding scale.

We propose a frugal, reproducible maize-ear phenotyping pipeline integrating established components (COLMAP pose estimation, Nerfstudio’s nerfacto model, and convex-hull surface reconstruction) with three design contributions:

1. A multi-seed COLMAP strategy that runs ten independent reconstructions and retains only 100 %-coverage solutions, averaging intrinsics across the retained runs.

2. A known-diameter cylindrical fiducial for automatic metric scaling without external calibration objects.

3. A single-command automation layer that wraps all stages from video to a trait CSV file.

The rest of the paper is structured as follows. Section 2 details the imaging hardware, data acquisition protocol, multi-seed COLMAP pose estimation, NeRF reconstruction, metric scaling, geometry quality control, and traitextraction procedures. Section 3 reports COLMAP registration outcomes, geometric validation against manual caliper and water-displacement references, and population-level trait distributions across the SAM diversity panel. Section 4 interprets these results in light of the pipeline’s practical impact, design advantages, biological relevance, and current limitations, and outlines directions for future work.

## 2. Materials and Methods

The overall workflow, from raw turntable video to a per-ear trait CSV, is summarized in Figure 1. Camera poses are first estimated by the multi-seed COLMAP strategy (Section 2.3.1) and used to train a nerfacto NeRF, from which a dense point cloud is exported. The point cloud is cleaned, metrically scaled using the cylindrical holder fiducial (Section 2.5), and passed through an automated geometry quality-control gate (Section 2.6) before being converted to a watertight convex-hull mesh, skeletonized, and reduced to the trait set defined in Section 2.7. A complementary, qualitative view of the intermediate 3-D output at each of these stages is shown in Figure 3 (Section 2.3).

![](images/740148c644bfa0087a5a87311dac5583559832bd286f99813a75ffc75fe9266c.jpg)  
Figure 1: Algorithmic pipeline for 3-D maize ear phenotyping. Camera poses from multi-seed COLMAP initialize nerfacto NeRF training, which yields a dense point cloud. The point cloud is cleaned, isolated to the cylindrical holder for metric scaling, and passed through the automated geometry quality-control gate. The scaled point cloud is converted to a watertight convex-hull mesh, voxelized and skeletonized, and divided into ten cross-sections, from which the size and shape traits of Table 3 are computed and written to a per-ear CSV.

## 2.1. Imaging hardware and cost breakdown

Figure 2 illustrates the acquisition rig. A consumer-grade DSLR (Canon EOS 6D) fitted with a 50 mm focal length lens is positioned a few decimeters from the specimen; the ear is mounted on a low-cost motorized turntable that rotates at approximately 3 rpm. The object is illuminated by two of-axis 12 V LED panels that provide uniform lighting, while a matte-black backdrop suppresses background clutter. A color-calibration chart is placed in the field of view to guarantee consistent photometric conditions across all captures. The sample holder is a cylindrical tube of 84 mm (3.3 in) outer diameter, 3-D printed in PLA. The cylindrical tube’s diameter serves as a metric fiducial for later scaling. All imaging components are commercially available and together cost around 607 USD.

## 2.2. Data acquisition

Maize ears were sourced from a diversity panel of 370 inbred maize genotypes (the SAM panel, Iowa State University), from which a random subset of 300 ears was selected for processing. Of the subset, 250 ears produced a complete end-to-end reconstruction; the remaining 50 ears were excluded automatically (15 at the COLMAP registration stage and 35 at the geometry quality-control stage) as described in Section 2.3.1 and Section 2.6. Data collection spanned multiple sessions on diferent days and was carried out by 5 independent operators; five operators followed a fixed capture protocol across sessions. Two motorized turntable models from the same manufacturer were used across sessions; both operated at approximately 3 rpm and produced comparable frame sequences.

Table 2: Bill of Materials (USD).
<table><tr><td>Component</td><td>Model / Specs</td><td>Cost</td></tr><tr><td>DSLR + lens</td><td>Canon EOS 6D + 50 mm</td><td>$500</td></tr><tr><td>Turn-table</td><td>PC-controlled turntable, 12.6 in platform, 350 lb load</td><td>$50</td></tr><tr><td>LED panels (2)</td><td>12 V, 500 lm each</td><td>$40</td></tr><tr><td>Black backdrop</td><td>Matte board, 1 m²</td><td>$10</td></tr><tr><td>Cylindrical holder</td><td>3-D printed PLA, 3.3 in diameter</td><td>$7</td></tr><tr><td>Total</td><td></td><td>$ 607</td></tr></table>

![](images/8b4d7ed8a636ac0f764c670f696d2b2a639087431d5608f05e50698bdcb06da5.jpg)  
Figure 2: Imaging hardware for 3-D maize ear phenotyping. The middle image illustrates the imaging setup: a fixed DSLR camera records a rotating maize ear mounted on a low-cost motorized turntable, illuminated by two of-axis LED panels against a matte-black backdrop. A color calibration chart ensures consistent photometric conditions across captures.

For each ear, the holder was placed on the turntable, and the camera and turntable (both programmatically controlled from and connected to a PC) were triggered together for synchronized video capture over a full 360° rotation. Videos were captured at 30 fps with a resolution of 1080p, yielding roughly 600 frames per ear (duration ≈20 s). Each recording was automatically renamed with a unique per-ear identifier and uploaded to cloud storage for downstream reconstruction. The raw video files were subsequently processed with ffmpeg as follows: frames were extracted at 3 fps, resulting in approximately 60 images per ear. The extracted image sequence together with the accompanying metadata form the input for the multi-pass COLMAP pose-estimation stage described in Section 2.3.1. All steps (synchronized capture, renaming, cloud upload, and frame extraction) are orchestrated by a small Python wrapper that creates the required directory hierarchy and logs the operations for reproducibility.

## 2.3. 3-D reconstruction

The 3-D reconstruction follows four sequential stages (Figure 3):

1. Camera pose estimation from the video using COLMAP with multi-seed initialization.

2. Volumetric reconstruction using a nerfacto Neural Radiance Field trained on the estimated poses.

3. Segmentation of the ear from the holder and background, followed by metric scaling using the known holder diameter.

4. Trait extraction from the scaled watertight mesh.

![](images/d2eb22e23e3b33109be5c851f4ff7b8989df2a879790cd58786c573afcfee690.jpg)  
Figure 3: Visual walkthrough of the 3D maize ear reconstruction pipeline. Whereas Figure 1 details the algorithmic steps of the pipeline, this figure shows the intermediate 3-D output produced at each stage. A 20-second video of the rotating ear is passed through COLMAP to estimate per-frame camera poses (Estimated Poses). The calibrated poses initialize a nerfacto Neural Radiance Field, which produces a dense 3-D reconstruction of the ear and its cylindrical holder (Reconstructed 3-D). Segmentation isolates the ear from the holder and background, and the known holder diameter provides a metric scale factor (Scaled 3-D Ear). Morphological traits, including volume, surface area, curvature score, and additional shape descriptors, are then extracted from the scaled watertight mesh.

## 2.3.1. Pose estimation: multi-seed COLMAP

Camera poses were estimated using COLMAP [7] via the ns-process-data front-end. For each video, ten independent COLMAP reconstructions were run using a sequential matcher and the default SIFT [23] ratio threshold (0.8); the only source of variability between runs was the random seed, which changes the order of feature-match sampling and RANSAC [24] initialization and can therefore produce diferent candidate models from the same input.

Post-processing proceeded as follows. For each run, frame-registration coverage was computed as

$$
\mathrm { C o v e r a g e } = \frac { N _ { \mathrm { r e g } } } { N _ { \mathrm { t o t } } } \times 1 0 0 \% .
$$

Runs with coverage below 100 % were discarded; if no run achieved full coverage, the ear was flagged for re-capture. Among retained runs, the run with the lowest mean reprojection error provided the camera extrinsics (per-frame pose transforms). Camera intrinsics (focal length, principal point, distortion coeficients) were averaged across all retained runs; in practice, intrinsic variation was small (<0.5 % relative change in focal length).

## 2.4. Neural radiancefield reconstruction

The camera poses and intrinsics from Section 2.3.1 were passed to nerfstudio (v0.4) using the nerfacto model, which uses a multi-resolution hash-grid MLP to represent a continuous radiance field. Training ran for 30 000 it erations (batch size 4096, learning rate $1 \times 1 0 ^ { - 2 } )$ on a single NVIDIA A100 (CUDA 12.1, PyTorch 2.2), requiring approximately fourteen minutes per ear.

The model optimizes a photometric loss with the inter-level and distortion regularizers introduced by Barron et al. [25]:

$$
\mathcal { L } = \frac { 1 } { \left| R \right| } \sum _ { \mathbf { r } \in R } \left\| \hat { C } ( \mathbf { r } ) - C _ { \mathrm { g t } } ( \mathbf { r } ) \right\| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { i n t e r } } \mathcal { L } _ { \mathrm { i n t e r } } + \lambda _ { \mathrm { d i s t } } \mathcal { L } _ { \mathrm { d i s t } } ,\tag{1}
$$

where $\hat { C } ( \mathbf { r } )$ is the rendered color for ray r and $C _ { \mathrm { { g t } } } ( { \bf r } )$ is the observed pixel color.

After training, a dense point cloud (5M points, outlier removal enabled) is exported with ns-export pointcloud and passed to the segmentation and scaling stage.

![](images/128bd67ca0ad5c730cf86097c4e474c1bb3c0ef6f62d19b069c092e0fd662b20.jpg)  
Figure 4: Automated metric scaling via the cylindrical holder. Left: Raw exported point cloud of the ear and its cylindrical holder. Middle: Smoothed point-density profile along the vertical (z) axis; the red dashed line marks the automatically detected cutof separating the ear from the holder base. Right: Points in the stable holder region are isolated and fitted with a circle by algebraic least squares; the fitted diameter relative to the known physical diameter (84 mm) yields a global scale factor applied to the entire reconstruction.

## 2.5. Metric scaling using the cylindrical holder

The nerfacto model reconstructs the scene in a normalized coordinate system; the exported point cloud therefore carries no intrinsic physical unit. We recover a metric scale factor automatically from the cylindrical sample holder, which is visible in every frame and whose outer diameter is known precisely $( d _ { \mathrm { t r u e } } = 8 4 $ mm), eliminating the need for a separate calibration target. Systematic geometric distortion is mitigated upstream by the averaged camera intrinsics from the multi-seed COLMAP strategy (Section 2.3.1).

The cylindrical holder points were isolated by retaining those whose z-coordinates fall within the region of nearly constant point density along the vertical axis (Figure 4, Middle). A thin horizontal slice $S = \{ { \bf p } \mid \mid z - z _ { c } \vert \leq \delta \}$ centered at z is projected onto the xy-plane and fitted with a circle by algebraic least squares [26], yielding estimated radius r and diameter $\hat { d } = 2 r$ . The global scale factor is then

$$
s = \frac { d _ { \mathrm { t r u e } } } { \hat { d } } ,\tag{2}
$$

and multiplying all point-cloud coordinates uniformly by s converts the reconstruction from normalized units to physical millimeters.

## 2.6. Geometry quality control via cylindrical holder cross-section analysis

Scale accuracy depends entirely on isolating the holder correctly, so the pipeline verifies that isolation before computing any trait. When the density profile along the vertical axis lacks a clear inflection point (due to partial holder occlusion, an ear positioned too close to the holder rim, or an unusually dense base region), the cutof boundary can fall in the wrong location. This produces three classes of failure: the isolated region may capture only a partial ear, include structural parts of the ear body alongside the holder, or correspond entirely to the wrong region of the point cloud. In all cases, the circle fit is applied to an incorrect point set, yielding a holder cross-section that is non-circular, an erroneous scale factor, or both.

Two scalar indicators are computed from the fitted holder cross-section to detect segmentation failures automatically. The first is the holder eccentricity

$$
e _ { \mathrm { h o l d e r } } = \sqrt { 1 - \left( \frac { \hat { b } } { \hat { a } } \right) ^ { 2 } } ,\tag{3}
$$

obtained by fitting a general conic shape [26] to the projected holder slice, yielding semi-axes $\hat { a } \ge \hat { b } > 0$ . A value of zero indicates a circular cross-section consistent with the physical holder; elevated values indicate that a non-holder region has been included in the fit. The second indicator is the fitted radius $\hat { r } = \hat { d } / 2$ in NeRF normalized units. Because the camera, lens, and camera-to-object distance are held constant across sessions, ˆr should fall within a narrow and reproducible range; a value that departs substantially from the expected range indicates that the circle fit has latched onto an entirely wrong region of the point cloud.

![](images/329933a14d9147e098b23a7c53e18d2307967b67dda36ce61fb355afc610c119.jpg)

B  
![](images/715b79e2af1e63194108d6721c617477492b1ad654f233021abfcb6fc81a9748.jpg)

C  
![](images/0c436b9df9ebd1ebd327075bba21648241282b7719f9287831a5603747a069fc.jpg)

D  
![](images/a3dc2f4954f07097fe1b62741fe45254a1a646c81fb41d843099d042e5434a8e.jpg)

![](images/26053632eb00c8c7b763fb3e4b86e9826ca01f35046924dac6374cf48af56440.jpg)

![](images/663be3fb38ee960e61a23e7b89d2e98572725ba7f410e354db6203cb451b69cd.jpg)

<table><tr><td>Trait</td><td>Value</td></tr><tr><td>Chord / Arc Ratio</td><td>0.9813</td></tr><tr><td>Taper to Tip</td><td>0.054</td></tr><tr><td>Curvature Score</td><td>0.0063</td></tr><tr><td>Max Bending</td><td>0.0077</td></tr></table>

Figure 5: Three-dimensional maize ear trait extraction pipeline. (A) Raw input point cloud with bounding box overlay. (B) Watertight solid mesh reconstructed from the point cloud; the mesh is converted to a volumetric solid prior to skeletonization to ensure topological correctness. (C) Topological skeleton extracted along the geodesic axis of the solid mesh, following the medial path from base to tip. (D) Trait landmark visualization showing key geometric measurements along the skeleton: the arc tracing the geodesic axis (red), the straight chord connecting tip to base (dashed blue), the point of maximum width (blue), and the point of maximum local bending (purple). (E) Major diameter profile measured at ten cross-sections distributed along the geodesic axis, with the position of maximum width indicated (dashed orange). (F) Cross-sectional eccentricity profile across the ten geodesic sections. Computed traits are summarized in the inset table: Chord/Arc Ratio, the ratio of the straight tip-to-base chord to the geodesic arc length (= 1 for a perfectly straight ear); Taper to Tip, the cross-sectional area at the distal section relative to the maximum section area (lower values indicate greater tapering); Curvature Score, the total turning angle along the skeleton normalized by geodesic length (higher values indicate more curved ears); and Max Bending, the strongest local bend rate at any single skeleton segment (higher values indicate a sharper localized bend).

An ear is automatically flagged if $e _ { \mathrm { h o l d e r } }$ exceeds the upper Tukey IQR fence

$$
e _ { \mathrm { h o l d e r } , i } > Q _ { 3 } ( e _ { \mathrm { h o l d e r } } ) + 1 . 5 \ \mathrm { I Q R } ( e _ { \mathrm { h o l d e r } } ) ,\tag{4}
$$

or if ˆr falls outside the expected session range, evaluated across all ears that passed COLMAP registration within the same processing batch. Ears not flagged automatically, but whose holder cross-section appeared visually anomalous during routine inspection were also excluded and recorded in the rejection log. All excluded ears are written to a rejection log with their $e _ { \mathrm { h o l d e r } }$ and ˆr values and the reason for exclusion. Representative failure cases are shown in the Supplementary Figure 14.

## 2.7. Trait extraction

Following metric scaling, the point cloud is processed through a series of geometric operations to extract morphological traits that characterize both the size and shape of the ear (Table 3). Trait definitions follow the vocabulary established by EarCV [3] and EARBOX [4]. The pipeline reports eleven whole-ear traits, six describing size and five describing shape, together with per-section values at ten cross-sections. The six size traits (skeleton length, maximum width, maximum-width position, convex-hull surface area, convex-hull volume, and bounding-box height) describe the overall dimensions of the ear. Shape traits capture geometry independently of absolute size: the Chord/Arc Ratio quantifies straightness; Taper to Tip describes the taper profile toward the tip; the Curvature Score and Max Bending measure axial bending; and Mean Eccentricity characterizes the cross-sectional roundness at ten sections distributed from base to tip. Each processing step is illustrated in Figure 5.

Table 3: Morphological trait definitions. Cross-sections are indexed 1 (base) to 10 (tip) along the centreline skeleton. Size traits are reported in physical units; shape traits are dimensionless ratios unless stated otherwise.
<table><tr><td>Trait</td><td>Unit</td><td>Definition</td></tr><tr><td>Size traits</td><td></td><td></td></tr><tr><td>Skeleton Length</td><td>mm</td><td>Arc-length of the centreline skeleton from base to tip</td></tr><tr><td>Max Width</td><td>mm</td><td>Largest major diameter across all 10 cross- sectional ellipses</td></tr><tr><td>Max Width Position</td><td>0-1</td><td>Fractional position of maximum width along the</td></tr><tr><td>Convex-Hull Surface Area</td><td> $\mathrm { c m } ^ { 2 }$ </td><td>skeleton (0 = base,  $1 = \operatorname { t i p } )$  Total surface area of the convex-hull mesh</td></tr><tr><td>Convex-Hull Volume</td><td> $\mathrm { c m } ^ { 3 }$ </td><td>Volume enclosed by the convex-hull mesh</td></tr><tr><td>BBox Height</td><td>mm</td><td>Height of the axis-aligned bounding box</td></tr><tr><td>Shape traits</td><td></td><td></td></tr><tr><td>Chord/Arc Ratio</td><td>0-1</td><td>Straight tip-to-base chord divided by skeleton arc-length; = 1 for a perfectly straight ear</td></tr><tr><td>Taper to Tip</td><td>0-1</td><td>Cross-sectional area at section 10 divided by the maximum section area; lower values indicate a</td></tr><tr><td>Curvature Score</td><td>rad mm−¹</td><td>sharper tip Sum of skeleton turning angles divided by total arc-length; higher values indicate a more curved</td></tr><tr><td>Max Bending</td><td> $\mathrm { r a d m m ^ { - 1 } }$ </td><td>ear Peak local turning angle divided by local seg- ment length; captures the sharpest single bend</td></tr><tr><td>Mean Eccentricity</td><td>0-1</td><td>Mean  $\sqrt { 1 - ( b / a ) ^ { 2 } }$  over 10 cross-sections; 0 = circular, 1 = flat ellipse</td></tr><tr><td colspan="3">Per-section traits (i = 1 .. . 10)</td></tr><tr><td>Section i major axis</td><td>mm</td><td>Major axis of the fitted ellipse at cross-section i</td></tr><tr><td>Section i minor axis</td><td>mm</td><td>Minor axis of the fitted ellipse at cross-section i</td></tr><tr><td>Section i eccentricity</td><td>0-1</td><td>Eccentricity of the fitted ellipse at cross-section i</td></tr></table>

Point-cloud cleaning (Panel A).. Before any geometric analysis, the raw point cloud is subjected to three automatic cleaning steps. First, DBSCAN clustering [27] (ε = 5 mm, minimum cluster size 50) isolates the largest connected component, discarding stray fragments and calibration artifacts. Second, a statistical outlier filter removes points whose mean distance to their $k = 2 0$ nearest neighbors exceeds 2 σ above the population mean, suppressing reconstruction noise on the surface. Third, turntable reflections are eliminated by a two-pass HSV color filter: the first pass removes globally blue pixels (hue $1 8 0 { - } 2 7 0 ^ { \circ }$ , saturation > 0 15, value < 0 85), and a second pass applies a relaxed saturation threshold (saturation > 0 08) to the bottom 25% of the scene where residual turntable color is most prevalent. The cleaned point cloud (Panel A) is carried forward to all subsequent steps.

Convex-hull reconstruction (Panel B).. The cleaned point cloud is converted directly to a three-dimensional convex hull using the incremental convex-hull algorithm in Open3D [28]. The resulting mesh $M = ( \mathcal { V } , \mathcal { F } )$ is watertight by construction, requires no normal estimation or surface-fitting parameters, and encloses the ear tightly while remaining robust to surface irregularities.

Solid interior and skeleton extraction (Panel C).. The convex-hull mesh is voxelized on a regular grid with spacing $\Delta = 3$ mm. For each grid point, occupancy is determined by ray-casting (threshold > 0 5), yielding a binary solid array $s \subset \mathbb { Z } ^ { 3 }$ . A Euclidean distance transform [29] is computed over S, giving the distance from every interior voxel to the nearest surface. For each slice along the longitudinal axis (identified as the axis of greatest bounding-box extent), voxels whose distance-transform value exceeds 0 75 of the slice maximum are retained as the medial core, and their centroid is computed with weights proportional to the squared distance value. The resulting centerline is smoothed by fitting a cubic polynomial in each transverse dimension as a function of the longitudinal coordinate, and near-duplicate consecutive points (separation < 0 25 ∆) are removed. The arc length of the smoothed centerline defines the skeleton length $L _ { \mathrm { s k e l } }$

Global geometry (Panel D).. The mesh is re-oriented so that the longitudinal axis coincides with z via principal component analysis. The skeleton arc length $L _ { \mathrm { s k e l } }$ serves as the primary measure of ear length. The straight-line distance between the skeleton endpoints defines the chord length $L _ { \mathrm { c h o r d } }$ . Surface area, A, is the sum of triangle areas over all faces of the convex hull. Volume, V, is obtained from the signed tetrahedral decomposition

$$
V = \frac { 1 } { 6 } \sum _ { ( i , j , k ) \in \mathcal { F } } \mathrm { d e t } \{ \mathbf { v } _ { i } , \mathbf { v } _ { j } , \mathbf { v } _ { k } \} ,\tag{5}
$$

applied to the convex-hull mesh, which is guaranteed watertight. The maximum major diameter across all ten crosssections (Section 2.7) defines the max width, and its relative position along the skeleton is also recorded.

Cross-sectional shape (Panels E–F).. The skeleton is divided into $K = 1 0$ equal-length sections. At each section t, mesh vertices within max $( 2 . 5 \% \times L _ { \mathrm { s k e l } } , 2$ mm) of the section plane are projected onto the plane perpendicular to the local skeleton tangent. A two-dimensional convex hull is computed from the projected points, and a direct leastsquares ellipse is fitted to its boundary vertices [30]. If the algebraic fit fails, a covariance-based fallback estimates semi-axes as $a _ { t } = 2 \sqrt { \lambda _ { 1 } ^ { ( t ) } }$ and $b _ { t } = 2 \sqrt { \lambda _ { 2 } ^ { ( t ) } }$ from the eigenvalues $\lambda _ { 1 } ^ { ( t ) } \geq \lambda _ { 2 } ^ { ( t ) }$ of the $2 \times 2$ scatter matrix.

Three section-wise quantities are reported: the major and minor diameters $D _ { \mathrm { m a j } } ^ { ( t ) } = 2 a _ { t }$ and $D _ { \operatorname* { m i n } } ^ { ( t ) } = 2 b _ { t }$ ; the crosssectional eccentricity

$$
e _ { t } = \sqrt { 1 - \left( \frac { b _ { t } } { a _ { t } } \right) ^ { 2 } } ,\tag{6}
$$

which equals zero for a circular section and approaches unity for a highly elongated one; and the cross-sectional area obtained from the two-dimensional convex hull.

Derived shape descriptors.. We use five shape descriptors to summarize ear morphology at the whole-ear level. The Chord/Arc Ratio $L _ { \mathrm { c h o r d } } / L _ { \mathrm { s k e l } }$ quantifies straightness and equals unity for a perfectly straight ear. The Taper to Tip index

$$
\mathrm { T a p } = { \frac { A _ { K } } { \operatorname* { m a x } _ { t } A _ { t } } }\tag{7}
$$

expresses the cross-sectional area at the distal (tip) section relative to the maximum section area, with lower values indicating greater tapering. The Curvature Score

$$
\mathrm { C u r } = \frac { \displaystyle \sum _ { t = 1 } ^ { K - 1 } \Delta \theta _ { t } } { \displaystyle L _ { \mathrm { s k e l } } }\tag{8}
$$

accumulates the turning angle $\Delta \theta _ { t }$ between consecutive skeleton tangent vectors and normalises by arc length; higher values indicate a more curved ear. The Max Bending is the peak local curvature along the skeleton,

$$
K _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { t } { \frac { \Delta \theta _ { t } } { \Delta s _ { t } } } ,\tag{9}
$$

where $\Delta s _ { t }$ is the segment length between consecutive skeleton points; it captures the single sharpest bend. Finally, Mean Eccentricity $\bar { e } = K ^ { - 1 } \sum _ { t = 1 } ^ { \bar { K } } e _ { t }$ summarizes the average departure from circularity across all ten sections.

All traits (global summaries and per-section values) are written to a single CSV file per ear. The full extraction pipeline is wrapped in a single command. Most trait vocabulary is inspired by the conceptual framework established by EarCV [3] and EARBOX [4]; the principal contribution lies in the integration, automation, and reproducibility packaging. Two traits warrant specific framing. First, the Chord/Arc Ratio is a formally defined 3-D shape descriptor with no direct analogue in prior ear-phenotyping tools: 2-D imaging records only the straight-line projection of the ear axis, making the geodesic arc (and therefore any arc-to-chord comparison) inaccessible; the metric is the reciprocal of the tortuosity index used in vascular biology and movement ecology [31, 32], here applied for the first time to quantify whole-ear axial straightness from a watertight mesh skeleton. Second, Taper-to-Tip is retained as a geometric descriptor of distal ear cross-sectional area but is explicitly interpreted as a shape proxy rather than a direct measure of kernel fill: without kernel-level segmentation of the mesh, it cannot distinguish natural geometric taper from tip kernel abortion, unlike the pixel-based fill scores of 2-D tools such as EarCV.

![](images/494ee1011fd14b5e1d6b7e5d26b274c67472cc3cd73a55ecce405c6ea4961bdc.jpg)  
Figure 6: Example 3-D reconstructions of maize ears generated by the proposed workflow. The method robustly recovers ear geometry across substantial variations in size, shape, curvature, and kernel color. Each ear is shown together with its automatically estimated bounding box, which defines the spatial extent used for subsequent trait extraction and comparative analysis. The diversity of reconstructed ears illustrates that the pipeline generalizes well to heterogeneous phenotypes and captures both morphological and color variation in a consistent 3-D representation.

Table 4: COLMAP success statistics (multi-seed initialization).
<table><tr><td></td><td>Number of videos 100 % registration Re-capture needed</td><td></td></tr><tr><td>300</td><td>285 (95 %)</td><td>15 (5 %)</td></tr></table>

## 3. Results

## 3.1. COLMAP registration and geometry quality control

Table 4 summarizes the pose-estimation of 300 processed ears. Of the ten independent COLMAP runs launched per ear, at least one achieved 100 % frame registration for 285 ears (95 %); the remaining 15 ears (5 %) required re-capture because none of the ten runs produced a fully registered reconstruction. Figure 7 illustrates the run-torun variability in camera pose trajectories across three representative ears. The three examples span the range of outcomes: a case where all ten seeds converge to consistent trajectories, a case with partial registration on some seeds, and a case with inconsistent trajectories that caused the ear to be flagged for re-capture. The multi-seed strategy is efective precisely because individual seeds can fail on challenging ears while other seeds succeed; retaining only 100 %-coverage reconstructions and selecting the lowest-reprojection-error run provides a stable pose set for NeRF training.

After registration, the 285 ears were passed through the automated geometry quality-control step (Section 2.6).

In total, 35 ears were excluded at this stage. After applying the geometry quality-control filter, 250 / 300 ears (83.3 %) passed to trait extraction. A set of excluded cases are shown in the Supplementary Figure 14.

![](images/2966eeb9ef735f63fceb07a7c3bc1cc45b0ebc62c3db43f3fb81513a12db8a9f.jpg)  
Sample 1 (1/9 successful)

![](images/6a93256a2e1b1ee47ea7d3b0be4c08af43ca23ddc3cbdc99a192ff5d47f588f4.jpg)  
Sample 2 (5/9 successful)

![](images/039eade1ca9efe4e951cee2584f3a9f317dd0919a8b1ba5a584c747b47bd61bd.jpg)  
Sample 3 (4/9 successful)  
Figure 7: Run-to-run variability in camera pose trajectories (A) 100% seed convergence to consistent trajectories (B) Partial registration on some seeds (C) Inconsistent trajectories; ear flagged for re-capture

A  
![](images/cbf79771c78873ba3f51d0455808ff9b8410481fe88ae3b7a72df5e56ba90b99.jpg)

![](images/73ba343a6e6e05953144957827930dd422a5a031fadcc96a1691006b4f83660b.jpg)

![](images/5e1b2174b39b00745b9becd69d1f3b9ae95d95891a94484450f14b06403ddbcc.jpg)  
Figure 8: (A) Automated skeleton length plotted against manual caliper measurement for all 250 ears $( R ^ { 2 } = 0 . 9 6 4 , { \mathrm { R M S E } } = 4 . 6 8 { \mathrm { m m } }$ ). Skeleton length traces the geodesic arc along the curved ear axis, which exceeds the straight-line chord measured by calipers for bent ears; the systematic divergence with curvature is quantified in Figure 9. (B) Photograph of the water-displacement apparatus. Each ear was submerged in a container filled to the brim and held flush with the opening by an acrylic plate; displaced water was collected and read to the nearest 1 mL in a graduated 100 mL cylinder. (C) Convex-hull volume derived from the reconstructed point cloud plotted against manually measured displacement volume $( R ^ { 2 } = 0 . \dot { 9 } 8 2 , \mathrm { R M S E } = 5 . 2 6 \mathrm { m L } )$ . The regression line lies slightly above the 1:1 reference (dotted), reflecting a small positive bias attributable to the convex hull enclosing empty space within the concavity of curved ears. In panels (A) and $( \mathbf { C } ) ,$ the solid line is the ordinary least-squares fit, the shaded band the 95 % confidence interval of the mean, and the dotted line the 1:1 reference; $R ^ { 2 }$ and RMSE are inset.

A  
![](images/d3912ab233ebbe39bbaaf5cb22bb3719d7d37c8d197d35f03148bcf435c6e15b.jpg)

B  
![](images/f0809e25c866f004647036f7f967bf390c1e008ef588f3c0f25ef40733dfdf24.jpg)  
Figure 9: (A) Absolute deviation of skeleton length from manual caliper measurement $( \Delta _ { \mathrm { s k e l } } .$ mm) plotted against Chord/Arc ratio $\mathrm { ( C / A ) }$ for all 250 ears $( r = - 0 . 5 4 , p < 0 . 0 0 1 )$ . Ears with lower C/A (indicating greater longitudinal curvature) show systematically larger disagreement between skeleton length and caliper measurement. (B) Absolute deviation of bounding-box height from manual caliper measurement $( \Delta _ { \mathrm { b b o x } } ,$ mm) plotted against the same $\mathrm { C } / \mathrm { A }$ axis $( r = - 0 . 0 5$ , ns). The regression slope is near zero, confirming that bounding-box height error is insensitive to ear curvature. The contrasting responses in panels (A) and (B) establish that the caliper–skeleton discrepancy arises from a diference in measurement definition: bounding-box height and manual calipers both record the straight-line chord from base to tip, whereas skeleton length traces the geodesic arc, which necessarily exceeds the chord as curvature increases. Significance codes: $^ { * * * } p < 0 . 0 0 1 ;$ ; ns $p \ge 0 . 0 5$ . ∆ = |skeleton length − manual|; ∆ = |bounding-box height − manual|.

## 3.2. Diverse Maize Ears (Qualitative Examples)

Figure 6 shows 3-D reconstructions of a curated set of diverse maize ears (diferent from the 300 ears used for quantitative experiments). The pipeline recovers ear geometry across substantial variation in size, shape, curvature, and kernel color. Each ear is shown with its automatically estimated axis-aligned bounding box. The reconstructions illustrate that the pipeline generalizes to heterogeneous phenotypes without manual parameter tuning: short and long ears, straight and strongly curved ears, and ears ranging from yellow to red-purple kernel pigmentation are all successfully reconstructed. These qualitative results support the quantitative validation in the following section.

## 3.3. Geometric validation

Figure 8 summarizes geometric validation of the pipeline against two independent manual references.

Skeleton length. Automated skeleton length was compared against manual caliper measurements for all $n = 2 5 0$ ears (Figure 8(A)), yielding $R ^ { 2 } = 0 . 9 6 4 ~ { \mathrm { a n d ~ R M S E } } = 4 . 6 8 { \mathrm { m m } }$ , of the same order as the RMSE = 4 4 mm reported by Oury et al. [4], although $R ^ { 2 }$ values are not directly comparable across panels with diferent trait ranges. The residual error reflects two contributions: natural caliper measurement variability and a fundamental diference in what each method measures. Manual calipers record the straight-line chord from base to tip; skeleton length traces the geodesic arc along the curved ear axis, which exceeds the chord for bent ears. Figure 9 quantifies this distinction. Panel (A) shows a strong negative relationship between $\Delta _ { \mathrm { s k e l } }$ and Chord/Arc ratio $( r = - 0 . 5 4 , p < 0 . 0 0 1 )$ : ears with lower $\mathrm { C } / \mathrm { A }$ (indicating greater curvature) exhibit systematically larger skeleton-length errors relative to the caliper reference. Panel (B) shows that bounding-box height error $\left( \Delta _ { \mathrm { b b o x } } \right)$ is insensitive to $\mathrm { C } / \mathrm { A } \left( r = - 0 . 0 5 , \mathrm { n s } \right)$ , with the regression slope near zero across the full curvature range. The divergent responses confirm that the caliper–skeleton discrepancy is a measurement-definition artifact rather than a reconstruction error: bounding-box height and manual calipers both measure the chord, so they agree regardless of ear shape, whereas skeleton length measures the arc and necessarily diverges from the chord as curvature increases.

Convex-hull volume. A subset of $n = 1 5$ ears spanning the full size range was measured by water displacement. Each ear was submerged in a container filled to the brim and held flush with the opening by an acrylic plate (Figure 8(B)); displaced water was collected and read to the nearest 1 mL in a graduated 100 mL cylinder. This method provides a direct physical volume reference independent of any geometric assumption. Convex-hull volume agrees closely with displacement volume (Figure $8 ( \mathbf { C } ) ; R ^ { 2 } = 0 . 9 8 2 , \mathrm { R M S E } = 5 . 2 6 \mathrm { m L } )$ , though the regression line lies slightly above the 1:1 reference, reflecting a small positive bias: the convex hull encloses empty space within the concavity of curved ears, contributing to hull volume but not to displacement. This is an inherent limitation of the convex-hull estimator rather than a reconstruction error, consistent with the curvature-driven discrepancies previously identified for skeleton length measurements. Because this bias stems from the convexity of the hull itself rather than from imprecision in integrating its volume, eliminating it would require replacing the convex hull with a non-convex, concavity-preserving surface reconstruction, which is planned for future work.

## 3.4. Population-level trait distributions

Figures 10-11 illustrate the phenotypic diversity captured by the pipeline through pairs of morphologically contrasting ears drawn from the size and shape trait spaces, respectively. Each pair is accompanied by a radar chart overlaying their normalized profiles across a subset of that group (five of the six size traits, four of the five shape traits; bounding-box height and Max Bending are omitted for legibility), revealing how variation in one trait co-varies with the remaining descriptors.

For size traits (Figure 10), a large, well-filled ear and a small, sparse ear (panels A–B) difer markedly across skeleton length, volume, and surface area simultaneously, consistent with overall ear size being the dominant axis of co-variation within this group. A wide ear and a narrow ear (panels C–D) show a more diferentiated profile: the wide ear scores broadly across all size traits, while the narrow ear retains moderate skeleton length, indicating that width and length can vary semi-independently across the diversity panel.

For shape traits (Figure 11), a straight ear and a curved ear (panels A–B) diverge primarily along the C/A axis while sharing comparable Tap and Ecc values, confirming that curvature is captured as an independent morphological dimension. A cylindrical ear with sustained distal width and a strongly tapered ear (panels C–D) illustrate the full range of the Taper-to-Tip trait.

Figure 12 shows the population-level distributions of seven of the eleven whole-ear traits as violin plots $( n = 2 5 0 ) ;$ the remaining four are reported in the per-ear CSV output. All traits show substantial spread across the diversity panel. Skeleton length ranges from approximately 75 to 225 mm with a roughly symmetric distribution, while maximum width is right-skewed, with most ears narrow and displaying a long upper tail. Chord/Arc ratio is tightly clustered near unity (0.965–1.000), indicating that the majority of ears are nearly straight, with curvature representing a minority phenotype. Taper-to-Tip and Curvature Score are each right-skewed, reflecting that strong tapering and high curvature are less common within this panel. Mean Eccentricity is broadly and approximately symmetrically distributed around 0.45, indicating that elliptic cross-sections are prevalent throughout the population.

Figure 13 provides further characterization of cross-sectional geometry and its association with ear size. Panels (A) and (B) show mean eccentricity and major-axis diameter profiles from base to tip, stratified by ear-length quartile. Eccentricity profiles are largely consistent across quartiles, with a shared mid-ear peak near section 5 and convergence toward the tip, suggesting that cross-sectional shape is conserved regardless of overall ear length. Major-axis diameter profiles show a clear quartile ordering throughout, confirming that longer ears are proportionally wider. Panels (C) and (D) show the association between convex-hull surface area and convex-hull volume against manually measured ear weight, yielding $R ^ { 2 } \ = \ 0 . 6 7 5 \ ( \mathrm { R M S E } \ = \ 1 8 . 7 0 \mathrm { c m } ^ { 2 } )$ and $R ^ { 2 } \ = \ 0 . 7 3 8 \ ( \mathrm { R M S E } \ = \ 1 7 . 4 1 \mathrm { c m } ^ { 3 } )$ , respectively. The moderate associations reflect a general size efect, with residual variance attributable to diferences in kernel moisture and maturity state at the time of weighing rather than geometric inaccuracy.

## 3.5. Human-efort comparison

Table 5 contrasts operator time for the conventional manual workflow with the proposed automated pipeline. Manual phenotyping (comprising caliper measurements of ear length, visual scoring, and data entry) requires approximately five minutes per ear. The automated pipeline reduces hands-on involvement to approximately one minute per ear (mounting the sample and initiating video capture); all downstream steps, including structure-from-motion reconstruction, point-cloud extraction, meshing, skeletonization, and trait computation, run unattended on a GPU node. Crucially, the pipeline simultaneously delivers traits (skeleton length, convex-hull volume, surface area, cross sectional profiles, and five shape descriptors) that are inaccessible to manual caliper measurement.

![](images/b8e299001a23a99979c76734c5a68e5dbe334d10c8f7d5d51e6ab70496bc1210.jpg)  
Figure 10: Size-trait variation across the 3D maize ear population illustrated by contrasting ear pairs. (A) RGB photographs of a large, well-filled ear (orange bounding box) and a small, sparse ear (blue bounding box), selected to contrast in volume. (B) Radar chart overlaying the normalised trait profiles of the two ears in panel (A) across five size descriptors (Skeleton Length, Volume, Max Width, Max Width Position, Convex-Hull Surface Area; axes normalised to the population range). The large ear scores broadly across all size traits; the small ear is uniformly compact. (C) RGB photographs of a wide ear (yellow bounding box) and a narrow ear (green bounding box), selected to contrast in maximum width. (D) Radar chart for the ear pair in panel (C), showing that maximum width co-varies with volume and surface area but not proportionally with skeleton length, indicating semi-independent variation in width and length across the panel. Ear photographs are not shown at a common scale; each image preserves the aspect ratio of the individual ear.

Table 5: Human-efort comparison (per ear).
<table><tr><td>Task</td><td>Manual workflow Proposed pipeline</td><td></td></tr><tr><td>Total operator time</td><td>≈5 min</td><td>≈1 min</td></tr></table>

Taken together, the validation results, population-level trait distributions, and biological associations confirm that the low-cost, fully automated pipeline delivers geometrically accurate and biologically relevant 3-D phenotypes across a diverse maize panel, while reducing operator involvement by approximately 80 % relative to conventional manual measurements.

## 4. Discussion

## 4.1. Practical impact

The system presented here meets three practical requirements for deployment in a breeding program: (i) a low cost hardware platform (\$607 total; Table 2), (ii) an approximately 80 % reduction in operator time relative to manual caliper measurement (Table 5), and (iii) measurements validated against independent manual references (Section 3.3). The imaging rig comprises a commodity DSLR, a motorized turntable, two LED panels, and a 3D-printed cylindrical sample holder, all components obtainable from standard laboratory suppliers. Following a 20-second video capture, the workflow executes autonomously on a GPU node, freeing the operator to prepare the next sample. The pipeline may be applicable to medium-scale breeding programs that require phenotypic characterization of hundreds of ears per season, though throughput at larger scales and performance across diverse environments remain to be evaluated.

## 4.2. Design advantages

Several technical choices contribute to the reliability of the approach.

![](images/20cfbd7e51f3ae9adcddfb8ce40b64b27852b671e05b383c83a8f8f62af79d7b.jpg)  
Figure 11: Shape-trait variation across the 3D maize ear population illustrated by contrasting ear pairs. (A) RGB photographs of a straight ear (orange bounding box; C/A ≈ 1) and a curved ear (blue bounding box), selected to contrast in Chord/Arc ratio. (B) Radar chart overlaying the normalised shape profiles of the two ears in panel (A) across four shape descriptors (C/A, Tap, Cur, Ecc; axes normalised to the population range). The two ears diverge on C/A while sharing comparable values for the remaining traits, confirming that curvature is an independent morphologica dimension. (C) RGB photographs of a cylindrical ear with sustained distal width (yellow bounding box) and a strongly tapered ear (green bounding box), selected to contrast in Taper-to-Tip. (D) Radar chart for the ear pair in panel (C), illustrating that Taper-to-Tip varies semi-independently of the other shape descriptors. Ear photographs are not shown at a common scale; each image preserves the aspect ratio of the individual ear. Trait abbreviations: C/A: Chord/Arc Ratio (= 1 for a perfectly straight ear); Tap: Taper-to-Tip (lower = greater tapering); Cur: Curvature Score (higher = more curved); Ecc: Mean Cross-sectional Eccentricity (higher = more elliptic cross-section).

Fixed-camera, rotating object.. A stationary camera reduces the need for multi-viewpoint synchronization and simplifies mechanical calibration. Constraining object motion to a single rotation axis limits the camera pose search space, which may improve reconstruction consistency across captures, though this has not been formally compared against a moving-camera setup.

Multi-seed COLMAP robustness.. Running COLMAP ten times with diferent random seeds (Section 2.3.1) and retaining only reconstructions with 100 % frame registration provides a safeguard against occasional feature-matching failures caused by low-texture or specular kernels. Averaging intrinsic parameters across successful runs reduces sensitivity to any single reconstruction and provides a more stable initialisation for subsequent NeRF training.

What the 3D mesh enables.. Bounding-box height is computable from any point cloud and agrees closely with manual caliper measurements because both measure the straight-line chord from base to tip. Skeleton length, by contrast, traces the geodesic arc along the curved ear axis and diverges from the caliper reference predictably with curvature (Figure 9); computing it requires skeletonization of a closed surface. The watertight mesh further enables (i) volume estimation via the signed tetrahedral formula, (ii) total surface area from triangle sums, and (iii) cross-sectional shape profiles that are less sensitive to the surface holes that can afect point-cloud slicing. These traits are not straightforwardly recoverable from a single 2D image, though direct comparison with alternative 3D acquisition methods is outside the scope of this work. Validation against water-displacement volume yields $R ^ { 2 } = 0 . 9 8 2$ and RMSE = 5 26 mL (Figure 8(C)), with the largest deviations occurring for curved ears where the convex hull overestimates the true surface-bounded volume (Section 3.3).

## 4.3. Biological relevance

Convex-hull volume and convex-hull surface area show moderate associations with manually measured ear weight $( R ^ { 2 } = 0 . 7 3 8$ and 0 675, respectively; Figure 13, panels C–D), confirming that the 3-D reconstruction captures biologically meaningful size variation. The slightly stronger association of volume relative to surface area is consistent with ear weight scaling more directly with a volumetric than a surface dimension. The residual variance reflects compounding sources: biologically, ear weight at harvest is sensitive to kernel moisture content, maturity state, and cob-to-kernel ratio diferences across genotypes, none of which are recoverable from surface geometry alone; geometrically, convex-hull volume and convex-hull surface area are approximations: the convex hull overestimates the true volume for curved ears by enclosing empty space, as established in Section 3.3, and the current mesh resolution does not resolve individual kernel geometry. Replacing the convex hull with a non-convex, concavity-preserving surface reconstruction would reduce the geometric component of this residual and is planned for future work. The shape descriptors (Curvature and Taper-to-Tip) are conceptually related to kernel distribution and grain-filling traits reported in the 2-D phenotyping literature [2, 4]; whether the 3-D versions of these indices provide predictive value beyond 2-D measurements is an important open question for future validation.

![](images/ab6eab3a417718258711caf4359be00a61c82f7e5907b82fcf6bec6635a9f8d5.jpg)  
Figure 12: Population-level distributions of seven of the eleven whole-ear morphometric traits (n = 250). Violin plots for skeleton length, maximum width, maximum width position, Chord/Arc ratio, Taper-to-Tip, Curvature Score, and mean cross-sectional eccentricity. The horizontal bar and crosshair inside each violin mark the median and interquartile range, respectively. Chord/Arc ratio is tightly clustered near unity, indicating that most ears are nearly straight; Taper-to-Tip and Curvature Score are right-skewed, reflecting that strong tapering and high curvature are minority phenotypes within this panel. All remaining traits show broad, continuous variation, confirming that the pipeline discriminates meaningfully across the diversity panel.

## 4.4. Limitations

Four constraints bound the results reported here. First, 50 of 300 ears did not reach trait extraction, and 35 of those were lost at holder segmentation rather than at reconstruction, which identifies metric scaling as the least robust stage of the pipeline. Flagged ears can be re-imaged rather than discarded, since a re-capture costs only a 20-second video and approximately one minute of operator time, so the efective loss in routine use is smaller than the 16.7 % first-pass figure suggests. We have not characterized the excluded ears against the retained ones, so we cannot yet rule out that attrition is correlated with ear geometry.

Second, the convex hull overestimates the volume of curved ears because it spans the concave flank, which sets a floor on the agreement achievable against water displacement. A concavity-preserving surface reconstruction would remove this bias and is the most direct route to improving volumetric accuracy.

Third, the automated quality-control gate is supplemented by visual inspection for one failure mode that the holder-eccentricity statistic does not detect, so the pipeline is automated with a single human checkpoint rather than fully unattended.

Fourth, all ears come from one diversity panel in a single season, imaged by five operators across sessions using two turntable models of the same specification. We did not isolate operator or session efects, and the Chord/Arc ratio spans only 0.965 to 1.000 in this panel, so the curvature descriptors are demonstrated over a narrow range and warrant evaluation on germplasm with more pronounced ear curvature.

## 4.5. Future Directions

Several directions are planned to address these limitations and to extend the scope of measurable traits.

COLMAP-free pose estimation via turntable geometry. Because the ear rotates on a motorized turntable at a known, approximately constant angular velocity, the rotation angle between successive frames provides a geometric prior on camera-pose increments. An approach under development uses per-frame rotation angles inferred from the turntable speed and frame timestamps to initialize camera poses, which are then refined jointly with the NeRF network weights through a diferentiable camera-parameter optimiser in the Nerfstudio training loop. If successful, this would reduce reliance on COLMAP and potentially shorten pre-processing time, though performance relative to the current pipeline has not yet been formally evaluated. Preliminary results on a held-out subset are encouraging and will be reported in a follow-up study.

![](images/68040134cca00a6de0c56212fda2d6b28754c352df2995c787c5362ba512f3ac.jpg)

B  
![](images/9f3bf8239753f4b7f43322103375354c40c86e065028b6e7becaf0efca22ff00.jpg)

C  
![](images/8a6cfa4ec84e60323ad0108a4756e34a92c2c61ef509eb10d07b171aba0f102f.jpg)

D  
![](images/c68dbaa032ff4b50027d60604fdaf536ad741f3bba03b399eec0eb6067bd5bd2.jpg)  
Figure 13: Cross-sectional profiles and biological associations of 3-D morphometric traits. (A) Mean cross-sectional eccentricity plotted from base to tip (sections 1–10) stratified by ear-length quartile (Q1: short; Q4: long). Profiles are consistent across quartiles, with a shared mid-ear peak near section 5 and convergence toward the tip, indicating that cross-sectional shape is conserved across ear sizes. (B) Mean major-axis diameter (mm) along the same ten sections, stratified by quartile. Longer ears show larger diameters throughout, confirming that ear length and girth co-vary within this panel. (C) Convex-hull surface area plotted against manually measured ear weight $( \breve { R ^ { 2 } } = 0 . 6 7 5 , { \mathrm { R M S E } } = 1 8 . 7 0 { \mathrm { c m } } ^ { 2 } )$ . (D) Convex-hull volume plotted against ear weight $( R ^ { 2 } = 0 . 7 3 8 , \mathbf { \dot { R } M S E } = 1 7 . 4 1 { \mathrm { c m } } ^ { 3 } )$ . The moderate associations in panels (C) and (D) reflect a general size efect; residual variance is attributable to diferences in kernel moisture and maturity state at the time of weighing rather than geometric inaccuracy. In all profile panels, shaded bands show the 95 % confidence interval of the mean.

Kernel-level trait extraction. The trait set reported here is intentionally limited to ear-level descriptors; a dedicated pipeline for kernel-level trait extraction (including kernel row number, kernels per row, and individual kernel dimensions) builds directly on the point cloud produced by this platform in a companion study [6], which jointly extracts eleven ear- and kernel-level traits by cylindrically unwrapping the calibrated point cloud and applying zero-shot instance segmentation, without requiring crop-specific model training. The dense color point cloud produced by the NeRF reconstruction provides a natural foundation for this extension, as full $3 6 0 ^ { \circ }$ surface color and geometry are available for localizing individual kernels without the occlusion constraints inherent to single-view approaches.

## 5. Conclusions

We have described a low-cost phenotyping pipeline that produces 3-D reconstructions of maize ears from a single 20-second video captured by a fixed consumer-grade DSLR. Combining a multi-seed COLMAP pose-estimation stage, a standard nerfacto NeRF model, and a cylindrical-fiducial scaling procedure, the pipeline extracts skeleton length $( R ^ { 2 } = 0 . 9 6 4 , \mathrm { R M S E } = 4 . 6 8$ mm against manual calipers), convex-hull volume $( R ^ { 2 } = 0 . 9 8 2 , \mathrm { R M S E } = 5 . 2 6$ mL against water-displacement), and a suite of shape descriptors from the reconstructed watertight mesh. Applied to 300 ears from the SAM diversity panel, 250 (83.3 %) passed automated processing and quality control. Of the 50 excluded ears, 15 failed at the COLMAP registration stage and 35 were excluded at the holder-segmentation qualitycontrol stage, so scale recovery rather than pose estimation is the dominant source of attrition. Operator involvement is approximately one minute per ear; all downstream processing runs unattended. Convex-hull volume and convex-hull surface area show moderate associations with manually measured ear weight $( R ^ { 2 } = 0 . 7 3 8$ and 0 675, respectively), with residual variance attributable to both biological factors (kernel moisture content and maturity state) and the geometric limitation of the convex-hull estimator for curved ears. The trait set reported here is deliberately ear-level; a companion study [6] extends this platform to eleven ear- and kernel-level traits. COLMAP-free pose estimation via turntable-rotation priors is also under evaluation, with performance relative to the current pipeline yet to be formally established.

## CRediT authorship contribution statement

Therin Young: Software, Validation, Formal analysis, Investigation, Writing – original draft, Writing – review & editing, Visualization. Elijah Rodriguez: Methodology, Software, Validation, Visualization, Writing – review & editing. Lisa Cofey: Data curation, Writing – review & editing. Talukder Zaki Jubery: Methodology, Software, Validation, Formal analysis, Investigation, Writing – original draft, Writing – review & editing, Visualization. Adarsh Krishnamurthy: Conceptualization, Supervision, Funding acquisition, Writing – review & editing. Patrick Schnable: Conceptualization, Resources, Supervision, Project administration, Funding acquisition, Writing – review & editing. Baskar Ganapathysubramanian: Conceptualization, Resources, Supervision, Project administration, Funding acquisition, Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This work was supported by the AI Institute for Resilient Agriculture (USDA-NIFA 2021-67021-35329) and Iowa State University’s Plant Science Institute. The funding sources had no involvement in study design; in the collection, analysis and interpretation of data; in the writing of the report; or in the decision to submit the article for publication.

## Acknowledgments

We thank a team of undergraduate assistants who imaged ears and generated ground truth data. Team members included: Owen Hildebrandt, Roland Stetz, Isaac Tharp, Haylee Thomas, and Ryan Wolf.

## Data availability

The source code for the reconstruction and trait-extraction pipeline is publicly available at <REPOSITORY-URL> (archived at <CODE-DOI>). The per-ear morphological trait tables and the manual caliper, water-displacement, and ear-weight reference measurements supporting the findings of this study are deposited at <DATADOI>. The raw turntable videos (approximately <N> GB for 300 ears) are available from the corresponding authors on reasonable request owing to their size.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used Claude to assist with language editing and organization. After using these tools, the authors reviewed and edited the content and take full responsibility for the final text.

## References

[1] R. Makanza, M. Zaman-Allah, J. E. Cairns, J. Eyre, J. Burgueño, Á. Pacheco, C. Diepenbrock, C. Magorokosho, A. Tarekegne, M. Olsen, B. M. Prasanna, High-throughput method for ear phenotyping and kernel weight estimation in maize using ear digital imaging, Plant Methods 14 (2018) 49. doi:10.1186/s13007-018-0317-4.

[2] N. D. Miller, N. J. Haase, J. Lee, S. M. Kaeppler, N. de Leon, E. P. Spalding, A robust, high-throughput method for computing maize ear, cob, and kernel attributes automatically from images, The Plant Journal 89 (2017) 169–178. doi:10.1111/tpj.13320.

[3] J. M. Gonzalez, N. Ghosh, V. Colantonio, F. de Cássia Pereira, R. A. Pinto, C. Wasson, K. A. Leach, M. F. R. Resende, EarCV: An open-source, computer vision package for maize ear phenotyping, The Plant Phenome Journal 5 (2022) e20055. doi:10.1002/ppj2.20055.

[4] V. Oury, T. Leroux, O. Turc, R. Chapuis, C. Palafre, F. Tardieu, S. A. Prado, C. Welcker, S. Lacube, Earbox, an open tool for high-throughput measurement of the spatial organization of maize ears and inference of novel traits, Plant Methods 18 (2022) 96. doi:10.1186/s13007-022-00925-8.

[5] S. Fan, G. Li, R. Bahitwa, Z. Jia, H. Zhang, J. Shao, Q. Yu, X. Chen, Y. Qian, M. Xu, L. Zhu, H. Wang, OpenEar: An ultra-afordable, high-throughput, and accurate maize ear phenotyping system, Plant Methods 22 (2026) 26. doi:10 1186/ 13007 026 01 04 .

[6] R. A. Kumar, S. Tripathi, P. Matthews, S. Reddy, T. Z. Jubery, P. S. Schnable, A. Krishnamurthy, B. Ganapathysubramanian, Automated maize ear phenotyping using 3D reconstructions, The Plant Phenome Journal (2026). Manuscript in preparation.

[7] J. L. Schönberger, J.-M. Frahm, Structure-from-motion revisited, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4104–4113. doi:10.1109/CVPR.2016.445.

[8] J. L. Schönberger, E. Zheng, J.-M. Frahm, M. Pollefeys, Pixelwise view selection for unstructured multi-view stereo, in: Computer Vision – ECCV 2016, volume 9907 of Lecture Notes in Computer Science, Springer, Cham, 2016, pp. 501–518. doi:10.1007/978-3-319-46487-9\_31.

[9] Y. Wang, W. Wen, S. Wu, C. Wang, Z. Yu, X. Guo, C. Zhao, Maize plant phenotyping: Comparing 3D laser scanning, multi-view stereo reconstruction, and 3D digitizing estimates, Remote Sensing 11 (2019) 63. doi:10. 3390/rs11010063.

[10] M. J. Feldmann, A. Tabb, Cost-efective, high-throughput phenotyping system for 3D reconstruction of fruit form, The Plant Phenome Journal 5 (2022) e20029. doi:10.1002/ppj2.20029.

[11] R. Qiu, Y. Kang, B. Ye, H. Sun, C. Xu, Sensors for measuring plant phenotyping: A review, International Journal of Agricultural and Biological Engineering 11 (2018) 1–17. doi:10.25165/j.ijabe.20181102.2696.

[12] R. Yang, Y. He, X. Lu, Y. Zhao, Y. Li, Y. Yang, W. Kong, F. Liu, 3D-based precise evaluation pipeline fo maize ear rot using multi-view stereo reconstruction and point cloud semantic segmentation, Computers and Electronics in Agriculture 216 (2024) 108512. doi:10.1016/j.compag.2023.108512.

[13] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, NeRF: Representing scenes as neural radiance fields for view synthesis, Communications of the ACM 65 (2022) 99–106. doi:10.1145/ 3503250.

[14] T. Müller, A. Evans, C. Schied, A. Keller, Instant neural graphics primitives with a multiresolution hash encoding, in: ACM SIGGRAPH 2022 Conference Proceedings, 2022. doi:10.1145/3528223.3530127.

[15] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, J. Kerr, T. Wang, A. Kristofersen, J. Austin, K. Salahi, A. Ahuja, D. McAllister, A. Kanazawa, Nerfstudio: A modular framework for neural radiance field development, SIG GRAPH 2023 Posters (2023). doi:10.1145/3588432.3591516.

[16] M. A. Arshad, T. Jubery, J. Aful, A. Jignasu, A. Balu, B. Ganapathysubramanian, S. Sarkar, A. Krishnamurthy, Evaluating neural radiance fields for 3D plant geometry reconstruction in field conditions, Plant Phenomics 6 (2024) 0235. doi:10.34133/plantphenomics.0235.

[17] K. Hu, W. Ying, Y. Pan, H. Kang, C. Chen, High-fidelity 3D reconstruction of plants using neural radiance fields, Computers and Electronics in Agriculture 220 (2024) 108848. doi:10.1016/j.compag.2024.108848.

[18] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, 3D gaussian splatting for real-time radiance field rendering, ACM Transactions on Graphics (SIGGRAPH) 42 (2023) 1–14. doi:10.1145/3592433.

[19] X. Sun, T. Huang, Z. Niu, C. Yang, Y. He, Z. Qiu, MEP3D: Improved clustering-based 3D point cloud method for comprehensive maize ear phenotypic trait extraction, Computers and Electronics in Agriculture 240 (2026) 111235. doi:10.1016/j.compag.2025.111235.

[20] C. Zhu, T. Miao, T. Xu, N. Li, H. Deng, Y. Zhou, Skeleton-based segmentation of the maize ear and phenotypic parameter extraction from 3D point clouds of maize plants [in chinese], Transactions of the Chinese Society of Agricultural Engineering 37 (2021) 295–301.

[21] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, R. Girshick, Segment anything, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4015–4026. doi:10.1109/ICCV51070.2023.00371.

[22] H. Zaremehrjerdi, L. Cofey, T. Z. Jubery, H. Liu, J. Turkus, K. Linders, J. C. Schnable, P. S. Schnable, B. Ganapathysubramanian, MaizeEar-SAM: Zero-shot maize ear phenotyping, arXiv preprint arXiv:2502.13399 (2025). doi:10.48550/arXiv.2502.13399

[23] D. G. Lowe, Distinctive image features from scale-invariant keypoints, International Journal of Computer Vision 60 (2004) 91–110. doi:10.1023/B:VISI.0000029664.99615.94.

[24] M. A. Fischler, R. C. Bolles, Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography, Communications of the ACM 24 (1981) 381–395. doi:10.1145/ 358669.358692.

[25] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman, Mip-NeRF 360: Unbounded anti-aliased neural radiance fields, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 5470–5479. doi:10.1109/CVPR52688.2022.00539.

[26] K. Kanatani, Statistical optimization and geometric inference in computer vision, Philosophical Transactions of the Royal Society A 356 (1998) 1303–1320. doi:10.1098/rsta.1998.0223.

[27] M. Ester, H.-P. Kriegel, J. Sander, X. Xu, A density-based algorithm for discovering clusters in large spatial databases with noise, in: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96), AAAI Press, 1996, pp. 226–231.

[28] Q.-Y. Zhou, J. Park, V. Koltun, Open3D: A modern library for 3D data processing, arXiv preprint arXiv:1801.09847 (2018). doi:10.48550/arXiv.1801.09847. arXiv:1801.09847.

[29] C. R. Maurer, R. Qi, V. Raghavan, A linear time algorithm for computing exact euclidean distance transforms of binary images in arbitrary dimensions, IEEE Transactions on Pattern Analysis and Machine Intelligence 25 (2003) 265–270. doi:10.1109/TPAMI.2003.1177156.

[30] A. Fitzgibbon, M. Pilu, R. B. Fisher, Direct least square fitting of ellipses, IEEE Transactions on Pattern Analysis and Machine Intelligence 21 (1999) 476–480. doi:10.1109/34.765658.

[31] W. E. Hart, M. Goldbaum, B. Côté, P. Kube, M. R. Nelson, Measurement and classification of retinal vascular tortuosity, International Journal of Medical Informatics 53 (1999) 239–252. doi:10.1016/S1386-5056(98) 00163-4.

[32] S. Benhamou, How to reliably estimate the tortuosity of an animal’s path: Straightness, sinuosity, or fractal dimension?, Journal of Theoretical Biology 229 (2004) 209–220. doi:10.1016/j.jtbi.2004.03.016.

B

## Supplementary Information

## Geometry quality-control failures

Figure 14 shows the reconstructed holder cross-sections for a representative sample of ears excluded by the geometry quality-control filter described in Section 2.6.

A  
![](images/0e14b00c69c24512e2f33724269d72369d234deb84434fa3198f90f3cb725d70.jpg)

![](images/687eeaab3a8a1775cbe3be98506e771198ed568ad8fb47bd7f5389ea076d8e1c.jpg)

![](images/5e518c696595a48bd6545351e486d40b23301f9ee5035cdef093c27fb02bd2bc.jpg)  
C  
Figure 14: Each column illustrates a distinct failure mode. (A) Partial ear detection: the cutof boundary underestimates the holder–ear junction, causing the upper portion of the ear to be excluded from the segmented region. The holder cross-section is approximately circular but the ear mesh passed to trait extraction is incomplete. This case was identified during manual inspection, as e<sub>holder</sub> alone does not flag it. (B) Contaminated segmentation: the cutof boundary falls within the ear body rather than at the holder rim, so structural parts of the ear are included in the region fitted as the holder. The resulting cross-section is non-circular $( e _ { \mathrm { h o l d e r } }$ elevated above the IQR fence) and the derived scale factor is unreliable. (C) Holder mis-identification: the isolated region does not correspond to the cylindrical holder, yielding a fitted radius ˆr that falls well outside the expected session range. The circle fit produces an erroneous scale factor that would propagate into all physical-unit traits if not caught. Failure modes (B) and (C) are detected automatically by the eccentricity and radius checks respectively $( \mathrm { E q s } . \ 3 . 4 )$ . Failure mode (A) requires manua review and motivates visual inspection of the segmented point cloud as a routine quality-control step. The underlying cause in each case (a flat z-density profile, partial holder occlusion, or an ear positioned too close to the holder rim) is noted below the respective panel.