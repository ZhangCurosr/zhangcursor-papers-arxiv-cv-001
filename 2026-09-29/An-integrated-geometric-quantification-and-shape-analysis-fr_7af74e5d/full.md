# An integrated geometric quantification and shape analysis framework for axillary lymph node metastasis in breast cancer patients<sup>⋆</sup>

Zixi Yi<sup>a</sup>, Limeng Qu<sup>b,c</sup> and Gary P. T. Choi<sup>d,∗</sup>

<sup>a</sup>School of Mathematics and Statistics, Central South University, Changsha, Hunan, China

<sup>b</sup>Department of General Surgery, The Second Xiangya Hospital, Central South University, Changsha, Hunan, China

<sup>c</sup>Clinical Research Centerfor Breast Disease in Hunan Province, Changsha, Hunan, China

<sup>d</sup>Department of Mathematics, The Chinese University of Hong Kong, Hong Kong SAR, China

## A R T I C L E I N F O

Keywords:   
axillary lymph node metastasis   
geometric quantification   
shape analysis   
breast cancer

## A BS T R AC T

Quantitative characterization of lymph node morphology is important for assessing axillary lymph node metastasis in breast cancer. However, surfaces reconstructed from computed tomography (CT) segmentation may contain geometric and topological defects that compromise subsequent analysis, while conventional shape descriptors predominantly characterize global morphology. To address these issues, we developed an integrated framework combining topology-aware surface processing with multi-resolution spherical harmonic (SH) analysis of CT-derived axillary lymph nodes. The processing pipeline produced topology-valid genus-0 surfaces with improved mesh quality, which were then represented at multiple SH degrees and characterized using 20 predefined geometric feature families. Geometric fidelity increased with SH degree, whereas predictive performance peaked at intermediate resolutions, indicating that maximal reconstruction fidelity did not coincide with maximal discriminative utility. Preferred SH degree also difered across feature families. A family-specific mixed-resolution model achieved an AUC of 0.918 (95% CI, 0.904–0.934), compared with 0.884 (95% CI, 0.868–0.900) for the conventional PyRadiomics Shape14 baseline, corresponding to an improvement of 0.0344 (95% paired bootstrap CI, 0.0212–0.0468). Controlled perturbation experiments showed that higher SH degrees transmitted more fine-scale geometric variation and yielded lower stability of curvature-based predictions. Representative geometric descriptors provided interpretable characterization of metastasis-associated surface morphology. Independent validation further supported the transportability of the framework: label-free replication in a multicenter lymph node cohort reproduced the family-specific resolution efects, while a labeled LIDC-IDRI lung-nodule experiment reproduced the resolution-dependent relationship between SH degree and predictive performance. Altogether, the proposed framework provides a topology-valid basis for multi-scale quantitative characterization of lymph node morphology and metastasis-associated imaging phenotypes.

## 1. Introduction

Axillary lymph node metastasis is a key component of regional staging in breast cancer and has important implications for locoregional treatment planning and prognosis (Marino et al., 2020). Quantitative imaging and radiomics provide a means of characterizing disease-related phenotypes beyond qualitative visual assessment (Lambin et al., 2017), and CT-derived lymph node features have demonstrated predictive value for metastatic status in both single-center and multicenter studies (Yang et al., 2021; Qu et al., 2024, 2025). In clinical imaging, metastatic lymph nodes may exhibit cortical thickening, focal cortical bulging, loss of the fatty hilum, and a transition from an oval or reniform shape toward a round, asymmetric, or irregular contour. These morphological changes may reflect partial or difuse replacement of the nodal architecture by metastatic tumor, with capsular invasion and extranodal extension occurring in more advanced nodal involvement (Dialani et al., 2018). However, lymph node morphology is commonly represented by image-derived radiomic variables or a limited set of global shape descriptors describing size, compactness, principalaxis geometry, and spatial extent. Such measurements may obscure localized or asymmetric changes and provide limited characterization of local surface geometry, symmetry, and spatially heterogeneous three-dimensional morphology. Metastatic involvement may alter nodal morphology in a spatially heterogeneous manner, producing asymmetric enlargement, focal contour bulging, lobulation, or contour irregularity that may not be fully reflected by global measurements of size, elongation, or compactness.

Explicit surface analysis introduces an additional computational requirement: meshes reconstructed from CT-derived lymph node segmentations must provide a numerically and topologically valid domain for subsequent geometric operators. Limited spatial resolution, anisotropic voxel sampling, segmentation boundaries, and surface reconstruction can introduce staircase artifacts, local geometric irregularities, disconnected structures, and topological inconsistencies (Škrinjar and Bistoquet, 2009; Mönch et al., 2011; Ito, 2019; Shirshin et al., 2021). Existing smoothing and topology-repair techniques address individual failure modes (Taubin, 1995; Desbrun et al., 1999; Chen et al., 2025), but smoothing alone does not guarantee a valid spherical domain, and genus alone does not ensure connectedness, closedness, manifoldness, or consistent orientation. SH-based reconstruction has also been explored for correcting topological defects in medically derived surface meshes, particularly for cortical surfaces (Yotter et al., 2009). Reliable surface-based analysis therefore requires coordinated control of geometric artifacts, topology, mesh quality, and geometric preservation.

Once a valid genus-0 surface has been obtained, spherical parameterization provides a natural common domain for representing and comparing three-dimensional geometry (Choi et al., 2015; Lyu et al., 2024). Spherical harmonics (SH) are particularly well suited to this setting because they provide an ordered spectral representation in which surface geometry can be reconstructed at progressively increasing spatial resolutions (Brechbühler et al., 1995; Styner et al., 2006; Khairy and Howard, 2008). Recent work has further investigated improved SH sampling and reconstruction strategies for general three-dimensional shape representation (Li et al., 2024). Low harmonic degrees primarily encode coarse, largescale morphology, whereas higher degrees progressively recover finer geometric structure. SH therefore provide two complementary advantages for quantitative surface analysis: a compact representation of genus-0 geometry and an explicit mechanism for controlling the spatial scale at which morphology is represented.

SH-derived representations have previously been used for medical shape classification, including three-dimensional lung-nodule malignancy analysis and myocardial-infarction screening (El-Baz et al., 2011; Valizadeh et al., 2021). However, these studies primarily used SH-derived representations or coeficients as predictive shape information rather than systematically examining how the finite reconstruction bandwidth influences downstream geometric measurements. This scale dependence raises a central methodological question: the SH degree required for faithful geometric reconstruction need not be the degree that most efectively represents morphology associated with metastatic status. Increasing the maximum harmonic degree expands representational capacity and improves the recovery of fine surface detail, but the additional high-frequency information may contain both disease-related morphology and variation that is only weakly associated with the clinical endpoint. Conversely, truncation at a lower or intermediate degree may suppress fine-scale variation while preserving larger-scale morphological organization. Geometric fidelity and discriminative utility are therefore distinct properties of an SH representation, and optimizing one does not necessarily optimize the other. Resolution may also influence robustness, because higher SH degrees can retain finer geometric perturbations arising from segmentation uncertainty, reconstruction artifacts, or other small surface variations, whereas spectral truncation may attenuate such perturbations. SH degree can therefore be viewed not only as a reconstruction parameter but also as a candidate geometric scale governing the balance among geometric fidelity, discriminative information, and sensitivity to perturbation.

The appropriate geometric scale may further depend on the quantity being measured. Conventional three-dimensional shape features, such as those defined in the PyRadiomics framework, primarily summarize global size, principal-axis geometry, compactness, and surface-to-volume relationships (van Griethuysen et al., 2017). Diferential-geometric descriptors characterize local surface bending and shape type (Meyer et al., 2003; Koenderink and van Doorn, 1992); Laplace–Beltrami spectra and difusion-based signatures characterize intrinsic geometry across multiple scales (Reuter et al., 2006; Sun et al., 2009; Aubry et al., 2011); geodesic measures describe intrinsic spatial extent (Crane et al., 2013); and shape-distribution and symmetry descriptors capture complementary aspects of three-dimensional spatial organization (Osada et al., 2001; Kazhdan et al., 2004). Because these descriptors depend on diferent geometric properties and spatial neighborhoods, there is little reason to expect a single SH degree to represent all feature families equally efectively. A uniformly low resolution may suppress information required by locally sensitive descriptors, whereas a uniformly high resolution may retain fine-scale variation that contributes little to more global measurements or to discrimination. This motivates a feature-dependent treatment of reconstruction scale rather than the assumption of a universally optimal SH degree.

In this study, we develop an integrated framework for topology-aware processing and multi-resolution geometric analysis of CT-derived axillary lymph node surfaces. We first establish a processing workflow that combines contextaware smoothing, adaptive topology correction, geometryconstrained smoothing, and local artifact reconstruction, and evaluate the resulting surfaces in terms of topological validity and computational quality. We then examine SH representations across multiple harmonic degrees to determine how reconstruction resolution afects geometric fidelity, spectral discrimination, and predictive performance. Geometric descriptors are organized into predefined feature families to evaluate feature-dependent resolution preferences and construct a family-specific mixed-resolution representation. Resolution-dependent robustness is further assessed through geometric perturbation transmission, harmonic-domain perturbation energy, and downstream predictive stability. Finally, independent multicenter lymph node and cross-organ LIDC-IDRI experiments are used to assess transportability of the observed geometric and resolution-dependent efects.

## 2. Data and geometric processing of lymph node surfaces

## 2.1. Study cohort and input data

The CT-derived lymph node label maps analyzed in this study were obtained from previously reported datasets (Qu et al., 2024, 2025). Specifically, the published Semi-ALNP cohort included 214 women with unilateral invasive breast cancer who underwent preoperative high-resolution thinsection contrast-enhanced CT and axillary lymph node dissection (ALND). The main eligibility criteria were the absence of distant metastasis and no neoadjuvant therapy before imaging. Briefly, individual axillary lymph nodes were delineated on consecutive thin-slice contrast-enhanced CT images. CT was performed within 1 month before surgery using a high-resolution thin-section contrast-enhanced protocol, with a 1-mm reconstructed slice thickness and 0- mm spacing. Histopathological examination of the ALND specimens served as the reference standard. The labels were defined using two patient groups: 1,769 ALNs from 214 patients, including 1,172 ALNs from 161 patients with no nodal metastasis and 597 ALNs from 53 patients in whom all dissected ALNs were metastatic. In the earlier dataset, three-dimensional lymph node localization was assisted by VitaWorks, followed by segmentation in 3D Slicer. In the subsequent Semi-ALNP dataset, ALNs were segmented in ITK-SNAP v3.8.0 using a region-growing procedure by two independent operators who were blinded to clinical and pathological information. The segmentations were subsequently reviewed and finalized by a senior radiologist. These procedures yielded complete three-dimensional lymph node label volumes.

An independent multicenter external-validation cohort was retrospectively collected from four hospitals between January 2023 and July 2024 (Qu et al., 2025). The external dataset comprised CT-derived three-dimensional axillary lymph node label maps from patients with breast cancer. Individual CT-visible lymph nodes in this cohort were not linked to node-level histopathological metastasis labels; therefore, metastatic status was unavailable for the external surfaces included in the present analysis. An additional cross-organ validation experiment was performed using segmented lung nodules from the publicly available LIDC-IDRI dataset (Armato III et al., 2011). After application of the predefined surface-availability and analysis criteria, 59 lung nodules with corresponding classification labels were retained. This cohort was analyzed independently from the axillary lymph node datasets to examine whether the relationship between SH reconstruction degree and predictive performance was preserved in a distinct anatomical structure and downstream classification task. The LIDC-IDRI data were not used for the development of the lymph node prediction models, the selection of lymph node feature families, or the determination of the SH degrees examined in the primary analysis.

For all cohorts, the resulting NIfTI label maps were converted into triangular surface meshes using a smoothing factor of 0.2. This setting provided mild suppression of voxel-scale irregularities while limiting excessive surface deformation and the introduction of additional geometric artifacts.

## 2.2. Overview of the geometric processing workflow

The reconstructed lymph node surfaces were processed using the sequential geometric workflow illustrated in Fig. 1. The workflow was designed to suppress segmentation- and reconstruction-related artifacts, enforce the topological requirements for spherical parameterization, and improve local mesh quality while limiting unnecessary modification of the underlying surface geometry.

The initial meshes underwent context-aware smoothing to attenuate staircase artifacts, followed by adaptive voxelization with strict topology validation. At each candidate voxel-grid resolution, natural voxelization was evaluated before explicit topology repair. A naturally voxelized surface was accepted whenever it satisfied all predefined validity criteria. Explicit topology repair was invoked only when no naturally valid candidate was available, and a convex-hullbased reconstruction was retained as a final fallback when necessary. A second context-aware smoothing stage was then applied to attenuate staircase-like irregularities introduced or retained during voxelization.

Residual geometric irregularities were further reduced using geometry-constrained adaptive Laplacian and Taubin smoothing. Local surface artifacts were subsequently identified using an initial �-shape-based candidate scan followed by complementary filtering procedures targeting inward folds, small recessed defects, and CT-related geometric artifacts. The selected artifact regions were locally excised and reconstructed using boundary-aware triangulation and geometric fairing, after which the repaired surfaces underwent strict post-repair validation. Only connected, closed, consistently oriented genus-0 surfaces satisfying all predefined meshintegrity criteria were retained for subsequent spherical parameterization and spherical harmonic analysis.

## 2.3. Context-aware smoothing for staircase-artifact suppression

To attenuate staircase artifacts while limiting deformation of unafected surface regions, we applied the contextaware smoothing method of Mönch et al. (2011). The method identifies staircase-prone vertices from local variations in incident-face orientation and propagates a distancedependent smoothing weight over the connected mesh neighborhood. Smoothing is therefore concentrated around detected artifacts rather than applied uniformly to the entire surface.

Using the displacement convention below, the local displacement at vertex $\mathbf { p } _ { i }$ was

$$
\mathbf { D } _ { i } = \frac { 1 } { | \mathcal { N } _ { i } | } \sum _ { j \in \mathcal { N } _ { i } } \left( \mathbf { p } _ { i } - \mathbf { p } _ { j } \right) ,\tag{1}
$$

where $\mathcal { N } _ { i }$ denotes the one-ring vertex neighborhood of $\mathbf { p } _ { i }$ . Each context-aware Taubin iteration consisted of two weighted updates:

$$
\mathbf p _ { i } ^ { \left( k + \frac { 1 } { 2 } \right) } = \mathbf p _ { i } ^ { \left( k \right) } - w _ { i } \lambda \mathbf D _ { i } ^ { \left( k \right) } ,\tag{2}
$$

$$
\mathbf { p } _ { i } ^ { ( k + 1 ) } = \mathbf { p } _ { i } ^ { \left( k + \frac { 1 } { 2 } \right) } - w _ { i } \mu \mathbf { D } _ { i } ^ { \left( k + \frac { 1 } { 2 } \right) } ,\tag{3}
$$

with $\lambda = 0 . 5$ and $\mu = - 0 . 5 3$ . The spatial weights $w _ { i }$ were computed once at the beginning of each smoothing stage using the artifact-detection and distance-decay formulation of Mönch et al. (2011).

In our implementation, the staircase-detection threshold, maximum influence distance, and minimum nonzero smoothing weight were set to $\tau _ { s } = 0 . 5 5 , d _ { \mathrm { m a x } } = 5$ , and $w _ { \mathrm { m i n } } = 0 . 4$ respectively. Context-aware smoothing was applied for 10 iterations before voxelization and reapplied for 5 iterations afterward to suppress staircase-like irregularities introduced or retained during voxelization. Representative examples demonstrating the efect of context-aware smoothing are provided in Fig. 2.

![](images/6a0e3c760693346aaf74c32897ab587663720168b43c90298279d523fdc77859.jpg)  
Figure 1: Overview of the proposed geometric processing workflow. The pipeline comprises initial context-aware smoothing, adaptive voxelization with strict topology correction, geometry-constrained adaptive smoothing, and local artifact detection and repair. The post-repair validation criteria ensure that the output surfaces are connected, closed, consistently oriented genus-0 surfaces for the subsequent spherical analysis.

## 2.4. Adaptive voxelization and strict genus-0 topology correction

Spherical parameterization requires a connected, closed, consistently oriented genus-0 surface (Choi et al., 2015). However, meshes reconstructed from CT-derived lymph node segmentations may contain disconnected components, cavities, handles, or other topological defects. We therefore adapted the voxel-based mesh-repair framework used in previous geometry-image studies (Sinha et al., 2016; Pumarola et al., 2019) by introducing an adaptive, naturalfirst resolution search together with strict topology and meshintegrity validation.

## 2.4.1. Voxel-based reconstruction and strict validation

For each candidate voxel-grid resolution �, the input triangular mesh was converted to a binary voxel representation. The largest 26-connected occupied component was retained after local dilation and hole filling, and a threedimensional �-shape was reconstructed from the occupied voxel centers (Edelsbrunner and Mücke, 1994). The initial � parameter was 0.9, and the resulting surface was transformed back to the coordinate system of the input mesh.

Candidate surfaces were evaluated using criteria stricter than genus alone. For a triangular mesh  with vertex, edge, and face sets �, �, and �, the Euler characteristic is

$$
\chi ( \mathcal { M } ) = | V | - | E | + | F | .\tag{4}
$$

For a connected, closed, orientable surface,

$$
g ( \mathcal { M } ) = 1 - \frac { \chi ( \mathcal { M } ) } { 2 } .\tag{5}
$$

Let � denote the number of connected components and let �, �, �, �, �, and � denote the numbers of boundary edges, non-manifold edges, orientation conflicts, duplicate faces, degenerate faces, and unreferenced vertices, respectively. A candidate was considered strictly valid only when

$$
\mathcal { Q } ( \mathcal { M } ) = \left\{ \begin{array} { l l } { 1 , } & { C = 1 , \ g ( \mathcal { M } ) = 0 , \ B = N = O = D = Z = U = 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{6}
$$

The genus was evaluated only after the mesh had been confirmed to be a closed, consistently oriented two-manifold. Surfaces failing these prerequisites were treated as topologically invalid.

## 2.4.2. Adaptive natural-first search and topology repair

The candidate voxel-grid resolutions were  = {150, 130, 110, 90, 70, 50}, and were evaluated from the finest to the coarsest resolution.

At each resolution, natural voxelization was evaluated first, and the first candidate satisfying Eq. (6) was retained. If no natural candidate was valid, topology repair was applied over the same resolution sequence; if this also failed, a convexhull reconstruction was used as the final fallback, subject to the same strict validity criterion.

The strict topology and mesh-integrity validation was repeated immediately before export.

## 2.5. Geometry-constrained adaptive smoothing

Although the preceding processing steps corrected major topological defects and staircase-like artifacts, residual high-frequency geometric irregularities could remain.

Geometric quantification and shape analysis for axillary lymph node metastasis

![](images/8a329338b6dee95df5dc885577cb43236a2aa90ea9d1fb6e703f8645eace9ada.jpg)  
Figure 2: Qualitative comparison of surface smoothing stages. Representative lymph node surface meshes are shown before smoothing, after context-aware (CA) smoothing, and after adaptive Laplacian–Taubin smoothing.

Standard Laplacian and Taubin smoothing can reduce such variation but may also alter the underlying geometry when applied excessively (Taubin, 1995; Desbrun et al., 1999; Vollmer et al., 1999). Rather than applying a fixed smoothing duration to every surface, we developed a two-stage adaptive procedure in which the number of iterations was selected independently for each mesh according to improvement in surface smoothness subject to predefined geometric-fidelity constraints. As illustrated in Fig. 2, the subsequent geometryconstrained smoothing further attenuated residual surface irregularities while preserving the overall morphology.

## 2.5.1. Adaptive Laplacian stage

The first stage used implicit cotangent-Laplacian smoothing (Desbrun et al., 1999; Meyer et al., 2003). Using the negative-semidefinite Laplacian convention of the implementation, the update was

$$
\left( \mathbf { I } - \lambda _ { \mathrm { L } } \mathbf { L } \right) \mathbf { X } ^ { ( t + 1 ) } = \mathbf { X } ^ { ( t ) } , \qquad \lambda _ { \mathrm { L } } = 0 . 0 5 .\tag{7}
$$

Surface smoothness was quantified using the integrated squared Laplace–Beltrami energy

$$
E _ { \mathrm { L } } ( \mathcal { M } ) = \sum _ { i = 1 } ^ { n } A _ { i } \left\| \Delta _ { \mathcal { M } } \mathbf { x } _ { i } \right\| ^ { 2 } ,\tag{8}
$$

where $A _ { i }$ is the lumped area associated with vertex �.

Geometric fidelity was constrained by the relative enclosed-volume change and the maximum relative change among the three principal-axis extents, both measured with respect to the input surface.

## 2.5.2. Adaptive Taubin stage

The Laplacian-smoothed surface was subsequently refined using standard Taubin smoothing (Taubin, 1995), with $\lambda _ { \mathrm { T } } = 0 . 1 0$ and $\mu _ { \mathrm { T } } = - 0 . 1 1$ . For adaptive stopping, local curvature variation was characterized using the absolute angle defect $\begin{array} { r } { K _ { i } = \left| 2 \pi - \sum _ { f \in \mathcal { F } _ { i } } \theta _ { i f } \right| } \end{array}$ . Following Wang et al. (2012), the area-weighted global roughness was

![](images/9f0c6c01c6a62bc92b66f8d60d908c0fac2930beca1e2d836f3f406a52fa5f9d.jpg)  
Figure 3: Qualitative examples of local artifact detection using three complementary filtering strategies. From left to right: raybased inward-fold detection, local small-hole detection, and CTinformed geometric screening. The top row shows the location of each detected region on the complete surface, and the bottom row shows the corresponding enlarged mesh patch.

$$
R ( \mathcal { M } ) = \frac { \sum _ { i = 1 } ^ { n } A _ { i } \rho _ { i } } { \sum _ { i = 1 } ^ { n } A _ { i } } ,\tag{9}
$$

where $\rho _ { i }$ measures the diference between $K _ { i }$ and its cotangentweighted local neighborhood average.

Spatial deviation was additionally constrained using a symmetric, area-weighted 95th-percentile Hausdorf distance following the mesh-based framework of MeshMetrics (Podobnik and Vrtovec, 2025). The distance was normalized by the efective voxel scale,

$$
h = \frac { \operatorname* { m a x } \{ \ell _ { x } , \ell _ { y } , \ell _ { z } \} } { r - 1 } , \qquad d _ { 9 5 } ^ { * } = \frac { \mathrm { H D } _ { 9 5 } ( \mathcal { M } _ { 1 } , \mathcal { M } _ { 2 } ) } { h } ,\tag{10}
$$

where $\ell _ { x } , \ell _ { y }$ , and $\ell _ { z }$ are the bounding-box dimensions of the reference surface and � is the selected voxel-grid resolution.

## 2.5.3. Adaptive stopping and rollback

For either stage, let $J ^ { ( t ) }$ denote the corresponding smoothness objective, $E _ { \mathrm { L } }$ for the Laplacian stage or $R ( \mathcal { M } )$ for the Taubin stage. The relative improvement was

$$
I ^ { ( t ) } = 1 0 0 \frac { J ^ { ( t - 1 ) } - J ^ { ( t ) } } { J ^ { ( t - 1 ) } } .\tag{11}
$$

An iteration was accepted only if the stage-specific objective did not worsen and all geometric-fidelity constraints were satisfied; otherwise, the procedure reverted to the best previously accepted state. The stopping threshold was 0.20% improvement with a patience of two evaluations. Volume/principal-axis limits were 0.6%∕3.0% during the Laplacian stage and 1.0%∕3.0% for the final surface, with normalized HD95 limits of 1.0 relative to the Laplaciansmoothed surface and 1.2 relative to the original surface. A maximum of 100 iterations per stage served only as a safety limit.

## 2.6. Detection of local surface artifacts

Although the preceding smoothing and topology correction steps removed major surface irregularities, small localized artifacts could remain without altering the global mesh topology. These defects typically appeared as narrow inward folds, steep-walled pits, or small recessed regions and therefore could not be identified reliably from genus or connectivity alone. We developed a three-stage local detection framework consisting of an initial �-shape-based candidate localization, followed by complementary ray-based fold detection, local small-hole detection, and CT-informed geometric screening (Fig. 3).

## 2.6.1. �-shape candidate localization

Let $\mathcal { M } = ( V , F )$ denote the processed triangular mesh. A three-dimensional �-shape was constructed from the mesh vertices as an external geometric reference (Edelsbrunner and Mücke, 1994). Rather than using a fixed dimensional value, the � parameter was scaled relative to the critical value required to obtain a single connected �-shape region,

$$
\alpha = 1 . 4 \alpha _ { \mathrm { o n e } } ,\tag{12}
$$

where $\alpha _ { \mathrm { o n e } }$ denotes the corresponding one-region critical value.

For each input vertex $\mathbf { v } _ { i } ,$ let $d _ { i }$ denote its distance to the �-shape boundary and let $\overline { { \ell } }$ denote the mean mesh-edge length. Vertices not belonging to the �-shape boundary were considered candidate vertices when $d _ { i } > 2 . 5 \overline { { \ell } }$ . The initial candidate mask was formed by all faces incident to at least one candidate vertex. Candidate faces were partitioned into edge-connected components and processed independently. The �-shape stage was used only to define a sensitive search domain and did not directly determine the faces subsequently removed.

## 2.6.2. Ray-based inward-fold detection

The first filtering stage targeted narrow inward folds and re-entrant surface regions, as illustrated in the left panel of Fig. 3. A locally regular surface is approximately singlevalued relative to its outward direction, whereas an inward fold may produce multiple separated intersections along the same inward-directed ray.

For each candidate component containing at least 10 faces, three rings of surrounding non-candidate faces were used to estimate a local reference frame from area-weighted face centroids and consistently oriented normals. Let $\widetilde { \ell } _ { \mathrm { l o c } }$ denote the median edge length within this local region. Parallel rays were sampled over the projected candidate component with spacing $h _ { \mathrm { r a y } } \ = \ 1 . 5 \widetilde { \ell _ { \mathrm { l o c } } } ,$ , using at least a $3 \times 3$ grid and at most 1200 rays per component. Each ray was cast inward along the negative local reference normal,

$$
\begin{array} { r } { { \bf x } _ { r } ( t ) = { \bf 0 } _ { r } - t { \bf n } _ { k } , \qquad t > 0 , } \end{array}\tag{13}
$$

where $\mathbf { 0 } _ { r }$ denotes the ray origin.

Ray–triangle intersections within the local region were computed using the two-sided Möller–Trumbore algorithm (Möller and Trumbore, 1997). Nearly tangential and numerically coincident intersections were excluded or merged before analysis. If a ray intersected the candidate component more than once, the normalized separation between its first and last candidate crossings was defined as

$$
\Delta _ { r } ^ { * } = \frac { s _ { r , m _ { r } } - s _ { r , 1 } } { \widetilde { \ell } _ { \mathrm { l o c } } } \ge 2 . 5 .\tag{14}
$$

Let $N _ { \mathrm { v } } , \ N _ { \mathrm { m } } ,$ and $N _ { \mathrm { d } }$ denote the numbers of valid, multi-hit, and deep multi-hit rays, respectively. A component was retained only when $N _ { \mathrm { v } } ~ \ge ~ 5 , ~ N _ { \mathrm { m } } ~ \ge ~ 3 , ~ N _ { \mathrm { m } } / N _ { \mathrm { v } } ~ \ge$ 0.05, $N _ { \mathrm { d } } ~ \ge ~ 2 .$ , and $N _ { \mathrm { d } } / N _ { \mathrm { v } } ~ \ge ~ 0 . 0 2$ . These joint count and proportion requirements reduced sensitivity to isolated numerical intersections.

Deep rays were grouped using 8-neighbor connectivity on the sampling grid, and clusters containing fewer than two deep rays were discarded. Intersected candidate faces from each retained cluster were used as seed faces, while candidate faces adjacent to the component exterior formed a two-ring mouth-anchor band.

A weighted face-adjacency graph was then constructed over the candidate component. Let $d _ { \mathrm { s } } ( f )$ and $d _ { \mathrm { m } } ( f )$ denote shortest-path distances from face $f$ to the seed and mouthanchor sets, respectively. We defined the normalized seed– mouth competition score

$$
q ( f ) = \frac { d _ { \mathrm { m } } ( f ) } { d _ { \mathrm { s } } ( f ) + d _ { \mathrm { m } } ( f ) } .\tag{15}
$$

Candidate thresholds from 0.90 to 0.05 were evaluated in steps of 0.025. The highest threshold was retained for which the selected region contained all seed faces, formed a single edge-connected patch with one unbranched boundary loop, and occupied no more than 65% of the original candidate component. Components for which no admissible region was found were not automatically included in the removal mask.

Finally, to restrict the mask to suficiently recessed geometry, inward depth was measured relative to the local reference plane. Faces were retained only when their inward depth was at least 40% of the 90th percentile of positive seedface depths. Connected regions containing fewer than 20 faces were discarded. The remaining inward region was expanded by at most four face-adjacency rings within the previously accepted graph-based region, and the largest connected patch was retained as the ray-selected artifact mask.

## 2.6.3. Local small-hole detection

Candidate regions not selected by the ray-based stage were subsequently examined for small, steep-walled pits (middle panel of Fig. 3). This detector combined relative triangle size and shape, adjacent-face normal variation, and a multiscale smoothing response.

For face $f _ { i }$ with area $A _ { i } ,$ , relative area was evaluated with respect to both the surrounding non-candidate surface and a local mesh neighborhood. The efective area ratio was

$$
r _ { i } = \operatorname* { m i n } \left\{ { \frac { A _ { i } } { { \bmod { \mathrm { i a n } } } _ { j \in S _ { i } } A _ { j } } } , { \frac { A _ { i } } { { \bmod { \mathrm { i a n } } } _ { j \in \mathcal { N } _ { i } } A _ { j } } } \right\} ,\tag{16}
$$

where $S _ { i }$ denotes the surrounding reference faces and $\mathcal { N } _ { i }$ a local multi-ring neighborhood. Triangle regularity was measured by

$$
q _ { i } = \frac { 4 \sqrt { 3 } A _ { i } } { \ell _ { i , 1 } ^ { 2 } + \ell _ { i , 2 } ^ { 2 } + \ell _ { i , 3 } ^ { 2 } } ,\tag{17}
$$

where $\ell _ { i , 1 } , \ell _ { i , 2 } , \ell _ { i , 3 }$ denote the side lengths of face $f _ { i } ;$ $q _ { i } \ = \ 1$ for an equilateral triangle. Let $\theta _ { i }$ denote the maximum normal-angle diference between $f _ { i }$ and its edgeadjacent faces. A geometry-based seed was generated when $\left( r _ { i } \le 0 . 5 5 \ \vee \ q _ { i } \le 0 . 4 0 \right) \wedge \left( \theta _ { i } \ge 3 5 ^ { \circ } \right)$

To detect recessed pits not characterized by unusually small or slender triangles, we additionally measured their response to mild implicit cotangent-Laplacian smoothing (Desbrun et al., 1999; Meyer et al., 2003). For vertex $\mathbf { v } _ { p } ,$ the normal displacement at smoothing scale � was normalized by the global mean edge length ${ \overline { { \ell } } } ,$

$$
s _ { p } ^ { ( t ) } = \frac { \left( \mathbf { v } _ { p } ^ { ( t ) } - \mathbf { v } _ { p } ^ { ( 0 ) } \right) ^ { \top } \mathbf { n } _ { p } } { \overline { { \ell } } } , \qquad t \in \{ 1 , 2 , 4 \} .\tag{18}
$$

The face response was the maximum value across its vertices and the three smoothing scales. A smoothing-response seed was retained when the response exceeded both 0.08 edge lengths and the local 95th percentile.

The union of the geometry- and smoothing-based seeds was expanded by three face rings within the local candidate domain. Enclosed internal islands were filled to obtain spatially continuous patches. A patch was retained only if its seed fraction was at least 0.02, it contained no more than 220 faces, and it occupied no more than 20% of its local growth domain.

## 2.6.4. CT-informed geometric screening

A third complementary stage screened residual candidate regions using geometric patterns motivated by common CT-derived surface artifacts, as described in the right panel of Fig. 3. Four evidence branches were considered: thin opposing surface sheets, local inward recession, risk of opposing-wall merging, and staircase or terrace patterns associated with the through-plane direction (Cevidanes et al., 2010; Ito, 2019; Shirshin et al., 2021; Mönch et al., 2011).

Let ℎ denote the efective voxel scale associated with the reconstructed surface. Thin-wall evidence was recorded when the estimated distance $t _ { i }$ between approximately opposing surface sheets satisfied $t _ { i } / h ~ \leq ~ 2 . 5$ . Local recession was defined relative to a fitted reference plane and required an inward depth of at least 0.5ℎ together with local spatial continuity.

A wall-break score was defined by

$$
b _ { i } = \frac { e _ { i } + e _ { i } ^ { \mathrm { { o p p } } } } { t _ { i } } ,\tag{19}
$$

where $e _ { i }$ and $e _ { i } ^ { \mathrm { { \mathrm { { o p p } } } } }$ denote multiscale surface-deviation responses on the two opposing sheets. Faces with $b _ { i } \geq$

![](images/591028fa261b88ce94f569c85ddc893378be939c0b5ad4770ae2af381d40e385.jpg)  
Figure 4: Local artifact excision and surface-reconstruction workflow. (A) Artifact excision. (B) Initial triangulation and conforming subdivision. (C) Quadratic reference-surface estimation. (D) Constrained cotangent bi-Laplacian fairing. (E) Final reconstructed surface after local Taubin smoothing.

0.70 provided wall-break evidence. Staircase evidence was derived from local variation in face orientation relative to the prescribed stack direction and from small, nearly planar terraces.

To reduce false-positive removal of anatomically plausible concavities, staircase evidence was not suficient by itself. A strict seed required at least two evidence branches, with at least one corresponding to thin-wall, recession, or wall-break evidence. Connected support regions were subsequently formed around these multi-evidence seeds.

## 2.7. Repair and reconstruction of local surface artifacts

Let $c _ { \mathrm { r a y } } , c _ { \mathrm { h o l e } } ,$ , and $c _ { \mathrm { { C T } } }$ denote the faces selected by the three detection stages. Their union

$$
C = C _ { \mathrm { r a y } } \cup C _ { \mathrm { h o l e } } \cup C _ { \mathrm { C T } }\tag{20}
$$

defined the candidate removal mask passed to the reconstruction stage. The local repair procedure consisted of artifact excision, boundary-aware triangulation, conforming subdivision, quadratic reference-surface estimation, constrained cotangent bi-Laplacian fairing, and final local Taubin smoothing. The complete repair sequence is summarized in Fig. 4.

## 2.7.1. Local excision and initial patch construction

The selected artifact faces were removed from the source mesh, the unreferenced vertices were discarded, and the local boundary loops were extracted from the edges incident to exactly one retained face. This excision step corresponds to Fig. 4A. Hole filling followed the standard boundarytriangulation paradigm used in mesh repair (Liepa, 2003).

For each boundary loop, a best-fitting plane was estimated by singular value decomposition. When the planar projection produced a valid non-self-intersecting polygon, the opening was filled using constrained Delaunay triangulation (Shewchuk, 1996). If planar triangulation was not admissible because of geometric or connectivity conflicts, a three-dimensional conflict-aware ear-clipping procedure was used instead. A centroid fan was retained only as a final fallback. New triangles were oriented consistently with the retained surface along the shared boundary. The initial patch obtained after this step is presented in Fig. 4B.

To provide suficient degrees of freedom for subsequent geometric refinement, long edges in the reconstructed patch were subdivided conformingly. Let $\ell _ { \mathrm { m e d } }$ denote the median edge length of the input mesh. Edges satisfying $\ell _ { e } > 1 . 5 \ell _ { \mathrm { m e d } }$ were split at their midpoints, for at most three subdivision passes.

## 2.7.2. Quadratic local reference surface

The initial triangulation determines patch connectivity but does not by itself recover the local surface geometry. We therefore estimated a smooth reference surface from five rings of measured faces surrounding each reconstructed patch. The resulting quadratic reference-surface construction is represented in Fig. 4C.

For patch $\mathcal { P } _ { k } .$ , an orthonormal local frame $\{ \mathbf { b } _ { 1 , k } , \mathbf { b } _ { 2 , k } , \mathbf { n } _ { k } \}$ was estimated from the surrounding measured vertices. In local coordinates (�, �, ℎ), the reference geometry was represented by the quadratic height field

$$
\widehat { h } _ { k } ( u , v ) = \phi \left( \frac { u } { s _ { k } } , \frac { v } { s _ { k } } \right) ^ { \top } \mathbf { a } _ { k } ,\tag{21}
$$

where $\phi ( \widetilde { u } , \widetilde { v } ) = \left| \widetilde { u } ^ { 2 } \quad \widetilde { u } \widetilde { v } \quad \widetilde { v } ^ { 2 } \quad \widetilde { u } \quad \widetilde { v } \quad 1 \right| ^ { \top }$ . The coeficient vector was obtained by distance-weighted, weakly regularized least squares,

$$
\mathbf { a } _ { k } = \underset { \mathbf { a } \in \mathbb { R } ^ { 6 } } { \arg \operatorname* { m i n } } \left[ \sum _ { i \in \mathcal { V } _ { k } ^ { \mathrm { r e f } } } \omega _ { i } \left( \boldsymbol { \phi } _ { i } ^ { \top } \mathbf { a } - h _ { i } \right) ^ { 2 } + \varepsilon _ { \mathrm { q } } \| \mathbf { a } \| _ { 2 } ^ { 2 } \right] ,\tag{22}
$$

with larger weights assigned to measured vertices closer to the reconstructed region. The fit was omitted when fewer than 12 measured reference vertices were available or when the local frame was numerically degenerate.

Only interior patch vertices were projected toward the fitted reference surface; boundary vertices remained fixed. The normal displacement of each interior vertex was clipped to $| d _ { i } | \leq 2 \ell _ { \mathrm { m e d } }$ , thereby limiting the efect of an unstable local fit.

## 2.7.3. Constrained cotangent bi-Laplacian fairing

Following reference-surface estimation, the reconstructed patch and four surrounding face rings were refined jointly. Cotangent weights were used to construct a discrete Laplace–Beltrami operator (Meyer et al., 2003). In this subsection, the positive-semidefinite sign convention was used. Vertices on the outer boundary of the local fairing region were fixed, providing Dirichlet boundary conditions and preventing repair-induced deformation from propagating into the distant measured surface. The constrained fairing stage is depicted in Fig. 4D.

Let $\mathcal { V } _ { k }$ and $D _ { k }$ denote the movable and fixed vertices, respectively, and partition the relevant Laplacian rows as $\begin{array} { l l l } { { { \cal L } _ { \Omega _ { k } } } } & { { = } } & { { \left[ A _ { k } \quad B _ { k } \right] } } \end{array}$ . The movable vertex positions were obtained by minimizing

$$
E _ { k } ( \mathbf { X } _ { \mathcal { V } _ { k } } ) = \left\| A _ { k } \mathbf { X } _ { \mathcal { V } _ { k } } + B _ { k } \mathbf { X } _ { \mathcal { D } _ { k } } \right\| _ { F } ^ { 2 } + \sum _ { i \in \mathcal { V } _ { k } } \gamma _ { i } \left\| \mathbf { x } _ { i } - \mathbf { x } _ { i } ^ { \mathrm { r e f } } \right\| _ { 2 } ^ { 2 } + \varepsilon _ { \mathrm { f } } \left\| \mathbf { X } _ { \mathcal { V } _ { k } } \right\| _ { F } ^ { 2 } .\tag{23}
$$

This formulation combines Laplacian fairing (Desbrun et al., 1999; Meyer et al., 2003) with soft positional constraints. The positional weights were

$$
\gamma _ { i } = { \left\{ \begin{array} { l l } { 0 . 5 , } & { i { \mathrm { ~ b e l o n g s ~ t o ~ t h e ~ r e c o n s t r u c t e d ~ p a t c h } } , } \\ { 1 0 , } & { i { \mathrm { ~ b e l o n g s ~ t o ~ t h e ~ m e a s u r e d ~ s u r r o u n d i n g s } } . } \end{array} \right. }\tag{24}
$$

Thus, reconstructed vertices were allowed to adapt to the inferred local geometry, whereas measured surrounding vertices were strongly constrained to their original positions. The numerical regularization factor was $1 0 ^ { - 8 }$ times the mean diagonal magnitude of $A _ { k } ^ { \top } A _ { k }$

## 2.7.4. Local smoothing andpost-repair validation

After bi-Laplacian fairing, the reconstructed patch and five surrounding face rings underwent five iterations of local Taubin smoothing (Taubin, 1995), with $\lambda _ { \mathrm { R } } ~ = ~ 0 . 2 5$ and $\mu _ { \mathrm { R } } ~ = ~ - 0 . 2 6$ . Vertices on the outer boundary of the local smoothing region remained fixed, ultimately yielding the final locally smoothed patch shown in Fig. 4E.

To prevent local element inversion during geometric refinement, quadratic projection, bi-Laplacian fairing, and each Taubin half-step were subjected to backtracking. Proposed displacements were progressively reduced until every afected triangle retained positive area and preserved its orientation relative to the previous state. If $\mathbf { a } _ { f } = \left( \mathbf { x } _ { f _ { 2 } } - \mathbf { x } _ { f _ { 1 } } \right) \times \left( \mathbf { x } _ { f _ { 3 } } - \mathbf { x } _ { f _ { 1 } } \right)$ denotes the oriented area vector of face $f ,$ an update was accepted only if

$$
\left\| \mathbf { a } _ { f } ^ { \mathrm { n e w } } \right\| _ { 2 } > \varepsilon _ { \mathrm { A } } , \qquad \left( \mathbf { a } _ { f } ^ { \mathrm { o l d } } \right) ^ { \mathsf { T } } \mathbf { a } _ { f } ^ { \mathrm { n e w } } > 0\tag{25}
$$

for all afected faces. Updates for which no admissible displacement scale could be found were rejected.

After reconstruction, vertex-touching floating fragments were removed by retaining the largest edge-connected surface component, and the mesh indices were compacted. The complete strict topology and mesh-integrity validation defined in Section 2.4 was then repeated. Altogether, the pipeline produces connected, closed, consistently oriented genus-0 meshes without boundary edges, non-manifold edges, orientation conflicts, duplicate faces, degenerate faces, or unreferenced vertices for the subsequent spherical parameterization and spherical harmonic analysis.

## 3. Shape analysis of lymph node surfaces

## 3.1. Spherical mapping

After geometric processing and topology validation, each genus-0 lymph node surface was parameterized onto the unit sphere to provide a common domain for subsequent spherical harmonic analysis. Three candidate parameterizations were considered: a spherical conformal map followed by Möbius area correction (Choi et al., 2015, 2020), an area-preserving spherical density-equalizing map initialized from the conformal parameterization (Lyu et al., 2024), and a spherical Tutte map (Tutte, 1963) included as a robust alternative.

Each candidate mapping was subjected to validity screening according to its parameterization characteristics. For the conformal and area-preserving parameterizations, a mapping was considered numerically valid only when the mapped surface exhibited consistent local face orientation, contained no degenerate spherical triangles or collapsed mapped vertices, and had a total spherical triangle area within 5% of 4�. When geometry-related validity issues occurred in the conformal or area-preserving parameterizations, the spherical Tutte mapping was used as a fallback.

Mapping selection was performed independently for each lymph node and each SH reconstruction degree (the SH reconstruction procedures are explained in the following section). For a given degree �, the SH-reconstructed surface was obtained using each admissible spherical mapping, and agreement with the processed reference surface was quantified using the normalized symmetric bidirectional nearest-neighbor root-mean-square distance

$$
\begin{array} { r } { E _ { \mathrm { R M S } } = \frac { 1 } { D _ { \mathrm { b b o x } } } \sqrt { \frac { 1 } { 2 } \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m i n } _ { j } \| { \bf x } _ { i } - \widehat { { \bf x } } _ { j } \| _ { 2 } ^ { 2 } + \frac { 1 } { \widehat { N } } \sum _ { j = 1 } ^ { \widehat { N } } \operatorname* { m i n } _ { i } \| \widehat { { \bf x } } _ { j } - { \bf x } _ { i } \| _ { 2 } ^ { 2 } \right] } , } \end{array}\tag{26}
$$

where $\mathbf { x } _ { i }$ and $\widehat { \mathbf { x } } _ { j }$ denote vertices of the processed reference and SH-reconstructed surfaces, respectively, and $D _ { \mathrm { b b o x } }$ is the diagonal length of the reference-surface bounding box. Among the admissible mappings, the mapping yielding the smallest $E _ { \mathrm { R M S } }$ was retained for that lymph node and SH degree. Symmetric mean nearest-neighbor distance and HD95 were additionally recorded for quality assessment but were not used for mapping selection. The entire mappingselection procedure was based solely on geometric criteria and did not use metastatic status, class labels, or predictive outcomes.

## 3.2. Spherical harmonic representation

Using the degree-specific spherical parameterization selected above, each lymph node surface was represented using real spherical harmonics (SH), which provide a hierarchical spectral representation of genus-0 three-dimensional surfaces (Brechbühler et al., 1995; Styner et al., 2006;

Kazhdan et al., 2003). Let $\mathbf { x } ( \theta , \phi ) ~ = ~ \left\lceil { x ( \theta , \phi ) } \right\rceil$ denote

L the Cartesian surface coordinates associated with spherical coordinates $( \theta , \phi )$ . The degree-� approximation was

$$
\mathbf { x } _ { L } ( \theta , \phi ) = \sum _ { \ell = 0 } ^ { L } \sum _ { m = - \ell } ^ { \ell } \mathbf { c } _ { \ell m } Y _ { \ell } ^ { m } ( \theta , \phi ) ,\tag{27}
$$

where $Y _ { \ell } ^ { m }$ denotes a real orthonormal spherical harmonic basis function of degree $\ell$ and order �, $\mathbf { c } _ { \ell m } \in \mathbb { R } ^ { 3 }$ is the corresponding coeficient vector, and � is the maximum harmonic degree.

In the numerical implementation, SH coeficients were estimated after weighted centroid removal using spherical triangle areas as quadrature weights. This weighting reduces sensitivity to nonuniform vertex sampling on the spherical parameterization. The weighted centroid was restored after reconstruction.

For the primary lymph node analyses, we evaluated $L \in \{ 5 , 8 , 1 0 , 1 5 , 2 0 , 3 4 \}$ , and a degree-� expansion contains $N _ { \mathrm { S H } } = ( L + 1 ) ^ { 2 }$ three-dimensional coeficient vectors. The corresponding reconstructions were subsequently used for geometric feature extraction, multi-resolution analysis, and evaluation of geometric fidelity and predictive performance. A separate degree range was used for the LIDC-IDRI crossorgan experiment, as specified in Section 3.6.2.

## 3.3. Surface feature extraction

To characterize lymph node morphology beyond conventional global shape measurements, geometric descriptors were extracted from the processed reference surfaces and from the SH-reconstructed surfaces. For the multiresolution analysis, identical feature-extraction procedures and numerical settings were applied independently to the SH reconstructions at diferent degrees.

## 3.3.1. Conventional geometric shapefeatures

Conventional global morphology was characterized using the 14 three-dimensional shape features defined in the PyRadiomics framework (van Griethuysen et al., 2017). The same descriptors were extracted from the original, processed-reference, and SH-reconstructed surfaces. These features, denoted $P y I 4 ,$ also served as the conventional globalmorphology baseline.

## 3.3.2. Extended surface descriptors andfeature-family organization

Extended descriptors were extracted to characterize complementary aspects of global geometry, local diferential geometry, intrinsic spectral geometry, shape distributions, symmetry, and SH spectral organization. The descriptors were organized a priori into 20 geometrically defined feature families (Table 1) using established formulations for diferential geometry, intrinsic spectral descriptors, geodesic measures, shape distributions, and symmetry (Meyer et al., 2003; Koenderink and van Doorn, 1992; Reuter et al., 2006; Sun et al., 2009; Aubry et al., 2011; Crane et al., 2013; Osada et al., 2001; Kazhdan et al., 2004). The extended set comprised 352 variables; together with $P y I 4 ,$ , this yielded 366 candidate variables for multi-resolution analysis.

## 3.4. Quantitative evaluation and statistical analysis

Quantitative analyses evaluated geometric validity, multi-resolution representation, predictive performance, perturbation robustness, and transportability across independent datasets. The complete set of 1, 769 surfaces was used for geometric processing and SH analyses, while predefined subsets were used for computationally intensive paired evaluations as specified below. The independent multicenter lymph node cohort and the LIDC-IDRI lung-nodule cohort were analyzed separately from the primary cohort for label-free external replication and cross-organ validation, respectively.

Feature families used in the multi-resolution SH analysis.
<table><tr><td>Feature family</td><td>Main descriptors</td><td>n</td></tr><tr><td>Convexity</td><td>Convex-hull measures</td><td>6</td></tr><tr><td>Radial morphology</td><td>Radial-distance statistics</td><td>15</td></tr><tr><td>Reflection symmetry</td><td>Principal-plane reflection errors</td><td>9</td></tr><tr><td>Mean curvature</td><td>Mean-curvature statistics</td><td>26</td></tr><tr><td>Gaussian curvature</td><td>Gaussian-curvature statistics</td><td>13</td></tr><tr><td>Shape Index and</td><td>Local shape and bending</td><td>28</td></tr><tr><td>Curvedness</td><td>measures</td><td></td></tr><tr><td>Curvature integrals</td><td>Integrated curvature measures</td><td>9</td></tr><tr><td>Curvature regions</td><td>Curvature-region descriptors</td><td>16</td></tr><tr><td>Laplace-Beltrami spectrum</td><td>Spectral eigenvalue descriptors</td><td>45</td></tr><tr><td>Heat Kernel Signature</td><td>Multi-scale HKS descriptors</td><td>60</td></tr><tr><td>Wave Kernel Signature</td><td>Multi-scale WKS descriptors</td><td>50</td></tr><tr><td>Geodesic geometry</td><td>Intrinsic extent measures</td><td>3</td></tr><tr><td>D2 shape distribution</td><td>Pairwise-distance distribution</td><td>10</td></tr><tr><td>D3 shape distribution</td><td>Triangle-area distribution</td><td>10</td></tr><tr><td>D4 shape distribution</td><td>Tetrahedral-volume distribution</td><td>10</td></tr><tr><td>Spherical extent</td><td>Directional extent and anisotropy</td><td>12</td></tr><tr><td>Central asymmetry</td><td>Opposite-direction asymmetry</td><td>10</td></tr><tr><td>180° rotational symmetry Py14</td><td>Rotational discrepancy measures</td><td>8</td></tr><tr><td>SH spectral energy</td><td>Conventional PyRadiomics shape features</td><td>14</td></tr><tr><td></td><td>SH energy-distribution descriptors</td><td>12</td></tr><tr><td>Total</td><td></td><td>366</td></tr></table>

The individual lymph node was the unit of prediction and performance evaluation, whereas the patient was the grouping and resampling unit used to prevent information leakage and account for within-patient dependence. For all predictive analyses, outer and inner cross-validation were performed at the patient level, with all lymph nodes from the same patient assigned to the same fold. Data-dependent preprocessing and hyperparameter selection used training data only, and identical outer partitions were retained for paired model comparisons. For descriptive interpretation of metastasis-associated geometric phenotypes, univariate AUC and Clif’s � were additionally used to summarize group separation between metastatic and non-metastatic lymph nodes.

## 3.4.1. Topological validity and downstream computational quality

Topological validity was evaluated on all 1, 769 surfaces before and after geometric processing using the strict validation criteria defined in Section 2.4. The primary endpoint was strict topology validity, which required a single connected, closed, consistently oriented genus-0 surface without boundary edges, non-manifold elements, orientation conflicts, duplicate faces, degenerate faces, or unreferenced vertices. Single-component structure, genus-0 topology, and outward orientation were additionally summarized as representative component criteria.

Downstream computational quality was evaluated on a randomly selected fixed subset of 500 paired original and processed surfaces using two complementary measures: triangle quality and adjacent-face normal variation. Triangle quality was quantified using the normalized area-based metric defined in Eq. (17), where values closer to one indicate more equilateral triangles. The reported metric, triangle quality (5th percentile), was defined as the fifth percentile of $q _ { i }$ over all mesh faces. Adjacent-face normal variation, quantified by the normal jump (95th percentile, degrees), was defined as the area-weighted 95th percentile of dihedral angles between neighboring face normals. Higher triangle quality and lower normal jump indicate improved numerical suitability for subsequent diferential-geometric computation.

## 3.4.2. Univariate discrimination of conventional shape descriptors

To assess whether SH reconstruction altered the discriminative behavior of conventional morphology, each Py14 descriptor was evaluated independently on the original surface and on the six SH reconstructions using a singlefeature L2-regularized logistic-regression model. This analysis used 10 repeats of five-fold patient-grouped stratified outer cross-validation, with five-fold patient-grouped inner crossvalidation for regularization selection. Identical outer splits were used across descriptors and surface representations, and out-of-fold probabilities were averaged across the 10 repeats before calculation of the final AUC.

For descriptor � and degree �, the change relative to the original surface was

$$
\Delta \mathrm { A U C } _ { f , L } = \mathrm { A U C } _ { f , L } ^ { \mathrm { S H } } - \mathrm { A U C } _ { f } ^ { \mathrm { o r i g i n a l } } .\tag{28}
$$

## 3.4.3. Degree-wise spectral localization of discriminative information

To characterize the discriminative information carried by diferent harmonic degrees, we performed a degree-wise spectral analysis using the coeficients from the degree-34 SH representation. This approach avoids reconstructing separate surfaces at each harmonic degree and directly evaluates the contribution of individual degrees within a common highresolution representation (Kazhdan et al., 2003). For degree $\ell ,$ the SH energy was defined as

$$
E _ { \ell } = \sum _ { m = - \ell } ^ { \ell } \left\| \mathbf { c } _ { \ell m } \right\| _ { 2 } ^ { 2 } ,\tag{29}
$$

and its relative contribution to nonconstant shape variation was

$$
P _ { \ell } = E _ { \ell } / ( \sum _ { k = 1 } ^ { 3 4 } E _ { k } ) , \qquad \ell = 1 , \dots , 3 4 .\tag{30}
$$

The degree-0 term was excluded because it represents the constant component and was not treated as shape variation.

The univariate discriminative strength of each degree was summarized using the folded AUC

$$
\begin{array} { r } { \mathrm { A U C } _ { \ell } ^ { \ast } = \operatorname* { m a x } \left\{ \mathrm { A U C } _ { \ell } , 1 - \mathrm { A U C } _ { \ell } \right\} , } \end{array}\tag{31}
$$

for which 0.5 represents chance-level discrimination and which summarizes discrimination irrespective of the direction of association. Degree-wise energy and discriminative

performance were examined jointly to distinguish the amount of geometric information represented at a given spectral scale from its relevance to metastatic-status discrimination.

## 3.4.4. Geometricfidelity and high-resolution representation eficiency

Direct reconstruction fidelity was evaluated on 500 lymph-node surfaces with complete measurements at all six SH degrees. For each reconstruction, four complementary geometric errors were calculated relative to the processed reference surface: normalized symmetric mean nearest-neighbor distance, normalized symmetric HD95, relative surface-area error, and relative volume error.

Because these measures have diferent numerical scales, each error was converted to a pooled empirical fidelity score. Let $e _ { i L q }$ denote geometric error metric � for lymph node � reconstructed at degree �, and let $r _ { i L q }$ denote its rank among all lymph-node–degree observations for that metric, with smaller errors assigned lower ranks. The corresponding fidelity score was

$$
F _ { i L q } = 1 - \frac { r _ { i L q } - 1 } { N _ { \mathrm { p o o l } } - 1 } ,\tag{32}
$$

where $N _ { \mathrm { p o o l } } ~ = ~ 3 0 0 0$ for the 500 lymph-node surfaces evaluated at six degrees. The composite geometric fidelity score was defined as

$$
F _ { i L } = \frac { 1 } { 4 } \sum _ { q = 1 } ^ { 4 } F _ { i L q } .\tag{33}
$$

Larger values indicate better overall agreement with the processed reference geometry. The score is a cohort-relative summary and should not be interpreted as an absolute physical measure of reconstruction error. Degree-specific medians and 95% confidence intervals were estimated using 5, 000 bootstrap resamples.

Because predictive optimization and faithful geometric representation need not favor the same SH degree, degree 34 was additionally examined as the high-resolution representation. Preservation of conventional morphology was assessed using Pearson correlation, Spearman correlation, and Lin’s concordance correlation coeficient across the 14 PyRadiomics shape features. Representation eficiency was evaluated by comparing the 1, 225 three-dimensional SH coeficient vectors at degree 34 with the number of vertices in the processed reference meshes. This comparison was interpreted as a reduction in subject-specific geometric representation elements rather than as a direct estimate of runtime or byte-level storage eficiency.

## 3.4.5. Fixed-resolution andfamily-specific mixed-resolution prediction

Predictive modeling was performed on all 1, 769 lymph nodes from 214 patients. At each fixed SH degree, all 366 candidate variables from the 20 predefined feature families were used to construct a fixed-resolution model. A familyspecific mixed-resolution model was constructed to allow each feature family to select its preferred SH degree using training data only.

All headline models used three repeats of five-fold patient-grouped stratified outer cross-validation, with identical outer partitions locked across all SH degrees and model comparisons. Within each outer training set, five-fold patientgrouped stratified inner cross-validation was used for degree selection and regularization tuning. The modeling pipeline consisted of median imputation with missing-value indicators, feature standardization, and class-balanced L2-regularized logistic regression. The regularization parameter was selected from $C \in \{ 0 . 0 1 , 0 . 1 , 1 , 1 0 ,$ , 100} according to mean innercross-validation AUC.

For the mixed-resolution model, degree selection was performed separately for each of the 20 feature families within each outer training set. For family �, models were evaluated at $L ~ \in ~ \{ 5 , 8 , 1 0 , 1 5 , 2 0 , 3 4 \}$ , and the degree producing the highest mean inner-cross-validation AUC was selected. Efectively exact ties, defined by a tolerance of $1 0 ^ { - 1 2 }$ , were resolved in favor of the lower degree. The feature blocks corresponding to the 20 selected family-specific degrees were then concatenated, after which the regularization parameter of the combined model was tuned again using inner crossvalidation. The held-out outer fold was used only for final prediction. Fixed-resolution comparators used the same 366 variables, modeling pipeline, and locked outer-fold assignments, difering only in that every feature family was represented at the same SH degree.

For each lymph node, the out-of-fold probabilities obtained from the three outer repeats were averaged before calculation of the final node-level AUC. Confidence intervals and paired model diferences were estimated using 2,000 stratified patient-cluster bootstrap resamples, with patients serving as the resampling units and all lymph nodes from each selected patient retained jointly.

The original-surface Py14 model was evaluated on 1,769 lymph nodes from 214 patients as the conventional global-morphology baseline. Its lymph-node-level out-offold predictions were compared with those of the mixedresolution model using paired bootstrap resampling.

## 3.5. Resolution-dependent perturbation robustness

To investigate how SH resolution afects sensitivity to small geometric variation, we performed a controlled perturbation analysis at geometric, harmonic, and predictive levels. A single perturbation realization was generated for each processed lymph node surface and propagated through all six SH reconstruction degrees: $L \in \{ 5 , 8 , 1 0 , 1 5 , 2 0 , 3 4 \}$ Complete paired data were available for a randomly selected subset of 324 lymph-node surfaces throughout the robustness analysis. Each degree retained its own degree-specific spherical parameterization selected from the corresponding clean surface; therefore, between-degree diferences reflect the combined behavior of the lower- or higher-resolution SH representation and its clean-selected mapping, rather than spectral truncation alone.

## 3.5.1. Controlled geometric perturbation

Let $\mathbf { v } _ { i }$ and ${ \bf n } _ { i }$ denote the position and outward unit normal of vertex �, respectively, and let $( s _ { x } , s _ { y } , s _ { z } )$ denote the voxel spacing in millimetres. To account for anisotropic image resolution, the physical scale corresponding to a one-voxel displacement along the local surface normal was defined as

$$
h _ { i } = \left[ \left( \frac { n _ { i , x } } { s _ { x } } \right) ^ { 2 } + \left( \frac { n _ { i , y } } { s _ { y } } \right) ^ { 2 } + \left( \frac { n _ { i , z } } { s _ { z } } \right) ^ { 2 } \right] ^ { - 1 / 2 } .\tag{34}
$$

A spatially correlated scalar perturbation field was generated from independent standard-normal samples � using cotangent finite-element heat difusion,

$$
( \mathbf { M } + t \mathbf { L } ) \mathbf { z } _ { 0 } = \mathbf { M } \varepsilon , \qquad t = \frac { \rho ^ { 2 } } { 2 } ,\tag{35}
$$

where � is the lumped mass matrix and � is the positivesemidefinite cotangent Laplacian. The correlation length was fixed a priori at $\rho _ { \mathrm { v o x } } = 1 . 7 5$ voxels for all lymph-node surfaces. The corresponding physical correlation length for each case was defined as

$$
\rho = \rho _ { \mathrm { v o x } } ( s _ { x } s _ { y } s _ { z } ) ^ { 1 / 3 } .\tag{36}
$$

The difused field was standardized using the lumped mass weights to have zero weighted mean and unit weighted root-mean-square magnitude. Extreme field values were then symmetrically clipped at $\pm 2 . 5$ standard deviations, followed by a second weighted standardization.

The normal displacement at vertex � was then defined as

$$
d _ { i } = \alpha h _ { i } z _ { i } ,\tag{37}
$$

and the perturbed vertex position was

$$
\mathbf { v } _ { i } ^ { \mathrm { p e r t } } = \mathbf { v } _ { i } + d _ { i } \mathbf { n } _ { i } .\tag{38}
$$

The perturbation amplitude was fixed at $\alpha = 0 . 2$ . The same standardized random field was used for all SH degrees of a given lymph node, thereby preventing degree-specific random perturbations from confounding the comparison. Perturbed surfaces retained the connectivity and vertex correspondence of the corresponding clean surfaces.

Candidate perturbations were accepted only when all afected triangles retained positive area according to the numerical validity criterion used in the implementation and no face-orientation reversal occurred. If these conditions were violated, the candidate was discarded and a new seeded realization was generated.

For each clean–perturbed pair, the spherical parameterization selected from the clean surface was kept fixed and directly applied to the perturbed surface without reestimation. Therefore, diferences between the clean and perturbed representations within each SH degree reflected only the efect of the imposed geometric perturbation, rather than changes caused by remapping. Comparisons across SH degrees used the corresponding degree-specific mappings selected from the clean surfaces, consistent with the main analysis.

## 3.5.2. Geometric perturbation transmission across SH degrees

Geometric perturbation transmission was quantified by comparing the RMS displacement before and after SH reconstruction at corresponding spherical sample locations.

For degree �, the perturbation transmission ratio was defined as

$$
T _ { L } = \frac { \mathrm { R M S } \left( \mathbf { x } _ { L } ^ { \mathrm { p e r t } } - \mathbf { x } _ { L } ^ { \mathrm { c l e a n } } \right) } { \mathrm { R M S } \left( \mathbf { x } ^ { \mathrm { p e r t } } - \mathbf { x } ^ { \mathrm { c l e a n } } \right) } ,\tag{39}
$$

where $\mathbf { x } ^ { \mathrm { c l e a n } }$ and $\mathbf { x } ^ { \mathrm { p e r t } }$ denote corresponding points on the clean and perturbed input surfaces, and the subscript � denotes their SH reconstructions at degree �.

Thus, $T _ { L } = 1$ indicates complete transmission of the imposed perturbation, whereas $T _ { L } < 1$ indicates attenuation.

## 3.5.3. Harmonic-domain perturbation analysis

To characterize the spectral origin of resolution-dependent perturbation transmission, the degree-34 SH representation was used as a common high-resolution reference. A single clean-selected degree-34 spherical parameterization was used for both the clean and perturbed surfaces so that the spectral comparison was not confounded by diferences in spherical mapping. Clean and perturbed surfaces were fitted independently using the fixed parameterization and identical spherical-area quadrature weights.

Let $\mathbf { c } _ { \ell m } ^ { \mathrm { { c l e a n } } }$ and $\mathbf { c } _ { \ell m } ^ { \mathrm { p e r t } }$ denote the clean and perturbed SH coeficient vectors at harmonic degree $\ell$ and order �. Their coeficient-space diference was

$$
\Delta \mathbf { c } _ { \ell m } = \mathbf { c } _ { \ell m } ^ { \mathrm { p e r t } } - \mathbf { c } _ { \ell m } ^ { \mathrm { c l e a n } } ,\tag{40}
$$

and the perturbation energy at degree � was defined as

$$
E _ { \ell } ^ { \Delta } = \sum _ { m = - \ell } ^ { \ell } \left\| \Delta \mathbf { c } _ { \ell m } \right\| _ { 2 } ^ { 2 } .\tag{41}
$$

Before calculation of spectral energies, SH coeficients were normalized by the equivalent-radius scale of the corresponding clean surface to reduce diferences arising solely from overall lymph node size. Degree 0 was excluded because it represents the constant translation component rather than shape variation.

The relative contribution of degree � to the total perturbation spectrum was

$$
P _ { \ell } ^ { \Delta } = E _ { \ell } ^ { \Delta } / ( \sum _ { k = 1 } ^ { 3 4 } E _ { k } ^ { \Delta } ) , \qquad \ell = 1 , \dots , 3 4 .\tag{42}
$$

The cumulative fraction of degree-34 perturbation energy retained after truncation at degree � was defined as

$$
R _ { L } = \left( \sum _ { \ell = 1 } ^ { L } E _ { \ell } ^ { \Delta } \right) \bigg / \left( \sum _ { \ell = 1 } ^ { 3 4 } E _ { \ell } ^ { \Delta } \right) .\tag{43}
$$

Accordingly, $1 - R _ { L }$ represents the fraction of coeficientspace perturbation energy excluded by truncation at degree $L .$

Because absolute perturbation energy does not indicate its magnitude relative to the geometric signal represented at the same spectral scale, we additionally calculated the perturbation-to-clean energy ratio

$$
Q _ { \ell } = \frac { E _ { \ell } ^ { \Delta } } { E _ { \ell } ^ { \mathrm { c l e a n } } } , \qquad E _ { \ell } ^ { \mathrm { c l e a n } } = \sum _ { m = - \ell } ^ { \ell } \left\| \mathbf { c } _ { \ell m } ^ { \mathrm { c l e a n } } \right\| _ { 2 } ^ { 2 } .\tag{44}
$$

Thus, $R _ { L }$ quantifies the cumulative fraction of perturbation preserved by truncation at degree $L ,$ whereas $Q _ { \ell }$ quantifies the perturbation burden relative to the clean geometric signal at an individual harmonic degree. In implementation, the denominator of $Q _ { \ell }$ was bounded below by machine epsilon to prevent numerical division by zero. Values of $Q _ { \ell } > 1$ indicate that perturbation energy exceeds the clean-shape energy represented at the same spectral degree. Because $Q _ { \ell }$ is normalized by degree-specific clean energy, it was interpreted jointly with $R _ { L }$ rather than as a measure of absolute perturbation magnitude.

## 3.5.4. Predictive robustness of curvature-derived features

To assess whether resolution-dependent perturbation transmission afected downstream geometric analysis, we evaluated curvature-based prediction using 92 descriptors consistently available for both clean and perturbed reconstructions at all six SH degrees across the 324 lymph-node surfaces. These descriptors comprised mean curvature (26), Gaussian curvature (13), Shape Index and Curvedness (28), curvature integrals (9), and curvature regions (16), with identical feature definitions and numerical settings applied to the paired surfaces.

The patient-grouped modeling framework described in Section 3.4.5 was retained. For each outer split, preprocessing, regularization selection, and model fitting were performed using only the clean training data. The fitted model was then applied without refitting to both the clean and perturbed versions of the held-out lymph nodes, thereby isolating perturbation-induced changes in prediction. The change in discriminative performance was defined as

$$
\Delta \mathrm { A U C } _ { L } = \mathrm { A U C } _ { L } ^ { \mathrm { p e r t } } - \mathrm { A U C } _ { L } ^ { \mathrm { c l e a n } } ,\tag{45}
$$

where negative values indicate reduced discriminative performance after perturbation.

Prediction-level stability was additionally quantified using the mean absolute probability shift,

$$
\Delta p _ { L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \hat { p } _ { i , L } ^ { \mathrm { p e r t } } - \hat { p } _ { i , L } ^ { \mathrm { c l e a n } } \right| ,\tag{46}
$$

together with the Pearson correlation $r _ { L }$ between paired clean and perturbed prediction probabilities. Larger $\Delta p _ { L }$ and lower $r _ { L }$ indicate reduced predictive stability.

Confidence intervals for $\Delta \mathrm { A U C } _ { L }$ and pairwise comparisons of AUC degradation between SH degrees were obtained using paired patient-level bootstrap resampling, with all lymph nodes from the same patient resampled jointly, followed by Benjamini–Hochberg correction for multiple comparisons.

## 3.6. Independent external and cross-organ validation

## 3.6.1. Multicenter label-free lymph node replication

All 2,644 external-cohort surfaces underwent the complete preprocessing pipeline and shape analysis described previously (Sections 2 and 3), as well as topology validation (Section 3.4.1). Subsequently, a simple random sample of 1,000 topology-eligible lymph nodes was drawn without replacement for the computational tests described in Section 3.4.1. The external replication analysis of degreedependent feature efects was conducted separately using the complete cohort of 2,644 lymph nodes.

Because pathological labels were unavailable, the external cohort was used to test replication of degree-dependent geometric efects rather than clinical prediction performance.

The internal cohort comprised 1,769 lymph nodes from 214 unique patients. All lymph nodes from the same patient were assigned to the same cross-validation fold and were resampled jointly in patient-level bootstrap analyses. The analysis was restricted a priori to the three geometric feature families available in both cohorts: convexity (6 features), mean curvature (26 features), and central asymmetry (10 features). SH-energy features were not included because the corresponding feature set was not available for matched external replication.

The degree set was fixed before examination of the external cohort. It was selected to cover the dominant familyspecific degrees identified in the leakage-safe internal modelselection analysis: $L = 5$ for convexity, $L = 1 0$ for mean curvature, and $L = 3 4$ for central asymmetry. In the internal analysis, degree selection was performed using inner-crossvalidation AUC within each outer training fold; these values were the modal selected degrees for their respective feature families. To enable matched cross-cohort evaluation, all three feature families were recomputed at each of the common degrees, $L = 5 , L = 1 0 ,$ and $L = 3 4$ , in both cohorts. The external cohort was therefore used to assess replication of prespecified degree-dependent geometric signatures rather than to re-select an optimal degree.

Within each cohort, measurements obtained at diferent reconstruction degrees were treated as repeated observations from the same lymph node. For each feature with complete measurements at all three degrees, an omnibus Friedman test was used to assess degree dependence. Pairwise degree efects were estimated for the three prespecified contrasts: $L \ = \ 5$ versus $L ~ = ~ 1 0 , ~ L ~ = ~ 1 0$ versus $L \ = \ 3 4 .$ and $L = 5$ versus $L = 3 4 .$ , using two-sided Wilcoxon signed-rank tests. For each contrast, the paired diference was defined as the value at the first degree minus the value at the second degree. Efect magnitude and direction were summarized using the rank-biserial correlation. Median paired diferences were accompanied by 95% percentile bootstrap confidence intervals based on 10,000 resamples. Benjamini–Hochberg false-discovery-rate correction was applied separately within each cohort across the feature-level omnibus tests and across the pairwise tests.

Each internal degree efect was matched to its external counterpart by feature family, feature identity, and degree contrast. Concordance between the internal and external rank-biserial efect sizes was quantified using Spearman’s correlation coeficient, with 95% percentile bootstrap confidence intervals obtained from $1 0 { , } 0 0 0$ resamples of the matched efect pairs. Directional replication was summarized as the proportion of matched non-zero efects having the same sign in both cohorts; a two-sided exact binomial test against a probability of 0.5 was used as a supplementary assessment. Family-specific concordance analyses were considered primary, whereas the pooled analysis across all matched efects was considered secondary because the three families contributed unequal numbers of efects. This featurelevel replication framework was consistent with established approaches for assessing replicability across multiple features while controlling multiplicity (Bogomolov and Heller, 2018).

To directly assess transferability of the internally identified family-specific degree preferences, we performed a prespecified directional composite analysis. For each feature � and each alternative degree $L ,$ the expected direction was fixed from the internal cohort as

$$
\begin{array} { r } { s _ { f , j , L } = \mathrm { s i g n } \left( \widehat { \Delta } _ { f , j , L _ { f } ^ { * } - L } ^ { \mathrm { i n t e r n a l } } \right) , } \end{array}\tag{47}
$$

where $L _ { f } ^ { * }$ denotes the internally preferred degree for feature family $\breve { f } .$ The external paired contrast was then directionaligned and scaled using internal reference variability:

$$
D _ { i , f , j , L } = s _ { f , j , L } \frac { x _ { i , f , j , L _ { f } ^ { * } } - x _ { i , f , j , L } } { \mathrm { M A D } _ { f , j } ^ { \mathrm { i n t e r n a l } } } ,\tag{48}
$$

where $x _ { i , f , j , L }$ is the feature value for lymph node �, and the median absolute deviation was calculated exclusively from the internal cohort across the three common degrees. Thus, neither the direction, scaling, nor preferred degree was determined using external data.

For each lymph node and family, the aligned featurelevel contrasts were aggregated using their median:

$$
C _ { i , f , L } = \mathrm { m e d i a n } _ { j \in f } D _ { i , f , j , L } .\tag{49}
$$

A positive value of $C _ { i , f , L }$ indicates that the external node supports the internally specified preferred degree relative to the alternative degree. For each family–contrast pair, the lymph-node-level composites were evaluated directly. The composite median and its 95% percentile confidence interval were estimated using 10,000 nonparametric bootstrap resamples of lymph nodes. The positive-node percentage was defined as the proportion of lymph nodes for which $C _ { i , f , L } > 0$ . One-sided Wilcoxon signed-rank tests against zero were used to test whether the family-level composite was directionally positive. Holm correction was applied across the six prespecified family–contrast tests. This composite analysis was intended to test transferability of the internal degree preference, not equivalence of efect magnitude between cohorts.

## 3.6.2. Cross-organ validation in LIDC-IDRI lung nodules

The LIDC-IDRI cohort was analyzed as an independent cross-organ experiment to determine whether the resolutiondependent relationship between SH representation and downstream discriminative performance extended beyond axillary lymph node morphology. SH-based three-dimensional shape analysis has previously been applied to lung-nodule malignancy discrimination (El-Baz et al., 2011); here, the purpose of the LIDC-IDRI experiment was specifically to evaluate whether predictive performance remained dependent on SH reconstruction resolution in a distinct anatomical structure and classification task. For the supervised classification task, only nodules with definite diagnostic labels were considered. Label 1 nodules (benign/non-malignant) and label 2 nodules (primary malignant) were retained, whereas label 0 nodules with unknown diagnosis and label 3 metastatic lesions were excluded to maintain a clinically well-defined binary classification task. After further requiring usable CT data and corresponding segmentation contours suitable for threedimensional surface reconstruction, 59 lung nodules from 56 unique patients were retained for analysis. Of these, 50 satisfied the strict topological criteria before geometric processing, while all 59 were retained for application of the complete preprocessing pipeline.

Following geometric processing and spherical parameterization, each lung-nodule surface was reconstructed using the prespecified SH degrees $L ~ \in ~ \{ 5 , 1 0 , 2 0 , 3 0 , 4 0 , 5 0 \}$ Degree-specific representations were generated independently using the SH reconstruction framework described in Section 3.2.

For each SH degree, the same predefined featureextraction procedure was applied to three geometric feature families: convexity, mean curvature, and central asymmetry. Degree-specific classifiers followed the same preprocessing and L2-regularized logistic-regression framework described in Section 3.4.5, with identical feature definitions and data partitions across all six degrees. Thus, SH reconstruction degree was the principal experimental factor difering among the compared representations.

Predictive performance was evaluated using five-fold outer cross-validation repeated three times, with all nodules from the same patient assigned to the same fold. Within each outer training set, the regularization parameter was selected by three-fold inner patient-grouped cross-validation using the same candidate grid described in Section 3.4.5. Identical outer partitions were retained across all six SH degrees.

Out-of-fold probabilities from the three outer-crossvalidation repeats were averaged for each nodule. For each SH degree, discriminative performance was summarized using the area under the receiver-operating-characteristic curve (AUC). Ninety-five percent confidence intervals were estimated using 2,000 bootstrap resamples of the averaged out-of-fold predictions. Degree-specific AUC diferences were assessed using paired bootstrap resampling with 2,000 iterations.

Table 2  
Topological validity before and after geometric processing.
<table><tr><td>Criterion</td><td>Original</td><td>Processed</td><td>Change (pp)</td></tr><tr><td>Strict topology pass</td><td> $\overline { { 1 3 8 3 / 1 7 6 9 \left( 7 8 . 1 8 \% \right) } }$ </td><td> $\overline { { 1 7 6 9 / 1 7 6 9 \left( 1 0 0 . 0 0 \% \right) } }$ </td><td>+21.82</td></tr><tr><td>Single-component surface</td><td>1531/1769 (86.55%)</td><td>1769/1769 (100.00%)</td><td>+13.45</td></tr><tr><td>Genus-0 topology</td><td>1537/1769 (86.89%)</td><td>1769/1769 (100.00%)</td><td> $+ 1 3 . 1 1$ </td></tr><tr><td>Outward orientation</td><td>1560/1769 (88.19%)</td><td>1769/1769 (100.00%)</td><td>+11.81</td></tr></table>

Downstream computational surface quality before and after geometric processing (internal cohort). Values are medians across 500 paired surfaces.
<table><tr><td>Measure</td><td>Original</td><td>Processed</td><td>Median paired change (95% CI)</td><td>FDR-adjusted P</td></tr><tr><td>Triangle quality (5th percentile)</td><td>0.7274</td><td>0.7787</td><td> $+ 0 . 0 4 5 3 \ ( 0 . 0 4 1 3 - 0 . 0 5 0 6 )$ </td><td> $\overline { { 2 . 2 9 \times 1 0 ^ { - 5 4 } } }$ </td></tr><tr><td>Normal jump (95th percentile, degrees)</td><td>38.7617</td><td>4.6759</td><td> $- 3 3 . 3 1 9 0 \ \left( - 3 3 . 9 1 9 6 -- 3 2 . 7 3 6 0 \right)$ </td><td> $5 . 0 6 \times 1 0 ^ { - 8 3 }$ </td></tr></table>

This cross-organ replication analysis evaluated whether the resolution-dependent relationship between SH representation and predictive performance generalized to a distinct anatomical structure and classification task.

## 4. Results

## 4.1. Geometric processing establishes a valid computational surface domain

Strict topology validity increased from 78.18% (1383 /1769) before processing to 100.00% (1769/1769) afterward, with all reported component criteria reaching 100% (Table 2). In the 500-surface computational-quality subset, processing increased fifth-percentile triangle quality and markedly reduced the 95th-percentile adjacent-face normal jump (Table 3). These results show that the processing workflow established a topology-valid and more regular computational surface domain for subsequent spherical and diferential-geometric analysis.

## 4.2. Multi-resolution spherical harmonic analysis 4.2.1. Conventional shape descriptors maintain stable discrimination across SH representations

Changes in univariate Py14 discrimination were small across most SH resolutions (Fig. 5). Median descriptorlevel ΔAUC remained close to zero from degrees 5 to 20, with a modest reduction at degree 34. Descriptor-specific responses were bidirectional, and Sphericity showed the largest degree dependence. Overall, conventional globalmorphology discrimination was largely preserved after SH reconstruction.

## 4.2.2. Degree-wise spectral discrimination is strongest at intermediate harmonic degrees

Degree-wise spectral analysis (Fig. 6) showed that relative spectral energy at intermediate harmonic degrees had the strongest univariate association with metastatic status. Discrimination was weak in the lowest harmonic components, increased progressively toward the intermediate-frequency range, and reached its strongest level around degrees 8– 12. The strongest individual degree was $\ell \ = \ 1 0 .$ , with a folded AUC of $\mathrm { A U C ^ { * } } = 0 . 6 4 5$ , while similar discrimination was observed at degrees 8 and 9. Beyond this intermediate range, degree-wise discrimination progressively declined, with folded AUC decreasing to 0.591 at degree 15 and 0.536 at degree 20 and approaching chance level in the highest harmonic components.

## 4.2.3. Geometricfidelity increases monotonically, whereas predictive utility does not

Composite geometric fidelity increased monotonically with SH degree and was highest at $L ~ = ~ 3 4 ~ ( \mathrm { F i g . } ~ 7 \mathrm { A } )$ Predictive performance followed a diferent pattern: fixedresolution model AUC increased from low resolution to the intermediate degrees and subsequently decreased at $L ~ = ~ 3 4 ~ ( \mathrm { F i g . ~ 7 B } )$ . Moreover, recovery of conventional shape measurements showed only a weak association with discriminative strength across feature–degree combinations (Spearman $\rho ~ \approx ~ 0 . 2 5 )$ . Numerical reconstruction fidelity and metastasis discrimination therefore followed distinct resolution-dependent behaviors.

## 4.2.4. Feature-dependent SH resolution preferences

The preferred SH degree varied substantially across the predefined geometric feature families (Fig. 8). Many curvature-related, spectral-energy, and shape-distribution descriptors favored intermediate resolutions, whereas several global and intrinsic descriptors favored lower degrees and selected families favored the highest resolution. These heterogeneous selection patterns demonstrate that the reconstruction scale most informative for prediction depends on the geometric quantity being measured.

The family-specific mixed-resolution model achieved the highest overall predictive performance among the evaluated representations (Table 4). Its performance was comparable to the fixed-resolution models at degrees 8, 10, 15, 20, and 34 after multiple-comparison correction, but significantly exceeded the low-resolution degree-5 model. Family-specific selection therefore primarily avoided clearly mismatched representation scales rather than substantially improving performance beyond the best intermediate fixed resolution.

![](images/33743b92bf49b50817bb45c5c2f12c6c0a563535bc654d13a49a04af2c919c4b.jpg)  
Conventional PyRadiomics shape feature  
Figure 5: Changes in univariate discrimination of conventional PyRadiomics shape descriptors across SH representations. For each descriptor and SH degree, ΔAUC was calculated relative to the corresponding descriptor extracted from the original surface. Positive values indicate higher discrimination after SH reconstruction, whereas negative values indicate lower discrimination.

![](images/6a39ecb423424d9df7e1f1448b227e176bcb6e97233d79c8c8f447937ebd1965.jpg)  
Figure 6: Degree-wise spectral discrimination across the SH spectrum. Folded univariate AUC is shown for relative spectral energy at each harmonic degree, summarizing discrimination irrespective of the direction of association. Discrimination was strongest at intermediate degrees and declined toward chance level at the highest degrees.

Compared with the original-surface Py14 baseline, the mixed-resolution model improved the AUC from 0.884 (95% CI, 0.868–0.900) to 0.918 (95% CI, 0.904–0.934), corresponding to an absolute improvement of 0.0344 (95% paired bootstrap CI, 0.0212–0.0468; � = 0.001). This improvement indicates that the extended geometric descriptors provide discriminative information complementary to conventional global morphology.

![](images/781eb291998825a1c88bdfcdcd4b533abdb2f1952456e67639ea0471d1f04904.jpg)

![](images/101ccec338956aa43279c1e763a498e6d80217ebe076bef105cc32b511d59d48.jpg)  
Figure 7: Divergence between geometric fidelity and predictive performance across SH degrees. (A) Composite geometric fidelity increased monotonically with reconstruction degree. (B) AUC of the fixed-resolution 366-feature models increased from low resolution to intermediate degrees but declined at degree 34. Error bars denote 95% confidence intervals.

Predictive performance of fixed- and mixed-resolution SH representations.
<table><tr><td>Representation</td><td>AUC (95% CI)</td><td>Mixed minus model ∆AUC</td><td>FDR-adjusted P</td></tr><tr><td>SH degree 5</td><td>0.9006 (0.8852-0.9165)</td><td>0.0177</td><td>0.006</td></tr><tr><td>SH degree 8</td><td>0.9120 (0.8966–0.9277)</td><td>0.0064</td><td>0.384</td></tr><tr><td>SH degree 10</td><td>0.9153 (0.9002-0.9304)</td><td>0.0031</td><td>0.588</td></tr><tr><td>SH degree 15</td><td>0.9178 (0.9031-0.9327)</td><td>0.0006</td><td>0.817</td></tr><tr><td>SH degree 20</td><td>0.9178 (0.9028–0.9331)</td><td>0.0006</td><td>0.817</td></tr><tr><td>SH degree 34</td><td>0.9139 (0.8985–0.9288)</td><td>0.0044</td><td>0.504</td></tr><tr><td>Mixed resolution</td><td>0.9183 (0.9037–0.9336)</td><td></td><td></td></tr></table>

## 4.2.5. High-resolution SH achieves compact geometric representation with highfidelity

Despite its lower predictive utility relative to intermediate resolutions, degree 34 provided the closest approximation to the processed reference geometry. In the 500-lymph-node fidelity subset, the median normalized symmetric mean nearest-neighbor distance and HD95 were 0.00887 and 0.01744, respectively, while relative surface-area and volume errors were 0.120% and 0.006%. Agreement of the 14 Py14 descriptors with the reference surfaces was also high (median Pearson, Spearman, and Lin’s concordance correlations: 0.9995, 0.9999, and 0.9995). A degree-34 representation used 1, 225 three-dimensional coeficient vectors compared with a median of 38, 492 mesh vertices, corresponding to a 31.42-fold reduction in representation elements. Thus, highresolution SH remained useful when faithful and structured geometric representation, rather than maximal discrimination, was the primary objective.

![](images/28f5ec7f9d62c69b59ff56dd7e93a89dfaac10a9d7cea2fdf02f07ac6cf4b1ef.jpg)

![](images/dc6eebc3231ae76793814bd83405f3d06e4cd6df636c49a118efc3cc16baaccf.jpg)  
Figure 8: Feature-family-specific SH degree selection. (A) Selection frequencies are shown across the locked outer crossvalidation folds for ten representative feature families; degree selection was performed for all 20 predefined feature families. (B) Distinct descriptor families favored diferent SH reconstruction degrees, demonstrating that a single common resolution was not uniformly optimal across geometric measurements.

## 4.3. Resolution-dependent perturbation robustness

Geometric perturbation transmission increased with SH degree (Fig. 9A), with the median transmission ratio rising from $T _ { 5 } = 0 . 7 9 4$ to $T _ { 3 4 } = 0 . 9 9 2$ . Thus, progressively higherresolution representations retained a larger fraction of the imposed surface displacement.

The harmonic-domain analysis showed the same resolution dependence. Cumulative retained perturbation energy increased with truncation degree (Fig. 9B), while the perturbation-to-clean energy ratio exceeded unity from approximately the intermediate harmonic range and remained elevated at high degrees (Fig. 9C). These results indicate that increasing SH bandwidth retains progressively more perturbation energy and increases its relative contribution at fine spectral scales.

The same pattern propagated to curvature-based prediction (Fig. 9D). Perturbation-induced AUC degradation was significantly greater than that at � = 5 for degrees 10, 15, 20, and 34 after FDR correction. Across the 324 lymph nodes, higher SH degrees produced larger mean absolute probability shifts and lower clean–perturbed prediction correlations.

## 4.4. Metastasis-associated geometric phenotypes

Beyond aggregate model performance, representative descriptors from the interpretable geometric families showed substantial separation between metastatic and nonmetastatic lymph nodes. The strongest examples included lower-tail curvedness and absolute mean-curvature descriptors, Gaussian-curvature statistics, radial morphology, and convexity-related measurements, with univariate AUCs ranging from 0.824 to 0.898. The corresponding Clif’s � values indicated large efect sizes across both global and local geometric properties.

These descriptors characterize diferent manifestations of the nodal surface phenotype. Radial and convexity-related measurements quantify departures from a regular overall contour, whereas curvedness and curvature statistics capture regional variations in surface bending and contour irregularity. Such geometric changes may be consistent with structural remodeling accompanying metastatic involvement, including asymmetric cortical thickening, hilar efacement, or heterogeneous tumor infiltration, which can alter the external contour of the lymph node. Their joint association with metastatic status therefore suggests that relevant morphological information extends beyond global enlargement or compactness to include spatially heterogeneous surface remodeling. Together, these measurements provide an interpretable quantitative description of metastasis-associated nodal morphology complementary to conventional global radiomic shape features.

## 4.5. Independent external and cross-organ validation

## 4.5.1. Multicenter label-free lymph node replication

The multicenter external cohort comprised 2,644 lymph nodes without node-level pathological metastasis labels and was used to assess label-free replication of the geometricprocessing and family-specific degree efects.

Geometric processing established complete strict topology validity in both independent datasets (Table 5). All 2,644 multicenter lymph node surfaces and all 59 LIDC-IDRI lung-nodule surfaces satisfied the predefined topological requirements after processing, with a larger pre-processing deficit observed in the LIDC-IDRI cohort.

Computational-quality measures also improved substantially after geometric processing in both independent datasets (Table 6). Processing increased lower-tail triangle quality and markedly reduced extreme adjacent-face normal variation in both the multicenter lymph node and LIDC-IDRI cohorts, indicating improved mesh regularity across distinct anatomical datasets.

The label-free replication analysis was restricted a priori to the three feature families available in both lymph node cohorts: convexity (6 features), mean curvature (26 features), and central asymmetry (10 features). Their internally specified preferred degrees were � = 5, $L = 1 0$ , and $L = 3 4$ respectively. All three families were therefore recomputed at the common degrees � = 5, � = 10, and $L = 3 4$ in both cohorts.

Table 6  
![](images/a24a7ef40449be6577d7b9aa5af529e032d858d4845117c4c32dd3015de77c08.jpg)

![](images/f2d7686fd5e5da7d00f508402cd8b79c3c9a41f9be6004d2b18bd8c0afeadbc9.jpg)

![](images/2447747ad7775ca56bc461c44aa03c49c883d8b9f3665f57640853faee0d1f5e.jpg)

![](images/867d3ab0287c985bd46d74e82df966008b8c62adc50917accb1880e6472abf49.jpg)  
Figure 9: Resolution-dependent perturbation robustness across SH degrees. (A) Geometric perturbation transmission ratio $T _ { L } .$ (B) Cumulative retained perturbation energy $R _ { L }$ . (C) Perturbation-to-clean spectral energy ratio $Q _ { \ell } . \ ( \mathsf { D } )$ Perturbation-induced change in AUC. Complementary predictive-stability measures included the mean absolute probability shift $\Delta p _ { L }$ and clean–perturbed prediction correlation.

Table 5  
Topological validity before and after geometric processing in the independent external and cross-organ cohorts.
<table><tr><td>Dataset</td><td>Criterion</td><td>Original</td><td>Processed</td><td>Change (percentage points)</td></tr><tr><td rowspan="4">External  $( n = 2 6 4 4 )$ </td><td>Strict topology pass</td><td>2607/2644 (98.60%)</td><td>2644/2644 (100.00%)</td><td>+1.40</td></tr><tr><td>Single-component surface</td><td>2623/2644 (99.21%)</td><td>2644/2644 (100.00%)</td><td>+0.79</td></tr><tr><td>Genus-0 topology</td><td>2628/2644 (99.40%)</td><td>2644/2644 (100.00%)</td><td>+0.61</td></tr><tr><td>Outward orientation</td><td>2644/2644 (100.00%)</td><td>2644/2644 (100.00%)</td><td>+0.00</td></tr><tr><td rowspan="4"> $\mathsf { L I D C - I D R l } \left( n = 5 9 \right)$ </td><td>Strict topology pass</td><td>50/59 (84.75%)</td><td>59/59 (100.00%)</td><td>+15.25</td></tr><tr><td>Single-component surface</td><td>58/59 (98.31%)</td><td>59/59 (100.00%)</td><td>+1.69</td></tr><tr><td>Genus-0 topology</td><td>50/59 (84.75%)</td><td>59/59 (100.00%)</td><td>+15.25</td></tr><tr><td>Outward orientation</td><td>58/59 (98.31%)</td><td>59/59 (100.00%)</td><td>+1.69</td></tr></table>

The external cohort reproduced the family-specific degree-efect signatures observed internally (Table 7). Featurelevel efect sizes showed strong internal–external concordance across all three families, with a pooled Spearman correlation of $\rho = 0 . 9 6 4$ and directional agreement for 121 of 126 matched efects. Both the direction and relative ordering of degree-dependent feature efects were therefore largely preserved across cohorts.

Paired computational-quality results before and after geometric processing in the independent external and cross-organ cohorts. Diferences are processed minus original values; confidence intervals were obtained by paired bootstrap resampling.
<table><tr><td>Dataset</td><td>Metric</td><td>Original median</td><td>Processed median</td><td>Median difference (95% CI)</td><td>FDR-adjusted q</td></tr><tr><td rowspan="2">External (n = 1000)</td><td>Triangle quality (5th percentile)</td><td>0.678</td><td>0.769</td><td>+0.088 (0.082 to 0.097)</td><td> $< 1 0 ^ { - 1 2 3 }$ </td></tr><tr><td>Normal jump (95th percentile, degrees)</td><td>32.74</td><td>3.38</td><td>-29.45 (-29.89 to -29.01)</td><td> $< 1 0 ^ { - 1 6 4 }$ </td></tr><tr><td rowspan="2">LIDC-IDRI (n = 59)</td><td>Triangle quality (5th percentile)</td><td>0.536</td><td>0.753</td><td>+0.178 (0.156 to 0.235)</td><td> $3 . 5 9 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>Normal jump (95th percentile, degrees)</td><td>68.20</td><td>4.92</td><td>-63.16 (-64.67 to -60.38)</td><td> $3 . 5 9 \times 1 0 ^ { - 1 1 }$ </td></tr></table>

Concordance of feature-level degree efects between the internal and independent external cohorts. Matched efects comprise all feature-by-degree-contrast pairs (�5 vs �10, �10 vs �34, and �5 vs �34). Spearman’s � quantifies concordance between internal and external paired rank-biserial efect sizes; direction agreement is the number of matched efects with the same sign.
<table><tr><td>Feature family</td><td>Matched effects</td><td>Spearman ρ (95% CI)</td><td>Direction agreement</td></tr><tr><td>Convexity</td><td>18</td><td>0.872 (0.569–0.987)</td><td>18/18</td></tr><tr><td>Central asymmetry</td><td>30</td><td>0.942 (0.830–0.986)</td><td>30/30</td></tr><tr><td>Mean curvature</td><td>78</td><td>0.979 (0.956–0.988)</td><td>73/78</td></tr><tr><td>Pooled</td><td>126</td><td>0.964 (0.941–0.977)</td><td>121/126</td></tr></table>

The prespecified direction-aligned family composites further supported transfer of the internally identified degree preferences (Table 8). All six family–contrast composites were positive, their 95% bootstrap confidence intervals excluded zero, and all comparisons remained significant after Holm correction. The strength of replication nevertheless difered across feature families, with the weakest separation observed for central asymmetry between � = 34 and � = 10.

Together, these results support the transportability of the geometric-processing framework and the family-specific multi-resolution efects to the independent multicenter lymph node cohort.

## 4.5.2. Family-specificfavored SH degrees in LIDC-IDRI lung nodules

The labeled LIDC-IDRI experiment showed featurefamily-specific dependence of predictive performance on SH degree (Table 9). Convexity achieved its highest observed AUC at $L \ = \ 5 ,$ , whereas mean curvature and central asymmetry favored $L ~ = ~ 4 0$ and $L \ = \ 3 0 .$ , respectively. Despite the distinct anatomy and classification endpoint, predictive utility therefore remained resolution-dependent, and the favored degree varied across feature families.

## 5. Discussion

This study shows that SH degree should be interpreted as a task-dependent geometric scale rather than a resolution parameter to be maximized uniformly. Across geometric fidelity, metastasis discrimination, perturbation robustness, and cross-organ prediction, diferent analytical objectives and anatomical settings favored diferent spectral resolutions.

## 5.1. Reliable surface representation enables quantitative geometric analysis

The increase in strict topology validity to 100%, together with the marked improvement in triangle regularity and adjacent-face normal consistency, demonstrates that preprocessing is not merely a cosmetic smoothing step but establishes the computational domain required for subsequent spherical and diferential-geometric analysis. This distinction is important because a nominal genus-0 classification alone does not guarantee the connectedness, manifoldness, orientation consistency, or local numerical regularity required by geometric operators (Brechbühler et al., 1995; Choi et al., 2015; Meyer et al., 2003). The reproduction of these improvements in the independent multicenter cohort further indicates that the processing procedure was not specific to the internal data source.

## 5.2. SH resolution defines a task-dependent geometric scale

The divergence between reconstruction fidelity and discrimination is central to the interpretation of SH resolution. Increasing degree systematically improved geometric recovery, as expected from the hierarchical structure of the SH basis (Kazhdan et al., 2003), but the strongest metastatic-status discrimination occurred at intermediate resolutions. Highfrequency geometric content therefore cannot be assumed to be uniformly task-informative. In this setting, spectral truncation is better interpreted as scale selection than as simple information loss: low degrees may omit diseaseassociated regional morphology, whereas high degrees may increasingly retain fine contour variations arising from both anatomy and image- or segmentation-related variability. Intermediate resolutions may consequently provide a more favorable balance between morphological information and fine-scale uncertainty.

The preferred scale also depended on the geometric quantity being measured. Global morphology, curvature, intrinsic geometry, spatial distributions, and symmetry characterize diferent aspects of surface organization and therefore need not respond similarly to spectral bandwidth. Accordingly, the family-specific selection results support incorporating representation scale into feature construction rather than imposing a single global reconstruction degree. The mixed-resolution model was comparable to the best intermediate fixed-degree models, suggesting that familyspecific selection primarily avoids mismatched representation scales across heterogeneous geometric measurements.

Previous medical applications have demonstrated the utility of SH-derived shape information for lung-nodule and cardiac classification (El-Baz et al., 2011; Valizadeh et al., 2021). The LIDC-IDRI experiment extended this observation to a distinct anatomical structure and classification task. Despite substantial diferences between lung nodules and axillary lymph nodes, predictive performance remained dependent on SH reconstruction degree. These results further support a task-dependent interpretation of spectral resolution, with the preferred bandwidth potentially varying with anatomy, feature definition, and downstream objective.

## 5.3. Resolution-dependent perturbation sensitivity and the fidelity–robustness trade-of

The perturbation experiments provide a complementary explanation for the resolution-dependent behavior of SH representations. Spectral truncation attenuated a larger fraction of the imposed geometric variation at lower degrees, whereas higher degrees preserved progressively more of the perturbation in both the reconstructed geometry and harmonic coeficients. The same trend propagated to curvature-derived predictions, whose stability decreased as resolution increased. Because local diferential quantities are particularly responsive to fine-scale surface variation, the additional bandwidth that improves high-resolution geometric fidelity can simultaneously increase sensitivity to perturbations.

External replication of internally specified family-specific degree preferences. Composite contrasts were direction-aligned using feature-specific directions defined in the internal cohort and scaled using internal median absolute deviations. Positive values support the internally preferred degree relative to the alternative degree. Confidence intervals were obtained using 10,000 nonparametric bootstrap resamples of lymph nodes. Positive lymph nodes are those with a direction-aligned composite greater than zero. One-sided � values were adjusted across the six prespecified family–contrast tests using the Holm procedure.
<table><tr><td>Feature family</td><td>Preferred L</td><td>Alternative L</td><td>Features</td><td>Composite median</td><td>95% Cl</td><td>Positive lymph nodes (%)</td><td>One-sided P</td><td>Holm-adjusted P</td></tr><tr><td>Convexity</td><td>5</td><td>10</td><td>6</td><td>0.116</td><td>0.110-0.122</td><td>98.6</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Convexity</td><td>5</td><td>34</td><td>6</td><td>0.223</td><td>0.213-0.232</td><td>99.2</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Mean curvature</td><td>10</td><td>5</td><td>26</td><td>0.216</td><td>0.210-0.220</td><td>98.8</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Mean curvature</td><td>10</td><td>34</td><td>26</td><td>0.970</td><td>0.943-1.004</td><td>99.7</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Central asymmetry</td><td>34</td><td>5</td><td>10</td><td>0.166</td><td>0.154-0.174</td><td>79.4</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Central asymmetry</td><td>34</td><td>10</td><td>10</td><td>0.024</td><td>0.021-0.028</td><td>65.4</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr></table>

Table 9

Observed family-specific SH-degree performance in the LIDC-IDRI cross-organ experiment. Values are shown as AUC with 95% confidence intervals. Paired ΔAUC values denote the diferences between the highest and second-highest observed AUCs and were calculated using the same out-of-fold predictions.
<table><tr><td>Feature family</td><td>Highest observed degree</td><td>AUC (95% CI)</td><td>Paired ΔAUC (95% CI)</td></tr><tr><td>Convexity</td><td> $L = 5$ </td><td>0.853 (0.731–0.945)</td><td>0.013 (-0.073–0.093)</td></tr><tr><td>Mean curvature</td><td> $L = 4 0$ </td><td>0.883 (0.788–0.964)</td><td>0.063 (-0.012–0.142)</td></tr><tr><td>Central asymmetry</td><td> $L = 3 0$ </td><td>0.745 (0.606–0.879)</td><td>0.002 (-0.034–0.042)</td></tr></table>

Higher geometric fidelity and lower perturbation attenuation are two consequences of increased representational bandwidth. Preserving finer geometry necessarily reduces spectral suppression of small-scale variation. Consequently, the appropriate SH degree depends on both the level of geometric detail to be retained and the stability requirements of the downstream measurement.

## 5.4. Limitations and future directions

The present framework has two main limitations. First, it relies on the fidelity of CT-derived surface reconstruction. Finite spatial resolution, anisotropic sampling, partialvolume efects, and segmentation variability may introduce geometric uncertainty that cannot be fully separated from true anatomical variation. This issue is particularly relevant for curvature-based descriptors, which are sensitive to local geometric perturbations and discretization efects (Meyer et al., 2003; Gatzke and Grimm, 2006). Although the proposed preprocessing improves mesh validity and numerical stability, it cannot recover anatomical information absent from the original imaging data.

Second, SH degree selection provides only an indirect mechanism for controlling the trade-of between geometric fidelity and robustness. Lower-degree representations suppress high-frequency variations but may remove informative geometry, whereas multi-resolution representations increase computational cost. Future work should investigate diferentialresponse-based representations that directly characterize and regulate the sensitivity of geometric measurements to surface perturbations.

## 6. Conclusion

We developed an integrated framework for topologyaware processing and multi-resolution geometric analysis of CT-derived axillary lymph node surfaces. The proposed processing pipeline established a connected, closed, consistently oriented genus-0 computational domain for spherical analysis while improving local mesh quality and limiting unnecessary modification of the underlying geometry. This provides a reproducible geometric basis for applying diferentialgeometric and spectral descriptors to clinically derived threedimensional surfaces.

The principal finding is that SH resolution should be regarded as a task-dependent geometric scale rather than as a reconstruction parameter that should be maximized uniformly. Increasing SH degree progressively improved reconstruction fidelity, but metastasis-associated discrimination was strongest at intermediate spectral scales, and diferent geometric feature families favored diferent resolutions. The family-specific mixed-resolution representation accommodated these heterogeneous scale preferences and captured discriminative information beyond conventional global shape descriptors, while high-resolution SH remained valuable for compact and faithful representation of detailed surface geometry. Thus, the resolution that best preserves the observed surface need not be the resolution that is most informative for a downstream prediction task.

Controlled perturbation analysis further showed that SH resolution also governs the transmission of fine-scale geometric variation: higher-degree representations preserved a larger fraction of the imposed perturbation, whereas spectral truncation provided greater attenuation and correspondingly greater stability of curvature-based predictions. Together, these findings distinguish geometric fidelity, discriminative utility, and perturbation robustness as complementary but non-equivalent objectives of quantitative shape representation. The proposed framework therefore provides a principled basis for selecting geometric resolution according to the downstream analytical objective and ofers a general strategy for multi-scale quantitative characterization of lymph node morphology and other clinically derived three-dimensional anatomical surfaces.

## CRediT authorship contribution statement

Zixi Yi: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing – original draft. Limeng Qu: Data curation, Methodology, Writing – review & editing. Gary P. T. Choi: Conceptualization, Funding acquisition, Investigation, Methodology, Project administration, Supervision, Writing – review & editing.

## References

Armato III, S.G., McLennan, G., Bidaut, L., McNitt-Gray, M.F., Meyer, C.R., Reeves, A.P., Zhao, B., Aberle, D.R., Henschke, C.I., Hofman, E.A., et al., 2011. The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans. Medical Physics 38, 915–931. doi:10.1118/1.3528204.

Aubry, M., Schlickewei, U., Cremers, D., 2011. The wave kernel signature: A quantum mechanical approach to shape analysis, in: 2011 IEEE International Conference on Computer Vision Workshops, pp. 1626– 1633. doi:10.1109/ICCVW.2011.6130444.

Bogomolov, M., Heller, R., 2018. Assessing replicability of findings across two studies of multiple features. Biometrika 105, 505–516. doi:10.1093/biomet/asy029.

Brechbühler, C., Gerig, G., Kübler, O., 1995. Parametrization of closed surfaces for 3-D shape description. Computer Vision and Image Understanding 61, 154–170. doi:10.1006/cviu.1995.1013.

Cevidanes, L.H., Tucker, S., Styner, M., Kim, H., Chapuis, J., Reyes, M., Profit, W., Turvey, T., Jaskolka, M., 2010. Three-dimensional surgical simulation. American Journal of Orthodontics and Dentofacial Orthopedics 138, 361–371. doi:10.1016/j.ajodo.2009.08.026.

Chen, J., Zhu, Q., Xie, B., Li, T., 2025. ToPoMesh: Accurate 3D surface reconstruction from CT volumetric data via topology modification. Medical & Biological Engineering & Computing 63, 3083–3098. doi:10 .1007/s11517-025-03381-3.

Choi, G.P.T., Leung-Liu, Y., Gu, X., Lui, L.M., 2020. Parallelizable global conformal parameterization of simply-connected surfaces via partial welding. SIAM Journal on Imaging Sciences 13, 1049–1083. doi:10.1137/19M125337X.

Choi, P.T., Lam, K.C., Lui, L.M., 2015. FLASH: Fast landmark aligned spherical harmonic parameterization for genus-0 closed brain surfaces. SIAM Journal on Imaging Sciences 8, 67–94. doi:10.1137/130950008.

Crane, K., Weischedel, C., Wardetzky, M., 2013. Geodesics in heat: A new approach to computing distance based on heat flow. ACM Transactions on Graphics 32, 152. doi:10.1145/2516971.2516977.

Desbrun, M., Meyer, M., Schröder, P., Barr, A.H., 1999. Implicit fairing of irregular meshes using difusion and curvature flow, in: Proceedings of the 26th Annual Conference on Computer Graphics and Interactive

Techniques, ACM Press/Addison-Wesley Publishing Co.. pp. 317–324. doi:10.1145/311535.311576.

Dialani, V., Westra, C., Venkataraman, S., Fein-Zachary, V., Brook, A., Mehta, T., 2018. Indications for biopsy of imaging-detected intramammary and axillary lymph nodes in the absence of concurrent breast cancer. The Breast Journal 24, 869–875. doi:10.1111/tbj.13009.

Edelsbrunner, H., Mücke, E.P., 1994. Three-dimensional alpha shapes. ACM Transactions on Graphics 13, 43–72. doi:10.1145/174462.156635.

El-Baz, A., Nitzken, M., Elnakib, A., Khalifa, F., Gimel’farb, G.L., Falk, R., Abou El-Ghar, M., 2011. 3D shape analysis for early diagnosis of malignant lung nodules, in: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2011, Springer. pp. 175–182. doi:10.1 007/978-3-642-23626-6\_22.

Gatzke, T.D., Grimm, C.M., 2006. Estimating curvature on triangular meshes. International Journal of Shape Modeling 12, 1–28. doi:10 .1142/S0218654306000810.

van Griethuysen, J.J.M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R.G.H., Fillion-Robin, J.C., Pieper, S., Aerts, H.J.W.L., 2017. Computational radiomics system to decode the radiographic phenotype. Cancer Research 77, e104–e107. doi:10.1158/ 0008-5472.CAN-17-0339.

Ito, T., 2019. Efects of diferent segmentation methods on geometric morphometric data collection from primate skulls. Methods in Ecology and Evolution 10, 1972–1984. doi:10.1111/2041-210X.13274.

Kazhdan, M., Funkhouser, T., Rusinkiewicz, S., 2003. Rotation invariant spherical harmonic representation of 3D shape descriptors, in: Proceedings of the Eurographics Symposium on Geometry Processing, Eurographics Association, Aachen, Germany. pp. 156–164.

Kazhdan, M., Funkhouser, T., Rusinkiewicz, S., 2004. Symmetry descriptors and 3D shape matching, in: Symposium on Geometry Processing, pp. 115–123. doi:10.1145/1057432.1057448.

Khairy, K., Howard, J., 2008. Spherical harmonics-based parametric deconvolution of 3D surface images using bending energy minimization. Medical Image Analysis 12, 217–227. doi:10.1016/j.media.2007.10.005.

Koenderink, J.J., van Doorn, A.J., 1992. Surface shape and curvature scales. Image and Vision Computing 10, 557–564. doi:10.1016/0262-8856(92)9 0076-F.

Lambin, P., Leijenaar, R.T.H., Deist, T.M., Peerlings, J., de Jong, E.E.C., van Timmeren, J., Sanduleanu, S., Larue, R.T.H.M., Even, A.J.G., Jochems, A., van Wijk, Y., Woodruf, H., van Soest, J., Lustberg, T., Roelofs, E., van Elmpt, W., Dekker, A., Mottaghy, F.M., Wildberger, J.E., Walsh, S., 2017. Radiomics: The bridge between medical imaging and personalized medicine. Nature Reviews Clinical Oncology 14, 749–762. doi:10.1038/nrclinonc.2017.141.

Li, Z., Huang, A., Jia, W., Wu, Q., Wei, M., Wang, J., 2024. FSH3D: 3D representation via fibonacci spherical harmonics. Computer Graphics Forum 43, e15231. doi:10.1111/cgf.15231.

Liepa, P., 2003. Filling holes in meshes, in: Proceedings of the Eurographics Symposium on Geometry Processing, The Eurographics Association. pp. 200–205. doi:10.2312/SGP/SGP03/200-206.

Lyu, Z., Lui, L.M., Choi, G.P.T., 2024. Spherical density-equalizing map for genus-0 closed surfaces. SIAM Journal on Imaging Sciences 17, 2110–2141. doi:10.1137/24M1633911.

Marino, M.A., Avendaño, D., Zapata, P., Riedl, C.C., Pinker, K., 2020. Lymph node imaging in patients with primary breast cancer: Concurrent diagnostic tools. The Oncologist 25, e231–e242. doi:10.1634/theoncol ogist.2019-0427.

Meyer, M., Desbrun, M., Schröder, P., Barr, A.H., 2003. Discrete diferentialgeometry operators for triangulated 2-manifolds, in: Hege, H.C., Polthier, K. (Eds.), Visualization and Mathematics III. Springer, Berlin, Heidelberg. Mathematics and Visualization, pp. 35–57. doi:10.1007/978-3-662-051 05-4\_2.

Möller, T., Trumbore, B., 1997. Fast, minimum storage ray–triangle intersection. Journal of Graphics Tools 2, 21–28. doi:10.1080/1086 7651.1997.10487468.

Mönch, T., Gasteiger, R., Janiga, G., Theisel, H., Preim, B., 2011. Contextaware mesh smoothing for biomedical applications. Computers & Graphics 35, 755–767. doi:10.1016/j.cag.2011.04.011.

Osada, R., Funkhouser, T., Chazelle, B., Dobkin, D., 2001. Matching 3D models with shape distributions, in: Proceedings of the International Conference on Shape Modeling and Applications, IEEE Computer Society. pp. 154–166. doi:10.1109/SMA.2001.923386.

Podobnik, G., Vrtovec, T., 2025. MeshMetrics: A precise implementation of distance-based image segmentation metrics. arXiv preprint arXiv:2509.05670 doi:10.48550/arXiv.2509.05670, arXiv:2509.05670.

Pumarola, A., Sanchez, J., Choi, G.P.T., Sanfeliu, A., Moreno-Noguer, F., 2019. 3Dpeople: Modeling the geometry of dressed humans, in: 2019 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE. pp. 2242–2251. doi:10.1109/ICCV.2019.00233.

Qu, L., Mei, X., Yi, Z., Zou, Q., Zhou, Q., Zhang, D., Zhou, M., Pei, L., Long, Q., Meng, J., Zhang, H., Chen, Q., Yi, W., 2024. An unsupervised learning model based on CT radiomics features accurately predicts axillary lymph node metastasis in breast cancer patients: Diagnostic study. International Journal of Surgery 110, 5363–5373. doi:10.1097/JS9.0000000000001778.

Qu, L., Zhu, J., Mei, X., Yi, Z., Luo, N., Yuan, S., Liu, X., Liu, M., Xie, H., Hu, X., Pan, L., Liang, Q., Li, Y., Zou, Q., Zhou, Q., Zhang, D., Zhou, M., Pei, L., Qian, K., Long, Q., Chen, Q., Chen, X., Plichta, J.K., Shang, Q., Ouyang, M., Xu, J., Yi, W., 2025. Evaluating axillary lymph node metastasis risks in breast cancer patients via Semi-ALNP: A multicenter study. eClinicalMedicine 85, 103311. doi:10.1016/j.eclinm.2025.103311.

Reuter, M., Wolter, F.E., Peinecke, N., 2006. Laplace–Beltrami spectra as ‘Shape-DNA’ of surfaces and solids. Computer-Aided Design 38, 342–366. doi:10.1016/j.cad.2005.10.011.

Shewchuk, J.R., 1996. Triangle: Engineering a 2D quality mesh generator and delaunay triangulator, in: Lin, M.C., Manocha, D. (Eds.), Applied Computational Geometry: Towards Geometric Engineering. Springer, Berlin, Heidelberg. volume 1148 of Lecture Notes in Computer Science, pp. 203–222. doi:10.1007/BFb0014497.

Shirshin, A.V., Zheleznyak, I.S., Malakhovsky, V.N., Kushnarev, S.V., Gorina, N.S., 2021. Evaluation of geometric deviations in rapid prototyped three-dimensional models created from computed tomography data. Digital Diagnostics 2, 277–288. doi:10.17816/DD63680.

Sinha, A., Bai, J., Ramani, K., 2016. Deep learning 3D shape surfaces using geometry images, in: European Conference on Computer Vision, Springer. pp. 223–240. doi:10.1007/978-3-319-46466-4\_14.

Škrinjar, O., Bistoquet, A., 2009. Generation of myocardial wall surface meshes from segmented MRI. International Journal of Biomedical Imaging 2009, 313517. doi:10.1155/2009/313517.

Styner, M., Oguz, I., Xu, S., Brechbühler, C., Pantazis, D., Levitt, J.J., Shenton, M.E., Gerig, G., 2006. Framework for the statistical shape analysis of brain structures using SPHARM-PDM. The Insight Journal , 242–250doi:10.54294/owxzil.

Sun, J., Ovsjanikov, M., Guibas, L., 2009. A concise and provably informative multi-scale signature based on heat difusion. Computer Graphics Forum 28, 1383–1392. doi:10.1111/j.1467-8659.2009.01515.x.

Taubin, G., 1995. A signal processing approach to fair surface design, in: Proceedings of the 22nd Annual Conference on Computer Graphics and Interactive Techniques, ACM. pp. 351–358. doi:10.1145/218380.218473.

Tutte, W.T., 1963. How to draw a graph. Proceedings of the London Mathematical Society 13, 743–767. doi:10.1112/plms/s3-13.1.743.

Valizadeh, G., Babapour Mofrad, F., Shalbaf, A., 2021. Parametric-based feature selection via spherical harmonic coeficients for the left ventricle myocardial infarction screening. Medical & Biological Engineering & Computing 59, 1261–1283. doi:10.1007/s11517-021-02372-4.

Vollmer, J., Mencl, R., Müller, H., 1999. Improved Laplacian smoothing of noisy surface meshes. Computer Graphics Forum 18, 131–138. doi:10.1111/1467-8659.00334.

Wang, K., Torkhani, F., Montanvert, A., 2012. A fast roughness-based approach to the assessment of 3D mesh visual quality. Computers & Graphics 36, 808–818. doi:10.1016/j.cag.2012.06.004.

Yang, C., Dong, J., Liu, Z., Guo, Q., Nie, Y., Huang, D., Qin, N., Shu, J., 2021. Prediction of metastasis in the axillary lymph nodes of patients with breast cancer: A radiomics method based on contrastenhanced computed tomography. Frontiers in Oncology 11, 726240. doi:10.3389/fonc.2021.726240.

Yotter, R.A., Dahnke, R., Gaser, C., 2009. Topological correction of brain surface meshes using spherical harmonics, in: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2009, Springer. pp. 125– 132. doi:10.1007/978-3-642-04271-3\_16.