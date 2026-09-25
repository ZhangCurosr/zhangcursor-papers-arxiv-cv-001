# Anatomy-Aligned Surface Field Learning for Myocardial Reconstruction from Sparse Short-Axis Cine MRI

Xiaohan Yuan<sup>a,b</sup>, Xuan Yang<sup>a</sup>, Qingya Li<sup>a</sup>, Yangang Wang<sup>b</sup>, Lei Li\*<sup>a</sup>

<sup>a</sup>Department of Biomedical Engineering, National University of Singapore, Singapore <sup>b</sup>School of Automation, Southeast University, Nanjing, China

## Abstract

Patient-specific 4D myocardial reconstruction from cine MRI supports quantitative functional assessment, regional motion analysis, and simulation-based modeling. However, routinely acquired short-axis (SAX) cine MRI is sparsely sampled2 along the through-plane direction, making dense and anatomically consistent surface reconstruction challenging. In this<sup>0</sup> study, we propose an anatomy-aligned surface learning framework that parameterizes the epicardial and endocardial surfaces on a shared circumferential-longitudinal UV domain. This formulation converts irregular 3D reconstruction intop structured coordinate-field completion with explicit correspondence across subjects and cardiac phases. Sparse SAX con-<sup>e</sup> tours are encoded as UV observation fields, coverage-aware sampling improves robustness to incomplete slice coverage, <sup>S</sup> and topology- and distortion-aware learning preserves circumferential continuity and local surface quality. Experiments<sup>4</sup> on three public cine MRI datasets showed that the proposed method consistently outperformed representative meshbased and implicit reconstruction approaches, achieving overall Chamfer distances of 2.887 mm on ACDC, 2.641 mm on M&Ms, and 2.810 mm on M&Ms-2. The reconstructed sequences also preserved ventricular function, with end-diastolicV volume and ejection fraction errors of 3.3 mL and 1.1%, respectively. These results demonstrate that anatomy-aligned UV learning provides an accurate, eficient, and correspondence-aware representation for sparse cine MRI reconstruction. and myocardial modeling. The source code will be available at https://github.com/yuan-xiaohan/SAX2MyoSurf.

Keywords: Cine MRI, 3D/4D Myocardial Reconstruction, UV Surface Parameterization, Anatomy-Aligned Representation.

1. Introduction  
![](images/a54fc92d8463ea09aaa3bbc2dbf266d21528d41774081af8cf9cf0785b563420.jpg)  
Figure 1: Overview of the anatomy-aligned UV representation for sparse short-axis (SAX) myocardial reconstruction. The epicardial (epi) and endocardial (endo) surfaces are unfolded onto shared circumferential-longitudinal UV domains, where each location stores the corresponding 3D surface coordinates.

Patient-specific 4D (3D+t) myocardial mesh reconstruction is fundamental for quantitative cardiovascular analysis (Qiao et al., 2025a; Liu et al., 2026; Li et al., 2024a). Compared with image-based measurements alone, surfacebased representations provide explicit anatomical geometry and enable downstream analyses such as spatial correspondence, regional functional mapping, and simulationbased modeling. In routine clinical imaging, cine MRI is commonly acquired as short-axis (SAX) slices, which provide high in-plane resolution but limited through-plane coverage along the long-axis direction (Li et al., 2023). Recovering a geometrically smooth and anatomically plausible myocardial surface from such sparse SAX observations remains challenging because geometric evidence is limited to sparse SAX planes.

Existing learning-based myocardial reconstruction methods can be broadly grouped into explicit mesh deformation and implicit field prediction. Mesh-based methods typically deform a predefined template by regressing vertexwise displacements, often using graph neural networks, mesh convolution, or diferentiable projection losses to constrain the reconstructed surface (Meng et al., 2022, 2023; Luo et al., 2026; Gaggion et al., 2025). These methods have shown promising performance when dense annotations, multi-view images, or strong projection constraints are available. However, under sparse SAX supervision, image evidence is only observed at a limited number of slice locations, making it dificult to propagate local contour information coherently across the entire ventricular surface. Moreover, vertex-space learning provides consistent topology but lacks an explicit anatomical coordinate system for organizing sparse SAX observations along the myocardial circumferential and longitudinal directions. Implicit reconstruction methods alleviate some of these limitations by predicting continuous occupancy, signed distance, or coordinate fields in 3D space, thereby improving smoothness and topology preservation (Chen et al., 2024; Fu et al., 2025; Sun et al., 2022; Yuan et al., 2023; Ye et al., 2023). Nevertheless, most implicit methods still operate in generic Euclidean coordinates and do not explicitly exploit the intrinsic anatomical organization of the myocardium. This is suboptimal for the left ventricle (LV), which exhibits a well-defined circumferential-longitudinal structure with approximately cylindrical topology. Such intrinsic coordinates have been used for regional partitioning, cardiac mechanics, and quantitative cardiac MRI analysis (Bayer et al., 2018; Schuler et al., 2021). In computer graphics and human/object surface modeling, surface parameterization has been widely used to represent complex 3D surfaces on structured 2D domains (G¨uler et al., 2018; Zufi et al., 2017; Huang et al., 2022; Zhao et al., 2023; Hu et al., 2024). However, using such a surface domain to jointly organize sparse SAX observations and dense reconstruction targets remains underexplored.

In this study, we introduce an anatomy-aligned surface learning framework for reconstructing dense 3D myocardial surfaces from sparse SAX cine MRI. The proposed method reformulates myocardial reconstruction from sparse slice observations as continuous 3D coordinate field estimation on a shared two-dimensional anatomical surface domain. Specifically, we parameterize the LV surface using a circumferential-longitudinal UV domain, where the u coordinate follows the circumferential direction and the v coordinate follows the longitudinal base-to-apex direction. This representation establishes anatomical correspondence across subjects and converts irregular 3D surface reconstruction into structured surface field learning. Sparse SAX contours are further lifted into the UV domain as observation fields, providing spatially coherent geometric conditioning from discontinuous slice measurements. To respect the cylindrical topology of the LV, we incorporate explicit periodic modeling along the circumferential direction. To our knowledge, this is the first sparse-SAX reconstruction framework to represent both observations and target surfaces as dense fields on the same anatomyaligned UV domain. The main contributions of this work are:

• We introduce an anatomy-aligned UV representation that reformulates sparse-SAX myocardial reconstruction as structured coordinate-field learning with explicit correspondence across subjects and cardiac phases.

• We propose a contour-to-UV encoder that maps discontinuous SAX contours into surface-aligned observation fields, together with coverage-aware sampling for incomplete slice coverage.

• We develop a topology- and distortion-aware completion network that combines circumferential wrap convolution with region-focused geometric regularization for coherent epicardial and endocardial reconstruction.

• We validate the framework on three public cine MRI datasets, demonstrating accurate and eficient 3D/4D myocardial reconstruction.

## 2. Related work

## 2.1. 3D/4D Cardiac Reconstruction from Cine MRI

Existing 3D/4D cardiac reconstruction methods can be broadly categorized into explicit surface modeling, implicit continuous representations, and generative shape modeling. Explicit approaches recover patient-specific anatomy by deforming fixed-topology meshes, templates, or statistical shape models. Representative methods include the MulViMotion series (Meng et al., 2022, 2023), Mesh4D (Qiao et al., 2025b), HeartSSM (Ma et al., 2026), and atlas-based frameworks (Sinclair et al., 2022). Subsequent studies incorporated point-cloud deformation (Beetz et al., 2024), CT-derived priors (Xu et al., 2024), diferentiable slicing (Luo et al., 2026), graph-based refinement (Gaggion et al., 2025), and statistical shape constraints (Joyce et al., 2022; Xia et al., 2022). Although these methods provide explicit and interpretable surfaces, they often depend on multi-view imaging, dense supervision, or strong geometric priors, which limits their applicability to sparse SAX-only acquisitions.

Implicit approaches instead represent cardiac anatomy and motion using continuous fields or deformable domains, including dynamic Gaussian representations (Fu et al., 2025), topology-preserving implicit registration (Sun et al., 2022), disentangled shape-motion models (Yuan et al., 2023) neural deformation fields (Ye et al., 2023), and deformable tetrahedral models (Chen et al., 2024, 2025). NIHC (Muffoletto et al., 2025) further introduced anatomically coherent implicit coordinates based on universal ventricular coordinates. In contrast, our formulation explicitly lifts sparse SAX contours into dense surface-aligned observation fields for direct UV-domain completion. Generative models have also been explored for dynamic shape synthesis, surface completion, and virtual population modeling (Ma et al., 2025; Qiao et al., 2025a; Zhang et al., 2025; Yang et al., 2026). Their emphasis, however, is primarily on distribution learning rather than directly reconstructing dense myocardial surfaces from sparse SAX observations while preserving explicit anatomical correspondence across subjects and cardiac phases.

## 2.2. Anatomical Coordinates and Canonical Surface Representations

Cardiac anatomical coordinate systems have been widely used for regional partitioning, functional analysis, and quantitative modeling. For example, the AHA 17-segment model provides a standardized regional reference for analyzing myocardial perfusion, motion abnormalities, and scar distribution (Li et al., 2024b). Universal ventricular coordinates (Bayer et al., 2018) and Cobiveco (Schuler et al., 2021) further construct continuous anatomical coordinate systems for establishing correspondences across cardiac mod els for cross-subject functional comparison. However, these coordinate systems are typically used as post-processing analysis tools, simulation coordinate systems, or registration reference spaces, and have rarely been adopted as the primary learning domain for reconstruction models under sparse observations. In general computer vision and graphics, canonical UV representations are widely used to unfold deformable 3D surfaces into cross-instance consistent 2D parameter domains, thereby transforming irregular surface learning into image-like representation learning. Representative works, such as DensePose (G¨uler et al., 2018), canonical animal/human surface modeling (Zufi et al., 2017), and SUPPLE (Zhao et al., 2023), demonstrate the advantages of UV representations in dense correspondence, geometric alignment, and cross-instance structural modeling. Nevertheless, these methods are mainly designed for natural images, dense surface observations, or texturebased supervision, and have not been fully explored for sparse medical image-based reconstruction (Geng et al., 2023; Hu et al., 2024).

## 3. Method

As illustrated in Fig. 2, the proposed framework reconstructs patient-specific myocardial surfaces from SAX contours through three key components. First, the epicardial and endocardial surfaces are represented as 3D coordinate fields on a shared anatomy-aligned UV domain (Sec. 3.1). Second, sparse SAX contours are transformed into surface-aligned observation fields through anatomical correspondence retrieval and soft cross-slice aggregation, with coverage-aware sampling introduced during training (Sec. 3.2). Third, a periodic UV-domain network completes the two surface fields, while topology- and distortionaware constraints preserve circumferential continuity and suppress local geometric artifacts. The predicted coordinate fields are finally converted into explicit meshes using fixed UV connectivity (Sec. 3.3).

## 3.1. Anatomy-Aligned UV Surface Representation

As illustrated in Fig. 1, we represent the epicardial and endocardial surfaces on a shared circumferential-longitudinal UV domain. A myocardial template (Bai et al., 2015) provides consistent topology and vertex correspondence across subjects and phases. Its UV parameterization is

constructed once and shared across all reference and predicted surfaces, such that the same UV location corresponds to the same anatomical position across subjects and cardiac phases. Specifically, we define the canonical domain as

$$
\Omega = \{ ( u , v ) \mid u \in [ 0 , 1 ) , v \in [ 0 , 1 ] \} ,\tag{1}
$$

where u is the periodic circumferential coordinate and v is the longitudinal coordinate from the base to the apex. For each surface $s \in \{ \mathrm { e p i } $ , endo}, the parameterization is defined as

$$
\Phi _ { s } : S _ { s } ^ { \mathrm { t e m p l a t e } } \to \Omega , \qquad { \bf x } \mapsto \left( u _ { s } ( { \bf x } ) , v _ { s } ( { \bf x } ) \right) .\tag{2}
$$

The two surfaces share the same centerline, septal reference, and seam convention, maintaining correspondence between the myocardial boundaries.

Let $\mathbf { c } ( \tau )$ denote the template centerline parameterized by normalized arc length $\tau \in [ 0 , 1 ]$ , from the base to the apex. The longitudinal coordinate of a surface point x is determined by its nearest centerline position:

$$
\boldsymbol { \tau } ^ { * } ( \mathbf { x } ) = \underset { \boldsymbol { \tau } \in [ 0 , 1 ] } { \arg \operatorname* { m i n } } ~ \| \mathbf { x } - \mathbf { c } ( \boldsymbol { \tau } ) \| _ { 2 } , \qquad \boldsymbol { v } _ { s } ( \mathbf { x } ) = \boldsymbol { \tau } ^ { * } ( \mathbf { x } ) .\tag{3}
$$

At $\mathbf { c } ( \tau ^ { * } )$ , the centerline tangent $\mathbf { t } ( \tau ^ { * } )$ defines the local cross-sectional plane. A septal reference projected onto this plane defines $\mathbf { e } _ { 1 } ( \tau ^ { * } )$ , and ${ \bf e } _ { 2 } ( \tau ^ { * } ) = { \bf t } ( \tau ^ { * } ) \times { \bf e } _ { 1 } ( \tau ^ { * } )$ completes the local orthonormal basis. The circumferential coordinate is then

$$
u _ { s } ( \mathbf { x } ) = \frac { [ \mathrm { a t a n 2 } \left( \mathbf { r } \cdot { \mathbf e } _ { 2 } ( \tau ^ { * } ) , \mathbf { r } \cdot { \mathbf e } _ { 1 } ( \tau ^ { * } ) \right) - \theta _ { \mathrm { s e a m } } ] \bmod 2 \pi } { 2 \pi } ,\tag{4}
$$

where $\mathbf { r } = \mathbf { x } - \mathbf { c } ( \tau ^ { * } )$ , and $\theta _ { \mathrm { s e a m } }$ fixes the seam at a consistent anatomical location. The resulting parameterization represents each surface as a 3D coordinate field:

$$
\mathbf { X } _ { s } : \Omega _ { s } \to \mathbb { R } ^ { 3 } , \qquad ( u , v ) \mapsto \mathbf { X } _ { s } ( u , v ) ,\tag{5}
$$

where $\Omega _ { s } \subseteq \Omega$ is the valid region of surface s, and ${ \bf X } _ { s } ( u , v )$ stores the corresponding 3D position. We denote the template and patient-specific fields by $\mathbf { X } _ { s } ^ { \mathrm { t } }$ <sup>emplate</sup> and $\mathbf { X } _ { s } ^ { * } ,$ respectively. Note that the coordinate fields are discretized on an $H \times W$ grid with a surface-specific validity mask $M _ { s }$ . Fixed connectivity over the valid grid locations converts each field into an explicit triangular mesh. Therefore, irregular 3D myocardial reconstruction is reformulated as structured coordinate-field learning on a shared anatomical domain.

## 3.2. Sparse SAX Contour-to-UV Observation Modeling

The canonical UV representation defines the reconstruction target but does not directly organize the sparse SAX contour observations. We therefore first transform the discontinuous contours into observation fields on the same anatomy-aligned domain, and further introduce coverageaware slice sampling to improve robustness to varying longitudinal coverage.

![](images/b060f3fb83fc77eeda3a51fd54535a5d5f3f09d6e9924c4396d7c7b5b8df82a3.jpg)  
Figure 2: Overall pipeline of the proposed anatomy-aligned UV myocardial surface reconstruction framework. Sparse SAX contours are aligned to the canonical template and encoded as epicardial and endocardial UV observation fields, with coverage-aware sampling used during training. A topology- and distortion-aware UV network completes the patient-specific surface fields through circumferential WrapConv2D and region-focused geometric supervision, followed by explicit mesh reconstruction using fixed UV connectivity.

## 3.2.1. Contour-to-UV Observation Encoding

The SAX contours are first rigidly aligned with the canonical template. Let $\mathcal { C } _ { s , k } ~ = ~ \{ \mathbf { p } _ { s , k , i } \} _ { i = 1 } ^ { N _ { s , k } }$ denote the aligned contour of surface s ∈ {epi, endo} on slice k. For each canonical UV location $( u , v )$ , the corresponding template point ${ \bf p } _ { s } ^ { \mathrm { q u e r y } } = { \bf X } _ { s } ^ { \mathrm { t e m p l a t e } } ( u , v )$ is used as the anatomical query. Since each UV location has a fixed anatomical meaning, the template query provides a stable anatomical prior for associating sparse contour observations with canonical surface locations.

Before retrieval, Ψ(·) places each contour at its anchorbased centerline position and aligns its normal with the local tangent using shared end-diastolic (ED) parameters. The transformed contour points can then be compared directly with the template query using Euclidean distance. For each slice, we retrieve the transformed contour point closest to the query and select its corresponding point from the rigidly aligned contour:

$$
\begin{array} { r } { i _ { s , k } ^ { * } = \underset { 1 \leq i \leq N _ { s , k } } { \arg \operatorname* { m i n } } \left\| \mathbf { p } _ { s } ^ { \mathrm { q u e r y } } - \Psi ( \mathbf { p } _ { s , k , i } ) \right\| _ { 2 } , \qquad { \tilde { \mathbf { p } } } _ { s , k } = \mathbf { p } _ { s , k , i _ { s , k } ^ { * } } . } \end{array}\tag{6}
$$

The transformed contour points determine the retrieval index. The selected point $\tilde { \bf p } _ { s , k }$ is taken from the rigidly aligned contour, preserving the patient-specific 3D position. Each slice contributes one candidate for the current query. Let $d _ { s , k } = \| \mathbf { p } _ { s } ^ { \mathrm { q u e r y } } - \tilde { \mathbf { p } } _ { s , k } \| _ { 2 }$ denote the query-tocandidate distance in the rigidly aligned coordinates. The candidates from the K available SAX slices are aggregated using distance-dependent soft weights:

$$
\begin{array} { c l } { { } } & { { w _ { s , k } = \displaystyle \frac { \exp \Big [ - d _ { s , k } ^ { 2 } / ( 2 \beta ^ { 2 } ) \Big ] } { \sum _ { k ^ { \prime } = 1 } ^ { K } \exp \Big [ - d _ { s , k ^ { \prime } } ^ { 2 } / ( 2 \beta ^ { 2 } ) \Big ] } , } } \\ { { } } & { { { \displaystyle \mathbf { X } _ { s } ^ { \mathrm { o b s } } ( u , v ) = \sum _ { k = 1 } ^ { K } w _ { s , k } \tilde { \mathbf { p } } _ { s , k } } . } } \end{array}\tag{7}
$$

where $\beta$ controls the aggregation bandwidth. Applying this procedure over the canonical UV domain yields the epicardial and endocardial observation fields $\mathbf { X } _ { \mathrm { e p i } } ^ { \mathrm { o b s } }$ and $\mathbf { X } _ { \mathrm { e n d o } } ^ { \mathrm { o b s } } .$

## 3.2.2. Coverage-Aware Slice Sampling

The number and longitudinal distribution of usable SAX slices vary across acquisitions. To reduce dependence on a fixed slice configuration, we perform coverage-aware sampling on the physical contours before UV encoding. Let V denote the ordered set of valid SAX slices. During training, a subset $\mathcal { V } _ { p } \subseteq \mathcal { V }$ is sampled according to the coverage patterns illustrated in Fig. 2, including full coverage, missing basal or apical slices, retained middle regions, and uniform sparse sampling. The corresponding observation field is written as

$$
\mathbf { X } _ { s , p } ^ { \mathrm { o b s } } = \mathcal { E } \left( \{ \mathcal { C } _ { s , k } \} _ { k \in \mathcal { V } _ { p } } ; \mathbf { X } _ { s } ^ { \mathrm { t e m p l a t e } } \right) ,\tag{8}
$$

where $\mathcal { E }$ denotes the contour-to-UV encoder. Because sampling is performed on the physical SAX contours before UV encoding, the resulting inputs preserve realistic longitudinal acquisition patterns. This exposes the network to varying slice coverage during training and improves robustness to incomplete observations.

## 3.3. Topology- and Distortion-Aware Surface Completion

Given the two observation fields, myocardial reconstruction is formulated as dense coordinate-field completion. The network input is obtained by concatenating the surface-specific observation fields:

$$
\mathcal { T } = \mathrm { C o n c a t } \left( \mathbf { X } _ { \mathrm { e p i } } ^ { \mathrm { o b s } } , \mathbf { X } _ { \mathrm { e n d o } } ^ { \mathrm { o b s } } \right) \in \mathbb { R } ^ { 6 \times H \times W } .\tag{9}
$$

As shown in Fig. 2, a shared encoder $E _ { \theta _ { \mathrm { e } } }$ extracts a joint representation of the myocardial geometry, which is subsequently processed by two surface-specific decoders:

$$
{ \bf Z } = E _ { \theta _ { \mathrm { e } } } ( { \mathcal { T } } ) , \qquad { \hat { \bf X } } _ { s } = D _ { \theta _ { s } } ( { \bf Z } ) , \quad s \in \{ \mathrm { e p i , e n d o } \} .\tag{10}
$$

The shared encoder captures the overall ventricular geometry and the relationship between the two myocardial boundaries, while each decoder reconstructs the complete coordinate field of its corresponding surface.

## 3.3.1. Circumferentially Periodic Feature Learning

The original myocardial surface is closed along the circumferential direction, but cutting it into a rectangular UV domain places anatomically adjacent locations at opposite lateral boundaries. Conventional convolution therefore breaks their intrinsic neighborhood. To preserve this topology, we introduce Circumferential Wrap Convolution:

$$
\mathrm { W r a p C o n v 2 D } ( \mathbf { F } ; \mathbf { G } ) = \mathbf { G } * \mathrm { P a d } _ { v } \left( \mathrm { P a d } _ { u } ^ { \mathrm { w r a p } } ( \mathbf { F } ) \right) ,\tag{11}
$$

where F and G denote the input feature map and learnable convolution kernel, respectively; periodic padding is applied along the circumferential coordinate $u ,$ and conventional padding along the longitudinal coordinate v. Wrap-Conv2D is used throughout the encoder and decoders, enabling multi-scale feature aggregation across the UV seam and maintaining circumferential continuity in the completed surface fields.

## 3.3.2. Region-Focused Geometric Learning

The predicted coordinate fields are first supervised by a surface-wide objective:

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { 2 } \sum _ { s \in \{ \mathrm { e p i } , \mathrm { e n d o } \} } \mathrm { M S E } \left( M _ { s } \odot \hat { \mathbf { X } } _ { s } , M _ { s } \odot \mathbf { X } _ { s } ^ { * } \right) ,\tag{12}
$$

where $M _ { s }$ denotes the valid UV mask. The base objective combines coordinate reconstruction with surface smoothness and global geometric consistency:

$$
\begin{array} { r l } & { { \mathcal { L } } _ { \mathrm { b a s e } } = { \mathcal { L } } _ { \mathrm { r e c } } + \lambda _ { \mathrm { t v } } { \mathcal { L } } _ { \mathrm { t v } } + \lambda _ { \mathrm { o v e r l a p } } { \mathcal { L } } _ { \mathrm { o v e r l a p } } } \\ & { ~ + \lambda _ { \mathrm { n o r m a l } } ^ { \mathrm { g } } { \mathcal { L } } _ { \mathrm { n o r m a l } } ^ { \mathrm { g } } + \lambda _ { \mathrm { f i p } } ^ { \mathrm { g } } { \mathcal { L } } _ { \mathrm { f i p } } ^ { \mathrm { g } } . } \end{array}\tag{13}
$$

Here, ${ \mathcal { L } } _ { \mathrm { t v } }$ smooths the UV coordinate fields, L<sub>overlap</sub> preserves the coincidence of template-defined shared epicardialendocardial vertices, and $\mathcal { L } _ { \mathrm { n o r m a l } } ^ { \mathrm { g } }$ and $\mathcal { L } _ { \mathrm { { f l i p } } } ^ { \mathrm { { g } } }$ constrain surfacewide normal consistency and face inversion, respectively.

$\lambda _ { \mathrm { t v } } .$ λ<sub>overlap</sub>, $\lambda _ { \mathrm { n o r m a l } } ^ { g } ,$ and $\lambda _ { \mathrm { f l i p } } ^ { g }$ are all balancing parameters.

Although the canonical UV representation provides a regular domain for structured surface learning, the contraction of the ventricular circumference near the apex still introduces pronounced metric distortion. Coordinate errors in this region are therefore more likely to be amplified into local mesh artifacts. Because these parameterizationsensitive locations occupy only a small portion of the surface, surface-wide averaging may not provide suficient emphasis. We thus precompute a focus core $\mathcal { R } _ { 0 }$ from canonical template locations exhibiting strong metric compression or UV-grid conflicts. Surrounding bufer regions $\{ \mathcal { R } _ { r } \} _ { r = 1 } ^ { R }$ are constructed through mesh-adjacency expansion. For each geometric term $q \in { \mathcal { Q } } = \{ \mathrm { e d g e }$ , area, normal, flip}, we compute a region-balanced loss as

$$
\begin{array} { r } { \mathcal { L } _ { q } ^ { \mathrm { f o c u s } } = \frac { \sum _ { r = 0 } ^ { R } \gamma _ { r } \mathcal { L } _ { q } ^ { ( r ) } } { \sum _ { r = 0 } ^ { R } \gamma _ { r } } , } \end{array}\tag{14}
$$

where $\mathcal { L } _ { q } ^ { ( r ) }$ is the mean penalty within $\mathcal { R } _ { r }$ and $\gamma _ { r }$ controls its contribution. This region-wise balancing prevents the compact focus core from being dominated by larger surrounding regions. The overall region-focused objective is

$$
\mathcal { L } _ { \mathrm { f o c u s } } = \sum _ { q \in \mathcal { Q } } \lambda _ { q } \mathcal { L } _ { q } ^ { \mathrm { f o c u s } } ,\tag{15}
$$

where $\lambda _ { q }$ is balancing parameter. Therefore, the complete objective is $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { b a s e } } + \mathcal { L } _ { \mathrm { f o c u s } }$ . Finally, the predicted epicardial and endocardial coordinate fields are converted into explicit triangular meshes using the fixed connectivity of the canonical UV grid.

## 4. Experiments and Results

## 4.1. Datasets and Experimental Protocols

We evaluated the reconstruction framework on three public cine MRI datasets: ACDC (Bernard et al., 2018), M&Ms (Campello et al., 2021), and M&Ms-2 (Mart´ın-Isla et al., 2023). ACDC contains SAX cine MRI from 150 cases evenly distributed into five diagnostic categories, including normal subjects (NOR), dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), myocardial infarction with altered left ventricular ejection fraction (MINF), and abnormal right ventricle (ARV). From M&Ms and M&Ms-2, we included 340 and 350 cases, respectively. These datasets provide complementary multi-center, multivendor, and multi-disease cohorts for evaluating reconstruction under heterogeneous acquisition conditions and cardiac anatomies. For reconstruction, each dataset was independently split at the subject level into approximately 70% training and 30% testing, with 20% of the training subjects reserved for validation. The test sets remained unseen until final evaluation.

For downstream cine MRI-based scar localization (Sec. 4.6.2), we additionally used the CineMyoPS dataset from the

Table 1: Quantitative comparison of reconstruction on ACDC, M&Ms, and M&Ms-2. Results are reported as mean (standard deviation).
<table><tr><td rowspan="2">Method</td><td colspan="2">ACDC</td><td colspan="2">M&amp;Ms</td><td colspan="2">M&amp;Ms-2</td><td rowspan="2">Inference Time (s/frame) ↓</td></tr><tr><td>CD (mm) ↓</td><td>F@2mm ↑</td><td>CD (mm) ↓</td><td>F@2mm ↑</td><td>CD (mm) ↓</td><td>F@2mm ↑</td></tr><tr><td>Voxel2Mesh</td><td>8.629 (2.773)</td><td>0.230 (0.126)</td><td>8.325 (2.742)</td><td>0.246 (0.120)</td><td>8.560 (2.761)</td><td>0.246 (0.125)</td><td>1.4</td></tr><tr><td>PN-GCN</td><td>3.183 (0.395)</td><td>0.741 (0.092)</td><td>2.873 (0.446)</td><td>0.808 (0.092)</td><td>3.144 (0.838)</td><td>0.770 (0.106)</td><td>1.0</td></tr><tr><td>4DMM</td><td>2.983 (0.398)</td><td>0.782 (0.067)</td><td>2.776 (0.339)</td><td>0.820 (0.064)</td><td>2.914 (0.387)</td><td>0.797 (0.066)</td><td>&gt; 30</td></tr><tr><td>GHDHeart</td><td>4.700 (1.060)</td><td>0.463 (0.112)</td><td>4.528 (1.169)</td><td>0.511 (0.131)</td><td>4.684 (1.012)</td><td>0.479 (0.117)</td><td>&gt; 30</td></tr><tr><td>Ours</td><td>2.887 (0.388)</td><td>0.796 (0.083)</td><td>2.641 (0.381)</td><td>0.839 (0.076)</td><td>2.810 (0.524)</td><td>0.817 (0.076)</td><td>0.8</td></tr></table>

Table 2: Quantitative comparison of left ventricular (LV) myocardium segmentation obtained by intersecting the reconstructed surfaces with the original SAX planes. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="3">ACDC</td><td colspan="3">M&amp;Ms</td><td rowspan="2">M&amp;Ms-2</td><td rowspan="2">ASSD (mm) ↓</td></tr><tr><td>Dice ↑</td><td>ASSD (mm) ↓</td><td>HD95 (mm) ↓</td><td>Dice ↑</td><td>ASSD (mm) ↓</td><td>HD95 (mm)↓</td></tr><tr><td>Voxel2Mesh</td><td>0.506 (0.200)</td><td>2.237 (1.513)</td><td>9.171 (5.059)</td><td>0.531 (0.162)</td><td>2.082 (1.345)</td><td>9.615 (4.185)</td><td>Dice ↑ 0.481 (0.132)</td><td>3.468 (1.245)</td></tr><tr><td>PN-GCN</td><td>0.899 (0.044)</td><td>0.329 (0.295)</td><td>2.616 (3.368)</td><td>0.923 (0.043)</td><td>0.280 (0.394)</td><td>2.664 (3.733)</td><td>0.893 (0.070)</td><td>0.398 (0.668)</td></tr><tr><td>4DMM</td><td>0.956 (0.024)</td><td>0.172 (0.205)</td><td>2.386 (3.011)</td><td>0.953 (0.028)</td><td>0.227 (0.316)</td><td>2.765 (3.574)</td><td>0.944 (0.031)</td><td>0.250 (0.323)</td></tr><tr><td>GHDHeart</td><td>0.806 (0.071)</td><td>0.790 (0.489)</td><td>4.448 (2.898)</td><td>0.786 (0.062)</td><td>1.216 (0.720)</td><td>8.979 (5.262)</td><td>0.750 (0.069)</td><td>1.256 (0.684)</td></tr><tr><td>Ours</td><td>0.939 (0.031)</td><td>0.212 (0.213)</td><td>2.429 (2.654)</td><td>0.965 (0.021)</td><td>0.114 (0.204)</td><td>1.181 (2.327)</td><td>0.949 (0.045)</td><td>0.193 (0.512)</td></tr></table>

CARE challenge (Ding et al., 2025), comprising 64 cases from two centers with cine MRI and manual annotations of the LV, myocardium, and myocardial scar at end diastole. The trained reconstruction model was used to generate UV-domain myocardial surface sequences from cine SAX inputs, from which abnormal motion features were subsequently used for scar prediction. For UV-domain generative modeling, the reconstructed UV sequences from 270 NOR, DCM, and HCM cases were used to learn diseaseconditioned myocardial motion generation, with 216 cases for training and 54 held-out cases for evaluation.

## 4.2. Reference Mesh Construction and Evaluation

All methods used SAX contours generated by a fixed pretrained nnU-Net (Isensee et al., 2021). Patient-specific reference meshes with consistent topology and vertex correspondence were generated ofline from the multi-phase SAX contours using a public cardiac atlas (Bai et al., 2015) through global alignment, non-rigid refinement, and temporal smoothing. Surface agreement with these references was evaluated using symmetric Chamfer distance (CD) and F-score at a 2-mm tolerance (F@2mm). SAXplane agreement was assessed against the corresponding segmentations using Dice, ASSD, and HD95 after intersecting the reconstructed surfaces with the SAX planes. Cross-phase motion agreement was evaluated using EDrelative motion endpoint error (EPE) (?) at corresponding UV locations. Functional agreement was evaluated using end-diastolic volume (EDV) and left ventricular ejection fraction (LVEF). For the ablation studies, we additionally report UV MSE, measuring coordinate-field reconstruction error over the valid UV domain, and Edge Ratio P95, measuring high-percentile local mesh distortion relative to the reference surface.

## 4.3. Implementation

All experiments were implemented in PyTorch and conducted on a single NVIDIA GeForce RTX 4090 GPU. All coordinate fields were represented on a 256×256 canonical UV grid, with an aggregation bandwidth of $\beta = 0 . 0 6$ . The concatenated epicardial and endocardial fields formed a six-channel input to a seam-aware 2D U-Net with a shared encoder and surface-specific decoders. Separate models were trained from scratch for ACDC, M&Ms, and M&Ms-2 using full- and incomplete-coverage samples generated by coverage-aware sampling. AdamW was used with a batch size of 16, learning rate 10<sup>−4</sup>, weight decay 10<sup>−2</sup>, gradient clipping at 1.0, and cosine annealing; training lasted 48,000, 75,000, and 80,000 steps, respectively. The surface-wide and region-focused loss weights were set to (0.01, 0.1, 0.003, 0.001) and (0.04, 0.005, 0.003, 0.002), respectively, with region weights (1.00, 0.75, 0.50, 0.25). Checkpoints were selected using the validation set only.

## 4.4. Comparison Study

We compared with four representative approaches reproducible under the same sparse-SAX setting: Voxel2Mesh (Wickramasinghe et al., 2020), using volumetric features for mesh deformation; a PN-GCN baseline implemented in this study, encoding the same contours as unordered point features without explicit UV correspondence; 4DMM (Yuan et al., 2023), using implicit shape and motion fields; and GHD-Heart (Luo et al., 2026), combining diferentiable slicing with global mesh deformation. All trainable baselines were retrained using the same subject-level splits. As shown in Table 1, our method achieved the best surface reconstruction accuracy on all three datasets. The CD values were 2.887 mm on ACDC, 2.641 mm on M&Ms, and 2.810 mm on M&Ms-2, with corresponding F@2mm values of 0.796, 0.839, and 0.817. Our method significantly outperformed the best baseline on both metrics across all datasets (Holm-corrected paired Wilcoxon tests, p ≤

![](images/4c6d31c831ce5590c2b3ad319d44bec504735e9155b9621f0c4fa042ceec710c.jpg)  
Figure 3: Qualitative comparison of disease-specific myocardial reconstruction: (a) ED and ES surfaces for representative NOR, DCM, HCM, and MINF cases, showing sparse SAX contours, reference surfaces, and predictions from the competing methods. (b) Corresponding SAX contours, UV surface-ofset maps, and reconstructed 3D surfaces. For visualization, UV fields show normalized template-relative 3D ofsets in RGB. NOR: normal subjects; DCM/HCM: dilated/hypertrophic cardiomyopathy; MINF: myocardial infarction with altered LV ejection fraction.

0.005). The qualitative results in Fig. 3 (a) further demonstrated stable recovery of disease-specific ED and ES morphologies. Compared with these approaches, the proposed contour-to-UV encoding explicitly associates sparse SAX observations with anatomically corresponding surface locations before completion, allowing local contour evidence to directly guide structured surface prediction.

Table 2 further evaluated the LVM segmentations obtained by intersecting the reconstructed surfaces with the original SAX planes. On ACDC, our method performed comparably to 4DMM, achieving a Dice of 0.939, an ASSD of 0.212 mm, and an HD95 of 2.429 mm. On the more heterogeneous M&Ms and M&Ms-2 datasets, it achieved the best results across all metrics, with Dice scores of 0.965 and 0.949, ASSD values of 0.114 and 0.193 mm, and HD95 values of 1.181 and 1.932 mm, respectively. These improvements were consistent with the basal, mid-cavity, and apical comparisons shown in Fig. 4. Moreover, the proposed method required 0.8 s per frame, compared with more than 30 s for 4DMM and GHDHeart. Fig. 3 (b) further demonstrated that patient- and phase-specific variations were represented within a shared circumferential-longitudinal

Table 3: Ablation study of the proposed framework on ACDC. UV MSE is reported in units of $( \times \bar { 1 0 } ^ { - 3 } )$
<table><tr><td>Variant</td><td>UV MSE ↓</td><td>CD (mm) ↓</td><td>Edge Ratio P95 ↓</td></tr><tr><td>Rigid / Full-only</td><td>1.565 (3.413)</td><td>3.569 (1.829)</td><td>1.447 (0.187)</td></tr><tr><td>Centerline / Full-only</td><td>1.491 (3.106)</td><td>3.538 (1.718)</td><td>1.449 (0.189)</td></tr><tr><td>w/o Region Focus</td><td>0.499 (0.541)</td><td>3.021 (0.571)</td><td>1.408 (0.053)</td></tr><tr><td>w/o WrapConv2D</td><td>0.535 (0.573)</td><td>2.999 (0.532)</td><td>1.406 (0.052)</td></tr><tr><td>Ours</td><td>0.487 (0.531)</td><td>2.997 (0.546)</td><td>1.393 (0.049)</td></tr></table>

UV domain, enabling explicit anatomical correspondence and eficient reconstruction. The Supporting Video visualizes disease-specific reconstruction, method comparisons, and robustness to varying slice coverage.

## 4.5. Ablation Study

Tables 3 and 4 evaluated the contributions of centerline alignment, coverage-aware training, region-focused supervision, and WrapConv2D. The Rigid / Full-only and Centerline / Full-only variants were trained only with fullcoverage samples, whereas the complete model incorporated all proposed components. Results were pooled over five coverage patterns and jointly computed for the epicardial and endocardial surfaces using UV MSE, CD, and

![](images/04700739f6f0d8e208c6bbf6069b2cfa7350a2b1d8aa7165f7e00a42230f9951.jpg)  
Figure 4: Qualitative comparison of LV myocardial segmentation obtained by intersecting the reconstructed surfaces with the original SAX planes.

Table 4: Reconstruction accuracy under five slice-coverage patterns for models trained with full coverage only and with coverage-aware sampling.
<table><tr><td></td><td colspan="2">UV MSE↓</td><td colspan="2">CD (mm) ↓</td></tr><tr><td>Pattern</td><td>Full-only</td><td>Coverage-aware</td><td>Full-only</td><td>Coverage-a</td></tr><tr><td>Full</td><td>0.383</td><td>0.408</td><td>2.852</td><td>2.888</td></tr><tr><td>Base-missing</td><td>1.380</td><td>0.544</td><td>3.913</td><td>3.089</td></tr><tr><td>Apex-missing</td><td>0.998</td><td>0.520</td><td>3.303</td><td>2.989</td></tr><tr><td>Mid-block</td><td>10.868</td><td>0.963</td><td>8.464</td><td>3.702</td></tr><tr><td>Uniform-sparse</td><td>2.122</td><td>0.598</td><td>4.302</td><td>3.139</td></tr></table>

Edge Ratio P95, where values closer to 1 indicated lower local edge distortion. The complete model achieved the best performance across all three metrics. Compared with Centerline / Full-only, it reduced UV MSE and CD by 67.3% and 15.3%, respectively. Table 4 and Fig. 5 (a) showed that coverage-aware training caused only a minor change under full coverage but consistently improved all incomplete-coverage settings. The largest gain occurred for the mid-block pattern, where UV MSE and CD were reduced by 91.1% and 56.3%, respectively, demonstrating improved robustness to base-missing, apex-missing, midblock, and uniformly sparse SAX inputs.

The similar performance of the two full-only variants suggested that both rigid and centerline alignment supported reliable reconstruction under complete coverage. Centerline alignment was nevertheless retained to provide a consistent geometric basis for correspondence retrieval across slices and subjects. As shown in Fig. 5 (b), the learned UV-domain completion model also preserved globally smooth and coherent surfaces in the presence of local contour misalignment or irregular variation. Fig. 5 (c) further showed that direct UV interpolation produced stripelike artifacts and that direct conversion of $\mathbf { X } ^ { \mathrm { o b s } }$ recovered only locally observed geometry, whereas the proposed network generated complete and smooth myocardial surfaces. WrapConv2D improved continuity across the circumferential seam, while region-focused supervision reduced triangle stretching and distortion near the apex. These observations were consistent with the improvements in UV MSE and Edge Ratio P95 and indicated that global CD alone did not fully reflect localized geometric defects.

## 4.6. Utility of the Canonical UV Representation

## 4.6.1. Myocardial Shape and Functional Quantification

The reconstructed surfaces were used to derive framewise LV volume and global wall-thickening trajectories, with wall thickening computed from temporal changes in the paired epicardial-endocardial distances on the shared UV domain. As shown in Fig. 6 (a), the predicted curves closely followed the references throughout the cardiac cycle and preserved disease-specific patterns, including enlarged ventricular volumes and reduced wall thickening in DCM and MINF, and increased thickening in HCM. Across the three datasets, the case-level motion EPE was 1.297 ± 0.643 mm, indicating accurate point-wise motion across cardiac phases. Fig. 6 (b) further showed strong agreement in functional measurements across the three datasets, with an EDV MAE of 3.3 mL $\left( R ^ { 2 } \ = \ 0 . 9 9 1 \right)$ and an LVEF MAE of 1.1% $( R ^ { 2 } ~ = ~ 0 . 9 8 8 )$ . These results demonstrated that the reconstructed 4D surfaces preserved both disease-related myocardial dynamics and clinically relevant ventricular function.

## 4.6.2. UV-Domain Myocardial Scar Localization from Cine MRI

To further investigate whether the reconstructed 4D myocardial geometry captured clinically relevant regional motion abnormalities, we evaluated myocardial scar footprint localization on the independent CineMyoPS dataset (Ding et al., 2025). Here, the scar footprint referred to the surface region obtained by projecting the myocardial scar along the wall onto the corresponding myocardial surface. From the reconstructed epicardial and endocardial sequences, we derived motion amplitude, fractional wall thickening, and cumulative displacement over the cardiac cycle. These features were compared with a healthy motion atlas constructed from approximately 200 normal cases to identify anatomically localized reductions in contraction. Because the reconstructed surfaces maintained point-wise correspondence across cardiac phases and subjects, the resulting abnormal-motion patterns could be directly mapped to LGE-derived scar regions in the shared UV domain. A lightweight predictor with WrapConv2D subsequently converted these motion-abnormality maps into continuous scar-footprint probability maps. As shown in Fig. 7, the predicted scar footprints were spatially consistent with the

M&Ms

![](images/e9da465d6ca78615956b54d454d4c7bead0233be75cb71c696fa99931e37b490.jpg)  
(a)

![](images/a89e3ce2df47fc53156adbda2173a27215bc2472dff0b6d084bde53fc2b8212c.jpg)  
Input

(b)  
![](images/063f9a5b5df1309ae370591d53517ad24fe111ffdfc2ea0e94fb9b83b25c3976.jpg)  
Interpolated

![](images/3f26ec867806eed2f2b6c30630d89a3641cec1dab575e7818bd67ec7416cd264.jpg)

![](images/446ad2635d09aab5278d873d9f96f6d5282f23d903069316efaa6218d35773d0.jpg)

![](images/f68ab975a0a4a168a1b8d81cc7e464790d1cd0d576cccf08409f00685f775166.jpg)  
�<sup>!"#</sup>  
Final Output

![](images/809001b81304561e437bc6637b0c4a58ff43cffa4f64af6a5b1487a51105a145.jpg)

![](images/d3332021b7e8ee69c32c600b6a464fbafad5bf4c7dc707a714b8fbbbaf09ced4.jpg)  
w/o Region Focus

![](images/29774dae9b49a18221a5fa94ead2f633be840c80dd71459573ccbe717a8b14b4.jpg)  
(c)  
Ours

Figure 5: (a) Full-only and coverage-aware reconstructions under five slice-coverage patterns. (b) Representative results for locally irregular contour observations. (c) Comparison of non-learning UV reconstruction and network completion (top), seam continuity without and with WrapConv2D (middle), and apical geometry without and with region-focused supervision (bottom).  
![](images/eb9b9b5ad62ac26e535b6022e2be938e5aa9d7121a188309abc8807337adf7f2.jpg)  
(a)

![](images/50183c6449b5ea59ec50867ee08ef65204afdfb6c872de9095a65ed19cfc52a1.jpg)  
Reference EDV (mL)

![](images/1e37dbe91e96e06c669ecd6db0280473a1ce3ddc90a4221173f51e5f8d162690.jpg)  
(b)  
Figure 6: Quantitative analysis of myocardial shape and function. (a) Predicted and reference LV volume and global wall-thickening trajectories for the five ACDC diagnostic groups; shaded regions indicate group-wise standard deviations. (b) Agreement between predicted and reference EDV and LVEF across ACDC, M&Ms, and M&Ms-2. ARV: abnormal right ventricle.

references in the AHA-17 representation, UV domain, and reconstructed 3D myocardium, indicating that the reconstructed 4D geometry preserved regional motion characteristics relevant to myocardial scar localization.

## 4.6.3. UV-Domain Generative Modeling

To demonstrate the broader utility of the canonical UV representation, temporally aligned epicardial and endocardial motion was converted into regular UV displacement fields and modeled using a sequence VAE followed by a conditional latent difusion model (LDM). Each UV sequence was first decomposed into the ED UV anatomy and 24 frames of ED-relative motion. The VAE compressed each 24×6×256×256 motion sequence into an 8×32×32 latent code. Conditioned on the ED UV and disease label, the LDM then generated a motion latent, which was decoded by the frozen VAE decoder and added back to the ED anatomy to recover a complete 25-frame UV sequence. Although the VAE was optimized solely for motion reconstruction, its latent space retained clear disease-related structure. As shown in Fig. 8 (a), NOR, DCM, and HCM exhibit distinguishable distributions along the first two linear discriminant axes. A classification probe trained on the motion latents achieved a held-out macro-F1 of 0.776 and a balanced accuracy of 0.778, indicating that the shared UV correspondence organizes temporal motion into a compact representation with consistent spatial semantics while preserving disease-related motion characteristics. The LDM generated complete 25-frame motion sequences for all heldout cases, achieving an overall CD of 3.929 mm. Fig. 8 (b) further shows that the generated sequences preserve the characteristic LV cavity-volume patterns of NOR, DCM, and HCM and remain consistent with the reference distributions. These results demonstrate that the canonical UV representation not only provides consistent anatomical correspondence across subjects and cardiac phases, but also enables conventional 2D generative models to capture disease-related 4D myocardial motion while retaining geometric and functional fidelity after reconstruction in 3D.

## 5. Discussion and Conclusion

This study introduced an anatomy-aligned UV representation for patient-specific 4D myocardial reconstruction from sparse SAX cine MRI. By unfolding the epicardial and endocardial surfaces onto a shared circumferentiallongitudinal domain, the framework transformed the irregular 3D reconstruction problem into structured twodimensional coordinate-field completion and directly associated sparse observations with anatomically corresponding surface locations. This shared organization preserved explicit anatomical correspondence across subjects and cardiac phases. The method achieved the best surface accuracy across ACDC, M&Ms, and M&Ms-2 with an inference time of only 0.8 s per frame, and showed particularly strong performance on the more heterogeneous M&Ms cohorts. The close agreement in EDV and LVEF, with MAEs of 3.3 mL and 1.1%, respectively, further indicated that the reconstructed sequences preserved clinically relevant ventricular geometry and function.

![](images/5a500161b037c8baf843ca8b4c3b4caf6fc8ccb1a0ec3cf008450dfca94d33a2.jpg)  
Figure 7: Qualitative myocardial scar localization in three representative CineMyoPS cases.

The experimental results clarified the contributions of the individual components. Coverage-aware sampling provided the largest improvement by reducing dependence on a fixed longitudinal slice distribution; compared with fullcoverage training, it substantially improved reconstruction under missing basal, apical, mid-block, and uniformly sparse observations. WrapConv2D maintained continuity across the circumferential seam, whereas region-focused supervision reduced localized distortion near the apex, demo strating that global surface-distance metrics alone were insuficient to characterize geometric quality in parameterizati sensitive regions. Beyond reconstruction, the shared UV correspondence supported several complementary applications. The reconstructed 4D geometry captured regional reductions in motion and wall thickening that could be mapped to LGE-derived scar regions in the independent CineMyoPS cohort. The same representation also converted myocardial motion into regular UV displacement fields compatible with conventional VAE and latent difusion architectures. The resulting latent features retained disease-related motion information, while the generated trajectories preserved both surface geometry and global ventricular function. Together, these experiments showed that the canonical UV domain served not only as a reconstruction space but also as a common representation for functional analysis, pathology localization, and generative modeling.

![](images/3f4cd356936ddeafba630a285f00391f4bc6e3ba98f05129edc3415bc3476a2a.jpg)  
Figure 8: Conditional myocardial motion generation and analysis in the UV domain. (a) Projection of the VAE motion latents onto the first two linear discriminant axes, showing disease-related distributions of NOR, DCM, and HCM. (b) Reference and generated normal ized LV cavity-volume curves, where the horizontal axis denotes the normalized cardiac phase and the vertical axis denotes $V ( t ) / V ( 0 )$ IQR: interquartile range.

Several limitations remain. Separate models were trained for each dataset, and the current formulation focused on the left-ventricular myocardium; extending the framework to a unified multi-dataset model and more complex wholeheart anatomy will require additional parameterization strate gies. Although periodic convolution and region-focused constraints reduced seam and apical artifacts, reconstruction quality still depended on the canonical template and its UV mapping. The scar-localization experiment was primarily qualitative and used LGE-derived labels for supervision, while the generative evaluation involved a relatively small cohort containing only NOR, DCM, and HCM cases. Future work should employ larger external cohorts and include quantitative scar assessment, uncertainty estima-<sup>n-</sup>tion, and evaluation in downstream simulation and prognostic tasks. Despite these limitations, anatomy-aligned UV surface-field learning provided an accurate, eficient, and extensible framework for correspondence-aware 4D myocardial reconstruction from sparse cine MRI.

## References

Bai, W., Shi, W., de Marvao, A., Dawes, T.J., O’Regan, D.P., Cook, S.A., Rueckert, D., 2015. A bi-ventricular cardiac atlas built from

1000+ high resolution mr images of healthy subjects and an analysis of shape and motion. Medical image analysis 26, 133–145.

Bayer, J., Prassl, A.J., Pashaei, A., Gomez, J.F., Frontera, A., Neic, A., Plank, G., Vigmond, E.J., 2018. Universal ventricular coordinates: A generic framework for describing position within the heart and transferring data. Medical image analysis 45, 83–93.

Beetz, M., Banerjee, A., Grau, V., 2024. Modeling 3d cardiac contraction and relaxation with point cloud deformation networks. IEEE Journal of Biomedical and Health Informatics 28, 4810– 4819.

Bernard, O., Lalande, A., Zotti, C., Cervenansky, F., Yang, X., Heng, P.A., Cetin, I., Lekadir, K., Camara, O., Ballester, M.A.G., et al., 2018. Deep learning techniques for automatic mri cardiac multi-structures segmentation and diagnosis: is the problem solved? IEEE transactions on medical imaging 37, 2514–2525.

Campello, V.M., Gkontra, P., Izquierdo, C., Martin-Isla, C., Sojoudi, A., Full, P.M., Maier-Hein, K., Zhang, Y., He, Z., Ma, J., et al., 2021. Multi-centre, multi-vendor and multi-disease cardiac segmentation: the m&ms challenge. IEEE Transactions on Medical Imaging 40, 3543–3554.

Chen, Y., Yang, J., Mercadier, D.S., Le, H., Fua, P., 2024. Medtet: An online motion model for 4d heart reconstruction. arXiv preprint arXiv:2412.02589 .

Chen, Y., Yang, J., Mercadier, D.S., Le, H., Schwitter, J., Fua, P., 2025. End-to-end 4d heart mesh recovery across full-stack and sparse cardiac mri. arXiv preprint arXiv:2509.12090 .

Ding, W., Li, L., Qiu, J., Lin, B., Yang, M., Huang, L., Wu, L., Wang, S., Zhuang, X., 2025. Cinemyops: Segmenting myocardial pathologies from cine cardiac mr. IEEE Transactions on Medical Imaging .

Fu, X., Wu, P., Li, Y., Luo, X., Jiang, Z., Mei, J., Lu, J., Teng, G.J., Zhou, S.K., 2025. Dyna3dgr: 4d cardiac motion tracking with dynamic 3d gaussian representation, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 164–174.

Gaggion, N., Matheson, B.A., Xia, Y., Bonazzola, R., Ravikumar, N., Taylor, Z.A., Milone, D.H., Frangi, A.F., Ferrante, E., 2025. Multi-view hybrid graph convolutional network for volume-tomesh reconstruction in cardiovascular mri. Medical Image Analysis , 103630.

Geng, C., Peng, S., Xu, Z., Bao, H., Zhou, X., 2023. Learning neural volumetric representations of dynamic humans in minutes, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8759–8770.

G¨uler, R.A., Neverova, N., Kokkinos, I., 2018. Densepose: Dense human pose estimation in the wild, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7297– 7306.

Hu, T., Hong, F., Liu, Z., 2024. Surmo: surface-based 4d motion modeling for dynamic human rendering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6550–6560.

Huang, B., Zhang, T., Wang, Y., 2022. Object-occluded human shape and pose estimation with probabilistic latent consistency. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 5010–5026.

Isensee, F., Jaeger, P.F., Kohl, S.A., Petersen, J., Maier-Hein, K.H., 2021. nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods 18, 203–211.

Joyce, T., Buoso, S., Stoeck, C.T., Kozerke, S., 2022. Rapid inference of personalised left-ventricular meshes by deformation-based differentiable mesh voxelization. Medical image analysis 79, 102445.

Li, L., Camps, J., Rodriguez, B., Grau, V., 2024a. Solving the inverse problem of electrocardiography for cardiac digital twins: A survey. IEEE Reviews in Biomedical Engineering 18, 316–336.

Li, L., Camps, J., Wang, Z.J., Beetz, M., Banerjee, A., Rodriguez, B., Grau, V., 2024b. Toward enabling cardiac digital twins of myocardial infarction using deep computational models for inverse inference. IEEE transactions on medical imaging 43, 2466–2478.

Li, L., Ding, W., Huang, L., Zhuang, X., Grau, V., 2023. Multimodality cardiac image computing: A survey. Medical image

analysis 88, 102869.

Liu, X., Yuan, X., Chan, M.Y., Sia, C.H., Li, L., 2026. Cinemesh4d: Personalized 4d whole heart reconstruction from sparse cine mri. arXiv preprint arXiv:2605.13994 .

Luo, Y., Sesia, D., Wang, F., Wu, Y., Ding, W., Hasan, K., Huang, J., Shi, F., Shah, A., Kaura, A., et al., 2026. Explicit diferentiable slicing and global deformation for cardiac mesh reconstruction. Medical image analysis , 103999.

Ma, Q., Meng, Q., Qiao, M., Matthews, P.M., O’Regan, D.P., Bai, W., 2025. Cardiacflow: 3d+ t four-chamber cardiac shape completion and generation via flow matching, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 89–99.

Ma, Q., Meng, Q., Wu, Y., Wang, S., Qiao, M., Niederer, S., O’Regan, D.P., Matthews, P.M., Bai, W., 2026. Learning a dynamic four-chamber shape model of the human heart for 95,695 uk biobank participants. arXiv preprint arXiv:2603.28711 .

Mart´ın-Isla, C., Campello, V.M., Izquierdo, C., Kushibar, K., Sendra-Balcells, C., Gkontra, P., Sojoudi, A., Fulton, M.J., Arega, T.W., Punithakumar, K., et al., 2023. Deep learning segmentation of the right ventricle in cardiac mri: the m&ms challenge. IEEE Journal of Biomedical and Health Informatics 27, 3302–3313.

Meng, Q., Bai, W., O’Regan, D.P., Rueckert, D., 2023. Deepmesh: Mesh-based cardiac motion tracking using deep learning. IEEE transactions on medical imaging 43, 1489–1500.

Meng, Q., Qin, C., Bai, W., Liu, T., De Marvao, A., O’Regan, D.P., Rueckert, D., 2022. Mulvimotion: Shape-aware 3d myocardial motion tracking from multi-view cardiac mri. IEEE transactions on medical imaging 41, 1961–1974.

Mufoletto, M., Hermida, U., Mauger, C., Suinesiaputra, A., Xu, Y., Burns, R., Pankewitz, L., McCulloch, A.D., Petersen, S.E., Rueckert, D., et al., 2025. Neural implicit heart coordinates: 3d cardiac shape reconstruction from sparse segmentations. arXiv preprint arXiv:2512.19316 .

Qiao, M., McGurk, K.A., Wang, S., Matthews, P.M., O’Regan, D.P., Bai, W., 2025a. A personalized time-resolved 3d mesh generative model for unveiling normal heart dynamics. Nature Machine Intelligence , 1–12.

Qiao, M., Zheng, J., Zhang, W., Ma, Q., Li, L., Kainz, B., O’Regan, D.P., Matthews, P.M., Niederer, S., Bai, W., 2025b. Mesh4d: A motion-aware multi-view variational autoencoder for 3d+ t mesh reconstruction, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 343–353.

Schuler, S., Pilia, N., Potyagaylo, D., Loewe, A., 2021. Cobiveco: Consistent biventricular coordinates for precise and intuitive description of position in the heart–with matlab implementation. Medical Image Analysis 74, 102247.

Sinclair, M., Schuh, A., Hahn, K., Petersen, K., Bai, Y., Batten, J., Schaap, M., Glocker, B., 2022. Atlas-istn: joint segmentation, registration and atlas construction with image-and-spatial transformer networks. Medical Image Analysis 78, 102383.

Sun, S., Han, K., Kong, D., Tang, H., Yan, X., Xie, X., 2022. Topology-preserving shape reconstruction and registration via neural difeomorphic flow, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20845– 20855.

Wickramasinghe, U., Remelli, E., Knott, G., Fua, P., 2020. Voxel2mesh: 3d mesh model generation from volumetric data, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 299–308.

Xia, Y., Chen, X., Ravikumar, N., Kelly, C., Attar, R., Aung, N., Neubauer, S., Petersen, S.E., Frangi, A.F., 2022. Automatic 3d+ t four-chamber cmr quantification of the uk biobank: integrating imaging and non-imaging data priors at scale. Medical Image Analysis 80, 102498.

Xu, Y., Xu, H., Sinclair, M., Puyol-Ant´on, E., Niederer, S.A., Chiribiri, A., Williams, S.E., Williams, M.C., Young, A.A., 2024. Improved 3d whole heart geometry from sparse cmr slices, in: International Workshop on Statistical Atlases and Computational Models of the Heart, Springer. pp. 43–52.

Yang, X., Yuan, X., Li, H., Chen, L., Liu, Y., Li, L., 2026. Repcm: Region-specific and phenotype-adaptive bi-ventricular cardiac motion synthesis. arXiv preprint arXiv:2605.21237 .

Ye, M., Yang, D., Kanski, M., Axel, L., Metaxas, D., 2023. Neural deformable models for 3d bi-ventricular heart shape reconstruction and modeling from 2d sparse cardiac magnetic resonance imaging, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14247–14256.

Yuan, X., Liu, C., Wang, Y., 2023. 4d myocardium reconstruction with decoupled motion and shape model, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21252–21262.

Zhang, Z., Liu, Z., Zhang, Z., Cui, Z., 2025. Mask2surface: Motion correction and super-resolution for cardiac surface reconstruction using latent difusion, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer. pp. 335–344.

Zhao, Z., Xie, W., Zuo, B., Wang, Y., 2023. Skeleton extraction for articulated objects with the spherical unwrapping profiles. IEEE Transactions on Visualization and Computer Graphics 30, 3731– 3748.

Zufi, S., Kanazawa, A., Jacobs, D.W., Black, M.J., 2017. 3d menagerie: Modeling the 3d shape and pose of animals, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6365–6373.