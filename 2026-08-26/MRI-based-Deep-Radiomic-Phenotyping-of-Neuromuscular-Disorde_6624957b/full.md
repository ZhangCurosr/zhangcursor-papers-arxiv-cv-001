# MRI-based Deep Radiomic Phenotyping of Neuromuscular Disorders: A Topology-driven Characterization

Martyna Żur<sup>1</sup>, Łukasz Piórecki<sup>1</sup>, Marek Socha<sup>1</sup>, Jordi Diaz-Manera<sup>2</sup>, Jose Verdu Diaz<sup>2</sup>, Volker Straub<sup>2</sup>, Rossella Tupler<sup>3</sup>, Joanna Polańska<sup>1</sup>

<sup>1</sup>Department of Data Science and Engineering, Silesian University of Technology, Gliwice, Poland <sup>2</sup>John Walton Muscular Dystrophy Research Centre, Newcastle University, Newcastle upon Tyne, UK <sup>3</sup>Department of Life Sciences, University of Modena and Reggio Emilia, Modena, Italy

## Abstract

Abstract: Quantitative assessment of muscle MRI is crucial for monitoring neuromuscular disorders (NMD). This study introduces an automated radiomic phenotyping framework based on original features engineered across five main architectural domains: quantitative morphometry, spatial distribution, geometric shape, interactions between progressive fat replacement stages, and graph-based topology. Utilizing 1184 MRI scans from the CoMPaSS-NMD project, we map the complex 3D architecture of heterogeneous intramuscular lipodegeneration into objective, morphologically interpretable biomarkers. We introduce a graph-based skeletonization of fat infiltrates to quantify muscle architectural changes, establishing a multi-dimensional extension of traditional, spatially-agnostic volume metrics by mapping topological networks across the entire 3D muscle volume. Statistical screening via non-parametric Kruskal-Wallis analysis confirmed the discriminative power of these novel descriptors across the genetic hierarchy. Notably, topological network metrics (e.g., SF1\_Skel\_Nodes, �<sup>2</sup> = 0.2656) and interface dynamics metrics (e.g., SF2\_To\_SF1\_Dist\_Min, �<sup>2</sup> = 0.2092) demonstrated substantial efect sizes, providing deeper structural insights than classical volumetric assessments. Post-hoc pairwise evaluations and UMAP projections further indicated the capability of these topologica and 3D geometric invariants to capture disease-specific macroscopic infiltration patterns. These results demonstrate that global architectural features represent a highly promising class of biomarkers for diferential diagnosis, ofering new avenues for tracking longitudinal disease dynamics in neuromuscular diagnostics. The developed automated feature extraction pipeline is integrated and available within the MUSCAT (MUSCle fAt Topology) library.

## 1 Introduction

The clinical evaluation of muscular dystrophies relies heavily on Magnetic Resonance Imaging (MRI) to assess progressive muscle degeneration and weakness. Over time, functional muscle tissue (low intensity) is gradually replaced by pathological manifestations, which can be radiologically stratified into moderate fat replacement (intermediate intensity) and severe fat replacement (high intensity). Identifying the precise morphological patterns of this replacement is paramount, as many neuromuscular disorders (NMDs) present with overlapping symptoms but require fundamentally diferent therapeutic interventions.

Historically, the evaluation of these scans has been guided by semi-quantitative visual grading systems, most notably the Mercuri scale. This foundational system successfully established a standard for radiological reporting and remains a cornerstone of clinical diagnostics. However, as the field moves towards early-stage interventions and gene therapies, there is a growing necessity to complement visual assessment with automated, continuous metrics. While traditional quantitative MRI (qMRI) techniques, such as Fat Fraction (FF) mapping—provide reproducible volumetric measurements, they compress the 3D complexity of the muscle architecture into a single scalar value. A muscle where 30% of fat constitutes a solid mass and one where 30% manifests as an aggressive, reticular "moth-eaten" network are clinically distinct, yet both generate identical FF values. Therefore, rather than seeking to replace FF, which remains a fundamental baseline metric, our Graph-Radiomics framework is designed to be highly complementary. It explicitly captures the 3D spatial heterogeneity that global FF mathematically dilutes.

## 2 State of the Art: Feature Extraction in Muscle MRI

## 2.1 Visual Grading and Quantitative Volumetry

The Mercuri scale remains a milestone in standardizing radiological reporting. Nonetheless, it lacks the sensitivity required for sub-clinical disease monitoring due to its discrete, categorical nature [1]. While Dixon-based Fat Fraction (FF) mapping overcomes inter-observer bias, it relies on global volumetry. As established in recent literature, relying exclusively on volumetric burden is a methodological limitation; FF is "blind" to the structural topology, failing to distinguish between benign focal fat accumulation and pathological, invasive networks. Furthermore, clinical practice often assumes symmetric progression. Mutations like DUX4 (FSHD) present with extreme, patchy asymmetry. Averaging standard volumetric scores from left and right limbs efectively masks these unique, genotypic signatures, highlighting the need for an advanced geometric profiling.

## 2.2 Radiomics and the "Bag-of-Pixels" Problem

Classical radiomics standardizes feature extraction using matrices like GLCM or GLSZM to identify textural changes. However, these methods operate on a "bag-ofpixels" paradigm. By aggregating pixels into global statistical matrices, they become spatially blind. They cannot distinguish between scattered "islands" of fat and a unified, invasive network. Consequently, due to the immense manual burden of 3D annotation, a vast majority of current clinical studies restrict their analysis to a single mid-thigh cross-section (e.g., at 33% of the femur length) [2, 3]. This "single-slice" paradigm intrinsically assumes longitudinal uniformity, severely hindering the evaluation of diseases that exhibit highly localized or progressing proximo-distal degeneration gradients. Comprehensive 3D spatial profiling is therefore mandatory to capture the true topological footprint of the pathology.

## 2.3 The Segmentation Bottleneck and Atlas Failure

The extraction of quantitative biomarkers intrinsically depends on accurate tissue segmentation. However, manual delineation is highly operator-dependent and impractically time-consuming for large 3D datasets. While automated atlas-based and active contour methods have shown eficacy in healthy cohorts, they often fail in advanced neuromuscular disorders [2]. Extreme fatty infiltration and severe muscle atrophy obscure the fascia lata and erase natural inter-muscular boundaries, rendering predefined healthy atlases inefective. This anatomical breakdown necessitates the use of adaptive, unsupervised segmentation frameworks that rely on probabilistic intensity distributions rather than rigid morphological priors, ensuring the extraction pipeline operates on accurately deconvoluted tissue masks.

## 2.4 Multi-Center Heterogeneity and Scale Information Leakage

The transition towards data-driven diagnostic models is further complicated by the heterogeneity of retrospective clinical annotations. As demonstrated by recent largescale multi-center frameworks, aggregating visual scores from various institutions introduces difering quantization schemes (e.g., 4-point vs. 6-point discrete scales) [4]. If specific rare diseases are predominantly annotated using a distinct scale within a historical cohort, machine learning models may inadvertently learn the grading scale artifact rather than the underlying pathophysiology, a phenomenon known as "scale information leakage." This highlights the critical need for automated pipelines that process continuous, image-driven spatial metrics directly, bypassing human quantization bias entirely.

## 2.5 The Deep Learning vs. Interpretability Gap

While Convolutional Neural Networks (CNNs) have achieved high classification accuracy, they face the "Data-Hunger" barrier in rare diseases, where expert-annotated 3D masks are practically unattainable [4]. Furthermore, end-to-end CNNs often sufer from poor cross-scanner generalizability. Models trained on specific hardware configurations or field strengths (e.g., 1.5T vs. 3.0T) tend to overfit to hardware-specific signal intensities and contrast variations, leading to suboptimal performance on external datasets [5]. In contrast, explicitly extracted topological invariants, such as skeletal graphs, map the underlying physical geometry and connectivity of the tissue, ofering a more robust and hardware-agnostic representation of the disease. In high-stakes diagnostics, clinicians require interpretable, physical metrics. Unlike "black-box" deep learning, our Graph-Radiomics framework is designed to yield biomarkers that are directly translatable to the biological reality of fascicular disruption.

## 2.6 The Tractography Limitation in Lipodegeneration

Historically, 3D structural analysis in muscle MRI has been strictly dominated by Difusion Tensor Imaging (DTI) and tractography. DTI maps the orientation of healthy contractile muscle fibers by tracking the anisotropic difusion of water molecules [6]. While highly efective in healthy cohorts or sports medicine, DTI breaks down in advanced muscular dystrophies. As macroscopic fat replaces muscle tissue, the water signal diminishes, introducing massive partial volume efects and signalto-noise (SNR) drops that render architectural tracking highly unreliable [6]. Rather than struggling to map the surviving muscle fibers through heavy noise, our Graph-Radiomics framework reverses the diagnostic paradigm: it directly models and parameterizes the 3D architecture of the invasive fat network itself, remaining robust where DTI loses reliability.

## 2.7 Statistical vs. Morphological Graph-Radiomics

To overcome the limitations of conventional, spatiallyblind radiomics, the broader medical imaging community is transitioning towards Graph-Radiomics. Recent breakthroughs in oncology have introduced graph-based descriptors (e.g., GrRAIL) to decode intra-tumoral heterogeneity [7]. However, in oncological applications, graph nodes typically represent abstract mathematical clusters of statistical intensity, and edges represent generalized feature similarity. Their graphs form an abstract manifold of statistical heterogeneity. This paper introduces the translation of Graph-Radiomics to rare neuromuscular disorders via a uniquely morphological approach. In our framework, the graph nodes and edges are not abstract statistical clusters, but direct representations of the physical skeletal infrastructure of fat infiltrates. This ensures that the extracted structural complexity directly mirrors the actual macroscopic anatomy, bridging the gap between abstract graph theory and clinically interpretable tissue morphology.

## 2.8 Statistical Rigor: Beyond P-Value

Finally, we address the common methodological pitfalls in radiomic discovery. Standard radiomic studies often extract hundreds of features and report only those with $p < 0 . 0 5$ . However, according to modern statistical guidelines, �-values are highly sensitive to sample size and do not reflect the clinical magnitude of a finding. We employ the Epsilon-squared $\bar { ( \epsilon ^ { 2 } ) }$ efect size, as recommended in contemporary biostatistics [8], to ensure that our proposed features are not merely statistical noise, but reflect substantial pathological variations between genetic phenotypes.

## 3 Methods: Feature Engineering and Mathematical Basis

## 3.1 Study Population and Tissue Segmentation

This study initially evaluated 1184 magnetic resonance imaging (MRI) scans acquired from patients enrolled in the multi-center CoMPaSS-NMD project. The analyzed cohort encompasses five distinct genetic neuromuscular phenotypes: DYSF (n = 458), CAPN3 $\left( \mathrm { n } = 3 4 3 \right)$ , GNE $( \mathrm { n } = 1 4 6 )$ , DMPK (n = 135), and DUX4 $\left( \mathrm { n } = \mathrm { 1 0 2 } \right)$ . Prior to topological feature engineering, all MRI volumes underwent an automated intensity-based segmentation pipeline within the MUSCAT framework.

This unsupervised preprocessing step utilizes a Gaussian Mixture Model (GMM) to decompose the tissue intensity distributions. It robustly isolates the macroscopic functional muscle structure from pathological infiltration. The lipodegeneration is stratified into three distinct masks for comprehensive 3D profiling: Subtype 1 Fat (SF1), representing moderate fat replacement (intermediate intensity); Subtype 2 Fat (SF2), representing severe, macroscopic fat (high intensity); and General Fat (GF), which is the union of SF1 and SF2, representing the unified pathological burden. To capture the holistic disease footprint and ensure statistical robustness, features extracted from both the left and right legs were mathematically aggregated (averaged using the mean) and treated as a single unified profile per patient.

The proposed feature extraction framework operates across five main architectural domains to holistically capture the 3D topology of the entire muscle volume. A rigorous correlation-based filter (Mutual Information) refined the initial feature set to 29 unique, orthogonal spatial biomarkers. The mathematical formulations below are uniformly applied to the GF, SF1, and SF2 masks independently, ensuring that the spatial representation is deterministic, reproducible, and hardware-agnostic. A comprehensive overview of the clinical and morphological interpretation of these five domains is provided in Table 1. Furthermore, the exact nomenclature and categorization of all extracted variables across the GF, SF1, and SF2 masks are explicitly detailed in the feature dictionary (Table 2).

## 3.2 Volumetric Macro-structure and Morphological Fragmentation

Basic volumetrics are initially derived from voxel counting multiplied by 3D spatial spacing to obtain the absolute volumetric extent $( V _ { t o t a l } )$ in $\mathrm { m m ^ { 3 } }$

$$
V _ { t o t a l } = N \times \left( s _ { x } \cdot s _ { y } \cdot s _ { z } \right)
$$

where $s _ { x } , s _ { y } , s _ { z }$ represent the physical voxel spacing of the MRI scan (in millimeters). Crucially, to eliminate anthropometric bias $( { \mathrm { i . e . } }$ , natural variations in patient height and limb size), this absolute volume is strictly not used as a standalone biomarker. Instead, it is normalized against the total anatomical compartment volume, yielding the relative disease burden (\_Global\_Pct).

Fragmentation is calculated using 3D Connected Component Analysis based on 26-connectivity to identify discrete, non-overlapping spatial structures (islands) $C =$ $\{ C _ { 1 } , C _ { 2 } , \ldots , C _ { k } \}$ , implemented computationally via the scikit-image library [13]. The 26-neighborhood criterion was deliberately chosen over 6- or 18-connectivity to prevent artificial computational fragmentation of thin, diagonally oriented fat infiltrates that spread across intermuscular fasciae.

## 3.3 Spatial Distribution and Vectoring

These features are based on exact Euclidean distance transforms and the classical theory of 3D image moments [12] for calculating geometric centroids. The Normalized Center of Mass (���) provides spatial coordinates tracking the absolute center of gravity of the pathology:

$$
C o M _ { x } = \frac { 1 } { N } \sum _ { i \in \Omega } i _ { x } \cdot V ( i )
$$

where Ω represents the segmented tissue region, $N$ is the total number of voxels within the mask, $i _ { x }$ is the specific coordinate, and $V ( i )$ is the binary mask value (1 or 0) at that coordinate. Similar equations apply to the Y and Z axes.

## 3.4 Geometric Shape and Orientation

Evaluated using Principal Component Analysis (PCA) on the voxel coordinates to extract the spatial covariance matrix describing spatial variance. By calculating the orthogonal eigenvalues $\left( \lambda _ { 1 } \ \geq \ \lambda _ { 2 } \ \geq \ \lambda _ { 3 } \right)$ , we quantify the macroscopic geometric shape. Elongation is defined as the square root of the ratio of the two largest eigenvalues:

$$
E l o n g a t i o n = \sqrt { \frac { { \lambda _ { 2 } } } { { \lambda _ { 1 } } } }
$$

Similarly, Flatness is defined as $\sqrt { \lambda _ { 3 } / \lambda _ { 2 } }$

Table 1: Clinical Translation of Engineered 3D Graph-Radiomic Biomarkers
<table><tr><td>tic</td><td>Topological Characteris- Morphological / Radiological Pat- Clinical Interpretation in NMDs tern</td><td></td></tr><tr><td>Topological Complexity [9, 7, 4] (Skel_Nodes &amp; Skel_Cycles)</td><td>Measures the density of branching inter- sections and closed loops within the fat skeleton.</td><td>High values indicate an aggressive, retic- ular (&quot;web-like&quot;) inter-fascicular perme- ation, heavily disrupting contractile ge- ometry. Explicitly identifies the highly</td></tr><tr><td>Fragmentation [10, 11, 4] (Blob_Count &amp; Size)</td><td>Quantifies the absolute number and me- dian size of disconnected, discrete spatial fat islands (Connected Components).</td><td>branched networks often seen in severe DUX4 phenotypes. A high count combined with small frag- ment size reflects a scattered, multi-focal &quot;moth-eaten&quot; pattern. This translates</td></tr><tr><td>Spatial Vectoring [12, 4]</td><td>the pathology within the anatomical com-</td><td>subjective visual grading into a continu- ous metric for early, patchy lipodegener- ation. Maps the exact 3D center of gravity of Provides objective coordinates to distin- guish specific asymmetrical involvement</td></tr><tr><td>(Center of Mass (CoM)) Interface Dynamics [1, 4]</td><td>partment.</td><td>(e.g., anterior vs. posterior dominance), tracking spatial disease progression vec- tors explicitly. Calculates the shortest Euclidean dis- Evaluates the synergistic growth at the</td></tr><tr><td>(SF2_To_SF1_Dist_Min) Geometric</td><td>tance between severe fat replacement (SF2) and moderate fat replacement (SF1) masks. Orientation Uses PCA eigenvectors to determine the</td><td>pathological tissue interface. Extremely low distances indicate densely inter- twined degenerative processes, separating LGMD variants (CAPN3 vs. DYSF).</td></tr><tr><td>[10, 4] (MainAxis_Z &amp; Flatness)</td><td>stretch. ping.</td><td>Determines if fat infiltration propa- primary axis of spatial variance and shape gates as a continuous longitudinal cylin- der or spreads horizontally across inter- muscular planes. This geometric profiling highlights the necessity of full 3D map-</td></tr></table>

## 3.5 Interface Dynamics Between Fat Replacement Stages

Calculated using 3D spatial dilation and distance mapping to evaluate the physical intersection and proximity between the severe fat replacement mask $\left( F _ { 2 } \right)$ and the moderate fat replacement mask $\left( F _ { 1 } \right)$ . The minimum interface distance is calculated as:

$$
D _ { m i n } = \operatorname* { m i n } _ { \substack { p \in F _ { 2 } , q \in F _ { 1 } } } \| p - q \| _ { 2 }
$$

where � and � represent the spatial coordinates belonging to the respective tissue classes, and $\| p - q \| _ { 2 }$ denotes the Euclidean distance between their boundaries.

## 3.6 3D Graph-based Topology (Network Architecture)

Derived from the Medial Axis Transform (MAT) theory initially proposed by Blum [9], the 3D volumetric fat mask undergoes morphological skeletonization. This complex morphological reduction is computationally executed producing a 1-voxel-thick medial line while strictly preserving the underlying topology.

Once skeletonized, the fat infrastructure is formally mapped into a mathematical undirected graph $\begin{array} { r l } { G } & { { } = } \end{array}$ $( V , E )$ , where � is the set of topological vertices (branching nodes and terminal endpoints) and � represents the set of connecting skeletal edges. The vertices � are defined by utilizing a continuous Coordinate k-D Tree to query voxel neighborhoods within a spatial radius of $r ~ \leq ~ 1 . 8$ voxels.

Crucially, the Total Network Length (Skel\_Len\_Total) is computed by multiplying the cumulative length of all edges by the mean spatial spacing of the MRI scan. Furthermore, the Number of Closed Loops $( S k e l \_ C y c l e s )$ is mathematically quantified using the cyclomatic complexity formula:

$$
C = | E | - | V | + N _ { c o m p }
$$

where $| E |$ is the total number of connected skeletal edges, |� | is the number of branch/terminal vertices, and $N _ { c o m p }$ signifies the number of fully disconnected spatial subgraphs.

## 4 Results

## 4.1 Global Variance and Efect Size

The statistical screening of the engineered biomarkers was conducted using the non-parametric Kruskal-Wallis H-test [14]. This method securely evaluated the global variance, testing the alternative hypothesis that at least one disease group difers significantly from the rest.

Table 2: Feature Dictionary: Explicit categorization of the engineered 3D spatial biomarkers by architectural domain. With the exception of Interface Dynamics and overall muscle percentage, features are systematically extracted for the General Fat (GF), Subtype 1 (SF1), and Subtype 2 (SF2) masks.
<table><tr><td>Architectural Domain</td><td>Engineered Biomarkers (Exact Nomenclature)</td></tr><tr><td>Topological Complexity</td><td>Skel_Nodes, _Skel_Cycles, _Skel_Components, _Skel_Len_Total (Each calculated independently for GF, SF1, and SF2)</td></tr><tr><td>Fragmentation &amp;</td><td>Blob_Count, _Global_Pct, _BlobSize_Mean, _BlobSize_Median,</td></tr><tr><td>Volumetry</td><td>BlobSize_Std, _BlobSize_Max (Each calculated independently for GF, SF1, and SF2), Muscle_Global_Pct</td></tr><tr><td>Spatial Vectoring &amp; Distribution</td><td>_CoM_X, _CoM_Y, _CoM_Z, _Depth_Mean, _Depth_Median, _Depth_Std, _Depth_Max</td></tr><tr><td></td><td>(Each calculated independently for GF, SF1, and SF2)</td></tr><tr><td>Geometric Orientation</td><td>Elongation, _Flatness, _MainAxis_Z (Each calculated independently for GF, SF1, and SF2)</td></tr><tr><td>Interface Dynamics</td><td>SF2_To_SF1_Dist_Min, SF2_To_SF1_Dist_Mean, SF2_To_SF1_Contact_Ratio (Calculated exclusively between the SF2 and SF1 tissue boundaries)</td></tr></table>

![](images/894680a5faa01c86ab060f7b8e02f0c486d0a64e82d67e15422338d9325ff7a9.jpg)  
(a) Sparse/Fragmented Pattern

![](images/429d319dab9b94b2fc6151fd768ceca87dbf289fee1c5e594016e0e73af9605f.jpg)  
(b) Structured Linear Streaks

![](images/90d155e01e1ece62e987e6b929117014b0c0a6e66326a81203799bfbd81e3fbd.jpg)  
(c) Complex Reticular Web  
Figure 1: Visual representation of the extracted 3D Graph-Radiomics topology from macroscopic lipodegeneration masks. The original volumetric fat segments are reduced to 1-voxel-thick medial axis skeletons, represented by connecting paths (cyan edges), with computationally extracted branching intersections and terminal points (red nodes). These 3D spatial networks allow for the exact mathematical quantification of structural complexity, such as the density of inter-fascicular permeation, which standard volumetric methods fail to capture.

To explicitly quantify the magnitude of diferentiation between the genetic phenotypes, independent of sample size bias, the Epsilon-squared $\left( \epsilon ^ { 2 } \right)$ efect size was calculated [8]. The top 10 discriminative global features demonstrating the highest statistical power are reported in Table 3. Notably, topological network features such as SF1\_Skel\_Nodes achieved a large efect size of $\epsilon ^ { 2 } = $ 0.2656, and GF\_Skel\_Nodes achieved $\epsilon ^ { 2 } \ = \ 0 . 2 5 4 2$ , indicating profound diferentiation based on the 3D graph complexity of the pathology rather than simple volumetric measures.

## 4.2 Pairwise Post-Hoc Analysis

To identify local diferences in rank sums between specific disease subgroups, Dunn’s Post-Hoc tests [15] were executed. The pairwise evaluations demonstrated the capability of the topological and geometric features to isolate specific phenotypes based on their entire 3D volume. For instance, the global topological interface distance $( S F 2 \_ T o \_ S F 1 \_ D i s t \_ M i n )$ successfully isolated specific phenotypes, indicating varying degrees of intertwining between moderate and severe fat. Furthermore, topological descriptors such as GF\_Skel\_Nodes and SF1\_Skel\_Nodes yielded large pairwise efect sizes (e.g., isolating DUX4 from DMPK and DYSF with efect sizes � > 0.88). A comprehensive visual representation of these pairwise magnitudes is presented in Figure 2.

Detailed descriptive statistics (including mean, median, standard deviation, and quantiles) along with comprehensive Kruskal-Wallis findings for all engineered biomarkers are provided in Supplementary Table S1. Furthermore, the complete pairwise efect size matrices derived from Dunn’s post-hoc analysis for all genetic phenotypes are accessible in Supplementary Table S2.

![](images/4675562d74a4321c662046861dec7bc0dc6ed1953db4096b0d5c146743333d8c.jpg)  
(a) General Fat Topological Nodes

![](images/b0de6aec22cef3edc7ed4562d2c0a946682bfe2b7fa445fdcfb91c55a2749c47.jpg)  
(b) Moderate Fat Topological Nodes

![](images/bc11834545c912310911c007fb4e3decf88281c1d5f6d3eaaa92e52be5503191.jpg)  
(c) Severe Fat Skeletal Cycles

![](images/ffb7216e6ee51ac38a994c6f96798e44302f09d07350cfc6b5e6b32d5a742135.jpg)  
(d) General Fat Median Fragment Size

![](images/a080fbde7f2e708d494894169b5b8f2aa1292bd427a4bfd5f52ccf04281bf8d8.jpg)  
(e) Severe Fat Median Fragment Size

![](images/0d488bc5669d933a66a115ea5f9b328e95f2a60d4845fbf63a189e74928de256.jpg)  
(f) Severe-to-Moderate Fat Interface  
Figure 2: Pairwise efect size heatmaps derived from Dunn’s post-hoc analysis across the top-performing global descriptors. The matrices visualize the magnitude of diferentiation between the neuromuscular phenotypes. The accompanying color bar on the right defines the standardized efect size scale (r).

## 4.3 UMAP Dimensionality Reduction

To evaluate the holistic, multi-dimensional discriminative potential of the engineered global features, an unsupervised Uniform Manifold Approximation and Projection (UMAP) [16] was applied to project the high-dimensional feature space into a 2D manifold while preserving local and global data structures. All spatial biomarkers demonstrating at least a small global efect size $( \epsilon ^ { 2 } \ge 0 . 0 1$ , yielding 28 features) were utilized. To prevent the curse of dimensionality and filter out statistical noise, this high-dimensional profile was first reduced via Principal Component Analysis (PCA, retaining 95% of variance) before being projected into a 2D manifold without any exposure to the genetic labels (Figure 3).

To visually validate the spatial separation observed on the UMAP manifold, specifically the isolation of the DUX4 cohort, a direct structural comparison was performed using the GF\_Skel\_Nodes gradient (Figure 4). The projection maps the 2D embedding directly to the physical 3D architecture of the extracted tissue. Patients in the isolated DUX4 cluster (green frame) exhibit an extreme density of topological branching nodes, forming a complex, multi-directional network. In contrast, patients located in the main continuous LGMD cluster, such as DYSF variants (yellow frame), present with significantly fewer branching nodes, characterized by parallel, longitudinal fat propagation.

Subsequent to the diagnostic projection, we aimed to objectively identify inherent structural subpopulations. Phenotyping by Accelerated Refined Classification (PARC) was applied to the manifold, revealing five distinct topological clusters (Figure 5a). Furthermore, to decode the biological drivers of this separation, the values of individual spatial biomarkers were projected back onto the UMAP embeddings as gradient maps (Figure 5b-f). This feature gradient mapping successfully isolates the specific architectural signatures—such as orientation, skeletal branching, and fragmentation—that define each disease sub-cohort. The complete repository of gradient maps for all 63 extracted features is provided in Supplementary Figure S1.

## 5 Discussion

The integration of morphological graph-radiomics advances muscle MRI evaluation from purely volumetric fat estimation into precise architectural profiling. While absolute fat quantity remains a critical baseline metric to assess general disease burden, relying solely on it obscures fundamental pathological diferences between NMDs. Our 3D global methodology overcomes the limitations of classical radiomics and "black-box" models, providing an interpretable foundation for deep phenotyping.

## 5.1 Clinical Phenotype Translation and Biological Basis

The statistical findings and pairwise efect sizes derived from the heatmaps structurally validate observed clinical and biological realities. The pronounced effect sizes generated by global fragmentation metrics (SF2\_BlobSize\_Median) efectively isolate specific phenotypes characterized by patchy, multi-focal infiltrates. This structurally mirrors the early clinical phase of the disease, frequently detected as muscle edema prior to irreversible fat replacement.

Table 3: Top 10 discriminative engineered biomarkers identified by the Kruskal-Wallis H-test, sorted by $\epsilon ^ { 2 }$ efect size. A comprehensive evaluation is provided in Supplementary Table S1.
<table><tr><td>Category</td><td>Topological Characteristic</td><td>H-Stat</td><td> $\overline { { { \epsilon } ^ { 2 } } }$ </td></tr><tr><td>Topology</td><td>SF1 Skel_ Nodes</td><td>314.17</td><td>0.2656</td></tr><tr><td>Topology</td><td>GF_Skel_Nodes</td><td>300.68</td><td>0.2542</td></tr><tr><td>Fragmentation SF2_BlobSize</td><td>Median</td><td>299.81</td><td>0.2534</td></tr><tr><td>Fragmentation GF_BlobSize</td><td>Median</td><td>277.95</td><td>0.2349</td></tr><tr><td>Interface</td><td>SF2_To_SF1_</td><td>247.44</td><td>0.2092</td></tr><tr><td>Topology</td><td>Dist_Min GF_Skel_</td><td>222.73</td><td>0.1883</td></tr><tr><td>Fragmentation SF1_BlobSize</td><td>Components</td><td>219.20</td><td>0.1853</td></tr><tr><td>Topology</td><td>Median SF2_Skel</td><td></td><td>0.1562</td></tr><tr><td></td><td>Cycles</td><td>184.82</td><td></td></tr><tr><td>Spatial Vect. Fragmentation SF1_BlobSize</td><td>SF1_CoM_Z</td><td>159.87 155.51</td><td>0.1351 0.1315</td></tr><tr><td></td><td>Max</td><td></td><td></td></tr></table>

Conversely, topological markers like GF\_Skel\_Nodes structurally validate whether the degeneration progresses via a uniform, cohesive, and continuous tissue replacement or a highly branched network. The visual translation of this phenomenon can be directly observed in our topological representations (Figure 1), and is visually demonstrated by the UMAP projection, where DUX4 patients form a highly isolated cluster characterized by an extreme density of branching nodes. This distinct topological footprint objectively maps the varying degrees of inter-fascicular permeation across the whole muscle volume. It is important to note that while DUX4 variants are clinically characterized by severe left-right asymmetry, our bilateral averaging methodology did not mask this specific genotypic signature. Although averaging mathematically dilutes unilateral severity, the topological features—such as extreme skeletal branching density (Skel\_Nodes) and distinct multi-focal fragmentation—are so morphologically pronounced in the afected limb that the patient’s aggregated topological mean remains starkly distinct from symmetric pathologies. This proves that 3D morphological parameters capture the underlying nature of the architectural disruption, persisting powerfully even when fused into a single unified patient profile.

Crucially, the engineered global biomarkers successfully separated phenotypes that are frequently challenging to distinguish using standard volumetry. For instance, the interface dynamics metrics (SF2\_To\_SF1\_Dist\_Min) and center of mass parameters (SF1\_CoM\_Z) generated notable efect sizes, indicating that spatial positioning and tissue proximity are defining characteristics of certain dystrophies. Since many variants belong to the Limb-Girdle Muscular Dystrophy (LGMD) spectrum, they frequently exhibit macroscopic similarities that confound standard visual grading. Variations in median blob sizes (GF\_BlobSize\_Median) and geometric shapes efectively segregated these broader cohorts, demonstrating that extracting explicit 3D geometric signatures provides a robust mechanism to diferentiate allied genetic variants.

![](images/a4e71b5cf6c69a934d799e5db5960f49ed733b3975d0923e948f3447f5d68147.jpg)  
Figure 3: UMAP Projection of the patient cohort based on the global biomarkers exhibiting at least a small efect size $( \epsilon ^ { 2 } \ge 0 . 0 1 )$ , pre-processed with PCA to retain 95% of spatial variance. The visualization confirms the capability of this rigorously selected 3D feature subset to diferentiate genetic phenotypes based on their entire volume topology, notably isolating DUX4 and GNE variants into distinct spatial clusters.

Furthermore, spatial vectoring parameters such as SF1\_CoM\_Z efectively map the directional propagation of the disease. High diferentiation power in this domain reflects whether the fat infiltration tracks longitudinally along the myofibers or distributes uniformly, highlighting the necessity of full 3D coordinate mapping over singleslice evaluations.

## 5.2 Dimensionality, Morphological Continuum, and Feature Gradients

The UMAP projection (Figure 3) based on the discriminative global features $( \dot { \epsilon } ^ { 2 } \geq 0 . 0 1 )$ reveals a distinct biological pattern regarding both phenotypic divergence and disease progression. Because UMAP operates as an unsupervised dimensionality reduction technique, it groups patients based purely on the mathematical similarity of their spatial biomarkers without prior knowledge of their clinical diagnoses. This intrinsic capability to cluster genetic cohorts highlights the robust diagnostic value of the engineered features.

For instance, the inclusion of biomarkers with smaller global efect sizes captured subtle structural variations. This allowed the algorithm to distinctly isolate the extreme architectural disruption of the DUX4 cohort into a separate topological "island," while also segregating a distinct sub-cluster of GNE cases. Meanwhile, the broad LGMD spectrum (CAPN3 and DYSF), alongside the remaining GNE variants, forms a more continuous main cluster. This overlap accurately reflects their shared overarching macroscopic similarities, typical for muscular dystrophies involving proximal limb segments, demonstrating that the topological profiling mirrors established clinical

![](images/0fafd6b10073f53c04dcca068d9aff79e60b4a33f884c628b377c497a58d677d.jpg)  
Figure 4: Visual validation of the 3D topological feature space. The central UMAP scatter plot displays the gradient for skeletal complexity (GF\_Skel\_Nodes). The green frame illustrates the physical 3D topological skeleton of a DUX4 patient from the isolated high-complexity cluster, demonstrating a dense, highly branched network. The yellow frame shows a contrasting DYSF patient from the main continuous cluster, exhibiting lower topological complexity with linear fat propagation.

classifications.

To characterize the specific 3D architectural signatures driving this phenotypic separation, we mapped individual spatial biomarkers onto the manifold using unsupervised PARC clustering (Figure 5). Each of the five resulting clusters is defined by a unique morphological domain.

The isolated dark blue cluster demonstrates the distinct phenotypic divergence of the DUX4 cohort, which is structurally validated by the direct feature mapping (Figure 4). Even as other diseases exhibit distinct morphological overlap across a continuous spectrum, unsupervised clustering consistently isolates DUX4 into an independent topological island. This spatial separation is dominated by extreme values of topological complexity (GF\_Skel\_Nodes), reflecting a multi-focal network where the pathology invades the fascicles with dense, multidirectional branching. Conversely, the continuous LGMD cluster (including DYSF) exhibits a substantially lower node density, reflecting a more ordered, longitudinal infiltration pattern. Crucially, the validity of this mathematical isolation is corroborated by our pairwise post-hoc evaluations (Supplementary Table S2). Engineered features capturing network architecture generated exceptionally large efect sizes (frequently $r > 0 . 8 5 )$ strictly when diferentiating DUX4 from all other genetic variants, supporting this highly branched 3D disruption as a key diagnostic marker of the disease.

Conversely, the violet cluster is structurally distinguished by shifts in spatial vectoring, notably the center of mass (SF1\_CoM\_Z, Figure 5c). This indicates that early moderate fat replacement is localized significantly higher (proximal) or lower (distal) within the anatomical compartment, capturing a clear proximo-distal gradient of degeneration.

The pink cluster is defined by atypically low geometric orientation values (GF\_MainAxis\_Z, Figure 5d). The infiltration in this sub-cohort lacks the typical longitudinal cylindrical growth; instead, the lipodegeneration propagates horizontally across inter-muscular fascial planes, resulting in a geometrically distinct, shortened vector of disease development.

Furthermore, macroscopic fragmentation defines the turquoise cluster. Characterized by extreme values of fragment size (SF1\_BlobSize\_Max, Figure 5e), this group highlights patients where early fat and edema do not present as scattered spots. Instead, the pathology pools into massive, continuous, and merging macroscopic lakes prior to irreversible severe fat replacement.

Finally, the broad morphological continuum of the light blue cluster is defined by high geometric elongation (GF\_Elongation, Figure 5f). Here, the fat explicitly propagates in long, narrow streaks, directly invading and replacing the longitudinal architecture of the surviving muscle fibers.

As the pathology progresses, the unique geometric and topological signatures of each specific variant become morphologically dominant, driving the cohorts to diverge into distinct spatial regions. This proves that our continuous topological features capture fluid, highly specific disease states rather than static, spatially-agnostic volumetric categories.

## 5.3 Longitudinal Dynamics and Disease Progression

A critical advantage of extracting continuous, objective topological networks is their considerable potential in longitudinal tracking. Quantitative volumetry (such as Fat Fraction) provides only a static snapshot of the tissue ratio. In contrast, topological variables, such as the exponential growth rate of closed loops (GF\_Skel\_Cycles) or the narrowing of the interface between fat replacement stages (SF2\_To\_SF1\_Dist\_Min), allow clinicians to measure the spatial velocity and dynamics of the disease over time. Rather than utilizing these features exclusively for static cross-sectional diferentiation, monitoring how the pathological "web" structurally permeates the fascicles (as visualized in Figure 1c) can serve as a highly sensitive quantitative endpoint. This framework could evaluate the eficacy of emerging gene therapies by capturing active disease velocity and structural remodeling long before measurable overall muscle volume is lost.

## 5.4 Algorithmic Robustness and Interpretability

A notable limitation of contemporary machine learning frameworks (e.g., deep neural networks or ensemble methods like XGBoost) in rare disease diagnostics is their substantial data requirements and susceptibility to class imbalance. Predictive models frequently misclassify ultrarare variants by favoring majority classes due to skewed training distributions. In contrast, by explicitly extracting topological invariants, our framework embeds 3D geometry directly into the mathematical formulation. Consequently, these metrics represent absolute physical properties (e.g., exact spatial distances in millimeters, specific branching node counts), making them less susceptible to cohort size discrepancies and robust against class imbalances.

While advanced predictive systems rely on post-hoc explainability frameworks, such as SHAP values, to approximate the reasoning behind a "black-box" decision, our spatial vectoring methodology is intrinsically interpretable.

a)  
b)  
![](images/31da052390f126320fc7c896bad653c69a0e363eab250aa959ab91f248994c4e.jpg)

![](images/a9ac72a73f7b4660a76110499d3dcd8a059c7721eb068b1a6e743851409b157c.jpg)  
c)

![](images/185f37ac47b580639a70dfccb55bf15dd6760deac4de6684976231352ef82d3e.jpg)

d)  
![](images/51e8be91b1f8da7e52630cd7b876336e0cbb1e542f9bcab7e21c695c13e05573.jpg)

e)  
![](images/3e04646d687c24c80b00d0dd22a14a5b40a9fb52b4963166f9d38ab6b286e741.jpg)  
f

![](images/2f8de3dc8a7cfbe4b8f5fc74172c28e4e60185efa0316550aee41fec212bc83c.jpg)  
Figure 5: Multi-dimensional UMAP projection and feature gradient mapping of the patient cohort. (a) Unsupervised PARC clustering identifying five distinct topological groups based on whole-muscle 3D spatial features. (b-f) Gradient maps of individual spatial biomarkers highlighting the structural signatures driving the manifold topology: (b) GF\_Skel\_Nodes isolating the highly branched dark blue cluster; (c) SF1\_CoM\_Z distinguishing the proximodistal gradient in the violet cluster; (d) GF\_MainAxis\_Z identifying the non-longitudinal propagation in the pink cluster; (e) SF1\_BlobSize\_Max highlighting massive edema pooling in the turquoise cluster; and (f) GF\_Elongation defining the streak-like infiltrates of the light blue cluster. The full set of 63 feature gradient maps is available in Supplementary Figure S1.

Metrics such as the center of mass directly output the exact mathematical coordinates of pathological shifts without requiring secondary approximation layers, ofering a robust and transparent foundation for clinical decisionmaking.

## 5.5 Limitations of the Proposed Methodology

Despite the considerable potential of morphological graphradiomics, several limitations must be acknowledged. First, the extraction of 3D topological networks is inherently dependent on the quality of the initial tissue segmentation; inaccuracies in the severe or moderate fat replacement masks directly propagate to the skeletal graphs. Second, while the methodology models physical distances, highly anisotropic MRI acquisitions (where slice thickness significantly exceeds in-plane resolution) may distort the 3D skeletonization process, potentially introducing artifacts into the node and cycle counts. Moreover, while our topological invariants are theoretically hardware-agnostic, this study lacks external validation. Future studies must rigorously confirm the stability of these graph features across independent, multi-center cohorts involving diverse MRI vendors and magnetic field strengths (e.g., 1.5T vs. 3.0T).

Furthermore, an important clinical consideration is the heterogeneity of the patient cohort regarding disease duration and clinical severity. Comprehensive clinical progression data (e.g., exact time since symptom onset) were not universally available, meaning our statistical evaluations are not strictly adjusted for disease stage. Consequently, the observed structural variations encapsulate a mixture of fundamental genotypic signatures and varying degrees of temporal disease progression. Finally, the computational complexity of computing exact Euclidean distance transforms and k-D tree-based graph networks across large 3D volumes is inherently higher than that of standard volumetric assessments.

## 6 Conclusion

This study presents a novel, topology-driven approach to muscle MRI analysis in neuromuscular disorders. By explicitly modeling the physical architecture of lipodegeneration (including structural network complexity, fragmentation, and interface dynamics), this methodology addresses the "bag-of-pixels" limitation of classical radiomics. By capturing the spatial heterogeneity that global Fat Fraction (FF) dilutes, these topological descriptors serve as a highly complementary extension to standard clinical volumetry.

The rigorous statistical evaluation confirmed that these global 3D descriptors yield substantial efect sizes. Furthermore, post-hoc analyses demonstrated that these spatial markers act as highly discriminative biomarkers, capable of isolating specific genetic phenotypes based on their unique macroscopic infiltration patterns. Ultimately, this morphological approach bridges the interpretability gap associated with black-box deep learning models, providing clinicians with a transparent, robust, and geometrically accurate foundation for deep radiomic phenotyping and diferential diagnosis.

Future investigations must address the realities of clinical deployment. To overcome the bottleneck of manual 3D segmentation, the integration of semi-supervised learning could leverage vast amounts of unannotated, raw scans. Moreover, to create a holistic Clinical Decision Support System (CDSS), these topological features should undergo multi-modal data fusion, integrating imaging biomarkers with clinical metadata (e.g., age, sex, genetic profiles) and extending the analysis to upper-body musculature to prevent accuracy degradation as new phenotypes are introduced. Transitioning to real-world hospital environments also necessitates robust strategies for missing data imputation, as comprehensive multi-sequence MRI protocols are frequently incomplete. Finally, to accumulate suficient statistical power for rare diseases without compromising patient privacy, the establishment of federated learning networks across international consortiums will be paramount for the next generation of rare disease diagnostics.

## Code Availability

The automated feature extraction pipeline, including the 3D graph-based topology modules, was developed as an integral component of the MUSCAT (MUSCle fAt Topology) library. The codebase and the associated feature engineering algorithms are available from the corresponding author upon reasonable request.

## Acknowledgments

This research was supported by the CoMPaSS-NMD project (Computational Models for new Patients Stratification Strategies of Neuromuscular Disorders), funded by the HORIZON RIA program (HORIZON-HLTH-2022- TOOL-12-two-stage) under Grant Agreement No. GAP-101080874. We gratefully acknowledge their invaluable support and data provision.

## References

[1] Pierre G Carlier et al. Skeletal Muscle Quantitative Nuclear Magnetic Resonance Imaging and Spectroscopy as an Outcome Measure for Clinical Trials. Journal of Neuromuscular Diseases, 3(1):1–28, 2016.

[2] Augustin C Ogier, Marc-Adrien Hostin, Marc-Emmanuel Bellemare, and David Bendahan. Overview of MR Image Segmentation Strategies in Neuromuscular Disorders. Frontiers in Neurology, 12:625308, 2021.

[3] J Kemnitz, CF Baumgartner, F Eckstein, A Chaudhari, A Ruhdorfer, W Wirth, et al. Clinical evaluation of fully automated thigh muscle and adipose tissue segmentation using a U-Net deep learning archi-

tecture in context of osteoarthritic knee pain. Magnetic Resonance Materials in Physics, Biology and Medicine, 33:483–493, 2020.

[4] Jose Verdu-Diaz et al. Myo-Guide: A Machine Learning-Based Web Application for Neuromuscular Disease Diagnosis With MRI. Journal of Cachexia, Sarcopenia and Muscle, 16(3):e13815, 2025.

[5] Taiya Chen, Haoran Zhu, Yingyi Hu, Yang Huang, Wengan He, Yizhen Luo, Zeqi Wu, Diangang Fang, et al. Machine learning-based radiomics using MRI to diferentiate early-stage Duchenne and Becker muscular dystrophy in children. BMC Musculoskeletal Disorders, 26(1):287, 2025.

[6] Martijn Froeling, Aart J Nederveen, Arend Heerschap, and Gustav J Strijkers. Difusion tensor imaging of skeletal muscle: extraction of architecture and pathology. Journal of Magnetic Resonance Imaging, 36(6):1528–1546, 2012.

[7] Dheerendranath Battalapalli et al. Graph-Radiomic Learning (GrRAiL) Descriptor to Characterize Imaging Heterogeneity in Confounding Tumor Pathologies. Preprint available on Research Square, 2025.

[8] Maciej Tomczak and Ewa Tomczak. The need to report efect size estimates revisited. An overview of some recommended measures of efect size. Trends in Sport Sciences, 21(1):19–25, 2014.

[9] Harry Blum. A transformation for extracting new descriptors of shape. In Models for the Perception of Speech and Visual Form, pages 362–380. MIT Press, Cambridge, MA, 1967.

[10] Jinzheng Cai, Fuyong Xing, Abhinandan Batra, Fujun Liu, Glenn A Walter, Krista Vandenborne, and Lin Yang. Texture Analysis for Muscular Dystrophy Classification in MRI with Improved Class Activation Mapping. Pattern Recognition, 86:368–375, 2019.

[11] Eugenio Mercuri, Heinz Jungbluth, and Francesco Muntoni. Muscle imaging in clinical practice: diagnostic value of muscle magnetic resonance imaging in inherited neuromuscular disorders. Current Opinion in Neurology, 18(5):526–537, 2005.

[12] Ming-Kuei Hu. Visual pattern recognition by moment invariants. IRE Transactions on Information Theory, 8(2):179–187, 1962.

[13] Stéfan van der Walt, Johannes L Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D Warner, Neil Yager, Emmanuelle Gouillart, and Tony Yu. scikit-image: image processing in Python. PeerJ, 2:e453, 2014.

[14] William H Kruskal and W Allen Wallis. Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association, 47(260):583– 621, 1952.

[15] Olive Jean Dunn. Multiple comparisons using rank sums. Technometrics, 6(3):241–252, 1964.

[16] Leland McInnes, John Healy, and James Melville. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv preprint arXiv:1802.03426, 2018.