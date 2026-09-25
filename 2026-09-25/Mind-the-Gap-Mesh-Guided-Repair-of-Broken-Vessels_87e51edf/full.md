# Mind the Gap: Mesh-Guided Repair of Broken Vessels

Gniewosz Drwiega<sup>1⋆[0000−0002−1968−2238]</sup>, Wojciech Szymanski<sup>1,2[0009−0001−2043−9104]</sup>, and Marek Wodzinski<sup>1,2[0000−0002−8076−6246]</sup>

<sup>1</sup> Sano – Centre for Computational Personalised Medicine International Research Foundation, Krakow, Poland

<sup>2</sup> AGH University of Krakow, Krakow, Poland {g.drwiega,w.szymanski,m.wodzinski}@sanoscience.org

Abstract. Vessel segmentation is commonly optimized as voxel-wise classification, but small local errors can strongly disrupt vascular connectivity while having little efect on overlap scores. This is particularly problematic for downstream analyzes that rely on centerlines, branches, connected components, or graph structure. We propose a mesh-guided post-processing framework for repairing broken vessel segmentations produced by nnU-Net. For each predicted binary mask, a deformable template mesh is fitted to the mask surface in physical space and used as a case-specific geometric scafold. The fitted mesh is not voxelized as the final segmentation; instead, it guides conservative reconnection of disconnected components by proposing or validating thin bridge candidates under foreground-growth constraints. We evaluated this approach in three vascular anatomies using AortaSeg24 and SEGA for the aorta, TopCoW for the Circle of Willis, and PARSE for the pulmonary arteries. Performance is measured using Dice, connected-component Dice (ccDice), and the Betti-0 number. Across these datasets, repair substantially improved connectivity while preserving overlap: Dice remained nearly unchanged, whereas ccDice increased from 0.596 to 0.992 for aorta, from 0.722 to 0.835 for TopCoW, and from 0.028 to 0.862 for PARSE. The FOMAML meta-initialization further accelerated the fitting per-case, supporting practical mesh-based repair of the vascular topology. These results suggest that explicit mesh representations can provide a useful geometric prior for correcting topological failures in otherwise accurate voxel segmentations.

Keywords: Vessel segmentation · Mesh deformation · Topology · Tubular Shapes

## 1 Introduction

Medical image segmentation is often formulated as dense voxel classification [21, 13], yet many target anatomies are more naturally described as geometric objects embedded in physical space [10, 3]. Organ boundaries, vascular trees, and

⋆ Corresponding author

anatomical loops are not only sets of foreground voxels. They have surfaces, centerlines, branches, connected components, and topological relationships that determine whether a segmentation is useful for measurement, planning, or modeling [15, 2]. This perspective is especially relevant for vascular segmentation, where anatomical shape, branching structure, and connectivity are central to clinical and computational usability. Explicit shape representations, such as surfaces, meshes, or statistical shape models, can complement image-based learning by making anatomical geometry directly accessible to the model and to downstream analysis pipelines [25, 14, 3]. Large-scale resources such as MedShapeNet further facilitate the development and evaluation of methods based on explicit 3D anatomical shape representations [16].

The segmentation of vessels exposes this gap particularly clearly. Large vessels and vascular networks are thin, elongated, and strongly anisotropic. Their clinically meaningful properties depend on continuity and branching as much as on local overlap [15, 23, 6]. A small missing bridge can split a vessel into multiple components while changing only a tiny fraction of the volume (see Fig. 1). These errors are easy to underweight when optimizing or reporting voxel-wise overlap, but they matter for centerline extraction, branch labeling, diameter measurement, computational flow analysis, and any application that interprets the vascular structure as a connected object or graph [19]. This has motivated shape-aware approaches that incorporate tubular geometry directly into feature extraction for coronary and cerebral vessel analysis [24]. Recent vascular benchmarks such as AortaSeg24, SEGA, TopCoW, and PARSE further emphasize that vessel segmentation is not only a regional labeling task, but also a geometric and topological reconstruction problem [12, 26, 17].

Convolutional segmentation networks have made this problem much more tractable. U-Net established the encoder-decoder paradigm for biomedical segmentation, and nnU-Net demonstrated how robust configuration choices can turn this paradigm into a strong general-purpose baseline across many medical tasks [21, 13]. However, these methods still produce masks on a voxel grid and are commonly trained with losses that primarily reward local agreement with the reference annotation. For tubular structures, high Dice scores can therefore coexist with broken vessels, missing thin branches, isolated false-positive islands, or incorrect loops. Such errors may have limited impact on volumetric overlap while substantially changing the connectivity and topology of the predicted anatomy. This discrepancy motivates evaluation criteria and learning objectives that explicitly account for structure, rather than relying only on voxel-wise agreement.

The community has responded with topology-aware losses, skeleton-based objectives, and topology-specific evaluation measures. Persistent-homology losses penalize Betti-number errors [11], while clDice and centerline-Cross Entropy encourage tubular continuity using skeleton or centerline information [23, 1]. On the evaluation side, ccDice extends the Dice principle from voxel overlap to spatially corresponding connected components, making it directly relevant for fragmented vessel predictions [22]. Recent benchmarking further shows that overlap scores alone do not capture centerline continuity, component structure, and topological correctness [6]. These works provide important training and evaluation signals, but they do not directly address post-processing of an already available voxel mask, where residual connectivity errors should be corrected while preserving the original prediction as much as possible.

![](images/958c06ba5e67d35fa808ad8602dab094fb080547bbace1c61dc1790778ecfdb4.jpg)  
Fig. 1. Examples of broken-vessel failure modes across vascular anatomies. Top row: reference vessel structures. Bottom row: broken nnU-Net predictions, with missing local connections highlighted. Such small local errors can fragment otherwise accurate masks.

Mesh representations provide a natural mechanism for such a correction. A mesh is a compact and explicit geometric scafold: its vertices, edges, and faces encode adjacency, surface smoothness, and allowable paths in continuous physical space [4, 25, 14]. Prior work has used learned mesh deformation to reconstruct anatomical surfaces directly from volumetric images, including Voxel2Mesh and MeshDeformNet-style architectures [25, 14]. Related work has also fitted parametric vessel models directly to segmentations, linking voxel, centerline-radius, and mesh representations through diferentiable voxelization [7]. Together, these methods illustrate why meshes are attractive for medical shape analysis: they can reduce staircase artifacts, preserve explicit connectivity, and expose geometry in a form that is easier to regularize than an unconstrained voxel map. However, most mesh-based segmentation methods use the mesh as the final anatomical representation. In vascular segmentation, extensive branching, variable caliber, fine-scale geometry, and substantial inter-patient variability make direct mesh-based replacement challenging. Mesh-deformation models must generalize across both global anatomical variation and small peripheral branches. If the fitted mesh is voxelized as a direct mask replacement, it may overfill the vessel volume or introduce anatomically implausible shortcuts.

Our contributions are fourfold. First, we propose a case-wise deformable-mesh fitting pipeline that transforms an existing nnU-Net vessel mask [21, 13] into an explicit, case-specific geometric scafold. Second, we use the fitted mesh to guide conservative path-based reconnection of disconnected components rather than replacing the mask through full mesh voxelization, thereby limiting unnecessary foreground growth. Third, we introduce FOMAML meta-initialization to accelerate per-case mesh fitting by reducing the required number of optimization steps. Finally, we evaluate the proposed repair framework on four vascular datasets, AortaSeg24, SEGA, PARSE, and TopCoW, using overlap, component-level, and topological metrics that capture the geometric failure modes motivating this work.

## 2 Method

## 2.1 Pipeline Overview

The proposed method is a post-processing framework for repairing connectivity errors in vessel segmentations. Given a binary mask predicted by nnU-Net, we fit a deformable template mesh to the predicted vessel surface and use the fitted mesh as a case-specific geometric scafold. The repaired output remains a voxel mask: the mesh is not used as the final segmentation, but only as a guide for targeted reconnection of disconnected components, as shown in Fig. 2.

![](images/5e8f234b100b35c5cbca0ab582107b5744a5410cda14e7ab984b896e9f60a7b3.jpg)  
Fig. 2. Overview of the proposed mesh-guided nnU-Net repair pipeline. A CT volume is first processed by a CNN segmentation branch based on nnU-Net, producing segmented vessel sections and an initial nnU-Net mask. When the predicted mask is fragmented or topologically inconsistent, its surface is extracted and used as the target for mesh fitting. Then, a fixed template mesh is passed through a GCN deformation branch. The mesh is progressively refined over multiple deformation stages, where each stage predicts bounded vertex ofsets and forwards the updated mesh to the next refinement stage. The final overfitted mesh approximates the surface geometry of the nnU-Net prediction while preserving the template mesh connectivity. This fitted mesh is then used as a geometric prior for topology-aware mask repair, producing a repaired nnU-Net mask with improved connectivity.

## 2.2 Datasets and Input Masks

We evaluate the proposed repair framework on vascular segmentation datasets covering diferent anatomical structures and topological regimes. For aortic anatomy, we use SEGA, which focuses on aortic vessel-tree segmentation from CT angiography with emphasis on robustness, visual quality, and downstream meshing, and AortaSeg24, which provides CTA volumes annotated for clinically relevant aortic branches and zones [20, 12]. For cerebral vessels, we use TopCoW, a topologyaware Circle of Willis benchmark with voxel-level annotations of multiple CoW vessel components in CTA and MRA [26]. For pulmonary vessels, we use PARSE, which targets multi-level pulmonary artery segmentation in CTPA scans [17]. Because the repair task concerns global vessel connectivity rather than multi-class anatomical labeling, all dataset-specific subregions or branch labels were merged into a single foreground vessel label before training and evaluation.

## 2.3 Mesh Fitting

The mesh-fitting stage of the pipeline is illustrated in Fig. 2. For each case, we fit a deformable template mesh to the surface of the predicted binary mask. Its associated NIfTI geometry, including voxel spacing, origin, and direction, is preserved so that all fitting operations are performed in physical coordinates. We extract the target surface with marching cubes and convert the resulting vertices from voxel-index coordinates to physical coordinates. To make optimization comparable across cases, the target surface is normalized by the center and scale of its physical bounding box.

The template mesh is deformed toward the normalized target surface using a graph-based mesh decoder. Starting from an anatomy-specific template, the decoder applies a sequence of deformation stages, where each stage predicts bounded vertex ofsets conditioned on the current vertex positions and trainable per-vertex features [25, 14]. In our experiments, we use a four-stage decoder and optimize its parameters separately for each case. We use a cylindrical template for aortic structures, a spherical template for PARSE, and a torus-like template for the Circle of Willis in TopCoW. The exact template resolution and optimization parameters are reported in the experimental protocol.

The fitting loss combines Chamfer distance between sampled points on the deformed mesh and the target mask surface with geometric regularization terms, including edge-length, Laplacian, normal-consistency, and face-area regularization. These terms encourage the mesh to follow the predicted vessel surface while avoiding irregular or degenerate deformations. The fitting is performed independently for each predicted mask, rather than training a single mesh model to generalize across patients. For each case, the mesh with the best Chamfer score during optimization is mapped back to the original physical image space and used only as a local geometric prior for the subsequent repair step.

Unless stated otherwise, all mesh-fitting experiments used CUDA execution, random target-point sampling, 8192 target points for both the coarse and detail stages, 6000 fitting iterations, a detail-stage learning rate of 0.006, a Chamferloss weight of 1.0, decoder-stage maximum ofsets of 0.35, 0.20, 0.10, and 0.05, and a decoder-stage auxiliary-loss weight of 0.1. For each dataset, per-case fitting was initialized from a mesh-model checkpoint obtained by overfitting one representative case. Dataset-specific template choices and regularization weights are reported in Table 1.

Table 1. Dataset-specific mesh-fitting hyperparameters.
<table><tr><td>Dataset</td><td>Template</td><td>Edge</td><td>Lap.</td><td>Normal</td><td>Area</td></tr><tr><td>Aorta Unified</td><td>Cylinder, 10k</td><td>0.03125</td><td>1.25</td><td></td><td> $2 . 5 { \times } 1 0 ^ { - 4 } ~ / ~ 2 . 5 { \times } 1 0 ^ { - 5 }$ </td></tr><tr><td>TopCoW</td><td>Torus-like, 10k</td><td>0.000625</td><td>0.025</td><td></td><td> $2 . 5 { \times } 1 0 ^ { - 4 } ~ / ~ 2 . 5 { \times } 1 0 ^ { - 5 }$ </td></tr><tr><td>PARSE</td><td>Sphere, 20k</td><td>0.1</td><td>5.0</td><td></td><td> $1 . 0 { \times } 1 0 ^ { - 3 } \ / \ 5 . 0 { \times } 1 0 ^ { - 4 }$ </td></tr></table>

## 2.4 Mesh-Guided Connectivity Repair

After fitting, connectivity repair is performed conservatively. Rather than voxelizing the entire fitted mesh, the method identifies disconnected mask components and adds only thin bridge candidates that satisfy geometric validation criteria. To generate these candidate bridges, we consider two complementary strategies: one that follows the connectivity of the fitted mesh graph, and one that proposes local endpoint-to-endpoint bridges validated by the fitted mesh. In the mesh-graph variant, disconnected mask components are anchored to nearby vertices of the fitted mesh, and candidate bridges are obtained as shortest paths on the mesh graph. In the endpoint variant, the method instead proposes short local bridges between nearby component boundary points and uses the fitted mesh only for validation, requiring the bridge to be supported by nearby mesh vertices or face centroids. In both variants, a candidate bridge is rasterized as a thin tube and accepted only if it merges the target components while keeping the number of newly added voxels below a prescribed per-bridge foreground-growth limit.

Optional component filtering can remove distant false-positive mask components before mesh fitting and repair. After repair, a mesh-supported cleanup step can remove small components that remain poorly supported by the fitted mesh.

In our experiments, component-distance artifact filtering and mesh-supported cleanup were enabled for all datasets. Unless stated otherwise, repair used 26- connectivity, adaptive local bridge radii with a 6.0 mm local radius window, minimum component size of one voxel, post-repair mesh distance threshold of 2.0 mm, and minimum mesh-close fraction of 0.01. Dataset-specific repair variants and thresholds are reported in Table 2.

Table 2. Dataset-specific mask-repair hyperparameters.
<table><tr><td>Parameter</td><td>Aorta Unified</td><td>TopCoW</td><td>PARSE</td></tr><tr><td>Repair variant</td><td>Mesh-graph path</td><td>Mesh-graph path</td><td>Endpoint + mesh validation</td></tr><tr><td>Artifact keep/remove 57.0 / 57.5 (mm)</td><td></td><td>57.0 / 57.5</td><td>20.0 / 25.0</td></tr><tr><td>Bridge chor/support</td><td>an- Anchor 3.0 mm</td><td>Anchor 3.0 mm</td><td>Gap 12.0 mm; sup- port 3.0 mm; frac-</td></tr><tr><td>Tube base/min/max (mm)</td><td>radius 1.0 / 2.0 / 5.0</td><td>0.25 / 0.35 / 1.0</td><td>tion 0.50 0.4 / 0.7 / 1.8</td></tr><tr><td>Adaptive radius per- 80 / 1.0</td><td></td><td>80 / 0.7</td><td>70 / 0.8</td></tr><tr><td>centile/scale Max added</td><td>fore- 0.03</td><td>0.15</td><td>0.06</td></tr><tr><td>ground fraction Cleanup max moved voxels</td><td>re- 1000</td><td>1000</td><td>200</td></tr></table>

## 2.5 FOMAML Meta-Initialization

Meta-learned initialization has previously been used to accelerate case-specific medical shape reconstruction with implicit neural representations [5]. Motivated by this approach, we evaluate a first-order model-agnostic meta-learning (FO-MAML) initialization for accelerating per-case deformable-mesh fitting [9, 18]. Each training case is treated as a separate fitting task. For a given case, surface points sampled from the predicted mask are split into support and query sets. The mesh model is first adapted to the support points using several inner optimization steps, and the shared initialization is then updated using the query loss after adaptation. Following the first-order approximation, the meta-update does not backpropagate through the full inner optimization trajectory. At inference time, the learned initialization is used as a warm start for case-specific mesh fitting, with the goal of reducing optimization time and improving fitting stability.

## 2.6 Evaluation Metrics

We evaluate the repaired masks using both voxel-wise segmentation accuracy and topology-oriented measures. Dice similarity coeficient is used to quantify foreground overlap with the reference segmentation. To assess connected-component consistency, we report ccDice, a topology-aware Dice score that extends the Dice principle from individual voxels to connected components and accounts for their spatial correspondence [22]. We also report the Betti-0 number, corresponding to the number of connected components in the binary vessel mask, as a direct measure of fragmentation.

Although additional overlap, skeleton, and component-level metrics were computed during analysis, we focus the main results on Dice, ccDice, and Betti-0 because they summarize the central trade-of addressed by the proposed method: preserving voxel-wise segmentation quality while improving vessel connectivity.

## 2.7 Experimental Protocol

The baseline for all experiments is the raw nnU-Net binary prediction before mesh-guided repair. nnU-Net models were trained using the 3D full-resolution configuration with the default nnUNetTrainer. Training was performed on the Helios PLGrid GPU cluster using one GPU per job, 16 CPU cores, 120 GB RAM, and a 48 h wall-time limit. As summarized in Fig. 2, the input to the repair stage is a hard binary vessel mask predicted by the nnU-Net 3D fullresolution configuration. The raw nnU-Net prediction is used as the baseline, and the proposed method is applied as a post-processing step to the same mask. We use five-fold outer evaluation protocol, ensuring that each repaired case was segmented by a model that did not see it during training. The repair stage does not use image intensities or nnU-Net probability maps.

Mesh fitting and connectivity repair were then run as a post-processing step on the predicted binary masks. These experiments were performed on a local workstation with an Intel Core Ultra 9 275HX CPU, 32 GB RAM, and an NVIDIA GeForce RTX 5090 GPU with 24 GB memory. The mesh-fitting hyperparameters were selected by grid-search experiments on the Helios cluster and then fixed for the reported out-of-fold evaluation. All post-processing experiments were implemented in Python using PyTorch for mesh deformation, SimpleITK for NIfTI image I/O and geometry handling, scikit-image for surface extraction, SciPy for connected-component and nearest-neighbor operations, and nnU-Net v2 for baseline segmentation. The source code is available in the accompanying repository [8].

## 3 Results

## 3.1 Visual Assessment of Vessel Reconnection

Qualitative examples are shown in Fig. 3. Broken-connectivity errors were observed across all evaluated anatomies, but with diferent visual patterns. In the aortic cases, errors typically appeared as small missing bridges along otherwise well-segmented large vessels or near branch regions (Fig. 3A). The TopCoW cases showed the most frequent local discontinuities in the posterior communicating artery region of the Circle of Willis (Fig. 3B). In PARSE, the pulmonary arterial tree exhibited severe fragmentation, especially in the thinner peripheral branches (Fig. 3C).

Across these examples, mesh-guided repair added localized connector regions rather than replacing the full mask by a voxelized mesh. The repaired masks therefore preserved the original nnU-Net segmentation in most regions while reconnecting components that were geometrically supported by the fitted mesh (Fig. 3D-L).

Aorta

A  
![](images/0ffcf92252f006101942d87cccf3fb31d4158356171fab72fc133caf081d4b12.jpg)

B  
![](images/9492995ab714c26e110417b355a15238d69307853ce8c7f2978e74e67a96ad2f.jpg)

Pulmonary Artery

C  
![](images/f153a49056da7c5d494b0a9a90e6c21149d65144576040addc51844e446ea80d.jpg)

![](images/c0f45864cb975f956f4d43afc6b6c26799db0d62bc024c7c6d0c2c804cc7aab2.jpg)  
E

![](images/4a5bb60385db632299470c37d708720faac84e5c3e874d73dad99c1eb12069de.jpg)

![](images/fc7ebe219716982d34a9379ad364185a40fd19da885ff4043060c49f9cda966d.jpg)

![](images/242391b58abd281be34c412632c0c5aa04b6b41910530cf7a3e84421acebbaf6.jpg)

H Repaired nnU-Net mask  
![](images/ef986d58df42c2ebf8a47fe5e59efc1f1b3d50bc0ceb281f31b563fc2e86f7bf.jpg)

一  
![](images/fb10724bf0c68e2fd50546f0bf4427ff68728578935b67c7cd0cbf735bfccff8.jpg)

J  
![](images/3974682a25016d5fc0e7075458ccf161fc5b7a458f11f535fcec69cf313198d9.jpg)  
K

![](images/9a3325c1076272176e0d0a4b0b35f04f926aca7617fba8f687f660cead2cce96.jpg)

L  
![](images/212668341ed9ed50e1ae233282d4d113640221a48a189309d3ea55d0c69b7c6f.jpg)  
Fig. 3. Qualitative examples of mesh-guided repair across vascular anatomies. The method reconnects fragmented vessel components using local mesh-guided bridges while preserving the original mask away from the repaired regions. Blue arrows indicate broken vessel regions repaired by the proposed method.

## 3.2 Efect on Overlap and Connectivity

The main quantitative results are summarized in Table 3. For the aortic cohort, including SEGA and AortaSeg cases, Dice remained essentially unchanged after repair, changing from $0 . 9 3 4 { \pm } 0 . 0 4 3$ for the raw mask to $0 . 9 3 4 { \pm } 0 . 0 4 3$ after repair. In contrast, ccDice increased from $0 . 5 9 6 { \pm } 0 . 2 7 5$ to $0 . 9 9 2 { \scriptstyle \pm 0 . 0 5 7 }$ , and the Betti-0 number decreased from $3 . 3 5 \pm 2 . 6 1$ to $1 . 0 1 \pm 0 . 1 2$

A similar pattern was observed for TopCoW. Dice changed only slightly, from 0.870±0.055 to 0.867±0.054, while ccDice improved from $0 . 7 2 2 { \scriptstyle \pm 0 . 2 4 7 }$ to 0.835± 0.195. The Betti-0 number decreased from $2 . 5 8 \pm 1 . 1 4$ to $1 . 0 2 \pm 0 . 1 3$ , indicating that most repaired predictions became nearly single-component structures.

The strongest connectivity efect was observed for PARSE, where raw pulmonaryartery masks were highly fragmented. Dice remained stable, changing from 0.877± 0.036 to $0 . 8 7 6 { \pm } 0 . 0 3 6$ , while ccDice increased from $0 . 0 2 8 { \pm } 0 . 0 1 2$ to $0 . 8 6 2 { \scriptstyle \pm 0 . 2 1 1 }$ The Betti-0 number decreased from $8 7 . 4 6 \pm 4 5 . 9 2$ to $1 . 5 6 \pm 1 . 0 6$ . The falsebranch fraction remained nearly unchanged for the aortic cohort $( 0 . 0 5 6 \pm 0 . 0 0 9$ to $0 . 0 5 5 \pm 0 . 0 0 9 )$ and PARSE (0.130 ± 0.020 to $0 . 1 3 2 \pm 0 . 0 2 0 )$ , but increased for TopCoW from $0 . 0 4 0 \pm 0 . 0 0 6$ to 0.063 ± 0.007, indicating a dataset-specific tradeof between restoring connectivity and avoiding spurious branches. These results support the central motivation of the method: large connectivity improvements can be achieved with negligible change in voxel-wise overlap.

Table 3. Segmentation and connectivity metrics before and after repair. Values are reported as mean ± 95% confidence interval. Best values within each dataset are shown in bold. FB denotes false-branch fraction.
<table><tr><td>Dataset</td><td>Mask</td><td>Dice ↑</td><td>ccDice ↑</td><td> $\overline { { \beta _ { 0 } \downarrow } }$ </td><td>FB↓</td></tr><tr><td rowspan="3">Aorta (n = 145)</td><td>Raw</td><td>0.934 ± 0.007</td><td> $\overline { { 0 . 5 9 6 \pm 0 . 0 4 5 } }$ </td><td> $\overline { { 3 . 3 5 \pm 0 . 4 3 } }$ </td><td>0.056 ± 0.009</td></tr><tr><td>Filtered</td><td>0.935 ± 0.007</td><td> $0 . 6 9 5 \pm 0 . 0 4 3$ </td><td> $2 . 4 7 \pm 0 . 3 0$ </td><td> $\mathbf { 0 . 0 4 4 \pm 0 . 0 0 8 }$ </td></tr><tr><td>Repaired</td><td> $0 . 9 3 4 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 9 9 2 \pm 0 . 0 0 9 }$ </td><td> ${ \bf 1 . 0 1 \pm 0 . 0 2 }$ </td><td> $0 . 0 5 5 \pm 0 . 0 0 9$ </td></tr><tr><td rowspan="3">TopCoW (n = 125)</td><td>Raw</td><td> $\mathbf { \overline { { 0 . 8 7 0 \pm 0 . 0 1 0 } } }$ </td><td> $\overline { { 0 . 7 2 2 \pm 0 . 0 4 3 } }$ </td><td> $\overline { { 2 . 5 8 \pm 0 . 2 0 } }$ </td><td> $\mathbf { \overline { { 0 . 0 4 0 \pm 0 . 0 0 6 } } }$ </td></tr><tr><td>Filtered</td><td> $\mathbf { 0 . 8 7 0 \pm 0 . 0 1 0 }$ </td><td> $0 . 7 2 3 \pm 0 . 0 4 3$ </td><td> $2 . 5 8 \pm 0 . 2 0$ </td><td> $\mathbf { 0 . 0 4 0 \pm 0 . 0 0 6 }$ </td></tr><tr><td>Repaired</td><td> $0 . 8 6 7 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 8 3 5 \pm 0 . 0 3 4 }$ </td><td> ${ \bf 1 . 0 2 \pm 0 . 0 2 }$ </td><td> $0 . 0 6 3 \pm 0 . 0 0 7$ </td></tr><tr><td rowspan="3">PARSE (n = 100)</td><td>Raw</td><td>0.877 ± 0.007</td><td> $\overline { { 0 . 0 2 8 \pm 0 . 0 0 2 } }$ </td><td> $\overline { { 8 7 . 4 6 \pm 9 . 0 0 } }$ </td><td> $\overline { { 0 . 1 3 0 \pm 0 . 0 2 0 } }$ </td></tr><tr><td>Filtered</td><td> $\mathbf { 0 . 8 7 7 \pm 0 . 0 0 7 }$ </td><td> $0 . 0 2 8 \pm 0 . 0 0 2$ </td><td> $8 6 . 5 6 \pm 8 . 9 7$ </td><td> $\mathbf { 0 . 1 2 9 \pm 0 . 0 2 0 }$ </td></tr><tr><td>Repaired</td><td> $0 . 8 7 6 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 8 6 2 \pm 0 . 0 4 1 }$ </td><td> ${ \bf 1 . 5 6 \pm 0 . 2 1 }$ </td><td> $0 . 1 3 2 \pm 0 . 0 2 0$ </td></tr></table>

## 3.3 Efect of FOMAML Meta-Initialization

One might expect fitting a mesh independently to each predicted mask to introduce substantial computational overhead during inference. To reduce this overhead, we use FOMAML to learn an initialization that can adapt to a new case in fewer optimization steps. Fig. 5 compares Chamfer-loss convergence between standard and FOMAML initialization on 25 held-out TopCoW cases. FOMAML achieved a lower Chamfer loss at every evaluated checkpoint and outperformed the standard initialization in all 25 cases, demonstrating faster and more consistent per-case mesh fitting.

![](images/ebb084b6e2d34a285b20b8c5bff128b5627de6ca35b5096f38433de83ff77212.jpg)  
Fig. 4. Quantitative summary of repair efects and mesh-fitting convergence. Meshguided repair improves connectivity-oriented metrics while preserving Dice, and metainitialization accelerates per-case mesh fitting.

![](images/ea975013960bf78a5db7737704d4a4cbbd379939693173fab7a2b9fe29d18b8a.jpg)  
Fig. 5. Efect of FOMAML meta-initialization on per-case mesh-fitting convergence for 25 held-out $\mathrm { T o p C o W }$ cases. Lines show mean Chamfer loss across cases, and shaded regions indicate ± SEM. The black horizontal line indicates a matched Chamfer-distance level, allowing comparison of the number of optimization steps required by FOMAML and standard initialization to reach the same reconstruction error.

The largest benefit was observed in the early optimization phase, indicating that FOMAML provides a better starting point for case-wise fitting. After 100 fitting iterations, mean Chamfer decreased from $7 . 7 9 \times 1 0 ^ { - 4 }$ with standard initialization to $4 . 9 9 \times 1 0 ^ { - 4 }$ with FOMAML. After 500 iterations, the corresponding values were $4 . 4 3 \times 1 0 ^ { - 4 }$ and $3 . 3 9 \times 1 0 ^ { - 4 }$ . This suggests that FOMAML can reduce the number of iterations needed to reach a given fitting quality, providing a useful warm start for case-specific mesh optimization.

Overall, the results show that the proposed repair strategy mainly afects connectivity rather than regional overlap. This behavior is desirable for brokenvessel repair: the method improves component structure and connectedness while preserving the strong voxel-wise segmentation performance of nnU-Net.

## 4 Discussion

## 4.1 Mesh-Guided Topology Repair

The results indicate that mesh representations can be useful as case-specific geometric priors for repairing vessel topology. Instead of replacing a voxel segmentation with a fully voxelized mesh, the proposed method uses the fitted mesh only to guide local reconnection. This distinction is important: the repaired output remains close to the original nnU-Net prediction, while the mesh provides additional geometric structure for identifying plausible paths between fragmented components. Because the mesh is fitted independently for each case, the method does not require a single mesh model to generalize directly across all patients and vascular anatomies. This per-case optimization is especially useful for vessels, where branching patterns, caliber, and topology vary substantially between cases. Across datasets, the main efect of the method was therefore observed in connectivity-sensitive metrics, whereas Dice changed only marginally.

The behavior was most consistent for aortic anatomy, where the target structure is comparatively large and the main connectivity errors are often localized. TopCoW and PARSE introduce more challenging topological settings. In the Circle of Willis, small vessels, variable anatomy, and clinically important communicating arteries mean that a local discontinuity can strongly afect graph connectivity. Moreover, some TopCoW ground-truth masks represent anatomically incomplete Circle of Willis variants; for example, absence of both posterior communicating arteries may leave the anterior and posterior circulations disconnected in the binary vascular mask [26]. PARSE is also demanding because pulmonary arteries form a dense, highly branching tree with many peripheral vessels. In these cases, mesh fitting and bridge selection must handle substantially more complex geometry than in the aorta.

The FOMAML meta-initialization further improves the practicality of this per-case strategy. In the aortic setting, nnU-Net inference required 44.17 s per case, while FOMAML-initialized mesh fitting with approximately 300 optimization steps followed by post-repair required about 30 s per case. Thus, the additional geometric repair stage remains comparable to the original segmentation inference time while providing a targeted improvement in connectivity.

## 4.2 Limitations

Several limitations remain. First, the fitted mesh can support an incorrect bridge if the mask contains misleading false-positive components or if the mesh deformation follows an anatomically implausible shortcut. Second, although optional component filtering and mesh-supported cleanup reduce the influence of distant false positives, they do not fully solve cases where false positives are close to true vessels. Third, the current repair stage uses hard binary masks and does not exploit image intensities or nnU-Net probability maps, which could help distinguish plausible missing vessels from unsupported bridges. Finally, the method relies on dataset-specific templates and hyperparameters, which may need adjustment for new vascular territories.

## 4.3 Future Work

Future work will focus on making the repair step more adaptive. Learned bridge scoring could replace hand-designed acceptance rules, while anatomical constraints could reduce implausible shortcuts in highly variable structures such as the Circle of Willis and pulmonary vasculature. Incorporating probability maps or image-based evidence may also improve bridge validation, especially in ambiguous peripheral regions. We also plan to conduct computational hemodynamic experiments to quantify how small topological corrections afect simulated blood flow and other downstream functional measurements. More broadly, combining per-case mesh fitting with learned topology-aware repair may provide a stronger framework for correcting vascular segmentation without retraining the original segmentation model.

## 5 Conclusion

We presented a mesh-guided post-processing method for repairing broken vessel segmentations. By fitting a deformable mesh to each predicted mask and using it as a local geometric scafold, the method improves vascular connectivity while preserving the original voxel-wise segmentation. With FOMAML initialization, the per-case repair remains practical in runtime while avoiding the need for a globally generalizing mesh prediction model. The results support the use of mesh representations as practical geometric priors for topology repair in vessel segmentation.

Acknowledgments. This work was supported by the National Science Centre, Poland, under Grant “MultiGeoMed” No. 2024/55/D/ST6/02081. We gratefully acknowledge the Polish high-performance computing infrastructure PLGrid, HPC Center: ACK Cyfronet AGH, for providing computational resources and support within computational grant No. PLG/2025/018770.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Acebes, C., Moustafa, A.H., Camara, O., Galdran, A.: The centerline-cross entropy loss for vessel-like structure segmentation: Better topology consistency without sacrificing accuracy. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. Lecture Notes in Computer Science, vol. 15008, pp. 710–720. Springer Nature Switzerland (2024). https://doi.org/10.1007/978-3-031- 72111-3\_67

2. Antiga, L., Piccinelli, M., Botti, L., Ene-Iordache, B., Remuzzi, A., Steinman, D.A.: An image-based modeling framework for patient-specific computational hemodynamics. Medical & Biological Engineering & Computing 46(11), 1097–1112 (2008). https://doi.org/10.1007/s11517-008-0420-1

3. Bohlender, S., Oksuz, I., Mukhopadhyay, A.: A survey on shape-constraint deep learning for medical image segmentation. arXiv preprint arXiv:2101.07721 (2021). https://doi.org/10.48550/arXiv.2101.07721

4. Botsch, M., Kobbelt, L., Pauly, M., Alliez, P., Lévy, B.: Polygon Mesh Processing. CRC Press (2010)

5. De Paolis, G.R., Lenis, D., Novotny, J., et al.: Fast medical shape reconstruction via meta-learned implicit neural representations. In: Shape in Medical Imaging. LNCS, vol. 15275, pp. 189–204. Springer (2025). https://doi.org/10.1007/978-3- 031-75291-9\_15

6. Decroocq, M., Poon, C., Schlachter, M., Skibbe, H.: Benchmarking evaluation metrics for tubular structure segmentation in biomedical images. In: Shape in Medical Imaging. Lecture Notes in Computer Science, vol. 16171, pp. 87–102. Springer Nature (2025). https://doi.org/10.1007/978-3-032-06774-6\_7

7. Dima, A.F., Shit, S., Qiu, H., et al.: Parametric shape models for vessels learned from segmentations via diferentiable voxelization. In: Shape in Medical Imaging. LNCS, vol. 16171, pp. 247–261. Springer (2026). https://doi.org/10.1007/978-3- 032-06774-6\_19

8. Drwiega, G., Szymanski, W., Wodzinski, M.: Source code for mesh-guided repair of broken vessel segmentations (2026), the repository will be made publicly accessible upon acceptance

9. Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: Proceedings of the 34th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 70, pp. 1126–1135 (2017)

10. Heimann, T., Meinzer, H.P.: Statistical shape models for 3d medical image segmentation: A review. Medical Image Analysis 13(4), 543–563 (2009). https://doi.org/10.1016/j.media.2009.05.004

11. Hu, X., Li, F., Samaras, D., Chen, C.: Topology-preserving deep image segmentation. In: Advances in Neural Information Processing Systems. vol. 32, pp. 5658– 5669 (2019)

12. Imran, M., Krebs, J.R., Sivaraman, V.B., Zhang, T., Kumar, A., Ueland, W.R., Fassler, M.J., Huang, J., Sun, X., Wang, L., et al.: Multi-class segmentation of aortic branches and zones in computed tomography angiography: The AortaSeg24 challenge. arXiv preprint arXiv:2502.05330 (2025). https://doi.org/10.48550/arXiv.2502.05330

13. Isensee, F., Jaeger, P.F., Kohl, S.A.A., Petersen, J., Maier-Hein, K.H.: nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. Nature Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z

14. Kong, F., Wilson, N., Shadden, S.C.: A deep-learning approach for direct whole-heart mesh reconstruction. Medical Image Analysis 74, 102222 (2021). https://doi.org/10.1016/j.media.2021.102222

15. Lesage, D., Angelini, E.D., Bloch, I., Funka-Lea, G.: A review of 3d vessel lumen segmentation techniques: Models, features and extraction schemes. Medical Image Analysis 13(6), 819–845 (2009). https://doi.org/10.1016/j.media.2009.07.011

16. Li, J., Pepe, A., Gsaxner, C., Luijten, G., Jin, Y., Ambigapathy, N., Nasca, E., Solak, N., Melito, G.M., Memon, A.R., et al.: Medshapenet–a large-scale dataset of 3d medical shapes for computer vision. arXiv preprint arXiv:2308.16139 (2023)

17. Luo, G., Wang, K., Liu, J., Li, S., Liang, X., Li, X., Gan, S., Wang, W., Dong, S., Wang, W., et al.: Eficient automatic segmentation for multi-level pulmonary arteries: The PARSE challenge. arXiv preprint arXiv:2304.03708 (2023). https://doi.org/10.48550/arXiv.2304.03708

18. Nichol, A., Achiam, J., Schulman, J.: On first-order metalearning algorithms. arXiv preprint arXiv:1803.02999 (2018). https://doi.org/10.48550/arXiv.1803.02999

19. Pepe, A., Li, J., Rolf-Pissarczyk, M., Gsaxner, C., Chen, X., Holzapfel, G.A., Egger, J.: Detection, segmentation, simulation and visualization of aortic dissections: A review. Medical Image Analysis 65, 101773 (2020). https://doi.org/10.1016/j.media.2020.101773

20. Pepe, A., Melito, G.M., Egger, J. (eds.): Segmentation of the Aorta. Towards the Automatic Segmentation, Modeling, and Meshing of the Aortic Vessel Tree from

Multicenter Acquisition, Lecture Notes in Computer Science, vol. 14539. Springer Cham (2024). https://doi.org/10.1007/978-3-031-53241-2

21. Ronneberger, O., Fischer, P., Brox, T.: U-Net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. Lecture Notes in Computer Science, vol. 9351, pp. 234–241. Springer (2015). https://doi.org/10.1007/978-3-319-24574-4\_28

22. Rougé, P., Merveille, O., Passat, N.: ccDice: A topology-aware dice score based on connected components. In: Topology- and Graph-Informed Imaging Informatics. Lecture Notes in Computer Science, vol. 15239, pp. 11–21. Springer (2025). https://doi.org/10.1007/978-3-031-73967-5\_2

23. Shit, S., Paetzold, J.C., Sekuboyina, A., Ezhov, I., Unger, A., Zhylka, A., Pluim, J.P.W., Bauer, U., Menze, B.H.: clDice: A novel topology-preserving loss function for tubular structure segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16560–16569 (2021)

24. Wang, Z., Ge, X., Chen, X., et al.: Deep combined computing of vascular images with tubular shape-guided convolution. In: Shape in Medical Imaging. LNCS, vol. 15275, pp. 48–58. Springer (2025). https://doi.org/10.1007/978-3-031-75291-9\_4

25. Wickramasinghe, U., Remelli, E., Knott, G., Fua, P.: Voxel2Mesh: 3d mesh model generation from volumetric data. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. pp. 299–308. Lecture Notes in Computer Science, Springer International Publishing (2020). https://doi.org/10.1007/978-3- 030-59719-1\_30

26. Yang, K., Musio, F., Ma, Y., Juchler, N., Paetzold, J.C., Al-Maskari, R., Höher, L., Li, H.B., Hamamci, I.E., Sekuboyina, A., et al.: Benchmarking the CoW with the TopCoW challenge: Topology-aware anatomical segmentation of the circle of willis for CTA and MRA. arXiv preprint arXiv:2312.17670 (2025). https://doi.org/10.48550/arXiv.2312.17670