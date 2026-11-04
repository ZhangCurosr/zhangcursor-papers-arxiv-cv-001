# EndoS<sup>3</sup>:Endoscopic Semantic Splatting SLAM

Mona Sheikh Zeinoddin<sup>1\*†</sup>, Benjamin Wright<sup>1†</sup>, Zafer Tandogdu<sup>2</sup>, Greg Shaw<sup>2</sup>, Evangelos Mazomenos<sup>1</sup>, Danail Stoyanov<sup>1</sup>

<sup>1</sup>Hawkes Institute, University College London, UK. <sup>2</sup>Dept of Urology, University College London Hospitals, UK.

\*Corresponding author(s). E-mail(s): mona.zeinoddin.22@ucl.ac.uk, e.mazomenos@ucl.ac.uk; <sup>†</sup>These authors contributed equally to this work.

## Abstract

Purpose: Despite advances in 3D surgical scene understanding, semantic simultaneous localisation and mapping (SLAM) remains under-explored for two key reasons. First, it requires solving three challenging tasks in parallel: camera pose tracking, high-fidelity deformable reconstruction, and semantic mapping, making progress particularly demanding. Second, the scarcity of datasets providing the necessary ground truth to evaluate all three tasks comprehensively under conditions that closely reflect real surgical procedures environments.

Method: We propose EndoS<sup>3</sup>, a Gaussian Splatting-based semantic SLAM framework that uses instance masks to guide all three tasks. An instancelevel motion decomposition module separates static and dynamic instances by evaluating residual flow after compensating for camera motion, accounting for flow noise and comparing motion across neighbouring instances. The camera is tracked against the static Gaussians, while an instance-aware warp field gives each dynamic instance its own control nodes and deformation network. A semantic node allocation module redistributes a fixed node budget across instances according to their size and non-rigidity. We also introduce CRCD++, an exvivo Cholecystectomy dataset with RGB-D frames, camera pose and instance annotations for full semantic SLAM evaluation.

Results: EndoS<sup>3</sup> surpasses all state-of-the-art general-purpose and surgical semantic SLAM methods in tracking, reconstruction and semantic mapping on the CRCD++ and SemSup datasets.

Conclusion: Semantic-aware motion and deformation modelling is vital to surgical semantic SLAM. EndoS<sup>3</sup> and CRCD++ ofer a method and benchmark for improved geometric and semantic understanding in endoscopy.

Keywords: Endoscopy, Semantic SLAM, Tracking, Reconstruction

## 1 Introduction

Spatial understanding of the endoscopic field supports surgical navigation and context-aware decision-making [1, 2]. Reaching it requires complementary geometric and semantic information: geometry describes the spatial relationships between the camera, anatomy, and instruments, while semantics identifies them and analyses their interactions. Surgical semantic simultaneous localisation and mapping (SLAM) achieves this understanding through jointly performing camera pose tracking, scene reconstruction, and semantic mapping, though it sufers from several key challenges.

First, image motion is inherently ambiguous: pixel displacements may arise from camera motion, tissue deformation, or independently moving instruments [3]. Misattributing scene motion to the camera corrupts pose estimates, while the opposite degrades the reconstruction. Previous approaches use geometric cues to decompose the sources of motion and limit the influence of tissue deformation on camera pose estimation [3], yet the role of instance-level semantics in motion decomposition remains under-explored, even though it can provide valuable information by associating pixels with entities that exhibit distinct motion characteristics.

Second, surgical scenes exhibit heterogeneous, time-varying deformations: Rigid instruments move quickly and independently of the tissue they manipulate, while organs deform slowly and largely coherently. Deformation models must capture these diferences and adapt to the content currently in view. Current approaches, however, model the motion of the entire scene with a single warp field, driven by a shared set of control nodes [4–6], which can blend instrument and tissue motion near their contact points. Third, a fixed set of control nodes [6] also limit adaptation to instruments entering or leaving the view and regions of greater tissue deformation. Existing methods primarily use semantics for map labelling and data association [7–10], while their use in motion decomposition and deformation modelling remains unexplored. Moreover, progress is further limited by the scarcity of evaluation data. Existing datasets lack joint ground truth for camera poses, depth, and segmentation [1, 3]. These annotations are needed to evaluate tracking, reconstruction, and semantic mapping jointly and determine how improvements in one task afect the others.

We propose EndoS<sup>3</sup>, a semantic SLAM framework that uses instance semantics to guide motion decomposition for camera tracking and deformable semantic mapping. We also present CRCD++, a cholecystectomy dataset with camera poses, depth, and instance annotations for a complete evaluation of semantic SLAM methods. Our code and data are at: https://bwright000.github.io/EndoS3/. Our main contributions are:

• Instance-level motion decomposition, which separates static and dynamic instances by investigating camera-compensated flow residuals, their signal-to-noise ratios, and motion diferences across instance boundaries.

• Instance-aware warp field, which assigns every instance its own control nodes and deformation network and restricts motion propagation to within that instance.

• Semantic node allocation, which redistributes a fixed node budget across instances at every frame according to their size and non-rigidity.

• CRCD++, an ex-vivo Cholecystectomy dataset with all of the necessary ground truth needed to perform an in-depth evaluation of surgical semantic SLAM methods.

## 2 Related Work

General-purpose semantic SLAM: Dense semantic SLAM builds the map using implicit fields or explicit primitives. SNI-SLAM [8] fuses appearance, geometry and semantic features in a neural implicit map through cross-attention; SGS-SLAM [9] jointly optimises the appearance, geometric and semantic features of a Gaussian map and uses semantics to select keyframes; and SemGauss-SLAM [10] embeds semantic features in the Gaussians and enforces consistency through semantic-informed bundle adjustment. All three regularise a static map and are evaluated on indoor scenes [11]. None model deformation or dynamics, so in scenes that feature either, they cannot disentangle scene from camera motion, typically using the latter to explain the former. Surgical semantic SLAM: Among sparse methods, Semantic-SuPer [1] builds a surfel map with a sparse embedded deformation graph and uses semantic masks within tissue tracking to pull surfels into their semantic region, but assumes a stationary camera. PERSEUS [2] combines learned segmentation and monocular depth with a learning-based SLAM backend to label a point cloud. It neither produces high-fidelity renderings nor models deformation, and is evaluated only on static sequences. Among dense methods, DDS-SLAM [7] models deformation as a hash-grid canonical scene with a time-dependent neural field and uses a semantic distance loss. Deformation is supervised only through rendering, and tracking does not separate static from moving regions, leaving camera and tissue motion entangled. It is evaluated on deforming scenes [1, 3] and outputs no semantic map; as they are used in its optimisation solely.

None of these methods use semantics to disentangle scene motion from camera motion. EndoS<sup>3</sup> uses a Gaussian Splatting framework where instance semantics guides motion decomposition: the camera is tracked against the static instances only, separating camera from scene motion, while an instance-aware warp field models moving organs and instruments individually and allows each to deform independently.

Surgical benchmarks: Semantic SLAM evaluation requires RGB images, depth, camera poses, and segmentation annotations. Existing datasets [12–14] ofer sparse segmentations or lack poses. StereoMIS [3] and D4D [15] provide poses but segment only tools; D4D also provides measured depth. eCRCD [16] annotates two tissue classes but lacks tool masks and propagates labels from one frame per clip. CRCD++ alone meets all four requirements. Unlike the general abdominal scenes in StereoMIS and D4D, CRCD++ provides clinically meaningful organ annotations specific to Cholecystectomy (Table 1).

Table 1 Overview of Semantic datasets. Sem.: semantic classes (<sup>s</sup>sparse; <sup>t</sup>tools only; <sup>nt</sup>no tools); Seq. len.: sequence length (frames); GT/Pseudo-GT: measured/stereo-estimated depth. Green: met; red: not met.
<table><tr><td></td><td>Cholec- Seg8k [12]</td><td>Endo- Vis18 [13]</td><td>Endo- scapes [14]</td><td>Semantic- SuPer [1]</td><td>Stereo- MIS [3]</td><td>D4D [15]</td><td>eCRCD [16]</td><td>CRCD++</td></tr><tr><td>Poses</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Depth</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>GT</td><td>No</td><td>Pseudo-GT</td></tr><tr><td>Sem.</td><td>13</td><td>12⁸</td><td>6⁸</td><td>3</td><td>1t</td><td>1t</td><td>2nt</td><td>4</td></tr><tr><td>Seq. len.</td><td>80</td><td>Sparse</td><td>Sparse</td><td>150</td><td>1.4–9.6k</td><td>~3k</td><td>120</td><td>0.2–2.1k</td></tr><tr><td>Frames</td><td>8,080</td><td>2,384</td><td>493</td><td>600</td><td>~58.7k</td><td>300k+</td><td>一</td><td>14,688</td></tr></table>

![](images/81f23f0a4fe95832d053f3cb2b4acda111759b00aecf5e7f45add0f436149115.jpg)  
Fig. 1 Overview of our framework - $\mathrm { E n d o S ^ { 3 } }$ uses semantics in both tracking and mapping.

## 3 Methods

As shown in Fig. 1, given an RGB-D sequence and per-frame instance segmentation masks $\{ \mathbf { C } _ { t } , \mathbf { D } _ { t } \mathbf { S } _ { t } \}$ , EndoS<sup>3</sup> adopts a 3D semantic Gaussian Splatting scene representation [10]. Our Instance-level motion decomposition module, separates the Gaussians into (semi-) static and dynamic. Tracking is then performed on the static part of the scene, while mapping utilises our Instance-aware warp field module to model deformations in the scene. This module operates on a fixed set of control nodes, distributed between instance regions using our Semantic node allocation module, and propagates the estimated deformation to all dynamic Gaussians in an instance region.

## 3.1 Instance-level motion decomposition.

The optical flow $\mathbf { F } _ { t ^ { \prime }  t }$ from an earlier frame $t ^ { \prime } ~ = ~ t - 5$ to the current frame t is estimated by zero-shot WAFT [17]. Each pixel u of frame t<sup>′</sup> with depth $\mathbf { D } _ { t ^ { \prime } }$ is back-projected to $\mathbf { X } ( \mathbf { u } ) \mathbf { \Psi } = \mathbf { \Psi } \mathbf { D } _ { t ^ { \prime } } ( \mathbf { u } ) \mathbf { K } ^ { - 1 } \tilde { \mathbf { u } }$ , with K the camera intrinsics and u˜ the homogeneous pixel coordinates. The optical flow provides its location $\mathbf { u } + \mathbf { F } _ { t ^ { \prime }  t } ( \mathbf { u } )$ in frame t. Leveraging these 3D–2D correspondences, PnP-RANSAC estimates the relative camera rotation R<sup>ˆ</sup> and translation <sup>ˆ</sup>t between $t ^ { \prime }$ and t, with independently moving pixels rejected as outliers. The noisy residual flow is:

$$
\mathbf { r } ( \mathbf { u } ) = \mathbf { F } _ { t ^ { \prime }  t } ( \mathbf { u } ) - \Big ( \pi \big ( \hat { \mathbf { R } } \mathbf { X } ( \mathbf { u } ) + \hat { \mathbf { t } } \big ) - \mathbf { u } \Big ) ,\tag{1}
$$

where π is the projection function. To obtain robust region-level evidence, the residuals are aggregated in SLIC superpixels split along the instance boundaries, so that each superpixel P lies within a single instance. Each $P$ is summarised by its median residual

$\bar { \mathbf { r } } _ { P }$ and its median forward–backward flow error $\bar { e } _ { P } ,$ which estimates the local flow uncertainty. An instance $\ell ,$ covered by the superpixels $\mathcal { P } _ { \ell } .$ is then tested with two complementary statistics:

$$
z _ { \ell } ^ { A } = \mathop { \mathrm { m e d } } _ { P \in \mathcal { P } _ { \ell } } \frac { \| \bar { \mathbf { r } } _ { P } \| _ { 2 } } { \bar { e } _ { P } } , \qquad z _ { \ell \ell ^ { \prime } } ^ { B } = \frac { \| \mathbf { m } _ { \ell } - \mathbf { m } _ { \ell ^ { \prime } } \| _ { 2 } } { \sqrt { s _ { \ell } ^ { 2 } / n _ { \ell } + s _ { \ell ^ { \prime } } ^ { 2 } / n _ { \ell ^ { \prime } } } } .\tag{2}
$$

$z _ { \ell } ^ { A }$ measures residual motion relative to flow noise, while $z _ { \ell \ell ^ { \prime } } ^ { B }$ measures motion diferences across the boundary with neighbouring instance $\ell ^ { \prime }$ . For boundary superpixels $\mathcal { Q } _ { \ell \ell ^ { \prime } } \subseteq \mathcal { P } _ { \ell }$ , we define the median residual, robust scale estimate, and superpixel count:

$$
\mathbf { m } _ { \ell } = \mathop { \mathrm { ~ m e d ~ } } _ { P \in \mathcal { Q } _ { \ell \ell ^ { \prime } } } \bar { \mathbf { r } } _ { P } , \qquad s _ { \ell } = M A D _ { P \in \mathcal { Q } _ { \ell \ell ^ { \prime } } } \big | \big | \bar { \mathbf { r } } _ { P } - \mathbf { m } _ { \ell } \big | \big | , \qquad n _ { \ell } = | \mathcal { Q } _ { \ell \ell ^ { \prime } } | ,\tag{3}
$$

The same quantities are computed on the neighbouring side. The tests are complementary: absolute residuals may reflect pose errors, whereas boundary diferences suppress shared errors but cannot determine which instance moved. An instance is classified as dynamic if both statistics exceed $\varepsilon$ for at least one neighbour and its residual is larger than the neighbour’s. We use one threshold across both tests and all sequences. The module outputs the static mask $M _ { t } ^ { \mathrm { s t } }$ for tracking, and its complement $M _ { t } ^ { \mathrm { { d y n } } } = \sim M _ { t } ^ { \mathrm { { s t } } }$ the bidirectional flow, and the per-instance $s _ { \ell }$ as a measure of non-rigidity for mapping.

## 3.2 Tracking

The pose $\mathbf { T } _ { t }$ is optimised from the previous pose against the the static part $M _ { t } ^ { \mathrm { s t } }$ [6], using the colour $\mathcal { L } _ { c }$ , depth $\mathcal { L } _ { d }$ and instance segmentation $\mathcal { L } _ { i d }$ losses [7]:

$$
{ \mathcal { L } } _ { \mathrm { t r a c k } } = \lambda _ { \mathrm { c } } { \mathcal { L } } _ { \mathrm { c } } + \left( 1 - \lambda _ { \mathrm { d } } \right) { \mathcal { L } } _ { d } + \lambda _ { \mathrm { i d } } { \mathcal { L } } _ { \mathrm { i d } }\tag{4}
$$

## 3.3 Mapping

Instance-aware Warp field: We model the motion of the dynamic Gaussians with a sparse set of control nodes [4], partitioned by instance, so that each instance deforms independently. The dynamic Gaussians are kept in a time-independent canonical space, and the nodes define a warp field that maps the canonical space to their state at time t. Each node k has a canonical position $\mathbf { p } _ { k }$ and an instance label $\ell _ { k }$ , and the deformation MLP $\Psi _ { \ell _ { k } }$ of its instance predicts its deformation at time t:

$$
( \delta \mathbf { x } _ { k } , \delta \mathbf { q } _ { k } , \delta \mathbf { s } _ { k } ) = \Psi _ { \ell _ { k } } \big ( \gamma ( \mathbf { p } _ { k } ) , \gamma ( t ) \big ) ,\tag{5}
$$

where $\delta \mathbf { x } _ { k } \in \mathbb { R } ^ { 3 } , \delta \mathbf { q } _ { k } \in \mathbb { R } ^ { 4 }$ and $\delta \mathbf { s } _ { k } \in \mathbb { R } ^ { 3 }$ are ofsets to the position, rotation and scale of the Gaussians it drives, and $\gamma$ denotes the frequency-based positional encoding [18]. Linear blend skinning [4] propagates these outputs to each dynamic Gaussian i from its neighbourhood ${ \mathcal { N } } _ { i } .$ the K nearest nodes in its own instance $\ell _ { i }$

Semantic node allocation: A fixed budget of B nodes is redistributed across instances every frame. Instances whose dynamic area $| \Omega _ { \ell } ^ { t } |$ , i.e. their number of pixels in the dynamic mask $M _ { t } ^ { \mathrm { { d y n } } }$ , exceeds a minimum, form the active set $\boldsymbol { A } _ { t }$ . All other instances keep at most $n _ { \mathrm { a b s } }$ nodes to preserve their past motion, and the remaining budget $B _ { \mathrm { f r e e } }$ is shared among the active instances with targets:

$$
B _ { \ell } = \operatorname* { m a x } \biggl ( n _ { \mathrm { m i n } } , \mathrm { r o u n d } \biggl ( B _ { \mathrm { f r e e } } \frac { w _ { \ell } } { \sum _ { \ell ^ { \prime } \in A _ { t } } w _ { \ell ^ { \prime } } } \biggr ) \biggr ) , \qquad w _ { \ell } = \sqrt { | \Omega _ { \ell } ^ { t } | } ( 1 + s _ { \ell } ) ,\tag{6}
$$

where $n _ { \mathrm { m i n } }$ is the minimum allocation and $s _ { \ell }$ the instance non-rigidity: the square root limits the advantage of large instances, while $s _ { \ell }$ favours deforming ones. Deficits are filled first from unused nodes, then from instances above their target, which give up their least-supported nodes that are the nearest node of the fewest Gaussians.

Optimisation: The mapping objective calculated on the dynamic part $M _ { t } ^ { \mathrm { { d y n } } }$ is:

$$
\mathcal { L } _ { \mathrm { m a p } } = \lambda \mathcal { L } _ { c } + \left( 1 - \lambda \right) \mathcal { L } _ { d } + \lambda _ { \mathrm { f } } \mathcal { L } _ { \mathrm { f l o w } } + \lambda _ { \mathrm { s f } } \mathcal { L } _ { \mathrm { s f } } + \lambda _ { \mathrm { i d } } \mathcal { L } _ { \mathrm { i d } } + \lambda _ { \mathrm { f e a t } } \mathcal { L } _ { \mathrm { f e a t } } + \mathcal { L } _ { \mathrm { r e g } } ,\tag{7}
$$

where $\mathcal { L } _ { c } ~ , ~ \mathcal { L } _ { d }$ and $\mathcal { L } _ { \mathrm { i d } }$ are the photometric, depth and instance segmentation losses [6]; ${ \mathcal { L } } _ { \mathrm { { f l o w } } }$ and $\mathcal { L } _ { \mathrm { s f } }$ are used following [6, 19, 20] for optical flow and scene flow; L<sub>feat</sub> compares features [21] of the rendered and input images, from the frozen DINO V3 encoder of WAFT following [8]; and Lreg $\mathcal { L } _ { \mathrm { r e g } }$ comprises the isotropy and ARAP terms [4, 20]. Following [20], dynamic Gaussians have a temporal GMM opacity, which we gate by instance. Loss weights and further details are given in the supplementary material.

## 3.4 CRCD++

The expanded Comprehensive Robotic Cholecystectomy Dataset (eCRCD) [16] records 18 ex-vivo porcine Cholecystectomies by seven surgeons on a dVRK-controlled da Vinci, with stereo video and camera kinematics. Its tissue masks are propagated by Track Anything [22] from one hand-labelled frame per 120-frame clip and therefore, degrade under occlusion, camera motion and deformation. Finally, eCRCD does not provide dense tool masks, and only provides keypoints for them. In order to address the limitations of eCRCD, we refine a subset of 20 sequences (200–2108 frames; 14,688 stereo pairs), and augment every frame as follows. Dense masks for Four semantic classes (background, liver, gallbladder, tool) are produced by a human annotator. Each tool carries its own instance ID(Fig. 2). Depth is estimated on every rectified pair with

![](images/219bafe80424b8bce39547d1a2808aa478955765a9d00029ed492dd352a1663e.jpg)  
Fig. 2 Semantic annotations. Input — eCRCD Track Anything — Our manual annotations.

![](images/3ad4627480fc09df67332d2a02c0e50c92c536e3e0e85b30bdd4b07e8bd33c0b.jpg)  
Fig. 3 Our camera tracking results on CRCD++ C1 and C3 - our Instance-level motion decomposition allows EndoS<sup>3</sup> to produce much smoother trajectories.

zero-shot FoundationStereo (FS) [23].

## 4 Experiments

Datasets: Five CRCD++ sequences: C1, C2, E3, C3 and G3 and Semantic-Super (SemSup) [1] test sequences with the pseudo GT depth we produce using FS. Implementation and Evaluation: We compare against DDS-SLAM [7], SNI-SLAM [8], SGS-SLAM [9], SemGauss-SLAM [10], Semantic-SuPer [1] and PERSEUS [2], using their public repositories. Semantic methods receive each dataset’s semantic maps as input (the ground-truth labels on CRCD++ and the released DeepLabV3+ masks on Semantic-SuPer). All experiments use one NVIDIA A100 (40 GB). In evaluation, We use ATE-RMSE for tracking, PSNR, SSIM and LPIPS for mapping. On Semsup, tissue tracking is evaluated by reprojection error using the tracked sparse pins. For additional visualisations, examples of complex scenarios, and a more detailed performance review, please refer to the supplementary material.

## 5 Results

Tracking on CRCD++: Table 2 shows EndoS<sup>3</sup> achieves the best ATE result in 3/5 sequences and the lowest average, with an improvement of 17% over the second best model. Fig. 3 demonstrates that the benchmarked models cannot separate camera motion from tissue deformation, and absorb deformation into the camera pose updates, producing jittery trajectories compared to our smoother predictions, the result of using our Instance-level motion decomposition module.

Reconstruction and Depth: Table 3 shows we achieve an average improvement of 2 dB and 0.81 dB over the second best model on CRCD++ and SemSup respectively. In depth estimation, Table 2 shows that on both datasets, EndoS<sup>3</sup> leads every tissue tracking (Pins) and depth metric. This is also visible in Fig. 5.

SNI-SLAM  
SGS-SLAM  
Table 2 Tracking, depth and semantics. ATE-RMSE and depth error in mm; mIoU in $\% ;$ SemSuP: pin reprojection error in px↓. “–”: not available.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="6">CRCD++</td><td>Sem.-SuPer</td></tr><tr><td>C1</td><td>C2</td><td>C3</td><td>G3</td><td>E3</td><td>Avg.</td><td>Pins</td></tr><tr><td rowspan="2">DDS-SLAM [7]</td><td>ATE↓</td><td>1.78</td><td>3.36</td><td>6.59</td><td>4.83</td><td>5.13</td><td>4.34</td><td>9.21</td></tr><tr><td>Depth↓ ATE↓</td><td>11.1 1.56</td><td>19.9 1.48</td><td>13.4 6.96</td><td>8.0 2.60</td><td>14.8 5.75</td><td>13.45 3.67</td><td></td></tr><tr><td rowspan="2">SNI-SLAM [8]</td><td>Depth↓</td><td>5.1</td><td>17.9</td><td>14.3</td><td>8.8</td><td>17.6</td><td>12.72</td><td>9.70</td></tr><tr><td>mIoU↑</td><td>63.6</td><td>52.1</td><td>60.2</td><td>32.6</td><td>41.5</td><td>50.0</td><td></td></tr><tr><td rowspan="2">SGS-SLAM [9]</td><td>ATE↓</td><td>2.13</td><td>5.34</td><td>8.54</td><td>4.24</td><td>5.21</td><td>5.09</td><td></td></tr><tr><td>Depth↓</td><td>3.1</td><td>16.8</td><td>6.2</td><td>2.3</td><td>21.7</td><td>10.04</td><td>9.04</td></tr><tr><td rowspan="2">SemGauss-SLAM [10]</td><td>mIoU↑</td><td>86.3</td><td>53.9</td><td>62.2</td><td>52.8</td><td>42.2</td><td>59.5</td><td></td></tr><tr><td>ATE↓</td><td>2.27</td><td>5.00</td><td>5.86</td><td>1.87</td><td>5.41</td><td>4.08</td><td>9.59</td></tr><tr><td rowspan="2"></td><td>Depth↓</td><td>3.2</td><td>18.5</td><td>8.4</td><td>3.2</td><td>26.0</td><td>11.86</td><td></td></tr><tr><td>mIoU↑</td><td>84.8</td><td>41.2</td><td>48.2</td><td>48.9</td><td>48.4</td><td>54.3</td><td></td></tr><tr><td>PERSEUS [2]</td><td>ATE↓ Depth↓</td><td>2.90 20.7</td><td>5.29 114.9</td><td>9.90 75.0</td><td>2.92 6.5</td><td>5.82 302.4</td><td>5.37 103.90</td><td>10.16</td></tr><tr><td>Semantic-SuPer [1]</td><td>Depth↓</td><td>10.2</td><td></td><td>16.1</td><td>3.1</td><td>35.3</td><td></td><td>23.17</td></tr><tr><td rowspan="2">Ours</td><td>ATE↓</td><td>1.31</td><td>2.24</td><td>4.48</td><td>1.80</td><td>5.29</td><td>3.03</td><td></td></tr><tr><td>Depth↓ mIoU↑</td><td>2.2 91.5</td><td>4.6 78.9</td><td>3.1 84.3</td><td>1.8 51.9</td><td>3.1 81.9</td><td>2.96 77.7</td><td>8.20</td></tr></table>

Semantics: EndoS<sup>3</sup> achieves the highest mIoU on both datasets in Table 2, while Fig. 4 shows the baselines’ semantic degradation on CRCD++ and SemSup.

![](images/3c5327d7e9927e24804e3cfd302239e7eb9b4994de699d13a37b5f610d68765f.jpg)  
Fig. 4 Semantic renders on CRCD++ and Semantic-SuPer.

Ablation studies: Table 4 identifies motion decomposition as the most impactful module, with its removal causing the largest degradation. After that, Instance-aware deformation contributes more to depth accuracy, whereas semantic node allocation has a greater impact on rendering, tracking, and semantics.

Table 3 Rendering quality on CRCD++ and Semantic-SuPer. PERSEUS has no renderer.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="6">CRCD++</td><td rowspan="2">Sem.-SuPer</td></tr><tr><td>C1</td><td>C2</td><td>C3</td><td>G3</td><td>E3</td><td>Avg.</td></tr><tr><td rowspan="3">DDS-SLAM [7]</td><td>PSNR↑</td><td>22.19</td><td>19.52</td><td>21.68</td><td>22.90</td><td>21.47</td><td>21.55</td><td>Avg. 30.81</td></tr><tr><td>SSIM↑</td><td>0.667</td><td>0.583</td><td>0.596</td><td>0.580</td><td>0.639</td><td>0.613</td><td>0.792</td></tr><tr><td>LPIPS↓</td><td>0.669</td><td>0.720</td><td>0.717</td><td>0.734</td><td>0.644</td><td>0.697</td><td>0.248</td></tr><tr><td rowspan="3">SNI-SLAM [8]</td><td>PSNR↑</td><td>19.32</td><td>16.09</td><td>16.60</td><td>16.39</td><td>18.61</td><td>17.40</td><td>22.47</td></tr><tr><td>SSIM↑</td><td>0.694</td><td>0.552</td><td>0.631</td><td>0.703</td><td>0.577</td><td>0.631</td><td>0.660</td></tr><tr><td>LPIPS↓</td><td>0.584</td><td>0.701</td><td>0.640</td><td>0.615</td><td>0.660</td><td>0.640</td><td>0.448</td></tr><tr><td rowspan="3">SGS-SLAM [9]</td><td>PSNR↑</td><td>20.79</td><td>13.81</td><td>18.30</td><td>18.44</td><td>12.13</td><td>16.69</td><td>22.72</td></tr><tr><td>SSIM↑</td><td>0.782</td><td>0.524</td><td>0.780</td><td>0.810</td><td>0.486</td><td>0.676</td><td>0.721</td></tr><tr><td>LPIPS↓</td><td>0.342</td><td>0.594</td><td>0.460</td><td>0.494</td><td>0.646</td><td>0.507</td><td>0.333</td></tr><tr><td rowspan="3">SemGauss-SLAM [10]</td><td>PSNR↑</td><td>20.21</td><td>15.03</td><td>17.05</td><td>17.67</td><td>11.95</td><td>16.38</td><td>23.01</td></tr><tr><td>SSIM↑</td><td>0.770</td><td>0.563</td><td>0.734</td><td>0.779</td><td>0.491</td><td>0.667</td><td>0.718</td></tr><tr><td>LPIPS↓</td><td>0.347</td><td>0.605</td><td>0.515</td><td>0.556</td><td>0.701</td><td>0.545</td><td>0.304</td></tr><tr><td rowspan="3">Semantic-SuPer [1]</td><td>PSNR↑</td><td>11.02</td><td>7.87</td><td>7.74</td><td>8.11</td><td>9.90</td><td>8.93</td><td>17.12</td></tr><tr><td>SSIM↑</td><td>0.257</td><td>0.058</td><td>0.154</td><td>0.135</td><td>0.230</td><td>0.167</td><td>0.542</td></tr><tr><td>LPIPS↓</td><td>0.819</td><td>0.888</td><td>0.992</td><td>0.941</td><td>0.858</td><td>0.900</td><td>0.611</td></tr><tr><td rowspan="3">Ours</td><td>PSNR↑</td><td>29.26</td><td>20.28</td><td>22.95</td><td>19.12</td><td>26.14</td><td>23.55</td><td>31.62</td></tr><tr><td>SSIM↑</td><td>0.893</td><td>0.783</td><td>0.813</td><td>0.867</td><td>0.868</td><td>0.845</td><td>0.867</td></tr><tr><td>LPIPS↓</td><td>0.250</td><td>0.419</td><td>0.383</td><td>0.375</td><td>0.321</td><td>0.350</td><td>0.217</td></tr></table>

Table 4 Ablation on CRCD++ - seq C4. MD: motion decomposition (otherwise only instruments are dynamic); IW: instance-aware warp field (otherwise one shared MLP); NA: semantic node allocation (otherwise a fixed node set).
<table><tr><td>MD IW NA</td><td></td><td></td><td>PSNR↑</td><td>Depth↓</td><td>ATE↓</td><td>mIoU↑</td></tr><tr><td>X</td><td>√</td><td>√</td><td>22.18</td><td>5.85</td><td>2.91</td><td>52.8</td></tr><tr><td>√</td><td>×</td><td>√</td><td>23.28</td><td>4.62</td><td>2.55</td><td>71.3</td></tr><tr><td>√</td><td>√</td><td>X</td><td>23.08</td><td>4.43</td><td>2.83</td><td>70.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td>24.45</td><td>4.35</td><td>2.75</td><td>73.4</td></tr></table>

Table 5 evaluates the two motion tests. Using $z ^ { A }$ alone gives the highest trajectory error, whereas $z ^ { B }$ alone tracks more accurately but renders slightly worse. Combining both tests gives the best result. Table 6 replaces the GT labels with predictions from

Table 5 Motion-decomposition CRCD++ - seq C4. $z ^ { A } { \mathrel { \mathop : } }$ instance-level test; $z ^ { B } ;$ boundary-band test. Removing either leaves the other as the sole criterion.
<table><tr><td> $\overline { { z ^ { A } } }$ </td><td> $\overline { { z ^ { B } } }$ </td><td>PSNR↑</td><td>ATE↓</td><td>mIoU↑</td></tr><tr><td> $\overline { { \checkmark } }$ </td><td>X</td><td>24.39</td><td>3.03</td><td>71.9</td></tr><tr><td> $\frac { \times } { \surd }$ </td><td>√</td><td>24.29</td><td>2.80</td><td>72.6</td></tr><tr><td></td><td>√</td><td>24.45</td><td>2.75</td><td>73.4</td></tr></table>

a pretrained Mask R-CNN [16] for organs and PFedSIS [24] for tools from previous works, proving that even with estimated segmentations our method outperforms the second-best DDS-SLAM that takes as input ground truth segmentations.

Table 6 Predicted segmentation Avg. performance on CRCD++.
<table><tr><td>Method</td><td>PSNR (dB)↑</td><td>Depth (mm)↓</td><td>ATE (mm)↓</td><td>mIoU (%)↑</td></tr><tr><td>DDS-SLAM (GT)</td><td>21.55</td><td>13.45</td><td>4.34</td><td></td></tr><tr><td>Ours (GT)</td><td>23.55</td><td>2.96</td><td>3.03</td><td>84.2</td></tr><tr><td>Ours (Pred.)</td><td>24.28</td><td>3.54</td><td>3.30</td><td>65.8</td></tr></table>

![](images/02de20a423d6bffd1762c670a69e1b548018997710d3d89bbe41be7b3e1af039.jpg)  
Fig. 5 CRCD++ and SemSuP: Rendered colour map, CRCD++: Rendered depth map, SemSuP: Depth rendered as a pointcloud error heatmap.

## 6 Conclusion

We introduce EndoS<sup>3</sup>, a semantic Gaussian splatting SLAM framework that uses instance semantics to guide motion decomposition and deformation modelling, alongside CRCD++, a robotic cholecystectomy benchmark for in-depth semantic SLAM evaluation. EndoS<sup>3</sup> outperforms SOTA on both CRCD++ and SemSup, demonstrating the value of semantic-awareness. Future work includes addressing surgical image artifacts, primarily low illumination conditions, in semantic SLAM.

## References

[1] Lin, S., Miao, A.J., Lu, J., Yu, S., Chiu, Z.-Y., Richter, F., Yip, M.C.: Semantic-SuPer: A semantic-aware surgical perception framework for endoscopic tissue identification, reconstruction, and tracking. In: ICRA, pp. 4739–4746 (2023). https://doi.org/10.1109/ICRA48891.2023.10160746

[2] Acar, A., et al.: Perseus: perception with semantic endoscopic understanding and SLAM. Int J Comput Assist Radiol Surg 21(7), 1465–1474 (2026) https://doi. org/10.1007/s11548-026-03717-w

[3] Hayoz, M., Hahne, C., Gallardo, M., Candinas, D., Kurmann, T., Allan, M., Sznitman, R.: Learning how to robustly estimate camera pose in endoscopic videos. Int J Comput Assist Radiol Surg 18(7), 1185–1192 (2023) https://doi.org/10.1007/ s11548-023-02919-w

[4] Huang, Y.-H., Sun, Y.-T., Yang, Z., Lyu, X., Cao, Y.-P., Qi, X.: SC-GS: Sparsecontrolled Gaussian splatting for editable dynamic scenes. In: CVPR, pp. 4220– 4230 (2024). https://doi.org/10.1109/CVPR52733.2024.00404

[5] Hayoz, M., et al.: Online 3D reconstruction and dense tracking in endoscopic videos. In: MICCAI, pp. 444–454. Springer, Cham (2024). https://doi.org/10. 1007/978-3-031-72089-5 42

[6] Li, Y., Fang, Y., Zhu, Z., Li, K., Ding, Y., Tombari, F.: 4D Gaussian splatting SLAM. In: ICCV, pp. 25019–25028 (2025). https://doi.org/10.1109/ICCV51701. 2025.02320

[7] Shan, J., Li, Y., Yang, L., Feng, Q., Han, L., Wang, H.: DDS-SLAM: Dense semantic neural SLAM for deformable endoscopic scenes. In: IROS, pp. 10837– 10842 (2024). https://doi.org/10.1109/IROS58592.2024.10802442

[8] Zhu, S., Wang, G., Blum, H., Liu, J., Song, L., Pollefeys, M., Wang, H.: SNI-SLAM: Semantic neural implicit SLAM. In: CVPR, pp. 21167–21177 (2024). https://doi.org/10.1109/CVPR52733.2024.02000

[9] Li, M., Liu, S., Zhou, H., Zhu, G., Cheng, N., Deng, T., Wang, H.: SGS-SLAM: Semantic Gaussian splatting for neural dense SLAM. In: ECCV, pp. 163–179. Springer, Cham (2024). https://doi.org/10.1007/978-3-031-72751-1 10

[10] Zhu, S., Qin, R., Wang, G., Liu, J., Wang, H.: SemGauss-SLAM: Dense semantic Gaussian splatting SLAM. In: IROS, pp. 21174–21181 (2025). https://doi.org/ 10.1109/IROS60139.2025.11246462

[11] Straub, J., et al.: The Replica dataset: A digital replica of indoor spaces. arXiv (2019) https://doi.org/10.48550/arXiv.1906.05797

[12] Hong, W.-Y., Kao, C.-L., Kuo, Y.-H., Wang, J.-R., Chang, W.-L., Shih, C.-S.: CholecSeg8k: A semantic segmentation dataset for laparoscopic cholecystectomy based on Cholec80. arXiv (2020) https://doi.org/10.48550/arXiv.2012.12453

[13] Allan, M., et al.: 2018 robotic scene segmentation challenge. arXiv (2020) https: //doi.org/10.48550/arXiv.2001.11190

[14] Mascagni, P., et al.: Endoscapes, a critical view of safety and surgical scene segmentation dataset for laparoscopic cholecystectomy. Sci Data 12, 331 (2025) https://doi.org/10.1038/s41597-025-04642-4

[15] Docea, R., et al.: The Dresden dataset for 4D reconstruction of non-rigid abdominal surgical scenes. Sci Data 13, 1334 (2026) https://doi.org/10.1038/ s41597-026-08289-7

[16] Oh, K.-H., et al.: Expanded Comprehensive Robotic Cholecystectomy Dataset. J Med Robot Res 10(03n04), 2550006 (2025) https://doi.org/10.1142/ S2424905X25500060

[17] Wang, Y., Deng, J.: WAFT: Warping-alone field transforms for optical flow. In: ICLR (2026)

[18] Mildenhall, B., et al.: NeRF: Representing scenes as neural radiance fields for view synthesis. In: ECCV, pp. 405–421. Springer, Cham (2020). https://doi.org/ 10.1007/978-3-030-58452-8 24

[19] Wu, T., et al.: EndoFlow-SLAM: Real-time endoscopic SLAM with flowconstrained Gaussian splatting. In: MICCAI, pp. 202–212. Springer, Cham (2025). https://doi.org/10.1007/978-3-032-05114-1 20

[20] Wang, Y., Lee, G.H.: Flow4DGS-SLAM: Optical flow-guided 4D Gaussian splatting SLAM. In: CVPR, pp. 33364–33373 (2026)

[21] Sim´eoni, O., et al.: DINOv3. arXiv (2025) https://doi.org/10.48550/arXiv.2508. 10104

[22] Yang, J., Gao, M., Li, Z., Gao, S., Wang, F., Zheng, F.: Track Anything: Segment Anything meets videos. arXiv (2023) https://doi.org/10.48550/arXiv.2304.11968

[23] Wen, B., Trepte, M., Aribido, J., Kautz, J., Gallo, O., Birchfield, S.: FoundationStereo: Zero-shot stereo matching. In: CVPR, pp. 5249–5260 (2025). https: //doi.org/10.1109/CVPR52734.2025.00495

[24] Xu, J., et al.: Personalizing federated instrument segmentation with visual trait priors in robotic surgery. IEEE Trans Biomed Eng 72(6), 1886–1896 (2025) https: //doi.org/10.1109/TBME.2025.3526667