# AURORA: Active Uncertainty-Driven Re-Orientation for In-Hand Reconstruction

Feiyu Zhao School of Information Science and Technology ShanghaiTech University, China zhaofy12024@shanghaitech.edu.cn

Yuetong Li<sup>∗</sup> School of Information Science and Technology ShanghaiTech University, China liyt2023@shanghaitech.edu.cn

Chenxi Xiao<sup>†</sup> School of Information Science and Technology ShanghaiTech University, China xiaochx@shanghaitech.edu.cn

![](images/d249177dbc38080ea3b5c9c8d748e57af7be76558e56cb75167e2963c6e9cdf6.jpg)  
Figure 1: AURORA, a framework for active in-hand object reconstruction using a fixed RGB-D camera. The hand reorients an object using an in-hand manipulation policy and an uncertaintydriven next-best-view planner, exposing unobserved regions and accelerating 3D reconstruction.

Abstract: Observing objects grasped by a robot hand is challenging due to severe visual occlusions. Although in-hand manipulation can expose hidden surfaces, existing approaches often rely on predefined or open-loop reorientation strategies that do not explicitly target under-observed regions. We propose AURORA, an active 3D reconstruction framework that closes the loop between online objectcentric reconstruction and in-hand reorientation. At its core, Ray-GPIS estimates direction-wise reconstruction uncertainty along candidate viewing rays and selects next-best-view targets using an uncertainty–novelty objective, which are realized through an axis-conditioned in-hand rotation policy. The resulting RGB-D observations are fused incrementally using CAD-free 6D pose tracking and lightweight geometric reconstruction. Experiments demonstrate that AURORA improves reconstruction quality and information-acquisition efficiency over non-active rotation strategies, while Ray-GPIS also outperforms active view-planning baselines in reconstruction performance, action-ranking quality, and planning efficiency. Targeted ablations further validate its robustness to hand occlusion and pose errors. The project webpage is available at https://aurorahand.github.io/.

Keywords: Probabilistic learning and uncertainty in robotics, Multimodal perception, sensor fusion, and robot vision, Active In-Hand Object Reconstruction

## 1 Introduction

Human perception is inherently active: when inspecting an object, humans deliberately reorient it to reveal surfaces occluded by the hand [1]. Robotic systems have a similar need during manipulation, where hand-object occlusions and limited camera viewpoints lead to partial observability. Unlike humans, who can simultaneously coordinate vision and manipulation to acquire novel views, most robotic systems rarely leverage their physical dexterity to resolve such occlusions, restricting the information available for understanding the object.

A primary reason for this limitation is the lack of a planning framework that translates perceptual needs into manipulation actions. While prior research has explored using robot manipulation data for object understanding, these approaches typically reconstruct objects from passively collected manipulation sequences, such as pick-and-place interactions [2] or in-hand rotation videos [3]. In these methods, the robot follows predefined action sequences or open-loop policies, serving merely as a passive data collector without determining where to observe next. Because they do not explicitly reason about viewpoint informativeness or convert perceptual uncertainty into action decisions, such pipelines are time-inefficient and may leave occluded regions persistently under-observed.

To efficiently and accurately recover the complete 3D geometry of unknown objects, we propose an active in-hand reconstruction framework that tightly couples visual perception with manipulation in a closed-loop manner. Rather than executing predefined motion sequences, the system follows an informative policy that actively selects actions to improve perceptual coverage. At each time step, the robot explicitly represents spatial geometric uncertainty over the object surface, which is then used to plan the next in-hand reorientation action, accelerating the exposure of under-observed regions. On the basis of the collected views, we implement a geometric fusion framework to integrate multi-view observations. Together, these components enable fully autonomous object reconstruction through coordinated hand–eye interactions.

In summary, the contributions of this paper include:

• AURORA, an informative framework for capturing complete object geometry in-hand, enabling autonomous exploration with improved efficiency and completeness.

• Ray-GPIS, a ray-conditioned planner that converts GPIS point-wise uncertainty into direction-wise view scores for active in-hand reorientation.

• An online object-centric fusion pipeline, and extensive real-world evaluations showing improved completeness and exploration efficiency under hand-object occlusion.

## 2 Related Works

## 2.1 In-Hand Manipulation for Object Reconstruction

In-hand manipulation enables a robot to reorient an object within its grasp, providing a natural mechanism for exposing previously unseen surfaces. Existing approaches have studied both model-based manipulation, which plans motions using contact mechanics [4, 5], and learning-based manipulation, which improves robustness to novel objects and complex dynamics [6, 7, 8, 9, 10]. Recent works further incorporate tactile sensing to improve contact perception and dexterous control [11, 12, 13].

However, most in-hand manipulation methods optimize for reorientation stability, speed, or task success, rather than reconstruction quality. A recent work NeuralFeels [3] is closely related, as it reconstructs in-hand object geometry from visuotactile observations. Despite this similarity, NeuralFeels reconstructs object pose and shape from a given sequence of visual and high-resolution tactile interactions, but does not include an active planning mechanism. Although a direct benchmark is not feasible due to differences in hardware settings, we discuss the relationship between NeuralFeels and our work along with limited comparisons in Appendix F.7.

![](images/b3a991b01623dd24c1071222d013379a6e418f8d1e7d9fce9060ec2180997617.jpg)  
Figure 2: Overview of our technical pipeline. The system integrates four modules: (a) in-hand object reorientation with the Leap Hand; (b) 6D pose tracking via BundleTrack [19]; (c–d) reconstructions; and (e) uncertainty-driven next-best-view planning.

## 2.2 Object Reconstruction and Tracking under Occlusion

Object reconstruction from image sequences has been widely studied, with approaches ranging from Structure-from-Motion [14] and RGB-D fusion to neural implicit representations [15, 16, 17, 18]. These methods usually assume static scene geometry and establish multi-view consistency primarily through camera motion. In contrast, in robotic in-hand object reconstruction, the object moves unpredictably under contact and is occluded by the hand, making such multi-view consistency more difficult to achieve. Recent model-free trackers and foundation-model-based pose estimators provide useful tools for addressing this issue [19, 20, 21, 22], and have been used to address similar occlusion challenges arising in human–object interaction reconstruction [23, 24, 25, 26]. However, these works primarily reconstruct objects from observed sequences and do not address active action or viewpoint selection for revealing previously unobserved object regions.

## 2.3 Active Exploration and Uncertainty-Driven View Planning

Active perception seeks to plan sensing actions that maximize task-relevant information acquisition. Most existing systems explore static, unknown environments using mobile sensors [27, 28, 29]. These methods use sensing modalities such as cameras or tactile sensors [30, 31] and typically optimize objectives that reduce geometric uncertainty or maximize expected information gain [32]. Common formulations include frontier-based exploration of unknown space [33], as well as information-theoretic criteria based on Fisher information [34], mutual information [35], and related measures. Beyond mobile-camera exploration, recent work has also studied active object reconstruction with fixed sensors and manipulable objects. For instance, a robot can grasp and re-pose an object in front of a static RGB-D camera to expose previously occluded surfaces [36]. Other interactive perception approaches use purposeful manipulation to incrementally reveal unseen geometry and improve reconstruction completeness [37]. However, these methods largely assume controlled object reposing or external manipulation, rather than continuous in-hand manipulation. As a result, they do not address the coupled challenges of persistent hand-induced occlusion, object motion during interaction, and action selection for revealing uncertain object regions in the robot hand.

![](images/bd17a351c10002185b1cf0577bb81eceea92b7fe43e6e60c5d9a49e772549111.jpg)  
Figure 3: Ray-GPIS estimates direction-wise reconstruction uncertainty. Rays are cast from the object center c. Each direction u is assigned an anchor point ${ \bf x } _ { 0 } ( { \bf u } )$ near the predicted surface intersection, and GPIS variance is aggregated in a local receptive field to obtain U(u).

## 3 Methodology

The overall pipeline of AURORA is illustrated in Fig. 2. We tackle in-hand 3D reconstruction as a closed-loop perception–action problem. From an initial observation, we iterate: (i) select a rotation axis and execute an in-hand rotation; (ii) track the object pose and select keyframes; (iii) fuse keyframes into an incremental point-cloud reconstruction; and (iv) update an uncertainty model and compute the next-best view (NBV) for the next action. When the loop terminates, we extract a watertight mesh as the final reconstruction.

## 3.1 In-Hand Object Reorientation

We use a tactile-based in-hand rotation policy adapted from Rotating without Seeing [11] as the low-level action executor. Given a commanded rotation axis, the policy reorients the grasped object accordingly. We deploy the policy on a Leap Hand [38]; hardware modifications, tactile sensing, and policy training details are provided in Appendix A.

We define our action space in a hand-centered world frame W, whose origin is at the palm center. The z-axis aligns with the outward palm normal, and the x- and y-axes lie in the palm plane. The feasible action set is:

$$
{ \mathcal { A } } \triangleq \{ - x , - y , + z \} ,\tag{1}
$$

where each element denotes a rotation-axis primitive in W. At each planning step, the active planner selects $\mathbf { a } ^ { \star } \in { \cal A } .$ , which is then executed by the low-level policy.

## 3.2 Ray-GPIS Active View Planning

Actions are selected based on reconstruction uncertainty over object geometry. While existing GPISbased methods [39, 30] estimate point-wise uncertainty in 3D space, they do not directly provide view-dependent scores for in-hand exploration. To address this, Ray-GPIS converts point-wise spatial GPIS uncertainty into direction-wise planning scores by casting object-centered rays, anchoring them near predicted surface locations, and aggregating posterior variance within local receptive fields, as shown in Fig. 3.

## 3.2.1 Ray-GPIS Uncertainty Estimation

Specifically, we use the center c of the oriented bounding box of $\mathcal { P } _ { t }$ as the ray origin and uniformly sample N unit directions $\{ { \mathbf { u } } _ { i } \} _ { i = } ^ { N }$ on the sphere. Each ray is parameterized by a normalized radial coordinate $t _ { i } { \mathrm { : } }$

$$
\mathbf { x } \big ( t _ { i } ; \mathbf { u } _ { i } \big ) = \mathbf { c } + t _ { i } r \mathbf { u } _ { i } , \qquad t _ { i } \in [ 0 , t _ { \operatorname* { m a x } } ] ,\tag{2}
$$

where r is the maximum distance from c to any point in $\mathcal { P } _ { t } ,$ and $t _ { i }$ denotes the radial distance normalized by $r _ { \ast }$ Thus, $t _ { i } = 1$ approximately corresponds to the current point-cloud extent along the object scale, while $t _ { \mathrm { m a x } } > 1$ allows the ray to probe slightly beyond the observed geometry.

For each direction, Ray-GPIS assigns an anchor point $\mathbf { x } _ { 0 } ( \mathbf { u } _ { i } ) = \mathbf { x } ( t _ { 0 } ( \mathbf { u } _ { i } ) ; \mathbf { u } _ { i } )$ , which approximates the surface location along the ray. A ray is classified as a hit ray if it intersects the observed surface neighborhood represented by $\mathcal { P } _ { t } ;$ its anchor is placed at the first observed surface intersection. A miss ray has no reliable observed intersection. For miss rays, the anchor depth is inferred by interpolating nearby hit-ray depths on the viewing sphere. This enables $\mathrm { R a y }$ -GPIS to query uncertainty both around observed surfaces and near plausible surface locations in unobserved directions. Details of the definitions and implementation are provided in Appendix B.1.

We then fit a GPIS [39] model to the observed anchors and use its posterior variance $\sigma ^ { 2 } ( \mathbf { x } )$ as the spatial reconstruction uncertainty. The GPIS is trained only on hit-ray anchors, while miss-ray anchors are used only as query locations for uncertainty evaluation. To obtain a direction-wise score, we evaluate the variance within a local receptive field around each anchor:

$$
\begin{array} { r } { \mathcal { V } ( \mathbf { u } _ { i } ) = \left\{ \mathbf { x } ( t _ { i } ; \mathbf { u } ) \vert t _ { i } \in [ t _ { 0 } ( \mathbf { u } _ { i } ) - \Delta t , ~ t _ { 0 } ( \mathbf { u } _ { i } ) + \Delta t ] , ~ \angle ( \mathbf { u } , \mathbf { u } _ { i } ) \le \delta _ { u } \right\} . } \end{array}\tag{3}
$$

Here, $\Delta t$ defines a narrow radial band around the anchor depth, and $\delta _ { u }$ defines the angular neigh borhood on the viewing sphere.

The direction-wise uncertainty is computed by integrating the GPIS variance over the receptive field:

$$
U ( \mathbf { u } _ { i } ) = \int _ { \mathcal { V } ( \mathbf { u } _ { i } ) } \sigma ^ { 2 } ( \mathbf { x } ) d \mathbf { x } ,\tag{4}
$$

Details of GPIS training and posterior inference are provided in Appendix B.2.

## 3.2.2 Next-Best-View Selection

Given the direction-wise uncertainty $U ( \mathbf { u } _ { i } )$ estimated by Ray-GPIS, the next view should target regions that are both uncertain and insufficiently observed. However, directly maximizing $U ( \mathbf { u } _ { i } )$ may select directions close to the current fused geometry, leading to redundant observations. We therefore introduce a novelty weight based on the distance between each candidate receptive field and the current point cloud $\mathcal { P } _ { t }$ . For a candidate direction $\mathbf { u } _ { i }$ , let $\bar { d } _ { \mathrm { n n } } ( \mathbf { u } _ { i } )$ be the average nearestneighbor distance from samples in $\mathcal { \ V } ( \mathbf { u } _ { i } )$ to $\mathcal { P } _ { t }$ . We normalize it by the median 1-NN spacing $\bar { d } _ { 1 \mathrm { { n n } } }$ of $\mathcal { P } _ { t }$ , which represents the typical point-cloud resolution, and define

$$
\eta ( \mathbf { u } _ { i } ) = 1 - \exp \left( - \left( \frac { \bar { d } _ { \mathrm { n n } } ( \mathbf { u } _ { i } ) } { \beta \bar { d } _ { \mathrm { 1 n n } } } \right) ^ { 2 } \right) ,\tag{5}
$$

where $\beta$ controls the scale at which a direction is considered novel. The final NBV direction is selected by maximizing the product of reconstruction uncertainty and geometric novelty:

$$
\mathbf { u } _ { \mathrm { N B V } } ^ { \star } = \arg \operatorname* { m a x } _ { \mathbf { u } _ { i } } U ( \mathbf { u } _ { i } ) \eta ( \mathbf { u } _ { i } ) .\tag{6}
$$

This encourages the planner to prioritize regions that are not only uncertain, but also far from already fused observations.

## 3.2.3 Action Selection

The selected NBV direction specifies which object-side region should be exposed to the fixed camera, but it must still be converted into an executable in-hand rotation primitive. Since the low-level controller operates in the hand-centered world frame $W$ , we first transform the object-frame NBV direction into W:

$$
{ \bf d } _ { W } = { \bf R } _ { W O } ( t ) { \bf u } _ { \mathrm { N B V } } ^ { \star } ,\tag{7}
$$

where ${ \mathbf { R } } _ { W O } ( t )$ is the rotation component of $\mathbf { T } _ { W O } ( t ) = \mathbf { T } _ { W C } \mathbf { T } _ { C O } ( t )$ . We define $\mathbf { v } _ { W }$ as the unit vector from the current object center to the camera, expressed in W. We compute the minimal object rotation that aligns the desired direction ${ \bf d } _ { W }$ with $\mathbf { v } _ { W }$ , and represent it by the Lie-algebra vector $\omega \in \mathbb { R } ^ { 3 }$ . Since the controller only supports a discrete action set A, we choose the primitive whose axis is most aligned with this desired rotation:

$$
\mathbf { a } ^ { \star } = \arg \operatorname* { m a x } _ { \mathbf { a } \in \mathcal { A } } \mathbf { a } ^ { \top } \boldsymbol { \omega } .\tag{8}
$$

Details of the frame transformation and rotation-vector computation are provided in Appendix C.

## 3.3 Object-Centric Tracking and Reconstruction

Object-centric fusion and Ray-GPIS planning require object poses in RGB-D frames. We use BundleTrack [19], a CAD-free RGB-D tracker, to estimate $T _ { C O } ( t ) \in S E ( 3 )$ from segmented observations [40] and fuse them in a consistent object frame. To reduce the effect of hand–object occlusions, we apply a lightweight visibility-aware keyframe filter that retains only frames with sufficient object visibility, stable pose estimates, and adequate motion from the latest keyframe. Detailed tracking and keyframe-selection rules are provided in Appendix D.

Given the selected keyframes K, we maintain a compact object-centric point cloud as the geometric state for closed-loop planning. For each keyframe $t \in \kappa$ , masked depth pixels are back-projected into the camera frame, transformed into the object frame, and fused incrementally:

$$
\mathscr { P } _ { t } \triangleq \mathscr { C } \big ( \mathscr { P } _ { t - 1 } \cup T _ { C O } ( t ) ^ { - 1 } \pi ^ { - 1 } \big (  { \mathbf { D } } _ { t } ,  { \mathbf { M } } _ { t } ;  { \mathbf { K } } _ { \mathrm { c a m } } \big ) \big ) ,\tag{9}
$$

where $\mathbf D _ { t } , \mathbf M _ { t }$ , and $\mathbf { K } _ { \mathrm { c a m } }$ denote the depth image, object mask, and camera intrinsics, respectively; $\pi ^ { - 1 } ( \cdot )$ back-projects masked depth pixels into 3D camera-frame points; and C(·) applies voxel downsampling and outlier removal. The resulting point cloud $\mathcal { P } _ { t }$ is used by Ray-GPIS for uncertainty estimation and NBV planning.

After exploration, we convert the final fused point cloud $\mathcal { P } _ { T }$ into a watertight mesh for quantitative evaluation. We estimate globally consistent normals using FaCE [41] and reconstruct the mesh with NKSR [42]; details are provided in Appendix E.

## 4 Experiments

We evaluate the proposed active in-hand reconstruction framework on six graspable real-world objects with varied geometries. Section 4.1 describes the hardware setup and experimental protocol. Section 4.2 evaluates reconstruction accuracy. Section 4.3 compares AURORA with both non-active manipulation strategies and active view-planning baselines, evaluating reconstruction efficiency and action-selection quality. Finally, Sec. 4.4 analyzes the contributions and robustness of the key Ray-GPIS components through targeted ablations.

## 4.1 System Setup and Task Protocol

Our system uses a fixed Azure Kinect DK RGB-D camera for visual sensing and a Leap Hand [38] for in-hand manipulation. During execution, the hand rotates the object for 6 s; the planner then updates the reconstruction and replans the next rotation direction. This closed-loop process is repeated under a fixed 30 s interaction budget. Then, we evaluate both the online point-cloud reconstruction used for active planning and the final mesh extracted after exploration. The 6 s interval is determined by the low-level hand controller, which requires this time to complete the commanded rotation and stabilize the grasp, rather than by planner computation. Detailed module runtimes are reported in Appendix F.2. For quantitative evaluation, we obtain ground-truth meshes using an EinScan Pro 2X high-resolution 3D scanner. More detailed hardware specifications and hyperparameters for the planner and reconstruction are provided in Appendix A.1 and F.1, respectively.

## 4.2 Reconstruction Accuracy

The first evaluation focuses on the reconstruction quality of the proposed AURORA framework. We evaluate reconstruction performance on six graspable real-world objects. For each, we report both the online reconstructed point cloud and the offline refined mesh obtained under a fixed manipulation budget of 30 s. Qualitative and quantitative results are shown in Fig. 4 and Tab. 1, respectively.

Reconstruction accuracy is evaluated using the F-score metric, where $F @ \tau$ denotes the harmonic mean of precision and recall within a distance tolerance τ (higher is better). As summarized in Tab. 1(a), AURORA achieves strong reconstruction performance under a fixed 30 s budget. The online point-cloud reconstruction already reaches high accuracy, with an average F@10 of 0.9671 and

![](images/927b3dded4062a98bed51f156811f905c46f1651ab51e0adfab79ab48b574013.jpg)  
Figure 4: Qualitative results on real-world objects. (a) Test objects. (b) Online point clouds. (c) Offline extracted meshes.

Table 1: Quantitative results after a fixed 30 s budget. F@τ is the harmonic mean of precision and recall at tolerance τ. Single-view meshes are evaluated only with Mesh–Mesh scores after post-hoc scale alignment to the ground-truth meshes.  
(a) Results of our active strategy across objects.
<table><tr><td rowspan="2">Obj.</td><td colspan="2">PCD-PCD (Online)</td><td colspan="2">Mesh–Mesh (Offline)</td></tr><tr><td>F@2↑ F@5↑</td><td>F@10↑</td><td>F@2↑</td><td>F@5↑ F@10↑</td></tr><tr><td>Cube</td><td>0.2895 0.9337</td><td>0.9977</td><td>0.6481</td><td>0.9557 0.9957</td></tr><tr><td>Corner Block</td><td>0.23530.7354</td><td>0.9303</td><td>0.5298</td><td>0.8559 0.9488</td></tr><tr><td>L-shaped Block 0.2306 0.8366</td><td></td><td>0.9886</td><td>0.4674</td><td>0.8367 0.9450</td></tr><tr><td>Pepper</td><td>0.17700.8321</td><td>0.9773</td><td>0.5480</td><td>0.8890 0.9913</td></tr><tr><td>Cylinder</td><td>0.44280.8400</td><td>0.9425</td><td>0.4035</td><td>0.8320 0.9372</td></tr><tr><td>Cross Block</td><td>0.14540.7718</td><td>0.9664</td><td>0.4601</td><td>0.8050 0.9850</td></tr><tr><td>Mean</td><td>0.25340.8249</td><td>0.9671</td><td>0.5095</td><td>0.8624 0.9672</td></tr><tr><td>Std.</td><td>0.1054 0.0679</td><td>0.0263</td><td>0.0855 0.0535</td><td>0.0262</td></tr></table>

(b) Mean comparison with baselines.
<table><tr><td rowspan="2">Method</td><td colspan="2">Mean PCD-PCD</td><td colspan="2">Mean Mesh-Mesh</td></tr><tr><td>F@2↑</td><td>F@5↑</td><td>F@10↑ F@2↑</td><td>F@5↑ F@10↑</td></tr><tr><td>x-axis</td><td>0.1735</td><td>0.6625 0.8655</td><td>0.3595</td><td>0.6211 0.8440</td></tr><tr><td>y-axis</td><td>0.1463</td><td>0.6074 0.7933</td><td>0.3082</td><td>0.5643 0.7498</td></tr><tr><td>z-axis</td><td>0.2079</td><td>0.7786 0.9228</td><td>0.4414</td><td>0.7231 0.8872</td></tr><tr><td>Fixed schedule</td><td>0.1550</td><td>0.7015 0.8995</td><td>0.4544</td><td>0.6948 0.8904</td></tr><tr><td>TRELLIS.2 [43]</td><td></td><td></td><td>0.6538</td><td>0.8315 0.9017</td></tr><tr><td>SPAR3D [44]</td><td></td><td></td><td>0.5571</td><td>0.7386 0.9483</td></tr><tr><td>Ours (Active)</td><td>0.2534 0.8249</td><td>0.9671</td><td>0.5095</td><td>0.8624 0.9672</td></tr></table>

F@5 of 0.8249, showing that active reorientation can efficiently acquire informative observations. Offline mesh refinement further improves the reconstruction quality, increasing the mesh-level F@5 to 0.8624 while preserving a high F@10 of 0.9672. As shown in the Tab. 1(b), AURORA also con sistently outperforms non-active rotation baselines, including single-axis rotations along the x-, y-, and z-axes and a predefined fixed rotation schedule. This is further supported by the qualitative temporal comparison in Fig. 5(a), where AURORA reconstructs more complete geometry with fewer missing regions and faster surface coverage. These results indicate that uncertainty-driven active reorientation improves both reconstruction efficiency and final surface completeness. More detailed qualitative comparisons with the x-, y-, and z-axis baselines are provided in Appendix F.4.

We further compare our method with recent single-view 3D reconstruction methods, TREL-LIS.2 [43] and SPAR3D [44], as reference baselines in Tab. 1(b). Although not designed for active in-hand reconstruction, these methods provide a useful reference for reconstruction from incomplete visual observations. For a well-defined comparison, we use the corresponding clean object image in Fig. 4(a) as input for each object. Because single-view methods do not recover metric scale, their raw outputs can have arbitrary sizes; we therefore align their mesh to the ground truth scale before computing F-scores. Thus, these scores reflect scale-normalized shape similarity, not directly usable metric reconstructions accuracy. Even after removing scale errors in their favor, our method still achieves higher mean F-scores at the 5 mm and 10 mm thresholds. Moreover, these baselines require clean, high-resolution inputs and may produce incomplete or non-watertight meshes under hand occlusion, limiting their suitability for metric-aware robotic applications. Detailed compar isons and qualitative results are provided in Appendix F.5.

![](images/b18b22a6f16814257adcd6635af62bad4f04d9247789a3e708f09fcaa7081c8a.jpg)

![](images/ec38b05f1f59fc05e2553b70574bcdaea1520ea6024df487c3eb587d5e530f02.jpg)  
Figure 5: (a) Qualitative comparison of the reconstruction result on Cross Block over time between our method and the Z-X-Y-Z-X baseline. (b) Online reconstruction efficiency under different rotation strategies, averaged over six objects. The curves show the $q _ { 9 5 }$ uncertainty and Mesh–Mesh F-score at the 5mm threshold within a 30 s budget.

## 4.3 Comparison with Active and Non-active Baselines

## 4.3.1 Comparison with Non-active Baselines

Analysis of Active Reconstruction Efficiency. We first evaluate online reconstruction efficiency by tracking the temporal evolution of both reconstruction uncertainty and geometric accuracy under the proposed closed-loop active strategy. We use $q _ { 9 5 }$ , the 95th percentile of the estimated uncertainty distribution, to indicate poorly observed, high-uncertainty regions (e.g., holes in the reconstruction). We also report the Mesh–Mesh F-score at 5 mm, denoted as F@5, to measure the quality of the intermediate reconstructed mesh against the scanned ground truth. As shown by the red curves in Fig. 5(b), our method reaches lower $q _ { 9 5 }$ and higher F@5 over time, indicating effective information acquisition and improved geometric reconstruction quality. The solid red $q _ { 9 5 }$ curve decreases stepwise: when exploration begins to plateau, each replanning step (every 6 s, marked by vertical lines) introduces further uncertainty reduction. The increasing F@5 curve further confirms that the reduced uncertainty translates into improved geometric reconstruction. After the final replanning step at $\mathrm { 2 4 s , }$ both curves gradually stabilize, suggesting that the most informative viewpoints have been sufficiently covered and additional observations yield diminishing returns.

Comparison with Open-loop Rotation Strategies. We further compare our method against openloop rotation schedules under the same fixed 30 s manipulation budget. The dashed F@5 curves in Fig. 5(b) show that our method improves reconstruction quality more efficiently. In particular, after the first replanning step at 6 s, our F@5 curve grows faster than those of the baselines, indicating that the actively selected rotations provide more informative observations. Single-axis rotations perform worst, with slower F-score improvement and higher remaining uncertainty, because they fail to expose surfaces outside their limited reachable sets. The predefined multi-axis schedule performs better but remains suboptimal because it cannot adapt to the current reconstruction state. Minor nonmonotonic fluctuations in the curves arise from the automatic fusion and mesh extraction pipeline, where denoising, point-cloud downsampling, and meshing can slightly change the reconstructed surface over time.

## 4.3.2 Comparison with Active View-Planning Baselines

To further evaluate active view-selection performance, we compare Ray-GPIS with ActNeRF [45] and PB-NBV [46], adapting both methods to our in-hand reorientation setting. We conduct 120 paired episodes in simulation with ground-truth object poses and exact RGB–depth registration to isolate planner performance from perception and tracking errors. We report F-AUC, final F@5, planner correlation (Corr.), and planning time. F-AUC measures reconstruction quality over the full exploration process, while Corr. measures the agreement between planner-estimated action scores and the actual geometric gains of executable actions.

Table 2: Active view-planning comparison and targeted ablations of Ray-GPIS. (a) Planner comparison over 120 paired simulation episodes. (b) Component ablations under targeted failure modes.  
(a) Active view-planning comparison  
(b) Targeted ablations
<table><tr><td>Metric</td><td>ActNeRF [45]</td><td>PB-NBV [46]</td><td>Ray-GPIS</td><td>Method</td><td>Score</td><td>∆</td></tr><tr><td>F-AUC↑</td><td> $0 . 8 7 { \pm } 0 . 0 1$ </td><td>0.75±0.02</td><td>0.90±0.01</td><td colspan="3">Finger occlusion: Unseen R-AUC↑</td></tr><tr><td>F@5↑</td><td> $0 . 9 7 { \pm } 0 . 0 1$ </td><td>0.85±0.02</td><td>0.98±0.01</td><td>Full Ray-GPIS</td><td>0.5715</td><td></td></tr><tr><td>Corr.↑</td><td> $0 . 1 6 { \pm } 0 . 0 3$ </td><td>-0.36±0.07</td><td>0.47±0.07</td><td>w/o miss-ray interp.</td><td>0.5498</td><td>-0.0217</td></tr><tr><td>Time (s)↓</td><td> $2 . 2 4 { \pm } 0 . 0 2$ </td><td>0.39±0.02</td><td>0.26±0.02</td><td colspan="3">Pose errors (6°/3 mm): Corr.↑</td></tr><tr><td></td><td></td><td></td><td></td><td>Full Ray-GPIS</td><td>0.5028</td><td></td></tr></table>

As shown in Table 2(a), Ray-GPIS achieves the highest F-AUC and final F@5, indicating more efficient exploration and better final reconstruction quality. More importantly, Ray-GPIS obtains a substantially higher Corr. than both adapted baselines, showing that its planner scores more reliably rank executable actions according to their actual geometric gains. Ray-GPIS also requires substantially less planning time than ActNeRF. These results show that the gains over open-loop strategies are not solely due to closed-loop replanning; the proposed Ray-GPIS planner also select more effective reconstruction actions than alternative active view-planning methods.

## 4.4 Ablation and Robustness Analysis

We further evaluate two key Ray-GPIS components through targeted ablations: (i) miss-ray interpolation, which estimates anchors for rays without surface hits, and (ii) receptive-field (RF) integration, which aggregates uncertainty over a local neighborhood. As shown in Table 2(b), removing missray interpolation reduces Unseen R-AUC under finger occlusion, indicating that interpolated anchors help the planner reason about directions whose surfaces are not directly observed. Removing RF integration causes a substantially larger degradation in Corr. under pose perturbations, showing that locally aggregating uncertainty makes the action ranking considerably less sensitive to small geometric misalignments. These results support the intended roles of the two components: missray interpolation improves reasoning over persistently unobserved regions, whereas RF integration stabilizes uncertainty estimation under pose errors.

## 5 Limitations

Our framework has three main limitations. First, reconstruction remains sensitive to 6D posetracking errors under weak visual texture or severe hand–object occlusion, motivating tactile or contact-aided tracking. Second, the fixed 6 s replanning interval could be adapted based on uncertainty or tracking confidence. Finally, occasional object slip or drops highlight the need for more robust control and sim-to-real transfer.

## 6 Conclusion

This paper presents AURORA, an active in-hand reconstruction framework that closes the loop between online object-centric reconstruction and uncertainty-driven reorientation. At its core, Ray-GPIS estimates direction-wise reconstruction uncertainty to guide executable in-hand rotation actions under severe hand–object occlusion. Together with CAD-free 6D pose tracking and lightweight RGB-D fusion, AURORA progressively exposes under-observed regions and improves reconstruction completeness. Experiments show that AURORA outperforms non-active rotation strategies in reconstruction quality and information-acquisition efficiency, while Ray-GPIS also surpasses adapted active view-planning baselines in reconstruction performance, action-ranking quality, and planning efficiency. Targeted ablations validate the roles of miss-ray interpolation and receptivefield integration under occlusion and pose errors, and pose-perturbation tests further demonstrate robustness to moderate tracking inaccuracies. Future work will focus on improving pose-tracking robustness and in-hand manipulation stability.

## Acknowledgments

We thank the Area Chair and the anonymous reviewers for their constructive feedback and helpful suggestions. This work was supported by the Natural Science Foundation of Shanghai under Grant 25ZR1402370 and in part by the MoE Key Laboratory of Intelligent Perception and Human– Machine Collaboration.

## References

[1] R. Bajcsy. Active perception. Proceedings of the IEEE, 76(8):966–1005, 1988.

[2] M. Krainin, P. Henry, X. Ren, and D. Fox. Manipulator and object tracking for in-hand 3d object modeling. The International Journal ofRobotics Research, 30(11):1311–1327, 2011.

[3] S. Suresh, H. Qi, T. Wu, T. Fan, L. Pineda, M. Lambeta, J. Malik, M. Kalakrishnan, R. Calandra, M. Kaess, et al. Neuralfeels with neural fields: Visuotactile perception for in-hand manipulation. Science Robotics, 9(96):eadl0628, 2024.

[4] N. Chavan-Dafle, R. Holladay, and A. Rodriguez. In-hand manipulation via motion cones. arXiv preprint arXiv:1810.00219, 2018.

[5] B. Liang, K. Ota, M. Tomizuka, and D. K. Jha. Robust in-hand manipulation with extrinsic contacts. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6544–6550. IEEE, 2024.

[6] H. Qi, A. Kumar, R. Calandra, Y. Ma, and J. Malik. In-hand object rotation via rapid motor adaptation. In Conference on Robot Learning, pages 1722–1732. PMLR, 2023.

[7] G. Khandate, S. Shang, E. T. Chang, T. L. Saidi, Y. Liu, S. M. Dennis, J. Adams, and M. Ciocarlie. Sampling-based exploration for reinforcement learning of dexterous manipulation. arXiv preprint arXiv:2303.03486, 2023.

[8] O. M. Andrychowicz, B. Baker, M. Chociej, R. Jozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, et al. Learning dexterous in-hand manipulation. The International Journal ofRobotics Research, 39(1):3–20, 2020.

[9] T. Chen, M. Tippur, S. Wu, V. Kumar, E. Adelson, and P. Agrawal. Visual dexterity: In-hand reorientation of novel and complex object shapes. Science Robotics, 8(84):eadc9244, 2023.

[10] Y. Qin, B. Huang, Z.-H. Yin, H. Su, and X. Wang. Dexpoint: Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation. In Conference on Robot Learning, pages 594–605. PMLR, 2023.

[11] Z.-H. Yin, B. Huang, Y. Qin, Q. Chen, and X. Wang. Rotating without seeing: Towards in-hand dexterity through touch. arXiv preprint arXiv:2303.10880, 2023.

[12] Y. Yuan, H. Che, Y. Qin, B. Huang, Z.-H. Yin, K.-W. Lee, Y. Wu, S.-C. Lim, and X. Wang. Robot synesthesia: In-hand manipulation with visuotactile sensing. In 2024 IEEE Interna tional Conference on Robotics and Automation (ICRA), pages 6558–6565. IEEE, 2024.

[13] X. Qin, F. Zhao, Y. Leng, R. Hu, and C. Xiao. Nlipscalib: An efficient calibration framework for high-fidelity 3d reconstruction of curved visuotactile sensors. arXiv preprint arXiv:2603.09319, 2026.

[14] J. L. Schonberger and J.-M. Frahm. Structure-from-motion revisited. In¨ Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[15] J. Lei, S. Sridhar, P. Guerrero, M. Sung, N. Mitra, and L. J. Guibas. Pix2surf: Learning parametric 3d surface models of objects from images. In European Conference on Computer Vision, pages 121–138. Springer, 2020.

[16] J. Munkberg, J. Hasselgren, T. Shen, J. Gao, W. Chen, A. Evans, T. Muller, and S. Fidler.¨ Extracting triangular 3d models, materials, and lighting from images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8280–8290, 2022.

[17] J. Sun, Y. Xie, L. Chen, X. Zhou, and H. Bao. Neuralrecon: Real-time coherent 3d reconstruction from monocular video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15598–15607, 2021.

[18] D. Azinovic, R. Martin-Brualla, D. B. Goldman, M. Nießner, and J. Thies. Neural rgb-d surface´ reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6290–6301, 2022.

[19] B. Wen and K. Bekris. Bundletrack: 6d pose tracking for novel objects without instance or category-level 3d models. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8067–8074. IEEE, 2021.

[20] B. Wen, W. Yang, J. Kautz, and S. Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17868–17879, 2024.

[21] T. Lee, B. Wen, M. Kang, G. Kang, I. S. Kweon, and K.-J. Yoon. Any6d: Model-free 6d pose estimation of novel objects. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11633–11643, 2025.

[22] M. Liu, S. Li, A. Chhatkuli, P. Truong, L. Van Gool, and F. Tombari. One2any: One-reference 6d pose estimation for any object. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6457–6467, 2025.

[23] S. Jiang, Q. Ye, R. Xie, Y. Huo, and J. Chen. Hand-held object reconstruction from rgb video with dynamic interaction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12220–12230, 2025.

[24] Y. Huang, O. Taheri, M. J. Black, and D. Tzionas. Intercap: Joint markerless 3d tracking of humans and objects in interaction. In DAGM German Conference on Pattern Recognition, pages 281–299. Springer, 2022.

[25] Y. Jiang, S. Jiang, G. Sun, Z. Su, K. Guo, M. Wu, J. Yu, and L. Xu. Neuralhofusion: Neural volumetric rendering under human-object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6155–6165, 2022.

[26] Z. Cao, I. Radosavovic, A. Kanazawa, and J. Malik. Reconstructing hand-object interactions in the wild. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12417–12426, 2021.

[27] C. Cao, J. Zhang, M. Travers, and H. Choset. Hierarchical coverage path planning in complex 3d environments. In 2020 IEEE International Conference on Robotics and Automation (ICRA), pages 3206–3212. IEEE, 2020.

[28] C. Feng, H. Li, M. Zhang, X. Chen, B. Zhou, and S. Shen. Fc-planner: A skeleton-guided planning framework for fast aerial coverage of complex 3d scenes. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 8686–8692. IEEE, 2024.

[29] B. Zhou, Y. Zhang, X. Chen, and S. Shen. Fuel: Fast uav exploration using incremental frontier structure and hierarchical planning. IEEE Robotics and Automation Letters, 6(2):779–786, 2021.

[30] F. Zhao and C. Xiao. Autonomous exploration for shape reconstruction and measurement via informative contact-guided planning. IEEE Robotics and Automation Letters, 11(2):1250– 1257, 2025.

[31] C. Xiao, S. Xu, W. Wu, and J. Wachs. Active multiobject exploration and recognition via tactile whiskers. IEEE Transactions on Robotics, 38(6):3479–3497, 2022.

[32] C. Stachniss, G. Grisetti, and W. Burgard. Information gain-based exploration using raoblackwellized particle filters. In Robotics: Science and systems, volume 2, pages 65–72, 2005.

[33] B. Yamauchi. A frontier-based approach for autonomous exploration. In Proceedings 1997 IEEE International Symposium on Computational Intelligence in Robotics and Automation CIRA’97.’Towards New Computational Principles for Robotics and Automation’, pages 146– 151. IEEE, 1997.

[34] W. Jiang, B. Lei, and K. Daniilidis. Fisherrf: Active view selection and mapping with radiance fields using fisher information. In European Conference on Computer Vision, pages 422–440. Springer, 2024.

[35] Y. Xie, Y. Cai, Y. Zhang, L. Yang, and J. Pan. Gauss-mi: Gaussian splatting shannon mutual information for active 3d reconstruction. arXiv preprint arXiv:2504.21067, 2025.

[36] M. Krainin, B. Curless, and D. Fox. Autonomous generation of complete 3d object models using next best view manipulation planning. In 2011 IEEE international conference on robotics and automation, pages 5031–5037. IEEE, 2011.

[37] J. Bohg, K. Hausman, B. Sankaran, O. Brock, D. Kragic, S. Schaal, and G. S. Sukhatme. Interactive perception: Leveraging action in perception and perception in action. IEEE Transactions on Robotics, 33(6):1273–1291, 2017.

[38] K. Shaw, A. Agarwal, and D. Pathak. Leap hand: Low-cost, efficient, and anthropomorphic hand for robot learning. Robotics: Science and Systems (RSS), 2023.

[39] O. Williams and A. Fitzgibbon. Gaussian process implicit surfaces. In Gaussian Processes in Practice, 2006.

[40] N. Carion and L. G. et al. Sam 3: Segment anything with concepts, 2025.

[41] D. Scrivener, D. Cui, E. Coldren, S. M. Abulnaga, M. Bessmeltsev, and E. Chien. Faraday cage estimation of normals for point clouds and ribbon sketches. ACM Trans. Graph., 44(4), 2025. ISSN 0730-0301. doi:10.1145/3731212.

[42] J. Huang, Z. Gojcic, M. Atzmon, O. Litany, S. Fidler, and F. Williams. Neural kernel surface reconstruction. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4369–4379, 2023.

[43] J. Xiang, X. Chen, S. Xu, R. Wang, Z. Lv, Y. Deng, H. Zhu, Y. Dong, H. Zhao, N. J. Yuan, and J. Yang. Native and compact structured latents for 3d generation. Tech report, 2025.

[44] Z. Huang, M. Boss, A. Vasishta, J. M. Rehg, and V. Jampani. Spar3d: Stable point-aware reconstruction of 3d objects from single images. 2025.

[45] S. Dasgupta et al. Uncertainty-aware active learning of NeRF-based object models for robot manipulators using visual and re-orientation actions. arXiv:2404.01812, 2024.

[46] Z. Jia et al. PB-NBV: Efficient projection-based next-best-view planning framework for reconstruction of unknown objects. IEEE Robotics and Automation Letters, 2025.

[47] M. Gualtieri and R. Platt. Robotic pick-and-place with uncertain object instance segmentation and shape completion. IEEE Robotics and Automation Letters, 6(2):1753–1760, 2021.

## Appendix

## A Details of In-Hand Reorientation Policy

## A.1 Hardware Setup

This section summarizes the hardware used for policy training, real-world deployment, tactile sensing, and ground-truth mesh acquisition. Our hardware setup consists of an Azure Kinect DK RGB-D camera for visual sensing and a Leap Hand [38] for in-hand manipulation. Since the manipulator i equipped with tactile sensors, the tactile sensor configuration is detailed in Sec. A.3. Policy training is conducted on a workstation equipped with four NVIDIA A40 GPUs (Sec. A.2), while real-world deployment runs on a desktop system with an Intel Core i7-13700 CPU, 32 GB of RAM, and an NVIDIA RTX 4090D GPU with 24 GB of memory. For metric computation in the main manuscript, ground-truth object meshes are captured using an EinScan Pro 2X V2 scanner.

## A.2 Policy Deployment on the Leap Hand

First, our system requires object to be actively manipulated. For achieving in-hand manipulation, we use the tactile-based in-hand rotation policy from Rotating without Seeing [11] as the low-level reorientation controller and adapt it to the Leap Hand [38]. Since the original policy was designed for a different hand morphology (Allegro Hand) and a different tactile layout, we modify both the simulation environment and the policy interface to match our hardware.

Specifically, we replace the original hand model with the Leap Hand model and update the joint limits, actuator parameters, and fingertip geometry. To improve contact coverage during rotation, we extend the distal fingertip links, as shown in Fig. A.1. We also express commanded rotation axes in the hand-centered world frame W, consistent with the main paper. The policy takes an axis command $\mathbf { a } \in { \mathcal { A } }$ as input and outputs joint-level motor commands to rotate the grasped object about the commanded axis.

Before deployment, we retrain the policy in simulation to match the Leap Hand morphology and our tactile sensing setup. The reward is

$$
R = w _ { \mathrm { r o t } } R _ { \mathrm { r o t } } + w _ { \mathrm { c o n t a c t } } R _ { \mathrm { c o n t a c t } } + w _ { \mathrm { s t a b l e } } R _ { \mathrm { s t a b l e } } - w _ { \mathrm { a c t } } R _ { \mathrm { a c t } } - w _ { \mathrm { s l i p } } R _ { \mathrm { s l i p } } - w _ { \mathrm { d r o p } } R _ { \mathrm { d r o p } } .\tag{A.1}
$$

Here, $R _ { \mathrm { r o t } }$ encourages angular velocity about the commanded axis, $R _ { \mathrm { { c o n t a c t } } }$ encourages sustained tactile contact, and $R _ { \mathrm { s t a b l e } }$ promotes stable object retention. The penalty terms $R _ { \mathrm { a c t } } , R _ { \mathrm { s l i p } }$ , and $R _ { \mathrm { d r o p } }$ discourage excessive actions, object slip, and dropping, respectively. This reward adaptation encourages the policy to produce reliable axis-conditioned rotations rather than merely maximizing rotation speed.

## A.3 FSR Sensor Layout and Usage

To provide tactile feedback needed by in-hand manipulation, we mount force-sensitive resistor (FSR) sensors across the inner surfaces of Leap Hand fingertips, PIP/DIP phalanges, and palm, as illustrated in Fig. A.1. These regions correspond to the primary contact areas during in-hand rotation. FSR signals are asynchronously updated at approximately 50 Hz in a background thread, providing the policy with the most recent contact state at each execution step.

The FSR signals are used only by the low-level in-hand reorientation policy. Although they are not used by the reconstruction pipeline, they benefit reconstruction indirectly by enabling more reliable object reorientation.

![](images/b5111ff43ef62962c23c3e25196353a3ba0c73f6d2c7b5d05d55746eb50deb29.jpg)  
Figure A.1: FSR sensor layout on the Leap Hand. Sensors are embedded across the fingertips, distal and middle phalanges, and the palm to provide tactile feedback for the in-hand rotation policy.

## A.4 FSR Sensor Activation and Utilization

![](images/bd8b5649e65f951f7fd415d410cc3fc61ab00b8da3e4272d7cae3bb3282d9db2.jpg)  
Figure A.2: Temporal activation of the 16-channel FSR tactile signals in simulation and realhardware deployment.

To better understand how tactile feedback is used by the in-hand rotation policy, we visualize the temporal activations of the 16-channel FSR tactile signals in both simulation and real-hardware deployment. Each curve corresponds to one FSR channel, and a pulse indicates that the corresponding tactile sensor is activated by contact during manipulation.

Fig. A.2 shows that the simulated policy produces rich and distributed tactile activations across multiple fingers and palm sensors, indicating that the learned policy actively exploits contact information from different regions of the hand during object rotation. In real-hardware deployment, the activations become sparser and are concentrated on several dominant contact channels. This discrepancy is mainly caused by the physical characteristics of the real tactile sensors, including the minimum force threshold required for activation, hysteresis during activation and deactivation, and the limited stiffness of the manipulated object, which differs from the simulation setting. Despite these differences, the real system can still reliably obtain consistent tactile feedback from key contact regions, leading to successful manipulation. These observations also suggest that further improvements in sensor placement, calibration, and tactile signal normalization may be a possible way to increase the number of informative channels and improve the stability of real-world in-hand manipulation.

## B Details of Ray-GPIS Uncertainty Estimation

This section provides the mathematical details omitted from Sec. 3.2, including hit/miss ray assignment, anchor interpolation, and GPIS posterior variance.

## B.1 Hit/Miss Ray Assignment and Anchor Interpolation

Given the current fused point cloud $\mathcal { P } _ { t }$ , Ray-GPIS samples a set of candidate viewing directions $\{ { \mathbf { u } } _ { i } \} _ { i = } ^ { N }$ and casts rays from the object center c:

$$
\mathbf { x } ( t ; \mathbf { u } _ { i } ) = \mathbf { c } + ( t r ) \mathbf { u } _ { i } , \qquad t \in [ 0 , t _ { \operatorname* { m a x } } ] ,\tag{A.2}
$$

where $r$ normalizes the ray extent. In our implementation, each ray is discretized into a set of depth samples $\{ t _ { i j } \} _ { j = 1 } ^ { N _ { t } }$ . We compute the minimum distance from each ray to the observed point cloud:

$$
d _ { \operatorname* { m i n } } ( \mathbf { u } _ { i } ) = \operatorname* { m i n } _ { j } \operatorname* { m i n } _ { \mathbf { p } \in \mathcal { P } _ { t } } \left\| \mathbf { x } ( t _ { i j } ; \mathbf { u } _ { i } ) - \mathbf { p } \right\| _ { 2 } .\tag{A.3}
$$

A ray is treated as a hit ray if this distance is below a threshold $\tau _ { \mathrm { h i t } }$ :

$$
m _ { i } = \mathbb { I } [ d _ { \mathrm { m i n } } ( \mathbf { u } _ { i } ) < \tau _ { \mathrm { h i t } } ] .\tag{A.4}
$$

For a hit ray, the anchor depth $t _ { 0 } ( { \mathbf { u } } _ { i } )$ is defined as the first depth at which the ray enters the observed surface neighborhood:

$$
t _ { 0 } ( \mathbf { u } _ { i } ) = \operatorname* { m i n } \left\{ t _ { i j } \left| \operatorname* { m i n } _ { \mathbf { p } \in \mathcal { P } _ { t } } \left\| \mathbf { x } ( t _ { i j } ; \mathbf { u } _ { i } ) - \mathbf { p } \right\| _ { 2 } < \tau _ { \mathrm { h i t } } \right\} . \right.\tag{A.5}
$$

The corresponding anchor point is

$$
\mathbf { x } _ { 0 } ( \mathbf { u } _ { i } ) = \mathbf { x } ( t _ { 0 } ( \mathbf { u } _ { i } ) ; \mathbf { u } _ { i } ) .\tag{A.6}
$$

For a miss ray, no reliable observed surface intersection is available. We therefore infer a plausible anchor depth by interpolating the depths of nearby hit-rays on the viewing sphere. Let $\mathcal { H } _ { i }$ denote the set of nearby hit directions:

$$
\mathcal { H } _ { i } = \left\{ j \big | m _ { j } = 1 , \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { j } \geq \gamma \right\} ,\tag{A.7}
$$

where $\gamma$ is an angular similarity threshold. We compute cosine-based interpolation weights:

$$
\alpha _ { i j } = \frac { \exp \bigl ( ( \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { j } - 1 ) / { \tau _ { \mathrm { i n t e r p } } } \bigr ) } { \sum _ { \ell \in \mathcal { H } _ { i } } \exp \bigl ( ( \mathbf { u } _ { i } ^ { \top } \mathbf { u } _ { \ell } - 1 ) / { \tau _ { \mathrm { i n t e r p } } } \bigr ) } , \qquad j \in \mathcal { H } _ { i } ,\tag{A.8}
$$

where $\tau _ { \mathrm { i n t e r p } }$ controls the angular interpolation bandwidth. The miss-ray anchor depth is then estimated as

$$
t _ { 0 } ( \mathbf { u } _ { i } ) = \sum _ { j \in \mathcal { H } _ { i } } \alpha _ { i j } t _ { 0 } ( \mathbf { u } _ { j } ) .\tag{A.9}
$$

If no valid neighboring hit ray exists, the ray is excluded from the current planning update. This anchor interpolation allows Ray-GPIS to evaluate uncertainty near plausible surface locations, even for directions without direct observations.

## B.2 GPIS Posterior Variance

Ray-GPIS uses a Gaussian Process (GP) model to estimate spatial reconstruction uncertainty. Given the anchor set

$$
{ \mathbf { X } } = \{ { \mathbf { x } } _ { 0 } ( { \mathbf { u } } _ { i } ) \} _ { i = 1 } ^ { N _ { X } } ,\tag{A.10}
$$

and the corresponding labels y, we model the implicit function as

$$
f ( \mathbf { x } ) \sim \mathcal { G P } \left( \mu _ { 0 } ( \mathbf { x } ) , k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \right) ,\tag{A.11}
$$

where $\mu _ { 0 } ( \cdot )$ is the prior mean and $k ( \cdot , \cdot )$ is the covariance kernel.

For a query point x, the posterior distribution is

$$
p ( f ( \mathbf { x } ) \mid \mathbf { X } , \mathbf { y } ) \sim { \mathcal { N } } \left( \mu ( \mathbf { x } ) , \sigma ^ { 2 } ( \mathbf { x } ) \right) .\tag{A.12}
$$

The predictive mean and variance are

$$
\mu ( \mathbf { x } ) = \mu _ { 0 } ( \mathbf { x } ) + \mathbf { k } _ { \mathbf { x } } ^ { \top } \left( \mathbf { K } + \sigma _ { n } ^ { 2 } \mathbf { I } \right) ^ { - 1 } \left( \mathbf { y } - \pmb { \mu } _ { 0 } \right) ,\tag{A.13}
$$

$$
\sigma ^ { 2 } ( \mathbf { x } ) = k ( \mathbf { x } , \mathbf { x } ) - \mathbf { k } _ { \mathbf { x } } ^ { \top } \left( \mathbf { K } + \sigma _ { n } ^ { 2 } \mathbf { I } \right) ^ { - 1 } \mathbf { k } _ { \mathbf { x } } ,\tag{A.14}
$$

where ${ \bf K } = k ( { \bf X } , { \bf X } ) , { \bf k _ { x } } = k ( { \bf X } , { \bf x } ) , \mu _ { 0 } = \mu _ { 0 } ( { \bf X } )$ , and $\sigma _ { n } ^ { 2 }$ is the observation noise variance.

In AURORA, we use the posterior variance $\sigma ^ { 2 } ( \mathbf { x } )$ as a geometric reconstruction uncertainty metric. This uncertainty is high in regions less covered by the current observations and low near wellobserved anchors.

## C Details of NBV-to-Action Mapping

This section details how the selected NBV direction is mapped to a feasible in-hand rotation primitive. The NBV module outputs a desired viewing direction $\mathbf { u } _ { \mathrm { N B V } } ^ { \star }$ in the object-centered planning frame. Since the low-level reorientation policy executes rotation-axis commands in the handcentered world frame W, we first transform the NBV direction into W.

Let $T _ { C O } ( t ) \in S E ( 3 )$ denote the tracked object pose in the camera frame, and let $T _ { W C } \in S E { ( 3 ) }$ denote the calibrated camera-to-world transform. The object pose in the hand-centered world frame is

$$
T _ { W O } ( t ) \triangleq T _ { W C } T _ { C O } ( t ) ,\tag{A.15}
$$

with rotation component ${ \bf R } _ { W O } ( t )$ . The desired viewing direction in W is then given by

$$
\mathbf { d } _ { W } = \frac { \mathbf { R } _ { W O } ( t ) \mathbf { u } _ { \mathrm { N B V } } ^ { \star } } { \| \mathbf { R } _ { W O } ( t ) \mathbf { u } _ { \mathrm { N B V } } ^ { \star } \| _ { 2 } } .\tag{A.16}
$$

We define $\mathbf { v } _ { W }$ as the unit vector from the current object center to the camera, expressed in W. We compute the minimal rotation that aligns d with $\mathbf { v } _ { W }$ . The rotation angle is

$$
\boldsymbol \theta = \operatorname { a r c c o s } \left( \operatorname { c l i p } \left( \mathbf { v } _ { W } ^ { \top } \mathbf { d } _ { W } , - 1 , 1 \right) \right) ,\tag{A.17}
$$

where clipping improves numerical stability. When $\mathbf { v } _ { W }$ and ${ \bf d } _ { W }$ are not parallel, the rotation axis is

$$
\mathbf { k } = { \frac { \mathbf { d } _ { W } \times \mathbf { v } _ { W } } { \lVert \mathbf { d } _ { W } \times \mathbf { v } _ { W } \rVert _ { 2 } } } .\tag{A.18}
$$

The corresponding minimal alignment rotation is

$$
R _ { \mathrm { e r r } } = \exp \left( \theta [ \mathbf { k } ] _ { \times } \right) ,\tag{A.19}
$$

where $[ \mathbf { k } ] _ { \times }$ is the skew-symmetric matrix of k.

We represent this rotation as a Lie algebra vector:

$$
\omega = \mathrm { v e e } \left( \log ( R _ { \mathrm { e r r } } ) \right) = \theta \mathbf { k } \in \mathbb { R } ^ { 3 } ,\tag{A.20}
$$

where $\mathrm { v e e } ( \cdot )$ maps a skew-symmetric matrix to its vector representation. In the main paper, we use $\log ( R _ { \mathrm { e r r } } )$ to denote this vectorized rotation representation for compactness.

Finally, we project the desired rotation vector onto the discrete action set supported by the low-level in-hand rotation policy:

$$
\mathbf { a } ^ { \star } = \arg \operatorname* { m a x } _ { \mathbf { a } \in \mathcal { A } } \mathbf { a } ^ { \top } \boldsymbol { \omega } .\tag{A.21}
$$

This selects the feasible rotation primitive whose axis is most aligned with the desired viewpointcorrecting rotation.

For numerical robustness, when $\left\| \mathbf { d } _ { W } \times \mathbf { v } _ { W } \right\| _ { 2 } < \epsilon ,$ we treat the two directions as nearly parallel. If ${ \bf v } _ { W } ^ { \top } \mathbf d _ { W } > 0 .$ , we set $\omega = 0$ , indicating that no corrective rotation is required. If ${ \bf v } _ { W } ^ { \top } \mathbf d _ { W } < 0$ , we choose an arbitrary unit vector orthogonal to $\mathbf { v } _ { W }$ as k and set $\theta = \pi .$ . In practice, this degenerate case rarely occurs because the NBV directions are discretely sampled and the action space is coarse.

## D Details of 6D Pose Tracking and Keyframe Selection

This section provides additional details on the object pose tracking and keyframe selection strategy used in Sec. 3.3. AURORA uses BundleTrack [19] to estimate the object pose in the camera frame, $T _ { C O } ( t ) \in S E ( 3 )$ , from segmented RGB-D observations. BundleTrack maintains a pool of masked keyframes and jointly refines their poses through bundle adjustment. The resulting object poses allow depth observations from different viewpoints to be transformed into a consistent object-centered frame for fusion.

However, during in-hand manipulation, hand–object occlusions and imperfect segmentation can introduce unreliable frames. Directly inserting these frames into the keyframe pool may corrupt the fused geometry and destabilize the pose graph. We therefore apply a lightweight keyframe filter based on visibility, pose stability, and viewpoint change.

Let $t ^ { - }$ denote the most recent selected keyframe. For a candidate frame at time $t ,$ we define the relative motion with respect to $t ^ { - }$ as

$$
\Delta T ( t ) \triangleq T _ { C O } ( t ^ { - } ) ^ { - 1 } T _ { C O } ( t ) ,\tag{A.22}
$$

with rotation and translation components $\Delta \mathbf { R } ( t )$ and $\Delta \mathbf { t } ( t )$ , respectively. The relative rotation and translation magnitudes are

$$
\Delta \theta ( t ) = \cos ^ { - 1 } \left( \frac { \operatorname { t r } ( \Delta \mathbf { R } ( t ) ) - 1 } { 2 } \right) , \qquad \Delta p ( t ) = \| \Delta \mathbf { t } ( t ) \| _ { 2 } .\tag{A.23}
$$

We also compute the pose-update magnitude

$$
e ( t ) = \| \log ( \Delta T ( t ) ) \| _ { 2 } ,\tag{A.24}
$$

where $\log ( \cdot )$ denotes the Lie algebra mapping from $S E ( 3 )$ to its tangent space.

A frame is retained as a keyframe if it satisfies both reliability and informativeness criteria:

$$
\begin{array} { c } { { t \in \mathcal { K } \Longleftrightarrow \left( A _ { t } \geq \tau _ { A } \right) \wedge \left( e ( t ) \leq \tau _ { E } \right) } } \\ { { \wedge \operatorname* { m a x } \Biggl ( { \frac { \Delta \theta ( t ) } { \tau _ { R } } } , { \frac { \Delta p ( t ) } { \tau _ { T } } } \Biggr ) \geq 1 . } } \end{array}\tag{A.25}
$$

Here, $A _ { t }$ denotes the visible object area measured from the segmentation mask $\mathbf { M } _ { t }$ . The threshold $\tau _ { A }$ rejects frames with insufficient visible object pixels, while $\tau _ { E }$ filters out frames with unstable pose updates. $\tau _ { R }$ and $\tau _ { T }$ are the rotation and translation thresholds, ensuring that the selected frame provides sufficient viewpoint change relative to the most recent keyframe, either through rotation or translation. This prevents redundant frames from being inserted while preserving informative observations for multi-view fusion.

In our implementation, this filtering step is applied before the online reconstruction update. Only the selected keyframes that pass the filter $\kappa$ are used to update the fused point cloud $\mathcal { P } _ { t } .$ . This improves the stability of both point-cloud fusion and subsequent Ray-GPIS uncertainty estimation.

## E Details of Mesh Extraction

AURORA represents the online reconstruction as a lightweight point cloud $\mathcal { P } _ { t } ^ { } .$ , enabling low-latency uncertainty estimation and active planning. After exploration terminates, we optionally run an offline post-processing stage to obtain a watertight mesh for quantitative evaluation and visualization.

To perform mesh reconstruction, given the final fused point cloud $\mathcal { P } _ { T }$ , we first estimate globally consistent surface normals using FaCE [41]:

$$
\begin{array} { r } { \hat { \bf N } \triangleq \mathcal { F } ( \mathcal { P } _ { T } ) , } \end{array}\tag{A.26}
$$

where $\mathcal F ( \cdot )$ denotes the FaCE-based normal estimation operator and $\hat { \bf N }$ is the estimated normal field. We use FaCE because the online fused point cloud may contain non-uniform density, positional noise, and residual misalignment from pose-tracking errors. More consistent normals improve the quality of the subsequent mesh reconstruction.

We then reconstruct a watertight mesh using NKSR [42]:

$$
\mathcal { M } \triangleq \mathcal { R } _ { \mathrm { n k s r } } \left( \mathcal { P } _ { T } , \hat { \mathbf { N } } \right) ,\tag{A.27}
$$

where $\mathcal { R } _ { \mathrm { n k s r } } ( \cdot )$ denotes the NKSR reconstruction operator. The resulting mesh $\mathcal { M }$ is used only for final evaluation and qualitative visualization, and is not used by the online planner. During active exploration, Ray-GPIS operates directly on the online point cloud $\mathcal { P } _ { t } .$ , avoiding the latency of repeated mesh extraction.

## F Experimental Details

## F.1 Planner and Reconstruction Hyperparameters

The main parameters used in our experiments are listed in Tab. A.1.

Table A.1: Planner and reconstruction hyperparameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Visibility threshold  $\tau _ { A }$  Pose stability threshold  $\tau _ { E }$  Rotation threshold  $\tau _ { R }$  Translation threshold  $\tau _ { T }$  Ray hit threshold  $\tau _ { \mathrm { h i t } }$  Angular similarity threshold  $\gamma$  Angular interpolation bandwidth  $\tau _ { \mathrm { i n t e r p } }$  Novelty scale  $\beta$  Maximum ray extent  $s _ { \operatorname* { m a x } } / t _ { \operatorname* { m a x } }$ </td><td>0.25 0.03 10° 0.008 m 0.005 m 0.6 0.08 4.0 2.2 0.01</td></tr></table>

![](images/a2f12dbf86a1bcd045c5dcbf14fb3c82cc88814e428bb574d907dafaa77df79c.jpg)  
Figure A.3: Qualitative comparison of reconstruction results on Cross Block between AURORA and fixed-axis baselines.

## F.2 Runtime and Replanning Interval

We report the runtime of the main online modules on the deployment system described in Sec. A.1. Ray-GPIS GP update, candidate scoring, and NBV-to-action mapping require approximately 261 ms, 0.072 ms, and 2.42 ms per planning update, respectively. Segmentation and Bundle-Track together require approximately 256.9 ms per RGB-D frame. These results show that the planner itself is sufficiently lightweight for online closed-loop replanning.

The 6 s replanning interval is determined primarily by the low-level in-hand manipulation controller rather than planner computation. After receiving a rotation-axis command, the hand requires approximately 6 s to complete the commanded rotation and stabilize the grasp. Replanning earlier would therefore change the target before the previous reorientation has been reliably executed. We consequently update the high-level planner after each 6 s manipulation interval.

## F.3 Robustness to 6D Pose-Tracking Errors

To evaluate sensitivity to pose-tracking errors, we manually perturb the estimated object poses on real RGB-D sequences before reconstruction. Specifically, we consider rotational/translational perturbations of $5 ^ { \circ } / 5$ mm and 10<sup>◦</sup>/10 mm. Under these perturbations, AURORA retains 96.3% and 90.3%, respectively, of its unperturbed F@5 performance. These results indicate that moderate pose errors degrade reconstruction only gradually, rather than causing immediate failure of the reconstruction and planning pipeline.

## F.4 Additional Comparison with Open-loop Rotation Baselines

Fig. A.3 provides additional qualitative comparisons between AURORA and the open-loop rotation baselines introduced in Sec 4.3.1. Single-axis rotation strategies i.e., the x-, y-, or z-axis, provide only a limited set of viewpoints. As a result, some surface regions remain persistently occluded by the hand or are never brought into view, leading to incomplete reconstructions and visible missing areas. The predefined Z-X-Y-Z-X rotation schedule exposes more diverse viewpoints, but it remains open-loop and cannot adapt to the current reconstruction state. Therefore, some fine geometric details remain insufficiently recovered, especially in regions that require targeted viewpoint changes. In contrast, AURORA actively selects the next rotation axis based on the estimated reconstruction uncertainty, allowing it to target under-observed and high-uncertainty regions and produce more complete surfaces over time.

![](images/ce1ad5a31130153230d7dcee788acc9a37a98399af63de9e4f8d6e0a7cf00d4c.jpg)  
(a)

![](images/6ff275b6836d0f132d9498d9c5b5e2c4e5a74f6e474a266f4215e1d704c33a2a.jpg)  
(b)

![](images/3b75aade5bb58da02ff99a0965d20f6fa1fbe9bfb144d14951d06f02c4e618f5.jpg)  
(c)  
Figure A.4: Temporal evolution of Mesh–Mesh F-scores for AURORA and non-active baselines under different distance tolerances: (a) 2 mm, (b) 5 mm, and (c) 10 mm, averaged over six objects.

We further evaluate reconstruction efficiency using the temporal Mesh–Mesh F-score curves. At each timestamp, we reconstruct a mesh from the online point cloud fused up to that time using the same post-processing pipeline and compare it with the scanned ground-truth mesh. Fig. A.4 reports the averaged F-scores over six objects under three distance thresholds: 2 mm, 5 mm, and 10 mm.

Because AURORA always starts with a rotation about the z-axis; therefore, its F-score curve nearly overlaps with other baselines during the first 6 s. After 6 s, the open-loop baselines improve more slowly because repeated observations along the same or predefined axes provide diminishing information gain. Although switching axes according to a predefined schedule can improve reconstruction and achieves performance close to AURORA in some trials, it is less reliable overall. For instance, object slip during in-hand manipulation can change the actual object pose, preventing the predefined schedule from consistently exposing the intended unseen surfaces. In contrast, AURORA replans from the current reconstruction state and is therefore more robust on average, maintaining faster F-score improvement and reaching higher accuracy sooner.

The curves tend to saturate after approximately 18 s for two main reasons. First, for simple objects, most surfaces have already been observed by this time, leaving limited room for further F-score improvement. For more complex geometries, such as the Cross Block shown in Fig. A.3, AURORA can still fill missing mesh regions after 18 s. Second, in practice, the camera-to-object distance, sensor resolution, accumulated 6D tracking errors, and the downsampling and denoising operations used during point-cloud fusion all limit the mesh resolution and introduce small geometric deviations. Consequently, even our method indeed completes previously missing regions, the mesh extraction process can introduce slight distortions or misalignments elsewhere. Under fixed distance thresholds, these deviations can offset the F-score gain brought by the newly observed regions. Thus, this saturation does not indicate a failure of the active strategy.

We also note that the three single-axis baselines behave differently. This is caused not only by viewpoint coverage, but also by hardware constraints. Due to the sim-to-real gap in tactile feedback utility, contact dynamics, and friction, rotations around the x- and y-axes are less efficient and stable than the z-axis rotation and are more likely to induce object slip or unstable motions. Consequently, their reconstruction performance can be worse even under the same manipulation budget.

Table A.2: Mesh–Mesh reconstruction comparison with single-view 3D reconstruction baselines. F@τ denotes the F-score under distance threshold τ, where higher is better.
<table><tr><td rowspan="2">Object</td><td>TRELLIS.2 [43]</td><td></td><td>SPAR3D [44]</td><td></td><td>Ours</td><td></td></tr><tr><td>F@2 F@5</td><td>F@10</td><td>F@2 F@5</td><td>F@10</td><td>F@2 F@5</td><td>F@10</td></tr><tr><td>Cube</td><td>0.8922 0.9324</td><td>0.9579</td><td>0.4510 0.5515</td><td>0.8968</td><td>0.6481 0.9557</td><td>0.9957</td></tr><tr><td>Corner Block</td><td>0.5140 0.6447</td><td>0.7833</td><td>0.4507 0.5331</td><td>0.8686 0.5298</td><td>0.8559</td><td>0.9488</td></tr><tr><td>L-shaped Block</td><td>0.8124 0.9251</td><td>0.9632</td><td>0.5209 0.7244</td><td>0.9987 0.4674</td><td>0.8367</td><td>0.9450</td></tr><tr><td>Pepper</td><td>0.6612 0.9504</td><td>0.9978</td><td>0.6998 0.9459</td><td>1.0000</td><td>0.5480 0.8890</td><td>0.9913</td></tr><tr><td>Cylinder</td><td>0.5884 0.8861</td><td>0.9423</td><td>0.9030 0.9963</td><td>1.0000</td><td>0.4035 0.8320</td><td>0.9372</td></tr><tr><td>Cross Block</td><td>0.4548 0.6509</td><td>0.7660</td><td>0.3176 0.6809</td><td>0.9260</td><td>0.4601 0.8050</td><td>0.9850</td></tr><tr><td>Mean</td><td>0.6538 0.8316</td><td>0.9018</td><td>0.5572 0.7387</td><td>0.9484</td><td>0.5095 0.8624</td><td>0.9672</td></tr></table>

## F.5 Comparison with Single-View 3D Reconstruction Baselines

![](images/d86c0d26402e109c7272bbc6b38e1faf36b7ecb1ebafe71037b14da843f3501b.jpg)  
Figure A.5: Visual comparison of reconstruction failure modes for single-view baselines (TREL-LIS.2 [43] and SPAR3D [44]) versus our method on the Cube, Cross Block, and Corner Block. These baseline reconstructions are generated from the high-quality input images shown on the left.

To further analyze the limitations of single-view 3D reconstruction under in-hand occlusion, we compare AURORA with two recent single-image baselines, TRELLIS.2 [43] and SPAR3D [44]. Although these methods are not designed for active in-hand reconstruction, they provide useful reference baselines because they also reconstruct 3D geometry from incomplete visual observations. We evaluate all methods using the Mesh–Mesh F-score at 2 mm, 5 mm, and 10 mm thresholds. The quantitative results are reported in Tab. A.2.

As shown in Tab. A.2, our method achieves higher mean Mesh–Mesh F-scores than TRELLIS.2 [43] and SPAR3D [44] at the 5 mm and 10 mm thresholds across all evaluated objects. This suggests that actively acquiring additional observations improves metric reconstruction over single-view priors. Nevertheless, single-view methods produce relatively clean meshes, highlighting a direction for us to make future improvements.

We observe the following failure modes for the two single-view baselines in Fig. A.5. TREL-LIS.2 [43] preserves global structure and planar surfaces well, such as the rectangular faces of the Cube and Cross Block. However, it often fails to maintain topological completeness for objects with sharp edges or cutouts, producing open surfaces, incomplete cutting planes, and over-smoothed details. In contrast, SPAR3D [44] tends to generate more watertight meshes but with poorer geometric alignment. It often hallucinates rear-side geometry from the front view without enforcing global consistency, causing distortions such as non-orthogonal angles and warped surfaces, as seen in the Cube and Corner Block examples. These failures show that single-view methods struggle to infer coherent 3D geometry from limited observations, whereas incremental viewpoint acquisition is important for both completeness and geometric fidelity.

Table A.3: Downstream manipulation success (%), averaged over 10 objects with 30 trials per object and method.
<table><tr><td>Reconstruction source</td><td>Placement↑</td><td>Regrasp↑</td><td>Placement&amp;Regrasp↑</td></tr><tr><td>SPAR3D [44]</td><td>30.0</td><td>15.0</td><td>5.0</td></tr><tr><td>TRELLIS.2 [43]</td><td>19.7</td><td>7.7</td><td>0.0</td></tr><tr><td>Fixed schedule</td><td>50.0</td><td>44.0</td><td>24.0</td></tr><tr><td>Adapted PB-NBV [46]</td><td>43.3</td><td>40.0</td><td>23.0</td></tr><tr><td>Adapted ActNeRF [45]</td><td>56.0</td><td>72.0</td><td>41.7</td></tr><tr><td>Full Ray-GPIS</td><td>57.7</td><td>77.7</td><td>45.0</td></tr></table>

Both baselines also share three limitations in our in-hand setting. First, their performance depends strongly on input image quality. The results in Tab. A.2 use relatively clean, high-resolution object images without dexterous-hand occlusion, whereas realistic in-hand observations often suffer from partial visibility and motion blur. Second, their meshes do not preserve metric scale. We therefore rescale each generated mesh to the corresponding ground truth before computing Mesh–Mesh metrics. This scale ambiguity prevents direct use in metric-aware downstream robotic tasks. Third, they cannot recover true object texture from occluded or unobserved regions; any missing appearance must be hallucinated from learned priors. In contrast, AURORA actively reorients the object, incrementally fuses newly observed geometry in a consistent object-centric frame, and produces metrically consistent reconstructions.

## F.6 Downstream Manipulation Evaluation

Beyond geometric reconstruction metrics, we evaluate whether the reconstructed meshes provide useful geometry for downstream manipulation. Following [47], we test stable object placement followed by regrasp using meshes reconstructed by different methods. We compare two singleview reconstruction baselines, a non-active fixed rotation schedule, adapted active view-planning baselines, and the full Ray-GPIS pipeline. Each method is evaluated with 30 trials per object over 10 objects.

As shown in Tab. A.3, Full Ray-GPIS achieves the highest success rates for placement, regrasp, and the complete placement-and-regrasp task, reaching 57.7%, 77.7%, and 45.0%, respectively. It consistently outperforms the fixed schedule, the adapted active planners, and the single-view reconstruction baselines. These results show that the geometry recovered through uncertainty-driven active exploration is not only more complete according to reconstruction metrics, but also more useful for downstream placement and regrasping.

## F.7 Comparison with NeuralFeels & Discussions

NeuralFeels [3] is closely related to our work, as both methods reconstruct object geometry during in-hand manipulation. However, their task formulations differ, making a controlled direct comparison difficult. Specifically, NeuralFeels uses a perceptually open-loop in-hand rotation policy that focuses on rotating the object along a fixed axis and reconstructs geometry from both visual and tactile interaction sequences to compensate for occlusion. In contrast, our method focuses on closed-loop planning: it actively selects the next in-hand reorientation based on current reconstruc tion uncertainty to observe visually occluded regions, but does not use tactile sensing.

Due to differences in hardware, tactile sensing, and manipulation policies, we cannot directly reproduce NeuralFeels on our platform. We therefore conduct a limited comparison on the common Pepper object, for which NeuralFeels provides released recordings. Both methods are evaluated under a matched 30 s interaction budget using the Mesh–Mesh F-score. Results are shown in Tab. A.4, with additional metric curves in Fig. A.6. The NeuralFeels results are reproduced using the authors released code and data without customization; we plot corresponding F-scores for comparison.

AURORA achieves an F@5 score of 0.89, while NeuralFeels achieves an average score of 0.76 under the same 30 s interaction budget. This comparison is subject to differences in hardware, tactile sensing, manipulation policies, task focus, and object instances; therefore, the higher score should not be interpreted as evidence that AURORA is strictly superior to NeuralFeels. In particular, we acknowledge that NeuralFeels can track textureless objects, whereas our approach requires sufficient visual texture and may fail under weakly textured conditions. The observed performance gap may also be partly attributable to object appearance: our pepper object has a similar shape but richer texture than the one shown in the NeuralFeels video. Thus, this result does not necessarily imply superior reconstruction quality of our framework.

Table A.4: Limited comparison with NeuralFeels on the common Pepper object under a matched 30 s interaction budget. F@τ denotes the Mesh–Mesh F-score, where higher is better.
<table><tr><td>Method</td><td>F@5</td></tr><tr><td>NeuralFeels [3]</td><td>0.76</td></tr><tr><td>Ours</td><td>0.89</td></tr></table>

![](images/f80d2e6ae80bde000fd26278e542770722bb3f6f53df77485e59825b1824e372.jpg)  
Figure A.6: Quality metrics on Pepper, comparing AURORA with NeuralFeels [3].

Nevertheless, this case study suggests two potential practical benefits of our design in this setting. First, active uncertainty-guided reorientation improves reconstruction efficiency (refer to the slope rate of curves): our planner explicitly selects actions that expose under-observed regions, whereas NeuralFeels does not use reconstruction uncertainty to choose the next in-hand action. In addition, our multi-axis in-palm reorientation provides more diverse viewpoints than the fingertips’ twisting motions used in NeuralFeels, which may leave the bottom surface and other occluded regions insufficiently observed. Second, our pipeline is more computationally efficient. For the same 30 s observation sequence, NeuralFeels requires 417.13 s of wall-clock reconstruction time, whereas our pipeline takes 71.82 s, including mesh extraction. This efficiency comes mainly from using a lightweight point-cloud representation for online fusion and planning instead of optimizing a dense neural SDF.