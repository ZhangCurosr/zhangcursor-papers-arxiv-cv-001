# PIVOT: A Multi-Trajectory Dataset and Testbed for Pose, Intrinsics, and Novel Viewpoint Evaluation in Real-World 3D Reconstruction

Mary Raymond

Independent Researcher

mary.raymond.n@gmail.com

Abstract—Neural radiance fields (NeRFs), 3D Gaussian Splat ting (3DGS), and related novel-view synthesis methods are commonly evaluated under capture and reconstruction conditions that can be substantially cleaner than those encountered by robots, drones, and autonomous systems. In particular, benchmark pipelines may rely on reconstruction-friendly camera trajectories, offline-optimized camera poses, scene-optimized intrinsics, and held-out views sampled from trajectories already represented during training. These choices make evaluation convenient, but couple several favorable assumptions and can obscure how reconstruction systems behave when deployed with measured poses, reusable camera calibration, or structurally different camera paths.

We introduce PIVOT (Pose, Intrinsics and Viewpoint Oriented Testbed), a multi-trajectory dataset, processing pipeline, and evaluation framework for studying these factors independently. PIVOT represents each scene using deliberately different camera trajectories and retains, where available, both sensor-derived measured poses and COLMAP-optimized poses for each frame, together with physical camera calibration and scene-optimized intrinsics. The testbed defines three benchmark families: (1) seen versus unseen trajectory novel-view generalization, (2) measured versus optimized pose sensitivity, and (3) calibrated versus optimized intrinsic sensitivity. We additionally introduce a directed pose-space Chamfer distance for describing how well a training pose distribution covers an evaluation trajectory. PIVOT v1 contains five real-world scenes captured using a DJI Mini 4 Pro drone and provides an open processing and Nerfstudio-based evaluation toolchain intended to be reusable for additional scenes and capture devices.

This manuscript accompanies the initial PIVOT release. Results from the benchmarks show a consistent quality gap between heldout views on represented trajectories and views from unseen trajectories. They also show substantial sensitivity to pose source and to the camera intrinsics used for reconstruction.

## I. INTRODUCTION

Modern 3D reconstruction and novel-view synthesis systems are increasingly relevant to robotics, autonomous platforms, inspection, mapping, and embodied perception. Yet common experimental pipelines can implicitly assume access to conditions that are difficult to reproduce online. A scene may be captured using a smooth inward-looking orbit with high image overlap; camera poses may be recovered and globally optimized offline by Structure-from-Motion (SfM); camera intrinsics may be optimized independently for every scene; and evaluation may use held-out images sampled from the same trajectory family as the training data.

These assumptions are individually reasonable for reconstruction research, but together they can make the experimental setting substantially cleaner than the conditions encountered by a deployed system. A robot or drone may instead receive poses from GPS/IMU, visual(-inertial) odometry, SLAM [13], LiDAR, radar, or another online localization source. It may reuse a physical camera calibration across scenes. Its motion may follow traversals rather than reconstruction-friendly orbits. Most importantly, a requested novel view may lie on a camera path that is structurally different from the paths represented during training.

PIVOT is designed to make these differences explicit, measurable, and reproducible. Rather than treating a scene as one undifferentiated image collection, PIVOT makes the trajectory a first-class unit of capture, processing, visualization, export, and evaluation. Each scene contains multiple named and typed trajectories covering reconstruction-friendly, robot-like, and extrapolation-oriented motion. Processed frames can retain both a sensor-derived measured pose and a COLMAP-optimized pose, while trajectories can retain both physical/offline camera calibration and COLMAP-optimized intrinsics.

Figure 1 provides an overview of the trajectory-aware PIVOT representation and an example reconstruction produced from the processed scene.

The resulting testbed supports four central questions:

1) How well does a reconstruction model generalize to camera trajectories that are structurally different from its training trajectories?

2) How does reconstruction quality change as an evaluation trajectory moves farther from the training pose distribution?

3) How much reconstruction quality is gained by replacing measured poses with offline-optimized poses?

4) How much does per-scene intrinsic optimization improve over a reusable physical camera calibration?

The principal contributions of this work are:

• a reusable multi-trajectory scene capture specification and processed dataset representation;

• a dual-pose representation that stores sensor-derived measured poses and COLMAP-optimized poses side by side;

![](images/97b55730bb37f7929029dda38f0c6d499700dde2e7e3de07819b75370cf9aaf1.jpg)  
(a)

![](images/b1739f86456ed52999faf33b65140299feac5ecb2c243a650da33fb8ad2a96e2.jpg)  
(b)  
Fig. 1. Overview of the PIVOT testbed and reconstruction pipeline. (a) PIVOT’s interactive trajectory-aware viewer jointly visualizes the COLMAP sparse reconstruction, camera trajectories, trajectory-level pose errors, and scene statistics. (b) Example Splatfacto reconstruction of the same scene, trained using 80% of the frames from all trajectories with COLMAP-optimized poses and intrinsics. PIVOT preserves trajectory identity together with measured and optimized camera parameters to support controlled reconstruction evaluation.

• support for both calibrated and scene-optimized camera intrinsics;

• a directed pose-space Chamfer distance for quantifying evaluation-trajectory coverage relative to training poses;

• an end-to-end raw-data processing, visualization, export, and Nerfstudio integration toolchain; and

• three benchmark families for novel-view trajectory generalization, pose-source sensitivity, and intrinsic-source sensitivity.

PIVOT does not propose a new NeRF, 3DGS, or SfM algorithm. Its purpose is to provide a controlled testbed for studying reconstruction under more realistic capture and evaluation conditions.

## II. RELATED WORK

a) Novel-view synthesis and neural reconstruction.: NeRF and subsequent neural rendering methods established highquality novel-view synthesis from posed image collections [1]. More recent explicit scene representations, including 3D Gaussian Splatting, provide high-quality rendering with substantially different optimization and rendering characteristics [2]. PIVOT is model-agnostic at the dataset level; the initial benchmark integration targets Nerfacto and Splatfacto through Nerfstudio.

b) Camera pose estimation and Structure-from-Motion.: COLMAP provides a widely used SfM and multi-view geometry pipeline for camera registration and sparse reconstruction [3], [4]. Many reconstruction datasets and pipelines use SfMoptimized camera parameters as model inputs. PIVOT retains these optimized estimates while also preserving sensor-derived measured poses, enabling controlled experiments in which translation and rotation sources can be independently selected.

c) Reconstruction benchmarks and trajectory generaliza tion.: Real-world novel-view synthesis datasets span several capture regimes. LLFF introduced forward-facing real-world captures together with practical sampling guidance [5], while the NeRF synthetic and LLFF-style evaluations helped establish interpolation-oriented held-out-view protocols. Mip-NeRF 360 extended evaluation to challenging unbounded 360-degree scenes [6]. These datasets have been important for measuring rendering quality, but evaluation commonly samples test views from the same capture distribution used to construct the scene. PIVOT instead makes the trajectory an explicit experimental unit and includes complete, independently captured trajectories whose motion structure differs from the training paths. The goal is not to replace existing NVS benchmarks, but to complement them with a controlled way to distinguish within-trajectory interpolation from trajectory-level generalization.

d) Calibration and pose robustness.: Several neural reconstruction methods relax the assumption of perfectly known cameras by jointly optimizing scene representation and camera parameters. BARF jointly refines camera poses and a NeRF representation from imperfect initialization [7], while NeRF– jointly optimizes both camera intrinsics and poses [8]. Such methods demonstrate that camera uncertainty can be absorbed or corrected during offline optimization. PIVOT asks a complementary deployment-oriented question: what happens when reconstruction is intentionally evaluated using sensorderived poses or a reusable physical calibration rather than allowing scene-specific camera optimization?

e) Reconstruction frameworks.: Nerfstudio provides a modular framework for NeRF development and includes Nerfacto as a practical combination of established components [9]. PIVOT uses Nerfstudio as the initial benchmark backend and adds trajectory-aware export and evaluation so that the same processed scene can be tested under controlled pose, intrinsic, and viewpoint conditions.

## III. PIVOT TESTBED DESIGN

## A. Scene as a Collection of Trajectories

A PIVOT scene is represented as a collection of deliberately different camera trajectories rather than a single reconstructionfriendly path. The trajectory protocol records properties including motion type, altitude band, whether the path is closed, camera direction, lens type, image resolution, capture mode, and capture device.

The core trajectory families include inward-looking orbits at multiple altitudes, outward-looking orbit at low altitude, directional traversals, a closed traversal loop, bird’s-eye-view capture, vertical ascent, scattered still viewpoints, and 360- degree panorama stations. Optional trajectories extend the same design with outward-looking orbits, additional traversal altitudes, and additional scattered viewpoints.

This structure is intended to support both conventional reconstruction-friendly coverage and motion patterns that better resemble deployed robotic or aerial systems.

## B. Dual Pose Representation

For each processed frame, PIVOT can store:

T<sup>measured</sup><sub>c2w</sub> : sensor-derived measured camera-to-world pose, (1)

T<sup>COLMAP</sup> : COLMAP-optimized camera-to-world pose.

(2)

Measured poses are derived from capture-device metadata without scene-level pose optimization. For the DJI Mini 4 Pro drone capture path used in PIVOT v1, GPS position, flight attitude, and gimbal attitude are converted into a North-East-Down (NED) world frame and then into the OpenGL-style camera convention used by the processed dataset.

The COLMAP reconstruction uses measured positions as soft position priors. This allows the reconstructed model to benefit from SfM optimization while retaining a common spatial relationship with the measured trajectory.

The dual representation allows downstream experiments to independently select measured or optimized translation and rotation. We denote the four pose-source configurations as:

<table><tr><td>Configuration</td><td>Translation</td><td>Rotation</td></tr><tr><td>00</td><td>optimized</td><td>optimized</td></tr><tr><td>OM</td><td>optimized</td><td>measured</td></tr><tr><td>MO</td><td>measured</td><td>optimized</td></tr><tr><td>MM</td><td>measured</td><td>measured</td></tr></table>

## C. Dual Intrinsic Representation

A physical robotic system generally carries a camera whose calibration is reused across scenes, whereas an offline reconstruction pipeline may optimize camera intrinsics for each scene. PIVOT therefore retains both a physical/offline camera calibration and COLMAP-optimized per-scene intrinsics. The exporter can select the intrinsic source independently of the pose source.

## D. Processing Pipeline

The PIVOT raw-data pipeline illustrated in Figure 2, transforms trajectory captures into a processed scene while preserving trajectory identity. The main stages are:

1) read trajectory metadata and raw video/photo captures;

2) sample video frames using translation and rotation thresholds, or use captured still images directly;

3) extract and write EXIF/XMP metadata;

4) compute measured camera poses from device position and orientation metadata;

5) transform poses into the NED world frame and OpenGL camera convention;

6) run COLMAP feature extraction and matching;

7) inject measured camera positions and covariance as soft priors;

8) run COLMAP pose-prior mapping and select the best reconstruction;

9) retain optimized poses and intrinsics alongside measured poses and calibrated intrinsics;

10) compute per-frame pose errors and trajectory/scene statistics; and

11) compute the directed trajectory-distance matrix and export the processed scene.

The processing core exposes capture-device metadata interfaces so that devices other than the DJI Mini 4 Pro drone can be integrated by implementing the required metadata mapping and image/video pose readers.

## IV. DIRECTED POSE CHAMFER DISTANCE

To describe how well one set of camera poses covers another, PIVOT extends the Chamfer distance commonly used for comparing point sets [12] to a directed pose-space measure. Let A be an evaluation trajectory and B a reference or training pose set. The directed distance is

$$
D ( A \to B ) = { \frac { 1 } { | A | } } \sum _ { a \in A } { \mathrm { k N N D i s t a n c e } } ( a , B ) .\tag{3}
$$

The pose distance combines normalized translation and rotation components. Translation-only and rotation-only variants are also retained. Because the measure is directed,

$$
D ( A \to B ) \neq D ( B \to A ) ,\tag{4}
$$

which is intentional: the question “how well does the training pose set cover the evaluation trajectory?” is different from asking how well the evaluation trajectory covers the training set.

At scene-processing time, pairwise trajectory distances are stored as a matrix for visualization and experiment design. During reconstruction evaluation, the same formulation is used to measure each evaluation trajectory against the complete training pose set.

a) Interpretation.: The metric should be interpreted as a pose-space coverage descriptor, not as a direct measure of novel-view difficulty. The experiments in this work test whether increasing pose-space displacement is empirically associated with reconstruction-quality degradation.

## V. DATASET

## A. PIVOT v1

PIVOT v1 is built around five real-world scenes captured using a DJI Mini 4 Pro drone. Released processed scenes contain trajectory images, per-frame measured and COLMAPoptimized poses, calibrated and optimized camera intrinsics, trajectory statistics, scene statistics, and sparse reconstruction assets.

![](images/467ad37fb0f22ad44014c5ea2f03a8e49bb4093bf101646e300e7c96981b797d.jpg)  
Fig. 2. PIVOT raw-data processing pipeline.  
TABLE I

PIVOT V1 SCENE SUMMARY. FRONTYARD AND BACKYARD STATISTICSARE PENDING COMPLETION OF THE CURRENTLY RUNNINGPROCESSING/BENCHMARK JOBS AND WILL BE POPULATED BEFORERELEASE.
<table><tr><td>Scene</td><td></td><td>Frames Registered Reg.</td><td>(%)</td><td>Sparse pts.</td><td>AABB (m)</td><td>Reproj. (px)</td></tr><tr><td>Church</td><td>1,612</td><td>1,538</td><td>95.4</td><td>882,387</td><td>28.68</td><td>1.06</td></tr><tr><td>Village</td><td>1,733</td><td>1,726</td><td>99.5</td><td>1,430,217</td><td>55.12</td><td>0.89</td></tr><tr><td>Street Victorian Garden</td><td>1,547</td><td>1,536</td><td>99.2</td><td>767,781</td><td>44.62</td><td>0.92</td></tr><tr><td>Frontyard</td><td>920</td><td>913</td><td>99.2</td><td>719,123</td><td>13.16</td><td>1.09</td></tr><tr><td>Backyard</td><td>1,536</td><td>1,527</td><td>99.4</td><td>110,1253</td><td>25.3</td><td>0.97</td></tr></table>

Some PIVOT trajectories are intentionally difficult for SfM. Consequently, the dataset records both total frames and COLMAP-registered frames. Registration rate is treated as useful information about the interaction between trajectory design and SfM rather than merely as a preprocessing detail.

## B. Trajectory Taxonomy

## VI. EXPERIMENTAL PROTOCOL

The PIVOT evaluation uses the separate PIVOT Nerfstudio integration environment and targets Nerfacto and Splatfacto. Evaluation is performed per trajectory and reports SSIM [11], PSNR, LPIPS [10], and directed pose Chamfer distance relative to the training pose set.

TABLE II  
REPRESENTATIVE CORE TRAJECTORY TYPES IN PIVOT.
<table><tr><td>Trajectory</td><td>Motion</td><td>Altitude</td><td>Camera direction</td></tr><tr><td>orbit_inward_low</td><td>orbit</td><td>low</td><td>scene inward</td></tr><tr><td>orbit_inward_mid</td><td>orbit</td><td>mid</td><td>scene inward</td></tr><tr><td>orbit_inward_high</td><td>orbit</td><td>high</td><td>scene inward</td></tr><tr><td>orbit_outward_low</td><td>orbit</td><td>low</td><td>scene outward</td></tr><tr><td>traversal_forward_low</td><td>traversal</td><td>low</td><td>along track</td></tr><tr><td>traversal_backward_low</td><td>traversal</td><td>low</td><td>along track</td></tr><tr><td>traversal_left_low</td><td>traversal</td><td>low</td><td>along track</td></tr><tr><td>traversal_right_low</td><td>traversal</td><td>low</td><td>along track</td></tr><tr><td>traverse_loop_low</td><td>closed traversal</td><td>low</td><td>along track</td></tr><tr><td>bev_orbit_area</td><td>BEV orbit</td><td>high</td><td>nadir</td></tr><tr><td>bev_traverse_area</td><td>BEV traverse</td><td>high</td><td>nadir</td></tr><tr><td>rocket_upward</td><td>vertical ascent</td><td>low-high</td><td>scene inward</td></tr><tr><td>scattered_low</td><td>scattered</td><td>low</td><td>multi-angle</td></tr><tr><td>panorama_360_station</td><td>panorama</td><td>low-mid</td><td>360°sweep</td></tr></table>

For reproducibility, the release will freeze the PIVOT and Nerfstudio-integration commits together with the model configurations, training iteration counts, random seeds, image resolution/scaling settings, and exact train/evaluation trajec tory selections used for all reported runs. These values are recorded from the executed benchmark configuration rather than reconstructed after the fact.

## A. Benchmark 1: Seen vs. Unseen Trajectories

This benchmark asks how reconstruction quality changes when evaluation moves from held-out frames on trajectories represented during training to complete camera trajectories not represented during training.

Training uses a mixture of inward-orbit and traversal trajectories. Evaluation is divided into:

• seen trajectories: held-out frames from trajectories represented in training;

• unseen trajectories: complete trajectories absent from the training set.

Per-trajectory image-quality metrics are reported together with directed pose Chamfer distance to the training pose set. The number of training iterations for benchmark 1 is 60k

## B. Benchmark 2: Measured vs. Optimized Poses

This benchmark asks how strongly reconstruction quality depends on offline pose optimization. Translation and rotation are independently selected from measured or optimized estimates, producing OO, OM, MO, and MM conditions. This separation is intended to reveal whether translation error, rotation error, or their combination dominates reconstruction degradation. The number of training iterations for benchmark 2 is 30k

## C. Benchmark 3: Calibrated vs. Optimized Intrinsics

This benchmark asks how much benefit is obtained by allowing COLMAP to optimize camera intrinsics for a scene rather than using the camera’s precomputed physical calibration. Pose source is held fixed while the intrinsic source changes. The number of training iterations for benchmark 3 is 30k

## VII. QUANTITATIVE RESULTS

## A. Novel-View Trajectory Generalization

Table III summarizes BM1 reconstruction quality for seen and unseen evaluation trajectories. The table reports aggregate SSIM, LPIPS, and PSNR for Nerfacto and Splatfacto.

## B. Trajectory Distance and Reconstruction Quality

Figure 3 shows the relationship between directed normalized pose Chamfer distance to the training pose set and LPIPS. Seen trajectories are held-out views from trajectories represented during training, while unseen trajectories are independently captured evaluation trajectories.

## C. Measured vs. Optimized Poses

Table IV reports BM2 results for the four translation/rotation source combinations. OO uses optimized translation and rotation, OM uses optimized translation and measured rotation, MO uses measured translation and optimized rotation, and MM uses measured translation and rotation.

## D. Calibrated vs. Optimized Intrinsics

Table V compares scene-optimized COLMAP intrinsics with the fixed OpenCV calibration used by the capture device. Pose source is held fixed at the COLMAP optimized pose while the intrinsic source changes.

## VIII. QUALITATIVE RESULTS

The following qualitative comparisons use representative views from two scenes. Each panel uses the same ground-truth view across compared reconstruction conditions so that changes in rendering quality can be inspected directly.

## A. BM1: Seen vs. Unseen Trajectories

Figure 4 compares representative seen and unseen views for Nerfacto and Splatfacto.

## B. BM2: Measured vs. Optimized Poses

Figure 5 compares the four measured/optimized pose-source combinations for representative views.

## C. BM3: Calibrated vs. Optimized Intrinsics

Figure 6 compares COLMAP-optimized scene intrinsics with the fixed OpenCV camera calibration.

## IX. DISCUSSION

## A. Trajectory-Level Evaluation

The central motivation of PIVOT is that held-out frames from a trajectory represented during training and views from an entirely different trajectory answer different evaluation questions. The former primarily probes interpolation within a sampled capture distribution; the latter probes how reconstruction behaves when the requested viewpoints depart structurally from that distribution.

Across all the five scenes, unseen trajectories degrade relative to seen held-out views for both evaluated models on all three reported image-quality metrics. For Nerfacto, mean PSNR decreases from 19.54 to 16.47 dB in Church, 17.01 to 15.15 dB in Victorian Garden, and 19.80 to 16.82 dB in Village Street. For Splatfacto, the corresponding decreases are 22.17 to 16.86 dB, 19.23 to 15.00 dB, and 24.50 to 16.08 dB. The magnitude is scene- and model-dependent, so these results should be interpreted as evidence from the current PIVOT v1 scenes rather than as a universal generalization law. Qualitatively, the independently captured unseen views also exhibit stronger blur, loss of detail, and rendering artifacts than representative seen views.

## B. Pose-Space Distance as an Evaluation Descriptor

If reconstruction quality degrades as directed pose Chamfer distance increases, the metric can provide a compact descriptor of how far an evaluation trajectory lies from the training pose distribution. However, such an empirical relationship should not be interpreted as establishing pose-space distance as a complete measure of novel-view difficulty.

In the five scenes, seen views cluster close to the training pose distribution and generally achieve lower LPIPS, whereas unseen trajectories occupy a broader range of pose-space distances and generally higher LPIPS. The relationship is not strictly monotonic: trajectories with similar pose-space distance can differ noticeably in rendering quality. This is expected because the metric describes camera-pose coverage rather than visibility, texture, occlusion, or scene content. We therefore use directed pose Chamfer distance as a descriptive covariate rather than claiming it is a complete predictor of novel-view difficulty.

![](images/d809b5730d5cce445e7919d4440c7d3f0eef7bd3cf4a828cc2d8089958728717.jpg)  
(e) Backyard  
Fig. 3. Relationship between directed normalized pose Chamfer distance to the training trajectories and reconstruction quality (LPIPS) across the PIVOT scenes. Left: Nerfacto. Right: Splatfacto. Seen trajectories (blue) correspond to held-out views from trajectories represented during training, while unseen (orange) trajectories correspond to independently captured evaluation trajectories. Lower LPIPS indicates better reconstruction quality.

![](images/47170f525c00f373a12d9d29a238a2bbe1397b5869223c3b12e016bd477e8c5c.jpg)  
Fig. 4. Qualitative BM1: Seen vs unseen reconstruction results for representative seen and unseen evaluation trajectories. Seen examples are sampled from a training trajectory, while unseen examples are sampled from an independently captured trajectory. Ground-truth images are shown alongside Nerfacto and Splatfacto reconstructions.

![](images/3724d57dfaa236492819b91a9974c130194e31c21abb76eed7be6d713af47ce7.jpg)  
Fig. 5. Qualitative BM2:Measured vs. Optimized Poses reconstruction results for two representative scenes under different combinations of optimized and measured camera pose components. OO denotes optimized translation and optimized rotation, MO denotes measured translation and optimized rotation, OM denotes optimized translation and measured rotation, and MM denotes measured translation and measured rotation. Ground-truth images are shown for reference.

TABLE III  
BM1 QUANTITATIVE RESULTS COMPARING SEEN AND UNSEEN EVALUATION TRAJECTORIES. HIGHER SSIM AND PSNR ARE BETTER (↑), WHILE LOWER LPIPS IS BETTER (↓).
<table><tr><td>Scene</td><td>Model</td><td>Eval. Type</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td></tr><tr><td rowspan="3">church</td><td>Nerfacto</td><td>Seen avg. Unseen avg.</td><td>0.44 0.31</td><td>0.53 0.70</td><td>19.53 16.47</td></tr><tr><td>Splatfacto</td><td>Seen avg. Unseen avg.</td><td>0.65</td><td>0.30</td><td>22.16</td></tr><tr><td>Nerfacto</td><td>Seen avg. Unseen avg.</td><td>0.40 0.33 0.29</td><td>0.57 0.62</td><td>16.86 17.00</td></tr><tr><td rowspan="2">victorian_garden</td><td>Splatfacto</td><td>Seen avg. Unseen avg.</td><td>0.51</td><td>0.71 0.38</td><td>15.15 19.22</td></tr><tr><td>Nerfacto</td><td>Seen avg. Unseen avg.</td><td>0.38 0.57</td><td>0.56 0.47</td><td>14.99 19.79</td></tr><tr><td rowspan="2">village_street</td><td>Splatfacto</td><td>Seen avg.</td><td>0.45 0.80</td><td>0.61 0.22</td><td>16.81 24.50</td></tr><tr><td>Nerfacto</td><td>Unseen avg. Seen avg.</td><td>0.56 0.51</td><td>0.49 0.44</td><td>16.07 19.64</td></tr><tr><td rowspan="2">frontyard</td><td></td><td>Unseen avg. Seen avg.</td><td>0.42 0.74</td><td>0.59</td><td>17.02</td></tr><tr><td>Splatfacto</td><td>Unseen avg.</td><td>0.53</td><td>0.18 0.43</td><td>23.40 16.96</td></tr><tr><td rowspan="2">backyard</td><td>Nerfacto</td><td>Seen avg. Unseen avg.</td><td>0.49 0.38</td><td>0.58 0.70</td><td>17.83 16.17</td></tr><tr><td>Splatfacto</td><td>Seen avg. Unseen avg.</td><td>0.73 0.46</td><td>0.29 0.56</td><td>22.12 15.63</td></tr></table>

## C. Sensitivity to Pose Source

The dual-pose representation allows translation and rotation to be changed independently. This is useful because sensorderived pose errors need not affect the two components equally, and reconstruction methods may exhibit different sensitivity to each.

The BM2 runs consistently favor OO, indicating a substantial benefit from the COLMAP-optimized camera trajectory. In Church, replacing either translation or rotation with measured values reduces PSNR by several decibels for both models, with MM approximately 6.7–7.6 dB below OO. Village Street shows an even larger effect: MM is 6.93 dB below OO for Nerfacto and 11.56 dB below OO for Splatfacto. The relative ordering of OM and MO is not consistent across scenes and models, so the current evidence does not support a general claim that translation or rotation error alone is dominant.

## D. Sensitivity to Intrinsic Source

The intrinsic benchmark contrasts the favorable offline setting in which intrinsics are optimized for the current scene with the deployment-oriented setting in which a fixed physical calibration is reused.

Using the fixed OpenCV calibration instead of COLMAPoptimized per-scene intrinsics reduces reconstruction quality in every all BM3 scene/model pair. The PSNR reduction ranges from 1.75 dB for Victorian Garden/Nerfacto to 9.33 dB for Village Street/Splatfacto. The effect is therefore substantial but strongly scene- and model-dependent. Importantly, this experiment measures sensitivity to the specific independently estimated calibration used in PIVOT v1; it should not be interpreted as evidence that reusable physical calibration is inherently inferior by the same amount in other systems.

## X. LIMITATIONS AND FUTURE WORK

## A. Geometry-Aware Trajectory Distance

The current directed pose Chamfer distance compares camera translation and orientation in pose space. Camera-pose similarity, however, does not necessarily imply similarity in scene visibility. Two camera poses can be spatially close and similarly oriented while lying on opposite sides of an occluding structure. Their pose-space distance can therefore be small even though they observe substantially different scene content.

This is a fundamental limitation of any trajectory descriptor based only on camera pose: pose-space proximity measures where cameras are and how they are oriented, but not which parts of the scene they can observe.

A promising extension is a geometry-aware trajectory similarity measure. Given an available scene reconstruction, such as the COLMAP sparse point cloud, visible scene geometry could be projected into each camera and the overlap between observations estimated, for example using an Intersection-over-Union-based visibility measure. A future trajectory metric could therefore combine:

• translation difference, describing camera separation;

• rotation difference, describing viewing-orientation difference; and

• visibility overlap, describing how much reconstructed scene geometry is jointly observed.

Such a metric could distinguish cameras that are close in pose space but observe different geometry from cameras that are farther apart while retaining substantial scene overlap. The current directed pose Chamfer distance nevertheless remains useful as a simple, scene-geometry-independent descriptor that can be computed directly from camera poses. Geometry-aware distance should therefore be viewed as complementary rather than as a replacement in all settings.

TABLE IV  
BM2 QUANTITATIVE RESULTS FOR OPTIMIZED AND MEASURED CAMERA POSES.
<table><tr><td>Scene</td><td>Model</td><td>Pose Type (TR)</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>∆ PSNR vs. OO</td></tr><tr><td rowspan="6">church</td><td rowspan="5">Nerfacto</td><td>00</td><td>0.44</td><td>0.46</td><td>20.25</td><td>0.00</td></tr><tr><td>OM</td><td>0.20</td><td>0.79</td><td>13.42</td><td>-6.82</td></tr><tr><td>MO</td><td>0.24</td><td>0.73</td><td>15.82</td><td>-4.43</td></tr><tr><td>MM</td><td>0.20</td><td>0.80</td><td>13.55</td><td>-6.69</td></tr><tr><td>00</td><td>0.63</td><td>0.29</td><td>22.13</td><td>0.00</td></tr><tr><td rowspan="4">Splatfacto</td><td>OM</td><td>0.25</td><td>0.60</td><td>14.64</td><td>-7.48</td></tr><tr><td>MO</td><td>0.31</td><td>0.54</td><td>16.58</td><td>-5.54</td></tr><tr><td>MM</td><td>0.25</td><td>0.61</td><td>14.56</td><td>-7.56</td></tr><tr><td>00</td><td>0.22</td><td>0.60</td><td>16.44</td><td>0.00</td></tr><tr><td rowspan="6">victorian_garden</td><td rowspan="3">Nerfacto</td><td>OM</td><td>0.11</td><td>0.84</td><td>13.92</td><td>-2.52</td></tr><tr><td>MO</td><td>0.12</td><td>0.82</td><td>14.82</td><td>-1.62</td></tr><tr><td>MM</td><td>0.10</td><td>0.84</td><td>13.68</td><td>-2.76</td></tr><tr><td rowspan="4">Splatfacto</td><td>00</td><td>0.39</td><td>0.36</td><td>17.28</td><td>0.00</td></tr><tr><td>OM</td><td>0.12</td><td>0.57</td><td>14.70</td><td>-2.59</td></tr><tr><td>MO</td><td>0.15</td><td>0.54</td><td>14.92</td><td>-2.37</td></tr><tr><td>MM</td><td>0.12</td><td>0.56</td><td>14.50</td><td>-2.78</td></tr><tr><td rowspan="6">village_street</td><td rowspan="3">Nerfacto</td><td>00</td><td>0.57</td><td>0.39</td><td>21.03</td><td>0.00</td></tr><tr><td>OM</td><td>0.26</td><td>0.75</td><td>14.64</td><td>-6.39</td></tr><tr><td>MO MM</td><td>0.25</td><td>0.76</td><td>14.10</td><td>-6.93</td></tr><tr><td rowspan="4">00</td><td></td><td>0.25</td><td>0.77</td><td>14.10</td><td>-6.93</td></tr><tr><td>OM</td><td>0.80</td><td>0.17</td><td>24.61</td><td>0.00</td></tr><tr><td>Splatfacto MO</td><td>0.36</td><td>0.58</td><td>16.09</td><td>-8.52</td></tr><tr><td></td><td>0.32</td><td>0.69</td><td>13.72</td><td>-10.89</td></tr><tr><td rowspan="6">frontyard</td><td rowspan="3">Nerfacto</td><td>MM</td><td>0.30</td><td>0.72</td><td>13.05</td><td>-11.56</td></tr><tr><td>00</td><td>0.47</td><td>0.44</td><td>19.51</td><td>0.00</td></tr><tr><td>OM</td><td>0.15</td><td>0.86</td><td>12.98</td><td>-6.54</td></tr><tr><td rowspan="3">MM</td><td>MO</td><td>0.16</td><td>0.85</td><td>13.37</td><td>-6.14</td></tr><tr><td></td><td>0.16</td><td>0.87</td><td>13.08</td><td>-6.43</td></tr><tr><td>00 OM</td><td>0.70</td><td>0.20</td><td>22.54</td><td>0.00</td></tr><tr><td rowspan="4"></td><td>Splatfacto MO</td><td>0.17</td><td>0.63</td><td>13.22</td><td>-9.32</td></tr><tr><td>MM</td><td>0.19</td><td>0.74</td><td>12.23</td><td>-10.31</td></tr><tr><td>00</td><td>0.17</td><td>0.74</td><td>11.96</td><td>-10.58</td></tr><tr><td>OM</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">backyard</td><td rowspan="4">Nerfacto</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MO</td><td></td><td></td><td></td><td></td></tr><tr><td>MM 00</td><td></td><td></td><td></td><td></td></tr><tr><td>OM</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Splatfacto</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MO</td><td></td><td></td><td></td><td></td></tr><tr><td>MM</td><td></td><td></td><td></td><td></td></tr></table>

O denotes COLMAP-optimized and M denotes measured pose components; the first and second letters correspond to translation (T) and rotation (R), respectively.

## B. Dataset Scale and Capture Platform

PIVOT v1 contains five real-world scenes captured with a single DJI Mini 4 Pro drone platform. This controlled setup is useful for isolating the target experimental variables, but it limits conclusions about other cameras, localization systems, environments, and motion platforms. The processing architecture is designed to support additional capture devices, and future releases can extend the scene and device diversity while retaining the same trajectory protocol.

## C. Physical Calibration Quality

The BM3 comparison depends on the quality of the indepen dently estimated physical camera calibration. The calibration used for PIVOT v1 has a relatively high reprojection error (approximately 4 pixels in the calibration run), so part of the observed gap between fixed and COLMAP-optimized intrinsics may reflect calibration quality rather than an unavoidable limitation of reusable calibration. BM3 should therefore be interpreted as a sensitivity experiment for the calibration available in this release. Future captures should use a higherquality calibration procedure and test calibration transfer across scenes and devices.

## D. Dependence on SfM Registration

Some trajectories are deliberately difficult for SfM and may not register completely. PIVOT records registration rates rather than silently discarding this behavior, since registration difficulty is itself relevant to trajectory design. Nevertheless, experiments requiring COLMAP-optimized poses cannot use missing optimized poses without either dropping those frames or explicitly falling back to measured poses. Benchmark configurations must therefore report how unregistered frames are handled.

TABLE V  
BM3 QUANTITATIVE RESULTS COMPARING COLMAP-OPTIMIZED AND OPENCV-CALIBRATED CAMERA INTRINSICS.
<table><tr><td>Scene</td><td>Model</td><td>Camera Calibration Type</td><td>SSIM ↑</td><td>LPIPS↓</td><td>PSNR ↑</td><td>∆ PSNR vs. Optimized</td></tr><tr><td rowspan="3">church</td><td rowspan="2">Nerfacto</td><td>COLMAP optimized</td><td>0.44</td><td>0.45</td><td>20.26</td><td>0.00</td></tr><tr><td>OpenCV calibrated</td><td>0.24</td><td>0.74</td><td>15.78</td><td>-4.47</td></tr><tr><td rowspan="2">Splatfacto</td><td>COLMAP optimized</td><td>0.63</td><td>0.29</td><td>22.13</td><td>0.00</td></tr><tr><td>OpenCV calibrated</td><td>0.29</td><td>0.54</td><td>16.42</td><td>-5.71</td></tr><tr><td rowspan="3">victorian_garden</td><td rowspan="2">Nerfacto</td><td>COLMAP optimized</td><td>0.22</td><td>0.60</td><td>16.46</td><td>0.00</td></tr><tr><td>OpenCV calibrated</td><td>0.11</td><td>0.84</td><td>14.71</td><td>-1.74</td></tr><tr><td rowspan="2">Splatfacto</td><td>COLMAP optimized</td><td>0.39</td><td>0.36</td><td>17.29</td><td>0.00</td></tr><tr><td rowspan="2"></td><td>OpenCV calibrated</td><td>0.12</td><td>0.55</td><td>14.69</td><td>-2.59</td></tr><tr><td rowspan="2">Nerfacto village_street</td><td>COLMAP optimized</td><td>0.57</td><td>0.39</td><td>21.00</td><td>0.00</td></tr><tr><td rowspan="2">Splatfacto</td><td>OpenCV calibrated</td><td>0.25 0.80</td><td>0.75</td><td>14.35</td><td>-6.65</td></tr><tr><td rowspan="2"></td><td>COLMAP optimized OpenCV calibrated</td><td>0.35</td><td>0.17 0.61</td><td>24.63</td><td>0.00</td></tr><tr><td rowspan="2">Nerfacto</td><td>COLMAP optimized</td><td>0.46</td><td>0.45</td><td>15.30 19.50</td><td>-9.33</td></tr><tr><td rowspan="2"></td><td>OpenCV calibrated</td><td>0.17</td><td>0.84</td><td>14.78</td><td>0.00</td></tr><tr><td rowspan="2">Splatfacto</td><td>COLMAP optimized</td><td>0.70</td><td>0.20</td><td>22.56</td><td>-4.72 0.00</td></tr><tr><td rowspan="2"></td><td>OpenCV calibrated COLMAP optimized</td><td>0.21</td><td>0.54</td><td>14.95</td><td>-7.61</td></tr><tr><td>Nerfacto Splatfacto</td><td>OpenCV calibrated COLMAP optimized OpenCV calibrated</td><td></td><td></td><td></td><td></td></tr></table>

## E. Sparse Geometry for Visibility Analysis

The proposed geometry-aware extension would itself depend on reconstruction quality. Sparse COLMAP points do not provide complete scene visibility and may be biased toward textured, repeatedly observed regions. Future geometry-aware metrics should therefore study sensitivity to the underlying geometric representation.

## XI. REPRODUCIBILITY AND RELEASE

The PIVOT source code is released under the MIT License, while the dataset is released under Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0). The project separates the core PIVOT processing environment from a dedicated Nerfstudio integration environment used for Nerfacto/Splatfacto training and benchmark execution.

a) Project repository.: https://github.com/maryraymond/ PIVOT/tree/v1.0.0

b) Nerfstudio integration.: https://github.com/ maryraymond/nerfstudio PIVOT integration/tree/v1.0.0

c) Dataset: https://huggingface.co/datasets/ MaryRaymond/PIVOT/tree/v1.0.0

d) containers:

• PIVOT ghcr.io/maryraymond/pivot:1.0.0

• Nerfstudio integration ghcr.io/maryraymond/nerfstudio pivot integ:1.0.0

## XII. CONCLUSION

We presented PIVOT, a multi-trajectory dataset and testbed designed to separate several favorable assumptions that are often coupled in 3D reconstruction evaluation. By preserving measured and optimized poses, calibrated and optimized intrin sics, and explicit trajectory identity, PIVOT enables controlled experiments on pose quality, calibration, capture trajectory, and novel-view generalization. Its directed pose Chamfer distance provides a simple pose-space description of evaluationtrajectory coverage, while the benchmark design explicitly distinguishes held-out views on represented trajectories from complete unseen trajectories.

Across all scenes, the initial experiments show a consistent gap between seen held-out views and independently captured unseen trajectories, substantial degradation when measured pose components replace COLMAP-optimized poses, and measurable sensitivity to the intrinsic calibration source. The magnitude of these effects varies by scene and model, and the pose-space distance is descriptive rather than a complete predictor of rendering difficulty. Together, these results support the use of trajectory identity, pose source, and intrinsic source as explicit evaluation dimensions. The broader goal of PIVOT is to make the gap between reconstruction benchmarks and real-world capture conditions easier to measure, reproduce, and study.

## REFERENCES

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In ECCV, 2020.

[2] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis. 3D Gaussian ¨ Splatting for Real-Time Radiance Field Rendering. ACM Transactions on Graphics, 2023.

[3] J. L. Schonberger and J.-M. Frahm. Structure-from-Motion Revisited. In¨ CVPR, 2016.

[4] J. L. Schonberger, E. Zheng, J.-M. Frahm, and M. Pollefeys. Pixelwise¨ View Selection for Unstructured Multi-View Stereo. In ECCV, 2016.

[5] B. Mildenhall, P. P. Srinivasan, R. Ortiz-Cayon, N. K. Kalantari, R. Ramamoorthi, R. Ng, and A. Kar. Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines. ACM Transactions on Graphics (SIGGRAPH), 2019.

[6] J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, and P. Hedman. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. In CVPR, 2022.

[7] C.-H. Lin, W.-C. Ma, A. Torralba, and S. Lucey. BARF: Bundle-Adjusting Neural Radiance Fields. In ICCV, 2021.

Scene Camera Intrinsics  
Nerfacto  
![](images/c2d97a5becc46d5d1ba5ad7a8222d2b5e11034bcca40bfb70d7138f28b1bd887.jpg)  
GT  
Village Street

Splatfacto  
![](images/33adfd46dca1ebe8d6e6149b8068b26663adca79bb0258d1f3c876c246170744.jpg)

COLMAP optimized  
![](images/99cae386c83943593be2a52384935b6da9cbe987dfb6ca87b08d9cbf2af9494e.jpg)

![](images/551dee7fee7c007b339048f96584a2aa00f07011813fb12284863d30fa5abdc4.jpg)

OpenCV calibrated  
![](images/3d837a17a29794ada5bd1a41b6c9357e4a15e247a5b785c875b405d09300fb0d.jpg)

![](images/110b0276d61e3d154dc4fa33cacdc57fe295ceffc5bbcfa005aa5e296eb59767.jpg)

GT  
![](images/9d8d9dd6ef86e072bbfc95e72b84408512a9310e5f01befbb0ba2f961bf85724.jpg)

![](images/3d48c9e73d326fa017242ef37985633c39db9f9acbf8e3c7d77bf198ade4cd52.jpg)

Church  
![](images/85cb829f48f345e45912dd758f610ad0465e05d4da1032ccfee559b1aaf857c7.jpg)

![](images/30dd1ca197b201df4ae015ae29b068a4285b313cedd6fa6c66b26c4f7699e7b2.jpg)

![](images/377df1bb2f6a460297da320341ee85f67d25a0c82c77d5e332ee7bfefcf7bfba.jpg)

![](images/40841c09b2a922db164b3c4737d9360db1d6c583bf55e0ffbbfd6e312a4c70fb.jpg)  
Fig. 6. Qualitative BM3 reconstruction results for two representative scenes using COLMAP-optimized and independently calibrated camera intrinsics. Ground-truth (GT) images are shown for reference. The COLMAP condition uses scene-optimized camera intrinsics, while the calibrated condition uses the fixed OpenCV camera calibration.

[8] Z. Wang, S. Wu, W. Xie, M. Chen, and V. A. Prisacariu. NeRF–: Neural Radiance Fields Without Known Camera Parameters. arXiv preprint arXiv:2102.07064, 2021.

[9] M. Tancik et al. Nerfstudio: A Modular Framework for Neural Radiance Field Development. ACM SIGGRAPH 2023 Conference Proceedings, 2023.

[10] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,” in CVPR, 2018.

[11] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[12] H. Fan, H. Su, and L. J. Guibas, “A Point Set Generation Network for 3D Object Reconstruction from a Single Image,” in CVPR, 2017.

[13] C. Campos, R. Elvira, J. J. Gomez Rodr´ ´ıguez, J. M. M. Montiel, and J. D. Tardos. ORB-SLAM3: An Accurate Open-Source Library for Visual,´ Visual–Inertial, and Multimap SLAM. IEEE Transactions on Robotics, 37(6):1874–1890, 2021.