# CLOCs-R: Radar-Assisted Camera-LiDAR Fusion for Long-Range Object Detection

Richard Beuck

U.A. Whitaker College of Engineering Florida Gulf Coast University Fort Myers, FL, USA rjbeuck1394@eagle.fgcu.edu

John Nguyen

U.A. Whitaker College of Engineering Florida Gulf Coast University Fort Myers, FL, USA jdnguyen1381@eagle.fgcu.edu

Sudip Dhakal U.A. Whitaker College of Engineering Florida Gulf Coast University Fort Myers, FL, USA sdhakal@fgcu.edu

## I. RELATED WORK

This section reviews the camera-LiDAR and radar fusion methods most relevant to our proposed approach, grouped by the sensors used and the stage at which they are fused. Table I summarizes the method, inputs, dataset, key result, and main limitation of each work referenced in this section.

## A. Camera-LiDAR Fusion

Most camera-LiDAR fusion methods build on a LiDAR detector. SECOND [1] replaced the dense 3D convolutions of earlier voxel networks with sparse convolutions, reaching 73.66 moderate difficulty car 3D AP on the KITTI test set [18] at about 20 frames per second. Its authors note, however, that it misses many distant cars that return fewer than 10 LiDAR points [1].

Fusion methods differ mainly in where the two modalities meet. At the point level, PointPainting [2] projects each LiDAR point into an image segmentation map and appends the class scores to the point, so that any LiDAR detector can use them. This raised PointPillars+ from 40.1 to 46.4 mAP on the nuScenes test set [2], [19], but painting can only help objects that already have LiDAR points. At the feature level, BEVFusion [5] lifts camera features into a bird’s-eye-view (BEV) grid shared with LiDAR and makes this transform about 40 times faster, reaching 70.2 mAP and 72.9 NDS on nuScenes test, although inaccurate camera depth leaves local misalignment between the two feature maps.

At the decision level, CLOCs [3] takes the candidates of a pretrained 2D detector and a pretrained 3D detector before non-maximum suppression (NMS). Each overlapping pair is encoded by its image-plane IoU, both confidence scores, and the object’s distance, and a small network learns to rescore the 3D candidates. Fusing SECOND with Cascade R-CNN raised moderate car 3D AP on the KITTI test set by 5.90 points, with the largest gains at 40-50 m [3]. Because late fusion only prunes and rescores, however, CLOCs cannot recover objects that the 3D detector never proposed [3]. Fast-CLOCs [4] replaces the separate 2D detector with a lightweight one that uses projected 3D candidates as proposals, running at 8-13 Hz on a desktop GPU and raising CenterPoint from 58.0 to 63.1 mAP on nuScenes test. This design makes the 2D branch depend even more on the recall of the 3D detector [4].

## B. Camera-Radar Fusion

Radar complements both cameras and LiDAR. Automotive radars detect targets at ranges of 250 m or more, measure radial velocity directly through the Doppler effect, and are less affected by poor lighting and weather [19], [20]. Their returns, however, are sparse, noisy, and lack reliable height information [8], [13].

Early work used radar to guide 2D detection. CRF-Net [6] projects radar returns into the image as additional input channels and lets the network learn at which feature level to fuse them, while RRPN [7] converts each projected radar return into distance-scaled anchor boxes, generating proposals far faster than Selective Search. Both produce only 2D boxes, and CRF-Net gained little from unfiltered radar [6], [7].

Object-level methods attach radar to camera detections. CenterFusion [8] associates radar returns with camera-detected object centers through a frustum and uses them to re-estimate depth and velocity. This improved NDS and sharply reduced velocity error on nuScenes, but mAP improved only marginally, since radar can only refine objects the camera has already found. RADIANT [9] instead predicts the offset from each radar return to the object center and fuses radar depth into a frozen monocular detector, although only object position is refined.

BEV-level methods fuse radar earlier so that it can help locate objects. CRN [10] uses radar occupancy to guide the camera-to-BEV transform and fuses the two maps with deformable cross-attention. For cars at 30-100 m it even outperforms the LiDAR detector CenterPoint, although its training still relies on depth supervision from projected LiDAR points [10]. CR3DT [11] concatenates radar pillars with camera BEV features and passes the improved velocity estimates to a tracker, improving both detection and tracking with little added latency, but it still trails CenterPoint by a wide margin. RCM-Fusion [12] adds an instance-level stage that refines each proposal using nearby radar points, which improves accuracy at the cost of a large and slow model.

TABLE I  
COMPARISON OF REVIEWED WORK (C = CAMERA, L = LIDAR, R = RADAR, NUSC. = NUSCENES, VOD = VIEW-OF-DELFT)
<table><tr><td>Work</td><td>Method</td><td>Input</td><td>Dataset</td><td>Key Result</td><td>Limitation</td></tr><tr><td>SECOND [1]</td><td>Sparse 3D convolution on voxels</td><td>L</td><td>KITTI</td><td>73.66 mod. car 3D AP at 20 FPS</td><td>Misses cars with &lt;10 points</td></tr><tr><td>PointPainting [2]</td><td>Image segmentation scores on points</td><td>C+L</td><td>KITTI, nuSc.</td><td>mAP 40.1 → 46.4 (nuSc. test)</td><td>Needs LiDAR points on the object</td></tr><tr><td>CLOCs [3]</td><td>Rescores pre-NMS 2D-3D pairs</td><td>C+L</td><td>KITTI</td><td>+5.90 mod. car 3D AP; best at 40–50 m</td><td>Cannot add missed 3D boxes</td></tr><tr><td>Fast-CLOCs [4]</td><td>Light 2D detector cued by 3D boxes</td><td>C+L</td><td>KITTI, nuSc.</td><td>mAP 58.0 → 63.1; 8–13 Hz</td><td>Needs high 3D recall</td></tr><tr><td>BEVFusion [5]</td><td>Shared BEV feature fusion</td><td>C+L</td><td>nuSc., Waymo</td><td>70.2 mAP, 72.9 NDS (test)</td><td>Depth errors misalign features</td></tr><tr><td>CRF-Net [6]</td><td>Radar image channels, multi-level</td><td>C+R</td><td>nuSc., TUM</td><td>mAP 43.47 → 43.95</td><td>2D only; small gain</td></tr><tr><td>RRPN [7]</td><td>Distance-scaled radar anchors</td><td>C+R</td><td>nuSc.</td><td>Beats Selective Search; &gt;100× faster</td><td>2D only; needs a radar return</td></tr><tr><td>CenterFusion [8]</td><td>Frustum radar-center association</td><td>C+R</td><td>nuSc.</td><td>NDS 40.0 → 44.9 (test)</td><td>Only marginal mAP gain</td></tr><tr><td>RADIANT [9]</td><td>Radar-to-center offsets, depth fusion</td><td>C+R</td><td>nuSc.</td><td>PGD mAP 36.0 → 38.0 (test)</td><td>Refines position only</td></tr><tr><td>CRN [10]</td><td>Radar-guided BEV transform</td><td>C+R</td><td>nuSc.</td><td>62.4 NDS; beats LiDAR at 30–100 m†</td><td>LiDAR depth used in training</td></tr><tr><td>CR3DT [11] RCM-Fusion [12]</td><td>Radar pillars + BEV features, tracking</td><td>C+R C+R</td><td>nuSc. nuSc.</td><td>+5.3 mAP, +14.9 AMOTA (val)</td><td>21.3 mAP below LiDAR</td></tr><tr><td></td><td>BEV + instance-level radar refinement</td><td></td><td></td><td>58.7 NDS, 50.6 mAP (test)</td><td>2.3 FPS; 87.6M parameters</td></tr><tr><td>RadarNet [13]</td><td>Early and attention-based late fusion</td><td>L+R</td><td>nuSc., DenseRadar</td><td>Car AP 82.3 → 84.5 (val)</td><td>Only +0.8 AP at 70–100 m</td></tr><tr><td>FUTR3D [14]</td><td>Sensor-agnostic query sampler</td><td>Any</td><td>nuSc.</td><td>C+R 39.9 vs. 33.2 mAP (val)</td><td>No C+L+R results</td></tr><tr><td>View-of-Delft [15]</td><td>4D radar dataset</td><td>C+L+R</td><td>VoD</td><td>Radar mAP 38.0 → 29.1 without Doppler</td><td>Labels only to 50 m</td></tr><tr><td>K-Radar [16]</td><td>4D radar tensor dataset</td><td>C+L+R</td><td>K-Radar</td><td>Heavy snow AP: radar 44.5 vs. LiDAR 29.7</td><td>Labels to 120 m; ~12 TB</td></tr><tr><td>aiMotive [17]</td><td>Long-range dataset with baselines</td><td>C+L+R</td><td>aiMotive</td><td>Night AP &gt;75 m: 0.464 → 0.550</td><td>Simple fusion; worse in rain</td></tr></table>

<sup>†</sup>CRN outperforms LiDAR-only CenterPoint on cars at 30–100 m, in the authors’ extended 100 m evaluation of nuScenes.

## C. Radar-LiDAR Fusion

Radar has also been fused directly with LiDAR. Radar-Net [13] combines the two sensors through early voxel-level fusion and late attention-based fusion, improving car AP on nuScenes over the best LiDAR-only model while lowering velocity error. Its authors observe that LiDAR point density decreases with range, whereas radar provides more observations at long range, yet on their 100 m DenseRadar dataset radar added little at 70-100 m [13]. FUTR3D [14] uses a modalityagnostic query sampler so that a single transformer decoder can use any combination of cameras, LiDAR, and radar. Its camera-radar model outperforms CenterFusion on nuScenes, but the paper reports no results using all three sensors.

## D. Datasets with Radar and Long-Range Evaluation

Long-range gains are also difficult to measure. KITTI has no radar [18], and in nuScenes the LiDAR is rated only to 70 m while each radar reaches 250 m [19]. Two recent datasets provide 4D radar, which measures elevation in addition to range, azimuth, and Doppler velocity. View-of-Delft [15] pairs a 3+1D radar with a 64-beam LiDAR and a stereo camera in dense urban traffic. A radar-only detector trained on it still trails its LiDAR counterpart, and removing Doppler velocity from the radar input markedly lowers accuracy, but annotations extend only to 50 m [15]. K-Radar [16] records the full 4D radar tensor together with LiDAR and cameras in fog, rain, and snow, with labels to 120 m, and its radaronly baseline outperforms a LiDAR detector in heavy snow. The aiMotive dataset [17] was designed specifically for longrange perception, with four cameras, a 64-beam LiDAR, two long-range radars, and annotations out to about 200 m across highway, urban, night, and rain scenes. In its baselines, adding radar to LiDAR barely changed overall AP beyond 75 m but raised it substantially at night, while it reduced accuracy in heavy rain, which the authors attribute to their simple fusion method [17].

## E. Research Gap

Radar helps most where cameras and LiDAR are weakest, at long range and at night [10], [17], yet long-range accuracy remains low: even CRN reaches only 7.0 car AP at 60- 100 m on nuScenes [10]. Most radar fusion methods pair radar with cameras alone and train the full network end to end [10]–[12], while the few that use all three sensors either report no three-sensor results [14] or rely on simple fusion [17]. Public datasets also limit long-range study: the 4D radar datasets View-of-Delft and K-Radar label objects only to 50 and 120 m [15], [16], so among the datasets reviewed here, only aiMotive combines camera, LiDAR, and long-range radar with annotations out to 200 m [17]. Candidate-level fusion such as CLOCs [3], [4] is lightweight, works with any pretrained detectors, and gains the most at long range, but it has been studied only with cameras and LiDAR, and its long-range benefit has been analyzed only out to 50 m [3]. To our knowledge, no prior work adds radar evidence to candidate-level fusion. CLOCs-R plans to address this gap by adding radar-derived features, namely the radar returns supporting each 3D candidate and their Doppler consistency, to the CLOCs pair encoding while keeping all detectors frozen, and by evaluating detection accuracy by distance out to 200 m on the aiMotive dataset.

## II. METHODOLOGY

CLOCs-R extends the candidate-level fusion of CLOCs [3] with radar evidence. A pretrained 2D image detector and a pretrained 3D LiDAR detector produce candidates before nonmaximum suppression (NMS), radar returns are associated with each 3D candidate, and a small fusion network rescores the 3D candidates using geometric, semantic, and radar cues. Both detectors remain frozen, so only the fusion network is trained. Because it is not known in advance which radar cues are most informative, CLOCs-R defines a set of candidate radar features and evaluates several combinations of them as inputs to the fusion network. This section describes candidate generation, the radar features, the fusion network and its training, and the evaluation protocol.

## A. Overview

For each frame, the inputs are a front camera image, a LiDAR sweep, and the radar returns from the most recent radar sweeps, together with the calibration between the sensors. The 2D detector outputs k axis-aligned image boxes with scores $s _ { i } ^ { 2 D }$ , and the 3D detector outputs n oriented boxes with scores $s _ { j } ^ { 3 D }$ . Following CLOCs [3], every 2D-3D pair is encoded as one cell of a sparse k × n tensor, a stack of 1×1 convolutions maps each cell to a logit, and max-pooling over the 2D axis yields one fused score per 3D candidate. CLOCs-R appends radar channels to each cell, drawn from the candidate features in Section II-C. Because these channels describe the 3D candidate rather than the pair, they are also available to candidates that have no overlapping 2D detection, which allows radar to support distant objects that the camera detector misses. The fused scores replace the original 3D scores, and standard NMS produces the final detections. As in CLOCs, box geometry is inherited from the 3D detector and only confidence scores change. Fig. 1 shows the pipeline on a clear daytime urban frame from the aiMotive dataset, using the front camera, the LiDAR, and the front long-range radar, with the bird’s-eye views drawn out to 100 m. LiDAR density falls quickly with range: a car at 14 m receives about 2,000 LiDAR points, whereas a car at 63 m receives 39. The radar returns in the same frame extend to 171 m, beyond the region shown. Radar returns are separated into moving objects and static clutter by their ego-compensated Doppler velocity and are matched to each 3D candidate through a gate that widens with range. The resulting radar features, namely the support r, Doppler speed v, Doppler spread e, range residual $\Delta ,$ and nearest-return offset $o$ defined in Section II-C, are appended to every cell of column $j ,$ including the no-camera row. Components that CLOCs-R adds are shown in orange, and the fusion network, whose input is widened, in purple. The aiMotive data are used under the CC BY-NC-SA 4.0 license.

## B. Candidate Generation

3D detector. We use SECOND [1] as the 3D detector, the same detector used in the CLOCs configuration that we reproduce on KITTI. For aiMotive [17], the detector is retrained with its point-cloud range extended to cover the dataset’s annotation range and its anchor sizes set from the aiMotive training labels. All anchor-level predictions are exported before NMS with no score threshold, so that the fusion stage sees every candidate the detector produces.

2D detector. We use Cascade R-CNN [21] as the 2D detector, implemented in MMDetection [22]. Because aiMotive provides only 3D annotations, 2D training labels are generated by projecting each annotated 3D box into the front camera image with the dataset’s calibration and taking the tightest enclosing axis-aligned rectangle. Boxes whose projections fall outside the image are discarded. At inference, the detector’s outputs are likewise taken before NMS and with a very low score threshold, and the top $K _ { 2 D } = 2 0 0$ candidates per image are kept, matching the CLOCs implementation.

Caching. Both detectors are run once over every frame, and their candidates are stored on disk. The fusion network is then trained and evaluated from these cached candidates, which keeps its training independent of the cost of the detectors.

## C. Radar Evidence Features

Preprocessing. Radar returns are transformed into the Li-DAR coordinate frame using the extrinsic calibration. Because a single radar sweep is sparse, returns from the last M sweeps are accumulated and transformed into the current frame using the ego-vehicle poses. Each return $q$ provides a bird’s-eyeview position ${ \bf p } _ { q }$ and a measured radial velocity $v _ { q } ^ { m }$ . The aiMotive radars report the range, azimuth, elevation, radial speed (in m/s), and reflectivity of every return [17], so a Doppler measurement is available for each return that CLOCs-R uses. This measurement is important: on View-of-Delft, removing Doppler velocity from the input of a radar-only detector lowered mAP from 38.0 to 29.1 [15]. The reported speed is relative to the ego vehicle: in the aiMotive sample sequence, returns from static structures read about −30 m/s at highway speed, close to the negative ego speed, while returns on annotated vehicles match their labeled relative velocities. The radial velocity is therefore compensated as

$$
\widetilde { v } _ { q } = v _ { q } ^ { m } + \mathbf { u } ^ { \top } \frac { \mathbf { p } _ { q } } { \| \mathbf { p } _ { q } \| } ,\tag{1}
$$

where u is the ego velocity in the BEV plane, obtained by differencing the consecutive GNSS-INS ego poses provided with the dataset, so that static objects have $\tilde { v } _ { q } \approx 0$ . We verify the compensation on each sequence by checking that returns from static structures, such as guardrails and signs, have nearzero $\tilde { v } _ { q }$

Association. For each 3D candidate $j ,$ we form a gate $\mathcal { G } _ { j }$ by enlarging its BEV footprint by a margin

$$
\begin{array} { r } { \delta _ { j } = \delta _ { 0 } + \kappa \rho _ { j } , } \end{array}\tag{2}
$$

where $\rho _ { j }$ is the candidate’s distance from the ego vehicle. The range-dependent term accounts for radar’s lateral position error, which grows with distance for a fixed angular resolution. The parameters $\delta _ { 0 }$ and κ are selected on a held-out subset of the training scenes. Let $\mathcal { Q } _ { j }$ be the set of radar returns inside $\mathcal { G } _ { j }$ , with $N _ { j } = | \mathcal { Q } _ { j } |$

Candidate features. We define five candidate features that summarize the radar evidence for candidate $j ,$ each computed only from the positions and compensated radial velocities of the returns in $\mathcal { Q } _ { j }$

• Support, $r _ { j } = \log ( 1 + N _ { j } )$ , which measures how many returns fall in or near the box. The logarithm limits the influence of large vehicles that produce many returns.

• Doppler speed, $v _ { j } = \mathrm { m e d i a n } _ { q \in \mathcal { Q } _ { j } } | \tilde { v } _ { q } | / v _ { \mathrm { m a x } }$ , which helps distinguish moving objects from static clutter such as guardrails and signs.

![](images/a035d1627c5ccd0f43151c6c8443e9fa38a398f38efeeaf963c7738294a9f52e.jpg)  
Fig. 1. CLOCs-R architecture, shown on a frame from the aiMotive dataset [17]. The frozen camera and LiDAR detectors produce candidates before NMS radar returns gated around each 3D candidate give its radar features $\phi _ { j } ,$ , and these are appended to every cell of column j of the pair tensor before the fusion network rescores each 3D box. Candidate boxes are generated from ground-truth labels for illustration.

• Doppler spread, $e _ { j } = \mathrm { m e d i a n } _ { q \in \mathcal { Q } _ { j } } | \tilde { v } _ { q } - \bar { v } _ { j } | / v _ { \mathrm { m a x } }$ , where $\bar { v } _ { j }$ is the median of $\tilde { v } _ { q }$ . Returns from a single rigid object share one velocity, so a small spread indicates that the returns are consistent with one object rather than a mix of neighboring objects and clutter.

• Range residual, $\Delta _ { j } = | \mathrm { m e d i a n } _ { q \in \mathcal { Q } _ { j } } | | { \bf p } _ { q } | | - \rho _ { j } | / \delta _ { j }$ , which compares radar’s direct range measurement with the range of the 3D box and can indicate a candidate placed at the wrong depth.

• Nearest-return offset, $o _ { j } = \operatorname* { m i n } _ { { q } \in \mathcal { Q } _ { j } } \| { \bf p } _ { q } - { \bf c } _ { j } \| / \delta _ { j }$ , where $\mathbf { c } _ { j }$ is the box center in BEV, which measures how closely the radar evidence agrees with the box location.

The medians reduce the effect of outlying returns, and dividing by the constant $v _ { \mathrm { m a x } }$ or the gate margin $\delta _ { j }$ keeps every feature on a comparable scale. When $N _ { j } = 0$ , we set $r _ { j } = 0$ and all other radar features to −1, following the CLOCs convention of using −1 to mark missing evidence so that it is distinguishable from a measured zero.

Feature combinations. Table II lists the radar feature sets we evaluate. They range from single features, which show what each cue contributes on its own, to the support and Doppler pair proposed for CLOCs-R and the full set of five. Each set is trained and evaluated separately with an otherwise identical network and training schedule. The set used for the final CLOCs-R model is selected by its center-distance AP beyond 50 m on held-out training scenes, and all sets are reported on the validation split.

## D. Fusion Network and Training

Each non-empty cell of the input tensor holds

$$
\begin{array} { r } { \mathbf { T } _ { i j } = \left[ \mathrm { I o U } _ { i j } , \ s _ { i } ^ { 2 D } , \ s _ { j } ^ { 3 D } , \ d _ { j } , \ \phi _ { j } ^ { \top } \right] , } \end{array}\tag{3}
$$

TABLE II  
RADAR FEATURE SETS EVALUATED
<table><tr><td>Set</td><td> $\boldsymbol { r } _ { j }$ </td><td> $v _ { j }$ </td><td> $e _ { j }$ </td><td> $\Delta _ { j }$ </td><td> $o _ { j }$ </td><td>Channels</td></tr><tr><td>RO (CLOCs, no radar)</td><td></td><td></td><td></td><td></td><td></td><td>4</td></tr><tr><td>R1</td><td>√</td><td></td><td></td><td></td><td></td><td>5</td></tr><tr><td>R2</td><td></td><td>√</td><td></td><td></td><td></td><td>5</td></tr><tr><td>R3</td><td>√</td><td>√</td><td></td><td></td><td></td><td>6</td></tr><tr><td>R4</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>7</td></tr><tr><td>R5</td><td>√</td><td></td><td></td><td>√</td><td>√</td><td>7</td></tr><tr><td>R6</td><td>√</td><td>√</td><td></td><td>√</td><td></td><td>7</td></tr><tr><td>R7 (all)</td><td>√</td><td>√</td><td>L</td><td>√</td><td>√</td><td>9</td></tr></table>

where $\mathrm { I o U } _ { i j }$ is the overlap between 2D box i and the image projection of 3D box $j , d _ { j }$ is the normalized BEV distance of candidate $j$ from the LiDAR, and $\phi _ { j } \in \mathbb { R } ^ { m }$ holds the m radar features of the feature set being evaluated. As in CLOCs, cells with zero IoU are discarded, and each 3D candidate without an overlapping 2D box is kept in a reserved row with IoU and $s ^ { 2 D }$ set to −1 [3]. The radar channels are filled in every cell of column $j ,$ including this reserved row. The fusion network keeps the CLOCs architecture of four 1×1 convolutions with ReLU activations ((4+m)→18→36→36→1), so only the input width changes with the feature set. The network therefore has between 2,161 and 2,233 parameters across the sets in Table II, compared with 2,143 for CLOCs. The fused score of candidate j is

$$
\hat { s } _ { j } = \operatorname* { m a x } _ { i } f _ { \theta } ( \mathbf { T } _ { i j } ) ,\tag{4}
$$

where $f _ { \theta }$ is the fusion network and the maximum is taken over all non-empty cells in column $j .$ Algorithm 1 summarizes the per-frame procedure.

Algorithm 1 CLOCs-R inference for one frame   
Require: image I, LiDAR sweep ${ \overline { { P , } } }$ last M radar sweeps,   
ego poses, radar feature set $s$   
1: $\{ ( \bar { B } _ { i } ^ { 2 D } , s _ { i } ^ { 2 D } ) \}  2 \bf { D }$ detector(I) ▷ pre-NMS, top $K _ { 2 D }$   
2: $\{ ( B _ { j } ^ { 3 D } , s _ { j } ^ { \bar { 3 } D } ) \}  3 \mathrm { D }$ detector $\cdot ( P )$ ▷ pre-NMS   
3: $\mathcal { Q } \dot { - }$ accumulate radar sweeps; compensate Doppler (1)   
4: for each 3D candidate $j$ do   
5: $\delta _ { j }  \delta _ { 0 } + \kappa \rho _ { j } \ ( 2 )$   
6: $\bar { \mathcal { Q } } _ { j }  \{ q \in \bar { \mathcal { Q } } : \mathbf { p } _ { q } \in \mathcal { G } _ { j } \}$   
7: $\phi _ { j } $ features in $s$ from $\mathcal { Q } _ { j }$ ▷ −1 if $\mathcal { Q } _ { j } = \emptyset$   
8: end for   
9: $\mathbf { T } $ cells $\mathbf { T } _ { i j }$ with $\mathrm { I o U } _ { i j } > 0 ,$ plus no-camera row (3)   
10: $\hat { s } _ { j } \gets \operatorname* { m a x } _ { i } f _ { \theta } \mathbf ( \mathbf { \bar { T } } _ { i j } )$ for all $j ~ ( 4 )$   
11: return NMS $\left( \{ ( \hat { B } _ { j } ^ { 3 D } , \hat { s } _ { j } ) \} \right)$

Training targets follow the CLOCs implementation: a 3D candidate is positive if its BEV IoU with a ground-truth box is at least 0.7 and negative if it is at most 0.5, and candidates in between are ignored. Because nearly all candidates are background, the network is trained with focal loss [23] with $\alpha \ = \ 0 . 2 5$ and $\gamma \ = \ 2$ . We use the Adam optimizer [24] with a one-cycle learning-rate schedule (maximum learning rate $3 \times 1 0 ^ { - 3 } )$ , weight decay of 0.01, and a batch size of one frame.

## E. Datasets and Evaluation Protocol

KITTI verification. Before modifying the method, we reproduce camera-LiDAR CLOCs on KITTI [18] with the official implementation, confirming that our software environment matches the published results. This step uses the standard split of 3,712 training and 3,769 validation frames.

aiMotive. All CLOCs-R experiments use the aiMotive dataset [17], which provides synchronized camera, 64-beam LiDAR, and long-range radar data, GNSS-INS ego poses, and 3D annotations out to about 200 m. We use the front camera, the LiDAR, and the front long-range radar. We use its official split of 151 training and 25 validation scenes (21,402 and 5,181 annotated frames), hold out a subset of training scenes for hyperparameter selection, and report all results on the validation split. Experiments begin with the car class.

Distance-binned metrics. Because the expected benefit of radar is at long range, every metric is reported separately for objects at 0–30 m, 30–50 m, 50–100 m, and 100–200 m from the ego vehicle. For each bin we report 3D average precision (AP), recall, and center-distance AP. Center-distance AP matches predictions to ground truth by BEV center distance rather than IoU, following nuScenes [19], because small position errors at long range would otherwise dominate an IoU-based score. We also report the latency that the radar features and fusion network add per frame.

Baselines and ablations. We compare against two baselines trained on aiMotive under the same split: the LiDAR-only 3D detector with its original scores, and camera-LiDAR CLOCs (set R0). Together they isolate the contributions of the camera and of radar. Each radar feature set in Table II is then compared with R0 to measure which radar cues carry the gain and whether adding more features helps or introduces noise. To measure how the method behaves when the camera is unavailable or degraded, for example at night, the selected feature set is also evaluated with all 2D candidates removed at test time, so that every 3D candidate uses the reserved no-camera row. The LiDAR-radar and three-sensor baselines released with aiMotive [17] serve as external reference points, although they use different detectors.

## F. Implementation Details

The fusion network and training pipeline build on the official CLOCs implementation [3], updated to run with PyTorch 1.10 and CUDA 11.3 on current GPUs. The fixed candidate dimensions of the original code are generalized so that the number of 3D candidates follows the extended aiMotive anchor grid. The 3D detector is trained with OpenPCDet [25]. Experiments run on NVIDIA A100 GPUs, and the latency added by computing the radar features and running the fusion network is measured for each feature set on a single GPU with batch size one, excluding the cost of the two detectors, which is reported separately.

## III. EXPERIMENTS

This section reports the experiments that follow the protocol of Section II, and Section III-B states where a run departs from it. Fusion results are means over three training seeds. Every number comes from the result files and logs in our repository. Changes made after the validation split had been scored are marked as post hoc (<sup>†</sup>).

## A. Dataset

Source and subset. We use the aiMotive dataset [17], restricted to its daytime clear-weather highway and urban scenes; night and rain scenes are left for future work. In each of the two conditions, every tenth official training scene is held out for selection, which leaves 109 training scenes and 13 holdout scenes. All 21 official validation scenes in these conditions are used for reporting. To limit storage, every second annotated frame of the training and holdout scenes is kept, giving 7,753 training, 923 holdout and 4,292 validation frames.

Labels and scored region. The car class combines the CAR, VAN and PICKUP labels. Trucks, buses and trailers are ignored: candidates that overlap them (BEV IoU of at least 0.1) count as neither true nor false positives, in training or in evaluation. An object is scored when its center lies inside the SECOND grid $( 0 \leq x < 2 0 4 . 8 \mathrm { m } , | y | < 2 5 . 6 \mathrm { m } )$ and inside the front camera’s view. The validation split then contains 17,669 cars, and holdout contains 6,242, of which only 466 are at 100–200 m.

Label shift. The training scenes are auto-annotated with manual checks, while the validation scenes were labeled by hired annotators [17]. The car labels differ accordingly. The median length is 4.00 m in training, where 43% of cars are exactly 4.00 m long (58% in holdout), and 4.37 m in validation, where none are. Detectors learn the training convention, so strict IoU metrics on validation reflect label differences as well as detection errors (Section III-D).

TABLE III  
RESULTS ON AIMOTIVE VALIDATION (DAYTIME CLEAR WEATHER, 4,292 FRAMES)
<table><tr><td></td><td></td><td colspan="6">APBEV@0.3</td><td></td><td colspan="6">Center-distance AP (APCD)</td><td></td></tr><tr><td></td><td>Method</td><td>0-30</td><td>30-50</td><td>50-100</td><td></td><td>100-200</td><td>&gt;50</td><td>All</td><td>0-30</td><td>30-50</td><td>50-100</td><td>100-200</td><td></td><td>&gt;50</td><td>All</td></tr><tr><td></td><td>SECOND</td><td>0.975</td><td>0.847</td><td>0.725</td><td></td><td>0.277</td><td>0.517</td><td>0.630</td><td>0.909</td><td>0.664</td><td>0.615</td><td>0.280</td><td>0.458</td><td>0.556</td><td></td></tr><tr><td></td><td>CLOCs (RO)</td><td>0.974</td><td>0.857</td><td>0.701</td><td>0.267</td><td></td><td>0.493</td><td>0.599</td><td>0.926</td><td>0.693</td><td>0.589</td><td>0.252</td><td>0.424</td><td>0.525</td><td></td></tr><tr><td></td><td> $\mathbf { C L O C s - R } \ ( \mathbf { R } 2 )$ </td><td>0.974</td><td>0.870</td><td>0.706</td><td>0.271</td><td></td><td>0.497</td><td>0.607</td><td>0.918</td><td>0.698</td><td>0.593</td><td>0.255</td><td>0.428</td><td>0.530</td><td></td></tr><tr><td></td><td> $\mathrm { R 0 } + \mathrm { r e s i d u a l } ^ { \dag }$   ${ \mathrm { R } } 2 \mathrm { ~ + ~ } \mathrm { r e s i d u a l } ^ { \dagger }$ </td><td>0.974 0.974</td><td>0.861</td><td>0.723</td><td>0.296</td><td></td><td>0.521</td><td>0.623</td><td>0.925</td><td>0.697</td><td>0.606</td><td>0.280</td><td>0.448</td><td>0.545</td><td></td></tr><tr><td></td><td colspan="2"></td><td></td><td>0.874</td><td>0.727</td><td>0.298</td><td>0.524</td><td>0.630</td><td>0.922</td><td>0.704</td><td>0.609</td><td>0.283</td><td></td><td>0.451</td><td>0.551</td></tr><tr><td></td><td colspan="5">AP3D@0.5</td><td colspan="5">AP3D@0.7</td><td colspan="5">Recall@BEV0.3</td></tr><tr><td>Method</td><td>0-30</td><td>30-50</td><td>50-100</td><td>100-200</td><td>All</td><td>0-30</td><td>30-50</td><td>50-100</td><td>100-200</td><td>All</td><td>0-30</td><td>30-50</td><td>50-100</td><td>100-200</td><td>All</td></tr><tr><td>SECOND</td><td>0.950</td><td>0.514</td><td>0.323</td><td>0.052</td><td>0.336</td><td>0.393</td><td>0.041</td><td>0.024</td><td>0.001</td><td>0.066</td><td>0.987</td><td>0.920</td><td>0.858</td><td>0.650</td><td>0.807</td></tr><tr><td> $\mathrm { C L O C s ~ ( R 0 ) }$ </td><td>0.958</td><td>0.603</td><td>0.328</td><td>0.047</td><td>0.349</td><td>0.413</td><td>0.047</td><td>0.018</td><td>0.001</td><td>0.063</td><td>0.987</td><td>0.924</td><td>0.858</td><td>0.614</td><td>0.794</td></tr><tr><td> $\mathbf { C L O C s - R } \ ( \mathbf { R } 2 )$ </td><td>0.957</td><td>0.595</td><td>0.329</td><td>0.047</td><td>0.350</td><td>0.381</td><td>0.044</td><td>0.018</td><td>0.001</td><td>0.062</td><td>0.988</td><td>0.924</td><td>0.860</td><td>0.623</td><td>0.798</td></tr><tr><td> $\mathrm { R 0 } + \mathrm { r e s i d u a l } ^ { \dag }$ </td><td>0.958</td><td>0.602</td><td>0.335</td><td>0.050</td><td>0.354</td><td>0.416</td><td>0.047</td><td>0.020</td><td>0.001</td><td>0.066</td><td>0.988</td><td>0.925</td><td>0.871</td><td>0.655</td><td>0.815</td></tr><tr><td> ${ \mathrm { R } } 2 \mathrm { ~ + ~ } \mathrm { r e s i d u a l } ^ { \dagger }$ </td><td>0.958</td><td>0.599</td><td>0.336</td><td>0.050</td><td>0.354</td><td>0.405</td><td>0.045</td><td>0.020</td><td>0.001</td><td>0.066</td><td>0.988</td><td>0.927</td><td>0.872</td><td>0.659</td><td>0.817</td></tr></table>

Distance bins in meters; validation has 17,669 cars (2,474, 1,470, 6,908 and 6,777 in the four bins), and >50 pools the last two bins, the selection metric. R0 is CLOCs withou radar and R2 is CLOCs-R with the selected feature set. Fusion rows are means of 3 seeds. <sup>†</sup>Post hoc residual scoring, Eq. (5). Recall is limited by the 100-box cap (Section III-B).

Composition. The 21 validation scenes come from five drives, and 83% of the validation cars beyond 50 m (11,352 of 13,685) come from nine consecutive urban scenes of one drive, about 2.25 minutes long. Five validation scenes and one holdout scene pass through tunnels, where the camera is badly underexposed, and no training scene does. Eleven of the 13 holdout scenes come from drives that also contribute training scenes, so holdout measures generalization to unseen scenes rather than unseen drives.

Preprocessing. 2D training labels are generated as in Section II. Returns from the front long-range radar are accumulated over $M = 3$ sweeps and compensated with Eq. (1). In 33 of the 12,876 validation sweeps (0.26%), the ego poses have no timestamps, and the ego velocity is then computed with the sequence’s frame period. We checked the compensation on all 143 sequences. In the median sequence, 88%, 90% and 91% of the returns away from labeled objects are near static (below 1 m/s) in training, holdout and validation, and Doppler speeds on labeled cars differ from their labeled velocities by a median of 0.06, 0.08 and 0.18 m/s. No sequence fails both checks.

## B. Experimental Setup

Hardware and software. All runs use one 20 GB NVIDIA A100 virtual GPU (A100X-20C) on the Jetstream2 cloud, with Python 3.8, PyTorch 1.10.2, CUDA 11.3 and spconv 1.2.1. SECOND is trained with OpenPCDet [25], and Cascade R-CNN with MMDetection 2.28.2 [22].

SECOND. The network follows the OpenPCDet SECOND configuration and is trained from random initialization for 40 epochs with batch size 4, using Adam with a onecycle schedule (maximum learning rate $3 \times 1 0 ^ { - 3 } )$ and weight decay 0.01, which took about 2 h 50 min. The grid covers $[ 0 , 2 0 4 . 8 ] \times [ - 2 5 . 6 , 2 5 . 6 ] \times [ - 1 . 5 , 2 . 5 ] \mathrm { m }$ with 0.1 m voxels, and one car anchor of $4 . 2 \times 2 . 0 \times 1 . 5 5$ m is placed at two yaw angles. Augmentation uses random left-right flips, rotations of up $\mathrm { t o \pm 5 ^ { \circ } }$ (larger rotations move distant cars out of the narrow grid), and scaling by 0.95–1.05.

Cascade R-CNN. A ResNet-50 FPN Cascade R-CNN pretrained on COCO [26] is fine-tuned for 4 epochs with SGD (learning rate 0.005, batch size 4) at an input size of $1 2 8 0 \times 7 0 4$ , which took about 1 h 10 min.

Candidates. Cascade R-CNN keeps its top $K _ { 2 D } ~ = ~ 2 0 0$ boxes per image. SECOND produces 32,768 anchor-level boxes per frame, and storing all of them did not fit on our disk, so the top 2,000 by score are kept, rather than every anchor as Section II states. These contain a box with BEV IoU of at least 0.3 for 85% of the validation cars at 100–200 m, and for 59% of the holdout cars.

Fusion network. The network, inputs and loss follow Section II. It is trained for 3 epochs over the 7,753 training frames with a batch size of one frame, a one-cycle schedule (maximum learning rate $3 \times 1 0 ^ { - 3 }$ , 40% warm-up) and weight decay 0.01. The optimizer is AdamW [27], the decoupledweight-decay form of Adam. Training targets use BEV IoU and the reserved row uses −1, as Section II states, whereas the official CLOCs code uses 3D IoU and −10. Each configuration is trained with seeds 1, 2 and 3, at about 2 minutes per seed. GPU training is not bit-identical between runs: retraining one seed reproduced its scores to within 0.005. The gate was selected on holdout with R3, from $\delta _ { 0 } ~ \in ~ \{ 0 . 2 5 , 0 . 5 , 1 . 0 \}$ m and $\kappa \in \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 4 \}$ , giving $\delta _ { 0 } = 0 . 5 \mathrm { m }$ and $\kappa = 0 . 0 2 .$ and $v _ { \mathrm { m a x } } = 4 0 \mathrm { m } / \mathrm { s }$

Evaluation. Detections are evaluated with no score threshold after BEV NMS at IoU 0.1, which keeps at most 1,000 boxes before and 100 after NMS per frame. Matching is greedy in score order, and AP is all-point interpolated. Each groundtruth box and detection falls in a distance bin by its own BEV range, and only pairs in the same bin can match. Besides 3D AP at IoU 0.5 and 0.7, recall, and center-distance AP (APCD, the mean AP at 0.5, 1, 2 and 4 m [19]), we report BEV AP at IoU 0.3 (APBEV), the matching threshold of the aiMotive baselines [17], because the label shift makes strict

TABLE IV  
OUR FUSION IMPLEMENTATION ON KITTI VALIDATION (3,769 FRAMES)
<table><tr><td></td><td colspan="3">APBEV@0.3</td><td colspan="3">AP3D@0.7</td></tr><tr><td>Model</td><td>All</td><td>30-50</td><td>50-70</td><td>All</td><td>30-50</td><td>50-70</td></tr><tr><td>SECOND</td><td>0.929</td><td>0.867</td><td>0.482</td><td>0.704</td><td>0.496</td><td>0.104</td></tr><tr><td>Ours (3 seeds)</td><td>0.942</td><td>0.906</td><td>0.621</td><td>0.719</td><td>0.542</td><td>0.160</td></tr><tr><td>Authors&#x27;CLOCs</td><td>0.941</td><td>0.906</td><td>0.622</td><td>0.717</td><td>0.536</td><td>0.156</td></tr></table>

Distance bins in meters; 50–70 ends at 70.4 m.

IoU thresholds unreliable on this dataset. Recall uses BEV IoU 0.3. The selection metric, APCD beyond 50 m, pools the 50–100 and 100–200 m bins. The 100-box limit is reached on 99% of validation frames. Raising it to 500, with 2,000 boxes before NMS, changes APBEV and APCD by at most 0.003, but raises recall at 100–200 m from 0.650 to 0.695 for SECOND and from 0.659 to 0.695 for CLOCs-R with residual scoring, so recall differences between methods should be read with care.

## C. KITTI Verification

With the authors’ released weights, CLOCs reproduces the published car 3D AP of 92.75 / 81.28 / 77.70 (easy / moderate / hard) on KITTI validation in our environment. SECOND alone gives 90.82 / 79.73 / 76.87, against 90.97 / 79.94 / 77.09 in the CLOCs repository, and a two-epoch training run of the official code reaches 92.01 / 80.82 / 77.42. We then ran our own fusion implementation, the one used on aiMotive, on the authors’ KITTI candidates (3 epochs, 3 seeds). It matches the authors’ CLOCs to within 0.006 on every metric and distance bin (Table IV).

## D. Detector Baselines

Table V compares SECOND on aiMotive with SECOND on KITTI. Within KITTI’s 70.4 m range, aiMotive is level with KITTI at lenient matching: APBEV is 0.975 against 0.973 at 0–30 m, 0.847 against 0.867 at 30–50 m, and 0.738 against 0.481 at 50–70.4 m. Its lower score over 0–70.4 m (0.867 against 0.929) comes from the distance mix, since 47% of aiMotive cars in that range are at 50–70.4 m, against 9% on KITTI. The full-range score falls to 0.630 because 38% of all validation cars are at 100–200 m, where APBEV is 0.277. At strict matching the gap is large (AP3D@0.7 of 0.066 against 0.704), and much of it is the label shift: matched validation boxes are about 0.2 m too wide at up to 100 m, and the median BEV IoU is about 0.7, so APBEV falls from 0.699 at IoU 0.1 to 0.207 at IoU 0.7.

At long range, LiDAR coverage is the main limit. On holdout, 49% of cars at 100–200 m have no LiDAR points and 66% have fewer than five, and counting only cars with at least five points raises APBEV there from 0.313 to 0.817 (158 cars). Both detectors also score much higher on frames they were trained on: on 900 training frames, SECOND reaches APBEV 0.835 and Cascade R-CNN AP2D@0.5 0.803, against 0.629 and 0.499 on holdout. Cascade R-CNN reaches AP2D@0.5 of 0.431 on validation against 0.827 on KITTI, but 0.827 against

TABLE V  
SECOND ALONE ON AIMOTIVE AND KITTI VALIDATION
<table><tr><td rowspan="2">Distance</td><td colspan="3">aiMotive</td><td colspan="3">KITTI</td></tr><tr><td>BEV@0.3</td><td>APCD</td><td>3D@0.7</td><td>BEV@0.3</td><td>APCD</td><td>3D@0.7</td></tr><tr><td>0-30m</td><td>0.975</td><td>0.909</td><td>0.393</td><td>0.973</td><td>0.971</td><td>0.854</td></tr><tr><td>30–50m</td><td>0.847</td><td>0.664</td><td>0.041</td><td>0.867</td><td>0.861</td><td>0.496</td></tr><tr><td>50–70.4 m</td><td>0.738</td><td>0.628</td><td>0.044</td><td>0.481</td><td>0.467</td><td>0.104</td></tr><tr><td>0-70.4 m</td><td>0.867</td><td>0.748</td><td>0.158</td><td>0.929</td><td>0.925</td><td>0.704</td></tr><tr><td>50-100 m</td><td>0.725</td><td>0.615</td><td>0.024</td><td></td><td></td><td></td></tr><tr><td>100–200 m</td><td>0.277</td><td>0.280</td><td>0.001</td><td></td><td></td><td></td></tr><tr><td>All</td><td>0.630</td><td>0.556</td><td>0.066</td><td>0.929</td><td>0.925</td><td>0.704</td></tr></table>

KITTI’s SECOND grid ends at 70.4 m.

0.812 for boxes at least 40 pixels tall, and 57% of aiMotive cars are under 25 pixels tall, against 12% on KITTI. At 100– 200 m, its candidates overlap only 32% of the validation cars at IoU 0.5.

## E. Main Results

Table III compares the fusion models with SECOND alone. Camera-LiDAR CLOCs (R0) improves on SECOND up to 50 m, most clearly in APCD (0.664 to 0.693 at 30–50 m) and in AP3D@0.5 at 30–50 m (0.514 to 0.603). Beyond 50 m it loses: APCD there falls from 0.458 to 0.424, and fullrange APBEV from 0.630 to 0.599. On KITTI the same code gains most at long range, and on aiMotive the loss also appears within KITTI’s range (APBEV at 50–70.4 m falls from 0.738 to 0.704). With the selected feature set R2, radar raises APBEV at 30–50 m from 0.857 to 0.870 but leaves the selection metric almost unchanged (0.428 against 0.424), and both fusion models stay below SECOND beyond 50 m.

Residual scoring (post hoc). At 100–200 m, CLOCs ranks SECOND’s boxes worse than SECOND’s own score does (AUC 0.873 against 0.909 for separating real cars from false detections). After this analysis, and therefore after the validation split had been scored, we tested six changes on holdout under a rule fixed in advance (Section III-G). Only one was adopted: the fusion network corrects SECOND’s score instead of replacing it,

$$
\hat { s } _ { j } = \mathrm { l o g i t } \left( s _ { j } ^ { 3 D } \right) + \operatorname* { m a x } _ { i } f _ { \theta } ( \mathbf { T } _ { i j } ) ,\tag{5}
$$

where the last layer of $f _ { \theta }$ is initialized to zero, so that training starts from SECOND’s ranking. This adds no parameters. Residual scoring recovers most of the loss beyond 50 m, where APCD rises to 0.448 without radar and to 0.451 with R2. APBEV beyond 50 m (0.524) and over the full range (0.630) then match or exceed SECOND. On top of residual scoring, radar adds little: 0.003 in APCD beyond 50 m, and 0.013 in APBEV at 30–50 m (0.861 to 0.874). SECOND keeps a lead in APCD beyond 50 m (0.458 against 0.451) and at 50–100 m, but Section III-H shows that this lead is not significant.

## F. Comparison with the aiMotive Baselines

Table VI places our models next to the baselines released with aiMotive [17]. These use different networks (a VoxelNet LiDAR branch and BEVFusion-style fusion), evaluate four object categories class-agnostically with Hungarian matching, and score a region that extends behind the vehicle, while our models score cars only, inside the front camera’s view, with greedy matching. The numbers therefore indicate the expected range rather than a ranking. Our SECOND scores below their LiDAR baseline on highways (0.639 against 0.755), slightly above it in urban scenes (0.633 against 0.608), and below it beyond 75 m in both (0.426 and 0.405 against 0.526 and 0.470). Their radar fusion raises AP beyond 75 m in urban scenes (0.470 to 0.525) but not on highways (0.526 to 0.522), whereas radar raises ours by at most 0.006 beyond 75 m in either condition. Their camera fusion raises full-range AP in both conditions, while camera-LiDAR CLOCs lowers ours until residual scoring is added.

TABLE VI  
REFERENCE COMPARISON WITH THE AIMOTIVE BASELINES [17] (ALL-POINT BEV AP@0.3, VALIDATION)
<table><tr><td></td><td></td><td colspan="2">Full range</td><td colspan="2">&gt;75m</td></tr><tr><td>Model</td><td>Sensors</td><td>Highway</td><td>Urban</td><td>Highway</td><td>Urban</td></tr><tr><td>aiMotive</td><td>L</td><td>0.755</td><td>0.608</td><td>0.526</td><td>0.470</td></tr><tr><td>aiMotive</td><td>L+C</td><td>0.772</td><td>0.656</td><td>0.509</td><td>0.466</td></tr><tr><td>aiMotive</td><td>L+R</td><td>0.769</td><td>0.617</td><td>0.522</td><td>0.525</td></tr><tr><td>aiMotive</td><td>L+R+C</td><td>0.762</td><td>0.644</td><td>一</td><td></td></tr><tr><td>SECOND</td><td>L</td><td>0.639</td><td>0.633</td><td>0.426</td><td>0.405</td></tr><tr><td>R0</td><td> $\mathrm { L } { + } \mathrm { C }$ </td><td>0.594</td><td>0.603</td><td>0.379</td><td>0.399</td></tr><tr><td>R2</td><td> $\mathrm { L + R + C }$ </td><td>0.599</td><td>0.613</td><td>0.385</td><td>0.403</td></tr><tr><td> $\mathrm { R 0 } + \mathrm { r e s . } ^ { \dagger }$ </td><td> $\mathrm { L } { + } \mathrm { C }$ </td><td>0.635</td><td>0.623</td><td>0.435</td><td>0.418</td></tr><tr><td> ${ \mathrm { R } } 2 + { \mathrm { r e s . } } ^ { \dagger }$ </td><td> $\mathrm { L + R + C }$ </td><td>0.639</td><td>0.632</td><td>0.441</td><td>0.420</td></tr></table>

The baselines’ >75 m values come from their longer-trained models, and no three-sensor value is reported there. Our rows (SECOND, R0 = CLOCs, R2 = CLOCs-R, res. = residual scoring) score cars only in daytime clear-weather scenes and are means of 3 seeds; our urban rows cover 14,625 cars, mostly from one drive.

## G. Ablation

Radar feature sets. Table VII reports every set of Table II. Beyond 50 m, all eight sets lie within 0.007 of each other on validation and within 0.006 on holdout, about the spread between seeds of one set (validation standard deviation up to 0.010). R2 (Doppler speed alone) was selected under the rule of Section II-C, with 0.4554 on holdout, but R7 (0.4550) and R3 (0.4547) are effectively tied with it. A retraining of R2 during tuning scored 0.4545, which would have ranked it below both, so the selection is not stable and does not show that Doppler speed is the best single cue. More features do not help: R7, with all five, is no better than R2 on either split. The consistent radar effect is at 30–50 m, where every set raises validation APBEV from 0.857 to between 0.866 and 0.871, and support alone (R1) already reaches 0.866. At strict closerange matching, the sets with Doppler speed lose AP3D@0.7 at 0–30 m on validation (0.375–0.386 against 0.413 for R0). This does not repeat on holdout, however, where R5, which has no Doppler speed, loses the most (0.465 against 0.492).

Camera removed. With all 2D candidates removed at test time, APCD beyond 50 m on validation rises for both R0 (0.424 to 0.444) and R2 (0.428 to 0.448). On holdout, it rises for R0 (0.450 to 0.460) but falls for R2 (0.455 to 0.439). The reserved row is almost untrained, because only 78 of about 50,000 confident training candidates lack an overlapping 2D box. The test therefore describes the camera’s effect on these models at long range, not a model trained to work without a camera, and it does not cover night driving, which our subset excludes.

TABLE VII  
RADAR FEATURE SETS (TABLE II): ACCURACY AND ADDED LATENCY
<table><tr><td></td><td colspan="2">APCD &gt;50 m</td><td>BEV@0.3</td><td>3D@0.7</td><td>Latency</td></tr><tr><td>Set</td><td>Holdout</td><td>Val.</td><td>30-50 m</td><td>0-30m</td><td>(ms)</td></tr><tr><td>SECOND</td><td>0.4601</td><td>0.458</td><td>0.847</td><td>0.393</td><td>一</td></tr><tr><td>R0</td><td>0.4504</td><td>0.424</td><td>0.857</td><td>0.413</td><td>1.5</td></tr><tr><td>R1</td><td>0.4509</td><td>0.424</td><td>0.866</td><td>0.417</td><td>4.8</td></tr><tr><td>R2 (selected)</td><td>0.4554</td><td>0.428</td><td>0.870</td><td>0.381</td><td>19.0</td></tr><tr><td>R3</td><td>0.4547</td><td>0.426</td><td>0.867</td><td>0.384</td><td>19.2</td></tr><tr><td>R4</td><td>0.4542</td><td>0.427</td><td>0.868</td><td>0.375</td><td>44.8</td></tr><tr><td>R5</td><td>0.4495</td><td>0.421</td><td>0.866</td><td>0.411</td><td>21.5</td></tr><tr><td>R6</td><td>0.4536</td><td>0.422</td><td>0.870</td><td>0.386</td><td>32.0</td></tr><tr><td>R7</td><td>0.4550</td><td>0.423</td><td>0.871</td><td>0.384</td><td>62.1</td></tr></table>

Means of 3 seeds. Holdout chose the set, and validation played no part. The BEV@0.3 and 3D@0.7 columns are on validation. Latency is the median per frame of the radar features (one CPU thread) and the fusion forward pass (GPU), excluding the detectors.

TABLE VIII  
POST HOC FUSION CHANGES, SELECTED ON HOLDOUT (APCD >50 M, MEAN OF 3 SEEDS)
<table><tr><td>Change to R2</td><td>Holdout</td><td>Adopted</td></tr><tr><td>None (R2 retrained)</td><td>0.4545</td><td></td></tr><tr><td>Far-range targets: positive at BEV IoU ≥0.5, negative ≤0.3 beyond 50 m</td><td>0.4433</td><td>No</td></tr><tr><td>Residual scoring, Eq. (5)</td><td>0.4619</td><td>Yes</td></tr><tr><td>SECOND&#x27;s score as a scaled logit input</td><td>0.4502</td><td>No</td></tr><tr><td>M = 5 sweeps with motion correction</td><td>0.4552</td><td>No</td></tr><tr><td>M = 10 sweeps with motion correction</td><td>0.4550</td><td>No</td></tr><tr><td>Median radar cross-section as an extra channel</td><td>0.4529</td><td>No</td></tr></table>

Adoption bar 0.4582; SECOND scores 0.4601 on holdout.

Gate. Across the nine gates, holdout APCD beyond 50 m spans 0.4522 to 0.4551, about the spread between the seeds of one gate (0.452–0.459). Every gate beats R0 (0.450), and none reaches SECOND (0.460), so the results are not sensitive to the gate within this grid.

Fusion changes (post hoc). Table VIII lists the six changes tried after the as-designed results. Each was tested alone against R2 on holdout and adopted only if its mean beat R2’s by more than the standard deviation of R2’s three seeds, a bar of 0.4582 fixed before the runs. The winner was scored once on validation and also trained without radar. Only residual scoring passed. Far-range targets lowered the score, and neither more radar sweeps with motion correction nor a radar crosssection channel raised it.

Runtime. Radar features dominate the added latency (Table VII), with medians of 3.7 to 61.0 ms per frame in unoptimized single-threaded Python, against 1.1 to 1.5 ms for the fusion forward pass. With batch size one on 200 validation frames, the detectors themselves take a median of 20.9 ms (SECOND, from GPU transfer to the top-2,000 candidates, with voxelization done by the data loader) and 50.6 ms (Cascade R-CNN at $1 2 8 0 \times 7 0 4$ , including preprocessing but not image decoding) per frame, so the 19.0 ms that R2 adds compares with 71.5 ms for the two detectors run one after the other.

TABLE IX  
DIFFERENCES WITH 95% SCENE-BOOTSTRAP INTERVALS (APCD >50 M UNLESS STATED)
<table><tr><td>Comparison</td><td>Validation</td><td>Holdout</td></tr><tr><td>R0 – SECOND</td><td></td><td>-.034 [−.057, -.011] -.010 [−.023, +.008]</td></tr><tr><td>R0 + residual - R0</td><td>+.024  $[ + . 0 1 8 , + . 0 3 3 ]$ </td><td>+.008  $[ + . 0 0 4 , + . 0 1 0 ]$ </td></tr><tr><td> ${ \mathrm { R } } 2 \mathrm { ~ + ~ } \mathrm { r e s i d u a l ~ - ~ } { \mathrm { R } } 0 \mathrm { ~ + ~ }$  residual (radar)</td><td>+.003 [+.000, +.005] +.004</td><td> $\left[ + . 0 0 2 , + . 0 0 6 \right]$ </td></tr><tr><td>R2 + residual - SECOND</td><td>-.007[−.026, +.012] +.002 [−.010, +.018]</td><td></td></tr><tr><td>Radar with residual,</td><td></td><td>+.014 [−.002, +.028] +.009 [+.002, +.017]</td></tr><tr><td>APBEV@0.3, 30–50 m R2 + residual − SECOND, APCD, 30–50 m</td><td>+.040 [+.016, +.063] +.010 [−.005, +.022]</td><td></td></tr></table>

## H. Analysis

Uncertainty. Table IX gives 95% intervals from resampling scenes with replacement (1,000 resamples), with the trained models held fixed and their three seeds averaged. CLOCs’ loss to SECOND beyond 50 m and the gain from residual scoring are clear on validation. CLOCs-R with residual scoring and SECOND are statistically tied. Radar’s effect beyond 50 m is small but positive on both splits. It is, however, smaller than the variation between training seeds (standard deviation about 0.004), and the intervals treat scenes as independent although many share a drive, so they are likely too narrow.

Why radar adds little at range. At 100–200 m on validation, 62% of the real cars among SECOND’s detections have a radar return in their gate, but so do 39% of the false detections, and the radar features alone separate the two only weakly (AUC 0.615 for support and 0.664 for Doppler speed, against 0.909 for SECOND’s score). A logistic model of SECOND’s score with and without radar, scored on scenes it was not fit on, gains at most 0.008 AUC there. The trained R3 networks also barely use radar at that range: replacing the radar inputs with “no return” changes the fused logit of real cars by about +0.5 at 30–50 m but by −0.02 at 100–200 m. Training offers little signal there, since only 7,232 of 136,332 training positives (5%) are at 100–200 m, and SECOND looks nearly perfect on its own training frames at that range (AUC 0.984, against 0.909 on validation).

Failure modes. Four effects limit the results. First, most of SECOND’s remaining lead comes from the tunnel scenes, where the camera is dark. Beyond 50 m there, APCD is 0.461 for SECOND against 0.431 for CLOCs-R with residual scoring (and 0.346 for R0), while without the tunnel scenes the two score 0.458 and 0.456. Second, camera evidence is weakest where fusion was meant to help: only 8% of real cars at 100–200 m on validation have a well-aligned 2D box (2D-3D IoU of at least 0.5) scoring above 0.5, against 97% at 0–30 m. Third, rescoring cannot recover cars that SECOND never proposed: among the 2,000 kept candidates, 15% of the validation cars at 100–200 m and 41% of those on holdout have none with BEV IoU of at least 0.3. Fourth, strict 3D AP is bounded by the label shift, which rescoring cannot correct because box geometry comes from SECOND.

Disclosure. The detectors were frozen, and the gate and the feature set were selected, using holdout only. Residual scoring was proposed after an analysis of validation results, however, so validation is not an untouched test set for it, and a clean estimate would need new data. For this reason, Table III reports the as-designed and post hoc models side by side.

## AI USAGE STATEMENT

Claude Opus 5.5 was used to format our original draft made in Google Docs to conform to the IEEE LaTeX template provided on Canvas. It was also used to generate LaTeX code for the comparison table and verify that this document followed the requirements outlined in Canvas for this phase of the project.

## REFERENCES

[1] Y. Yan, Y. Mao, and B. Li, “SECOND: Sparsely embedded convolutional detection,” Sensors, vol. 18, no. 10, p. 3337, 2018.

[2] S. Vora, A. H. Lang, B. Helou, and O. Beijbom, “PointPainting: Sequential fusion for 3D object detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 4604–4612.

[3] S. Pang, D. Morris, and H. Radha, “CLOCs: Camera-LiDAR object candidates fusion for 3D object detection,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2020, pp. 10 386–10 393.

[4] ——, “Fast-CLOCs: Fast camera-LiDAR object candidates fusion for 3D object detection,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2022, pp. 187–196.

[5] Z. Liu, H. Tang, A. Amini, X. Yang, H. Mao, D. L. Rus, and S. Han, “BEVFusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2023, pp. 2774–2781.

[6] F. Nobis, M. Geisslinger, M. Weber, J. Betz, and M. Lienkamp, “A deep learning-based radar and camera sensor fusion architecture for object detection,” in Proc. Sensor Data Fusion: Trends, Solutions, Applications (SDF), 2019, pp. 1–7.

[7] R. Nabati and H. Qi, “RRPN: Radar region proposal network for object detection in autonomous vehicles,” in Proc. IEEE Int. Conf. Image Process. (ICIP), 2019, pp. 3093–3097.

[8] ——, “CenterFusion: Center-based radar and camera fusion for 3D object detection,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2021, pp. 1527–1536.

[9] Y. Long, A. Kumar, D. Morris, X. Liu, M. Castro, and P. Chakravarty, “RADIANT: Radar-image association network for 3D object detection,” in Proc. AAAI Conf. Artif. Intell., vol. 37, no. 2, 2023, pp. 1808–1816.

[10] Y. Kim, J. Shin, S. Kim, I.-J. Lee, J. W. Choi, and D. Kum, “CRN: Camera radar net for accurate, robust, efficient 3D perception,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 17 615–17 626.

[11] N. Baumann, M. Baumgartner, E. Ghignone, J. Kuhne, T. Fischer, Y.-H.¨ Yang, M. Pollefeys, and M. Magno, “CR3DT: Camera-RADAR fusion for 3D detection and tracking,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2024, pp. 4926–4933.

[12] J. Kim, M. Seong, G. Bang, D. Kum, and J. W. Choi, “RCM-Fusion: Radar-camera multi-level fusion for 3D object detection,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 18 236–18 242.

[13] B. Yang, R. Guo, M. Liang, S. Casas, and R. Urtasun, “RadarNet: Exploiting radar for robust perception of dynamic objects,” in Proc. Eur. Conf. Comput. Vis. (ECCV). Springer, 2020, pp. 496–512.

[14] X. Chen, T. Zhang, Y. Wang, Y. Wang, and H. Zhao, “FUTR3D: A unified sensor fusion framework for 3D detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2023, pp. 172–181.

[15] A. Palffy, E. Pool, S. Baratam, J. F. P. Kooij, and D. M. Gavrila, “Multiclass road user detection with 3+1D radar in the View-of-Delft dataset,” IEEE Robot. Autom. Lett., vol. 7, no. 2, pp. 4961–4968, 2022.

[16] D.-H. Paek, S.-H. Kong, and K. T. Wijaya, “K-Radar: 4D radar object detection for autonomous driving in various weather conditions,” in Proc. Neural Inf. Process. Syst. (NeurIPS) Datasets Benchmarks Track, 2022.

[17] T. Matuszka, I. Barton, A. Butykai, P. Hajas, D. Kiss, D. Kov<sup>´</sup> acs,´ S. Kunsagi-M´ at´ e, P. Lengyel, G. N´ emeth, L. Pet´ o, D. Ribli, D. Szeghy,˝ S. Vajna, and B. V. Varga, “aiMotive dataset: A multimodal dataset for robust autonomous driving with long-range perception,” in Proc. Int. Conf. Learn. Represent. (ICLR) Workshop Scene Represent. Auton. Driving (SR4AD), 2023.

[18] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? The KITTI vision benchmark suite,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2012, pp. 3354–3361.

[19] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A multimodal dataset for autonomous driving,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 11 621–11 631.

[20] S. Yao, R. Guan, X. Huang, Z. Li, X. Sha, Y. Yue, E. G. Lim, H. Seo, K. L. Man, X. Zhu, and Y. Yue, “Radar-camera fusion for object detection and semantic segmentation in autonomous driving: A comprehensive review,” IEEE Trans. Intell. Veh., vol. 9, no. 1, pp. 2094– 2128, 2024.

[21] Z. Cai and N. Vasconcelos, “Cascade R-CNN: Delving into high quality object detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018.

[22] K. Chen, J. Wang, J. Pang, Y. Cao, Y. Xiong, X. Li, S. Sun, W. Feng, Z. Liu, J. Xu, Z. Zhang, D. Cheng, C. Zhu, T. Cheng, Q. Zhao, B. Li, X. Lu, R. Zhu, Y. Wu, J. Dai, J. Wang, J. Shi, W. Ouyang, C. C. Loy, and D. Lin, “MMDetection: Open MMLab detection toolbox and benchmark,” arXiv:1906.07155, 2019.

[23] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2980–2988.

[24] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.

[25] OpenPCDet Development Team, “OpenPCDet: An open-source toolbox for 3D object detection from point clouds,” https://github.com/ open-mmlab/OpenPCDet, 2020.

[26] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick, “Microsoft COCO: Common objects in´ context,” in Proc. Eur. Conf. Comput. Vis. (ECCV). Springer, 2014, pp. 740–755.

[27] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.