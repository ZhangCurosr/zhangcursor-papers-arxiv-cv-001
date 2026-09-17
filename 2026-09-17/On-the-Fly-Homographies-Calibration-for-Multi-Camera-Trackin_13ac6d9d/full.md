# On-the-Fly Homographies Calibration for Multi-Camera Tracking

David Voihanski and Mor Sinai

Juganu

Or Yehuda, Israel

{davidv, mors}@juganu.com

Ben Zion Bobrovsky Tel Aviv University Israel bobrov@tauex.tau.ac.il

© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

Abstract—Precise multi-camera tracking traditionally relies on rigorous 3D site calibration, yet this requirement is often operationally impossible in large-scale deployments. Privacy regulations frequently prohibit recording video for offline calibration; limited bandwidth precludes synchronizing high-resolution streams from hundreds of cameras; and covering immense physical sites with calibration targets is logistically infeasible.

We present a multi-camera homography calibration system designed to overcome these barriers through ”on-the-fly” geometric refinement. Starting from coarse manual homographies, we introduce a centroid-based projection optimization (PO) that continuously aligns the ground-plane geometry using live detec tion streams. Because PO operates asynchronously on alreadytransmitted, lightweight metadata, it adds zero computational latency to the real-time tracker. This allows the system to adapt automatically to camera movements or environmental changes without human intervention. This optimized geometry feeds a multi-camera bird’s-eye-view (BEV) tracker that fuses detections and unifies trajectories across zones. Crucially, by operating strictly on live anonymous metadata, our solution ensures a privacy-safe, zero-overhead, and resilient tracking pipeline that maintains global consistency in dynamic environments where static, recorded-video calibration is impossible.

## I. INTRODUCTION

Multi-camera tracking enables global identity consistency across camera networks, improving robustness under occlusion and expanding coverage. However, many multi-camera systems assume accurate 3D calibration (intrinsics/extrinsics, surveyed landmarks, or specialized procedures). In real-world deployments, relying on such precision is not merely expensive; it is often operationally infeasible [5].

Three primary barriers prevent the adoption of traditional calibration in large-scale facilities. First, privacy regulations (e.g., GDPR) often strictly prohibit the recording and storage of video footage required for offline calibration bundles. Second, bandwidth constraints in large retail or industrial networks preclude the transmission and synchronization of high-resolution video streams from hundreds of cameras to a central server. Third, covering thousands of square meters with physical calibration targets (e.g., checkerboards) is logistically impossible without disrupting site operations.

We target deployments where operators can provide only manual homographies from image coordinates to a site map. Such homographies are fast to obtain but are typically imperfect due to occlusions, imperfect floor plans, and extrapolation errors. Crucially, static homographies fail when cameras are inevitably nudged or moved during daily operations.

Key idea. We propose a system that requires neither video recording nor physical site access. By refining initial manual homographies using a projection optimization (PO) procedure that operates on live detection metadata, we align multi-camera projections in BEV space ”on-the-fly.” This approach ensures privacy compliance and allows the system to autonomously heal geometric alignment errors caused by camera movement.

a) Contributions:

• A zero-overhead projection optimization method that refines manual homographies asynchronously. By running exclusively on lightweight, clustered cross-view detection metadata, it continuously adapts to environmental changes without adding computational latency to the real-time pipeline.

• A global BEV multi-camera tracker that associates multi-view observations using combined motion and appearance cues before fusing them into a shared geometric state, operating purely on privacy-safe metadata.

• A trajectory-level cross-camera unification stage that merges duplicate global tracks using temporal alignment, motion dynamics, and appearance costs.

b) Scope of Tracking Framework: It is important to note that while we present an end-to-end tracking pipeline, our core scientific contribution is the on-the-fly Projection Optimization (PO) calibration module, not a novel tracking architecture. The custom BEV tracker described herein is provided strictly as an evaluation vehicle to demonstrate the tangible improvements in trajectory alignment resulting from our geometric refinement. Because PO operates purely on the coordinate mapping layer, any multi-camera tracker that relies on spatial gating or trajectory alignment will inherently benefit from the restored geometric fidelity our module provides.

## II. RELATED WORK

A well-structured multi-camera tracking system must balance geometric precision with deployment feasibility. Below, we contextualize our approach against the three primary calibration paradigms in the tracking literature, highlighting their operational shortcomings and how our proposed system explicitly addresses them.

a) Multi-Camera Tracking with 3D Calibration: Previous work often assumes an accurate 3D camera calibration to triangulate targets or fuse observations in a shared 3D space [10], [11]. Obtaining such precision in the wild typically requires costly manual setup or specialized procedures, such as surveying architectural landmarks or placing physical checkerboards across the site [5]. Failure Mode: These systems are inherently brittle in active surveillance environments. If a camera is nudged or moved during daily operations, the rigorous static calibration is immediately invalidated, requiring a costly site visit to repeat the setup. Our Solution: Our system abandons the need for static, surveyed 3D calibration. By continuously running our Projection Optimization (PO) on live metadata, the system autonomously heals geometric alignment errors on-the-fly, gracefully recovering from camera movements without human intervention.

b) Planar (Homography) Multi-Camera Tracking: To bypass full 3D calibration, projecting multi-view observations to a common ground plane (bird’s-eye view) using planar homographies is a standard approach for fusing data [6], [4]. Failure Mode: These systems typically assume the provided homographies are highly precise. In practice, manual homographies frequently exhibit residual misalignment due to extrapolation errors, lens distortion, and imperfect floor plans [5]. This residual spatial scatter is catastrophic for tracking: it causes multi-view observations of the same target to fail spatial distance gating, severely fragmenting trajectories and causing rampant identity switches. Our Solution: We treat manual homographies merely as a coarse warm start. By dynamically clustering cross-view Re-ID correspondences, our centroid-anchored optimizer continuously pulls the projection matrices into alignment. This drastically reduces spatial scatter and restores the integrity of the tracking system’s spatial gating logic.

c) Weak Calibration and Self-Calibration: Methods for learning cross-view alignment or self-calibration from data typically attempt to deduce geometry by observing moving targets over time [3]. Failure Mode: These approaches generally require strong environmental assumptions (e.g., vehicles moving in perfectly straight lines to find vanishing points), centralized video processing, and long sequences of recorded footage to converge. While effective for traffic surveillance, these assumptions fail completely in retail or pedestrian environments where targets move erratically. Furthermore, in largescale deployments, strict privacy regulations (e.g., GDPR) and bandwidth constraints prohibit the recording and transmission of the raw video required for these offline calibration algorithms. Our Solution: Our approach differs by being fully online and privacy-compliant. It relies exclusively on lightweight, anonymized detection metadata to refine the geometry via a stable BEV consistency objective. This eliminates the need for video transmission entirely, making it ideal for bandwidth constrained, privacy-sensitive networks.

## III. SYSTEM OVERVIEW

At each time step, each camera produces detections (bounding boxes, confidence scores, and appearance embeddings). A chosen image point (e.g. leg point / bottom-center) is mapped to BEV via a per-camera projection. Crucially, the edge nodes transmit only this lightweight metadata to the central tracker, not video frames. This design respects strict privacy policies regarding video persistence and minimizes bandwidth usage. We then (i) match per-camera detections to shared BEV tracks, (ii) fuse the matched spatial detections and appearance embeddings to continuously update the global track states, and (iii) periodically merge duplicate BEV tracks via trajectory unification.

## IV. METHOD

## A. Projection from Manual Homographies

For camera $i \in \{ 1 , \ldots , N \}$ , we initialize a projective mapping (homography) $\mathbf { \bar { A } } _ { i } ^ { ( 0 ) }$ from the image coordinates to a shared BEV plane [5]. Given an image point $ { \mathbf { p } } \in \mathbb { R } ^ { 2 }$ , we compute its projection as $\mathbf { x } = \Pi ( \mathbf { p } ; \mathbf { A } _ { i } ) \in \mathbb { R } ^ { 2 }$ , where Π denotes the homogeneous projection followed by dehomogenization.

In practice, ${ \bf \ddot { A } } _ { i } ^ { ( 0 ) }$ is obtained from a small set of manual correspondences. While fast to annotate, these suffer from well-known critical limitations: (i) down-scaling reduces point placement accuracy, (ii) occlusions force landmark estimation, (iii) imperfect floor plans mismatch physical reality, (iv) floorcontact ambiguity arises due to perspective differences, (v) extrapolation errors grow significantly outside the annotated region, and (vi) lens distortion cannot be perfectly modeled by simple projective transforms. In a static system, these errors are permanent; in our system, they serve merely as a warm start for online optimization.

## B. Matching Point Acquisition

The projection optimization described in Section IV-C relies on clusters of corresponding points $\{ \mathcal { C } _ { c } \}$ observed across multiple views. To acquire these correspondences in live deployments where video recording is prohibited or infeasible, we employ an online matching method driven by visual Re-Identification (Re-ID). We run an appearance-based tracker across the camera network to associate detections based on embedding similarity.

Crucially, because projection optimization is highly sensitive to outliers, we prioritize precision over recall. We apply a strictly low matching distance threshold (or high similarity score), rejecting any ambiguous associations. This ensures that we harvest a smaller set of highly robust correspondences rather than a large volume of noisy matches, preventing false associations from corrupting the spatial alignment.

This online approach is critical for ”on-the-fly” adaptation. If a camera is accidentally nudged, the Re-ID stream continues to find correspondences between the (now shifted) view and its neighbors. The subsequent optimization step naturally pulls the projection matrix back into alignment without requiring a site visit or manual recalibration, ensuring long-term system resilience.

a) Spatial Subsampling: Since the transformation between the image plane and the ground plane is modeled as a homography (a smooth, planar mapping), dense clusters of points within a small image neighborhood provide redundant geometric constraints. To improve computational efficiency and prevent overfitting to local clusters, we apply spatial subsampling on the image plane. We enforce a configurable minimum pixel distance between selected points. This ensures a uniform distribution of constraints across the field of view without unnecessary redundancy.

## C. Projection Optimization (PO)

![](images/43aaf23f9d7aa0ec50d9747ffdd674aa3739a9c0b6b84b9e2a6048f99218bebf.jpg)  
Fig. 1. Projection inconsistency before optimization. A single person is observed simultaneously across multiple overlapping camera views (left). When mapped into the shared bird’s-eye view (bottom right) using initial manual homographies, the individual camera projections (colored circles) exhibit significant spatial scatter around their computed centroid (black dot). The black arrow illustrates the multi-view alignment error that our projection optimization aims to minimize per camera.

a) Centroid-based optimization objective: As shown in Figure 1, initial manual homographies often result in significant spatial disagreement when the same object is viewed from different perspectives. To resolve this, calibration points are grouped into clusters $\{ \mathcal { C } _ { c } \}$ , where each cluster corresponds to one physical landmark observed in two or more cameras. Using the original manual projection matrices ${ \bf A } _ { i } ^ { ( 0 ) }$ , we compute an initial BEV centroid for each cluster:

$$
\mu _ { c } ^ { ( 0 ) } = \frac { 1 } { N _ { c } } \sum _ { p \in \mathcal { C } _ { c } } \mathbf { x } _ { p } ^ { ( 0 ) } ,\tag{1}
$$

where $N _ { c } ~ = ~ | \mathcal { C } _ { c } |$ , p represents an individual image-space observation (e.g., a bounding box foot point from a specific camera) belonging to cluster $\mathcal { C } _ { c } .$ , and $\mathbf { x } _ { p } ^ { ( 0 ) }$ is its corresponding BEV projection under the original matrices.

b) Optimization over camera matrices: Because the system continuously adapts on-the-fly, the projection matrix for each camera is updated iteratively. The optimization cost at iteration k is defined as the sum of the squared distances from each projected point to the reference centroid of its cluster:

$$
\mathcal { L } ^ { ( k ) } = \sum _ { c } \sum _ { p \in \mathcal { C } _ { c } } \left\| \mathbf { x } _ { p } ^ { ( k ) } - \pmb { \mu } _ { c } ^ { ( k _ { \mathrm { r e f } } ) } \right\| ^ { 2 } .\tag{2}
$$

An optimizer such as Adam is applied to the current camera matrix parameters ${ \bf A } _ { i } ^ { ( k ) }$ to minimize this loss $\mathcal { L } ^ { ( k ) }$ across all clusters and calibration points. The centroid-based loss naturally extends from two cameras to N cameras: any number of views that contribute points to a cluster are all pulled toward the same cluster centroid in BEV space, yielding the updated matrices ${ \bf A } _ { i } ^ { ( k + 1 ) }$ for the next iteration.

c) Slowly updated centroids: To avoid trivial collapse and excessive drift, the reference centroids $\mu _ { c } ^ { \left( k _ { \mathrm { r e f } } \right) }$ used in Equation 2 are refreshed only intermittently. Initially, $\mu _ { c } ^ { ( k _ { \mathrm { r e f } } ) } =$ $\mu _ { c } ^ { ( 0 ) }$ . Every K optimization iterations we recompute:

$$
\mu _ { c } ^ { ( k _ { \mathrm { r e f , n e w } } ) } = \frac { 1 } { N _ { c } } \sum _ { p \in \mathcal { C } _ { c } } \mathbf { x } _ { p } ^ { ( k ) } .\tag{3}
$$

## D. Multi-Camera Tracking in BEV

a) Design principle: a shared global track pool in BEV: The multi-camera (MC) tracker maintains a single set of global tracks whose latent state lives in a shared bird’s-eye-view (BEV) plane. Each camera acts as an independent measurement source that can update any global track.

b) Per-camera inputs: At each timestamp t, each camera i produces a set of detections $\mathcal { D } _ { t } ^ { ( i ) } = \{ d _ { t , m } ^ { ( i ) } \} _ { m = 1 } ^ { \tilde { M _ { i } } }$ with: (i) a 2D image box and confidence score, (ii) a chosen image point $\mathbf { p } _ { t , m } ^ { ( i ) }$ that approximates ground contact (e.g. “leg point” / bottomcenter), and (iii) an appearance embedding $\mathbf { e } _ { t , m } ^ { ( i ) }$ . The image point is projected to BEV using the per-camera projection model (Section IV-A–IV-C), which yields BEV measurements $\mathbf { z } _ { t , m } ^ { ( i ) } \in \mathbb { R } ^ { 2 }$

c) Refining Leg Points via Single-View Tracking: Since our system relies on planar homographies rather than full 3D calibration, multi-camera association is highly sensitive to input noise; a few pixels of vertical jitter in the image plane can translate into large spatial displacements in the BEV. We mitigate this by feeding the multi-camera tracker with smoothed state estimates from the single-view tracker rather than raw detections. This temporal stabilization reduces projection variance and provides better-calibrated uncertainty covariances, significantly improving the recall of cross-camera matching under tight Mahalanobis gates.

## 1) Per-camera association against the shared pool:

a) Parallel matching: For each timestamp t, we first predict all global tracks to obtain $\{ \mathbf { x } _ { t | t - 1 } ^ { ( j ) } , \mathbf { P } _ { t | t - 1 } ^ { ( j ) } \} _ { j }$ . For each camera i, we then independently match its BEV measurements $\{ \mathbf { z } _ { t , m } ^ { ( i ) } \} _ { m }$ to the same global track pool. This step can be parallelized across cameras, allowing for efficient multi-sensor fusion.

b) Motion cost (Mahalanobis): For detection m at time t in camera i and track $j ,$ define innovation and covariance:

$$
\mathbf { f } _ { m } ^ { ( i , j ) } = \mathbf { z } _ { t , m } ^ { ( i ) } - \mathbf { H } \mathbf { x } _ { t \mid t - 1 } ^ { ( j ) } , \quad \mathbf { S } ^ { ( j ) } = \mathbf { H } \mathbf { P } _ { t \mid t - 1 } ^ { ( j ) } \mathbf { H } ^ { \top } + \mathbf { R } ^ { ( j ) } ,\tag{4}
$$

where $\mathbf { R } ^ { ( j ) }$ may be adapted over time (below). The motion distance is

$$
d _ { \mathrm { m o t } } ( m , j ) = ( { \bf f } _ { m } ^ { ( i , j ) } ) ^ { \top } ( { \bf S } ^ { ( j ) } ) ^ { - 1 } { \bf f } _ { m } ^ { ( i , j ) } .\tag{5}
$$

c) Appearance cost: We use Euclidean distance between the detection embedding $\mathbf { e } _ { t , m } ^ { ( i ) }$ and the track embedding $\mathbf { e } ^ { ( j ) }$ Embeddings are computed following standard Re-ID practices [9].

d) Combined cost and assignment: Since MC tracks a BEV point (not a 2D box), we do not use IoU-based fallback matching. We construct a combined cost $c ( m , j ) =$ α $d _ { \mathrm { m o t } } ( m , j ) + \beta d _ { \mathrm { a p p } } ( m , j )$ . After applying gating thresholds for both motion and total cost, we resolve the one-to-one assignment for camera i using the Hungarian algorithm [7].

2) Cross-camera fusion into the global state: After parallel matching, a global track $j$ may receive concurrent measurements from multiple cameras. We fuse these observations via sequential Kalman measurement updates [2], where the posterior state and covariance from one camera’s update immediately serve as the prior for the next. To handle heterogeneous camera error (e.g., varying projection variance or detector noise), we adapt the measurement noise covariance $\mathbf { R } ^ { ( j ) }$ online based on recent residual magnitudes. Finally, the global track’s appearance embedding $\mathbf { e } ^ { ( j ) }$ is updated via an exponential moving average (EMA) over the matched multiview detections.

## 3) Track initiation and lifecycle:

a) Unmatched detections and new track creation: After association and fusion, each camera may have unmatched detections. We reuse the same logic as single-view for new track initiation: only detections above a class-specific opening threshold can spawn new tracks. This prevents low-confidence clutter from creating global IDs. Tracks transition from tentative to confirmed on the basis of configured hit/miss logic.

## E. Trajectory-Level Cross-Camera Track Unification

To resolve temporary track fragmentation (e.g., during crosscamera handovers or severe occlusions), we run a trajectorylevel merge stage after the per-frame fusion update. We first filter candidates using a strict physical gate: tracks observed by the same camera at the same timestamp are assigned infinite cost, preventing the collapse of distinct neighboring objects into a single identity.

For admissible candidate pairs $( i , j )$ with a set of overlapping timestamps $\Omega _ { i j }$ , we aggregate their historical trajectories to assess similarity. Rather than relying on a single frame, we compute the average relative state difference $\bar { \mathbf { d } } _ { i j }$ and the average combined covariance $\bar { \Sigma } _ { i j }$ across the entire temporal overlap. The structural motion cost is then evaluated via the Mahalanobis distance $d _ { i j } ^ { \mathrm { m o t i o n } } = \bar { \mathbf { d } } _ { i j } ^ { \top } \bar { \Sigma } _ { i j } ^ { - 1 } \bar { \mathbf { d } } _ { i j }$

This motion cost is fused with a cosine distance $d _ { i j } ^ { \mathrm { { e m b } } }$ between the tracks’ appearance embeddings. Pairs exceeding configured spatial or appearance thresholds are discarded, and the remaining valid pairs are resolved via greedy assignment. Upon merging, the retained track absorbs the duplicate, updating its appearance embedding via a weighted average proportional to the number of multi-view detections each track has accumulated.

## F. Implementation Details

a) Initial Homography Estimation: The initial projection matrices ${ \bf A } _ { i } ^ { ( 0 ) }$ are established via a dedicated annotation interface that displays the camera view alongside a 2D site map. Operators select a minimum of four corresponding point pairs (anchors) between the image plane and the bird’s-eye-view (BEV) map, deliberately distributing these points as widely as possible across the field of view to maximize coverage and minimize extrapolation errors. The exact number of annotated points is strictly scene-dependent, varying based on the availability of distinct physical landmarks. To ensure a robust initialization, the tool provides a real-time interactive preview: as the operator hovers over the image, the corresponding BEV projection is actively displayed. This feedback loop allows users to iteratively add, move, or delete anchor points until a satisfactory baseline alignment is achieved.

b) Re-ID Model and Matching Policy: For cross-camera visual association in the live deployment (Section IV-B), the system utilizes a PLR-OSNet [12] backbone to extract 2560- dimensional appearance embeddings. Given a frame processing rate of 5 fps, object detections from different cameras are temporally synchronized by matching their metadata timestamps within a narrow tolerance window of ±100 ms. To prioritize precision over recall during point acquisition, cross-camera associations are evaluated using Euclidean distance and require a strict similarity threshold.

c) Outlier Rejection: To further mitigate false Re-ID matches and prevent erroneous associations from corrupting the geometric alignment, we enforce a strict, configurable spatial gating mechanism. Any projected point that yields a spatial distance greater than a defined threshold from its calculated cluster centroid in the shared bird’s-eye-view space is rejected as an outlier and excluded from the optimization constraints.

d) Optimization Parameters and Scheduling: The projection optimization is performed using the Adam optimizer. Because the BEV projection is highly sensitive to small perturbations in the homography matrix—where minor parameter shifts can cause large spatial displacements and lead to divergence— we employ a very low learning rate of $1 \times 1 0 ^ { - 5 }$ . We run the optimization for 300,000 epochs; because it optimizes only a minimal parameter space (a $3 \times 3$ matrix per camera) over lightweight metadata, this executes rapidly. To stabilize convergence, the reference cluster centroids (Equation 3) are updated every 15,000 epochs, but these updates are intentionally frozen after the first 30,000 epochs (10% of the schedule) to allow the matrix parameters to settle into a stable geometric state. Furthermore, to avoid any computational bottleneck, this optimization service runs asynchronously (e.g., once per day) on a separate machine using the already-transmitted metadata. This architecture ensures that continuous geometric refinement adds zero computational overhead or latency to the real-time multi-view tracking pipeline.

## V. EXPERIMENTS

## A. Setup and Datasets

We evaluated on three proprietary multi-camera datasets collected in real retail environments. Due to privacy and business constraints, the raw videos and annotations cannot be publicly released. All datasets are recorded at 5 FPS with fixed cameras and are time-synchronized. The ground truth is labeled with global identities across all cameras: the same person observed in multiple views is assigned the same GT track ID.

a) Dataset A (Supermarket): This dataset contains 11 cameras and spans 2:20 minutes. Cameras are installed at approximately 45<sup>◦</sup> view angle and around 3 m height. It includes 53 global ground-truth tracks.

b) Dataset B (Mall): This dataset contains 5 cameras and spans 3:50 minutes. Cameras are installed at approximately $4 0 ^ { \circ }$ view angle and around 4.5 m height. It includes 80 global ground-truth tracks.

c) Dataset C (General store): This dataset contains 8 cameras and spans 2:50 minutes. Cameras are installed at approximately $4 5 ^ { \circ }$ view angle and around 4 m height. It includes 45 global ground-truth tracks.

## B. Evaluation Metrics

We report multi-object tracking metrics (e.g., IDF1 / HOTA / MOTA) [10], [8], [1], ID switches, and calibration/alignment metrics.

• BEV alignment error: for each landmark cluster, compute its BEV centroid as the mean of all projected points in that cluster; report the mean $\ell _ { 2 }$ distance from each projected point to its cluster centroid, averaged over all points across all clusters.

• Tracking: IDF1, HOTA; ID switches.

## C. Baselines and Evaluation Strategy

Our experimental objective is to isolate and quantify the impact of geometric calibration on multi-camera association. Therefore, rather than comparing disparate tracking architectures—which conflates calibration quality with variations in data association heuristics—we evaluate the identical multicamera tracking pipeline under two different geometric conditions:

• Manual homographies (no PO). We use the operatorprovided per-camera homographies as-is to project detec tions into BEV and run the multi-camera tracker. This baseline measures how much tracking performance is lost when a standard system relies solely on raw, unrefined manual calibration.

• Manual + PO (ours). Starting from the exact same manual homographies, we run projection optimization to refine the per-camera projections, and then run the identical multicamera tracker. By keeping the tracking logic completely static, any improvement over the previous baseline is strictly attributable to better cross-camera BEV alignment.

## D. Results

a) Quantitative Analysis and the Impact of Alignment: Quantitative results for Datasets A, B, and C are presented in Table I. Across all environments, replacing raw manual homographies with our Projection Optimization (PO) yields consistent improvements in both geometric fidelity and tracking consistency.

The direct correlation between BEV alignment error and tracking stability is profoundly demonstrated when comparing the dynamics of Datasets B and C. Prior to optimization, both datasets suffered from a substantial BEV alignment error of 42 cm. However, the manifestation of this error depends heavily on scene density. Dataset B represents a busy mall environment with high pedestrian density (80 unique trajectories). In crowded scenes, a 42 cm projection discrepancy is catastrophic: it not only pushes cross-camera observations outside their correct Mahalanobis matching gate (Equation 5), but frequently pushes them into the spatial gates of neighboring pedestrians. This leads to rampant identity mixing, resulting in an extremely high identity switch count (182 IDSW) under the manual baseline. By applying our centroid-anchored PO, the alignment error in Dataset B is reduced threefold to 14 cm. This spatial convergence resolves the identity collisions, directly driving the dramatic reduction in ID switches down to 38, while boosting IDF1 from 0.717 to 0.892.

Interestingly, Dataset C presents a different but equally important insight. Despite starting with the same 42 cm error, its lower target density (a general store with 45 trajectories) means that spatial offsets rarely caused projections to collide with neighboring identities. Consequently, the raw ID switch count remained relatively stable (25 to 24). However, the 42 cm error still caused the tracker to drop associations and break trajectories. By reducing the alignment error from 42 cm to 24 cm, PO ensures that targets maintain their correct global identity for much longer durations without fragmenting, which is reflected in the significant improvements in both IDF1 (0.806 to 0.906) and HOTA (0.683 to 0.810). This confirms that mitigating projection scatter is the primary mechanism for maintaining global trajectory purity, regardless of scene density.

TABLE I  
QUANTITATIVE TRACKING AND ALIGNMENT RESULTS.
<table><tr><td>Dataset Method</td><td></td><td>IDF1 ↑ HOTA ↑ IDSW ↓ AlignErr↓</td><td></td><td></td><td></td></tr><tr><td rowspan="2">A</td><td>Manual</td><td>0.938</td><td>0.864</td><td>7</td><td>37 cm</td></tr><tr><td>+ PO (ours)</td><td>0.966</td><td>0.926</td><td>0</td><td>20 cm</td></tr><tr><td rowspan="2">B</td><td>Manual</td><td>0.717</td><td>0.544</td><td>182</td><td>42 cm</td></tr><tr><td>+ PO (ours)</td><td>0.892</td><td>0.783</td><td>38</td><td>14 cm</td></tr><tr><td rowspan="2">C</td><td>Manual</td><td>0.806</td><td>0.683</td><td>25</td><td>42 cm</td></tr><tr><td>+ PO (ours)</td><td>0.906</td><td>0.810</td><td>24</td><td>24 cm</td></tr></table>

## VI. DISCUSSION AND LIMITATIONS

a) Planar Ground Assumption: Our homography-based formulation strictly assumes that all targets move on a dominant ground plane (z = 0). Although this approximation holds for the majority of retail and indoor environments, the projection model is inherently limited in scenes with significant elevation changes, such as ramps or split-level flooring. In such nonplanar scenarios, a single homography cannot accurately map foot points to the bird’s-eye view. Future work could address this by modeling complex sites as piecewise-planar spaces, or by replacing the rigid planar matrix with a lightweight neural projection model capable of learning non-linear, terrain-aware mappings directly from the sparse detection metadata.

![](images/f255dbcf143a8d57e26b3580548a8fcc647ae319f1922233b9e2406e073e0e65.jpg)  
(a) Before Optimization  
(b) After Optimization

Fig. 2. Detailed anatomy of a single spatial cluster before (a) and after (b) Projection Optimization. In this instance, the blue point represents a target’s projection from Camera A, while the yellow point represents the exact same target at the same timestamp from Camera B. The central black point designates the computed cluster centroid, with the grey lines visualizing the spatial distance from each camera’s projection to this shared center. The number “2” denotes that this specific cluster fuses observations from two overlapping cameras. Crucially, while this highlights a single synchronized instance, the full-scale maps (Figure 3) plot all clusters generated across the entire video timeline for all people. Because the optimization is agnostic to time and global identity—focusing purely on pulling corresponding multiview points toward their local centroids—these macroscopic maps effectively visualize the complete geometric input space and convergence objective of the PO algorithm.  
![](images/0b44d4efc4a904b3e177bd99e159cadc60176430bdf52a4049f934d1ddb06b95.jpg)  
(a) Before Optimization

![](images/1f5aa711f42250afba88175730609bc60cbb2e83c6ed7b258d94c5b81f358a09.jpg)  
(b) After Optimization  
Fig. 3. Global BEV projection consistency across the entire video timeline before (a) and after (b) optimization.

b) Dependence on View Overlap: The efficacy of Projection Optimization (PO) relies on the existence of a connected graph of overlapping views. The optimization constraints are derived entirely from joint observations—either shared static landmarks or clustered detections visible in multiple cameras. If a camera is spatially isolated (i.e., it shares no common field of view with the network), its projection parameters cannot be geometrically refined relative to the global frame. In these disjoint cases, the system safely falls back to the initial manual calibration, relying more on the visual Re-ID embeddings rather than spatial gating for cross-zone associations.

c) Initialization Sensitivity: Finally, centroid-based optimization is designed as a local refinement step rather than a global geometric solver. It assumes that the initial manual homographies provide a roughly correct starting point regarding scale and orientation. If the initial manual input is grossly incorrect (e.g., flipped axes or massive scale errors), the optimization may converge to a degenerate local minimum. However, in practice, the real-time visual feedback provided by our annotation UI (Section IV-F) effectively prevents these severe initialization errors, ensuring the starting geometry is well within the convergence basin of the PO algorithm.

## VII. CONCLUSION

We presented a calibration-light multi-camera tracking framework that successfully bridges the gap between geometric precision and real-world deployment constraints. By introducing a centroid-anchored Projection Optimization (PO) method, we demonstrated that coarse, manual planar homographies can be dynamically refined into a robust spatial coordinate system using exclusively live, anonymous detection metadata. This continuous, on-the-fly alignment eliminates multi-view projection scatter, restoring the integrity of spatial distance gating and significantly reducing global identity switches.

Furthermore, because the optimization operates asynchronously on lightweight metadata, it completely decouples geometric self-healing from the real-time tracking loop, incurring zero computational overhead. Ultimately, our approach yields a highly scalable, privacy-compliant pipeline capable of maintaining stable global trajectories in dynamic environments where traditional static 3D calibration and centralized video processing are fundamentally prohibitive or infeasible.

## REFERENCES

[1] Keni Bernardin and Rainer Stiefelhagen. Evaluating multiple object tracking performance: the clear mot metrics. EURASIP Journal on Image and Video Processing, 2008(1):246309, 2008.

[2] Robert Grover Brown and Patrick YC Hwang. Introduction to random signals and applied kalman filtering: with matlab exercises and solutions. Introduction to random signals and applied Kalman filtering: with MATLAB exercises and solutions, 1997.

[3] Marketa Dubsk´ a, Adam Herout, Roman Jur´ anek, and Jakub Sochor. Fully´ automatic roadside camera calibration for traffic surveillance. IEEE Transactions on Intelligent Transportation Systems, 16(3):1162–1171, 2014.

[4] Francois Fleuret, Jerome Berclaz, Richard Lengagne, and Pascal Fua. Multicamera people tracking with a probabilistic occupancy map. IEEE transactions on pattern analysis and machine intelligence, 30(2):267–282, 2008.

[5] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003.

[6] Saad M Khan and Mubarak Shah. Tracking multiple occluding people by localizing on multiple scene planes. IEEE transactions on pattern analysis and machine intelligence, 31(3):505–519, 2008.

[7] Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83–97, 1955.

[8] Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe, and Bastian Leibe. Hota: A higher order metric´ for evaluating multi-object tracking. International journal of computer vision, 129(2):548–578, 2021.

[9] Hao Luo, Youzhi Gu, Xingyu Liao, Shenqi Lai, and Wei Jiang. Bag of tricks and a strong baseline for deep person re-identification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, 2019.

[10] Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multicamera tracking. In European conference on computer vision, pages 17–35. Springer, 2016.

[11] Zheng Tang, Milind Naphade, Ming-Yu Liu, Xiaodong Yang, Stan Birchfield, Shuo Wang, Ratnesh Kumar, David Anastasiu, and Jenq-Neng Hwang. Cityflow: A city-scale benchmark for multi-target multi-camera vehicle tracking and re-identification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8797–8806, 2019.

[12] Ben Xie, Xiaofu Wu, Suofei Zhang, Shiliang Zhao, and Ming Li. Learning diverse features with part-level resolution for person re-identification. In Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pages 16–28. Springer, 2020.