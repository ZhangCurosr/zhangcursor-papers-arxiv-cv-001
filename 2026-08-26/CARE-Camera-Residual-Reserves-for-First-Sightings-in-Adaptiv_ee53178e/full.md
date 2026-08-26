# CARE: Camera-Residual Reserves for First Sightings in Adaptive LiDAR Sensing

Jiachen Gong<sup>1</sup> , Yun Li<sup>1</sup> , Ehsan Javanmardi<sup>1</sup> , Wencan Mao<sup>2</sup> , and Manabu Tsukada<sup>1</sup>

<sup>1</sup> The University of Tokyo, Tokyo, Japan {jiachengong,li-yun,ejavanmardi,mtsukada}@g.ecc.u-tokyo.ac.jp 2 National Institute of Informatics, Tokyo, Japan wencan\_mao@nii.ac.jp

Abstract. Adaptive LiDAR scanning, which concentrates a limited sensing budget on informative regions of interest based on past object tracks, is becoming a key enabler in autonomous driving by lowering data volume while maintaining detection accuracy. However, existing scanning policies face three key challenges. First, history-driven approaches depend heavily on past object tracks, leading to delayed or missed detection of unseen objects. Second, methods leveraging random or uniform sampling outside the predicted regions have no awareness of where new objects appear. Third, camera-guided alternatives spend their budget on all camera detections, resampling objects that are already covered, which costs overall recall in crowded scenes and detection range when budgets are scarce. To address these limitations, this paper introduces the CAmera-REsidual reserve (CARE), a training-free allocation rule that reserves part of a fixed ray budget for the directions of current camera detections that the track forecasts cannot explain, while the remaining budget follows the base history policy and unused reserve returns to a shared random floor. The paper makes three key contributions: First, a leakage-free ray-budget evaluation on nuScenes (150 scenes, 4,148 events) that measures the first-sighting loss of history-driven scanning, with a strict-causal variant using the preceding keyframe. Second, CARE raises first-sighting recall by 5.2, 5.2, and 4.3 points at 10%, 20%, and 35% budgets over the history policy with paired intervals excluding zero; the camera cue drives this gain, and the first-sighting versus overall trade-of is reported openly as a budget-dependent Pareto choice. Third, a safetybounded forgetting module that releases budget from receding or static tracks beyond a speed-dependent guard distance; at tight budgets forgetting without the guard significantly harms near-field recall, and the guard is what keeps forgetting safe at the tightest budget. The pipeline runs end to end on a real vehicle and, in a closed-loop simulation, detects an occluded pedestrian earlier and brakes more reliably than history-driven scanning.

Keywords: Adaptive LiDAR sensing · Sensing budget allocation · Cameraguided scanning · 3D object detection · Autonomous driving

## 1 Introduction

LiDAR is the main range sensor of most autonomous driving stacks [5, 28], but a dense scan of the full field of view at every frame spends much of the sensing budget on empty road, building walls, and objects the system already tracks with high confidence. Temporal cues are now standard inside detection networks [12, 13,24], and adaptive LiDAR scanning brings the same idea to the sensor itself: it concentrates a limited ray budget on regions of interest that are predicted from the recent past, and recent systems report large savings in scanning density [15, 20], the most recent at almost no loss in detection accuracy [21].

However, existing history-driven scanning policies face three key challenges. First, an object that has no track yet produces no region of interest, so pedestrians and vehicles that newly enter the field of view receive the least sensing budget; we call this failure first-sighting under-allocation. Second, the usual fallback outside the predicted regions is random or uniform sampling, which is not directed at where new objects appear and, as our experiments show, can even fall below the plain history policy. Third, camera cues are either fused inside the detector [3, 14, 27] or steer the scan without asking whether the tracker already covers the object [15], so part of the budget goes to objects that need no extra sensing.

To address these limitations, this paper introduces the CAmera-REsidual reserve (CARE), a training-free allocation rule built on one observation: a camera detection is useful for scan allocation only to the extent that the temporal memory cannot already explain it. At each decision time CARE projects the current track forecasts into the camera image, removes every detection that a same-class forecast already explains, and reserves part of the ray budget for the viewing directions of the remaining detections. The rest of the budget follows the base history policy, and any budget the reserve leaves unused is filled by a shared random floor, so the mechanism falls back to the history policy when the camera sees nothing new or fails entirely. A safety-bounded forgetting module (SBF) further releases budget from tracked objects that are receding or static and outside a speed-dependent guard distance, and hands the released cells to the same reserve; the guard is what keeps this forgetting safe at the tightest budget.

The main contributions of this paper are summarized as follows.

1. A leakage-free evaluation protocol that measures first-sighting under-allocation on a fixed angular ray-cell budget: cells are chosen before the returns are observed, all policies share the same detector, budget, and random floor, and first sightings are scored with paired per-scene bootstrap intervals; a strictcausal variant delays the camera cue by one keyframe.

2. The CARE allocation rule. On nuScenes validation (150 scenes, 4,148 events), CARE raises first-sighting recall by 5.2, 5.2, and 4.3 points at 10%, 20%, and 35% budgets over the history policy, with paired confidence intervals excluding zero. An all-camera control attributes this gain to the camera cue rather than the history state, and residual filtering directs the reserve only at objects the memory cannot explain, spending nothing on tracked ones. The first-sighting versus overall trade-of is reported as a budget-dependent Pareto choice.

3. The safety-bounded forgetting module, which combines time-decay forgetting of low-risk tracks with a speed-dependent guard distance and a localizationnoise margin, all set from published trafic rules and models rather than tuning. On the ablation subset at the tightest budget, forgetting without the guard loses 1.8 points of near-field recall against CARE without forgetting, with an interval excluding zero, while the guarded default shows no significant loss. The pipeline is further demonstrated on a real vehicle and in a closed-loop simulation study.

CARE is further compared against policy-level re-implementations of recent and standard alternatives, including history-driven scanning [21], an uncertaintydriven reserve [2], and beam reduction [25], all under the same allocator, budget, and detector.

![](images/d5e856989c98007e325b12264e44c6eb89ce0a9a916ff61d2c0a338d0ad1a366.jpg)  
Fig. 1: Overview of CARE at one keyframe. Camera detections $D _ { t }$ that the projected track forecasts (angular hulls $H _ { i } )$ cannot explain form the residual layer U (Algorithm 2), which fills the reserve of the shared allocator (Algorithm 1) ahead of the seeded floor; the SBF stage (Algorithm 3) shrinks the hulls of low-criticality tracks. The output mask $S _ { t }$ is held over the sweeps, and the 3D detections update the track memory.

## 2 Related Work

Adaptive and temporal LiDAR scanning. Programmable LiDAR can spread its rays unevenly over the field of view, and early systems steered this freedom with image cues [15], a scene prior [20], or human visual attention [18]. The most recent and strongest line derives the regions of interest from the tracks the perception system already holds: a history-aware predictor with a learned mask generator concentrates dense scanning on forecast object regions, cutting LiDAR energy by more than 65% at comparable accuracy [21], and forecast positions have been fed directly as detection queries [29]. These history-driven policies share one structural property: every dense region traces back to an object the system has already seen, so an object at its first appearance, which has no track yet, is given no priority. CARE keeps this history policy as its base and adds the one cue that the history state cannot contain, a current-frame camera detection, so that newly entering objects are covered.

Active perception with uncertainty. A second family points the sensor where the current estimate is least certain: programmable light curtains place a thin sensing surface at the least certain depth, with safety-envelope variants tracking the closest possible obstacle [1, 2], building on uncertainty estimation in deep vision models [10]. Uncertainty is a useful cue, but it can only be formed where the sensor has already returned points, so an object that has never been measured produces no uncertainty to react to. Our experiments include a reserve of this kind, and it reaches the highest overall recall of any policy we test, yet it trails CARE at every budget on first sightings, exactly the gap between revisiting measured but uncertain regions and directing sensing toward new objects.

Post-acquisition sampling and beam reduction. A diferent line reduces LiDAR data after the full scan is captured. Farthest point sampling is the standard geometric choice [16], learned samplers keep the task-relevant points [11,30], beam reduction drops whole scan lines and is the common protocol for crossdensity studies [25], and related methods prune tokens or voxels inside the network [17, 31]. All of these subsample a cloud that has already been acquired, so they save downstream computation but not a single ray of sensing; we keep random and farthest point sampling as such references, and adapt beam reduction and a learned scorer into causal scan policies for comparison.

Camera-LiDAR fusion. Modern 3D detectors combine camera and LiDAR features inside the network [3,6,14,26,27], and the strongest systems on nuScenes are multi-modal [5]. This fusion happens after the scan pattern is already fixed, so the camera improves what is done with the returned points but never changes where the LiDAR looks. CARE uses the camera one step earlier, as the cue that decides where the next rays go; it reads only boxes from a frozen 2D detector, leaves the downstream detector untouched, and could sit in front of a fusion model as well.

## 3 Problem Formulation

Budgeted ray-cell scanning. We discretize the LiDAR field of view into an angular grid G of A azimuth columns and E elevation rows, giving $N = A \cdot E$ firing cells. A scanning policy selects a subset $S _ { t } \subseteq \mathcal G$ at decision time t under a budget

$$
| S _ { t } | \le B , \qquad B = \mathrm { r o u n d } ( \beta N ) ,\tag{1}
$$

where $\beta \in ( 0 , 1 ]$ is the budget fraction. Only points whose viewing direction falls in a selected cell are retained, and the masked cloud is passed to a frozen 3D detector. Selection is leakage-free: $S _ { t }$ is fixed before the returns of frame t are observed, so a policy may use the past and the camera but never the cloud it is about to acquire.

Causality of the camera cue. The scan mask is computed once per keyframe, before the returns of that keyframe are observed, and held over the intervening sweeps. The main evaluation uses the synchronized camera frame of each keyframe; we additionally evaluate a strict-causal variant that uses detections from the preceding keyframe, which are half a second stale and complete well before the current decision, and the vehicle study measures the real camera-todecision lag.

First-sighting events and metrics. A first sighting is the first keyframe at which an annotated object within range R appears in the scene, excluding objects present at scene start. The reported events are new entries; reappearances after annotation gaps are not scored separately, although they present the same absence of track support. For each policy we report: first-sighting recall, the fraction of such events detected at their first keyframe; time to first detection (TTFD) in keyframes, censored at $K ;$ and overall recall over all in-range object frames. Contrasts use paired per-scene bootstrap intervals, and results are stratified by range and azimuth.

Remark 1. Throughout the experiments the grid is A=512 azimuth columns by $E { = } 3 2$ elevation rows, matching the channel count of the rotating LiDAR in the evaluation data [5], with each azimuth column grouping a few neighboring firings. The budgets are $\beta \in \{ 0 . 1 0 , 0 . 2 0 , 0 . 3 5 \}$ : the largest matches recent adaptive scanning, which cuts LiDAR energy by more than 65% [21], and the smaller two probe the tighter regime. The diagnostic uses range $R { = } 5 0$ m and a classconsistent 2 m center-distance match; the oficial nuScenes metrics apply the full protocol and are reported separately. TTFD is censored at K=6 keyframes, three seconds at 2 Hz [5]. The paired bootstrap uses $1 0 ^ { 4 }$ per-scene resamples, a standard choice for percentile intervals [8]. All values were fixed before the evaluation runs.

## 4 Method

Figure 1 shows the pipeline at one keyframe. The latest camera frame passes through a frozen 2D detector to give the detections $D _ { t } .$ and the track memory produces constant-velocity forecasts with their angular hulls $H _ { i }$ . The residual stage (Algorithm 2) projects the forecasts into the image and keeps the detections that no forecast explains. The shared allocator (Algorithm 1) then fills the exploitation share from the hulls, the reserve from the residual set, and the remainder from a seeded floor to produce the mask $S _ { t } ;$ the SBF stage (Algorithm 3) shrinks the hulls of low-criticality tracks before allocation. The mask is held over the sweeps, a frozen 3D detector runs on the retained points, and its detections update the track memory.

Algorithm 1 Shared budget allocator (one keyframe, one policy)   
Require: budget $B ;$ reserve fraction $\rho ;$ exploitation sets H (full hulls before shrunk   
hulls, nearest first within each); reserve layers $\mathcal { L } _ { 1 } , \ldots , \mathcal { L } _ { m }$ (priority order); frame   
seed s   
1: $S \gets \emptyset$   
2: fill S round-robin from H up to round $( ( 1 - \rho ) B )$ cells   
3: for $k = 1 , \ldots , m$ do   
4: fill S round-robin from $\mathcal { L } _ { k }$ up to $B - | S |$ cells   
5: end for   
6: fill S with the seeded random floor Floor(s) up to B cells   
7: return S

## 4.1 Shared Budget Allocator

Every policy receives the same structure: an exploitation share of the budget, a reserve share, and a common random floor. Let $\rho \in ( 0 , 1 )$ be the reserve fraction. The allocator first fills at most $( 1 - \rho ) B$ cells from the exploitation sets, ordered nearest first and interleaved round-robin so that no single large object exhausts the share. It then fills the reserve layers in priority order, which keep at least $\rho B$ of capacity and inherit any unused exploitation; whatever remains is filled by a seeded random floor drawn with a seed shared by all policies at the same frame. This random floor is the only stochastic part of an otherwise deterministic pipeline. An unused reserve flows on to the floor, and a policy whose reserve is empty is exactly the history policy. Ties are broken by ascending cell index, and the total never exceeds B.

The exploitation set of the base policy is built from track forecasts. Each tracked object i with current position estimate is forecast one keyframe ahead by a constant-velocity rule, and its angular hull $H _ { i }$ is the set of cells covered by the forecast box, dilated by a small azimuth margin and one elevation row to absorb forecast error. This base policy is our policy-level re-implementation of history-driven scanning [21]: dense attention on forecast regions, sparse floor elsewhere.

## 4.2 CARE: the Camera-Residual Reserve

Let $D _ { t }$ be the 2D boxes of the latest causally eligible camera frame from the frozen 2D detector. Each forecast is projected into the image; a detection $d \in D _ { t }$ is explained if the projected center of a same-class forecast falls inside its box. The residual set is

$$
R _ { t } = \{ d \in D _ { t } \ : \ \mathrm { n o ~ s a m e - c l a s s ~ f o r e c a s t ~ c e n t e r ~ l i e s ~ i n } \ d \} ,\tag{2}
$$

```latex
Algorithm 2 Residual reserve construction (CARE)
Require: camera detections $D _ { t } ;$ track forecasts $\mathcal { F } ;$ intrinsics and extrinsics
1: $U \gets \emptyset ; M \gets \emptyset$
2: for $d \in D _ { t }$ do
3: if some $f \in { \mathcal { F } }$ of the same class projects into the box of d then
4: $M \gets M \cup \{ W ( d ) \}$ \triangleright explained by memory
5: else
6: $U \gets U \cup \{ W ( d ) \}$ \triangleright residual: memory cannot explain it
end if
8: end for
9: return U for CARE; (U, M) for the priority-ordering variant and CARE+SBF
```

and each residual box is mapped to a depth-free angular wedge $W ( d )$ by backprojecting its corners and center to rays, with the same margins as the hulls; the wedge set $U = \{ W ( d ) : d \in R _ { t } \}$ is the residual reserve layer, and M collects the wedges of the explained detections. CARE uses the wedge set U as its camera reserve with strict priority over the shared floor; the explained set M is kept for the priority-ordering variant and CARE+SBF, whose layers (U, M) fill in that order.

Two design choices matter. First, matching uses the full forecasts, never the shrunk hulls of SBF, so forgetting can only reduce where the scanner looks and can never create a false residual. Second, the exploitation share is protected by its own cap and is filled first, so a burst of false 2D detections competes only for the reserve and never displaces tracked-object cells.

Remark 2. The reserve fraction is ρ=0.3, fixed before the evaluation runs and swept in the ablation over {0.15, 0.3, 0.5}. The 2D detector is an of-the-shelf driving-domain YOLOX [9] at its public confidence threshold of 0.25; the angular margin is two azimuth cells and one elevation row, absorbing forecast error of roughly one object radius at mid range; both are swept in the ablation. The 3D detector is a frozen CenterPoint [28] with its public configuration on the standard ten-sweep input. None of these values was tuned on the evaluation split.

## 4.3 Safety-Bounded Forgetting

The exploitation share itself contains waste: a parked car that has been rescanned for many frames does not need its full hull every frame. SBF shrinks the hulls of tracks that the motion and guard rules classify as low criticality and leaves everything else untouched. A track is a candidate for forgetting only if it is receding faster than a threshold $\tau _ { \mathrm { r e c } }$ or quasi-static below a speed threshold $\tau _ { \mathrm { s t } } .$ and, independently, only if its range exceeds a speed-dependent guard distance

$$
d _ { \mathrm { g u a r d } } = k _ { \mathrm { s a f e } } \left( s _ { 0 } + { v } _ { \mathrm { e g o } } T _ { h } \right) ,\tag{3}
$$

so that nothing inside the ego interaction envelope is ever forgotten; every other track keeps its full forecast hull. Each consecutive safe frame increases a streak

Algorithm 3 Safety-bounded forgetting (SBF)   
Require: tracks with two-frame history; ego speed $v _ { { \mathrm { e g o } } } ;$ streak counters   
1: for each track i do   
2: lowcrit ← (receding ∨ static ) $\land ( r _ { i } > d _ { \mathrm { g u a r d } } )$ \triangleright Eq. (3)   
3: update streak: $c _ { i } \gets c _ { i } + 1$ if lowcrit else 0   
4: if lowcrit then   
5: shrink hull margin by the streak schedule, keep $m _ { \mathrm { u n c } }$ of Eq. (4)   
6: else   
7: keep the full forecast hull   
8: end if   
9: end for   
10: return exploitation sets (full hulls first, shrunk hulls second) for Algorithm 1

counter, and the hull margin decays with the streak on a fixed schedule that never drops the elevation pad. Because a real tracker carries localization noise, the shrunk hull keeps a residual margin of

$$
m _ { \mathrm { u n c } } = \left\lceil \frac { \sigma _ { \mathrm { p o s } } / r } { \varDelta \theta } \right\rceil\tag{4}
$$

azimuth cells for a track at range r, where $\sigma _ { \mathrm { p o s } }$ is the position uncertainty and ∆θ the cell width. The cells released by shrinking flow to the residual reserve through Algorithm 1, which couples retention and exploration in a single budget.

Remark 3. The guard follows published trafic rules and models: the headway term is $T _ { h } { = } 2 \ \mathrm { s } .$ , the two-second rule, whose empirical support as a safety indicator is given by Vogel [23]; the standstill term is $s _ { 0 } { = } 2$ m, the jam distance of the intelligent driver model [22]; and the form of a speed-dependent safe envelope follows the responsibility-sensitive safety model [19]. The scale factor $k _ { \mathrm { s a f e } }$ is swept in the ablation over {0.5, 1.0, 1.5, 2.0} and fixed at 2.0 beforehand; $\sigma _ { \mathrm { p o s } }$ is zero under ground-truth simulation and 0.4 m on the vehicle. The motion thresholds are set against adult walking speed, 1.2 to 1.4 m/s [4]: the static threshold $\tau _ { \mathrm { s t } } { = } 0 . 5 ~ \mathrm { m / s }$ stays below it, so a normally walking pedestrian is not classified as static, and the receding threshold $\tau _ { \mathrm { r e c } } { = } 1 \ \mathrm { m } / \mathrm { s }$ requires clear radial departure. At standstill the guard reduces to $k _ { \mathrm { s a f e } } s _ { 0 }$ , which the vehicle experiment confirms exactly.

Remark 4. CARE is the recommended operating point and its first-sighting gains do not depend on SBF. The value of SBF lies in making the retention side of the budget explicit and safe: the guard evidence in Section 5.4 shows that forgetting without the guard significantly harms near-field recall, whereas the guarded default releases budget with no significant loss at the tightest budget.

Failure handling. When the camera produces no eligible detections, $R _ { t }$ is empty and the reserve returns to the shared floor, so CARE equals the history policy by construction; Section 5.5 documents this fallback under a real camera failure. False 2D detections cannot displace the exploitation cells, which are filled first under their own cap.

## 5 Experiments

## 5.1 Setup

Data and detectors. The primary evaluation uses the nuScenes validation split [5] with 150 scenes, 6,019 keyframes, and 4,148 first-sighting events; 20 disjoint scenes only train the learned-allocation baseline. Detections of the frozen YOLOX are precomputed on all six cameras, and all emulation, protocol, and parameter values follow Remarks 1–3. Both detectors and the ground truth share the five driving classes of the LiDAR head (car, truck, bus, bicycle, pedestrian); the other 2D classes are dropped, and the remaining five nuScenes classes lie outside the evaluation.

Compared policies. All policies run through Algorithm 1 at identical budget, identical floor seeds, and identical detector. A floor seed fixes the random fill of the leftover budget, so all policies draw their floor with the same seed at a frame. We average history, the all-camera reserve, and CARE with and without SBF over three floor seeds, with per-seed values within 0.4 points of the mean, far below the diferences we report; the remaining policies, the oficial metrics, and the ablations use a single seed. The cell-budget policies are: history, our policy-level re-implementation of adaptive scanning [21]; history+uniform, which spends the reserve on a fixed uniform pattern; all-camera, which reserves for every camera detection and thus removes only the residual filter; uncertainty, a light-curtain style reserve on cells that were occupied but undetected in the previous frame [2]; beam-reduce, which keeps the elevation rows nearest the horizon [25]; a learned scorer trained on the calibration scenes [11]; and CARE with and without SBF. As non-causal downsampling references at matched point fraction we report random point sampling and, at small scale, farthest point sampling [16]; these see the full cloud and save no sensing.

## 5.2 Main Results

CARE raises first-sighting recall over the history policy by 5.2, 5.2, and 4.3 points at the 10%, 20%, and 35% budgets (Table 1, Figure 2), all intervals excluding zero; equal scene weights give 5.3, 5.2, and 4.2 points, so event-rich scenes do not drive it. A full scan reaches 0.245 first-sighting and 0.494 overall recall; CARE at 35% recovers 88% and 94% of these ceilings against 70% and 93% for history, so it closes most of the allocation-controlled gap; the low ceiling reflects a detector and return limit. The all-camera control shows that the camera cue, not the history state, drives the first-sighting gain: the two are tied on first sightings at all three budgets. Residual filtering keeps the reserve on objects the memory cannot explain instead of on tracked ones; on average the two reserves difer on overall recall by only −0.1, +0.5, and +0.7 points, a small deficit at 10% and small gains at the larger budgets, each interval excluding zero (Table 1). The near-tie is mechanical: the filter removes about a thousand cells of explained wedges per keyframe, but at 10% the unexplained wedges alone fill the reserve on 79% of keyframes, and the realized allocation changes on only 17% of keyframes, against 68% at 35%. The value appears where scenes are crowded: in the densest third of the scenes, ranked by tracked-object count, CARE gains 0.7 and 1.0 points of overall recall over the all-camera reserve at 20% and 35%. The denseminus-sparse diference of 0.7 and 1.3 points excludes zero, and the same pattern appears when scenes are ranked by explained detections instead. Sparse scenes are tied, and the stratified first-sighting contrasts are tied apart from a 1.3-point loss in the middle third at 20%; Section 5.6 adds the closed-loop counterpart at scarce budgets. The depth-free class gate falsely explains 32.5% of new entries with a 2D detection, yet reserving those directions recovers only four events at 10%: the detector misses 85% of them even when their directions are scanned, and CARE already detects nearly all of the remainder through the floor and neighboring coverage, so the ambiguity is nearly outcome-neutral. Against the history policy, the first-sighting gain is not free at tight budgets: in the paired per-scene contrast used for all intervals, overall recall moves by −0.8, −0.5, and +0.6 points at 10%, 20%, and 35%, a budget-dependent Pareto choice that we report as such rather than as a uniform win. The learned scorer matches CARE on first-sighting recall but loses 0.6 to 2.4 points of overall recall at every budget (Table 1); that a learned allocator does not beat the rules also indicates the history baseline, our concept-level re-implementation of a learned region predictor, is not understating such methods here. An oracle variant sharpens this point: replacing the forecast of every tracked object with its true next position, an upper bound on any learned region predictor, moves first-sighting recall by at most 0.5 points in a single-seed diagnostic run, and every interval contains zero. Within the same run CARE stays 3.7 to 4.7 points above this oracle, so what it adds is information about objects that have no track yet, not a better forecast. The oracle raises overall recall by 0.5 to 0.7 points, as expected from perfect placement on tracked objects. Descriptively, on a single seed CARE lowers the mean censored TTFD of the history policy from 3.76 to 3.64, 3.60 to 3.47, and 3.45 to 3.33 keyframes, a shift of about 0.12 keyframes (60 ms); no interval claim is made for TTFD. The efect-size gate fixed before the runs, five points of first-sighting recall, is met at 10% and 20%; the 35% gain stays positive but below it. Under the oficial protocol (Table 2), mAP shows the same transition, with CARE below history at 10% and 20% and slightly above at 35%, while NDS stays below history throughout, since these frame-averaged metrics dilute a first-sighting gain and thereby motivate the separate protocol.

![](images/e72af48a3dea4bcf50e0f63fbe5a9698f73da36141ec17c51423dc9a80fb8719.jpg)  
Fig. 2: First-sighting recall versus overall recall at the (a) 10%, (b) 20%, and (c) 35% budgets on nuScenes validation. The gray staircase traces the Pareto frontier; CARE sits at its knee in every panel, and Table 1 gives the paired contrasts.

![](images/031f6368909f41cbc6d8f97b52276f1767d462fb4a9c27597dcf4613d7ca7dae.jpg)

![](images/e3831591d1929c7ff504306d8272eb5766b7daa4a50b8b0bd9c1bb718338e2f5.jpg)  
Fig. 3: First-sighting recall (left) and overall recall (right) versus ray-cell budget for CARE, CARE+SBF, and the policy-level re-implementations: history-driven scanning [21], the uncertainty reserve [2], beam reduction [25], the all-camera reserve, and the learned scorer [11].

Table 1: Paired per-scene bootstrap contrasts, CARE minus the named policy, in points of recall with 95% confidence intervals; positive favors CARE; bold marks intervals excluding zero in favor of CARE. First-sighting contrasts not shown are statistically tied, except beam reduction at 10%, where CARE leads by 2.0 points; the significant overall-recall contrasts against history and the uncertainty reserve appear in the text. Cells use one decimal, or two where an endpoint would round to zero; the identical history cells at 10% and 20% difer before rounding. Absolute levels appear in Figure 3.
<table><tr><td>Contrast</td><td>10%</td><td>20%</td><td></td><td>35%</td></tr><tr><td colspan="5">First-sighting recall (points)</td></tr><tr><td>vs. History [21]</td><td>+5.2 [4.3,6.2]</td><td>+5.2 [4.3,6.2]</td><td></td><td>+4.3 [3.5,5.1]</td></tr><tr><td>vs. Uncertainty reserve [2]</td><td>+3.5 [2.6,4.4]</td><td>+2.7 [1.7,3.8]</td><td></td><td>+2.1 [1.1,3.1]</td></tr><tr><td colspan="5">Overall recall (points)</td></tr><tr><td>vs. Beam reduction [25]</td><td>+17.4 [15.9,19.0] +11.5 [10.4,12.7] +6.4 [5.4,7.5]</td><td></td><td></td><td></td></tr><tr><td>vs. Learned scorer [11]</td><td>+2.4 [1.9,3.0]</td><td>+0.7 [0.4,1.0]</td><td></td><td>+0.6 [0.3,0.9]</td></tr><tr><td>vs. All-camera reserve</td><td>-0.13 [−0.26,-0.01]</td><td>+0.5 [0.3,0.8]</td><td></td><td>+0.7 [0.4,1.0]</td></tr></table>

## 5.3 Baseline Comparison

The uncertainty reserve confirms the central distinction. It attains the best overall recall of all policies, exceeding CARE by 1.0, 2.2, and 0.5 points with intervals excluding zero, because it re-examines regions that returned points without a confident detection. Yet it trails CARE on first sightings by 3.5, 2.7, and 2.1 points at all three budgets (Table 1), since an object that has never returned a point creates no uncertainty to react to. Beam reduction shows the opposite behavior: its first-sighting recall is statistically tied with CARE at 20% and 35% and 2.0 points behind at 10%, while it gives up 17.4, 11.5, and 6.4 points of overall recall (Table 1), so CARE gives the better joint trade-of. The history+uniform control falls below plain history at every budget, so an undirected reserve costs more than it returns. Across the comparisons (Figure 2), CARE combines the best or statistically tied first-sighting recall at every budget with the highest overall recall among the policies that match its first sightings at 20% and 35%, conceding at most 2.2 points of overall recall to the uncertainty reserve; this joint behavior is the design goal of the residual reserve.

Table 2: Oficial nuScenes metrics under the ray-cell budgets. Because these metrics average over every in-range object frame, and most of those frames belong to objects the tracker already follows, a gain that is confined to first sightings carries little weight in the average, and CARE stays close to the history policy on mAP while trailing it on NDS. This dilution is the reason we measure first sightings separately. The detector head emits five of the ten protocol classes, so the absolute values are not comparable with standard ten-class reports.
<table><tr><td rowspan="2">Policy</td><td colspan="3">mAP</td><td colspan="3">NDS</td></tr><tr><td>10%</td><td>20%</td><td>35%</td><td>10%</td><td>20%</td><td>35%</td></tr><tr><td>History</td><td>0.076</td><td>0.097</td><td>0.110</td><td>0.131</td><td>0.164</td><td>0.169</td></tr><tr><td>All-camera</td><td>0.070</td><td>0.090</td><td>0.107</td><td>0.124</td><td>0.130</td><td>0.162</td></tr><tr><td>CARE+SBF (ours)</td><td>0.066</td><td>0.088</td><td>0.106</td><td>0.120</td><td>0.128</td><td>0.162</td></tr><tr><td>CARE (ours)</td><td>0.068</td><td>0.091</td><td>0.111</td><td>0.124</td><td>0.149</td><td>0.164</td></tr></table>

## 5.4 Ablations

Each mechanism is swept on a fixed quarter of the validation scenes, every fourth scene, 38 scenes with 925 events; diferences below about 1.5 points are within the subset noise and read as ties. The reserve fraction needs no tuning: firstsighting recall moves by at most one point across the swept fractions, since the exploitation cap protects tracked objects and unused reserve returns to the floor. Priority ordering is structural: on the full split the variant changes first-sighting recall by at most 0.2 points, with overall recall of 0.380, 0.418, and 0.457. The forgetting knobs are insensitive on first sightings: the guard scale, the localization margin, and the four decay schedules stay within the subset noise, because at urban speeds the guard of Eq. (3) already covers most nearby objects; the preregistered defaults of Remark 3 are kept. The guard itself, however, is load bearing. At the 10% budget, relative to CARE without forgetting, unguarded forgetting loses 1.8 points of near-field recall on objects within 20 m, interval [−2.9, −0.5], a quarter-scale guard loses 1.5 points [−2.7, −0.2], and the default shows no significant loss, $- 0 . 5 \left[ - 1 . 7 , + 0 . 4 \right]$ , while releasing 24 cells per keyframe. At 20% even the default pays 3.1 points [−4.7, −1.5], so forgetting is a tool for the tightest budgets.

![](images/0602ef65b2f3622875e09883cca5d38be2c0dc911f4a702ddcecd29c82940989.jpg)  
Fig. 4: The vehicle platform: (a) drive-by-wire chassis with LiDAR, forward camera, and GNSS; (b) camera view and (c) LiDAR cloud from a separate drive.

Three further sweeps probe the camera pathway. Delaying the camera detections by one full keyframe, so the reserve only sees half-second-old evidence, still recovers 4.6, 5.3, and 2.6 points over history: same-frame camera input is not required. Halving or doubling the angular margin keeps the paired gap between 3.3 and 6.3 points, and raising the 2D confidence threshold from 0.25 to 0.5 costs one point at the tightest budget and nothing above; all three sweeps leave the policy ordering unchanged and are descriptive.

## 5.5 Vehicle Case Study

The platform is a drive-by-wire vehicle with one rotating LiDAR and one forward camera (Figure 4); we analyze a driving sequence with the camera operating and a garage session during which the camera failed and delivered no frames.

On the driving sequence the pipeline runs end to end with real YOLOX detections and ego-motion compensated forecasts. The reserve uses the newest camera frame at or before each decision, rotating wedge directions by the ego yaw change over the gap; this evidence is 0.11 to 0.14 s old on average against the 0.30 s keyframe spacing, and 0.42 s at the 95th percentile, a staleness the delayedcamera sweep covers. Across 104 new-entry events CARE detected the entry earlier than the history policy on 17 and later on 8, with the rest tied. Matched detections were rare, 19 keyframes of 648: at first appearance the camera mostly sees objects the tracker does not hold, so CARE and the all-camera reserve are nearly identical here; intrinsics are approximate.

In the garage session the perception path had a steady-state compute p99 of 97.9 ms against a 100 ms budget over 1,000 frames; the 2D stage used synthetic inputs because the camera recorded no images; acquisition and transport lie outside the boundary. At standstill the guard of Eq. (3) equaled $k _ { \mathrm { s a f e } } s _ { 0 } = 4 . 0 0$ m on all 570 keyframes, and the margin of Eq. (4) kept the net release near zero, the intended conservative behavior.

With the camera absent, CARE, all-camera, and history produced identical allocations on all 37 stable new-entry events, while newly appearing clusters received as few as 1 to 5 cells at appearance against 117 to 123 inside established coverage: the diagnosed under-allocation is visible on the vehicle. Several close or side entries were missed by every policy, so equality under the fallback does not imply detection safety; all vehicle results are descriptive.

![](images/3e1a30a7239e7be183ab5fc2f60ee4b2f284ce63a76905afc07e788b3bd09baa.jpg)

![](images/47ed30f4255a6a87ba548b25757f0fd5ced4923587d2e3fae3e959ae0471ab27.jpg)  
Fig. 5: The dense CARLA scene. (a) The ego follows a street lined with tracked vehicles; the pedestrian crossing from behind the parked truck (dashed track, magnified inset) is the first sighting the reserve must catch. (b) Median first-detection distance of the crosser, five seeds per point, interquartile bars. The all-camera reserve collapses at 3%, spread over the five tracked vehicles; CARE and CARE+SBF match the full-scan bound throughout, so the three top curves coincide.

## 5.6 Closed-Loop Simulation Study

To test closed-loop consequences we add a controlled CARLA study [7]: a fixed controller brakes on detected objects, the same allocator masks the cloud, the camera cue uses ground-truth boxes with injectable false negatives and latency, and detection needs enough masked returns. Scenario physics were fixed before any policy ran; all 245 episodes over five seeds are identical across policies.

When a pedestrian darts out from behind a parked vehicle, at the 10% budget the camera policies detect it at about 15 m, brake in every episode, and pass at 5.7 m or more; history detects at 11.9 m, brakes in three of its five episodes, and passes at 4.0 m. In the dense scene of Figure 5, at 3% CARE detects the crosser at 15.1 m, the full-scan bound, while the all-camera reserve spreads over the tracked vehicles and detects at 10.3 m; they converge from 5% upward. A vehicle cutin shows no separation, since a large vehicle returns enough points even when starved: the camera cue matters for small or sparse first sightings. At 10%, falsenegative rates up to 0.4 and a two-frame latency leave the ordering unchanged. No episode collides under the fixed physics, so we report detection and braking; the proxy upper-bounds a learned detector, and we make no deployment claim.

## 6 Conclusion

History-driven adaptive LiDAR scanning under-samples exactly the objects it has never seen. This paper measured that failure with a leakage-free protocol and repaired it with CARE, a training-free reserve for camera detections that memory cannot explain. On nuScenes validation CARE improves first-sighting recall over history at every budget with intervals excluding zero; the camera cue drives this gain; residual filtering costs about 0.1 points of overall recall at the tightest budget and pays of in crowded scenes. The mechanism runs on a real vehicle and, in simulation, sees an occluded pedestrian earlier and brakes more reliably; the evidence is limited to one frozen detector pair, ray-cell emulation, and new-entry events; steerable hardware is the next step.

## References

1. Ancha, S., Pathak, G., Narasimhan, S.G., Held, D.: Active safety envelopes using light curtains with probabilistic guarantees. In: Robotics: Science and Systems (RSS) (2021)

2. Ancha, S., Raaj, Y., Hu, P., Narasimhan, S.G., Held, D.: Active perception using light curtains for autonomous driving. In: ECCV (2020)

3. Bai, X., Hu, Z., Zhu, X., Huang, Q., Chen, Y., Fu, H., Tai, C.L.: TransFusion: Robust lidar-camera fusion for 3d object detection with transformers. In: CVPR (2022)

4. Bohannon, R.W.: Comfortable and maximum walking speed of adults aged 20–79 years: reference values and determinants. Age and Ageing 26(1), 15–19 (1997)

5. Caesar, H., Bankiti, V., Lang, A.H., Vora, S., Liong, V.E., Xu, Q., Krishnan, A., Pan, Y., Baldan, G., Beijbom, O.: nuScenes: A multimodal dataset for autonomous driving. In: CVPR (2020)

6. Chen, X., Zhang, T., Wang, Y., Wang, Y., Zhao, H.: FUTR3D: A unified sensor fusion framework for 3d detection. In: CVPR Workshops (2023)

7. Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., Koltun, V.: CARLA: An open urban driving simulator. In: CoRL (2017)

8. Efron, B., Tibshirani, R.J.: An Introduction to the Bootstrap. Chapman & Hall/CRC (1994)

9. Ge, Z., Liu, S., Wang, F., Li, Z., Sun, J.: YOLOX: Exceeding YOLO series in 2021. arXiv:2107.08430 (2021)

10. Kendall, A., Gal, Y.: What uncertainties do we need in bayesian deep learning for computer vision? In: NeurIPS (2017)

11. Lang, I., Manor, A., Avidan, S.: SampleNet: Diferentiable point cloud sampling. In: CVPR (2020)

12. Lin, X., Lin, T., Pei, Z., Huang, L., Su, Z.: Sparse4D: Multi-view 3d object detection with sparse spatial-temporal fusion. arXiv:2211.10581 (2022)

13. Liu, H., Teng, Y., Lu, T., Wang, H., Wang, L.: SparseBEV: High-performance sparse 3d object detection from multi-camera videos. In: ICCV (2023)

14. Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D.L., Han, S.: BEVFusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation. In: ICRA (2023)

15. Pittaluga, F., Tasneem, Z., Folden, J., Tilmon, B., Chakrabarti, A., Koppal, S.J.: Towards a MEMS-based adaptive LIDAR. In: International Conference on 3D Vision (3DV) (2020)

16. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: PointNet++: Deep hierarchical feature learning on point sets in a metric space. In: NeurIPS (2017)

17. Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.J.: DynamicViT: Eficient vision transformers with dynamic token sparsification. In: NeurIPS (2021)

18. Scarì, F., Myers, N.J., Quan, C., Zgonnikov, A.: Hybrid human-machine perception via adaptive lidar for advanced driver assistance systems. arXiv:2502.17309 (2025)

19. Shalev-Shwartz, S., Shammah, S., Shashua, A.: On a formal model of safe and scalable self-driving cars. arXiv:1708.06374 (2017)

20. Shomer, A., Avidan, S.: Prior based sampling for adaptive lidar. arXiv:2304.07099 (2023)

21. Shoouri, S., Taba, M.T., Kim, H.S.: Adaptive LiDAR scanning: Harnessing temporal cues for eficient 3D object detection via multi-modal fusion. arXiv:2508.01562 (2025), AAAI 2026, to appear

22. Treiber, M., Hennecke, A., Helbing, D.: Congested trafic states in empirical observations and microscopic simulations. Physical Review E 62(2), 1805–1824 (2000)

23. Vogel, K.: A comparison of headway and time to collision as safety indicators. Accident Analysis & Prevention 35(3), 427–433 (2003)

24. Wang, S., Liu, Y., Wang, T., Li, Y., Zhang, X.: Exploring object-centric temporal modeling for eficient multi-view 3d object detection. In: ICCV (2023)

25. Wei, Y., Wei, Z., Rao, Y., Li, J., Zhou, J., Lu, J.: LiDAR distillation: Bridging the beam-induced domain gap for 3d object detection. In: ECCV (2022)

26. Xie, Y., Xu, C., Rakotosaona, M.J., Rim, P., Tombari, F., Keutzer, K., Tomizuka, M., Zhan, W.: SparseFusion: Fusing multi-modal sparse representations for multisensor 3d object detection. In: ICCV (2023)

27. Yan, J., Liu, Y., Sun, J., Jia, F., Li, S., Wang, T., Zhang, X.: Cross modal transformer: Towards fast and robust 3d object detection. In: ICCV (2023)

28. Yin, T., Zhou, X., Krähenbühl, P.: Center-based 3d object detection and tracking. In: CVPR (2021)

29. Zhang, S., Chen, H., Wang, R.: A prediction-as-perception framework for 3d object detection. arXiv:2603.12599 (2026)

30. Zhang, Y., Hu, Q., Xu, G., Ma, Y., Wan, J., Guo, Y.: Not all points are equal: Learning highly eficient point-based detectors for 3d lidar point clouds. In: CVPR (2022)

31. Zhao, T., Ning, X., Hong, K., Qiu, Z., Lu, P., Zhao, Y., Zhang, L., Zhou, L., Dai, G., Yang, H., Wang, Y.: Ada3D: Exploiting the spatial redundancy with adaptive inference for eficient 3d object detection. In: ICCV (2023)