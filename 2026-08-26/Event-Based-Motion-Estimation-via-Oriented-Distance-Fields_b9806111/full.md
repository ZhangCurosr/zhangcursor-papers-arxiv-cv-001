# Event-Based Motion Estimation via Oriented Distance Fields

Lei Sun<sup>∗</sup>, Yuqin Ma<sup>∗</sup>, Weilun Li, Haoran Liang, Runyi Yang, Kaiwei Wang, Danda Pani Paudel, and Luc Van Gool

Abstract—Event-based motion estimation is central to tasks that demand high temporal resolution and robustness to fast motion. Existing methods typically rely on iterative optimization or repeated hypothesis comparison, offsetting the sensor’s lowlatency advantage. We propose Oriented Distance Field Motion Estimation (ODF Motion Estimation), which replaces this optimization with a single averaging step over a precomputed field of event distance vectors, combined with an adaptive event-count selection strategy and a parameter-free trail filter. On public and self-collected datasets, ODF motion estimation reaches sub-pixel accuracy at the lowest latency among compared methods. We validate its generality on two downstream applications rather than treating them as separate contributions. First, the estimated trajectory is converted into a blur kernel and paired with a compact iterative-unfolding network, trained on simulated motion-estimation noise, for real-time non-blind image deblurring, attaining competitive or superior PSNR/SSIM with under 1M parameters. Second, the same precomputed field is repurposed for directional event filtering in a low-power asynchronous pupil and glint tracker, sustaining stable tracking for tens of seconds while lowering a near-eye module’s power draw.

Index Terms—Event cameras, motion estimation, image deblurring, eye tracking.

## I. INTRODUCTION

ECOVERING the relative motion between a camera and its scene within a short observation window is a fundamental perception capability for platforms that must operate while moving quickly, mobile robots equipped with gimbal or Pan-Tilt-Zoom (PTZ) camera systems chief among them, which rely on rapid panning and tilting for wide-area situational awareness. An event camera is well matched to this need: rather than fixed-rate frames, it reports per-pixel brightness changes asynchronously and with microsecond resolution [11], so the stream itself directly encodes the motion trajectory rather than requiring it to be inferred after the fact. Recovering this trajectory means aligning the edges implied by early events against those implied by each subsequent event, and existing solutions fall into two families: an explicit optimization objective solved iteratively [12], [17], [18], [25], or a discrete search over hypothesized states compared per incoming event [3], [4].

Both families reintroduce the very latency that motivates using an event camera in the first place. Iterative optimization requires repeatedly evaluating and refining an energy function before it converges, and hypothesis-based search requires comparing several candidate states for every event rather than resolving it directly. For a platform whose onboard perception must keep pace with its own fast angular motion, this perevent computational burden, not the sensor’s raw temporal resolution, becomes the binding constraint on how quickly motion can actually be recovered.

This paper removes that bottleneck by replacing optimization or search with a single closed-form operation. We propose Oriented Distance Field Motion Estimation (ODF motion estimation), which precomputes, once per edge template, a field whose value and descent direction at every location directly encode the translation that would align that location with the template. So estimating a new event’s motion reduces to a lookup into this field, and combining many events reduces to averaging those lookups, with no per-event optimization, search, or learned inference. Under the common case of a dominant 2-DoF translation, stated explicitly and examined further in Sec. III, this closed-form average recovers a trajectory competitive with what iterative or search-based solvers need many more operations to reach.

Because the trajectory ODF motion estimation recovers is useful wherever motion itself must be known, not specific to one task, we validate it through two applications that consume it in different ways rather than presenting them as independent contributions. The trajectory is converted into a blur kernel to test whether the estimate is accurate enough to support non-blind image deblurring, and the same underlying field is repurposed as a directional event filter to test whether the estimator generalizes to asynchronous feature tracking, demonstrated on pupil and glint tracking for eye tracking. Both serve to validate the estimator under different accuracy and robustness requirements rather than to propose new deblurring or eye-tracking architectures in their own right (Fig. 1).

In summary, we deliver the following contributions:

• ODF motion estimation, a closed-form algorithm for event-based 2-DoF motion estimation that replaces iterative optimization or hypothesis search with a single distance-vector average, an adaptive event-count selection strategy, and a parameter-free trail filter.

• Validation of ODF motion estimation’s generality through two downstream consumers of its output, a non-blind deblurring network and a directional event filter for asynchronous tracking, evaluated as tests of the estimator rather than as separate contributions.

• Extensive evaluation on public and self-collected datasets, showing that ODF motion estimation reaches state-of-theart or competitive accuracy at substantially lower latency than optimization- or search-based alternatives, and that this accuracy carries through to competitive downstream deblurring and tracking performance at a fraction of the parameter count.

![](images/a4c7d810bdd9fbb1bdfba83a35bb5c7be23722e278375735f6de65258540a83c.jpg)  
Fig. 1. Overview of the proposed pipeline. (a) ODF motion estimation (Sec. III). An edge template built from events or a reference frame is used to precompute a distance field and its descent direction. Averaging the distance vectors of subsequent events against this field gives the motion trajectory during exposure, which yields a temporally-centered blur kernel and an event mask marking where events occurred. (b) Image deblurring with the estimated kernel (Sec. IV-A). The kernel and event mask from (a), together with the blurry image, are passed to an iterative deep-unfolding network that alternates a closed-form data module with a learned denoising network to recover the sharp image. The same motion trajectory from (a) is also reused as a directional filter for asynchronous pupil and glint tracking (Sec. IV-B).

## II. RELATED WORK

## A. Motion Estimation with Event Cameras

Event-based motion estimation methods either repurpose classical optical flow machinery [2], [23] or exploit the event generation model directly, as in contrast maximization [12], time-surface alignment [17], and closed-form line solvers [13]. Multi-hypothesis trackers [4] compare many candidate states per event. Frame-fusion methods such as EKLT [14] and edgealigned tracking [20] extend feature lifetime through iterative alignment. ODF motion estimation instead precomputes the distance vector field once per template, reducing motion estimation to a single average per event.

## B. Blurry Kernel Estimation and Deblurring

Image deblurring methods split by whether the blur kernel is known: non-blind deblurring recovers the sharp image from a known or estimated kernel via classical solvers or learned priors such as USRNet [37], DPIR [38], and DWDN [9], while blind deblurring estimates both the kernel and the sharp image from the blurred input alone. Event cameras give nonblind deblurring a direct route to the kernel: the event double integral model [22], [26] recovers it in closed form but is noise-sensitive, while end-to-end fusion networks [30], [31] skip kernel estimation entirely yet generalize poorly beyond their training event statistics [29]. ODF motion estimation keeps the pipeline compact and non-blind, closing this gap with an explicit motion-estimation-noise kernel model rather than iterative optimization or heavy fusion.

## C. Eye Tracking

Eye tracking most commonly locates the pupil via pupilcenter-corneal-reflection (PCCR) [15] or appearance-based models, both requiring fast, stable pupil (and, for PCCR, glint) detection. Event cameras avoid the power-accuracy trade-off of a fixed frame rate, and event eye tracking has progressed from asynchronous model fitting [5] to learned pupil segmentation and gaze regression [21], [40], with spiking implementations on neuromorphic hardware [6] reaching milliwatt-level power but still trailing frame-based PCCR in accuracy. We build on this line of work by reusing the translation vectors from Sec. III for directional event filtering rather than learning a separate pupil model, keeping the tracker parameter-free.

## III. EVENT-BASED MOTION ESTIMATION

ODF motion estimation relies on three conditions holding over the events used for a single trajectory estimate. The scene motion is globally consistent, so one 2-DoF translation explains every tracked edge rather than independently moving regions. That translation is well approximated as 2-DoF rather than a general rotation, affine, or projective warp. And the template’s edges span enough distinct orientations to constrain both components of the translation. These conditions are what make a closed-form average a substitute for iterative optimization, and Sec. III returns to each: dynamic-scene failure for the first, specific camera platforms for the second, and Sec. III-C for the third, where it sets the accuracy of the averaged estimate. Violating a condition degrades the estimate in a specific way rather than uniformly: an independently moving object biases the average toward a mixture of motions, unmodeled rotation or higher-order warp aliases into the estimate as systematic error, and insufficient orientation diversity biases the under-constrained component toward the dominant edges’ direction, consequences we revisit in our limitations (Sec. VI).

## A. Edge Alignment

Motion events are triggered predominantly near scene edges, where the spatial gradient is large, since the intensity increment during motion is

$$
\Delta L \approx - \nabla L \cdot \mathbf { v } \Delta t ,\tag{1}
$$

where L is the (log) image intensity, v is the instantaneous velocity of the scene relative to the camera, and ∆t is a small time interval. The resulting event stream therefore records these edges’ trajectory during exposure, which we recover through edge alignment: matching the edges formed by the earliest events against those formed by each subsequent event. Edge or feature alignment has been applied between frame images [20], point clouds, and event streams [17], [18], [25], [41]. We apply the same principle between an edge template and incoming events.

![](images/c7528dc7def92469948822952657291ee05d6f1ea95ec458d0eb6d0bd5d1ed62.jpg)  
(a)

![](images/b6fe2f4f27eaf92bad0752fa3f893d83d389fc5f8064c7f28f7de1ae15addcf5.jpg)  
(b)

![](images/6100d5f820d98d42046ce79a34505bace80d5e4aa7d409fe649893d0f8fbe038.jpg)  
(c)  
Fig. 2. Event-based trajectory extraction. (a) Blurred frame. (b) Events within the exposure window. (c) Event stream (red and blue dots) with the extracted motion trajectory (green line).

Blur induced by 2-degree-of-freedom (2-DoF) translation, such as handshake or panning, is the common case we target. Existing event-based edge tracking methods address this and more general motion via an optimization objective solved through iterative [12], [17], [18], [25] or discrete search [3], [4] procedures. We show this optimization can be replaced entirely under the 2-DoF assumption: trajectory recovery reduces to exploiting event temporal continuity together with an oriented distance field, without any iterative or discrete search step (Fig. 2).

## B. Pattern Generation

```tcl
Algorithm 1 Pattern Generation With Adaptive Batch Strategy
1: Input: events ${ \overline { { E , } } }$ down sampling factor N, overlap ratio
$r _ { i }$
2: create event count frame $I _ { c o u n t }$ to record the overlap
3: slice $E$ into event bundles $\xi _ { 1 } , \xi _ { 2 } , . . . ,$ each containing $n _ { e }$
events
4: for $\xi _ { i }$ in $\xi _ { 1 } , \xi _ { 2 } , \ldots$ do
5: n<sub>overlap</sub> $ 0$ ▷ reset for the current bundle
6: for each event $( x , y , t , p ) \in \xi _ { i }$ do
7: $( x ^ { \prime } , y ^ { \prime } ) \gets ( x / / N , y / / N )$
8: if $I _ { c o u n t } ( x ^ { \prime } , y ^ { \prime } ) \neq 0$ then
9: $n _ { o v e r l a p }  n _ { o v e r l a p } + 1$
10: end if
11: $I _ { c o u n t } ( x ^ { \prime } , y ^ { \prime } ) \gets I _ { c o u n t } ( x ^ { \prime } , y ^ { \prime } ) + 1$
12: end for
13: if $n _ { o v e r l a p } / n _ { e } > r _ { i }$ then
14: collect the preceding events $\xi _ { 1 } , . . . , \xi _ { i }$
15: break
16: end if
17: end for
```

Building an edge template from a fixed time window or event count is unreliable: edge density and motion speed vary across scenes, so a fixed threshold yields templates too sparse to constrain motion or too dense to compute efficiently. Prior adaptive schemes [17], [23] address this but add their own overhead. We instead exploit an adjacency effect already present in spatial down sampling: as events are down sampled, nearby events increasingly collide with alreadyoccupied bins, and the fraction of colliding events within each newly processed bundle signals when the template has accumulated sufficient spatial coverage. We track this fraction per bundle, resetting it before each new bundle, and stop once it exceeds the ratio $r _ { i } ( \mathrm { A l g o r i t h m }$ 1), set empirically rather than from a coverage model since collision rate depends on scene content in ways we do not model explicitly. The choice is not delicate: once a template nears sufficient coverage, further events collide with occupied bins at a fast-rising rate, so small changes to $r _ { i }$ shift the stopping point by a few events rather than which templates are accepted.

![](images/a83eb7a1da9b4d2f363aeeaf07364de4fec42000a5ece356d19b9200e4b1b093.jpg)

![](images/62af155a21ade78b93ede9aa569bb94322bb046678968611ec5111537a2d585a.jpg)  
(b)

(a)  
![](images/7bcf0eb5a09645c04ff6d14fa963ca9d99f93d047bc53a6b378f02431b986ab0.jpg)  
(c)

![](images/dac94bbdf4142eab14e8347c5d87d70c082c5454c8f68c538fc2439bcd898030.jpg)  
(d)  
Fig. 3. Distance field and direction field. (a-b) Toy example around a single edge point, showing how the distance vector at each location encodes translation magnitude and direction. (c-d) The same fields computed on a real graphics scene, used to precompute the distance vectors that give the translation direction and amount for subsequent events.

## C. Oriented Distance Field

Given the edge template of Sec. III-B, we construct a field over it that encodes, at every pixel, the translation needed to align that pixel with the reference edge. Distance fields built from events have proven effective for optical flow estimation [2] and related motion tasks. Following the use of time surfaces as distance fields [41], we build a field from the events nearest the start of exposure, align each subsequent event against it by shifting to minimize distance, and take the accumulated shift over all processed events as the edge trajectory.

The underlying idea is geometric: if the edge has moved by translation t since the reference template was built, every point on the moved edge sits exactly t away from its corresponding reference point, so a field storing this direction and distance at every location implicitly stores a translation estimate everywhere at once, precomputed rather than solved per event.

A standard distance field measures alignment quality but does not by itself yield the warp parameters needed to align two edges. Under the 2-DoF assumption, we instead construct an oriented distance field $D : \Omega \to \mathbb { R } _ { \geq 0 }$ over the image domain Ω that directly encodes the translation parameters at every location.

Let $\hat { d } ( x ) = - \nabla D ( x ) / \| \nabla D ( x ) \|$ denote the unit gradientdescent direction of the field at pixel x. The translation vector implied by $x ,$ referred to elsewhere in this paper simply as its distance vector, is then $v ( x ) = D ( x ) { \hat { d } } ( x )$ , precomputed and stored as the pair $( D , { \hat { d } } ) \ ( \mathrm { F i g . } \ 3 )$ . Each new event reads v(x) directly from these fields, and averaging v(x) across the current batch gives the displacement that shifts the field forward.

![](images/e73bf1e08f48ad27c670e73c0dc0805600cc2aa1ac5090ada94bb7380b580860.jpg)

![](images/eac095cb2e0a2e30c97c13817ff75a2ccac6bbc6d49406fd96558dc3324937bd.jpg)

![](images/ae34c68f2d384a87012a36246c117ab5442111e1b9916f5af5f21c53502bcfda.jpg)  
(a)  
(b)

![](images/ac321e1266b64a1bcf89e1171cff5aad7e424dc81810d583c62d3cbb57b274dc.jpg)  
(c)  
(d)  
${ \mathrm { F i g . ~ } } 4 .$ Valid edge region selection. (a) reference image. (b) edge template. (c) distance map from the edge template. (d) valid region after dilating sparseedge regions and intersecting with the edge active region.

A single event, however, cannot fully determine t: picture a straight edge sliding behind a small aperture, where motion toward or away from the edge’s line is visible, but sliding along its direction leaves the view unchanged. This is the classical aperture problem, and it applies to v(x) identically, since reading v(x) from one event constrains only the component of $t \in \mathbb { R } ^ { 2 }$ along that event’s local edge normal. Orientation diversity within a batch resolves the ambiguity: edges of differing orientation constrain different components of t, so their normal components reinforce each other under averaging while no single direction dominates, a good approximation precisely when a batch’s orientations are reasonably spread, and unreliable parallel to a template’s dominant orientation otherwise, for instance one long straight boundary, a failure mode we return to in Sec. VI. A separate problem arises in edge-dense regions, where field values are small everywhere and a raw average is biased toward zero regardless of orientation diversity. We correct this by dilating the sparseedge regions where the field exceeds 5 pixels and intersecting the result with the edge active region, restricting the average to this reduced valid region (Fig. 4). Both thresholds are empirical, chosen by inspecting when the dense-region bias became visually apparent, and are not delicate to tune since they only need to separate this small-value regime from the larger, informative values elsewhere.

Events generated far from any edge are noise that would otherwise pull the average off the true trajectory, so we discard events whose distance from the edge exceeds this same 5- pixel scale, since genuine correspondences rarely deviate this far under our sensors’ noise and quantization, while spurious events, not concentrated at any particular distance, are thinned out by the cutoff.

## D. Extracting Trajectory

We now recover the exposure-interval trajectory from the per-event translation vectors v(x) of Sec. III-C. As noted in [13], a single readout only recovers the component of motion perpendicular to the local edge, so we average v(x) over a batch of M events, $\begin{array} { r } { \bar { v } = \frac { 1 } { M } \sum _ { i = 1 } ^ { \tilde { M } } v ( x _ { i } ) } \end{array}$ , to approximate the full 2-DoF displacement via the orientation-diversity argument of Sec. III-C.

Choosing M matters: large batches blur the translation estimate, while small batches let event noise dominate and the trajectory oscillate. We set M to 2.5% of the event count used to build the edge template (Sec. III-B), empirically chosen since we lack a closed-form model of how noise scales with batch size, by observing where the trajectory stopped oscillating without yet showing visible blur. We have not formally swept its sensitivity, but the two failure modes are visually distinct enough that the setting should not be delicately poised between them.

Trailing events left behind by the moving edge likewise blur v¯, which we suppress with a Spatio-Temporal-Contrast filter for event-stream denoising before averaging.

The full trajectory s(t) is reconstructed by processing events sequentially within the exposure interval, concatenating each batch’s incremental displacement v¯. This process is noniterative, so each incremental estimate carries some error, but the error does not compound, since every v¯ is read from the field anchored to the fixed initial template rather than accumulated relative to the previous, already noisy estimate. Discussion. The efficiency of ODF motion estimation follows from where its computation is spent, not a constant-factor engineering optimization. Building the distance and direction fields over an $H \times W$ template costs O(HW), paid once per template rather than once per event: every subsequent event then costs only a field lookup and an accumulation into a running average, independent of scene complexity or motion magnitude. An iterative solver instead pays a cost that depends on how many refinement steps convergence takes, and a multihypothesis tracker pays a cost proportional to the states it compares per event. Memory follows the same pattern, two fixed $O ( H W )$ fields per template rather than a hypothesis set or optimizer state that grows with tracking duration. ODF motion estimation’s per-event cost is thus decoupled from motion magnitude, scene texture, and event count, the very factors that determine the cost of optimization- or search-based alternatives, a property Sec. V validates through update rate and latency.

## IV. APPLICATIONS

## A. Image Deblurring with Estimated Kernel

1) Kernel Generator: The motion blur point spread function (PSF) follows from the trajectory s(t) over exposure time T [7]:

$$
P S F ( \cdot ) = \int _ { 0 } ^ { T } \delta _ { s ( t ) } ( \cdot ) d t ,\tag{2}
$$

where $\delta _ { s ( t ) }$ is the Dirac delta function at $s ( t )$

To build the kernel, each position along the trajectory is assigned a value equal to the fraction of exposure time spent there, with sub-pixel linear interpolation at fractional positions. Unlike typical 8-bit kernels, we keep it in continuous, unquantized form throughout non-blind deblurring.

Because the clean-image ground truth in most deblurring datasets corresponds to the exposure’s middle rather than the kernel’s geometric center, we center the kernel on the trajectory position at the temporal midpoint of exposure, which ODF motion estimation gives directly since it recovers the exact pixel position at any instant during exposure.

2) Deep Unfolding Networkfor Deblurring with Noisy Blur Kernels: Event noise, sensor bandwidth limits, and kernel quantization mean the estimated kernel inevitably deviates from the ground truth. Prior noise models for non-blind deblurring [9], [34] assume error distributions inconsistent with ours and do not transfer directly.

Inspired by the half-quadratic splitting scheme in USR-Net [37] and DPIR [38], we build an iterative deep-unfolding network that alternates a data module with a learned prior module. Given the blur kernel $k ,$ the blurred image $y ,$ and the prior module’s previous output $z _ { i - 1 }$ , the data module solves the data-fidelity term in closed form,

$$
x _ { i } = \mathcal { F } ^ { - 1 } \left( \frac { \overline { { \mathcal { F } ( k ) } } \mathcal { F } ( y ) + \alpha _ { i } \mathcal { F } ( z _ { i - 1 } ) } { \overline { { \mathcal { F } ( k ) } } \mathcal { F } ( k ) + \alpha _ { i } } \right) ,\tag{3}
$$

where $\mathcal { F }$ is the Fourier transform, the overline denotes complex conjugation, and $\alpha _ { i }$ is a per-iteration regularization weight. The prior module then denoises $x _ { i }$ into $z _ { i } ,$ , conditioned in USRNet and DPIR on a known noise level unavailable here since the true kernel-estimation noise is not directly observable. We instead condition on the original blurred image, which carries less noise than the network’s own intermediate estimate, together with an event mask, a binary map marking pixels where the event count during exposure exceeds a threshold. Since these are the only pixels where the blurred and sharp image can differ, the mask focuses the network’s correction on genuinely blurred regions.

3) Training on Synthetic Data: Replaying synthetic trajectories through an event simulator would give blur kernels with realistic estimation noise, but current simulators emit frame-locked event timing that biases the estimated kernel. We instead generate motion-noise-infused blur kernels directly from the constructed trajectories without simulating events, convolving a low-quality image with the ground-truth kernel and noise to form the network input, supervised by the original high-quality image.

To avoid cyclic-boundary artifacts, we blur an oversized patch before cropping to the training resolution. Kernel size varies from 11 to 81 pixels across batches using diverse motion trajectories [7], with noise from simplified BSRGAN [39] and Real-ESRGAN [35] pipelines and simulated pixel saturation.

## B. Pupil Motion Estimation for Eye Tracking

Eye tracking estimates the visual axis, most commonly by locating the pupil center together with one or more corneal reflection glints under the PCCR framework [15]. The pupil and glints change shape as the eye rotates, so ODF motion estimation (Sec. III) does not apply directly, but the translation vectors it introduces still generalize to event-based spatial filtering of these features.

1) Pupil and Glint Event Signatures: An initial frame localizes the pupil ellipse and the corneal glints, and the subsequent event stream tracks both asynchronously. Because the pupil is darker than the surrounding iris, its motion generates negative events on the leading edge and positive events on the trailing edge, while the smaller, brighter glints generate the opposite pattern. As in Sec. III, only events near the previous feature boundary are processed.

2) Distance-Vector-Based Pupil Event Filtering: Pupil events are corrupted by trailing events left behind by the moving edge and by events from nearby glints. A fixedpolarity trail filter fails whenever the same pixel crosses two edges of the same sign in a row (Fig. 5(a,b)), which happens routinely as the pupil moves. We instead estimate the pupil motion direction by summing, over nearby events, the vector from each event to the previous pupil center, with negativeevent vectors flipped in sign so both polarities reinforce the same direction. An event is discarded as a trail event when its distance vector against the previous pupil ellipse (Sec. III-C) opposes this direction (Fig. 5(c)). Events inside the glint template (Sec. IV-B4) are filtered separately.

![](images/8ab25d775ade22287ad1d1bba55fd3b47a6df6b6429e31a740fce82c9fb455a8.jpg)  
(a)

![](images/f2ebf82eb0db2298a52a7af29e6b2d5bb198f69733e94e147a75cedd901eaab8.jpg)  
(b)

![](images/ffac6c44acca2c61c57dc40b9f86b1ae5294a0aeaed9318795a9650e1450bdfe.jpg)  
(c)

![](images/3f60ef2d9ed6ce05b2ad278db83f01d1f8cf0495466f6ba1d39affca7a8568fc.jpg)  
(d)

![](images/2a3da9b5368fe36f8e8c4eab8c9ac6687235fe74ab01a940ecbbaee0cb64d5b9.jpg)  
(e)  
Fig. 5. Pupil event filtering and device comparison. (a) Raw events during pupil tracking. (b) Valid events after a fixed-polarity trail filter, which still leaks trailing events. (c) Valid events after the proposed distance-vector direction filter. (d) Pupil events captured by a non-near-eye device, where the pupil occupies few pixels. (e) Pupil events captured by a near-eye device, where the pupil occupies many more pixels.

3) Pupil Tracking: Valid events are treated as new boundary points and refit to an ellipse with direct least-squares fitting, updated every fixed number of events, tuned to the initial ellipse size and to whether the device is near-eye (Fig. 5(d,e)) to balance fit stability against tracking drift. Points sampled from the previous ellipse boundary are added to each new fit to further stabilize it against event noise, removing short-axis oscillation and tracking failures.

4) Glint Localization and Blink Detection: Because glint polarity is reversed relative to the pupil, a directional template split at the axis perpendicular to the pupil motion direction (0 on the axis, +1/−1 on each side) is accumulated event by event to score candidate glint regions. The accumulator is thresholded and decayed rather than reset, since glints move less than the pupil and can disappear intermittently. Blinks are detected from an abrupt, sustained drop in the pupil ellipse’s minor axis, which suspends tracking.

## V. EXPERIMENTS

## A. Event-Based Motion Estimation

We evaluate motion estimation accuracy and update rate against contrast maximization (CM) [12] and HASTE [4], both pure-event methods, on four Event-Camera Dataset [24] sequences where a motorized slider translates the camera along one axis (slider close, slider $f a r ,$ and their high-dynamicrange counterparts). Motion is horizontal, so every method is constrained to 1 degree of freedom to avoid the aperture problem of Sec. III. ODF motion estimation and HASTE build their edge template with the adaptive event-count selection of Sec. III-B. For CM, we sweep its energy function, event-count window, and blur parameter and report the best configuration. Ground truth comes from the slider’s timestamped physical displacement.

Table I shows all three methods reaching sub-pixel error, confirming that a closed-form average does not by itself sacrifice accuracy. ODF motion estimation updates at the highest rate on every sequence and attains the lowest error on slider far and slider hdr far. HASTE and CM edge out a lower error on the remaining two sequences, since each can be

TABLE I  
MOTION ESTIMATION ACCURACY AND UPDATE RATE ON EVENT-CAMERA DATASET [24] (BOLD: BEST PER COLUMN).
<table><tr><td>Sequence</td><td>Method</td><td>Rate ↑ (Hz)</td><td>Error ↓ (px)</td></tr><tr><td rowspan="3">slider_close</td><td>HASTE [4]</td><td>467.2</td><td>0.396</td></tr><tr><td>CM [12]</td><td>19.5</td><td>0.734</td></tr><tr><td>Ours</td><td>1066.0</td><td>0.584</td></tr><tr><td rowspan="3">slider_far</td><td>HASTE [4]</td><td>188.9</td><td>0.457</td></tr><tr><td>CM [12]</td><td>17.2</td><td>0.558</td></tr><tr><td>Ours</td><td>679.4</td><td>0.328</td></tr><tr><td rowspan="3">slider_hdr_close</td><td>HASTE [4]</td><td>632.1</td><td>0.555</td></tr><tr><td>CM [12]</td><td>15.5</td><td>0.426</td></tr><tr><td>Ours</td><td>1036.5</td><td>0.543</td></tr><tr><td rowspan="3">slider_hdr_far</td><td>HASTE [4]</td><td>175.9</td><td>0.496</td></tr><tr><td>CM [12]</td><td>12.2</td><td>0.564</td></tr><tr><td>Ours</td><td>533.9</td><td>0.381</td></tr></table>

tuned to a sequence’s noise characteristics, whereas our fixed, precomputed field is not, trading the last fraction of a pixel iterative refinement can buy for an order-of-magnitude higher update rate, favorable when latency matters more than peak accuracy.

## B. Image Deblurring with Estimated Kernel

We evaluate deblurring on the global-motion sequences of EventAid-B [10] (box, building, desk, jingjin, dog, global, and xiaoying), spanning the blur magnitude and scene content in Fig. 6 and covering degradations such as row-scanning artifacts and low-light noise, with blur from near-sharp to over 100 pixels. No method is fine-tuned on this dataset, to measure generalization to real-world data.

For kernel-accuracy evaluation, we additionally collect a real dataset of 70 blurred images with corresponding events and ground-truth blur kernels, using a synchronized, colocated event-and-frame-camera rig and a motorized stage providing the true motion trajectory from its calibrated position during exposure.

We train on 800 DIV2K [1] and 2750 Flickr2K [33] images at 512×512 patch size, validating without test-time fine-tuning on Urban100 [16] and a color image set simulated the same way. Following USRNet [37], we use Adam at batch size 12, first with an $\ell _ { 1 }$ loss and learning rate $1 \times 1 0 ^ { - 4 }$ halved every $2 \times 1 0 ^ { 5 }$ iterations to about $6 \times 1 0 ^ { - 6 }$ , then fine-tuned for $2 \times 1 0 ^ { 3 }$ iterations with ℓ , VGG perceptual, and relativistic adversarial losses weighted [1, 1, 0.005].

We compare against blind deblurring baselines NAFNet [8] and Restormer [36], which take only the blurred image, and event-assisted baselines EDI [26] and EFNet [31] (pretrained on GoPro and, separately, fine-tuned on REBlur). To isolate ODF motion estimation’s contribution from the restoration network’s, we also pair our estimated kernel with two existing non-blind networks, DWDN [9] and RAM [32], in place of our own, and separately pair it with the plain USRNettiny [37] backbone at matched parameter count to isolate our architectural modifications from the kernel and the capacity budget. All pretrained models are used without further finetuning on either test set, reflecting generalization rather than dataset-specific adaptation.

Table II reports results on EventAid-B. Image-only methods cannot recover severely blurred regions and trail the eventassisted methods in PSNR. EFNet is sensitive to the mismatch between its training event statistics and EventAid-B’s degraded events: its PSNR falls below the parameter-tuned EDI baseline, though fine-tuning on REBlur raises its SSIM above every other method, including ours. With only 0.6M parameters, our method attains the highest PSNR, attributable to the ODFestimated kernel’s accuracy rather than network capacity. At matched capacity, pairing the same estimated kernel with the plain USRNet-tiny backbone reaches only 26.35/0.843, so our event-mask conditioning still contributes beyond capacity or kernel accuracy alone. Our SSIM trails EFNet-REBlur, plausibly because that fine-tuning targets REBlur’s specific structural statistics, a trade-off between a more accurate motion-derived kernel and a dataset-tuned prior (qualitative comparison in Fig. 7).

![](images/cfa78edd7cd00d0305abac7d2e7027ca08c574705528ac742693f869a2fe5f58.jpg)  
Fig. 6. Blurred EventAid-B images (main) with the corresponding kernel estimated by ODF motion estimation (inset), showing the range of blur magnitude and scene content covered by the benchmark. TABLE II TABLE II

COMPARISON OF SINGLE IMAGE MOTION DEBLURRING METHODS ON EVENTAID-B-GLOBAL MOTION. (BOLD: BEST PER COLUMN)
<table><tr><td>Method</td><td>Events</td><td>PSNR ↑</td><td>SSIM ↑</td><td>#Param</td></tr><tr><td>EDI [26]</td><td>V</td><td>26.45</td><td>0.800</td><td></td></tr><tr><td>Restormer [36]</td><td>x</td><td>26.25</td><td>0.842</td><td>26.1M</td></tr><tr><td>NAFNet-64 [8]</td><td>x</td><td>25.52</td><td>0.816</td><td>67.9M</td></tr><tr><td>EFNet-GoPro [31]</td><td>V</td><td>25.65</td><td>0.806</td><td>8.5M</td></tr><tr><td>EFNet-REBlur [31]</td><td></td><td>25.56</td><td>0.874</td><td>8.5M</td></tr><tr><td>ODF kernel + DWDN [9]</td><td></td><td>26.66</td><td>0.856</td><td>7.0M</td></tr><tr><td>ODF kernel + USRNet-tiny [37]</td><td></td><td>26.35</td><td>0.843</td><td>0.6M</td></tr><tr><td>ODF kernel + RAM [32]</td><td></td><td>26.31</td><td>0.839</td><td>35.6M</td></tr><tr><td>Ours (ODF kernel + event mask)</td><td>V</td><td>27.23</td><td>0.860</td><td>0.6M</td></tr></table>

TABLE III

COMPARISON OF SINGLE IMAGE MOTION DEBLURRING METHODS ON BLURRY IMAGES WITH EVENT DATA AND GROUND-TRUTH BLUR KERNELS. BOLD MARKS THE BEST RESULT PER COLUMN (GT KERNEL ROWS ARE AN ORACLE UPPER BOUND).
<table><tr><td>Method</td><td>Events</td><td>PSNR ↑</td><td>SSIM ↑</td><td>#Param</td></tr><tr><td>EDI [26]</td><td>V</td><td>30.54</td><td>0.856</td><td></td></tr><tr><td>Restormer [36]</td><td>x</td><td>30.34</td><td>0.861</td><td>26.1M</td></tr><tr><td>NAFNet-32 [8]</td><td>x</td><td>29.93</td><td>0.848</td><td>17.1M</td></tr><tr><td>EFNet [31]</td><td>V</td><td>31.67</td><td>0.854</td><td>8.5M</td></tr><tr><td>ODF kernel + DWDN [9]</td><td></td><td>31.79</td><td>0.874</td><td>7.0M</td></tr><tr><td>ODF kernel + RAM [32]</td><td></td><td>33.00</td><td>0.897</td><td>35.6M</td></tr><tr><td>Ours (ODF kernel + event mask)</td><td></td><td>32.33</td><td>0.885</td><td>0.6M</td></tr><tr><td>DWDN w/ gt kernel [9]</td><td></td><td>32.58</td><td>0.890</td><td>7.0M</td></tr><tr><td>RAM w/ gt kernel [32]</td><td></td><td>33.39</td><td>0.903</td><td>35.6M</td></tr><tr><td>Ours w/ gt kernel</td><td></td><td>32.92</td><td>0.895</td><td>0.6M</td></tr></table>

Table III evaluates the same networks on our real dataset, with both our estimated kernel and the ground-truth trajectory. EDI’s closed-form reconstruction attains the second-highest PSNR among the baselines (30.54, behind EFNet’s 31.67) and a competitive SSIM, yet still trails every ODF-kernelbased method. Replacing the ground-truth kernel with our estimated one costs each network under 1 dB in PSNR and 0.02 in SSIM, confirming ODF motion estimation recovers kernels close to the true motion. Our network trails the much larger RAM by under 1 dB in PSNR at both kernel settings, though it outperforms DWDN despite an order of magnitude fewer parameters than either, a capacity trade-off rather than a kernel-quality one: both receive the identical kernel, so the gap traces to RAM’s 35.6M parameters and plug-and-play restoration affording more capacity than our 0.6M network spends (qualitative comparison in Fig. 8).

EFNet-GoPro  
![](images/dc15ed4ba5622463b80113f6a5869282b4c3b8201c8f3fd018fca2f017e09936.jpg)

![](images/bde995c890a2917743b1f852a655d5b83ea6a20dd68b75fb5eee6e62549cdb86.jpg)

Blurred  
EDI  
NAFNet  
Fig. 7. Qualitative comparison on EventAid-B. Blind methods (NAFNet, Restormer) leave residual blur, EFNet introduces over-sharpening and brightness shifts, and our method restores sharp detail while preserving the original tone  
![](images/fc17abd4b2e18b9b2b6c60d41cbffe9ac12f57a5fcbdc697108de80d2bd12a1e.jpg)  
Blurred

![](images/3d4dbc215f156547a5702c687bc26e5db4fb81bab39fb94f8711da1136b7e300.jpg)  
ODF kernel + DWDN

Restormer  
![](images/9d2c0770fc20e6ca9dc5ea9bf46870d76f5a09177c76874e7177b5cd194b0824.jpg)  
ODF kernel + RAM

EFNet-REBlur  
Ours  
![](images/e3ba0242d112ed6dd4fc4f58e006a5213103e2c94660765f7be5c3236f4be8ab.jpg)  
Ours

GT  
![](images/3e89671217161f96613d4a7574a55cfc2677f88c42fba4f08973a19795dba186.jpg)  
GT  
Fig. 8. Qualitative comparison on our real captured dataset, using our estimated blur kernel. Our lightweight network controls edge artifacts better than RAM, though RAM recovers marginally sharper texture.

Speed. ODF motion estimation processes each event in under 0.7 µs on a single CPU thread (14th Gen Intel Core i7- 14650HX), since it only looks up and averages precomputed distance vectors instead of solving an optimization problem. The deblurring network runs at 55 ms per $1 2 8 0 \times 7 2 0$ image on an Nvidia A800 GPU and reaches real time, above 60 fps, at $5 1 2 \times 5 1 2$

We do not report a separate ablation study: ODF motion estimation follows from a closed-form derivation rather than empirically tuned modules, and the deblurring network’s two main design choices are already isolated within the main comparison tables. Training on motion-estimation-noise kernels is isolated by the estimated- and ground-truth-kernel rows in Table III, which differ by under 1 dB in PSNR and 0.02 in SSIM; the event-mask conditioning is isolated by the plain-USRNet-tiny row in Table II, which trails our network by 0.88 dB PSNR and 0.017 SSIM at matched parameter count.

## C. Pupil Motion Estimation for Eye Tracking

We evaluate pupil tracking accuracy on the public near-eye event dataset of Angelopoulos et al. [5] (24 subjects), using EllSeg [19] pseudo-labels on the frame images as reference, since the original dataset’s per-frame pupil extraction is not public. Tracking is initialized from the first valid ellipse after each blink and evaluated against the temporally nearest EllSeg label. Across subjects, the median pupil IoU is 0.85 (left) and 0.86 (right), with a median center error of 1.45 and 1.41 pixels, and the longest continuous track reaches 1745 frames (about 70 s at 25 fps), showing tracking sustained through long fixations. On a near-eye AR prototype using a Prophesee GenX320 event sensor [27], initialized from an E2VID [28] reconstruction, the tracker recovers both pupil and glint positions throughout the recorded sequences (Fig. 9), processing 28 µs per event on a single thread, dropping to 10 µs on separate threads, comfortably real time. We further compare the power draw of a near-eye module built around the GenX320 against one around an OmniVision OVM6211 frame sensor, with the frame sensor’s LED lit only during its own exposure window (Table IV). The event sensor draws under a third of the frame sensor’s power at every rate tested, but its continuously-lit LED draws more than the frame sensor’s duty-cycled LED at 60 fps, since our glint detection is not yet optimized for duty cycling. The sensor-side saving still dominates, so the event-driven module draws less overall, 127.1 mW against 215.0 mW at 60 fps, widening as the frame rate increases: the tracker’s low data rate lowers system power even though it does not win on every sub-component.

TABLE IV  
HARDWARE POWER COMPARISON OF A FRAME-CAMERA AND AN EVENT-CAMERA NEAR-EYE TRACKING MODULE (BOLD: LOWEST TOTAL POWER).
<table><tr><td>Sensor</td><td>Rate (fps)</td><td>Sensor ↓ (mW)</td><td>LED duty</td><td>LED ↓ (mW)</td><td>Total ↓ (mW)</td></tr><tr><td rowspan="3">Frame camera</td><td>60</td><td>161.8</td><td>30%</td><td>53.2</td><td>215.0</td></tr><tr><td>90</td><td>163.6</td><td>45%</td><td>75.6</td><td>239.2</td></tr><tr><td>120</td><td>165.4</td><td>60%</td><td>100.8</td><td>266.2</td></tr><tr><td>Event camera</td><td></td><td>51.2</td><td>100%</td><td>75.9</td><td>127.1</td></tr></table>

![](images/5f3dd96abbffd0b6bbb6de95464903dec52607796f743b6b44edbfffd148f040.jpg)  
t<sub>1</sub>

![](images/5148f3c105aa9d5b9e47bde681497c1a4c86739bd227aeef4ce660869d5c9863.jpg)

![](images/2bdb8785b333f1c440d9d6a61184f4b4e52cdc1834e553382bd51b01b4f919d4.jpg)  
t<sub>2</sub>  
t<sub>3</sub>  
Fig. 9. Pupil and glint tracking on near-eye event data at three time steps t<sub>1</sub>–t<sub>3</sub>. Events are shown in red (positive) and blue (negative), with the fitted pupil ellipse in yellow and located glints in green.

## VI. CONCLUSION

We presented ODF motion estimation, which replaces iterative optimization with a single averaging step over a precomputed distance vector field, and showed this core module generalizes to two downstream tasks. For deblurring, the recovered trajectory yields a blur kernel for a lightweight network trained on motion-estimation-noise simulation, matching or exceeding prior event-assisted methods at a fraction of their parameter count. For eye tracking, the same distance vectors filter events directionally for stable, low-power pupil and glint tracking. Motion estimation itself runs at sub-microsecond latency per event, so the low-latency advantage of event cameras is preserved end to end rather than traded away for accuracy.

Limitations. Our method assumes globally consistent motion, sufficient orientation diversity among tracked edges, and enough events to constrain the estimate, so dynamic scenes, templates dominated by one edge orientation, and sparse event regions each degrade accuracy. Both applications also assume 2-DoF translation rather than general 6-DoF motion, valid for gimbal- or PTZ-mounted cameras and similar platforms but not for larger rotation or affine motion, which we leave to future work.

## REFERENCES

[1] Eirikur Agustsson and Radu Timofte. NTIRE 2017 challenge on single image super-resolution: Dataset and study. In IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 1122–1131, 2017. 6

[2] Mohammed Almatrafi, Raymond Baldwin, Kiyoharu Aizawa, and Keigo Hirakawa. Distance surface for event-based optical flow. IEEE transactions on pattern analysis and machine intelligence, 42(7):1547–1556, 2020. 2, 3

[3] I. Alzugaray and M. Chli. Asynchronous multi-hypothesis tracking of features with event cameras. In 2019 International Conference on 3D Vision (3DV), pages 269–278, 2019. 1, 3

[4] I. Alzugaray and M. Chli. HASTE: multi-hypothesis asynchronous speeded-up tracking of events. In 31st British Machine Vision Conference 2020, BMVC 2020. BMVA Press, 2020. 1, 2, 3, 5, 6

[5] Anastasios N. Angelopoulos, Julien N. P. Martel, Amit P. Kohli, et al. Event-based near-eye gaze tracking beyond 10,000 hz. IEEE Transactions on Visualization and Computer Graphics, 27(5):2577–2586, 2021. 2, 7

[6] Pietro Bonazzi, Sizhen Bian, Giovanni Lippolis, et al. Retina: Lowpower eye tracking with event camera and spiking hardware. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 5684–5692, 2024. 2

[7] Giacomo Boracchi and Alessandro Foi. Modeling the performance of image restoration from motion blur. IEEE Transactions on Image Processing, 21(8):3502–3517, 2012. 4, 5

[8] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun. Simple baselines for image restoration. arXiv preprint arXiv:2204.04676, 2022. 6

[9] Jiangxin Dong, Stefan Roth, and Bernt Schiele. Dwdn: Deep wiener deconvolution network for non-blind image deblurring. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12):9960–9976, 2021. 2, 4, 6

[10] Peiqi Duan, Boyu Li, Yixin Yang, et al. EventAid: Benchmarking event-aided image/video enhancement algorithms with real-captured hybrid dataset. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(8):6959–6973, 2025. 6

[11] Guillermo Gallego, Tobi Delbruck, Garrick Michael Orchard, Chiara Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger, Andrew Davison, Jorg Conradt, Kostas Daniilidis, et al. Event-based vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020. 1

[12] Guillermo Gallego, Henri Rebecq, and Davide Scaramuzza. A unifying contrast maximization framework for event cameras, with applications to motion, depth, and optical flow estimation. In CVPR, 2018. 1, 2, 3, 5, 6

[13] Ling Gao, Daniel Gehrig, Hang Su, Davide Scaramuzza, and Laurent Kneip. An n-point linear solver for line and motion estimation with event cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14596–14605, 2024. 2, 4

[14] Daniel Gehrig, Henri Rebecq, Guillermo Gallego, et al. EKLT: Asynchronous photometric feature tracking using events and frames. International Journal of Computer Vision, 128(3):601–618, 2020. 2

[15] Elias Daniel Guestrin and Moshe Eizenman. General theory of remote gaze estimation using the pupil center and corneal reflections. IEEE Transactions on Biomedical Engineering, 53(6):1124–1133, 2006. 2, 5

[16] Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Single image super-resolution from transformed self-exemplars. In IEEE Conference on Computer Vision and Pattern Recognition, pages 5197–5206, 2015. 6

[17] Xueyan Huang, Yueyi Zhang, and Zhiwei Xiong. Progressive spatiotemporal alignment for efficient event-based motion estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1537–1546, 2023. 1, 2, 3

[18] Haram Kim and H Jin Kim. Real-time rotational motion estimation with contrast maximization over globally aligned events. IEEE Robotics and Automation Letters, 6(3):6016–6023, 2021. 1, 3

[19] Rakshit S. Kothari, Aayush K. Chaudhary, Reynold J. Bailey, et al. EllSeg: An ellipse segmentation framework for robust gaze tracking. IEEE Transactions on Visualization and Computer Graphics, 27(5):2757–2767, 2021. 7

[20] Manohar Kuse and Shaojie Shen. Robust camera motion estimation using direct edge alignment and sub-gradient method. In 2016 IEEE international conference on robotics and automation (ICRA), pages 573– 579. IEEE, 2016. 2, 3

[21] Nealson Li, Muya Chang, and Arijit Raychowdhury. E-Gaze: Gaze estimation with event camera. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(7):4796–4811, 2024. 2

[22] Shijie Lin, Yingjun Zhang, Daoyuan Huang, et al. Fast event-based double integral for real-time robotics. In IEEE International Conference on Robotics and Automation (ICRA), pages 796–803, 2023. 2

[23] Min Liu and Tobi Delbruck. Adaptive time-slice block-matching optical flow algorithm for dynamic vision sensors. In BMVC, 2018. 2, 3

[24] Elias Mueggler, Henri Rebecq, Guillermo Gallego, et al. The eventcamera dataset and simulator: Event-based data for pose estimation, visual odometry, and SLAM. The International Journal of Robotics Research, 36(2):142–149, 2017. 5, 6

[25] Urbano Miguel Nunes and Yiannis Demiris. Robust event-based vision model estimation by dispersion minimisation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(12):9561–9573, 2021. 1, 3

[26] Liyuan Pan, Cedric Scheerlinck, Xin Yu, Richard Hartley, Miaomiao Liu, and Yuchao Dai. Bringing a blurry frame alive at high frame-rate with an event camera. In Proc. CVPR, pages 6820–6829, 2019. 2, 6

[27] Prophesee. Event-based metavision sensor GenX320, 2023. https:// www.prophesee.ai/event-based-sensor-genx320/. 7

[28] Henri Rebecq, Rene Ranftl, Vladlen Koltun, et al. High speed and´ high dynamic range video with an event camera. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(6):1964–1980, 2021. 7

[29] Yeqing Shen, Shang Li, and Kaiwei Song. Restoring real-world degraded events improves deblurring quality. In ACM International Conference on Multimedia, pages 4957–4966, 2024. 2

[30] Lei Sun, Daniel Gehrig, Christos Sakaridis, Mathias Gehrig, Jingyun Liang, Peng Sun, Zhijie Xu, Kaiwei Wang, Luc Van Gool, and Davide Scaramuzza. A unified framework for event-based frame interpolation with ad-hoc deblurring in the wild. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2

[31] Lei Sun, Christos Sakaridis, Jingyun Liang, Qi Jiang, Kailun Yang, Peng Sun, Yaozu Ye, Kaiwei Wang, and Luc Van Gool. Event-based fusion for motion deblurring with cross-modal attention. In Proc. ECCV, pages 412–428. Springer, 2022. 2, 6

[32] Matthieu Terris, Samuel Hurault, Maxime Song, and Julian Tachella.´ Reconstruct anything model: a lightweight foundational model for computational imaging. arXiv preprint arXiv:2503.08915, 2025. 6

[33] Radu Timofte, Eirikur Agustsson, Luc Van Gool, et al. NTIRE 2017 challenge on single image super-resolution: Methods and results. In IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 1110–1121, 2017. 6

[34] Subeesh Vasu, Venkatesh Reddy Maligireddy, and AN Rajagopalan. Non-blind deblurring: Handling kernel uncertainty with cnns. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3272–3281, 2018. 4

[35] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 1905–1914, October 2021. 5

[36] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In Proc. CVPR, pages 5728–5739, 2022. 6

[37] Kai Zhang, Luc Van Gool, and Radu Timofte. Deep unfolding network for image super-resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3217–3226, 2020. 2, 5, 6

[38] Kai Zhang, Yawei Li, Wangmeng Zuo, et al. Plug-and-play image restoration with deep denoiser prior. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10):6360–6376, 2022. 2, 5

[39] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4791–4800, October 2021. 5

[40] Guangrong Zhao, Yiran Yang, Jingwei Liu, et al. EV-Eye: Rethinking high-frequency eye tracking through the lenses of event cameras. Advances in Neural Information Processing Systems, 36:62169–62182, 2023. 2

[41] Yi Zhou, Guillermo Gallego, and Shaojie Shen. Event-based stereo visual odometry. IEEE Transactions on Robotics, 37(5):1433–1450, 2021. 3