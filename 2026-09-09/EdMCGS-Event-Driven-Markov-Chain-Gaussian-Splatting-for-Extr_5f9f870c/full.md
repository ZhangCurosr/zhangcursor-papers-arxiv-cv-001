# EdMCGS: Event-Driven Markov Chain Gaussian Splatting for Extreme-Low-Frame-Rate Dynamic Scene Reconstruction

Yuzhong Wang<sup>a</sup>, Wenmin Wang<sup>a,∗</sup>, Xinxing Yu<sup>a</sup>

<sup>a</sup>Macau University ofScience and Technology, Macau, China

## Abstract

We present EdMCGS (Event-driven Markov chain Gaussian Splatting), an end-to-end method for reconstructing dynamic 3D scenes from extreme-low-frame-rate RGB together with an event stream, which can then be rendered at any intermediate timestamp. Methods relying solely on RGB images generate numerous artifacts due to the lack of evidence from between consecutive frames. To supply this missing evidence, we model the scene motion as an event-driven Markov chain, in which the sparse RGB frames anchor the state at their own timestamps while the events recorded within an interval drive the transition across it. Since the transition reads the events of the current interval, it remains active at inference and produces the in-between motion of the 3D Gaussians directly from the events rather than by interpolation, which sets our method apart from prior work that uses events only as training-time supervision. The state is carried by a compact set of control points, each driven by the events sampled in the neighborhood of its own image projection, and a temporal local isometry term keeps the propagated motion locally rigid. Experiments on synthetic and real-world scenes show that EdM-CGS outperforms both RGB-based and event-based baselines, while rendering in real time with far fewer Gaussians than the strongest event-based baseline. We release our source code and a new dataset at https://github.com/joseclipse/EdMCGS.

Keywords: dynamic scene reconstruction, 3D Gaussian Splatting, event camera, Markov chain, extreme-low-frame-rate, novel view synthesis

## 1. Introduction

High-quality 3D scene reconstruction and novel view synthesis have far-reaching applications across a wide range of domains, including autonomous driving, embodied intelligence and virtual reality. Neural Radiance Fields (NeRF) [1] demonstrated photorealistic rendering quality for static scenes, inspiring a substantial line of follow-up work that extended this framework to dynamic settings [2, 3, 4] and made the reconstruction of time-varying scenes possible. However, because NeRF-based methods rely on implicit functions to jointly encode scene geometry and appearance, they sufer from prohibitively slow rendering speeds and high memory consumption.

The recent advent of 3D Gaussian Splatting (3DGS) [5] has enabled real-time, high-quality novel view synthesis by replacing the implicit volumetric representation with an explicit, point-based one, a collection of 3D Gaussian ellipsoids whose parameters are optimized end-to-end via diferentiable rasterization. Building on this framework, 3DGS has quickly proven versatile across a broad range of tasks, from faster optimization [6, 7] to 3D scene segmentation [8, 9]. Leveraging the expressiveness and eficiency of 3DGS, numerous approaches have further extended it to dynamic scenes, representing time-varying content as a set of deformable Gaussian primitives whose geometry and appearance evolve over time, as in Deformable 3DGS [10] and 4D-GS [11]. Owing to fully diferentiable, end-to-end optimization, 3DGS-based methods have rapidly surpassed NeRF-based methods on dynamic novel view synthesis. All of these methods nonetheless rely on densely sampled RGB input, where consecutive frames are close enough in time for the deformation field to interpolate the motion between them.

In many practical settings this assumption breaks down and the available RGB frames are sparse in time. Low frame rates arise from bandwidth and storage constraints, from long exposures in low-light capture, and from commodity hardware. Storage is an especially common pressure: full-frame-rate video is so bulky that it is often kept on a rolling bufer and overwritten after a short retention window, so that only recent footage can be retrieved. Keeping only a fraction of the RGB frames, for example one third to one sixth, together with the compact stream of an event camera is therefore an attractive way to retain reconstructable dynamic footage for far longer. Under such extreme-low-frame-rate input, however, a model trained on RGB frames alone has no evidence about what happens between two consecutive frames; it tends to overfit the few observed timestamps and produces torn or ghosted geometry when asked to render the unobserved in-between motion. The core dificulty is not the representation but the lack of temporal evidence in the gaps between frames.

Event cameras ofer exactly this missing evidence. As neuromorphic sensors, they asynchronously report per-pixel brightness changes at microsecond temporal resolution, with high dynamic range and low power consumption [12]. Rather than synchronous intensity frames at fixed intervals, an event camera produces a continuous stream that densely samples precisely the intervals where RGB frames are absent. This makes the event stream a natural complement to low-frame-rate RGB for dynamic reconstruction: the two modalities are deeply complementary, as RGB frames provide absolute appearance anchors at a coarse temporal resolution, while the event stream encodes fine-grained brightness changes that bridge the large temporal gaps between frames.

The promise of this complementarity has begun to be explored. A growing body of work reconstructs static scenes from events [13, 14, 15], and a handful of very recent methods target dynamic scenes [16, 17, 18, 19]. Yet across this nascent line three characteristics are common. First, the events shape the deformation only during training, where the deformation is parameterized as a function of time alone, so that at inference it is queried without events and the unobserved in-between motion is produced by temporal interpolation rather than driven by event evidence. Second, these methods tend to be heavy. DEGS [16], for instance, relies on a pretrained external event-flow network with per-scene low-rank fine-tuning, a geometry-aware event–Gaussian association, an explicit decomposition of scene and camera motion, and a separate pose network; other methods depend on dynamic-static or binary motion masks [17, 18] or on multi-view event rigs [19]. Finally, most of these methods and their datasets are not publicly available, and the one line whose code is released [17] targets static-dominant captures, an assumption that we show empirically to leave much of the event evidence unexploited once the foreground motion dominates (Sec. 4.3), so the community still lacks a reproducible baseline for the fully dynamic, extreme-low-frame-rate regime.

To address these limitations, we propose EdMCGS (Event-driven Markov chain Gaussian Splatting), a lightweight, end-to-end, and mask-free framework that reconstructs dynamic 3D scenes from sparse extreme-low-frame-rate RGB together with an event stream, and that can be rendered at any intermediate timestamp. Our central idea is to cast the motion as an event-driven Markov chain, in which the sparse RGB frames partition the timeline into intervals and anchor the state at their timestamps, while within each interval the event stream drives the transition from one state to the next. Because the transition is conditioned on the events, it remains active at inference, so that at an unobserved timestamp the in-between motion is produced directly from the events rather than guessed by interpolation, which is the key distinction from prior event-supervised methods. The state that the chain transitions is carried by a compact set of control points which drive the canonical Gaussians through linear blend skinning [20], a low-rank motion representation that we adopt from SC-GS [21]. A temporal local isometry regularizer keeps the propagated motion locally rigid. Our main contributions are summarized as follows:

• We propose EdMCGS, which models the motion of a dynamic scene as an eventdriven Markov chain, where the events recorded between two consecutive RGB frames drive the transition of the scene state from one frame to the next.

• We drive the chain by local event sampling, where every control point reads its own motion code from the accumulated event map at its image projection, so that the events fired by a moving part act on precisely the points covering that part instead of being collapsed into a single global descriptor.

• We introduce a temporal local isometry (TLI) regularizer that keeps the propagated motion locally rigid by preserving inter-point distances across nearby deformed timestamps, which stabilizes the rendering at unobserved in-between timestamps.

• We release our source code together with a new synthetic benchmark for eventbased monocular dynamic scene reconstruction. On both synthetic and realworld scenes EdMCGS outperforms RGB-based and event-based methods at every frame rate we test, and it does so while training faster, rendering faster and using several times fewer Gaussians than the strongest event-based baseline.

## 2. Related Work

## 2.1. RGB-based Dynamic Scene Reconstruction

Novel view synthesis of dynamic scenes from RGB input has been studied intensively since Neural Radiance Fields (NeRF) [1]. One line of work extends NeRF to the temporal domain, either by conditioning an implicit field on a deformation that maps each observation back to a canonical space [2, 4], or by adopting explicit space-time structures such as planar or grid factorizations for eficiency [22, 23, 24]. Although these methods achieve high-quality interpolation, their implicit volumetric representation makes both training and rendering slow. 3D Gaussian Splatting (3DGS) [5] instead represents the scene with an explicit set of anisotropic Gaussians that are rendered by diferentiable rasterization, which enables real-time and high-fidelity synthesis. To extend 3DGS to dynamic scenes, some methods attach a time-conditioned deformation field to a canonical set of Gaussians [10, 11, 25], while others model the motion directly with 4D primitives or per-Gaussian trajectories [26, 27]. Among the deformation-based methods, several drive the dense Gaussians through a sparse set of learnable control points whose motion is regularized by an as-rigid-as-possible term [21, 28]. More recently, MCGS [29] explored temporal motion propagation based on a Markov chain over control points, where the transition is predicted from a memory of past states with temporal attention and therefore still relies on densely sampled RGB observations. Our method keeps that sparse control point formulation, but addresses a diferent problem with a diferent transition mechanism. A related line of work tackles intra-frame motion blur in dynamic 3DGS [30, 31, 32], which is orthogonal to our setting because we address the inter-frame gaps left by sparse but sharp frames rather than blur within a single exposure.

## 2.2. Event-based Static Scene Reconstruction

Event cameras report asynchronous, per-pixel brightness changes at microsecond resolution with high dynamic range [12]. Beyond low-level sensing, they have also been used for higher-level vision tasks such as object recognition [33], motion feature extraction [34], and knowledge transfer across event domains [35], and a large body of work has incorporated them into static 3D reconstruction. Within the NeRF framework, E-NeRF [13] couples the event generation model with volumetric rendering, and later work improves its robustness to event noise and varying contrast thresholds [36], to inaccurate poses and uneven event density [37], and to motion blur [38, 39]. More recently, the explicit and eficient nature of 3DGS has been combined with events for static scenes, reconstructing from the event stream on its own or together with blurry or sparse RGB frames, as in recent event-based 3DGS methods [40, 14, 41]. These methods show that the event stream is a powerful geometric cue, but because they assume a static scene they do not model the temporal deformation that is central to our problem.

## 2.3. Event-based Dynamic Scene Reconstruction

Reconstructing dynamic scenes with events is a nascent and rapidly moving direction, and it is here that our work is positioned. DE-NeRF [42], which first combined events with RGB for a dynamic radiance field, estimates a per-event color and jointly optimizes a deformable NeRF, although it inherits the slow rendering of NeRF and uses the events only as a supervision signal. DEGS [16] brings event-guided deformation to 3DGS by extracting event-flow trajectories with a pretrained external flow estimator that is fine-tuned per scene through low-rank adaptation, by building a geometry-aware association between events and Gaussians from unprojected depth, and by decomposing object motion from camera ego-motion with a separate pose network. Although efective, this pipeline is heavy because it depends on several external components. Event-boosted Deformable 3DGS [17] focuses instead on modeling the event contrast threshold and on a dynamic-static decomposition for faster rendering, which requires explicit dynamic-region masks. Ev4DGS [18] targets non-rigid reconstruction from a monocular event stream alone, using a low-rank deformation basis together with binary masks generated from the events, while E-4DGS [19] addresses the multi-view setting in which the scene is captured by a 360<sup>◦</sup> rig of event cameras.

Across this line, the events serve only as training-time supervision, so at inference the in-between motion is interpolated rather than driven by evidence, and, apart from [17], neither the code nor the data is released. In contrast, EdMCGS reconstructs dynamic scenes from sparse RGB and a monocular event stream by casting the deformation as an event-driven Markov chain over control points whose transitions remain active at inference, and it is lightweight and mask-free, requiring no external opticalflow model, no per-scene fine-tuning, and no pose network. We release our code and dataset as a reproducible baseline for the extreme-low-frame-rate setting.

## 3. Method

We propose Event-driven Markov chain Gaussian Splatting (EdMCGS), shown in Fig. 1, a new method for reconstructing a dynamic 3D scene from an extreme-lowframe-rate RGB sequence and an asynchronous event stream, so that the scene can be rendered at any intermediate timestamp not covered by an RGB frame. EdMCGS represents the scene as a canonical set of 3D Gaussians whose motion is carried by a sparse set of control points, and its central idea is to cast the deformation as an eventdriven Markov chain in which the sparse RGB frames anchor the control points’ state at their timestamps while the event stream drives the transition within each interval. Because the transition is driven by the events, it remains active at inference, so that the motion at the unobserved in-between timestamps is produced directly from the events rather than interpolated from the RGB frames.

In this section, we first briefly review the necessary background to establish the foundation for our approach (Sec. 3.1). We then introduce our event-driven Markov chain (Sec. 3.2), followed by local event sampling (Sec. 3.3), the temporal local isometry regularizer (Sec. 3.4) and the training objective (Sec. 3.5).

![](images/b4789216c71464c8eb97dc3748083ca04a0ade02baae06b02d404e1d74975715.jpg)  
Figure 1: Overview of EdMCGS. (a) We initialize the control points by sampling from the canonical space. (b) We cast the deformation as an event-driven Markov chain, where the events accumulated in the interva $[ \tau _ { n } , t ]$ between two RGB anchors drive a Markov transition of the control point state. (c) At a query time t, the coarse RGB-driven deformation predicted by the MLP and the fine event-driven correction from the Markov transition are added to displace the control points, which in turn deform the Gaussians by linearblend skinning before rasterization. (d) Local event sampling: the events in $[ \tau _ { n } , t ]$ are accumulated into a polarity map, encoded by a shallow CNN Φ, and bilinearly sampled at each control point’s projection to form its local motion code.

## 3.1. Preliminaries

## 3.1.1. Deformable 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) [5] represents a scene by a set of N anisotropic Gaussians $\mathcal { G } = \{ G _ { i } \} _ { i = 1 } ^ { N }$ . At time t, Gaussian i is parameterized by a center $\boldsymbol { \mu } _ { i } ^ { t } \in \mathbb { R } ^ { 3 }$ ， a rotation quaternion $q _ { i } ^ { t } ,$ a scale $s _ { i } ^ { t } \in \mathbb { R } ^ { 3 }$ , an opacity $\sigma _ { i } ,$ and spherical-harmonic color coeficients $s h _ { i }$ , where the last two are time-invariant. The spatial response is

$$
\begin{array} { c } { { G _ { i } ( \boldsymbol { x } ) = \exp \big ( - \frac 1 2 ( \boldsymbol { x } - \boldsymbol { \mu } _ { i } ^ { t } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( \boldsymbol { x } - \boldsymbol { \mu } _ { i } ^ { t } ) \big ) , } } \\ { { \Sigma _ { i } = R ( q _ { i } ^ { t } ) S ( s _ { i } ^ { t } ) S ( s _ { i } ^ { t } ) ^ { \top } R ( q _ { i } ^ { t } ) ^ { \top } , } } \end{array}\tag{1}
$$

with $R ( \cdot )$ the rotation matrix and $S \left( \cdot \right)$ the diagonal scale matrix. Images are formed by projecting the Gaussians to 2D image plane and α-blending them front to back, which is fully diferentiable and real-time.

To model a dynamic scene, Deformable 3DGS [10] keeps $\mathcal { G }$ as a canonical (timezero) configuration and predicts, for a normalized time $t \in [ 0 , 1 ]$ , a per-Gaussian deformation

$$
( \delta \mu _ { i } , \delta q _ { i } , \delta s _ { i } ) = f _ { \theta } ( \gamma ( s g ( \mu _ { i } ) ) , \gamma ( t ) )\tag{2}
$$

that maps each canonical Gaussian to its state at t, where $s g ( \cdot )$ indicates a stop-gradient operation and $\gamma ( \cdot )$ is the standard sinusoidal positional encoding. This MLP must be evaluated for all $N \sim 1 0 ^ { 5 }$ Gaussians at every step, and, more importantly for our setting, under extreme-low-frame-rate input it overfits the few observed timestamps and tears apart at unobserved ones. This is the starting point of our method.

## 3.1.2. Markov Chain

A discrete-time stochastic process $\{ S _ { t } \mid S _ { t } \in S$ and $t \in T \}$ is a first-order Markov chain if its future is conditionally independent of its past given the present,

$$
P ( S _ { t + 1 } = s _ { t + 1 } \ | \ S _ { t } = s _ { t } , S _ { t - 1 } = s _ { t - 1 } , \ldots , S _ { 1 } = s _ { 1 } ) = P ( S _ { t + 1 } = s _ { t + 1 } \ | \ S _ { t } = s _ { t } ) ,\tag{3}
$$

so that the evolution of the process is fully described by its current state and a transition operator. In Sec. 3.2 we instantiate the state as the configuration of control points and the transition as an event-driven operator, so that the motion within an interval is propagated from the preceding RGB frame by the events alone.

## 3.1.3. Event Generation Model

An event camera reports asynchronous events $e \ : = \ : ( x , y , t , p )$ , with pixel location $\boldsymbol { u } \ = \ ( x , y )$ , timestamp t, and polarity $p \in \{ + 1 , - 1 \}$ . An event is fired whenever the log-luminance $L ( u , t ) = \log { \left( \mathcal { I } ( u , t ) + \epsilon \right) }$ at a pixel changes by a contrast threshold C. Consequently, integrating the signed events over an interval $[ t _ { a } , t _ { b } ]$ predicts the logluminance diference,

$$
L ( u , t _ { b } ) - L ( u , t _ { a } ) = C \sum _ { e \in [ t _ { a } , t _ { b } ] } p _ { e } ,\tag{4}
$$

which is the event generation model (EGM). It links a rendered brightness change to the accumulated events and is the basis of our event supervision.

## 3.2. Event-Driven Markov Chain

We begin by sampling a sparse set of points from the canonical Gaussians by farthest point sampling, which we define as a set of control points $M = \{ M _ { j } \} _ { j = 1 } ^ { M } \subset \mathcal { G }$ inspired by the sparse control points in SC-GS [21]. These control points are connected to their nearby Gaussians by K-nearest-neighbor search and drive the deformation of those Gaussians through linear-blend skinning [20], weighting each Gaussian by a Gaussian kernel of its distance to the control point. While control points are widely used [21, 28, 29], our contribution is not the control points themselves but how they are moved.

Let the extreme-low-frame-rate RGB frames be captured at the sparse timestamps $\tau _ { 1 } < \tau _ { 2 } < \cdots < \tau _ { N _ { \mathrm { r g b } } }$ , which partition [0 1] into intervals. We take the state of the Markov chain at time t to be the configuration of control points $S ^ { t } = \{ \mu _ { j } ^ { t } \} _ { j = 1 } ^ { M }$ , namely the deformed control point positions. For a query time t, let $\tau ( t ) = \operatorname* { m a x } \{ \tau _ { n } : \tau _ { n } \leq t \}$ denote the immediately preceding RGB anchor. Since an RGB frame is observed at each $\tau _ { n } .$ , the state at an anchor is well constrained by the photometric loss, whereas the in-between states must be inferred.

We model the evolution between consecutive anchors as a first-order Markov transition: the state at a query time t depends on the past only through the preceding anchor state $S ^ { \tau ( t ) }$ and the events $E [ \tau ( t ) .$ t] observed in between, without any earlier history,

$$
S ^ { t } = \mathcal { T } ( S ^ { \tau ( t ) } , E [ \tau ( t ) , t ] ) .\tag{5}
$$

This instantiates the Markov property of Eq. (3), with the RGB frames acting as the anchors that pin the state and the events driving the transition operator T within each interval, so that the deformation remains active at inference.

We realize this transition independently for each control point. To the RGB prior $\delta \mu _ { j } ^ { f , t }$ , obtained by evaluating the deformation network of Eq. (2) at the M control points rather than at all N Gaussians so that the RGB-driven motion stays low-dimensional, we add an event-driven term, so that the total displacement of control point j is

$$
\delta \mu _ { j } ^ { t } = \delta \mu _ { j } ^ { f , t } + g _ { \phi } ( \gamma ( \mu _ { j } ) , m _ { j } ^ { t } ) ,\tag{6}
$$

where $g _ { \phi }$ is a small MLP, and $m _ { j } ^ { t }$ is a motion code built from the events accumulated since the anchor, defined below. Since $\delta \mu _ { j } ^ { f , t }$ is supervised only by the RGB frames, it is a smooth interpolator in t that gives a sensible baseline near the observed timestamps but carries no evidence of the true motion in the gaps between sparse frames. The Markov transition is thus realized entirely by the event term $g _ { \phi } \colon$ at $t = \tau ( t )$ the accumulation window is empty and the zero-initialized $g _ { \phi }$ vanishes, so that the state reduces to the anchor $S ^ { \tau ( t ) }$ , and as t advances the accumulated events propagate it forward.

The canonical control points $\{ \mu _ { j } \}$ are fixed, so the anchor state $S ^ { \tau ( t ) }$ is a deterministic function of $\tau ( t )$ , which is in turn determined by t; conditioning on $( \mu _ { j } , t )$ is therefore informationally equivalent to conditioning on $S ^ { \tau ( t ) }$ , and the two parameterizations describe the same operator. What makes the process Markovian is not the parameterization of the prior but the accumulation window of the motion code, since $m _ { j } ^ { t }$ reads only the events fired in [τ(t) t]. The increment that carries the state from the anchor to t therefore depends on the current interval alone and is conditionally independent of everything that occurred before $\tau ( t )$ , which is exactly the property required by Eq. (3). Each RGB frame resets the window, so errors accumulated inside one interval are not propagated across the anchors, a property that a transition conditioned on a memory of past states does not have.

## 3.3. Local Event Sampling

It remains to define the motion code $m _ { j } ^ { t }$ that drives the transition, which we build from the interval events in three steps. First, the events fired in the anchored window [τ(t) t] are accumulated, by polarity, into a 2D map $I _ { E } ^ { t } = \mathsf { A c c } ( E [ \tau ( t ) , t ] ) \in \mathbb { R } ^ { H \times W }$ on the event camera image plane. Second, a shallow CNN Φ encodes this map into a feature map that preserves its spatial layout. Third, we project each control point onto the event camera and read its feature by bilinear sampling at that location,

$$
m _ { j } ^ { t } = \mathrm { b i l i n e a r } \big ( \Phi ( I _ { E } ^ { t } ) , \pi _ { \mathrm { e v } } ( \mu _ { j } + \delta \mu _ { j } ^ { f , t } ) \big ) ,\tag{7}
$$

where $\pi _ { \mathrm { e v } }$ is the projection onto the event camera and $\mu _ { j } + \delta \mu _ { j } ^ { f , t }$ is the prior-deformed control point.

Each control point samples the event feature in the neighborhood of its own image projection, rather than from a global summary of the event stream. Because the events triggered by a moving part fall near the projection of the control points that cover it, this local sampling lets the events of a region drive exactly the control points of that region, instead of being averaged together with camera and background events as in a global pooling. The neighborhood is defined implicitly by the receptive field of Φ and the bilinear sampling, without any explicit radius.

The branch $g _ { \phi }$ is zero-initialized at its last layer and warmed in gradually, so optimization first establishes the RGB-anchored field before the events contribute, and sharing $g _ { \phi }$ across control points together with the local sampling in Eq. (7) keeps the transition spatially coherent, which is further encouraged by a KNN Laplacian smoothness term (Sec. 3.5).

## 3.4. Temporal Local Isometry Regularization

To keep the propagated motion locally rigid without imposing a global rigid model, we regularize the deformation to be locally isometric in time. On the canonical control points we build a KNN graph with edge set E. For an edge $( j , k ) \in \mathcal { E } _ { \mathrm { {  } } }$ , let

$$
\ell _ { j k } ( t ) = | | \mu _ { j } ^ { t } - \mu _ { k } ^ { t } | |\tag{8}
$$

be its deformed length at time t. At a sampled query t we take the two nearby times $\begin{array} { r } { t _ { 1 , 2 } = t \mp \frac { \Delta t } { 2 } } \end{array}$ and penalize the change of edge length between them,

$$
\mathcal { L } _ { \mathrm { T L I } } = \frac { 1 } { \vert \mathcal { E } \vert } \sum _ { ( j , k ) \in \mathcal { E } } \left( \ell _ { j k } ( t _ { 1 } ) - \ell _ { j k } ( t _ { 2 } ) \right) ^ { 2 } .\tag{9}
$$

A locally rigid motion preserves inter-point distances and is therefore not penalized, while only stretching and shearing are. We stress that ${ \mathcal { L } } _ { \mathrm { T L I } }$ difers in substance from the ARAP regularizer of SC-GS, and not merely in name. First, it enforces local isometry, namely the preservation of inter-point distances, rather than fitting a per-point rotation by SVD and penalizing the residual, and it requires no rotation estimation at all. Second, it compares two deformed timestamps $( t _ { 1 }$ and $t _ { 2 } )$ against each other, rather than comparing a deformed configuration against the static canonical one. The result is a rotation-free term that constrains how the geometry evolves between adjacent instants, which is exactly the property needed for stable in-between interpolation.

## 3.5. Training Objective

Photometric loss. At each RGB timestamp the rendering is supervised with the standard photometric loss, a combination of an $\ell _ { 1 }$ term and a D-SSIM term.

Event loss. We supervise the geometry and motion in the gaps with the EGM of Eq. (4). We render the scene at two dense sub-step times $t _ { a } ~ < ~ t _ { b } .$ , convert each rendering to log-luminance $\hat { L } = \log ( \hat { \cal J } + \epsilon )$ , and match the rendered diference to the accumulated events,

$$
\mathcal { L } _ { \mathrm { e v e n t } } = \Big \| \left( \hat { L } ( t _ { b } ) - \hat { L } ( t _ { a } ) \right) - C \sum _ { e \in [ t _ { a } , t _ { b } ] } p _ { e } \Big \| _ { 1 } .\tag{10}
$$

Since a single brightness-change constraint cannot disentangle geometry from chromaticity, we detach the color gradient when forming $\hat { L } ,$ so that $\mathcal { L } _ { \mathrm { e v e n t } }$ updates only geometry and motion and does not contaminate the Gaussian colors, an issue we observed when the events were allowed to back-propagate into color.

Full objective. Together with the two regularizers, the full objective is

$$
\begin{array} { r } { \mathcal { L } = \left( 1 - \lambda _ { d } \right) \mathcal { L } _ { 1 } + \lambda _ { d } \mathcal { L } _ { \mathrm { { D } \mathrm { - } S S I M } } + \lambda _ { e } \mathcal { L } _ { \mathrm { e v e n t } } + \lambda _ { t } \mathcal { L } _ { \mathrm { T L I } } + \lambda _ { s } \mathcal { L } _ { \mathrm { s m o o t h } } , } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { s m o o t h } }$ is the KNN Laplacian smoothness on the event-driven transition term $g _ { \phi }$ that keeps neighboring control points coherent, and $\lambda _ { d } , \lambda _ { e } , \lambda _ { t } , \lambda _ { s }$ balance the terms.

## 4. Experiments

## 4.1. Datasets

Synthetic dataset. We construct a synthetic event-based monocular dataset in Blender, consisting of four sequences, Jumpingjack, Kick, Mutant and $L e g o { \mathrm { . } }$ , which together cover articulated full-frame motion, localized limb motion and rigid mechanical motion. For each sequence we simulate a continuous camera trajectory around a deformable object and render a high-frame-rate RGB sequence with motion blur enabled at a shutter setting of 0 5 (180<sup>◦</sup> shutter angle), so that the exposure of fast motion is realistic. All sequences are rendered at $3 4 6 \times 2 6 0$ to match the resolution of a DAVIS346 event camera, and the rendered sequence is fed to ESIM [43] with a contrast threshold of 0 2 to obtain the event stream. To emulate extreme-low-frame-rate capture we retain only the frames sampled at 10 fps or 5 fps for training, which amounts to roughly 20 and 10 input frames per sequence, and the discarded frames serve as ground truth at the unobserved intermediate timestamps.

Real-world dataset. For real captures we use the real-world dataset released by E-D3DGS [17], which provides four dynamic scenes, Excavator, Jeep, Flowers and Eagle, recorded with a real event camera together with the RGB frames. We downsample each RGB stream to 5 fps to match our extreme-low-frame-rate setting and hold out the remaining frames as ground truth at the intermediate timestamps.

## 4.2. Experimental Setup

Baselines. We compare EdMCGS against two RGB-based deformable baselines, Deformable 3DGS (D-3DGS) [10] and SC-GS [21]. Since RGB-based methods cannot consume events, for a fair comparison we reconstruct intensity frames from the events with E2VID [44] and feed D-3DGS as additional input, which is reported as E2VID + D-3DGS. We additionally compare against Event-boosted Deformable 3DGS (E-D3DGS) [17]. To the best of our knowledge it is the only event-based deformable Gaussian method in this setting whose implementation is publicly available, as DEGS [16], Ev4DGS [18] and E-4DGS [19] have released neither code nor data at the time of writing, so a direct comparison with them is not possible. Substituting the methods that cannot be reproduced by RGB baselines fed with E2VID reconstructions is common practice in this literature [19].

We further include an internal baseline, denoted Ours w/o Markov, which disables the event-driven transition $g _ { \phi }$ so that the events act only as supervision through the event loss. This variant follows the events-as-supervision paradigm of prior work and lets us isolate the gain brought by the event-driven Markov transition that is the core of EdMCGS.

Metrics. We report the three standard image quality metrics, PSNR, SSIM, and LPIPS (VGG). E-D3DGS [17] reports LPIPS with an AlexNet backbone by default, which we switch to VGG so that all methods are evaluated under the same metric.

Implementation details. All experiments are conducted on a single GeForce RTX 4070 Ti Super with 16 GB of memory. Every method, ours and the baselines alike, is trained for 30k iterations on the synthetic datasets and 40k iterations on the real datasets. We use M = 512 control points and bind each Gaussian to its K = 3 nearest control points, while the temporal local isometry graph connects each control point to its $K _ { \mathrm { c o n n } } ~ =$ 10 neighbors. The event map is encoded by a shallow three-layer CNN into a 32- channel feature map at a quarter of the input resolution, without any global pooling and with a receptive field of about 23 pixels, so that each control point reads only its local neighborhood. The decoder $g _ { \phi }$ is a small two-layer MLP whose last layer is zero-initialized; the exact layer specifications are given in our released code. The loss weights are $\lambda _ { d } = 0 . 2 , \lambda _ { e } = 0 . 0 5 , \lambda _ { t } = 1 0 ^ { - 5 }$ , and $\lambda _ { s } = 1 0 ^ { - 2 }$

![](images/24d75a5a0b767e48f8da94258aca3b0a42506a71864e2017884d7d7ee796aeac.jpg)

Figure 2: Qualitative results on our synthesis dataset.  
![](images/42d405db6a876c1d3916fc21813ac7dcdafe9d2eeb5382fdaeb03648d1220a18.jpg)  
Figure 3: Qualitative results on real-world dataset.

## 4.3. Comparison

Figures 2 and 3 show the same pattern in image space. The RGB-based methods either tear the moving parts apart or smear them into an artifact that overlays the positions of the two neighbouring observed frames, while E-D3DGS suppresses the tearing but leaves the fast-moving regions blurred, consistent with an approach whose stable evidence lies in the static background. Compared to previous methods, our method can reconstruct better details.

Quantitative results on our synthesis real-world datasets are presented in Table 1 and Table 2, the per-scene breakdown behind these averages is deferred to Appendix A. On our synthesis datasets, our method reaches 31 56 dB at 10 fps and 29 37 dB at 5 fps, exceeding all the baselines. Against E-D3DGS the relative gain is larger on LPIPS than on PSNR. This distinction matters here, because a method that cannot recover the motion inside an unobserved interval falls back on a temporally averaged, blurred rendering of the moving part, a failure that PSNR tolerates and LPIPS does not. On the four real scenes EdMCGS again ranks first on all three metrics. The absolute margins are smaller than on the synthetic data because these captures place a moderately sized moving object in an otherwise static scene, so much of every test image is explained by the static reconstruction and is never endangered by the low frame rate.

Table 1: Quantitative comparison on our synthesis dataset, averaged over the four scenes. Per-scene results are given in Appendix A (Table A.1). We highlight the best and second-best values for each column.
<table><tr><td rowspan="2">Method</td><td colspan="3">Average (5 fps)</td><td colspan="3">Average (10 fps)</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>D-3DGS [10]</td><td>22.95</td><td>.8975</td><td>.1066</td><td>26.60</td><td>.9302</td><td>.0739</td></tr><tr><td>E2VID + D-3DGS</td><td>21.47</td><td>.8923</td><td>.1637</td><td>21.67</td><td>.8925</td><td>.1633</td></tr><tr><td>SC-GS [21]</td><td>22.99</td><td>.9052</td><td>.1254</td><td>26.10</td><td>.9204</td><td>.0958</td></tr><tr><td>E-D3DGS [17]</td><td>26.83</td><td>.9298</td><td>.0703</td><td>28.60</td><td>.9546</td><td>.0662</td></tr><tr><td>Ours w/o Markov</td><td>25.44</td><td>.9280</td><td>.0877</td><td>28.11</td><td>.9515</td><td>.0624</td></tr><tr><td>Ours</td><td>29.37</td><td>.9495</td><td>.0551</td><td>31.56</td><td>.9668</td><td>.0426</td></tr></table>

The advantage grows as the input becomes sparser, which is the regime the method is built for. Halving the frame rate doubles the interval that must be filled without observations while leaving the event evidence inside it unchanged, so a method that draws its in-between motion from the events should degrade more slowly than one that interpolates over time.

Table 2: Quantitative comparison on the real-world dataset of E-D3DGS [17] at 5 fps, averaged over the four scenes. Per-scene results are given in Appendix A (Table A.2). We highlight the best and second-best values for each column.
<table><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>D-3DGS [10]</td><td>29.07</td><td>.8829</td><td>.2780</td></tr><tr><td>E2VID + D-3DGS</td><td>21.96</td><td>.8165</td><td>.5266</td></tr><tr><td>E-D3DGS [17]</td><td>29.75</td><td>.8920</td><td>.2583</td></tr><tr><td>Ours w/o Markov</td><td>30.22</td><td>.8940</td><td>.2575</td></tr><tr><td>Ours</td><td>31.13</td><td>.9015</td><td>.2398</td></tr></table>

E2VID + D-3DGS falls below our method in every setting, and its scores are almost invariant to the RGB frame rate, which shows that in extreme-low-frame-rate settings, the reconstructed frames dominate the optimization. Event-to-video reconstruction recovers relative brightness up to a drifting ofset and carries no chrominance, so it is photometrically inconsistent with the genuine frames and corrupts the appearance those frames had already pinned down.

E-D3DGS instead segments the scene into a dynamic and a static part and uses the static background as a photometric anchor, which works only while the moving object stays small in the image. The per-scene results in Table A.1 follow that boundary, as our largest margin over E-D3DGS occurs at both frame rates on Jumpingjack, the only sequence in which an articulated body fills the frame and leaves almost no static background to anchor on. EdMCGS needs no decomposition, no motion mask and no assumption about how much of the frame moves, and it ranks first on every scene at both frame rates.

EdMCGS is at the same time the lightest of the event-based methods. As Tables 3 and 4 show, it trains faster than E-D3DGS on both datasets, renders faster, and represents the scene with several times fewer Gaussians, while still reconstructing at higher quality. Relative to the D-3DGS backbone the event branch does cost training time and rendering speed, yet it leaves the number of Gaussians essentially unchanged, so the improvement comes from the motion model rather than from additional representational capacity. Event evidence is therefore better exploited by a light deformation model that reads it at inference than by a heavy scene representation that only absorbs it during training.

Table 3: Eficiency comparison on our synthesis dataset.
<table><tr><td>Method</td><td>Time (mm:ss)↓</td><td>FPS↑</td><td>#Gaussians</td><td>PSNR↑</td></tr><tr><td>D-3DGS [10]</td><td>5:49</td><td>566</td><td>7373</td><td>26.60</td></tr><tr><td>E2VID + D-3DGS</td><td>8:00</td><td>448</td><td>12097</td><td>21.67</td></tr><tr><td>SC-GS [21]</td><td>17:34</td><td>249</td><td>37699</td><td>26.10</td></tr><tr><td>E-D3DGS [17]</td><td>15:04</td><td>277</td><td>46044</td><td>28.60</td></tr><tr><td>EdMCGS (Ours)</td><td>13:34</td><td>309</td><td>8119</td><td>31.56</td></tr></table>

Table 4: Eficiency comparison on real-world dataset (5 fps).
<table><tr><td>Method</td><td>Time (mm:ss)↓</td><td>FPS↑</td><td>#Gaussians</td><td>PSNR↑</td></tr><tr><td>D-3DGS [10]</td><td>14:02</td><td>296</td><td>18971</td><td>29.07</td></tr><tr><td>E2VID + D-3DGS</td><td>18:38</td><td>221</td><td>23790</td><td>21.96</td></tr><tr><td>E-D3DGS [17]</td><td>198:58</td><td>176</td><td>183710</td><td>29.75</td></tr><tr><td>EdMCGS (Ours)</td><td>16:42</td><td>218</td><td>18051</td><td>31.13</td></tr></table>

## 4.4. Ablation Study

Table 5 removes each of the three main components in turn. Discarding the event stream entirely, which reduces the model to a control-point deformation field driven by time alone, gives 24 05 dB on the synthetic data and 30 04 dB on the real data, 5 32 dB and 1<sub>.</sub>09 dB below the full model, and this diference measures everything the events are worth to our method.

The internal baseline Ours w/o Markov keeps the control-point deformation field and the event loss but removes the event-driven transition, leaving the events as pure training-time supervision in the manner of prior work. It collapses to 21 67 dB on Jumpingjack at 5 fps, 10 14 dB behind the full model, yet trails by only 0 24 dB on Lego at 10 fps. When little happens between two frames, conditioning the deformation on time alone is already close to suficient, and it is when a great deal happens, which extreme-low-frame-rate capture guarantees, that the events must drive the transition at inference.

Table 5: Ablation of the main components of EdMCGS, with per-metric averages on the synthetic dataset (5 fps) and the real-world dataset. Full denotes the complete model.
<table><tr><td rowspan="2">Variant</td><td colspan="3">Synthetic (5 fps)</td><td colspan="3">Real-world</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>w/o event stream</td><td>24.05</td><td>.8953</td><td>.1015</td><td>30.04</td><td>.8933</td><td>.2533</td></tr><tr><td>w/o Markov chain</td><td>25.44</td><td>.9280</td><td>.0877</td><td>30.22</td><td>.8940</td><td>.2575</td></tr><tr><td>w/o local sampling</td><td>24.61</td><td>.9033</td><td>.0998</td><td>29.87</td><td>.8899</td><td>.2645</td></tr><tr><td>w/o TLI</td><td>24.98</td><td>.9220</td><td>.0921</td><td>30.56</td><td>.8985</td><td>.2489</td></tr><tr><td>Full</td><td>29.37</td><td>.9495</td><td>.0551</td><td>31.13</td><td>.9015</td><td>.2398</td></tr></table>

Replacing the local event sampling of Eq. (7) by a global descriptor, obtained by average-pooling the encoded event map into a single vector shared by all control points, is worse still. The reason is that an accumulated event map mixes together the events fired by the moving object, by the static background under camera motion, and by sensor noise, and pooling collapses all of them into one descriptor that is then applied identically to every control point. Points on the static parts of the scene are pushed by the object’s events and points on the object are dragged by the background’s, which is precisely the confusion that reading the map at each point’s own image projection avoids.

Removing the temporal local isometry term is comparably damaging. The eventdriven transition produces an independent displacement for each control point, and the accumulated event map is noisy and unevenly distributed over the object, so nothing in the transition alone prevents neighbouring points from moving inconsistently.

Preserving inter-point distances across nearby deformed timestamps is what keeps the propagated motion locally rigid, and the term matters most on the synthetic sequences, where the transition is doing the most work. The two components are therefore complementary rather than additive, since the transition supplies the motion and the isometry constraint makes it usable, and the full model needs both to reach its quality.

## 5. Conclusion

We present EdMCGS, a lightweight and end-to-end framework for reconstructing dynamic 3D scenes from extreme-low-frame-rate RGB images together with an event stream. Unlike previous methods that rely on temporal interpolation, the event-driven Markov chain remains active during inference, enabling intermediate deformations to be predicted directly from the event stream. Experiments on both synthetic and realworld datasets demonstrate that EdMCGS consistently outperforms existing baselines. We also release our code and dataset to facilitate future research on event-based dynamic scene reconstruction.

A limitation of EdMCGS is that our pipeline assumes known camera poses, an assumption that it shares with the deformable Gaussian splatting methods we compare against [10, 21, 17] and that DEGS [16] lifts only by adding a separate pose network to its pipeline. Recent work reconstructs static scenes from events without poses [41, 15], and carrying that line over to a deforming scene, so that the camera poses and the dynamic geometry are recovered jointly, is a natural next step. Further directions include exploring higher-order event-driven temporal models and evaluating the method on larger real-world event datasets with more diverse motions.

## CRediT authorship contribution statement

Yuzhong Wang: Conceptualization, Methodology, Software, Validation, Investigation, Data curation, Writing – original draft, Visualization. Wenmin Wang: Supervision, Writing – review & editing, Project administration. Xinxing Yu: Supervision, Writing – review & editing.

## Declaration of Generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used Claude (Anthropic) in order to improve the language and readability of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Acknowledgements

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Appendix A. Per-scene Results

For completeness, we report the per-scene breakdown behind the averages in Tables 1 and 2. Table A.1 gives the per-scene results on the synthetic dataset, and Table A.2 gives the per-scene results on the real-world dataset. We highlight the best and second-best values for each column.

Table A.1: Per-scene quantitative comparison on our synthesis dataset.
<table><tr><td></td><td colspan="3">Jumpingjack (5 fps)</td><td colspan="3">Kick (5 fps)</td><td colspan="3">Mutant (5 fps)</td><td colspan="3">Lego (5 fps)</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>D-3DGS [10]</td><td>21.44</td><td>.8850</td><td>.1309</td><td>21.53</td><td>.8955</td><td>.1170</td><td>25.86</td><td>.8950</td><td>.1152</td><td>22.96</td><td>.9146</td><td>.0632</td></tr><tr><td>E2VID + D-3DGS</td><td>19.12</td><td>.8853</td><td>.1717</td><td>22.14</td><td>.9110</td><td>.1681</td><td>24.67</td><td>.8975</td><td>.1776</td><td>19.94</td><td>.8752</td><td>.1374</td></tr><tr><td>SC-GS [21]</td><td>22.62</td><td>.8944</td><td>.1622</td><td>20.27</td><td>.8919</td><td>.1033</td><td>25.19</td><td>.8908</td><td>.1752</td><td>23.87</td><td>.9437</td><td>.0609</td></tr><tr><td>E-D3DGS [17]</td><td>26.69</td><td>.9161</td><td>.0763</td><td>26.11</td><td>.9276</td><td>.0732</td><td>26.29</td><td>.9190</td><td>.0719</td><td>28.23</td><td>.9564</td><td>.0599</td></tr><tr><td>Ours w/o Markov</td><td>21.67</td><td>.9030</td><td>.1253</td><td>25.14</td><td>.9295</td><td>.0826</td><td>26.49</td><td>.9216</td><td>.0916</td><td>28.44</td><td>.9580</td><td>.0511</td></tr><tr><td>Ours</td><td>31.81</td><td>.9618</td><td>.0422</td><td>28.08</td><td>.9423</td><td>.0623</td><td>28.44</td><td>.9332</td><td>.0716</td><td>29.13</td><td>.9608</td><td>.0442</td></tr><tr><td></td><td colspan="3">Jumpingjack (10 fps)</td><td colspan="3">Kick (10 fps)</td><td colspan="3">Mutant (10 fps)</td><td colspan="3">Lego (10 fps)</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>D-3DGS [10]</td><td>24.83</td><td>.9211</td><td>.0801</td><td>24.51</td><td>.9119</td><td>.0982</td><td>25.94</td><td>.9116</td><td>.0940</td><td>31.11</td><td>.9761</td><td>.0232</td></tr><tr><td>E2VID + D-3DGS</td><td>19.12</td><td>.8850</td><td>.1715</td><td>22.14</td><td>.9111</td><td>.1683</td><td>24.67</td><td>.8974</td><td>.1775</td><td>20.76</td><td>.8766</td><td>.1358</td></tr><tr><td>SC-GS [21]</td><td>24.65</td><td>.9166</td><td>.1343</td><td>23.59</td><td>.8723</td><td>.1349</td><td>25.69</td><td>.9187</td><td>.0898</td><td>30.46</td><td>.9740</td><td>.0241</td></tr><tr><td>E-D3DGS [17]</td><td>28.42</td><td>.9571</td><td>.0738</td><td>28.21</td><td>.9555</td><td>.0619</td><td>27.45</td><td>.9376</td><td>.0820</td><td>30.30</td><td>.9681</td><td>.0470</td></tr><tr><td>Ours w/o Markov</td><td>24.36</td><td>.9443</td><td>.0829</td><td>26.08</td><td>.9363</td><td>.0748</td><td>27.96</td><td>.9429</td><td>.0709</td><td>34.03</td><td>.9826</td><td>.0208</td></tr><tr><td>Ours</td><td>32.91</td><td>.9707</td><td>.0363</td><td>29.70</td><td>.9622</td><td>.0538</td><td>29.35</td><td>.9499</td><td>.0628</td><td>34.27</td><td>.9842</td><td>.0175</td></tr></table>

Table A.2: Per-scene quantitative comparison on the real-world dataset of E-D3DGS [17] at 5 fps. The average over the four scenes is reported in Table 2.
<table><tr><td></td><td colspan="3">Excavator (5 fps)</td><td colspan="3">Jeep (5 fps)</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>D-3DGS [10]</td><td>28.89</td><td>.8953</td><td>.3045</td><td>28.98</td><td>.8665</td><td>.2754</td></tr><tr><td>E2VID + D-3DGS</td><td>20.42</td><td>.8239</td><td>.5669</td><td>21.34</td><td>.8085</td><td>.5116</td></tr><tr><td>E-D3DGS [17]</td><td>30.88</td><td>.9173</td><td>.2528</td><td>29.16</td><td>.8747</td><td>.2497</td></tr><tr><td>Ours w/o Markov</td><td>30.63</td><td>.9107</td><td>.2667</td><td>30.03</td><td>.8780</td><td>.2490</td></tr><tr><td>Ours</td><td>32.17</td><td>.9270</td><td>.2350</td><td>30.49</td><td>.8813</td><td>.2402</td></tr><tr><td></td><td colspan="3">Flowers (5 fps)</td><td colspan="3">Eagle (5 fps)</td></tr><tr><td>Method</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td></tr><tr><td>D-3DGS [10]</td><td>27.40</td><td>.8708</td><td>.2681</td><td>31.01</td><td>.8989</td><td>.2639</td></tr><tr><td>E2VID + D-3DGS</td><td>20.95</td><td>.8204</td><td>.5145</td><td>25.15</td><td>.8133</td><td>.5134</td></tr><tr><td>E-D3DGS [17]</td><td>27.85</td><td>.8767</td><td>.2673</td><td>31.12</td><td>.8992</td><td>.2633</td></tr><tr><td>Ours w/o Markov</td><td>28.57</td><td>.8852</td><td>.2431</td><td>31.66</td><td>.9021</td><td>.2711</td></tr><tr><td>Ours</td><td>29.51</td><td>.8917</td><td>.2237</td><td>32.36</td><td>.9060</td><td>.2604</td></tr></table>

## References

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, R. Ng, Nerf: Representing scenes as neural radiance fields for view synthesis, Communications of the ACM 65 (1) (2021) 99–106.

[2] A. Pumarola, E. Corona, G. Pons-Moll, F. Moreno-Noguer, D-nerf: Neural radiance fields for dynamic scenes, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 10318–10327.

[3] Z. Yan, C. Li, G. H. Lee, Nerf-ds: Neural radiance fields for dynamic specular objects, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 8285–8295.

[4] K. Park, U. Sinha, P. Hedman, J. T. Barron, S. Bouaziz, D. B. Goldman, R. Martin-Brualla, S. M. Seitz, Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields, ACM Trans. Graph. 40 (6) (dec 2021).

[5] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, 3d gaussian splatting for realtime radiance field rendering., ACM Trans. Graph. 42 (4) (2023) 139–1.

[6] A. Hanson, A. Tu, G. Lin, V. Singla, M. Zwicker, T. Goldstein, Speedy-splat: Fast 3d gaussian splatting with sparse pixels and sparse primitives, in: Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 21537– 21546.

[7] S. Ren, T. Wen, Y. Fang, B. Lu, Fastgs: Training 3d gaussian splatting in 100 seconds, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 26094–26103.

[8] Q. Shen, X. Yang, X. Wang, Flashsplat: 2d to 3d gaussian splatting segmentation solved optimally, in: European Conference on Computer Vision, Springer, 2024, pp. 456–472.

[9] M. Ye, M. Danelljan, F. Yu, L. Ke, Gaussian grouping: Segment and edit anything in 3d scenes, in: European conference on computer vision, Springer, 2024, pp. 162–179.

[10] Z. Yang, X. Gao, W. Zhou, S. Jiao, Y. Zhang, X. Jin, Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 20331–20341.

[11] G. Wu, T. Yi, J. Fang, L. Xie, X. Zhang, W. Wei, W. Liu, Q. Tian, X. Wang, 4d gaussian splatting for real-time dynamic scene rendering, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 20310–20320.

[12] G. Gallego, T. Delbrück, G. Orchard, C. Bartolozzi, B. Taba, A. Censi, S. Leutenegger, A. J. Davison, J. Conradt, K. Daniilidis, et al., Event-based vision: A survey, IEEE transactions on pattern analysis and machine intelligence 44 (1) (2020) 154–180.

[13] S. Klenk, L. Koestler, D. Scaramuzza, D. Cremers, E-nerf: Neural radiance fields from a moving event camera, IEEE Robotics and Automation Letters 8 (3) (2023) 1587–1594.

[14] W. Yu, C. Feng, J. Li, J. Tang, J. Yang, Z. Tang, M. Cao, X. Jia, Y. Yang, L. Yuan, et al., Evagaussians: Event stream assisted gaussian splatting from blurry images, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 24780–24790.

[15] Y. Kim, C. Sung, D. Hong, H. Myung, E2egs: Event-to-edge gaussian splatting for pose-free 3d reconstruction, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 4922–4931.

[16] J. He, J. Wang, J. Li, M. Sun, Q. Zhang, J. Cao, Z. Zhang, Y. Gu, J. Sun, R. Xu, Degs: Deformable event-based 3d gaussian splatting from rgb and event stream, IEEE Transactions on Visualization and Computer Graphics (2025).

[17] W. Xu, W. Weng, Y. Zhang, R. Xu, Z. Xiong, Event-boosted deformable 3d gaussians for dynamic scene reconstruction, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 28334–28343.

[18] T. Nakabayashi, N. Kairanda, H. Saito, V. Golyanik, Ev4dgs: Novel-view rendering of non-rigid objects from monocular event streams, in: Proceedings of the British Machine Vision Conference (BMVC), 2025.

[19] C. Feng, Z. Tang, W. Yu, Y. Pang, Y. Zhao, J. Zhao, L. Yuan, Y. Tian, E-4dgs: High-fidelity dynamic reconstruction from the multi-view event cameras, in: Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 7356–7365.

[20] R. W. Sumner, J. Schmid, M. Pauly, Embedded deformation for shape manipulation, in: ACM SIGGRAPH 2007 Papers, Association for Computing Machinery, 2007, pp. 80–es.

[21] Y.-H. Huang, Y.-T. Sun, Z. Yang, X. Lyu, Y.-P. Cao, X. Qi, Sc-gs: Sparsecontrolled gaussian splatting for editable dynamic scenes, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024, pp. 4220–4230.

[22] A. Cao, J. Johnson, Hexplane: A fast representation for dynamic scenes, CVPR (2023).

[23] S. Fridovich-Keil, G. Meanti, F. R. Warburg, B. Recht, A. Kanazawa, K-planes: Explicit radiance fields in space, time, and appearance, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 12479–12488.

[24] J. Fang, T. Yi, X. Wang, L. Xie, X. Zhang, W. Liu, M. Nießner, Q. Tian, Fast dynamic radiance fields with time-aware neural voxels, in: SIGGRAPH Asia 2022 Conference Papers, 2022.

[25] K. Katsumata, D. M. Vo, H. Nakayama, A compact dynamic 3d gaussian representation for real-time dynamic view synthesis, in: European Conference on Computer Vision, Springer, 2024, pp. 394–412.

[26] J. Luiten, G. Kopanas, B. Leibe, D. Ramanan, Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis, in: 2024 International Conference on 3D Vision (3DV), IEEE, 2024, pp. 800–809.

[27] S. Kwak, J. Kim, J. Y. Jeong, W.-S. Cheong, J. Oh, M. Kim, Modec-gs: Globalto-local motion decomposition and temporal interval adjustment for compact dynamic 3d gaussian splatting, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 11338–11348.

[28] D. Wan, R. Lu, G. Zeng, Superpoint gaussian splatting for real-time high-fidelity dynamic scene reconstruction, in: Proceedings of the 41st International Conference on Machine Learning, 2024, pp. 49957–49972.

[29] Y. Wang, W. Wang, S. Zhang, X. Yu, Z. Chen, Mcgs: Markov chain gaussian splatting for dynamic scenes reconstruction, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, 2026, pp. 10341–10348.

[30] R. Wu, Z. Zhang, M. Chen, Z. Yan, W. Zuo, Deblur4dgs: 4d gaussian splatting from blurry monocular video, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, 2026, pp. 10727–10735.

[31] Y. Lu, Y. Zhou, D. Liu, T. Liang, Y. Yin, Bard-gs: Blur-aware reconstruction of dynamic scenes via gaussian splatting, in: Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 16532–16542.

[32] M.-Q. V. Bui, J. Park, J. L. Gonzalez, J. Moon, J. Oh, M. Kim, Mobgs: Motion deblurring dynamic 3d gaussian splatting for blurry monocular video, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40, 2026, pp. 2480–2489.

[33] Y. Liu, Y. Deng, B. Xie, H. Liu, Z. Yang, Y. Li, Neuromorphic event-based recognition boosted by motion-aware learning, Neurocomputing 630 (2025) 129678. doi:https://doi.org/10.1016/j.neucom.2025.129678. URL https://www.sciencedirect.com/science/article/pii/ S0925231225003509

[34] Z. Liu, J. Wu, G. Shi, W. Yang, J. Ma, An event-based motion scene feature extraction framework, Pattern Recognition 161 (2025) 111320. doi:https://doi.org/10.1016/j.patcog.2024.111320. URL https://www.sciencedirect.com/science/article/pii/ S0031320324010719

[35] Y. Lin, J. Zhang, S. Yu, J. Xiao, J. Lu, Ffevent: Fast fourier-based knowledge transfer for event cameras, Expert Systems with Applications 299 (2026) 130055.

doi:https://doi.org/10.1016/j.eswa.2025.130055. URL https://www.sciencedirect.com/science/article/pii/ S0957417425036711

[36] W. F. Low, G. H. Lee, Robust e-nerf: Nerf from sparse & noisy events under non-uniform motion, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 18335–18346.

[37] C. Feng, W. Yu, X. Cheng, Z. Tang, J. Zhang, L. Yuan, Y. Tian, Ae-nerf: Augmenting event-based neural radiance fields for non-ideal conditions and larger scenes, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39, 2025, pp. 2924–2932.

[38] W. F. Low, G. H. Lee, Deblur e-nerf: Nerf from motion-blurred events under highspeed or low-light conditions, in: European Conference on Computer Vision, Springer, 2024, pp. 192–209.

[39] Y. Qi, L. Zhu, Y. Zhang, J. Li, E2nerf: Event enhanced neural radiance fields from blurry images, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 13254–13264.

[40] J. Wu, S. Zhu, C. Wang, E. Y. Lam, Ev-gs: Event-based gaussian splatting for eficient and accurate radiance field rendering, in: 2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP), IEEE, 2024, pp. 1–6.

[41] J. Huang, C. Dong, X. Chen, P. Liu, Inceventgs: Pose-free gaussian splatting from a single event camera, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 26933–26942.

[42] Y. Qi, L. Zhu, Y. Zhang, J. Li, Deformable neural radiance fields using rgb and event cameras, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[43] H. Rebecq, D. Gehrig, D. Scaramuzza, Esim: an open event camera simulator, in: Conference on robot learning, PMLR, 2018, pp. 969–982.

[44] H. Rebecq, R. Ranftl, V. Koltun, D. Scaramuzza, Events-to-video: Bringing modern computer vision to event cameras, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 3857–3866.