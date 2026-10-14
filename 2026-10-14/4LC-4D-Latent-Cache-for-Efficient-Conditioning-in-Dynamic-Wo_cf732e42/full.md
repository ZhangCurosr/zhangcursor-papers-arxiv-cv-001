# 4LC: 4D Latent Cache for Efficient Conditioning in Dynamic World Modeling

Jianhui Zhang<sup>1,2</sup> Zuyao Lin<sup>3,2</sup> Pengxin Xu<sup>4,2</sup>

Jingdong Wang<sup>2,B</sup> Xingyu Chen<sup>2,B</sup>

<sup>1</sup>Shanghai Jiaotong University <sup>2</sup> Zhongguancun Academy <sup>3</sup>Institute of Automation, Chinese Academy of Science <sup>4</sup>Harbin Institute of Technology

https://zgca-hmi-lab.github.io/4LC

https://github.com/ZGCA-HMI-Lab/4LC

https://huggingface.co/jianhuizhang/4LC-Video

https://huggingface.co/spaces/jianhuizhang/4LC-Demo

![](images/1461b3de332d9287048f4426895ed0d3fdd372fbdbe5f86c5e8486a761f9c3da.jpg)

![](images/b1d22eab359ae2cf7718495b692efb925ae972a26d990761cc40a4b23a993cbc.jpg)  
(c) Efficiency of Condition Acquisition

![](images/865f77577f1eb60decd301b5f89ace0cf66c16586049bfb37de06825771d89be.jpg)

![](images/4ff0687a68054ec82faca95899d75d8a35295cae173a776a7607222f5353a7af.jpg)  
(b) Non-Cache vs. Cache-Level Video Generation  
Figure 1. Overview of 4LC. (a) In 4LC, static and dynamic scene content is retained in multiple visual representations for reuse across time and viewpoint requests. (b) Non-cache pipelines render and re-encode a reconstructed scene for each generation request; 4LC encodes source observations once and reads out the cache for subsequent requests. (c) With the same Wan2.1 VAE, 4LC is superior to 4DGS and RGB point clouds in condition-acquisition latency and peak GPU memory. (d) 4LC-Video shows superior performance across five video generation tasks.

## Abstract

For video generation in dynamic world modeling, observed static and dynamic content should be efficiently reused to provide conditions for repeated generation at different times and viewpoints. We introduce 4LC, a 4D latent cache with composite visual representations and a flexible spatiotemporal access interface. As a latent cache, 4LC directly reuses visual representations of static and dynamic content to provide generation conditions, avoiding per-request RGB rendering and re-encoding. As a composite cache, 4LC binds visual representations from multiple visual encoders (e.g., VAE, DINO, V-JEPA, VGGT) to shared geometry and motion, allowing the composite representations to describe the dynamic world with rich information. Besides, we design a flexible spatiotemporal interface, which directly reads the cache at independently specified times and viewpoints and combines readouts across observations, providing a common conditioning format for diverse downstream tasks. Building on 4LC, we develop 4LC-Video to address multiple video generation tasks through different configurations of source observations, spatiotemporal requests, and historical reuse, including camera control, temporal control, novel-view synthesis, scene revisits, and long-term generation. Our experiments show more efficient condition acquisition, effective spatiotempora controllability, and scene consistency across tasks using a single 4LC-Video model without task-specific fine-tuning.

## 1 Introduction

In dynamic world modeling, observed scene information provides conditions for video generation at specified times and viewpoints. These conditions should convey persistent static structure together with dynamic content at the requested scene time. Geometry and motion help relate the observed static and dynamic content across viewpoints and scene times [1, 2], supporting consistency in the generated video [3]. Because the same observations serve different requests and subsequent generation segments, reusing previously extracted information can reduce the recurring cost of condition acquisition.

Existing conditioning pipelines retain scene information in explicit 3D reconstructions or spatial memories and render it into target views [4, 5, 6]. Static and dynamic content can be maintained in separate components [7, 8] or represented together through shared 4D geometry, as in 4D Gaussian Splatting (4DGS) [9]. This shared geometry allows both to be rendered at requested times and viewpoints [10]. However, these geometric conditioning pipelines still require RGB rendering and re-encoding for each new generation request (Fig. 1b). This repeated processing can consume a substantial share of total generation time. Latent reuse already reduces repeated computation in video world models, both in history processing [11] and in denoising [12]. For conditioning, Mirage caches latent features with scene geometry and warps them directly to target views [13]. These features lie in the latent space of a specific generator, and dynamic content is excluded from the cache. These limitations motivate us to investigate a cache mechanism that extends latent reuse to dynamic content, enabling efficien conditioning in dynamic world modeling.

To this end, we introduce 4LC, a 4D latent cache that jointly stores visual representations of static and dynamic content extracted by multiple visual encoders (Fig. 1a). Motion determines the 3D positions associated with cached representations a the requested scene time, with static content held at its reference position. These visual representations are bound to the same geometry and motion while retaining their own formats, allowing their complementary information to be used together for conditioning. 4LC provides a spatiotemporal interface that directly reads these representations at independently specified times and viewpoints, combining readouts across observations to supplement uncovered regions. Once observations are cached, subsequent requests read the stored representations directly instead of rendering and re-encoding them (Fig. 1b), reducing condition-acquisition latency and peak GPU memory (Fig. 1c).

We build 4LC-Video to use these cache readouts as conditions for a shared video generator. Visual representations of generated observations are appended to the cache as new entries for subsequent segments. The same model can be configured for camera control, temporal control, novel-view synthesis, scene revisits, and long-term generation by specifying source observations, spatiotemporal requests, and historical reuse. Across these tasks, a single checkpoint achieves effective spatiotemporal controllability and scene consistency without task-specific fine-tuning (Fig. 1d).

Our main contributions are as follows:

• We introduce 4LC, a 4D latent cache with rich visual representations from composite encoders. Its spatiotemporal interface directly reads the cache at independently requested times and viewpoints, without per-request RGB rendering and re-encoding, thereby forming an efficient dynamic cache mechanism for 4D scenes.

• We develop 4LC-Video, which includes a spatiotemporal condition adapter to align the rich visual representations of 4LC with a pretrained video diffusion model for downstream video generation.

• Without task-specific fine-tuning, a single 4LC-Video model achieves state-of-the-art control accuracy and scene consistency across five generation tasks spanning spatiotemporal control, view synthesis, and long-term generation. 4LC accelerates condition acquisition by up to 68.5× with 12.6× lower peak GPU memory and reduces its share of end-to-end generation time from 14.9% to 0.25%, paving a new route for efficient conditioning and multi-task video generation.

## 2 Related Work

Dynamic World Modeling. Dynamic world modeling aims at controllable video generation within a persistent scene that evolves over time [14, 15, 16, 17]. BulletTime [18] and SpaceTimePilot [19] control viewpoint and scene time as separate inputs, including frozen and reversed motion. Spatia [5] maintains consistency across segments by updating a point-cloud memory. WorldMem [20] and VMem [21] retrieve past views by field-of-view overlap and surfel indexing, respectively. Interactive world models support long-term generation over successive segments [22, 23, 24, 25, 26]. WorldKV [11] retrieves relevant cached history and GIM-World [27] compresses past observations into compact memory. When scene time advances between visits, dynamic content must also remain coherent [28, 29]. LiveWorld [7] simulates out-of-sight dynamics, and VerseCrafter [8] separates a static point cloud from per-object trajectories. To avoid the complexity of maintaining separate components, 4LC binds representations of static and dynamic content to shared geometry and motion, reading them flexibly a independently specified times and viewpoints.

Reconstruction-based Video Generation. Reconstruction-based video generation guides video models with renderings of reconstructed geometry. Several methods condition generation on point-cloud or 3D-cache renderings to complete unseen or poorly reconstructed regions [4, 30]. For dynamic scenes, 4DGS [9] and feed-forward 4D reconstruction [31, 32, 33, 34] model geometry and motion, and NeoVerse [10] renders them at requested times and viewpoints as conditions. TrajectoryCrafter [35] and Vista4D [36] redirect the camera of monocular videos through point-cloud renderings that preserve source motion. These pipelines re-encode rendered RGB views for every generation request. Feature-space approaches instead condition on features carried by reconstructed geometry [37, 38, 39]. MVSplat360 [40] and Splatent [41] render Gaussian features into diffusion latents for novel-view synthesis. Mirage [13] warps cached latent features to target views. MVSplat360, Splatent, and Mirage avoid target RGB encoding, yet their features lie in a specific generator’s latent space and cover only static content. For efficient conditioning, 4LC reads out cached representations directly without per-request re-encoding, and its multi-encode representations add complementary information.

## 3 Method

The core of 4LC is a cache $\mathcal { C } = \{ \mathcal { C } _ { q } \} _ { q = 0 } ^ { Q - 1 }$ and a spatiotemporal interface equipped with three operations. Each entry $\mathcal { C } _ { q }$ stores the visual representations of its source observations, bound to their 3D geometry and motion, and C organizes these entries by insertion order $q .$ Construction encodes source observations with multiple visual encoders and stores their visua representations in a new cache entry (Fig. 2a). Readout propagates an entry’s geometry to requested times and projects it through independently specified target cameras, reusing stored representations to produce readout features and coverage masks for static and dynamic content (Fig. 2b). Composition uses these masks to combine readouts across selected entries, filling uncovered regions while retaining existing readouts (Fig. 2c). 4LC-Video conditions a pretrained video diffusion model on these readouts (Sec. 3.4). Appendix Sections A.1 and A.2 detail the implementation of 4LC and 4LC-Video, respectively.

![](images/b865e2a3952704087be1635092d427d974701a4d9be6bb68db296fd63f5b053c.jpg)  
Figure 2. Architecture of 4LC. (a) Construction binds visual representations from multiple encoders to shared geometry and motion in one cache entry. (b) Readout propagates the entry to a requested time and projects it through a target camera, yielding readout features and coverage masks. (c) Composition fills uncovered regions with readouts from further entries.

## 3.1 4LC Construction

Construction builds an entry $\mathcal { C } _ { q } = \{ u _ { i } \} _ { i = 0 } ^ { P - 1 }$ of $P$ cache units from $K$ timestamped source images (keyframes) $\displaystyle \{ \mathbf { I } _ { k } \} _ { k = 0 } ^ { K - 1 }$ in shared world coordinates and appends it to C. Each unit is defined as

$$
\begin{array} { r } { u _ { i } = \big ( \mathbf { x } _ { i } , t _ { i } , \mathbf { v } _ { i } ^ { + } , \mathbf { v } _ { i } ^ { - } , b _ { i } , \{ \mathbf { f } _ { i } ^ { ( r ) } \} _ { r } \big ) . } \end{array}
$$

Geometry. Using depth and cameras from a pretrained estimator, we back-project source pixels to reference positions $\mathbf { x } _ { i } \in \mathbb { R } ^ { 3 }$ in shared world coordinates. Reference time $t _ { i }$ is the source timestamp.

Motion. Motion estimates from the same estimator toward adjacent observations give local forward and backward world velocities $\mathbf v _ { i } ^ { + } , \mathbf v _ { i } ^ { - } \in \mathbb R ^ { 3 }$ . The motion mask $b _ { i } \in \{ 0 , 1 \}$ retains the estimated motion when $b _ { i } = 1$ and holds the reference position when $b _ { i } = 0 . \mathrm { ~ A ~ }$ single source image $( K = 1 )$ ) provides no motion estimates between observations, so we set $b _ { i } = 0$

Visual representations. Visual encoders produce visual representations as feature maps at different spatial and temporal resolutions. For each visual representation $r ,$ we associate source frame k with step $\alpha _ { r } ( k )$ of its encoded feature sequence $\mathbf { Z } ^ { ( r ) }$ . For unit $u _ { i } .$ , its source frame $k _ { i } \in \{ 0 , \ldots , K - 1 \}$ and source-pixel coordinates $\mathbf { p } _ { i } \in \mathbb { R } ^ { 2 }$ locate the bound feature vector:

$$
\mathbf { F } _ { k } ^ { ( r ) } = \mathbf { Z } _ { \alpha _ { r } ( k ) } ^ { ( r ) } , \qquad \mathbf { f } _ { i } ^ { ( r ) } = \mathbf { F } _ { k _ { i } } ^ { ( r ) } \left[ \left\lfloor \frac { \mathbf { p } _ { i } } { d _ { r } } \right\rfloor \right] \in \mathbb { R } ^ { D _ { r } } .\tag{1}
$$

Here, $d _ { r }$ is the spatial stride and $D _ { r }$ the feature dimension. Units sharing a source feature cell reuse the same vector while retaining their own geometry and motion.

## 3.2 4LC Readout

The target sequence $\{ ( t _ { j } , \Gamma _ { j } ) \} _ { j = 0 } ^ { N - 1 }$ specifies N output frames, with scene time $t _ { j }$ and camera configuration $\Gamma _ { j }$ (intrinsics and world pose) at index j. For a cache entry $\mathcal { C } _ { q }$ and a pair (t, Γ), the readout features and coverage for representation r are

$$
\left( \mathbf { H } ^ { ( r ) } , \mathbf { M } ^ { ( r ) } \right) = \mathrm { P r o j e c t i o n } \bigl ( \mathrm { P r o p a g a t i o n } ( \mathcal { C } _ { q } , t ) , \Gamma ; r \bigr ) .\tag{2}
$$

Temporal propagation. The local support $\mathcal { T } _ { i }$ contains $t _ { i }$ and the open intervals to adjacent source times with available motion estimates. Under a locally linear approximation, the target position is

$$
\mathbf { x } _ { i } ( t ) = \left\{ \begin{array} { l l } { \mathbf { x } _ { i } , } & { b _ { i } = 0 \mathrm { o r } t = t _ { i } , } \\ { \mathbf { x } _ { i } + ( t - t _ { i } ) \mathbf { v } _ { i } ^ { + } , } & { b _ { i } = 1 , t \in \mathcal { T } _ { i } , t > t _ { i } , } \\ { \mathbf { x } _ { i } + ( t _ { i } - t ) \mathbf { v } _ { i } ^ { - } , } & { b _ { i } = 1 , t \in \mathcal { T } _ { i } , t < t _ { i } . } \end{array} \right.\tag{3}
$$

The backward velocity points toward the past, hence the positive interval $t _ { i } - t .$

The propagated entry $\mathcal { C } _ { q } ( t ) = \mathrm { P r o p a g a t i o n } ( \mathcal { C } _ { q } , t )$ retains units with defined target positions and preserves their feature bindings. The source entry $\mathcal { C } _ { q }$ remains unchanged and available for subsequent requests. For $b _ { i } = 0 .$ , the reference position remains defined beyond $\textstyle { \mathcal { T } } _ { i } .$ . To suppress spurious motion on static structure, we set $b _ { i }$ by thresholding projected displacement in the fixed source camera. The threshold ϵ is measured in source-image pixels per video frame.

Camera projection. For each visual representation r, we project $ { \mathcal { C } } _ { q } ( t )$ through Γ onto its target grid, obtaining candidate units at each cell $ { \mathbf { g } } \in  { \mathbb { Z } } ^ { 2 }$ . Depth alone would let closer units from other times hide moving content, so we take moving candidates from the nearest source time first and fill remaining cells with static ones. The selected unit index $i ^ { \star } ( { \bf g } )$ determines the readout features:

$$
\mathbf { H } ^ { ( r ) } [ \mathbf { g } ] = \mathbf { f } _ { i ^ { \star } ( \mathbf { g } ) } ^ { ( r ) } .\tag{4}
$$

Coverage $\mathbf { M } ^ { ( r ) } [ \mathbf { g } ] = 1$ when a unit is selected; otherwise, both are zero, and the coverage mask distinguishes missing conten from valid zero-valued features.

An entry constructed from K keyframes can provide conditions for an N-frame target sequence, including $N > K .$ , by reading each representation r at its sampled target indices $\mathcal { T } _ { r } \subseteq \{ 0 , \dots , N - 1 \}$ to obtain readouts $( \mathbf { H } _ { j } ^ { ( r ) } , \mathbf { M } _ { j } ^ { ( \bar { r } ) } )$ . Different visual representations therefore share one access procedure with their own feature grids, dimensions, and temporal sampling

## 3.3 4LC Composition

Composition applies Readout to selected cache entries in the specified order and combines their readouts to supplement coverage for the same target time and camera sequences (Fig. 2c). For an entry $\mathcal { C } _ { q } ,$ , let $( \mathbf { H } _ { q , j } ^ { ( r ) } , \mathbf { M } _ { q , j } ^ { ( r ) } )$ denote its readout at target index j for representation r. The subscript q retains the entry’s insertion index in C, independently of the order of access. For each representation r and $j \in \mathcal { I } _ { r }$ , we initialize the accumulated outputs as $\mathbf { H } _ { j } ^ { ( r ) } = \mathbf { 0 }$ and $\mathbf { M } _ { j } ^ { ( r ) } = \mathbf { 0 }$ on that representation’s grid. Each selected entry contributes through the updates

$$
\mathbf { H } _ { j } ^ { ( r ) }  \mathbf { H } _ { j } ^ { ( r ) } + ( \mathbf { 1 } - \mathbf { M } _ { j } ^ { ( r ) } ) \odot \mathbf { H } _ { q , j } ^ { ( r ) } , \qquad \mathbf { M } _ { j } ^ { ( r ) }  \mathbf { M } _ { j } ^ { ( r ) } \lor \mathbf { M } _ { q , j } ^ { ( r ) } .\tag{5}
$$

We apply the feature update before the coverage update, using the previous coverage to retain the first valid readout at overlapping locations. The mask is broadcast across channels in the elementwise product ⊙; ∨ denotes elementwise logical OR. For each representation, the coverage ratio is the fraction of covered grid cells across its sampled target sequence. Composition returns when every representation reaches the prescribed coverage-ratio threshold or no selected entries remain. Entries therefore need not be accumulated into a single entry, avoiding the cost and redundancy of accumulation, and can be combined or evicted independently across requests.

## 3.4 4LC-Video

Representation adaptation. Visual representations differ in spatial resolution, temporal sampling, and feature dimension. A condition adapter aligns the outputs of Readout or Composition to the generator’s spatiotemporal latent grid, using coverage masks to exclude missing values (Fig. 3). Spatial alignment interpolates masked features and normalizes by interpolated coverage, keeping uncovered locations at zero. Temporal alignment maps the initial readout to the first latent step and aggregates subsequent readouts within each latent time interval, averaging valid contributions and recording their coverage. Separate linear projections map the aligned representations, merged coverage mask, and target-camera rays before concatenation and fusion into $\mathbf { h } _ { \mathrm { c o n d } }$

Table 1. Task configurations for 4LC-Video. $\scriptstyle \mathrm { R e - }$ quests use one entry $\mathcal { C } _ { q }$ or selected entries from C. $\bar { t } _ { j }$ preserves the source or default single-image timeline, and $\bar { \Gamma } _ { j }$ the source camera trajectory. Scene revisits use one readout along a camera path that returns to its start.
<table><tr><td>Task</td><td>Initial frames</td><td>Target request  $( t _ { j } , \Gamma _ { j } )$ </td><td>Condition source</td></tr><tr><td>Camera control</td><td> $\{ \mathbf { I } _ { k } \} _ { k = 0 } ^ { K - 1 }$ </td><td> $( \bar { t } _ { j } , \Gamma _ { j } )$ </td><td> $\mathcal { C } _ { q }$ </td></tr><tr><td>Temporal control</td><td> $\{ \mathbf { I } _ { k } \} _ { k = 0 } ^ { K - 1 }$ </td><td> $( t _ { j } , \bar { \Gamma } _ { j } )$ </td><td> $\mathcal { C } _ { q }$ </td></tr><tr><td>Novel-view synthesis</td><td> $\mathbf { I } _ { 0 }$ </td><td> $( \bar { t } _ { j } , \Gamma _ { j } )$ </td><td> $\mathcal { C } _ { q }$ </td></tr><tr><td>Scene revisits</td><td> $\mathbf { I } _ { 0 }$ </td><td> $( \bar { t } _ { j } , \Gamma _ { j } )$ </td><td> $\mathcal { C } _ { q }$ </td></tr><tr><td>Long-term generation</td><td> $\mathbf { I } _ { 0 }$ </td><td> $( \bar { t } _ { j } , \Gamma _ { j } )$ </td><td> $\mathcal { C }$ </td></tr></table>

![](images/3732b6cd49fe4c7ad1de06c157c1ce1b0967a3a0faed150deed276886c01110e.jpg)  
Figure 3. 4LC-Video framework. The condition adapter aligns readouts to the Wan2.1 VAE latent grid and fuses them with masks and camera rays for the control branch. Generated observations extend the cache.

Conditional generation. A control branch injects this condition into the diffusion model to guide generation of target observations and complete uncovered regions. Construction converts generated observations and their prescribed cameras and timestamps into new entries in C for reuse. Because Readout takes times and cameras independently and Composition combines entries, the tasks in Table 1 share one model and differ only in initial frames, target requests, and condition sources.

Training. We train the condition adapter and control branch with a weighted flow-matching objective [42], keeping the visual encoders and video diffusion backbone frozen. Monocular training videos supply target sequences and sampled source observations, whose cache entries are queried at the original frame times and cameras.

## 4 Experiments

## 4.1 Experimental Setup

Implementation details. 4LC-Video uses Wan2.1-T2V-14B [43] with a VACE control branch [44] and the pretrained NeoVerse reconstructor [10] for geometry and motion estimation. We select FLUX.2 VAE [45] and V-JEPA 2.1 [46] as the default encoders for extracting visual representations. Training uses 200K monocular clips from SpatialVID [47], RealCam-Vid [48], and OpenVid [49] without paired multiview supervision. Inference generates 81-frame 560 × 336 segments with the LightX2V 4-step LoRA [50].

Benchmarks and baselines. Camera control compares with SpaceTimePilot [19], TrajectoryCrafter [35], GEN3C [4], and NeoVerse [10], and temporal control with SpaceTimePilot and NeoVerse. Both tasks report camera errors and six VBench [51] scores. Novel-view synthesis on RealEstate10K [52] and scene revisits on WorldScore [53] compare with ViewCrafter [54], FlexWorld [30], Voyager [55], Spatia [5], Mirage [13], and LiveWorld [7] with PSNR, SSIM [56], and LPIPS [57]. Long-term generation on MBench-A compares with published results [58] of HY-WorldPlay [23], Infinite-World [22], Yume-1.5 [26], and LingBot-World [24]. Appendix A.3 and Appendix A.4 provide evaluation protocols and additional results.

![](images/f44f9871b8da9dba032bb755ed48c0ce5c144cf3e2120a3cd68f2cc2eb9a5fd4.jpg)  
Figure 4. Qualitative comparison for camera and temporal control. Left: dynamic scene; right: static scene. Top: Static camera. The camera is fixed while scene time follows the source timeline. Bottom: Frozen time. Scene time is fixed at source frame 0 while the camera follows the source trajectory. Input frame labels identify each column; red boxes highlight distortions.

## 4.2 Main Results on Multiple Video Generation Tasks

We evaluate a single 4LC-Video checkpoint on the five tasks in Table 1 without task-specific fine-tuning. Camera and temporal control read 4LC built from a source video with static and dynamic content, novel-view synthesis and scene revisits read an entry built from a single image, and long-term generation reads entries accumulated across generated segments.

Camera control. 4LC-Video attains the lowest camera errors together with the highest subject and background consistency (Table 2), whereas the baselines trade one for the other: NeoVerse follows the cameras with lower consistency, GEN3C and SpaceTimePilot stay consistent but close to static views with several-fold camera errors, and TrajectoryCrafter trails on both. 4LC-Video avoids this trade-off because 4LC keeps static structure persistent and dynamic content accurate. GEN3C, TrajectoryCrafter, and NeoVerse instead render conditions from reconstructed point clouds or 4DGS, whose artifacts and holes appear on moving subjects and static structures in NeoVerse outputs, while 4LC-Video stays faithful to the source observations (Fig. 4).

Temporal control. 4LC-Video keeps this advantage under temporal control (Table 2), while the baselines show the same trade-off. For time and camera pairs never observed together, 4LC reads each moving subject from its own observations and moves it by its estimated motion, while static structure stays in place. SpaceTimePilot generates the retimed motion withou

Table 2. Quantitative comparison for camera and temporal control. We evaluate 100 randomly sampled unseen videos under four camera trajectories (400 cases) and three temporal trajectories (300 cases). $r _ { \mathrm { d i s t } }$ measures rotation error in radians and $t _ { \mathrm { d i s t } }$ scale-normalized translation error; higher VBench scores are better. Best and second-best results per task are bold and underlined.
<table><tr><td>Method</td><td>rdist↓</td><td>tdist↓</td><td>Subj. Consist.</td><td>Back. Consist.</td><td>Temp. Flick.</td><td>Motion Smooth.</td><td>Aesth. Quality</td><td>Imag. Quality</td></tr><tr><td>Camera control</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SpaceTimePilot</td><td>0.179</td><td>0.321</td><td>89.79</td><td>93.43</td><td>95.26</td><td>98.69</td><td>47.60</td><td>57.23</td></tr><tr><td>TrajectoryCrafter</td><td>0.244</td><td>0.395</td><td>86.37</td><td>92.19</td><td>92.69</td><td>96.17</td><td>45.16</td><td>60.97</td></tr><tr><td>GEN3C</td><td>0.203</td><td>0.285</td><td>90.45</td><td>93.79</td><td>95.66</td><td>97.55</td><td>45.62</td><td>56.39</td></tr><tr><td>NeoVerse</td><td>0.059</td><td>0.120</td><td>88.63</td><td>92.77</td><td>94.43</td><td>98.17</td><td>50.38</td><td>67.84</td></tr><tr><td>4LC-Video (ours)</td><td>0.037</td><td>0.089</td><td>90.71</td><td>94.14</td><td>95.47</td><td>98.06</td><td>48.16</td><td>65.49</td></tr><tr><td>Temporal control</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SpaceTimePilot</td><td>0.163</td><td>0.192</td><td>88.42</td><td>92.65</td><td>94.64</td><td>98.37</td><td>47.44</td><td>56.40</td></tr><tr><td>NeoVerse</td><td>0.057</td><td>0.080</td><td>87.55</td><td>92.11</td><td>93.38</td><td>97.88</td><td>49.65</td><td>66.85</td></tr><tr><td>4LC-Video (ours)</td><td>0.041</td><td>0.060</td><td>88.58</td><td>92.85</td><td>93.78</td><td>97.41</td><td>48.75</td><td>65.14</td></tr></table>

explicit conditions, and NeoVerse renders it from 4DGS with the artifacts noted above. In the frozen-time examples of Fig. 4, 4LC-Video holds the subject at frame 0 and keeps later-observed background intact, whereas NeoVerse deforms the subject inconsistently.

Novel-view synthesis and Scene revisits. 4LC-Video obtains the best novel-view LPIPS with PSNR within 0.1 dB of LiveWorld, and it exceeds ViewCrafter, FlexWorld, Voyager, and Spatia on all three metrics (Table 3). For static scenes, 4LC reuses the cached visual representations of the image without RGB rendering and re-encoding, so its conditions avoid the information loss of these steps. Mirage attains higher SSIM at a higher output resolution (1280 × 704 versus $5 6 0 \times 3 3 6 )$ , while its PSNR and LPIPS are worse than those of 4LC-Video. Scene revisits expose accumulated drift. 4LC-Video leads LiveWorld by 1.84 dB in PSNR<sub>C</sub> and attains the best LPIPS<sub>C</sub>. The baselines update their point clouds or spatial memories with generated views along the path, allowing intermediate

Table 3. Quantitative comparison for novel-view synthesis and scene revisits. We use the protocols and baseline results reported by Spatia [5] and Mirage [13], and evaluate LiveWorld under the same protocols.
<table><tr><td rowspan="2">Method</td><td colspan="3">Novel-view synthesis</td><td colspan="3">Closed-loop consistency</td></tr><tr><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>PSNRc↑</td><td>SSIMc↑</td><td>LPIPSC↓</td></tr><tr><td>ViewCrafter</td><td>15.78</td><td>0.580</td><td>0.396</td><td>14.79</td><td>0.481</td><td>0.365</td></tr><tr><td>FlexWorld</td><td>16.25</td><td>0.593</td><td>0.370</td><td>12.20</td><td>0.428</td><td>0.598</td></tr><tr><td>Voyager</td><td>17.79</td><td>0.636</td><td>0.297</td><td>17.66</td><td>0.540</td><td>0.380</td></tr><tr><td>Spatia</td><td>18.58</td><td>0.646</td><td>0.254</td><td>19.38</td><td>0.579</td><td>0.213</td></tr><tr><td>Mirage</td><td>18.38</td><td>0.779</td><td>0.250</td><td>20.05</td><td>0.825</td><td>0.228</td></tr><tr><td>LiveWorld</td><td>19.04</td><td>0.645</td><td>0.191</td><td>20.07</td><td>0.653</td><td>0.186</td></tr><tr><td>4LC-Video (ours)</td><td>18.94</td><td>0.650</td><td>0.188</td><td>21.91</td><td>0.705</td><td>0.161</td></tr></table>

errors to re-enter the conditions of return views. Reusing one source-image entry, 4LC gives each revisited camera the same condition regardless of intermediate outputs.

Long-term generation. Reuse across generated segments separates the MBench-A baselines, which either preserve the environment or evolve the scene (Table 4). HY-WorldPlay and Infinite-World score high on spatial and rendering metrics but low on state evolution, as their scenes remain overly static [58]. Yume-1.5 and LingBot-World evolve scene states but lose spatial consistency. 4LC-Video bridges the two with the highest Environment and Causal scores. Each generated segment is appended to 4LC as a new entry, and later segments reuse only the static units of earlier entries as history through Composition. Revisited structure is thus constrained, while past motion is not replayed and scene states can keep evolving. Return views reproduce the outbound content, whereas HY-WorldPlay shifts the layout of revisited regions (Fig. 6 in Appendix A.4).

Table 4. Quantitative comparison for long-term generation on MBench-A [58]. We compare Environment and Causal scores with published leaderboard results.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Env. score</td><td rowspan="2">Causal score</td><td colspan="2">Spatial</td><td colspan="2">Rendering</td><td colspan="2">Self-evolution</td><td>Interaction</td></tr><tr><td>Epipolar</td><td>Reproj.</td><td>Lighting</td><td>Style</td><td>State</td><td>Correct.</td><td>Action</td></tr><tr><td>HY-WorldPlay</td><td>83.23</td><td>65.86</td><td>83.86</td><td>68.17</td><td>82.67</td><td>98.23</td><td>49.50</td><td>62.40</td><td>85.69</td></tr><tr><td>Infinite-World</td><td>73.76</td><td>70.92</td><td>74.04</td><td>61.51</td><td>62.63</td><td>96.87</td><td>48.00</td><td>78.40</td><td>86.37</td></tr><tr><td>Yume-1.5</td><td>54.92</td><td>85.03</td><td>51.86</td><td>24.55</td><td>51.21</td><td>92.05</td><td>97.90</td><td>95.00</td><td>62.20</td></tr><tr><td>LingBot-World</td><td>38.90</td><td>82.91</td><td>22.12</td><td>7.57</td><td>40.06</td><td>85.87</td><td>96.00</td><td>89.40</td><td>63.32</td></tr><tr><td>4LC-Video (ours)</td><td>85.56</td><td>89.06</td><td>71.11</td><td>75.98</td><td>97.52</td><td>97.66</td><td>85.00</td><td>93.20</td><td>88.97</td></tr></table>

![](images/eab8c0ff71f3b6b1bf944cf1fd01f70e0d97b3b4ee79fa596150d1e579564573.jpg)

![](images/2a6db68227a62aa35a9f21736c4b12cadff72bf5190fa5d15b7360539c2dfec9.jpg)

![](images/a417d4f4509d28b28043488a7c5d54b4513bf9f70a33f8d3fdc46f78d84c642c.jpg)  
Figure 5. Efficiency and feature similarity across condition-acquisition paths. Left: Latency and peak GPU memory of 4LC readout and of RGB rendering and VAE encoding from NeoVerse’s 4DGS and LiveWorld’s RGB point clouds, with the end-to-end latency of each framework. Rendering + Encoding includes rendering time. Solid and dashed 4LC Readout bars use Wan2.1 VAE alone and with DINOv2, VGGT, and V-JEPA 2.1. Latency uses a square-root scale. Right: Feature similarity between direct readouts and re-encoded features, as per-case mean cosine similarity in the original (Absolute) and relative-representation (Relative) spaces; black ticks mark medians.

## 4.3 Conditioning Efficiency and Quality

Conditioning efficiency. Condition acquisition accounts for 14.9% of the 18.47 s end-to-end generation time in NeoVerse, compared with 0.25% of 16.19 s for 4LC-Video under the same generation settings, and this overhead recurs across segments in long-term generation. In Fig. 5, NeoVerse, LiveWorld, and 4LC-Video all use the Wan2.1 backbone and VAE encoder for 81-frame requests at 560 × 336 (Appendix A.3). The conditioning paths in NeoVerse [10] and LiveWorld [7] render image-resolution RGB views from 4DGS and RGB point clouds, respectively, followed by VAE encoding (Fig. 5, left). 4LC readout directly projects cached representations onto lower-resolution feature grids without repeated encoding. Compared with 4DGS rendering followed by VAE encoding in NeoVerse, 4LC readout is 68.5× faster, with 12.6× lower peak GPU memory. Adding DINOv2, VGGT, and V-JEPA 2.1 to Wan2.1 VAE raises readout latency from 40.14 to 196.14 ms and peak GPU memory from 0.385 to 1.245 GiB. The default FLUX.2 VAE with V-JEPA 2.1 takes 162.03 ms and 0.763 GiB, and adding DINOv2 and VGGT raises them to 255.20 ms and 1.605 GiB. All remain below the cost of rendering and encoding one representation.

Conditioning quality. To test whether direct readout preserves the information of re-encoding, we compare it with reencoding views rendered from cached source pixels under the same Readout and encoders. All features retain comparable absolute similarity (Fig. 5, right), while semantic and geometric features also preserve stable relative structure [59], with VGGT [60] and V-JEPA 2.1 concentrating near one in relative space and DINOv2 [61] showing intermediate similarity. VAE features show lower relative similarity (medians 0.85 for FLUX.2 VAE and 0.71 for Wan2.1 VAE), as projection, resampling, and re-encoding alter their pixel-level detail. Sec. 4.4 examines the effect of condition-acquisition paths on generation.

## 4.4 Ablation Studies

Condition-acquisition path. For each VAE, we compare direct readout with RGB rendering and re-encoding (Table 5). Re-encoding updates features in the rendered target context, while direct readout retains the source encoding as content moves and visibility changes. With FLUX.2 VAE, direct readout leaves subject and background consistency nearly unchanged, with a small difference in camera errors. For Wan2.1 VAE, direct readout lowers both consistency scores and raises camera errors more. This matches FLUX.2 VAE’s higher readout similarity in Sec. 4.3 and supports its selection.

Composite visual representations. A single visual representation cannot meet the representational demands of dynamic world modeling, which span appearance, semantics, geometry, and dynamics, whereas existing encoders each specialize in one of them: FLUX.2 VAE reconstructs appearance, DINOv2 learns semantic features, VGGT infers multi-view geometry, and V-JEPA 2.1 models video dynamics. Since 4LC binds these representations to shared geometry and motion,

Table 5. Ablations of condition-acquisition paths and composite visual representations. All variants share a frozen Wan2.1-T2V-1.3B backbone and the same 100 unseen videos with all trajectories. ALL denotes the other three encoders.
<table><tr><td>Visual representations</td><td>rdist↓ tdist↓</td><td></td><td>Subj. Consist.</td><td>Back. Consist.</td></tr><tr><td colspan="5">RGB rendering and re-encoding</td></tr><tr><td>Wan2.1 VAE</td><td>0.046</td><td>0.106</td><td>89.86</td><td>93.69</td></tr><tr><td>FLUX.2 VAE</td><td>0.044</td><td>0.106</td><td>90.00</td><td>93.80</td></tr><tr><td>Direct readout</td><td></td><td></td><td></td><td></td></tr><tr><td>Wan2.1 VAE</td><td>0.056</td><td>0.128</td><td>89.03</td><td>93.03</td></tr><tr><td>FLUX.2 VAE</td><td>0.049</td><td>0.114</td><td>90.06</td><td>93.81</td></tr><tr><td colspan="5">Direct readout with composite representations</td></tr><tr><td>FLUX.2 VAE + DINOv2</td><td>0.051</td><td>0.116</td><td>89.89</td><td>93.69</td></tr><tr><td>FLUX.2 VAE + VGGT</td><td>0.048</td><td>0.120</td><td>89.89</td><td>93.70</td></tr><tr><td>FLUX.2 VAE + V-JEPA 2.1</td><td>0.046</td><td>0.112</td><td>90.05</td><td>93.85</td></tr><tr><td>FLUX.2 VAE + ALL</td><td>0.044</td><td>0.101</td><td>90.15</td><td>93.90</td></tr></table>

we add DINOv2, VGGT, and V-JEPA 2.1 to FLUX.2 VAE under direct readout, individually and together (Table 5). Individually, DINOv2 and VGGT keep camera errors and consistency close to FLUX.2 VAE, indicating that their semantic and geometric cues are not fully exploited in pairwise combinations. V-JEPA 2.1 lowers both rotation and translation errors while leaving consistency nearly unchanged: unlike the image and multi-view encoders, it learns from video, and its temporally consistent features [46] keep high similarity after readout, bringing temporal context into the condition. Combining all four lets appearance, semantic, geometric, and dynamic cues complement one another, making it the strongest configuration: it reduces the translation error of FLUX.2 VAE combined with V-JEPA 2.1 from 0.112 to 0.101, and all four metrics match o exceed FLUX.2 VAE re-encoding. Because Readout reuses cached representations, it adds no efficiency bottleneck (Fig. 5, Sec. 4.3), so 4LC is both composite and efficient, yielding rich, high-quality 4D conditions.

## 5 Conclusion

We presented 4LC, a 4D latent cache that stores composite visual representations of static and dynamic content and reads them directly at independently requested times and viewpoints. Conditioned on these readouts, a single 4LC-Video checkpoint achieved state-of-the-art control accuracy and scene consistency across five generation tasks. By avoiding per-request RGB rendering and re-encoding, 4LC turns condition acquisition from a recurring cost into a one-time encoding of observations. Its composite representations add complementary information from multiple encoders, giving dynamic world modeling conditions that are both efficient and rich.

Limitations and future work. 4LC relies on a feed-forward 4D reconstruction model to estimate depth, camera parameters, and motion, and its estimation errors can affect readout quality despite training augmentation. While 4LC supports both static and dynamic content, 4LC-Video reuses only static historical content for long-term generation. We leave dynamic reuse across generation segments to future work, as a further step toward efficient conditioning for multi-task video generation.

## References

[1] David Yifan Yao, Albert J. Zhai, and Shenlong Wang. Uni4D: Unifying visual foundation models for 4D modeling from a single video. In CVPR, 2025.

[2] Feng Qiao, Zhaochong An, Zhexiao Xiong, Serge Belongie, and Nathan Jacobs. Track2View: 4D-Consistent cameracontrolled video generation via paired 3D point tracks. arXiv preprint arXiv:2606.15534, 2026.

[3] Shaoheng Fang, Hanwen Jiang, Yunpeng Bai, Niloy J. Mitra, and Qixing Huang. WorldReel: 4D video generation with consistent geometry and motion modeling. In CVPR, 2026.

[4] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Muller, Alexander¨ Keller, Sanja Fidler, and Jun Gao. GEN3C: 3D-Informed world-consistent video generation with precise camera control. In CVPR, 2025.

[5] Jinjing Zhao, Fangyun Wei, Zhening Liu, Hongyang Zhang, Chang Xu, and Yan Lu. Spatia: Video generation with updatable spatial memory. In CVPR, 2026.

[6] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. In NeurIPS, 2025.

[7] Zicheng Duan, Jiatong Xia, Zeyu Zhang, Wenbo Zhang, Gengze Zhou, Chenhui Gou, Yefei He, Feng Chen, Xinyu Zhang, and Lingqiao Liu. LiveWorld: Simulating out-of-sight dynamics in generative video world models. In ECCV, 2026.

[8] Sixiao Zheng, Minghao Yin, Wenbo Hu, Xiaoyu Li, Ying Shan, and Yanwei Fu. VerseCrafter: Dynamic realistic video world model with 4D geometric control. In CVPR, 2026.

[9] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4D Gaussian splatting for real-time dynamic scene rendering. In CVPR, 2024.

[10] Yuxue Yang, Lue Fan, Ziqi Shi, Junran Peng, Feng Wang, and Zhaoxiang Zhang. NeoVerse: Enhancing 4D world mode with in-the-wild monocular videos. In CVPR, 2026.

[11] Jung Yi, Minjae Kim, Paul Hyunbin Cho, Wooseok Jang, Sangdoo Yun, and Seungryong Kim. WorldKV: Efficient world memory with world retrieval and compression. arXiv preprint arXiv:2605.22718, 2026.

[12] Weilun Feng, Guoxin Fan, Haotong Qin, Mingqiang Wu, Yuqi Li, Xiangqi Li, Zhulin An, Libo Huang, Dingrui Wang, Longlong Liao, Michele Magno, Yongjun Xu, and Chuanguang Yang. WorldCache: Accelerating world models for free via heterogeneous token caching. In ICML, 2026.

[13] Weijie Wang, Haoyu Zhao, Yifan Yang, Feng Chen, Zeyu Zhang, Yefei He, Zicheng Duan, Donny Y. Chen, Yuqing Yang, and Bohan Zhuang. Latent spatial memory for video world models. arXiv preprint arXiv:2606.09828, 2026.

[14] Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, and Tong He. Aether: Geometric-aware unified world modeling. In ICCV, 2025.

[15] Yabo Chen, Yuanzhi Liang, Jiepeng Wang, Tingxi Chen, Junfei Cheng, Zixiao Gu, Yuyang Huang, Zicheng Jiang, Wei Li, Tian Li, Weichen Li, Zuoxin Li, Guangce Liu, Jialun Liu, Junqi Liu, Haoyuan Wang, Qizhen Weng, Xuan’er Wu, Xunzhi Xiang, Xiaoyan Yang, Xin Zhang, Shiwen Zhang, Junyu Zhou, Chengcheng Zhou, Haibin Huang, Ch Zhang, and Xuelong Li. TeleWorld: Towards dynamic multimodal synthesis with a 4D world model. arXiv preprint arXiv:2601.00051, 2025.

[16] InSpatio Team, Donghui Shen, Guofeng Zhang, Haomin Liu, Haoyu Ji, Hujun Bao, Hongjia Zhai, Jialin Liu, Jing Guo, Nan Wang, Siji Pan, Weihong Pan, Weijian Xie, Xianbin Liu, Xiaojun Xiang, Xiaoyu Zhang, Xinyu Chen, Yifu Wang, Yipeng Chen, Zhenzhou Fan, Zhewen Le, Zhichao Ye, and Ziqiang Zhao. INSPATIO-WORLD: A real-time 4D world simulator via spatiotemporal autoregressive modeling. arXiv preprint arXiv:2604.07209, 2026.

[17] Junchao Huang, Xinting Hu, Boyao Han, Shaoshuai Shi, Zhuotao Tian, Tianyu He, and Li Jiang. Memory forcing: Spatio-temporal memory for consistent scene generation on Minecraft. arXiv preprint arXiv:2510.03198, 2025.

[18] Yiming Wang, Qihang Zhang, Shengqu Cai, Tong Wu, Jan Ackermann, Zhengfei Kuang, Yang Zheng, Frano Rajic, Siyuˇ Tang, and Gordon Wetzstein. BulletTime: Decoupled control of time and camera pose for video generation. In CVPR, 2026.

[19] Zhening Huang, Hyeonho Jeong, Xuelin Chen, Yulia Gryaditskaya, Tuanfeng Y. Wang, Joan Lasenby, and Chun-Hao Huang. SpaceTimePilot: Generative rendering of dynamic scenes across space and time. In CVPR, 2026.

[20] Zeqi Xiao, Yushi LAN, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. WorldMem: Long-term consistent world simulation with memory. In NeurIPS, 2025.

[21] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. VMem: Consistent interactive video scene generation with surfel-indexed view memory. In ICCV, 2025.

[22] Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, and Ming-Ming Cheng. Infinite-World: Scaling interactive world models to 1000-frame horizons via pose-free hierarchical memory. In ICML, 2026.

[23] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. WorldPlay: Towards long-term geometric consistency for real-time interactive world modeling. In ICML, 2026.

[24] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[25] Junchao Huang, Ziyang Ye, Xinting Hu, Tianyu He, Guiyu Zhang, Shaoshuai Shi, Jiang Bian, and Li Jiang. LIVE: Long-horizon interactive video world modeling. In ICML, 2026.

[26] Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, and Kaipeng Zhang. Yume1.5: A text-controlled interactive world generation model. In CVPR, 2026.

[27] Zhengxuan Wei, Xu Guo, Xinghui Li, Xunzhi Xiang, Min Wei, Yiran Zhu, Qiulin Wang, Xintao Wang, Pengfei Wan, Xiangwang Hou, and Qi Fan. Geometry-aware implicit memory for video world models. arXiv preprint arXiv:2606.02436, 2026.

[28] Wei Yu, Runjia Qian, Yumeng Li, Liquan Wang, Songheng Yin, Sri Siddarth Chakaravarthy P, Dennis Anthony, Yang Ye, Yidi Li, Weiwei Wan, and Animesh Garg. MosaicMem: Hybrid spatial memory for controllable video world models. arXiv preprint arXiv:2603.17117, 2026.

[29] Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, and Xiang Bai. Out of sight but no out of mind: Hybrid memory for dynamic video world models. arXiv preprint arXiv:2603.25716, 2026.

[30] Luxi Chen, Zihan Zhou, Min Zhao, Yikai Wang, Ge Zhang, Wenhao Huang, Hao Sun, Ji-Rong Wen, and Chongxuan LI. FlexWorld: Progressively expanding 3D scenes for flexible-view exploration. In NeurIPS, 2025.

[31] Chuhan Zhang, Guillaume Le Moing, Skanda Koppula, Ignacio Rocco, Liliane Momeni, Junyu Xie, Shuyang Sun, Rahul Sukthankar, Joelle K. Barral, Raia Hadsell, Zoubin Ghahramani, Andrew Zisserman, Junlin Zhang, and Mehdi S. M.¨ Sajjadi. Efficiently reconstructing dynamic scenes one D4RT at a time. In CVPR, 2026.

[32] Edgar Sucar, Eldar Insafutdinov, Zihang Lai, and Andrea Vedaldi. V-DPM: 4D video reconstruction with dynamic poin maps. In CVPR, 2026.

[33] Yihang Luo, Shangchen Zhou, Yushi Lan, Xingang Pan, and Chen Change Loy. 4RC: 4D reconstruction via conditiona querying anytime and anywhere. In ICML, 2026.

[34] Zhen Xu, Zhengqin Li, Zhao Dong, Xiaowei Zhou, Richard Newcombe, and Zhaoyang Lv. 4DGT: Learning a 4D Gaussian transformer using real-world monocular videos. In NeurIPS, 2025.

[35] Mark Yu, Wenbo Hu, Jinbo Xing, and Ying Shan. TrajectoryCrafter: Redirecting camera trajectory for monocular videos via diffusion models. In ICCV, 2025.

[36] Kuan Heng Lin, Zhizheng Liu, Pablo Salamanca, Yash Kant, Ryan Burgert, Yuancheng Xu, Koichi Namekata, Yiwe Zhao, Bolei Zhou, Micah Goldblum, Paul Debevec, and Ning Yu. Vista4D: Video reshooting with 4D point clouds. In CVPR, 2026.

[37] Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lempitsky. Neural point-based graphics. In ECCV, 2020.

[38] Christopher Wewer, Kevin Raj, Eddy Ilg, Bernt Schiele, and Jan Eric Lenssen. latentSplat: Autoencoding variational Gaussians for fast generalizable 3D reconstruction. In ECCV, 2024.

[39] Chaoyi Zhou, Xi Liu, Feng Luo, and Siyu Huang. Latent radiance fields with 3D-aware 2D representations. In ICLR, 2025.

[40] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. MVSplat360: Feed-forward 360 scene synthesis from sparse views. In NeurIPS, 2024.

[41] Or Hirschorn, Omer Sela, Inbar Huberman-Spiegelglas, Netalee Efrat, Eli Alshan, Ianir Ideses, Frederic Devernay, Yochai Zvik, and Lior Fritz. Splatent: Splatting diffusion latents for novel view synthesis. In CVPR, 2026.

[42] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.

[43] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwe Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[44] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. VACE: All-in-one video creation and editing. In ICCV, 2025.

[45] Black Forest Labs. FLUX.2: Frontier visual intelligence. Official model release, 2025. URL https://bfl.ai/ blog/flux-2.

[46] Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-JEPA 2.1: Unlocking dense features in video self-supervised learning. arXiv preprint arXiv:2603.14482, 2026.

[47] Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Chang Zeng, Yanxi Zhou, Xiao-Xiao Long, Hao Zhu, Zhaoxiang Zhang, Xun Cao, and Yao Yao. SpatialVID: A large-scale video dataset with spatial annotations. In CVPR, 2026.

[48] Guangcong Zheng, Teng Li, Xianpan Zhou, and Xi Li. RealCam-Vid: High-resolution video dataset with dynamic scenes and metric-scale camera movements. arXiv preprint arXiv:2504.08212, 2025.

[49] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. OpenVid-1M: A large-scale high-quality dataset for text-to-video generation. In ICLR, 2025.

[50] LightX2V Contributors. LightX2V: Light video generation inference framework. GitHub repository, 2025. URL https://github.com/ModelTC/lightx2v.

[51] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.

[52] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. ACM TOG, 37(4):1–12, 2018.

[53] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. WorldScore: A unified evaluation benchmark fo world generation. In ICCV, 2025.

[54] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. ViewCrafter: Taming video diffusion models for high-fidelity novel view synthesis. IEEE TPAMI, pages 1–18, 2025.

[55] Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson Lau, Wangmeng Zuo, and Chunchao Guo. Voyager: Long-range and world-consistent video diffusion for explorable 3D scene generation. ACM TOG, 44(6):1–15, 2025.

[56] Zhou Wang, Alan Conrad Bovik, Hamid Rahim Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE TIP, 13(4):600–612, 2004.

[57] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

[58] Shengjun Zhang, Zhang Zhang, Simin Huang, Zhenyu Tang, Hanyang Wang, Chensheng Dai, Min Chen, Yifan Li, Yuxin Li, Yingjie Chen, Hao Liu, Chen Li, Jing Lyu, and Yueqi Duan. MBench: A comprehensive benchmark on memory capability for video world models. arXiv preprint arXiv:2606.00793, 2026.

[59] Luca Moschella, Valentino Maiorca, Marco Fumero, Antonio Norelli, Francesco Locatello, and Emanuele Rodola.\` Relative representations enable zero-shot latent space communication. In ICLR, 2023.

[60] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visua geometry grounded transformer. In CVPR, 2025.

[61] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,´ Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou,´ Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024.

[62] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo DAI, Hongsheng Li, and Ceyuan Yang. CameraCtrl: Enabling camera control for video diffusion models. In ICLR, 2025.

[63] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019.

[64] Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T. Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, and Charles Herrmann. WonderJourney: Going from anywhere to everywhere. In CVPR, 2024.

[65] Paul Engstler, Andrea Vedaldi, Iro Laina, and Christian Rupprecht. Invisible Stitch: Generating smooth 3D scenes with depth inpainting. In 3DV, 2025.

[66] Hong-Xing Yu, Haoyi Duan, Charles Herrmann, William T. Freeman, and Jiajun Wu. WonderWorld: Interactive 3D scene generation from a single image. In CVPR, 2025.

[67] Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying Shan. VideoCrafter2: Overcoming data limitations for high-quality video diffusion models. In CVPR, 2024.

[68] Jiaqi Xu, Kunzhe Huang, Xinyi Zou, Yunkuo Chen, Bo Liu, Mengli Cheng, Jun Huang, and Xing Shi. EasyAnimate: High-performance video generation framework with hybrid windows attention and reward backpropagation. In ACM MM, 2025.

[69] Yuan Zhou, Qiuyue Wang, Yuxuan Cai, and Huan Yang. Allegro: Open the black box of commercial-level video generation model. arXiv preprint arXiv:2410.15458, 2024.

[70] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-video diffusion models with an expert transformer. In ICLR, 2025.

[71] Weichen Fan, Chenyang Si, Junhao Song, Zhenyu Yang, Yinan He, Long Zhuo, Ziqi Huang, Ziyue Dong, Jingwen He, Dongwei Pan, Yi Wang, Yuming Jiang, Yaohui Wang, Peng Gao, Xinyuan Chen, Hengjie Li, Dahua Lin, Yu Qiao, and Ziwei Liu. Vchitect-2.0: Parallel transformer for scaling up video diffusion models. arXiv preprint arXiv:2501.08453, 2025.

[72] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. LTX-Video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2024.

## A Appendix

## A.1 4LC Details

Visual encoders. Source observations are resized and center-cropped to 560 × 336 in the coordinates of the geometry, and each encoder follows the settings in Table 6. FLUX.2 VAE [45], DINOv2 [61], and VGGT [60] produce one feature map per keyframe, so $\alpha _ { r } ( k ) = k$ in Equation 1. V-JEPA 2.1 [46] produces one feature step per two frames, so we pair each keyframe with its predecessor, repeating the first keyframe, to obtain one step per keyframe. Wan2.1 VAE [43] encodes the first frame separately and compresses later frames by four; we fill each frame with the most recent keyframe and associate keyframe k at clip index $n _ { k }$ with step $\lceil n _ { k } / 4 \rceil$

Motion filtering. The motion mask $b _ { i }$ in Sec. 3.2 marks a unit as dynamic when its estimated motion displaces its projection in the source camera $\Gamma _ { k _ { i } }$ by more than ϵ pixels per frame toward either adjacent observation:

$$
b _ { i } = \mathbf { 1 } \left[ \operatorname* { m a x } _ { \pm } \frac { \left\| \pi _ { \Gamma _ { k _ { i } } } \left( \mathbf { x } _ { i } + \Delta t _ { i } ^ { \pm } \mathbf { v } _ { i } ^ { \pm } \right) - \mathbf { p } _ { i } \right\| _ { 2 } } { \Delta n _ { i } ^ { \pm } } > \epsilon \right] ,\tag{6}
$$

where $\pi _ { \Gamma }$ denotes pixel projection through camera Γ, and $\Delta t _ { i } ^ { \pm }$ and $\Delta n _ { i } ^ { \pm }$ are the time and frame gaps to the adjacen observations. When motion is estimated from multiple observations, units with motion toward neither neighbor are excluded from readout.

Camera projection. Each unit is assigned to the feature cell nearest to its projection in the target camera. At each cell, dynamic units from the source time closest to the requested time take priority, and static units fill the remaining cells (Sec. 3.2). Among the remaining candidates, we prefer the source camera closest to the target camera, then the frontmost surface, and finally the unit whose projections lie closest to the source and target cell centers.

## A.2 4LC-Video Details

Condition adapter. Wan2.1 latents encode the first frame alone and each later group of four frames, so readouts with target intervals of one, two, and four frames contribute four, two, and one readouts to each later latent step; for 81 frames, they give 81, 41, and 21 readouts over 21 latent steps. Target-camera rays use Plucker coordinates [¨ 62].

Training. The 200K training clips comprise 100K from SpatialVID [47], 40K from RealCam-Vid [48], and 60K from OpenVid [49]. We train the condition adapter and control branch for 20K steps on 16 NVIDIA A100 GPUs with a globa batch size of 16, using AdamW [63] at a learning rate of $1 0 ^ { - 5 }$

Data augmentation. Inspired by NeoVerse [10], which simulates degraded conditions from monocular videos, training augmentations vary the available conditions while preserving the target video, times, and cameras.

Each training clip provides K uniformly spaced keyframes, with K drawn from 11 to 21. To expose the generator to incomplete coverage, we use two geometric augmentations. Source-frame dropout excludes selected frames from the readout

Table 6. Visual encoder settings. $d _ { r }$ is the spatial stride in pixels, $D _ { r }$ the feature dimension, and the target interval the number of frames between readouts.
<table><tr><td>Encoder</td><td>Features</td><td> $d _ { r }$ </td><td> $D _ { r }$ </td><td>Target interval</td></tr><tr><td>FLUX.2 VAE</td><td>Posterior mean of RGB and depth</td><td>8</td><td>32</td><td>1</td></tr><tr><td>DINOv2-L</td><td>Patch tokens</td><td>14</td><td>1024</td><td>1</td></tr><tr><td>VGGT-1B</td><td>Aggregator layer-11 tokens</td><td>14</td><td>2048</td><td>1</td></tr><tr><td>V-JEPA 2.1 ViT-L</td><td>Patch tokens</td><td>16</td><td>1024</td><td>2</td></tr><tr><td>Wan2.1 VAE</td><td>Latents of RGB and depth</td><td>8</td><td>16</td><td>4</td></tr></table>

Table 7. Training augmentation parameters. Geometric augmentation applies outside first-frame sampling. Lateral translation is in world units; focal-length factors are relative to the source camera.
<table><tr><td>Type</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="6">Geometric</td><td>Number of keyframes K</td><td>11-21</td></tr><tr><td>Augmentation probability</td><td>0.6</td></tr><tr><td>Maximum source-frame drop fraction</td><td>0.3</td></tr><tr><td>Lateral translation</td><td>0.1-0.3</td></tr><tr><td>Yaw and pitch</td><td>5°-25°</td></tr><tr><td>Zoom-in / zoom-out focal-length factor</td><td>1.1–1.3 / 0.7–0.9</td></tr><tr><td rowspan="2">Motion</td><td>First-frame sampling probability</td><td>0.1</td></tr><tr><td>Motion threshold € (pixels/frame)</td><td>0.05-0.2</td></tr><tr><td rowspan="3">Representation and text</td><td>Dropout probability, all representations</td><td>0.1</td></tr><tr><td>Dropout probability, per representation</td><td>0.1</td></tr><tr><td>Text-dropout probability</td><td>0.1</td></tr></table>

candidates. Visibility culling retains source units visible from a virtual camera near each source camera, which translates laterally while looking at the scene, yaws, pitches, zooms in, or zooms out.

To account for uncertainty in motion estimation, we perturb the motion threshold ϵ and construct training conditions with different motion masks. First-frame sampling restricts the observations to the initial image and constructs geometric conditions from its static state.

Representation and text dropout vary the visual and textual information available to the conditioning path. Representation dropout removes selected representations and recomputes coverage from the retained ones, and text dropout replaces the tex condition with the empty-prompt embedding. Table 7 lists the probabilities and ranges.

## A.3 Experimental Details

Spatiotemporal control. As in NeoVerse [10], test videos use K = 21 uniformly spaced keyframes, and its reconstructor estimates camera poses of generated videos. Each generated and target trajectory is expressed relative to its first frame. Camera trajectories are pan left, reversed, static, and dolly out at the original scene times; temporal trajectories are reversal, freezing, and pause-and-resume under the source cameras. Translation error excludes static-camera cases.

Long-term generation. Each MBench-A [58] case generates 401 frames. Historical conditions use up to eight uniformly sampled cache entries and their static units $( b _ { i } = 0 )$ . Composition visits them from earliest to latest for Environment cases and from latest to earliest for Causal cases, with a coverage-ratio threshold of 0.95.

Conditioning efficiency. Acquisition latency and peak GPU memory are measured on an NVIDIA A100, averaging 50 runs after three warmup runs. Scene construction differs widely across frameworks and cannot be fully aligned: the NeoVerse reconstructor used by NeoVerse and 4LC takes 1.35 s, whereas LiveWorld takes 20.43 s. For a fair comparison, acquisition and end-to-end timing exclude scene construction, including source reconstruction and encoding, for all frameworks; memory includes weights used for condition acquisition.

Feature readout analysis. For each video, request, and encoder, we uniformly sample 2,048 tokens from the region covered by both acquisition paths. Relative representations [59] use 128 corresponding anchor features with ten anchor seeds, keeping the statistical tokens fixed across seeds. We average relative cosine similarities over seeds within each case and give all cases equal weight in Fig. 5 (right). The five requests preserve the source time and camera sequence, reverse scene time, freeze scene time, hold the camera fixed, or reverse the camera trajectory.

Ablations. Each ablation configuration, including the RGB variants, trains its condition adapter and control branch independently on 10K clips for 4K steps on four GPUs with one sample per GPU, and samples with 20 steps.

![](images/e5ff8dc9ecd4aeb771f8121c6fa9152df38cd1ee05318dca766bd09219c6cf48.jpg)  
Figure 6. Qualitative comparison for long-term generation on MBench-A. Each method occupies two rows: the top row follows the outbound trajectory from left to right, and the bottom row follows the return trajectory from right to left, so vertically aligned frames share a camera pose. Red boxes mark structures that change between the first and final frames.

## A.4 Additional Results

Long-term generation. Fig. 6 complements the MBench-A results in Sec. 4.2 with two cases whose camera trajectory reverses midway and returns to the initial viewpoint. HY-WorldPlay [23] regenerates returned regions with shifted layouts and missing structures, and its final frame departs from the first. 4LC-Video adds each generated segment to the cache, and Composition reads out the static units of these entries for the return requests, so return views reproduce the outbound content. This consistency agrees with its highest Reprojection and Environment scores in Table 4.

Generalization to WorldScore. We evaluate the main model on WorldScore without fine-tuning, using the official pipeline, and compare it with published results [53] of static scene generators [64, 65, 66, 55] and foundation video models [67, 68, 69, 70, 71, 72, 43] (Table 8). 4LC-Video obtains the highest Dynamic score and the best content alignment and style consistency among these methods. Compared with Wan2.1, it raises camera controllability from 23.53 to 86.87 and 3D consistency from 78.74 to 87.97, and its Static score approaches those of static scene generators.

Table 8. Quantitative comparison on WorldScore [53]. We compare against its published leaderboard results and, following Spatia [5], report Avg. as the mean of the Static and Dynamic scores; missing motion metrics of static scene generators count as zero.
<table><tr><td rowspan="2">Method</td><td colspan="3">WorldScore</td><td colspan="3">Controllability</td><td colspan="4">Quality</td><td colspan="3">Dynamics</td></tr><tr><td>Avg.</td><td>Static</td><td>Dynamic</td><td>Camera</td><td>Object</td><td>Content</td><td>3D</td><td>Photo.</td><td>Style</td><td>Subj.</td><td>Acc.</td><td>Mag.</td><td>Smooth.</td></tr><tr><td colspan="10">Static scene generation</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>WonderJourney</td><td>54.19</td><td>63.75</td><td>44.63</td><td>84.60</td><td>37.10</td><td>35.54</td><td>80.60</td><td>79.03</td><td>62.82</td><td>66.56</td><td></td><td></td><td></td></tr><tr><td>InvisibleStitch</td><td>51.95</td><td>61.12</td><td>42.78</td><td>93.20</td><td>36.51</td><td>29.53</td><td>88.51</td><td>89.19</td><td>32.37</td><td>58.50</td><td></td><td></td><td></td></tr><tr><td>WonderWorld</td><td>61.79</td><td>72.69</td><td>50.88</td><td>92.98</td><td>51.76</td><td>71.25</td><td>86.87</td><td>85.56</td><td>70.57</td><td>49.81</td><td></td><td></td><td></td></tr><tr><td>Voyager</td><td>66.08</td><td>77.62</td><td>54.53</td><td>85.95</td><td>66.92</td><td>68.92</td><td>81.56</td><td>85.99</td><td>84.89</td><td>71.09</td><td></td><td></td><td></td></tr><tr><td colspan="10">Foundation video generation</td><td colspan="3"></td><td colspan="3"></td></tr><tr><td>VideoCrafter2</td><td>50.03</td><td>52.57</td><td>47.49</td><td>28.92</td><td>39.07</td><td>72.46</td><td>65.14</td><td>61.85</td><td>43.79</td><td>56.74</td><td>47.12</td><td>30.40</td><td>29.39</td></tr><tr><td>EasyAnimate</td><td>52.25</td><td>52.85</td><td>51.65</td><td>26.72</td><td>54.50</td><td>50.76</td><td>67.29</td><td>47.35</td><td>73.05</td><td>50.31</td><td>75.00</td><td>31.16</td><td>40.32</td></tr><tr><td>Allegro</td><td>53.64</td><td>55.31</td><td>51.97</td><td>24.84</td><td>57.47</td><td>51.48</td><td>70.50</td><td>69.89</td><td>65.60</td><td>47.41</td><td>54.39</td><td>40.28</td><td>37.81</td></tr><tr><td>CogVideoX-I2V</td><td>60.64</td><td>62.15</td><td>59.12</td><td>38.27</td><td>40.07</td><td>36.73</td><td>86.21</td><td>88.12</td><td>83.22</td><td>62.44</td><td>69.56</td><td>26.42</td><td>60.15</td></tr><tr><td>Vchitect-2.0</td><td>40.38</td><td>42.28</td><td>38.47</td><td>26.55</td><td>49.54</td><td>65.75</td><td>41.53</td><td>42.30</td><td>25.69</td><td>44.58</td><td>33.59</td><td>33.81</td><td>21.31</td></tr><tr><td>LTX-Video</td><td>55.99</td><td>55.44</td><td>56.54</td><td>25.06</td><td>53.41</td><td>39.73</td><td>78.41</td><td>88.92</td><td>53.50</td><td>49.08</td><td>76.22</td><td>29.95</td><td>71.09</td></tr><tr><td>Wan2.1</td><td>55.21</td><td>57.56</td><td>52.85</td><td>23.53</td><td>40.32</td><td>45.44</td><td>78.74</td><td>78.36</td><td>77.18</td><td>59.38</td><td>54.27</td><td>33.26</td><td>38.05</td></tr><tr><td>4LC-Video (ours)</td><td>68.46</td><td>72.18</td><td>64.73</td><td>86.87</td><td>41.04</td><td>76.84</td><td>87.97</td><td>68.02</td><td>86.41</td><td>58.10</td><td>55.66</td><td>35.78</td><td>50.65</td></tr></table>

Table 9. Motion filtering and threshold robustness. The main model is evaluated on the same 100 unseen videos with camera trajectories. ϵ is in pixels/frame; ϵ = 0 disables filtering, and ϵ = 0.1 is the default. Bold and red values mark the best and worst results, respectively.
<table><tr><td>Threshold €</td><td>rdist↓</td><td>tdist↓</td><td>Subj. Consist.</td><td>Back. Consist.</td><td>Temp. Flick.</td><td>Motion Smooth.</td><td>Aesth. Quality</td><td>Imag. Quality</td></tr><tr><td>0</td><td>0.065</td><td>0.138</td><td>89.91</td><td>93.53</td><td>94.44</td><td>97.88</td><td>49.08</td><td>67.07</td></tr><tr><td>0.05</td><td>0.035</td><td>0.092</td><td>90.61</td><td>94.06</td><td>94.81</td><td>98.03</td><td>48.50</td><td>66.88</td></tr><tr><td>0.1</td><td>0.037</td><td>0.089</td><td>90.71</td><td>94.14</td><td>95.47</td><td>98.06</td><td>48.16</td><td>65.49</td></tr><tr><td>0.2</td><td>0.035</td><td>0.083</td><td>90.76</td><td>94.24</td><td>94.88</td><td>98.04</td><td>48.15</td><td>66.61</td></tr><tr><td>0.4</td><td>0.036</td><td>0.083</td><td>90.74</td><td>94.23</td><td>94.88</td><td>98.04</td><td>48.13</td><td>66.61</td></tr></table>

Motion threshold. Using the main model on camera trajectories, we compare disabled motion filtering (Equation 6) with filtering at different thresholds (Table 9). Without filtering, estimated motion on static structure enters the conditions as spurious displacement, yielding the largest camera errors and the lowest subject consistency, background consistency, temporal flickering, and motion smoothness. Training samples the threshold from 0.05 to 0.2 to account for motion-estimation uncertainty (Table 7), and performance remains stable from 0.05 to 0.4, including values beyond this range, indicating robustness to the condition changes induced by threshold selection.