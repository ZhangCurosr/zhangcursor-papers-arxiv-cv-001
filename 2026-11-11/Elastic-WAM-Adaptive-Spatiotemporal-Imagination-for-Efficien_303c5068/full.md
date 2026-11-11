# Elastic-WAM: Adaptive Spatiotemporal Imagination for Efficient Robot Control

Haitian Li<sup>1,2,∗</sup> Tianfu Li<sup>2,3,∗</sup> Xuxi Zhao<sup>2,4</sup> Zizheng Liu<sup>2,5</sup> Yangcheng Yu<sup>6</sup> Yiwen Hou<sup>1,2</sup> Zeyu Xiao<sup>1</sup> Mutian Xu<sup>7</sup> Ziqian Yang<sup>2,8</sup> Haonan Chen<sup>1</sup> Lin Shao<sup>1,2,†</sup>

<sup>1</sup>National University of Singapore <sup>2</sup>RoboScience <sup>∗</sup>Equal contribution <sup>†</sup>Corresponding author

![](images/83036561ab9bbc85c51ff9eff1dae77c20dbd067545309ad4554eb5f4e3adf2d.jpg)  
Figure 1. Overview of Elastic-WAM. (a) Empirical observations reveal heterogeneous temporal refinement requirements across tasks and states, as well as varying spatial relevance to action generation. (b) So our unified spatiotemporal refinement field determines WHEN to stop world refinement, HOW FAR to advance denoising, and WHERE to refresh future video regions. (c) As a result, Elastic-WAM improves control performance and inference efficiency across simulation benchmarks and real-world long-horizon manipulation.

## Abstract

World-Action Models improve robot control by predicting future visual dynamics, but iterative video denoising incurs substantial inference overhead. Recent approaches reduce

Affiliations 3–8: <sup>3</sup>The Hong Kong University of Science and Technology (Guangzhou); <sup>4</sup>Zhejiang University; <sup>5</sup>Harbin Institute of Technology (Shenzhen); <sup>6</sup>Tsinghua University; <sup>7</sup>Nanyang Technological University; <sup>8</sup>Xi’an Jiaotong-Liverpool University.

this cost through partial future denoising, yet typically rely on fixed refinement budgets. Our empirical observations reveal that preferred refinement depths vary across tasks and states, while different future regions exhibit varying relevance to action generation. Motivated by these findings, we propose Elastic-WAM, which adaptively allocates world refinement computation across temporal denoising time and spatial future space. A unified spatiotemporal refinement demand field governs when to stop world refinement, how far to advance each denoising step, and which regions to refresh. We parameterize this field with a shared temporal basis and spatial amplitudes, and learn it through a three-stage action-grounded supervised procedure. At inference, the learned field guides recurrent sparse world refinement to improve task success while reducing inference overhead. Evaluations across diverse simulation benchmarks and real-world long-horizon manipulation demonstrate state-of-the-art control performance with reduced inference latency. Project page: https://lihaitian. com/Elastic-WAM

## 1. Introduction

World-Action Models (WAMs) have shown that predicting future visual dynamics can improve robot control [1, 3, 9, 31, 37]. However, iterative denoising of future videos incurs substantial inference overhead [2, 20]. Recent studies show that intermediate video representations can already support action prediction without complete denoising [19, 24, 25]. Explicit video generation can even be removed at inference while retaining future prediction as a training objective [40]. Nevertheless, these approaches typically adopt a fixed future refinement budget, regardless of the current task or execution state [10, 13]. Consequently, a fixed budget may waste computation in some states while limiting action quality in others [30].

To investigate this limitation, we examine how future video refinement affects downstream action quality across different tasks and states on 10 selected tasks from LIBERO-Plus benchmark [12]. As demonstrated in Fig. 1 (a.1), the number of denoising steps yielding the highest success rate varies from one to four. Even within the same task, different states favor different refinement depths, as shown in Fig. 1 (a.2). Beyond refinement depth, actionto-video attention maps reveal uneven attention across future video regions (shown in Fig. 1 (a.3)), motivating selective refinement of regions most relevant to action generation [38]. Together, these observations motivate adaptive allocation of world refinement computation across both temporal denoising time and spatial future video regions.

To this end, we propose Elastic-WAM, which models the demand for world refinement through a unified spatiotemporal field, as illustrated in Fig. 1 (b). Its total mass governs when to stop world refinement, its temporal profile determines how far to advance each denoising step, and its spatial variation identifies which future video regions to refresh. We parameterize this field with a shared temporal basis and spatial amplitudes that separate global refinement demand from relative regional priorities. We learn this field through three stages of action-grounded supervision, progressively optimizing the temporal profile, relative spatial priorities, and overall refinement demand. At inference, we start with a dense joint denoising step, use the learned field to guide subsequent sparse world refinement, and continue action denoising after world refinement stops.

We evaluate Elastic-WAM across in-distribution [21], out-of-distribution [12, 49], and more complex manipulation simulation benchmarks [4], as well as a real-world long-horizon tea preparation task. Our results demonstrate state-of-the-art control performance among methods without large-scale embodied pretraining, while reducing inference latency by 32.2% compared with Joint-WAM on RoboCasa-GR1, as shown in Fig. 1(c).

Our contributions are summarized as follows:

• We formulate adaptive world refinement through a unified spatiotemporal demand field, jointly governing when to stop, how far to advance, and where to refresh future video representations.

• We develop an action-grounded supervised learning framework and recurrent sparse inference strategy, enabling adaptive refinement with a frozen WAM backbone.

• We demonstrate state-of-the-art performance across simulation benchmarks and real-world manipulation tasks, while substantially reducing inference latency.

## 2. Related Work

World-Action Models. Vision-Language-Action (VLA) models have evolved from directly predicting robot actions [5, 6, 14, 15, 27, 35] to leveraging future visual predictions to guide action generation [8, 33, 34, 43, 45]. World-Action Models (WAMs) further extend the VLA paradigm by jointly modeling future visual dynamics and robot actions [3, 17, 31, 37]. To improve inference efficiency, recent approaches exploit partially denoised video representations [19, 24, 25], modality-aware step distillation [2], or compact video experts with sparse latents and asymmetric denoising [18]. Other methods eliminate explicit future generation at inference [40] or preserve future conditioning through sparse video–action interactions [47]. Complementary to these approaches, our Elastic-WAM adaptively allocates computation within the recurrent world refinement process of a frozen WAM, across both temporal denoising time and spatial future video regions.

Temporally Adaptive Denoising. Diffusion sampling has been accelerated through implicit samplers, efficient ODE solvers, and consistency distillation [22, 23, 28], typically using predefined schedules across inputs. Recent approaches further adapt denoising computation to individual inputs: AdaDiff learns sample-specific step budgets [41], while AdaptiveDiffusion skips redundant noise predictions based on intermediate latent changes [36]. For robotic control, D3P dynamically allocates action denoising steps according to state-dependent action importance [39], while SANTS learns adaptive stopping and noise progression for WAMs based on downstream action quality [30]. While these methods adapt computation along the denoising trajectory, Elastic-WAM jointly governs temporal refinement and spatial future token updates through a unified spatiotemporal refinement field.

![](images/3300ede1ffa13737415e459dcafd772a4217b105bea5455d23410f0b9c1b6d91.jpg)  
Figure 2. Pipeline of Elastic-WAM. Top: The inference pipeline starts with dense joint refinement, followed by adaptive sparse world refinement with state-dependent denoising progression and spatial token selection. Once the world is committed, action refinement con tinues using the cached world representations. Bottom: The unified spatiotemporal refinement field consists of (a) a temporal branch tha predicts a shared Beta distribution over denoising progression, (b) a spatial branch that estimates group-wise refinement amplitudes, and (c) a unified field that determines WHEN to commit the world, HOW FAR to advance denoising, and WHERE to refresh future

Spatially Adaptive Computation. Spatial redundancy in diffusion transformers has motivated methods that concentrate computation on selected tokens or attention interactions. VSA introduces trainable sparse attention by identifying important token interactions [42], while VORTA dynamically routes computation among sparse attention patterns [29]. Jenga combines dynamic attention carving with progressive resolution generation [44]. Beyond attention sparsification, DyDiT dynamically adjusts computation across denoising timesteps and spatial locations [46], while token-wise feature caching selectively reuses intermediate features across denoising steps [50]. Elastic-WAM learns to selectively refresh future video queries based on their contribution to downstream action prediction, preserving full logical attention context and coordinating spatial refinement with temporal decisions through a unified field.

## 3. Methodology

As illustrated in Fig. 2, Elastic-WAM allocates world refinement computation according to the remaining refinement demand at each state. We model this demand over denoising progression and future video space through a unified spatiotemporal field, where geometry-aware local curves describe the refinement needs of different regions (Sec. 3.1). Since future video regions share the same denoising trajectory, we use a common temporal profile while allowing each region to have a different refinement amplitude. Together, these components determine when to stop world refinement, how far to advance each denoising step, and which future video regions to refresh (Sec. 3.2).

With the pretrained WAM frozen, we learn the field through three stages of action-grounded supervision: first its temporal profile, then the relative refinement priorities of future video regions, and finally the overall demand for continued world refinement (Sec. 3.3). At inference, Elastic-WAM begins with a dense joint denoising step, recurrently refreshes selected future video queries while preserving the full logical attention context, and continues action denoising using the cached video representation after world refinement stops (Sec. 3.4).

## 3.1. Unified Spatiotemporal Refinement Field

We unify the temporal and spatial refinement demand within a single spatiotemporal field. To model spatially varying refinement demand, we partition the $N _ { F } ~ = ~ 1 9 6$ future video tokens into G = 64 geometry-aware groups that preserve temporal and camera-view boundaries. Each group g contains $n _ { g } \in \{ 1 , 2 , 4 \}$ tokens, with $\textstyle \sum _ { g = 1 } ^ { G } n _ { g } =$ $N _ { F }$ . For each group, we define a local refinement curve that describes how its remaining demand is distributed along the normalized denoising progression $z \in [ 0 , 1 ]$

$$
q _ { g } ( z \mid s ) = a _ { g } ( s ) \mathrm { B e t a } ( z ; m _ { g } ( s ) , c _ { g } ( s ) ) ,\tag{1}
$$

Here, $a _ { g } ( s )$ represents the remaining refinement demand of group g, while $\operatorname { B e t a } ( { \cdot } ; m _ { g } ( s ) , c _ { g } ( s ) )$ denotes a Beta density parameterized by its mode $m _ { g } ( s )$ and concentration $c _ { g } ( s )$

Since the Beta density is normalized, the total mass of each local refinement curve equals its amplitude $a _ { g } ( s )$ . We aggregate these local curves according to their respective token counts to obtain the global refinement demand:

$$
\begin{array} { c } { { { \displaystyle { \int _ { 0 } ^ { 1 } { q _ { g } ( z \mid s ) d z } } = a _ { g } ( s ) } , } } \\ { { { } } } \\ { { { Q ( z \mid s ) = \displaystyle { \sum _ { g = 1 } ^ { G } \frac { n _ { g } } { N _ { F } } q _ { g } ( z \mid s ) } } . } } \end{array}\tag{2}
$$

Together, the local curves $\{ q _ { g } ( z \mid s ) \} _ { g = 1 } ^ { G }$ form the unified spatiotemporal field, while $Q ( z \mid s )$ represents its spatially aggregated refinement demand. The total mass of $Q ( z \mid s )$ captures the overall remaining refinement demand, its distribution along z characterizes preferred denoising progression, and the local amplitudes $a _ { g } ( s )$ indicate how demand varies across future video regions.

## 3.2. Refinement Field Parameterization

While the general formulation allows each region to have an independent temporal profile, all future video tokens follow a shared denoising trajectory in the underlying WAM. We therefore parameterize each local refinement curve using a shared temporal basis $\textstyle B _ { 0 } ( z \mid s )$ and a region-specific amplitude $a _ { g } ( s )$

$$
\begin{array} { r l } & { q _ { g } ( z \mid s ) = a _ { g } ( s ) B _ { 0 } ( z \mid s ) , } \\ & { B _ { 0 } ( z \mid s ) = \mathrm { B e t a } ( z ; m _ { 0 } ( s ) , c _ { 0 } ( s ) ) . } \end{array}\tag{3}
$$

Substituting this parameterization into the global aggregation yields

$$
\begin{array} { c } { { Q ( z \mid s ) = D ( s ) B _ { 0 } ( z \mid s ) , } } \\ { { { } } } \\ { { D ( s ) = \displaystyle \sum _ { g = 1 } ^ { G } \frac { n _ { g } } { N _ { F } } a _ { g } ( s ) , } } \end{array}\tag{4}
$$

Since $B _ { 0 } ( z \mid s )$ is normalized, $D ( s )$ represents the total remaining refinement demand, while $B _ { 0 } ( z \mid s )$ describes its distribution along the progression of denoising.

To separate overall refinement demand from relative spatial priorities, we center the region-wise scores by their token-weighted mean:

$$
\begin{array} { c } { { \Delta u _ { g } ( s ) = u _ { g } ^ { \mathrm { r e l } } ( s ) - \displaystyle \sum _ { j = 1 } ^ { G } \frac { n _ { j } } { N _ { F } } u _ { j } ^ { \mathrm { r e l } } ( s ) , } } \\ { { { } } } \\ { { \displaystyle \sum _ { g = 1 } ^ { G } \frac { n _ { g } } { N _ { F } } \Delta u _ { g } ( s ) = 0 . } } \end{array}\tag{5}
$$

We then combine the relative spatial scores with a shared offset $b _ { 0 } ( s )$ to define each group’s refinement amplitude:

$$
a _ { g } ( s ) = \mathrm { s i g m o i d } ( b _ { 0 } ( s ) + \mathrm { s t o p g r a d } [ \Delta u _ { g } ( s ) ] ) .\tag{6}
$$

The shared offset $b _ { 0 } ( s )$ controls the overall refinement demand $D ( s )$ , while $\Delta u _ { g } ( s )$ determines the relative priority of each region. The stop-gradient operation prevents the learning of absolute demand from modifying the relative spatial scores.

WHEN. We interpret the global refinement demand $D ( s _ { k } )$ as the conditional continuation probability and construct the cumulative World Commit probability across refinement steps:

$$
\begin{array} { c } { { C _ { k } = D ( s _ { k } ) , } } \\ { { \displaystyle \Delta H _ { k } = - \log ( \mathrm { c l a m p } ( C _ { k } , \varepsilon , 1 ) ) , } } \\ { { \displaystyle F _ { k } = 1 - \exp \left( - \sum _ { j \le k } \Delta H _ { j } \right) } } \\ { { \displaystyle \quad = 1 - \prod _ { j \le k } C _ { j } . } } \end{array}\tag{7}
$$

We trigger World Commit when $F _ { k } \geq \eta ,$ with $\eta = 0 . 5$

HOW FAR. Given continuation, we normalize the global refinement field to obtain the temporal progression distribution and use its mode to determine the denoising step size:

$$
\begin{array} { r } { p ( z \mid s _ { k } , \mathrm { C O N T I N U E } ) = \cfrac { Q ( z \mid s _ { k } ) } { D ( s _ { k } ) } = B _ { 0 } ( z \mid s _ { k } ) , } \\ { z _ { k } = m _ { 0 } ( s _ { k } ) , \qquad } \\ { \Delta \sigma _ { k } = \delta _ { \operatorname* { m i n } } + z _ { k } ( \delta _ { \operatorname* { m a x } } - \delta _ { \operatorname* { m i n } } ) . } \end{array}\tag{8}
$$

We set $\delta _ { \mathrm { m i n } } = 0 . 0 5$ and $\delta _ { \mathrm { m a x } } = \mathrm { m i n } ( 0 . 2 0 , \sigma _ { k } - \sigma _ { \mathrm { f l o o r } } )$ to constrain the denoising progression within the admissible noise interval.

WHERE. Since the shared temporal basis and global offset do not affect spatial ordering, we select future video regions according to their relative refinement demand under an exact-token budget $K _ { 0 } i$

$$
\begin{array} { l } { { \displaystyle M _ { k } ^ { * } = \arg \operatorname* { m a x } _ { M \in \{ 0 , 1 \} ^ { G } } \sum _ { g = 1 } ^ { G } n _ { g } M _ { g } \Delta u _ { g } ( s _ { k } ) , } } \\ { { \mathrm { s . t . } \quad \displaystyle \sum _ { g = 1 } ^ { G } n _ { g } M _ { g } = K _ { 0 } . } } \end{array}\tag{9}
$$

We use an externally specified budget of $K _ { 0 } = 9 6$ tokens per sparse update by default, while the learned spatial scores determine which regions to refresh.

## 3.3. Action-Grounded Refinement Field Learning

With the pretrained WAM frozen, we learn the refinement field in three stages, successively optimizing the temporal parameters $( m _ { 0 } , c _ { 0 } )$ , the relative spatial scores $u _ { g } ^ { \mathrm { r e l } }$ , and the global offset $b _ { 0 }$

Temporal Profile (HOW FAR). To learn the temporal profile, we compare immediate World Commit with continuations of different denoising distances from the same recurrent state s. Each continuation performs one additional joint denoising step, followed by a forced World Commit and complete Action Finish. We evaluate the resulting action sequence y using a terminal action error:

$$
E ( s , y ) = 0 . 6 E _ { \mathrm { s e q } } ( s , y ) + 0 . 4 E _ { \Delta } ( s , y ) ,\tag{10}
$$

where $E _ { \mathrm { s e q } }$ measures action-sequence error and $E _ { \Delta }$ measures the error in consecutive action differences. We convert the candidate errors into a soft target over denoising distances, assigning higher probabilities to candidates with lower terminal errors, to supervise the temporal distribution $\textstyle B _ { 0 } ( z \mid s )$

The same comparisons also provide stopping supervision by evaluating immediate World Commit against the expected error of continuing under the learned temporal policy:

$$
\begin{array} { r l } & { E _ { \mathrm { c o n t i n u e } } ^ { \pi } ( s ) = \displaystyle \sum _ { \delta } P _ { \pi } ( \delta \mid s ) E _ { \delta } ( s ) , } \\ & { ~ A _ { \mathrm { s t o p } } ( s ) = E _ { \mathrm { c o n t i n u e } } ^ { \pi } ( s ) - E _ { \mathrm { c o m m i t } } ( s ) , } \\ & { ~ p _ { \mathrm { s t o p } } ^ { * } ( s ) = \mathrm { s i g m o i d } \displaystyle \bigg ( \frac { A _ { \mathrm { s t o p } } ( s ) } { T ( s ) } \bigg ) . } \end{array}\tag{11}
$$

Here, $E _ { \delta } ( s )$ and $E _ { \mathrm { c o m m i t } } ( s )$ denote the terminal action errors of continuation by $\delta$ and immediate World Commit, respectively, while $T ( s )$ controls the softness of the target. The resulting stopping supervision is used to train a temporal teacher, whose frozen hazard predictions guide the learning of overall refinement demand in the final stage.

Spatial Priorities (WHERE). With the temporal profile fixed, we learn the relative spatial scores $u _ { g } ^ { \mathrm { r e l } } ( s )$ by comparing dense and sparse future video updates at the same denoising state. Starting from a dense joint update at noise level $\sigma _ { 0 } ,$ , we obtain the recurrent context and use the frozen temporal policy to select the next noise level $\sigma _ { 1 } . \ \mathrm { A t } \ \sigma _ { 1 }$ , we compare a no-gradient dense teacher with a differentiable sparse student using the same teacher-forced state $x _ { \mathrm { c u r } } .$ . To optimize the discrete exact-budget mask, we use a straightthrough relaxation that preserves hard selection in the forward pass while allowing gradients to flow through its soft counterpart:

$$
\begin{array} { r l } & { M ^ { \mathrm { S T } } = M ^ { \mathrm { h a r d } } + M ^ { \mathrm { s o f t } } - \mathrm { s t o p g r a d } ( M ^ { \mathrm { s o f t } } ) , } \\ & { \mathcal { L } _ { \mathrm { w h e r e } } = \mathcal { L } _ { \mathrm { a c t i o n - G T } } + \mathcal { L } _ { \mathrm { a c t i o n - D e n s e } } } \\ & { ~ + 0 . 1 \mathcal { L } _ { \mathrm { v i d e o - G T } } + 0 . 0 1 \mathcal { L } _ { \mathrm { b u d g e t } } . } \end{array}\tag{12}
$$

The two action losses supervise the sparse student’s predicted action flow against ground-truth flow and the dense teacher, respectively, while the video and budget terms provide auxiliary flow supervision and soft-budget regularization. The WAM remains frozen while gradients propagate through its computations to optimize the spatial scores. $\mathbf { A f } -$ ter training, we freeze the relative spatial branch to preserve the learned region priorities.

Overall Demand (WHEN). With the temporal profile and relative spatial scores fixed, we learn the global offset $b _ { 0 } ( s )$ to control the overall refinement demand $D ( s )$ . We use hazard predictions from the frozen temporal teacher to construct target continuation masses $C _ { k } ^ { * }$ , which are mapped to the corresponding global offsets $b _ { 0 } ^ { * } ( s _ { k } )$ through the aggregated regional amplitudes:

$$
\begin{array} { l } { { \displaystyle C _ { k } ^ { * } = \exp \bigl ( { - \Delta H _ { k } ^ { \mathrm { t e a c h e r } } } \bigr ) \ : , } } \\ { { \displaystyle C _ { k } ^ { * } = \sum _ { g = 1 } ^ { G } \frac { n _ { g } } { N _ { F } } \mathrm { s i g m o i d } \bigl ( b _ { 0 } ^ { * } ( s _ { k } ) + \Delta u _ { g } ( s _ { k } ) \bigr ) \ : . } } \end{array}\tag{13}
$$

Since the aggregated regional amplitudes increase monotonically with $b _ { 0 } ^ { * } ( s _ { k } )$ , this constraint uniquely determines the target global offset.

Since World Commit depends on accumulated continuation probabilities, we train $b _ { 0 }$ using complete recurrent prefixes rather than isolated states. We optimize the following objective to supervise cumulative stopping behavior while retaining a terminal action-quality correction:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { s e q } } = \mathcal { L } _ { b _ { 0 } } + \mathcal { L } _ { \mathrm { c o n d i t i o n a l } } + \mathcal { L } _ { \mathrm { c u m u l a t i v e } } } \\ & { ~ + 0 . 5 \mathcal { L } _ { \mathrm { c o m m i t - t i m e } } , } \\ & { \mathcal { L } _ { \mathrm { f i n a l } } = \mathcal { L } _ { \mathrm { s e q } } + 0 . 1 \mathcal { L } _ { \mathrm { t e r m i n a l - C F } } . } \end{array}\tag{14}
$$

Here, $\mathcal { L } _ { b _ { 0 } }$ supervises the target global offset, $\mathcal { L } _ { \mathrm { c o n d i t i o n a l } }$ aligns the conditional continuation mass, and $\mathcal { L } _ { \mathrm { c u m u l a t i v e } }$ and $\mathcal { L } _ { \mathrm { { c o m m i t - t i m e } } }$ constrain cumulative stopping probabilities and commit timing across recurrent prefixes. Finally, $\mathcal { L } _ { \mathrm { t e r m i n a l - C F } }$ provides a low-weight correction based on terminal action quality at newly visited states, while the temporal profile and relative spatial branch remain frozen.

## 3.4. Refinement Field-Guided Sparse Inference

At each replanning cycle, Elastic-WAM begins with a dense joint denoising step over all future video tokens, establishing the recurrent state and layer-wise key–value cache. The learned refinement field then determines whether to continue world refinement and, if so, selects the next denoising progression and the future video regions to refresh (Sec. 3.2). All video and action latents are advanced to the next noise level, while the selected exact-token mask determines which future video queries are recomputed in the subsequent step. This process repeats with updated refinement decisions at each step.

During sparse joint updates, observation and action queries remain dense, while the selected future video queries retain access to the full logical key–value context. Unselected future video tokens preserve their cached hidden states, keys, values, and last-refresh velocities, which are reused as their latents continue to evolve. Once the cumulative World Commit threshold is reached or a safety condition is triggered, Elastic-WAM stops world refinement and directly reuses the current video representation and layerwise cache for subsequent action denoising. No additional video prefill is required after World Commit.

Table 1. LIBERO success rates (%). The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Spat.</td><td>Obj.</td><td>Goal</td><td>Long</td><td>Avg.</td></tr><tr><td>π0.5 [27]</td><td>98.8</td><td>98.2</td><td>98.0</td><td>92.4</td><td>96.9</td></tr><tr><td>LingBot-VA [19]</td><td>98.5</td><td>99.6</td><td>97.2</td><td>98.5</td><td>98.5</td></tr><tr><td>Motus [3]</td><td>96.8</td><td>99.8</td><td>96.6</td><td>97.6</td><td>97.7</td></tr><tr><td>Fast-WAM [40]</td><td>98.2</td><td>100.0</td><td>97.0</td><td>95.2</td><td>97.6</td></tr><tr><td>Joint-WAM [47]</td><td>99.6</td><td>99.4</td><td>98.2</td><td>96.8</td><td>98.5</td></tr><tr><td>Elastic-WAM (Ours)</td><td>99.4</td><td>99.6</td><td>98.0</td><td>98.2</td><td>98.8</td></tr></table>

Table 2. LIBERO-Pro success rates (%). The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Obj.</td><td> $\mathrm { P o s . }$ </td><td>Sem.</td><td>Task</td><td>Avg.</td></tr><tr><td>π0.5 [27]</td><td>96.0</td><td>20.8</td><td>95.8</td><td>0.8</td><td>53.4</td></tr><tr><td>OpenVLA [15]</td><td>93.0</td><td>0.0</td><td>97.3</td><td>0.0</td><td>47.6</td></tr><tr><td>X-VLA [48]</td><td>79.0</td><td>0.8</td><td>96.8</td><td>6.8</td><td>45.9</td></tr><tr><td>Fast-WAM [40]</td><td>78.0</td><td>1.0</td><td>83.2</td><td>15.8</td><td>44.5</td></tr><tr><td>Joint-WAM [47]</td><td>84.5</td><td>6.3</td><td>96.2</td><td>24.2</td><td>52.8</td></tr><tr><td>Elastic-WAM (Ours)</td><td>85.6</td><td>8.4</td><td>97.7</td><td>24.7</td><td>54.1</td></tr></table>

Table 3. LIBERO-Plus success rates (%). The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Cam.</td><td>Rob.</td><td>Lang.</td><td>Light</td><td>Backg.</td><td>Noise</td><td>Layout</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>UniVLA [7]</td><td>1.8</td><td>46.2</td><td>69.6</td><td>69.0</td><td>81.0</td><td>21.2</td><td>31.9</td><td>42.9</td></tr><tr><td>OpenVLA-OFT [16]</td><td>56.4</td><td>31.9</td><td>79.5</td><td>88.7</td><td>93.3</td><td>75.8</td><td>74.2</td><td>69.6</td></tr><tr><td>π0 [5]</td><td>13.8</td><td>6.0</td><td>58.8</td><td>85.0</td><td>81.4</td><td>79.0</td><td>68.9</td><td>53.6</td></tr><tr><td>π0-Fast [26]</td><td>65.1</td><td>21.6</td><td>61.0</td><td>73.2</td><td>73.2</td><td>74.4</td><td>68.8</td><td>61.6</td></tr><tr><td>WorldVLA [8]</td><td>0.1</td><td>27.9</td><td>41.6</td><td>43.7</td><td>17.1</td><td>10.9</td><td>38.0</td><td>25.0</td></tr><tr><td>Fast-WAM [40]</td><td>18.8</td><td>45.7</td><td>70.1</td><td>83.2</td><td>45.7</td><td>29.8</td><td>62.7</td><td>49.1</td></tr><tr><td>Joint-WAM [47]</td><td>37.5</td><td>64.5</td><td>93.0</td><td>95.0</td><td>55.9</td><td>47.3</td><td>79.6</td><td>66.3</td></tr><tr><td>LingBot-VA [19]</td><td>40.9</td><td>83.0</td><td>86.4</td><td>82.3</td><td>53.1</td><td>64.4</td><td>76.2</td><td>69.5</td></tr><tr><td>Elastic-WAM (Ours)</td><td>52.1</td><td>76.3</td><td>95.5</td><td>95.1</td><td>60.0</td><td>61.1</td><td>81.4</td><td>73.7</td></tr></table>

Table 4. RoboCasa-GR1 success rates (%) and inference latency (ms). The best and second-best success rates are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Latency ↓|</td><td>Avg.</td></tr><tr><td>π0.5 [27]</td><td>80.5</td><td>37.0</td></tr><tr><td>GR00T-N1.6 [32]</td><td>77.0</td><td>47.6</td></tr><tr><td>StarVLA [11]</td><td>82.0</td><td>48.8</td></tr><tr><td>Fast-WAM [40]</td><td>331.1</td><td>54.9</td></tr><tr><td>Joint-WAM [47]</td><td>572.6</td><td>55.0</td></tr><tr><td>Elastic-WAM (Ours)</td><td>388.2</td><td>57.4</td></tr></table>

## 4. Experiments

Our experiments address four key questions:

• Q1: How well does Elastic-WAM perform under increasingly challenging manipulation conditions?

• Q2: How does our Elastic-WAM balance control performance and inference cost?

• Q3: How does adaptive temporal and spatial refinement contribute to control performance?

• Q4: How robust is our Elastic-WAM in real-world longhorizon manipulation under lighting and object changes?

## 4.1. Experimental Setup

We evaluate Elastic-WAM on LIBERO [21] for standard manipulation, LIBERO-Plus [12] for robustness under visual and environmental perturbations, and LIBERO-Pro [49] for generalization across object, position, language, and task variations. We further evaluate complex manipulation performance and inference efficiency on RoboCasa-GR1 [4]. For real-world evaluation, we consider a four-stage bimanual tea preparation task under nominal conditions, lighting changes, and object replacement.

We compare Elastic-WAM against existing policies, including Joint-WAM [47] and Fast-WAM [40], while keeping the pretrained WAM frozen during refinement field learning. On LIBERO-Plus and LIBERO-Pro, the LIBERO-trained policy is evaluated without additional adaptation. We report task success rates across simulation benchmarks, inference latency on RoboCasa-GR1, and the full-task success rate (SR) and the average task progress (PS) in real-world experiments.

## 4.2. Comparison with State-of-the-Art Methods

In this section, we address Q1 and Q2 by evaluating control performance across four simulation benchmarks and inference efficiency on the most challenging benchmark, RoboCasa-GR1.

LIBERO. Elastic-WAM achieves the highest average success rate on LIBERO (Table 1), with notable improvements over Joint-WAM on the Long suite. These results demonstrate that adaptive refinement can maintain strong contro performance without uniformly refining the future.

LIBERO-Pro and LIBERO-Plus. On LIBERO-Pro (Table 2), Elastic-WAM achieves the highest average success rate, consistently outperforming Joint-WAM across object, position, semantic, and task variations. On LIBERO-Plus (Table 3), Elastic-WAM demonstrates superior robustness across all perturbation categories, with particularly notable gains under camera viewpoint changes, robot initialization changes, and sensor noise. These improvements suggest that adapting refinement depth and spatial allocation to the current state provides more effective future representations for action generation under diverse distribution shifts.

RoboCasa-GR1. Elastic-WAM achieves the highest success rate on RoboCasa-GR1 (Table 4), while reducing inference latency by 32.2% compared with Joint-WAM. These results highlight the benefit of selectively refining actionrelevant future representations, improving control in complex manipulation while avoiding redundant world updates.

## 4.3. Effectiveness of Adaptive Refinement

To address Q3, we investigate the contributions of spatial selection, adaptive temporal control, and token-refresh budget through ablation studies, complemented by a qualitative analysis of refinement behavior.

Spatial Refinement. On RoboCasa-GR1 (Table 5), learned spatial selection outperforms random selection under the same token budget, as well as all-active refinement with a larger budget and FirstKV, which reuses the initial video cache. These results demonstrate the benefit of selectively refreshing action-relevant future regions rather than relying on random allocation, broader refinement, or cached representations alone.

Temporal Refinement. On LIBERO-Plus (Table 6(a)), we progressively introduce adaptive World Commit (WHEN) and denoising progression (HOW FAR), while keeping learned spatial selection and the token budget fixed. Adaptive stopping improves upon the WHERE-only baseline, while further adapting denoising progression yields the best overall performance, demonstrating the complementary benefits of temporal adaptation.

Refinement Budget. We further examine the effect of different token-refresh budgets on LIBERO-Plus (Table 6(b)), while keeping adaptive temporal control fixed. The default budget K = 96 achieves the highest overall success rate, with smaller budgets remaining competitive and larger budgets yielding lower performance. These results indicate that increasing the number of refreshed future video tokens does not necessarily improve downstream control.

Qualitative Analysis. Figure 3 compares Elastic-WAM and Joint-WAM under noisy observations on LIBERO-Plus. Although both methods exhibit similar action-to-video attention patterns, Elastic-WAM selectively refreshes future video regions while retaining cached representations for others. This allows different regions to undergo different levels of refinement while preserving the full logical attention context for action generation. In the illustrated case, Elastic-WAM succeeds where Joint-WAM fails, suggesting that selective refinement can preserve useful action information while avoiding redundant world updates.

![](images/c668fd28f027b2b32658bc7f299cf8b7b502bfb87e07555263e008945f01f334.jpg)

Figure 3. Qualitative comparison between Joint-WAM (top) and Elastic-WAM (bottom). (a) Task outcomes. (b) Action-to-video attention maps. (c) Spatial distribution of future video token update counts. Elastic-WAM achieves successful control through selective refinement while preserving the full attention context.  
![](images/abaec2ce72186ee42a42c4a57e0787f29eb0b811fa2052a33f47842190626eb1.jpg)  
Figure 4. Real-world evaluation settings. We evaluate under three conditions: Nominal, Lighting OOD, and Object OOD. Lighting OOD introduces strong illumination changes, while Object OOD replaces the white cup with a silver-colored cup of a slightly different shape.

## 4.4. Real-World Long-Horizon Manipulation

To address Q4, we conduct real-world experiments on a bimanual robotic platform equipped with two UFACTORY xArm 7 manipulators.

Setup. We evaluate a four-stage tea preparation task involving cup placement, tea pouring, and object return. As shown in Fig. 4, we consider three evaluation conditions: nominal, lighting Out-of-Distribution (OOD) with neon illumination, and object OOD with cup replacement. We compare Elastic-WAM against Joint-WAM and Fast-WAM, evaluating each method over 20 trials per condition. We report full-task success rate (SR) and progress score (PS), measuring complete task success and the average fraction of completed stages, respectively. Additional experimental details are provided in Appendix C.

Results. As shown in Table 7, Elastic-WAM achieves the highest full-task SR, while consistently attaining the best task PS across all evaluation settings. Under object OOD, although Elastic-WAM and Joint-WAM achieve comparable full-task success, Elastic-WAM completes more manipulation stages on average, revealing improvements that are not fully captured by the full-task success metric. In contrast, Fast-WAM fails to complete the entire task under any condition. These results demonstrate that Elastic-WAM enables more reliable long-horizon execution, even under challenging visual distribution shifts.

Table 5. Spatial selection and future refinement on RoboCasa-GR1 (success rates, %). The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Variant</td><td>Cab.</td><td>Drawer</td><td>Micro.</td><td>CutBoard</td><td>Placemat</td><td>Plate</td><td>Tray</td><td>Overall</td></tr><tr><td>Learned WHERE,  $K = 9 6$ </td><td>65.00</td><td>65.00</td><td>50.00</td><td>63.60</td><td>48.50</td><td>60.00</td><td>53.20</td><td>57.42</td></tr><tr><td>Random WHERE,  $K = 9 6$ </td><td>61.00</td><td>63.00</td><td>50.00</td><td>55.60</td><td>47.50</td><td>59.00</td><td>48.00</td><td>53.83</td></tr><tr><td>All-active,  $K = 1 2 8$ </td><td>61.00</td><td>60.00</td><td>45.00</td><td>60.40</td><td>50.00</td><td>59.50</td><td>50.40</td><td>55.17</td></tr><tr><td>FirstKV</td><td>62.00</td><td>54.00</td><td>53.00</td><td>59.60</td><td>46.00</td><td>58.50</td><td>50.80</td><td>54.50</td></tr></table>

<sup>\*</sup> The first three variants use adaptive temporal control. K counts refreshed future tokens after the initial dense joint denoising update. FirstKV follows the video update strategy of Faster-WAM [47], reusing the initial video KV cache without further refinement.

Table 6. Temporal control and token-refresh budget on LIBERO-Plus (success rates, %). The best and second-best results within each panel are highlighted in bold and underlined, respectively.
<table><tr><td>Variant</td><td>Cam.</td><td>Rob.</td><td>Lang.</td><td>Light</td><td>Backg.</td><td>Noise</td><td>Layout</td><td>Overall</td></tr><tr><td colspan="9">(a) Temporal control: learned WHERE, K = 96</td></tr><tr><td>WHERE-only</td><td>47.92</td><td>71.72</td><td>94.67</td><td>93.21</td><td>59.46</td><td>58.75</td><td>81.82</td><td>71.71</td></tr><tr><td>Fixed step</td><td>47.50</td><td>75.25</td><td>94.26</td><td>94.44</td><td>61.08</td><td>59.14</td><td>80.91</td><td>72.31</td></tr><tr><td>Adaptive</td><td>52.08</td><td>76.26</td><td>95.49</td><td>95.06</td><td>60.00</td><td>61.09</td><td>81.36</td><td>73.71</td></tr><tr><td colspan="9">(b) Token-refresh budget: adaptive temporal control</td></tr><tr><td>K = 49</td><td>48.75</td><td>73.23</td><td>94.26</td><td>96.30</td><td>62.16</td><td>59.14</td><td>82.73</td><td>72.84</td></tr><tr><td>K = 96</td><td>52.08</td><td>76.26</td><td>95.49</td><td>95.06</td><td>60.00</td><td>61.09</td><td>81.36</td><td>73.71</td></tr><tr><td>K = 147</td><td>46.67</td><td>68.69</td><td>94.67</td><td>94.44</td><td>60.54</td><td>59.92</td><td>82.73</td><td>71.71</td></tr><tr><td>K = 196</td><td>46.25</td><td>69.19</td><td>94.26</td><td>95.06</td><td>59.46</td><td>57.98</td><td>80.45</td><td>70.92</td></tr></table>

<sup>\*</sup> In (a), Fixed step uses adaptive commit with fixed progression; Adaptive uses both adaptive commit and progression. In (b), K counts refreshed future tokens after the initial dense joint denoising update. The Adaptive and K = 96 rows share the same configuration.

Table 7. Real-world long-horizon manipulation results. SR denotes full-task success rate (%), and PS denotes average task progress (%). The best results are highlighted in bold.
<table><tr><td></td><td colspan="2">Nominal</td><td colspan="2">Lighting OOD</td><td colspan="2">Object OOD</td></tr><tr><td>Method</td><td>SR</td><td>PS</td><td>SR</td><td>PS</td><td>SR</td><td>PS</td></tr><tr><td>Fast-WAM [40]</td><td>0.0</td><td>12.5</td><td>0.0</td><td>5.0</td><td>0.0</td><td>10.0</td></tr><tr><td>Joint-WAM [47]</td><td>30.0</td><td>37.5</td><td>30.0</td><td>35.0</td><td>30.0</td><td>35.0</td></tr><tr><td>Elastic-WAM (Ours)</td><td>50.0</td><td>57.5</td><td>50.0</td><td>52.5</td><td>30.0</td><td>47.5</td></tr></table>

Long-Horizon Analysis. Long-horizon manipulation requires policies to recognize task progress from evolving visual observations and generate actions appropriate to each execution stage. Without an explicit task-progress memory module, the evaluated policies must largely infer their current stage from visual information. We hypothesize that jointly adapting the temporal progression and spatial allocation of future refinement helps Elastic-WAM capture stagedependent visual changes more effectively, thereby facilitating task-state discrimination and stage-appropriate action generation. This interpretation is consistent with its superior performance on the LIBERO Long suite (Table 1) and higher task progress in real-world experiments (Table 7), suggesting that adaptive spatiotemporal refinement may also benefit long-horizon task execution.

## 5. Conclusion

In this paper, we presented Elastic-WAM, a unified framework for adaptively allocating world refinement computation across temporal denoising time and spatial future video space. By modeling the remaining refinement demand through a spatiotemporal field, Elastic-WAM jointly determines when to stop world refinement, how far to advance each denoising step, and which future video regions to refresh. We learn this field through three stages of action-grounded supervision and enable efficient inference through recurrent sparse refinement while preserving the full logical attention context. Extensive experiments across diverse simulation benchmarks and real-world long-horizon manipulation demonstrate state-ofthe-art control performance with reduced inference latency.

## References

[1] Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026. 2

[2] Arman Akbari, Ci Zhang, Arash Akbari, Lin Zhao, Yixiao Chen, Weiwei Chen, Xuan Zhang, Geng Yuan, and Yanzhi Wang. Flash-WAM: Modality-aware distillation for world action models. arXiv preprint arXiv:2606.05254, 2026. 2

[3] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 35101–35113, 2026. 2, 6

[4] Johan Bjorck, Fernando Castaneda, Nikita Cherniadev,˜ Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. GR00T-N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. 2, 6, 12

[5] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164, 2024. 2, 6

[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. 2

[7] Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. UniVLA: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025. 6

[8] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. WorldVLA: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025. 2, 6

[9] Haonan Chen, Sicheng Isabella Lu, Haitian Li, Jiajin Wu, Hexiang Wei, Jiawei Zhang, Xin Fei, Yiwen Hou, Chenrui Tie, Chongkai Gao, Sheng Wang, and Lin Shao. ObjectWorld: Scaling object-centric world models with a large-scale 3D full-object flow dataset. arXiv preprint arXiv:2611.11085, 2026. 2

[10] Wenbo Chen, Tianfu Li, Haoxuan Xu, Zhihao Cao, Zhenghan Chen, Zhengming Zhu, Zizhou Luo, Guosheng Yang, Yuan Liu, Lujia Wang, et al. MVG-WAM: Multiple view geometry-aware world-action modeling for robotic manipulation. arXiv preprint arXiv:2609.37793, 2026. 2

[11] StarVLA Community. StarVLA: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026. 6, 12

[12] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. LIBERO Plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025. 2, 6, 12

[13] Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, and Huaping Liu. Unified 4d world action modeling from video priors with asynchronous denoising. arXiv preprint arXiv:2604.26694, 2026. 2

[14] Zeyuan He, Bowen Yang, Zhirui Fang, Keru Zhou, Lei Jiang, Jingjing Qian, Fan Mo, Junchi Yan, Philip Torr, Xiu Li, et al. ELAN4D: Embodiment-centric 4d supervision for visionlanguage-action models via plug-and-play adaptation. arXiv preprint arXiv:2605.30484, 2026. 2

[15] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2, 6

[16] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and suc cess. arXiv preprint arXiv:2502.19645, 2025. 6

[17] Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming Yu Liu, Chelsea Finn, et al. Cosmos Policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026. 2

[18] Jiajun Li, Tiecheng Guo, Yifan Ye, Rongyu Zhang, Xiaowei Chi, Qianpu Sun, Ying Li, Yunfan Lou, Yan Huang, Zhihe Lu, et al. Efficient-WAM: A 1b-parameter world-action model with low-cost future imagination. arXiv preprint arXiv:2606.10040, 2026. 2

[19] Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026. 2, 6

[20] Tianfu Li, Haoxuan Xu, Wenbo Chen, Haitian Li, Changchuan Yang, Xinhu Zheng, Jun Ma, Yuan Liu, Lujia Wang, and Haoang Li. SLIP-VLA: Single-step latent imagination for policy learning in vision-language-action models. arXiv preprint arXiv:2609.33575, 2026. 2

[21] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023. 2, 6, 12

[22] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775– 5787, 2022. 2

[23] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent Consistency Models: Synthesizing high resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2

[24] Teli Ma, Jia Zheng, Zifan Wang, Chunli Jiang, Andy Cui, Junwei Liang, and Shuo Yang. Dit4Dit: Jointly modeling

video dynamics and actions for generalizable robot control. arXiv preprint arXiv:2603.10448, 2026. 2

[25] Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692, 2025. 2

[26] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025. 6

[27] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. $\pi _ { 0 . 5 } { : }$ a Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054, 2025. 2, 6

[28] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2

[29] Wenhao Sun, Rong-Cheng Tu, Yifu Ding, Jingyi Liao, Zhao Jin, Shunyu Liu, and Dacheng Tao. VORTA: Efficient video diffusion via routing sparse attention. Advances in Neural Information Processing Systems, 38:7837–7863, 2026. 3

[30] Yirui Sun, Guangyu Zhuge, Keliang Liu, Jie Gu, Shiqin Dai, Xinyu Bing, Zhongxue Gan, and Chunxu Tian. SANTS: A state-adaptive scheduler for world action models. arXiv preprint arXiv:2605.27947, 2026. 2

[31] MotuBrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, Jingrui Pang, Kiro Jing, et al. Motubrain: An advanced world action model for robot control. arXiv preprint arXiv:2604.27792, 2026. 2

[32] NVIDIA GEAR Team, N1 GR00T, et al. GR00T N1.6: An improved open foundation model for generalist humanoid robots, 2025. 6

[33] Xinkai Wang, Chenyi Wang, Yifu Xu, Mingzhe Ye, Fucheng Zhang, Jialin Tian, Xinyu Zhan, Lifeng Zhu, Cewu Lu, and Lixin Yang. LaMP: Learning vision-language-action policy with 3d scene flow as la tent m otion p rior. In European Conference on Computer Vision, pages 227–244. Springer, 2026. 2

[34] Yuran Wang, Siqiao Huang, Mingleyang Li, Chenhao Zhang, Jiaqi Liang, Weiyang Jin, Yue Chen, Xuemin Chi, Donghao Zhou, Qize Yu, Yu-Kai Wang, Yuhan Rui, Shenzhe Yao, Zhen Yuan, Zhenhao Shen, Kefei Zhu, Zijie Zhu, Ning Gao, Xiaowei Chi, Guanqi He, Shanghang Zhang, Hao Dong, Lin Shao, and Hang Zhao. OpenWAM: An open, modular exploration towards systematic world-action model pretrain ing. arXiv preprint arXiv:2609.07398, 2026. 2

[35] Haozhe Xie, Beichen Wen, Jiarui Zheng, Zhaoxi Chen, Fangzhou Hong, Haiwen Diao, and Ziwei Liu. DynamicVLA: A vision-language-action model for dynamic object manipulation. In Advances in Neural Information Processing Systems, 2026. 2

[36] Hancheng Ye, Jiakang Yuan, Renqiu Xia, Xiangchao Yan, Tao Chen, Junchi Yan, Botian Shi, and Bo Zhang. Trainingfree adaptive diffusion with bounded difference approxima-

tion strategy. Advances in Neural Information Processing Systems, 37:306–332, 2024. 2

[37] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. 2

[38] Hanyang Yu, Haitao Lin, Jingbo Zhang, Wenyao Zhang, Chenghao Gu, Heng Li, and Ping Tan. MaskWAM: Unify ing mask prompting and prediction for world-action models. arXiv preprint arXiv:2606.13515, 2026. 2

[39] Shu-Ang Yu, Feng Gao, Yi Wu, Chao Yu, and Yu Wang. D3P: Dynamic denoising diffusion policy via reinforcement learning. arXiv preprint arXiv:2508.06804, 2025. 2

[40] Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026. 2, 6, 8

[41] Hui Zhang, Zuxuan Wu, Zhen Xing, Jie Shao, and Yu-Gang Jiang. Adadiff: Adaptive step selection for fast diffusion models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 9914–9922, 2025. 2

[42] Peiyuan Zhang, Yongqi Chen, Haofeng Huang, Will Lin, Zhengzhong Liu, Ion Stoica, Eric Xing, and Hao Zhang. Faster video diffusion with trainable sparse attention. Ad vances in Neural Information Processing Systems, 38: 152509–152534, 2026. 3

[43] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, et al. DreamVLA: a visionlanguage-action model dreamed with comprehensive world knowledge. Advances in Neural Information Processing Systems, 38:24195–24228, 2026. 2

[44] Yuechen Zhang, Jinbo Xing, Shaoteng Liu, Bohao PENG, Xin Tao, Pengfei Wan, Eric Lo, Jiaya Jia, et al. Trainingfree efficient video generation via dynamic token carving. Advances in Neural Information Processing Systems, 38: 81913–81946, 2026. 3

[45] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. CoT-VLA: Visual chain-of-thought reasoning for vision-language-action models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1702–1713. IEEE, 2025. 2

[46] Wangbo Zhao, Yizeng Han, Jiasheng Tang, Kai Wang, Yibing Song, Gao Huang, Fan Wang, and Yang You. Dynamic diffusion transformer. In International Conference on Learn ing Representations, pages 65520–65552, 2025. 3

[47] Weiheng Zhao, Haoyi Jiang, Xin Shi, Liu Liu, Fan Huang, Zhizhong Su, Wei Sui, and Xinggang Wang. Faster-WAM: Efficient inference-time future conditioning for robust world action models. arXiv preprint arXiv:2608.04404, 2026. 2, 6, 8, 12

[48] Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-VLA: Soft-prompted transformer as scalable cross-embodiment vision-language-action model.

In International Conference on Learning Representations, pages 60580–60606, 2026. 6

[49] Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. LIBERO-Pro: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025. 2, 6, 12

[50] Chang Zou, Xuyang Liu, Ting Liu, Siteng Huang, and Linfeng Zhang. Accelerating diffusion transformers with tokenwise feature caching. arXiv preprint arXiv:2410.05317, 2024. 3

## A. Implementation Details

## A.1. Base World-Action Model

We build Elastic-WAM upon FastWAM-Joint, a pretrained world-action model that jointly refines future video representations and robot actions through iterative denoising. FastWAM-Joint is the same model referred to as Joint-WAM in Faster-WAM [47]; we use the former name to describe our backbone and the latter for baseline comparisons. The model follows the joint video-action denoising configuration used by the Joint-WAM baseline [47]. As illustrated in Fig. 2, language instructions are processed by a T5 encoder, visual observations are encoded through a VAE, and proprioceptive states are mapped into the model’s feature space through a projection layer. The joint denoising backbone processes observation, future-video, and action tokens, allowing future visual representations to provide conditioning for action generation. In our configuration, the future-video stream contains 196 tokens, which are organized into 64 geometry-aware groups for adaptive spatial refinement. Throughout the three-stage refinement field training, all pretrained WAM parameters remain frozen. Elastic-WAM preserves the underlying action generation mechanism while adaptively controlling the temporal and spatial computation devoted to future video refinement.

## A.2. Simulation Benchmarks

We evaluate Elastic-WAM across four simulation benchmarks covering standard manipulation, out-of-distribution generalization, and complex robotic interactions.

LIBERO. LIBERO [21] comprises four manipulation suites: Spatial, Object, Goal, and Long. Each suite contains 10 language-conditioned tasks, covering spatial reasoning, object manipulation, goal-conditioned execution, and extended action sequences. We evaluate each task over 50 rollouts, resulting in 2,000 evaluation episodes across all four suites. We report the success rate of each suite and the average performance across suites.

LIBERO-Plus. LIBERO-Plus [12] evaluates robustness under seven categories of distribution shifts: camera viewpoint, robot initialization, language instruction, illumination, background appearance, sensor noise, and object layout. We directly evaluate the LIBERO-trained policy without additional adaptation to these perturbations. We report success rates for individual perturbation categories and aggregate overall performance across evaluation episodes, accounting for the number of episodes in each category.

LIBERO-Pro. LIBERO-Pro [49] evaluates generalization across four categories: Object, Position, Semantic, and Task. These settings examine robustness to changes in manipulated objects, object positions, language semantics, and task configurations. As with LIBERO-Plus, we evaluate the LIBERO-trained policy without additional fine-tuning. We report category-wise success rates and their unweighted average.

RoboCasa-GR1. RoboCasa-GR1 [4] evaluates complex tabletop manipulation across 24 tasks involving object rearrangement and interactions with articulated objects. The benchmark covers diverse manipulation scenarios, including cabinets, drawers, microwaves, cutting boards, placemats, plates, and trays. We report task success rates and overall performance to assess the effectiveness of adaptive refinement in challenging manipulation settings. We follow the evaluation setting used by StarVLA [11].

## A.3. Inference Efficiency and Hardware

We measure inference latency on an NVIDIA H800 GPU using the RoboCasa-GR1 benchmark. We report inference latency in milliseconds alongside task success rates to evaluate the trade-off between control performance and computational efficiency. As reported in Table 4, Elastic-WAM reduces inference latency by 32.2% compared with Joint-WAM while achieving higher task success. This efficiency improvement is enabled by adaptive temporal refinement, selective future-video token updates, and cache reuse during action generation.

## B. Additional Simulation Experiments

## B.1. Precision-Sensitive Manipulation

We provide additional qualitative comparisons on the RoboCasa-GR1 to examine manipulation behaviors requiring precise object placement. As shown in Fig. 5, we compare Elastic-WAM with Fast-WAM on two representative tasks: placing a bottle into a cabinet and placing a can into a drawer.

In both examples, Fast-WAM moves the objects toward their target locations but fails to complete the final placement, whereas Elastic-WAM successfully executes the manipulation. These observations suggest that selectively continuing future refinement can provide more effective conditioning for precision-sensitive actions, where reaching the target vicinity alone is insufficient for successful task completion.

## C. Additional Real-World Experiments

## C.1. Robotic Platform and Data Collection

We conduct real-world experiments on a bimanual robotic platform equipped with two UFACTORY xArm 7 manipulators. The tabletop workspace contains two teapots and multiple cups, requiring coordinated manipulation of different objects throughout a long-horizon tea-preparation task. The evaluation involves successive grasping, placement, pouring, and object-return operations, with changes in object configurations across execution stages. We additionally record head-camera and wrist-camera views to visualize the robot’s execution behavior.

![](images/122afa0c01a70402050af5c1c630ab21c991daa1d30882dd45a771bc8efa9207.jpg)  
Figure 5. Precision-sensitive manipulation on RoboCasa-GR1. Qualitative comparisons between Fast-WAM (top) and Elastic-WAM (bottom) on (a) Bottle into Cabinet and (b) Can into Drawer. Keyframes progress from left to right, highlighting the final placement stage. Fast-WAM fails to complete the placement, whereas Elastic-WAM successfully finishes both tasks.

We collect 200 demonstration trajectories for the fourstage tea-preparation task. The underlying world-action policies are prepared following the standard FastWAM-Joint training procedure. For Elastic-WAM, the pretrained WAM backbone remains frozen during subsequent refinement field learning, as described in Sec. 3. This setup allows us to evaluate the effect of adaptive world refinement on long-horizon manipulation without modifying the underlying action generation architecture.

## C.2. Task and Evaluation Protocol

Long-Horizon Task. We design a four-stage teapreparation task requiring the robot to execute a sequence of coordinated bimanual operations. The complete language instruction is: “Place the white cup on top ofthe uncovered teapot, pour tea from the covered teapot into the white cup, return the white cup to its original position, then pour tea from the uncovered teapot into the black cup and return the teapot to its original position.”

The task consists of four sequential stages. In the first stage, the robot grasps the white cup and places it on the uncovered teapot. In the second stage, it manipulates the covered teapot to pour tea into the white cup. In the third stage, the robot retrieves the white cup and returns it to its original position. Finally, it pours tea from the uncovered teapot into the black cup and returns the teapot to its original position. Successful task execution requires completing all four stages in sequence without human intervention.

Evaluation Conditions. We evaluate Elastic-WAM, Joint-WAM, and Fast-WAM under three conditions: Nominal,

Lighting OOD, and Object OOD, as illustrated in Fig. 4. The Nominal condition uses the standard tabletop environment and object configuration. Lighting OOD introduces neon illumination to alter the visual appearance of the scene. Object OOD replaces the original white cup with a silver-colored cup of a slightly different shape throughout the manipulation sequence. These perturbations evaluate the robustness of long-horizon execution to changes in scene appearance and object characteristics.

Each method is evaluated over 20 trials per condition. All methods are required to complete the same four-stage task under the corresponding evaluation setting.

Evaluation Metrics. We report two complementary metrics: full-task success rate (SR) and progress score (PS). A trial is considered successful only when all four stages are completed. The full-task success rate is defined as the percentage of trials satisfying this criterion.

The progress score measures the average fraction of successfully completed stages across evaluation trials. Each stage contributes one point only when all operations within that stage are completed. For a four-stage task, the number of completed stages is divided by four and averaged across trials, with the resulting score expressed as a percentage. Unlike full-task success, PS captures partial task completion and provides a more detailed measure of execution progress when a policy fails before completing the entire sequence. The quantitative results are presented in Table 7.

## C.3. Qualitative Long-Horizon Execution

We provide additional qualitative results illustrating a successful long-horizon tea-preparation rollout with Elastic-WAM. Figure 6 presents representative keyframes from all four execution stages, including the corresponding wristcamera observations. The visualization captures the progression of the manipulation sequence from initial cup placement to the final pouring and object-return operations.

Throughout the rollout, the robot successfully performs successive grasping, placement, pouring, and object-return operations as the scene configuration evolves. Each completed stage changes the positions or states of task-relevant objects, requiring subsequent actions to be generated according to the updated visual observations. This example illustrates Elastic-WAM’s ability to sustain coordinated manipulation across multiple execution stages and complements the quantitative long-horizon results reported in Sec. 4.4.

HEAD CAMERA - ACTION PROGRESSION  
![](images/f74f845b1763d1d1722f8251df4ae7a810a89822510fa327e6c3339b0941145a.jpg)  
Figure 6. Qualitative visualization of real-world long-horizon manipulation. Keyframes illustrate a successful four-stage teapreparation rollout using Elastic-WAM. Stage 1: Grasping, lifting, aligning, and placing the white cup on the uncovered teapot. Stage 2: Grasping and manipulating the covered teapot to pour tea into the white cup. Stage 3: Retrieving and returning the white cup to its origina position. Stage 4: Pouring tea into the black cup and returning the uncovered teapot. The head-camera keyframes and corresponding wrist-camera views illustrate the progression of robot actions throughout the task.