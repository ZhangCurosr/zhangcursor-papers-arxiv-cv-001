# Beyond Visual Quality: Evaluating Physical Consistency under Ego-Motion with EGOGENEVAL

Yilin Long<sup>1,2</sup>, Chenming Zhu<sup>1,3</sup>, Zitang Gou<sup>1,2</sup>, Jingli Lin<sup>1,4</sup>, Tai Wang<sup>1,‡</sup>

<sup>1</sup>Shanghai AI Laboratory, <sup>2</sup>Fudan University, <sup>3</sup>The University of Hong Kong, <sup>4</sup>Shanghai Jiao Tong University

<sup>‡</sup>Corresponding author

Recent visual generators produce high-fidelity images yet often violate physical consistency under ego-motion, limiting their use for spatial reasoning and embodied planning. Existing benchmarks largely focus on isolated images or single-step quality, leaving this challenge underexplored. We introduce EgoGenEval, a geometry-grounded, pose-free benchmark designed to evaluate the physical consistency of visual generators under ego-motion, and organize our study into two parts. (1) Ego-GenEval contains 1,400 cases and 2,360 target views spanning single-step and multi-step ego-motion. It separately measures Camera Motion Grounding (CMG) and Scene State Preservation (SSP), with both metrics validated against blinded human judgments. Evaluating 16 pose-free generators together with two pose-conditioned references reveals that current models struggle to execute camera motion while maintaining scene state, and that no system performs well on both axes at once. (2) To examine whether benchmark-derived data can improve these capabilities, we build EgoGen-Train from the same geometry-grounded pipeline and run controlled SFT studies. These show that pairwise supervision does not reliably improve camera-motion grounding and scene-state preservation together: even at the full training pool and the longest budget, scene preservation gains a fraction of what camera motion does. This points to the pairwise teacher-forced objective itself as the binding constraint, motivating a trajectory-centric paradigm that couples self-conditioned rollouts with explicit pose and visibility supervision.

Code: https://github.com/InternRobotics/EgoGenEval

## 1 Introduction

Recent visual generators ofer increasingly photorealistic and controllable synthesis [1, 2], motivating their use as visual simulators for spatial reasoning and agentic visual imagination [3, 4]. For such uses, the generator should execute the instructed camera motion while preserving object presence, spatial layout, and appearance across views. However, a plausible output can drift toward an input view, mis-scale the requested motion, or lose objects during a rollout. We therefore ask: how reliably do pose-free visual generators ground natural-language camera motions while preserving scene state over short rollouts?

Existing benchmarks only partially address this question. They emphasize isolated-image composition, single image editing, explicit pose or trajectory control, or holistic world-model scores [2, 5–10]. No existing setting evaluates both capabilities in a unified pose-free rollout protocol while reporting motion realization and target-view scene preservation as separate outcomes. Table 1 summarizes this distinction.

To address this gap, we introduce EgoGenEval, comprising 1,400 cases and 2,360 target views from posed RGB-D indoor scenes. It spans four atomic motions, three-step chains, inverse cycles, and K=1–4 input view settings (Figure 1). Evaluated models receive only images and magnitude-specified natural-language instructions. We measure camera-motion fidelity with the Camera Motion Grounding Score (CMG-Score) and environment fidelity with the Scene State Preservation Score (SSP-Score), while reporting conventional image metrics only as auxiliary diagnostics. Both scores closely track aggregate rankings from blinded human evaluations across six representative systems.

Our evaluation of 16 pose-free systems, contextualized by two pose-conditioned references, reveals a consistent motion–state score gap. No evaluated system excels at both camera-motion grounding and scene-state preservation, and similar Overall scores can conceal markedly diferent CMG–SSP profiles. Within cameramotion grounding, directional compliance is comparatively tractable, yet most direction-correct outputs still miss the requested displacement by more than ±20%, so motion failure is dominated by scale rather than sign. Moreover, executing the camera motion correctly is not suficient to preserve scene state: systems can reach the right viewpoint yet still lose objects, distort layout, or corrupt appearance. The gap also widens with rollout: all 16 pose-free systems obtain lower SSP on Chain than on Atomic cases, with model-mean drops of 0.164 for SSP and 0.059 for CMG. Inverse cycles further expose return-action suppression after self-conditioning, with 66.3% of return steps remaining near-static, indicating that single-step accuracy does not carry over to multi-step consistency. Finally, for the seven multi-image systems, increasing context from K=1 to K=4 raises mean CMG from 0.519 to 0.582 while SSP decreases from 0.539 to 0.504, with outputs frequently attracted toward auxiliary views. Together, these results show that current generators do not reliably couple motion execution with persistent scene state, establishing the need for axis-specific, rollout-aware evaluation.

To test whether the benchmark can also guide model improvement, we follow the same construction principles to build EgoGen-Train, a scene-disjoint resource containing 66,214 trajectories and 108,213 teacher-forced edit pairs. As a diagnostic probe, pairwise SFT raises Qwen-Image-Edit’s Overall from 0.495 to 0.681, but the gains are highly asymmetric: +0.303 in CMG versus +0.069 in SSP. Matched controls further show that this transfer is backbone-dependent: on Qwen, SFT reliably improves CMG but yields no confirmed SSP gain, whereas on OmniGen2 the same recipe fails to improve CMG and even reduces SSP—so pairwise supervision does not transfer uniformly across models. Crucially, this asymmetry persists at the full training pool and the longest budget—the most data and training this resource provides—and even on the easiest single-step transitions, where a perfect previous frame precludes error accumulation; it is therefore the pairwise teacherforced objective, not a shortage of examples, that binds scene preservation. By making this objective-level bottleneck measurable, EgoGenEval motivates a shift from single-step pairs to full-trajectory supervision.

## Our contributions are as follows.

• We introduce EgoGenEval, a 1,400-case benchmark with 2,360 target views for evaluating physical consistency under ego-motion through atomic, chained, and inverse-cycle generation with controlled visual context.

• We decompose physical consistency into camera-motion grounding and target-view scene-state preservation, operationalized by CMG-Score and SSP-Score and validated against blinded human judgments.

• We benchmark 16 pose-free generators together with two pose-conditioned references, revealing a motion– state gap obscured by visual quality, one-step success, additional visual context, or a single aggregate score.

• Using EgoGen-Train, a scene-disjoint set built from the same pipeline, we run controlled SFT studies as a diagnostic probe: pairwise teacher-forced supervision yields asymmetric, backbone-dependent CMG–SSP responses and does not reliably improve both axes, with the asymmetry persisting at the full pool and longest budget—isolating the pairwise teacher-forced objective as the binding constraint—and motivating self-conditioned trajectory supervision.

## 2 Related Work

Image-generation and geometric-consistency benchmarks. Prior image-generation evaluation has largely focused on compositional prompt following, spatial relations, and single-output editing [5, 7, 10]. PDI-Bench extends this focus to generated videos by measuring projective-geometry residuals for scale–depth alignment, 3D motion consistency, and structural rigidity [11]. EgoGenEval complements these settings by jointly scoring language-instructed camera motion and target-view object state across short rollouts.

Novel-view synthesis and camera-controlled generation. Novel-view synthesis and 3D reconstruction typically assume calibrated multi-view observations with known or recoverable poses [12–14]. Related camera-controlled generation methods and world-model benchmarks likewise condition on or evaluate explicit camera trajectories [2, 8]. EgoGenEval instead evaluates pose-free generators driven only by magnitude-specified natural-language camera instructions.

![](images/f11646f532921400737f2166c6da5cbe3d48bb1d464db412b53a629fdb33fa6b.jpg)

![](images/2b6305e4b6da981092b99249cd823cbdc2db6c3aff6c65368dcb8f2096f204c9.jpg)

Figure 1 Overview of EgoGenEval. Atomic, Chain, and inverse-Cycle protocols evaluate magnitude-specified camera motions under controlled visual context (left). CMG and SSP separately measure camera-motion execution and targetview scene-state preservation using geometry- and object-centric components (center and bottom). Representative leaderboard results illustrate the distinct performance profiles induced by the two axes (right).
<table><tr><td>Evaluation setting</td><td>Ctx. NL cam. Multi State</td><td></td><td></td><td></td></tr><tr><td>T2I composition [5]</td><td>x</td><td>x</td><td>x</td><td>△</td></tr><tr><td>Spatial gen./editing [7, 10]</td><td>△</td><td>△</td><td>x</td><td>△</td></tr><tr><td>Pose-conditioned NVS/control [2]</td><td>√</td><td>x</td><td>△</td><td>△</td></tr><tr><td>Camera/world evaluation [8, 9]</td><td>√</td><td>△</td><td>√</td><td>△</td></tr><tr><td>EGOGENEVAL (ours)</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1 Scope comparison across six representative benchmarks grouped into four evaluation settings. Columns indicate support for visual context (Ctx.), natural-language camera control (NL cam.), multi-step evaluation (Multi), and explicit cross-view scene-state scoring (State). ✓/✗/△ denote full, absent, and partial coverage, respectively.

World models and embodied evaluation. World models use action-conditioned visual imagination for reasoning and control [3, 4, 15, 16]. Recent benchmarks evaluate multi-turn controllability, structural consistency, interaction dynamics, or simulator-grounded execution [9, 17–19]. Unlike these families, EgoGenEval evaluates pose-free, language-conditioned camera motion and target-view object-state preservation in static indoor scenes; Table 1 summarizes these diferences. Interaction dynamics and simulator task success remain out of scope.

![](images/a272b68215ed28a936d61020f3cc4b72f8d81b53ef483bd467d2d894aaf09531.jpg)  
Figure 2 Benchmark construction and composition. (a) The construction pipeline mines geometry-grounded view transitions, instantiates language-conditioned atomic/chain/cycle protocols, and applies quality filtering and quotabalanced sampling. (b) The benchmark covers four atomic, three chain, and two cycle subtypes, each allocating K=1, 2, 3, 4 cases in a 50:35:10:5 ratio.

## 3 EGOGENEVAL: Benchmark Design and Construction

## 3.1 Task, Context, and Motion Taxonomy

Task formulation. Given a current view $I _ { s } ,$ a natural-language camera-motion instruction T specifying the motion direction and an approximate numerical magnitude $( \mathrm { e . g . } , $ “move forward by 0.3 m”), and optional same-scene auxiliary views C, the model generates a target view $\hat { I } _ { t } \colon$

$$
( I _ { s } , C , T ) \ \to \ { \hat { I } } _ { t } .\tag{1}
$$

Visual context. We vary the total number of input images $K \in \{ 1 , 2 , 3 , 4 \}$ : K=1 uses $I _ { s }$ alone, whereas $K { > } 1$ adds K−1 same-scene auxiliary views. The current view is always placed first, and auxiliary views provide broader scene evidence without coinciding with any evaluated target.

Actions and protocols. We retain real-scene transitions dominated by four components: forward/backward translation (z), lateral translation (x), yaw, and pitch. Because real trajectories contain residual 6-DoF motion, instructions and human judgments consistently evaluate only the specified dominant component. We define three protocols: atomic $f _ { 0 }  f _ { 1 }$ , autoregressive three-step chain $f _ { 0 } {  } f _ { 1 } {  } f _ { 2 } {  } f _ { 3 }$ , and inverse cycle $f _ { 0 } {  } f _ { 1 } {  } f _ { 0 }$ Each transition is scored, yielding three targets per chain and two per cycle.

## 3.2 Benchmark Construction

EgoGenEval is constructed in three phases (Figure 2(a)). Phase 1 mines source–target view pairs with controlled gaps from ScanNet++, ScanNet, HyperSim, and Matterport3D [13, 14, 20, 21], computes sourceframe relative 6-DoF motion, and retains transitions with a clear dominant forward/backward translation, lateral translation, yaw, or pitch that satisfy overlap and visibility constraints. Phase 2 forms atomic, three-step chain, and inverse-cycle cases; each GT relative pose determines the motion, direction, and approximate magnitude, while a frozen overlap-, pose-, and novelty-based ranking selects K−1 auxiliary views to support target recovery, without using the ground-truth target views as input. Phase 3 combines z-bufer checks, VLM [22] and manual quality filtering, and quota-balanced sampling over subtype and context size, yielding 1,400 cases and 2,360 target views. Appendix B details the mining gates, task assembly, and quota-balanced selection of all three phases.

The final benchmark contains 1,400 cases from 771 scenes (at most seven per scene) and 2,360 scored target views; Figure 2(b) shows their subtype and context-size distribution. Each unique GT target is annotated once with Qwen3-VL [23]; the resulting labels are then frozen and unioned with a fixed 36-class indoor vocabulary. The same per-step prompt union is used for GT and generated-image detection and is never exposed to evaluated models.

## 4 Evaluation Setup and Metrics

We operationalize physical consistency in its most basic, static-scene form—geometric consistency under egomotion: whether a generated image realizes the requested camera motion while remaining a valid observation of the same, unchanged 3D environment. Dynamics and lighting are out of scope; this static form is a prerequisite that current generators already fail. We separate spatial fidelity into camera-motion fidelity, measured by CMG-Score, and environment fidelity, measured by SSP-Score. A realistic image may satisfy either requirement without satisfying the other.

## 4.1 Evaluation Setup

We evaluate 16 pose-free systems spanning image, multimodal, and video models. Pose-free describes only their interface: models receive images and a language instruction but no pose/depth; construction and scoring remain geometry-grounded. Two trajectory-conditioned world models serve as references. Each Chain/Cycle step is a fresh call under a shared wrapper: the first input is the anchor image at step 1 and the preceding output thereafter, followed by retained auxiliaries. For video models, the final frame is scored and becomes the next step’s first input. At K=4, three max-3 systems retain the two highest-overlap auxiliaries; single-reference systems use their native path. Full details are provided in Appendix C.

## 4.2 CMG: Is the Requested Motion Realized?

CMG measures whether the generated transition realizes the requested ego-motion. For each step m, DA3Nested-Giant-Large [24] estimates relative motion for both the physical GT transition and the evaluated transition. The latter is source-to-generation for the first step and previous-generation-to-current-generation thereafter.

Translation is calibrated per target transition, not per scene or model:

$$
\alpha _ { m } = \frac { \Vert \mathbf { t } _ { m } ^ { * } \Vert _ { 2 } } { \Vert \hat { \mathbf { t } } _ { m } ^ { \mathrm { G T } } \Vert _ { 2 } } , \qquad \hat { \mathbf { t } } _ { m } = \alpha _ { m } \hat { \mathbf { t } } _ { m } ^ { \mathrm { r a w } } .\tag{2}
$$

Each transition uses the same $\alpha _ { m }$ for all systems, leaving direction and rotation unchanged. Translation magnitude remains target-geometry-assisted, preserving under- or over-scaling only when the estimator scale is consistent across physical and generated pairs.

Let $a _ { m } ^ { * }$ be the signed GT value of the instructed dominant component, $\hat { a } _ { m }$ its estimate, and $d _ { m } = 1$ if their signs agree and 0 otherwise. We jointly score direction and magnitude as

$$
q _ { m } = \left( 1 + \frac { \left| \hat { a } _ { m } - a _ { m } ^ { * } \right| } { \operatorname* { m a x } ( \left| a _ { m } ^ { * } \right| , \epsilon _ { a } ) } \right) ^ { - 1 } .\tag{3}
$$

$$
\begin{array} { r } { g _ { m } = \frac { 1 } { 2 } d _ { m } ( 1 + q _ { m } ) . } \end{array}\tag{4}
$$

Since $d _ { m } , q _ { m }$ , and $g _ { m }$ lie in [0, 1] with higher values better, we directly average steps within a case:

$$
\mathrm { C M G } _ { c } = \frac { 1 } { T _ { c } } \sum _ { m = 1 } ^ { T _ { c } } g _ { m } .\tag{5}
$$

where $\epsilon _ { a } = 0 . 1$ m for translation and $5 ^ { \circ }$ for rotation. All instructed magnitudes are nonzero by construction (translation $\ge 0 . 1 5 \mathrm { m } .$ , yaw $\geq 8 ^ { \circ }$ , and pitch $\geq 6 ^ { \circ } )$ ; these floors stabilize relative error rather than define

direction, and an exactly zero prediction is direction-wrong. CMG isolates the instructed dominant component; supplementary full-pose diagnostics report unintended of-axis drift, which can lower SSP by displacing the viewpoint without necessarily corrupting the scene. Appendix H.6 reports these of-axis and full-pose errors.

## 4.3 SSP: Is the Environment Preserved?

SSP measures environment consistency at the intended target view. Grounding DINO [25] detects objects at box/text thresholds 0.25/0.25 using a fixed 36-class vocabulary augmented with frozen GT-target labels, with the same detection prompt for GT and generation. A Qwen3-VL [23] propose–demote–recover matcher with a DINOv3 [26] identity guard produces M one-to-one matches between G evaluable GT objects and $P$ filtered detections. Retention is $F _ { 1 } = 2 M / ( G + P )$ , and unmatched GT objects receive zero spatial and integrity credit. GT pose and depth only determine input-supported target objects and physical depth order. Thus, low SSP may reflect scene corruption, viewpoint error, or evaluator error.

For a target pair $( a , b )$ , the planar score measures whether its relative image-plane direction is preserved:

$$
O _ { a b } ^ { x y } = \left\{ \begin{array} { l l } { \operatorname* { m a x } \ ( 0 , \cos ( \mathbf { c } _ { b } ^ { * } - \mathbf { c } _ { a } ^ { * } , \hat { \mathbf { c } } _ { b } - \hat { \mathbf { c } } _ { a } ) ) , } & { a , b \mathrm { m a t c h e d , } } \\ { 0 , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{6}
$$

where $\mathbf { c } _ { i } ^ { * }$ and $\hat { \mathbf { c } } _ { i }$ are box centers and $\cos ( \mathbf { u } , \mathbf { v } ) = \mathbf { u } ^ { \top } \mathbf { v } / ( \| \mathbf { u } \| _ { 2 } \| \mathbf { v } \| _ { 2 } )$ . GT pairs with zero displacement are rejected by data validation; a generated displacement below $1 0 ^ { - 8 }$ receives zero.

The depth score checks whether the same object remains in front:

$$
O _ { a b } ^ { z } = \mathbb { I } \left[ \begin{array} { c } { a , b \mathrm { m a t c h e d ~ w i t h ~ v a l i d , n o n . t i e d ~ d e p t h s , } } \\ { \mathrm { s g n } ( \hat { z } _ { a } - \hat { z } _ { b } ) = \mathrm { s g n } ( z _ { a } ^ { * } - z _ { b } ^ { * } ) } \end{array} \right] .\tag{7}
$$

The score applies only to GT pairs with reliable depth separation and compares order rather than distance, so generated monocular depth need not be metric. A GT pair is depth-eligible only when its relative depth ratio exceeds 0.03.

Pairwise $O ^ { x y }$ and $O ^ { z }$ form topology, averaged with diagonal-normalized box-center accuracy to obtain S. Appearance integrity I weights identity, structure, edges, sharpness, color, and shape by (.30, .20, .15, .12, .13, .10). For multi-step cases, $A _ { c } ( x )$ equally weights the step mean and worst step. SSP then averages its three coverage-aware pillars:

$$
\mathrm { S S P } _ { c } = \textstyle { \frac { 1 } { 3 } } \left[ A _ { c } ( F _ { 1 } ) + A _ { c } ( S ) + A _ { c } ( { \mathbb Z } ) \right] .\tag{8}
$$

Exact matching, position normalization, depth eligibility, appearance weights, and thresholds are provided in Appendix D.2.

## 4.4 Aggregation and Auxiliary Image Metrics

Within each axis, case scores are normalized to [0, 1] with higher values better, so we average cases within Atomic, Chain, and Cycle and then macro-average the protocols. A model-independent GT-only mask removes 44 cases with no evaluable target objects at a required step, retaining 1,356/1,400 cases and 2,265/2,360 steps for every system. CMG and SSP retain this common normalized scale and direction, so Overall weights them equally:

$$
\mathrm { O v e r a l l } = { \frac { 1 } { 2 } } ( \mathrm { C M G } + \mathrm { S S P } ) .\tag{9}
$$

Auxiliary Reference Similarity (RefSim) aggregates PSNR [27] and LPIPS [28]; Visual Quality (VisQual) aggregates NIQE [29], MUSIQ [30], and CLIP-IQA [31]. Before averaging, each metric is min–max normalized across the fixed 18-system panel and oriented so higher is better. RefSim measures target-view alignment, whereas VisQual measures perceptual realism; both are excluded from Overall and ranking.

## 4.5 Metric Robustness and Human Validation

Human alignment. To confirm that CMG and SSP rank models reliably, three rubric-trained annotators independently rank six anonymized systems on 340 steps sampled across protocols and context sizes, scoring camera-motion realization and scene-state preservation separately. At the system level, both metrics align well with human judgment, closely tracking the aggregate human ranking (Spearman ρ=0.943, Kendall $\tau _ { b } { = } 0 . 8 6 7$ exact $\scriptstyle { p = 0 . 0 1 6 7 }$ . A qualitative comparison further shows that the VLM-assisted matcher resolves identity ambiguities and recovers correspondences a DINO-only baseline misses. These results validate the reliability of CMG and SSP as automatic measures of camera-motion grounding and scene-state preservation; full step-level statistics and rubrics are provided in Appendix E.

Evaluator robustness. To test for evaluator-dependency, we re-score the fixed 18-system panel after substituting each learned component of CMG and SSP and correlate every resulting ranking with the default. Replacing the DA3 relative-pose estimator with VGGT preserves the CMG ranking $_ { \left( \rho = 0 . 9 4 3 \right) }$ , and SSP is equally stable across depth backends $( \rho { = } 1 . 0 0 0 )$ , Grounding DINO box/text thresholds $\left( \rho { \geq } 0 . 9 2 9 \right)$ , detection vocabularies $\scriptstyle \left( \rho = 0 . 9 7 6 \right)$ , and mean-only temporal aggregation $\scriptstyle ( \rho = 0 . 9 9 4 )$ . The consistency of these rankings validates the robustness of the benchmark to the choice of evaluator, demonstrating that its relative results are not driven by any single backend or configuration; full ablations are provided in Appendices F and G.

## 5 Experimental Results

## 5.1 Main Results

Overall findings of EGOGENEVAL. Table 2 reports CMG, SSP, and auxiliary metrics for all 18 systems: no system is strong on both axes, the best pose-free Overall (0.662) trails the GT-target oracle (0.940), and scores fall further from Atomic to Chain and Cycle. The diagnostics below trace these gaps to failures in getting the motion magnitude right and keeping the scene intact, which visual quality does not reveal; Figure 4 shows representative cases, one per protocol, chosen to illustrate these failures rather than to enumerate every error type.

• Motion and scene fidelity are distinct axes: neither score implies the other. The CMG–SSP scatter in Figure 3(a) shows that a high score on one axis does not imply a high score on the other. HY-WorldMirror attains the highest CMG (0.847) yet only 0.481 SSP, below eight pose-free systems; conversely, FLUX.2-dev reaches 0.554 SSP while its CMG is only 0.448. The two axes are largely decoupled across the panel, so a single scalar average like Overall can hide which capability a system actually lacks—motion execution and scene preservation must be read separately.

• Magnitude failure is systematic, not random. Getting the direction of motion right is the comparatively tractable part of camera-motion grounding: direction accuracy reaches 0.663, well above the 0.525 obtained by always predicting the single most frequent direction, and ranking systems by direction accuracy alone almost perfectly reproduces their ranking by full CMG $( \rho = 0 . 9 9 7 )$ so what separates one system’s CMG from another’s is mostly direction, not magnitude. Magnitude, by contrast, is a weakness shared across the board: among direction-correct outputs, 59.5%–70.7% fall outside a ±20% window around the requested displacement, and this shortfall worsens monotonically as the requested motion grows (61.1%→68.9%→70.7% for small, medium, and large displacements; Figure 3(c)). Models thus learn the sign of motion but not its magnitude, uniformly and regardless of the scale requested. Privileged conditioning does not remove the error either—it only flips its sign: HY-WorldMirror attains the smallest angular error yet the largest full-translation error, from systematic over-scaling rather than under-execution.

• Multi-step consistency is not inherited from single-step execution. All 16 pose-free systems lose SSP from Atomic to Chain (mean drop 0.164), and the degradation spans all three SSP pillars with comparable declines for large furniture and small portable objects (0.195 vs. 0.191). Inverse-Cycle rollout reveals a complementary failure: within a single cycle, 66.3% of return steps are near-static, versus only 37.4% on the outbound step. Appendix H.3 defines the near-static criterion and reports the per-step response ratios and direction accuracies behind these rates. Because both are single-step actions of comparable dificulty, this near-doubling of the failure rate isolates self-conditioning, not single-step ability, as the cause: the return step must condition on a self-generated view, and that is where execution collapses. Rollout consistency is therefore a separate capability, not a downstream consequence of single-step quality, and motivates training the model on its own generated intermediate views, the inputs it must actually condition on at inference, rather than only on pairs of adjacent real frames.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Overall↑</td><td colspan="3">CMG↑</td><td colspan="4">SSP↑</td><td rowspan="2"></td><td rowspan="2">RefSim*↑ VisQual*↑</td></tr><tr><td>Avg Atomic Chain Cycle Avg Atomic Chain Cycle</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Pose-Free Track (Primary Evaluation)</td></tr><tr><td></td><td></td><td>Closed-Source Image Generators</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-Image-2 [2026]</td><td>0.66</td><td>0.72</td><td>0.74</td><td>0.72</td><td>0.71</td><td>0.60</td><td>0.66</td><td>0.55 0.60</td><td></td><td>0.59</td><td>0.90</td></tr><tr><td>Seedream-5.0 [2026] Gemini-3-Pro-Image [2025]</td><td>0.62</td><td>0.69</td><td>0.72</td><td>0.67</td><td>0.68</td><td>0.54</td><td>0.61</td><td>0.48</td><td>0.53</td><td>0.63</td><td>0.69</td></tr><tr><td></td><td>0.57</td><td>0.59</td><td>0.60</td><td>0.60</td><td>0.58</td><td>0.55</td><td>0.62</td><td>0.49</td><td>0.55</td><td>0.71</td><td>0.57</td></tr><tr><td colspan="10">Open-Source Image Generators</td></tr><tr><td>HiDream-O1 [2026]</td><td>0.52</td><td>0.48</td><td>0.57</td><td>0.44</td><td>0.44</td><td>0.56</td><td>0.61</td><td>0.50</td><td>0.57</td><td>0.60</td><td>0.08</td></tr><tr><td>FLUX.2-dev [2025]</td><td>0.50</td><td>0.45</td><td>0.56</td><td>0.36</td><td>0.42</td><td>0.55</td><td>0.61</td><td>0.49</td><td>0.56</td><td>0.60</td><td>0.68</td></tr><tr><td>HunyuanImage-3.0† [2026]</td><td>0.50</td><td>0.46</td><td>0.52</td><td>0.41</td><td>0.45</td><td>0.53</td><td>0.62</td><td>0.41</td><td>0.57</td><td>0.52</td><td>0.58</td></tr><tr><td>Qwen-Image-Edit-2511 [2025]</td><td>0.50</td><td>0.47</td><td>0.57</td><td>0.41</td><td>0.43</td><td>0.52</td><td>0.62</td><td>0.40</td><td>0.54</td><td>0.47</td><td>0.56</td></tr><tr><td>OmniGen2 [2026]</td><td>0.44</td><td>0.42</td><td>0.50</td><td>0.39</td><td>0.38</td><td>0.45</td><td>0.52</td><td>0.35</td><td>0.49</td><td>0.42</td><td>0.38</td></tr><tr><td>Step1X-Edit [2025]</td><td>0.44</td><td>0.30</td><td>0.32</td><td>0.31</td><td>0.27</td><td>0.57</td><td>0.60</td><td>0.49</td><td>0.63</td><td>0.91</td><td>0.35</td></tr><tr><td>ACE++ [2025]</td><td>0.40</td><td>0.39</td><td>0.41</td><td>0.38</td><td>0.38</td><td>0.41</td><td>0.44</td><td>0.30</td><td>0.48</td><td>0.23</td><td>0.19</td></tr><tr><td>FireRed-Image-Edit-1.1† [2026]</td><td>0.39</td><td>0.41</td><td>0.48</td><td>0.38</td><td>0.36</td><td>0.37</td><td>0.50</td><td>0.21</td><td>0.42</td><td>0.48</td><td>0.47</td></tr><tr><td>ICEdit [2026]</td><td>0.32</td><td>0.34</td><td>0.36</td><td>0.33</td><td>0.33</td><td>0.30</td><td>0.33</td><td>0.24</td><td>0.31</td><td>0.00</td><td>0.48</td></tr><tr><td colspan="10">Unified Multimodal Models</td></tr><tr><td>BAGEL-7B-MoT [2025]</td><td>0.45</td><td>0.44</td><td>0.46</td><td>0.43</td><td>0.41</td><td>0.47</td><td>0.55</td><td>0.39</td><td>0.46</td><td>0.57</td><td>0.32</td></tr><tr><td>Emu3.5-Image† [2025]</td><td>0.32</td><td>0.44</td><td>0.42</td><td>0.47</td><td>0.45</td><td>0.21</td><td>0.32</td><td>0.10</td><td>0.19</td><td>0.42</td><td>0.39</td></tr><tr><td colspan="10">Generic Video Models</td></tr><tr><td>Kling-2.1 [2025]</td><td>0.60</td><td>0.74</td><td>0.75</td><td>0.72</td><td>0.76</td><td>0.47</td><td>0.56</td><td>0.32</td><td>0.51</td><td>0.52</td><td>0.50</td></tr><tr><td>Seedance-1.0-Pro-Fast [2025]</td><td>0.53</td><td>0.61</td><td>0.61</td><td>0.57</td><td>0.65</td><td>0.45</td><td>0.54</td><td>0.36</td><td>0.44</td><td>0.42</td><td>0.50</td></tr><tr><td colspan="10">Pose-Conditioned Track (Reference Only)</td></tr><tr><td>HY-WorldMirror-2.0 [2026]</td><td></td><td>0.85</td><td></td><td></td><td></td><td></td><td>0.48</td><td>0.42</td><td>0.55</td><td>0.90</td><td></td></tr><tr><td>Lingbot-World [2026]</td><td>0.66 0.64</td><td>0.75</td><td>0.84 0.77</td><td>0.88 0.74</td><td>0.82 0.74</td><td>0.48 0.53</td><td>0.60</td><td>0.44</td><td>0.55</td><td>0.80</td><td>0.37 0.77</td></tr><tr><td colspan="10">Unranked Score Calibration</td></tr><tr><td>GT-target oracle</td><td>0.94</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.98</td><td>0.90</td><td>0.92</td><td>0.87</td><td>0.90</td><td></td><td></td></tr></table>

Table 2 Main leaderboard. Overall is the unweighted mean of the CMG and SSP protocol averages. <sup>∗</sup>RefSim and VisQual are normalized auxiliary aggregates and do not afect Overall or ranking. Pose-conditioned systems and calibration rows are reported as references only; the GT-target oracle is a practical evaluator ceiling rather than a mathematical upper bound. <sup>†</sup>These models support at most three total inputs; Appendix C describes their K=4 interface. Appendix J provides per-protocol breakdowns.

• Additional context views are imitated, not fused. For seven multi-image models, increasing context from K=1 to K=4 raises mean CMG from 0.519 to 0.582, while SSP decreases from 0.539 to 0.504 (Figure 3(b)). To rule out that this drop merely reflects a change in which objects are scored as K grows, we fix the evaluated object set to those objects present at every K and recompute SSP: the decline persists on this fixed set (∆ = −0.035 to −0.037). At K=4, 62.2% of outputs are CLIP-closest to an auxiliary view rather than the current target view, while only 0.6% are near-pixel copies; auxiliary-confused rows have systematically lower SSP (∆ = −0.064, 95% CI [−0.119, −0.009]). The additional views carry genuine evidence about the scene, yet current models cannot exploit it as such: rather than fusing these views into a more consistent target, they merely imitate them. More context is therefore not a monotone path to physical consistency.

(a) Motion–State Capability Profiles  
![](images/e36817639e39ec77cc599f93146ffdaf12e66c3fb168ee082dbfd584640801ef.jpg)  
(c) Direction-Correct Motions Are Under-Scaled

(b) More Views Improve CMG, Not SSP  
![](images/35e90c34bbdd42a560bebf63b5f4e52e12b907942f58396f08272c67ba6ff7e6.jpg)

![](images/9e69794be708d58373b502d223826d18bf2dc22a875ebcfcc78b583adadc4aa5.jpg)

(d) Object Size Affects Retention, Not Integrity  
![](images/baebb5a5f3273f735ba2c2fca12a89113ea71d86bd10e8ac50dfc86bdeaf6834.jpg)  
Figure 3 Capability profiles and diagnostic results. (a) CMG–SSP profiles separate motion execution from scene preservation across 18 systems. (b) For systems supporting all four context settings, additional views improve CMG on average but reduce SSP and increase feature-space attraction toward input views. (c) Directionally correct outputs commonly under-execute the requested magnitude. (d) Recall increases with projected object size, whereas integrity among successfully matched objects remains nearly constant. These diagnostics show why visual quality, direction, additional visual context, or matched-object integrity alone is insuficient to establish physical consistency.

• Perceptual quality is statistically decoupled from spatial consistency. Across the 14 image and editing systems with full quality signals, VisQual correlates with SSP at only $\rho = 0 . 2 9 2 \ ( p = 0 . 3 1 0 , \mathrm { n . s . } )$ and with CMG at $\rho = 0 . 5 8 2 \ ( p = 0 . 0 3 2 )$ . The most striking instance: Step1X-Edit holds the highest RefSim in the full panel (0.908) while recording the lowest pose-free CMG (0.298): its outputs are near-static, remaining visually close to the target yet failing to execute the requested action. A system that looks polished is statistically indistinguishable from one that is spatially incoherent. Checkpoint selection based on no-reference quality metrics will not surface magnitude under-execution or multi-step scene dissolution; the CMG–SSP frontier is the necessary signal.

Correlation with other benchmarks. Table 3 compares system rankings on EgoGenEval against a metaranking aggregated from four external editing benchmarks. The model rankings on EgoGenEval closely align with the meta-rankings, which validates EgoGenEval as a reliable measure of editing quality rather than an artifact of our protocol. At the same time, EgoGenEval diverges most from the spatial-editing benchmark closest to our setting, SpatialEdit. Existing benchmarks reward appearance preservation without asking whether the requested viewpoint was physically realized, so this divergence shows that EgoGenEval captures a capability the closest prior benchmark misses.

<table><tr><td>Model</td><td>GEdit Img v2</td><td>Edit</td><td>RISE</td><td>Sp. Edit</td><td>Meta Rank</td><td>Rank (Ours)</td></tr><tr><td>Qwen-Image-Edit-2511</td><td>1</td><td>1</td><td>1</td><td>2</td><td>1</td><td>1</td></tr><tr><td>Step1X-Edit</td><td>2</td><td>2</td><td>3</td><td>1</td><td>2</td><td>4</td></tr><tr><td>BAGEL</td><td>4</td><td>4</td><td>2</td><td>4</td><td>4</td><td>2</td></tr><tr><td>OmniGen2</td><td>3</td><td>3</td><td>=4</td><td>3</td><td>3</td><td>3</td></tr><tr><td>ICEdit</td><td>5</td><td>5</td><td>=4</td><td>5</td><td>5</td><td>5</td></tr></table>

Table 3 Rank comparison with four external benchmarks [10, 50–52]. Sp. Edit denotes SpatialEdit. Meta-Rank aggregates the four external rankings; Ours ranks models by EgoGenEval Overall, and “=” denotes tied ranks.

## 5.2 Supervised Fine-Tuning

To further examine the utility of EgoGenEval beyond evaluation, this section investigates whether its construction pipeline can produce a supervised fine-tuning (SFT) dataset, EgoGen-Train, for improving existing generators, from which we draw several observations.

EGOGEN-TRAIN construction. EgoGen-Train uses the same geometry-grounded pipeline as EgoGenEval, but it is scene-disjoint: when assembling EgoGen-Train, we exclude every scene used in EgoGenEval, so all reported gains are measured on scenes never seen during fine-tuning. Candidate trajectories first pass the same motion-dominance, overlap, visibility, depth-layer, and pose-separation gates as the benchmark. A curation stage then drops trajectories with ambiguous motion, unusable images, cross-scene sequences, dominant dynamic content, or no trackable spatial anchors, and a bidirectional depth-consistency check keeps only trajectories whose cross-view overlap stays within [0.35, 0.85] at every step. The resulting pool has 66,214 trajectories and 108,213 teacher-forced transition pairs. We form the pairs by taking each multi-step trajectory and splitting it into its adjacent frames $f _ { i - 1 } \to f _ { i }$ , then training each pair on its own with the ground-truth previous frame as context.

Training details. We fine-tune Qwen-Image-Edit-2511 with rank-16 LoRA on this pool. As a cross-backbone control, we also fine-tune OmniGen2 on the same pool against a matched base run. All variants are evaluated on the same 1,400 held-out cases. Appendix I reports the full provenance, budgets, and confidence intervals.

SFT results. Full SFT raises Qwen Overall from 0.495 to 0.681, above the best of-the-shelf pose-free model at 0.662, and Chain SSP rises from 0.397 to 0.505. But the two axes move very unevenly: CMG rises by +0.303 while SSP rises by only +0.069. The gap is not simply a matter of training longer (Figure 5a): at a 4k-update budget SSP has barely moved (+0.015), and tripling the budget to 13.5k is what lifts it to +0.069—while CMG has already gained +0.189 by 4k. Scene preservation does improve, but slowly and always far behind motion: fine-tuning readily teaches the model where to move, not how to hold the scene together while moving.

Cross-backbone transfer. The same recipe does not carry over to a second model. Comparing both backbones at a matched 4k-update budget on the same cleaned pool: on Qwen, fine-tuning improves CMG but not SSP; on OmniGen2 (Figure 5b), it does the opposite, gaining nothing on CMG while losing scene state across object retention, spatial relations, and appearance. On one model the transition-pair recipe helps motion, on another it hurts the scene. The direction of the CMG efect is thus backbone-dependent, but the two backbones agree on the axis that matters here: at matched compute, neither gains scene preservation.

The bottleneck is the objective, not the data. The asymmetry survives every more-favorable condition this resource allows. It holds at the full 108k-pair pool trained for the longest budget—the most data and training available—where SSP still trails CMG several times over. It holds when the data is made cleaner: at a matched 4k-update budget and matched pool size, quality-filtering the 40k pool raises CMG by +0.045 but SSP by only +0.003 over the unfiltered pool. And it holds on the easiest cases: on single-step Atomic transitions, where the model is handed a perfect previous frame and no error can accumulate, SSP reaches only 0.669 (a +0.051 gain) while the matched CMG gain is several times larger. When more data, cleaner data, and the (c) Inverse cycle (GPT-Image-2): a target-like return frame hides a wrong-sign return.

(a) Atomic yaw, five models: plausible frames hide five distinct failures.

Prompt: Rotate the camera mainly to the right by about 37.3 degrees, with only minor camera translation.

![](images/051ba7dadb23010e5212895dd97869e1d08f63c6d966b3a5dde698795c73984d.jpg)  
input view f₀

![](images/47f953cc43440114364faccd3219ef104e047ef9a673fc10d249cc1e8ff9d19e.jpg)  
reference — / 36.2°

![](images/7c6d49d8b1b05e61bb3451e25f6796c365dc3bf3c8bf8adef038ea33478989b2.jpg)  
under-rotation 24.5° / 36.2° 0.88 · 0.89

![](images/39f6aaf060a9042806f2d91620eb115141f4e8d20528fd2589815de4f9365dd1.jpg)  
half of sofa lost 37.0° / 36.2° 0.99 · 0.55

![](images/0fcb4c14f7d51d59c0ba2aefce5f9b113fb92d7284675d2f9536a2fe2a42b789.jpg)  
scene corrupted 40.6° / 36.2° 0.95 · 0.54

![](images/ae75951e7db9669a1b7b43fd9167e90ce5ba5418e7bc69e9bf79f2f87c5fab6d.jpg)  
wrong direction −2.1° / 36.2° 0.00 · 0.58

![](images/96c8a3a9cf7da6a63262b6fb3d3e0e5499c66580bed825ce7787c83583a2362c.jpg)  
near-static copy −0.2° / 36.2° 0.00 · 0.59

Prompt: Move the camera mainly to the right by about [0.69 / 0.72 / 0.68] meters, with only minor forward/backward shift and minor rotation.

(b) Three-step chain (Gemini-3-Pro): motion stays calibrated while the scene dissolves step by step.  
![](images/6cde7ec74bd1364844b3e94cb250c764624f0d7efc271acd024191be5a1c1358.jpg)  
input view f₀

![](images/a5bccc8dad214d6eae2c5a6ab4726349cf2496c3b8691f92a760561f0487c0e4.jpg)  
auxiliary context

![](images/8ec05d79ba45968807dd1e113a7cb1543eee96f4b57d7698bda6ee1805f9e0fa.jpg)  
under-translation +0.550 / +0.691 0.92 · 0.61

![](images/9e69b06e3c42c7c83821ab88664a1bdec7becf29c53491550cc223e3126e4964.jpg)  
round table gone +0.742 / +0.725 0.99 · 0.40

![](images/aec738018cdca2122266cd60c8eb214943e64b340218ef559ecdec41a02e62f6.jpg)  
sofa reoriented +0.717 / +0.680 0.97 · 0.15

Prompt: Move the camera mainly to the [left → right] by about 0.40 meters, with only minor forward/backward shift and minor rotation.

input / context

![](images/254ff2f9fe5d67a0d564f4d422f9936b3f1b4fc729ea2bf0ac322ba521ef7831.jpg)  
input view f₀

![](images/eebf07f1774ca1545a5d250efaa9b0efbbf58ff5107ea71ee5eba82da283062b.jpg)

![](images/71e06e5b6054c3cd81892d81890a70232ba0aa27bcbb2199df7b8443cc06ee25.jpg)  
correct outbound −0.362 / −0.401 0.96 · 0.61

![](images/f0c811a4ceca0450c7400162a5994373cacee832394ee2902a934faeb6b7c9ed.jpg)  
wrong-sign return −0.132 / +0.398 0.00 · 0.69

Figure 4 Representative failure cases. (a) One atomic yaw instruction given to five models: every frame is individually plausible, yet each fails diferently—under-rotation, object loss, visual corruption, wrong direction, and near-static copying. (b) A three-step chain (Gemini-3-Pro): after an under-translated first step the motion stays well calibrated (CMG ≥ 0.97) while the scene dissolves, SSP falling 0.61 → 0.40 → 0.15. (c) An inverse cycle (GPT-Image-2): a correct outbound step is followed by a return frame that resembles the target but moves the wrong way (−0.13 m against a target of +0.40 m, CMG 0.000). Under each frame we report the predicted and ground-truth motion along the instructed axis (degrees or metres), then CMG and SSP. Each prompt reproduces the motion clause verbatim and elides the shared scafold: every instruction additionally ends with the scene-preservation clause, and multi-image cases are further wrapped with the auxiliary-view preamble and the instruction not to copy any input image. Bracketed slots mark the single token that difers between the steps of a chain or cycle—the magnitude in (b), the direction word in (c). Appendix C.2 gives the complete template and all eight motion clauses.

easiest error-free version of the task all leave scene preservation this far behind, the ceiling is set by what the training signal rewards: pairwise next-view prediction scores getting the viewpoint right far more than keeping objects, their relations, and their appearance intact. We therefore read this as a limit of the pairwise SFT objective, not a call for a larger or better-curated dataset.

Toward better supervision. If the objective is the binding constraint, the remedy is to change what the objective supervises. Scene state can be supervised directly, using the pose, depth, visibility, and cross-step object correspondences that EgoGen-Train already carries but plain SFT ignores, so that preservation becomes an explicit target instead of something the model must infer from next-view prediction alone. Training on the model’s own multi-step rollouts, rather than always on ground-truth frames, would in addition build robustness to the errors that accumulate over a sequence.

(a) Motion outgains scene state in every protocol  
![](images/53536d143c2b581bb65c283474e510a8638c2a400d11fca4ad2fc7264677d8c6.jpg)

(b) Same recipe, opposite effect across backbones  
![](images/f9d667da6522837c0b2a475d41917d2bfaf511e5877e5212eebf553dcf4657d1.jpg)  
Figure 5 SFT efects on the two axes. (a) More training moves motion, not scene state. On the full cleaned pool, deltas relative to the Qwen base as the budget grows (base→ 4k → 13.5k updates): CMG climbs steeply (+0.189 → +0.303) while SSP crawls up from near zero $( + 0 . 0 1 5  + 0 . 0 6 9 )$ . This is the most data and training the resource provides, yet the axes stay far apart. (b) Same recipe, opposite effect across backbones. Paired deltas (∆CMG, ∆SSP) relative to each model’s base at a matched 4k-update budget on the same cleaned pool, under 20,000 scene-cluster bootstrap resamples: Qwen SFT gains CMG strongly but its SSP change is indistinguishable from zero, while OmniGen2 SFT gains nothing on CMG and loses SSP. Hatched bars mark deltas whose 95% interval includes zero. At matched compute, neither backbone gains scene preservation. Intervals quantify benchmark sampling uncertainty, not training-run variance.

## 6 Discussion and Limitations

A persistent-state bottleneck. Across the benchmark, current systems often generate the appearance of a camera action without maintaining the persistent scene state on which that action operates: direction can be predicted without metric control, auxiliary views are imitated rather than integrated, and a self-generated view is a weak anchor for the inverse action. Small, portable objects are also retained less often than large furniture. These observations do not establish a particular internal representation, but they localize the dificulty to coupling camera transformation with cross-view scene state, and the SFT study (Section 5.2) shows the same gap persists under pairwise supervision. Overall remains a useful summary, yet close scores should be read alongside CMG–SSP profiles and protocol-level uncertainty.

Limitations. EgoGenEval covers static indoor scenes, four camera-action families, and rollouts of up to three steps; dynamic scenes, wider motions, and longer horizons remain open. Because CMG and SSP are computed from learned perception models, a low score can reflect the limits of these evaluators rather than a true failure of the generator, even though both metrics track blinded human rankings. Appendix N details the statistical treatment, evaluator, licensing, interface, and contamination boundaries.

## 7 Conclusion

We introduced EgoGenEval, a 1,400-case benchmark of physical consistency under ego-motion, measuring camera-motion and target-view environment fidelity with metrics validated against aggregate human system rankings. Across 16 pose-free systems, similar Overall scores conceal distinct CMG–SSP profiles, scene state degrades consistently in Chain rollouts, and additional views improve motion grounding without improving preservation—together establishing EgoGenEval as an axis-specific, rollout-aware diagnostic for physically consistent generation under ego-motion. As a controlled study of whether benchmark-derived data can narrow this gap, we built EgoGen-Train from the same pipeline and ran matched SFT experiments across backbones. Pairwise teacher-forced supervision improves camera-motion grounding but does not reliably improve scene-state preservation, its efect does not transfer uniformly across backbones, and the asymmetry persists at the full pool and the longest budget and even on the easiest single-step cases—isolating the pairwise teacher-forced objective as the binding constraint. Together, the evaluation and training results identify persistent scene state as a central open challenge and motivate self-conditioned trajectory training with explicit pose and visibility supervision.

## References

[1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image difusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022

[2] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for video difusion models. In The Thirteenth International Conference on Learning Representations, 2025.

[3] Yuncong Yang, Jiageng Liu, Zheyuan Zhang, Siyuan Zhou, Reuben Tan, Jianwei Yang, Yilun Du, and Chuang Gan. Mindjourney: Test-time scaling with world models for spatial reasoning. Advances in Neural Information Processing Systems, 38:109855–109885, 2026.

[4] Chenming Zhu, Jingli Lin, Yilin Long, Peizhou Cao, Tai Wang, Jiangmiao Pang, and Xihui Liu. Thinking with imagination: Agentic visual spatial reasoning with world simulators, 2026. URL https://arxiv.org/abs/2606.06476.

[5] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

[6] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36: 78723–78747, 2023.

[7] Zengbin Wang, Xuecai Hu, Yong Wang, Feng Xiong, Man Zhang, and Xiangxiang Chu. Everything in its place: Benchmarking spatial intelligence of text-to-image models, 2026. URL https://arxiv.org/abs/2601.20354.

[8] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27713–27724, 2025.

[9] Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding. Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation, 2026. URL https://arxiv.org/abs/2605.25874.

[10] Yicheng Xiao, Wenhu Zhang, Lin Song, Yukang Chen, Wenbo Li, Nan Jiang, Tianhe Ren, Haokun Lin, Wei Huang, Haoyang Huang, Xiu Li, Nan Duan, and Xiaojuan Qi. Spatialedit: Benchmarking fine-grained image spatial editing, 2026. URL https://arxiv.org/abs/2604.04911.

[11] Jiaxin Wu, Yihao Pi, Yinling Zhang, Yuheng Li, and Xueyan Zou. Quantitative video world model evaluation for geometric-consistency, 2026. URL https://arxiv.org/abs/2605.15185.

[12] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911, 2021.

[13] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017.

[14] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niebner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. In 2017 International Conference on 3D Vision (3DV), pages 667–676. IEEE, 2017.

[15] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. Advances in neural information processing systems, 31, 2018.

[16] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, pages 1–7, 2025.

[17] Anurag Bagchi, Zhipeng Bao, Homanga Bharadhwaj, Yu-Xiong Wang, Pavel Tokmakov, and Martial Hebert. Walk through paintings: Egocentric world models from internet priors, 2026. URL https://arxiv.org/abs/2601.15284.

[18] Dayou Li, Lulin Liu, Bangya Liu, Shijie Zhou, Jiu Feng, Ziqi Lu, Minghui Zheng, Chenyu You, and Zhiwen Fan. Egocentric world model for photorealistic hand-object interaction synthesis, 2026. URL https://arxiv.org/abs/ 2603.13615.

[19] Zeyu Liu, Zhangzhe Zhu, Yang Zhang, Chenyou Fan, Chenjia Bai, and Xuelong Li. Kinebench: Benchmarking embodied world models via idm-free kinematic grounding, 2026. URL https://arxiv.org/abs/2607.19876.

[20] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023.

[21] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021.

[22] Google. Gemini 3.1 Pro Preview. https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview, 2026. Model ID: gemini-3.1-pro-preview; accessed July 27, 2026.

[23] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025. URL https://arxiv.org/abs/2511.21631.

[24] Haotong Lin, Sili Chen, Junhao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views, 2025. URL https://arxiv.org/abs/2511.10647.

[25] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.

[26] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothée Darcet, Théo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. Dinov3, 2025. URL https://arxiv.org/abs/2508.10104.

[27] Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pages 2366–2369. IEEE, 2010.

[28] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

[29] Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a “completely blind” image quality analyzer. IEEE Signal processing letters, 20(3):209–212, 2012.

[30] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021.

[31] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In AAAI, 2023.

[32] OpenAI. GPT Image 2. https://developers.openai.com/api/docs/models/gpt-image-2, 2026. Model ID: gptimage-2; accessed July 27, 2026.

[33] Volcano Engine. Seedream 4.0–5.0 prompt guide. https://www.volcengine.com/docs/82379/1829186, 2026. Model ID: doubao-seedream-5-0-260128; accessed July 27, 2026.

[34] Google DeepMind. Gemini 3 Pro Image model card. https://deepmind.google/models/gemini-image/pro/, 2025. Model ID: gemini-3-pro-image-preview; accessed July 27, 2026.

[35] Qi Cai, Jingwen Chen, Chengmin Gao, Zijian Gong, Yehao Li, Yingwei Pan, Yi Peng, Zhaofan Qiu, Kai Yu, Yiheng Zhang, Hao Ai, Siying Bai, Yang Chen, Zhihui Chen, Fengbin Gao, Ying Guo, Dong Li, Zhen Shen, Leilei Shi, Jing Wang, Siyu Wang, Yimeng Wang, Rui Zheng, Ting Yao, and Tao Mei. Hidream-o1-image: A natively unified image generative foundation model with pixel-level unified transformer, 2026. URL https://arxiv.org/abs/2605.11061.

[36] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

[37] Tencent Hunyuan Foundation Model Team. Hunyuanimage 3.0 technical report, 2026. URL https://arxiv.org/ abs/2509.23951.

[38] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. URL https://arxiv.org/abs/2508.02324.

[39] Chenyuan Wu, Jiahao Wang, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, et al. Omnigen2: Towards instruction-aligned multimodal generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21964–21975, 2026.

[40] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, Guopeng Li, Yuang Peng, Quan Sun, Jingwei Wu, Yan Cai, Zheng Ge, Ranchen Ming, Lei Xia, Xianfang Zeng, Yibo Zhu, Binxing Jiao, Xiangyu Zhang, Gang Yu, and Daxin Jiang. Step1x-edit: A practica framework for general image editing, 2025. URL https://arxiv.org/abs/2504.17761.

[41] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instructionbased image creation and editing via context-aware content filling. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1958–1966, 2025.

[42] Super Intelligence Team, Changhao Qiao, Chao Hui, Chen Li, Cunzheng Wang, Dejia Song, Jiale Zhang, Jing Li, Qiang Xiang, Runqi Wang, Shuang Sun, Wei Zhu, Xu Tang, Yao Hu, Yibo Chen, Yuhao Huang, Yuxuan Duan, Zhiyi Chen, and Ziyuan Guo. Firered-image-edit-1.0 technical report, 2026. URL https://arxiv.org/abs/2602.13344.

[43] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. Enabling instructional image editing with incontext generation in large scale difusion transformer. Advances in Neural Information Processing Systems, 38: 139195–139227, 2026.

[44] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining, 2025. URL https://arxiv.org/abs/2505.14683.

[45] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, Yueze Wang, Chengyuan Wang, Fan Zhang, Yingli Zhao, Ting Pan, Xianduo Li, Zecheng Hao, Wenxuan Ma, Zhuo Chen, Yulong Ao, Tiejun Huang, Zhongyuan Wang, and Xinlong Wang. Emu3. 5: Native multimodal models are world learners, 2025. URL https://arxiv.org/abs/2510.26583.

[46] Kuaishou Technology. Kling AI 2.1 model series. https://ir.kuaishou.com/news-releases/news-release-details/ kling-ai-celebrates-first-anniversary-achieves-annualized/, 2025. Model ID: kling-v2-1; accessed July 27, 2026.

[47] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, Xunsong Li, Yifu Li, Shanchuan Lin, Zhijie Lin, Jiawei Liu, Shu Liu, Xiaonan Nie, Zhiwu Qing, Yuxi Ren, Li Sun, Zhi Tian, Rui Wang, Sen Wang, Guoqiang Wei, Guohong Wu, Jie Wu, Ruiqi Xia, Fei Xiao, Xuefeng Xiao, Jiangqiao Yan, Ceyuan Yang, Jianchao Yang, Runkai Yang, Tao Yang, Yihang Yang, Zilyu Ye, Xuejiao Zeng, Yan Zeng, Heng Zhang, Yang Zhao, Xiaozheng Zheng, Peihao Zhu, Jiaxin Zou, and Feilong Zuo. Seedance 1.0: Exploring the boundaries of video generation models, 2025. URL https://arxiv.org/abs/2506.09113.

[48] Team HY-World, Chenjie Cao, Xuhui Zuo, Zhenwei Wang, Yisu Zhang, Junta Wu, Zhenyang Liu, Yuning Gong, Yang Liu, Bo Yuan, Chao Zhang, Coopers Li, Dongyuan Guo, Fan Yang, Haiyu Zhang, Hang Cao, Jianchen Zhu, Jiaxin Lin, Jie Xiao, Jihong Zhang, Junlin Yu, Lei Wang, Lifu Wang, Lilin Wang, Linus, Minghui Chen, Peng He, Penghao Zhao, Qi Chen, Rui Chen, Rui Shao, Sicong Liu, Wangchen Qin, Xiaochuan Niu, Xiang Yuan, Yi Sun, Yifei Tang, Yifu Sun, Yihang Lian, Yonghao Tan, Yuhong Liu, Yuyang Yin, Zhiyuan Min, Tengfei Wang, and Chunchao Guo. Hy-world 2.0: A multi-modal world model for reconstructing, generating, and simulating 3d worlds, 2026. URL https://arxiv.org/abs/2604.14268.

[49] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models, 2026. URL https://arxiv.org/abs/2601.20540.

[50] Zhangqi Jiang, Zheng Sun, Xianfang Zeng, Yufeng Yang, Xuanyang Zhang, Yongliang Wu, Wei Cheng, Gang Yu, Xu Yang, and Bihan Wen. Geditbench v2: A human-aligned benchmark for general image editing, 2026. URL https://arxiv.org/abs/2603.28547.

[51] Yang Ye, Xianyi He, Zongjian Li, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, Li Yuan, et al. Imgedit: A unified image editing dataset and benchmark. Advances in Neural Information Processing Systems, 38, 2026.

[52] Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. Advances in Neural Information Processing Systems, 38, 2026.

[53] Zhen Han, Zeyinzi Jiang, Yulin Pan, Jingfeng Zhang, Chaojie Mao, Chen-Wei Xie, Yu Liu, and Jingren Zhou. Ace: All-round creator and editor following instructions via difusion transformer. In The Thirteenth International Conference on Learning Representations, 2025.

[54] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.

[55] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, Xuanmao Li, Xingpeng Sun, Rohan Ashok, Aniruddha Mukherjee, Hao Kang, Xiangrui Kong, Gang Hua, Tianyi Zhang, Bedrich Benes, and Aniket Bera. DL3DV-10K: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024.

[56] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025.

## A Appendix Roadmap and Evaluation Scope

This appendix is included in the same arXiv document so that the complete benchmark contract can be read and cited together with the main results.

The primary evaluation comprises 16 pose-free generators; HY-WorldMirror-2.0 and Lingbot-World receive explicit 6-DoF trajectories and are reported separately as pose-conditioned references, yielding 18 evaluated systems in total. Unless stated otherwise, diagnostic conclusions use the 16-system primary scope, and pose-conditioned references are italicized in appendix tables rather than ranked.

## B Benchmark Construction

## B.1 Source data and candidate mining

EgoGenEval draws posed indoor RGB-D observations from ScanNet++ [20], ScanNet [13], HyperSim [21], and Matterport3D [14]. The final 1,400 cases contain 1,003, 210, 165, and 22 cases from these sources, respectively. These counts are the outcome of feasibility-constrained balanced selection rather than a target population estimate.

For a candidate transition from frame i to frame $j ,$ we compute the relative six-degree-of-freedom (6-DoF) transform in the source-camera frame,

$$
\Delta P _ { i  j } = ( t _ { x } , t _ { y } , t _ { z } , \psi , \theta , \phi ) ,\tag{10}
$$

where $t _ { x }$ and $t _ { z }$ represent lateral and forward/backward translation, while $\psi$ and θ represent yaw and pitch. Candidate mining removes near-duplicate views, excessive motion, low-overlap pairs, and transitions that fail motion-clarity, visibility, or pose-gap gates. Among legal components, the normalized largest component defines the dominant action. Residual translation, roll, or rotation may remain because the data come from real trajectories; they do not create an additional instruction class or scoring axis.

Translation and rotation components are normalized by fixed minimum-action gates, and the largest legal component determines the action label only after overlap, visibility, and motion-separation checks. Translation candidates span 0.15/0.18–1.20 m (source-dependent lower bound), yaw $8 ^ { \circ } { - } 4 5 ^ { \circ }$ , and pitch $6 ^ { \circ } - 3 5 ^ { \circ }$ . Candidate overlap lies in [0.35, 0.85], the dominant component must exceed competing components by at least 1.5, and translation–rotation separation must be at least 1.8. Atomic and Chain frame gaps are 2–50 and 2–25 frames; pose-NMS uses 0.30 m and $8 ^ { \circ }$ bins. The per-case metadata records the source-specific translation minimum and every gate outcome.

## B.2 Task assembly and instruction construction

Atomic cases contain one real current-to-target transition. Three-step chains use three valid consecutive transitions. C1 contains three translations from the same atomic family, C2 contains three rotations from the same atomic family, and C3 contains at least two atomic action families. Inverse cycles contain a planned two-step path

$$
f _ { 0 } \stackrel { a } {  } f _ { 1 } \stackrel { - a } { \longrightarrow } f _ { 0 } .\tag{11}
$$

The return action is computed from the inverse GT transform in the $f _ { 1 }$ camera frame. Y1 uses a translational outbound action and Y2 a rotational outbound action. The return is fixed before inference and is not adapted to the model’s realized outbound motion.

For every step, the GT relative pose determines the dominant action class, sign, and magnitude-specified value in meters or degrees. A deterministic template adds a preservation instruction covering scene identity, layout, relative depth, and perspective. The full pose, target image, depth, object labels, and detector prompts are never exposed to pose-free models.

The total input count is $K \in \{ 1 , 2 , 3 , 4 \}$ . The first image is the current view at step 1 and the previous generated output at later steps. The remaining $K - 1$ images are same-scene auxiliary views. They are ranked by target-visible support, overlap with the current/target views, pose distance, and view novelty; an evaluated target is never supplied as context. This target-aware selection is evaluator-side metadata used to choose evidence, not target-image access by the evaluated model.

## B.3 Filtering and quota-balanced selection

The final selection uses 36 subtype-by-K buckets. Each of the four atomic subtypes contributes 200 cases, while each of the three chain and two cycle subtypes contributes 120. Within every subtype, the $K = 1 , 2 , 3 ,$ 4 allocation follows 50:35:10:5, yielding 700, 490, 140, and 70 cases. Quota-constrained selection balances source datasets and limits scene concentration, producing 771 scenes with at most seven cases per scene. All selected cases pass z-bufer, VLM, and manual visual-quality checks. Figure 6 summarizes the resulting benchmark: the top row reports the dominant-component magnitude distribution for each atomic family, and the bottom row reports how cases divide across the four source datasets within every subtype. The source split reflects feasibility-constrained balanced selection rather than a source-population estimate.

## B.4 Examples and data separation

The planned release will provide versioned case manifests, construction metadata, evaluator contracts, and the inputs required to reproduce the reported aggregate results. For EgoGen-Train, it will additionally provide row-level trajectory metadata, deterministic teacher-forced pair conversion, and the scene-disjointness audit. It will not reproduce every intermediate candidate generated during early data mining, and restricted source pixels will remain governed by the licenses of the underlying datasets. The evaluation scenes are disjoint from the fine-tuning data used in Section I; pretraining exposure of the evaluated foundation models to the source datasets remains unknown.

Motion histograms use the benchmark-defined dominant component. Source bars describe the feasibility-constrained benchmark selection, not source-dataset population frequencies  
b
<table><tr><td>Protocol</td><td>Subtypes</td><td>Cases</td><td>Steps</td><td>K1</td><td>K2</td><td>K3</td><td>K4</td></tr><tr><td>Atomic</td><td>A1-A4</td><td>800</td><td>800</td><td>400</td><td>280</td><td>80</td><td>40</td></tr><tr><td>Chain</td><td>C1-C3</td><td>360</td><td>1080</td><td>180</td><td>126</td><td>36</td><td>18</td></tr><tr><td>Cycle</td><td>Y1-Y2</td><td>240</td><td>480</td><td>120</td><td>84</td><td>24</td><td>12</td></tr><tr><td>Total</td><td>9 subtypes</td><td>1400</td><td>2360</td><td>700</td><td>490</td><td>140</td><td>70</td></tr></table>

Table 4 Benchmark composition. K counts all model inputs, including the current or previous rollout view. Each case has one K setting in the main evaluation; the controlled Context-K study separately repeats matched cases across K.  
Frozen benchmark composition: 1,400 cases, 2,360 target steps

![](images/c7753c5d72cc9c1bae9ad049093687e50ffb8094c9c73268de16575af460ef77.jpg)

![](images/b92428b327661f1361365801103923cd662cfa78f7e4392cc909e1d96272419d.jpg)

![](images/e17ce57b0cae135f93b1e67d63e8122dbcc955ab07b7988610c6721ddbbf566a.jpg)

![](images/ea68966396c8edace0ccdc0cdc71c33de568e4eaeddee6a9503240b06fbe7508.jpg)

Case distribution by source dataset and benchmark subtype (Atomic 800 | Chain 360 | Cycle 240)  
![](images/42072d0c63a1c76a315903469e02c38953b05b776a8b7a05265679ecbe17f8b3.jpg)  
Figure 6 Frozen benchmark composition. (a) Per-family histograms of the benchmark-defined dominant component: forward/backward and lateral translation in meters, yaw and pitch rotation in degrees, each annotated with its step count and median. (b) Case distribution across the four source datasets within every Atomic, Chain, and Cycle subtype. The bars describe the feasibility-constrained balanced selection actually shipped, not source-dataset population frequencies.

## C Model Interfaces and Rollout Protocol

## C.1 Evaluation scope

Each transition is a fresh model call without conversation history. The rollout is nevertheless autoregressive: the generated output at step m − 1 replaces the current view at step m. Auxiliary views remain available according to the native interface, but the original $f _ { 0 }$ is not re-added on Cycle step 2. Consequently, a cycle cannot close

![](images/6448606cc3c18f0034858b1333916f2510b7ed0b09655a3ca6f93122b4a72f23.jpg)

![](images/01e0661622708530700369530feb7f1a51df06eabaeead40ceee2b5713606d88.jpg)

![](images/0735ae48d24eb496d700748028dda2cb32a822d2f298532010bf9c30632ca0cf.jpg)

![](images/7650328fc4f9c29de067d0e00d7b5f27f0bbc8c37c8aeec19129af1f7067d9ce.jpg)

## A1: Forward/Backward

## Prompt:

Move the camera mainly backward by about 1.06 meters, with only minor sideways shift and minor rotation. Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency.

## (Multi Contexts) Prompt:

Image 1 is the current view and should be used as the starting point. The other images are auxiliary context views of the same scene. Starting from Image 1, generate the new view after the following camera motion: Move the camera mainly backward by about 1.06 meters, with only minor sideways shift and minor rotation. Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency. Use the auxiliary images only to preserve scene identity and spatial layout. Do not simply copy any input image.

## A3: Yaw Rotation

## A3: Prompt:

Rotate the camera mainly to the left by about 24.3 degrees, with only minor sideways shift and minor rotation. Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency.

## (Multi Contexts) Prompt:

Image 1 is the current view and should be used as the starting point. The other images are auxiliary context views of the same scene. Starting from Image 1, generate the new view after the following camera motion: Rotate the camera mainly to the left by about 24.3 degrees, with only minor camera translation. Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency. Use the auxiliary images only to preserve scene identity and spatial layout. Do not simply copy any input image.

![](images/31614dce74a39ddfa5c076f1008aee769d3940acd8826c5c2799eed64d0c5493.jpg)

![](images/52b3b9c4247ec48496b38b8218f8d7d4849f5937478acc19cde7527b573eeb64.jpg)

![](images/4cefd574161676d70c82883e8093deafe381fcd1288e0377563499ac7d62b162.jpg)

![](images/7b2df7574768e9f8c01027efc5f37b2f6b482a195de78c19920d5d1d213ea0e2.jpg)

![](images/cddf8bfd9659c5c939287f5e85a5b9778ae09a493218e0a1fc7e0d69dfa3acb4.jpg)

## A2: Lateral / Strafe

## Prompt:

Move the camera mainly to the left by about 1.15 meters, with only minor sideways shift and minor rotation. Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency.

## (Multi Contexts) Prompt:

Image 1 is the current view and should be used as the starting point. The other images are auxiliary context views of the same scene. Starting from Image 1, generate the new view after the following camera motion: Move the camera mainly to the left by about 1.15 meters, with only minor forward/backward shift and minor rotation. Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency. Use the auxiliary images only to preserve scene identity and spatial layout. Do not simply copy any input image.

## A4: Pitch Rotation

## Prompt:

Tilt the camera mainly upward by about 16.3 degrees, with only minor sideways shift and minor rotation. Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency.

## (Multi Contexts) Prompt:

Image 1 is the current view and should be used as the starting point. The other images are auxiliary context views of the same scene. Starting from Image 1, generate the new view after the following camera motion: Tilt the camera mainly upward by about 16.3 degrees with only minor camera translation and minor yaw rotation. Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency. Use the auxiliary images only to preserve scene identity and spatial layout. Do not simply copy any input image.

Figure 7 Representative benchmark cases for forward/backward translation, lateral translation, yaw, and pitch. Each case contains the current view, optional same-scene context, and a magnitude-specified instruction. Target images and evaluator metadata are not shown to pose-free models.
<table><tr><td>Family</td><td>Systems</td></tr><tr><td>Closed image</td><td>GPT-Image-2, Seedream-5.0, Gemini-3-Pro-Image</td></tr><tr><td>Open gen./ edit</td><td>HiDream-O1, HunyuanImage-3.0, Qwen-Image-Edit-2511, FLUX.2-dev, OmniGen2, Step1X-Edit,</td></tr><tr><td>Unified</td><td>ACE++, FireRed-Image-Edit-1.1, ICEdit BAGEL-7B-MoT, Emu3.5-Image</td></tr><tr><td>Pose-free</td><td>Kling-2.1, Seedance-1.0-Pro-Fast</td></tr><tr><td>video</td><td>Pose-cond. ref. HY-WorldMirror-2.0, Lingbot-World</td></tr></table>

Table 5 Evaluated systems. The first four rows contain the 16-model Pose-Free Track; the final row contains two pose-conditioned references reported separately from the primary ranking.

by receiving the answer view as an extra reference. Every model–step contributes one frozen output; case bootstrap therefore quantifies benchmark-sampling uncertainty, not stochastic generation variance.

Multi-image models receive the current/previous view followed by retained auxiliaries. Step1X-Edit and ICEdit use their single-reference paths; ACE++ uses its single-reference path and builds on ACE and FLUX.1- dev [41, 53, 54]. Emu3.5, FireRed-Edit, and HunyuanImage-3.0 support at most three total images. On a K = 4 case, these models retain the current/previous rollout view and the two auxiliaries with highest mean target-visible overlap; original order breaks ties. The controlled Context-K study excludes native max-three and single-reference paths.

The two pose-free video models are invoked once per benchmark step; the final video frame is scored and becomes the next current view. Their table labels carry an “Autoreg” sufix that names this rollout wrapper rather than a diferent model checkpoint. HY-WorldMirror-2.0 and Lingbot-World receive native cumulative 6-DoF trajectories and are scored at the same step boundaries. Their scores are references rather than directly comparable pose-free baselines.

Table 6 The eight motion clauses. Axis is the component of the relative pose that determines both the magnitude and the direction word; d is in meters and a in degrees.
<table><tr><td>Subtype</td><td>Axis</td><td>Clause</td></tr><tr><td>A1 forward/backward</td><td> $t _ { z }$ </td><td>Move the camera mainly {forward |backward} by about d me- ters, with only minor sideways shift and minor rotation.</td></tr><tr><td>A2 lateral strafe</td><td> $t _ { x }$ </td><td>Move the camera mainly to the {right |left} by about d meters, with only minor forward/backward shift and minor rotation.</td></tr><tr><td>A3 yaw rotation</td><td>yaw</td><td>Rotate the camera mainly to the {right |left} by about a degrees, with only minor camera translation.</td></tr><tr><td>A4 pitch rotation</td><td>pitch</td><td>Tilt the camera mainly {upward |downward} by about a de- grees, with only minor camera translation and minor yaw rota- tion.</td></tr></table>

Evaluator preprocessing is component-specific. The oficial CMG pipeline preserves native aspect ratio; the aspect-ratio sensitivity test in Section F applies a center crop only as an intervention. SSP uses the frozen detector and depth-estimator preprocessing described in Section D.2. No blanket resize-to-GT rule defines the benchmark.

## C.2 Prompt templates

Every instruction in the benchmark is produced by filling one fixed template with the measured relative camera pose of the step. No instruction is written by hand or by a language model, so the mapping from pose to text is exact and reproducible.

Let $\Delta P _ { m } = P _ { m - 1 } ^ { - 1 } P _ { m }$ be the GT relative transform for step m, and let $a _ { m } ^ { * }$ be its dominant scalar component. A deterministic template converts sgn $( a _ { m } ^ { * } ) , | a _ { m } ^ { * } |$ , and the unit into a motion clause: four clause templates, one per atomic family, each carrying a binary direction slot, so eight realized clauses in total (Table 6). The magnitude is $| a _ { m } ^ { * } |$ rounded to two decimals for translation (meters; observed range 0.15–1.20) and to one decimal for rotation (degrees; observed range 8.1–44.8 for yaw and 6.1–25.0 for pitch); the direction word is selected by $\mathrm { s g n } ( a _ { m } ^ { * } )$ . Checked against pose\_metadata for all 2,360 instructions, both fill rules hold with zero exceptions.

Let ⟨motion⟩ denote that clause. For K = 1 the instruction is the clause followed by the scene-preservation sentence, with no preamble:

⟨MOTION⟩ Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency.

For $K \geq 2$ the same clause is wrapped so that the current view is unambiguous and the auxiliary views cannot be copied:

Image 1 is the current view and should be used as the starting point. The other images are auxiliary context views of the same scene. Starting from Image 1, generate the new view after the following camera motion: ⟨MOTION⟩ Keep the same static 3D scene, object identities, spatial layout, relative depth order, and perspective consistency. Use the auxiliary images only to preserve scene identity and spatial layout. Do not simply copy any input image.

The current view is always referred to as Image 1, independently of K. Multi-step tasks introduce no additional wording: each step of a chain (3 steps) or of a cycle (2 steps) is an independent instantiation of the same template. The two clauses of a cycle therefore difer only in the direction word (240/240 cycles), and within the single-family chains C1 and C2 the three clauses difer only in the magnitude (240/240); C3 mixes action families by construction, so its clauses difer in the clause template as well. Blanking the numerals in all 2,360 instructions leaves exactly 16 distinct forms—two wrappers × eight motion clauses—with no seventeenth. Figure 7 shows instantiated single- and multi-image prompts rather than schematic paraphrases.

For a chain, $T _ { 1 } , T _ { 2 } , T _ { 3 }$ are independently derived from the three GT transitions:

$$
\begin{array} { r } { \hat { f } _ { 1 } = M ( f _ { 0 } , C , T _ { 1 } ) , } \\ { \hat { f } _ { 2 } = M ( \hat { f } _ { 1 } , C , T _ { 2 } ) , } \\ { \hat { f } _ { 3 } = M ( \hat { f } _ { 2 } , C , T _ { 3 } ) . } \end{array}\tag{12}
$$

The GT sequence determines each instruction, while the generated sequence determines the visual current view. The model never receives a GT intermediate frame. For a cycle, the return prompt is derived from $\Delta P _ { 1  0 }$ rather than from a string-level direction swap.

## D Complete Evaluation Protocol

## D.1 Camera Motion Grounding (CMG)

CMG uses the camera decoder of DA3Nested-Giant-Large [24] at resolution 504 with upper-bound resizing, native aspect ratio, saddle-balanced reference selection, and ray-pose disabled. At step m, it estimates the relative motion of the physical GT pair and the evaluated pair. For translation, monocular scale is calibrated from the GT pair,

$$
\alpha _ { m } = \frac { \Vert \mathbf { t } _ { m } ^ { * } \Vert _ { 2 } } { \Vert \hat { \mathbf { t } } _ { m } ^ { \mathrm { G T } } \Vert _ { 2 } } , \qquad \hat { \mathbf { t } } _ { m } = \alpha _ { m } \hat { \mathbf { t } } _ { m } ^ { \mathrm { r a w } } .\tag{13}
$$

The same factor is applied to the current-to-generated estimate; it changes neither direction nor rotation. Calibration is performed independently for each physical target transition, never per model, scene, or global dataset. Hence every system evaluated on a transition receives the identical $\alpha _ { m }$ . This makes translation magnitude interpretable for that transition but leaves magnitude-dependent conclusions conditional on the learned pose backend and GT-assisted calibration.

Let $a _ { m } ^ { * }$ and $\hat { a } _ { m }$ be the GT and estimated signed values of the instructed component. Direction and magnitude agreement are

$$
\begin{array} { l } { { d _ { m } = \mathbb { I } [ \mathrm { s g n } ( \hat { a } _ { m } ) = \mathrm { s g n } ( a _ { m } ^ { * } ) ] , } } \\ { { q _ { m } = \left( 1 + \displaystyle \frac { | \hat { a } _ { m } - a _ { m } ^ { * } | } { \operatorname* { m a x } ( | a _ { m } ^ { * } | , \epsilon _ { a } ) } \right) ^ { - 1 } . } } \end{array}\tag{14}
$$

where $\epsilon _ { a } = 0 . 1$ m for translation and $5 ^ { \circ }$ for rotation. The step score is

$$
g _ { m } = { \frac { 1 } { 2 } } d _ { m } ( 1 + q _ { m } ) .\tag{15}
$$

Wrong-direction estimates receive zero; correct-direction estimates receive between 0.5 and 1 according to magnitude. The benchmark action label selects the scored component for every step, and all 2,360 modelevaluation steps have a defined CMG score. The separately generated score-calibration references in Section J use a model-independent 2,358-step mask because two near-zero Chain actions are undefined in that evaluation run.

Temporal and protocol aggregation. For a case with $T _ { c }$ steps, the case-level score $\mathrm { C M G } _ { c }$ is the plain mean of its step scores,

$$
\mathrm { C M G } _ { c } = \frac { 1 } { T _ { c } } \sum _ { m = 1 } ^ { T _ { c } } g _ { m } .\tag{16}
$$

An atomic case reduces to its single step $( T _ { c } = 1 )$ , while a three-step chain and a two-step cycle average their steps. Unlike the SSP pillars below, CMG uses no worst-step term: every step contributes equally within the case, and a within-case failure is diluted rather than pinned. Cases are then averaged within Atomic, Chain, and Cycle, and the three protocol means receive equal final weight. Thus a three-step case does not receive three times the weight of an atomic case.

GT detection

SSP is inspectable: detections → correspondences → spatial / depth checks → score  
![](images/e3361a686729896dcf612537b0c3d8e40694e0d14caa153d9da3d7b1fa0c30df.jpg)  
matched prediction

![](images/57e49cc56839574d0157d1c5049397e4d1c70bebb265cc271033c6cb0dea3876.jpg)  
unmatched prediction  
HiDream-O1

![](images/94a5d8c956f9149b09ec196242792786e3cb3c53701eadb27e69ad86fed02445.jpg)  
HYworld

![](images/32e8ef915f178d4edc1378d94063d908011452db9599c8d0330df956aeb020ff.jpg)

Depth cells compare front/behind sign only; GT physical depth and generated DA3 depth are never placed on a shared numeric scale.  
![](images/c71bcbb96ee18e3e7682392c8b78d81b5909c91f168eeeb0b6fb13f92d85a374.jpg)

![](images/1fe60f919ddb5e4009bbc5dd4c1df880f179f9dbadc12219d771e320fe87a6a1.jpg)

![](images/e3c0e7ff6fe40d47ab415754462fc3bb939258829cdd9d9505caeab154be5c26.jpg)  
Depth object IDs: 0=sofa 1=table 2=window 1 3=window 2

![](images/b50dfcf59de54e98705828ff5a4ada827310a417bd5549707ddf3c1e6781ef1a.jpg)  
Figure 8 Worked SSP example on one target step. (a) GT detections define the evaluable object set; $^ { ( \mathtt { b } , \mathtt { c } ) }$ each model’s generated view is scored by its matched, missing, and unmatched-prediction counts. (d) The pillar formulas. (e,f) The lower-triangular matrix records the front/behind check for every eligible GT pair (green correct, vermillion wrong or tied, light gray an eligible pair with a missing endpoint), and the bars give the resulting $F _ { 1 }$ , Spatial, Integrity, and SSP values. Depth cells compare front/behind sign only; GT physical depth and generated monocular depth are never placed on a shared numeric scale.

## D.2 Scene State Preservation (SSP)

Figure 8 traces the complete SSP computation on a single target step for two models: detector boxes on the target and generated views, the resulting one-to-one correspondences, the pairwise depth-order checks, and the three pillar scores that combine into the step score. The definitions below make each stage precise.

Detection and observable GT objects. Grounding DINO [25] detects objects in GT and generated images with the same prompt union. The union contains a fixed 36-class indoor vocabulary plus at most 20 normalized labels produced once for the GT target by Qwen3-VL [23]; the deduplicated list is capped at 64 prompts and then held fixed for both images. Each evaluated step contributes 1–8 target-derived labels, producing 36–41 prompts, so neither cap is active. Box and text thresholds are 0.25/0.25, and SAM3 is disabled; SSP therefore uses rectangular detector boxes rather than segmentation silhouettes.

GT poses and physical depth identify target objects that are supported by the supplied input views and pass the visibility and matchability filters. Let G and P be the resulting evaluable GT-object count and filtered generated-object count. A propose–demote–recover procedure based on Qwen3-VL, with DINOv3 [26] appearance verification, produces M one-to-one matches. Unmatched generated proposals pass correspondence exclusion and box deduplication before being included in the false-positive count. They do not incur a second penalty because they already reduce precision through P in F . Objects outside the shared prompt union are outside the declared open-vocabulary scope. A model-independent GT-only mask removes an entire case if any required step has G = 0. The same mask excludes 44 cases and retains 1,356 cases/2,265 steps for all 18 evaluated systems.

Presence, position, and integrity. Object presence is

$$
F _ { 1 } = \frac { 2 M } { G + P } .\tag{17}
$$

For the matched set M, normalized center error and coverage-aware position are

$$
\begin{array} { c } { { \mathrm { P E } = \displaystyle \frac { 1 } { M } \sum _ { ( i , j ) \in { \mathcal { M } } } \frac { \| { \bf c } _ { i } ^ { * } - \hat { \bf c } _ { j } \| _ { 2 } } { \sqrt { H ^ { 2 } + W ^ { 2 } } } , } } \\ { { \mathcal { P } = \displaystyle \frac { M } { G } \exp ( 1 - \mathrm { P E } , 0 , 1 ) . } } \end{array}\tag{18}
$$

The matched-object appearance score $\bar { I }$ combines DINO identity, structural SSIM, edge IoU, sharpness, color, and shape with weights (0.30, 0.20, 0.15, 0.12, 0.13, 0.10). If a signal is unavailable, the available weights are renormalized to sum to one. Coverage-aware integrity is

$$
{ \mathcal { T } } = { \frac { M } { G } } { \bar { I } } .\tag{19}
$$

If $M = 0 ;$ , both $\mathcal { P }$ and $\mathcal { T }$ are zero. The factor $M / G$ ensures that unmatched GT objects receive no position or integrity credit.

Planar and depth topology. For GT object pair $( a , b )$ , the planar score is the non-negative cosine agreement between GT and generated box-center ofsets:

$$
O _ { a b } ^ { x y } = \left\{ \begin{array} { l l } { \operatorname* { m a x } \ ( 0 , \cos ( \mathbf { c } _ { b } ^ { * } - \mathbf { c } _ { a } ^ { * } , \hat { \mathbf { c } } _ { b } - \hat { \mathbf { c } } _ { a } ) ) , } & { a , b \mathrm { m a t c h e d , } } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{20}
$$

The cosine uses the product of the two ofset norms in its denominator. Benchmark validation rejects a GT ofset norm below $1 0 ^ { - 8 } ;$ a generated ofset norm below $1 0 ^ { - 8 }$ receives zero planar credit rather than an undefined cosine. GT depth is the median physical RGB-D depth in the central 80% of each frozen GT box. Generated depth is the median DA3 single-image depth in the central 80% of each matched generated box. Both require at least 16 finite pixels in the numeric range [0.05, 20]. For depths $z _ { a } , z _ { b }$ , define

$$
r _ { a b } = \frac { z _ { a } - z _ { b } } { 0 . 5 ( | z _ { a } | + | z _ { b } | ) + \varepsilon } .\tag{21}
$$

A GT pair is depth-eligible when $| r _ { a b } ^ { \mathrm { G T } } | > 0 . 0 3$ . Its depth score $O _ { a b } ^ { z } \in \{ 0 , 1 \}$ is one only if both endpoints are matched, generated depths are valid and non-tied under the same 0.03 gate, and the front–behind sign agrees; otherwise it is zero. Only relative order is compared, so GT and generated depth need not share a metric scale.

The per-pair topology score is

$$
T _ { a b } = \left\{ \begin{array} { l l } { { \frac { 1 } { 2 } ( O _ { a b } ^ { x y } + O _ { a b } ^ { z } ) , } } & { { \mathrm { G T ~ d e p t h – e l i g i b l e , } } } \\ { { O _ { a b } ^ { x y } , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right.\tag{22}
$$

For $G \geq 2 .$

$$
\mathcal { T } = \binom { G } { 2 } ^ { - 1 } \sum _ { 1 \leq a < b \leq G } T _ { a b } , \qquad \mathcal { S } = \frac { 1 } { 2 } ( \mathcal { P } + \mathcal { T } ) .\tag{23}
$$

For $G = 1 , S = \mathcal { P }$ . An unmatched endpoint therefore receives zero in every incident topology pair.

Temporal and protocol aggregation. For a case with $T _ { c }$ steps, SSP applies the mean-plus-worst operator separately to each pillar,

$$
\mathrm { M P W } _ { c } ( x ) = \frac { 1 } { 2 } \left( \frac { 1 } { T _ { c } } \sum _ { m = 1 } ^ { T _ { c } } x _ { m } + \operatorname* { m i n } _ { m } x _ { m } \right) ,\tag{24}
$$

and defines

$$
\mathrm { S S P } _ { c } = \frac { 1 } { 3 } \sum _ { x \in \{ F _ { 1 } , S , \mathbb { Z } \} } \mathrm { M P W } _ { c } ( x ) .\tag{25}
$$

Cases are averaged within protocol, and Atomic/Chain/Cycle receive equal final weight. MPW makes transient state loss visible without allowing longer cases to dominate by step count.

<table><tr><td rowspan="2">Model</td><td colspan="3">Atomic</td><td colspan="3">Chain</td><td colspan="3">Cycle</td></tr><tr><td>CMG</td><td>Dir.</td><td>Gated</td><td>CMG</td><td>Dir.</td><td>Gated</td><td>CMG</td><td>Dir.</td><td>Gated</td></tr><tr><td>GPT-Image-2</td><td>0.735</td><td>0.887</td><td>0.582</td><td>0.717</td><td>0.870</td><td>0.564</td><td>0.712</td><td>0.848</td><td>0.576</td></tr><tr><td>Seedream-5</td><td>0.716</td><td>0.859</td><td>0.573</td><td>0.674</td><td>0.819</td><td>0.530</td><td>0.675</td><td>0.808</td><td>0.541</td></tr><tr><td>Gemini-3-Pro</td><td>0.598</td><td>0.723</td><td>0.473</td><td>0.603</td><td>0.753</td><td>0.454</td><td>0.578</td><td>0.700</td><td>0.456</td></tr><tr><td>HiDream-O1</td><td>0.566</td><td>0.691</td><td>0.441</td><td>0.443</td><td>0.561</td><td>0.326</td><td>0.443</td><td>0.552</td><td>0.334</td></tr><tr><td>FLUX.2-dev</td><td>0.560</td><td>0.680</td><td>0.440</td><td>0.362</td><td>0.467</td><td>0.257</td><td>0.421</td><td>0.525</td><td>0.317</td></tr><tr><td>HunyuanImage-3.0</td><td>0.524</td><td>0.647</td><td>0.401</td><td>0.413</td><td>0.529</td><td>0.296</td><td>0.450</td><td>0.565</td><td>0.335</td></tr><tr><td>Qwen-Image-Edit</td><td>0.568</td><td>0.685</td><td>0.450</td><td>0.409</td><td>0.531</td><td>0.288</td><td>0.432</td><td>0.540</td><td>0.325</td></tr><tr><td>Step1X-Edit</td><td></td><td>0.3170.421</td><td>0.214</td><td>0.306</td><td>0.406</td><td>0.205</td><td>0.271</td><td>0.360</td><td>0.182</td></tr><tr><td>BAGEL-7B-MoT</td><td>0.460</td><td>0.566</td><td>0.355</td><td>0.431</td><td>0.539</td><td>0.323</td><td>0.414</td><td>0.515</td><td>0.314</td></tr><tr><td>OmniGen2</td><td>0.495</td><td>0.616</td><td>0.374</td><td>0.392</td><td>0.502</td><td>0.282</td><td>0.382</td><td>0.483</td><td>0.280</td></tr><tr><td>ACE++</td><td>0.409</td><td>0.521</td><td>0.297</td><td>0.384</td><td>0.499</td><td>0.269</td><td>0.382</td><td>0.498</td><td>0.267</td></tr><tr><td>FireRed-Edit</td><td>0.480</td><td>0.603</td><td>0.357</td><td>0.383</td><td>0.496</td><td>0.269</td><td>0.356</td><td>0.456</td><td>0.256</td></tr><tr><td>ICEdit</td><td>0.359</td><td>0.469</td><td>0.249</td><td>0.330</td><td>0.440</td><td>0.219</td><td>0.325</td><td>0.423</td><td>0.227</td></tr><tr><td>Emu3.5</td><td>0.415</td><td>0.499</td><td>0.330</td><td>0.469</td><td>0.573</td><td>0.365</td><td>0.446</td><td>0.531</td><td>0.360</td></tr><tr><td>Kling-v2.1-Autoreg</td><td>0.746</td><td>0.956</td><td>0.536</td><td>0.724</td><td>0.955</td><td>0.494</td><td>0.760</td><td>0.965</td><td>0.556</td></tr><tr><td>Seedance-1.0-Pro-Fast-Autoreg</td><td>0.609</td><td>0.756</td><td>0.462</td><td>0.568</td><td>0.738</td><td>0.397</td><td>0.645</td><td>0.792</td><td>0.498</td></tr><tr><td>HY-WorldMirror-2.0</td><td>0.839</td><td>0.989</td><td>0.688</td><td>0.883</td><td>0.981</td><td>0.784</td><td>0.821</td><td>0.983</td><td>0.658</td></tr><tr><td>Lingbot-World</td><td></td><td>0.765 0.915</td><td>0.616</td><td>0.742</td><td>0.877</td><td>0.606</td><td>0.735</td><td>0.892</td><td>0.578</td></tr></table>

Table 7 CMG decomposition for all 18 systems. Gated is direction-gated magnitude $d q ;$ CMG is $( d + d q ) / 2 .$ . Values use the oficial case-first protocol aggregation and match the headline CMG scores in the main text. Italicized rows are pose-conditioned references reported separately from the primary pose-free ranking.

## D.3 Overall and auxiliary quality

Let $\mathcal { C } _ { p }$ be all cases in protocol $p$ and $\nu _ { p }$ its fixed SSP-valid subset. The reported scores are

$$
\mathrm { C M G } = \frac { 1 } { 3 } \sum _ { p } \frac { 1 } { | \mathcal { C } _ { p } | } \sum _ { c \in \mathcal { C } _ { p } } \mathrm { C M G } _ { c } ,
$$

$$
\mathrm { S S P } = \frac { 1 } { 3 } \sum _ { p } \frac { 1 } { | \mathcal { V } _ { p } | } \sum _ { c \in \mathcal { V } _ { p } } \mathrm { S S P } _ { c } .\tag{26}
$$

$$
\mathrm { O v e r a l l } = { \frac { 1 } { 2 } } ( \mathrm { C M G } + \mathrm { S S P } ) .\tag{27}
$$

PSNR, LPIPS, NIQE [29], MUSIQ [30], and CLIP-IQA [31] are auxiliary. RefSim averages min–max-normalized PSNR and inverted LPIPS; VisQual averages normalized inverted NIQE, MUSIQ, and CLIP-IQA. Both are oriented so higher is better and are excluded from Overall and model ranking.

## D.4 Full metric decomposition

Tables 7 and 8 report the per-protocol submetrics for all 18 systems, exposing the structure hidden by the two headline averages. Two patterns hold throughout. First, direction accuracy is consistently higher than direction-gated magnitude, so the gap between the Dir. and Gated columns measures how much CMG credit comes from sense alone: GPT-Image-2 reaches 0.887 Atomic direction accuracy but only 0.582 gated magnitude, and every pose-free system shows the same ordering. Second, SSP degradation on Chain cases spreads across presence $\left( F _ { 1 } \right)$ , spatial organization, and integrity rather than concentrating in one pillar, which is why no single submetric explains the Chain drop.

The pose-conditioned references confirm the interpretation. Privileged 6-DoF control lifts HY-WorldMirror-2.0 and Lingbot-World well above the pose-free field on CMG, yet their SSP remains within the pose-free range because state preservation depends on target-view content that trajectory control alone does not guarantee.

<table><tr><td rowspan="3">Model</td><td colspan="3">Atomic</td><td colspan="4">Chain</td><td colspan="4">Cycle</td></tr><tr><td>SSP</td><td>F1</td><td> $\mathrm { S p . }$ </td><td></td><td>Int. SSP</td><td>F1</td><td> $\mathrm { S p . }$ </td><td>Int. SSP</td><td></td><td>F1</td><td></td><td>Sp. Int.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-Image-2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.66 0.72 0.66 0.59 0.55 0.61 0.54 0.50 0.60 0.65 0.60 0.55</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seedream-5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.61 0.70 0.60 0.54 0.480.57 0.45 0.42 0.53 0.61 0.51 0.47</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini-3-Pro</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.62 0.70 0.60 0.55 0.49 0.56 0.45 0.45 0.55 0.62 0.53 0.50</td><td></td><td></td><td></td><td></td></tr><tr><td>HiDream-O1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.61 0.69 0.61 0.54 0.50 0.57 0.48 0.45 0.57 0.63 0.57 0.52</td><td></td><td></td><td></td><td></td></tr><tr><td>FLUX.2-dev</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.61 0.69 0.60 0.54 0.49 0.56 0.47 0.45 0.56 0.62 0.55 0.51</td><td></td><td></td><td></td><td></td></tr><tr><td>HunyuanImage-3.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.62 0.69 0.61 0.55 0.41 0.49 0.39 0.36 0.57 0.64 0.56 0.51</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-Image-Edit</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.62 0.69 0.610.560.40 0.480.37 0.350.540.61 0.530.49</td><td></td><td></td><td></td><td></td></tr><tr><td>Step1X-Edit</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.60 0.69 0.580.540.49 0.57 0.450.450.630.70 0.61 0.58</td><td></td><td></td><td></td><td></td></tr><tr><td>BAGEL-7B-MoT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.550.640.520.490.390.470.340.350.460.540.420.41</td></tr><tr><td>OmniGen2 ACE++</td><td></td><td>0.44 0.55 0.40 0.380.30 0.380.260.260.480.570.44 0.42</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.52 0.61 0.49 0.47 0.35 0.43 0.31 0.31 0.49 0.56 0.46 0.44</td></tr><tr><td>FireRed-Edit</td><td></td><td>0.50 0.60 0.46 0.43 0.21 0.27 0.18 0.17 0.42 0.51 0.39 0.36</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ICEdit</td><td></td><td>0.33 0.43 0.30 0.28 0.24 0.30 0.20 0.20 0.31 0.39 0.29 0.27</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Emu3.5</td><td></td><td>0.32 0.44 0.25 0.28 0.10 0.15 0.07 0.09 0.19 0.26 0.14 0.16</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kling-v2.1-Autoreg</td><td></td><td>0.56 0.64 0.54 0.51 0.32 0.40 0.28 0.29 0.51 0.57 0.49 0.47</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seedance-1.0-Pro-Fast-Autoreg 0.54 0.62 0.52 0.480.360.44 0.320.320.44 0.51 0.42 0.40</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HY-WorldMirror-2.0 Lingbot-World</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.60 0.680.580.540.440.52 0.410.40 0.550.62 0.530.50</td><td></td><td></td><td>0.480.580.440.430.420.50 0.380.370.550.610.520.51</td><td></td></tr></table>

Table 8 SSP decomposition on the fixed GT-only mask for all 18 systems. Spatial $\left( \mathrm { S p . } \right)$ and Integrity (Int.) are coverage-aware All-GT quantities, so unmatched GT objects contribute zero. Italicized rows are pose-conditioned references.
<table><tr><td>Evidence</td><td>Axis</td><td>Statistic</td><td>Scope</td></tr><tr><td>Aggregate rank</td><td>CMG</td><td> $\rho = 0 . 9 4 3 , \tau _ { b } = 0 . 8 6 7 , \mathrm { e x a c t } p = 0 . 0 1 6 7$ </td><td>6 systems, 340 steps</td></tr><tr><td>Aggregate rank</td><td>SSP</td><td> $\rho = 0 . 9 4 3 , \tau _ { b } = 0 . 8 6 7 , \mathrm { e x a c t } p = 0 . 0 1 6 7$ </td><td>6 systems, 340 steps</td></tr><tr><td>Aggregate rank</td><td>Overall</td><td> $\rho = 0 . 9 4 3$   $\tau _ { b } = 0 . 8 6 7$  , exact  $p = 0 . 0 1 6 7$ </td><td>6 systems, 340 steps</td></tr><tr><td>Item alignment</td><td>CMG</td><td>mean  $\rho = 0 . 3 4 8 \ [ 0 . 3 0 2 , 0 . 3 9 2 ] ;$ </td><td>pairwise 0.639 [0.621,0.657] 340 steps/180 cases</td></tr><tr><td>Item alignment</td><td>SSP</td><td>mean ρ = 0.334 [0.289,0.377]; pairwise 0.634</td><td> $\rho \colon 3 3 3 / 1 7 7 ;$  pairwise: 340/180</td></tr><tr><td>Reannotation</td><td>CMG</td><td> $\mathrm { I C C ( A , 1 ) { = } 0 . 5 9 6 ; \mathrm { I C C ( A , 3 ) { = } 0 . 8 1 6 ; } }$  cross-group</td><td>102 steps, 6 systems</td></tr><tr><td>Reannotation</td><td>SSP</td><td>ICC(A,1)=0.632; ICC(A,3)=0.838; cross-group</td><td>102 steps, 6 systems</td></tr></table>

Table 9 Human-validation evidence. Brackets are 95% parent-case cluster-bootstrap intervals from 10,000 resamples (seed 20260619). Pairwise agreement gives half credit to automatic ties on human-strict pairs.

## E Human Validation and Ranking Uncertainty

## E.1 Blinded protocol

The primary study samples 180 protocol-stratified parent cases (80 Atomic, 60 Chain, 40 Cycle), yielding 340 steps. Three rubric-trained annotators independently rank the same six anonymized systems per step, with ties allowed. CMG judgments emphasize action direction, magnitude, and target-view agreement; SSP judgments emphasize target-visible objects, image-plane organization, relative depth, and recognizable appearance. Visual polish is not a primary criterion, and model identities and automatic scores are hidden.

Ties receive average ranks and rank r maps to utility $( 6 - r ) / 5$ . Utility is first averaged across annotators for each item–model pair. Automatic metrics are recomputed on identical examples using the oficial stepto-case-to-protocol aggregation, SSP mask, and mean-plus-worst rule. The matched data contain exactly $3 4 0 \times 6 = 2 { , } 0 4 0$ model–step rows. Three additional annotators independently re-evaluate a stratified 102-step subset (24 Atomic, 54 Chain, 24 Cycle).

<table><tr><td>Automatic metric</td><td>Spearman ρ</td><td>Kendall  $\tau _ { b }$ </td></tr><tr><td>Overall</td><td>0.943</td><td>0.867</td></tr><tr><td>CLIP-IQA</td><td>0.886</td><td>0.733</td></tr><tr><td>MUSIQ</td><td>0.714</td><td>0.467</td></tr><tr><td>PSNR</td><td>0.600</td><td>0.333</td></tr><tr><td>-NIQE</td><td>0.429</td><td>0.467</td></tr><tr><td>-LPIPS</td><td>0.314</td><td>0.200</td></tr></table>

Table 10 System-level agreement with aggregate human ordering on the same six-system panel. Negative NIQE/LPIPS orient all metrics as higher-is-better. The comparison is descriptive because $n = 6 .$

## E.2 Agreement at system and item level

The automatic ordering is GPT-Image-2, Seedream-5, Gemini-3-Pro, Qwen-Image-Edit, BAGEL-7B-MoT, and OmniGen2; human consensus swaps only the last two. The high six-system correlation supports aggregate ordering on this matched panel. The moderate item-level values—including the reported 0.348 CMG mean correlation—show why the metrics are used for case-, protocol-, and system-level comparison rather than exact per-example grading or separate validation of every SSP subcomponent.

## E.3 Comparison with conventional image metrics

To test whether the aggregate human agreement merely reflects generic image quality or reference similarity, we recompute each candidate metric on the identical six-system, 340-step panel and apply the same stepto-case-to-protocol aggregation. Table 10 orients NIQE and LPIPS so that higher values are better. Overall best matches the human system ordering, followed by CLIP-IQA. With only six systems, these diferences are descriptive rather than a powered significance claim.

The component-level boundary is equally important. At item level, CMG reaches mean $\rho = 0 . 3 4 8$ , while negative dominant-action absolute error reaches 0.371; their paired parent-case-bootstrap diference is $[ - 0 . 0 3 9 , - 0 . 0 0 7 ]$ SSP reaches 0.334, comparable to Spatial (0.357), Integrity (0.356), and PSNR (0.356), whose paired diferences from SSP include zero. The composite scores are therefore justified as pre-defined summaries for aggregate capability comparison, not as universally superior per-item perceptual metrics.

## E.4 Paired case-bootstrap uncertainty

We resample parent cases within subtype, use identical draws across systems, and recompute the complete step-to-case-to-protocol aggregation. CMG and SSP use independent subtype-stratified draws; Overall averages their replicate scores. Table 11 reports 2,000 percentile-bootstrap resamples (seed 20260619). The one-sided $p _ { \mathrm { f i i p } }$ is the fraction of paired replicates in which each displayed row reverses or ties with the following row.

The median Overall interval half-width is approximately 0.010. In particular, HY-WorldMirror-2.0 and GPT-Image-2 are not statistically ordered $( p _ { \mathrm { f l i p } } = 0 . 3 4 4 )$ , nor are several close middle and lower pairs. These intervals quantify benchmark case sampling only: one frozen output exists per model–step, so they do not include stochastic generation variance.

## F Evaluator Robustness

Each intervention changes one evaluator component while holding generated outputs, GT actions, and remaining aggregation fixed. The tests establish broad rank robustness, not numerical equivalence or ground-truth accuracy of one learned evaluator.

CMG pose and preprocessing. Replacing DA3 with VGGT-1B on a controlled six-model panel yields $\rho = 0 . 9 4 3$ and $\tau _ { b } = 0 . 8 6 7$ , with unchanged Top-1 and Top-3. Native-aspect versus GT-aspect center-crop preprocessing changes CMG by 0.0019 on average and 0.0044 at most, with $\rho = \tau _ { b } = 1 . 0 0 0$ . Across the 16 pose-free systems, direction-only aggregation agrees with full CMG at $\rho = 0 . 9 9 7$ and $\tau _ { b } = 0 . 9 8 3$ . The broad ordering is therefore not created by the magnitude term alone.

<table><tr><td>System</td><td>Overall [95% CI]</td><td>CMG [95% CI]</td><td>SSP [95% CI]</td><td>pflip</td></tr><tr><td>HY-WorldMirror-2.0</td><td>0.664 [0.655,0.673]</td><td>[0.838,0.857]</td><td>[0.466,0.495]</td><td>0.344</td></tr><tr><td>GPT-Image-2</td><td>0.662 [0.654,0.670]</td><td>[0.708,0.734]</td><td>[0.592,0.612]</td><td>0.001</td></tr><tr><td>Lingbot-World</td><td>0.639 [0.629,0.648]</td><td>[0.734,0.760]</td><td>[0.517,0.544]</td><td>0.001</td></tr><tr><td>Seedream-5</td><td>0.615 [0.606,0.625]</td><td>[0.673,0.703]</td><td>[0.530,0.554]</td><td>0.028</td></tr><tr><td>Kling-v2.1-Autoreg</td><td>0.605 [0.597,0.612]</td><td>[0.735,0.753]</td><td>[0.453,0.477]</td><td>0.000</td></tr><tr><td>Gemini-3-Pro</td><td>0.572 [0.561,0.583]</td><td>[0.576,0.610]</td><td>[0.538,0.563]</td><td>0.000</td></tr><tr><td>Seedance-1.0-Pro-Fast-Autoreg</td><td>0.528 [0.517,0.538]</td><td>[0.591,0.623]</td><td>[0.435,0.461]</td><td>0.290</td></tr><tr><td>HiDream-O1</td><td>0.524 [0.513,0.534]</td><td>[0.468,0.501]</td><td>[0.552,0.575]</td><td>0.000</td></tr><tr><td>FLUX.2-dev</td><td>0.501 [0.491,0.511]</td><td>[0.431,0.464]</td><td>[0.543,0.565]</td><td>0.325</td></tr><tr><td>HunyuanImage-3.0</td><td>0.498 [0.487,0.509]</td><td>[0.445,0.479]</td><td>[0.521,0.545]</td><td>0.314</td></tr><tr><td>Qwen-Image-Edit</td><td>0.495 [0.485,0.505]</td><td>[0.452,0.487]</td><td>[0.508,0.531]</td><td>0.000</td></tr><tr><td>BAGEL-7B-MoT</td><td>0.451 [0.440,0.461]</td><td>[0.417,0.453]</td><td>[0.452,0.479]</td><td>0.035</td></tr><tr><td>OmniGen2</td><td>0.438 [0.427,0.449]</td><td>[0.406,0.439]</td><td>[0.440,0.465]</td><td>0.421</td></tr><tr><td>Step1X-Edit</td><td>0.436 [0.426,0.446]</td><td>[0.283,0.314]</td><td>[0.563,0.586]</td><td>0.000</td></tr><tr><td>ACE++</td><td>0.400 [0.389,0.410]</td><td>[0.376,0.408]</td><td>[0.394,0.422]</td><td>0.080</td></tr><tr><td>FireRed-Edit</td><td>0.390 [0.380,0.399]</td><td>[0.391,0.422]</td><td>[0.361,0.385]</td><td>0.000</td></tr><tr><td>Emu3.5</td><td>0.324 [0.314,0.335]</td><td>[0.425,0.459]</td><td>[0.196,0.215]</td><td>0.157</td></tr><tr><td>ICEdit</td><td>0.316 [0.306,0.327]</td><td>[0.321,0.354]</td><td>[0.282,0.309]</td><td></td></tr></table>

Table 11 Case-bootstrap uncertainty. Italicized rows are pose-conditioned references, not members of the primary pose-free ranking. Adjacent rows with $p _ { \mathrm { f l i p } } \geq 0 . 0 5$ should be read as local ties.

SSP geometry and detection. Replacing generated-object DA3 depth with VGGT on an eight-model, 342-case common panel preserves the complete ordering $( \rho = 1 . 0 0 0 )$ while shifting mean macro SSP by −0.0124. Jointly changing detector box/text thresholds from 0.25/0.25 to 0.20/0.20 and 0.30/0.30 gives Overall $\rho = 0 . 9 2 9 / 0 . 9 7 6$ unchanged Top-1, and maximum absolute score change 0.00873. Comparing the 36 fixed prompts with their deterministic union with frozen GT labels gives $\rho = 0 . 9 7 6$ , unchanged Top-1, and maximum change 0.00542. Common masks prevent alternatives from gaining evaluation coverage.

Correspondence and temporal aggregation. Replacing propose–demote–recover with DINO-Hungarian assignment preserves Overall Top-1 but yields $\rho = 0 . 8 5 7$ overall and 0.595 on Cycle; assignments and close ranks remain sensitive. Without human pair labels, this does not establish either matcher as more accurate. Replacing pillar-wise MPW with mean-only aggregation gives SSP $\rho = 0 . 9 9 4$ and Overall $\rho = 0 . 9 9 6$ on 18 systems with unchanged Top-3.

## G Benchmark-Design and Sampling Robustness

Evaluator ablations ask whether a learned backend creates the ranking. This section asks a complementary benchmark-design question: whether the principal conclusions depend on the temporal functional, caseindependence assumption, dominant data source, score weights, or selected benchmark size. All analyses reuse the frozen outputs and the 16 pose-free primary systems.

## G.1 Matched temporal aggregation

The oficial axes intentionally answer diferent questions: CMG averages action execution across steps, whereas SSP uses $\mathrm { M P W } ( x ) = \textstyle { \frac { 1 } { 2 } } ( \operatorname* { m e a n } ( x ) + \operatorname* { m i n } ( x ) )$ within each pillar to expose a corrupted intermediate view. Comparing their Atomic-to-Chain drops therefore mixes capability and functional sensitivity. We repeat the comparison with both axes using the same functional. Confidence intervals use 5,000 dataset-stratified scene-cluster bootstrap resamples shared across systems and axes.

<table><tr><td>CMG/SSP temporal rule</td><td>CMG drop</td><td>SSP drop</td><td>SSP-CMG</td><td>Gap 95% scene Cl</td><td>SSP larger</td></tr><tr><td>Mean MPW (official)</td><td>0.0593</td><td>0.1643</td><td>+0.1050</td><td>[0.0837, 0.1250]</td><td>14/16</td></tr><tr><td>Mean Mean</td><td>0.0593</td><td>0.0792</td><td>+0.0198</td><td>[-0.0007, 0.0396]</td><td>11/16</td></tr><tr><td>MPW MPW</td><td>0.1826</td><td>0.1643</td><td>-0.0182</td><td>[-0.0396, 0.0021]</td><td>5/16</td></tr></table>

Table 12 Model-mean Atomic-to-Chain drops under matched and oficial temporal aggregation. “SSP larger” counts systems whose SSP drop exceeds their CMG drop.

The oficial 0.059/0.164 drops are correct descriptive statistics, but the 0.105 contrast is not aggregationinvariant. Under matched means, SSP still has the larger point drop by 0.020, but its interval slightly crosses zero; under matched MPW, CMG has the larger point drop by 0.018. A shared temporal-weight sweep changes the sign near a minimum-step weight of 0.26. The robust finding is that Chain degrades both axes. The additional oficial SSP drop quantifies the tail failure that its worst-step-sensitive definition is designed to expose; it should not be interpreted alone as evidence for an independent state-memory mechanism.

## G.2 Scene-cluster ranking uncertainty

The primary case bootstrap stratifies by nine instruction subtypes. Because 1,400 cases come from 771 scenes, we additionally resample scenes within each source dataset and retain every case from a selected scene; 5,000 shared resamples preserve paired model comparisons. For the 16-system primary ranking, the median Overall interval half-width increases from 0.0100 under case sampling to 0.0117 under scene sampling, a median factor of 1.180. The pose-free Top-1 comparison remains unambiguous $( p _ { \mathrm { H i p } } = 0 )$ , and the Seedream–Kling Top-3 boundary has $p _ { \mathrm { f i i p } } = 0 . 0 3 5$ . However, six of 15 adjacent pairs have $p _ { \mathrm { f l i p } } \geq 0 . 0 5$ . These results support the leading system and broad tiers while reinforcing that several middle and lower neighbors are local ties.

## G.3 Source coverage and leave-one-source-out stability

Source coverage is feasibility-constrained rather than balanced: ScanNet++ contributes 1,003 of 1,400 cases, while Matterport3D contributes 22. Hypersim and Matterport3D contain no Chain cases, so we compare sources on their common Atomic+Cycle protocols instead of conflating source and protocol composition.

All four source-specific panels retain GPT-Image-2 as Top-1, with common-protocol rank correlations of $\rho = 0 . 8 9 7  – 0 . 9 5 9$ against the full-data Atomic+Cycle ordering. Removing any one source leaves $\rho = 0 . 9 6 8 –$ –1.000 against the full primary ranking; even after removing ScanNet++, which contributes 71.6% of the cases, the pose-free Top-3 remains GPT-Image-2, Seedream-5, and Kling, and equal weighting of the four sources also retains GPT-Image-2 as Top-1. The broad ordering is therefore not solely a ScanNet++ artifact.

## G.4 Score-weight sensitivity and benchmark size

Let Overal $\mathrm { l } _ { \lambda } = \lambda \mathrm { C M G } + ( 1 - \lambda )$ SSP. For $\lambda \in [ 0 . 2 5 , 0 . 7 5 ]$ , the minimum rank correlation with equal weighting is $\rho = 0 . 9 2 6$ and GPT-Image-2 remains Top-1; it remains Top-1 over the wider [0, 0.85] range, while Kling becomes Top-1 at the motion-dominant λ = 0.90. Equal weighting is therefore not uniquely necessary, but extreme preferences appropriately change the ranking. For SSP, 20,000 Dirichlet(1, 1, 1) pillar-weight draws retain GPT-Image-2 as Top-1 in 100% of draws, and the 95% range of rank correlation with equal-pillar SSP is [0.991, 1.000]. Dropping any one pillar gives $\rho = 0 . 9 9 7 , 0 . 9 9 7$ , and 1.000, with unchanged Top-1.

Subsampling within the nine instruction subtypes (1,000 no-replacement draws per size) shows the 16-system ordering is already stable at 140 cases (median ρ = 0.985, Top-1 match 1.000) and that the full 1,400 cases mainly reduce score error (MAE 0.0121 → 0.0023). This post-hoc check is not a power analysis for future systems and does not guarantee every adjacent order.

<table><tr><td colspan="6">K Native Fixed/filtered Fixed/all Source recall Aux.-only recall</td></tr><tr><td>1</td><td>0.539</td><td>0.584</td><td>0.537</td><td>0.725</td><td></td></tr><tr><td>2</td><td>0.533</td><td>0.582</td><td>0.531</td><td>0.735</td><td>0.632</td></tr><tr><td>3</td><td>0.514</td><td>0.556</td><td>0.506</td><td>0.725</td><td>0.650</td></tr><tr><td>4</td><td>0.504</td><td>0.549</td><td>0.500</td><td>0.710</td><td>0.631</td></tr></table>

Table 13 Model-balanced Context-K results. Source recall uses the stable cross-K intersection; auxiliary-only objects are $\mathcal { O } _ { K } \backslash \mathcal { O } _ { 1 }$
<table><tr><td>Family</td><td>Bin</td><td>Direction</td><td>Under</td><td>Near</td><td>Over</td></tr><tr><td>Translation Small</td><td></td><td>0.565</td><td>0.611</td><td>0.083</td><td>0.306</td></tr><tr><td>Translation Medium</td><td></td><td>0.609</td><td>0.689</td><td>0.093</td><td>0.218</td></tr><tr><td>Translation Large</td><td></td><td>0.610</td><td>0.707</td><td>0.117</td><td>0.175</td></tr><tr><td>Rotation</td><td>Small</td><td>0.663</td><td>0.561</td><td>0.106</td><td>0.333</td></tr><tr><td>Rotation</td><td>Medium</td><td>0.655</td><td>0.595</td><td>0.137</td><td>0.268</td></tr><tr><td>Rotation</td><td>Large</td><td>0.659</td><td>0.680</td><td>0.133</td><td>0.187</td></tr></table>

Table 14 Magnitude diagnostics over 16 pose-free systems. Parent cases are averaged before systems.

## H Claim-Focused Diagnostics

All diagnostics reuse frozen evaluated outputs. They explain score behavior and do not create additional leaderboards or causal claims.

## H.1 Controlled Context-K: fixed objects and view confusion

Seven native multi-image models are evaluated on 70 matched base cases at K = 1–4; a common mask retains 69 cases/116 steps. Under the model-balanced aggregation used in Figure 3(b), CMG at $K = 1 , 2 , 3 , \scriptscriptstyle { \cdot }$ 4 is 0.519, 0.551, 0.580, and 0.582. Native SSP evaluates the object set supported at each K, which changes on 45/118 base steps as auxiliaries add evidence. We therefore recompute SSP on 440 exact GT label–box instances present at every K. “Fixed/filtered” also filters predictions to this intersection; “fixed/all predictions” retains all generated predictions and is the conservative variant because unmatched predictions outside the intersection still reduce precision.

From K = 1 to K = 4, native, fixed/filtered, and fixed/all-prediction SSP change by −0.0354, −0.0347, and −0.0368, respectively. The latter two changes have 95% CIs of [−0.0653, −0.0033] and [−0.0644, −0.0084], with one-sided $p _ { \geq 0 } = 0 . 0 1 4 7$ and 0.0047. Fixed-object F1, Spatial, and Integrity change by −0.0433, −0.0327, and −0.0280. Intervals use 10,000 subtype-stratified parent-case bootstrap resamples shared across K and models (seed 20260729). Thus the small decline persists under a fixed-object comparison, while stable source-visible recall changes only from 0.725 to 0.710 and auxiliary-only recall remains 0.631–0.650. The result indicates imperfect integration of added views, not that context is intrinsically harmful.

CLIP ViT-L/14@336 provides a complementary view-confusion diagnostic. At K = 4, 62.2% of step-1 outputs are feature-nearest to an auxiliary but only 0.6% are near-pixel copies (RGB RMSE ≤ 0.02 after deterministic resizing). Strict auxiliary-confused outputs score 0.064 lower SSP than other rows (95% CI [0.009,0.119]); across paired K = 2–4 rows, auxiliary attraction and SSP change correlate only weakly $( \rho = - 0 . 1 0 9 )$ . This is evidence of partial feature-space attraction, not widespread literal copying.

## H.2 Motion magnitude and residual pose

Translation and rotation steps are binned by tertiles of absolute GT action magnitude. Under/near/over denote predicted-to-GT magnitude ratios below 0.8, within [0.8, 1.2], and above 1.2, conditional on correct direction.

<table><tr><td>Step</td><td>N</td><td>Median r</td><td>Near-static</td><td>Direction</td></tr><tr><td>Outbound 3,840</td><td></td><td>0.412</td><td>37.4%</td><td>66.4%</td></tr><tr><td>Return</td><td>3,840</td><td>0.013</td><td>66.3%</td><td>53.1%</td></tr></table>

Table 15 Observed Cycle action behavior across the 16 pose-free systems. N counts transitions. Near-static is a descriptive $r \leq 0 . 2$ diagnostic; Direction is signed dominant-action accuracy.
<table><tr><td>Subset</td><td>N</td><td></td><td>SSP [95% CI]</td><td> $F _ { 1 }$ </td><td>Spatial</td><td>Integrity</td></tr><tr><td>All valid</td><td>36,240</td><td></td><td>0.510 [0.504,0.517]</td><td>0.589</td><td>0.489</td><td>0.452</td></tr><tr><td>CMG ≥ 0.8</td><td>8,956</td><td></td><td>0.561 [0.553,0.569]</td><td>0.634</td><td>0.551</td><td>0.499</td></tr><tr><td>Loose full pose</td><td>1,515</td><td></td><td>0.595 [0.579,0.610]</td><td>0.669</td><td>0.589</td><td>0.526</td></tr><tr><td>Strict full pose</td><td>418</td><td></td><td>0.597 [0.571,0.624]</td><td>0.668</td><td>0.591</td><td>0.533</td></tr></table>

Table 16 Target-view SSP conditioned on pose accuracy. Loose thresholds are translation-direction/SO(3)/translationnorm errors $\leq 3 0 ^ { \circ } / 1 5 ^ { \circ } / 0 . 3$ m; strict thresholds $\mathrm { a r e } \leq 2 0 ^ { \circ } / 1 0 ^ { \circ } / 0 . 2 \mathrm { m }$ .

Small-rotation direction accuracy is 0.663 versus the 0.525 majority-sign baseline. For medium and large actions, 59.5–70.7% of direction-correct outputs under-execute and only 9.3–13.7% lie within 20% of the requested magnitude. Full-pose records also show that dominant-axis success is not full-pose fidelity: GPT-Image-2 has mean translation-direction and SO(3) errors of 43.68<sup>◦</sup> and $1 5 . 8 0 ^ { \circ }$ ; pose-conditioned HY-WorldMirror-2.0 reaches 10.51<sup>◦</sup> translation-direction and $2 . 5 2 ^ { \circ } \ \mathrm { S O ( 3 ) }$ error but a 1.491 m full-translation error due to over-scaling.

## H.3 Cycle return behavior

Cycle tests whether action responsiveness persists after one self-generated step. For this diagnostic only, we define the absolute response ratio $r = | \hat { a } _ { m } | / | a _ { m } ^ { * } |$ on the instructed dominant component and call a response near-static when $r \leq 0 . 2$ . This describes estimated camera response, not pixel identity.

Ten of the 16 systems are near-static on more than half of their return steps. Among the 3,840 Cycle cases, 32.2% are near-static on both outbound and return, and another 34.1% move outbound but become near-static on return. The first group barely executes the requested camera cycle; the second moves to an intermediate viewpoint but then fails to invert the action. Individual outputs also include wrong-direction and over-motion failures, so the aggregate should not be read as uniform behavior across models. These observations motivate free-running inverse-action supervision; they do not establish whether the suppression arises from the generator, accumulated visual drift, or the learned pose evaluator.

## H.4 SSP conditioned on pose accuracy

To separate wrong-view and state efects observationally, Table 16 pools transitions from the 16 pose-free systems under increasingly strict pose criteria. Confidence intervals cluster 10,000 bootstrap resamples by parent case (seed 20260729).

SSP rises with pose accuracy but remains near 0.60 in the strictest stratum, so wrong view is not the only observed source of SSP deficit.

## H.5 Object retention by projected size

On 1,351 cases/2,255 steps with identical GT instance catalogs across systems, confirmed retention is 0.604 [0.592,0.615], 0.635 [0.623,0.645], and 0.696 [0.686,0.705] for small, medium, and large projected objects. Conditional position varies by at most 0.011 and integrity by 0.006, so the 0.093 large–small gap mainly reflects detection/matching-confirmed retention. From Chain step 1 to step 3, retention drops by 0.195 for large furniture and 0.191 for small/portable objects; cumulative degradation is not specific to small objects.

<table><tr><td>Model</td><td>Full trans. (m)</td><td>Off-axis trans. (m)</td><td>Trans. dir. (°)</td><td>SO(3) error (°)</td><td>Cross-axis rot. (°)</td></tr><tr><td>GPT-Image-2</td><td>0.682</td><td>0.363</td><td>43.68</td><td>15.80</td><td>6.41</td></tr><tr><td>Seedream-5</td><td>0.715</td><td>0.302</td><td>60.43</td><td>15.40</td><td>6.95</td></tr><tr><td>Gemini-3-Pro</td><td>1.004</td><td>0.466</td><td>69.02</td><td>24.28</td><td>9.93</td></tr><tr><td>HiDream-O1</td><td>0.865</td><td>0.364</td><td>84.35</td><td>19.66</td><td>7.57</td></tr><tr><td>FLUX.2-dev</td><td>1.077</td><td>0.570</td><td>81.76</td><td>23.08</td><td>8.77</td></tr><tr><td>HunyuanImage-3.0</td><td>0.784</td><td>0.272</td><td>80.03</td><td>18.86</td><td>6.40</td></tr><tr><td>Qwen-Image-Edit</td><td>0.903</td><td>0.411</td><td>80.00</td><td>23.57</td><td>7.77</td></tr><tr><td>BAGEL-7B-MoT</td><td>0.963</td><td>0.479</td><td>73.64</td><td>30.83</td><td>9.30</td></tr><tr><td>OmniGen2</td><td>0.875</td><td>0.374</td><td>83.99</td><td>21.91</td><td>8.61</td></tr><tr><td>Step1X-Edit</td><td>0.784</td><td>0.221</td><td>90.62</td><td>19.87</td><td>5.78</td></tr><tr><td>ACE++</td><td>1.174</td><td>0.570</td><td>87.09</td><td>29.92</td><td>13.25</td></tr><tr><td>FireRed-Edit</td><td>0.881</td><td>0.341</td><td>89.53</td><td>20.28</td><td>7.10</td></tr><tr><td>Emu3.5</td><td>1.279</td><td>0.638</td><td>78.03</td><td>31.67</td><td>12.36</td></tr><tr><td>ICEdit</td><td>1.414</td><td>0.730</td><td>85.22</td><td>26.26</td><td>9.99</td></tr><tr><td>Kling-v2.1-Autoreg</td><td>1.020</td><td>0.303</td><td>30.30</td><td>20.93</td><td>6.55</td></tr><tr><td>Seedance-1.0-Pro-Fast-Autoreg</td><td>1.167</td><td>0.590</td><td>45.42</td><td>28.62</td><td>8.35</td></tr><tr><td>HY-WorldMirror-2.0</td><td>1.491</td><td>0.512</td><td>10.51</td><td>2.52</td><td>1.35</td></tr><tr><td>Lingbot-World</td><td>0.635</td><td>0.236</td><td>17.64</td><td>14.03</td><td>5.48</td></tr></table>

Table 17 Full-pose diagnostics from the corrected DA3 records; lower is better. Eligible steps are averaged within case, then protocol, followed by an equal Atomic/Chain/Cycle macro. Translation direction and both rotation columns cover all eligible records. Full-vector translation fields cover 1,007 translation-action steps per model except GPT-Image-2, Seedream-5, and Gemini-3-Pro (887 each); missing vectors are left undefined rather than imputed. Italicized rows are pose-conditioned references.

## H.6 Off-axis and full-pose error

CMG deliberately scores only the frozen dominant action component. The corrected all-step pose records also permit a stricter diagnostic of unintended or mismatched residual motion. For a translation instruction whose dominant axis is $k \in \{ x , z \}$ , let $\mathbf { e } _ { t } = \hat { \mathbf { t } } - \mathbf { t } ^ { * }$ and define

$$
E _ { t } ^ { \mathrm { f u l l } } = \| \mathbf { e } _ { t } \| _ { 2 } , \qquad E _ { t } ^ { \mathrm { o f f } } = \| ( e _ { t , j } ) _ { j \neq k } \| _ { 2 } .\tag{28}
$$

This subtracts the complete GT translation before measuring the residual, so legitimate of-axis motion in the real trajectory is not itself penalized. We additionally report the angle between the full predicted and GT translation vectors. For rotation, $E _ { R } ^ { \mathrm { f u l l } }$ is the SO(3) geodesic error. The cross-axis diagnostic is $| \hat { \theta } - \theta ^ { * } |$ for a yaw instruction and $| \hat { \psi } - \psi ^ { * } |$ for a pitch instruction; the geodesic term additionally captures roll error and axis coupling.

These diagnostics explain why dominant-axis success is not equivalent to full-pose fidelity. HY-WorldMirror-2.0 has the smallest translation-direction and SO(3) errors, yet its full translation error is large because it often over-scales motion; conversely, a near-static model can have modest of-axis error while failing the instructed component. Among pose-free systems, GPT-Image-2 combines the strongest CMG with relatively low full-translation and rotation errors, but its mean translation-direction error remains 43.68<sup>◦</sup>. The table is therefore diagnostic only: it does not enter CMG, SSP, Overall, or model ranking, and unequal full-vector coverage precludes treating small numerical diferences as a separate leaderboard.

## H.7 Qualitative case studies

Figures 9 and 10 illustrate the score behavior discussed above on concrete cases from the frozen bundle. Every panel reuses the oficial records: green and vermillion borders mark correct and wrong action direction, and each caption reports the estimated action and the matched-over-evaluable GT-object count $( \mathrm { M } / \mathrm { G } )$ used by CMG and SSP.

Figure 9 shows one atomic case per action family evaluated across five systems. A recurring pattern from the magnitude diagnostics in Table 14 is visible directly: models frequently commit to the correct sense yet miss the requested magnitude, for example overshooting a forward translation or under-rotating a yaw, so the green border coexists with a large action error. Wrong-direction outputs and degraded renders lose object matches even when the scene is otherwise plausible.

Atomic action gallery: diverse models fail in different ways  
![](images/2d78093a4ebb3a8135e12b11663961883979ef60cadc7c63f439801849fd1226.jpg)  
Figure 9 Atomic action gallery. Rows are the four action families and columns are the shared input, the geometry-defined target, and five representative systems. Captions give the estimated dominant action, whether its direction matches the instruction, and the matched/evaluable GT-object count. Diverse systems fail in diferent ways: correct-direction magnitude errors, wrong-direction outputs, and object loss under degraded renders.

Figure 10 shows chain and cycle rollouts. The chain rows make the cumulative degradation of Section H concrete: individual steps can keep the correct direction while the matched-object count falls toward zero by the final step, so a trajectory drifts even when no single step is grossly wrong. The cycle rows show that a correct outbound step can be followed by a wrong-direction or weak return, consistent with the return-action failures in Appendix H.3.

![](images/ff09703c8c8dda2309d5f4913dfdca919c1aaf6ffc2f5621851cf854c7fb19c5.jpg)  
Figure 10 Rollout gallery. Rows alternate the ground-truth and generated frames of three-step chains and two-step cycles. Chains expose cumulative object-retention loss as the matched/evaluable count drops across steps; cycles show cases in which a correct outbound motion is followed by an incorrect return. Correct single steps therefore do not guarantee trajectory consistency.

## I EGOGEN-TRAIN as a Post-Training Resource

These experiments ask whether the benchmark can guide model development; they are separate from both the 16-system primary ranking and the pose-conditioned references. Every row is evaluated on all 1,400 cases/2,360 steps with the same CMG implementation, SSP exclusion mask, detector/matcher records, depth protocol, and aggregation as the main study. Each system produces one frozen output per step, and each SFT variant is trained once; the rows are therefore not averages over repeated training runs.

## I.1 Data construction and pair formation

EgoGen-Train is mined from training-side scenes in DL3DV [55], HyperSim, Matterport3D, ScanNet, and ScanNet++. The eligible source pool contains 8,315 scenes; geometry mining constructs the same four Atomic, three Chain, and two Cycle subtypes used by the benchmark. Candidate trajectories must satisfy the motion-dominance, overlap, visibility, depth-layer, and pose-separation gates. A subsequent visual and semantic filter rejects ambiguous motion, unusable images, diferent-scene sequences, dominant dynamic content, and cases without trackable spatial anchors. A final bidirectional depth-consistent check requires visible overlap in [0.35, 0.85] at every trajectory step. Exact (dataset, scene ID) matching against the terminal blocklist leaves zero evaluation-scene overlap.

The final clean pool contains 4,140 scenes and 66,214 trajectories. Table 18 separates trajectories from optimizer examples because each multi-step trajectory produces more than one edit pair.

<table><tr><td>Protocol</td><td>Trajectories</td><td>Pairs per trajectory</td><td>Edit pairs</td></tr><tr><td>Atomic</td><td>33,144</td><td>1</td><td>33,144</td></tr><tr><td>Chain</td><td>8,929</td><td>3</td><td>26,787</td></tr><tr><td>Cycle</td><td>24,141</td><td>2</td><td>48,282</td></tr><tr><td>Total</td><td>66,214</td><td>一</td><td>108,213</td></tr></table>

Table 18 EgoGen-Train composition. Multi-step trajectories are converted into teacher-forced image-edit pairs while remaining intact during trajectory-level subset sampling.

The trajectory counts by source are 33,972 for DL3DV, 405 for HyperSim, 96 for Matterport3D, 6,627 for ScanNet, and 25,114 for ScanNet++. Instructions are generated from the calibrated relative pose and state the dominant action, direction, and approximate magnitude. At step i > 1, teacher forcing uses the physical target from step i − 1 as the first edit reference; it never uses a model prediction. Fixed same-scene auxiliary references remain available at every step, and current and target frames are excluded from auxiliary sampling.

The scale subsets are sampled from the final clean pool at the trajectory level with subtype stratification. The clean/raw comparison instead pairs a 40k sample from this final pool with a subtype-matched, best-efort sample from the raw geometry-mined pool before the visual/semantic and depth-consistent visibility filters. The raw pool contains too few C3 mixed chains to fill its requested quota, so the nominal raw-40k variant contains 39,761 trajectories and 64,653 edit pairs. This comparison tests the complete filtering pipeline; it does not isolate any single filtering rule.

## I.2 Qwen development probes

Qwen-Image-Edit-2511 is fine-tuned after removing every evaluation scene. The full clean pool contains 66,214 trajectories and 108,213 teacher-forced edit pairs. Training uses eight NVIDIA A800 80-GB GPUs, global batch eight, BF16, gradient checkpointing, a 589,824-pixel dynamic-resolution cap, and rank-16 LoRA (α = 16, zero dropout) on the difusion transformer. AdamW uses learning rate $1 0 ^ { - 4 }$ , weight decay 0.01, default betas, and frozen non-adapter weights.

Fixed-budget variants run for 4,000 updates. The full-clean continuation starts from the 4k LoRA weights and performs 9,527 additional updates, for 13,527 cumulative weight updates—approximately one pass over the 108,213 edit pairs. Because this continuation changes optimization budget, it is a development checkpoint rather than a controlled data-scaling result.
<table><tr><td>Variant</td><td>Train samples</td><td>Edit pairs</td><td>Budget</td><td>Overall</td><td>CMG</td><td>SSP</td></tr><tr><td>Base</td><td></td><td></td><td>no SFT</td><td>0.495</td><td>0.470</td><td>0.520</td></tr><tr><td>Clean 20k</td><td>20,000</td><td>32,684</td><td>4k</td><td>0.590</td><td>0.651</td><td>0.530</td></tr><tr><td>Clean 40k</td><td>40,000</td><td>65,370</td><td>4k</td><td>0.589</td><td>0.664</td><td>0.513</td></tr><tr><td>Clean full</td><td>66,214</td><td>108,213</td><td>4k</td><td>0.597</td><td>0.659</td><td>0.535</td></tr><tr><td>Clean full continuation</td><td>66,214</td><td>108,213</td><td>~13.5k</td><td>0.681</td><td>0.773</td><td>0.589</td></tr><tr><td>Matched clean 40k</td><td>40,000</td><td>65,370</td><td>4k</td><td>0.615</td><td>0.698</td><td>0.531</td></tr><tr><td>Matched raw 40k</td><td>39,761</td><td>64,653</td><td>4k</td><td>0.591</td><td>0.653</td><td>0.528</td></tr></table>

Table 19 Benchmark-guided fine-tuning probe. Data-pool rows hold optimizer updates fixed; the continuation changes the training budget and is not a pure data-scale comparison.

At 4k updates, increasing the cleaned pool from 20k to the full set does not produce a monotonic SSP trend, so these rows do not establish a scaling law. The per-protocol records show that Chain SSP rises from 0.397 for the base model to 0.442 for Clean full at 4k updates and to 0.505 after the 13.5k-update continuation. The single-step case improves far less: Atomic SSP rises only from 0.618 for the base model to 0.669 after the full continuation, even though this case gives the model a ground-truth previous frame and admits no rollout accumulation, and it does not rise as the pool grows. The full continuation reaches 0.681 Overall but uses substantially more optimization. These point estimates establish that benchmark-derived data can produce a competitive held-out-scene checkpoint; the matched controls below are required to attribute axis-specific diferences.

## I.3 Matched endpoint and second-backbone controls

Training and generation controls. The Qwen comparison continues the independently trained clean/raw 40k adapters for exactly 9,527 additional local updates, yielding 13,527 updates from the same base for both endpoints. Continuation restores adapter weights only and reinitializes optimizer and scheduler state; the original 4k adapters were trained without an explicit seed. The comparison is therefore update-matched, not bitwise-equivalent to uninterrupted training and not averaged over training randomness.

For the second backbone, we convert the same 65,370 cleaned 40k Qwen edit pairs one-to-one to the OmniGen2 format without resampling, reordering, or prompt rewriting. OmniGen2 trains for 4,000 updates on eight A800 80-GB GPUs with global batch eight, BF16, gradient checkpointing, seed 2233, and rank-8 LoRA (α = 8, zero dropout) on attention query/key/value/output projections. AdamW uses learning rate $8 \times 1 0 ^ { - 7 }$ , betas (0.9, 0.95), weight decay 0.01, gradient clipping at one, and a 500-update warmup. The matched base and SFT generations use the same benchmark inputs, seed 42, sampler, target-size rule, and strict Cycle step-2 conditioning. Qwen generation uses $1 0 2 4 ^ { 2 }$ outputs; OmniGen2 uses 50 Euler steps, text guidance 5.0, and image guidance 2.0.

All four matched rows contain 1,400 cases and 2,360 steps with zero missing outputs and identical validity masks. Reconstructed headline and pillar scores match the frozen final metric files to below $7 \times 1 0 ^ { - 1 6 }$ absolute error.
<table><tr><td>Matched endpoint</td><td>Overall</td><td>CMG</td><td>SSP</td></tr><tr><td>Qwen raw 40k, 13.5k updates</td><td>0.669</td><td>0.766</td><td>0.572</td></tr><tr><td>Qwen clean 40k, 13.5k updates</td><td>0.673</td><td>0.783</td><td>0.563</td></tr><tr><td>OmniGen2 matched base</td><td>0.449</td><td>0.440</td><td>0.457</td></tr><tr><td>OmniGen2 clean-40k SFT, 4k updates</td><td>0.430</td><td>0.450</td><td>0.410</td></tr></table>

Table 20 Absolute scores for the matched SFT controls. The Qwen rows isolate clean versus raw training data at the same cumulative update count; the OmniGen2 base is regenerated under the SFT row’s inference configuration.

Paired uncertainty. The primary analysis uses 20,000 paired scene-cluster bootstrap resamples (seed 20260810): scenes are sampled with replacement within each source dataset, all cases inherit their scene multiplicity, and the full protocol macro is recomputed before subtraction. A 20,000-resample subtype-stratified case bootstrap gives the same headline decisions. Intervals characterize paired benchmark sampling for these frozen runs, not decoding or training-run variance.
<table><tr><td>Paired comparison</td><td></td><td>∆Overall</td><td>△CMG</td><td></td><td>ΔSSP</td><td>△CMG-△SSP</td></tr><tr><td>Qwen clean - raw, 13.5k</td><td>+0.004</td><td>[−0.003,+0.011]</td><td>+0.017 [+0.007, +0.026]</td><td>-0.009</td><td>[−0.019, +0.002]</td><td>+0.025</td><td>[+0.011,+0.040]</td></tr><tr><td>OmniGen2 SFT – base</td><td>-0.019</td><td>[−0.031, -0.008]</td><td>+0.010</td><td>-0.010, +0.029]</td><td>-0.048</td><td>-0.059,−0.037]</td><td>+0.057 [+0.036, +0.080]</td></tr></table>

Table 21 Paired diferences with 95% scene-cluster bootstrap intervals. Qwen cleaning reliably improves CMG but not Overall at the long endpoint; OmniGen2 SFT leaves CMG unresolved while reliably reducing SSP and Overall. The positive axis-gap intervals are the cross-backbone evidence that pairwise supervision changes action and state at diferent rates.

For Qwen, both CMG components support the action gain: direction changes by $+ 0 . 0 1 9 \ [ + 0 . 0 0 8 , + 0 . 0 3 0 ]$ and gated magnitude by +0.015 [+0.005, +0.024]. SSP F1 decreases by −0.012 [−0.023, −0.001], whereas spatial and integrity changes remain unresolved. For OmniGen2, SSP F1, spatial relations, and integrity fall by −0.037, −0.058, and −0.049, respectively, with all intervals below zero. Hence the second backbone supports an action–state divergence, not a claim that SFT improves CMG.

<table><tr><td>Comparison</td><td>Protocol</td><td colspan="4">△CMG</td></tr><tr><td rowspan="3">Qwen clean - raw</td><td>Atomic</td><td>-0.001</td><td>[−0.006, +0.004]</td><td></td><td>+0.003 [-0.011, +0.016]</td></tr><tr><td>Chain</td><td>+0.036</td><td>[+0.017, +0.056]</td><td>-0.011</td><td>[-0.028, +0.005]</td></tr><tr><td>Cycle</td><td>+0.015</td><td>[-0.003, +0.033]</td><td>-0.017</td><td>[-0.040, +0.006]</td></tr><tr><td rowspan="3">OmniGen2 SFT − base</td><td>Atomic</td><td>+0.004</td><td>[-0.023, +0.031]</td><td>-0.026</td><td>[-0.040, -0.011]</td></tr><tr><td>Chain</td><td>+0.030</td><td>[+0.001, +0.058]</td><td>-0.056</td><td>[-0.073, -0.040]</td></tr><tr><td>Cycle</td><td>-0.005</td><td>[-0.046, +0.036]</td><td></td><td>-0.062 [−0.088, -0.036]</td></tr></table>

Table 22 Protocol-level paired diferences and 95% scene-cluster intervals. Qwen’s reliable CMG gain localizes to Chain, while OmniGen2 SSP declines under every protocol and most strongly in sequential settings. Protocol rows are diagnostic decompositions of the headline efects.

## I.4 Budget, subgroup, and qualitative diagnostics

Budget. The early Qwen cleaning advantage does not grow under continued optimization. At 4k updates, clean minus raw is +0.024 [+0.015, +0.033] Overall and +0.045 [+0.032, +0.058] CMG, but this advantage contracts at the long endpoint until Overall is no longer separated from zero. SSP is +0.003 [−0.008, +0.014] at 4k and −0.009 [−0.019, +0.002] at 13.5k. Combined with the fixed-update 20k/40k/full-pool probe, this rules out the simple prescription that exposing the same pairwise objective to more distinct pairs or more updates is suficient for state preservation.

Source, subtype, action, and concentration. Source-level estimates are heterogeneous and source is not fully crossed with protocol. Qwen’s resolved SSP loss appears in ScanNet++ (−0.012 [−0.023, −0.002]); OmniGen2 SSP falls on ScanNet (−0.046 [−0.075, −0.017]) and ScanNet++ (−0.043 [−0.055, −0.031]), while the much smaller Matterport3D slice is inconclusive. The subtype map shows Qwen’s largest axis gap in C1 translation chains and OmniGen2 gaps across translation, rotation, and mixed sequential subtypes. In an explicitly exploratory step-level action slice, forward motion gives Qwen +0.072 CMG and +0.013 SSP, but OmniGen2 +0.083 CMG and −0.052 SSP. The top 10% of scenes account for at most 43.9% of absolute headline contribution across the four model–metric combinations, so no headline efect is carried by a handful of scenes. Dataset, subtype, and action intervals are uncorrected diagnostics rather than independent discoveries.

Prespecified qualitative audit. Before visual inspection, we select two distinct-scene cases for each comparison in each of three categories: joint improvement (∆CMG and ∆SSP both above 0.05), action–state tradeof (∆CMG above 0.05 and ∆SSP below −0.05), and joint regression (both below −0.05). Cases are ranked deterministically within category, and every input, GT target, and generated step is retained. The full audit contains 12 cases. In the top tradeof case for each backbone (Qwen Cycle, ∆CMG = +0.502 and ∆SSP = −0.323; OmniGen2 Atomic, ∆CMG = +0.956 and ∆SSP = −0.491), the treatment executes the requested motion more successfully but preserves fewer target-view objects or relations, so a positive Overall change (+0.089 for Qwen; +0.232 for OmniGen2) hides a large SSP loss. These metric-extreme examples illustrate the exchange and are not frequency estimates.

Interpretation boundary. The completed controls support the conclusion that pairwise SFT changes the two axes at diferent rates. They do not support universal data-cleaning gains, an OmniGen2 CMG improvement, or a causal claim averaged over training seeds. Although the data contain Chain and Cycle trajectories, teacher forcing conditions every later step on the preceding physical target and therefore does not train recovery from model-induced drift. A stronger baseline should select checkpoints on the CMG–SSP Pareto frontier or enforce SSP non-degradation, then test explicit identity/count, planar/depth topology, integrity, free-running sequence, and worst-step losses under matched compute.

## J Reference Baselines and Score Calibration

Four unranked references calibrate the metric scale without defining mathematical lower or upper bounds. GT-target oracle emits the physical target at every step. Source-copy recursively emits the initial view without executing an action. Oracle nearest-visible-input copy uses the GT target only to select the most similar image among the current and auxiliary views exposed by the benchmark interface; it never inserts the target into the candidate set. Scene-disjoint NN retrieval-copy selects a training transition after removing all 771 evaluation scenes, then restricts candidates by K, action direction, and a training-only magnitude tertile before CLIP-based selection. No evaluation target image, feature, pose, depth, mask, or label enters the scene-disjoint retrieval procedure.

<table><tr><td>Reference</td><td>Overall</td><td>CMG</td><td>SSP</td></tr><tr><td>GT-target oracle</td><td>0.940</td><td>0.980</td><td>0.899</td></tr><tr><td>Source-copy</td><td>0.391</td><td>0.181</td><td>0.601</td></tr><tr><td>Oracle nearest-visible-input copy</td><td>0.457</td><td>0.302</td><td>0.613</td></tr><tr><td>Scene-disjoint NN retrieval-copy</td><td>0.377</td><td>0.539</td><td>0.215</td></tr></table>

Table 23 Unranked score-calibration references. CMG uses the model-independent 2,358-step reference mask; SSP uses the same 44-case exclusion mask as the primary evaluation.

The ranked set contains 16 pose-free systems; HY-WorldMirror-2.0 and Lingbot-World remain separate pose-conditioned references. GPT-Image-2 reaches 0.662 Overall, leaving a 0.278 gap to the practical oracle. Source-copy shows that preserving the input without moving can retain a moderate SSP but receives little CMG credit. The two retrieval rows are diagnostics rather than deployable methods: oracle nearest-visible-input copy uses the target only to select among already exposed views, whereas scene-disjoint NN retrieval-copy never accesses evaluation targets or scenes.

## K Reproduction Scope

The planned release is designed around stable case/step identifiers that connect permitted model inputs and instructions with evaluator-only targets, poses, depth, visibility, and object metadata. Pose-free inference reads only the permitted images and language instruction. Versioned manifests will record file hashes and the fixed SSP mask; accompanying scripts will recompute CMG, SSP, aggregation, human-alignment statistics, bootstrap intervals, and claim-focused diagnostics from frozen outputs and per-step records. The evaluation release will cover the selected 1,400 cases; exhaustive intermediate candidates from early mining are not required to reproduce the reported evaluation. For EgoGen-Train, the release scope comprises the row-level trajectory manifest, source-relative frame identifiers, pose-derived instructions, deterministic teacher-forced conversion, construction metadata, frozen hashes, and the exact scene-leakage audit. Restricted source pixels, private absolute paths, and third-party model weights are not redistributed.

## L Evaluator Configuration and Compute

Table 24 lists the frozen models that implement the two metrics and the auxiliary diagnostics. Every component runs with fixed weights and fixed hyperparameters, and no evaluator is trained or fine-tuned on benchmark outputs. Given the frozen generated images and per-step records, the pipeline is deterministic, so re-running it reproduces the reported scores exactly.

Evaluator inference runs on a single 80-GB GPU per component, and the detection, matching, and pose stages dominate wall-clock time. Generation cost is not separately metered here because it varies by provider and interface: closed systems (GPT-Image-2, Seedream-5, Gemini-3-Pro) are queried through their public APIs, whereas open systems run locally under their native decoding settings. We therefore report evaluator configuration rather than a single normalized cost figure. The fine-tuning probe in Section I uses eight NVIDIA A800 80-GB GPUs, as stated there.

## M Data Licensing and Ethics

EgoGenEval is assembled from four established research RGB-D datasets—ScanNet++ [20], ScanNet [13], HyperSim [21], and Matterport3D [14]. EgoGen-Train additionally uses DL3DV [55] as a training-only source. Both resources are used for non-commercial academic research under the original licenses and access terms. No new scenes are captured for this work. Source scans may contain sensitive visual details inherited from the provider datasets. The release plan includes a reporting and removal route for afected records; we do not assume that the upstream datasets are free of person-specific content.

<table><tr><td>Component</td><td>Role in evaluation</td></tr><tr><td>DA3Nested-Giant-Large [24]</td><td>Relative camera pose for CMG; single-image depth for SSP depth topology</td></tr><tr><td>Grounding DINO [25]</td><td>Open-vocabulary object detection in GT and generated images</td></tr><tr><td>Qwen3-VL [23]</td><td>GT target-label proposal and propose-demote-recover matching</td></tr><tr><td>DINOv3 [26]</td><td>Appearance-identity verification for matched objects</td></tr><tr><td>CLIP ViT-L/14@336</td><td>Feature-space view-confusion diagnostic (Context-K)</td></tr><tr><td>VGGT-1B [56]</td><td>Alternate pose/depth backend for robustness ablations</td></tr></table>

Table 24 Frozen evaluator components and their roles. All run with fixed weights; SAM3 segmentation is disabled, so SSP uses rectangular detector boxes.

Release boundary. The planned release will provide versioned metadata, source-relative identifiers, scenedisjoint split records, and local materialization scripts rather than republishing restricted source pixels. Users will obtain the underlying datasets from their oficial providers and remain subject to the corresponding licenses and access agreements.

The human validation study in Section E uses rubric-trained annotators who rank anonymized system outputs and collect only ordinal quality judgments, with no personal data recorded. Released material links cases to source frames through stable identifiers and provides construction metadata and per-step evaluator records needed to reproduce the reported results; redistribution of the underlying scans follows each source dataset’s own terms rather than republishing them. Because the benchmark targets physical camera-motion consistency, it is intended as a diagnostic for controllable and world-model generation and carries the standard dual-use considerations of generative-model research; it adds no generative capability beyond evaluation.

## N Scope and Interpretation Boundaries

Scope. EgoGenEval evaluates static indoor scenes, four dominant camera actions, and rollouts of at most three steps; it does not cover dynamics, physical interaction, long-horizon memory, or open-world navigation. Evaluator dependence. CMG depends on learned relative-pose estimation and per-transition GT-assisted translation-scale calibration. Direction-only and backend tests support broad rank stability, but magnitude conclusions remain calibration-dependent. SSP depends on frozen detection, target-derived prompts, correspondence, and monocular depth. Its mean-plus-worst aggregation intentionally emphasizes intermediate failure; matched aggregation shows that the large oficial CMG/SSP Chain-drop contrast is not invariant to the temporal functional. Ablations support broad ranks, while close ranks and Cycle correspondence remain evaluator-sensitive.

Observable objects. SSP covers only objects supported by the supplied views and formal visibility/matchability filters; it is not an exhaustive open-world inventory, and an unmatched detection is not proof of physical absence.

Human evidence. Human validation covers a six-system panel rather than all 18 evaluated systems. Agreement is strongest after case/protocol aggregation, so the metrics support aggregate system comparison rather than exact per-example grading.

Attribution and context. SSP is a joint target-view outcome that can fall through scene corruption, wrong pose, visibility, or evaluator error; pose-conditioned strata are observational and do not isolate state memory. The fixed-object Context-K decline does not causally estimate the value of extra context because interfaces, evidence sets, and behavior remain intertwined.

Evaluation scope and interfaces. Bootstrap intervals characterize benchmark sampling for one frozen output per model–step, not decoding variation. Source-removal analyses support broad rank stability, but source coverage is imbalanced and Matterport3D has only 22 cases. The benchmark compares complete systems through predefined native interfaces, whose image limits, resolutions, and decoding remain part of each system configuration.

Training probe. The SFT study contains one training run per variant. The Qwen clean/raw adapters were independently trained without an explicit seed, their continuations restore adapter weights without optimizer state, and the OmniGen2 matched base controls generation configuration rather than training randomness. Bootstrap intervals quantify paired benchmark sampling only. The two backbones support axis-selective score changes under pairwise supervision; repeated-seed claims and comparisons with free-running, state-aware, or full-pose objectives require matched experiments. Because later training steps use preceding physical targets, the present data do not establish how the same objectives behave under model-generated histories.

Data provenance and contamination. Evaluation scenes are disjoint from the SFT data; pretraining exposure to the source datasets remains unknown.

Reading model ranks. Scene-cluster resampling leaves several adjacent systems tied, so scores should be read as capability profiles and broad tiers, not exact ranks. Weight and sample-eficiency tests do not guarantee future rankings; pose-conditioned references receive privileged 6-DoF controls and remain separate.