## Highlights

R4Tun: LLM-guided adaptive segmental tunnel lining segmentation in point clouds Xinghui Tao,Zehao Ye,Guangming Wang,Jelena Ninić,Brian Sheil

• Propose R4Tun for label-free adaptation of tunnel lining segmentation pipelines.

• Integrate memory, state, and knowledge for multi-agent parameter tuning.

• Achieve mIoU of 0.784–0.796 on near-reference tunnel subsets across diferent LLMs.

• Raise overall mIoU from 0.18 to 0.43–0.48 without labels or retraining.

• Enable controlled cross-LLM adaptation without retraining the expert pipeline.

# R4Tun: LLM-guided adaptive segmental tunnel lining segmentation in point clouds

Xinghui Tao<sup>a</sup>, Zehao Ye<sup>b</sup>, Guangming Wang<sup>a,∗</sup>, Jelena Ninić<sup>b</sup> and Brian Sheil<sup>a</sup>

<sup>a</sup>Construction Engineering, University ofCambridge, Trumpington Street, Cambridge, CB2 1PZ, United Kingdom

<sup>b</sup>Department of Engineering, Durham University, Stockton Road, Durham, DH1 3LE, United Kingdom

## A R T I C L E I N F O

Keywords: Segmental tunnel lining Point cloud segmentation Tunnel inspection Large language models Parameter adaptation Multi-agent systems

## A BS T RA C T

Automated inspection of segmental tunnel linings requires adaptive segmentation from 3D point clouds, yet expert-tuned pipelines often degrade when tunnel conditions vary. This paper presents R4Tun, a large language model (LLM)-driven adaptation framework that extends an expert-designed pipeline (SAM4Tun) with bounded parameter tuning informed by structured context: memory (�), state (�), and knowledge (�). Evaluated on 30 selected Seg2Tunnel subsets (13 regular, 17 complex) across three LLMs, the full �+�+� design raised mean Intersection-over-Union (mIoU) from 0.18 to 0.43–0.48 and overall accuracy (OA) from 0.42 to 0.59–0.65 relative to the static SAM4Tun baseline, with the near-reference regular (staggered) subsets reaching mIoU 0.784–0.796 across LLMs. Across 270 (30 tunnels × 3 diferent LLMs × 3 context settings) runs, the LLMs showed similar parameteradjustment trends (with overlapping 95% CIs on mean gains) and consistently adjusted a shared set of critical parameters. These results support R4Tun as a controlled, label-free, cross-LLM adaptation mechanism in the tested SAM4Tun–Seg2Tunnel setting, demonstrating consistent accuracy gains; we position R4Tun as a mechanism contribution rather than a deployable final-inspection system, in which each bounded parameter change is auditable via logged rationales.

## 1. Introduction

Automated inspection of segmental tunnel linings is required to support structural health assessment and long-term operational safety, given the scale, frequency, and access constraints of modern tunnel networks [1]. In practice, engineers routinely encounter projects in which tunnel geometry, lining properties, and data acquisition settings vary [2, 3]. Furthermore, although modern laser scanning enables highfidelity tunnel capture, the resulting measurements often contain mixed structural elements, occlusions, and noise, making direct extraction of lining components challenging [4, 5]. Consequently, segmentation methods need to be adaptive enough to handle this complexity and variability [6, 7].

Existing tunnel point-cloud segmentation methods are grouped into three broad categories: feature engineering, deep learning and foundation-model pipelines (see Fig. 1 and Section 2). Feature engineering rules are interpretable but lack robustness under changed conditions [2]. Supervised deep-learning models generalise through data but require large labelled datasets and periodic retraining, while lacking auditability [8, 9, 10]. Foundation-model pipelines bypass annotation by using models pre-trained on large datasets [11, 12], but remain sensitive to expert-tuned processing parameters. Existing tunnel-segmentation approaches have not yet demonstrated, in a single workflow, both robust adaptation to changed conditions and an auditable adaptation process. Among foundation-model pipelines, SAM4Tun [13] combines point-cloud preprocessing with SAM-based prompting [12] and has shown strong performance under expert tuning. However, the pipeline is highly sensitive to its many processing parameters. When tunnel geometry or scanning conditions depart from the tuning reference, performance can degrade, and identifying which parameters need adjustment can require repeated expert intervention.

Given that recent LLMs can follow multi-step instructions over structured inputs, parameter adaptation can be formulated as a diagnostic task in which the model receives context and proposes bounded adjustments. LLMs have been applied to engineering workflows including chemical synthesis planning [14], multi-agent coordination for complex project tasks [15, 16, 17], and manufacturing decision support [18]. However, these studies do not address parameter adaptation in infrastructure inspection pipelines. In this study, we propose R4Tun, an LLM-guided adaptation system that extends the SAM4Tun pipeline by adjusting stage parameters in response to changing tunnel conditions. The framework provides each agent with structured context � + � + � (memory, state, and knowledge), from which it produces bounded parameter updates via stepwise prompting. We evaluate whether this contextual information improves segmentation adaptability across both regular and complex tunnels, and whether the resulting adaptation behaviour is consistent across LLMs. To isolate the contribution of the LLM beyond deterministic tuning, we also benchmark R4Tun against a rule-based adaptation derived from the same per-stage knowledge documents.

We position R4Tun as a mechanism contribution rather than a deployable final-inspection system: its strength lies in integrating LLM-driven bounded parameter adaptation with an expert-designed pipeline, with each change auditable through logged rationales. Absolute segmentation accuracy remains bounded by the open-source SAM4Tun ceiling and by the single expert reference used here, as discussed in Section 4.6.

![](images/8230bfac43645dc090bb43dc65b94d2362de0da1e0e99f02287d8f0e1cdc064e.jpg)  
Figure 1: Segmentation approaches arranged by their adaptation mechanisms. Expert-guided, LLM-driven methods can extend foundation-model pipelines by adding parameter adaptation with structured context and encoded expert knowledge. In this work, we test one such mechanism (R4Tun) while keeping the underlying SAM4Tun pipeline fixed.

This paper makes the following contributions:

1. A framework design in which LLM agents adapt the parameters of the SAM4Tun pipeline using structured context $m + s + k \colon$ memory of the reference configuration, state from intermediate pipeline outputs, and a knowledge document oftunnel category-specific configuration shared across all tunnels.

2. A controlled and auditable adaptation mechanism rather than a deployable final-inspection system, in which each bounded parameter update is accompanied by a logged rationale for post-hoc expert review.

3. Empirical evidence from cumulative ablation across 30 tunnels suggesting that intermediate pipeline state is an important contributor to adaptation, while the knowledge component adds a smaller increment concentrated on the complex category.

4. Cross-LLM evidence showing that three diferent LLMs (Opus-4.6, GPT-5.4, Gemini-3-Flash) produce overlapping efect ranges and adjust a consistent subset of critical parameters under the same prompt structure.

5. Evidence that deterministic, rule-based adaptation explains part, but not all, of the observed mIoU increase, which is consistent with an additional contribution from the LLM-based adaptation process in the tested setting.

The rest of this paper is organised as follows. Section 2 reviews related work. Section 3 presents methodology, including the R4Tun architecture, dataset, experimental design, and evaluation. Section 4 reports results. Section 5 discusses findings, implications, and limitations. Section 6 concludes.

## 2. Related work

This study situates R4Tun within tunnel point-cloud segmentation by contrasting (i) feature-engineered methods and supervised deep learning, (ii) foundation-model pipelines that reduce annotation but remain parameter-sensitive, and (iii) LLM reasoning as a way to produce auditable, bounded per-tunnel parameter updates.

## 2.1. Feature engineering versus deep learning

Tunnel point-cloud segmentation has advanced through two generations of methods, which trade of between auditability and adaptability. Feature-engineering approaches encode domain knowledge into deterministic geometric rules: centreline fitting via RANSAC [19], density-based surface clustering [20], and point-sampled surface processing (e.g., simplification and resampling) [21]. Those methods remain widely used in safety-critical inspection because their logic is explicit and auditable [2]. However, each rule embeds assumptions about tunnel geometry and point-cloud characteristics (diameter, ring spacing, joint pattern, point density), and changing any of these requires manual reconfiguration by a domain expert. Supervised deep-learning methods, such as PointNet++ [22], RandLA-Net [23], Mask3D [24], and TD3D [25], learn features directly from annotated point clouds and can generalise across similar conditions present in the training set. Recent work has also extended tunnel and infrastructure understanding with attention mechanisms: for 3D point clouds, UnrollingNet [26] unfolds the cylindrical surface into a structured 2D representation for attention-based semantic segmentation, and SerialFormer [27] uses transformerstyle attention to capture long-range context for pointcloud semantic segmentation; for 2D inspection imagery, TransUNet [28] leverages self-attention within a U-Net-style encoder–decoder for tunnel defect segmentation, and ECA-YOLO [29] augments a YOLO detector with eficient channel attention and tiling to improve multi-defect detection. Yet supervised deep-learning models require large labelled tunnel datasets that remain scarce [10], demand retraining for new tunnel conditions, and provide no mechanism for an engineer to trace or override a segmentation decision [9].

![](images/44c109431cdc918f46f34c0a0c0d24713601ed8633924e0f3ea37dabf543b4b3.jpg)  
Figure 2: The R4Tun multi-agent architecture for tunnel segmentation. Four blue-boxed stage agents (Stage 1: Unfolding, Stage 2: Denoising, Stage 3: Enhancing, and Stage 4: Segmenting) adapt their stage-specific parameters via LLM reasoning, passing intermediate outputs forward to produce the final segmentation. Arrows indicate the workflow direction from the raw point cloud through intermediate representations to 2D segmentation and 3D reprojection.

Therefore, for inspection operators, deploying to a new tunnel type requires either re-engineering rules, collecting new training data, or accepting degraded performance without a systematic diagnostic mechanism.

## 2.2. Foundation-model pipelines

To the best of our knowledge, direct 3D point-cloud segmentation of infrastructure at tunnel scene scale remains an open challenge for current foundation models. Existing 3D foundation models target common object recognition on curated datasets and do not transfer to large, noisy tunnel point clouds [30, 31, 32, 33]. As a result, practical tunnel pipelines often project 3D data into 2D representations to use mature vision foundation models pre-trained on largescale image datasets, thereby bypassing the annotation bottleneck [34, 35]. SAM [12] performs class-agnostic segmentation from spatial prompts without task-specific training, and extensions such as SAM2 [36] for temporal consistency and SEEM [37] for natural-language-guided segmentation are broadening prompt-based segmentation further. These models have been adopted in infrastructure contexts including crack detection [38, 39] and Scan-to-BIM workflows [40, 41, 42].

For tunnel linings, SAM4Tun [13] combines geometric preprocessing (unfolding, denoising, enhancing) with foundation-model 2D segmentation, achieving strong segmentation without training data. However, this pipeline depends on numerous stage-specific parameters that encode assumptions about the reference tunnel’s diameter, ring length, point density, and joint patterns. When a new tunnel departs from the reference, these parameters become misspecified and the pipeline can degrade without indicating which parameters fail or why. This sensitivity is not unique to SAM4Tun: most multi-stage pipelines that chain domainspecific preprocessing with a foundation model tend to inherit it. While foundation models have reduced reliance on labelled data, they have made the pipeline’s dependence on hand-tuned parameters more explicit, without removing the need for expert parameter tuning.

## 2.3. LLM reasoning

Recent LLM advances have introduced explicit reasoning capabilities relevant to the parameter-adaptation problem. Reasoning-oriented training [43, 44, 45, 46, 47, 48] produces models capable of decomposing complex problems and performing prompted consistency checks. Chain-ofthought (CoT) reasoning [49, 50] enables step-by-step logical traces. Multi-agent architectures coordinate specialised roles through shared context [16, 15, 17, 51, 52], and context engineering influences reasoning quality through the careful design of information provided to models [53, 54, 55].

In tunnelling segmentation, however, no prior work has applied LLM-based reasoning to a challenge in expertdesigned pipelines: re-tuning a parameter-sensitive system to new conditions while preserving the engineer’s ability to inspect and override each decision. We therefore introduce an LLM-based reasoning framework that retunes parametersensitive pipelines per tunnel using expert-guided context and a logged rationale for each proposed change. Meanwhile, whether LLM reasoning produces gains beyond what handcoded rules can deliver from the same knowledge text remains an open question. Our experimental design therefore tests this directly by including a deterministic rule-based adaptation of those documents as a non-LLM control.

![](images/c9ae0f618cf9cb9c2e2b8c02604659ae1bcfd62774f205d572379ddea76e026f.jpg)  
Figure 3: Stage 1: Unfolding. Left: fitted tunnel cross-section used to determine the reference centre and radius for cylindrica unfolding. Centre: 2D panoramic depth map derived from the 3D point cloud. Right: zoomed region of the unfolded map showing radial-distance encoding.

## 3. Methodology

Our methodology defines the tunnel-lining segmentation task, summarises the fixed SAM4Tun pipeline, and specifies the R4Tun adaptation layer (structured context � + � + � and bounded parameter updates), followed by the dataset, baselines, ablations, metrics, and sensitivity analysis.

## 3.1. Task definition

The task is semantic segmentation of segmental tunnel linings from terrestrial laser scanning (TLS) point clouds. Given a raw point cloud of a tunnel section, the goal is to assign each point a structural label: background, key segment (K), adjacent segments (B), and standard segments (A), with an additional A4 class for 7-segment complex tunnels. The unit of analysis is a tunnel subset spanning multiple rings selected from the Seg2Tunnel benchmark.

A key limitation is that the fixed configuration of the open-source SAM4Tun achieves an mIoU of 0.27 on regular tunnels, but only 0.04 on complex tunnels whose geometry departs from the tuning reference. As noted above, the pipeline provides no built-in signal indicating which parameters become misspecified.

## 3.2. R4Tun

R4Tun supplements the SAM4Tun pipeline [13] with an LLM-guided adaptation layer that adjusts parameters per tunnel without modifying the underlying algorithms (Fig. 2). Given a new tunnel point cloud, a characteriser first extracts raw geometric properties (diameter, density, coordinate ranges; full field list in Appendix 2). Each stage (Stage 1: Unfolding, Stage 2: Denoising, Stage 3: Enhancing, and Stage 4: Segmenting) has a dedicated LLM agent that adapts its parameters. In Stage 4, the adapted prompts are passed to SAM, and the resulting 2D masks are reprojected onto the 3D point cloud without further LLM intervention.

Stage 1: Unfolding (Fig. 3) establishes a cylindrical coordinate system for the tunnel. Similar cylindrical unfolding ideas have been explored for tunnel point-cloud semantic segmentation (e.g., UnrollingNet [26]). It slices the point cloud at regular longitudinal intervals, fits an ellipse to each cross-section via RANSAC [19], and selects the model with the highest inlier support. The per-slice ellipse centres are interpolated into a smooth centreline that defines the cylindrical frame (�, �, ℎ). The 3D tunnel surface is then projected into a 2D panoramic image in which each pixel encodes the radial distance to the centreline, producing a stable depth representation for downstream filtering and segmentation.

Stage 2: Denoising (Fig. 4) removes non-structural artefacts (rails, cables, and scattered points) that would otherwise interfere with segmentation. The algorithm applies grid-based radial-density filtering inspired by the DBSCAN clustering principle [20], grouping points by local density in cylindrical coordinates. Dense, connected regions are retained as tunnel surfaces while isolated points are rejected as noise. A pair of radial masks $( R _ { \mathrm { l o w } } , R _ { \mathrm { h i g h } } )$ constrains filtering to the expected tunnel radius, preservingjoint boundaries and ring edges while discarding outliers beyond the lining surface. The result is a clean, continuous surface representation suitable for curvature analysis and interpolation in the next stage.

Stage 3: Enhancing (Fig. 5) improves geometric continuity and surface completeness before projection into the 2D depth map. Local curvature is estimated on neighbourhood points to guide point insertion: new points are placed between neighbours whose curvature diference is small (smooth panel regions), while high-curvature areas (joint boundaries and ring edges) are preserved intact. A threestage progressive upsampling scheme inserts additional points at progressively finer resolutions to maintain surface continuity without blurring structural boundaries. Pixellevel interpolation then refines outlier points at joint locations. The enhanced surface is projected into a panoramic depth map that balances smooth panel regions with sharply defined joint boundaries.

![](images/39ed64cd061eacaf485c17bb2e2cc605ec41b9ee38a69dbd31525c0ac5480394.jpg)  
Figure 4: Stage 2: Denoising. Left: radial-distance distribution used to filter points outside the expected tunnel lining surface. Centre: unfolded 2D depth map after filtering, showing the cleaned point distribution. Right: zoomed region of the denoised map.

![](images/417671dc744eebf472949bf43939cfd1f45f5f3df0987ce1b72a84a859551442.jpg)  
Figure 5: Stage 3: Enhancing. Left: curvature-guided surface and boundary interpolation along the tunnel lining, showing original sampling, curvature-guided insertion, and progressive three-stage upsampling (shown as three sequential upsampling steps). Centre: unfolded 2D depth map after enhancement, where inserted points improve geometric continuity; the red box highlights a local region of interest. Right: zoomed views of the highlighted region, showing added points and pixel-level interpolation.

Stage 4: Segmenting (Fig. 6) combines boundary detection and SAM segmentation into a single workflow. First, a Hough-transform-based detector [56] extracts linear features from the enhanced depth map that correspond to ring joint boundaries. Detected lines are filtered by orientation (oblique, horizontal, vertical) with separate Hough thresholds and merged when closer than a distance tolerance. The confirmed boundaries are then used to construct templatebased prompts for key, adjacent, and standard segments, which are passed to SAM [12]. SAM produces 2D segment masks on the panoramic image, which are reprojected into 3D point labels via the stored pixel-to-point mapping from Stage 1.

The pipeline contains 81 tunable processing parameters, spanning geometric defaults, filtering and interpolation settings, and boundary-detection thresholds. All parameters were expert-tuned on a reference tunnel (mIoU = 0.88); full parameter tables are provided in Appendix 1. In this study, the underlying pipeline stages remain fixed across all conditions; only the parameter values change. The SAM4Tun baseline applies this single expert-tuned configuration uniformly to all 30 tunnels without per-tunnel adaptation, serving as the static baseline against which LLM-guided adaptation is measured.

## 3.3. Agent design

Each of the four stage agents is a self-contained unit comprising (i) a context (Section 3.3.1; Fig. 7a) and (ii) a Python analyst that constructs the full LLM prompt and parses the returned JSON. For each agent-driven stage, the analyst assembles a prompt from the current context level, calls one of the selected LLM APIs (Opus-4.6, GPT-5.4, or Gemini-3-Flash) to follow a five-step CoT protocol (Section 3.3.2; Fig. 7b): referencing, diagnostic inspection, parameter adaptation, validation, and JSON output. Each agent returns a single schema-conformant parameter JSON, which the corresponding pipeline stage executes unchanged. A characteriser plugin then extracts statistics from the stage output, updating the cumulative state available to subsequent stages. Each stage’s parameters are adapted from the expert-tuned reference configuration together with this cumulative state; the stage scripts and evaluation code are identical across all ablation conditions, with only the parameter JSONs changing. Because every parameter change is logged alongside a generated textual rationale, engineers can review and, where necessary, override adaptation decisions. A worked example is provided in Appendix 5. After the final segmentation, per-point labels are evaluated against ground truth using mean Intersection over Union (mIoU) as the primary metric.

![](images/1dbd9ac855cb1a4130e582226b3299430acd4c7a92acc2d3fcda262f391e3ac2.jpg)  
Figure 6: Stage 4: Segmenting. The segmenting stage applies Hough-transform line detection [56] to identify ring boundaries, then constructs template-based prompts and feeds them to SAM [12]. SAM produces 2D segment masks that are reprojected into 3D. K denotes the key segment, B denotes adjacent segments, and A denotes standard segments around the ring.

## 3.3.1. Context design

Each stage agent receives structured context comprising three components: memory, state and knowledge (Fig. 7a). A worked example of the three components for the denoising agent is given in Appendix 4.

Memory (Appendix 4.1) stores the reference tunnel’s characteristics (a JSON file containing geometry, point density, coordinate ranges, nearest-neighbour distances) alongside the expert-tuned reference parameters. The agent compares the current tunnel’s characteristics against this reference to quantify deviation and adjust its parameters via CoT reasoning (Section 3.3.2). Memory is static: it does not change between stages.

State (Appendix 4.2) captures cumulative geometric and statistical properties after each pipeline stage executes. These JSON summaries are injected into subsequent agents prompts, and the state grows as stages execute. Importantly, state alone does not prescribe parameter changes: the numeric summaries of the actual point distribution after upstream processing must be interpreted in the context of parameter semantics and tunnel conditions before they can inform parameter updates.

Knowledge (Appendix 4.3) supplies domain-specific guidance in human-readable markdown documents. Each stage has its own knowledge document covering three types of guidance: (i) cross-tunnel variation, which describes common lining layouts and structural conditions; (ii) parameter notes, which define each tunable parameter, its empirically validated range, and proven defaults; and (iii) diagnostic rules, which provide constrained guidelines for recommended parameter adjustments. Knowledge is authored once and shared across all tunnels.

![](images/890daeedf483ac6a55f4da66af5986cfb832dee1850d94dd645c3f987c0f8afe.jpg)  
Figure 7: The left column (a) shows R4Tun stage agent architecture. Each agent maintains a shared and cumulative context that informs an LLM analyst component performing CoT reasoning. The right column (b) shows the five CoT phases: (1) Referencing, (2) Diagnostic inspection, (3) Parameter adaptation, (4) Validation, and (5) JSON output.

## 3.3.2. CoT design

CoT reasoning provides the analytical structure for each agent’s decision-making (Fig. 7b). The agent is prompted to follow a five-step protocol (Algorithm 1). A worked example of the full five-step trace is provided in Appendix 5.

1. Referencing: The agent compares the current tunnel’s characteristics against the stored reference to quantify deviation (e.g., +39% diameter increase).

2. Diagnostic inspection: The agent attributes the deviation to a plausible cause and identifies which specific tunable parameters are implicated. If signals conflict, the agent is instructed to prioritise State over Memory, as State summarises geometry after upstream processing.

3. Parameter adaptation: The agent proposes updates for the implicated parameters. These updates are instructed to stay within the empirical ranges defined in the Knowledge component.

4. Validation: A self-correction step performs a consistency check to confirm that the proposed values satisfy logical constraints (e.g. a radius lower bound must be smaller than the corresponding upper bound). If a constraint is violated, the value is prompted to move toward to the nearest valid bound. This checkand-correction is executed as an LLM reasoning step, rather than a deterministic software routine, and therefore does not guarantee constraint satisfaction.

5. JSON output: The selected parameters and their reasoning trace are packaged into a single schemaconformant JSON object.

Each stage agent reads the tunnel characteristics and the latest pipeline state to identify deviations and decide which parameters to adjust. Starting from the expert reference value, it computes an updated setting from the post-stage statistics and clips it to the empirically observed range in the knowledge document when needed. This difers from the deterministic rule baseline (Appendix 3), which assigns each tunnel to a discrete category based on raw characteristics and then applies one fixed parameter value per family. By contrast, the LLM diagnoses parameters on the fly, resolves conflicts between live state and static memory, and adapts values continuously from the observed statistics.

Notation for Algorithm 1. $C _ { \mathrm { t a r g e t } }$ denotes the raw characteristics of the current tunnel, while � stores the reference configuration $( C _ { \mathrm { r e f } } , P _ { \mathrm { r e f } } )$ . � is the cumulative pipeline state, and $S _ { \mathrm { c u r r e n t } }$ is its most recent summary. � is a shared knowledge document that encodes parameter semantics, per-parameter admissible bounds �, and cross-parameter constraints �. The adaptation procedure computes a geometric deviation descriptor $\Delta _ { \mathrm { g e o m } } ,$ selects an implicated parameter subset $P _ { \mathrm { i m p l i c a t e d } } ,$ proposes an updated parameter map $P _ { \mathrm { p r o p o s e d } }$ (indexed as $P _ { \mathrm { p r o p o s e d } } [ p ] )$ , and finally returns the schemaformatted JSON object $P _ { \mathrm { a d a p t e d } } .$

Algorithm 1 LLM-driven Parameter Adaptation via CoT   
Require: Target tunnel raw characteristics $C _ { \mathrm { t a r g e t } }$   
Require: Memory � (reference characteristics $C _ { \mathrm { r e f } } ,$ , refer  
ence parameters $P _ { \mathrm { r e f } } )$   
Require: State � (cumulative pipeline outputs / intermedi   
ate summaries)   
Require: Knowledge � (parameter semantics, tunable   
bounds �, and cross-parameter constraints $R )$   
Ensure: Adapted stage parameter JSON $P _ { \mathrm { a d a p t e d } }$   
1: Phase 1: Referencing   
2: $\Delta _ { \mathrm { g e o m } }  \mathrm { C o m p a r e } ( C _ { \mathrm { t a r g e t } } , C _ { \mathrm { r e f } } )$ ⊳ Quantify deviation   
from reference   
3: Phase 2: Diagnostic inspection   
4: $S _ { \mathrm { c u r r e n t } }  \mathrm { E x t r a c t L a t e s t } ( S )$ ⊳ Most recent stage   
statistics   
5: Reconciliation: If $S _ { \mathrm { c u r r e n t } }$ conflicts with �, prioritise   
$S _ { c }$ urrent   
6: $P _ { \mathrm { i m p l i c a t e d } }$ ← IdentifyParameters $( \Delta _ { \mathrm { g e o m } } , S _ { \mathrm { c u r r e n t } } , K )$ ⊳   
Select parameters relevant to observed deviations   
7: Phase 3: Parameter adaptation   
8: $P _ { \mathrm { p r o p o s e d } }  \emptyset$   
9: for each parameter $p \in P _ { \mathrm { i m p l i c a t e d } }$ do   
10: $P _ { \mathrm { p r o p o s e d } } [ p ]  \mathrm { A d j u s t } ( p , \Delta _ { \mathrm { g e o m } } , S _ { \mathrm { c u r r e n t } } , K )$ ⊳   
Propose new value within � guided by �   
11: end for   
12: Phase 4: Validation   
13: for each parameter $p \in P _ { \mathrm { p r o p o s e d } }$ do   
14: $P _ { \mathrm { p r o p o s e d } } [ p ]  \mathrm { S e l f V }$ alidate $\left( P _ { \mathrm { p r o p o s e d } } [ p ] , p , K \right)$ ⊳   
Execute an LLM internal reasoning step conditioned on   
� (semantics, bounds spec �, constraints $R ) .$   
15: end for   
16: Phase 5: JSON output   
17: $P _ { \mathrm { a d a p t e d } }  \mathrm { F o r m a t A s J S O N } ( P _ { \mathrm { p r o p o s e d } } )$   
18: return $P _ { \mathrm { a d a p t e d } }$

## 3.4. Experimental setup

## 3.4.1. Dataset

We selected 30 tunnel subsets from the Seg2Tunnel benchmark (Table 1). These subsets cover the main variations in segmental tunnel geometry (diameter, ring length, segment count, key-segment layout, and point-density distribution), providing a controlled basis for evaluating adaptation across tunnel categories.

As shown in Fig. 8, staggered and continuous tunnels are grouped as “regular” (� = 13): they share the 5.60 m nominal diameter, 1.2 m ring spacing, six segments per ring, and, most importantly, regular patterns of key-segment positions similar to the reference. Complex tunnels $( n = 1 7 )$ depart from that reference in multiple coupled ways: nominal inner diameter increases by 34% (5.60 m → 7.50 m), ring length by 50% (1.2 m → 1.8 m), and each ring adds a seventh segment with an interleaved key-segment arrangement that does not repeat ring-to-ring. In addition, of-axis scanning yields non-uniform sampling, partial circumferential coverage, and angle-dependent range and incidence. Together, these diferences alter both the geometry visible in the point cloud and the semantic targets the pipeline must recover. Consequently, a fixed parameter set cannot be assumed to transfer from the regular reference; these conditions constitute the primary stress test for whether R4Tun can adapt the same algorithmic pipeline to of-design tunnel conditions. Ground-truth labels are per-point structural class annotations provided by the Seg2Tunnel benchmark; they are used for evaluation only and are not provided to the LLM or pipeline during adaptation.

## 3.4.2. Experimental design

The experiment follows a cumulative ablation design with four conditions, each adding one context component (Table 2). This design isolates the cumulative contribution of each component while maintaining a controlled comparison. Level 0 (SAM4Tun) is the fixed expert-tuned baseline with no LLM involvement. Levels 1–3 progressively provide the LLM with richer context, enabling the cumulative ablation to attribute improvements to specific components. A separate Non-LLM condition (level 0a in Table 2) is included as the Non-LLM adaptive control, allowing the LLMs’ contribution to be read directly from the gap between the Non-LLM column and the LLM columns of Table 5. We codified the per-stage knowledge documents into a deterministic Python rule table as a non-LLM baseline. It maps characterisation fields (e.g., diameter, ring length, segments-per-ring, joint type, point density, and station configuration) to the 18 critical parameters adjusted by the LLMs (Table 9) using explicit $\mathrm { ^ { 6 6 } i f . . . }$ then . . . ” rules. The table and script are fixed for all 30 tunnels (regular and complex), with no evaluationset tuning or subset-specific settings, isolating the value of LLM per-tunnel reasoning. The complete rule specification is provided in Appendix 3.

To assess whether adaptation behaviour depends on a specific LLM, each condition was run with three diferent LLMs (Opus-4.6, GPT-5.4, and Gemini-3-Flash) under identical prompts. Across models, the pipeline code, evaluation scripts, and prompt structure were held constant, while the context content varied by ablation level. Other than setting temperature to 0 to minimise stochasticity (Table 3), all other API settings used vendor defaults. This corresponds to 270 unique LLM-adaptation configurations: 30 tunnels $\times \textsf { 3 } \mathrm { L L M s } \times \textsf { 3 }$ context settings (m, m+s, m+s+k). Each configuration was executed a second time for repeatability checking, yielding $2 7 0 \times 2 = 5 4 0$ pipeline executions, plus 30 baseline runs with fixed parameters. Headline analyses (e.g., Table 5) use the first execution only. All three LLMs were accessed via their respective commercial APIs (Table 3). No model-specific prompt tuning was performed.

Pipeline execution used a single NVIDIA RTX 5060 GPU. Adapted parameter files and CoT reasoning traces were logged for all runs, enabling post-hoc sensitivity analysis (Section 3.4.4).

![](images/d202868198c38e9fe0af1804eb6201a4fd69dbe74a81429afd00cd56ef360951.jpg)  
Regular (staggered)

![](images/0dee033d8172cf70fbfd185cb57f89270094108f18185bfa4c4fea756812bc84.jpg)  
Regular (continuous)

![](images/3b258a1bf9b37de508ab80a13e52ec737faff395a081a552027178316284258e.jpg)  
Complex  
Figure 8: Tunnel lining pattern categories. From left to right: regular (staggered) tunnels exhibit a repeating ring-to-ring pattern; regular (continuous) tunnels maintain a fixed segment order around each ring; complex tunnels show no consistent repeating pattern.

Table 1  
Dataset properties.
<table><tr><td rowspan="2">Property</td><td colspan="2">Regular (T1, T2, T3)</td><td rowspan="2">Complex (T4, T5)</td></tr><tr><td>Staggered (T1, T2)</td><td>Continuous (T3)</td></tr><tr><td>Inner diameter</td><td>5.5m</td><td>5.9 m</td><td>7.5 m</td></tr><tr><td>Ring length</td><td>1.2 m</td><td>1.2 m</td><td>1.8 m</td></tr><tr><td>Segments/ring</td><td>6</td><td>6</td><td>7</td></tr><tr><td>Joint type</td><td>Staggered</td><td>Continuous</td><td>Complex interleaved</td></tr><tr><td>Key-segment position</td><td>A ring-to-ring repeating pattern</td><td>A fixed ring-wise order</td><td>No repeating pattern</td></tr><tr><td>Scanning</td><td>Single-station</td><td>Multi-station</td><td>Single-station, offset</td></tr><tr><td>Eval. schema</td><td>6-class</td><td>6-class</td><td>7-class</td></tr><tr><td>Count</td><td>10 subsets</td><td>3 subsets</td><td>17 subsets</td></tr></table>

Table 2  
Ablation conditions.
<table><tr><td>Level</td><td>Code</td><td>Condition</td><td>What the LLM sees</td></tr><tr><td>0</td><td>SAM4Tun</td><td>Baseline</td><td>Default parameters</td></tr><tr><td>0a</td><td>Non-LLM</td><td>Rule table</td><td>18 parameters set by lookup</td></tr><tr><td>1</td><td>m</td><td>Memory</td><td>Reference characteristics +</td></tr><tr><td>2</td><td>m+s</td><td>Memory+State</td><td>parameters + intermediate pipeline outputs</td></tr><tr><td>3</td><td>m+s+k</td><td>Memory + State + Knowledge</td><td>+ domain knowledge</td></tr></table>

## 3.4.3. Evaluation metrics

Segmentation quality is measured primarily by mean Intersection-over-Union (mIoU). For class �,

$$
\mathrm { I o U } _ { c } = \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F P } _ { c } + \mathrm { F N } _ { c } } , \qquad \mathrm { m I o U } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } \mathrm { I o U } _ { c } ,\tag{1}
$$

where � is the number of segmental segments (classes) within one ring $( C ~ = ~ 6$ for regular tunnels; $C \ = \ 7$ for complex tunnels). Equation (1) defines the per-tunnel mIoU (the mean IoU over the � classes of a single tunnel); unless stated otherwise, mIoU values reported for a group of tunnels are the arithmetic mean of these per-tunnel scores across the relevant subsets. As a complementary global metric we also report overall accuracy (OA),

Table 3  
LLM configuration.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Models</td><td>Opus-4.6, GPT-5.4, Gemini-3-Flash</td></tr><tr><td>Max tokens</td><td>16,384</td></tr><tr><td>Temperature</td><td>0 (override to minimise stochasticity)</td></tr><tr><td>Timeout</td><td>300 s per call</td></tr><tr><td>Prompt format</td><td>Markdown with JSON code</td></tr><tr><td>Failure handling</td><td>JSON extraction failure</td></tr></table>

$$
\mathrm { O A } = \frac { \sum _ { c = 1 } ^ { C } \mathrm { T P } _ { c } } { N } ,\tag{2}
$$

where � is the total number of points. Because OA can be dominated by majority classes, we treat mIoU as the primary score. We also report per-class IoU to assess whether improvements are broad or concentrated in specific structural classes.

Segmentation performance on the reference tunnel: expert configuration versus LLM-adapted (� + � + �).
<table><tr><td>Metric</td><td>Expert</td><td>GPT-5.4</td><td> $\mathsf { O p u s } { \mathsf { - } } 4 . 6$ </td><td>Gemini-3-Flash</td></tr><tr><td>OA</td><td>0.9369</td><td>0.9390</td><td>0.9419</td><td>0.9354</td></tr><tr><td>F1</td><td>0.9361</td><td>0.9389</td><td>0.9431</td><td>0.9362</td></tr><tr><td>mloU</td><td>0.8801</td><td>0.8852</td><td>0.8926</td><td>0.8805</td></tr></table>

For each condition and LLM, we computed paired pertunnel improvements as $\Delta _ { i } = \mathrm { m I o U } _ { \mathrm { c o n d i t i o n } , i } \mathrm { - m I o U }$ <sub>baseline,�</sub>, with $\ n \ = \ 1 3$ for regular tunnels, $n ~ = ~ 1 7$ for complex tunnels, and $n ~ = ~ 3 0$ overall. We summarise these paired improvements using three standard statistics: (i) two-sided paired �-test �-values $( \alpha = 0 . 0 5 )$ for statistical significance, (ii) paired Cohen’s � for efect size, and (iii) bootstrap 95% confidence intervals (CIs) for the mean improvement.

## 3.4.4. Sensitivity analysis

To assess the robustness of adapted parameters, for each parameter, we report the coeficient ofvariation (CV):

$$
\mathrm { C V } = { \frac { s } { | { \bar { v } } | } } ,\tag{3}
$$

where $\begin{array} { r c l } { \bar { v } } & { = } & { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } v _ { i } } \end{array}$ and $\begin{array} { r c l } { s } & { = } & { \sqrt { \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N } ( v _ { i } - \bar { v } ) ^ { 2 } } } \end{array}$ are the mean and standard deviation of the adapted values $\{ v _ { i } \} _ { i = 1 } ^ { N }$ across $N = 3 0$ tunnels. CV measures how much a parameter changes from tunnel to tunnel: (i) a high CV (≥ 0.06) identifies parameters that are tunnel-responsive; (ii) CV ≈ 0 indicates baseline corrections that take the same improved value on every tunnel regardless of geometry. We further examine cross-LLM consistency: for parameters that always trigger adaptation, we compare the adapted values, tunnel counts, and tunnel categories produced by each LLM independently.

## 4. Results

By comparing segmentation performance against the static SAM4Tun baseline and a deterministic non-LLM control, we quantify the contributions of memory (m), state (s), and knowledge (k) across LLMs, identify consistently adapted parameters, and analyse residual errors and runtime trade-ofs.

## 4.1. Overall performance

Before assessing adaptation across the 30 diverse subsets, we verified the corrected SAM4Tun implementation on the single curated tuning tunnel. Under the expert-tuned configuration, it reaches mIoU ≈ 0.88 (Table 4); under the same $m + s + k$ adaptation, all three LLMs match or slightly surpass this expert anchor.

Table 5 summarises mIoU across conditions and LLMs. Under the full m+s+k design, overall mIoU rises from 0.18 (baseline) to 0.43–0.48 across the three LLMs $( p \ <$ 0.0001, paired Cohen’s $d = 1 . 5 1 \mathrm { - } 1 . 9 5 )$ ; overall OA rises from 0.42 to 0.59–0.65. The improvement holds across both tunnel categories (Fig. 10). Regular tunnels improve from 0.27 to 0.68–0.71 (paired Cohen’s $d = 1 . 4 – 2 . 2 )$ . Within the regular category, the near-reference staggered subsets (T1, T2), which match the reference tunnel in diameter, ring spacing, and key-segment layout, reach mIoU 0.784–0.796 across LLMs, whereas the continuous subset (T3), which departs from the reference in key-segment ordering, reaches only mIoU 0.270–0.309 and remains the main source of the lower combined regular figure. Complex tunnels, where the baseline drops to mIoU = 0.04 because several pipeline assumptions no longer match the tunnel conditions, improve to 0.15–0.20 (paired Cohen’s $d = 1 . 2 – 2 . 5 )$ ; complex-tunnel OA rises from 0.30 to 0.33–0.42 (Fig. 9).

Wall-clock runtime, API-call counts, and per-tunnel input/output token usage with the corresponding indicative USD cost for each LLM under m+s+k are reported in Appendix 6 (Table 19). Per-class IoU analysis (Appendix 8) supports broad improvement across structural classes for regular tunnels and progressive recovery from near-zero baselines for complex tunnels. The full performance distribution is summarised in Appendix 9.

Comparison against the non-LLM adaptation. The rulebased adaptation (Table 5, Non-LLM) increases overall mIoU from 0.18 to 0.25, which shows some gains are deterministic.

On complex tunnels, the rules raise mIoU from 0.04 to 0.15 by applying the large-tunnel specification (diameter, ring spacing, segment count, key-segment layout). The LLM then adds a smaller lift to 0.15–0.20 (10/17 improved). The largest gap is on regular tunnels, where the LLM improve the mIoU from 0.27 to 0.68–0.71. Both non-LLM and LLM approaches remain low in absolute terms on complex tunnels under the single-reference setup.

## 4.2. Ablation analysis

To isolate each context component’s contribution, Table 6 reports cumulative mIoU gains at each ablation step. Fig. 10 visualises the step-wise progression for regular and complex categories across all three LLMs.

Memory alone provides an initial reference but is insufficient. On regular tunnels, especially the staggered subsets, memory alone changes mIoU by 0.04–0.07 across the three LLMs (Table 6); all three LLMs show small positive averages. Without intermediate feedback, the agent can propose parameter updates but cannot verify whether they improve or degrade the pipeline outputs.

State accounts for the largest observed increment in the ablation study. The m+s condition yields the strongest gain relative to baseline across all three LLMs (all $p \_ <$ 0.0001; paired Cohen’s $d = 1 . 3 2 – 1 . 7 7 )$ , while the memoryonly efect is small or inconsistent, indicating that state is the main contributor to the observed improvement. State provides the agent with explicit quantitative evidence of how each stage has transformed the data: radial percentiles for mask bounds, retention rates for denoising aggressiveness, coverage uniformity for upsampling targets. Without state, the agent has only pre-pipeline statistics that may not reflect conditions after unfolding, denoising, or enhancing. The same conclusion holds when the comparator is the Non-LLM baseline rather than the static SAM4Tun: state alone (m+s) lifts overall mIoU by 0.16–0.25 above the rule-based adaptation.

![](images/3f4cc7e7dc674b306217004cc23abc5fbaba62e510769e3b24b256614a9b977d.jpg)  
Figure 9: Point-cloud visualisation for the tunnels with the largest mIoU gain in each category: regular (staggered; 2-1), regular (continuous; 3-3-1), and complex (5-1)  
. Top (a): SAM4Tun baseline; bottom (b): LLM-adapted pipeline (m+s+k, Opus-4.6). Colours indicate classes; red denotes error; mIoU is shown per example.

One plausible explanation is that the gain associated with state depends on how the model maps raw numeric summaries (percentiles, counts, ratios) to parameter values in light of parameter semantics. We did not test alternative mapping strategies (e.g., regression models); however, the continuous, multidimensional nature of the characteristic space and the non-trivial parameter interactions suggest that capturing this mapping through static rules alone would require considerable manual engineering efort per tunnel category.

Knowledge provides a small additional gain, concentrated in the complex category. On top of m+s, knowledge raises mIoU by up to 0.05 across the three LLMs. The mean increments are small, but the per-tunnel direction is nevertheless positive. The increment is concentrated where the knowledge document supplies specific tuning guidance that neither memory nor state can reliably infer from numerical summaries alone (e.g., a larger-diameter tunnel often uses wider rings).

## 4.3. Cross-model consistency and repeatability

To assess whether the adaptation behaviour is modeldependent under a fixed prompt and input structure, Table 7 compares the three LLMs on the full m+s+k condition. The three models show similar efect ranges under the full m+s+k condition, although overlapping 95% confidence intervals do not by themselves rule out between-model differences. A repeatability check under the full m+s+k setting indicates low but non-zero stochastic drift. Across 30 tunnels, each LLM was inferred twice: on average, 90.9% of the 18 critical parameters remain unchanged, and the mean absolute inter-run ΔmIoU is 0.029 (median 0.000), as reported in Table 20 (Appendix 7).

![](images/708ace75860ca9d1634abd4c9eb35e3c0c7f1a3b4e2511f97eb9a6df5f800339.jpg)

![](images/b5f438e9723b14bd249b48a4ec79d2b197550d7bffe496fb7636114368a86229.jpg)  
Figure 10: Step-wise adaptation results split by tunnel categories.Bars left to right: SAM4Tun (static baseline), Non-LLM (rulebased adaptation), m, m+s, m+s+k. The rule baseline closes part of the gap on complex tunnels but stays flat on regular tunnels; LLM (m+s+k) is highest in both categories. Error bars are bootstrap 95% CIs.

Table 5  
Overall segmentation results (� = 30): mIoU and paired ΔmIoU vs. baseline with �-values, efect sizes, and bootstrap 95% CIs. Δ is paired against SAM4Tun. OA is reported as a secondary metric, shown as the first line in each condition segment.
<table><tr><td>Condition</td><td>Statistic</td><td>Non-LLM</td><td>Opus-4.6</td><td>GPT-5.4</td><td>Gemini-3-Flash</td></tr><tr><td rowspan="2">Baseline (SAM4Tun)</td><td>Mean OA</td><td>0.42</td><td>0.42</td><td>0.42</td><td>0.42</td></tr><tr><td>mloU</td><td>0.18</td><td>0.18</td><td>0.18</td><td>0.18</td></tr><tr><td rowspan="7">Non-LLM</td><td>Mean OA</td><td>0.48</td><td></td><td></td><td></td></tr><tr><td>mloU</td><td>0.25</td><td></td><td></td><td></td></tr><tr><td>Mean ∆mloU</td><td>+0.08</td><td></td><td></td><td></td></tr><tr><td>p-value</td><td>0.0035</td><td></td><td></td><td></td></tr><tr><td>paired Cohen&#x27;s d</td><td>0.58</td><td></td><td></td><td></td></tr><tr><td>std (∆)</td><td>0.13</td><td></td><td></td><td></td></tr><tr><td>95% CI</td><td>[0.028, 0.128]</td><td></td><td></td><td></td></tr><tr><td rowspan="6">m</td><td>Mean OA</td><td></td><td>0.46</td><td>0.48</td><td>0.43</td></tr><tr><td>mloU</td><td></td><td>0.22</td><td>0.25</td><td>0.22</td></tr><tr><td>Mean ∆mloU</td><td></td><td>+0.05</td><td>+0.07</td><td>+0.04</td></tr><tr><td>p-value</td><td></td><td>0.558</td><td>0.028</td><td>0.056</td></tr><tr><td>paired Cohen&#x27;s d</td><td></td><td>0.11</td><td>0.42</td><td>0.36</td></tr><tr><td>95% Cl</td><td></td><td>[-0.113, 0.207]</td><td>[0.008, 0.134]</td><td>[-0.002, 0.090]</td></tr><tr><td rowspan="6">m+s</td><td>Mean OA</td><td></td><td>0.60</td><td>0.58</td><td>0.63</td></tr><tr><td>mloU</td><td></td><td>0.42</td><td>0.40</td><td>0.47</td></tr><tr><td>Mean ∆mloU</td><td></td><td>+0.25</td><td>+0.23</td><td>+0.29</td></tr><tr><td>p-value</td><td></td><td>&lt; 0.0001</td><td>&lt; 0.0001</td><td>&lt; 0.0001</td></tr><tr><td>paired Cohen&#x27;s d</td><td></td><td>1.77</td><td>1.32</td><td>1.46</td></tr><tr><td>95% Cl</td><td></td><td>[0.196, 0.300]</td><td>[0.163, 0.291]</td><td>[0.215, 0.363]</td></tr><tr><td rowspan="6">m+s+k</td><td>Mean OA</td><td></td><td>0.59</td><td>0.62</td><td>0.65</td></tr><tr><td>mloU</td><td></td><td>0.43</td><td>0.45</td><td>0.48</td></tr><tr><td>Mean ∆mloU</td><td></td><td>+0.25</td><td>+0.27</td><td>+0.30</td></tr><tr><td>p-value</td><td></td><td>&lt; 0.0001</td><td>&lt; 0.0001</td><td>&lt; 0.0001</td></tr><tr><td>paired Cohen&#x27;s d</td><td></td><td>1.95</td><td>1.95</td><td>1.51</td></tr><tr><td>95% Cl</td><td></td><td>[0.205, 0.301]</td><td>[0.222, 0.327]</td><td>[0.228, 0.378]</td></tr></table>

Table 6  
Cumulative mIoU contribution (mean Δ vs. previous level).
<table><tr><td>Transition</td><td>Opus-4.6</td><td>GPT-5.4</td><td>Gemini-3-Flash</td></tr><tr><td>Baseline → Non-LLM</td><td>+0.08</td><td>+0.08</td><td>+0.08</td></tr><tr><td>Baseline → m</td><td>+0.05</td><td>+0.07</td><td>+0.04</td></tr><tr><td> $\mathbf { m }  \mathbf { m } { + } \pmb { \mathscr { s } }$ </td><td>+0.20</td><td>+0.16</td><td> $+ 0 . 2 5$ </td></tr><tr><td> $\mathsf { m } + \mathsf { s } \to \mathsf { m } + \mathsf { s } + \mathsf { k }$ </td><td>+0.00</td><td>+0.05</td><td>+0.01</td></tr></table>

Table 7  
Cross-model summary (m+s+k vs baseline, overall).
<table><tr><td>LLM</td><td>Mean ∆mloU</td><td>95% Cl</td><td>d</td></tr><tr><td>Opus-4.6</td><td>+0.25</td><td>[0.21, 0.30]</td><td>1.95</td></tr><tr><td>GPT-5.4</td><td>+0.27</td><td>[0.22, 0.33]</td><td>1.95</td></tr><tr><td>Gemini-3-Flash</td><td>+0.30</td><td>[0.23, 0.38]</td><td>1.51</td></tr></table>

Table 8  
Runtime trade-of of the tested R4Tun m+s+k setting.
<table><tr><td>LLM</td><td>Extra time</td><td>Total time</td><td>Mean ∆mloU</td></tr><tr><td>Gemini-3-Flash</td><td> $+ 9 6 \thinspace \mathsf { s } \ ( 0 . 4 \mathsf { x } )$ </td><td>~331 s</td><td>+0.30</td></tr><tr><td>Opus-4.6</td><td> $+ 1 4 0 \thinspace 5 \thinspace \left( 0 . 6 \times \right)$ </td><td>~375 s</td><td>+0.25</td></tr><tr><td>GPT-5.4</td><td> $+ 3 0 7 \thinspace \mathsf { s } \left( 1 . 3 \mathsf { x } \right)$ </td><td>~542s</td><td>+0.27</td></tr></table>

## 4.4. Runtime trade-of

R4Tun trades additional API latency for bounded, logged parameter decisions. In this implementation, the SAM4Tun processing takes approximately 235 s per tunnel, and the LLM adaptation under m+s+k adds 96–307 s depending on the LLM, for total times of roughly 331–542 s per tunnel (Table 8). The added runtime is therefore not negligible, but it is modest relative to the cost of annotating data or retraining a supervised model.

## 4.5. Parameter sensitivity

This analysis separates parameters that vary across tunnels from those that remain constant, clarifying which aspects of the adaptation respond to specific tunnel conditions.

Analysis of 270 adapted parameter runs identified 18 parameters that all three LLMs consistently adjust (Table 9). Of these, 11 are tunnel-responsive $\mathrm { ( C V ~ \geq ~ 0 . 0 6 ) }$ , in that their adapted values vary with tunnel geometry, clustering by tunnel category. The remaining 7 parameters behave as baseline corrections $\left( \mathrm { C V } \approx 0 \right)$ , with near-identical values across tunnels, indicating a shared correction trend relative to the SAM4Tun defaults.

## 4.6. Error analysis

To understand why the absolute metrics stay low, we sorted every predicted point into four outcomes against the ground truth: correct;false negative (a segment point left as background, i.e. under-segmentation);false positive (a background point pulled into a segment, i.e. over-segmentation, rare here); and class swap (a correctly located segment point assigned to the wrong segment type).

Regular and complex tunnels fail diferently (Table 10). On regular tunnels, adaptation reduces under-segmentation, lowering FN from 22% to 6% of ground-truth points, and reduces class swaps from 20% to 7%. On complex tunnels, the baseline predicts almost all points as background; adaptation recovers the blocks (FN 70%→21%). However, the fixed segment-ofset template (anchored at K) turns a per-ring angular shift into systematic adjacent-class swaps (0%→32%). Fig. 12 shows this spatially: after adaptation, swaps dominate the residual error on complex tunnels.

The failure modes trace back to a mismatch between how the current open-source SAM4Tun pipeline is parameterised and how segmental rings actually vary. The pipeline applies one parameter set per tunnel and labels each ring by first detecting the key segment (K) and then placing every other segment at a fixed angular ofset from K, in a fixed order. Neither of these two structural constraints can be removed by tunnel-level parameter tuning.

1. Per-ring density variation under tunnel-level parameterisation. R4Tun adapts one configuration per tunnel, but point density is far from uniform within a tunnel: the per-ring point count varies by up to 19-fold (regular) and 38-fold (complex) between the dense central rings and the sparse end rings. A single tunnel-level setting cannot keep both ends of this range inside the detector’s working band, so the sparse rings systematically lose ring-boundary and K detection no matter how the tunnel-level parameters are chosen. This is a limitation of the adaptation scope (one configuration per tunnel), not of any individual parameter value, and it can only be lifted by moving from tunnel-level to ring-level adaptation.

2. Fixed segment-ofset template. Once K is located, the rule places the B segments one step on either side of K and the A segments at successive multiples of that same fixed step, in a fixed order (Fig. 11a). The step size and the segment order are hard-coded, so when the true geometry departs from the reference the fixed steps no longer line up with the real joints and correctly detected points are still given the wrong label. This is the single largest source of class swaps, visible as the red bands in the right column of Fig. 12, which grow from 10% (regular) to 42% (complex) even where points are correctly located.

Lifting these limits is future work rather than a parameter change, because they are baked into SAM4Tun’s one-configuration-per-tunnel, detect-K-then-fixed-template logic. The clearest direction is ring-level adaptation: densityadaptive preprocessing and per-ring K re-anchoring would address the density and K-detection problems above; a variable-length segment template and handedness inference would relax the fixed labelling rule. Implementing this requires rewriting the open-source pipeline and lies outside the bounded-parameter adaptation studied here. These directions extend SAM4Tun and R4Tun rather than invalidate

Table 9  
Critical parameters identified across all three LLMs (18 total). Tunnel-responsive parameters $( \mathsf { C V } \geq 0 . 0 6 )$ vary with tunnel geometry; baseline corrections (CV ≈ 0) take near-identical values across tunnels. The meaning/control column briefly explains what each parameter does in the SAM4Tun pipeline.
<table><tr><td>Stage</td><td>Parameter</td><td>Meaning/Control</td><td>Behaviour</td><td>Tunnels</td><td>CV</td><td>Adapted range correction</td></tr><tr><td>Unfolding</td><td>diameter</td><td>Cylindrical scale.</td><td>Responsive</td><td>27/30</td><td>0.072</td><td>[5.31, 7.6]</td></tr><tr><td>Denoising</td><td>mask_r_low</td><td>Inner radial gate.</td><td>Responsive</td><td>30/30</td><td>0.082</td><td>[2.09, 3.75]</td></tr><tr><td>Denoising</td><td>mask_r_high</td><td>Outer radial gate.</td><td>Responsive</td><td>30/30</td><td>0.147</td><td>[2.78, 4.38]</td></tr><tr><td>Denoising</td><td>default_cutoff_z</td><td>Fallback radial cutoff (low density).</td><td>Responsive</td><td>29/30</td><td>0.142</td><td>[2.65, 6.27]</td></tr><tr><td>Denoising</td><td>z_step</td><td>Radial histogram bin width,</td><td>Responsive</td><td>30/30</td><td>0.181</td><td>[0.003, 0.005]</td></tr><tr><td>Denoising</td><td>smooth_win</td><td>Cutoff-curve smoothing win- dow.</td><td>Correction</td><td>30/30</td><td>≈0</td><td>3 → 5</td></tr><tr><td>Denoising</td><td>smooth_offset</td><td>Post-smoothing cutoff bias.</td><td>Correction</td><td>30/30</td><td>≈0</td><td>-0.003 → -0.002</td></tr><tr><td>Denoising</td><td>grad_thresh</td><td>Outer-edge gradient thresh- old.</td><td>Correction</td><td>30/30</td><td>≈0</td><td>0.2 → 0.15</td></tr><tr><td>Denoising</td><td>y_step</td><td>Angular bin width (density fil- tering).</td><td>Correction</td><td>30/30</td><td>≈0</td><td>0.5 → 0.4</td></tr><tr><td>Enhancing</td><td>inter_radius</td><td>Interpolation neighbour ra- dius.</td><td>Responsive</td><td>30/30</td><td>0.130</td><td>[0.03, 0.08]</td></tr><tr><td>Enhancing</td><td>upsampling_stage1</td><td>Stage-1 upsampling spacing.</td><td>Responsive</td><td>30/30</td><td>0.064</td><td>[0.055, 0.11]</td></tr><tr><td>Enhancing</td><td>curv_thresh</td><td>Panel-joint curvature thresh- old.</td><td>Correction</td><td>30/30</td><td>≈0</td><td>0.0005 → 0.005</td></tr><tr><td>Enhancing</td><td>depth_low</td><td>Lower interpolation depth tol- erance.</td><td>Correction</td><td>30/30</td><td>≈0</td><td>0.003 → 0.005</td></tr><tr><td>Enhancing</td><td>depth_high</td><td>Upper interpolation depth tol- erance.</td><td>Correction</td><td>30/30</td><td>≈0</td><td>0.008 → 0.015</td></tr><tr><td>Segmenting</td><td>hough_thresh_obliq</td><td>Oblique joint-line vote thresh- old.</td><td>Responsive</td><td>30/30</td><td>0.188</td><td>[20,83]</td></tr><tr><td>Segmenting</td><td>hough_thresh_horiz</td><td>Horizontal joint-line vote threshold.</td><td>Responsive</td><td>30/30</td><td>0.204</td><td>[20, 83]</td></tr><tr><td>Segmenting</td><td>hough_thresh_vert</td><td>Vertical ring-boundary vote threshold.</td><td>Responsive</td><td>28/30</td><td>0.219</td><td>[320, 980]</td></tr><tr><td>Segmenting</td><td>processing.padding</td><td>Segment-crop horizontal padding.</td><td>Responsive</td><td>29/30</td><td>0.265</td><td>[160, 419]</td></tr></table>

## Table 10

Error composition as a fraction of ground-truth points (Opus-4.6 � + � + � versus SAM4Tun baseline; 13 regular, 17 complex tunnels). FN: segment predicted as background; FP: background predicted as segment; Swap: segment predicted as the wrong segment class. Adaptation reduces false negatives; on complex tunnels the residual error is dominated by class swaps.

<table><tr><td>Category</td><td>Method</td><td>Correct</td><td>FN</td><td>FP</td><td>Swap</td></tr><tr><td>Regular</td><td>SAM4Tun m + s + k</td><td>55% 83%</td><td>22% 6%</td><td>3% 4%</td><td>20% 7%</td></tr><tr><td></td><td>SAM4Tun</td><td>30%</td><td>70%</td><td>0%</td><td>0%</td></tr><tr><td>Complex</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $m + s + k$ </td><td>41%</td><td>21%</td><td>5%</td><td>32%</td></tr></table>

them: structured context already enables LLM adaptation within fixed bounds, and pinpoints which structural parts must change next to lift the absolute ceiling.

## 5. Discussion

Taken together, the results support a mechanism-level claim rather than a deployment-level one. With the SAM4Tun pipeline and the expert reference held fixed, structured context improves bounded parameter adaptation across three LLMs. In practical terms, � + � + � can guide a reviewable parameter update without labelled retraining; however, at current accuracy it supports assisted analysis, not autonomous inspection.

## 5.1. Key findings

The experimental evaluation yields two primary findings. First, as noted in the ablation study (Section 4.2), the ablation results suggest that the context design improves adaptation across models and tunnel categories, with state providing the main gains (+0.16 to +0.25 mIoU on top of memory). Second, the LLM (m+s+k) gain over the non-LLM rule-based adaptation is concentrated on the regular category, where the rule table cannot improve over the static baseline (Section 4.1). This concentration is consistent with the single-reference design: both comparators start from a configuration tuned on a tunnel that is similar to the regular category. Near the reference, the LLM has more informative context to condition on, and its advantage over deterministic lookup becomes visible; far from the reference (the complex category), neither approach has a matching anchor and the two converge toward similarly low absolute performance, with the LLMs retaining a small mean advantage. As reported in Section 4.1, the regular category gap occurs where the deterministic control does not improve over the static baseline, whereas the complex category convergence more likely reflects a shared ceiling than a specific failure of LLM reasoning (Section 5.3).

![](images/071175f136c429c4da3c678d070bfe0ae7e9a5bd358ed1474d4f4f841daf7d06.jpg)

![](images/f48bf3a7b67e15959fceea435326ff6a84b36f00decb45c11baf4cf39a5e3284.jpg)  
Figure 11: Structural constraints of the fixed SAM4Tun labelling rule. (a) Labelling rule: after K is detected, B segments are placed one step either side of K and A segments at successive multiples of the same fixed ofset, in a fixed order, so the template mislabels rings whose segment widths or count difer from the reference. (b) Ground-truth K position versus ring index for one regular (blue, 2-1) and one complex (red, 5-1) tunnel: K jumps from ring to ring, much more on the complex tunnel, which the current one-configuration-per-tunnel setting does not address.

In practice, a domain expert calibrates the pipeline on a single reference tunnel, encodes the tuning rationale into structured documents, and deploys R4Tun for subsequent tunnels. Under this single-reference setup, the framework is best suited to targets that remain close to the calibrated reference, and it can also flag when the fixed pipeline begins to fail. The framework supports post-hoc review by logging a rationale alongside each parameter change, and requires no model retraining or labelled domain data. Experts must still define the reference configuration and bounds, review rationales, and judge whether low-confidence outputs are acceptable. Within this scope, R4Tun does not yet support autonomous final inspection or downstream tasks (e.g. structural health monitoring or Scan-to-BIM) without human verification.

## 5.2. Comparison with state-of-the-art

We position R4Tun against established tunnel-segmentation paradigms in Table 11. The comparison is presented as

an operating-point and setup-cost comparison rather than a direct accuracy ranking, because the methods are not evaluated under comparable conditions. Supervised 3D deeplearning models on Seg2Tunnel (LiningNet, SparseUNet, SCF-Net, FA-ResNet) report mIoU 0.83–0.90 [57], but are evaluated in-distribution and require per-point labels and GPU retraining. Geometric feature-engineering methods avoid training but depend on manually tuned thresholds and typically report cross-section fitting error rather than per-segment mIoU, so they are not directly comparable at the component level. Expert-tuned SAM4Tun reaches mIoU above 0.90, but only on curated, per-case-tuned test rings [13]; under our protocol (30 diverse subsets, single fixed reference, no per-case tuning) the corrected opensource pipeline reaches mIoU 0.88 on its own reference tunnel but only 0.18 when that same fixed configuration is applied static across all subsets.

The only directly comparable pair (same evaluation protocol, with no labels, no training, and no per-case tuning) is R4Tun (0.43–0.48) versus static SAM4Tun (0.18); R4Tun improves the directly comparable baseline. With the corrected implementation and anchor, R4Tun also reaches mIoU 0.784–0.796 across LLMs on near-reference (regularstaggered) tunnels. Supervised and expert-tuned figures come from more favourable settings and bound the remaining gap, not direct competitors. Deterministic, non-LLM rule baseline (Section 4.1) reaches only 0.25, and every higher figure in Table 11 requires labels, training, or per-case expert tuning.

![](images/e1a6d69cb42956a80409fb61e6e0dc0023174af52374d1081df51ce5ed023263.jpg)

![](images/0ef33bc53f6eeecd3681139fef6de82ee2025222d9fd0848abfc4e510131672f.jpg)

![](images/111cc977d8670c941c7515319509e8191d7e43b3dd0d5f2186a2c384dde40a8d.jpg)

![](images/b83b9750bf92513c290dbf04be46d9b6f7a6bac226165dd0a2d8e696efe8bfe7.jpg)  
Figure 12: Per-point error maps for a representative regular tunnel (a, regular tunnel 2-1) and complex tunnel (b, complex tunnel 5-1) under $m + s + k \ ( \mathsf { O p u s } { - } 4 . 6 )$ , drawn on the unfolded surface (x: ring index; y: circumferential angle). Left column—detection outcomes: correct (grey), false negatives (blue, segment→background) and false positives (orange, background→segment), with the ground-truth K (green) and predicted K (red) overlaid per ring; titles give FN, FP, and mIoU. Right column—class swaps (red) among otherwise correctly located points. On the regular tunnel the predicted K stays on the GT K and swaps remain low (10%). On the complex tunnel the predicted K is nearly constant and drifts away from the GT K, which jumps ring-to-ring; each afected ring’s labels then rotate together and swaps rise to 42%. The maps localise the two constraints: the K mismatch (left) drives the moving-anchor errors, and the resulting label rotation (right) is the dominant residual error.

Table 11  
Positioning R4Tun against representative tunnel-segmentation paradigms. The comparison is on setup cost as well as accuracy: ✓ indicates the requirement (or capability) applies, “–” that it does not. mIoU figures are not measured under comparable settings (see the “setting” column): supervised and expert-tuned figures are taken from [57] and [13] under their original, more favourable protocols, whereas the static-SAM4Tun, non-LLM, and R4Tun figures are from this study under a single fixed reference across 30 diverse subsets. R4Tun is the only approach that is simultaneously label-free, training-free, and expert-tuning-free while still adapting per tunnel.
<table><tr><td>Method</td><td>Labels</td><td>Training</td><td>Per-case expert tuning</td><td>Per-tunnel adaptation</td><td>mloU (setting)</td></tr><tr><td>LiningNet/SparseUNet/SCF-Net/FA-ResNet</td><td>√</td><td>1</td><td>一</td><td></td><td>0.83-0.90 (in-distribution)</td></tr><tr><td>Expert-tuned SAM4Tun</td><td></td><td></td><td>√</td><td></td><td>above 0.90 (curated rings)</td></tr><tr><td>Static SAM4Tun (this protocol)</td><td></td><td></td><td></td><td></td><td>0.18</td></tr><tr><td>Non-LLM rules (this study)</td><td></td><td></td><td></td><td>√</td><td>0.25</td></tr><tr><td>R4Tun (near-reference)</td><td></td><td></td><td></td><td>√</td><td>0.784–0.796 (across LLMs)</td></tr><tr><td>R4Tun (all tunnels)</td><td></td><td></td><td></td><td>√</td><td>0.43-0.48</td></tr></table>

## 5.3. Limitations

The limitations of this work are driven by the fixed SAM4Tun operator design and by the single-reference adaptation protocol. In particular, R4Tun inherits SAM4Tun’s assumptions and failure modes, so the LLM can only reparameterise the pipeline rather than replace it. All parameter adaptation is anchored to a single expert-tuned configuration, which creates a low ceiling for both the LLMguided and the rule-based methods. When no contextual anchor resembles the target tunnel, neither approach has suficient information to recover strong performance. Expanding R4Tun to encode multiple expert-tuned reference configurations (e.g., one per tunnel category) may further narrow the absolute gap.

Our study was evaluated exclusively on the SAM4Tun pipeline and the Seg2Tunnel dataset; potentially, its transferability is most plausible when four conditions hold: (i) each stage exposes bounded, tunable parameters (i.e., adaptation is a re-parameterisation problem rather than a redesign); (ii) expert reference configurations exist whose target characterisation is comparable to the new deployment setting; (iii) each stage admits a compact, summarised intermediate state that can be passed between agents; and (iv) stage-specific knowledge can be encoded (e.g., as bounds, rules, and failure signatures) for use during adaptation.

## 6. Conclusions

In this paper, we presented R4Tun, positioned as a mechanism contribution rather than a deployable final-inspection system, a multi-agent framework that extends a fixed, expertdesigned tunnel-lining segmentation pipeline (SAM4Tun) with LLM-guided bounded parameter adaptation. Rather than replacing the underlying geometric operators, each stage agent compares a new tunnel against the reference configuration, reads compact summaries of intermediate pipeline outputs, and proposes bounded parameter updates accompanied by a logged rationale, preserving the deterministic pipeline structure while supporting post-hoc expert review. We validated R4Tun on 30 Seg2Tunnel subsets (13 regular, 17 complex) across three diferent LLMs. The main findings of this study are:

• Structured context improves bounded adaptation across LLMs. The full � + � + � design raised mIoU from 0.18 to 0.43–0.48 and OA from 0.42 to 0.59–0.65; on the near-reference regular (staggered) subsets it reached mIoU 0.784–0.796, approaching the expert anchor (0.88).

• Intermediate pipeline state is the dominant contributor. Adding state increased mIoU by 0.16–0.25 over memory alone, whereas memory or knowledge in isolation produced only small increments.

• The adaptation behaviour is consistent across models. The three LLMs produced overlapping efect ranges, adjusted a shared set of 18 critical parameters under identical prompts, and showed limited run-to-run drift (90.9% of critical parameters unchanged across repeats).

• The framework is label-free and auditable. R4Tun shifts expert efort from repeated per-tunnel intervention toward an upfront authoring step (reference calibration plus per-stage knowledge documents), runs on commercial LLM APIs without labelled retraining, logs a rationale for each parameter change, and adds only modest runtime (96–307 s per tunnel).

These results also reveal limitations that are inherent to the present design. First, all adaptation is anchored to a single expert-tuned reference, which sets a low ceiling away from that anchor: the regular-continuous (T3) and complex subsets difer from the reference in segment arrangement, segment count, and diameter, and remain challenging (mIoU 0.15–0.31), so at current accuracy R4Tun supports assisted, expert-verified parameter re-configuration for near-reference tunnels rather than autonomous final inspection or unverified downstream use such as structural health monitoring or Scan-to-BIM. Second, R4Tun adapts but does not replace SAM4Tun’s operators; in particular, its one-configuration-per-tunnel parameterisation and fixed ofset-and-order labelling template turn per-ring geometric variation into systematic class swaps that persist regardless of parameter choice (Section 4.6). Finally, all evidence is confined to the SAM4Tun–Seg2Tunnel setting, and transferability to other parameter-controllable pipelines is argued conceptually (Section 5.3) but not yet empirically validated.

In the future, encoding multiple expert-tuned reference anchors (for example, one per tunnel category) would narrow the single-anchor gap for of-reference tunnels. Replacing the fixed-template labelling logic with dynamic, ring-level adaptatio would lift the structural ceiling identified in the error analysis. Validating the � + � + � strategy on other parameter-controllable backbones would test its expected backbone-agnostic behaviour, and broader validation across tunnel typologies and acquisition conditions, together with human-in-the-loop integration into downstream inspection tasks, would strengthen generalisability and move the framework toward verified deployment.

## Data availability

The source code, adapted parameters, and evaluation scripts are available at https://github.com/Tao-Robomin ds/R4Tun. The Seg2Tunnel dataset is publicly available.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI use

During this study, the authors evaluated large language models (Opus-4.6, GPT-5.4, and Gemini-3-Flash) as the main experimental systems. The LLMs were also used for language editing and for improving manuscript structure. The authors reviewed and edited all content and take full responsibility for the content of the publication.

## References

[1] L. Attard, C. J. Debono, G. Valentino, M. D. Castro, Tunnel inspection using photogrammetric techniques and image processing: A review, ISPRS Journal of Photogrammetry and Remote Sensing 144 (2018) 180–188.

[2] L. Weidner, G. Walton, Generalized extraction of bolts, mesh, and rock in tunnel point clouds: a critical comparison of geometric featurebased methods using random forest and neural networks, Remote Sensing 16 (2024) 4466.

[3] Y. Yue, S. Zhang, H. Liu, F. Hong, C. Liu, J. Cui, et al., Damage mechanism of a shield tunnel with cavities behind the concrete lining: an insight from a scaled model test, Tunnelling and Underground Space Technology 153 (2024) 105998.

[4] M. Q. Huang, J. Ninić, Q. B. Zhang, BIM, machine learning and computer vision techniques in underground construction: current status and future perspectives, Tunnelling and Underground Space Technology 108 (2021) 103677.

[5] A. Sjölander, V. Belloni, A. Ansell, E. Nordström, Towards automated inspections of tunnels: a review of optical inspections and

autonomous assessment of concrete tunnel linings, Sensors 23 (2023) 3189.

[6] R. Montero, J. G. Victores, S. MartÃŋnez, A. JardÃşn, C. Balaguer, Past, present and future of robotic tunnel inspection, Automation in Construction 59 (2015) 99–112.

[7] A. Strauss, J. Bien, H. Neuner, C. Harmening, C. Seywald, M. Österreicher, et al., Sensing and monitoring in tunnels: testing and monitoring methods for the assessment of tunnels, Structural Concrete 21 (2020) 1356–1376.

[8] Y. Duan, S. Qiu, W. Jin, T. Lu, X. Li, High-speed rail tunnel panoramic inspection image recognition technology based on improved YOLOv5, Sensors 23 (2023) 5986.

[9] Y.-J. Cha, R. Ali, J. Lewis, O. Büyüköztürk, Deep learning-based structural health monitoring, Automation in Construction 161 (2024) 105328.

[10] C. Su, Q. Hu, Z. Yang, R. Huo, A review of deep learning applications in tunneling and underground engineering in china, Applied Sciences 14 (2024) 1720.

[11] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, et al., On the opportunities and risks of Foundation Models, 2021. doi:10.48550/arXiv.2108.07258. arXiv:2108.07258.

[12] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, et al., Segment anything, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023, pp. 4015–4026. doi:10.1109/ICCV51070.2023.00371.

[13] Z. Ye, W. Lin, A. Faramarzi, X. Xie, J. Ninić, SAM4Tun: notraining model for tunnel lining point cloud component segmentation, Tunnelling and Underground Space Technology 158 (2025) 106401.

[14] A. M. Bran, S. Cox, O. Schilter, C. Baldassari, A. D. White, P. Schwaller, Augmenting large language models with chemistry tools, Nature Machine Intelligence 6 (2024) 525–535.

[15] C. Qian, W. Liu, H. Liu, N. Chen, Y. Dang, J. Li, et al., ChatDev: communicative agents for software development, 2024. doi:10.48550 /arXiv.2307.07924. arXiv:2307.07924.

[16] S. Hong, M. Zhuge, J. Chen, X. Zheng, Y. Cheng, C. Zhang, et al., MetaGPT: meta programming for a multi-agent collaborative framework, 2024. doi:10.48550/arXiv.2308.00352. arXiv:2308.00352.

[17] P. Chen, B. Han, S. Zhang, CoMM: collaborative multi-agent, multireasoning-path prompting for complex problem solving, 2024. doi:10 .48550/arXiv.2404.17729. arXiv:2404.17729.

[18] C. I. Garcia, M. A. DiBattista, T. A. Letelier, H. D. Halloran, J. A. Camelio, Framework for LLM applications in manufacturing, Manufacturing Letters 41 (2024) 253–263.

[19] M. A. Fischler, R. C. Bolles, Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography, Communications of the ACM 24 (1981) 381–395.

[20] M. Ester, H.-P. Kriegel, J. Sander, X. Xu, A density-based algorithm for discovering clusters in large spatial databases with noise, in: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96), AAAI Press, 1996, pp. 226– 231. doi:10.5555/3001460.3001507.

[21] M. Pauly, M. Gross, L. P. Kobbelt, Eficient simplification of point-sampled surfaces, in: Proceedings of the IEEE Visualization Conference (VIS 2002), IEEE, 2002, pp. 163–170. doi:10.1109/VISU AL.2002.1183771.

[22] C. R. Qi, L. Yi, H. Su, L. J. Guibas, PointNet++: deep hierarchical feature learning on point sets in a metric space, in: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 2017, pp. 5105– 5114. doi:10.48550/arXiv.1706.02413. arXiv:1706.02413.

[23] Q. Hu, B. Yang, L. Xie, S. Rosa, Y. Guo, Z. Wang, et al., RandLA-Net: eficient semantic segmentation of large-scale point clouds, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 11105–11114. doi:10.1109/ CVPR42600.2020.01112.

[24] J. Schult, F. Engelmann, A. Hermans, O. Litany, S. Tang, B. Leibe, Mask3D: mask transformer for 3D semantic instance segmentation, in: Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 8216–8223. doi:10.1109/ICRA48891.20

23.10160590.

[25] M. Kolodiazhnyi, A. Vorontsova, A. Konushin, D. Rukhovich, Topdown beats bottom-up in 3D instance segmentation, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 3566–3574. doi:10.1109/WACV57701.2024 .00353.

[26] [verify author list]. Zhang, Unrollingnet: [verify title], Automation in Construction (2022).

[27] Q. Wang, W. Ding, F. Li, Y. Qiao, K. Khoshelham, F. Zhang, et al., Serialized point cloud segmentation with dilated patchwise attention for generating geometric twins of shield metro tunnels, Automation in Construction 187 (2026) 106908.

[28] X. Li, Y. Zhang, J. Wang, et al., Tunnel surface defect segmentation using TransUNet with self-attention for complex underground environments, PLOS ONE 21 (2026) e0322859.

[29] H. Y. Liang, S. L. Shen, A. Zhou, ECA-enhanced YOLO integrated with SAHI for multi-defect inspection of tunnel linings, Tunnelling and Underground Space Technology 174 (2026) 107705.

[30] E. Camufo, D. Mari, S. Milani, Recent advancements in learning algorithms for point clouds: an updated overview, Sensors 22 (2022) 1357.

[31] M. Liu, R. Shi, K. Kuang, Y. Zhu, X. Li, S. Han, et al., OpenShape: scaling up 3D shape representation towards open-world understanding, in: Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023, pp. 44860–44879. doi:10.52202/075280-1944.

[32] Z. Guo, R. Zhang, X. Zhu, Y. Tang, X. Ma, J. Han, et al., Point-Bind & Point-LLM: aligning point cloud with multi-modality for 3D understanding, generation, and instruction following, 2023. doi:10.4 8550/arXiv.2309.00615. arXiv:2309.00615.

[33] R. Xu, X. Wang, T. Wang, Y. Chen, J. Pang, D. Lin, PointLLM: empowering large language models to understand point clouds, in: Computer Vision – ECCV 2024, Springer, 2024, pp. 131–147. doi:10 .1007/978-3-031-72698-9\_8.

[34] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, et al., Emerging properties in self-supervised vision transformers, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 9650–9660. doi:10.1109/ICCV48922.2021.0 0951.

[35] G. Franceschelli, C. Cevenini, M. Musolesi, Training Foundation Models as data compression: on information, model weights and copyright law, 2024. doi:1 0 . 4 8 5 5 0 / a r X i v . 2 4 0 7 . 1 3 4 9 3. arXiv:2407.13493.

[36] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, et al., SAM 2: segment anything in images and videos, 2025. URL: https://procee dings.iclr.cc/paper\_files/paper/2025/hash/45c1f6a8cbf2da59ebf2 c802b4f742cd-Abstract-Conference.html, international Conference on Learning Representations (ICLR).

[37] X. Zou, J. Yang, H. Zhang, F. Li, L. Li, J. Wang, et al., Segment everything everywhere all at once, 2023. URL: https://proceedings. neurips.cc/paper\_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c72 6172b86-Abstract-Conference.html, advances in Neural Information Processing Systems (NeurIPS 2023).

[38] K. Ge, C. Wang, Y. Guo, Y. Tang, Z. Hu, H. Chen, Fine-tuning vision foundation model for crack segmentation in civil infrastructures, Construction and Building Materials 431 (2024) 136573.

[39] Z. Ye, L. Lovell, A. Faramarzi, J. Ninić, Sam-based instance segmentation models for the automation of structural damage detection, Advanced Engineering Informatics 62 (2024) 102826.

[40] B. Wang, Z. Chen, M. Li, Q. Wang, C. Yin, J. C. P. Cheng, Omni-Scan2BIM: a ready-to-use Scan2BIM approach based on vision foundation models for MEP scenes, Automation in Construction 162 (2024) 105384.

[41] F. Pan, S. Jeon, B. Wang, F. McKenna, S. X. Yu, Zero-shot building attribute extraction from large-scale vision and language models, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024, pp. 8647–8656. doi:10.1109/WACV 57701.2024.00845.

[42] Z. Ye, Q. Li, G. Desiderio, M. Huang, W. Liu, V. Villa, J. Ninić, Maintenance-oriented tunnel digital model generation via panoptic segmentation of ultra-high-resolution images, Computer-Aided Civil and Infrastructure Engineering (2026) 100032.

[43] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, et al., Training language models to follow instructions with human feedback, in: Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022, pp. 27730–27744. doi:10.48550/a rXiv.2203.02155. arXiv:2203.02155.

[44] DeepSeek-AI, DeepSeek-R1: incentivizing reasoning capability in LLMs via reinforcement learning, 2025. doi:10.48550/arXiv.2501. 12948. arXiv:2501.12948.

[45] J. Chua, O. Evans, Are DeepSeek R1 and other reasoning models more faithful?, 2025. doi:1 0 . 4 8 5 5 0 / a r X i v . 2 5 0 1 . 0 8 1 5 6. arXiv:2501.08156.

[46] V. Xiang, C. Snell, K. Gandhi, A. Albalak, A. Singh, C. Blagden, et al., Towards System 2 reasoning in LLMs: learning how to think with meta chain-of-thought, 2025. doi:10.48550/arXiv.2501.04682. arXiv:2501.04682.

[47] OpenAI, OpenAI o3 and o4-mini system card, 2025. URL: https: //openai.com/index/o3-o4-mini-system-card/.

[48] OpenAI, Reasoning best practices, https://platform.openai.com/do cs/guides/reasoning-best-practices, 2025. Accessed: 2025-12-01.

[49] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, et al., Chain-of-thought prompting elicits reasoning in large language models, in: Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022, pp. 24824–24837. doi:10.52202/068431-1800.

[50] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, Y. Iwasawa, Large language models are zero-shot reasoners, in: Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022, pp. 22199–22213. doi:10.48550/arXiv.2205.11916. arXiv:2205.11916.

[51] Z. Zhang, Y. Yao, A. Zhang, X. Tang, X. Ma, Z. He, et al., Igniting language intelligence: the hitchhikerś guide from chain-of-thought reasoning to language agents, 2023. doi:10.48550/arXiv.2311.11797. arXiv:2311.11797.

[52] Y. Zhang, R. Sun, Y. Chen, T. Pfister, R. Zhang, S. Ö. Arik, Chain of agents: large language models collaborating on long-context tasks, 2024. doi:10.48550/arXiv.2406.02818. arXiv:2406.02818.

[53] P. Xu, W. Ping, X. Wu, L. McAfee, C. Zhu, Z. Liu, et al., Retrieval meets long-context large language models, 2024. doi:10.48550/arXiv .2310.03025. arXiv:2310.03025.

[54] L. Mei, J. Yao, Y. Ge, Y. Wang, B. Bi, Y. Cai, et al., A survey of context engineering for large language models, 2025. doi:10.48550/a rXiv.2507.13334. arXiv:2507.13334.

[55] Anthropic, Efective context engineering for AI agents, https://www. anthropic.com/engineering/effective-context-engineering-for-a i-agents, 2025. Accessed: 2025-12-01.

[56] R. O. Duda, P. E. Hart, Use of the Hough transformation to detect lines and curves in pictures, Communications of the ACM 15 (1972) 11–15.

[57] W. Lin, B. Sheil, P. Zhang, B. Zhou, C. Wang, X. Xie, Seg2Tunnel: a hierarchical point cloud dataset and benchmarks for segmentation of segmental tunnel linings, Tunnelling and Underground Space Technology 147 (2024) 105735.

## 1. Baseline parameter tables

Tables 12–15 report the SAM4Tun baseline parameter values used as the reference configuration for all adaptation experiments.

## 2. Characteriser fields

Table 16 summarises the characteriser fields used to describe each tunnel and to populate the per-stage state provided to the agents.

## 3. Non-LLM rule-based pseudocode

The non-LLM baseline reads the same per-stage knowledge documents the LLM agents read, transcribed into a deterministic Python lookup. The parameter-selection logic for the denoising stage is reproduced below. Other stages follow the same pattern.

Table 12  
Stage 1 — Unfolding parameters (sam4tun baseline).
<table><tr><td>Parameter Value</td><td></td></tr><tr><td>delta</td><td>0.005</td></tr><tr><td>slice_spacing_factor</td><td>1.2</td></tr><tr><td>vertical filter window</td><td>4.5</td></tr><tr><td>ransac_threshold</td><td>1</td></tr><tr><td>ransac__probability</td><td>0.9</td></tr><tr><td>ransac_inlier__ratio</td><td>0.75</td></tr><tr><td>ransac_sample_size</td><td>5</td></tr><tr><td>polynomial_degree</td><td>3</td></tr><tr><td>num_samples_factor</td><td>1210</td></tr><tr><td>diameter</td><td>5.60</td></tr></table>

Table 13  
Stage 2 — Denoising parameters (sam4tun baseline).
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>mask_r_low</td><td>2.7</td></tr><tr><td>mask_r_high</td><td>2.8</td></tr><tr><td>y_step</td><td>0.5</td></tr><tr><td>z_step</td><td>0.001</td></tr><tr><td>grad_threshold</td><td>0.2</td></tr><tr><td>smoothing_window_size</td><td>3</td></tr><tr><td>smoothing_offset</td><td>-0.003</td></tr><tr><td>default_cutoff_z</td><td>2.7</td></tr></table>

Table 14  
Stage 3 — Enhancing parameters (sam4tun baseline).
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>upsamp_stage1_target_dist</td><td>0.08</td></tr><tr><td>upsamp_stage2_target_dist</td><td>0.04</td></tr><tr><td>upsamp_stage3_target_dist</td><td>0.02</td></tr><tr><td>curvature threshold</td><td>0.0005</td></tr><tr><td>depth threshold low</td><td>0.003</td></tr><tr><td>depth threshold high</td><td>0.01</td></tr><tr><td>inter_radius</td><td>0.06</td></tr><tr><td>duplicate_threshold</td><td>0.02</td></tr><tr><td>num neighbors</td><td>20</td></tr><tr><td>num_interpolations</td><td>2</td></tr><tr><td>resolution</td><td>0.005</td></tr><tr><td>window_size</td><td>9</td></tr></table>

```hcl
def select_denoising(chars):
diameter = chars["estimated_diameter"]
density = chars["density"]
p10 = chars["unfolded_p10"]
p99 = chars["unfolded_p99"]
if diameter > 6.5:
family = "large"
elif chars["joint_type"] == "continuous":
family = "continuous"
else:
family = "base"
```

Table 15  
Stage 4 — Segmenting parameters (sam4tun baseline). <sup>\*</sup>
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Boundary detection</td><td></td></tr><tr><td>binary_threshold</td><td>127</td></tr><tr><td>morph kernel size</td><td>[3, 3]</td></tr><tr><td>dilation iterations</td><td>1</td></tr><tr><td>hough_thresh_oblique</td><td>50</td></tr><tr><td>minLineLength_oblique</td><td>100</td></tr><tr><td>maxLineGap_oblique</td><td>40</td></tr><tr><td>hough thresh horiz</td><td>50</td></tr><tr><td>minLineLength_horiz</td><td>100</td></tr><tr><td>maxLineGap_horiz</td><td>10</td></tr><tr><td>hough thresh vert</td><td>500</td></tr><tr><td>angle_range_obliq__pos</td><td>[6, 9]</td></tr><tr><td>angle_range_obliq_neg</td><td>[−9, -6]</td></tr><tr><td>merge_distance</td><td>3</td></tr><tr><td>ring_spacing_constant resolution</td><td>1.2 0.005</td></tr><tr><td></td><td></td></tr><tr><td>SAM template</td><td>6</td></tr><tr><td>segment_per_ring</td><td>[K, B1, A1, A2, A3, B2]</td></tr><tr><td>segment_order</td><td></td></tr><tr><td>segment width</td><td>1200</td></tr><tr><td>K_height</td><td>1079.92</td></tr><tr><td>AB_height</td><td>3239.77</td></tr><tr><td>angle</td><td>7.52</td></tr><tr><td>processing.padding</td><td>150</td></tr><tr><td>processing.y_bounds</td><td>[4200, 13100]</td></tr><tr><td>processing.crop_margin</td><td>50</td></tr></table>

```python
params = REFERENCE_DENOISING.copy()
if family == "large":
params["mask_r_low"] = p10
params["mask_r_high"] = p99 + 0.05
params["smooth_win"] = 5
elif family == "continuous":
params["mask_r_high"] = max(p99, 2.85)
params["smooth_win"] = 6
return params
```

## 4. Context components: denoising agent example

This appendix reproduces the three context components (memory, state, knowledge) that the agent receives as input.

## 4.1. Memory: reference vs target raw characteristics

Memory pairs the reference tunnel’s raw characteristics with those of the target tunnel using an identical schema, so the agent can compute deviations field-by-field (Table 17).

Memory also bundles the SAM4Tun reference parameters for the stage (Table 13), so the agent always has a knowngood baseline to deviate from.

Table 16  
Characteriser fields by stage. Raw fields are available to all stages; each subsequent group becomes available after the corresponding stage completes.
<table><tr><td>Source</td><td>Fields</td></tr><tr><td>Raw</td><td>estimated diameter, tunnel length, tunnel height z-range (min, max) mean / median / min nearest-neighbour</td></tr><tr><td>Unfolded</td><td>distance r-percentiles  $\left( { p _ { 1 0 } , \ p _ { 9 9 } } \right)$  , h-span, θ-span, θ- range median / std nearest-neighbour distance</td></tr><tr><td>Denoised</td><td>intensity median, intensity min mean / median /std nearest-neighbour dis- tance estimated diameter, tunnel length, surface completeness surface regularity, average curvature, section</td></tr><tr><td>Enhanced</td><td>curvatures total points, median / mean nearest- neighbour distance coverage uniformity template spacing suitability, current median spacing</td></tr></table>

## Table 17

Memory excerpt (raw characteristics). Reference is the experttuned tunnel; target is tunnel 4-1.
<table><tr><td>Field</td><td>Reference</td><td>Target (4-1)</td><td>Δ</td></tr><tr><td>Total points</td><td>1,109,768</td><td>1,872,537</td><td>+69%</td></tr><tr><td>Estimated diameter (m)</td><td>5.32</td><td>7.41</td><td>+39%</td></tr><tr><td>Tunnel length (m)</td><td>12.16</td><td>18.10</td><td>+49%</td></tr><tr><td>Tunnel height (m)</td><td>5.08</td><td>7.72</td><td>+52%</td></tr><tr><td>Mean NN distance (m)</td><td>0.0082</td><td>0.0081</td><td>-1%</td></tr><tr><td>Median NN distance (m)</td><td>0.0065</td><td>0.0065</td><td>≈ 0%</td></tr></table>

State excerpt (unfolded characteristics, after the unfolding stage).
<table><tr><td>Field</td><td>Reference</td><td>Target (4-1)</td></tr><tr><td>r-percentile  $p _ { 1 0 }$  (m)</td><td>2.30</td><td>2.38</td></tr><tr><td>r-percentile  $p _ { 9 9 }$  (m)</td><td>2.77</td><td>3.93</td></tr><tr><td>h-span (m)</td><td>12.08</td><td>18.86</td></tr><tr><td>θ-span (rad)</td><td>17.28</td><td>23.28</td></tr><tr><td>Median NN distance (m)</td><td>0.047</td><td>0.066</td></tr></table>

## 4.2. State: cumulative outputs of upstream stages

State grows as the pipeline executes. For the denoising agent, state consists of the unfolded characteristics produced by the preceding unfolding stage, supplied as a reference/target pair (Table 18).

## 4.3. Knowledge: parameter semantics, ranges, and constraints

Knowledge is a stage-specific Markdown document, authored once and shared across all tunnels. It enumerates each parameter together with its empirically validated range, defaults, and inter-parameter constraints. The denoising knowledge document is reproduced below.

Cross-tunnel variation. Tunnel lining datasets commonly vary along several axes: tunnel scale, from smaller metro-scale tunnels to larger-diameter tunnels; ring geometry, including shorter or longer ring lengths; segment layout, including diferent numbers and ordering of lining segments per ring; joint assembly, such as staggered, continuous, or interleaved joint arrangements; and scanning configuration, including single-station scans, multi-station registration, or uneven scanner placement.

Tunable parameters. mask\_r\_low (m), inner radial gate before depth histogramming, range [2.09, 3.75] (baseline 2.7); mask\_r\_high (m), outer radial gate, range [2.78, 4.38] (baseline 2.8); default\_cutoff\_z (m), fallback radial cutof when a �-bin lacks reliable counts, range [2.65, 6.27] (baseline 2.7); z\_step (m), radial bin width per histogram column, range [0.003, 0.005] (baseline 0.001).

Proven defaults. Proven defaults.

```prolog
smoothing_window_size = 5,
smoothing_offset = −0.002,
grad_threshold = 0.15,
y_step = 0.4.
Diagnostic rules. mask_r_low < mask_r_high; default_cutoff_z
should stay coherent with the upstream unfolding
diameter (order of magnitude ≈ diameter/2, with
diameter in metres).
```

The three other stages (unfolding, enhancing, segmenting) follow the same four-block template; the per-stage knowledge documents are released in the project’s code repository.

## 5. Worked CoT trace

The following trace continues the worked example from Appendix 4: the denoising agent receives the memory, state, and knowledge inputs above, executes the five-step CoT protocol (Section 3.3.2), and emits a single schema-conformant JSON object that the denoising stage consumes unchanged.

Step 1 — Referencing. Estimated diameter is 7.41 m vs reference 5.32 m (+39%). Unfolded �-percentiles are $p _ { 1 0 } =$ $2 . 3 8 , p _ { 9 9 } = 3 . 9 3$ , vs reference $p _ { 1 0 } = 2 . 3 0 , p _ { 9 9 } = 2 . 7 7$ . Median nearest-neighbour distance after unfolding is 0.066 m vs reference 0.047 m (+41%), indicating lower areal density on the tunnel wall.

Step 2 — Diagnostic inspection. Large-diameter, moderatesparse target. The reference mask\_r\_low/mask\_r\_high (2.7∕2.8) would clip all wall points beyond � = 2.8 m, which is below the target $p _ { 1 0 } ~ = ~ 2 . 3 8$ and well below $p _ { 9 9 } ~ = ~ 3 . 9 3 $ the radial gate must widen substantially. default\_cutoff\_z (2.7) is below the target � (3.93) and would force the fallback path to discard valid wall points; it must rise. Lower density (+41% NN distance) requires z\_step to relax from 0.001 so each radial bin retains enough samples for the gradient test.

Step 3 — Parameter adaptation. Set mask\_r\_low = 2.37 (just below target $p _ { 1 0 } = 2 . 3 8 )$ ; mask\_r\_high = 3.80 (just below target $p _ { 9 9 } = 3 . 9 3 .$ , retaining ≥ 99% of wall points while staying within the empirical range [2.78, 4.38]); default\_cutoff\_z = 3.7 (close to mask\_r\_high and order ≈ diameter/2 = 3.71); z\_step = 0.005 (proven default for sparse scans). All other parameters take the proven defaults from the knowledge block.

Step 4 — Validation. mask\_r\_low (2.37) < mask\_r\_high (3.80); both lie within their empirical ranges. default\_cutoff\_z (3.7) is consistent with the unfolding diameter (7.41 m, half-diameter 3.71 m). z\_step (0.005) lies in [0.003, 0.005]. All keys present, all types match Table 13.

Step 5 — JSON output.

```json
{
"mask_r_low": 2.37,
"mask_r_high": 3.80,
"y_step": 0.4,
"z_step": 0.005,
"grad_threshold": 0.15,
"smoothing_window_size": 5,
"smoothing_offset": -0.002,
"default_cutoff_z": 3.7
}
```

The same parameter keys are adapted by GPT-5.4 and Gemini 3 Flash in qualitatively the same direction for this tunnel.

## 6. Runtime, API calls, and cost

Table 19 reports the compute setup and, for one m+s+k run, the per-tunnel input/output token counts and indicative USD cost for each LLM (averaged over the 30 tunnels, four stage calls per tunnel).

## 7. Repeatability analysis

Table 20 reports a basic repeatability check for the full m+s+k condition (temperature 0). Each tunnel was reinferred twice per LLM and the resulting adapted parameters and mIoU were compared between run 1 and run 2.

Table 19  
Runtime, API calls, and per-tunnel API cost. Token counts are as reported by each vendor API; USD figures use vendor list prices at submission and are indicative.
<table><tr><td>Metric Value</td></tr><tr><td>LLM API calls per tunnel 4 (one per adapted stage agent Total API calls (full ablation) 120 per condition per LLM GPU Single NVIDIA RTX 5060 Retraining required None Labelled data required None (GT for evaluation only)</td></tr><tr><td colspan="2">Per-tunnel tokens and cost (m+s+k, mean over 30 tunnels) 64,854 in / 6,599 out, $1.47 64,414 in / 35,263 out, $0.43</td></tr><tr><td>Opus 4.6 GPT-5.4</td></tr></table>

Table 21

Per-class IoU—Regular tunnels (� = 13, 6-class, Opus 4.6).
<table><tr><td>Class</td><td>sam4tun</td><td>rules</td><td>memory</td><td>m+s</td><td>m+s+k</td></tr><tr><td>Background</td><td>0.511</td><td>0.639</td><td>0.576</td><td>0.671</td><td>0.698</td></tr><tr><td>K-block</td><td>0.151</td><td>0.295</td><td>0.372</td><td>0.599</td><td>0.558</td></tr><tr><td>B1-block</td><td>0.217</td><td>0.256</td><td>0.338</td><td>0.643</td><td>0.613</td></tr><tr><td>A1-block</td><td>0.268</td><td>0.234</td><td>0.317</td><td>0.681</td><td>0.651</td></tr><tr><td>A2-block</td><td>0.198</td><td>0.159</td><td>0.180</td><td>0.403</td><td>0.398</td></tr><tr><td>A3-block</td><td>0.302</td><td>0.272</td><td>0.320</td><td>0.667</td><td>0.679</td></tr><tr><td>B2-block</td><td>0.229</td><td>0.412</td><td>0.317</td><td>0.710</td><td>0.723</td></tr></table>

Table 20  
Repeatability under m+s+k (temperature 0). Metrics are computed over 30 tunnels, with two runs per tunnel per LLM.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Mean critical parameters unchanged (18 total)</td><td>90.9%</td></tr><tr><td>Mean |∆mIoU| (run 1 vs run 2)</td><td>0.029</td></tr><tr><td>Median |∆mIoU| (run 1 vs run 2)</td><td>0.000</td></tr></table>

## 8. Per-class IoU breakdown

Tables 21 and 22 report per-class IoU for Opus 4.6 (representative; other LLMs show the same pattern) alongside the rules baseline. For regular tunnels, rules achieve perclass IoU comparable to sam4tun while all LLM conditions improve roughly uniformly. For complex tunnels (rules � = 17 including 3 failed tunnels scored as zero), rules recover segment structure from near-zero for most classes but not K-block; the LLM conditions achieve further improvement on regular tunnels. B2-block remains the hardest class, requiring the 7-segment layout guidance from the knowledge component.

Table 22  
Per-class IoU—Complex tunnels (� = 17, 7-class, Opus 4.6). Rules include 3 failed tunnels scored as zero.
<table><tr><td>Class</td><td>sam4tun</td><td>rules</td><td>memory</td><td>m+s</td><td>m+s+k</td></tr><tr><td>Background</td><td>0.337</td><td>0.534</td><td>0.358</td><td>0.513</td><td>0.520</td></tr><tr><td>K-block</td><td>0.000</td><td>0.000</td><td>0.034</td><td>0.183</td><td>0.159</td></tr><tr><td>B1-block</td><td>0.000</td><td>0.041</td><td>0.001</td><td>0.084</td><td>0.135</td></tr><tr><td>A1-block</td><td>0.000</td><td>0.072</td><td>0.005</td><td>0.128</td><td>0.155</td></tr><tr><td>A2-block</td><td>0.000</td><td>0.083</td><td>0.006</td><td>0.143</td><td>0.116</td></tr><tr><td>A3-block</td><td>0.000</td><td>0.149</td><td>0.017</td><td>0.092</td><td>0.119</td></tr><tr><td>A4-block</td><td>0.000</td><td>0.116</td><td>0.019</td><td>0.100</td><td>0.104</td></tr><tr><td>B2-block</td><td>0.000</td><td>0.099</td><td>0.000</td><td>0.000</td><td>0.043</td></tr></table>

## 9. Performance distribution

Table 23 summarises the distribution of mIoU across tunnels (reported as the mean across the three LLMs for each condition).

Table 23  
Performance distribution (mean across 3 LLMs).
<table><tr><td>Metric</td><td>sam4tun</td><td>rules</td><td>memory</td><td>m+s</td><td>m+s+k</td></tr><tr><td>Mean mloU</td><td>0.176</td><td>0.254</td><td>0.230</td><td>0.430</td><td>0.452</td></tr><tr><td>Std</td><td>0.173</td><td>0.165</td><td>0.135</td><td>0.304</td><td>0.282</td></tr><tr><td>Min</td><td>0.037</td><td>0.119</td><td>0.068</td><td>0.099</td><td>0.156</td></tr><tr><td>Max</td><td>0.451</td><td>0.562</td><td>0.397</td><td>0.829</td><td>0.791</td></tr></table>