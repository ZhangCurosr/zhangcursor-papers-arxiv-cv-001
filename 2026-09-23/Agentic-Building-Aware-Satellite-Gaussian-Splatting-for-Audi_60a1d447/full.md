# Agentic Building-Aware Satellite Gaussian Splatting for Auditable Urban DSM Reconstruction

Wentao Sun<sup>1</sup>, Zhengsen Xu<sup>2</sup>, Yiping Chen<sup>3</sup>,

John S. Zelek<sup>1</sup>, Jonathan Li<sup>1</sup>

<sup>1</sup>University of Waterloo, Department of Systems Design Engineering, Waterloo, Canada

<sup>2</sup>University of Calgary, Department of Geomatics Engineering, Calgary, Canada

<sup>3</sup>Sun Yat-sen University, School of Geospatial Engineering and Science, Zhuhai, China

wentao.sun@uwaterloo.ca, chenyp79@mail.sysu.edu.cn, zhengsen.xu@ucalgary.ca, jzelek@uwaterloo.ca, junli@uwaterloo.ca

## Abstract

Urban-scale 3D reconstruction from satellite imagery supports disaster response, city monitoring, and geospatial digital twins, yet neural rendering methods typically optimize average visual fidelity rather than the structures that analysts inspect first: buildings. We present an agentic buildingaware satellite Gaussian Splatting workflow that uses Segment Anything-derived building masks as semantic priors and an Agentic Reconstruction Controller to select, verify, and record DSM reconstruction policies. On the DFC2019 JAX\_004 scene, building-aware weighting reduces buildingregion DSM MAE from 0.844 m to 0.806 m, showing that semantic priors can shift reconstruction capacity toward analystcritical regions. A staged schedule provides a balanced operating point, improving full-scene MAE from 1.362 m to 1.349 m while retaining a building gain. Across four JAX scenes, the Agent selects validated policies for both general DSM and building-focused DSM objectives, and produces building-inventory metadata and per-scene decision records. The system combines semantic priors, policy selection, region-specific DSM metrics, and DSM-derived GIS surface products for auditable urban 3D analysis.

Project Page — https://w27sun.github.io/agentdsm/

## Introduction

Satellite image collections ofer wide-area coverage for cities, disaster zones, and changing infrastructure, but converting those images into reliable 3D products remains dificult. Operational users often need DSMs, roof-level cues, and inspectable 3D surfaces from imagery collected under heterogeneous viewing angles, seasonal efects, shadows, and changing illumination. Modern neural rendering and Gaussian Splatting methods can represent complex appearance, yet their optimization is usually driven by image reconstruction losses rather than by the semantic structures that geospatial users inspect first. In urban analysis, a meter-level improvement on buildings can matter more than a visually plausible texture on roads, parking lots, or vegetation.

This paper asks a practical question: can foundation-model semantic masks and an Agentic Reconstruction Controller make satellite Gaussian Splatting more useful for operationa DSM reconstruction? We focus on buildings because they are visible in satellite imagery, application-critical, and measurable with public truth layers. The method is designed as a deployable batch workflow: generate per-view SAM-style building masks (Kirillov et al. 2023), inject them as photometric weights during Earth-observation Gaussian Splatting (EOGS), render DSMs, evaluate semantic-region errors, and export analyst-facing visual products.

The main finding is building-centric. On JAX\_004, semantic weighting reduces building MAE from 0.844 m to 0.806 m. A staged schedule gives a balanced operating point that improves full-scene MAE while retaining a building gain. Additional JAX scenes are used to evaluate an Agent-guided readiness policy: the controller observes mask quality, inventory metadata, and validation metrics, then selects buildingaware enhancement or the baseline operating mode according to the requested product. The contribution is therefore a deployable decision workflow for urban DSM production, not a single global semantic weight.

The paper makes four application-oriented contributions. First, it frames foundation segmentation as a task prior for satellite Gaussian Splatting rather than as a stand-alone segmentation product. Second, it introduces a region-weighted photometric objective and staged semantic-weight schedule for building-aware EOGS optimization. Third, it adds an Agentic Reconstruction Controller that turns mask QA, validation metrics, and building-inventory metadata into recorded policy decisions. Fourth, it evaluates DSM error by application region and exports analyst-facing DSM, GIS surface previews, and decision records.

## Related Work and Positioning

Urban 3D products support planning, energy analysis, disaster management, visualization, and digital-twin maintenance across many city-model use cases (Biljecki et al. 2015; Ketzler et al. 2020). Classical satellite 3D reconstruction has long relied on stereo and multi-view photogrammetry, with DFC2019/US3D providing a widely used benchmark for large-scale semantic 3D reconstruction from incidental satellite imagery (Bosch et al. 2019; Le Saux et al. 2019). Neural radiance fields introduced continuous scene representations for view synthesis (Mildenhall et al. 2020), and satellitespecific variants such as Sat-NeRF and EO-NeRF model RPC cameras, shadows, and transient appearance to improve DSM recovery from multi-date imagery (Marí, Facciolo, and Ehret 2022, 2023). Eficiency-oriented follow-ups, including SAT-NGP and EOGS, reduce training time and adapt neural rendering or Gaussian Splatting to Earth observation (Billouard et al. 2024; Aira, Facciolo, and Ehret 2025). Recent satellite Gaussian Splatting work further explores generalizable sparse-view reconstruction and semantic feature fusion (Huang et al. 2026; Reed et al. 2026).

<table><tr><td>Family</td><td>Main emphasis</td><td>Difference in this work</td></tr><tr><td>Sat-NeRF / EO-</td><td>RPC NeRF,</td><td>shadows, Adds building-task prior</td></tr><tr><td>NeRF SAT-NGP / EOGS</td><td>DSM/NVS</td><td>and Agent policy Efficient satellite neural/GS Uses EOGS as engine, adds</td></tr><tr><td>Recent satellite GS</td><td>reconstruction Season, shadow, sparse- Selects</td><td>semantic product control AOI-specific</td></tr><tr><td></td><td>modules</td><td>view, or generalization building-aware DSM prod- ucts Surface-oriented GS Explicit surface and mesh Reports DSM-derived GIS</td></tr></table>

Table 1: External method positioning. The comparison is by system role rather than a direct SOTA leaderboard because the proposed contribution is a semantic-prior and Agentcontrol layer around EOGS.

Foundation segmentation models provide a complementary direction. SAM introduced promptable segmentation at broad scale (Kirillov et al. 2023); subsequent remote-sensing studies demonstrate its promise for overhead imagery and motivate domain-specific adaptation (Ren et al. 2024; Osco et al. 2023; Wu and Osco 2023). Building-specific SAM adaptations improve footprint extraction and boundary quality (Feng et al. 2025; Wang et al. 2024), but they primarily evaluate 2D segmentation. Our work instead uses building masks as task priors for 3D satellite Gaussian Splatting and couples them with a scene-level operating policy.

Recent language-agent work provides a useful abstraction for this operating policy. ReAct-style systems interleave reasoning with tool actions (Yao et al. 2023), Reflexion uses feedback from previous attempts (Shinn et al. 2023), and multi-agent frameworks such as AutoGen organize planner, executor, and verifier roles (Wu et al. 2023). We adapt this Agent pattern to geospatial reconstruction: the Agent does not replace the renderer, but observes mask QA and DSM validation outputs, selects reconstruction policies, and writes a decision record for analyst review.

Mesh extraction is also related to the downstream product goal. Surface-aligned Gaussian Splatting and 2D Gaussian Splatting show that Gaussian representations can support explicit surfaces and editable meshes (Guédon and Lepetit 2024; Huang et al. 2024). Recent satellite Gaussian variants further improve season handling, shadow modeling, sparseview robustness, and surface reconstruction (Xu and Dong 2026; Luo et al. 2026; Kim et al. 2026; Chen et al. 2026). In contrast to full surface-aligned retraining, our system treats the satellite DSM/altitude output as the reliable deployment geometry and visualizes it with hillshade, building-footprint overlays, and DSM-derived oblique surface previews, leaving stronger surface regularization as future work.

Table 1 positions the work as a semantic-prior and policycontrol layer around EOGS. The application gap is turning foundation segmentation into a controlled reconstruction prior for metric satellite geometry, with AOI-specific reporting instead of one global hyperparameter.

![](images/e0cf8c0811f5c498c8a8de5f49a745c33c75cb4a117ec3b68ad2404fd8f21bc0.jpg)  
Figure 1: System overview. Multi-view satellite images and SAM-derived building masks feed candidate EOGS reconstructions. The Agentic Reconstruction Controller observes mask QA and inventory metadata, acts by selecting a policy, verifies region-specific DSM metrics, and reports an auditable product bundle.

## Operational Setting

The target user is a geospatial analyst or urban digital-twin engineer who needs a registered DSM plus interpretable 3D evidence from multi-view satellite imagery. Because such users inspect GIS-ready products rather than neural representations, the workflow treats Gaussian Splatting as a reconstruction engine inside a product pipeline.

The application is also asymmetric: not every pixel is equally important. Building roofs, roof edges, and building blocks are often the first regions inspected in post-disaster mapping, urban growth monitoring, and infrastructure inventory. Vegetation can dominate height error in many scenes, but it is less stable across dates and less likely to support crisp planar geometry. A single full-scene MAE can therefore hide the efect that matters to the user. The central design choice in this paper is to make the optimization and the evaluation region-aware.

The deployment constraints are practical: no manual polygon annotation per AOI, inspectable mask diagnostics, a retained baseline mode, and artifacts viewable without neuralrendering tools. These constraints motivate soft photometric weights rather than hard geometric constraints.

Figure 1 summarizes the intended workflow. Given multiview satellite images and camera models, a SAM-family segmenter produces per-view building masks. EOGS optimizes a Gaussian scene representation with a weighted photometric objective. The Agentic Reconstruction Controller sits around this renderer: it observes mask coverage and inventory signals, chooses the candidate schedule to run, verifies fullscene and region-specific DSM metrics, and records a decision trace. The trained representation then supports rendered images, DSM extraction, DSM error maps, and DSM-derived GIS surface visualization.

Table 2 states the system contract: observe mask plausibility and inventory, select a candidate EOGS schedule, verify DSM metrics, and report the selected operating mode.

<table><tr><td>Deployment item</td><td>System contract</td></tr><tr><td>Target user</td><td>Geospatial analyst inspecting urban DSMs and 3D surfaces</td></tr><tr><td>Input</td><td>Multi-view satellite images, camera models, op- tional SAM-derived masks</td></tr><tr><td>Output</td><td>Registered DSM, building-region errors, inventory metadata, decision record, GIS surface preview</td></tr><tr><td>Agent observations</td><td>Mask coverage, inventory metadata, overlay QA, canonical-view DSM validation</td></tr><tr><td>Agent action</td><td>Select baseline, staged, or building-specialist se-</td></tr><tr><td>Agent verifier</td><td>mantic schedule Report selected policy, full/building/non-building metrics, and decision trace</td></tr></table>

Table 2: Deployment contract for the system workflow.

## Agentic Semantic-Prior EOGS

Let $L _ { 1 } ( p )$ denote the per-pixel photometric reconstruction loss and $M _ { b } ( p )$ the SAM-derived building-mask value for pixel p. The baseline EOGS objective is modified by a perpixel weight

$$
w ( p ) = 1 + ( \lambda _ { b } - 1 ) M _ { b } ( p ) ,
$$

so building-labeled pixels receive stronger photometric supervision. In practice, the masks are generated ofline for each training image and read during optimization. This keeps the EOGS renderer, camera model, and DSM evaluation pipeline unchanged; only the loss weighting changes. If no mask is provided, or if $\lambda _ { b } = 1$ , the method reduces to the original baseline.

The weighting has an application-oriented interpretation: a building mask does not assert known height, but marks pixels whose photometric residuals matter more to the product. The mask can therefore be soft, imperfect, and view-dependent, which is important under satellite shadows, occlusion, and multi-date appearance changes.

We evaluate static and staged weights. Static weights characterize the building-vs.-global trade-of; the staged setting increases the weight from 1.3 to 1.5 to 2.0 within the same 5000-iteration budget, first establishing broad geometry and then emphasizing buildings. Stronger weights such as 2.5 test building-specialist settings.

The second part of the method is a mask-quality gate. For a scene s, we compute a mean mask coverage score

$$
c _ { s } = \frac { 1 } { N _ { s } } \sum _ { i } \frac { \sum _ { p \in \Omega _ { i } } M _ { i } ( p ) } { | \Omega _ { i } | } ,
$$

where $N _ { s }$ is the number of training views. Coverage is not used as a formal proof of correctness; it is an early quality signal. High coverage can suggest that the segmenter may be labeling broad hard surfaces, roads, or urban texture as buildings. Low coverage is therefore paired with validation before selecting aggressive building-specialist weights.

The Agentic Reconstruction Controller turns this gate into an executable policy with four steps. Observe computes mask coverage, overlay QA, building coverage/count/area/relief, and candidate-validation DSM metrics. Act selects baseline, staged, or building-specialist candidates under the requested product objective. Verify records full-scene, building, vegetation, and non-building MAE for the selected policy. Report writes a CSV summary and JSON audit card, so the decision can be inspected independently of the renderer. In the current ofline run, the action step uses candidate-set validation metrics; we do not treat it as a learned policy that generalizes without validation.

The inventory module reports connected building components, footprint area, and DSM relief as analyst-facing metadata rather than cadastral labels. Vegetation downweighting and depth-prior losses serve as calibration checks; the controller centers building masks because they best match the urban DSM product.

## Experimental Setup

Experiments use DFC2019/IARPA-style satellite scenes with ground-truth DSM and semantic class maps. We report full-scene DSM MAE and semantic-region MAE for buildings (CLS=6), vegetation (CLS=5), and non-building pixels. Predicted DSMs are registered to ground truth using the existing EOGS DSM evaluation pipeline before computing errors. The metric is simple by design: it matches the kind of height error that downstream mapping users inspect, and it can be computed for every ablation without requiring a new learned evaluator.

The primary semantic experiments use SAM-derived building masks generated ofline for each training view, without manual mask correction. All reported semantic-prior runs use 5000 EOGS iterations and the same train/test split as the baseline. We evaluate a controlled JAX\_004 ablation and cross-scene staged experiments on JAX\_068, JAX\_214, and JAX\_260; canonical DSM test indices are fixed per scene before evaluation. Baseline multi-scene EOGS reproduction metrics provide context for how scene dificulty varies before semantic priors are added.

JAX\_004 is the primary building-enhancement scene because its building mask is visually plausible and its baseline building error leaves room for improvement. JAX\_068 and JAX\_214 provide high-coverage cases for testing the readiness policy, while JAX\_260 provides a low-coverage case with a strong baseline. Together, these scenes support both the positive building result and the policy-selection analysis used for deployment. The Agent report script uses the same completed metric tables, SAM-mask coverage logs, and benchmark CLS/DSM rasters to generate inventory metadata and policy decision records.

## Results

Table 3 shows the central trade-of. On JAX\_004, a static weight of 2.0 gives the best building-region MAE, improving from 0.844 m to 0.806 m, but it prioritizes building accuracy over full-scene error. A stronger static weight of 2.5 is nearly tied on buildings at 0.808 m and lowers full-scene MAE to 1.352 m, making it a useful deployment compromise. The staged schedule is the best current global compromise, improving full MAE from 1.362 m to 1.349 m and nonbuilding MAE from 1.424 m to 1.409 m while still improving building error relative to baseline.

![](images/d01189274d71c0180beed7f8d0434abc89df4b3d52970725a144ab7d0dc115cd.jpg)

![](images/7b21368d802585b7f3ff12ba8a36d3a834196f78c7cbae4202ede970d77c7c61.jpg)  
Figure 2: Building/global operating points and cross-scene regional DSM deltas. Deltas below zero indicate improvement over each baseline.

<table><tr><td>Scene</td><td>Variant</td><td>Full MAE</td><td>Bldg. MAE</td><td></td><td>Veg. MAE Non-bldg. MAE</td></tr><tr><td>JAX_004 Baseline</td><td></td><td>1.3625</td><td>0.8440</td><td>3.2068</td><td>1.4238</td></tr><tr><td>JAX_004 SAM 1.3</td><td></td><td>1.3615</td><td>0.8326</td><td>3.2307</td><td>1.4241</td></tr><tr><td>JAX_004 SAM 2.0</td><td></td><td>1.3757</td><td>0.8061</td><td>3.1939</td><td>1.4432</td></tr><tr><td>JAX_004 SAM 2.5</td><td></td><td>1.3515</td><td>0.8077</td><td>3.1493</td><td>1.4159</td></tr><tr><td>JAX_004 Staged</td><td></td><td>1.3486</td><td>0.8347</td><td>3.1710</td><td>1.4094</td></tr><tr><td>JAX_068 Baseline</td><td></td><td>1.0926</td><td>1.0478</td><td>1.6069</td><td>1.1266</td></tr><tr><td>JAX_068 Staged</td><td></td><td>1.1038</td><td>1.0685</td><td>1.7942</td><td>1.1305</td></tr><tr><td>JAX_068 S1.1</td><td></td><td>1.1039</td><td>1.0609</td><td>1.7362</td><td>1.1365</td></tr><tr><td>JAX_214 Baseline</td><td></td><td>1.7493</td><td>1.3635</td><td>3.0016</td><td>2.1635</td></tr><tr><td>JAX_214 Staged</td><td></td><td>1.7541</td><td>1.4252</td><td>2.7626</td><td>2.1072</td></tr><tr><td>JAX_214 S1.1</td><td></td><td>1.7589</td><td>1.3777</td><td>3.1045</td><td>2.1680</td></tr><tr><td>JAX_260 Baseline</td><td></td><td>1.5502</td><td>0.8303</td><td>2.2945</td><td>1.6452</td></tr><tr><td>JAX_260 Staged</td><td></td><td>1.5726</td><td>0.8665</td><td>2.3114</td><td>1.6658</td></tr><tr><td>JAX_260 S2.5</td><td></td><td>1.7282</td><td>0.9527</td><td>2.3232</td><td>1.8305</td></tr></table>

Table 3: DSM MAE in meters. JAX\_004 shows the buildingregion gain, while the additional scenes support readinesspolicy selection for semantic weighting.

The JAX\_004 ablation shows distinct operating points. Weight 1.3 is conservative, improving buildings with little global change. Weight 2.0 is the building-specialist setting, giving the best building MAE with a stronger buildingpriority trade-of. Weight 2.5 retains near-best building accuracy while improving full-scene MAE relative to weight 2.0. The staged setting favors full-scene and non-building MAE while preserving a smaller building gain. This supports policy selection because deployment settings may prefer diferent trade-ofs depending on the user task.

The cross-scene results motivate policy selection rather than a fixed semantic weight. JAX\_004 selects the staged building-aware mode, while JAX\_068, JAX\_214, and JAX\_260 select baseline or conservative modes under the readiness policy. This behavior is useful in an applied reconstruction system: semantic priors are activated when they improve the requested product, and the baseline remains available as a high-quality operating point. Figure 2 visualizes the regional error deltas used by this policy.

Figure 3 shows how mask coverage and validation complement each other. JAX\_068 and JAX\_214 have mean buildingmask coverage near 0.405 and 0.442, much higher than JAX\_004 and JAX\_260, so the readiness policy treats them as candidates for conservative operation. JAX\_260 illustrates the value of the validation step: even with low coverage, the baseline can remain the preferred product setting. The resulting gate combines coverage, overlay inspection, and a cheap validation run.

The auxiliary-prior experiments further support the design choice. The vegetation downweight experiment was motivated by the fact that trees and seasonal vegetation can be unstable across views; it serves as a calibration check for whether semantic reweighting should target all unstable regions or the application-critical region. The current results favor building masks because they align directly with the target geometry, the sensor evidence, and the metric evaluation. These comparisons sharpen the contribution: the useful prior is not “any foundation model output,” but a prior that matches the task region, the sensor geometry, and the requested product.

## Agent-Generated Reconstruction Report

To make the readiness policy measurable, we run an ofline Agentic Reconstruction Controller over the four-scene result set. The Agent receives the candidate-validation metrics in Table 3, SAM-mask coverage, and AOI inventory metadata computed from the benchmark building layer. It then selects two products per AOI: a general DSM policy that minimizes full-scene MAE, and a building-focused policy that minimizes building-region MAE. A policy-regret check shows the value of AOI-specific selection: for the general DSM objective, the Agent selects a mean MAE of 1.4352, compared with 1.4386 for fixed baseline and 1.4448 for fixed staged operation; for the building-focused objective, it selects 1.0119, compared with 1.0214 and 1.0487. The run also writes perscene decision records to disk, recording the observations, selected policies, and metrics used by the decision.

Table 4 reframes the multi-scene experiment as product selection. For JAX\_004, the controller selects Staged for the best general DSM and Static 2.0 for the best building DSM. For JAX\_068, JAX\_214, and JAX\_260, it keeps the validated baseline product. In all cases, the output is a recorded policy decision rather than a manually chosen hyperparameter.

![](images/42b6f8e7cae7b689678c2bcf647c17ce09a9d815fc1ac71ffc9876ef26cfb4a1.jpg)

![](images/297b62d4e33ff9d63112cd080730d945689914818ec0a551c2000b2dc5003e8c.jpg)

Figure 3: Mask-QA diagnostics. Coverage provides an interpretable prior-quality signal, and validation ablations complete the operating-policy decision.
<table><tr><td>Scene</td><td>Bldg. area</td><td>Components</td><td>SAM cov.</td><td>General policy</td><td>Full MAE</td><td>Building policy</td><td>Bldg. MAE</td></tr><tr><td>JAX_004</td><td> $6 , 9 1 0 \mathrm { ~ m } ^ { 2 } / 1 0 . 5 \%$ </td><td>63</td><td>0.131</td><td>Staged</td><td>1.3486</td><td>Static 2.0</td><td>0.8061</td></tr><tr><td>JAX_068</td><td> $2 8 , 2 2 2 \mathrm { m } ^ { 2 } / 4 3 . 1 \%$ </td><td>9</td><td>0.405</td><td>Baseline</td><td>1.0926</td><td>Baseline</td><td>1.0478</td></tr><tr><td>JAX_214</td><td>4 32,959 m2 / 50.3%</td><td>3</td><td>0.442</td><td>Baseline</td><td>1.7493</td><td>Baseline</td><td>1.3635</td></tr><tr><td>JAX_260</td><td> $5 , 6 2 6 ~ \mathrm { m } ^ { 2 } / 8 . 6 \%$ </td><td>22</td><td>0.127</td><td>Baseline</td><td>1.5502</td><td>Baseline</td><td>0.8303</td></tr></table>

Table 4: Agent-generated AOI reconstruction report. Building area and component count are benchmark-grid inventory metadata (CLS=6, 0.5 m grid, connected components $\geq 2 5  { \mathrm { m } } ^ { 2 } )$ . The Agent selects the validated policy for both general DSM and buildingfocused DSM objectives on all four scenes.

The inventory fields add operational metadata–footprint area, roof-block count, mask coverage, selected policy, and region-specific error–that make the product easier to triage, compare, and archive.

## Deployment Lessons

Figure 4 summarizes the intended product view. For JAX\_004, the ofline AOI run emits a registered DSM, DSM error view, building inventory, selected general and building policies, JSON/CSV decision records, a hillshade overlay, and a DSM-derived oblique surface preview. The key design choice is that the selected policy is reported with the evidence that justified it. Analysts can therefore inspect not only the final surface but also the mask coverage, inventory context, and regional DSM errors that led the Agent to select staged, static, or baseline operation.

The experiments suggest three deployment lessons. First, foundation-model masks can steer reconstruction without retraining the segmenter or the renderer, but the gain is scene dependent. Second, visually plausible masks are not suficient for metric geometry; strong semantic weighting should be selected only after DSM validation confirms that the requested region improves. Third, auxiliary priors must match the product objective. In these experiments, building masks are useful because the region is visible, measurable, and operationally important; vegetation downweighting and monocular depth priors require stronger confidence calibration before deployment.

Operationally, the Agent should run a QA gate before full training: compute mask coverage, inspect mask overlays, summarize building inventory, and run cheap canonical-view validation before schedule selection. High-coverage scenes are routed to baseline or conservative settings unless validation supports stronger building weights; low-coverage scenes can advance to stronger settings such as 2.5 when the requested product prioritizes roof accuracy. A deployment dashboard should report the selected weight, selection rationale, inventory metadata, full-scene MAE, building MAE, and non-building MAE so the trade-of is visible instead of hidden behind a single aggregate score.

The adoption path is incremental. A first product release can provide DSMs, error maps, mask overlays, DSM-derived surface previews, and an artifact manifest. Later runs can add cached inventory, validation metrics, and policy histories across AOIs. This keeps the learning-based renderer replaceable: future satellite Gaussian Splatting engines, stronger segmentation models, or surface-aligned mesh extractors can be inserted while preserving the same validation record and analyst-facing decision contract.

## Scope and Future Work

This study focuses on four JAX scenes, building-focused masks, DSM-region metrics, and Agent-generated decision records. The scope is suficient to test the proposed operating policy, but not to claim a universal semantic weighting rule. JAX\_004 establishes the building-aware enhancement case, while the other scenes show why policy selection matters under diferent mask coverage, inventory, and baseline-strength conditions.

Future work should add calibrated mask-quality prediction from coverage, connected-component statistics, crossview agreement, rendered-geometry alignment, and validation loss. The weighting schedule should also become adaptive: rather than choosing from a fixed set of 1.3, 2.0, and 2.5 settings, the controller should increase building emphasis only while a full-scene accuracy bound is satisfied. This would turn the current candidate-selection procedure into a more scalable AOI policy.

![](images/f80413affcac46492c055eed70f0673b6a40c40ed412fb4773bb8e9fc04260cc.jpg)  
Figure 4: Agent-generated JAX\_004 product sheet combining DSM hillshade overlays, a DSM-derived oblique surface preview, AOI inventory, selected policies, readiness criteria, and emitted artifacts.

A second direction is stronger surface export. The current product uses the DSM/altitude output as dependable geometry and avoids treating normalized Gaussian RGB as a photorealistic mesh texture. Surface-aligned Gaussian methods suggest a path toward meshes that better preserve roof planes and boundaries, but satellite deployment should combine them with DSM alignment, georeferenced texture projection, semantic QA, and GIS-ready metadata.

## Deployment Validation Plan

Before field deployment, the Agent should be validated on a larger AOI queue that represents the intended operating environment. Each AOI should run the baseline, a conservative semantic setting, and at most one stronger building-focused setting selected by the QA gate. The validation record should store mask coverage, inventory metadata, overlay thumbnails, available DSM errors, and the final policy decision. Acceptance criteria are product-specific: building-focused requests require building-region improvement with a full-scene accuracy bound, while general DSM requests prefer the staged or baseline policy unless both building and full-scene metrics improve.

This validation queue should also separate model performance from product readiness. A scene may have acceptable global MAE but fail a building-focused request, while another scene may justify stronger semantic weighting only after regional DSM validation. Recording these cases gives operators a calibration set for deciding when mask coverage is reliable enough to trigger semantic weighting and gives analysts a repeatable basis for accepting, comparing, or rerunning DSM products across scenes.

For each accepted AOI, the audit card should identify the candidate policies considered, the selected product objective, the mask-coverage and inventory signals used by the gate, and the regional DSM metrics used for verification. Failed or baseline-selected cases should be retained rather than discarded, because they define the operational boundary of the semantic prior. This record is especially important when the same reconstruction pipeline is run across heterogeneous city blocks, where dense roofs, sparse suburbs, vegetation, and shadowed views can produce diferent policy choices. Over time, these records can support threshold tuning, operator review, and comparison of new reconstruction engines without changing the analyst-facing product contract.

## Conclusion

This paper presents an agentic building-aware foundationprior workflow for satellite Gaussian Splatting. On maskvetted scenes, semantic weighting can improve building DSM accuracy; across scenes, the Agentic Reconstruction Controller converts mask QA, inventory metadata, and validation metrics into explicit operating policies. The main application contribution is therefore not a single fixed weight, but a product workflow that reports when semantic priors help and when the baseline is the better reconstruction choice.

The results support a practical principle for urban DSM production: reconstruction systems should expose their policy decisions as clearly as their surfaces. By pairing regionspecific DSM metrics with decision records and analystfacing visual products, the proposed workflow makes neural satellite reconstruction more auditable, easier to triage, and better aligned with GIS deployment needs.

## References

Aira, L. S.; Facciolo, G.; and Ehret, T. 2025. Gaussian Splatting for Eficient Satellite Image Photogrammetry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5959–5969.

Biljecki, F.; Stoter, J.; Ledoux, H.; Zlatanova, S.; and Çöltekin, A. 2015. Applications of 3D City Models: State of the Art Review. ISPRS International Journal of Geo-Information, 4(4): 2842–2889.

Billouard, C.; Derksen, D.; Sarrazin, E.; and Vallet, B. 2024. SAT-NGP: Unleashing Neural Graphics Primitives for Fast Relightable Transient-Free 3D Reconstruction from Satellite Imagery. In IEEE International Geoscience and Remote Sensing Symposium, 8749–8753.

Bosch, M.; Foster, K.; Christie, G.; Wang, S.; Hager, G. D.; and Brown, M. 2019. Semantic Stereo for Incidental Satellite Images. In Proceedings of the IEEE Winter Conference on Applications ofComputer Vision, 1524–1532.

Chen, M.; Guo, W.; Wang, B.; Li, W.; Fang, T.; Zhang, J.; Zhao, J.; Kuang, H.; Hu, H.; Ge, X.; Zhu, Q.; and Xu, B. 2026. SatSurfGS: Generalizable 2D Gaussian Splatting for Sparse-View Satellite Surface Reconstruction. arXiv:2605.07181.

Feng, W.; Guan, F.; Tu, J.; and Xu, W. 2025. BuildingSAM: A Dual-Branch Feature-Augmented Segment Anything Model for Remote Sensing Building Extraction. IEEE Geoscience and Remote Sensing Letters, 22: 6008605.

Guédon, A.; and Lepetit, V. 2024. SuGaR: Surface-Aligned Gaussian Splatting for Eficient 3D Mesh Reconstruction and High-Quality Mesh Rendering. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5354–5363.

Huang, B.; Yu, Z.; Chen, A.; Geiger, A.; and Gao, S. 2024. 2D Gaussian Splatting for Geometrically Accurate Radiance Fields. In ACM SIGGRAPH Conference Papers.

Huang, X.; Liu, X.; Wan, Y.; Zheng, Z.; Zhang, B.; Xiong, M.; Pei, Y.; and Zhang, Y. 2026. SkySplat: Generalizable 3D Gaussian Splatting from Multi-Temporal Sparse Satellite Images. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, 5158–5166.

Ketzler, B.; Naserentin, V.; Latino, F.; Zangelidis, C.; Thuvander, L.; and Logg, A. 2020. Digital Twins for Cities: A State of the Art Review. Built Environment, 46(4): 547–573.

Kim, H.-G.; Yun, S.; Park, J.; and Kwon, D. 2026. GeoGS: Geospatial Gaussian Splatting for Robust 3D Reconstruction from Sparse Satellite Imagery. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 7980–7989.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; Dollár, P.; and Girshick, R. 2023. Segment Anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 4015–4026.

Le Saux, B.; Yokoya, N.; Hänsch, R.; Brown, M.; and Hager, G. 2019. 2019 IEEE GRSS Data Fusion Contest: Large-Scale Semantic 3D Reconstruction. IEEE Geoscience and Remote Sensing Magazine, 7(4): 33–36.

Luo, F.; Pan, H.; Yang, X.; Jiang, B.; Liu, F.; and Huang, T. 2026. ShadowGS: Shadow-Aware 3D Gaussian Splatting for Satellite Imagery. arXiv:2601.00939.

Marí, R.; Facciolo, G.; and Ehret, T. 2022. Sat-NeRF: Learning Multi-View Satellite Photogrammetry With Transient Objects and Shadow Modeling Using RPC Cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 1311–1321.

Marí, R.; Facciolo, G.; and Ehret, T. 2023. Multi-Date Earth Observation NeRF: The Detail Is in the Shadows. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2035–2045.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2020. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. In European Conference on Computer Vision, 405–421.

Osco, L. P.; Wu, Q.; de Lemos, E. L.; Gonçalves, W. N.; Ramos, A. P. M.; Li, J.; and Marcato Junior, J. 2023. The Segment Anything Model (SAM) for Remote Sensing Applications: From Zero to One Shot. International Journal ofApplied Earth Observation and Geoinformation, 124: 103540.

Reed, A.; Nagle-McNaughton, T.; Elliott, S.; and Mapel, J. 2026. Fusing Semantic Features with Gaussian Splatting for Enhanced Satellite Image Surface Reconstruction. Remote Sensing, 18(10): 1563.

Ren, S.; Luzi, F.; Lahrichi, S.; Kassaw, K.; Collins, L. M.; Bradbury, K.; and Malof, J. M. 2024. Segment Anything, From Space? In IEEE/CVF Winter Conference on Applications ofComputer Vision, 8355–8365.

Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning. In Advances in Neural Information Processing Systems.

Wang, C.; Chen, J.; Meng, Y.; Deng, Y.; Li, K.; and Kong, Y. 2024. SAMPolyBuild: Adapting the Segment Anything Model for Polygonal Building Extraction. ISPRS Journal of Photogrammetry and Remote Sensing, 218: 707–720.

Wu, Q.; Bansal, G.; Zhang, J.; Wu, Y.; Zhang, S.; Zhu, E.; Li, B.; Jiang, L.; Zhang, X.; and Wang, C. 2023. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework. arXiv:2308.08155.

Wu, Q.; and Osco, L. P. 2023. samgeo: A Python Package for Segmenting Geospatial Data with the Segment Anything Model (SAM). Journal of Open Source Software, 8(89): 5663.

Xu, Y.; and Dong, Q. 2026. SA-GS: Season-aware afine 3D Gaussian Splatting for satellite image rendering. ISPRS Journal of Photogrammetry and Remote Sensing, 236: 474– 486.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations.