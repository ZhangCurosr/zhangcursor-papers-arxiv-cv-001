# Hardware-Accelerated Instance Segmentation for Resource-Constrained Space Robotics with Criticality Analysis

Siddhant Shete<sup>1</sup>, Hilmi Dogu Kuc¨ uker¨ <sup>1</sup>, Udo Frese<sup>1,2</sup>, Frank Kirchner<sup>1,2</sup>

Abstract— Autonomous lunar missions require real-time perception under three coupled constraints: extreme low-light conditions, limited onboard compute, and radiation-induced hardware faults that can silently corrupt inference. We present a deployment-oriented instance segmentation framework for resource-constrained lunar robotics that jointly addresses quantization calibration and system-level fault exposure under strict compute constraints. First, we introduce Activation Variance Informative Sampling (AVIS), a label-free calibration strategy that deterministically selects calibration samples based on activation variance statistics. Second, we deploy a YOLO-based segmentation model on a Deep Learning Processor Unit (DPU) with architectural modifications that reduce CPU fallback paths and enable statically compiled execution with bounded latency in low-lighting conditions. We further introduce a software-level criticality analysis to estimate fault exposure and guide mitigation under radiation-constrained operation. On a lunar micro-rover platform, AVIS with bias correction recovers 69.8% of quantization-induced accuracy loss while achieving 309 ms inference latency and 5.7 W power consumption. Targeted mitigation reduces global criticality by 31.7%. The results demonstrate an integrated approach and a blueprint for a reliable and safe AI perception framework under space deployment constraints.

## I. INTRODUCTION

Autonomous lunar robots must perceive their environment with high reliability to detect hazards, plan safe trajectories, and execute mission-critical tasks without human intervention [2]. Communication latencies of several seconds, harsh environmental extremes, and the complete absence of in-situ repair make real-time onboard autonomy a prerequisite for mission success [26]. However, standard AI models are unsuitable for the lunar operating envelope, where three linked limitations—extreme low-light illumination, extremely limited onboard computing, and radiation-induced hardware defects—can decrease both perception accuracy and runtime predictability.

Perception failures in autonomous systems can have mission-critical consequences. The Chandrayaan-2 Vikram lander crash, which has been widely analyzed in the literature as highlighting the importance of real-time fault handling during powered descent [17], resulted in complete mission loss; the engineering lessons directly informed the successful Chandrayaan-3 redesign [33]. These missions highlight the need for deterministic execution, autonomous fault tolerance, and graceful degradation, because in-flight retraining, manual intervention, and hardware repair are infeasible [29]. From a deployment perspective, three tightly coupled constraints dominate system design:

1) Illumination: Lunar scenes exhibit extreme low-light regions, high-contrast cast shadows, and specular regolith reflections that confound conventional visual perception [11].

2) Compute constraints: Onboard platforms are severely limited in power dissipation, memory capacity, and thermal headroom [8], [35], making computationally efficient deployment mandatory [18].

3) Radiation: Single-event effects—transient bit flips, memory corruption, and logic upsets—can silently corrupt inference without immediate observable indications [5].

Consequently, flight-grade perception must be deterministic, resilient to transient faults, and capable of self-recovery, prioritizing mission safety over peak benchmark performance.

Deep neural networks achieve state-of-the-art segmentation accuracy on terrestrial benchmarks, but their behavior degrades unpredictably under space-deployment conditions: quantization noise from reduced precision, activation distortion from model compression, and silent data corruption from radiation [7], [24]. Post-training quantization (PTQ) reduces model size and latency—prerequisites for embedded deployment—but introduces layer-dependent accuracy loss that is difficult to bound analytically [37]. Instance segmentation provides the dense, geometry-aware scene understanding required for hazard avoidance: pixel-level delineation of rocks and craters supports landing site assessment, realtime obstacle detection, and autonomous target selection [2], [9], [19]. Yet the computational intensity of segmentation networks significantly complicates their deployment, particularly under strict power and memory constraints.

In this work, we present a deployment-oriented instance segmentation framework for lunar robotics that jointly addresses quantization fidelity, execution determinism, and radiation-related fault exposure. The framework integrates three components:

• a label-free calibration strategy (AVIS) that selects informative inputs using activation variance statistics to improve INT8 quantization stability,

• a hardware-aware deployment pipeline that reduces runtime variability through statically compiled execution on a DPU accelerator, and

• a software-level criticality analysis that ranks functional blocks by memory footprint and execution time to guide radiation-informed mitigation.

These components are designed to operate jointly under strict compute, power, and reliability constraints of lunarclass embedded robotics. AVIS reduces quantization sensitivity by improving calibration stability, while the criticality analysis identifies software components most exposed to radiation-induced faults. Together, they improve robustness under the combined constraints of lunar deployment.

Evaluated on a lunar micro-rover platform, the framework recovers 69.8% of quantization-induced accuracy loss, achieves 309 ms inference at 5.7 W, and reduces radiationinduced criticality by 31.7% through targeted mitigation. This demonstrates a deployable instance segmentation system for lunar robotics under combined constraints of quantization, deterministic execution, and radiation-induced faults.

## II. RELATED WORK

We situate our contribution at the intersection of planetary perception, post-training quantization for edge deployment, and radiation-tolerant AI execution.

a) Planetary Perception: Deep learning-based perception is increasingly explored for planetary navigation [3], hazard avoidance [32], and terrain segmentation from orbital and surface imagery [21], [25], [28]. However, these approaches are evaluated exclusively on offline datasets with no consideration of deterministic execution, bounded latency, or deployment on radiation-informed hardware. The implicit assumption of unlimited floating-point compute and safe execution environments limits their applicability to flightgrade systems.

b) Post-Training Quantization and Calibration: PTQ and model compression enable deployment on space-grade edge hardware including FPGAs and DPUs [4], [27]. INT8 quantization reduces memory footprint and latency, but its effectiveness depends on calibration data selection [14]. Poorly chosen inputs cause activation range misestimation, leading to clipping artifacts and unpredictable accuracy degradation. Recent work compares quantization against pruning [20] and explores label-free calibration strategies [36], [37], but existing methods do not guarantee the deterministic, reproducible quantization behavior required by safety-critical flight software.

This gap motivates AVIS, which provides deterministic sample selection without labels or retraining. Unlike prior post-training quantization methods that rely on activation range coverage, entropy-based sampling, or reconstruction loss minimization, AVIS constructs a deterministic ranking of calibration samples based on aggregated layer-wise activation variance. This enables reproducible sample selection without stochastic sampling or label dependence.

c) Radiation-Tolerant AI Acceleration: The space radiation environment induces single-event upsets that silently corrupt weights, activations, or control logic in embedded accelerators [5], [10]. Hardware-level mitigations—Triple Modular Redundancy (TMR), Error Detection and Correction (EDAC) codes, and memory scrubbing—significantly improve resilience [16], [30], [34], but address radiation tolerance at the accelerator level in isolation. However, these methods primarily focus on hardware-level mitigation and largely neglect software-level fault propagation across inference pipelines. This limits efficient mitigation resource allocation under the severe power, area, and weight budgets of space platforms [12], [15]. Recent studies on fault injection in neural networks highlight sensitivity to bit-level perturbations, but do not provide deployment-level criticality prioritization.

d) Gap and Positioning: Table I summarizes coverage across key deployment dimensions. To the best of our knowledge, no prior work simultaneously integrates deterministic label-free PTQ calibration, DPU-accelerated instance segmentation execution, and function-wise radiation-informed criticality analysis within a unified perception framework for space robotics.

TABLE I: Comparison of prior work across key deployment dimensions. Our framework integrates instance segmentation, post-training quantization, DPU deployment, radiationinformed analysis, and label-free calibration within a unified system.

<table><tr><td>Work</td><td>Instance Segmentation</td><td>PTQ</td><td>DPU</td><td>R.A</td><td>Label-Free Calibration</td></tr><tr><td>Petrakis et al. [25]</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Liu et al. [21]</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Prieur et al. [28]</td><td>√</td><td></td><td></td><td></td><td></td></tr><tr><td>Blacker et al. [4]</td><td></td><td>√</td><td></td><td></td><td></td></tr><tr><td>Williams et al. [36]</td><td></td><td>√</td><td>一</td><td></td><td>√</td></tr><tr><td>Shao et al. [30]</td><td></td><td></td><td>√</td><td>√</td><td></td></tr><tr><td>Tedeschi et al. [34]</td><td></td><td></td><td>1</td><td>√</td><td></td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Note: R.A = Radiation Analysis

## III. SYSTEM DESIGN AND PROBLEM FORMULATION

The perception framework targets resource-constrained lunar surface platforms where autonomous hazard detection must operate without ground-in-the-loop support. The reference platform is the Lunar Micro Rover for Night Survival (LuNiS)<sup>1</sup>, a micro-class rover designed for semi-autonomous navigation under extreme low-angle illumination near permanently shadowed regions, with a continuous power budget below 10 W for all onboard computation. LuNiS Rover exemplifies systems where perception is expected to be simultaneously accurate, deterministic, energy-efficient, and resilient to radiation-induced faults.

The deployable perception pipeline, illustrated in Fig. 1, performs fully onboard instance segmentation of rocks and craters—the primary geometric hazards during surface traversal. All model weights are frozen after offline training; no in-flight retraining, fine-tuning, or dynamic model modification is performed.

## A. Problem Formulation

The objective is to deploy a high-accuracy instance segmentation model on a resource-constrained DPU such that it is designed to operate deterministically under fixed execution graphs, bounded in latency, and robust to both quantizationinduced accuracy loss and radiation-induced hardware faults. The deployment is designed to address the following requirements:

1) Quantization fidelity: Segmentation quality must be preserved under INT8 post-training quantization with bounded, characterizable accuracy degradation relative to the FP32 baseline. Non-reproducible accuracy loss is undesirable for reliable deployment.

2) Deterministic real-time execution: End-to-end inference is expected to complete within a fixed latency bound dictated by the rover’s motion profile. Identical inputs are expected to produce identical outputs under fixed deployment conditions and static execution graphs, with no dependence on dynamic resource allocation.

3) Data-efficient, label-free calibration: INT8 calibration must operate without labeled data and select samples deterministically to ensure repeatable quantized behavior, reflecting the scarcity of representative lunar imagery and the impossibility of post-launch adjustment.

4) Radiation-aware fault robustness: The framework estimates per-block vulnerability to radiation-induced single-event effects and supports allocation of mitigation resources to the highest-risk components.

Quantization actions impact radiation-induced bit flip sensitivity, while hardware partitioning between CPU and DPU affects latency and spatial radiation exposure. Calibration strategy additionally influences INT8 dynamic range tightness, affecting fidelity and robustness. This connection drives the co-design approach outlined in Section IV.

## IV. METHODOLOGY

This section presents the four methodological components of our framework, designed to jointly satisfy the deployment requirements defined in Section III-A: segmentation architecture (Section IV-A), hardware-deterministic deployment (Section IV-B), activation-variance-based calibration (Section IV-C), and radiation-aware criticality analysis (Section IV-E).

## A. Instance Segmentation Architecture

We employ YOLOv8m-based instance segmentation to detect and delineate rocks and craters—the primary geometric hazards during surface traversal [9], [19]. Instance segmentation is preferred over bounding-box detection because pixellevel boundary delineation is essential for trajectory planning over irregular terrain, where box approximations introduce unacceptable collision-risk margins. The medium variant was selected for its trade-off between mask accuracy and inference efficiency: larger variants exceed DPU resource limits, while smaller variants exhibit measurable degradation on thin or partially occluded rock boundaries. The model is trained offline on 25,000 images comprising synthetic renders and custom-collected data [31] under lunar-representative conditions (low-light illumination, high-contrast shadows, specular regolith reflections). All weights are frozen after training, ensuring deterministic and reproducible inference throughout the mission lifetime.

## B. Hardware-Aware Model Adaptation for DPU Deployment

Direct deployment of YOLOv8m on the target DPU is not feasible without architectural adaptation. The Vitis AI toolchain does not natively support segmentation network compilation, and several architectural components produce compilation failures or CPU fallbacks that violate latency bounds and introduce non-deterministic execution. The deployment targets a Zynq UltraScale+ MPSoC with the DPUCZDX8G ISA1 B4096 DPU. Three categories of adaptation achieve full-graph compilation:

a) Activation function replacement: YOLOv8’s SiLU and Swish activations require element-wise sigmoid, which is unsupported or poorly quantized on the target DPU. We replace all activations with HardSwish, which is natively supported and preserves nonlinear expressiveness [13]. Alternatives were systematically evaluated: ReLU variants introduced excessive sparsity degrading mask completeness, while HardSigmoid produced asymmetric quantization noise in deeper layers.

b) Elimination of unsupported operations: The attention mechanism’s scaled dot-product computation employs a DPU-unsupported operator (nndct elemwise mul), resulting in CPU fallback. We redesign this as a fully DPUcompatible additive projection that preserves cross-channel feature reweighting using only supported operations [6].

c) Static tensor dimensions: Dynamic shapedependent operations (nndct shape, aten::meshgrid, nndct arange) default to CPU execution. We replace runtime shape queries with fixed dimensions derived from the network’s multi-scale feature pyramid structure, enabling most operations to execute on the DPU under the target compilation constraints and ensuring bounded execution latency.

Together, these adaptations enable full-graph XMODEL compilation, eliminate dynamic execution paths, and support highly predictable and reproducible inference latency under the target toolchain constraints.

## C. Activation Variance Informative Sampling (AVIS)

PTQ accuracy is sensitive to calibration data selection: non-representative inputs may cause activation saturation or dynamic range under-utilization, leading to unpredictable accuracy degradation [14], [37]. We propose AVIS, a deterministic, label-free strategy that selects highly informative inputs for calibration stability by evaluating activation variance across network layers. The core hypothesis is that inputs producing higher spatial variance in intermediate feature maps tend to activate a broader subset of learned representations, which empirically improves calibration stability and INT8 scale estimation.

![](images/fc555b2094dff48f2d581a7b0899f01112512237584cb38695591912a1329f95.jpg)  
Fig. 1: System architecture. Left: Offline phase—training, AVIS calibration selection, INT8 quantization with bias correction, and XMODEL (Vitis-AI model representation) compilation. Right: Onboard phase—DPU-accelerated convolution and deterministic CPU post-processing (non-maximum suppression (NMS), mask reconstruction). All weights are frozen; no in-flight modification is performed.

a) Activation variance scoring: Let $\mathcal { X } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ denote the candidate pool and L the number of layers. For each candidate $x _ { i } { \mathrm { : } }$

$$
s _ { i } = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \mathrm { m e a n } _ { c } \Big ( \mathrm { V a r } _ { H , W } \left( A _ { l } ^ { c } ( x _ { i } ) \right) \Big )\tag{1}
$$

where $A _ { I } ^ { c } ( x _ { i } ) \in \mathbb { R } ^ { H \times W }$ is the spatial activation map.

Higher s<sub>i</sub> is interpreted as indicating inputs that activate a broader subset of learned representations and are therefore more informative for quantization scale estimation.

b) Deterministic Top-K selection: Calibration images are selected via:

$$
\mathcal { X } _ { \mathrm { A V I S } } = \left\{ x _ { i } \ \middle | \ s _ { i } \in \mathrm { T o p K } \left( \{ s _ { j } \ \middle | \ s _ { j } > 0 \} , K \right) \right\}\tag{2}
$$

The positivity constraint excludes near-uniform or underexposed inputs. The resulting set is fully determined by the frozen network’s activation statistics, with no dependence on labels or stochastic sampling, ensuring deterministic calibration under fixed model parameters. AVIS reduces calibration data requirements by approximately 2× relative to random sampling while improving INT8 accuracy, and operates independently of training or architectural modifications.

## D. Quantized Deployment and CPU–DPU Partitioning

Following AVIS calibration, the network undergoes INT8 PTQ with bias correction, compensating for systematic layerwise activation shifts that degrade mask boundary precision [23]. The resulting model is compiled into an XMODEL for DPU deployment. At runtime, the pipeline executes as a structured CPU–DPU hybrid:

• DPU: Backbone, neck, and segmentation head compiled as a single INT8 XMODEL graph with bounded latency.

• CPU: Deterministic post-processing (confidence decoding, NMS, mask reconstruction) with statically defined memory and no dynamic control flow or heap allocation.

This partitioning ensures deterministic, bounded-latency execution at system level by eliminating CPU fallbacks and uncontrolled execution paths. It also enables structured isolation of compute-intensive kernels for downstream radiationaware analysis.

## E. Radiation-Aware Criticality Analysis

This section quantifies vulnerability of each functional block to radiation-induced single-event effects (SEE), enabling targeted allocation of mitigation resources.

1) Scope and Assumptions: The analysis evaluates SEE susceptibility of memory-resident components (RAM and associated runtime buffers) under onboard execution. Missionlevel particle flux and device cross-sections are not explicitly modeled. Flash memory effects are neglected due to demonstrated FPGA robustness [22]. We focus on the functional impact of SEE at the software level. The analysis provides a relative criticality ranking rather than absolute failure probability.

2) Occurrence Model: Memory-level SEE events are approximated as spatially uniform at the memory layout level. The short inference cycle (1.2 s) relative to mission duration supports a time-invariant approximation within each execution window:

$$
p ( x ) = { \frac { 1 } { \mathrm { A r e a } ( R _ { M } \cap R _ { T } ) } } { \mathbf { 1 } } _ { R _ { M } \cap R _ { T } } ( x )\tag{3}
$$

where $R _ { M }$ and $R _ { T }$ denote memory and temporal execution regions. This expression follows from modeling memory residency and execution time as independent uniform distributions over $R _ { M }$ and $R _ { T }$ , respectively, which reduces to a uniform distribution over their intersection.

3) Functional Block Decomposition: We aggregate operations at the library level into functional blocks:

• CPU-static: Persistent allocations (OpenCV, runtime, math libraries), profiled via pmap.

• CPU-dynamic: Heap allocations during postprocessing, profiled via massif.

• DPU: XMODEL kernel footprint and execution time, profiled via vaitrace.

Temporal proportions are extracted via callgrind.

![](images/837d9fe8e949dec8ae3d23a259ee27cb1d86ea9d7749bae16d541c184c3155ce.jpg)  
Fig. 2: Risk assessment pipeline following ECSS methodology.

4) Criticality Classification: Criticality follows ECSS methodology [1] as shown in Fig. 2. Occurrence is derived from measured memory footprint and execution time. Severity reflects functional impact under fault conditions and is assigned based on detectability and propagation potential. Dynamic heap allocations are conservatively assigned high severity due to susceptibility to silent corruption and lack of deterministic recovery mechanisms. After mitigation, severity is reassigned based on software stack hierarchy, where lower-level libraries exhibit higher propagation impact.

The final criticality metric is defined as follows:

$$
C = O \times S\tag{4}
$$

This formulation provides a first-order risk approximation consistent with standard FMEA practice rather than a physical failure probability model.

## V. EXPERIMENTAL RESULTS

We evaluate the framework across three dimensions: segmentation accuracy under quantization (Section V-A), deployment efficiency (Section V-C), and radiation-informed robustness (Section V-D).

## A. Segmentation Performance

Table II reports segmentation performance across configurations, isolating the effects of platform transfer, quantization, bias correction, and AVIS. Results are reported on a held-out test set representative of the deployment domain.

TABLE II: Segmentation accuracy across platforms and quantization configurations. GPU FP32 serves as the reference baseline (0% loss). All INT8 models run on the hybrid CPU–DPU platform.
<table><tr><td>Platform</td><td>Model</td><td>mAP / IoU</td><td>mAP drop</td></tr><tr><td>GPU</td><td>Y8m FP32</td><td>0.802 / 0.782</td><td>0.0%</td></tr><tr><td></td><td>CPU-DPU Y8m INT8 (RC)</td><td>0.749 / 0.731</td><td>-6.6%</td></tr><tr><td></td><td>CPU-DPU Y8m INT8 + Bias Corr. (RC)</td><td>0.768 / 0.741</td><td>-4.2%</td></tr><tr><td></td><td>CPU-DPU Y8m INT8 + AVIS + Bias Corr. 0.786 / 0.767</td><td></td><td>-2.0%</td></tr></table>

Note: RC = Random Calibration, Y8m = YOLOv8m

TABLE III: End-to-end inference latency, power, and energy per frame.
<table><tr><td>Platform</td><td>Latency [ms]</td><td>Power [W]</td><td>Energy [J]</td></tr><tr><td>GPU</td><td>121</td><td>11.8</td><td>1.43</td></tr><tr><td>CPU</td><td>537</td><td>8.3</td><td>4.46</td></tr><tr><td>CPU-DPU</td><td>309</td><td>5.7</td><td>1.76</td></tr></table>

a) Platform transfer: FP32 results across CPU and GPU show negligible variation, indicating that segmentation accuracy is primarily sensitive to quantization rather than execution backend.

b) Quantization impact: INT8 with random calibration incurs a 6.6% relative mAP reduction (0.749 vs. 0.802), concentrated in mask boundary precision and small-object completeness, consistent with activation range misestimation under uninformative calibration.

c) Bias correction: Adding bias correction improves mAP to 0.768 (+2.5%), indicating that systematic layer-wise activation shifts contribute measurably to INT8 accuracy degradation and can be partially compensated.

d) AVIS + bias correction: The combined strategy achieves the highest INT8 performance (mAP = 0.786, IoU = 0.767), recovering 69.8% of the accuracy loss observed under random calibration relative to the FP32 baseline. This improvement is attributed to two complementary effects: AVIS improves activation range estimation via informative sample selection, while bias correction compensates residual distributional shifts in layer-wise activations.

## B. Qualitative Analysis

Figure 3 compares segmentation outputs under representative low-light conditions. INT8 with random calibration produces mask fragmentation and boundary distortion due to activation clipping in fine-detail layers. AVIS + bias correction improves mask completeness and geometric fidelity, producing outputs closer to FP32-level structure, consistent with the quantitative trends in Table II.

## C. Inference Efficiency

The GPU achieves the lowest latency (121 ms) but its 11.8 W power draw exceeds the LuNiS Rover compute budget, making it unsuitable for typical lunar micro-rover power budgets. The CPU-only baseline exhibits the highest latency (537 ms) and energy (4.46 J/frame). The hybrid CPU-DPU configuration achieves 309 ms at 5.7 W (1.76 J/frame), corresponding to a 60% energy reduction versus CPU-only. Under the X Rover operational profile (0.1 m/s velocity, 5 s capture interval, 0.5 m per frame), this provides more than 16× temporal margin for hazard detection within the perception-action cycle. All reported latencies include full on-device execution, covering DPU inference and CPUbased post-processing, with no offloading.

![](images/883283c14149756a2af68418b62ca16c1b93a002ca76ce7c2c6ee05cdcd21e3d.jpg)  
Fig. 3: Instance segmentation under low-light lunar conditions. Left to right: input, FP32 reference, INT8 with random calibration (mask fragmentation and boundary distortion), INT8 with AVIS + bias correction producing outputs closer to FP32 reference-level structure. Bottom: zoomed boundary crops.

## D. Criticality Analysis

Function-wise criticality analysis was performed following Section IV-E, using vaitrace, pmap, massif, and callgrind for profiling. The DPU-to-CPU execution time ratio is computed as the ratio of DPU kernel execution time to CPU postprocessing time, measured using vaitrace and averaged over the evaluation set, yielding 0.2644. The DPU executes as a monolithic kernel and is modeled as a single functional entity.

![](images/08621cf3417c483483271babb01c116d6aa9936f515e629f5a371e9bffa9d697.jpg)  
Fig. 4: Per-library criticality under static and dynamic allocation. The DPU kernel exhibits the highest estimated criticality due to memory footprint and modeled propagation sensitivity.

TABLE IV: Per-block occurrence(O) and severity(S) ratings (1–4 scale; 4 = highest).
<table><tr><td></td><td colspan="2">0</td><td colspan="2">S</td></tr><tr><td>Block</td><td>Stat.</td><td>Dyn.</td><td>Stat.</td><td>Dyn.</td></tr><tr><td>DPU Kernel</td><td>4</td><td>0</td><td>4</td><td>4</td></tr><tr><td>C/C++ Backbone</td><td>2</td><td>0</td><td>3</td><td>4</td></tr><tr><td>OpenCV</td><td>2</td><td>2</td><td>2</td><td>4</td></tr><tr><td>Xilinx DPU Runtime</td><td>2</td><td>3</td><td>4</td><td>4</td></tr><tr><td>Math Libraries</td><td>2</td><td>0</td><td>2</td><td>4</td></tr></table>

a) Criticality distribution: Figure 4 and Table IV show the per-block criticality distribution. The DPU kernel contributes most to the estimated system-level criticality due to its memory footprint and modeled propagation sensitivity, followed by the XIR runtime due to semantic fault sensitivity, while OpenCV and math libraries exhibit lower impact due to more localized fault propagation. Severity assignment follows a discrete ECSS-inspired ordinal scale, where 1 denotes negligible impact and 4 represents system-level or non-recoverable failure modes under radiation-induced faults. Dynamic allocations are assigned higher severity values due to reduced observability and recovery complexity under heap corruption, whereas the DPU kernel is assigned the maximum static severity (S = 4) due to its dominant propagation potential.

TABLE V: Cumulative mitigation effect on global criticality. Each row includes all preceding mitigations.
<table><tr><td>Case</td><td>C</td><td>Δ</td></tr><tr><td>Unmitigated</td><td>0.3389</td><td></td></tr><tr><td>+ Static-only allocation</td><td>0.3161</td><td>-6.7%</td></tr><tr><td>+ Selective TMR</td><td>0.2610</td><td>-17.4%</td></tr><tr><td>+ EDAC &amp; memory scrubbing</td><td>0.2316</td><td>-11.3%</td></tr><tr><td>Cumulative</td><td></td><td>-31.7%</td></tr></table>

Note: EDAC = Error Detection and Correction, ∆ = relative to the preceding row.

b) Progressive mitigation: Progressive mitigation is evaluated in Table V, while Figure 5 visualizes the cumulative reduction in system-level criticality across successive mitigation stages. Static allocation reduces baseline criticality by 6.7% by eliminating heap-associated risk. Selective TMR further reduces system vulnerability by 17.4% by protecting high-impact compute blocks, while EDAC and memory scrubbing contribute an additional 11.3% reduction by mitigating transient memory faults. Overall, these staged interventions reduce global criticality from 0.3389 to 0.2316, corresponding to a total reduction of 31.7% relative to the unmitigated baseline.

![](images/2541325c3972d667df78c06f522aacb9c179f3d55ec290559adf309390bf56d2.jpg)  
Fig. 5: Cumulative criticality reduction under progressive mitigation. The DPU kernel and XIR runtime exhibit the largest reductions, validating the effectiveness of function-wise prioritization.

AVIS and criticality-driven mitigation address complementary robustness dimensions. AVIS reduces sensitivity to quantization-induced perturbations in activation distributions, while criticality-driven mitigation reduces the estimated impact of radiation-induced faults at the system level under the adopted model.

## VI. CONCLUSION

This work presents a deployment-oriented framework for instance segmentation on resource-constrained lunar platforms. By integrating calibration-aware quantization, deterministic hardware execution, and function-wise fault anal ysis, the approach improves both inference reliability and system robustness. AVIS with bias correction recovers 69.8% of quantization-induced accuracy loss (mAP 0.786 vs. 0.749 random calibration and 0.802 FP32), while enabling deterministic DPU execution at 309 ms and 5.7 W. Functionwise criticality analysis identifies the DPU kernel and Xilinx runtime as dominant contributors to system-level risk, and targeted mitigation—static allocation, selective TMR, and EDAC—reduced global criticality by 31.7%. These mechanisms are complementary: AVIS reduces quantization sensitivity by tightening activation ranges, while criticalitydriven hardening mitigates radiation-induced faults that exceed model tolerance. The framework satisfies four key deployment requirements: quantization fidelity, deterministic execution, data-efficient calibration, and radiation-aware robustness. These results demonstrate that calibration-aware quantization and function-level analysis can significantly improve the reliability of AI perception under the extreme constraints of space robotics, where safety-critical autonomy is essential for rover operation.

a) Limitations: The criticality analysis relies on analytical fault modeling rather than physical fault injection; empirical validation under proton or heavy-ion irradiation remains necessary. AVIS assumes a representative offline dataset; performance under distributional shifts (e.g., unfamiliar terrain types not included in calibration) remains unexplored. The 309 ms latency is sufficient for the current 0.1 m/s profile but would require further optimization for faster traversal or temporal fusion.

b) Future work: Key directions include adaptive quantization scales responsive to accumulated radiation dose, lightweight uncertainty estimation for per-prediction confidence bounds, and multi-sensor fusion for landing-phase redundancy. Formal bit-flip injection campaigns comparing AVIS-calibrated and randomly calibrated models would empirically validate the hypothesized relationship between activation range tightness and radiation fault tolerance.

## ACKNOWLEDGMENT

This research was done in the SAMLER-KI project, funded by the German Federal Ministry for Economic Affairs and Energy (BMWE, grant number 50RA2203A).

## REFERENCES

[1] “Failure Modes, Effects (and Criticality) Analysis (FMEA/FMECA),” European Cooperation for Space Standardization (ECSS), ESTEC, P.O. Box 299, 2200 AG Noordwijk, The Netherlands, Space Product Assurance Standard ECSS-Q-ST-30-02C, Mar. 2009.

[2] L. M. Amaya-Mej´ıa, M. Ghita, J. Dentler, M. Olivares-Mendez, and C. Martinez, “Visual servoing for robotic on-orbit servicing: A survey,” in 2024 International Conference on Space Robotics (iSpaRo). IEEE, June 2024, p. 178–185.

[3] M. S. Bahraini, A. Zenati, and N. Aouf, “Autonomous cooperative visual navigation for planetary exploration robots,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 9653–9658.

[4] P. Blacker, C. Bridges, and S. Hadfield, “Rapid prototyping of deep learning models on radiation hardened cpus,” 07 2019, pp. 25–32.

[5] C. Cai, Y. Chi, and L. Cai, “Radiation effects of advanced electronic devices and circuits, 2nd edition,” Electronics, vol. 14, no. 14, 2025.

[6] T. Cai, S. Luo, K. Xu, D. He, T.-Y. Liu, and L. Wang, “Graphnorm: A principled approach to accelerating graph neural network training,” 2021.

[7] E. Caroselli, F. Belien, A. Falke, and F. Curti, “Deep learningbased passive hazard detection for asteroid landing in unexplored environment,” 02 2022.

[8] T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Yan, M. Cowan, H. Shen, L. Wang, Y. Hu, L. Ceze, C. Guestrin, and A. Krishnamurthy, “Tvm: An automated end-to-end optimizing compiler for deep learning,” 2018.

[9] M. Durner, W. Boerdijk, Y. Fanger, R. Sakagami, D. L. Risch, R. Triebel, and A. Wedler, “Autonomous rock instance segmentation for extra-terrestrial robotic missions,” in 2023 IEEE Aerospace Conference, 2023, pp. 01–14.

[10] F. D’Aniello, M. Tettamanti, S. A. A. Shah, S. Mattiazzo, S. Bonaldo, V. Vadala, and A. Baschirotto, “Single-event upset characterization of\` a shift register in 16 nm finfet technology,” Electronics, vol. 14, no. 7, 2025.

[11] P. Glaser, J. Oberst, G. Neumann, E. Mazarico, E. Speyerer, and ¨ M. Robinson, “Illumination conditions at the lunar poles: Implications for future exploration,” Planetary and Space Science, vol. 162, 07 2017.

[12] J. Goodwill, C. Wilson, and J. MacKinnon, “Chapter 16 - current ai technology in space,” in Precision Medicine for Long and Safe Permanence of Humans in Space, C. Krittanawong, Ed. Academic Press, 2025, pp. 239–250.

[13] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” 2017.

[14] I. Hubara, Y. Nahshan, Y. Hanani, R. Banner, and D. Soudry, “Accurate post training quantization with small calibration sets,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 4466–4475.

[15] M. Isenkul, “Energy-aware deep learning for real-time video analysis through pruning, quantization, and hardware optimization,” Journal of Real-Time Image Processing, vol. 22, 06 2025.

[16] N. Jonckers, T. Vinck, P. Karsmakers, and J. Prinzie, “Analysis of single event induced bit faults in a deep neural network accelerator pipeline,” arXiv, 2025.

[17] S. Kosambe, “Chandrayaan-2: India’s second lunar exploration mission,” Journal of Aircraft and Spacecraft Technology, vol. 3, pp. 221– 236, 01 2019.

[18] S. Krishnan, A. Elmore, M. Franklin, J. Paparrizos, Z. Shang, A. Dziedzic, and R. Liu, “Artificial intelligence in resource-constrained and shared environments,” ACM SIGOPS Operating Systems Review, vol. 53, pp. 1–6, 07 2019.

[19] B. Kuang, C. Gu, Z. A. Rana, Y. Zhao, S. Sun, and S. G. Nnabuife, “Semantic terrain segmentation in the navigation vision of planetary rovers—a systematic literature review,” Sensors, vol. 22, no. 21, 2022.

[20] A. Kuzmin, M. Nagel, M. van Baalen, A. Behboodi, and T. Blankevoort, “Pruning vs quantization: Which is better?” 2024.

[21] J. Liu, S. Liu, Y. Shao, X. Wan, and H. Zhao, “Mars terrain semantic segmentation using zhurong rover imagery based on transfer learning of historical mission data,” in 2022 International Conference on Service Robotics (ICoSR), 2022, pp. 139–144.

[22] Microchip Technology Inc., “Understanding single event effects (sees) in fpgas: A backgrounder,” Microchip Technology Inc.,” Technical Backgrounder, August 2011.

[23] M. Nagel, M. van Baalen, T. Blankevoort, and M. Welling, “Data-free quantization through weight equalization and bias correction,” 2019.

[24] T. Pang, C. Du, and J. Zhu, “Robust deep learning via reverse crossentropy training and thresholding test,” ArXiv, vol. abs/1706.00633, 2017.

[25] G. Petrakis and P. Partsinevelos, “Lunar ground segmentation using a modified u-net neural network,” 09 2023.

[26] P. Prasad and A. Hameed, “Autonomous ai in space exploration: Navigating the challenges of off-earth missions,” International Journal of Scientific Research and Engineering Trends, 01 2024.

[27] R. Prete, P. Thind, A. Mazzeo, M. Whitley, L. Papa, N. Longepe, and G. Meoni, “Optimizing deep learning models for on-orbit deployment through neural architecture search,” 05 2025.

[28] N. C. Prieur, B. Amaro, E. Gonzalez, H. Kerner, S. Medvedev, L. Rubanenko, S. C. Werner, Z. Xiao, D. Zastrozhnov, and M. G. A. Lapotre, “Automatic characterization of boulders on planetary sur-ˆ faces from high-resolution satellite images,” Journal of Geophysical Research: Planets, vol. 128, no. 11, p. e2023JE008013, 2023, e2023JE008013 2023JE008013.

[29] M. Quoos, S. Kay, J. Wajoras, R. Field, E. Ntagiou, F. Mohammad, and Y. Gao, “Survey on ai-enabled computer vision technologies and applications for space robotic missions,” Journal of Field Robotics, pp. n/a–n/a, 01 2026.

[30] Y. Shao, J. Wang, X. Han, Y. Li, Y. Li, and Z. Tao, “Research on spaceborne neural network accelerator and its fault tolerance design,” Remote Sensing, vol. 17, no. 1, 2025.

[31] S. Shete, “Lunar rocks and craters dataset for instance segmentation (yolo and coco format),” Jan. 2025.

[32] S. Shete, R. Dom´ınguez, R. Selvaraju, M. D. L. Alvarez, and F. Kirch-<sup>´</sup> ner, “Prediction-based tip-over prevention for planetary exploration rovers,” Engineering Proceedings, vol. 90, no. 1, 2025.

[33] C. Stout, “Chandrayaan-3: A landmark mission of resilience,” 12 2025.

[34] R. Tedeschi, L. Ghionda, and et al., “Safe-neureka: a hybrid modular redundant dnn accelerator for on-board satellite ai processing,” arXiv, 2026.

[35] A. Williams and S. Palo, “Issues and implications of the thermal control system on responsive space missions,” p. 13, 08 2006.

[36] M. Williams and N. Aletras, “On the impact of calibration data in posttraining quantization and pruning,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2024, p. 10100–10118.

[37] Y. Zhou, S.-M. Moosavi-Dezfooli, N.-M. Cheung, and P. Frossard, “Adaptive quantization for deep neural network,” vol. 32, Apr. 2018.