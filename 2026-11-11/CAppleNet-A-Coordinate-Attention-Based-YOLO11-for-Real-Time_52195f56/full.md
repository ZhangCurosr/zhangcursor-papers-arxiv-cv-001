Article

# CAppleNet: A Coordinate Attention-Based YOLO11 for Real-Time Apple Stem Cavity and Calyx Detection

Alican Güne¸s<sup>1,2</sup> , Robert Selim Hill<sup>2</sup> , Gökhan Hasano˘glu<sup>2</sup> and Murat Kurt <sup>1,∗</sup>

International Computer Institute, Ege University, <sup>˙</sup>Izmir, Türkiye; alican.acg@gmail.com (A.G.); murat.kurt@ege.edu.tr (M.K.)

2 Tarend, <sup>˙</sup>Izmir, Türkiye; alican.acg@gmail.com (A.G.); rs.hill@tarend.com (R.S.H.); g.hasanoglu@tarend.com (G.H.)

Correspondence: murat.kurt@ege.edu.tr; Tel.: +90-232-311-3227 (M.K.)

## Abstract

Automated fruit grading, sorting, and orientation systems require real-time, accurate detection of fruit-level morphological landmarks rather than the fruit boundary alone. This paper presents a custom, fully self-collected and self-annotated dataset of 1506 apple images (1054 training, 301 validation, 151 test) captured under controlled, fixed-camera conditions and annotated for three target classes: fruit body, stem cavity, and calyx. A YOLO11s detector trained with default hyperparameters is adopted as the baseline, and a systematic, one-factor-at-a-time ablation study is conducted across input resolution, data augmentation policy, optimizer configuration, and lightweight architectural enhancement to identify the factors that most improve detection of these fine-grained landmarks under high inter-cultivar color and texture variability. Neither increased input resolution, nor any of six tested augmentation variants, nor manual optimizer tuning improved on the untuned baseline (F1 = 0.9047); only a same-slot architectural substitution of the backbone’s C2PSA block with Coordinate Attention (CA) matched the baseline’s overall F1 (0.9038) while reducing parameter count by 10.2% and achieving the study’s best mAP@0.5 (0.935) and best stem-cavity F1 (0.8217). A confidence-threshold sweep identified 0.15 as the deployment operating point for this model, and a confirmatory experiment combining the CA-swap architecture with the best-performing augmentation variant showed the two effects do not combine additively, reversing the architectural gain rather than compounding it. The resulting design recommendations — the CA-swap architecture alone, at a 0.15 confidence threshold — target real-time apple landmark detection for orientation-aware industrial sorting and alignment systems.

Keywords: deep learning; object detection; agricultural automation; apple stem cavity detection; apple calyx detection; ablation study; mean average precision (mAP); YOLO11; machine vision; coordinate attention

Received: Revised: Accepted: Published:

Copyright: © 2026 by the authors. Submitted to Sensors for possible open access publication under the terms and conditions of the Creative Commons Attribution (CC BY) license.

## 1. Introduction

Automated fruit grading, sorting, and orientation systems depend on the accurate, real-time localization of fruit-level morphological landmarks rather than the fruit boundary alone. For apples specifically, reliable detection of the stem cavity and calyx (flower end), together with the overall fruit body, is a prerequisite for orientation-aware packaging, defect inspection, and robotic handling, where misidentification of these regions directly degrades downstream mechanical and quality-control decisions.

Deep-learning-based object detectors have progressively replaced classical imagesegmentation pipelines for fruit localization tasks [1,2]. Early convolutional approaches demonstrated the feasibility of in-field and orchard-based fruit detection and yield estimation [1,3,4], while the introduction of the YOLO family provided a favorable accuracy–speed trade-off that made real-time deployment practical [5,6]. Building on this foundation, a substantial body of work has adapted successive YOLO variants specifically to apple detection: improved YOLOv3 and YOLOv5 variants for orchard and picking-robot scenarios [3,7], channel-pruned and lightweight YOLOv5/YOLOv4 architectures for fruitlet detection and complex backgrounds [8,9], ShuffleNetV2-YOLOX and attention-augmented YOLOv5 (CA-YOLOv5) models for natural-environment robustness [10,11], and, most recently, lightweight embedded-system and improved YOLO11-based detectors such as YOLO-WAS [12,13]. Broader surveys of the YOLO family and its agricultural applications [14,15] confirm a consistent trend toward progressively lighter, faster architectures without sacrificing detection accuracy.

Closest to the present work, Baek et al. proposed AS-YOLO, which combines a Ghost Bottleneck design with a global attention mechanism specifically for apple stem-region segmentation [16], confirming that fine-grained, sub-fruit landmarks such as the stem region require architectural consideration beyond whole-fruit bounding-box detection. However, this and related studies typically report a single, already-optimized architecture rather than a systematic, controlled comparison of the individual design choices that led to it. In particular, it remains unclear which factors — data augmentation policy, input resolution, optimizer configuration, or lightweight architectural modules — contribute most to detecting small, fine-grained landmarks such as the stem cavity and calyx, especially under an imaging setup in which camera distance and illumination are fixed, but fruit appearance (color, blemishes, bicoloration) varies substantially across cultivars.

To address this gap, the present study makes the following contributions:

Custom, fully self-collected and self-annotated dataset: Unlike studies relying on public, synthetic, or orchard-captured imagery [1,2,4], all images in this study were captured under controlled, fixed-camera conditions and manually annotated by the authors on the Roboflow platform, yielding 1506 images (1054 training, 301 validation, 151 test) across three classes: fruit body, stem cavity, and calyx.

A rigorously defined baseline: A YOLO11s model trained with the architecture’s default hyperparameters is adopted as the baseline, avoiding premature tuning that would confound the subsequent ablation analysis.

A systematic, one-factor-at-a-time ablation study: Starting from the baseline, individual components — augmentation policy, input resolution, optimizer configuration, and lightweight architectural enhancements inspired by prior attention-based appledetection work [11,16] — are each varied in isolation to quantify their individual contribution to detection performance.

Design recommendations for controlled-environment deployment: The resulting analysis yields practical guidance for real-time apple landmark detection systems operating under fixed-distance, fixed-illumination imaging conditions with high intraclass appearance variability.

## 2. Related Work

## 2.1. Deep Learning for Fruit Detection and the Evolution of YOLO

Early fruit-detection research relied on image segmentation and classical computervision pipelines to localize fruit in orchard imagery and estimate yield [1], with related efforts extending to RGB-D-based in-field detection for other crops such as citrus [2]. The subsequent shift toward single-stage deep detectors, beginning with the YOLO family, was motivated by the need to jointly satisfy accuracy and real-time inference constraints [5]. YOLOv4 in particular established an architecture-level balance between speed and accuracy that was widely adopted as a backbone for downstream agricultural detection tasks [5], and related single-stage detectors were concurrently adapted to other fruit types, such as tomato [6], demonstrating the generality of the YOLO paradigm beyond apples.

## 2.2. YOLO-Based Apple Detection

A substantial line of work has adapted successive YOLO versions specifically to apple detection under varying acquisition conditions. Tian et al. improved YOLOv3 for apple detection across different growth stages in orchards, addressing appearance changes over the growing season [3]. Kang and Chen combined detection and segmentation for apple harvesting using a visual sensor mounted on a harvesting platform [4]. Yan et al. proposed a real-time apple-target detection method based on an improved YOLOv5 for picking-robot applications [7], while Wang and He introduced a channel-pruned YOLOv5s model to accelerate apple fruitlet detection prior to fruit thinning [8]. Zhang et al. developed a lightweight YOLOv4 variant to maintain detection accuracy in complex orchard backgrounds [9], and Ji et al. combined ShuffleNetV2 with YOLOX to achieve a favorable speed–accuracy trade-off for picking-robot deployment [10]. More recently, Yang et al. proposed CA-YOLOv5, which incorporates coordinate attention to improve apple detection in natural, unconstrained environments [11], Olguín-Rojas et al. presented a lightweight YOLO-based architecture targeting embedded-system deployment [12,13]. Comprehensive surveys of the YOLO family [14] and of deep learning applications across the apple production cycle [15] situate these contributions within a broader trend toward progressively lighter and faster architectures for agricultural deployment.

## 2.3. Landmark-Level and Attention-Based Apple Detection

While the studies above primarily address whole-fruit bounding-box detection, fewer works target sub-fruit morphological landmarks. The most directly related study is AS-YOLO, in which Baek et al. combined a Ghost Bottleneck module with a global attention mechanism to improve apple stem-region segmentation [16]. This confirms that finegrained landmarks such as the stem region benefit from architectural modifications beyond the detection backbones used for whole-fruit localization, a finding consistent with the coordinate-attention enhancement explored for whole-apple detection in natural environments [11]. However, neither study isolates the individual contribution of each architectural or training-level modification through a controlled ablation procedure.

## 2.4. Ablation Studies and Research Gap

Across the reviewed literature, the predominant reporting practice is to present a single, already-optimized model configuration — whether a pruned backbone [8], a lightweight variant [9,12,13], or an attention-augmented architecture [11,16] — without a systematic, controlled comparison of the individual design choices that produced it. Consequently, it remains unclear, particularly for small and fine-grained landmark classes such as the stem cavity and calyx, which individual factors (augmentation policy, input resolution, optimizer configuration, or lightweight architectural modules) drive the reported performance gains, and whether these effects generalize to imaging conditions with fixed camera distance and controlled illumination but high intra-class color and texture variability. The present study addresses this gap through a one-factor-at-a-time ablation study built on a rigorously defined YOLO11s baseline.

## 3. Materials and Methods

## 3.1. Dataset Acquisition and Annotation

Unlike the majority of the reviewed studies, which rely on public, synthetic, or orchardcaptured datasets [1,2,4], the dataset used in this study was collected entirely by the authors and is not derived from any existing public source. Images were captured under controlled, indoor conditions with a fixed camera-to-fruit distance and consistent illumination, eliminating variability due to ambient lighting or camera perspective. Within this controlled acquisition setup, the apples themselves exhibit substantial natural appearance variability, including bicoloration, surface blemishes, and a range of cultivar-dependent hues from deep red to pale yellow-green, which the subsequent ablation study is designed to account for.

All images were manually annotated by the authors using the Roboflow platform and exported in YOLO format, with three target classes: fruit body, stem cavity and calyx. The annotated dataset comprises 1506 images in total, partitioned into 1054 training, 301 validation, and 151 test images.

## 3.2. Baseline Model Configuration

Following the trend toward lightweight YOLO variants for real-time agricultural deployment [8–10,12,13], the small variant of the YOLOv11 architecture (YOLO11s) is adopted as the baseline detector. To ensure that the subsequent ablation study isolates the effect of each individually varied factor, the baseline is trained using the architecture’s default hyperparameters without prior tuning: an input resolution of 640 × 640 pixels, the framework’s default optimizer and learning-rate schedule, and the default augmentation policy. Training is run for up to 100 epochs, consistent with established practice for datasets of comparable scale [5,7], with early stopping applied via a validation-performance patience criterion (patience = 30 epochs without improvement). This patience value is held fixed across the baseline and every subsequent ablation run; it is a training-control setting rather than an ablated factor, so fixing it does not compromise the one-factor-at-a-time design. This untuned configuration serves as the fixed reference point against which every subsequent ablation variant is compared.

## 3.3. Ablation Study Design

To decouple confounding interactions and establish the precise marginal utility of each engineering intervention, the ablation experiments follow a strict One-Factor-at-a-Time (OFAT) protocol. In complex computer vision pipelines, multi-parameter grid searches or simultaneous feature stacking often obscure whether performance improvements stem from genuine representational enhancements or accidental regularizing cross-cancellations. By modifying exactly one operational variable per experiment relative to the baseline configuration (Section 3.2) while clamping all remaining parameters to their default states, performance variations can be attributed unambiguously to that isolated intervention. This controlled experimental methodology directly addresses the methodological limitation identified in Section 2.4, where prior agricultural vision studies predominantly present composite, post-hoc optimized models without isolating the causal trajectory of each individual component [8,9,11–13,16].

The experimental dimensions were selected to systematically probe the trade-offs between spatial representation, photometric robustness, convergence dynamics, and computational efficiency in high-speed sorting machinery:

1. Spatial Resolution Scaling: Small anatomical landmarks occupy only a minor fraction of the total image area. Higher spatial resolutions increase sampling density across fine morphological contours (e.g., the concentric creasing of the stem cavity), but scale computational complexity quadratically and may introduce spatial quantization noise in self-attention layers.

2. Photometric and Geometric Augmentations: In factory grading lines operating with fixed-focus industrial optics and constant LED strobes, geometric viewpoint variance is mechanically constrained. However, biological variations (e.g., anthocyanin blush gradients, russeting, and chlorophyll fading across cultivars) represent extreme photometric diversity, raising the question of whether synthetic color perturbations aid or corrupt subtle structural landmarks.

3. Optimization Trajectory: Adaptive gradient methods interact sensitively with smallto-medium agricultural datasets. Evaluating hand-crafted weight decay schedules against automated framework heuristics determines whether convergence stability can be decoupled from empirical overparameterization.

4. Direction-Aware Structural Attention: Morphological depressions possess pronounced directional symmetries. Evaluating whether coordinate-aware attention modules can outperform isotropic self-attention blocks without computational inflation addresses the core embedded latency requirement.

5. Post-Hoc Decision Thresholding: Actuator triggering in industrial mechanical sorters operates under asymmetric cost matrices, where landmark false negatives lead directly to orientational disorientation and line recirculation.

The five investigated factors, their specific parametric alterations, and their operational hypotheses are summarized in Table 1.

Table 1. Ablation factors, tested variants, and engineering rationale.
<table><tr><td># Factor</td><td></td><td>Variants Tested</td><td>Rationale</td></tr><tr><td></td><td>1 Input resolution</td><td>640 × 640; 768 × 768</td><td>High resolution may improve small landmark sampling but incurs quadratic compute scaling.</td></tr><tr><td></td><td>2 Augmentation policy</td><td>CutMix; Mixup; HSV jitter; Rota- tion (15°)</td><td>Fixed industrial geometry versus biological surface hue and texture variability across cultivars.</td></tr><tr><td></td><td>3 Optimizer / LR</td><td>Default (auto) vs. AdamW + cosine LR</td><td>Assesses convergence stability and risk of overfitting on a clean, mod- erately sized dataset.</td></tr><tr><td></td><td>4 Architectural modules</td><td>Same-slot Coordinate Attention (CA) swap</td><td>Position-sensitive channel atten- tion replaces isotropic self-attention (C2PSA) for landmarks.</td></tr><tr><td></td><td>5 Decision threshold</td><td>Post-hoc confidence sweep (0.10 ≤ τ ≤ 0.70)</td><td>Balances precision and recall against asymmetric costs of sorting actuator false negatives.</td></tr></table>

Experimental Rationale. The structured breakdown in Table 1 establishes an isolated empirical progression across all critical phases of the visual pipeline. By sequentially interrogating data representation (Factor 1), data manifold regularization (Factor 2), loss-surface optimization (Factor 3), structural inductive bias (Factor 4), and deployment operating semantics (Factor 5), each component’s distinct contribution to fine-grained sub-fruit localization is explicitly mapped without conflation.

## 3.4. Hardware and Software Infrastructure

The baseline model and all subsequent ablation variants were trained and evaluated on a workstation equipped with a 13th Gen Intel Core i7-13700HX CPU and an NVIDIA GeForce RTX 5060 Laptop GPU. Training was implemented in Python using the PyTorch deep learning framework and the Ultralytics YOLO11 implementation.

## 4. Results

## 4.1. Baseline Model Performance

Training of the baseline YOLO11s model converged before reaching the 100-epoch budget: early stopping (patience = 30) triggered at epoch 88, with the best checkpoint retained from epoch 58. Table 2 reports precision, recall, mAP@0.5, and mAP@0.5:0.95 for each class on the held-out test set (151 images, 233 annotated instances).

Table 2. Baseline performance on the held-out test split.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.980</td><td>1.000</td><td>0.992</td><td>0.991</td><td>0.9901</td></tr><tr><td>Calyx</td><td>0.909</td><td>0.923</td><td>0.918</td><td>0.507</td><td>0.9160</td></tr><tr><td>Stem cavity</td><td>0.672</td><td>0.963</td><td>0.853</td><td>0.382</td><td>0.7918</td></tr><tr><td>All classes</td><td>0.854</td><td>0.962</td><td>0.921</td><td>0.627</td><td>0.9047</td></tr></table>

Baseline Analysis. As indicated in Table 2, the macro-level fruit body is localized with near-perfect reliability (F1 = 0.9901), indicating that whole-object bounding is virtually solved by modern single-stage backbones under controlled lighting. Conversely, a pronounced performance disparity is evident among the morphological sub-landmarks: while the calyx achieves a strong F1 of 0.9160, the stem cavity demonstrates substantially depressed precision (0.672) and tight IoU precision (mAP@0.5:0.95 = 0.382). This confirms that visual ambiguity, shallow depth-of-field shading, and skin-tone transitions predominantly degrade the smallest anatomical depression, underscoring the stem cavity as the primary bottleneck for orientation alignment.

## 4.2. Ablation Factor 1 — Input Resolution

The 768 × 768 variant was trained with the same baseline hyperparameters (patience = 30), converging via early stopping at epoch 95 with the best checkpoint retained from epoch 65. Table 3 reports per-class test-set performance.

Table 3. Input-resolution variant $( 7 6 8 \times 7 6 8 )$ performance on the test set.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.983</td><td>1.000</td><td>0.985</td><td>0.985</td><td>0.9905</td></tr><tr><td>Calyx</td><td>0.963</td><td>0.808</td><td>0.944</td><td>0.489</td><td>0.9301</td></tr><tr><td>Stem cavity</td><td>0.827</td><td>0.745</td><td>0.790</td><td>0.377</td><td>0.7382</td></tr><tr><td>All classes</td><td>0.924</td><td>0.851</td><td>0.907</td><td>0.617</td><td>0.8896</td></tr></table>

Contrary to the hypothesis in Section 3.3 that a higher input resolution would improve detection of the small stem-cavity and calyx landmarks, increasing the resolution from 640 × 640 to 768 × 768 degrades overall performance relative to the baseline: mAP@0.5 drops from 0.921 to 0.907, mAP@0.5:0.95 from 0.627 to 0.617, and the overall F1-score from 0.9047 to 0.8896. The effect is strongest for the stem cavity class, whose recall falls from 0.963 to 0.745 and mAP@0.5 from 0.853 to 0.790. Precision increases slightly overall (0.854 to 0.924), indicating the higher-resolution model becomes more conservative, missing more true instances rather than producing more false positives. Inference speed is also substantially affected, with throughput nearly halving (142.8 to 141.0 FPS; 7.00 to 7.09 ms per image). Based on this result, $6 4 0 \times 6 4 0$ is retained as the preferred resolution.

## 4.3. Ablation Factor 2 — Augmentation Policy

With the input resolution fixed at 640 × 640, this factor evaluates whether modifying the default photometric, geometric, or blend-based augmentation parameters improves landmark detection across varying apple cultivars.

Table 4. Ablation Factor 2, Run E variant tested.
<table><tr><td>Run</td><td></td><td>Changed parameter(s)</td><td>Default → Tested</td></tr><tr><td>E — HSV</td><td>Stronger hsv_s, hsv_v</td><td></td><td>0.7 / 0.4 → 0.9 / 0.6</td></tr></table>

Run E increases HSV saturation and value jitter beyond the default, testing whether a stronger synthetic color perturbation can further improve robustness to the dataset’s natural inter-cultivar color variation (Table 4). Vertical-flip augmentation was excluded because vertical flipping would produce physically implausible fruit orientations for a task centered on detecting orientation itself.

## Run E — Stronger HSV.

Training converged via early stopping at epoch 85, with the best checkpoint retained from epoch 55.

Table 5. Run E (stronger HSV) performance on the test set.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.983</td><td>1.000</td><td>0.990</td><td>0.990</td><td>0.9915</td></tr><tr><td>Calyx</td><td>0.938</td><td>0.876</td><td>0.964</td><td>0.501</td><td>0.9062</td></tr><tr><td>Stem cavity</td><td>0.747</td><td>0.812</td><td>0.811</td><td>0.330</td><td>0.7781</td></tr><tr><td>All classes</td><td>0.889</td><td>0.896</td><td>0.922</td><td>0.607</td><td>0.8928</td></tr></table>

Run E produces a mixed result relative to the baseline (Table 5). Overall mAP@0.5 edges marginally higher (0.922 vs. 0.921), driven almost entirely by a calyx-class improvement (mAP@0.5 of 0.964 vs. 0.918, the best calyx result of any variant tested so far). However, every other aggregate metric is worse than the baseline: mAP@0.5:0.95 falls from 0.627 to 0.607, recall from 0.962 to 0.896, and overall F1 from 0.9047 to 0.8928. The stem cavity class in particular regresses on mAP@0.5:0.95 (0.382 to 0.330) despite a small recall gain.

## Run F — Rotation augmentation.

Random image rotation is left at 0<sup>◦</sup> (disabled) in the Ultralytics default and in every run above. Since the apple’s orientation relative to the stem/calyx axis is exactly what the downstream alignment system must determine, introducing rotation during training may help the model learn landmark appearance independent of in-plane orientation, rather than relying on an implicit “upright” prior from the fixed camera setup (Table 6).

Table 6. Ablation Factor 2, Run F variant tested.
<table><tr><td>Run</td><td>Changed parameter(s)</td><td>Default → Tested</td></tr><tr><td>F — Rotation on</td><td>degrees</td><td>0.0 → 15.0</td></tr></table>

Run F results.

Training converged via early stopping at epoch 98, with the best checkpoint retained from epoch 68.

Adding rotation augmentation does not exceed the baseline either (Table 7): overall mAP@0.5 (0.912 vs. 0.921), mAP@0.5:0.95 (0.623 vs. 0.627), and F1 (0.8932 vs. 0.9047) are all slightly lower. The pattern resembles earlier non-default-exceeding runs — a modest calyx-class improvement (F1 0.9248, the best calyx F1 of any run so far, edging out even Run E) paired with a stem-cavity precision drop (0.629 vs. 0.672 baseline), suggesting rotation introduces more false positives for the smallest, most ambiguous class even as it helps the model generalize calyx appearance.

Table 7. Run F (rotation on) performance on the test set.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.982</td><td>1.000</td><td>0.994</td><td>0.987</td><td>0.9908</td></tr><tr><td>Calyx</td><td>0.959</td><td>0.893</td><td>0.916</td><td>0.497</td><td>0.9248</td></tr><tr><td>Stem cavity</td><td>0.629</td><td>0.906</td><td>0.827</td><td>0.386</td><td>0.7428</td></tr><tr><td>All classes</td><td>0.857</td><td>0.933</td><td>0.912</td><td>0.623</td><td>0.8932</td></tr></table>

Run G results (Mixup, 0.1).

Training converged via early stopping at epoch 90, with the best checkpoint retained from epoch 60.

Table 8. Run G (mixup on) performance on the test set.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.984</td><td>1.000</td><td>0.990</td><td>0.990</td><td>0.9918</td></tr><tr><td>Calyx</td><td>0.948</td><td>0.904</td><td>0.937</td><td>0.480</td><td>0.9253</td></tr><tr><td>Stem cavity</td><td>0.795</td><td>0.726</td><td>0.833</td><td>0.341</td><td>0.7586</td></tr><tr><td>All classes</td><td>0.909</td><td>0.877</td><td>0.920</td><td>0.604</td><td>0.8923</td></tr></table>

Mixup at a conservative blend strength neither confirms nor refutes the original blurring concern cleanly (Table 8): precision rises to the highest of any run so far (0.909, exceeding the baseline’s 0.854), but recall drops accordingly (0.877 vs. 0.962), and mAP@0.5:0.95 falls to 0.604, among the lowest recorded. Overall F1 (0.8923) remains below the baseline (0.9047). The precision/recall trade-off is consistent with the blurring hypothesis: the model appears to become more conservative and miss more true instances, rather than localizing them less accurately, when trained on blended images.

Run H results (CutMix, 0.1).

Training ran the full 100-epoch budget without early stopping triggering.

Table 9. Run H (CutMix on) performance on the test set.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.983</td><td>1.000</td><td>0.990</td><td>0.989</td><td>0.9915</td></tr><tr><td>Calyx</td><td>0.920</td><td>0.904</td><td>0.925</td><td>0.498</td><td>0.9117</td></tr><tr><td>Stem cavity</td><td>0.782</td><td>0.719</td><td>0.756</td><td>0.350</td><td>0.7492</td></tr><tr><td>All classes</td><td>0.895</td><td>0.874</td><td>0.890</td><td>0.612</td><td>0.8845</td></tr></table>

CutMix produces the weakest result of the two blend-based augmentations (Table 9): mAP@0.5 falls to 0.890 (the lowest of any run tested), and F1 (0.8845) also trails both the baseline and Run G. Unlike mixup, CutMix does not raise precision noticeably (0.895 vs. baseline 0.854) while still losing substantial recall (0.874 vs. 0.962), indicating the hard, rectangular region replacement used by CutMix is more disruptive to this task than mixup’s soft blending — plausibly because it can fully occlude a small landmark (stem cavity or calyx) rather than merely blending its appearance.

Summary of Ablation Factor 2 (Augmentation Policy) — complete.

Six variants (Runs E–H, plus two earlier removal tests, both superseded) were tested against the baseline; none exceeded it on the primary metrics (F1, mAP@0.5:0.95). Results are ranked in Table 10 by overall F1:

Table 10. Ablation Factor 2 summary: all augmentation variants ranked by F1.
<table><tr><td>Configuration</td><td>F1</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td></tr><tr><td>Baseline (default augmentation)</td><td>0.9047</td><td>0.921</td><td>0.627</td></tr><tr><td>Run F — Rotation (15°)</td><td>0.8932</td><td>0.912</td><td>0.623</td></tr><tr><td>Run E — Stronger HSV</td><td>0.8928</td><td>0.922</td><td>0.607</td></tr><tr><td>Run G — Mixup (0.1)</td><td>0.8923</td><td>0.920</td><td>0.604</td></tr><tr><td>Run H — CutMix (0.1)</td><td>0.8845</td><td>0.890</td><td>0.612</td></tr></table>

The default Ultralytics augmentation policy is therefore retained as the final configuration for Ablation Factor 2, and carried forward unchanged into all subsequent ablation factors. While no single modification improved the aggregate metrics, several runs (E, F, G) produced the best calyx-class results observed in this study, suggesting that further gains for that specific class — rather than the model as a whole — may be achievable with a more targeted, class-weighted training strategy, a direction left for future work.

## 4.4. Ablation Factor 3 — Optimizer / LR Schedule

With the augmentation policy fixed at the Ultralytics default (Section 4.3), this factor tests whether a manually configured optimizer and learning-rate schedule outperform the framework’s automatic selection. All other settings (augmentation, imgsz = 640, patience = 30) are held at the baseline configuration (Table 11).

Table 11. Ablation Factor 3 variant tested.
<table><tr><td>Run</td><td>Changed parameter(s)</td><td>Default → Tested</td></tr><tr><td>cosine LR</td><td>I — AdamW + optimizer, lrO, cos_lr</td><td>auto / 0.01 / off → AdamW / 0.001 / on</td></tr></table>

The baseline’s optimizer=auto setting lets Ultralytics select between SGD and AdamW based on dataset size and epoch count, with a linear learning-rate decay. Run I instead forces AdamW with a lower initial learning rate (0.001, a common default for AdamW) and a cosine annealing schedule, testing whether this more commonly handtuned configuration converges to a better optimum than the automatic default.

Run I results (AdamW, lr0 = 0.001, cosine LR).

Training ran the full 100-epoch budget without early stopping triggering.

Table 12. Run I (AdamW + cosine LR) performance on the test set.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.982</td><td>1.000</td><td>0.984</td><td>0.984</td><td>0.9911</td></tr><tr><td>Calyx</td><td>0.936</td><td>0.865</td><td>0.922</td><td>0.513</td><td>0.8992</td></tr><tr><td>Stem cavity</td><td>0.713</td><td>0.875</td><td>0.856</td><td>0.320</td><td>0.7859</td></tr><tr><td>All classes</td><td>0.877</td><td>0.913</td><td>0.921</td><td>0.606</td><td>0.8949</td></tr></table>

The manually configured AdamW optimizer with cosine learning-rate annealing matches the baseline almost exactly on mAP@0.5 (0.921 vs. 0.921) but does not exceed it on either of the primary metrics (Table 12): mAP@0.5:0.95 falls from 0.627 to 0.606, and

F1 from 0.9047 to 0.8949. Precision improves modestly (0.877 vs. 0.854), but at the cost of recall (0.913 vs. 0.962), the same precision/recall trade-off pattern observed in several augmentation runs. The stem cavity class shows the sharpest mAP@0.5:0.95 decline (0.382 to 0.320), the weakest result for that metric on this class across all ablation factors tested to date.

Summary of Ablation Factor 3 (Optimizer / LR Schedule).

Ultralytics’ automatic optimizer selection (optimizer=auto) outperforms the manually configured AdamW + cosine-annealing alternative on every primary metric. The automatic default is therefore retained, and the baseline configuration remains the bestperforming setup identified across all ablation factors examined so far (input resolution, augmentation policy, and optimizer/LR schedule).

## 4.5. Ablation Factor 4 — Lightweight Architectural Module

Rather than stacking an additional attention mechanism onto YOLO11s’s existing C2PSA block (which already performs spatial self-attention at the end of the backbone), this factor swaps C2PSA for Coordinate Attention (CA) in the same architectural slot — same position, same input/output channel count — following the design used in CA-YOLOv5 for apple detection [11]. This same-slot substitution avoids both the index-renumbering risk of inserting a new layer and the redundancy risk of running two different attention mechanisms back to back, isolating the architectural comparison to attention type rather than attention quantity. All other settings (augmentation, imgsz = 640, patience = 30, default optimizer) match the baseline exactly. Figure 1 illustrates (a) where this substitution occurs within the YOLO11s backbone and (b) the internal dataflow of the Coordinate Attention module itself.

Results. Training converged via early stopping at epoch 73, with the best checkpoint retained from epoch 43. The CA-swap model is also lighter than the baseline: 8,451,449 parameters and 20.5 GFLOPs, versus 9,413,961 parameters and 21.4 GFLOPs for the unmodified YOLO11s — a 10.2% parameter reduction and 4.2% fewer GFLOPs, since Coordinate Attention is a cheaper mechanism than C2PSA (Table 13).

Table 13. Per-class test-set results for the Coordinate Attention (CA) swap (Ablation Factor 4).
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.982</td><td>1.000</td><td>0.984</td><td>0.984</td><td>0.9908</td></tr><tr><td>Calyx</td><td>0.903</td><td>0.891</td><td>0.949</td><td>0.494</td><td>0.8969</td></tr><tr><td>Stem cavity</td><td>0.777</td><td>0.872</td><td>0.870</td><td>0.342</td><td>0.8217</td></tr><tr><td>All classes</td><td>0.887</td><td>0.921</td><td>0.935</td><td>0.607</td><td>0.9038</td></tr></table>

Discussion. The CA-swap model achieves an overall F1 of 0.9038, within 0.1 percentage points of the baseline (0.9047) — the closest any tested variant has come to matching the baseline across every ablation factor examined so far — while using roughly 10% fewer parameters. Overall mAP@0.5 is the highest recorded in this study (0.935 vs. baseline 0.921), driven by gains in both calyx (0.949 vs. 0.918) and stem cavity (0.870 vs. 0.853). Most notably, the stem cavity class — the weakest class in every prior configuration — achieves its best F1 of the entire study (0.8217, driven by a large precision gain from 0.672 to 0.777), suggesting that coordinate attention’s position-sensitive channel weighting is specifically well suited to localizing this small, fine-grained, variably textured landmark. The trade-off is a lower mAP@0.5:0.95 (0.607 vs. 0.627) and a modest calyx F1 decline (0.8969 vs. 0.9160), indicating the swap sharpens coarse localization (mAP@0.5) and the hardest class (stem cavity) at some cost to the tightest bounding-box precision (mAP@0.5:0.95) and the already-strong calyx class. Training diagnostics and confusion matrix behavior for this model are shown in Figures 2 and 3.

Summary of Ablation Factor 4. Replacing C2PSA with Coordinate Attention is the strongest result identified across all ablation factors tested (resolution, augmentation, optimizer/LR, and architecture): it matches the baseline’s overall F1 almost exactly, improves mAP@0.5 and stem-cavity detection specifically, and reduces model size and compute — a meaningful advantage for the real-time, embedded deployment scenario motivating this study. This configuration is carried forward as a candidate for the final combined model, alongside the retained default augmentation policy (Section 4.3) and default optimizer (Section 4.4).

![](images/468a7828ac02d76d75d8eff1cc5c2b34556383e4e1edbb96f1a27299689b5704.jpg)  
(a) YOLOv11s backbone with the CoordAtt substitution

![](images/2e885ddfd9489a8871d3b363b1de118b1dec4e825181d0e22b04f8c147f63d38.jpg)  
(b) Coordinate Attention (CoordAtt) module

Figure 1. (a) YOLO11s backbone with the Coordinate Attention (CoordAtt) substitution: CoordAtt occupies the same architectural slot as the baseline’s C2PSA block, with identical input/output channel counts, so no other layer in the backbone, neck, or detection heads is modified. (b) Internal dataflow of the CoordAtt module: the input feature map is pooled separately along the height and width axes, concatenated, passed through a shared 1 × 1 convolution–batch-norm–h-swish bottleneck, split back into two branches, and used to generate a pair of sigmoid attention maps $( \boldsymbol { a } _ { h } , \boldsymbol { a } _ { w } )$ that rescale the original input via element-wise multiplication.  
![](images/368ee30519a691cb875d60cc4de004a9cf69bc07b003a46fcb77daf4631daccb.jpg)

![](images/ce06cad8ce188337c32669bfe6be1f9f18bbf43a3c50ba6e942f7144b03f7a4d.jpg)

![](images/3bd1b00a76722115db7a6a0876e81e2a96c0c693e805f87c5f8744c1b56b5cbd.jpg)

![](images/f7f11279784a626e458e9e14add246c44bcec042f8d7d238eb91bbeef2363a73.jpg)

![](images/74feb382b6692f7eeed866faa22e85fa0cb7284a62a209107e60e524e9e19916.jpg)

![](images/32fdbafdc5670b20fda8c475010e5325bc61a744e690af76d5ac8fd45bb5be4a.jpg)

![](images/5832945eb613d55d7deae60f6df04779d75a3b09eb2c575c8d26c9589f3aa8d5.jpg)

![](images/557952e9497cc6e26b9ae4378a1fda99566ebc3d3892bd653ef767f6f8d475e1.jpg)

![](images/cbb98dcdd544350b1f6cf2e501e0942905d77fd67fe6aa93739b4574647203dc.jpg)

![](images/c345941531a5bdd498ef45f471525ed69f04e8c70562487d24e1832973a6d28d.jpg)  
Figure 2. Training and validation curves for the final CA-swap model across 73 training epochs (early stopping triggered at epoch 73, best checkpoint retained from epoch 43): box, classification, and DFL loss (train and validation), and validation-split precision, recall, mAP@0.5, and mAP@0.5:0.95.

![](images/1a47fc2919439e235a16d1bccbda657845bd7219773ee6fad8eed30bd3374e88.jpg)  
Figure 3. Normalized confusion matrix for the final CA-swap model on the validation split (301 images), showing per-class prediction accuracy and the degree to which each class is confused with background.

## 4.6. Ablation Factor 5 — Inference Decision Threshold

This factor evaluates post-hoc confidence thresholding on the CA-swap model (Section 4.5) without retraining, holding the NMS IoU threshold at its default (0.7) across 0.10 to 0.70 on the held-out test split (Table 14).

Table 14. Confidence-threshold sweep on the CA-swap model’s test-set predictions (Ablation Factor 5).
<table><tr><td>Conf. threshold</td><td>Precision</td><td>Recall</td><td>F1</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td></tr><tr><td>0.10</td><td>0.8885</td><td>0.9189</td><td>0.9035</td><td>0.9108</td><td>0.5959</td></tr><tr><td>0.15</td><td>0.8888</td><td>0.9205</td><td>0.9043</td><td>0.9108</td><td>0.5959</td></tr><tr><td>0.20</td><td>0.8888</td><td>0.9199</td><td>0.9041</td><td>0.8900</td><td>0.5875</td></tr><tr><td>0.25</td><td>0.8967</td><td>0.8990</td><td>0.8979</td><td>0.8743</td><td>0.5802</td></tr><tr><td>0.30</td><td>0.9106</td><td>0.8882</td><td>0.8992</td><td>0.8666</td><td>0.5764</td></tr><tr><td>0.35</td><td>0.9107</td><td>0.8886</td><td>0.8995</td><td>0.8666</td><td>0.5764</td></tr><tr><td>0.40</td><td>0.9085</td><td>0.8778</td><td>0.8929</td><td>0.8559</td><td>0.5729</td></tr><tr><td>0.50</td><td>0.9458</td><td>0.8405</td><td>0.8901</td><td>0.8238</td><td>0.5634</td></tr><tr><td>0.60</td><td>0.9629</td><td>0.7989</td><td>0.8732</td><td>0.7860</td><td>0.5478</td></tr><tr><td>0.70</td><td>0.9465</td><td>0.6883</td><td>0.7970</td><td>0.6787</td><td>0.5022</td></tr></table>

Discussion. F1 peaks at a confidence threshold of 0.15 (P = 0.8888, R = 0.9205, F1 = 0.9043), which also ties the highest mAP@0.5 (0.9108) recorded in the sweep. As the threshold rises, precision climbs steadily (reaching 0.9629 at 0.60) while recall falls sharply (to 0.7989 at the same point and 0.6883 at 0.70), and F1 declines monotonically past 0.15 — confirming that, for this model, low-to-moderate thresholds dominate the precision–recall trade-off rather than Ultralytics’ conventional default of 0.25, which already underperforms (F1 = 0.8979) relative to 0.10–0.20 (Figure 4). Because missed stem-cavity or calyx detections directly translate into failed orientation decisions downstream in the sorting pipeline, the cost of a false negative is higher than that of a false positive, which further favors operating near the recall-leaning end of the curve rather than strictly at the F1-maximizing point.

Summary of Ablation Factor 5. A confidence threshold of 0.15 is selected as the deployment operating point for the CA-swap model: it maximizes F1, matches the sweep’s best mAP@0.5, and sits at the recall-favorable end of the precision–recall trade-off appropriate for an orientation system where missed landmark detections are more costly than occasional false positives. This closes the five-factor ablation study; the final recommended configuration combines the default augmentation policy (Section 4.3), the default (automatic) optimizer/LR schedule (Section 4.4), the CoordAtt architectural swap (Section 4.5), and this 0.15 confidence threshold (Section 4.6).

![](images/5b001080691e04e41af4935677d67bdc9bafdbee817bc01eb4c2b8b3d10bc09c.jpg)  
(a)

![](images/49a77333a1ef0010c0896272fe04daa503b248a54a4eb276b3948094efaae668.jpg)  
(b)  
Figure 4. (a) Precision–recall curve and (b) F1-confidence curve for the final CA-swap model, computed on the validation split (301 images) as a training-time diagnostic. Note: the per-point optimum shown in (b) (confidence ≈ 0.336) reflects the validation split and is not the deployment recommendation; the 0.15 confidence threshold adopted in this study was instead selected via an explicit sweep on the held-out test split (Table 14), the split used for every other reported result in this paper.

## 4.7. Final Combined Model — Testing Additivity

Sections 4.3 and 4.5 identified two variants that each improved on the baseline’s calyxclass performance in isolation: Run E (stronger HSV augmentation) produced the best calyx mAP@0.5 of any augmentation variant (0.964), while the CA-swap model (Section 4.5) was the single strongest overall configuration but showed a modest calyx F1 decline relative to the baseline (0.8969 vs. 0.9160). Since ablation effects are not guaranteed to compose additively, this was tested directly rather than assumed: a final combined model was trained using the CA-swap architecture together with Run E’s stronger HSV settings (hsv\_s = 0.9, hsv\_ $_ - \mathbf { v } = 0 . 6 )$ , with every other setting — imgsz = 640, patience = 30, default optimizer — unchanged (Table 15).

Table 15. Per-class test-set results for the final combined model (CA-swap architecture + stronger HSV augmentation, Section 4.7).
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>mAP@0.5</td><td>mAP@0.5:0.95</td><td>F1</td></tr><tr><td>Fruit body</td><td>0.984</td><td>1.000</td><td>0.994</td><td>0.994</td><td>0.9921</td></tr><tr><td>Calyx</td><td>0.957</td><td>0.853</td><td>0.933</td><td>0.483</td><td>0.9018</td></tr><tr><td>Stem cavity</td><td>0.827</td><td>0.656</td><td>0.789</td><td>0.313</td><td>0.7319</td></tr><tr><td>All classes</td><td>0.923</td><td>0.836</td><td>0.906</td><td>0.597</td><td>0.8775</td></tr></table>

Discussion. The combination underperforms both of its constituent configurations on every aggregate metric: overall F1 falls to 0.8775, below both the baseline (0.9047) and the CA-swap-alone model (0.9038), and is the lowest F1 recorded across the entire study. The calyx class does improve slightly relative to CA-swap-alone (F1 0.9018 vs. 0.8969), consistent with Run E’s isolated calyx benefit, but this gain is purchased at a severe cost to the stem cavity class: recall collapses from 0.872 (CA-swap-alone) to 0.656, pushing stem-cavity F1 to 0.7319 — below even the untuned baseline (0.7918), and far below the CA-swap model’s best-in-study result (0.8217). In other words, combining the two bestperforming isolated interventions reverses the architectural factor’s primary benefit rather than compounding it. This is consistent with the recall-suppressing pattern observed across several augmentation variants in Section 4.3 (Runs E, F, and G all traded recall for precision relative to the baseline); layering that same tendency onto the CA-swap architecture appears to compound against the smallest, most instance-sparse class rather than complementing its architectural gain.

Summary of Section 4.7. This result provides direct empirical confirmation — rather than inference from separate ablation runs — that the factors identified as individually beneficial or neutral in this study do not compose additively, and in this case interact destructively. The CA-swap architecture alone (Section 4.5), without any augmentation modification, is therefore retained as the final recommended configuration; no further combination of ablation factors is pursued.

## 4.8. Qualitative Results

To illustrate how the final recommended model (the CA-swap architecture, Section 4.5, at the 0.15 confidence threshold selected in Section 4.6) performs under the dataset’s natural appearance variability, Figure 5 shows three representative detections on held-out test images spanning the color range described in Section 3.1: (a) a pale yellow-green apple with a correctly detected stem cavity (confidence 0.76); (b) a pale pink-blush apple with a correctly detected calyx (confidence 0.80); and (c) a deep red apple with a correctly detected calyx at comparatively lower confidence (0.67), illustrating how darker, more saturated skin tones reduce the model’s detection confidence for this class without yet causing a missed detection. All three examples show a correctly localized fruit-body box alongside the landmark detection, consistent with the near-perfect fruit-body performance reported throughout this study (Table 13).

![](images/2e493f95387f630f64e013e331e18b65824fdb46bb2f4511a391ee278856fbaf.jpg)  
(a)

![](images/c2795f627d9697af3d20331c8a9822b07d3faf0c070b701b9a6a8f8c30966027.jpg)  
(b)

![](images/82dce6e3d1f75732b97a5454a89cf0b1af125be9e3ba0cb8923f08a95e6c25ab.jpg)  
(c)  
Figure 5. Representative detections of the final CA-swap model (at the 0.15 confidence threshold) on held-out test images across varying apple cultivars and skin tones: (a) pale yellow-green apple with detected stem cavity (confidence 0.76); (b) pale pink-blush apple with detected calyx (confidence 0.80); and (c) deep red apple with detected calyx (confidence 0.67).

## 5. Discussion

Across the five factors examined, a clear pattern emerges: training- and data-level interventions (input resolution, augmentation policy, optimizer/LR schedule) consistently failed to improve on the untuned baseline, while the one architectural intervention tested — the CoordAtt swap — was the only factor to match or exceed it, and the only factor to meaningfully improve the stem cavity class. This asymmetry is informative beyond the specific numbers. In a fixed-camera, controlled-illumination acquisition setup, the geometric and photometric variability that augmentation is designed to simulate (viewpoint, lighting, occlusion) is largely already absent from the data distribution the model will see at inference time; synthetic perturbations beyond the Ultralytics default therefore add noise relative to the deployment distribution rather than useful invariance. The optimizer result points in the same direction: with a moderately sized, relatively clean dataset, Ultralytics’ automatic optimizer selection is already well matched to the problem, leaving little room for a manually tuned alternative to improve on it. Architecture, by contrast, changes what the model is capable of representing regardless of the data distribution, which is consistent with coordinate attention’s specific benefit to the stem cavity class — the smallest, most texturally variable, and consistently weakest-performing landmark across every other factor tested.

This finding is consistent with, and extends, the two most closely related studies in Section 2.3. Yang et al.’s CA-YOLOv5 [11] showed that coordinate attention improves whole-apple detection in uncontrolled natural environments; the present results show the same mechanism also improves detection of a specific sub-fruit landmark under controlled imaging conditions, suggesting the benefit of position-sensitive channel attention generalizes across both environmental variability and landmark granularity. Baek et al.’s AS-YOLO [16] similarly found that an attention mechanism, rather than training-level tuning, was necessary to improve stem-region segmentation — but reported only a single already-optimized architecture. The controlled, one-factor-at-a-time design used here isolates that same conclusion from first principles: for this class of fine-grained landmark detection problem, architectural modification appears to be a more productive lever than augmentation or optimizer tuning, at least within the ranges tested.

From an industrial deployment perspective, the mechanistic failure of additivity in Section 4.7 offers critical guidance for engineering high-speed grading lines. Morphological depressions such as the stem cavity rely on delicate, low-contrast specular shading gradients around the basin edge. While global fruit detection remains robust across extreme color perturbations, aggressive synthetic jittering (Run E) disrupts the spatial-channel codependencies that Coordinate Attention explicitly encodes via 1D direction-aware pooling, destroying the subtle shadow contours required to localize the cavity. Furthermore, in mechanical orientation systems utilizing bi-axial roller bobbins or pneumatic flippers, undetected landmarks represent an irreversible sorting failure, forcing fruit recirculation and degrading conveyor throughput. Operating the lighter CA-swap architecture at a calibrated 0.15 threshold maximizes landmark recall while eliminating the quadratic memory overhead of native C2PSA attention blocks, ensuring deterministic, low-latency execution on embedded edge hardware (e.g., NVIDIA Jetson platforms) without sacrificing orientation reliability.

Several limitations constrain how far these findings should be generalized. All data were collected under a single controlled, fixed-camera, fixed-illumination setup; whether the same factor rankings hold under variable camera distance, lighting, or an orchard/conveyor environment is untested. The dataset, while fully self-annotated and class-balanced within the constraints of natural fruit morphology, remains moderate in scale (1506 images), and the stem cavity class in particular had the fewest training instances of the three, which may compound its sensitivity to the factors tested. Only one backbone family (YOLO11s) and one attention mechanism (coordinate attention, tested as a single same-slot substitution) were evaluated; other lightweight attention or backbone modifications were not explored and may interact differently with this dataset. Inference-speed measurements also showed session-to-session variance (Section 4.2) large enough that the latency figures reported here should be read as indicative rather than tightly reproducible; a dedicated, single-session benchmarking pass across the baseline and final configuration is identified as necessary follow-up work before any latency claim is used for deployment sizing. Finally, the augmentation and optimizer ablations, while systematic, tested a representative rather than exhaustive set of variants within each factor, and only one cross-factor combination (CA-swap with stronger HSV, Section 4.7) was tested directly; it remains possible that a different combination, or a configuration outside the tested range entirely (for example, a more targeted, class-weighted training strategy for the calyx and stem cavity classes, as noted in Section 4.3), could outperform the baseline where the tested variants and this one combination did not.

![](images/4a92ebdc774b0e7bef4f3fd1c86db9f7017bf241e95db9972375e49486d6f84b.jpg)  
Figure 6. Per-class F1 score comparison across the baseline, the final CA-swap architecture (Section 4.5), and the combined CA-swap-plus-stronger-HSV model (Section 4.7). The architecture’s stem-cavity gain and the combination’s stem-cavity loss are directly visible as the one class whose bar moves in opposite directions between the second and third configurations, while fruit body remains essentially unchanged across all three.

## 6. Conclusions and Future Works

This study presented a custom, fully self-collected and self-annotated dataset of 1506 apple images annotated for three orientation-relevant landmarks — fruit body, stem cavity, and calyx — and used it to conduct a systematic, one-factor-at-a-time ablation study on a YOLO11s baseline, addressing a gap in the apple-detection literature where prior work typically reports a single, already-optimized configuration rather than isolating the contribution of individual design choices. Across five factors — input resolution, augmentation policy, optimizer/LR schedule, lightweight architectural modules, and inference decision threshold — the results show that, for this fine-grained, controlledenvironment landmark-detection task, architectural modification was the only lever that improved on the untuned baseline, while increased resolution, every tested augmentation variant, and manual optimizer tuning each underperformed it on the primary metrics.

The recommended final configuration combines the default (baseline) augmentation policy, the default automatic optimizer/LR schedule, a coordinate-attention substitution for YOLO11s’s native C2PSA block, and a 0.15 confidence threshold at inference. This configuration matches the baseline’s overall F1 (0.9038 vs. 0.9047) while achieving the study’s highest mAP@0.5 (0.935), its best stem-cavity F1 (0.8217), and a roughly 10% reduction in parameter count and GFLOPs — a combination of comparable accuracy and reduced computational cost directly relevant to real-time, potentially embedded deployment in industrial apple sorting and orientation systems. Future work should extend this analysis beyond the single controlled imaging environment studied here, benchmark inference latency in a single consistent session on representative embedded hardware, and explore class-targeted training strategies and alternative attention or backbone modules for the stem cavity and calyx classes specifically.

In the future,we would like to implement our deep learning model on mobile platforms. We believe that our CAppleNet will work on mobile platforms at real-time frame rates.

Additionally, there is potential for upgrading and optimizing our model by incorporating state of the art deep learning methods and techniques [17–24].

To improve accuracy of CAppleNet, we are also interested in implementing realistic Bidirectional Reflectance Distribution Function (BRDF) [25–42], Bidirectional Scattering Distribution Function (BSDF) [43–48], Bidirectional Surface Scattering Reflectance Distribution Function (BSSRDF) [49–55] and multi-layered material models [43,46,56] into our framework.

Author Contributions: Conceptualization, A.G. and M.K.; methodology, A.G. and M.K.; software, A.G.; validation, A.G., R.S.H. and G.H.; formal analysis, A.G. and M.K.; investigation, A.G., R.S.H. and G.H.; resources, R.S.H. and G.H.; data curation, A.G.; writing—original draft preparation, A.G.; writing—review and editing, A.G. and M.K.; visualization, A.G.; supervision, M.K.; project administration, M.K., R.S.H. and G.H. All authors have read and agreed to the published version of the manuscript.

Funding: This research was funded by the Scientific and Technical Research Council of Turkey grant number 7250234.

Data Availability Statement: The custom dataset and implementation source codes used in this study are proprietary to Tarend Agricultural and Industrial Machinery.

Acknowledgments: The authors express gratitude to Tarend Agricultural and Industrial Machinery for providing the industrial facility, prototype machinery, and testing hardware used during this project.

Conflicts of Interest: Authors Robert Selim Hill, Gökhan Hasano˘glu and Alican Güne¸s are employed by Tarend. The remaining author declares no conflict of interest.

## References

1. Bargoti, S.; Underwood, J.P. Image Segmentation for Fruit Detection and Yield Estimation in Apple Orchards. Journal ofField Robotics 2017, 34, 1039–1060. https://doi.org/10.1002/rob.21699.

2. Lin, G.; Tang, Y.; Zou, X.; Li, J.; Xiong, J. In-field citrus detection and localisation based on RGB-D image analysis. Biosystems Engineering 2019, 186, 34–44. https://doi.org/10.1016/j.biosystemseng.2019.06.019.

Tian, Y.; Yang, G.; Wang, Z.; Wang, H.; Li, E.; Liang, Z. Apple detection during different growth stages in orchards using the improved YOLO-V3 model. Computers and Electronics in Agriculture 2019, 157, 417–426. https://doi.org/10.1016/j.compag.2019.0 1.012.

4. Kang, H.; Chen, C. Fruit Detection and Segmentation for Apple Harvesting Using Visual Sensor in Orchards. Sensors 2019, 19, 4599. https://doi.org/10.3390/s19204599.

5. Bochkovskiy, A.; Wang, C.Y.; Liao, H.Y.M. YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv preprint arXiv:2004.10934 2020. https://doi.org/10.48550/arXiv.2004.10934.

6. Liu, G.; Nouaze, J.C.; Touko Mbouembe, P.L.; Kim, J.H. YOLO-Tomato: A Robust Algorithm for Tomato Detection Based on YOLOv3. Sensors 2020, 20, 2145. https://doi.org/10.3390/s20072145.

7. Yan, B.; Fan, P.; Lei, X.; Liu, Z.; Yang, F. A Real-Time Apple Targets Detection Method for Picking Robot Based on Improved YOLOv5. Remote Sensing 2021, 13, 1619. https://doi.org/10.3390/rs13091619.

8. Wang, D.; He, D. Channel Pruned YOLO V5s-Based Deep Learning Approach for Rapid and Accurate Apple Fruitlet Detection before Fruit Thinning. Biosystems Engineering 2021, 210, 271–281. https://doi.org/10.1016/j.biosystemseng.2021.08.015.

9. Zhang, C.; Kang, F.; Wang, Y. An Improved Apple Object Detection Method Based on Lightweight YOLOv4 in Complex Backgrounds. Remote Sensing 2022, 14, 4150. https://doi.org/10.3390/rs14174150.

10. Ji, W.; Pan, Y.; Xu, B.; Wang, J. A Real-Time Apple Targets Detection Method for Picking Robot Based on ShufflenetV2-YOLOX. Agriculture 2022, 12, 856. https://doi.org/10.3390/agriculture12060856.

11. Yang, R.; He, Y.; Hu, Z.; Gao, R.; Yang, H. CA-YOLOv5: A YOLO Model for Apple Detection in the Natural Environment. Systems Science & Control Engineering 2024, 12, 2278905. https://doi.org/10.1080/21642583.2023.2278905.

12. Olguín-Rojas, J.C.; Vasquez, J.I.; López-Canteñs, G.d.J.; Herrera-Lozada, J.C.; Mota-Delfin, C. A Lightweight YOLO-Based Architecture for Apple Detection on Embedded Systems. Agriculture 2025, 15, 838. https://doi.org/10.3390/agriculture15080838.

13. Du, X.; Zhang, X.; Li, T.; Chen, X.; Yu, X.; Wang, H. YOLO-WAS: A Lightweight Apple Target Detection Method Based on Improved YOLO11. Agriculture 2025, 15, 1521. https://doi.org/10.3390/agriculture15141521.

14. Alif, M.A.R.; Hussain, M. YOLOv1 to YOLOv10: A Comprehensive Review of YOLO Variants and Their Application in the Agricultural Domain. arXiv preprint arXiv:2406.10139 2024. https://doi.org/10.48550/arXiv.2406.10139.

15. Lv, M.; Xu, Y.X.; Miao, Y.H.; Su, W.H. A Comprehensive Review of Deep Learning in Computer Vision for Monitoring Apple Tree Growth and Fruit Production. Sensors 2025, 25, 2433. https://doi.org/10.3390/s25082433.

16. Baek, N.R.; Lee, Y.; Noh, D.h.; Lee, H.M.; Cho, S.W. AS-YOLO: Enhanced YOLO Using Ghost Bottleneck and Global Attention Mechanism for Apple Stem Segmentation. Sensors 2025, 25, 1422. https://doi.org/10.3390/s25051422.

17. Gök, G.; Küçük, S.; Kurt, M.; Tarı, E. A U-Net Based Segmentation and Classification Approach over Orthophoto Maps of Archaeological Sites. In Proceedings of the Proceedings of the IEEE 31st Signal Processing and Communications Applications Conference, Istanbul, Turkey, July 2023; SIU ’23, pp. 1–4.

18. Azadvatan, Y.; Kurt, M. MelNet: A Real-Time Deep Learning Algorithm for Object Detection. arXiv preprint arXiv:2401.17972 2024, p. arXiv:2401.17972, [arXiv:cs.CV/2401.17972]. https://doi.org/10.48550/arXiv.2401.17972.

19. Jabbarlı, G.; Kurt, M. LightFFDNets: Lightweight Convolutional Neural Networks for Rapid Facial Forgery Detection. arXiv preprint arXiv:2411.11826 2024, p. arXiv:2411.11826, [arXiv:cs.CV/2411.11826]. https://doi.org/10.48550/arXiv.2411.11826.

20. Akdo˘gan, A.; Kurt, M. Feature Engineering for Robust Post-OCR Document Information Extraction. In Proceedings of the Proceedings of the 14th International Artemis Scientific Research Congress; Karakurt, P.; Giurgiu, G., Eds., Bucharest, Romania, 2025; pp. 346–357.

21. Türkün, A.E.; Kalender, M.E.; Kurt, M.; Do˘gan, S. Clinical Equivalence of a CNN-Based Automated Soft Tissue Landmark Detection System on 2D Facial Images. Diagnostics 2026, 16, 1464:1–1464:19. https://doi.org/10.3390/diagnostics16101464.

22. Akdo˘gan, A.; Kurt, M. ExTTNet: A Deep Learning Algorithm for Extracting Table Texts from Invoice Images. Mathematics 2026, 14, 2258:1–2258:17. https://doi.org/10.3390/math14132258.

23. Azadvatan, Y.; Kurt, M. SiStNet: A Single-Stage Convolutional Neural Network for Vehicle Detection. Applied Sciences 2026, 16, 6476:1–6476:22. https://doi.org/10.3390/app16136476.

24. Çangırılı, E.; Kurt, M. TRACK: Telemetry-Based Racing Analysis and Coaching Kit in Sim Racing Games. arXiv preprint arXiv:2610.10061 2026, p. arXiv:2610.10061, [arXiv:cs.GR/2610.10061]. https://doi.org/10.48550/arXiv.2610.10061.

25. Öztürk, A.; Bilgili, A.; Kurt, M. Polynomial Approximation of Blinn-Phong Model. In Proceedings of the Proceedings of the 4th Theory and Practice of Computer Graphics; Lever, L.M.; McDerby, M., Eds., Middlesbrough, United Kingdom, 2006; TPCG ’06, pp. 55–61. https://doi.org/10.2312/LocalChapterEvents/TPCG/TPCG06/055-061.

26. Kurt, M. A New Illumination Model in Computer Graphics. Master’s thesis, International Computer Institute, Ege University, Izmir, Turkey, 2007. 140 pages.

27. Ozturk, A.; Kurt, M.; Bilgili, A.; Gungor, C. Linear approximation of Bidirectional Reflectance Distribution Functions. Computers & Graphics 2008, 32, 149–158.

28. Kurt, M.; Cinsdikici, M.G. Representing BRDFs using SOMs and MANs. SIGGRAPH Computer Graphics 2008, 42, 1–18. https://doi.org/10.1145/1408626.1408630.

29. Kurt, M.; Edwards, D. A survey of BRDF models for computer graphics. SIGGRAPH Computer Graphics 2009, 43, 1–7. https://doi.org/10.1145/1629216.1629222.

30. Kurt, M.; Szirmay-Kalos, L.; Kˇrivánek, J. An Anisotropic BRDF Model for Fitting and Monte Carlo Rendering. SIGGRAPH Computer Graphics 2010, 44, 1–15. https://doi.org/10.1145/1722991.1722996.

31. Öztürk, A.; Kurt, M.; Bilgili, A. Modeling BRDF by a Probability Distribution. In Proceedings of the Proceedings of the 20th International Conference on Computer Graphics and Vision, St. Petersburg, Russia, 2010; pp. 57–63.

32. Szécsi, L.; Szirmay-Kalos, L.; Kurt, M.; Csébfalvi, B. Adaptive sampling for environment mapping. In Proceedings of the Proceedings of the 26th Spring Conference on Computer Graphics, New York, NY, USA, 2010; SCCG ’10, pp. 69–76. https://doi.org/10.1145/1925059.1925073.

33. Öztürk, A.; Kurt, M.; Bilgili, A. A Copula-Based BRDF Model. Computer Graphics Forum 2010, 29, 1795–1806.

34. Bilgili, A.; Öztürk, A.; Kurt, M. A General BRDF Representation Based on Tensor Decomposition. Computer Graphics Forum 2011, 30, 2427–2439.

35. Bilgili, A.; Öztürk, A.; Kurt, M. Representing BRDF by wavelet transformation of pair-copula constructions. In Proceedings of the Proceedings of the 28th Spring Conference on Computer Graphics, New York, NY, USA, 2012; SCCG ’12, pp. 63–69. https://doi.org/10.1145/2448531.2448539.

36. Ergun, S.; Kurt, M.; Öztürk, A. Real-time kd-tree based importance sampling of environment maps. In Proceedings of the Proceedings of the 28th Spring Conference on Computer Graphics, New York, NY, USA, 2012; SCCG ’12, pp. 77–84. https://doi.org/10.1145/2448531.2448541.

37. Töral, O.A.; Ergun, S.; Kurt, M.; Öztürk, A. Mobile GPU-Based Importance Sampling. In Proceedings of the Proceedings of the IEEE 22nd Signal Processing and Communications Applications Conference, Trabzon, Turkey, April 2014; SIU ’14, pp. 510–513.

38. Tongbuasirilai, T.; Unger, J.; Kurt, M. Efficient BRDF Sampling Using Projected Deviation Vector Parameterization. In Proceedings of the Proceedings of the IEEE International Conference on Computer Vision Workshops, Venice, Italy, 2017; ICCVW ’17, pp. 153–158. https://doi.org/10.1109/ICCVW.2017.26.

39. Kurt, M. Real-Time Shading with Phong BRDF Model. Journal of Science and Engineering 2019, 21, 859–867.

40. Tongbuasirilai, T.; Unger, J.; Kronander, J.; Kurt, M. Compact and intuitive data-driven BRDF models. The Visual Computer 2020, 36, 855–872. https://doi.org/10.1007/s00371-019-01664-z.

41. Akleman, E.; Kurt, M.; Akleman, D.; Bruins, G.; Deng, S.; Subramanian, M. Hyper-Realist Rendering: A Theoretical Framework. arXiv preprint arXiv:2401.12853 2024, p. arXiv:2401.12853, [arXiv:cs.GR/2401.12853]. https://doi.org/10.48550/arXiv.2401.12853.

42. Kurt, M. Mul2MAR: A Multi-Marker Mobile Augmented Reality Application for Improved Visual Perception. arXiv preprint arXiv:2502.05953 2025, p. arXiv:2502.05953, [arXiv:cs.GR/2502.05953]. https://doi.org/10.48550/arXiv.2502.05953.

43. Ward, G.; Kurt, M.; Bonneel, N. A Practical Framework for Sharing and Rendering Real-World Bidirectional Scattering Distribution Functions. Technical Report LBNL-5954E, Lawrence Berkeley National Laboratory, 2012.

44. Ward, G.; Kurt, M.; Bonneel, N. Reducing Anisotropic BSDF Measurement to Common Practice. In Proceedings of the Proceedings of the 2nd Eurographics Workshop on Material Appearance Modeling: Issues and Acquisition; Klein, R.; Rushmeier, H., Eds., Lyon, France, 2014; MAM ’14, pp. 5–8. https://doi.org/10.2312/mam.20141292.

45. Kurt, M. Grand Challenges in BSDF Measurement and Modeling. The Workshop on Light Redirection and Scatter: Measurement, Modeling, Simulation, 2014. (Invited Talk).

46. Kurt, M.; Ward, G.; Bonneel, N. A Data-Driven BSDF Framework. In Proceedings of the Proceedings of the ACM SIGGRAPH 2016, Posters, New York, NY, USA, 2016; SIGGRAPH ’16, pp. 31:1–31:2. https://doi.org/10.1145/2945078.2945109.

47. Kurt, M. Experimental Analysis of BSDF Models. In Proceedings of the Proceedings of the 5th Eurographics Workshop on Material Appearance Modeling: Issues and Acquisition; Klein, R.; Rushmeier, H., Eds., Helsinki, Finland, 2017; MAM ’17, pp. 35–39. https://doi.org/10.2312/mam.20171330.

48. Kurt, M. A Survey of BSDF Measurements and Representations. Journal of Science and Engineering 2018, 20, 87–102.

49. Kurt, M.; Öztürk, A.; Peers, P. A Compact Tucker-Based Factorization Model for Heterogeneous Subsurface Scattering. In Proceedings of the Proceedings of the 11th Theory and Practice of Computer Graphics; Czanner, S.; Tang, W., Eds., Bath, United Kingdom, 2013; TPCG ’13, pp. 85–92. https://doi.org/10.2312/LocalChapterEvents.TPCG.TPCG13.085-092.

50. Kurt, M.; Öztürk, A. A Heterogeneous Subsurface Scattering Representation Based on Compact and Efficient Matrix Factorization. In Proceedings of the Proceedings of the 24th Eurographics Symposium on Rendering, Posters, Zaragoza, Spain, June 2013; EGSR ’13.

51. Kurt, M. An Efficient Model for Subsurface Scattering in Translucent Materials. PhD thesis, International Computer Institute, Ege University, Izmir, Turkey, 2014. 122 pages.

52. Önel, S.; Kurt, M.; Öztürk, A. An Efficient Plugin for Representing Heterogeneous Translucent Materials. In Contemporary Topics in Computer Graphics and Games: Selected Papersfrom the Eurasia Graphics Conference Series; <sup>˙</sup>I¸sler, V.; Gürçay, H.; Süher, H.K.; Çatak, G., Eds.; Peter Lang GmbH, Internationaler Verlag der Wissenschaften, 2019; chapter 18, pp. 309–321. (Book Chapter).

53. Kurt, M. A Genetic Algorithm Based Heterogeneous Subsurface Scattering Representation. In Proceedings of the Proceedings of the 8th Eurographics Workshop on Material Appearance Modeling: Issues and Acquisition; Klein, R.; Rushmeier, H., Eds., London, UK, 2020; MAM ’20, pp. 13–16. https://doi.org/10.2312/mam.20201140.

54. Kurt, M. GenSSS: a genetic algorithm for measured subsurface scattering representation. The Visual Computer 2021, 37, 307–323. https://doi.org/10.1007/s00371-020-01800-0.

55. Yıldırım, B.; Kurt, M. GenPluSSS: A Genetic Algorithm-Based Plugin for Measured Subsurface Scattering Representation. Applied Sciences 2026, 16, 6249:1–6249:19. https://doi.org/10.3390/app16126249.

56. Mir, S.; Yıldırım, B.; Kurt, M. An Analysis of Goniochromatic and Sparkle Effects on Multi-Layered Materials. Journal of Science and Engineering 2022, 24, 737–746.

Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.