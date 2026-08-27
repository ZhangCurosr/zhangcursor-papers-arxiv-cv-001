# CropCop: An Auditable 120-Class Plant-Health Model from Benchmark Reconstruction to a Quantised Runtime Artifact

Rana Muhammad Ahmed<sup>1</sup> Sabahat Abbas<sup>1</sup>

<sup>1</sup>Department of Computer Science, Bahria University Islamabad, Islamabad, Pakistan

Preprint · August 2026

## Abstract

A plant-health score can appear precise while resting on duplicated image families, a long-tailed label space, or a runtime file that was never evaluated. We present CropCop, a closed-set recognition system spanning 120 operational plant-health classes and an evidence chain from corpus reconstruction to direct execution of the final quantised artifact. Starting from 117,546 audited images, we rejected the inherited partition after confirming 3,233 duplicate relationships across split boundaries and froze a 109,107-image benchmark with zero crossings among the audited trusted leakage groups and a 151.7× largest-to-smallest class ratio. A fully fine-tuned DINOv3 ConvNeXt-Tiny reference achieved 98.51% accuracy and 96.87% macro-F1 on the locked internal test. A compact MobileNetV4 Conv-Medium derivative achieved 98.46% accuracy and 96.27% macro-F1 without being presented as evidence for a new distillation method. Validation-only post-training quantisation selected dynamic activations with per-channel weights, and the final 22.60 MiB ExecuTorch/XNNPACK PTE achieved 98.46% accuracy and 96.23% macro-F1 when executed directly. Only six of 16,363 top-1 decisions changed between the converted INT8 graph and the PTE, while paired analysis showed a modest class-balanced loss; an exploratory post hoc fruit-label slice localized a larger recall decline than aggregate accuracy revealed. CropCop establishes strong leakage-controlled internal recognition and software-runtime fidelity; it does not establish performance on unseen farms, camera pipelines, or physical Android hardware.

109,107 frozen images

120 operational classes

22.60 MiB executed PTE

6 / 16,363 changed top-1 decisions

Keywords plant-health recognition; dataset leakage; duplicate forensics; long-tailed classification; DINOv3; MobileNetV4; post-training quantisation; ExecuTorch; XNNPACK; runtime fidelity

## 1 Introduction

The impressive number in a plant-disease paper is usually the accuracy. The consequential numbers are often elsewhere: how many test images have relatives in training, how many classes have only a handful of examples, and how many predictions change after export. Plant-health recognition is attractive because cameras are inexpensive and already embedded in agricultural workflows, but a defensible result requires more than fitting a classifier. The benchmark must have a stable identity, the evaluation must resist selection leakage, and the file intended for inference must be shown to behave like the model described in the paper.

Plant-image collections make these requirements dificult. Public corpora are assembled from controlled repositories, internet searches, extension websites, derivative datasets, and field photographs. The resulting images vary in background, framing, compression, resolution, plant organ, cultivar, disease severity, and diagnostic specificity. Byte-identical copies are only the simplest source of contamination. Crops, recompressions, colour transformations, and burst-like views can cross nominal train and test boundaries, while label-correlated borders, dimensions, and acquisition styles can remain predictive after duplicate removal. PlantVillage demonstrated that deep networks could classify controlled imagery with striking internal accuracy, yet its external-image experiment also revealed how sharply performance can change across acquisition domains [1, 2]. PlantDoc, PlantWild, PlantWild\_v2, PlantSeg, and Deep-Plant-Disease have since broadened the field toward natural imagery and larger taxonomies [3–7].

Compact inference creates a second identity problem. A fine-tuned checkpoint, a selected float student, a quantised graph, a backend-lowered program, and a serialised runtime file are diferent computational objects. A changed class map, preprocessing contract, unsupported operator, backend rewrite, or quantisation decision can alter predictions without changing the project name. Reporting the checkpoint score beside an unevaluated deployment claim conceals where the result was preserved or lost. CropCop therefore treats model conversion as a sequence of paired evaluations. Every state is tied to the same class order and locked rows, and the final runtime claim applies to one exact PTE identified by its bytes and SHA-256 digest.

The project began with 117,546 audited images and 121 candidate labels. The inherited folders were abandoned after 3,233 confirmed duplicate relationships crossed historical train, validation, and test boundaries. Global reconstruction produced CropCop Final v1: 109,107 images across 120 operational classes, divided into 76,376 training, 16,368 validation, and 16,363 test rows, with zero crossings among the audited trusted leakage groups. The dataset remains dificult. Its largest class is 151.7 times the size of its smallest; its labels mix specific diseases with broader condition categories; and its images range from isolated leaves and fruits to whole plants and field scenes.

CropCop asks three research questions. RQ1 — benchmark validity: can a heterogeneous plant-image aggregate be reconstructed so that confirmed duplicate families do not cross the final partitions? RQ2 — model retention: how much class-balanced performance remains when a strong DINOv3 reference is carried into a compact MobileNetV4 lineage? RQ3 — runtime fidelity: does the selected quantised graph survive serialisation into an ExecuTorch/XNNPACK PTE without materially changing its decisions or ground-truth performance? The questions are deliberately narrower than “does the model work in the field?” That claim requires a source-independent cohort and is outside the completed evidence. Figure 1 summarises the connected evidence chain.

![](images/f63659aec55e6cc97421490a45b2ff92e11b62ebf86ba469d2e2450ac51b711a.jpg)  
Figure 1: CropCop audit-to-runtime evidence chain.  
Note. Each transition is tied to a persisted manifest, prediction record, configuration fingerprint, or artifact identifier.

The completed system answers the three questions with a connected result. The DINOv3 ConvNeXt-Tiny reference reached 98.51% accuracy and 96.87% macro-F1 on the locked internal test. The MobileNetV4 float state retained 98.46% accuracy and 96.27% macro-F1. The final PTE reached 98.46% accuracy and 96.23% macro-F1; only six top-1 decisions difered from the converted INT8 graph. The contribution is not a new backbone, loss, or quantiser. It is an auditable model study that connects benchmark repair, class-balanced evaluation, compact transfer, validation-only PTQ selection, and direct runtime execution while preserving the boundary between internal evidence and external validity.

## 2 Related Work

## 2.1 Plant-health datasets and domain realism

Early plant-recognition work benefited from controlled image collections whose clean backgrounds and centred leaves made large-scale supervised learning practical [1, 2]. Those datasets established the feasibility of image-based disease recognition, but they also made acquisition conditions part of the task. PlantDoc introduced field and internet imagery [3]. PlantWild and its refined v2 release emphasised natural variation, fine-grained ambiguity, and broader label spaces [4, 5]. PlantSeg extended in-the-wild work to localisation and segmentation [6], while Deep-Plant-Disease assembled a larger multi-crop identification benchmark [7]. These resources show why raw accuracy across diferent datasets is not a meaningful leaderboard: ontology, source composition, image granularity, duplicate policy, and split construction all change the question being measured.

## 2.2 Benchmark contamination, shortcut learning, and reporting

Near-duplicate contamination can alter benchmark estimates even in canonical vision datasets. The ciFAIR analysis showed that CIFAR train–test near-duplicates changed reported performance and motivated explicitly deduplicated variants [8]. Label-error studies have likewise shown that test-set defects can destabilise comparisons [9], while representation-based audit methods treat duplicates, of-topic images, and suspect labels as distinct data-quality problems [10]. A leakage-free partition still does not guarantee causal learning. Models can exploit acquisition signatures that correlate with labels, as shortcut-learning studies in medical imaging have demonstrated [11]. Datasheets and model cards provide complementary reporting frameworks for documenting composition, intended use, limitations, and model identity [12, 13]. CropCop applies these ideas to one agricultural corpus and links the audit directly to the downstream runtime lineage.

## 2.3 Foundation transfer, long-tailed recognition, and compact students

Self-supervised visual representations have raised the transfer baseline for tasks with limited or heterogeneous supervision. DINOv2 demonstrated broad transfer from self-supervised features [14], and DINOv3 extended that family to multiple architectures and resource points [15]. CropCop uses DINOv3 twice: a frozen ViT-S/16 representation diagnoses the dataset before task adaptation, and a ConvNeXt-Tiny model [16] provides the fully fine-tuned reference. The dataset’s 151.7× class-size ratio also places it in the long-tail regime, where class-balanced losses and logit adjustment have been proposed to reduce head-class dominance [17, 18]. CropCop does not claim a new imbalance method. Macro-F1, balanced accuracy, disjoint support bins, and per-class tables are used to prevent aggregate accuracy from hiding minority behaviour.

Knowledge distillation transfers information from a teacher into a smaller model [19, 20], and recent plant-recognition work has studied distillation as an explicit eficiency tool [21]. MobileNetV4 was designed for hardware-aware performance across diverse mobile backends [22], while recent plant benchmarks have compared lightweight architectures across broad taxonomies [23]. Quantisation methods reduce numerical precision to lower storage and execution costs [24]. Calibration distinguishes predictive correctness from confidence quality [25]. ExecuTorch then introduces further states—export, backend lowering, quantisation, serialisation, and runtime execution—with XNNPACK-specific quantisation support [26, 27]. CropCop’s position across these lines of work is summarised in Table 1.

Table 1: Positioning of CropCop relative to adjacent literature.
<table><tr><td>Literature line</td><td>Primary object</td><td>What it establishes</td><td>Gap relative to CropCop&#x27;s question</td></tr><tr><td>Controlled and in-the-wild plant benchmarks [1-7]</td><td>Dataset and recognition task</td><td>Scale, taxonomy, acquisition diffi- culty, or localisation</td><td>The benchmark is normally the end- point; compact runtime identity is not the central object.</td></tr><tr><td>Benchmark audit and reporting [8-13]</td><td>Contamination, labels, short- cuts, documentation</td><td>Why data identity, provenance, and These works are domain-general and do limitations matter</td><td>not trace one agricultural model through quantised execution.</td></tr><tr><td>Foundation transfer and long-tail learning [14-18]</td><td>Representations and class imbalance</td><td>Transfer baselines and class- balanced objectives</td><td>They do not by themselves establish benchmark integrity or runtime fidelity.</td></tr><tr><td>Distillation and efficient vision [19-24]</td><td>Compact-model training and quantisation</td><td>How accuracy can be transferred or compressed</td><td>Dataset forensics and exact serialized- artifact evaluation are usually separate concerns.</td></tr><tr><td>Calibration and runtime tooling [25-27]</td><td>Confidence and executable backends</td><td>How predictions can be calibrated, exported, lowered, and quantised</td><td>They supply methods and infrastructure rather than an end-to-end application evidence chain.</td></tr><tr><td>CropCop</td><td>One audited plant-health model lineage</td><td>Benchmark reconstruction, locked evaluation, compact transfer, PTQ selection, direct PTE execution, and not a claim beyond the evaluated distri- artifact identity</td><td>The contribution is the connected evi- dence chain, not a new backbone and bution.</td></tr></table>

The table is an evidence-scope comparison, not a claim that earlier work should have implemented every stage. CropCop’s contribution lies in making the transitions jointly observable for one model. The cited literature supplies the data, learning, calibration, and runtime foundations; the present study tests whether their outputs can remain attached to one identifiable result.

## 3 Dataset and Benchmark Construction

## 3.1 Source pool and operational ontology

The audited source pool contained 117,546 images arranged under 121 candidate labels. The labels were operational categories rather than a uniform botanical taxonomy: some named specific diseases, others described healthy states, broad fruit conditions, nutrient deficiency, or a generic plant\_healthy category. Images included leaves, fruits, whole plants, close-ups, and field scenes. This heterogeneity is part of the completed task and one reason the paper uses the phrase plant-health recognition rather than claiming 120 distinct diseases.

The frozen class map contains 120 operational labels spanning 35 named crop or plant groups; nutrient\_deficiency and plant\_healthy do not encode a crop. Under the literal label structure, 72 labels name a specific disease or named condition, 9 use a broad condition category, and 39 denote a healthy state. The broad group comprises generic fruit- or leaf-disease labels, bean\_fungal\_disease, corn\_fungal\_leaf, and nutrient\_deficiency. This breakdown substantiates the ontology boundary: the task is a closed-set plant-health vocabulary rather than a uniform set of 120 disease entities. The full class inventory and split supports are reported in Appendix E.

Table 2: Operational class-map composition.

<table><tr><td>Dimension Category</td><td>Count</td></tr><tr><td>Crop scope</td><td>Named crop or plant groups 35</td></tr><tr><td>Crop scope</td><td>Crop-unspecified labels 2</td></tr><tr><td>Label type</td><td>Specific named disease† 72</td></tr><tr><td>Label type</td><td>Broad condition category 9</td></tr><tr><td>Label type</td><td>Generic healthy state 39</td></tr><tr><td>Label type</td><td>Total operational classes 120</td></tr></table>

Note. Crop/species and label type are derived from the literal operational labels. The four pest-labelled classes marked in Appendix E are retained in the requested specific named-disease category because the three-way scheme has no separate pest category.

The historical assignment placed 57,426 images in training, 12,272 in validation, and 47,848 in test. Those folders were retained as provenance, not accepted as a scientific split. The audit treated image identity as a graph problem. Nodes represented files; trusted edges represented direct evidence that two files belonged to the same image family. Edge sources included exact identity, strong hash evidence, and corrected feature/geometric verification. Cross-label edges were handled separately because a duplicate family carrying diferent labels creates an annotation conflict rather than an ordinary redundancy.

## 3.2 Why the inherited partition was rejected

The audit found 8,672 trusted duplicate relationships. Of these, 3,233 crossed historical split boundaries: 2,876 exact and 357 near-duplicate relationships. This was not a marginal defect. A model evaluated on the inherited folders could encounter a transformed or byte-identical member of a training image family in its test set. The historical split was therefore discarded before the final model lineage was selected. The intervention changes the semantics of the test: it no longer rewards known duplicate-family overlap, although it still represents the reconstructed aggregate rather than an independent acquisition domain.

## 3.3 Corrected near-duplicate verification

An earlier feature route used an ORB-derived score whose denominator could produce invalid ratios above one. Rather than preserving its decisions, the reconstruction reopened all 2,031 candidate edges that depended on that route. The corrected ORB/RANSAC verifier limited the maximum image side to 800 pixels, extracted up to 1,200 ORB features, required Hamming-compatible matches and a normalised good-match ratio of at least 0.12, and accepted a geometric relation only when homography support, inlier ratio, spatial coverage, and reprojection error passed fixed thresholds. The corrected route retained 1,932 edges and rejected 99. This history matters because an audit is not made reliable by hiding a defective detector; reliability comes from reopening the afected decisions and recording the correction.

## 3.4 Trusted graph, direct-cover removal, and human review

The final trusted graph contained 6,196 exact edges, 445 strong-hash edges, 1,932 corrected feature edges, and 17 confirmed cross-label edges. Deduplication used a direct-cover rule: every removed image required a direct trusted relation to a retained representative. This is more conservative than deleting an entire transitive component when some links may be uncertain. The rule removed 8,355 duplicate files. Thirty-four images in unresolved cross-label families, six severe-quality failures, and the one-image rice\_neck\_blast category were quarantined.

A model-readiness stage then prioritised 180 images for human review. Manual review retained 137 images, deleted 43 black-screen or severely distorted images, and relabelled none. Model disagreement was never treated as automatic proof of annotation error. The surviving archive does not preserve reviewer count, blinding, independent duplicate review, or inter-rater agreement, so no such protocol claim is made. Figure 2 summarises the historical contamination and frozen replacement.

![](images/0a2716b904e4ef8d12b778be7738669fc03b63a74d1de1db1be9154891c549c5.jpg)

![](images/fb92f3b33c349e1ce96839c71092a8b06faa33be639304236db38bf4eb8795cb.jpg)  
Figure 2: Historical contamination and the frozen replacement.  
Note. Counts refer to audited trusted relationships and groups; undetected duplicate relationships remain possible.

## 3.5 Leakage-group-aware split and final freeze

Surviving files were assigned by leakage group, not independently by image, against 70/15/15 target shares. The frozen archive records seed 42, five group-safe cross-validation folds over train plus validation, and the resulting assignmen of 76,376 training, 16,368 validation, and 16,363 test images across 120 classes. The final build contains 109,107 unique SHA-256 identities and 109,090 audited trusted leakage groups; none of those audited groups spans train, validation, and test. The later manual-review freeze was removal-only: 43 rejected images were deleted without moving any retained row or re-running the split. The archive preserves the final row/group assignment and fingerprints but not a standalone implementation or complete tie-breaking trace of the earlier v5 assignment heuristic; exact historical split regeneration from aggregate summaries is therefore not claimed. Table 3 records the resulting audit and freeze ledger.

Table 3: Dataset audit and frozen benchmark ledger.
<table><tr><td rowspan=1 colspan=1>Property                                                                                                          Value</td></tr><tr><td rowspan=1 colspan=1>Audited source images                                                                                            117,546</td></tr><tr><td rowspan=1 colspan=1>Final images                                                                                                     109,107</td></tr><tr><td rowspan=1 colspan=1>Operational classes                                                                                                   120</td></tr><tr><td rowspan=1 colspan=1>Train / validation / test                                                                               76,376 /16,368/16,363</td></tr><tr><td rowspan=1 colspan=1>Historical duplicate relationships                                                                                     8,672</td></tr><tr><td rowspan=1 colspan=1>Historical cross-split duplicate relationships                                                                           3,233</td></tr><tr><td rowspan=1 colspan=1>Direct-cover duplicate removals                                                                                      8,355</td></tr><tr><td rowspan=1 colspan=1>Final leakage groups                                                                                              109,090</td></tr><tr><td rowspan=1 colspan=1>Leakage groups crossing final splits                                                                                       0</td></tr><tr><td rowspan=1 colspan=1>Largest:smallest class ratio                                                                                         151.7×</td></tr></table>

The class distribution remained severely imbalanced. The smallest total class contained 28 images and the largest

4,248. Fifteen classes had fewer than 20 test examples. Consequently, one error can move a four-example class by 25 percentage points. This is why class support accompanies per-class metrics and why the paper avoids presenting rare-class percentages as equally stable estimates.

## 3.6 Model-readiness diagnosis and residual risk

Cleaning answered whether known image families crossed the split. It did not answer what a model might learn instead of disease morphology. Before full fine-tuning, frozen DINOv3 ViT-S/16 embeddings were used for a linear probe, exact-neighbour analysis, centroid comparisons, and review prioritisation. The test linear probe reached 85.39% accuracy but only 68.70% macro-F1. The 16.69-point gap exposed the long tail before task-specific adaptation.

Separate probes measured label information in acquisition-associated features. Technical metadata reached 20.62% accuracy, 16×16 coarse imagery reached 30.59%, and border/framing features reached 23.74%, compared with 0.83% chance across 120 classes. By contrast, a split-membership classifier reached 32.76% against 33.33% chance. The combination is revealing: source and composition cues are label-predictive, yet the final partitions are well mixed in the frozen representation. Internal test performance can therefore be valid for the aggregate distribution while remaining optimistic for new sources. Figure 3 places the support distribution beside the diagnostic probes.

![](images/4f1ea13a4aa0000ef7d02703e0eb55da88ba47ef85b812984ff15477fc02f147.jpg)  
B Diagnostic probes before final training

![](images/8abd2ecb0e63c95292398feec32b6d9235030062d7806f90e1dc05264bc3d0b6.jpg)  
Figure 3: Long-tail training support and pre-training diagnostic probes.  
Note. Shortcut-probe performance above chance motivates the internal-validity boundary.

## 4 Model and Training Methodology

## 4.1 DINOv3 ConvNeXt-Tiny reference

The reference model was facebook/dinov3-convnext-tiny-pretrain-lvd1689m with a 120-way classifier at 256×256 input resolution. The entire backbone was fine-tuned. Validation macro-F1 selected the checkpoint because the diagnostic probe had already shown that accuracy and class-balanced performance could diverge. The completed configuration is recorded below. Optimiser betas and epsilon were not explicitly preserved in the reviewed notebook and are not reconstructed by assumption.

Table 4: Completed DINOv3 reference-training configuration.
<table><tr><td>Field</td><td>Completed reference configuration</td></tr><tr><td>Backbone</td><td>facebook/dinov3-convnext-tiny-pretrain-lvd1689m</td></tr><tr><td>Input / classes</td><td>256×256/120</td></tr><tr><td>Optimiser</td><td>AdamW (betas and epsilon not explicitly preserved)</td></tr><tr><td>Backbone / head learning rate</td><td>5e-05/3e-04</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Epochs / best epoch</td><td>40/14</td></tr><tr><td>Warm-up / minimum LR fraction</td><td>0.05/0.02</td></tr><tr><td>Scheduler</td><td>cosine with warm-up</td></tr><tr><td>Gradient clip norm</td><td>1.0</td></tr><tr><td>Micro-batch / effective batch</td><td>32/64</td></tr><tr><td>AMP/EMA</td><td>True / True (decay 0.9998)</td></tr><tr><td>Label smoothing</td><td>0.02</td></tr><tr><td>Seed</td><td>42</td></tr><tr><td>Selection rule</td><td>validation macro-F1</td></tr><tr><td>Geometric augmentation</td><td>RRC p=0.78, scale 0.70–1.00, ratio 0.75–1.33; flip p=0. 5; affine p=0. 4, ±12°, translation 0.05, scale 0.9-1.1</td></tr><tr><td>Photometric / codec augmentation</td><td>brightness 0.12, contrast 0.12, saturation 0.08, hue 0.01; JPEG p=0.10, quality 65–100; blur/noise p=0.05 each</td></tr><tr><td>Evaluation preprocessing</td><td>EXIF orientation, deterministic RGB conversion, aspect-preserving mean-colour square padding, bicubic resize, normalization</td></tr></table>

The augmentation policy was intentionally conservative for disease morphology. It combined controlled crop, afine, illumination, JPEG, blur, and noise variation without making grayscale, solarisation, large hue shifts, or extreme crops mandatory. Deterministic evaluation used EXIF-aware RGB conversion, aspect-preserving square padding, bicubic resize, and the frozen normalisation contract. The selected reference checkpoint was the human-readable epoch 14 and is identified by SHA-256 in Appendix C.

## 4.2 Deterministic and resumable training engine

Training ran under Kaggle session limits, so resumption was designed as part of experimental identity. Startup checks validated dataset and class-map fingerprints, split counts, duplicate identities, and leakage groups. Augmentation views were derived from the seed, epoch, and frozen row identifier. Epoch shufling was deterministic, and mid-epoch checkpoints were written only at optimiser-step boundaries. A resumable state included model, optimiser, scheduler, AMP scaler, EMA, random-number generators, epoch, sample ofset, global step, and configuration fingerprint. Atomic temporary-to-final replacement prevented partially written checkpoints from being treated as valid states.

These controls do not raise accuracy by themselves. They prevent an interrupted session from silently becoming a diferent experiment. Ten AMP-skipped updates were recorded; optimiser-dependent state was not advanced when those updates were skipped. The final evidence also includes AMP-versus-FP32 and batch-size parity checks, allowing numerical implementation efects to be separated from model-selection efects.

## 4.3 Locked evaluation and uncertainty

The test loader was disabled by default and required explicit authorization after the reference configuration, class map, preprocessing contract, and evaluation plan were frozen. Validation and test outputs preserved stable row identifiers, labels, logits, class order, and model/data fingerprints. The metric suite included accuracy, balanced accuracy, macro- and weighted F1, top-3 and top-5 accuracy, negative log-likelihood, Brier score, 15-bin expected calibration error, support slices, and complete per-class reports.

A class-stratified bootstrap over the fixed test rows produced uncertainty intervals for the reference estimate. Within each true class, rows were sampled with replacement at the original class size; paired comparisons applied the same sampled indices to both states. The reported intervals are percentile intervals from the 2.5th and 97.5th quantiles. For the reference-to-PTE analysis, 2,000 replicates used seed 20260821, and calibration temperatures and selective-prediction thresholds were not re-estimated inside replicates. This bootstrap describes sensitivity to the composition of the locked test sample; it does not measure retraining variability. The completed lineage contains one archived training seed for the reference and one for the compact model.

Numeric reporting. Headline prose, stat cards, and ordinary descriptive tables report percentages to two decimal places. Additional recorded precision is retained where it is necessary to distinguish PTQ candidates, model-state transitions, paired intervals, row-level confidences, or exact artifact identity. Appendix statistical tables preserve the released full precision.

## 4.4 Compact MobileNetV4 derivative

The compact state used mobilenetv4\_conv\_medium.e500\_r256\_in1k from timm 1.0.26, a 256×256 input, ImageNet normalisation, bicubic interpolation, and a 120-way linear classifier. Direct counting from the archived selected state gives 8,588,232 parameter tensors when batch-normalisation running statistics and counters are excluded. Exact MACs, activation memory, and peak resident memory were not measured for the frozen 120-class graph and are not inferred from a generic ImageNet model card. The 30-epoch history records total loss, hard-label loss, teacher-logit loss, feature-transfer loss, teacher reliability, and teacher top-1 correctness. The selected state was the raw best checkpoint. Train-only batch-normalisation recalibration and delayed-EMA-plus-BN alternatives did not improve the selected validation result

## Evidence boundary

The archive identifies the exact selected state and configuration fingerprint, but it does not preserve a complete human-readable record of the mobile optimiser, learning-rate schedule, distillation temperature, or objective coeficients. We therefore report the components that are directly recorded and do not claim exact recipe-level reproducibility for the historical mobile run. We also do not attribute the student’s performance causally to distillation: a matched direct MobileNetV4 run is absent. The supported statement is descriptive—the completed teacher-guided lineage reached the reported metrics.

## 4.5 State identity and comparison design

Four model states are kept separate throughout the analysis: the DINOv3 reference, the selected float MobileNetV4 state, the converted INT8 graph, and the serialised PTE. The float and quantised states operate on the same locked examples with the same class ordering. Pairwise comparisons therefore report both ground-truth metrics and top-1 agreement. Agreement is not treated as accuracy: two states can agree on the same wrong class. Every state-specific result is attached to raw predictions or logits, and the final artifact is attached to an exact SHA-256 digest.

## 5 Quantisation and Runtime Methodology

## 5.1 Validation-only PTQ selection

Post-training quantisation was treated as a model-selection problem rather than a file-conversion formality. Three predeclared XNNPACK-compatible candidates were evaluated on the full validation split before test access. The selection rule was pass-first and macro-F1-first, with a small tie preference defined in the frozen recipe. The candidates included dynamic activation quantisation with per-channel weights and two calibrated static alternatives. Only the selected recipe was carried into the locked test path.

Table 5: Validation-only post-training quantisation candidates.
<table><tr><td colspan="2"></td><td rowspan="2">Val. accuracy (%)</td><td rowspan="2">Val. balanced accuracy (%)</td><td rowspan="2">Val. macro-F1 (%)</td><td rowspan="2">Agreement with float (%)</td><td rowspan="2">Calibration rows</td><td rowspan="2">Gate</td></tr><tr><td>Candidate</td><td>Scheme</td></tr><tr><td>Dynamic</td><td>Dynamic activations; per-channel weights</td><td>98.4726</td><td>96.5189</td><td>96.4882</td><td>99.9389</td><td>0</td><td>√ PASS</td></tr><tr><td>Cal16</td><td>Static; 2,880 calibration rows</td><td>98.4299</td><td>96.4088</td><td>96.3458</td><td>99.6884</td><td>2,880</td><td>√ PASS</td></tr><tr><td>Cal8</td><td>Static; 1,440 calibration rows</td><td>98.4421</td><td>96.3787</td><td>96.3461</td><td>99.7006</td><td>1,440</td><td>X FAIL</td></tr></table>

Note. Repository identifiers: pc\_dynamic\_full, pc\_full\_cal16, and pc\_full\_cal8.

The dynamic candidate achieved the strongest validation macro-F1, class-balanced result, and float agreement, while requiring no calibration rows. The 2,880-row static candidate passed the quality gate but was weaker. The 1,440-row static candidate failed the predeclared gate. These results do not establish that dynamic quantisation is universally superior; they justify the selected path for this model, backend, and validation distribution.

## 5.2 Converted-graph evaluation

The selected converted graph was evaluated on the same locked test rows as the float state. Its accuracy changed by −0.006 percentage points, balanced accuracy by −0.048 points, and macro-F1 by −0.022 points. Float and converted INT8 predictions agreed on 99.951% of rows. A paired class-stratified bootstrap placed the converted-minus-float macro-F1 change between −0.097 and +0.024 percentage points at 95% confidence. The interval includes zero, supporting the practical conclusion that the selected conversion did not produce a clear aggregate class-balanced loss on the fixed test.

## 5.3 Serialisation and direct PTE execution

The converted graph was lowered to ExecuTorch with the XNNPACK backend and serialised as CropCop\_Mobile\_INT8\_ XNNPACK\_PRODUCTION\_v1.pte. The exact artifact contains 23,696,352 bytes, or 22.5986 MiB, and is identified by SHA-256 7c70d0f307f0d9578310600913cb8ff171b294ae4de0813f7cd1d6a53628bdf1. It is 31.92% smaller than the selected 34,805,405-byte prequant state file. That comparison is between two diferent containers, so it is reported as an artifact-size reduction rather than a theoretical fourfold INT8 compression claim.

Direct PTE execution generated a new prediction record on all 16,363 locked rows. The PTE is therefore not assigned the converted graph’s metrics by inheritance. Its accuracy, macro-F1, agreement, and disagreements are measured from the serialised program itself.

## 5.4 Runtime environment and quantisation coverage

Table 6: Archived runtime environment and quantisation coverage.
<table><tr><td>Evidence domain Archived value</td><td></td></tr><tr><td>Execution context</td><td>Kaggle-host CPU software-runtime evaluation; not Android device evidence</td></tr><tr><td>Software stack</td><td>Python 3.12.13;PyTorch / torchvision 2.12.1 / 0.27.1;ExecuTorch /torchao1.3.1 / 0.17.0;timm/</td></tr><tr><td>Backend / input</td><td>FlatBuffers 1.0.26 / 25.12.19 XNNPACK/[1, 3, 256, 256]</td></tr><tr><td>Quantisation contract</td><td>Requested scope ful1; dynamic activation / per-channel weights; explicit module exclusions []</td></tr><tr><td>Coverage evidence</td><td>√PASS Lowered backend modules / delegate tokens: 1 /1</td></tr><tr><td>Unarchived host details Physical Android evaluation</td><td>CPU model, thread count, and OS build</td></tr></table>

Note. The complete field-by-field environment register is retained for Appendix D in the full manuscript.

The archived graph records a full requested quantisation scope, no explicit module exclusions, a passed project coverage gate, and one lowered XNNPACK backend module/delegate token. These records support the intended backend path, but they do not provide an operator-level delegation dump: total delegated nodes, portable-fallback nodes, and partition boundaries were not archived. The paper therefore does not claim complete XNNPACK operator coverage. A host-only timing record exists, yet the CPU model, thread count, and operating-system build were not archived, so those latencies are excluded from the main performance claims. These omissions prevent host-latency reproduction but do not alter the reported INT8 prediction-fidelity comparison, which is computed from archived converted-graph and PTE outputs on the same locked rows. Android latency, memory, energy, delegate fallback, and thermal behaviour remain unmeasured.

## 6 Results

## 6.1 End-to-end model-state performance

The primary result is the retention of class-balanced performance across identified model states. The DINOv3 reference achieved 98.51% accuracy and 96.87% macro-F1. The float MobileNetV4 state finished 0.05 accuracy points and 0.60 macro-F1 points lower. Conversion to INT8 produced little further movement. Direct execution of the final PTE returned to the float state’s accuracy and finished 0.04 macro-F1 points below it.

Table 7: Performance across identified model states on the locked internal test.
<table><tr><td></td><td></td><td>Accuracy (%)</td><td>Balanced accuracy (%)</td><td>Macro-F1 (%)</td><td>Top-3 (%)</td><td>Top-5 (%)</td><td>Errors</td></tr><tr><td>State Reference</td><td>Architecture / object DINOv3 ConvNeXt-Tiny</td><td>98.5088</td><td>96.6836</td><td>96.8700</td><td>99.7250</td><td>99.8533</td><td>244</td></tr><tr><td>Float student</td><td>MobileNetV4 Conv-Medium</td><td>98.4599</td><td>96.2435</td><td>96.2710</td><td>99.7494</td><td>99.8594</td><td>252</td></tr><tr><td>Converted INT8</td><td>XNNPACK-compatible converted graph</td><td>98.4538</td><td>96.1957</td><td>96.2492</td><td>99.7372</td><td>99.8533</td><td>253</td></tr><tr><td>PTE runtime</td><td>Serialised ExecuTorch/XNNPACK program</td><td>98.4599</td><td>96.2017</td><td>96.2267</td><td>99.7433</td><td>99.8717</td><td>252</td></tr></table>

The table should not be read as an architecture tournament. The reference and compact states difer in architecture, training procedure, and role. It answers a narrower engineering question: how much of the strong reference result survived the completed compact and runtime path? On aggregate accuracy, nearly all of it survived. Macro-F1 reveals a small but measurable redistribution across classes. The four-state and PTQ results are visualised in Figure 4.

## 6.2 Benchmark reconstruction and task diagnosis

The benchmark result precedes the model result logically even though it follows it in presentation. The historical split was abandoned because 3,233 confirmed duplicate relationships crossed its boundaries. The frozen replacement contains 109,107 unique images and zero crossings among the audited trusted leakage groups. This removes a known source of train–test family overlap, but it does not remove source bias or ontology ambiguity.

The diagnostic probes clarify the remaining problem. A frozen DINOv3 representation reached 85.39% accuracy and 68.70% macro-F1, showing that much of the label structure was visible before fine-tuning while minority performance lagged. Metadata, 16×16 imagery, and border probes achieved 20.62%, 30.59%, and 23.74% accuracy, all far above chance. Split membership remained at chance. The final split is internally coherent, yet it shares label-correlated acquisition cues across partitions. This is why the paper uses the phrase leakage-controlled internal performance rather than field generalisation.

## 6.3 Paired reference-to-PTE analysis

Point estimates alone understate the structure of the reference-to-runtime change. Across the same 16,363 rows, the reference and PTE difered on 240 top-1 predictions. The reference was uniquely correct on 105 of those rows and the PTE uniquely correct on 97; 38 rows were wrong under both states but assigned diferent labels. The PTE made 252 errors, eight more than the reference. An exact two-sided McNemar test on the correctness indicators gave p = 0.622 [28], so the aggregate accuracy diference is not distinguishable in that paired test.

Class-balanced behaviour tells a sharper story. In 2,000 class-stratified paired bootstrap replicates, the PTE’s accuracy change had a 95% interval spanning zero, as did the balanced-accuracy change. The macro-F1 change was −0.64 percentage points with a 95% interval of −1.24 to −0.05 points. An exploratory post hoc slice defined mechanically by labels containing the literal substring fruit showed a 6.97-point mean-recall decline, with a percentile interval of −12.41 to −1.71 points. Its exact membership is released with the evidence package; because the slice was examined during error analysis rather than preregistered, it is descriptive and hypothesis-generating. The overall result is therefore not “no loss.” It is near-parity in aggregate correctness with a modest, concentrated reduction in class-balanced performance.

A Retention across identified model states  
![](images/70d995c6dc4716c97e5d7fcb90c0057aa6f199dcc6d4e005ae0b488b48cb490f.jpg)

B Validation-only PTQ selection  
![](images/fba5c3e7ecf34f64656e5c8a66b65f83041d9cc6a2c806d4623fb059cfc95e5c.jpg)  
Selected: dynamic activations with per-channel weights.  
Figure 4: Accuracy, balanced accuracy, and macro-F1 across the four identified model states, with validation-only PTQ candidate results.

Table 8: Paired reference-to-PTE changes with class-stratified percentile intervals.
<table><tr><td>Metric</td><td>Reference (%)</td><td>PTE (%)</td><td>PTE – reference</td><td>Paired 95% interval</td></tr><tr><td>Accuracy</td><td>98.509</td><td>98.460</td><td>(pp) -0.049</td><td>(pp) [−0.214,+0.116]</td></tr><tr><td>Balanced accuracy</td><td>96.684</td><td>96.202</td><td>-0.482</td><td>[-1.097, +0.113]</td></tr><tr><td>Macro-F1</td><td>96.870</td><td>96.227</td><td>-0.643</td><td>[-1.239,-0.047]</td></tr><tr><td>Fruit-category mean recall</td><td>85.936</td><td>78.971</td><td>-6.965</td><td>[−12.409,-1.711]</td></tr></table>

Note. Full-precision estimates and support-bin recall intervals are reported in Appendix C.1.

## 6.4 Long-tail retention and class-level redistribution

Disjoint training-support bins show where the compact runtime paid the largest cost. Classes with fewer than 100 training examples lost 2.56 macro-F1 points on average, and the 100–199 bin lost 2.78 points. The three higher-support bins changed by less than one tenth of a point in magnitude, except that the ≥1,000 bin improved slightly. These are descriptive averages over heterogeneous classes; support alone does not determine dificulty.

Table 9: Performance retention by disjoint training-support bin.
<table><tr><td>Training support</td><td>Classes</td><td>Reference macro-F1 (%)</td><td>PTE macro-F1 (%)</td><td>Change (pp)</td><td>Reference recall (%)</td><td>PTE recall (%)</td><td>Change (pp)</td></tr><tr><td>&lt; 100</td><td>17</td><td>91.17</td><td>88.60</td><td>-2.56</td><td>89.37</td><td>87.27</td><td>-2.10</td></tr><tr><td>100-199</td><td>12</td><td>92.84</td><td>90.05</td><td>-2.78</td><td>93.09</td><td>90.27</td><td>-2.82</td></tr><tr><td>200-499</td><td>38</td><td>97.91</td><td>97.85</td><td>-0.06</td><td>98.04</td><td>98.34</td><td>+0.30</td></tr><tr><td>500-999</td><td>30</td><td>99.07</td><td>98.98</td><td>-0.09</td><td>99.10</td><td>99.01</td><td>-0.09</td></tr><tr><td>≥ 1000</td><td>23</td><td>98.60</td><td>98.81</td><td>+0.21</td><td>98.58</td><td>98.71</td><td>+0.13</td></tr></table>

The per-class deltas reinforce that warning, but their ranking is descriptive: several classes have only 7–30 test examples, so small changes in TP, FP, or FN can move F1 sharply. Exact reconstructed TP/FP/FN counts are released beside the delta table. banana\_disease\_fruit fell from 0.7778 to 0.6000 F1 on 11 test images. Broad fruit-condition categories account for several of the largest declines, including mango\_disease\_fruit, guava\_healthy\_fruit, and papaya\_fruit\_healthy. By contrast, plant\_healthy improved from 0.7609 to 0.8824 F1. Compression did not merely lower every score by a fixed amount; it altered a small set of class boundaries.

Table 10: Largest class-level F1 declines.
<table><tr><td>Class</td><td>Train support</td><td>Test support</td><td>Reference F1 (0-1)</td><td>PTE F1 (0-1)</td><td>Δ F1 (0-1)</td></tr><tr><td>banana_disease_fruit</td><td>51</td><td>11</td><td>0.7778</td><td>0.6000</td><td>-0.1778</td></tr><tr><td>mango_disease_fruit</td><td>143</td><td>30</td><td>0.8788</td><td>0.7213</td><td>-0.1575</td></tr><tr><td>guava_healthy_fruit</td><td>35</td><td>7</td><td>1.0000</td><td>0.8571</td><td>-0.1429</td></tr><tr><td>guava_disease_fruit</td><td>110</td><td>23</td><td>0.8000</td><td>0.7234</td><td>-0.0766</td></tr><tr><td>papaya_fruit_healthy</td><td>96</td><td>20</td><td>0.9231</td><td>0.8500</td><td>-0.0731</td></tr><tr><td>tomato_mosaic_virus</td><td>240</td><td>51</td><td>0.9703</td><td>0.9231</td><td>-0.0472</td></tr><tr><td>guava_healthy_leaf</td><td>105</td><td>22</td><td>1.0000</td><td>0.9545</td><td>-0.0455</td></tr><tr><td>mango_healthy_fruit</td><td>227</td><td>49</td><td>0.9216</td><td>0.8824</td><td>-0.0392</td></tr><tr><td>aloevera_healthy</td><td>63</td><td>14</td><td>1.0000</td><td>0.9630</td><td>-0.0370</td></tr><tr><td>raspberry_healthy</td><td>264</td><td>57</td><td>0.9655</td><td>0.9298</td><td>-0.0357</td></tr></table>

Note. The ranking is descriptive: several classes have only 7–30 test examples, so small changes in TP, FP, or FN can move F1 sharply.

The class-level redistribution is visualised in Figure 5.

A Per-class retention  
![](images/0697da731b52dc07ce77d4ccabe81d69674f837ef223f78279987bc8746c761a.jpg)

B Effect concentration by training support  
![](images/7bb56ff3d344c786c2b3f53601986d057fe4808528d15ca98a28a73f00b9d005.jpg)  
Figure 5: Reference-versus-PTE class F1; point area is proportional to test support.  
Note. Labelled declines are descriptive for small classes.

## 6.5 From converted graph to serialised PTE: six changed decisions

Converted INT8 and PTE predictions difered on exactly six of 16,363 rows. Two changes turned correct INT8 predictions into PTE errors, three corrected INT8 errors, and one moved between two wrong classes. As a result, PTE accuracy was one correct prediction higher than converted INT8 accuracy, even though macro-F1 was 0.023 points lower. The diference is possible because each changed prediction afects class precision and recall diferently

None of the six changed decisions was high-margin. Converted confidences ranged from 0.262 to 0.530, and every margin was below 0.13. The serialisation changes therefore occurred near contested decision boundaries rather than among unequivocal predictions. This audit is more informative than agreement alone: 99.963% agreement becomes a concrete list of six cases that can be inspected and reproduced.

Row-level audit. The complete six-row converted-INT8-to-PTE disagreement table is retained in Appendix C.2. Figure 6B–C preserves the 2 / 3 / 1 transition summary and the margin distribution in the main results narrative.

## 6.6 Calibration, selective prediction, and error taxonomy

Raw PTE ECE15 was 1.58% and NLL was 0.0799 on the internal test. Calibration numbers describe the frozen distribution, not confidence under domain shift. To examine whether confidence could support review, thresholds were fixed independently from each state’s validation outputs at target coverages of 80%, 90%, 95%, and 98%, then applied once to test outputs. At the 95% target, the reference covered 94.88% of test rows at 99.81% accepted-set accuracy; the PTE covered 94.82% at 99.67%. No threshold from this internal analysis is proposed as a field-deployment policy.

Table 11: Validation-derived selective-prediction operating points.
<table><tr><td>Validation target coverage (%)</td><td>Reference test coverage (%)</td><td>Reference accepted accuracy (%)</td><td>PTE test coverage (%)</td><td>PTE accepted accuracy (%)</td></tr><tr><td>80</td><td>79.80</td><td>99.824</td><td>79.72</td><td>99.678</td></tr><tr><td>90</td><td>89.79</td><td>99.816</td><td>89.90</td><td>99.681</td></tr><tr><td>95</td><td>94.88</td><td>99.813</td><td>94.82</td><td>99.671</td></tr><tr><td>98</td><td>97.99</td><td>99.451</td><td>97.90</td><td>99.407</td></tr></table>

An explicit threshold sweep further clarifies the phrase high-confidence error. For the PTE, 82 wrong predictions had raw softmax confidence above 0.95, 18 exceeded 0.99, and none exceeded 0.999. These counts should not be compared directly with the reference without accounting for its temperature scaling and confidence distribution. Their purpose is to show that even a highly accurate internal model can make confident mistakes

The descriptive error taxonomy was derived mechanically from operational label strings, not from biological adjudication. Same-crop condition confusions dominated both states. Fruit-condition involvement increased from 47 reference errors to 64 PTE errors, while errors involving the generic plant\_healthy label decreased. This pattern agrees with the per-class analysis: the main compression cost is concentrated in broad, visually heterogeneous fruit categories rather than spread evenly across the ontology.

Operational taxonomy. The complete six-category reference-to-PTE error taxonomy is retained in Appendix C.3. The main-text interpretation preserves the two largest directional changes: fruit-condition involvement increased from 47 to 64 errors, while generic plant\_healthy involvement decreased from 22 to 12.

Figure 6 summarises the validation-derived operating points and the six converted-INT8-to-PTE changes.

A Validation-derived selective prediction  
![](images/97dfc85e355f47b481b4aab8e67e6aa283a365e8366f94956b7279b9dff46d99.jpg)

B Six converted-INT8-to-PTE changes  
![](images/9ec9a52cb41a38b2090fda00e4017a3ab6fa948b709e9baad4b10ef58f4e4ee1.jpg)

C All changes occurred near contested boundaries  
![](images/d28bf96a3ef2ae13e1c1c1c6451dda54bf297661b7e7ecf347cdaedc6f5e03bd.jpg)  
Figure 6: Validation-derived selective-prediction operating points and the six converted-INT8-to-PTE decision changes.

## 7 Discussion

## 7.1 What the paper establishes

CropCop establishes that a broad, long-tailed plant-health model can retain high internal performance across a traceable sequence of benchmark reconstruction, foundation-model adaptation, compact training, quantisation, backend lowering, and direct runtime execution. The final 98.46% accuracy is not assigned to a project name; it is attached to a particular split, class map, model lineage, prediction record, PTE file, and SHA-256 digest. That traceability is the paper’s central contribution. It allows a reviewer to ask where a change occurred and answer with paired rows rather than with speculation.

## 7.2 The model result is strong, but not lossless

The PTE finished only 0.05 accuracy points below the reference, and the paired correctness comparison did not show a clear aggregate accuracy diference. Macro-F1, however, declined by 0.64 points with a paired interval excluding zero. This is an important distinction. A compact system can preserve nearly every top-1 decision while changing the distribution of those decisions across classes. Reporting only accuracy would erase the main cost of compression; reporting only macro-F1 would obscure how small the row-level change actually was.

## 7.3 What the error redistribution teaches

The weakest changes are not random. Low-support groups and broad fruit-condition labels lose more than high-suppor groups, while some classes improve. The result points to two distinct constraints. Limited examples make a boundary sensitive to model capacity and training variation. Broad or inconsistent labels impose ambiguity that a larger backbone cannot fully remove. Additional tuning on the same internal split may therefore have less value than clearer ontology definitions and independently collected examples for the afected categories.

The operational taxonomy also separates same-crop disease confusion from cross-crop mistakes. Same-crop condition confusions remain the largest category under both reference and PTE states, which is consistent with fine-grained visual similarity. Fruit-condition involvement grows under the PTE, aligning with the paired decline in fruit recall. These analyses do not prove why an image was misclassified, but they identify where the next data and review efort should be directed.

## 7.4 What foundation transfer and teacher guidance do not prove

DINOv3 supplied the completed reference lineage and the frozen diagnostic representation, but the study does not include a matched non-DINO ConvNeXt control. The evidence therefore shows that a DINOv3-pretrained reference performed strongly, without isolating the improvement attributable to DINOv3. The compact history records hard-label, teacherlogit, and feature-transfer components, yet no matched direct MobileNetV4 baseline survives. The paper accordingly describes a teacher-guided training path without claiming a causal distillation gain. These are natural experiments for an expanded model study, but they are not required to support the current audit-to-runtime result.

## 7.5 Internal validity and the next evidence boundary

The reconstructed split addresses known duplicate-family leakage. It does not create a new acquisition domain. Metadata, low-frequency imagery, and borders remain label-predictive, and train, validation, and test originate from the same audited pool. The next decisive evaluation is therefore not another decimal place on the consumed internal test. It is a source-independent smartphone cohort with a prespecified mapping to the 120-class ontology, followed by actual Android measurements of latency, memory, energy, delegate coverage, and thermal behaviour. Until those studies exist, CropCop is best understood as a strongly verified internal recognition system and executable software artifact, not an autonomous field diagnostic.

## 8 Limitations

## RESEARCH LIMITATIONS STATEMENT

External validity. The test is controlled against the audited trusted leakage graph but internal to the reconstructed source pool. Zero observed group crossings do not exclude false negatives in duplicate detection. It cannot establish performance on unseen farms, regions, cultivars, cameras, backgrounds, or acquisition protocols. The shortcut probes make this boundary visible rather than eliminating it.

Provenance and ontology. The surviving archive does not reconstruct complete source URLs, annotation authority, cultivar, geography, severity, or redistribution rights for every image. The 120 operational labels mix disease specificity and visual target types. CropCop is therefore not presented as a legally homogeneous public image release or a definitive botanical taxonomy.

Causal model analysis. The completed evidence lacks a matched direct MobileNetV4 baseline and a matched non-DINO reference control. The study cannot assign the compact result to distillation, feature transfer, or DINOv3 pretraining as isolated causes.

Historical mobile configuration. The selected mobile state, fingerprint, component loss histories, preprocessing contract, and predictions are preserved. The exact optimiser, schedule, distillation temperature, and objective coeficients are not available in a complete human-readable configuration. This limits exact recipe-level reproduction of the historical student run.

Training variability. Each completed final lineage is represented by one archived training seed. Bootstrap intervals quantify uncertainty over the fixed test sample, not variation across retraining. Multi-seed claims are therefore withheld.

Runtime scope. Direct PTE execution proves software-runtime predictions in the archived Kaggle-host environment. The archive does not contain an operator-level XNNPACK delegation/fallback dump, and it does not establish Android latency, memory, energy, delegate fallback, or thermal behaviour. Host-only timings are not used as phone performance evidence.

Consumed test. The final test has been opened for the frozen lineage and the paired analyses reported here. It must not be used to choose new architectures, losses, thresholds, or quantisation recipes. Later studies should rely on validation-only ablations and a new external cohort for consequential comparisons.

## 9 Data and Artifact Availability

The GitHub companion repository provides the public reproducibility package. It contains the manuscript source, canonical PDF, metric registry, claim-evidence matrix, complete per-class reference and PTE tables, PTQ summaries, paired analyses, deterministic figure-generation code, sanitized notebooks and certificates, model/data cards, and cryptographic identifiers. The repository validator checks manuscript/evidence consistency, relative links, notebook sanitation, both checksum ledgers, citation metadata, and release boundaries. An immutable preprint tag and release manifest will be frozen only after the arXiv identifier is assigned; until then, the repository is a release candidate rathe than the archival citation target.

The consolidated image corpus is not distributed with this manuscript because complete image-level provenance and redistribution rights have not been reconstructed. The final PTE and model states are identified by hashes but should no be released until upstream-model and dataset-derived distribution obligations have been reviewed. A release manifest will bind the paper PDF, source archive, repository commit, evidence package, and checksums to the same version.

## 10 Intended Use and Safety

## INTENDED-USE AND SAFETY STATEMENT

CropCop is a closed-set research classifier. It always chooses among 120 known labels and has not been evaluated on unsupported crops, novel diseases, non-plant images, or a source-independent field cohort. A high-confidence output can still be wrong, particularly when the image lies outside the frozen distribution. The system must not be presented as autonomous agronomic diagnosis or as a treatment recommender.

Any consequential application should disclose uncertainty, provide abstention and human-review paths, validate the complete image-preprocessing pipeline on target devices, and test performance on independently collected smartphone data. Agronomic decisions should remain subject to qualified expertise and local context.

## 11 Reproducibility Statement

## REPRODUCIBILITY STATEMENT

The evidence package preserves the final dataset and class fingerprints, group-safe split counts, reference configuration, checkpoint and artifact hashes, stable row identifiers, raw prediction records, per-class reports, PTQ selection record, and direct PTE outputs. Reference training can be reconstructed from the archived notebook and configuration values reported in Section 4. The paired analyses in Section 6 are derived from the locked rows and are released as machine-readable tables.

The compact model’s exact historical optimiser and objective coeficients are the principal reproducibility gap. The manuscript does not hide that gap or fill it by reverse-engineering loss curves. A future reproduced mobile recipe would require a new version identifier and should not be presented as proof of the missing historical settings.

## 12 AI-Assistance Disclosure

## AI-ASSISTANCE DISCLOSURE

Generative-AI tools assisted literature organisation, language editing, code inspection, and consistency checking during manuscript preparation. All empirical claims were checked against archived project evidence or explicitly derived from released prediction records. The authors remain responsible for the study design, analyses, claims, and final text.

## 13 Conclusion

CropCop began as a broad model-building project and became a benchmark-reconstruction project before it could become a model-building project again. Rejecting a partition crossed by 3,233 duplicate relationships produced a frozen 109,107-image, 120-class benchmark with zero crossings among the audited trusted leakage groups. On that benchmark, a DINOv3 ConvNeXt-Tiny reference achieved 98.51% accuracy and 96.87% macro-F1, while a compact MobileNetV4 lineage preserved 98.46% accuracy and 96.27% macro-F1.

The final result is an identified runtime object rather than an inherited checkpoint number. The 22.60 MiB Execu-Torch/XNNPACK PTE was executed directly, reached 98.46% accuracy and 96.23% macro-F1, and changed only six decisions relative to the converted INT8 graph. Paired analysis shows where the remaining cost lies: not in a clear aggregate accuracy loss, but in a modest reduction in class-balanced performance concentrated among low-support and fruit-condition categories. CropCop closes the internal loop from data identity to runtime identity. New farms, phones, and acquisition pipelines remain the next loop to test.

## A Complete per-class results

The complete 120-class reference and PTE precision, recall, F1, and support tables are distributed as machine-readable files in the companion repository:

<sup>•</sup> metrics/reference\_per\_class\_test.csv

<sup>•</sup> metrics/pte\_runtime\_per\_class\_test.csv

<sup>•</sup> evidence/derived/reference\_vs\_pte\_per\_class\_delta.csv

<sup>•</sup> evidence/derived/reference\_vs\_pte\_per\_class\_counts.csv

These files preserve class order and full numerical precision. The main paper reports the support-bin summaries and the largest descriptive class-level changes rather than reproducing seven pages of dense tables. Appendix E reports the corresponding human-readable crop, label-type, and split-support inventory in the same class order.

## B Model and dataset identity register

Table 12: Model and dataset identity registry.

<table><tr><td>Object Identifier</td><td></td></tr><tr><td>Final retained dataset semantic fingerprint</td><td>7c368e6e3d8be3bb3a9a3a5f961075d4 faa125bcac2e98a3b55e1a1c61f1c523</td></tr><tr><td>Final dataset build fingerprint</td><td>2d7c237981b8943d9b08a522db048984</td></tr><tr><td>Dataset manifest SHA-256</td><td>9a4bfe7f488dfb6461489dd36dcdc12b bdb82211ccc2059153724eea178a1680</td></tr><tr><td>Class-map SHA-256</td><td>893a6b38ecc243fae484baa91dbf68e2 46f7811726c19c42bd7213b2d8178b19</td></tr><tr><td></td><td>a5a182a1b763f60a94ee2c0e5f6688d2</td></tr><tr><td>Reference checkpoint SHA-256</td><td>74b4701b8931976c9227845ead50788a</td></tr><tr><td>Mobile configuration fingerprint</td><td>e47e3596f575f2817b7352a715f53b79</td></tr><tr><td rowspan="2"></td><td>a75442087ee1c148b1dcd68441f2fd87</td></tr><tr><td></td></tr><tr><td rowspan="2">Selected float-state SHA-256</td><td>8d8738844532ece4d5606a0038ffb91e</td></tr><tr><td>b41e76660bcc1042991282aed2ea4067</td></tr><tr><td rowspan="2">PTQ selection fingerprint</td><td>ba94677116604b622eda6a8f1eb00c00</td></tr><tr><td>bb2ccae4c03f27099cdd1bcbf392676</td></tr><tr><td rowspan="2">Final PTE SHA-256</td><td>355b637df4c9f6c15b49d12884994e20f</td></tr><tr><td>7c70d0f307f0d9578310600913cb8ff1</td></tr></table>

## C Paired state-transition details

## C.1 Reference-to-PTE paired bootstrap

Table 13: Paired bootstrap state-transition summary.

<table><tr><td>Metric</td><td>Reference point</td><td>PTE point</td><td>PTE reference</td><td>95% lower</td><td>95% upper</td></tr><tr><td>Accuracy</td><td>0.985088</td><td>0.984599</td><td>-0.000489</td><td>-0.002140</td><td>+0.001161</td></tr><tr><td>Balanced accuracy</td><td>0.966836</td><td>0.962017</td><td>-0.004819</td><td>-0.010967</td><td>+0.001128</td></tr><tr><td>Macro-F1</td><td>0.968700</td><td>0.962267</td><td>-0.006433</td><td>-0.012386</td><td>-0.000467</td></tr><tr><td>Fruit mean recall</td><td>0.859364</td><td>0.789711</td><td>-0.069653</td><td>-0.124086</td><td>-0.017108</td></tr><tr><td>Recall &lt; 100</td><td>0.893703</td><td>0.872686</td><td>-0.021017</td><td>-0.057207</td><td>+0.013036</td></tr><tr><td>Recall 100-199</td><td>0.930925</td><td>0.902687</td><td>-0.028238</td><td>-0.056711</td><td>+0.000623</td></tr><tr><td>Recall 200-499</td><td>0.980353</td><td>0.983360</td><td>+0.003007</td><td>-0.002884</td><td>+0.008774</td></tr><tr><td>Recall 500-999</td><td>0.991013</td><td>0.990140</td><td>-0.000873</td><td>-0.003038</td><td>+0.001292</td></tr><tr><td>Recall ≥ 1000</td><td>0.985762</td><td>0.987056</td><td>+0.001294</td><td>-0.001100</td><td>+0.003858</td></tr></table>

## C.2 Converted-INT8-to-PTE disagreement audit

Table 14: Converted-INT8-to-PTE disagreement audit.
<table><tr><td>Row ID True class</td><td></td><td>Converted INT8</td><td>PTE</td><td>INT8 confidence</td><td>PTE confidence</td></tr><tr><td>1455</td><td>banana_healthy_fruit</td><td>banana_healthy_fruit√</td><td>banana_disease_fruit ×</td><td>0.479</td><td>0.477</td></tr><tr><td>3891</td><td>corn_northern_leaf_blight</td><td>corn_northern_leaf_blight√</td><td>wheat_leaf_rust X</td><td>0.303</td><td>0.358</td></tr><tr><td>6292</td><td>nutrient_deficiency</td><td>pear_black_spot ×</td><td>nutrient_deficiency√</td><td>0.486</td><td>0.468</td></tr><tr><td>11743</td><td>sugarcane_yellow_leaf</td><td>sugarcane_mosaic ×</td><td>sugarcane_yellow_leaf √</td><td>0.530</td><td>0.472</td></tr><tr><td>12361</td><td>tomato_early_blight</td><td>tomato_septoria_leaf_spot ×</td><td>tomato_early_blight√</td><td>0.481</td><td>0.477</td></tr><tr><td></td><td>12750 tomato_healthy</td><td>apple_healthy ×</td><td>cherry_healthy ×</td><td>0.262</td><td>0.267</td></tr></table>

## C.3 Reference-to-PTE operational error taxonomy

Table 15: Reference-to-PTE operational error taxonomy.

<table><tr><td>Operational error category</td><td>Reference errors</td><td>PTE errors</td><td>Change</td></tr><tr><td>Cross-crop condition confusion</td><td>52</td><td>48</td><td>-4</td></tr><tr><td>Cross-crop healthy/condition confusion</td><td>6</td><td>9</td><td>+3</td></tr><tr><td>Fruit-condition label involved</td><td>47</td><td>64</td><td>+17</td></tr><tr><td>Generic plant_healthy involved</td><td>22</td><td>12</td><td>-10</td></tr><tr><td>Same-crop condition confusion</td><td>88</td><td>90</td><td>+2</td></tr><tr><td>Same-crop healthy/condition confusion</td><td>29</td><td>29</td><td>+0</td></tr></table>

## D Runtime and quantisation environment

Table 16: Archived runtime and quantisation environment.
<table><tr><td>Field</td><td>Archived value</td></tr><tr><td>Evaluation context</td><td>Kaggle-host CPU software-runtime evaluation; not Android device evidence</td></tr><tr><td>Python</td><td>3.12.13</td></tr><tr><td>PyTorch / torchvision</td><td>2.12.1 / 0.27.1</td></tr><tr><td>ExecuTorch / torchao</td><td>1.3.1 / 0.17.0</td></tr><tr><td>timm / FlatBuffers</td><td>1.0.26 / 25.12.19</td></tr><tr><td>Backend /input</td><td>XNNPACK/[1, 3, 256, 256]</td></tr><tr><td>Requested quantisation scope</td><td>full</td></tr><tr><td>Activation / weight scheme</td><td>dynamic activation / per-channel weights</td></tr><tr><td>Explicit module exclusions</td><td>[]</td></tr><tr><td>Coverage gate</td><td>√PASS</td></tr><tr><td>Lowered backend modules / delegate tokens</td><td>1  /  1</td></tr><tr><td>Unarchived host details</td><td>CPU model, thread count, and OS build</td></tr><tr><td>Physical Android evaluation</td><td>Not evaluated</td></tr></table>

## E Operational class taxonomy and support inventory

Table 17 preserves the authoritative class order and reports the frozen train and test support for every operational label. Class labels follow data\_card/class\_to\_idx.json; supports follow data\_card/class\_distribution.csv and agree row-for-row with both per-class metric tables and the two reference-to-PTE derived tables.

Table 17: Operational class taxonomy and split support.
<table><tr><td>Operational class</td><td>Crop/species</td><td>Label type</td><td>Train</td><td>Test</td></tr><tr><td>aloevera_healthy</td><td>aloevera</td><td>Generic healthy state</td><td>63</td><td>14</td></tr><tr><td>amla_healthy</td><td>amla</td><td>Generic healthy state</td><td>49</td><td>10</td></tr><tr><td>apple_black_rot</td><td>apple</td><td>Specific named disease</td><td>1,426</td><td>306</td></tr><tr><td>apple_black_spot</td><td>apple</td><td>Specific named disease</td><td>218</td><td>47</td></tr><tr><td>apple_brown_spot</td><td>apple</td><td>Specific named disease</td><td>696</td><td>149</td></tr><tr><td>apple_cedar_rust</td><td>apple</td><td>Specific named disease</td><td>202</td><td>43</td></tr><tr><td>apple_disease_fruit</td><td>apple</td><td>Broad condition category</td><td>197</td><td>42</td></tr><tr><td>apple_healthy</td><td>apple</td><td>Generic healthy state</td><td>2,143</td><td>459</td></tr><tr><td>apple_scab</td><td>apple</td><td>Specific named disease</td><td>399</td><td>86</td></tr><tr><td>apricot_blight</td><td>apricot</td><td>Specific named disease</td><td>165</td><td>35</td></tr><tr><td>apricot_healthy</td><td>apricot</td><td>Generic healthy state</td><td>418</td><td>89</td></tr><tr><td>apricot_shot_hole</td><td>apricot</td><td>Specific named disease</td><td>546</td><td>117</td></tr><tr><td>banana_cordana</td><td>banana</td><td>Specific named disease</td><td>185</td><td>39</td></tr><tr><td>banana_disease_fruit</td><td>banana</td><td>Broad condition category</td><td>51</td><td>11</td></tr><tr><td>banana_healthy_fruit</td><td>banana</td><td>Generic healthy state</td><td>89</td><td>19</td></tr><tr><td>banana_healthy_leaf</td><td>banana</td><td>Generic healthy state</td><td>68</td><td>15</td></tr><tr><td>bean_fungal_disease</td><td>bean</td><td>Broad condition category</td><td>495</td><td>106</td></tr><tr><td>bean_healthy</td><td>bean</td><td>Generic healthy state</td><td>560</td><td>120</td></tr><tr><td>bean_rust</td><td>bean</td><td>Specific named disease</td><td>324</td><td>69</td></tr><tr><td>bean_shot_hole</td><td>bean</td><td>Specific named disease</td><td>250</td><td>54</td></tr><tr><td>blueberry_healthy</td><td>blueberry</td><td>Generic healthy state</td><td>894</td><td>191</td></tr><tr><td>cherry_brown_spot</td><td>cherry</td><td>Specific named disease</td><td>642</td><td>138</td></tr><tr><td>cherry_healthy</td><td>cherry</td><td>Generic healthy state</td><td>772</td><td>165</td></tr><tr><td>cherry_leaf_scorch</td><td>cherry</td><td>Specific named disease</td><td>632</td><td>135</td></tr><tr><td>cherry_powdery_mildew</td><td>cherry</td><td>Specific named disease</td><td>576</td><td>124</td></tr><tr><td>cherry_purple_leaf_spot</td><td>cherry</td><td>Specific named disease</td><td>665</td><td>143</td></tr><tr><td>cherry_shot_hole</td><td>cherry</td><td>Specific named disease</td><td>275</td><td>59</td></tr><tr><td>citrus_canker_leaf</td><td>citrus</td><td>Specific named disease</td><td>117</td><td>25</td></tr><tr><td>citrus_fruit_disease</td><td>citrus</td><td>Broad condition category</td><td>87</td><td>19</td></tr><tr><td>citrus_healthy_fruit</td><td>citrus</td><td>Generic healthy state</td><td>128</td><td>28</td></tr><tr><td>citrus_healthy_leaf</td><td>citrus</td><td>Generic healthy state</td><td>96</td><td>21</td></tr><tr><td>corn_common_rust</td><td>corn</td><td>Specific named disease</td><td>1,556</td><td>334</td></tr><tr><td>corn_fungal_leaf</td><td>corn</td><td>Broad condition category</td><td>147</td><td>32</td></tr><tr><td>corn_gray_leaf_spot</td><td>corn</td><td>Specific named disease</td><td>674</td><td>144</td></tr><tr><td>corn_healthy</td><td>corn</td><td>Generic healthy state</td><td>1,555</td><td>333</td></tr><tr><td>corn_holcus_leaf_spot</td><td>corn</td><td>Specific named disease</td><td>219</td><td>47</td></tr><tr><td>corn_northern_leaf_blight</td><td>corn</td><td>Specific named disease</td><td>1,577</td><td>338</td></tr><tr><td>crape_jasmine_healthy</td><td>crape_jasmine</td><td>Generic healthy state</td><td>32</td><td>7</td></tr><tr><td>fig_blight</td><td>fig</td><td>Specific named disease</td><td>514</td><td>110</td></tr><tr><td>fig_brown_spot</td><td>fig</td><td>Specific named disease</td><td>261</td><td>57</td></tr><tr><td>fig_healthy</td><td>fig</td><td>Generic healthy state</td><td>400</td><td>86</td></tr><tr><td>fig_rust</td><td>fig</td><td>Specific named disease</td><td>331</td><td>71</td></tr><tr><td>grape_anthracnose</td><td>grape</td><td>Specific named disease</td><td>675</td><td>144</td></tr><tr><td>grape_black_rot</td><td>grape</td><td>Specific named disease</td><td>694</td><td>149</td></tr><tr><td>grape_brown_spot</td><td>grape</td><td>Specific named disease</td><td>432</td><td>93</td></tr><tr><td>grape_downy_mildew</td><td>grape</td><td>Specific named disease</td><td>418</td><td>89</td></tr><tr><td>grape_esca</td><td>grape</td><td>Specific named disease</td><td>775</td><td>166</td></tr><tr><td>grape_healthy</td><td>grape</td><td>Generic healthy state</td><td>1,239</td><td>266</td></tr><tr><td>grape_leaf_blight</td><td>grape</td><td>Specific named disease</td><td>603</td><td>129</td></tr><tr><td>grape_mites</td><td>grape</td><td>Specific named disease†</td><td>282</td><td>60</td></tr><tr><td>grape_powdery_mildew</td><td>grape</td><td>Specific named disease</td><td>803</td><td>172</td></tr><tr><td>grape_shot_hole</td><td>grape</td><td>Specific named disease</td><td>557</td><td>120</td></tr><tr><td>guava_disease_fruit</td><td>guava</td><td>Broad condition category</td><td>110</td><td>23</td></tr><tr><td>guava_healthy_fruit</td><td>guava</td><td>Generic healthy state</td><td>35</td><td>7</td></tr><tr><td>guava_healthy_leaf</td><td>guava</td><td>Generic healthy state</td><td>105</td><td>22</td></tr><tr><td>guava_red_rust</td><td>guava</td><td>Specific named disease</td><td>63</td><td>13</td></tr><tr><td>hibiscus_healthy</td><td>hibiscus</td><td>Generic healthy state</td><td>58</td><td>13</td></tr><tr><td>loquat_healthy</td><td>loquat</td><td>Generic healthy state</td><td>334</td><td>71</td></tr><tr><td>loquat_leaf_spot</td><td>loquat</td><td>Specific named disease</td><td>437</td><td>93</td></tr><tr><td>mango_bacterial_canker</td><td>mango</td><td>Specific named disease</td><td>273</td><td>59</td></tr><tr><td>mango_disease_fruit</td><td>mango</td><td>Broad condition category</td><td>143</td><td>30</td></tr><tr><td>mango_healthy</td><td>mango</td><td>Generic healthy state</td><td>58</td><td>12</td></tr><tr><td>mango_healthy_fruit</td><td>mango</td><td>Generic healthy state</td><td>227</td><td>49</td></tr><tr><td>mango_healthy_leaf</td><td>mango</td><td>Generic healthy state</td><td>212</td><td>46</td></tr><tr><td>money_plant_healthy</td><td>money_plant</td><td>Generic healthy state</td><td>63</td><td>14</td></tr><tr><td>nutrient_deficiency</td><td>unspecified</td><td>Broad condition category</td><td>83</td><td>18</td></tr><tr><td>orange_citrus_greening</td><td>orange</td><td>Specific named disease</td><td>2,885</td><td>618</td></tr><tr><td>papaya_fruit_disease</td><td>papaya</td><td>Broad condition category</td><td>73</td><td>16</td></tr><tr><td>papaya_fruit_healthy</td><td>papaya</td><td>Generic healthy state</td><td>96</td><td>20</td></tr><tr><td>papaya_healthy_leaf</td><td>papaya</td><td>Generic healthy state</td><td>208</td><td>45</td></tr><tr><td>papaya_ring_spot</td><td>papaya</td><td>Specific named disease</td><td>471</td><td>101</td></tr><tr><td>peach_bacterial_spot</td><td>peach</td><td>Specific named disease</td><td>1,285</td><td>276</td></tr><tr><td>peach_healthy</td><td>peach</td><td>Generic healthy state</td><td>264</td><td>57</td></tr><tr><td>pear_black_spot</td><td>pear</td><td>Specific named disease</td><td>553</td><td>119</td></tr><tr><td>pear_fire_blight</td><td>pear</td><td>Specific named disease</td><td>250</td><td>53</td></tr><tr><td>pear_healthy</td><td>pear</td><td>Generic healthy state</td><td>478</td><td>103</td></tr><tr><td>pepper_bacterial_spot</td><td>pepper</td><td>Specific named disease</td><td>586</td><td>125</td></tr><tr><td>pepper_healthy</td><td>pepper</td><td>Generic healthy state</td><td>846</td><td>181</td></tr><tr><td>persimmon_brown_spot</td><td>persimmon</td><td>Specific named disease</td><td>432</td><td>92</td></tr><tr><td>plant_healthy</td><td>unspecified</td><td>Generic healthy state</td><td>229</td><td>46</td></tr><tr><td>potato_early_blight</td><td>potato</td><td>Specific named disease</td><td>766</td><td>164</td></tr><tr><td>potato_healthy</td><td>potato</td><td>Generic healthy state</td><td>105</td><td>23</td></tr><tr><td>potato_late_blight</td><td>potato</td><td>Specific named disease</td><td>763</td><td>163</td></tr><tr><td>raspberry_healthy</td><td>raspberry</td><td>Generic healthy state</td><td>264</td><td>57</td></tr><tr><td>rice_blast</td><td>rice</td><td>Specific named disease</td><td>409</td><td>88</td></tr><tr><td>rice_brown_spot</td><td>rice</td><td>Specific named disease</td><td>425</td><td>91</td></tr><tr><td>rice_healthy</td><td>rice</td><td>Generic healthy state</td><td>420</td><td>90</td></tr><tr><td>rose_healthy</td><td>rose</td><td>Generic healthy state</td><td>20</td><td>4</td></tr><tr><td>soybean_healthy</td><td>soybean</td><td>Generic healthy state</td><td>2,750</td><td>589</td></tr><tr><td>squash_powdery_mildew</td><td>squash</td><td>Specific named disease</td><td>1,108</td><td>237</td></tr><tr><td>strawberry_healthy</td><td>strawberry</td><td>Generic healthy state</td><td>306</td><td>65</td></tr><tr><td>strawberry_leaf_scorch</td><td>strawberry</td><td>Specific named disease</td><td>618</td><td>132</td></tr><tr><td>sugarcane_bacterial_blight</td><td>sugarcane</td><td>Specific named disease</td><td>2,974</td><td>637</td></tr><tr><td>sugarcane_healthy</td><td>sugarcane</td><td>Generic healthy state</td><td>1,191</td><td>256</td></tr><tr><td>sugarcane_mosaic</td><td>sugarcane</td><td>Specific named disease</td><td>874</td><td>188</td></tr><tr><td>sugarcane_red_rot</td><td>sugarcane</td><td>Specific named disease</td><td>1,636</td><td>350</td></tr><tr><td>sugarcane_rust</td><td>sugarcane</td><td>Specific named disease</td><td>1,466</td><td>314</td></tr><tr><td>sugarcane_yellow_leaf</td><td>sugarcane</td><td>Specific named disease</td><td>1,203</td><td>258</td></tr><tr><td>tomato_bacterial_spot</td><td>tomato</td><td>Specific named disease</td><td>2,126</td><td>455</td></tr><tr><td>tomato_early_blight</td><td>tomato</td><td>Specific named disease</td><td>1,403</td><td>300</td></tr><tr><td>tomato_fusarium_wilt</td><td>tomato</td><td>Specific named disease</td><td>140</td><td>30</td></tr><tr><td>tomato_healthy</td><td>tomato</td><td>Generic healthy state</td><td>1,941</td><td>416</td></tr><tr><td>tomato_late_blight</td><td>tomato</td><td>Specific named disease</td><td>1,854</td><td>397</td></tr><tr><td>tomato_leaf_curl</td><td>tomato</td><td>Specific named disease</td><td>553</td><td>119</td></tr><tr><td>tomato_leaf_miner</td><td>tomato</td><td>Specific named disease†</td><td>606</td><td>130</td></tr><tr><td>tomato_leaf_mold</td><td>tomato</td><td>Specific named disease</td><td>1,351</td><td>290</td></tr><tr><td>tomato_mosaic_virus</td><td>tomato</td><td>Specific named disease</td><td>240</td><td>51</td></tr><tr><td>tomato_septoria_leaf_spot</td><td>tomato</td><td>Specific named disease</td><td>1,710</td><td>367</td></tr><tr><td>tomato_spider_mites</td><td>tomato</td><td>Specific named disease†</td><td>1,281</td><td>275</td></tr><tr><td>tomato_target_spot</td><td>tomato</td><td>Specific named disease</td><td>982</td><td>211</td></tr><tr><td>tomato_verticillium_wilt</td><td>tomato</td><td>Specific named disease</td><td>170</td><td>37</td></tr><tr><td>tomato_yellow_leaf_curl</td><td>tomato</td><td>Specific named disease</td><td>2,840</td><td>608</td></tr><tr><td>walnut_anthracnose</td><td>walnut</td><td>Specific named disease</td><td>436</td><td>93</td></tr><tr><td>walnut_blotch</td><td>walnut</td><td>Specific named disease</td><td>852</td><td>183</td></tr><tr><td>walnut_gall_mite</td><td>walnut</td><td>Specific named disease†</td><td>269</td><td>58</td></tr><tr><td>walnut_healthy</td><td>walnut</td><td>Generic healthy state</td><td>413</td><td>88</td></tr><tr><td>walnut_shot_hole</td><td>walnut</td><td>Specific named disease</td><td>623</td><td>133</td></tr><tr><td>wheat_healthy</td><td>wheat</td><td>Generic healthy state</td><td>623</td><td>133</td></tr><tr><td>wheat_leaf_rust</td><td>wheat</td><td>Specific named disease</td><td>395</td><td>85</td></tr><tr><td>wheat_stripe_rust</td><td>wheat</td><td>Specific named disease</td><td>241</td><td>51</td></tr></table>

Note. Crop/species is inferred from the literal label prefix; nutrient\_deficiency and plant\_healthy are crop-unspecified. Label types follow the paper’s three-way naming logic: labels containing healthy are generic healthy states; generic fruit/leaf disease, fungal disease/leaf, and nutrient-deficiency labels are broad condition categories; the remaining non-healthy labels are assigned to the specific named disease category. grape\_mites, tomato\_leaf\_miner, tomato\_spider\_mites, and walnut\_gall\_mite are named pest or arthropod conditions and are flagged because the requested three-way scheme has no separate pest category. rice\_neck\_blast is not one of the 120 operational classes; it was a one-image candidate category quarantined before the final freeze.

## References

[1] David P. Hughes and Marcel Salathé. An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060, 2015.

[2] Sharada P. Mohanty, David P. Hughes, and Marcel Salathé. Using deep learning for image-based plant disease detection. Frontiers in Plant Science, 7:1419, 2016. doi: 10.3389/fpls.2016.01419.

[3] Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra. Plantdoc: A dataset for visual plant disease detection. In Proceedings ofthe 7th ACM IKDD CoDS and 25th COMAD, pages 249–253, 2020. doi: 10.1145/3371158.3371196.

[4] Tianqi Wei, Zhi Chen, Zi Huang, and Xin Yu. Benchmarking in-the-wild multimodal disease recognition and a versatile baseline. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pages 1593–1601, 2024. doi: 10.1145/3664647.3680599.

[5] Tianqi Wei, Zhi Chen, Zi Huang, and Xin Yu. Plantwild\_v2 dataset. https://tqwei05.github.io/PlantWild/access\_v2, 2025. Expert-refined 115-class release; accessed August 2026.

[6] Tianqi Wei, Zhi Chen, Xin Yu, Scott Chapman, Paul Melloy, and Zi Huang. A large-scale in-the-wild dataset for plant disease segmentation. Scientific Data, 13:205, 2026. doi: 10.1038/s41597-025-06513-4.

[7] Abel Yu Hao Chai, Kelly Li Zhen Jee, Sue Han Lee, Fei Siang Tay, Jules Vandeputte, Hervé Goëau, Pierre Bonnet, and Alexis Joly. Deep-plant-disease dataset is all you need for plant disease identification. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 12578–12584, 2025. doi: 10.1145/3746027.3758192.

[8] Björn Barz and Joachim Denzler. Do we train on test data? purging cifar of near-duplicates. Journal ofImaging, 6(6):41, 2020. doi: 10.3390/jimaging6060041.

[9] Curtis G. Northcutt, Anish Athalye, and Jonas Mueller. Pervasive label errors in test sets destabilize machine learning benchmarks. In NeurIPS Datasets and Benchmarks Track, 2021.

[10] Fabian Gröger, Simone Lionetti, Philippe Gottfrois, Alvaro Gonzalez-Jimenez, Ludovic Amruthalingam, Matthew Groh, Alexander A. Navarini, and Marc Pouly. Intrinsic self-supervision for data quality audits. Advances in Neural Information Processing Systems, 37, 2024.

[11] Alex J. DeGrave, Joseph D. Janizek, and Su-In Lee. Ai for radiographic covid-19 detection selects shortcuts over signal. Nature Machine Intelligence, 3:610–619, 2021. doi: 10.1038/s42256-021-00338-7.

[12] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets. Communications ofthe ACM, 64(12):86–92, 2021. doi: 10.1145/3458723.

[13] Margaret Mitchell, Simone Wu, Andrew Zaldivar, Parker Barnes, Lucy Vasserman, Ben Hutchinson, Elena Spitzer, Inioluwa Deborah Raji, and Timnit Gebru. Model cards for model reporting. In Proceedings ofthe Conference on Fairness, Accountability, and Transparency, pages 220–229, 2019. doi: 10.1145/3287560.3287596.

[14] Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[15] Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[16] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11976–11986, 2022. doi: 10.1109/CVPR52 688.2022.01167.

[17] Yin Cui, Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie. Class-balanced loss based on efective number of samples. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9268–9277, 2019. doi: 10.1109/CVPR.2019. 00949.

[18] Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. Long-tail learning via logit adjustment. In International Conference on Learning Representations, 2021.

[19] Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network. In NIPS Deep Learning and Representation Learning Workshop, 2015.

[20] Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation: A survey. International Journal of Computer Vision, 129:1789–1819, 2021. doi: 10.1007/s11263-021-01453-z.

[21] Ilyass Moummad, Reda Bensaid, Kawtar Zaher, Hervé Goëau, Jean-Christophe Lombardo, Joseph Salmon, Pierre Bonnet, and Alexis Joly. Energy-eficient plant monitoring via knowledge distillation. arXiv preprint arXiv:2604.27178, 2026.

[22] Danfeng Qin, Chas H. Leichner, Manolis Delakis, et al. Mobilenetv4: Universal models for the mobile ecosystem. In European Conference on Computer Vision, pages 78–96, 2024. doi: 10.1007/978-3-031-73661-2\_5.

[23] Anand Kumar, Harminder Pal Monga, Tapasi Brahma, Satyam Kalra, and Navas Sherif. Mobile-friendly deep learning for plant disease detection: A lightweight cnn benchmark across 101 classes of 33 crops. arXiv preprint arXiv:2508.10817, 2025.

[24] BenoitJacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko Quantization and training of neural networks for eficient integer-arithmetic-only inference. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 2704–2713, 2018. doi: 10.1109/CVPR.2018.00286.

[25] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70, pages 1321–1330, 2017.

[26] PyTorch ExecuTorch. Model export and lowering. Oficial documentation, 2026. Accessed August 2026.

[27] PyTorch. Executorch xnnpack quantization documentation. Oficial documentation, 2026. Accessed August 2026.

[28] Quinn McNemar. Note on the sampling error of the diference between correlated proportions or percentages. Psychometrika, 12(2): 153–157, 1947. doi: 10.1007/BF02295996.