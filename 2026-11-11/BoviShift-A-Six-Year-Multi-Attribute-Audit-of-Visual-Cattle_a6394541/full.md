# BoviShift: A Six-Year Multi-Attribute Audit of Visual Cattle Phenotyping Across Animals and Time

Mobasshir Bhuiya Shagor Graaho Technologies LLC Dhaka, Bangladesh mobasshir.bhuiya@graaho.com

Nilambar Halder Tonmoy Indiana University Bloomington Bloomington, IN, USA nhaldert@iu.edu

Md Saef Ullah Miah American International University Bangladesh Dhaka, Bangladesh saef@aiub.edu

## Abstract

Computer vision ofers a scalable, non-contact means of estimating livestock traits, but correlated images and video frames make reliable evaluation dificult. Imagerandom splitting can place views of the same animal in both training and test sets, while deployment further requires transfer to animals captured in future years. We introduce BoviShift, a corpus reconstructed from six consecutive annual cohorts of an operational cattle marketplace collected from 2021 to 2026. It contains 2,657 cattle, 10,610 still images, and 2,566 walk-around videos across nine recorded breed categories. We eval uate three splitting strategies through four configurations: image-random and pooled cattle-disjoint splits over the same development cohorts, followed by separate future-2025 and locked future-2026 evaluations. Using a fixed ResNet-50 multi-task baseline, we assess weight, age, height, breed, and coat color. Diferences between image-random and cattle-disjoint splitting are task-dependent rather than a universal leakage penalty. Future-cohort behavior is also attribute- and cohort specific: breed declines relative to pooled evaluation; mean weight and age performance fails to beat constant baselines in 2025; height remains above baseline; and color transfers well to 2025 but degrades in 2026. Multi-task training improves all five point estimates in a single-seed ablation, still-view aggregation helps selectively, and zero-shot video-frame aggregation exposes a still-to-video domain gap. To our knowledge, BoviShift is the longest-spanning cattle-vision resource combining consecutive annual cohorts with physical and visual attributes. Upon acceptance, we will release the corpus with its 79,390 sampled video frames, alongside repro ducible metadata reconstruction, data-quality audits, leakage-checked split manifests, and evaluation code for realistic, repeatable visual livestock phenotyping.

## 1 Introduction

Estimating livestock traits from images could reduce reliance on manual measurement and specialized equipment. Weighing cattle conventionally requires a crush, scale, and trained handler, whereas photographs can be acquired with a phone. This distinction matters where weighing infrastructure is unavailable and sale decisions depend on visual judgment. Recent methods report promising errors for image-based weight estimation [1, 7, 12].

Their interpretation, however, depends on whether observations of the same animal can occur in both training and test sets.

Livestock datasets rarely contain only one frame per animal. A marketplace listing may include four still photographs and a walk-around video captured in the same session and setting. An image-level partition can therefore place views of the same animal on both sides of the split.

The model is then evaluated on new observations, but often of animals already represented during training. This identity overlap is a common source of overoptimistic evaluation in applied machine learning [10].

It may allow a model to exploit animal- or capturespecific cues rather than generalizable phenotype evidence.

A deployed phenotyping system must also process future animals after population and acquisition conditions have changed. Benchmarks assembled during one collection campaign cannot directly test this setting because their partitions share a common acquisition period. Although temporal distribution shift is well studied elsewhere [11, 14], it has received little systematic study across multi-year livestock-vision cohorts.

We study both problems using six consecutive annual cohorts of a commercial cattle marketplace collected from 2021 to 2026. BoviShift contains 2,657 cattle, 10,610 still images, 2,566 walk-around videos, and 79,390 sampled frames across nine recorded breed categories, with no cattle identifier occurring in more than one cohort.

The design is therefore repeated cross-sectional rather than longitudinal: annual diferences may combine changes in population, acquisition, and annotation rather than within-animal growth.

We evaluate three splitting strategies through four configurations. P1 randomly partitions images from the 2021–2024 development cohorts, whereas P2 partitions the same cohorts by cattle identity. P3 trains on 2021– 2023, validates on 2024, and tests on 2025. The locked P4 run trains through 2024, validates on 2025, and tests once on 2026. Because partition units, test identities, and cohort composition change across configurations, their gaps are operational diferences rather than causal efects of identity leakage or elapsed time.

Using a fixed ResNet-50 model family, we evaluate weight, age, height, breed, and coat color.

Future-cohort performance is attribute- and cohortdependent: breed declines; the mean weight and age results fail to beat constant baselines in 2025; height remains above baseline; and color transfers well to 2025 but declines in 2026. We also compare single- and multi-task training, still-view aggregation, and zeroshot aggregation of sparse video frames.

Our contributions are:

1. BoviShift, a six-year corpus of 2,657 cattle with multi-view stills, walk-around videos, and five physical and visual attributes, released in full on acceptance.

2. A four-configuration evaluation framework comparing image-random, pooled cattle-disjoint, and two future-cohort settings using leakage-checked manifests, cattle-level metrics, and simple baselines.

3. A reproducible integrity audit that repairs 9,295 media references and identifies size-dependent missing weight and price labels in 40% of the 2024 cohort.

## 2 Background and Related Work

## 2.1 Visual estimation of livestock traits

Body-weight estimation from imagery has a long lineage in precision livestock farming, and the best-performing systems are consistently those with the most controlled capture. Point-cloud approaches using depth sensors report mean absolute percentage errors near 3% on herds of around one hundred animals [7], and depthimage pipelines for dairy calves reach comparable accuracy [12]. More recently, lightweight networks have brought the task to commodity hardware: one such system reports accuracy above 97% for weight estimation from a mobile phone [1].

These are real results, but the evaluation base is narrow. The mobile-phone system above is validated on 108 animals of a single breed; the point-cloud work on roughly 100. Neither the efect of breed heterogeneity nor the efect of elapsed time between training and deployment can be assessed at that scale, because the variation simply is not present in the data. Whether a 97% figure survives a change of breed, season or year is, at present, an open question rather than a settled one. Part of our purpose here is to make it answerable.

## 2.2 Cattle image datasets

Public cattle datasets have grown substantially, but they have grown along the identity axis rather than the phenotype axis. BECA [15] is the largest recent release, with 16,889 images of 5,661 beef cattle assembled for long-term individual recognition; its long-term subset follows 103 animals for up to five months, a span its authors describe as unusually long for the field. MOO [5] contributes multi-view observations for viewpoint analysis in re-identification. Both are strong resources for recognizing which animal is in frame. Neither carries body-weight, height or price labels, and neither spans multiple years.

On the phenotype side, CID [13] released 17,899 images of 513 cattle with weight, height and breed annotations for joint regression and classification, drawn from a single collection campaign. A recent survey of public computer vision datasets for precision livestock farming [2] catalogues the wider landscape and reaches a similar conclusion: datasets are either large and identity-focused, or trait-annotated and small.

BoviShift sits in the gap. It is trait-annotated, comparable in image count to the identity datasets, released in full on acceptance, and — unlike any of the above — organized into six consecutive annual cohorts drawn from one continuously operating marketplace.

## 2.3 Reconstruction-based approaches

A parallel line of work estimates traits by first recovering geometry. Agreement-driven fusion of multiview segmentations with a 3D foundation model reports R<sup>2</sup> = 0.69 and MAPE of 2.22% for live cattle weight [3], and monocular mesh recovery has been applied to body measurement in goats [9]. The recent SAM 3D Animal [8] makes promptable animal reconstruction in the wild practical, building on a SMAL-family parametric model.

Dulal et al. [3] report that the bottleneck for multiview 3D reconstruction is data: producing 3D-suitable capture at scale on working farms is dificult. BoviShift contributes 2,566 walk-around videos collected during routine marketplace operations using a simple handheld acquisition process. Although these are not calibrated

3D captures, their scale and viewpoint variation provide a useful resource for developing and evaluating reconstruction methods under less controlled conditions. The SAM 3D Animal model of Hu et al. [8] builds on the SMAL family, which recovers shape only up to an unknown scale and was fitted predominantly to taurine and general quadruped morphology. Whether it faithfully represents the thoracic hump and dewlap of Bos indicus cattle is not established, and a metric weight estimate cannot be obtained without an explicit calibration step. We therefore treat reconstruction as a direction the corpus enables rather than a baseline we adopt uncritically.

## 2.4 Evaluation protocols, leakage and temporal shift

The methodological framing of this paper draws on work outside livestock vision. WILDS [11] assembled distribution shifts that arise in deployment rather than in construction, and Wild-Time [14] extended this to shift over time, reporting substantial degradation between in-distribution and temporally outof-distribution evaluation across domains including healthcare and remote sensing. Neither includes an agricultural or livestock benchmark.

On the leakage side, Kapoor and Narayanan [10] document how train–test contamination produces irreproducible results across a range of applied fields, and subject-level splitting is standard practice in medical imaging for precisely this reason. Livestock vision has not generally adopted the convention, in part because many datasets contain one image per animal and the question does not arise. Once multi-view stills and video frames are introduced, it does.

What is missing in both literatures is a setting where the two efects can be measured on the same data, with the same models, at the same time. That is what the cohort structure of BoviShift permits.

## 3 The BoviShift Corpus

## 3.1 Source and collection

BoviShift extends CID [13] with five further annual cohorts. The 2021 cohort corresponds to the collection released as CID, recovered here as 514 animals after re-indexing; images, videos, and cattle properties for the remaining cohorts were sourced directly from the Bengal Meat marketplace. Each listing represents one animal and typically includes seller-recorded attributes, four still photographs captured against a prepared backdrop, and a handheld walk-around video. The six annual cohorts span 2021–2026, and the marketplace operator granted written permission for redistribution (Section 3.8).

The corpus represents a narrow production setting: 2,646 of 2,657 animals are intact male beef-finishing bulls, with six heifers and five steers. Accordingly, results should not be extrapolated to dairy cattle, breed ing stock, or other regions without additional validation. Recorded health fields are above 99.8% positive and are not used as targets. No cattle identifier recurs across cohorts. BoviShift is therefore treated as repeated cross-sectional rather than longitudinal: annual diferences combine changes in sampled population, acquisition, and annotation, not within-animal growth.

## 3.2 Media inventory

Table 1 summarizes the corpus. Coverage is close to uniform: 2,651 of 2,657 animals have exactly four still images, and the remaining six have one. Video coverage is 96.6% overall but uneven by year. In 2025, 288 of 344 cattle have an archived video, leaving 56 without one. After still-image eligibility filtering, the P3 test set contains 342 cattle, of which 288 have video; this is the 54-cattle shortfall reported in the video experiment. Video results are therefore restricted to the covered subset, with its sample size stated.

Frame extraction. The released video frames are sampled at a fixed count of 31 frames per video with random timing, spanning the full duration of each video. Frames are stored as JPG and match the source video resolution. We remove frames flagged as corrupt or truncated during decoding, yielding 79,390 usable frames across 2,566 videos (mean 30.9 frames per video). This release-time extraction is independent of the eight-frame evaluation protocol in Section 4.4, which draws a separate, sparser set of frames at fixed relative timestamps for that specific experiment.

## 3.3 Attributes

We predict five attributes (Tab. 2). Weight and height are continuous, with ranges of 140–826 kg and 30– 65 inches after validity checks. Age is recorded in five ordered levels from 24 to 48 months and is treated as regression. It closely follows seller-recorded incisor count, so it should be interpreted as an estimated dental-age category rather than a verified birth record. Missing or invalid continuous targets are masked per task.

Breed is highly imbalanced: LOCAL accounts for 80.1% of labeled cattle, while PABNA, MIR KADIM, DHANNI, and BRAHMA contain six or fewer each. Categories with fewer than 20 training cattle are pooled into Other, yielding six evaluation classes, and performance is reported with macro-F1. Color is mapped to Red versus Non-Red; the three unsupported Cross-Red records are masked. Its relative visual simplicity makes color a useful control for acquisition and cohort efects.

<table><tr><td></td><td>2021</td><td>2022</td><td>2023</td><td>2024</td><td>2025</td><td>2026</td><td>Total</td></tr><tr><td>Media</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cattle</td><td>514</td><td>629</td><td>399</td><td>385</td><td>344</td><td>386</td><td>2,657</td></tr><tr><td>Still images</td><td>2,056</td><td>2,516</td><td>1,596</td><td>1,537</td><td>1,370</td><td>1,535</td><td>10,610</td></tr><tr><td>Walk-around videos</td><td>512</td><td>621</td><td>378</td><td>384</td><td>288</td><td>383</td><td>2,566</td></tr><tr><td>Sampled video frames</td><td>15,843</td><td>19,155</td><td>11,687</td><td>11,904</td><td>8,928</td><td>11,873</td><td>79,390</td></tr><tr><td>Animals without video</td><td>2</td><td>8</td><td>21</td><td>1</td><td>56</td><td>3</td><td>91</td></tr><tr><td>Labels</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Usable weight labels</td><td>510</td><td>629</td><td>399</td><td>231</td><td>344</td><td>385</td><td>2,498</td></tr><tr><td>Weight µ (kg)</td><td>256.4</td><td>271.7</td><td>294.5</td><td>254.4†</td><td>270.2</td><td>288.0</td><td>272.9</td></tr><tr><td>Weight σ (kg)</td><td>74.3</td><td>81.2</td><td>107.4</td><td>83.4</td><td>92.1</td><td>94.5</td><td>89.4</td></tr><tr><td>Height µ (in)</td><td>46.7</td><td>42.4</td><td>43.0</td><td>42.6</td><td>42.7</td><td>43.3</td><td>43.5</td></tr><tr><td>Median price (BDT k)</td><td>101</td><td>139</td><td>160</td><td>152</td><td>156</td><td>161</td><td></td></tr><tr><td>Composition (%)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LOCAL</td><td>68.9</td><td>81.2</td><td>78.2</td><td>93.0</td><td>87.5</td><td>75.8</td><td>80.1</td></tr><tr><td>SAHIWAL</td><td>11.5</td><td>10.5</td><td>14.0</td><td>2.1</td><td>1.5</td><td>13.5</td><td>9.3</td></tr><tr><td>SINDHI</td><td>7.4</td><td>1.3</td><td>0.3</td><td>0.0</td><td>0.0</td><td>1.0</td><td>1.9</td></tr><tr><td>RED coat color</td><td>59.9</td><td>59.0</td><td>53.6</td><td>56.1</td><td>50.9</td><td>49.2</td><td>55.5</td></tr></table>

Table 1: The corpus by cohort. Media totals cover 92,566 objects and 37.3 GB. Label statistics exclude records flagged as missing (Sec. 3.5). <sup>†</sup>2024 weight is biased downward by non-random missingness. <sup>‡</sup>2021 height reflects a measurement-convention change and is not comparable with later cohorts.

<table><tr><td>Attribute</td><td>Target</td><td>Output</td><td>Metric</td></tr><tr><td>Weight</td><td>continuous</td><td>kg</td><td>MAE</td></tr><tr><td>Age</td><td>ordered numeric</td><td>months</td><td>MAE</td></tr><tr><td>Height</td><td>continuous</td><td>inches</td><td>MAE</td></tr><tr><td>Breed</td><td>categorical</td><td>6 classes</td><td>macro-F1</td></tr><tr><td>Color</td><td>categorical</td><td>2 classes</td><td>macro-F1</td></tr></table>

Table 2: Benchmark targets and primary metrics. Regression mean and classification majority baselines are fit separately on each training split.

## 3.4 Metadata reconstruction and integrity audit

The corpus as originally archived was not usable as a benchmark, and repairing it was a substantial part of this work. We report the process because the resulting manifests are part of what we release, and because the failure modes are likely to recur in any dataset assembled incrementally over several years.

Four defect classes were resolved. Space normalization repaired 2,263 video and 7,032 image references, and 64,131 of 92,566 objects were brought to one path convention. The missing 2026 index was reconstructed for 386 cattle. Six misnamed videos were reconciled using source identifiers and hashes, one byte-identical video was deduplicated, and one conflicting metadata row was removed. The audit also found 30 still-image rows in 13 exact-hash groups; these files are documented, and any hash associated with multiple cattle IDs is excluded before benchmark splitting.

The final media index contains 92,566 paths joined to annual metadata by normalized cattle identifier, with no unmatched image records. Checksums and byte sizes permit verification after re-hosting; path occurrence should not be confused with unique file content because the duplicate-hash audit is retained.

<table><tr><td>Stratum</td><td>Labeled</td><td>Missing</td><td>Difference</td></tr><tr><td>size = MINIMUM</td><td>88.2%</td><td>72.1%</td><td>-16.1 pt</td></tr><tr><td>size = MEDIUM</td><td>8.7%</td><td>23.4%</td><td>+14.7 pt</td></tr><tr><td>size = EXTRA_LARGE</td><td>0.4%</td><td>3.2%</td><td>+2.8 pt</td></tr><tr><td>mean height (in)</td><td>42.1</td><td>43.1</td><td>+1.0 in</td></tr><tr><td>n</td><td>231</td><td>154</td><td>385 total</td></tr></table>

Table 3: Composition of the 2024 cohort split by label availability. Labeled: nonzero weight and price. Missing: weight and price recorded as zero. Diference: Missing minus Labeled, in percentage points (pt) for the size shares and inches (in) for mean height. The MIN-IMUM share is 16.1 pt lower among the unlabeled animals, while MEDIUM and EXTRA\_LARGE are 14.7 and 2.8 pt higher, and mean height is 1.0 in greater. Missingness is therefore size-dependent, with the unlabeled animals systematically larger. Because larger animals weigh more, the 2024 mean weight computed over the labeled subset alone (Tab. 1) is biased downward.

## 3.5 Missingness and the 2024 cohort

Among the 2,657 deduplicated cattle, 159 non-positive weight and price entries are treated as missing rather than imputed; 154 occur in 2024 and afect 40.0% of that cohort. Table 3 shows size-dependent missingness: unlabeled cattle contain fewer Minimum and more Medium or Extra-Large records, and are one inch taller on average. The observed 2024 mean weight of 254.4 kg is therefore biased downward.

Invalid labels contribute neither loss nor metrics. Accordingly, we do not select 2024 as a future evaluation cohort, and its observed weight is not interpreted as evidence of temporal shift. It serves as P3 validation data and enters P4 training with missing targets masked. P3 contains 1,540 eligible training cattle from 2021–2023, of which 1,538 have valid weight labels. The manifests also mask the implausible 4.6-inch height record and document three cattle with incomplete media and obsolete source URLs.

## 3.6 Cohort composition drift

The shift the corpus is built to measure is present in the labels before any model is trained, and the lower panels of Tab. 1 make it concrete.

Breed composition moves substantially. The LOCAL share ranges from 68.9% to 93.0%; SAHIWAL falls from 14.0% in 2023 to 1.5% in 2025 before returning to 13.5% in 2026; SINDHI declines from 7.4% in 2021 to complete absence in 2024 and 2025. Four of the nine breed categories appear in only one or two cohorts. A future-year protocol on this corpus therefore encounters class disappearance and reappearance, not merely reweighted frequencies. The Red share declines overall from 59.9% to 49.2%, with a temporary increase in 2024.

Prices rise, as expected in a nominal currency over a six-year window: the median listing moves from 101,000 to 161,000, a nominal increase of roughly 59%. This matters for any economically labeled target, and it is the reason we do not treat price as a phenotype in this study. Price is nonetheless released with the corpus, since it supports questions this paper does not attempt.

Two discontinuities indicate acquisition change. Mean recorded height falls from 46.7 inches in 2021 to 42.4 inches in 2022 and remains between 42.4 and 43.3 thereafter, suggesting a change in measurement convention; 2021 height is therefore flagged as not directly comparable. The feed schema also changes after 2022 from an ingredient list to one categorical type. Both changes are documented in the datasheet and exemplify the operational drift the temporal protocols are designed to expose.

## 3.7 Benchmark organization

We instantiate three splitting strategies through four configurations (Tab. 4). P1 and P2 use the 2021–2024 development cohorts with image-random and cattledisjoint partitions, respectively. P3 trains on 2021– 2023, validates on 2024, and tests on 2025; locked P4 trains through 2024, validates on 2025, and tests once on 2026. P2–P4 are cattle-disjoint. Explicit manifests record cattle IDs and content hashes, and the verification script rejects cross-partition identity or exact-file overlap. Protocol contrasts are descriptive because partition units and test populations change together.

## 3.8 Ethics, licensing and availability

Ethics. The corpus derives from commercial listings published by the marketplace operator – Bengal Meat. Seller and buyer identifiers are not retained, and the benchmark contains no associated personal records. No animal was handled, restrained, or otherwise afected for this research; the marketplace produced the imagery in the ordinary course of trade.

Permission and license. Redistribution rights for the imagery, video, derived frames, and attribute metadata were granted in writing by the marketplace operator for research use. The specific license terms, including any media-specific restrictions stated separately for stills, videos, and derived frames, will accompany the release. The operator is acknowledged in the camera-ready version in accordance with the terms of that agreement.

What is released. The release package comprises 10,610 still images, 2,566 walk-around videos, and 79,390 sampled frames (extraction protocol detailed in Section 3.2; 92,566 objects, 37.3 GB); attributes for 2,657 cattle; the checksummed media index; audit and exclusion records; manifests for all four configurations; leakage verification; and baseline training and evaluation code. Upon acceptance, it will be deposited in a persistent public archive with a DOI and no registration requirement. A datasheet following Gebru et al. [4] documents collection, composition, preprocessing, recommended uses, and limitations.

Reproducibility. The reported dataset summaries and evaluations can be regenerated from the release without contacting the authors. Split manifests contain explicit cattle identifiers rather than only a random seed, allowing future implementations to reuse the same partitions and verify identity overlap directly.

Intended use and limitations. Automated appraisal can shift bargaining power between buyers and sellers, and subgroup error should be audited before deployment. This study does not validate a commercial valuation system. The corpus covers one marketplace, one country, and one narrow production category; its intended use is research on visual phenotyping, evaluation methodology, and distribution shift. Deployment in other populations requires local validation.

## 4 Evaluation Protocol

We examine how image-random splitting afects measured performance, how cattle-disjoint models transfer to future cohorts, and whether multi-task and multiobservation inference improve that transfer. We hold the model family fixed, but interpret protocol diferences descriptively because partition unit and testcohort composition change together.

## 4.1 Leakage and temporal protocols

Table 4 defines four configurations. P1 randomly assigns images from 2021–2024; consequently, 899 of 913 test cattle also occur in training. P2 uses the same cohorts and approximately $7 0 / 1 5 / 1 5$ proportions but partitions by cattle ID. P3 trains on 2021–2023, validates on 2024, and tests on 2025; the locked P4 run trains on 2021–2024, validates on 2025, and tests once on 2026. P2–P4 have no cattle overlap between partitions. Before splitting, we remove 30 images whose content hash occurs under multiple cattle IDs; no remaining file hash crosses a partition. This leaves 10,580 eligible images, with 2,647 of 2,657 cattle represented. All manifests and preprocessing rules are fixed before training.

Breeds represented by fewer than 20 training cattle are mapped to Other, yielding six common evaluation classes from the nine recorded categories. Color is evaluated as Red versus Non-Red; weight, age, and height retain their recorded units. Unsupported categorical and invalid continuous labels are masked. Label maps and normalization statistics are fit on training data only.

## 4.2 Model and optimization

Our reference model uses an ImageNet-pretrained ResNet-50 encoder [6] followed by dropout $( p = 0 . 2 )$ ， one linear classification head for each of breed and color, and a joint linear head for the three continuous attributes. Regression targets are standardized using the training-set mean and standard deviation. The multitask objective is the equally weighted sum of (i) the mean inverse-frequency-weighted cross-entropy over the two classification tasks, with weights computed from training image rows, and (ii) Smooth- $. L _ { 1 }$ loss over all valid standardized regression targets. Thus, the classification and regression components each have coeficient one; no test-set metric is used to tune their relative weighting.

More precisely, for classification tasks ${ \mathcal { C } } ,$ regression tasks $\mathcal { R } _ { : }$ , standardized target $z _ { i r }$ , and validity mask $m _ { i r } .$ we optimize

$$
\begin{array} { r l r } {  { \mathcal { L } = \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \mathrm { C E } _ { w _ { c } } ( y _ { c } , \hat { y } _ { c } ) } } \\ & { } & { \quad + \frac { \sum _ { i , r } m _ { i r } \mathrm { ~ S m o o t h L 1 } ( z _ { i r } , \hat { z } _ { i r } ) } { \sum _ { i , r } m _ { i r } } . } \end{array}\tag{1}
$$

For residual $e = \hat { z } - z$ , Smooth-L<sub>1</sub> with $\beta = 1$ uses $e ^ { 2 } / 2$ when $| e | < 1$ and $| e | - 1 / 2$ otherwise. It is quadratic for small errors but less sensitive than squared error to large outliers. Because targets are standardized, the transition occurs at one training-set standard deviation. The mask ensures that an unavailable continuous target contributes neither loss nor a regression metric; unsupported categorical targets are ignored by cross-entropy.

The shorter image side is resized to 256 pixels for $\mathrm { ~ a ~ } 2 2 4 \times 2 2 4$ input. Training uses a random resized crop (scale 0.75–1.0, aspect ratio 0.75–1.33), horizontal flipping with probability 0.5, and color jitter with brightness, contrast, saturation, and hue magnitudes of 0.15, 0.15, 0.10, and 0.02; evaluation uses a center crop. We optimize all parameters for 30 epochs with AdamW, batch size 64, initial learning rate $3 \times 1 0 ^ { - 4 }$ weight decay $1 0 ^ { - 4 }$ , gradient-norm clipping at 1.0, cosine learning-rate decay, and bfloat16 mixed precision. The checkpoint with minimum total validation loss is selected without inspecting the test set. P1–P3 are run with seeds 42, 123, and 2026. P4 uses the frozen configuration and seed 42 as a single pre-designated held-out run. The three P1–P3 seeds vary model initialization and training order, not the fixed split manifest.

## 4.3 Metrics and uncertainty

Breed and color are evaluated with macro-F1; weight, age, and height use MAE in kilograms, months, and inches. Cattle-level predictions average class probabilities or regression outputs across available stills. We report mean and sample standard deviation across seeds where available. Conditional 95% percentile intervals use 2,000 cattle-bootstrap resamples, with paired resampling for ablations. They capture test-cohort sampling uncertainty for a fitted checkpoint, not retraining, split, or cohort-selection variation. Training-cattle majority and mean predictors provide classification and regression baselines.

P1 contains 1.26 test views per animal on average, whereas P2 keeps approximately four; the two configurations also contain diferent test identities. We therefore use image-level metrics only for the descriptive P1– P2 split-policy comparison and cattle-level metrics for P2–P4. The P1–P2 contrast is an operational diference between splitting policies, not a causal estimate of identity leakage.

## 4.4 Multi-observation and task ablations

Single- versus multi-task learning. On P3 with seed 42, we train five single-task models, one per attribute. Each uses the same encoder, split, augmentation, optimizer, epoch budget, and checkpoint rule as the multi-task model. This tests joint supervision with a shared representation for the seed-42 training configuration.

Table 4: Evaluation configurations. Each split reports images/cattle. P1 is intentionally leakage-prone; P2–P4 are cattle-disjoint.
<table><tr><td>Config.</td><td>Split rule</td><td>Train</td><td>Validation</td><td>Test</td></tr><tr><td>P1</td><td>image-random, 2021–2024</td><td>5,382/1,908</td><td>1,153/919</td><td>1,153/913</td></tr><tr><td>P2</td><td>cattle-disjoint, 2021–2024</td><td>5,381/1,347</td><td>1,156/289</td><td>1,151/288</td></tr><tr><td>P3</td><td>future 2025</td><td>6,152/1,540 (2021–2023)</td><td>1,536/384 (2024)</td><td>1,368/342 (2025)</td></tr><tr><td>P4</td><td>locked future 2026</td><td>7,688/1,924 (2021–2024)</td><td>1,368/342 (2025)</td><td>1,524/381 (2026)</td></tr></table>

Table 5: Image-level P1–P2 comparison, reported as mean ± sample standard deviation over three optimization seeds. Higher is better for F1; lower is better for MAE. MAE units are kg, months, and inches.
<table><tr><td>Config.</td><td>Br. F1 ↑</td><td>Col. F1 ↑</td><td>Wt. MAE ↓</td><td>Age MAE ↓</td><td>Ht. MAE ↓</td></tr><tr><td>P1 image-random</td><td>0.500±0.029</td><td>0.885±0.020</td><td>35.256±3.173</td><td>2.291±0.040</td><td>1.948±0.393</td></tr><tr><td>P2 cattle-disjoint</td><td>0.440±0.026</td><td>0.896±0.011</td><td>36.831±12.075</td><td>2.422±0.206</td><td>1.753±0.314</td></tr></table>

Still-view aggregation. We compare all-view cattle predictions with one randomly selected still. The single-view point estimate averages 2,000 random selections; each paired bootstrap replicate resamples cattle and draws a new view, avoiding dependence on one favorable selection.

Sparse video frames. We test whether the frozen still-image model transfers directly to the walk-around videos. In contrast to the 79,390 release frames (Section 3.2), we sample eight candidate frames at evenly spaced relative timestamps from 10% to 90% of each video. The one-frame set uses 44.3%, while the eightframe set uses all candidates. Predictions are aggregated by averaging class probabilities and regression outputs. Still–video comparisons use exactly the cattle with an available, successfully decoded video: 288/342 in P3 and 381/381 in P4. This is zero-shot still-to-video transfer; no video frame is used for training or model selection.

## 5 Results

## 5.1 Image-random versus cattledisjoint splitting

Table 5 compares P1 and P2 at the image level, avoiding the diferent number of test views in their cattle-level aggregates. Relative to P1, P2 has 0.060 lower breed macro-F1 and 1.575 kg and 0.131-month higher weight and age MAE. The direction is not universal: P2 has 0.011 higher color F1 and 0.195-inch lower height MAE. Weight also varies substantially across seeds. We therefore do not report a single “leakage penalty.” These are descriptive diferences between fixed configurations with matched years and similar data budgets, but different test identities.

Table 6: Cattle-level performance. Model rows for P2– P3 are mean ± sample standard deviation over three optimization seeds; P4 is one locked run. Base denotes the training-cattle majority/mean predictor. Weight, age, and height MAE use kg, months, and inches.
<table><tr><td>Config.</td><td>Br. F1 ↑</td><td>Col. F1 ↑</td><td>Wt. MAE ↓</td><td>Age MAE ↓</td><td>Ht. MAE ↓</td></tr><tr><td>P2 model</td><td>0.469±0.037</td><td>0.910±0.007</td><td>33.836±13.255</td><td>2.401±0.219</td><td>1.603±0.368</td></tr><tr><td>P2 base</td><td>0.145</td><td>0.372</td><td>53.222</td><td>2.750</td><td>2.603</td></tr><tr><td>P3 model</td><td>0.402±0.067</td><td>0.920±0.008</td><td>64.427±16.399</td><td>2.824±0.279</td><td>2.328±0.471</td></tr><tr><td>P3 base</td><td>0.234</td><td>0.340</td><td>61.056</td><td>2.790</td><td>3.375</td></tr><tr><td>P4 model</td><td>0.312</td><td>0.657</td><td>38.915</td><td>2.969</td><td>1.815</td></tr><tr><td>P4 base</td><td>0.144</td><td>0.329</td><td>64.536</td><td>3.020</td><td>3.437</td></tr></table>

## 5.2 Cattle-level future-cohort evaluation

Table 6 uses the animal as the evaluation unit for the cattle-disjoint and future-cohort configurations. Relative to P2, P3 is associated with 0.067 lower breed F1 and 30.591 kg, 0.423-month, and 0.725-inch higher MAE for weight, age, and height. Color remains similar (0.910 versus 0.920). This is an operational configuration gap, not a causal estimate of calendar time: P2 randomly partitions 2021–2024, whereas P3 trains on 2021–2023, validates on 2024, and tests on a diferent 2025 population.

The large weight standard deviations in P2 and P3 reveal optimization sensitivity. The P3 seed-42 checkpoint reaches 45.936 kg MAE, whereas the three-seed mean (64.427 kg) is worse than the 61.056 kg trainingmean baseline; the mean age result similarly fails to improve on its baseline. In contrast, P3 breed, color, and height and all five P4 point estimates improve on their corresponding baselines. However, the P3 height baseline is inflated by the 2021 measurement-convention change: including that cohort raises the training mean, so the margin over baseline narrows when 2021 is excluded. We retain the cohort and report the baseline explicitly rather than adjusting it. These outcomes make the seed distribution and simple baselines essential parts of the benchmark rather than optional diagnostics.

P4 is a second held-out future cohort rather than a direct repeat of P3: it adds 2024 to training, uses 2025 for validation, and tests a diferent 2026 population. Its breed and color F1 are 0.312 and 0.657; conditional 95% cattle-bootstrap intervals are [0.252, 0.375] and [0.605, 0.706]. Weight, age, and height MAE are

Table 7: P3 single-task versus multi-task learning (seed 42). The final column is single-task minus multi-task under paired cattle bootstrap. Thus, positive MAE differences and negative F1 diferences favor multi-task learning.
<table><tr><td>Attribute</td><td>Metric</td><td>Multi-task</td><td>Single-task</td><td colspan="2">Difference [95% CI]</td></tr><tr><td>Breed</td><td>macro-F1</td><td>0.421</td><td>0.362</td><td>-0.059</td><td>[-0.103, +0.074]</td></tr><tr><td>Color</td><td>macro-F1</td><td>0.915</td><td>0.904</td><td>-0.010</td><td>[−0.046, +0.027]</td></tr><tr><td>Weight</td><td>MAE (kg)</td><td>45.936</td><td>63.603</td><td></td><td>+17.667 [+15.325, +19.828]</td></tr><tr><td>Age</td><td>MAE (months)</td><td>2.516</td><td>2.691</td><td>+0.175</td><td>[+0.090, +0.257]</td></tr><tr><td>Height</td><td>MAE (inches)</td><td>1.799</td><td>1.928</td><td>+0.129</td><td>[+0.025, +0.233]</td></tr></table>

Table 8: Gain from all-view cattle aggregation over a randomly selected single still. Positive values favor aggregation. P2 and P3 show mean ± standard deviation over three seeds; P4 is one locked run. MAE units are kg, months, and inches.
<table><tr><td>Config.</td><td>Br. F1</td><td>Col. F1</td><td>Wt. MAE</td><td>Age MAE</td><td>Ht. MAE</td></tr><tr><td>P2</td><td>+0.028±0.018</td><td>+0.014±0.014</td><td>+2.994±1.977</td><td>+0.023±0.022</td><td>+0.150±0.055</td></tr><tr><td>P3</td><td>+0.039±0.035</td><td>+0.031±0.008</td><td>+1.782±1.117</td><td>+0.064±0.041</td><td>+0.119±0.069</td></tr><tr><td>P4</td><td>-0.041</td><td>+0.008</td><td>+1.539</td><td>+0.009</td><td>+0.115</td></tr></table>

38.915 kg, 2.969 months, and 1.815 inches, with conditional 95% intervals [35.560, 42.415], [2.737, 3.228], and [1.639, 1.990]. The P4 regression rebound alongside the color decline illustrates cohort-specific behavior, but the single P4 training seed precludes a claim about its optimization variance.

## 5.3 Does multi-task learning help?

Table 7 compares seed-42 models on exactly the same P3 test cattle. Multi-task learning improves every point estimate for this training seed. Conditional on the fitted seed-42 checkpoints, the paired 95% cattle-bootstrap interval excludes zero for weight, age, and height. Breed and color also favor the multi-task model, but their intervals include zero. Since only one optimization seed is available for this ablation, these intervals do not establish that the multi-task advantage persists across retraining.

## 5.4 Aggregation across still views

Table 8 reports the gain from using all available still views instead of one randomly selected view. The point estimates show positive weight, height, and color gains in P2, P3, and P4. The age gains are small, and breed improves in P2 and P3 but reverses on P4. We therefore treat multi-view aggregation as task- and cohortdependent rather than universally beneficial.

For the locked P4 checkpoint, paired cattlebootstrap intervals exclude zero only for the weight gain (+1.539 kg [0.072, 3.048]) and height gain (+0.115 inches [0.040, 0.186]). The P4 breed, color, and age intervals include zero. These intervals reflect test-cattle resampling for one trained model and do not measure retraining uncertainty.

Table 9: Zero-shot sparse-frame evaluation using frozen still-image models. Still results use the identical videoavailable cattle subset. MAE units are kg, months, and inches.
<table><tr><td>Config.</td><td>Input</td><td>Br. F1</td><td>Col. F1</td><td>Wt. MAE</td><td>Age MAE</td><td>Ht. MAE</td></tr><tr><td>P3</td><td>one video frame</td><td>0.307</td><td>0.775</td><td>52.272</td><td>2.869</td><td>3.003</td></tr><tr><td>P3</td><td>eight video frames</td><td>0.312</td><td>0.836</td><td>60.031</td><td>2.835</td><td>3.231</td></tr><tr><td>P3</td><td>all still views</td><td>0.432</td><td>0.912</td><td>43.668</td><td>2.462</td><td>1.790</td></tr><tr><td>P4</td><td>one video frame</td><td>0.250</td><td>0.680</td><td>48.158</td><td>2.867</td><td>2.354</td></tr><tr><td>P4</td><td>eight video frames</td><td>0.174</td><td>0.596</td><td>50.933</td><td>2.903</td><td>2.507</td></tr><tr><td>P4</td><td>all still views</td><td>0.312</td><td>0.657</td><td>38.915</td><td>2.969</td><td>1.815</td></tr></table>

## 5.5 Zero-shot transfer to video frames

Table 9 compares uniformly sampled video frames with curated still views on the matched cattle subset. Eightframe video aggregation is worse than all-still aggregation for every P3 metric. The paired intervals exclude zero for all five diferences conditional on the seed-42 checkpoint. In P4, video is worse for breed, weight, and height; color is worse with eight frames but better with one; the small age improvement is inconclusive. Eight frames underperform the predefined one-frame sample on all five P4 point estimates, although the paired age diference includes zero.

This result does not show that video is intrinsically less informative. Because the model is trained only on curated stills, the gap may reflect motion blur, occlusion, unfavorable poses, compression, or other acquisition diferences in uniformly sampled frames. Videospecific training and quality-aware frame selection remain open benchmark directions.

## 6 Discussion

The experiments yield three main observations. First, P1 and P2 produce task-dependent diferences rather than a universal leakage penalty. Because P1 nevertheless places the same cattle in training and test, it remains an unsuitable estimate of deployment performance. Second, the future-2025 configuration is markedly harder for breed and the physical attributes than pooled cattle-disjoint evaluation, while color behaves diferently in 2025 and 2026. Generalization is therefore both attribute- and cohort-specific. Third, the large across-seed variation in weight estimation shows that a single training run is insuficient evidence of robustness. Additional observations are also selective: curated still aggregation benefits some physical attributes, whereas naive video-frame averaging exposes a still-to-video domain gap.

These results motivate a practical reporting standard: partition by cattle identity, include a held-out future cohort, report multiple optimization seeds, aggregate and evaluate at cattle level, and obtain uncertainty by resampling cattle rather than correlated images. Image-random results may be reported as a diagnostic, but not as evidence of deployment generalization.

BoviShift also has important limitations. It is reconstructed from one operational marketplace, so annual diferences combine changes in population, capture practice, operations, and annotation rather than isolating calendar time. P1–P2 and P2–P3 consequently compare fixed experimental configurations, not causal efects, and each uses one split realization. Breed labels are imbalanced, color is binary, and marketplace metadata may contain measurement or transcription error. Our baseline uses one encoder family; P4 and the single-task ablation use one training seed, and checkpoint bootstrap intervals do not capture retraining variation. Finally, video evaluation is zero-shot, and 54 P3 test cattle lack videos. Future work should test additional model families, split realizations, and videospecific methods under the same fixed identity-disjoint temporal evaluation strategy.

## References

[1] Lili Bai, Chaopeng Guo, and Jie Song. Cattle weight estimation model through readily photos. Engineering Applications of Artificial Intelligence, 143:109976, 2025.

[2] Anil Bhujel, Yibin Wang, Yuzhen Lu, Daniel Morris, and Mukesh Dangol. A systematic survey of public computer vision datasets for precision livestock farming. Computers and Electronics in Agriculture, 229: 109718, 2025.

[3] Rabin Dulal, Wenfeng Jia, Lihong Zheng, and Jane Quinn. Agreement-driven multi-view 3D reconstruction for live cattle weight estimation. arXiv preprint arXiv:2601.17791, 2026.

[4] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets. Communications of the ACM, 64(12):86–92, 2021.

[5] William Grolleau, Achraf Chaouch, Astrid Sabourin, Guillaume Lapouge, and Catherine Achard. MOO: A multi-view oriented observations dataset for viewpoint analysis in cattle re-identification. arXiv preprint arXiv:2603.04314, 2026.

[6] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[7] Zixia Hou, Lyuwen Huang, Qi Zhang, and Yuanshuang Miao. Body weight estimation of beef cattle with 3d deep learning model: PointNet++. Computers and Electronics in Agriculture, 213:108184, 2023.

[8] Xuyi Hu, Jin Lyu, Jiuming Liu, Yebin Liu, Silvia Zufi, Liang An, and Stefan Goetz. SAM 3D Animal: Promptable animal 3D reconstruction from images in the wild. arXiv preprint arXiv:2605.07604, 2026.

[9] Bo Jin, Shichao Zhao, Jin Lyu, Bin Zhang, Tao Yu, Liang An, Yebin Liu, and Meili Wang. Monocular

mesh recovery and body measurement of female saanen goats. In Proc. AAAI Conference on Artificial Intelligence, 2026.

[10] Sayash Kapoor and Arvind Narayanan. Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 2023.

[11] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, et al. WILDS: A benchmark of in-the-wild distribution shifts. In Proc. International Conference on Machine Learning (ICML), 2021.

[12] Mingsi Liao, Gota Morota, Ye Bi, and Rebecca R. Cockrum. Predicting dairy calf body weight from depth images using deep learning (yolov8) and threshold segmentation with cross-validation and longitudinal analysis. Animals, 15(6):868, 2025.

[13] Mobasshir Bhuiya Shagor, Md Zahidul Haque Alvi, and Khandaker Tabin Hasan. CID: Cow images dataset for regression and classification. In Proc. International Conference on Computing Advancements (ICCA), 2022.

[14] Huaxiu Yao, Caroline Choi, Bochuan Cao, Yoonho Lee, Pang Wei Koh, and Chelsea Finn. Wild-time: A benchmark of in-the-wild distribution shift over time. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2022.

[15] Yuqi Zhang, Longxiang Li, Chunyang Li, Sen Wang, Yue Rong, Kai Niu, and Zhiqiang He. BECA: A computer vision dataset for long-term recognition in beef cattle. Scientific Data, 13:7, 2026.