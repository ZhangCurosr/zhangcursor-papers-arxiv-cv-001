# Frugal Handwritten Text Recognition for Sub-Saharan Civil-Status Registers: A Smartphone-to-Records Pipeline with CPU-Only Inference

<sub>2</sub>Ousmane Hamadou<sup>1</sup> 0ousmanehamadou776@gmail.com <sup>2</sup>Jean Louis Kedieng Ebongue Fendji<sup>2,3</sup> <sup>t</sup>jl.fendji@egcim-univ-ndere.cm

<sup>1</sup> Department of Computer Science Faculty of Science, University of Ngaoundéré Ngaoundéré, Cameroon

<sup>2</sup> Centre of Research, Experimentation and Production SCEMI, University of Ngaoundéré Ngaoundéré, Cameroon

<sup>3</sup> Stellenbosch Institute for Advanced Study Wallenberg Research Centre at Stellenbosch University Stellenbosch, South Africa

## Abstract

Handwritten civil-status registers across Sub-Saharan Africa are being photographed, but rarely transcribed, which leaves their content unsearchable. Local administrations that hold these records typically lack GPUs, reliable connectivity, and the budget for external digitization services. We study whether a frugal pipeline, with consumer-grade capture and CPU-only inference, can make such registers machine-readable. We introduce a corpus of 1,568 smartphone images of handwritten birth, marriage, and death records from a municipal registry in Chad, with line-level transcriptions for 185 records. Our pipeline combines a fully classical preprocessing stage, running at about 2.6 images per second on a laptop CPU, with a Kraken recognizer fine-tuned from a pretrained model in about four hours on a free-tier GPU. On manually corrected line segmentation and a single-registry test set whose writers also appear in training, fine-tuning reduces the character error rate from 14.53% to 0.50% and the word error rate from 56.93% to 3.37%. On a six-core laptop CPU, the recognizer weighs 61 MiB, peaks below 2.6 GiB of RAM, and processes an image in under 6 s on average, excluding line segmentation. Finally, a downstream information-extraction task shows that residual recognition errors cost about 9 F1 points on role resolution, compared with a 36.7-point gap between the fine-tuned and pretrained recognizers. These results are evidence of feasibility within one registry, not of performance on unseen writers or sites; we discuss these limitations in detail.

## 1 Introduction

Civil registration is the documentary basis of legal identity. Birth, marriage, and death records establish age, filiation, and nationality, and they feed the demographic statistics that public policy depends on [40, 42]. In Sub-Saharan Africa, a large share of these records exists only as handwritten registers kept in municipal offices, often in poor physical condition and exposed to humidity, fire, and loss [1, 31]. Coverage is also low: UNICEF estimates that only about half of children under five in the region have their births registered [41]. For people without recent documents, these older registers can be the only surviving proof of identity or filiation [43].

Several countries have launched large modernization programmes, such as Senegal’s NEKKAL [9], which are funded at a scale most local administrations cannot match. These efforts mainly computerize new registrations or index scanned documents at the level of whole records. The historical handwritten stock is typically photographed, when it is digitized at all, but its content remains unreadable by machines: an image without a transcription cannot be searched by name, date, or place.

Handwritten text recognition (HTR) is the obvious remedy, but two obstacles stand in the way in this setting. First, domain mismatch: publicly available HTR models are trained predominantly on European historical documents. They have never seen the handwriting styles, administrative templates, or, most importantly, the personal and place names that dominate African civil records. Second, resource constraints: most recent HTR work assumes GPU training and inference, and often cloud processing. That is incompatible with offices that have only standard laptops, intermittent connectivity, and legitimate concerns about sending identity data to external providers.

In this paper we ask whether a frugal pipeline, meaning one captured with consumer hardware, adapted with a small amount of annotated data, and run for inference entirely on a CPU, can reach transcription quality good enough to be useful. We make the following contributions:

1. A new corpus of Sub-Saharan civil-status records. We collected 1,568 smartphone images of handwritten birth, marriage, and death records from a municipal registry in Chad, using a low-cost capture protocol. The records follow five distinct administrative templates. We provide line-level transcriptions for 185 records (Section 3).

2. A classical, CPU-only preprocessing pipeline for smartphone RAW captures. It handles page detection, separation of two records captured in one image, bindingmargin removal, and upside-down image detection without any learned model. It ran on the full corpus without errors at about 2.6 images per second on a laptop CPU; the accuracy of its outputs was not measured (Section 4.1).

3. An evaluation of frugal HTR adaptation. Fine-tuning a Kraken recognizer on this small ground truth for about four hours on a free-tier GPU reduces CER from 14.53% to 0.50% on corrected line segmentation and writers seen in training. The resulting model runs on a laptop CPU with 61 MiB of weights, under 2.6 GiB of peak memory, and under 6 s per image (Sections 4.2–5.3).

4. An error-propagation analysis on a downstream information-extraction task. It quantifies how much of the end-task error comes from recognition rather than from later stages (Section 5.4).

Capture, preprocessing, and inference run on consumer hardware without a GPU; only finetuning used a free-tier cloud GPU. The stages are evaluated separately: line segmentation was corrected manually, so the pipeline is not evaluated end to end. We also report the limitations of our study openly, including the single collection site, the small test set, and the absence of inter-annotator agreement (Section 6), and we describe how we handled the personal data involved (Section 7).

## 2 Related Work

Handwritten text recognition. Handwriting recognition has long been harder than printed OCR because cursive letters join, shapes vary across writers, and character boundaries are ambiguous. This “segmentation problem” was the main obstacle for offline recognition [32]. The field moved toward segmentation-free recognition of whole text lines: recurrent networks trained with Connectionist Temporal Classification learn the alignment between image positions and output characters implicitly [17, 18]. Convolutional-recurrent networks with CTC (CRNN-CTC) remain a standard, lightweight baseline. Kraken [25] implements this design for historical documents and, together with the eScriptorium platform [26], supports the full workflow from segmentation to transcription and training. More recently, encoderdecoder Transformers such as TrOCR [27] have achieved strong results by combining a vision Transformer encoder with a pretrained language-model decoder, at a substantially higher computational cost. We use CRNN-CTC because its cost fits our CPU inference setting. Comparing it under the same resource constraints with a Transformer recognizer, with Kraken models already trained on French handwriting (e.g. from the HTR-United or CAT-MuS collections), with Transkribus, and with compact vision-language models run on a CPU without training remains future work.

Historical records and domain adaptation. Much HTR research on historical materia relies on curated ground truth from European archives [39], typically produced by a limited number of writers. Closest to our setting is work on demographic records: the ESPOS-ALLES database of historical marriage licences [36] and the associated ICDAR 2017 competition on information extraction from historical handwritten records [15]. Both address recognition and the extraction of names and roles from structured civil or parish records, which is essentially the task we study downstream. The French civil registers of births of Belfort (1807–1919) also combine pre-printed forms with handwritten fields [5]. On Dutch Curaçao death certificates (1879–1949) transcribed with a custom-trained Transkribus model, Hoek et al. found that names and professions still contain errors when the rest of the transcription appears flawless, and that whole-text quality indicators say little about the entities scholars need [20]. To our knowledge, comparable resources for Sub-Saharan African civil registers do not exist. Our registers differ from European corpora in three ways: many officers wrote them with no common calligraphic standard, they mix pre-printed forms with handwritten fields, and they are dominated by local personal and place names.

Name bias in pretrained models. Pretrained language models such as CamemBERT [28] are trained on web text in which Sub-Saharan African names are likely under-represented. Subword tokenization splits rare names into many fragments, which weakens their representation [4, 7]. Multilingual NER performance also drops sharply for under-represented African languages [2]. Our setting differs: the text is in French, and it is the names rather than the language that are out of distribution. The same concern plausibly applies to recognition models trained mostly on European material; we measure the effect of adaptation on the extraction model per entity type, but not on name spans in the recognizer.

Efficient deployment. The “Green AI” position argues that model quality should always be reported alongside computational cost [38]. This matters especially where access to compute and connectivity is limited, as in much of Sub-Saharan Africa, where cloud-based AI risks widening existing digital divides [13, 22]. Compression techniques such as posttraining quantization [23, 29], pruning [19], and distillation [37] are complementary and have been applied, for example, to deep-learning models for agricultural plants [16]; their speed-ups, however, depend on hardware and implementation. Parameter-efficient finetuning [12, 21] reduces training memory, yet QLoRA’s 4-bit kernels typically assume a CUDA-capable GPU. We therefore follow a simple recipe suited to low-resource settings: full fine-tuning of a small model, with every cost dimension reported.

Civil-registration digitization in Africa. Continental and national programmes are modernizing civil registration [3, 10], for example Senegal’s NEKKAL [9]. These efforts focus mainly on computerizing new registrations and indexing scanned documents at the record level, not on transcribing the content of historical handwritten registers. Our work addresses this gap with an approach sized for local administrations rather than national programmes.

## 3 Dataset and Capture Protocol

Source and context. The corpus comes from the civil-status archive of a municipal registry in Chad. Chad’s recent civil-registration reforms focus on secure identity documents, late registration of children, and a new biometric register [11, 33]. None of them addresses the existing stock of handwritten registers, which remains the only record of many past births, marriages, and deaths. The registry we worked with lies outside the urban areas targeted by these programmes and is representative of the many local offices where paper registers are the sole copy.

The registers were kept by a succession of civil-status officers, each with their own handwriting and no common calligraphic standard. This makes the writer population far more varied than in many reference HTR datasets. Most records are pre-printed forms, several of them bilingual French–Arabic, whose handwritten fields are filled in French. The handwritten fields are dominated by local personal and place names that rarely appear in European training corpora. The registers also show physical wear typical of such archives, including stains, tears, faded ink, and damage from humidity and handling.

Capture protocol. Collection took place on site over five consecutive days, producing 1,568 images, an average of about 314 per day. This throughput requirement shaped the protocol, which uses only consumer hardware: a Google Pixel 6a smartphone capturing in RAW (DNG) format at an image resolution of 2268×4032 pixels; a stand holding the phone parallel to the register, whose height was adjusted for each record; and a ThinkPad T460 laptop running scrcpy (v4.1), which mirrors the phone screen for real-time framing control and remote triggering. We captured in RAW rather than JPEG to avoid the phone’s built-in sharpening, denoising, and tone mapping, which can destroy the fine intensity variations our cropping pipeline relies on. Holding the phone parallel to the page and triggering remotely reduce motion blur and perspective distortion, which is therefore limited at capture time rather than corrected in software. Lighting was not controlled: captures were made under ambient conditions, so the page-detection stage must cope with uneven illumination. The full setup (laptop, phone, and capture stand) costs about 150,000 FCFA (roughly 230 EUR) and requires no specialized scanning equipment.

<table><tr><td>Record type</td><td>Images</td><td>Share</td></tr><tr><td>Birth</td><td>1,101</td><td>70.2%</td></tr><tr><td>Marriage</td><td>152</td><td>9.7%</td></tr><tr><td>Death</td><td>315</td><td>20.1%</td></tr><tr><td>Total</td><td>1,568</td><td>100%</td></tr></table>

Table 1: Corpus composition by record type (number of images).

Corpus composition. Table 1 gives the distribution of images by record type. Birth records dominate, and marriage records are scarce. The legal record type is only a rough guide to layout: manual inspection revealed five distinct templates, namely (G1) narrative marriage record, (G2) birth record by direct declaration, (G3) birth record transcribing a supplementary court judgment, (G4) narrative death record, and (G5) birth declaration on a pre-printed form filled in by hand. Birth records alone span three structurally different templates, so the recognizer must handle different line lengths, field arrangements, and mixes of printed and handwritten text.

Annotation. We transcribed 185 records covering all three record types in eScriptorium [26], producing line-level transcriptions aligned with the image. Lines were first segmented automatically and then corrected manually once; this corrected segmentation was stored and reused every time it was needed (training, evaluation, benchmarking). A line is transcribed if and only if it contains handwritten text: printed text is part of the ground truth only when it appears on a line that also contains handwriting, and purely printed lines are excluded. The data were split 70/15/15 into training, validation, and test sets at the line level, with the constraint that lines from the same record never appear in more than one set. For the downstream task, a separate set of 179 records was annotated with eight entity types in Label Studio (Section 4.4). All annotation was carried out by a single annotator, so we cannot report inter-annotator agreement.

Data availability. The records contain personal information about real individuals. Only two isolated text lines, with personal names and record identifiers masked, are shown in this paper (Figure 2); the corpus (images, transcriptions, and annotations) is not publicly released (Section 7). We will release the preprocessing code; the recognizer weights will be released after a memorization check and the required authorization.

## 4 Method

Our pipeline has three stages (Figure 1): classical preprocessing that turns raw smartphone captures into cropped, correctly oriented images of single records; handwritten text recogni-

![](images/06479486d5d651fda16aa77f9fffa2b6977cbeb0d434bac147f286cd4ed97ddb.jpg)  
Figure 1: Overview of the pipeline, with CPU-only inference. Top: the vision pipeline evaluated in this paper. Bottom: the downstream extraction stages, used only as a probe to measure how recognition errors propagate; feeding them ground-truth text gives the upper bound in Table 5.

tion with a fine-tuned Kraken model; and a downstream extraction stage that we use only to measure how recognition errors propagate. Inference in every stage runs on a laptop CPU.

## 4.1 Classical preprocessing

We deliberately use no learned model in preprocessing: classical operations need no training data or annotation, have predictable failure modes, and run fast on commodity hardware.

Normalization and orientation. Raw DNG files are renamed deterministically using their MD5 hash and grouped into batches. A portrait/landscape check found no image that needed rotation in practice. Upside-down captures occurred in a subset of the images of the marriage sub-corpus: 87 images in total, or 5.5% of the corpus. We detect them by normalized crosscorrelation template matching of the letter “e”, a frequent character in the corpus, with a detection threshold of 0.80, and rotate the detected images by 180°. Finally, we remove a fixed strip along the right edge that contains no document content.

Page cropping. Smartphone captures taken under uncontrolled lighting have uneven illumination, so a single global threshold such as Otsu’s [30] is unreliable. Instead, after smoothing with a 7 × 7 Gaussian kernel, we combine adaptive Gaussian thresholding with an HSV criterion that keeps low-saturation, bright pixels (S < 70 and V > 110 on an 8-bit scale) to separate page from background. We then compute a Canny edge map [8] (thresholds 30 and 90), reinforce it with directional morphological closing (vertical 3×60 and horizontal 60×3 kernels), and take horizontal and vertical projection profiles of the result to locate the page boundaries.

Separating adjacent records. When the detected region is too wide for a single record, because two records were captured in one frame, a disambiguation step chooses the split that best separates them, scoring candidates on fill ratio, relative width, and edge strength. A second pass targeted at the left margin then removes the binding area: after a 21 × 21 Gaussian blur and contrast-limited adaptive histogram equalization (clip limit 3.0), a Canny edge map (thresholds 100 and 200) locates the binding, and a fixed offset of 45 pixels is applied to the left boundary.

<table><tr><td>Stage</td><td>Parameter (code name)</td><td>Value</td></tr><tr><td rowspan="6">Page detection</td><td>blur_ksize</td><td>(7,7)</td></tr><tr><td>s_max,v_min</td><td>70,110</td></tr><tr><td>canny_t1,canny_t2</td><td>30,90</td></tr><tr><td>close_v,close_h</td><td>(3,60), (60,3)</td></tr><tr><td>prof_thr</td><td>0.5</td></tr><tr><td>ratio_min_page,ratio_cible</td><td>1.9,2.35</td></tr><tr><td rowspan="4">Left-margin cleaning</td><td>snap_tol,margin</td><td>45,10</td></tr><tr><td>blur_ksize_s2</td><td>(21,21)</td></tr><tr><td>clahe_clip_s2</td><td>3.0</td></tr><tr><td>canny_t1_s2,canny_t2_s2</td><td>100,200</td></tr><tr><td>Upside-down detection</td><td>decalage_gauche_px template character, NCC threshold</td><td>45 “e&quot;, 0.80</td></tr></table>

Table 2: Preprocessing parameters (code names as in our implementation).

No perspective correction. We do not apply homography-based rectification. The capture protocol keeps the phone parallel to the page, so perspective distortion is limited at the source: we prevent distortion during capture rather than correct it with extra computation afterwards. Some residual curvature near the binding remains (Section 6).

The full pipeline ran on all 1,568 images without errors, at about 2.6 images per second on the Dell 7750 laptop used for the benchmarks (Section 5). We did not measure the accuracy of page crops, record splits, or upside-down detection against manual references; the parameters (Table 2) were tuned by hand on this corpus and may need retuning elsewhere.

## 4.2 Handwritten text recognition

Architecture. We use Kraken [25], an open-source HTR system built on a hybrid convolutional recurrent network trained with the CTC loss [18]. It is small enough to run on a CPU, can be fine-tuned on a free-tier GPU, integrates directly with the eScriptorium annotation platform, and is widely used for historical documents.

Line segmentation. Lines were segmented automatically and then corrected manually once (Section 3); this corrected segmentation is reused throughout. All reported CER and WER values are therefore computed on known, ground-truth line boundaries and do not include segmentation errors. End-to-end evaluation with automatic segmentation is left to future work.

Fine-tuning. Training from scratch is infeasible with a few hundred annotated records. We therefore fully fine-tune all weights of the pretrained Kraken model kraken-ppocrv6-mediu (Hugging Face: small-models-for-glam/kraken-ppocrv6-medium). Because the pretrained output alphabet does not cover every character and diacritic in our corpus, we

merge it with the alphabet of our data using Kraken’s --resize union option. We train for 40 epochs with a learning rate of $1 0 ^ { - 3 }$ and a batch size of 7. Training took about four hours on a free-tier Google Colab GPU; it is the only stage of the pipeline that used a GPU. We did not use parameter-efficient methods: the model is small enough that full fine-tuning is already inexpensive.

Data split. Ground truth is split 70/15/15 at the line level, Kraken’s native unit, with the constraint that lines from one record never appear in more than one split. This prevents record-level leakage but not writer-level leakage (Section 6).

## 4.3 Cost measurement protocol

We report quality jointly with cost. We compare two configurations that share the same architecture and FP32 precision: the pretrained model and the fine-tuned model. Fine-tuning changes the weights but not the network, so any difference in inference cost between them reflects measurement noise rather than model size. We assume an operational ceiling of 2% CER, above which we expect checking and correcting the output to approach the effort of partial manual transcription. This threshold is a working assumption, not a measured correction cost, and the downstream results (Section 5.4) suggest that the error rate on names matters more than overall CER.

Benchmark protocol. We process 50 images drawn at random (seed 42) in five steps of ten images. Both models process the same images, and two additional images are processed beforehand as warm-up and excluded. Inference uses batches of 8 lines, line-length bucketing (64) with sorting, a frozen computation graph, channels-last layout, batch-normalization fusion, 6 threads, 2 line workers, and a prefetch of 2. The measured time covers recognition on the stored, manually corrected line segmentation and excludes automatic line segmentation.

Metrics. We report weight file size, model loading time, one-time warm-up time (first inference after loading), peak resident memory (RSS) of the process, time per image and per line, throughput, and mean CPU utilization. Time per image is the mean over the five steps, with its standard deviation across steps. We also report the training cost of fine-tuning.

We did not evaluate reduced-precision inference (e.g. INT8 quantization). Its benefit depends on hardware support and on the inference runtime, and our target machines are ordinary laptop CPUs without dedicated low-precision instructions; we leave it to future work on the actual deployment hardware.

## 4.4 Downstream extraction (evaluation probe)

To measure how recognition quality affects a real task, we feed HTR output into an extraction stage that identifies the people in each record and their roles (father, mother, spouse, witness, etc.). We fully fine-tune camembert-ner-with-dates [24], a French model of roughly 110M parameters derived from CamemBERT [28], on 179 annotated records, with eight entity types: PER, LOC, ORG, and DATE (inherited from the base model) and PROFES-SION, NUM\_REF (reference numbers), ACTE\_TYPE (record type), and ANCRE (lexical role markers such as fils de, “son of”). The record’s template is inferred with simple rules from the detected ACTE\_TYPE entities. Roles are then assigned in linear time by a deterministic algorithm that walks through the entities in reading order and uses a templatespecific dictionary mapping each marker to a role; each person entity inherits the role set by the most recent marker, and a block-counting mechanism handles the repeated field groups of template G5. Because these rules are fixed, role errors come from only three sources, namely recognition, NER, or template classification, which lets us isolate the contribution of HTR (Section 5.4).

<table><tr><td>Model</td><td>CER (%) ↓</td><td>WER  $( \% ) \downarrow$ </td></tr><tr><td>Kraken, pretrained (no adaptation)</td><td>14.53</td><td>56.93</td></tr><tr><td>Kraken, fine-tuned</td><td>0.50</td><td>3.37</td></tr></table>

Table 3: HTR results on the test set, computed on ground-truth line segmentation.

## 5 Experiments and Results

## 5.1 Setup

Preprocessing and inference benchmarks run on a Dell 7750 mobile workstation with an Intel Core i7-10850H CPU (6 cores), 15.6 GiB of RAM, and no GPU. This machine is faster than the ThinkPad T460 used for capture, on which we did not run the benchmarks. Fine-tuning ran on a free-tier Google Colab GPU. Software: Python 3.13, PyTorch 2.14.0 (CPU build), Kraken 7.1.1, OpenCV 5.0.0.93, jiwer 4.0.0, seqeval 1.2.2. Character error rate is

$$
\mathrm { C E R } = \frac { S + D + I } { N } ,\tag{1}
$$

where S, D, I are character substitutions, deletions, and insertions and N is the number of reference characters; WER is defined analogously at word level.

## 5.2 Recognition quality

Fine-tuning reduces CER by a factor of about 29 and WER by a factor of about 17 (Table 3). In absolute terms WER falls by 53.6 points, against 14.0 points for CER: with the pretrained model more than half of the words are affected by errors, whereas after fine-tuning fewer than 4% are. On inspection, many of the pretrained model’s errors fall on local personal and place names, as reported for other civil records [20], but we did not measure error rates on name spans. Part of its 14.53% CER also comes from characters absent from its output alphabet (Section 4.2), which it cannot produce at all. Both error rates are computed on the manually corrected line segmentation, so they measure recognition alone, not segmentation followed by recognition. Figure 2 shows representative test lines. Two further factors make the 0.50% CER optimistic: test lines are written by officers who also wrote training lines, and the ground truth includes printed field labels $( e . g . \ ^ {  } \mathrm { N } \acute { \mathbf { e } } \ ( \mathbf { e } ) ^ { \prime \prime } )$ whenever they share a line with handwriting; these labels recur across records and are easy to learn. We therefore read the figure as evidence of feasibility within this registry rather than as an estimate of performance on unseen writers or other registries (Section 6).

![](images/3775f30c9384aef76aabf655abb96ee8d772c7430ba6e59c43fd7ad5555efef4.jpg)

CER 37.5 %

CER 0.0 %

Figure 2: Two example test lines. Personal names, the record number and the office code are masked in the images (black) and replaced character by character in the transcriptions (grey); red marks show recognition errors.
<table><tr><td>Model</td><td>Disk (MiB)</td><td>Load / warm-up (s)</td><td>Peak RSS (GiB)</td><td>Time / image (s)</td><td>Time / line (ms)</td><td>Images / min</td></tr><tr><td>Pretrained</td><td>60.8</td><td>1.1 / 26.1</td><td>2.30</td><td>5.84±1.13</td><td>282</td><td>10.3</td></tr><tr><td>Fine-tuned</td><td>60.8</td><td>1.1 / 25.3</td><td>2.49</td><td>5.60±1.09</td><td>270</td><td>10.7</td></tr></table>

Table 4: Inference cost on a six-core laptop CPU (Intel Core i7-10850H, no GPU), for 50 images in five steps of 10 images (1,037 lines). Peak RSS is the mean over steps; ± is the standard deviation across steps.

## 5.3 Frugality of the recognizer

As expected for two models with the same architecture, their inference costs are similar (Table 4); the table mainly describes what one Kraken recognizer costs on a laptop CPU. Both models have the same weight file (60.8 MiB) and load in about one second. Peak memory stays below 2.6 GiB for both, far under the 15.6 GiB available; we did not test smaller machines. The small gaps between the two models (about 4% in time, about 0.2 GiB in peak RSS) are of the same order as the step-to-step variation and may depend on execution order, so we do not interpret them. The first inference after loading takes about 25 s and is a one-time cost paid at startup.

Throughput is about 10 images per minute, or 3.5–3.7 lines per second, using on average 5.3 of the 6 cores. Extrapolated to the full corpus, recognition of the 1,568 images takes roughly 2.5 hours on a single laptop. The first two steps were slower for both models (95thpercentile latency around 15 s, about 4.5 busy cores) than the last three (about 4.8–5.1 s per image, 5.8 busy cores); since both models process the same images, we attribute this to system effects such as thermal throttling or cache state rather than to image content.

Training cost. Fine-tuning (40 epochs) took about four hours on a free-tier Google Colab GPU. This is a one-off cost, but it is the only step that required a GPU; we did not measure

<table><tr><td>Input text</td><td>Role-resolution F1 (%)</td></tr><tr><td>Ground-truth transcription (upper bound)</td><td>90.3</td></tr><tr><td>Fine-tuned Kraken</td><td>81.2</td></tr><tr><td>Pretrained Kraken (no adaptation)</td><td>44.5</td></tr></table>

Table 5: Role-resolution F1 as a function of input text quality (n = 46 records, none from the HTR test set).

CPU-only training.

Since the fine-tuned model reaches 0.50% CER on this test set, below our assumed 2% ceiling, and runs as a small recurrent-convolutional model on a laptop CPU, it meets our working criteria on the dimensions we measured, subject to the caveats of Section 5.2.

## 5.4 How recognition errors propagate

We keep the extraction stage fixed and vary only its input text, on a subsample of 46 records that are not part of the HTR test set (Table 5). Because these records are outside the test set, the fine-tuned recognizer may have seen some of them during training, so its score is best read as an optimistic estimate and the recognition cost as a lower bound. Three observations follow. First, the extraction stage has its own ceiling: even with perfect text, role resolution reaches only 90.3% F1, owing to NER and template-classification errors. Second, residual recognition errors cost about 9 F1 points despite a test-set CER of only 0.50%: we expect recognition errors to concentrate on names and role markers, exactly the tokens the downstream task depends on. Third, with the pretrained recognizer F1 drops by a further 36.7 points, making the task essentially unusable. For information extraction, where recognition errors fall matters as much as how many there are.

## 5.5 Downstream extraction (summary)

The fine-tuned NER model reaches 98.0% micro-F1 over the eight entity types, evaluated on ground-truth transcriptions of a held-out test split (1,242 entities; the 179 records are split 70/15/15 by record). On the three entity types shared with the base model (PER, LOC, and DATE; ORG is left out because it is too rare in our data to be informative), adaptation improves weighted F1 from 86.8% to 98.6%, with most of the gain on PER entities (+18.0 points, against +8.0 for DATE). This suggests that the base NER model is weakest on local personal names. Rule-based template classification reaches 87.0% accuracy (n = 100); 8 of its 13 errors confuse templates G2 and G3, which share the same role sequence, so this confusion does not harm role resolution.

## 6 Discussion and Limitations

What the CER gain does and does not show. The 29-fold CER reduction shows that adaptation helps on this registry, but it does not explain why. At least three factors are confounded: the pretrained model is a general-purpose recognizer rather than a French handwriting model; it cannot output some characters of our corpus; and the fine-tuned model is tested on writers it has already seen. Our observation that errors fall on names is qualitative, and the stronger NER gain on PER than on DATE (+18.0 vs. +8.0 F1) concerns a different model. That generic models are poorly suited to local names is therefore a hypothesis to test, with name-span CER, an out-of-alphabet breakdown and a handwriting-specific baseline, rather than a result of this paper.

Frugality is not a single number. Fine-tuning leaves the architecture, and hence the inference cost, unchanged by design. The costs that matter for deployment lie elsewhere: a one-time warm-up of about 25 s, a peak memory of about 2.5 GiB, a throughput of about 10 images per minute, and, for adaptation itself, about four hours of GPU training plus the human effort of transcribing 185 records, correcting their segmentation and annotating 179 records for NER, which we did not time. A record-holding administration can run the resulting model on a laptop CPU, but adapting it to a new registry currently requires access to a GPU, or CPU training that we did not evaluate. Frugality claims should report every cost dimension, and we did not test lower-end machines or reduced-precision inference.

Limitations. Single site and possible writer overlap. All records come from one registry. The line-level split keeps each record within one split, but lines by the same officer, or from the same register, can appear in both training and test data, so the 0.50% CER may overestimate performance on unseen writers or other registries. A split by register or by writer, and a second site, are the most important next steps. Small test set. The HTR test set contains about 28 records, and marriage and death records are under-represented; per-type results are diagnostic at best, and we report no confidence intervals. What the CER measures. Printed field labels are part of the ground truth when they share a line with handwriting, which lowers CER; we do not report CER on handwritten characters only, on names, or per template. The recognizer was trained once, with no repeated runs or variance estimate, and compared with a single, general-purpose baseline. Single annotator. We cannot estimate how much residual error reflects annotation ambiguity. Line segmentation. CER and WER assume the manually corrected line segmentation; errors of automatic segmentation are not measured, and the timings exclude it. Cost measurements. Timings come from 50 images on a single six-core laptop, with the two models run sequentially, and show a systematic slowdown in the first two steps; they are orders of magnitude, not precise figures. Older machines, such as the laptop used for capture, will likely be slower. Quality and cost were measured on different image samples, and the cost-benchmark images were drawn from the corpus and may overlap with the annotated records. Preprocessing coverage. Upside-down detection was developed and tested only on the marriage sub-corpus, where such captures occurred, and no perspective correction is applied near the binding. Downstream probe. The 46 probe records may overlap the training data of the recognizer and of the NER model, so its scores are optimistic. Unablated choices. The benefit of RAW over JPEG capture was not tested. Frozen downstream rules. Role resolution relies on rules tied to this registry’s five templates and will likely need rewriting elsewhere.

## 7 Ethics Statement

The corpus consists of civil-status records of real people, containing names, family relationships, dates and places of birth, marriage, and death, and in some cases national identity numbers. Most of the people concerned were not informed that their records would be used for research. We did not anonymize or pseudonymize the working corpus: models were trained and evaluated on the original, identifying text. To limit risk, we (i) processed the data locally on a single machine, except for fine-tuning, which ran on a free-tier cloud GPU and therefore involved uploading the training images and transcriptions to that service, (ii) show no full record in this paper, only two isolated text lines in which personal names and record identifiers are masked (Figure 2), and (iii) do not release the images, transcriptions, or annotations. The cloud upload runs counter to the aim of this work, which is to keep identity data under the control of the administration and the community that hold it [14]; future adaptation should use hardware controlled by the institution.

Chad’s data-protection framework requires processing of personal data to be declared to, or authorized by, the national agency ANSICE [6, 34, 35]. Data collection was carried out with the written permission of the Municipality of Koumra, which authorizes use of the records for academic purposes only, requires personal information in published images to be masked, and excludes any commercial use. We do not claim that this study has been declared to or authorized by ANSICE, nor that research use of public civil-status archives is exempt from individual consent under this framework; we will seek the corresponding authorization before any release of data or models.

Fine-tuning on a cloud GPU required uploading the training images and transcriptions to that service. Future adaptation will run on hardware controlled by the institution, so that identity data remain under the control of the administration and the community that hold them [14].

Systems that make identity records searchable at scale could also support surveillance or profiling. Local, CPU-only deployment under the control of the record-holding administration reduces this risk compared with outsourced processing, but does not remove it.

## 8 Conclusion

We asked whether a frugal pipeline, built from consumer capture hardware, a few hundred annotated records, and CPU-only inference, can make handwritten civil-status registers from Sub-Saharan Africa machine-readable. On a new corpus of 1,568 images of records from a registry in Chad, fine-tuning a Kraken recognizer for about four hours on a free-tier GPU reduced CER from 14.53% to 0.50% on corrected line segmentation and writers seen in training, and the model runs at under 6 s per image and under 2.6 GiB of memory on a laptop CPU. A downstream extraction task showed that even this small residual error costs about 9 F1 points. These results suggest that recognition for such archives need not require GPUs or cloud services at deployment time, but they come from a single registry, a small test set with writer overlap, and a single baseline. Future work includes evaluation across registers, writers, and sites; error rates on handwritten text and on names; handwriting-specific and Transformer baselines; a learning curve over annotation size; a second annotator to measure agreement; end-to-end evaluation with automatic line segmentation; and, above all, an anonymization scheme that still permits record linkage, which must be in place before any data or trained models are shared with, or deployed by, the administrations that hold these records.

## Acknowledgements

We thank the Municipality of Koumra (Mandoul Oriental, Chad) for granting access to the civil-status registers used in this study and authorizing their use for academic research.

## References

[1] A. Abdoulahi. Contribution à l’archivage électronique et à la valorisation des manuscrits ouest-africains en langue arabe : la numérisation des manuscrits de Tombouctou. PhD thesis, Université Lumière Lyon 2, 2019.

[2] D. I. Adelani et al. MasakhaNER 2.0: Africa-centric transfer learning for named entity recognition. arXiv preprint arXiv:2210.12391, 2022.

[3] African Union. Innovation et numérisation de l’enregistrement des faits d’état civil et des statistiques de l’état civil en Afrique. African Union working document, 2019.

[4] Orevaoghene Ahia, Sachin Kumar, Hila Gonen, Jungo Kasai, David R Mortensen, Noah A Smith, and Yulia Tsvetkov. Do all languages cost the same? tokenization in the era of commercial language models. In Proceedings of the 2023 Conference on empirical methods in natural language processing, pages 9904–9923, 2023.

[5] Wissam AlKendi, Franck Gechter, Laurent Heyberger, and Christophe Guyeux. Advancements and challenges in handwritten text recognition: A comprehensive survey. Journal ofImaging, 10(1):18, 2024. doi: 10.3390/jimaging10010018.

[6] S. Bassounda and O. Bustin. Tchad : effectivité du régime de protection des données personnelles. L’Essentiel Droits africains des affaires, (3):4–5, 2020.

[7] K. Bostrom and G. Durrett. Byte pair encoding is suboptimal for language model pretraining. arXiv preprint arXiv:2004.03720, 2020.

[8] J. Canny. A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6):679–698, 1986.

[9] Civipol. SENEGAL: NEKKAL – projet d’appui au renforcement du système d’information de l’état civil et à la consolidation d’un fichier national d’identité biométrique. Project fact sheet, Civipol / Enabel / European Union, Paris, 2023.

[10] Commission économique pour l’Afrique. Rapport sur les progrès accomplis dans la mise en œuvre du programme africain d’amélioration accélérée des systèmes d’enregistrement des faits d’état civil et d’établissement des statistiques de l’état civil. United Nations, 2024.

[11] Y. Dangourbe. Modernisation de l’état civil au Tchad : un atelier pour connecter tribunaux et registre biométrique. Tchadinfos (online news), 2026.

[12] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pages 10088–10115, 2023.

[13] Jean Louis Kedieng Ebongue Fendji. From left behind to left out: Generative ai or the next pain of the unconnected. Harvard Data Science Review, (Special Issue 5), 2024.

[14] Jean Louis Fendji Kedieng Ebongue. From community network to community data: Towards combining data pool and data cooperative for data justice in rural areas. arXiv preprint arXiv:2503.05950, 2025.

[15] Alicia Fornés, Verónica Romero, Arnau Baró, Juan Ignacio Toledo, Joan Andreu Sánchez, Enrique Vidal, and Josep Lladós. Icdar2017 competition on information extraction in historical handwritten records. In 2017 14th IAPR international conference on document analysis and recognition (ICDAR), volume 1, pages 1389–1394. IEEE, 2017.

[16] A. N. Fountsop, J. L. Fendji Kedieng Ebongue, and M. Atemkeng. Deep learning models compression for agricultural plants. Applied Sciences, 10(19):6866, 2020.

[17] A. Graves and J. Schmidhuber. Offline handwriting recognition with multidimensional recurrent neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 21, 2008.

[18] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. In Proc. International Conference on Machine Learning (ICML), pages 369–376, 2006.

[19] S. Han, J. Pool, J. Tran, and W. J. Dally. Learning both weights and connections for efficient neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 28, pages 1135–1143, 2015.

[20] Lisa Hoek, Björn Quanjer, Thunnis van Oort, Erik Tjong Kim Sang, and Matthias Rosenbaum-Feldbrügge. Defeating the haystack: Lessons learned from extracting entities from Dutch Curaçao civil certificates, 1879–1949. Digital Scholarship in the Humanities, 2026. doi: 10.1093/llc/fqag052.

[21] E. J. Hu et al. LoRA: Low-rank adaptation of large language models. In Proc. International Conference on Learning Representations (ICLR), 2022.

[22] International Development Research Centre (IDRC). Africa compute initiative: Expanding African access to compute for AI research. IDRC project brief, 2026.

[23] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 2704–2713. IEEE, 2018.

[24] Jean-Baptiste. camembert-ner-with-dates. Hugging Face model card, huggingface.co/Jean-Baptiste/camembert-ner-with-dates, 2023.

[25] B. Kiessling. Kraken – an universal text recognizer for the humanities. In Proc. 2nd International Conference on Digital Access to Textual Cultural Heritage (DATeCH), pages 91–96, 2019.

[26] B. Kiessling, R. Tissot, D. Stökl Ben Ezra, and P. A. Stokes. eScriptorium: An open source platform for historical document analysis. In International Conference on Document Analysis and Recognition Workshops (ICDARW), volume 2, 2019.

[27] M. Li et al. TrOCR: Transformer-based optical character recognition with pre-trained models. In Proc. AAAI Conference on Artificial Intelligence, volume 37, pages 13094– 13102, 2023.

[28] L. Martin et al. CamemBERT: a tasty French language model. In Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL), pages 7203–7219, 2020.

[29] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart Van Baalen, and Tijmen Blankevoort. A white paper on neural network quantization, 2021.

[30] N. Otsu. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1):62–66, 1979.

[31] A. Ougock. Cameroun : Les archives de l’État civil détruits après l’incendie de la mairie de Yaoundé VI. Koaci (online news), 2023.

[32] R. Plamondon and S. N. Srihari. On-line and off-line handwriting recognition: A comprehensive survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 22:63–84, 2000.

[33] Realites Info. Le Tchad entre dans l’ère du numérique avec la modernisation de son état civil. Realites Info (online news), 2025.

[34] République du Tchad. Loi n<sup>◦</sup> 006/PR/2015 portant création de l’Agence Nationale de Sécurité Informatique et de Certification Électronique (ANSICE), 2015.

[35] République du Tchad. Loi n<sup>◦</sup> 007/PR/2015 portant protection des données à caractère personnel, 2015.

[36] Verónica Romero, Alicia Fornés, Nicolás Serrano, Joan Andreu Sánchez, Alejandro H Toselli, Volkmar Frinken, Enrique Vidal, and Josep Lladós. The esposalles database: An ancient marriage license corpus for off-line handwriting recognition. Pattern Recognition, 46(6):1658–1669, 2013.

[37] V. Sanh, L. Debut, J. Chaumond, and T. Wolf. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.

[38] R. Schwartz, J. Dodge, N. A. Smith, and O. Etzioni. Green AI. Communications of the ACM, 63(12):54–63, 2020.

[39] U. Springmann, C. Reul, S. Dipper, and J. Baiter. Ground truth for training OCR engines on historical documents in German Fraktur and Early Modern Latin. arXiv preprint arXiv:1809.05501, 2018.

[40] UN DESA, UNSD, and Vital Strategies. Guidelines on the legislative framework for civil registration, vital statistics and identity management systems. International Standards and Recommendations, United Nations, New York, 2023.

[41] United Nations Children’s Fund (UNICEF). The right start in life: Global levels and trends in birth registration, 2024 update. Global Progress Report, UNICEF, New York, 2024.

[42] United Nations Department of Economic and Social Affairs. Handbook on Civil Registration and Vital Statistics Systems: Management, Operation and Maintenance, Revision 1. United Nations, 2021.

[43] United Nations Economic Commission for Africa. Report on the status of civil registration and vital statistics in Africa. Technical report, United Nations, Addis Ababa, Ethiopia, 2017.