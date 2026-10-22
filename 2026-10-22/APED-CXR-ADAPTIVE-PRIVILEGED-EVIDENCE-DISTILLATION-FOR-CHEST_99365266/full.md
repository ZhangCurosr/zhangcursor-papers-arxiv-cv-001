# APED-CXR: ADAPTIVE PRIVILEGED EVIDENCE DISTILLATION FOR CHEST X-RAY GROUNDING WITH IMAGE-ONLY INFERENCE

Md Shihabul Islam Shovo<sup>⋆</sup>

Nishi Kanta Paul<sup>⋆</sup>

Jamil Fayyad†

<sup>⋆</sup>NIMISHES Lab, Dhaka, Bangladesh

†American University of Sharjah, Sharjah, UAE

shihabul900@gmail.com, nishikantapaul108@gmail.com, jfayyad@aus.edu

## ABSTRACT

Compact chest X-ray (CXR) models that localize findings suit deployment, but box annotations are limited and report-aware multimodal models are expensive at inference. We introduce APED-CXR, which distills report-informed spatial evidence from a frozen multimodal teacher into a student with 11.1M parameters that predicts diseases and boxes from shared evidence maps and requires the image alone at test time. During training, the teacher uses the radiograph, report, and queried class to evaluate regions proposed from the student’s current evidence maps, and its support scores serve as training targets for the student’s pooled evidence scores within these regions. Trained on PadChest and evaluated on held-out PadChest-GR, refreshing the student’s proposals for teacher evaluation improves box-level localization F1 from 0.302 with fixed warm-up proposals to 0.338 (IoU 0.5; gain 0.036, 95% CI [0.022, 0.049]), while maintaining diagnostic AUC. This gain depends on the teacher checking current proposals: with equally many random boxes, a fixed dense box set, teacher-generated boxes, or report rules instead of the teacher, F1 stays near the fixed-proposal level, and proposals two refreshes old recover only part of the gain. On NIH ChestXray14, a secondary test set, box recall also rises from 0.423 to 0.447.

Index Terms— Chest X-ray, knowledge distillation, privileged information, visual grounding, weakly supervised localization

## 1. INTRODUCTION AND RELATED WORK

Automated chest X-ray (CXR) interpretation is increasingly expected to localize where a finding occurs, not only whether it is present. For routine use, such a model should also be compact and independent of the radiology report, which does not yet exist when the image is first read. Report-aware vision–language models (VLMs) can localize findings, but are too large and text-dependent for use at the time of the read, while compact image-only models rarely receive enough box supervision to do so reliably. We thus introduce Adaptive Privileged Evidence Distillation for Chest X-Rays (APED-CXR), in which a frozen multimodal teacher verifies the student’s proposed regions during training only, distilling reportinformed spatial knowledge into a compact image-only model.

Prior CXR localization techniques mostly used image-level supervision. Class activation maps (CAM) [1] project classifier weights onto convolutional features to highlight class-discriminative regions, while pooling-based multiple-instance learning [2, 3] combines local predictions into image-level scores, for example with log-sum-exp pooling; ChestX-ray8 [3] applied the approach to textmined labels and evaluated localization on a small boxed subset. However, a saliency benchmark found such maps inferior to radiologists at localizing pathology [4], which motivates supervising the evidence map directly. Li et al. [5] showed that a few boxes can supplement image-level supervision, but expert-curated datasets are small: PadChest-GR provides radiologist boxes in 4,555 CXR studies [6], while PadChest provides over 160,000 [7]. Thus, most training samples have labels and reports but lack boxes.

Recent VLMs have used reports to localize findings. BioViL [8] grounds phrases using image–report pairs, CoNNS [9] suppresses noisy negatives for zero-shot classification and grounding, while MAIRA-2 [10] and ChEX [11] generate grounded reports or region descriptions. MedGemma [12] answers localized questions about radiographs. Although these advances exist, such models require text at inference or have billions of parameters, which limits their use as compact first-read tools. RegionCLIP [13] suggests another route: a vision–language model assigns scores to region proposals, creating pseudo-labels so that its knowledge supervises a separate model instead of being used directly at inference.

Knowledge distillation transfers soft targets from the teacher to a smaller student [14], and learning with privileged information uses data, such as reports, available to the teacher during training but not during testing [15, 16]. Existing CXR distillation methods transfer labels, embeddings, or explanations rather than spatial evidence. Termritthikun et al. [17] distill a classifier with explainable outputs for on-device inference. X-Plain [18] distills the embeddings and text explanations of a vision–language model into an EfficientNet-B3 student. Thus, transferring spatial evidence raises a design question – which regions should the teacher evaluate? Evaluating fixed proposals generated at warm-up is simple and cheap, but the student’s evidence changes during training, so these proposals may no longer match the regions it currently emphasizes. DAgger [19] trains on states observed by the learner, on-policy distillation [20] supervises sequences generated by the student, and Soft Teacher [21] weights student-proposed boxes by teacher scores in semi-supervised detection. It is unclear whether such learner-derived supervision helps pathology localization, where teacher queries are costly and teacher judgments of box extent are imperfect.

APED-CXR therefore uses a frozen multimodal teacher that reads the radiograph, report, and queried class to verify proposals from the current student, and turns its support scores into targets for the student’s evidence maps (Fig. 1). Since diagnosis and localization use the same maps, the student requires the image alone at test time, and the teacher’s cost is restricted to training. As the teacher scores each proposal independently, its targets can amplify evidence in supported regions and dampen it when all proposals are poor. Our contributions are threefold: (1) adaptive privileged verification, whose independent support targets directly modify the evidence maps used for image-only diagnosis and grounding; (2) paired controls that separate adaptive feedback from query exposure, recency, scorer type, dense proposal coverage, and cached teacher-generated boxes, with the teacher cost of each; and (3) a patient-disjoint held-out evaluation using box-level grounded F1 on PadChest-GR, with five paired seeds and patient-level bootstrap intervals, plus a secondary evaluation on NIH ChestX-ray14 boxes [3].

## 2. METHOD

## 2.1. Evidence-Map Student

An EfficientNet-B3 encoder [22] and a lightweight adapter compose features from strides 8, 16, and 32 of a radiograph x into a stride-8 map F. A 1 1 convolution predicts the evidence logits and the bounded evidence map of class c,

$$
A _ { c } ( u , v ) = w _ { c } ^ { \top } F _ { u , v } + b _ { c } , \qquad H _ { c } = \sigma ( A _ { c } ) ,\tag{1}
$$

where σ is the sigmoid function. Log-sum-exp (LSE) pooling [2, 3] over the n valid, non-padded cells produces the disease probability,

$$
z _ { c } = \frac { 1 } { \rho } \log \Big ( \frac { 1 } { n } \sum _ { u , v } e ^ { \rho A _ { c } ( u , v ) } \Big ) , \qquad \hat { y } _ { c } = \sigma ( z _ { c } ) ,\tag{2}
$$

where $\rho \ : = \ : 1$ . Since there is no extra classifier, spatial supervision of $A _ { c }$ also affects the diagnostic output. At inference, $H _ { c }$ is thresholded, connected components are turned into boxes, and nonmaximum suppression (NMS) with a box cap produces $\hat { B } _ { c } .$ Boxes are generated only if $\hat { y } _ { c } \ge \tau _ { c } ,$ , thus a single class might give multiple boxes or none.

## 2.2. Proposals and Privileged Verification

For each explicitly positive image–class pair, proposals are extracted without backpropagation by applying the inference extraction at thresholds 0.2, 0.4, 0.6, 0.8 , merging duplicates, and keeping the top three components by mean evidence. One exploration box from a fixed multi-scale bank completes the set $( \bar { K } \ = \ 4 )$ and lets the teacher also support regions where the student has no focus yet. The student scores the proposal $B _ { j }$ by its mean evidence, $r _ { S } ^ { ( j ) } = \sigma \big ( | B _ { j } | ^ { - 1 } \sum _ { ( u , v ) \in B _ { i } } A _ { c } ( u , v ) \big )$ . The frozen teacher, MedGemma 1.5 4B [12], was selected since it was trained on CXRs, works locally to satisfy the data agreements, and exposes answer likelihoods. It receives the radiograph x with $B _ { j }$ drawn as an outline, the normalized box coordinates, the preprocessed report text ${ \mathcal { R } } ,$ and the class name, and is asked whether the box is sufficiently localized to capture one visible occurrence of the finding. If $\ell _ { \mathrm { y e s } }$ and $\ell _ { \mathrm { n o } }$ are the log-likelihoods of two fixed answer tokens, the support score is

$$
p _ { T } ^ { ( j ) } = \sigma \big ( ( \ell _ { \mathrm { y e s } } - \ell _ { \mathrm { n o } } ) / T \big ) ,\tag{3}
$$

where the temperature $T$ is fitted on development data. The prompt is fixed before training since answer likelihoods depend on prompt format [23]. As scores are independent across proposals, a set of consistently bad proposals can be awarded uniformly low support.

## 2.3. Training Objective

Let $m _ { i , c } \in \{ 0 , 1 \}$ be a mask that marks explicitly positive or negative labels in the report-derived labels; uncertain and unmentioned labels are masked rather than considered negative. The classification loss $\mathcal { L } _ { \mathrm { c l s } }$ is the average of the binary cross-entropy (BCE) between $\hat { y } _ { i , c }$ and $y _ { i , c }$ over the labels with $m _ { i , c } = 1$ . On images with manually labeled boxes, the union $M _ { i , c }$ of same-class boxes supervises $H _ { c }$ through ${ \mathcal { L } } _ { \mathrm { b o x } } .$ , the sum of the BCE and Dice [24] losses between $H _ { c }$ and $M _ { i , c } ;$ the filled boxes are coarse targets, but not lesion masks. The privileged loss is the average of the soft-target

BCE first on the $K _ { i , c }$ proposals of the pair and then on the set $\mathcal { P }$ of eligible pairs,

$$
\mathcal { L } _ { \mathrm { p r i v } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , c ) \in \mathcal { P } } \frac { 1 } { K _ { i , c } } \sum _ { j = 1 } ^ { K _ { i , c } } \mathrm { B C E } \left( r _ { S } ^ { ( j ) } , p _ { T } ^ { ( j ) } \right) ,\tag{4}
$$

and the total loss is $\mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \beta \mathcal { L } _ { \mathrm { b o x } } + \gamma \mathcal { L } _ { \mathrm { p r i v } } ,$

## 2.4. Adaptive and Static Proposal Schedules

Stage A trains on diagnostic labels only; Stage B additionally uses $\mathcal { L } _ { \mathrm { b o x } }$ and produces a warm-up checkpoint $S _ { 0 }$ for each seed. During Stage $\mathrm { C } ,$ each branch begins with the same $S _ { 0 }$ and optimizer state. The static comparator extracts and scores proposals just once from $S _ { 0 }$ but leaves them constant, while the current evidence inside them is pooled again at every update. The adaptive approach regenerates proposals from the current student $S _ { t }$ every $N$ updates, scores the new proposals, and reuses cached scores for duplicate queries (the same image, class, report, and box). The branches share images, sampling order, loss weights, and update budget, so the only difference between them is the proposal source.

## 3. EXPERIMENTAL SETUP

## 3.1. Datasets

We train on frontal adult PadChest radiographs [7] with their preprocessed Spanish reports and report-derived labels (27% annotated by physicians, the rest by a recurrent network). We employ the official stratified patient split of PadChest-GR [6], which also covers the other PadChest studies, so no patient overlaps partitions. PadChest-GR provides boxes by up to two radiologists; its 3,185 training and 455 validation studies act as grounding-train and grounding-val, and its 915 test studies form the primary held-out test set, never used for training or tuning. For a secondary cross-dataset set, we use the NIH ChestX-ray14 boxes [3], where a radiologist boxed one queried finding per image; since the teacher was trained on ChestX-ray14 [12], this set is not independent of the supervision. After auditing label mappings and per-class statistics, we retain only atelectasis (ATL), cardiomegaly (CMG), consolidation (CON), edema (EDM), pleural effusion (EFF), and pneumothorax (PTX); lung opacity and pneumonia are excluded because they overlap other classes. Distillation uses a fixed, patient-stratified subset of 40,000 training images with at least one positive target finding; images of grounding-val and test patients are excluded before it is drawn.

## 3.2. Controls

C0 uses diagnostic labels only, C1 extends them with manual boxes without a teacher, and C2 serves as the static comparator. C3 refreshes random boxes matched in number and size, so it receives as many teacher queries without learner-derived regions (query exposure). C5 scores a dense fixed bank of 32 boxes per pair once (coverage rather than refreshing), and C6 replaces the teacher with report-derived laterality and zone rules on current proposals (scorer type). C7 uses proposals from two refreshes earlier (recency), and C8 transforms boxes generated once by the teacher into targets on the fixed bank (generation rather than verification). C4 scores current proposals by their overlap with manual boxes (annotation-based feedback) and is thus assessed on grounding-train images only.

![](images/0c94776f3ff9a315494a0d7e86f23b93de411cbf489235403d462aa008de57a1.jpg)  
Fig. 1. Overview of APED-CXR. The student predicts disease probabilities $\hat { y } _ { c }$ and boxes $\hat { B } _ { c }$ from evidence maps $A _ { c }$ using the image alone. In training, a frozen teacher verifies proposals $\{ B _ { j } \}$ ; its support scores p<sub>T</sub> supervise the box-pooled scores $\mathbf { \Delta } _ { \mathbf { \Delta } } \mathbf { r } _ { S } .$ . The switch selects proposals from the current student S<sub>t</sub> (APED-CXR) or the warm-up checkpoint S<sub>0</sub> (static comparator C2). Maps and boxes are schematic.

## 3.3. Evaluation Protocol

The prespecified primary evaluation metric is class-aware box-level micro-gF1 at an intersection-over-union (IoU) threshold of 0.5 on the PadChest-GR test partition. Eligible pairs include 341 fully annotated positive pairs with 402 boxes from the designated primary reader and 1,691 reliable-negative pairs; uncertain, unmentioned, and non-localizable positives are excluded. Boxes are accepted with $\hat { y } _ { c } ~ \ge ~ \tau _ { c }$ and matched one-to-one to same-class reference boxes by maximizing matches with $\mathrm { I o U } \ge 0 . 5$ Then micro-gF1 is defined as $\mathrm { 2 g T \bar { P } / ( 2 g T P + g F P + g F N ) }$ , where unmatched predictions, including all boxes on negative pairs, constitute gFP and unmatched reference boxes, gFN. We also report macro-gF1 and the diagnostic macro-AUC over known labels. NIH lacks reliable negatives and boxes only the queried finding, so we report classconditional box recall at $\mathrm { I o U } \ge 0 . 5$ for ATL, CMG, EFF, and PTX (577 boxes, 550 images, 471 patients) using the ungated boxes of the annotated class. Results are the mean standard deviation (SD) over five paired seeds for C1, C2, and Full and three for the other controls, and 95% confidence intervals (CIs) for paired differences over their common seeds come from 2,000 patient-level bootstrap samples, conditional on the trained seeds and resampling patients jointly for both branches.

## 3.4. Implementation Details

Images are padded to 512 512 pixels with the aspect ratio conserved, giving 64 64 evidence maps. We use AdamW with a weight decay of 0.01, a learning rate of $\mathrm { 3 \tilde { \times } 1 0 ^ { - 4 } }$ during Stage A and $1 0 ^ { - 4 }$ afterwards, and cosine decay, with 32 diagnostic and 16 groundingtrain images drawn per update. Stages A, B, and C run 40,000, 6,000, and 12,000 updates, with a refresh every $N = 1 { , } 2 0 0$ updates $( R = 1 0 )$ . We set $\beta = 1$ and select $\gamma \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 1 . 0 \} , \tau _ { c } ,$ and the heatmap threshold per branch on grounding-val with the same budget; NMS uses an IoU of 0.5 and at most three boxes per class. The teacher runs in bfloat16 without gradients. All experiments use PyTorch on one NVIDIA A100 GPU (40 GB), and latency is averaged over 100 timed images after warm-up.

Table 1. Teacher scoring of grounding-val proposals from image x, report , class c, and box B; generated boxes are mapped to proposals by overlap. Acceptable: IoU 0.5 with a same-class box. AU-ROC: acceptable vs. other; Top-1: best-scored is acceptable; ρ<sub>IoU</sub>: Spearman correlation with IoU; All-bad: mean support if none is acceptable. Time: ms per image–class pair.
<table><tr><td>Mode</td><td>Inputs</td><td>AUROC↑ Top-1↑ρIoU ↑ All-bad↓</td><td></td><td></td><td></td><td>ms</td></tr><tr><td>Image only</td><td> $_ { { \mathbf { x } } , c , B }$ </td><td>0.716</td><td>0.46</td><td>0.37</td><td>0.41</td><td>318</td></tr><tr><td>Report only</td><td> ${ \mathcal { R } } , c , B$ </td><td>0.689</td><td>0.43</td><td>0.31</td><td>0.46</td><td>86</td></tr><tr><td>Report rules</td><td> ${ \mathcal { R } } , c , B$ </td><td>0.664</td><td>0.40</td><td>0.28</td><td>0.38</td><td>&lt;1</td></tr><tr><td>Generation</td><td> ${ \mathbf { } } x , \mathcal { R } , c$ </td><td>0.652</td><td>0.39</td><td>0.30</td><td>0.19</td><td>1,380</td></tr><tr><td>Image+report</td><td> ${ \mathbf { \mathit { x } } } , \mathcal { R } , c , B$ </td><td>0.781</td><td>0.55</td><td>0.46</td><td>0.27</td><td>333</td></tr></table>

## 4. RESULTS AND DISCUSSION

## 4.1. Teacher Verification

On grounding-val (Table 1), combined image and report verification gives the best proposal AUROC (0.781) and top-1 hit rate (0.55, candidate oracle recall 0.71), and gives less support to all-bad sets than image-only or report-only scoring. Moving a proposal to an incorrect location with the report unchanged reduced its score by 0.24 on average, so the teacher responds to box location and not just report content. Report rules cover only 58% of pairs, while generation, although conservative on all-bad sets, ranks proposals less accurately and returned a valid output for 88.9% of queries, so all teacher branches use combined verification.

## 4.2. Adaptive versus Static Supervision

In Table 2, manual boxes roughly double the micro-gF1 of C0, while static privileged supervision (C2) adds a further 0.017 (95% CI [0.004, 0.030]). The adaptive procedure achieves 0.338, giving a paired gain of 0.036 over C2 (95% CI [0.022, 0.049]) that is positive in all 5 seeds (0.019 to 0.057). Diagnostic AUC remains between 0.809 and 0.815 across branches.

Table 2. Box-level grounding on the held-out PadChest-GR test partition $( \mathrm { I o U } \ge 0 . 5 ;$ mean SD over five seeds for C1, C2, and Full, three otherwise). ∆: paired difference from C2 over common seeds (patient-bootstrap 95% CI). All rows except C0 use the manual box loss.
<table><tr><td>ID</td><td>Spatial supervision</td><td> $\mathbf { M i c r o - g F 1 } \uparrow$ </td><td>∆ vs. static [95% CI]</td><td>Macro-gF1 ↑</td><td>AUC↑</td><td>Teacher GPU-h↓</td></tr><tr><td>C0</td><td>None (image-level labels only)</td><td> $0 . 1 4 0 \pm 0 . 0 1 4$ </td><td> $- 0 . 1 6 5 [ - 0 . 1 8 8 , - 0 . 1 4 4 ]$ </td><td>0.101</td><td>0.812</td><td>0</td></tr><tr><td>C1</td><td>Manual boxes only</td><td> $0 . 2 8 4 \pm 0 . 0 0 9$ </td><td> $- 0 . 0 1 7 \left[ - 0 . 0 3 0 , - 0 . 0 0 4 \right]$ </td><td>0.201</td><td>0.809</td><td>0</td></tr><tr><td>C2</td><td>Teacher on fixed warm-up proposals (S0)</td><td> $0 . 3 0 2 \pm 0 . 0 0 5$ </td><td></td><td>0.215</td><td>0.813</td><td>5.9</td></tr><tr><td>C3</td><td>Teacher on refreshed random boxes</td><td> $0 . 3 1 1 \pm 0 . 0 2 2$ </td><td> $+ 0 . 0 0 6 \left[ - 0 . 0 1 0 , + 0 . 0 2 2 \right]$ </td><td>0.223</td><td>0.811</td><td>57.5</td></tr><tr><td>C5</td><td>Teacher on a fixed dense proposal bank</td><td> $0 . 3 0 8 \pm 0 . 0 0 8$ </td><td> $+ 0 . 0 0 4 \left[ - 0 . 0 1 6 , + 0 . 0 2 2 \right]$ </td><td>0.224</td><td>0.813</td><td>47.4</td></tr><tr><td>C6</td><td>Report rules on current proposals  $( S _ { t } )$ </td><td> $0 . 3 1 1 \pm 0 . 0 0 9$ </td><td> $+ 0 . 0 0 6 [ - 0 . 0 1 1 , + 0 . 0 2 3 ]$ </td><td>0.222</td><td>0.810</td><td>0</td></tr><tr><td>C7</td><td>Teacher on lagged proposals  $( S _ { t - 2 } )$ </td><td> $0 . 3 2 2 \pm 0 . 0 2 2$ </td><td> $+ 0 . 0 1 8 \left[ - 0 . 0 0 1 , + 0 . 0 3 6 \right]$ </td><td>0.240</td><td>0.814</td><td>35.6</td></tr><tr><td>C8</td><td>Cached teacher-generated boxes</td><td> $0 . 3 1 0 \pm 0 . 0 0 8$ </td><td> $+ 0 . 0 0 5 \left[ - 0 . 0 1 2 , + 0 . 0 2 2 \right]$ </td><td>0.221</td><td>0.812</td><td>6.1</td></tr><tr><td>Full</td><td>Teacher on current proposals  $( S _ { t } )$ </td><td> ${ \bf 0 . 3 3 8 \pm 0 . 0 1 8 }$ </td><td> $+ 0 . 0 3 6 \left[ + 0 . 0 2 2 , + 0 . 0 4 9 \right]$ </td><td>0.253</td><td>0.815</td><td>34.4</td></tr><tr><td colspan="7">†Matched manual subset: distillation restricted to grounding-train images;  $\Delta \nu s . \ C 2 ^ { \dagger }$ </td></tr><tr><td> $C 2 ^ { \dagger }$ </td><td>Teacher on fixed warm-up proposals  $( S _ { 0 } )$ </td><td> $0 . 2 9 7 \pm 0 . 0 1 1$ </td><td></td><td>0.213</td><td>0.810</td><td></td></tr><tr><td> ${ \mathrm { C } } 4 ^ { \dagger }$ </td><td>Manual-overlap score on current proposals</td><td> $0 . 3 1 3 \pm 0 . 0 0 9$ </td><td> $+ 0 . 0 1 6 \left[ - 0 . 0 0 0 , + 0 . 0 3 2 \right]$ </td><td>0.226</td><td>0.811</td><td></td></tr><tr><td>Full†</td><td>Teacher on current proposals  $( S _ { t } )$ </td><td> $0 . 3 0 6 \pm 0 . 0 0 5$ </td><td> $+ 0 . 0 0 9 \left[ - 0 . 0 0 7 , + 0 . 0 2 4 \right]$ </td><td>0.218</td><td>0.811</td><td></td></tr></table>

Table 3. Inference cost per image (A100, batch 1; ms, peak GB) and PadChest-GR test results; MedGemma generates boxes zeroshot from the image and class names.
<table><tr><td>Model</td><td>Params GMACs</td><td>ms GB</td><td>gF1↑AUC↑</td></tr><tr><td>MedGemma</td><td>4B</td><td></td><td>1,910 9.6 0.189</td></tr><tr><td>APED-CXR</td><td>11.1M</td><td>5.7</td><td>9.6 0.38 0.338 0.815</td></tr></table>

C3, C5, C6, and C8 stay close to C2, with paired differences of at most 0.006, and the adaptive procedure exceeds each by at least 0.034 (all 95% CIs above zero). Refreshed random boxes (C3) receive as many teacher queries but are not derived from the student’s evidence, so query exposure alone does not explain the gain. The dense bank (C5) is scored once and cannot react to the student’s changing evidence, rules (C6) lack box extent, and generated boxes (C8) inherit the weaker ranking of generation (Table 1). Lagged proposals (C7) recapture part of the gain (0.018 over C2), but remain 0.023 below current proposals (95% CI [0.004, 0.040]), so recency contributes. In the matched subset, teacher-scored and annotationscored proposals (C4) gain similarly over C1 (0.020 versus 0.027) and are not separated here (C4 Full<sup>†</sup> +0.007, 95% CI [ 0.009, +0.024]), so the teacher is directionally consistent with annotationbased feedback, not shown to match it.

On grounding-val (Fig. 2a), C2 saturates after roughly two refresh intervals, while Full and C7 keep improving, with C7 about two refreshes behind. Relative to C1, the largest gains are for atelectasis, consolidation, and effusion (Fig. 2b), while cardiomegaly, where C1 already performs best, changes less; edema and pneumothorax have only 12 and 10 positive test pairs, so their estimates are unstable. The gain over C2 is comparable under a common threshold across branches (0.035, 95% CI [0.021, 0.049]) and on the doubleannotated subset averaged across readers (0.037, 95% CI [0.020, 0.055]), and positive predictions without any box drop from 7.4% to 6.1%. On NIH ChestX-ray14, adaptive supervision improves box recall from 0.423 (C2) to 0.447 (0.024, 95% CI [0.011, 0.038]), while static supervision adds little over manual boxes (0.007, 95% CI [ 0.005, 0.019]), matching its smaller in-domain gain.

## 4.3. Limitations and Efficiency

The main limitation is that the primary test set is from the training hospital. NIH ChestX-ray14 only partly addresses this, as it covers four classes without negatives and was used to train the teacher, so it is a secondary cross-dataset check rather than independent external validation. The supervision is also noisy: the teacher reads stemmed Spanish report text without stop words, and most PadChest labels were assigned automatically. Filled boxes are coarse targets, and an $\mathrm { I o U } \ge 0 . 5$ match neither measures lesion delineation nor establishes clinical faithfulness. Finally, the adaptive branch queries the teacher more often than C2, so they are compared at unequal teacher cost, although C3 receives even more queries and stays near C2.

![](images/1cb277084dc84ac0952698e11b61964d855a260e7a1aa2c28f75b8ba6a8fdf84.jpg)  
Fig. 2. (a) Grounding-val micro-gF1 across refreshes and (b) perclass gF1 on the PadChest-GR test partition (mean SD over seeds).

The student needs 11.1M parameters and 5.7 GMACs and takes 9.6 ms per image, including box extraction (Table 3). Without the report and proposals, the zero-shot teacher grounds findings less accurately than the student and is about 200 times slower. The cost of adaptivity appears during training: 34.4 teacher GPU-hours per seed, 5.8 times that of C2, although 42% of its queries are cache hits.

## 5. CONCLUSION

APED-CXR distills report-informed spatial evidence into a compact image-only CXR student using the teacher’s verification of the student’s own proposals. On held-out PadChest-GR data, refreshed learner-derived proposals outperformed fixed proposals and the other proposal sources on box-level grounding, the lagged ones recouped part of the gain, and the direction held on NIH ChestXray14. Future work will cut teacher cost by selective refreshing and use larger multi-institution test sets.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

PadChest [7] and PadChest-GR [6] were obtained from BIMCV under their research use agreements, and NIH ChestX-ray14 [3] was downloaded from the NIH site https://nihcc.app.box. com/v/ChestXray-NIHCC. Ethical approval was not required for this retrospective study of de-identified data, in accordance with the data use agreements of the source datasets. The radiograph in Fig. 1 is a public NIH ChestX-ray14 image, and no PadChest images or reports are reproduced.

## 7. ACKNOWLEDGMENTS

No funding was received for conducting this study. The authors have no relevant financial or non-financial interests to disclose. We acknowledge the NIH Clinical Center as the provider of the ChestXray14 data. ChatGPT (OpenAI) was used to assist with writing and Claude Opus 5.5 (Anthropic) to assist with coding; the authors reviewed and edited all AI-assisted output and take full responsibility for the content of this paper.

## 8. REFERENCES

[1] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba, “Learning deep features for discriminative localization,” in Proc. IEEE CVPR, 2016, pp. 2921–2929.

[2] Pedro O. Pinheiro and Ronan Collobert, “From image-level to pixel-level labeling with convolutional networks,” in Proc. IEEE CVPR, 2015, pp. 1713–1721.

[3] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M. Summers, “ChestXray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases,” in Proc. IEEE CVPR, 2017, pp. 2097–2106.

[4] Adriel Saporta et al., “Benchmarking saliency methods for chest X-ray interpretation,” Nat. Mach. Intell., vol. 4, no. 10, pp. 867–878, 2022.

[5] Zhe Li et al., “Thoracic disease identification and localization with limited supervision,” in Proc. IEEE/CVF CVPR, 2018, pp. 8290–8299.

[6] Daniel Coelho de Castro et al., “PadChest-GR: A bilingual chest X-ray dataset for grounded radiology report generation,” NEJM AI, vol. 2, no. 7, 2025.

[7] Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vaya, “PadChest: A large chest x-ray im-´ age dataset with multi-label annotated reports,” Med. Image Anal., vol. 66, pp. 101797, 2020.

[8] Benedikt Boecking et al., “Making the most of text semantics to improve biomedical vision–language processing,” in Proc. ECCV, 2022, vol. 13696 of LNCS, pp. 1–21.

[9] Chenyu Lian, Hong-Yu Zhou, Chun-Ka Wong, and Jing Qin, “Concept-guided noisy negative suppression for zero-shot classification and grounding of chest X-ray findings,” arXiv preprint arXiv:2605.19374, 2026.

[10] Shruthi Bannur et al., “MAIRA-2: Grounded radiology report generation,” arXiv preprint arXiv:2406.04449, 2024.

[11] Philip Muller, Georgios Kaissis, and Daniel Rueckert, “ChEX:¨ Interactive localization and region description in chest X-rays,” in Proc. ECCV, 2024.

[12] Andrew Sellergren et al., “MedGemma technical report,” arXiv preprint arXiv:2507.05201, 2025.

[13] Yiwu Zhong et al., “RegionCLIP: Region-based languageimage pretraining,” in Proc. IEEE/CVF CVPR, 2022, pp. 16772–16782.

[14] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[15] Vladimir Vapnik and Akshay Vashist, “A new learning paradigm: Learning using privileged information,” Neural Netw., vol. 22, no. 5–6, pp. 544–557, 2009.

[16] David Lopez-Paz, Leon Bottou, Bernhard Sch´ olkopf, and¨ Vladimir Vapnik, “Unifying distillation and privileged information,” in Proc. ICLR, 2016.

[17] Chakkrit Termritthikun, Ayaz Umer, Suwichaya Suwanwimolkul, Feng Xia, and Ivan Lee, “Explainable knowledge distillation for on-device chest X-ray classification,” IEEE/ACM Trans. Comput. Biol. Bioinform., vol. 21, no. 4, pp. 846–856, 2024.

[18] Supriyo Sadhya and Xiaojun Qi, “X-Plain: Vision-language knowledge distillation for interpretable chest X-ray diagnosis,” in Proc. IEEE ISBI, 2026, pp. 1–5.

[19] Stephane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell,´ “A reduction of imitation learning and structured prediction to no-regret online learning,” in Proc. AISTATS, 2011, vol. 15 of PMLR, pp. 627–635.

[20] Rishabh Agarwal et al., “On-policy distillation of language models: Learning from self-generated mistakes,” in Proc. ICLR, 2024.

[21] Mengde Xu et al., “End-to-end semi-supervised object detection with soft teacher,” in Proc. IEEE/CVF ICCV, 2021, pp. 3040–3049.

[22] Mingxing Tan and Quoc V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in Proc. ICML, 2019, vol. 97 of PMLR, pp. 6105–6114.

[23] Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh, “Calibrate before use: Improving few-shot performance of language models,” in Proc. ICML, 2021, vol. 139 of PMLR, pp. 12697–12706.

[24] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” in Proc. 3DV, 2016, pp. 565– 571.