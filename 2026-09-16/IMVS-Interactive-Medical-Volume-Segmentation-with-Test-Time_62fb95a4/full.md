# IMVS: Interactive Medical Volume Segmentation with Test-Time Adaptation - A New Method for Annotating Radiology Datasets<sup>⋆</sup>

Abhilaksh Singh Reen<sup>⋆⋆,1(B)</sup>, Kushal Borkar<sup>⋆⋆,2</sup>, and Ritvik Mahapatra<sup>⋆⋆,3</sup>

<sup>1</sup> Independent Researcher, Delhi, India abhilakshsinghreen@gmail.com

<sup>2</sup> Independent Researcher, Kolkata, India kushalborkar.11@gmail.com

3 California State University, Fresno, CA, USA ritvik123@mail.fresnostate.edu

Abstract. Constructing large annotated radiology datasets is bottlenecked by the manual efort of delineating structures slice-by-slice in 3D volumes. Interactive methods reduce this efort but stay interactionineficient: slice-wise methods (including many foundation models) ignore inter-slice continuity, while 3D and video-based methods propagate a prompt with a fixed propagator that never adapts to the target volume, so it drifts on low-contrast or pathological structures and must be re-prompted. We present IMVS, a human-in-the-loop annotation framework that, rather than a new segmentation primitive, composes three components into a closed loop: a lightweight 2D Slice Mask Adapter (SMA) fine-tuned online from user scribbles, a frozen Volume Mask Tracker (VMT) that propagates corrected masks across adjacent slices, and a soft teacher–student alignment that limits forgetting. The SMA is backbone-agnostic (UNet++, DeepLabV3, TransUNet). Across 8 public CT/MRI datasets, IMVS matches strong interactive baselines in quality while substantially cutting annotation efort: 14.4× faster than a proficient copy-based manual workflow (22.3× over naive manual), 4.6× over slice-wise and 1.9× over 3D interactive methods. Foundation models such as MedSAM2 and ScribblePrompt stay competitive or stronger on well-delineated organs; IMVS’s advantage is largest on challenging targets and on overall interaction eficiency. Source code and Demo Video: https://github.com/AbhilakshSinghReen/imvs.

Keywords: Human-in-the-Loop Annotation · Interactive Segmentation · Annotation Eficiency · Test-Time Adaptation · Medical Image Analysis

## 1 Introduction

Interactive segmentation frameworks, including foundation models such as SAM [6] MedSAM [13], and ScribblePrompt [22], reduce annotation efort by taking human prompts (boxes [24], clicks [16, 17, 11], or scribbles [23, 1]) directly into the inference loop. Applied to volumes, slice-wise methods refine individual masks well but ignore inter-slice continuity, producing redundant interactions across a volume. Recent 3D and video-based interactive methods: PRISM [8], nnInteractive [5], and video-propagation systems such as SAM2/MedSAM2, instead, carry a prompt through the volume, but rely on a fixed, pre-trained propagator that is never adapted to the volume being annotated. Under distribution shift, low contrast, and pathology typical of clinical targets, such static propagators drift and must be re-prompted, so efort is still spent correcting the same kinds of errors repeatedly.

Motivated by the temporal consistency exploited in Video Object Segmentation (VOS), we build a human-in-the-loop framework, IMVS, that likewise propagates masks across adjacent slices but unlike the fixed 3D propagators above, IMVS couples propagation with online adaptation: each correction fixes the current slice and updates a lightweight per-slice model for the slices ahead, so an interaction is reused over many slices instead of being repeated on each one, and the system stops repeating errors rather than merely re-propagating them. In the taxonomy of interactive medical segmentation [15], IMVS is scribble-driven and iteratively refined, distinguished by where human efort is spent: it amortizes corrections over a volume through learned propagation and online adaptation. We therefore position it not as a new segmentation primitive but as a human-AI collaboration system for annotation eficiency, whose building blocks: interactive 2D segmentation, teacher-student test-time adaptation [18], soft alignment against forgetting [7], and attention-based propagation - are adapted from established paradigms; the contribution is how they compose into an eficient closed loop.

Concretely, we provide: (1) a closed-loop pipeline that couples a lightweight, online-adapted 2D Slice Mask Adapter (SMA) with a frozen Volume Mask Tracker (VMT), confining backpropagation to a small network while a fixed propagator enforces volumetric consistency; (2) a soft teacher-student alignment enabling continual refinement from user scribbles without catastrophic forgetting across slices and volumes; and (3) an empirical study on 8 CT/MRI datasets with a 9-annotator user study, quantifying annotation-eficiency gains and characterizing where the approach helps and where it does not. The SMA is backboneagnostic (validated on UNet++, DeepLabV3, TransUNet).

## 2 Related Work

Interactive methods refine masks from user inputs: boxes [24], clicks [16, 17, 11, 12, 2], or scribbles [23, 1]; f-BRS [16] and RiTM [17] introduced iterative inference-time refinement, and iSegFormer [11] a transformer backbone for medical images. Foundation models (SAM [6], MedSAM [13, 14]) generalize across targets but degrade under medical distribution shift and are typically applied per-slice; lighter 3D approaches such as PRISM [8] and promptable systems like nnInteractive [5] address volumes more directly. We refer to Marinov et al. [15] for a systematic taxonomy; within it IMVS is scribble-driven and iterativelyrefined, distinguished by amortizing interactions across slices rather than maximizing single-mask accuracy [21, 10].

![](images/a3a079336b7df0772955899d9502f8e126583473957c35ac8273999a005d33e2.jpg)  
Fig. 1. Overview of the IMVS closed annotation loop. The SMA predicts a mask the user refines with scribbles (teacher–student online adaptation); the frozen VMT propagates the corrected mask across adjacent slices via long/short-term attention and a memory bank.

Test-time and continual adaptation avoid target-domain retraining using uncertainty [18, 9] or teacher pseudo-labels [20]; mean-teacher formulations are effective for continual TTA [3] but are unsupervised. IMVS difers by being interactive and weakly supervised: the dominant signal is the user’s corrective scribble, with teacher/propagation terms only stabilizing, which counteracts the pseudolabel drift to which unsupervised teacher-student adaptation is prone [20], while soft alignment guards against forgetting [7, 4]. Propagating an annotation to neighbouring slices is long-standing—Wang et al. [19] used online random forests for placental segmentation, and VOS trackers later formalized memory/attentionbased propagation. IMVS shares this intuition and, like modern VOS trackers, uses a learned attention-based propagator with a memory bank: pretrained on video object segmentation [25] and fine-tuned for medical propagation. What distinguishes IMVS is that this propagator is deliberately kept lightweight and frozen during the annotation loop, with its errors recovered by online SMA adaptation rather than by a heavier tracker. This division of labour - a fixed propagator plus an adaptive per-slice model, is the main design choice.

## 3 Method

Problem setting and loop. IMVS couples a lightweight 2D SMA, fine-tuned online from scribbles, with a frozen VMT propagator, confining backpropagation to the small SMA while the fixed propagator enforces volumetric consistency. Let $f _ { \theta _ { 0 } }$ be a model pre-trained (parameters $\theta _ { 0 } )$ on source data $D _ { s }$ . At step t the model sees a slice $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { H \times W }$ , the previous slice/mask $( x _ { t - 1 } , \hat { y } _ { t - 1 } )$ , and (on intervention) scribbles $S _ { t } .$ , and the user refines the prediction into an accepted mask $m _ { t } ;$ the goal is to segment an unseen volume with as few interactions as possible without forgetting $D _ { s }$ . The annotator begins at the slice $x _ { 0 }$ where the target first becomes clearly visible - a protocol choice (mirroring how experts scroll to a structure’s onset), not a tuned hyperparameter. Because the loop re-seeds from any corrected slice, a diferent start costs at most one early correction, not final quality. From $x _ { 0 }$ the framework alternates an Interactive Mode (scribbles drive online SMA adaptation) and an Automatic Mode (the VMT propagates the accepted mask forward for up to K slices. We use $K { = } 1 7$ , trading propagation error against correction frequency. Before a correction is required; K is justified empirically in Sec. 4.4). Algorithm 1 summarizes the loop for one volume: the user intervenes only when a propagated mask is unsatisfactory, and each correction fixes the current slice and adapts the student for the slices ahead.

## 3.1 Slice Mask Adapter (SMA)

The SMA is a standard, interchangeable 2D segmentation network; we evaluate UNet++, DeepLabV3+, and TransUNet, and adopt UNet++ as default. It is instantiated as a teacher–student pair: a frozen teacher $f _ { \theta ^ { T } }$ and an adaptive student $f _ { \theta ^ { S } }$ , both initialized from the same weights $\theta ^ { T } = \theta ^ { S } = \theta _ { 0 }$ . The teacher preserves source-domain knowledge; only the student is updated online. The teacher and the VMT are never updated at test time.

How scribbles enter (early fusion). Foreground/background scribbles $\boldsymbol { S _ { t } } = \{ s _ { t } ^ { f g } , s _ { t } ^ { b g } \}$ are rasterized (dilation radius 3 px) into two binary guidance channels $M _ { f g } , M _ { b g } \in \{ 0 , 1 \} ^ { H \times }$ , set to 1 on foreground/background strokes respectively, and concatenated with the grayscale slice at the input, giving a 3- channel input $( \mathrm { C T } / \mathrm { M R I } + M _ { f g } + M _ { b g } )$ to the SMA. Interactions are thus fused early, keeping them backbone-agnostic.

## 3.2 Volume Mask Tracker (VMT)

The VMT maps $( x _ { t } , m _ { t - 1 } , \mathcal { M } ) \mapsto \hat { y } _ { t } ^ { V M T }$ , propagating the accepted mask of slice t−1 onto slice t using a memory bank M of recent (feature, mask) pairs. Its encoder is a ViT-B/16 pretrained on video object segmentation (YouTube-VOS 2019 [25]) and fine-tuned on medical volumes (Sec. 3.5), producing

$$
F _ { t } ^ { e n c } = \mathrm { E n c o d e r } ( x _ { t } ) \in \mathbb { R } ^ { H / 1 6 \times W / 1 6 \times D } , \qquad D = 7 6 8 .\tag{1}
$$

The previous mask is encoded to the same resolution (a 1×1 convolution on the resized mask) and concatenated with slice features to inject temporal, objectspecific context. A transformer with L = 6 Long+Short-Term Attention (LSTA) layers applies scaled dot-product attention in two scopes: short-term over the two preceding slices (frame-to-frame smoothness) and long-term over a memory bufer of the six most recent slices (appearance consistency and drift prevention). A decoder upsamples propagated mask $\hat { y } _ { t } ^ { V M T }$ . VMT is frozen at inference.

Algorithm 1 IMVS interaction loop for one volume   
Require: volume $\overline { { \{ x _ { i } \} ; \operatorname { S M A } \ f _ { \theta _ { 0 } } ; } }$ frozen VMT $^ { g ; }$ threshold $\gamma ;$ span K   
1: $\bar { \theta } ^ { T } , \theta ^ { S }  \theta _ { 0 } ;$ user seeds $x _ { 0 }$ with $ { \boldsymbol { S } } _ { 0 }$ (Interactive Mode); refine and update $\theta ^ { S } ;$   
m<sub>0</sub> $\gets \hat { y } _ { 0 } ^ { c }$ <sup>orrected</sup>; push to $\mathcal { M }$   
2: for $t = 1 , 2 , \dots$ until the volume is covered do   
3: $\hat { y } _ { t } ^ { V M T } \gets g ( x _ { t } , m _ { t - 1 } , \mathcal { M } )$ ▷ automatic propagation   
4: if user accepts $\hat { y } _ { t } ^ { V M T }$ and span since correction $< K$ then   
5: m<sub>t</sub> $\gets \hat { y } _ { t } ^ { \dot { V } M T }$   
6: else ▷ correct and adapt   
7: user draws $S _ { t } \mathrm { { ; \ i f \ } } \mathcal { L } _ { t o t a l } > \gamma$ update $\theta ^ { S } ~ ( \mathrm { E q . ~ 3 } ) ; m _ { t }  \hat { y } _ { t } ^ { c }$ orrected   
8: end if   
9: push $( F _ { t } , m _ { t } )$ to M, evict oldest   
10: end for   
11: carry $\theta ^ { S }$ to the next volume ▷ warm-start

## 3.3 Online Adaptation Objective

We call this test-time adaptation to indicate when the SMA is updated (at inference, on the target volume), not unsupervised adaptation [18, 3]: the dominant signal is the user’s corrective scribble, with propagated/teacher pseudo-labels only for stabilization. At each interactive step the student minimizes three terms. (A) A consistency term distills the frozen VMT mask and the augmentationaveraged teacher prediction $\hat { y } _ { t } ^ { a u g }$ into the student $\hat { y } _ { t } ^ { S }$

$$
\mathcal { L } _ { 1 } = - \sum _ { c } \big [ \beta _ { 1 } \hat { y } _ { t } ^ { V M T } [ : , c ] \log ( \hat { y } _ { t } ^ { S } [ : , c ] ) + \beta _ { 2 } \hat { y } _ { t } ^ { a u g } [ : , c ] \log ( \hat { y } _ { t } ^ { S } [ : , c ] ) \big ] ,\tag{2}
$$

with class index c and weights $\beta _ { 1 } , \beta _ { 2 }$ biased toward VMT guidance. (B) An interactive cross-entropy term $\begin{array} { r } { \mathcal { L } _ { 2 } = - \sum _ { c } \hat { y } _ { t } ^ { c o r r e c t e d } [ : , c ] \log ( \hat { y } _ { t } ^ { S } [ : , c ] ) } \end{array}$ matches the user-corrected mask $\hat { y } _ { t } ^ { c o r r e c t e d }$ , and (C) a soft alignment term prevents forgetting by anchoring the student’s batch-norm (BN) parameters to the teacher, $\begin{array} { r } { \dot { \mathcal { L } } _ { r e g } ( \dot { \theta _ { t } } ) = \sum _ { l } \mathcal { k } [ \bar { l } \in \mathrm { B N } ] \| \theta _ { l } ^ { S } - \theta _ { l } ^ { T } \| _ { 2 } ^ { 2 } } \end{array}$ . The total objective is

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { 1 } + \mathcal { L } _ { 2 } + \kappa \mathcal { L } _ { r e g } .\tag{3}
$$

The student is updated by SGD only when $\mathcal { L } _ { t o t a l } > \gamma = 0 . 0 5$ (suppressing noisy updates), with $\kappa = 0 . 1$ . Freezing the propagator prevents pseudo-label drift from compounding; the adapted student is carried to subsequent volumes as an eficiency warm-start (supervision stays user-driven, no bias to annotation).

## 3.4 Human-AI Interaction Workflow

IMVS is designed around iterative human-AI collaboration rather than one-shot segmentation. The annotator intervenes only when propagated masks become unacceptable; each correction is immediately incorporated by the adaptive SMA and reused across subsequent slices through propagation, a capability missing in existing methods. Thus, user input serves not only as local error correction but also as guidance that adapts future model behavior, reducing repeated corrections within the same session. This shifts the annotator’s role from refining every slice to supervising and steering an adaptive segmentation assistant.

Table 1. Annotation eficiency per method and dataset, shown as time (minutes) with # interactions in parentheses. Manual baselines have no discrete interactions (−). Manual With Copy is a proficient copy-and-edit workflow (the realistic manual reference). Best (lowest) time and interactions per column in bold.
<table><tr><td>Method</td><td>CHAOS CT</td><td>CHAOS MRI</td><td>AMOS CT</td><td>MSD Prostate</td><td>LiTS</td><td>MSD Hep.Ves.</td><td>MSD Pancreas</td><td>BraTS</td></tr><tr><td>Manual</td><td>44.79 (−)</td><td>21.5 (−)</td><td>65.5 (−)</td><td>8.53 (−)</td><td>73.53 (−)</td><td>96.3 (−)</td><td>8.11 (−)</td><td>55.26 (−)</td></tr><tr><td>Manual w/ Copy</td><td>28.83 (−)</td><td>13.68 (−)</td><td>42.79 (−)</td><td>5.4 (−)</td><td>47.56 (一)</td><td>61.85 (−)</td><td>5.26 (−)</td><td>35.52 (−)</td></tr><tr><td>f-BRS</td><td>7.23 (10)</td><td>3.41 (9)</td><td>10.55 (34)</td><td>1.64 (16)</td><td>17.57 (27)</td><td>22.8 (126)</td><td>1.59 (23)</td><td>11.49 (30)</td></tr><tr><td>Med SAM</td><td>12.61 (13)</td><td>6.02 (11)</td><td>18.51 (31)</td><td>2.39 (18)</td><td>20.51 (28)</td><td>28.25 (122)</td><td>2.27 (22)</td><td>15.55 (30)</td></tr><tr><td>ScribblePrompt</td><td>7.79 (20)</td><td>3.65 (12)</td><td>11.36 (29)</td><td>1.45 (21)</td><td>12.68 (25)</td><td>18.4 (121)</td><td>1.39 (24)</td><td>9.41 (29)</td></tr><tr><td>iSegFormer</td><td>7.12 (8)</td><td>3.36 (10)</td><td>10.42 (10)</td><td>1.35 (6)</td><td>11.58 (10)</td><td>15.4 (32)</td><td>1.32 (8)</td><td>8.73 (9)</td></tr><tr><td>PRISM</td><td>5.49 (6)</td><td>2.65 (4)</td><td>8.06 (9)</td><td>1.07 (3)</td><td>3.93 (5)</td><td>14.7 (28)</td><td>1.03 (3)</td><td>6.79 (8)</td></tr><tr><td>MedSAM2</td><td>1.89 (3)</td><td>0.96 (2)</td><td>2.74 (3)</td><td>0.93 (3)</td><td>4.17 (5)</td><td>11.3 (19)</td><td>0.76 (3)</td><td>3.1 (4)</td></tr><tr><td>nnInteractive</td><td>1.94 (4)</td><td>0.965 (3)</td><td>2.81 (4)</td><td>0.7 (3)</td><td>3.705 (4)</td><td>11 (23)</td><td>0.61 (4)</td><td>2.75 (5)</td></tr><tr><td>Ours (UNet++)</td><td>1.99 (4)</td><td>0.97 (3)</td><td>2.88 (4)</td><td>0.47 (2)</td><td>3.24 (4)</td><td>10.7 (20)</td><td>0.46 (2)</td><td>2.4 (3)</td></tr></table>

## 3.5 Training and Implementation

Data / zero-shot split. Both networks are trained only on stratified 70:15:15 splits of three datasets (BraTS, LiTS, MSD Pancreas; the “consolidated” set); the other five (CHAOS-CT/MRI, AMOS-CT, MSD Prostate, MSD HepaticVessel) are never seen in training and evaluated zero-shot, so cross-dataset results measure generalization. SMA: ImageNet-initialized backbones, slices $2 5 6 \times 2 5 6$ 110 epochs (AdamW, lr $1 \times 1 0 ^ { - 4 }$ , batch 16), loss $\mathcal { L } _ { D i c e } + \mathcal { L } _ { C E } ;$ training scribbles simulated from ground truth (boundary default). Online adaptation uses SGD (momentum 0.9, lr $1 \times 1 0 ^ { - 4 } )$ with horizontal-flip and intensity-jitter test-time augmentation on the teacher. VMT: the ViT-B propagator is pretrained on YouTube-VOS 2019 [25] (natural-video object segmentation) and then fine-tuned as a mask propagator on the consolidated set for 58 epochs (Adam, lr $2 \times 1 0 ^ { - 4 }$ batch 8) with a CE+Dice loss between propagated mask and ground truth, then frozen at inference. Hardware: a single NVIDIA RTX 5090 (CUDA 12.8).

Training uses simulated scribbles. For training only, metrics use a deterministic simulator that reacts to the current error region until the stopping criterion is met. This is reproducible and annotator-independent.

## 4 Experiments and Results

We evaluate across 8 CT/MRI benchmark datasets, focusing on challenging cases (liver tumors, pancreatic neoplasms, hepatic vasculature); training uses only the BraTS/LiTS/MSD-Pancreas splits, and the other five datasets are held out zero-shot. We benchmark seven interactive methods: MedSAM, MedSAM2, ScribblePrompt, PRISM, f-BRS, iSegFormer, and nnInteractive [5]. Interaction modality is heterogeneous (MedSAM/MedSAM2 and PRISM use boxes/clicks; ScribblePrompt, f-BRS, iSegFormer, and IMVS use scribbles/clicks; nnInteractive supports all three). We account for this when attributing the eficiency gains.

## 4.1 User Study and Stopping Criterion

Annotation times were measured in a user study with nine residents, reflecting a realistic annotator population for radiology dataset construction under expert supervision. Participants had varying experience and evaluated all methods using same acceptance criterion: advancing once the segmentation was qualitatively satisfactory. Aggregated accepted masks defined the automatic stopping threshold (DSC = 0.885, NSD = 0.894, HD95 = 4 mm). Inter-annotator agreement was high (mean pairwise $\mathrm { D S C } = 0 . 8 9 \pm 0 . 0 4 )$ , with consistent acceptance thresholds across raters $( \mathrm { D S C } = 0 . 8 8 5 \pm 0 . 0 3 )$ , supporting its use as a shared comparison rule. The outcome of the study (Table 1) is human workload reduction - both in time as well as number of interactions.

## 4.2 Annotation Eficiency

Time and interactions. Table 1 reports both annotation time and interaction counts. IMVS is fastest on 5 of 8 datasets and fastest in aggregate, but not uniformly best. MedSAM2 is faster on the well-delineated healthy organs of CHAOS-CT/MRI and AMOS-CT, this is reported in the per-dataset study rather than being averaged away. On interaction count, IMVS is most eficient on 4 of 8 datasets (2 on MSD Prostate/Pancreas, 3 on BraTS, 4 on LiTS) and stays within one interaction of MedSAM2 elsewhere. The tortuous hepatic vessels are the hard case for every method: IMVS still needs 20 interactions there (vs. 19 for MedSAM2 and 28-126 for the slice-wise baselines), consistent with our claim that the approach helps least on discontinuous structures.

Sources of the gain. We state speedups conservatively: IMVS is 22.3× faster than naive Manual annotation but 14.4× faster than the fairer Manual With Copy (copy-and-edit the previous mask). Part of the per-interaction gain over click/box baselines is modality (a scribble carries more information than a click); comparing against scribble baselines under the identical criterion isolates the algorithmic contribution: amortizing interactions over a propagated window plus online SMA adaptation. IMVS also uses the least compute (3.23 GB VRAM, 15.93 s GPU/volume). A paired Wilcoxon signed-rank test (per volume, Med-SAM2) locates the real advantage: significant over the four challenging datasets (LiTS, MSD Pancreas, MSD HepaticVessel, BraTS; $p = 0 . 0 2 )$ but similar over the well-delineated ones (CHAOS-CT/MRI, AMOS, MSD Prostate; $p = 0 . 2 6 )$

## 4.3 Segmentation Quality

Table 2 complements Table 1: per dataset, how many interactions each method needs to reach a demanding target $( \mathrm { D S C } \ge 0 . 9 0$ , near the operating point). IMVS needs the fewest on every dataset: 7.1 on average vs. 10.1 for MedSAM2 and 13.1 for slice-wise iSegFormer (1.4–1.8× fewer). Final quality is closely clustered once enough interactions are spent, so IMVS’s contribution is reaching it sooner, not exceeding the plateau. IMVS thus matches strong baselines in accuracy while being far more interaction and VRAM-eficient; it is not claimed best in raw quality on every dataset. Robustness: On tortuous hepatic vessels the VMT tracks reliably for only ∼4–6 slices (vs. up to 17 on well-defined organs), so more corrections are needed. Every method degrades here. IMVS stays competitive because online SMA adaptation recovers tracking with typically a single scribble.

Table 2. Interaction eficiency at matched quality: number of user interactions needed to reach $\mathrm { D S C } \geq 0 . 9 0$ (near the stopping operating point) per dataset; lower is better. This complements Table 1 (efort to satisfy the stopping criterion). IMVS reaches the target with the fewest interactions on every dataset.
<table><tr><td>Method</td><td>CHAOS-CT</td><td>CHAOS-MRI</td><td>AMOS</td><td>Prostate</td><td>LiTS</td><td>Pancreas</td><td>BraTS</td><td>Mean</td></tr><tr><td>iSegFormer</td><td>13</td><td>14</td><td>13</td><td>12</td><td>11</td><td>15</td><td>13.5</td><td>13.1</td></tr><tr><td>PRISM</td><td>12</td><td>13</td><td>12</td><td>11</td><td>10</td><td>14</td><td>12.5</td><td>12.1</td></tr><tr><td>MedSAM2</td><td>10</td><td>11</td><td>10</td><td>9</td><td>8</td><td>12</td><td>10.5</td><td>10.1</td></tr><tr><td>Ours</td><td>7</td><td>8</td><td>7</td><td>6</td><td>6</td><td>8</td><td>7.5</td><td>7.1</td></tr></table>

Table 3. Ablation of adaptation strategies on the consolidated set (per-volume averages). SMA (adaptive), VMT frozen is our method; Tracked Slices is the average span before a correction is needed.

<table><tr><td>Approach</td><td># Int. # Tracked Dice NSD HD95 Max VRAM GPU Time</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No SMA, VMT frozen</td><td>18</td><td>3</td><td>0.86</td><td>0.87</td><td>2.7</td><td>1.7 GB</td><td>4ms</td></tr><tr><td>Ours (Memory Size 6)</td><td>4</td><td>12</td><td>0.92</td><td>0.91</td><td>1.6</td><td>3.23 GB</td><td>129ms</td></tr><tr><td>SMA (adaptive), VMT adapted</td><td>8</td><td>7</td><td>0.92</td><td>0.91</td><td>1.6</td><td>20.7 GB</td><td>1857ms</td></tr></table>

## 4.4 Ablation: Contribution of Each Component

Table 3 isolates the components on the consolidated set. Online SMA adaptation is the largest factor: without it (No SMA, VMT frozen) the loop needs 18 interactions at DSC 0.86; adding it (our method) cuts interactions to 4 at DSC 0.92 and raises the tracked span from 3 to 12 slices. Freezing the VMT is essential: adapting it as well gives no accuracy gain but 6× the VRAM, 14× the latency, and a shorter tracked span (caused by temporal drift). Among backbones, UNet++ beats DeepLabV3/TransUNet across all metrics (dense skip connections aid boundary refinement under sparse scribbles) and is the default; a memory-bank size of 6 is optimal (4–5 degrade tracking, 7–9 add compute without gains). The propagation length K=17 is likewise chosen from a sweep $K \in \{ 5 , \ldots , 2 5 \}$ : smaller K forces frequent corrections (7.2 interactions at K=5 vs. 4.1 at $K { = } 1 7 )$ , while larger K raises the propagation failure rate (29% at K=17 to 55% at K=25) as masks drift.

## 5 Scope and Limitations

IMVS segments a single target per pass; multi-label volumes use sequential single-label sessions sharing one adaptation context, and native multi-object propagation is future work. Eficiency comes from amortizing interactions over a propagated span, so IMVS helps least on tortuous or discontinuous structures, where more interactions are needed.

## 6 Conclusion

We have presented IMVS: a human-in-the-loop framework that composes an online-adapted 2D SMA, a frozen VMT propagator, and soft teacher-student alignment into a closed annotation loop. Its benefit is annotation eficiency: propagating each correction across slices and adapting the SMA online cuts interactions, time, and VRAM. We view IMVS as a new method in human-AI collaboration for eficient dataset construction. Future extensions include topologyaware and multi-object propagation.

## References

1. Agustsson, E., Uijlings, J.R., Ferrari, V.: Interactive full image segmentation by considering all regions jointly. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11622–11631 (2019)

2. Chen, X., Zhao, Z., Zhang, Y., Duan, M., Qi, D., Zhao, H.: Focalclick: Towards practical interactive image segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1300–1309 (2022)

3. Döbler, M., Marsden, R.A., Yang, B.: Robust mean teacher for continual and gradual test-time adaptation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7704–7714 (2023)

4. French, R.M.: Catastrophic forgetting in connectionist networks. Trends in cognitive sciences 3(4), 128–135 (1999)

5. Isensee, F., Rokuss, M., Krämer, L., Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., Maier-Hein, K.: nninteractive: Redefining 3d promptable segmentation. arXiv preprint arXiv:2503.08373 (2025)

6. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.Y., et al.: Segment anything. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4015–4026 (2023)

7. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al.: Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences 114(13), 3521–3526 (2017)

8. Li, H., Liu, H., Hu, D., Wang, J., Oguz, I.: Prism: A promptable and robust interactive segmentation model with visual prompts. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 389–399. Springer (2024)

9. Liang, J., Hu, D., Feng, J.: Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation. In: International conference on machine learning. pp. 6028–6039. PMLR (2020)

10. Lin, D., Dai, J., Jia, J., He, K., Sun, J.: Scribblesup: Scribble-supervised convolutional networks for semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3159–3167 (2016)

11. Liu, Q., Xu, Z., Jiao, Y., Niethammer, M.: isegformer: interactive segmentation via transformers with application to 3d knee mr images. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. pp. 464–474. Springer (2022)

12. Liu, Q., Zheng, M., Planche, B., Karanam, S., Chen, T., Niethammer, M., Wu, Z.: Pseudoclick: Interactive image segmentation with click imitation. In: European Conference on Computer Vision. pp. 728–745. Springer (2022)

13. Ma, J., He, Y., Li, F., Han, L., You, C., Wang, B.: Segment anything in medical images. Nature Communications 15(1), 654 (2024)

14. Ma, J., Kim, S., Li, F., Baharoon, M., Asakereh, R., Lyu, H., Wang, B.: Segment anything in medical images and videos: Benchmark and deployment. arXiv preprint arXiv:2408.03322 (2024)

15. Marinov, Z., Jäger, P.F., Egger, J., Kleesiek, J., Stiefelhagen, R.: Deep interactive segmentation of medical images: A systematic review and taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(12), 10998–11018 (2024)

16. Sofiiuk, K., Petrov, I., Barinova, O., Konushin, A.: f-brs: Rethinking backpropagating refinement for interactive segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8623–8632 (2020)

17. Sofiiuk, K., Petrov, I.A., Konushin, A.: Reviving iterative training with mask guidance for interactive segmentation. In: 2022 IEEE International Conference on Image Processing (ICIP). pp. 3141–3145. IEEE (2022)

18. Wang, D., Shelhamer, E., Liu, S., Olshausen, B., Darrell, T.: Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726 (2020)

19. Wang, G., Zuluaga, M.A., Pratt, R., Aertsen, M., Doel, T., Klusmann, M., David, A.L., Deprest, J., Vercauteren, T., Ourselin, S.: Slic-seg: A minimally interactive segmentation of the placenta from sparse and motion-corrupted fetal mri in multiple views. Medical Image Analysis 34, 137–147 (2016)

20. Wang, Q., Fink, O., Van Gool, L., Dai, D.: Continual test-time domain adaptation. In: Proceedings of Conference on Computer Vision and Pattern Recognition (2022)

21. Wong, H.E., Rakic, M., Guttag, J., Dalca, A.V.: Scribbleprompt: Fast and flexible interactive segmentation for any medical image. arXiv e-prints pp. arXiv–2312 (2023)

22. Wong, H.E., Rakic, M., Guttag, J., Dalca, A.V.: Scribbleprompt: fast and flexible interactive segmentation for any biomedical image. In: European Conference on Computer Vision. pp. 207–229. Springer (2024)

23. Wu, J., Zhao, Y., Zhu, J.Y., Luo, S., Tu, Z.: Milcut: A sweeping line multiple instance learning paradigm for interactive image segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 256–263 (2014)

24. Xu, N., Price, B., Cohen, S., Yang, J., Huang, T.: Deep grabcut for object selection. arXiv preprint arXiv:1707.00243 (2017)

25. Xu, N., Yang, L., Fan, Y., Yue, D., Liang, Y., Yang, J., Huang, T.: Youtube-vos: A large-scale video object segmentation benchmark. arXiv preprint arXiv:1809.03327 (2018)