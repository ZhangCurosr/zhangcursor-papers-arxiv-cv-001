# Cross-Architecture Knowledge Distillation from a Vision Foundation Model to a Lightweight Visual State Space Model for Tea Leaf Disease Classification

Zibo Zhou, Zongsen Qiu, Rui Chen, Yujie Yao, Yue Zhou, Jianjun Wang

Abstract—Automated tea leaf disease classification supports precision agriculture, yet deploying accurate models on edge devices remains challenging under tight compute budgets. Selfsupervised vision foundation models such as DINOv2 provide strong features but are too large for field deployment, while lightweight models trained from scratch on small agricultural datasets often underfit. We study cross-architecture knowledge distillation (KD) from a fine-tuned DINOv2 teacher (a Vision Transformer) to a compact bidirectional Visual State Space Model (LVSSM) student, a direction that is comparatively underexplored because the two architectures use fundamentally different token-mixing mechanisms. We first identify and fix two training-stability problems that otherwise prevent our fromscratch SSM student from learning on limited data: a single large patch-embedding convolution and a fusion layer that severs the residual path. With a progressive convolutional stem and a gated bidirectional selective-scan block, the 4.45M-parameter student trains stably. Averaged over three seeds, temperature-scaled logit distillation raises test accuracy from 92.32±2.14% (standalone) to 95.41±1.17% (best single run 96.20%, macro-F1 94.45%), a +3.09 percentage-point mean gain that also reduces run-torun variance, while the model uses 5.0× fewer parameters than the 22M-parameter teacher and retains 98.3% of its accuracy. Through controlled ablations that hold the distillation weight fixed, we find that adding intermediate feature-alignment losses reduces accuracy, so simple logit-level KD is the strongest configuration for this ViT→SSM transfer. A fair from-scratch comparison shows the gain is specific to students that start below the teacher: the same recipe helps the SSM but not fromscratch CNNs that already match the teacher. We report per-class metrics, confusion matrices, bootstrap confidence intervals, and FLOPs/latency measurements, and we discuss the limitations of the study, including its single-dataset scope and the use of a simplified (non-official) SSM implementation.

Index Terms—Knowledge distillation, state space models, cross-architecture transfer, tea leaf disease classification, precision agriculture, edge deployment

## I. INTRODUCTION

Tea is among the most widely consumed beverages worldwide, and its yield and quality depend on timely detection of foliar diseases. Manual inspection is labor-intensive and subjective, which motivates automated image-based classification. While deep learning has advanced plant disease recognition substantially, deploying high-accuracy models on the resource-constrained edge devices typical of agricultural settings remains an open problem.

Self-supervised vision foundation models, particularly DI-NOv2 [1], achieve strong transfer accuracy across visual tasks.

However, even their smaller variants (∼22M parameters) are expensive for embedded inference. Conversely, compact architectures trained from scratch on small agricultural datasets frequently fail to learn discriminative features and generalize poorly.

State Space Models (SSMs), and the selective-scan Mamba formulation [2] in particular, offer linear-time sequence modeling and have been adapted to vision by Vision Mamba (Vim) [3] and VMamba [4]. Their favorable complexity makes SSMs attractive for edge deployment. However, training vision SSMs from scratch on small datasets is difficult: the bidirectional scan is prone to unstable gradient flow, and naive patch embeddings provide little local inductive bias.

Knowledge distillation (KD) [5] is a natural way to transfer the knowledge of a large teacher into a compact student. Most KD literature targets same-family transfer (CNN→CNN, ViT→ViT), where teacher and student share token-mixing mechanisms and intermediate representations can be aligned directly. Cross-architecture distillation between families with different inductive biases — here, a ViT teacher (global selfattention over patches) and an SSM student (sequential state propagation over a scanned patch order) — is less understood, and it is unclear whether feature-level alignment helps or hurts.

In this work we make the following contributions:

1) We diagnose two concrete failure modes that prevent a from-scratch bidirectional visual SSM from learning on a small dataset, and we fix them with a progressive convolutional stem and a gated-residual bidirectional selective-scan block. The resulting 4.45M-parameter student (which we call LVSSM, for Lightweight Visual State Space Model) trains stably to 92.16% test accuracy without any pretraining.

2) We study cross-architecture KD from a fine-tuned DI-NOv2 teacher to this SSM student. Averaged over three seeds, logit-level distillation improves student accuracy by +3.09 points (92.32%→95.41%; best single run 96.20%) and reduces variance, demonstrating that dark knowledge transfers effectively even across strongly dissimilar architectures.

3) Through controlled ablations with a fixed CE/KD weighting, we find that adding intermediate featurealignment losses does not improve over logit-only KD at this scale, and we analyze why. We report a complete evaluation (per-class metrics, confusion matrices, bootstrap confidence intervals, FLOPs and GPU/CPU latency) and discuss the study’s limitations honestly.

## II. RELATED WORK

## A. Deep Learning for Plant Disease Classification

CNN-based transfer learning from ImageNet-pretrained backbones (ResNet [6], EfficientNet [7], MobileNet [8]) is the dominant paradigm for plant disease recognition, and large curated datasets such as PlantVillage [9] have driven progress [10]. These approaches achieve high accuracy but depend on large-scale external pretraining and are typically benchmarked without explicit attention to deployment cost. Tea leaf disease datasets are comparatively small and classimbalanced, which makes both from-scratch training and fair efficiency comparison more delicate.

## B. Vision State Space Models

Mamba [2] introduced input-dependent (selective) state space parameters, giving linear-time sequence modeling with a hardware-aware scan. Vision Mamba [3] and VMamba [4] adapt selective scanning to images by serializing patches and scanning them bidirectionally or along multiple spatial orders. These models are efficient but, in our experience, require careful architectural choices (stem design, residual structure) to train stably on small datasets. Our student is a deliberately simplified, self-contained SSM block inspired by this line of work rather than the official mamba-ssm kernel; we therefore name it LVSSM to avoid overstating equivalence with Vim/VMamba.

## C. Knowledge Distillation

KD [5] transfers a teacher’s softened output distribution to a student. Feature-based extensions align intermediate representations, and review-based methods [11] match features across multiple stages. Most such methods assume architecturally compatible teacher–student pairs. Cross-architecture transfer — e.g., ViT→CNN or, as here, ViT→SSM — is less studied; the central difficulty is that intermediate features encode information in structurally different coordinate systems (spatial attention maps vs. scan-order state trajectories), so direct feature alignment may inject noise rather than signal. Our ablations speak directly to this question.

## III. METHODOLOGY

## A. Problem Formulation

Given a teacher T (fine-tuned DINOv2) and a student S (LVSSM), we train S to approximate $T \ ' _ { \mathbf { S } }$ predictive distribution while using far fewer parameters. Let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be the labeled training set of tea leaf images.

## B. Teacher Model: Fine-tuned DINOv2

We use DINOv2-Small (22.06M parameters) as the teacher, fine-tuned end-to-end on the tea disease dataset with a linear classification head over the class token. Under our unified evaluation pipeline the teacher attains 97.86% test accuracy and 96.76% macro-F1, serving as a strong reference.

## C. Student Model: Lightweight Visual SSM (LVSSM)

Our student (4,454,982 parameters) addresses two trainingstability failure modes we observed when training a bidirectional visual SSM from scratch.

1) Diagnosed failure modes: A baseline design using (i) a single 16×16 patch-embedding convolution and (ii) a bidirectional block whose forward/backward outputs are merged by a linear “fusion” layer followed by LayerNorm exhibited severe vanishing gradients: measured gradient norms decayed from ∼0.85 at the first block to ∼0.03 at the second and to near zero in deeper blocks, and the model stayed at chance accuracy $\left( \sim 1 / 6 \right)$ . We attribute this to (i) the single large convolution providing no local inductive bias before sequential scanning, and (ii) the fusion layer routing all information through a nonidentity transform, which destroys the residual pathway.

2) Progressive Convolutional Stem: We replace the single convolution with a 4-layer progressive stem of stride-2 convolutions,

$$
\mathrm { C o n v S t e m } : \mathbb { R } ^ { 3 \times H \times W } \to \mathbb { R } ^ { d \times \frac { H } { 1 6 } \times \frac { W } { 1 6 } } ,\tag{1}
$$

with channel widths $d / 4 \to d / 2 \to d \to d ,$ each followed by BatchNorm and GELU. This provides local feature extraction before sequential processing, mirroring the stems used in successful vision SSMs.

3) Gated-Residual Bidirectional Selective Scan: Each block scans the sequence forward and backward and combines the two directions through a learned gate while preserving an explicit identity path:

$$
\mathrm { \mathbf { c } = [ S S M _ { f w d } ( \mathbf { x } ) ; \nabla S S M _ { b w d } ( \mathbf { f l i p } ( \mathbf { x } ) ) ] , }\tag{2}
$$

$$
{ \bf g } = \sigma ( W _ { g } { \bf c } ) , \qquad { \bf o } = { \bf x } + { \bf g } \odot W _ { p } { \bf c } .\tag{3}
$$

The gate g learns how much bidirectional context to inject, while the additive residual guarantees gradient flow through the identity path. This single change restored stable training in all our runs.

4) Configuration: The student stacks 6 gated bidirectional SSM blocks with embedding dimension $\begin{array} { l l l } { d } & { = } & { 1 9 2 } \end{array}$ , state dimension 16, expansion factor 2, and convolution kernel size 4, producing 196 tokens (14 × 14) at 224 × 224 input and 4,454,982 parameters in total.

## D. Distillation Framework

1) Logit-level Distillation: The primary objective is temperature-scaled soft-target matching:

$$
\mathcal { L } _ { \mathrm { K D } } = \tau ^ { 2 } \cdot \mathrm { K L } ( \sigma ( z _ { s } / \tau ) \parallel \sigma ( z _ { t } / \tau ) ) ,\tag{4}
$$

where $z _ { s } , z _ { t }$ are student/teacher logits, τ is the temperature, and σ is the softmax. We use $\tau = 2 . 0 ;$ ; the $\tau ^ { 2 }$ factor keeps the KD gradient magnitude comparable to the cross-entropy term.

2) Combined Objective: The total loss combines groundtruth cross-entropy and KD:

$$
\begin{array} { r } { \mathcal { L } = \alpha _ { \mathrm { C E } } \mathcal { L } _ { \mathrm { C E } } + \alpha _ { \mathrm { K D } } \mathcal { L } _ { \mathrm { K D } } , } \end{array}\tag{5}
$$

with $\alpha _ { \mathrm { C E } } = \alpha _ { \mathrm { K D } } = 0 . 5$ in the main configuration. The ablation study (Section IV-E) additionally considers two optional feature-alignment terms: a scan-attention alignment loss $\mathcal { L } _ { \mathrm { S A A } }$ that matches a pooled teacher attention signal to a student scan-state summary, and a progressive feature-distillation loss $\mathcal { L } _ { \mathrm { P r o g } }$ that aligns projected intermediate features at several depths.

## E. Training Strategy

We use AdamW (weight decay 0.05), a peak learning rate of $5 \times 1 0 ^ { - 4 }$ , linear warmup (start factor 0.01) followed by cosine annealing to $1 0 ^ { - 6 }$ , and gradient clipping at norm 1.0. Augmentation comprises random resized crop, horizontal/vertical flips, color jitter, and mild rotation. The distilled students use batch size 12 (bounded by the bidirectional-scan memory footprint); the standalone student uses batch size 24. All models train for 30 epochs.

## IV. EXPERIMENTS

## A. Dataset and Setup

We evaluate on a real-world tea leaf disease dataset (Roboflow, CC BY 4.0) with six classes: Algal Leaf Spot, Brown Blight, Gray Blight, Healthy, Helopeltis, and Red Leaf Spot. The dataset provides a group-aware split by source photograph, which we verified: cross-split overlap of sourcephoto identifiers is exactly zero between train, validation, and test, so there is no data leakage. After converting the original detection-format annotations to classification labels (discarding 79 images with empty annotation files, i.e. no bounding box), the splits contain 4,259 training, 421 validation, and 421 test images at 224 × 224. The test set is class-imbalanced (Healthy: 192; the other five classes: 41–50), which is why we emphasize macro-F1 and per-class metrics rather than accuracy alone. All experiments run on a single NVIDIA RTX 4090.

## B. Implementation Details

The teacher (DINOv2-Small) is fine-tuned for 30 epochs at batch size 32. CNN baselines (ResNet18, EfficientNet-B0, MobileNetV3-Small) are ImageNet-pretrained and fine-tuned for 30 epochs. The LVSSM student is trained for 30 epochs as described above. To ensure comparability, every model is evaluated with a single shared pipeline that computes accuracy, macro/weighted-F1, per-class precision/recall/F1, confusion matrices, bootstrap 95% confidence intervals (2,000 resamples), FLOPs (via ptflops), and GPU/CPU single-image latency.

## C. Main Results

Table I summarizes the main results. Three observations stand out.

Cross-architecture KD is effective. For the best single run (seed 42), logit distillation improves the SSM student from 92.16% to 96.20% accuracy and from 89.82% to 94.45% macro-F1; averaged over three seeds the gain is +3.09 accuracy / +3.44 macro-F1 points (Table VI). Either way, the teacher’s softened outputs transfer useful “dark knowledge” despite the ViT→SSM architecture gap. Table I reports best-run values for each model; multi-seed means for the student are in Table VI.

TABLE I  
MAIN COMPARISON ON THE TEA LEAF DISEASE TEST SET (421 IMAGES), ALL UNDER ONE EVALUATION PIPELINE. VALUES ARE BEST SINGLE-RUN (SEED 42); STUDENT MULTI-SEED MEANS ARE IN TABLE VI. BEST STUDENT IN BOLD.
<table><tr><td>Model</td><td>Params (M)</td><td>Acc (%)</td><td>Macro-F1 (%)</td><td>Pretrain</td></tr><tr><td>DINOv2-S (Teacher)</td><td>22.06</td><td>97.86</td><td>96.76</td><td>SSL-142M</td></tr><tr><td>ResNet18</td><td>11.18</td><td>98.34</td><td>97.52</td><td>ImageNet</td></tr><tr><td>EfficientNet-B0</td><td>4.02</td><td>98.57</td><td>97.89</td><td>ImageNet</td></tr><tr><td>MobileNetV3-S</td><td>1.52</td><td>98.10</td><td>97.12</td><td>ImageNet</td></tr><tr><td>LVSSM (standalone)</td><td>4.45</td><td>92.16</td><td>89.82</td><td>None</td></tr><tr><td>LVSSM + Full distill</td><td>4.45</td><td>95.49</td><td>93.79</td><td>None</td></tr><tr><td>LVSSM + Logit KD</td><td>4.45</td><td>96.20</td><td>94.45</td><td>None</td></tr></table>

![](images/21d0e15836969500674f5025803a2cb098de6529ee31eb0954f00f3d84143482.jpg)  
Fig. 1. Validation accuracy over training. Distillation converges faster and to a higher plateau than standalone training.

The efficiency tradeoff is favorable. The distilled student uses 5.0× fewer parameters than the teacher and retains 98.3% of its accuracy (96.20 vs. 97.86).

Pretraining still matters. ImageNet-pretrained CNNs (98.1–98.6%) lead the from-scratch SSM student. This gap reflects large-scale external pretraining data unavailable to the student rather than architectural inferiority; the distilled SSM, trained only on 4,259 tea images, comes within ∼2.4 points of EfficientNet-B0.

## D. Per-Class Analysis and Error Structure

Table II shows that distillation helps most on the classes the standalone student handled worst: Helopeltis (+12.7 F1) and Red Leaf Spot (+8.6 F1). The dominant residual error, visible in the confusion matrix (Fig. 3), is mutual confusion between Brown Blight and Gray Blight (5 misclassifications in each direction) — two visually similar necrotic-lesion diseases that the teacher itself also confuses (Table II). This suggests the error is intrinsic to the visual similarity of the two diseases rather than a distillation artifact.

## E. Ablation: Does Feature Alignment Help?

We consider two optional feature-alignment terms on top of logit KD: a scan-attention alignment loss $( \mathcal { L } _ { \mathrm { { S A A } } } )$ and a progressive feature-distillation loss $( \mathcal { L } _ { \mathrm { P r o g } } )$ . A naive ablation that redistributes a fixed loss budget (lowering $\alpha _ { \mathrm { K D } }$ to make room for the new terms) confounds two effects — the feature term’s contribution and the weaker KD signal. We therefore run a controlled ablation that holds $\alpha _ { \mathrm { C E } } = \alpha _ { \mathrm { K D } } = 0 . 5$ fixed and adds each feature term at weight 0.5, changing nothing about the primary objective.

![](images/ee8bbffc24d75ad5fee5889a0d1c70aa1dad17813bc99bcc1ab27f7145aee30d.jpg)  
Fig. 2. Accuracy vs. parameters. Distillation lifts the SSM student (arrow) without any external pretraining data.

TABLE II  
PER-CLASS F1 (%) ON THE TEST SET. SUPPORT IN PARENTHESES.
<table><tr><td>Class (support)</td><td>Standalone</td><td>Logit KD</td><td>Teacher</td></tr><tr><td>Algal Leaf Spot (42)</td><td>96.39</td><td>95.35</td><td>98.82</td></tr><tr><td>Brown Blight (41)</td><td>85.00</td><td>87.80</td><td>91.57</td></tr><tr><td>Gray Blight (47)</td><td>86.32</td><td>88.42</td><td>92.47</td></tr><tr><td>Healthy (192)</td><td>96.62</td><td>99.22</td><td>99.74</td></tr><tr><td>Helopeltis (49)</td><td>84.21</td><td>96.91</td><td>98.97</td></tr><tr><td>Red Leaf Spot (50)</td><td>90.38</td><td>98.99</td><td>98.99</td></tr><tr><td>Macro-F1</td><td>89.82</td><td>94.45</td><td>96.76</td></tr></table>

Table III gives an unambiguous answer: even with the KD signal held at full strength, adding feature-alignment losses reduces accuracy (96.20%→94.77–95.49%). This rules out the “diluted KD weight” explanation — the feature terms are not merely neutral, they actively interfere. We attribute this to a representational mismatch: the SAA loss forces the student’s scan-state summary toward the teacher’s patch-attention geometry, and the progressive loss aligns intermediate features across a ViT and an SSM whose depth-wise representations are structurally different. Constraining the SSM to imitate ViTshaped intermediate features conflicts with the representations the scan mechanism would otherwise learn. Logit-only KD, which constrains only the output distribution, avoids this conflict and is the strongest configuration.

## F. Is Cross-Architecture KD Special to the SSM Student?

A natural question is whether the distillation gain reflects something specific about the SSM student or whether the same teacher would help any compact from-scratch model. To test this, we distill the same DINOv2 teacher into two CNN students (MobileNetV3-Small, EfficientNet-B0) trained from scratch (no ImageNet), using the identical logit-KD recipe $( \tau = 2 . 0 , \alpha _ { \mathrm { C E } } = \alpha _ { \mathrm { K D } } = 0 . 5 )$ . This is the apples-to-apples comparison: all three students start from random initialization.

![](images/e802ff81b472de0ee90062bcd474ff488f129b3495cd7942a6ecd3932ce83134.jpg)

Fig. 3. Confusion matrix of the distilled LVSSM (logit KD). The main residual error is Brown Blight ↔ Gray Blight.  
![](images/481b32c69bca1126dae7388b65506caabba5e30f7b41ba75ed84ae0f8811eb8e.jpg)  
Fig. 4. Per-class F1 before and after distillation. Gains concentrate on the hardest standalone classes.

Table IV shows a clear pattern: distillation from this teacher yields a large gain for the SSM student (+3.09 points, mean over seeds) but is neutral-to-negative for the CNN students (−1.19 and +0.24). The explanation is that the from-scratch CNNs already reach 96.4–97.2% standalone — at or above the teacher’s own 97.86% — so there is little useful “dark knowledge” left to transfer, and a not-strictly-stronger teacher can even mislead them. The SSM student, by contrast, starts well below the teacher and has substantial headroom. This clarifies the scope of our claim: cross-architecture KD is most valuable precisely when the compact student would otherwise underperform the teacher, as is the case for from-scratch vision SSMs on small data.

## G. Efficiency and Complexity

Table V reports the efficiency picture honestly. On parameters the distilled LVSSM (4.45M) is 5.0× smaller than the teacher. However, its measured latency is not competitive: the pure-PyTorch bidirectional selective scan runs at 10.05 ms on GPU and 1410 ms on CPU, far slower than the CNN baselines, because our simplified SSM does not use the hardware-aware fused scan kernel of the official Mamba implementation. Parameter count and FLOPs (2.19 GFLOPs) therefore understate real inference cost. Closing this gap requires a fused scan kernel and is important future work before any edgedeployment claim.

TABLE III  
CONTROLLED ABLATION: CE AND KD WEIGHTS HELD FIXED AT 0.5, FEATURE-ALIGNMENT TERMS added ON TOP (NOT TRADED AGAINST KD). IDENTICAL SCHEDULE, $\tau = 2 . 0$ , 30 EPOCHS.
<table><tr><td>Config</td><td>αCE</td><td>αKD</td><td>αSAA</td><td> $\alpha _ { \mathrm { P r o g } }$ </td><td>Acc (%)</td></tr><tr><td>No distillation</td><td>1.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>92.16</td></tr><tr><td>Logit-only</td><td>0.5</td><td>0.5</td><td>0.0</td><td>0.0</td><td>96.20</td></tr><tr><td>+ SAA</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.0</td><td>94.77</td></tr><tr><td>+ Progressive</td><td>0.5</td><td>0.5</td><td>0.0</td><td>0.5</td><td>95.49</td></tr><tr><td> $+ \ \mathrm { S A A } \ + \ \mathrm { P r o g }$ </td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>95.01</td></tr></table>

Ablation Study: Effect of Distillation Components  
![](images/f9a70a5533d04fba9f8ca159fd94d3f43801dfbcb19d6271f7c28bc26323d641.jpg)  
Fig. 5. Controlled ablation. With KD held at full weight, adding feature alignment reduces accuracy; logit-only KD is strongest.

## H. Multi-Seed Stability

Because a single-seed gain could partly reflect initialization noise, we repeat the standalone and logit-KD configurations across three random seeds (42, 123, 456) and report mean ± std (Table VI). The distillation gain is robust: averaged over seeds, logit KD improves accuracy by +3.09 points (92.32→95.41) and macro-F1 by +3.44 points (90.19→93.63). Distillation also reduces run-to-run variance (accuracy std 2.14→1.17), consistent with the teacher’s soft targets providing a more stable optimization signal than one-hot labels alone.

Note that the best single logit-KD run (seed 42, 96.20%) and the mean (95.41%) differ by less than one standard deviation; we report both, and use the mean for all crossmodel comparisons where variance matters.

## V. DISCUSSION

Why does logit KD transfer across such different architectures? Logit-level KD only constrains the student’s output distribution, leaving it free to develop internal representations suited to sequential scanning. Feature-level alignment, by contrast, would ask the SSM to reproduce the teacher’s patchattention geometry, which has no natural analogue in scanorder state trajectories. This is consistent with our ablation and with the broader intuition that output-space transfer is more architecture-agnostic than feature-space transfer.

TABLE IV  
FAIR CROSS-ARCHITECTURE DISTILLATION: SAME TEACHER, ALL STUDENTS TRAINED FROM SCRATCH (NO PRETRAINING), IDENTICAL LOGIT-KD RECIPE.
<table><tr><td>Student (from scratch)</td><td>Standalone</td><td>+ Logit KD</td><td>∆</td></tr><tr><td>MobileNetV3-Small</td><td>96.44</td><td>95.25</td><td>-1.19</td></tr><tr><td>EfficientNet-B0</td><td>97.15</td><td>97.39</td><td>+0.24</td></tr><tr><td>LVSSM (ours)</td><td>92.32</td><td>95.41</td><td>+3.09</td></tr></table>

TABLE V

COMPLEXITY AND LATENCY (SINGLE $2 2 4 ^ { 2 }$ IMAGE; RTX 4090 FOR GPU,SERVER CPU FOR CPU). FLOPS VIA PTFLOPS.
<table><tr><td>Model</td><td>Params (M)</td><td>GFLOPs</td><td>GPU (ms)</td><td>CPU (ms)</td></tr><tr><td>DINOv2-S (Teacher)</td><td>22.06</td><td>11.06</td><td>5.07</td><td>102.4</td></tr><tr><td>ResNet18</td><td>11.18</td><td>3.65</td><td>1.42</td><td>53.6</td></tr><tr><td>EfficientNet-B0</td><td>4.02</td><td>0.78</td><td>4.55</td><td>43.3</td></tr><tr><td>MobileNetV3-S</td><td>1.52</td><td>0.11</td><td>1.96</td><td>5.2</td></tr><tr><td>LVSSM (Ours)</td><td>4.45</td><td>2.19</td><td>10.05</td><td>1410</td></tr></table>

Practical guidance. For distillation across dissimilar families (ViT→SSM, and likely CNN→SSM), a well-tuned logitonly KD is a strong default; our controlled ablation shows that adding ViT-style feature alignment hurts even when the KD weight is held at full strength, so it should not be adopted without careful per-task validation against a logit-only baseline. Equally important is when to distill at all: our fairbaseline experiment (Section IV-F) shows the gain is large only when the from-scratch student would otherwise fall below the teacher (the SSM), and vanishes when it already matches the teacher (the CNNs). Cross-architecture KD is therefore best viewed as a tool for closing a student–teacher gap, not a universal accuracy boost.

Limitations. (1) Single dataset. We evaluate on one 6- class tea dataset (∼4K training images); the balance between logit and feature KD may shift with scale or task complexity. (2) Simplified SSM. Our student is a self-contained SSM approximation, not the official mamba-ssm kernel; accordingly we name it LVSSM and do not claim parity with Vim/VMamba, and its unfused scan is slow at inference (Table V). (3) Grouping granularity. The split is group-aware by source photograph (verified leakage-free), but we cannot rule out near-duplicate specimens photographed in separate sessions. (4) No edge hardware. We report GPU/CPU latency on server hardware, not on embedded devices (e.g., Jetson); real deployment suitability requires on-device measurement and a fused kernel. (5) Teacher ceiling. Under our unified pipeline the fine-tuned teacher reaches 97.86%, below the ImageNet-pretrained CNNs, so the student inherits a teacher that is not the strongest available model.

## VI. CONCLUSION

We presented a cross-architecture knowledge distillation study transferring knowledge from a DINOv2 teacher to a lightweight bidirectional Visual State Space Model for tea leaf disease classification. We identified and fixed two failure modes that block from-scratch SSM training on small data, and showed that simple logit-level distillation lifts the 4.45Mparameter student by +3.09 points on average over three seeds (to 95.41%; best run 96.20%, 94.45% macro-F1) — retaining 98.3% of the teacher’s accuracy with 5.0× fewer parameters.Controlled ablations, with the KD weight held at full strength, show that adding ViT-style feature-alignment losses reduces accuracy, so logit KD alone is the strongest configuration for this ViT→SSM transfer. A fair-baseline comparison further shows the distillation gain is specific to students that start below the teacher: the same recipe helps the from-scratch SSM (+3.09) but not from-scratch CNNs that already match the teacher. We release complete perclass metrics, confusion matrices, confidence intervals, and complexity measurements, and we identify a fused selectivescan kernel and multi-dataset validation as the key steps toward practical edge deployment.

TABLE VI  
MULTI-SEED RESULTS (SEEDS 42, 123, 456). MEAN ± STD OVER THREE RUNS.
<table><tr><td>Config</td><td>Accuracy (%) </td><td>Macro-F1 (%)</td></tr><tr><td>Standalone</td><td> $9 2 . 3 2 \pm 2 . 1 4$ </td><td> $9 0 . 1 9 \pm 2 . 5 0$ </td></tr><tr><td>Logit KD</td><td> ${ \bf 9 5 . 4 1 \pm 1 . 1 7 }$ </td><td> ${ \pm } 3 . 6 3 \pm 1 . 4 5$ </td></tr><tr><td>∆ (KD — standalone)</td><td> $+ 3 . 0 9$ </td><td> $+ 3 . 4 4$ </td></tr></table>

## ACKNOWLEDGMENT

This work was supported by the Innovation Training Program for College Students in Sichuan Province (No.202510626013).

## REFERENCES

[1] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.

[2] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[3] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, “Vision mamba: Efficient visual representation learning with bidirectional state space model,” arXiv preprint arXiv:2401.09417, 2024.

[4] Y. Liu, Y. Tian, Y. Zhao, H. Yu, L. Xie, Y. Wang, Q. Ye, and Y. Liu, “Vmamba: Visual state space model,” arXiv preprint arXiv:2401.10166, 2024.

[5] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[7] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in International Conference on Machine Learning, 2019, pp. 6105–6114.

[8] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan et al., “Searching for mobilenetv3,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019, pp. 1314–1324.

[9] S. P. Mohanty, D. P. Hughes, and M. Salathe, “Using deep learning for´ image-based plant disease detection,” Frontiers in Plant Science, vol. 7, p. 1419, 2016.

[10] K. P. Ferentinos, “Deep learning models for plant disease detection and diagnosis,” Computers and Electronics in Agriculture, vol. 145, pp. 311– 318, 2018.

[11] P. Chen, S. Liu, H. Zhao, and J. Jia, “Distilling knowledge via knowledge review,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 12, pp. 9163–9176, 2022.