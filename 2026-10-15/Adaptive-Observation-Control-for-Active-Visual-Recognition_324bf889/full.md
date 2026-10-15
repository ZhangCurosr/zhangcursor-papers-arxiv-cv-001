# Adaptive Observation Control for Active Visual Recognition

Dikai Li, Weijia Wang<sup>\*</sup>, and Yuan-Gen Wang<sup>⋆</sup>

School of Artificial Intelligence, Guangzhou University, Guangzhou, China dikaili@e.gzhu.edu.cn, wwj@gzhu.edu.cn, wangyg@gzhu.edu.cn

Abstract. Visual recognition models usually process a fixed view under a fixed computation budget, although images often benefit from diferent scales and amounts of spatial evidence. Active vision can adapt by observing a coarse scene and then selecting informative fixations, but the public AdaptiveNN-DeiT-S pipeline still fixes both field of view and observation count. We extend this fixed-step recognizer with a controllable observation interface that adapts both the field of view and the number of fixations to the input. Specifically, the model predicts a per-fixation scale that sets its field of view, refines each selected crop with optional scale-gated foveation and a residual high-resolution detail-token branch, and uses label-grounded stop heads for input-dependent early termination. Experiments across fine-grained benchmarks, full ImageNet-1K, and ImageNet-A/R-derived classification splits show that the interface remains competitive with fixed-step references while exposing a controllable observation budget. The same checkpoint enables thresholdcontrolled accuracy-compute trade-ofs without per-threshold retraining. On sequential ImageNet-1K, the learned stop rule outperforms fixed-step truncation of the same checkpoint, sustaining 81.9% top-1 using 2.45 of the four focus steps and exceeding three-step truncation at lower cost. On each benchmark, a variant combining scale control and detail-token fusion achieves the highest accuracy among the tested component-added non-exit variants. Our code is available at https://github.com/GZHU-DVL/LI-Dikai.

Keywords: Adaptive vision · Fine-grained recognition · Adaptive inference

## 1 Introduction

Modern image classifiers typically process a fixed-resolution view of the whole image. This design is eficient on parallel hardware, but it assigns the same observation budget to easy and hard examples, and small discriminative evidence can be diluted by background clutter. Active visual recognition ofers a complementary design, in which the model first builds a global representation, then samples task-relevant regions before deciding. AdaptiveNN [33] instantiates this coarse-to-fine idea, yet the public DeiT-S implementation still uses a fixed crop size across a fixed number of focus steps.

This fixed schedule imposes three limitations: (i) the field of view is fixed, although discriminative evidence ranges from local texture to larger object parts; (ii) every crop is encoded uniformly, so high-resolution local detail can be missed and peripheral context can be overemphasized; and (iii) the observation count is fixed, so easy examples still incur the full computational cost of the focus sequence. We address these limitations by adding a controllable observation interface to the same recognizer while maintaining competitive recognition accuracy. We organize the interface around the corresponding three observation decisions: geometry control determines the field of view of each fixation, evidence control determines how selected regions are encoded, and stopping control decides how many observations an input needs.

We introduce a scale-adaptive extension of AdaptiveNN, summarized in Fig. 1. The policy predicts location and scale for each fixation, allowing the model to zoom in on discriminative regions or zoom out for broader context. Selected centers are revisited by a residual detail-token path at higher resolution, with optional scale-gated foveation that suppresses peripheral detail in expanded views. A label-grounded stop head maps the current representation to a scalar stop logit $q _ { t }$ . At inference time, the model stops at the first focus step whose score $\sigma ( q _ { t } )$ reaches a chosen threshold τ, or at the final step if the threshold is never reached. Varying τ changes the stopping criterion without retraining the model, allowing us to evaluate the resulting accuracy-compute trade-of on each dataset.

## Our contributions are:

a controllable observation interface for active recognition that unifies geometry, evidence, and stopping control on one sequential state in the public AdaptiveNN-DeiT-S pipeline. It is implemented as a scale-adaptive fixation action, residual detail-token fusion with optional scale-gated foveation, and label-grounded per-step stop heads;

– an evaluation protocol that separates realized sequential exits from simulated stop-rule diagnostics, anchors component efects to a matched-budget same-codebase baseline, and compares against our reruns of the unmodified AdaptiveNN implementation released by its authors; and

– an empirical analysis across fine-grained datasets, full ImageNet-1K, and ImageNet-A/R-derived classification splits that identifies where scale and detail improve accuracy and where thresholded stopping trades computation for accuracy.

## 2 Related Work

Adaptive recognition and attention. Vision Transformers [5] and DeiT [29] are strong fixed-computation backbones, whereas AdaptiveNN [33] casts recognition as sequential decision-making that chooses focus regions after an initial glance, in the spirit of recurrent attention models [22] and spatial transformers [13] that learn where to sample. We retain this coarse-to-fine structure and extend its action space and inference policy.

![](images/5a040c52c7f1a05c643fca0c96367f1abd30809270860fc84d4220efdb6b654d.jpg)  
Fig. 1: Overview of the controllable observation loop. A low-resolution glance and the previous-step state drive a policy that predicts crop location and field-of-view scale (+S). The selected crop is optionally foveated (+F), encoded by the shared focus branch, augmented by an inner high-resolution detail-token branch (+D), and routed to classifier and label-grounded stop heads (+EE). Larger s<sub>t</sub> gives a wider field of view. Here, $( u _ { t } , v _ { t } )$ denotes the normalized top-left crop position, q<sub>t</sub> is the stop logit, σ is the sigmoid function, and τ is the exit threshold. The dashed path continues only when the stopping score $\sigma ( q _ { t } ) < \tau$ and $t < T ,$ , where T is the final focus step. The inner detail crop is sampled directly from I at the outer crop’s center, with scale 0.6s<sub>t</sub>. Green and orange boxes illustrate the outer and inner crop regions, respectively.

Foveated and multi-scale computation. Foveated vision allocates high fidelity to the gaze center and less to the periphery, as studied in foveal-peripheral networks [4,20], foveated rendering and magnification [24,28], and active recognition systems [14, 16]. Fine-grained categorization [17, 21, 30], which our fixation policy targets, commonly relies on part features [35], attention [7], or feature pyramids [18] rather than gaze-centered sensing. We treat foveation as a scalegated crop refinement rather than a standalone sensor or fixed pyramid, and our continuous scale coordinate selects one adaptive field of view per fixation, optimized by the policy reward.

Recent work also studies non-uniform visual representations under limited pixel budgets. Seeing More with Less [8] evaluates single-fixation foveated sampling for visual question answering and object detection. LLMind [2] adapts Möbius-parameterized sampling through test-time semantic feedback from a frozen vision-language model. Our method addresses sequential image recognition: a trained fixation policy uses the evolving state to choose the next crop’s location and scale, while a stop head determines whether to acquire another observation.

Early exit and adaptive inference. Early-exit networks attach auxiliary classifiers or stopping policies to intermediate representations so that inference terminates early for easy examples [6,12,27], and dynamic vision models prune or halt tokens within a fixed view, adapt depth, width, and routing, or cascade input resolutions per image [9,25,32,34]. These decisions act along network depth or token count within a fixed passive view, under heterogeneous backbones, resolutions, and recipes, so their accuracy-compute frontiers are not directly comparable to an active observation loop. We therefore evaluate controllability within one public active-recognition pipeline. We build on the glance-and-focus paradigm for active recognition [31, 33], which stops once the classifier passes a confidence threshold; our label-grounded stop heads instead interact with the learned per-fixation scale and provide multiple thresholds from the same checkpoint.

## 3 Method

Overall framework. The method builds on the AdaptiveNN-DeiT-S [33] glance-and-focus structure and extends its fixed four-step focus sequence into an adaptive observation controller. A low-resolution glance initializes a spatial token state. At each focus step, the policy predicts a crop location and field-ofview scale; the selected crop is transformed by scale-gated foveation, encoded by the focus branch, optionally complemented by an inner high-resolution detail descriptor, and then classified. A per-step stop head maps the same representation to a stopping score. The trained model therefore controls three quantities at inference time: field of view, local evidence quality, and observation count.

## 3.1 Controllable Observation Loop

Throughout the paper, we index focus steps starting from 1, with the glance prediction denoted as step 0. Let $m _ { 0 }$ denote the spatial token state produced by the glance branch. At focus step t, the policy reads the detached state $m _ { t - 1 }$ and predicts an action

$$
a _ { t } \sim \pi _ { \psi } ( \cdot \mid m _ { t - 1 } ) , \qquad a _ { t } = ( u _ { t } , v _ { t } , z _ { t } ) .\tag{1}
$$

The first two coordinates select the normalized top-left crop position, and the third coordinate controls the field of view. The observation path can be summa-

rized as

$$
\begin{array} { r l } & { P _ { t } = \mathrm { C r o p } ( I ; u _ { t } , v _ { t } , s _ { t } ) , } \\ & { \widetilde { P } _ { t } = \mathrm { F o v e a } ( P _ { t } ; s _ { t } , t , e ) , } \\ & { m _ { t } = U _ { \theta } ( m _ { t - 1 } , E _ { \theta } ( \widetilde { P } _ { t } ) , a _ { t } ) , } \end{array}\tag{2}
$$

where $E _ { \theta }$ is the focus encoder, $U _ { \theta }$ is the feature-recovery update, e is the training epoch, and $\widetilde { P } _ { t } = P _ { t }$ when foveation is disabled or has zero strength. A pooled classifier feature $h _ { t }$ is obtained from $m _ { t }$ . When enabled, the detail path adds a residual descriptor to $h _ { t }$ before the classifier and stop heads produce $y _ { t }$ and $q _ { t }$ The exit step under threshold $\tau$ is

$$
t ^ { \star } = \operatorname* { m i n } \{ t : \sigma ( q _ { t } ) \geq \tau \} ,\tag{3}
$$

with $t ^ { \star } = T$ if no earlier step satisfies the condition.

## 3.2 Baseline AdaptiveNN

Given image I, a glance branch processes a low-resolution global view and produces the spatial token state used by the fixation policy. At focus step $t \in \{ 1 , \ldots , T \}$ , the policy predicts a fixation location, the focus branch encodes the selected patch, the feature-recovery module writes the local evidence back into the spatial state, and a classifier predicts logits $y _ { t }$ . We use the default four-step setting $( T = 4 )$ . The original policy outputs two continuous action components,

$$
a _ { t } = ( u _ { t } , v _ { t } ) , \qquad u _ { t } , v _ { t } \in [ 0 , 1 ] ,\tag{4}
$$

which define the top-left crop coordinate after scaling to the image extent.

Training combines supervised classification losses on the glance and focus predictions, knowledge distillation from the final focus prediction to earlier predictions, and a PPO [26] update for the non-diferentiable fixation policy. We retain this optimization structure, modifying only the action space, observation transforms, and stopping head described below.

## 3.3 Scale-Adaptive Fixations

The baseline policy chooses only where to look: the crop size is fixed, so the field of view cannot adapt to objects of diferent apparent size. We extend the action space from two-dimensional location to location plus scale,

$$
a _ { t } = ( u _ { t } , v _ { t } , z _ { t } ) , \qquad s _ { t } = s _ { \operatorname* { m i n } } + ( s _ { \operatorname* { m a x } } - s _ { \operatorname* { m i n } } ) z _ { t } ,\tag{5}
$$

where $z _ { t } = \sigma ( \ell _ { t } ) \in [ 0 , 1 ]$ is produced by a third policy output and $s _ { t }$ is the relative field-of-view scale. The extracted crop has side length proportional to $s _ { t }$ and is resized to the fixed focus input size, so the encoder cost per focus step is unchanged by the chosen scale. When the scale head is disabled or fixed at $s _ { t } = 1$ , the model recovers the original fixed-field-of-view behavior. We initialize the scale-head weight to zero and its bias to

$$
\log \mathrm { i t } \left( \frac { s _ { \mathrm { a n c h o r } } - s _ { \mathrm { m i n } } } { s _ { \mathrm { m a x } } - s _ { \mathrm { m i n } } } \right) , \qquad s _ { \mathrm { a n c h o r } } = 1 . 0 ,\tag{6}
$$

so training starts from the baseline field of view. The scale coordinate shares the same PPO objective as the location coordinates. Because the reward is unchanged, scale is learned from recognition improvements rather than from an auxiliary scale loss.

For the fine-grained datasets and ImageNet-A we use $s _ { \mathrm { m i n } } = 0 . 7 0$ and $s _ { \mathrm { m a x } } =$ 1.50, allowing the policy to zoom in on small discriminative parts or zoom out for broader context; for ImageNet-1K and ImageNet-R we use the narrower range [0.75, 1.35] to limit scale variation on datasets with more uniform object sizes.

## 3.4 Foveated Focus Crop

We use foveation as a scale-gated transform of the selected focus crop rather than as a standalone sensor. Given the focus crop $P _ { t }$ at scale $s _ { t } ,$ the model preserves the central region and blends the crop with a degraded version of itself:

$$
\widetilde { P } _ { t } = ( 1 - \alpha _ { t } ) P _ { t } + \alpha _ { t } \left( M \odot P _ { t } + ( 1 - M ) \odot D ( P _ { t } ) \right) ,\tag{7}
$$

where M is a radial center mask and $D ( \cdot )$ downsamples the crop to 50% of its side length before upsampling it back. The mask is 1 inside radius 0.35, decreases linearly across a 0.10 blending band centered at radius 0.40, and is 0 outside radius 0.45 in normalized crop coordinates. This implements the centralversus-peripheral fidelity trade-of used in foveated vision models [4,16,20]; when $\alpha _ { t } = 0$ , the operation reduces to the original crop.

The efective foveation strength is scale-gated:

$$
\alpha _ { t } = \alpha _ { \mathrm { m a x } } r ( e ) \mathbf { 1 } [ t \geq t _ { 0 } ] \mathrm { c l i p } \left( \frac { s _ { t } - s _ { \mathrm { a n c h o r } } } { s _ { \mathrm { m a x } } - s _ { \mathrm { a n c h o r } } } , 0 , 1 \right) ,\tag{8}
$$

where $\alpha _ { \mathrm { m a x } } = 0 . 5 , s _ { \mathrm { a n c h o r } } = 1 . 0$ , s<sub>max</sub> is the dataset-specific maximum scale from Eq. $( 5 ) , t _ { 0 } = 2$ under the step indexing defined above, and $r ( e )$ is the training ramp

$$
r ( e ) = 0 ~ ( e < 1 0 ) , \qquad r ( e ) = { \frac { e - 1 0 } { 1 0 } } ~ ( 1 0 \leq e < 2 0 ) , \qquad r ( e ) = 1 ~ ( e \geq 2 0 ) .\tag{9}
$$

Early-exit fine-tuning starts from the trained $+ \mathrm { S + F + D }$ checkpoint with $r ( e ) =$ 1. The scale gate makes foveation strongest for expanded views and nearly inactive near the anchor scale, acting as a conditional refinement of the evidence path whose dataset-dependent efect is quantified in Sec. 5.

## 3.5 Detail-Token Fusion

Foveation controls peripheral fidelity, but fine-grained evidence can still be under-represented in the selected center. We therefore use a detail branch that extracts an inner crop aligned to the center of the selected outer crop. The inner crop is sampled from the original image using the same fixation center as $P _ { t }$ , with scale $0 . 6 s _ { t }$ in the main experiments, giving a higher-resolution view of the innermost region.

Because the inner crop shares the outer crop’s center, the policy makes one location decision for both paths. The detail branch then samples a smaller region from the original image, rather than magnifying the resized outer crop. This distinction preserves access to image detail that outer-crop resizing may discard. The fixed scale ratio also lets scale control change the two fields of view together.

The inner crop is processed by the shared focus encoder, and its features are pooled around the spatial center into a descriptor $p _ { t }$ . Scale-gated foveation therefore modulates peripheral fidelity within the main focus crop, whereas the detail path injects a separate high-resolution central descriptor. In the reported setting, the descriptor is the average of the central $3 \times 3$ token window after the focus encoder. We detach the inner features before the detail gate so that gradients from the auxiliary detail path do not overwrite the main focus encoder updates:

$$
p _ { t } = \mathrm { C e n t e r P o o l } \bigl ( \mathrm { s g } \bigl [ E _ { \theta } \bigl ( P _ { t } ^ { \mathrm { i n n e r } } \bigr ) \bigr ] \bigr ) , \qquad d _ { t } = \mathrm { t a n h } \bigl ( G _ { \phi } ( p _ { t } ) \bigr ) , \qquad \widehat { h } _ { t } = h _ { t } + \lambda _ { t } d _ { t } ,\tag{10}
$$

where sg[·] denotes stop-gradient, $G _ { \phi }$ is a two-layer MLP with hidden width 2C and GELU activation, $C$ is the focus feature dimension, and $\lambda _ { t }$ ramps from 0 to 0.5 over epochs 10–18 in non-exit runs. The main experiments use one pooled detail token, center pooling, residual fusion, and no write-back into the next policy state. The detail branch adds one focus-encoder pass per detail-enabled executed step; we enable detail from the second focus step onward. Unlike recurrent attention CNNs that stack additional sub-networks for magnified regions [7], our design reuses the shared focus encoder and injects only one stop-gradient pooled token. For conservative accounting, the supplementary encoder-FLOP estimate counts a detail pass at every executed focus step, including the first.

## 3.6 Confidence-Conditioned Early Exit

The full model adds a scalar stop head to each focus step in the four-step observation path. The stop head consumes the pooled feature before classification and outputs a stop logit $q _ { t }$ . At evaluation, the model exits at the first step satisfying

$$
\sigma ( q _ { t } ) \geq \tau ,\tag{11}
$$

or at the final step $T = 4$ if no earlier step passes the threshold. The threshold $\tau$ is an inference-time hyperparameter, distinct from the confidence threshold used to construct training labels. Since the reported early-exit runs use hard class labels (no mixup), we use a hard-label stop target. Let $\rho _ { t } = \mathrm { s o f t m a x } ( y _ { t } )$ 2 $\widehat { c } _ { t } = \operatorname* { m a x } _ { c } \rho _ { t } ( c ) , \widehat { y } _ { t } = \arg \operatorname* { m a x } _ { c } \rho _ { t } ( c )$ , and $y ^ { \star }$ be the ground-truth class. The stop target is

$$
b _ { t } = { \bf 1 } [ \widehat { y } _ { t } = y ^ { \star } ] { \bf 1 } [ \widehat { c } _ { t } \geq 0 . 7 ] .\tag{12}
$$

This label-grounded target trains the stop head against task correctness rather than against agreement with the final step.

At inference, the stopping rule uses only the learned score and does not require ground-truth labels. The confidence threshold in the training target therefore does not replace the inference threshold τ. Changing τ selects an intermediate prediction without altering its logits. Because later observations may change the predicted class, neither the stopping score nor the final-step accuracy establishes the accuracy of the returned predictions; this must be evaluated at the selected exits.

The stop head is trained with binary cross-entropy,

$$
\mathcal { L } _ { \mathrm { s t o p } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { B C E W i t h L o g i t s } ( q _ { t } , b _ { t } ) ,\tag{13}
$$

added to the main classification loss with auxiliary weight 0.05. Early-exit finetuning initializes from the $+ \mathrm { S + F + D }$ checkpoint and runs for 15–20 additional epochs (dataset schedules in the supplementary material). At inference, varying $\tau$ traces out a range of accuracy-compute operating points from the same checkpoint.

Because a stop decision changes how many observations are executed per sample, we report two protocols separately. In simulation mode the model runs the full four-step pass and applies the stop rule post hoc, analyzing the policy without claiming realized savings. In sequential-step mode samples terminate when the threshold fires, and the realized step count is measured. Since $y _ { t }$ and $q _ { t }$ are produced before any later step, gathering canonical step-t logits from a full unrolled pass introduces no temporal leakage; we use these for protocolcomparable accuracy at the measured exit steps and additionally record physically truncated logits for consistency checks. Only sequential-step rows support realized compute-saving claims.

We report accuracy and focus-step ratio (average selected focus steps divided by $T = 4 )$ ; the supplementary cost breakdown translates this ratio into a conservative encoder-FLOP estimate, including the always-on glance overhead and excluding crop sampling and other non-encoder operations.

## 3.7 Training Objective

Training separates diferentiable prediction losses from the non-diferentiable fixation policy update. The backbone, classifiers, detail branch, and stop heads are optimized with

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s u p } } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { k d } } \mathcal { L } _ { \mathrm { k d } } + \lambda _ { \mathrm { s t o p } } \mathcal { L } _ { \mathrm { s t o p } } , } \end{array}\tag{14}
$$

where $\mathcal { L } _ { \mathrm { c l s } }$ is the sum of supervised cross-entropy losses on the glance prediction, focus predictions, and focus-only auxiliary prediction, and $\mathcal { L } _ { \mathrm { k d } }$ distills the final focus prediction to the glance and earlier focus predictions using temperature 1. We set $\lambda _ { \mathrm { k d } } = 1$ . The stop term is enabled only for early-exit fine-tuning; for these runs, $\lambda _ { \mathrm { s t o p } } = 0 . 0 5$ , and otherwise $\lambda _ { \mathrm { s t o p } } = 0$

The fixation policy is updated separately with PPO because crop selection is non-diferentiable. For each sample and focus step, the reward is defined as the reduction in cross entropy from the previous prediction to the current focus prediction:

$$
R _ { t } = \mathrm { C E } _ { t - 1 } - \mathrm { C E } _ { t } ,\tag{15}
$$

where $\mathrm { C E _ { 0 } }$ is computed from the glance prediction and $\mathrm { C E } _ { t }$ from focus step t after the enabled observation transforms and detail fusion. The learned scale coordinate shares this policy reward, with no auxiliary scale regression loss or scale-cost penalty.

The components are trained in stages under a shared recipe. All non-exit variants start from the same AdaptiveNN checkpoint. During scale warmup, the scale coordinate in both the executed action and PPO latent action is fixed to the anchor so the crop field of view stays at the baseline; after warmup, scale is sampled and optimized with location. Foveation and detail fusion use the ramps in Eqs. (9) and (10). Early-exit fine-tuning starts from the trained $+ \mathrm { S + F + D }$ checkpoint and adds only the stop supervision in Eqs. (12) and (13). Datasetspecific schedules are kept in the supplementary material.

## 4 Experiments

## 4.1 Datasets and Protocol

We evaluate on eight recognition benchmarks: FGVC-Aircraft [21], Stanford Cars [17], CUB-200-2011 [30], Stanford Dogs [15], Oxford-IIIT Pet [23], ImageNet-A [11], ImageNet-R [10], and the full ImageNet-1K validation set [3], covering part-sensitive fine-grained recognition, ImageNet-A/R-derived classification splits, and large-vocabulary recognition. Following the AdaptiveNN protocol, we report accuracy on each dataset’s public evaluation split, which we call validation throughout even when a release names it “test”.

All non-exit variants initialize from the same AdaptiveNN checkpoint and fine-tune DeiT-S glance and focus branches for four focus steps $( T = 4 )$ with AdamW [19], cosine decay, RandAugment [1], and PPO updates with incremental cross-entropy reward; Full+EE initializes from the trained $+ \mathrm { S + F + D }$ checkpoint. Dataset-specific schedules are summarized in the supplementary material. We select checkpoints by best public-validation Acc@1 and report Acc@5 for context. Evaluation uses crop\_pct=1.0 with input size 336 for Aircraft, Cars, CUB, Dogs, Oxford-Pet, and ImageNet-A, and 288 for ImageNet-1K and ImageNet-R; ImageNet-R uses a fixed 10% validation split (seed 0), matching the split used during training.

## 4.2 Compared Variants

We compare our AdaptiveNN rerun against additive variants:

– Orig: the original AdaptiveNN configuration, rerun inside our modified codebase with all new components disabled. It uses fixed-scale focus crops and no foveation, detail fusion, or stop head, but shares the training recipe used by the variants below;

– +S: learned scale action;

– +S+F: learned scale with foveated focus crops;

– +S+D: learned scale with detail-token fusion;

– +S+F+D: the non-exit full adaptive crop model; and

– Full+EE: +S+F+D with the confidence-conditioned stop head.

For Aircraft, Cars, CUB, Dogs, ImageNet-A, ImageNet-1K, ImageNet-R, and Oxford-Pet we also rerun the unmodified oficial AdaptiveNN codebase [33] (“Official” in Tab. 1) as the external fixed-step reference. Orig remains the internal same-codebase anchor for additive component efects in Tab. 2; the supplementary material reports its diference from Oficial separately.

## 5 Results

We first compare accuracy against the strongest available fixed-step baselines (Tab. 1; budget-matched diagnostic in the supplementary material), then isolate the geometry and evidence controls (Tab. 2, with supplementary scale statistics), and finally evaluate stopping as a thresholded accuracy-compute trade-of (Tab. 3, Fig. 3).

## 5.1 Accuracy and Budget Diagnostics

Table 1 summarizes the main validation results for the scale-adaptive detail model. To keep the main comparison compact, we show the external Oficial rerun and the strongest completed in-codebase non-exit row, selected from Orig and the four additive non-exit variants. Our primary metric, ∆best, subtracts the stronger of these fixed/non-exit references from Full+EE. Fine-grained rows report the seed-0 run and the ImageNet-A/-1K/-R rows are single runs; three-seed paired checks are summarized below. Accuracy rows are not compute-neutral— a full four-step Full+EE unroll costs 9.90 versus 5.50 encoder-pass GFLOPs for Orig (supplementary cost breakdown)—so realized savings arise only under sequential early exit (Tab. 3). The supplementary material reports the Orig– Oficial gap and the budget-matched continuation diagnostic.

Full+EE exceeds the oficial rerun on seven of eight datasets and is within 0.002 points on ImageNet-1K. The only negative ∆best values occur on ImageNet-1K (−0.158) and ImageNet-R (−0.267), where +S+D is stronger. Small margins below 0.1 points are treated as parity-level diferences.

The additional training stage also afects how improvements should be attributed. Full+EE receives further fine-tuning, so its diference from an originalschedule non-exit model combines the stopping objective with additional optimization. The continuation diagnostic in the supplementary material compares models after matched additional training to assess this source of variation.

Table 1: Main validation accuracy $( \operatorname { A c c @ 1 } / \operatorname { A c c @ 5 }$ at the selected 1-based epoch). ∆best is Full+EE minus the stronger of the Oficial rerun and the best in-codebase non-exit variant (selected row named in parentheses). <sup>†</sup>+S+F also achieves 66.138% Acc@1 on ImageNet-A. The highest Acc@1 in each row is shown in bold; Table 2 uses the same convention.
<table><tr><td>Dataset</td><td>Official</td><td>Best in-codebase non-exit</td><td>Full+EE (ours)</td><td>∆best</td></tr><tr><td>Aircraft</td><td>77.318 /97.000 @75</td><td>77.078/96.550 @69  $( + \mathrm { { S + F + D } ) }$ </td><td>77.558/96.910 @10</td><td>+0.240</td></tr><tr><td>Cars</td><td>83.076 /97.525 @79</td><td>82.741/97.675 @77  $( + \mathbf { S } + \mathbf { F } + \mathbf { D } )$ </td><td>83.711/97.650 @15</td><td>+0.635</td></tr><tr><td>CUB</td><td>84.524/96.912 @65</td><td>84.369/96.929 @79 (Orig)</td><td>84.800/97.084 @17</td><td>+0.276</td></tr><tr><td>Dogs</td><td>91.981/99.452 @30</td><td>92.401/99.464 @22  $( + \mathrm { S } + \dot { \mathrm { D } } )$ </td><td>92.413/99.534@3</td><td>+0.012</td></tr><tr><td>ImageNet-1K</td><td>82.126/ /95.946 @1</td><td>82.282/96.014 @1 (+S+D)</td><td>82.124/95.896 @5</td><td>-0.158</td></tr><tr><td>ImageNet-A</td><td>64.815/80.688 @46</td><td>66.138/82.011 @42  $( + \mathbf { S } + \mathbf { D } ) ^ { \dagger }$ </td><td>66.534/81.746 @17</td><td>+0.396</td></tr><tr><td>ImageNet-R</td><td>85.133/94.500 @37</td><td>85.900/95.033 @32  $( + \mathrm { S } \mathrm { + } \dot { \mathrm { D } } )$ </td><td>85.633/94.933 @14</td><td>-0.267</td></tr><tr><td>Oxford-Pet</td><td>96.402/99.838 @68</td><td>96.537/99.865 @66 (Orig)</td><td>96.591/99.919 @20</td><td>+0.054</td></tr></table>

Paired ImageNet-1K tests quantify uncertainty over aligned validation examples for fixed checkpoints, whereas the repeated-seed studies examine variation across independently trained models. These analyses answer diferent questions. A single-run ranking alone therefore does not establish that one configuration reliably outperforms another across training seeds.

## 5.2 Geometry and Evidence Controls

Table 2 gives the component ablation behind the dataset-dependent accuracy behavior. We read +S, +D, and +F as coupled controls: scale action sets fieldof-view geometry, detail-token fusion injects higher-resolution local evidence, and foveation re-weights periphery with strength gated by the learned scale through Eq. (8).

Scale alone improves Acc@1 on Cars (+0.560), Dogs (+0.327), Aircraft $\left( + 0 . 3 9 0 \right)$ , and ImageNet-1K (+0.118). Among the non-exit variants with added components, +S+D or $+ \mathrm { S + F + D }$ achieves the highest Acc@1 on each benchmark: +S+D is strongest on Dogs, ImageNet-1K, ImageNet-R, and Oxford-Pet and ties ${ + \mathrm { S + F } }$ on ImageNet-A, while $+ \mathrm { S + F + D }$ is strongest on Aircraft, Cars, and CUB.

Foveation is activated through the scale gate and is not uniformly helpful. For the Full+EE checkpoints, the supplementary statistics show mean scales near the anchor on Aircraft, CUB, Dogs, and ImageNet-A; these means do not determine the strength of foveation at individual fixations. In the non-exit ablation, ${ + \mathrm { S + F } }$ performs close to +S on Aircraft, CUB, and Dogs, whereas ImageNet-A has a 0.397-point diference in the single-run comparison. Cars benefits from $+ \mathrm { S + F + D }$ , while ImageNet-1K, ImageNet-R, and Oxford-Pet favor +S+D over $+ \mathrm { S + F + D }$ . Thus, the component comparisons show a dataset-dependent efect rather than a uniform benefit from foveation.

Figure 2 illustrates the detail, foveation, scale, and stopping controls at the example level.

![](images/3830fe0511ad26f1d6192d6471900d4e93e9481f3b1e72723fae5bcca25adb30.jpg)  
Fig. 2: Qualitative module contrasts on held-out examples. The Cars rows compare detail fusion $\mathrm { ( + S + F \ t o \ + S + F + D ) }$ and scale-gated foveation (+S+D to $+ \mathrm { S } { \left. + \mathrm { F } { + \mathrm { D } } \right) }$ The ImageNet-1K rows compare Orig with Full+EE as joint-model contrasts rather than controlled, single-component comparisons. Colored boxes and numbers mark focus order; the heat overlay is per-example fixation density; dashed inner boxes mark detail crops (top row) and the sharp foveal core (second row) from step 2 onward.

Table 2: Acc@1 ablation across additive model variants. +S, +F, and +D denote learned scale, scale-gated foveation, and detail-token fusion, respectively; Full+EE adds early-exit fine-tuning on top of +S+F+D.
<table><tr><td>Dataset</td><td>Orig</td><td>+S</td><td>+S+F</td><td>+S+D</td><td>+S+F+D</td><td>Full+EE</td></tr><tr><td>Aircraft</td><td>76.358</td><td>76.748</td><td>76.898</td><td>76.628</td><td>77.078</td><td>77.558</td></tr><tr><td>Cars</td><td>81.696</td><td>82.256</td><td>81.771</td><td>82.467</td><td>82.741</td><td>83.711</td></tr><tr><td>CUB</td><td>84.369</td><td>84.179</td><td>84.300</td><td>84.300</td><td>84.334</td><td>84.800</td></tr><tr><td>Dogs</td><td>92.051</td><td>92.378</td><td>92.238</td><td>92.401</td><td>92.238</td><td>92.413</td></tr><tr><td>ImageNet-1K</td><td>82.116</td><td>82.234</td><td>82.146</td><td>82.282</td><td>82.246</td><td>82.124</td></tr><tr><td>ImageNet-A</td><td>65.873</td><td>65.741</td><td>66.138</td><td>66.138</td><td>65.873</td><td>66.534</td></tr><tr><td>ImageNet-R</td><td>85.467</td><td>85.300</td><td>85.400</td><td>85.900</td><td>85.133</td><td>85.633</td></tr><tr><td>Oxford-Pet</td><td>96.537</td><td>96.402</td><td>96.429</td><td>96.456</td><td>96.429</td><td>96.591</td></tr></table>

## 5.3 Stopping-Control Trade-Of

Table 3 reports confidence-conditioned stopping at thresholds $\tau \in \{ 0 . 7 0 , 0 . 9 0 \}$ Sequential-step rows measure realized exits; simulation rows apply the stopping rule to full rollouts and do not establish realized savings. The supplementary material reports the complete threshold sweep and encoder-pass estimates.

![](images/6dabfff593f7b463e96c0799e0411b7a173075e7bc291692f0e2f0304fb96f65.jpg)

Table 3: Confidence-conditioned early exit at $\tau \in \{ 0 . 7 0 , 0 . 9 0 \}$ . Seq. steps: realized exits (Acc@1 from canonical logits at the exit step; steps measured and encoder FLOPs estimated from those steps). Sim.: diagnostic stop rule on full four-step outputs only; these rows do not measure realized inference cost or FLOP savings. Ratio is average focus steps over four. Full is the full-unroll Acc@1 of the same checkpoint measured in the corresponding protocol environment; for ImageNet-1K this is the sequential evaluation, hence 82.040 versus the main-table 82.124.
<table><tr><td></td><td></td><td></td><td colspan="3">τ = 0.70</td><td colspan="3">τ = 0.90</td></tr><tr><td>Dataset</td><td>Protocol</td><td>Full</td><td>Acc@1</td><td>Avg. step</td><td>Ratio</td><td>Acc@1</td><td>Avg. step</td><td>Ratio</td></tr><tr><td>Aircraft</td><td>Sim.</td><td>77.558</td><td>77.378</td><td>3.669</td><td>0.917</td><td>77.558</td><td>4.000</td><td>1.000</td></tr><tr><td>Cars</td><td>Sim.</td><td>83.711</td><td>83.686</td><td>3.988</td><td>0.997</td><td>83.711</td><td>4.000</td><td>1.000</td></tr><tr><td>CUB</td><td>Sim.</td><td>84.800</td><td>84.800</td><td>3.710</td><td>0.927</td><td>84.800</td><td>4.000</td><td>1.000</td></tr><tr><td>Dogs</td><td>Sim.</td><td>92.413</td><td>92.424</td><td>3.992</td><td>0.998</td><td>92.413</td><td>4.000</td><td>1.000</td></tr><tr><td>ImageNet-1K</td><td>Seq. steps</td><td>82.040</td><td>81.404</td><td>1.731</td><td>0.433</td><td>81.888</td><td>2.446</td><td>0.611</td></tr><tr><td>ImageNet-A</td><td>Sim.</td><td>66.534</td><td>66.534</td><td>3.959</td><td>0.990</td><td>66.534</td><td>4.000</td><td>1.000</td></tr><tr><td>ImageNet-R</td><td>Sim.</td><td>85.633</td><td>84.333</td><td>1.437</td><td>0.359</td><td>85.033</td><td>1.926</td><td>0.481</td></tr><tr><td>Oxford-Pet</td><td>Seq. steps</td><td>96.591</td><td>96.050</td><td>1.343</td><td>0.336</td><td>96.591</td><td>4.000</td><td>1.000</td></tr></table>

Fig. 3: Realized sequential accuracy versus average focus steps for the Full+EE checkpoint. Red markers are realized confidence-conditioned exits from one checkpoint with measured average focus steps; dashed gray curves are fixed-step truncation at integer budgets. The supplementary material reports the full mixed diagnostic, including ImageNet-R simulation-only points.

Figure 3 compares confidence-conditioned exits with fixed-step truncation of the same checkpoint under realized sequential execution. On ImageNet-1K the sequential $\tau = 0 . 7 0$ point outperforms fixed two-step truncation by 0.298 percentage points in Acc@1 while using fewer than two focus steps on average, and $\tau = 0 . 9 0$ outperforms fixed three-step truncation in Acc@1 while using fewer focus steps on average. Oxford-Pet shows a low-cost $\tau = 0 . 7 0$ operating point and conservatively returns to full-step behavior at $\tau = 0 . 9 0$ . The stop head thus reallocates observations across examples, with realized savings supported only by the sequential-step rows of Tab. 3.

The cost comparison depends on the reference model and the selected step count. For example, the ImageNet-1K $\tau = 0 . 9 0$ setting uses about 2.45 focus steps but costs 6.48 encoder-pass GFLOPs under the conservative supplementary estimate, compared with 5.50 for four-step Orig. It thus saves computation relative to four-step Full+EE while remaining above Orig under this proxy. The supplementary runtime study uses a separately fitted max-softmax stopping rule and measures model-forward execution. Its timings characterize that profiling setup and should not be assigned directly to the learned-head threshold points in Tab. 3. Step reduction, encoder cost, and measured latency describe related but distinct quantities.

## 6 Limitations

Evaluation scope. Three-seed studies cover Cars, Aircraft, CUB, Dogs, and Oxford-Pet; the original ImageNet-A, ImageNet-1K, and ImageNet-R table rows use single runs. A separate single-GPU study in the supplementary material examines three training seeds for ImageNet-R. Checkpoints are selected by best public-split Acc@1, so parity-level margins are sensitive to epoch selection (bestversus-last table in the supplementary material).

Accuracy and cost. Extra training of the fixed-step model is itself a strong baseline, so matched-budget gains are dataset-dependent. Detail-token fusion adds one focus-encoder pass from the second focus step onward, so rarely-exiting datasets incur most of the additional cost; the supplementary runtime measurements cover ImageNet-1K model-forward execution on one GPU.

Component and threshold generality. Interactions are not uniform— ImageNet-R favors +S+D over active foveation—and the thresholds in Tab. 3 are set manually.

## 7 Conclusion

We extended a fixed-step AdaptiveNN recognizer with a controllable observation interface: learned scale controls field of view, detail-token fusion with optional scale-gated foveation controls local evidence, and confidence-conditioned stop heads control observation count. Our method shows competitive performance across benchmark datasets while providing a threshold-controlled accuracycompute frontier from one checkpoint: the measured sequential ImageNet-1K run sustains 81.9% top-1 at τ = 0.90 using 2.45 focus steps. Since the detail path adds per-step cost, net compute benefit depends on realized early termination. The resulting framework ofers explicit, tunable control over observation geometry, evidence, and stopping.

## References

1. Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: Randaugment: Practical automated data augmentation with a reduced search space. In: IEEE Conference on Computer Vision and Pattern Recognition Workshops. pp. 702–703 (2020)

2. Debnath, S., Manh, B.D., Liu, Z., Wang, L.: LLMind: Bio-inspired training-free adaptive visual representations for vision-language models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3133–3142 (2026)

3. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 248–255 (2009)

4. Deza, A., Konkle, T.: Emergent properties of foveated perceptual systems. arXiv preprint arXiv:2006.07991 (2020)

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

6. Elbayad, M., Gu, J., Grave, E., Auli, M.: Depth-adaptive transformer. In: International Conference on Learning Representations (2020)

7. Fu, J., Zheng, H., Mei, T.: Look closer to see better: Recurrent attention convolutional neural network for fine-grained image recognition. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 4438–4446 (2017)

8. Gizdov, A., Ullman, S., Harari, D.: Seeing more with less: Human-like representations in vision models. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4408–4417 (2025)

9. Han, Y., Huang, G., Song, S., Yang, L., Wang, H., Wang, Y.: Dynamic neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 44(11), 7436–7456 (2022)

10. Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., Song, D., Steinhardt, J., Gilmer, J.: The many faces of robustness: A critical analysis of out-of-distribution generalization. In: IEEE International Conference on Computer Vision. pp. 8340–8349 (2021)

11. Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., Song, D.: Natural adversarial examples. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 15262–15271 (2021)

12. Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., Weinberger, K.Q.: Multiscale dense networks for resource eficient image classification. In: International Conference on Learning Representations (2018)

13. Jaderberg, M., Simonyan, K., Zisserman, A., Kavukcuoglu, K.: Spatial transformer networks. In: Advances in Neural Information Processing Systems (2015)

14. Jonnalagadda, A., Wang, W., Zhang, W., Jha, S.: FoveaTer: Foveated transformer for image classification. arXiv preprint arXiv:2105.14173 (2021)

15. Khosla, A., Jayadevaprakash, N., Yao, B., Li, F.F.: Novel dataset for fine-grained image categorization. In: CVPR Workshop on Fine-Grained Visual Categorization (2011)

16. Killick, T., Bammey, Q., Pitie, F.: Foveation in the era of deep learning. In: British Machine Vision Conference (2023)

17. Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for finegrained categorization. In: IEEE International Conference on Computer Vision Workshops. pp. 554–561 (2013)

18. Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S.: Feature pyramid networks for object detection. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 2117–2125 (2017)

19. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019)

20. Lukanov, H., Krawczyk, B., Wozniak, M.: Biologically inspired deep learning model for eficient foveal-peripheral vision. Frontiers in Computational Neuroscience 15, 746204 (2021)

21. Maji, S., Rahtu, E., Kannala, J., Blaschko, M., Vedaldi, A.: Fine-grained visual classification of aircraft. Tech. rep., University of Oxford (2013)

22. Mnih, V., Heess, N., Graves, A., Kavukcuoglu, K.: Recurrent models of visual attention. In: Advances in Neural Information Processing Systems (2014)

23. Parkhi, O.M., Vedaldi, A., Zisserman, A., Jawahar, C.V.: Cats and dogs. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 3498–3505 (2012)

24. Patney, A., Salvi, M., Kim, J., Kaplanyan, A., Wyman, C., Benty, N., Luebke, D., Lefohn, A.: Towards foveated rendering for gaze-tracked virtual reality. ACM Transactions on Graphics 35(6), 1–12 (2016)

25. Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.J.: Dynamicvit: Eficient vision transformers with dynamic token sparsification. In: Advances in Neural Information Processing Systems (2021)

26. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O.: Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017)

27. Teerapittayanon, S., McDanel, B., Kung, H.T.: Branchynet: Fast inference via early exiting from deep neural networks. In: International Conference on Pattern Recognition. pp. 2464–2469 (2016)

28. Thavamani, S., Asari, V.K., Nguyen, K., Medioni, G.: FOVEA: Foveated image magnification for autonomous navigation. In: IEEE/CVF International Conference on Computer Vision. pp. 15539–15548 (2021)

29. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jegou, H.: Training data-eficient image transformers and distillation through attention. In: International Conference on Machine Learning. pp. 10347–10357 (2021)

30. Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.: The caltech-ucsd birds-200-2011 dataset. Tech. Rep. CNS-TR-2011-001, California Institute of Technology (2011)

31. Wang, Y., Huang, G., Song, S., Pan, X., Xia, Y., Wu, C.: Glance and focus: A dynamic approach to reducing spatial redundancy in image classification. In: Advances in Neural Information Processing Systems (2020)

32. Wang, Y., Huang, R., Song, S., Huang, Z., Huang, G.: Not all images are worth 16x16 words: Dynamic transformers for eficient image recognition. In: Advances in Neural Information Processing Systems (2021)

33. Wang, Y., Yue, Y., Wang, H., Jiang, H., Han, Y., Ni, Z., Pu, Y., Shi, M., Lu, R., Yang, Q., Zhao, A., Xia, Z., Song, S., Huang, G.: Emulating human-like adaptive vision for eficient and flexible machine visual perception. Nature Machine Intelligence (2025)

34. Yin, H., Vahdat, A., Alvarez, J.M., Mallya, A., Kautz, J., Molchanov, P.: A-vit: Adaptive tokens for eficient vision transformer. In: IEEE Conference on Computer Vision and Pattern Recognition. pp. 10809–10818 (2022)

35. Zhang, N., Donahue, J., Girshick, R., Darrell, T.: Part-based R-CNNs for finegrained category detection. In: European Conference on Computer Vision. pp. 834–849 (2014)

# Adaptive Observation Control for Active Visual Recognition Supplementary Material

Dikai Li, Weijia Wang<sup>\*</sup>, and Yuan-Gen Wang<sup>⋆</sup>

School of Artificial Intelligence, Guangzhou University, Guangzhou, China dikaili@e.gzhu.edu.cn, wwj@gzhu.edu.cn, wangyg@gzhu.edu.cn

This supplement provides the training schedules and reproducibility notes, best-versus-last checkpoint sensitivity, the budget-matched continuation diagnostic, per-step early-exit diagnostics, learned-scale statistics, qualitative fixation examples, and encoder-cost breakdown referenced by the main paper.

## 1 Additional Reproducibility Details

Table 1 lists the dataset-specific schedules used for the original main-table checkpoints. Those runs use four focus steps, AdamW, weight decay 0.05, cosine learning rate decay with 5 warmup epochs, label smoothing 0.0, no mixup/cutmix/random erasing, RandAugment, and PPO updates with clip ratio 0.2. The PPO policy standard deviation is scheduled from 0.20 to 0.07 with a cosine decay, except on ImageNet-R, which uses 0.25 to 0.10 and disables the grid-action clipping used elsewhere. In non-exit runs, both the executed scale action and the PPO latent scale coordinate are clamped to the anchor for the first 8 epochs, so the crop field of view stays at the baseline and the scale head is efectively optimized only after warmup; foveation ramps from 0 to 0.5 over epochs 10–20, and detail fusion ramps from 0 to 0.5 over epochs 10–18; early-exit fine-tuning starts from +S+F+D with all three components at full strength. Reported epochs are 1-based throughout. Checkpoints are selected by the public validation/reporting split Acc@1, matching the released training logs. Distributed evaluation follows the released AdaptiveNN-style sampler; when a validation split is not divisible by the number of processes, duplicate padding samples can introduce tiny changes, so noise-level margins in the main paper are not interpreted as accuracy gains. The original accuracy tables are regenerated directly from the training logs by a script included with the code; training scripts and evaluation code will be released.

Table 4 reports the three-seed paired results (seeds 0–2) summarized in the main paper; the matched-budget component gain +S+F+D−Orig is small but positive on Cars, Aircraft, CUB, and Dogs, and negative-to-zero on Oxford-Pet, which we interpret as near-parity rather than established equivalence.

Table 1: Training schedules for the reported runs. BS is efective batch size. EE denotes early-exit fine-tuning from the +S+F+D checkpoint.
<table><tr><td>Dataset</td><td>Non-exit BS</td><td>Non-exit epochs</td><td>Non-exit LR</td><td>EE BS</td><td>EE epochs</td><td>EE LR</td></tr><tr><td>Aircraft</td><td>108</td><td>80</td><td>5e-5</td><td>108</td><td>20</td><td>2e-5</td></tr><tr><td>Cars</td><td>108</td><td>80</td><td>5e-5</td><td>108</td><td>20</td><td>2e-5</td></tr><tr><td>CUB</td><td>108</td><td>80</td><td>5e-5</td><td>108</td><td>20</td><td>2e-5</td></tr><tr><td>Dogs</td><td>108</td><td>80</td><td>5e-5</td><td>108</td><td>20</td><td>2e-5</td></tr><tr><td>ImageNet-1K</td><td>144</td><td>50</td><td>5e-5</td><td>144</td><td>15</td><td>2e-5</td></tr><tr><td>ImageNet-A</td><td>108</td><td>60</td><td>5e-5</td><td>108</td><td>20</td><td>2e-5</td></tr><tr><td>ImageNet-R</td><td>144</td><td>80</td><td>1e-4</td><td>144</td><td>20</td><td>5e-5</td></tr><tr><td>Oxford-Pet</td><td>108</td><td>80</td><td>5e-5</td><td>108</td><td>20</td><td>2e-5</td></tr></table>

Table 2: External oficial-codebase rerun versus the same-codebase Orig anchor used for ablations (Acc@1 at the selected epoch). The main paper reports Oficial as the external fixed-step reference and uses Orig only inside in-codebase ablations; this table records their gap explicitly.
<table><tr><td>Dataset</td><td>Official</td><td>Orig</td><td>Orig-Official</td></tr><tr><td>Aircraft</td><td>77.318</td><td>76.358</td><td>-0.960</td></tr><tr><td>Cars</td><td>83.076</td><td>81.696</td><td>-1.380</td></tr><tr><td>CUB</td><td>84.524</td><td>84.369</td><td>-0.155</td></tr><tr><td>Dogs</td><td>91.981</td><td>92.051</td><td>+0.070</td></tr><tr><td>ImageNet-1K</td><td>82.126</td><td>82.116</td><td>-0.010</td></tr><tr><td>ImageNet-A</td><td>64.815</td><td>65.873</td><td>+1.058</td></tr><tr><td>ImageNet-R</td><td>85.133</td><td>85.467</td><td>+0.334</td></tr><tr><td>Oxford-Pet</td><td>96.402</td><td>96.537</td><td>+0.135</td></tr></table>

## 1.1 Paired uncertainty on ImageNet-1K

We evaluate fixed Full+EE, +S+F+D, and +S+D checkpoints on all 50,000 validation images with the same prediction-dump protocol (evaluation batch size 48) and align their predictions by sample identity. Their full-step Acc@1 values under this protocol are 82.048%, 82.188%, and 82.260%, respectively. The paired statistics below use these predictions; the main-paper accuracy tables report results from the original evaluation runs.

We obtain unadjusted 95% percentile confidence intervals from 10,000 paired bootstrap resamples (seed 0). We also apply two-sided exact McNemar tests to the discordant predictions, with Holm correction for the two comparisons with Full+EE. Diferences and interval endpoints are in percentage points.

<table><tr><td>Full+EE minus Difference</td><td>95% CI Exact  / Holm p</td></tr><tr><td>+S+F+D -0.140</td><td>[-0.332, 0.056]0.1611 0.1611</td></tr><tr><td> $+ \mathrm { S } { + } \mathrm { D }$ </td><td>-0.212 [−0.404, -0.020] 0.0319 / 0.0638</td></tr></table>

Neither McNemar comparison reaches the Holm-adjusted 0.05 threshold. The intervals describe image-level sampling uncertainty for the fixed models, whereas variation across training seeds requires separate repeated runs.

Table 3: Best-versus-last Acc@1 sensitivity for the main table rows. Each cell is selected-best / final-epoch Acc@1. The middle column selects among component-added non-exit variants, excluding Orig. +S+F ties +S+D on ImageNet-A.
<table><tr><td>Dataset</td><td colspan="2">Orig</td><td colspan="2">Best added non-exit variant</td><td colspan="2">Full+EE</td></tr><tr><td>Aircraft</td><td>76.358</td><td>75.848</td><td>77.078</td><td>76.838  $( + \mathrm { { S + F + D } ) }$ </td><td>77.558</td><td>77.258</td></tr><tr><td>Cars</td><td>81.696</td><td>81.248</td><td>82.741</td><td> $8 2 . 4 4 2 \ ( + \mathrm { S + F + D } )$ </td><td>83.711</td><td>83.648</td></tr><tr><td>CUB</td><td>84.369</td><td>84.265</td><td>84.334</td><td>84.110  $( + \mathrm { { S + F + D } ) }$ </td><td>84.800</td><td>84.455</td></tr><tr><td>Dogs</td><td>92.051</td><td>91.702</td><td>92.401</td><td>91.748  $( + \mathrm { S } + \mathrm { D } )$ </td><td>92.413</td><td>92.016</td></tr><tr><td>ImageNet-1K</td><td>82.116</td><td>81.280</td><td>82.282</td><td>80.896  $( + \mathrm { S } { + } \mathrm { D } )$ </td><td>82.124</td><td>81.946</td></tr><tr><td>ImageNet-A</td><td>65.873</td><td>64.550</td><td>66.138</td><td>65.212  $( + \mathrm { S } { + } \mathrm { D } )$ </td><td>66.534</td><td>65.873</td></tr><tr><td>ImageNet-R</td><td>85.467</td><td>84.367</td><td>85.900</td><td>84.867  $( + \mathrm { S } { + } \mathrm { D } )$ </td><td>85.633</td><td>84.900</td></tr><tr><td>Oxford-Pet</td><td>96.537</td><td>96.266</td><td>96.456</td><td>96.131  $( + \mathrm { S } { + } \mathrm { D } )$ </td><td>96.591</td><td>96.591</td></tr></table>

Table 4: Three-seed paired results (seeds 0–2) for the three main arms. Cells are mean Acc@1 ± standard deviation; the last three columns are paired deltas. These support the seed-0 main table and the near-parity reading of Oxford-Pet.
<table><tr><td>Dataset</td><td> $\mathrm { O r i g }$ </td><td> $+ \mathrm { S } { \mathrm { + F } } { \mathrm { + D } }$ </td><td>Full+EE</td><td> $\mathrm { F u l l + E E { \mathrm { - } } O r i g }$ </td><td> $\mathrm { F u l l + E E - ( + S { + } F { + } D ) }$ </td><td> $+ \mathrm { { S + F + D - O r i g } }$ </td></tr><tr><td>Aircraft</td><td>76.398±0.035</td><td>76.748±0.330</td><td>77.308±0.312</td><td>+0.910±0.337</td><td>+0.560±0.092</td><td> $+ 0 . 3 5 0 { \scriptstyle \pm 0 . 3 6 0 }$ </td></tr><tr><td>CUB</td><td>84.210±0.348</td><td>84.302±0.183</td><td>84.745±0.187</td><td>+0.535±0.165</td><td>+0.443±0.020</td><td> $+ 0 . 0 9 2 { \scriptstyle \pm 0 . 1 7 6 }$ </td></tr><tr><td> $\mathrm { D o g s }$ </td><td>92.273±0.248</td><td>92.331±0.107</td><td>92.514±0.155</td><td> $+ 0 . 2 4 1 { \pm } 0 . 1 0 8$ </td><td>+0.183±0.059</td><td> $+ 0 . 0 5 8 { \scriptstyle \pm 0 . 1 4 1 }$ </td></tr><tr><td>Oxford-Pet</td><td>96.564±0.098</td><td> $9 6 . 4 4 7 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td>96.537±0.054</td><td> $- 0 . 0 2 7 \pm 0 . 0 9 8$ </td><td>+0.090±0.062</td><td> $- 0 . 1 1 7 \pm 0 . 0 6 8$ </td></tr><tr><td>Cars</td><td>81.632±0.457</td><td> $8 2 . 7 3 1 { \pm } 0 . 0 6 3$ </td><td>84.074±0.315</td><td> $+ 2 . 4 4 2 { \pm } 0 . 5 9 5$ </td><td> $+ 1 . 3 4 3 { \pm } 0 . 3 2 7$ </td><td> $+ 1 . 0 9 8 { \pm } 0 . 5 1 8$ </td></tr></table>

## 1.2 Training-seed variation on ImageNet-R

We further examine training-seed variation on the fixed ImageNet-R split (split seed 0), using training seeds 0–2. All runs use one GPU with gradient accumulation to an efective batch of 144. This study is separate from the original three-GPU seed-0 runs.

<table><tr><td></td><td>Variant Acc@1 (%, mean ± sample SD)</td></tr><tr><td> $+ \mathrm { S } { + } \mathrm { D }$ </td><td> $8 5 . 1 6 7 \pm 0 . 3 7 9$ </td></tr><tr><td> $+ \mathrm { S + F + D }$ </td><td> $8 5 . 2 6 7 \pm 0 . 1 2 0$ </td></tr><tr><td> $\mathrm { F u l l { + } E E }$ </td><td> $8 5 . 1 3 3 \pm 0 . 3 6 7$ </td></tr></table>

We also compute paired 95% Student-t intervals across the three seeds, with two degrees of freedom. Diferences and interval endpoints are in percentage points.
<table><tr><td>Comparison</td><td>Mean difference 95% paired interval</td><td></td></tr><tr><td> $+ \mathrm { S + F + D }$  minus +S+D</td><td>+0.100</td><td>[−0.622, 0.822]</td></tr><tr><td>Full+EE minus  $+ \mathrm { S + F + D }$ </td><td>-0.133</td><td>[-1.168,0.901]</td></tr><tr><td>Full+EE minus  $+ \mathrm { S } { + } \mathrm { D }$ </td><td>-0.033</td><td>[−1.786, 1.719]</td></tr></table>

The mean accuracies are close, but the three-seed intervals remain wide. This study characterizes the observed seed variation without establishing equivalent performance.

Table 5: Budget-matched continuation diagnostic: 30-epoch single-GPU continuation with matched efective batch and learning-rate schedule (Acc@1/Acc@5 at the selected epoch). ∆Orig+30 and $\Delta { + } \mathrm { S + } \mathrm { F + } \mathrm { D + } 3 0$ are Full+EE+30 minus the corresponding continued baseline. Scale-cost+30 is a diagnostic +S+F+D continuation with an auxiliary scale-cost penalty; it is not used by any reported main-paper run and was run only for Aircraft, CUB, and ImageNet-A. This diagnostic was run only for Aircraft, CUB, and ImageNet-A.
<table><tr><td>Dataset</td><td>Orig+30</td><td> $+ \mathrm { S } + \mathrm { F } + \mathrm { D } + 3 0$ </td><td>Full+EE+30</td><td>Scale-cost+30</td><td>∆Orig+30</td><td>Δ+S+F+D+30</td></tr><tr><td>Aircraft</td><td>79.718 /96.910 @24</td><td>80.168 /97.120 @30</td><td>80.378 /97.000 @28</td><td>79.748/97.030 @27</td><td>+0.660</td><td>+0.210</td></tr><tr><td>Cars</td><td>86.121 /98.222 @27</td><td>86.668 /98.408 @25</td><td>86.880 /98.346 @24</td><td></td><td>+0.759</td><td>+0.211</td></tr><tr><td>CUB</td><td>85.554/97.428 @30</td><td>85.623 /97.204 @30</td><td>85.692 /97.273 @24</td><td>85.433/97.290 @29</td><td>+0.138</td><td>+0.069</td></tr><tr><td>Dogs</td><td>92.296/99.510 @4</td><td>92.226 /99.510 @5</td><td>92.378 /99.499 @3</td><td></td><td>+0.082</td><td>+0.152</td></tr><tr><td>ImageNet-A</td><td>67.328 /82.407 @23</td><td>66.402 /82.804 @18</td><td>66.005/82.011 @15</td><td>66.270/81.746@29</td><td>-1.323</td><td>-0.397</td></tr><tr><td>Oxford-Pet</td><td>96.645 /99.919 @22</td><td>96.429 /99.946 @20</td><td>96.456/99.756 @14</td><td></td><td>-0.189</td><td>+0.027</td></tr></table>

## 2 Budget-Matched Continuation Diagnostic

Table 5 reports the budget-matched continuation summarized in the main paper: a 30-epoch single-GPU continuation of Orig, +S+F+D, and Full+EE with matched efective batch and learning-rate schedule. Extra training of the fixedstep Orig model is itself a strong accuracy baseline, so once the budget is matched the proposed components yield dataset-dependent gains rather than a uniform improvement. This controls only the accuracy axis and does not give the fixedstep model the learned per-fixation scale or the confidence-conditioned operating points of the early-exit frontier.

## 3 Early-Exit Diagnostics

## 3.1 Early exit initialized from ${ \bf + S + D }$

We fine-tune early-exit models from +S+D and +S+F+D on ImageNet-R for 20 epochs at learning rate $5 \times 1 0 ^ { - 5 }$ . The efective training batches are 48 and 144, respectively. The models therefore difer in both their starting configurations and training batches.

We fit stopping thresholds by five-fold cross-fitting on the same 3,000-image evaluation split and apply the rules to full-rollout predictions. The table reports Acc@1 of the +S+D-initialized model minus that of the +S+F+D-initialized model, in percentage points. The largest diference in simulated average exit steps between configurations is 0.00534.

<table><tr><td>Stopping rule / target steps</td><td>1.5</td><td>2.0</td><td>2.5</td><td>3.0</td></tr><tr><td>Max-softmax</td><td></td><td> $+ 0 . 1 6 7 + 0 . 0 6 7$ </td><td>0.000-0.067</td><td></td></tr><tr><td>Learned stop head</td><td></td><td>-0.167-0.300-0.167-0.433</td><td></td><td></td></tr></table>

The +S+D-initialized model does not consistently improve accuracy across stopping rules and budgets. The comparison evaluates an alternative early-exit configuration, with training-batch diferences retained as a source of confounding.

Table 6: Per-step Acc@1 for the full early-exit checkpoint without early stopping.
<table><tr><td>Dataset</td><td>Step 1</td><td>Step 2</td><td>Step 3</td><td>Step 4</td></tr><tr><td>Aircraft</td><td>71.347</td><td>74.377</td><td>76.508</td><td>77.558</td></tr><tr><td>Cars</td><td>76.200</td><td>80.291</td><td>82.753</td><td>83.711</td></tr><tr><td>CUB</td><td>82.678</td><td>83.834</td><td>84.438</td><td>84.800</td></tr><tr><td>Dogs</td><td>90.688</td><td>91.713</td><td>92.040</td><td>92.413</td></tr><tr><td>ImageNet-1K</td><td>79.704</td><td>81.106</td><td>81.810</td><td>82.040</td></tr><tr><td>ImageNet-A</td><td>56.746</td><td>61.772</td><td>63.757</td><td>66.534</td></tr><tr><td>ImageNet-R</td><td>82.033</td><td>83.700</td><td>84.533</td><td>85.633</td></tr><tr><td>Oxford-Pet</td><td>95.942</td><td>95.969</td><td>96.402</td><td>96.591</td></tr></table>

Table 7: Mean learned scale per focus step for the full early-exit checkpoint without early stopping. The last column is the standard deviation of the learned scale across validation images, averaged over the four steps.
<table><tr><td>Dataset</td><td>Step 1</td><td>Step 2</td><td>Step 3</td><td>Step 4</td><td>Avg. std</td></tr><tr><td>Aircraft</td><td>1.006</td><td>1.015</td><td>1.011</td><td>1.008</td><td>0.007</td></tr><tr><td>Cars</td><td>1.110</td><td>1.167</td><td>1.167</td><td>1.215</td><td>0.077</td></tr><tr><td>CUB</td><td>1.006</td><td>1.010</td><td>1.008</td><td>1.012</td><td>0.008</td></tr><tr><td>Dogs</td><td>1.002</td><td>1.003</td><td>1.002</td><td>1.002</td><td>0.002</td></tr><tr><td>ImageNet-1K</td><td>1.078</td><td>1.162</td><td>1.175</td><td>1.106</td><td>0.055</td></tr><tr><td>ImageNet-A</td><td>1.004</td><td>1.003</td><td>1.004</td><td>1.003</td><td>0.003</td></tr><tr><td>ImageNet-R</td><td>1.150</td><td>1.155</td><td>1.194</td><td>1.175</td><td>0.059</td></tr><tr><td>Oxford-Pet</td><td>1.055</td><td>1.076</td><td>1.051</td><td>1.070</td><td>0.058</td></tr></table>

Table 6 reports per-step Acc@1 for the full early-exit checkpoint evaluated without early stopping. Accuracy generally increases across focus steps, supporting the use of confidence-conditioned stopping as an inference-time operating point.

Table 7 gives the mean learned scale over the four focus steps for the same checkpoints. Aircraft, CUB, Dogs, and ImageNet-A mostly stay near the anchor scale, while Cars, ImageNet-1K, ImageNet-R, and Oxford-Pet use larger average fields of view.

Figure 3 illustrates the detail-token contribution on Stanford Cars, the dataset with the most consistent detail gains. The comparison uses +S+F and +S+F+D; both enable scale and foveation and execute four focus steps, with detail enabled only in the latter. A deterministic re-evaluation of the 8,041-image validation split finds 379 examples where detail fusion corrects a no-detail error against 290 reverse cases. The shown examples are same-make confusions (Leaf vs. Juke, Continental trim variants, F-150 model years) whose discriminative evidence is fine-grained. An analogous comparison of foveation configurations (+S+D versus +S+F+D, detail branch fixed) flips 328 validation examples right against 290 reverse cases and supplies the foveation row of the main-paper qualitative figure.

Figure 4 reproduces the full accuracy-compute diagnostic: confidenceconditioned exits against fixed-step truncation of the same Full+EE checkpoint.

![](images/7669694c5780654610e89f28bee1e4ddc184c768799e66860cc91e2ba415d0b8.jpg)  
Fig. 1: Qualitative fixation trajectories on ImageNet validation examples at $\tau = 0 . 7 0 .$ Each row shows the input and four predicted focus boxes from the Full+EE checkpoint. The overlay reports learned scale s and stop probability $p = \sigma ( q _ { t } )$ ; asterisks identify the displayed stop-probability annotations.

The main paper isolates the realized sequential panels; here, ImageNet-R is included as a simulation-only stop-rule diagnostic.

![](images/ffdfb55bec1493bbf93173ddfffe22228411776b30352457b07073e91a4160a1.jpg)  
Fig. 2: Additional ImageNet-1K fixation-density contrasts, including the adaptive failure case. Fixed density accumulates four fixed-scale crops; adaptive density accumulates only the crops executed at $\tau = 0 . 7 0$ . Heat intensity remains a per-example fixationdensity map rather than a dataset-level fixation distribution.

case

Detail win GT: Nissan Leaf Hatchback 2012 same make: Nissan +S+F: err +S+F+D: ok

input  
![](images/54a4aa00c55bc1f65729a0b9b1693f4f9d4814561b4b7e971ba958b3b8d48e8b.jpg)

+S+F (no detail)  
![](images/bda6f799b26016e2929548ad9a13cd4ceaded6c438724db131f124b5145c7892.jpg)

+S+F+D (detail)  
![](images/30690ee4bdbde15d8a8ff53c171c4b62780cf12e8342d609c4575767ac15616e.jpg)  
Detail win GT: Bentley Continental Supersports Conv. Convertible 2012 same make: Bentley +S+F: err +S+F+D: ok

![](images/1164bc86ba79054752f054763c25a4c6af88c91906a635c04345efb3e67b0d0b.jpg)

![](images/c5aef1521c1dbfafd403e59767693402cd6be10bf3b5919e114b72b95df73cee.jpg)

![](images/82809cf32e6250df8448af58c15590b0e9d788f8a6d2baa26721488512139e38.jpg)  
Detail win GT: Ford F-150 Regular Cab 2012 same make: Ford +S+F: err +S+F+D: ok

![](images/b480fbc52afe64f3081c536278a7b3e628403cd72fd4b2a90e077e96565816ed.jpg)

![](images/08c7796e7de98767aa0dcdd5a0fc8ee0066d0ec5f60c54a7a83e607cca85773e.jpg)

![](images/2f369d2cd9569c26ac9f23a4082150e6e445e6ac54b2660e3f3892d1d10b6fed.jpg)  
Fig. 3: Stanford Cars validation cases where detail fusion flips the prediction. Each row shows the input and the executed four-step fixations of the no-detail (+S+F) and detail $\mathrm { ( + S { + } F { + } D ) }$ ablation checkpoints; numbered boxes mark fixation order, heat shading marks accumulated crop coverage, and each label reports the prediction with its confidence. All three cases are same-make confusions corrected by the detail-enabled checkpoint.

![](images/2cca92c041efb9aad49c4f914351c678e18964a8cd7c3e3952c7d69c94708823.jpg)

![](images/a5079e039b0c62a4a49f34a35e545d22a67011deaff91c74944eff779d27d207.jpg)  
-- fixed-step truncationconfidence-conditioned exit

![](images/7b7cb30acb60cbdf1d2e4819688b0bc5152b52365f665529997641e111d034d2.jpg)  
Fig. 4: Accuracy versus average focus steps for the Full+EE checkpoint. Dashed: truncating every sequence at a fixed step (1–4). Red markers: confidence-conditioned exit for $\tau \in \{ 0 . 3 0 , 0 . 5 0 , 0 . 7 0 , 0 . 9 0 \}$ , using realized sequential-step measurements where available and otherwise the full-forward stop-rule simulation. Sequential-step points measure realized exits; simulated points analyze the stop rule without claiming realized savings.

![](images/10c5837f0f0343a4bd301d36f835f2fbd527e1658c5e54d15cbcff066d1ad941.jpg)  
Fig. 5: Simulation-only ImageNet-R stop-rule diagnostic from full four-step rollouts. This panel helps separate the shape of the learned stopping criterion from realized sequential FLOP savings, whose estimates use physically executed step counts only for Seq. rows in the main table.

## 4 Computational Cost

The cost breakdown uses a conservative encoder-pass estimate. Each glance, focus, or detail pass contributes 1.10 GFLOPs under the DeiT-S multiply-add convention used for this breakdown. For Full+EE, the estimate is 1.10 + 2.20K<sup>¯</sup> GFLOPs, where K<sup>¯</sup> is the average number of selected focus steps. It counts one detail pass at every selected step, including the first, although the implementation enables detail only from the second focus step onward. Four-step estimates are therefore 5.50 GFLOPs for Orig and 9.90 GFLOPs for Full+EE under the conservative convention. Crop sampling, policy and classification heads, feature recovery, and other non-encoder operations are outside this estimate. The Seq. rows use physically executed step counts; Sim. rows give analytic costs for stops selected from full four-step rollouts and do not measure realized savings.

## 4.1 Model-forward runtime and memory

We profile ImageNet-1K inference on an NVIDIA RTX 4090 with PyTorch 2.5.1+cu121 and CUDA 12.1. CUDA-event timing covers model-forward execution and excludes disk I/O, preprocessing, and host-to-device transfer. Each policy is measured after 30 warmup batches, followed by 100 timed batches.

For adaptive inference, we use classifier max-softmax as the stopping score. At each target step budget, five-fold threshold fitting produces five thresholds, whose mean is used as one fixed threshold for profiling. At the two-step target, this threshold is 0.914015, and the timed batch-1 images execute 2.04 focus steps on average. Timing uses this single threshold and a profiling subset; accuracy evaluation with fold-specific thresholds is a separate protocol.

Table 8: Conservative encoder-pass cost estimate (GFLOPs, multiply-adds; nonencoder operations excluded) of the Full+EE checkpoint at each early-exit threshold, computed as glance + average steps × (focus + detail). Simulated rows are analytic stop-rule costs rather than realized sequential savings. The original model at four fixed steps costs 5.50 GFLOPs under the same proxy.
<table><tr><td></td><td></td><td colspan="2">τ = 0.70</td><td colspan="2">τ = 0.90</td><td>Full</td></tr><tr><td>Dataset</td><td>Protocol</td><td>Avg. step</td><td>GFLOPs</td><td>Avg. step</td><td>GFLOPs</td><td>GFLOPs</td></tr><tr><td>Aircraft</td><td>Sim.</td><td>3.67</td><td>9.17</td><td>4.00</td><td>9.90</td><td>9.90</td></tr><tr><td>Cars</td><td>Sim.</td><td>3.99</td><td>9.87</td><td>4.00</td><td>9.90</td><td>9.90</td></tr><tr><td>CUB</td><td>Sim.</td><td>3.71</td><td>9.26</td><td>4.00</td><td>9.90</td><td>9.90</td></tr><tr><td>Dogs</td><td>Sim.</td><td>3.99</td><td>9.88</td><td>4.00</td><td>9.90</td><td>9.90</td></tr><tr><td>ImageNet-1K</td><td>Seq. steps</td><td>1.73</td><td>4.91</td><td>2.45</td><td>6.48</td><td>9.90</td></tr><tr><td>ImageNet-A</td><td>Sim.</td><td>3.96</td><td>9.81</td><td>4.00</td><td>9.90</td><td>9.90</td></tr><tr><td>ImageNet-R</td><td>Sim.</td><td>1.44</td><td>4.26</td><td>1.93</td><td>5.33</td><td>9.90</td></tr><tr><td>Oxford-Pet</td><td>Seq. steps</td><td>1.34</td><td>4.05</td><td>4.00</td><td>9.89</td><td>9.90</td></tr></table>

<table><tr><td>Batch-1 policy</td><td>Mean (ms) P90 (ms)</td><td>Peak allocated (MiB)</td></tr><tr><td>Orig, fixed 4</td><td>52.23</td><td>543.46</td></tr><tr><td>+S+D, fixed 4</td><td>71.88</td><td>547.06</td></tr><tr><td>Full+EE, fixed 2</td><td>37.63</td><td>515.91</td></tr><tr><td>Full+EE, adaptive target 2</td><td>38.40</td><td>547.64</td></tr><tr><td> ${ \mathrm { F u l l } } + { \mathrm { E E } } ,$  fixed 4</td><td>71.33</td><td>547.64</td></tr></table>

Adaptive inference reduces mean batch-1 latency relative to fixed four-step Full+EE inference, while P90 changes little. Fixed two-step inference has similar mean latency and a lower P90.
<table><tr><td>Policy</td><td>Batch 32 Batch 48 (images/s) (images/s)</td><td>Batch 48 peak (MiB)</td></tr><tr><td>Orig, fixed 4</td><td>562.1</td><td>855.1 718.85</td></tr><tr><td>+S+D, fixed 4</td><td>411.4</td><td>616.0</td></tr><tr><td>Full+EE, fixed 2</td><td>756.6</td><td>732.14 1115.9 691.19</td></tr><tr><td>Full+EE, adaptive target 2</td><td>409.0</td><td>620.3 701.59</td></tr><tr><td>Full+EE, fixed 4</td><td>423.5</td><td>613.1 734.37</td></tr></table>

The batch-32 and batch-48 measurements cover 3,200 and 4,800 timed images, respectively. Adaptive execution has no consistent throughput advantage over fixed four-step Full+EE inference in this implementation. The Orig and +S+D rows provide additional runtime controls under the same profiler.

## 4.2 Recorded training cost

In the recorded three-GPU ImageNet-1K +S+D run, training takes approximately 1.4 wall-clock hours per epoch, corresponding to 4.2 GPU-hours. The 50-epoch run therefore costs approximately 210 GPU-hours. The logged earlyexit fine-tuning epochs take approximately 1.7 wall-clock hours each on three GPUs, or 5.1 GPU-hours per epoch. The early-exit figure is a per-epoch estimate from the available log segment.