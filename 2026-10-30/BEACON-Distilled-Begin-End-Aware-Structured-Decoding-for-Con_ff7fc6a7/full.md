# BEACON: Distilled Begin/End-Aware Structured Decoding for Continuous Sign Language Segmentation

Guohou Zhang<sup>1</sup> Kalin Stefanov<sup>2</sup>

<sup>1</sup>Monash University, Australia <sup>2</sup>Japan Advanced Institute of Science and Technology (JAIST), Japan gzha0067@student.monash.edu kalins@jaist.ac.jp

## Abstract

In segmentation-based continuous sign language recognition, an untrimmed video is first partitioned into lexical units; segmentation is therefore a means to an end— downstream recognition. However, segmenters are usually judged by local boundary detection rather than by the qual ity of the lexical segments and the recognition they enable. We present BEACON, a begin/end-aware segmenter that combines four-class Outside/Begin/Inside/End (BIOE) labeling, constrained Viterbi decoding over eight legal transitions, and boundary knowledge distilled from a binary teacher into a shared student. Under strict no-cut evaluation with validation-based model selection, BEACON improves segment-F1 by 14.58/14.59 points and boundary-F1 by 2.17/1.22 points, and reduces the downstream word error rate (WER) by 9.10/4.15 points relative to the state-ofthe-art in the PHOENIX-14/BSL Corpus. To examine which segmentation metric better reflects downstream recognition, we evaluatedfive distinct architecturefamilies. Segment-F1 aligns more consistently with lower WER than boundary-F1 in both corpora (Spearman ρ = −1.0/ − 0.9 versus −0.9/ − 0.8). These results support evaluating sign language segmentation in terms of segment quality and downstream recognition utility rather than local boundary accuracy alone.

## 1. Introduction

In segmentation-based recognition pipelines, untrimmed sign language videos are first partitioned into lexical units that can be passed to an isolated sign language recognizer. Since signers produce fluid, co-articulated motions without explicit pauses, automatic segmentation remains an important bottleneck for pipelines that require localized lexical units; segmentation errors propagate into downstream recognition and contribute to a higher word error rate (WER) [14, 19]. Segmentation is therefore a means to an end, a perspective we make central both to how we evaluate segmenters and to how we design them.

![](images/6f10970acc6fbbbc01655ca99171e40e3c0bf95a8de7847f7139d7fea80ddade.jpg)  
Figure 1. BEACON replaces Renz et al. [19]’s binary boundary mask—which collapses sign onset/offset—with a 4-class BIOE labeling (explicit Outside/Begin/Inside/End states) and a constrained Viterbi decoder whose eight legal transitions make every output a well-formed segment sequence by construction. This design improves segment quality while retaining boundary accuracy and lowers downstream recognition WER.

Renz et al. [19] formulate segmentation as binary boundary detection: each frame is labeled either “inside a sign” or “at a boundary”. A multi-stage temporal convolutional network (MS-TCN) [1] is trained with binary cross-entropy, followed by simple argmax decoding and an endpointcutting post-processing step. The released evaluation procedure uses ground-truth evaluation endpoints for this operation; we revisit it in Sec. 4. This approach established a reproducible protocol on PHOENIX-14 [7] and the BSL Corpus [21] using boundary-F1 and segment-F1. Its binary representation, however, treats each frame only as either sign-internal or boundary, collapsing onset and offset into a single “boundary” class. It therefore does not explicitly represent how frames form a complete sign segment—where the segment begins, continues, and ends.

We make two departures. First, we recast segmentation as structured sequence labeling: every frame is tagged O/B/I/E (Outside/Begin/Inside/End), inspired by BIO chunk-tagging [17]. The scheme is binarycompatible—it maps exactly to Renz et al. [19]’s labels, as verified on both corpora—while explicitly distinguishing sign onset and offset. Fig. 1 previews the central representational change from an undifferentiated boundary mask to a structurally valid Begin/Inside/End path. These explicit states define a small finite-state grammar; constrained Viterbi decoding [24] over eight legal BIOE transitions restricts predictions to structurally valid paths and returns a well-formed segment sequence by construction.

Global path validity does not by itself provide sharp local endpoints. A direct solution would be to pair the BIOE segmenter with a separate binary boundary model, but this would duplicate most of the temporal architecture. BEA-CON instead combines structured segment modeling and boundary modeling within a single student through knowledge distillation [6]. A trained binary boundary model is frozen as a teacher, and its boundary information is distilled into a lightweight IO branch built on the shared student encoder. The teacher is used only during training, leaving a single 5M-parameter student at inference for structured segment prediction and endpoint refinement. The two components serve complementary purposes: the fixed grammar guarantees structurally valid output sequences, while boundary distillation provides local information for more precise endpoint refinement. Matched ablations examine the contribution of each component.

Second, we ask a question the field has left implicit: Which metric should a segmenter optimize? Boundary-F1 (mF1B) is the de facto headline metric, but if segmentation serves downstream recognition, the more relevant target is the metric most closely aligned with WER. Under a matched frozen recognizer control, we compare five distinct architecture families on both PHOENIX-14 [7] and the BSL Corpus [21]. Among the tested architecture families, segment-F1 (mF1S) orders downstream WER more consistently than mF1B on both corpora. We assess this claim through each metric’s association with the independent downstream WER measure, rather than assume that segment-level metrics are inherently preferable. Accordingly, we evaluate BEACON not only by segmentation metrics, but also by whether its explicit segment structure and boundary-sensitive refinement translate into lower WER. Concretely, our contributions are threefold:

1. Single structured segmenter with distilled boundary knowledge. BEACON combines four-class BIOE prediction, constrained decoding over eight legal transitions, and training-only boundary distillation, while requiring only a single 5M-parameter model at inference. Under strict no-cut evaluation, it improves mF1S by 14.58/14.59 points and mF1B by 2.17/1.22 points over Renz et al. [19] on the PHOENIX-14/BSL Corpus.

2. Recognition under strict no-cut protocol. With validation-based checkpoint and refinement-parameter selection, three seeds, and a frozen matched recognizer, BEACON without ground-truth endpoint cutting reduces WER by 9.10/4.15 points relative to Renz et al. [19] on the PHOENIX-14/BSL Corpus.

3. Architecture-level analysis of segmentation metrics and downstream recognition. Across five architecture families, mF1S orders downstream WER more consistently than mF1B on both PHOENIX-14 and the BSL Corpus, providing an architecture-level test of which segmentation metric better reflects recognition utility.

## 2. Related Work

## 2.1. Sign Language Segmentation

Directly comparable baselines. Renz et al. [19] established the baseline protocol for this task using binary boundary masks, an MS-TCN [1] decoded by argmax, and an endpoint-cutting post-processing step that uses ground-truth endpoints. Performance is evaluated with mF1B/mF1S on PHOENIX-14 [7] and the BSL Corpus [21]. We use the same metrics, datasets, and I3D [3] representations, while adopting validation-based model selection and strict no-cut endpoint protocol (see Sec. 4).

Hands-On [13] is the closest recent baseline. It also casts segmentation as sequence labeling, but uses three-class BIO with argmax decoding and stronger HaMeR [16] features on the BSL Corpus. In contrast, BEACON uses an explicit E (End) class (four-class BIOE), replaces argmax with constrained Viterbi decoding [24] (valid segments by construction), and adds soft B/E auxiliary heads, a boundary-aware temporal smoothness loss, and boundary knowledge distillation into a shared student. Since Hands-On uses HaMeR features, whereas our controlled setting uses the released I3D representations, we treat it as a feature-heterogeneous reference rather than a feature-matched comparison.

Related but not directly comparable. Other segmentation work differs from our setting in dataset, supervision, or evaluation protocol. CMPL [20] uses changepointmodulated pseudo-labeling with additional unlabeled target data in a source-free domain adaptation setting. Since this differs from the fully supervised setting used by our method and primary baselines, we do not treat its reported results as a head-to-head comparison.

Belissen et al. [2] released Dicta-Sign-LSF-v2 with a sliding-window-F1 baseline rather than the mF1B/mF1S protocol. Moryossef et al. [14] studied linguistically motivated segmentation of German Sign Language using perdocument frame-F1/IoU. Farfan et al. [4] applied MS-TCN and diffusion-based temporal segmentation to Peruvian Sign Language under a different dataset and evaluation setting. Sim et al. [22] performed unsupervised changepoint segmentation. MHB [26] incorporated handshape pretraining for boundary detection but evaluated on ASLLRP-S [15] rather than the PHOENIX-14/BSL Corpus.

Since these studies differ in datasets, frame rates, annotations, supervision, or evaluation metrics, we do not treat them as directly comparable baselines. Few prior methods are evaluated under the same mF1B/mF1S protocol.

The closest approaches are positioned in Table 1 by learning setting, sequence modeling, and evaluation.

Table 1. Positioning of the closest sign-segmentation approaches. DA and PL denote domain adaptation and pseudo-labeling. Matched WER uses a fixed corpus-matched recognizer; Sec. 5 reports our controlled re-evaluation.
<table><tr><td rowspan="2">Property</td><td rowspan="2">Renz et al. [19]</td><td rowspan="2">Hands- On [13]</td><td rowspan="2">CMPL [20]</td><td rowspan="2">BEACON</td></tr><tr><td></td></tr><tr><td>Setting</td><td>Supervised</td><td>Supervised</td><td>Source-free DA</td><td>Supervised</td></tr><tr><td>Input</td><td>I3D</td><td>HaMeR + 3D angles</td><td>I3D</td><td>I3D</td></tr><tr><td>Temporal model MS-TCN</td><td></td><td>Transformer</td><td>MS-TCN</td><td>TCN + Transformer</td></tr><tr><td>Labels</td><td>Binary</td><td>BIO</td><td>Binary</td><td>BIOE + IO</td></tr><tr><td>Readout</td><td>argmax</td><td>argmax</td><td>argmax</td><td>Constrained Viterbi</td></tr><tr><td rowspan="2">Boundary cue</td><td>GT endpoint</td><td></td><td>CP-</td><td>Distilled IO</td></tr><tr><td>cut</td><td></td><td>modulated PL</td><td></td></tr><tr><td>Matched WER</td><td>No</td><td>No</td><td>No</td><td>Yes</td></tr></table>

## 2.2. Structured Prediction with Transition Graphs

Structured sequence labeling commonly combines explicit label schemes such as BIO [17] and BIOES [9, 18] with structured prediction models such as CRF [8]. Our BIOE scheme adapts BIOES by retaining explicit outside, begin, inside, and end states while omitting a separate singleton S state, yielding a compact four-state grammar with eight legal frame-to-frame transitions.

Because this label grammar is known a priori, we enforce these transitions directly with a non-learned constrained Viterbi decoder [24], rather than learning transition scores from data. This guarantees that the decoded BIOE path satisfies the declared segment grammar by construction. We compare this fixed transition graph with a learned-transition CRF control in Sec. 5.3.

## 2.3. Downstream Impact on Recognition

Prior segmentation methods [2, 13, 19, 26] mostly stop at boundary- or segment-level scores; for example, MHB [26] reports boundary detection on ASLLRP-S but does not report downstream recognition WER under a matched recognizer. To our knowledge, none of these methods reports downstream recognition WER under matched recognizer control or examines which segmentation metric best aligns with it. We close the loop on both corpora with a frozen per-corpus recognizer: a partial continuous sign language recognition pipeline (segmenter followed by an isolated sign recognizer), not an end-to-end CTC [5] system. Freezing the recognizer isolates the segmentation effect, and the five-architecture-family analysis tests whether mF1S orders WER more consistently than mF1B (see Sec. 5.2).

## 3. Method

## 3.1. BIOE Labeling Scheme

We replace the binary boundary mask $( y _ { t } \in \{ 0 , 1 \} )$ proposed by Renz et al. [19] with a four-class sequence $y _ { t } \in$ {O, B, I, E} (Outside/Begin/Inside/End). For multi-frame signs, the first and last frames are labeled B and E, with intervening frames labeled I; single-frame annotations (only 0.31%/0.35% of PHOENIX-14/BSL training frames) are assigned E. Because the eight-transition grammar has no singleton path, their evidence contributes to the highestscoring valid path; their rarity motivates the compact fourstate representation rather than a dedicated S class. The map O 7→ 1, {B, I, E} 7→ 0 recovers the binary labels exactly, so any binary boundary metric remains computable while providing explicit begin, inside, and end states for structured prediction. At inference, we restrict decoding to eight legal frame-to-frame transitions, and a constrained Viterbi decoder [24] (see Sec. 3.5) keeps the highest emission path among them, so every decoded path satisfies the declared BIOE transition grammar by construction. Fig. 2 illustrates the legal transition decoding, and Fig. 3 summarizes the complete BEACON pipeline.

![](images/abc59f409b5763dc78bc8f40e4cd78258bb26c0ff9114a11751f3b84e265b31e.jpg)  
Figure 2. Constrained Viterbi decoding [24] over the BIOE grammar. A framewise argmax path may contain illegal transitions; the decoder instead returns the highest emission path restricted to the eight legal BIOE moves, so the decoded sequence satisfies the declared grammar. Emission totals are illustrative.

## 3.2. Encoder

Throughout our main experiments, the frame features X ∈ $\mathbb { R } ^ { T \times d _ { \mathrm { f e a t } } } ~ ( d _ { \mathrm { f e a t } } = 1 0 2 4 )$ are extracted using I3D [3]. The features are projected by a linear layer to 256 dimensions, passed through a 5-block residual TCN [10] with dilations [1, 2, 4, 8, 16] (receptive field 63 frames ≈ 2.5 s at 25 fps), augmented with sinusoidal positional encoding, and finally processed by a 4-layer pre-LN Transformer encoder [23] (8 heads, FFN 1024). The resulting hidden states $H \in$ $\mathrm { \mathbb { R } } ^ { T \times 2 5 6 }$ are shared by the BIOE and soft Begin/End heads and also serve as input to the distilled binary IO branch. The IO branch adds 3 residual temporal convolution blocks with dilations [1, 2, 4], layer normalization, and a binary sign/boundary classifier, but does not duplicate the input projection, 5-block TCN, or Transformer. The student has 5M trainable parameters; the frozen teacher is used only during training and discarded at inference.

![](images/8b6f66fd327bf3cf03b5bda7047cd5518ebd9865dac83bde9987c9ffff5f632f.jpg)  
Figure 3. The BEACON pipeline. A frozen I3D [3] backbone supplies frame features to a shared temporal encoder (Stage 1, standard) BIOE and soft Begin/End predictions are fused and decoded over the eight legal BIOE transitions (inset), producing a preliminary segment path that satisfies the declared grammar (Stage 2, ours). A binary IO branch reads the same shared hidden states H (no extra encoder) and refines endpoints within a validation-selected radius of at most ±3 frames (Stage 3, ours). During training only, a separately trained, frozen binary teacher (shown dashed, with its own encoder, and discarded after training) distills boundary evidence into this student IO branch via a KL loss; inference uses a single 5M-parameter model with one encoder. The lower strip visualizes the per-frame BIOE labels collapsing into the final segment spans.

## 3.3. Prediction Heads

The structured decoder uses three heads on the shared states H. The fine BIOE head $P _ { \mathrm { B I O E } } = \mathrm { s o f t m a x } ( W _ { f } H + b _ { f } ) \in$ $\mathbb { R } ^ { T \times 4 }$ is the primary output (class-weighted cross-entropy, label smoothing 0.1). Two soft B/E heads $\begin{array} { r l } { p _ { B } , p _ { E } } & { { } = } \end{array}$ $\sigma ( \cdot ) \in \mathbb { R } ^ { T }$ are each trained by binary cross-entropy $( \lambda _ { B } =$ $\lambda _ { E } = 0 . 4 )$ against Gaussian-shaped targets centered on the ground-truth begin/end frames.

## 3.4. Loss Function

Supervised training combines the 4-class loss, binary IO loss, Begin/End auxiliaries, and temporal regularization:

$$
L _ { \mathrm { s u p } } = L _ { \mathrm { B I O E } } + L _ { \mathrm { I 0 } } + 0 . 4 ( L _ { B } + L _ { E } ) + 0 . 0 5 L _ { \mathrm { s m o o t h } } .
$$

Here, $L _ { \mathrm { I O } }$ is a class-weighted 2-class cross-entropy loss using the binary boundary masks of Renz et al. [19] as targets. Its classes are sign/internal and boundary/outside: ${ y _ { t } ^ { \mathrm { I O } } } \bar { = } 1$ for O frames and 0 for B/I/E frames, with respective class weights [1, 4]. $L _ { \mathrm { s m o o t h } }$ is a boundary-aware temporal MSE on BIOE log-probabilities,

$$
\begin{array} { r } { \left\| \log P _ { \mathrm { B I O E } } ( t ) - \log P _ { \mathrm { B I O E } } ( t - 1 ) \right\| _ { 2 } ^ { 2 } , } \end{array}
$$

masked to zero within ±2 frames of any ground-truth boundary, penalizing interior jitter while permitting legitimate transitions. For binary IO logits $z _ { k }$ define,

$$
q _ { k } ^ { ( T _ { d } ) } = \mathrm { s o f t m a x } ( z _ { k } / T _ { d } ) , \qquad k \in \{ \mathrm { t e a c h e r , s t u d e n t } \} .
$$

The total distillation objective [6] is,

$$
L = L _ { \mathrm { s u p } } + \lambda _ { \mathrm { K D } } T _ { d } ^ { 2 } \mathrm { K L } \Big ( q _ { \mathrm { t e a c h e r } } ^ { ( T _ { d } ) } \lVert q _ { \mathrm { s t u d e n t } } ^ { ( T _ { d } ) } \Big ) ,
$$

where $T _ { d } = 2$ and $\lambda _ { \mathrm { K D } } = 1$ . Padding frames are excluded from all loss terms.

## 3.5. Constrained Viterbi Decoder with B/E Fusion

At inference, the three structured-decoding heads are combined in steps (see Fig. 2):

1. B/E fusion gate $( \alpha = 0 . 5 , \tau = 0 . 4 ) \colon$

For the B class, the fused probability is,

$$
\tilde { P } _ { B } ( t ) = \left\{ \begin{array} { l l } { \alpha p _ { B } ( t ) + ( 1 - \alpha ) P _ { \mathrm { B I O E } } [ B ] ( t ) , } & { p _ { B } ( t ) \geq \tau } \\ { P _ { \mathrm { B I O E } } [ B ] ( t ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

with the same operation applied to the E class using $p _ { E } ( t )$ and $P _ { \mathrm { B I O E } } [ E ] ( t )$ The fused per-frame probabilities are re-normalized and log-transformed into the Viterbi emissions. In the full BEACON model, the distilled IO branch additionally contributes its boundary prediction to the O class emission before decoding, as described in Sec. 3.6.

2. Constrained Viterbi finds the highest emission path through the BIOE trellis subject to eight legal transitions:

$$
{ \begin{array} { r l } { 0 \to \{ \ 0 , \mathtt { B } \} , } & { \mathtt { B } \to \{ \mathtt { I } , \mathtt { E } \} , } \\ { \mathtt { I } \to \{ \mathtt { I } , \mathtt { E } \} , } & { \mathtt { E } \to \{ \mathtt { O } , \mathtt { B } \} . } \end{array} }
$$

We use a fixed O class emission bias $o _ { \mathrm { b i a s } } = - 1 . 0$ and transition bias $b _ { \mathrm { t r a n s } } = 0 . 0$

## 3. Endpoint policy:

The primary strict no-cut protocol uses no ground-truth endpoint cleanup. The cutting rule used by Renz et al. [19] is evaluated only in a paired compatibility audit. The final BIOE path is converted to segments by extracting every maximal B I<sub>\*</sub> E run.

## 3.6. Distilled Endpoint Refinement

The constrained decoder produces structured preliminary segments, but their endpoint precision still depends on the boundary evidence available to the BIOE predictions. To improve boundary localization while restricting changes to local endpoint positions, the student uses a binary IO branch on the shared hidden states. For each corpus and seed, a 4.41M-parameter binary teacher is trained on the same training indices and selected by validation mF1B. During student training, this teacher remains frozen and supplies softened binary targets in addition to the ground-truth binary labels.

At inference, the student’s P(boundary) contributes to the O class emission before constrained decoding using the validation-selected $\alpha _ { \mathrm { I O } }$ . A boundary run is a maximal contiguous run of boundary/outside frames in the decoded binary mask. After decoding, each run is shifted as a block toward the nearest sufficiently confident student boundary peak within the displacement radius, clipped to the video bounds, and combined with the other shifted runs; its width is preserved unless clipping or merging occurs. For each corpus and seed, after checkpoint selection we choose $\alpha _ { \mathrm { I O } } ~ \in ~ \{ 0 , 0 . 3 , 0 . 7 \}$ , peak threshold $\tau _ { \mathrm { I O } } \in \{ 0 . 3 , 0 . 4 , 0 . 5 \}$ and maximum displacement $r \in \{ 1 , 2 , 3 \}$ using the heldout validation score. The selected setting is then applied once to the test set; all six selections are reported in Supplementary Tab. 7.

This prediction-only refinement adjusts endpoint locations locally; the strict BIOE validity guarantee applies to the Viterbi path before relocation. A 2-encoder implementation is evaluated only as a computational baseline.

## 3.7. Training Protocol

We train for 50 epochs using AdamW [12] $( \mathrm { l r } = 3 \times 1 0 ^ { - 4 }$ weight decay 0.01), with a cosine learning-rate schedule ending at $1 0 ^ { - 5 }$ , batch size 8, and gradient clipping 1. We run three seeds (42, 43, 44) and report mean ± SD. Student checkpoints are selected by the balanced validation score (mF1B+mF1S)/2 under a fixed checkpoint-selection configuration: B/E fusion $\alpha \ = \ 0 . 5$ and $\tau ~ = ~ 0 . 4 .$ , O class bias −1.0, transition bias 0, and IO refinement $\alpha _ { \mathrm { I O } } = 0 . 7 \ :$ $\tau _ { \mathrm { I O } } = 0 . 4$ , and $r = 3$ . After checkpoint selection, only the IO refinement parameters in Sec. 3.6 are retuned on validation; test data are reserved for final reporting.

## 4. Experimental Setup

Datasets and features. We use the released PHOENIX-14 [7] package (4,556 train/1,115 test videos; German Sign Language) and BSL Corpus [21] package (5,413/702; British Sign Language). Both provide frame-level lexical annotations and the same 1024-D I3D [3] features used by Renz et al. [19], keeping the visual representation fixed so that the controlled comparison reflects changes in the segmentation model rather than a different visual backbone.

Neither package defines a validation split for the segmentation task. We therefore deterministically hold out 10% of the original training videos using split seed 100: 4,100/456 train/validation videos for PHOENIX-14 and 4,872/541 for the BSL Corpus. Every trained model— teachers, students, matched ablations, learned CRFs, and architecture controls—uses exactly these video indices. The train and validation partitions are video-disjoint and together cover the original training set, although recording groups can occur in both partitions (see Sec. 6).

Segmentation metrics. Following the evaluation protocol of Renz et al. [19], mF1B averages boundary F1 at temporal tolerances of 1–4 frames, while mF1S averages segment-IoU F1 over thresholds 0.40–0.75. mF1B measures local boundary placement, whereas mF1S additionally penalizes fragmented, merged, or incorrectly extended segments. All primary predictions are evaluated without endpoint cutting. The post-processing step for endpoint cutting used by Renz et al. [19] reads ground-truth endpoints; we retain a paired cut/no-cut audit in the Supplementary only to quantify its effect, while all primary results use the no-cut protocol.

Baselines and controls. The primary baseline is the checkpoint linked by the official Renz et al. [19] repository, evaluated with the same strict no-cut protocol used throughout our experiments. Because the released files do not reconstruct the paper’s multi-seed aggregate, we report it as one public checkpoint rather than as the exact published mean. Supplementary Sec. A.5 separately reports the published aggregates and the cut protocol compatibility audit.

On the BSL Corpus, we additionally evaluate a threeseed full-resolution reproduction of Hands-On [13] with normalized HaMeR [16] features at the native frame rate of 25 fps, without label cleanup or endpoint cutting. Seed 42 uses the recovered checkpoint; seeds 43 and 44 follow the same recipe with fixed epoch-40 checkpoints. Their predicted segments are passed to the same frozen recognizer used for the other downstream comparisons. Because Hands-On uses HaMeR features while the controlled comparisons use I3D, it is treated as a feature-heterogeneous reference rather than a controlled I3D comparison.

Three additional architecture families use binary inside/boundary supervision: a local-attention Transformer with temporal refinement (2.51M), a BiLSTM-CRF (1.05M), and MS-TCN++ (0.89M) [11]. They share I3D inputs, train/validation indices, seeds, 50 epochs, AdamW training setup, balanced score checkpoint selection, endpoint policy, metrics, and downstream recognizer. Model depth and parameter count vary by architecture; the local Transformer is an ASFormer-inspired control, not an exact reproduction of its published configuration [25].

Table 2. Strict no-cut segmentation results. BEACON and Hands-On [13]<sup>†</sup> are three-seed mean ± SD; Renz et al. [19] is the public checkpoint. Hands-On<sup>†</sup> uses full-resolution HaMeR features; all other rows use I3D.
<table><tr><td>Corpus</td><td>Segmenter</td><td>mF1B↑</td><td>mF1S↑</td></tr><tr><td rowspan="2">PHX [7]</td><td>Renz et al. [19]</td><td>69.76</td><td>51.12</td></tr><tr><td>BEACON</td><td> $\mathbf { 7 1 . 9 3 { \pm } 0 . 1 9 }$ </td><td> ${ \bf 6 5 . 7 0 { \scriptstyle \pm 0 . 1 5 } }$ </td></tr><tr><td rowspan="2"></td><td>Renz et al. [19]</td><td>67.28</td><td>43.83</td></tr><tr><td>BSL [21] Hands-On [13]† 62.39±0.06 BEACON</td><td> $\mathbf { 6 8 . 5 0 { \scriptstyle \pm 0 . 6 4 } }$ </td><td> $4 7 . 0 6 { \pm } 0 . 4 1$   ${ \bf 5 8 . 4 2 \pm 1 . 0 3 }$ </td></tr></table>

Downstream probe. Each predicted segment sequence is passed to one frozen recognizer per corpus: the released PHOENIX-14 classifier [19] and a 979-class BSL classifier shared across all BSL experiments. The classifier receives the frames selected by a segmenter, while its weights, vocabulary, and decoding remain unchanged. We report corpus ${ \mathrm { W E R } } = ( S + D + I ) / N$ and its substitution, deletion, and insertion components, with fixed denominators N: 10,987 (PHOENIX-14) and 4,214 (BSL Corpus). Test WER and recognition accuracy are never used for checkpoint or decoder selection.

Statistics. New models report three-seed mean and sample standard deviation. The BIO/BIOE comparison is paired by seed and emphasizes paired differences because three pairs cannot support a high-powered significance test. For the five family metric analysis, the unit is one architecturefamily mean; with $n = 5 ,$ we use exact two-sided permutation p-values for Spearman ρ and report the small-sample limitation explicitly.

## 5. Results

## 5.1. No-Cut Segmentation and Recognition

Under the same strict no-cut endpoint protocol, BEA-CON improves over the public Renz et al. [19] checkpoint by $+ 2 . 1 7 / + 1 4 . 5 8$ mF1B/mF1S on PHOENIX-14 and +1.22/+14.59 on the BSL Corpus (see Tab. 2). On the BSL Corpus, it also exceeds our three-seed full-resolution reproduction of Hands-On [13] by +6.11/+11.36 mF1B/mF1S; because Hands-On uses HaMeR while the other rows use I3D, this is a feature-heterogeneous reference rather than a controlled I3D comparison.

In downstream recognition, BEACON reduces WER relative to the public Renz et al. [19] checkpoint by 9.10 points (18.3% relative) on PHOENIX-14 and 4.15 points (5.36% relative) on the BSL Corpus. Under the same frozen recognizer, the Hands-On [13] reproduction yields 77.52 ± 0.41 WER, whereas BEACON achieves $7 3 . 3 3 \pm 0 . 3 1$ , a 4.19-point (5.40% relative) reduction in mean WER (see Tab. 3).

![](images/34aea9c9f16eca9c9412b8f03c72c457a36cce778fee03b7ae215dd472540917.jpg)

![](images/c8990a5d1569c25f3871806adf8fd9ce9a66305df5735f47d9435a9133c80498.jpg)  
Figure 4. PHOENIX-14 [7] example (no-cut; seed 42). Enlarged B/E frames anchor each clip to the frame-aligned GT and predictions. Numbers report segment IoU with the corresponding GT sign.

Across three seeds, on PHOENIX-14 mF1S/WER spans 65.54–65.83/40.41–40.79, and on the BSL Corpus spans 57.39–59.45/73.02–73.64. On PHOENIX-14, insertion/substitution errors fall by 7.48/4.97 points relative to Renz et al. [19] while deletion rises by 3.35, consistent with fewer fragmented predictions and occasional missed short signs. On the BSL Corpus, insertion/substitution fall by 2.28/2.42 points while deletion rises by 0.55.

Figure 4 gives a frame-aligned view of two neighboring ground-truth signs. Renz et al. [19] divide each sign into two unmatched fragments (IoUs 0.45/0.18 and 0.23/0.45), whereas BEACON produces one matched segment per sign (IoUs 0.83/0.71).

Endpoint cutting has little effect on BEACON: relative to no-cut, the paired cut evaluation changes WER by $- 0 . 0 5 / - 0 . 1 6$ and mF1S by $+ 0 . 0 8 / + 0 . 2 4$ on the PHOENIX-14/BSL Corpus. This indicates that the primary results do not depend on ground-truth endpoint cleanup. For the public Renz et al. [19] checkpoint, cutting increases mF1S by $+ 0 . 5 4 / + 1 . 1 7 $ , showing that the cut and no-cut protocols are not equivalent for this baseline. Per-seed results and the full paired compatibility audit are provided in the Supplementary.

Table 3. Downstream recognition under a matched recognizer. WER and substitution/deletion/insertion rates are corpus-level percentages; lower is better. BEACON and $\mathrm { H a n d s  – O n ^ { \dagger } }$ use three-seed $\mathrm { m e a n } \pm \mathrm { S D }$ for WER and three-seed mean for S/D/I. Hands-On<sup>†</sup> uses HaMeR features; dashes denote an unavailable PHOENIX-14 model. None uses test labels for model/decoder selection or endpoint cutting.
<table><tr><td></td><td colspan="4">PHX [7]</td><td colspan="4">BSL [21]</td></tr><tr><td>Segmenter</td><td>WER↓</td><td>S</td><td>D</td><td>I</td><td>WER↓</td><td>S</td><td>D</td><td>I</td></tr><tr><td>Renz et al. [19]</td><td>49.73</td><td>32.03</td><td>5.55</td><td>12.15</td><td>77.48</td><td>57.85</td><td>12.13</td><td>7.50</td></tr><tr><td>Hands  $. \mathrm { O n } [ 1 3 ] ^ { \dagger }$ </td><td></td><td></td><td></td><td></td><td> $7 7 . 5 2 { \pm } 0 . 4 1 $ </td><td>55.55</td><td>16.55</td><td>5.42</td></tr><tr><td>BEACON</td><td> $\mathbf { 4 0 . 6 3 } \pm \mathbf { 0 . 2 0 }$ </td><td>27.06</td><td>8.90</td><td>4.67</td><td> $\mathbf { 7 3 . 3 3 \pm 0 . 3 1 }$ </td><td>55.43</td><td>12.68</td><td>5.22</td></tr></table>

Table 4. Architecture families comparison under strict no-cut evaluation (three-seed mean $\pm \thinspace \mathrm { S D }$ except Renz et al. [19]’s MS-TCN). Local Transformer is an architecture control, not an exact AS-Former [25] reproduction.
<table><tr><td colspan="4">PHX [7]</td></tr><tr><td>Family</td><td>mF1B↑</td><td>mF1S↑</td><td>WER↓</td></tr><tr><td>MS-TCN</td><td>69.76</td><td>51.12</td><td>49.73</td></tr><tr><td>Local Transformer</td><td> $4 8 . 2 9 { \pm } 0 . 5 4 $ </td><td> $3 7 . 0 3 { \pm } 0 . 0 7$ </td><td> $5 0 . 9 0 { \scriptstyle \pm 0 . 9 2 }$ </td></tr><tr><td>BiLSTM-CRF</td><td> $6 7 . 8 0 { \pm } 0 . 1 0 $ </td><td> $5 8 . 0 8 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $4 7 . 1 5 { \pm } 0 . 7 5 $ </td></tr><tr><td>MS-TCN++</td><td> $7 0 . 6 0 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $6 0 . 1 0 { \pm } 1 . 7 1 $ </td><td> $4 2 . 5 4 { \pm } 0 . 2 9$ </td></tr><tr><td>BEACON</td><td> $\mathbf { 7 1 . 9 3 { \pm } 0 . 1 9 }$ </td><td> ${ \bf 6 5 . 7 0 { \scriptstyle \pm 0 . 1 5 } }$ </td><td> $\mathbf { 4 0 . 6 3 } \pm \mathbf { 0 . 2 0 }$ </td></tr></table>

<table><tr><td colspan="4">BSL [21]</td></tr><tr><td>Family</td><td>mF1B↑</td><td>mF1S↑</td><td>WER↓</td></tr><tr><td>MS-TCN</td><td>67.28</td><td> $4 3 . 8 3 $ </td><td>77.48</td></tr><tr><td>Local Transformer</td><td> $4 9 . 9 7 { \pm } 1 . 1 6 $ </td><td> $3 6 . 3 6 { \pm } 0 . 8 4$ </td><td> $7 9 . 4 1 { \pm } 0 . 4 9$ </td></tr><tr><td>BiLSTM-CRF</td><td> $6 2 . 5 9 { \pm } 0 . 2 6 $ </td><td> $5 0 . 9 4 { \pm } 0 . 1 5 $ </td><td> $7 6 . 1 5 { \pm } 0 . 1 3$ </td></tr><tr><td>MS-TCN++</td><td> $6 7 . 6 8 { \pm } 1 . 3 3 $ </td><td> $5 5 . 7 8 { \pm } 0 . 9 3 $ </td><td> ${ \bf 7 3 . 1 2 \pm 0 . 6 3 }$ </td></tr><tr><td>BEACON</td><td> $\mathbf { 6 8 . 5 0 { \scriptstyle \pm 0 . 6 4 } }$ </td><td> ${ \pm } 8 . 4 2 { \pm } 1 . 0 3 $ </td><td> $7 3 . 3 3 { \pm } 0 . 3 1 $ </td></tr></table>

## 5.2. Architecture Families and Metric Associations

The association analysis (see Tab. 4) uses one representative point from each of five distinct architecture families (see Supplementary Tab. 15). All newly trained controls share features, split, training budget, selection rule, no-cut evaluation, and frozen recognizer.

Across these architectures, mF1S has a stronger negative Spearman association with WER: $\rho = - 1 . 0$ versus −0.9 on PHOENIX-14 and −0.9 versus −0.8 on BSL. The exact test reaches $p < 0 . 0 5$ only for PHOENIX-14 mF1S; hence, the evidence supports only that mF1S orders WER more consistently among the tested families. Leave-one-familyout mF1S results are $\rho = - 1 . 0$ throughout on PHOENIX-14 and $[ - 1 . 0 0 , - 0 . 8 0 ]$ on BSL (see Supplementary). On BSL, MS-TCN++ has 0.21 lower mean WER than BEA-CON despite 2.64 lower mF1S, within either method’s reported seed variation.

## 5.3. Controlled Analyses

Matched BIO/BIOE. To isolate labels and decoding, we hold backbone, optimizer, split, loss, seeds, and 50-epoch budget fixed and cross BIO/BIOE labeling with argmax or the corresponding grammar-constrained decoder (see Tab. 5). Under argmax, changing BIO to BIOE changes mF1S by $- 0 . 0 1 / \mathrm { ~ + ~ } 0 . 5 8$ on PHOENIX-14/BSL; with each representation’s matching grammar, the changes are $- 0 . 0 4 / + 0 . 9 2$ , and WER changes by $- 0 . 5 3 / 0 . 0 0$ . The isolated effect is modest and dataset-dependent. BIOE nevertheless represents a complete segment directly as $\mathrm { ~ a ~ B ~ I \star E ~ }$ run. The full BEACON improvement therefore reflects the combined system rather than the label alphabet alone.

Table 5. Matched label/readout comparison, strict no-cut (threeseed mean). Other factors are fixed; each constrained row uses the matching grammar. WER is reported for constrained readout.
<table><tr><td>Corpus</td><td>Labels/Readout mF1B↑</td><td></td><td>mF1S↑</td><td>∆mF1S WER↓</td><td></td></tr><tr><td rowspan="3">PHX [7]</td><td>BIO/argmax BIO/grammar</td><td>70.78</td><td>62.26 62.80</td><td></td><td>43.82</td></tr><tr><td>BIOE/argmax</td><td>70.68 70.56</td><td>62.25</td><td></td><td></td></tr><tr><td>BIOE/8-edge</td><td>70.38</td><td>62.76</td><td>-0.04</td><td>43.29</td></tr><tr><td rowspan="4">BSL [21]</td><td>BIO/argmax</td><td>64.79</td><td>54.85</td><td></td><td></td></tr><tr><td>BIO/grammar</td><td>63.98</td><td>54.54</td><td></td><td>73.95</td></tr><tr><td>BIOE/argmax</td><td>66.27</td><td>55.43</td><td></td><td></td></tr><tr><td>BIOE/8-edge</td><td>65.48</td><td>55.45</td><td>+0.92</td><td>73.95</td></tr></table>

Relative to argmax, matching-grammar decoding changes mF1S by $+ 0 . 5 1 / \mathrm { ~ + ~ } 0 . 0 2$ for BIOE and $+ 0 . 5 4 / - 0 . 3 1$ for BIO. Its numerical effect is also modest and dataset-dependent; structurally, however, argmax can leave the declared grammar whereas constrained decoding cannot.

Relative to BIO/argmax in Tab. 5, the full model gains $+ 1 . 1 5 / \ : + \ : 3 . 7 1$ mF1B and $+ 3 . 4 4 / \mathrm { ~ + ~ } 3 . 5 7$ mF1S on PHOENIX-14/BSL. Tab. 6 separates endpoint modeling from distillation. The supervised IO branch and refinement add +1.17/ + 2.62 mF1B and +2.46/ + 2.72 mF1S; KD then adds +0.38/ + 0.40 and +0.48/ + 0.25, respectively. The total gains over the structured base are $+ 1 . 5 5 / + 3 . 0 2$ mF1B and $+ 2 . 9 4 / + 2 . 9 7$ mF1S; the table also reports the corresponding WER.

Fixed grammar and learned transitions. A samebackbone unrestricted CRF is numerically competitive (see Supplementary Tab. 14). It is close to fixed decoding on PHOENIX-14 and improves the matched nondistilled BIOE control on the BSL Corpus. However, 12.65%/12.63% of its decoded test sequences contain at least one transition outside the fixed grammar, whereas constrained decoding guarantees zero grammar-invalid sequences. Learned transitions are viable and can be numerically competitive; the fixed graph instead provides deterministic validity under the declared BIOE grammar.

Table 6. Component comparison, strict no-cut (three-seed mean). Each entry reports PHOENIX-14/BSL. The last two rows have the same shared-student architecture and endpoint readout, isolating KD.
<table><tr><td>System</td><td>mF1B↑</td><td>mF1S↑</td><td>WER↓</td></tr><tr><td>BIOE/8-edge base</td><td>70.38/65.48</td><td>62.76/55.45</td><td>43.29/73.95</td></tr><tr><td>IO branch, no KD</td><td>71.55/68.10</td><td>65.22/58.17</td><td>40.72/73.74</td></tr><tr><td>Full BEACON</td><td>71.93/68.50</td><td>65.70/58.42</td><td>40.63/73.33</td></tr></table>

Inference cost. Supplementary Tab. 16 compares a twoencoder implementation with the shared student. At the p50/p90 sequence lengths of both corpora, the student reduces neural-forward latency by 36.5%–36.9% and peak allocation by roughly 29%–33%. The teacher is never loaded at inference. Absolute timings are hardware-dependent, making the relative reductions more informative.

## 6. Discussion

## 6.1. Segment Quality and Downstream Recognition

BEACON improves boundary and segment metrics under strict no-cut evaluation, with the larger gain at the segment level, and the same predictions lower WER under the frozen recognizers. The paired audit confirms that this pattern does not depend on ground-truth endpoint cleanup.

On PHOENIX-14, fewer insertions are consistent with fewer fragmented or spurious segments, while more deletions indicate occasional missed signs. On BSL, gains span substitutions and insertions. Together with Sec. 5.2, this motivates reporting mF1B, mF1S, and downstream WER jointly.

## 6.2. Structure and Boundary Synergy

Sec. 5.3 shows modest standalone numerical effects for BIOE and constrained decoding. BIOE directly represents completion as a B I<sub>\*</sub> E run; constrained decoding enforces the declared grammar. Learned CRF transitions can be competitive, whereas the fixed graph guarantees validity under that grammar.

Structural validity does not determine exact endpoints. The supervised IO branch and refinement provide most of the gain beyond the structured base; distillation adds a smaller, consistent gain without changing inference. BEA-CON thereby combines valid structure and precise endpoint localization in one model.

## 6.3. Limitations

Validation and generalization. Neither dataset defines an official segmentation validation split, so we construct a fixed video-disjoint partition from its training pool. It is not recording-group-disjoint; generalization to unseen recording groups remains untested.

Scope of the metric–WER evidence. The five-family analysis uses one frozen recognizer per corpus. This small sample is corpus- and recognizer-specific and cannot establish universal recognition optimality.

Dense supervision and training complexity. BEACON requires frame-level annotations and does not address corpora with only sentence- or gloss-level supervision. Inference uses one student, but training also requires a binary teacher and validation-selected refinement parameters.

Annotation scope. Frame endpoints are operational annotations, not unique linguistic boundaries: co-articulation and transitions may be ambiguous, and the corpora provide no inter-annotator analysis.

## 7. Conclusion

We presented BEACON, a single-encoder segmenter that combines four-class BIOE labeling, eight-transition constrained decoding, and distilled boundary knowledge. The design separates two complementary requirements: the fixed grammar restricts predictions to valid BIOE paths, while the shared IO branch supplies local evidence for endpoint refinement. Its binary teacher is used only during training, so inference retains one 5M-parameter student rather than a separate boundary encoder.

Under strict no-cut evaluation with validation-based selection, BEACON improves mF1S by 14.58/14.59 points and reduces downstream WER by 9.10/4.15 points relative to the state-of-the-art on PHOENIX-14/BSL. The matched ablations show that the full gain is a system-level result: the shared endpoint branch provides the largest improvement beyond the structured base, and distillation adds a smaller consistent gain without increasing inference cost. Across five architecture families, mF1S orders WER more consistently than mF1B. Although this evidence is limited to the tested corpora and recognizers, it supports reporting segment quality, boundary accuracy, and downstream WER together rather than selecting models by boundary precision alone.

Beyond sign segmentation, fixed-grammar decoding and training-only boundary distillation may be useful when temporal outputs must satisfy a known structure but local endpoints remain uncertain. Future work should model annotation ambiguity and transition regions, reduce dependence on dense frame-level supervision, broaden matched evaluation across datasets, recognizers, and vocabularies, and train segmenters more directly for downstream utility.

## References

[1] Yazan Abu Farha and Juergen Gall. MS-TCN: Multi-stage temporal convolutional network for action segmentation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3575– 3584, 2019.

[2] Valentin Belissen, Annelies Braffort, and Michele Gouiff\` es. Dicta-\` Sign-LSF-v2: Remake of a continuous French Sign Language dialogue corpus and a first baseline for automatic sign language processing. In Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC), pages 6040–6048. European Language Resources Association, 2020.

[3] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? A new model and the Kinetics dataset. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 6299– 6308, 2017.

[4] Summy Farfan, Juan J. Choquehuanca-Zevallos, Ana Aguilera, Irvin Dongo, and Yudith Cardinale. Temporal video segmentation approach for peruvian sign language. Sensors, 25(17):5217, 2025.

[5] Alex Graves, Santiago Fernandez, Faustino Gomez, and J´ urgen¨ Schmidhuber. Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. In International Conference on Machine Learning (ICML), 2006.

[6] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network, 2015.

[7] Oscar Koller, Jens Forster, and Hermann Ney. Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. Computer Vision and Image Understanding, 141:108–125, 2015.

[8] John D. Lafferty, Andrew McCallum, and Fernando C. N. Pereira. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In International Conference on Machine Learning (ICML), pages 282–289, 2001.

[9] Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, and Chris Dyer. Neural architectures for named entity recognition. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages 260– 270, 2016.

[10] Colin Lea, Michael D. Flynn, Rene Vidal, Austin Reiter, and Gregory D. Hager. Temporal convolutional networks for action segmentation and detection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1003–1012, 2017.

[11] Shijie Li, Yazan Abu Farha, Yun Liu, Ming-Ming Cheng, and Juergen Gall. MS-TCN++: Multi-stage temporal convolutional network for action segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(6):6647–6658, 2023.

[12] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.

[13] Jian He Low, Harry Walsh, Ozge Mercanoglu Sincan, and Richard Bowden. Hands-On: Segmenting individual signs from continuous

sequences. In IEEE International Conference on Automatic Face and Gesture Recognition (FG), pages 1–5, 2025.

[14] Amit Moryossef, Zifan Jiang, Mathias Muller, Sarah Ebling, and ¨ Yoav Goldberg. Linguistically motivated sign language segmenta tion. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12703–12724. Association for Computational Linguistics, 2023.

[15] Carol Neidle and Dimitris Metaxas. American Sign Language Linguistic Research Project (ASLLRP) Sign Bank: ASLLRP Continuous Signing Corpora — version 3, 2025. Videos provided by Dawn SignPress (www.dawnsign.com).

[16] Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, and Jitendra Malik. Reconstructing hands in 3D with transformers. In IEEE/CVF Conference on Computer Vi sion and Pattern Recognition (CVPR), 2024.

[17] Lance A. Ramshaw and Mitchell P. Marcus. Text chunking using transformation-based learning. In Third Workshop on Very Large Corpora, pages 82–94, 1995.

[18] Lev Ratinov and Dan Roth. Design challenges and misconceptions in named entity recognition. In Proceedings of the Thirteenth Conference on Computational Natural Language Learning (CoNLL), pages 147–155, 2009.

[19] Katrin Renz, Nicolaj C. Stache, Samuel Albanie, and Gul Varol.¨ Sign language segmentation with temporal convolutional networks. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2135–2139, 2021.

[20] Katrin Renz, Nicolaj C. Stache, Neil Fox, Gul Varol, and Samuel¨ Albanie. Sign segmentation with changepoint-modulated pseudolabelling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3403–3412, 2021.

[21] Adam Schembri, Jordan Fenlon, Ramas Rentelis, Sally Reynolds, and Kearsy Cormier. Building the British Sign Language corpus. Language Documentation & Conservation, 7:136–154, 2013.

[22] HoHyun Sim, Hyeonjoong Cho, and Hankyu Lee. Changepoint detection-assisted nonparametric clustering for unsupervised temporal sign segmentation. Engineering Applications of Artificial Intelli gence, 127:107323, 2024.

[23] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), pages 5998–6008, 2017.

[24] Andrew J. Viterbi. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Transactions on Information Theory, 13(2):260–269, 1967.

[25] Fangqiu Yi, Hongyu Wen, and Tingting Jiang. ASFormer: Transformer for action segmentation. In British Machine Vision Conference (BMVC), 2021.

[26] Mingyu Zhao, Zhanfu Yang, Yang Zhou, Zhaoyang Xia, Can Jin, Xiaoxiao He, Carol Neidle, and Dimitris N. Metaxas. MHB: Multi modal handshape-aware boundary detection for continuous sign lan guage recognition. arXiv preprint arXiv:2511.19907, 2025.

## A. Additional Experimental Details

## A.1. Canonical No-Cut Results by Seed

Tab. 7 gives every run behind the primary results. For each dataset, the binary teacher and distilled student use the same fixed 90/10 split of the original training set (split seed 100). The teacher and student checkpoints and IO refinement parameters are selected on validation segmentation metrics; the B/E fusion gate is fixed across runs. Seed-43 on PHOENIX-14 selects $\alpha _ { \mathrm { I O } } = 0 ;$ , yet still retains the IO/KD training losses. This suggests that distillation can regularize the shared encoder even when validation does not select IO fusion at inference.

Table 7. Distilled BEACON per-seed segmentation results and validation-selected IO refinement settings (α<sub>IO</sub>, r, τ<sub>IO</sub>), all under strict no-cut evaluation.
<table><tr><td>Corpus/Seed</td><td>1 Epoch</td><td>α</td><td>r</td><td>τ</td><td>mF1B↑</td><td>mF1S↑</td></tr><tr><td>PHX/42</td><td>12</td><td>.7</td><td>2</td><td>.5</td><td>71.80</td><td>65.54</td></tr><tr><td>PHX/43</td><td>6</td><td>.0</td><td>1</td><td>.3</td><td>71.85</td><td>65.83</td></tr><tr><td>PHX/44</td><td>10</td><td>.7</td><td>3</td><td>.3</td><td>72.15</td><td>65.74</td></tr><tr><td>mean±SD</td><td></td><td></td><td></td><td></td><td>71.93±.19</td><td>65.70±.15</td></tr><tr><td>BSL/42</td><td>10</td><td>.7</td><td>3</td><td>.5</td><td>67.92</td><td>57.39</td></tr><tr><td>BSL/43</td><td>6</td><td>.7</td><td>1</td><td>.3</td><td>69.18</td><td>58.42</td></tr><tr><td>BSL/44</td><td>3</td><td>.7</td><td>3</td><td>.5</td><td>68.41</td><td>59.45</td></tr><tr><td>mean±SD</td><td></td><td></td><td></td><td></td><td>68.50±.64</td><td>58.42±1.03</td></tr></table>

Table 8. Hands- ${ \boldsymbol { \cdot } } { \boldsymbol { \mathrm { O n } } } ^ { \dagger }$ reproduction by seed using full-resolution HaMeR features under strict no-cut evaluation. Seed-42 uses the recovered checkpoint; seeds-43 and 44 use fixed epoch-40 checkpoints from the same recipe.
<table><tr><td>Seed</td><td>mF1B↑</td><td>mF1S↑</td></tr><tr><td>42</td><td>62.45</td><td>47.44</td></tr><tr><td>43</td><td>62.38</td><td>47.11</td></tr><tr><td>44</td><td>62.33</td><td>46.63</td></tr><tr><td>mean±SD</td><td>62.39±.06</td><td>47.06±.41</td></tr></table>

## A.2. Per-Run Downstream Recognition

Tab. 9 separates the downstream error decomposition from the segmentation and decoder selection results above. Hands-On<sup>†</sup> uses seeds 42–44 at full 25 fps resolution with checkpoint-matched normalized HaMeR features, raw BIO argmax, no label cleanup, and no endpoint cutting. Seed-42 is the recovered checkpoint; seeds 43 and 44 follow the same recipe and use fixed epoch-40 checkpoints. Their segments index the same I3D stream consumed by the frozen recognizer. All BSL rows use $N = 4 { , } 2 1 4$ and all PHX rows use $N = 1 0 { , } 9 8 7$

Table 9. Per-run downstream decomposition under the same frozen recognizer per corpus. Rates are corpus-level percentages. Renz et al. is one public checkpoint; Hands-On<sup>†</sup> and BEACON report all three seeds, with mean ± SD for WER and mean S/D/I.
<table><tr><td>Corpus</td><td>Segmenter/Run</td><td>WER</td><td>S</td><td>D</td><td>I</td></tr><tr><td rowspan="5">PHX</td><td>Renz et al.</td><td>49.73</td><td>32.03</td><td>5.55</td><td>12.15</td></tr><tr><td>BEACON/42</td><td>40.79</td><td>27.21</td><td>8.87</td><td>4.71</td></tr><tr><td>BEACON/43</td><td>40.41</td><td>26.60</td><td>10.13</td><td>3.68</td></tr><tr><td>BEACON/44</td><td>40.69</td><td>27.37</td><td>7.71</td><td>5.62</td></tr><tr><td>BEACON mean±SD</td><td>40.63±.20</td><td>27.06</td><td>8.90</td><td>4.67</td></tr><tr><td rowspan="8">BSL</td><td>Renz et al.</td><td>77.48</td><td>57.85</td><td>12.13</td><td>7.50</td></tr><tr><td>Hands-On†/42</td><td>77.98</td><td>55.93</td><td>15.95</td><td>6.10</td></tr><tr><td>Hands-On†/43</td><td>77.41</td><td>55.67</td><td>16.94</td><td>4.79</td></tr><tr><td>Hands-On†/44</td><td>77.17</td><td>55.05</td><td>16.75</td><td>5.36</td></tr><tr><td>Hands-On† mean±SD</td><td>77.52±.41</td><td>55.55</td><td>16.55</td><td>5.42</td></tr><tr><td>BEACON/42</td><td>73.33</td><td>55.08</td><td>13.50</td><td>4.75</td></tr><tr><td>BEACON/43</td><td>73.64</td><td>55.65</td><td>11.96</td><td>6.03</td></tr><tr><td>BEACON/44</td><td>73.02</td><td>55.55</td><td>12.58</td><td>4.89</td></tr><tr><td></td><td>BEACON mean±SD</td><td>73.33±.31</td><td>55.43</td><td>12.68</td><td>5.22</td></tr></table>

## A.3. Selection and Test-Use Audit

Tab. 10 makes the selection boundary explicit. No test segmentation metric or downstream WER selects a checkpoint or decoder setting.

Table 10. Training, selection, and test-use separation for the reported systems.
<table><tr><td>Item</td><td>Chosen by</td><td>Test Set Role</td></tr><tr><td>Teacher checkpoint</td><td>validation mF1B</td><td>final reporting only</td></tr><tr><td>Student checkpoint</td><td>validation balanced score</td><td>final reporting only</td></tr><tr><td>IO refinement (α, r, τ)</td><td>validation grid</td><td>final reporting only</td></tr><tr><td>B/E fusion and decoder biases</td><td>fixed across primary runs</td><td>final reporting only</td></tr><tr><td>Architecture controls</td><td>validation balanced score</td><td>final reporting only</td></tr><tr><td>Renz et al. checkpoint released artifact</td><td></td><td>evaluation only</td></tr><tr><td></td><td>Hands-On checkpoints seed 42 recovered; seeds 43/44 fixed epoch 40</td><td>evaluation only</td></tr></table>

## A.4. Distillation Component Diagnostic

The non-distilled and distilled students use the same 5M-parameter architecture and strict no-cut endpoint policy. Across three matched seeds, distillation changes mean mF1B/mF1S/WER from 71.55/65.22/40.72 to 71.93/65.70/40.63 on PHOENIX-14 and from 68.10/58.17/73.74 to 68.50/58.42/73.33 on the BSL Corpus. Tab. 11 reports every paired run.

Table 11. Matched distillation ablation under strict no-cut evalu ation. The non-distilled and distilled rows use the same sharedstudent architecture and readout.
<table><tr><td>Corpus</td><td>Training</td><td>Seed</td><td>mF1B↑</td><td>mF1S↑</td><td>WER↓</td></tr><tr><td rowspan="6">PHX</td><td>No KD</td><td>42</td><td>71.74</td><td>65.26</td><td>40.57</td></tr><tr><td>No KD</td><td>43</td><td>71.25</td><td>65.05</td><td>40.68</td></tr><tr><td>No KD</td><td>44</td><td>71.65</td><td>65.34</td><td>40.92</td></tr><tr><td>KD</td><td>42</td><td>71.80</td><td>65.54</td><td>40.79</td></tr><tr><td>KD</td><td>43</td><td>71.85</td><td>65.83</td><td>40.41</td></tr><tr><td>KD</td><td>44</td><td>72.15</td><td>65.74</td><td>40.69</td></tr><tr><td rowspan="6">BSL</td><td>No KD</td><td>42</td><td>67.97</td><td>57.88</td><td>73.80</td></tr><tr><td>No KD</td><td>43</td><td>68.35</td><td>58.37</td><td>74.42</td></tr><tr><td>No KD</td><td>44</td><td>67.99</td><td>58.26</td><td>72.99</td></tr><tr><td>KD</td><td>42</td><td>67.92</td><td>57.39</td><td>73.33</td></tr><tr><td>KD</td><td>43</td><td>69.18</td><td>58.42</td><td>73.64</td></tr><tr><td>KD</td><td>44</td><td>68.41</td><td>59.45</td><td>73.02</td></tr></table>

## A.5. Endpoint-Cut Compatibility Check

The endpoint-cutting post-processing step in Renz et al. reads the ground-truth start/end extent of each evaluation video. It is therefore inappropriate for deployable headline inference, but paired evaluation quantifies its effect (see Tab. 12). The official paper reports multi-run cut aggregates of 71.50/52.78 on PHOENIX-14 and 68.68/47.71 on the BSL Corpus, but only one linked checkpoint per corpus was available to us. We therefore do not present our paired values as exact no-cut conversions of the published means.

Table 12. Paired no-cut/cut results from identical predictions. BEACON is a three-seed mean; Renz et al. is one checkpoint.
<table><tr><td>Corpus</td><td>Segmenter/Protocol</td><td>mF1B↑</td><td>mF1S↑</td><td>WER↓</td></tr><tr><td rowspan="4">PHX</td><td>Renz et al./no-cut</td><td>69.76</td><td>51.12</td><td>49.73</td></tr><tr><td>Renz et al./cut</td><td>70.28</td><td>51.66</td><td></td></tr><tr><td>BEACON/no-cut</td><td>71.93</td><td>65.70</td><td>40.63</td></tr><tr><td>BEACON/cut</td><td>72.08</td><td>65.78</td><td>40.58</td></tr><tr><td rowspan="4">BSL</td><td>Renz et al./no-cut</td><td>67.28</td><td>43.83</td><td>77.48</td></tr><tr><td>Renz et al./cut</td><td>68.32</td><td>45.00</td><td></td></tr><tr><td>BEACON/no-cut</td><td>68.50</td><td>58.42</td><td>73.33</td></tr><tr><td>BEACON/cut</td><td>68.87</td><td>58.66</td><td>73.17</td></tr></table>

## A.6. Fair Labeling and Decoder Factorial

The factorial comparison fixes the 1024-D input, 5-block TCN plus 4-layer Transformer, optimizer, losses, 50 epochs, split, paired seeds, selection rule, and no-cut evaluator. BIO and BIOE differ by 257 output-head parameters (0.0058%); no IO head, teacher, distillation, Begin/End auxiliary loss, or relocation is used. Each constrained row uses the grammar matching its label representation: the BIO grammar disallows an I start or O→I, while BIOE uses the declared eight-edge grammar. Paired BIOE-minus-BIO differences under these matching grammars are −0.30 ± $0 . 3 9 / - 0 . 0 4 \pm 0 . 6 2 / - 0 . 5 3 \pm 1 . 1 6$ (mF1B/mF1S/WER) on $\mathrm { P H O E N I X - 1 4 a n d + 1 . 5 0 \pm 0 . 4 6 / + 0 . 9 2 \pm 0 . 7 7 / 0 . 0 0 \pm 0 . 8 8 }$ on the BSL Corpus (see Tab. 13). With three paired seeds, exact sign-flip tests have little resolution; we therefore emphasize effect estimates and direction consistency rather than significance.

Table 13. Fair ablation, three-seed mean ± SD. WER is reported for grammar-constrained readout only.
<table><tr><td>Corpus</td><td>Labels</td><td>Readout</td><td>mF1B↑</td><td>mF1S↑</td><td>WER↓</td></tr><tr><td rowspan="4">PHX</td><td>BIO</td><td>argmax</td><td>70.78±.14</td><td>62.26±.19</td><td rowspan="4">43.82±.15</td></tr><tr><td>BIO</td><td>grammar</td><td>70.68±.16</td><td>62.80±.37</td></tr><tr><td>BIOE</td><td>argmax</td><td>70.56±.34</td><td>62.25±.41</td></tr><tr><td>BIOE</td><td>8-edge</td><td>70.38±.51</td><td>62.76±.27 43.29±1.01</td></tr><tr><td rowspan="4">BSL</td><td>BIO</td><td>argmax</td><td>64.79±.58</td><td>54.85±.58</td><td rowspan="4">73.95±.25</td></tr><tr><td>BIO</td><td>grammar</td><td>63.98±.55</td><td>54.54±.75</td></tr><tr><td>BIOE</td><td>argmax</td><td>66.27±.24</td><td>55.43±.29</td></tr><tr><td>BIOE</td><td>8-edge</td><td>65.48±.09</td><td>55.45±.03 73.95±.67</td></tr></table>

## A.7. Learned-Transition CRF

The learned transition control uses the same BIOE encoder and training split but an unrestricted linear-chain CRF. We count a decoded sequence as invalid when it contains any start, end, or adjacent-state transition outside the fixed eight-edge grammar. The CRF is competitive, but learned scores do not enforce the known grammar. Constrained decoding always returns a path in that grammar and adds no learned transition parameters.

Table 14. Learned transition CRF, strict no-cut, three-seed mean ± SD. Invalid is the mean percentage of test sequences containing at least one transition outside the fixed grammar.
<table><tr><td>Corpus</td><td>mF1B↑</td><td>mF1S↑</td><td>WER↓</td><td>Invalid (%)</td></tr><tr><td>PHX</td><td>70.13±1.29</td><td>62.78±2.09</td><td>43.77±2.75</td><td>12.65</td></tr><tr><td>BSL</td><td>66.40±1.06</td><td>56.53±1.45</td><td>73.43±1.15</td><td>12.63</td></tr></table>

## A.8. Architecture Families Metric–WER Analysis

Tab. 15 retains sample-SD variation for three-seed models and reports no artificial variance for the public Renz et al. checkpoint. Exact permutation tests are used because n = 5. The Spearman (ρ, p) pairs for mF1B/mF1S are $( - . 9 0 , . 0 8 3 ) / ( - 1 . 0 0 , . 0 1 7 )$ on PHOENIX-14 and $( - . 8 0 , . 1 3 3 ) / ( - . 9 0 , . 0 8 3 )$ on the BSL Corpus. Leave-onefamily-out ranges are $[ - 1 . 0 0 , - . 8 0 ] / [ - 1 . 0 0 , - 1 . 0 0 ]$ and $[ - . 8 0 , - . 6 0 ] / [ - 1 . 0 0 , - . 8 0 ]$ , respectively. These sensitivity results reinforce the ordering observation but not a universa claim.

Table 15. Architecture families comparison (three-seed mean ± SD except the single public Renz et al. checkpoint). Local Transformer is an architecture control, not an exact ASFormer reproduction. Exact permutation p-values accompany Spearman ρ.
<table><tr><td>Family</td><td>mF1B↑</td><td>mF1S↑</td><td>WER↓</td></tr><tr><td>PHX</td><td></td><td></td><td></td></tr><tr><td>MS-TCN</td><td>69.76</td><td>51.12</td><td>49.73</td></tr><tr><td>Local Transformer</td><td>48.29±.54</td><td>37.03±.07</td><td>50.90±.92</td></tr><tr><td>BiLSTM-CRF</td><td>67.80±.10</td><td>58.08±.26</td><td>47.15±.75</td></tr><tr><td>MS-TCN++</td><td>70.60±.67</td><td>60.10±1.71</td><td>42.54±.29</td></tr><tr><td>BEACON</td><td>71.93±.19</td><td>65.70±.15</td><td>40.63±.20</td></tr><tr><td>ρ with WER (p)</td><td>−.90 (.083)</td><td>-1.00 (.017)</td><td></td></tr><tr><td>BSL</td><td></td><td></td><td></td></tr><tr><td>MS-TCN</td><td>67.28</td><td>43.83</td><td>77.48</td></tr><tr><td>Local Transformer</td><td>49.97±1.16</td><td>36.36±.84</td><td>79.41±.49</td></tr><tr><td>BiLSTM-CRF</td><td>62.59±.26</td><td>50.94±.15</td><td>76.15±.13</td></tr><tr><td>MS-TCN++</td><td>67.68±1.33</td><td>55.78±.93</td><td>73.12±.63</td></tr><tr><td>BEACON</td><td>68.50±.64</td><td>58.42±1.03</td><td>73.33±.31</td></tr><tr><td>ρ with WER (p)</td><td>−.80 (.133)</td><td>-.90 (.083)</td><td></td></tr></table>

## A.9. Inference Efficiency

We measured the neural forward pass on an NVIDIA RTX 5090, FP32, batch size 1, at the p50/p90 test lengths of each corpus, with 20 warm-up and 100 CUDA-synchronized repetitions. CPU Viterbi and endpoint relocation are excluded from both systems. The shared student reduces parameters by 43.3%, resident allocation by 43.5%, peak allocation by approximately 29%–33%, and forward latency by 36.5%– 36.9%.

Table 16. Inference efficiency. Latency ranges cover both corpora and sequence lengths.
<table><tr><td>System</td><td>Params</td><td>Resident (MiB)</td><td>Peak (MiB)</td><td>Latency (ms)</td></tr><tr><td>Two encoders</td><td>8.82M</td><td>34.65</td><td>45.25-47.71</td><td>1.41-1.43</td></tr><tr><td>Single distilled</td><td>5.00M</td><td>19.59</td><td>30.19–32.65</td><td>0.89-0.90</td></tr></table>

## A.10. Reproducibility Notes

All 12 fair ablation runs, 18 architecture control runs, 6 teachers, 6 non-distilled students, 6 distilled students, and 6 learned CRF runs completed without NaN, traceback, or out-of-memory failure. The fixed split has zero video overlap and covers the original training set exactly. Teacher and student use identical indices; teachers are frozen and excluded from the inference checkpoint. Corpus WER denominators are constant across segmenters (N = 10,987 PHOENIX-14; N = 4,214 BSL Corpus). The split is not recording group-disjoint because most available recording groups occur on both sides of the original training pool; this is disclosed as a limitation rather than treated as a crossgroup validation result. Code and evaluation scripts are included in the Supplementary archive and will be released publicly upon acceptance.

## A.11. Data and Ethical Scope

The experiments use previously released sign language research corpora and derived features under their existing access conditions. We do not infer signer identity or demographic attributes. Automatic lexical segmentation can support annotation and retrieval, but errors may disproportionately affect signing styles underrepresented in the training corpora; the present two-corpus evaluation does not quantify that disparity. Predicted boundaries and recognition outputs should therefore not be treated as authoritative linguistic annotation without human review.