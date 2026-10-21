# Auditing Where Concept Bottleneck Accuracy Comes From on CUB

Sascha Saralajew\* NEC Labs Europe Heidelberg, Germany sascha.saralajew@neclab.eu

Eliana Pastor Politecnico di Torino Turin, Italy eliana.pastor@polito.it

Eleonora Poeta\* Politecnico di Torino Turin, Italy eleonora.poeta@polito.it

Ammar Shaker NEC Labs Europe Heidelberg, Germany ammar.shaker@neclab.eu

## Abstract

Concept Bottleneck Models (CBMs) promise predictions that humans can inspect and correct through concepts such as “curved beak". On CUB-200-2011, their standard visual benchmark, progress is nevertheless measured like that of ordinary classifiers: by higher task and concept accuracy. Neither number reveals whether the label is predicted from the exposed concept scores at all, or whether a predicted concept is supported by visual evidence a human would recognize as that concept, rather than by other image features that co-occur with it in the training data, such as cues to the species as a whole. We audit where the accuracy of CBMs and Concept Embedding Models (CEMs) on CUB comes from. We first show that the CUB concept annotations are too noisy to explain this accuracy: the class-level labels used for training agree with the per-image annotations on only ≈82% of image-concept pairs, and classifiers trained on the per-image annotations reach only ≈37% species accuracy. Leakage probes show that CEM predictions cannot be reconstructed from their exposed concept scores, with the missing class information stored in internal embeddings. We identify four design axes through which class information can enter concept scores—receptive field, shared versus concept-specific representations, norm-dependent scoring, and task supervision— and a systematic ablation shows that most of the class information in concept scores enters through non-local features: restricting a ResNet50 backbone to local BagNet patches lowers the accuracy recoverable from concept scores from 72% to 43%, close to what the ground-truth concept annotations themselves support, while joint task training raises it again at the expense of concept accuracy. Spatial localization and part occlusion show that locality makes grounding measurable but does not guarantee it: under class-level labels, local concept scores use the head as a proxy for body, wing, and tail concepts, and CEM's concept scores barely respond when their supporting part is occluded. Task and concept accuracy alone are therefore poor proxies for whether a CBM delivers the interpretability and intervenability it promises: they should support, not drive, the evaluation of architectural changes, and given the fuzziness of the CUB annotations, differences in CUB accuracy should not be overinterpreted.

## 1 Introduction

Concept Bottleneck Models (CBMs) [15; 23] are designed to make predictions through humanunderstandable concepts, such as “has red wing" or “long beak". Their appeal rests on two promises beyond predictive accuracy: interpretability – a decision can be inspected through the concepts that produced it – and intervenability – a decision can be corrected by correcting those concepts.

The CUB-200-2011 dataset [26] is the standard visual benchmark for this paradigm. As a result progress in CBMs is often summarized by higher classification accuracy on CUB, whether reported at the class (species) level or at the instance level – the same metric used to evaluate ordinary classifiers. This creates a tension: neither of these accuracy numbers confirms the interpretability promise above – that predictions are actually mediated by concepts a human can recognize in the image, e.g., actually seeing a curved beak, not merely a pattern that correlates with the label.

The interpretability promise can fail in two distinct ways. First, the label predictor could rely on information that the concepts do not express: either by bypassing the exposed concept scores – the model's concept interface – altogether (e.g., through the internal concept embedding in CEMs), or through structured variation encoded within the concept scores themselves, as soft concept scores are known to do [17; 10]. Second, even when the label predictor uses only the concept scores, those scores could be predicted from image features that co-occur with the concept in the training data but that a human would not perceive as evidence for it – e.g., predicting “curved beak" from cues to the species as a whole. Such a concept score is predictive, but not visually grounded in the sense a human inspecting the image would expect.

This second failure mode also undermines intervenability. The label predictor learns what a concept score means from the model's own scores: if the score for “curved beak" in fact tracks species cues elsewhere in the image, the label predictor learns to read it as species evidence. A human who corrects this score after inspecting the beak then feeds the label predictor a value whose meaning differs from the one it was trained on, so the effect of the correction on the prediction no longer reflects the human's judgment about the beak. We do not evaluate interventions directly; instead, we test their precondition, namely whether concept scores are driven by the evidence a human would inspect.

Both failure modes allow a model to be more accurate than its concepts can explain. As a first check, we therefore estimate how much task information the CUB concept annotations themselves provide, by training classifiers on ground-truth instance-level concept vectors. The resulting concept-only reference accuracy is only ≈37%. Because the annotations are noisy – a single Amazon Mechanical Turk annotator per image – this value marks a reference band rather than a strict upper bound on faithful concept reasoning – yet even allowing for this noise, it is far below the 70%+ accuracies reported by widely used CBM and CEM models, which suggests that concept annotations alone do not explain those numbers.

This motivates our central question: where does the extra accuracy come from, and is it consistent with the interpretability and intervenability promises introduced above? We address the two failure modes in turn. For the first, leakage probes test whether the native predictions of CBMs and CEMs can be reconstructed from their exposed concept scores alone. For the second, we ask how class information enters the concept scores themselves. We identify four design choices through which this can happen – the receptive field of the features, shared versus concept-specific representations, norm-dependent scoring, and task supervision of the concept scores – and sweep receptive field, head sharing, and task supervision systematically within one model template, measuring how much class information a probe recovers from the frozen concept scores of each configuration. The receptive field turns out to be the main lever, and the configuration in which all four choices are constrained carries roughly as much class information as the ground-truth concept annotations. Because this configuration computes each concept score from a local image patch, it also makes grounding measurable: spatial localization tests whether its concept scores peak on the image regions a human would use to recognize the concept. Part occlusion, which applies to any model, tests whether concept scores respond when that evidence is removed; we apply it to the constrained configuration and to CEM, the model with the largest bypass gap.

Contributions. Our contributions are fourfold: (i) We show that the CUB concept annotations alone cannot account for the accuracies reported by CBM/CEM models: classifiers trained directly on ground-truth concept vectors reach only ≈37% task accuracy – far below the 70%+ typically reported – and this gap partly reflects a structural mismatch between the class-level concept labels used for training (identical for every image of a species) and the instance-level concept evidence actually visible in each image; (ii) We identify four design axes – receptive field, shared concept representations, norm-dependent scoring, and task supervision – through which CBMs and CEMs can place class information in their concept scores that is not backed by visual evidence for the annotated concepts; (iii) We audit where CBM and CEM accuracy on CUB comes from: leakage probes show that CEM predictions partly bypass the exposed concept scores, and a systematic ablation over these axes shows that non-local features and task supervision are the main sources of class information in concept scores, while closing all four pathways brings this information down to the level of the concept annotations; (iv) We show that locality makes grounding measurable but does not guarantee it: under class-level supervision, local concept scores use the head as a proxy for body, wing, and tail concepts, and CEM concept scores barely respond when the evidence that should support them is occluded.

## 2 Related Work

Concept models and the CUB benchmark. CBMs [15] decompose prediction into a concept predictor followed by a downstream label predictor, making the intermediate concept vector available for inspection and intervention. Subsequent models increase flexibility through continuous concept embeddings [8], probabilistic concept uncertainty [14], and broader concept-based architectures [23]. CUB-200-2011 [26] has become the standard visual benchmark for this line of work because it provides bird species labels, attribute annotations, and part locations. This makes it a natural testbed for CBMs, but species accuracy on CUB is a weak signal of concept use: images of one species share many visual cues beyond the annotated attributes, so a model can recognize the species from those cues even when its concept scores are not grounded in the attributes they name.

Information leakage in concept models. Mahinpei et al. [17] and Havasi et al. [10] show that soft concept scores can encode task information unavailable to hard concept predictors; discretizing the scores [10] or deriving them from discrete prototypes [5] is therefore often taken to prevent leakage. Under class-level labels, however, a hard concept vector can still identify the species (Section 4), so discreteness alone does not remove class information from the concept interface. Parisini et al. [22] develop an information-theoretic account of concept-task and interconcept leakage and show that CEM-style representations can carry task information beyond scalar concept activations. Related work proposes direct leakage metrics and mitigations, including irrelevant-concept contribution scores [25] and information-theoretic leakage measures [18]. Espinosa Zarlenga [7] argues that leakage is not always harmful: when the concept set is incomplete, some leakage can be necessary for accurate and intervenable models, provided it is properly localized. We share this view and do not treat every gain in accuracy as a defect. CUB is such a concept-incomplete setting, and deciding whether its leakage is benign requires knowing where the additional information comes from and whether the concept scores stay grounded. Our audit provides exactly this decomposition, across exposed concept scores, internal representations, receptive-field choices, and supervision regimes.

Evaluating concept-interface faithfulness. Intervention metrics [15; 8], concept-representation metrics [29], and completeness scores [28] are common ways to evaluate whether concepts are useful. These metrics answer different questions from the one studied here. Interventions test whether user corrections improve predictions; completeness compares concepts to a reference predictor; leakage metrics quantify specific forms of unintended information. We instead start from the benchmark itself: given the CUB annotations and the exposed concept interface, how much of the reported task accuracy is visible at that interface, and which architectural choices make the remainder available?

Spatial grounding and local evidence. For visual concepts, faithfulness also has a spatial component: a model may predict a concept correctly while using evidence from the wrong image region, a form of shortcut learning [9]. On CUB, Margeloiu et al. [20] give qualitative saliency-map examples in which concepts such as wing pattern or leg color attend to the whole bird rather than to the corresponding part. Huang et al. [12] turn this observation into a benchmark: using the CUB part keypoints, they test whether the peak of a concept's localization map falls near the annotated part, and find that most concepts of standard CBMs fail this test across eight backbones. Raman et al. [24] formalize the property as locality and measure it with perturbation, intervention, and masking metrics; their CUB masking results for CEM and ProbCBM change concept accuracy by at most 3.5 points, whether the relevant or an irrelevant region is masked, and Jeon et al. [13] find that the concept scores of label-free CBMs retain 87–90% of their value when the concept's region is deleted. We read these results as insensitivity of the concept predictions to the concept region, in line with our part-occlusion results for CEM (Section 6.4).

![](images/debabcc5448e731d4cec4a3ed5791caa641b413f2d43c5df31cf1ddd083b971d.jpg)  
Figure 1: Comparison of class-level and instance-level annotations for two sample images. Concepts are color-coded by source: class-only (orange; not visually observable to the annotator), instance-only (blue), and shared (green). In yellow, we highlight annotated concepts judged by the authors to be not visually observable. A subset of the concepts is shown for visualization purposes.

Several methods aim to restore locality: part-prototype CBMs in the style of ProtoPNet [3] with feature-alignment losses [12], CLIP-guided prototypes with a localization loss [13], and concept prototypes computed on segmented image parts [5]. These approaches encourage locality through training objectives or external segmenters rather than guaranteeing it: deep feature maps need not be spatially aligned with the input [11; 12], and on CUB, where no segmenter is used, PGCM treats the whole image as a single part [5]. Following BagNet-style models [2; 6], we instead restrict the receptive field by construction: each concept score is computed from a single image patch of known size, so locality holds exactly and the patch behind every score can be traced without attribution methods. We use this restriction as a controlled ablation axis to ask how much of the task accuracy reported on CUB depends on non-local evidence. Our localization and occlusion tests build on the keypoint- and removal-based protocols above [12; 24], but break results down by anatomical part and compare class- with instance-level supervision. This comparison shows that locality alone does not ensure grounding: under class-level labels, local concept scores use the head as a proxy for other parts.

## 3 Preliminaries

Concept Bottleneck Models Concept-bottleneck models (CBMs) [8; 15] can be written as a composition $f \circ s \circ g : \mathcal { X } \ \overset { g } { \to } \ \mathcal { Z } \ \overset { s } { \to } \ \mathcal { C } \ \overset { f } { \to } \ \mathcal { y }$ , where $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { D }$ is the input space, Z a latent space, ${ \mathcal { C } } \subseteq [ { \dot { 0 } } , 1 ] ^ { k }$ the concept space, and  the label space. Concept-based data take the form $\{ ( x _ { i } , c _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , where $c _ { i }$ is the concept annotation vector for $x _ { i }$ . The defining constraint—the concept bottleneck—is that the predicted class, Î, depends on x only through the predicted concepts ê, limiting the task-relevant information accessible to the classifier.

The CUB Benchmark and Its Annotations The Caltech-UCSD Birds (CUB) dataset [26] is the principal benchmark for CBMs. CUB contains 11,788 images across 200 bird species, each annotated with 312 visual attributes (concepts). In the original dataset, annotations are at the instance level: each image is annotated by a single annotator who records per concept a binary label (present or absent), a visibility indicator via a certainty score κ ∈ {1, 2, 3, 4} ranging from not observable (1) to definitely (4). CUB also provides part keypoints for 15 anatomical regions with visibility flags, which we use for spatial faithfulness evaluation.

In practice, CBM evaluations rely on a processed version introduced by Koh et al. [15]: 312 attributes are reduced to 112 binary concepts, and class-level concept labels are obtained via majority voting so that all images of the same species share identical concept labels. While intended to reduce noise, this introduces aggregation artifacts. We refer to the two protocols as by-class (majority-voted) and by-instance (original per-image annotations).

## 4 Annotation Limits and a Concept-Only Reference for CUB

The premise of CBMs is that predictions are mediated by human-annotated concepts; a model that adheres strictly to this premise can only be as accurate as the concepts allow. We show that the CUB concept annotations [26], both in their original form and as processed by prior work [15], limit how reliably concepts can be supervised and how much task information they can carry. The by-class protocol [15] assigns unique but identical concept vectors to all images of a species, meaning a model that identifies the correct species can achieve 100% concept accuracy without inspecting any image-level evidence, simply by using the predicted class label as a key for per-class lookup table (see Appendix C). The by-instance protocol [26] provides visually grounded supervision but introduces its own noise via single-annotator labels and subjective certainty scores. Both regimes, therefore, impose a structural limit on performance.

By-Instance vs. By-Class Supervision Faithful concept-based reasoning requires attributes to be visible in each image, making instance-level annotations the most appropriate supervision signal. To compare by-class and by-instance protocols, we measure their agreement as the average conceptwise accuracy between instance-level annotations and class-level labels across all image-concept pairs, corresponding to an oracle that predicts true image-level concepts evaluated under class-level supervision. Agreement is bounded at ≈82% (at κ ≥ 1) and increases only marginally to ≈84% under high-certainty annotations κ = 4 (retaining ≈55% of pairs; Appendix A, Fig. 4(a)), indicating a structural mismatch rather than one driven by low-confidence labels. Concretely, for roughly one in five image-concept pairs, the class-level label and the annotator's judgment for that image disagree: either the class-level label marks a concept as present that the annotator did not see, or the annotator saw a concept that the class-level label marks as absent. Three sources contribute to this disagreement; Figure 1 shows examples, which occur frequently across the dataset.

Limited observability. Many concepts that are typical of a species are not visible in individual images because of pose, occlusion, or resolution, yet the class-level label still marks them as present (Figure 1, left: leg color and size). Non-visibility rates are particularly high for fine-grained attributes such as wing or tail concepts, and these concepts consistently show lower agreement (Fig. 4(b) in Appendix A).

Aggregation. Koh et al. [15] construct class-level labels by majority voting: a concept is marked present for all images of a species if it is annotated as present in most of them, and concepts present in too few classes after voting are removed, which reduces the 312 attributes to 112. Majority voting removes intra-class variation, e.g., between sexes or plumages: a concept that the annotator saw in an image is marked absent if it occurs in only a minority of the species' images, and vice versa.

Annotation noise. Each image is annotated by a single Amazon Mechanical Turk worker, so instancelevel labels themselves contain errors: a concept can be marked as present although the corresponding part is not visible (Figure 1, right: body and tail concepts annotated although only the head is visible).

These observations raise the question of what near-perfect concept accuracy under by-class labels means. Concept predictors that follow the per-image evidence would agree with the class-level labels on only ≈82% of pairs; accuracies well above this level are only reachable by predicting class-typical concepts that are not visible in the image, i.e., from cues to the species rather than from the concept itself.

Concept-Only Reference Accuracy The annotation limits also bound how much task information the concepts can carry. We estimate this by training predictors directly on the binary ground-truth by-instance concept vectors (by-class vectors are excluded because they are identical per species and allow the lookup solution described above). Across linear logistic regression, a two-layer MLP, and gradient boosting (XGBoost), the best test accuracy is 37.5% (two-layer MLP; Table 1, full results in Appendix C). This value is not a strict upper bound: the instance-level annotations are noisy, and image models may recover graded visual evidence that the binary labels do not capture. We therefore treat ≈37% as the centre of a reference band: concept scores that carry only the information expressed by the annotated concepts should support task accuracy near it. Accuracies in the 70%+ range therefore do not by themselves show that concept scores are bypassed or ungrounded, but they call for direct audits of where the additional information comes from (Section 6).

Table 1: Concept-only reference accuracy on CUB. Classifiers receive only ground-truth instancelevel binary concept annotations as input (all certainty levels, i.e., no concept is masked). We report the best test accuracy across concept-to-label predictors. The full table, including certainty-based masking, is available in Appendix C.
<table><tr><td>Encoding</td><td>Best predictor</td><td>Test accuracy</td></tr><tr><td>Binary (±1)</td><td>MLP-2x256</td><td>37.50</td></tr></table>

## 5 How Task Information Can Enter Concept Scores

Many CBMs report task accuracy well above the reference band of Section 4. Part of this gap may reflect legitimate visual evidence that the binary annotations do not capture, but it raises the question of which design choices let task information into the concept scores without that information being backed by visual evidence for the concepts. We identify four such design axes (Section 5.1). Each axis has a standard setting, used by common CBMs and CEMs, and a constrained setting that closes the corresponding pathway. Section 6.2 sweeps these axes within a single model template (Section 5.2).

## 5.1 Four Design Axes

Receptive field of the features. Many CUB concepts describe spatially localized attributes (e. g. has a brown head, has an orange beak), yet they are standardly predicted from backbones such as ResNet18/50 whose receptive fields cover most or all of the image. A concept predictor can then exploit contextual or co-occurring evidence (e. g., background, pose, or other body parts) instead of the attribute itself. Constrained setting: a BagNet backbone [2] with a fixed k× k receptive field per spatial location, k ∈ {9, 17, 33}, so that every feature depends only on a local image patch.

Shared vs. concept-specific representations. Standard CBMs predict all concepts from one shared embedding g(x). Since all concept predictors read the same features, the model is encouraged to encode features that are jointly useful for many concepts, which can lead to correlated activations: several concepts fire together even when only a subset is present. This does not directly imply leakage, but concept predictions are then not disentangled and can share class-level information. Constrained setting: concept-specific heads, i.e., a separate projection per concept on top of the backbone, so that each concept is scored from its own feature vector rather than from an embedding shared with all other concepts.

Norm-dependent scoring. Standard concept scores are dot products between a shared embedding and a concept weight vector,

$$
s _ { i } ( \boldsymbol x ) = w _ { i } ^ { \top } g ( \boldsymbol x ) = \| w _ { i } \| \| g ( \boldsymbol x ) \| \cos ( \theta _ { i } ) ,
$$

so a score can increase through the feature norm $\| g ( x ) \|$ rather than through better alignment cos $( \theta _ { i } )$ with the concept. On a standard CBM, feature norm and the number of saturated concepts are strongly correlated $( r = 0 . 5 7 5 )$ : large norms activate many concepts at once. Normalizing both vectors removes saturation but collapses task accuracy, indicating that magnitude carries class information beyond concept alignment (Appendix B). Constrained setting: spherical scoring, i.e., cosine similarity between $\ell _ { 2 } \cdot$ -normalized vectors. This constraint presupposes concept-specific feature vectors. On the unit sphere, a single feature vector can be close to two concept vectors only if these are close to each other. A shared feature vector could therefore give high scores to several distinct concepts at once only by making their concept vectors nearly identical. Concept-specific vectors avoid this: each concept is scored against its own feature vector, so several concepts can be active in the same image region.

Task supervision of the concept scores. In joint training, the label loss backpropagates into the concept predictor. Concept scores can then be shaped toward class-discriminative features regardless of whether these features support the concept; Margeloiu et al. [20] report that jointly trained CBMs encode target information beyond the concepts. Constrained setting: concept-only training, i.e., the first stage of sequential training [15], with task information measured afterwards from frozen concept scores.

## 5.2 A Common Template for the Ablation

To vary these axes in a controlled way, we use a single concept-model template; implementation details and an overview figure are in Appendix F.1. A backbone (ResNet50 or BagNet-k) computes a feature map $\boldsymbol { g } ( \boldsymbol { x } ) \in \mathbb { R } ^ { C \times \breve { H } \times W }$ . A projection maps it to concept latents $z _ { i } ( u , v ) \in \mathbb { R } ^ { d } , i = 1 , \cdot . . , K$ at every spatial location $( u , v )$ , either through one head $\phi _ { i }$ per concept or through a single shared head. Each latent is compared with a learnable concept vector $p _ { i } \in \mathbb { R } ^ { d }$ and max-pooled into a concept score,

$$
\begin{array} { r l r } & { } & { s _ { i } ( \boldsymbol { x } ) = \displaystyle \operatorname* { m a x } _ { u , v } \hat { z } _ { i } ( u , v ) ^ { \top } \hat { p } _ { i } \in [ - 1 , 1 ] , \qquad \hat { z } _ { i } = z _ { i } / \| z _ { i } \| , \hat { p } _ { i } = p _ { i } / \| p _ { i } \| , } \end{array}
$$

i.e., spherical scoring. Concept-only variants minimize a weighted binary cross-entropy

$$
- \sum _ { ( x _ { j } , c _ { j } ) \in D } \sum _ { i = 1 } ^ { K } \lambda _ { j , i } \left[ c _ { j , i } \log \left( \hat { c } _ { j , i } \right) + \left( 1 - c _ { j , i } \right) \log \left( 1 - \hat { c } _ { j , i } \right) \right] ,\tag{1}
$$

where $\hat { c } _ { j , i } = \sigma ( s _ { i } ( x _ { j } ) )$ . With by-class labels, $\lambda _ { j , i } = 1$ . With by-instance labels, $\lambda _ { j , i }$ down-weights uncertain annotations according to the annotator certainty $\kappa _ { j , i } ~ \in ~ \{ 1 , 2 , 3 , 4 \}$ (Section 3), with $\lambda = 0 . 1 , 0 . 4 , 0 . 7 5 , 1$ respectively. Joint variants add a cross-entropy loss on a linear class head.

Measuring class information in concept scores. For every variant, task accuracy is measured by a bias-free linear probe $\hat { y } = W ^ { \top } s$ fit on frozen concept scores. The probe is the only path from the image to a class prediction, so no variant can bypass its concept scores; the probe accuracy therefore measures how much class information the scores themselves carry. We omit the bias so that the probe has no class preference when no concept evidence is present.

We refer to the configuration in which every axis is set to its constrained value—BagNet-33, conceptspecific heads, spherical scoring, and concept-only training—as the fully constrained configuration, denoted LC-CBM in tables. It is not proposed as a competitive model but is the end point of the sweep; the choice of BagNet-33 among the local backbones is itself an outcome of the receptive-field sweep (Section 6.2).

## 6 Experiments

The experiments follow the two failure modes of Section 1. Leakage probes first test whether CBM and CEM predictions bypass their exposed concept scores (Section 6.1). A controlled ablation over the design axes of Section 5 then measures how much class information enters the concept scores themselves (Section 6.2). Finally, spatial localization and part occlusion test whether concept scores are driven by the image regions that should support them (Sections 6.3 and 6.4).

Setup. We audit Joint CBM [15] and CEM [8], both with a ResNet18 backbone on full images (Appendix F), and the template variants of Section 5.2, which use a bounding-box crop. Results are mean ± std over three seeds with early stopping on validation concept accuracy. All experiments use CUB-200-2011 [26] with the $K \stackrel { \cdot } { = } 1 1 \stackrel { \cdot } { 2 }$ concepts of Koh et al. [15] and 200 classes, under the by-class and by-instance regimes of Section 3. Both regimes use the same 112 concepts: by-instance labels are the original per-image CUB annotations of these concepts, so the two regimes differ only in the labels, not in the concept set. Chance level is 0.5% (1/200 classes).

## 6.1 Do CBM and CEM Predictions Bypass the Concept Scores?

We begin with the first failure mode: whether a model's native predictions can be reconstructed from its exposed concept scores. Let $s ( x )$ denote the continuous concept scores predicted by a model, and let $d ( x )$ denote their discretized version. We fit linear probes $\pi _ { \mathrm { c o n t } } : s ( x ) \mapsto y$ and $\pi _ { \mathrm { d i s c } } : d ( x ) \mapsto y$ using only concept outputs. For CEM, we also fit a probe $\pi _ { \mathrm { e m b } } : e ( x ) \mapsto y$ on the internal concept embedding $e ( x )$ $\operatorname { A c c } _ { \mathrm { p r o b e , d i s c } }$ and $\mathrm { A c c } _ { \mathrm { p r o b e , c o n t } }$ quantify the ease of reconstructing class labels from discrete and continuous predicted concepts, respectively. $\operatorname { A c c } _ { \mathrm { n a t i v e } }$ denotes the model's own task-head accuracy. All probes are fit with scikit-learn logistic regression to convergence (no solver warnings across settings), so probe gaps are not an artifact of optimizer underfitting.

We report three gaps: the bypass gap $\Delta _ { \mathrm { b y p a s s } } = \mathrm { A c c } _ { \mathrm { n a t i v e } } - \mathrm { A c c } _ { \mathrm { p r o b e , c o n t } } ,$ capturing whether the model relies on information outside the predicted concepts (a positive value indicates a non-concept decision path); the encoding gap $\Delta _ { \mathrm { e n c } } = \mathrm { A c c } _ { \mathrm { p r o b e , c o n t } } - \mathrm { A c c } _ { \mathrm { p r o b e , d i s c } } ,$ measuring information lost when concepts are discretized; and, for CEM, the embedding leakage $\Delta _ { \mathrm { e m b } } = \mathrm { A c c } _ { \mathrm { p r o b e , e m b } } -$ $\mathrm { A c c } _ { \mathrm { p r o b e , c o n t } }$ , quantifying label information present in internal representations but not in the exposed concept outputs.

Table 2: Concept accuracy, AUC, and leakage probe results (% accuracy, mean ± std). Nat.: native task accuracy; Cont./Disc./Emb.: linear probes on continuous, discretized, and internal-embedding concept outputs. $\Delta _ { b } , \Delta _ { e n c } , \Delta _ { e m b }$ quantify bypass, encoding, and embedding leakage. †: fully constrained configuration of the ablation (Section 5.2), shown for reference; its predictions come only from the probe, so $\Delta _ { b } = 0$ by construction. Gaps are computed before rounding.
<table><tr><td>Model</td><td>| Concept Acc.</td><td>AUC</td><td>Nat.</td><td>Cont.</td><td>Disc.</td><td>Emb.</td><td> $\Delta _ { b }$ </td><td> $\Delta _ { e n c }$ </td><td> $\Delta _ { e m b }$ </td></tr><tr><td colspan="10">by-instance</td></tr><tr><td>CEM</td><td> $7 9 . 8 3 { \pm } 0 . 4 3$ </td><td> $6 6 . 5 7 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $6 7 . 5 2 { \scriptstyle \pm 2 . 4 8 }$ </td><td> $5 2 . 2 1 { \pm } 1 . 9 0 $ </td><td> $3 8 . 7 6 { \scriptstyle \pm 2 . 1 5 }$ </td><td> $6 8 . 5 6 { \pm } 1 . 2 3 $ </td><td>15.31</td><td>13.45</td><td>16.34</td></tr><tr><td>CBM</td><td> $8 0 . 2 2 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $6 3 . 9 7 { \scriptstyle \pm 0 . 4 8 }$ </td><td>58.20±2.67</td><td> $5 7 . 1 6 { \pm } 1 . 4 8 $ </td><td> $5 2 . 0 6 { \pm } 2 . 3 5 $ </td><td>一</td><td>1.04</td><td>5.11</td><td>1</td></tr><tr><td>LC-CBM†</td><td> $7 4 . 8 4 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $7 0 . 2 0 { \scriptstyle \pm 0 . 1 3 }$ </td><td>一</td><td> $2 5 . 2 3 { \scriptstyle \pm 0 . 4 8 }$ </td><td> $2 1 . 8 4 { \pm } 1 . 4 5$ </td><td>1</td><td>0.00</td><td>3.39</td><td>一</td></tr><tr><td colspan="10">by-class</td></tr><tr><td>CEM</td><td>94.89±0.15</td><td> $9 0 . 2 3 { \scriptstyle \pm 0 . 5 4 }$ </td><td>76.91±0.18</td><td>69.68±0.83</td><td> $6 9 . 5 4 { \pm } 1 . 0 7$ </td><td> $7 4 . 9 8 { \scriptstyle \pm 0 . 7 4 }$ </td><td>7.23</td><td>0.14</td><td>5.30</td></tr><tr><td>CBM</td><td>94.78±0.12</td><td>89.39±0.08</td><td>72.64±0.16</td><td>69.03±0.68</td><td> $6 8 . 4 5 { \scriptstyle \pm 0 . 4 7 }$ </td><td></td><td>3.61</td><td>0.58</td><td></td></tr><tr><td>LC-CBM†</td><td>90.35±0.38</td><td> $8 3 . 1 7 { \scriptstyle \pm 0 . 0 6 }$ </td><td>一</td><td> $4 2 . 8 2 { \pm } 0 . 1 3$ </td><td> $4 5 . 9 7 { \scriptstyle \pm 1 . 1 2 }$ </td><td>一</td><td>0.00</td><td>-3.15</td><td>一</td></tr></table>

Table 2 reveals qualitatively different behaviors across models. Across both evaluation settings, CEM shows a substantial bypass gap (e.g., +7.23 by-class and +15.31 by-instance), indicating that its native predictions cannot be reconstructed from exposed concepts alone. Moreover, the embedding probe consistently outperforms the concept probe (e.g., $\Delta _ { \mathrm { e m b } } = + 5 . 3 0$ by-class and +16.34 by-instance), demonstrating that class information is stored in internal representations that are not reflected in the concept outputs. This gap widens significantly in the more challenging by-instance setting, suggesting increased reliance on non-concept information.

The embedding probe also serves as a capacity check on the probing procedure itself: fit on the full internal representation, $\pi _ { \mathrm { e m b } }$ matches (by-instance: 68.56% vs. native 67.52%) or nearly matches (by-class: 74.98% vs. native 76.91%, within 2pp) native accuracy. Since this is the same linear probing procedure applied to a strictly higher-dimensional representation, its ability to recover nativelevel accuracy indicates that the probe is not the limiting factor; the smaller gaps observed for the continuous and discretized concept-score probes are therefore unlikely to reflect probe underfitting rather than genuine information loss at the exposed interface.

CBM has a smaller bypass gap than CEM (e.g., +3.61 by-class), but does not eliminate it, indicating residual dependence on non-concept pathways. We note that this gap is not directly comparable to CEM's: CBM's label predictor $f$ is trained jointly with the backbone rather than fit post-hoc on frozen scores, so part of the gap may reflect this training-regime mismatch rather than a distinct non-concept pathway; we do not have an analogous embedding-level probe for CBM to disentangle the two. Additionally, non-trivial encoding gaps (e.g., +0.58 by-class and +5.11 by-instance) suggest that discretized concepts lose task-relevant information, particularly in the by-instance regime.

Table 2 also lists the fully constrained configuration of Section 5.2 for reference. Its class predictions come only from the probe, so $\Delta _ { \mathrm { b y p a s s } } = 0$ holds by construction; its probe accuracy is discussed with the ablation below.

## 6.2 Where Does the Class Information in Concept Scores Come From?

The probes address the first failure mode. The second one does not require a bypass: the concept scores can themselves carry class information that is not backed by visual evidence for the concepts they represent. We therefore sweep the design axes of Section 5 within the template of Section 5.2 and measure, for each configuration, concept quality and the class information that a probe recovers from frozen concept scores (Table 3). Because no configuration can bypass its scores, differences in probe accuracy reflect differences in what the scores encode, not in how the scores are used.

Receptive field is the main lever. With concept-specific heads and concept-only training fixed, probe accuracy falls monotonically as the receptive field shrinks: 72% for ResNet50, 43% for BagNet-33, 32% for BagNet-17, and 16% for BagNet-9. Most of the class information that ResNet50 concept scores carry therefore comes from visual context beyond a local patch. The sweep also shows that locality has a cost of its own: concept accuracy drops from 95.8% (ResNet50) to 90.4% (BagNet-33), 84.0% (BagNet-17), and 76.2% (BagNet-9), because very small patches miss the spatial extent needed for shape, pattern, or multi-part color attributes. BagNet-33 is the most local setting in the sweep that still predicts concepts reliably, which is why the fully constrained configuration uses it.

Table 3: Systematic ablation of the design axes of Section 5 (mean ± std, by-class annotations). All configurations use the template of Section 5.2 with spherical scoring; task accuracy is measured by linear probes on continuous (Cont.) and discretized (Disc.) frozen concept scores, so no configuration can bypass its scores. †: fully constrained configuration (LC-CBM).
<table><tr><td>Backbone</td><td>Heads</td><td>Training</td><td>Concept quality Conc. Acc</td><td>Conc. AUC</td><td>Task Acc (probe)</td><td></td></tr><tr><td>Head sharing</td><td></td><td></td><td></td><td></td><td>Cont.</td><td>Disc.</td></tr><tr><td colspan="7"></td></tr><tr><td>ResNet50 ResNet50</td><td>Shared Specific</td><td>Concept-only</td><td>82.53±0.16</td><td>84.03±0.49 92.85±0.21</td><td>64.66±0.62 72.11±0.55</td><td>60.57±0.57 72.59±0.68</td></tr><tr><td>BagNet-33</td><td>Shared</td><td>Concept-only Concept-only</td><td>95.79±0.13 77.21±0.52</td><td>70.47±3.79</td><td>20.63±5.30</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>13.40±4.33</td></tr><tr><td colspan="7">Task supervision</td></tr><tr><td>BagNet-33</td><td>Specific Joint</td><td></td><td>70.58±3.16</td><td>72.43±2.40</td><td>54.95±4.29</td><td>55.29±3.60</td></tr><tr><td colspan="7">Receptive field (specific heads, concept-only)</td></tr><tr><td>BagNet-33†</td><td></td><td>Specific Concept-only</td><td>90.35±0.38</td><td>83.17±0.06</td><td>42.82±0.13</td><td>45.97±1.12</td></tr><tr><td>BagNet-17</td><td></td><td>Specific Concept-only</td><td>84.03±0.61</td><td>79.95±0.27</td><td>32.04±0.70</td><td>31.81±0.54</td></tr><tr><td>BagNet-9</td><td></td><td>Specific Concept-only</td><td>76.21±0.47</td><td>73.96±0.28</td><td>15.92±1.23</td><td>12.62±1.27</td></tr></table>

Concept-specific heads are a precondition, not a cost. Replacing concept-specific heads with a single shared head lowers both concept quality and probe accuracy, with ResNet50 (95.8 → 82.5% concept accuracy; 72 → 65% probe accuracy) and much more strongly with BagNet-33 (90.4 → 77.2%; 43 → 21%). The shared-head variants keep spherical scoring, so a shared latent at a given location can align only with concept vectors that are close to each other (Section 5). Distinct concepts that occur in the same patch, such as the color and pattern of a wing, then compete for one direction. This interaction is strongest when each location sees only a small patch, which is consistent with the larger drop for BagNet-33. Concept-specific heads are therefore needed for usable concept scores under spherical scoring rather than being a source of lost accuracy.

Norm-dependent scoring. All template variants use spherical scoring, so the ablation does not isolate the scoring axis. Evidence for it comes from the normalization study on a standard CBM (Appendix B), where removing the norm dependence removes concept saturation but also collapses task accuracy, i.e., feature magnitude carries class information beyond concept alignment.

Task supervision injects class information at the expense of the concepts. Adding joint task training to BagNet-33 with concept-specific heads raises probe accuracy from 43% to 55% while lowering concept accuracy from 90.4% to 70.6%. Even with local features and separate heads, task gradients push the concept scores toward class-discriminative features that the concept labels do not support.

The constrained end point. Setting every axis to its constrained value yields the fully constrained configuration (LC-CBM). Its probe accuracy (43% by-class, 25% by-instance; Table 2) lies within the reference band around 37%: once the pathways of Section 5 are closed, the concept scores carry roughly as much class information as the ground-truth concept annotations do. Its lower task accuracy is therefore not a failure of this configuration but the expected outcome of the sweep.

The grid is not fully factorial, and swapping the backbone may change more than the receptive field. We therefore read the sweep as mechanistic evidence for where class information enters the concept scores, not as an exact causal decomposition of CBM or CEM accuracy.

## 6.3 Is Locality Enough for Grounding? Spatial Localization

Closing the pathways of Section 5 limits how much class information the concept scores carry, but it does not guarantee that each score is driven by the evidence a human would use to recognize the concept. Local features, however, make this question measurable: because every BagNet-33 feature depends only on a 33×33 patch, a concept score can be traced back exactly to the patch that produced it, without attribution approximations. We use this to test whether the concept activations of the fully constrained configuration peak near the expected anatomical part. Training provides no incentive for this: part keypoints are never used, and the only signal is the image-level concept label applied to the max-pooled score. Correct localization therefore has to emerge from which patches best predict the concept label, which is exactly what the test probes. CBM and CEM are excluded because their global features do not expose comparable spatial concept maps. We compare the configuration trained with class-level and with instance-level annotations, isolating the effect of the supervision signal. We localize each concept at its peak activation and, following Huang et al. [12], compare it to the CUB part keypoints, reporting PCK@0.15.2 Evaluation is restricted to samples where the concept is present, annotator certainty is above threshold, the model predicts the concept, and the keypoint is visible (full protocol in Appendix D). Figure 2 shows a clear difference between training regimes. Under instance-level supervision, localization is more balanced across parts (PCK@0.15: 0.59 head, 0.46 body, 0.08 tail, 0.04 leg), whereas class-level supervision produces a strong head bias (PCK@0.15: 0.80 head, 0.18 body). The same holds for a stricter test that asks whether an activation lies closer to the mapped part than to a competing part sharing the same attribute: 0.569 under instance-level supervision versus near-chance 0.519 under class-level supervision.

![](images/fdbb9feaa0901f064074aa0b3868d54492fbc20d26d088496a82d6ccf4343463.jpg)

![](images/376062d44ad2ff489f608ebc939d4519210488757e088a510a029bc1c9751258.jpg)  
Figure 2: Spatial faithfulness (PCK@0.15) by anatomical region. Models trained with by-class annotations (left) exhibit strong head bias, while by-instance supervision (right) yields more balanced localization across body parts.

Head-proxy behavior. Under class-level supervision, body, wing, and tail concepts are predicted predominantly from the head region (head/body PCK ratio 4.4×). Since all images of a species share the same concept vector, the head, which is highly discriminative for species identity, becomes a proxy for every concept of that species. Instance-level supervision weakens this shortcut, reducing the ratio to 1.3×. Locality therefore constrains where the evidence for a concept can come from, but the supervision signal decides which local evidence is used.

## 6.4 Do Concept Scores Respond to Their Evidence? Part Occlusion

Spatial localization only applies to models with local features. As a behavioral test that applies to any model, we adapt the part-removal tests of [12; 13]: we blur the anatomical part mapped to each positive concept and compare the resulting change of the concept score against random and unrelated-part masks. A grounded concept score should drop when its supporting part is removed, and more so than under control masks. We compare the fully constrained configuration with CEM.

Because the two models differ in backbone, preprocessing, score scale, and evaluated pairs (Section 6), we compare within-model contrasts rather than raw score drops across models (Table 4). For the fully constrained configuration, whose scores are cosine similarities in [–1, 1], occluding the mapped part lowers the concept score by 0.437 on average, well above the drop under random masks (target– random contrast 0.355). For CEM, whose scores are probabilities in [0, 1], occluding the mapped part lowers the concept score by only 0.034, barely more than a random mask (0.017 contrast), even though occlusion flips the native class prediction in 19.4% of evaluated pairs. CEM's predictions thus change under visual intervention while its exposed concept scores barely reflect the removed evidence—precisely the situation in which a human inspecting or correcting these scores would be misled.

Table 4: Part-occlusion audit under class-level CUB supervision. Scores are macro-averages over mapped anatomical groups. Target drop is the change in the exposed concept score when the mapped part is blurred; target-random and target-unrelated are specificity contrasts against control masks. Flip rate reports changes in the native class prediction after target occlusion. Backbone, preprocessing, score scale ([–1, 1] for LC-CBM, [0, 1] for CEM), and evaluated pairs differ between rows, so contrasts should be read within each row.
<table><tr><td>Model</td><td>Pairs</td><td>Target drop</td><td>Target-random</td><td>Target-unrelated</td><td>Flip rate</td></tr><tr><td>LC-CBM</td><td>69,228</td><td>0.437</td><td>0.355</td><td>0.153</td><td>29.1</td></tr><tr><td>CEM</td><td>25,994</td><td>0.034</td><td>0.017</td><td>0.003</td><td>19.4</td></tr></table>

The occlusion results also corroborate the head-proxy behavior. Under class-level supervision, the fully constrained configuration is highly sensitive to head masks (target-score drop 0.755; flip rate 49.4%) but much less sensitive to masks of non-head parts (target drop 0.135; flip rate 13.9%): its concept scores do respond to local evidence, but that evidence is often the head rather than the annotated part.

## 7 Discussion and Implications

Our main conclusion is not that high CUB accuracy is meaningless, nor that concept models should ignore task performance. Rather, CUB task accuracy is underdetermined as evidence for concept-based reasoning: the same number can arise from visible concept scores, internal embeddings, non-local image features, shared representations, or task-gradient shaping. For CBMs, this distinction matters because the model's scientific and practical promise is not only classification, but an inspectable and intervenable concept interface. In line with the view that some leakage can be benign when the concept set is incomplete [7], we do not argue that every source of additional accuracy should be removed; we argue that it should be identified before the accuracy is credited to the concepts.

The concept-only reference provides the first point of calibration. The CUB concept annotations explain far less task performance than modern CBM/CEM variants achieve. Because the annotations are noisy and incomplete, this is not a hard ceiling, but it is a warning that task accuracy alone cannot tell us whether a model reasons through the concepts a user can inspect.

The probes and the ablation then locate the additional accuracy. CEM's native predictions are not fully reconstructable from its scalar concept scores, and its internal embeddings contain the missing class information—the first failure mode. The ablation shows that the second failure mode does not need a bypass: within a single template, the class information carried by the concept scores rises from the level of the annotations to CEM-like values when the receptive field is enlarged, and rises again under joint task training, which at the same time lowers concept accuracy. High accuracy on CUB can therefore be recovered by relaxing these constraints, but doing so changes what the concept scores mean. The receptive-field sweep also shows that locality is not an unqualified virtue: very small receptive fields miss the visual extent needed for shape, pattern, or multi-part color attributes, while global ones make it easy to predict a concept from correlated evidence elsewhere in the image. The useful design question is therefore not “the smallest possible receptive field,"but whether the evidence used for a concept matches the evidence a user expects that concept to use.

The grounding results show that closing these pathways is necessary but not sufficient. Under class-level supervision, local concept scores often predict non-head attributes from the head, which is highly discriminative for species identity on CUB. Such head-proxy behavior is exactly the kind of mismatch that accuracy numbers hide: a concept score can be predictive at the scalar level while its evidence comes from a different region than a human would expect, so a user who corrects it after inspecting the annotated part is not correcting what the model used. Instance-level supervision reduces this bias but does not remove all localization failures, especially for small or peripheral parts such as tails and legs. Part occlusion makes the same point for CEM: its predictions change under targeted masks while its exposed concept scores remain nearly flat, so inspecting those scores would not reveal why the prediction changed.

Implications for evaluating CBMs. Our results suggest reporting, alongside native task accuracy, (i) the accuracy of a probe on frozen concept scores and the resulting bypass gap, (ii) where applicable, the class information in internal concept embeddings, (iii) the concept-only accuracy of the annotations used for supervision as a reference band, and (iv) for models with spatial concept maps, localization or occlusion checks against part annotations. Together, these quantities separate accuracy that is carried by grounded concept scores from accuracy that arises elsewhere.

The role of concept annotations. Our findings highlight the central role of concept annotations in determining what a faithful CBM can learn and explain. On CUB, the available concepts are noisy, partially observable, and incomplete, and class-level labels disagree with the per-image annotations often enough that near-perfect concept accuracy under these labels itself signals prediction from species cues rather than from visible concepts. This is not a failure of concept bottlenecks in general. It means that concept vocabularies and annotation protocols define the interpretive contract a model can satisfy. More precise and visually grounded annotations could support higher faithful accuracy, while unsupervised or generative concept discovery methods [1; 4; 16; 19; 27; 21] may expand the concept interface; their faithfulness, however, should be audited with the same care.

Limitations. We focus on CUB-200-2011 with 112 processed concepts, so the results should not be read as a universal estimate of leakage or faithful accuracy for all CBM datasets. The concept-only reference depends on noisy single-annotator instance labels encoded as binary concept vectors; it is a reference band, not a ground-truth ceiling. The ablation grid is not fully factorial, and swapping backbones may change more than the receptive field, so it supports mechanistic rather than exact causal conclusions. The CBM and CEM baselines use a different backbone (ResNet18) and preprocessing (no bounding-box crop) than the template configurations, and their concept scores live on different scales; we therefore interpret occlusion effects within each model rather than comparing raw effects across models. The spatial evaluation is limited to concepts with well-defined anatomical anchors and to correctly predicted concept instances, and therefore does not capture false positives or all forms of mislocalized evidence. Finally, we argue that ungrounded concept scores undermine interventions but do not evaluate interventions directly; our grounding analyses test their precondition.

## References

[1] David Alvarez Melis and Tommi Jaakkola. Towards robust interpretability with self-explaining neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

[2] Wieland Brendel and Matthias Bethge. Approximating CNNs with bag-of-local-features models works surprisingly well on imagenet. In International Conference on Learning Representations, 2019.

[3] Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K Su. This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32, 2019.

[4] Chaofan Chen, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K Su. This looks like that: Deep learning for interpretable image recognition. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[5] Stefano Colamonaco, David Debot, Pietro Barbiero, and Giuseppe Marra. Prototype-grounded concept models for verifiable concept alignment. In International Conference on Machine Learning, 2026.

[6] Kerol Djoumessi, Bubacarr Bah, Laura Kühlewein, Philipp Berens, and Lisa Koch. This actually looks like that: Proto-bagnets for local and global interpretability-by-design. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 718–728. Springer, 2024.

[7] Mateo Espinosa Zarlenga. In defense of information leakage in concept-based models. In International Conference on Machine Learning, 2026.

[8] Mateo Espinosa Zarlenga, Pietro Barbiero, Gabriele Ciravegna, Giuseppe Marra, Francesco Giannini, Michelangelo Diligenti, Zohreh Shams, Frederic Precioso, Stefano Melacci, Adrian Weller, et al. Concept embedding models: Beyond the accuracy-explainability trade-off. Advances in neural information processing systems, 35:21400–21413, 2022.

[9] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

[10] Marton Havasi, Sonali Parbhoo, and Finale Doshi-Velez. Addressing leakage in concept bottleneck models. Advances in Neural Information Processing Systems, 35:23386–23397, 2022.

[11] Adrian Hoffmann, Claudio Fanconi, Rahul Rade, and Jonas Kohler. This looks like that... does it? shortcomings of latent space prototype interpretability in deep networks. arXiv preprint arXiv:2105.02968, 2021.

[12] Qihan Huang, Jie Song, Jingwen Hu, Haofei Zhang, Yong Wang, and Mingli Song. On the concept trustworthiness in concept bottleneck models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 21161–21168, 2024.

[13] Sujin Jeon, Hyundo Lee, Eungseo Kim, Sanghack Lee, Byoung-Tak Zhang, and Inwoo Hwang. Locality-aware concept bottleneck model. arXiv preprint arXiv:2508.14562, 2025.

[14] Eunji Kim, Dahuin Jung, Sangha Park, Siwon Kim, and Sungroh Yoon. Probabilistic concept bottleneck models. In International Conference on Machine Learning, pages 16521–16540. PMLR, 2023.

[15] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International conference on machine learning, pages 5338–5348. PMLR, 2020.

[16] Oscar Li, Hao Liu, Chaofan Chen, and Cynthia Rudin. Deep learning for case-based reasoning through prototypes: A neural network that explains its predictions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

[17] Anita Mahinpei, Justin Clark, Isaac Lage, Finale Doshi-Velez, and Weiwei Pan. Promises and pitfalls of black-box concept learning models. arXiv preprint arXiv:2106.13314, 2021.

[18] Mikael Makonnen, Moritz Vandenhirtz, Sonia Laguna, and Julia E Vogt. Measuring leakage in concept-based methods: An information theoretic approach. arXiv preprint arXiv:2504.09459, 2025.

[19] Emanuele Marconato, Andrea Passerini, and Stefano Teso. Glancenets: Interpretable, leak-proof concept-based models. Advances in Neural Information Processing Systems, 35:21212–21227, 2022.

[20] Andrei Margeloiu, Matthew Ashman, Umang Bhatt, Yanzhi Chen, Mateja Jamnik, and Adrian Weller. Do concept bottleneck models learn as intended? In ICLR 2021 Workshop, 2021. arXiv:2105.04289.

[21] Tuomas Oikarinen, Subhro Das, Lam M. Nguyen, and Tsui-Wei Weng. Label-free concept bottleneck models. In The Eleventh International Conference on Learning Representations, 2023.

[22] Enrico Parisini, Tapabrata Chakraborti, Chris Harbron, Ben D MacArthur, and Christopher RS Banerji. Leakage and interpretability in concept-based models. arXiv preprint arXiv:2504.14094, 2025.

[23] Eleonora Poeta, Gabriele Ciravegna, Eliana Pastor, Tania Cerquitelli, and Elena Baralis. Conceptbased explainable artificial intelligence: A survey. ACM Computing Surveys, 2023.

[24] Naveen Janaki Raman, Mateo Espinosa Zarlenga, Juyeon Heo, and Mateja Jamnik. Do concept bottleneck models respect localities? Transactions on Machine Learning Research, 2025.

[25] Raffael Schoen, Baptiste Abeloos, and Stéphane Herbin. Measuring and addressing information leakage in concept bottleneck models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 624–632, 2025.

[26] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.

[27] Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

[28] Chih-Kuan Yeh, Been Kim, Sercan Arik, Chun-Liang Li, Tomas Pfister, and Pradeep Ravikumar. On completeness-aware concept-based explanations in deep neural networks. Advances in neural information processing systems, 33:20554–20565, 2020.

[29] Mateo Espinosa Zarlenga, Pietro Barbiero, Zohreh Shams, Dmitry Kazhdan, Umang Bhatt. Adrian Weller, and Mateja Jamnik. Towards robust metrics for concept representation evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 11791– 11799, 2023.

## A Analysis of Concept Annotation Reliability on CUB

This appendix provides additional analysis supporting the results in Sec . 4, focusing on the reliability of concept annotations under the standard CUB preprocessing pipeline [15].

Data and preprocessing We use the processed CUB dataset introduced by Koh et al. [15], with 112 binary concepts derived from the original 312 attributes via filtering, grouping, binarization, and majority voting. For each image, we extract: (i) class-level concept annotations used in prior work, (ii) instance-level concept annotations from the original dataset [26], and (iii) annotator certainty scores provided in [26].

Agreement metric Let $y _ { c } \in \{ 0 , 1 \}$ denote the class-level label for concept c, and let $\hat { y } _ { i , c } \in \{ 0 , 1 \}$ denote the instance-level annotation for concept c in image i. Each instance-level annotation is associated with a certainty score $\kappa _ { i , c } \in \{ 1 , 2 , 3 , 4 \}$

We evaluate agreement as a function of a certainty threshold t, retaining only annotations with $\kappa _ { i , c } \geq t$ . We define the set of retained image-concept pairs as

$$
K _ { t } = \{ ( i , c ) \mid \kappa _ { i , c } \geq t \} .
$$

Agreement is then defined as

$$
\mathrm { A g r e e m e n t } ( t ) = P ( \hat { y } _ { i , c } = y _ { c } \mid ( i , c ) \in K _ { t } ) = \frac { 1 } { \lvert K _ { t } \rvert } \sum _ { ( i , c ) \in K _ { t } } \mathbf { 1 } [ \hat { y } _ { i , c } = y _ { c } ] .
$$

This corresponds to the fraction of retained image-concept pairs for which instance-level annotations match class-level labels. It defines an oracle ceiling for class-level evaluation: a perfect model that predicts the true image-level concepts would achieve exactly this accuracy when evaluated against class-level labels.

Coverage-agreement trade-off We define Coverage as the fraction of image-concept pairs that survive the certainty filter $\kappa \geq t .$ More formally: $\begin{array} { r } { \mathrm { C o v e r a g e } ( t ) = \frac { \left| K _ { t } \right| } { \left| K _ { 1 } \right| } \mathrm { A t } t = 1 } \end{array}$ , coverage is 100% by definition (no pairs are rejected). $\ \mathrm { A t } \ t = 4 .$ , only pairs for which the annotator was fully certain are retained. Figures 4(a) and Fig. 3(a) show the trend of Agreement and Coverage with respect to the certainty threshold t. Specifically, we can see that increasing t leads to only modest improvements in Agreement (from $\approx 8 2 \% \ 1 0 \approx 8 4 \% )$ , while substantially reducing Coverage $( \mathrm { f r o m } \approx 1 0 0 \%$ to $\approx 5 0 \% )$ . This confirms that the mismatch between class- and instance-level annotations is structural rather than driven by low-confidence labels.

Disagreement decomposition To better understand the sources of annotation errors, we analyze the direction of Disagreement between image-level and class-level labels. Let $D _ { t }$ denote the set of disagreeing image-concept pairs in $K _ { t }$

We distinguish two complementary types of disagreement. The quantity $\delta _ { 0  1 }$ (class-positive miss) measures the fraction of disagreements where the image-level annotation indicates absence, while the class-level label indicates presence. This form of disagreement typically occurs when a concept characteristic of the class is not visible in a particular image (e.g., due to pose, occlusion, or viewpoint).

Conversely, $\delta _ { 1  0 }$ (class-negative contradiction) measures the fraction of disagreements where the image-level annotation indicates presence, but the class-level label indicates absence. This reflects when class-level concept prevalence is thresholded (e.g., at 50%), forcing minority instances within a class to be incorrectly labeled as negative.

Formally, these quantities are defined as:

$$
\delta _ { 0  1 } ( t ) = \mathbb { P } \big ( \hat { y } _ { i , c } = 0 , y _ { c } = 1 | ( i , c ) \in D _ { t } \big )\tag{2}
$$

$$
\delta _ { 1 \to 0 } ( t ) = \mathbb { P } \big ( \hat { y } _ { i , c } = 1 , y _ { c } = 0 \mid ( i , c ) \in D _ { t } \big )\tag{3}
$$

![](images/6bec273aeca793f8ed685306d7c72888d1f0e1d996244e7d5dde349b1abb9d92.jpg)  
(a) Coverage vs. certainty

![](images/ae3e04476e9ed434bf68ccbeb1419d1b536a4911536109bdab8d188115c50e0e.jpg)  
(b) Disagreement direction

Figure 3: Additional analysis of the class-instance annotation mismatch on CUB. (a) Increasing the certainty threshold substantially reduces coverage, dropping to roughly 55% at the highest threshold. (b) Disagreement direction reveals two sources of mismatch: class-positive concepts that are not visible in individual images, and instance-positive concepts suppressed by class-level aggregation.  
![](images/97efd7f293f1a2e465014e32946d39b183c8b38cf1b206b15c727c44897f290a.jpg)  
(a) Agreement vs. certainty

![](images/536e6b59b0535fb4d38bc7f3646ffcdaeebe2f706966a974fe4c070bcfb463e9.jpg)  
(b) Non-visibility by concept group  
Figure 4: Annotation mismatch in CUB. (a) Agreement between instance-level and class-level concept labels remains bounded around 82–84% across certainty thresholds, showing that filtering uncertain annotations only marginally improves alignment. class\_label=0 denotes disagreement between class-level (green) and instance-level annotations, while class\_label=1 denotes agreement; results over both cases are shown in blue. (b) Non-visibility rates vary substantially across concept groups, especially for fine-grained attributes, indicating that limited observability is a major source of mismatch.

By construction, $\delta _ { 0 \to 1 } ( t ) + \delta _ { 1 \to 0 } ( t ) = 1$

Figure 3(b) shows that the Disagreements are clearly asymmetric and become more so with higher certainty. $\delta _ { 1  0 }$ dominates (64–65%), meaning most conflicts occur when the image shows the concept but the class label says it is absent.

Concept observability We next analyze how agreement varies across concept groups. A concept group is a set of semantically related attributes describing a common part or property of the object (e.g., wing\_shape, $b i l l \_ c o l o r )$ , which are analyzed jointly due to their shared observability and noise characteristics.

We find that concepts that are difficult to observe (e.g., fine-grained shape or tail attributes) exhibit substantially lower agreement than consistently visible attributes such as global color or size (Figure 4(b)). This pattern closely mirrors the variation in non-visibility rates across groups, indicating that observability is a primary driver of annotation noise.

Table 5: Controlled normalization study on CUB. We independently normalize $g$ and w and report score decomposition, norm-saturation correlations, and downstream performance. In the standard configuration, \~99.7% of score variance is explained by the norm product. Fixing one norm shifts the dependency to the other, while fixing both removes norm-driven variability.
<table><tr><td colspan="2"></td><td colspan="3">Unconstrained g normalized w normalized Both normalized</td></tr><tr><td></td><td colspan="4">Concept score decomposition</td></tr><tr><td>Norm product contribution</td><td>0.997</td><td>0.651</td><td>0.995</td><td>0.000</td></tr><tr><td>Cosine contribution</td><td>0.003</td><td>0.349</td><td>0.005</td><td>1.000</td></tr><tr><td colspan="5">Norm-saturation correlation</td></tr><tr><td>Feature norm vs. saturation</td><td>0.575</td><td>-0.017</td><td>0.520</td><td>&lt;0.001</td></tr><tr><td>Weight norm vs. saturation</td><td>0.197</td><td>0.382</td><td>-0.039</td><td>&lt;0.001</td></tr><tr><td colspan="3"></td><td>Downstream performance</td><td></td></tr><tr><td>Concept accuracy</td><td>0.828</td><td>0.812</td><td>0.836</td><td>0.816</td></tr><tr><td>Task accuracy</td><td>0.398</td><td>0.183</td><td>0.404</td><td>0.034</td></tr></table>

Summary Overall, we find that: (i) Agreement between instance- and class-level annotations is fundamentally bounded $( \approx 8 2 \% ) ,$ (ii) Increasing annotation certainty yields only marginal improvements, (ii) Disagreement arises from both observability limitations and aggregation artifacts, and (iv) Annotation reliability varies substantially across concepts.

These results support the conclusion that the concept representation used in standard CUB benchmarks is inherently noisy and imposes a structural limit on concept-based learning performance.

## B Norm Effects in Concept Scoring

In this appendix, we analyze the role of feature and weight magnitudes in concept-bottleneck models (CBMs) and provide additional evidence supporting the claims in Section 5.

Score decomposition. Concept-bottleneck models (CBMs) [15] are composed as $f \circ g \circ h : \mathcal { X } \ { \overset { h } {  } }$ $\mathcal { Z } \overset { g } { \to } \mathcal { C } \overset { f } { \to } \mathcal { Y }$ , with predictions made from concept outputs rather than directly from image features. In standard CBMs, concepts are predicted via linear classifiers on the shared representation $z = h ( x )$ The score for concept i is $s _ { i } ( x ) = w _ { i } ^ { \top }$ z where $w _ { i }$ is the concept weight vector. This admits the decomposition

$$
s _ { i } ( x ) = \| z \| \| w _ { i } \| \cos ( \theta _ { i } ) ,\tag{4}
$$

where $\theta _ { i }$ is the angle between z and $w _ { i } ,$ separating semantic alignment $\left( \cos ( \theta _ { i } ) \right)$ from the scaling effects of the feature norm $\| z \|$ and the weight norm $\lVert \boldsymbol { w } _ { i } \rVert$

Concept scores in standard CBMs take the form

$$
s _ { i } ( \boldsymbol x ) = w _ { i } ^ { \top } g ( \boldsymbol x ) = \| g ( \boldsymbol x ) \| \| w _ { i } \| \cos ( \theta _ { i } ) ,\tag{5}
$$

where $g ( x )$ is the shared feature embedding, $w _ { i }$ is the concept weight vector, and $\theta _ { i }$ is the angle between $g ( x )$ and $w _ { i }$ . This decomposition separates semantic alignment $( \cos ( \theta _ { i } ) )$ from two independent scaling factors: the feature norm $\| g ( x ) \|$ and the weight norm $\lVert \boldsymbol { w } _ { i } \rVert$

Experimental setup. To isolate the contribution of each factor, we perform a controlled normalization study in which we independently normalize $\| g ( x ) \|$ and $\lVert \boldsymbol { w } _ { i } \rVert$ to 1. This results in four configurations: (i) standard CBM, (ii) g normalized $( \| g ( x ) \| = 1 )$ , (iii) w normalized $( \| w _ { i } \| = 1 )$ and (iv) both normalized (pure cosine similarity). All models are trained under identical conditions on CUB dataset by [15].

Norm-driven saturation. Figure 5 presents the relationship between feature norm $\| g ( x ) \|$ and the number of saturated concepts for all four configurations. The standard model (top-left) exhibits a strong positive correlation $( r = 0 . 5 7 5 )$ , indicating that larger feature norms systematically lead to more saturated predictions.

Norm Domination: Large |g(x)|| → Many Concepts Saturate (Standard — smoking-gun plot)  
![](images/13a6ddbffe307a752d811a91f45ad56cf837e43feac70e6ac9f854e7a9203805.jpg)

Norm Domination: Large |lg(x)|| → Many Concepts Saturate (g-fix — smoking-gun plot) 1e-7+1  
![](images/2a82335fe0a2ed93705767d16d8e811aea350a0acd1d2763c401f363d2db17cb.jpg)

Norm Domination: Large |lg(x)|| → Many Concepts Saturate (w-fix — smoking-gun plot)  
![](images/8435deef3c579d6ed8ff343b7810fba24516a24301aa81df2a45c7be37728b82.jpg)

Norm Domination: Large |g(x)|| → Many Concepts Saturate (Both-fix — smoking-gun plot) 1e-7+1  
![](images/2a0847344cd438bef3791f272b3b5151756bdd05662680772d06d102447c1f52.jpg)  
Figure 5: Feature norm $\| g ( x ) \|$ versus the number of saturated concepts across the four ablation settings. Top- $. l e f t i$ standard CBM. Top-right: $\operatorname { g - } \operatorname { f i x } .$ , where $\| g ( x ) \| = 1$ . Bottom- $. l e f t i$ w-fix, where $\| w _ { i } \| = 1$ . Bottom-right: both-fix, where both norms are constrained. The standard model exhibits a strong positive relation between feature norm and saturation. Fixing $\| g ( x ) \|$ removes this dependency, while fixing only $\lVert \boldsymbol { w } _ { i } \rVert$ leaves it largely intact. When both norms are fixed, saturation disappears entirely.

Fixing $\| g ( x ) \|$ (top-right) removes this dependency entirely $( r \approx 0 )$ , confirming that feature magnitude is a primary driver of saturation. In contrast, fixing $\lVert \boldsymbol { w } _ { i } \rVert$ (bottom-left) leaves the dependency largely unchanged $( r = 0 . 5 2 0 )$ , showing that constraining weights alone is insufficient. When both norms are fixed (bottom-right), saturation disappears completely, as scores are bounded and no longer depend on magnitude.

Compensatory behavior. Constraining a single norm does not eliminate magnitude effects, as the model compensates through the remaining degree of freedom. When $\| g ( x ) \|$ is fixed, weight norms increase substantially (mean $0 . 9 7 8  4 . 2 9 9 )$ , and their correlation with saturation rises $( r = 0 . 1 9 7  0 . 3 8 2 )$ . Conversely, fixing $\lVert \boldsymbol { w } _ { i } \rVert$ removes variability in weight norms but leaves feature norms free, preserving their influence on saturation. This behavior, visible across the four panels, indicates that the two norm factors act as interchangeable channels for controlling score magnitude.

Variance decomposition. We quantify the relative contribution of magnitude and alignment by decomposing score variance. In the standard model, the norm product $\| g ( x ) \| \cdot \| w _ { i } \|$ explains 99.7% of the variance, while cosine similarity contributes only $0 . 3 \bar { \% }$ . Fixing $\| g ( x ) \|$ increases the cosine contribution to 34.9%, but magnitude still dominates, whereas fixing $\lVert \boldsymbol { w } _ { i } \rVert$ has negligible effect. Only when both norms are fixed does cosine similarity account for the full variance (100%), confirming that standard CBMs rely almost entirely on magnitude rather than alignment.

Table 6: Concept-only task accuracy on CUB using ground-truth concepts. We report test accuracy (%) for different concept encodings, certainty thresholds t, and concept-to-label predictors. The best result per encoding is highlighted in bold.
<table><tr><td>Encoding</td><td>Model</td><td> $t = 1$ </td><td> $t = 2$ </td><td> $t = 3$ </td><td> $t = 4$ </td></tr><tr><td rowspan="3">Masked sign</td><td>Linear</td><td>32.46</td><td>32.46</td><td>31.50</td><td>19.62</td></tr><tr><td>MLP (2x256)</td><td>37.50</td><td>37.50</td><td>35.45</td><td>25.34</td></tr><tr><td>XGBoost</td><td>36.02</td><td>36.02</td><td>34.05</td><td>22.82</td></tr></table>

Impact on performance. While removing norm effects leads to a geometrically well-defined formulation, it comes at a high cost. Task accuracy drops from 0.398 in the standard model to 0.034 under both-fix, indicating that magnitude is critical for class separation in high-dimensional feature space. Partial fixes exhibit intermediate behavior: w-fix preserves performance, while g-fix degrades it due to compensatory growth in weight norms.

Summary. These results support three key conclusions: (i) Concept scores in standard CBMs are dominated by magnitude rather than semantic alignment, (ii) Feature and weight norms contribute independently and compensate for each other when constrained, and (iii) Fully removing magnitude eliminates saturation but severely degrades performance, motivating softer forms of norm control.

## C Concept-Only Reference Accuracy: Full Results

In this appendix, we provide additional details and robustness analyses for the concept-only reference accuracy reported in Section 4.

Experimental protocol. To estimate the task performance achievable from concepts alone, we derive a dataset of pairs $( c ^ { * } , y )$ from the CUB dataset by Koh et al. [15], where $c ^ { * }$ is the masked-sign concept vector constructed from the ground-truth instance-level annotations of image x, and y denotes the corresponding class label [26]. Specifically, concepts with annotator certainty below a threshold t are masked to zero; retained concepts take values in $\{ - 1 , + 1 \}$ according to the binary annotation. We vary the certainty threshold $t \in \left\{ 1 , 2 , 3 , 4 \right\}$ , producing multiple annotation regimes with different degrees of visual reliability. $\mathbf { A } \mathbf { t } \ t = 1$ no concept is masked, so the encoding is the plain binary annotation recoded from $\{ \bar { 0 } , 1 \}$ to $\{ - 1 , + 1 \}$ ; all three predictor families are invariant to this affine recoding. The reference accuracy of Section 4 is obtained at t = 1.

Model families. To ensure that the estimated reference accuracy is not limited by the capacity of the concept-to-label mapping, we evaluate multiple predictor classes: (i) Linear logistic regression, (ii) A two-layer MLP with 256 hidden units per layer, and (iii) Gradient-boosted decision trees (XGBoost). All models are selected based on validation performance and evaluated on the held-out test set.

Results. Table 6 reports the full results across all configurations. We make the following three observations: (i) Low concept-only accuracy. Across all encodings and predictors, the maximum achievable test accuracy remains below 40%, with a peak of 37.5%. This demonstrates that the annotated instance-level concepts alone do not explain the classification performance reported by many CBMs, motivating direct audits of what additional information the trained models use. (ii) Robustness across model classes. The reference accuracy is stable across linear, neural, and treebased predictors. While more expressive models (MLP, XGBoost) slightly outperform linear models, the improvement is marginal, indicating that the limitation arises from the concept representation itself rather than the classifier capacity. (iii) Effect of annotation filtering. Performance varies with the certainty threshold t. Moderate filtering $( t = 1 , 2 )$ yields the highest accuracy, while aggressive filtering $( t = 4 )$ significantly degrades performance. This reflects a trade-off between annotation noise and information loss: removing uncertain concepts improves reliability but reduces the amount of available signal.

Implications. These results establish the center of the concept-only reference band for conceptbased classification on CUB. Because the band has fuzzy boundaries (single annotator per image, subjectivity of certainty scores), it should be treated as a reference point rather than a strict bound. Models operating substantially above it may use legitimate graded visual evidence, non-local proxy cues, or internal information not exposed in scalar concept scores; distinguishing these cases is the purpose of the leakage probes and ablations in Section 6.

## D Spatial Faithfulness

For visual concepts, interface faithfulness also has a spatial component: a model may correctly predict a concept while relying on evidence from an unexpected region. To evaluate this, we measure spatial faithfulness, i.e., whether concept activations localize near the anatomical region associated with the concept.

Architecture and backprojection. The fully constrained configuration (LC-CBM) uses a BagNet-33 backbone, producing a spatial feature map $\mathbf { \bar { \boldsymbol { f } } } ( \boldsymbol { x } ) \in \mathbb { R } ^ { U \times V \times C }$ with stride $s = 8$ and receptive field size $r = 3 3$ . For location $( u , v )$ , the receptive field center in input coordinates is:

$$
\boldsymbol { c } _ { x } = \frac { \boldsymbol { r } - 1 } { 2 } + \boldsymbol { s } \cdot \boldsymbol { v } , \qquad \boldsymbol { c } _ { y } = \frac { \boldsymbol { r } - 1 } { 2 } + \boldsymbol { s } \cdot \boldsymbol { u } .
$$

For each concept c, we localize it via the peak activation:

$$
( u ^ { * } , v ^ { * } ) = \arg \operatorname* { m a x } _ { u , v } f _ { c } ( x ) _ { u , v } ,
$$

and map it to $( c _ { x } , c _ { y } )$

Setup. We compare LC-CBM trained with class-level (Koh) annotations and instance-level annotations. Concepts are mapped to CUB keypoints across 17 anatomical regions (see mapping below). For bilateral parts (e.g., wings), the nearest visible keypoint is used.

Evaluation filtering. A pair $( x , c )$ is evaluated only if: (i) the concept is present, (ii) annotator certainty $\geq 3 .$ (iii) the model predicts it as present, and (iv) the keypoint is visible. This yields ≈68k (Koh) and ≈77k (instance-level) valid pairs.

Coordinate normalization. Distances are computed in 224 × 224 model-input space and normalized by the bounding-box diagonal:

$$
d _ { \mathrm { n o r m } } = \frac { \| ( c _ { x } , c _ { y } ) - ( x _ { \mathrm { k p } } , y _ { \mathrm { k p } } ) \| _ { 2 } } { 2 2 4 } .
$$

PCK (Percentage of Correct Keypoints). We measure localization using

$$
\mathrm { P C K } ( \alpha ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \big [ d _ { \mathrm { n o r m } } ^ { ( i ) } < \alpha \big ] .
$$

We evaluate $\alpha \in [ 0 . 0 5 , 0 . 5 0 ]$ and report PCK@0.15. At $2 2 4 \times 2 2 4 , \alpha = 0 . 1 5$ corresponds to ≈34 pixels (about four BagNet strides), balancing robustness and spatial precision

Spatial discrimination. To assess fine-grained localization, we define a relative metric. For concept pairs $c _ { 1 } , c _ { 2 }$ sharing a descriptor but mapped to different parts:

$$
\mathrm { D i s c } ( c _ { 1 } , c _ { 2 } ) = \mathbb { E } \left[ \mathbf { 1 } \big ( d ( \mathrm { p e a k } _ { c _ { 1 } } , \mathrm { p a r t } _ { c _ { 1 } } ) < d ( \mathrm { p e a k } _ { c _ { 1 } } , \mathrm { p a r t } _ { c _ { 2 } } ) \big ) \right] .
$$

A score of 0.5 is chance. Values $> 0 . 5$ indicate that activations are closer to the mapped region even for visually similar concepts. This is stricter than PCK, as it requires discriminating between nearby parts.

Results. Table 7 shows that both models achieve $\mathrm { P C K @ 0 . 5 0 \approx 0 . 9 5 }$ , indicating on-object activations. However, fine localization differs: PCK@0.15 improves from 0.42 (class-level) to 0.48 (instance-level). Table 11 also shows the extracted anatomical groups. Refer to Appendix Section F.3 for more details.

Table 7: PCK across thresholds α for LC-CBM trained with class-level and instance-level annotations. Instance-level supervision consistently improves localization, particularly at stricter thresholds.
<table><tr><td rowspan=1 colspan=1>α</td><td rowspan=1 colspan=1>LC-CBM (class-level)</td><td rowspan=1 colspan=1>LC-CBM (instance-level)</td></tr><tr><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.113</td><td rowspan=1 colspan=1>0.130</td></tr><tr><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.284</td><td rowspan=1 colspan=1>0.326</td></tr><tr><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.422</td><td rowspan=1 colspan=1>0.476</td></tr><tr><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.541</td><td rowspan=3 colspan=1>0.5960.7770.841</td></tr><tr><td rowspan=1 colspan=1>0.30</td><td rowspan=1 colspan=1>0.748</td></tr><tr><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.829</td></tr><tr><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.883</td><td rowspan=3 colspan=1>0.8860.9210.946</td></tr><tr><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.919</td></tr><tr><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.946</td></tr></table>

Table 8: PCK@0.15 by anatomical region. Instance-level supervision reduces head bias and improves localization on body parts.
<table><tr><td>Region</td><td>Koh</td><td>Instance</td><td>∆</td></tr><tr><td>Head / Bill</td><td>0.80</td><td>0.59</td><td>-0.21</td></tr><tr><td>Body</td><td>0.18</td><td>0.46</td><td> $+ 0 . 2 8$ </td></tr><tr><td>Wing</td><td>0.21</td><td>0.39</td><td> $+ 0 . 1 8$ </td></tr><tr><td>Tail</td><td>0.01</td><td>0.08</td><td> $+ 0 . 0 7$ </td></tr><tr><td>Leg</td><td>0.03</td><td>0.04</td><td> $+ 0 . 0 1$ </td></tr></table>

Anatomical breakdown. Table 8 highlights a strong anatomical bias under class-level supervision: activations concentrate on head regions (0.80) while remaining low on body parts (0.18), indicating reliance on class-correlated features. Instance-level supervision substantially reduces this bias, improving localization on body and peripheral regions and yielding a more balanced spatial distribution. Instance-level supervision reduces this bias (body 0.46), lowering the head/body ratio from 4.4× to 1.3×. Tail and leg concepts remain challenging $( \mathrm { P C K } @ 0 . 1 5 < 0 . 0 9 )$ , likely due to small spatial extent, peripheral location, and grid resolution.

Spatial discrimination results.The class-level model operates near chance (0.519), while instancelevel supervision achieves 0.569, confirming improved discrimination between nearby anatomical regions.

Interpretation. These results show that high concept accuracy does not imply spatially faithful reasoning. Class-level supervision encourages shortcuts via global correlations (e.g., head features), whereas instance-level supervision encourages activations closer to the mapped anatomical regions.

Importantly, architectural locality alone is insufficient: BagNet constrains where information comes from, but not where the model attends. Spatial faithfulness therefore depends critically on the supervision signal.

Limitations. Evaluation is conditioned on correct predictions and does not capture false positives. Localization is also limited by the discrete grid (stride = 8). Finally, performance varies across regions due to dataset properties (occlusion, size, discriminativeness).

## E Codebook Fidelity

In this section, we evaluate whether model predictions can be explained as decisions over discrete concept representations. While the probe analysis in Section 6.1 tests whether predictions are reconstructible from continuous concept scores, we now assess whether decisions are consistent with a codebook-based reasoning mechanism operating on discretized concepts.

Method. We construct a class-level concept codebook $\{ k _ { c } \} _ { c = 1 } ^ { C }$ , where each class c is associated with a binary concept vector derived from the class-level CUB annotations. For each test image x we first discretize the predicted concept vector $d ( x )$ and binarize it via thresholding $( d _ { i } ( x ) > 0 )$ . We then assign a predicted class by selecting the nearest codebook entry under Hamming distance:

Table 9: Codebook-fidelity diagnostics.
<table><tr><td>Metric</td><td>LC-CBM</td><td>CEM</td></tr><tr><td>Mean Hamming distance</td><td>5.35</td><td>3.93</td></tr><tr><td>Nearest-code / native agreement</td><td>0.03</td><td>79.50</td></tr><tr><td>Nearest-code / true-class match</td><td>45.05</td><td>70.50</td></tr></table>

$$
{ \hat { y } } _ { \mathrm { c o d e } } ( x ) = \arg \operatorname* { m i n } _ { c } { \mathrm { ~ H a m } ( d ( x ) , k _ { c } ) } .
$$

Metrics. We report three diagnostics:

• Mean Hamming distance: average distance between $d ( x )$ and the nearest class code.

• Nearest-code / native agreement: fraction of test samples where the nearest codebook class matches the model's native prediction.

• Nearest-code / true-class match: classification accuracy obtained via nearest-code assignment.

Results. Table 9 reports the codebook-fidelity diagnostics. For the fully constrained configuration the nearest-code classification accuracy is 45.05%, closely matching both the continuous-concept probe accuracy (44.99%) and the discrete-concept probe accuracy (47.58%) on the checkpoint analyzed here (seed-averaged values: Table 3). This consistency indicates that predictions can be well approximated as nearest-code decisions in the discrete concept space, supporting the interpretation that the model routes decisions through the exposed concept representation.

The nearest-code / native agreement is close to zero for this configuration. This is expected and not informative: the native class head is not trained in the sequential setting and operates at chance level, so its predictions do not reflect the concept-based decision pathway.

For CEM, the nearest-code classification accuracy is substantially higher (70.50%), reflecting that its concept outputs are strongly class-informative. However, the native model still achieves higher accuracy than both the codebook-based prediction and the discrete-concept probe, resulting in a non-zero gap (Table 2). This indicates that, despite the informativeness of the concept representation, the final prediction is not fully determined by the exposed concept channel.

Interpretation. Taken together, these results provide a complementary view of leakage. While probe-based analysis reveals whether predictions are reconstructible from concept outputs, codebook fidelity evaluates whether predictions follow a discrete concept-level decision rule. The fully constrained configuration satisfies both criteria, whereas CEM exhibits residual performance beyond what can be explained by concept-based reasoning, consistent with the leakage gaps observed in the main text.

## F Common Implementation and Training Details

## F.1 Ablation Template and Fully Constrained Configuration

We describe the template in its fully constrained configuration; the ablated configurations change one component at a time (backbone: ResNet50 or BagNet-k; heads: one shared head instead of grouped per-concept projections; scoring: dot product instead of cosine similarity; training: joint instead of concept-only). The fully constrained configuration processes an input image through a five-stage pipeline:

• BagNet-33 backbone: produces spatial feature maps with a hard $3 3 \times 3 3$ pixel receptive field per position, yielding a tensor of shape $( B , C , \overleftarrow { H } , W )$

![](images/d4d45cf5f29173bf7c6232987732405975382a8f2548a8386b942e80ed444e26.jpg)  
Figure 6: Fully constrained configuration of the ablation template (LC-CBM). A BagNet-33 backbone extracts local features, projects them into concept-specific representations, compares them with learned concept vectors via cosine similarity, and aggregates them via global max pooling into concept scores. The other configurations of Table 3 replace the backbone, the heads, the scoring, or the training objective.

• Latent mapper: a stack of $1 \times 1$ convolutions projects backbone features into per-concept latent subspaces. The first stage is a dense projection; subsequent stages use grouped convolutions with n groups so that each concept operates on its own d-dimensional channel slice. Each spatial position is L2-normalized per concept, yielding latents of shape $( B , n , d , H , W )$ on the unit sphere.

• Prototype similarity maps: l prototype vectors per concept (each of dimension d) are compared via cosine similarity at every spatial location using grouped convolution, yielding similarity maps of shape $( B , \dot { n } , l , H , \dot { W } )$ with values in $[ - \bar { 1 } , \bar { 1 } ]$

• Global max pooling: spatial and prototype dimensions are collapsed by taking the maximum over all $( H \times W )$ locations and over all l prototypes per concept, producing one continuous concept score per concept: $s \in \mathbb { R } ^ { n }$

• (Optional) Linear classification head $\mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ elasses maps concept scores to class logits.

Discretization of s into $\{ 0 , 1 \} ^ { n }$ is applied post-hoc during evaluation and does not participate in gradient updates.

Training Procedure All main-text LC-CBM results use concept-only training: the linear classification head is frozen (never updated) and the backbone, latent mapper, and prototypes are optimized with Adam using only the concept loss,

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { c o n c e p t } } ,
$$

where $\mathcal { L } _ { \mathrm { c o n c e p t } }$ is the weighted binary cross-entropy of Eq. (1) (certainty weights for by-instance labels, unit weights for by-class labels). Model checkpointing uses best validation concept accuracy. Because the head is never trained under this protocol, its own predictions are at chance level; task accuracy for LC-CBM is instead obtained by the post-hoc linear probe of Section 6.1, fit on frozen concept scores exported from the checkpoint. This is also why $\Delta _ { \mathrm { b y p a s s } } = 0$ holds by construction for LC-CBM: the only path from concept scores to a class prediction is the probe itself.

The joint-training ablation reported in Table 3 (§6.2) instead optimizes all parameters end-to-end every epoch, including the class head, with

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { C E } } ( \hat { y } , y ) + \lambda \cdot \mathcal { L } _ { \mathrm { c o n c e p t } } ,
$$

where $\mathcal { L } _ { \mathrm { C E } }$ is cross-entropy on class logits. This variant is used only to test whether task supervision changes concept scores (§6.2); it is not the protocol behind the main LC-CBM rows elsewhere in the paper.

## Hyperparameters All reported LC-CBM results use the configuration in Table 10.

Weight decay is applied to backbone and mapper parameters only; the prototype weight tensor receives no decay.

Table 10: LC-CBM hyperparameters used for all reported experiments.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>epochs</td><td>150</td></tr><tr><td>learning_rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>weight_decay</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>batch_size</td><td>32</td></tr><tr><td>image_size</td><td>224 × 224</td></tr><tr><td>bbox_center_crop</td><td>True</td></tr><tr><td>latent_out_channels</td><td>128, 64, 32</td></tr><tr><td>bagnet_latent_out_channels</td><td>1024,256</td></tr><tr><td>n_prototypes_per_concept</td><td>1</td></tr><tr><td>optimizer</td><td>Adam</td></tr><tr><td>lr_scheduler</td><td>ReduceLROnPlateau (factor=0.05, patience=10, min  $\scriptstyle 1 \ r = 1 0 ^ { - 5 } )$ </td></tr></table>

Augmentation and Pre-processing: LC-CBM vs. CEM LC-CBM uses the CUB bounding box annotation (bbox\_center\_crop=True) to centre-crop each image around the annotated bird bounding box before resizing to 224 × 224. This preserves the bird's aspect ratio and ensures that patch-level features at every spatial position correspond to predictable, bird-relative image regions. Shape and size attributes, which are aspect-ratio-sensitive, are not distorted, and part-level attributes map to consistent spatial positions across images.

CEM (and standard CBM) uses a standard ImageNet-style preprocessing pipeline (resize + random crop) that does not account for the bounding box. This pipeline can significantly distort the bird's aspect ratio, shift the bird within the crop, and truncate body parts. Shape and size attributes are confounded by crop geometry, and spatial precision for all part-level attributes is reduced because the relationship between pixel position and bird anatomy is not consistent across images.

This preprocessing difference is a key experimental variable: LC-CBM's spatial faithfulness results (Appendix D) are only interpretable under bbox\_center\_crop=True.

## F.2 Baselines: CEM and Joint CBM

CEM [8] is trained using the authors' published implementation with a ResNet-18 feature extractor, embedding size 16, concept\_loss\_weight=1, and training\_intervention\_prob=0.25. The optimizer is SGD (lr=0.01, momentum=0.9, weight\_decay=4 × 10−6) with early stopping (patience=15) up to 150 epochs.

Joint CBM [15] is trained end-to-end with concept\_1oss\_weight=3 using the same ResNet-18 backbone and preprocessing pipeline as CEM.

Both baselines are trained with 3 independent seeds (42, 43, 44). For the by-class evaluation, all three seeds use Koh et al.'s class-level concept labels. For the by-instance evaluation, the same three seeds are retrained on per-image CUB instance annotations. Val/test splits follow the standard Koh et al. preprocessing (4,796 train 7 1,000 val / 5,794 test).

## F.3 Concept-to-CUB-Part Mapping

The Spatial faithfulness evaluation requires mapping each CUB concept to one or more anatomical keypoints. The canonical mapping covers 17 anatomical groups and is shown in Table 11 and stored in notebooks/concept\_to\_cub\_part\_mapping.json. Three concept families are excluded from the C5 evaluation: has\_size\_\*, has\_shape\_\*, and has\_primary\_color\_\*. These are global descriptors with no anatomical part anchor. For bilateral parts (eyes, legs, wings), the nearest visible keypoint is used as the evaluation target rather than the mean of left and right keypoints.

Table 11: Concept-to-CUB-part mapping used in the C5 spatial faithfulness evaluation. Concepts has\_size\_\*, has\_shape\_\*, and has\_primary\_color\_\* are excluded (no anatomical anchor).
<table><tr><td>Concept prefix</td><td>CUB part IDs</td><td>Bilateral?</td></tr><tr><td>bill_*</td><td>beak (2)</td><td>No</td></tr><tr><td>belly_*</td><td>belly (3)</td><td>No</td></tr><tr><td>breast_*</td><td>breast (4)</td><td>No</td></tr><tr><td>crown_*</td><td>crown (5)</td><td>No</td></tr><tr><td>forehead_*</td><td>forehead (6)</td><td>No</td></tr><tr><td>eye_color</td><td>left eye (7), right eye (11)</td><td>Yes</td></tr><tr><td>leg_*</td><td>left leg (8), right leg (12)</td><td>Yes</td></tr><tr><td>wing_*</td><td>left wing (9), right wing (13)</td><td>Yes</td></tr><tr><td>nape_*</td><td>nape (10)</td><td>No</td></tr><tr><td>tail_*</td><td>tail (14)</td><td>No</td></tr><tr><td>throat_*</td><td>throat (15)</td><td>No</td></tr><tr><td>back_*</td><td>back (1)</td><td>No</td></tr><tr><td>head_pattern</td><td>crown (5), forehead (6), nape (10)</td><td>No</td></tr><tr><td>underparts_*</td><td>belly (3), breast (4)</td><td>No</td></tr><tr><td>upperparts_*</td><td>back (1), crown (5)</td><td>No</td></tr><tr><td>under_tail_*</td><td>tail (14), belly (3)</td><td>No</td></tr><tr><td>upper_tail_*</td><td>tail (14), back (1)</td><td>No</td></tr></table>

## G Cross-Label Concept Accuracy

This appendix reports concept accuracy when models are evaluated against a label regime different from their training regime, establishing that Koh (class-wise) and instance-wise annotations are substantially consistent.

Setup. Koh-trained model scores are evaluated against instance-wise CUB ground truth; instancetrained model scores are evaluated against Koh class-level labels. Three model families (LC-CBM, CEM, CBM) are evaluated across three data splits each.
<table><tr><td>Family</td><td>Koh-trained → Instance GT</td><td>Instance-trained → Koh GT</td><td>Koh→ Koh (matched)</td><td>Instance → Instance (matched)</td></tr><tr><td>LC-CBM</td><td>0.800</td><td>0.777</td><td>0.904</td><td>0.748</td></tr><tr><td>CEM</td><td>0.812</td><td>0.841</td><td>0.949</td><td>0.798</td></tr><tr><td>CBM</td><td>0.811</td><td>0.841</td><td>0.946</td><td>0.800</td></tr></table>

Table 12: Cross-label concept accuracy (mean across 3 splits). Matched-label baselines are shown for reference only.

## Interpretation

• Label regimes are substantially consistent. A model trained exclusively on Koh labels reaches 80–84% accuracy when evaluated on instance-wise CUB GT, and vice versa. This confirms that both annotation protocols capture the same underlying bird attributes despite differing in granularity and derivation.

• LC-CBM generalizes less across label regimes than CEM and CBM (—1 to —6 pp). The concept representations learned by LC-CBM are more annotation-regime-dependent, particularly in the instance-trained → Koh GT direction (—6.4 pp). This is consistent with the instance-trained LC-CBM being a weaker concept predictor overall (matched accuracy 74.8% vs. 79.8–80.0% for CEM/CBM).

• For CEM and CBM, instance-trained → Koh GT (≈84%) outperforms Koh-trained → instance GT (≈81%) by ≈3 pp. Models trained on noisier instance-wise labels still recover the Koh-measurable concept signal at high accuracy, supporting instance-wise training as a more faithful annotation protocol without sacrificing Koh-measurable concept quality.

![](images/d4595c81e4d232ab1be982678bcc2da5bfeb535ce7a6a93e5d2f7fee226db058.jpg)

![](images/7bef737f4d24fc7913ccae29d482e33e05ace789aaaf31565ee00fd7218cc2ed.jpg)  
(a) Concept 14

Class-level (Koh)  
![](images/2a62bd4d868b6819b3b5262d271f277ff066d7841d8b918ccdf8ef93c54b0bb6.jpg)  
(b) Concept 1

![](images/a87f42acfbfd554afed81e79dda2a190b4bb7a0bd4a2149112c3c31a7c42cdde.jpg)  
(c) Concept 6

![](images/d376a1afa42fe06a6416b9c565b7ae621ee127a2610d06c23b1d3537986a3e98.jpg)  
(d) Concept 14

(e) Concept 1  
![](images/8ee55e3d0a3f1aa4ed20dcaa0e610f39dea920bacf77ee0254be920bd77f39b3.jpg)  
(f) Concept 6  
Figure 7: Top-5 highest-activation patches for selected concepts under class-level (top) and instancelevel (bottom) supervision. Class-level training shows a strong bias toward head regions, while instance-level supervision produces more spatially aligned activations.

This analysis confirms that the two label sets are not measuring incompatible things, and that results under different labeling regimes are directly comparable. The ≈18% cross-label disagreement rate bounds the expected concept accuracy of any faithfully grounded model, as discussed in Section 4.

## H Qualitative Visualisations

## H.1 Concept Representation

We provide qualitative evidence of concept grounding by visualizing top-activating patches for a small set of representative concepts. For each concept, we show the top-5 highest-activation patches, corresponding to the spatial locations that most strongly support the prediction.

Selection. Weselectconceptsthathighlightdifferentbehaviors: (i) has\_upperparts\_color::white (global color), (ii) has\_bill\_shape::hooked\_seabird (localized shape), and (iii) has\_wing\_color: :yellow (mid-level attribute). These examples illustrate both success and failure modes.

## Results. Figure 7 compares class-level (Koh) and instance-level supervision.

Under class-level supervision, activations are often concentrated on head regions, even for concepts unrelated to the head, indicating reliance on class-correlated cues. In contrast, instance-level supervision yields more spatially aligned activations, particularly for localized concepts such as bill shape and wing color.

Interpretation. These visualizations support the quantitative findings: leakage manifests as a reuse of shared discriminative regions, while instance-level supervision encourages concept-specific, spatially grounded representations.

## H.2 Overlay of Cub key point and BagNet RF

Figure 8 visualizes both the keypoint provided by the original CUB dataset, and the corresponding receptive field, illustrating how the model grounds concepts in local image evidence. Consistent with our spatial faithfulness analysis, instance-level supervision leads to better alignment between receptive fields and anatomical keypoints, whereas class-level supervision often relies on less localized, proxy regions.

Koh\_1 (class-wise Koh labels) | sample=89 | image\_id=205 one concept / visible part (seed 7) | plotted rows=3 (candidates before dedup=21, after dedup=12) (GT+, certainty>=3, pred+, keypoint-visible)  
![](images/d837c028ccb33b587060bf96112aab2ae9c4fc9432460220c737b2dbc0d8739e.jpg)  
(a) Koh (by class), idx 000089

run 0 pkl (instance-wise) | sample=89 | image id=205 one concept / visible part (seed 7) | plotted rows=3 (candidates before dedup=12, after dedup=9) (GT+, certainty>=3, pred+, keypoint-visible)  
![](images/a667b2d74883a03039b9fb8dcfc3dd6eaf0aa144b791026f939f4874a9787cb9.jpg)  
(b) Instance-wise, idx 000089

Koh\_1 (class-wise Koh labels) | sample=296 | image\_id=638 one concept / visible part (seed 7) | plotted rows=3 (candidates before dedup=20, after dedup=10) (GT+, certainty>=3, pred+, keypoint-visible)  
![](images/d9c953d32fc27ab44b3711740a8d89e63186406b31f6792080b876b0ba5d019a.jpg)  
(c) Koh (by class), idx 000296  
run 0 pkl (instance-wise) | sample=296 | image id=638 one concept / visible part (seed 7) | plotted rows=3 (candidates before dedup=22, after dedup=12) (GT+, certainty>=3, pred+, keypoint-visible)

![](images/36a3da7e48bc5466a04f278c6a3da141f9dea342d95da906f44e7c3a556d66bb.jpg)  
(d) Instance-wise, idx 000296  
Figure 8: Qualitative visualization of concept localization. For each sample, we show the keypoint (center) and the corresponding receptive field (bounding box) used by the network to infer the concept. We compare class-level (Koh) and instance-level supervision across two examples. Instance-level supervision yields activations that are more spatially aligned with the expected anatomical regions.