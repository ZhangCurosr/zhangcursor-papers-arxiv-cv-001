# MuViRL: Multimodal View-Invariant Visual Speech Representation Learning via Calibrated View Aggregation

Fei Su<sup>1,2</sup>, Kai Qiu<sup>3</sup> and Juan Liu<sup>2,1\*</sup>

<sup>1</sup>School of Computer Science, Wuhan University, Wuhan, China. <sup>2</sup>School of Artificial Intelligence, Wuhan University, Wuhan, China.

<sup>3</sup>Xiaomi EV Technology Co., Ltd., Beijing, China.

## Abstract

Multimodal representation learning combines complementary signals to capture shared semantic information. For dynamic scenes, these representations need to remain consistent across views while preserving distinctions between temporal states. In visual speech representation learning, view variation changes the visibility and reliability of articulation cues, making it dificult to learn such representations from synchronized audio and video. To address this challenge, we propose MuViRL, a self-supervised framework for multimodal view-invariant visual speech representation learning via calibrated view aggregation. The framework combines synchronized 3D Gaussian view synthesis with multimodal self-supervision to learn consistent representations from real and synthesized data. A multimodal decision model learns relative view utility from teacher prediction losses, and its calibrated scores guide the aggregation of complementary information. Aggregation-to-Single Distillation transfers the aggregated predictions to a single branch. Real-synthetic alignment matches synchronized windows across views while preserving their temporal relations. Experiments on real and synthesized multiview data demonstrate that MuViRL improves multimodal representation consistency and visual speech recognition performance across views.

Keywords: Multimodal representation learning, self-supervised learning, calibrated view aggregation, multimodal decision model, 3D Gaussian splatting

## 1 Introduction

Multimodal representation learning combines complementary observations to capture semantic information shared across modalities. Self-supervised Multi-Modal Versatile networks [1] and ImageBind [2] use naturally paired signals to learn transferable representations. In dynamic scenes, these representations must also preserve changes over time. Visual speech representation learning draws on synchronized sound and lip motion to capture evolving articulation. The same articulation can appear diferent across views, while distinct states may difer only in subtle lip movements. Learning useful representations requires separating variation caused by viewpoint from changes that distinguish articulation states.

Multiview learning provides mechanisms for both view invariance and the fusion of complementary information. MVContrast [3] and ViewCLR [4] learn consistency across object views and generated latent video views, respectively. For heterogeneous inputs, robust multiview learning [5] combines samplelevel attention with alignment under simulated perturbations. Decomposition and composition [6] jointly train fused and individual modality features for skeleton-based action learning. Applying these ideas to visual speech requires correspondence within an utterance. Two segments can share the same utterance identity yet depict diferent articulation states. Agreement between whole-sequence representations does not establish which segments correspond across views or explicitly constrain their temporal relations.

Audiovisual pretraining supplies temporal supervision within a recording. AV-HuBERT [7] predicts masked multimodal units, while RAVen [8] and BRAVEn [9] learn through prediction across modalities. CAV-MAE Sync [10] models fine-grained correspondence between sound and video. These approaches associate audio with visible events along an observed sequence. However, monocular datasets such as LRS3 [11] and VoxCeleb2 [12] lack synchronized visual observations of the same articulation from diferent viewpoints. Synchronization between audio and video does not directly supervise how that articulation should be represented from another view.

Synchronized view synthesis can supply these observations, but matching timestamps does not ensure that each view retains useful articulation cues. Occlusion can hide mouth structures, and reconstruction errors can distort their shape or motion. Giving such views equal influence can propagate these errors through aggregation and alignment. Research on prediction quality [13] and uncertainty [14] addresses unreliable inputs in multimodal fusion. For synthesized visual speech, image sharpness alone cannot determine whether a view preserves the articulation needed for prediction. A sharp rendering may still distort a lip contour, while a valid perspective change may lower pixel similarity. The challenge is to assess predictive usefulness at corresponding times and use it to guide learning across views while preserving temporal distinctions.

To address these challenges, we propose MuViRL, a self-supervised framework for multimodal view-invariant visual speech representation learning via

Calibrated View Aggregation. Region-Focused 3D Gaussian Splatting (3DGS) View Synthesis uses dynamic head reconstruction [15, 16] to generate synchronized observations, with fitting focused on articulatory structures. A frozen teacher derives unit targets from the original audio and video. We use these targets to supervise corresponding sequences and measure each view’s predictive usefulness.

Calibrated View Aggregation uses a multimodal decision model to assess synchronized lip grids under a language rubric for articulation. Teacher prediction losses provide grade and ranking supervision, and temperature calibration adjusts the resulting grade probabilities. Calibrated scores and geometric correspondence guide Adaptive Weighting, linking each view’s contribution to its usefulness for multimodal prediction.

Aggregation-to-Single Distillation (Agg2S) transfers aggregated predictions to a single branch that shares the encoder and unit targets. Temporal Real– Synthetic Representation Alignment (RSA) matches synchronized windows and their temporal relations. Reliability weighting reduces the influence of unreliable synthesized features during alignment. The encoder processes an individual video at inference without view aggregation or decision scoring.

The main contributions are as follows:

1) We propose MuViRL for multimodal view-invariant visual speech representation learning. Synchronized 3DGS views and shared audiovisual targets extend self-supervision across viewpoints while retaining temporal correspondence.

2) We introduce Calibrated View Aggregation to learn view utility from a frozen teacher’s prediction losses. Calibrated decision scores, geometric correspondence, and Adaptive Weighting jointly determine each view’s contribution.

3) We combine Agg2S with Temporal RSA to transfer aggregated predictions to individual views and align synchronized representations and their temporal relations. The learned encoder operates on a single video at inference.

4) We evaluate MuViRL on real and synthesized multiview data through representation analyses and recognition experiments. The results demonstrate improved correspondence across viewpoints and modalities while preserving discriminative information.

## 2 Related Work

## 2.1 Multimodal Representation Learning

Multimodal representation learning uses correspondence between inputs to learn shared content and retain complementary information. Shared embeddings in MultiModal Versatile networks [1] and ImageBind [2] connect modalities through paired observations. This principle supports multilingual pretraining [17] and image–text retrieval [18], and supplies semantic correspondences for object navigation [19]. Reconstruction and fusion complement semantic matching by retaining cues that can help recover or interpret another modality. MultiMAE [20] learns to reconstruct masked inputs across modalities, whereas attention bottlenecks [21] restrict feature exchange during fusion. Decomposition and composition [6] supervise both fused and individual representations, linking the information retained by each for action learning.

For audiovisual sequences, synchronization provides correspondence at the level of temporal states. AV-HuBERT [7] uses shared unit targets, while RAVen [8] and BRAVEn [9] learn through feature prediction. CAV-MAE Sync [10] and AlignVSR [22] explicitly align local audio and visual information. Temporal attention [23] further models their dependencies. Synthetic supervision in SynthVSR [24] expands recognition training data through speech-driven lip animation. Although these methods exploit the relationship between sound and visible articulation, pairing modalities within a recording does not supply observations of the same articulation from diferent viewpoints. MuViRL extends this supervision to synchronized original and synthesized views, using shared audiovisual targets to preserve articulation distinctions across views.

## 2.2 Self-Supervised Multiview Learning

The choice of paired views determines which variations a self-supervised objective suppresses. Contrastive Multiview Coding [25] extracts shared information from sensory channels. For physical viewpoint variation, MVContrast [3] learns associations within views, across views, and between aggregated shape representations. ViewCLR [4] generates latent viewpoints for video learning. Robust multiview learning [5] targets imperfect heterogeneous inputs, combining sample attention with alignment under simulated noise and missing information. Beyond matching paired views, aggregation determines how their complementary information is retained. Deep Sets [26] formalizes invariance to input order, while Set Transformer [27] models interactions among unordered inputs.

For dynamic sequences, aggregation must also preserve the information needed to distinguish temporal states. Supervised multiview lipreading combines synchronized streams [28] or shares information through multitask learning [29]. Temporal prediction captures dependencies within a sequence. Model compression [30] transfers ensemble predictions to a single model, while similarity-preserving distillation [31] transfers relations among representations. However, agreement between whole sequences does not specify which articulation states should match across views. MuViRL combines shared target prediction and Agg2S with Temporal RSA to align synchronized windows and their relations. This preserves temporal correspondence during multiview learning while supporting inference from a single video.

## 2.3 Calibrated Multimodal Decisions

Reliability estimation guides fusion when inputs difer in quality. QMF [13] relates dynamic fusion weights to prediction losses, while EAU [14] uses aleatoric uncertainty to stabilize unimodal features before fusion. For ocular recognition, image quality estimation [32] weights fusion using embedding robustness. Geometric alignment also matters for fusion: cooperative segmentation of LiDAR and image inputs [33] preserves semantic and geometric correspondence during feature interaction. For synthesized speech views, reliability assessment must distinguish reconstruction defects from valid appearance changes across viewpoints.

Language-guided assessment specifies which visual structures should be preserved. MLLM-as-a-Judge [34] examines scoring and ranking, while Q-Align [35] and DeQA-Score [36] express image quality through ordered levels and score distributions. SmolVLM [37] provides compact image–language encoding. Pairwise logistic ranking [38] learns relative order, whereas temperature scaling [39] and Dirichlet calibration [40] adjust probabilities. For synthesized speech views, general quality scores may miss articulation cues needed for multimodal prediction. We therefore use a frozen teacher’s prediction losses to supervise utility grades and rankings, then calibrate the grade probabilities. Calibrated View Aggregation combines these scores with geometric correspondence and learned feature weights.

## 2.4 Dynamic 3D Gaussian View Synthesis

View synthesis provides additional observations when recordings cover few head orientations. Pose augmentation for lipreading [41] uses a fitted 3D morphable model to generate training views for supervised recognition. Dynamic avatars model facial motion and appearance: FLAME [42] parameterizes facial shape and expression, and Neural Head Avatars [43] combines deformable geometry with learned appearance. With eficient rendering provided by 3D Gaussian Splatting [44], mesh binding in GaussianAvatars [15] and separate face and mouth deformations in TalkingGaussian [45] preserve facial structure during animation. STAvatar [16] further improves motion and occluded detail through soft binding and temporal density control.

Geometric consistency also determines whether synthesized observations can be aligned across views. Two-dimensional Gaussian splatting [46] emphasizes surface accuracy, and VCR-GauS [47] couples depth and normal regularization. DepthSplat [48] connects depth estimation with Gaussian rendering. Robust cross-view consistency [49] and video depth adaptation [50] further constrain geometry across observations. Reconstruction fidelity, however, does not establish whether rendered articulation remains informative for audiovisual prediction. MuViRL uses region-focused fitting and rendering along the original timeline to preserve local articulation and audio correspondence. The synchronized views supervise multimodal representations, and their rendered geometry supplies reliability evidence for aggregation and alignment.

![](images/59de0789404e1503cf71a95abfd38b05401386e46c59a74a2ed7d3d8fbc5456f.jpg)  
Fig. 1 Overview of the proposed framework. (a) Target extraction uses a frozen audiovisual teacher to generate shared unit targets Y. (b) Multimodal self-supervised pretraining uses Calibrated View Aggregation to combine visual features according to reliability ${ { q } _ { i , t } }$ and learned coeficients $s _ { i , t }$ . Aggregation-to-Single Distillation transfers aggregated predictions to the single branch, while MMP and Temporal RSA provide target supervision and temporal alignment. (c) Fine-tuning optimizes the pretrained encoder and a sequence decoder with transcript supervision.

## 3 Method

We consider multimodal representation learning from audiovisual videos. Each video provides an original lip image sequence $X _ { 0 }$ and a synchronized audio signal $X _ { a }$ . The objective is to learn representations that remain consistent across views while distinguishing diferent articulation states.

An overview of our framework is shown in Fig. 1. Synchronized 3DGS synthesis provides corresponding views for multimodal masked prediction (MMP). Calibrated View Aggregation integrates features from these views. Aggregation-to-Single Distillation (Agg2S) transfers aggregated predictions to a single branch, while Temporal Real–Synthetic Representation Alignment (RSA) preserves synchronized states and their temporal relations.

## 3.1 Region-Focused 3DGS View Synthesis

We generate synchronized view sequences to expand viewpoint coverage while preserving lip structure and temporal correspondence. As illustrated in Fig. 2, head rotations guide clip selection to expose complementary surfaces [43, 51]. Foreground masks [52] and landmarks [53] support tracking with VHAP<sup>1</sup> and FLAME [42]. The recovered pose and expression remain fixed during appearance fitting.

![](images/916de948b1f46e9c02931c5c131812af339549aa5593ee0fb1633a08b2ddd0ac.jpg)  
Fig. 2 Region-Focused 3DGS View Synthesis pipeline. (a) Pose-guided clip selection identifies clips with complementary yaw and pitch coverage from head pose trajectories. (b) Region-focused 3DGS fitting uses original motion and restored appearance for reconstruction, with errors averaged over the mouth box, lip rim, and inner mouth regions. (c) Synchronized view synthesis renders the same facial state through virtual cameras at each timestamp. Frame correspondence, lip motion, and rendering artifacts are then checked.

After motion tracking, we restore the appearance targets using Code-Former [54] while preserving the original frame indices. With the mesh binding strategy of STAvatar [16], a U-Net [55] predicts dynamic Gaussian attributes from appearance, UV positions, and frame displacements. Gaussian projection and compositing [44] produce the renderings. The regional constraint $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e g i o n } } = \sum _ { r } \lambda _ { r } \operatorname { M A E } ( \overleftarrow { I } _ { \mathrm { r e n d e r } } [ \Omega _ { r } ] , I _ { \mathrm { t a r g e t } } [ \Omega _ { r } ] ) } \end{array}$ emphasizes the mouth box, lip rim, and inner mouth. Here, $\Omega _ { r }$ selects region $r , \lambda _ { r }$ weights its error, and MAE averages over its pixels and color channels. This constraint complements image reconstruction, SSIM [56], LPIPS [57], and Gaussian regularization during avatar fitting.

Using the fitted avatar, we render each facial state from multiple virtual viewpoints at its original timestamp, retaining frame order and correspondence with the audio. Frame counts, lip motion, and rendering artifacts are checked before constructing training pairs. The resulting lip sequences are denoted by $X _ { i } = \{ X _ { i , t } \} _ { t = 1 } ^ { T } , i = 1 , \ldots , V$ , where V is the number of synthesized views and t indexes the $T$ frames. View index zero denotes the original sequence. Rendered geometry supports reliability assessment. Section S1.1 of the supplementary material (Online Resource 1) provides fitting and selection details, and Fig. S1 illustrates the synthesized views and mouth reconstructions.

![](images/25cb4dce0b7e75d2a31b264f0c5b78e8c645bc185602dbf4c3260bcac4950b91.jpg)  
Fig. 3 Calibrated View Aggregation. (a) Decision model adaptation uses synchronized lip grids and a language rubric to predict articulation grades, supervised by teacher loss ordering and ordinal grades. (b) Weighted aggregation converts grade logits $\ell _ { k }$ into calibrated utility $u _ { i , b } .$ , then combines it with depth correspondence to obtain reliability $q _ { i , t } .$ . Here, $u _ { i , b }$ abbreviates $u _ { i , b } ( \tau _ { \mathrm { c a l } } )$ , and $\mathrm { E } [ k ] / 3$ denotes the normalized expected grade. Reliability and learned coeficients $^ { s _ { i , t } }$ determine weights $\alpha _ { i , t }$ for aggregating view features into $H _ { \mathrm { a g g } }$

## 3.2 Calibrated View Aggregation

To integrate complementary visual information for multimodal target prediction, we aggregate synchronized view features according to calibrated reliability. A multimodal decision model uses a language rubric to assess articulatory structures and motion in lip images afected by blur, occlusion, or rendering artifacts. Predictive utility supervises these assessments during adaptation. Temperature calibration adjusts the grade probabilities used to calculate reliability. As illustrated in Fig. 3, calibrated reliability and Adaptive Weighting jointly determine each view’s contribution. A shared visual frontend $F _ { v }$ produces feature sequence $H _ { i }$ , with frame features $\mathbf { h } _ { i , t }$ , for aggregation.

Multimodal decision model adaptation. For each temporal segment b, paired original and candidate RGB grids contain frames at matching timestamps. The decision model $Q _ { \eta }$ uses the Laya Vision<sup>2</sup> readout with SmolVLM [37] to jointly encode the grids, view angles, a language rubric, and four ordered options. The rubric specifies lip contours, opening changes, and visible teeth, allowing normal viewpoint variation and occlusion. Following $\mathrm { Q - }$ Align [35], the four grades $k \in \{ 0 , 1 , 2 , 3 \}$ range from insuficient to clearly preserved articulation. The model returns one logit $\ell _ { k }$ per grade in a single forward pass. Applying softmax at temperature $\tau$ gives grade probabilities. Their expected grade, divided by three, defines the predictive utility score $u _ { i , b } ( \tau ) \in [ 0 , 1 ] \ [ 3 5 ]$ . A temperature of one gives the uncalibrated distribution, and $\tau _ { \mathrm { c a l } }$ gives the calibrated distribution.

Adaptation uses shared unit targets $Y = \{ y _ { t } \} _ { t = 1 } ^ { T }$ generated by the frozen audiovisual teacher from the original pair $( X _ { 0 } , X _ { a } )$ . Its visual prediction path evaluates each candidate at the same masked positions, producing segment loss $e _ { i , b }$ . Lower loss indicates greater predictive utility [13]. For $e _ { i , b } ~ < ~ e _ { j , b } .$ pairwise logistic ranking [38] encourages $u _ { i , b } ( 1 ) \ > \ u _ { j , b } ( 1 )$ by minimizing log $( 1 + \exp ( ( u _ { j , b } ( 1 ) - u _ { i , b } ( 1 ) ) / \tau _ { \mathrm { r a n k } } ) )$ , where $\tau _ { \mathrm { r a n k } } > 0$ controls score diferences. Averaging the pairwise losses with weights derived from teacher loss diferences gives $\mathcal { L } _ { \mathrm { r a n k } }$ . Candidate ranks define relative utility grades on the ordered readout scale within each segment, supplying soft targets for ordinal cross-entropy $\mathcal { L } _ { \mathrm { o r d } }$ . After ordinal initialization, the readout and score head parameters $\eta$ are updated with $\mathcal { L } _ { \mathrm { d e c } } = \mathcal { L } _ { \mathrm { r a n k } } + \lambda _ { \mathrm { o r d } } \mathcal { L } _ { \mathrm { o r d } }$ . Temperature scaling [39] on independent calibration utterances determines $\tau _ { \mathrm { c a l } }$ . We then freeze the decision model and cache its scores for student training. Supplementary Section S2 details the prompts, grade targets, and calibration.

Depth reprojection. Rendered geometry supplies complementary evidence for correspondence [47, 49]. Mouth pixels are backprojected and transformed between views to compare depth and surface normals. Valid support, coverage, and residual errors determine a coeficient $g _ { i , b } \in [ 0 , 1 ]$ using training statistics for each angle. The coeficient is one for geometrically consistent pairs and decreases as discrepancies increase. Views without rendered geometry also use one. Supplementary Section S1.3 specifies the construction. Segment reliability is

$$
q _ { i , b } = q _ { \operatorname* { m i n } } + ( 1 - q _ { \operatorname* { m i n } } ) u _ { i , b } ( \tau _ { \mathrm { c a l } } ) g _ { i , b } .\tag{1}
$$

Here, $q _ { \mathrm { m i n } } \in ( 0 , 1 )$ sets a positive floor. Segment values map to frame weights ${ q } _ { i , t }$ by timestamp, with $q _ { 0 , t } = 1$ for the original view. Utility estimates the candidate’s relative teacher grade. Geometry reduces the contribution of views with inconsistent correspondence.

Weighted aggregation. Adaptive Weighting learns each view’s contribution from the selected group [3, 5]. Let $s$ denote the sampled aggregation subset and $S _ { t }$ its valid views at time t. A temporal convolution produces local descriptors. Each descriptor is concatenated with its diference from the group mean, then processed by a linear mapping, ReLU, and a two-layer MLP with a sigmoid output to obtain $s _ { i , t } > 0$ . The final weights and aggregated features are

$$
\alpha _ { i , t } = \frac { s _ { i , t } \sqrt { q _ { i , t } } } { \sum _ { j \in S _ { t } } s _ { j , t } \sqrt { q _ { j , t } } } , \qquad \mathbf { h } _ { \mathrm { a g g } , t } = \sum _ { i \in S _ { t } } \alpha _ { i , t } \mathbf { h } _ { i , t } .\tag{2}
$$

The square root moderates reliability ratios. Temperature scaling changes the contrast among candidate scores and their weights relative to the original view when it is included. Cached ${ { q } _ { i , t } }$ remains fixed, whereas student gradients update Adaptive Weighting through $s _ { i , t }$ . Aggregation retains the timestamp and feature dimension, yielding sequence $H _ { \mathrm { a g g } }$ . Jointly reordering views and their scores leaves the result unchanged [26].

## 3.3 Aggregation-to-Single Distillation (Agg2S)

Agg2S transfers complementary information learned during multiview training to representations extracted from an individual video. The aggregation and single branches share the student encoder and predict the same unit targets. A separately sampled synthesized view $j ,$ , which may also belong to the aggregation subset, supplies features $H _ { j }$ to the single branch.

The audio frontend $F _ { a }$ maps $X _ { a }$ to synchronized features $H _ { a }$ . The aggregation branch combines $H _ { a }$ and $H _ { \mathrm { a g g } }$ through concatenation, normalization, projection, masking, and temporal encoding. Audio dropout sets $H _ { a }$ to zero on sampled updates. The single branch uses $H _ { j }$ with audio fixed to zero. With student parameters $\theta _ { ; }$ the shared prediction path $F _ { \theta }$ produces $p _ { \mathrm { a g g } , t } ( \tau )$ and $p _ { \mathrm { s i n g l e } , t } ( \tau )$ by applying softmax to logits divided by temperature $\tau$ . Both branches predict targets at the same valid masked positions $\mathcal { M }$

Agg2S matches their predictive distributions using Kullback–Leibler (KL) divergence [58]:

$$
\mathcal { L } _ { \mathrm { A g g 2 S } } = \frac { \tau _ { \mathrm { d } } ^ { 2 } } { N _ { \mathrm { m } } } \sum _ { t \in \mathcal { M } } \mathrm { K L } ( p _ { \mathrm { a g g } , t } ( \tau _ { \mathrm { d } } ) \| p _ { \mathrm { s i n g l e } , t } ( \tau _ { \mathrm { d } } ) ) .\tag{3}
$$

Here, $\tau _ { \mathrm { d } } > 0$ is the distillation temperature and $N _ { \mathrm { m } }$ counts shared valid masked positions. The temperature squared compensates for gradient rescaling. The aggregation distribution is held fixed in this loss, so gradients update the single prediction path through the shared parameters. In the same iteration, the aggregation branch learns from target prediction, while the single branch learns from both unit targets and predictions based on complementary views. At inference, the shared encoder extracts representations from an individual video without view aggregation or decision scoring.

## 3.4 Reliability-Aware Real–Synthetic Alignment

Alignment encourages consistent representations of the same articulation across original and synthesized views while preserving distinctions between successive states. Temporal RSA aligns synchronized windows and their relations, combining correspondence across views [3, 5] with similarity supervision [31]. Calibrated reliability weights both constraints to reduce the influence of unreliable synthesized frames.

Synchronized window alignment. To align articulation at corresponding times, we encode visual sequences without feature masking to obtain $Z _ { 0 }$ and $Z _ { i }$ . Pooling jointly valid frames within window $w$ and applying L2 normalization gives original vector ${ \bf z } _ { \mathrm { r e f } , i , w }$ and candidate vector ${ \bf z } _ { i , w }$ . The original vector is pooled over the same timestamps as candidate i and detached along the alignment path. Weight $a _ { i , w }$ averages cached ${ { q } _ { i , t } }$ over these frames. Invalid windows are excluded, and uniform alignment uses $a _ { i , w } = 1$ . The weighted cosine distance between corresponding windows is

$$
\mathcal { L } _ { \mathrm { s y n c } } = \frac { \sum _ { i , w } a _ { i , w } \left[ 1 - \mathbf { z } _ { \mathrm { r e f } , i , w } ^ { \top } \mathbf { z } _ { i , w } \right] } { \sum _ { i , w } a _ { i , w } } .\tag{4}
$$

The inner product measures similarity between normalized vectors. Normalization by the total weight preserves the loss scale across valid windows. Gradients update the synthesized representation to align corresponding states.

Temporal relation alignment. To preserve distinctions between states, we also match the similarities between windows within each sequence:

$$
\mathcal { L } _ { \mathrm { r e l a t i o n } } = \frac { \displaystyle \sum _ { i , w , w ^ { \prime } } a _ { i , w } a _ { i , w ^ { \prime } } \left[ \mathbf { z } _ { \mathrm { r e f } , i , w } ^ { \top } \mathbf { z } _ { \mathrm { r e f } , i , w ^ { \prime } } - \mathbf { z } _ { i , w } ^ { \top } \mathbf { z } _ { i , w ^ { \prime } } \right] ^ { 2 } } { \displaystyle \sum _ { i , w , w ^ { \prime } } a _ { i , w } a _ { i , w ^ { \prime } } } .\tag{5}
$$

Here, $w , w ^ { \prime }$ identify windows of the same utterance. The squared diference compares temporal relations in the original and synthesized sequences, while $a _ { i , w } a _ { i , w ^ { \prime } }$ weights both endpoints. Sums include valid pairs across the minibatch. The combined objective is

$$
\mathcal { L } _ { \mathrm { R S A } } = \beta _ { \mathrm { s y n c } } \mathcal { L } _ { \mathrm { s y n c } } + \beta _ { \mathrm { r e l } } \mathcal { L } _ { \mathrm { r e l a t i o n } } .\tag{6}
$$

Nonnegative $\beta _ { \mathrm { s y n c } }$ and $\beta _ { \mathrm { r e l } }$ balance the two reliability weighted constraints on synchronized windows and temporal relations.

## 3.5 Real–Synthetic Self-Supervised Pretraining

Target extraction. The frozen audiovisual teacher fuses each original audio and lip sequence into $H _ { \mathrm { a v } }$ , encodes it as $Z _ { \mathrm { a v } }$ , and selects the most probable unit at each timestamp. The resulting targets $Y$ also supervise synthesized sequences through their shared frame indices.

Multimodal masked prediction. At shared positions M, projected features are replaced by features from other valid timestamps. The aggregation branch uses audiovisual context with audio dropout, and the single branch uses visual context [7]. Their target cross-entropies form

$$
\mathcal { L } _ { \mathrm { M M P } } = \frac { \mathcal { L } _ { \mathrm { a g g } } + \lambda _ { \mathrm { s i n g l e } } \mathcal { L } _ { \mathrm { s i n g l e } } } { 1 + \lambda _ { \mathrm { s i n g l e } } } .\tag{7}
$$

Here, $\lambda _ { \mathrm { s i n g l e } } ~ \ge ~ 0$ controls the single branch contribution. Normalization preserves the combined loss scale.

Overall objective. Prediction, distillation, and alignment jointly optimize the student:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { M M P } } + \lambda _ { \mathrm { A g g 2 S } } \mathcal { L } _ { \mathrm { A g g 2 S } } + \lambda _ { \mathrm { R S A } } \mathcal { L } _ { \mathrm { R S A } } .\tag{8}
$$

Algorithm 1 Self-supervised optimization of MuViRL   
Decision model adaptation   
Input: $\mathcal { D } _ { \mathrm { t r } } , \mathcal { D } _ { \mathrm { c a l } }$ and frozen teacher.   
Output: Adapted $Q _ { \eta }$ and $\tau _ { \mathrm { c a l } }$   
Initialize: $\eta  \eta _ { 0 } .$   
Extract $Y$ and candidate losses $\{ \boldsymbol { e } _ { i , b } \}$   
Form pairs with $e _ { i , b } < e _ { j , b } .$   
Construct ordinal targets.   
Initialize $Q _ { \eta }$ with $\mathcal { L } _ { \mathrm { o r d } } .$   
for $n _ { \mathrm { d e c } } = 1 , \ldots , N _ { \mathrm { d e c } }$ do   
Sample candidate groups from ${ \mathcal { D } } _ { \mathrm { t r } }$   
$\begin{array} { r } { \mathcal { L } _ { \mathrm { d e c } }  \mathcal { L } _ { \mathrm { r a n k } } + \lambda _ { \mathrm { o r d } } \mathcal { L } _ { \mathrm { o r d } } . } \end{array}$   
Update η using $\nabla _ { \eta } \mathcal { L } _ { \mathrm { d e c } } .$   
end for   
Fit $\tau _ { \mathrm { c a l } }$ on $\mathcal { D } _ { \mathrm { c a l } }$ and freeze $Q _ { \eta }$   
Multimodal self-supervised pretraining   
Input: $\{ X _ { i } \} , X _ { a } , Y$ , geometry, and $Q _ { \eta } .$   
Output: Pretrained student parameters $\theta .$   
Initialize: $\theta  \theta _ { 0 }$ . Freeze $F _ { v } , F _ { a }$   
Cache $\{ q _ { i , t } \}$ using Eq. (1).   
for $n _ { \mathrm { t r a i n } } = 1 , \dots , N _ { \mathrm { t r a i n } }$ do   
Sample a minibatch and views $s , j .$   
Sample shared mask $\mathcal { M } .$   
for $i \in \mathcal { S } \cup \{ 0 , j \}$ do   
$H _ { i } \gets F _ { v } ( X _ { i } )$   
end for   
$H _ { a } \gets F _ { a } ( X _ { a } )$ . Apply audio dropout.   
$\{ \alpha _ { i , t } \} , H _ { \mathrm { a g g } } \gets \mathrm { E q . ~ ( 2 ) }$   
$p _ { \mathrm { a g g } }  F _ { \theta } ( H _ { \mathrm { a g g } } , H _ { a } ; \mathcal { M } )$   
$p _ { \mathrm { s i n g l e } }  F _ { \theta } ( H _ { j } , 0 ; \mathcal { M } )$   
Encode $Z _ { 0 } , \{ Z _ { i } \}$ without masking.   
Evaluate $\mathcal { L } _ { \mathrm { M M P } } , \mathcal { L } _ { \mathrm { A g g 2 S } } , \mathcal { L } _ { \mathrm { R S A } }$   
Update $\theta$ using $\nabla _ { \theta } \mathcal { L } \ [ \mathrm { E q . \ ( 8 ) } ]$   
end for

The coeficients $\lambda _ { \mathrm { A g g 2 S } }$ and $\lambda _ { \mathrm { R S A } }$ balance distillation and alignment.

For recognition, a sequence decoder maps $Z _ { i }$ to the predicted transcript $\widehat { \tau }$ and is fine-tuned with the encoder using transcript supervision.

Optimization. Algorithm 1 separates decision adaptation from student updates. Training and calibration utterances are denoted by ${ \mathcal { D } } _ { \mathrm { t r } }$ and $\mathcal { D } _ { \mathrm { c a l } }$ Student parameters $\theta$ comprise the updated encoder, projection, and aggregation parameters. The initial parameters are $\eta _ { 0 } , \theta _ { 0 }$ , and $ { N _ { \mathrm { d e c } } } ,  { N _ { \mathrm { t r a i n } } }$ specify the respective numbers of updates.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate our proposed framework on LRS3 [11], VoxCeleb2 [12], TCD-TIMIT [59], and OuluVS2 [60]. LRS3 provides 433 h of transcribed English speech. Following the AV-HuBERT [7] protocol, we combine pretrain and trainval for training, reserve approximately 1 h from trainval for validation, and use the oficial test set. We assess label eficiency using the 30 h trainval subset. LRS3 and 1,326 h of English VoxCeleb2 [12] data form a 1,759 h collection used for large-scale pretraining. TCD-TIMIT contains 6,913 sentences from 62 speakers, recorded from frontal and 30<sup>◦</sup> viewpoints. OuluVS2 provides synchronized video at five fixed viewpoints $( 0 ^ { \circ } , 3 0 ^ { \circ } , 4 5 ^ { \circ } , 6 0 ^ { \circ } , 9 0 ^ { \circ } )$ . Both datasets support evaluation of robustness to view variation on real multiview data. ORL [61], Yale [62], Notting Hill [63], and MEAD [64] provide quality grade supervision for decision model fine-tuning. The first three provide static face crops with variations in pose, expression, illumination, and occlusion, while MEAD provides synchronized multiview facial sequences.

Multiview data synthesis. We fit dynamic 3DGS avatars [16] with 1.5 million Gaussians for 300 epochs, followed by 30 refinement epochs. Original mouth pixels are retained in the reconstruction targets. Each selected original clip yields twelve synchronized sequences, with horizontal ofsets at $5 ^ { \circ }$ intervals within $[ - 2 5 ^ { \circ } , 2 5 ^ { \circ } ]$ and pitch ofsets within $[ - 1 0 ^ { \circ } , 1 5 ^ { \circ } ]$ . View-Augmented (VA) training draws 60% of base samples from original clips and 40% from synthesized clips. VA-30h, VA-433h, and VA-1759h denote the respective original data collections. Original and synthesized views of an utterance remain in the same dataset partition. Supplementary Section S1 provides synthesis and sampling details, and Fig. S1 shows synthesized inputs.

Metrics. We use accuracy, recall, and cosine similarity to evaluate representation learning and aggregation. Visual speech recognition (VSR) uses word error rate (WER), $\mathrm { W E R } = ( S + D + I ) / N _ { \mathrm { r e f } } \times 1 0 0 \%$ , where S, D, and I count substitutions, deletions, and insertions, and $N _ { \mathrm { r e f } }$ counts reference words.

## 4.2 Implementation Details

Backbone and initialization. We initialize from the 5th-iteration AV-HuBERT model [7], which provides cluster pseudo-labels for the 6th iteration. Base and Large use 12 layers of dimension 768 and 24 layers of dimension 1,024, respectively, with a ResNet-18 [65] visual frontend. Preprocessing converts 96 × 96 lip crops to grayscale 88 × 88 inputs. Audio preprocessing follows the AV-HuBERT configuration. Experiments are conducted on NVIDIA GeForce RTX 4090 GPUs.

Decision inputs and prompts. We use the Laya Vision 201M checkpoint. We divide each clip into three chronological segments and sample up to six evenly spaced frames per segment. Synchronized original and candidate frames are arranged in separate 2×3 RGB grids. The text specifies frame order, view angles, the assessment criterion, and four ordered grades. The predictive utility instruction includes “Judge how useful the second view is for recovering the articulation shown in the reference: changes in lip contour, opening and visible teeth.” The rubric accepts normal viewpoint variation and occlusion. The structural variant assesses preservation under blur, occlusion, and rendering defects. The model returns four logits for either prompt in one forward pass. Grades range from zero to three, and temperature scaling followed by the normalized expected grade produces the utility score. Supplementary Section S2.1 provides the complete prompts.

Decision adaptation and calibration. Vision and language backbones remain frozen while approximately 8.31M readout and score head parameters are updated. Ordinal initialization uses 300 steps at $1 0 ^ { - 4 }$ with batch size eight, followed by $N _ { \mathrm { d e c } } = 3 0 0$ ranking steps at $3 \times \bar { 1 } 0 ^ { - 5 }$ using twelve synchronized candidates. We set $\lambda _ { \mathrm { o r d } } = 0 . 1$ and $\tau _ { \mathrm { r a n k } } = 0 . 2 5$ , cap normalized pair weights at three, and apply label smoothing of 0.05. For the quality grade experiment, MEAD and the auxiliary face datasets supply original inputs and three levels of blur or occlusion. We fine-tune the readout for 1,500 updates at $3 \times 1 0 ^ { - 5 }$ with four ordered examples per update. Calibration and evaluation use separate utterances or identities, and pixelation is reserved for evaluation. With the decision model frozen, we fit $\tau _ { \mathrm { c a l } }$ by minimizing grade negative log-likelihood and cache scores. Reliability uses $q _ { \mathrm { m i n } } = 0 . 2 5$ and geometric coeficient one when rendered geometry is unavailable. Supplementary Section S2 gives label construction and partition details.

Self-supervised pretraining and fine-tuning. MuViRL uses the full objective in Eq. 8, with teacher grade and ranking supervision followed by temperature calibration. Reliability weights guide both view aggregation and Temporal RSA. The MuViRL configurations in Table 1 use Large encoders pretrained on VA-433h or VA-1759h and fine-tuned on 433 h of labeled data. Optimization and recognition decoding follow AV-HuBERT [7]. OuluVS2 finetuning follows the protocol of Han et al. [29]. Representation, distillation, and ablation comparisons share twenty LRS3 training and eight validation utterances, with 600 pretraining updates and 150 recognition fine-tuning updates. AdamW [66] updates the final two Base encoder layers, projection, normalization, and Adaptive Weighting while frontend features and unit prototypes remain frozen. We set $\lambda _ { \mathrm { s i n g l e } } = 2 , \lambda _ { \mathrm { A g g 2 S } } = 0 . 1 , \lambda _ { \mathrm { R S A } } = 0 . 5 , \tau _ { \mathrm { d } } = 2$ , and $\beta _ { \mathrm { s y n c } } = \beta _ { \mathrm { r e l } } = 1$ . Supplementary Section S1.2 gives optimization budgets, sampling, and evaluation settings.

## 4.3 Overall Performance Comparison

To evaluate MuViRL’s view-invariant multimodal representations, we compare its performance with representative supervised and self-supervised methods on LRS3. Table 1 presents results for Base and Large encoders to assess the efect of model capacity. MuViRL and AV-HuBERT are compared using the same Large encoder and VA training data.

Table 1 Overall performance on LRS3. The $0 ^ { \circ }$ column uses original test data. Other columns add synthesized views up to the indicated yaw magnitude in both directions, with zero pitch ofset. VA denotes pretraining with synthesized multiview data.
<table><tr><td rowspan="2"></td><td rowspan="2">Encoder Criterion</td><td rowspan="2"></td><td colspan="2">Training data (h)</td><td colspan="3">WER (%) ↓</td></tr><tr><td></td><td>Labeled Unlabeled</td><td> $\mathbf { 0 } ^ { \circ }$ </td><td> $\mathbf { 5 } ^ { \circ }$ </td><td> $\mathbf { 1 0 } ^ { \circ }$  15°</td></tr><tr><td colspan="8"></td></tr><tr><td colspan="8">Supervised</td></tr><tr><td>CM-seq2seq [67]</td><td>Conformer CTC+S2S</td><td></td><td>433</td><td></td><td></td><td>43.3 64.6 70.8 83.9</td><td></td></tr><tr><td>XMD [68]</td><td>CNN</td><td>CTC</td><td>433</td><td></td><td></td><td>68.8 74.1 78.6 82.0</td><td></td></tr><tr><td>MSR [69]</td><td>RNN</td><td>S2S</td><td>433</td><td></td><td></td><td>57.8 66.4 72.5 77.6</td><td></td></tr><tr><td>Auto-AVSR [70]</td><td>Conformer CTC+S2S</td><td></td><td>818</td><td></td><td></td><td>33.0 51.5 64.8 73.0</td><td></td></tr><tr><td colspan="8"></td></tr><tr><td></td><td></td><td>Self-supervised S2S</td><td>433</td><td>1,759</td><td></td><td></td><td></td></tr><tr><td>CMA [23]</td><td>Transformer</td><td></td><td>433</td><td>1,759</td><td></td><td></td><td>32.5 40.4 50.5 73.8</td></tr><tr><td>RAVen [8]</td><td>Transformer CTC+S2S</td><td></td><td>433</td><td></td><td></td><td></td><td>27.8 35.9 49.1 52.3</td></tr><tr><td>AV-HuBERT-B [7]</td><td>Transformer</td><td>S2S</td><td></td><td>1,759</td><td></td><td></td><td>34.8 45.7 66.2 72.1</td></tr><tr><td>AV-HuBERT-L [7]</td><td>Transformer</td><td>S2S</td><td>433</td><td>1,759</td><td></td><td></td><td>28.2 36.3 43.6 50.5</td></tr><tr><td>AV-HuBERT-L (VA-433h) Transformer</td><td></td><td>S2S</td><td>433</td><td>433</td><td></td><td></td><td>44.663.771.582.4</td></tr><tr><td>AV-HuBERT-L (VA-1759h) Transformer</td><td></td><td>S2S</td><td>433</td><td>1,759</td><td></td><td></td><td>28.6 40.2 46.6 52.7</td></tr><tr><td colspan="8">Our Method</td></tr><tr><td>MuViRL-L (VA-433h)</td><td>Transformer</td><td>S2S</td><td>433</td><td>433</td><td></td><td>42.2 61.4 68.1 71.4</td><td></td></tr><tr><td>MuViRL-L (VA-1759h)</td><td>Transformer</td><td>S2S</td><td>433</td><td>1,759</td><td></td><td>27.034.340.846.9</td><td></td></tr></table>

With synthesized yaw ofsets up to $\pm 1 5 ^ { \circ }$ , MuViRL-L pretrained on VA-1759h achieves 46.9% WER, outperforming CMA (73.8%) and RAVen (52.3%). Using the same Large encoder and VA-1759h training data, it achieves 27.0% WER on the original test set and 46.9% with the $\pm 1 5 ^ { \circ }$ range, compared with 28.6% and 52.7% for AV-HuBERT-L. The relative improvement is greater with synthesized views than on the original test set, averaging 10.9% across the four settings. Improvements are also obtained with VA-433h, where MuViRL-L achieves 71.4% WER with the $\pm 1 5 ^ { \circ }$ range, compared with 82.4% for AV-HuBERT-L. The improvements under both pretraining settings, with encoder capacity and training data matched, indicate robustness to viewpoint variation.

## 4.4 Evaluation of Multimodal View-Invariant Representations

We use AV-HuBERT as the baseline to assess view invariance and generalization. The qualitative and quantitative comparisons cover recognition on real multiview data, spatial responses, feature distributions, and temporal correspondence.

Qualitative evaluation. We adopt a spatiotemporal occlusion sensitivity method similar to receptive-field visualization approaches [71, 72] to identify regions contributing to prediction. As shown in Fig. 4, the baseline’s spatial responses vary across views, with additional prediction errors. In the illustrated sequence, MuViRL maintains sensitivity around the mouth opening and lip contours and makes fewer word prediction errors. These localized responses and predictions indicate preserved articulatory evidence across views.

Table 2 Multimodal representation evaluation. Here, MuViRL denotes the configuration trained without decision or depth weighting. Values are means ± standard deviations over three pretraining seeds.
<table><tr><td rowspan="2">Configuration</td><td colspan="3">LRS3</td><td colspan="3">OuluVS2</td></tr><tr><td>Prediction accuracy (%) ↑</td><td>Cross-view Cross-modal Prediction Recall (%) ↑</td><td>Recall (%) ↑</td><td>accuracy (%) ↑</td><td>Recall (%) ↑</td><td>Cross-view Cross-modal Recall (%) ↑</td></tr><tr><td>Baseline</td><td> $3 . 8 5 { \pm } 0 . 0 3$ </td><td> $6 4 . 5 9 { \pm } 0 . 2 7 \ $ </td><td> $5 3 . 5 6 { \pm } 0 . 2 3 $ </td><td> $2 9 . 1 7 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $8 3 . 4 0 { \pm } 0 . 3 5 $ </td><td> $6 4 . 0 2 { \pm } 0 . 6 0 $ </td></tr><tr><td>Random view</td><td> ${ \bf 3 . 9 3 } \pm 0 . 0 6$ </td><td> $6 5 . 3 2 { \pm } 0 . 3 4$ </td><td> $5 3 . 9 3 { \pm } 0 . 2 8 $ </td><td> $3 2 . 8 4 { \pm } 0 . 4 8$ </td><td> $8 5 . 6 8 { \pm } 0 . 5 1 $ </td><td> $6 7 . 2 3 { \pm } 1 . 0 8 $ </td></tr><tr><td>MuViRL</td><td> $3 . 9 1 { \pm } 0 . 0 5$ </td><td> ${ \bf 6 7 . 6 4 } \pm 0 . 0 6$ </td><td> ${ \bf 5 5 . 3 2 } { \pm } 0 . 0 9$ </td><td> $\mathbf { 3 5 . 0 9 } { \scriptstyle \pm 0 . 3 3 }$ </td><td> $\mathbf { 8 8 . 6 6 { \pm } 0 . 2 2 }$ </td><td> ${ \bf 6 8 . 6 0 { \scriptstyle \pm 0 . 6 7 } }$ </td></tr></table>

Quantitative evaluation. Fig. 5 presents results on real and synthesized multiview data. After OuluVS2 fine-tuning from matched VA-1759h pretraining and VA-433h recognition checkpoints, MuViRL-L achieves 3.65% overall WER on OuluVS2, compared with 4.47% for the baseline, an 18.3% relative improvement. WER is also lower at $6 0 ^ { \circ }$ and $9 0 ^ { \circ }$ , and the overall improvement remains with VA-30h fine-tuning. Direct transfer to TCD-TIMIT uses all 5,782 paired utterances from 59 volunteers with identical preprocessing and decoding. Overall WER is 79.85% and 59.78% with 30 h and 433 h finetuning, compared with 81.48% and 61.06% for the baseline. Lower WER at both viewpoints without additional training on the transfer corpus indicates generalization to real multiview data. Evaluation on synthesized data covers diferent angular ofsets and proportions of synthesized test clips. With 433 h fine-tuning, both Base and Large configurations outperform the baseline in most settings. The recognition gains extend across diferent angular ranges and proportions of synthesized clips.

Representation structure analysis. We assess whether representations remain consistent within multimodal units and across synchronized views while preserving temporal distinctions. Table 2 presents prediction and retrieval results, and Fig. 6 shows feature structure. Experiments on both datasets use the Base configurations. Baseline continues prediction on the original view, Random view samples one view per update, and MuViRL uses MMP, Adaptive Weighting, Agg2S, and Temporal RSA without decision or depth weighting.

Prediction accuracy is the proportion of valid masked positions with a correctly predicted unit. Both retrieval metrics [10, 18] report Recall@1. For each candidate view, cosine similarity ranks its valid frames within the same utterance. Cross-view Recall uses original visual queries, and Cross-modal Recall uses audio queries. A query is correct when the highest-ranked frame has the same timestamp. As shown in Table 2, Cross-view Recall improves from 64.59% to 67.64% on LRS3 and from 83.40% to 88.66% on OuluVS2. OuluVS2 Cross-modal Recall increases from 64.02% to 68.60%. The gains in both recalls indicate more accurate temporal matching across viewpoints and modalities.

![](images/f211bc0cd69504d6f82580bcc1a7817ccaccdb25b08685d52f1ddeacee1bcff2.jpg)  
re(b) Synthesized view (yaw = +20°)  
(b) Synthesized view (yaw = +20°)<sup>(b) Synthesized view (yaw = +20°)</sup>Fig. 4 Qualitative evaluation of view-invariant representations. We overlay sensitivity (b) Synthesized view (yaw = +20°)<sub>heatmaps on an LRS3 test sequence for (a) the original view and (b) a synthesized view with</sub> yaw $+ 2 0 ^ { \circ }$ , showing the contribution of diferent mouth regions to prediction. Colors range from blue (low sensitivity) to red (high sensitivity), according to the loss change ∆L under local occlusion. Word predictions are aligned below the frames. Red words mark transcript mismatches, and [miss] denotes a deletion.

Unit consistency averages cosine similarity between distinct normalized features sharing a teacher label. View consistency averages it over synchronized feature pairs within each utterance. MuViRL improves similarity for 436 of 447 units and 353 of 360 test utterances in Fig. 6(a,b). Across three seeds, the respective means increase from 0.6379 to 0.6997 and from 0.8514 to 0.9202. Unit centers also become more separated, with squared distance increasing from 1.0403 to 1.1300. For the same LRS3 sequence, the bright diagonal and peaks at zero ofset in Fig. $6 ( \mathrm { c } , \mathrm { d } )$ indicate temporal alignment. Ofsets are synthesized visual frame indices minus original visual or audio frame indices.

![](images/cc33d285845c892305ce08f4a662a462f5faeca5caaa669486b22912a834ac8d.jpg)  
(a) Angular Offset (Base)

![](images/0a6ce7da325a9dac11e564edd5ffb7a774e3ef930f060371c801ef0d7bf4b7c5.jpg)  
(b) Random Proportions (Base)

![](images/45fa20864c4419bbe4f7422e7b8f78cec80dff2aafb76d5ee1314bfe7b79d8cf.jpg)  
(c) Angular Offset (Large)

![](images/3fc928bf5398fe5f79d4d5383cfadbb5d2e7fbb90b99e7ea3cfc890fc57343eb.jpg)  
(d) Random Proportions (Large)

![](images/27fcb720a40dad6f97653884a8db798408686fb551a57a4ab8eb87d0269a88c4.jpg)  
(e) TCD­TIMIT

![](images/66622e122491db1f404388133f576308cb2735ecf214b468cbb203a2155a3e6a.jpg)  
(f) OuluVS2  
Fig. 5 Quantitative evaluation on multiview data. B-30h/B-433h denote AV-HuBERT Base pretrained on 433 h and fine-tuned on 30 h/433 h, respectively. L-30h/L-433h denote AV-HuBERT Large pretrained on 1,759 h and fine-tuned on 30 h/433 h, respectively. MuViRL uses the corresponding view augmented data. (a,c) Angular Ofset: performance as the range of synthesized yaw ofsets expands. (b,d) Random Proportions: performance with varying proportions of synthesized test clips. (e,f) Results on TCD-TIMIT and OuluVS2, respectively.

Supplementary Table S2 reports prediction and retrieval results for component comparisons.

![](images/521feba47a5b313dc38a5551c3a851b8bc329e0837f20d935758cb4230a92e24.jpg)  
(a) Unit consistency

![](images/ba3b3a35da5a8bb096f5c25a79e671c6d165fbbff319d89b0f779825c2fc7001.jpg)  
(b) View consistency

![](images/e42fa7d599796ffd118eba53f9f44a74b0adb2c8749c8721a648b88631b00d49.jpg)  
(c) Cross-modal correspondence

![](images/5a826a2bace3ca05f687dce02b347ffa8c92de1e605a9d36ad2b628e598ae16e.jpg)  
(d) Temporal alignment  
Fig. 6 Multimodal representation structure. (a) Unit consistency and (b) view consistency on OuluVS2. Points below the diagonal indicate higher similarity with MuViRL. (c) Correspondence between audio and synthesized visual frames on LRS3. The bright diagonal indicates matching timestamps. (d) Mean similarity at diferent frame ofsets for visual and audiovisual pairs from the same sequence. Peaks at zero ofset indicate temporal alignment.

## 4.5 Evaluation of Calibrated View Aggregation

To assess how Calibrated View Aggregation supports multimodal representation learning, we evaluate aggregate prediction and recognition from individual views. Table 3 presents aggregation results with the encoder and Adaptive Weighting fixed, and Fig. 7 shows weight allocation and prediction.

Decision model adaptation analysis. The aggregation comparison uses scores from a decision model fine-tuned with four quality grades on ORL, Yale, Notting Hill, and MEAD. Training uses blur and occlusion, while evaluation on pixelation tests generalization. Actors and identities are disjoint across adaptation, calibration, and evaluation. Supplementary Table S1 presents grading accuracy across the four datasets. Aggregation evaluation uses 120 OuluVS2 test utterances covering all twelve test speakers. Distortion is applied to two oblique views at one of three levels.

![](images/512717cabdc5dbf1c455b95bff4f1ca4fd1d03d38de6b801e75bd6b9f093f906.jpg)  
(a) View weight allocation

![](images/3d5a861d5783905941668a31e2cb1c794a476dc4c1b8504c298fc8eacc1a0046.jpg)  
(b) Aggregation performance  
Fig. 7 View aggregation analysis. (a) Weight gains for views without added distortion. Boxes show quartiles and medians, and diamonds mark means. (b) Changes in similarity and accuracy per utterance relative to aggregation without reliability weighting. Similarity on the horizontal axis is the paired change, scaled by $1 0 ^ { 3 }$ . Points in the upper right indicate improved stability and prediction.

Table 3 Aggregation evaluation with diferent input configurations. Original inputs are evaluated without added distortion. Adaptation means fine-tuning with four quality grades. $\mathrm { w / o }$ reliability means $q _ { i , t } = 1$ . Similarity means cosine agreement with the reference aggregate computed from original inputs, with larger values indicating greater stability.
<table><tr><td rowspan="2">Configuration</td><td rowspan="2">Adaptation</td><td colspan="3">Prediction accuracy (%) ↑</td><td rowspan="2">Similarity ↑</td></tr><tr><td>Original inputs</td><td>Blur/ occlusion</td><td>Pixelation</td></tr><tr><td> $\mathrm { w / o }$  reliability</td><td></td><td>37.29</td><td>36.23</td><td>37.05</td><td>0.9702</td></tr><tr><td>weighting Laplacian</td><td></td><td>37.32</td><td>36.73</td><td>37.05</td><td>0.9723</td></tr><tr><td>Pixel similarity</td><td></td><td>37.35</td><td>36.73</td><td>37.11</td><td>0.9854</td></tr><tr><td>Decision model</td><td>X</td><td>37.44</td><td>36.82</td><td>36.87</td><td>0.9918</td></tr><tr><td>Decision model</td><td>V</td><td>38.22</td><td>37.87</td><td>36.93</td><td>0.9926</td></tr></table>

View aggregation analysis. As shown in Table 3, aggregation with the adapted decision model achieves 37.87% accuracy under blur and occlusion, compared with 36.82% without adaptation and 36.23% without reliability weighting. It also gives the highest accuracy on original inputs (38.22%) and similarity to the reference aggregate (0.9926). The reference uses inputs without added distortion and uniform reliability. Table 3 pools frame contributions across distortion conditions within each seed, then averages over three seeds. Fig. 7 shows weight allocation and paired changes per utterance, averaged over three distortion types, three severity levels, and three seeds. Under unseen pixelation, pixel similarity gives the highest accuracy (37.11%), compared with 36.93% using adapted decisions. The adapted decision scores improve aggregate prediction under blur and occlusion.

Table 4 Evaluation of Aggregation-to-Single Distillation (Agg2S). $\mathcal { L } _ { \mathrm { s i n g l e } }$ and $\mathcal { L } _ { \mathrm { a g g } }$ mean target prediction from one view and aggregated views, respectively.
<table><tr><td>Configuration Cross-view Recall (%) ↑</td></tr><tr><td>WER (%) ↓</td></tr><tr><td>Prediction objective</td></tr><tr><td> $\mathcal { L } _ { \mathrm { s i n g l e } }$  65.32 69.50 64.42 68.84</td></tr><tr><td> $\mathcal { L } _ { \mathrm { a g g } }$  MMP 65.25 69.50</td></tr><tr><td>67.63 66.21</td></tr><tr><td> $\mathrm { M M P + A g g 2 S }$ </td></tr><tr><td>Aggregation method</td></tr><tr><td>Mean 66.18 69.47</td></tr><tr><td>Learned 67.21 69.55</td></tr><tr><td>Cross-view Attention 67.32 69.43</td></tr><tr><td>Adaptive Weighting 67.63 66.21</td></tr></table>

Evaluation of decision scores. Supplementary Table S3 evaluates teacher supervision, ranking, and temperature calibration under matched data, initialization, and training budgets. Ranking improves pairwise accuracy from 52.90% to 54.02%, while calibration lowers expected calibration error from 0.68% to 0.49%. We compare aggregation using uniform reliability, ranking scores, and calibrated scores on the same OuluVS2 split. Base encoder initialization, view sampling, training budgets, Agg2S, and uniformly weighted Temporal RSA remain fixed. Across three seeds, mean WER is 17.52% with uniform reliability, 17.10% with ranking scores, and 16.48% with calibrated scores. Calibrated scores also yield the highest aggregate accuracy (37.56%). These results show that calibrated decision scores improve aggregate prediction and reduce recognition errors.

## 4.6 Evaluation of Aggregation-to-Single Distillation

To evaluate Agg2S, we compare prediction objectives and aggregation methods in Table 4. Both comparisons reuse the LRS3 split and Base training settings from Section 4.4, with uniform reliability and Temporal RSA disabled. Adaptive Weighting is used for the aggregation branch in the objective comparison. The aggregation comparison retains MMP and Agg2S.

Prediction objective comparison. MMP with Agg2S achieves 67.63% Cross-view Recall and 66.21% WER, compared with 65.25% and 69.50% for MMP alone. The OuluVS2 comparison reuses the representation experiment’s split and training settings, with uniform reliability and Temporal RSA disabled. Increasing the Agg2S coeficient from zero to 0.1 raises mean Cross-view Recall from 86.73% to 86.91% over three seeds (Supplementary Table S2). The retrieval gains on both datasets and lower LRS3 WER support transferring aggregated predictions to individual views.

Table 5 Ablation studies of MuViRL on LRS3. Decision and Depth mean decision and geometric weighting of Temporal RSA, respectively.
<table><tr><td>Agg2S RSA Decision Depth</td><td></td><td></td><td>Cross-view Recall (%)↑1</td><td>Cross-modal Recall (%)↑</td><td>WER (%) ↓</td></tr><tr><td></td><td>Baseline</td><td></td><td>65.25</td><td>53.93</td><td>69.50</td></tr><tr><td>√</td><td></td><td></td><td>67.63</td><td>55.12</td><td>66.21</td></tr><tr><td></td><td>√</td><td></td><td>66.22</td><td>54.30</td><td>68.54</td></tr><tr><td>√</td><td>√</td><td></td><td>67.64</td><td>55.32</td><td>65.87</td></tr><tr><td>√</td><td>√</td><td>√</td><td>68.17</td><td>56.03</td><td>65.07</td></tr><tr><td>√</td><td>√</td><td>√</td><td>67.84</td><td>55.35</td><td>65.95</td></tr><tr><td>√</td><td>√</td><td>√ √</td><td>68.69</td><td>56.40</td><td>64.75</td></tr></table>

Aggregation method comparison. Mean assigns equal weights to all views. Learned estimates weights from individual views, while Cross-view Attention models interactions across views. Adaptive Weighting achieves the highest recall and lowest WER. Cross-view Recall increases from 66.18% with Mean to 67.63% with Adaptive Weighting. With the distillation objective fixed, Adaptive Weighting improves temporal correspondence and reduces recognition errors.

## 4.7 Ablation Study

We conduct an ablation study of MuViRL using the LRS3 dataset. As shown in Table 5, all configurations use the same data, initialization, and training settings. Baseline uses MMP and Adaptive Weighting with uniform aggregation reliability. Decision and Depth apply utility $u _ { i , b }$ and geometry $g _ { i , b }$ to RSA weights.

Agg2S. As shown in Table 5, Agg2S improves Cross-view Recall from 65.25% to 67.63% and Cross-modal Recall from 53.93% to 55.12%. WER falls from 69.50% to 66.21%. With Temporal RSA retained, Agg2S further improves WER from 68.54% to 65.87%. This improvement indicates that prediction distillation provides supervision beyond temporal feature alignment.

Reliability-Aware Real–Synthetic Alignment. Temporal RSA alone achieves 66.22% Cross-view Recall, 54.30% Cross-modal Recall, and 68.54% WER, improving all three metrics over Baseline. When combined with Agg2S, WER improves from 66.21% to 65.87%, with recalls of 67.64% and 55.32%. The joint window and relation objective improves recognition when used with distillation. With decision weighting of Temporal RSA, WER falls from 65.87% to 65.07%, while both recalls increase to 68.17% and 56.03%. Depth weighting alone achieves 67.84% and 55.35% recall but gives 65.95% WER. Combining decision and depth weighting yields 68.69% and 56.40% recall and 64.75% WER. Weighting alignment by both predictive usefulness and geometric consistency improves correspondence across views and modalities.

Compared with Baseline, Agg2S with weighted Temporal RSA increases Cross-view Recall from 65.25% to 68.69% and Cross-modal Recall from 53.93% to 56.40%. WER decreases from 69.50% to 64.75%. With masked prediction supervision held fixed, these gains support the combined use of Agg2S and reliability weighted temporal alignment for learning consistent and discriminative representations.

## 5 Conclusion

In this paper, we proposed MuViRL, a self-supervised framework for multimodal view-invariant visual speech representation learning. Synchronized 3DGS synthesis provides corresponding views for learning from shared audiovisual targets. Calibrated View Aggregation uses calibrated multimodal decisions and Adaptive Weighting to select informative visual features. Agg2S transfers aggregated predictions to individual views, and Temporal RSA preserves temporal correspondence between original and synthesized representations. Experiments show improved multimodal representation consistency and visual speech recognition under view variation, including generalization to real multiview data. Ablation studies support the contributions of Agg2S and reliability weighted alignment.

## Declarations of Conflict of Interest

The authors declare that they have no financial or nonfinancial conflicts of interest related to this work.

## References

[1] J.-B. Alayrac, A. Recasens, R. Schneider, R. Arandjelovi´c, J. Ramapuram, J. De Fauw, L. Smaira, S. Dieleman, A. Zisserman. Self-supervised Multi-Modal versatile networks. In Advances in Neural Information Processing Systems, vol. 33, pp. 25–37, 2020.

[2] R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, I. Misra. ImageBind: One embedding space to bind them all. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15180–15190, 2023.

[3] L. Wang, H. Xu, W. Kang. MVContrast: Unsupervised pretraining for multi-view 3D object recognition. Machine Intelligence Research, vol. 20, no. 6, pp. 872–883, 2023.

[4] S. Das, M. S. Ryoo. ViewCLR: Learning self-supervised video representation for unseen viewpoints. In Proceedings of IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5573–5583, 2023.

[5] J. Xu, N. Zhao, G. Niu, M. Sugiyama, X. Zhu. Robust multi-view learning via representation fusion of sample-level attention and alignment of simulated perturbation. In Proceedings of IEEE/CVF International Conference on Computer Vision, pp. 4232–4241, 2025.

[6] H. Wang, H. Fei, B. Dai, J. Gui. Multimodal skeleton-based action representation learning via decomposition and composition. Machine Intelligence Research, vol. 23, no. 5, pp. 1028–1040, 2026.

[7] B. Shi, W.-N. Hsu, K. Lakhotia, A. Mohamed. Learning audio-visual speech representation by masked multimodal cluster prediction. In Proceedings of International Conference on Learning Representations, 2022.

[8] A. Haliassos, P. Ma, R. Mira, S. Petridis, M. Pantic. Jointly learning visual and auditory speech representations from raw data. In Proceedings of International Conference on Learning Representations, 2023.

[9] A. Haliassos, A. Zinonos, R. Mira, S. Petridis, M. Pantic. BRAVEn: Improving self-supervised pre-training for visual and auditory speech recognition. In Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 11431–11435, 2024.

[10] E. Araujo, A. Rouditchenko, Y. Gong, S. Bhati, S. Thomas, B. Kingsbury, L. Karlinsky, R. Feris, J. R. Glass, H. Kuehne. CAV-MAE Sync: Improving contrastive audio-visual mask autoencoders via fine-grained alignment. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18794–18803, 2025.

[11] T. Afouras, J. S. Chung, A. Zisserman. LRS3-TED: A large-scale dataset for visual speech recognition, [Online], Available: https://arxiv.org/abs/ 1809.00496, 2018.

[12] J. S. Chung, A. Nagrani, A. Zisserman. VoxCeleb2: Deep speaker recognition. In Proceedings of Interspeech, pp. 1086–1090, 2018.

[13] Q. Zhang, H. Wu, C. Zhang, Q. Hu, H. Fu, J. T. Zhou, X. Peng. Provable dynamic fusion for low-quality multimodal data. In Proceedings of International Conference on Machine Learning, pp. 41753–41769, 2023.

[14] Z. Gao, X. Jiang, X. Xu, F. Shen, Y. Li, H. T. Shen. Embracing unimodal aleatoric uncertainty for robust multimodal fusion. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26876–26885, 2024.

[15] S. Qian, T. Kirschstein, L. Schoneveld, D. Davoli, S. Giebenhain, M. Nießner. GaussianAvatars: Photorealistic head avatars with rigged 3D Gaussians. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20299–20309, 2024.

[16] J. Zhao, X. Zhu, Z. Wang, Z. Lei. STAvatar: Soft binding and temporal density control for monocular 3D head avatars reconstruction. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10996–11005, 2026.

[17] L. Zhang, L. Ruan, A. Hu, Q. Jin. Multimodal pretraining from monolingual to multilingual. Machine Intelligence Research, vol. 20, no. 2, pp. 220–232, 2023.

[18] H. Lu, Y. Huo, M. Ding, N. Fei, Z. Lu. Cross-modal contrastive learning for generalizable and eficient image-text retrieval. Machine Intelligence Research, vol. 20, no. 4, pp. 569–582, 2023.

[19] H. Yuan, Y. Huang, N. Yu, D. Zhang, Z. Du, Z. Liu, K. Zhang. Multimodal pretrained knowledge for real-world object navigation. Machine Intelligence Research, vol. 22, no. 4, pp. 713–729, 2025.

[20] R. Bachmann, D. Mizrahi, A. Atanov, A. Zamir. MultiMAE: Multi-modal multi-task masked autoencoders. In Proceedings of European Conference on Computer Vision, pp. 348–367, 2022.

[21] A. Nagrani, S. Yang, A. Arnab, A. Jansen, C. Schmid, C. Sun. Attention bottlenecks for multimodal fusion. In Advances in Neural Information Processing Systems, vol. 34, pp. 14200–14213, 2021.

[22] Z. Liu, X. Li, C. Chen, L. Guo, L. Li, D. Wang. AlignVSR: Audio-visual cross-modal alignment for visual speech recognition. In Proceedings of11th International Conference on Communication and Information Processing, pp. 161–165, 2025.

[23] S. Kim, K. Jang, S. Bae, H. Kim, S.-Y. Yun. Learning video temporal dynamics with cross-modal attention for robust audio-visual speech recognition. In Proceedings of IEEE Spoken Language Technology Workshop, pp. 447–454, 2024.

[24] X. Liu, E. Lakomkin, K. Vougioukas, P. Ma, H. Chen, R. Xie, M. Doulaty, N. Moritz, J. Kol´aˇr, S. Petridis, M. Pantic, C. Fuegen. SynthVSR: Scaling up visual speech recognition with synthetic supervision. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18806–18815, 2023.

[25] Y. Tian, D. Krishnan, P. Isola. Contrastive multiview coding. In Proceedings of European Conference on Computer Vision, pp. 776–794, 2020.

[26] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. P´oczos, R. Salakhutdinov, A. J. Smola. Deep sets. In Advances in Neural Information Processing Systems, vol. 30, pp. 3391–3401, 2017.

[27] J. Lee, Y. Lee, J. Kim, A. R. Kosiorek, S. Choi, Y. W. Teh. Set Transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings of International Conference on Machine Learning, pp. 3744–3753, 2019.

[28] S. Petridis, Y. Wang, Z. Li, M. Pantic. End-to-end multi-view lipreading. In Proceedings of British Machine Vision Conference, pp. 161.1–161.12, 2017.

[29] H. Han, S. Kang, C. D. Yoo. Multi-view visual speech recognition based on multi task learning. In Proceedings of IEEE International Conference on Image Processing, pp. 3983–3987, 2017.

[30] C. Bucilu˘a, R. Caruana, A. Niculescu-Mizil. Model compression. In Proceedings of ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 535–541, 2006.

[31] F. Tung, G. Mori. Similarity-preserving knowledge distillation. In Proceedings of IEEE/CVF International Conference on Computer Vision, pp. 1365–1374, 2019.

[32] Z. Yan, Y. Wang, K. Zhang, Z. Sun, L. He. Boosting multi-modal ocular recognition via spatial feature reconstruction and unsupervised image quality estimation. Machine Intelligence Research, vol. 21, no. 1, pp. 197–214, 2024.

[33] H. Guan, C. Song, Z. Zhang. LiDAR-camera cooperative semantic segmentation. Machine Intelligence Research, vol. 22, no. 5, pp. 956–968, 2025.

[34] D. Chen, R. Chen, S. Zhang, Y. Wang, Y. Liu, H. Zhou, Q. Zhang, Y. Wan, P. Zhou, L. Sun. MLLM-as-a-Judge: Assessing multimodal LLM-asa-judge with vision-language benchmark. In Proceedings of International Conference on Machine Learning, pp. 6562–6595, 2024.

[35] H. Wu, Z. Zhang, W. Zhang, C. Chen, L. Liao, C. Li, Y. Gao, A. Wang, E. Zhang, W. Sun, Q. Yan, X. Min, G. Zhai, W. Lin. Q-Align: Teaching LMMs for visual scoring via discrete text-defined levels. In Proceedings of International Conference on Machine Learning, pp. 54015–54029, 2024.

[36] Z. You, X. Cai, J. Gu, T. Xue, C. Dong. Teaching large language models to regress accurate image quality scores using score distribution. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14483–14494, 2025.

[37] A. Marafioti, O. Zohar, M. Farr´e, M. Noyan, E. Bakouch, P. Cuenca, C. Zakka, L. Ben Allal, A. Lozhkov, N. Tazi, V. Srivastav, J. Lochner, H. Larcher, M. Morlon, L. Tunstall, L. von Werra, T. Wolf. SmolVLM: Redefining small and eficient multimodal models. In Proceedings of Second Conference on Language Modeling, 2025.

[38] C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, G. Hullender. Learning to rank using gradient descent. In Proceedings of International Conference on Machine Learning, pp. 89–96, 2005.

[39] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On calibration of modern neural networks. In Proceedings of International Conference on Machine Learning, pp. 1321–1330, 2017.

[40] M. Kull, M. Perello-Nieto, M. K¨angsepp, T. Silva Filho, H. Song, P. Flach. Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration. In Advances in Neural Information Processing Systems, vol. 32, pp. 12295–12305, 2019.

[41] S. Cheng, P. Ma, G. Tzimiropoulos, S. Petridis, A. Bulat, J. Shen, M. Pantic. Towards pose-invariant lip-reading. In Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 4357–4361, 2020.

[42] T. Li, T. Bolkart, M. J. Black, H. Li, J. Romero. Learning a model of facial shape and expression from 4D scans. ACM Transactions on Graphics, vol. 36, no. 6, Article number 194, 2017.

[43] P.-W. Grassal, M. Prinzler, T. Leistner, C. Rother, M. Nießner, J. Thies. Neural head avatars from monocular RGB videos. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18653–18664, 2022.

[44] B. Kerbl, G. Kopanas, T. Leimk¨uhler, G. Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, vol. 42, no. 4, Article number 139, 2023.

[45] J. Li, J. Zhang, X. Bai, J. Zheng, X. Ning, J. Zhou, L. Gu. Talking-Gaussian: Structure-persistent 3D talking head synthesis via Gaussian splatting. In Proceedings of European Conference on Computer Vision, pp. 127–145, 2024.

[46] B. Huang, Z. Yu, A. Chen, A. Geiger, S. Gao. 2D Gaussian splatting for geometrically accurate radiance fields. In Proceedings of ACM SIGGRAPH Conference Papers, Article number 32, 2024.

[47] H. Chen, F. Wei, C. Li, T. Huang, Y. Wang, G. H. Lee. VCR-GauS: View consistent depth-normal regularizer for Gaussian surface reconstruction. In Advances in Neural Information Processing Systems, vol. 37, pp. 139725–139750, 2024.

[48] H. Xu, S. Peng, F. Wang, H. Blum, D. Barath, A. Geiger, M. Pollefeys. DepthSplat: Connecting Gaussian splatting and depth. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16453–16463, 2025.

[49] H. Zhao, J. Zhang, Z. Chen, B. Yuan, D. Tao. On robust cross-view consistency in self-supervised monocular depth estimation. Machine Intelligence Research, vol. 21, no. 3, pp. 495–513, 2024.

[50] J. Xia, L. Sun, I. Reid, J.-W. Bian. Self-supervised video depth adaptation for eficient sparse-view NeRF. Machine Intelligence Research, vol. 23, no. 5, pp. 969–979, 2026.

[51] T. Hempel, A. A. Abdelrahman, A. Al-Hamadi. 6D rotation representation for unconstrained head pose estimation. In Proceedings of IEEE International Conference on Image Processing, pp. 2496–2500, 2022.

[52] S. Lin, L. Yang, I. Saleemi, S. Sengupta. Robust high-resolution video matting with temporal guidance. In Proceedings of IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 238–247, 2022.

[53] Z. Zhou, H. Li, H. Liu, N. Wang, G. Yu, R. Ji. STAR loss: Reducing semantic ambiguity in facial landmark detection. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15475–15484, 2023.

[54] S. Zhou, K. C. K. Chan, C. Li, C. C. Loy. Towards robust blind face restoration with codebook lookup Transformer. In Advances in Neural Information Processing Systems, vol. 35, pp. 30599–30611, 2022.

[55] O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional networks for biomedical image segmentation. In Proceedings of Medical Image Computing and Computer-Assisted Intervention, pp. 234–241, 2015.

[56] Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[57] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 586–595, 2018.

[58] T. Kim, J. Oh, N. Y. Kim, S. Cho, S.-Y. Yun. Comparing Kullback-Leibler divergence and mean squared error loss in knowledge distillation. In Proceedings of International Joint Conference on Artificial Intelligence, pp. 2628–2635, 2021.

[59] N. Harte, E. Gillen. TCD-TIMIT: An audio-visual corpus of continuous speech. IEEE Transactions on Multimedia, vol. 17, no. 5, pp. 603–615, 2015.

[60] I. Anina, Z. Zhou, G. Zhao, M. Pietik¨ainen. OuluVS2: A multi-view audiovisual database for non-rigid mouth motion analysis. In Proceedings of IEEE International Conference and Workshops on Automatic Face and Gesture Recognition, pp. 1–5, 2015.

[61] F. S. Samaria, A. C. Harter. Parameterisation of a stochastic model for human face identification. In Proceedings of IEEE Workshop on Applications of Computer Vision, pp. 138–142, 1994.

[62] P. N. Belhumeur, J. P. Hespanha, D. J. Kriegman. Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 19, no. 7, pp. 711–720, 1997.

[63] X. Cao, C. Zhang, C. Zhou, H. Fu, H. Foroosh. Constrained multi-view video face clustering. IEEE Transactions on Image Processing, vol. 24, no. 11, pp. 4381–4393, 2015.

[64] K. Wang, Q. Wu, L. Song, Z. Yang, W. Wu, C. Qian, R. He, Y. Qiao, C. C. Loy. MEAD: A large-scale audio-visual dataset for emotional talking-face generation. In Proceedings of European Conference on Computer Vision, pp. 700–717, 2020.

[65] K. He, X. Zhang, S. Ren, J. Sun. Deep residual learning for image recognition. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

[66] I. Loshchilov, F. Hutter. Decoupled weight decay regularization. In Proceedings of International Conference on Learning Representations, 2019.

[67] P. Ma, S. Petridis, M. Pantic. End-to-end audio-visual speech recognition with Conformers. In Proceedings of IEEE International Conference on

Acoustics, Speech and Signal Processing, pp. 7613–7617, 2021.

[68] T. Afouras, J. S. Chung, A. Zisserman. ASR is all you need: Crossmodal distillation for lip reading. In Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 2143–2147, 2020.

[69] B. Xu, C. Lu, Y. Guo, J. Wang. Discriminative multi-modality speech recognition. In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14433–14442, 2020.

[70] P. Ma, A. Haliassos, A. Fernandez-Lopez, H. Chen, S. Petridis, M. Pantic. Auto-AVSR: Audio-visual speech recognition with automatic labels. In Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 1–5, 2023.

[71] M. D. Zeiler, R. Fergus. Visualizing and understanding convolutional networks. In Proceedings of European Conference on Computer Vision, pp. 818–833, 2014.

[72] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, A. Torralba. Object detectors emerge in deep scene CNNs. In Proceedings of International Conference on Learning Representations, 2015.