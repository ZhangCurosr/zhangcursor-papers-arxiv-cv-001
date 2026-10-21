# CLOSER: Evidence-Aware Active Visual Verification for Embodied Object-Goal Navigation

Pengying Wu, Mingye Jiang, Yunze Hu, Zhejia Wen, Kangjie Zhou, Zongru Yang, Zhe Peng, and Chang Liu

Abstract—Object-goal navigation (ObjectNav) requires an embodied agent to find an instance of a specified category in an unseen environment and stop near it. The agent typically relies on an object detector but must judge online from a limited egocentric video stream, sometimes a single frame, whether a detected candidate is the target. Detectors often confidently mistake similar objects, such as a sofa for a bed, and stopping at a false positive fails the episode. Prevailing systems also approach candidates greedily, leaving them occluded or partially observed. We present CLOSER, an evidence-aware active verification framework that treats stopping as closed-loop evidence acquisition, coupling vision–language model (VLM) verification with active perception. CLOSER fine-tunes an open-source VLM to label each view of a candidate tracked across frames as target, non-target, or uncertain, selects complementary views with an embedding-based diversity gate, and fuses their per-view probabilities with a lightweight network. If the fused judgment remains uncertain, a diffusion policy drives the agent toward the candidate using only the tracked mask and depth until the decision becomes decisive. CLOSER runs locally and plugs into existing ObjectNav systems without retraining them. On Gibson and Habitat-Matterport 3D, it raises average success rate by 3.6 points and success weighted by path length by 2.0 points, showing that verifying evidence before committing improves embodied navigation.

Index Terms—Object-goal navigation, embodied AI, spatial reasoning, scene understanding, vision for robotics, object detection and recognition.

## I. INTRODUCTION

O <sup>BJECT-GOAL</sup> <sup>navigation</sup> <sup>(ObjectNav)</sup> <sup>is</sup> <sup>a</sup> <sup>fundamental</sup>task in embodied AI [1] in which an agent searches an unseen environment for an object of a specified target category and stops near it [2], [3]. To find the target, the agent must explore the environment efficiently and infer where the target is likely to be located. When an object resembling the target comes into view, the agent must further judge whether it is the target. ObjectNav therefore demands both efficient exploration and reliable perception from its online egocentric video stream, and remains a challenging problem despite extensive study [4].

To meet these demands, prevailing modular systems adopt an exploration strategy to search promising regions and an offthe-shelf object detector to recognize the observed objects [2], [5], [6]. Whenever a candidate for the target category is detected, the agent must decide from limited local observations whether the candidate is the target and hence whether to stop; in the worst case, only a single partial view is available. Since the episode ends once the agent stops, a wrong stop cannot be corrected, and this stopping decision has been identified as a major bottleneck of modular systems [7], [8]. Although modern open-vocabulary detectors such as YOLO-World and Grounded-SAM [9], [10] achieve high recall on common indoor categories, they frequently mistake semantically similar objects for the target. For example, a sofa observed from a distance may be detected as a bed (Fig. 1), and a stool partially occluded by a table may be detected as a chair. Such confusers are often detected with tight bounding boxes and high confidence, so raising the confidence threshold removes true targets before it removes these errors. Therefore, each detected candidate needs to be further verified before the agent commits to a stop.

![](images/0900b9a3951bc8509d003224c120aafa8a1a5743df8bdf22b177da0806e7f81b.jpg)  
Fig. 1. With CLOSER, the agent approaches an uncertain candidate and decides from closer views. Given the goal bed, the view at (1) shows a distant, partially occluded candidate and does not suffice for a decision. CLOSER therefore labels the candidate uncertain, and the agent approaches it to gather additional views (orange, dotted). At (2), the closer view reveals a sofa confuser; CLOSER labels it no, and the agent resumes exploration (blue, solid). At (3), the view of the next candidate suffices; CLOSER labels it yes, and the agent moves to the bed and stops near it. The bottom panels show the agent’s view and CLOSER’s label at each numbered location.

To this end, recent systems introduce a closed-source vision–language model (VLM) such as GPT-4o-mini as a verifier that confirms or rejects each detected candidate [11]. Such a model provides open-world visual reasoning that the detector lacks, and can reject confusers even when the detection confidence is high. However, the verifier is accessible only through a remote API, which incurs per-query cost and latency and leaves its weights unavailable for local adaptation. A smaller open-source VLM avoids these constraints, yet its verification accuracy drops markedly on hard cases such as distant or partially occluded confusers. Moreover, existing verifiers are binary and must decide even when the current view supports neither confirmation nor rejection. Such a forced decision may stop the agent at a non-target or reject a candidate that a closer view could confirm. Recent work has proposed ternary uncertainty scores and explicit NOT-FOUND outcomes [12], [13], but neither mechanism moves the agent to acquire more views of the candidate for verification.

In this paper, we propose CLOSER (Closed-Loop Observation Sensing for Embodied veRification), an evidenceaware active verification framework in which the agent decides whether to stop by reasoning over the visual evidence on a candidate and acting to improve it. Our key insight is that a reliable stopping decision depends not only on what the candidate appears to be but also on whether the available views suffice to identify it, and this sufficiency depends on the spatial relationship between the agent and the candidate, including its distance, viewing angle, and degree of occlusion. The agent therefore commits to a decision only when the evidence suffices and otherwise approaches the candidate to acquire more informative views (Fig. 1). Realizing this closed loop poses two challenges. First, the agent must learn when its evidence is insufficient, which confidence cannot reveal since distant or occluded views often yield confident predictions. Second, it must learn how to reach a more informative view without a complete map of the unseen environment.

To learn when the evidence is insufficient, CLOSER finetunes an open-source VLM with LoRA [14] to assess both the identity of the candidate and the sufficiency of each view. During navigation, CLOSER tracks the candidate across egocentric frames with ByteTrack [15] and caches complementary views of it, and a lightweight MLP fuses the perview assessments of the VLM to decide whether the evidence suffices. To learn how to reach a more informative view, CLOSER imitates an expert that plans maximum-clearance paths along the Voronoi medial axis of the complete map and smooths them into Bezier trajectories, so that walls are less´ likely to occlude the candidate along the way. To eliminate this dependence on the map, we distill the expert trajectories into a Diffusion Transformer policy [16], [17] that conditions only on the tracked candidate mask and depth, which encode the distance and visibility of the candidate relative to the agent. As the agent follows this policy, verification and active perception form a closed loop, in which every view cached along the trajectory refines the assessment and the approach phase ends as soon as the evidence supports a decision. CLOSER is invoked only when a candidate for the target category is detected and leaves the exploration and mapping components of existing ObjectNav systems unchanged. It can therefore be inserted into these systems as a plug-in verification layer without retraining them.

The main contributions of this paper are summarized as follows:

• We propose CLOSER, a plug-in layer for existing ObjectNav systems that formulates the stopping decision as an active verification problem. Closer judges not only whether a candidate is the target but also whether its views suffice for that judgment, and it drives the agent to acquire more views by approaching the candidate when the current views are insufficient.

• Since a confident prediction on a distant or occluded view can be wrong, we design the verifier in Closer to assess the sufficiency of each view in addition to the identity of the candidate. We fine-tune an open-source VLM for this assessment under a closed-source VLM teacher and train a lightweight MLP that fuses the assessments of complementary views into a single decision.

• We train a map-free approach policy by distilling an expert into a Diffusion Transformer that generates a trajectory toward the candidate from its tracked mask and depth. To keep the candidate in view throughout the approach phase, the expert plans each path along the Voronoi medial axis of the free space for maximum clearance from walls and smooths the path into a Bezier´ trajectory.

• Simulation experiments confirm that CLOSER improves ObjectNav performance. Across diverse ObjectNav systems on Gibson, HM3D v1, and HM3D v2, CLOSER improves success rate (SR), with an average gain of 3.6 points and a maximum gain of 4.9 points. CLOSER also improves success weighted by path length (SPL) by 2.0 points on average.

## II. RELATED WORK

## A. Object-Goal Navigation and Target Verification

Existing ObjectNav methods range from end-to-end reinforcement learning, which learns continuous environment representations with multi-step look-ahead planning [18] or agentcentric relation graphs that encode object layout and target distance [19], to modular pipelines that combine semantic mapping with goal selection [2], [3] and zero-shot variants [4], supported by large-scale simulators [20]. Recent methods use vision–language models to score exploration frontiers [6] or large language models to prioritize search regions [21], [22]; E<sup>2</sup>BA [23] further fuses LLM commonsense scores with geometric costs through particle filtering and uses a cascaded discriminator to decide when to backtrack. These advances concentrate on where to explore, and a separate mechanism is still required to verify whether a detected candidate is the target. For the stopping decision, DITA [7] decouples depthbased termination from the navigation policy, RATE-Nav [8] models spatial regions to trigger stopping, and ACRG [19] encodes the depth-derived agent–target distance into the policy. Classical pipelines verify candidates with detector confidence thresholds [2], which fail under partial observability [24]. Memory [3], [24] and blacklist mechanisms [25] improve decision consistency, SPENav [26] filters task-irrelevant objects with room-guided semantic priors, and recent systems query a closed-source VLM to verify each candidate [11]. For image goals, RSRNav [27] reasons about the spatial relationship between the goal and the current observation through direction-aware correlation and remains robust when the goal viewpoint differs from the test viewpoint. Beyond binary decisions, QAsk-Nav [12] and VLN-NF [13] introduce ternary uncertainty scores and an explicit NOT-FOUND outcome, while IEVE [28] and DRIVE-Nav [29] verify potential targets by keypoint matching and cross-frame VLM inspection. However, these systems either depend on closed-source VLM APIs or do not collect further observations when the evidence on a candidate is insufficient. In CLOSER, verification runs on local hardware, and the agent approaches a candidate with insufficient evidence to acquire additional views.

## B. Vision–Language Model Adaptation and Chain-of-Thought Reasoning

Large VLMs can be adapted by training only a small fraction of their parameters, either a lightweight query module with the image encoder and language model frozen [30] or low-rank updates such as LoRA [14], and EverWalker [31] generates LoRA parameters on the fly from observation prototypes for lifelong navigation. Chain-of-thought supervision provides training signal beyond the final label: CL-CoTNav [32] fine-tunes a VLM on hierarchical chain-ofthought annotations so that it reasons about ObjectNav decisions, and distilling rationales from closed-source teachers into open models is now standard practice [33], [34]. Distillation is likewise used to compress navigation models: Chen et al. [35] distill both where to look and how to move from a teacher into a student with 93% fewer parameters. However, CURE [36] shows that VLM rationales can be inconsistent with the visual evidence even when the final answer is correct, and programbased distillation [37] mitigates this inconsistency through executable verification. CLOSER adapts Qwen2.5-VL-7B with LoRA and keeps a base-model rationale only when its label is correct and a rule-based validator accepts it, so an external teacher annotates only the remaining samples (Sec. III-D3). Rendering novel viewpoints with 3D Gaussian Splatting also improves the viewpoint robustness of VLN agents [38]; likewise, CLOSER renders confusers as hard negatives and refines the perceiver on its residual errors (Sec. III-D2).

## C. Active Perception and Uncertainty-Driven Exploration

Active perception selects sensor placements to maximize information gain. Next-best-view (NBV) methods optimize hand-crafted metrics [39] or predict the downstream value of prospective viewpoints [40], [41]. Uncertainty-aware methods complement view selection through conformal prediction [42] and uncertainty-driven intrinsic motivation [43], and recent work applies them to open-vocabulary object search [44]. Diffusion models have also become expressive policy representations that capture multimodal action distributions [16] and generate navigation trajectories [45]. The approach policy of CLOSER connects these two lines without an explicit information-gain objective (Sec. III-E). A Diffusion Transformer [17] distilled from privileged expert trajectories takes only the candidate mask and depth as input, and the verification stack decides from the newly acquired views whether the evidence suffices.

## III. METHODOLOGY

## A. Problem Formulation

We consider the standard ObjectNav setting, in which an agent placed in a previously unseen indoor environment must find an instance of a given target category c. At each step, the agent receives an egocentric RGB-D observation together with its pose relative to the start of the episode. It then executes either a discrete movement action or the stop action, which ends the episode. The episode succeeds if the agent calls the stop action within 1 m of an instance of the target category under a budget of 500 steps.

A candidate event begins with the detection of a candidate for the target category and ends when the agent decides to stop at the candidate or to reject it and resume exploration. The agent may defer this decision to gather further views of the candidate. The event ends without a decision if the agent loses track of the candidate or the episode terminates first. Since deferral consumes steps from the episode budget, the goal in each event is to reach a correct decision with as few deferral steps as possible.

## B. Framework Overview

CLOSER controls the agent only during a candidate event, so the host ObjectNav system keeps its exploration and mapping modules unchanged. As shown in Fig. 2, the candidate is tracked across frames, and a CLIP-based diversity gate maintains an evidence cache of complementary views. The single-view perceiver is a fine-tuned VLM that predicts tristate probabilities for each retained view (yes for a target, no for a non-target, and uncertain for insufficient evidence). A lightweight MLP fuses these probabilities into a tri-state decision (Sec. III-C). The perceiver and the MLP are trained offline on labels that encode both the semantic identity and the evidence sufficiency of each detection (Sec. III-D). If the fused decision is uncertain, the agent enters the approach phase, in which the approach policy generates a local trajectory toward the candidate from its tracked mask and depth. The policy is trained to imitate expert trajectories planned on a privileged navigability map (Sec. III-E). Views retained along the local trajectory update the fused decision, so the agent can end the approach phase before reaching the candidate (Sec. III-F).

## C. Tri-State Verification Stack

At each step t of a candidate event, the verification stack outputs a fused decision $y _ { t } \in \mathcal { Y } = \{ \mathrm { y e s , u n c e r t a i n , n o } \}$ from the cached views. An uncertain decision indicates that these views support neither yes nor no, so targets and confusers alike can receive it. A binary verifier can also defer by withholding its decision whenever its confidence falls below a threshold. However, the threshold is selected on the validation split and is thus tied to the confidence distribution of that split. When this distribution shifts at deployment, confident predictions on insufficient views can exceed the threshold and become forced decisions, while no labels are available to re-tune the threshold. Moreover, a low confidence does not indicate whether additional views of the candidate would enable a decision. CLOSER therefore supervises uncertain as an explicit class. The training labels of this class are derived from view conditions that the agent can improve by moving, namely the image scale, truncation or occlusion, and viewing distance of the candidate (Sec. III-D1).

![](images/adbd5be0e489f86ff6f7843c9f1fc4cf0d93b9e3b7c12f23aef4430142ee6666.jpg)  
Fig. 2. Closed-loop inference and training of CLOSER. (A) Grounded-SAM, ByteTrack, and SAM track the candidate and refresh its mask in each RGB-D frame. (B) The evidence cache retains views selected by a CLIP-based diversity gate. (C) Per-view tri-state probabilities from the single-view perceiver are fused by a lightweight MLP. (D) After an uncertain decision, a Diffusion Transformer generates waypoints from the mask and depth. Frames observed along the waypoints re-enter panels A and B (uncertain loop). The lower panels depict the perceiver training pipeline (Verifier Training) and the construction of expert waypoint chunks (Policy Training).

1) Candidate Tracking: When Grounded-SAM [10] identifies a candidate for the target category in the RGB frame $I _ { t } ,$ , it returns a bounding box $B _ { t }$ and a segmentation mask $M _ { t } ; D _ { t }$ denotes the aligned depth image. ByteTrack [15] associates the ffi<sub>candidate across subsequent frames, and SAM [46] refreshes</sub> $M _ { t }$ from the tracked box in each frame.

2) Evidence Cache and Diversity Gate: Consecutive frames of a tracked candidate are often nearly identical, and admitting every frame would waste perceiver queries and bias the fused decision toward repeated views. The diversity gate therefore decides which frames enter the evidence cache of the tracked candidate. After frame $I _ { t }$ , the cache is

$$
\mathcal { K } _ { t } = \big \{ ( I ^ { ( j ) } , \mathbf { v } ^ { ( j ) } , \mathbf { p } ^ { ( j ) } ) \big \} _ { j = 1 } ^ { m _ { t } } , \qquad m _ { t } \leq K ,\tag{1}
$$

where $m _ { t }$ is the number of retained views and $K$ is the cache capacity. Entries are indexed in admission order, and entry $j$ stores the retained view $I ^ { ( j ) }$ , its CLIP ViT-L/14 [47] feature $\mathbf { v } ^ { ( j ) }$ , and its tri-state probabilities $\mathbf { p } ^ { ( j ) }$ from the perceiver (Sec. III-C3). The gate measures the novelty of the current frame by the minimum cosine distance between its CLIP feature $\mathbf { v } _ { t }$ and the features in $\boldsymbol { \mathcal { K } } _ { t - 1 }$

$$
\rho _ { t } = \operatorname* { m i n } _ { 1 \leq j \leq m _ { t - 1 } } \bigl [ 1 - \cos \bigl ( \mathbf { v } _ { t } , \mathbf { v } ^ { ( j ) } \bigr ) \bigr ] ,\tag{2}
$$

where $\cos ( \cdot , \cdot )$ denotes cosine similarity. The frame is eligible only if $\rho _ { t }$ exceeds the threshold $\delta _ { \mathrm { v i e w } }$ selected on the validation split. Each track starts with an empty cache, and its first frame is always eligible. If $m _ { t - 1 } ~ < ~ K$ , an eligible frame is appended as the newest entry $\left( I _ { t } , \mathbf { v } _ { t } , \mathbf { p } _ { t } \right)$ . If $m _ { t - 1 } = K$ the gate keeps the size-K subset of $\mathcal { V } _ { t } = \{ \mathbf { v } ^ { ( 1 ) } , \ldots , \mathbf { v } ^ { ( K ) } , \mathbf { v } _ { t } \}$ with the largest diversity,

$$
S _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { S \subseteq \mathcal { V } _ { t } , \ | S | = K } \operatorname { D i v } ( S ) ,\tag{3}
$$

where Div(S) is the mean cosine distance over all pairs of distinct features in S. If ${ \bf v } _ { t } \notin \ S _ { t } ^ { \star }$ , the frame is discarded without a perceiver query. Otherwise, the entry whose feature lies outside $S _ { t } ^ { \star }$ is evicted, and the frame is appended as the newest entry.

3) Single-View Perceiver: Each frame that enters the cache is evaluated exactly once by the perceiver $h _ { \theta } .$ , a VLM that runs on local hardware and is fine-tuned as described in Sec. III-D3. The perceiver receives the full frame $I _ { t }$ and the target category $c ,$ so that the scale, occlusion, and context of the candidate remain visible. A constrained prompt restricts its output to the three labels. Their probabilities, renormalized over $\mathcal { V } ,$ form $\mathbf { p } _ { t } = h _ { \theta } ( I _ { t } , c ) \in \mathbb { R } ^ { 3 }$

4) Cross-View Fusion: The fusion MLP $g _ { \omega }$ receives the probability vectors of all cached views, which together retain how confident each view is and whether the views agree. Its input $\mathbf { P } _ { t }$ stacks these vectors and fills the $K - m _ { t }$ empty slots with zero rows,

$$
\mathbf { P } _ { t } = \left[ \mathbf { p } ^ { ( 1 ) } , \ \ldots , \ \mathbf { p } ^ { ( m _ { t } ) } , \ \mathbf { 0 } , \ \ldots , \ \mathbf { 0 } \right] ^ { \top } \in \mathbb { R } ^ { K \times 3 } .\tag{4}
$$

As every cached probability vector sums to one, a zero row marks an empty slot without a separate indicator. The MLP $g _ { \omega }$ maps the flattened input vec $( \mathbf { P } _ { t } ) \in \mathbb { R } ^ { 3 K }$ to logits $\mathbf { z } _ { t }$ with one entry $z _ { t , a }$ per label $a \in \mathcal { V }$ , and the fused decision is the label with the largest logit,

$$
\mathbf { z } _ { t } = g _ { \omega } \big ( \mathrm { v e c } ( \mathbf { P } _ { t } ) \big ) \in \mathbb { R } ^ { 3 } , \qquad y _ { t } = \arg \operatorname* { m a x } _ { a \in \mathcal { Y } } z _ { t , a } .\tag{5}
$$

## D. Training the Verification Stack

1) Evidence-Sufficiency Labeling: We assign training labels to Grounded-SAM detections, so that the training samples carry the same localization errors and imperfect masks as the candidates encountered at deployment. Each detection $( I , B , M , D )$ comprises the frame, box, mask, and depth image of Sec. III-C1 with the time index omitted. It is assessed along two axes. Semantic identity is derived from simulator annotations, whereas evidence sufficiency is computed from the detection alone.

Each detection is obtained by prompting Grounded-SAM with a target category c. Let $G _ { c }$ be the pixels annotated as c in the simulator, and let $| \cdot |$ count pixels. The semantic overlap is

$$
p _ { \mathrm { t a r g e t } } ( M , G _ { c } ) = \frac { | M \cap G _ { c } | } { | M | } .\tag{6}
$$

An overlap of at least $\eta _ { + }$ indicates a target, an overlap below $\eta _ { - }$ indicates a non-target, and intermediate values leave the identity ambiguous.

Evidence sufficiency is measured by three quantities that correspond to the view conditions of Sec. III-C: the maskto-image ratio $r _ { \mathrm { i m g } } = | M | / | I |$ for image scale, the box-fill ratio $r _ { \mathrm { f i l l } } = | M | / | B |$ for truncation or occlusion, and the mean mask depth $\textstyle { \bar { d } } = \sum _ { u \in M } D ( u ) / | M |$ for viewing distance. The sufficiency predicate requires all three quantities to satisfy their thresholds,

$$
q ( I , M , \mathcal { B } , D ) = \mathbf { 1 } [ r _ { \mathrm { i m g } } \geq \alpha ] \mathbf { 1 } [ r _ { \mathrm { f i l l } } \geq \beta ] \mathbf { 1 } [ \bar { d } \leq \gamma ] ,\tag{7}
$$

where the indicator 1[·] equals 1 if its condition holds and 0 otherwise. The labeling function then assigns the label ℓ by checking sufficiency before identity,

$$
\ell = \left\{ \begin{array} { l l } { \mathrm { u n c e r t a i n , } } & { q = 0 , } \\ { \tt y e s , } & { q = 1 \land p _ { \mathrm { t a r g e t } } \geq \eta _ { + } , } \\ { \mathrm { n o , } } & { q = 1 \land p _ { \mathrm { t a r g e t } } < \eta _ { - } , } \\ { \mathrm { d i s c a r d , } } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{8}
$$

where $\eta _ { \pm } , \alpha , \beta ,$ and $\gamma$ are the thresholds of the labeling function. Table I summarizes the four regimes of this assignment.

TABLE I  
EVIDENCE-AWARE LABEL ASSIGNMENT.
<table><tr><td></td><td>Sufficient Evidence</td><td>Insufficient Evidence</td></tr><tr><td>Target candidate</td><td>yes</td><td>uncertain</td></tr><tr><td>Non-target candidate</td><td>no</td><td>uncertain</td></tr></table>

2) Detector-Aligned Data and Render Augmentation: Natural training data come from collection trajectories that follow shortest paths through HM3D [20] training scenes in Habitat [48]. Grounded-SAM runs on sampled frames, and each detection is labeled by Eq. (8). These data populate the four regimes of Table I unevenly. Most non-target detections are easy to reject, whereas sufficient views of confusers are scarce and insufficient views of confusers are rarer still. We therefore supplement these two regimes with rendered confusers, such as a sofa for the target bed or a stool for the target chair (Verifier Training panel of Fig. 2). Each confuser is rendered with varied camera distance, azimuth, image-plane placement, and occlusion, so that it is observed in both sufficient and insufficient views. Rendered frames pass through the same detector and labeling function as natural frames. A rendered frame therefore yields a training sample only when Grounded-SAM identifies the confuser as a candidate for the target category, and the sample is labeled from the detected mask.

3) Rationale-Supervised Adaptation and Refinement: The perceiver is adapted from Qwen2.5-VL-7B-Instruct [49] with LoRA [14] on the resulting single-view corpus. Its training target is a short rationale followed by the tri-state label. The rationale names the category-relevant attributes visible in the frame and states whether the view suffices for a decision, so the supervision covers both axes of the labeling function. Since only the label is decoded at deployment, the rationale adds no inference latency.

To limit queries to a closed-source teacher, we first prompt the base Qwen model to generate a rationale $r _ { i } ^ { \mathrm { b a s e } }$ and a prediction $\boldsymbol { \hat { \ell } } _ { i }$ for each sample i from its frame $I _ { i }$ and category $c _ { i } .$ , without access to its label $\ell _ { i } .$ . A correct prediction can still be accompanied by a malformed or label-inconsistent rationale. A rule-based validator V therefore checks each base rationale for format, length, label consistency, and prohibited patterns, and outputs 1 only if all checks pass. The rationale used for training is

$$
r _ { i } = \left\{ \begin{array} { l l } { r _ { i } ^ { \mathrm { b a s e } } , } & { \hat { \ell } _ { i } = \ell _ { i } \wedge V \big ( r _ { i } ^ { \mathrm { b a s e } } , \ell _ { i } \big ) = 1 , } \\ { f _ { \mathrm { e x t } } ( I _ { i } , c _ { i } , \ell _ { i } ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{9}
$$

where $f _ { \mathrm { e x t } }$ queries GPT-4o to generate the rationale from the image, the category, and the label.

In the first round, the perceiver is trained with autoregressive cross-entropy over the rationale and label tokens. A hardexample refinement round then repeats the LoRA fine-tuning at a reduced learning rate. Its training set is drawn from the training split and combines every sample that the first-round perceiver misclassifies with correctly classified samples at a fixed ratio.

4) Fusion MLP Training: The fusion MLP is trained with the perceiver frozen. When a candidate stays tracked over several frames of a collection trajectory (Sec. III-D2), these frames pass through the diversity gate of Sec. III-C2, and the retained views form a multi-view training sequence. For sequence $i ,$ the perceiver probabilities of these views are stacked into $\mathbf { P } _ { i }$ as in Eq. (4). Each sequence is labeled by a best-evidence rule over the sufficiency values $q _ { j }$ of its retained views, given by Eq. (7). A sequence without a sufficient view $( \operatorname* { m a x } _ { j } q _ { j } \ = \ 0 )$ is labeled uncertain. Otherwise, it is labeled yes or no by applying Eq. (8) to its sufficient views. Sequences whose sufficient views yield ambiguous or conflicting identities are discarded. Since a single sufficient view makes the sequence label decisive, the MLP is trained to let that view outweigh the insufficient views cached with it. The training objective is three-class cross-entropy between $g _ { \omega } ( \mathrm { v e c } ( \mathbf { P } _ { i } ) )$ and the sequence label.

## E. Active Approach Policy

During the approach phase, the approach policy $\pi _ { \phi }$ generates waypoints toward the candidate. Since shortest paths often pass close to walls and corners, where walls can block the line of sight to the candidate, the policy imitates expert trajectories smoothed from maximum-clearance paths.

1) Privileged Expert Trajectory Construction: Each expert demonstration starts from a simulated state in which a target instance is detected as a candidate whose view fails the sufficiency predicate of Eq. (7). The goal of the expert is obtained by back-projecting the candidate mask and depth to a target position and projecting this position onto the navigable plane. The expert has privileged access to the ground-truth navigability map (Policy Training panel of Fig. 2), on which it plans a piecewise-linear path to the goal along the Voronoi medial axis of the free space. A Bezier curve fitted to this´ path yields a smooth trajectory, from which waypoints are sampled at a fixed spacing. Sliding windows of H consecutive waypoints form the expert chunks. The chunk $\mathbf { A } _ { t } ^ { 0 } \in \mathbb { R } ^ { H \times 2 }$ that starts at step t stacks the planar offsets of its waypoints in the egocentric frame of the agent at step t.

2) Mask-and-Depth-Conditioned Diffusion Transformer: At deployment, the policy uses no map and observes only the tracked mask $M _ { t }$ and the aligned depth image $D _ { t }$ , which encode the relative position, scale, and visibility of the candidate with respect to the agent. These inputs are concatenated channel-wise and processed by encoder $E _ { \psi }$ into the condition ${ \bf e } _ { t } ~ = ~ E _ { \psi } ( [ M _ { t } ; D _ { t } ] )$ . Appearance reasoning is left to the perceiver, so the policy takes no RGB input. Omitting RGB is also intended to reduce the sensitivity of the policy to the appearance gap between simulation and reality. As the policy receives no category input, it approaches an uncertain confuser in the same way as an uncertain target, although all demonstrations end at targets.

Similar mask and depth observations in different room layouts can correspond to geometrically distinct expert chunks, and direct regression would average them. We therefore model the distribution of expert chunks conditioned on $\mathbf { e } _ { t }$ with a Diffusion Transformer [16], [17]. Each chunk ${ \bf A } _ { t } ^ { 0 }$ is paired with the condition $\mathbf { e } _ { t }$ computed at its start step t. The transformer $\epsilon _ { \phi }$ predicts the Gaussian noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ added to ${ \bf A } _ { t } ^ { 0 }$ at diffusion timestep k and is trained jointly with $E _ { \psi }$ by minimizing

$$
\mathcal { L } _ { \mathrm { D P } } ( \phi , \psi ) = \mathbb { E } _ { \mathbf { A } _ { t } ^ { 0 } , k , \epsilon } \left[ \left| \left| \epsilon - \epsilon _ { \phi } ( \mathbf { A } _ { t } ^ { k } , k , \mathbf { e } _ { t } ) \right| \right| _ { F } ^ { 2 } \right] ,\tag{10}
$$

where ${ \bf A } _ { t } ^ { k } = \sqrt { \bar { \alpha } _ { k } } { \bf A } _ { t } ^ { 0 } + \sqrt { 1 - \bar { \alpha } _ { k } } \epsilon$ is the noised chunk and $\bar { \alpha } _ { k }$ is the cumulative noise schedule. At inference, reverse diffusion conditioned on the current mask and depth samples a chunk $\mathbf { A } _ { t } \sim \pi _ { \phi } ( \cdot \mid M _ { t } , D _ { t } )$ , and a low-level waypoint controller converts its waypoints into discrete movement actions.

## F. Closed-Loop Active Verification

Algorithm 1 summarizes the inference procedure for one candidate event. Throughout the event, the perceiver evaluates each frame admitted by the diversity gate, and the fusion MLP updates the fused decision $y _ { t }$ from the evidence cache (Sec. III-C). If $y _ { t }$ is yes or no, the agent stops at the candidate or rejects it, respectively. Otherwise, the agent enters the approach phase and follows a chunk ${ \bf A } _ { t }$ sampled from the approach policy $\pi _ { \phi }$ (Sec. III-E). Since verification continues during chunk execution, the agent cancels the remaining waypoints and acts on y<sub>t</sub> as soon as a newly retained view makes it decisive. When the agent completes a chunk while $y _ { t }$ remains uncertain, the approach policy samples the next chunk from the current mask and depth. Once the mean mask depth ${ \bar { d } } _ { t }$ of $M _ { t }$ falls below the close-range threshold $d _ { \mathrm { c l o s e } } ,$ moving closer is unlikely to reveal substantially more detail. Therefore, if $y _ { t }$ is still uncertain at this range, the agent applies a close-range fallback that selects whichever of yes and no has the larger logit in $\mathbf { z } _ { t }$

## IV. EXPERIMENTS

We first report ObjectNav performance across systems, then analyze closed-loop verification and its components, and finally examine efficiency and remaining failures.

## A. Experimental Setup

Datasets and evaluation splits. We evaluate navigation in Habitat on the standard validation episodes of Gibson [50] (1,000 episodes), HM3D v1 (2,000), and HM3D v2 [20], [51] (1,000). The verification stack is trained, refined, validated, and tested on scene-disjoint HM3D splits, and the approach policy is trained on expert demonstrations from the same training scenes. MP3D [52] uses a 21-category protocol, whereas the stack is trained on the six categories shared by Gibson and HM3D, so MP3D serves only for the candidatelevel distribution-shift evaluation of Table VI.

Metrics. At the episode level, we report success rate (SR) and Success weighted by Path Length (SPL) [53]; closed-loop results are means over three evaluation seeds. Every failed episode is attributed to exactly one cause, in priority order: false stop (FS), where the episode terminates by accepting a non-target; false reject (FR), where at least one truetarget candidate is detected and then rejected or abandoned; exploration failure, where no true target is detected within the budget; or other, covering timeouts with a pending candidate and collision-induced termination. Episode-level FS and FR are therefore exclusive shares of all episodes. Candidate-level FS and FR are instead class-conditional rates: FS is the fraction of non-target candidate events accepted with yes, and FR is the fraction of target candidate events rejected with no; events that end in abandonment or episode termination with a pending candidate remain in the denominator but enter neither numerator. In the approach phase, a candidate is correctly resolved only if a correct yes/no decision is reached before the close-range fallback; fallback-triggered decisions, track loss, abandonment, and episode termination with a pending candidate do not count as correct resolution.

Algorithm 1 Closed-Loop Active Verification   
Require: target category $c ;$ RGB-D frames $( I _ { t } , D _ { t } ) ;$ trained   
$h _ { \theta } , \ g _ { \omega } , \ \pi _ { \phi } ;$ cache capacity K; view-diversity threshold   
$\delta _ { \mathrm { v i e w } } ;$ fallback depth $d _ { \mathrm { c l o s e } }$   
1: Grounded-SAM detects a candidate for c in $I _ { t } ; \mathcal { K } _ { t - 1 }  \emptyset$   
2: while ByteTrack keeps tracking the candidate do   
3: Refresh $M _ { t }$ with SAM from the tracked box; extract   
the CLIP feature $\mathbf { v } _ { t }$ of $I _ { t } ; \mathcal { K } _ { t } \gets \mathcal { K } _ { t - 1 }$   
4: if $( m _ { t - 1 } ~ = ~ 0$ or $\rho _ { t } ~ > ~ \delta _ { \mathrm { v i e w } } )$ and $( m _ { t - 1 } < K$ or   
$\mathbf { v } _ { t } \in S _ { t } ^ { \star } )$ then {Eqs. (2), (3)}   
5: If $m _ { t - 1 } = K ,$ , evict the entry outside $S _ { t } ^ { \star }$ from $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$   
6: $\mathbf { p } _ { t }  h _ { \theta } ( I _ { t } , c ) ;$ append $\left( I _ { t } , \mathbf { v } _ { t } , \mathbf { p } _ { t } \right)$ to $\textstyle { \mathcal { K } } _ { t }$   
7: end if   
8: $\mathbf { z } _ { t } \gets g _ { \omega } ( \mathrm { v e c } ( \mathbf { P } _ { t } ) ) ; y _ { t } \gets \arg \operatorname* { m a x } _ { a \in \mathcal { V } } z _ { t , a } \ \big \{ \mathrm { E q . ~ } ( 5 ) \big \}$   
9: if y = uncertain and $\bar { d } _ { t } < d _ { \mathrm { c l o s e } }$ then   
10: $y _ { t } \gets \arg \operatorname* { m a x } _ { a \in \{ \mathrm { y e s } , \mathrm { n o } \} } z _ { t , a }$ {close-range fallback}   
11: end if   
12: if $y _ { t } = \mathrm { y } \in S$ then   
13: Call the stop action; return   
14: else if $y _ { t } = \mathtt { n o }$ then   
15: Reject the candidate and resume exploration; return   
16: end if   
17: If no waypoints remain, sample $\mathbf { A } _ { t } \sim \pi _ { \phi } ( \cdot \mid M _ { t } , D _ { t } )$   
18: Execute one action toward the next waypoint; t ← t+1   
19: end while   
20: Resume exploration {track lost}

Baselines and system integration. The eight baselines span end-to-end and modular ObjectNav systems based on learned policies, semantic exploration, frontier scoring, value maps, topological reasoning, and target-centric fusion. CLOSER attaches to each as a self-contained verification layer that runs its own Grounded-SAM, ByteTrack, and SAM (Sec. III-F) on the RGB-D stream of the baseline, so every system receives candidates from the same detection-and-tracking pipeline. Exploration, mapping, and the low-level planner remain unchanged, and CLOSER replaces only the native stopping rule. The 500-step budget, action space, and success criterion of each dataset are likewise unchanged, so verification steps, including the approach phase, count against the same budget. Simulator annotations are used only for training labels and privileged expert construction. Systems with an LLM decision module use the same locally deployed Qwen3-30B-A3B-Instruct-2507 model across datasets.

TABLE II IMPLEMENTATION SETTINGS FOR CLOSER.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Target-overlap thresholds  $( \eta _ { + } , \eta _ { - } )$ </td><td>(0.25, 0.05)</td></tr><tr><td>Minimum mask-to-image ratio α</td><td>0.02</td></tr><tr><td>Minimum box-fill ratio β</td><td>0.40</td></tr><tr><td>Maximum mean mask depth γ</td><td>3.0 m</td></tr><tr><td>Cache capacity K</td><td>4 views</td></tr><tr><td>View-diversity threshold  $\delta _ { \mathrm { v i e w } }$ </td><td>0.18</td></tr><tr><td>Cached-view order</td><td>Oldest to newest</td></tr><tr><td>Fusion MLP input</td><td> $4 \times 3$  raw probabilities</td></tr><tr><td>LoRA rank / scaling factor</td><td>64/128</td></tr><tr><td>LoRA target modules</td><td>Attention and MLP projections</td></tr><tr><td>Perceiver SFT learning rate</td><td>2 × 10−5</td></tr><tr><td>Refinement correct-to-misclassified ratio</td><td>1.5</td></tr><tr><td>Expert Bézier degree</td><td>7</td></tr><tr><td>Waypoint horizon H</td><td>8</td></tr><tr><td>Waypoint spacing</td><td>0.25 m</td></tr><tr><td>DDIM denoising steps</td><td>10</td></tr><tr><td>Policy learning rate</td><td>1 × 10−4</td></tr><tr><td>Fallback mean-depth threshold  $d _ { \mathrm { c l o s e } }$ </td><td>1.0 m</td></tr></table>

Implementation details. The perceiver is trained on 78,350 labeled single-view detections from 108 HM3D training scenes (60,030 natural Grounded-SAM detections and 18,320 rendered samples covering the two rare regimes of Sec. III-D2), and the fusion MLP on 8,070 tracked multi-view sequences. Table II lists the remaining settings; the view-diversity threshold $\delta _ { \mathrm { v i e w } }$ is selected on the validation split.

## B. Overall ObjectNav Performance

Table III compares eight ObjectNav systems with their CLOSER variants on Gibson, HM3D v1, and HM3D ${ \bf v } 2 ;$ each “+ CLOSER” row differs from its baseline only in the stopping rule under the protocol of Sec. IV-A. CLOSER improves both SR and SPL in all 24 system–dataset pairs (Fig. 3), with SR gains averaging 3.6 points (up to 4.9) and SPL gains averaging 2.0 points (up to 3.1). Since approach steps count against the unchanged 500-step budget, the consistent SPL gains show that the path spent on additional views is more than repaid by the failures they prevent.

The size of each gain reflects how much candidate verification the baseline already performs. DD-PPO, ZSON, SemExp, L3MVN, and VoroNav stop on detector or policy outputs and gain 3.4–4.9 SR points on each dataset; ESC and VLFM use map-level scoring without a candidate-level check and gain 2.2–3.4 points; ApexNav already fuses passively acquired views of the candidate [58] and gains the least, yet CLOSER still improves its SR and SPL on all three datasets and makes it the strongest system in Table III. This ordering is consistent with attributing the baseline failures to the stopping decision (Sec. IV-C), and the gains over ApexNav indicate that the views gathered actively during the approach phase add evidence that passive acquisition does not provide.

## C. Failure Cause Analysis

We analyze native VLFM on HM3D v2 using the episodelevel failure attribution of Sec. IV-A. Its SR is 63.6%, with 19.4% false stops and 13.8% false rejects. These stopattributable causes sum to 33.2 percentage points of the overall 36.4% failure rate, so the stopping decision accounts for over 90% of the failures; exploration and other failures account for the remaining 3.2 points.

TABLE III  
END-TO-END OBJECTNAV RESULTS. SR/SPL OF EIGHT OBJECTNAV SYSTEMS AND THEIR CLOSER PLUG-IN VARIANTS ON GIBSON, HM3D V1, AND HM3D V2 (OV: OPEN-VOCABULARY). EACH SHADED “+ CLOSER” ROW REPORTS THE ABSOLUTE GAIN OVER ITS BASELINE IN PARENTHESES; CLOSER RESULTS ARE MEANS OVER THREE SEEDS (STANDARD DEVIATIONS ≤0.9 SR AND ≤0.6 SPL POINTS). CLOSER IMPROVES BOTH METRIC FOR EVERY SYSTEM ON EVERY DATASET.
<table><tr><td rowspan="2">Method</td><td rowspan="2">OV</td><td colspan="2">Gibson</td><td colspan="2">HM3D v1</td><td rowspan="2"></td><td colspan="2">HM3D v2</td></tr><tr><td>SR(%) ↑</td><td>SPL(%) ↑</td><td>SR(%) ↑</td><td>SPL(%) ↑</td><td>SR(%) ↑</td><td>SPL(%) ↑</td></tr><tr><td colspan="10">Plug-in comparison</td></tr><tr><td>ZSON [54]</td><td>V</td><td>31.3</td><td>12.0</td><td>25.5</td><td>12.6</td><td></td><td>28.7</td><td>13.1</td></tr><tr><td>+ CLOSER</td><td>√</td><td>35.6(+4.3)</td><td>14.5 (+2.5)</td><td>29.8(+4.3)</td><td>15.1 (+2.5)</td><td></td><td>32.5 (+3.8)</td><td>14.7(+1.6)</td></tr><tr><td>DD-PPO [55]</td><td>X</td><td>34.8</td><td>14.0</td><td>27.9</td><td>14.2</td><td></td><td>33.5</td><td>15.6</td></tr><tr><td>+ CLOSER</td><td>X</td><td>38.9(+4.1)</td><td>16.3 (+2.3)</td><td>32.1 (+4.2)</td><td>15.6(+1.4)</td><td></td><td>36.9(+3.4)</td><td>17.8(+2.2)</td></tr><tr><td>ESC [5]</td><td>V</td><td>66.5</td><td>40.2</td><td>43.4</td><td>24.1</td><td></td><td>51.6</td><td>26.4</td></tr><tr><td>+ CLOSER</td><td>√</td><td>69.9(+3.4)</td><td>42.0(+1.8)</td><td>46.7(+3.3)</td><td>26.0(+1.9)</td><td></td><td>54.8(+3.2)</td><td>28.2(+1.8)</td></tr><tr><td>VoroNav [56]</td><td>V</td><td>72.8</td><td>46.5</td><td>46.2</td><td>28.1</td><td></td><td>54.9</td><td>30.2</td></tr><tr><td>+ CLOSER</td><td>√</td><td>76.9(+4.1)</td><td>48.8(+2.3)</td><td>50.3 (+4.1)</td><td>30.5(+2.4)</td><td></td><td>59.0(+4.1)</td><td>32.5 (+2.3)</td></tr><tr><td>VLFM [6]</td><td>V</td><td>84.0</td><td>52.2</td><td>52.5</td><td>30.4</td><td></td><td>63.6</td><td>32.5</td></tr><tr><td>+ CLOSER</td><td>√</td><td>86.2(+2.2)</td><td>53.7 (+1.5)</td><td>55.4(+2.9)</td><td>32.2(+1.8)</td><td></td><td>66.1 (+2.5)</td><td>34.1 (+1.6)</td></tr><tr><td>L3MVN [57]</td><td>×</td><td>79.5</td><td>39.4</td><td>53.9</td><td>25.2</td><td></td><td>65.1</td><td>28.4</td></tr><tr><td>+ CLOSER</td><td>×</td><td>84.4(+4.9)</td><td>41.8(+2.4)</td><td>58.8(+4.9)</td><td>28.3(+3.1)</td><td></td><td>69.6(+4.5)</td><td>31.2(+2.8)</td></tr><tr><td>SemExp [2]</td><td>X</td><td>54.4</td><td>19.9</td><td>38.2</td><td>19.0</td><td></td><td>44.8</td><td>20.3</td></tr><tr><td>+ CLOSER</td><td>×</td><td>58.9(+4.5)</td><td>22.6(+2.7)</td><td>42.6(+4.4)</td><td>21.6(+2.6)</td><td></td><td>49.1 (+4.3)</td><td>22.9(+2.6)</td></tr><tr><td>ApexNav [58]</td><td>√</td><td>88.6</td><td>55.4</td><td>58.6</td><td>32.4</td><td></td><td>71.2</td><td>35.7</td></tr><tr><td>+ CLOSER</td><td>√</td><td>89.3(+0.7)</td><td>55.6(+0.2)</td><td>60.4(+1.8)</td><td>33.6(+1.2)</td><td></td><td>72.5 (+1.3)</td><td>36.7(+1.0)</td></tr></table>

CLOSER reduces the stop-attributable failures by more than half, from 33.2 to 15.7 points, with false stops falling to 6.4% and false rejects to 9.3%, and SR increases to 66.1%. Preventing an early false stop or false reject allows navigation to continue, and some of the continued episodes subsequently time out, collide, or end with a pending candidate. The exploration/other share therefore rises from 3.2% to 18.2%, and more than half of the remaining failures fall outside the stopping decision that CLOSER addresses.

## D. Qualitative Results

Fig. 4 shows how the closed loop prevents both error types in one successful episode with target category sofa. In the first candidate event, the fused decision on the initial view at step 34 is uncertain, so the agent enters the approach phase. The view retained at step 37 shows that the candidate is a bed; the fused decision becomes no, and the agent discards the candidate and resumes exploration instead of ending the episode with a false stop. In the second event, the initial view at step 63 is likewise deferred, and two further views retained at steps 72 and 75 add evidence during the approach phase until the fused decision becomes yes. The agent then calls the stop action, and the episode succeeds. In the navigation context of Fig. 4(b), the colored segments of the top-down path mark the steps spanned by the two events, and the black segments correspond to exploration by the baseline. Neither map in the panel is an input to CLOSER (Sec. III-E).

Real-world verification. Fig. 5 shows the verification stack deployed on a Unitree Go2 quadruped in two real indoor scenes, running the same components as in Sec. III-F on the onboard camera stream without any fine-tuning on real data. In both scenes, stacked blocks occlude the candidate at the initial viewpoint, so the initial views are deferred; as the robot moves around the blocks, clearer views enter the cache, and the fused decision becomes decisive once the candidate is fully visible. The tri-state verification thus transfers from simulation to real camera streams without adaptation.

## E. Ablation Studies

1) Closed-loop Verification: Table IV compares the native VLFM stopping rule, two binary verifiers, tri-state verification without the approach policy, and the complete system on the same HM3D v2 episodes. Binary GPT-4o-mini follows the API-based verification of prior work [11]; the local binary variant uses the same cache implementation but has no uncertain state, so its first decision terminates the candidate event. The static tri-state variant keeps the candidate tracked while VLFM continues to explore and recomputes the fused decision whenever the cache admits a new view, without a dedicated approach policy; the close-range fallback still applies if VLFM brings the agent within 1 m of the candidate. Cache construction and approach policies are evaluated separately in Tables VII and VIII.

(a)  
![](images/52b5b170a3a7de6b3efad24287b9073451fc0998eb21c7f45b4614cb2aedbe7b.jpg)

(b)  
![](images/49f194ea44a60b9af0c5acf7907c7a5dafa792b4496c04fa3cf5029746c1a5f2.jpg)  
Fig. 3. Paired ObjectNav gains from CLOSER. (a) Absolute SR gains grouped by dataset and by the native candidate-verification mechanism (shading, left to right: none, map-level heuristic, and candidate-level fusion). (b) Changes in SR and SPL for the same 24 system–dataset pairs; each arrow starts at the baseline and ends at its “+ CLOSER” result. Data from Table III.

![](images/a6fe5a3a36ce30b0548002d77480da1144c9447cef591070ead6d4243a16f505.jpg)  
Fig. 4. Closed-loop verification in one successful episode with target category sofa. (a) Evidence timeline of the two candidate events (S: step index). In event A, the initial view is deferred as uncertain, and the view retained at step 37 reveals the candidate to be a bed, so the agent discards the candidate and resumes exploration. In event B, the fused decision remains uncertain at step 72 and becomes yes at step 75, so the agent stops. The quoted text shows the perceiver rationales for the two views of event A, decoded here for illustration. (b) Top-down path and semantic map of the episode; the colored path segments correspond to the timeline arrows in (a).

The binary API and local verifiers reduce false stops from 19.4% to 11.2% and 13.6%, respectively, but improve SR by at most 1.3 points. The local binary variant also increases false rejects from 13.8% to 14.2%, since it has no deferral state and must assign a binary outcome when the available evidence is insufficient.

TABLE IV  
CLOSED-LOOP ABLATION OF THE VERIFICATION LAYER. ALL VARIANTS ARE BUILT ON VLFM AND EVALUATED ON THE HM3D V2 VALIDATION EPISODES. FS / FR / FB: EPISODE-LEVEL FALSE STOP, FALSE REJECT, AND FALLBACK TRIGGER (%).
<table><tr><td>Variant</td><td>SR↑</td><td>SPL↑</td><td>FS↓</td><td>FR↓</td><td>FB↓</td></tr><tr><td>VLFM native stop</td><td>63.6</td><td>32.5</td><td>19.4</td><td>13.8</td><td></td></tr><tr><td>Binary GPT-4o-mini (API)</td><td>64.9</td><td>33.3</td><td>11.2</td><td>12.1</td><td>一</td></tr><tr><td>Binary local</td><td>64.4</td><td>33.0</td><td>13.6</td><td>14.2</td><td>一</td></tr><tr><td>Tri-state + diverse cache (static)</td><td>65.3</td><td>33.6</td><td>8.7</td><td>15.1</td><td>4.2</td></tr><tr><td>+ Diffusion policy (CLOSER)</td><td>66.1</td><td>34.1</td><td>6.4</td><td>9.3</td><td>9.8</td></tr></table>

Adding an uncertain state without the approach policy further reduces false stops to 8.7%, while false rejects increase to 15.1%: deferred target candidates are abandoned when the views that VLFM subsequently observes do not provide sufficient evidence. With the approach policy, false stops and false rejects decrease to 6.4% and 9.3%, and SR/SPL reach 66.1/34.1 (Fig. 6). During the approach phase, the agent acquires additional views of deferred candidates, which accounts for the reduction in false rejects relative to the static tri-state variant. The complete system triggers the close-range fallback in 9.8% of episodes; fallback-induced false stops and false rejects are included in the reported rates.

2) Evidence-aware Verification and View Selection: Table V compares verifier variants on 5,184 held-out Grounded-SAM candidate events from the scene-disjoint HM3D test split, separating semantic decisions on clear observations from deferral decisions on insufficient ones. Binary SFT shares the backbone and corpus of the tri-state models but assigns each insufficient observation its latent semantic label [11]; the threshold variant defers when its maximum label-token probability falls below a threshold selected on the validation split.

Binary SFT reaches 87.4% and 78.6% on clear target and

![](images/22830425c9a3baa5fb3f6add3d17c882d4946dbec5b50f7cad7af29238e645b1.jpg)

Fig. 5. Real-world verification on a quadruped robot. The verification stack runs on the onboard camera stream of a Unitree Go2 in two indoor scenes (one scene per row, frames ordered left to right; insets show the onboard view). Stacked blocks occlude the candidate (a chair, top; an armchair, bottom), so the initial views are deferred as uncertain. As the robot moves around the blocks, clearer views enter the cache, and the fused decision becomes decisive once an unobstructed view is retained (rightmost insets).  
![](images/c7d7fe52385aeabec2f48a85b82dec25f386a671c47d042fa5d35f7d1dbe9ff0.jpg)  
Fig. 6. Episode outcomes of the closed-loop ablation. Each episode is assigned to exactly one outcome; the right column shows the total stopattributable error (FS + FR). Data from Table IV.

## TABLE V

VERIFIER COMPARISON ON 5,184 HELD-OUT GROUNDED-SAM DETECTIONS. CLEAR COLUMNS: CORRECT SEMANTIC DECISIONS ON CLEAR OBSERVATIONS; LOW-Q COLUMNS: CORRECT DEFERRAL OF INSUFFICIENT OBSERVATIONS (ZERO FOR THE PURE BINARY VERIFIER, WHICH CANNOT DEFER). THE LOWER BLOCK REPORTS THE CLOSER TRAINING STAGES, THE FIRST TWO ON SINGLE VIEWS; PARENTHESIZED VALUES GIVE THE GAIN OVER THE PREVIOUS STAGE. ALL VALUES IN %.
<table><tr><td>Verifier / variant</td><td>Clear→ yes ↑</td><td>Low-Q tgt→ unc. ↑</td><td>Clear→ no ↑</td><td>Low-Q cnf→ unc. ↑</td></tr><tr><td>Qwen2.5-VL zero-shot</td><td>68.7</td><td>42.3</td><td>55.1</td><td>39.8</td></tr><tr><td>Binary SFT (yes/no)</td><td>87.4</td><td>0.0</td><td>78.6</td><td>0.0</td></tr><tr><td>Binary SFT + threshold</td><td>79.8</td><td>82.1</td><td>69.4</td><td>80.6</td></tr><tr><td>Direct-label SFT, single view</td><td>83.6</td><td>67.4</td><td>73.2</td><td>65.8</td></tr><tr><td>+ Rendered regimes + rationale 85.9 (+2.3)</td><td></td><td>78.7(+11.3)</td><td>77.1 (+3.9)</td><td>77.3 (+11.5)</td></tr><tr><td>Full CLOSER stack</td><td>89.1 (+3.2)</td><td>85.4(+6.7)</td><td>84.6(+7.5)</td><td>83.8(+6.5)</td></tr></table>

confuser observations but cannot defer; threshold deferral raises the two low-quality rates to 82.1% and 80.6% at the cost of the clear-observation rates, which drop to 79.8% and 69.4%. Among the CLOSER stages, rendered regimes with rationale supervision raise the average low-quality deferral of direct-label SFT from 66.6% to 78.0%, and sequence-level fusion with hard refinement brings the full stack to 85.4% and 83.8% while also improving the clear-observation rates. At the deployment operating point of Table IX, the local stack reaches 86.1% tri-state accuracy, compared with 82.4% for the single-view GPT-4o-mini API comparator.

TABLE VI  
DEFERRAL QUALITY UNDER DISTRIBUTION SHIFT. THE DEFERRAL THRESHOLD OF THE BINARY VERIFIER IS SELECTED ONCE ON THE HM3D  
VALIDATION SPLIT AND KEPT FIXED. DEFER: FRACTION OFINSUFFICIENT-EVIDENCE CANDIDATES CORRECTLY DEFERRED; FS / FR:CANDIDATE-LEVEL FALSE STOP AND FALSE REJECT. ALL VALUES IN %.
<table><tr><td rowspan="2">Evaluation domain</td><td colspan="3">Binary SFT + threshold</td><td colspan="3">CLOSER (tri-state)</td></tr><tr><td>Defer ↑</td><td>FS↓</td><td>FR↓</td><td>Defer ↑ FS ↓1</td><td></td><td>FR↓</td></tr><tr><td>HM3D held-out (in-domain)</td><td>81.4</td><td>12.6</td><td>11.2</td><td>84.6</td><td>7.5</td><td>8.4</td></tr><tr><td>MP3D candidates (cross-dataset)</td><td>63.7</td><td>17.9</td><td>15.8</td><td>82.1</td><td>8.6</td><td>9.3</td></tr><tr><td>Rendered hard regimes</td><td>46.8</td><td>23.4</td><td>20.7</td><td>79.2</td><td>9.8</td><td>10.9</td></tr></table>

Distribution shift. Table VI evaluates threshold deferral and tri-state verification across three domains, with the binary threshold selected once on the HM3D validation split and kept fixed.

From HM3D to MP3D and the rendered regimes, the deferral rate of the thresholded model drops from 81.4% to 63.7% and 46.8%, whereas that of CLOSER drops only from 84.6% to 82.1% and 79.2% (Fig. 7); over the same range, the false-stop rate rises from 12.6% to 23.4% for the threshold but only from 7.5% to 9.8% for CLOSER. The explicitly supervised uncertain state is therefore far less sensitive to domain shift than a fixed confidence threshold.

View selection. Table VII holds the learned components fixed and compares a single latest frame with the latest K frames and CLIP-diverse selection, both with K=4.

The latest-K cache raises tri-state accuracy from 80.7% to 83.5%, and CLIP-diverse selection raises it further to 86.1% while lifting low-quality deferral to 84.6% and lowering candidate-level false stops from 9.2% to 7.5%. Selecting complementary views thus improves the evidence supplied to the fusion MLP at the same cache budget.

![](images/b48eb02eb81ba94f69117d74131ad757f06a11e3f09b798f7172cac8ac53cc62.jpg)

![](images/59b1d91150c1a345d8830d3925ef116fd4f86e16fe9982634c8d42597ed584f1.jpg)  
Evaluation domain (increasing shift → )  
Fig. 7. Deferral under distribution shift. The fixed binary threshold and CLOSER are evaluated on held-out HM3D candidates, MP3D candidates, and the rendered hard regimes of Table I. Annotations show changes from the indomain to the rendered setting; the shaded region shows the gap between the methods. Data from Table VI.

TABLE VII  
ABLATION OF CACHE CONSTRUCTION. THE LEARNED COMPONENTS ARE FIXED; MULTI-VIEW ROWS USE K=4. ACC.: TRI-STATE ACCURACY; LOW-Q: FRACTION OF INSUFFICIENT-EVIDENCE CANDIDATES CORRECTLY DEFERRED; FS: CANDIDATE-LEVEL FALSE STOP. ALL VALUES IN %.
<table><tr><td>Evidence input</td><td>Acc. ↑</td><td>Low-Q ↑</td><td>FS↓</td></tr><tr><td>Latest frame only</td><td>80.7</td><td>78.9</td><td>11.4</td></tr><tr><td>Latest-K frames</td><td>83.5</td><td>81.2</td><td>9.2</td></tr><tr><td>CLIP-diverse (ours)</td><td>86.1</td><td>84.6</td><td>7.5</td></tr></table>

3) Approach Policy: Table VIII compares policies on candidates initially assigned uncertain. All variants share the verifier, evidence rule, and close-range fallback and differ only in how they acquire additional observations; the fallback rate is conditioned on this subset and differs from the episode-level rate in Table IV.

In-place look-around resolves only 32.4% of candidates, and the straight-line policy raises correct resolution to 61.7% at the cost of 2.84 m of extra path and a 37.6% fallback rate; selecting a single re-verification viewpoint as in STRIVE [11] reaches 70.2% with 2.21 m. The deterministic and diffusion policies imitate the same Voronoi–Bezier demonstrations, and´ the diffusion policy raises correct resolution from 76.5% to 83.6%, shortens the extra path from 2.08 m to 1.69 m, and lowers the fallback rate from 18.1% to 12.7% (Fig. 8). It also rejects 78.4% of initially uncertain confusers, although it is trained only on target-present demonstrations.

## F. Efficiency and Training Cost

Table IX compares the single-view API and local perceivers with the complete verification stack. Local measurements use batch size 1 on one NVIDIA A100 40 GB GPU and include visual preprocessing and decoding; API latency is measured end to end. Because the perceiver is queried once per retained view and the MLP fuses the cached probabilities after each cache update, the latency of the stack is reported per retained view.

APPROACH-POLICY COMPARISON ON INITIALLY UNCERTAINCANDIDATES. CORRECT: CORRECT YES/NO DECISION BEFORE THECLOSE-RANGE FALLBACK; CONFUSER: REJECTION RATE ON INITIALLYUNCERTAIN CONFUSERS; FB: FALLBACK TRIGGER RATE ON THISCANDIDATE SUBSET. ALL VALUES EXCEPT PATH ARE IN %.
<table><tr><td>Policy</td><td></td><td></td><td>Correct ↑ Confuser ↑ Path (m) ↓ FB ↓</td><td></td></tr><tr><td>In-place look-around</td><td>32.4</td><td>26.1</td><td>0.00</td><td>0.0</td></tr><tr><td>Straight-line policy</td><td>61.7</td><td>54.3</td><td>2.84</td><td>37.6</td></tr><tr><td>Viewpoint re-verif. (STRIVE) [11]</td><td>70.2</td><td>63.8</td><td>2.21</td><td>25.4</td></tr><tr><td>Voronoi-Bézier det. BC</td><td>76.5</td><td>70.9</td><td>2.08</td><td>18.1</td></tr><tr><td>Voronoi-Bézier diffusion</td><td>83.6</td><td>78.4</td><td>1.69</td><td>12.7</td></tr></table>

![](images/9af1b6c079f99823c53a941579528bf6aee924a5fc085416586c5665db33bc88.jpg)  
Fig. 8. Resolution accuracy, extra path, and fallback rate of approach policies on the initially uncertain candidate subset. Data from Table VIII.

The single-view perceiver requires 0.28 s and 17.2 GB per retained view, compared with 1.38 s and \$0.41 per 1,000 queries for GPT-4o-mini. Fusing up to four cached views adds 0.1 ms and raises tri-state accuracy from 80.7% to 86.1%, above the 82.4% of the API comparator, while leaving perview latency and peak memory unchanged, since each retained view is evaluated exactly once.

Training cost. LoRA adaptation updates 160.8 M parameters (2.1% of the backbone). The perceiver and policy stages require 111 GPU-hours on four A100-80GB GPUs: 76 hours for first-round SFT on the 78,350 single-view samples, 19 hours for hard-example refinement, and 16 hours for diffusionpolicy training on 14,680 demonstration chunks (fusion-MLP training excluded). The triage rule of Eq. (9) queries the GPT-4o teacher for 42.3% of the corpus and reduces annotation cost by 2.4× (\$518 vs. \$1,226).

## G. Remaining Failure Analysis

Table X compares the conditional candidate-level false-stop rates of the detector-only rule, the binary SFT verifier, and CLOSER on four non-target confusion subsets.

For the sofa–bed, stool–chair, and sink–toilet confusions, binary verification reduces false stops from 64.8–79.2% to 42.7– 53.1%, and CLOSER further reduces them to 27.1–35.4%; on the remaining confusions, the rates are 46.3%, 30.2%, and 19.6%. Binary verification handles clear confusers, while the closed loop acquires additional views for candidates that remain uncertain. As each row is conditioned on a different subset, these rates are not comparable with the aggregate 7.5% false-stop rate of Tables VI and VII.

TABLE IX  
VERIFIER RUNTIME AND DEPLOYMENT COST. LOCAL VLM LATENCYAND MEMORY ARE MEASURED ON A SINGLE NVIDIA A100 40 GB GPU;API COST IS IN USD PER 1,000 QUERIES. LATENCY OF THEPERCEIVER–MLP STACK IS PER RETAINED VIEW; EACH MLP FUSIONADDS 0.1 MS.
<table><tr><td>Verifier</td><td>Local</td><td>Views</td><td>Tri-state accuracy ↑</td><td>Latency (s) ↓</td><td>Memory (GB) ↓</td><td>Cost/1k queries ↓</td></tr><tr><td>GPT-4o-mini API (tri-state)</td><td>No</td><td>1</td><td>82.4</td><td>1.38</td><td></td><td>0.41</td></tr><tr><td>Qwen2.5-VL zero-shot</td><td>Yes</td><td>1</td><td>52.6</td><td>0.31</td><td>16.9</td><td>0</td></tr><tr><td>ČLOSER single-view perceiver</td><td>Yes</td><td>1</td><td>80.7</td><td>0.28</td><td>17.2</td><td>0</td></tr><tr><td>CLOSER perceiver + raw-probability MLP</td><td>Yes</td><td>≤ 4</td><td>86.1</td><td>0.28/view</td><td>17.2</td><td>0</td></tr></table>

TABLE X

CANDIDATE-LEVEL FALSE-STOP RATE (%) ON CHARACTERISTIC CONFUSER PAIRS (VLFM+CLOSER, HM3D V2). “A AS B” DENOTES A CANDIDATE OF CATEGORY A DETECTED WHILE SEARCHING FOR TARGET CATEGORY B; “OTHER CONFUSIONS” POOLS THE UNLISTED NON-TARGET CONFUSION TYPES.
<table><tr><td>Confusion</td><td>Detector only</td><td>Binary verifier</td><td>CLOSER</td></tr><tr><td>Sofa as bed</td><td>79.2</td><td>53.1</td><td>35.4</td></tr><tr><td>Stool as chair</td><td>70.6</td><td>45.3</td><td>28.8</td></tr><tr><td>Sink as toilet</td><td>64.8</td><td>42.7</td><td>27.1</td></tr><tr><td>Other confusions</td><td>46.3</td><td>30.2</td><td>19.6</td></tr></table>

On VLFM with CLOSER, the remaining episode-level false stops and false rejects are 6.4% and 9.3%, respectively (Table IV). Qualitative inspection identifies three recurrent modes.

Track switches between adjacent similar instances. Byte-Track can transfer a candidate identity to a neighboring object with a similar appearance. The evidence cache may then mix views of different instances.

Persistent occlusion at close range. When a candidate remains occluded within the 1 m region, the binary fallback makes a decision from insufficient evidence. The episode-level fallback trigger rate is 9.8%, while 19.7% of fallback-triggered decisions on initially uncertain candidates are incorrect; these rates use different denominators.

Confusers with similar appearance and context. Some instances, such as daybeds staged as sofas, remain ambiguous at close range and require cues beyond local appearance.

The remaining 18.2% of episodes belong to the combined exploration/other category. They include episodes in which no target is detected within the budget, timeouts with a pending candidate, and collision-induced termination.

## V. CONCLUSION

We presented CLOSER, an evidence-aware active visual verification framework that treats the terminal decision of object-goal navigation as a closed-loop evidence acquisition problem rather than a one-shot binary filter. A CLIP gate maintains four complementary views, a locally deployable single-view VLM reports their raw tri-state probabilities, and a lightweight MLP learns when the chronologically ordered cache supports a category commitment; a privilegedto-observation Diffusion Transformer learns how to acquire a more informative view from only the tracked mask and depth. Inserted as a plug-in layer, CLOSER consistently improves success rate and path efficiency across heterogeneous ObjectNav harnesses; ablations show that the full gain requires the closed loop, since deferral without resolution reduces false stops but raises false rejects. Looking ahead, extending evidence-aware deferral from candidate verification to exploration commitments, and validating the mask-anddepth policy on physical platforms, are natural next steps toward irreversible decisions under partial observability.

## ACKNOWLEDGMENT

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## REFERENCES

[1] Y. Liu et al., “Aligning cyber space with physical world: A comprehensive survey on embodied AI,” IEEE/ASME Trans. Mechatronics, vol. 30, no. 6, pp. 7253–7274, 2025.

[2] D. S. Chaplot, D. Gandhi, A. Gupta, and R. Salakhutdinov, “Object goal navigation using goal-oriented semantic exploration,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 4247–4258.

[3] R. Fukushima, K. Ota, A. Kanezaki, Y. Sasaki, and Y. Yoshiyasu, “Object memory transformer for object goal navigation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2022, pp. 11 288–11 294.

[4] J. Sun, J. Wu, Z. Ji, and Y.-K. Lai, “A survey of object goal navigation,” IEEE Trans. Autom. Sci. Eng., vol. 22, pp. 2292–2308, 2025.

[5] K. Zhou et al., “ESC: Exploration with soft commonsense constraints for zero-shot object navigation,” in Proc. Int. Conf. Mach. Learn. (ICML), 2023.

[6] N. Yokoyama, S. Ha, D. Batra, J. Wang, and B. Bucher, “VLFM: Visionlanguage frontier maps for zero-shot semantic navigation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 42–48.

[7] Y. Song, A. Nguyen, and C.-Y. Lee, “Learning to terminate in object navigation,” in Proc. Asian Conf. Mach. Learn. (ACML), 2024, pp. 1247–1262.

[8] J. Li et al., “RATE-Nav: Region-aware termination enhancement for zero-shot object navigation with vision-language models,” in Findings Assoc. Comput. Linguist.: ACL, 2025, pp. 6564–6574.

[9] T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, “YOLO-World: Real-time open-vocabulary object detection,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16 901–16 911.

[10] T. Ren et al., “Grounded SAM: Assembling open-world models for diverse visual tasks,” arXiv preprint arXiv:2401.14159, 2024.

[11] H. Zhu et al., “STRIVE: Structured representation integrating VLM reasoning for efficient object navigation,” arXiv preprint arXiv:2505.06729, 2025.

[12] E. Zorzi et al., “Benchmarking interaction, beyond policy: a reproducible benchmark for collaborative instance object navigation,” arXiv preprint arXiv:2604.00265, 2026.

[13] H.-T. Su, T.-J. Wang, J.-F. Yeh, M. Sun, and W. H. Hsu, “VLN-NF: Feasibility-aware vision-and-language navigation with false-premise instructions,” in Proc. 64th Annu. Meeting Assoc. Comput. Linguist. (ACL), 2026, pp. 25 134–25 150.

[14] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[15] Y. Zhang et al., “ByteTrack: Multi-object tracking by associating every detection box,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022, pp. 1–21.

[16] C. Chi et al., “Diffusion policy: Visuomotor policy learning via action diffusion,” Int. J. Robot. Res., vol. 44, no. 10–11, pp. 1684–1704, 2025.

[17] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023.

[18] B. Chen et al., “Think holistically, act down-to-earth: A semantic navigation strategy with continuous environmental representation and multi-step forward planning,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 5, pp. 3860–3875, 2024.

[19] X. Hu, Y. Lin, S. Wang, Z. Wu, and K. Lv, “Agent-centric relation graph for object visual navigation,” IEEE Trans. Circuits Syst. Video Technol., vol. 34, no. 2, pp. 1295–1309, 2024.

[20] K. Yadav et al., “Habitat-matterport 3D semantics dataset,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023.

[21] Y. Zhu et al., “ChatNav: Leveraging LLM to zero-shot semantic reasoning in object navigation,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 3, pp. 2369–2381, 2025.

[22] Y. Cao et al., “CogNav: Cognitive process modeling for object goal navigation with LLMs,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 9550–9560.

[23] Y. Shi, J. Liu, L. Sun, and X. Zheng, “E<sup>2</sup>BA: Environment exploration and backtracking agent for visual language object navigation,” IEEE Trans. Circuits Syst. Video Technol., vol. 35, no. 7, pp. 6231–6244, 2025.

[24] F. Wu, W. Zuo, W. Yang, J. Xiao, Y. Liu, and X. Zeng, “ReMemNav: Memory-based decision correction and target verification for zero-shot object navigation,” arXiv preprint arXiv:2603.26788, 2026.

[25] H. Yin, X. Xu, L. Zhao, Z. Wang, J. Zhou, and J. Lu, “UniGoal: Towards universal zero-shot goal-oriented navigation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 19 057–19 066.

[26] S. Yuan, H. Zhang, L. Liu, L. Zhu, and X. Dong, “SPENav: Dynamic object filtering with spatial perception enhancement for vision-language navigation,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 5, pp. 6905–6919, 2026.

[27] Z. Qin, L. Wang, Y. Wang, S. Zhou, G. Hua, and W. Tang, “RSRNav: Reasoning spatial relationship for image-goal navigation,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 8, pp. 12 492–12 503, 2026.

[28] X. Lei, M. Wang, W. Zhou, L. Li, and H. Li, “Instance-aware exploration-verification-exploitation for instance ImageGoal navigation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16 329–16 339.

[29] M. Gao et al., “DRIVE-Nav: Directional reasoning, inspection, and verification for efficient open-vocabulary navigation,” arXiv preprint arXiv:2603.28691, 2026.

[30] W. Dai et al., “InstructBLIP: Towards general-purpose vision-language models with instruction tuning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2023.

[31] G. Li et al., “Evolving the prototype journey: Lifelong vision-andlanguage navigation with prototype adaptation,” IEEE Trans. Circuits Syst. Video Technol., 2026, early access.

[32] Y. Cai, X. He, M. Wang, H. Guo, W.-Y. Yau, and C. Lv, “CL-CoTNav: Closed-loop hierarchical chain-of-thought for zero-shot object-goal navigation with vision-language models,” arXiv preprint arXiv:2504.09000, 2025.

[33] Z. Zhang, G. P. Meyer, Z. Lu, A. Shrivastava, A. Ravichandran, and E. M. Wolff, “VLM-KD: Knowledge distillation from VLM for longtail visual recognition,” arXiv preprint arXiv:2408.16930, 2024.

[34] L. H. Li, J. Hessel, Y. Yu, X. Ren, K.-W. Chang, and Y. Choi, “Symbolic chain-of-thought distillation: Small models can also “think” step-bystep,” in Proc. 61st Annu. Meeting Assoc. Comput. Linguist. (ACL), 2023, pp. 2665–2679.

[35] Z. Chen, Y. Ge, Z. Wang, P. Cao, and L. Yang, “Query, align, and distill: Navigation-aware cross-modal interaction for efficient vision-andlanguage navigation,” IEEE Trans. Circuits Syst. Video Technol., 2026, early access.

[36] Y. Chen, K. Sikka, M. Cogswell, H. Ji, and A. Divakaran, “Measuring and improving chain-of-thought reasoning in vision-language models,” in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguist. (NAACL), 2024, pp. 192–210.

[37] Y. Hu et al., “Visual program distillation: Distilling tools and programmatic reasoning into vision-language models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 9590–9601.

[38] Z. Zhong et al., “Spatial-aware and viewpoint-robust vision-language navigation,” IEEE Trans. Circuits Syst. Video Technol., vol. 36, no. 8, pp. 11 717–11 728, 2026.

[39] M. Breyer, L. Ott, R. Siegwart, and J. J. Chung, “Closed-loop next-bestview planning for target-driven grasping,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2022, pp. 1411–1416.

[40] X. Chen, Q. Li, T. Wang, T. Xue, and J. Pang, “GenNBV: Generalizable next-best-view policy for active 3D reconstruction,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16 436–16 445.

[41] L. Jin, X. Chen, J. Ruckin, and M. Popovi¨ c, “NeU-NBV: Next best view´ planning using uncertainty estimation in image-based neural rendering,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2023, pp. 11 305–11 312.

[42] A. Z. Ren et al., “Robots that ask for help: Uncertainty alignment for large language model planners,” in Proc. Conf. Robot Learn. (CoRL), 2023, pp. 661–682.

[43] D. S. Chaplot, M. Dalal, S. Gupta, J. Malik, and R. Salakhutdinov, “SEAL: Self-supervised embodied active learning using exploration and 3D consistency,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2021.

[44] U. Bajpai, J. Ruckin, C. Stachniss, and M. Popovi ¨ c, “Uncertainty-´ informed active perception for open vocabulary object goal navigation,” in Proc. Eur. Conf. Mobile Robots (ECMR), 2025, pp. 1–7.

[45] A. Sridhar, D. Shah, C. Glossop, and S. Levine, “NoMaD: Goal masked diffusion policies for navigation and exploration,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2024, pp. 63–70.

[46] A. Kirillov et al., “Segment anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 4015–4026.

[47] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn. (ICML), 2021, pp. 8748–8763.

[48] M. Savva et al., “Habitat: A platform for embodied AI research,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 9339–9347.

[49] S. Bai et al., “Qwen2.5-VL technical report,” arXiv preprint arXiv:2502.13923, 2025.

[50] F. Xia, A. R. Zamir, Z. He, A. Sax, J. Malik, and S. Savarese, “Gibson env: Real-world perception for embodied agents,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 9068–9079.

[51] S. K. Ramakrishnan et al., “Habitat-matterport 3D dataset (HM3D): 1000 large-scale 3D environments for embodied AI,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS) Datasets and Benchmarks Track, 2021.

[52] A. Chang et al., “Matterport3D: Learning from RGB-D data in indoor environments,” in Proc. Int. Conf. 3D Vis. (3DV), 2017, pp. 667–676.

[53] P. Anderson et al., “On evaluation of embodied navigation agents,” arXiv preprint arXiv:1807.06757, 2018.

[54] A. Majumdar, G. Aggarwal, B. Devnani, J. Hoffman, and D. Batra, “ZSON: Zero-shot object-goal navigation using multimodal goal embeddings,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2022.

[55] E. Wijmans et al., “DD-PPO: Learning near-perfect PointGoal navigators from 2.5 billion frames,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2020.

[56] P. Wu et al., “VoroNav: Voronoi-based zero-shot object navigation with large language model,” in Proc. Int. Conf. Mach. Learn. (ICML), 2024, pp. 53 757–53 775.

[57] B. Yu, H. Kasaei, and M. Cao, “L3MVN: Leveraging large language models for visual target navigation,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2023.

[58] M. Zhang et al., “ApexNav: An adaptive exploration strategy for zeroshot object navigation with target-centric semantic fusion,” IEEE Robot. Autom. Lett., 2025.