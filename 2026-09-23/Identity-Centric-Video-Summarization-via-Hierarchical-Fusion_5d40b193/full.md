# Identity-Centric Video Summarization via Hierarchical Fusion of Biometric, Appearance, and 3D Body Features<sup>1</sup>

Milad Mirjalili<sup></sup>, Enrique Alegre Gutiérrez, Eduardo Fidalgo Fernández, Víctor González Castro, Rocío Alaiz Rodríguez, Manuel Castejón Limas

Group for Vision and Intelligent Systems, Institute for Research and Innovation in Engineering (I4), Universidad de León, 24071, León, Spain

## Abstract

This work presents a video summarization algorithm based on multi-object tracking and person reidentification. We integrate facial embeddings, 3D body-shape features, and visual appearance into a unified tracking framework. These representations enable hierarchical identity assignment and tracking through bidirectional anchoring, which robustly recovers trajectories under severe occlusion or low visual quality. From these stable trajectories, we generate a compact set of summaries for each identity. We select keyframes using a multi-factor weighting scheme that optimizes biometric clarity, social interaction, and motion dynamics, while Adaptive Non-Maximum Suppression ensures temporal diversity. Evaluation on a custom dataset demonstrates tracking stability, achieving an IDF1 of 97.89% and a MOTA of 95.79%. Compared to Top-K selection, our algorithm also increases visual diversity by 146%, temporal coverage by 89%, and information retrievability by 3.5%.

Keywords: Video summarization, Multi-object tracking, Biometric feature fusion, Human-centric perception, Computer vision, Deep learning.

## 1. Introduction

The growing availability of cameras and recording devices generates an enormous amount of video data worldwide every day, far exceeding human capacity for manual review and analysis (Meena et al., 2023). This has created a pressing need to automate the creation of concise video summaries that capture the main events and individuals present in long recordings (Argaw et al., 2024; Li et al., 2023). Video summarization (VS) reduces extensive footage to its most essential parts in a structured and compact format (Alaa et al., 2024; Sultani et al., 2018). VS has clear potential in domains such as surveillance, where recordings are simply stored (Tank, 2023) and only a minimal fraction of the footage is actually relevant (Danesh Pazho et al., 2023). A typical surveillance feed captures hours of normal, everyday behavior, with only a small fraction containing anomalous activity or footage involving specific individuals of interest. Manual review of this footage is therefore not only laborious but also prone to error due to operator fatigue, which creates a critical need for automated systems reliable enough to detect meaningful frames and present them in a structured, compact form. VS is also used to generate content for social media (Narwal et al., 2025), films, or sports, adding value (Wistia, 2025) through short, content-rich videos (Gujar, 2024).

The quality of person-centric summarization depends in part on the Multi-Object Tracking (MOT) algorithm employed (Mirjalili et al., 2025). These algorithms aim to detect and track individuals (S. Li et al., 2025) despite occlusions and changes in appearance (Sharghi et al., 2017; Zhang et al., 2022). However, their performance can degrade in crowded and dynamic environments, where challenges such as occlusion, identity switches, and appearance variation persist.

To address this limitation, we propose a method that combines person tracking with identity-centric video summarization, integrating facial biometrics, appearance, and 3D body shape. This combination helps maintain identity consistency even when one feature becomes unreliable (e.g., during severe facial occlusion). We additionally provide a comprehensive evaluation of both the system's tracking performance using standard metrics and the representativeness of the generated summaries.

The main contributions of this work are:

• Robust feature integration: We combine AdaFace (facial biometrics) (Kim et al., 2022), TransReID (visual appearance) (He et al., 2021), and SMPL parameters (3D body shape) (Goel et al., 2023), reducing identification ambiguity in scenarios with occlusion or visual similarity.

Hierarchical association framework: We propose a two-stage tracking strategy that prioritizes highconfidence facial anchors, followed by a matching stage that recovers occluded subjects. This strategy significantly reduces identity switches by ensuring that tracklets are initialized only from biometrically verified detections.

• Dynamic multi-objective cost function: We define a custom association cost function that weighs facial, appearance, body-structure, and spatial costs based on feature availability. This allows the tracker to alternate between matching modes without manual tuning.

• Persistent gallery and trajectory recovery: We maintain a stable representation of each identity and refine trajectories accordingly. This allows the system to close temporal gaps caused by long-term occlusions and ensures narrative continuity in the final summary.

Identity-centric summarization: We generate individual summaries through multi-factor weighting that optimizes biometric clarity, social interaction, and motion dynamics. We additionally introduce Adaptive Non-Maximum Suppression (A-NMS) to improve temporal coverage and produce summaries that balance visual quality and diversity.

## 2. Related work

VS and Multiple Object Tracking (MOT) are broad research fields with diverse methodologies (Argaw et al., 2024; Lee et al., 2025; Saini et al., 2023). In this section, we first review the main categories and common

architectures of VS. We then examine tracking algorithms proposed in the literature. Finally, we analyze prior work that attempts to link summarization with tracking, highlighting the limitations that our work aims to address.

## 2.1. Video summarization paradigms

VS algorithms can be supervised, unsupervised, or weakly supervised (H. Li et al., 2025).

Supervised learning: Model training is guided by frame importance (Apostolidis et al., 2021c; Hsu et al., 2023; Jiang and Lan, 2025; Qaroush et al., 2025). These methods are limited by the scarcity of annotated data and the subjectivity inherent to the annotation process (Meena et al., 2023).

Unsupervised learning: In the absence of annotations, techniques such as Generative Adversarial Networks (GANs) or Reinforcement Learning (RL) are used (Li et al., 2024; Liu et al., 2022; Yuan and Zhang, 2023). In the former, a Generator (G) predicts frame importance while a Discriminator (D) attempts to distinguish the generated summary from a reconstruction of the original video (Apostolidis et al., 2021a). In the latter, agents learn to select frames that maximize a reward function based on representativeness and diversity (Li and Yang, 2021).

Weakly supervised learning: These methods use annotations such as video tags or titles instead of framelevel labels (Ramos et al., 2023). For example, textual descriptors can be used to align video segments with semantic concepts, and the model learns frame importance based on its relevance to the overall content of the video (Argaw et al., 2024).

## 2.2. Query-based and multi-modal summarization

To mitigate the subjectivity of generic summarization (Narwal et al., 2022), query-based approaches adapt the output using text descriptions or reference images (Bhute et al., 2025; Wu et al., 2022). These approaches rely on integrating diverse features and semantically relating different types of data. One approach aligns natural language queries (e.g., “show me the goal”) with visual content (Narasimhan et al., 2021; Xiao et al., 2020). This typically involves using separate encoders for visual frames and text, which are later merged into a shared latent space, or using cross-modal attention (Argaw et al., 2024). Another approach is to generate more context-aware summaries (Zhao et al., 2023) by incorporating audio information.

However, few works explicitly focus on identity-centric summarization, where the query corresponds to the biometric or morphological identity of a specific individual (Mirjalili et al., 2025). While early attempts relied on simple object or face detection filters, extracting only the frames where a face was detected with high confidence (Biswas et al., 2021), these methods lack the ability to differentiate specific individuals or maintain narrative continuity. Our work addresses this gap with a multi-cue tracking method that serves as a query mechanism for filtering and summarizing footage based on individual identities.

## 2.3. Spatiotemporal architectures

Video processing involves capturing both spatial and temporal information (Apostolidis et al., 2021b). Early deep learning approaches used 3D CNNs to extract spatiotemporal features, or hybrid architectures combining 2D CNNs (for spatial features) with RNNs, such as Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRU), to model temporal dependencies (Alomar et al., 2025). More recently, Vision Transformers (ViT) and multimodal Transformers have become state-of-the-art, using self-attention to capture long-range dependencies across the entire video sequence, often outperforming RNNs in modeling long-range dependencies (Tang et al., 2025).

## 2.4. Multi-Object Tracking (MOT) and re-identification

In MOT, the dominant paradigm is tracking-by-detection, in which targets are detected in each frame and subsequently associated into continuous trajectories (Guan et al., 2025). Most deep learning MOT systems are based on this paradigm (Hassan et al., 2023; S. Li et al., 2025).

Algorithms such as SORT (Simple Online and Realtime Tracking) use a Kalman filter to locate objects and the Hungarian algorithm for association, resulting in a simple motion-based tracking framework (Bewley et al., 2016). DeepSORT adds CNN-based appearance embeddings to help preserve identity through occlusions and abrupt motion changes (Wojke et al., 2017). ByteTrack improves tracking by leveraging both high- and low-confidence detections to recover occluded objects and maintain continuity (Zhang et al., 2022). Methods such as DeepOCSORT employ a fusion strategy (Maggiolino et al., 2023) that weighs motion and appearance, prioritizing visual identity when motion is erratic, or spatial overlap when visual features are ambiguous.

Early deep re-identification (ReID) methods relied on CNNs such as ResNet50 (He et al., 2016) to compute appearance descriptors that support identity consistency across frames (Wojke et al., 2017). More recently, they rely on Transformers such as TransReID to capture fine-grained, part-aware visual features (e.g., clothing, accessories) with stronger discriminative power than standard global CNN embeddings (He et al., 2021). ReID performance can degrade for individuals wearing similar clothing or under severe occlusion. Human Mesh Recovery (HMR), which estimates the parameters of a parametric body model (SMPL) (Goel et al., 2023), can provide additional identification cues when visual features are unreliable.

To synthesize this review and establish the context of our contribution, Table 1 presents a structural comparison of standard MOT trackers against our proposal. As the analysis shows, the considered tracking methods primarily rely on kinematic or 2D visual associations and do not explicitly incorporate 3D structural or biometric constraints. Our proposal addresses this gap by incorporating biometric and 3D structural cues for long-term tracking.

## 2.5. Linking tracking and summarization

Existing evaluation protocols and datasets indicate that MOT and VS are often addressed separately (Dendorfer et al., 2021; Otani et al., 2019). VS selects keyframes by optimizing coverage and diversity on datasets such as SumMe and TVSum (Gygli et al., 2014; Yale Song et al., 2015). MOT maintains the trajectory consistency of individual targets on datasets such as MOTChallenge (Dendorfer et al., 2021). Despite this separation, several studies have attempted to combine these two fields to produce human-centric summaries. Video Synopsis extracts moving objects (tubes) and rearranges them temporally to display multiple events simultaneously, condensing hours of footage into minutes (Baskurt and Samet, 2019; Zhang et al., 2024). Other methods employ tube-level scoring, retaining the segments with the highest representativeness (Shoitan et al., 2023). However, these methods may depend on hand-crafted energy functions or generic metrics that do not explicitly maintain identity consistency in complex environments (Li et al., 2018).

Table 1: Structural comparison of standard MOT tracking methodologies against our proposal.
<table><tr><td>Method</td><td>Appearance extraction</td><td>Association mechanism</td><td>Main strengths</td><td>Critical limitations</td></tr><tr><td>al., 2016)</td><td>SORT (Bewley et None. Purely based on linear kinematics.</td><td>Kalman filter + Hungarian algorithm. Assumes linear, constant-velocity motion for spatial association (IoU).</td><td>Extremely fast and efficient.</td><td>Performance degrades substantially during prolonged occlusions.</td></tr><tr><td>DeepSORT (Wojke et al., 2017)</td><td>Yes. Offline CNN network for strategy and Mahalanobis roughly 45% relative to appearance metrics.</td><td>distance to the SORT motion model.</td><td>Adds a cascaded matching Reduces ID switches by SORT; handles short occlusions.</td><td>Slower; trajectories can jump to false positives generated by static background geometry.</td></tr><tr><td>ByteTrack (Zhang et al., 2022)</td><td>Conditional. Only on high- confidence detections.</td><td>BYTE association: associates all boxes, using the low-score ones to recover occluded objects through pure IoU.</td><td>Recovers heavily occluded objects smoothly.</td><td>For low-confidence boxes, visual appearance is unusable, forcing total reliance on kinematics (IoU).</td></tr><tr><td>Deep OC-SORT (Maggiolino et al., 2023)</td><td>Yes (adaptive). Appearance weighting is adjusted dynamically according to confidence.</td><td>Improves the SORT linear motion model by incorporating Momentum Recovery and Camera Motion Compensation (CMC) to associate non- linear trajectories under occlusion.</td><td>Strong performance in complex dynamics; very robust to non-linear motion.</td><td>High computational and architectural complexity; requires tuning multiple modules and operates on 2D image-space cues.</td></tr><tr><td>Our proposal</td><td>Yes (multiple features). Facial biometrics (AdaFace), visual appearance (TransReID), and 3D body shape (SMPL).</td><td>Bi-directional hierarchical tracking: a dynamic cost function adaptively integrates spatial distance, facial error, visual similarity, and 3D geometric consistency.</td><td>Viewpoint invariance; robust recovery after 180° Dependence on a simple turns and severe 3D structure. Optimized for the post-processing stage in high-precision video summary extraction.</td><td>linear kinematic model; occlusions by validating higher computational cost due to the parallel extraction of multiple networks.</td></tr></table>

In a manner analogous to tracking, Table 2 contrasts summary generation across the different approaches found in the literature and our proposal. An analysis of these methodologies reveals that methods relying on isolated 2D attributes or purely kinematic cues remain fragile under complex interactions. Our approach overcomes these paradigms by using biometric and structural validation to extract a narrative summary centered exclusively on the individual of interest.

In our previous work (Mirjalili et al., 2025), we linked person-centric tracking with video summarization, combining intermediate YOLO layers (Varghese and M., 2024), 2D pose landmarks to extract structural motion, and ArcFace embeddings for biometric identity (Deng et al., 2019). Rather than performing strict frame-by-frame tracking, we used a clustering-based approach to group detections into unique identity clusters.

Table 2: Methodological comparison of the main summarization approaches in the literature against our proposal.
<table><tr><td>Approach in the literature</td><td>Main selection method</td><td>Identity handling</td><td>Main strengths</td><td>Limitations</td></tr><tr><td>Traditional video Lan, 2025; Qaroush et al., based on global scene 2025; Li et al., 2024)</td><td>summarization (Jiang and skips redundant segments variance.</td><td>Keyframe extraction that None. Depends on global visual features of the whole frame, without associating subjects.</td><td>Computationally fast; maintains a real timeline without artificial overlaps</td><td>May disrupt action continuity; does not allow the summary to focus on a specific person.</td></tr><tr><td>Filter-based summarization, face/body (Biswas et al., 2021)</td><td>face detection).</td><td>Filtering through isolated Weak. Relies purely on local thresholds (e.g., the point-wise visibility of high-confidence frontal 2D attributes in isolated frames.</td><td>Allows some personalization; reduces duration by showing the resolution, or when the frontally.</td><td>Performance degrades under occlusion, low subject only when visible subject turns their back to the camera.</td></tr><tr><td>Shoitan et al., 2023)</td><td>Extraction and temporal Video synopsis, motion shifting of moving objects tubes (Zhang et al., 2024; (“tubes&quot;) to display them simultaneously over a static background.</td><td>trajectories based on pure kinematics or background subtraction.</td><td>Moderate. Maintains short Can substantially reduce crowded scenes, separate viewing time by displaying sequential events in parallel.</td><td>May produce visually people who are moving together, and lose identity consistency during dense crossings.</td></tr><tr><td>Our proposal</td><td>Keyframe extraction for each detected ID, guided by Adaptive Non- Maximum Suppression (A-NMS).</td><td>Designed for robust identity association. Dynamic cost function anchored in biometrics, visual features, and 3D structure.</td><td>Its selection criteria let the can introduce different summary focus on activity people into a subject&#x27;s level, social interaction, and visual resolution.</td><td>Query-based generation. An error in ID assignment summary; higher computational load.</td></tr></table>

While this previous work demonstrated the feasibility of identity-based summarization, it highlighted several critical limitations that we now aim to address:

• Feature specificity: Intermediate YOLO layers provide less discriminative identity features than dedicated ReID architectures for distinguishing individuals with similar appearance.

2D versus 3D structure: Relying on 2D pose landmarks makes the system susceptible to viewpoint variation. A person's 2D skeleton can differ substantially between side and frontal views, potentially contributing to identity switches. In contrast, 3D body shape provides a view-invariant biometric signature that remains stable regardless of the subject's orientation relative to the camera.

Biometric robustness: Although ArcFace is effective, its fixed angular margin is independent of image quality, which can limit its effectiveness on blurry or noisy inputs. We replace ArcFace with AdaFace, whose adaptive margin is based on image quality and is designed to improve recognition for lowresolution faces commonly encountered in surveillance footage (Kim et al., 2022).

Table 3 summarizes the methodological changes from our preliminary work and their intended effects, summarizing the improvements introduced in feature extraction, temporal association, and summary generation. Notably, while our previous approach generated global, statistical summaries, the robustness of our new architecture enables a query-based summarization paradigm, generating independent, high-quality visual narratives for each individual.

Table 3: Architectural comparison detailing the methodological evolution from the preliminary work to the current proposal.
<table><tr><td>Algorithm component</td><td>Previous work (Mirjalili et al., 2025)</td><td>Proposed method</td><td>Impact / solution provided</td></tr><tr><td>Facial biometrics</td><td>ArcFace. Applies a fixed angular AdaFace, an adaptive margin margin regardless of image quality.</td><td>based on image quality and resolution.</td><td>Improves detection and recognition under occlusion and low-resolution conditions.</td></tr><tr><td>Visual appearance</td><td>Intermediate YOLO layers. Extract very general features.</td><td>TransReID (Vision Transformer).</td><td>Self-attention captures fine- grained detail (clothing, accessories), surpassing global CNN descriptors.</td></tr><tr><td>Body structure</td><td>2D keypoints.</td><td>4D-Humans (3D SMPL parameters).</td><td>Viewpoint invariance. The 2D skeleton fails when the subject turns, while the 3D signature remains stable.</td></tr><tr><td>Association logic</td><td>Direct, static clustering.</td><td>Hierarchical, bi-directional anchoring with a dynamic cost function.</td><td>Reweights association dynamically when the face is hidden, relying instead on body shape and motion.</td></tr><tr><td>Summary selection</td><td></td><td>Individual per-ID summary Global, statistical summary. guided by a multi-factor function and A-NMS.</td><td>Moves from a generic statistical summary to an independent narrative for each person, optimizing biometric clarity, activity, and interactions.</td></tr></table>

## 3. Proposed algorithm

This work extends the framework presented in (Mirjalili et al., 2025) by introducing significant improvements: (i) features that are more robust to occlusion and pose variation, (ii) a hierarchical association scheme for stable tracking, and (iii) an automated, identity-centric summarization method. Our proposal prioritizes the extraction of semantically informative keyframes for social media content and identity-centric analysis. To this end, we adopt a feature-centric architecture rather than a motion-centric one. This choice is intended to reduce unnecessary computation while maintaining the visual quality required in the summaries. The overview of the method is illustrated in Figure 1.

## 3.1. Person localization and detection

The first stage requires precise localization of the individuals in the scene. Given an input video sequence $V = \{ I _ { 1 } , I _ { 2 } , \dots , I _ { T } \}$ , where $I _ { i }$ denotes the i-th frame, we employ an object detection model to identify people. The output of this module is a set of bounding boxes $B _ { t } \ = \ \{ b _ { 1 } , b _ { 2 } , \ldots , b _ { n } \}$ for the individuals detected in each frame, which serves as input for feature extraction.

![](images/4297afa8638968dc21bd1aa5596ce101b0552f2d6661dd0c48ec0a311997a3c6.jpg)  
Figure 1: Overview of the proposed hierarchical tracking and summarization framework. The pipeline begins by extracting features from video frames, where quality filters are applied before computing the AdaFace (biometrics), TransReID (visual appearance), and SMPL (3D body shape) embeddings. The tracking process consists of four hierarchical steps: (1) Face Anchor Clustering uses DBSCAN on high-quality faces to find unique identities; (2) Gallery Construction establishes a robust profile for each cluster; (3) Candidate Matching links the remaining detections (e.g., occluded or back-turned) to these profiles; and (4) Recovery and Refinement handles unmatched tracklets and smooths the trajectories. Finally, the system generates an Identity-centric summary containing selected keyframes and metadata for each person.

## 3.2. Feature extraction

For each detected individual, we extract features representing facial biometrics, visual appearance, and 3D body shape. For computational efficiency, frames without detected people are skipped. We use two stages to verify identity. First, we apply the Sample and Computation Redistribution Face Detector (SCRFD) (Guo et al., 2021), which is designed to provide high recall for small and occluded faces commonly encountered in surveillance footage. This step yields a facial detection score that allows low-confidence detections to be discarded. The face image is then aligned using facial landmarks and fed into AdaFace to extract the identity embedding $f _ { f a c e }$ , using a loss function whose margin is adaptive to image quality (Kim et al., 2022).

Semantic pose filtering: To apply computationally expensive body feature extractors only to meaningful data, we implement a validity filter leveraging the YOLO detector's pose estimation output. We analyze the confidence scores and spatial distribution of the predicted skeletal keypoints to assess structural integrity. This module acts as a gating mechanism, distinguishing valid human poses, including challenging back views or headless torsos, from detector artifacts such as disembodied hands or background clutter. By filtering out detections with sparse or low-confidence keypoints, this step reduces redundant computation and prevents the re-identification gallery from being corrupted with noisy embeddings. For detections that pass this filter, we extract the remaining body-centric features.

For subjects with occluded faces or facing away from the camera, we employ the Vision Transformer-based TransReID (He et al., 2021). Unlike CNN-based approaches, which focus on global texture, this algorithm uses self-attention to capture fine-grained details (e.g., shoes, backpacks, patterns). We extract a robust appearance feature map $f _ { a p p }$ that encodes the subject's clothing and overall body semantics. This design helps maintain the trajectory during short-term facial occlusions.

Finally, to improve robustness to viewpoint variation, we use 4D Humans (HMR 2.0) to reconstruct the 3D human mesh from 2D images (Goel et al., 2023). Specifically, we extract the SMPL β parameters (shape parameters), denoted $f _ { s h a p e }$ . These parameters represent aspects of the person's physical structure, such as height, limb proportions, and girth, which are relatively stable across camera viewpoints. $f _ { s h a p e }$ acts as a geometric consistency constraint, serving as a negative filter to prevent matching individuals with very different morphologies even if their clothing looks similar.

## 3.3. Hierarchical tracking

To maintain consistent identities in challenging scenarios, such as occlusions, $1 8 0 ^ { \circ }$ turns, and side-profile views, we propose a multi-stage, bi-directional association strategy that uses high-confidence biometric anchors.

Step 1: Face anchor clustering: We identify detections with high-quality facial features across the sequence. Rather than relying on a local threshold, we aggregate all valid face embeddings $E _ { f a c e }$ and apply DBSCAN (Ester et al., 1996) to identify unique identities globally.

Step 2: Gallery construction: For each cluster $C _ { k }$ , we establish a gallery entry $G _ { k }$ containing:

• The centroid of the facial embeddings $( \mu _ { f a c e } )$

• A set of K distinct appearance embeddings $( E _ { a p p } )$ , sampled from frames where the subject's face was clearly visible. To ensure diversity, we keep the K most distinctive embeddings (e.g., front, back, side) based on cosine dissimilarity, minimizing feature redundancy.

• The median of the body shape parameters $( \beta _ { m e d i a n } )$ , as a robust structural signature.

Detections within these clusters serve as high-confidence anchors for subsequent association.

Step 3: Candidate matching: We link the remaining detections, including those with occluded or non-frontal faces, to the established anchors using a bidirectional matching strategy. We define a dynamic cost function $C ( d , g )$ , shown in Eq. (1), to compute the dissimilarity between a candidate detection � and a tracked identity �,

$$
\begin{array} { r } { C ( d , g ) = \frac { w _ { a } D _ { a } + w _ { s } D _ { s } + w _ { l } D _ { l } + \alpha _ { f } . w _ { f } D _ { f } } { w _ { a } + w _ { s } + w _ { l } + \alpha _ { f } . w _ { f } } \rho ( \Delta t ) , } \end{array}\tag{1}
$$

where:

$D _ { a } \colon$ cosine distance between the candidate's ReID embedding and the closest match within gallery g.

$D _ { s } \mathrm { : }$ Euclidean distance between the SMPL shape parameters $\beta .$

$D _ { l } \colon$ spatial distance derived from a linear velocity prediction model.

$D _ { f } \colon$ cosine distance for facial embeddings, active only for visible faces (setting $\alpha _ { f } = 1 )$ .

$\rho ( \Delta t )$ : a penalty that increases as the temporal gap (∆�) between frames grows. Empirically, this function is defined as $\rho ( \Delta t ) = 1 . 0 \mathrm { ~ + ~ } \alpha \Delta t$ , with $\alpha = 0 . 0 2$ . This linearly scales the association cost, gradually reducing the probability of linking subjects that have been absent from the scene for extended periods.

The dynamic weighting is governed by the indicator function $\alpha _ { f }$ . When a face is detected $( \alpha _ { f } = 1 )$ , the highconfidence facial score is included in the weighted average. When the subject turns away or the face is occluded $( \alpha _ { f } = 0 )$ , the facial term is dropped, and the weights are automatically redistributed to prioritize $D _ { a }$ $D _ { s } ,$ , and $D _ { l } ,$ , ensuring robust tracking based on body features and motion.

Step 4: Unmatched recovery and post-processing: We process unidentified detections without reliable facial features by clustering their appearance embeddings with DBSCAN to identify additional tracklets. Fragmented tracklets are then merged and trajectories are smoothed. This step unifies fragmented identities by analyzing the temporal and visual proximity of disconnected segments, closing short occlusion gaps, and improving spatiotemporal continuity.

## 3.4. Summary selection

We formulate the summarization of long-duration sequences as an optimization problem: for each unique identity $( I D _ { i } )$ , we seek a subset of keyframes that maximizes semantic information while minimizing

temporal redundancy. To this end, we introduce a frame-level importance score, prioritizing frames according to biometric quality, social context, resolution, and motion dynamics.

Importance scoring function: For a target person P at frame t, the total importance $S ( P , t )$ is defined from four normalized factors, as shown in Eq. (2):

$$
S ( P , t ) = w _ { 1 } Q _ { f a c e } + w _ { 2 } I _ { s o c i a l } + w _ { 3 } Q _ { a p p e a r } + w _ { 4 } M _ { a c t i o n }\tag{2}
$$

where:

$\boldsymbol { Q } _ { f a c e } \mathrm { : }$ scores confidence in face detection, favoring frames in which the subject's face is clearly visible and frontal.

$I _ { s o c i a l } \mathrm { : }$ frames containing social interactions are semantically more relevant. We measure interaction across all concurrent detections and select the maximum value corresponding to the nearest neighbor, as shown in Eq. (3):

$$
I _ { s o c i a l } = \exp { \left( - \frac { \left\| c _ { p } - c _ { j } \right\| _ { 2 } ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } ,\tag{3}
$$

where $c _ { p }$ and $c _ { j }$ are the centroids of the bounding boxes of the target and the nearest neighbor, and σ is a scale parameter. The Gaussian kernel naturally models human social zones, where proximity decays gradually rather than abruptly.

$Q _ { a p p e a r } .$ the ratio between the bounding-box area and the frame area, or pixel density of the subject.   
High values prioritize close-ups with clearly visible detail over distant, low-resolution detections.

$M _ { a c t i o n }$ : the normalized velocity. Let $c _ { t }$ and $c _ { t - 1 }$ be the bounding-box centroids in the current and previous frames, and $\delta _ { m a x }$ an empirical velocity limit. We define $M _ { a c t i o n }$ as shown in Eq. (4):

$$
M _ { a c t i o n } = m i n \{ 1 ; \ S _ { \mathrm { m a x } } ^ { - 1 } | | c _ { t } - c _ { t - 1 } | | _ { 2 } \}\tag{4}
$$

Restricting the velocity to the [0, 1] interval prevents very high-speed movements from dominating the weighted sum. This factor prioritizes frames in which the subject is performing an action (e.g., walking) over stationary frames, adding dynamic context to the summary.

Temporal diversity via adaptive NMS: Selecting frames purely by score tends to produce redundancy, with all selected frames clustering around a single high-quality moment. To ensure temporal coverage, we apply Adaptive Non-Maximum Suppression (A-NMS). Selecting a keyframe at time t with the highest score suppresses the score of all neighboring frames within a temporal window $\left[ { t - \Delta , t + \Delta } \right]$ (with $\Delta =$ 2 seconds). This forces the selection algorithm to identify the next most informative frame from a distinct time segment, ensuring that the storyline covers the entire duration of the track.

Summary generation: For each processed identity, we generate a structured summary package containing:

• Representative thumbnail: the single frame with maximal $Q _ { f a c e }$ , serving as the biometric profile image.

Chronological timeline: a sequence of � keyframes sorted in temporal order that represents the subject's narrative. The timeline can be easily personalized by adjusting the weighting terms; for instance, a user may prioritize social interactions for security investigations, or motion dynamics for sports analytics, by adjusting the corresponding coefficients in the importance function.

• Semantic metadata: a profile card that includes the specific ID, total track duration, a list of concurrent identities derived from the associated detections, and the dominant motion state derived from the average $M _ { a c t i o n } .$

## 4. Experiments and results

Our experiments focused on two key aspects: tracking accuracy and the representative quality of the generated summaries. We evaluated the tracking algorithm, feature set, and post-processing pipeline through an incremental ablation study.

## 4.1. Experimental setup

Datasets: We used Custom-YT, our own dataset extended from Mirjalili et al. (2025) with more processed frames and ground-truth bounding boxes. For reproducibility, it is publicly available in the repository of the Group for Vision and Intelligent Systems (GVIS)<sup>2</sup>. This evaluation was not intended to compete with massive tracking benchmarks, but rather to assess the improvements of this proposal in scenarios representative of the intended application. We therefore use a dataset that presents variability consistent with social media content. Large-scale evaluation is left for future work.

Custom-YT is diverse in its challenges, such as lighting conditions, camera angles, and backgrounds, and is representative of social media content, acting as a demanding “micro-benchmark” for the algorithm. The sequences include: (i) frames with degraded facial quality, simulating video surveillance; (ii) severe occlusions or subjects with their backs to the camera, to evaluate re-identification through visual appearance and 3D shape; (iii) lighting changes; and (iv) multiple subjects disappearing for long periods, to evaluate the consistency of long-term matching. Figure 2 shows examples from the dataset that directly illustrate these visual challenges.

![](images/a811a1582be02e5f9796f45a72231ca294f468be8fc2f4c937ca58fff13c8eca.jpg)  
Figure 2: Visual challenges in Custom-YT. From left to right: the first column illustrates subjects with their backs turned and lighting changes; the second, facial absence and prolonged disappearances used to evaluate long-term matching; the third, low resolution with degraded quality; and the fourth, variable poses and camera angles.

Beyond visual complexity, this selection makes the dataset a rigorous stress test at the quantitative level. Table 4 provides the video statistics, trajectory duration, and facial availability, whose absence (16% of detections) requires the system to rely exclusively on body and appearance features to maintain tracking.

Table 4: Statistics of the dataset used, including the number of videos, resolution range (in pixels), number of frames, number of unique identities (#IDs), average number of people per frame (density), track length (in frames), and facial availability, illustrating the complexity of the subjects to be tracked.
<table><tr><td>#Videos</td><td>Resolution range (px)</td><td>#Frames</td><td>#IDs</td><td>Density</td><td>Track length (min-max)</td><td>Facial availability (%)</td></tr><tr><td>5</td><td>320×240 to 3840×2160</td><td>5664</td><td>39</td><td>1.93</td><td>2 - 1305</td><td>84.07</td></tr></table>

## Evaluation metrics:

Tracking: We report standard MOT metrics. The main one, Higher Order Tracking Accuracy (HOTA) (Luiten et al., 2021), jointly evaluates object detection quality and temporal association. We additionally report Multiple Object Tracking Accuracy (MOTA) (Bernardin and Stiefelhagen, 2008), which reflects detection accuracy considering false positives, false negatives, and identity errors. To evaluate long-term identity consistency, we report the Identification F-Score (IDF1) (Ristani et al., 2016). Finally, we measure the frequency of incorrect trajectory reassignments using the number of Identity Switches (ID Sw).

Video summarization: We evaluated optimization success, information preservation, and spatiotemporal diversity. We compared these metrics against uniform temporal sampling and Top-K selection of the highestquality frames, to analyze the trade-offs between raw quality and representative selection. To verify that high-quality observations are prioritized, we report the average AdaFace confidence and the coverage of the selected bounding boxes. We evaluated the representative power of the summary using a Retrievability metric. Keyframes are treated as a gallery and the remaining frames as queries, evaluating the average cosine similarity between queries and summaries. To avoid redundancy, we also measure the average pairwise cosine distance between ReID embeddings, as well as the temporal coverage.

Implementation details: The critical hyperparameters of the architecture were tuned empirically. In the detection stage, the SCRFD confidence threshold was set to 0.50, while the YOLO threshold was set to 0.45. For the initial clustering of facial anchors, DBSCAN used a maximum distance (eps) of 0.40 with a minimum of 4 samples (min\_samples). In the secondary ReID phase, a stricter DBSCAN was applied (eps = 0.25, min\_samples = 3) to avoid erroneously merging distinct subjects wearing similar clothing. During tracking association, the cost function prioritizes spatial continuity, giving the highest weight to spatial proximity $( w _ { l } = 1 . 5 )$ , followed by facial and visual consistency $( w _ { f } = 1 . 0 , w _ { a } = 1 . 0 )$ , and using 3D body shape as a structural safety constraint $( w _ { s } = 0 . 8 )$ , all evaluated under a matching threshold of 0.70. Finally, for video summary selection, biometric clarity dominates the equation $( w _ { 1 } = 0 . 4 0 )$ , while the contextual factors, social interaction, visual resolution, and motion dynamics, contribute equally with a weight of 0.20 each $( w _ { 2 } = w _ { 3 } = w _ { 4 } = 0 . 2 0 )$

## 4.2. Comparative baselines and ablation study

All experiments used the same base object detection architecture (YOLOv8 Pose) and preprocessing.

## Tracking algorithms:

• Global clustering: Used in Mirjalili et al. (2025). Aggregates features from all detections into a single graph and applies DBSCAN for identity clustering.

• Bi-directional anchoring: Proposed in this work and described in Section 3, it uses anchor-based trajectory initialization, dynamic association costs, and bi-directional propagation.

## Feature sets:

• Legacy features: The baseline set from our previous work, consisting of standard CNN-based appearance embeddings (intermediate YOLO layers), flattened 2D pose keypoints from the YOLO pose model, and ArcFace embeddings.

• Enriched features: The set proposed in this work, comprising TransReID (appearance), 4D-Humans SMPL (3D body structure), and AdaFace (biometrics).

Post-processing: The final stage of the pipeline refines the trajectory, smoothing it by eliminating temporal gaps.

Experimental configurations: We measure the contribution of each component by evaluating five tracking configurations:

• Legacy baseline: Global clustering with the legacy features already used in Mirjalili et al. (2025).

• Feature upgrade: Global clustering with enriched features.

• Tracking upgrade: Bi-directional anchoring with legacy features, to evaluate the impact of the new association strategy alone.

• Proposed combination: Bi-directional anchoring with enriched features, without post-processing.

• Proposed full system: Adds the post-processing module to the proposed combination.

## 4.3. Tracking results

In Table 5, the legacy baseline achieves considerable performance (HOTA 84.36%). However, it proves fragile under occlusion, since identity is not preserved in the intermediate YOLO layers and 2D pose keypoint estimation is unstable, which translates into identity switches (30 ID Sw) in the absence of faces.

The feature upgrade produces an immediate improvement in identity consistency, reducing identity switches to 9. This is due to AdaFace's facial embeddings and their robustness to low quality, together with the reduced ambiguity achieved by the dedicated ReID and 3D shape features in challenging scenes.

In the tracking upgrade, although MOTA and IDF1 showed slight fluctuations compared to the baseline, identity switches were reduced to only two. This suggests that the legacy features lack the discriminative capacity to fully exploit the potential of the algorithm, limiting gains in recall and precision.

The proposed combination improved performance, eliminating identity switches and increasing MOTA to 95.06% and IDF1 to 97.51%. By hierarchically integrating complementary feature spaces, ReID and 3D shape manage to maintain consistency despite the absence of faces.

The proposed full system produced an additional performance gain, increasing HOTA to 85.13% and MOTA to 95.79%. This result indicates that offline refinement recovers missed detections and smooths trajectories without introducing additional identity errors.

Table 5: Comparative baselines and ablation study. We evaluate the incremental contribution of each component on our custom dataset (Custom-YT). The legacy baseline represents our previous approach using global clustering and legacy features. The proposed full system integrates the bi-directional anchor tracker, the new feature stack (AdaFace, TransReID, 4D-Humans), and the post-processing refinement module.
<table><tr><td>Method</td><td>Tracker</td><td>Features</td><td>Post-proc.</td><td>HOTA↑(%)</td><td>MOTA↑(%)</td><td>IDF1↑(%)</td><td>ID Sw ↓</td></tr><tr><td>Legacy baseline</td><td>Global clustering</td><td>Legacy</td><td>No</td><td>84.36</td><td>94.55</td><td>97.00</td><td>30</td></tr><tr><td>Feature upgrade</td><td>Global clustering</td><td>Enriched</td><td>No</td><td>84.53</td><td>94.86</td><td>97.22</td><td>9</td></tr><tr><td>Tracking upgrade</td><td>Bi-directional anchoring</td><td>Legacy</td><td>No</td><td>84.46</td><td>92.90</td><td>96.33</td><td>2</td></tr><tr><td>Proposed combination</td><td>Bi-directional anchoring</td><td>Enriched</td><td>No</td><td>84.75</td><td>95.06</td><td>97.51</td><td>0</td></tr><tr><td>Proposed full system</td><td>Bi-directional anchoring</td><td>Enriched</td><td>Yes</td><td>85.13</td><td>95.79</td><td>97.89</td><td>0</td></tr></table>

## 4.4. Feature ablation study

To isolate the individual contributions of each modality within the enriched feature set, we performed a component-wise ablation study using the final model configuration. To ensure that the evaluation reflects the intrinsic discriminative power of the online tracking features rather than offline refinement, all postprocessing is disabled. We also evaluated the impact on computational cost (frames per second, FPS) of each modality. We define three configurations for this analysis, whose results are presented in Table 6:

• Face only: Uses high-confidence facial anchors for association and discards faceless detections.

• Face + shape: Integrates 3D shape parameters to maintain tracking when faces are unavailable.

• Face + shape + appearance (full model): Adds appearance embeddings (TransReID) to resolve complex visual ambiguities.

Using only high-confidence facial anchors produces a highly precise baseline with zero identity switches. However, this configuration exhibits reduced recovery (MOTA 92.49%), since tracks terminate immediately when faces are occluded or subjects turn around, limiting overall trajectory coverage.

Integrating the 3D shape feature improved all accuracy metrics (HOTA +0.32, IDF1 +0.22, MOTA +0.39). This suggests that body shape provides critical discriminative signals in the absence of faces. Notably, this configuration produced a small increase in identity switches (2), reflecting improved recovery: whereas the face-only baseline discarded difficult trajectories, avoiding errors, Face + shape successfully closed ambiguous segments, occasionally at the cost of a switch, but keeping the trajectory alive for longer.

The full model achieves the highest overall performance, increasing MOTA to 95.06% and IDF1 to 97.51% while eliminating identity switches. Integrating appearance-based ReID features resolves the residual ambiguities left by shape matching alone, confirming that robust long-term identity preservation requires the joint exploitation of face, shape, and appearance.

Computational cost analysis: Finally, we evaluated hardware requirements and runtimes. All inference was performed on a workstation equipped with an NVIDIA A100 GPU with 40 GB of VRAM. The system showed high memory efficiency, with a peak of 4.9 GB in the heaviest configuration.

Table 6 shows a clear trade-off between empirical accuracy and processing speed. The Face-only baseline is computationally very efficient, reaching 25.04 FPS. However, once body shape is processed with 4D-Humans, speed drops to 4.90 FPS. For the full model we obtain 4.65 FPS. As this minimal difference shows, appearance extraction through TransReID turns out to be extremely lightweight. This suggests that 3D mesh estimation is the primary computational bottleneck. Therefore, while the full architecture guarantees maximum accuracy for offline video summarization or forensic analysis applications, strict real-time deployment will require further optimization.

Table 6: Feature set ablation study. We analyze the incremental contribution of each component and its computational impact.
<table><tr><td>Configuration</td><td>HOTA↑(%)</td><td>MOTA↑(%)</td><td>IDF1↑(%)</td><td>ID Sw ↓</td><td>Avg. FPS ↑</td></tr><tr><td>Face only</td><td>84.12</td><td>92.49</td><td>96.10</td><td>0</td><td>25.04</td></tr><tr><td>Face + shape</td><td>84.44</td><td>92.88</td><td>96.32</td><td>2</td><td>4.90</td></tr><tr><td>Face + shape + appearance</td><td>84.75</td><td>95.06</td><td>97.51</td><td>0</td><td>4.65</td></tr></table>

## 4.5. Quantitative analysis of the summary

Table 7 shows that the quantitative evaluation reveals distinct performance trade-offs across the three selection strategies. The Top-K baseline, which prioritizes frames with the highest facial scores, obtained the highest biometric confidence and resolution ratio. However, the proposed method proved competitive, outperforming the uniform baseline. In retrievability, the proposed method achieved the highest score, surpassing both uniform and Top-K. This demonstrates that the adaptive selection strategy captures the most representative visual features of each identity, ensuring that the summary serves as a reliable query source for downstream re-identification. The lower score for Top-K suggests that selecting only the best frames can omit critical appearance variations needed for comprehensive matching.

The largest performance differences appear in the diversity metrics. Uniform sampling naturally yields high temporal coverage but exhibits redundancy, as indicated by a low visual diversity score. Top-K exhibits severe redundancy, producing the lowest visual diversity and a temporal coverage that tends to concentrate frames around a single high-quality segment. In contrast, our method achieves the best balance, with the highest visual diversity and temporal coverage, distributing keyframes across the entire trajectory and minimizing redundant frames.

Together, these results show that, while simplistic strategies such as Top-K or uniform sampling maximize specific individual metrics, they fail to provide a holistic summary. The proposed method balances biometric quality and narrative coverage, creating a concise and diverse summary optimized for both human interpretation and automated analysis.

Table 7: Evaluation of VS on Custom-YT. We compare our proposed identity-centric approach against the Uniform and Top-K selection baselines across three evaluation categories.
<table><tr><td rowspan="2">Method</td><td colspan="2">Optimization</td><td>Information</td><td colspan="2">Diversity</td></tr><tr><td>Biometric confidence</td><td>Resolution ratio</td><td>Retrievability</td><td>Visual</td><td>Temporal</td></tr><tr><td>Uniform</td><td>0.7362</td><td>0.2381</td><td>0.9648</td><td>0.0832</td><td>0.8688</td></tr><tr><td>Top-K</td><td>0.8138</td><td>0.2524</td><td>0.9333</td><td>0.0370</td><td>0.4712</td></tr><tr><td>Proposed</td><td>0.7637</td><td>0.2438</td><td>0.9663</td><td>0.0909</td><td>0.8921</td></tr></table>

## 4.6. Qualitative analysis of summary selection

We performed a qualitative analysis using video sequences from the CHAD dataset (Danesh Pazho et al., 2023). Figure 3 presents the temporal distribution of keyframes for two distinct identities (ID 1 and ID 2).

The shaded area shows the continuous evolution of the importance score $S ( P , t )$ , while the markers denote the frames extracted by each selection strategy.

The Top-K strategy (green triangles) clusters its selections around the highest absolute peaks of the importance curve. While this guarantees high biometric quality, it produces severe temporal redundancy, capturing the subject within a very limited time span and completely missing the rest of their trajectory. In contrast, Uniform sampling (blue crosses) ensures temporal spread but is independent of the semantic content of the scene. As the plots show, uniform selections frequently fall into low-importance “valleys,” extracting suboptimal frames where the subject may be occluded, distant, or blurred, simultaneously missing critical high-scoring events.

In contrast, the proposed method (red dots), driven by Adaptive Non-Maximum Suppression, achieves an optimal balance. It successfully anchors its selections to the semantic peaks of the sequence, capturing highquality visual moments, while applying a temporal penalty to avoid redundancy. As the sequence illustrates, this ensures that the final summary captures a diverse and representative visual narrative of the subject's entire presence in the scene.

Summarization guided by semantic queries: In addition, our framework is not limited to a single, rigid summary output. As illustrated by the radar chart in Figure 3 (right), the proposed scoring function enables dynamic, query-guided summarization through the use of Semantic Profiles. By adjusting the weight distribution $( w _ { 1 } , w _ { 2 } , w _ { 3 } , w _ { 4 } )$ , users can tailor the summary to specific observation intents. For example, an operator performing a “Forensic Search” can maximize the biometric and resolution weights to extract only the sharpest facial crops. Alternatively, selecting a “Social Dynamics” or “Action” profile shifts the algorithmic focus toward behavior, prioritizing frames where subjects interact with others or exhibit significant motion, even if their faces are partially obscured. This flexibility allows the system to adapt smoothly to diverse analytical requirements.

![](images/fce965ae4107ee568fde9ae69920070f285dc6588c969ea93dbdab7e91b69d96.jpg)  
Figure 3: Qualitative evaluation of summary selection strategies for two distinct identities. (A, C) Temporal evolution of the importance score S(P,t) for ID 1 and ID 2, respectively. Our proposed method (red circles) effectively captures semantically rich frames while maintaining temporal diversity, avoiding the severe redundancy of the Top-K baseline (green triangles) and the suboptimal semantic performance of Uniform sampling (blue crosses). (B, D) The resulting visual summaries generated by the proposed method, demonstrating a complete and diverse narrative sequence for each identity. (E) Semantic profiles illustrating the system's ability to generate query-guided summaries, enabling the dynamic creation of personalized summaries based on specific analytical intents.

## 4.7. Failure analysis and limitations

Although the proposal demonstrates notable robustness to occlusion and appearance variation through 3D structural and fine-grained visual features, extreme stress scenarios reveal specific vulnerabilities. We detail below the most frequent failure modes, as well as the main limitation of the architecture. These factors are tied to the physical limits of the descriptors, the constraints of our temporal association model, and the processing load.

Prolonged occlusion with non-linear motion: The system struggles with prolonged occlusions if the subject changes trajectory. Because spatial tracking relies on simple kinematics, drastic changes in motion behind an obstacle cause the location prediction to fail when the subject reappears, fragmenting the trajectory.

Spatial overlap and feature entanglement: In dense crowds with overlapping bounding boxes, visual features become corrupted. This reduces the discriminative power of the network, introduces bias into the dynamic cost calculations, and can force a tracking failure.

Image degradation: Extreme lighting degradation or severe motion blur directly impacts feature quality. In addition, 3D shape estimation is vulnerable to scale ambiguity. For very distant subjects, small bounding boxes lack detail, causing the 4D-Humans network to generate a noisy or generic mesh, temporarily neutralizing the advantage of our structural cues.

Error propagation to the summary: If an identity switch is unrecoverable, the error contaminates the final summary, introducing several different people into the narrative of a single subject.

Real-time processing: Despite the pre-filtering steps, the main limitation of the proposal is its computational cost, due to the expensive extraction of descriptors.

## 5. Conclusions

This work presents an extension of a previous identity-centric video summarization proposal. We incorporate a hierarchical association strategy based on biometric anchors, together with a feature set that integrates adaptive facial biometrics, Transformer-based appearance, and 3D body shape. This combination maintains identity consistency in scenarios with occlusion, pose changes, and visual ambiguity, producing summaries that are more coherent at the narrative level.

The results show that the hierarchical fusion of complementary signals reduces trajectory fragmentation and eliminates identity switches on the evaluated dataset, improving the overall stability of tracking. The proposed summarization module balances biometric quality, social context, and temporal diversity, generating compact representations that preserve the relevant information of each individual.

Future work: The results and observations of this study open several priority lines of research:

Dataset expansion and large-scale validation: While the presented evaluation acts as a rigorous stress test, it represents a bounded scenario. As a first line of work, we will expand the dataset to increase representativeness with respect to real-world diversity. We will also scale the evaluation to public, highdensity benchmarks.

Semantic activity classification (HAR): To enrich the summary scoring function, we propose integrating a human action recognition module. This will make it possible to distinguish between routine and anomalous events, an improvement that is especially relevant in video surveillance scenarios, where it will allow sequences of higher contextual value to be prioritized.

• Optimization for real-time processing: To enable online deployment without sacrificing accuracy, we will investigate architectural optimization and quantization techniques that accelerate inference.

Subjective perceptual evaluation: Finally, since the narrative quality of a summary has an inherently subjective component that conventional objective metrics do not fully capture, we will design a perceptual evaluation framework with independent human users. This cross-study protocol will allow the summaries generated automatically by our architecture to be compared against manual selections created by participants, to formally evaluate the cognitive and semantic quality of the results using standardized qualitative assessment instruments and user perception scales.

## Acknowledgements

This work has been funded by the Recovery, Transformation, and Resilience Plan, financed by the European Union (Next Generation), through the LUCIA project (Fight against Cybercrime through the application of Artificial Intelligence), granted by INCIBE to the Universidad de León.

## References

Alaa, T., Mongy, A., Bakr, A., Diab, M., Gomaa, W., 2024. Video Summarization Techniques: A Comprehensive Review. https://doi.org/10.48550/ARXIV.2410.04449

Alomar, K., Aysel, H.I., Cai, X., 2025. CNNs, RNNs and Transformers in human action recognition: a survey and a hybrid model. Artif Intell Rev 58, 387. https://doi.org/10.1007/s10462-025-11388-3

Apostolidis, E., Adamantidou, E., Metsai, A.I., Mezaris, V., Patras, I., 2021a. AC-SUM-GAN: Connecting Actor-Critic and Generative Adversarial Networks for Unsupervised Video Summarization. IEEE Trans. Circuits Syst. Video Technol. 31, 3278–3292. https://doi.org/10.1109/TCSVT.2020.3037883

Apostolidis, E., Adamantidou, E., Metsai, A.I., Mezaris, V., Patras, I., 2021b. Video Summarization Using Deep Neural Networks: A Survey. Proc. IEEE 109, 1838–1863. https://doi.org/10.1109/JPROC.2021.3117472

Apostolidis, E., Balaouras, G., Mezaris, V., Patras, I., 2021c. Combining Global and Local Attention with Positional Encoding for Video Summarization, in: 2021 IEEE International Symposium on Multimedia (ISM). Presented at the 2021 IEEE International Symposium on Multimedia (ISM), IEEE, Naple, Italy, pp. 226–234. https://doi.org/10.1109/ISM52913.2021.00045

Argaw, D.M., Yoon, S., Heilbron, F.C., Deilamsalehy, H., Bui, T., Wang, Z., Dernoncourt, F., Chung, J.S., 2024. Scaling Up Video Summarization Pretraining with Large Language Models, in: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Presented at the 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Seattle, WA, USA, pp. 8332–8341. https://doi.org/10.1109/CVPR52733.2024.00796

Baskurt, K.B., Samet, R., 2019. Video synopsis: A survey. Computer Vision and Image Understanding 181, 26–38. https://doi.org/10.1016/j.cviu.2019.02.004

Bernardin, K., Stiefelhagen, R., 2008. Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics. EURASIP Journal on Image and Video Processing 2008, 1–10. https://doi.org/10.1155/2008/246309

Bewley, A., Ge, Z., Ott, L., Ramos, F., Upcroft, B., 2016. Simple online and realtime tracking, in: 2016 IEEE International Conference on Image Processing (ICIP). Presented at the 2016 IEEE International Conference on Image Processing (ICIP), IEEE, Phoenix, AZ, USA, pp. 3464–3468. https://doi.org/10.1109/ICIP.2016.7533003

Bhute, M.M., Tare, S.S., S, S.R., 2025. Query-Driven Video Summarization for Long Video Footage Analysis Using Faster-RCNN and Determinantal Point Processes. Procedia Computer Science 258, 3989–3999. https://doi.org/10.1016/j.procs.2025.04.650

Biswas, R., Chaves, D., Fernández-Robles, L., Fidalgo, E., Alegre, E., 2021. A Video Summarization Approach to Speed-up the Analysis of Child Sexual Exploitation Material, in: XLII JORNADAS DE AUTOMÁTICA : LIBRO DE ACTAS. Servizo de Publicacións da UDC, pp. 648–654. https://doi.org/10.17979/spudc.9788497498043.648

Danesh Pazho, A., Alinezhad Noghre, G., Rahimi Ardabili, B., Neff, C., Tabkhi, H., 2023. CHAD: Charlotte Anomaly Dataset, in: Gade, R., Felsberg, M., Kämäräinen, J.-K. (Eds.), Image Analysis, Lecture Notes in Computer Science. Springer Nature Switzerland, Cham, pp. 50–66. https://doi.org/10.1007/978-3-031-31435-3\_4

Dendorfer, P., Os̆ep, A., Milan, A., Schindler, K., Cremers, D., Reid, I., Roth, S., Leal-Taixé, L., 2021. MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking. Int J Comput Vis 129, 845–881. https://doi.org/10.1007/s11263-020-01393-0

Deng, J., Guo, J., Xue, N., Zafeiriou, S., 2019. ArcFace: Additive Angular Margin Loss for Deep Face Recognition, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Presented at the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Long Beach, CA, USA, pp. 4685–4694. https://doi.org/10.1109/CVPR.2019.00482

Ester, M., Kriegel, H.-P., Sander, J., Xu, X., 1996. A density-based algorithm for discovering clusters in large spatial databases with noise, in: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, KDD’96. AAAI Press, Portland, Oregon, pp. 226–231.

Goel, S., Pavlakos, G., Rajasegaran, J., Kanazawa, A., Malik, J., 2023. Humans in 4D: Reconstructing and Tracking Humans with Transformers, in: 2023 IEEE/CVF International Conference on Computer Vision (ICCV). Presented at the 2023 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE, Paris, France, pp. 14737–14748. https://doi.org/10.1109/ICCV51070.2023.01358

Guan, Z., Wang, Z., Zhang, G., Li, L., Zhang, M., Shi, Z., Jiang, N., 2025. Multi-object tracking review: retrospective and emerging trend. Artif Intell Rev 58, 235. https://doi.org/10.1007/s10462-025-11212-y

Gujar, P., 2024. How Marketers Can Improve Short-Form Video Ads In The Age Of GenAI. URL https://forbes.com/councils/forbestechcouncil/2024/09/12/how-marketers-can-improve-short-form-video-ads-in-the-ageof-genai/ (accessed 5.23.25).

Guo, J., Deng, J., Lattas, A., Zafeiriou, S., 2021. Sample and Computation Redistribution for Efficient Face Detection. https://doi.org/10.48550/ARXIV.2105.04714

Gygli, M., Grabner, H., Riemenschneider, H., Van Gool, L., 2014. Creating Summaries from User Videos, in: Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (Eds.), Computer Vision – ECCV 2014, Lecture Notes in Computer Science. Springer International Publishing, Cham, pp. 505–520. https://doi.org/10.1007/978-3-319-10584-0\_33

Hassan, S., Mujtaba, G., Rajput, A., Fatima, N., 2023. Multi-object tracking: a systematic literature review. Multimed Tools Appl 83, 43439–43492. https://doi.org/10.1007/s11042-023-17297-3

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep Residual Learning for Image Recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

He, S., Luo, H., Wang, P., Wang, F., Li, H., Jiang, W., 2021. TransReID: Transformer-based Object Re-Identification, in: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). Presented at the 2021 IEEE/CVF International Conference on Computer Vision (ICCV), IEEE, Montreal, QC, Canada, pp. 14993–15002. https://doi.org/10.1109/ICCV48922.2021.01474

Hsu, T.-C., Liao, Y.-S., Huang, C.-R., 2023. Video Summarization With Spatiotemporal Vision Transformer. IEEE Trans. on Image Process. 32, 3013–3026. https://doi.org/10.1109/TIP.2023.3275069

Jiang, L., Lan, L., 2025. MVS-SupCon: multimodal video summarization with supervised contrastive learning, in: Déniz, L.G., Meng, H., Carli, R. (Eds.), Fifth International Conference on Image Processing and Intelligent Control (IPIC 2025). Presented at the Fifth International Conference on Image Processing and Intelligent Control, SPIE, Qingdao, China, p. 75. https://doi.org/10.1117/12.3074773

Kim, M., Jain, A.K., Liu, X., 2022. AdaFace: Quality Adaptive Margin for Face Recognition, in: 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Presented at the 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, New Orleans, LA, USA, pp. 18729–18738. https://doi.org/10.1109/CVPR52688.2022.01819

Lee, M.J., Gong, D., Cho, M., 2025. Video Summarization with Large Language Models, in: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18981–18991. https://doi.org/10.1109/CVPR52734.2025.01768

Li, H., Klabjan, D., Utke, J., 2024. Unsupervised Video Summarization via Iterative Training and Simplified GAN, in: Proceedings of the Asian Conference on Computer Vision (ACCV). pp. 1585–1601.

Li, H., Zhu, Y., Shang, Z., Wang, Z., Wu, X., 2025. A Comprehensive Survey on Video Summarization: Challenges and Advances. IEEE Trans. Circuits Syst. Video Technol. 1–1. https://doi.org/10.1109/TCSVT.2025.3596006

Li, Q., Chen, J., Xie, Q., Han, X., 2023. Video summarization for event-centric videos. Neural Networks 161, 359–370. https://doi.org/10.1016/j.neunet.2023.01.047

Li, S., Ren, H., Xie, X., Cao, Y., 2025. A Review of Multi‐Object Tracking in Recent Times. IET Computer Vision 19, e70010. https://doi.org/10.1049/cvi2.70010

Li, X., Wang, Z., Lu, X., 2018. Video Synopsis in Complex Situations. IEEE Trans Image Process 27, 3798–3812. https://doi.org/10.1109/TIP.2018.2823420

Li, Z., Yang, L., 2021. Weakly Supervised Deep Reinforcement Learning for Video Summarization With Semantically Meaningful Reward, in: 2021 IEEE Winter Conference on Applications of Computer Vision (WACV). Presented at the

2021 IEEE Winter Conference on Applications of Computer Vision (WACV), IEEE, Waikoloa, HI, USA, pp. 3238– 3246. https://doi.org/10.1109/WACV48630.2021.00328

Liu, T., Meng, Q., Huang, J.-J., Vlontzos, A., Rueckert, D., Kainz, B., 2022. Video Summarization Through Reinforcement Learning With a 3D Spatio-Temporal U-Net. IEEE Trans. on Image Process. 31, 1573–1586. https://doi.org/10.1109/TIP.2022.3143699

Luiten, J., Os̆ep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., Leibe, B., 2021. HOTA: A Higher Order Metric for Evaluating Multi-object Tracking. Int J Comput Vis 129, 548–578. https://doi.org/10.1007/s11263-020-01375-2

Maggiolino, G., Ahmad, A., Cao, J., Kitani, K., 2023. Deep OC-Sort: Multi-Pedestrian Tracking by Adaptive Re-Identification, in: 2023 IEEE International Conference on Image Processing (ICIP). Presented at the 2023 IEEE International Conference on Image Processing (ICIP), IEEE, Kuala Lumpur, Malaysia, pp. 3025–3029. https://doi.org/10.1109/ICIP49359.2023.10222576

Meena, P., Kumar, H., Kumar Yadav, S., 2023. A review on video summarization techniques. Engineering Applications of Artificial Intelligence 118, 105667. https://doi.org/10.1016/j.engappai.2022.105667

Mirjalili, M., Alegre Gutiérrez, E., Fidalgo Fernández, E., González Castro, V., Tanveer, W., 2025. Human-Centric Video Summarization via Identity-Aware Tracking. JA-CEA. https://doi.org/10.17979/ja-cea.2025.46.12249

Narasimhan, M., Rohrbach, A., Darrell, T., 2021. Clip-it! language-guided video summarization. Advances in neural information processing systems 34, 13988–14000.

Narwal, P., Duhan, N., Bhatia, K.K., 2025. Dynamic and personalized video summarization towards sports entertainment. Entertainment Computing 55, 100999. https://doi.org/10.1016/j.entcom.2025.100999

Narwal, P., Duhan, N., Kumar Bhatia, K., 2022. A comprehensive survey and mathematical insights towards video summarization. Journal of Visual Communication and Image Representation 89, 103670. https://doi.org/10.1016/j.jvcir.2022.103670

Otani, M., Nakashima, Y., Rahtu, E., Heikkilä, J., 2019. Rethinking the Evaluation of Video Summaries, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). Presented at the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Long Beach, CA, USA, pp. 7588–7596. https://doi.org/10.1109/CVPR.2019.00778

Qaroush, A.M., Jubran, M., Olayyan, Q., 2025. A novel supervised framework for multi-video summarization: Addressing dataset biases and enhancing feature representation. Neurocomputing 653, 131180. https://doi.org/10.1016/j.neucom.2025.131180

Ramos, W., Silva, M., Araujo, E., Moura, V., Oliveira, K., Marcolino, L.S., Nascimento, E.R., 2023. Text-Driven Video Acceleration: A Weakly-Supervised Reinforcement Learning Method. IEEE Trans. Pattern Anal. Mach. Intell. 45, 2492– 2504. https://doi.org/10.1109/TPAMI.2022.3157198

Ristani, E., Solera, F., Zou, R., Cucchiara, R., Tomasi, C., 2016. Performance Measures and a Data Set for Multi-target, Multi-camera Tracking, in: Hua, G., Jégou, H. (Eds.), Computer Vision – ECCV 2016 Workshops, Lecture Notes in Computer Science. Springer International Publishing, Cham, pp. 17–35. https://doi.org/10.1007/978-3-319-48881-3\_2

Saini, P., Kumar, K., Kashid, S., Saini, A., Negi, A., 2023. Video summarization using deep learning techniques: a detailed analysis and investigation. Artif Intell Rev 56, 12347–12385. https://doi.org/10.1007/s10462-023-10444-0

Sharghi, A., Laurel, J.S., Gong, B., 2017. Query-Focused Video Summarization: Dataset, Evaluation, and a Memory Network Based Approach, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Shoitan, R., Moussa, M.M., Gharghory, S.M., Elnemr, H.A., Cho, Y.-I., Abdallah, M.S., 2023. User Preference-Based Video Synopsis Using Person Appearance and Motion Descriptions. Sensors 23, 1521. https://doi.org/10.3390/s23031521

Sultani, W., Chen, C., Shah, M., 2018. Real-World Anomaly Detection in Surveillance Videos, in: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 6479–6488. https://doi.org/10.1109/CVPR.2018.00678

Tang, Y., Bi, J., Xu, S., Song, L., Liang, S., Wang, T., Zhang, D., An, J., Lin, J., Zhu, R., Vosoughi, A., Huang, C., Zhang, Z., Liu, P., Feng, M., Zheng, F., Zhang, J., Luo, P., Luo, J., Xu, C., 2025. Video Understanding with Large Language Models: A Survey. IEEE Trans. Circuits Syst. Video Technol. 1–1. https://doi.org/10.1109/TCSVT.2025.3566695

Tank, C., 2023. The Data Bonanza: Exploring the Untapped Riches of Video Surveillance. Daten & Wissen. URL https://datenwissen.com/blog/the-data-bonanza-exploring-the-untapped-riches-of-video-surveillance/ (accessed 5.23.25).

Varghese, R., M., S., 2024. YOLOv8: A Novel Object Detection Algorithm with Enhanced Performance and Robustness, in: 2024 International Conference on Advances in Data Engineering and Intelligent Computing Systems (ADICS). Presented at the 2024 International Conference on Advances in Data Engineering and Intelligent Computing Systems (ADICS), IEEE, Chennai, India, pp. 1–6. https://doi.org/10.1109/ADICS58448.2024.10533619

Wistia, 2025. State of Video Report: Video Marketing Statistics for 2025. URL https://wistia.com/learn/marketing/videomarketing-statistics (accessed 10.15.25).

Wojke, N., Bewley, A., Paulus, D., 2017. Simple online and realtime tracking with a deep association metric, in: 2017 IEEE International Conference on Image Processing (ICIP). Presented at the 2017 IEEE International Conference on Image Processing (ICIP), IEEE, Beijing, pp. 3645–3649. https://doi.org/10.1109/ICIP.2017.8296962

Wu, G., Lin, J., Silva, C.T., 2022. Intentvizor: Towards generic query guided interactive video summarization, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10503–10512.

Xiao, S., Zhao, Z., Zhang, Z., Yan, X., Yang, M., 2020. Convolutional Hierarchical Attention Network for Query-Focused Video Summarization. AAAI 34, 12426–12433. https://doi.org/10.1609/aaai.v34i07.6929

Yale Song, Vallmitjana, J., Stent, A., Jaimes, A., 2015. TVSum: Summarizing web videos using titles, in: 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Presented at the 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Boston, MA, USA, pp. 5179–5187. https://doi.org/10.1109/CVPR.2015.7299154

Yuan, Y., Zhang, J., 2023. Unsupervised Video Summarization via Deep Reinforcement Learning With Shot-Level Semantics. IEEE Trans. Circuits Syst. Video Technol. 33, 445–456. https://doi.org/10.1109/TCSVT.2022.3197819

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., Wang, X., 2022. ByteTrack: Multi-object Tracking by Associating Every Detection Box, in: Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., Hassner, T. (Eds.), Computer Vision – ECCV 2022, Lecture Notes in Computer Science. Springer Nature Switzerland, Cham, pp. 1– 21. https://doi.org/10.1007/978-3-031-20047-2\_1

Zhang, Y., Zhu, P., Zheng, T., Yu, P., Wang, J., 2024. Surveillance video synopsis framework base on tube set. Journal of Visual Communication and Image Representation 98, 104057. https://doi.org/10.1016/j.jvcir.2024.104057

Zhao, B., Gong, M., Li, X., 2023. AudioVisual Video Summarization. IEEE Trans. Neural Netw. Learning Syst. 34, 5181– 5188. https://doi.org/10.1109/TNNLS.2021.3119969