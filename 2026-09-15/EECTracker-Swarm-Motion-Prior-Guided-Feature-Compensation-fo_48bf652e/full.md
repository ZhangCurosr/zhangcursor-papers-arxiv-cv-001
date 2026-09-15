# EECTracker: Swarm Motion Prior-Guided Feature Compensation for Airborne Optical UAV Swarm Tracking

Zhaochen Chu, Tao Song, Ren Jin\*, Mingdong Jia, Defu Lin

Abstract—Airborne optical tracking of uncrewed aerial vehicle (UAV) swarms is challenging due to extremely small target scales, rapid viewpoint changes, and cluttered backgrounds, which can weaken target feature responses and lead to intermittent or temporarily missing detector responses. Existing multiobject tracking methods generally depend on reliable targetspecific detector responses to maintain target states and identities across frames. When such responses become unreliable, target states cannot be reliably updated and cross-frame association cues become ambiguous, resulting in fragmented trajectories and identity switches. To address this problem, we propose EECTracker, a swarm-motion-prior-guided joint detection-andtracking framework for airborne optical UAV swarm tracking. EECTracker constructs a probabilistic swarm motion prior from reliable historical tracklets to capture the shared short-term image-plane motion tendency of the swarm and its uncertainty, providing spatial guidance for cross-frame feature compensation. Building on this prior, we introduce Energy–Entropy Consistency Activation (EEC Activation) to evaluate motion-prior-conditioned feature consistency using feature residual energy and local residual entropy. The resulting Local EEC score guides pixellevel feature compensation by enhancing motion-prior-consistent feature responses in potential target regions while suppressing inconsistent background responses. Experiments on AIRMOT and UAVSwarm show that EECTracker achieves superior overall tracking performance compared with state-of-the-art methods. Compared with the strongest competing method SCT-MOT, EECTracker improves MOTA/IDF1 by 3.89/1.79 percentage points on AIRMOT and by 2.81/1.74 percentage points on UAVSwarm, while maintaining online inference speed.

Index Terms—UAV swarm tracking, multiple-object tracking, motion prior, feature compensation.

## I. INTRODUCTION

IRBORNE optical tracking of uncrewed aerial vehicle (UAV) swarms is an important perception task for persistent airspace monitoring, cooperative UAV operations, and low-altitude situational awareness [1], [2]. The task requires continuously localizing multiple UAV targets and maintaining their identities over time. Compared with ground-based optical sensing, airborne optical platforms provide more flexible viewpoints and broader observation coverage [3], but are also challenged by extremely small target scales, rapid viewpoint changes, and complex background interference [4]. These factors can weaken target feature responses, making

target observations intermittent or temporarily missing across consecutive frames and thereby increasing the difficulty of target localization and identity association over time.

Existing multi-object tracking (MOT) methods can be broadly categorized into tracking-by-detection (TBD) and joint detection and tracking (JDT) frameworks [5], [6]. TBD methods first localize candidate targets and then associate them across frames using appearance and motion cues, whereas JDT methods share feature representations for detection and identity association within a unified framework. More recently, transformer-based trackers [7], [8] have further unified detection and tracking in end-to-end architectures, where persistent track queries interact with current-frame features through cross-frame attention to maintain target states over time. In addition, closed-loop detection-tracking methods [9], [10] have strengthened the interaction between detection and tracking by feeding historical detections, motion predictions, or propagated features back to the current frame, thereby guiding current-frame detection and cross-frame feature propagation through detection-derived spatial references. Nevertheless, despite their architectural differences, the effectiveness of existing MOT methods generally depends on sufficiently reliable observations and historical states maintained for individual targets to propagate target states across frames, provide spatial cues for temporal feature interaction, and maintain identity associations.

Such dependence on sufficiently reliable target observations and historical states becomes particularly limiting in airborne optical UAV swarm tracking, where extremely small target scales, rapid viewpoint changes, and complex background interference can weaken target feature responses and make target observations intermittent or temporarily missing across adjacent frames [11], [12]. When target observations or historical states become unreliable, temporal feature propagation and cross-frame association are weakened, leading to trajectory fragmentation and identity switches. However, UAVs within a swarm often exhibit shared short-term motion tendencies in the image plane rather than moving independently. Even when some UAVs are weakly observed, the reliable tracklets that remain available can still provide shared motion information, which can be aggregated into a swarm-level motion prior to indicate potential UAV regions consistent with the inferred short-term motion tendency. Historical feature information can then be guided toward these regions to supplement weakened current-frame target responses. We refer to this swarm-motionguided supplementation of weakened target feature responses as feature compensation.

To exploit this swarm-level motion information, we develop a swarm-motion-prior-guided feature compensation strategy for airborne optical UAV swarm tracking. Rather than deriving target-specific spatial guidance for feature compensation from reliable observations and historical states of every UAV, the strategy models individual image-plane motions using available reliable historical tracklets and aggregates them into a probabilistic swarm motion prior. The resulting prior captures the shared short-term motion tendency and uncertainty of the swarm, thereby characterizing statistically reachable imageplane regions of swarm targets in the current frame. It further provides dense pixel-level guidance across adjacent-frame feature maps to direct historical feature transfer and compensate the feature responses in potential target regions.

With the probabilistic swarm motion prior, the key challenge becomes how to identify reliable target-associated historical feature responses for current-frame feature compensation. Existing temporal feature alignment and enhancement methods commonly align or aggregate historical features using detection-derived locations or motion-predicted positions; therefore, their effectiveness degrades when these spatial references become unavailable under intermittent or temporarily missing target observations [13]. To address this limitation, we design Energy–Entropy Consistency Activation (EEC Activation), a pixel-level feature compensation mechanism built on motion-prior-conditioned feature consistency. Under the swarm motion prior, displacement-conditioned historical feature responses are compared with current-frame responses to form feature residuals. Feature residual energy measures the expected magnitude of these residuals, while local residual entropy characterizes their spatial dispersion. These two complementary statistics are then combined to construct the Local EEC score, which represents the local feature consistency between prior-guided historical responses and current-frame responses. This score is further converted into a channelwise soft consistency mask to selectively weight featureconsistent historical responses during feature fusion. In this way, EEC Activation enhances feature-consistent responses in potential UAV target regions while suppressing featureinconsistent background responses.

Building upon the proposed feature compensation strategy and EEC Activation, we further develop EECTracker, a swarm-motion-prior-guided joint detection and tracking framework for airborne optical UAV swarm tracking. At each frame, reliable historical tracklets are used to construct a probabilistic swarm motion prior, which guides feature compensation across adjacent-frame feature maps, while EEC Activation selectively controls historical feature fusion. The compensated feature representation is then used for target detection and association, and the updated tracklets are fed back to construct the motion prior for subsequent frames. Within this closed loop, historical tracklets provide swarm-level motion guidance for feature compensation, whereas the compensated features in turn support more reliable detection and association, thereby improving detection reliability and trajectory continuity under weakened target feature responses and intermittent target observations.

The main contributions of this work are summarized as follows:

1. We develop a swarm-motion-prior-guided feature compensation strategy that constructs a probabilistic swarm motion prior from available reliable historical tracklets. By capturing the shared short-term motion tendency and uncertainty of the swarm, the prior provides dense pixel-level guidance for feature compensation without requiring reliable target-specific observations for every UAV.

2. We propose Energy–Entropy Consistency Activation (EEC Activation), a pixel-level feature compensation mechanism that adaptively weights historical responses over potential target regions according to motion-priorconditioned feature consistency. It combines feature residual energy and local residual entropy into a Local EEC score and converts the score into a channel-wise soft consistency mask for selective feature fusion.

3. Building upon the above two components, we further develop EECTracker, a swarm-motion-prior-guided joint detection and tracking framework. Extensive experiments on UAV swarm tracking benchmarks demonstrate improved overall tracking performance and detection robustness compared with state-of-the-art methods.

![](images/badbe71143be2da5601fe5e247f5c6f2d7b02a072ece407c61b9114a0fee4526.jpg)  
Fig. 1. Conceptual comparison of tracking frameworks for airborne optical UAV swarm tracking. Detection-driven frameworks rely on detections for cross-frame association, while detection-conditioned closed-loop frameworks further exploit detection-derived spatial references for temporal feature propagation; both become vulnerable when target observations are intermittent or temporarily missing. In contrast, EECTracker introduces swarm-motion-priorguided feature compensation, in which a probabilistic swarm motion prior provides spatial guidance and EEC Activation selectively compensates featureconsistent potential target responses for subsequent detection and association.

## II. RELATED WORK

## A. Airborne Optical UAV Swarm Tracking

Recent aerial remote-sensing studies have devoted increasing attention to visual UAV tracking under challenging imaging conditions [14]–[16]. By contrast, airborne optical tracking of multiple UAVs in swarm scenarios remains relatively underexplored, with limited dedicated benchmarks and tracking algorithms. MOT-FLY [17] provides an early airborne multi-UAV tracking benchmark, but it mainly contains independently moving UAV targets. UAVSwarm [18] and AIRMOT [19] more directly characterize UAV swarm tracking with small long-range targets, cluttered backgrounds, homogeneous appearances, and coordinated motion patterns. Based on these benchmarks, UAVS-MOT [20] improves weak UAV target representation through coordinate attention, while BELGTracker [21] and HOMATracker [19] exploit geometric, motion, or association cues to improve instance discrimination. More recently, GL-DT [22] combines spatio-temporal feature fusion and trajectory-aware tracking to improve trajectory continuity. SCT-MOT [10] uses target-wise motion predictions as explicit spatial references to guide historical feature alignment and fusion. Despite these advances, intermittent or temporarily missing target observations can still interrupt trajectory continuity and identity maintenance, which remains insufficiently addressed.

## B. Motion-Prior-Guided Visual MOT

Image-plane motion modeling has been widely exploited in online visual MOT to maintain temporal target information and improve cross-frame tracking. Classical recursive estimators, such as Kalman and particle filters, predict target states from historical observations [1], while learning-based trajectory prediction methods further capture interaction-aware or cooperative motion patterns among multiple agents [23].

In visual MOT, motion cues and priors are mainly incorporated into tracking pipelines in three forms. First, displacement-based methods estimate inter-frame offsets for target localization or temporal feature propagation, as in CenterTrack [24], TraDeS [25] and IMANet [9]. Second, historical-state and query-based methods maintain temporal target information through explicit positions, track queries, or trajectory hypotheses. PPTracker [26] exploits historical target positions as spatial priors, while TransTrack [27] introduces track queries for cross-frame target matching. MOTR [7] and MOTRv3 [8] maintain persistent track queries across frames to couple detection and association, whereas TCT [28] further introduces trajectory-level hypotheses for temporal tracking. Third, prior-fusion methods explicitly inject motion information into feature matching or attention. For example, P2FTrack [29] integrates detection and tracking priors with feature posteriors through prior–posterior attention. However, most existing motion priors are constructed or updated from the observation history and state estimates of individual targets. When target observations become intermittent or temporarily missing, the available target-specific evidence is reduced, weakening the observational constraints on the corresponding motion estimates and limiting the reliability of motion-prior-guided current-frame localization and temporal feature propagation in MOT.

## C. Motion-Aware Feature Modeling and Fusion

Motion-aware feature modeling and fusion have been widely studied in video object detection and multi-object tracking. Existing approaches mainly exploit motion information in two forms. One line of work uses target-specific motion or tracking references to guide feature enhancement or detection. MP2Net [30] combines mask propagation with motion prediction for satellite video object detection, while OmniTracker [13] uses tracking priors to guide current-frame detection. Another line establishes cross-frame motion correspondence to propagate and aggregate historical features. FGFA [31] and STSN [32] perform temporal feature aggregation through motion-guided propagation or learned spatio-temporal sampling. Recent aerial and smalltarget methods, including DFA-MOT [33], StreamFlow [34], and LMAFormer [35], further exploit optical flow, deformable alignment, or motion-aware attention for temporal feature fusion. In satellite-video MOT, FMA-Net [36] employs pixellevel flow-driven motion modeling to guide inter-frame feature fusion. Feature discrimination and reliability modeling have also been explored to identify informative responses and quantify motion uncertainty. OIFS [37] employs entropy energy to select informative appearance features for local target– background classification. ProbFlow [38] estimates opticalflow uncertainty from an energy-based posterior, where entropy characterizes the motion uncertainty. However, these criteria focus on appearance-feature selection or motion-field uncertainty estimation, rather than on discriminating target responses from background responses under a given motion prior.

Overall, motion-aware feature modeling exploits motion information either to provide target-specific spatial guidance for feature enhancement or to establish cross-frame correspondence for historical feature propagation and fusion. The former depends on reliable target-specific references, whereas the latter may propagate background or spatially mismatched responses together with useful target information, with the reliability of these propagated responses remaining insufficiently characterized before fusion.

## III. METHOD

## A. Problem Formulation and Framework Overview

Given an online airborne optical video stream $\left\{ { { I } _ { t } } \right\}$ , UAV swarm tracking aims to localize multiple UAV targets in each incoming frame and maintain their identities over time. At frame t, the detector produces a detection set $\mathcal { D } _ { t } = \{ d _ { j , t } \} _ { j = 1 } ^ { M _ { t } }$ where $d _ { j , t } ~ = ~ ( \mathbf { b } _ { j , t } , s _ { j , t } )$ , and $\mathbf { b } _ { j , t } ~ = ~ [ x _ { j , t } , y _ { j , t } , w _ { j , t } , h _ { j , t } ] ^ { \top }$ denotes the image-plane bounding box, $s _ { j , t }$ is the detection confidence. After data association, these observations are used to update the track set $\mathcal { T } _ { t } = \{ \tau _ { i , t } \} _ { i = 1 } ^ { N _ { t } }$ , where $\tau _ { i , t } = ( \mathrm { i d } _ { i } , \mathbf { b } _ { i , t } )$ and ${ \mathrm { i d } } _ { i }$ denotes the object identity.

In airborne optical UAV swarm tracking, small target scales, dynamic viewpoints, and background interference can weaken current-frame target feature responses, resulting in intermittent or temporarily missing target observations and an increased risk of trajectory fragmentation and identity switches. To address this challenge, we propose EECTracker, a swarmmotion-prior-guided joint detection and tracking framework that introduces a probabilistic swarm motion prior for currentframe feature compensation. Rather than relying solely on target-specific motion estimates, the prior characterizes the shared short-term motion tendency of the swarm together with its uncertainty, thereby defining statistically reachable imageplane regions for historical feature projection. The prior is estimated online from available reliable historical tracklets and used to guide historical feature information toward these regions. EEC Activation then selectively incorporates reliable projected historical responses into the current representation before downstream detection and association.

![](images/818b76f232d7d204abecec20287724b363a89f02939a1c8357a503b514bc212a.jpg)  
Fig. 2. Overall architecture of EECTracker. The probabilistic swarm motion prior construction, pixel-level feature compensation via EEC Activation, and downstream detection and association are detailed in Sec. III-B, Sec. III-C, and Sec. III-D, respectively.

As illustrated in Fig. 2, EECTracker integrates probabilistic swarm motion prior construction, feature extraction, motionprior-conditioned pixel-level feature compensation via EEC Activation, and downstream detection and association into a closed-loop online tracking framework.

The framework consists of four components: (1) probabilistic swarm motion prior construction, which takes the historical image-plane motion state $\mathbf { S } _ { t - 1 }$ derived from available reliable tracklets as input and estimates a swarm-level motion prior $\mathcal { M } _ { t }$ characterizing the statistically reachable image-plane regions of swarm targets and their associated uncertainty in the current frame; (2) feature extraction, which encodes adjacent frames $I _ { t - 1 }$ and $I _ { t }$ with a shared backbone to obtain multi-scale feature maps $\mathbf { F } _ { t - 1 }$ and $\mathbf { F } _ { t } ; \mathbf { \Xi } ( 3 )$ pixel-level feature compensation via EEC Activation, which evaluates motionprior-conditioned consistency between historical and current feature responses, projects historical features to the current frame according to $\mathcal { M } _ { t } ,$ , and uses the resulting consistency to selectively weight the projected features for compensating weakened current-frame responses, yielding the compensated feature maps $\hat { \mathbf { F } } _ { t } ;$ and (4) detection and tracking, which uses $\hat { \mathbf { F } } _ { t }$ for target localization, confidence prediction, appearance embedding extraction, and cross-frame association, producing the current detection set $\mathcal { D } _ { t }$ and updated track set $\mathcal { T } _ { t } .$ . The resulting tracking states are subsequently used to update the historical motion state from $\mathbf { S } _ { t - 1 }$ to $\mathbf { S } _ { t } ,$ , which is used for swarm motion prior construction at the next frame.

For adjacent frames $I _ { t - 1 }$ and $I _ { t } ,$ , the shared backbone extracts multi-scale feature maps $\mathbf { F } _ { t - 1 } ~ = ~ \{ \mathbf { F } _ { t - 1 } ^ { l } \} _ { l = 1 } ^ { m }$ and $\mathbf { F } _ { t } = \{ \mathbf { F } _ { t } ^ { l } \} _ { l = 1 } ^ { m }$ . For the l-th feature level, $\mathbf { F } _ { t } ^ { l } \in \bar { \mathbf { R } } ^ { \bar { C } _ { l } \times \bar { H _ { l } } \times W _ { l } }$ where $C _ { l }$ denotes the number of channels and $( H _ { l } , W _ { l } )$ denotes the spatial resolution. For an input image of resolution $H _ { 0 } \times W _ { 0 }$ , each level has a down-sampling stride $r _ { l } ,$ , such that $H _ { l } = H _ { 0 } / r _ { l }$ and $W _ { l } = W _ { 0 } / r _ { l }$ . In our implementation, $r _ { l } ~ \in ~ \{ 8 , 1 6 , 3 2 \}$ . Since UAV targets are typically small in airborne optical images, feature compensation is performed on the highest-resolution feature level $\mathbf { F } _ { t } ^ { 1 }$ to preserve finegrained target responses. Guided by $\mathcal { M } _ { t }$ , the historical feature $\mathbf { F } _ { t - 1 } ^ { 1 }$ is projected to the current frame and selectively incorporated through EEC Activation, yielding the compensated feature map $\breve { \hat { { \mathbf { F } } } } _ { t } ^ { 1 } \in { \mathbf { { R } } ^ { C _ { 1 } \times H _ { 1 } \times W _ { 1 } } }$ <sup>1</sup>. The compensated feature map replaces F<sup>1</sup><sub>t</sub> in the multi-scale feature set to form $\hat { \mathbf { F } } _ { t } =$ $\{ \hat { \mathbf { F } } _ { t } ^ { \bar { 1 } } , \mathbf { F } _ { t } ^ { \bar { 2 } } , \ldots , \mathbf { F } _ { t } ^ { \bar { m } } \}$ , which is subsequently used for detection and association. The updated track set $\mathcal { T } _ { t }$ is then fed back to update the historical motion state for swarm motion prior construction at the next frame, thereby closing the online loop from motion-prior construction to feature compensation, detection and association, and track-state update.

## B. Probabilistic Swarm Motion Prior Construction

UAVs participating in coordinated swarm maneuvers often exhibit a shared short-term image-plane motion tendency. This collective motion property provides a transferable cue for estimating the current-frame motion of the swarm when direct observations of some targets become weak or temporarily unavailable. Based on this observation, we construct a probabilistic swarm motion prior from reliable historical tracklets to characterize the current-frame swarm motion and its uncertainty. The construction consists of five stages. First, reliable historical tracklets are selected from the online tracking results. Second, their image-plane center trajectories and velocity sequences within a temporal window are used to represent the historical motion states. Third, the historical motion states are encoded by projecting the position sequences and decomposing the velocity sequences into global swarm and residual components, yielding embedded position and motion representations for subsequent trajectory prediction. Fourth, an EqMotion-based image-plane trajectory predictor estimates the future locations of reliable tracklets and their target-wise spatial distributions. Fifth, the predicted locations are used to estimate the global swarm motion, while the target-wise spatial distributions are aggregated through motion-consistencyweighted moment matching into a compact swarm-level distribution; together, they jointly form the probabilistic swarm motion prior $\mathcal { M } _ { t }$

Before processing frame t, let $\mathcal { R } _ { t }$ denote the subset of reliable historical tracklets used for current swarm motion modeling, and let $n _ { T } = | \mathcal { R } _ { t } | . \mathrm { ~ A ~ }$ tracklet is regarded as reliable if it has valid observations in every frame within the historical temporal window $[ t - T , t - 1 ]$ and maintains the same identity throughout the entire window. Here, $T$ denotes the temporal window length.

For each reliable tracklet $i ,$ its image-plane center is denoted by $\mathbf { c } _ { i , t _ { j } } = [ x _ { i , t _ { j } } , y _ { i , t _ { j } } ] ^ { \top }$ , and its inter-frame velocity is $\mathbf { v } _ { i , t _ { j } } =$ ${ \bf c } _ { i , t _ { j } } - { \bf c } _ { i , t _ { j - 1 } } ,$ for $t _ { j } \in \{ t - T , \dots , t - 1 \}$ }. The first velocity is initialized as $\mathbf { v } _ { i , t - T } = \mathbf { v } _ { i , t - T + 1 } .$ Accordingly, the historical position and velocity sequences of tracklet i are represented as $\mathbf { c } _ { i } ~ = ~ \{ \mathbf { c } _ { i , t _ { j } } \} _ { t _ { i } = t - T } ^ { t - 1 } ~ \in ~ \mathbb { R } ^ { T \times 2 }$ and $\mathbf { v } _ { i } = \{ \mathbf { v } _ { i , t _ { j } } \} _ { t _ { i } = t - T } ^ { t - 1 } \in$ $\mathbb { R } ^ { T \times 2 }$ , respectively. The historical image-plane motion state of the reliable tracklets within the swarm is therefore $\mathbf { S } _ { t - 1 } =$ $\{ ( \mathbf { c } _ { i } , \mathbf { v } _ { i } ) \} _ { i = 1 } ^ { n _ { T } }$

To explicitly encode the shared swarm-motion information, we first estimate the average swarm velocity at each time step:

$$
\mathbf { v } _ { s w , t _ { j } } = \frac { 1 } { n _ { T } } \sum _ { i = 1 } ^ { n _ { T } } \mathbf { v } _ { i , t _ { j } } , \qquad t _ { j } \in \{ t - T , \dots , t - 1 \} .\tag{1}
$$

The corresponding historical swarm-velocity sequence is denoted as $\begin{array} { r } { \bar { \bf v } _ { s w } = \bar { \bf \{ v } } _ { s w , t _ { j } } \} _ { t _ { i } = t - T } ^ { t - 1 } \in { \bf R } ^ { T \times 2 }  \end{array}$

The position and velocity sequences are then encoded according to their respective roles in image-plane motion modeling. The position sequence describes the historical spatial evolution of each reliable tracklet, whereas the velocity sequence characterizes its short-term motion. Specifically, the position state is directly projected into the embedding space, while the velocity state is encoded by combining the global swarm velocity with the residual velocity relative to the swarm:

$$
\begin{array} { r } { \begin{array} { c } { \tilde { \mathbf { c } } _ { i } = \eta _ { c } \mathbf { c } _ { i } \in \mathbf { R } ^ { T \times f } , } \\ { \tilde { \mathbf { v } } _ { i } = \eta _ { v } ( \mathbf { v } _ { i } - \mathbf { v } _ { s w } ) + \eta _ { v } ^ { s w } \mathbf { v } _ { s w } \in \mathbf { R } ^ { T \times f } } \end{array} } \end{array}\tag{2}
$$

where $\eta _ { c } , \ \eta _ { v } , \ \eta _ { v } ^ { s w }$ denote learnable linear projection layers, and $f$ denotes the embedding dimension.

Let $\tilde { \textbf { C } } = \ \{ \tilde { \mathbf { c } } _ { i } \} _ { i = 1 } ^ { n _ { T } } \ \in \ \mathbf { R } ^ { n _ { T } \times T \times f }$ and $\tilde { \textbf { V } } = \{ \tilde { \mathbf { v } } _ { i } \} _ { i = 1 } ^ { n _ { T } } \in$ $\mathbf { R } ^ { n _ { T } \times T \times f }$ denote the embedded position and velocity states of reliable tracklets. An EqMotion-based trajectory predictor [23] is then employed to estimate their future image-plane motion. EqMotion jointly captures the temporal motion evolution of each reliable tracklet and its interactions with other targets, enabling interaction-aware prediction for coordinated swarm motion. Since the original EqMotion produces deterministic trajectory predictions, we retain its original trajectoryprediction architecture and append a lightweight uncertainty head to provide a probabilistic spatial characterization relative to the deterministic location predicted for frame $t { : }$

$$
\left\{ \hat { \mathbf { c } } _ { i } , \left( \hat { \mu } _ { i } , \hat { \Sigma } _ { i } \right) \right\} _ { i = 1 } ^ { n _ { T } } = \phi _ { \mathrm { t r a j } } \left( \tilde { \mathbf { C } } , \tilde { \mathbf { V } } \right)\tag{3}
$$

Here, $\phi _ { \mathrm { t r a j } }$ denotes the EqMotion-based predictor with the appended uncertainty head. For each reliable tracklet $i ,$ the trajectory head predicts its future image-plane locations $\hat { \mathbf { c } } _ { i } \in$ $\bar { \mathbf { R } ^ { P \times 2 } }$ , while the uncertainty head, implemented using a twolayer multilayer perceptron (MLP), predicts a target-wise probabilistic spatial distribution $\mathcal { N } _ { i } ( \hat { \mu } _ { i } , \hat { \Sigma } _ { i } )$ at frame $t .$ Here, $\hat { \mu } _ { i }$ denotes the mean offset relative to the predicted position $\hat { \mathbf { c } } _ { i , t }$ and $\hat { \Sigma } _ { i }$ denotes the corresponding spatial covariance.

Based on the prediction, we further derive the predicted velocity of each reliable swarm tracklet at frame t:

$$
\hat { \mathbf { v } } _ { t } = \{ \hat { \mathbf { v } } _ { 1 , t } , \cdot \cdot \cdot , \hat { \mathbf { v } } _ { n _ { T } , t } \} , \hat { \mathbf { v } } _ { i , t } = \hat { \mathbf { c } } _ { i , t } - \mathbf { c } _ { i , t - 1 }\tag{4}
$$

The predicted global swarm velocity is then estimated as

$$
\hat { \mathbf { v } } _ { s w , t } = \frac { 1 } { n _ { T } } \sum _ { i = 1 } ^ { n _ { T } } \hat { \mathbf { v } } _ { i , t }\tag{5}
$$

Here, $\hat { \mathbf { v } } _ { s w , t }$ summarizes the shared deterministic motion of the swarm at frame t.

To aggregate the target-wise probabilistic predictions according to their agreement with the shared swarm motion, we compute a directional consistency score and its normalized weight:

$$
s _ { i } = \frac { \langle \hat { \mathbf { v } } _ { i , t } , \hat { \mathbf { v } } _ { s w , t } \rangle } { \| \hat { \mathbf { v } } _ { i , t } \| _ { 2 } \| \hat { \mathbf { v } } _ { s w , t } \| _ { 2 } + \epsilon } , \alpha _ { i } = \frac { \exp ( \beta s _ { i } ) } { \sum _ { j = 1 } ^ { n _ { T } } \exp ( \beta s _ { j } ) }\tag{6}
$$

where ϵ is a small constant for numerical stability and $\beta$ controls the concentration of the normalized weights. The resulting $\alpha _ { i }$ determines the contribution of each reliable tracklet to the swarm-level probabilistic representation.

Using these weights, we construct a compact swarm-level motion distribution by moment-matching the target-wise spatial distributions:

$$
\hat { \mu } _ { s w } = \sum _ { i = 1 } ^ { n _ { T } } \alpha _ { i } \hat { \mu } _ { i } , \hat { \Sigma } _ { s w } = \sum _ { i = 1 } ^ { n _ { T } } \alpha _ { i } \left[ \hat { \Sigma } _ { i } + ( \hat { \mu } _ { i } - \hat { \mu } _ { s w } ) ( \hat { \mu } _ { i } - \hat { \mu } _ { s w } ) ^ { \top } \right]\tag{7}
$$

Finally, the probabilistic swarm motion prior at time t is defined as:

$$
\mathcal { M } _ { t } = \{ \hat { \mathbf { v } } _ { s w , t } , \mathcal { N } ( \hat { \mu } _ { s w } , \hat { \Sigma } _ { s w } ) \}\tag{8}
$$

where $\hat { \mathbf { v } } _ { s w , t }$ provides the predicted global swarm motion, and $\hat { \mathcal { N } } ( \hat { \mu } _ { s w } , \hat { \Sigma } _ { s w } )$ denotes a compact swarm-level probabilistic characterization of its spatial tendency and uncertainty. The $\mathcal { M } _ { t }$ guides the subsequent EEC-based feature compensation.

C. Motion-Prior-Conditioned Feature Compensation via EEC Activation

Cooperative swarm UAVs often exhibit shared short-term motion patterns, leading to coherent cross-frame displacement of target regions in airborne optical imagery. In contrast, background regions and noise are generally not constrained by the same swarm-specific motion pattern, and therefore tend to exhibit weaker or less stable correspondence. This difference provides a useful cue for distinguishing potential UAV target regions and suppressing motion-inconsistent background interference.

To characterize this difference in feature space, we evaluate the cross-frame feature residuals under swarm-motion-guided correspondence. If a historical target response remains consistent with its current-frame counterpart after accounting for the shared swarm motion, the resulting cross-frame residuals are expected to exhibit both smaller magnitudes and a more spatially concentrated local distribution; in contrast, motioninconsistent background and noise responses tend to produce larger residuals with more dispersed local distributions. We therefore use residual energy to characterize the magnitude of the cross-frame feature mismatch and residual entropy to measure the local spatial dispersion of the residual responses. Lower residual energy and entropy generally indicate stronger cross-frame motion consistency. Together, these complementary statistics provide an interpretable feature-space measure for selectively exploiting motion-consistent historical information.

Motivated by the above measurements, we propose Energy-Entropy Consistency Activation (EEC Activation), a pixellevel feature compensation mechanism guided by motionprior-conditioned cross-frame feature consistency. As shown in Fig. 3, EEC Activation consists of three stages. First, historical features are projected toward the current frame according to the probabilistic swarm motion prior. Second, cross-frame residuals between the current response and displacementconditioned historical responses are evaluated, from which the expected residual energy and local residual entropy are computed and combined into a Local EEC score, which is further converted into a channel-wise soft consistency mask. Third, the resulting mask is used to selectively weight the projected historical features before fusion with the current feature map, thereby compensating feature responses in potential UAV target regions and suppressing motion-inconsistent background responses.

To propagate historical features under the probabilistic swarm motion prior, we first define a cross-frame spatial mapping function U. This function models pixel transport between adjacent frames. Specifically, given a pixel $\mathbf { p } _ { t - 1 }$ in frame $t - 1$ , its current-frame correspondence is represented by a possible pixel position $\mathbf { p } _ { t }$ and modeled as:

$$
\begin{array} { r } { \mathbf { p } _ { t } = \mathcal { U } ( \mathbf { p } _ { t - 1 } | t - 1 \to t ) = \mathbf { p } _ { t - 1 } + \Delta , } \\ { \Delta = \hat { \mathbf { v } } _ { s w , t } + \delta ~ } \end{array}\tag{9}
$$

where $\hat { \mathbf { v } } _ { s w , t }$ represents the predicted global swarm displacement, and δ is the probabilistic spatial offset sampled from the Gaussian component $\mathcal { N } ( \hat { \mu } _ { s w } , \bar { \hat { \Sigma } } _ { s w } )$ of the swarm motion prior. The former provides a deterministic reference for the dominant cross-frame swarm displacement, whereas the latter characterizes the spatial uncertainty around this reference. $\Delta$ denotes the overall probabilistic displacement. In this way, the probabilistic swarm motion prior characterizes a set of statistically reachable image-plane locations for historical feature projection in the current frame, with the corresponding probability density reflecting their relative likelihood under the predicted swarm motion.

Under intermittent or temporarily missing observations, local feature alignment based on detection-derived spatial references may become unavailable or inaccurate. Instead of aligning features around detected target regions, we apply motion-prior-guided projection over the entire feature map. Let $\mathbf { F } _ { t } = \{ \mathbf { F } _ { t } ^ { \bar { 1 } } , \cdots , \bar { \mathbf { F } _ { t } ^ { m } } \}$ denote the multi-scale feature maps at frame t, where $\mathbf { F } _ { t } ^ { 1 }$ is the highest-resolution level. Because UAV targets are usually extremely small in airborne optical images, feature compensation is performed only on $\mathbf { F } _ { t } ^ { 1 }$ with stride $r _ { 1 } = 8$ to preserve fine spatial details of potential target regions while balancing effectiveness and computational efficiency. The image-plane motion parameters are transformed into the coordinate system of this feature level:

$$
\hat { \bf v } _ { s w , t } ^ { 1 } = \frac { \hat { \bf v } _ { s w , t } } { r _ { 1 } } , \hat { \mu } _ { s w , t } ^ { 1 } = \frac { \hat { \mu } _ { s w , t } } { r _ { 1 } } , \hat { \Sigma } _ { s w , t } ^ { 1 } = \frac { \hat { \Sigma } _ { s w , t } } { r _ { 1 } ^ { 2 } } .\tag{10}
$$

Based on the spatial mapping function $u ,$ we project the historical feature map $\mathbf { F } _ { t - 1 } ^ { 1 }$ into the current feature coordinate system and obtain the motion-prior-projected feature map $\tilde { \mathbf { F } } _ { t - 1  t } ^ { 1 }$ . This projected feature map statistically characterizes the expected current-frame response of historical features under the swarm motion prior.

$$
\begin{array} { r l } & { \tilde { \mathbf { F } } _ { t - 1 \to t } ^ { 1 } ( \mathbf { p } _ { t } ) = \mathbb { E } _ { \Delta \sim \mathcal { N } ( \hat { \mathbf { v } } _ { s w , t } ^ { 1 } + \hat { \mu } _ { s w } ^ { 1 } , \hat { \Sigma } _ { s w } ^ { 1 } ) } [ \mathbf { F } _ { t - 1 } ^ { 1 } ( \mathbf { p } _ { t } - \Delta ) ] = } \\ & { \qquad \displaystyle \int _ { \delta } \mathbf { F } _ { t - 1 } ^ { 1 } ( \mathbf { p } _ { t } - \delta - \hat { \mathbf { v } } _ { s w , t } ^ { 1 } ) \mathcal { N } ( \delta ; \hat { \mu } _ { s w } ^ { 1 } , \hat { \Sigma } _ { s w } ^ { 1 } ) d \delta } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad . } \end{array}\tag{11}
$$

For a target-associated location whose image-plane motion follows the swarm motion prior, the historical feature response sampled at its inverse-mapped position is expected to be close to the corresponding current-frame response. This approximate motion-prior-conditioned feature consistency can be expressed as:

$$
\mathbf { F } _ { t - 1 } ^ { 1 } ( \mathcal { U } _ { \Delta } ^ { - 1 } ( \mathbf { p } _ { t } ) ) \approx \mathbf { F } _ { t } ^ { 1 } ( \mathbf { p } _ { t } )\tag{12}
$$

Accordingly, we further define a channel-wise expected feature residual over the displacement distribution. For pixel $\mathbf { p } _ { t }$ and channel $c ,$ the channel residual under a displacement $\Delta$ is defined as:

$$
\begin{array} { r l } & { \mathbf { r } _ { c } ( \mathbf { p } _ { t } ; \Delta ) = \mathbf { F } _ { t - 1 , c } ^ { 1 } ( \mathcal { U } _ { \Delta } ^ { - 1 } ( \mathbf { p } _ { t } ) ) - \mathbf { F } _ { t , c } ^ { 1 } ( \mathbf { p } _ { t } ) = } \\ & { \qquad \mathbf { F } _ { t - 1 , c } ^ { 1 } ( \mathbf { p } _ { t } - \Delta ) - \mathbf { F } _ { t , c } ^ { 1 } ( \mathbf { p } _ { t } ) } \end{array}\tag{13}
$$

![](images/7d3cf4ecd6123e807082ab6b3c01463f45c2bad300af2bffad33c53b56fe00a7.jpg)  
Fig. 3. Pixel-level feature compensation via EEC Activation. The process consists of three stages: (1) probabilistic swarm motion prior-guided projection of historical feature responses toward the current frame; (2) Local EEC estimation from residual energy and local residual entropy, followed by channel-wise soft consistency mask construction; and (3) fusion of the EEC-weighted projected historical feature with the current feature map to obtain the compensated feature representation.

where $\mathbf { F } _ { t , c } ^ { 1 }$ and $\mathbf { F } _ { t - 1 , c } ^ { 1 }$ denote the feature responses of the current and historical feature maps at channel c, respectively. The residual $\mathbf { r } _ { c } ( \mathbf { p } _ { t } ; \Delta )$ measures the cross-frame feature mismatch between the current response and the historical response sampled at the corresponding inverse-mapped position. Based on the channel residual, we then define the channel-wise residual energy as the expected squared feature residual over the displacement distribution induced by the swarm motion prior:

$$
E _ { c } ( \mathbf { p } _ { t } ) = \mathbb { E } _ { \Delta \sim \mathcal { N } ( \hat { \mathbf { v } } _ { s w , t } ^ { 1 } + \hat { \mu } _ { s w } ^ { 1 } , \hat { \Sigma } _ { s w } ^ { 1 } ) } [ \mathbf { r } _ { c } ( \mathbf { p } _ { t } ; \Delta ) ^ { 2 } ]\tag{14}
$$

This residual energy measures the expected magnitude of cross-frame feature mismatch under plausible swarm motion displacements. Target regions consistent with the swarm motion typically exhibit lower residual energy, whereas motioninconsistent background regions generally produce larger residual energy.

On the discrete pixel grid, we approximate the above channel-wise residual energy by discretizing the displacement distribution within a local neighborhood $\Omega ( \mathbf { p } _ { t } )$ . Since feature maps are defined on discrete pixel locations, the continuous expectation in Eq. 14 is implemented by a normalized weighted summation over the discrete sampling locations. Specifically, for a pixel $\mathbf { p } _ { t }$ , we consider candidate sampling locations $\textbf { u } \in \Omega ( { \mathbf { p } } _ { t } )$ and use the corresponding feature responses to construct a discrete approximation of the displacementinduced feature distribution.

$$
w ( \mathbf { u } | \mathbf { p } _ { t } ) = \frac { \mathcal { N } \big ( \mathbf { p } _ { t } - \mathbf { u } ; \hat { \mu } _ { s w } ^ { 1 } , \hat { \Sigma } _ { s w } ^ { 1 } \big ) } { \sum _ { \mathbf { u ^ { \prime } } \in \Omega ( \mathbf { p } _ { t } ) } \mathcal { N } \big ( \mathbf { p } _ { t } - \mathbf { u ^ { \prime } } ; \hat { \mu } _ { s w } ^ { 1 } , \hat { \Sigma } _ { s w } ^ { 1 } \big ) }\tag{15}
$$

Accordingly, the discrete approximation of the motionprior-projected historical feature at $\mathbf { p } _ { t }$ is given by:

$$
\tilde { \mathbf { F } } _ { t - 1  t } ^ { 1 } ( \mathbf { p } _ { t } ) = \sum _ { \mathbf { u } \in \Omega ( \mathbf { p } _ { t } ) } w ( \mathbf { u } | \mathbf { p } _ { t } ) \mathbf { F } _ { t - 1 } ^ { 1 } ( \mathbf { u } - \hat { \mathbf { v } } _ { s w , t } ^ { 1 } )\tag{16}
$$

The corresponding channel-wise expected residual energy is approximated as:

$$
E _ { c } ( \mathbf { p } _ { t } ) = \sum _ { \mathbf { u } \in \Omega ( \mathbf { p } _ { t } ) } w ( \mathbf { u } | \mathbf { p _ { t } } ) ( \mathbf { F } _ { t - 1 , c } ^ { 1 } ( \mathbf { u } - \hat { \mathbf { v } } _ { s w , t } ^ { 1 } ) - \mathbf { F } _ { t , c } ^ { 1 } ( \mathbf { p } _ { t } ) ) ^ { 2 }\tag{17}
$$

Residual energy characterizes the magnitude of the crossframe feature mismatch but does not describe the spatial concentration of local consistency responses. We further normalize the channel-wise residual energy within the local neighborhood to obtain a local residual-consistency distribution:

$$
\pi _ { c } ( \mathbf { q } _ { t } | \mathbf { p } _ { t } ) = \frac { \exp ( - E _ { c } ( \mathbf { q } _ { t } ) ) } { \sum _ { \mathbf { k } _ { t } \in \Omega ( \mathbf { p } _ { t } ) } \exp ( - E _ { c } ( \mathbf { k } _ { t } ) ) } , \mathbf { q } _ { t } \in \Omega ( \mathbf { p } _ { t } )\tag{18}
$$

The channel-wise residual entropy is then defined as:

$$
S _ { c } ( \mathbf { p } _ { t } ) = - \sum _ { \mathbf { q } _ { t } \in \Omega ( \mathbf { p _ { t } } ) } \pi _ { c } ( \mathbf { q } _ { t } | \mathbf { p } _ { t } ) \log \pi _ { c } ( \mathbf { q } _ { t } | \mathbf { p } _ { t } )\tag{19}
$$

Here, residual entropy measures the local spatial dispersion of residual-consistency responses. For target regions consistent with the swarm motion prior, low-residual responses are usually more spatially concentrated, leading to lower entropy. In contrast, motion-inconsistent background regions tend to produce more dispersed residual responses and therefore higher entropy.

Based on the channel-wise residual energy and residual entropy, we construct a Local EEC score to jointly characterize the magnitude and local spatial dispersion of the cross-frame feature residuals. Specifically, for pixel $\mathbf { p } _ { t }$ and channel c, the Local EEC score is defined as:

$$
G _ { c } ( \mathbf { p } _ { t } ) = E _ { c } ( \mathbf { p } _ { t } ) + \lambda _ { s } S _ { c } ( \mathbf { p } _ { t } )\tag{20}
$$

where $\lambda _ { s }$ balances the contributions of the residual energy and entropy terms. A lower $G _ { c } ( \mathbf { p } _ { t } )$ indicates stronger motionprior-conditioned consistency between the historical and current feature responses in channel $c ,$ whereas a higher value indicates weaker consistency.

We then construct a learnable channel-wise soft consistency mask ${ \cal M } ( { \bf p } _ { t } ) \in \mathrm { { \bf ~ R } } _ { 1 } ^ { C }$ to adaptively activate potential target features based on $G _ { c }$ . For channel $c ,$ the mask is defined as:

$$
M _ { c } ( \mathbf { p } _ { t } ) = \sigma ( \alpha _ { c } ( \tau _ { c } - G _ { c } ( \mathbf { p } _ { t } ) ) )\tag{21}
$$

where $\tau _ { c } \in \mathbf { R }$ and $\alpha _ { c } > 0$ denote the learnable consistency threshold and scale factor for channel $c ,$ respectively. Both parameters can be optimized end-to-end during training. $\sigma ( \cdot )$ denotes the HardSigmoid activation function. With this design, each channel can adaptively modulate its activation according to its response to motion-prior-conditioned consistency, enabling finer-grained control of historical feature contributions.

We then apply the mask $M ( \mathbf { p } _ { t } )$ to selectively weight the motion-prior-projected historical feature and fuse it with the current-frame feature, producing the swarm-motion-guided compensated feature map $\hat { \mathbf { F } } _ { t } ^ { 1 }$

$$
\hat { \mathbf { F } } _ { t } ^ { 1 } = \phi _ { f u } \bigl ( [ M \odot \tilde { \mathbf { F } } _ { t - 1  t } ^ { 1 } ; \mathbf { F } _ { t } ^ { 1 } ] \bigr ) + \mathbf { F } _ { t } ^ { 1 }\tag{22}
$$

where $\phi _ { f u }$ denotes a $3 \times 3$ convolutional fusion layer, $[ \cdot , \cdot ]$ denotes channel-wise concatenation, and ⊙ denotes the Hadamard product. This operation retains historical responses with stronger motion-prior-conditioned consistency while attenuating inconsistent responses, thereby enabling pixel-level compensation of potential target features under intermittent or temporarily missing observations.

Finally, the compensated feature map $\hat { \mathbf { F } } _ { t } ^ { 1 }$ replaces $\mathbf { F } _ { t } ^ { 1 }$ in the multi-scale feature set and is fed into the subsequent detection and tracking branches for target detection and association.

## D. Detection and Tracking Integration

To integrate probabilistic swarm motion prior construction and feature compensation into an online tracking loop, we adopt an existing swarm tracking framework, HOMATracker [19] as the downstream detection and tracking framework. The compensated feature set $\hat { \mathbf { F } } _ { t }$ is fed into this framework for detection, appearance embedding extraction, and crossframe association, producing the current tracking result $\mathcal { T } _ { t }$ for subsequent motion state update.

Specifically, $\hat { \mathbf { F } } _ { t }$ is processed by two parallel branches for target detection and appearance embedding extraction. The detection branch predicts the detection set $\mathcal { D } _ { t }$ , including bounding boxes and confidence scores. In parallel, the appearance branch extracts target-level embeddings $\mathbf { E } _ { t }$ from $\hat { \mathbf { F } } _ { t }$ for identity discrimination and data association.

During association, we follow the multi-frame homogeneous association strategy of HOMATracker and construct the association cost using appearance similarity and motion consistency. The current detections $\mathcal { D } _ { t }$ are then associated with historical tracklets $\mathcal { T } _ { t - 1 }$ to obtain the updated track set $\mathcal { T } _ { t } .$ This process is performed within a sliding temporal window to improve association stability.

The resulting track set $\mathcal { T } _ { t }$ is then used to update the historical motion state $\mathbf { S } _ { t }$ . Before processing frame $t + 1$ , tracklets that remain continuously observed and maintain consistent identities throughout the temporal window are selected to form the reliable set $\mathcal { R } _ { t + 1 }$ , which is then used to construct the nextframe probabilistic swarm motion prior $\mathcal { M } _ { t + 1 }$ . This feedback process closes the online loop among motion-prior construction, feature compensation, detection, and tracking, enabling the tracker to continuously exploit reliable historical swarm motion information and thereby improve tracking robustness under intermittent or temporarily missing observations.

## IV. EXPERIMENTS

## A. Datasets

We evaluate EECTracker on two representative public benchmarks for airborne optical UAV swarm tracking: AIR-MOT and UAVSwarm. AIRMOT is a simulation-based benchmark, whereas UAVSwarm is collected from real flight scenarios.

AIRMOT is an open-source simulation benchmark for airborne optical swarm UAV tracking. It contains 8 highly dynamic RGB video sequences with a resolution of 1920×1080 and a total of 7,844 frames. A major challenge of AIRMOT is the extremely small apparent target scale: more than 97% of targets are smaller than $3 2 \times 3 2$ pixels, which leads to weak visual responses and intermittent detections. Each sequence typically contains 5–16 UAV instances undergoing coordinated maneuvers and formation changes.

UAVSwarm is a real-world UAV swarm tracking benchmark collected from flight scenarios. It contains 72 video sequences with a total of 12,598 annotated frames, covering 13 different scenes and more than 19 UAV types, and is officially split into 36 training sequences and 36 testing sequences. Compared with AIRMOT, UAVSwarm presents more complex background interference, including urban buildings, clouds, and ground clutter, together with multiple viewpoints such as top-down and horizontal views. It also includes representative swarm behaviors such as formation reconfiguration and cooperative maneuvers.

We follow the official train/test splits for all experiments, with models trained on the training set and evaluated on the test set. To train the probabilistic swarm motion prior construction module, we further construct auxiliary trajectoryprediction sequences from the original training annotations of both benchmarks. For each video sequence, we extract continuous trajectory segments of length $L = 2 0$ using a spatiotemporal sliding window. Following a standard trajectoryprediction setting, the first $L _ { o b s } = 8$ frames are used as historical observations and the remaining $L _ { p r e d } = 1 2$ frames are used for future-motion supervision. During sequence construction, the frame-level temporal alignment among UAV tracklets is preserved, so that the predictor can jointly learn individual image-plane motion evolution and short-term swarm coordi nation patterns. This procedure yields two auxiliary trajectoryprediction subsets, denoted as AIRMOT-STP (Swarm Trajectory Prediction) and UAVSwarm-STP, respectively. These subsets are used only for pretraining the swarm motion-prior construction module and do not alter the official tracking benchmark splits.

## B. Evaluation Metrics

We evaluate EECTracker from three aspects: detection accuracy, tracking performance, and runtime efficiency. For detection evaluation, we report Average Precision at an IoU threshold of 0.5 (AP50). For tracking evaluation, we report Multiple Object Tracking Accuracy (MOTA), Identification F1 Score (IDF1), Higher Order Tracking Accuracy (HOTA), and the number of identity switches (IDSW). Runtime efficiency is measured by the end-to-end inference speed in frames per second (FPS) [39].

AP50 measures the detection accuracy by integrating the precision–recall curve at an IoU threshold of 0.5:

$$
\mathrm { A P _ { 5 0 } } = \int _ { 0 } ^ { 1 } P ( R ) d R\tag{23}
$$

where P and R denote the precision and recall evaluated at an IoU threshold of 0.5, respectively.

MOTA measures the overall tracking accuracy by accounting for false positives, false negatives, and identity switches:

$$
\mathrm { M O T A } = 1 - \frac { \sum _ { t } \left( \mathrm { F N } _ { t } + \mathrm { F P } _ { t } + \mathrm { I D S W } _ { t } \right) } { \sum _ { t } \mathrm { G T } _ { t } }\tag{24}
$$

where $\mathrm { F P } _ { t }$ $\mathrm { F N } _ { t } .$ , IDSW , and GT denote the numbers of false positives, false negatives, identity switches, and groundtruth objects at frame t, respectively.

IDF1 evaluates identity preservation over the entire sequence and is defined as

$$
\mathrm { I D F 1 } = \frac { 2 \cdot \mathrm { I D T P } } { 2 \cdot \mathrm { I D T P } + \mathrm { I D F P } + \mathrm { I D F N } } ,\tag{25}
$$

where IDTP, IDFP, and IDFN denote the numbers of true-positive, false-positive, and false-negative identity assignments, respectively. IDSW directly counts the total number of identity switches:

$$
\mathrm { I D S W } = \sum _ { t } \mathrm { I D S W } _ { t } .\tag{26}
$$

HOTA jointly evaluates detection and association quality by averaging the corresponding scores over a set of localization thresholds, and is computed as

$$
\mathrm { { H O T A } } = \frac { 1 } { | \cal { A } | } \sum _ { \alpha \in \cal { A } } \sqrt { \mathrm { { D e t A } } _ { \alpha } \mathrm { { A s s A } } _ { \alpha } } ,\tag{27}
$$

where Det $\mathrm { A } _ { \alpha }$ and $\mathrm { A s s A } _ { \alpha }$ denote the detection and association accuracy at threshold $\alpha ,$ respectively, and A denotes the set of localization thresholds used for evaluation.

Finally, runtime efficiency is evaluated using the end-to-end inference speed in frames per second (FPS). Higher AP50, MOTA, IDF1, HOTA, and FPS indicate better performance, whereas lower IDSW is preferred.

## C. Implementation Details

We train and evaluate all models on a single NVIDIA RTX 3090 GPU.

We first pretrain the probabilistic swarm motion prior construction (PSMP) module. An EqMotion-based [23] trajectory predictor is employed to predict target-wise future imageplane locations, together with a lightweight uncertainty head for estimating the corresponding local spatial uncertainty. The predictor takes $L _ { \mathrm { o b s } } = 8$ historical frames as input and outputs two complementary motion representations: a deterministic future trajectory over the subsequent $P \ = \ L _ { \mathrm { p r e d } } \ = \ 1 2$ frames and a target-wise Gaussian spatial offset distribution parameterized by $( \hat { \mu } _ { i } , \hat { \Sigma } _ { i } )$ . The former provides the predicted image-plane motion reference, while the latter characterizes the corresponding local spatial uncertainty. We optimize this module using stochastic gradient descent (SGD) with an initial learning rate of $5 \times 1 0 ^ { - 4 } ,$ , momentum of 0.9, and weight decay of $1 \times 1 0 ^ { - 4 }$ for 60 epochs.

To supervise trajectory prediction, we use a mean squared position error loss:

$$
\mathcal { L } _ { p o s } = \frac { 1 } { n _ { T } P } \sum _ { t _ { j } = 1 } ^ { P } \sum _ { i = 1 } ^ { n _ { T } } \lvert \lvert \hat { \mathbf { c } } _ { i } [ t _ { j } ] - \mathbf { c } _ { i } ^ { * } [ t _ { j } ] \rvert \rvert _ { 2 } ^ { 2 }\tag{28}
$$

where $\mathbf { c } _ { i } ^ { * }$ denotes the ground truth future trajectory of target i over the next $P$ frames, and $\hat { \mathbf { c } } _ { i }$ denotes the corresponding prediction.

To supervise the probabilistic spatial prediction at frame t, we construct a local spatial reference by averaging the groundtruth positions within a temporal neighborhood:

$$
\bar { \mathbf { c } } _ { i , t } ^ { * } = \frac { 1 } { 2 P _ { u } + 1 } \sum _ { q = - P _ { u } } ^ { P _ { u } } \mathbf { c } _ { i , t + q } ^ { * } ,\tag{29}
$$

where $P _ { u } = 3$ determines the temporal support of the local spatial reference. The uncertainty head predicts the mean $\hat { \mu } _ { i }$ and covariance $\hat { \Sigma } _ { i }$ of the target-wise spatial offset and is trained using the negative log-likelihood (NLL) loss:

$$
\mathcal { L } _ { \mathrm { n l l } } = - \frac { 1 } { n _ { T } } \sum _ { i = 1 } ^ { n _ { T } } \log \mathcal { N } \left( \bar { \mathbf { c } } _ { i , t } ^ { * } - \hat { \mathbf { c } } _ { i , t } ; \hat { \mu } _ { i } , \hat { \Sigma } _ { i } \right)\tag{30}
$$

The overall training loss of the PSMP module is:

$$
\mathcal { L } _ { t r a j } = \lambda _ { 1 } \mathcal { L } _ { p o s } + \lambda _ { 2 } \mathcal { L } _ { n l l }\tag{31}
$$

where $\lambda _ { 1 } = 1$ and $\lambda _ { 2 } = 0 . 0 1$ are fixed for all experiments.

After pretraining the PSMP module, we train the full MOT framework. We adopt HOMATracker as the base framework and integrate both the PSMP module and EEC Activation into it. During training, all input images are resized while preserving the original aspect ratio and padded to an input size of $1 0 8 8 \times 1 0 8 8$ , and the network is optimized using SGD with an initial learning rate of 0.002, momentum of 0.9, and weight decay of $1 \times 1 0 ^ { - 4 }$

A sliding temporal window of $T = 8$ frames is used to maintain the historical motion states for probabilistic swarm motion prior construction. During full-framework training, the prior is constructed from historical tracklets and used to guide EEC-based historical feature compensation. The pretrained PSMP module remains frozen, while EEC Activation is optimized end-to-end together with the downstream tracking framework through the original detection and association losses. Specifically, we follow the multi-branch loss design of HOMATracker, including bounding-box regression, category prediction, objectness prediction, and appearance association losses, collectively denoted as $\mathcal { L } _ { \mathrm { t r a c k } } .$ . No additional supervision is introduced for EEC Activation.

For EEC Activation, the probabilistic displacement distribution is discretized over a fixed rectangular neighborhood around each query pixel. The local neighborhood size is set to $3 \times 3$ on AIRMOT and $7 \times 7$ on UAVSwarm based on the ablation results. The balance factor of the Local EEC score is fixed to $\lambda _ { s } ~ = ~ 0 . 1$ . The learnable channel-wise gating parameters $\alpha _ { c }$ and $\tau _ { c }$ are initialized as 1.0 and 0.8, respectively, and constrained to [0.001, 2.0] during training for numerical stability. These settings are fixed for all sequences within each dataset.

During inference, the framework processes the video stream online and predicts swarm motion prior using a sliding window of length $T \ = \ 8$ and stride 1. In the first T frames, the framework performs conventional detection and tracking to initialize target trajectories and historical motion-state buffer. From frame $T + 1$ onward, tracklets satisfying the reliabletracklet selection rule are used by the PSMP module to construct the probabilistic swarm motion prior $\mathcal { M } _ { t }$ . If no reliable tracklet is available at frame t, PSMP and EEC Activation are bypassed, and the original current-frame features $\mathbf { F } _ { t }$ are retained. Otherwise, EEC Activation projects historical features toward the current frame under $\mathcal { M } _ { t } ,$ computes the Local EEC score, and applies the resulting channel-wise soft consistency mask to weight the projected historical features before fusion with the current-frame feature map. The compensated feature set $\hat { \mathbf { F } } _ { t }$ is subsequently fed into the downstream detection and tracking branches.

All input images are resized with their original aspect ratios preserved and then padded to $1 0 8 8 \times 1 0 8 8$ . The confidence threshold for candidate detection is 0.01, and non-maximum suppression (NMS) is applied with a threshold of 0.7. The detection branch outputs the candidate target set $\mathcal { D } _ { t }$ for the current frame. For runtime evaluation, we report the end-toend inference speed of the complete online framework in FPS, including image preprocessing, feature extraction, probabilistic swarm motion prior construction, EEC-based feature compensation, downstream detection, association and post-processing.

For data association, we follow the multi-frame homogeneous association strategy of HOMATracker and maintain the appearance features and motion states of each tracklet within the sliding window. Detections in each frame are first divided into high-confidence and low-confidence sets according to their confidence scores. High-confidence detections are assigned to existing tracklets using the Hungarian algorithm with a cost matrix computed from appearance similarity and motion difference. Unmatched high-confidence detections are initialized as new tracklets. Low-confidence detections are associated only with unmatched tracklets from the previous frame based on spatial IoU to reduce false associations.

## D. Comparison with State-of-the-Art Methods

To comprehensively evaluate the effectiveness of EEC-Tracker for airborne optical UAV swarm tracking, we compare it with representative methods from three mainstream MOT paradigms: tracking-by-detection (TBD), joint detection and tracking (JDT), and transformer-based end-to-end tracking. The compared methods include DeepSORT [40], Byte-Track [6], OC-SORT [41], Hybrid-SORT [42], BELGTracker [21], HOMATracker [19], FairMOT [5], UAVS-MOT [20], MOTRv3 [8] and SCT-MOT [10].

To ensure a fair comparison, we control irrelevant factors other than the tracking framework as much as possible. For TBD methods, we employ YOLOX-X as the unified front-end detector, and all methods are trained and evaluated under the same train/test split and input resolution. For JDT and end-toend methods, we preserve their original framework designs and follow their standard training and inference pipelines, while using the same data splits, input resolution, and evaluation protocol whenever applicable.

The quantitative results on AIRMOT and UAVSwarm are shown in Table I. EECTracker achieves the best MOTA, IDF1, HOTA, and IDSW on both benchmarks, demonstrating consistent improvements in overall tracking accuracy and identity preservation. On AIRMOT, compared with SCT-MOT, EECTracker improves MOTA, IDF1, and HOTA by 3.89, 1.79, and 0.61 percentage points, respectively, while reducing IDSW by 76. On UAVSwarm, the corresponding improvements are 2.81, 1.74, and 0.60 percentage points, respectively, with IDSW further reduced from 56 to 54.

Overall, EECTracker yields stable improvements across both datasets, with more pronounced benefits on AIRMOT. This trend is consistent with the more severe small-target observation degradation in AIRMOT, where current-frame UAV responses are more frequently weakened or intermittently missed. In terms of efficiency, EECTracker maintains online inference speed and remains comparable to the most relevant joint tracking baselines such as HOMATracker and SCT-MOT, indicating that the framework provides a favorable balance between tracking performance and online computational efficiency.

To complement the quantitative evaluation, we present qualitative comparisons on the AIRMOT and UAVSwarm datasets in Figs. 4 and 5. For visual clarity, we compare EECTracker with the most relevant competitors, including SCT-MOT as the strongest prior method, HOMATracker as the direct base framework, and Hybrid-SORT as a representative strong TBD baseline when applicable. In the figures, different tracked targets are marked by bounding boxes with different colors; red arrows denote false positives, while yellow boxes and arrows indicate missed detections. The selected examples are taken from representative sequences involving weak target responses, cluttered backgrounds, and severe appearance degradation to highlight the robustness differences among the compared methods.

In the AIRMOT-02 sequence, targets are difficult to perceive between frames 250 and 352 due to small scale, weak appearance, and cloud-background interference. Under these conditions, the compared methods exhibit varying false alarms and missed detections. At frame 250, for example, SCT-MOT and HOMATracker produce one to two false positives. At frame 352, multiple targets are missed by SCT-MOT, HOMATracker, and Hybrid-SORT. In contrast, EEC-Tracker maintains visibly more stable detection and tracking results over the selected interval, with fewer false alarms, fewer missed detections, and more consistent identities. On UAVSwarm, we further analyze two representative clips. In UAVSwarm-20, the combination of small target size and complex ground clutter causes noticeable missed detections for competing methods; for example, HOMATracker misses one target at frame 48. In UAVSwarm-32, at frame 42, SCT-MOT and HOMATracker produce false alarms for nearly half of the targets because of interference from background border structures. Meanwhile, partial occlusion among adjacent UAVs causes missed detections for multiple competing methods. By comparison, EECTracker maintains more stable detection and tracking results for all targets under these challenging conditions. These qualitative observations are consistent with the quantitative results and further illustrate the role of the probabilistic swarm motion prior and EEC Activation. The motion prior provides statistically reachable image-plane regions for historical feature projection, while EEC Activation performs feature compensation according to motion-prior-conditioned feature consistency, strengthening useful historical responses in potential UAV target regions while attenuating motioninconsistent responses. Together, these mechanisms improve detection reliability and tracking continuity under weak and intermittent target observations in airborne optical imagery.

TABLE I  
QUANTITATIVE COMPARISON WITH STATE-OF-THE-ART METHODS ON AIRMOT AND UAVSWARM. ↑/↓: HIGHER/LOWER IS BETTER. BEST RESULTS ON EACH DATASET ARE SHOWN IN BOLD
<table><tr><td>Methods</td><td>Publication</td><td>MOTA(%)↑</td><td>IDF1(%)↑</td><td>HOTA(%)↑</td><td>IDSW↓</td><td>FPS↑</td></tr><tr><td colspan="7">AIRMOT</td></tr><tr><td>FairMOT [5]</td><td>IJCV 2021</td><td>18.19</td><td>17.41</td><td>17.56</td><td>476</td><td>35.2</td></tr><tr><td>DeepSORT [40]</td><td>ICIP 2017</td><td>19.60</td><td>20.45</td><td>19.42</td><td>2841</td><td>33.6</td></tr><tr><td>OC-SORT [41]</td><td>CVPR 2023</td><td>20.88</td><td>23.27</td><td>22.07</td><td>522</td><td>40.9</td></tr><tr><td>ByteTrack [6]</td><td>ECCV 2022</td><td>24.35</td><td>25.72</td><td>23.56</td><td>1707</td><td>39.2</td></tr><tr><td>MOTRv3 [8]</td><td>ArXiv 2023</td><td>23.52</td><td>24.89</td><td>22.59</td><td>1805</td><td>20.6</td></tr><tr><td>Hybrid-SORT [42]</td><td>AAAI 2024</td><td>29.01</td><td>24.08</td><td>21.70</td><td>652</td><td>28.9</td></tr><tr><td>BELGTracker [21]</td><td>Acta Aero. Sin. (CN) 2024</td><td>29.27</td><td>23.55</td><td>23.47</td><td>1144</td><td>34.6</td></tr><tr><td>HOMATracker [19]</td><td>CJA 2025</td><td>30.08</td><td>29.31</td><td>25.55</td><td>506</td><td>20.0</td></tr><tr><td>SCT-MOT [10] EECTracker</td><td>TAES 2026</td><td>32.30 36.19</td><td>30.15 31.94</td><td>25.54 26.15</td><td>476</td><td>21.8 22.4</td></tr><tr><td>UAVSwarm</td><td>ours</td><td></td><td></td><td></td><td>400</td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>FairMOT [5]</td><td>IJCV 2021</td><td>67.7</td><td>73.2</td><td>59.2</td><td>590</td><td>35.2</td></tr><tr><td>DeepSORT [40]</td><td>ICIP 2017</td><td>61.2</td><td>70.3</td><td>58.8</td><td>221</td><td>33.6</td></tr><tr><td>OC-SORT [41]</td><td>CVPR 2023</td><td>75.7</td><td>81.8</td><td>64.7</td><td>709</td><td>40.9</td></tr><tr><td>ByteTrack [6]</td><td>ECCV 2022</td><td>65.0</td><td>76.5</td><td>60.6</td><td>67</td><td>39.2</td></tr><tr><td>MOTRv3 [8]</td><td>ArXiv 2023</td><td>62.3</td><td>71.9</td><td>57.9</td><td>72</td><td>20.6</td></tr><tr><td>Hybrid-SORT [42]</td><td>AAAI 2024</td><td>77.4</td><td>80.1</td><td>62.8</td><td>459</td><td>28.9</td></tr><tr><td>UAVS-MOT [20]</td><td>Acta Aero. Sin. (CN) 2024</td><td>73.4</td><td>76.1</td><td>65.8</td><td>740</td><td>34.6</td></tr><tr><td>HOMATracker [19]</td><td>CJA 2025</td><td>79.2</td><td>87.1</td><td>67.0</td><td>58</td><td>20.3</td></tr><tr><td>SCT-MOT [10]</td><td>TAES 2026</td><td>81.90</td><td>88.45</td><td>68.56</td><td>56</td><td>22.3</td></tr><tr><td>EECTracker</td><td>ours</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td><td>22.9</td></tr></table>

## E. Effectiveness of Probabilistic Swarm Motion Prior-Guided Feature Compensation

To evaluate the effectiveness of the proposed swarm-motionprior-guided feature compensation strategy, we conduct an ablation study under a unified HOMATracker-based framework. All variants share the same backbone, detection branch, appearance branch, and association strategy. For variants requiring motion prediction, the same PSMP module is used, so that the comparison focuses on the feature compensation strategy. We compare four variants: (1) Baseline, which uses HOMATracker without feature compensation; (2) Temporal feature concatenation, which directly concatenates and fuses adjacent-frame features; (3) Explicit detector-location-guided compensation, which uses detector-provided target locations to guide feature compensation and is implemented using the representative IMA fusion mechanism from IMANet [9]; and (4) Swarm Motion-prior-guided feature compensation (EEC-Tracker). The results are reported in Table II.

As shown in Table II, EECTracker achieves the best AP50, MOTA, IDF1, and HOTA on both AIRMOT and UAVSwarm. On AIRMOT, compared with explicit detector-location-guided compensation, EECTracker improves AP50, MOTA, IDF1, and HOTA by 1.30, 4.35, 3.56, and 1.49 percentage points, respectively, and reduces IDSW by 89. On UAVSwarm, EECTracker improves AP50, MOTA, IDF1, and HOTA over explicit detector-location-guided compensation by 2.20, 4.66, 2.56, and 1.87 percentage points, while achieving the same lowest IDSW of 54.

The comparison among the compensation strategies further reveals their different behaviors. Temporal feature concatenation improves AP50 and MOTA over the baseline, but it brings limited or even negative gains in IDF1 and HOTA, especially on AIRMOT. This indicates that simply introducing adjacent-frame features does not necessarily provide reliable temporal information for downstream tracking. Explicit detector-location-guided compensation improves AP50, IDF1, HOTA, and IDSW over temporal feature concatenation, but slightly decreases MOTA on both datasets, suggesting that detector-provided target locations can make feature alignment more discriminative when these locations are reliable, while their dependence on current target locations may constrain compensation under weak or intermittent observations. In contrast, EECTracker combines probabilistic swarm-level spatial guidance with motion-prior-conditioned feature consistency, resulting in more consistent improvements across detection and tracking metrics.

![](images/aa7b6bc4b36da6895842ecbef52fe882d7dab004b86414fc9d6dd0a45383e1a0.jpg)  
Fig. 4. Qualitative comparison on AIRMOT under small target scales, weak visual responses, and cloud-background interference. Different tracked objects are marked by bounding boxes with different colors. Red arrows denote false positives, while yellow boxes and arrows indicate missed detections.

SCT-MOT [10] is a closely related motion-prior-guided UAV swarm tracking method built on SMTP for targetwise trajectory prediction and TG-STFF for trajectory-guided historical feature fusion. To compare this target-specific deterministic guidance with the swarm-motion-prior-guided feature compensation in EECTracker, we keep the remaining HOMATracker-based framework unchanged and evaluate three variants: (1) SCT-MOT with SMTP and TG-STFF; (2) PSMP+TG-STFF, which replaces SMTP with PSMP while retaining trajectory-guided fusion; and (3) PSMP+EEC Activation, which further replaces TG-STFF with consistency-guided feature compensation. The results are reported in Table III.

As shown in Table III, when coupled with the same TG-STFF fusion module, PSMP improves AP50, MOTA, and IDF1 on both datasets and reduces IDSW relative to SMTP, while maintaining comparable HOTA. This indicates that PSMP provides more effective current-frame motion guidance than the target-wise deterministic predictions used in SCT-MOT. More importantly, under the same PSMP, EEC Activation consistently outperforms TG-STFF across all reported metrics. On AIRMOT, it further improves MOTA, IDF1, and HOTA by 1.27, 1.09, and 0.58 percentage points, respectively, while reducing IDSW by 62; on UAVSwarm, the corresponding gains are 2.26, 0.98, and 0.88 percentage points, while reducing IDSW by 1. These results show that motionprior-conditioned feature consistency enables more effective historical feature utilization for downstream detection and tracking than trajectory-guided fusion alone.

The precision-recall curves in Fig. 6 provide complementary detection-level evidence. On both datasets, the compared methods perform similarly in the low-recall region, where detections are mainly determined by relatively salient targets. As recall increases, the detector includes more challenging UAV instances, such as weak or ambiguous targets. In this highrecall region, EECTracker preserves higher precision than the competing variants, indicating that swarm-motion-prior-guided feature compensation is more effective in maintaining reliable detections for challenging targets. This advantage emerges earlier on AIRMOT, becoming evident beyond approximately 0.3 recall, which is consistent with its smaller target scales and more frequent weak target responses. On UAVSwarm, the advantage mainly emerges beyond approximately 0.7 recall, suggesting that the compared variants remain close on easy targets, while EECTracker becomes more effective as increasingly challenging targets are included. These results are consistent with the AP50 improvements in Table II and further support the effectiveness of the proposed compensation strategy under weak and intermittent target observations.

![](images/a58b5f053e68203f449f197ab9c6024d2aa403c399ec121156725d1b4dd42de7.jpg)  
Fig. 5. Qualitative comparison on UAVSwarm under weak target responses, complex background clutter, and partial occlusion. Different tracked objects are marked by bounding boxes with different colors. Red arrows denote false positives, while yellow boxes and arrows indicate missed detections.

F. Cross-Tracker Plug-in Evaluation of the Proposed Feature Compensation

To evaluate whether the proposed feature compensation strategy can be integrated into different tracking frameworks, we conduct plug-in experiments on ByteTrack, OC-SORT, and FairMOT. ByteTrack and OC-SORT are representative TBD methods, while FairMOT is a representative JDT framework. For ByteTrack and OC-SORT, PSMP and EEC Activation are inserted into the detector used by the tracker, between the backbone/neck features and the detection head. The compensated feature representation is then fed into the original detection head, while the subsequent association pipeline remains unchanged. For FairMOT, the two components are inserted between the shared feature map and the detection/ReID heads, so that feature compensation is performed before joint detection and appearance prediction. In all cases, reliable historical tracklets produced by the original tracker are used to construct the probabilistic swarm motion prior, while the original detection heads, ReID branches, matching logic, and association strategies are kept unchanged. All experiments follow the same data splits, input resolution, training protocol, and inference settings as the corresponding base trackers.

As shown in Table IV, adding PSMP and EEC Activation improves all three trackers on both datasets, indicating that the proposed feature compensation strategy is not restricted to the HOMATracker-based implementation of EECTracker. For example, the ByteTrack enhanced with the proposed feature compensation improves MOTA, IDF1, and HOTA by 2.90, 0.93, and 0.61 percentage points on AIRMOT, and by 3.18, 2.26, and 0.82 percentage points on UAVSwarm, respectively. For FairMOT, the corresponding improvements are 2.53, 4.57, and 3.33 percentage points on AIRMOT, and 1.61, 1.32, and 0.83 percentage points on UAVSwarm. Consistent gains are also observed on OC-SORT, while IDSW decreases for all three trackers on both datasets. These results suggest that, when reliable historical tracklets are available, the proposed feature compensation strategy can be integrated into different MOT frameworks and provide consistent performance gains without modifying their original association strategies.

TABLE II  
ABLATION OF FEATURE COMPENSATION STRATEGIES ON AIRMOT AND UAVSWARM. “DET. LOC.” DENOTES WHETHER FEATURE COMPENSATION REQUIRES DETECTOR-PROVIDED TARGET LOCATIONS. AP50, MOTA, IDF1, AND HOTA ARE REPORTED AS PERCENTAGES.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">Det. loc.</td><td colspan="5">AIRMOT</td><td colspan="5">UAVSwarm</td></tr><tr><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td>Baseline Tracker</td><td>No</td><td>41.60</td><td>30.08</td><td>29.31</td><td>25.55</td><td>506</td><td>85.40</td><td>79.20</td><td>87.10</td><td>67.00</td><td>58</td></tr><tr><td>+ Temporal Feature Concatenation</td><td>No</td><td>42.90</td><td>32.23</td><td>28.06</td><td>24.01</td><td>497</td><td>86.10</td><td>80.61</td><td>87.35</td><td>67.15</td><td>64</td></tr><tr><td>+ Explicit Detector-Location- Guided Compensation</td><td>Yes</td><td>44.80</td><td>31.84</td><td>28.38</td><td>24.66</td><td>489</td><td>87.10</td><td>80.05</td><td>87.63</td><td>67.29</td><td>54</td></tr><tr><td>+ Swarm Motion Prior- Guided Compensation</td><td>No</td><td>46.10</td><td>36.19</td><td>31.94</td><td>26.15</td><td>400</td><td>89.30</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td></tr></table>

TABLE III  
COMPONENT-LEVEL COMPARISON WITH SCT-MOT ON AIRMOT AND UAVSWARM UNDER THE SAME HOMATRACKER-BASED FRAMEWORK.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">Motion Prior Feature Fusion</td><td rowspan="2"></td><td colspan="5">AIRMOT</td><td colspan="5">UAVSwarm</td></tr><tr><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td>SCT-MOT</td><td>SMTP</td><td>TG-STFF</td><td>45.30</td><td>32.30</td><td>30.15</td><td>25.54</td><td>476</td><td>86.30</td><td>81.90</td><td>88.45</td><td>68.56</td><td>56</td></tr><tr><td>PSMP + TG-STFF</td><td>PSMP</td><td>TG-STFF</td><td>45.40</td><td>34.92</td><td>30.85</td><td>25.57</td><td>462</td><td>87.40</td><td>82.45</td><td>89.21</td><td>68.28</td><td>55</td></tr><tr><td>EECTracker</td><td>PSMP</td><td>EEC Activation</td><td>46.10</td><td>36.19</td><td>31.94</td><td>26.15</td><td>400</td><td>89.30</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td></tr></table>

![](images/cdf35e06708d14bdd455d54d7dcbe7ce7a95c48f0264380ba7d62015fc0d4144.jpg)

(a) AIRMOT  
![](images/1c025de4ebee5703ff7419c4e51b0d0875e83ad6d2503a9d8559744c56aeb461.jpg)  
(b) UAVSwarm  
Fig. 6. Precision-recall curves of different feature compensation strategies on AIRMOT and UAVSwarm at IoU=0.5.

## G. Effectiveness and Robustness of Probabilistic Swarm Motion Prior Construction

The PSMP module constructs the image-plane probabilistic swarm motion prior used by EEC Activation for motion-priorguided feature compensation. We evaluate PSMP from three aspects: (1) the effectiveness of probabilistic swarm motion prior construction; (2) robustness to historical tracklet inputs, including tracklet availability and the train–inference gap; and (3) the effect of prior representation.

1) Effectiveness of Swarm-Aware Probabilistic Prior Construction: The probabilistic swarm motion prior constructed by PSMP serves as the common motion reference for both historical feature projection and Local EEC score computation. Its quality directly affects the reliability of subsequent feature compensation. Therefore, we evaluate the effectiveness of the proposed prior construction strategy by comparing three probabilistic motion-prior variants while keeping the downstream feature compensation module and the detection-and-tracking framework unchanged: (1) Kalman probabilistic prior, which estimates target-wise displacement and covariance using a classical Kalman filter. (2) Individual learned probabilistic prior, which removes the swarm-aware motion encoding from PSMP and learns each target’s displacement distribution and uncertainty from its individual historical motion. (3) Swarmaware learned probabilistic prior (EECTracker), which uses PSMP to model swarm-aware motion relations and target-wise uncertainty for constructing the swarm motion prior. For all variants, the resulting target-wise probabilistic estimates are aggregated using the same weighting strategy to construct the final motion prior for downstream feature compensation. The results are reported in Table V.

TABLE IV  
CROSS-TRACKER PLUG-IN EVALUATION OF EEC-BASED FEATURE COMPENSATION ON AIRMOT AND UAVSWARM.“COMP.” INDICATES WHETHER PSMP AND EEC ACTIVATION ARE ENABLED.
<table><tr><td rowspan="2">Tracker</td><td rowspan="2">Comp.</td><td colspan="4">AIRMOT</td><td colspan="4">UAVSwarm</td></tr><tr><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td rowspan="2">ByteTrack</td><td>× &gt;</td><td>24.35</td><td>25.72</td><td>23.56</td><td>1707</td><td>65.00</td><td>76.50</td><td>60.60</td><td>67</td></tr><tr><td></td><td>27.25</td><td>26.65</td><td>24.17</td><td>1254</td><td>68.18</td><td>78.76</td><td>61.42</td><td>65</td></tr><tr><td rowspan="2">OC-SORT</td><td>×</td><td>20.88</td><td>23.27</td><td>22.07</td><td>522</td><td>75.70</td><td>81.80</td><td>64.70</td><td>709</td></tr><tr><td>V</td><td>21.53</td><td>23.31</td><td>22.15</td><td>481</td><td>76.44</td><td>82.33</td><td>64.88</td><td>578</td></tr><tr><td rowspan="2">FairMOT</td><td>× &gt;</td><td>18.19</td><td>17.41</td><td>17.56</td><td>476</td><td>67.70</td><td>73.20</td><td>59.20</td><td>590</td></tr><tr><td></td><td>20.72</td><td>21.98</td><td>20.89</td><td>425</td><td>69.31</td><td>74.52</td><td>60.03</td><td>561</td></tr></table>

TABLE V

ABLATION OF PROBABILISTIC MOTION PRIOR CONSTRUCTION ON AIRMOT AND UAVSWARM. “S.A.”, “GEN.”, AND “KF” DENOTE SWARM-AWARE MODELING, PRIOR GENERATOR, AND KALMAN FILTER, RESPECTIVELY.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">S.A.</td><td rowspan="2">Gen.</td><td colspan="5">AIRMOT</td><td colspan="5">UAVSwarm</td></tr><tr><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td> $\mathbf { I D S W } \downarrow$ </td><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td>Kalman Probabilistic Prior</td><td>X</td><td>KF</td><td>44.30</td><td>34.86</td><td>30.49</td><td>25.56</td><td>468</td><td>87.50</td><td>83.95</td><td>89.06</td><td>68.43</td><td>73</td></tr><tr><td>Individual Learned Prior</td><td>X</td><td>Learned</td><td>45.20</td><td>35.68</td><td>30.86</td><td>25.47</td><td>447</td><td>88.20</td><td>84.23</td><td>89.57</td><td>68.68</td><td>67</td></tr><tr><td>Swarm-Aware Learned Prior</td><td>√</td><td>Learned</td><td>46.10</td><td>36.19</td><td>31.94</td><td>26.15</td><td>400</td><td>89.30</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td></tr></table>

As shown in Table V, the proposed swarm-aware learned probabilistic prior achieves the best overall performance on both AIRMOT and UAVSwarm. Compared with the individual learned prior, it improves AP50, MOTA, IDF1, and HOTA by 0.90, 0.51, 1.08, and 0.68 percentage points on AIRMOT, respectively, while reducing IDSW by 47. On UAVSwarm, the corresponding improvements are 1.10, 0.48, 0.62, and 0.48 percentage points, respectively, with 13 fewer ID switches. These results indicate that, under the same probabilistic prediction and aggregation framework, incorporating shared short-term swarm motion information provides more effective probabilistic guidance for current-frame feature compensation than modeling individual target motion independently.

Meanwhile, the individual learned prior also outperforms the Kalman probabilistic prior on most metrics. On AIRMOT, it improves AP50, MOTA, and IDF1 by 0.90, 0.82, and 0.37 percentage points, respectively, and reduces IDSW by 21, while HOTA remains comparable. On UAVSwarm, it improves AP50, MOTA, IDF1, and HOTA by 0.70, 0.28, 0.51, and 0.25 percentage points, respectively, with 6 fewer ID switches. These results suggest that learned target-wise motion and uncertainty provide more effective probabilistic guidance for feature compensation than classical target-wise recursive filtering.

2) Robustness to Historical Tracklet Inputs: Since PSMP constructs the swarm motion prior from reliable historical tracklets, we analyze its robustness from two aspects: reliabletracklet availability and the train–inference gap of historical tracklet inputs. The former analyzes how PSMP behaves when the number of reliable tracklets decreases, while the latter examines the gap between online predicted tracklets and annotated historical tracklets.

Tracklet availability: To evaluate the sensitivity of PSMP to reduced reliable-tracklet availability, we vary the proportion of reliable tracklets used for motion-prior construction. During inference, the overall EECTracker framework and inference pipeline are kept unchanged. The 100% setting uses all tracklets satisfying the reliable-tracklet selection rule within the 8-frame historical window, whereas the 75%, 50%, and 25% settings randomly retain the corresponding proportions of these tracklets. The 0% setting disables PSMP and reduces to the baseline tracker without the swarm motion prior.

SENSITIVITY ANALYSIS OF PSMP TO RELIABLE TRACKLET AVAILABILITY.  
TABLE VI
<table><tr><td>Dataset</td><td>Ratio</td><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td rowspan="5">AIRMOT</td><td>100%</td><td>46.10</td><td>36.19</td><td>31.94</td><td>26.15</td><td>400</td></tr><tr><td>75%</td><td>45.60</td><td>35.86</td><td>31.33</td><td>26.06</td><td>405</td></tr><tr><td>50%</td><td>44.80</td><td>35.27</td><td>30.92</td><td>25.97</td><td>413</td></tr><tr><td>25%</td><td>43.30</td><td>34.63</td><td>29.75</td><td>25.24</td><td>458</td></tr><tr><td>0%</td><td>41.60</td><td>30.08</td><td>29.31</td><td>25.55</td><td>506</td></tr><tr><td rowspan="5">UAVSwarm</td><td>100%</td><td>89.30</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td></tr><tr><td>75%</td><td>89.10</td><td>84.29</td><td>89.62</td><td>68.90</td><td>62</td></tr><tr><td>50%</td><td>88.50</td><td>83.35</td><td>89.23</td><td>68.54</td><td>69</td></tr><tr><td>25%</td><td>86.60</td><td>82.82</td><td>89.08</td><td>67.15</td><td>73</td></tr><tr><td>0%</td><td>85.40</td><td>79.20</td><td>87.10</td><td>67.00</td><td>58</td></tr></table>

As shown in Table VI, tracking performance degrades gradually as the proportion of reliable tracklets decreases from 100% to 25%. On AIRMOT, the 75% and 50% settings remain close to the full setting, with MOTA drops of only 0.33 and 0.92 percentage points, respectively. Even with only 25% reliable tracklets, PSMP still improves AP50, MOTA, and IDF1 over the 0% baseline. A similar trend is observed on UAVSwarm, where the 50% setting retains clear performance gains, and the 25% setting still outperforms the baseline in AP50, MOTA, IDF1, and HOTA. These results indicate that PSMP remains effective with limited reliable-tracklet availability, although its benefit gradually decreases as fewer historical tracklets are available for motion-prior construction.

TABLE VII  
TRAIN–INFERENCE GAP ANALYSIS OF HISTORICAL TRACKLET INPUTS FOR PSMP.
<table><tr><td>Dataset</td><td>Input</td><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td rowspan="2">AIRMOT</td><td>Online</td><td>46.10</td><td>36.19</td><td>31.94</td><td>26.15</td><td>400</td></tr><tr><td>GT</td><td>46.10</td><td>36.27</td><td>31.97</td><td>26.20</td><td>398</td></tr><tr><td rowspan="2">UAVSwarm</td><td>Online</td><td>89.30</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td></tr><tr><td>GT</td><td>89.30</td><td>84.86</td><td>90.22</td><td>69.17</td><td>53</td></tr></table>

Train–inference gap of historical tracklet inputs: PSMP is trained with annotated trajectories but constructs the motion prior from online tracklets during inference. To evaluate this train–inference gap, we compare two settings: (1) Online PSMP, the default EECTracker setting, which constructs the motion prior from online tracklets generated by the tracking pipeline; and (2) GT-tracklet PSMP, which constructs the motion prior from the corresponding ground-truth trajectories within the same historical window. The GT setting is used only for sensitivity analysis and does not use future-frame annotations. All other settings are kept unchanged.

As shown in Table VII, the Online setting achieves performance very close to the GT setting on both datasets. The MOTA/IDF1/HOTA gaps are only 0.08/0.03/0.05 percentage points on AIRMOT and 0.15/0.03/0.01 percentage points on UAVSwarm, with only 2 and 1 additional ID switches, respectively, while AP50 remains unchanged. These results indicate that PSMP can use online tracklets for motionprior construction with only marginal degradation in the final tracking performance.

3) Effect of Probabilistic Prior Representation: To examine whether explicitly retaining target-wise probabilistic components provides additional benefit, we compare the default moment-matched compact Gaussian prior with an explicit Gaussian mixture representation. All training and inference settings remain unchanged, and the two variants differ only in the probabilistic representation used for motion-prior-guided feature compensation. The default PSMP aggregates the targetwise distributions into a single Gaussian through moment matching, whereas the comparison retains them as an explicit mixture:

$$
p _ { \operatorname* { m i x } } ( \delta ) = \sum _ { i = 1 } ^ { n _ { T } } \alpha _ { i } \mathcal { N } ( \delta ; \hat { \mu } _ { i } , \hat { \Sigma } _ { i } ) , \mathcal { M } _ { t , \operatorname* { m i x } } = \left\{ \hat { \mathbf { v } } _ { s w , t } , p _ { \operatorname* { m i x } } ( \delta ) \right\}\tag{32}
$$

where $\hat { \mu } _ { i }$ and $\hat { \Sigma } _ { i }$ are the predicted mean and covariance of the spatial offset distribution for the i-th reliable tracklet, and $\alpha _ { i }$ is the corresponding normalized aggregation weight. The normalized mixture distribution is then used within the same EEC Activation pipeline.

As shown in Table VIII, the compact Gaussian prior achieves slightly but consistently better performance than the explicit Gaussian mixture prior on both datasets. On AIRMOT, it improves MOTA, IDF1, and HOTA by 0.11, 0.54, and 0.66 percentage points, respectively, and reduces IDSW by 21. On UAVSwarm, the corresponding improvements are 0.43, 0.77, and 0.33 percentage points, with 12 fewer ID switches.

TABLE VIII  
COMPARISON BETWEEN THE COMPACT GAUSSIAN PRIOR AND THE EXPLICIT GAUSSIAN MIXTURE PRIOR IN PSMP. “COMPACT” DENOTES THE COMPACT GAUSSIAN PRIOR, AND “MIXTURE” DENOTES THE GAUSSIAN MIXTURE PRIOR.
<table><tr><td>Dataset</td><td>Prior</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td rowspan="2">AIRMOT</td><td>Compact</td><td>36.19</td><td>31.94</td><td>26.15</td><td>400</td></tr><tr><td>Mixture</td><td>36.08</td><td>31.40</td><td>25.49</td><td>421</td></tr><tr><td rowspan="2">UAVSwarm</td><td>Compact</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td></tr><tr><td>Mixture</td><td>84.28</td><td>89.42</td><td>68.83</td><td>66</td></tr></table>

These results indicate that explicitly preserving the targetwise Gaussian components does not provide additional benefit for the proposed feature compensation under the evaluated settings.

## H. Ablation Study of EEC Activation

To further analyze the effectiveness of EEC Activation, we conduct ablation studies from four aspects: the Local EEC score, the local window size used for residual statistics, the channel-adaptive gating strategy, and qualitative visualization of the intermediate feature compensation process.

1) Effectiveness of the Local EEC Score: To evaluate the effectiveness of the Local EEC score, we compare different residual-statistics-based activation strategies within the same HOMATracker-based compensation framework. All variants use the same PSMP module and downstream detection-andtracking framework, while differing in how the motion-priorprojected historical features are evaluated and weighted before fusion with the current-frame features.

Specifically, we compare four variants: (1) Direct fusion, which directly fuses motion-prior-projected historical features with current-frame features without EEC Activation; (2) Entropy-only, which employs only the residual entropy term in EEC Activation; (3) Energy-only, which uses only the feature residual energy term in EEC Activation; (4) Local EEC, which jointly incorporates residual energy and residual entropy. For the entropy-only, energy-only, and Local EEC variants, the same channel-adaptive gating strategy is used to generate the soft consistency mask. The weighted historical features are then fused with the current-frame features and fed into the downstream detection and association modules. The results are reported in Table IX.

As shown in Table IX, the variants with residual-statisticsbased activation generally outperform direct fusion on both datasets. This indicates that motion-prior-guided projection alone is insufficient, because the projected historical features may also carry background interference and spatially mismatched responses caused by motion uncertainty. Compared with direct fusion, both the Entropy-only and Energyonly variants improve most detection and tracking metrics, demonstrating the benefit of explicit cross-frame consistency evaluation. Among them, the energy-only variant is generally more effective than the entropy-only variant, indicating that residual magnitude provides a stronger cue than spatialdispersion information alone for evaluating cross-frame feature consistency.

TABLE IX  
ABLATION OF THE LOCAL EEC SCORE ON AIRMOT AND UAVSWARM. E, S, AND C.A. DENOTE RESIDUAL ENERGY, RESIDUAL ENTROPY, AND CHANNEL-ADAPTIVE GATING, RESPECTIVELY.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">E</td><td rowspan="2">S</td><td rowspan="2">C.A.</td><td colspan="5">AIRMOT</td><td colspan="5">UAVSwarm</td></tr><tr><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td> $\mathbf { I D S W } \downarrow$ </td><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td> $\mathbf { I D S W } \downarrow$ </td></tr><tr><td>Direct Fusion</td><td>X</td><td>X</td><td>X</td><td>42.60</td><td>33.15</td><td>30.48</td><td>25.14</td><td>470</td><td>85.80</td><td>82.49</td><td>88.96</td><td>67.79</td><td>69</td></tr><tr><td>Entropy-only</td><td>X</td><td>√</td><td>√</td><td>44.90</td><td>33.78</td><td>31.03</td><td>25.48</td><td>447</td><td>86.70</td><td>83.57</td><td>89.05</td><td>67.65</td><td>68</td></tr><tr><td>Energy-only</td><td>√</td><td>X</td><td>√</td><td>45.40</td><td>35.60</td><td>31.50</td><td>25.62</td><td>428</td><td>87.60</td><td>83.87</td><td>89.08</td><td>68.21</td><td>67</td></tr><tr><td>Local EEC</td><td>√</td><td>√</td><td>√</td><td>46.10</td><td>36.19</td><td>31.94</td><td>26.15</td><td>400</td><td>89.30</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td></tr></table>

The full Local EEC score achieves the best overall performance on both datasets. On AIRMOT, compared with the energy-only variant, Local EEC improves AP50, MOTA, IDF1 and HOTA by 0.70, 0.59, 0.44 and 0.53 percentage points, respectively, while reducing IDSW by 28. On UAVSwarm, the corresponding improvements are 1.70, 0.84, 1.11, 0.95 percentage points, with 13 fewer ID switches. This demonstrates that the joint use of residual energy and residual entropy enables more effective motion-prior-conditioned feature consistency evaluation for selective feature compensation.

2) Effect of Local Window Size in Local EEC: To analyze the effect of different local support regions in Local EEC, we conduct an ablation study on the neighborhood window used to compute local residual statistics. In this experiment, the PSMP module, EEC Activation and the downstream detectionand-tracking framework are kept unchanged, and only the neighborhood size used in Local EEC is varied. Specifically, we evaluate four window sizes, $1 \times 1 , 3 \times 3 , 5 \times 5$ , and $7 \times 7$ on AIRMOT and UAVSwarm. The results are reported in Table X.

On AIRMOT, performance improves from 1 × 1 to $3 \times 3$ and then decreases with larger windows, with $3 \times 3$ achieving the best overall results. Compared with the $5 \times 5$ setting, it improves AP50, MOTA, IDF1, and HOTA by 1.10, 0.37, 0.82 and 0.42 percentage points, respectively, while reducing IDSW by 20. This suggests that, on AIRMOT, where UAV targets are relatively small and feature responses are weak, the $1 \times 1$ window provides insufficient spatial support, whereas larger windows may introduce more background interference into the residual statistics. Therefore, the $3 \times 3$ window achieves a better balance between spatial support and interference suppression. In contrast, UAVSwarm favors a larger spatial support, with $7 \times 7$ yielding the best performance despite minor fluctuations at intermediate window sizes. Compared with the 3×3 setting, the $7 \times 7$ window improves AP50, MOTA, IDF1, and HOTA by 0.90, 0.28, 0.70, and 0.51 percentage points, respectively, and reduces IDSW by 12. This suggests that the relatively larger target-response regions in UAVSwarm benefit from a wider local window, which captures more complete spatial information for consistency estimation. Overall, these results indicate that the optimal local window size depends on the spatial extent of target responses: overly small windows may provide insufficient spatial support, whereas overly large windows may introduce excessive background responses.

3) Effect of Channel-Adaptive Gating in EEC Activation: To analyze the effect of channel-adaptive gating in EEC Activation, we compare two gating strategies while keeping the PSMP module, the local window size, the feature compensation framework, and the downstream detection and tracking framework unchanged. Shared gating uses a common activation threshold and scaling factor for all channels, whereas channel-adaptive gating learns channel-specific activation parameters. The results are reported in Table XI.

TABLE X  
EFFECT OF LOCAL WINDOW SIZE IN LOCAL EEC ON AIRMOT AND UAVSWARM.
<table><tr><td>Window</td><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td colspan="6">AIRMOT</td></tr><tr><td> $1 \times 1$ </td><td>43.50</td><td>35.49</td><td>30.62</td><td>25.35</td><td>423</td></tr><tr><td> $3 \times 3$ </td><td>46.10</td><td>36.19</td><td>31.94</td><td>26.15</td><td>400</td></tr><tr><td> $5 \times 5$ </td><td>45.00</td><td>35.82</td><td>31.12</td><td>25.73</td><td>420</td></tr><tr><td> $7 \times 7$ </td><td>44.20</td><td>35.74</td><td>30.58</td><td>25.40</td><td>418</td></tr><tr><td colspan="6">UAVSwarm</td></tr><tr><td> $1 \times 1$ </td><td>87.40</td><td>83.58</td><td>89.42</td><td>68.41</td><td>73</td></tr><tr><td> $3 \times 3$ </td><td>88.40</td><td>84.43</td><td>89.49</td><td>68.65</td><td>66</td></tr><tr><td> $5 \times 5$ </td><td>88.30</td><td>84.31</td><td>89.23</td><td>68.54</td><td>67</td></tr><tr><td> $7 \times 7$ </td><td>89.30</td><td>84.71</td><td>90.19</td><td>69.16</td><td>54</td></tr></table>

TABLE XI

EFFECT OF CHANNEL-ADAPTIVE GATING IN EEC ACTIVATION ON AIRMOT AND UAVSWARM. “ADAPTIVE” DENOTES CHANNEL-ADAPTIVE GATING.
<table><tr><td>Dataset</td><td>Gating</td><td>AP50↑</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑</td><td>IDSW↓</td></tr><tr><td>AIRMOT</td><td>Shared Adaptive</td><td>45.30 46.10</td><td>35.69 36.19</td><td>30.67 31.94</td><td>25.28 26.15</td><td>434 400</td></tr><tr><td>UAVSwarm</td><td>Shared Adaptive</td><td>88.50 89.30</td><td>84.47 84.71</td><td>89.73 90.19</td><td>68.94 69.16</td><td>73 54</td></tr></table>

As shown in Table XI, channel-adaptive gating consistently outperforms shared gating on both AIRMOT and UAVSwarm. On AIRMOT, it improves AP50, MOTA, IDF1, and HOTA by 0.80, 0.50, 1.27, and 0.87 percentage points, respectively, while reducing IDSW by 34. On UAVSwarm, it improves AP50, MOTA, IDF1, and HOTA by 0.80, 0.24, 0.46, and 0.22 percentage points, respectively, and reduces IDSW by 19. These results suggest that residual-consistency responses vary across feature channels, making shared activation parameters less effective in capturing channel-wise differences. Channelspecific parameters allow EEC Activation to adapt to these channel-wise variations, thereby preserving motion-consistent historical responses while suppressing interfering responses more selectively.

4) Qualitative Analysis of EEC Activation: To provide qualitative evidence for EEC Activation, we visualize the intermediate feature compensation process on a representative frame from the AIRMOT-02 sequence, as shown in Fig. 7. The figure presents the following stages: the input image patch, the current-frame feature before compensation, the motion-prior-projected historical feature, the Local EEC score map, the channel-wise soft consistency mask, and the final compensated feature map. These intermediate results illustrate how EEC Activation performs feature compensation through motion-prior-guided projection, Local EEC-based consistency evaluation, soft mask construction, and selective weighting followed by feature fusion.

![](images/399ac02ba1526d23b21fec37d1254174270030554848e99f7ce9a11dddff1f74.jpg)  
Fig. 7. Visualization of the intermediate feature compensation process of EEC Activation on AIRMOT. The subfigures show (a) the input image patch, (b) the current feature before compensation, (c) the motion-prior-projected historical feature, (d) the Local EEC score map, (e) the channel-wise soft consistenc mask, and (f) the final compensated feature map. The color bar indicates the response magnitude from low to high.

Before feature compensation, the target response in the current frame is weak and partially obscured by background responses. After projecting the historical feature toward the current frame under the probabilistic swarm motion prior, the projected feature retains a clearer target response but also carries noticeable background interference. This indicates that motion-prior-guided projection alone is insufficient. The Local EEC score map further shows that the target response exhibits a relatively low score, whereas many interfering responses have higher values. This is consistent with the intended design of Local EEC: motion-consistent cross-frame target responses tend to produce lower residual energy and entropy, while less-consistent interfering responses generally exhibit larger residual magnitudes or higher spatial dispersion. Consequently, Local EEC provides a discriminative featurespace measure for separating motion-consistent potential target responses from interfering historical responses.

Based on this score map, the soft consistency mask converts low-score, high-consistency regions into larger activation weights for feature compensation. After applying the soft mask weighting and feature fusion, motion-consistent potential UAV target responses are strengthened and background interference is reduced in the compensated current-frame feature. These results show that EEC Activation does not simply propagate historical features. Instead, it uses the Local EEC score map to selectively incorporate motion-prior-projected historical information, thereby strengthening weak target feature responses and reducing background interference.

## I. Runtime Efficiency and Embedded Deployment

For airborne optical UAV swarm tracking, online inference efficiency and embedded deployability are important systemlevel considerations. Therefore, in addition to tracking accuracy, we further evaluate a lightweight implementation of EECTracker on an NVIDIA Jetson Orin NX platform to assess its embedded deployment performance.

Considering the limited computational resources of embedded devices, we implement a lightweight variant, denoted as EECTracker-Light. It adopts HOMATracker-Light as the base framework, using YOLOX-S for detection together with a lightweight appearance branch for appearance embedding. PSMP and EEC Activation are then integrated into this lightweight tracking pipeline. We select the representative TBD method ByteTrack as the comparison method due to its efficiency and deployment flexibility. ByteTrack is also configured with the same YOLOX-S detector and denoted as ByteTrack-Light. The neural-network inference components of both methods are accelerated with TensorRT on Jetson Orin NX.

As shown in Table XII, EECTracker-Light achieves 31.68% MOTA, 30.03% IDF1, and 24.79% HOTA at 21.0 FPS on AIR-MOT, improving MOTA, IDF1, and HOTA over ByteTrack-Light by 8.22, 5.45, and 1.93 percentage points, respectively.

TABLE XII  
EMBEDDED DEPLOYMENT PERFORMANCE ON JETSON ORIN NX.
<table><tr><td>Method</td><td>Dataset</td><td>MOTA↑</td><td>IDF1↑</td><td>HOTA↑ FPS↑</td><td></td></tr><tr><td>ByteTrack-Light</td><td>AIRMOT UAVSwarm</td><td>23.46 63.71</td><td>24.58 75.13</td><td>22.86</td><td>24.6</td></tr><tr><td>EECTracker-Light</td><td>AIRMOT</td><td>31.68</td><td>30.03</td><td>59.60 24.79</td><td>23.9 21.0</td></tr></table>

On UAVSwarm, it achieves 78.73% MOTA, 86.35% IDF1, and 65.60% HOTA at 21.3 FPS, with corresponding improvements of 15.02, 11.22, and 6.00 percentage points over ByteTrack-Light. Although EECTracker-Light incurs a modest reduction in inference speed, it maintains an end-to-end rate above 20 FPS on both datasets while providing substantial trackingperformance gains. These results indicate that EECTracker can be adapted to a lightweight online tracking pipeline while retaining both competitive tracking accuracy and practical inference efficiency, supporting its deployment in embedded airborne optical UAV swarm tracking systems.

## V. CONCLUSION

This paper addresses stable online UAV swarm tracking under challenging airborne optical imaging conditions, where small target scales and complex background interference can weaken target feature responses and disrupt temporal observations. We presented EECTracker, a swarm-motion-priorguided joint detection and tracking framework that exploits swarm-level historical motion information to compensate potential UAV target responses. Within EECTracker, the proposed feature compensation strategy constructs a probabilistic swarm motion prior from available reliable historical tracklets by modeling the shared short-term motion tendency of the swarm and its uncertainty. The resulting prior characterizes statistically reachable image-plane regions and provides spatial guidance for historical feature projection. EEC Activation further evaluates motion-prior-conditioned cross-frame feature consistency and selectively incorporates projected historical responses over potential target regions through channeladaptive soft gating. The compensated feature representation is then used for detection and association, while updated tracklets support subsequent motion-prior construction in an online closed-loop manner.

Extensive experiments on AIRMOT and UAVSwarm demonstrate that the proposed swarm-motion-prior-guided feature compensation strategy provides robust swarm-level motion guidance for downstream compensation, while EEC Activation effectively modulates the influence of projected historical responses according to motion-prior-conditioned feature consistency. Building on these components, EECTracker achieves superior overall tracking performance compared with state-of-the-art methods. The Jetson Orin NX experiments further indicate the potential of EECTracker for embedded deployment on airborne platforms. A current limitation is that the benefit of probabilistic swarm motion guidance decreases when reliable historical tracklets become highly sparse. Future work will focus on more robust swarm motion modeling under severely limited historical observations.

## ACKNOWLEDGMENTS

This study was co-supported by the National Key Research and Development Program of China (2023YFC3341100), National Natural Science Foundation of China (No. 52672527).

## REFERENCES

[1] G. Ding, Y. Ren, Y. Liu, Q. Zhao, and S. Li, “Vision-based antiunmanned aerial technology: Opportunities and challenges,” IEEE Geoscience and Remote Sensing Magazine, vol. 13, no. 4, pp. 382–405, 2025.

[2] B. Huang, J.-A. Li, J.-J. Chen, G. Wang, J. Zhao, and T.-F. Xu, “Antiuav410: A thermal infrared benchmark and customized scheme for tracking drones in the wild,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 5, pp. 2852–2865, 2024.

[3] S. Li, X. Yang, X. Wang, D. Zeng, H. Ye, and Q. Zhao, “Learning targetaware vision transformers for real-time uav tracking,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–18, 2024.

[4] X. Yuan, G. Cheng, K. Yan, Q. Zeng, and J. Han, “Small object detection via coarse-to-fine proposal generation and imitation learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 6317–6327.

[5] Y. Zhang, C. Wang, X. Wang, W. Zeng, and W. Liu, “Fairmot: On the fairness of detection and re-identification in multiple object tracking,” Int. J. Comput. Vis., vol. 129, pp. 3069–3087, 2021.

[6] Y.-F. Zhang, P.-Z. Sun, Y. Jiang, D.-D. Yu, F.-C. Weng, Z.-H. Yuan, P. Luo, W.-Y. Liu, and X.-G. Wang, “Bytetrack: Multi-object tracking by associating every detection box,” in ECCV 2022: Proceedings of the European conference on computer vision. Springer, 2022, pp. 1–21.

[7] F. Zeng, B. Dong, Y. Zhang, T. Wang, X. Zhang, and Y. Wei, “Motr: End-to-end multiple-object tracking with transformer,” in European conference on computer vision. Springer, 2022, pp. 659–675.

[8] E. Yu, T. Wang, Z. Li, Y. Zhang, X. Zhang, and W. Tao, “Motrv3: Release-fetch supervision for end-to-end multi-object tracking,” ArXiv, vol. abs/2305.14298, 2023.

[9] Z. Shen, K. Cai, P. Zhao, and X. Luo, “An interactively motion-assisted network for multiple object tracking in complex traffic scenes,” IEEE Transactions on Intelligent Transportation Systems, vol. 25, no. 2, pp. 1992–2004, 2023.

[10] Z. Chu, T. Song, R. Jin, S. He, D. Lin, and S. Cheng, “Sct-mot: Enhancing air-to-air multiple uavs tracking with swarm-coupled motion and trajectory guidance,” IEEE Transactions on Aerospace and Electronic Systems, pp. 1–18, 2026.

[11] J. Liu, L. Plotegher, E. Roura, C. d. S. Junior, and S. He, “Realtime detection for small uavs: Combining yolo and multiframe motion analysis,” IEEE Transactions on Aerospace and Electronic Systems, vol. 61, no. 5, pp. 13 419–13 433, 2025.

[12] F. Chen, C. Gao, F. Liu, Y. Zhao, Y. Zhou, D. Meng, and W. Zuo, “Local patch network with global attention for infrared small target detection,” IEEE Transactions on Aerospace and Electronic Systems, vol. 58, no. 5, pp. 3979–3991, 2022.

[13] J. Wang, Z. Wu, D. Chen, C. Luo, X. Dai, L. Yuan, and Y.-G. Jiang, “Omnitracker: Unifying visual object tracking by tracking-withdetection,” IEEE transactions on pattern analysis and machine intelligence, vol. 47, no. 4, pp. 3159–3174, 2025.

[14] B. Huang, Z. Dou, J. Chen, J. Li, N. Shen, Y. Wang, and T. Xu, “Searching region-free and template-free siamese network for tracking drones in tir videos,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[15] H. Fang, C. Wu, X. Wang, F. Zhou, Y. Chang, and L. Yan, “Online infrared uav target tracking with enhanced context-awareness and pixelwise attention modulation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–17, 2024.

[16] X. Cui, X. Li, P. Wu, X. Liu, and S. He, “TAPTrack: An efficient temporal-aware prompt tracker for infrared anti-uav tracking,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, pp. 1–14, 2026.

[17] Z. Chu, T. Song, R. Jin, and T. Jiang, “An experimental evaluation based on new air-to-air multi-uav tracking dataset,” in 2023 IEEE International Conference on Unmanned Systems (ICUS). IEEE, 2023, pp. 671–676.

[18] C. Wang, Y. Su, J. Wang, T. Wang, and Q. Gao, “Uavswarm dataset: An unmanned aerial vehicle swarm dataset for multiple object tracking,” Remote Sensing, vol. 14, no. 11, p. 2601, 2022.

[19] Z. Chu, T. Song, R. Jin, D. Lin, H. Shen, and M. LYU, “Vision-based swarm tracking of multiple uavs in air-to-air scenarios,” Chinese Journal of Aeronautics, p. 103558, 2025.

[20] C. Wang, Y. Su, L. Wang, T. Wang, J. Wang, and Q. Gao, “Multiobject continuous robust tracking algorithm for anti-uav swarm,” Acta Aeronautica et Astronautica Sinica, vol. 45, no. 7, 2024.

[21] Z. Chu, T. Song, R. Jin, and D. Lin, “Vision-based air-to-air multi-uavs tracking,” Acta Aeronautica et Astronautica Sinica, vol. 45, no. 14, 2024.

[22] J. Liu, L. Plotegher, E. Roura, and S. He, “Gl-dt: multi-uav detection and tracking with global-local integration,” IEEE Transactions on Geoscience and Remote Sensing, 2026.

[23] C. Xu, R. T. Tan, Y. Tan, S. Chen, Y. G. Wang, X. Wang, and Y. Wang, “Eqmotion: Equivariant multi-agent motion prediction with invariant interaction reasoning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 1410–1420.

[24] X. Zhou, V. Koltun, and P. Krahenb ¨ uhl, “Tracking objects as points,” in¨ European conference on computer vision. Springer, 2020, pp. 474–490.

[25] J. Wu, J. Cao, L. Song, Y. Wang, M. Yang, and J. Yuan, “Track to detect and segment: An online multi-object tracker,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12 352–12 361.

[26] H. Q. Qin, T. Li, T. Xu, J. Xu, Y. Fang, and J. Li, “Pptracker: Tracking uav swarms with prior prompt,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 6583–6590.

[27] P. Sun, J. Cao, Y. Jiang, R. Zhang, E. Xie, Z. Yuan, C. Wang, and P. Luo, “Transtrack: Multiple object tracking with transformer,” arXiv preprint arXiv:2012.15460, 2020.

[28] Q. Xu, L. Wang, W. Sheng, Q. Zhang, and W. An, “Trajectory-centric transformer for multiple objects tracking in satellite videos,” IEEE Transactions on Multimedia, 2026.

[29] H. Zhang, J. Wan, J. Zhang, D. Yuan, X. Li, and Y. Yang, “P2ftrack: Multi-object tracking with motion prior and feature posterior,” ACM Transactions on Multimedia Computing, Communications and Applications, vol. 21, no. 1, pp. 1–22, 2024.

[30] M. Zhao, S. Li, H. Wang, J. Yang, Y. Sun, and Y. Gu, “Mp2net: Mask propagation and motion prediction network for multiobject tracking in satellite videos,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[31] X. Zhu, Y. Wang, J. Dai, L. Yuan, and Y. Wei, “Flow-guided feature aggregation for video object detection,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 408–417.

[32] G. Bertasius, L. Torresani, and J. Shi, “Object detection in video with spatiotemporal sampling networks,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 331–346.

[33] Y. Zheng, C. He, X. Chen, H. Zhang, T. Qu, and D. Wang, “Dfa-mot: A dynamic field-aware multi-object tracking framework for unmanned aerial vehicles,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.

[34] T. Wang, L. Zhu, and H. Huang, “Enhancing real-time object detection with optical flow-guided streaming perception,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 35, no. 5, pp. 4816– 4830, 2025.

[35] Y. Huang, X. Zhi, J. Hu, L. Yu, Q. Han, W. Chen, and W. Zhang, “Lmaformer: Local motion aware transformer for small moving infrared target detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–17, 2024.

[36] W. Lu, B. Sun, S. Li, and X. Li, “Fma-net: Flow-driven motion-aware network for multi-object tracking in satellite videos,” IEEE Transactions on Geoscience and Remote Sensing, 2025.

[37] H. Song, “Robust visual tracking via online informative feature selection,” Electronics Letters, vol. 50, no. 25, pp. 1931–1933, 2014.

[38] A. S. Wannenwetsch, M. Keuper, and S. Roth, “Probflow: Joint optical flow and uncertainty estimation,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 1173–1182.

[39] W.-H. Luo, J.-L. Xing, A. Milan, X.-Q. Zhang, W. Liu, and T.-K. Kim, “Multiple object tracking: A literature review,” Artif. Intell., vol. 293, p. 103448, 2021.

[40] N. Wojke, A. Bewley, and D. Paulus, “Simple online and realtime tracking with a deep association metric,” in ICIP 2017: Proceedings of the IEEE international conference on image processing. IEEE, 2017, pp. 3645–3649.

[41] J.-K. Cao, J.-M. Pang, X.-S. Weng, R. Khirodkar, and K. Kitani, “Observation-centric sort: Rethinking sort for robust multi-object track-

ing,” in CVPR 2023: Proceedings of the IEEE /CVF conference on computer vision and pattern recognition, 2023, pp. 9686–9696.

[42] M.-Z. Yang, G.-X. Han, B. Yan, W.-H. Zhang, J.-Q. Qi, H.-C. Lu, and D. Wang, “Hybrid-sort: Weak cues matter for online multi-object tracking,” in AAAI 2024: Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 7, 2024, pp. 6504–6512.

![](images/701cde554fdb18ca3710bc490fd795e955c4f3b06f2ff9b912f61e5866c475c8.jpg)

Zhaochen Chu received the B.E. degree in Flight Vehicle Design and Engineering from Beijing Institute of Technology, Beijing, China, in 2021. He is currently pursuing the Ph.D. degree with the China-UAE Belt and Road Joint Laboratory on Intelligent Unmanned Systems with the School of Aerospace Engineering, Beijing Institute of Technology. His research interests include small target UAV detection and swarm UAV tracking based on visual imagery.

![](images/d4f26dd88d3cc4ce64dd2c56ad920547c401fd36e80ef42a3d4d6fae3be933fa.jpg)

Tao Song received the B.E. degree in Guidance, Navigation and Control, and Ph.D. degree in Aircraft Design from Beijing Institute of Technology, Beijing, China, in 2008 and 2014, respectively. He is currently an Associate Professor with the School of Aerospace Engineering, Beijing Institute of Technology. His research interests include UAV swarm systems, intelligent aircraft modeling, and guidance and control.

![](images/746b5ae77bfa6b676fab8931b199def4fa7ca0430e0f67a7bf5bd4f563d0aea3.jpg)

Ren Jin received the M.E. degree in Computer Application Technology from Hefei University of Technology, Hefei, China, in 2016, and the Ph.D. degree in Aerospace Science and Technology from Beijing Institute of Technology, Beijing, China, in 2020. He is currently a Tenure-Track Assistant Professor with the School of Aerospace Engineering, Beijing Institute of Technology. His research interests include onboard visual object detection, recognition and tracking, and UAV visual navigation.

![](images/01e67a8628476d4af7af352a1801df646fc5bb0b035bb292826ed0bfdbf34f6a.jpg)

Mingdong Jia received the B.E. degree in Information Security from Tianjin University of Technology, Tianjin, China, in 2024. He is currently pursuing the M.S. degree with Beijing Institute of Technology, Beijing, China. His research interests include UAV vision, object detection and multi-object tracking.

![](images/0e85b802d5e63764da34cb727b7a8357c457a7d7a61cfdbce86c1bf986f45ee7.jpg)

Defu Lin received the M.E. and Ph.D. degrees in Aircraft Design from Beijing Institute of Technology, Beijing, China, in 1999 and 2005, respectively. He is currently a Professor with the School of Aerospace Engineering, Beijing Institute of Technology. He directs the Beijing Key Laboratory for UAV Autonomous Control and the China-UAE Belt and Road Joint Laboratory on Intelligent Unmanned Systems. His research interests include aircraft system design, flight vehicle guidance, and control technologies.