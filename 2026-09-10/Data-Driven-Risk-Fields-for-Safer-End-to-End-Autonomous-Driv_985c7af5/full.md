# Data-Driven Risk Fields for Safer End-to-End Autonomous Driving

Yuanxin Tian<sup>1,†</sup>, Zhiyuan Liu<sup>1,†</sup>, Jinhao Li<sup>1</sup>, Zhenhua Xu<sup>1,∗</sup>, Wenhao Yu<sup>1,∗</sup>, and Jianqiang Wang<sup>1</sup>

Abstract— Safety is a fundamental requirement for autonomous driving, yet existing end-to-end driving models still lack explicit risk-aware learning capacities. Existing rule-based risk models provide interpretable safety priors, yet their absolute risk scores depend on handcrafted functions, coefficients, and thresholds. Learning-based risk representations reduce part of this manual design, but their supervision often relies on occupancy-derived labels or heuristic cost values, which may not capture ego-conditioned planning risk. In this paper, we propose DRiF, a data-driven risk-field framework for safer end-to-end autonomous driving. DRiF learns a shared BEV feature with static map segmentation, dynamic risk prediction, and vehicle planning. For dynamic risk learning, DRiF converts rule-based safety priors into pairwise risk labels, and trains the risk field to preserve relative risk ordering instead of regressing handcrafted absolute scores. Experiments on Bench2Drive show that DRiF achieves competitive overall performance, with consistent improvements in driving score, success rate, and collision-related metrics. These results establish relative risk supervision as an effective way to connect explicit safety structure with end-to-end planning. The data and code will be publicly available.

## I. INTRODUCTION

Autonomous driving has progressed rapidly in both academia and industry, with data-driven end-to-end driving becoming a representative paradigm. Given sensory observations and route commands, an end-to-end model directly predicts future trajectories [1], [2], [3], [4]. Early methods established the feasibility of supervised sensor-to-action learning. Subsequent systems strengthen this formulation by introducing structured scene modeling into the learning pipeline. BEV encoders provide a unified top-down space for multi-view perception [5], [6], [7]; online mapping and vectorized scene modeling recover road geometry and topology [8], [9], [10]; motion prediction further extends the representation from static layout to future scene evolution [11], [12], [13], [14]. Finally, planning-oriented architectures connect these intermediate outputs with ego trajectory generation [15], [16], [17]. Recent planners further improve trajectory modeling through generative policies, temporal-consistency modeling, and driving world models [18], [19], [20], [21], [22], [23], [24]. Taken together, these advances make end-to-end driving an increasingly powerful paradigm.

Safety is the central requirement of autonomous driving. A reliable vehicle must not only perceive the scene, but also understand how surrounding objects, road structures, predicted motions, and route constraints affect its own future behavior.

Existing end-to-end planners usually learn this perceptionto-planning relation implicitly through latent features and final trajectory losses [16], [9], [17], [18]. Such supervision can capture common expert driving patterns, but it does not explicitly indicate which BEV regions are dangerous to the current ego vehicle, why they are dangerous, or how their risk should influence planning. Without explicit risk-oriented supervision, an end-to-end planner may fail to learn reliable safety structure, especially in long-tail scenarios such as cutin, merging, intersections, pedestrian crossing, lane-boundary violation, and off-road behavior.

A direct way to introduce safety structure is to model driving risk explicitly. Classical traffic safety metrics assign criticality to interactions from different perspectives. Timeto-collision estimates collision imminence under continued motion [25]; time headway measures longitudinal following safety [26]; post-encroachment time evaluates temporal separation at a conflict point [27]; deceleration rate to avoid collision measures the braking effort required to prevent a crash [28], [29]. Beyond scalar metrics, risk-field methods provide a more general way to represent safety as spatial cost. Artificial potential fields first model obstacles as repulsive forces [30], while probabilistic risk fields further incorporate motion uncertainty and collision probability [31], [32], [33]. Among these methods, driving safety fields offer a particularly relevant formulation for autonomous driving, as they encode driver–vehicle–road interactions in a unified continuous field and provide an interpretable spatial representation for trajectory evaluation and safety-aware planning [34], [35]. Although these formulations are interpretable and auditable, their risk magnitudes usually depend on hand-designed coefficients, thresholds, and scenario-specific calibration. This manual design limits their scalability and makes them difficult to use as direct supervision for end-to-end representation learning.

Learning-based risk representations reduce part of this manual design, but their risk modeling and supervision remain limited. RiskMap learns a differentiable risk field as a driving cost prior for motion planning [36]; Risk Occupancy extends occupancy-style modeling with spatial, temporal, and risk dimensions under vehicle-road-cloud collaboration [37]; RiskMM uses risk maps as middleware for cooperative endto-end planning [38]. These methods bring risk modeling closer to data-driven planning systems. However, driving risk does not have an objective ground truth comparable to depth, occupancy, or object location. Existing models therefore often rely on occupancy-derived labels or heuristic planner scores. Such supervision may simplify risk into occupied-or-costly regions, without explicitly deriving how the surrounding agents jointly determine planning risk. As a

![](images/775a6835efd612c86d7f5df814b07cce3f55dc3d3feab28fe6d6ebdb23b0e3f1.jpg)  
The risk field directly models how the surroundings influence ego planning.

Fig. 1: Explicit risk modeling beyond scene understanding. Conventional scene representations describe objects, lanes, and semantic regions, but do not explicitly reveal how they affect ego planning. Our method learns an ego-conditioned data-driven risk field that models how surrounding agents influence the ego vehicle, enabling safer end-to-end planning.

result, the learned risk map may remain an occupancy-like or cost-map-like surrogate, rather than a genuine ego-conditioned risk representation. This can bind the network to a particular handcrafted risk scale and limit its ability to identify and avoid risks in complex or long-tail driving scenarios.

We address the aforementioned problems by decomposing BEV driving risk into a static map and a dynamic interactive risk field. The static map captures hard constraints from route feasibility and ego reachability, which is supervised with dense labels. The dynamic interactive risk field models planning risk induced by surrounding-agent futures and potential conflicts, where calibrated dense labels are unavailable. We therefore learn it from a relative ranking loss between sampled BEV locations. Based on this design, we propose DRiF, an end-to-end autonomous driving framework relying on Datadriven Risk-Field. DRiF encodes multi-source driving inputs into a shared BEV representation and predicts both static map and dynamic interactive risk. During training, a pairwise risk label generator converts rule-based safety priors into pairwise ranking labels to supervise the dynamic risk head, thereby injecting explicit risk structure into the representation used by the end-to-end planner.

Our contributions are summarized as follows:

• We propose DRiF, an end-to-end autonomous driving framework that learns ego-conditioned risk representations in a data-driven manner for trajectory planning.

• We introduce relative-risk supervision, which learns driving risk through pairwise comparisons instead of hand-crafted absolute scalar labels.

• We conduct extensive experiments on the large closedloop autonomous driving benchmark Bench2Drive. DRiF achieves competitive performance across overall, safety, and multi-ability metrics, demonstrating the effectiveness of explicit risk modeling for safer end-to-end planning.

## II. RELATED WORK

End-to-End Autonomous Driving. End-to-end autonomous driving learns ego actions from onboard observations and expert demonstrations, with outputs ranging from control commands to future trajectories [1], [2], [3], [4], [39], [40]. Recent systems improve this formulation by introducing stronger scene representations. BEV-based methods build a unified spatial feature space for multi-view perception [5], [6], [7]; online mapping and vectorized scene modeling encode road structures for downstream reasoning [8], [9]; motion and occupancy prediction further enable the planner to reason about future scene evolution [11], [12], [13], [14]. Representative planning-oriented frameworks include UniAD, which jointly models perception, prediction, occupancy, and planning [16]; VAD, which uses vectorized scene elements for efficient trajectory generation [9]; and SparseDrive, which formulates end-to-end planning with sparse scene representation [17]. More recent methods strengthen trajectory modeling through generative policies, temporal consistency, and world models [18], [19], [20], [21]. Safety-oriented planners further improve interaction handling through path-conditioned longitudinal planning and safety-critical augmentation [41], ego-centric joint-causal modeling and policy alignment [42], or futurescene priors from 4D occupancy world models [43]. These works improve planning structure, policy alignment, or futurescene modeling, whereas ego-conditioned risk supervision in a shared representation remains under-explored.

Risk Representations for Driving. Classical safety metrics quantify interaction criticality using time-to-collision, time headway, post-encroachment time, deceleration rate, or formal safe-distance constraints [25], [26], [27], [28], [29], [44]. Riskfield methods represent safety as spatial cost through artificial potential fields, probabilistic risk fields, or continuous driver– vehicle–road interaction fields [30], [31], [32], [33], [34], [35]. Recent learning-based approaches establish complementary risk-to-planning interfaces: RiskMap learns a differentiable risk field [36]; Risk Occupancy extends occupancy representations with risk dimensions [37]; RiskMM learns an interpretable spatiotemporal risk representation and feeds it to learning-based MPC [38]. Although these studies demonstrate the value of risk-aware planning, their supervision often relies on manually calibrated risk functions or occupancy-derived labels, requiring models to regress uncertain absolute risk values.

Relative Supervision. Relative supervision is effective when absolute labels are ambiguous or difficult to calibrate. Ranking formulations learn from pairwise preferences [45], [46], and ordinal depth methods show that useful spatial structure can be learned without a globally calibrated scale [47], [48], [49]. These works show that relative constraints can provide robust supervision when absolute values lack a stable scale. DRiF brings this principle to autonomous driving risk representation by learning whether one BEV location is riskier or safer than another, thereby converting rule-based safety knowledge into planner-compatible supervision.

## III. PROBLEM DEFINITION

## A. Ego-Conditioned BEV Risk Field

At time t, an end-to-end driving system receives multisource observations $o _ { t }$ , ego state $s _ { t } .$ , navigation command $c _ { t } ,$ , and navigation point $p _ { t }$ . These inputs are encoded into a bird’s-eye-view (BEV) domain $\Omega \subset \mathbb { R } ^ { 2 }$ , which is discretized into an $H ^ { \prime } \times W ^ { \prime }$ grid $F _ { t }$ . We denote a continuous BEV location by $\boldsymbol { x } = ( x , y )$ and its corresponding grid index by $\boldsymbol { q } = \left( \boldsymbol { u } , \boldsymbol { v } \right)$

The goal of DRiF is to learn an ego-conditioned BEV risk field:

$$
R _ { t } = \Psi ( F _ { t } ) = \Psi ( \Phi _ { \theta } ( o _ { t } , s _ { t } , c _ { t } , p _ { t } ) ) ,
$$

where $R _ { t }$ describes how each BEV location affects the safety of the current ego vehicle. Different from occupancy or object detection, this representation is not an objective scene attribute. Its value depends on ego state, route intention, reachability, road structure, surrounding-agent futures, their interactions, etc.

## B. Risk Field Decomposition

We decompose BEV driving supervision into two complementary outputs:

$$
R _ { t } = \left( M _ { t } , R _ { t } ^ { d y n } \right) ,
$$

where $M _ { t } \in \mathbb { R } ^ { H \times W \times K }$ and $R _ { t } ^ { d y n } \in \mathbb { R } ^ { H \times W }$

The static map representation $M _ { t }$ describes dense BEV map semantics, such as road boundary, lane marking, lane centerline, etc. Since these semantics can be obtained from map annotations or BEV segmentation labels, this branch can be supervised with absolute values. Although $M _ { t }$ is not itself a risk field, it provides structured road-geometry and constraint information that supports ego-conditioned risk learning and downstream planning.

The dynamic interactive risk field $R _ { t } ^ { d y n }$ represents planning risk induced by surrounding-agent futures and potential conflicts. Unlike static map semantics, this risk does not have calibrated dense ground-truth values. For example, a future conflict region may be riskier than a free lane-center region, but the exact numerical risk gap is not objectively defined. Therefore, we define $R _ { t } ^ { d y n }$ through relative ordering rather than absolute values.

## C. Relative-Risk Supervision

For dynamic interactive risk, we construct pairwise supervi sion over planning-relevant BEV locations. Given a sampled pair $( x _ { i } , x _ { j } )$ , its label is $y _ { i j } \in \{ + 1 , 0 , - 1 \}$ , where $y _ { i j } = + 1$ means $x _ { i }$ is riskier than $x _ { j } , y _ { i j } = - 1$ means $x _ { i }$ is safer than $x _ { j }$ , and $y _ { i j } = 0$ means the two locations are comparable. The desired ordering is $R _ { t } ^ { d y n } ( q _ { i } ) > R _ { t } ^ { d y n } ( q _ { j } )$ if $y _ { i j } = + 1$ , where $q _ { i }$ and $q _ { j }$ are the BEV grid indices of $x _ { i }$ and $x _ { j }$

In short, DRiF uses dense supervision for static constraints and relative supervision for ambiguous dynamic risk. This avoids forcing static constraints and dynamic interactions onto the same numerical scale, while still providing explicit ego-conditioned risk supervision for end-to-end planning.

## IV. METHODOLOGY

## A. Overview

DRiF is a risk-centric end-to-end autonomous driving framework. Given multi-source driving inputs, DRiF first extracts a shared BEV feature $F _ { t }$ . The feature is then used by four branches: an auxiliary perception branch, a static map branch, a dynamic risk branch, and a planning branch. The static map branch preserves the model’s understanding of road structure, while the dynamic risk branch learns egoconditioned interaction risk that is difficult to annotate with dense scalar labels. The overall structure of DRiF is visualized in Fig. 2.

DRiF is trained in two stages. Stage 1 uses auxiliary perception tasks, such as detection, segmentation, and depth prediction, to pretrain the encoder with directly observable scene knowledge. Stage 2 removes the auxiliary branch and jointly trains static map segmentation, dynamic risk learning, and trajectory planning. During this stage, a training-only risk pair generator converts rule-based safety priors into pairwise ranking labels for supervising the dynamic risk branch. At inference time, the label generator is removed, the learned risk representation supports planning through shared BEV features.

![](images/75b58302f7a089b7aa453fbf27e0cc5edf3a5d0eed84914b2e6d8b94c1fff325.jpg)  
Fig. 2: Overview of DRiF. DRiF encodes multi-source driving inputs into a shared BEV feature $F _ { t }$ for auxiliary perception, static map segmentation, dynamic risk prediction, and trajectory planning. The model is trained in two stages. Stage 1 uses auxiliary tasks to pretrain the encoder with directly observable scene knowledge (e.g., object states). Stage 2 removes the auxiliary perception branch and jointly trains the static map, dynamic risk, and planning branches. The dynamic risk head is supervised by pairwise risk labels generated from rule-based safety priors, while the static map head maintains the model’s understanding of static surroundings.

## B. Model Structure

BEV Encoder. At time t, the model receives observations $o _ { t } ,$ ego state $s _ { t } .$ , navigation command $c _ { t } ,$ and navigation point $p _ { t }$ . An encoder maps them into a BEV feature $F _ { t } =$ $\Phi _ { \theta } \big ( o _ { t } , s _ { t } , c _ { t } , p _ { t } \big )$ , where $\dot { \boldsymbol { F } _ { t } } \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times C }$ . The encoder can be instantiated with camera, LiDAR, or multi-modal BEV backbones. Ego state and navigation information are injected so that the feature is conditioned on the current driving context.

Auxiliary Perception Branch. In stage 1, the auxiliary branch predicts directly observable scene quantities, including object states, semantic segmentation, and depth. It provides strong pretraining knowledge for geometry and object-level scene understanding, and is removed during stage 2.

Static Map Branch. The static map branch predicts dense BEV map semantics $\begin{array} { r c l } { \hat { M } _ { t } } & { = } & { \Psi _ { \mathrm { m a p } } ( F _ { t } ) } \end{array}$ , where $\begin{array} { r l } { \hat { M } _ { t } } & { { } \in } \end{array}$ $\mathbb { R } ^ { H \times W \times K }$ . It is supervised by BEV map labels, such as drivable area, road boundary, lane marking, and lane centerline. This branch maintains the model’s understanding of static road structures during risk learning.

Dynamic Risk Branch. The dynamic risk branch predicts an ego-conditioned risk field $\hat { R } _ { t } ^ { d \dot { y } n } = \Psi _ { \mathrm { r i s k } } ( F _ { t } )$ , where $\hat { R } _ { t } ^ { d y n } \in$ $\mathbf { \mathbb { R } } ^ { H \times W }$ . Unlike occupancy or semantic maps, $\hat { R } _ { t } ^ { d y n }$ represents relative planning risk caused by surrounding-agent futures and potential conflicts. Since dense calibrated risk labels are unavailable, this branch is learned with relative supervision. Planning Branch. The planning branch predicts the future ego trajectory $\hat { \tau } _ { t } = \Psi _ { \mathrm { p l a n } } ( F _ { t } ) = \{ \hat { x } _ { t + k } \} _ { k = 1 } ^ { T }$ . It follows the original end-to-end driving objective, while benefiting from the risk-aware BEV feature shaped by static map and dynamic risk supervisions.

## C. Relative Supervision

Since calibrated dense labels for dynamic risk are unavailable, DRiF learns the dynamic risk field from pairwise relative supervision. The supervision pipeline contains three steps: sampling candidate points, constructing risk pairs, and generating pairwise risk labels with rule-based safety priors. The generated labels supervise the dynamic risk branch to preserve relative risk ordering.

Candidate Sampling. For each frame, DRiF first samples a set of planning-relevant BEV locations $S _ { t }$ around the ego future trajectory, surrounding-agent futures, potential conflict regions, and route-consistent reachable areas. This step only collects candidate points that may affect ego planning.

Pair Construction. After obtaining the candidate set, DRiF constructs a compact pair set $\mathcal { Q } _ { t } ~ = ~ \{ ( x _ { i } , x _ { j } ) ~ \vert ~ x _ { i } , x _ { j } ~ \in$ $S _ { t } , i \neq j \}$ . In practice, we do not enumerate all pairs. Instead, we sample pairs from different candidate sources and egorelevant regions to avoid redundant easy comparisons and reduce computation.

Pairwise Risk Scoring. For each candidate pair $( x _ { i } , x _ { j } ) \in$ $\mathcal { Q } _ { t } .$ , rule-based safety priors are used to evaluate their relative risk. Each point in the pair is assigned a risk signature $\phi _ { t } ( x ) = [ s _ { 1 } ( x ) , s _ { 2 } ( x ) , s _ { 3 } ( x ) ]$ , where $s _ { 1 }$ measures same-time ego-agent conflict (i.e., overlap risk), $s _ { 2 }$ measures crosstime trajectory-corridor conflict (i.e., corridor risk), and $s _ { 3 }$ measures surrounding-agent future occupancy (i.e., occupancy risk). These scores are not used as dense regression targets, but only as local evidence for pairwise label generation. The labeling process is visualized in Fig. 3.

Pairwise Label Generation. For a pair $( x _ { i } , x _ { j } )$ , DRiF first assigns each point to a risk class C according to the matched risk source. We use three ordered classes, $R _ { 1 } > R _ { 2 } > R _ { 3 }$ where $R _ { 1 }$ denotes overlap risk, $R _ { 2 }$ denotes corridor risk, and $R _ { 3 }$ denotes occupancy risk. For a point $x ,$ the generator checks the risk scores in this priority order: if $s _ { 1 } ( x )$ exceeds its threshold, x is assigned to $R _ { 1 } ;$ otherwise, it checks $s _ { 2 } ( x )$ and then $s _ { 3 } ( x )$ . These thresholds are derived from the rulebased driving risk field and its corresponding safety priors [34], [35]. Given two points, if their assigned classes differ, the pairwise label is determined by the class priority. If they belong to the same class, the label is further determined by comparing their detailed risk scores:

$$
y _ { i j } = \left\{ \begin{array} { l l } { + 1 , } & { C _ { i } > C _ { j } ; C _ { i } = C _ { j } , \ s _ { i } - s _ { j } > \epsilon , } \\ { - 1 , } & { C _ { i } < C _ { j } ; C _ { i } = C _ { j } , \ s _ { j } - s _ { i } > \epsilon , } \\ { 0 , } & { C _ { i } = C _ { j } , \ \lvert s _ { i } - s _ { j } \rvert \le \epsilon . } \end{array} \right.
$$

The resulting pairwise risk label set is denoted as $P _ { t } \ =$ $\{ ( x _ { i } , x _ { j } , y _ { i j } ) \}$ . This two-level design uses reliable class-level safety priors while avoiding direct regression to uncertain absolute risk scores.

## D. Training Losses

DRiF is optimized with planning, auxiliary perception, static map, and dynamic risk losses:

$$
\mathcal { L } = \mathcal { L } _ { p l a n } + \lambda _ { a u x } \mathcal { L } _ { a u x } + \lambda _ { m a p } \mathcal { L } _ { m a p } + \lambda _ { r i s k } \mathcal { L } _ { r i s k } .
$$

Planning and Auxiliary Losses. $\mathcal { L } _ { p l a n }$ follows the original planning objective of the base end-to-end model, such as trajectory, waypoint, target-speed, or control-related losses. $\mathcal { L } _ { a u x }$ is used only in stage 1 and includes auxiliary detection, segmentation, or depth prediction losses for pretraining.

Static Map Loss. The static map branch is supervised by dense BEV semantic labels. We use a standard combination of cross-entropy and Dice losses:

$$
\mathcal { L } _ { m a p } = \mathcal { L } _ { C E } ( \hat { M } _ { t } , M _ { t } ) + \mathcal { L } _ { D i c e } ( \hat { M } _ { t } , M _ { t } ) .
$$

This loss maintains the model’s understanding of static road structures during risk learning.

Pairwise Ranking Loss. For each risk pair $( x _ { i } , x _ { j } , y _ { i j } ) \in P _ { t }$ we map $x _ { i } , x _ { j }$ to BEV grid cells $q _ { i } , q _ { j } .$ , and read their predicted dynamic risk values $\hat { r } _ { i } ~ = ~ \hat { R } _ { t } ^ { d y n } ( q _ { i } )$ and ${ \hat { r } } _ { j } \ =$ $\hat { R } _ { t } ^ { d y n } ( q _ { j } )$ . For ordered pairs with $y _ { i j } \in \{ + 1 , - 1 \}$ , we use the ranking loss [46], [47]:

$$
\mathcal { L } _ { r a n k } = \frac { 1 } { | P _ { t } | } \sum _ { ( x _ { i } , x _ { j } , y _ { i j } ) \in P _ { t } } \operatorname* { m a x } _ { { \bf ( } 0 , \gamma - y _ { i j } ( \hat { r } _ { i } - \hat { r } _ { j } ) \ ) } .
$$

Comparable pairs with $y _ { i j } = 0$ are ignored at this stage. This objective supervises the relative ordering of risk values without regressing handcrafted absolute risk scores.

Stage-Wise Training. Stage 1 pretrains the encoder with direct scene supervision:

$$
\begin{array} { r } { \pmb { \mathcal { L } } ^ { ( 1 ) } = \lambda _ { a u x } \mathcal { L } _ { a u x } . } \end{array}
$$

Stage 2 removes the auxiliary perception branch and jointly optimizes planning, static map understanding, and dynamic risk learning:

$$
\mathcal { L } ^ { ( 2 ) } = \mathcal { L } _ { p l a n } + \lambda _ { m a p } \mathcal { L } _ { m a p } + \lambda _ { r i s k } \mathcal { L } _ { r i s k } .
$$

## V. EXPERIMENTS

## A. Experiment Settings

Training Dataset. We train DRiF on a CARLA dataset collected across multiple towns using the TF++ data collection protocol [50]. Each frame contains a front-view camera image, LiDAR point cloud, ego state, route command, map information, and expert driving trajectory.

We generate DRiF labels from the raw logs. Static map labels are obtained from GT BEV segmentations with dynamic objects (e.g., vehicles, walkers) removed. For dynamic risk field labels, we sample 300 BEV candidates per frame with 50% strategy-guided sampling and 50% random sampling over reachable lanes. Pairs are ranked by staged safety comparators covering overlap risk, corridor risk and occupancy risk.

Closed-Loop Benchmark. We evaluate on Bench2Drive [51], a closed-loop CARLA benchmark with 220 routes across diverse towns, weather, traffic, and scenarios. Given online observations and route commands, the agent should predict future trajectories for closed-loop vehicle control.

Evaluation Criteria. Following the official Bench2Drive protocol [51], we report Driving Score (DS), Success Rate (SR), and Driving Efficiency (Eff.); DS combines route progress and infraction penalties, while Eff. measures the route-averaged ego speed relative to surrounding traffic. We also report CRoute, the percentage of routes with at least one vehicle, pedestrian, or layout collision, and collision events per driven kilometer (Coll./km), followed by Multi-Ability (MA) scores and their mean.

Training Configuration. DRiF is trained on 16 NVIDIA A100 GPUs for 60 epochs in total. The first 30 epochs are used for auxiliary perception pretraining, and the next 30 epochs are used for joint training with static map segmentation, dynamic risk learning, and trajectory planning objectives. We use an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , and the full training process takes about 1.5 days.

## B. Comparison Experiments

Baselines. We compare DRiF with representative end-toend autonomous driving methods, including UniAD [16], VAD [9], ThinkTwice [52], DriveAdapter [53], TransFuser++ (TF++) [54], HiP-AD [55], and SimLingo [56]. Among them, TF++ is a powerful baseline and achieves the best performance.

For a fair comparison, we reproduce most of the baselines from their official open-source implementations and train/evaluate them under the same data split, sensor setting, and closed-loop benchmark protocol. For UniAD and VAD, we retain the officially released benchmark scores and compute the safety statistics from their officially released route-level logs. HiP-AD and SimLingo are not BEV-based methods and mainly operate on front-view observations, we therefore adapt their input interface and training configuration to the dataset setting while preserving their original model design.

![](images/acafaf82e16f6486df8b477c1b1c4112100bf6e687d15508162e6ebfb435dd24.jpg)  
Fig. 3: Pairwise risk label generation. Each point is first assigned to a risk class according to the matched risk type. The comparator then determines the pairwise risk relation by class priority; only when the two points belong to the same risk class, their detailed risk scores are further compared to generate the final label $y _ { i j } \in \{ + 1 , 0 , - 1 \}$

Comparison Results. Our reproduced methods are evaluated on all 220 routes of Bench2Drive in a closed-loop manner; officially released scores are retained for UniAD and VAD. The results are shown in Tab. I. DRiF achieves the best performance among the evaluated non-expert methods, with 88.78 DS and 75.91 SR, outperforming the strongest baseline TF++ multi-frame (4 frames) by 3.13 DS (+3.7%) and 6.82 SR (+9.9%), while improving Eff. from 246.69 to 252.20 and reducing CRoute from 22.73% to 15.91%. DRiF also obtains the highest non-expert scores on most multiability tasks. Together, these gains demonstrate that explicit risk-field supervision improves interaction-aware planning and enables safer closed-loop driving without compromising driving efficiency.

Qualitative Results. As shown in Fig. 4, DRiF assigns higher values to more risky areas, while keeping irrelevant occupied regions relatively low-risk. The planned trajectories tend to avoid high-risk regions and follow safer drivable corridors, indicating that the learned risk field provides interpretable safety structure for end-to-end planning.

## C. Ablation Studies

All ablation studies use 4-frame inputs and start from the same baseline stage-1 checkpoint. The default DRiF setting uses relative risk supervision from all three sources $R _ { 1 } +$ $R _ { 2 } + R _ { 3 }$ , and static map semantics supervision.

Static Map and Risk Supervision. We ablate the two main supervision signals in DRiF. Removing the static map tests the effect of static road-context supervision, while removing the risk branch tests the contribution of interaction-aware relative risk learning. The results are listed in Tab. II. The results show that both branches are beneficial, indicating that static road-context cues and dynamic risk cues provide complementary information for closed-loop driving.

Risk Sources. We ablate the risk sources used for pair ranking. $R _ { 1 } , R _ { 2 } .$ , and $R _ { 3 }$ denote overlap risk, corridor risk, and occupancy risk, respectively. We compare variants using $R _ { 3 }$ (i.e., occupancy), $R _ { 1 } + R _ { 2 }$ , and $R _ { 1 } + R _ { 2 } + R _ { 3 } \ ( \mathrm { i . e . , D R i F } )$ The results in Tab. III show that combining all three sources achieves the best performance, indicating the necessity of all risk classes.

Absolute Value Supervision. We replace pairwise relative supervision with absolute score regression. This examines whether learning risk ordering between BEV locations is more effective than directly fitting handcrafted scalar risk scores. The results are shown in Tab. IV. Relative supervision outperforms absolute-score regression by 3.37 DS, 5.46 SR, and 7.19 MA-Mean, supporting the advantage of learning risk ordering without fitting manually calibrated magnitudes. Auxiliary Tasks. Finally, we add the original auxiliary perception tasks in the 2nd training stage, including semantic, depth, and detection supervision. This tests whether DRiF can perceive and understand the surroundings only based on risk-oriented training objective. Based on the results in Tab. V, we find that the proposed risk-oriented supervision already provides compact and task-relevant scene understanding for driving.

## D. Limitations

Currently, the pairwise label generator only covers several representative and easy-to-validate risk sources (i.e., overlap, corridor and occupancy risks). This simplified design is mainly for quick proof-of-concept validation in CARLA. Thus, the current risk-pair labels do not yet reflect a fully comprehensive risk field for real-world applications. Future work will use more complete risk theories to guide the data-driven risk-field learning, especially on large-scale realvehicle data.

## VI. CONCLUSION

We presented DRiF, a data-driven risk-field framework that learns a shared BEV representation for static-map understanding, dynamic risk prediction, and trajectory planning. Pairwise risk labels supervise relative ordering instead of handcrafted absolute values, retaining interpretable safety structure while enabling data-driven learning. Closed-loop Bench2Drive results show improvements in driving score, success rate, and interaction abilities, supporting explicit riskfield supervision for end-to-end planning.

TABLE I: Closed-loop Bench2Drive results. Eff. follows the official Bench2Drive report [51]; CRoute denotes routes with at least one collision (%), and Coll./km denotes collision events per driven kilometer. Bold and underline mark the best non-expert and baseline results, respectively. Four-frame input is used in multi-frame mode; “†” marks reproduced results, “‡” marks official route logs, and “–” denotes unavailable verifiable route records.
<table><tr><td rowspan="2">Method</td><td colspan="3">Overall</td><td colspan="2">Safety</td><td colspan="6">Multi-Ability</td></tr><tr><td>DS ↑</td><td>SR ↑</td><td>Eff. ↑</td><td>CRoute↓</td><td>Coll./km ↓</td><td>Merge ↑</td><td>Overtake ↑</td><td>EmgBrake ↑</td><td>GiveWay ↑</td><td>TSign ↑</td><td>Mean ↑</td></tr><tr><td>PDM-Lite expert</td><td>97.02</td><td>92.27</td><td>250.04</td><td>3.64</td><td>0.347</td><td>88.75</td><td>93.33</td><td>98.33</td><td>90.00</td><td>93.68</td><td>92.82</td></tr><tr><td>UniAD [16]‡</td><td>45.81</td><td>16.36</td><td>129.21</td><td>49.08</td><td>9.636</td><td>14.10</td><td>17.78</td><td>21.67</td><td>10.00</td><td>14.21</td><td>15.55</td></tr><tr><td>VAD [9]‡</td><td>42.35</td><td>15.00</td><td>157.94</td><td>42.72</td><td>10.674</td><td>8.11</td><td>24.44</td><td>18.64</td><td>20.00</td><td>19.15</td><td>18.07</td></tr><tr><td>ThinkTwice [52]</td><td>62.44</td><td>31.23</td><td>69.33</td><td>一</td><td></td><td>27.38</td><td>18.42</td><td>35.82</td><td>50.00</td><td>54.23</td><td>37.17</td></tr><tr><td>DriveAdapter [53]</td><td>64.22</td><td>33.08</td><td>70.22</td><td></td><td></td><td>28.82</td><td>26.38</td><td>48.76</td><td>50.00</td><td>56.43</td><td>42.08</td></tr><tr><td>HiP-AD [55]†</td><td>81.88</td><td>62.73</td><td>251.33</td><td>26.36</td><td>3.528</td><td>53.75</td><td>60.00</td><td>73.33</td><td>50.00</td><td>75.26</td><td>62.47</td></tr><tr><td>SimLingo [56]†</td><td>84.63</td><td>67.73</td><td>250.88</td><td>23.18</td><td>2.991</td><td>61.25</td><td>53.33</td><td>83.33</td><td>50.00</td><td>84.74</td><td>66.53</td></tr><tr><td>TF++ [54]†</td><td>84.41</td><td>69.09</td><td>240.76</td><td>20.45</td><td>2.714</td><td>66.25</td><td>57.78</td><td>76.67</td><td>50.00</td><td>82.63</td><td>66.67</td></tr><tr><td>TF++ multi-frame†</td><td>85.65</td><td>69.09</td><td>246.69</td><td>22.73</td><td>2.757</td><td>57.50</td><td>64.44</td><td>81.67</td><td>50.00</td><td>85.79</td><td>67.88</td></tr><tr><td>DRiF single-frame</td><td>85.21</td><td>67.27</td><td>243.86</td><td>25.45</td><td>2.861</td><td>58.75</td><td>55.56</td><td>78.33</td><td>50.00</td><td>82.11</td><td>64.95</td></tr><tr><td>DRiF multi-frame</td><td>88.78</td><td>75.91</td><td>252.20</td><td>15.91</td><td>1.955</td><td>70.00</td><td>68.89</td><td>85.00</td><td>50.00</td><td>88.95</td><td>72.57</td></tr></table>

Front View Image  
![](images/f3b695a52bc5cbc84f7b4903a49fd3646232131e5e38ed1d3e3722916e7478a1.jpg)  
Fig. 4: Qualitative visualizations. DRiF highlights ego-relevant risks and provides interpretable planning cues.

TABLE II: Static map and dynamic risk ablations.
<table><tr><td>Setting</td><td>DS</td><td>SR</td><td>MA-Mean</td></tr><tr><td>w/o Static</td><td>83.23</td><td>60.00</td><td>63.97</td></tr><tr><td>w/o Risk</td><td>62.40</td><td>45.45</td><td>35.03</td></tr><tr><td>DRiF</td><td>88.78</td><td>75.91</td><td>72.57</td></tr></table>

TABLE IV: Absolute value supervision ablation.
<table><tr><td>Setting</td><td>DS</td><td>SR</td><td>MA-Mean</td></tr><tr><td>Absolute</td><td>85.41</td><td>70.45</td><td>65.38</td></tr><tr><td>Relative</td><td>88.78</td><td>75.91</td><td>72.57</td></tr></table>

TABLE III: Risk sources ablations.
<table><tr><td>Sources</td><td>DS</td><td>SR</td><td>MA-Mean</td></tr><tr><td>R3 (Occupancy)</td><td>86.29</td><td>70.45</td><td>65.90</td></tr><tr><td> $R _ { 1 } + R _ { 2 }$ </td><td>87.77</td><td>72.73</td><td>69.28</td></tr><tr><td> $R _ { 1 } + R _ { 2 } + R _ { 3 }$ </td><td>88.78</td><td>75.91</td><td>72.57</td></tr></table>

TABLE V: Auxiliary task ablation.
<table><tr><td>Setting</td><td>DS</td><td>SR</td><td>MA-Mean</td></tr><tr><td>w/ Stage-2 Aux</td><td>88.88</td><td>74.55</td><td>70.02</td></tr><tr><td>w/o Stage-2 Aux</td><td>88.78</td><td>75.91</td><td>72.57</td></tr></table>

## REFERENCES

[1] M. Bojarski, et al., “End to end learning for self-driving cars,” arXiv preprint arXiv:1604.07316, 2016.

[2] M. Bansal, A. Krizhevsky, and A. S. Ogale, “ChauffeurNet: Learning to drive by imitating the best and synthesizing the worst,” in RSS, 2019.

[3] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger, “Trans-Fuser: Imitation with transformer-based sensor fusion for autonomous driving,” IEEE TPAMI, 2023.

[4] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li, “End-to-end autonomous driving: Challenges and frontiers,” IEEE TPAMI, 2024.

[5] J. Philion and S. Fidler, “Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d,” in ECCV, 2020.

[6] Z. Li, et al., “BEVFormer: Learning bird’s-eye-view representation from multi-camera images via spatiotemporal transformers,” in ECCV, 2022.

[7] Z. Liu, et al., “BEVFusion: Multi-task multi-sensor fusion with unified bird’s-eye view representation,” in ICRA, 2023.

[8] B. Liao, et al., “MapTR: Structured modeling and learning for online vectorized HD map construction,” in ICLR, 2023.

[9] B. Jiang, et al., “VAD: Vectorized scene representation for efficient autonomous driving,” in ICCV, 2023.

[10] Z. Xu, K.-Y. K. Wong, and H. Zhao, “InsMapper: Exploring innerinstance information for vectorized HD mapping,” in ECCV, 2024.

[11] A. Hu, et al., “FIERY: Future instance prediction in bird’s-eye view from surround monocular cameras,” in ICCV, 2021.

[12] S. Shi, L. Jiang, D. Dai, and B. Schiele, “Motion transformer with global intention localization and local movement refinement,” in NeurIPS, 2022.

[13] Y. Wei, L. Zhao, W. Zheng, Z. Zhu, J. Zhou, and J. Lu, “SurroundOcc: Multi-camera 3d occupancy prediction for autonomous driving,” in ICCV, 2023.

[14] Y. Zhang, Z. Zhu, and D. Du, “OccFormer: Dual-path transformer for vision-based 3d semantic occupancy prediction,” in ICCV, 2023.

[16] Y. Hu, et al., “Planning-oriented autonomous driving,” in CVPR, 2023.

[15] S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao, “ST-P3: End-to-end vision-based autonomous driving via spatial-temporal feature learning,” in ECCV, 2022.

[17] W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng, “SparseDrive: End-to-end autonomous driving via sparse scene representation,” in ICRA, 2025.

[18] B. Liao, et al., “DiffusionDrive: Truncated diffusion model for end-toend autonomous driving,” in CVPR, 2025.

[19] Z. Xing, et al., “GoalFlow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving,” in CVPR, 2025.

[20] Z. Song, et al., “Don’t shake the wheel: Momentum-aware planning in end-to-end autonomous driving,” in CVPR, 2025.

[21] Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang, “End-to-end driving with online trajectory evaluation via BEV world model,” in ICCV, 2025.

[22] Y. Zheng, et al., “World4Drive: End-to-end autonomous driving via intention-aware physical latent world model,” in ICCV, 2025.

[23] B. Zhang, N. Song, J. Li, X. Zhu, J. Deng, and L. Zhang, “Future-aware end-to-end driving: Bidirectional modeling of trajectory planning and scene evolution,” in NeurIPS, 2025.

[24] J. Zhang, Z. Fu, Z. Xu, W. Dai, Q. Liu, and Y. Wang, “ResWorld: Temporal residual world model for end-to-end autonomous driving,” arXiv preprint arXiv:2602.10884, 2026.

[25] J. C. Hayward, “Near-miss determination through use of a scale of danger,” Highway Research Record, 1972.

[26] K. Vogel, “A comparison of headway and time to collision as safety indicators,” Accident Analysis & Prevention, 2003.

[27] P. J. Cooper, “Experience with traffic conflicts in canada with emphasis on “post encroachment time” techniques,” in International Calibration Study of Traffic Conflict Techniques, 1984.

[28] D. F. Cooper and N. Ferguson, “Traffic studies at T-junctions. 2. a conflict simulation record,” Traffic Engineering & Control, 1976.

[29] C. Fu and T. Sayed, “Comparison of threshold determination methods for the deceleration rate to avoid a crash (DRAC)-based crash estimation,” Accident Analysis & Prevention, 2021.

[30] O. Khatib, “Real-time obstacle avoidance for manipulators and mobile robots,” IJRR, 1986.

[31] F. A. Mullakkal-Babu, M. Wang, X. He, B. van Arem, and R. Happee, “Probabilistic field approach for motorway driving risk assessment,” TR-C, 2020.

[32] X. Wang, J. Alonso-Mora, and M. Wang, “Probabilistic risk metric for highway driving leveraging multi-modal trajectory predictions,” IEEE T-ITS, 2022.

[33] Z. Wang, et al., “A data-driven spatio-temporal driving risk field mechanism for path planning,” Expert Systems with Applications, 2026.

[34] J. Wang, J. Wu, and Y. Li, “The driving safety field based on driver– vehicle–road interactions,” IEEE T-ITS, 2015.

[35] J. Wang, J. Wu, X. Zheng, D. Ni, and K. Li, “Driving safety field theory modeling and its application in pre-collision warning system,” TR-C, 2016.

[36] R. Xin, S. Wang, Y. Chen, J. Cheng, M. Liu, and J. Ma, “RiskMap: A unified driving context representation for autonomous motion planning in urban driving environment,” in ROBIO, 2024.

[37] J. Chen, et al., “Risk occupancy: A new and efficient paradigm through vehicle-road-cloud collaboration,” arXiv preprint arXiv:2408.07367, 2024.

[38] M. Lei, Z. Zhou, H. Li, J. Ma, and J. Hu, “Risk map as middleware: Towards interpretable cooperative end-to-end autonomous driving for risk-aware planning,” IEEE RA-L, 2026.

[39] Z. Xu, et al., “DriveGPT4: Interpretable end-to-end autonomous driving via large language model,” IEEE RA-L, 2024.

[40] Z. Xu, et al., “DriveGPT4-V2: Harnessing large language model capabilities for enhanced closed-loop autonomous driving,” in CVPR, 2025.

[41] Y. Wu, et al., “Aligndrive: Aligned lateral-longitudinal planning for end-to-end autonomous driving,” arXiv preprint arXiv:2601.01762, 2026.

[42] S. Moon, M. Lee, J. Seo, J. Kim, and J. Lee, “Causality-aware end-toend autonomous driving via ego-centric joint scene modeling,” arXiv preprint arXiv:2605.13646, 2026.

[43] J. Cheng, R. Song, Y. Wu, N. Zeng, X. Li, and Y. Ai, “Owmdrive: Causality-aware end-to-end autonomous driving via 4d occupancy world model,” arXiv preprint arXiv:2606.30421, 2026.

[44] S. Shalev-Shwartz, S. Shammah, and A. Shashua, “On a formal model of safe and scalable self-driving cars,” arXiv preprint arXiv:1708.06374, 2017.

[45] R. A. Bradley and M. E. Terry, “Rank analysis of incomplete block designs: I. the method of paired comparisons,” Biometrika, 1952.

[46] C. J. C. Burges, et al., “Learning to rank using gradient descent,” in ICML, 2005.

[47] W. Chen, Z. Fu, D. Yang, and J. Deng, “Single-image depth perception in the wild,” in NeurIPS, 2016.

[48] H. Fu, M. Gong, C. Wang, K. Batmanghelich, and D. Tao, “Deep ordinal regression network for monocular depth estimation,” in CVPR, 2018.

[49] L. Yang, et al., “Depth anything v2,” in NeurIPS, 2024.

[50] J. Zimmerlin, J. Beißwenger, B. Jaeger, A. Geiger, and K. Chitta, “Hidden biases of end-to-end driving datasets,” arXiv preprint arXiv:2412.09602, 2024.

[51] X. Jia et al., “Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving,” in NeurIPS, 2024.

[52] X. Jia, et al., “Think twice before driving: Towards scalable decoders for end-to-end autonomous driving,” in CVPR, 2023.

[53] X. Jia, Y. Gao, L. Chen, J. Yan, P. L. Liu, and H. Li, “Driveadapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving,” in ICCV, 2023.

[54] B. Jaeger, K. Chitta, and A. Geiger, “Hidden biases of end-to-end driving models,” in ICCV, 2023.

[55] Y. Tang, Z. Xu, Z. Meng, and E. Cheng, “Hip-ad: Hierarchical and multi-granularity planning with deformable attention for autonomous driving in a single decoder,” in ICCV, 2025.

[56] K. Renz, L. Chen, E. Arani, and O. Sinavski, “Simlingo: Vision-only closed-loop autonomous driving with language-action alignment,” in CVPR, 2025.