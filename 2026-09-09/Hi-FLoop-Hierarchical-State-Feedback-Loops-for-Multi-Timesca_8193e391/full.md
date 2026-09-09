# Hi-FLoop: Hierarchical State-Feedback Loops for Multi-Timescale World Modeling

Rx Fan Z Han<sup>∗</sup>

School of Systems Science, Beijing Normal University

Gyro@mail.bnu.edu.cn zhan@bnu.edu.cn

## Abstract

Multi-agent trafic simulation seeks diverse, coordinated, and physically realistic futures from maps and observed history. Long-horizon closed-loop generation must reconcile multiple decision time scales while its context evolves with generated states. Existing methods often unfold long futures from the initial scene and resolve intent, interaction, and motion monolithically, weakening cross-scale consistency and adaptation. We present HI-FLOOP, a branchconsistent multi-timescale state-feedback framework. Eight scene-level Worlds represent joint hypotheses, and all agents share the selected World identity throughout an 8-second rollout. Within the branch, an 8-second Goal anchors intent, a 2-second Preview coordinates interactions, and 1- second Control produces physical motion. Every 0.5-second commit feeds back only its executed prefix as new facts, while unexecuted hypotheses never enter factual memory. Joint Preview Interaction (JPI) induces a sparse directed future graph from Preview and uses conflict probabilities and signed arrival-time diferences to gate interaction refinement. For generated-state recovery, a prefix-frozen A→B cascade lets frozen Model A generate 0–1 seconds, then transfers typed physical state, admissible context, and the branch index—but no latent state—to an independent Model B for reencoding and 1–2-second recovery. On the full H-D publicvalidation split of 955 scenarios, FDE-selected scenariojoint evaluation, where one World explains all evaluated agents, yields ADE-at-joint-minFDE@8/joint-minFDE@8 of 2.377028/7.529927 m. Agent-centric selection gives ADE-at-minFDE@8/minFDE@8 of 1.203089/3.879052 m and, when independently optimized for ADE, oracleminADE@8 of 1.196636 m.

## 1 Introduction

Open-loop motion prediction infers future motion from a fixed observed history. Closed-loop trafic simulation instead feeds model outputs back into the system, so the prediction problem itself changes with the generated state. Errors in position, velocity, or heading during the first few steps not only afect an individual agent’s subsequent trajectory, but may also alter agent neighborhoods, map relations, conflict locations, and right-of-way order, thereby changing the appropriate responses of other trafic participants. Closed-loop error therefore accumulates over time and propagates through scene interactions. In heterogeneous scenes containing vehicles, pedestrians, and cyclists, a model must additionally reconcile long-term behavioral direction, local interaction coordination, and type-specific motion feasibility at diferent time scales; marginal trajectories that appear plausible in isolation may still conflict when composed. Long-horizon simulation should consequently be assessed not only by single-trajectory errors, but also by joint interaction consistency, safety, map and kinematic compliance, and coverage of the generated distribution [16].

Multi-agent motion predictors now provide strong factual encoding, intent conditioning, and local trajectory refinement. QCNet models scene relations with query-centric representations [45], while MTR and MTR++ organize multimodal prediction and global-to-local trajectory refinement around motion-intention queries [24, 25]. For predictors conditioned on a fixed history, however, naively invoking the model in a rolling fashion does not by itself define state read/write semantics across commits. Meanwhile, trafic simulators such as TraficSim, TraficBots, and BehaviorGPT have introduced generated-state feedback or reactive roll out [28, 42, 46]. Building on these advances, we focus on explicit causal interfaces among long-term joint intent, rolling interaction plans, and physical execution: which joint branch should retain its identity across commits, which local plans may be revised as new facts arrive, how continuous actions are grounded into executable states, and what information is eligible to enter the factual memory. Without these bound aries, short-term avoidance may override long-term direction, per-agent mode choices may break scene-level branch consistency, and unexecuted futures may be conflated with observed facts. Section 2 provides a more detailed comparison of tasks and methods.

We formulate this problem as persistent multi-timescale joint world modeling. Long-term joint intent, mid-term interaction plans, and short-term executable motion are not merely diferent output heads attached to a shared latent state; they are generative states with distinct predictive horizons, temporal persistence, and adaptation scopes. The long-term layer maintains the joint future hypothesis underlying an entire rollout, the mid-term layer continually revises local coordination among agents according to executed facts, and the short-term layer converts the current plan into physical motion. After each short prefix is executed, the model constructs the next generation condition from the updated scene facts. Unexecuted futures always remain planning hypotheses and are never mixed with physical events that have already occurred. Long-horizon closed-loop modeling must therefore jointly address cross-scale decision consistency, temporal continuity of a joint future branch, and dynamic adaptation of interactions to new states.

Under this formulation, HI-FLOOP adopts a QCNetstyle query-centric factual encoder and draws on the intentconditioning and local-refinement paradigms of the MTR family. These mature components form the implementation basis rather than our novelty claim. On top of them, a scenelevel World provides all agents with a shared long-term future condition and preserves its branch identity throughout the rollout, while the Goal may adapt continuously to new facts within that branch. A mid-term Preview instantiates the long-term intent as a local route and passing order. Across two rounds of interaction refinement, the model alternates between reconstructing a sparse relation topology from the updated joint Preview and updating continuous conflict and timing attributes; local Control then translates the interactionconditioned plan into executable physical states. On the learning side, we further map temporal responsibilities for prefix generation and sufix recovery to parameter ownership: a frozen model produces a closed-loop model-generated prefix, while a separately parameterized recovery model is optimized only on the generated-state distribution induced by that prefix. In this way, explicit causal boundaries unify longterm joint hypotheses, rolling interaction adaptation, shortterm physical execution, and generated-state training.

Our main contributions are as follows:

• A multi-timescale decoupled state-feedback closedloop generation framework. We formulate long-term behavioral intent, mid-term interactions, and short-term motion evolution as generative states with heterogeneous predictive horizons, lifetimes, and causal scopes, and close the recurrence through executed-state feedback. This formulation reconciles cross-scale decision consistency with continual adaptation to generated-state evolution within a unified generation process.

• A temporally consistent representation of joint future branches. We elevate multi-agent multimodality to scene-level joint World hypotheses, explicitly modeling the persistence of a discrete branch identity across both agents and rolling time while allowing continuous withinbranch states to evolve with new facts. This decouples joint-branch consistency from the adaptivity of local generation.

• Preview-driven alternating structure–attribute interaction reasoning. Over a rolling Preview, we alternate between reconstructing sparse directed relation topology and updating continuous conflict and timing attributes. Plan changes can thus revise the interaction structure, while interaction reasoning in turn refines the plan, making mid-term coordination an explicit condition for shortterm motion generation.

• Temporally specialized learning on generated-state distributions. We separate generation of a learned prefix from recovery over its generated continuation. A frozen prefix model induces the actual input distribution for an independently parameterized recovery model, while disjoint parameter sets prevent sufix optimization from altering the learned prefix. This casts long-horizon closedloop recovery as a conditional temporal learning problem with explicit ownership of each interval.

On the full H-D public-validation split of 955 scenarios, the step-120k checkpoint from a single complete S1 training run attains a scenario-joint 8-second ADE-at-jointminFDE@8/joint-minFDE@8 of 2.377028/7.529927 m. In the supplementary agent-centric evaluation, FDEbased selection yields ADE-at-minFDE@8/minFDE@8 of 1.203089/3.879052 m, while independent ADE-based selection yields oracle-minADE@8 of 1.196636 m. Section 3 details the 8-/2-/1-second horizons, 0.5-second commits,

World indexing, JPI refinement, and A→B generated-state recovery.

## 2 Related Work

## 2.1 Multi-Agent Motion Prediction and Closed-Loop Trafic Generation

Multi-agent motion prediction has progressed from per-agent marginal trajectories to multimodal joint futures. AgentFormer and Scene Transformer jointly model social– temporal relations or multi-agent dependencies within a scene [17, 40]. M2I and FJMP factor interactive prediction through directed relations, whereas JFP directly learns mutually consistent joint futures for multiple agents [15, 22, 27]. MotionLM and MotionDifuser represent joint uncertainty using discrete motion tokens and difusion distributions, respectively [9, 23]. Query-conditioned predictors are most closely related to our architecture: QCNet encodes scene relations with query-centric representations, MTR/MTR++ use intention queries for global target localization and local motion refinement, and BiFF further fuses multi-level future information [24, 25, 45, 47]. We adopt QCNet’s query centric factual encoding and the intent-conditioning and local-refinement paradigms of the MTR family; query representations, multimodal goals, and hierarchical decoding are not themselves our contributions. These prediction methods primarily address whatfutures are possible given a fixed history, whereas state read/write semantics and continual replanning across multiple execution commits pose a distinct problem.

Data-driven trafic generation additionally requires multiple agents to respond continually on states produced by the model itself. TraficSim and TraficBots learn scene-level stochasticity and reactive driving policies, respectively; BehaviorGPT generates behavior autoregressively in next-patch form, while Trajeglish and SMART cast trafic generation as next-token modeling [20, 28, 36, 42, 46]. Among generative approaches, CTG produces controllable trajectories with guided difusion, SceneDifuser supports constraintaware initialization and closed-loop rollout, and CCDif further introduces causal compositional reasoning [10, 13, 44]. SceneDifuser++ scales the generation of scenes, dynamic agents, and trafic signals to the city level, while ProSim controls closed-loop multi-agent rollouts through numerical, categorical, or textual prompts [30, 31]. Rather than reintroducing hierarchical structure, interaction modeling, or generated state feedback, we define a unified causal contract for these components across commits: the selected World index and its associated query define a persistent joint branch for all agents throughout a rollout, the Goal persists as a slow state across commits, the Preview updates local routes and passing order from new facts, and Control writes only its executed prefix back to the factual memory after dynamics propagation. These boundaries among a joint hypothesis, a revisable plan, and physical facts are orthogonal to whether the underlying model uses queries, tokens, or difusion.

Distinction from Hierarchical Closed-Loop Simulators.. Hierarchical planning and closed-loop feedback both have clear precedents. BITS separates high-level intent inference from low-level driving behavior, while MixSim and HMSim adopt hierarchical trafic-simulation frameworks [14, 29, 38].

Beyond Self-Play conditions low-level continuous motion on high-level multi-agent interaction reasoning and adapts to closed-loop deployment through recovery supervision [41]. These methods demonstrate the benefit of policy–motion hierarchies for closed-loop behavior, but do not specify how a scene-level multimodal joint branch retains its identity across agents and repeated commits, nor do they distinguish the lifecycles of a long-term Goal, a mid-term interaction Preview, short-term Control, and executed facts. This set of crosscommit state semantics is precisely the focus of HI-FLOOP: World identity persists, Goals and interactions adapt to new facts within the branch, and only the prefix executed by the dynamics can be written back to history.

For longer horizons, TraficBots, ProSim, and AutoWorld advance trafic simulation through reactive policies, promptable closed-loop generation, and self-supervised world modeling, respectively [21, 30, 42]. RosettaSim organizes scene topology, agent states, and the introduction of new agents into variable-length structured autoregressive sequences, and evaluates long-horizon simulation with retrieval-based references [37]. Its central concerns are longsequence representation, dynamic agent cardinality, and long-horizon evaluation. In contrast, HI-FLOOP targets persistent causal evolution within a fixed joint World: the same World retains its routing identity across all agents and 16 commits; the long-term Goal, rolling interaction Preview, and short-term Control evolve according to their respective state semantics; and unexecuted plans remain isolated from executed facts. Our distinction is therefore not generic longsequence autoregression, but joint branch consistency across agents and rolling time together with a causal contract for multi-timescale state feedback.

## 2.2 Training and Post-Training for Closed-Loop Trafic Generation

Closed-loop training can be characterized along three separable dimensions: the learning signal, the source of states, and the gradient scope. BITS decomposes high-level intent inference and low-level driving behavior through hierarchical imitation, while SMART supervises multi-agent motion generation with next-token prediction [36, 38]. To address the covariate shift between log-state training and the states encountered in model rollouts, LASIL constructs learner-aware supervision examples from the learner’s state distribution. CAT-K instead selects, from the policy’s top-K action tokens, the token whose resulting next state is closest to the ground truth and continues closed-loop fine-tuning from that state [7, 43]. TraficSim and ProSim perform differentiable closed-loop rollout with backpropagation through time (BPTT) on generated states [28, 30]. TraficBots also rolls out autoregressively, but stops gradients through the action path while retaining cross-time gradients through the state path [42]. Rectify, Don’t Regret detaches the computation graph between adjacent replanning steps to prevent future supervision from creating a non-causal gradient shortcut through induced states [39]. Classical truncated BPTT (TBPTT) reduces the computation and memory costs of long sequences with a finite backward window, at the cost of truncating credit assignment across windows [34].

Rectify focuses on gradient paths when the same predictor is trained against receding target trajectories, while surrounding agents continue to follow log replay. Our S2.1 training extension instead assigns the learned prefix and generated-state recovery to disjointly parameterized Models A and B, and transfers only observable states, admissible exogenous context, and discrete routing labels. Sufix gradients therefore cannot alter the prefix model at the parameter level. The methodological claim lies in jointly isolating temporal responsibility, parameter ownership, and the causal state interface, rather than in detachment or TBPTT itself; the branch-consistent multi-timescale generation framework of HI-FLOOP likewise does not depend on this gradient operation for its definition.

Reinforcement learning and post-training can complement supervised objectives with closed-loop signals for collision avoidance, compliance, behavioral quality, or diversity. Shiroshita et al. select policy sets subject to drivingcompetence constraints and use intrinsic rewards to diversify behavior across policies, evaluating behavioral coverage with trajectory-based metrics [26]. RL fine-tuning and TraficRLHF refine trafic behavior models using closedloop rewards and human preferences, respectively [2, 19]. ForSim introduces step-wise forward simulation into grouprelative policy fine-tuning: at each virtual step, the focal traffic agent propagates candidate modes that are spatiotemporally aligned with a reference trajectory, while other agents repeatedly repredict from the updated state, balancing withinmode continuity, physical feasibility, and interactive responsiveness [3]. The related Plan-R1 aligns ego planning for safety and feasibility using rule-based rewards and VD-GRPO after expert-trajectory pretraining, whereas SMART-R1 applies R1-style reinforcement fine-tuning to multi-agent token-based trafic simulation [18, 32]. These approaches emphasize reward alignment and post-training rollouts; our focus is the joint-branch and multi-timescale state interface in supervised closed-loop generation, making the two directions orthogonal at the level of optimization objectives.

## 3 Method

## 3.1 Overall Architecture and Closed-Loop State

Figure 1 provides an overview of the proposed framework. Given a scene with N vehicles, pedestrians, and cyclists, the input history $\pmb { H } _ { 0 }$ contains 11 frames of agent states sampled at 10 Hz, while the map M contains polylines, boundaries, topology, and available semantics. The objective is to jointly generate 80 future frames per agent over an 8-second horizon. A complete rollout is divided into $C ~ = ~ 1 6$ commits. At each commit, the model generates one second, or 10 control steps, but executes only the first $K _ { e } = 5$ steps (0.5 seconds). This plan-longer-than-commit design preserves a local interaction look-ahead while allowing the model to condition on newly generated states every 0.5 seconds.

Table 1 defines the semantics of the hierarchy. Slow, Medium, and Fast refer to prediction horizons, state persistence, and write-back permissions rather than three independently scheduled inference rates; all three levels are accessed at every 0.5-second commit. Control serves as the action interface from a plan to physical facts, rather than a fourth persistent semantic state.

![](images/51c8f824d63cd76e34fc09a6fbea5261b78c73a0126299c1b218c57260179fc0.jpg)  
Figure 1. Architecture of HI-FLOOP. At initialization, eight scene-level Worlds propose joint future branches and the scorer selects one physical World whose identity persists through all 16 commits. The recurrent core updates an 8-s Goal, alternates two rounds of 2-s Preview–Interaction refinement, decodes 1-s Control, and writes only the executed 0.5-s prefix back to factual history. The S2.1 path uses a frozen A B recovery cascade: its typed boundary transfers physical state and causal context, but no latent state, cache, or gradient graph.

Table 1. Closed-loop read–write contract for the three semantic states. $\mathcal { G } _ { c }$ denotes the future interaction graph at the current commit; the persistent branch state $B _ { c }$ is isolated from the factual history $H _ { c } .$
<table><tr><td>Semantic state</td><td>Carrier and horizon</td><td>Lifecycle and update</td><td>Write-back permission</td></tr><tr><td>Long-horizon branch intent</td><td> $( w , \widetilde { \pmb { g } } _ { c } , \pmb { h } _ { c } ^ { w } ) ;$  8 seconds</td><td>w remains fixed throughout the rollout; the Goal and hidden state receive bounded updates at each</td><td>Written only to the branch state  $\mathbf { \delta } _ { B _ { c } , \mathrm { ~ ~ } }$  never to the factual encoder</td></tr><tr><td>Rolling interaction plan</td><td> $( P _ { c } , \mathcal { G } _ { c } , \overline { { P } } _ { c } , \overline { { r } } _ { c } ) ; 2$  2 seconds</td><td>commit Replanned and warm-shifted every 0.5 seconds</td><td>Stores only the current plan and its branch summary</td></tr><tr><td>Executed physical facts</td><td> $( \pmb { x } _ { c } , \pmb { H } _ { c } ) ; 1 0 \mathrm { H z }$ </td><td>Commits only the first five steps of the one-second Control</td><td>The only state allowed to enter  $H _ { c + 1 }$  and be re-encoded</td></tr></table>

At runtime, these semantic states are maintained in two isolated containers. The factual state $\pmb { H } _ { c }$ contains only physical states that have been observed or executed up to commit $c .$ The persistent branch state

$$
\begin{array} { c } { B _ { c } = ( w ^ { * } , q _ { w ^ { * } } , k _ { w ^ { * } } , \widetilde { g } _ { c - 1 } , { h _ { c } ^ { w ^ { * } } } , } \\ { \overline { { P } } _ { c - 1 } , \overline { { r } } _ { c - 1 } , \chi _ { c } ) , } \\ { k _ { w } = ( k _ { w , i } ) _ { i = 1 } ^ { N } } \end{array}\tag{1}
$$

stores the fixed World identity $w ^ { * }$ , its World Query $\mathbf { \Delta } \mathbf { q } _ { w } \mathbf { * }$ and per-agent Anchor indices $k _ { w ^ { \ast } }$ , together with the slow Goal Region, the World hidden state, the previous Preview warm start, an interaction summary, and the committed quality prefix $x _ { c }$ . Although $B _ { c }$ persists across commits, it is never treated as an observation by the factual encoder. Let E, G, P, C, and Φ denote factual encoding, slow-goal updating, Preview-based interaction planning, control decoding, and joint dynamics, respectively. One commit then follows

$$
\begin{array} { c } { F _ { c } = E ( \pmb { H } _ { c } , \pmb { M } ) , } \\ { \widetilde { g } _ { c } = G ( \pmb { F } _ { c } , \pmb { B } _ { c } ) , } \\ { ( \pmb { P } _ { c } , \mathcal { G } _ { c } , \pmb { h } _ { c + 1 } ^ { w } ) = P ( \pmb { F } _ { c } , \widetilde { \pmb { g } } _ { c } , \pmb { B } _ { c } ) , } \end{array}
$$

$$
\begin{array} { r l } & { \quad U _ { c } \sim C ( { \boldsymbol { F } } _ { c } , \widetilde { { \boldsymbol { g } } } _ { c } , P _ { c } , \boldsymbol { h } _ { c + 1 } ^ { w } ) , } \\ & { \quad { \boldsymbol { X } } _ { c } ^ { 1 : 1 0 } = \Phi ( \boldsymbol { x } _ { c } , { \boldsymbol { U } } _ { c } ) , } \\ & { \quad { \boldsymbol { H } } _ { c + 1 } = \mathrm { R o l l } ( { \boldsymbol { H } } _ { c } , { \boldsymbol { X } } _ { c } ^ { 1 : 5 } ) . } \end{array}\tag{2}
$$

Consequently, P and the unexecuted states $X _ { c } ^ { 6 : 1 0 }$ remain planning hypotheses; only $X _ { c } ^ { 1 : 5 }$ may enter $H _ { c + 1 }$ . State feedback neither reselects the World nor changes the Goal Anchor. Instead, it adapts the continuous evolution within a persistent branch identity according to newly established facts. This distinction gives the hierarchy its causal meaning, beyond merely stacking modules with diferent prediction horizons.

Agent sets serving diferent purposes are also kept explicit. The existence mask determines which agents participate in physical updates, the output mask determines which agents require generation, the supervision mask selects entries with a valid ground-truth prefix, and the oficial mask is used only for final evaluation. Context agents still participate in factual encoding, interaction graphs, and safety computation. A trajectory that terminates early in future ground truth merely shortens the supervised prefix; it is neither padded as a stationary trajectory nor interpreted as a motion mode. During training, the model constructs eight lightweight World plans but recursively executes only one physical World per scene. For the oficial H-D export, each w is combined with four control-noise streams that remain temporally correlated and persistent across commits, yielding 32 independently executed, scene-level joint rollouts. Within each rollout, all agents share the same w and sample stream while maintaining separate dynamic histories; per-agent, per-commit, or post-hoc stitching across Worlds is prohibited. The four samples within a World represent within-branch execution stochasticity rather than four distinct semantic Worlds.

## 3.2 Factual Scene Encoder with Causal Inputs Only

The factual encoder adopts a QCNet-style factorized relation model. It first applies Temporal Self-Attention to the 11-frame history of each agent, retaining both sequence memory and a history summary. A polyline encoder then represents lanes, road boundaries, drivable regions, and map topology, followed by dynamic Agent–Map Attention and directed Agent–Agent relation attention. Relation edges are computed in the querycentric coordinate frame of the receiving agent and augmented with directed type-pair embeddings to distinguish, for example, vehicle→pedestrian, pedestrian→vehicle, and vehicle→vehicle interactions. Rather than producing a single scene vector, the encoder returns the multi-source factual memory

$$
\mathcal { F } _ { c } = \{ H _ { c } ^ { \mathrm { h i s t } } , H ^ { \mathrm { m a p } } , H _ { c } ^ { a 2 m } , H _ { c } ^ { a 2 a } , H _ { c } ^ { \mathrm { s i g } } \} .\tag{3}
$$

For the ℓ-th layer, the map and agent relation updates can be summarized as

$$
\begin{array} { r l } & { \pmb { h } _ { i , c } ^ { a 2 m , \ell } = \mathrm { A t t n } _ { a 2 m } ( \pmb { H } _ { c } ^ { \mathrm { m a p } } , \pmb { h } _ { i , c } ^ { a 2 a , \ell - 1 } , \{ e _ { p  i , c } ^ { m } \} ) , } \\ & { \pmb { h } _ { i , c } ^ { a 2 a , \ell } = \mathrm { A t t n } _ { a 2 a } ( \{ \pmb { h } _ { j , c } ^ { a 2 m , \ell } \} , \pmb { h } _ { i , c } ^ { a 2 m , \ell } , \{ e _ { j  i , c } ^ { a } , t _ { j  i } \} ) , } \\ & { \quad \pmb { f } _ { i , c } = \phi _ { f } [ \pmb { h } _ { i , c } ^ { \mathrm { h i s t } } , \pmb { h } _ { i , c } ^ { a 2 m } , \pmb { h } _ { i , c } ^ { a 2 a } ] , } \end{array}\tag{4}
$$

where $t _ { j  i }$ encodes the directed agent-type pair. Keeping the memory sources separate allows the downstream Goal, Preview, and Control modules to access history, map, agent, and signal facts as needed, without relying on a single compressed scene vector.

Static map representations may be cached within a continuous rollout. After each commit, the model appends five generated states to the rolling history, retains the most recent 11 frames, and recomputes dynamic relations. During TBPTT training, static representations may be recomputed across chunks to instantiate a new computation graph, but their numerical semantics do not change with the generated future. Diferent Worlds may share only the static map and the initial read-only observations; they cannot share dynamic memory after their trajectories diverge.

Causal contract for observed elevation.. Elevation z is used only as an optional relative quantity in factual relations, not as a target for full 3D dynamics. The model encodes asinh $( \Delta z / 1 \mathrm { m } )$ and a separate availability bit only when both endpoints of a relation are marked as observed by a trusted data converter. If elevation is missing at either endpoint, both the elevation value slot and the availability bit are set to zero, while the 2D relative geometry is retained. This distinguishes a genuine $\Delta z = 0$ from missing elevation. Because closedloop states generated by the model do not predict future z, their agent elevation is marked unobserved; neither future ground-truth elevation nor map-projected elevation may be reinjected into the factual encoder. Genuine observed relative elevation in the static map can nevertheless be preserved.

3.3 Goal Regions and World-Conditioned Coordination Type-specific Goal proposals.. For each agent, the model generates $K = 3 2$ candidate Goal Regions over the 8-second horizon. Vehicles, pedestrians, and cyclists use separate typespecific prior banks. A pedestrian query is oriented using its most recent reliable direction of motion, falling back to the current heading at low speed. Define the element-wise smooth bound $\begin{array} { r } { \mathcal { T } _ { L } ( z ) = L \operatorname { t a n h } ( z / L ) } \end{array}$ . The center and scale of candidate k for agent i are

$$
\begin{array} { r l } & { \pmb { g } _ { i , k } ^ { q } = \pmb { a } _ { i , k } + \mathcal { T } _ { L _ { i } } ( \pmb { r } _ { i , k } ) , \qquad \pmb { g } _ { i , k } ^ { s } = \mathcal { R } _ { i } \pmb { g } _ { i , k } ^ { q } + \pmb { p } _ { i } , } \\ & { \pmb { s } _ { i , k } = \mathrm { c l i p } \big ( \pmb { s } _ { i , k } ^ { 0 } \odot \mathrm { e x p } [ 0 . 7 5 \mathrm { t a n h } ( \Delta \pmb { s } _ { i , k } ) ] , \pmb { s } _ { \mathrm { m i n } } , \pmb { s } _ { \mathrm { m a x } } \big ) } \end{array}\tag{5}
$$

where $q / s$ denote query/scene coordinates, ${ \mathbf { } } a _ { i , k }$ and $\pmb { s } _ { i , k } ^ { 0 }$ are type-specific priors, and $\mathcal { R } _ { i }$ is defined by the current factual heading. Each proposal additionally carries map context, a coarse ETA interval, a unary score $u _ { i , k }$ , and a learned feasibility logit $f _ { i , k }$ . Its canonical score before World coordination is

$$
s _ { i , k } = u _ { i , k } + \alpha _ { \mathrm { f e a s } } \sigma ( f _ { i , k } ) .\tag{6}
$$

This proposal stage reads only factual memory and therefore cannot exploit interaction outcomes that have not yet been generated.

World Queries.. Eight learnable World Queries serve as scene-level seeds for future hypotheses. For scene b, the conditioning vector of World w is

$$
\pmb { q } _ { b , w } = \mathrm { L N } \Bigg ( e _ { w } + \phi _ { s } \Bigg [ \frac { 1 } { N _ { b } } \sum _ { i \in b } \pmb { h } _ { i } \Bigg ] \Bigg ) ,\tag{7}
$$

where $\boldsymbol { h } _ { i }$ is the factual agent representation. Each World first adds a bounded conditional residual to the proposal scores and then performs two layers of sparse soft coordination over the factual agent graph. For World w and agent i, the candidate distribution and its soft summary are

$$
\begin{array} { r l } & { p _ { w , i , k } = \mathrm { s o f t m a x } _ { k } ( \ell _ { w , i , k } ) , \qquad \bar { \pmb { e } } _ { w , i } = \displaystyle \sum _ { k = 1 } ^ { K } p _ { w , i , k } \pmb { e } _ { i , k } , } \\ & { \ell _ { w , i , k } ^ { 0 } = s _ { i , k } + \Delta _ { w , i , k } ^ { 0 } . } \end{array}\tag{8}
$$

For each factual edge $j ~  ~ i$ , the message jointly reads $( h _ { j } ^ { w } , h _ { i } ^ { w } , \bar { e } _ { w , j } , \bar { e } _ { w , i } )$ , relative geometry, and the directed type pair, and is aggregated at the receiving agent through a learned gate. Each layer further produces a centered, bounded logit residual over valid proposals:

$$
\ell _ { w , i , k } ^ { \prime } = \ell _ { w , i , k } + \Delta _ { \mathrm { m a x } } \operatorname { t a n h } \Biggl ( \delta _ { w , i , k } - \frac { 1 } { | \mathcal { K } _ { i } | } \sum _ { k ^ { \prime } \in \mathcal { K } _ { i } } \delta _ { w , i , k ^ { \prime } } \Biggr ) .\tag{9}
$$

This enables the Goal distribution of one agent to influence those of its neighbors within the same World without enumerating $3 2 ^ { N }$ combinations. It is a tractable sparse-coupling approximation, not an exactly normalized global distribution. In this paper, a “joint World” specifically denotes a branch identity in which all agents share scene-level conditioning and are coupled through message passing. We do not claim to explicitly represent or normalize the full $3 2 ^ { N }$ joint distribution, and we do not permit per-agent argmax selections to be stitched across Worlds.

After coordination, each World makes a single hard selection for every agent:

$$
k _ { w , i } = \arg \operatorname* { m a x } _ { k } \ell _ { w , i , k } .\tag{10}
$$

The Anchor class and its topological basin remain fixed throughout a standard 8-second rollout, preventing mode switches caused by commit-wise reselection. To permit continuous adaptation of the target as execution proceeds, we define the radial bound

$$
\mathcal { B } _ { R } ( z ) = \left\{ \begin{array} { l l } { R \frac { \operatorname { t a n h } ( \| z \| / R ) } { \| z \| } z , } & { \| z \| > 0 , } \\ { \mathbf { 0 } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{11}
$$

and, for $c > 0 .$ , update the center ofset as

$$
\begin{array} { r l } & { \pmb { d } _ { i , c } = \mathcal { R } _ { i , c } \mathcal { B } _ { R _ { i } ^ { \mathrm { s t e p } } } ( \Delta \pmb { o } _ { i , c } ^ { q } ) , } \\ & { \pmb { o } _ { i , c } = \mathcal { B } _ { R _ { i } ^ { \mathrm { t o u l } } } \big ( \pmb { o } _ { i , c - 1 } + \pmb { d } _ { i , c } \big ) , \qquad \widetilde { \pmb { g } } _ { i , c } = \pmb { g } _ { i , k _ { w ^ { * } , i } } ^ { s } + \pmb { o } _ { i , c } . } \end{array}\tag{12}
$$

The update reads the current facts, the persistent World condition, the remaining horizon, and the Preview endpoint and interaction summary from the previous commit. For any bounded scalar or vector x $\in ( a , b )$ ), define

$$
\begin{array} { l } { { \mathrm { U p d } _ { [ a , b ] } ( { \pmb x } , \delta ; \eta ) = a + ( b - a ) \sigma \displaystyle \biggl [ \log \mathrm { i t } \biggl ( \frac { { \pmb x } - a } { b - a } \biggr ) } } \\ { { + \eta \operatorname { t a n h } ( \delta ) ] , } } \end{array}\tag{13}
$$

which is used to update the Region scale and confidence separately. After several consecutive failed commits, a state is only marked as degraded or invalid; the discrete Anchor is not reselected. The Goal termination time is always fixed at the initial $t _ { 0 } + 8$ seconds, preventing recurrent planning from continually pushing the long-horizon target into the future.

## 3.4 Preview-Induced Sparse Future Interaction

For each World and each agent, the model generates four Preview nodes at +0.5, +1.0, +1.5, and +2.0 seconds. We refer to the two alternating rounds of Preview graph construction, interaction message passing, and plan refinement as the Joint Preview Interaction (JPI) block. Specifically, $P _ { c } ^ { 0 }$ induces $\mathcal { E } _ { c } ^ { 0 }$ , and $\mathrm { J P I } _ { 1 }$ produces $P _ { c } ^ { 1 }$ ; the model then reconstructs $\mathcal { E } _ { c } ^ { 1 }$ from $P _ { c } ^ { 1 }$ , and $\mathrm { J P I _ { 2 } }$ yields the final $P _ { c } ^ { 2 }$ . JPI is an operator that updates the rolling plan, not a fourth persistent state, and it never writes directly to the factual history. Let $R _ { i , c }$ be the remaining time to the Goal, $\pmb { v } _ { i , c } ^ { q }$ the current query-frame velocity, and $d _ { i , c } ^ { q }$ the query-frame displacement from the current position to the center of the Slow Goal. For Preview time $t _ { m }$ , we first construct a motion baseline with the correct terminal boundary condition:

$$
\begin{array} { r l r } & { \tau _ { i , m } = \mathrm { m i n } ( t _ { m } , R _ { i , c } ) , \quad } & { s _ { i , m } = \tau _ { i , m } / R _ { i , c } , } \\ & { h ( s ) = s ^ { 2 } ( 3 - 2 s ) , } \\ & { P _ { i , m } ^ { \mathrm { b a s e , } q } = { v _ { i , c } ^ { q } } \tau _ { i , m } + h ( s _ { i , m } ) \left( d _ { i , c } ^ { q } - { v _ { i , c } ^ { q } } R _ { i , c } \right) . } \end{array}\tag{14}
$$

The initial nodes and the refinement at round r are respectively

$$
\begin{array} { r l } & { P _ { i , c } ^ { q , 0 } = P _ { i , c } ^ { \mathrm { b a s e } , q } + \mathcal { T } _ { L _ { P } } ( r _ { i , c } ^ { 0 } ) , } \\ & { P _ { i , c } ^ { q , r + 1 } = P _ { i , c } ^ { q , r } + \mathcal { T } _ { 0 . 3 5 L _ { P } } ( r _ { i , c } ^ { r + 1 } ) . } \end{array}\tag{15}
$$

Velocities are obtained by diferencing adjacent 0.5-second nodes and are represented jointly with heading, progress, occupancy scale, and the nearest map topology. After 0.5 seconds are executed, the old $+ 1 . 0 , + 1 . 5 .$ , and +2.0 second nodes are shifted to become the first three warm-start nodes of the next commit, while a new +2.0 second node is predicted. The warm start is used only as a conditioning feature and is never copied into the physical state.

Sparse future graph.. For each directed agent pair $j  i ,$ candidates are formed using the current envelope clearance $c _ { j i } ^ { 0 } .$ , the minimum Preview-path clearance $c _ { j i } ^ { f } ,$ closing speed, TTC, whether the current nearest polygons of the two agents coincide, and the geometric ETA diference $\Delta \eta _ { j i } ^ { \mathrm { g e o } }$ . The ranking score is

$$
\begin{array} { r l } & { \rho _ { j i } = \mathrm { ~ - ~ } \operatorname* { m i n } ( c _ { j i } ^ { 0 } , c _ { j i } ^ { f } ) - 0 . 2 5 | \Delta \eta _ { j i } ^ { \mathrm { g e o } } | } \\ & { \mathrm { ~ ~ } + \frac { 2 } { \operatorname* { m i n } ( { \mathrm { T T C } _ { j i } , 2 0 } ) + 1 } + 0 . 2 5 \mathbb { I } _ { \mathrm { s a m e - m a p } } . } \end{array}\tag{16}
$$

An edge is eligible whenever the current clearance lies within the local radius or the future clearance enters the warning range. Each receiving agent retains its highest-ranked neigh bors, while edges whose current or future envelopes overlap are exempt from the neighbor cap. At round $r \in \{ 0 , 1 \}$ , the discrete edge set $\mathcal { E } _ { c } ^ { r }$ is constructed from the current Preview $P _ { c } ^ { r }$ , and continuous edge attributes are recomputed on that graph. Thus, the Preview modified by the first refinement can afect both the candidate topology and interaction strength in the second round. Discrete edge selection itself is not diferentiated, whereas continuous geometry and message updates remain diferentiable.

At round $r \in \{ 0 , 1 \}$ , edge attributes $\pmb { a } _ { j i } ^ { r }$ on $\mathcal { E } _ { c } ^ { r }$ are used to predict the conflict logit $\boldsymbol { \kappa } _ { j i } ^ { r }$ , signed arrival-time diference, and its scale:

$$
\begin{array} { r l r } & { \widehat { \Delta \eta } _ { j i } ^ { r } = \Delta \eta _ { j i } ^ { \mathrm { g e o } , r } + 1 . 5 \operatorname { t a n h } ( \delta _ { j i } ^ { r } ) , } & \\ & { \gamma _ { j i } ^ { r } = \sigma ( \kappa _ { j i } ^ { r } ) , \qquad b _ { j i } ^ { r } = \exp \bigl ( \exp ( \beta _ { j i } ^ { r } , - 4 , 2 ) \bigr ) , } & \\ & { p ^ { r } ( j \prec i ) = \sigma \Bigl ( - \widehat { \Delta \eta } _ { j i } ^ { r } / b _ { j i } ^ { r } \Bigr ) . } & { \qquad ( \mathrm { l i p } ^ { r } ) } \end{array}\tag{7}
$$

The conflict probability $\gamma _ { j i } ^ { r }$ also gates the sparse message from j to i. Aggregated messages update the agent-level World hidden state and refine the four Preview nodes:

$$
( { P } _ { c } ^ { r + 1 } , { \pmb { h } } _ { c } ^ { w , r + 1 } ) = \mathbf { J } \mathbf { P } \mathrm { I } _ { r + 1 } ( { P } _ { c } ^ { r } , { \pmb { h } } _ { c } ^ { w , r } , { \pmb { F } } _ { c } , \widetilde { { \pmb { g } } } _ { c } ; { \mathcal { E } } _ { c } ^ { r } , { \pmb { a } } _ { c } ^ { r } , \gamma _ { c } ^ { r } ) .\tag{18}
$$

Precedence is derived monotonically from signed ∆ETA rather than predicted by an unconstrained priority head that could contradict the timing diference. Importantly, the two JPI rounds and their intermediate graph reconstruction update only the rolling Preview, interaction structure, and World hidden state; they do not alter the fixed Goal Anchor. The Slow Goal is updated only at the beginning of the next commit, where it uses the interaction summary left by the current commit in Eq. (12). Ground-truth positive conflict edges added during training are used solely to compute interaction losses and never enter inference-time message passing, thereby preventing future-topology leakage.

## 3.5 Causal World Plan Scoring

At $\begin{array} { r l r } { c } & { { } = } & { 0 } \end{array}$ , after all eight Worlds have completed lightweight Goal and Preview planning, the WorldPlanScorer predicts a six-dimensional quality vector $\widehat { \mathbf { q } } _ { b , w }$ and an aggregate score ${ \mathbf { } } ^ { S } b , w$ bfor every scene–World pair. The six components correspond to Goal, Preview, Interaction, Map, Dynamics, and Closed-loop quality. Let $z _ { b , w }$ aggregate the Preview and World representations, Goal geometry, interaction timing, and committed quality prefix of that World. Then

$$
\widehat { \pmb q } _ { b , w } = \psi _ { q } ( \mathrm { s g } [ z _ { b , w } ] ) , \qquad s _ { b , w } = \psi _ { s } ( \mathrm { s g } [ z _ { b , w } ] , \widehat { \pmb q } _ { b , w } ) ,\tag{19}
$$

where sg stops planning gradients from the Scorer. Map, safety, dynamics, and trajectory losses still supervise the generator directly, preventing the planner from altering its plan features merely to please the internal Scorer. At inference time, the model selects $w ^ { * } = \arg \operatorname* { m a x } _ { w } s _ { b , w }$ and retains that World identity for all 16 commits of the same scene. Subsequent causal scores evaluate only the selected branch and never stitch together agent-wise Worlds.

Supervised training likewise executes only one physical World per scene. Scene-level WTA routing, the causal Scorer winner, and balanced exploration are mixed to select the branch. For a given scene, only the routed World receives full physical-execution supervision; routing coverage over training gives diferent Worlds opportunities to receive such labels. WTA selects a branch but never injects a ground-truth Anchor into the forward Goal process. Unexecuted Worlds receive only the same lightweight plan-level supervision, and a two-second proxy cannot stand in for realized eight-second closed-loop quality. Scorer supervision combines the initial lightweight quality over eight Worlds, the per-commit causal quality of the selected branch, and its accumulated realized quality:

$$
\mathcal { L } _ { \mathrm { s c o r e } } = \beta _ { \mathrm { l i g h t } } \mathcal { L } _ { \mathrm { l i g h t } } ^ { 1 : W } + \mathcal { L } _ { \mathrm { c a u s a l } } ^ { w ^ { * } } + \beta _ { \mathrm { r e a l } } \mathcal { L } _ { \mathrm { r e a l i z e d } } ^ { w ^ { * } } .\tag{20}
$$

The final term uses stop-gradient initial plans and accumulated realized quality to calibrate whether the initial plan predicts long-horizon execution quality. Since only $w ^ { * }$ receives a complete label, this term must not be interpreted as genuine full-ranking supervision over all eight Worlds.

## 3.6 Continuous Control and Diferentiable Dynamics

The control head reads the final Preview, Slow Goal, World state, factual memory, and current physical state, and predicts a continuous distribution over the next second. For each agent, the latent variable of the flattened 30-dimensional control sequence is

$$
\begin{array} { r } { z = \mu + \exp ( \log \sigma ) \odot \epsilon + L \xi , \qquad } \\ { \epsilon \sim \mathcal { N } ( 0 , I ) , \qquad \xi \sim \mathcal { N } ( 0 , I _ { 2 } ) , } \end{array}\tag{21}
$$

where L is a rank-2 temporal covariance factor. Stochastic export further uses temporally correlated noise,

$$
\epsilon _ { t } = \rho \epsilon _ { t - 1 } + \sqrt { 1 - \rho ^ { 2 } } \eta _ { t } , \qquad \eta _ { t } \sim \mathcal { N } ( \mathbf { 0 } , I ) ,\tag{22}
$$

rather than independent white noise at every step. At each 0.1- second step, the 2D query-frame acceleration is mapped onto a disk, while the yaw acceleration is mapped to an interval:

$$
\begin{array} { r } { { \pmb a } _ { t } ^ { q } = { A _ { i } } \frac { \operatorname { t a n h } \left( \lVert { \boldsymbol z } _ { t } ^ { x y } \rVert \right) } { \lVert { \boldsymbol z } _ { t } ^ { x y } \rVert } { \boldsymbol z } _ { t } ^ { x y } , \qquad \alpha _ { t } = \Omega _ { i } \operatorname { t a n h } ( { \boldsymbol z } _ { t } ^ { \omega } ) . } \end{array}\tag{23}
$$

The continuous limit is used as $\| z _ { t } ^ { x y } \| \to 0$ . Vehicles and cyclists retain all three channels, whereas the yaw-control channel is masked for pedestrians, which use only 2D holonomic acceleration. The invertible transformations and their Jacobians are included in the control NLL so that bounded physical controls are not incorrectly modeled as unconstrained Gaussian variables.

After rotating query-frame controls into the scene frame, all agents are integrated synchronously in FP32:

$$
\begin{array} { r l } & { \pmb { v } _ { t + 1 } = \pmb { v } _ { t } + \pmb { a } _ { t } ^ { s } \Delta t , } \\ & { p _ { t + 1 } = \pmb { p } _ { t } + \frac { 1 } { 2 } ( \pmb { v } _ { t } + \pmb { v } _ { t + 1 } ) \Delta t , } \\ & { \omega _ { t + 1 } = \mathrm { c l i p } ( \omega _ { t } + \alpha _ { t } \Delta t ) , } \\ & { \theta _ { t + 1 } = \mathrm { w r a p } \big ( \theta _ { t } + \frac { 1 } { 2 } ( \omega _ { t } + \omega _ { t + 1 } ) \Delta t \big ) . } \end{array}\tag{24}
$$

(25)

Vehicles and cyclists explicitly maintain yaw and yaw rate. A pedestrian’s heading is determined by its direction of motion when its speed is suficiently high and otherwise retains the previous heading. Extreme clipping in the dynamics is used only as an emergency numerical safeguard and is not responsible for learned collision avoidance. Preview nodes and local ground-truth states supervise the Preview, while the trajectory obtained by integrating the mean controls is supervised by continuous ground-truth states. Because Control is conditioned on the final Preview representation, state, safety, and map losses can backpropagate to the Preview through the control path; however, the model does not introduce an additional direct alignment objective between the four twosecond Preview nodes and the one-second control trajectory. At inference time, Control does not overwrite Preview coordinates, and physical states are produced only by Eq. (25).

## 3.7 Learning Objectives and Streaming Generated-State Training

The learning objectives are grouped by the interfaces they supervise: Goal proposal and Region; World assignment and diversity; Slow Goal region and continuity; Preview state, continuity, and topology; interaction conflict, timing, and reciprocity; World Scorer; control distribution; integrated state; kinematic priors; safety; map feasibility; and closed-loop error. To prevent the number of vehicles from overwhelming rare agent types, agent-level terms are first averaged over time and agents and then macro-averaged across agent types present in the current batch. Interaction terms are macro-averaged across supervised directed type pairs, whereas global safety terms retain the within-scene pair average. In general, let $\mathcal { G } _ { m } ^ { + }$ be the set of reduction groups containing valid elements for loss m. We compute

$$
\begin{array} { r l } & { \overline { { \ell } } _ { m , g } = \frac { { \sum _ { j \in \mathcal { V } _ { m , g } } \ell _ { m , j } } } { { \operatorname* { m a x } ( 1 , | \mathcal { V } _ { m , g } | ) } } , } \\ &  ~ { \overline { { \mathcal { L } } } _ { m } = \frac { { 1 } } { { \operatorname* { m a x } ( 1 , | \mathcal { G } _ { m } ^ { + } | ) } } \displaystyle \sum _ { g \in \mathcal { G } _ { m } ^ { + } } \overline { { \ell } } _ { m , g } , } \end{array}
$$

$$
\mathcal { L } = \sum _ { m } \lambda _ { m } ( c ) \frac { \overline { { \mathcal { L } } } _ { m } } { \operatorname* { m a x } \{ 1 , \mathrm { E M A } ( | \overline { { \mathcal { L } } } _ { m } | ) \} } .\tag{26}
$$

Terms with empty support produce no gradient. Future interaction geometry, dynamics, key probability normalizations, and loss denominators are evaluated in FP32. The nonamplifying EMA balances scale diferences without magnifying auxiliary terms that are intrinsically small. Kinematic terms supervise ground-truth acceleration and regularize jerk, sideslip, and lateral acceleration; safety and map terms operate on scene-level joint trajectories after dynamics integration. Missing future ground truth, missing elevation, and invalid interaction timing are controlled by their respective masks.

Unlike training exclusively on logged states, the training rollout writes the first five states generated by Eq. (2) into the history of the next commit. Our evaluation uses the step-120k Stage 1 checkpoint learned from the first two commits starting from a real 11-frame history. Streaming training partitions the C commits into finite TBPTT windows. Numerical state evolves continuously from model-generated commits, while sg at a window boundary truncates only the computation graph; it neither reinjects ground truth nor resets the physical state. TBPTT limits memory usage and the temporal extent of gradient credit assignment, while preserving continuity of generated states across windows.

## 3.8 Prefix-Frozen Generated-State Recovery

When a single shared model is optimized only through later commits, gradients still update the same parameters that generate the early prefix. Consequently, the input distribution faced by the recovering sufix changes during optimization, and sufix gradients can degrade already learned prefix behavior. S2.1 introduces a prefix-frozen temporal cascade that aligns temporal responsibilities with parameter ownership. Models A and B have identical architectures but disjoint parameter sets, both initialized from S1 step 120k. A remains permanently in evaluation mode and is excluded from the optimizer, scheduler, and AMP scaler; B is the only model updated. With zero-based global commit sets and their executed time intervals defined as

$$
\begin{array} { c } { { \mathcal { C } _ { A } = \{ 0 , 1 \} , \qquad \mathcal { C } _ { B } = \{ 2 , 3 \} , } } \\ { { \tau ( c ) = \left[ 0 . 5 c , 0 . 5 ( c + 1 ) \right) \mathrm { s } , } } \end{array}\tag{27}
$$

A executes the first two commits from the real 11-frame history to produce a model-generated prefix over 0–1 seconds. B neither executes nor optimizes C1/C2; it starts from the handof state at t = 1 second and learns recovery over 1–2 seconds.

The boundary is not a hidden-state distillation interface, but an auditable typed causal interface. Let $z _ { 1 } ^ { \mathrm { p h y s } }$ contain the latest 11-frame generated history, current kinematic state, active/lifecycle state, map context, and causally available signal context; let $w _ { A }$ be the integer World index selected by A; and let µ encode scene identity and generated-history provenance. Then

$$
\begin{array} { r } { \pmb { s } _ { 1 } ^ { A } = \mathrm { R o l l o u t } _ { \theta _ { A } } ( \pmb { H } _ { 0 } ; \mathcal { C } _ { A } ) , \quad } \\ { \pmb { z } _ { 1 } ^ { A  B } = \mathrm { s g } ( \pmb { z } _ { 1 } ^ { \mathrm { p h y s } } ( \pmb { s } _ { 1 } ^ { A } ) , w _ { A } , \pmb { \mu } ) , } \end{array}
$$

$$
\left( F _ { 1 } ^ { B } , \{ b _ { 1 } ^ { B , w } \} _ { w = 1 } ^ { 8 } \right) = \mathrm { R e b u i l d } _ { \theta _ { B } } \left( z _ { 1 } ^ { \mathrm { p h y s } } \right) , \qquad w _ { B } \gets w _ { A } .\tag{28}
$$

All tensors passed through the boundary are detached and cloned. Encoder hidden states, KV/cache, decoder memory, Slow Goal, Goal/World features, interaction summaries, Preview warm starts, and the computation graph of A are discarded. B reruns the factual encoder, regenerates Goal proposals and World representations as well as its own Slow Goal, and reconstructs interaction from the new Preview and physical state. The active state, causal-exit state, and its confirmation count are carried forward as explicit lifecycle facts. Equation (28) preserves routing through the same discrete World index but neither transfers nor presupposes semantic continuity between the neural hidden states of A and B.

S2.1 optimizes B only on the generated-state continuation at global Commits 3/4:

$$
\begin{array} { c } { { \displaystyle \mathcal { L } _ { \mathrm { S 2 . 1 } } \big ( \theta _ { B } ; \theta _ { A } \big ) = \sum _ { c = 2 } ^ { 3 } \sum _ { m \in \mathcal { M } _ { \mathrm { r e c } } } \lambda _ { m } \overline { { { \mathcal { L } } } } _ { m , c } , } } \\ { { \mathcal { L } _ { 0 } ^ { B } = \mathcal { L } _ { 1 } ^ { B } = 0 , \qquad \theta _ { A } ^ { k + 1 } = \theta _ { A } ^ { k } , } } \\ { { \theta _ { B } ^ { k + 1 } = \theta _ { B } ^ { k } - \eta \nabla _ { \theta _ { B } } \mathcal { L } _ { \mathrm { S 2 . 1 } } . } } \end{array}\tag{29}
$$

Here, $\mathcal { M } _ { \mathrm { r e c } }$ includes Slow Goal continuity as well as the recursive Preview, Interaction, Control, State, Closed-loop, Safety, Map, and Kinematic objectives; static Goal proposal, World Scorer, and World diversity terms are disabled. Supervision for the two B commits begins at future steps 10 and 15, respectively. During training, a detached GT-WTA proxy that depends only on future indices 14/19 may select the branch and supply supervised edges for C3/C4. Perturbing future[0:10] does not change the winner, and ground truth is never written into generated history. During target-free validation and deployment, A selects w<sub>A</sub> using its causal Scorer and, after re-encoding, B is routed only along the same index.

## 4 Experiments

We evaluate the step-120k checkpoint obtained from a complete S1 training run on the full H-D public-validation split of 955 scenarios using an eight-second closed-loop rollout. Scenario-joint displacement is the primary result because it measures whether one World explains all evaluated agents in a scene, while per-agent World selection is retained as a complementary diagnostic of marginal mode coverage.

## 4.1 Datasets and Implementation Details

Five datasets are converted into a unified ragged scene representation through dataset-specific adapters [1, 4, 5, 11, 35]. The step-120k S1 checkpoint uses fixed Waymo/AV2/H D training splits, whereas S2.1 uses a five-source, train-only manifest. Table 2 summarizes the scheduled source allocation over 100k optimization steps and the corresponding heldout boundaries. Approximately 3.2% of the Waymo scenarios that do not strictly follow the 10 Hz sampling rate are removed according to the predefined filtering rule. Future trafic signals are supplied exogenously by the environment when available and otherwise fully masked. Missing elevation is likewise strictly masked; the model does not impute either elevation or future signal states.

Throughout S2.1, source weights are reported in the fixed order H-D/WOMD/AV2/inD/uniD. The respective weights are 35%, 25%, 15%, 15%, and 10% during steps 0–40k; 50%, 18%, 10%, 14%, and 8% during steps 40–80k; and 70%, 10%, 5%, 10%, and 5% during steps 80–100k. The manifest contains no validation entries, and the training entry point fails closed if any path or name contains valid, validation, audit, or test.

The main model uses an embedding dimension of $d =$ 384, 32 Goal Regions, eight Worlds, four Preview nodes, and two rounds of JPI. The S1 instance contains 114,106,461 trainable parameters. Learnable modules operate in BF16, whereas future-interaction geometry, dynamics, key probabilistic operations, and loss reductions are evaluated in FP32. The probability-domain BCE for the Slow Goal is computed in an autocast-disabled FP32 region without changing the mathematical objective or gradient definition. S2.1 is configured for 100k optimization steps with a physical batch size of 16 on a single 80-GB H100. Models A and B are two complete, independent model instances, and the optimizer contains only the parameters of B.

## 4.2 Evaluation Protocol and Metrics

We perform a full eight-second closed-loop rollout on all 955 scenarios in the H-D public-validation split; all scenarios complete successfully, with no skipped or failed cases. The model produces eight learned Worlds and four temporally persistent Control samples per World, giving the 32 stochastic rollouts required by the oficial evaluator. Probabilities used by the World-level Brier metrics are defined over the eight learned Worlds; the four within-World Control samples are not treated as additional semantic modes.

For scenario s, horizon τ, evaluated-agent set $A _ { s } ( \tau )$ and World w, let $\mathrm { A D E } _ { s , i , w } ( \tau )$ and $\mathrm { F D E } _ { s , i , w } ( \tau )$ denote the displacement errors of agent i. Scenario-joint evaluation selects one World for the entire scene,

$$
w _ { s } ^ { \star } ( \tau ) = \arg \operatorname* { m i n } _ { 1 \leq w \leq 8 } \frac { 1 } { \lvert \mathcal { A } _ { s } ( \tau ) \rvert } \sum _ { i \in \mathcal { A } _ { s } ( \tau ) } \mathrm { F D E } _ { s , i , w } ( \tau ) ,\tag{30}
$$

and reports both ADE and FDE from this same $w _ { s } ^ { \star } ( \tau )$ , denoted ADE-at-joint-minFDE@8 and joint-minFDE@8, respectively. Thus, every agent in a scene is evaluated under a shared joint-future hypothesis. For the secondary agent-centric diagnostic, each agent instead selects $w _ { s , i } ^ { \dagger } =$ arg $\begin{array} { r } { \operatorname* { m i n } _ { w } \mathrm { F D E } _ { s , i , w } ; } \end{array}$ we report FDE at this mode and ADE from the same mode, denoted ADE-at-minFDE. We additionally define $\widehat { w } _ { s , i } = \arg \operatorname* { m i n } _ { w } \mathrm { A D E } _ { s , i , w }$ and report the indepenbdently ADE-selected oracle-minADE. Neither diagnostic is a jointly realizable World because diferent agents may select diferent modes.

## 4.3 Full-955 Closed-Loop Results

Scenario-joint displacement. Table 3 evaluates the eight learned Worlds under the shared selection rule in Eq. (30). The short-horizon values quantify accurate local continuation, while the complete eight-second rollout reaches a scenario-joint ADE-at-joint-minFDE@8/joint-minFDE@8 of 2.377028/7.529927 m. Crucially, each number is obtained from one World selected for all evaluated agents in a scene rather than from a post-hoc composition of agent-wise modes.

Table 3. Scenario-joint displacement over the full H-D 955-scenario evaluation. At every horizon, a single selected World explains all evaluated agents in the scene.
<table><tr><td>Horizon</td><td>ADE-at-joint- minFDE@8 (m)</td><td>joint-minFDE@8 (m)</td></tr><tr><td>1 s</td><td>0.051287</td><td>0.067154</td></tr><tr><td>2 s</td><td>0.130339</td><td>0.342070</td></tr><tr><td>3s</td><td>0.290057</td><td>0.848773</td></tr><tr><td>4 s</td><td>0.528915</td><td>1.585686</td></tr><tr><td>5s</td><td>0.849769</td><td>2.579771</td></tr><tr><td>8 s</td><td>2.377028</td><td>7.529927</td></tr></table>

Agent-centric diagnostic. Table 4 reports the agent-centric oracle displacement across evaluation horizons. When each agent is allowed to select its own World among all eight modes, FDE-based selection gives an eight-second ADEat-minFDE@8/minFDE@8 of 1.203089/3.879052 m. Independently selecting the ADE-optimal World gives oracleminADE@8 of 1.196636 m. These lower agent-centric errors diagnose marginal mode coverage; because the selected World may difer across agents, they do not replace the scenario-joint results above.

Cross-benchmark context. To place the displacement scale in context, Table 5 juxtaposes public open-loop motionforecasting results with the H-D closed-loop diagnostics above while retaining each benchmark’s native protocol. The HI-FLOOP agent-centric rows use the same $K = 8$ selection budget at both six and eight seconds. The additional eightsecond Scene K = 8 row instead requires one World to explain all evaluated agents jointly.

## 4.4 Mechanism-Level Structural Checks

The complete HI-FLOOP contract enforces the scenelevel branch identity $w _ { c } = w _ { 0 }$ , current-fact encoding $\pmb { F } _ { c } =$ $E ( H _ { c } , M )$ , and the executed-prefix transition $\pmb { H } _ { c + 1 } =$ Rol $| ( \pmb { H } _ { c } , \pmb { X } _ { c } ^ { 1 : 5 } )$ . Table 6 records definition-level mechanism checks obtained by changing one contract condition at a time. These checks clarify what each mechanism guarantees; they are not empirical performance ablations.

Table 2. Five-source train-only sampling contract for S2.1 and the H-D evaluation split. The allocation column reports the planned source allocation accumu lated over 100k optimization steps.
<table><tr><td>Dataset</td><td>Training identity</td><td>100k allocation</td><td>Evaluation role</td></tr><tr><td>H-D</td><td>Official training split, S1-matched release</td><td>48.0k</td><td>Public-valid 955 as an independent evaluation set</td></tr><tr><td>WOMD</td><td>Official training split</td><td>19.2k</td><td>Official validation split as a held-out evaluation set</td></tr><tr><td>Argoverse 2</td><td>Official training split</td><td>11.0k</td><td>Official validation split as a held-out evaluation set</td></tr><tr><td>inD</td><td>Train only</td><td>13.6k</td><td>Valid/audit/test remain held out</td></tr><tr><td>uniD</td><td>Train only</td><td>8.2k</td><td>Valid/audit/test remain held out</td></tr><tr><td>H-D public-valid</td><td>Evaluation only, 955 scenarios</td><td></td><td>Eight-second closed-loop evaluation cohort</td></tr></table>

Table 4. Agent-centric oracle displacement at every integer-second horizon on H-D full-955. Each column independently selects its best World per agent; only agents with a valid endpoint at that horizon are included.
<table><tr><td>Horizon</td><td>oracle-minADE@8 (m)</td><td>minFDE@8 (m)</td></tr><tr><td>1 s</td><td>0.034846</td><td>0.049019</td></tr><tr><td>2 s</td><td>0.086638</td><td>0.223141</td></tr><tr><td>3s</td><td>0.181491</td><td>0.520086</td></tr><tr><td>4 s</td><td>0.314824</td><td>0.934945</td></tr><tr><td>5s</td><td>0.486963</td><td>1.484945</td></tr><tr><td>6s</td><td>0.696436</td><td>2.169126</td></tr><tr><td>7 s</td><td>0.934998</td><td>2.970606</td></tr><tr><td>8s</td><td>1.196636</td><td>3.879052</td></tr></table>

Together, these checks isolate branch persistence, executed-state feedback, temporal decomposition, and explicit future-interaction conditioning as distinct properties of the proposed computation contract.

## 4.5 S1 Optimization Diagnostic

Figure 2 records the composite training objective at 10kstep intervals for the S1 run used above. We include this curve solely as an optimization diagnostic; its values are training objectives and are not used as evidence of held-out or closedloop performance.

![](images/e9d514b6d2dd2014e01006613f84a26c1377bc3ee9ca5c582a6702e7f061daed.jpg)  
Figure 2. S1 optimization diagnostic. The curve reports the composite training objective in Eq. (26) at 10k-step checkpoints during the 120k-step run; markers denote logged values, and the connecting line is not smoothed.

## 5 Conclusion

We presented HI-FLOOP, a branch-consistent, multitimescale state-feedback framework for closed-loop generation. Within a fixed scene-level joint World, the long-horizon Goal, rolling JPI plan, and short-horizon Control operate with distinct horizons, temporal persistence, and update semantics. At each commit, only the dynamically executed 0.5- s prefix is appended to the history and re-encoded as factual context for the next iteration. The scene-level World preserves a shared branch identity across agents and commits, while the Goal, interaction relations, and local motion adapt to newly generated facts within that branch. As an extension for generated-state training, the A→B temporal cascade isolates sufix gradients through a frozen prefix model and a recovery model with disjoint parameters, with re-encoding across a typed physical-state boundary. Under the requirement that a single World jointly explain every evaluated agent in a scene, the complete S1 run obtains an eight-second ADE-at-joint-minFDE@8/jointminFDE@8 of 2.377028/7.529927 m on all 955 H-D publicvalidation scenarios; independent per-agent ADE selection gives oracle-minADE@8 of 1.196636 m.

## Acknowledgments

We gratefully acknowledge the support of the School of Systems Science, Beijing Normal University. The unified data interface in this work covers the Waymo Open Motion Dataset, Argoverse 2, and H-D (HetroD), while the S2.1 train-only mixture additionally uses inD and uniD to complement heterogeneous road-user and shared-space traffic scenarios. We thank the respective dataset maintainers for making these resources available [1, 4, 5, 11, 35]. We further acknowledge Joint Metrics Matter and the Waymo Open Sim Agents Challenge for advancing the evaluation of joint futures and closed-loop realism [16, 33], as well as Waymax and ScenarioNet for providing open infrastructure that accelerates simulation and cross-dataset scene management [6, 12]. The precise data splits, adaptation rules, and scope of tool usage follow the experimental setup described above.

## References

[1] Julian Bock, Robert Krajewski, Tobias Moers, Stefen Runde, Lennart Vater, and Lutz Eckstein. The inD dataset: A drone dataset of natural istic road user trajectories at German intersections. In IEEE Intelligent Vehicles Symposium, pages 1929–1934, 2020.

[2] Yulong Cao, Boris Ivanovic, Chaowei Xiao, and Marco Pavone. Reinforcement learning with human feedback for realistic trafic simulation. In IEEE International Conference on Robotics and Automation, pages 14428–14434, 2024.

[3] Keyu Chen, Wenchao Sun, Hao Cheng, Zheng Fu, and Sifa Zheng. ForSim: Stepwise forward simulation for trafic policy fine-tuning. In IEEE International Conference on Robotics andAutomation, 2026. Paper TuI1I.138; arXiv:2602.01916.

[4] Yu-Hsiang Chen, Wei-Jer Chang, Christian Kotulla, Thomas Keutgens, Stefen Runde, Tobias Moers, Christoph Klas, Wei Zhan, Masayoshi Tomizuka, and Yi-Ting Chen. HetroD: A high-fidelity drone dataset and benchmark for autonomous driving in heterogeneous trafic. In IEEE International Conference on Robotics andAutomation, 2026.

[5] Scott Ettinger, Shuyang Cheng, Benjamin Caine, Chenxi Liu, Hang Zhao, Sabeek Pradhan, Yuning Chai, Ben Sapp, Charles R. Qi, Yin Zhou, Zoey Yang, Aurélien Chouard, Pei Sun, Jiquan Ngiam, Vijay Vasudevan, Alexander McCauley, Jonathon Shlens, and Dragomir Anguelov. Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset. In IEEE/CVF International Conference on Computer Vision, pages 9710–9719, 2021.

Table 5. Contextual displacement reference under native benchmark protocols. Published rows use their original test protocols; HI-FLOOP uses the H-D public valid full-955 closed-loop protocol. Values are not directly comparable across datasets, horizons, aggregation schemes, or selection rules. The NDPNet WOMD entry follows the corrected epoch-31 result released with the oficial implementation.
<table><tr><td>Method</td><td>Dataset</td><td>Horizon</td><td>Selection</td><td>minADE (m)</td><td>minFDE (m)</td></tr><tr><td>NDPNet [8]</td><td>WOMD</td><td>8s</td><td>Agent K = 6</td><td>0.840</td><td>1.751</td></tr><tr><td>NDPNet [8]</td><td>AV2</td><td>6 s</td><td>Agent K = 6</td><td>0.610</td><td>1.170</td></tr><tr><td>MTR++ ensemble [25]</td><td>WOMD</td><td>3/5/8 s avg.</td><td>Agent K = 6</td><td>0.558</td><td>1.117</td></tr><tr><td>MTR++ [25]</td><td>WOMD</td><td>3/5/8 s avg.</td><td>Agent K = 6</td><td>0.591</td><td>1.194</td></tr><tr><td>QCNet ensemble [45]</td><td>AV2</td><td>6s</td><td>Agent K = 6</td><td>0.620</td><td>1.190</td></tr><tr><td>QCNet [45]</td><td>AV2</td><td>6 s</td><td>Agent K = 6</td><td>0.650</td><td>1.290</td></tr><tr><td>MTR [24, 45]</td><td>AV2</td><td>6s</td><td>Agent K = 6</td><td>0.730</td><td>1.440</td></tr><tr><td>Hi-FLoop (ours)</td><td>H-D</td><td>6 s</td><td>Agent K = 8</td><td>0.696</td><td>2.169</td></tr><tr><td>Hi-FLoop (ours)</td><td>H-D</td><td>8s</td><td>Agent K = 8</td><td>1.197</td><td>3.879</td></tr><tr><td>Hi-FLoop (ours, joint)</td><td>H-D</td><td>8 s</td><td>Scene K = 8</td><td>2.377</td><td>7.530</td></tr></table>

Table 6. Definition-level checks of the mechanisms in HI-FLOOP. Each row changes one contract condition and identifies the structural property that is consequently removed; the table does not report measured performance ablations.
<table><tr><td>Mechanism check</td><td>Controlled alteration</td><td>Consequence under the model definition</td></tr><tr><td>Persistent joint branch</td><td>Reselect wc = arg maxw sc,w at every commit</td><td>Adjacent commits may follow different long-horizon hypotheses, so temporal branch consistency is no longer guaranteed</td></tr><tr><td>World semantics</td><td>Replace eight Worlds by one World with 32 Control samples</td><td>Execution stochasticity remains, but scene-level diversity of joint hypotheses is removed; sample count alone does not recover multi-World semantics</td></tr><tr><td>Executed-state feedback</td><td>Hold the factual representation fixed at E(H0, M) while dynamics advances</td><td>Subsequent Goals, Previews, and Controls cannot condition on generated changes in position, neighborhood, or conflict timing</td></tr><tr><td>Multi-timescale state</td><td>Remove the persistent Goal and two-second Preview and decode one-second Control directly</td><td>No separate state carries long-term behavioral direction or a mid-term interaction plan, collapsing the explicit temporal decomposition</td></tr><tr><td>Future-graph message</td><td>Zero the JPI graph-message residual and remove conflict/∆ETA conditioning from Control</td><td>Agents remain coupled by factual attention and the shared World, but predicted future conflicts and passing order cease to condition local motion</td></tr></table>

[6] Cole Gulino, Justin Fu, Wenjie Luo, George Tucker, Eli Bronstein, Yiren Lu, Jean Harb, Xinlei Pan, Yan Wang, Xiangyu Chen, John D. Co-Reyes, Rishabh Agarwal, Rebecca Roelofs, Yao Lu, Nico Montali, Paul Mougin, Zoey Yang, Brandyn White, Aleksandra Faust, Rowan McAllister, Dragomir Anguelov, and Benjamin Sapp. Waymax: An accelerated, data-driven simulator for large-scale autonomous driving research. In Advances in Neural Information Processing Systems, pages 7730–7742, 2023.

[7] Ke Guo, Zhenwei Miao, Wei Jing, Weiwei Liu, Weizi Li, Dayang Hao, and Jia Pan. LASIL: Learner-aware supervised imitation learning for long-term microscopic trafic simulation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15386–15395, 2024.

[8] Hua Hu, Zikang Zhou, Qian Zhou, Zihao Wen, Junjie Hu, Xinhong Chen, Zhengmin Jiang, Yung-Hui Li, and Jianping Wang. Perceiving the near, reasoning the distant: Coherent long-horizon trajectory prediction for autonomous driving. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24875–24884, 2026.

[9] Chiyu Max Jiang, Andre Cornman, Cheolho Park, Benjamin Sapp, Yin Zhou, and Dragomir Anguelov. MotionDifuser: Controllable multiagent motion prediction using difusion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9644–9653, 2023.

[10] Chiyu Max Jiang, Yijing Bai, Andre Cornman, Christopher Davis, Xiukun Huang, Hong Jeon, Sakshum Kulshrestha, John Lambert, Shuangyu Li, Xuanyu Zhou, Carlos Fuertes, Chang Yuan, Mingxing Tan, Yin Zhou, and Dragomir Anguelov. SceneDifuser: Eficient and controllable driving simulation initialization and rollout. In Advances in Neural Information Processing Systems, pages 55729–55760, 2024.

[11] leveLXData by fka GmbH. uniD: The University Drone Dataset. https://levelxdata.com/unid-dataset/, 2021.

[12] Quanyi Li, Zhenghao Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo, and Bolei Zhou. ScenarioNet: Open-source platform for large-scale trafic scenario simulation and modeling. In Advances in Neural Information Processing Systems, pages 3894–3920, 2023.

[13] Haohong Lin, Xin Huang, Tung Phan, David Hayden, Huan Zhang, Ding Zhao, Siddhartha Srinivasa, Eric Wolf, and Hongge Chen. Causal composition difusion model for closed-loop trafic generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27542–27552, 2025.

[14] Haolan Liu, Jishen Zhao, and Liangjun Zhang. HMSim: A hierarchical multi-agent simulator for autonomous vehicles. In IEEE International Conference on Robotics and Automation, 2026.

[15] Wenjie Luo, Cheol Park, Andre Cornman, Benjamin Sapp, and Dragomir Anguelov. JFP: Joint future prediction with interactive multi-agent modeling for autonomous driving. In Proceedings of the 6th Conference on Robot Learning, pages 1457–1467, 2023.

[16] Nico Montali, John Lambert, Paul Mougin, Alex Kuefler, Nicholas Rhinehart, Michelle Li, Cole Gulino, Tristan Emrich, Zoey Yang, Shimon Whiteson, Brandyn White, and Dragomir Anguelov. The waymo open sim agents challenge. In Advances in Neural Information Processing Systems, pages 59151–59171, 2023.

[17] Jiquan Ngiam, Benjamin Caine, Vijay Vasudevan, Zhengdong Zhang, Hao-Tien Lewis Chiang, Jefrey Ling, Rebecca Roelofs, Alex Bewley, Chenxi Liu, Ashish Venugopal, David Weiss, Ben Sapp, Zhifeng Chen, and Jonathon Shlens. Scene transformer: A unified architecture for predicting multiple agent trajectories. In International Conference on Learning Representations, 2022.

[18] Muleilan Pei, Shaoshuai Shi, and Shaojie Shen. Advancing multiagent trafic simulation via R1-style reinforcement fine-tuning. In International Conference on Learning Representations, 2026.

[19] Zhenghao Peng, Wenjie Luo, Yiren Lu, Tianyi Shen, Cole Gulino, Ari Sef, and Justin Fu. Improving agent behaviors with RL fine-tuning for autonomous driving. In European Conference on Computer Vision, pages 165–181. Springer, 2024.

[20] Jonah Philion, Xue Bin Peng, and Sanja Fidler. Trajeglish: Trafic modeling as next-token prediction. In International Conference on Learning Representations, 2024.

[21] Mozhgan Pourkeshavarz, Tianran Liu, and Nicholas Rhinehart. AutoWorld: Learning multi-agent trafic simulation with self-supervised world models. arXiv preprint arXiv:2603.28963, 2026.

[22] Luke Rowe, Martin Ethier, Eli-Henry Dykhne, and Krzysztof Czarnecki. FJMP: Factorized joint multi-agent motion prediction over learned directed acyclic interaction graphs. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13745–13755, 2023.

[23] Ari Sef, Brian Cera, Dian Chen, Mason Ng, Aurick Zhou, Niga maa Nayakanti, Khaled S. Refaat, Rami Al-Rfou, and Benjamin Sapp. MotionLM: Multi-agent motion forecasting as language modeling. In IEEE/CVF International Conference on Computer Vision, pages 8579–8590, 2023.

[24] Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele. Motion trans former with global intention localization and local movement refine ment. In Advances in Neural Information Processing Systems, pages 6531–6543, 2022.

[25] Shaoshuai Shi, Li Jiang, Dengxin Dai, and Bernt Schiele. MTR++: Multi-agent motion prediction with symmetric scene modeling and guided intention querying. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):3955–3971, 2024.

[26] Shinya Shiroshita, Shirou Maruyama, Daisuke Nishiyama, Mario Yno cente Castro, Karim Hamzaoui, Guy Rosman, Jonathan A. DeCastro, Kuan-Hui Lee, and Adrien Gaidon. Behaviorally diverse trafic simulation via reinforcement learning. In IEEE/RSJ International Confer ence on Intelligent Robots and Systems, pages 2103–2110, 2020.

[27] Qiao Sun, Xin Huang, Junru Gu, Brian C. Williams, and Hang Zhao. M2I: From factored marginal trajectory prediction to interactive prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6543–6552, 2022.

[28] Simon Suo, Sebastian Regalado, Sergio Casas, and Raquel Urtasun. TraficSim: Learning to simulate realistic multi-agent behaviors. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10400–10409, 2021.

[29] Simon Suo, Kelvin Wong, Justin Xu, James Tu, Alexander Cui, Sergio Casas, and Raquel Urtasun. MixSim: A hierarchical framework for mixed reality trafic simulation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9622–9631, 2023.

[30] Shuhan Tan, Boris Ivanovic, Yuxiao Chen, Boyi Li, Xinshuo Weng, Yulong Cao, Philipp Kraehenbuehl, and Marco Pavone. Promptable closed-loop trafic simulation. In Proceedings of the 8th Conference on Robot Learning, pages 5087–5105, 2025.

[31] Shuhan Tan, John Lambert, Hong Jeon, Sakshum Kulshrestha, Yijing Bai, Jing Luo, Dragomir Anguelov, Mingxing Tan, and Chiyu Max Jiang. SceneDifuser++: City-scale trafic simulation via a generative world model. In IEEE/CVF Conference on Computer Vision and Pat tern Recognition, pages 1570–1580, 2025.

[32] Xiaolong Tang, Meina Kan, Shiguang Shan, and Xilin Chen. Plan-R1: Safe and feasible trajectory planning as language modeling. In International Conference on Learning Representations, 2026.

[33] Erica Weng, Hana Hoshino, Deva Ramanan, and Kris M. Kitani. Joint metrics matter: A better standard for trajectory forecasting. In IEEE/CVF International Conference on Computer Vision, pages 20315–20326, 2023.

[34] Ronald J. Williams and Jing Peng. An eficient gradient-based algorithm for on-line training of recurrent network trajectories. Neural Computation, 2(4):490–501, 1990.

[35] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jag jeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, Deva Ramanan, Peter Carr, and James Hays. Argoverse 2: Next generation datasets for self-driving perception and forecasting. In Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

[36] Wei Wu, Xiaoxin Feng, Ziyan Gao, and Yuheng Kan. SMART: Scal able multi-agent real-time motion generation via next-token prediction. In Advances in Neural Information Processing Systems, pages 114048– 114071, 2024.

[37] Lingyu Xiao, Zexin Feng, and Xintao Yan. Long-term trafic simulation via structured autoregressive modeling. In European Conference on Computer Vision, 2026. arXiv:2606.31209.

[38] Danfei Xu, Yuxiao Chen, Boris Ivanovic, and Marco Pavone. BITS: Bilevel imitation for trafic simulation. In IEEE International Conference on Robotics and Automation, pages 2929–2936, 2023.

[39] Harsh Yadav, Christian Bohn, and Tobias Meisen. Rectify, don’t regret: Avoiding pitfalls of diferentiable simulation in trajectory prediction. arXiv preprint arXiv:2603.23393, 2026.

[40] Ye Yuan, Xinshuo Weng, Yanglan Ou, and Kris M. Kitani. Agent-Former: Agent-aware transformers for socio-temporal multi-agent forecasting. In IEEE/CVF International Conference on Computer Vision, pages 9813–9823, 2021.

[41] Weifan Zhang, Xiaofeng Zhao, Adel Bazzi, Mingrui Li, Yifan Wei, and Dengfeng Sun. Beyond self-play: Hierarchical reasoning for

continuous motion in closed-loop trafic simulation. arXiv preprint arXiv:2605.09153, 2026.

[42] Zhejun Zhang, Alexander Liniger, Dengxin Dai, Fisher Yu, and Luc Van Gool. TraficBots: Towards world models for autonomous driving simulation and motion prediction. In IEEE International Conference on Robotics and Automation, pages 1522–1529, 2023.

[43] Zhejun Zhang, Peter Karkus, Maximilian Igl, Wenhao Ding, Yuxiao Chen, Boris Ivanovic, and Marco Pavone. Closed-loop supervised fine-tuning of tokenized trafic models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5422–5432, 2025.

[44] Ziyuan Zhong, Davis Rempe, Danfei Xu, Yuxiao Chen, Sushant Veer, Tong Che, Baishakhi Ray, and Marco Pavone. Guided conditional diffusion for controllable trafic simulation. In IEEE International Conference on Robotics and Automation, pages 3560–3566, 2023.

[45] Zikang Zhou, Jianping Wang, Yung-Hui Li, and Yu-Kai Huang. Querycentric trajectory prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17863–17873, 2023.

[46] Zikang Zhou, Haibo Hu, Xinhong Chen, Jianping Wang, Nan Guan, Kui Wu, Yung-Hui Li, Yu-Kai Huang, and Chun Jason Xue. BehaviorGPT: Smart agent simulation for autonomous driving with nextpatch prediction. In Advances in Neural Information Processing Systems, pages 79597–79617, 2024.

[47] Yiyao Zhu, Di Luan, and Shaojie Shen. BiFF: Bi-level future fusion with polyline-based coordinate for interactive trajectory prediction. In IEEE/CVF International Conference on Computer Vision, pages 8260– 8271, 2023.