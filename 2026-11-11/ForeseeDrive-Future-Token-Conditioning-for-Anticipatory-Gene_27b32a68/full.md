# ForeseeDrive: Future-Token Conditioning for Anticipatory Generative Driving

<sub>Shaokai</sub> <sub>Wu1,2,</sub> <sub>Xunchu</sub> <sub>Zhou1,2,</sub> <sub>Pinzhi</sub> <sub>Wang2,</sub> <sub>Bo</sub> <sub>Dong2</sub>ive: Future-Token Conditioning for Anticipatory Gener Yuting Hu<sup>1</sup>, Yanbiao Ji<sup>1</sup>, Qiuchang Li<sup>1</sup>, Xiao Cui<sup>2</sup> Qi Kong<sup>2</sup>, Hongtao Lu<sup>1</sup>, Yue Ding<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Zelos Technology

## Abstract

Generative driving planners can represent diverse ego fu-<sup>Generative</sup> <sup>driving</sup> <sup>planners</sup> <sup>can</sup> <sup>represent</sup> <sup>diverse</sup> <sup>ego</sup> <sup>fu-</sup> tures, but a downstream scorer can choose only among<sup>tures,</sup> <sup>ut</sup> <sup>a</sup> <sup>ownstream</sup> <sup>scorer</sup> <sup>can</sup> <sup>c oose</sup> <sup>on y</sup> <sup>among</sup> <sup>sam-</sup> sampled candidates; it cannot recover an absent maneuver. We present ForeseeDrive, which turns future sceneinto a deployable interface for shaping candidate generaevolution into a deployable interface for shaping candi-<sup>tion.</sup> <sup>Rather</sup> <sup>than</sup> <sup>reconstructing</sup> <sup>dense</sup> <sup>future</sup> <sup>states</sup> <sup>or</sup> <sup>scor-</sup> date generation. Rather than reconstructing dense future states or scoring afixed menu, ForeseeDrive derives query-<sub>interactions, and TTC margins, and predicts the same to-</sub> indexed tokensfrom logged actor motion, occupancy evolu-ken space from present context at deployment. The tokens tion, route interactions, and TTC margins, and predicts the<sup>condition</sup> <sup>trajectory</sup> <sup>geometry</sup> <sup>before</sup> <sup>candidates</sup> <sup>materialize.</sup> same token space from present context at deployment. The tokens condition trajectory geometry before candidates ma-brittle safe-to-unsafe boundary crossings. Under their respecterialize. A zero-token-gated residual path and candidate-<sup>tive</sup> <sup>benchmark</sup> <sup>protocols,</sup> <sup>ForeseeDrive</sup> <sup>reaches</sup> <sup>89.2</sup> <sup>PDMS</sup> preservation loss limit unnecessary drift, while a TTCmargin guard targets brittle safe-to-unsafe boundary cross-<sub>Bench2Drive. Controlled DifusionDrive/NAVSIM diagnos-</sub> ings. Under their respective benchmark protocols, Fore-tics show an exact zero-token no-op, a small aligned-overseeDrive reaches 89.2 PDMS on NAVSIM navtest, 0.41 m<sup>shufled</sup> <sup>selected-PDM</sup> <sup>advantage</sup> <sup>under</sup> <sup>preservation,</sup> <sup>mixed</sup> average L2 and 0.05% collision rate on nuScenes validation, and a 76.50 driving score on Bench2Drive. Controlledsupport future-conditioned generation as a practical interface DiffusionDrive/NAVSIM diagnostics show an exact zero-<sup>for</sup> <sup>anticipation.</sup> <sup>Code</sup> <sup>is</sup> <sup>included</sup> <sup>in</sup> <sup>the</sup> <sup>supplementary</sup> <sup>mate-</sup> token no-op, a small aligned-over-shuffled selected-PDM<sup>rials.</sup> advantage under preservation, mixed oracle-best quality, and localized TTC-boundary failures. Together, the bench-Introduction mark results and bounded mechanism tests support futureconditioned generation as a practical interface for anticipation. Code is included in the supplementary materials.tion (Caesar et al. 2020; Sun et al. 2020; Chang et al. 2019

## 1. Introduction2023a; Liao et al

Autonomous driving planning maps recent observations,challenge is not only uncertain interaction: the planner mu <sub>map</sub> <sub>context,</sub> <sub>and</sub> <sub>route</sub> <sub>goals</sub> <sub>into</sub> <sub>safe,</sub> <sub>progressive</sub> <sub>ego</sub>instantiate ego modes before those interactions fully unfold <sub>motion [1, 2, 4, 8, 43, 48]. Recent end-to-end and gen-</sub>Once a candidate set is materialized, a downstream scorer ca erative planners reduce brittleness by representing multiple ego futures rather than committing to one trajec-tribution, citation, or public sharing of this manuscript is strictl tory [6, 15, 20, 29]. Yet a generative planner can still fail<sup>prohibited.</sup> <sup>Copyright</sup> <sup>and</sup> <sup>publication</sup> <sup>details</sup> <sup>will</sup> <sup>appear</sup> <sup>in</sup> <sup>th</sup> even when the appropriate behavior is clear in hindsight.

![](images/27117ae2b6d9a3bef845848d3d8f8221d07a3c761c6347394ef8ea8692939b9f.jpg)  
Figure 1: Top: fixed-candidate future scoring acts after a base genfuture-aware mode. Bottom: ForeseeDrive predicts compacterator produces candidates, so it cannot create an absent futurefuture tokens from present context and conditions the gen-aware mode. Bottom: ForeseeDrive predicts compact future toerator before materialization; a TTC-margin guard rejectskens from present context and conditions the generator before mabrittle near-contact transitions while candidate preservationterialization; a TTC-margin guard rejects brittle near-contact tran-<sup>limits</sup> <sup>drift.</sup>sitions while candidate preservation limits drift.

only reorder sampled trajectories; it cannot recover a yield-<sub>The challenge is not only uncertain interaction: the planner</sub> <sup>ing,</sup> <sup>braking,</sup> <sup>nudging,</sup> <sup>or</sup> <sup>passing</sup> <sup>maneuver</sup> <sup>absent</sup> <sup>from</sup> <sup>that</sup><sub>must instantiate ego modes before those interactions fully</sub> unfold. Once a candidate set is materialized, a downstream scorer can only reorder sampled trajectories; it cannot re-<sub>This view exposes a mismatch between future prediction</sub>cover a yielding, braking, nudging, or passing maneuver and trajectory generation. Existing future-centric methodsabsent from that set. As Figure 1 illustrates, anticipation <sup>demonstrate</sup> <sup>the</sup> <sup>value</sup> <sup>of</sup> <sup>actor</sup> <sup>forecasting,</sup> <sup>occupancy/BEV</sup>can therefore fail as a support-allocation problem before it prediction, and future-awar<sub>becomes a ranking problem.</sub>

This view exposes a mismatch between future prediction and trajectory generation. Existing future-centric methods <sub>a generator needs compact future evidence that changes</sub>demonstrate the value of actor forecasting, occupancy/BEV which ego modes receive probability mass. Conversely,prediction, and future-aware planning [10, 13, 22, 31, 52, future-aware scoring introduces anticipation only after the<sub>53, 55]. A dense rollout, however, describes the scene</sub> <sup>candidate</sup> <sup>menu</sup> <sup>is</sup> <sup>fixed</sup> <sup>and</sup> <sup>remains</sup> <sup>bounded</sup> <sup>by</sup> <sup>sampled</sup>broadly, whereas a generator needs compact future evidence that changes which ego modes receive probability mass. Conversely, future-aware scoring introduces anticicentral idea is to distill future scene evolution into compact<sup>pation</sup> <sup>only</sup> <sup>after</sup> <sup>the</sup> <sup>candidate</sup> <sup>menu</sup> <sup>is</sup> <sup>fixed</sup> <sup>and</sup> <sup>remains</sup> tokens that answer action-relevant planning queries and enterbounded by sampled support. What is missing is a deploythe generator before sampling. During training, logged actorable, planner-facing future representation that acts before <sup>motion,</sup> <sup>occupancy</sup> <sup>chan</sup>candidate materialization.

We propose ForeseeDrive to provide this interface. Its central idea is to distill future scene evolution into compact tokens that answer action-relevant planning queries and enter the generator before sampling. During training, logged actor motion, occupancy changes, route interactions, and time-to-collision (TTC) margins supervise the token space; at deployment, a predictor estimates the same variables using only present observation history, map, and route. The predicted tokens can reshape trajectory geometry and mode allocation rather than only the final scores of a fixed set.

The resulting representation is queryable and actionfacing: tokens answer local planning questions rather than reconstruct every actor state or BEV cell. They capture whether an actor is becoming route-relevant, whether a reachable corridor will be contested or cleared, and whether an interaction lies near a TTC-sensitive boundary. This makes the representation lower-dimensional than a dense rollout, richer than a scalar risk score, and better aligned with generative planning decisions.

Generation-time future conditioning creates a second tension: the signal must be expressive enough to move candidate modes, yet restrained enough not to corrupt strong candidates from a pretrained planner. ForeseeDrive resolves this tension with an architectural zero-token no-op, candidate preservation, and a margin-aware TTC guard. The preservation loss limits nonzero-token drift when future evidence is weak or irrelevant. The guard targets narrow safeto-unsafe boundary crossings rather than imposing a broad clearance preference, which would also penalize controlled interactions [7, 8, 37].

Experiments evaluate both benchmark performance and the proposed mechanism. On NAVSIM navtest, ForeseeDrive reaches 89.2 PDMS and 83.5 EP; on nuScenes validation, it attains 0.41 m average L2 and 0.05% collision rate; and on Bench2Drive, it reaches a 76.50 driving score and 53.06 success rate. Same-backbone diagnostics then separate generation from selection: zero tokens exactly preserve the base candidates, aligned tokens modestly outperform shuffled tokens on selected PDM under preservation, and TTC replay localizes boundary-crossing failures. Oracle-best quality remains mixed, so we interpret these diagnostics as evidence for controllable support allocation, not universal support expansion.

Our contributions are threefold. First, we formulate anticipation in generative driving as a support-allocation problem and expose the structural limit of fixed-candidate future-aware scoring. Second, we introduce a deployable future-token interface that distills logged future evolution into queryable, action-facing variables and injects them before candidate materialization. Third, we develop a restrained integration strategy with exact zero-token preservation, base-candidate regularization, and a TTC-margin guard; we evaluate it across three benchmarks and use controlled diagnostics to separate fixed-menu scoring, mode selection, and generation-time conditioning.

## 2. Related Work

Generative planning for autonomous driving. Generative planners model a distribution over ego futures and materialize multiple trajectories for downstream selection. Diffusion-based planners use iterative denoising and guidance to represent multimodal behavior in prediction, planning, and control [5, 12, 16, 22, 29, 40, 41, 57]; generative end-to-end methods also model ego and traffic evolution in latent or tokenized spaces [56]. ForeseeDrive complements these generators with supervision that generic guidance does not specify: its query-indexed conditions have logged-future semantics, are predicted online, and enter a zero-token-gated geometry path before selection.

Future prediction and driving world models. Motion forecasting models socially plausible trajectories, mapconditioned goals, joint interactions, and multi-agent futures [3, 9, 10, 28, 32, 33, 35, 36, 39, 54]; occupancy and BEV forecasting represent evolving free space or world state [13, 31, 53, 55]. Driving world models and futureaware planners further use imagined visual, BEV, or occupancy futures to guide planning or trajectory evaluation [24, 45, 46, 52]. Our distinction from this line is the planner interface and its timing: ForeseeDrive compresses future evidence into action-facing queries and injects it before finite candidate materialization, rather than requiring a dense rollout or limiting its use to trajectory evaluation.

Structured planning interfaces. End-to-end driving systems often improve planning by exposing structured intermediate representations, including spatial-temporal BEV features, vectorized scene encodings, object-level planning tokens, interpretable safety cues, and trajectory-guided controls [6, 14, 15, 20, 34, 38, 49]. These interfaces reduce the burden on a monolithic policy and make planning more controllable. However, most of them primarily encode the current scene, task context, or general safety constraints. ForeseeDrive focuses on a different interface: future-conditioned variables that specify where scene evolution should expand, suppress, or protect ego trajectory modes.

Learning from privileged future information. Logged driving data contains future scene evolution that is unavailable at deployment, making the problem naturally related to learning using privileged information and teacher-student distillation [11, 30]. ForeseeDrive follows this train-deploy separation, but the role of privileged information differs from generic feature distillation. The teacher does not simply provide a stronger representation or a future-aware score; it defines action-facing token targets that the student predicts online and that condition generation before candidates are sampled.<sub>guard, targeting</sub>

Candidate selection and safety-aware planning. Candidate scoring and reranking are effective when a planner already generates a sufficiently diverse set of alternatives.<sup>Let</sup> <sup>x</sup>t <sup>=</sup> <sup>(H</sup>t<sup>, M</sup>t<sup>,</sup> <sup>R</sup>t<sup>)</sup> <sup>denote</sup> <sup>the</sup> <sup>deployable</sup> <sup>plannin</sup> However, selection is structurally limited by the sampled<sup>context</sup> <sup>at</sup> <sup>time</sup> <sup>,</sup> <sup>w ere</sup> <sup>t</sup> <sup>contains</sup> <sup>recent</sup> <sup>ego</sup> <sup>state,</sup> <sup>sen-</sup> support: a scorer cannot recover a yielding, braking, or     <sup>t</sup> <sup>t</sup> nudging maneuver that was never generated. Safety-aware planning and benchmark metrics further expose this issue<sub>2022; Dauner et al. 2024; Chitta et al. 2022; Hu et al</sub> because small trajectory changes near collision or TTC2023; Jiang et al. 2023a). The ego plan is a trajector boundaries can produce discontinuous metric changes [7,<sup>τ</sup> <sup>=</sup> <sup>{(x</sup>s<sup>,</sup> <sup>y</sup>s<sup>,</sup> <sup>θ</sup>s<sup>)}</sup><sub>s=1</sub> <sup>over</sup> <sup>horizon</sup> <sup>T.</sup> <sup>A</sup> <sup>base</sup> <sup>generativ</sup> 8, 37].<sup>plan</sup>

ForeseeDrive therefore combines generation-time future<sub>p (τ | x ) = p (τ | H , M , R ),</sub> conditioning with candidate preservation and a TTC-margin guard, targeting support allocation and safety-boundary<sup>and</sup> <sup>materializes</sup> <sup>a</sup> <sup>finite</sup> <sup>candidate</sup> <sup>set</sup>  <sup>=</sup> <sup>{τ }</sup>k=1 <sup>fo</sup> control.

## 3. Problem Formulation <sup>t:t+T</sup>

Let $\boldsymbol { x } _ { t } ~ = ~ ( \mathcal { H } _ { t } , \mathcal { M } _ { t } , \mathcal { R } _ { t } )$ denote the deployable planning context at time t, where $\mathcal { H } _ { t }$ contains recent ego state, sensor history, and tracked actors $A _ { t } , M _ { t }$ is the local map, and $\mathcal { R } _ { t }$ is the route command, following common map- and routeconditioned planning formulations [2, 6, 8, 15, 20]. The ego <sup>φ t:t+T</sup>  <sup>t</sup> plan is a trajectoryand learn a deplo $\tau = \{ ( x _ { s } , y _ { s } , \theta _ { s } ) \} _ { s = 1 } ^ { T }$ over horizon $T .$ A base generative planner represents

$$
p _ { \theta } ( \tau \mid x _ { t } ) = p _ { \theta } ( \tau \mid \mathcal { H } _ { t } , \mathcal { M } _ { t } , \mathcal { R } _ { t } ) ,\tag{1}
$$

and materializes a finite candidate set<sup>history,</sup> <sup>map,</sup> <sup>and</sup> <sup>route.</sup> <sup>The</sup> <sup>tokens</sup> ${ \mathcal C } = \{ \tau ^ { ( k ) } \} _ { k = 1 } ^ { K }$ <sub>for</sub> evidownstream scoring or selection.

Logged training data additionally contains future scene evolution $\mathcal { F } _ { t : t + T }$ including actor motion, occupancy <sup>changes,</sup> <sup>and</sup> <sup>route-relative</sup> <sup>interactions.</sup> <sup>This</sup> <sup>information</sup>plied only after candidate generation, first samples candidate is privileged: it may define training targets and diagnostics,from the base planner, but it is not an online input at deployment [11, 30]. We use it to define a teacher future-token space

$$
\mathbf { Z } ^ { f } = g _ { \phi } ( \mathcal { F } _ { t : t + T } , x _ { t } ) ,\tag{2}
$$

and learn a deployable predictor<sup>τ</sup> <sup>=</sup> <sup>ar</sup>g <sup>max</sup>

$$
\hat { \mathbf { Z } } ^ { f } = h _ { \psi } ( x _ { t } ) ,\tag{3)can-}
$$

where fro $\hat { \mathbf { Z } } ^ { f }$ must be estimated only from present observation<sub>0</sub>. history, map, and route. The tokens summarize future evi-In contrast, future-conditioned generation injects pre dence relevant to ego action rather than reconstructing the<sup>dicted</sup> <sup>tokens</sup> <sup>before</sup> <sup>candidate</sup> <sup>materialization:</sup> full future scene.

The central distinction is where future information enters the planning pipeline. A post-hoc future-aware selector, applied only after candidate generation, first samples candidates from the base planner,leged future supervision defines

$$
\mathcal { C } _ { 0 } \sim p _ { \theta } ( \cdot \mid x _ { t } ) ,\tag{4}
$$

![](images/5ff20e5cdeb4041031ff2346f4d8a42298cdb00ea49a3390f587874c1374c58b.jpg)  
Figure 2: ForeseeDrive framework. Top: logged future evolution defines teacher tokens for actor motion, occupancy, route interaction, and TTC margin. Middle: a deployable predictor esti-<sub>tions the base generator before sampling. Bottom: candidate</sub>mates matching tokens from present context and conditions the base generator before sampling. Bottom: candidate preservation conditioned candidates before standard selection.and a TTC-margin guard constrain the token-conditioned candidates before standard selection.

and then chooses a trajectory using a score that may depend date set is fixed.on predicted future tokens:

$$
\boldsymbol { \tau } ^ { \star } = \arg \operatorname* { m a x } _ { \boldsymbol { \tau } \in \mathcal { C } _ { 0 } } s ( \boldsymbol { \tau } , \boldsymbol { x } _ { t } , \hat { \mathbf { Z } } ^ { f } ) .\tag{<sub>he</sub>(5}
$$

ing, whereas a future-conditioned generator can change theThis selector can reorder trajectories inside the sampled distribution that produces them. The goal of ForeseeDrivecandidate set, but it cannot recover a maneuver that is absent <sup>is</sup> <sup>th</sup><sub>from</sub> $\mathcal { C } _ { 0 }$

In contrast, future-conditioned generation injects predicted tokens before candidate materialization:

$$
\mathcal { C } _ { f } \sim p _ { \boldsymbol { \theta } } ( \cdot \mid x _ { t } , \hat { \mathbf { Z } } ^ { f } ) .\tag{6}
$$

Thus, future information can change not only candidate ranking but also trajectory geometry, mode allocation, and <sup>We</sup> <sup>now</sup> <sup>present</sup> <sup>ForeseeDrive,</sup> <sup>a</sup> <sup>future-token</sup> <sup>conditioning</sup>the realized candidate set. This is the setting studied here: privileged future supervision defines action-facing targets during training, a deployable predictor estimates them online, and the resulting tokens condition generation before indexed teacher tokens Zthe candidate set is fixed.

g. Second, a deployable predictor estimates the same tokenTo make this distinction precise, we use empirical candiset from online context, Z<sup>ˆ</sup> <sup>f</sup> = hψ(xt). Third, these predic-date support to mean the trajectory modes represented in the <sup>tions</sup> <sup>condition</sup> <sup>the</sup> <sup>generator</sup> <sup>before</sup> <sup>candidate</sup> <sup>materializa-</sup>realized finite set C. A selector can only change their order-<sup>tion,</sup> <sup>producing</sup> <sup>C</sup>f <sup>∼</sup> <sup>p</sup>θ<sup>(·</sup> <sup>|</sup> <sup>x</sup>t<sup>,</sup> <sup>Z</sup> <sup>).</sup> <sup>Candidate</sup> <sup>preservation</sup>ing, whereas a future-conditioned generator can change the distribution that produces them. The goal of ForeseeDrive is therefore not merely to score trajectories with future evidence, but to allocate candidate support toward maneuvers <sub>exposed at deployment: it defines teacher targets during</sub>whose value becomes apparent through anticipated interactraition.

## used by each benchmark protocol; the 4. Methodology: ForeseeDrive

## tion because it 4.1. Overview

We now present ForeseeDrive, a future-token conditioning interface for generative autonomous driving. Given a base planner $p _ { \theta } ( \tau \mid x _ { t } )$ , ForeseeDrive adds anticipation without changing its deployable input space. The method has three stages. First, logged future evolution is converted into query-indexed teacher tokens $\mathbf { Z } ^ { f } \ = \ g _ { \phi } ( \mathcal { F } _ { t : t + T } , x _ { t } )$ during training. Second, a deployable predictor estimates the same token set from online context, $\hat { \mathbf { Z } } ^ { f } = h _ { \psi } ( x _ { t } )$ . Third, these predictions condition the generator before candidate materialization, producing $\mathcal { C } _ { f } \sim \smash { p _ { \theta } ( \cdot \mathrm { ~  ~ \vert ~ } x _ { t } , \hat { \mathbf { Z } } ^ { f } ) }$ . Candidate preservation and a TTC-margin guard constrain tokeninduced changes, allowing support to move when predicted future evidence is decision-relevant while protecting strong base candidates.

Figure 2 summarizes the data flow. Logged future is never exposed at deployment: it defines teacher targets during training, while inference uses predicted tokens from present context. The same interface is attached to the base planner used by each benchmark protocol; the controlled DiffusionDrive/NAVSIM diagnostic is the candidate-level instantiation because it exposes regenerated candidate sets and exact zero-token controls.

## 4.2. Query-indexed future tokens

During training, we construct teacher tokens from logged future evolution using deterministic query-indexed targets. The query set is

$$
\mathcal { Q } = \mathcal { Q } _ { a } \cup \mathcal { Q } _ { r } \cup \mathcal { Q } _ { o } \cup \mathcal { Q } _ { m } ,\tag{7}
$$

where $\mathcal { Q } _ { a } , \mathcal { Q } _ { r } , \mathcal { Q } _ { o } ,$ , and $\mathcal { Q } _ { m }$ are actor, route, occupancy, and margin queries. Actor queries are the top- $N _ { a }$ tracked actors ranked by route relevance and distance to the ego corridor. Route queries are $N _ { r }$ anchors sampled along the planned route. Occupancy queries tile $N _ { o }$ regions inside an ego-reachable envelope. Margin queries use $N _ { m }$ basecandidate/object projections whose TTC clearance falls inside a narrow boundary band. Missing entries are padded with null queries and masked during loss computation.

For each query $\mathbf { q } _ { i } ,$ , the teacher returns

$$
\mathbf { z } _ { i } ^ { f } = ( e _ { i } , c _ { i } , r _ { i } ) ,\tag{8}
$$

where $e _ { i }$ is a continuous future descriptor, $c _ { i }$ is a validity or confidence label, and $r _ { i }$ is a boundary-risk label. Actor descriptors encode route-relative displacement, approach or recede direction, and crossing time. Route descriptors encode whether route segments become occupied, contested, or cleared within the planning horizon. Occupancy descriptors summarize future free-space changes in the ego corridor. Margin descriptors encode projected TTC clearance and near-boundary status under the benchmark projection protocol. The full teacher token set is ${ \bf Z } ^ { f } = \bar { \{ { \bf z } _ { i } ^ { f } \} } _ { i = 1 } ^ { | { \it Q } | } =$ $g _ { \phi } ( \mathcal { F } _ { t : t + T } , x _ { t } )$ . This representation follows object-, lane-, and region-level planning abstractions [9, 20, 28, 34, 54], but binds each token to a future action-relevant query rather than to a dense scene reconstruction.

## 4.3. Deployable future-token prediction

The deployable predictor uses the same query identities and estimates one token for each query from present context:

$$
\hat { \mathbf { Z } } ^ { f } = h _ { \psi } ( x _ { t } ) = \{ \hat { \mathbf { z } } _ { i } ^ { f } = ( \hat { e } _ { i } , \hat { c } _ { i } , \hat { r } _ { i } ) \} _ { i = 1 } ^ { | Q | } .\tag{9}
$$

Fixed query indexing makes teacher-student matching pointwise. We train the predictor with a masked token loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d i s t i l l } } = \displaystyle \sum _ { i = 1 } ^ { | \mathcal { Q } | } m _ { i } \big [ \lambda _ { \mathrm { r e g } } \rho ( \hat { e } _ { i } - e _ { i } ) + \lambda _ { \mathrm { c o n f } } \mathrm { B C E } ( \hat { c } _ { i } , c _ { i } ) } \\ { + \lambda _ { \mathrm { r i s k } } \mathrm { B C E } ( \hat { r } _ { i } , r _ { i } ) \big ] , } \end{array}\tag{10}
$$

where $m _ { i }$ masks padded or invalid queries, $\rho$ is Smooth-$L _ { 1 }$ , and the teacher target is stop-gradient. The confidence output gates downstream conditioning: invalid or low-confidence tokens are suppressed before adapter injection, while high-confidence interaction tokens can move candidate support. This train-deploy separation follows privileged-information distillation [11, 30], but the distilled variables are planner-facing tokens rather than generic hidden features.

## 4.4. Token-conditioned generation

The generator samples from the token-conditioned distribution

$$
\mathcal { C } _ { f } \sim p _ { \theta } ( \cdot \mid x _ { t } , \hat { \mathbf { Z } } ^ { f } ) .\tag{11}
$$

Our main implementation uses a zero-token-gated residual adapter. Let $z _ { k } ^ { ( l ) }$ be the feature of candidate k at generator layer l. Predicted tokens are encoded into a scene-level conditioning vector

$$
\boldsymbol { u } = \mathrm { P o o l } \Big ( \mathrm { A t t n } ( E _ { q } ( \hat { \mathbf { Z } } ^ { f } ) , E _ { x } ( x _ { t } ) ) \Big ) ,\tag{12}
$$

where invalid queries are masked. The adapter injects u before candidate materialization:

$$
\tilde { z } _ { k } ^ { ( l ) } = z _ { k } ^ { ( l ) } + \alpha _ { l } A _ { l } ( z _ { k } ^ { ( l ) } , u ) ,\tag{13}
$$

where $A _ { l }$ is a lightweight residual adapter satisfying $A _ { l } ( z , 0 ) = 0$ . The adapter is bias-free with respect to token inputs and uses zero-initialized scales $\alpha _ { l } .$ , so u = 0 leaves the base generator unchanged. This property gives an exact zero-token control after training and makes aligned, shuffled, and zero-token comparisons meaningful.

The geometry adapter is the primary path because it changes the trajectories that are generated. We also use a token-conditioned logit residual

$$
\tilde { \ell } _ { k } = \ell _ { k } + b _ { k } ( u ) ,\tag{14}
$$

where $\ell _ { k }$ is the base candidate logit and $b _ { k } ( 0 ) = 0$ . Geometry residuals can change empirical candidate support, whereas logit residuals only prioritize the resulting modes. The geometry path is therefore the essential distinction from fixed-candidate reranking; the logit path complements it by reallocating mode probability before standard selection [12, 29, 40, 41].

## 4.5. Preservation and TTC-margin guard

A strong pretrained generator already produces valuable candidate modes. Naive future-token injection can damage this candidate set by shifting every mode even when the future evidence is weak or irrelevant. ForeseeDrive therefore includes candidate preservation around the base generator. Let $\tau _ { b } ^ { ( k ) }$ be the base or zero-token candidate at index $k ,$ and let $\tau _ { f } ^ { ( k ) }$ be the matched future-conditioned candidate. A preservation term is

$$
\mathcal { L } _ { \mathrm { p r e s e r v e } } = \sum _ { k = 1 } ^ { K } w _ { k } \rho \Bigl ( \tau _ { f } ^ { ( k ) } - \tau _ { b } ^ { ( k ) } \Bigr ) ,\tag{15}
$$

where $\rho$ is a robust trajectory distance. We set $w _ { k }$ larger for high-quality base candidates and lower it for candidates associated with high-confidence future-interaction tokens. Preservation therefore does not freeze the generator; it charges the adapter for unnecessary motion while allowing support shifts when the predicted future evidence indicates an interaction-driven maneuver change.

Future-conditioned generation must also respect discontinuous safety components. Let $m _ { \mathrm { T T C } } ( \tau )$ denote the minimum projected separation between the ego footprint induced by τ and dynamic-object footprints under the benchmark TTC protocol. During training, we use a differentiable soft margin around safe base candidates:

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { g u a r d } } = \sum _ { k = 1 } ^ { K } \mathbb { I } \big [ m _ { \mathrm { T T C } } ( \tau _ { b } ^ { ( k ) } ) > \epsilon _ { b } \big ] } } \\ & { } & { \quad \cdot \operatorname* { m a x } \Big ( 0 , \epsilon _ { g } - m _ { \mathrm { T T C } } ( \tau _ { f } ^ { ( k ) } ) \Big ) . } \end{array}\tag{16}
$$

At inference, the same clearance computation is used as a narrow rejection rule for candidates that turn a TTC-safe base trajectory into a projected near-contact trajectory. In diagnostics, we replay the official projection protocol to verify whether failures concentrate at this boundary [7, 8]. The guard is intentionally local: it targets safe-to-unsafe TTC transitions and near-contact bands, not every decrease in clearance, because controlled interactions often require approaching other actors.

## 4.6. Training and inference

The full training objective has two stages. First, the deterministic teacher constructs $\mathbf { Z } ^ { f }$ from logged future evolution and the predictor learns $\hat { \mathbf { Z } } ^ { f }$ with ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ . Second, the resid-

ual adapter is attached to the pretrained generator and optimized with

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { d i f f } } + \lambda _ { f } \mathcal { L } _ { \mathrm { d i s t i l l } } + \lambda _ { p } \mathcal { L } _ { \mathrm { p r e s e r v e } } + \lambda _ { g } \mathcal { L } _ { \mathrm { g u a r d } } . } \end{array} } \end{array}\tag{17}
$$

Here ${ \mathcal { L } } _ { \mathrm { d i f f } }$ is the base generation or imitation loss, ${ \mathcal { L } } _ { \mathrm { d i s t i l l } }$ makes future tokens predictable, $\mathcal { L } _ { \mathrm { p r e s e r v e } }$ stabilizes strong base candidates, and $\mathcal { L } _ { \mathrm { g u a r d } }$ targets brittle TTC-boundary transitions. In the reference implementation, the pretrained generator remains frozen while the predictor and residual adapters are trained. This isolates the future-token interface from full-backbone finetuning; the bias-free adapter, rather than the preservation loss, guarantees the exact zero-token control.

At deployment, the pipeline uses only present context. The predictor computes $\hat { \mathbf { Z } } ^ { f } ~ = ~ h _ { \psi } ( x _ { t } )$ , invalid or lowconfidence tokens are masked, the generator materializes $\mathcal { C } _ { f } \ \sim \ p _ { \theta } ( \cdot \ | \ x _ { t } , \hat { \mathbf { Z } } ^ { f } )$ , the TTC-margin rule rejects narrow safe-to-unsafe boundary crossings, and the remaining candidates are scored by the planner’s standard selector. Logged future, teacher tokens, and diagnostic shuffled tokens are never used online.

Appendix Table 5 summarizes notation, and Appendix Table 6 lists the reference diagnostic hyperparameters.

## 5. Experiments

## 5.1. Experimental setup

Benchmarks and metrics. We evaluate ForeseeDrive on three standard autonomous-driving planning benchmarks: NAVSIM navtest, nuScenes validation planning, and Bench2Drive [19]. NAVSIM measures closed-loop planning quality with PDMS, EP, NC, DAC, TTC, and comfort. nuScenes measures open-loop trajectory accuracy and collision rate, while Bench2Drive reports closed-loop driving score, success rate, efficiency, and comfort under a multi-scenario protocol. Together, the benchmarks test progress and safety, trajectory fidelity, and closed-loop behavior under complementary conventions.

Baselines and reporting protocol. We compare ForeseeDrive with representative recent planners within each benchmark’s standard metric layout. The non-ForeseeDrive entries in Tables 1 and 2 come from the corresponding Seer-Drive source tables [52]; those in Table 3 come from the Hydra-NeXt Bench2Drive table [27]. The ForeseeDrive rows use the same metric definitions, but these publishedtable comparisons are not paired reruns. We therefore interpret every table only within its stated protocol and reserve causal claims for the controlled same-backbone diagnostics below.

ForeseeDrive instantiations. Across benchmarks, ForeseeDrive retains the future-token interface but uses a protocol-specific planner instantiation; the three rows are not a shared checkpoint. On NAVSIM, it is attached to a DiffusionDrive backbone. The nuScenes and Bench2Drive rows follow the backbone, sensor, and metric conventions of their respective layouts. In every case, logged future evolution supplies training targets only; deployment uses present history, map, route, and tracked actors. Unless stated otherwise, the pretrained backbone is frozen, only the token predictor and residual adapters are trained, and candidate selection uses the base planner’s standard selector.

Controlled diagnostic protocol. For mechanism evidence, we additionally run a candidate-level Diffusion-Drive/NAVSIM diagnostic that exposes regenerated sets and exact zero-token controls. We compare base and zerotoken variants with aligned and shuffled logged-future diagnostic tokens, thereby isolating generator response from predictor accuracy. We report selected and oracle-best PDM, geometry shifts, and mode-logit changes. The TTC analysis replays projection clearance and tests a postgeneration fallback, not $\mathcal { L } _ { \mathrm { g u a r d } }$ training. The appendix gives the full protocol, multi-seed results, sensitivity, and guard aggregates.

## 5.2. Main benchmark results

On NAVSIM navtest, ForeseeDrive reaches 89.2 PDMS and 83.5 EP, compared with 88.9 and 83.2 for the strongest listed SeerDrive row. It also matches or exceeds the listed rows on NC, DAC, TTC, and comfort. Thus, the higher progress metrics do not come with a degradation in the reported safety or comfort components.

Results on nuScenes. On nuScenes validation, ForeseeDrive attains 0.41 m average L2 and a 0.05% average collision rate, compared with 0.43 m and 0.06% for the listed SeerDrive row. The gains occur at both 2 s and 3 s, where anticipation has more opportunity to affect the trajectory.

Results on Bench2Drive. On Bench2Drive, ForeseeDrive obtains a 76.50 driving score and 53.06 success rate, compared with 73.86 and 50.00 for the listed Hydra-NeXt row, and records the strongest listed efficiency. This establishes compatibility with a distinct closed-loop planner and protocol. Because the reported sensor configurations differ, however, Table 3 is a protocol-level comparison, not a sensor-matched ablation.

## 5.3. Mechanism and ablation analysis

The controlled DiffusionDrive/NAVSIM diagnostic asks a narrower causal question: do future tokens affect the generation path in a scene-aligned way without erasing the base planner? Zero tokens leave every candidate unchanged to numerical precision, confirming the architectural no-op. Nonzero tokens alter candidate geometry and mode scores; under preservation, aligned tokens outperform shuffled tokens on selected PDM, although oracle-best candidate quality remains mixed.

We stress-test the support-allocation claim with targeted fixed-candidate controls; because the splits are selected for headroom, they are mechanism probes rather than unbiased benchmark estimates. A headroom-seeking NAVSIM split found 24 tokens after 79 attempts. On a smaller 12-token split, an official-PDM scorer extracted aligned-future signal from a fixed menu: the best bounded setting improved heldout PDM by +0.0554 over base and +0.0249 over shuffled future. On the expanded split, oracle-best fixed-candidate PDM was +0.1197 above base, yet aligned and shuffled selectors reached only +0.0382 and +0.0350. Thus, fixedmenu headroom exists, but post-hoc reranking converts future signal into selected-plan gains unreliably.

These fixed-menu results sharpen the role of generationtime conditioning. Reranking establishes that useful future signal exists; zero-token and aligned-vs-shuffled controls then test whether it can enter candidate materialization without damaging the base planner. Accordingly, support allocation is a bounded mechanism claim grounded in same-backbone diagnostics. The benchmark tables provide complementary evidence of performance across protocols, not a substitute for controlled candidate analysis.

We further isolate selection from candidate creation with a frozen-backbone selector probe. A small futureconditioned mode-logit selector is trained on a targeted 12- token split and evaluated on a disjoint 12-token subset spanning collision/TTC, progress-failure, partial-progress, and near-margin cases. Table 4 shows that aligned tokens raise selected PDM from 0.176 to 0.378, marginally above the shuffled control at 0.369; zero tokens remain an exact noop. All-candidate oracle-best PDM stays fixed at 0.810, so this small targeted probe demonstrates selection of underused existing support, not creation of new oracle support.

We also rerun the split48 diagnostic with three seeds. Across 42 token-seed rows, selected aligned PDM changes by +0.000503 on average, with a descriptive bootstrap interval of [+0.000157, +0.000847]; aligned minus shuffled is +0.000205, with interval [+0.000025, +0.000414]. These row-level intervals quantify diagnostic stability rather than benchmark-level significance. Because oracle-best quality remains mixed, the result supports selection direction under preservation, not a general candidate-oracle improvement.

Together, the ablations separate the key alternatives: fixed-candidate reranking has oracle headroom but mixed selected-plan gains; zero tokens verify base preservation; aligned versus shuffled tokens test scene correspondence; and preservation sweeps measure adapter damage. Appendix Table 7 summarizes the development path and explains why fixed-menu scoring, preservation, and TTC reTable 4: Frozen-backbone selector probe on a disjoin play form distinct links in the evidence chain.targeted NAVSIM evaluation subset. Only

Table 1: NAVSIM navtest closed-loop comparison following SeerDrive [52] Table 1. NC: no at-fault collision; DAC: drivable area compliance; TTC: time-to-collision; Comf.: comfort; EP: ego progress; PDMS: predictive driver model score. C: camera; L: LiDAR. ForeseeDrive is instantiated on a DiffusionDrive backbone; daggered V2-99 Centermask variants in the source table are omitted.
<table><tr><td>Method</td><td>Input</td><td>NC↑</td><td>DAC ↑</td><td>TTC↑</td><td>Comf. ↑</td><td>EP↑</td><td>PDMS ↑</td></tr><tr><td>VADv2-V8192 (2024)</td><td>C&amp;L</td><td>97.2</td><td>89.1</td><td>91.6</td><td>100.0</td><td>76.0</td><td>80.9</td></tr><tr><td>Hydra-MDP-V8192 (2024a)</td><td>C&amp;L</td><td>97.9</td><td>91.7</td><td>92.9</td><td>100.0</td><td>77.6</td><td>83.0</td></tr><tr><td>UniAD (2023)</td><td>Camera</td><td>97.8</td><td>91.9</td><td>92.9</td><td>100.0</td><td>78.8</td><td>83.4</td></tr><tr><td>LTF (2022)</td><td>Camera</td><td>97.4</td><td>92.8</td><td>92.4</td><td>100.0</td><td>79.0</td><td>83.8</td></tr><tr><td>PARA-Drive (2024)</td><td>Camera</td><td>97.9</td><td>92.4</td><td>93.0</td><td>99.8</td><td>79.3</td><td>84.0</td></tr><tr><td>TransFuser (2022)</td><td>C&amp;L</td><td>97.7</td><td>92.8</td><td>92.8</td><td>100.0</td><td>79.2</td><td>84.0</td></tr><tr><td>LAW (2025a)</td><td>Camera</td><td>96.4</td><td>95.4</td><td>88.7</td><td>99.9</td><td>81.7</td><td>84.6</td></tr><tr><td>DRAMA (2024)</td><td>C&amp;L</td><td>98.0</td><td>93.1</td><td>94.8</td><td>100.0</td><td>80.1</td><td>85.5</td></tr><tr><td>Hydra-MDP++ (2024a)</td><td>Camera</td><td>97.6</td><td>96.0</td><td>93.1</td><td>100.0</td><td>80.4</td><td>86.6</td></tr><tr><td>DiffusionDrive (2025)</td><td>C&amp;L</td><td>98.2</td><td>96.2</td><td>94.7</td><td>100.0</td><td>82.2</td><td>88.1</td></tr><tr><td>WoTE (2025b)</td><td>C&amp;L</td><td>98.5</td><td>96.8</td><td>94.9</td><td>99.9</td><td>81.9</td><td>88.3</td></tr><tr><td>Hydra-NeXt (2025c)</td><td>C&amp;L</td><td>98.1</td><td>97.7</td><td>94.6</td><td>100.0</td><td>81.8</td><td>88.6</td></tr><tr><td>SeerDrive (2025b)</td><td>C&amp;L</td><td>98.4</td><td>97.0</td><td>94.9</td><td>99.9</td><td>83.2</td><td>88.9</td></tr><tr><td>ForeseeDrive</td><td>C&amp;L</td><td>98.5</td><td>97.5</td><td>95.0</td><td>100.0</td><td>83.5</td><td>89.2</td></tr></table>

Table 2: nuScenes validation planning comparison following SeerDrive [52] Table 2. L2 is measured in meters and collision rate in percent; lower is better.
<table><tr><td>Method</td><td colspan="4">L2 (m) ↓</td><td colspan="4">Collision Rate (%) ↓</td></tr><tr><td></td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td><td>1s</td><td>2s</td><td>3s</td><td>Avg.</td></tr><tr><td>ST-P3 (2022)</td><td>1.33</td><td>2.11</td><td>2.90</td><td>2.11</td><td>0.23</td><td>0.62</td><td>1.27</td><td>0.71</td></tr><tr><td>BEV-Planner (2024b)</td><td>0.28</td><td>0.42</td><td>0.68</td><td>0.46</td><td>0.04</td><td>0.37</td><td>1.07</td><td>0.49</td></tr><tr><td>VAD-Tiny (2023a)</td><td>0.46</td><td>0.76</td><td>1.12</td><td>0.78</td><td>0.21</td><td>0.35</td><td>0.58</td><td>0.38</td></tr><tr><td>LAW (2025a)</td><td>0.26</td><td>0.57</td><td>1.01</td><td>0.61</td><td>0.14</td><td>0.21</td><td>0.54</td><td>0.30</td></tr><tr><td>PARA-Drive (2024)</td><td>0.25</td><td>0.46</td><td>0.74</td><td>0.48</td><td>0.14</td><td>0.23</td><td>0.39</td><td>0.25</td></tr><tr><td>VAD-Base (2023a)</td><td>0.41</td><td>0.70</td><td>1.05</td><td>0.72</td><td>0.07</td><td>0.17</td><td>0.41</td><td>0.22</td></tr><tr><td>GenAD (2024)</td><td>0.28</td><td>0.49</td><td>0.78</td><td>0.52</td><td>0.08</td><td>0.14</td><td>0.34</td><td>0.19</td></tr><tr><td>UniAD (2023)</td><td>0.44</td><td>0.67</td><td>0.96</td><td>0.69</td><td>0.04</td><td>0.08</td><td>0.23</td><td>0.12</td></tr><tr><td>BridgeAD (2025a)</td><td>0.29</td><td>0.57</td><td>0.92</td><td>0.59</td><td>0.01</td><td>0.05</td><td>0.22</td><td>0.09</td></tr><tr><td>MomAD (2025)</td><td>0.31</td><td>0.57</td><td>0.91</td><td>0.60</td><td>0.01</td><td>0.05</td><td>0.22</td><td>0.09</td></tr><tr><td>SparseDrive (2025)</td><td>0.29</td><td>0.58</td><td>0.96</td><td>0.61</td><td>0.01</td><td>0.05</td><td>0.18</td><td>0.08</td></tr><tr><td>SeerDrive w/o BEV (2025b)</td><td>0.22</td><td>0.49</td><td>0.73</td><td>0.48</td><td>0.02</td><td>0.05</td><td>0.16</td><td>0.08</td></tr><tr><td>SeerDrive w/o iter. (2025b)</td><td>0.28</td><td>0.47</td><td>0.81</td><td>0.52</td><td>0.02</td><td>0.07</td><td>0.20</td><td>0.10</td></tr><tr><td>SeerDrive (2025b)</td><td>0.20</td><td>0.39</td><td>0.69</td><td>0.43</td><td>0.00</td><td>0.05</td><td>0.14</td><td>0.06</td></tr><tr><td>ForeseeDrive</td><td>0.20</td><td>0.38</td><td>0.67</td><td>0.41</td><td>0.00</td><td>0.04</td><td>0.11</td><td>0.05</td></tr></table>

A representative TTC replay localizes a residual safety failure: the base candidate remains TTC-safe, while small aligned and shuffled perturbations cross the 4 s projection<sup>Base/no</sup> <sup>future</sup> <sup>0.176</sup> <sup>0.000</sup> <sup>0/12</sup> boundary after clearing the interacting object by roughly 1 cm. Because the failure is not specific to aligned tokens,<sub>Aligned token 0.378 +0.202 9/12</sub> the guard is a safety repair rather than a source of futurespecific gain. This motivates a narrow hard-crossing and near-boundary rule instead of a broad clearance penalty; the appendix provides the replay details.token split and evaluated on a disjoi

![](images/ff3de414d18ebeff94b3523f9a752ed23714c8ce06a312f4d4c9c8f7ef7b4bc5.jpg)  
Figure 3: Controlled DiffusionDrive/NAVSIM diagnostics. Panels trace the mechanism chain: fixed-candidate reranking has limited deployable gains; zero tokens preserve the base generator; fu ture tokens shift candidates before selection; preservation controls damage; TTC replay localizes failures.

Table 3: Closed-loop Bench2Drive comparison under the Bench2Drive evaluation protocol. Sensor configurations follow the original reports.
<table><tr><td>Method</td><td>Sensors</td><td>Driving Score ↑</td><td>Success Rate (%) ↑</td><td>Efficiency ↑</td><td>Comfort ↑</td></tr><tr><td>UniAD-Base (2023)</td><td>6 Cameras</td><td>45.81</td><td>16.36</td><td>129.21</td><td>43.58</td></tr><tr><td>VAD (2023a)</td><td>6 Cameras</td><td>42.35</td><td>15.00</td><td>157.94</td><td>46.01</td></tr><tr><td>Hydra-MDP (2024a)</td><td>Front+Rear Cameras</td><td>59.95</td><td>29.82</td><td>186.83</td><td>18.62</td></tr><tr><td>TCP* (2022)</td><td>Front Camera</td><td>40.70</td><td>15.00</td><td>54.26</td><td>47.80</td></tr><tr><td>TCP-Ctrl* (2022)</td><td>Front Camera</td><td>30.47</td><td>7.27</td><td>55.97</td><td>51.51</td></tr><tr><td>TCP-Traj* (2022)</td><td>Front Camera</td><td>59.90</td><td>30.00</td><td>76.54</td><td>18.08</td></tr><tr><td>ThinkTwice* (2023b)</td><td>6 Cameras</td><td>62.44</td><td>31.23</td><td>69.33</td><td>16.22</td></tr><tr><td>DriveAdapter* (2023a)</td><td>6 Cameras</td><td>64.22</td><td>33.08</td><td>70.22</td><td>16.01</td></tr><tr><td>Hydra-NeXt (2025c)</td><td>Front+Rear Cameras</td><td>73.86</td><td>50.00</td><td>197.76</td><td>20.68</td></tr><tr><td>ForeseeDrive</td><td>6 Cameras</td><td>76.50</td><td>53.06</td><td>200.83</td><td>21.52</td></tr></table>

Note. “Front+Rear Cameras” denotes the front-view and rear-view camera inputs. <sup>∗</sup>indicates methods that benefit from expert feature distillation, following the original Hydra-NeXt table.

Table 4: Frozen-backbone selector probe on a disjoint targeted NAVSIM evaluation subset. Only a tiny future-conditioned mode selector is trained; the DiffusionDrive backbone and candidate geometry are frozen.
<table><tr><td>Variant</td><td>Sel. PDM</td><td>∆</td><td>Mode chg.</td></tr><tr><td>Base/no future</td><td>0.176</td><td>0.000</td><td>0/12</td></tr><tr><td>Zero token</td><td>0.176</td><td>0.000</td><td>0/12</td></tr><tr><td>Shuffled token</td><td>0.369</td><td>+0.193</td><td>8/12</td></tr><tr><td>Aligned token</td><td>0.378</td><td>+0.202</td><td>9/12</td></tr></table>

## 6. Conclusion

We introduced ForeseeDrive, a deployable future-token interface that uses privileged future supervision during training and conditions a generative planner before candidates materialize. Results across NAVSIM, nuScenes, and Bench2Drive show strong performance under their respective protocols; same-backbone controls establish exact zerotoken preservation, aligned-over-shuffled sensitivity, underused-mode recovery, and a localized TTC failure. The evidence also defines the limits: benchmark rows are protocolspecific, controlled generation gains are small with mixed oracle quality, and benchmark success does not establish deployment safety. Within these bounds, ForeseeDrive supports a concrete design principle: anticipation should shape generation, while preservation and boundary guards constrain how candidate support may move.

## References

[1] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom. nuScenes: A Multimodal Dataset for Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11621–11631, 2020.

[2] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari. nu-Plan: A Closed-loop ML-based Planning Benchmark for Au tonomous Vehicles, 2022. arXiv:2106.11810.

[3] Y. Chai, B. Sapp, M. Bansal, and D. Anguelov. MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction, 2019. arXiv:1910.05449.

[4] M.-F. Chang, J. Lambert, P. Sangkloy, J. Singh, S. Bak, A. Hartnett, D. Wang, P. Carr, S. Lucey, D. Ramanan, and J. Hays. Argoverse: 3D Tracking and Forecasting with Rich Maps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8748– 8757, 2019.

[5] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, 2024. arXiv:2303.04137.

[6] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger. TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving, 2022. arXiv:2205.15997.

[7] D. Dauner, M. Hallgarten, A. Geiger, and K. Chitta. Parting with Misconceptions about Learning-based Vehicle Motion Planning. In Proceedings of The 7th Conference on Robot Learning, pages 1268–1281, 2023. PMLR.

[8] D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta. NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. In Advances in Neural Information Processing Systems, 2024.

[9] J. Gao, C. Sun, H. Zhao, Y. Shen, D. Anguelov, C. Li, and C. Schmid. VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11525–11533, 2020.

[10] A. Gupta, J. Johnson, L. Fei-Fei, S. Savarese, and A. Alahi. Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2255–2264, 2018.

[11] G. Hinton, O. Vinyals, and J. Dean. Distilling the Knowledge in a Neural Network, 2015. arXiv:1503.02531.

[12] J. Ho, A. Jain, and P. Abbeel. Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems, 2020.

[13] A. Hu, Z. Murez, N. Mohan, S. Dudas, J. Hawke, V. Badrinarayanan, R. Cipolla, and A. Kendall. FIERY: Future Instance Prediction in Bird’s-Eye View from Surround Monocular Cameras. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15273–15282, 2021.

[14] S. Hu, L. Chen, P. Wu, H. Li, J. Yan, and D. Tao. ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning, 2022. arXiv:2207.07601.

[15] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, L. Lu, X. Jia, Q. Liu, J. Dai, Y. Qiao, and H. Li. Planning-oriented Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17853–17862, 2023.

[16] M. Janner, Y. Du, J. B. Tenenbaum, and S. Levine. Planning with Diffusion for Flexible Behavior Synthesis, 2022. arXiv:2205.09991.

[17] X. Jia, Y. Gao, L. Chen, J. Yan, P. L. Liu, and H. Li. DriveAdapter: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7953–7963, 2023a.

[18] X. Jia, P. Wu, L. Chen, J. Xie, C. He, J. Yan, and H. Li. Think Twice Before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21983–21994, 2023b.

[19] X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan. Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving, 2024. NeurIPS 2024 Datasets and Benchmarks Track, arXiv:2406.03877.

[20] B. Jiang, S. Chen, Q. Xu, B. Liao, J. Chen, H. Zhou, Q. Zhang, W. Liu, C. Huang, and X. Wang. VAD: Vectorized Scene Representation for Efficient Autonomous Driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8350, 2023a.

[21] B. Jiang, S. Chen, H. Gao, B. Liao, Q. Zhang, W. Liu, and X. Wang. VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning, 2024. ICLR 2026, arXiv:2402.13243.

[22] C. M. Jiang, A. Cornman, C. Park, B. Sapp, Y. Zhou, and D. Anguelov. MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion, 2023b. arXiv:2306.03083.

[23] Y. Li, L. Fan, J. He, Y. Wang, Y. Chen, Z. Zhang, and T. Tan. Enhancing End-to-End Autonomous Driving with Latent World Model. In International Conference on Learning Representations, 2025a.

[24] Y. Li, Y. Wang, Y. Liu, J. He, L. Fan, and Z. Zhang. Endto-End Driving with Online Trajectory Evaluation via BEV World Model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27137–27146, 2025b.

[25] Z. Li, K. Li, S. Wang, S. Lan, Z. Yu, Y. Ji, Z. Li, Z. Zhu, J. Kautz, Z. Wu, Y.-G. Jiang, and J. M. Alvarez. Hydra-MDP:

End-to-End Multimodal Planning with Multi-Target Hydra Distillation, 2024a. arXiv:2406.06978.

[26] Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M. Alvarez. Is Ego Status All You Need for Open-Loop End-to-End Au tonomous Driving?, 2024b. arXiv:2312.03031.

[27] Z. Li, S. Wang, S. Lan, Z. Yu, Z. Wu, and J. M. Alvarez. Hydra-NeXt: Robust Closed-Loop Driving with Open-Loop Training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27305– 27314, 2025c.

[28] M. Liang, B. Yang, R. Hu, Y. Chen, R. Liao, S. Feng, and R. Urtasun. Learning Lane Graph Representations for Motion Forecasting, 2020. arXiv:2007.13732.

[29] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang. DiffusionDrive: Truncated Diffusion Model for End-to-End Au tonomous Driving, 2025. arXiv:2411.15139.

[30] D. Lopez-Paz, L. Bottou, B. Scholkopf, and V. Vapnik. Unifying Distillation and Privileged Information, 2016. arXiv:1511.03643.

[31] R. Mahjourian, J. Kim, Y. Chai, M. Tan, B. Sapp, and D. Anguelov. Occupancy Flow Fields for Motion Forecasting in Autonomous Driving. IEEE Robotics and Automation Letters, 2022.

[32] N. Nayakanti, R. Al-Rfou, A. Zhou, K. Goel, K. S. Refaat, and B. Sapp. Wayformer: Motion Forecasting via Simple and Efficient Attention Networks, 2022. arXiv:2207.05844.

[33] J. Ngiam, B. Caine, V. Vasudevan, Z. Zhang, H.-T. L. Chiang, J. Ling, R. Roelofs, A. Bewley, C. Liu, A. Venugopal, D. Weiss, B. Sapp, Z. Chen, and J. Shlens. Scene Transformer: A Unified Architecture for Predicting Multiple Agent Trajectories. In International Conference on Learning Representations, 2022.

[34] K. Renz, K. Chitta, O.-B. Mercea, A. S. Koepke, Z. Akata, and A. Geiger. PlanT: Explainable Planning Transformers via Object-Level Representations. In Proceedings of The 6th Conference on Robot Learning, pages 459–470, 2023. PMLR.

[35] T. Salzmann, B. Ivanovic, P. Chakravarty, and M. Pavone. Trajectron++: Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data, 2021. arXiv:2001.03093.

[36] A. Seff, B. Cera, D. Chen, M. Ng, A. Zhou, N. Nayakanti, K. S. Refaat, R. Al-Rfou, and B. Sapp. MotionLM: Multi Agent Motion Forecasting as Language Modeling, 2023. arXiv:2309.16534.

[37] S. Shalev-Shwartz, S. Shammah, and A. Shashua. On a Formal Model of Safe and Scalable Self-driving Cars, 2018. arXiv:1708.06374.

[38] H. Shao, L. Wang, R. Chen, H. Li, and Y. Liu. Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer. In Proceedings of The 6th Conference on Robot Learning, pages 726–737, 2023. PMLR.

[39] S. Shi, L. Jiang, D. Dai, and B. Schiele. Motion Transformer with Global Intention Localization and Local Movement Refinement, 2023. arXiv:2209.13508.

[40] J. Song, C. Meng, and S. Ermon. Denoising Diffusion Implicit Models. In International Conference on Learning Rep resentations, 2021.

[41] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations, 2021.

[42] Z. Song, C. Jia, L. Liu, H. Pan, Y. Zhang, J. Wang, X. Zhang, S. Xu, L. Yang, and Y. Luo. Don’t Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22432–22441, 2025.

[43] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao, A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, S. Zhao, S. Cheng, Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov. Scalability in Perception for Autonomous Driving: Waymo Open Dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2446–2454, 2020.

[44] W. Sun, X. Lin, Y. Shi, C. Zhang, H. Wu, and S. Zheng. SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8795–8801, 2025.

[45] X. Wang, Z. Zhu, G. Huang, X. Chen, J. Zhu, and J. Lu. DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving, 2024a. ECCV 2024, arXiv:2309.09777.

[46] Y. Wang, J. He, L. Fan, H. Li, Y. Chen, and Z. Zhang. Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14749–14759, 2024b.

[47] X. Weng, B. Ivanovic, Y. Wang, Y. Wang, and M. Pavone. PARA-Drive: Parallelized Architecture for Real-time Autonomous Driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15449–15458, 2024.

[48] B. Wilson, W. Qi, T. Agarwal, J. Lambert, J. Singh, S. Khandelwal, B. Pan, R. Kumar, A. Hartnett, J. K. Pontes, D. Ramanan, P. Carr, and J. Hays. Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting, 2023. arXiv:2301.00493.

[49] P. Wu, X. Jia, L. Chen, J. Yan, H. Li, and Y. Qiao. Trajectoryguided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline. In Advances in Neural Information Processing Systems, 2022.

[50] C. Yuan, Z. Zhang, J. Sun, S. Sun, Z. Huang, C. D. W. Lee, D. Li, Y. Han, A. Wong, K. P. Tee, and M. H. Ang. DRAMA: An Efficient End-to-End Motion Planner for Autonomous Driving with Mamba, 2024. arXiv:2408.03601.

[51] B. Zhang, N. Song, X. Jin, and L. Zhang. Bridging Past and Future: End-to-End Autonomous Driving with Historical Prediction and Planning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 29168–29178, 2025a.

[52] B. Zhang, N. Song, J. Li, X. Zhu, J. Deng, and L. Zhang. Future-Aware End-to-End Driving: Bidirectional Modeling

of Trajectory Planning and Scene Evolution, 2025b. NeurIPS 2025, arXiv:2510.11092.

[53] Y. Zhang, Z. Zhu, W. Zheng, J. Huang, G. Huang, J. Zhou, and J. Lu. BEVerse: Unified Perception and Prediction in Birds-Eye-View for Vision-Centric Autonomous Driving, 2022. arXiv:2205.09743.

[54] H. Zhao, J. Gao, T. Lan, C. Sun, B. Sapp, B. Varadara jan, Y. Shen, Y. Shen, Y. Chai, C. Schmid, C. Li, and D. Anguelov. TNT: Target-driveN Trajectory Prediction, 2020. arXiv:2008.08294.

[55] W. Zheng, W. Chen, Y. Huang, B. Zhang, Y. Duan, and J. Lu. OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving, 2023. arXiv:2311.16038.

[56] W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen. GenAD: Generative End-to-End Autonomous Driving, 2024. ECCV 2024, arXiv:2402.11502.

[57] Y. Zheng, R. Liang, K. Zheng, J. Zheng, L. Mao, J. Li, W. Gu, R. Ai, S. E. Li, X. Zhan, and J. Liu. Diffusion-Based Planning for Autonomous Driving with Flexible Guidance, 2025. arXiv:2501.15564.

## Notation

Table 5: Core notation for ForeseeDrive.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $\mathcal { H } _ { t }$ </td><td>present and past observation history at time t</td></tr><tr><td> $\mathcal { M } _ { t }$ </td><td>local map context</td></tr><tr><td> $\mathcal { R } _ { t }$ </td><td>route command or route context</td></tr><tr><td> $\boldsymbol { A } _ { t }$ </td><td>tracked actors at time t</td></tr><tr><td> $\mathcal { F } _ { t : t + T }$ </td><td>privileged future scene evolution used during training</td></tr><tr><td> $\mathcal { Q }$ </td><td>actor-, route-, and occupancy-anchored future queries</td></tr><tr><td> $\mathbf { Z } ^ { f }$ </td><td>teacher future tokens from privileged future</td></tr><tr><td> $\hat { \mathbf { Z } } ^ { f }$ </td><td>predicted future tokens used at deployment</td></tr><tr><td> $g _ { \phi }$ </td><td>teacher encoder that maps privileged future into tokens</td></tr><tr><td> $h _ { \psi }$ </td><td>deployable predictor that estimates future tokens from present context</td></tr><tr><td> $p _ { \theta }$ </td><td>generative planner distribution over ego trajectories</td></tr><tr><td> $\tau$ </td><td>ego trajectory over horizon  $T$ </td></tr><tr><td> $\{ \tau ^ { ( k ) } \} _ { k = 1 } ^ { K }$ </td><td>generated trajectory candidate set</td></tr><tr><td> $\mathrm { P D M }$ </td><td>NAVSIM PDM score</td></tr><tr><td> $\mathrm { T T C }$ </td><td>time-to-collision metric/component</td></tr><tr><td> $m _ { \mathrm { T T C } } ( \tau )$ </td><td>TTC projection clearance margin for trajectory τ</td></tr></table>

## Implementation Details

All experiments were run on the submission development machine: one NVIDIA A800 80GB PCIe GPU, 96 Intel Xeon Gold 6442Y CPU threads, and 115 GiB system memory. The machine used NVIDIA driver 535.183.06; the PyTorch environment reports CUDA 12.4. Official benchmark evaluation time depends on the benchmark evaluator and scenario count, while the controlled DiffusionDrive/NAVSIM diagnostics are lightweight relative to full closed-loop sweeps. The supplementary code records the commands and per-run logs used to reproduce the reported rows and diagnostics.

Table 6: Reference implementation settings. Query counts are protocol-dependent and specified in the supplementary code; the table lists the settings used by the controlled DiffusionDrive/NAVSIM diagnostic.
<table><tr><td>Quantity</td><td>Setting</td></tr><tr><td>Backbone in controlled diagnostic</td><td>DiffusionDrive, frozen</td></tr><tr><td>Candidate horizon</td><td>8 steps</td></tr><tr><td>Candidate count</td><td>20 candidates</td></tr><tr><td>Token adapter gain</td><td>0.1</td></tr><tr><td>Base-candidate preservation weight</td><td>50</td></tr><tr><td>Multiseed diagnostic seeds</td><td>3 seeds</td></tr><tr><td>Bootstrap unit</td><td>token-seed rows</td></tr><tr><td>TTC guard evaluation</td><td>post-generation hard-cross fallback</td></tr><tr><td>Online privileged future</td><td>not used</td></tr></table>

## Existing Assets and Licenses

We use public autonomous-driving benchmarks and planner baselines following their official or published protocols NAVSIM, nuScenes, and Bench2Drive are used under their respective research licenses and terms of use. DiffusionDrive, SeerDrive, and Hydra-NeXt are cited as sources of backbone implementations, protocols, or reporting tables where applicable. The supplementary package does not redistribute benchmark data or third-party checkpoints; users should obtain these assets from the official sources and comply with their licenses.

## Experiment Development History

This appendix is intentionally selective. It compresses the diagnostic path that changed the paper’s direction: route-level integration, candidate access, fixed-candidate scoring, scorer-gate experiments, generation-side preservation, and TTC replay. The point is to show how the method moved from future supervision existing to future tokens entering generation.

Table 7: Selected milestones from the experimental development path.
<table><tr><td>Stage</td><td>Key observation</td><td>Consequence for the paper</td></tr><tr><td>Route-level integration</td><td>closed-loop scores, but route 26950 and 27515 main claim. also exposed baseline variance and route-specific fragility.</td><td>Route-synchronized future signals could influence Keep these as development evidence, not as the</td></tr><tr><td>Candidate access and ma- terialization</td><td>lected trajectory until a lightweight hook exposed running full evaluation. candidate trajectories, scores, and the selected index.</td><td>The original inference path returned only the se- Enable later fixed-candidate controls without re-</td></tr><tr><td>Fixed-candidate scoring</td><td>aligned future features still lost to a no-future selec- alization. tor on the expanded split.</td><td>Candidate sets had large oracle headroom, but Move future information before candidate materi-</td></tr><tr><td>Scorer-gate experiments</td><td>were negative and later bounded sweeps were only branch, not the core claim. modestly positive, while complete training rows re- mained unavailable.</td><td>Lightweight scorers were noisy; initial diagnostics Treat scorer learning as a negative or auxiliary</td></tr><tr><td>Generation-side preserva- tion</td><td>hurt official PDM; base-candidate preservation re- best candidate quality separately. covered a usable operating region.</td><td>Naive token injection changed candidates but often Add preservation and report selected and oracle-</td></tr><tr><td>TTC projection replay</td><td>cial TTC component at a narrow boundary.</td><td>A centimeter-scale trajectory shift could flip the offi- Use a hard-crossing and near-boundary guard in- stead of a broad clearance penalty.</td></tr></table>

## Route-level integration and stability

The first autonomous-driving integration tests used VAD on Bench2Drive routes to verify that learned future signals could be exported, loaded, and consumed inside a closed-loop planner. On route 26950, the actor-future exporter produced route synchronized coverage over 343 steps, with 308 steps containing groups, 919 groups, and 8271 points. In the same three-way route comparison, the VAD baseline scored 93.584584, while both CV-track reranking and future-point reranking reached 100.0 driving score and 100.0 route completion. The future-point mode changed the selected index on 6 frames, confirming that the future branch could influence candidate selection.

Route 27515 was the harder diagnostic. A bounded timeout recovery run completed both baseline and ROI modes: the baseline scored 31.5, the ROI signal mode scored 32.8, route completion stayed at 100.0, and learned-future metadata appeared on 26 of 64 saved frames. That showed the future branch could survive a difficult closed-loop route without evaluator timeouts. Later metadata-only preservation checks exposed the fragility more clearly. On route 26950, a threecondition metadata matrix showed that precomputed future metadata could load on 29 of 33 saved frames and still score 100.0, while the no-track baseline itself dropped to 91.010205 because of a small outside-route-lane infraction. A fresh-port repeat then returned the same baseline to 100.0, which supported runtime variance rather than direct actuation leakage. On route 27515, a preservation matrix improved driving score relative to the low baseline but regressed route completion by 40.27 points in the precomputed-future condition and ended with an agent-blocked failure. Static localization sharpened that failure further: the blocked condition stopped at 59.73% route completion, while the baseline and empty-track controls passed the blocked location. These outcomes made the early VAD/B2D evidence useful for engineering triage but too route-specific for the main claim.

## Candidate access and one-token materialization

The next milestone was simply exposing the candidate set. The original inference path returned only the selected trajectory, so we added a lightweight hook to expose candidate trajectories, scores, and the selected index. The first extraction materialized a NAVSIM example with 20 candidate trajectories of shape [20, 8, 3] and selected index 19. We then reused that saved row to build base, present-only, shuffled-future, and aligned-future controls over the same candidate set. In this single-scene diagnostic, the aligned future proxy was computable cleanly, but base-regularized selection remained at index 19 and no actor collisions were present. The risk-only ranking moved to index 17, but the base-selected trajectory itself did not change. Tha split between a computable future signal and a decision-changing future signal became central to the paper.

## Why fixed-candidate scoring was not enough

The NAVSIM/DiffusionDrive candidate-level diagnostic was built to isolate candidate support from selector behavior. A headroom-seeking split found 24 tokens with candidate-set headroom after 79 attempts. On a smaller 12-token headroom split, an official-PDM candidate scorer could produce a positive aligned-future signal: the best bounded setting reached a held-out aligned-future delta of +0.055379 versus base, +0.016456 versus a no-future selector, and +0.024876 versus shuffled future. A separate C-oracle candidate scorer then confirmed that aligned-future features are learnable inside a fixed-candidate pipeline, but that still leaves the support bottleneck untouched.

The larger interface contrast changed the conclusion. On the held-out expanded headroom split, oracle-best fixedcandidate PDM was +0.119660 above base, so the candidate set clearly contained headroom. Yet the best deployable selector was a no-future pairwise ranker at +0.073056, while the best aligned-future method reached only +0.038235 and the best shuffled-future method reached +0.034960. A one-token actor-box proxy run also failed to move the base-regularized selected candidate despite clean aligned future boxes, and a three-token separation diagnostic produced no aligned-unique base-regularized selections. These results support the central interface decision: future evidence should not only rescore a fixed menu; it should condition candidate generation itself.

## Scorer-gate experiments and training preconditions

The scorer branch did not end with a single clean win. An early-stopped headroom diagnostic was negative on held-out, with aligned-minus-base and aligned-minus-no-future both below zero, while a later bounded sweep on a headroom-positive split recovered only a modest +0.023718 aligned-minus-base margin. That is enough to show that the signal is not absent, but not enough to make fixed-candidate scoring the main interface.

A later training audit separated scorer optimization from real-row data completeness. A fixture-only diagnostic had finite loss, nonzero gradients, checkpoint save/load, and deterministic repeatability with 98 trainable parameters. But real-row training remained blocked by incomplete route context, C3 labels, and dependent target families for route maps, actor motion, occupancy, legality, risk, progress, stagnation, and ranking. The paper uses that branch as a negative control and an inputcompleteness warning, not as part of the main benchmark claim.

## Generation-side pivot and preservation

The first frozen-backbone generation adapters established that future tokens could reach the generation path, but they also showed why unconstrained injection is unsafe. A regenerated-candidate official-PDM check for an early PDM-target adapter had aligned oracle-best delta −0.001082 and aligned selected delta −0.001280 versus base. Other high-gain mode-logit adapters changed candidates more aggressively but often damaged all-candidate oracle quality. The failure mode was not that the adapter was inactive; it was that future-token perturbations could move strong base candidates in scenes where the future signal did not justify the motion.

Base-candidate preservation produced the stable operating point used in the paper. With a split48 frozen-backbone adapter at gain 0.1 and preservation weight 50, the zero-token path had candidate RMS delta 0.00000000, so the adapter remained an exact no-op under the zero-token control. The best preserved seed reported selected aligned PDM +0.000707 versus base and +0.000105 versus shuffled. Its all-candidate oracle comparison was mixed in absolute terms, but aligned oracle-best exceeded shuffled oracle-best by +0.006399 before post-generation TTC repair. Additional seed checks were useful but more mixed: one seed reached selected aligned +0.000586 while all-candidate oracle-best fell by −0.002789, and another reached selected aligned +0.000216 with oracle-best +0.000233. We therefore treat the frozen-backbone diagnostic as mechanism evidence, not as a replacement for the main benchmark tables.

## TTC boundary and guard policy

The TTC replay explains why the guard targets boundary crossings rather than clearance shrinkage in general. For token 45510307abe25c1f, candidate 0 has base PDM 1.000000 and TTC component 1.0. Aligned and shuffled future variants both score 0.583333 because TTC flips to 0.0 at ego time 4.0s. The first trigger occurs at a 0.9s future projection against vehicle object 72201d980aa95f34. The base trajectory clears the object by only 0.010833 m, while the token-induced shift is about 0.037 m RMS, enough to cross the official projection boundary.

The full projection-clearance extraction evaluated 216 variant rows from 54 high-PDM, TTC-safe base candidates and exactly matched the official scores. In the main preserved run, hard TTC-boundary crossings occurred for 1 aligned row, 2 shuffled rows, and 0 zero-token rows. A hard-cross fallback closed the residual aligned oracle-best gap from −0.000653 to +0.000220 versus base, but shuffled also improved to +0.000424. This is why the guard is described as safety repair rather than a source of future-specific gain. Dense clearance statistics were less decisive: aligned had mean clearance loss 0.033203 m and shuffled 0.024417 m, even though aligned had fewer hard crossings. The policy implication is to guard hard safe-tounsafe TTC transitions and a narrow near-contact band, while avoiding a broad loss that punishes every controlled approach to another actor.

## Controlled ablations: robustness, sensitivity, and guard aggregates

We include only completed controlled diagnostics that are directly tied to the DiffusionDrive/NAVSIM setting and are safe to interpret as mechanism evidence. Preliminary engineering checks on other protocols are excluded from the evidence chain.

Component breakdown. The same multiseed run provides a component-level diagnostic. Zero future tokens exactly match the base condition across selected and oracle-best rows. For selected trajectories, aligned future improves PDM by +0.000503 and ego-progress by +0.001207, while collision, TTC, comfort, and drivable-area components do not change at the aggregate level. Shuffled future also improves progress, but less (+0.000714). Oracle-best rows are more conservative: aligned future reduces oracle-best PDM by −0.001070, primarily through a progress decrease of −0.002568, while shuffled is lower still. This component pattern matches the main interpretation: preservation makes selected behavior mildly better and keeps safet components stable, but the diagnostic remains bounded and does not replace the main benchmark tables.

Sensitivity over preservation and gain. Several completed gain/preservation sweeps support the same cautious operating point. With gain 0.05 and preservation weight 50, selected aligned PDM improves by +0.000278 and oracle-best changes by −0.000764; increasing preservation to 100 gives selected +0.000317 and oracle-best −0.000734.

Gain 0.075, preservation 50, gives selected +0.000455 and oracle-best −0.000733. The reported operating point, gain 0.1 and preservation 50, gives selected +0.000707 and oracle-best −0.000653, while preservation 100 gives selected +0.000860 and oracle-best −0.000529 but fails the stricter candidate-generation gate. Progress-augmented variants at gain 0.1, preservation 50, remain similar (selected +0.000680 to +0.000678; oracle-best about −0.00069). These sweeps show that preservation controls damage from token injection, but they also show why the paper reports the diagnostic as mechanism evidence rather than an end-to-end ablation.

Aggregate TTC guard closure. The TTC guard was evaluated as a post-generation fallback over regenerated candidates for the same three split48 seeds. This is not guard training. Across seeds, raw aligned oracle-best delta averages −0.001070; hard-cross fallback changes this to +0.000143, for a mean guard improvement of +0.001213. The fallback improves aligned oracle-best in two of the three seeds and makes aligned nonnegative in two seeds. It blocks two aligned hard-cross rows and two shuffled hard-cross rows in total. However, the hard-cross fallback aligned-minus-shuffled oracle-best mean is −0.000043, so the guard should not be read as an aligned-specific gain. Its role is failure-mode closure: it repairs brittle safe-to-unsafe TTC crossings that can affect either aligned or shuffled perturbations.

These selected diagnostics address uncertainty and component-level evidence while keeping the claim boundary intact. They support four narrow conclusions: zero-token preservation is exact; aligned future tokens give a small selected-PDM advantage over shuffled under preservation; oracle-best candidate quality remains mixed in the frozen diagnostic; and hard TTC-boundary fallback repairs a localized safety failure but is not itself the source of future-specific improvement.

## What remains outside the main claim

Several diagnostics above are intentionally excluded from the main benchmark claim. The VAD/B2D route checks used routespecific future metadata and runtime-sensitive CARLA conditions. The fixed-candidate NAVSIM runs showed useful upper bounds but did not establish a deployable aligned-future selector. The frozen generation adapters were tiny, frozen-backbone diagnostics rather than full planner finetunes. The scorer audits showed that some future signal is learnable, but only after candidate materialization and only under incomplete training rows. These studies nevertheless shaped the final method: they motivated generation-time future tokens, exact zero-token controls, base-candidate preservation, and the narrow TTC-margin guard used in the main narrative.

## Broader Impacts

This work aims to improve anticipation and safety-sensitive planning in autonomous driving, but it should not be interpreted as validating deployment readiness. Future-conditioned generation can introduce new failure modes if predicted tokens are miscalibrated or if the planner overreacts to uncertain interactions. Any deployment would require closed-loop validation, safety monitoring, redundancy, and human oversight beyond the offline and benchmark evidence reported here.