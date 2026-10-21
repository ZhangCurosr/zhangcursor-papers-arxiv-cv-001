# Cross-Visibility-Gap Identity Persistence for 4D Scene Graphs

Jiahui Hu<sup>1</sup>, Li Gao<sup>2</sup>, Yu Liu<sup>3</sup>, Pengzhan Zhou<sup>2</sup>

<sup>1</sup>Department of Computer Science, The University of Hong Kong <sup>2</sup>Chongqing University

<sup>3</sup>Department of Computing, The Hong Kong Polytechnic University

## Abstract

4D scene graphs require persistent object identity, but a track may terminate when an object is transported inside an opaque carrier and later reappear as a fresh detection. In this regime, target appearance, self-motion, and last-seen geometry cannot determine its historical identity. Our key insight is that identity evidence need not remain attached to the invisible target: it can be transferred to the observable carrier and recovered after the gap. We therefore formulate closed-set cross-visibilitygap identity persistence as historical-node recovery through a remembered target–carrier binding. This turns a visibility discontinuity into a graph-routing problem: when direct target evidence disappears, identity can persist through a relation anchored to an observable carrier. Unlike extrapolative tracking, the method does not hallucinate target motion inside the carrier; it stores the last observed binding and delays identity assignment until reappearance supplies new routing evidence. We introduce CVG-SGM, a perception-frontend-compatible transfer–maintain–read memory. Geometry-weighted crossnode interaction writes target-specific evidence into carrier states, a shared per-node GRU maintains the binding during transport, and a two-time readout combines stored carrier– identity compatibility with current detection–carrier geometry to re-link a fresh detection. This separates identity evidence— who was carried—from routing evidence—where the returning detection belongs—and treats graph relations as memory paths rather than only prediction outputs.

In the controlled one-identity-per-carrier regime, CVG-SGM reaches 1.000 re-link accuracy versus 0.570 for no-memory and parameter-count-matched no-write controls, and gener alizes to scene-disjoint unseen longer gaps. On real RGB-D/SAM2 feature-level episodes, it re-links 36/36 held-out detections, compared with 28/36 for the strongest fixed carrier– identity control. Counterfactual carrier-memory swaps redirect re-linking according to the swapped binding, confirming that stored relational content drives recovery.

## Introduction

A temporal scene graph depends critically on persistent object identity: when a node no longer refers to the same physical object, its accumulated relations, attributes, and task history can no longer be reliably associated with that object. This problem becomes especially acute when an object is placed inside a moving carrier, disappears from view, and later reappears as a new detection. During transport, the target’s appearance is unavailable and its trajectory is no longer directly observable; if the returning detection is instantiated as a fresh node, it cannot inherit the state accumulated under the original identity. Scene graphs and their 3D/4D extensions model objects, relations, geometry, and temporal change for visual reasoning and embodied planning (Johnson et al. 2015; Armeni et al. 2019; Yang et al. 2023; Rana et al. 2023; Gu et al. 2024). However, existing 4D scenegraph pipelines primarily maintain identity through continuous tracking, appearance, or short-term geometric association, and do not provide an explicit mechanism for restoring a historical graph identity once a returning detection’s observation track has broken after a prolonged visibility gap. We refer to this under-addressed capability as cross-visibility-gap identity persistence.

A particularly diagnostic case involves indistinguishable objects carried inside identical moving containers. Appearance is uninformative; the target’s trajectory cannot be reliably extrapolated from its own history because its motion is governed by the carrier; and even the containment relation may become temporarily unobservable. The carrier, however, remains trackable in this diagnostic setting. If the graph records before disappearance that target o<sub>i</sub> entered carrier c<sub>j</sub>, tracks $c _ { j }$ through the gap, and retrieves this binding when the target reappears as a fresh detection, it can re-link that detection to the original target identity—persistence through the remembered object–carrier relation, who was carried by what, rather than through target appearance or trajectory. Our focus is the distinct broken-association setting in which the target track has terminated and a returning detection is instantiated as a fresh node that must be actively re-linked to an existing graph identity. We introduce CVG-SGM, a perception-frontend-compatible identity re-linking module built around a transfer–maintain–read cycle. Cross-node interaction transfers target-specific evidence into observable carrier states, per-node recurrence maintains it through the visibility gap, and a two-time readout combines stored identity compatibility with current routing geometry. Figure 1 illustrates the diagnostic setting and the role of remembered carrier relations.

Experimentally, FULL reaches 1.000±0.000 in controlled 2t2c, while NOMEM and the parameter-count-matched NOWRITE control remain at 0.570 ± 0.000. A single model trained on 5- and 10-frame gaps retains perfect accuracy on scene-disjoint 15- and 20-frame gaps. On real RGB-

![](images/44b83ada8faf6f2dd01fccd84df452b2f59de2d20058f4149d7a86a25afed834.jpg)  
Figure 1: Cross-visibility-gap identity persistence through remembered carrier relations. (a) Controlled 2t2c diagnostic: two identical targets bind to carriers c and c , are occluded while the carriers swap, and reappear near their current carriers at locations that are uninformative with respect to their own pre-gap positions. (b) The bindings persist as memory through the gap, re-linking each fresh detection to its identity; on unseen long gaps, FULL improves re-link accuracy from 0.576 (NOMEM) to 1.000. (c) Complementary human-operated cup-shufle videos provide ofline carrier-recovery evidence, while full real RGB-D/SAM2 feature-level 2t2c re-linking is reported in Table 4.

D/SAM2 feature-level 2t2c episodes, Neural FULL correctly re-links 36/36 held-out detections, compared with 28/36 for the strongest fixed carrier–identity control. Identity-key permutation, carrier-memory swapping, propagated read, poolorder reversal, and structured corruptions support the same carrier-conditioned binding mechanism. Our instance-level claim is limited to the one-identity-per-carrier regime.

## Contributions.

• We formulate cross-visibility-gap identity persistence as historical-node recovery when target tracking, appearance, and target-centric motion are unavailable.

• We propose CVG-SGM, a transfer–maintain–read memory that stores target evidence in carrier states and relinks fresh detections through a two-time memory-andgeometry readout.

• We isolate the mechanism through parameter-countmatched no-write, identity-permutation, carrier-memoryswap, propagated-read, and pool-order controls, and evaluate it on independently generated canonical scenes, scene-disjoint unseen gaps, structured node-level corruptions, real RGB-D/SAM2 full re-linking, roster scaling, and shared-carrier ambiguity.

## Related Work

## Occlusion Tracking and Object Memory

Object-permanence and occlusion-tracking methods maintain target continuity during disappearance using appearance cues, propagated masks, or hidden-motion estimates (Shamsian et al. 2020; Tokmakov et al. 2021, 2022; Oh et al. 2019; Cheng and Schwing 2022). Promptable video trackers and tracking-through-occlusion benchmarks further improve target-centric continuity under prolonged occlusion (Ravi et al. 2025; Van Hoorick et al. 2023). Recent shell-game diagnostics evaluate whether models can preserve the identities of visually indistinguishable entities through spatiotemporal continuity (Liu and Lee 2026), while object-memory systems maintain explicit entity representations for interaction, hidden-object reasoning, or scene-state anticipation (Huang et al. 2024; Teng et al. 2021; Peddi et al. 2024). In contrast, CVG-SGM addresses a broken-association regime in which the original target track has terminated and a fresh detection must be re-linked to its historical graph node through a remembered target–carrier binding.

## Temporal, World, and 4D Scene Graphs

Scene graphs and their 3D/4D extensions model objects, relations, geometry, and temporal change for visual reasoning and embodied use (Krishna et al. 2017; Rosinol et al. 2020; Yang et al. 2023; Rana et al. 2023; Gu et al. 2024). Dynamic graph and incremental mapping systems maintain identity through geometric fusion, frontend associations, interaction tracking, or semantic matching (Ji et al. 2020; Cong et al. 2021; Wu et al. 2021; Hughes, Chang, and Carlone 2022; Behrens et al. 2025; Al Mdfaa et al. 2025). CVG-SGM targets the complementary case where visually indistinguishable objects reappear as fresh nodes and must be re-linked through carrier-mediated relation memory.

## Method

## Problem Formulation

We model a dynamic scene as a 4D scene graph whose nodes are objects and whose edges are pairwise relations, both indexed by time. Each node n carries a frontend-provided feature $\mathbf { x } _ { n } ^ { i } \in \mathbb { R } ^ { D }$ containing geometric, observability, and category-related cues, including its image-plane position $\mathbf { p } _ { n } ^ { t }$ and a visibility indicator $v _ { n } ^ { t } \in \{ 0 , 1 \}$

When an object’s track is interrupted by a long visibility gap—for example, because it becomes contained in or transported by an observable carrier and is consequently occluded—it may re-enter the scene as afresh detection with no persistent identifier. The cross-visibility-gap re-linking task is to assign each fresh detection to the historical node it continues.

In our appearance-degenerate diagnostic, visual appearance is uninformative and target-centric motion is unavailable through the gap; recovery must therefore rely on relational history accumulated while the target was associated with an observable carrier.

Let H denote the set of historical target nodes, $\mathcal { P }$ the set of fresh detections, and C the set of candidate carriers. For each fresh detection $k \in \mathcal { P }$ , the model predicts

$$
\hat { \boldsymbol { \imath } } _ { k } = \arg \operatorname* { m a x } _ { i \in \mathcal { H } } \boldsymbol { s } _ { k i } ,\tag{1}
$$

where $s _ { k i }$ is the re-link compatibility between fresh detection k and historical identity i, defined in the readout below.

## CVG-SGM Overview

As shown in Fig. 2, CVG-SGM follows a transfer–maintain– read cycle: geometry-weighted cross-node transfer writes target-specific evidence into an observable carrier state, pernode recurrent states updated by a shared GRU maintain it through the gap, and a two-time readout combines stored carrier–identity compatibility with reappearance geometry. Masked intra-frame attention adds an implicit cross-node path; the auxiliary relation logits do not enter the inferencetime re-linking, though the relation loss supervises shared representations during training.

## Node Encoding and Masked Intra-Frame Interaction

Each node feature is encoded and augmented with a learned track-indexed embedding,

$$
\mathbf { z } _ { n } ^ { t } = \phi _ { \mathrm { e n c } } ( \mathbf { x } _ { n } ^ { t } ) + \mathbf { e } _ { \kappa ( n ) }\tag{2}
$$

where $\phi _ { \mathrm { e n c } }$ is an MLP, $\kappa ( n )$ denotes the episode-local track address assigned to node n, and ${ \mathbf e } _ { \kappa ( n ) }$ is the corresponding learned embedding. These embeddings serve as episodelocal graph addresses rather than semantic identity descriptors. Historical target-track embeddings are reused as candidate identity keys, while fresh detections receive a shared reserved pool address. The intended computation is therefore to transfer address-coded evidence into carrier states and later recover the corresponding historical node; we do not claim appearance-based or open-vocabulary re-identification. Relinking consequently depends on stored carrier memory and current routing geometry rather than a persistent identifier attached to the fresh detection. We evaluate an episode-wise identity-key permutation control to test whether the model exploits a globally fixed identity-slot ordering. Visibility $\boldsymbol { v } _ { n } ^ { t }$ masks intra-frame interaction.

Within each frame, nodes exchange information through masked multi-head self-attention. Let $\mathbf { Z } _ { \mathrm { v i s } } ^ { t } = \{ \mathbf { z } _ { j } ^ { t } \mid v _ { j } ^ { t } = \bar { 1 } \}$ denote the set of currently visible node features. Then,

$$
\begin{array} { r } { \widetilde { \mathbf Z } _ { n } ^ { t } = \mathrm { M H A } \left( \mathbf Z _ { n } ^ { t } , \mathbf Z _ { \mathrm { v i s } } ^ { t } \right) . } \end{array}\tag{3}
$$

Currently invisible nodes are excluded from $\mathbf { Z } _ { \mathrm { v i s } } ^ { t }$ and therefore do not serve as attention keys or values. This creates an implicit cross-node interaction path complementary to the explicit geometry-weighted message.

## Geometry-Weighted Cross-Node Relational Transfer

In addition to masked attention, CVG-SGM applies a general proximity-weighted message over ordered node pairs,

$$
\begin{array} { r } { a _ { n j } ^ { t } = 1 [ n \neq j ] ~ v _ { j } ^ { t } \exp \Big ( - \frac { \| { \bf p } _ { n } ^ { t } - { \bf p } _ { j } ^ { t } \| _ { 2 } } { \tau _ { w } } \Big ) } \end{array}\tag{4}
$$

$$
\mathbf { m } _ { n } ^ { t } = \sum _ { j } a _ { n j } ^ { t } W _ { \mathrm { w r i t e } } \tilde { \mathbf { z } } _ { j } ^ { t }\tag{5}
$$

$$
{ \bf g } _ { n } ^ { t } = \widetilde { { \bf z } } _ { n } ^ { t } + { \bf m } _ { n } ^ { t }\tag{6}
$$

The exchange is defined over all ordered node pairs, excludes self-messages, and is gated by the visibility of the source $j ,$ so an invisible node contributes no new message. The architecture imposes no typed target→carrier edge and no carrier-only receiver mask.

When a target and a carrier are spatially associated before disappearance, target-specific information can enter the carrier state through this generic transfer. The resulting carrierspecific content is therefore a task-induced consequence of the learned interaction rather than a typed target-to-carrier architectural constraint. This explicit geometry-weighted message provides a second cross-node transfer path in addition to masked intra-frame attention.

## Recurrent Memory and Auxiliary Relation Prediction

A shared GRU is applied independently along each node trajectory to integrate the contextualized features and geometryweighted messages over time,

$$
\mathbf { M } _ { n } ^ { 1 : T } = \mathrm { G R U } \bigl ( \mathbf { g } _ { n } ^ { 1 : T } \bigr ) ,\tag{7}
$$

![](images/f64f6ace112faff6e0d8409d2dfd5ff1e083f6c7ebd398eddcc2318abf3889c1.jpg)  
Figure 2: Overview of CVG-SGM. (1) Identical targets disappear with separately trackable carriers and reappear without continuous target tracks. (2) Nodes from GT, SAM2, or 4DSG pass through a track-indexed encoder and masked intra-frame attention. (3) Geometry-weighted cross-node transfer lets target-specific evidence enter a carrier’s recurrent state, updated by a shared GRU that maintains it through the gap; an auxiliary relation head reads these states under relation supervision. (4) At reappearance, routing weights $r _ { k c }$ from current detection–carrier geometry combine with identity compatibility lBci $B _ { c i }$ from end-of-carry memory as $\begin{array} { r } { s _ { k i } = \sum _ { c \in \mathcal { C } } r _ { k c } B _ { c i } } \end{array}$ to recover the historical identity.

This yields one recurrent state $\mathbf { M } _ { n } ^ { t }$ per node while sharing the GRU parameters across nodes. The recurrent state maintains transferred information after the target’s observation track becomes unavailable.

In the NOMEM memory-free ablation, both the explicit geometry-weighted message and the recurrent GRU update are bypassed. The contextualized per-frame representation $\tilde { \mathbf { z } } _ { n } ^ { t }$ is used directly in place of the recurrent state, while masked intra-frame attention is retained.

An auxiliary relation head reads the recurrent node states,

$$
\begin{array} { r } { \boldsymbol { \ell } ^ { t } = \phi _ { \mathrm { r e l } } ( \mathbf { M } ^ { t } ) \in \mathbb { R } ^ { N \times N \times | \mathcal { R } | } , } \end{array}\tag{8}
$$

where N is the number of nodes and $\boldsymbol { \ell } _ { n j } ^ { t } \in \mathbb { R } ^ { | \mathcal { R } | }$ contains the logits for ordered node pair $( n , j )$ over relation categories $\mathcal { R }$ These logits are supervised by binary cross-entropy against the ground-truth pairwise relation labels.

The predicted relation logits do not gate the cross-node transfer and do not enter the identity re-linking score at inference time. The auxiliary relation loss nevertheless backpropagates through the recurrent node states and shared upstream representations, providing additional supervision during training.

## Two-Time Carrier-Memory Readout

Re-linking draws on two decoupled sources of evidence observed at diferent frames. Identity evidence is read from the canonical end-of-carry carrier-memory snapshot, denoted by $t _ { g } .$ , through the bilinear compatibility

$$
B _ { c i } = \mathrm { B i } [ \mathbf { M } _ { c } ^ { t _ { g } } , \mathbf { e } _ { i } ] = ( \mathbf { M } _ { c } ^ { t _ { g } } ) ^ { \top } W _ { \mathrm { b i l } } \mathbf { e } _ { i }\tag{9}
$$

between carrier $c \mathbf { \hat { s } }$ recurrent memory $\mathbf { M } _ { c } ^ { t _ { g } }$ and the identity key e<sub>i</sub> of historical target i, where $W _ { \mathrm { b i l } }$ is a learned weight matrix. A higher $B _ { c i }$ indicates that carrier $c \mathbf { \hat { s } }$ memory more strongly encodes a binding to target i—i.e. that c is more likely the carrier that held target i during the gap. In the controlled diagnostic, $t _ { g }$ is provided by the scripted end-of-carry event and is used only as a canonical analysis snapshot. The inference rule need not know this event boundary: replacing $\mathbf { M } _ { c } ^ { t _ { g } }$ with the latest pre-reappearance state $\mathbf { M } _ { c } ^ { t _ { r , k } - 1 }$ produces identical predictions in the propagated-read diagnostic.

Routing evidence is obtained from the geometry at fresh detection k’s reappearance frame $t _ { r , k }$

$$
d _ { k c } = \| \mathbf { p } _ { k } ^ { t _ { r , k } } - \mathbf { p } _ { c } ^ { t _ { r , k } } \| _ { 2 } + \lambda _ { \mathrm { v i s } } \left( 1 - v _ { c } ^ { t _ { r , k } } \right)\tag{10}
$$

$$
r _ { k c } = \frac { \exp ( - d _ { k c } / \tau _ { a } ) } { \sum _ { c ^ { \prime } } \exp ( - d _ { k c ^ { \prime } } / \tau _ { a } ) }\tag{11}
$$

where invisible carriers incur the fixed penalty $\lambda _ { \mathrm { v i s } } = 1 0$

The two evidence sources are combined through a routingweighted aggregation,

$$
s _ { k i } = \sum _ { c \in \mathcal { C } } r _ { k c } B _ { c i } ,\tag{12}
$$

where the normalized routing weights determine how strongly the identity compatibility stored in each candidate carrier contributes to fresh detection k. The fresh detection is then re-linked to the historical node maximizing $s _ { k i }$ according to Eq. (1).

The predicted historical-node index provides the relinking interface through which a downstream graph manager can attach the fresh observation track to an existing node.

We evaluate recovery of this historical identity; executing the subsequent online graph update is outside the scope of this work.

## Training Objective and Complexity

The canonical re-linking model is trained with

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { i d } } + \mathcal { L } _ { \mathrm { r e l } }\tag{13}
$$

where ${ \mathcal { L } } _ { \mathrm { i d } }$ is cross-entropy over the historical-identity scores $s _ { k i }$ against the ground-truth slot, and $\mathcal { L } _ { \mathrm { r e l } }$ is binary crossentropy over pairwise relation logits.

For the latency benchmark configuration $( N { = } 1 3 , T { = } 3 0 ,$ $d _ { \mathrm { h i d } } { = } 1 2 8 )$ , the shared GRU adds 99,072 parameters $( + 5 6 . 6 \% )$ and raises median forward latency from 1.752 to 2.150 ms. Online inference stores $O ( N d _ { \mathrm { h i d } } )$ recurrent state independent of gap duration, while training uses $O ( T N d _ { \mathrm { h i d } } )$ storage and $O ( \breve { T } N ^ { 2 } )$ pairwise computation.

## Experiments

We evaluate full historical-identity re-linking and mechanism isolation in controlled Kubric scenes, independently generated canonical scenes, structured node-level corruptions, real RGB-D/SAM2 two-target/two-carrier re-linking, complementary real-video carrier recovery, roster scaling and shared-carrier collision, and a PSG4D scope-boundary test.

## Controlled-Diagnostic Setup

Task. The controlled two-target/two-carrier (2t2c) diagnostic is generated with Kubric (Gref et al. 2022). Two visually identical targets are each associated with one of two identical carriers while visible; the targets become occluded as the carriers exchange positions and later reappear near their current carriers at locations that are uninformative with respect to their own pre-gap positions. Each reappearance is instantiated as a fresh detection without a persistent identifier, so recovery must use the remembered target–carrier binding. Each carrier holds exactly one target, with $| \mathcal { H } | = | \mathcal { C } | = 2$

Frontends. The same CVG-SGM architecture consumes two interchangeable frontends. Clean-GT provides exact geometry, visibility, and category, but a reappearing target is still instantiated as a fresh detection with no persistent identifier. The SAM2 frontend supplies SAM2-derived pre-gap target geometry and carrier tracks, together with a ground-truth detector proxy for reappearing detections. It therefore tests the identity-persistence mechanism under SAM2-derived geometry, not an end-to-end perceptual reappearance pipeline.

Splits. Each gap length contains 500 episodes, partitioned into 400 training and 100 validation episodes by a deterministic filename-based split fixed across seeds. A gap-invariant frame-0 node-layout audit confirms zero training–validation scene-signature overlap at every gap length. For OOD evaluation, one model is trained on 750 episodes from 375 scene signatures at gaps 5 and 10 and tested on 250 episodes from 125 disjoint scene signatures at unseen gaps 15 and 20; zero train–test overlap is verified.

Independent test and structured corruptions. After fixing the training protocol, architecture, and source checkpoints, we generated a separate 200-episode canonical gap-10 test set using scene seeds 10200–10399. All episodes pass generator verification, contain 34 frames, and have no discovered scene-seed overlap with previously indexed raw datasets. The set contains 400 re-link decisions per checkpoint and is reserved for final evaluation; it is not used for training, hyperparameter tuning, checkpoint selection, or protocol modification.

Read jitter adds Gaussian noise with $\sigma = 0 . 0 2$ to normalized coordinates at each detection’s actual re-link frame. Maintenance dropout removes carrier observations with probability $q \ : = \ : 0 . 2 0$ only after all historical targets have disappeared and before the end of carry. Each condition uses three deterministic corruption seeds.

Training and reporting. Canonical mechanism, gap, OOD, and permutation results average epochs 60, 70, and 80 within each seed and report mean ± sample SD over seeds 3–5. Independent tests use frozen epoch-80 checkpoints, averaging corruption realizations within each training seed before aggregation. SAM2 and scaling experiments use the stated fixed-learning-rate protocol with seeds 0–2.

Ablations. All variants share inputs, readout, losses, training, and instantiated modules. FULL uses masked attention, the explicit geometry-weighted message, and the shared pernode GRU. NOMEM bypasses the explicit message and GRU while retaining masked attention. NOWRITE retains the GRU but disables both cross-node paths, whereas NOREL-WRITE retains masked attention and the GRU but removes the explicit message. Parameter matching is structural; bypassed branches receive no efective gradients.

Metrics and oracle. Re-link accuracy measures whether a fresh detection recovers the correct historical identity. In 2t2c it coincides with carrier-association accuracy, since each carrier holds one target. Per-carrier set recovery is used only for the shared-carrier analysis. The oracle replaces learned carrier memory with the ground-truth target–carrier binding while retaining geometric routing; its value of 1.0 is specific to clean one-identity-per-carrier 2t2c.

## Main Cross-Gap Re-linking Results

Controlled and OOD results. Table 1 summarizes the canonical, frontend, and scene-disjoint long-gap protocols, while the independently generated frozen test set is reported separately below. The SAM2 setting uses a groundtruth detector proxy at reappearance and therefore evaluates frontend-compatible identity persistence rather than an endto-end perception pipeline. Clean-GT and OOD use cosine annealing with seeds 3–5, whereas SAM2 uses the separately reported fixed-learning-rate protocol with seeds 0–2.

Robustness across gap length. Across separately trained models for gaps 5, 10, 15, and 20, FULL remains at 0.997– 1.000, whereas NOMEM remains at 0.570. Last-position (0.550) and mask-propagation (0.149–0.203) baselines are also inefective. Unlike these per-gap models, Table 1 evaluates one model trained on gaps 5 and 10 on scene-disjoint scenes with unseen gaps 15 and 20.

<table><tr><td>Protocol</td><td>NOMEM</td><td>FULL</td><td>Oracle</td><td>Gain</td></tr><tr><td>Clean-GT, gap 10</td><td> $0 . 5 7 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td>1.000</td><td>+0.430</td></tr><tr><td>SAM2 geometry</td><td> $0 . 5 5 0 \pm 0 . 0 8 7$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td>1.000</td><td> $+ 0 . 4 5 0$ </td></tr><tr><td> $\mathrm { O O D } 5 , 1 0  1 5 , 2 0$ </td><td> $0 . 5 7 6 \pm 0 . 0 0 0$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td>1.000</td><td> $+ 0 . 4 2 4$ </td></tr></table>

Table 1: Re-link accuracy over three seeds. SAM2 uses derived pre-gap geometry with a ground-truth reappearance proxy; OOD trains on gaps 5 and 10 and evaluates scenedisjoint gaps 15 and 20.
<table><tr><td>Condition</td><td>NOMEM</td><td>FULL</td><td>Gain</td></tr><tr><td>Clean</td><td> $0 . 5 3 0 \pm 0 . 0 0 0$ </td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td>+0.470</td></tr><tr><td>Read jitter  $( \sigma = . 0 2 )$ </td><td> $0 . 5 2 9 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 9 9 \pm 0 . 0 0 1 \ + 0 . 4 7 0 }$ </td><td></td></tr><tr><td>Maint. dropout  $( q = . 2 0 )$ </td><td> $0 . 5 1 6 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 8 8 7 \pm 0 . 0 4 4 \ + 0 . 3 7 1 }$ </td><td></td></tr></table>

Table 2: Re-link accuracy on 200 independently generated gap-10 scenes. Corruption seeds are averaged within each training seed before reporting mean ± sample SD over three frozen training seeds.

Independent-scene robustness. We next evaluate the frozen epoch-80 checkpoints on the 200 independently generated canonical scenes and their structured corruptions.

FULL remains perfect on the independent clean set and changes by less than 0.1 percentage points under read jitter. Maintenance dropout causes an 11.3-percentage-point decrease, but FULL remains 37.1 percentage points above NOMEM. This asymmetric degradation identifies persistent carrier maintenance as a principal sensitivity of the memory pathway, not the exclusive source of all errors.

## What Drives Identity Recovery?

The main results show that the complete persistent pathway improves identity recovery, but do not determine whether the gain arises from recurrence alone or from transferring information across nodes before recurrent maintenance. Table 3 therefore compares variants that selectively remove masked attention, the explicit geometry-weighted message, or recurrent state maintenance.

NOWRITE matches NOMEM despite retaining the GRU, showing that recurrence without cross-node transfer is insufficient. NORELWRITE is unstable across seeds, whereas FULL reaches 1.000 ± 0.000, showing that the explicit geometry-weighted message makes carrier-conditioned storage reliable.

Mechanism controls. Episode-wise identity-key permutation leaves FULL at $1 . 0 0 0 \pm 0 . 0 0 0$ on canonical and OOD evaluation, while NOMEM remains near 0.57, ruling out a fixed identity-slot shortcut. Removing relation supervision leaves FULL at 0.999 versus 0.570 for NOMEM. Swapping only the two carrier-memory rows changes instance accuracy from 1.000 to 0.000 and makes all 1,800 predictions follow the swapped binding; this read-path intervention does not rerun the recurrent trajectory and is not a real-world causal experiment. Reading the latest pre-reappearance carrier state instead of the canonical snapshot retains 1.000 accuracy and prediction agreement across all nine checkpoints. Reversing fresh-pool order likewise leaves every prediction unchanged, with a maximum score diference of zero and no track-IDorder mismatch.

<table><tr><td>Variant</td><td></td><td>Masked attn. Explicit msg.</td><td>GRU</td><td>Re-link acc.</td></tr><tr><td>NOMEM</td><td>Yes</td><td>No</td><td>No</td><td> $0 . 5 7 0 \pm 0 . 0 0 0$ </td></tr><tr><td>NOWRITE</td><td>No</td><td>No</td><td>Yes</td><td> $0 . 5 7 0 \pm 0 . 0 0 0$ </td></tr><tr><td>NORELWRITE</td><td>Yes</td><td>No</td><td>Yes</td><td> $0 . 7 1 6 \pm 0 . 2 4 6$ </td></tr><tr><td>FULL</td><td>Yes</td><td>Yes</td><td>Yes</td><td> $\mathbf { 1 . 0 0 0 \pm 0 . 0 0 0 }$ </td></tr></table>

Table 3: Mechanism ablations on controlled 2t2c re-linking. “Explicit msg.” denotes the general geometry-weighted cross-node message in Eq. (5).
<table><tr><td>Method</td><td>All (desc.)</td><td>Held-out test</td></tr><tr><td>Side-only prior</td><td>0.508</td><td>0.556</td></tr><tr><td>Carrier prior</td><td>0.483</td><td>0.222</td></tr><tr><td>Fixed carrier-identity</td><td>0.517</td><td>0.778</td></tr><tr><td>Neural FULL</td><td>0.983</td><td>1.000</td></tr></table>

Table 4: Real-data full re-linking over 120 RGB-D/SAM2 episodes. $\mathbf { \ddot { A } } \mathbf { l } \mathbf { l } ^ { \prime \prime }$ descriptively combines 240 train/validation/test decisions; the held-out test contains 36 decisions. Controls are deterministic, while Neural FULL is identical across five training seeds.

## Real RGB-D/SAM2 2t2c Full Re-linking

We evaluate full re-linking on 120 frozen real RGB-D/SAM2 2t2c episodes, yielding 240 decisions split by episode into 84/18/18 train/validation/test episodes. The feature-level model consumes frozen geometry and relation-distance features, while fresh identities appear only as labels; this is not raw-video end-to-end ReID.

Across all 240 decisions, the deterministic side-only, carrier-prior, and fixed carrier–identity controls obtain 122/240 (50.8%), 116/240 (48.3%), and 124/240 (51.7%), respectively. The explicit symbolic relation-chain realization of CVG-SGM reaches 236/240 (98.3%) overall and 36/36 on the held-out test. The feature-level Neural FULL realization matches this behavior over five training seeds, reaching 97.6%, 100.0%, and 100.0% on the train, validation, and test splits, respectively; neural NOMEM gives $4 8 . 9 \pm 5 . 4 \%$ test accuracy over five seeds, reported using population SD. Symbolic and neural FULL should therefore be viewed as complementary realizations of the same carrier-conditioned write–maintain–read principle rather than competing methods.

The real-data neural realization optionally uses an auxiliary write loss during training. Setting its weight to zero leaves FULL unchanged at 97.6%, 100.0%, and 100.0% on train, validation, and test, showing that final re-link supervision alone is suficient to learn the feature-level carrierconditioned write–read pathway. This real-data-specific write loss is separate from the auxiliary relation loss $\mathcal { L } _ { \mathrm { r e l } }$

<table><tr><td>Setting</td><td>NOMEM</td><td>Sym. Range</td><td>FULL</td></tr><tr><td>4 tgt / 4 car</td><td> $0 . 2 1 7 \pm 0 . 0 5 6$ </td><td>0.792–0.875</td><td> $0 . 9 1 7 \pm 0 . 0 0 7$ </td></tr><tr><td>4 tgt / 3 car</td><td> $0 . 2 7 1 \pm 0 . 0 6 2$ </td><td>0.479-0.742</td><td> $0 . 7 3 8 \pm 0 . 0 1 3$ </td></tr></table>

Table 5: Instance-level roster scaling and carrier collision. “Sym. Range” spans Hard-mean, Hard-max, and Hard-last; entries report mean ± sample SD over seeds 0–2.

in the controlled model.

Stress diagnostics on the symbolic relation-chain realization show that the real-data result is not an artifact of oracle event timing. Timing inferred from observed relation records matches oracle timing at 236/240 overall and 36/36 on the held-out test. The result also remains 236/240 under 2D-only distance, stride-5 or stride-10 subsampling, 0.10 m distance noise, and 50% relation dropout. Severe stride-20 subsampling reduces performance to 196/240 overall and 34/36 on test. Over five perturbation seeds, 70% relation dropout yields $9 6 . 8 \pm 1 . 1 \%$ overall and $9 8 . 9 \pm 2 . 2 \%$ test accuracy. These diagnostics establish robustness of the explicit relation-chain realization under moderate feature-level perturbations, not end-to-end raw-video robustness of the neural model.

## Roster Scaling and Carrier Collision

We next increase the roster to four targets and test both distinct-carrier and shared-carrier settings. Models use the sum readout under the fixed-learning-rate protocol (seeds 0– 2, epoch 80). Instance accuracy is computed independently for each fresh detection, while episode exact-match requires the complete assignment to be correct.

With four distinct carriers, CVG-SGM reaches 0.917 ± 0.007 instance accuracy, numerically higher than the strongest deterministic variant at 0.875. When two identities share one carrier, CVG-SGM reaches 0.738 ± 0.013, comparable to the strongest symbolic variant at 0.742, while the symbolic variants span 0.479–0.742. Carrier-association accuracy nevertheless remains 0.983 ± 0.014 and per-carrier set recovery remains $1 . 0 0 0 \pm 0 . 0 0 0$ , indicating that the remaining dificulty primarily concerns within-carrier identity resolution.

On the same frozen FULL score matrices, Hungarian oneto-one assignment raises episode exact-match from 0.833 to 0.983 without collision and from 0.033 to 0.567 under collision. Global assignment eliminates duplicate identity assignments by construction; residual errors may additionally reflect routing, calibration, or representation failures. These results do not imply universal dominance over symbolic implementations. They instead show that carrier-conditioned transfer–maintain–read can be instantiated either symbolically with reliable relations or diferentiably inside a learned scene-graph model.

Complementary Real-Video Evidence and Scope Boundary. Beyond the full real-data re-linking experiment, a separate training-free SAM2 relation-anchored pipeline improves TCOW cup-game container selection from 0.049 to

0.570 and container IoU from 0.070 to 0.492. On humanoperated cup-shufle videos, ofline carrier selection yields a leave-one-action-category-out pooled gain of +0.458 over NOMEM. These experiments provide complementary evidence for carrier recovery but do not evaluate full historicalnode re-linking. On PSG4D-GTA, adding the targeted memory module reduces generic relation recall, with video meanrecall at 100 (vmR@100) decreasing from 0.298 to 0.156. CVG-SGM is therefore not claimed as a universal improvement for generic 4D scene-graph relation prediction.

## Limitations and Conclusion

Limitations. Our instance-level claim assumes one identity per carrier. Under the current observation model, visually indistinguishable targets sharing one carrier provide no additional cue that determines their within-carrier identity ordering, making the instance-level bijection underdetermined. Hungarian one-to-one decoding removes duplicate assignments and raises 4t3c episode exact-match from 0.033 to 0.567, but the residual errors are consistent with withincarrier ambiguity and may also reflect routing, calibration, or representation failures. Carrier-association accuracy and per-carrier set recovery nevertheless remain high.

Gap robustness is verified only over gaps 5–20 and one scene-disjoint short-to-long protocol, not arbitrary visibilitygap duration. The independent corruption study operates on structured graph observations: read jitter perturbs node coordinates at reappearance, while maintenance dropout removes carrier observations and incident relations after the historical targets disappear. These tests do not perturb raw RGB, SAM2 masks, detector recall or precision, or camera calibration.

The real RGB-D/SAM2 experiment evaluates full featurelevel historical-node re-linking over frozen geometry and relation-distance features, but it does not establish end-to-end raw-video ReID, detector, segmentation, sensor, or closedloop robot robustness. The controlled SAM2 frontend also uses a ground-truth detector proxy at reappearance. Finally, identity keys are learned episode-local addresses; dynamic address creation remains open.

Conclusion. We identify cross-visibility-gap identity persistence as historical-node recovery after track breakage and introduce CVG-SGM, a carrier-conditioned transfer– maintain–read memory for 4D scene graphs. Mechanism ablations, identity permutation, counterfactual memory swapping, propagated read, and pool-order reversal show that recovery depends on transferred and persistently maintained carrier-bound content. Frozen checkpoints retain perfect accuracy on 200 independently generated canonical scenes, tolerate moderate read-side coordinate noise, and generalize from gaps 5 and 10 to scene-disjoint gaps 15 and 20. The mechanism is further validated on real RGB-D/SAM2 feature-level full 2t2c re-linking, where Neural FULL recovers all 36/36 held-out identities from 18 episodes. Resolving shared-carrier instance ambiguity, extending the feature-level result to end-to-end raw-video re-linking, dynamically creating identity addresses, and integrating the module with closed-loop embodied agents remain future work.

## References

Al Mdfaa, M.; Lukina, S.; Akhtyamov, T.; Nigmatzyanov, A.; Nalberskii, D.; Zagoruyko, S.; and Ferrer, G. 2025. VL-KnG: Persistent Spatiotemporal Knowledge Graphs from Egocentric Video for Embodied Scene Understanding. arXiv preprint arXiv:2510.01483.

Armeni, I.; He, Z.-Y.; Gwak, J.; Zamir, A. R.; Fischer, M.; Malik, J.; and Savarese, S. 2019. 3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Behrens, T.; Zurbrügg, R.; Pollefeys, M.; Bauer, Z.; and Blum, H. 2025. Lost & Found: Tracking Changes from Egocentric Observations in 3D Dynamic Scene Graphs. IEEE Robotics and Automation Letters, 10(4): 3739–3746.

Cheng, H. K.; and Schwing, A. G. 2022. XMem: Long-Term Video Object Segmentation with an Atkinson-Shifrin Memory Model. In Proceedings of the European Conference on Computer Vision.

Cong, Y.; Liao, W.; Ackermann, H.; Rosenhahn, B.; and Yang, M. Y. 2021. Spatial-Temporal Transformer for Dynamic Scene Graph Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Gref, K.; Belletti, F.; Beyer, L.; Doersch, C.; Du, Y.; Duckworth, D.; Fleet, D. J.; Gnanapragasam, D.; Golemo, F.; Herrmann, C.; et al. 2022. Kubric: A Scalable Dataset Generator. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 3749–3761.

Gu, Q.; Kuwajerwala, A.; Morin, S.; Jatavallabhula, K. M.; Sen, B.; Agarwal, A.; Rivera, C.; Paul, W.; Ellis, K.; Chellappa, R.; Gan, C.; de Melo, C. M.; Tenenbaum, J. B.; Torralba, A.; Shkurti, F.; and Paull, L. 2024. ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning. In Proceedings of the IEEE International Conference on Robotics and Automation, 5021–5028.

Huang, Y.; Yuan, J.; Kim, C.; Pradhan, P.; Chen, B.; Li, F.; and Hermans, T. 2024. Out of Sight, Still in Mind: Reasoning and Planning about Unobserved Objects with Video Tracking Enabled Memory Models. In Proceedings of the IEEE International Conference on Robotics andAutomation, 2116–2123.

Hughes, N.; Chang, Y.; and Carlone, L. 2022. Hydra: A Real-Time Spatial Perception System for 3D Scene Graph Construction and Optimization. In Robotics: Science and Systems.

Ji, J.; Krishna, R.; Fei-Fei, L.; and Niebles, J. C. 2020. Action Genome: Actions as Compositions ofSpatio-Temporal Scene Graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Johnson, J.; Krishna, R.; Stark, M.; Li, L.-J.; Shamma, D. A.; Bernstein, M. S.; and Fei-Fei, L. 2015. Image Retrieval Using Scene Graphs. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Krishna, R.; Zhu, Y.; Groth, O.; Johnson, J.; Hata, K.; Kravitz, J.; Chen, S.; Kalantidis, Y.; Li, L.-J.; Shamma, D. A.; Bernstein, M. S.; and Fei-Fei, L. 2017. Visual Genome: Connecting Language and Vision Using Crowdsourced Dense

Image Annotations. International Journal of Computer Vision, 123(1): 32–73.

Liu, T.; and Lee, W. S. 2026. Can Vision-Language Models Solve the Shell Game? arXiv:2603.08436.

Oh, S. W.; Lee, J.-Y.; Xu, N.; and Kim, S. J. 2019. Video Object Segmentation Using Space-Time Memory Networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Peddi, R.; Singh, S.; Saurabh; Singla, P.; and Gogate, V. 2024. Towards Scene Graph Anticipation. In Proceedings ofthe European Conference on Computer Vision.

Rana, K.; Haviland, J.; Garg, S.; Abou-Chakra, J.; Reid, I.; and Sünderhauf, N. 2023. SayPlan: Grounding Large Language Models Using 3D Scene Graphs for Scalable Robot Task Planning. In Proceedings of the 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, 23–72.

Ravi, N.; Gabeur, V.; Hu, Y.-T.; Hu, R.; Ryali, C.; Ma, T.; Khedr, H.; Rädle, R.; Rolland, C.; Gustafson, L.; Mintun, E.; Pan, J.; Alwala, K. V.; Carion, N.; Wu, C.-Y.; Girshick, R.; Dollár, P.; and Feichtenhofer, C. 2025. SAM 2: Segment Anything in Images and Videos. In International Conference on Learning Representations.

Rosinol, A.; Gupta, A.; Abate, M.; Shi, J.; and Carlone, L. 2020. 3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans. In Robotics: Science and Systems.

Shamsian, A.; Kleinfeld, O.; Globerson, A.; and Chechik, G. 2020. Learning Object Permanence from Video. In Proceedings of the European Conference on Computer Vision.

Teng, Y.; Wang, L.; Li, Z.; and Wu, G. 2021. Target Adaptive Context Aggregation for Video Scene Graph Generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 13688–13697.

Tokmakov, P.; Jabri, A.; Li, J.; and Gaidon, A. 2022. Object Permanence Emerges in a Random Walk along Memory. In Proceedings of the International Conference on Machine Learning.

Tokmakov, P.; Li, J.; Burgard, W.; and Gaidon, A. 2021. Learning to Track with Object Permanence. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Van Hoorick, B.; Tokmakov, P.; Stent, S.; Li, J.; and Vondrick, C. 2023. Tracking through Containers and Occluders in the Wild. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Wu, S.-C.; Wald, J.; Tateno, K.; Navab, N.; and Tombari, F. 2021. SceneGraphFusion: Incremental 3D Scene Graph Prediction from RGB-D Sequences. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Yang, J.; Cen, J.; Peng, W.; Liu, S.; Hong, F.; Li, X.; Zhou, K.; Chen, Q.; and Liu, Z. 2023. 4D Panoptic Scene Graph Generation. In Advances in Neural Information Processing Systems.