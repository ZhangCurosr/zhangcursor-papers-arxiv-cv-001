# Beyond End-Task Success: How to Audit Visual Experience Retrieval in Robotics

Eshika Pathak<sup>∗†</sup> and Leela Krishna<sup>†</sup>

<sup>∗</sup>University of Illinois Urbana-Champaign <sup>†</sup>Centific

Abstract—Robots that store past experiences must select which one to reuse in a new scene. Most systems select by visual similarity, and most evaluations report only the success of the selected experience. That number does not show whether the selection was good: a rule can score well by repeatedly using one broadly transferable experience, or poorly because its preferred experience is weak. This matters because robots increasingly adapt by reuse rather than retraining: every such adaptation passes through this selection step, and a score that describes the library rather than the rule misleads what the field builds next. We contribute an audit methodology: execute every stored experience in every query scene, yielding the complete table of transfer outcomes, over two manipulation tasks, three reuse mechanisms, and libraries of $K = 3 ,$ 10, and 50. The complete table is what makes the confound measurable: every alternative’s outcome is known, so a score can be traced to per-scene selection or to library quality. The audited rules select by nearest-neighbor distance in five visual embeddings, from raw pixels to CLIP. We see four results. (1) One fixed experience, chosen with hindsight, captures 30–58% of the gap between random selection and an oracle; per-scene selection competes for the remaining 0.07–0.15 in success rate. (2) At $K \geq 1 0 ,$ visual rules concentrate on one experience 1.5–3 times more than the oracle does, and their scores then follow that experience’s quality. (3) Wherever a rule differs significantly from a shuffle that keeps its selection rates but pairs them with scenes at random, the rule is worse, for every learned image policy. (4) Visual distance predicts well whether a given pair will succeed (AUROC up to 0.96), yet ranks the candidates within one scene no better than chance for four of five embeddings at $K = 5 0 \left( { \bf A U R O C 0 . 4 5 { - } 0 . 5 2 } \right)$ . Exhaustive execution is usually infeasible, so the audit reduces to two cheap reports any study can give: the distribution of selected experiences, and the success of the best single experience in hindsight.

## I. INTRODUCTION

Robots increasingly store demonstrations, trajectories, and policies for reuse. Reuse is becoming how robots adapt in deployment: a retrieved demonstration seeds few-shot imitation, a similar past episode conditions a policy, a stored skill is executed directly. Each of these passes through one selection step. Given a new scene, the robot must select which stored experience to use. A common rule is nearest-neighbor retrieval: embed the current scene, embed each stored scene, and select the experience whose stored scene is closest [1], [2], [3]. The rule assumes that visually similar scenes call for the same experience.

We study how to evaluate this selection. For a query scene $q$ and a stored experience $e _ { i } ,$ define the transfer probability

$$
T ( i , q ) = P ( \operatorname { s u c c e s s } \mid e _ { i } , q ) .\tag{1}
$$

An ideal retriever selects arg max<sub>i</sub> $T ( i , q )$ . A similarity rule instead selects arg min<sub>i</sub> $\mathinner { l \mathopen { \left( f \mathopen { \left( q \right) } , f \mathopen { \left( e _ { i } \right) } \right) } }$ , where f is an embedding and d a distance. The audit asks how well the second quantity approximates the first.

Standard evaluations execute only the selected experience and report its success. This number mixes two things: the quality of the decision and the quality of the library. A rule can score well by always selecting one strong experience. A rule can score poorly because it keeps selecting one weak experience. In both cases the score describes an experience, not the rule.

We therefore ask a methodological question: how can we audit whether a robot retriever actually chose well? Our an swer is to measure the alternatives. In simulation, we execute every stored experience in every query scene. This yields the complete transfer matrix of outcomes, which no ordinary deployment can observe. We also vary scene appearance (lighting, colour, texture, camera pose) independently of scene layout (object poses, articulation, obstacles). The physics is identical across appearance changes, so appearance effects and layout effects can be separated.

The audit has one central finding. Predicting transfer and selecting an experience are different problems. Visual distance predicts well whether a given experience-scene pair will succeed. It ranks poorly which of the K candidates to use for a given scene. The reason is that most of the outcome variance is shared within a scene: a hard scene is hard for every candidate. A score can detect hard scenes and still not separate the candidates. This one distinction explains three observations that would otherwise conflict: visual rules concentrate on few experiences, equalizing library quality restores positive discrimination, and high pooled prediction accuracy coexists with negative selection value.

The contribution is methodological. We propose no new retrieval method, and we claim no general verdict on visual similarity. We propose a way to measure whether a retriever chose well, and we show that this measurement changes the conclusions an ordinary evaluation would draw, including several of our own.

Contributions. (1) An audit protocol that scores retrieval against the complete transfer matrix, not only the selected trajectory. (2) Direct measurements separating pooled transfer prediction from within-query ranking, showing that the first does not imply the second. (3) Controls that isolate the effect of library composition: an oracle concentration reference, a best-fixed reference, frequency-matched shuffles, random sublibraries, and quality-equalized libraries. (4) A directional result for learned image policies: whenever visual selection differs significantly from the frequency-matched baseline, it is worse. (5) Two reporting diagnostics for ordinary studies: the selection distribution and the best-fixed reference.

## II. AUDIT PROTOCOL

Tasks and scenes. We use Stack and Door in robosuite [4] with a Panda arm under pose control. A scripted expert records each stored experience in its own base scene. Each query set contains 40 physical states. The states span four layout levels: object displacement up to ±12 cm, relative displacement or articulation, and obstacles. Each state is rendered under five appearance draws. Appearance draws change lighting, colour, texture, and small camera rotations. They do not change physics: the simulator state hash is identical across the five renders of each layout.

Reuse mechanisms. We evaluate three ways to reuse an experience. Replay retargets the recorded end-effector path to the current object poses and executes it open-loop [5]. Replay never reads the camera, so appearance cannot affect it. This is confirmed in the data: 0 of 600 labelled cases change across renders. Policy trains one behavior-cloned image policy per experience (BC-RNN [6]). Both appearance and layout affect a Policy execution. Policy+DR trains the same policy class with randomized lighting and texture. On Stack, this reduces the share of failures caused by appearance from 26% to 2%, and layout-caused failures remain matched pair by pair. On Door, domain randomization degrades layout performance by a measured amount.

Ground truth. We execute every pair $( e _ { i } , q )$ with five seeds and record the success rate $y _ { i q } .$ . The result is an answer that ordinary deployment cannot see: for each scene, the outcome of the selected experience, of every experience not selected, of random selection, and of oracle selection. The study uses 114,500 seeded executions plus 250 own-base replays. The unit of analysis is the transfer matrix, not the execution count.

Selection rules. We test nearest-neighbor retrieval in five embeddings: raw $3 2 \times 3 2$ pixels, ImageNet ResNet-18, DI-NOv2 [7], CLIP [8], and the Policy+DR visual encoder. We also test recency and two structural scores: object-position proximity and path clearance. Finally, a track-record rule ranks experiences by their outcomes on past scenes. Track record uses outcome data that a zero-shot visual rule does not have. We use it only as a reference for how quickly outcome data can find a strong experience.

Metrics. We report three quantities. First, the normalized advantage

$$
A = \frac { S _ { \mathrm { r u l e } } - S _ { \mathrm { r a n d o m } } } { S _ { \mathrm { o r a c l e } } - S _ { \mathrm { r a n d o m } } } ,\tag{2}
$$

where S is mean success: A=0 matches random selection and A=1 matches the oracle. Second, raw success rates. Third, the discrimination gap. To compute it, we generate 2,000 shuffled rules. Each shuffled rule selects each experience exactly as often as the real rule, but assigns those selections to scenes at random. The gap is the real rule’s success minus the mean shuffled success. A positive gap means the rule’s scene assignment adds value beyond its selection frequencies. Intervals are clustered by scene (n=40). We omit A when its denominator falls below a threshold fixed in advance.

TABLE I: Best-fixed advantage, oracle and random success, headroom (oracle minus best-fixed success), and the oracle’s top-pick share (the concentration reference of Sec. IV).
<table><tr><td>set</td><td></td><td>K best-fixed</td><td>oracle</td><td>random</td><td>headroom</td><td>oracle modal</td></tr><tr><td>Stack Replay, orig.</td><td>3</td><td>.58</td><td>.80</td><td>.53</td><td>.110</td><td>.45</td></tr><tr><td>Stack Replay, fresh</td><td>3</td><td>.45</td><td>.88</td><td>.63</td><td>.135</td><td>.38</td></tr><tr><td>Stack Replay, A</td><td>10</td><td>.45</td><td>.82</td><td>.55</td><td>.150</td><td>.20</td></tr><tr><td>Stack Replay, B</td><td>10</td><td>.51</td><td>.86</td><td>.62</td><td>.115</td><td>.20</td></tr><tr><td>Stack Replay</td><td>50</td><td>.51</td><td>.83</td><td>.56</td><td>.130</td><td>.10</td></tr><tr><td>Door P/+DR, orig.</td><td>3</td><td>.36/.55</td><td>.781.77</td><td>.64/.61</td><td>.088/.068</td><td>.41/.50</td></tr><tr><td>Door P/+DR, fresh</td><td>3</td><td>.30/.40</td><td>.81/.82</td><td>.66/.67</td><td>.103/.092</td><td>.47/.50</td></tr></table>

## III. HOW MUCH CAN SELECTION ADD?

Selection matters only if different scenes need different experiences. We first measure how much room the benchmark leaves for this.

Define best-fixed as the single experience with the highest mean success on the query set, chosen after seeing all outcomes. Best-fixed uses no per-scene information. It is not a deployable method; it is a reference.

Best-fixed is strong (Table I). It captures 30–58% of the gap between random and oracle selection in every stable setting; in the advantage metric, $A = 0 . 3 0 { - } 0 . 5 8$ (Table I). No tested rule beats it with non-overlapping intervals, and point excesses are at most 0.04. The intervals are wide, so the correct conclusion is that no rule is distinguishable from best-fixed, not that all rules equal it.

The headroom above best-fixed is small (Table I, headroom column). The oracle exceeds best-fixed by only 0.07–0.15 in success rate, in every stable setting. The headroom does not grow with library size: at K=50 it is 0.130, below our advance prediction of at least 0.15. The conclusion is direct. In these libraries, the main opportunity is not to match experiences to scenes. It is to find the experiences that transfer broadly. Section VI shows the same fact from the variance side.

Outcome data finds strong experiences quickly. The trackrecord rule selects the best or second-best experience in every set, and it locks onto a strong experience after about six scenes. On fresh image-policy scenes it matches or beats every visual rule in the warm-start setting. At K=50 it still trails bestfixed (A=0.26 versus 0.51): estimating 50 candidates from 40 scenes has an exploration cost. Track record is not a fair competitor to visual retrieval, because it sees outcomes. Its role here is to show that a little outcome data identifies the quantity that matters most, the mean success of each experience.

## IV. CONCENTRATION: THE SCORE FOLLOWS THE TOP PICK

Visual rules keep selecting the same experience. We call the experience a rule selects most often its top pick. This section measures how concentrated the rules are, and what the top pick does to the score.

TABLE II: The policy encoder on Stack Replay. Its advantage follows the quality of its top pick.
<table><tr><td>set</td><td>top pick</td><td>advantage</td></tr><tr><td> $K = 3 , \operatorname { o r i g } .$ </td><td>below library mean</td><td> $- 0 . 7 7$ </td></tr><tr><td> $K { = } 3 , \mathrm { f r e s h }$ </td><td>below library mean</td><td> $- 0 . 7 3$ </td></tr><tr><td> $K { = } 1 0 , \mathrm { { A } }$ </td><td>above library mean</td><td> $+ 0 . 2 9$ </td></tr><tr><td> $K = 1 0 , \mathrm { ~ B ~ }$ </td><td>above library mean</td><td> $+ 0 . 3 2$ </td></tr><tr><td> $K { = } 5 0$ </td><td>weakest in library (mean 0.28)</td><td> $- 0 . 5 7$ </td></tr></table>

Concentration must be compared to the right reference. The oracle itself is concentrated, because several experiences often tie for the best outcome. The oracle’s top-pick share is 0.38– $0 . 5 2$ at $K { = } 3 , \ 0 . 2 0$ at $K { = } 1 0$ , and 0.10 at $K { = } 5 0$ (Table I, last column). The visual rules’ top-pick share is 0.50–0.68 in several $K { = } 3$ settings, $0 . 3 0 – 0 . 6 0$ at $K { = } 1 0 .$ , and 0.25–0.30 at $K { = } 5 0$ . Against the oracle reference, the excess at $K { = } 3$ is small and often absent. At $K \geq 1 0$ it is clear: visual rules are 1.5–3 times as concentrated as the oracle (Fig. 3, left).

Concentration is costly when the top pick is weak. Define the quality margin of an experience as its mean success minus the library mean. If a score merely follows the top pick, the sign of the advantage A should match the sign of the top pick’s margin: a strong top pick gives $A > 0 ,$ , a weak one gives $A \ < \ 0 .$ . We tested this across the whole benchmark. A setting is one combination of task, reuse mechanism, and library; 13 settings have a denominator large enough to report A (Sec. II). Seven rules report A in each of them: the five visual embeddings and the two structural scores. That gives $7 \times 1 3 = 9 1$ cases. The signs match in 69 of the 91. The policy encoder is the clearest example (Table II). Its advantage is negative when its top pick is weak, positive when its top pick is strong, and negative again at $K { = } 5 0$ , where its top pick is the weakest experience in the library. We predicted the signs of the two $K { = } 1 0$ rows in advance, from the top pick’s quality alone. The pattern is not a law. Raw pixels at $K { = } 1 0$ are a counterexample: their top pick is weak, yet they select it mainly on the scenes where it works.

Does a rule’s scene assignment add anything beyond its frequencies? The discrimination gap answers this. Figure 1 shows each visual rule’s gap with the 95% envelope of its frequency-matched shuffles. In 42 of 65 cases, the rule lies inside the envelope. This does not prove those rules are equivalent to frequency-matched random assignment. With 40 scenes the envelope is wide, and we have not measured its power to detect small gaps. It shows compatibility, nothing stronger.

The cases outside the envelope are the informative ones. Under Replay, nine are positive and two are negative. So visual assignment can add per-scene value even when selections are concentrated. Under the learned image policies, the result is one-directional: all twelve significant Policy and Policy+DR cases lie below the envelope. For these policies, similarity assigns the selections to the wrong scenes. Holding the frequencies fixed, random assignment would have scored higher.

Why would similarity assign scenes wrongly? The appearance/layout design suggests an answer. Image policies fail under appearance changes. Generic encoders mostly measure appearance. So a similarity score can flag which scenes are hard, without saying which stored policy is safest there. We fixed a test of this in advance: recompute the gap separately on appearance-perturbed scenes and on layout-perturbed scenes, for the twelve significant cases. The negative gap is larger on the appearance side in 9 of 12 cases. The three exceptions are the fresh-scene Door Policy+DR cases, where the smaller layout subsets $\scriptstyle ( n = 3 0 )$ carry the larger negative gap with wide intervals. The mechanism holds for most cases, not all.

![](images/a0edb03e8e37ae3caca303e56aae0499e7c0d9ec2a0227ad4140493168dd30db.jpg)  
Fig. 1: Discrimination gap versus top-pick share for the visual rules. Error bars show the 95% envelope of frequency-matched shuffles. Red points lie outside the envelope. The gap shows no clear relation to concentration: concentration measures how often a rule repeats its top pick, not whether its scene assignment helps.

## V. THE LIBRARY CHANGES WHAT THE SCORE MEANS

The same rule gets different scores in different libraries. This section measures that dependence with three labelinformed procedures. All three are measurement tools, not methods: each uses outcome data that a deployed retriever would not have. Estimating experience quality without exhaustive labels remains open.

Equalize the library. We build sub-libraries whose experiences have nearly equal mean success. Labels choose the sub-library; the rule never sees them. At K=10, equalization uncovers discrimination that unequal libraries hide. Four of five visual rules become positive, with top-pick quality margins near zero; the policy encoder stays negative (Table III, top). The same rules score 0.12–0.21 lower in the unequal libraries. At $K { = } 3 ,$ , the equalized advantages are near zero. We do not conclude that equalization removes the signal there: A is unstable when its denominator is small, and we could not confirm the denominator in those sets. The supported statement is narrower. At $K { = } 1 0$ , real discrimination survives once library quality is equalized.

Re-rank the scores. Hubness corrections adjust similarity scores to reduce the dominance of frequent nearest neighbors [9]. Untuned CSLS [10] and mutual proximity [11] barely move the results (Table III, middle). Their effect is smaller than the effect of equalizing the library.

TABLE III: Effect of interventions on the advantage A. Equalization and deletion use labels; they are diagnostics, not methods.
<table><tr><td>intervention</td><td>setting</td><td>effect on A</td></tr><tr><td>equalize library</td><td>K=10</td><td>ResNet +0.22, DINOv2 +0.18, raw +0.13, CLIP +0.10, policy enc. −0.09 (unequal libraries: 0.12–0.21 lower)</td></tr><tr><td>re-rank scores (CSLS, mutual prox.) random</td><td>K=50  $K { = } 1 0$ </td><td>at most 0.08; concentration unchanged mean absolute effect 0.06–0.14</td></tr><tr><td>delete top pick</td><td> $K { = } 5 0$ </td><td> $\mathrm { C L I P - 0 . 4 2  + 0 . 1 5 }$   $\mathrm { p o l i c y ~ e n c . ~ } - 0 . 5 7  + 0 . 3 6$ </td></tr></table>

![](images/23530164ac84d8817a077198a0cdd05e8a179f71ef68be7d5af9a8978f136402.jpg)  
Fig. 2: Own-base success versus mean cross-scene success for the 50 Replay experiences. The two are uncorrelated $( \rho = - 0 . 0 4 )$ . Seven experiences have own-base success $\ge 0 . 8$ and cross-scene success $\leq 0 . 4 5$

Delete the top pick. Removing one experience, the rule’s top pick, flips two rules from negative to positive at $K { = } 5 0$ (Table III, bottom). Deletion uses labels, so it is not a method. But it locates the failure. The failure lives in the pairing of a concentrated rule with one weak experience. It is not a geometric artifact that re-ranking can fix.

Local success does not imply transfer. Figure 2 shows a library problem that exists before any retrieval. Of the 50 Replay experiences, 49 repeat their own base scene with success at least 0.8 (mean 0.92). Yet own-base success does not predict cross-scene success: the correlation is $\rho { = } - 0 . 0 4$ Seven experiences are reliable at home (own-base $\geq 0 . 8 )$ and weak elsewhere (cross-scene ≤ 0.45). The standard acceptance test for a demonstration asks whether it works where it was collected. That test does not measure what reuse depends on, which is how well the demonstration transfers.

## VI. PREDICTING TRANSFER IS NOT RANKING CANDIDATES

The results so far pose a question. If visual selection is weak, does visual distance carry no information? It carries a lot. This section shows where the information goes.

We treat the negated distance $- d ( f ( q ) , f ( e _ { i } ) )$ as a score for whether the pair $( e _ { i } , q )$ succeeds, and we measure the area under the ROC curve (AUROC). AUROC is the probability that a random successful pair gets a smaller distance than a random failed pair: 0.5 is chance, 1 is perfect. Pooled over all pairs, the prediction is good for image policies: AUROC reaches 0.85–0.96 under appearance perturbations and 0.70– 0.91 under layout perturbations (see also Table IV). Replay values are lower, as its causal structure requires: appearance cannot affect an open-loop replay.

TABLE IV: Pooled versus within-query AUROC (ranges over the visual representations). Within-query is lower in 62 of 65 eligible cases.
<table><tr><td>setting</td><td>pooled</td><td>within-query</td></tr><tr><td>Stack Replay,  $K { = } 1 0$ </td><td> $0 . 7 0 – 0 . 8 0$ </td><td>0.35–0.71</td></tr><tr><td>Stack Replay,  $K { = } 5 0$ </td><td>0.71-0.76</td><td>0.45–0.52 (all but ResNet)</td></tr><tr><td>Image policy,  $K { = } 3$ </td><td>0.68-0.94</td><td>0.26–0.77 (7–19% eligible)</td></tr></table>

Pooled AUROC is not selection accuracy. The reason is a decomposition:

$$
T ( i , q ) = \underbrace { g ( q ) } _ { \mathrm { s c e n e ~ d i f f c u l t y } } + \underbrace { h ( i , q ) } _ { \mathrm { c a n d i d a t e - s p e c i f i c } } + \epsilon .\tag{3}
$$

A hard scene lowers T for every candidate at once. A score that tracks $g ( q )$ predicts many pair outcomes correctly. But selection never compares across scenes. Selection compares the K candidates within one scene, and for that only $h ( i , q )$ matters.

So we measure ranking within the scene: for each scene $q ,$ whether $- d ( f ( q ) , f ( e _ { i } ) )$ ranks $T ( i , q )$ across the K candidates. For each scene, candidates with $y \geq 0 . 8$ are positives and $y \leq 0 . 2$ are negatives; scenes without both are excluded; per-scene AUROC is averaged with scene-clustered intervals. We fixed this rule in advance and recomputed pooled AUROC with the same labels for comparison. The result is consistent (Table IV): within-query AUROC is below pooled AUROC in 62 of 65 eligible cases. The gap is widest at $K { = } 5 0 { : }$ pooled AUROC stays above 0.7, while within-query AUROC is at or near chance for every representation except ResNet. The $K { = } 3$ image-policy rows carry a caveat. Only 7–19% of those scenes are eligible, and with one positive and one negative a per-scene AUROC can only be 0, <sup>1</sup> , or 1.

A variance decomposition says where the mismatch is worst. Let $g ( q )$ be the mean of $T ( i , q )$ over the K candidates, and let $R _ { q } ^ { 2 }$ be the share of the total variance of T explained by $g ( q )$ . Large $R _ { q } ^ { 2 }$ means scene difficulty dominates; small $R _ { q } ^ { 2 }$ means the candidates genuinely differ. We predicted $R _ { q } ^ { 2 } > 0 . 5$ everywhere. The prediction failed in exactly two settings, the Stack Replay $K { = } 3$ sets, where $R _ { q } ^ { 2 }$ is $0 . 4 2$ and 0.45. Those are also the settings with the strongest positive visual-selection results. Everywhere else the prediction held: $R _ { q } ^ { 2 }$ is 0.52–0.57 for Replay at $K \geq 1 0$ , and 0.71–0.81 for the image policies, exactly where all twelve significant gaps are negative. So $R _ { q } ^ { 2 }$ varies across settings in step with where visual selection works. Our advance prediction covered only the 0.5 threshold, not this ordering, so we report the ordering as supporting evidence, not a confirmed relationship.

A second check agrees. ResNet is the only rule with a positive gap at every Replay library size (Sec. VII). It is also the only representation above chance in within-query AUROC at K=50. CLIP and the policy encoder show the reverse: large advantages in magnitude at $K { = } 5 0 \ ( - 0 . 4 2 , - 0 . 5 7 )$ with within-query AUROC near 0.5. Two different summaries of the matrix single out the same representation.

The design target follows. A retrieval representation does not need to predict which scenes are hard. It needs to order the K candidates within one scene by transferability. Selectors should be trained and evaluated with query-grouped ranking objectives, not only with pooled similarity.

The same distinction recurs beyond retrieval. World foundation models for physical AI [12] are trained and benchmarked on prediction fidelity, while control requires the conditional analogue: ranking candidate actions or plans by outcome within a single state. Our results do not evaluate such systems, but the audit template transfers directly: pooled predictive skill should not be reported as evidence of decision-relevant ranking until the within-state statistic is measured.

## VII. LIBRARY SIZE CHANGES THE CONCLUSION

The same rule supports opposite conclusions at different library sizes (Fig. 3). The policy encoder’s gap is −0.79 at K=3, +0.35 and $+ 0 . 4 7$ at $K { = } 1 0$ , and −0.28 at K=50 (Fig. 3, middle). We predicted that the positive K=10 gaps would persist at K=50. That held for one rule of four. ResNet is the only rule positive at all three sizes. We report that as a fact about one rule.

Two caveats limit this section. First, the nested libraries extend one recording stream, and the expert’s rejection rate changes with size (0.40, 0.33, 0.153 at K=3, 10, 50). Size is therefore confounded with recording quality, and the trend is not a scaling law. Random sub-libraries address the concentration part of this concern: excess concentration appears in 96–99% of 200 random K=10 compositions, with the identity of the top pick varying widely. Those sub-libraries still share one pool and one query set. Independent recording streams are the stronger test.

Second, the size sensitivity changes how to read our smalllibrary results. Two early conclusions at K=3 did not survive at K=10: that recency beat every visual rule, and that one encoder was negatively predictive in general. All image-policy experiments here use K=3. The learned-policy results are therefore evidence about small libraries, not about scaling.

## VIII. TWO REPORTS EVERY RETRIEVAL STUDY CAN GIVE

Exhaustive execution is usually infeasible, so the methodology must reduce to cheap reports. The question a reader needs answered is simple. Does the score reflect per-scene selection, or the quality of one repeatedly selected experience? Two reports answer most of it, and any retrieval study can give them.

Report the selection distribution. State how often each experience was selected. This is free: it is a log of the evaluation the study already runs. If exhaustive labels exist, compare against the oracle’s distribution, not against uniform. Heavy concentration warns that the score may describe a few experiences.

TABLE V: Reporting audit of seven retrieval systems. Inclusion criterion: a stored demonstration, trajectory, skill, or training example is selected by an explicit rule before reuse. “Partial” means a distribution over source categories, not per-item selections.
<table><tr><td>system</td><td>pick dist. best-fixed what it does report</td><td></td><td></td></tr><tr><td>VINN [1]</td><td>no</td><td>no</td><td>downstream success vs. baselines</td></tr><tr><td>DINOBot [2]</td><td>no</td><td>no</td><td>per-task success, ablations</td></tr><tr><td>R+X [14]</td><td>no</td><td>no</td><td>success; varies retrieved-clip count</td></tr><tr><td>RT-Cache [3]</td><td>no</td><td>no</td><td>success and operation time; varies snippet horizon N; latency</td></tr><tr><td>Behavior Retrieval [15]</td><td>no</td><td>no</td><td>relevance separation vs. GT labels; GT same-task filter</td></tr><tr><td>FlowRetrieval [16]</td><td>partial</td><td>no</td><td>composition: useful / adversarial / non-harmful</td></tr><tr><td>STRAP [17]</td><td>partial</td><td>no</td><td>source-task distribution; varies K</td></tr></table>

Report best-fixed. State the success of always using the single best experience, chosen in hindsight. This does not require the full transfer matrix. It requires only the mean success of each library item on the evaluation scenes: K times the rollouts of the standard evaluation, with no per-scene resolution and no repeated seeds. In simulation this is routine. On hardware, a subsample of scenes per item suffices, because identifying the best item needs only rough means. Best-fixed is not a competitor. It shows how much success needed no perscene selection at all, and how much headroom the retriever can claim.

Table V shows the current gap. In seven surveyed systems, the two reports never appear together. The two partial cases report the composition of retrieved sources, not per-item selections. This is a reporting gap, not a claim that prior results are wrong. One scope note: the two reports are defined for top-1 selection, while Behavior Retrieval, FlowRetrieval, STRAP, and arguably R+X retrieve a set of data to train on. For set retrieval the analogues are the distribution over retrieved items and training on a single hindsight-chosen subset; the entries for these systems mark reports that would need this translation, not simple omissions. Exhaustive measurement of alternatives has precedent: SRSA [13] executes every source policy on every prior task to train a transfer predictor, and reports a per-task oracle reference, in a state-based assembly setting with policylevel retrieval. It reports neither the selection distribution nor a best-fixed reference, which is the gap the two reports close.

The two reports also give a way to read any new result. A rule that spreads its selections and beats best-fixed has shown per-scene skill. A rule that concentrates and lands near bestfixed has shown that one experience is good. A rule below its frequency-matched baseline, like the image-policy cases here, is assigning its selections to the wrong scenes.

Where trial history exists, the next step is practical: estimate each experience’s cross-scene quality before selecting per scene. Collection-time success is not that estimate (Fig. 2). Future work should pair quality estimation with query-conditional ranking, and test both on independent libraries, larger policy libraries, more tasks, and hardware. The audit template also extends beyond retrieval: any system benchmarked on prediction fidelity, including world foundation models, can be tested for the within-state ranking its decisions require.

![](images/775502fe193e18d94b0e7c305ae0b13d8151a14224a25dec2b00a55cbd09b213.jpg)

![](images/aefb7d08e6538eab8925aba345c8ff530ad7b1d6c8f2b1bb3411bce08b40e247.jpg)

![](images/a8bc80f76cf4bd37e7b6f0ebc4f0145633320d7ec00ee022ae13b385b3552156.jpg)  
Fig. 3: Stack Replay at K=3, 10, 50: top-pick share against the oracle reference, discrimination gap, and advantage against the best-fixed reference.

## IX. LIMITATIONS AND CONCLUSION

This is an audit of one benchmark. All K=50 results use Stack under Replay. All learned-policy results use $K { = } 3 ,$ one policy class, and three policies per level. Forty scenes give wide intervals. The frequency-matched envelope has no power analysis, so the 42 of 65 count shows compatibility, not equivalence. The random sub-libraries share one recording pool and one query set. The nested libraries differ in expert rejection rate. All experiments are in simulation. The variance decomposition supports the proposed mechanism but does not confirm it, and the $K { = } 3$ within-query values rest on few eligible scenes.

Within this scope, the conclusion is methodological. Endtask success does not show whether a retriever chose well. Applying the audit overturned conclusions that ordinary evaluation had suggested, including several of our own. A representation can predict which experience-scene pairs succeed and still fail to rank the candidates for one scene. When library quality is unequal, that failure makes the score follow the rule’s top pick. At $K \geq 1 0 .$ , visual rules are 1.5–3 times as concentrated as the oracle, and changing the library moves scores more than re-ranking does. For learned image policies, every significant departure from the frequency-matched baseline is a departure downward.

The recommendation costs little and changes how retrieval results are read. Report which experiences were selected. Report best-fixed. Where counterfactual labels exist, add the frequency-matched and within-query analyses. For future methods, the target is the capability this audit measured and found missing: ranking the candidates within one scene. At $K { = } 5 0$ , visual similarity performs that ranking no better than chance (AUROC 0.45–0.52; chance is 0.5) for four of the five embeddings we tested.

## REFERENCES

[1] J. Pari, N. M. Shafiullah, S. P. Arunachalam, and L. Pinto, “The surprising effectiveness of representation learning for visual imitation,” in Robotics: Science and Systems (RSS), 2022, arXiv:2112.01511.

[2] N. Di Palo and E. Johns, “DINOBot: Robot manipulation via retrieval and alignment with vision foundation models,” in IEEE International Conference on Robotics and Automation (ICRA), 2024, arXiv:2402.13181.

[3] O. Kwon, A. George, A. Bartsch, and A. Barati Farimani, “RT-Cache: Training-free retrieval for real-time manipulation,” in IEEE-RAS International Conference on Humanoid Robots (Humanoids), 2025, arXiv:2505.09040.

[4] Y. Zhu, J. Wong, A. Mandlekar, R. Mart´ın-Mart´ın, A. Joshi, S. Nasiriany, Y. Zhu, and K. Lin, “robosuite: A modular simulation framework and benchmark for robot learning,” arXiv preprint arXiv:2009.12293, 2020.

[5] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox, “MimicGen: A data generation system for scalable robot learning using human demonstrations,” in Conference on Robot Learning (CoRL), 2023, arXiv:2310.17596.

[6] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline human demonstrations for robot manipulation,” in Conference on Robot Learning (CoRL), 2021.

[7] M. Oquab, T. Darcet, T. Moutakanni, et al., “DINOv2: Learning robust visual features without supervision,” Transactions on Machine Learning Research, 2024.

[8] A. Radford, J. W. Kim, C. Hallacy, et al., “Learning transferable visual models from natural language supervision,” in International Conference on Machine Learning (ICML), 2021.

[9] M. Radovanovic, A. Nanopoulos, and M. Ivanovi ´ c, “Hubs in space:´ Popular nearest neighbors in high-dimensional data,” Journal ofMachine Learning Research, vol. 11, pp. 2487–2531, 2010.

[10] A. Conneau, G. Lample, M. Ranzato, L. Denoyer, and H. Jegou,´ “Word translation without parallel data,” in International Conference on Learning Representations (ICLR), 2018, arXiv:1710.04087.

[11] D. Schnitzer, A. Flexer, M. Schedl, and G. Widmer, “Local and global scaling reduce hubs in space,” Journal of Machine Learning Research, vol. 13, pp. 2871–2902, 2012.

[12] NVIDIA, N. Agarwal, et al., “Cosmos world foundation model platform for physical AI,” arXiv preprint arXiv:2501.03575, 2025.

[13] Y. Guo, B. Tang, I. Akinola, D. Fox, A. Gupta, and Y. Narang, “SRSA: Skill retrieval and adaptation for robotic assembly tasks,” in International Conference on Learning Representations (ICLR), 2025, arXiv:2503.04538.

[14] G. Papagiannis, N. Di Palo, P. Vitiello, and E. Johns, “R+X: Retrieval and execution from everyday human videos,” in IEEE International Conference on Robotics and Automation (ICRA), 2025, arXiv:2407.12957.

[15] M. Du, S. Nair, D. Sadigh, and C. Finn, “Behavior retrieval: Few-shot imitation learning by querying unlabeled datasets,” in Robotics: Science and Systems (RSS), 2023, arXiv:2304.08742.

[16] L.-H. Lin, Y. Cui, A. Xie, T. Hua, and D. Sadigh, “FlowRetrieval: Flowguided data retrieval for few-shot imitation learning,” in Conference on Robot Learning (CoRL), 2024, arXiv:2408.16944.

[17] M. Memmel, J. Berg, B. Chen, A. Gupta, and J. Francis, “STRAP: Robot sub-trajectory retrieval for augmented policy learning,” in International Conference on Learning Representations (ICLR), 2025, arXiv:2412.15182.