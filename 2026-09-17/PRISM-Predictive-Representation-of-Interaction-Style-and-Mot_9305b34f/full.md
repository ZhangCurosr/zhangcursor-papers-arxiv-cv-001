# PRISM: Predictive Representation of Interaction Style and Motion for Social Robot Navigation

Bo-Han Chen<sup>1</sup>, Hiromu Taketsugu<sup>2</sup>, and Norimichi Ukita<sup>2</sup>

<sup>1</sup> National Chung Hsing University 2 Toyota Technological Institute

Abstract. Humans often observe others before interacting and adjust their behavior accordingly. Robot navigation in crowds, however, often represents pedestrians mainly by observed geometric states, leaving individual diferences in interaction tendencies implicit. We propose PRISM (Predictive Representation of Interaction Style and Motion), a framework that infers interaction traits from passive observations of human-human interactions. PRISM encodes human trajectories into a continuous ordinal latent space with a transformer encoder trained by Rank-N-Contrast loss, and pairs each inferred trait with a temporal-stability score supplied to the navigation policy. In randomized crowd simulations, PRISM reduces collision rates over the geometry-only baseline and yields small improvements in navigation-time and path-length metrics. These results suggest the utility of passive latent-trait inference for social navigation in dynamic crowds.

Keywords: Social navigation · Interactive agents · Human-human interaction · Behavioral heterogeneity · Robot navigation

## 1 Introduction

As mobile robots move into human-centric environments, social acceptability becomes as important as eficiency [12, 20, 27]. A key dificulty is latent behavioral heterogeneity [26]: individuals difer in how strongly they yield, keep distance, or push forward in shared spaces. Current socially-aware navigation frameworks often represent pedestrians mainly through their trajectories represented by positions and velocities [3, 5, 15]. Such trajectories obtained by multi-object tracking [8, 25, 31] can contain useful cues, although individual yielding or assertive tendencies remain entangled with geometric motion. Active probing can identify interaction parameters by disturbing an agent [22], yet this violates the non-disturbance principle desirable in human-robot interaction.

We instead follow social eavesdropping [9, 21, 30], in which a robot passively observes human-human interactions and infers an agent’s Interaction Style from observed interaction histories without perturbing that agent. We propose PRISM (Predictive Representation of Interaction Style and Motion), which learns a continuous ordinal representation of interaction styles and provides it to a navigation policy. Figure 1 illustrates this passive observation setting.

![](images/95cc5e2b417af602cefa6bd10c16bcd54c051af593cdd652ee3867508bf684c7.jpg)  
Fig. 1: Overview. Colored pedestrian trajectories indicate observed interaction histories used to infer heterogeneous interaction styles, including unyielding and yielding behavior shown in red and green, respectively. (a) Without personalization, the robot relies on geometric states and may delay avoidance. (b) PRISM augments the navigation policy with inferred Interaction Style Vectors and confidence scores, producing a robot trajectory that better anticipates diferent yielding tendencies.

PRISM is designed for interactive agents in dynamic crowds, and encodes relative motion during encounters as continuous state variables, without probing pedestrians or requesting labels.

Our contributions are as follows: (1) a passive mechanism for inferring behavioral traits from trajectories using Rank-N-Contrast loss, (2) a temporal confidence estimate that is provided to the policy alongside the trait, and (3) an evaluation in heterogeneous crowd simulations showing improved collision-based safety metrics when our interaction-style representation is integrated into robot navigation.

## 2 Related Work

Early methods such as the Social Force Model [7] and Reciprocal Velocity Obstacles [2] use fixed interaction rules, whereas crowd-aware attention models [3, 5], intention-aware interaction graphs [14], and the Heterogeneous Interaction Graph Transformer (HEIGHT) [15] learn richer social dynamics. These methods use observed pedestrian trajectories for navigation and forecasting without exposing per-person latent traits to the policy.

Trajectory and motion prediction, such as Social-LSTM [1], Social-GAN [6], and related models [10, 13, 17–19, 24, 29], captures social interactions for anticipating pedestrian motion. Recent studies [11, 16] address style variation and distribution shift. These works target future-motion prediction, whereas PRISM infers an interaction-style variable.

Personalized prediction methods such as T4P [23], Interactive Adjustment [28], DisDis [4], meta-learning-based prediction [36], and MemoNet [34] adapt prediction to individual histories or memory. PRISM difers by using interaction history for personalization and learning a temporally consistent interaction-style representation for downstream planning. The goal is not to improve open-loop trajectory forecasting, but to provide a compact trait variable that a closed-loop navigation policy can use while observing agents.

![](images/23bc8d457da9ecef0f606102fbcaa01332168599022126f20e6013c9813f585d.jpg)  
Fig. 2: PRISM architecture. A transformer encoder maps trajectories $X _ { i }$ to an Interaction Style Vector $z _ { i } ,$ which is structured by RnC loss and decoded for stifness regression and input reconstruction.

## 3 Proposed Method

## 3.1 PRISM

Figure 2 shows the PRISM architecture. Given agents $\boldsymbol { S _ { t } } = \left\{ P _ { 0 } , P _ { 1 } , \ldots , P _ { K } \right\}$ ， $P _ { 0 }$ denotes the ego-agent, i.e., the robot, and $P _ { 1 } , \ldots , P _ { K }$ denote neighboring agents. For each $P _ { i }$ , PRISM constructs a trajectory history $\pmb { X } _ { i } \in \mathbb { R } ^ { T \times 4 }$ with $T = 3 0$ frames, corresponding to 3.0 seconds at 10 frames per second. Its feature at time step τ is $\pmb { x } _ { i , \tau } = [ p _ { i , \tau } ^ { x } , p _ { i , \tau } ^ { y } , v _ { i , \tau } ^ { x } , v _ { i , \tau } ^ { y } ] ^ { \top }$ , where $( p _ { i , \tau } ^ { x } , p _ { i , \tau } ^ { y } )$ and $( v _ { i , \tau } ^ { x } , v _ { i , \tau } ^ { y } )$ denote the position and velocity of $P _ { i }$ , respectively. The trajectories are represented in a local coordinate frame centered at the robot’s initial position and aligned with its initial velocity, providing invariance to global translation and rotation.

For a target neighbor $P _ { i }$ , the interaction context $\mathcal C _ { i } = \mathcal S _ { t } \setminus \{ P _ { i } \}$ consists of the robot and the other neighbors. A dual-stage Transformer encoder [33] first applies temporal self-attention to the target trajectory and then uses crossattention to incorporate the context trajectories. Invalid and padded context tokens are masked. Standard multi-head attention, positional encoding, residual connections, layer normalization, and feed-forward layers are used. The resulting contextualized features are average-pooled over time and projected by a two-layer multilayer perceptron with a GELU activation to obtain the Interaction Style Vector $z _ { i } \in \mathbb { R } ^ { D }$ , where $D = 1 2 8$

PRISM uses relative motion during encounters, including how agents approach, change velocity, and adjust their paths, as evidence of interaction style. It requires only the position and velocity histories available to the navigation system, without semantic interaction labels. PRISM processes the most recent $T = 3 0$ frames and updates the interaction-style estimate every ten control steps, holding it constant in between. Passive observation means that the robot does not perturb pedestrians or request behavior labels.

## 3.2 Rank-N-Contrast Supervision

In simulation, each agent has a social stifness label $\mu \in [ 0 , 1 ]$ , where low values indicate aggressive, weakly yielding behavior, and high values indicate cautious, strongly yielding behavior. A binary contrastive loss would require discretizing this trait, incorrectly separating similar agents near an arbitrary threshold. We use Rank-N-Contrast (RnC) loss [35] to encourage ordinal geometry.

For an anchor i in a mini-batch B, all other samples are sorted by label distance $| \mu _ { i } - \mu _ { k } |$ . Let $\varOmega _ { i } = \{ k _ { 1 } , k _ { 2 } , \dots \}$ be this sorted index set, ordered such that $| \mu _ { i } - \mu _ { k _ { 1 } } | \leq | \mu _ { i } - \mu _ { k _ { 2 } } | \leq \cdot \cdot \cdot$ . The RnC loss is

$$
\mathcal { L } _ { \mathrm { R n C } } = - \sum _ { i \in \mathcal { B } } \sum _ { m = 1 } ^ { | \Omega _ { i } | } \log \frac { \exp ( - \Vert z _ { i } - z _ { k _ { m } } \Vert _ { 2 } / \tau _ { \mathrm { R n C } } ) } { \sum _ { j = m } ^ { | \Omega _ { i } | } \exp ( - \Vert z _ { i } - z _ { k _ { j } } \Vert _ { 2 } / \tau _ { \mathrm { R n C } } ) } ,\tag{1}
$$

where $\tau _ { \mathrm { R n C } } = 2 . 0$ denotes the temperature parameter. This loss penalizes cases in which a sample with a dissimilar stifness score is embedded closer to the anchor than a sample with a similar score. It does not guarantee manifold smoothness. It provides a rank-based training signal that encourages nearby stifness values to have nearby embeddings and distant values to be separated. This structure is useful for reinforcement learning because the policy receives $z _ { i }$ as part of its state.

The encoder is trained with reconstruction and regression objectives:

$$
\mathcal { L } _ { \mathrm { { R e c o n } } } = | | \hat { X } _ { i } - X _ { i } | | _ { 2 } ^ { 2 } ,\tag{2}
$$

$$
\mathcal { L } _ { \mathrm { R e g } } = | | \hat { \mu } _ { i } - \mu _ { i } | | _ { 2 } ^ { 2 } ,\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { T o t a l } } = \lambda _ { \mathrm { R e c o n } } \mathcal { L } _ { \mathrm { R e c o n } } + \lambda _ { \mathrm { R n C } } \mathcal { L } _ { \mathrm { R n C } } + \lambda _ { \mathrm { R e g } } \mathcal { L } _ { \mathrm { R e g } } ,\tag{4}
$$

where ${ \hat { X } } _ { i }$ denotes the reconstructed trajectory, $\hat { \mu } _ { i }$ denotes the regression-head stifness prediction, and $\lambda _ { \mathrm { R e c o n } } = 1 . 0 , \lambda _ { \mathrm { R n C } } = 1 . 0$ , and $\lambda _ { \mathrm { R e g } } = 0 . 1$ . The regression head is used only during representation learning. During policy learning, the planner receives a projection of the latent vector and the confidence score defined later in Eq. (6), rather than the ground-truth simulator label, so the closed-loop policy does not receive privileged stifness labels during inference.

## 3.3 Temporal Uncertainty Estimation

To quantify the stability of its trait estimates, PRISM tracks embedding volatility with an exponentially weighted moving variance:

$$
\begin{array} { r } { \sigma _ { t } ^ { 2 } = ( 1 - \beta ) \sigma _ { t - 1 } ^ { 2 } + \beta \| \hat { z } _ { t } - z _ { t - 1 } \| _ { 2 } ^ { 2 } , } \end{array}\tag{5}
$$

where $\hat { z } _ { t }$ denotes the instantaneous encoder output, $z _ { t - 1 }$ denotes the previous smoothed estima ${ \mathrm { ; e , } }$ and $\beta = 0 . 1$ denotes the smoothing factor. This recursive estimator maintains a fading memory of embedding stability while emphasizing recent observations. The resulting volatility is mapped to a confidence score $\alpha _ { t } \in [ 0 , 1 ]$ using a radial basis function (RBF) kernel:

$$
\alpha _ { t } = \exp \left( - \sigma _ { t } ^ { 2 } / ( 2 \gamma ^ { 2 } ) \right) ,\tag{6}
$$

where $\gamma = 1 . 0$ denotes the RBF kernel width. A high volatility produces a small $\alpha _ { t } .$ , indicating that the style estimate is currently unstable. The planner receives the interaction-style representation together with $\alpha _ { t } .$ , allowing the navigation policy to account for its temporal stability. Before $T = 3 0$ frames of history are available, the policy receives a zero vector with $\alpha _ { t } = 0$ . The estimate is also held fixed while no neighbor is within 2.5 m, so that frames without a nearby agent do not perturb it.

Table 1: Phase 1 ablation study on latent representation.
<table><tr><td>Configuration</td><td colspan="3">Stiffness MAE↓ Manifold Rank  $( \rho ) \uparrow$  Recon. ADE (m) ↓</td></tr><tr><td>Recon Only</td><td>0.248</td><td>0.019</td><td>0.006</td></tr><tr><td>Reg Only</td><td>0.214</td><td>0.038</td><td>0.738</td></tr><tr><td>RnC Only</td><td>0.247</td><td>0.162</td><td>0.858</td></tr><tr><td> $\mathrm { R e c o n } + \mathrm { R e g }$ </td><td>0.207</td><td>0.043</td><td>0.011</td></tr><tr><td> $\mathrm { R n C } + \mathrm { R e g }$ </td><td>0.219</td><td>0.189</td><td>0.771</td></tr><tr><td> $\mathrm { R e c o n } + \mathrm { R n C }$ </td><td>0.247</td><td>0.166</td><td>0.028</td></tr><tr><td>Full Model (PRISM)</td><td>0.203</td><td>0.183</td><td>0.027</td></tr></table>

## 4 Experiments

## 4.1 Simulation Environment and Crowd Heterogeneity

We evaluate PRISM in a custom PyBullet simulator based on HEIGHT [15]. The arena is a 9m × 9m square with 8–12 randomly placed static obstacles modeled as furniture. The robot is a Turtlebot with non-holonomic kinematics and maximum linear velocity $v _ { \mathrm { m a x } } = 0 . 5 ~ \mathrm { m / s }$ . It observes the environment through simulated two-dimensional light detection and ranging (LiDAR) with a 360<sup>◦</sup> field of view (FOV) and a 25m range. This observation is used by the navigation policy only. PRISM instead consumes ground-truth pedestrian positions with no observation noise, and velocities are finite diferences of consecutive positions. Neighbor identity is provided by the simulator, so detection and tracking errors are outside the scope of this study. Dynamic pedestrians are governed by an Optimal Reciprocal Collision Avoidance (ORCA) policy [32]. ORCA provides controllable interactive pedestrian motion and interpretable parameters. Each agent has a latent social stifness $\mu _ { i } \in [ 0 , 1 ]$ that linearly interpolates its navigation parameters: $\phi _ { i } = \mu _ { i } \varPhi _ { \mathrm { c a u t i o u s } } + ( 1 - \mu _ { i } ) \varPhi _ { \mathrm { a g g r e s s i v e } }$ . Here, $\varPhi _ { i }$ comprises the ORCA time horizon, safety bufer, and neighbor-detection range. Cautious agents $( \mu = 1 )$ use a long time horizon $( T _ { \mathrm { h o r } } = 1 0 . 0 \mathrm { s } )$ , a safety bufer (0.15m), and a far detection range (10.0m). Aggressive agents $( \mu = 0 )$ use a short time horizon $( T _ { \mathrm { h o r } } = 2 . 0 \mathrm { s } )$ , zero safety bufer, and a 2.0m detection range. Note that $\mu$ is a controllable simulation attribute, not a directly observed label in real crowds.

## 4.2 Training and Evaluation

Training has two phases. First, the PRISM encoder is trained on $1 0 ^ { 5 }$ interaction frames using ground-truth $\mu$ for the RnC ordering signal. The regression and reconstruction heads are used only during this representation-learning phase. Second, the encoder is frozen and integrated into the HEIGHT navigation policy. The HEIGHT architecture and reward design follow the original paper [15].

For each neighbor $P _ { i }$ , PRISM augments the original HEIGHT geometric feature $h _ { i , t } ^ { \mathrm { g e o m } }$ with the inferred style vector and confidence score: $h _ { i , t } ^ { \mathrm { { \tiny ~ P R I S M } } } =$ $[ { h } _ { i , t } ^ { \mathrm { g e o m } } ; \tilde { z } _ { i , t } ; { \alpha _ { i , t } } ]$ , where $\tilde { z } _ { i , t } \in \mathbb { R } ^ { 8 }$ is a linear projection of $z _ { i , t } ,$ so each neighbor contributes nine additional values. While $h _ { i , t } ^ { \mathrm { P R I S M } }$ is augmented, the remaining HEIGHT architecture is unchanged. The frozen PRISM encoder is evaluated every 10 control steps, and policy-learning gradients are not propagated into it.

![](images/3f36faa4ef24d671b5e9750e356dd81c749d05afc452fb1226adb95ab902449b.jpg)  
(a) T4P

![](images/20d417ae74e32617098e7b632fee97f2c692e9e094bfb1662125499212154745.jpg)  
(b) Recon Only

![](images/d455345508ec2a92bac1db02597d74248a6a8822d7db7fba2753cc98a54c20a0.jpg)  
(c) Reg Only

![](images/bce9411287524417adbde748d11b3a1f98ffdf39cf21deafac07dcb52a626fdb.jpg)  
(d) RnC Only

![](images/095bc5b8357491c4c97da580d70bd00b4d4b2d6d287398d1e77dcbde8e5b15ae.jpg)  
(e) Recon + Reg

![](images/82c3c821ea1db2371e3d116015c47a198fb83917d5be0bf11dc9e0b8b02ef918.jpg)  
(f) RnC + Reg

![](images/d4680670a0ec987f3163526c4151237a38158668a6d69f88d6e7c661f0fdac75.jpg)

![](images/941043e78a1b516e6740213366924c7d68af1cb985ba3d04b4d1ae95995297b2.jpg)  
(g) Recon + RnC (h) Full Model (PRISM)  
Fig. 3: Latent embedding structure visualized by t-SNE. Embeddings are colored by ground-truth social stifness µ (µ ≈ 0: aggressive; µ ≈ 1: cautious). RnC-based configurations show a clearer ordinal gradient than reconstruction- or regression-only variants.

We evaluate 500 randomized episodes per setting. Success rate is the ratio of episodes in which the robot reaches its goal without collision. The overall collision rate is the ratio of episodes ending in collision, with human and obstacle collisions reported separately. The timeout rate is the ratio of episodes that neither reach the goal nor collide within the maximum duration.

## 4.3 Results: Latent Representation

Table 1 shows the quantitative trade-of among three requirements for an interactionstyle representation. It should predict the simulated stifness value, preserve motion information, and organize the latent space according to stifness ordering. Manifold Rank $\rho$ is the Spearman rank correlation (denoted by corr<sub>S</sub>) between all pairwise embedding distances and the corresponding pairwise stifness differences: $\rho = \mathrm { c o r r } _ { \mathrm { S } } ( \{ \| z _ { i } - z _ { j } \| _ { 2 } \} _ { i < j } , \{ | \mu _ { i } - \mu _ { j } | \} _ { i < j } )$ . Figure 3 provides the complementary geometric view of the same trade-of using t-SNE and coloring each point by the ground-truth ORCA stifness $\mu$ between 0 (blue) and 1 (red).

Figure 3 highlights the efect of RnC supervision. The T4P [23] representation in (a), Recon Only in (b), Reg Only in (c), and $\mathrm { R e c o n } + \mathrm { R e g }$ in (e) are weakly organized by color: blue and red samples remain broadly mixed. Table 1 gives the same conclusion quantitatively because their Manifold Rank values remain close to zero $( \rho = 0 . 0 1 9 , 0 . 0 3 8$ , and 0.043 for Recon Only, Reg Only, and Recon $+ \ \mathrm { R e g } )$ . Thus, accurate reconstruction or scalar regression alone does not create a latent geometry that reflects relative behavioral similarity. This is important because the policy needs a latent state structured by interaction-style similarity.

Table 2: Navigation results under diferent human and obstacle densities.
<table><tr><td rowspan="2">Environment</td><td rowspan="2">Method</td><td rowspan="2">Success↑</td><td colspan="3">Collision Rate↓</td><td rowspan="2"></td><td rowspan="2">Timeout↓ Nav Time (s)↓ Path Len (m)↓</td><td rowspan="2"></td></tr><tr><td></td><td>Overall w/ Humans w/ Obs</td><td></td></tr><tr><td>Training distribution</td><td>HEIGHT (Baseline)</td><td>0.81</td><td>0.19</td><td>0.13</td><td>0.06</td><td>0.00</td><td>18.78</td><td>10.30</td></tr><tr><td>(5-9 humans, 8-12 obs.)</td><td>PRISM (Ours)</td><td>0.88</td><td>0.12</td><td>0.08</td><td>0.04</td><td>0.00</td><td>18.08</td><td>10.20</td></tr><tr><td>Less crowded</td><td>HEIGHT (Baseline)</td><td>0.89</td><td>0.11</td><td>0.06</td><td>0.05</td><td>0.00</td><td>17.82</td><td>10.42</td></tr><tr><td>(0-4 humans, 8-12 obs.)</td><td>PRISM (Ours)</td><td>0.95</td><td>0.05</td><td>0.02</td><td>0.03</td><td>0.00</td><td>17.46</td><td>10.37</td></tr><tr><td>More crowded</td><td>HEIGHT (Baseline)</td><td>0.74</td><td>0.26</td><td>0.20</td><td>0.06</td><td>0.00</td><td>19.29</td><td>10.18</td></tr><tr><td>(10-14 humans, 8-12 obs.) PRISM</td><td>(Ours)</td><td>0.80</td><td>0.20</td><td>0.16</td><td>0.04</td><td>0.00</td><td>18.93</td><td>10.20</td></tr><tr><td>Less constrained (5-9 humans, 3-7 obs.)</td><td>HEIGHT (Baseline)</td><td>0.87</td><td>0.13</td><td>0.10</td><td>0.02</td><td>0.00</td><td>18.04</td><td>10.59</td></tr><tr><td></td><td>PRISM (Ours)</td><td>0.89</td><td>0.11</td><td>0.09</td><td>0.01</td><td>0.00</td><td>17.79</td><td>10.43</td></tr><tr><td>More constrained</td><td>HEIGHT (Baseline)</td><td>0.73</td><td>0.27</td><td>0.11</td><td>0.16</td><td>0.01</td><td>19.30</td><td>10.41</td></tr><tr><td>(5-9 humans, 13-17 obs.)</td><td>PRISM (Ours)</td><td>0.77</td><td>0.23</td><td>0.08</td><td>0.15</td><td>0.00</td><td>18.85</td><td>10.26</td></tr></table>

The panels that include RnC (i.e., Fig. 3 (d), (f), (g), and (h)) change this structure. They show a more coherent blue-to-red gradient. Table 1 reports a corresponding increase in Manifold Rank $( \rho > 0 . 1 6 )$ . This comparison indicates the role of RnC. It supplies the ordinal behavioral structure that is absent from reconstruction-only and regression-only objectives. However, RnC alone is not suficient for navigation. Figures 3 (d) and (f) show strong stifness ordering, whereas Table 1 shows poor Average Displacement Error (ADE) [6] for RnC Only and $\mathrm { R n C } + \mathrm { R e g }$ . They capture the stifness ordering of simulated agents, while losing much of the motion information needed for trajectory grounding.

The efectiveness of PRISM follows from the balance in Fig. 3 (h). Compared with Recon + RnC in (g), the full model adds regression supervision and obtains the best MAE of stifness (0.203). Compared with RnC + Reg in (f), it restores motion grounding, reducing reconstruction ADE from 0.771m to 0.027m. Compared with Recon + Reg in (e), it maintains an ordinal latent layout, raising Manifold Rank from 0.043 to 0.183. These linked comparisons show why all three objectives are used. RnC organizes the space by interaction style, reconstruction ties the style vector to observable trajectories, and regression stabilizes the stifness estimate. The full PRISM representation is therefore suitable as a compact state variable for downstream social navigation.

## 4.4 Results: Navigation Performance

Table 2 evaluates whether the representation in Table 1 and Fig. 3 improves closed-loop navigation when inserted into HEIGHT.

Across all five evaluation regimes, PRISM reduces the overall collision rate. In the training-distribution setting, the collision rate decreases from 0.19 to 0.12, a relative reduction of 37%, while success increases from 0.81 to 0.88. This consistency suggests that the inferred social stifness variable helps the robot adjust its behavior around agents with diferent yielding tendencies.

The obstacle-density tests show a similar pattern, though the gain is smaller under dense obstacles, as geometric bottlenecks cannot be fully resolved by interaction-style reasoning alone. Nevertheless, PRISM consistently improves or maintains performance across regimes where social and geometric constraints are entangled.

Eficiency metrics show a modest secondary benefit. Navigation time is lower across all settings. Path length is also lower in four of the five settings, likely because better anticipation of yielding or assertive pedestrians reduces late evasive actions. Since Table 2 reports aggregate rates and means without confidence intervals, these results should be viewed as empirical evidence in ORCA simulations rather than as a statistical significance claim. Even with this conservative interpretation, the results support the central claim that passively inferred heterogeneous interaction traits can improve reinforcement-learning-based robot navigation in dynamic crowds.

## 5 Discussion

PRISM treats crowds as dynamic multi-agent worlds in which pedestrians are interactive agents rather than passive obstacles. PRISM infers interaction styles from passively observed motion histories and uses them as state variables for closed-loop robot planning. This formulation is closely aligned with interactiveagent and embodied-AI settings, where an agent must reason about how other agents respond during interaction.

In the ORCA-based simulator, PRISM reduces collision rates across all five evaluation regimes and yields small changes in navigation time and path length. These results suggest that interaction-style augmentation can improve navigation beyond geometric observations alone. Since personal-space violations are penalized independently of the inferred style, the policy is not explicitly rewarded for exploiting pedestrians who tend to yield.

The current evidence is limited to synthetic heterogeneity defined by an ORCA parameter, whereas real behavior is context-dependent and multidimensional. The results also lack confidence intervals, independent PPO training seeds, and a closed-loop ablation of its components, including an oracle-stifness control. PRISM should therefore be viewed as a promising simulated interactionrepresentation module. Future work will evaluate it on real pedestrian data and learn multidimensional traits from naturally occurring interactions.

## 6 Conclusion

This paper presented PRISM, a framework for behavior-aware robot navigation through passive inference of interaction styles. PRISM uses Rank-N-Contrast loss to organize trajectory embeddings by a simulated social stifness parameter and injects the inferred representation into a robot navigation planner. Table 1 and Fig. 3 show a latent representation that is both ordered by interaction style and grounded in observed motion, and Table 2 shows improved collision-avoidance metrics over a geometry-only baseline. The present results do not solve the simto-real problem of obtaining stifness labels for real humans. Within controlled ORCA-based crowd simulations, however, PRISM demonstrates that passive latent-trait inference can support reinforcement-learning-based social navigation.

## References

1. Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., Savarese, S.: Social lstm: Human trajectory prediction in crowded spaces. In: CVPR (2016)

2. van den Berg, J., Lin, M.C., Manocha, D.: Reciprocal velocity obstacles for realtime multi-agent navigation. In: ICRA (2008)

3. Chen, C., Liu, Y., Kreiss, S., Alahi, A.: Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning. In: ICRA (2019)

4. Chen, G., Li, J., Zhou, N., Ren, L., Lu, J.: Personalized trajectory prediction via distribution discrimination. In: ICCV (2021)

5. Chen, Y.F., Everett, M., Liu, M., How, J.P.: Socially aware motion planning with deep reinforcement learning. In: IROS (2017)

6. Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., Alahi, A.: Social gan: Socially acceptable trajectories with generative adversarial networks. In: CVPR (2018)

7. Helbing, D., Molnár, P.: Social force model for pedestrian dynamics. Physical Review E 51(5), 4282–4286 (1995)

8. Hirano, S., Ukita, N.: Samidare: Advanced tracking-by-segmentation for dense scenarios. In: CVPRW (2026)

9. Johnstone, R.A.: Eavesdropping and animal conflict. Proceedings of the National Academy of Sciences 98(16), 9177–9180 (2001)

10. Kosaraju, V., Sadeghian, A., Martín-Martín, R., Reid, I., Rezatofighi, H., Savarese, S.: Social-bigat: Multimodal trajectory forecasting using bicycle-gan and graph attention networks. In: NeurIPS (2019)

11. Kothari, P., Li, D., Liu, Y., Alahi, A.: Motion style transfer: Modular low-rank adaptation for deep motion forecasting. In: CoRL (2022)

12. Kruse, T., Pandey, A.K., Alami, R., Kirsch, A.: Human-aware robot navigation: A survey. Robotics and Autonomous Systems 61(12), 1726–1743 (2013)

13. Lee, N., Choi, W., Vernaza, P., Choy, C.B., Torr, P.H.S., Chandraker, M.: DESIRE: Distant future prediction in dynamic scenes with interacting agents. In: CVPR (2017)

14. Liu, S., Chang, P., Huang, Z., Chakraborty, N., Hong, K., Liang, W., Livingston McPherson, D., Geng, J., Driggs-Campbell, K.: Intention aware robot crowd navigation with attention-based interaction graph. In: ICRA. pp. 12015– 12021 (2023)

15. Liu, S., Xia, H., Cheraghi Pouria, F., Hong, K., Chakraborty, N., Hu, Z., Biswas, J., Driggs-Campbell, K.: Height: Heterogeneous interaction graph transformer for robot navigation in crowded and constrained environments. IEEE Transactions on Automation Science and Engineering 23, 1211–1230 (2026)

16. Liu, Y., Cadei, R., Schweizer, J., Bahmani, S., Alahi, A.: Towards robust and adaptive motion forecasting: A causal representation perspective. In: CVPR (2022)

17. Maeda, T., Cao, J., Ukita, N., Kitani, K.: Cacheflow: Fast human motion prediction by cached normalizing flow. Trans. Mach. Learn. Res. 2026 (2026)

18. Maeda, T., Ukita, N.: Fast inference and update of probabilistic density estimation on trajectory prediction. In: ICCV (2023)

19. Mangalam, K., Girase, H., Agarwal, S., Lee, K.H., Adeli, E., Malik, J., Gaidon, A.: It is not the journey but the destination: Endpoint conditioned trajectory prediction. In: ECCV (2020)

20. Möller, R., Furnari, A., Battiato, S., Härmä, A., Farinella, G.M.: A survey on human-aware robot navigation. Robotics and Autonomous Systems 145, 103837 (2021)

21. Nowak, M.A., Sigmund, K.: Evolution of indirect reciprocity. Nature 437(7063), 1291–1298 (2005)

22. Pandya, R., Liu, C.: Safe and eficient exploration of human models during humanrobot interaction. In: IROS (2022)

23. Park, D., Jeong, J., Yoon, S.H., Jeong, J., Yoon, K.J.: T4p: Test-time training of trajectory prediction via masked autoencoder and actor-specific token memory. In: CVPR (2024)

24. Salzmann, T., Ivanovic, B., Chakravarty, P., Pavone, M.: Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data. In: ECCV (2020)

25. Shim, K., Ko, K., Yang, Y., Kim, C.: Focusing on tracks for online multi-object tracking. In: CVPR (2025)

26. Sih, A., Bell, A.M., Johnson, J.C., Ziemba, R.E.: Behavioral syndromes: an integrative overview. The Quarterly Review of Biology 79(3), 241–277 (2004)

27. Singamaneni, P.T., Bachiller-Burgos, P., Manso, L.J., Garrell, A., Sanfeliu, A., Spalanzani, A., Alami, R.: A survey on socially aware robot navigation: Taxonomy and future challenges. The International Journal of Robotics Research 43(10), 1533–1572 (2024)

28. Sun, J., Li, Y., Chai, L., Lu, C.: Interactive adjustment for human trajectory prediction with individual feedback. In: ICLR (2025)

29. Taketsugu, H., Oba, T., Maeda, T., Nobuhara, S., Ukita, N.: Physical plausibilityaware trajectory prediction via locomotion embodiment. In: CVPR (2025)

30. Tibbetts, E.A., Wong, E., Bonello, S.: Wasps use social eavesdropping to learn about individual rivals. Current Biology 30(15), 3007–3010.e2 (2020)

31. Ukita, N., Okada, A.: High-order framewise smoothness-constrained globallyoptimal tracking. Comput. Vis. Image Underst. 153, 130–142 (2016)

32. Van Den Berg, J., Guy, S.J., Lin, M., Manocha, D.: Optimal reciprocal collision avoidance for multi-agent navigation. In: ICRA (2010)

33. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: NIPS (2017)

34. Xu, C., Mao, W., Zhang, W., Chen, S.: Remember intentions: Retrospectivememory-based trajectory prediction. In: CVPR (2022)

35. Zha, K., Cao, P., Son, J., Yang, Y., Katabi, D.: Rank-n-contrast: Learning continuous representations for regression. In: NeurIPS (2023)

36. Zhu, H., Zhang, L., Fan, Z.: Personalized individual trajectory prediction via metalearning. In: SIGSPATIAL (2022)