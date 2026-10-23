# Asking Less, Adapting Better: Feedback-Efficient Test-Time Adaptation for Vision-and-Language Navigation

Zixu Wang<sup>∗</sup>, Huaquan Zhang<sup>∗</sup>, Shoukai Xu<sup>∗</sup>, Jinwu Hu, Zhiquan Wen<sup>†</sup>, Mingkui Tan, Member IEEE

Abstract—Vision-and-Language Navigation (VLN) requires agents to follow instructions and navigate to target locations. However, current VLN agents often suffer severe performance degradation in unseen environments due to distribution shifts. Test-Time Adaptation (TTA) offers a promising paradigm to bridge this gap, yet existing methods that rely on unsupervised entropy minimization can propagate errors through the trajectory, causing cascading failures in navigation. Recent methods mitigate this by incorporating human feedback, but they apply trajectory-level supervision uniformly across all steps without localizing the actions is truly informative, leading to inefficient feedback utilization and mismatched optimization granularity. To address these limitations, we propose FETA, a Feedback-Efficient Test-Time Adaptation method that improves both feedback efficiency and optimization reliability. To reduce unnecessary human interventions, we introduce a hierarchical sample selection strategy to localize informative samples. Specifically, we first filter trajectories whose normalized policy entropy exhibits large fluctuations. Within each selected trajectory, we further select the critical action via an entropy-gradient saliency criterion and query a single binary feedback on its correctness. A discrepancyaware entropy optimization objective then updates the policy only at the queried step, with adaptive scaling that assigns larger relative weights to samples with greater prediction–feedback discrepancy. Extensive experiments across multiple benchmarks demonstrate that FETA achieves state-of-the-art performance with fewer human interventions, while remaining robust under noisy feedback and continual adaptation settings.

Index Terms—Vision-and-Language Navigation, Test-Time Adaption, Active Learning.

## I. INTRODUCTION

damental embodied AI task in which an agent interprets natural-language instructions and navigate through a 3D environment to reach a target location. In practical applications, VLN agents are often deployed under conditions that differ from those seen during training. Such deployment shifts may arise from new room layouts, varying illumination or different instruction granularity, leading to substantial discrepancies between the training and test distributions [5–9]. As a result, even strong VLN models often suffer marked performance drop when transferred to new environments, making online adaptation crucial for robust embodied navigation [10–14].

Test-Time Adaptation (TTA) [15–18] offers a natural way to mitigate such distribution shifts by updating model parameters on incoming test data. Motivated by its success in static vision tasks such as image classification and semantic segmentation, recent studies have begun to extend TTA to VLN. However, existing methods [6, 19] typically rely on unsupervised objectives, such as entropy minimization, and perform policy updates at every step along the navigation trajectory. This strategy is particularly brittle in VLN, where navigation unfolds as a sequential decision-making process: an erroneous update at one step can change the subsequent state distribution, distort later optimization signals, and eventually cause cascading failures. Therefore, the key challenge in VLN test-time adaptation is not simply how to adapt online, but how to derive optimization signals that remain reliable throughout sequential decision-making.

To obtain more reliable optimization signals, recent VLN-TTA methods have begun to incorporate human feedback at test time. FeedTTA [20] queries human whether each trajectory is successful or not, and uses the feedback as a reward signal to update the policy via reinforcement learning. ATENA [7] requests feedback only for trajectories with high entropy, calibrating policy uncertainty by boosting decision confidence on successful trajectories and suppressing it on failed ones. Despite their promise, these methods still suffer from two key limitations. 1) Inefficient Feedback Utilization: Once a trajectory-level label (i.e., success or failure) is obtained, existing methods generally use every obtained label to update the policy on the corresponding trajectory, even when the resulting adaptation gains are limited. 2) Mismatched Optimization Granularity: Existing methods use the same trajectory-level feedback signal across all steps, even though successful trajectories may still contain suboptimal actions and failed ones may still include locally correct decisions. This creates a mismatch between trajectory-level supervision and local action decisions being optimized.

To address these challenges, we propose a Feedback-Efficient Test-Time Adaptation method (FETA) that improves both feedback efficiency and optimization reliability. Our key observation is that the choice of trajectories and steps for feedback acquisition affects adaptation performance (as shown in Fig. 2 and Table IV). Based on this observation, we design a hierarchical feedback selection strategy that improves both feedback efficiency and optimization granularity. At the trajectory level, we prioritize trajectories with large fluctuations in normalized policy entropy, as such trajectories indicate unstable decision confidence and may benefit from external correction. At the step level, for each selected trajectory, we select the critical decision state using an entropy-gradient saliency criterion, which measures the sensitivity of normalized policy entropy to the action logits. We then query a single binary human feedback signal on the correctness of the selected action. This hierarchical design significantly reduces unnecessary human intervention while refining supervision from the trajectory level to the action level.

Based on the queried feedback, we design a discrepancyaware objective for policy optimization. Instead of updating the entire trajectory from a single final outcome, our method corrects the policy only at the queried step, where the feedback determines both the update magnitude and direction of entropy optimization. Specifically, we introduce an adaptive scaling strategy that assigns larger relative weights to samples with greater mismatch between the current policy prediction and the feedback. We then apply directional entropy optimization: for actions verified as correct, the objective sharpens the policy distribution to reinforce the current decision. Otherwise, it relaxes overconfident predictions and encourages alternative actions. Our contributions can be summarized as follows:

• We propose a Feedback-Efficient Test-Time Adaptation method for VLN and substantially reduce the number of human interventions through a hierarchical sample selection strategy, achieving a favorable trade-off between feedback cost and performance gain.

• We design a discrepancy-aware entropy optimization loss that dynamically adjusts the direction of entropy optimization and the relative sample weights according to the human feedback, enabling efficient utilization of sparse feedback signals.

• Extensive experiments show that FETA achieves the stateof-the-art performance across multiple benchmarks with fewer human interventions, including an 18.30% relative improvement in SPL over the strongest existing navigation TTA baseline, using DUET as the backbone on the REVERIE val unseen split, while remaining robust under noisy feedback and continual adaptation.

## II. RELATED WORK

## A. Vision-and-language navigation

Vision-and-language navigation (VLN) [1, 4, 21, 22] is a fundamental task in embodied AI, requiring an agent to translate natural-language instructions into grounded perception and sequential actions [23]. Early VLN methods were largely based on recurrent seq2seq architectures, augmented with mechanisms such as speaker-based data augmentation, progress estimation, and search to improve instruction following and long-horizon decision making [24–27]. Subsequent work shifted toward pretraining and Transformer-based models [5, 11, 28–32] which substantially improved multimodal fusion, history modeling, and benchmark performance. Despite this progress, most VLN models are still trained offline and tend to suffer significant performance degradation when deployed in unseen environments [33]. Recent efforts explore zero-shot navigation with Large Language Models (LLMs) [34–39] to improve generalization without task-specific training. However, their reliance on large models incurs high inference latency, limiting practicality for realtime navigation. Another line of work studies test-time adap tation through online updates at deployment. Existing methods typically optimize self-supervised objectives such as entropy minimization [6, 15, 16, 40], which are weakly aligned with navigation objectives and can lead to brittle adaptation under distribution shifts. In contrast, our work leverages binary human feedback to directly align adaptation with the navigation goal, enabling more stable updates in unseen environments.

## B. Test-Time Adaptation

Test-Time Adaptation (TTA) aims to update the model parameters directly using target domain data during inference, without access to source domain labels [41]. Most methods [15, 16, 42–44] adopt entropy minimization as the optimization objective, improving the confidence of policy predictions to enable online adaptation to unseen domains, and have demonstrated strong performance on tasks such as image classification [45, 46] and semantic segmentation [47, 48]. Recent work has begun to explore the application of TTA to Vision-and-Language Navigation. FSTTA [6] achieves a balance between fast adaptation and long-term stability through gradient and parameter decomposition. FeedTTA [20] incorporates binary trajectory-level human feedback as a reinforcement learning reward signal to update the navigation policy online. ATENA [7] selectively requests feedback only for high-entropy trajectories and calibrates policy confidence accordingly. However, existing methods that rely on human feedback use the same trajectory-level feedback signal across all steps, potentially reinforcing suboptimal actions or suppressing locally correct ones. By contrast, our method uses hierarchical feedback selection and improving adaptation performance while requiring fewer feedback queries.

## III. PROBLEM STATEMENT AND MOTIVATION

Vision-and-Language Navigation (VLN) is a partially observable sequential decision-making task, where an agent follows the instruction to navigate in a 3D environment. Given an instruction I, at each time step t the agent receives a visual observation $o _ { t }$ and maintains a history representation $h _ { t } ,$ forming the agent state $s _ { t } = ( o _ { t } , h _ { t } , I )$ . The agent then selects the highest-probability action $a _ { t }$ from the candidate set $\boldsymbol { A } _ { t }$ according to its policy $\pi _ { \boldsymbol { \theta } } .$ . The navigation continues until the agent predicts STOP or reaches the maximum step limit, yielding a trajectory ${ \boldsymbol { \tau } } = \{ ( o _ { t } , a _ { t } ) \} _ { t = 1 } ^ { T } .$

In real deployment, VLN agents often encounter environments that differ from those seen during training, which can degrade navigation performance. This motivates test-time adaptation (TTA), which updates model parameters online using incoming test samples. Existing TTA methods typically rely on unsupervised surrogate objectives, $e . g .$ , entropy minimization. However, unlike static perception tasks, VLN is inherently sequential: each action affects future states, observations, and subsequent optimization signals. As a result, once the adaptation signal is unreliable, an erroneous update can propagate through the trajectory and cause cascading failures. The challenge is how to obtain reliable optimization signals under sequential decision-making and environment shift.

Algorithm 1 The pipeline of Feedback-Efficient Test-Time   
Adaptation method for VLN   
Require: Pretrained policy $\pi _ { \theta } ,$ threshold $\delta _ { \sigma } ,$   
scaling exponent $\gamma ,$ learning rate η   
1: for each episode do   
2: Collect trajectory ${ \boldsymbol { \tau } } = \{ ( o _ { t } , a _ { t } ) \} _ { t = } ^ { T }$ <sub>1</sub> using $\pi _ { \theta }$   
3: Compute $\sigma _ { \hat { H } } ( \tau )$ via Eq. (3)   
4: if $\sigma _ { \hat { H } } ( \tau ) > \delta _ { \sigma }$ then   
5: Compute $G _ { t }$ via Eq. (5)   
6: Select step $t ^ { * } = \arg \operatorname* { m a x } _ { t } G _ { t }$   
7: Query binary feedback $r _ { t ^ { * } }$ for action $a _ { t ^ { * } }$   
8: $\alpha _ { t ^ { * } } \gets \mathrm { s g } ( | r _ { t ^ { * } } - \pi _ { \theta } ( a _ { t ^ { * } } \ | \ s _ { t ^ { * } } ) | ^ { \gamma } )$   
9: Compute $\mathcal { L } ( \tau ) = \alpha _ { t ^ { * } } \cdot \mathcal { L } _ { \mathrm { e n t } } ( t ^ { * } )$ via Eq. (8)   
10: Update $\theta  \theta - \eta \nabla _ { \theta } \mathcal { L } ( \tau )$   
return Adapted policy $\pi _ { \theta ^ { \star } }$

To improve adaptation reliability, recent VLN test-time adaptation methods have begun to introduce reliable human feedback during deployment. However, existing humanfeedback-driven methods still remain limited in two important aspects. First, the feedback they acquire may not always translate into effective adaptation, as policy updates on some labeled trajectories yield only limited gains. Second, they typically use a single trajectory-level feedback signal to optimize all actions within that trajectory in the same direction. Such coarse supervision is mismatched to the fine-grained nature of sequential decision-making: successful trajectories may still contain suboptimal actions, while failed trajectories may still include locally correct decisions. Applying the same feedback to all steps can therefore provide inappropriate optimization signals for individual actions, reducing the effectiveness of human guidance for policy adaptation. These limitations suggest that human-feedback-driven TTA for VLN requires a more structured use of sparse supervision, with both feedbackefficient querying and fine-grained action-level correction.

## IV. FEEDBACK-EFFICIENT TEST-TIME ADAPTATION

We propose FETA, a Feedback-Efficient Test-Time Adaptation method for VLN. At test time, after the agent executes a navigation trajectory in the target environment, FETA first performs Hierarchical Sample Selection $( c . f .$ Section IV-B) to determine where human supervision should be used, including trajectory-level selection to identify informative trajectories and step-level selection to locate a promising decision point for feedback within each selected trajectory. FETA then queries a binary feedback signal on the selected action $( c . f .$ Section IV-A). Based on the acquired feedback, FETA finally performs Feedback-Driven Policy

Optimization $( c . f .$ Section IV-C) to convert sparse supervision into discrepancy-aware step-level policy updates. In this way, FETA concentrates limited human supervision on the informative decisions and enables efficient and reliable adaptation under distribution shift. Fig. 1 provides an overview of our method, and Algorithm 1 summarizes the complete procedure.

## A. Preliminary

Progress-Based Binary Feedback. FETA performs adaptation after each completed navigation episode, rather than updating the policy at every decision step. Accordingly, human feedback is queried only after the full trajectory is available. For each queried action, the human provides a binary signal indicating whether the action is beneficial for reaching the navigation goal. In the simulator, we instantiate this supervision with a progress-based oracle. Let $d _ { t }$ denote the geodesic distance from the agent’s position to the goal at time step t. For a non-stop action, the feedback is positive if the action reduces the agent-to-goal geodesic distance; for the stop action, the feedback is positive if the agent has successfully reached the goal. The binary reward is defined as:

$$
r _ { t } = \left\{ \begin{array} { l l } { \mathbb { I } [ d _ { t } > d _ { t + 1 } ] , } & { \mathrm { i f ~ } a _ { t } \neq \mathrm { S T O P } , } \\ { \mathbb { I } [ d _ { T } \leq d _ { \mathrm { t h } } ] , } & { \mathrm { i f ~ } a _ { t } = \mathrm { S T O P } , } \end{array} \right.\tag{1}
$$

where $d _ { T }$ is the final agent-to-goal geodesic distance and $d _ { \mathrm { t h } }$ is the success threshold. $r _ { t } ~ = ~ 1$ and $r _ { t } ~ = ~ 0$ denote positive and negative feedback, respectively. This oracle enables reproducible evaluation, while local progress serves as an approximation to human judgment on action usefulness.

## B. Hierarchical Sample Selection

Requesting human feedback on every action is impractical at test time. Under a limited feedback budget, we must answer two questions: which trajectories are worth querying, and where to query within each selected trajectory. We address these questions through a hierarchical sample selection strategy, whose design is motivated by two validation studies. As shown in Fig. 2, under comparable numbers of feedback, selecting trajectories with larger fluctuations in normalized policy entropy yields better adaptation gains, motivating our trajectory-level criterion. As shown in Fig. 3, although querying additional steps can further improve performance, the gains quickly diminish while the query cost continues to increase, indicating that one carefully selected step per trajectory provides the favorable trade-off. Guided by these observations, FETA first performs Trajectory-Level Selection to identify informative trajectories and then performs Step-Level Selection to select a critical decision step for feedback querying.

Trajectory-Level Selection. Our goal is to identify the trajectories that are most likely to benefit from external feedback under a limited query budget. Since uncertainty is a natural indicator of adaptation potential, we measure the policy uncertainty along each trajectory using normalized entropy. Let $\pi _ { \boldsymbol { \theta } } ( a _ { t } \ \mid \ s _ { t } )$ denote the softmax probability assigned to action $a _ { t }$ at time step $t ,$ and let $| \mathcal { A } _ { t } |$ denote the number of feasible actions at step t. The policy entropy and its normalized form are defined as:

![](images/79d35cb8b1d7f9fedffd8224a6b79a1a8edfc5cd45d18e2f0f7b1e811691d21d.jpg)  
Fig. 1. Overview of FETA. For the trajectory executed by the agent, we hierarchically select informative samples by filtering trajectories with higher normalized-entropy standard deviation $\sigma _ { \hat { H } } ( \tau )$ and querying the step with the largest entropy-gradient saliency $\bar { G _ { t } } .$ . The queried feedback is then used to update the policy via directional entropy optimization modulated by discrepancy-aware scaling, enabling efficient adaptation in novel environments.

$$
H _ { t } = - \sum _ { a \in \mathcal { A } _ { t } } \pi _ { \theta } ( a \mid s _ { t } ) \log \pi _ { \theta } ( a \mid s _ { t } ) , \quad \quad \hat { H } _ { t } = \frac { H _ { t } } { \log \mid \mathcal { A } _ { t } \mid } ,\tag{2}
$$

where $\hat { H } _ { t } ~ \in ~ [ 0 , 1 ]$ captures the relative uncertainty of the policy over the current action space. Based on the per-step normalized entropy within a trajectory τ of length T, we further define the mean and standard deviation of normalized entropy as:

$$
\mu _ { \hat { H } } ( \tau ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \hat { H } _ { t } , \qquad \sigma _ { \hat { H } } ( \tau ) = \sqrt { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left( \hat { H } _ { t } - \mu _ { \hat { H } } ( \tau ) \right) ^ { 2 } } .\tag{3}
$$

To compare the two criteria with closely matched feedback counts, we set their respective dynamic thresholds $\delta _ { \sigma }$ and $\delta _ { \mu }$ to the same p-th percentile of separate sliding windows, each containing the 150 most recent values of the corresponding criterion. As shown in Fig. 2, using the standard deviation $\sigma _ { \hat { H } } ( \tau )$ is more effective than using the mean entropy $\mu _ { \hat { H } } ( \tau )$ Intuitively, a large entropy standard deviation indicates that the agent alternates between confident and uncertain phases along the trajectory, suggesting the presence of localized unreliable decisions that can benefit more from sparse external correction. In contrast, high mean entropy indicates globally diffuse uncertainty, which is harder to correct with limited feedback. Based on this observation, we use $\sigma _ { \hat { H } } ( \tau )$ as the trajectory-level selection criterion:

$$
S _ { \tau } = \mathbb { I } \big ( \sigma _ { \hat { H } } ( \tau ) > \delta _ { \sigma } \big ) ,\tag{4}
$$

where $\delta _ { \sigma }$ is a dynamic threshold derived from recent trajectories (see Section VII) and I(·) denotes the indicator function. Only trajectories with $S _ { \tau } = 1$ are forwarded to the next stage.

Step-Level Selection. After selecting informative trajectories, the next question is where to query within each trajectory. Fig. 3 shows that the first queried step yields the largest performance gain. While additional queried steps can further improve the success rate, their marginal benefits diminish quickly, whereas the feedback count increases nearly linearly. This makes a single carefully selected step the most costeffective choice. Thus, FETA queries only one step for each selected trajectory.

Given this one-query design, we further need to determine which step to query for correction. Intuitively, a useful query target should allow the feedback signal to guide policy optimization effectively. A natural alternative is to query the step with the highest entropy. However, high entropy only indicates that the policy is uncertain, and may simply correspond to a near-uniform distribution, where the entropy gradient with respect to the logits approaches zero. Instead, we select the step whose normalized entropy is most sensitive to the logits, using this sensitivity as a heuristic for identifying steps that may benefit from the correction. We capture this property using the gradient norm of the normalized entropy with respect to the action logits, which we term entropy-gradient saliency.

![](images/13706359f689be823aa103e7c06a1238a26b3a1915a0d006078df8b029ac6479.jpg)  
Fig. 2. Trajectory-level selection criteria comparison. Solid/dashed lines select trajectories above/below threshold. $\delta _ { \sigma }$ and $\delta _ { \mu }$ are thresholds for $\sigma _ { \hat { H } } ( \tau )$ and $\mu _ { \hat { H } } ( \tau )$ , respectively.

Let $\mathbf { z } _ { t }$ denote the action logits at step t. The saliency score is defined as:

$$
\begin{array} { l } { { \displaystyle G _ { t } = \left\| \nabla _ { z _ { t } } \hat { H } _ { t } \right\| } } \\ { { \displaystyle \quad = \frac 1 { \log | \mathcal { A } _ { t } | } \sqrt { \sum _ { a _ { j } \in \mathcal { A } _ { t } } \pi _ { \theta } \left( a _ { j } \mid s _ { t } \right) ^ { 2 } \left( \log \pi _ { \theta } \left( a _ { j } \mid s _ { t } \right) + H _ { t } \right) ^ { 2 } } } . } \end{array}\tag{5}
$$

For each selected trajectory, we rank all decision steps by $G _ { t }$ and query feedback only for the executed action at the critical step $t ^ { * } = \arg \operatorname* { m a x } _ { t \in \tau } G _ { t }$

## C. Feedback-Driven Policy Optimization

To effectively exploit sparse human feedback, we decompose each queried signal into two roles in policy adaptation. The prediction–feedback discrepancy modulates the update magnitude by reweighting how strongly a queried sample contributes to optimization, while the feedback label determines the direction of entropy optimization to encourage reinforcement or relaxation of the queried decision. In this way, feedback modulates the strength and determines the direction of entropy optimization.

Discrepancy-Aware Scaling. Applying uniform-weight updates to all queried samples overlooks the fact that different feedback samples are not equally informative for policy adaptation. To address this, we introduce a discrepancy-aware scaling factor according to the discrepancy between the current policy prediction and the queried feedback:

$$
| r _ { t } - \pi _ { \theta } ( a _ { t } \mid s _ { t } ) | ^ { \gamma } , \gamma > 0 ,\tag{6}
$$

where $\pi _ { \theta } { \big ( } a _ { t } \ { \big | } \ s _ { t } { \big ) }$ denotes the probability assigned by the policy to the executed action $a _ { t } .$ . When $\pi _ { \boldsymbol { \theta } } { \left( a _ { t } \mid s _ { t } \right) }$ is close to $r _ { t } ,$ the current policy already agrees with the feedback, and the corresponding sample should contribute less to optimization. In contrast, when the discrepancy is large, the queried sample receives a larger relative weight, allowing the update to focus more on decision points where the current policy disagrees with the feedback signal.

![](images/f3aca043eb2b101a3fbb4630f616c97778f0cd35e0de65bbfc3b868395ba322d.jpg)  
Fig. 3. Effect of maximum number of feedback per trajectory (K) on success rate and total feedback count. Our method uses $K = 1$ to balance feedback cost and performance.

Directional Entropy Optimization. Different feedback signals imply different local adjustment needs. For correct actions, the current decision should be reinforced; for incorrect ones, the current preference should be relaxed so that alternative actions can be explored. To realize this intuition, we use the policy entropy $H _ { t }$ as the optimization objective and introduce a directional weight $w _ { t } \mathbf { : }$

$$
\mathcal { L } _ { \mathrm { e n t } } ( t ) = w _ { t } \cdot H _ { t } , \qquad \mathrm { w h e r e } \quad w _ { t } = \left\{ + 1 , \quad r _ { t } = 1 , \right.\tag{7}
$$

When the feedback is positive, $w _ { t } = + 1$ encourages the local policy distribution to become more concentrated around the current action, thereby reinforcing the verified decision. When the feedback is negative, $w _ { t } ~ = ~ - 1$ encourages the current preference to be relaxed, allowing probability mass to shift toward alternative actions.

Full Optimization Objective. Combining the hierarchical sample filtering mechanism described in Section IV-B, the overall loss is defined as:

$$
\mathcal { L } ( \tau ) = \mathbb { I } [ \sigma _ { \hat { H } } ( \tau ) > \delta _ { \sigma } ] \cdot \mathrm { s g } ( | r _ { t ^ { * } } - \pi _ { \theta } ( a _ { t ^ { * } } \mid s _ { t ^ { * } } ) | ^ { \gamma } ) \cdot \mathcal { L } _ { \mathrm { e n t } } ( t ^ { * } ) ,\tag{8}
$$

where $t ^ { * } = \arg \operatorname* { m a x } _ { t \in \tau } G _ { t }$ and $\operatorname { s g } ( \cdot )$ denotes the stopgradient operator, which treats the discrepancy term as a fixed weighting factor during optimization. This objective is computed only at the single queried step within trajectorie that pass the trajectory-level filter. Here, the discrepancy term re-weights each queried sample according to its consistency with the received feedback, while $w _ { t ^ { * } }$ determines whether the local decision is reinforced or relaxed. Together, these components enable FETA to convert sparse binary feedback into effective local policy adaptation.

## V. EXPERIMENTS

## A. Benchmark and Metrics

We evaluate our method on four benchmarks: REVERIE [49], R2R [1], GSA-R2R [19], and R2R-CE [50]. REVERIE requires the agent to navigate to a target region and identify the referred object based on a high-level instruction. R2R focuses on fine-grained instruction following in discrete indoor environments. GSA-R2R extends R2R with more diverse environments and instruction styles to evaluate agent adaptability in both in-distribution and out-of-distribution settings. R2R-CE transfers the R2R task to continuous environments, requiring low-level action execution. We report SR (Success Rate), OSR (Oracle Success Rate), SPL (Success rate weighted by Path Length), NDTW (Normalized Dynamic Time Warping), TL (trajectory length) and NE (Navigation Error). For REVERIE, we additionally report RGS and RGSPL, which measure remote object grounding success without and with path efficiency weighting. We also introduce FC (Feedback Count) to quantify the total number of human feedback queries during evaluation, directly reflecting the annotation cost of test-time adaptation process. R2R results are provided in the Appendix A.

TABLE I  
EXPERIMENTAL RESULTS ON REVERIE. THE LAST COLUMN REPORTS THE AVERAGE RUNTIME PER EPISODE.
<table><tr><td rowspan="2">Methods</td><td colspan="5">Val Seen</td><td colspan="5">Val Unseen</td><td rowspan="2">Time(ms)</td></tr><tr><td>OSR ↑</td><td>SR ↑</td><td>SPL ↑</td><td>RGSPL ↑</td><td>FC↓</td><td>OSR ↑</td><td>SR↑</td><td>SPL ↑</td><td>RGSPL ↑</td><td>FC↓</td></tr><tr><td>DUET [3]</td><td>73.86</td><td>71.75</td><td>63.94</td><td>51.14</td><td></td><td>51.07</td><td>46.98</td><td>33.73</td><td>23.03</td><td>一</td><td>156.5</td></tr><tr><td>• FSTTA [6]</td><td>73.79</td><td>71.68</td><td>64.00</td><td>51.21</td><td></td><td>50.93</td><td>46.86</td><td>33.71</td><td>23.00</td><td></td><td>457.7</td></tr><tr><td>• FeedTTA [20]</td><td>82.83</td><td>77.28</td><td>65.39</td><td>54.28</td><td>1424</td><td>62.60</td><td>53.85</td><td>28.66</td><td>19.02</td><td>3522</td><td>433.1</td></tr><tr><td>• ATENA [7]</td><td>82.62</td><td>81.40</td><td>72.85</td><td>58.90</td><td>573</td><td>65.78</td><td>61.95</td><td>42.73</td><td>28.39</td><td>2595</td><td>321.3</td></tr><tr><td>• FETA</td><td>84.31</td><td>82.29</td><td>73.75</td><td>59.01</td><td>519</td><td>73.27</td><td>68.33</td><td>50.55</td><td>33.15</td><td>2475</td><td>243.5</td></tr><tr><td>GOAT [13]</td><td>82.36</td><td>80.74</td><td>73.44</td><td>58.82</td><td>一</td><td>57.97</td><td>53.82</td><td>37.52</td><td>27.00</td><td>一</td><td>144.2</td></tr><tr><td>• FSTTA [6]</td><td>82.85</td><td>80.67</td><td>72.65</td><td>58.11</td><td></td><td>57.54</td><td>53.48</td><td>36.76</td><td>25.80</td><td></td><td>461.1</td></tr><tr><td>• FeedTTA [20]</td><td>84.92</td><td>82.08</td><td>69.54</td><td>59.12</td><td>1424</td><td>65.12</td><td>56.75</td><td>33.68</td><td>26.11</td><td>3522</td><td>390.0</td></tr><tr><td>• ATENA [7]</td><td>84.68</td><td>83.02</td><td>76.46</td><td>61.46</td><td>803</td><td>63.66</td><td>60.52</td><td>42.96</td><td>30.43</td><td>3012</td><td>342.3</td></tr><tr><td>• FETA</td><td>90.02</td><td>87.65</td><td>77.98</td><td>62.14</td><td>533</td><td>72.75</td><td>68.65</td><td>54.22</td><td>38.18</td><td>2477</td><td>307.9</td></tr><tr><td>ScaleVLN [11]</td><td>73.30</td><td>70.98</td><td>60.86</td><td>44.30</td><td>1</td><td>61.43</td><td>54.25</td><td>41.28</td><td>25.00</td><td>一</td><td>187.7</td></tr><tr><td>• FSTTA [6]</td><td>73.23</td><td>70.91</td><td>60.82</td><td>44.25</td><td></td><td>61.45</td><td>54.15</td><td>41.24</td><td>25.00</td><td></td><td>544.7</td></tr><tr><td>• FeedTTA [20]</td><td>78.71</td><td>72.10</td><td>55.53</td><td>45.92</td><td>1424</td><td>71.71</td><td>56.18</td><td>34.85</td><td>23.83</td><td>3522</td><td>437.5</td></tr><tr><td>• ATENA [7]</td><td>77.93</td><td>76.04</td><td>66.01</td><td>48.06</td><td>965</td><td>68.49</td><td>63.32</td><td>49.08</td><td>29.20</td><td>2870</td><td>344.4</td></tr><tr><td>• FETA</td><td>83.02</td><td>80.86</td><td>70.76</td><td>51.67</td><td>562</td><td>75.50</td><td>69.84</td><td>56.61</td><td>33.52</td><td>2465</td><td>300.0</td></tr></table>

TABLE II  
EXPERIMENTAL RESULTS ON GSA-R2R.
<table><tr><td rowspan="2">Methods</td><td colspan="4">Test Residential Basic</td><td colspan="4">Test Non-residential Scene</td></tr><tr><td></td><td>SR ↑SPL ↑NDTW↑NE</td><td></td><td></td><td></td><td></td><td>E ↓ SR ↑ SPL ↑NDTW ↑NE ↓</td><td></td></tr><tr><td>GR-DUET [19]</td><td>69</td><td>64</td><td>71</td><td>3.09</td><td>47</td><td>42</td><td>54</td><td>5.55</td></tr><tr><td>• FSTTA [6]</td><td>69</td><td>62</td><td>70</td><td>3.14</td><td>48</td><td>42</td><td>53</td><td>5.46</td></tr><tr><td>• FeedTTA [20]</td><td>69</td><td>64</td><td>72</td><td>3.07</td><td>48</td><td>43</td><td>55</td><td>5.43</td></tr><tr><td>• ATENA [7]</td><td>67</td><td>64</td><td>71</td><td>3.34</td><td>48</td><td>40</td><td>50</td><td>5.45</td></tr><tr><td>• FETA</td><td>71</td><td>67</td><td>74</td><td>2.93</td><td>50</td><td>44</td><td>55</td><td>4.94</td></tr><tr><td>BEVBert [12]</td><td>58</td><td>45</td><td>54</td><td>4.11</td><td>39</td><td>27</td><td>36</td><td>6.69</td></tr><tr><td>• FSTTA [6]</td><td>58</td><td>44</td><td>53</td><td>4.22</td><td>40</td><td>28</td><td>37</td><td>6.50</td></tr><tr><td>• FeedTTA [20]</td><td>59</td><td>44</td><td>53</td><td>4.09</td><td>39</td><td>26</td><td>35</td><td>6.70</td></tr><tr><td>• ATENA [7]</td><td>58</td><td>44</td><td>53</td><td>4.20</td><td>40</td><td>28</td><td>37</td><td>6.49</td></tr><tr><td>• FETA</td><td>59</td><td>45</td><td>54</td><td>3.91</td><td>42</td><td>28</td><td>37</td><td>5.75</td></tr></table>

## B. Main Navigation Results

Results on REVERIE. Table I reports the comparative results on the REVERIE [49] benchmark. We conduct adaptation using DUET [3], GOAT [13], and ScaleVLN [11] as base models. Across all base models and data splits, FETA consistently outperforms ATENA [7], the current state-of-theart test-time adaptation method for navigation. In particular, on the more challenging val unseen split, FETA achieves relative SPL improvements over ATENA of 18.30%, 26.21%, and 15.34% for DUET, GOAT, and ScaleVLN, respectively. Meanwhile, FETA also achieves lower feedback count (FC) across all settings, as the hierarchical selection strategy filters informative trajectories and queries critical navigation step within each selected trajectory. Furthermore, FETA requires consistently less execution time per episode than ATENA, demonstrating that the proposed method improves both effectiveness and efficiency.

TABLE III  
EXPERIMENTAL RESULTS ON R2R-CE.
<table><tr><td rowspan="2">Methods</td><td colspan="4">Val Seen</td><td colspan="4">Val Unseen</td></tr><tr><td>OSR ↑ SR ↑ SPL ↑ NE↓</td><td></td><td></td><td></td><td></td><td></td><td></td><td>OSR ↑ SR↑ SPL ↑ NE↓</td></tr><tr><td>ETPNav [10]</td><td>71</td><td>64</td><td>57</td><td>4.12</td><td>64</td><td>57</td><td>49</td><td>4.72</td></tr><tr><td>• FSTTA [6]</td><td>70</td><td>63</td><td>56</td><td>4.19</td><td>63</td><td>55</td><td>47</td><td>4.86</td></tr><tr><td>• FeedTTA [20]</td><td>72</td><td>65</td><td>58</td><td>4.06</td><td>64</td><td>57</td><td>49</td><td>4.73</td></tr><tr><td>• ATENA [7]</td><td>70</td><td>65</td><td>59</td><td>4.06</td><td>63</td><td>57</td><td>50</td><td>4.69</td></tr><tr><td>• FETA</td><td>72</td><td>66</td><td>59</td><td>4.00</td><td>65</td><td>59</td><td>51</td><td>4.52</td></tr><tr><td>BEVBert [12]</td><td>78</td><td>71</td><td>63</td><td>3.22</td><td>66</td><td>58</td><td>49</td><td>4.53</td></tr><tr><td>• FSTTA [6]</td><td>75</td><td>68</td><td>60</td><td>3.52</td><td>67</td><td>59</td><td>49</td><td>4.59</td></tr><tr><td>• FeedTTA [20]</td><td>77</td><td>72</td><td>64</td><td>3.21</td><td>67</td><td>58</td><td>49</td><td>4.47</td></tr><tr><td>• ATENA [7]</td><td>77</td><td>71</td><td>64</td><td>3.24</td><td>67</td><td>60</td><td>48</td><td>4.34</td></tr><tr><td>• FETA</td><td>79</td><td>73</td><td>64</td><td>3.08</td><td>69</td><td>61</td><td>50</td><td>4.20</td></tr></table>

Results on GSA-R2R. We further evaluate FETA on the scene-diverse GSA-R2R [19] benchmark. As shown in Table II, ATENA brings only limited gains and even degrades the performance of GR-DUET on the residential split. In contrast, FETA delivers consistent improvements across all settings and achieves up to a 10.00% relative improvement in SPL over ATENA on the more challenging non-residential split. This suggests that FETA remains effective even when deployment scenarios differ substantially from the training distribution.

TABLE IV  
ABLATION STUDY OF TRAJECTORY-LEVEL SELECTION (TLS), STEP-LEVEL SELECTION (SLS), AND DISCREPANCY-AWARE SCALING (DAS) ON THE REVERIE VAL UNSEEN SPLIT. BASELINE DENOTES THE UNADAPTED MODEL.
<table><tr><td colspan="3">Module</td><td colspan="4">Val Unseen</td></tr><tr><td>TLS</td><td>SLS</td><td>DAS</td><td>SR ↑</td><td>OSR ↑</td><td>SPL ↑</td><td>RGSPL ↑</td><td>RGS ↑</td></tr><tr><td></td><td>Baseline</td><td></td><td>46.98</td><td>51.07</td><td>33.73</td><td>23.03</td><td>32.15</td></tr><tr><td> $\checkmark$ </td><td></td><td></td><td>55.54</td><td>64.21</td><td>36.10</td><td>24.12</td><td>37.06</td></tr><tr><td></td><td>√</td><td></td><td>61.41</td><td>69.01</td><td>43.32</td><td>29.14</td><td>41.47</td></tr><tr><td>一</td><td></td><td> $\checkmark$ </td><td>56.83</td><td>62.52</td><td>37.63</td><td>25.11</td><td>37.91</td></tr><tr><td></td><td> $\checkmark$ </td><td>√</td><td>63.19</td><td>67.52</td><td>46.86</td><td>31.32</td><td>42.34</td></tr><tr><td> $\checkmark$ </td><td></td><td>√</td><td>57.66</td><td>62.69</td><td>41.20</td><td>27.77</td><td>38.90</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td>1</td><td>65.46</td><td>72.05</td><td>48.42</td><td>32.62</td><td>44.08</td></tr><tr><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>68.33</td><td>73.27</td><td>50.55</td><td>33.15</td><td>45.08</td></tr></table>

Results on R2R-CE. Unlike the discrete navigation settings above, R2R-CE [50] requires the agent to execute lowlevel continuous control commands. As shown in Table III, FETA achieves the best results with both ETPNav [10] and BEVBert [12]. On the val unseen split, it achieves relative SR improvements over ATENA of 3.51% and 1.67% with ETPNav and BEVBert, respectively, demonstrating consistent gains in continuous environments.

## C. Ablation Study

To verify the effectiveness of each component in our method, we conduct ablation studies on the REVERIE Val Unseen split, with results reported in Table IV. To ensure a fair comparison under an equal feedback budget, the ablations of Trajectory-Level Selection (TLS) and Step-Level Selection (SLS) replace the respective selection method with uniform random sampling while keeping the total number of feedback queries identical. Disabling Discrepancy-Aware Scaling (DAS) assigns unit weight to each queried sample. Relative to the full method, replacing TLS with random trajectory sampling leads to a performance drop across all reported metrics, supporting the effectiveness of trajectory selection based on the standard deviation of normalized entropy. Replacing SLS with random step sampling also decreases performance, indicating that entropy-gradient saliency helps make effective use of sparse feedback. Removing DAS while retaining TLS and SLS also decreases performance, supporting the benefit of weighting each queried sample according to the prediction– feedback discrepancy. The variants with only TLS, SLS, or DAS enabled also outperform the unadapted baseline, showing that adaptation remains effective when only one of the three components is enabled. These results collectively demonstrate that our method efficiently exploits sparse human feedback to enable effective agent adaptation in unseen environments.

## D. More Discussions

Comparison of Step-Level Selection Methods. Table VII compares entropy-gradient saliency against querying feedback at the step with highest normalized entropy. Under an equal budget, entropy gradient saliency outperforms the entropybased selection method across all metrics, yielding an 8.97% relative improvement in SPL on the REVERIE Val Unseen split. A possible reason is that time steps with high entropy may correspond to policy outputs that approach a uniform distribution. Negative feedback at such steps provides a weak entropy-maximization signal, as the entropy gradient with respect to the logits approaches zero near a uniform distribution. By contrast, selecting the step with the largest entropy-gradient saliency G<sub>t</sub> directs feedback toward steps where normalized entropy is more sensitive to logit changes, which may facilitate feedback-driven policy adjustment.

TABLE V  
EXPERIMENTAL RESULTS OF CROSS-DATASET ADAPTATION. MODELS ARE TRAINED ON ONE DATASET AND ADAPTED ON THE OTHER.
<table><tr><td rowspan="2">Data (Train → Test)</td><td rowspan="2">Methods</td><td colspan="4">Val Unseen</td></tr><tr><td>SR ↑</td><td>SPL ↑</td><td>NE↓</td><td>TL ↓</td></tr><tr><td rowspan="2">R2R → REVERIE</td><td>DUET [3]</td><td>24.91</td><td>20.00</td><td>7.22</td><td>14.67</td></tr><tr><td>• ATENA • FETA</td><td>29.42 40.02</td><td>19.43 23.53</td><td>6.75 5.20</td><td>24.45 39.09</td></tr><tr><td rowspan="2">REVERIE → R2R</td><td>DUET [3]</td><td>29.84</td><td>9.03</td><td></td><td></td></tr><tr><td>• ATENA</td><td>31.25</td><td>8.41</td><td>7.28 6.98</td><td>44.20 48.40</td></tr><tr><td rowspan="2"></td><td>• FETA</td><td>36.91</td><td>19.06</td><td>6.23</td><td>34.65</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE VI  
EXPERIMENTAL RESULTS IN REAL-WORLD ENVIRONMENTS.
<table><tr><td rowspan="2">Method</td><td colspan="2">Office</td><td colspan="2">Laboratory</td></tr><tr><td>SR↑</td><td>NE↓</td><td>SR↑</td><td>NE↓</td></tr><tr><td>BEVBert [12]</td><td>40</td><td>4.30</td><td>30</td><td>4.20</td></tr><tr><td>• FETA</td><td>55</td><td>3.87</td><td>40</td><td>4.03</td></tr></table>

Robustness of the Method. In practical applications, binary feedback provided by humans is rarely perfectly accurate. To evaluate the robustness of our method under noisy feedback, we simulate noise by randomly flipping feedback labels with varying probabilities, gradually reducing the feedback accuracy from 100% to 50%. As shown in Fig. 4, the navigation success rate decreases as the noise level increases. Even at 60% feedback accuracy, our method still outperforms the unadapted baseline. Among the tested noise levels, performance falls below the baseline only when feedback becomes random (50% accuracy). These results demonstrate that FETA can tolerate substantial simulated label noise while retaining an advantage over the unadapted baseline.

Continual Adaptation. We evaluate continual adaptation by progressively adapting the model over multiple rounds and measuring performance after each round, starting from the unadapted model (Round 0). As shown in Fig. 5, ATENA improves over the baseline after the first round but degrades in subsequent rounds, eventually falling below the unadapted performance. In contrast, FETA improves in each of the three adaptation rounds. This contrast may be related to the granularity of supervision: ATENA uses the same trajectorylevel feedback signal across all steps, potentially reinforcing suboptimal actions or suppressing locally correct ones. By contrast, FETA computes the adaptation loss at a selected step using feedback on the corresponding action, aligning the supervision granularity with the decision being optimized.

TABLE VII  
COMPARISON OF STEP-LEVEL SELECTION METHODS ON THE REVERIE DATASET.
<table><tr><td rowspan="2">Selection Method</td><td colspan="4">Val Seen</td><td colspan="4">Val Unseen</td></tr><tr><td>OSR↑</td><td>SR↑</td><td>SPL↑</td><td>RGSPL↑</td><td>OSR↑</td><td>SR↑</td><td>SPL↑</td><td>RGSPL↑</td></tr><tr><td>Entropy</td><td>81.19</td><td>79.03</td><td>71.31</td><td>57.41</td><td>69.63</td><td>63.97</td><td>46.39</td><td>31.13</td></tr><tr><td>Entropy-Gradient Saliency</td><td>84.31</td><td>82.29</td><td>73.75</td><td>59.01</td><td>73.27</td><td>68.33</td><td>50.55</td><td>33.15</td></tr></table>

![](images/ee81ea6d6be4e8fa43ebc0f38afddcbaa62867f772c0d52ea09f8a37cf37efdd.jpg)  
Fig. 4. Effect of noisy binary feedback on navigation performance in the REVERIE Val Unseen split.

Cross-Dataset Adaptation. We evaluate cross-dataset adaptation by training the model on one dataset and then adapting it to the other. Table V reports cross-dataset generalization under two transfer directions: R2R→REVERIE and REVERIE→R2R. Compared with the competing methods, FETA shows strong transferability across datasets and tasks, achieving better adaptation performance. These results confirm that our adaptation strategy is not dataset-specific but generalizes robustly across heterogeneous task distributions.

Real-Robot Navigation Results. We evaluate FETA on a physical robot in an office and a laboratory, with 20 navigation episodes per environment. We use BEVBert [12] as the baseline and retain the same settings as in simulation. As shown in Table VI, FETA improves success rate by 15 and 10 percentage points in the office and laboratory, respectively, while also reducing navigation error in both environments. These results provide initial evidence that FETA can improve navigation performance in physical environments beyond simulation.

## VI. HUMAN–SIMULATOR FEEDBACK AGREEMENT

To assess how closely the progress-based simulator feedback reflects human judgments of action usefulness, we conduct a human annotation study on 90 feedback queries from the REVERIE Val Unseen split. Three annotators each label a disjoint set of 30 queries, so each query receives one human annotation. For each query, the annotator is shown the navigation instruction, the complete observation sequence along the trajectory, the observations immediately before and after the queried action, and the action itself. The annotator then assigns a binary label indicating whether the action is helpful for completing the task. Human annotations agree with the simulator feedback on 78.9% of queries, with agreement rates of 81.8% for simulator-positive queries and 76.1% for simulator-negative queries. The label distributions are approximately balanced: 48.9% of simulator labels are positive, while annotators judge 52.2% of actions as helpful. These results provide empirical support for using the simulator feedback as a proxy for human judgments in this setting. The annotation interface is shown in Fig. 8 in the appendix.

![](images/8f4a631bb3d4a2218dd907f6ae8841f388ba99bcd08636f070ed694a8d686c10.jpg)  
Fig. 5. Continual adaptation results on REVERIE Unseen split, averaged over 3 random seeds.

## VII. IMPLEMENTATION DETAILS

To simulate real-world deployment conditions, we adopt an online test-time adaptation paradigm with a batch size of 1. All adaptation methods are evaluated three times with distinct random orderings of test samples, and we report the mean performance across runs. We evaluate our method against several representative VLN architectures, including DUET [3], GOAT [13], ScaleVLN [11], ETPNav [10], and BEVBert [12]. All baseline results are reproduced using the official implementations. During adaptation, we perform fullparameter updates with learning rates chosen from $\{ 6 ~ \times$ $\mathrm { \bar { 1 0 ^ { - 6 } } , 2 \times 1 \bar { 0 } ^ { - 6 } , 6 \times 1 0 ^ { - 7 } } \}$ . A gradient update is triggered at the end of an episode only if the standard deviation of the normalized entropy $\sigma _ { \hat { H } } ( \tau )$ across the trajectory exceeds a dynamic threshold $\delta _ { \sigma }$ . Specifically, we maintain a sliding window of the $W = 1 5 0$ most recent standard deviations. Before the window is filled, all episodes pass the filter since feedback is valuable when the agent first encounters a novel environment. On REVERIE, R2R and R2R-CE, $\delta _ { \sigma }$ is set to

Shared Path Before-Adaptation Path FETA-Adapted Path

![](images/7b04278449ce3f36c5d97a366786aa6844d627c50b77f3dc6e1162e4a920dcbc.jpg)  
Shared Path Before-Adaptation Path FETA-Adapted Path

![](images/eddf782f15c8b22fcd70a2f6d8db855447ac5d6dc86227a547bf2bd808f83e28.jpg)

Fig. 6. Qualitative comparison of navigation trajectories between the base model and our test-time adaptation method (FETA). Red waypoints indicate th trajectory predicted by the base model, while green waypoints denote the trajectory predicted after applying FETA at test time. Blue waypoints are the shared prefix visited by both methods before their trajectories diverge.  
(a) Walk past the grey sofa, turn left, and go to the blue standing display board.  
![](images/a0e8a8179699b87d9ed2bb184279b381d839c0f4cfb7bffafb5739822928ac92.jpg)  
Fig. 7. Real-robot navigation examples. (a) BEVBert fails to reach the blue display board, while FETA proceeds past the sofas and stops near the board. (b) BEVBert remains near the starting location, while FETA successfully stops in front of the glass door.

the 70th percentile of this window for Val Seen and the 30th percentile for Val Unseen. On GSA-R2R, $\delta _ { \sigma }$ is set to the 30th percentile for all evaluations. These settings balance feedback cost and navigation performance. For episodes that pass this filter, the update is applied at the navigation step exhibiting the highest entropy-gradient saliency $G _ { t }$ . The discrepancyaware scaling factor γ in Eq. (6) is set to 0.6. The results of FSTTA [6], FeedTTA [20], and ATENA [7] are reproduced using its officially released codebase. All experiments are conducted on a single NVIDIA Tesla A800 GPU.

## VIII. QUALITATIVE ANALYSIS

Fig. 6 visualizes the navigation trajectories of the base policy (DUET [3]) and the policy adapted by FETA on two representative episodes from the REVERIE validation unseen split. In the left panel of Fig. 6, the instruction asks the agent to go to the dining room and identify the table support closest to the tray of food. The unadapted model navigates toward a plausible dining area containing a round table, but ultimately stops at an incorrect location. After adaptation, FETA steers the agent toward the correct dining room, yielding a trajectory that better matches the instructed destination. In the right panel of Fig. 6, the instruction requires the agent to navigate to the balcony and locate the book closest to the lantern on the table. The unadapted model terminates in the living room rather than reaching the requested balcony. After adaptation, FETA leads the agent to the balcony specified by the instruction. Across both examples, a consistent pattern emerges: the unadapted policy makes a plausible but erroneous decision at a branching point and subsequently fails to reach the target, whereas the adapted policy follows a route toward the instructed destination.

The improvement in reaching the instructed destination is also reflected in the real-robot examples in Fig. 7, where BEVBert serves as the base policy. In example (a), the adapted policy proceeds past the sofas and stops near the blue display board, which the baseline fails to reach. In example (b), the adapted policy reaches the specified glass door, while the baseline remains near the starting location. Together, the simulation and real-robot examples provide qualitative evidence that FETA can help navigation policies follow routes toward instruction-specified destinations in both simulated and physical environments.

## IX. CONCLUSION

We propose FETA, a Feedback-Efficient Test-Time Adaptation method for Vision-and-Language Navigation. To address the inefficient feedback utilization and mismatched optimization granularity of prior methods, FETA performs hierarchical sample selection that filters informative trajectories via the standard deviation of normalized entropy and selects a critical step for targeted feedback via entropygradient saliency. The queried feedback is then converted into a fine-grained step-level update through a discrepancy-aware entropy optimization objective. Extensive experiments show that FETA achieves state-of-the-art performance with fewer human queries and lower per-episode latency.

## REFERENCES

[1] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sunderhauf, I. Reid, S. Gould, and A. Van¨ Den Hengel, “Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3674– 3683.

[2] Y. Qiao, Y. Qi, Y. Hong, Z. Yu, P. Wang, and Q. Wu, “Hop: History-and-order aware pre-training for vision-and-language navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 15 418–15 427.

[3] S. Chen, P.-L. Guhur, M. Tapaswi, C. Schmid, and I. Laptev, “Think global, act local: Dual-scale graph transformer for vision-and-language navigation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16 537–16 547.

[4] Y. Hong, Z. Wang, Q. Wu, and S. Gould, “Bridging the gap between learning in discrete and continuous environments for vision-and-language navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2022.

[5] P.-L. Guhur, M. Tapaswi, S. Chen, I. Laptev, and C. Schmid, “Airbert: In-domain Pretraining for Visionand-Language Navigation,” 2021.

[6] J. Gao, X. Yao, and C. Xu, “Fast-slow test-time adaptation for online vision-and-language navigation,” in ICML, 2024.

[7] H. Ko, S. Kim, G. Oh, J. Yoon, H. Lee, S. Jang, S. Kim, and S. Kim, “Active test-time vision-language navigation,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[8] B. Lin, Y. Zhu, Y. Long, X. Liang, Q. Ye, and L. Lin, “Adversarial reinforced instruction attacker for robust vision-language navigation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 10, pp. 7175–7189, 2021.

[9] X. Wang, W. Wang, J. Shao, and Y. Yang, “Learning to follow and generate instructions for language-capable navigation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 3334–3350, 2023.

[10] D. An, H. Wang, W. Wang, Z. Wang, Y. Huang, K. He, and L. Wang, “Etpnav: Evolving topological planning for vision-language navigation in continuous environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[11] Z. Wang, J. Li, Y. Hong, Y. Wang, Q. Wu, M. Bansal, S. Gould, H. Tan, and Y. Qiao, “Scaling data generation in vision-and-language navigation,” in ICCV, 2023, pp. 12 009–12 020.

[12] D. An, Y. Qi, Y. Li, Y. Huang, L. Wang, T. Tan, and J. Shao, “Bevbert: Multimodal map pre-training for language-guided navigation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

[13] L. Wang, Z. He, R. Dang, M. Shen, C. Liu, and Q. Chen, “Vision-and-language navigation via causal learning,” in The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[14] X. Wang, J. Dong, B. Liu, Q. Lyu, L. Liu, and Z. Han, “Lifelong embodied navigation learning,” in The Fourteenth International Conference on Learning Representations, 2026. [Online]. Available: https:// openreview.net/forum?id=PaYo96rjij

[15] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell, “Tent: Fully test-time adaptation by entropy minimization,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=uXl3bZLkr3c

[16] S. Niu, J. Wu, Y. Zhang, Y. Chen, S. Zheng, P. Zhao, and M. Tan, “Efficient test-time model adaptation without forgetting,” in The Internetional Conference on Machine Learning, 2022.

[17] S. Niu, C. Miao, G. Chen, P. Wu, and P. Zhao, “Test-time model adaptation with only forward passes,” in Forty-first International Conference on Machine Learning, 2024.

[18] Y. Su, X. Xu, T. Li, and K. Jia, “Revisiting realistic test-time training: Sequential inference and adaptation by anchored clustering regularized self-training,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5524–5540, 2024.

[19] H. Hong, Y. Qiao, S. Wang, J. Liu, and Q. Wu, “General

scene adaptation for vision-and-language navigation,” in The Thirteenth International Conference on Learning Representations, 2025.

[20] S. Kim, G. Oh, H. Ko, D. Ji, D. Lee, B.-J. Lee, S. Jang, and S. Kim, “Test-time adaptation for online visionlanguage navigation with feedback-based reinforcement learning,” in Forty-second International Conference on Machine Learning, 2025.

[21] X. Shi, Z. Li, W. Lyu, J. Xia, F. Dayoub, Y. Qiao, and Q. Wu, “Smartway: Enhanced waypoint prediction and backtracking for zero-shot vision-and-language navigation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 16 923–16 930.

[22] B. Lin, Y. Nie, K. L. Zai, Z. Wei, M. Han, R. Xu, M. Niu, J. Han, H. Zhang, L. Lin et al., “Evolvenav: Empowering llm-based vision-language navigation via self-improving embodied reasoning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[23] K. He, Y. Huang, Y. Jing, Q. Wu, and L. Wang, “Fine-grained alignment supervision matters in visionand-language navigation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[24] D. Fried, R. Hu, V. Cirik, A. Rohrbach, J. Andreas, L.-P. Morency, T. Berg-Kirkpatrick, K. Saenko, D. Klein, and T. Darrell, “Speaker-follower models for vision-and-language navigation,” in Advances in Neural Information Processing Systems, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds., vol. 31. Curran Associates, Inc., 2018. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2018/ file/6a81681a7af700c6385d36577ebec359-Paper.pdf

[25] C.-Y. Ma, J. Lu, Z. Wu, G. AlRegib, Z. Kira, R. Socher, and C. Xiong, “Self-monitoring navigation agent via auxiliary progress estimation,” in Proceedings of the International Conference on Learning Representations (ICLR), 2019. [Online]. Available: https://arxiv.org/abs/ 1901.03035

[26] H. Wang, W. Wang, T. Shu, W. Liang, and J. Shen, “Active visual information gathering for vision-language navigation,” in ECCV, 2020.

[27] X. Wang, Q. Huang, A. Celikyilmaz, J. Gao, D. Shen, Y.- F. Wang, W. Y. Wang, and L. Zhang, “Vision-language navigation policy learning and adaptation,” IEEE transactions on pattern analysis and machine intelligence, no. 12, pp. 4205–4216, 2020.

[28] Y. Hong, Q. Wu, Y. Qi, C. Rodriguez-Opazo, and S. Gould, “A recurrent vision-and-language bert for navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2021, pp. 1643–1653.

[29] S. Chen, P.-L. Guhur, C. Schmid, and I. Laptev, “History aware multimodal transformer for vision-and-language navigation,” in Advances in Neural Information Processing Systems, 2021.

[30] J. Zhang, K. Wang, S. Wang, M. Li, H. Liu, S. Wei, Z. Wang, Z. Zhang, and H. Wang, “Uni-navid: A video-

based vision-language-action model for unifying embodied navigation tasks,” in Robotics: Science and Systems, 2025.

[31] M. Wei, C. Wan, X. Yu, T. Wang, Y. Yang, X. Mao, C. Zhu, W. Cai, H. Wang, Y. Chen et al., “Streamvln: Streaming vision-and-language navigation via slowfast context modeling,” in 2026 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2026.

[32] Y. Qiao, Y. Qi, Y. Hong, Z. Yu, P. Wang, and Q. Wu, “Hop+: History-enhanced and order-aware pre-training for vision-and-language navigation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 7, pp. 8524–8537, 2023.

[33] B. Lin, Y. Nie, Z. Wei, Y. Zhu, H. Xu, S. Ma, J. Liu, and X. Liang, “Correctable landmark discovery via large models for vision-language navigation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 8534–8548, 2024.

[34] Y. Qiao, W. Lyu, H. Wang, Z. Wang, Z. Li, Y. Zhang, M. Tan, and Q. Wu, “Open-nav: Exploring zero-shot vision-and-language navigation in continuous environment with open-source llms,” in ICRA, 2025.

[35] G. Zhou, Y. Hong, and Q. Wu, “Navgpt: Explicit reasoning in vision-and-language navigation with large language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, 2024, pp. 7641–7649.

[36] P. Chen, X. Sun, H. Zhi, R. Zeng, T. H. Li, G. Liu, M. Tan, and C. Gan, “a<sup>2</sup>nav: Action-aware zeroshot robot navigation by exploiting vision-and-language ability of foundation models,” 2023. [Online]. Available: https://arxiv.org/abs/2308.07997

[37] Y. Long, W. Cai, H. Wang, G. Zhan, and H. Dong, “Instructnav: Zero-shot system for generic instruction navigation in unexplored environment,” in 8th Annual Conference on Robot Learning, 2024. [Online]. Available: https://openreview.net/forum?id=fCDOfpTCzZ

[38] Y. Qiao, H. Hong, W. Lyu, D. An, S. Zhang, Y. Xie, X. Wang, and Q. Wu, “Navbench: Probing multimodal large language models for embodied navigation,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[39] K. Chen, D. An, Y. Huang, R. Xu, Y. Su, Y. Ling, I. Reid, and L. Wang, “Constraint-aware zero-shot visionlanguage navigation in continuous environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[40] Q. Zhang, Y. Bian, X. Kong, P. Zhao, and C. Zhang, “Come: Test-time adaption by conservatively minimizing entropy,” in Internetional Conference on Learning Representations, 2025.

[41] H. Dong, M. Liu, K. Zhou, E. Chatzi, J. Kannala, C. Stachniss, and O. Fink, “Advances in multimodal adaptation and generalization: From traditional approaches to foundation models,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[42] Q. Wang, O. Fink, L. Van Gool, and D. Dai, “Continual test-time domain adaptation,” in Proceedings of Conference on Computer Vision and Pattern Recognition, 2022.

[43] S. Niu, J. Wu, Y. Zhang, Z. Wen, Y. Chen, P. Zhao, and M. Tan, “Towards stable test-time adaptation in dynamic wild world,” in Internetional Conference on Learning Representations, 2023.

[44] M. Tan, G. Chen, J. Wu, Y. Zhang, Y. Chen, P. Zhao, and S. Niu, “Uncertainty-calibrated test-time model adaptation without forgetting,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 47, no. 8, pp. 6274–6289, 2025.

[45] T. Lee, S. Chottananurak, J. Kim, J. Shin, T. Gong, and S.-J. Lee, “Test-time adaptation with binary feedback,” in Forty-second International Conference on Machine Learning, 2025.

[46] A. Karmanov, D. Guan, S. Lu, A. El Saddik, and E. Xing, “Efficient test-time adaptation of vision-language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 14 162–14 171.

[47] H. Cao, Y. Xu, J. Yang, P. Yin, S. Yuan, and L. Xie, “Multi-modal continual test-time adaptation for 3d semantic segmentation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 18 809–18 819.

[48] J. Lee, D. Das, J. Choo, and S. Choi, “Towards open-set test-time adaptation utilizing the wisdom of crowds in entropy minimization,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 16 380–16 389.

[49] Y. Qi, Q. Wu, P. Anderson, X. Wang, W. Y. Wang, C. Shen, and A. van den Hengel, “Reverie: Remote embodied visual referring expression in real indoor environments,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

[50] J. Krantz, E. Wijmans, A. Majundar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision and language navigation in continuous environments,” in European Conference on Computer Vision (ECCV), 2020.

![](images/10d9affed84513b2eac8ca8cad9c8542441adcc14a75fb5f68cfbf61c871947e.jpg)

Huaquan Zhang is currently pursuing the M.S. degree at South China University of Technology, Guangzhou, China. He received his B.Sc. degree from South China University of Technology, Guangzhou, China, in 2025. His research interests include vision-and-language navigation and test-time adaptation.

![](images/3eaa8db1eff34787509576c408fea0be4571a0d44022db98122d77973aa3b9bc.jpg)  
Zixu Wang is currently pursuing a Ph.D. degree at South China University of Technology, Guangzhou, China. He received the B.E. degree in Automation from Central South University, China, in 2022. His research interests include vision-and-language navigation, test-time adaptation and continual learning.

![](images/86695fbc72d0ca5acabc56192fa16cf615a177a2e8c5be785c60824e560d5ff0.jpg)

Shoukai Xu is Postdoctoral Researcher with School of Software Engineering at South China University of Technology. He received his Ph.D. degree in the School of Software Engineering in 2025 from South China University of Technology in Guangzhou, China. His research interests include computer vision and lightweight neural model optimization. He has published papers in top AI venues, including ICML, ICLR, ECCV and IEEE TCSVT.

Jinwu Hu is currently pursuing a Ph.D. degree at South China University of Technology, Guangzhou, China. He received the M.S. degree from Chongqing University of Posts and Telecommunications, Chongqing, China, in 2023. His research interests include LLMs, computer vision, and reinforcement learning. He has published papers in top venues, including ICML, NeurIPS, CVPR, IJCAI, ACM MM, ECCV, IEEE TKDE, IEEE TIP, IEEE TIFS, IEEE TMM, IEEE TCYB, IEEE TMI, etc. He has served as a reviewer for many journals/conferences, including TPAMI, ICML, NeurIPS, CVPR, etc.

![](images/2fd8d2082b042b503f1a240f959dd16dbd80538841b3ef86faf3c39a27c59ca7.jpg)

![](images/9c7846d72357c404b4ad7e1f5754ed8d4d3de8af42d9bd944b5af6690fefd6a4.jpg)

Zhiquan Wen is recently a Post-doctoral Researcher with Shien-Ming Wu School of Intelligent Engineering at South China University of Technology. He received his Ph.D. degree in the School of Software Engineering in 2024 from South China University of Technology in Guangzhou, China. His research interests include vision-and-language and test-time adaptation. He has published papers in top venues, including NeurIPS, ICLR, ECCV, AAAI, ACL, IEEE TMM, IEEE TCSVT and Neural Networks. He has been invited as a reviewer for top-tier conferences and journals including IEEE TPAMI, IEEE TMM, ICLR, ICML, NeurIPS, CVPR, ICCV, AAAI, etc.

![](images/ede70dee1fe1112a42866a19b07befc8ca3ec99ecbc4876058bf7806cff5017a.jpg)

Mingkui Tan is currently a Professor with the School of Software Engineering at South China University of Technology. He received his B.Sc. degree in Environmental Science and Engineering and M.Eng. degree in Control Science and Engineering from Hunan University, Changsha, China, in 2006 and 2009, respectively. He obtained his Ph.D. degree in Computer Science from Nanyang Technological University, Singapore, in 2014. From 2014 to 2016, he worked as a Senior Research Associate in the School of Computer Science at the University of

Adelaide, Australia, where his research focused on computer vision. He currently serves as an Associate Editor for both IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) and Machine Intelligence Research (MIR). His research interests include machine learning, deep learning, computer vision, sparse analysis, and large-scale optimization.

# Supplementary Materials for “Asking Less, Adapting Better: Feedback-Efficient Test-Time Adaptation for Vision-and-Language Navigation”

## APPENDIX A ADAPTATION RESULTS ON R2R

We further evaluate FETA on the classic R2R benchmark using DUET, GOAT, and ScaleVLN as navigation backbones. As shown in Table VIII, FETA improves overall navigation performance across all three backbones. On the Val Unseen split, it achieves higher navigation success than ATENA while maintaining or improving path efficiency. These results demonstrate the effectiveness of FETA on R2R across different navigation backbones.

TABLE VIII  
EXPERIMENTAL RESULTS ON R2R DATASET.
<table><tr><td rowspan="2">Methods</td><td colspan="3">Val Seen</td><td colspan="4">Val Unseen</td></tr><tr><td>OSR ↑SR ↑ SPL ↑NE↓OSR</td><td></td><td></td><td></td><td></td><td></td><td>↑SR↑SPL ↑NE↓</td><td></td></tr><tr><td>DUET [3]</td><td>86</td><td>79</td><td>73</td><td>2.28</td><td>81</td><td>72</td><td>60</td><td>3.31</td></tr><tr><td>• FSTTA [6]</td><td>83</td><td>76</td><td>69</td><td>2.51</td><td>78</td><td>68</td><td>57</td><td>3.47</td></tr><tr><td>• FeedTTA [20]</td><td>85</td><td>78</td><td>73</td><td>2.29</td><td>80</td><td>72</td><td>61</td><td>3.26</td></tr><tr><td>• ATENA [7]</td><td>85</td><td>79</td><td>74</td><td>2.28</td><td>79</td><td>72</td><td>62</td><td>3.23</td></tr><tr><td>• FETA</td><td>86</td><td>80</td><td>74</td><td>2.20</td><td>82</td><td>74</td><td>62</td><td>3.01</td></tr><tr><td>GOAT [13]</td><td>89</td><td>85</td><td>80</td><td>1.67</td><td>86</td><td>78</td><td>68</td><td>2.31</td></tr><tr><td>• FSTTA [6]</td><td>88</td><td>83</td><td>78</td><td>1.82</td><td>85</td><td>77</td><td>66</td><td>2.45</td></tr><tr><td>• FeedTTA [20]</td><td>89</td><td>85</td><td>80</td><td>1.66</td><td>86</td><td>78</td><td>68</td><td>2.30</td></tr><tr><td>• ATENA [7]</td><td>89</td><td>85</td><td>80</td><td>1.67</td><td>86</td><td>78</td><td>69</td><td>2.27</td></tr><tr><td>• FETA</td><td>90</td><td>85</td><td>80</td><td>1.65</td><td>86</td><td>79</td><td>69</td><td>2.21</td></tr><tr><td>ScaleVLN [11]</td><td>87</td><td>81</td><td>75</td><td>2.12</td><td>88</td><td>81</td><td>70</td><td>2.09</td></tr><tr><td>• FSTTA [6]</td><td>84</td><td>78</td><td>72</td><td>2.32</td><td>86</td><td>78</td><td>67</td><td>2.30</td></tr><tr><td>• FeedTTA [20]</td><td>87</td><td>82</td><td>76</td><td>2.10</td><td>87</td><td>80</td><td>71</td><td>2.12</td></tr><tr><td>• ATENA [7]</td><td>86</td><td>82</td><td>77</td><td>2.04</td><td>86</td><td>79</td><td>71</td><td>2.18</td></tr><tr><td>• FETA</td><td>88</td><td>82</td><td>77</td><td>1.96</td><td>88</td><td>81</td><td>72</td><td>2.08</td></tr></table>

## APPENDIX B

## DERIVATION OF ENTROPY-GRADIENT SALIENCY

We derive the closed-form expression for the entropygradient saliency $G _ { t } ~ = ~ \left\| \nabla _ { \mathbf { z } _ { t } } \hat { H _ { t } } \right\|$ used in the step-level selection criterion (Section IV-B). Let $\mathbf { z } _ { t } = ( z _ { 1 } , z _ { 2 } , \ldots , z _ { N } ) ^ { \top }$ denote the action logits at step t, where $N \mathrm { ~ = ~ } | A _ { t } |$ is the number of feasible actions. The policy is defined via the softmax function:

$$
\pi _ { j } \triangleq \pi _ { \boldsymbol \theta } ( a _ { j } \mid s _ { t } ) = \frac { \exp ( z _ { j } ) } { \sum _ { k = 1 } ^ { N } \exp ( z _ { k } ) } , \quad j = 1 , \ldots , N .\tag{9}
$$

The policy entropy and its normalized form are:

$$
H _ { t } = - \sum _ { j = 1 } ^ { N } \pi _ { j } \log \pi _ { j } , \hat { H } _ { t } = \frac { H _ { t } } { \log N } .\tag{10}
$$

a) Step 1: Softmax Jacobian.: The partial derivative of the softmax output $\pi _ { i }$ with respect to the logit $z _ { j }$ is:

$$
\frac { \partial \pi _ { i } } { \partial z _ { j } } = \pi _ { i } \left( \delta _ { i j } - \pi _ { j } \right) ,\tag{11}
$$

where $\delta _ { i j }$ is the Kronecker delta.

b) Step 2: Gradient of the entropy with respect to logits.: Taking the partial derivative of $H _ { t }$ with respect to $z _ { j } { \mathrm { : } }$

$$
\begin{array} { l } { \displaystyle \frac { \partial H _ { t } } { \partial z _ { j } } = - \sum _ { i = 1 } ^ { N } \frac { \partial } { \partial z _ { j } } \left[ \pi _ { i } \log \pi _ { i } \right] } \\ { \displaystyle \quad = - \sum _ { i = 1 } ^ { N } \left[ \frac { \partial \pi _ { i } } { \partial z _ { j } } \log \pi _ { i } + \pi _ { i } \cdot \frac { 1 } { \pi _ { i } } \frac { \partial \pi _ { i } } { \partial z _ { j } } \right] } \\ { \displaystyle \quad = - \sum _ { i = 1 } ^ { N } \frac { \partial \pi _ { i } } { \partial z _ { j } } \left( \log \pi _ { i } + 1 \right) . } \end{array}\tag{12}
$$

Substituting the softmax Jacobian from Eq. (11):

$$
\begin{array} { c } { { { \displaystyle \frac { \partial H _ { t } } { \partial z _ { j } } } = - \sum _ { i = 1 } ^ { N } \pi _ { i } \left( \delta _ { i j } - \pi _ { j } \right) \left( \log \pi _ { i } + 1 \right) } } \\ { { = - \pi _ { j } \left( \log \pi _ { j } + 1 \right) + \pi _ { j } \displaystyle \sum _ { i = 1 } ^ { N } \pi _ { i } \left( \log \pi _ { i } + 1 \right) . } } \end{array}\tag{13}
$$

Note that $\textstyle \sum _ { i = 1 } ^ { N }$ π<sub>i</sub> log $\pi _ { i } = - H _ { t }$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { N } { \pi } _ { i } = 1 } \end{array}$ . Therefore:

$$
\begin{array} { c } { { \displaystyle \frac { \partial { \cal H } _ { t } } { \partial z _ { j } } = - \pi _ { j } \left( \log \pi _ { j } + 1 \right) + \pi _ { j } \left( - { \cal H } _ { t } + 1 \right) } } \\ { { { } } } \\ { { = - \pi _ { j } \log \pi _ { j } - \pi _ { j } + \pi _ { j } - \pi _ { j } { \cal H } _ { t } } } \\ { { { } } } \\ { { = - \pi _ { j } \left( \log \pi _ { j } + { \cal H } _ { t } \right) . } } \end{array}\tag{14}
$$

c) Step 3: Gradient of the normalized entropy.: Since $\hat { H } _ { t } = \hat { H } _ { t } / \log | \mathcal { A } _ { t } |$ and $| \mathcal { A } _ { t } |$ is treated as a constant at step t: $\frac { \partial \hat { H } _ { t } } { \partial z _ { j } } = \frac { 1 } { \log | \mathcal { A } _ { t } | } \cdot \frac { \partial H _ { t } } { \partial z _ { j } } = - \frac { \pi _ { \theta } ( a _ { j } \mid s _ { t } ) \left( \log \pi _ { \theta } ( a _ { j } \mid s _ { t } ) + H _ { t } \right) } { \log | \mathcal { A } _ { t } | } .$

(15)

d) Step 4: Entropy-gradient saliency.: The saliency score is defined as the ℓ<sub>2</sub>-norm of the gradient vector. Substituting Eq. (15):

$$
\begin{array} { l } { { \displaystyle { G _ { t } = \left\| \nabla _ { { \bf z } _ { t } } { \hat { H } } _ { t } \right\| = \sqrt { \displaystyle \sum _ { a _ { j } \in { \cal A } _ { t } } \left( \frac { \partial { \hat { H } } _ { t } } { \partial z _ { j } } \right) ^ { 2 } } } } } \\ { { \displaystyle { \quad = \frac { 1 } { \log | { \cal A } _ { t } | } \sqrt { \displaystyle \sum _ { a _ { j } \in { \cal A } _ { t } } \pi _ { \theta } \left( a _ { j } \mid s _ { t } \right) ^ { 2 } \left( \log \pi _ { \theta } \left( a _ { j } \mid s _ { t } \right) + H _ { t } \right) ^ { 2 } } . } } } \end{array}\tag{16}
$$

This expression can be computed in closed form from the softmax probabilities and the entropy, requiring no backpropagation through the model.

Fig. 8. Human annotation interface for binary action feedback. Annotators assess action usefulness using the task instruction, trajectory video, and panoramic observations before and after the queried action.

3/10√0

TaskInstruction

Go to the level 1 kitchen and wipe down the black table closest to the lobby

Panoramic observations before and after the action (Step 0 → Step 1) — the red arrow indicates the agent's current heading

![](images/df52668059c76c9b357b4fa48d200f590f02f020f3778a0a79337c11aa99c861.jpg)  
Position before action BEFORE

![](images/b62fa67d3f92b00a01810cf62c17eea30158995ebda18bd0883f04a02134aa91.jpg)

Agent trajectory video  
![](images/f2e7da42c431aea6ff7c7469b991a07f97809ec53b3fb4bbdcc2d9e5b3c4642b.jpg)

Local observation before the action to be annotated / local observation after the action to be annotated

Annotation Question

The agent performed one step of a navigation action (BEFORE → AFTER). Do you think this action helps complete the final task?

No

Yes

← Previous

Submit

Next →