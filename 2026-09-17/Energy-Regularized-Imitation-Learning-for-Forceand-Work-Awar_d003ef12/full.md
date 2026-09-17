# Energy-Regularized Imitation Learning for Forceand Work-Aware Robotic Manipulation

Toshiki Otani, Hiromu Taketsugu, Norimichi Ukita

Toyota Technological Institute, Japan

Abstract. This paper studies energy-aware manipulation as a physically grounded learning problem. We define a joint-space mechanicalwork proxy from joint torque and angular displacement, and train a differentiable energy predictor that estimates this work from robot states and actions. The predictor converts a non-diferentiable simulator-side physical quantity into a diferentiable regularizer for fine-tuning a pretrained manipulation policy. We instantiate the framework with RVT-2 on RLBench and evaluate 12 manipulation tasks involving object contact, articulated motion, placement, pushing, and sweeping. The proposed fine-tuning reduces the average mechanical work from 208.8 J to 204.4 J (i.e., 2.1% reduction), while the mean task success rate also increases slightly from 86.2% to 86.9%. These results show that work-aware policy optimization can suppress physically ineficient motion without requiring an explicit diferentiable dynamics model.

## 1 Introduction

Imitation learning, including PerAct [21], Q-Transformer [4], and RVT-2 [8], can learn object manipulation from demonstrations. Their training objectives are usually defined around imitation accuracy without explicitly evaluating the physical efort required by the executed motion. A policy may therefore complete a task through redundant accelerations, unnecessarily large joint motion, or avoidable contact with the environment. Figure 1 shows this contrast.

This issue is relevant to force-grounded articulated manipulation, where task completion alone is insuficient to characterize the physical quality of a robot motion. In contact-rich settings, force, torque, and mechanical work are central quantities because they determine how much physical efort is exerted on the robot, the object, and the surrounding environment. In this paper, we address this problem from the policy-learning side by using mechanical work as an additional training signal for a manipulation policy.

Directly using mechanical work as a training loss is dificult because torque and contact computations are produced inside physics engines [19, 23]. These computations are not diferentiable functions of a high-level neural policy. In real robots, dynamics parameters and contact states are also dificult to identify. We therefore introduce a learned surrogate predictor that converts simulator-side work measurements into a diferentiable objective.

![](images/7ab53b75c3b99f9916b8c1c1df903a13128fbee2ba8a2c3c07d7721b7d5a57f3.jpg)  
Fig. 1: A manipulation task can be completed with either (Upper) a short, physically eficient trajectory or (Lower) a redundant trajectory involving unnecessary contact.

We propose energy-regularized imitation learning. First, we compute actionlevel work by accumulating joint-space work over the simulator steps required to execute each key action. Second, we train an energy predictor that approximates this quantity from robot states and actions. Third, we freeze the predictor and fine-tune a pretrained policy network with an additional energy loss.

Our contributions are threefold. (1) We formulate manipulation energy consumption as mechanical work computed from joint torque and angular displacement, aligning policy learning with force-, torque-, and work-aware manipulation. (2) We introduce a diferentiable energy predictor and integrate it into a pretrained imitation-learning policy as a regularizer. (3) We evaluate the method on 12 RLBench tasks and show an average 2.1% reduction in mechanical work with no degradation in average success rate.

## 2 Related Work

Imitation learning for visual manipulation. Behavioral cloning learns a policy from expert demonstrations [2]. It is simple and data eficient, yet suffers from distribution shift when the policy visits states outside that distribution [20]. Recent methods [1, 8, 15–17, 21] improve policy expressivity through difusion models, transformers, 3D scene representations, and multi-view observations. These methods mainly optimize action prediction and task success. Recent reinforcement-learning methods incorporate torque- and mechanical-energy penalties into policy optimization for reaching, box pushing, mobile manipulation, and articulated-object manipulation [6,7,18,22]. Unlike these RL methods, our method learns a diferentiable surrogate of simulator-computed joint-space work and uses it to fine-tune a pretrained visual imitation-learning policy.

Force-, torque-, and work-aware manipulation. Classical robot control uses dynamics models, torque limits, and trajectory optimization to reduce control cost or mechanical efort [13]. In learned manipulation, however, exact dynamics are dificult to obtain for contact-rich tasks. Existing benchmarks such as RL-Bench [11] and Meta-World [25] mainly evaluate task success. We instead use mechanical work as an additional optimization target.

Surrogate physical models. When a physical quantity is computed inside a simulator, it is often unavailable as a diferentiable function of a neural policy. Neural surrogate models can approximate such quantities and provide gradients for policy optimization [9,12,14]. In our work, the surrogate predicts manipulator work from robot states and actions, without simulating full contact dynamics.

## 3 Method

## 3.1 Mechanical Work as the Energy Target

We define the energy target as a joint-space mechanical-work proxy computed from joint torque and joint displacement. Let J be the number of robot joints and $T$ be the number of control steps. At step t, the simulator provides the joint torque vector $\boldsymbol { \tau } _ { t } \in \mathbb { R } ^ { J }$ and joint angle vector $\pmb q _ { t } \in \mathbb { R } ^ { J }$ . The per-step mechanical work $e _ { t }$ and the trajectory-level work $E ( \tau _ { 1 : T } , \pmb { q } _ { 0 : T } )$ are defined as follows:

$$
\begin{array} { r } { e _ { t } = \sum _ { j = 1 } ^ { J } | \tau _ { t , j } | | q _ { t , j } - q _ { t - 1 , j } | , } \end{array}\tag{1}
$$

$$
\begin{array} { r } { E ( \tau _ { 1 : T } , \pmb { q } _ { 0 : T } ) = \sum _ { t = 1 } ^ { T } e _ { t } . } \end{array}\tag{2}
$$

The policy network predicts key actions at a higher level than the simulator control step. Let $\mathcal { T } _ { k }$ be the set of simulator steps required to execute key action k. The action-level work is defined as follows:

$$
\begin{array} { r } { E _ { \mathrm { a c t } } ( k ) = \sum _ { t \in \mathcal { T } _ { k } } e _ { t } . } \end{array}\tag{3}
$$

These definitions follow the physical form of work, torque multiplied by angular displacement. The proposed quantity is an absolute joint-space work proxy for manipulator efort, rather than the net mechanical work transferred to the object or the electrical energy consumed by the actuators. It is complementary to contact-gated work metrics, which integrate force along distance during contact. We use this joint-space formulation because RLBench provides robot joint states and simulator-side torques for all tasks.

Although this work measure is physically meaningful, it is not directly usable as a gradient-based loss for a high-level policy. The torque values are computed inside the simulator after executing actions. Contacts, collision handling, and low-level control introduce discontinuities, and the simulator is not part of the neural computation graph. We therefore learn a diferentiable approximation of the action-level work in Eq. (3), as described in Sec. 3.2.

## 3.2 Diferentiable Energy Predictor

Let $\mathbf { s } _ { k }$ denote the robot state at key action $k ,$ and let ${ \bf a } _ { k }$ denote an input key action. We train an energy prediction network parameterized by $\phi$ (denoted by $g _ { \phi } )$

$$
\hat { E } _ { \phi } = g _ { \phi } ( \mathbf { s } _ { k } , \mathbf { a } _ { k } )\tag{4}
$$

![](images/6513f4bf0e33957c4b653ac114e36a44067d161eaed1d1792ed8ed0575c4ec1c.jpg)  
Fig. 2: Network architecture of our energy prediction network.

$g _ { \phi }$ estimates the action-level work in Eq. (3). The predictor is trained on demonstration key actions, using simulator-computed work labels obtained during action execution. It does not take simulator torque as input; torque and joint displacement are used only to compute the supervised work label. Figure 2 shows the predictor architecture. The predictor input consists of a 21-D robot-state vector, a 9-D action, and a 1-D step-count feature. Specifically, the 21-D robot-state vector consists of the angles and angular velocities of seven joints (14-D), the end-efector xyz position (3-D), and the end-efector quaternion (4-D). The 9-D action consists of the target xyz position (3-D), target quaternion (4-D), gripper open/close command (1-D), and collision flag (1-D). The 1-D step-count feature denotes the current task-step index and provides task-progress information to the non-recurrent predictor. This predictor is trained with RAdamSchedule-Free [5], a learning rate of $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 1024, and early stopping.

We use a Huber loss to make $\hat { E } _ { \phi }$ robust to rare high-energy episodes:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e d } } ( \phi ) = \mathrm { H u b e r } \left( g _ { \phi } ( \mathbf { s } _ { k } , \mathbf { a } _ { k } ) , E _ { \mathrm { a c t } } ( k ) \right) . } \end{array}\tag{5}
$$

Before training the predictor, we filter atypical successful steps using HDB-SCAN [3]. For each task, clustering is performed in a two-dimensional space of arm-motion magnitude and measured action-level work. Atypical successful behaviors, such as re-grasping, repeated approaches, accidental contact, or temporary stuck states, often deviate from the normal motion–work correlation by producing high torque with small motion. Samples in low-density outlier regions are removed before predictor training, keeping the predictor focused on representative successful motions.

## 3.3 Energy-Regularized Fine-Tuning

We integrate $g _ { \phi }$ into policy-network fine-tuning. Figure 3 shows the overall finetuning framework. Let $\mathcal { L } _ { \mathrm { t a s k } }$ be the original imitation-learning loss that supervises the predicted action $\hat { \mathbf { a } } _ { k }$ with the expert action $\mathbf { a } _ { k } ^ { * }$ from the demonstration. During fine-tuning, $g _ { \phi }$ is frozen and only the manipulation policy parameters θ are updated. At each training step, the policy network predicts an action $\hat { \mathbf { a } } _ { k } = \pi _ { \theta } ( \mathbf { o } _ { k } )$ , which is supervised by the corresponding expert action $\mathbf { a } _ { k } ^ { * }$ from the demonstration. The same predicted action, together with the robot state, is fed to the frozen $g _ { \phi }$ to compute the work regularization term.

![](images/3e809d8361c1cc8df19373ee0f869ec541ab848d2049505cb9551b6d66279311.jpg)  
Fig. 3: Overview of energy-regularized fine-tuning.

The energy loss is the min-max normalized predicted work using Eq. (4):

$$
\mathcal { L } _ { \mathrm { e n e r g y } } = \frac { \hat { E } _ { \phi } - E _ { \mathrm { m i n } } } { E _ { \mathrm { m a x } } - E _ { \mathrm { m i n } } } .\tag{6}
$$

where $E _ { \mathrm { m i n } }$ and $E _ { \mathrm { m a x } }$ denote fixed constants computed once as the minimum and maximum of $E _ { \mathrm { a c t } } ( k )$ over the action-level work labels in the energy-predictor training set; they are kept fixed during policy fine-tuning. The total fine-tuning loss is defined with Eq. (6):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t o t a l } } ( \theta ) = \mathcal L _ { \mathrm { t a s k } } \big ( \hat { \mathbf { a } } _ { k } , \mathbf { a } _ { k } ^ { * } \big ) + \lambda \mathcal L _ { \mathrm { e n e r g y } } , } \end{array}\tag{7}
$$

where λ denotes a weight that controls the tradeof between action imitation and work reduction. We use λ = 10 as the main setting. Fine-tuning uses LAMB [24], learning rate $1 0 ^ { - 4 }$ , a batch size of 2 per GPU, cosine annealing with warmup [10], and early stopping. Since the predictor is frozen, the policy cannot reduce the loss by changing the physical model. It must instead produce actions that the predictor estimates as lower work.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate on 12 RLBench tasks [11]: Close Jar, Light Bulb In, Open Drawer, Place Shape in Shape Sorter, Place Wine at Rack Location, Push Buttons, Put Item in Drawer, Reach and Drag, Stack Blocks, Stack Cups, Sweep to Dustpan of Size, and Turn Tap. These tasks cover articulated objects, placing, button interaction, reaching, stacking, sweeping, and object rearrangement. We use RVT-2 as the baseline policy and report task success rate and mechanical work in Joules. For evaluation, Work[J] denotes the trajectory-level work in Eq. (2), computed after each policy rollout and averaged over successful episodes only.

Table 1: Energy-predictor accuracy on 12 RLBench tasks. Rel. denotes MAE divided by the mean ground-truth action-level work.
<table><tr><td>Task</td><td>MAE[J]</td><td>Rel.[%] |</td><td>|Task</td><td>MAE[J]</td><td>Rel.[%]</td></tr><tr><td>Close Jar</td><td>1.221</td><td>4.05</td><td>Put Item Drawer</td><td>0.862</td><td>3.16</td></tr><tr><td>Light Bulb In</td><td>4.775</td><td>13.81</td><td>Reach and Drag</td><td>1.178</td><td>5.72</td></tr><tr><td>Open Drawer</td><td>4.136</td><td>11.71</td><td>Stack Blocks</td><td>1.198</td><td>5.15</td></tr><tr><td>Shape Sorter</td><td>4.787</td><td>14.59</td><td>Stack Cups</td><td>0.878</td><td>4.33</td></tr><tr><td>Place Wine</td><td>2.864</td><td>7.47</td><td>Sweep Dustpan</td><td>0.565</td><td>2.13</td></tr><tr><td>Push Buttons</td><td>0.211</td><td>1.86</td><td>Turn Tap</td><td>6.184</td><td>10.20</td></tr><tr><td></td><td></td><td></td><td>Average</td><td>2.40</td><td>7.02</td></tr></table>

Table 2: Comparison between RVT-2 and the proposed energy-regularized policy (λ = 10). Work is averaged over successful episodes only.
<table><tr><td rowspan="2">Task</td><td colspan="2">Success[%]</td><td colspan="2">Work[J]</td><td rowspan="2">Red.[%]</td></tr><tr><td>RVT-2</td><td>Ours</td><td>RVT-2</td><td>Ours</td></tr><tr><td>Close Jar</td><td>100</td><td>100</td><td>171.11</td><td>170.71</td><td>0.2</td></tr><tr><td>Light Bulb In</td><td>89</td><td>88</td><td>223.02</td><td>222.22</td><td>0.4</td></tr><tr><td>Open Drawer</td><td>78</td><td>82</td><td>131.23</td><td>131.44</td><td>-0.2</td></tr><tr><td>Shape Sorter</td><td>41</td><td>47</td><td>260.69</td><td>219.48</td><td>15.8</td></tr><tr><td>Place Wine</td><td>89</td><td>91</td><td>207.00</td><td>205.84</td><td>0.6</td></tr><tr><td>Push Buttons</td><td>100</td><td>100</td><td>115.40</td><td>115.45</td><td>0.0</td></tr><tr><td>Put Item Drawer</td><td>96</td><td>93</td><td>357.04</td><td>348.78</td><td>2.3</td></tr><tr><td>Reach and Drag</td><td>100</td><td>100</td><td>145.12</td><td>145.02</td><td>0.1</td></tr><tr><td>Stack Blocks</td><td>70</td><td>74</td><td>403.39</td><td>408.15</td><td>-1.2</td></tr><tr><td>Stack Cups</td><td>74</td><td>72</td><td>238.90</td><td>240.31</td><td>-0.6</td></tr><tr><td>Sweep Dustpan</td><td>100</td><td>100</td><td>125.60</td><td>125.51</td><td>0.1</td></tr><tr><td>Turn Tap</td><td>97</td><td>96</td><td>126.50</td><td>119.92</td><td>5.2</td></tr><tr><td>Average</td><td>86.2</td><td>86.9</td><td>208.8</td><td>204.4</td><td>2.1</td></tr></table>

## 4.2 Energy Predictor Accuracy

Table 1 reports the prediction accuracy of the energy predictor. The predictor achieves an average mean absolute error (MAE) of 2.40 J and an average relative error of 7.02% across 12 tasks. Errors are smallest for button interaction, sweeping, and simple reaching tasks, and larger for articulated-object manipulation and long-horizon placement tasks. This accuracy is suficient for regularization because the fine-tuning objective mainly requires a stable low-versus-high work signal rather than exact physical reconstruction at every step.

## 4.3 Energy-Regularized Fine-Tuning

Table 2 compares the baseline RVT-2 policy and our energy-regularized finetuned policy with λ = 10. The proposed method reduces the average mechanical work from 208.8 J to 204.4 J, corresponding to a 2.1% reduction, while the average success rate also increases from 86.2% to 86.9%. This suggests that the energy term acts as a mild eficiency prior without harming task completion. The reduction is conservative because the policy is already pretrained for high success and the regularization weight is kept moderate.

Table 3: Ablation of atypical-data filtering over the 12 RLBench tasks. Representative dificult and simple tasks are selected by the largest and smallest filtered predictor errors, respectively.
<table><tr><td rowspan="2">Task group</td><td colspan="2">Predictor MAE[J]</td><td colspan="2">Fine-tuned work[J]</td></tr><tr><td>Filtered</td><td>Unfiltered</td><td>Filtered</td><td>Unfiltered</td></tr><tr><td>Average over 12 tasks</td><td>2.40</td><td>2.80</td><td>204.4</td><td>214.8</td></tr><tr><td>Four difficult tasks</td><td>4.97</td><td>6.04</td><td>173.3</td><td>202.7</td></tr><tr><td>Four simple tasks</td><td>0.63</td><td>0.74</td><td>207.5</td><td>207.7</td></tr></table>

The energy reduction is large for tasks where the baseline tends to produce redundant motion or unnecessary contact. Shape Sorter reduces work by 15.8%, Turn Tap by 5.2%, and Put Item Drawer by 2.3%. Several tasks preserve success completely while slightly reducing work, including Close Jar, Reach and Drag, and Sweep Dustpan. Some tasks show a negative energy reduction, such as Open Drawer, Stack Blocks, and Stack Cups, indicating that the regularizer does not uniformly improve all manipulation modes. These failures suggest that taskdependent contact strategies and success constraints should be incorporated into future energy-aware losses.

## 4.4 Ablation Studies

Efect of atypical-data filtering. Table 3 shows the efect of filtering atypical successful trajectories before predictor training. HDBSCAN filtering improves the average predictor MAE from 2.80 J to 2.40 J over the 12 tasks. We further report group averages for representative dificult and simple tasks to examine whether this efect holds across diferent prediction-error regimes. The dificult group contains the four tasks with the largest filtered predictor errors: Turn Tap, Place Shape in Shape Sorter, Light Bulb In, and Open Drawer. The simple group contains the four tasks with the smallest filtered predictor errors: Push Buttons, Sweep to Dustpan of Size, Put Item in Drawer, and Stack Cups. The baseline policy (i.e., RVT-2) has a mean work of 208.8 J, as shown in Table 2. Fine-tuning with the unfiltered predictor increases it to 214.8 J, whereas fine-tuning with the filtered predictor reduces it to 204.4 J, as shown in Table 3. This supports the importance of estimating energy from representative successful behavior rather than from all successful rollouts indiscriminately.

Using internal RVT-2 features. We also tested an energy predictor that uses internal RVT-2 features in addition to robot state and action features. This improves the average predictor MAE from 2.40 J to 2.30 J. However, the downstream fine-tuning does not improve average work: the mean work becomes 207.8 J with an average success rate of 87.3%. This contrast indicates that predictor MAE alone is not the only criterion for an efective regularizer. The predictor must also provide gradients that are well aligned with controllable action changes.

Sensitivity to the energy weight. The coeficient λ controls the tradeof in Eq. (7). Moderate weights can reduce work while preserving task completion for several tasks. Excessively large weights harm success, especially in contact-rich insertion and articulated-object tasks, because the policy can minimize predicted work by avoiding decisive contact or by producing overly conservative actions. In our experiments, λ = 10 ofers the clearest average work reduction, while λ = 5 and λ = 7 are often safer for success-critical tasks.

## 5 Discussion

Our method does not modify the visual encoder of the base policy. Instead, it adds a physically grounded supervision signal to the action prediction head. The approach is complementary to visual representation learning and can be applied to visual manipulation policies without changing their perception backbone.

The proposed method connects imitation learning with physically grounded manipulation in two ways. First, it uses a work quantity that depends on joint torque and motion, rather than a purely kinematic smoothness penalty. Smooth motion can still be ineficient when the robot pushes against an object or applies unnecessary torque. Second, our method directly targets policy learning. The predictor is trained from physics simulation and then used as a diferentiable energy term, avoiding the need to backpropagate through contact simulation.

The current study has limitations. The work definition is joint-space work and does not explicitly gate object contact. Thus, the metric captures the robot’s mechanical efort rather than the exact energy transferred to the object. For force-grounded manipulation, a natural extension is to combine joint-space work with contact detection and force/torque sensing. The success-rate drop on several tasks shows that energy regularization should be constrained by task-specific contact requirements. For example, insertion and stacking may require high work to satisfy geometric constraints. Future work should use contact-aware losses, task-conditioned weights, and energy-predictor uncertainty estimates.

## 6 Conclusion

We presented energy-regularized imitation learning for force- and work-aware manipulation. Our method defines a joint-space mechanical-work proxy from joint torque and angular displacement, learns a diferentiable predictor, and uses the frozen predictor as an additional loss to fine-tune a pretrained policy. On 12 RLBench tasks, it reduces average mechanical work from 208.8 J to 204.4 J, while the mean success rate slightly increases from 86.2% to 86.9%. This modest reduction reflects the conservative role of the energy term, which regularizes a pretrained high-success policy without harming task performance. The results show that physically meaningful work estimates can be incorporated into imitation learning without a diferentiable simulator. This direction aligns with force-grounded manipulation, where torque, contact, and mechanical work provide more deployment-relevant signals than task success alone.

## References

1. Aizu, T., Oba, T., Kondo, Y., Ukita, N.: Robot motion planning using one-step diffusion with noise-optimized approximate motions. CoRR abs/2504.19652 (2025) 2

2. Argall, B.D., Chernova, S., Veloso, M., Browning, B.: A survey of robot learning from demonstration. Robotics and Autonomous Systems 57(5), 469–483 (2009) 2

3. Campello, R.J., Moulavi, D., Zimek, A., Sander, J.: Hierarchical density estimates for data clustering, visualization, and outlier detection. ACM Transactions on Knowledge Discovery from Data 10(1), 1–51 (2015) 4

4. Chebotar, Y., Vuong, Q., Hausman, K., Xia, F., Lu, Y., Irpan, A., Kumar, A., Yu, T., Herzog, A., Pertsch, K., Gopalakrishnan, K., Ibarz, J., Nachum, O., Sontakke, S.A., Salazar, G., Tran, H.T., Peralta, J., Tan, C., Manjunath, D., Singh, J., Zitkovich, B., Jackson, T., Rao, K., Finn, C., Levine, S.: Q-Transformer: Scalable ofline reinforcement learning via autoregressive q-functions. In: Conference on Robot Learning (2023) 1

5. Defazio, A., Yang, X., Mehta, H., Mishchenko, K., Khaled, A., Cutkosky, A.: The road less scheduled. In: Advances in Neural Information Processing Systems (2024) 4

6. Deniz, N.N., Parsons, S., Auat Cheein, F.: Energy-eficient arm reaching for a humanoid robot via deep reinforcement learning with identified power models (2026) 2

7. Fu, Z., Cheng, X., Pathak, D.: Deep whole-body control: Learning a unified policy for manipulation and locomotion. In: CORL (2022) 2

8. Goyal, A., Blukis, V., Xu, J., Guo, Y., Chao, Y.W., Fox, D.: RVT-2: Learning precise manipulation from few demonstrations. In: Robotics: Science and Systems (2024) 1, 2

9. Hu, Y., Anderson, L., Li, T.M., Sun, Q., Carr, N., Ragan-Kelley, J., Durand, F.: Diftaichi: Diferentiable programming for physical simulation. In: International Conference on Learning Representations (2020) 3

10. Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J.E., Weinberger, K.Q.: Snapshot ensembles: Train 1, get M for free. In: International Conference on Learning Representations (2017) 5

11. James, S., Ma, Z., Arrojo, D.R., Davison, A.J.: RLBench: The robot learning benchmark and learning environment. IEEE Robotics and Automation Letters 5(2), 3019–3026 (2020) 2, 5

12. Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., Yang, L.: Physics-informed machine learning. Nature Reviews Physics 3, 422–440 (2021) 3

13. Kufner, J.J., LaValle, S.M.: RRT-Connect: An eficient approach to single-query path planning. In: IEEE International Conference on Robotics and Automation. pp. 995–1001 (2000) 2

14. Lusch, B., Kutz, J.N., Brunton, S.L.: Deep learning for universal linear embeddings of nonlinear dynamics. Nature Communications 9(1), 4950 (2018) 3

15. Oba, T., Ukita, N.: Data-driven stochastic motion evaluation and optimization with image by spatially-aligned temporal encoding. In: ICRA (2023) 2

16. Oba, T., Ukita, N.: R2-dif: Denoising by difusion as a refinement of retrieved motion for image-based motion prediction. Neurocomputing 647, 130489 (2025) 2

17. Oba, T., Walter, M.R., Ukita, N.: READ: retrieval-enhanced asymmetric difusion for motion planning. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024) 2

18. Otto, F., Celik, O., Zhou, H., Ziesche, H., Ngo, V.A., Neumann, G.: Deep black-box reinforcement learning with movement primitives. In: CORL (2022) 2

19. Rohmer, E., Singh, S.P.N., Freese, M.: V-rep: A versatile and scalable robot simulation framework. In: IEEE/RSJ International Conference on Intelligent Robots and Systems (2013) 1

20. Ross, S., Gordon, G., Bagnell, D.: A reduction of imitation learning and structured prediction to no-regret online learning. In: International Conference on Artificial Intelligence and Statistics. pp. 627–635 (2011) 2

21. Shridhar, M., Manuelli, L., Fox, D.: Perceiver-actor: A multi-task transformer for robotic manipulation. In: Conference on Robot Learning (2022) 1, 2

22. Tao, X., Wang, Y., Ding, H., Qi, Y., Song, Z.: Energy-aware reinforcement learning for robotic manipulation of articulated components in infrastructure operation and maintenance. Computer-Aided Civil and Infrastructure Engineering 46, 100015 (2026) 2

23. Todorov, E., Erez, T., Tassa, Y.: MuJoCo: A physics engine for model-based control. In: IEEE/RSJ International Conference on Intelligent Robots and Systems. pp. 5026–5033 (2012) 1

24. You, Y., Li, J., Reddi, S., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X., Demmel, J., Keutzer, K., Hsieh, C.J.: Large batch optimization for deep learning: Training BERT in 76 minutes. In: International Conference on Learning Representations (2020) 5

25. Yu, T., Quillen, D., He, Z., Julian, R., Hausman, K., Finn, C., Levine, S.: Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning. In: Conference on Robot Learning (2019) 2