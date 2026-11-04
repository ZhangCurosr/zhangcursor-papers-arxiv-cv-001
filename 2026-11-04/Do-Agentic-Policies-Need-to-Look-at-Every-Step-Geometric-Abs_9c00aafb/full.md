# Do Agentic Policies Need to Look at Every Step? Geometric Abstraction for Reasoning and Action

Yihang Liu<sup>1</sup> Yiming Chen<sup>2</sup> Mingzhe Zheng<sup>3</sup>

<sup>1</sup>Independent Researcher

<sup>2</sup>Georgia Institute of Technology

<sup>3</sup>The Hong Kong University of Science and Technology yihang.liu.pc@gmail.com ychen3868@gatech.edu mzhengar@connect.ust.hk

Abstract: Recent multimodal language models can reason directly about robot scenes, but it is unclear whether they need to repeatedly reinterpret raw visual observations at every step of an interaction. We propose shifting routine spatial reasoning from images to text and symbols through a 3D geometric abstraction that is refreshed locally from observations and read directly by the language models. Specifically, our interface represents scene geometry via typed, evidenceaware symbolic references to object parts, grounds them from calibrated RGB-D, and returns compact textual checkpoints with metric residuals after each action. Symbolic task structure can thus be reused across decisions while metric values are rebound to the current scene, with direct visual reasoning invoked only on demand. Across offline policy synthesis and continuous online execution, reasoning over grounded text and symbols improves multi-step manipulation completion while substantially reducing repeated multimodal queries, positioning explicit geometric state as an effective substrate for agentic robot policies.

Project page: https://geometric-abstraction.github.io

Keywords: Agentic Robotics, 3D Abstractions for Robotics, Spatial Reasoning

## 1 Introduction

Recent frontier multimodal models have demonstrated increasingly strong 3D spatial reasoning and, in some cases, can directly drive robot manipulation without task-specific policy training [1, 2, 3, 4]. However, deploying such models in closed-loop manipulation often requires repeatedly reinterpreting visual observations as the scene evolves, even though many manipulation decisions are governed by a compact set of task-relevant geometric quantities [5, 6]. Repeatedly recovering these quantities from raw images introduces both reasoning uncertainty and additional computational cost: metric and structural relations remain challenging for vision-language models, whereas explicitly representing spatial information in text can improve reasoning reliability [7, 8, 9, 10]. Moreover, visual tokens often carry redundancy, incurring unnecessary overhead during reasoning [11]. This raises a central question: does a robot agent need to reinterpret the visual scene at every decision step?

Existing interfaces trade off richness and compactness. Raw images retain detailed scene information but require repeated visual interpretation, while conventional perception APIs often expose only coarse object-level quantities such as centers or bounding boxes. ReKep [12] shows that explicit 3D keypoints and relational constraints can provide an effective interface for manipulation. Meanwhile, languagemodel-based robotics is moving toward persistent and structured policies: Code as Policies [13] synthesizes executable programs, ASPIRE [14] extends this paradigm through execution–diagnosis– repair and skill reuse, and Graph-as-Policy [15] represents policies as modular graphs. These works make the policy itself structured and reusable, raising a complementary question: what representation ofthe 3D physical scene should such policies reason over?

![](images/077ef2f5bde4d28c318809cc630128ad31fc54eb7afe49004325a5fb81ce5eea.jpg)  
Figure 1: Geometric abstraction between observation, reasoning, and action. Visual observations are grounded locally into metric 3D geometry and exposed to the language model through symbolic references and compact textual checkpoints. Policies reason in text and symbols over stable geometric roles whose metric values are rebound to the current scene during execution.

To address this question, we propose shifting routine spatial reasoning from images to text and symbols through an explicit 3D geometric abstraction (Fig. 1). Rather than reducing objects to single 3D centroids or vague natural-language descriptions, our abstraction models scene geometry as a structured set of typed 3D object parts—including centers, vertices, edges, and oriented surfaces— annotated with timestamps and evidence status (observed, inferred from shape priors, or unavailable under occlusion). Local perception grounds this 3D state from calibrated RGB-D and exposes it to the language model in two forms. First, symbolic references (e.g., red.bottom\_face, blue.top\_face) and geometric operators (such as surface alignment and support checks) let policies compose actions independently of specific object coordinates. Second, compact textual checkpoints report updated 3D coordinates, metric goal residuals, and evidence status after each action. Separating reusable symbolic structure from scene-dependent metric grounding lets the language model reason directly in text and symbols, invoking visual reasoning on demand when richer scene context is needed.

We study this textual and symbolic 3D interface in two regimes. In policy synthesis, the language model revises an executable policy between rollouts using execution feedback, while geometric grounding and control run locally during rollouts. In online execution, the agent repeatedly reasons and selects skills within a single continuing episode using the updated textual checkpoint after each action, without resetting the scene after failures. Both settings use matched sensors, actuators, and initial skills. Across both regimes, our experiments show that reasoning over grounded text and symbols accelerates policy acquisition, substantially improves multi-step online construction completion, and reduces repeated multimodal queries and interaction latency. We use these two settings to examine when explicit textual and symbolic 3D state is sufficient for manipulation and when direct visual reasoning remains necessary.

Our contributions are threefold: (i) we formulate a 3D geometric abstraction that exposes taskrelevant scene geometry to language models as structured text and symbols—combining typed, evidence-aware part references with reusable symbolic structure and local metric grounding; (ii) we instantiate this abstraction as a reasoning-to-action interface with composable geometric operators, local RGB-D grounding, and compact textual checkpoints; and (iii) we empirically evaluate the interface across policy synthesis and continuous online execution, characterizing its impact on manipulation completion, model interaction cost, and the remaining role of direct visual input.

## 2 Related Work

Geometric representations and spatial reasoning for manipulation. Prior work bridges highlevel reasoning and low-level robot control through structured intermediate representations. Vox-Poser [16] synthesizes composable 3D value maps for motion planning, ReKep [12] formulates manipulation as relational constraints over sparse 3D keypoints, and KALM [17], RT-Affordance [5], and ATK [6] extract task-driven keypoints or affordances to guide policy learning. Meanwhile, recent VLM evaluations show that recovering metric and structural 3D relations directly from images remains error-prone and token-intensive, whereas making spatial relations explicit in text improves reasoning reliability [7, 8, 9, 10, 11]. Rather than optimizing a single-step spatial objective or relying on coarse centroids, we maintain a persistent, evidence-aware 3D scene state of typed object parts (centers, vertices, edges, and oriented faces) that is continuously re-grounded from RGB-D and exposed to the language model as composable symbols and compact text.

![](images/ce249b13e33c3427988a708f98a8fb6a25b1f5f4858bec3f4595f695a7729799.jpg)  
Figure 2: Geometric abstraction between perception and reasoning. RGB-D observations are grounded into object geometry and exposed to the LLM through typed symbolic references and compact textual checkpoints. In policy synthesis, the LLM revises executable policies between rollouts, while perception and control run locally. In online execution, the same interface supplies updated geometry for successive decisions within a continuing episode.

Programmatic, agentic, and closed-loop robot policies. Code as Policies [13] and Inner Monologue [18] pioneered composing perception-control APIs in code and incorporating closed-loop language feedback. Recent agentic frameworks extend this paradigm through coding-agent benchmarks and VLM module orchestration [19, 20], modular policy graphs and evolving skill libraries [15, 21], execution-driven repair in simulation and the real world [14, 22], and hierarchical VLA steering and failure correction [23, 24, 25]. While these works focus on how policies are synthesized, orchestrated, or repaired, we focus on the spatial state interface that closes the loop: re-grounding symbolic references after each action and returning metric goal residuals and evidence status in text so subsequent decisions reason over the observed post-action scene rather than assumed action outcomes.

## 3 Geometric Abstraction as a Reasoning-to-Action Interface

Our interface represents scene geometry through typed symbolic references that are grounded in current RGB-D observations at execution time. Updated geometry and goal residuals are returned as compact textual checkpoints for subsequent reasoning (Fig. 2).

## 3.1 Grounding geometry from visual observations

At time t, calibrated RGB-D observations from the side and wrist cameras, together with robot proprioception $q _ { t } ,$ , define the current geometric state:

$$
\begin{array} { r } { o _ { t } = \big ( \{ I _ { t } ^ { c } , D _ { t } ^ { c } , K ^ { c } , T _ { W C } ^ { c } \} _ { c \in \{ \mathrm { s i d e } , \mathrm { w r i s t } \} } , q _ { t } \big ) , \qquad \mathcal { Z } _ { t } = \Phi ( o _ { t } ; M ) , } \end{array}\tag{1}
$$

where M stores object shape information acquired during initialization. Depth observations are back-projected into the calibrated world frame and combined with M to estimate each object’s current center, orientation, dimensions, and, when supported by the observation, its vertices, edges, and faces. Each geometric element additionally records whether it is directly observed, inferred from the initialized shape model, or unavailable. A core invariant is that shape may persist, but pose must be refreshed: after an object moves, its pose must be re-estimated from a current observation rather than copied from the previous state or inferred from commanded motions. If the required geometry cannot be recovered, the corresponding reference is marked unavailable.

Table 1: Operators on the geometric state.
<table><tr><td>Operation</td><td>Inputs</td><td>Output</td></tr><tr><td>reference</td><td>object and role</td><td>geometric entity with evidence</td></tr><tr><td>relation</td><td>predicate and objects</td><td>residuals and true/false/unknown</td></tr><tr><td>align</td><td>source and target geometry</td><td>desired rigid transform</td></tr><tr><td>checkpoint</td><td>active object and goal</td><td>compact textual state</td></tr></table>

## 3.2 Symbolic references and geometric operators

Policies written by the language model access the geometric state through typed references. A reference $r = ( i , \rho )$ identifies object i and a geometric role $\rho$ (such as a center, vertex, segment, or face) and is evaluated against the current state:

$$
[ i . . \rho ] \boldsymbol { \ ] } _ { \mathcal { Z } _ { t } } = \left\{ \begin{array} { l l } { ( v , \tau , \eta , t ) , } & { \mathrm { i f ~ t h e ~ r e f e r e n c e ~ i s ~ c u r r e n t l y ~ s u p p o r t e d } , } \\ { \perp , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{2}
$$

where v is the metric geometric value, τ its type, η its evidence status, and t its observation timestamp.

This turns scene geometry into a compositional vocabulary (Table 1). Instead of reasoning directly from images, a policy can refer to red.bottom\_face and blue.top\_face, align the two faces, and request the corresponding placement transform. The same program generalizes across object configurations because symbolic references are dynamically rebound to the current scene geometry.

For placement, the policy queries the mating face centers and outward normals $( p _ { a } , n _ { a } ) = \operatorname { b o t t o m } ( a )$ and $( p _ { b } , n _ { b } ) = \mathrm { t o p } ( b )$ and computes a rigid transform $( R _ { \Delta } , t _ { \Delta } )$ satisfying

$$
R _ { \Delta } n _ { a } = - n _ { b } , ~ t _ { \Delta } = p _ { b } - R _ { \Delta } p _ { a } .\tag{3}
$$

Since a point and a normal leave rotation about $n _ { b }$ unconstrained, align returns the minimum-angle rotation and reports the remaining twist. This provides a reusable geometric primitive rather than a hardcoded coordinate target. Spatial relations are evaluated similarly: a support query checks relative height, planar overlap, and face orientation, returning both the relation status and its metric residuals. The status is unknown when a required axis or surface is unobserved, and a true result is treated as a geometric candidate rather than a certificate of physical contact or stability.

## 3.3 Textual checkpoints for reasoning

Symbolic references support action composition, while textual checkpoints expose the current state in a compact form for reasoning. For active object i and target position $g _ { i }$ , we write

$$
\begin{array} { r } { x _ { t } = \Psi ( \mathcal { Z } _ { t } , i , g _ { i } ) , \qquad e _ { t } = g _ { i } - p _ { i } ( t ) , } \end{array}\tag{4}
$$

where $x _ { t }$ summarizes the relevant current geometry, its evidence status, and the residual $e _ { t }$ between $g _ { i }$ and the current position $p _ { i } ( t )$ of the selected reference. For example, after placing a block, the checkpoint reports whether the target face is aligned, the remaining translational displacement, and whether the current estimate is visually supported. If the object becomes occluded, the goal is retained but the residual is explicitly marked unavailable rather than computed from a stale pose estimate.

This representation allows successive reasoning steps to reuse grounded geometry without repeatedly sending raw images to the language model, while keeping visual observations available on demand when new scene context is needed or when the geometric abstraction does not explain a failure.

![](images/1af5f4a68e0a06648c0fd9c6e83a9426220814ca92e6576ea5039d3f9caf7b3f.jpg)  
Figure 3: Single-policy acquisition. Policies are revised between reset rollouts and evaluated after training freezes. Geometric Abstraction additionally uses supplied geometric references. Right: illustrative start and goal configurations, constructed in simulation rather than recorded policy trajectories.

## 3.4 Policy synthesis and online execution

We evaluate the same interface in two operational regimes. In policy synthesis (single-policy acquisition), the language model revises an executable program between batches of rollouts; within each rollout, geometry is grounded locally and the program executes without further language-model calls.

In online execution, the language model instead participates directly in the continuing interaction loop:

$$
u _ { t } = \mathcal { A } _ { \mathrm { L L M } } ( g , x _ { t } , \mathcal { H } _ { t } , \mathcal { E } ) ,\tag{5}
$$

where $x _ { t }$ is the current textual checkpoint, H the interaction history, and E available prior skill experience. After each action, new observations re-ground the relevant references, ensuring that recovery decisions start from the observed post-action scene rather than from the intended action outcome. Images remain available on demand when the geometric state is insufficient.

## 4 Experiments

We evaluate the geometric abstraction in simulation with an SO-100 robot arm equipped with calibrated side and wrist RGB-D cameras. Across all experiments, the agent receives the same task specification, sensor access, actuator interface, and initial skills; no simulator object poses or private evaluator state are exposed. We compare a Pixel Reasoning agent that reasons directly from visua observations with a Geometric Abstraction agent that additionally receives our grounded geometric references and textual checkpoints. We study the interface in two regimes: policy synthesis across repeated rollouts and continuous online execution within a single, non-resetting episode.

## 4.1 Single-policy acquisition

In the policy synthesis regime, we test whether geometric abstraction helps an LLM acquire effective single-program policies within a small revision budget. The simulated SO-100 performs three manipulation tasks with randomized object poses and colors (Figure 3): Bridge+cylinder (placing two base blocks, spanning them with a beam, and placing an upright cylinder on top), Three-cube stacking (assembling a three-cube tower in a prescribed color order), and Ring-on-peg (grasping a hexagonal ring and threading it onto a vertical peg).

Comparison and procedure. Pixel Reasoning and Geometric Abstraction share the task specification, calibrated RGB-D access, and action interface, both using gpt-6-astra with high reasoning effort. Pixel Reasoning implements custom perception routines directly in its generated program, whereas Geometric Abstraction additionally queries sensor-grounded 3D part references, geometric operators, and textual checkpoints. Neither condition has access to simulator ground-truth poses or private evaluator state.

![](images/28d1aa68a43a755c4500d39c69fd1d4e95a023c0305b161c06554e0a84f30f39.jpg)  
Figure 4: Test success and acquisition cost. Lines show mean held-out test SR of training-retained policies; bars show cumulative input-plus-output tokens through each iteration, including rejected proposals. Both average over five chains. Ring-on-peg Pixel Reasoning excludes unknown usage from two interrupted requests.

For each task, we run five paired training chains over five shared training layouts with separate persistent LLM contexts. Starting from an observation-only scaffold (R0), the LLM generates three policy revisions (R1–R3). Each revision executes across all five training layouts with a scene reset between rollouts and no LLM calls during execution; rollout outcomes and observations inform the subsequent revision. A candidate policy is retained only if it succeeds on every previously solved training layout plus at least one additional layout; otherwise, the incumbent policy is kept. After training freezes, each retained version is evaluated on 100 shared held-out layouts per task without further revision. Physical success requires achieving the target arrangement, releasing the object, and maintaining stability (Figure 4; Table 2 in Appendix A).

Findings. Geometric abstraction substantially accelerates policy acquisition on both construction tasks. The largest gap appears at the first revision (R1), where Geometric Abstraction reaches 45.2% held-out test SR on Bridge+cylinder (vs. 21.2% for Pixel Reasoning) and 57.2% on Three-cube stacking (vs. 17.2%). By R2, Geometric Abstraction’s mean test SR (63.4% and 72.2%) already surpasses Pixel Reasoning’s final R3 performance (56.0% and 70.8%) on both tasks, while expending roughly half the cumulative tokens (67.3k vs. 125.7k on Bridge+cylinder; 64.7k vs. 138.1k on Stacking). Although the gap narrows at R3 as Pixel Reasoning refines its perception code, Geometric Abstraction retains higher final success (65.4% and 81.4%) and lower chain-to-chain variance, with comparable total tokens on Bridge+cylinder and lower token usage on Stacking. Ring-on-peg follows a different pattern: both conditions remain at 0.0% retained test SR through R2, before Geometric Abstraction reaches 38.0% at R3 while Pixel Reasoning remains at 0.0%. Here, geometric abstraction enables successful policy acquisition within the three-revision budget where image-only synthesis fails to converge, as fine ring-to-peg alignment benefits directly from grounded part references. Note that these gains reflect the combined effect of policy synthesis and local geometric grounding during execution, rather than isolating LLM inference speed or text formatting alone.

## 4.2 Continuous online reasoning and execution

We next evaluate whether geometric abstraction helps an LLM (gpt-6-astra, medium reasoning effort) orchestrate prior skills and recover from failures online under a real-time wall-clock budget (Figure 5). Failed actions are not reset, so model waiting time shares the same 200-second episode budget as physical execution and recovery.

Tasks and protocol. We evaluate 50 paired layouts per task (300 episodes total) across three tasks: Bridge-tower (stacking two cubes atop a constructed bridge), Ring-on-peg (aligning and releasing a pre-lifted ring over a peg), and Tower relocation (transferring the upper two cubes of a column onto a new foundation while preserving order and foundation poses). Both conditions share identical local skills (bridge/stacking primitives, a shared sensor-driven ring grasp-and-lift prefix, and placement routines) and can request camera images and depth samples on demand; Geometric Abstraction additionally receives compact textual checkpoints and symbolic references after each action. Each episode runs continuously for up to 200 s with physics advancing during model waits (see Appendix B–C for full protocol and audited repeat details).

![](images/f78484619a4b76ad23be5cec7d6bb4a73e5aaca36e15ddc32c5151829cb7cf17.jpg)

Figure 5: Online execution and tasks. The agent acts in a single continuing scene until completion or the 200 s timeout; Geometric Abstraction receives grounded symbolic references and textual checkpoints. Right: illustrative start and goal states.  
![](images/4a30208ca16fa1e2e79cc9da88e5c9d04933f6882a3190ac8e94a0472175c1ec.jpg)  
Figure 6: Online outcomes and cost. Completion rate within 200 s (left), mean capped completion time (middle; timeouts assigned 200 s), and mean recorded tokens (right) over 50 paired layouts per task (†: excludes unknown interrupted tokens).

Findings. Geometric Abstraction substantially increases completion rates on multi-step construction, solving Bridge-tower on 28/50 layouts (56% vs. 12/50, 24%) and Tower relocation on 42/50 layouts (84% vs. 15/50, 30%) with lower mean capped completion times (Figure 6; Table 3 in Appendix B). On Ring-on-peg, where both conditions reliably succeed (50/50 vs. 49/50), Geometric Abstraction finishes 33.3% faster (77.2 s vs. 115.8 s), completing earlier on 45 of 50 pairs.

Simultaneously, Geometric Abstraction reduces repeated multimodal querying (Figure 7). On Ringon-peg, model-request time drops by 49.4% (45.6 s vs. 90.0 s), recorded tokens by 52.8% (101.3k vs. 214.5k), completed requests from 11.88 to 6.90, and attached images from 10.94 to 2.72. Tower relocation shows a 60.5% reduction in model time and 52.5% in tokens while physical execution time remains similar (104.5 s vs. 103.2 s). On Bridge-tower, Geometric Abstraction more than doubles completion with 6.7% less model time and 15.1% more tokens, showing that the interface improves performance both by cutting visual query overhead and by enabling effective corrective actions within the time budget. Across all tasks, local RGB-D sensing refreshes the 3D state while drastically reducing image attachments to the LLM (Table 3; Figure 8).

![](images/37d8fb3d34ee3f170f00589321c4ef421e19368f9588e369d8a51db625c9773c.jpg)

![](images/d089ffc6f33d08e44f61dc21bdb437d0b875c9f41bc344527a70a371b5e710b4.jpg)  
Figure 7: Model interaction cost. Mean (bars) and per-episode (markers) request wall-clock time (left) and recorded tokens (right) over 50 paired episodes per task (△: excludes unknown interrupted tokens).

## 5 Limitations and Discussion

Our evaluation focuses on simulated table-top manipulation with regular object geometries. Realworld deployment remains an essential next step to assess robustness under realistic sensor noise, heavy occlusion, and camera calibration drift. We deliberately used lightweight CPU-based geometric estimators to keep local grounding fast and self-contained, leaving learned 3D perception modules and articulated or deformable objects for future work.

In continuous online execution, geometric abstraction reduced decision rounds and on-demand image attachments across tasks, indicating that its primary benefit lies in providing direct, actionable spatia state throughout the reasoning–action loop rather than merely compressing prompt length. More concise checkpoint schemas and learned symbolic abstractions could further reduce interaction overhead. For novel object categories, VLM-assisted shape initialization could broaden coverage, provided its one-time grounding cost is accounted for alongside policy execution and amortized across subsequent interactions.

Finally, decoupling local visual grounding from high-level policy reasoning opens the possibility of driving manipulation with fast text-only language models. By exposing verified, sensor-grounded 3D state in text, the interface allows a language model to reason over physical geometry without processing dense visual tokens at every step. Evaluating diverse models on standardized geometric checkpoints could clarify how effectively language models perform spatial reasoning when metric relations are made explicit, and help delineate which manipulation decisions are well served by structured geometric state versus direct visual inspection.

## 6 Conclusion

We investigated whether agentic robot policies need to reinterpret visual observations at every decision step. Across both offline policy synthesis and continuous online execution, our results demonstrate that maintaining an explicit geometric abstraction improves manipulation completion while substantially reducing repeated multimodal queries. By pairing reusable symbolic references with metric values refreshed from local RGB-D observations, the interface allows visual sensing to remain continuously active in the control loop without requiring the language model to process images at every step—positioning structured geometric state as a practical substrate for agentic reasoning and action, with direct visual reasoning invoked on demand.

## References

[1] W. Zhang, K. Wang, Y. Ouyang, X. Huang, L. Li, K. Su, W. Jin, W. Chai, H. Liang, Z. Dou, et al. An unexpected robot policy: Early evaluations of gpt-6 astra on robodojo and beyond. arXiv preprint arXiv:2609.24170, 2026.

[2] M. Jia, Y. Lin, X. Zhang, Z. Zhang, X. Liu, and M. Jiang. Agent as policy for robotic manipulation. arXiv preprint arXiv:2609.12541, 2026.

[3] W. Tang, D. Jing, J.-H. Pan, Z. Lu, Y.-H. Liu, L. E. Li, M. Ding, and C.-W. Fu. Incentivizing multimodal reasoning in large models for direct robot manipulation. arXiv preprint arXiv:2505.12744, 2025.

[4] C. H. Song, V. Blukis, J. Tremblay, S. Tyree, Y. Su, and S. Birchfield. RoboSpatial: Teaching spatial understanding to 2D and 3D vision-language models for robotics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15768–15780, June 2025.

[5] S. Nasiriany, S. Kirmani, T. Ding, L. Smith, Y. Zhu, D. Driess, D. Sadigh, and T. Xiao. Rtaffordance: Affordances are versatile intermediate representations for robot manipulation. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8249–8257. IEEE, 2025.

[6] Y. Zhang, S. Mittal, Z. Zhang, L. Ke, S. Srinivasa, and A. Gupta. Atk: Automatic task-driven keypoint selection for robust policy learning. arXiv preprint arXiv:2506.13867, 2025.

[7] J. Wang, Y. Ming, Z. Shi, V. Vineet, X. Wang, Y. Li, and N. Joshi. Is a picture worth a thousand words? delving into spatial reasoning for vision language models. Advances in Neural Information Processing Systems, 37:75392–75421, 2024.

[8] A. Pothiraj, E. Stengel-Eskin, J. Cho, and M. Bansal. Capture: Evaluating spatial reasoning in vision language models via occluded object counting. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8001–8010. IEEE, 2025.

[9] Z. Song, X. Lin, Q. Huang, S. Qin, G. Wang, and L. Lin. Siri-bench: Challenging vlms’ spatial intelligence through complex reasoning tasks. arXiv preprint arXiv:2506.14512, 2025.

[10] J. Yang, S. Yang, A. W. Gupta, R. Han, L. Fei-Fei, and S. Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10632–10643. IEEE, 2025.

[11] L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer, 2024.

[12] W. Huang, C. Wang, Y. Li, R. Zhang, and L. Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024.

[13] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International conference on robotics and automation (ICRA), pages 9493–9500. IEEE, 2023.

[14] R. Lu, Y. Wu, E. Kou, L. Fu, W. Xiao, A. Mandlekar, Y. Xu, G. Shi, K. Goldberg, A. Chen, et al. Aspire: Agentic/skills discovery for robotics. arXiv preprint arXiv:2607.00272, 2026.

[15] K. Chen, S. Xie, L. Fu, J. Yu, W. Pacini, S. Bajamahal, H. Kim, J. Drake, D. Kim, H. Xue, et al. Gap: A graph-as-policy multi-agent self-learning harness for variational automation tasks. arXiv preprint arXiv:2607.05369, 2026.

[16] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023.

[17] X. Fang, B.-R. Huang, J. Mao, J. Shone, J. B. Tenenbaum, T. Lozano-Pérez, and L. P. Kaelbling. Kalm: Keypoint abstraction using large models for object-relative imitation learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8307–8314. IEEE, 2025.

[18] W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022.

[19] L. Fu, J. Yu, K. El-Refai, E. Kou, H. Xue, H. Huang, W. Xiao, G. Wang, D. Niu, F.-F. Li, et al. Cap-x: A framework for benchmarking and improving coding agents for robot manipulation. arXiv preprint arXiv:2603.22435, 2026.

[20] J. Shi, R. Yang, K. Chao, B. S. Wan, Y. S. Shao, J. Lei, J. Qian, L. Le, P. Chaudhari, K. Daniilidis, et al. Maestro: Orchestrating robotics modules with vision-language models for zero-shot generalist robots. In NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI, 2025.

[21] D. Fu, C. Jiang, S. Qian, H. Wang, and Z. Hao. Se-gos: Self-evolving graph-of-skills for skill library at scale. arXiv preprint arXiv:2609.08228, 2026.

[22] W. Xiao, J. Xie, T. Zhang, H. Lin, L. Fu, H. Xue, J. Lu, Y. Yang, C. Dai, Z. Wang, et al. Enpire: Agentic robot policy self-improvement in the real world. arXiv preprint arXiv:2606.19980, 2026.

[23] Y. Zhang, H. Zhang, F. Gao, X. Li, Z. Liu, C. Zhu, J. Qiu, Y. Yan, J. Liu, W. Tang, et al. Harness vla: Steering frozen vlas into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448, 2026.

[24] J. Hu, M. Shridhar, C. Lu, D. Shah, H.-T. L. Chiang, J. Tan, and A. Xie. What matters in orchestrating robot policies: A systematic study of hierarchical vla agents. arXiv preprint arXiv:2606.10267, 2026.

[25] O. Kwon, P. Ortega-Kral, A. Bucker, and J. Oh. Training-free action correction for vla model failures via language feedback. arXiv preprint arXiv:2608.29967, 2026.

[26] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Condition</td><td colspan="3">Retained test SR (%)</td><td colspan="3">Cumulative tokens (k)</td></tr><tr><td>R1</td><td>R2</td><td>R3</td><td>R1</td><td>R1-R2</td><td>R1-R3</td></tr><tr><td rowspan="2">Bridge + cylinder</td><td>Pixel Reasoning</td><td> $2 1 . 2 \pm 2 2 . 0$ </td><td> $5 3 . 0 \pm 1 1 . 7$ </td><td> $5 6 . 0 \pm 9 . 1$ </td><td>24.6</td><td>66.2</td><td>125.7</td></tr><tr><td>Geometric Abstraction (ours)</td><td> ${ \bf 4 5 . 2 \pm 2 5 . 4 }$ </td><td> ${ \bf 6 3 . 4 \pm 5 . 9 }$ </td><td> ${ \bf 6 5 . 4 \pm 3 . 5 }$ </td><td>25.8</td><td>67.3</td><td>127.0</td></tr><tr><td rowspan="2">Three-cube stacking</td><td>Pixel Reasoning</td><td> $1 7 . 2 \pm 2 9 . 4$ </td><td> $5 3 . 6 \pm 2 6 . 2$ </td><td> $7 0 . 8 \pm 2 6 . 5$ </td><td>28.0</td><td>74.1</td><td>138.1</td></tr><tr><td>Geometric Abstraction (ours)</td><td> ${ \bf 5 7 . 2 \pm 3 2 . 6 }$ </td><td> ${ \bf 7 2 . 2 \pm 7 . 2 }$ </td><td> ${ \bf 8 1 . 4 \pm 1 2 . 3 }$ </td><td>25.7</td><td>64.7</td><td>123.4</td></tr><tr><td rowspan="2">Ring-on-peg</td><td>Pixel Reasoning</td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td>37.8</td><td>93.5</td><td>166.8</td></tr><tr><td>Geometric Abstraction (ours)</td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> ${ \bf 3 8 . 0 \pm 3 4 . 9 }$ </td><td>36.3</td><td>91.5</td><td>163.1</td></tr></table>

Table 2: Single-policy acquisition outcomes and token usage. Held-out test SR (mean ± sample SD) across five training chains on 100 shared held-out layouts per task, using training-only policy retention. Tokens report cumulative completed-request input plus output per chain (in thousands), including rejected proposals and counting cached input once. Ring-on-peg Pixel Reasoning excludes unknown token usage from two interrupted requests. Bold highlights where Geometric Abstraction (ours) improves over Pixel Reasoning.
<table><tr><td>Task</td><td>Condition</td><td>Success</td><td>Time (s)</td><td>Model (s)</td><td>Tokens (k)</td><td>Requests</td><td>Images</td></tr><tr><td rowspan="2">Bridge-tower</td><td>Pixel Reasoning</td><td>12/50</td><td>198.0</td><td>55.5</td><td>112.5</td><td>7.88</td><td>7.50</td></tr><tr><td>Geometric Abstraction (ours)</td><td>28/50</td><td>185.3</td><td>51.8</td><td>129.5</td><td>7.32</td><td>4.28</td></tr><tr><td rowspan="2">Ring-on-peg</td><td>Pixel Reasoning</td><td>49/50</td><td>115.8</td><td>90.0</td><td>214.5</td><td>11.88</td><td>10.94</td></tr><tr><td>Geometric Abstraction (ours)</td><td>50/50</td><td>77.2</td><td>45.6</td><td>101.3</td><td>6.90</td><td>2.72</td></tr><tr><td rowspan="2">Tower relocation</td><td>Pixel Reasoning</td><td>15/50</td><td>186.4</td><td>80.3</td><td>185.9</td><td>11.58</td><td>11.34</td></tr><tr><td>Geometric Abstraction (ours)</td><td>42/50</td><td>139.2</td><td>31.7</td><td>88.3</td><td>5.36</td><td>3.00</td></tr></table>

Table 3: Online outcomes and interaction cost. Each task evaluates 50 paired layouts. Success reports completed episodes out of 50; all other columns report per-episode means across all 50 episodes, including timeouts. Time is capped completion time, assigning 200 seconds to timeouts. Model is cumulative request wall-clock time, including interrupted waits. Tokens are recorded input plus output tokens (in thousands), counting cached input once and excluding unknown interrupted usage. Requests counts completed model calls; Images counts newly attached camera images across all request attempts (excluding local sensing frames not sent to the LLM). Both conditions use audited repeats for nine runtime-flagged pairs. Bold highlights where Geometric Abstraction (ours) improves over Pixel Reasoning.

## A Additional single-policy results

Evaluation notes. All experiments use MuJoCo 3.11.0 [26] with a simulated SO-100 robot arm. Bridge+cylinder and Three-cube stacking require achieving the prescribed support structure and maintaining a five-second stable hold after release. Ring-on-peg requires the peg to pass through the ring opening, the ring center to lie below the peg top, full release from robot contact, and a two-second stable hold. Syntactically or runtime-invalid programs count as failures. Bridge+cylinder and Stacking contact checks can miss unnamed robot collision geometries, whereas Ring-on-peg uses full body-ancestry contact checks.

## B Additional online results

Table 3 and Figure 8 expand the online evaluation results presented in Section 4.2.

Evaluation notes. Bridge-tower and Tower relocation require a five-second stable hold; Ring-onpeg requires two seconds. Tower relocation must preserve or restore both foundation cubes within 5 mm of their initial positions. Ring-on-peg executes the same sensor-driven grasp-and-lift prefix in both conditions and hands control to the online agent prior to alignment and release; completion times and all interaction costs include this shared prefix. Each layout is attempted once per condition (with audited repeats replacing nine pairs affected by runtime timing or GPU faults). Because paired episodes can differ in outcome, completion-time means include all 50 episodes (with timeouts capped at 200 s) rather than conditioning on successful episodes only. Model-request time reflects end-to-end API waiting time rather than isolated model compute.

![](images/6ad112e2db4921cf776e705ea8d917118f17a5d32f888efa24f1c8ac86d49108.jpg)  
Figure 8: Cumulative success over elapsed wall-clock time. Each curve summarizes the 50 paired 200-second episodes for Pixel Reasoning and Geometric Abstraction (ours), stepping at the first verified completion time. Timeouts do not convert to successes at the 200 s deadline. Intermediate values are read directly from the 200-second episodes rather than separate evaluations under shorter budgets.

## C Geometric abstraction implementation

The geometric abstraction converts calibrated RGB-D observations into metric entities, symbolic references, and compact textual checkpoints. It runs locally on the CPU using fixed geometric estimators throughout both policy synthesis and online execution.

From pixels to metric geometry. Color segmentation masks associate visible pixels with scene objects. For a valid pixel $\tilde { \mathbf { u } } = ( u , v , 1 ) ^ { \top }$ , its 3D world position is recovered via back-projection:

$$
\mathbf { p } _ { W } = R _ { W C } \big ( D ( u , v ) K ^ { - 1 } \tilde { \mathbf { u } } \big ) + \mathbf { t } _ { W C } ,\tag{6}
$$

where $D ( u , v )$ is optical-axis depth, K is the camera intrinsic matrix, and $\left( R _ { W C } , \mathbf { t } _ { W C } \right)$ transforms camera coordinates to the world frame. Background depth points estimate the tabletop plane, while known robot link geometry masks out the manipulator. Cuboid and cylinder estimators fit object pose and visible support surfaces using dimensions initialized from RGB-D under declared shape and table-support priors. Simulator ground-truth object poses and dimensions are never accessed.

References and current evidence. Symbolic references such as red.top\_face resolve to typed geometric primitives: points, axes, or oriented faces with centers and outward normals. Each record stores world-frame coordinates, physical units, observation timestamps, and evidence status (observed, inferred, or unavailable). Initialized shape dimensions persist across an episode, whereas a movable object’s pose must be refreshed from current visual evidence after movement. For partially occluded rings, initialized boundary templates are registered to currently visible contour edges; underconstrained or ambiguous fits return unavailable. Unobserved object faces retain model-derived labels marked as inferred, while the stationary peg may retain an earlier verified measurement with its original timestamp. Geometric support predicates report metric residuals and missing evidence rather than certifying physical contact or dynamic stability.

From geometry to agent input. The Geometric Abstraction agent receives task-relevant object geometry, symbolic references, evidence metadata, and requested goal residuals formatted as structured text (with dense boundary point arrays omitted from online Ring-on-peg checkpoints for compactness). Multiple geometric queries can reuse a single RGB-D capture, and local sensing can refresh the geometric state after an action without attaching new images to the LLM prompt. Initial and on-demand camera images remain accessible whenever visual inspection is requested. In policy synthesis, generated programs invoke geometric operators locally during rollouts; in online execution, the agent reads updated textual checkpoints to select its next skill or recovery action.