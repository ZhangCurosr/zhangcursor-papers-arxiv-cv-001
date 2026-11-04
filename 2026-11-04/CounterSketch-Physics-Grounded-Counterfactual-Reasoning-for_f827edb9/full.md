# CounterSketch: Physics-Grounded Counterfactual Reasoning for Sketch-to-Video Generation

Hien Trinh<sup>1,2</sup> <sup>⋆</sup>, Tan-Hiep To<sup>1,2</sup> <sup>⋆</sup>, Dinh-Khoi Vo<sup>1,2</sup> , Tam V. Nguyen<sup>3</sup> , Minh-Triet Tran<sup>1,2</sup> , and Trung-Nghia Le<sup>1,2</sup> <sup>⋆⋆</sup>

<sup>1</sup> University of Science, Ho Chi Minh City, Vietnam <sup>2</sup> Vietnam National University, Ho Chi Minh City, Vietnam 3 University of Dayton, Ohio, USA {23120254, 25C1103979}@student.hcmus.edu.vn, tamnguyen@udayton.edu, {tmtriet, ltnghia}@fit.hcmus.edu.vn

Abstract. Counterfactual sketch animation explores how a static sketch evolves under a hypothetical intervention. We present CounterSketch, a physics-grounded framework that transforms natural-language “What if...?” queries into physically plausible sketch animations. The framework integrates VLM-guided scene grounding, causal event-chain reasoning, constraint-based trajectory generation, and verification. It models physical transitions to update the scene graph, computes continuous object trajectories, and validates them against physical, geometric, and semantic constraints. The verified trajectories are then projected into object-level motion bounding boxes to guide downstream sketch-video synthesis. Experiments on CounterSketchBench (80 questions) demonstrate that physics-grounded trajectories provide stronger motion guidance than direct VLM predictions. Project page: https: //tanhiep-to.github.io/CounterSketch/.

Keywords: Counterfactual Reasoning · Sketch-to-Video Generation · Physical Simulation · Physics-Grounded Generation

## 1 Introduction

Sketch-to-video generation uses sketches to define scene structure and text to specify motion. Recent methods have explored scene decomposition, objectaware motion planning, and trajectory optimization to improve visual coherence [12,14,23]. However, they largely overlook how physical relations, including support, contact, and collision, evolve during animation [16,15,2]. Moreover, treating scene conditions as fixed limits their ability to reason about changes and subsequent physical consequences.

A complementary direction models physical phenomena through causal event chains. CoECT [17] introduces Physics-driven Event Chain Reasoning (PECR), representing physical processes as sequential events with evolving scene graphs and physics-based constraints. While this formulation captures causal state transitions, it does not address how hypothetical interventions modify scene relations and propagate into object motion.

![](images/afb04df5e07e2a438c99bf6502bf56a429fde76a526bf6022f49e279335aaa7c.jpg)  
Fig. 1: CounterSketch interprets a counterfactual intervention, reasons about its physical consequences, and generates a sketch video whose motion follows the resulting trajectory.

Building on this perspective, we introduce CounterSketch for counterfactual sketch-to-video generation. As shown in Fig. 1, given a sketch and a naturallanguage “What if...?” query, the system must infer how a hypothetical intervention changes physical relations and triggers subsequent motion. Unlike conventional motion captions, such queries modify the conditions governing object dynamics, requiring causal reasoning to generate physically plausible animations.

The central challenge lies in propagating intervention efects across interacting objects [4,3,5]. For example, removing a table breaks its support relation and causes a ball to fall, while removing a wall eliminates a collision constraint. Similarly, removing the bottom box of a stack may destabilize other objects. These examples, illustrated in Fig. 2, emphasize the need to track evolving physical relations and generate trajectories consistent with counterfactual constraints.

Our contributions are as follows:

– A formulation of counterfactual sketch-to-video generation that transforms a static sketch and a natural-language “What if...?” prompt into a physically plausible sketch video.

A physics-driven counterfactual event-chain reasoning framework that interprets hypothetical interventions, propagates their consequences through evolving scene relations, and identifies the physical events governing the resulting motion.

– A unified constrained trajectory framework that models support, contact, collision, gravity, rolling, and related interactions, with trajectory-level verification, bounded repair, and local semantic replanning to enforce geometric and physical consistency before visual synthesis.

– A trajectory-conditioning mechanism that projects verified object dynamics into temporally aligned image-space motion constraints, providing explicit spatial guidance for downstream sketch-video synthesis.

– CounterSketchBench, comprising 80 counterfactual questions across 20 base scenes to evaluate counterfactual sketch animation across visual and physical dimensions.

![](images/5f6271badfd182f33e68067474ee1d7037614003ad727ae3e929da518f4aaea1.jpg)  
Fig. 2: Representative six types of counterfactual scenes in CounterSketchBench.

## 2 Related Work

Sketch-to-Video and Sketch Animation. Recent methods generate temporally coherent animations from static sketches [13,18]. FlipSketch [1] adapts video difusion models, while Live-Sketch [6] preserves sketch identity through stroke-space transformations. GroupSketch [12] supports multi-object animation using keyframes, and MoSketch [14] combines LLM-based motion planning with trajectory optimization. However, these approaches largely rely on predefined motion descriptions, without reasoning about how hypothetical interventions alter physical relations and subsequent motion [19].

Event-Centric Physical Video Generation. Recent approaches model physical processes through causal event chains [16,17]. CoECT [17] introduces Physics-driven Event Chain Reasoning (PECR), representing physical transitions through evolving scene graphs and physics-based conditions. Building on this formulation, CounterSketch infers event chains from hypothetical interventions and converts their physical consequences into constrained trajectories for sketch animation [19,11].

Constraint-Based Motion Planning. Existing methods optimize trajectories under contact and physical constraints, including contact-implicit control [9], joint contact-trajectory optimization [21], and sequential convex optimization [10]. Unlike these methods, CounterSketch must infer how interventions modify the constraint set before trajectory planning. Although counterfactual physical reasoning has been explored for outcome prediction and world-model evaluation [3,4,5], its application to sketch animation remains unexplored. As summarized in Table 1, CounterSketch combines intervention-driven constraint reasoning with MoSketch-based trajectory optimization [14].

Table 1: Comparison of representative methods for sketch animation, physicsaware generation, and constraint-based planning. ✓ indicates support and × indicates no support. Optimization denotes per-sketch iterative optimization.
<table><tr><td>Method</td><td>Counterfactual Reasoning</td><td>Planning Basis</td><td>Motion Optimization</td></tr><tr><td colspan="4">Sketch animation</td></tr><tr><td>FlipSketch [1]</td><td>X</td><td>T2V prior</td><td>X</td></tr><tr><td>Live-Sketch [6]</td><td>X</td><td>T2V prior</td><td>√</td></tr><tr><td>GroupSketch [12]</td><td>X</td><td>User keyframes</td><td>√</td></tr><tr><td>MoSketch [14]</td><td>X</td><td>LLM</td><td>√</td></tr><tr><td colspan="4">Physics-aware generation</td></tr><tr><td>PhysAnimator [19]</td><td>×</td><td>Physics simulation</td><td>×</td></tr><tr><td>CoECT [17]</td><td>X</td><td>LLM event chain</td><td>X</td></tr><tr><td colspan="4">Constraint-based planning</td></tr><tr><td>Contact-implicit [10,21]</td><td>×</td><td>Fixed constraints</td><td></td></tr><tr><td>CounterSketch</td><td>√</td><td>Physics-aware</td><td></td></tr></table>

## 3 Problem Formulation

Let S be a sketch image and $p _ { c f }$ a counterfactual prompt, the goal is to generate a physically plausible sketch video $V _ { c f } \colon$

$$
( S , p _ { c f } )  V _ { c f } .\tag{1}
$$

We represent S as a scene graph $G _ { S } = ( O , R )$ , comprising objects O and physical relations R [20]. The prompt defines an intervention $\varDelta ,$ preservation constraints $P ,$ and afected objects A [3]. Applying $\varDelta$ yields a counterfactual scene graph $G _ { c f }$ and a sequence of physical events ${ e _ { t } } _ { t = 1 } ^ { T }$ , which guide the generation of constrained object trajectories τ.

$$
G _ { c f } = F ( G _ { S } , \varDelta , P ) , \qquad \lbrace e _ { t } \rbrace _ { t = 1 } ^ { T } = \operatorname { R e a s o n } ( G _ { c f } , A ) ,\tag{2}
$$

$$
\tau = \operatorname { R o l l o u t } ( G _ { c f } , \{ e _ { t } \} _ { t = 1 } ^ { T } ) , \qquad V _ { c f } = \operatorname { R e n d e r } ( S , \tau ) .\tag{3}
$$

## 4 Proposed Method

CounterSketch consists of five stages: (1) sketch scene grounding, (2) physicsdriven counterfactual event-chain reasoning, (3) constraint compilation and physicsbased rollout, (4) trajectory verification and closed-loop repair, and (5) trajectory conditioning for video synthesis. The first four stages operate on an explicit physical representation of the scene. The final stage translates the verified trajectories and event states into conditions for downstream sketch-video generation.

![](images/2dd8c123291448f9a6793b58daadcac66d747b44660147e3ca6c859717eeaef0.jpg)  
Fig. 3: CounterSketch overview. The physical reasoning layer is separated from the downstream sketch-video generator.

## 4.1 Sketch Scene Grounding

CounterSketch begins by recovering a structured representation of the sketched scene. A vision-language model (VLM) identifies candidate objects, estimates their spatial extent, and infers qualitative relations such as support, contact, attachment, containment, and relative position. These semantic predictions are grounded in the underlying vector strokes to associate each stroke with its corresponding object. Strokes with uncertain object membership are assigned using geometric cues, spatial proximity, or connected components. Each grounded object is then represented by its polygonal or contour-based geometry, centroid, bounds, and motion-relevant physical properties.

Conditioned on these grounded entities and their spatial configurations, we formalize the parsed environment into a scene graph:

$$
G _ { S } = ( O , R ) ,\tag{4}
$$

where R encodes relations such as support, contact, attachment, relative position, containment, and potential collision. This explicit representation allows counterfactual interventions to modify object relations and physical constraints rather than operating merely on visual appearance. This grounded scene graph provides the structured state on which counterfactual interventions are subsequently applied.

## 4.2 Physics-Driven Counterfactual Event-Chain Reasoning

As illustrated in Fig. 4, CounterSketch translates a “What $\operatorname { i f } \dots \ ? ^ { 9 }$ intervention into a causal sequence of physical events:

$$
E = \{ e _ { 1 } , e _ { 2 } , \ldots , e _ { T } \} , \qquad e _ { t } = ( A _ { t } , a _ { t } , C _ { t } , G _ { t } ) ,\tag{5}
$$

where $A _ { t }$ denotes afected objects, $a _ { t }$ the event type, $C _ { t }$ the physical conditions, and $G _ { t }$ the updated scene graph.

![](images/b4d949a5af93c06b1ad39a5c50ccde279d8a3e91e4eb63e707c647b675424799.jpg)  
Fig. 4: Physics-driven counterfactual event-chain reasoning. The intervention is grounded in the sketch scene graph and propagated through physically meaningful event states.

Vector Intervention The intervention parser extracts $\varDelta .$ , specifying changes to objects, geometry, or relations. These modifications are applied to vector strokes, and the scene graph is updated to reflect immediate efects, such as lost support, new contact, or removed collision boundaries.

Counterfactual Reasoning From the modified scene graph, CounterSketch infers cascading physical events involving gravity, contact, collision, rolling, inclined motion, and pulley constraints. Each event captures the conditions and state transitions governing subsequent motion, providing the causal basis for constraint compilation and trajectory generation.

## 4.3 Constraint Compilation and Physics-Based Rollout

Figure 5 illustrates constraint compilation, physics-based rollout, and trajectory verification. Each event is converted into geometric and physical constraints over generalized coordinates $q .$ . For an active constraint k,

$$
\phi _ { k } ( q , t ) = 0 , \qquad C ( t ) = \{ \phi _ { k } \mid e _ { k } \mathrm { ~ i s ~ a c t i v e ~ a t } t \} .\tag{6}
$$

These constraints encode support, contact, non-penetration, rolling, and pulley coupling. Given the active constraint Jacobian $J = \partial \phi / \partial q$ , mass matrix $M ,$ and external force $f ,$ the constrained acceleration a is computed as

$$
{ \left[ \begin{array} { l l } { M \ J ^ { \top } } \\ { J } & { 0 } \end{array} \right] } { \left[ \begin{array} { l } { a } \\ { \lambda } \end{array} \right] } = { \left[ \begin{array} { l l } { M a _ { \mathrm { f r e e } } } \\ { \qquad b } \end{array} \right] } , \qquad a _ { \mathrm { f r e e } } = M ^ { - 1 } f\tag{7}
$$

, where λ denotes the constraint forces and b accounts for the time-dependent constraint terms. Contact transitions are detected from the signed geometric gap $d _ { i j } ( t )$ , updating the active constraint set as the event chain evolves. The resulting rollout produces continuous trajectories $\boldsymbol { \tau } = \{ ( q _ { t } , \dot { q } _ { t } ) \}$ for subsequent verification and visual synthesis.

![](images/648fc0529a10c51174f11f9a579d4fbbfdbbad91005b107a75efa9680c78d747.jpg)  
Fig. 5: Constraint compilation, unified dynamics, hybrid rollout, and trajectory verification.

## 4.4 Trajectory Verification and Closed-Loop Repair

CounterSketch verifies generated trajectories against geometric, physical, and semantic constraints. Numerical violations trigger parameter or constraint adjustments, while semantic violations require local replanning. Updates are accepted only if they reduce violations. The verifier also ensures counterfactual compliance: interventions must occur, afected objects must respond accordingly, and unafected objects must remain unchanged unless causally influenced.

## 4.5 Trajectory Conditioning for Video Synthesis

Verified trajectories are projected into per-frame image-space bounding boxes to guide video synthesis. Static objects retain their original positions, while dynamic objects follow simulated paths. These spatial controls translate physical rollouts into explicit motion guidance consistent with the inferred counterfactual event chain.

## 5 Experiments

## 5.1 CounterSketchBench

We introduce CounterSketchBench to evaluate the visual quality and physical plausibility of counterfactual sketch animation. It contains 80 “What if...?” questions across 20 vector-sketch scenes, covering geometric, relational, and kinematic interventions. We assess CounterSketch through qualitative comparisons and an ablation study contrasting physics-grounded trajectory planning with direct VLM prediction.

## 5.2 Ablation Study

Experimental Setup. Because CounterSketch utilizes MoSketch as its animation generation backbone, we investigate whether our physics-grounded reasoning and constrained rollout provide more efective motion guidance than direct trajectory inference by a vision-language model. We evaluate three progressive configurations while keeping the downstream MoSketch renderer fixed:

Table 2: Counterfactual intervention categories in CounterSketchBench.
<table><tr><td>Category</td><td>Intervention Example question</td></tr><tr><td>Object removal</td><td>Remove one or multiple ob- What if the table beneath the 19 jects or supporting structures ball is removed?</td></tr><tr><td>Position</td><td>Translate, swap, reorder, or What if the right ball is placed 21 rearrange objects to the left of the left ball?</td></tr><tr><td>Orientation</td><td>Rotate an object, surface, or What if the ground is tilted by 10 the entire scene 15°?</td></tr><tr><td>Size / geometry</td><td>Scale an object or modify its What if the pendulum string 15 geometric extent is 50% longer?</td></tr><tr><td></td><td>Relation / constraint Modify support, contact, at- What if the ball detaches from 13 tachment, or relative relations the string?</td></tr><tr><td>Dynamics</td><td>Modify an initial dynamic What if the ball starts with a 2 condition very large initial velocity?</td></tr><tr><td>Total</td><td>80</td></tr></table>

Table 3: Ablation of motion guidance for downstream synthesis using MoSketch. All variants share the identical renderer.
<table><tr><td>Scene Trajectory Modification Generation</td><td>T2V</td><td>S2V</td><td>Motion Alignment ↑ Alignment ↑ Smoothness ↑ Degree ↑</td><td>Dynamic</td></tr><tr><td>Direct VLM Direct VLM</td><td>0.1682</td><td>0.9415</td><td>0.9830</td><td>0.5625</td></tr><tr><td>CounterSketch Direct VLM</td><td>0.1729</td><td>0.9438</td><td>0.9825</td><td>0.7250</td></tr><tr><td>CounterSketch CounterSketch</td><td>0.2011</td><td>0.9580</td><td>0.9834</td><td>0.7750</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

1. Direct VLM + Direct VLM: A VLM-only upstream baseline where the VLM performs both counterfactual scene modification and trajectory prediction before passing motion guidance to MoSketch.

2. CounterSketch + Direct VLM: CounterSketch applies the structured vector intervention to update the scene, after which a VLM directly predicts object bounding-box paths from the modified drawing.

3. CounterSketch + CounterSketch: Our complete framework, where the intervention updates the scene graph, causal event chains compile into physical constraints, and verified trajectories condition MoSketch.

We adopt Gemini 3.8 Flash model<sup>4</sup> as the backbone for both VLM baselines, supplying the identical counterfactual prompt, detected object labels, frame budget, and bounding-box trajectory formatting employed by CounterSketch.

Evaluation Metrics. Following standard sketch animation benchmarks, we assess video synthesis quality using four criteria derived from VBench [7,8,22]. We instruction fidelity (Text–Video Alignment), sketch preservation (Sketch– Video Alignment), temporal coherence (Motion Smoothness), and motion intensity (Dynamic Degree).

![](images/2bfc0b4e93c23a5439ee558ba60761831abed4f62a54779cf904787597bcc021.jpg)  
Fig. 6: Qualitative results across five representative counterfactual intervention types.

Ablation Results. Table 3 shows that replacing physics-grounded rollout with direct VLM prediction reduces T2V alignment (0.2011 to 0.1729) and Dynamic Degree (0.7750 to 0.7250), with minimal changes in S2V alignment and Motion Smoothness. Removing structured scene modification further reduces these metrics to 0.1682 and 0.5625, respectively, highlighting the importance of structured interventions and physics-grounded trajectories.

## 5.3 Qualitative Results

Figure 6 illustrates counterfactual interventions involving object removal, repositioning, detachment, scene reorientation, and physical-property changes. The generated keyframes reflect their expected physical consequences: removing a table causes a ball to fall, while detaching an object from a string alters its motion. Geometric changes similarly trigger updated contacts and trajectories. These results demonstrate CounterSketch’s ability to propagate interventions into physically coherent, temporally smooth animations.

## 6 Discussion

CounterSketch separates causal reasoning and trajectory planning from visual synthesis. Event-chain reasoning models physical transitions, while constrained rollout generates physically consistent trajectories. Ablation results show that replacing physics-grounded rollout with direct VLM predictions maintains visual quality and temporal smoothness but weakens semantic alignment and motion dynamism. This highlights the importance of explicit physical reasoning in translating counterfactual interventions into accurate motion.

## 7 Limitations

CounterSketch depends on reliable sketch grounding, as ambiguous drawings may produce diferent scene interpretations. Latent physical properties, such as mass, friction, and elasticity, are dificult to infer from sketches alone. Moreover, complex object interactions increase reasoning costs, while practical performance remains dependent on the conditioning capabilities of video generators.

## 8 Conclusion and Future Work

We introduced CounterSketch, a physics-grounded framework that transforms sketches and “What if...?” queries into physically plausible animations through causal reasoning and verified object trajectories. By separating physical reasoning from visual synthesis, CounterSketch provides interpretable and controllable motion guidance. Future work will explore inferring latent physical properties from multimodal priors and extending the framework to non-rigid motion.

Acknowledgments. This research is funded by Vietnam National University - Ho Chi Minh City (VNU-HCM) under Grant Number B2026-18-17.

## References

1. Bandyopadhyay, H., Song, Y.Z.: Flipsketch: Flipping static drawings to text-guided sketch animations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2025)

2. Bansal, H., Peng, C., Bitton, Y., Goldenberg, R., Grover, A., Chang, K.W.: Videophy-2: A challenging action-centric physical commonsense evaluation in video generation. In: International Conference on Learning Representations (2026)

3. Baradel, F., Neverova, N., Mille, J., Mori, G., Wolf, C.: Cophy: Counterfactual learning of physical dynamics. arXiv preprint arXiv:1909.12000 (2019)

4. Begiristain, L., Dünkel, O., Kortylewski, A.: Cronos: Benchmarking counterfactual physical consistency in video models. arXiv preprint arXiv:2605.23699 (2026)

5. Cai, K., Song, R., Zhang, J., Zhang, K., Bodapati, P., Yu, A., Suya, F., Rostami, M., Ma, J., Tian, Y.: What-if world: A causal benchmark for general world models in embodied scenarios. arXiv preprint arXiv:2605.27589 (2026)

6. Gal, R., et al.: Breathing life into sketches using text-to-video priors. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

7. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024)

8. Huang, Z., Zhang, F., Xu, X., He, Y., Yu, J., Dong, Z., Ma, Q., Chanpaisit, N., Si, C., Jiang, Y., Wang, Y., Chen, X., Chen, Y.C., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

9. Le Cleac’h, S., Howell, T.A., Yang, S., Lee, C.Y., Zhang, J., Bishop, A., Schwager, M., Manchester, Z.: Fast contact-implicit model predictive control. IEEE Transactions on Robotics 40, 1617–1629 (2024)

10. Li, Y., Han, H., Kang, S., Ma, J., Yang, H.: On the surprising robustness of sequential convex optimization for contact-implicit motion planning. arXiv preprint arXiv:2502.01055 (2025)

11. Li, Z., Yu, H.X., Liu, W., Yang, Y., Herrmann, C., Wetzstein, G., Wu, J.: Wonderplay: Dynamic 3d scene generation from a single image and actions. In: 2025 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9080–9090. IEEE (2025)

12. Liang, G., et al.: Multi-object sketch animation with grouping and motion trajectory priors. In: Proceedings of the ACM International Conference on Multimedia (2025)

13. Liu, F.L., Fu, H., Wang, X., Ye, W., Wan, P., Zhang, D., Gao, L.: Sketchvideo: Sketch-based video generation and editing. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 23379–23390. IEEE (2025)

14. Liu, J., et al.: Multi-object sketch animation by scene decomposition and motion planning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2025)

15. Meng, F., Liao, J., Tan, X., Shao, W., Lu, Q., Zhang, K., Cheng, Y., Li, D., Qiao, Y., Luo, P.: Towards world simulator: Crafting physical commonsense-based benchmark for video generation. arXiv preprint arXiv:2410.05363 (2024)

16. Satish, S.N.K., Jaiswal, D., Chen, H., Bakshi, A.: Physvideogenerator: Towards physically aware video generation via latent physics guidance. arXiv preprint arXiv:2601.03665 (2026)

17. Wang, Z., Hu, Y., Wang, H., Chen, F., Liu, Y., Li, W., Lei, Y.: Chain of eventcentric causal thought for physically plausible video generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 38122–38131 (2026)

18. Wu, R., Su, W., Ma, K., Liao, J.: Aniclipart: Clipart animation with text-to-video priors. International Journal of Computer Vision 133(6), 3149–3165 (2025)

19. Xie, T., Zhao, Y., Jiang, Y., Jiang, C.: Physanimator: Physics-guided generative cartoon animation. In: 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10793–10804. IEEE (2025)

20. Xu, K., Chen, K., Fu, H., Sun, W.L., Hu, S.M.: Sketch2scene: Sketch-based coretrieval and co-placement of 3d models. ACM Transactions on Graphics (TOG) 32(4), 1–15 (2013)

21. Zhang, M., Jha, D.K., Raghunathan, A.U., Hauser, K.: Simultaneous trajectory optimization and contact selection for contact-rich manipulation with high-fidelity geometry. IEEE Transactions on Robotics 41, 2677–2690 (2025)

22. Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Zhang, Y., He, J., Zheng, W.S., Qiao, Y., Liu, Z.: VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755 (2025)

23. Zhu, X., Yang, X., Zheng, S., Zhang, Z., Gao, F., Huang, J., Chen, J.: Vector sketch animation generation with diferentiable motion trajectories. In: Computer Graphics Forum. p. e70335. Wiley Online Library (2026)