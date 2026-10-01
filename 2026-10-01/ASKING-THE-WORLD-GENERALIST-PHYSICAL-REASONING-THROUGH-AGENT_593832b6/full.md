# ASKING THE WORLD: GENERALIST PHYSICAL REASONING THROUGH AGENTIC WORLD MODELING AND PROBING

Shenxiang Zeng<sup>1∗</sup>, Chen Yang<sup>1∗</sup>, Peiyao Chen<sup>1</sup>, Guohui Zhang<sup>1</sup>, Jiansheng Fan<sup>1†</sup>, Chen Wang<sup>1†</sup> <sup>1</sup>Tsinghua University

![](images/778d9d117dec9da398d3c23f24c0c94caf47c154def9462556a3e67b5e60abaf.jpg)  
Figure 1: Physical reasoning through world modeling and probing. Given videos and questions, ATW builds and revises executable worlds, probes them with PolyWorld Engine, and reasons from execution evidence. Examples span rigid bodies, fluids, and cloth.

## ABSTRACT

Physical reasoning from video requires inferring latent physical properties and dynamics beyond direct observation. Direct VLM inference remains unreliable on complex physical tasks without explicit modeling and validation, while predefined tool pipelines rely on task- and domain-specific priors that limit generalization across materials, dynamics, and reasoning tasks. We introduce Asking the World (ATW), a generalist agent that constructs and interrogates task-relevant executable worlds through two adaptive stages: World Modeling calibrates a world from video, while World Probing queries, simulates, and intervenes on it to obtain question-relevant evidence. Rather than prescribing the operations in either stage, ATW determines how to model and probe according to the scene and question. We develop PolyWorld Engine, a lightweight and highly programmable Warp-based multiphysics simulator for constructing and probing worlds with rigid bodies, soft bodies, cloth, ropes, fluids, and their coupled interactions. CEM-based system identification recovers task-relevant dynamics during World Modeling. The resulting world becomes an active workspace for question-directed physical experiments rather than a predetermined downstream tool. We evaluate ATW on CLEVRER, ContPhy, and three real-world scenarios. Using Gemini-3-Flash as its base VLM, ATW achieves 80.82% overall per-question accuracy on CLEVRER, improving direct Gemini-3-Flash by 46.50 points, GPT-5.5 by 13.58 points, and PhysMind by 8.27 points. On ContPhy, it reaches 70.56% overall accuracy, surpassing Gemini-3-Flash by 28.10 points and GPT-5.5 by 3.53 points. Across the three real-world scenarios, ATW achieves 71.67% accuracy, 28.33 points above GPT-5.5. These results establish agentic world modeling and probing as an effective, execution-grounded approach to generalist physical reasoning.

## 1 INTRODUCTION

Physical questions about latent properties, future outcomes, and counterfactual events often require evidence beyond the observed video. A capable system must therefore construct a task-relevant account of scene dynamics and obtain physical evidence from it. Doing so across different materials, dynamics, and question types is the central ambition of generalist physical reasoning, from rigidbody collisions in CLEVRER to continuum phenomena in ContPhy (Yi et al., 2019; Zheng et al., 2024).

Existing approaches largely follow two paradigms. VLM-based methods exploit broad visual and semantic priors, either through direct inference or task-specific adaptation (Chow et al., 2025; Wang et al., 2025; Feng et al., 2025). Yet recent evaluations continue to expose failures in physical consistency and causal evolution across multimodal and generative video models (Li et al., 2025c; Bansal et al., 2024; Meng et al., 2024; Li et al., 2025a). Physics-grounded methods instead recover dynamics, call specialized tools, or execute simulators (Ding et al., 2021; Li et al., 2023; Xie et al., 2024; Zhang et al., 2024; Fan et al., 2025; Cherian et al., 2026; Yang et al., 2026). Although they provide explicit physical evidence, they usually commit in advance to a particular representation, physical model, or inference procedure, limiting adaptation across materials and questions. What remains missing is a general agent that can construct an appropriate executable world for the scene and determine how to interrogate it for the question at hand.

Recent agents show that tools, generated programs, and execution can support iterative reasoning rather than only terminal prediction (Yao et al., 2023; Gupta & Kembhavi, 2023; Sur´ıs et al., 2023; Liang et al., 2023; Yin et al., 2026). Physical question answering, however, has no fully observed target scene against which every update can be compared; the agent must decide for itself what hidden mechanisms and interventions matter. We therefore introduce Asking the World (ATW), a generalist physical reasoning agent that organizes its reasoning into two stages. World Modeling constructs a task-relevant executable world by identifying relevant entities, selecting physical representations, and recovering latent dynamics. World Probing queries, simulates, and intervenes on that world to gather evidence for the associated questions. Rather than prescribing the operations in either stage, ATW determines how to model and probe according to the scene and question. In a nutshell, ATW asks the world by building and probing it (Figure 1). This process is enabled by PolyWorld Engine, our lightweight and highly programmable multiphysics simulator built on Warp (Macklin, 2022). It provides a unified environment for rigid bodies, soft bodies, cloth, ropes, fluids, and their coupled interactions, which ATW can instantiate and modify on demand. CEM-based system identification supports World Modeling by recovering task-relevant parameters from visual evidence. Together, these capabilities turn simulation from a predetermined downstream tool into an active workspace for constructing and interrogating physical explanations.

Without task-specific training, ATW applies the same agentic formulation across rigid-body, continuum, and real-world settings. It achieves 80.82% overall per-question accuracy on CLEVRER, improving Gemini-3-Flash by 46.50 points, GPT-5.5 by 13.58 points, and PhysMind by 8.27 points. On ContPhy, it reaches 70.56%, surpassing Gemini-3-Flash by 28.10 points and GPT-5.5 by 3.53 points; across three real-world scenarios, it achieves 71.67% accuracy, 28.33 points above GPT-5.5. These results establish agentic world modeling and probing, supported by a general multiphysics engine, as an effective path toward generalist physical reasoning.

In summary, our contributions are threefold:

• Agentic physical reasoning. We introduce ATW, a generalist physical reasoning agent that organizes reasoning into World Modeling, which constructs task-relevant executable worlds, and World Probing, which conducts question-directed queries, simulations, and interventions.

• Generalist multiphysics execution. We develop PolyWorld Engine, a lightweight and highly programmable environment that unifies rigid bodies, soft bodies, cloth, ropes, fluids, and their coupled interactions for on-demand world construction, intervention, and validation.

• Broad empirical validation. We demonstrate substantial improvements over direct VLM reasoning and strong baselines across CLEVRER, ContPhy, and three real-world scenarios, spanning multiple materials, dynamical systems, and reasoning tasks.

![](images/f25d86eda19390ffe9bd3bb8a077ebdf299f70f654cc914dca5ff768fbcd55e2.jpg)  
Figure 2: ATW system overview. A billiards counterfactual illustrates the one-way transition from World Modeling to World Probing. The modeling agent corrects a missing object mask, tracks object poses, and fits dynamics before committing the executable world. The probing agent then removes the brown ball, queries collision events, and answers from the resulting evidence.

## 2 RELATED WORK

Visual Physical Reasoning. Object-centric models learn structured representations and dynamics (Chen et al., 2021; Ding et al., 2021; Wu et al., 2023; Li et al., 2025b). VLM methods use targeted training, memories, perception modules, spatiotemporal tools, or simulator search (Balazadeh et al., 2025; Wang et al., 2025; Feng et al., 2025; Ghazanfari et al., 2025; Chow et al., 2025; Fan et al., 2024; 2025; Cherian et al., 2026), while complementary benchmarks assess physical consistency and world-model behavior in generated videos (Bansal et al., 2024; Meng et al., 2024; Li et al., 2025a). Their physics nevertheless remains implicit or procedurally fixed; ATW instead lets execution feedback select what to model and test.

Executable Physical Worlds. Differentiable simulators and video-conditioned models optimize parameters within specified dynamics (Howell et al., 2022; Ding et al., 2021; Li et al., 2023), while black-box approaches search simulator parameters from trajectory error (Cherian et al., 2026). Video-to-simulation methods reconstruct geometry and physical properties (Xie et al., 2024; Zhang et al., 2024; Chen et al., 2025a; Zhao et al., 2025), and executable-world methods use reconstructed dynamics for question answering (Yang et al., 2026). These systems assume a fixed representation, material family, or objective; ATW instead builds task-sufficient worlds with heterogeneous materials and coupled interactions.

Agents Reasoning through Engines. Language and vision agents interleave reasoning with actions, tools, or generated programs (Yao et al., 2023; Gupta & Kembhavi, 2023; Sur´ıs et al., 2023; Liang et al., 2023). Agentic reconstruction combines visual models, code generation, execution, and feedback to construct explicit scenes (Yao et al., 2025; Yin et al., 2026); related agents use 3D programs for spatial reasoning (Luo et al., 2026; Chen et al., 2025c) or interactive environments for manipulation planning (Liu et al., 2025; Xu et al., 2026). These agents target observable scenes or prespecified goals, whereas physical questions may concern hidden, future, or counterfactual dynamics. ATW therefore makes world construction question-conditioned and chooses which physica hypotheses to instantiate and probe.

## 3 METHOD

ATW turns physical reasoning into interaction with an executable world. Given a video and its associated questions, the agent first constructs a task-sufficient physical world and then uses that world to conduct question-directed experiments. These are two consecutive stages: World Modeling revises perceptual and physical assumptions until it commits a world, after which World Probing adapts experiments within that world. Figure 2 illustrates this progression. We first describe the two-stage agentic framework, then present the PolyWorld Engine that makes executable world programmable by the agent, and finally explain how video evidence grounds and calibrates those worlds.

![](images/0ce913870a30974daeb2b1afe7b1f1efdc83bf1a7755273a1f1abc7d007390f2.jpg)  
Figure 3: Stage-local feedback and correction. Within World Modeling, observed deformation causes the agent to replace a rigid-body hypothesis with a soft-body model. Within World Probing, execution feedback reveals an incorrect intervention target and causes the agent to probe the intended cyan wall. Feedback therefore changes the agent’s decisions within each stage.

## 3.1 ATW: TWO-STAGE AGENTIC PHYSICAL REASONING

Let V denote an input video and $\mathcal { Q } = \{ q _ { i } \} _ { i = 1 } ^ { N }$ its physical questions. Rather than reconstructing every visible detail, ATW seeks an executable world $\dot { \mathcal { W } } ^ { \star }$ containing the entities, physical representations, states, and latent parameters needed by these questions. We denote the stage-specific agent policies by $\mathcal { A } _ { \mathrm { M } }$ and $A _ { \mathrm { P } }$ . The former constructs the world, while the latter obtains evidence $\mathcal { E } _ { i }$ for each question and produces the answer:

$$
\begin{array} { r } { \mathcal { W } ^ { \star } = \mathcal { A } _ { \mathrm { M } } ( \mathcal { V } , \mathcal { Q } ) , \qquad \mathcal { E } _ { i } = \mathcal { A } _ { \mathrm { P } } ( \mathcal { W } ^ { \star } , q _ { i } ) , \qquad \hat { a } _ { i } = \mathrm { A n s w e r } ( q _ { i } , \mathcal { E } _ { i } ) . } \end{array}\tag{1}
$$

Both $\mathcal { A } _ { \mathrm { M } }$ and $A _ { \mathrm { P } }$ are adaptive agents rather than fixed workflows: within its stage, each agent chooses its next operation from the current state, prior tool results, and execution feedback.

World Modeling. The modeling agent decides what must be represented before deciding how to recover it. It selects question-relevant entities, invokes visual tools to establish their geometry and motion, assigns a physical representation to each entity, and identifies parameters whose values must be inferred. The agent can inspect intermediate masks, tracks, reconstructions, and simulated trajectories; a mismatch can trigger another observation, a corrected object binding, a different physical representation, or renewed parameter fitting. This feedback remains internal to World Modeling. The stage terminates by committing $\mathcal { W } ^ { \star }$ , including object identities, geometry, initial states, physical models, and fitted parameters.

World Probing. The probing agent treats $\mathcal { W } ^ { \star }$ as a fixed base world and decides which physical experiment will resolve each question. It can read an inferred property, continue the unmodified dynamics, instantiate an intervened branch, inspect events or trajectories, and compare outcomes across branches. Thus property questions can query the fitted world directly, predictive questions can extend its trajectory, and counterfactual or goal-driven questions can test edited worlds. Execution feedback may change the next probe or correct its intervention target, but it does not reopen World Modeling or alter the committed base world. Probing ends when the accumulated evidence supports an answer or the interaction budget is exhausted.

## 3.2 POLYWORLD ENGINE: AGENT-OPERABLE MULTIPHYSICS WORLDS

PolyWorld Engine is our lightweight and highly programmable multiphysics simulator built on Warp (Macklin, 2022). It places rigid bodies, soft bodies, cloth, ropes, and fluids in a common executable scene and supports their coupled interactions. Each physical system retains the state variables and dynamics appropriate to its material, while the shared scene allows the agent to compose heterogeneous systems when required by the observed interaction. Reusable scene data, simulation buffers, and explicit parameterization make repeated fitting and branched rollouts practical within an agent trajectory.

Table 1: PolyWorld as an agent-facing semantic interface. High-level operations connect agent decisions to physical execution while preserving explicit world and evidence provenance. Concrete tool signatures and backend-specific details are provided in Appendix A.
<table><tr><td>Interface role</td><td>Representative capabilities</td><td>Returned state or evidence</td></tr><tr><td>World construction</td><td>Bind objects, assign physical models, initialize states, fit parameters</td><td>Identities, geometry, model assignments, fitted dynamics</td></tr><tr><td>World inspection</td><td>Inspect frames or rollouts, read properties, query states</td><td>Visual observations, properties, trajectories, events</td></tr><tr><td>World</td><td>Continue dynamics, intervene, branch</td><td>Predictive and counterfactual outcomes</td></tr><tr><td>experimentation Execution control</td><td>rollouts, compare worlds</td><td>with temporal scope Tool constraints, execution status,</td></tr><tr><td></td><td>Describe operations, revisit evidence recover from failure, terminate</td><td>retained evidence</td></tr></table>

PolyWorld exposes physics through semantic operations rather than requiring the agent to manipulate low-level simulator code. Table 1 summarizes the interface used across the two stages. The available operations are conditioned on the instantiated physical systems, and every result records its source world, objects, temporal scope, and execution status. This provenance lets the agent revisit earlier evidence without rerunning an experiment or confusing observations from different branches.

The committed world is reusable but not destructively edited during probing. Predictive rollouts continue the base world, whereas interventions create branches that inherit its fitted parameters and initial conditions. Multiple questions and candidate interventions can therefore share the same calibrated world, while their trajectories and events remain separately addressable. This design turns simulation into a persistent workspace for physical inquiry rather than a terminal tool invocation.

## 3.3 GROUNDING AND CALIBRATING EXECUTABLE WORLDS

World Modeling converts video evidence into a world that is physically executable and sufficient for the questions at hand. The agent obtains object masks, motion tracks, and scene geometry from visual tools, and retains explicit bindings between referenced entities and objects in the executable scene. These observations play complementary roles: masks delimit objects and their visible deformation, tracks constrain temporal motion, and geometry establishes spatial configuration and initial conditions. The agent requests or refines only the evidence required by the selected representation, avoiding a mandatory reconstruction pipeline shared by all scenes.

For each instantiated physical representation h, PolyWorld maps an initial state $s _ { 0 }$ and parameters θ to a simulated trajectory, from which a model-specific readout is compared with visual evidence. Rigid systems use tracked positions and orientations; deformable systems additionally use shape, surface, centerline, or occupancy observations as appropriate. We fit the exposed parameters with the cross-entropy method (CEM). At iteration $k ,$ CEM evaluates bounded parameter samples under the corresponding trajectory discrepancy and uses the lowest-loss elite set to update its diagonal Gaussian proposal:

$$
\begin{array} { r l } & { \pmb { \mu } _ { k + 1 } = \alpha \pmb { \mu } _ { k } + ( 1 - \alpha ) \widehat { \pmb { \mu } } _ { k } , } \\ & { \pmb { \sigma } _ { k + 1 } = \operatorname* { m a x } ( \pmb { \sigma } _ { \operatorname* { m i n } } , \alpha \pmb { \sigma } _ { k } + ( 1 - \alpha ) \widehat { \pmb { \sigma } } _ { k } ) , } \end{array}\tag{2}
$$

where $\widehat { \mu } _ { k }$ and $\widehat { \pmb { \sigma } } _ { k }$ are the elite statistics, α controls smoothing, and $\sigma _ { \mathrm { m i n } }$ preserves exploration. Fitting progressively incorporates longer observation windows so that later interactions constrain the recovered dynamics (system-identification details: Appendix A.2).

Calibration also tests whether the current world hypothesis is adequate. Persistent motion or shape discrepancies can indicate that the error lies in the representation or object binding rather than its numerical parameters. The modeling agent can therefore revise those choices and repeat fitting before committing $\mathcal { W } ^ { \star }$ as the shared physical basis for subsequent experiments.

## 4 EXPERIMENTS

We evaluate ATW on CLEVRER (Yi et al., 2019), ContPhy (Zheng et al., 2024), and real-world videos against foundation VLMs and physical-reasoning methods. We also present qualitative results and ablations of adaptive reasoning, physical simulation, and system identification.

## 4.1 EXPERIMENTAL PROTOCOL

Benchmarks and metrics. CLEVRER (Yi et al., 2019) tests explanatory, predictive, and counterfactual rigid-body reasoning. Its validation subset contains 1,000 videos, 4,280 questions, and 14,228 options; we report per-option and per-question accuracy, with the latter requiring every option to be correct. ContPhy (Zheng et al., 2024) covers physical-property, predictive, counterfactual, and goal-driven reasoning. On its 600 videos and 1,950 questions, we report category accuracy and a question-count-weighted overall score. We additionally report category-wise accuracy on three real-world scenarios totaling 60 videos and questions (evaluation details: Appendix B).

Baselines and controlled variants. Baselines comprise foundation VLMs—Qwen3-VL-235B-A22B (Bai et al., 2025), GLM-4.6V (Z.ai, 2025), Gemini-3-Flash (Google DeepMind, 2025), Gemini-3.1-Pro (Google DeepMind, 2026), GPT-4o (OpenAI, 2024), and GPT-5.5 (OpenAI, 2026); training-based video reasoners—VideoRFT (Wang et al., 2025), Video-R1 (Feng et al., 2025), VideoThinker-R1 (Wu et al., 2026), and Chain-of-Frames (Ghazanfari et al., 2025); and trainingfree methods—VideoAgent (Fan et al., 2024), STAR (Fan et al., 2025), and PhysMind (Yang et al., 2026). Controlled variants isolate structured perception, physical execution, and adaptive reasoning.

Implementation details. ATW uses Gemini-3-Flash and PolyWorld Engine, without training or fine-tuning on either benchmark. Gemini baselines receive the video; GPT, Qwen, and GLM receive eight uniformly sampled frames (implementation details: Appendix A.1; prompts: Appendix D).

## 4.2 MAIN RESULTS

CLEVRER. Table 2 shows that ATW achieves 80.82% overall per-question accuracy, outperforming Gemini-3-Flash by 46.50 percentage points, GPT-5.5 by 13.58 points, and PhysMind, the strongest training-free baseline, by 8.27 points. ATW leads all three categories with 83.87% explanatory, 88.60% predictive, and 75.16% counterfactual accuracy. Its largest gain over GPT-5.5 is on counterfactual questions (+23.83 points), where it also exceeds PhysMind by 4.58 points.

ContPhy. Table 3 evaluates continuous-media and diverse-material tasks. ATW achieves 70.56% overall accuracy, 28.10 percentage points above Gemini-3-Flash and 3.53 points above GPT-5.5; it also outperforms Gemini-3-Flash in each of the four categories. Its gains over GPT-5.5 concentrate in physical-property and predictive reasoning, reaching 81.73% and 62.20% (+6.26 and +11.39 points). It nearly matches GPT-5.5 on goal-driven questions (a 1.93-point gap) but trails it by 6.46 points on counterfactual questions.

Real-world evaluation. Across the three realworld tasks, ATW achieves 85.00%, 70.00%, and 60.00% per-question accuracy on billiards counterfactual, bouncing-ball prediction, and toy-car prediction, respectively, outperforming both GPT-5.5 and Gemini-3-Flash in every category. Its overall accuracy is 71.67%, compared with 43.33% for GPT-5.5 and 33.33% for Gemini-3-Flash. Across these 60 videos, the gains show that executable-world reasoning extends to the three evaluated real-world settings despite appearance variation. The lower predictive-question score also reflects the greater difficulty of forecasting future trajectories in real-world scenes (detailed results: Appendix C.1).

![](images/360978517f530f4448539296550fcf1a5acaf700f43ac4bc27c7a272aab873fe.jpg)  
Figure 4: Real-world per-question accuracy (CF: counterfactual; Pred.: prediction).

Table 2: CLEVRER accuracy (%) on a validation subset of 1,000 videos. Explanatory questions identify causes, while predictive and counterfactual questions concern future and altered outcomes. Each category reports per-question and per-option accuracy; a question is correct only when all associated options are correct. Bold and underlined values indicate the best and second-best results in each column, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="2">Explanatory</td><td colspan="2">Predictive</td><td colspan="2">Counterfactual</td><td colspan="2">Overall</td></tr><tr><td>per ques. per opt.</td><td></td><td>per ques. per opt.</td><td></td><td>per ques.</td><td>per opt.</td><td>per ques. per opt.</td><td></td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>7.19</td><td>49.33</td><td>25.40</td><td>49.71</td><td>9.75</td><td>50.02</td><td>11.26</td><td>49.69</td></tr><tr><td>Blind Gemini-3-Flash</td><td>27.00</td><td>64.64</td><td>24.68</td><td>48.92</td><td>10.07</td><td>50.13</td><td>19.21</td><td>56.31</td></tr><tr><td colspan="9">Foundation VLMs</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>53.19</td><td>75.84</td><td>40.40</td><td>62.12</td><td>22.17</td><td>61.79</td><td>37.52</td><td>67.92</td></tr><tr><td>GLM-4.6V</td><td>39.33</td><td>73.75</td><td>63.64</td><td>66.74</td><td>24.31</td><td>58.70</td><td>36.68</td><td>66.02</td></tr><tr><td>Gemini-3-Flash</td><td>50.32</td><td>74.66</td><td>42.71</td><td>53.46</td><td>16.63</td><td>58.38</td><td>34.32</td><td>64.97</td></tr><tr><td>Gemini-3.1-Pro</td><td>60.90</td><td>79.17</td><td>61.76</td><td>71.28</td><td>25.37</td><td>63.50</td><td>45.47</td><td>71.06</td></tr><tr><td>GPT-40</td><td>36.76</td><td>70.48</td><td>35.64</td><td>54.98</td><td>15.57</td><td>56.54</td><td>27.29</td><td>62.44</td></tr><tr><td>GPT-5.5</td><td>77.21</td><td>89.55</td><td>85.71</td><td>90.69</td><td>51.33</td><td>76.74</td><td>67.24</td><td>83.66</td></tr><tr><td colspan="9">Training-Based Methods</td></tr><tr><td>VideoRFT</td><td>16.66</td><td>61.98</td><td>39.25</td><td>47.40</td><td>15.35</td><td>54.98</td><td>19.74</td><td>57.28</td></tr><tr><td>Video-R1</td><td>32.79</td><td>71.14</td><td>41.56</td><td>46.25</td><td>19.62</td><td>59.10</td><td>28.43</td><td>63.08</td></tr><tr><td>VideoThinker-R1</td><td>17.53</td><td>63.60</td><td>41.56</td><td>42.50</td><td>15.14</td><td>53.71</td><td>20.37</td><td>56.92</td></tr><tr><td>Chain-of-Frames</td><td>44.71</td><td>77.26</td><td>84.42</td><td>87.88</td><td>33.48</td><td>70.84</td><td>46.21</td><td>75.29</td></tr><tr><td colspan="9">Training-Free Methods</td></tr><tr><td>VideoAgent</td><td>34.42</td><td>61.55</td><td>14.29</td><td>48.56</td><td>7.57</td><td>51.61</td><td>19.39</td><td>55.63</td></tr><tr><td>STAR</td><td>55.58</td><td>78.62</td><td>24.24</td><td>57.14</td><td>7.57</td><td>53.47</td><td>29.46</td><td>64.75</td></tr><tr><td>PhysMind</td><td>76.97</td><td>87.38</td><td>66.96</td><td>82.32</td><td>70.58</td><td>88.08</td><td>72.55</td><td>87.22</td></tr><tr><td>ATW (Ours)</td><td>83.87</td><td>92.31</td><td>88.60</td><td>91.41</td><td>75.16</td><td>90.29</td><td>80.82</td><td>91.28</td></tr></table>

Table 3: ContPhy accuracy (%) over 600 videos and 1,950 questions. Property questions test physical attributes, predictive and counterfactual questions test future and intervened outcomes, and goaldriven questions test action selection toward a target state. Bold and underlined values indicate the best and second-best results in each column, respectively.
<table><tr><td>Method</td><td>Property</td><td>Predictive</td><td>Counterfactual</td><td>Goal-driven</td><td>Overall</td></tr><tr><td colspan="6">Baselines</td></tr><tr><td>Random</td><td>40.53</td><td>31.50</td><td>12.92</td><td>13.90</td><td>28.36</td></tr><tr><td>Blind Gemini-3-Flash</td><td>47.47</td><td>41.26</td><td>23.16</td><td>29.34</td><td>37.90</td></tr><tr><td colspan="6">Foundation VLMs</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>58.80</td><td>46.54</td><td>30.51</td><td>27.41</td><td>45.03</td></tr><tr><td>Gemini-3-Flash</td><td>60.00</td><td>43.90</td><td>23.16</td><td>22.39</td><td>42.46</td></tr><tr><td>GPT-5.5</td><td>75.47</td><td>50.81</td><td>69.49</td><td>69.11</td><td>67.03</td></tr><tr><td colspan="6">Training-Based Methods</td></tr><tr><td>Video-R1-7B</td><td>43.47</td><td>34.15</td><td>11.14</td><td>15.44</td><td>29.95</td></tr><tr><td>Chain-of-Frames-8B</td><td>50.27</td><td>48.17</td><td>10.24</td><td>20.85</td><td>36.62</td></tr><tr><td colspan="6">Training-Free Methods</td></tr><tr><td>ATW (Ours)</td><td>81.73</td><td>62.20</td><td>63.03</td><td>67.18</td><td>70.56</td></tr></table>

Figure 5 visualizes the selected modeling steps across three real-world tasks. Segmentation identifies task-relevant objects, pose tracking preserves identities and motion across key frames, and dynamic reconstruction forms executable states. Together, they provide the geometry, trajectories, and interactions needed for targeted prediction and counterfactual probes.

![](images/a198f3d21efae06095a611edfc2e95382e7af209edb60060cc13a2f9e1104726.jpg)

Figure 5: Agentic world modeling from real-world videos of billiards, toy cars, and bouncing balls. Selected key frames show object segmentation, pose tracking, and dynamic reconstruction, producing executable worlds for subsequent probing.  
![](images/790f71e472e8ec70e6885f34d7fc0aa95e7ddf2f88ed0b024f2ad8ffaf5b74cb.jpg)  
Figure 6: Ablation of agentic world reasoning on the CLEVRER and ContPhy subsets. All four variants use Gemini-3-Flash. Bars and labels show per-question accuracy; overall scores aggregate all questions in each subset.

## 4.3 ABLATION STUDIES

We conduct ablation studies on CLEVRER and ContPhy subsets, each containing 100 scenes. CLEVRER includes 435 questions and 1,439 answer options, while ContPhy contains 325 questions. These studies examine the roles of executable worlds and adaptive reasoning, the effectiveness of physical simulation and system identification, and the applicability of ATW across foundation models. Within each comparison, the data and evaluation protocol remain fixed so that performance changes isolate the component under study.

Agentic world reasoning. Figure 6 compares four variants: Video-only VLM receives only the video and question; Perception-only (w/o Executable World) additionally receives ATW's structured detections, masks, tracks, and geometry, but neither builds nor executes a physical world; Single-pass World Model builds and executes a world once without feedback-driven revision or additional probing; and Full ATW (ours) adaptively updates its world representation, physical parameters, and subsequent probes based on the question and accumulated feedback. Overall accuracy increases across these variants from 35.40% to 42.07%, 75.40%, and 82.30% on CLEVRER, and from 47.08% to 51.38%, 59.08%, and 68.00% on ContPhy. Structured perception improves video-only reasoning, while physical execution brings a larger gain. Relative to Perception-only, Single-pass World Model raises CLEVRER counterfactual accuracy by 61.08 percentage points, from 9.73% to 70.81%. Feedback-guided revision in Full ATW then adds 6.90 and 8.92 points over Single-pass on CLEVRER and ContPhy, respectively. Despite some variation across individual reasoning categories, the overall trend across both datasets shows that executable worlds provide substantial gains, while feedback-guided revision brings further improvements (see Appendix C.2).

System identification and physical backends. Figure 7 evaluates both factors on the CLEVRER ablation subset. With Warp fixed and equal simulation budgets, CEM achieves a substantially lower median trajectory RMSE than Random Search, while raising per-question accuracy from 43.45% to 82.76% and per-option accuracy from 69.49% to 93.19%. With CEM fixed, Warp achieves the lowest trajectory error and the highest per-question accuracy among the PhysMind physical model (Yang et al., 2026), MuJoCo (Todorov et al., 2012), and Warp. This paired design separates optimization quality from backend fidelity: stronger search improves the same simulator, while a more suitable backend improves reasoning under the same optimizer. Across both ablations, lower fitting error is consistently associated with stronger downstream reasoning, demonstrating that reliable physical grounding depends on both system identification and simulation (see Appendix C.3).

![](images/624be09b26d4702852bd9125ffb75e58b75746be9f6b3210fc1ea8d33a9ff752.jpg)

![](images/9ab1081d58216ffe8874c5ad45ac5cf841f5100f1e51dba8a7ea5d5591ec82e0.jpg)  
Figure 7: System-identification and physicsbackend ablations on CLEVRER. Overall perquestion accuracy versus median trajectory RMSE (log scale). System identification fixes Warp; backend comparisons fix CEM. Both ablations share CEM + Warp. CEM and Random Search use equal simulation budgets.  
Figure 8: Across foundation models. Overall per-question accuracy; open: Video-only VLM, filled: ATW. Values below the lines show gains in percentage points.

Generalization across foundation models. Figure 8 extends the Gemini-3-Flash evaluation to Qwen3-VL-235B-A22B, replacing the backbone in both world modeling and world probing. With Qwen, ATW improves overall per-question accuracy over Video-only VLM from 38.85% to 76.32% on CLEVRER and from 45.23% to 62.46% on ContPhy. Across the two evaluated backbones, every paired comparison improves in the same direction, demonstrating backbone portability in both the tested rigid-body and continuous-physics settings.

## 5 CONCLUSION

We presented Asking the World (ATW), a training-free framework for generalist physical reasoning through agentic world modeling and probing. Rather than treating simulation as a fixed downstream tool, ATW constructs task-relevant executable worlds and uses observations and execution feedback to revise modeling decisions and guide question-directed probes. PolyWorld Engine supports diverse materials and coupled interactions, while system identification grounds world dynamics in visual evidence. Across CLEVRER, ContPhy, and three real-world scenarios, ATW improves reasoning accuracy, while controlled ablations show complementary contributions from physical execution and feedback-guided adaptation. These findings suggest that actively building and interrogating executable worlds provides a promising path toward physical reasoning beyond direct visual inference.

## AI USE STATEMENT

Generative AI tools, including large language models, were used to assist with implementing portions of the research code, drafting portions of the manuscript, and improving clarity and readability. They were not used to generate synthetic datasets or to formulate or prove mathematical claims. The authors reviewed and tested all AI-assisted code, verified the reported results, and take responsibility for the final content.

## ETHICS STATEMENT

This work uses public benchmarks and author-recorded tabletop scenes and involves no human participants, personal data, or decisions about individuals. Because errors in perception or simulation can produce incorrect conclusions, the method should not be used in safety-critical settings without independent validation.

## REPRODUCIBILITY STATEMENT

The paper describes the method, experimental protocol, metrics, baselines, and ablations. Appendices A–D provide additional implementation context, benchmark and real-world protocols, supplementary experimental results, and system-prompt summaries to support interpretation and assessment of the reported findings.

## REFERENCES

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report, 2025. URL https: //arxiv.org/abs/2511.21631.

Vahid Balazadeh, Mohammadmehdi Ataei, Hyunmin Cheong, Amir Hosein Khasahmadi, and Rahul G. Krishnan. Physics context builders: A modular framework for physical reasoning in vision-language models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 7318–7328, 2025.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. VideoPhy: Evaluating physical commonsense for video generation, 2024. URL https://arxiv.org/abs/2406.03520.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. SAM 3: Segment anything with concepts, 2025. URL https://arxiv.org/abs/2511.16719.

Chuhao Chen, Zhiyang Dou, Chen Wang, Yiming Huang, Anjun Chen, Qiao Feng, Jiatao Gu, and Lingjie Liu. Vid2Sim: Generalizable, video-based reconstruction of appearance, geometry and physics for mesh-free simulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26545–26555, 2025a.

Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J. Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, et al. SAM 3D: 3dfy anything in images, 2025b. URL https://arxiv.org/abs/2511.16624.

Zeren Chen, Xiaoya Lu, Zhijie Zheng, Pengrui Li, Lehan He, Yijin Zhou, Jing Shao, Bohan Zhuang, and Lu Sheng. Geometrically-constrained agent for spatial reasoning, 2025c. URL https: //arxiv.org/abs/2511.22659.

Zhenfang Chen, Jiayuan Mao, Jiajun Wu, Kwan-Yee Kenneth Wong, Joshua B. Tenenbaum, and Chuang Gan. Grounding physical concepts of objects and events through dynamic visual reasoning. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=bhCDO\_cEGCz.

Anoop Cherian, Radu Corcodel, Siddarth Jain, and Diego Romeres. LLMPhy: Parameteridentifiable physical reasoning combining large language models and physics engines. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings ofMachine Learning Research. PMLR, 2026.

Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Campagnolo Guizilini, and Yue Wang. PhysBench: Benchmarking and enhancing vision-language models for physical world understanding. In International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=Q6a9W6kzv5.

Mingyu Ding, Zhenfang Chen, Tao Du, Ping Luo, Joshua B. Tenenbaum, and Chuang Gan. Dynamic visual reasoning by learning differentiable physics models from video and language. In Advances in Neural Information Processing Systems, volume 34, pp. 887–899, 2021.

Sunqi Fan, Jiashuo Cui, Meng-Hao Guo, and Shuojin Yang. Tool-augmented spatiotemporal reasoning for streamlining video question answering task. In Advances in Neural Information Processing Systems, volume 38, pp. 129243–129273, 2025.

Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. VideoAgent: A memory-augmented multimodal agent for video understanding. In Computer Vision – ECCV 2024, pp. 75–92. Springer, 2024. doi: 10.1007/978-3-031-72670-5 5.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-R1: Reinforcing video reasoning in MLLMs. In Advances in Neural Information Processing Systems, volume 38, pp. 99114–99137, 2025.

Sara Ghazanfari, Francesco Croce, Nicolas Flammarion, Prashanth Krishnamurthy, Farshad Khorrami, and Siddharth Garg. Chain-of-frames: Advancing video understanding in multimodal LLMs via frame-aware reasoning, 2025. URL https://arxiv.org/abs/2506.00318.

Google DeepMind. Gemini 3 Flash: frontier intelligence built for speed. https://blog. google/products-and-platforms/products/gemini/gemini-3-flash/, 2025. Accessed: 2026-09-24.

Google DeepMind. Gemini 3.1 Pro: A smarter model for your most complex tasks. https://blog.google/innovation-and-ai/models-and-research gemini-models/gemini-3-1-pro/, 2026. Accessed: 2026-09-24.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14953–14962, 2023. doi: 10.1109/CVPR52729.2023.01436.

Taylor A. Howell, Simon Le Cleac’h, Jan Brudigam, Qianzhong Chen, Jiankai Sun, J. Zico Kolter,¨ Mac Schwager, and Zachary Manchester. Dojo: A differentiable physics engine for robotics, 2022. URL https://arxiv.org/abs/2203.00806.

Dacheng Li, Yunhao Fang, Yukang Chen, Shuo Yang, Shiyi Cao, Justin Wong, Michael Luo, Xiaolong Wang, Hongxu Yin, Joseph Gonzalez, Ion Stoica, Song Han, and Yao Lu. WorldModel-Bench: Judging video generation models as world models. In Advances in Neural Information Processing Systems, volume 38, pp. 61064–61088, 2025a. doi: 10.52202/085713-1834.

Jian Li, Wan Han, Ning Lin, Yu-Liang Zhan, Ruizhi Chengze, Haining Wang, Yi Zhang, Hongsheng Liu, Zidong Wang, Fan Yu, et al. SlotPi: Physics-informed object-centric reasoning models, 2025b. URL https://arxiv.org/abs/2506.10778.

Puyin Li, Tiange Xiang, Ella Mao, Shirley Wei, Xinye Chen, Adnan Masood, Fei-Fei Li, and Ehsan Adeli. QuantiPhy: A quantitative benchmark evaluating physical reasoning abilities of visionlanguage models, 2025c. URL https://arxiv.org/abs/2512.19526.

Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. PAC-NeRF: Physics augmented continuum neural radiance fields for geometry-agnostic system identification, 2023. URL https://arxiv.org/abs/2303. 05512.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In IEEE International Conference on Robotics and Automation, pp. 9493–9500, 2023. doi: 10.1109/ ICRA48891.2023.10160591.

Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

Haowen Liu, Shaoxiong Yao, Haonan Chen, Jiawei Gao, Jiayuan Mao, Jia-Bin Huang, and Yilun Du. SIMPACT: Simulation-enabled action planning using vision-language models, 2025. URL https://arxiv.org/abs/2512.05955.

Jiahao Lu, Jiayi Xu, Wenbo Hu, Ruijie Zhu, Chengfeng Zhao, Sai-Kit Yeung, Ying Shan, and Yuan Liu. Track4world: Feedforward world-centric dense 3d tracking of all pixels. arXiv preprint arXiv:2603.02573, 2026.

Zhanpeng Luo, Ce Zhang, Silong Yong, Cunxi Dai, Qianwei Wang, Haoxi Ran, Guanya Shi, Katia P. Sycara, and Yaqi Xie. pySpatial: Generating 3d visual programs for zero-shot spatial reasoning. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=yv15C8ql24.

Miles Macklin. Warp: A high-performance python framework for GPU simulation and graphics. NVIDIA GPU Technology Conference (GTC), March 2022. URL https://github.com/ NVIDIA/warp.

Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation, 2024. URL https://arxiv.org/abs/2410.05363.

OpenAI. Hello GPT-4o. https://openai.com/index/hello-gpt-4o/, 2024. Accessed: 2026-09-24.

OpenAI. Introducing GPT-5.5. https://openai.com/index/ introducing-gpt-5-5/, 2026. Accessed: 2026-09-24.

D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. ViperGPT: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11854–11864, 2023. doi: 10.1109/ICCV51070.2023.01092.

Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012.6386109.

Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys. GeoCalib: Learning single-image calibration with geometric optimization. In Computer Vision – ECCV 2024, pp. 1–20. Springer Nature Switzerland, 2024. doi: 10.1007/978-3-031-73661-2 1.

Qi Wang, Yanrui Yu, Ye Yuan, Rui Mao, and Tianfei Zhou. VideoRFT: Incentivizing video reasoning capability in MLLMs via reinforced fine-tuning. In Advances in Neural Information Processing Systems, volume 38, pp. 4350–4376, 2025.

Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. FoundationPose: Unified 6d pose estimation and tracking of novel objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17868–17879, 2024. doi: 10.1109/CVPR52733.2024.01692. URL https://openaccess.thecvf.com/content/CVPR2024/html/Wen\_ FoundationPose\_Unified\_6D\_Pose\_Estimation\_and\_Tracking\_of\_Novel\_ Objects\_CVPR\_2024\_paper.html.

Jingze Wu, Quan Zhang, Hongfei Suo, Zeqiang Cai, and Hongbo Chen. Beyond perceptual shortcuts: Causal-inspired debiasing optimization for generalizable video reasoning in lightweight MLLMs, 2026. URL https://arxiv.org/abs/2605.01324.

Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, and Animesh Garg. SlotFormer: Unsupervised visual dynamics simulation with object-centric models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.05861.

Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. PhysGaussian: Physics-integrated 3d gaussians for generative dynamics. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4389–4398, 2024. doi: 10.1109/CVPR52733.2024.00420.

Wenjiang Xu, Mingkang Zhang, Cindy Wang, Rui Fang, Lusong Li, Jing Xu, Jiayuan Gu, Zecui Zeng, and Rui Chen. Embodied tree of thoughts: Deliberate manipulation planning with embodied world model. IEEE Robotics and Automation Letters, pp. 1–8, 2026. doi: 10.1109/LRA.2026.3716133.

Chen Yang, Shenxiang Zeng, Haoyang Zhao, Zhouyuan Xu, Youquan He, Haoyu Li, Mingyi Deng, Jiansheng Fan, and Chen Wang. PhysMind: From video to executable worlds for training-free physical reasoning, 2026. URL https://arxiv.org/abs/2608.04575.

Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Lan Xu, Wei Yang, Jiayuan Gu, and Jingyi Yu. CAST: Component-aligned 3d scene reconstruction from an RGB image. ACM Transactions on Graphics, 44(4):1–19, 2025. doi: 10.1145/3730841.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.03629.

Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B. Tenenbaum. CLEVRER: Collision events for video representation and reasoning, 2019. URL https://arxiv.org/abs/1910.01442.

Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Chenyang Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, and Haiwen Feng. Vision-as-inverse-graphics agent via interleaved multimodal reasoning, 2026. URL https://arxiv.org/abs/2601.11109.

Z.ai. GLM-V: GLM-4.6V, GLM-4.5V, and GLM-4.1V-Thinking. https://github.com/ zai-org/GLM-V, 2025. Accessed: 2026-09-24.

Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y. Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T. Freeman. PhysDreamer: Physics-based interaction with 3d objects via video generation. In Computer Vision – ECCV 2024, pp. 388–406, 2024. doi: 10.1007/ 978-3-031-72627-9 22.

Haoyu Zhao, Hao Wang, Xingyue Zhao, Hao Fei, Hongqiu Wang, Chengjiang Long, and Hua Zou. PhysSplat: Efficient physics simulation for 3d scenes via MLLM-guided gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5242–5252, 2025.

Zhicheng Zheng, Xin Yan, Zhenfang Chen, Jingzhou Wang, Qin Zhi Eddie Lim, Joshua B. Tenenbaum, and Chuang Gan. ContPhy: Continuum physical concept learning and reasoning from videos. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 61526–61558. PMLR, 2024.

## A MORE IMPLEMENTATION DETAILS

## A.1 MODELS AND VISUAL FOUNDATION MODULES

Reasoning models. Gemini-3-Flash (Google DeepMind, 2025) serves as the default backbone for both world modeling and world probing. The cross-backbone experiment replaces the model in both stages with Qwen3-VL-235B-A22B (Bai et al., 2025). These models select operations, interpret returned evidence, and produce answers. We do not train or fine-tune the reasoning models or visual modules on CLEVRER or ContPhy.

SAM 3. SAM 3 (Carion et al., 2025) supports concept-prompted segmentation of objects in images and videos. In ATW, the agent supplies object descriptions to obtain masks and associated video identities. These masks delimit the visual evidence used for object reconstruction and motion tracking, and connect image regions to retained scene entities. The agent can inspect the segmentation results and request another segmentation when the current masks do not adequately capture the objects relevant to modeling.

Track4World. Track4World (Lu et al., 2026) provides dense 3D tracking in a shared world coordinate system. We use its Depth Anything 3 variant to obtain geometric observations and inter-frame correspondences from the input video. For dynamic rigid objects, correspondences initialize pose propagation, followed by point-cloud registration against the observed geometry. The resulting motion estimates are retained with object identities and confidence information, providing tempora evidence for fitting physical parameters and comparing simulated motion with observations.

Depth Anything 3. Depth Anything 3 (Lin et al., 2025) recovers scene geometry from visual inputs and serves as the geometric backbone of our Track4World configuration. Through this integration, the perception pipeline obtains camera intrinsics, camera poses, and dense 3D points in camera and world coordinates. These outputs provide the spatial reference for aligning reconstructed object geometry with the video. Geometry confidence and validity information support the selection of usable observations for downstream reconstruction and tracking.

SAM 3D Objects. SAM 3D Objects (Chen et al., 2025b) reconstructs object geometry from an image and an object mask. ATW applies it to selected reference observations to obtain canonical object geometry, which is aligned with the recovered scene geometry. For rigid objects, this reference shape is retained while subsequent frames update the object’s pose. Keeping geometry and motion separate provides a consistent object representation for tracking, physical modeling, and visualization of the reconstructed world.

FoundationPose. FoundationPose (Wen et al., 2024) estimates and refines the 6D pose of an object using its geometry and visual observations. In our implementation, it refines the reference-frame alignment between the reconstructed object and the observed scene. This pose anchors the canonical object geometry in the recovered coordinate system. Subsequent dynamic poses use Track4World correspondences and point-cloud registration, while static objects retain their aligned reference pose. FoundationPose therefore supplies the initial geometric alignment used by temporal tracking.

## A.2 WARP-BASED SIMULATION AND CEM FITTING

Shared simulation interface. PolyWorld Engine uses Warp (Macklin, 2022) to execute physical models for rigid bodies, soft bodies, cloth, ropes, and fluids. Each model receives scene geometry, initial conditions, and a candidate parameter vector, and produces a simulated trajectory. Warp kernels accelerate the forward dynamics on the GPU, while reusable scene data and simulation buffers support repeated candidate evaluation. The fitting interface expresses this process as

$$
\begin{array} { r } { \widehat { Y } _ { 1 : T } ( \theta ) = \mathcal { O } _ { h } ( \mathcal { F } _ { h } ( s _ { 0 } , \theta ; T ) ) , } \end{array}\tag{3}
$$

where h denotes the physical representation, $s _ { 0 }$ its initial state, $\mathcal { F } _ { h }$ the forward simulator, and $\mathcal { O } _ { h }$ the readout that converts simulated states into quantities comparable to visual observations.

Observation-driven fitting. System identification fits the parameters exposed by the instantiated physical model to the retained visual evidence. These parameters describe motion, material response, contact, or boundary conditions, depending on the system. Across the implementations, the fitting objective can be expressed as

$$
\theta ^ { \star } = \arg \operatorname* { m i n } _ { \theta \in \Theta _ { h } } \mathcal { L } _ { h } ( \theta ) , \qquad \mathcal { L } _ { h } ( \theta ) = D _ { h } \Big ( \widehat { Y } _ { 1 : T } ( \theta ) , Y _ { 1 : T } \Big ) + \lambda _ { h } R _ { h } ( \theta ) ,\tag{4}
$$

where $Y _ { 1 : T }$ is the observed evidence, $D _ { h }$ measures the corresponding motion or shape discrepancy, and $\Theta _ { h }$ specifies admissible parameters. $R _ { h }$ represents a parameter prior when used, with $\lambda _ { h } = 0$ otherwise. Observation validity and confidence determine which measurements contribute to the discrepancy. This formulation allows each physical model to use the evidence that directly describes its dynamics.

Evidence across physical systems. For rigid bodies, fitting compares simulated and tracked positions and rotations to recover motion and contact-response parameters. For soft bodies, it combines center motion with directional deformation, linking material response to observed changes in shape. For cloth, it compares simulated surface points with tracked trajectories to fit motion, stiffness, damping, and boundary conditions. For ropes, it uses centerline geometry and the trajectories of connected objects to fit constrained dynamics. Forfluids, it compares simulated spatial distributions with observed fluid regions to fit flow behavior and source conditions, including relative density and emission parameters. These observation choices adapt the shared fitting procedure to the representation of each system.

CEM search. CEM maintains a diagonal Gaussian proposal over the fitted parameters. At iteration $k ,$ it samples a population, projects candidates onto their bounds, and selects the lowest-loss candidates as an elite set:

$$
\begin{array} { r } { \theta _ { k } ^ { ( j ) } = \Pi _ { \Theta _ { h } } \big ( \mu _ { k } + \sigma _ { k } \odot \epsilon _ { j } \big ) , \quad \epsilon _ { j } \sim \mathcal { N } ( 0 , I ) , \qquad { \mathcal { E } } _ { k } = \mathrm { T o p K } _ { \mathrm { l o w e s t } } \_ { \mathcal { L } _ { h } } \{ \theta _ { k } ^ { ( j ) } \} _ { j = 1 } ^ { N } . } \end{array}\tag{5}
$$

Here N is the population size and K is the elite count. The elite mean and standard deviation update the proposal through the smoothed rule in Equation 2, with a variance floor maintaining exploration. The search reuses promising candidates and progressively extends the fitting horizon so that later interactions contribute to parameter fitting. Model-specific parameterizations and readouts connect this common search procedure to the different forward simulators.

Reusable fitted worlds. The fitted parameters are retained with the modeled geometry and initial conditions, yielding an executable world that can be replayed and queried. World probing uses these assets to continue the dynamics or execute an intervention, such as removing an object or changing an exposed physical condition. The same forward model therefore connects parameter fitting with subsequent question-conditioned experiments.

## A.3 QUERY-CONDITIONED PHYSICAL ROLLOUT TOOLS

The probing agent receives the retained world assets, object identities, available capabilities, and the current physical question. Tool availability and temporal scope depend on the question and the instantiated world. The following signatures are pseudocode abstractions of general operations; each backend exposes the subset supported by its physical representation.

Asset and property queries. InspectWorld(objects, fields) retrieves identities, geometry, visibility, and available capabilities. ReadProperty(objects, property) reads supported physical quantities from the fitted world, allowing property questions to use existing parameters without an additional rollout. These queries preserve the distinction between observed attributes and fitted physical properties.

Rollouts and interventions. Rollout(world, intervention, horizon) executes an unedited or modified world and returns a rollout identifier. Predictive questions use the unedited world and read the interval after the observation cutoff. Counterfactual questions specify an intervention before execution, while goal-directed questions can test candidate interventions in separate worlds. Object removal is one supported edit; other edits are exposed according to backend capabilities. In the rigid implementation, removal applies from the modeled beginning of the replay. Fitted parameters of the retained objects are reused for the edited rollout.

Evidence readout. ReadEvents(rollout, objects, interval) and ReadTrajectory(rollout, object, interval) retrieve time-localized outcomes. ReadTimeline(rollout, interval) organizes collision, entry, and exit events chronologically, retaining event times and participating object identities to support reasoning about their order. QueryWorld(rollout, predicate, interval) requests semantic conditions, such as contact, motion, or region occupancy, with the appropriate temporal aggregation. Readouts refer to an explicit world and time interval. Several queries can accompany a rollout request, and retained rollouts can be queried repeatedly.

Cross-world comparison. CompareWorlds(base, edited, objects) compares selected objects across an original world and an intervened world. It aligns their trajectories at shared frames and reports the maximum center displacement for each object, together with the two event sequences and simulation horizons. These outputs show how an intervention changes object motion and event occurrence, providing evidence for comparing the consequences of alternative physical conditions.

Observation-based checks. InspectObservedMotion(object, interval) retrieves motion-related measurements from the object’s masks in the input video. The returned observations include image-space centroid positions, visible mask areas, and image-boundary contact over the requested interval. Together with the original frames, these measurements let the agent inspect where and when an object moves and compare the observed changes with its tracked or simulated trajectory.

Inspection and control. DescribeTool(tool) provides an operation’s inputs, parameter constraints, and conditions of use, helping the agent prepare its next call. InspectFrames(times, object) retrieves visual observations, while InspectRollout(rollout, interval) presents the simulated interval for visual inspection of motion and deformation alongside numerical readouts. ReadEvidence(id) revisits an earlier tool result, and UpdateBindings(correction, reason) revises question-toobject associations using accumulated evidence. Finally, Finish(evidence) ends probing and passes the evidence to answer generation. Tool returns retain their observation or simulation source, object and rollout identities, temporal coverage, and execution status.

## A.4 SAM 3 AGENT FOR GENERAL-PURPOSE VIDEO SEGMENTATION

Object discovery and seed selection. The SAM 3 agent combines VLM-based visual reasoning with SAM 3 (Carion et al., 2025) segmentation to produce object-level mask tracks. Given a video and indexed frame samples, the VLM identifies distinct entities in the interaction workspace and generates discriminative descriptions and segmentation prompts. It selects a reference frame for each entity, prioritizing complete outlines, visibility, and identity clarity. SAM 3 then generates candidate masks, which the VLM inspects alongside the original image to select a mask covering the target while excluding neighboring objects and background.

Feedback-guided refinement and propagation. When no candidate is accepted, the agent checks the target’s visibility and can select another reference frame or propose alternative descriptions of the same entity. Candidate masks and rejection feedback guide these adjustments while preserving the target identity. Accepted seeds initialize bidirectional video-mask propagation in a shared multi-object state. The module returns indexed mask sequences, object-to-track associations, and temporal quality summaries for downstream reconstruction and motion tracking. This interface supports reusable object segmentation across scenes without requiring an object mesh or a rigid-motion assumption.

## B EVALUATION BENCHMARK DETAILS

## B.1 CLEVRER

CLEVRER (Yi et al., 2019) is a video reasoning benchmark designed to study causal understanding of physical events. It pairs synthetic videos of interacting rigid objects with questions linking object identities, motion, and collisions. Objects vary in shape, color, and material appearance, allowing questions to refer to specific entities and their interactions. We evaluate ATW on a validation subset containing 1,000 videos, 4,280 questions, and 14,228 answer options.

Explanatory questions. These questions ask which earlier events are responsible for an observed outcome. Answering them requires identifying the relevant objects, recovering the temporal order of their interactions, and relating a target event to preceding motion or collisions. The candidate answers refer to possible explanatory events, so the model must distinguish the interactions that account for the outcome from other events in the video.

Predictive questions. These questions ask what will happen after the observed video ends. The model must infer how the objects will continue moving and whether their subsequent trajectories will lead to the events described by the answer options. This category connects the visible history of motion and collisions with future physical interactions, testing whether the observed evidence supports an accurate continuation of the scene.

Counterfactual questions. These questions ask how events would change under a specified intervention, such as removing an object from the scene. The model must identify the intervention target and reason about how the remaining objects would interact under the modified conditions. Because removing an object can alter subsequent collisions and motion, answering requires considering the consequences of the edit throughout the relevant event sequence.

Metrics. We report per-option and per-question accuracy for each category and overall. Per-option accuracy is the fraction of answer options whose correctness is judged correctly. Per-question accuracy counts a question as correct only when every associated option is judged correctly. We use overall per-question accuracy as the primary metric. Overall scores are computed across all evaluated questions or options, respectively.

## B.2 CONTPHY

ContPhy (Zheng et al., 2024) is a benchmark for learning and reasoning about physical concepts from videos of continuum systems. It connects observed dynamics with questions about physical properties and the consequences of changing a scene. Its diverse materials and interactions support evaluation of reasoning across different physical systems. We evaluate ATW on 600 videos containing 1,950 questions.

Physical-property questions. These questions ask about physical attributes that govern the behavior of the objects or materials in a scene. The model must connect observable motion, deformation, or interaction responses with the underlying property being queried. Different scene categories provide different forms of evidence, making this category a test of whether visual dynamics can support judgments about the physical characteristics of the participating entities.

Predictive questions. These questions ask how a physical system will evolve after the observed interval. The model must interpret the current configuration and motion, then anticipate the subsequent behavior of the materials and objects involved. Depending on the scene, relevant evidence may include deformation, contact, constrained movement, or flow. The requested answer concerns the future outcome of the system under its existing conditions.

Counterfactual questions. These questions ask about the outcome of changing a condition in the observed scene. The model must interpret the specified change, identify the affected entities, and reason about how their interactions would unfold in the modified setting. Across the materia families, this requires connecting an intervention with changes in motion, deformation, or flow and evaluating the resulting outcome described by the question.

Goal-driven questions. These questions specify a desired physical outcome and ask which intervention would produce it. The model must relate each candidate change to its consequences for the system and select the option that satisfies the stated goal. Reasoning therefore proceeds from a target outcome to a suitable change in the scene, using the observed configuration and dynamics to assess the available choices.

Cloth scenes. Cloth scenes feature flexible surfaces that bend and deform as they move and interact with other objects. Their evolution depends on the initial configuration, material response, and contact geometry. Understanding these scenes requires following changes in surface shape and spatial relationships over time. They provide visual evidence for questions about material behavior, subsequent deformation, and the effects of changing the physical setup.

Rope scenes. Rope scenes feature connected flexible structures whose motion is constrained by their geometry and attachments, including interactions with pulleys and attached objects. Movement in one part of the system can affect other connected parts. Reasoning about these scenes requires tracking these relationships and understanding how forces and constraints shape the resulting motion, supporting questions about physical properties and the consequences of interventions.

Soft-ball scenes. Soft-ball scenes feature deformable bodies whose shapes change during motion and contact. Their responses to interactions provide evidence about material behavior and influence subsequent trajectories. Understanding these scenes requires relating object-level movement to visible deformation, including how the body responds around contact. The questions use this evidence to probe physical attributes, future outcomes, and changes induced by modifying the scene conditions.

Fluid scenes. Fluid scenes feature flowing materials that interact with containers, boundaries, and obstacles. Their behavior is expressed through changes in spatial distribution and the movement of material between regions. Reasoning requires following the evolving flow and relating it to scene geometry and physical conditions. These scenes support questions about material properties, future distributions, and how changes to the environment affect the resulting flow.

Metrics. We report question-answering accuracy for each question category and overall. Category accuracy is the number of correctly answered questions divided by the number of questions in that category. Overall accuracy is the total number of correctly answered questions divided by 1,950, so each evaluated question contributes equally to the final score.

## B.3 REAL-WORLD

We collect a real-world dataset using consumer-grade cameras to evaluate physical reasoning from recorded interactions. The dataset contains 60 videos and 60 questions across three scene categories, with 20 videos and 20 questions per category. It focuses on two challenging reasoning tasks: counterfactual reasoning and prediction. Billiards scenes provide counterfactual questions, while bouncing-ball and toy-car scenes provide predictive questions.

Billiards scenes. These scenes capture balls moving and colliding on a billiards table. Counterfactual questions ask how the interactions would change under a specified intervention, such as removing a ball. Answering requires identifying the relevant balls, interpreting their observed motion, and reasoning about how the intervention changes subsequent collisions. This category evaluates the connection between object-level interventions and the resulting sequence of physical events.

Bouncing-ball scenes. These scenes capture the motion of bouncing balls and their interactions with surrounding surfaces. Predictive questions ask about physical outcomes after the observed motion. The model must connect the ball’s trajectory and contact response with its subsequen movement, accounting for changes in direction caused by interactions. This category evaluates whether observed dynamics support predictions of future motion and contact events.

Toy-car scenes. These scenes capture toy cars moving through a physical environment and interacting with other scene elements. Predictive questions concern the subsequent motion or interaction of the cars. Answering requires identifying the relevant vehicles, interpreting their trajectories, and relating their movement to the surrounding geometry. This category evaluates prediction from real object motion under the spatial constraints of the recorded scene.

Ground-truth acquisition. Each scene is recorded in full from multiple synchronized viewpoints. For predictive questions, the reference answers are determined from the future portion of the complete recording, with complementary views used to recover the subsequent trajectory and resolve occlusions. Counterfactual outcomes are not directly observed; we therefore retain only questions for which human observers can determine the answer with high confidence from the multi-view recordings. In particular, cases involving unresolved spatial ambiguity or viewpoint occlusion are excluded.

Metrics. We report per-question accuracy for each scene category and overall. Each category score is the number of correctly answered questions divided by 20, and the overall score is the total number of correct answers divided by 60. Since the categories contain equal numbers of questions, overall accuracy also equals the mean of the three category accuracies.

## C MORE EXPERIMENTAL RESULTS

## C.1 DETAILED REAL-WORLD RESULTS

Table 4 reports detailed accuracy on the 60 real-world videos and questions. In addition to perquestion accuracy, we report per-option accuracy for billiards and toy-car tasks. Each bouncing-ball question requires one yes/no judgment. Overall per-judgment accuracy aggregates the 80 billiards options, 20 bouncing-ball judgments, and 60 toy-car options, for 160 judgments in total.

Table 4: Detailed real-world accuracy (%). Each scenario contains 20 videos and 20 questions. Q: per-question; O: per-option; J: per-judgment. Counts are shown in parentheses. Per-question accuracy requires all associated option judgments to be correct. CF: counterfactual; Pred.: predictive. Bold denotes the best result in each column.
<table><tr><td rowspan="2">Model</td><td colspan="2">Billiards (CF)</td><td>Bouncing ball</td><td colspan="2">Toy car (Pred.)</td><td colspan="2">Overall</td></tr><tr><td>Q (20)</td><td>0 (80)</td><td>Pred. / Q (20)</td><td>Q (20)</td><td>0 (60)</td><td>Q (60)</td><td>J (160)</td></tr><tr><td colspan="8">Foundation VLMs</td></tr><tr><td>GPT-5.5</td><td>35.00</td><td>61.25</td><td>45.00</td><td>50.00</td><td>65.00</td><td>43.33</td><td>60.63</td></tr><tr><td>Gemini-3-Flash</td><td>20.00</td><td>60.00</td><td>60.00</td><td>20.00</td><td>26.67</td><td>33.33</td><td>47.50</td></tr><tr><td colspan="8">Training-Free Methods</td></tr><tr><td>ATW (Ours)</td><td>85.00</td><td>91.25</td><td>70.00</td><td>60.00</td><td>71.67</td><td>71.67</td><td>81.25</td></tr></table>

## C.2 ADDITIONAL ABLATION RESULTS

Figure 9 reports additional ablations on the same 100-scene subsets used in Section 4.3, with full ATW included for comparison. The identity-binding ablation removes the appearance cards and identity-binding table.

![](images/0525456818c7b8099b0ef215558a503e20361690e1db3e0a4cfb74b7cd2dd7df.jpg)

![](images/747b968cdd3be9143a8b5c4340136b45f4dad21bec652f813c3866691aa223e9.jpg)  
Figure 9: Additional ablations on CLEVRER (left) and ContPhy (right). Bars show per-question accuracy for each reasoning category and overall, evaluated on 100 scenes per benchmark (435 CLEVRER questions and 325 ContPhy questions). Hatched bars denote full ATW.

Effects of identity binding and perception modules. Figure 9 shows that removing appearance cards and identity bindings reduces overall accuracy from 82.30% to 70.11% on CLEVRER and from 68.00% to 62.46% on ContPhy, with declines across every question category. These results highlight the importance of maintaining object identities when connecting visual observations, question references, and executable worlds. On CLEVRER, FoundationPose (Wen et al., 2024) tracking achieves 80.46% overall accuracy and the highest predictive accuracy of 97.14%, while matching ATW on counterfactual questions. Its tracking benefits from an input object mesh and a rigid-body prior, making it effective for these rigid-object scenes. However, its rigid-pose formulation does not capture the evolving shapes of cloth, fluids, or soft bodies. We therefore use Track4World (Lu et al., 2026) for trajectory tracking across the diverse physical systems considered in ATW. The GeoCalib (Veicht et al., 2024) variant, which uses a learning-based single-image camera calibration method to estimate camera intrinsics and gravity direction, achieves 78.62% overall accuracy. Full ATW obtains the highest overall score, while different module choices produce distinct accuracy profiles across reasoning categories.

## C.3 DETAILED PHYSICAL-GROUNDING ABLATIONS

Tables 5 and 6 provide the category-level results underlying the system-identification and physicsbackend ablations. Overall accuracies are aggregated over all questions or options rather than averaged across categories. These experiments are run independently of the agentic world-reasoning ablation.

Table 5: Detailed system-identification ablation on the 100-scene ContPhy subset. Category columns report question-answering accuracy (%). Random Search and CEM use equal simulation budgets.
<table><tr><td>Method</td><td>Property</td><td>Predictive</td><td>Counterfactual</td><td>Goal-driven</td><td>Overall</td></tr><tr><td colspan="6">System Identification (Warp Fixed)</td></tr><tr><td>Default parameters</td><td>70.40</td><td>39.53</td><td>56.16</td><td>80.49</td><td>60.31</td></tr><tr><td>Random Search</td><td>70.40</td><td>51.16</td><td>63.01</td><td>46.34</td><td>60.62</td></tr><tr><td>CEM (Ours)</td><td>80.00</td><td>54.65</td><td>63.01</td><td>65.85</td><td>67.69</td></tr></table>

ContPhy trends. CEM achieves the strongest aggregate performance among the three configurations. Relative to Random Search, it improves physical-property, predictive, goal-driven, and overall accuracy, while matching counterfactual accuracy at 63.01%. These results indicate that CEM-based system identification improves aggregate reasoning performance, with category-specific differences.

Table 6: Detailed system-identification and physics-backend ablations on the 100-scene CLEVRER subset. Accuracy is reported in percent (Q: per-question; O: per-option), and the final column reports median trajectory RMSE in meters. The CEM and Warp + CEM rows are the same shared configuration.
<table><tr><td rowspan="2">Method</td><td colspan="2">Explanatory</td><td colspan="2">Predictive</td><td colspan="2">Counterfactual</td><td colspan="2">Overall</td><td rowspan="2">Median trajectory RMSE (m)</td></tr><tr><td>Q</td><td>0</td><td>Q</td><td>0</td><td>Q</td><td>0</td><td>Q</td><td>0</td></tr><tr><td>System Identification (Warp Fixed)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Default parameters</td><td>25.56</td><td>50.62</td><td>97.14</td><td>98.57</td><td>1.08</td><td>50.23</td><td>26.67</td><td>55.11</td><td>0.292053</td></tr><tr><td>Random Search</td><td>41.67</td><td>64.77</td><td>91.43</td><td>92.86</td><td>27.03</td><td>69.18</td><td>43.45</td><td>69.49</td><td>0.034992</td></tr><tr><td>CEM (Ours)</td><td>86.11</td><td>94.31</td><td>94.29</td><td>97.14</td><td>75.14</td><td>91.22</td><td>82.76</td><td>93.19</td><td>0.007655</td></tr><tr><td>Physics Backends (CEM Fixed)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PhysMind model + CEM</td><td>58.33</td><td>76.15</td><td>94.29</td><td>95.71</td><td>33.51</td><td>72.88</td><td>53.56</td><td>76.58</td><td>0.059138</td></tr><tr><td>MuJoCo + CEM</td><td>68.89</td><td>83.38</td><td>91.43</td><td>94.29</td><td>71.89</td><td>87.52</td><td>73.79</td><td>86.31</td><td>0.016017</td></tr><tr><td>Warp + CEM (Ours)</td><td>86.11</td><td>94.31</td><td>94.29</td><td>97.14</td><td>75.14</td><td>91.22</td><td>82.76</td><td>93.19</td><td>0.007655</td></tr></table>

CLEVRER trends. Across the system-identification variants, trajectory RMSE decreases from default parameters to Random Search and then to CEM, while overall per-question and per-option accuracy increase in the same order. The largest reasoning gains appear in explanatory and counterfactual questions, while predictive accuracy remains high across configurations. With CEM fixed, the comparison from the PhysMind model through MuJoCo to Warp shows the same broad relationship between lower trajectory error and higher overall accuracy. Together, these trends show that both system identification and the physical backend determine the quality of the executable world used for downstream reasoning.

## D SYSTEM PROMPTS

This section presents the principal system instructions used in ATW. We organize them according to the main stages of the framework: world-representation routing, object discovery and segmentation, world probing, and evidence-based answer generation.

## World-representation routing.

Inspect the complete video together with all associated questions, but do not answer them at this stage. First identify whether the main interactions involve free collision and support, constrained relative displacement, or another form of material motion. Examine depth ordering and outline truncation during contact. Use a three-dimensional representation when occlusion makes a visible mask boundary different from the physical contact surface, or when folding, covering, and other out-of-plane states cannot be described by planar outlines. Use a planar representation when the relevant connections, endpoints, displacements, and boundaries provide sufficient interaction geometry. Base the decision on visible evidence rather than object names, dataset conventions, or rendering style, and report the observations and remaining uncertainty that support the route.

## Object discovery and SAM 3 segmentation.

Inspect the entire video and indexed frame samples to enumerate every distinct physical entity in the interaction workspace. Include stationary obstacles, supports, and dividers that constrain the experiment, while excluding continuous environment surfaces, shadows, reflections, annotations, and rendering artifacts. Keep same-category instances separate and assign each entity a stable opaque identity. Classify its physical type as rigid, soft body, cloth, or rope from temporal evidence, distinguishing real deformation from occlusion, viewpoint change, and rigid rotation. For each entity, generate short image-grounded descriptions and select a seed frame that prioritizes a complete in-frame outline, visibility, and identity clarity. Accept a SAM 3 candidate only when it covers the intended foreground entity and excludes neighboring objects and background. If no candidate is adequate, refine the description or reselect the seed frame while preserving the target identity.

## World probing (shared instructions).

Investigate the literal question before producing an answer and treat its options independently. Match the referenced entities, world, intervention, and time interval. Missing evidence is unknown rather than false, so inspect returned results before deciding that the evidence is sufficient. Use only the tools exposed by the selected route and issue one to six independent tool calls in a round. A rollout identifier must come from an earlier successful operation, and subsequent queries must read from the corresponding world. Follow returned guidance when evidence is missing and do not repeat an unchanged failed request. Termination or video-fallback requests are issued alone. Because the answer stage receives compact textual evidence, preserve every necessary visual observation together with the evidence reference from which it was obtained.

## Evidence-based answer generation.

Answer the literal question using only its retained evidence. Match the referenced entities, intervention, and time window, and handle negated statements explicitly. Evaluate each relevant claim independently: unknown evidence does not imply a negative decision, and a claim that an event did not occur requires adequate coverage of the requested interval. Cite the evidence supporting the answer and provide a short factual reason without repeating the full tool history. Return one object that follows the specified output contract. If the retained evidence cannot determine the answer, return an explicit unresolved result that identifies the missing evidence rather than inventing a conclusion.