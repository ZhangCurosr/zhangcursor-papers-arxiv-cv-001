# DreamStream: Towards Policy-Oriented Generative Simulation for End-to-End Driving

Ziyang Leng<sup>1∗</sup> Sicheng Mo<sup>1∗</sup> Seth Z. Zhao<sup>1</sup> Haoyuan Cai<sup>1</sup>

Yu Zeng<sup>2</sup> Rowan McAllister<sup>2</sup> Bolei Zhou<sup>1</sup>

<sup>1</sup>University of California, Los Angeles <sup>2</sup>Toyota Research Institute

Abstract: Faithfully evaluating end-to-end driving policies in simulation requires observations that are not merely photo-realistic, but preserve the scene features a policy relies on to make decisions. Existing platforms, however, exhibit a simto-real visual gap that corrupts policy perception, undermining their ability to assess a policy’s closed-loop decision-making. To this end, we propose Dream-Stream, a generative, closed-loop simulator that achieves policy-orientedfidelity using a simulator-grounded autoregressive video model. Our video model is distilled from a large pretrained video model via traffic layout guidance, varying visual appearance while preserving policy-relevant features such as scenario layout and the temporal consistency of dynamic objects. We further observe that perceptual metrics like FID misrank how well these features are preserved. To tackle this, we introduce FDπ, a new multi-representation metric that measures the sim-to-real gap as the Frechet distance over scene-context features from pub-´ lic E2E policies. Under FDπ, DreamStream improves over the strongest prior closed-loop simulator by 1.6× on nuScenes and 4.7× on NAVSIM, and induces the least perturbation to policy’s perceptual observability. Based on Dream-Stream, we construct Navhard-CL benchmark, which turns non-reactive realworld benchmark NAVSIM into interactive testing environments with adversarial driving behaviors and weather variations. This benchmark exposes many failure modes of driving policies, such as scorer bias and lack of recovery behaviors, that prior closed-loop benchmarks overlook. Code and data are available at https://github.com/VAIL-UCLA/DreamStream.

Keywords: Autonomous Driving, Video Model, Closed-loop Simulation

## 1 Introduction

Recent studies [1, 2, 3] reveal a substantial open-loop (OL) to closed-loop (CL) evaluation gap, making closed-loop simulation necessary for the reliable assessment of end-to-end (E2E) driving policies. Since E2E policies map sensor observations directly to actions, a closed-loop simulator must provide controllable traffic scenarios, namely physically grounded layouts and agent behaviors, while rendering observations that are visually realistic and preserve the scene features a policy relies on to make decisions. Controllability of traffic scenarios is already well supported by physics-based simulators [4, 5], yet these leave a substantial sim-to-real visual appearance gap [6, 7, 3]; conversely, generative and reconstruction-based simulators improve realism but operate within narrow visual domains and often forfeit controllability [8, 9, 10, 11, 12]. No existing platform delivers both, leaving closed-loop evaluation unable to thoroughly assess a policy’s driving performance.

![](images/d1d787f0900e5e58ea33b106e28e5288f10c66384b0e2a93571dc45ec49a2b55.jpg)  
Figure 1: Motivation: existing benchmarks for E2E driving either omit behavioral testing or eval uate unfaithfully due to the sim-to-real gap. DreamStream bridges sim-to-real via policy-oriented visual alignment and grounded traffic for faithful closed-loop evaluation.

In this work, we present DreamStream, a generative closed-loop simulator that bridges the aforementioned evaluation gaps as illustrated in Fig. 1. DreamStream couples a physics simulator with an autoregressive video model with the following protocol: the physics simulator constructs and rolls out scenarios with interactive agents, and the autoregressive video model translates the simulator’s symbolic state into photorealistic visual observations. Such a design ensures visual realism for ego-agent observation, while background traffic is logged using real-world dynamics.

Building DreamStream’s video model imposes three requirements that previous generative simulators cannot meet simultaneously. (i) Autoregressive long-horizon rollout. Closed-loop evaluation requires generating hundreds of consecutive rollout frames. We adopt an autoregressive design with early-timestep latent states as the KV cache to maintain consistency across the entire scenario. (ii) Scene layout alignment. E2E policies attend to visual cues, such as lane geometry and trafficelement state, for decisions. Without explicit conditioning on the simulator’s layout, the video model may produce misaligned lane geometry and agent positions. We therefore introduce traffic layout guidance during distillation, applying classifier-free guidance on the layout condition so that rendered frames preserve simulator state. (iii) Diverse visual appearance. Policies must be stresstested under diverse and scarce visual conditions, but a video model trained only on small in-domain datasets [11, 12] inherits their narrow visual distribution. We propose diverse-scene distillation to retain broad visual priors that are absent from the training data. Together, these designs enable DreamStream to render policy-decision-relevant, long-horizon, and visually diverse observations for closed-loop policy evaluation.

To verify a simulator achieves policy-oriented visual alignment, we introduce FDπ. Prior simulators measure visual quality with FID [13] and controllability with 3D detection or map segmentation accuracy [14, 11, 12], and recent benchmarks expand to multiple aspects [15]. However, FID computed on small in-domain driving datasets is biased [16, 15, 17], and controllability metrics reflect limited alignment dimensions, which may not be relevant to the policy’s decision at all. FDπ instead measures visual alignment using the scene context feature from E2E policies, computing the Frechet´ distance between this feature for paired real and simulated scenes. As shown in Fig. 3, FDπ exhibits a substantially stronger correlation with the driving performance than FID and FVD, confirming that it captures visual cues policies depend on. Under FDπ, DreamStream outperforms the strongest prior closed-loop simulator by 1.6× on nuScenes and 4.7× on NAVSIM.

Based on the diversity of visual appearance and the scenario customizability of DreamStream, we construct Navhard-CL, a closed-loop benchmark that turns the non-reactive, static openloop benchmark NAVSIM [18] into reactive, dynamic closed-loop scenario testing environments. Navhard-CL could be used to systematically diagnose the observation and behavior gaps that current E2E policies face under closed-loop deployment. It consists of a real-world log-replay base set Navhard-Base [19] and two challenging variations: Navhard-AdvBehavior introduces adversarial agents to manifest safety-critical interactions, and Navhard-AdvWeather renders the same scenarios under diverse weather, lighting, and road-surface conditions. The benchmark reveals three failure modes under closed-loop settings: scorer miscalibration, proposal-coverage failure, and vision encoder brittleness under appearance shifts. These findings suggest that Navhard-CL could complement existing benchmarks for evaluating driving performance. We summarize our contributions as follows:

• We design DreamStream, a generative closed-loop simulator that pairs a physics simulator with an autoregressive video model. DreamStream delivers both visual realism and scenario realism necessary for reliable closed-loop simulation for E2E driving policies.

• We propose FDπ, a policy-oriented visual alignment metric for driving world models. Under FDπ, DreamStream outperforms the strongest prior simulator by 1.6× on nuScenes and 4.7× on NAVSIM.

• We construct a closed-loop benchmark, Navhard-CL, built on DreamStream, with behavioral and visual variations. It reveals performance gaps and failure modes of existing driving policies that prior closed-loop benchmarks overlook.

## 2 Related Work

Autoregressive Video Generation for Autonomous Driving. Generating realistic driving observations for interactive evaluation remains an open problem, as closed-loop simulators must render action-conditioned camera views autoregressively while preserving real-world appearance and temporal coherence. Three lines of work address this by predicting future frames with generative world models [20, 21, 22, 14, 23, 24, 25, 26], synthesizing novel views from reconstructed dynamic 3D scenes [27, 8, 9, 10], and re-rendering physics-based simulator state with large pretrained video models [28, 16, 12, 11]. However, existing pipelines are typically trained on small driving datasets such as nuScenes and inherit their narrow visual style, failing to support diverse weather, lighting, and road conditions during closed-loop evaluation [29, 11, 12]. DreamStream mitigates this with diverse-scene distillation in a multi-stage autoregressive pipeline distilled from a large pretrained video foundation model [30], enabling weather and lighting variations for evaluations.

End-to-End Driving Policies Evaluation. Open-loop evaluation emerged as a practical proxy for closed-loop simulation, enabling efficient benchmarking of planning policies on large-scale datasets [29, 31, 18, 19, 32, 33], where interactive evaluation is computationally expensive and difficult to standardize [34, 35]. However, because it evaluates policies under fixed, non-reactive observations rather than sequential decision-making, strong open-loop performance may not be indicative of reliable driving behavior in closed-loop deployment [6, 36, 1, 3, 37]. This motivates closed-loop evaluations of current E2E policies that are widely evaluated in an open-loop manner [38, 39, 40, 18, 41]. To this end, DreamStream provides a generative simulator that bridges the sim-to-real visual gap while grounded in closed-loop simulation dynamics, thus enabling fair and comprehensive closed-loop evaluations of E2E driving policies.

## 3 DreamStream Approach

We give an overview of our closed-loop generative simulator DreamStream in Sec. 3.1, and then describe the multi-stage distillation pipeline of the autoregressive video diffusion model in Sec. 3.2.

## 3.1 Overview

DreamStream consists of closed-loop interaction between three components: a driving simulator S, our autoregressive (AR) video model $\mathcal { G } _ { \phi }$ (Sec. 3.2), and an E2E driving policy π given for evaluation. The simulator maintains the underlying scene state, including ego pose, surrounding agents, and an HD map. The video model translates the simulator state into the camera frames that the policy actually consumes. This decoupling lets DreamStream inherit the simulator grounding while providing high visual fidelity. Fig. 2 illustrates the iteration cycle.

Scenario initialization. A rollout begins from a scenario specified by an initial world state $\mathbf { s } _ { 0 }$ together with an initial camera frame $\mathbf { o } _ { 0 }$ . We support two scenario sources: (i) real-world driving logs converted into the simulator [42], where $\mathbf { o } _ { 0 }$ is the corresponding logged frame, and (ii) generated scenarios such as safety-critical variants [43], where $\mathbf { o } _ { 0 }$ is synthesized by an image generation model to match the desired initial conditions.

![](images/55aca9e44ea352826328e985aee87901350807661e23b5d472dc946f324404e9.jpg)  
Figure 2: Overview of the DreamStream framework. At each iteration of closed-loop simulation, it performs autoregressive rollouts with layout from a physics simulator and diverse visual appearance.

Closed-loop iteration. Each iteration i starts from time $t _ { i }$ with the world state $\mathbf { s } _ { t _ { i } }$ and the most recent K camera frames $\begin{array} { r } { \mathbf { O } _ { t _ { i } - K + 1 : t _ { i } } , } \end{array}$ and proceeds in three steps.

(1) Plan. The E2E policy produces a planned trajectory of length H steps, $\tau _ { t _ { i } } = \pi ( \mathbf { o } _ { t _ { i } - K + 1 : t _ { i } } ) =$ $( \mathbf { a } _ { t _ { i } + 1 } , \dots , \mathbf { a } _ { t _ { i } + H } )$

(2) Simulator rollout. The simulator executes only the first $\Delta _ { r } \ \leq \ H$ steps of $\tau _ { t _ { i } }$ , where $\Delta _ { r }$ is a configurable replan period that controls how often the policy is re-queried: $\mathbf { s } _ { t _ { i } + k } ~ =$ $\begin{array} { r } { \boldsymbol { S } ( \mathbf { s } _ { t _ { i } + k - 1 } , \mathbf { a } _ { t _ { i } + k } ) , \boldsymbol { k } = 1 , \ldots , \Delta _ { r } } \end{array}$ . For each new state, the simulator renders a traffic-layout condition $\mathbf { c } _ { t _ { i } + k }$ , which contains the perspective projection of the high-definition (HD) map (lane geometry and traffic elements) together with 3D bounding boxes of traffic agents (vehicles and pedestrians).

(3) Observation generation. The video model autoregressively generates the next $\Delta _ { r }$ camera frames conditioned on the rendered layouts and a text prompt p describing the scene: $\begin{array} { r } { \mathbf { O } _ { t _ { i } + 1 : t _ { i } + \Delta _ { r } } = } \end{array}$ $\mathcal { G } _ { \phi } \left( \mathbf { c } _ { t _ { i } + 1 : t _ { i } + \Delta _ { r } } , \mathbf { p } \ \middle | \ K _ { i } \right)$ , where $\kappa _ { i }$ is the video model’s KV cache that grows with each iteration’s generated frames. $\kappa _ { 0 }$ is seeded with the initial frame $\mathbf { o } _ { 0 }$ . The policy then takes the most recent K camera frames as its next input, and the loop continues from $t _ { i + 1 } = t _ { i } + \Delta _ { r }$

Closed-loop scoring. During rollout, surrounding agents are controlled via log-replay, intelligent driver model (IDM), or adversarial modes. After the scenario terminates, the full executed trajectory {s<sub>t</sub>} is scored with closed-loop metrics.

## 3.2 Autoregressive Video Diffusion Model Distillation

As training a few-step autoregressive video diffusion model from scratch is difficult, we follow common practice and adopt a three-stage distillation recipe [44, 45, 46]: Stage-1 builds a controllable conditional video generator; Stage-2 turns it into a few-step autoregressive model for efficient rollout; Stage-3 finally improves its long-horizon rollout stability.

Starting from a pretrained video model, we define the input as $c = ( \mathbf { c } , \mathbf { o } _ { 0 } )$ , where c denotes trafficlayout conditions and $\mathbf { o } _ { 0 }$ is the first-frame anchor, and train $G _ { \mathrm { T } }$ to generate future video $\hat { \mathbf { o } } _ { 1 : F }$ that follows c while preserving realistic appearance and dynamics. Stage-1 teaches the model to generate the future video from explicit conditions, so controllability is learned before causal distillation.

Stage-2 and Stage-3 share the same goal: obtaining a stable few-step causal AR model $G _ { \theta }$ for long rollout. Stage-2 performs teacher-to-student distillation with ground-truth context. It uses $( \mathbf { x } _ { t } ^ { \mathrm { { T } } } , c , t )$ as input and $\mathbf { x } _ { \mathrm { 0 } }$ as target, where $c = ( \mathbf { c } , \mathbf { o } _ { 0 } )$ , which gives the causal student a strong initialization for chunk-level AR prediction while inheriting the teacher’s generation quality. Stage-3 then addresses the remaining train-test gap by training on self-generated context (Self Forcing) rather than groundtruth history. In this stage, the rollout is conditioned on $c ,$ the AR clean output is $\hat { \mathbf { x } } _ { 0 } ^ { 1 : F } = \mathcal { R } _ { \theta } ( c )$

and training is applied on its noised version $\hat { \mathbf { x } } _ { t }$ , so the model is explicitly optimized under its own rollout distribution. Detailed formulations are provided in the Appendix Sec. B.1.

Alongside the multi-stage distillation pipeline, we further introduce two key components to enhance the distilled autoregressive video diffusion model to preserve policy-oriented fidelity.

Traffic-guided distillation. In closed-loop simulation, the policy plans based on world-modelgenerated camera frames, so misalignment in lanes or agents can change decisions even when the simulator state is correct. The distilled model must therefore reliably follow the traffic layout in c. Following classifier-free guidance [47], extrapolating between conditional and unconditional estimates steers denoising toward greater satisfaction of the conditioning. Thus, it mimics a classifier gradient without training a separate classifier. During Stage-1 training, we replace the layout in c with $c _ { \emptyset }$ with probability $p { = } 0 . 1$ where $c _ { \emptyset }$ removes traffic layout but keeps $\mathrm { C L I P } ( \mathbf { o } _ { 0 } )$ . When generating videos with $G _ { \mathrm { T } }$ , we combine predictions under c and c<sub>∅</sub> with guidance scale w,

$$
\hat { G } _ { \mathrm { T } } ( \mathbf { x } _ { t } , c , t ) = G _ { \mathrm { T } } ( \mathbf { x } _ { t } , c _ { \emptyset } , t ) + w \cdot \big ( G _ { \mathrm { T } } ( \mathbf { x } _ { t } , c , t ) - G _ { \mathrm { T } } ( \mathbf { x } _ { t } , c _ { \emptyset } , t ) \big ) .\tag{1}
$$

Without additional data, this guidance improves traffic layout alignment in $G _ { \mathrm { T } }$ distillation pipeline.

Diverse-scene distillation. Pretrained video models already learned to generate diverse weather and lighting conditions. However, our distillation data comes from driving logs with limited coverage, and most training scenes share similar weather and lighting. When we distill the teacher into the causal student model, the autoregressive student model gradually loses the ability to render adverse or visually diverse conditions that were present in the pretrained backbone but absent from the driving dataset.

To obtain visually diverse distillation data without new driving logs, we build synthetic clips from existing driving scenarios detailed in Sec. B.3. Training Stage-2 and 3 on these clips enables Dream Stream to roll out under conditions outside the narrow visual domain of small-scale driving data.

## 4 Policy-oriented World Model Evaluation

Existing world-model evaluation metrics fail to capture whether a model preserves the information downstream policies rely on for decisions. Perceptual metrics such as FID [13] were designed for visual quality rather than whether the generated world supports downstream autonomy. They saturate against a single feature space and are biased on small in-domain driving datasets [16, 15, 17]. Controllability evaluations, such as 3D detection or map segmentation accuracy [14, 11, 12, 15], measure task-specific alignment from a modular rather than an E2E perspective. Therefore, we need a direct and unified visual alignment metric from the E2E policies’ perspective.

## 4.1 Metric Design

We design FDπ that quantifies the sim-to-real visual gap for closed-loop policy evaluation from the policy’s perspective. For a driving policy, we extract the scene-context features it uses to generate actions from real camera frames and world-model-rendered frames of the same scenes, and compute the Frechet distance between the resulting feature distributions. This captures how much the world´ model perturbs the visual information the policy uses to act. Different E2E policies attend to various aspects (appearance, geometry, critical objects, traffic semantics) of a driving scene differently; we therefore aggregate the measurement across a panel of public E2E policies [39, 41, 48, 38, 49].

Formulation. Let $\Pi = \{ \pi _ { 1 } , \ldots , \pi _ { N } \}$ denote a panel of E2E driving policies, T a validation scene token set, and v the world model under evaluation. For each token $t \in \tau$ , we extract the scene context feature of the original camera frame as $\Phi _ { \pi _ { k } } ( t ) \in \mathbb { R } ^ { D _ { \pi _ { k } } }$ , with $D _ { \pi _ { k } }$ policy-dependent. We also extract scene context feature $\tilde { \Phi } _ { \pi _ { k } } ^ { ( v ) } ( t )$ of the generated frame. Modeling $\Phi _ { \pi _ { k } }$ and $\tilde { \Phi } _ { \pi _ { k } } ^ { ( v ) }$ as samples from multivariate Gaussians with moments $\left( \mu _ { \pi _ { k } } , \Sigma _ { \pi _ { k } } \right)$ and $( \tilde { \mu } _ { \pi _ { k } } ^ { ( v ) } , \tilde { \Sigma } _ { \pi _ { k } } ^ { ( v ) } )$ , we compute the Frechet´ distance

$$
\widetilde \mathrm { F D } \pi _ { k } ( v ) \ = \ \lVert \mu _ { \pi _ { k } } - \tilde { \mu } _ { \pi _ { k } } ^ { ( v ) } \rVert _ { 2 } ^ { 2 } \ + \ \mathrm { T r } \Big ( \Sigma _ { \pi _ { k } } + \tilde { \Sigma } _ { \pi _ { k } } ^ { ( v ) } - 2 \big ( \Sigma _ { \pi _ { k } } \tilde { \Sigma } _ { \pi _ { k } } ^ { ( v ) } \big ) ^ { 1 / 2 } \Big ) \ .\tag{2}
$$

<table><tr><td rowspan="2">Method</td><td rowspan="2">FDπ↓</td><td colspan="4"></td><td rowspan="2">SDv2</td><td rowspan="2">FID↓</td><td rowspan="2">FVD↓</td></tr><tr><td>DrivoR</td><td>DD</td><td>FDπk↓ LTF</td><td>RAP</td></tr><tr><td colspan="9">nuScenes val</td></tr><tr><td>MagicDrive [14]</td><td>17.18</td><td>11.76</td><td>31.74</td><td>4.57</td><td>32.00</td><td>5.84</td><td>16.20</td><td>218.12</td></tr><tr><td>Panacea [24]</td><td>31.46</td><td>19.68</td><td>57.44</td><td>8.47</td><td>54.81</td><td>16.89</td><td>16.96</td><td>139.00</td></tr><tr><td>Dreamland [16]</td><td>25.28</td><td>30.69</td><td>30.69</td><td>7.40</td><td>24.18</td><td>33.46</td><td>47.93</td><td>670.90</td></tr><tr><td>DriveArena* [11]</td><td>15.68</td><td>9.95</td><td>27.55</td><td>4.68</td><td>25.66</td><td>10.54</td><td>34.74</td><td>665.17</td></tr><tr><td>DreamForge* [12]</td><td>18.29</td><td>11.71</td><td>31.91</td><td>4.09</td><td>36.70</td><td>7.04</td><td>14.61</td><td>209.90</td></tr><tr><td>HUGSIM* [8]</td><td>11.68</td><td>12.39</td><td>23.76</td><td>5.79</td><td>8.25</td><td>8.21</td><td>27.95</td><td>147.18</td></tr><tr><td>DreamStream</td><td>7.27</td><td>9.42</td><td>8.31</td><td>2.25</td><td>9.24</td><td>7.12</td><td>19.58</td><td>272.22</td></tr><tr><td colspan="9">NAVSIM navtest</td></tr><tr><td>BridgeSim [3]</td><td>56.13</td><td>57.04</td><td>77.00</td><td>22.15</td><td>60.92</td><td>63.58</td><td>175.53</td><td></td></tr><tr><td>DriveArena [11]</td><td>25.45</td><td>25.18</td><td>38.96</td><td>10.17</td><td>30.78</td><td>22.16</td><td>41.80</td><td></td></tr><tr><td>DreamStream</td><td>5.47</td><td>5.21</td><td>9.69</td><td>2.65</td><td>5.05</td><td>4.74</td><td>11.78</td><td></td></tr></table>

![](images/43da05ef60ce3e0219cc871e437f3160ef20404d660ff635943656aa029b7a04.jpg)  
Figure 3: Correlation of FDπ, FID, and FVD with OL PDMS.

Table 1: FDπ and FDπ between original and generated/rendered frames. Bold, underline, and wavy marks first, second, third place, respectively. <sup>∗</sup> trained on evaluation set.  
![](images/2515d1416047b3ada81a759a6101dd77ec3dd1bdab9f34e6954b420f7ebb8fb0.jpg)  
Figure 4: Qualitative comparison of our DreamStream and baselines on nuScenes val scenarios.

This measures the shift in the visual representation that policy $\pi _ { k }$ actually consumes. The raw $\widetilde { \mathrm { F D } \pi _ { k } } ( v )$ scales with the feature norm of policy $\pi _ { k } .$ , which varies across model architectures. To make it comparable across policies, we normalize it with $\mathrm { F D } \pi _ { k } ( v ) = \widetilde { \mathrm { F D } } \pi _ { k } ( v ) / \mathrm { T r } ( \Sigma _ { \pi _ { k } } )$ , and report FDπ by averaging across the policy panel:

$$
\mathrm { F D } \pi ( v ) = \frac { 1 } { | \Pi | } \sum _ { \pi _ { k } \in \Pi } \mathrm { F D } \pi _ { k } ( v ) .\tag{3}
$$

Lower FDπ indicates that the generated frames preserve more of the visual information that downstream policies rely on, with 0 marking perfect alignment between real and rendered representations. Following existing distributional distance metrics [50, 51], we report FDπ $\times 1 0 ^ { 2 }$ for readability. As demonstrated in Fig. 3, FDπ correlates substantially more strongly with driving performance than FID and FVD, suggesting that it better captures the visual cues relevant to policy behavior.

## 4.2 Results

We compare our DreamStream against prior works on nuScenes val [29] and NAVSIM navtest [18], including MagicDrive [14], Panacea [24], Dreamland [16], DriveArena [11], DreamForge [12], and HUGSIM [8]. The policy panel comprises five distinct E2E architectures: DrivoR [39], DiffusionDrive/DiffusionDriveV2 (DD/DDv2) [41, 52], LTF [48], RAP [38], and SparseDriveV2 (SDv2) [49]. Results are in Tab. 1; FID and FVD are reported alongside for reference.

On nuScenes val, DreamStream achieves FDπ of 7.27, which is 1.6× lower than the strongest baseline. The advantage holds on three of the five per-policy columns, with the largest reduction on

![](images/e5db3640454f63fcc7cee7e2237da1b5999218711f9f6358683b4915435a0ab2.jpg)  
Figure 5: Qualitative example of closed-loop policy evaluation using DreamStream.

DiffusionDrive. On RAP, reconstruction-based HUGSIM performs on par with ours, which demonstrates our world model’s ability to preserve the 3D geometry that RAP’s spatial cross-attention relies on. The averaged FDπ across diverse E2E policies unifies different architectures, and our model consistently outperforms previous baselines. On NAVSIM navtest, our model achieves the highest performance consistently. DriveArena’s FDπ increases 1.6× than its nuScenes value, indicating its wider visual gap as the evaluation domain spans. Qualitative comparison in Fig. 4 furthe confirms the strong visual alignment of DreamStream compared with baseline methods.

## 4.3 Metric Ablation

Correlation. To further verify FDπ correlation, we compute correlation in a held-out manner: for each policy π, we correlate the FD computed excluding π with π’s PDMS, so the features and the behavior come from different networks. Held-out FDπ remains strongly correlated $( r = - 0 . 6 5 )$ , whereas FID averages $r ~ = ~ + 0 . 1 1$ and FVD $r ~ = ~ + 0 . 0 8$ , indicating that FDπ captures policy independent corruption of scene information.

Sensitivity. We evaluate the sensitivity of FDπ to challenging scenes and safety-critical local errors. When applying it to Navhard subsets of navtest with 540 challenging scenes, FDπ increases for 20% more compared with a randomly sampled subset, which corresponds to the larger visual gap. FID on Navhard subsets only captures average representation shift. We further inject local corruptions using [53] on 1,517 nuScenes val samples, which includes agent displacement for 2m and lane removal. Measuring on the corrupted samples, FDπ responds 2.3× more strongly than FID, showing higher sensitivity to policy-relevant local errors.

## 5 Navhard-CL Closed-loop Benchmark

Built on our DreamStream generative simulator, we create Navhard-CL, a closed-loop benchmark for evaluating end-to-end driving policies with realistic and diverse traffic scenarios and visual appearance. Navhard-CL comprises three scenario buckets based on NAVSIM navhard [19]: Navhard-Base from real-world logs, Navhard-AdvBehavior which introduces adversarial agents to manifest safety-critical interactions, and Navhard-AdvWeather that exposes policies to adverse and scarce weather, lighting, and road-surface conditions. We detailed scenario curation in Appendix Sec. C.4.

## 5.1 E2E Policies Performance

We evaluate E2E driving policies on the Navhard-Base scenarios of Navhard-CL, comparing three observation sources: simulator RGB rendering from BridgeSim [3], DriveArena-rendered [11], and DreamStream-rendered in Fig. 5. Tab. 2a reports the closed-loop performance [19, 3], including driving score (DS), extended PDM score (EPDMS), and route completion (RC). We find that better visual alignment in DreamStream yields higher closed-loop scores across evaluated policies, indicating that preserving policy-oriented visual fidelity reduces simulation-induced perturbation to policy behavior and is essential for faithful closed-loop evaluation. Notably, policies score lower under the more photorealistic DriveArena than under the game-engine-rendered BridgeSim, suggesting that photorealism without policy-relevant scene-context preservation can degrade evaluation faithfulness even relative to a simulator with a visible sim-to-real gap.

(a) E2E policy performance on Navhard-Base. (b) Scorer bias & proposal coverage failure.  
Table 2: Closed-loop performance and analysis on Navhard-CL. (a) E2E driving policy performance on Navhard-Base under three observation sources. Sim: simulator RGB; DA: DriveArena; Ours: DreamStream. (b) Driving score (DS) gap due to scorer bias (Oracle−Learned) and proposal coverage (AdvBehavior−Base with oracle scorer) on Navhard-AdvBehavior. <sup>†</sup> has no scoring head.
<table><tr><td>Policy</td><td>Obs.</td><td>DS</td><td>EPDMS</td><td>RC</td></tr><tr><td rowspan="2">DrivoR</td><td>Sim</td><td>42.32</td><td>66.99</td><td>62.39</td></tr><tr><td>DA Ours</td><td>38.92 46.21</td><td>66.16 68.64</td><td>57.74 66.28</td></tr><tr><td rowspan="2">DiffusionDrive</td><td>Sim</td><td>46.19</td><td>61.56</td><td>72.67</td></tr><tr><td>DA Ours</td><td>32.02 59.68</td><td>56.39 72.98</td><td>55.56 81.37</td></tr><tr><td>DiffusionDriveV2</td><td>Sim DA Ours</td><td>45.87 21.48 58.35</td><td>57.60 53.06 67.52</td><td>77.67 38.54 85.48</td></tr><tr><td>LTF</td><td>Sim DA Ours</td><td>40.64 38.27 53.28</td><td>58.60 60.25 67.84</td><td>68.26 63.84 77.93</td></tr></table>

<table><tr><td rowspan="2">Policy</td><td colspan="2">Scorer Bias</td><td rowspan="2">Coverage Gap</td></tr><tr><td>Base</td><td>Adv</td></tr><tr><td>DiffusionDrive</td><td>+3.64</td><td>+8.98</td><td>-12.98</td></tr><tr><td>DrivoR</td><td>+14.17</td><td>+7.19</td><td>-16.66</td></tr><tr><td>LTF†</td><td></td><td></td><td>-14.61</td></tr></table>

(c) DrivoR on a high driving score subset of Navhard-AdvWeather.
<table><tr><td>Condition</td><td>DS</td><td>EPDMS</td><td>RC</td></tr><tr><td>Original</td><td>88.44</td><td>91.86</td><td>96.38</td></tr><tr><td>Rain</td><td>81.70</td><td>86.30</td><td>94.10</td></tr><tr><td>Snow</td><td>81.96</td><td>86.25</td><td>94.50</td></tr><tr><td>Night</td><td>85.92</td><td>90.57</td><td>94.92</td></tr></table>

## 5.2 Closed-loop Gap Analysis

Navhard-CL reveals closed-loop performance gap and failure modes that prior benchmarks overlook. Navhard-AdvBehavior decomposes the performance drop into two distinct observations by analyzing the policy’s scorer and trajectory. Navhard-AdvWeather measures policy brittleness under appearance shifts, such as adverse weather and on-road conditions. We refer to Sec. C.5 for detailed setup, results, and analysis.

Scorer bias and proposal coverage failure (Tab. 2b). We replace the policy’s learned scorer with an oracle scorer that ranks proposals by ground-truth EPDMS calculated in the simulator. The driving score difference reveals two gaps: (i) Scorer bias is already large on Navhard-Base, and widens under Navhard-AdvBehavior for certain architectures. (ii) Proposal coverage fails even when scoring is optimal. The driving score decreases dramatically from Navhard-Base to Navhard AdvBehavior. Decomposing by subscores shows lower TTC, LK, and HC, indicating the policy’s inability to react safely. LTF, a policy with no scoring head, shows that the gap lies in the trajectory decoder. These extend the open-loop scorer-mismatch analysis [54] into the closed-loop regime.

Visual robustness gap under appearance shift (Tab. 2c). For high driving score (≥ 80) scenarios in Navhard-AdvWeather, rain and snow degrade DrivoR’s performance more than night, especially on road-surface related performance (drivable-area compliance, lane keeping). It reveals a visual robustness gap that closed-loop benchmarks on a narrow distribution-matched visual domain cannot expose.

## 6 Ablations

We ablate our key design for enhancing the distilled autoregressive video model, including the traffic-layout guidance and diverse-scene distillation using the Wan 2.1 backbone.

Traffic-guided distillation (TGD) enhances the traffic layout signal in the generated frames, as shown on the left of Fig. 6. This results in better scene layout alignment between the generated frames and simulator states, which benefits the evaluated E2E policies for more accurate decisions. We ablate this design on nuScenes val dataset and observe FDπ improves from 12.18 to 9.50.

Diverse-scene distillation (DSD) is crucial for preserving visual diversity in generated frames. We ablate this design on Navhard by conditioning on an edited first frame, and report the CLIP similarity between the generated frame at rollout step 100 and the weather/lighting text prompt. DSD consistently improves CLIP similarity across diverse weather and lighting conditions. Without DSD, the autoregressive model fails to preserve the weather and lighting in the first frame and quickly reverts to the narrow appearance of the training distribution, as shown on the right of Fig. 6.

Table 3: Ablation on guidance. Our guidance improves layout fidelity over standard CFG.
<table><tr><td>Variant</td><td>DrivoR</td><td>DD LTF</td><td>RAP</td><td>SDv2</td><td>FDπ↓</td></tr><tr><td>w/o TGD</td><td>11.25</td><td>22.48 6.92</td><td>10.87</td><td>9.38</td><td>12.18</td></tr><tr><td>w/TGD</td><td>8.53</td><td>18.92 3.56</td><td>9.65</td><td>6.86</td><td>9.50</td></tr></table>

Table 4: Ablation on diverse-scene distillation, reporting text-image CLIP score.
<table><tr><td>Variant</td><td>Night</td><td>Snow</td><td>Rain</td></tr><tr><td>w/o DSD</td><td>0.2679</td><td>0.2516</td><td>0.2421</td></tr><tr><td>w/ DSD</td><td>0.2796</td><td>0.2722</td><td>0.2475</td></tr></table>

![](images/2452baf4552fe4f352339e39aa1980e874353a8c4847d90658473f3972652ba6.jpg)  
(a) Traffic-guided distillation

![](images/791f9300ba9ce510b0cf3a4e73367ea927584fa4c5a543757448f920947d5823.jpg)

![](images/aa49af32414e8186c9c88fd14cc6fb0d69b3b06992b199d6dc5a071a5ceb0c32.jpg)  
(b) Diverse-scene distillation (night)

Figure 6: Qualitative ablation on traffic-layout guidance and diverse-scene distillation.  
![](images/65a7c10e8d51ddb1dd743b439bf9b1e2b4fe7cc3d32b4c013f128a38b728557d.jpg)  
Figure 7: FDπ over long-horizon rollout.

<table><tr><td>1.3B</td><td>Wan2.1 Wan2.2 5B</td></tr><tr><td>ms/frame</td><td>49.5 59</td></tr><tr><td>Gen. fps 20.2</td><td>16.9</td></tr><tr><td>VRAM</td><td>32 GB 67 GB</td></tr><tr><td>Sim. fps</td><td>10.1 8.8</td></tr></table>

Table 5: DreamStream efficiency.

Long-horizon rollout stability. We quantify autoregressive quality drift by computing FDπ as a function of rollout length in Fig. 7. DreamStream’s drift is bounded and remains below all compared baselines on both nuScenes and NAVSIM. Using early-timestep latent states as the KV cache mitigates drift by 68% on nuScenes and 48% on NAVSIM for longer rollouts.

Efficiency. Tab. 5 reports the runtime of DreamStream on a single RTX PRO 6000 GPU. Our design enables linear multi-GPU scaling, real-time generation, and batch benchmarking. Full model training costs ∼200 A100 GPU-days.

## 7 Limitations

DreamStream bridges the visual gap between simulator and real-world camera observations, but our closed-loop evaluation still runs in simulation instead of on-road real-world evaluation or hardwarein-the-loop deployment given the cost and safety. Besides, the autoregressive video model also accumulates drift over extensive long-horizon rollouts, which we leave as future work.

## 8 Conclusion

In this paper, we presented DreamStream, a generative closed-loop simulator pairing a physics simulator with an autoregressive video model; FDπ, a policy-oriented Frechet distance that better mea-´ sures visual alignment, on which DreamStream improves over the best prior closed-loop simulators by 1.6× on nuScenes and 4.7× on NAVSIM; and Navhard-CL, a closed-loop benchmark built on DreamStream with behavior and visual variations. Our closed-loop benchmark reveals performance gaps and failure modes that prior closed-loop benchmarks miss, and points scorer calibration and proposal coverage as concrete directions for training stronger closed-loop end-to-end driving poli cies.

## Acknowledgments

This work was supported by NSF grants CNS-2235012 and IIS-2339769, and Toyota Research Institute. Seth Z. Zhao was supported by Qualcomm Innovation Fellowship. Sicheng Mo was supported by Amazon AI PhD Fellowship through the Science Hub for Humanity and Artificial Intelligence.

## References

[1] P. Karkus, M. Igl, Y. Chen, K. Chitta, J. Packer, B. Douillard, R. Tian, A. Naumann, G. Garcia-Cobo, S. Tan, et al. Beyond behavior cloning in autonomous driving: a survey of closed-loop training techniques. Authorea Preprints.

[2] Y. Wang, A. Jiang, S. Wang, Y. Heng, H. Yang, Y. Chen, and H. Sun. Do open-loop metrics predict closed-loop driving? a cross-benchmark correlation study of NAVSIM and Bench2Drive. arXiv preprint arXiv:2605.00066, 2026.

[3] S. Z. Zhao, L. Wang, H. Ruan, Y. Bao, Y. Chen, Z. Leng, A. Ravichandran, H. He, Z. Zhou, X. Han, et al. BridgeSim: Unveiling the OL-CL gap in end-to-end autonomous driving. arXiv preprint arXiv:2604.10856, 2026.

[4] Q. Li, Z. Peng, L. Feng, Q. Zhang, Z. Xue, and B. Zhou. MetaDrive: Composing diverse driving scenarios for generalizable reinforcement learning. TPAMI, 2022.

[5] S. Kazemkhani, A. Pandya, D. Cornelisse, B. Shacklett, and E. Vinitsky. GPUDrive: Datadriven, multi-agent driving simulation at 1 million FPS. arXiv preprint arXiv:2408.01584, 2024.

[6] X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan. Bench2drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving. In NeurIPS 2024 Datasets and Benchmarks Track, 2024.

[7] S. Gerstenecker, A. Geiger, and K. Renz. Fail2drive: Benchmarking closed-loop driving generalization. arXiv preprint arXiv:2604.08535, 2026.

[8] H. Zhou, L. Lin, J. Wang, Y. Lu, D. Bai, B. Liu, Y. Wang, A. Geiger, and Y. Liao. HUGSIM: A real-time, photo-realistic and closed-loop simulator for autonomous driving. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[9] C. Ni, G. Zhao, X. Wang, Z. Zhu, W. Qin, G. Huang, C. Liu, Y. Chen, Y. Wang, X. Zhang, et al. Recondreamer: Crafting world models for driving scene reconstruction via online restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1559–1569, 2025.

[10] G. Zhao, C. Ni, X. Wang, Z. Zhu, X. Zhang, Y. Wang, G. Huang, X. Chen, B. Wang, Y. Zhang, et al. DriveDreamer4D: World models are effective data machines for 4D driving scene representation. In Proceedings of the computer vision and pattern recognition conference, pages 12015–12026, 2025.

[11] X. Yang, L. Wen, T. Wei, Y. Ma, J. Mei, X. Li, W. Lei, D. Fu, P. Cai, M. Dou, et al. Drivearena: A closed-loop generative simulation platform for autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 26933–26943, 2025.

[12] J. Mei, T. Hu, X. Yang, L. Wen, Y. Yang, T. Wei, Y. Ma, M. Dou, B. Shi, and Y. Liu. Dream-Forge: Motion-aware autoregressive video generation for multi-view driving scenes. arXiv preprint arXiv:2409.04003, 2024.

[13] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in neural information processing systems, 30, 2017.

[14] R. Gao, K. Chen, E. Xie, L. Hong, Z. Li, D.-Y. Yeung, and Q. Xu. Magicdrive: Street view generation with diverse 3D geometry control, 2024. URL https://arxiv.org/abs/2310. 02601.

[15] A. Liang, L. Kong, T. Yan, H. Liu, W. Yang, Z. Huang, W. Yin, J. Zuo, Y. Hu, D. Zhu, et al. WorldLens: Full-spectrum evaluations of driving world models in real world. arXiv preprint arXiv:2512.10958, 2025.

[16] S. Mo, Z. Leng, L. Liu, W. Wang, H. He, and B. Zhou. Dreamland: Controllable world creation with simulator and generative models. arXiv preprint arXiv:2506.08006, 2025.

[17] J. Yang, Z. Geng, X. Ju, Y. Tian, and Y. Wang. Representation Frechet loss for visual genera-´ tion. arXiv preprint arXiv:2604.28190, 2026.

[18] D. Dauner, M. Hallgarten, T. Li, X. Weng, Z. Huang, Z. Yang, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta. NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[19] W. Cao, M. Hallgarten, T. Li, D. Dauner, X. Gu, C. Wang, Y. Miron, M. Aiello, H. Li, I. Gilitschenski, B. Ivanovic, M. Pavone, A. Geiger, and K. Chitta. Pseudo-simulation for autonomous driving. In Conference on Robot Learning (CoRL), 2025.

[20] S. W. Kim, J. Philion, A. Torralba, and S. Fidler. DriveGAN: Towards a controllable highquality neural simulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5820–5829, 2021.

[21] A. Hu, L. Russell, H. Yeo, Z. Murez, G. Fedoseev, A. Kendall, J. Shotton, and G. Corrado. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

[22] S. Gao, J. Yang, L. Chen, K. Chitta, Y. Qiu, A. Geiger, J. Zhang, and H. Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. Advances in Neural Information Processing Systems, 37:91560–91596, 2024.

[23] R. Gao, K. Chen, B. Xiao, L. Hong, Z. Li, and Q. Xu. MagicDriveDiT: High-resolution long video generation for autonomous driving with adaptive control. arXiv preprint arXiv:2411.13807, 2024.

[24] Y. Wen, Y. Zhao, Y. Liu, F. Jia, Y. Wang, C. Luo, C. Zhang, T. Wang, X. Sun, and X. Zhang. Panacea: Panoramic and controllable video generation for autonomous driving. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6902–6912, 2024.

[25] Y. Lu, X. Ren, J. Yang, T. Shen, Z. Wu, J. Gao, Y. Wang, S. Chen, M. Chen, S. Fidler, et al. InfiniCube: Unbounded and controllable dynamic 3D driving scene generation with worldguided video models. arXiv preprint arXiv:2412.03934, 2024.

[26] W. Zheng, R. Song, X. Guo, C. Zhang, and L. Chen. GenAD: Generative end-to-end autonomous driving. In European Conference on Computer Vision, pages 87–104. Springer, 2024.

[27] J. Yang, B. Ivanovic, O. Litany, X. Weng, S. W. Kim, B. Li, T. Che, D. Xu, S. Fidler, M. Pavone, et al. EmerNeRF: Emergent spatial-temporal scene decomposition via self-supervision. In International Conference on Learning Representations, volume 2024, pages 16739–16766, 2024.

[28] Y. Zhou, M. Simon, Z. Peng, S. Mo, H. Zhu, M. Guo, and B. Zhou. SimGen: Simulatorconditioned driving scene generation. Advances in Neural Information Processing Systems, 37:48838–48874, 2024.

[29] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom. nuScenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11621–11631, 2020.

[30] T. Wan, A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[31] H. Caesar, J. Kabzan, K. S. Tan, W. K. Fong, E. Wolff, A. Lang, L. Fletcher, O. Beijbom, and S. Omari. nuPlan: A closed-loop ml-based planning benchmark for autonomous vehicles. arXiv preprint arXiv:2106.11810, 2021.

[32] Y. Li, S. Z. Zhao, C. Xu, C. Tang, C. Li, M. Ding, M. Tomizuka, and W. Zhan. Pre-training on synthetic driving data for trajectory prediction. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5910–5917, 2024. doi:10.1109/IROS58592. 2024.10802492.

[33] S. Z. Zhao, H. Zhang, Z. Li, J. Peng, A. Chui, Z. Zhou, Z. Meng, H. Xiang, Z. Huang, F. Wang, et al. QuantV2X: A fully quantized multi-agent system for cooperative perception. arXiv preprint arXiv:2509.03704, 2025.

[34] D. Dolgov, S. Thrun, M. Montemerlo, and J. Diebel. Practical search techniques in path planning for autonomous driving. ann arbor, 1001(48105):18–80, 2008.

[35] L. Claussmann, M. Revilloud, D. Gruyer, and S. Glaser. A review of motion planning for highway autonomous driving. IEEE Transactions on Intelligent Transportation Systems, 21 (5):1826–1848, 2019.

[36] Z. Li, Z. Yu, S. Lan, J. Li, J. Kautz, T. Lu, and J. M. Alvarez. Is ego status all you need for open-loop end-to-end autonomous driving? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14864–14873, 2024.

[37] M. Coscoy, Z. Zhou, S. Z. Zhao, H. Wei, A. Magtoto, J. Liu, R. Song, W. Zimmer, Z. Huang, C. Tang, B. Zhou, and J. Ma. Mdrive: Benchmarking closed-loop cooperative driving for end-to-end multi-agent systems. arXiv preprint arXiv:2605.10904, 2026.

[38] L. Feng, Y. Gao, E. Zablocki, Q. Li, W. Li, S. Liu, M. Cord, and A. Alahi. RAP: 3D rasterization augmented end-to-end planning, 2025. URL https://arxiv.org/abs/2510.04333.

[39] E. Kirby, A. Boulch, Y. Xu, Y. Yin, G. Puy, E. Zablocki, A. Bursuc, S. Gidaris, R. Marlet, F. Bartoccioni, A.-Q. Cao, N. Samet, T.-H. Vu, and M. Cord. Driving on registers. preprint, 2026.

[40] Z. Zhou, T. Cai, S. Z. Zhao, Y. Zhang, Z. Huang, B. Zhou, and J. Ma. AutoVLA: A visionlanguage-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. Advances in Neural Information Processing Systems (NeurIPS), 2025.

[41] B. Liao, S. Chen, H. Yin, B. Jiang, C. Wang, S. Yan, X. Zhang, X. Li, Y. Zhang, Q. Zhang, and X. Wang. DiffusionDrive: Truncated diffusion model for end-to-end autonomous driving. pages 12037–12047, 2025. doi:10.1109/CVPR52734.2025.01124. URL https: //openaccess.thecvf.com/content/CVPR2025/html/Liao\_DiffusionDrive\_ Truncated\_Diffusion\_Model\_for\_End-to-End\_Autonomous\_Driving\_CVPR\_2025\_ paper.html.

[42] Q. Li, Z. Peng, L. Feng, Z. Liu, C. Duan, W. Mo, and B. Zhou. ScenarioNet: Open-source platform for large-scale traffic scenario simulation and modeling. Advances in Neural Information Processing Systems, 2023.

[43] Y. Liu, Z. Peng, X. Cui, and B. Zhou. Adv-BMT: Bidirectional motion transformer for safetycritical traffic scenario generation, 2025. URL https://arxiv.org/abs/2506.09485.

[44] X. Huang, Z. Li, G. He, M. Zhou, and E. Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems, 38: 167283–167308, 2026.

[45] J. Shin, Z. Li, R. Zhang, J.-Y. Zhu, J. Park, E. Shechtman, and X. Huang. MotionStream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266, 2025.

[46] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, K. Mo, C.-H. Lin, et al. DreamDojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026.

[47] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[48] K. Chitta, A. Prakash, B. Jaeger, Z. Yu, K. Renz, and A. Geiger. TransFuser: Imitation with transformer-based sensor fusion for autonomous driving. Pattern Analysis and Machine Intelligence (PAMI), 2023.

[49] W. Sun, X. Lin, K. Chen, Z. Pei, X. Li, Y. Shi, and S. Zheng. SparseDriveV2: Scoring is all you need for end-to-end autonomous driving. arXiv preprint arXiv:2603.29163, 2026.

[50] M. Binkowski, D. J. Sutherland, M. Arbel, and A. Gretton. Demystifying MMD GANs. In ´ International Conference on Learning Representations, 2018.

[51] S. Jayasumana, S. Ramalingam, A. Veit, D. Glasner, A. Chakrabarti, and S. Kumar. Rethinking FID: Towards a better evaluation metric for image generation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9307–9315, 2024.

[52] J. Zou, S. Chen, B. Liao, Z. Zheng, Y. Song, L. Zhang, Q. Zhang, W. Liu, and X. Wang. DiffusionDriveV2: Reinforcement learning-constrained truncated diffusion modeling in endto-end autonomous driving. arXiv preprint arXiv:2512.07745, 2025.

[53] J. Zhao et al. Precise object and effect removal with adaptive target-aware attention. In CVPR, pages 19370–19379, 2026.

[54] S. Ang, Y. Yang, C. Chen, and Y. Wang. CLOVER: Closed-loop value estimation and ranking for end-to-end autonomous driving planning. arXiv preprint arXiv:2605.15120, 2026.

[55] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun. CARLA: An open urban driving simulator. In Conference on robot learning, pages 1–16. PMLR, 2017.

[56] M. Martinez, C. Sitawarin, K. Finch, L. Meincke, A. Yablonski, and A. Kornhauser. Beyond grand theft auto v for training, testing and enhancing deep learning in self driving cars, 2017. URL https://arxiv.org/abs/1712.01397.

[57] M. Muller, V. Casser, J. Lahoud, N. Smith, and B. Ghanem. Sim4cv: A photo-realistic simu-¨ lator for computer vision applications. IJCV, 2018.

[58] S. Shah, D. Dey, C. Lovett, and A. Kapoor. Airsim: High-fidelity visual and physical simulation for autonomous vehicles. In FSR, 2018

[59] D. Team. Deepdrive: a simulator that allows anyone with a pc to push the state-of-the-art in self-driving. https://github.com/deepdrive/deepdrive.

[60] P. Kothari, C. Perone, L. Bergamini, A. Alahi, and P. Ondruska. Drivergym: Democratising reinforcement learning for autonomous driving. arXiv preprint arXiv:2111.06889, 2021.

[61] C. Gulino, J. Fu, W. Luo, G. Tucker, E. Bronstein, Y. Lu, J. Harb, X. Pan, Y. Wang, X. Chen, et al. Waymax: An accelerated, data-driven simulator for large-scale autonomous driving research. NeurIPS, 2024.

[62] H. Gao, S. Chen, B. Jiang, B. Liao, Y. Shi, X. Guo, Y. Pu, H. Yin, X. Li, X. Zhang, et al. Rad: Training an end-to-end driving policy via large-scale 3dgs-based reinforcement learning. arXiv preprint arXiv:2502.13144, 2025.

[63] C. Ni, G. Zhao, X. Wang, Z. Zhu, W. Qin, X. Chen, G. Jia, G. Huang, and W. Mei. Recondreamer-rl: Enhancing reinforcement learning via diffusion-based scene reconstruction. arXiv preprint arXiv:2508.08170, 2025.

[64] A. Fortin, G. Vernade, K. Kampf, and A. Reshi. Introducing gemini 2.5 flash image, our stateof-the-art image model. Google Developers Blog, Aug. 2025. URL https://developers. googleblog.com/en/introducing-gemini-2-5-flash-image/. Accessed: 2026-06- 03.

[65] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[66] W. G. Najm, J. D. Smith, and M. Yanagisawa. Pre-crash scenario typology for crash avoidance research. Technical Report DOT HS 810 767, National Highway Traffic Safety Administration, 2007.

[67] E. D. Swanson, F. Foderaro, M. Yanagisawa, W. G. Najm, and P. Azeredo. Statistics of lightvehicle pre-crash scenarios based on 2011–2015 national crash data. Technical Report DOT HS 812 745, National Highway Traffic Safety Administration, 2019.

## A More Related Work

Driving Simulators and Closed-loop Benchmarks. Closed-loop (CL) simulators for E2E driving differ mainly in how they render the observations for policy. Game-engine simulators [55, 56, 57, 58, 59, 60, 4, 5, 61] and the CL benchmarks built on them [6, 7, 3] offer full control over layout and reactive, adversarial behavior, but render using hand-built 3D assets and thus exhibit a sim-to-real visual gap that can corrupt a vision-based policy’s perception. Two recent lines of work narrow this gap, but each exhibits its own limitations. Reconstruction-based renderers [27, 8, 62, 19, 9, 63, 10] lift captured driving logs into 3D for photo-realistic rollout, but are limited to pre-captured scenarios with fixed appearance, and exhibit rendering artifacts under off-trajectory viewpoints and dynamicobject insertion. Generative re-rendering of simulator state [11, 12, 28, 16, 14, 24, 26, 22, 21, 25] produces photo-realistic, controllable rollouts but is typically trained on small in-domain datasets such as nuScenes [29], inheriting a narrow visual style with weak 3D grounding. DreamStream is a hybrid approach that keeps a physics simulator for grounded, controllable, reactive state and distills an autoregressive video model from a large pretrained video model to render diverse, photo-realistic observations. It uniquely combines photorealism and visual diversity with grounded, reactive CL simulation, as shown in Tab. 6.

Table 6: Comparison of DreamStream with representative open-loop and closed-loop benchmarks. Photoreal.: renders photo-realistic, low sim-to-real-gap observations. Visual Div.: can render diverse visual appearance. Adv.: supports reactive adversarial agents.
<table><tr><td>Name</td><td>Render Domain</td><td>Closed-loop</td><td>Long-horizon</td><td>Photoreal.</td><td>Visual Div.</td><td>Adv.</td></tr><tr><td>NAVSIM [18, 19]</td><td>Real-world</td><td>×</td><td>X</td><td>√</td><td>×</td><td>X</td></tr><tr><td>Bench2Drive [6]</td><td>Game-engine</td><td>√</td><td>√</td><td>×</td><td>√</td><td>X</td></tr><tr><td>Fail2Drive [7]</td><td>Game-engine</td><td>√</td><td>√</td><td>X</td><td>X</td><td>√</td></tr><tr><td>BridgeSim [3]</td><td>Game-engine</td><td>√</td><td>√</td><td>×</td><td>X</td><td>√</td></tr><tr><td>HUGSIM [8]</td><td>3DGS Recon.</td><td>√</td><td>X</td><td>√</td><td>X</td><td>√</td></tr><tr><td>DriveArena [11]</td><td>World Model</td><td>√</td><td>√</td><td>√</td><td>X</td><td>×</td></tr><tr><td>DreamStream</td><td>World Model</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## B Methodology Details

We provide the training details of our autoregressive video model in this section. Sec. B.1 presents the three-stage distillation pipeline and its training objectives. Sec. B.2 reports the training hyperparameters and related implementation settings for each stage.

## B.1 Autoregressive Video Model Distillation Details

Stage-1: Bidirectional Model. The goal of this stage is to train a stronger teacher model that can follow the scene layout to generate coherent and continuous video frames. Following the latent video diffusion framework, we operate on VAE latents ${ \bf x } _ { 0 } \in \mathbb { R } ^ { F \times H \times W \times C }$ , where F is the number of latent frames, H is the height, W is the width, and C is the number of channels. Building upon the flow-matching framework, we sample Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ and form noised latents $\mathbf { x } _ { t }$ by interpolating between $\mathbf { x } _ { \mathrm { 0 } }$ and ϵ over flow time $t \in [ 0 , 1 ]$ . We train a bidirectional generator $G _ { \mathrm { T } }$ with the following loss:

$$
\mathcal { L } _ { \mathrm { f l o w } } = \mathbb { E } \left[ \| G _ { \mathrm { T } } ( \mathbf { x } _ { t } , c , t ) - ( \epsilon - \mathbf { x } _ { 0 } ) \| _ { 2 } ^ { 2 } \right] ,\tag{4}
$$

where the condition $c = ( \mathbf { c } , \mathbf { C L I P } ( \mathbf { o } _ { 0 } ) )$ combines the scene layout c with the CLIP embedding of the initial camera frame $\mathbf { o } _ { 0 } .$ , and t is the flow-matching time step. After training, this model can follow the conditions and iteratively denoise random noise into visually realistic video frames.

Stage-2: Causal ODE init. The goal of this stage is to distill the Stage-1 bidirectional teacher $G _ { \mathrm { T } }$ into a few-step causal student $G _ { \theta }$ that supports chunk-level autoregressive prediction: each chunk contains one or more latent frames with bidirectional attention inside the chunk and causal attention across chunks [44]. Under the diffusion forcing layout, each frame is denoised with an independently noised causal context while the current frame stays noisy. Following Self Forcing [44], for each x we simulate the teacher reverse-time PF-ODE with $G _ { \mathrm { T } }$ to obtain $\{ \bar { \mathbf { x } } _ { \tau } ^ { \mathrm { T } } \} _ { \tau \in \mathcal { T } }$ , sample $t \in \tau$ and $\mathbf { x } _ { t } ^ { \mathrm { { T } } }$ and optimize the ODE regression objective

$$
\mathcal { L } _ { \mathrm { o d e } } = \mathbb { E } _ { { \mathbf { x } } _ { 0 } , t } \left[ \left\| G _ { \theta } ( { \mathbf { x } } _ { t } ^ { \mathrm { T } } , c , t ) - { \mathbf { x } } _ { 0 } \right\| _ { 2 } ^ { 2 } \right] ,\tag{5}
$$

which distills $G _ { \mathrm { T } }$ into the few-step causal student while retaining diffusion-forcing context noise during training.

Stage-3: Self Forcing. During inference, the autoregressive model must condition each step on its own previously generated latents, but Stage-2 trains the student with ground-truth context, creating a train-test mismatch. We close this gap with Self Forcing [44]: during training we unroll the causal student with KV caching, denoising each chunk from previously self-generated latents rather than ground truth history, and optimizing a holistic distribution-matching objective on the rollout video. Let $\hat { \mathbf { x } } _ { 0 } ^ { 1 : F } = \mathcal { R } _ { \theta } ( \overset { \cdot } { c } , \mathcal { K } )$ denote the clean latents from autoregressively rolling out $G _ { \theta }$ under c. Following the DMD generator objective in Self Forcing [44], we define

$$
\mathcal { L } _ { \mathrm { s f } } = \mathbb { E } _ { t , \epsilon } \left[ \frac { 1 } { 2 } \left. \hat { \mathbf { x } } _ { 0 } - \mathrm { s g } \big ( \hat { \mathbf { x } } _ { 0 } - \left( s _ { \mathrm { f a k e } } \big ( \hat { \mathbf { x } } _ { t } , c , t \big ) - s _ { \mathrm { r e a l } } \big ( \hat { \mathbf { x } } _ { t } , c , t \big ) \big ) \right) \right. _ { 2 } ^ { 2 } \right] ,\tag{6}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient, $\hat { \mathbf { x } } _ { t } = ( 1 - t ) \hat { \mathbf { x } } _ { 0 } + t \epsilon$ for $\epsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) , s _ { \mathrm { r e a l } }$ is a frozen score network initialized from $G _ { \mathrm { T } }$ , and $s _ { \mathrm { f a k e } }$ is a trainable critic on the student’s self-rollout.

We observe that using fully denoised latents as the KV cache leads to rapid quality drift, as highfrequency errors from the final denoising steps compound over rollout and dominate long-horizon degradation. To mitigate this, we condition on early-timestep latents only, which empirically stabilizes rollout.

## B.2 Implementation Details

Our training setups closely follow MotionStream [45]. We use two Wan backbones [30]: Wan 2.1 (1.3B) at $8 3 2 \times 4 8 0$ and Wan 2.2 (5B) at $1 2 8 0 \times 7 0 4 ;$ all other settings are shared. We initialize the model weights from its Wan-Fun-Control checkpoint.

In our training, Stage-1 uses batch size 64 and learning rate $1 \times 1 0 ^ { - 5 }$ with 10K steps; Stage-2 trains for 20K steps with batch size 64 and learning rate $2 \times 1 0 ^ { - 6 }$ ; Stage-3 trains for $1 K$ steps with batch size 32, generator learning rate $2 \times 1 0 ^ { - 6 }$ , and critic learning rate $4 \times 1 0 ^ { - 7 }$ (1:5 updates). For trafficguided distillation, we use a guidance scale of 7.5 at generation time to sample ODE trajectories and provide guidance to the student model.

## B.3 Synthetic Clip Construction for Diverse-Scene Distillation

We detail the recipe for constructing the diverse synthetic clips used in Stages 2–3.

Source scenarios. We uniformly sample 500 driving scenarios from the nuPlan-based training set (Sec. C.1) as layout sources. The traffic layout c (HD map projection and 3D bounding boxes) of each sampled clip is kept untouched, so lanes, agents, and ego motion remain aligned with the original log.

First-frame re-rendering. We re-render only the initial frame $\mathbf { o } _ { 0 }$ of each clip with the Nano-Banana [64], prompted to change the visual appearance while preserving scene geometry. The editing prompts are sampled from the three appearance axes used in Navhard-AdvWeather: lighting conditions, weather conditions, and road-surface conditions. We show the prompt template for changing the weather condition as follows:

This is a driving-camera photo. Edit the weather and lighting only.

Do NOT move, add, remove, resize, or redraw any structure. Every lane line, every building, every billboard, every palm tree, every traffic signal, every electric pole, and every vehicle must stay in exactly the same position with exactly the same shape and the same identity. Preserve the camera angle, framing, perspective, and proportions perfectly.

Change only the season and lighting: make it a heavy snowy winter day. Cover the road shoulders, sidewalks, rooftops, billboard ledges, and palm fronds with snow. Add gentle snowflakes in the air. Replace the bright sunny sky with a flat overcast gray sky and tone down the warm sunlight to cool, diffuse winter daylight.

Output only the edited image, same resolution and same aspect ratio.

Clip generation and filtering. The Stage-1 teacher $G _ { \mathrm { T } }$ rolls out a full video from the edited $\mathbf { o } _ { 0 }$ and the original layout c, producing clips with appearance follows the edited frame while ego motion and surrounding traffic follow the log. We manually review the edited samples to ensure the alignment with the appearance edit instruction, retaining 500 synthetic scenes that are added to the Stage-2 and Stage-3 training sets.

## C Experimental Details

## C.1 Training Dataset

To train and distill our autoregressive world model, we curate the dataset based on nuPlan [31]. It contains more than 20,000 driving scenario videos, each ranging from 15–20 seconds, with a total of around 90 hours of training data. To obtain the traffic-layout condition c, we project the HDMap and 3D bounding boxes annotations into perspective view according to the camera intrinsic and extrinsic. It includes map elements like lane lines, road boundaries, crosswalks, etc., and traffic agents like vehicles and pedestrians, with each object type color-coded. The text prompt p for each video is generated using Qwen3-VL [65] with targeted engineered prompt emphasize factual scene details like objects, geometry, motion, and context.

## C.2 Metric Details

DreamStream adopts open-loop metrics for non-reactive open-loop simulations and closed-loop metrics for closed-loop simulations. For more motivations about metric designs, we refer the reader to [18, 19, 8, 3].

Open-loop metrics: PDMS & EPDMS. The planned trajectory output by E2E policy is scored against the logged future without being executed. The score is implemented as a weighted com bination of hard safety constraints C and driving quality features ${ \mathcal F } ,$ following

$$
\mathrm { P D M S / E P D M S } = \Big ( \prod _ { s c \in \mathcal { C } } \mathbb { I } _ { s c } \Big ) \cdot \frac { \sum _ { f \in \mathcal { F } } w _ { f } v _ { f } } { \sum _ { f \in \mathcal { F } } w _ { f } } \ \in \ [ 0 , 1 ] ,\tag{7}
$$

where $\mathbb { I } _ { s c } \in \{ 0 , 1 \}$ marks safety compliance with constraint sc, $v _ { f } ~ \in ~ [ 0 , 1 ]$ is the value of quality term $f ,$ and $w _ { f }$ its weight. The constraints C comprise NC (No At-Fault Collisions), DAC (Drivable Area Compliance), TLC (Traffic Light Compliance), and DDC (Driving Direction Compliance); the quality features $\mathcal { F }$ comprise EP (Ego Progress), LK (Lane Keeping), TTC (Timeto-Collision), C (Comfort), HC (History Comfort), and EC (Extended Comfort). For Predictive Driver Model Score (PDMS) [18], it uses C = {NC, DAC, DDC} and $\mathcal { F } = \{ \mathrm { E P } , \mathrm { T T C } , \mathrm { C } \}$ . Extended Predictive Driver Model Score (EPDMS) [19] extends it to $\mathcal { C } = \{ \mathrm { N C } , \mathrm { D A C } , \mathrm { D D C } , \mathrm { T L C } \}$ and F = {EP, TTC, LK, HC, EC}.

Closed-loop metric: Driving Score. For reactive closed-loop simulation, we score the full executed trajectory of the scenario by Driving Score (DS): global route completion (RC) times the mean per-frame EPDMS over the episode:

$$
\mathrm { D S } = R _ { c } \cdot \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathrm { E P D M S } ^ { ( t ) } ,\tag{8}
$$

where $R _ { c } \in [ 0 , 1 ]$ represents the percentage of the route completed by the agent relative to the expert driver’s path or the goal destination, T is the total number of frames in the simulation episode, and $\mathrm { E P D M S } ^ { ( t ) }$ is the EPDM score at time step t.

## C.3 Closed-loop Simulation Setup

All closed-loop evaluations run with BridgeSim [3] as the backend physics simulator, with a replan period of $\Delta _ { r } ~ = ~ 5$ frames and a simulation horizon of 8 seconds. At each iteration, the policy consumes the most recent $K = 1$ world-model-rendered frames and outputs a planned trajectory, of which the first $\Delta _ { r }$ steps are executed. Surrounding agents are initialized at their ground-truth pose, and their subsequent behaviors are controlled by the intelligent driver model (IDM), so they react to the ego vehicle while adhering to traffic rules.

## C.4 Navhard-CL Scenario Curation

We construct Navhard-CL, a closed-loop benchmark to systematically diagnose the observation and behavior gaps that current E2E policies face under closed-loop deployment. It consists of a realworld log-replay base set Navhard-Base [19] and two challenging variations: Navhard-AdvBehavior and Navhard-AdvWeather.

Navhard-Base. The base set comprises the 421 scenarios of NAVSIM navhard [19], each ported into the MetaDrive backend via ScenarioNet [42, 3]. These scenarios cover dense urban driving with a wide range of map topologies and surrounding-agent densities, and serve as our distributionmatched reference.

Navhard-AdvBehavior. For each base scenario, we use Adv-BMT [43] default setting to introduce an adversarial agent that maneuvers to provoke a collision with the ego vehicle. We review and filter to retain 325 highquality safety-critical variants. Each variant inherits the ego route and map of its base scenario but exposes the policy to a reactive adversary, enabling a paired comparison against the base set on the same map. These variants span five categories following the National Highway Traffic Safety Administration (NHTSA) pre-crash scenario typology [66, 67], including Rear-End, Straight Crossing Paths, Opposite Direction (head-on), Changing

![](images/72cd0f813573d8436b7630aae329f77cf9833403701060f0c4cdf3ec52a61597.jpg)  
Figure 8: Navhard-AdvBehavior distribution over scenario categories.

Lanes (cut-in), and Left Turn Across Path. The resulting distribution is shown in Fig. 8.

Navhard-AdvWeather. To evaluate the policy’s robustness under appearance shift, we render the base scenarios under diverse conditions that vary on three axes: lighting conditions (sunrise, sunset, twilight, golden hour, blue hour, night), weather conditions (overcast, snow, rain, fog), and road-surface conditions (snow-covered, sand-covered, puddles). The text prompt and initial frame input are adapted and re-rendered correspondingly. This yields more than 5k scenario-appearance combinations for evaluating the policy.

## C.5 Closed-loop Gap Analysis

Navhard-CL exposes performance gap and failure modes that open-loop scoring and prior closedloop benchmarks cannot reveal. On Navhard-AdvBehavior we decompose the closed-loop performance drop to scorer bias and proposal coverage; on Navhard-AdvWeather we measure a policy’s visual robustness under appearance shift.

For the Navhard-AdvBehavior decomposition, we keep each scoring policy’s proposal set fixed and replace its learned scorer with an oracle that selects the proposal with the highest ground-truth EPDMS computed in the simulator. This enables us to measure the scorer bias gap as the driving score decrease due to scorer mis-ranking proposals the policy generated. The coverage gap measures the policy’s proposal degradation under adversarial scenarios when scoring is already optimal.

Table 7: Decomposing the closed-loop performance gap on Navhard-CL into scorer bias and proposal coverage. For each scoring policy we report driving score (DS) under its learned scorer (Learned) and an oracle scorer that selects the proposal with the highest ground-truth EPDMS (Oracle), on the base set Navhard-Base and the adversarial set Navhard-AdvBehavior. Gray arrows mark the two gaps: ↕ scorer bias gap, ←→ the proposal coverage gap. <sup>†</sup> has no scoring head.
<table><tr><td colspan="2">Policy Scorer</td><td colspan="2">Base DS↑</td><td colspan="2">Adv DS↑</td></tr><tr><td rowspan="4">DiffusionDrive [41]</td><td>Learned</td><td>58.76</td><td></td><td>40.44</td></tr><tr><td></td><td>3.64</td><td></td><td>8.98</td></tr><tr><td>Oracle</td><td>62.40</td><td>12.98</td><td>49.42</td></tr><tr><td>Learned</td><td>58.06</td><td></td><td>38.34</td></tr><tr><td rowspan="3">DiffusionDriveV2 [52]</td><td></td><td>-0.99</td><td></td><td>3.13</td></tr><tr><td>Oracle</td><td>57.07</td><td>15.60</td><td>41.47</td></tr><tr><td>Learned</td><td>46.04</td><td></td><td>36.36</td></tr><tr><td rowspan="3">DrivoR [39]</td><td></td><td>14.17</td><td></td><td>7.19</td></tr><tr><td>Oracle</td><td>60.21</td><td>16.66</td><td>43.55</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>LTF† [48]</td><td></td><td>51.97</td><td>14.61 人</td><td>37.36</td></tr></table>

Table 8: Closed-loop evaluation of DrivoR across appearance variations in Navhard-AdvWeather, comparing against the base scenarios. Gray is the DS gap versus base. The degradation mainly resides in route completion (RC) and drivable-area compliance (DAC) under road-surface-obscuring conditions.
<table><tr><td>Condition</td><td>DS</td><td>EPDMS</td><td>RC</td><td>NC</td><td>DAC</td><td>TTC</td><td>LK</td><td>EC</td></tr><tr><td>Base</td><td>一 47.06</td><td>69.73</td><td>66.38</td><td>95.90</td><td>85.37</td><td>87.07</td><td>95.11</td><td>59.87</td></tr><tr><td colspan="9">Lighting</td></tr><tr><td>Blue hour</td><td>47.49 (+0.43)</td><td>69.82</td><td>66.74</td><td>96.14</td><td>84.08</td><td>88.47</td><td>95.37</td><td>61.10</td></tr><tr><td>Night</td><td>46.79 (−0.27)</td><td>70.08</td><td>65.10</td><td>96.16</td><td>83.44</td><td>89.74</td><td>95.98</td><td>65.22</td></tr><tr><td>Golden hour</td><td>46.64 (−0.42)</td><td>69.70</td><td>65.53</td><td>95.98</td><td>84.26</td><td>87.94</td><td>95.51</td><td>61.25</td></tr><tr><td>Twilight</td><td>46.47 (−0.59)</td><td>70.38</td><td>64.32</td><td>96.38</td><td>84.39</td><td>89.44</td><td>95.24</td><td>61.03</td></tr><tr><td>Sunrise</td><td>46.29 (−0.77)</td><td>69.48</td><td>65.12</td><td>95.80</td><td>84.55</td><td>87.49</td><td>95.14</td><td>59.70</td></tr><tr><td>Sunset</td><td>46.28 (−0.78)</td><td>70.06</td><td>64.17</td><td>96.09</td><td>84.19</td><td>89.23</td><td>95.66</td><td>61.68</td></tr><tr><td colspan="9">Weather</td></tr><tr><td>Rain</td><td>47.21 (+0.15)</td><td>69.08</td><td>66.78</td><td>96.38</td><td>83.03</td><td>89.15</td><td>95.80</td><td>61.45</td></tr><tr><td>Overcast</td><td>47.13 (+0.07)</td><td>69.13</td><td>66.75</td><td>96.26</td><td>84.06</td><td>86.49</td><td>94.92</td><td>59.06</td></tr><tr><td>Snow</td><td>44.70 (−2.36)</td><td>67.71</td><td>64.89</td><td>96.22</td><td>83.01</td><td>86.61</td><td>95.24</td><td>57.91</td></tr><tr><td>Fog</td><td>44.50 (−2.56)</td><td>69.73</td><td>62.57</td><td>96.18</td><td>84.17</td><td>87.40</td><td>95.59</td><td>62.21</td></tr><tr><td colspan="9">Road surface</td></tr><tr><td>Puddles</td><td>45.75 (-1.31)</td><td>67.96</td><td>65.88</td><td>96.09</td><td>81.95</td><td>88.31</td><td>95.17</td><td>60.33</td></tr><tr><td>Sand-covered</td><td>43.42 (−3.64)</td><td>68.47</td><td>62.03</td><td>95.88</td><td>82.53</td><td>89.03</td><td>95.63</td><td>63.38</td></tr><tr><td>Snow-covered</td><td>42.54 (−4.52)</td><td>66.52</td><td>62.76</td><td>95.76</td><td>79.37</td><td>91.92</td><td>95.94</td><td>64.58</td></tr></table>

Scorer bias widely exists across E2E policies. Tab. 7 shows that scorer bias leads to varying performance gaps, with DrivoR decreasing the most, while DiffusionDriveV2’s is effectively zero. We attribute the small scorer bias of DiffusionDriveV2 to its carefully finetuned scorer using EPDMS, which effectively mitigates the scorer gap under Navhard-Base normal scenarios. Under safetycritical distributions, this gap shifts differently for different policies. The DiffusionDrive and DiffusionDriveV2 scorer bias doubles, while DrivoR’s narrows due to the Oracle performance collapses. Further analysis on DrivoR shows that on 14.1% of Navhard-AdvBehavior, its proposal set contains no candidate with positive EPDMS, versus 7.9% on Navhard-Base. Navhard-CL’s reactive closedloop rollouts compound the cost of scorer mis-ranking, which is different from open-loop evaluation on logged trajectories. This extends the previous open-loop scorer-mismatch analysis [54] into the closed-loop regime, revealing the scorer bias that open-loop scoring cannot expose.

Proposals fail to cover feasible and recovery trajectories. The Oracle scorer represents the upper bound that scoring can recover; the remaining performance gap is attributable to the proposal set failing to contain a feasible safe trajectory. Tab. 7 shows large Oracle DS decreases from Navhard-Base to Navhard-AdvBehavior. Furthermore, we observe a similar scale of performance gap for a policy such as LTF that emits a single trajectory without a scorer. This verifies that this gap resides in the trajectory decoder. Analysis on DrivoR’s 64 proposal set reveals the average EPDMS of the best proposal decreases from 70.6 to 66.7 under safety-critical distribution. These results show the coverage gap of proposal generation, a failure mode that only surfaces when forcing the proposal coverage outside the policy’s training distribution.

Visual robustness is limited more by road-surface appearance than by lighting. We evaluate DrivoR across appearance variations in Navhard-AdvWeather, with results shown in Tab. 8. We observe that lighting and common weather variations have little impact on the driving performance, as these conditions are also present in the training distribution. However, for conditions that obscure road elements, such as fog, snow, sand, and puddles, they result in larger performance degradation. Closed-loop safety submetrics, including NC and TTC, remain stable across conditions, suggesting that the performance drop is not driven by a safety regression. The decrease mainly comes from route completion and drivable-area compliance, indicating that the policy perceives the road semantics less reliably when its perception departs from the training distribution. This reveals a visual robustness gap related to road-surface appearance, which closed-loop benchmarks with limited visual diversity are unlikely to expose.

## D More Visualization

We present additional qualitative examples of the closed-loop policy evaluation using DreamStream below in Fig. 9, demonstrating its visual realism and alignment.

![](images/183cdba508f16ccdb4718a85ccc517763b03eb4304006e07cf38584f7f0900ac.jpg)  
Figure 9: More visualizations of closed-loop policy evaluation using DreamStream.