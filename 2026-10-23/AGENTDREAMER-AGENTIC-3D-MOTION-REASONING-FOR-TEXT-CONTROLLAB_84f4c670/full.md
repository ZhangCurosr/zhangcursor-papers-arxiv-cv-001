# AGENTDREAMER: AGENTIC 3D MOTION REASONING FOR TEXT-CONTROLLABLE PHYSICS SIMULATION

Mijeong Kim<sup>∗</sup> Suho Park<sup>∗</sup> Junha Park Bohyung Han

Equal contribution

## ABSTRACT

We propose AgentDreamer, an agent-guided optimization framework that aligns text instructions with simulated motion through explicit 3D motion planning. Given a text instruction and scene geometry, an off-the-shelf agent translates the instruction into a structured semantic motion prior, specifying what should move, in which direction, and in what temporal order. The agent represents this prior using parametric functions over a series of temporal chunks, yielding dense 3D trajectories. We then optimize the parameters of a differentiable MPM simulator such that its rollout best matches the prior while remaining constrained by the scene dynamics. This allows the agent to provide semantic guidance from language, while the simulator grounds it in physically feasible motion. We further introduce Motion-IG, an information-theoretic metric for text–motion alignment based on masked text recovery. By comparing the likelihood of recovering masked motion-related content with and without the simulated motion, Motion-IG measures the information contributed by the motion beyond that available from the remaining text and scene context. Experiments show that AgentDreamer improves text alignment over video-based guidance, producing simulations that more closely follow the requested physical behavior. Our project page is available at https://cv.snu.ac.kr/research/AgentDreamer.

## 1 INTRODUCTION

Controlling physical motion in simulated scenes is important for creating dynamic environments for embodied AI and interactive content generation. Without an intuitive control interface, users must manually tune simulation parameters through repeated simulation and adjustment to obtain the desired motion. Text-controllable physics simulation offers a promising alternative by allowing users to directly describe the intended behavior in natural language. Recent approaches, including PhysFlow (Liu et al., 2025) and MotionPhysics (Wang et al., 2026), leverage video generative priors to guide the optimization of simulation parameters. In this way, they connect high-level user specifications to the dynamics of a 3D scene.

Despite this progress, the resulting simulation may not reproduce the physical behavior described by the text. A central difficulty lies in the motion reference itself: visually plausible video generation does not ensure that the requested dynamics are represented correctly. For example, a request for a rolling ball may produce sliding, omitting the rotation that distinguishes the two behaviors. Optimizing a simulator to follow such a reference can therefore yield physically plausible motion that remains inconsistent with the text. This discrepancy motivates explicit 3D motion planning that describes how the object should move to reproduce the requested behavior.

We propose AgentDreamer, a framework for text-controllable physics simulation that bridges text and simulated motion through agent-based 3D motion planning. Our key idea is to formulate motion planning as a physical reasoning problem. Given a text instruction and scene geometry, an off-theshelf multimodal agent draws on its language understanding and physical knowledge to infer how the object should move, as shown in Figure 1. The agent thus takes a direct role in constructing the motion plan, extending its use beyond estimating material parameters (Liu et al., 2025; Wang et al., 2026). This plan provides explicit 3D guidance for translating the text into simulated motion.

![](images/2177cfa917e3c4ed994f66756cd1980dc6ce5fc5db6a1bf527106f9a2e434a0e.jpg)  
Figure 1: 3D motion planning as physical reasoning. Given a text instruction and scene geometry, the agent reasons about the physical behavior required by the instruction and constructs a corresponding parametric motion plan. The plan produces dense 3D trajectories that guide physics optimization toward the requested motion.

To support motion reasoning, we introduce automatic semantic injection that associates sampled points with visible object parts. We provide a scene image with colored point markers corresponding to the 3D point input, together with coordinate-system information. This correspondence helps the agent interpret the role of each point in the requested motion. The agent then divides the time horizon into temporal chunks and infers a parametric motion function for each. These functions describe distinct motion stages and provide dense trajectories without predicting each future position independently. The resulting trajectories serve as pseudo supervision for a differentiable Material Point Method (MPM) simulator. We optimize simulation parameters to match these trajectories, guiding the simulator toward physically plausible motion aligned with the text.

We further introduce Motion-IG, an information-gain-based metric for measuring text-motion alignment through masked text recovery. An MLLM-as-a-judge score (Zheng et al., 2023; Chen et al., 2024), read as the model’s verbalized confidence, considers the complete text and simulated motion together, without isolating the information provided by the motion. We instead mask motion-related content in the text and compare its recovery likelihoods with and without simulated motion under a fixed evaluation model. The difference measures how much the motion contributes to recovering the masked content beyond the remaining text.

Experiments demonstrate that AgentDreamer improves text alignment over video-based guidance, producing simulations that more closely follow the requested physical behavior. Our contributions are summarized as follows:

• We propose AgentDreamer, a framework for text-controllable physics simulation that bridges text and simulated motion through agent-based 3D motion planning.

• We formulate motion planning as physical reasoning through parametric functions over temporal chunks, providing dense trajectories as pseudo supervision for differentiable physics optimization.

• We introduce Motion-IG, an information-gain-based metric for text-motion alignment that compares masked text recovery likelihoods with and without simulated motion.

## 2 RELATED WORKS

Dynamic 3D representations Recent methods extend 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) to reconstruct dynamic scenes from video sequences. These methods (Luiten et al., 2024; Wu et al., 2024; Yang et al., 2024; Lin et al., 2024a; Katsumata et al., 2024; Jiang et al., 2025a; Lee et al., 2024) represent temporal changes through persistent Gaussian trajectories, deformation fields, or space-time primitives. The particle-based structure of Gaussian primitives further enables these representations to be extended to physics-based dynamics. PhysGaussian (Xie et al., 2024) couples Gaussian primitives with Material Point Method (MPM) simulation, using a shared representation for physical dynamics and rendering. Subsequent works develop elastic deformation models (Zhong et al., 2024), estimate material properties (Zhang et al., 2024; Cai et al., 2024), and support diverse material behaviors and interactions (Lin et al., 2025; Jiang et al., 2025b).

![](images/c48e39341b180e7ed81b984812c7c481b65ac5e57f404267838c52c2c8f94b97.jpg)  
Figure 2: Overview of AgentDreamer. Given a text instruction and scene geometry, an off-theshelf agent constructs a structured semantic motion prior using parametric functions over temporal chunks. The resulting dense 3D trajectories guide optimization of a differentiable MPM simulator under the scene dynamics.

Text-conditional physics simulation Text provides an intuitive interface for specifying object dynamics and interactions. Recent approaches (Wang et al., 2026; Liu et al., 2025; 2024; Lin et al., 2025) translate these instructions into simulation guidance through video diffusion priors. Phys-Flow (Liu et al., 2025) uses optical-flow guidance to optimize physical parameters, while Motion-Physics (Wang et al., 2026) employs learnable motion distillation to guide simulation toward textspecified dynamics. These approaches primarily derive motion guidance in 2D space from generated videos. In contrast, our method constructs an explicit 3D motion prior through physical reasoning with an off-the-shelf multimodal agent, directly specifying the desired scene dynamics. Prior works have also used multimodal models to construct explicit motion priors, including bounding-box plans in VLIPP (Yang et al., 2025) and 2D point trajectories in C-Drag (Li et al., 2025). However, these representations remain sparse and confined to image space. Our approach instead produces dense 3D trajectories over the scene geometry, which serve as pseudo supervision for differentiable physics simulation.

Agent-guided physics-aware motion evaluation Vision-language models provide semantic and physical feedback for evaluating and improving generated motion (Lin et al., 2024b; Bansal et al., 2025; Meng et al., 2024). PhyGDPO (Cai et al., 2026) uses VLM-based scores of semantic adherence and physical commonsense to guide groupwise preference optimization. PhyCo (Narayanan et al., 2026) evaluates generated videos through targeted physical questions and converts response logits into differentiable rewards for refining video generation. These approaches assess motion through semantic or physical judgments (Zheng et al., 2023; Chen et al., 2024), but do not explicitly separate the evidence supplied by motion from information available in the textual context alone. We instead formulate text-motion alignment as conditional information gain (Wyner, 1978), measuring how much a simulated trajectory improves the recovery of masked motion semantics over the remaining text.

## 3 METHOD

We present AgentDreamer, a framework for text-controllable physics simulation through agentgenerated 3D motion planning, as shown in Figure 2. The agent translates the requested behavior into parametric motion functions, whose trajectories provide pseudo supervision for physics optimization. We further introduce Motion-IG to measure text–motion alignment through masked text recovery.

## 3.1 AGENT-GUIDED PHYSICS SIMULATION

Given a text instruction X and initial scene geometry G, we construct a 3D motion plan and optimize simulation parameters to follow its trajectories.

![](images/d44face5c8f02f9145e6f424026bfdaaffee729a9be29740359672ff5b45d024.jpg)  
Figure 3: Effect of parametric motion planning. Direct prediction of future point positions produces irregular trajectories and distorts the object structure (top). Parametric functions yield coherent trajectories that better preserve spatial relationships between points (bottom). Colors identify corresponding points across time.

## 3.1.1 AGENT-BASED MOTION PLANNING

Automatic semantic injection Point coordinates alone provide limited semantic information about the object parts relevant to the instruction X. Let ${ \bf p } _ { k } ^ { 0 }$ be the initial position of MPM particle $k ,$ and let $\bar { \kappa }$ be the set of particle indices selected for guidance. The set K can contain any selected MPM particles; in our experiments, it consists of sampled particles. For each initial view $I ^ { v }$ , we project the selected points with camera projection $\pi ^ { v }$ and mark their indices as follows:

$$
I _ { \mathrm { s e m } } ^ { v } = \mathrm { A n n o t a t e } \left( I ^ { v } , \{ ( k , \pi ^ { v } ( \mathbf { p } _ { k } ^ { 0 } ) ) : k \in \mathcal { K } \} \right) .\tag{1}
$$

Let $\mathbf { I } _ { \mathrm { s e m } } = ( I _ { \mathrm { s e m } } ^ { 1 } , \dots , I _ { \mathrm { s e m } } ^ { V } )$ denote the annotated views, where V is the number of views. Each marker displays the particle index k consistently across views, allowing the agent to associate its 3D position with a visible object part. We also specify the coordinate origin, axis directions, and spatial units. Examples of annotated inputs are given in the Appendix.

Parametric motion planning Direct prediction of future point positions can distort the point configuration over time, as shown in Figure 3. We instead ask the agent to infer motion functions and their parameters. Since a single function may not capture distinct motion stages, the agent divides the time horizon into temporal chunks and specifies a function for each. The agent groups points with similar motion and divides the time horizon into J chunks with boundaries $0 = t _ { 0 } < \cdot \cdot \cdot < t _ { J } = T$ Given the instruction, selected point coordinates ${ \mathcal { C } } = \{ ( k , \mathbf { p } _ { k } ^ { 0 } ) : k \in { \mathcal { K } } \}$ , and annotated views $\mathbf { I } _ { \mathrm { s e m } } .$ the agent A produces the plan as follows:

$$
\Pi = \mathcal { A } ( X , \mathcal { C } , \mathbf { I } _ { \mathrm { s e m } } ) = \{ ( \phi _ { j } , t _ { j - 1 } , t _ { j } ) \} _ { j = 1 } ^ { J } .\tag{2}
$$

For chunk $j , \phi _ { j }$ combines a center trajectory with the local motion of each point group. The function $\phi _ { j } ( k , { \bf x } , \tau )$ gives the target 3D position for point k from its position x at the chunk start after elapsed time τ. Starting from $\widehat { \mathbf { p } } _ { k } ( 0 ) = \mathbf { p } _ { k } ^ { 0 }$ , the target trajectory is given by

$$
\widehat { \mathbf { p } } _ { k } ( t ) = \phi _ { j } \bigl ( k , \widehat { \mathbf { p } } _ { k } ( t _ { j - 1 } ) , t - t _ { j - 1 } \bigr ) , \qquad t \in [ t _ { j - 1 } , t _ { j } ] .\tag{3}
$$

The condition $\phi _ { j } ( k , { \bf x } , 0 ) = { \bf x }$ ensures continuity between chunks. Evaluating these functions at the desired times yields dense 3D trajectories.

## 3.1.2 MPM OPTIMIZATION

MPM simulation The planned trajectories provide pseudo supervision for a differentiable Material Point Method (MPM) simulator. We optimize the simulation parameters θ, including objectlevel material parameters and per-particle initial velocities, to reproduce the planned motion under the scene dynamics. Let $\mathbf { s } ^ { n }$ denote the MPM state after simulation step n, including particle positions, velocities, and deformation-related quantities, and let $\Delta t _ { \mathrm { s i m } }$ be the step size. The state evolves as

$$
\mathbf { s } ^ { n + 1 } = \mathrm { M P M } ( \mathbf { s } ^ { n } ; \pmb { \theta } , \Delta t _ { \mathrm { s i m } } ) .\tag{4}
$$

Let $\mathbf { p } _ { k } ( t ; \pmb \theta )$ be the simulated position of particle k at output time t. The particle index k establishes correspondence with the target position $\widehat { \mathbf { p } } _ { k } ( t )$

Table 1: Examples of word-level masking for Motion-IG. Bold words in the original instruction X are replaced by [MASK] in the masked instruction $X _ { m } .$
<table><tr><td>Sequence</td><td>Instruction</td></tr><tr><td>X</td><td>The soft fox sways from side to side while remaining attached at the rear.</td></tr><tr><td> $X _ { m }$ </td><td>The [MASK] fox [MASK] from [MASK] to side while [MASK] [MASK] at the rear.</td></tr><tr><td> $X$   $X _ { m }$ </td><td>The basketball rolls forward in a straight line, rotating naturally without slipping. The basketball [MASK] [MASK] in a [MASK] [MASK], [MASK] [MASK] [MASK] [MASK].</td></tr><tr><td> $X$   $X _ { m }$ </td><td>The rigid playdoh ball falls straight down and bounces several times without deforming. The [MASK] playdoh ball [MASK] [MASK] [MASK] and [MASK] [MASK] [MASK] without [MASK].</td></tr></table>

Motion reconstruction loss We match the simulated and target positions of the selected particles at output times $t _ { 1 } , \dots , t _ { F } \colon$

$$
\mathcal { L } _ { \mathrm { m o t i o n } } ( \pmb { \theta } ) = \frac { 1 } { K F } \sum _ { k \in { \cal K } } \sum _ { f = 1 } ^ { F } \Vert \mathbf { p } _ { k } ( t _ { f } ; \pmb { \theta } ) - \widehat { \mathbf { p } } _ { k } ( t _ { f } ) \Vert _ { 2 } ^ { 2 } ,\tag{5}
$$

$$
\pmb { \theta } ^ { * } = \arg \operatorname* { m i n } _ { \pmb { \theta } } \mathcal { L } _ { \mathrm { m o t i o n } } ( \pmb { \theta } ) .\tag{6}
$$

Here $K = | { \boldsymbol { K } } |$ . Although the loss is evaluated only for particles in K, MPM evolves the full scene. The motion functions provide target positions for the selected particles at each output time. We keep the plan fixed and update θ using gradients through the MPM rollout.

## 3.2 MOTION-IG

We introduce Motion-Conditioned Information Gain (Motion-IG) to measure text–motion alignment through masked text recovery. It measures how much observing the simulated motion improves recovery of motion-related content beyond the remaining text.

Notation Let $X = ( x _ { 1 } , \dots , x _ { L } )$ be the instruction word sequence and $\mathcal { M } = \{ m _ { 1 } , . . . , m _ { R } \}$ the set of masked positions, where $m _ { 1 } < \cdots < m _ { R }$ . Replacing these words with [MASK] yields the masked sequence $X _ { m }$ ; the removed words form the ordered subsequence $X ^ { \mathcal { M } } = ( x _ { m _ { 1 } } , \dots , x _ { m _ { R } } )$ We mask motion words and attributes while retaining object and part names (Table 1). Let Y denote the simulated motion, represented by rendered frames and serialized 3D trajectories. The masked text $X _ { m }$ and annotated initial views $\mathbf { I } _ { \mathrm { s e m } }$ remain identical in both conditions. For brevity, we omit the shared conditioning on $\mathbf { I } _ { \mathrm { s e m } }$ in the equations below.

Information gain The masked content may be partially inferred from the remaining text and annotated views. We therefore measure the additional information provided by the motion through conditional mutual information as follows:

$$
I ( X ^ { \mathcal { M } } ; Y \mid X _ { m } ) = \operatorname { \mathbb { E } } \left[ \log \frac { p ( X ^ { \mathcal { M } } \mid X _ { m } , Y ) } { p ( X ^ { \mathcal { M } } \mid X _ { m } ) } \right] .\tag{7}
$$

Since these probabilities are unknown, we use a fixed vision-language model $p _ { \psi }$ to compute a pointwise score as follows:

$$
s _ { \psi } ( X , Y ) = \log \frac { p _ { \psi } ( X ^ { \mathcal { M } } \mid X _ { m } , Y ) } { p _ { \psi } ( X ^ { \mathcal { M } } \mid X _ { m } ) } .\tag{8}
$$

The denominator gives the recovery likelihood without the simulated motion. A positive score indicates that observing the motion increases the likelihood of the removed words; a negative score indicates that it decreases this likelihood.

Score computation We provide the masked text as the query and teacher-force the original instruction as the response, with and without simulated motion. Let N be the number of evaluated text–motion pairs. For pair $( X _ { r } , Y _ { r } )$ , let $\left( z _ { r , 1 } , \ldots , z _ { r , L _ { r } } \right)$ denote the response tokens. The token log-probability gain is given by

$$
\delta _ { \psi } ( r , \ell ) = \log p _ { \psi } \left( z _ { r , \ell } \mid X _ { m , r } , Y _ { r } , z _ { r , < \ell } \right) - \log p _ { \psi } \left( z _ { r , \ell } \mid X _ { m , r } , z _ { r , < \ell } \right) .\tag{9}
$$

The preceding response tokens $z _ { r , < \ell }$ are identical in both conditions. Let $\mathcal { M } _ { r } ^ { \mathrm { s c o r e } } \subseteq \mathcal { M } _ { \tau }$ contain the masked positions selected for scoring, excluding words such as “is” and “the” but retaining negations such as “not”. For each selected word at position m, we score its first subword token at position $\ell _ { r } ( m )$ . Averaging first over words and then over N instructions gives Motion-IG as follows:

$$
\mathrm { M o t i o n - I G } : = \frac { 1 } { N } \sum _ { r = 1 } ^ { N } \frac { 1 } { | \mathcal { M } _ { r } ^ { \mathrm { s c o r e } } | } \sum _ { m \in \mathcal { M } _ { r } ^ { \mathrm { s c o r e } } } \delta _ { \psi } \left( r , \ell _ { r } ( m ) \right) .\tag{10}
$$

This first-subword score is a proxy for Eq. 7. We compute Motion-IG for each method using its simulated motions while keeping the instructions, masks, annotated views, and evaluation models identical across methods.

## 4 EXPERIMENTS

We evaluate whether agent-generated 3D motion plans improve text-controllable physics simulation across diverse object motions. We use GPT-5.6 Sol as the off-the-shelf agent for 3D motion planning throughout our experiments. We compare the simulations with video-guided methods using Motion-IG and blinded human rankings of text–motion alignment.

## 4.1 EXPERIMENTAL SETUP

Scenes and instructions. We conduct experiments on eight scenes with 20 text instructions. The dataset comprises four real-world scenes used by PhysFlow (Liu et al., 2025) (Alocasia, Carnation, Telephone and Fox) and four synthetic scenes, one from PAC-NeRF (Li et al., 2023) (Playdoh) and three from GaussianVerse (Xiang et al., 2025) (Basketball, Soccer Ball and Can). The instructions cover falling and bouncing, rolling, slowing rotation, oscillation, and bending or deformation, with 10 applied to the real-world scenes and 10 to the synthetic scenes.

Baselines and evaluation. We compare AgentDreamer with PhysFlow (Liu et al., 2025) and MotionPhysics (Wang et al., 2026), which use text-conditioned video diffusion to guide physics simulation. We use four open-weight evaluators: InternVL3.5-8B (Wang et al., 2025), Qwen3-VL-8B (Qwen Team, 2025), Qwen3.5-9B (Qwen Team, 2026), and Gemma-4-12B (Team et al., 2026). We additionally report results with five API-based evaluators: GPT-5.6 Sol and GPT-5.6 Terra (OpenAI, 2026a), GPT-6 Astra (OpenAI, 2026b), Gemini 3.1 Pro (Google DeepMind, 2026a), and Gemini 3.8 Flash (Google DeepMind, 2026b). To compute Motion-IG, we use teacher-forced token likelihoods for open-weight evaluators and directly requested probabilities for API-based evaluators. For human evaluation, participants rank three anonymized simulations for each instruction by how well the motion follows the text.

## 4.2 COMPARISON WITH EXISTING METHODS

## 4.2.1 QUALITATIVE RESULTS

Figure 4 compares the three methods on a bouncing playdoh ball and a fox instructed to rock while its rear remains attached. For the ball, PhysFlow and MotionPhysics deform on impact but remain near the ground, whereas AgentDreamer rebounds after compression as requested. For the fox, the shown frames of AgentDreamer display the requested rocking motion while the rear remains attached; the baseline sequences show less of this motion. Both examples depend on the motion sequence specified in the instruction, which the agent represents through a 3D trajectory prior before simulation.

## 4.2.2 QUANTITATIVE RESULTS

Motion-IG with open-weight evaluators Table 2a compares text–motion alignment using four open-weight evaluators. On the 20 evaluated instructions, AgentDreamer achieves the highest score under all four evaluators. Its positive scores indicate that providing the simulated motion increase the likelihood of the correct masked words, on average, compared with the condition without motion. These results suggest that motion generated by AgentDreamer provides more useful information for recovering the requested behavior than the compared methods.

![](images/e792728aa1404deaff2001cbef101b1290a6de2fe52d93d239625ccb7ab95943.jpg)  
“The soft playdoh ball falls down, deforms on impact, and bounces several times while squashing.”

![](images/905617325a165ef7da39cae7ea1518651cbec5c95bd1975f6bf04a6ee2c5f020.jpg)  
Figure 4: Qualitative comparison of text-controllable physics simulation. AgentDreamer more closely follows the requested motion than PhysFlow and MotionPhysics in the shown examples. The shown examples contrast the requested motion with the behaviors produced by video-guided optimization and explicit 3D planning.

Table 2: Comparison of text–motion alignment using Motion-IG with (a) token likelihoods from open-weight evaluators and (b) verbalized probabilities from black-box API evaluators (higher is better). Best in counts the evaluator–instruction pairs in which each method ranks first.  
(a) Open-weight evaluators
<table><tr><td>Method</td><td>Gemma-4</td><td>Qwen3.5</td><td>Qwen3-VL</td><td>InternVL3.5</td><td>Average</td><td>Best in</td></tr><tr><td>PhysFlow (Liu et al., 2025)</td><td>0.36</td><td>0.62</td><td>0.58</td><td>0.52</td><td>0.52</td><td>15/80</td></tr><tr><td>MotionPhysics (Wang et al., 2026)</td><td>0.13</td><td>0.58</td><td>0.81</td><td>0.56</td><td>0.52</td><td>22/80</td></tr><tr><td>AgentDreamer (ours)</td><td>0.63</td><td>0.67</td><td>0.92</td><td>0.64</td><td>0.71</td><td>43/80</td></tr></table>

(b) Black-box evaluators
<table><tr><td></td><td colspan="3">GPT</td><td colspan="3">Gemini</td><td></td></tr><tr><td>Method</td><td>6 Astra</td><td>5.6 Sol</td><td>5.6 Terra</td><td>3.8 Flash</td><td>3.1 Pro</td><td>Average</td><td>Best in</td></tr><tr><td>PhysFlow (Liu et al., 2025)</td><td>-0.13</td><td>-0.45</td><td>+0.04</td><td>-0.69</td><td>+0.14</td><td>-0.22</td><td>23/100</td></tr><tr><td>MotionPhysics (Wang et al., 2026)</td><td>-0.49</td><td>-0.96</td><td>-0.44</td><td>-0.93</td><td>-0.32</td><td>-0.63</td><td>15/100</td></tr><tr><td>AgentDreamer (ours)</td><td>+0.56</td><td>+0.25</td><td>+0.48</td><td>+0.12</td><td>+1.00</td><td>+0.48</td><td>62/100</td></tr></table>

Motion-IG with API-based evaluators Table 2b reports results using verbalized probabilities from five API-based evaluators. The lower average scores may partly reflect stronger recovery from the masked text and annotated views alone: if an evaluator already predicts the masked words with high probability, observing motion provides less additional gain. Nevertheless, AgentDreamer achieves the highest score under all five evaluators and ranks first in 62/100 evaluator–instruction pairs. This consistent ranking suggests that its simulated motions provide more useful evidence for recovering the requested behavior across both evaluation approaches. We compare rankings within each evaluator group because verbalized probabilities and teacher-forced token likelihoods have different scales.

Human evaluation We conduct a human evaluation of text–motion alignment with 25 participants. For each instruction, participants view the results of all three methods and rank them by how well they match the requested behavior. AgentDreamer is ranked first most often (73.6%), compared with PhysFlow (14.0%) and MotionPhysics (12.4%). Figure 5 further shows that AgentDreamer is preferred over each baseline in pairwise comparisons. This agrees with the Motion-IG ranking, providing further evidence that AgentDreamer follows the text instructions more closely than the baselines.

![](images/3e3cb76ceb2d3a4292c750d90eafcdfb56b1cc448d3f8f5b148860d5b948be08.jpg)  
Figure 5: Pairwise human preference for text–motion alignment. Win rate is the percentage of trials in which one method is ranked above the other. AgentDreamer is preferred over PhysFlow in 82.6% of trials and over MotionPhysics in 85.6% of trials.

Table 3: Correspondence between agent-generated motion functions and expected forms across 20 instructions. The expected forms follow classical mechanics for rolling, spin decay, falling, and oscillation, while deformation uses smoothstep interpolation. Expressions show representative components, and Matched / Total counts plans that use the expected form in each category.
<table><tr><td>Motion cue</td><td>Function family</td><td>Representative expression</td><td>Matched / Total</td></tr><tr><td>Roll</td><td>Coupled translation and rotation</td><td> $d ( \tau ) = v \tau , \quad \theta ( \tau ) = ( v / R ) \tau$ </td><td>2/2</td></tr><tr><td>Spin, slow down, stop</td><td>Quadratic</td><td> $\begin{array} { r } { \theta ( \tau ) = \omega _ { 0 } \tau + \frac { 1 } { 2 } \alpha \tau ^ { 2 } , \quad \alpha < 0 } \end{array}$ </td><td>2/2</td></tr><tr><td>Fall, bounce, jump</td><td>Quadratic</td><td> $z ( \tau ) = z _ { 0 } + v _ { z , 0 } \tau - \textstyle { \frac { 1 } { 2 } } g \tau ^ { 2 }$ </td><td>4/4</td></tr><tr><td>Sway, oscillate, rock</td><td>Sinusoid</td><td> $\theta ( \tau ) = A e ^ { - \lambda \tau } \sin ( \omega \tau { \bar { ) } }$ </td><td>8/8</td></tr><tr><td>Pull, bend, squash, deform</td><td>Smoothstep interpolation</td><td> $s ( u ) = 3 u ^ { 2 } - 2 u ^ { 3 }$ </td><td>4/4</td></tr></table>

τ : elapsed time within a chunk; $u = \tau / ( t _ { j } - t _ { j - 1 } ) \colon$ normalized time.  
d: travel distance; θ: rotation angle; R: ball radius.

![](images/5586c8700bad1214f31d706e0dae3430feee0e82685cdfc2cd222a507657fb7a.jpg)

![](images/3b4bfa9f680938d43077ecb0d09ceccd3c19975edf874c1702ad7262e4b04760.jpg)  
Figure 6: Text–motion alignment across motion categories measured by Motion-IG (higher is better). AgentDreamer achieves the highest score in every category; for rolling, both baselines have negative information gain.  
Figure 7: Consistency of Motion-IG with human evaluation across motion categories. Both show higher first-rank rates for rolling and spinning and lower rates for oscillation and deformation.

## 4.3 ANALYSIS

Correspondence to classical mechanics. We examine whether the agent-generated functions follow motion equations derived from classical mechanics. Table 3 reports how often the plans match the expected function family. The generated forms resemble the classical relations for rolling, falling, spin decay, and damped oscillation, including coupled translation and rotation, quadratic trajectories, and damped sinusoids. For deformation, the agent uses smoothstep interpolation to describe the temporal change rather than a classical mechanics equation. These similarities suggest that the agent draws on familiar physical motion patterns when constructing its 3D prior.

Benefits of 3D motion planning across motion categories. To examine the benefit of 3D planning over 2D video-based guidance, we compare Motion-IG across five motion categories in Figure 6. AgentDreamer achieves the highest score in every category, with the largest improvements for rolling, spin decay, and falling and bouncing. Notably, both baselines obtain negative scores for rolling, indicating that their simulated motions make the masked behavior harder to recover than from the shared context without motion. Rolling and sliding can appear similar in video, although rolling requires coupled translation and rotation in 3D. The larger gain for rolling suggests that explicit 3D planning is especially useful when the requested motion is difficult to distinguish from video appearance alone. Gains are smaller for oscillation and deformation, where the baselines already achieve relatively high scores.

Consistency with human evaluation. We examine whether Motion-IG reflects human preferences by comparing how often AgentDreamer ranks first among the evaluated methods in each motion category. As shown in Figure 7, both human evaluation and Motion-IG select our method most frequently for rolling and spinning. The first-rank rates decrease for oscillation and deformation, where the baselines are selected more often. Although the absolute rates differ, this shared trend suggests that Motion-IG captures category-dependent differences in text–motion alignment observed by human evaluators. These category-level trends do not measure agreement on individual instruction rankings.

![](images/3bb851b1e2aa869827c9c1a60e2558e29082a5f5e7de429d88c84febcff64fdf.jpg)  
Figure 8: Temporal decomposition of an agent-generated motion plan. The agent uses separate parametric functions to describe the ball movement, compression on impact, and partial shape recovery. Dashed lines mark temporal chunks, and colored points show target positions over time.

Table 4: Effect of planning components including semantic injection and parametric motion predictions. Parametric planning provides the largest individual gain, while combining both components achieves the highest score.  
![](images/340d9be076ab35ea6935278a235817467323e9eed41f1ee9adbd51c98872bf72.jpg)

<table><tr><td>Semantic injection</td><td>Parametric planning</td><td>Motion-IG ↑</td></tr><tr><td>√</td><td></td><td>0.37 0.38</td></tr><tr><td></td><td>√</td><td>0.63</td></tr><tr><td></td><td>1</td><td>0.72</td></tr></table>

Figure 9: Score distributions on identical text–motion pairs, scored by gemma-4. Motion-IG gives finer score resolution than an MLLM judge on the same text–motion pairs.

Temporal visualization of motion planning Figure 8 shows how the agent divides the requested motion into temporal chunks and constructs a parametric function for each. The planned sequence captures the ball motion, strong compression on impact, and subsequent partial shape recovery. The chunks correspond to these behavioral changes, suggesting that the agent identifies distinct motion stages within the instruction. By describing each stage separately, the plan captures changes in physical motion that are difficult to represent with a single parametric function.

Effect of planning components Table 4 shows that both components improve text–motion alignment, with parametric planning providing the larger gain. The small gain from semantic injection alone suggests that point annotations are most useful with parametric planning, where they help associate selected 3D points with visible object parts. Without parametric planning, independently predicted point positions lose coherence over time (Figure 3). Motion functions instead provide coherent, dense target trajectories for physics optimization.

Comparison with MLLM judge Figure 9 compares Motion-IG with MLLM judgments (Zheng et al., 2023; Chen et al., 2024) on the same 60 text–motion pairs after scaling each score type to its own range.<sup>1</sup> The direct judge scores the instruction and motion together, whereas Motion-IG measures the additional evidence that the motion provides for recovering masked motion words. Although asked for a real-valued score, the MLLM judge assigns only 11 distinct values to the 60 pairs, giving the same score to multiple motions. In contrast, Motion-IG assigns a distinct score to every pair in this comparison. The reduced number of ties gives finer resolution when ranking these pairs; Figure 7 separately compares category-level first-rank rates with human evaluation.

## 5 CONCLUSION

We presented AgentDreamer, a framework that uses an agent to generate 3D motion priors for textcontrollable physics simulation. Given a text instruction and scene geometry, the agent reasons about the underlying physics and constructs parametric functions to describe motion consistent with the instruction. The resulting trajectories guide differentiable physics optimization to reproduce the requested motion. We have also introduced Motion-IG to measure how much simulated motion helps recover masked motion words beyond the remaining text. Experiments show improved text alignment over video-guided baselines, demonstrating the effectiveness of agentic motion priors.

## AI USE STATEMENT

In this work, we used generative AI tools to help develop theoretical models or conceptual frameworks, design or provide feedback on research methodology or experiments, implement methods, and assist with translation. We have not used generative AI tools to generate synthetic datasets, formulate mathematical claims, clean or reformat datasets, or interpret results, and tasks related to providing critical ingredients for proving mathematical claims, assisting in the writing of proofs, proposing or refining hypotheses, and supporting qualitative or thematic data analysis are not applicable to this work.

Additionally, we used generative AI tools to create or modify scientific figures or images, suggest experimental parameters, create or edit software code and research artifacts, draft parts of the manuscript, summarize or analyze existing literature, discover research topics or identify gaps, brainstorm research ideas, search for information, improve manuscript readability, identify relevant literature, format references, and suggest manuscript structure.

We have reviewed all AI-assisted work. AI-assisted code was verified through execution and testing, and all methodological suggestions and translations were manually reviewed by the authors before use. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work uses existing 3D scenes to study text-controlled physical simulation. The human evaluation involved 25 participants who ranked simulated motions by their alignment with text instructions. Method identities were hidden as A, B, and C, their presentation positions were balanced across instructions, and results are reported in aggregate. Because the motion plans and automated evaluations rely on pretrained multimodal models, they may reflect model errors or biases. We therefore report human judgments alongside model-based scores and do not treat the latter as a sole measure of physical correctness.

## REPRODUCIBILITY STATEMENT

Sections 3.1 and 3.2 describe the motion-planning and physics-optimization procedure and define the Motion-IG metric. Section 4.1 identifies the scene sources, text instructions, baselines, and evaluation models. The appendix provides additional results (Appendices A and B), examples of semantic injection (Appendix C), the prompts used for motion planning, code generation, and evaluation (Appendix D), and the human evaluation procedure (Appendix E).

## REFERENCES

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. In ICLR, 2025.

Junhao Cai, Yuji Yang, Weihao Yuan, Yisheng He, Zilong Dong, Liefeng Bo, Hui Cheng, and Qifeng Chen. Gic: Gaussian-informed continuum for physical property identification and simulation. Advances in Neural Information Processing Systems, 2024.

Yuanhao Cai, Kunpeng Li, Menglin Jia, Jialiang Wang, Junzhe Sun, Feng Liang, Weifeng Chen, Felix Juefei-Xu, Chu Wang, Ali Thabet, et al. Phygdpo: Physics-aware groupwise direct preference optimization for physically consistent text-to-video generation. In ECCV, 2026.

Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. MLLM-as-a-judge: Assessing multimodal LLM-as-a-judge with vision-language benchmark. In ICML, 2024.

Google DeepMind. Gemini 3.1 Pro: A smarter model for your most complex tasks, 2026a. URL https://blog.google/innovation-and-ai/models-and-research/ gemini-models/gemini-3-1-pro/.

Google DeepMind. Gemini 3.8 Flash, 2026b. URL https://ai.google.dev/ gemini-api/docs/latest-model.

Dadong Jiang, Zhi Hou, Zhihui Ke, Xianghui Yang, Xiaobo Zhou, and Tie Qiu. Timeformer: Capturing temporal relationships of deformable 3d gaussians for robust reconstruction. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8721–8732. IEEE, 2025a.

Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, and Yunzhu Li. Phystwin: Physics-informed reconstruction and simulation of deformable objects from videos. In ICCV, 2025b.

Kai Katsumata, Duc Minh Vo, and Hideki Nakayama. A compact dynamic 3d gaussian representation for real-time dynamic view synthesis. In ECCV, 2024.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, George Drettakis, et al. 3d gaussian splat-¨ ting for real-time radiance field rendering. ACM Trans. Graph., 2023.

Junoh Lee, ChangYeon Won, Hyunjun Jung, Inhwan Bae, and Hae-Gon Jeon. Fully explicit dynamic gaussian splatting. NeurIPS, 2024.

Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. Pac-nerf: Physics augmented continuum neural radiance fields for geometry-agnostic system identification. In ICLR, 2023.

Yuhao Li, Mirana Claire Angel, Salman Khan, Yu Zhu, Jinqiu Sun, Yanning Zhang, and Fahad Shahbaz Khan. C-drag: chain-of-thought driven motion controller for video generation. arXiv preprint arXiv:2502.19868, 2025.

Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In CVPR, 2024a.

Yuchen Lin, Chenguo Lin, Jianjin Xu, and Yadong Mu. Omniphysgs: 3d constitutive gaussians for general physics-based dynamics generation. In ICLR, volume 2025, 2025.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In ECCV, 2024b.

Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan. Physics3d: Learning physical properties of 3d gaussians via video diffusion. arXiv preprint arXiv:2406.04338, 2024.

Zhuoman Liu, Weicai Ye, Yan Luximon, Pengfei Wan, and Di Zhang. Unleashing the potential of multi-modal foundation models and video diffusion for 4d dynamic physical scene simulation. In CVPR, 2025.

Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. In 3DV, 2024.

Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsensebased benchmark for video generation. arXiv preprint arXiv:2410.05363, 2024.

Sriram Narayanan, Ziyu Jiang, Srinivasa Narasimhan, and Manmohan Chandraker. Phyco: Learning controllable physical priors for generative motion. In CVPR, 2026.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition, 2026a. URL https: //openai.com/index/gpt-5-6/.

OpenAI. GPT-6 Astra: A new generation of intelligence, 2026b. URL https://openai.com/ index/gpt-6-astra/.

Qwen Team. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, 2026. URL https://qwen.ai/ blog?id=qwen3.5.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 technical report.˘ arXiv preprint arXiv:2607.02770, 2026.

Miaowei Wang, Jakub Zadrozny, Oisin Mac Aodha, and Amir Vaxman. Motionphysics: Learnable˙ motion distillation for text-guided simulation. In AAAI, 2026.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, 2024.

Aaron D Wyner. A definition of conditional mutual information for arbitrary ensembles. Information and Control, 1978.

Tiange Xiang, Kai Li, Chengjiang Long, Christian Hane, Peihong Guo, Scott Delp, Ehsan Adeli,¨ and Li Fei-Fei. Repurposing 2d diffusion models with gaussian atlas for 3d generation. In ICCV, 2025.

Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physics-integrated 3d gaussians for generative dynamics. In CVPR, 2024.

Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, Zhiyong Wang, Jianfei Cai, Tien-Tsin Wong, Huchuan Lu, et al. Vlipp: Towards physically plausible video generation with vision and language informed physical prior. In ICCV, 2025.

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In CVPR, 2024.

Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T Freeman. Physdreamer: Physics-based interaction with 3d objects via video generation. In ECCV, 2024.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In NeurIPS Datasets and Benchmarks Track, 2023.

Licheng Zhong, Hong-Xing Yu, Jiajun Wu, and Yunzhu Li. Reconstruction and simulation of elastic objects with spring-mass 3d gaussians. In ECCV, 2024.

## A ADDITIONAL QUALITATIVE RESULTS

Figure 10 and Figure 11 present additional comparisons of PhysFlow, MotionPhysics and Agent-Dreamer under the same text instructions. In the shown examples, AgentDreamer produces motion that more closely follows the instructions than the baselines. These results complement the main-paper comparisons and further support the use of agent-generated 3D motion priors for textcontrollable physics simulation.

![](images/d69c2675b1a8a331fc5101dd49480b55637cbfec00101acf0b26b2919032b3b6.jpg)  
“One leaf receives an initial sideways velocity, oscillates, and gradually returns to its resting pose.”

![](images/47574015d83534b5d5cee3250340ff89440ade6c05945b3fba958dea6e638f5d.jpg)

“The basketball rolls forward in a straight line, rotating naturally without slipping.”  
![](images/66774fffbf208decf86c8ec64b6fc379340e64e870cddff110edbfcfe4d15255.jpg)  
“The upright flower twists back and forth around its vertical stem axis with strong initial speed.”

Figure 10: Additional qualitative comparisons with PhysFlow (Liu et al., 2025) and Motion-Physics (Wang et al., 2026) under the same text instructions. The left panels show frames from the simulated videos, while the right panels show the corresponding sampled-point trajectories. In the shown examples, AgentDreamer produces simulated motion that more closely follows the requested behavior.

“The soccer ball jumps upward from the floor, then lands and bounces several times.”  
![](images/54c13ca8ad121734ea1c3227cf7f484f26f1edc891b7ce7ef8f2f0f281958124.jpg)  
“The upright can rocks left and right with medium, decreasing amplitude, then settles upright.”

![](images/3356d2783f6c294d51ca41e18ff80679fa2c8a9d08fbbb0bfb07de0a37645fd9.jpg)

![](images/b467bc0a3a35a2ab71b48eb9064bbe1d6f603ddbefb2670ce61f77709c6308d3.jpg)  
“The soccer ball spins moderately in place and gradually slows down until it comes to a complete stop.”

![](images/39cfb3cc9f0e5c53cc210370d0263243a23be5ede90509d6f50a3482d83d4ce2.jpg)  
“A smooth S-shaped wave propagates repeatedly along the soft telephone cord.”

Figure 11: Additional qualitative comparisons with PhysFlow (Liu et al., 2025) and Motion-Physics (Wang et al., 2026) under the same text instructions. The left panels show frames from the simulated videos, while the right panels show the corresponding sampled-point trajectories.

## B ADDITIONAL QUANTITATIVE RESULTS

Text–motion alignment across motion categories Table 5 breaks the Motion-IG scores down by the motion categories of the main paper. AgentDreamer ranks first in every category under both open-weight and black-box evaluators. The gap is clear for simple rigid-body motions governed by a single physical law, namely rolling, falling and bouncing, and spin decay: on rolling, Agent-Dreamer reaches +0.46 with open-weight evaluators while both baselines stay negative (−0.22 and −0.24). This is where the benefit of planning shows most, since the plan can state the motion by its governing law. On more complex motions that repeat or change shape, oscillation and bending and deformation, AgentDreamer still scores highest but by a narrower margin; with black-box evaluators it reaches +0.35 on oscillation, close to PhysFlow at +0.33.

Table 5: Text–motion alignment per motion category measured by Motion-IG (higher is better), averaged over four open-weight and five black-box evaluators.
<table><tr><td></td><td></td><td colspan="3">Open-weight (avg.)</td><td colspan="3">Black-box (avg.)</td></tr><tr><td>Category</td><td>n</td><td>PhysFlow</td><td>MotionPhysics</td><td>Ours</td><td>PhysFlow</td><td>MotionPhysics</td><td>Ours</td></tr><tr><td>Fall and bounce</td><td>4</td><td>+0.18</td><td>+0.36</td><td>+0.63</td><td>-1.46</td><td>-0.87</td><td>+0.77</td></tr><tr><td>Roll</td><td>3</td><td>-0.22</td><td>-0.24</td><td>+0.46</td><td>-1.23</td><td>-2.01</td><td>+0.92</td></tr><tr><td>Spin decay</td><td>2</td><td>+0.29</td><td>+0.20</td><td>+0.55</td><td>-0.15</td><td>-0.23</td><td>+0.62</td></tr><tr><td>Oscillation</td><td>9</td><td>+0.77</td><td>+0.75</td><td>+0.83</td><td>+0.33</td><td>-0.50</td><td>+0.35</td></tr><tr><td>Bend and deform</td><td>5</td><td>+0.67</td><td>+0.70</td><td>+0.74</td><td>+0.34</td><td>-0.12</td><td>+0.39</td></tr></table>

Human evaluation details We collect rankings from 25 participants on 20 instructions, for 500 trials in total. In each trial, participants read the instruction and watch the results of the three methods, stacked vertically and labeled A, B, and C without method names. The position of each method is balanced across instructions. Participants rank the three results from first to third by how well the motion matches the instruction. Table 6 reports how often each method receives each rank. AgentDreamer is ranked first in 73.6% of trials and last in only 5.4%, whereas PhysFlow and MotionPhysics are ranked last in 40.2% and 54.4% of trials, respectively.

Table 6: Rank distribution in the human evaluation (25 participants × 20 instructions = 500 trials). Each entry is the percentage of trials in which the method receives the rank.
<table><tr><td>Method</td><td>1st (%) ↑</td><td>2nd (%)</td><td>3rd (%) ↓</td></tr><tr><td>PhysFlow</td><td>14.0</td><td>45.8</td><td>40.2</td></tr><tr><td>MotionPhysics</td><td>12.4</td><td>33.2</td><td>54.4</td></tr><tr><td>Ours</td><td>73.6</td><td>21.0</td><td>5.4</td></tr></table>

## C SEMANTIC INJECTION EXAMPLES

Figure 12 presents examples of annotated images for two objects. The numerical identifiers show where the sampled points appear in each initial scene image, helping the agent associate the points with visible object parts relevant to the text instruction.

View 1  
View 2  
![](images/8867e3d2d0123baa852d5dd0e10b424d3b040edf366e945a23e909e6ac39bfad.jpg)  
Figure 12: Examples of annotated images used for automatic semantic injection. Control points sampled from the initial scene geometry are marked with numerical identifiers in two views of a fox (top) and a soccer ball (bottom).

## D DETAILS OF AGENT PROMPTS

## D.1 PROMPTS FOR MOTION PLANNING

Figure 13 presents the prompt used to construct the motion plan. The agent groups the sampled points, divides the time horizon into temporal chunks, and specifies a center function and a local function for each group in each chunk. Figure 14 presents the prompt used to translate the resulting motion plan into deterministic Python code. The code computes target point positions from the specified functions without modifying the motion plan.

![](images/f542ded1d0e37e0c2d55f3cbcb87049af2ef90ec3705bf3ef4100bc4d25ed052.jpg)  
Figure 13: Prompt for agent-based motion planning. The agent divides the time horizon into temporal chunks and specifies parametric motion functions for groups of sampled points.

![](images/9d4527f6b2786060a57881e96d48716fa3d30bb40b56738bf728d500b218370c.jpg)  
Figure 14: Prompt for translating the motion plan into deterministic Python code. The code implements the specified motion functions to compute target point positions.

## D.2 PROMPTS FOR EVALUATION

Figure 15 presents the prompt given to the open-weight evaluators. The evaluator receives the annotated initial views, the generated motion as a trajectory and as rendered frames, and the description with its motion words masked; the masked words are scored by teacher forcing rather than by sampling the completion. Figure 16 presents the prompt given to the black-box evaluators. Since current API models expose no teacher-forced log-probabilities, the evaluator states a probability for each masked word, and these ratings replace the log-probabilities in the same gain.

![](images/c3ac970bc31dfbfc16830ab91378e70a446afc88b5f5ad53c6a2f2f7eaa42abc.jpg)  
Figure 15: Prompt for the open-weight evaluators. The masked words are scored by teacher forcing, and the same prompt without the motion gives the reference term.

![](images/aacd729b4b1a2e08e34ead6641d8325a4770974f4570da3e6917dd01f77bdd25.jpg)  
Figure 16: Prompt for the black-box evaluators. Each masked word receives a stated probability, collected in one JSON response per query.

## E DETAILS OF USER STUDY

We conducted a user study to assess how closely the simulated motions follow their text instructions. For each of 20 instructions, participants viewed three videos of the same object, corresponding to PhysFlow, MotionPhysics, and AgentDreamer, and ranked them from first to third according to their text–motion alignment. At the beginning of the survey, we present the following instructions:

In each of the 20 questions, you will see three short videos (A, B, C) of the same object and a text description below them. The videos loop automatically. For each question, please rank the three videos 1st, 2nd, and 3rd: 1st is the one whose motion best matches the text, and 3rd is the one that matches it the least. Each rank can be used only once.

Please judge only how well the motion follows the description: what moves, in which way, and how strongly. Do not judge image quality, lighting, or rendering artifacts. The key words in each description are highlighted in orange. Please watch all three videos at least once before answering.

Figure 17 shows an example question presented to participants in the user study.  
![](images/d9069505d67cd2759655a81b7c3e8c42abfa14cc19d2fec50ddec71ea9d8f54c.jpg)

The flower base stays fixed. The upright flower twists back and forth around its vertical stem axis with strong initial speed and a large, gradually decreasing amplitude. No lateral motion or additional torque is applied.
<table><tr><td>A</td><td>B</td><td>C</td><td></td></tr><tr><td>1st O</td><td>O</td><td></td><td>O</td></tr><tr><td>2nd</td><td>O</td><td>O</td><td>O</td></tr><tr><td>3rd</td><td>O</td><td>O</td><td>O</td></tr></table>

Figure 17: Example question in the human evaluation of text–motion alignment. Given a text instruction and three simulation results for the same object, participants rank the motions by how closely they follow the requested behavior.