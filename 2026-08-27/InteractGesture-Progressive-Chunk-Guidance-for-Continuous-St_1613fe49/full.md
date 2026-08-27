# InteractGesture: Progressive Chunk Guidance for Continuous Streaming Co-Speech Gesture Control

Ekkasit Pinyoanuntapong<sup>1,2</sup>, Ajinkya Deogade<sup>1</sup>, Paul Streli<sup>1</sup>, Wenjing Zhang<sup>1</sup>, Joanna Materzynska<sup>1</sup>, Pu Wang<sup>2</sup>, Vittorio Ferrari<sup>1</sup>, and Jie Shen<sup>1</sup>

<sup>1</sup> Meta <sup>2</sup> University of North Carolina at Charlotte

![](images/d398e32945da291001dd1967b1c6a0bf916413847410dff94bbd30a857dd51d6.jpg)  
(a) Speech Condition  
(b) Speech + Sparse Joint Position Control

![](images/dd39d3a1c813b0a1c5b11618fce44b982766808a0b8d03cecd981742fac7d06f.jpg)  
(c) Speech + Dense Joint Trajectroy Control

![](images/e46bdbc6c56f5b1ab20d33cc0b3bf29c1871ef3576412a1087219764f6afcee4.jpg)  
(d) Speech + Pointing Direction Control  
Fig. 1: InteractGesture enables spatial gesture control for a speech-conditioned gesture generation model. (a) Speech Condition Only. (b) Joint Position Control: directs specified joints to point at described objects at designated frames. (c) Dense Joint Trajectory Control: guides joints along a continuous path for sophisticated gesture control. (d) Pointing Direction Control: orients the arm toward a target direction.

Abstract. Co-speech gesture generation has made significant progress toward realistic full-body motion from speaker audio, yet existing models lack fine-grained spatial controllability of individual joints. To address this, we introduce InteractGesture, a model-agnostic, inferencetime method for spatially controllable gesture generation. InteractGesture guides target latent estimates of a difusion sampler through a differentiable RVQ-VAE decoder, backpropagating spatial control gradients to adjust motion latents during sampling. A primary challenge in streaming co-speech generation is chunk-wise dependency: standard sequential inference freezes prior chunks, preventing spatial constraints in future chunks from adjusting preceding trajectories and causing boundary inconsistencies. To overcome this limitation, we propose Progressive Chunk Guidance, a chunk-window strategy that maintains an active set of editable chunk latents with staggered delays, enabling spatial constraints to propagate gradients backward across chunk boundaries during streaming generation. Experiments on the BEAT2 dataset show that InteractGesture improves multi-joint spatial control while preserving overall gesture quality. Furthermore, our approach supports diverse applications, including sparse joint positioning, dense joint trajectory control, and directional pointing. Our project page is available at https://exitudio.github.io/interactgesture-page.

E. Pinyoanuntapong et al.

Keywords: Co-speech gesture generation · Controllable motion synthesis · Continuous streaming generation

## 1 Introduction

Speech-driven gesture synthesis is an essential capability for digital humans, virtual agents, and embodied assistants. The task is challenging because movements must be synchronized with speech rhythm, semantically compatible with spoken content, and natural and coherent across the upper body, hands, lower body, and global root motion. Previous approaches have improved realism through larger datasets, increasingly powerful temporal models, expressive body representations, and generative gesture priors [17, 18, 23, 54]. Yet these methods are still primarily designed for unconstrained generation: given speech and optional style information, they produce plausible gestures, but do not provide fine-grained spatial controllability of individual body parts.

Spatial controllability is crucial for practical gesture editing and embodied interaction. An animator may want the right wrist to land near a prop, both hands to frame an object during an explanation, or an elbow trajectory to avoid an obstacle. An embodied agent may need to point toward an interface element while continuing to gesture naturally with speech. Efective spatial control should preserve speech-gesture synchrony while enabling precise direct control over generated motions (see Figure 1).

We introduce InteractGesture, a model-agnostic, inference-time method for spatially controllable co-speech gesture generation. InteractGesture treats a pretrained gesture generator as a denoising process whose target latent estimates can be guided during inference. At selected sampling steps, the predicted target latent estimate is decoded through a diferentiable RVQ-VAE decoder into body poses, a masked spatial loss is computed against user-specified control points, and gradients update the target latent estimate rather than the model parameters. The adjusted estimate is then used to denoise the current motion latent, allowing subsequent sampling steps to pull the guided state back toward the learned speech-conditioned gesture distribution.

Prior work in controllable motion synthesis [14, 20, 35, 43] demonstrates that generative models can follow spatial trajectories, keyframes, or programmable constraints. However, these methods operate on text-to-motion tasks, where the conditioning signal is static and the output is typically a short, ofline clip. Cospeech motion synthesis requires a fundamentally diferent inference paradigm: gestures must accompany long speech sequences whose continuous audio is typically processed in finite, overlapping chunks, requiring spatial controls to remain consistent across chunk boundaries.

For standard chunk-wise inference, chunk k must be fully generated before chunk k + 1 can use it as historical context. This strict sequential dependency complicates spatial control: reaching a target in chunk k + 1 may require chunk k to adjust its trajectory, but chunk k is already locked (see Figure 2a). Therefore, independently guiding chunks produces inconsistent control efects across boundaries.

![](images/4b4d1640d4afe441180482b59e38f0108ac184429941dffd086f7e07d2986fec.jpg)  
Fig. 2: Comparison of chunk-guidance schedules. (a) Sequential Chunk Guidance: Fully completes and freezes chunk k before initializing chunk k + 1. While causal, guidance from later chunks cannot adjust earlier motion. (b) Synchronous Chunk Guidance: Denoises all chunks simultaneously. Controls across the entire sequence influence overlapping context, but this requires full-sequence audio ofline. (c) Progressive Chunk Guidance (Ours): Introduces chunks with a staggered delay and continuously refreshes context frames from intermediate target estimates xˆ<sub>0</sub>. This maintains streaming compatibility while propagating spatial control across chunk boundaries.

To avoid this, given the full audio stream, all chunks can be denoised synchronously (see Figure 2b). This allows spatial controls to propagate gradients backward across earlier chunk latents during intermediate sampling steps, adjusting preceding trajectories before they are finalized.

To combine the online capability of chunk-wise generation with the global coherence of joint optimization, we propose Progressive Chunk Guidance. Rather than fixing past context or waiting for entire sequences to finalize, InteractGesture maintains an active window of editable chunk latents that are denoised concurrently. As continuous audio streams in, newly active chunks are introduced with staggered delays and conditioned on the evolving, guided latent estimates of preceding chunks (see Figure 2c). This continuous backward propagation of gradients across active chunks enables future spatial controls to adjust preceding motion, reconciling long-horizon controllability with online streaming execution.

Our method is model-agnostic at the control interface: it does not depend on specific training objective or generator architecture, provided the model exposes an editable target latent estimate (or equivalent latent update) and a diferentiable decoder mapping latents to joint poses. Our implementation uses the pretrained RVQ-VAE body-part decoders and SMPL-X joint recovery from a co-speech generator, which provides an end-to-end diferentiable gradient path from a joint-space control loss back to the latent estimates. The same representation supports diverse control tasks, including absolute scene-space targets, root-relative body-centric targets, sparse joint positioning, dense trajectory control, and directional pointing.

## Collectively, we contribute:

1. InteractGesture, a model-agnostic, inference-time framework for fine-grained spatial control of pretrained co-speech gesture generators.

2. Progressive Chunk Guidance, a chunk-window sampling strategy that maintains editable active chunks to enforce spatial control consistency across streaming chunk boundaries.

3. An evaluation on the BEAT2 dataset with quantitative and qualitative results, demonstrating the efectiveness of InteractGesture and its diverse applications, including sparse joint positioning, dense trajectory tracking, and directional pointing.

## 2 Related Work

## 2.1 Co-Speech Gesture Generation

Co-speech gesture generation synthesizes body motion from speech audio, text, speaker identity, and conversational context. Early work learned multimodal mappings for upper-body motion or video-based gesture transfer [2, 10, 41, 47, 48], while later skeleton-based methods improve speech-gesture alignment with stronger audio-motion representations, semantic cues, contrastive objectives, and conversational context [1, 16, 21, 22, 24, 25, 45, 50, 52]. Dataset and representation advances such as BEAT/CaMN [18, 19], BEAT2 [17], TalkSHOW [32, 46], EMAGE [17], and latent representations such as VQ-VAE/RVQ-VAE [23,31,49] further broaden the task from upper-body prediction to holistic body, hand, and facial generation.

Recent work also explores difusion and masked generative modeling [3, 5, 8, 54], personalization and emotion control [6, 15, 38, 51, 53], semantic planning [9], photoreal rendering [29], and dyadic or spatially aware interaction [27, 28, 30, 33, 34]. These methods broaden speech-driven generation, but their controls are primarily identity, style, emotion, semantic intent, or interaction context rather than direct 3D spatial targets for individual joints. For streaming generation, CaMN-style autoregressive models [19], MambaTalk [44], and LiveGesture [40] generate gestures from incoming audio. InteractGesture shares this streaming motivation, but focuses on fine-grained inference-time spatial control while chunks are still being generated.

## 2.2 Controllable Human Motion Generation

Text-to-motion control methods add trajectories, inbetweening, body-part masks, programmable constraints, or joint keyframes to ofline motion generation [11,

![](images/aaf45476a3ad934271276f5f30d7c55640eff755a88e036afb856ee422a0f488.jpg)  
Fig. 3: InteractGesture adds spatial control guidance to a pretrained co-speech gesture generation model at inference time. At every selected sampling step, the predicted target latent estimate is decoded into SMPL-X joints, a masked joint loss is minimized, and the edited target latent estimate is converted back into the sampler update used by the generator. The same mechanism supports absolute 3D targets, root-relative targets, and Progressive Chunk Guidance for continuous streaming generation from overlapping latent chunks.

14, 20, 35–37, 43]. These methods show that generative motion priors can follow spatial constraints, but they typically assume static text prompts, fixed sequence lengths, and ofline generation. Controllable co-speech work has begun to study pointing, trajectory control, multimodal examples, and object-aware interaction [4, 7, 39, 42], yet it does not provide fine-grained multi-joint 3D spatial control during continuous streaming generation.

At the algorithmic level, difusion and flow samplers can be steered with classifier guidance, classifier-free guidance, energy guidance, or optimization-based editing [12,13,26]. For human motion, diferentiable kinematics and body models allow geometric losses to be evaluated on decoded poses during sampling. InteractGesture adapts this principle to co-speech latent generation by optimizing target latent estimates through a diferentiable decoder, while Progressive Chunk Guidance propagates these edits across active streaming chunk boundaries.

## 3 Method

Our method synthesizes speech-synchronized full-body gestures while enforcing spatial constraints during continuous streaming generation. InteractGesture augments a pretrained chunk-level latent sampler conditioned on audio with a spatial control specification C = (H, M, Φ), where H stores target joint positions, M denotes active joint-frame masks, and Φ projects generated skeleton joints into the target coordinate space. Treating the underlying generator as a black box, our formulation backpropagates control gradients exclusively into the target latent estimates zˆ, leaving model parameters untouched.

## 3.1 Pretrained Latent Generator

Our pretrained motion generator synthesizes streaming motion as a sequence of motion chunks using a fixed-step DDIM sampler. Each generated motion chunk contains 128 pose frames. Because the underlying RVQ-VAE uses a temporal squeeze factor of 4, the sampler operates on latent windows of length 32. The latent channel is partitioned into three body-part streams:

$$
\mathbf { z } = [ \mathbf { z } ^ { \mathrm { { u p } } } , \mathbf { z } ^ { \mathrm { { h a n d } } } , \mathbf { z } ^ { \mathrm { { l o w } } } ] \in \mathbb { R } ^ { 3 2 \times 3 8 4 } ,\tag{1}
$$

where each stream $\mathbf { z } ^ { m } \in \mathbb { R } ^ { 3 2 \times 1 2 8 }$ corresponds to upper-body, hand, or lowerbody dynamics, respectively. The upper-body decoder reconstructs upper-body 6D rotations, the hand decoder reconstructs hand joint rotations, and the lowerbody decoder predicts lower-body rotations alongside root translation velocity. Global root translation r is recovered by cumulative summation of the predicted velocities across temporal frames.

These decoded body-part poses are mapped into their corresponding joint indices within the full 55-joint SMPL-X pose vector. Given body shape parameters $\beta$ and global root translation r, the end-to-end diferentiable SMPL-X layer computes 3D joint positions:

$$
\begin{array} { r } { \mathbf { J } ( \mathbf { z } ; \boldsymbol { \beta } ) = \mathrm { S M P L X } \left( \mathrm { D e c } _ { \mathrm { u p } } ( \mathbf { z } ^ { \mathrm { u p } } ) , \mathrm { D e c } _ { \mathrm { h a n d } } ( \mathbf { z } ^ { \mathrm { h a n d } } ) , \mathrm { D e c } _ { \mathrm { l o w } } ( \mathbf { z } ^ { \mathrm { l o w } } ) , \mathbf { r } , \boldsymbol { \beta } \right) . } \end{array}\tag{2}
$$

This fully diferentiable decoding path enables spatial control loss gradients to backpropagate directly from 3D joint space into the target latent estimates z, serving as the primary interface required for guidance.

## 3.2 Control Representation

InteractGesture specifies spatial controls using a target position tensor and a corresponding binary indicator mask. For a gesture sequence of length T frames and J = 55 SMPL-X joints, the spatial hint tensor is defined as:

$$
\mathbf { H } \in \mathbb { R } ^ { T \times J \times 3 } ,\tag{3}
$$

and the spatial mask tensor as:

$$
\mathbf { M } \in \{ 0 , 1 \} ^ { T \times J } ,\tag{4}
$$

where entries $\mathbf { M } _ { t , j } ~ = ~ 1$ denote active spatial constraints contributing to the control loss, while $\mathbf { M } _ { t , j } = 0$ indicates unconstrained joint-frame positions. This tensorized formulation naturally accommodates arbitrary spatio-temporal constraint configurations, ranging from sparse joint keyframes to continuous multijoint trajectories.

To enforce constraints during continuous streaming generation with overlapping windows, let P denote the pose chunk length (P = 128 frames) and S denote the number of historical context frames. The non-overlapping stride is given by $R = P - S$ . For the k-th chunk, spatial constraints from the sequencelevel hint tensor are sliced across local frame indices kR to $k R + P$ . For all noninitial chunks $( k > 0 )$ , constraints that overlap with the historical context region $( 1 \leq t \leq S )$ are masked out in M. This prevents double-counting optimization loss on previously generated context while preserving a unified, sequence-level control specification across streaming chunk boundaries.

## 3.3 Guidance on Target Latent Estimates

The DDIM sampler maintains intermediate latent states along a reverse denoising trajectory $\mathbf { x } _ { N } , \ldots , \mathbf { x } _ { 0 }$ over denoising steps $i \in \{ N , \ldots , 1 \}$ . At denoising step $i ,$ the generator predicts the clean target latent estimate $\hat { \mathbf { x } } _ { 0 }$ . Our spatial control loss evaluates this prediction across motion frames t and joints $j \colon$

$$
\mathcal { L } _ { \mathrm { c t r l } } ( \hat { \mathbf { x } } _ { 0 } ) = \frac { 1 } { 2 } \sum _ { b , t , j } \mathbf { M } _ { b , t , j } \left\| \boldsymbol { \varPhi } \left( \mathbf { J } ( \hat { \mathbf { x } } _ { 0 , b } ) _ { t , j } \right) - \mathbf { H } _ { b , t , j } \right\| _ { 2 } ^ { 2 } .\tag{5}
$$

The optimization variable is $\hat { \mathbf { x } } _ { 0 }$ itself. All generator parameters, RVQ-VAE decoders, and SMPL-X body model weights remain fixed.

We optimize $\hat { \mathbf { x } } _ { 0 }$ using the Adam optimizer. Because the number of active constraints varies across joint groups and keyframe densities, we normalize the base learning rate $\eta _ { 0 }$ by the active constraint count $N _ { \mathrm { a c t i v e } } \colon$

$$
\eta = \frac { \eta _ { 0 } } { g ( N _ { \mathrm { a c t i v e } } ) } ,\tag{6}
$$

where $g ( N _ { \mathrm { a c t i v e } } ) = N _ { \mathrm { a c t i v e } }$ for linear normalization, $g ( N _ { \mathrm { a c t i v e } } ) = \sqrt { N _ { \mathrm { a c t i v e } } }$ for square-root normalization, and $g ( N _ { \mathrm { a c t i v e } } ) = 1$ when unnormalized. $\mathrm { B y }$ default, we use linear normalization.

Following gradient optimization, the edited target estimate $\hat { \mathbf { x } } _ { 0 } ^ { * }$ is plugged directly into the standard denoising step formula to update the next latent state $\mathbf { x } _ { i - 1 }$ . While spatial edits temporarily push the estimated trajectory away from the unconstrained motion distribution, subsequent denoising steps sample from this guided state, allowing the learned score network to continually regularize the motion back toward the pretrained gesture prior. Because this framework relies solely on optimizing the estimated target state $\hat { \mathbf { x } } _ { 0 }$ , this formulation naturally extends to other generative paradigms, such as Flow Matching and standard DDPM samplers, without architectural modification.

## 3.4 Guidance Schedule

Not all reverse sampling steps require equal optimization efort. In early sampling steps, decoded intermediate joints are noisy, making spatial gradients unstable. In later sampling steps, the latent estimate converges toward the clean target manifold, enabling more precise spatial edits. To reflect this dynamic, we schedule the number of latent optimization iterations $K ( i )$ over the reverse sampler

index $i \in \{ N , \ldots , 1 \}$ as:

$$
K ( i ) = \left\lfloor K _ { \mathrm { e a r l y } } + ( K _ { \mathrm { l a t e } } - K _ { \mathrm { e a r l y } } ) \cdot \operatorname* { m a x } \left( 0 , \operatorname* { m i n } \left( 1 , \frac { \tau - i } { \tau } \right) \right) \right\rceil ,\tag{7}
$$

where τ denotes the late-start threshold in denoising step units, $K _ { \mathrm { e a r l y } }$ and $K _ { \mathrm { l a t e } }$ specify the initial and final iteration budgets, and $\lfloor \cdot \rceil$ denotes rounding to the nearest integer. This schedule concentrates spatial guidance toward the final stage of the sampling trajectory, where joint decoding is most reliable.

Additionally, InteractGesture supports an optional post-sampling refinement stage. After completing the final denoising step, the spatial control loss ${ \mathcal L } _ { \mathrm { c t r l } }$ can be optimized directly on the final clean target latent estimate $\mathbf { x } _ { \mathrm { 0 } }$ for $K _ { \mathrm { p o s t } }$ iterations. While this refinement further reduces spatial position error, excessive post-sampling optimization bypasses the generative sampler’s regularization steps and can diminish gesture naturalness.

## 3.5 Progressive Chunk Guidance

Continuous gesture generation processes audio incrementally via sliding temporal windows of stride $R = P - S$ frames. In standard unconstrained rollouts, chunk k is fully generated before its S historical context frames are copied to initialize chunk $k + 1$ . While suitable for real-time streaming, this rigid boundary poses a challenge for spatial control: a constraint in chunk $k + 1$ may require an earlier arm trajectory adjustment in chunk $k ,$ which is already frozen. Consequently, independently guided chunks can satisfy local targets while inducing motion discontinuities across chunk boundaries.

To address this, we compare three chunk-guidance schedules illustrated in Figure 2: a baseline Sequential strategy, an ofline Synchronous scheme, and our proposed Progressive Chunk Guidance.

Sequential chunk guidance. This causal baseline completes and freezes chunk k before using its final clean context frames to initialize chunk $k { + 1 }$ . Because chunk k is immutable when chunk $k + 1$ begins denoising, spatial guidance in chunk $k + 1$ cannot adjust the incoming motion, leading to abrupt trajectory changes across window boundaries.

Synchronous Chunk Guidance. When the entire audio stream is available ofline, all chunks are activated from the start and guided concurrently. Spatial loss gradients backpropagate through all active chunks simultaneously, allowing future constraints to adjust preceding trajectories across overlapping context regions. While this provides the highest spatial accuracy and trajectory smoothness, it cannot support streaming generation.

Progressive Chunk Guidance. For continuous streaming generation, future chunks arrive incrementally. Progressive Chunk Guidance introduces subsequent chunks with a staggered ofset of $\varDelta i$ sampler steps. While chunk $k + 1$ waits to initialize, preceding chunk k continues to optimize its target latent estimate $\hat { \mathbf { x } } _ { 0 } ^ { ( k ) }$ under spatial guidance. Crucially, at each active denoising step i, chunk k + 1 refreshes its S context frames from the latest guided estimate of chunk k. This creates a two-way control cascade: spatial loss gradients in chunk $k + 1$ flow backward into $\hat { \mathbf { x } } _ { 0 } ^ { ( k ) }$ across the decoder’s temporal receptive field, while updated estimates in chunk k dynamically refresh the context consumed by chunk $k + 1$ . This bidirectional propagation allows future spatial constraints to adjust preceding arm trajectories, enforcing smooth inter-chunk motion continuity within a fixed, user-defined streaming latency budget.

## 3.6 Applications

The spatial control tensor H supports three key editing modes: Joint Position Control, Figure 1(b), activates a small number of joint-frame targets, such as placing a wrist at a scene position during a spoken reference. This serves as our primary quantitative benchmark. Dense Trajectory Control, Figure 1(c), activates targets across continuous frame sequences to enforce hand or elbow paths using a temporally dense mask M. Pointing Direction Control, Figure 1(d), maps pointing rays or arm vector angles to target coordinates along a directional vector, enabling expressive pointing while maintaining speech rhythm.

## 4 Experiments

## 4.1 Dataset and Evaluation Setup

Base Generator & Feature Space. We conduct evaluation experiments on the BEAT2 dataset using GestureLSM as the pretrained co-speech motion generator. All GestureLSM generator weights and RVQ-VAE body-part decoders remain frozen throughout evaluation. Motion is synthesized at 30 FPS using 128-frame pose chunks with a fixed-step DDIM reverse sampling schedule.

Control Benchmarks & Settings. We evaluate spatial control using absolute 3D joint targets extracted from ground-truth SMPL-X poses. Evaluation covers three constraint densities (1, 2, and 5 keyframe targets per chunk) across five joint configurations: left wrist, right wrist, left elbow, right elbow, and all four joints combined, yielding 15 distinct control settings. Guidance optimization executes $K _ { \mathrm { l a t e } } = 3 0$ iterations during late-stage reverse sampling steps, followed by $K _ { \mathrm { p o s t } } = 4 0$ post-sampling refinement iterations on target latents $\hat { \mathbf { x } } _ { 0 }$

Evaluated Schedules. We compare three chunk-guidance schedules: (i) Sequential Chunk Guidance (causal baseline freezing prior chunks), (ii) Synchronous Chunk Guidance (ofline upper bound guiding all chunks concurrently), and (iii) Progressive Chunk Guidance (our proposed streaming schedule using a staggered ofset of $\varDelta i = 1$ sampling wave).

## 4.2 Control Protocol

The evaluation benchmark is constructed directly from ground-truth SMPL-X poses. For each sequence, target global translations and joint rotations are converted to 3D joint positions, which are then sampled according to the selected joint group and keyframe density. All reported metrics are aggregated over all controlled joints and active constraint frames.

## 4.3 Metrics

We evaluate motion quality and spatial control accuracy:

– Fréchet Gesture Distance (FGD ↓) measures the distributional feature distance to ground-truth motion.

– Beat Consistency (BC →) evaluates the temporal synchronization between speech beats and motion beats (closer to Ground-Truth is preferred).

– Diversity (→) evaluates the global pose variation across generated sequences (closer to Ground-Truth is preferred).

– Average Control Error (Avg. Err. cm ↓) is the mean Euclidean distance (in centimeters) between controlled joints and spatial targets.

– Location Error (Loc. Error ↓) is the fraction of keyframe targets with Euclidean error exceeding 10 cm.

– Trajectory Error (Traj. Error ↓) is the fraction of active chunks containing at least one target exceeding 10 cm error.

## 4.4 Implementation Details

The guidance function receives latent tensors from the sampler, re-formats them into temporal chunk windows, and refreshes overlapping context from the latest guided estimates when needed. Active latents are decoded into body-part streams to extract joint coordinates for loss evaluation. For root-relative control, root rotations are applied prior to loss calculation.

Optimization is performed on latent tensors using Adam. We configure $K _ { \mathrm { e a r l y } } =$ $0 , K _ { \mathrm { l a t e } } = 3 0$ , and start threshold τ = 100, followed by $K _ { \mathrm { p o s t } } = 4 0$ post-sampling refinement steps. We use linear step-size normalization $( \eta = \eta _ { 0 } / N _ { \mathrm { a c t i v e } } )$ to normalize gradient magnitudes between single-joint controls and dense multi-joint groups. Without this normalization, dense multi-joint controls produce excessively large gradient updates that distort gesture naturalness.

## 4.5 Main Quantitative Results

Table 1 compares InteractGesture across all three chunk-guidance schedules as well as against alternative spatial control paradigms. Specifically, we compare against a post-hoc Inverse Kinematics (IK) baseline, which analytically solves joint rotations post-generation to reach spatial targets without respecting the speech-conditioned gesture distribution, and a Chunk-Wise ControlNet baseline, which incorporates spatial conditioning via a trainable feed-forward adapter applied independently to each chunk. As shown in Table 1, IK preserves high motion diversity but severely degrades gesture naturalness (FGD 1.043) and produces large average control errors (19.97 cm) because geometric post-fitting introduces unnatural body poses. Chunk-Wise ControlNet achieves better FGD (0.491), but fails to reach keyframe targets (46.16 cm average error; 97.6% location failure rate).

Table 1: Main spatial control comparison on BEAT2 evaluation. Lower is better for FGD, trajectory/location errors, and average control error. For metrics marked with →, closer to Ground-Truth is better. Red bold marks the best generated method in each column, and blue underlined marks the second best.
<table><tr><td>Method</td><td colspan="6">FGD (↓) BC (→) Diversity (→) Traj. Error (↓) Loc. Error (↓) Avg. Err. cm (↓)</td></tr><tr><td>Ground-Truth</td><td></td><td>0.703</td><td>11.970</td><td></td><td></td><td></td></tr><tr><td>Inverse Kinematic</td><td>1.043</td><td>0.723</td><td>16.379</td><td>0.582</td><td>0.453</td><td>19.973</td></tr><tr><td>Chunk-Wise ControlNet</td><td>0.491</td><td>0.778</td><td>12.034</td><td>0.996</td><td>0.976</td><td>46.159</td></tr><tr><td>Sequential Chunk Guidance</td><td>0.690</td><td>0.816</td><td>13.845</td><td>0.574</td><td>0.470</td><td>11.701</td></tr><tr><td>Synchronous Chunk Guidance (Ours)</td><td>0.442</td><td>0.750</td><td>12.234</td><td>0.092</td><td>0.100</td><td>4.669</td></tr><tr><td>Progressive Chunk Guidance (Ours)</td><td>0.431</td><td>0.762</td><td>11.760</td><td>0.267</td><td>0.161</td><td>6.335</td></tr></table>

In contrast, Sequential Chunk Guidance improves spatial accuracy (11.70 cm error) through test-time latent optimization, though freezing prior chunks still prevents future constraints from adjusting incoming motion trajectories. Synchronous Chunk Guidance achieves the strongest overall accuracy in the ofline setting (4.67 cm average error; 10.0% location error). Finally, our proposed Progressive Chunk Guidance enables continuous streaming execution while achieving an average error of 6.34 cm and improving gesture naturalness (FGD 0.431, Diversity 11.760) over Synchronous guidance. Overall, optimizing target latent estimates through inference-time guidance strikes a superior balance between spatial control precision and gesture naturalness compared to post-hoc geometric fitting or feed-forward chunk-level conditioning.

## 4.6 Qualitative Results

Figure 4 qualitatively compares the visualization of the baselines and our method under identical spatial keyframe controls. The Inverse Kinematics baseline directly calculate joint rotation from control position after generation, without considering the speech-conditioned motion distribution. As a result, it can place controlled joints closer to the targets, but the surrounding motion often looks less natural because the correction is not regularized by the learned gesture prior.

Chunk-Wise ControlNet applies control independently within each generated chunk. Its joint placement is less accurate because a chunk-wise perturbation has limited access to neighboring chunk motion and future control positions, preventing corrections near chunk boundaries from being coordinated over a longer gesture trajectory. Sequential Chunk Guidance improves over chunk-wise control but freezes each previous chunk before optimizing the next one. Therefore, control targets in chunk k+1 cannot propagate influence back to chunk k, even when the motion should be adjusted across the chunk boundary. InteractGesture with Progressive Chunk Guidance keeps overlapping chunks editable as new chunks are introduced. This allows later controls to influence the active previous chunk through refreshed context, producing both accurate spatial control and natural co-speech motion in Figure 4.

![](images/d9aa944fd799ee5369986332c9bfdae9c8a14afd057ef1781ba6a329254a3035.jpg)  
Fig. 4: Main qualitative result. InteractGesture uses spatial controls to steer co-speech gestures toward target hand and arm positions while preserving the speech-conditioned gesture prior. The visualization shows the guided generations alongside target positions, demonstrating whether the generated wrists and elbows adhere to the requested spatial constraints.

## 5 Ablation Study

We perform an ablation study on the BEAT2 dataset under absolute 3D spatial control to evaluate the individual contributions of our guidance components, optimization scales, and execution schedules (Table 2).

Table 2: Ablation study on spatial guidance components and optimization scales on BEAT2. Panel (a) evaluates guidance modules at scale 0.5. Panel (b) evaluates guidance scale sweeps. Delay 0 corresponds to Synchronous Chunk Guidance. Delay 1 corresponds to Progressive Chunk Guidance (∆i = 1).
<table><tr><td>Setting</td><td>Sampling Post Scale Guidance Refine</td><td></td><td>FGD (↓) BC (→) Div (→) Traj10 (↓) Loc10 (↓) Avg. cm (↓)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">(a) Control Component Ablation</td></tr><tr><td colspan="9">Delay 0: Synchronous Chunk Guidance</td></tr><tr><td>Uncontrolled Baseline</td><td>0.5</td><td>X</td><td>X</td><td>0.439</td><td>0.724</td><td>12.243</td><td>0.984</td><td>0.965</td><td>59.308</td></tr><tr><td>Sampling Guidance only</td><td>0.5</td><td>√</td><td>X</td><td>0.438</td><td>0.744</td><td>12.175</td><td>0.426</td><td>0.427</td><td>11.471</td></tr><tr><td>Post-Refinement only</td><td>0.5</td><td>X</td><td>√</td><td>0.438</td><td>0.743</td><td>12.195</td><td>0.338</td><td>0.338</td><td>9.428</td></tr><tr><td>Full Guidance (Ours)</td><td>0.5</td><td>√</td><td>√</td><td>0.442</td><td>0.750</td><td>12.234</td><td>0.092</td><td>0.100</td><td>4.669</td></tr><tr><td colspan="10">Delay 1: Progressive Chunk Guidance</td></tr><tr><td>Uncontrolled Baseline</td><td>0.5</td><td>X</td><td>X</td><td>0.418</td><td>0.726</td><td>12.320</td><td>0.992</td><td>0.960</td><td>64.984</td></tr><tr><td>Sampling Guidance only</td><td>0.5</td><td>√</td><td>X</td><td>0.434</td><td>0.766</td><td>12.151</td><td>0.733</td><td>0.484</td><td>12.740</td></tr><tr><td>Post-Refinement only</td><td>0.5</td><td>X</td><td>√</td><td>0.417</td><td>0.746</td><td>12.272</td><td>0.524</td><td>0.524</td><td>12.361</td></tr><tr><td>Full Guidance (Ours)</td><td>0.5</td><td>√</td><td>√</td><td>0.431</td><td>0.762</td><td>11.760</td><td>0.267</td><td>0.161</td><td>6.335</td></tr><tr><td colspan="10">(b) Guidance Scale Sweep</td></tr><tr><td colspan="10">Delay 0: Synchronous Chunk Guidance</td></tr><tr><td>Scale 0.1</td><td>0.1</td><td>√</td><td>√</td><td>0.435</td><td>0.739</td><td>12.224</td><td>0.299</td><td>0.376</td><td>10.464</td></tr><tr><td>Scale 0.5</td><td>0.5</td><td>√</td><td>√</td><td>0.442</td><td>0.750</td><td>12.234</td><td>0.092</td><td>0.100</td><td>4.669</td></tr><tr><td>Scale 1.0</td><td>1.0</td><td>√</td><td>√</td><td>0.452</td><td></td><td>0.766 11.970</td><td>0.080</td><td>0.050</td><td>4.072</td></tr><tr><td colspan="10">Delay 1: Progressive Chunk Guidance</td></tr><tr><td>Scale 0.1</td><td>0.1</td><td>√</td><td>√</td><td>0.417</td><td>0.743</td><td>12.257</td><td>0.283</td><td>0.307</td><td>8.900</td></tr><tr><td>Scale 0.5</td><td>0.5</td><td>√</td><td>√</td><td>0.431</td><td>0.762</td><td>11.760</td><td>0.267</td><td>0.161</td><td>6.335</td></tr><tr><td>Scale 1.0</td><td>1.0</td><td>√</td><td>√</td><td>0.463</td><td>0.778</td><td>11.283</td><td>0.355</td><td>0.191</td><td>7.308</td></tr></table>

## 5.1 Impact of Guidance Components

Table 2(a) evaluates four guidance configurations: Uncontrolled Baseline, Sampling Guidance only, Post-Refinement only, and Full Guidance (Ours). Each configuration is reported for both Delay 0 and Delay 1 so that the component efects can be compared under the same execution settings.

Without guidance, spatial control fails in both schedules, with average errors of 59.308 cm for Delay 0 and 64.984 cm for Delay 1. Sampling Guidance only reduces the error to 11.471 cm and 12.740 cm, respectively, showing that optimizing target latent estimates during denoising provides the main correction signal. Post-Refinement only also improves accuracy, reaching 9.428 cm and 12.361 cm, but it lacks the iterative sampler regularization available during denoising.

Full Guidance combines both components and gives the best control accuracy within each delay setting, reducing the average error to 4.669 cm for Delay 0 and 6.335 cm for Delay 1. This confirms that Sampling Guidance provides coarse trajectory alignment during denoising, while Post-Refinement fine-tunes the final target latent estimates where decoded joint coordinates are most reliable.

## 5.2 Guidance Scale Sensitivity

Table 2(b) evaluates guidance step scales across three settings: 0.1, 0.5, and 1.0. In the ofline Synchronous setting, increasing guidance scale monotonically improves spatial precision, dropping average control error from 10.46 cm at scale 0.1 down to 4.07 cm at scale 1.0. For continuous streaming execution under Progressive Chunk Guidance, scale 0.5 provides the optimal trade-of between spatial precision (6.34 cm) and gesture naturalness (FGD 0.431), whereas scale 1.0 yields slightly higher control error (7.31 cm).

## 5.3 Efect of Progressive Chunk Guidance

Progressive Chunk Guidance enables spatial control during continuous streaming generation. Section 3.5 defines the two execution schedules evaluated in Table 2: Delay 0 corresponds to ofline Synchronous Chunk Guidance, which must wait for the full speech sequence so that all chunks can be optimized concurrently; Delay 1 corresponds to Progressive Chunk Guidance, which introduces chunks with a one-wave delay (∆i = 1) and keeps generation compatible with streaming audio.

The ofline Synchronous setting provides the strongest control precision because future controls are available when earlier chunks are still editable. With Full Guidance, it reaches 4.669 cm average error, Traj10 0.092, and Loc10 0.100. Progressive Chunk Guidance maintains comparable control accuracy under the streaming constraint, with 6.335 cm average error, Traj10 0.267, and Loc10 0.161.

Progressive Chunk Guidance also maintains high generation quality: its FGD is slightly lower than the Synchronous setting (0.431 vs. 0.442), with Diversity 11.760 compared with 12.234. Thus, Synchronous Chunk Guidance serves as an ofline accuracy reference, while Progressive Chunk Guidance preserves most of the control benefit without requiring the full speech sequence in advance.

## 6 Conclusion

We presented InteractGesture, a model-agnostic inference-time method for spatial control of co-speech gesture generation. The method guides a pretrained co-speech generative sampler by optimizing predicted target latent estimates through a diferentiable RVQ-VAE decoder. We proposed Progressive Chunk Guidance to support continuous streaming generation: instead of freezing each chunk before the next one is guided, it keeps overlapping chunk latents editable, introduces future chunks with a staggered delay, and refreshes context from the latest guided estimates. This lets spatial constraints in later chunks influence preceding motion across chunk boundaries without requiring the full speech sequence in advance. When the complete audio stream is available, our synchronous setting can produce high-quality motion with precise joint control. InteractGesture supports diverse applications including sparse joint control, dense joint trajectory control, and pointing-direction control. Experiments on BEAT2 demonstrate that latent guidance significantly reduces joint errors while preserving speech-conditioned gesture naturalness.

## References

1. Ao, T., Gao, Q., Lou, Y., Chen, B., Liu, L.: Rhythmic gesticulator: Rhythmaware co-speech gesture synthesis with hierarchical neural embeddings. ACM Trans. Graph. 41(6) (Nov 2022). https://doi.org/10.1145/3550454.3555435, https://doi.org/10.1145/3550454.3555435

2. Chan, C., Ginosar, S., Zhou, T., Efros, A.A.: Everybody dance now. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (October 2019)

3. Chen, B., Li, Y., Ding, Y.X., Shao, T., Zhou, K.: Enabling Synergistic Full-Body Control in Prompt-Based Co-Speech Motion Generation. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 6774–6783 (2024). https: //doi.org/10.1145/3664647.3680847, https://doi.org/10.1145/3664647. 3680847

4. Chen, B., Li, Y., Zheng, Y., Ding, Y.X., Zhou, K.: Motion-example-controlled co-speech gesture generation leveraging large language models. In: Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. SIGGRAPH Conference Papers ’25, Association for Computing Machinery, New York, NY, USA (2025). https://doi.org/10. 1145/3721238.3730611, https://doi.org/10.1145/3721238.3730611

5. Chen, J., Liu, Y., Wang, J., Zeng, A., Li, Y., Chen, Q.: Difsheg: A difusion-based approach for real-time speech-driven holistic 3d expression and gesture generation. In: CVPR (2024)

6. Chhatre, K., Daněček, R., Athanasiou, N., Becherini, G., Peters, C., Black, M.J., Bolkart, T.: AMUSE: Emotional speech-driven 3D body animation via disentangled latent difusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1942–1953 (June 2024), https://amuse.is. tue.mpg.de

7. Deichler, A., Alexanderson, S., Beskow, J.: Incorporating spatial awareness in datadriven gesture generation for virtual agents. In: Proceedings of the 24th ACM International Conference on Intelligent Virtual Agents. IVA ’24, Association for Computing Machinery, New York, NY, USA (2024). https://doi.org/10.1145/ 3652988.3673936, https://doi.org/10.1145/3652988.3673936

8. Deichler, A., Mehta, S., Alexanderson, S., Beskow, J.: Difusion-based co-speech gesture generation using joint text and audio representation. In: Proceedings of the 25th International Conference on Multimodal Interaction. p. 755–762. ICMI ’23, Association for Computing Machinery, New York, NY, USA (2023). https://doi. org/10.1145/3577190.3616117, https://doi.org/10.1145/3577190.3616117

9. Gao, N., Zhao, Z., Zeng, Z., Zhang, S., Weng, D., Bao, Y.: Gesgpt: Speech gesture synthesis with text parsing from chatgpt. IEEE Robotics and Automation Letters 9(3), 2718–2725 (2024). https://doi.org/10.1109/LRA.2024.3359544

10. Ginosar, S., Bar, A., Kohavi, G., Chan, C., Owens, A., Malik, J.: Learning individual styles of conversational gesture. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2019)

11. Guo, C., Mu, Y., Javed, M.G., Wang, S., Cheng, L.: MoMask: Generative masked modeling of 3d human motions. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 1900–1910 (2024). https://doi. org/10.1109/CVPR52733.2024.00186

12. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 6840–6851. Curran Associates,

Inc. (2020), https://proceedings.neurips.cc/paper\_files/paper/2020/file/ 4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

13. Ho, J., Salimans, T.: Classifier-free difusion guidance. In: NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications (2021), https: //openreview.net/forum?id=qw8AKxfYbI

14. Karunratanakul, K., Preechakul, K., Suwajanakorn, S., Tang, S.: Guided motion difusion for controllable human motion synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2151–2162 (2023)

15. Li, X., Lin, J., Zhang, B., Qi, Y., Wang, C., He, G.: Emodifges: Emotionaware co-speech holistic gesture generation with progressive synergistic difusion. Computer Graphics Forum 44, e70261 (2025). https://doi.org/https: //doi.org/10.1111/cgf.70261, https://onlinelibrary.wiley.com/doi/abs/ 10.1111/cgf.70261

16. Liu, H., Yang, X., Akiyama, T., Huang, Y., Li, Q., Kuriyama, S., Taketomi, T.: TANGO: Co-speech gesture video reenactment with hierarchical audio motion embedding and difusion interpolation. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id= LbEWwJOufy

17. Liu, H., Zhu, Z., Becherini, G., Peng, Y., Su, M., Zhou, Y., Zhe, X., Iwamoto, N., Zheng, B., Black, M.J.: Emage: Towards unified holistic co-speech gesture generation via expressive masked audio gesture modeling. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1144–1154 (June 2024)

18. Liu, H., Zhu, Z., Iwamoto, N., Peng, Y., Li, Z., Zhou, Y., Bozkurt, E., Zheng, B.: Beat: A large-scale semantic and emotional multi-modal dataset for conversational gestures synthesis. In: European Conference on Computer Vision (ECCV) (2022)

19. Liu, H., Zhu, Z., Iwamoto, N., Peng, Y., Li, Z., Zhou, Y., Bozkurt, E., Zheng, B.: Beat: A large-scale semantic and emotional multi-modal dataset for conversational gestures synthesis. arXiv preprint arXiv:2203.05297 (2022)

20. Liu, H., Zhan, X., Huang, S., Mu, T.J., Shan, Y.: Programmable motion generation for open-set motion control tasks. In: 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1399–1408 (2024). https://doi. org/10.1109/CVPR52733.2024.00139

21. Liu, L., Ghaleb, E., Özyürek, A., Yumak, Z.: SemGes: Semantics-aware co-speech gesture generation using semantic coherence and relevance learning (2025)

22. Liu, P., Liu, H., Song, L., Corso, J.J., Xu, C.: Intentional gesture: Deliver your intentions with gestures for speech. arXiv preprint arXiv:2505.15197 (2025)

23. Liu, P., Song, L., Huang, J., Xu, C.: Gesturelsm: Latent shortcut based co-speech gesture generation with spatial-temporal modeling. In: IEEE/CVF International Conference on Computer Vision (2025)

24. Liu, P., Zhang, P., Kim, H., Garrido, P., Sharpio, A., Olszewski, K.: Contextual gesture: Co-speech gesture video generation through context-aware gesture representation (2025), https://arxiv.org/abs/2502.07239

25. Liu, X., Wu, Q., Zhou, H., Xu, Y., Qian, R., Lin, X., Zhou, X., Wu, W., Dai, B., Zhou, B.: Learning hierarchical cross-modal association for co-speech gesture generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10462–10472 (June 2022)

26. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum?id=XVjTT1nw5z

27. Mughal, M.H., Dabral, R., Demberg, V., Theobalt, C.: Miburi: Towards expressive interactive gesture synthesis. In: Computer Vision and Pattern Recognition (CVPR) (2026)

28. Nagano, K., Liu, H., Park, S., Li, T., Mazumdar, A., Jacobsen, C., Wang, S., Stengel, M., Roy, R., Cheung, K.C., See, S., De Mello, S.: Dyaplex: Full-duplex speechmotion model for dyadic interaction. arXiv preprint arXiv:2606.03874 (2026)

29. Ng, E., Romero, J., Bagautdinov, T., Bai, S., Darrell, T., Kanazawa, A., Richard, A.: From audio to photoreal embodiment: Synthesizing humans in conversations. In: ArXiv (2024)

30. Ng, E., Zhang, S., Chen, Z., Zollhoefer, M., Richard, A.: Sarah: Spatially aware real-time agentic humans (2026), https://arxiv.org/abs/2602.18432

31. van den Oord, A., Vinyals, O., kavukcuoglu, k.: Neural discrete representation learning. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017), https://proceedings.neurips.cc/ paper \_ files / paper / 2017 / file / 7a98af17e63a0ac09ce2e96d03992fbc - Paper . pdf

32. Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A.A.A., Tzionas, D., Black, M.J.: Expressive body capture: 3D hands, face, and body from a single image. In: Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). pp. 10975–10985 (2019)

33. Peng, Y., Song, J.T., Jung, S., Liu, R., Liu, H., Chu, X., Liu, R., Wu, E., Koike, H., Kitani, K.: Dyadit: A multi-modal difusion transformer for socially favorable dyadic gesture generation (2026), https://arxiv.org/abs/2602.23165

34. Peng, Z., Fan, Y., Wu, H., Wang, X., Liu, H., He, J., Fan, Z.: Dualtalk: Dual-speaker interaction for 3d talking head conversations. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2025)

35. Pinyoanuntapong, E., Saleem, M., Karunratanakul, K., Wang, P., Xue, H., Chen, C., Guo, C., Cao, J., Ren, J., Tulyakov, S.: Maskcontrol: Spatio-temporal control for masked motion synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9955–9965 (2025)

36. Pinyoanuntapong, E., Saleem, M.U., Wang, P., Lee, M., Das, S., Chen, C.: BAMM: Bidirectional autoregressive motion model. In: ECCV (2024), https://www.ecva. net/papers/eccv\_2024/papers\_ECCV/papers/02316.pdf

37. Pinyoanuntapong, E., Wang, P., Lee, M., Chen, C.: Mmm: Generative masked motion model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

38. Qi, X., Liu, C., Li, L., Hou, J., Xin, H., Yu, X.: Emotiongesture: Audio-driven diverse emotional co-speech 3d gesture generation (2023)

39. Rajan, S., Bhosikar, K., Sharma, C.: Interactalker: Prompt-based human-object interaction with co-speech gesture generation. In: 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1438–1447. IEEE (2026)

40. Saleem, M.U., Patel, M.J., Pinyoanuntapong, E., Qin, Z., Yang, L., Xue, H., Helmy, A., Chen, C., Wang, P.: Livegesture streamable co-speech gesture generation model. arXiv preprint arXiv:2604.10927 (2026)

41. Wang, T.C., Liu, M.Y., Zhu, J.Y., Tao, A., Kautz, J., Catanzaro, B.: Highresolution image synthesis and semantic manipulation with conditional gans. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)

42. Wang, W., Gao, D., Hu, Y.: Synthesizing eficient trajectory-controllable co-speech gesture with latent consistency model. In: ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5 (2025). https://doi.org/10.1109/ICASSP49660.2025.10890673

43. Xie, Y., Jampani, V., Zhong, L., Sun, D., Jiang, H.: Omnicontrol: Control any joint at any time for human motion generation. In: The Twelfth International Conference on Learning Representations (2024)

44. Xu, Z., Lin, Y., Han, H., Yang, S., Li, R., Zhang, Y., Li, X.: Mambatalk: Eficient holistic gesture synthesis with selective state space models. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) Advances in Neural Information Processing Systems. vol. 37, pp. 20055–20080. Curran Associates, Inc. (2024). https://doi.org/10.52202/ 079017- 0633, https://proceedings.neurips.cc/paper\_files/paper/2024/ file/23c9c94227f937cfb50592a15e7fbb63-Paper-Conference.pdf

45. Xu, Z., Zhang, Y., Yang, S., Li, R., Li, X.: Chain of generation: Multi-modal gesture synthesis via cascaded conditional control (2024)

46. Yi, H., Liang, H., Liu, Y., Cao, Q., Wen, Y., Bolkart, T., Tao, D., Black, M.J.: Generating holistic 3d human motion from speech. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 469–480 (June 2023)

47. Yoon, Y., Cha, B., Lee, J.H., Jang, M., Lee, J., Kim, J., Lee, G.: Speech gesture generation from the trimodal context of text, audio, and speaker identity. ACM Trans. Graph. 39(6) (Nov 2020). https://doi.org/10.1145/3414685.3417838, https://doi.org/10.1145/3414685.3417838

48. Yoon, Y., Ko, W.R., Jang, M., Lee, J., Kim, J., Lee, G.: Robots learn social skills: End-to-end learning of co-speech gesture generation for humanoid robots. In: 2019 International Conference on Robotics and Automation (ICRA). pp. 4303–4309 (2019). https://doi.org/10.1109/ICRA.2019.8793720

49. Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., Tagliasacchi, M.: Soundstream: An end-to-end neural audio codec. Transactions on Audio, Speech and Language Processing (2021), https://arxiv.org/abs/2107.03312

50. Zhang, P., Liu, P., Garrido, P., Kim, H., Chaudhuri, B.: KinMo: Kinematic-aware Human Motion Understanding and Generation (2025)

51. Zhang, X., Cai, Y., Li, K., Yang, K., Zhou, Y., Li, Z., Chu, X., Zhang, J., Liu, H.: Personagesture: Single-reference co-speech gesture personalization for unseen speakers. arXiv preprint arXiv:2605.06064 (2026)

52. Zhang, Z., Ao, T., Zhang, Y., Gao, Q., Lin, C., Chen, B., Liu, L.: Semantic gesticulator: Semantics-aware co-speech gesture synthesis. ACM Transactions on Graphics (TOG) 43(4), 1–17 (2024)

53. Zhao, J., Liang, Q., Wang, Y.: Personagest: Personalized co-speech gesture generation with semantic-guided hierarchical motion representation. arXiv preprint arXiv:2605.07252 (2026)

54. Zhao, W., Hu, L., Zhang, S.: Difugesture: Generating human gesture from twoperson dialogue with difusion models (2023), https://openreview.net/forum? id=swc28UDR8Wk