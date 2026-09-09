# ACTIONSPLICE: IN-FLIGHT ACTION EDITING FOR INTERACTIVE WORLD MODELS

Pardis Taghavi Tingyu Guo Jonas Lossner Gaurav Pandey Reza Langari

Texas A&M University

Code Checkpoints Project Page

Abstract. Chunk-autoregressive video world models typically condition each generated chunk on one action. An action received during sampling must therefore wait for the next chunk, condition future solver evaluations on a state produced under the previous action, or trigger rollback that repeats completed evaluations. We introduce ActionSplice, an inference framework that formulates this problem as Counterfactual State Transport (CST). A lightweight corrector transports the interrupted backbone-native representation toward the matched state induced by the revised action at the same solver step. The world model and sampler remain frozen, and sampling resumes without replaying completed evaluations. The retargeting variant CST<sub>R</sub> updates the entire active chunk, while the temporal-splicing variant CST preserves a temporal prefix and updates only the suffix. Across minWM–Wan Action2V and HY-WM1.5, CST<sub>R</sub> reduces rollback-relative LPIPS by 61.5% and 75.9% relative to direct condition swapping. CST reduces suffix LPIPS by 56.1% and 77.5%, respectively, while providing 2.73× and 1.69× pixel-ready speedups over waiting. Under the HY-WorldPlay protocol, $\mathrm { C S T } _ { R }$ obtains 25.66 PSNR, 0.6902 SSIM, and 0.1337 LPIPS against the original rollout.

## 1 INTRODUCTION

Interactive video world models must be both fast and responsive. Recent diffusion world models support action-conditioned visual simulation and real-time streaming generation (Valevski et al., 2025; Wang et al., 2026; Zhao et al., 2026a). Yet high frame throughput does not guarantee low control latency. A chunk-autoregressive model generates several latent frames under one action and denoises them through K evaluation steps before display. If a new action arrives after solver step r, the active state has already been shaped by the previous action. Waiting for the next chunk delays the visible response and limits control changes to the chunk rate. We study interruptibility, the ability to incorporate an action received during sampling into the active chunk while retaining completed computation and committed history.

The main obstacle is state misalignment. Waiting preserves the current trajectory but postpones the action. Direct condition swapping supplies the updated action to future evaluations, but leaves the active state on the trajectory induced by the previous action. Full rollback reconstructs the desired trajectory, but repeats the first r solver evaluations. This suggests a more useful target. At the interruption step, we seek the matched counterfactual state $s _ { r } ^ { \star }$ that rollback would have reached under the revised action, with the same initial state, committed history, solver schedule, and stochastic inputs when applicable. A valid in-flight edit must approximate this state at the same solver step, preserve the non-editable portion of the transport representation, make the revised action visible within the active chunk, and avoid replaying completed evaluations.

We introduce ActionSplice, an inference framework built around Counterfactual State Transport (CST). CST learns a same-step correction from matched rollback pairs. Given an interrupted old action state, a lightweight corrector predicts a masked residual in the backbone-native transport representation, reconstructs a valid solver state, and resumes the remaining $K - r$ evaluations with the world model and sampler frozen. The retargeting variant $\mathrm { C S T } _ { R }$ applies the revised action to the entire active chunk. The temporal-splicing variant $\mathrm { C S T } _ { T }$ preserves an old-action prefix and applies the revised action only to the remaining suffix, allowing control to change at finer temporal granularity than one action per chunk. Full rollback is used only to construct training targets and evaluation references; it is not executed during ActionSplice inference.

CST does not assume a shared latent space or transfer corrector weights across models. Instead, it defines the same counterfactual intervention in each backbone’s resumable state space: cleanprediction transport for minWM–Wan Action2V (Zhao et al., 2026a) and Euler solver-state transport for HY-WM1.5 (HunyuanWorld, 2025). We train a separate corrector for each backbone and variant because the variants impose different intervention structures: $\mathrm { C S T } _ { R }$ transports the entire active state for asynchronous action revisions from a user or planner, whereas $\mathrm { C S T } _ { T }$ handles a scheduled action transition at a specified boundary within the active chunk.

Our contributions are threefold. First, we formulate in-flight action editing as matched counterfactual transport between resumable sampler states. Second, we introduce whole-chunk retargeting and within-chunk temporal splicing for frozen world models. Third, we establish a matched evaluation protocol across two independently implemented backbones that covers action receipt steps, directed action transitions, temporal boundaries, and repeated interruptions. Relative to direct condition swapping, CST reduces rollback-relative LPIPS by 61.5% on minWM–Wan Action2V and 75.9% on HY-WM1.5. $\mathrm { C S T } _ { T }$ reduces suffix LPIPS by 56.1% and 77.5%, respectively, while providing 2.73× and 1.69× pixel-ready speedups over waiting. On the HY-WorldPlay benchmark, $\mathrm { C S T } _ { R }$ obtains 25.66 PSNR, 0.6902 SSIM, and 0.1337 LPIPS against the original rollout, improving on the best reported comparison value for each metric.

## 2 RELATED WORK

Interactive video world models. Diffusion-based world modeling demonstrated actionconditioned visual simulation in interactive game environments (Alonso et al., 2024; Valevski et al., 2025). Recent systems target open-ended, camera-controllable streaming generation. WorldPlay uses dual action representations and reconstituted context memory for long-term geometric consistency (Sun et al., 2025), while Matrix-Game 3.0 combines long-horizon memory with multi-segment autoregressive distillation (Wang et al., 2026). HY-World 1.5 targets real-time interactive generation with geometric consistency (HunyuanWorld, 2025), while minWM and BiWM adapt bidirectional video diffusion backbones into camera-controllable autoregressive world models (Zhao et al., 2026a; Rui et al., 2026). Causal Forcing, Causal Forcing++, Self Forcing, and Causal-rCM improve the training and distillation of one- to four-step causal rollouts (Zhu et al., 2026; Zhao et al., 2026b; Huang et al., 2026; Zheng et al., 2026). These methods improve rollout quality, speed, and longhorizon consistency, but assume that the action remains fixed while a generation unit is sampled. ActionSplice addresses an action change during sampling by revising the active solver state rather than waiting for the next generation unit.

Interactive condition changes and memory. Existing methods handle condition changes through steering objectives, memory updates, or cache reconstruction. Delta Forcing constrains teacherguided steering with a trust region to improve event responsiveness without destabilizing autoregressive rollout (Wu et al., 2026b). Echo-Forcing separates stable, recent, and recalled memory for prompt switching and scene recall (Wu et al., 2026a). LongLive refreshes cached states after prompt switches, while Anchor Forcing reconstructs the cache from anchor memories (Yang et al., 2025; 2026). Visko Orbis supports live prompt updates during long-form streaming generation (Gao et al., 2026). These approaches update future conditioning or attention memory, but do not estimate the denoising state that a revised action would induce at the current solver step. ActionSplice learns this counterfactual state from matched rollback pairs and resumes the remaining solver evaluations from the interruption point.

Inference-time reuse and acceleration. Methods for accelerating video diffusion reuse computation across denoising steps, attention operations, autoregressive chunks, and generation requests. TeaCache and FasterCache adaptively reuse model features across denoising steps. Pyramid Attention Broadcast reuses attention outputs over stable timestep intervals (Liu et al., 2025; Lyu et al., 2025; Zhao et al., 2025). Sparse VideoGen and SparsePR reduce attention cost through structured spatiotemporal sparsity (Xi et al., 2025; Taghavi et al., 2026). Recent methods extend reuse to autoregressive and interactive settings. Light Interaction combines adaptive context selection, denoising reuse, and sparse attention over blocks (Lu et al., 2026). X-Cache and $C ^ { 3 }$ ache reuse computation across chunks, while Chorus reuses intermediate states across related requests (Zeng et al., 2026; Zhao et al., 2026c; Liu et al., 2026). These methods reuse computation that remains valid when conditioning does not change. An action update received during sampling makes the current solver state stale for the revised action. ActionSplice transports that state toward the trajectory induced by the new action and resumes the remaining solver evaluations without replaying completed ones.

![](images/8449f6aab89166ad96eabb0fd3af4b33c3133a764cbaeb1a40a5e3d370da0146.jpg)  
Figure 1: ActionSplice overview. (a) Direct condition swapping applies the revised action $a ^ { + }$ only to the remaining evaluations, leaving the interrupted state $s _ { r } ^ { - }$ on the trajectory induced by $a ^ { - }$ . (b) At receipt step r, the corrector $C _ { \phi _ { q } }$ transports $s _ { r } ^ { - }$ to ${ \widehat { s } } _ { r } ,$ , after which the frozen sampler executes the remaining $K - r$ steps. $\mathrm { C S T } _ { R }$ uses $M _ { 0 } = \mathbf { 1 }$ to retarget the entire chunk. CST uses the suffix mask $M _ { m }$ to leave the prefix coordinates unchanged during transport. (c) Matched rollback provides the target $s _ { r } ^ { \star }$ at the same solver step. CST uses ordinary new-action rollback, while CST clamps the old-action prefix after each replayed evaluation. Rollback is used only for training and evaluation. (d) The receipt step r indexes solver time, while the boundary m indexes video time.

Revisable denoising and editing of intermediate states. Diffusion Forcing uses independent noise levels for individual tokens to support flexible causal denoising, while Diffusion ReRoll selectively re-noises stable regions to revise predictions across a temporal horizon (Chen et al., 2024; Kim et al., 2026). SDEdit edits images by re-noising a source, EDICT recovers an editable diffusion trajectory through inversion, and Layered Diffusion Brushes caches intermediate latents for localized edits (Meng et al., 2021; Wallace et al., 2023; Gholami & Xiao, 2025). For video, FateZero preserves intermediate attention maps during inversion to guide temporally consistent edits (Qi et al., 2023). These methods revise generated content through flexible noise schedules, re-noising, inversion, or spatial masking. They do not estimate the active solver state that a revised action would have produced at the current step in a frozen world model. ActionSplice learns this transport from matched rollback pairs. Its temporally masked variant restricts transport to the editable suffix and places the action boundary inside the active chunk. Both variants resume from the interruption poin without replaying completed solver evaluations.

## 3 METHOD

ActionSplice edits the state of an active chunk when the control changes before sampling is complete. Its core operation, Counterfactual State Transport (CST), uses a learned corrector to estimate the backbone-native transport state that the revised control would have produced at the current solver step. The world model and sampler remain frozen, and generation resumes from the corrected state without replaying completed solver evaluations. A valid in-flight edit must satisfy four requirements. It must remain at the current solver step, preserve committed history, make the updated action visible within the active chunk, and avoid replaying completed solver evaluations.We consider two settings that differ in the temporal support of the correction. $\mathrm { C S T } _ { R }$ retargets the entire active chunk when an action arrives during solver execution. $\mathrm { C S T } _ { T }$ leaves the prefix coordinates of the transport representation unchanged and applies the revised action only to the suffix beginning at temporal boundary m. Figure 1 illustrates an overview of ActionSplice method.

## 3.1 PROBLEM FORMULATION

Let a frozen world model generate an active chunk of T latent frames with a K-step sampler. An action update arrives after $r \in \{ 1 , \ldots , K - 1 \}$ completed solver evaluations. We denote the backbonenative internal representation corrected by ActionSplice at solver step r by $s _ { r } ^ { - }$ . Depending on the sampler parameterization, $s _ { r } ^ { - }$ is either a cached clean prediction $x _ { r } ^ { - }$ or a solver state $z _ { r } ^ { - }$ . The state $s _ { r } ^ { - }$ lies on the trajectory induced by the previous action $a ^ { - }$ , while the update requests a revised action $a ^ { + }$

Let m be the first latent position in the active chunk controlled by $a ^ { + }$ . The requested control schedule and editable mask are

$$
c _ { \tau } ^ { ( m ) } = \{ a ^ { - } , \tau < m , \mathrm { ~ } \quad \quad M _ { m } [ \tau ] = { \bf 1 } [ \tau \geq m ] , \mathrm { ~ } \quad \tau \in \{ 0 , . . . , T - 1 \} .\tag{1}
$$

The receipt step r identifies the interruption along the solver trajectory, while m identifies the action boundary along the temporal axis of the chunk. Setting $m = 0$ applies $a ^ { + }$ to the entire active chunk. For $0 < m < T$ , latent positions before m remain under $a ^ { - }$ and positions from m onward follow $a ^ { + }$

We define $s _ { r } ^ { \star }$ as the matched counterfactual representation at solver step r under the requested control schedule $c ^ { ( m ) }$ . For $\mathrm { C S T } _ { R } ,$ , full rollback constructs this target by replaying the first r solver evaluations under $\grave { a } ^ { + }$ . For $\bar { \mathrm { C S T } } _ { T }$ , we use the prefix-clamped rollback construction described in Section 3.3. The matched branches share the prompt, initial state, committed history, solver schedule, receipt step, and stochastic inputs when applicable. Direct condition swapping instead leaves $s _ { r } ^ { - }$ on the trajectory induced by $a ^ { - }$ . The corrector must approximate $s _ { r } ^ { \star }$ over the editable positions while preserving the remaining coordinates and resumability at step r.

![](images/3579a55ecdd3aebb26c4006a9e2c3643b06f67bfc58b2110849b0681595d7f51.jpg)  
Figure 2: Action-update trade-offs after r of $K$ solver evaluations. Condition swapping resumes from stale $s _ { r } ^ { - }$ , waiting defers update, and full rollback replays r evaluations to recover $s _ { r } ^ { \star }$ $\mathrm { C S T } _ { R }$ predicts $\widehat { s } _ { r }$ and resumes without replay.

## 3.2 ACTIONSPLICE: SAME-STEP STATE EDITING

ActionSplice predicts a residual that transports the interrupted representation $s _ { r } ^ { - }$ toward its matched counterfactual target $s _ { r } ^ { \star }$ . Let $\mathcal { T } _ { r }$ denote the backbone-specific context available at receipt step r. It includes $s _ { r } ^ { - }$ , the initial state, recent committed history, the previous and revised action conditioning, and any sampler information retained at the interruption. For recurrent editing, $\mathcal { T } _ { r }$ may also encode the intervention history. For edit type $q \in \{ R , T \}$ , the corrector $C _ { \phi _ { q } }$ predicts a transport residual from $\mathcal { T } _ { r }$ and $M _ { m }$

$$
\widehat { s } _ { r } = s _ { r } ^ { - } + M _ { m } \odot C _ { \phi _ { q } } ( \mathbb { Z } _ { r } , M _ { m } ) ,\tag{2}
$$

The temporal mask is broadcast over the channel and spatial dimensions. It restricts the predicted residual to the requested action interval and leaves the remaining coordinates unchanged.

Whole-chunk editing. For $\mathrm { C S T } _ { R } ,$ the updated action applies to the entire active chunk. We set $m = 0$ and $M _ { 0 } = \mathbf { 1 }$ , allowing the corrector to update every active latent position. This variant supports asynchronous action revisions that should affect the full active chunk with the new control.

Prefix-preserving editing. For $\mathrm { C S T } _ { T }$ , the action changes at a temporal boundary $0 < m < T$ The hard output mask preserves positions $0 , \ldots , m - 1$ and restricts the correction to positions $m , \ldots , T - 1 .$ . The corrected representation therefore retains the old-action prefix and applies the new action to the suffix of the same active chunk. This variant supports scheduled action transitions at a specified boundary within the active chunk.

State reconstruction and resumption. The corrected representation $\widehat { s } _ { r }$ is inserted into a valid solver state at the same solver step r. For min $\mathbf { W } \mathbf { M } , \widehat { s } _ { r } = \widehat { x } _ { i }$ is the corrected clean prediction. The native scheduler reconstructs $\widehat { z } _ { r }$ from $\widehat { x } _ { r }$ and the stored transition noise. For HY-WM1. $5 , \widehat { s } _ { r } = \widehat { z } _ { r }$ is the corrected Euler solver state, so no reconstruction is required. ActionSplice invalidates temporary action-dependent cache entries while preserving committed-history entries. Sampling then resumes from $\widehat { z } _ { r }$ under $c ^ { ( m ) }$ for the remaining $K - r$ backbone evaluations.

After the update, ActionSplice performs one corrector evaluation and $K - r$ backbone evaluations. Full rollback instead performs K backbone evaluations, comprising r replayed evaluations and the remaining $K - r$ evaluations. ActionSplice therefore avoids replaying the r backbone evaluations completed before the update. separately from backbone evaluations.

## 3.3 MATCHED COUNTERFACTUAL SUPERVISION

Each training example pairs an interrupted source representation $s _ { r } ^ { - }$ with the matched counterfactual target $s _ { r } ^ { \star }$ defined in Section 3.1. The source and target branches share the prompt, initial state, committed history, solver schedule, receipt step, and stochastic inputs when applicable. They differ only in the action schedule applied to the active chunk. This pairing isolates the change in the backbone-native representation induced by the action update.

For $\mathrm { C S T } _ { R } ,$ the source branch reaches solver step r under $a ^ { - }$ . The matched rollback branch replays the first r solver evaluations from the same initial state under $a ^ { + }$ . Its backbone-native representation at step r defines $s _ { r } ^ { \star }$

For $\mathrm { C S T } _ { T } .$ , we construct the target using matched prefix-clamped rollback. The target branch starts from the same initial state as the source branch and is evaluated under the requested-action conditioning. After every replayed solver evaluation, we replace its latent positions $0 , \ldots , m - 1$ with the corresponding source-branch positions and retain the updated suffix $m , \ldots , T - 1$ . The source and target representations therefore agree exactly before the temporal boundary:

$$
s _ { r } ^ { \star } [ \tau ] = s _ { r } ^ { - } [ \tau ] , \qquad \tau < m .\tag{3}
$$

Consequently, the target correction is identically zero over the clamped prefix. The transport residual and its masked training objective are defined in Section 3.4.

Recurrent training exposes each corrector to histories produced by its earlier predictions. After each student correction, sampling completes the active chunk and the rollout continues to the next interruption. At each interruption, the matched source and teacher branches share the same studentproduced committed history. The teacher uses ordinary new-action rollback for $\mathrm { C S T } _ { R }$ and prefixclamped rollback for $\mathrm { C S T } _ { T } .$ . This construction reduces the mismatch between training and recurrent inference. Rollback is used only to construct supervision and evaluation references and is not executed during ActionSplice inference.

## 3.4 TRANSPORT CORRECTOR AND TRAINING OBJECTIVES

We train a separate corrector for each backbone and edit type. Each corrector is a six-block residual 3D encoder-decoder. FiLM conditioning (Perez et al., 2018) modulates its hidden features using embeddings of the camera controls and solver step. For $\mathrm { C S T } _ { T }$ , the corrector also receives the temporal mask $M _ { m }$ . Applying this mask to the predicted correction enforces exact preservation of the prefix at the receipt step.

We measure transport error over the editable elements using masked normalized mean squared error:

$$
\mathrm { N M S E } _ { M } ( u , v ) = \frac { \Vert M \odot ( u - v ) \Vert _ { 2 } ^ { 2 } / N _ { M } } { \operatorname* { m a x } ( \Vert M \odot v \Vert _ { 2 } ^ { 2 } / N _ { M } , \epsilon ) } ,\tag{4}
$$

where $N _ { M }$ is the number of selected scalar elements after broadcasting the temporal mask over channels and spatial locations. We write NMSE for the unmasked case $M = \mathbf { 1 }$

Let $\widehat { \Delta s } _ { r } = \widehat { s } _ { r } - s _ { r } ^ { - }$ denote the predicted transport and $\Delta s _ { r } ^ { \star } = { s } _ { r } ^ { \star } - { s } _ { r } ^ { - }$ its matched target. State and residual matching have the same error numerator because $\widehat { \Delta } \widehat { s } _ { r } - \Delta s _ { r } ^ { \star } = \widehat { s } _ { r } - s _ { r } ^ { \star }$ . We use the residual form because its normalization measures error relative to the required action-induced transport. The transport loss is $\mathcal { L } _ { \mathrm { t r } } = \mathrm { N M S E } _ { M } ( \widehat { \Delta s } _ { r } , \Delta s _ { r } ^ { \star } )$

For $\mathrm { C S T } _ { T }$ , let $\widehat { d } _ { m } = \widehat { s } _ { r } [ m ] - \widehat { s } _ { r } [ m - 1 ]$ and $d _ { m } ^ { \star } = s _ { r } ^ { \star } [ m ] - s _ { r } ^ { \star } [ m - 1 ]$ denote the predicted and target changes across the temporal boundary. We define $\mathcal { L } _ { \mathrm { m i d } } = \mathrm { N M S E } ( \widehat { d } _ { m } , d _ { m } ^ { \star } )$ . This term is omitted for $\mathrm { C S T } _ { R }$ . The core transport objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { t r } } + \lambda _ { \mathrm { m i d } } \mathcal { L } _ { \mathrm { m i d } } , \qquad ( M , \lambda _ { \mathrm { m i d } } ) = \left\{ \begin{array} { l l } { ( { \bf 1 } , 0 ) , } & { \mathrm { C S T } _ { R } , } \\ { ( M _ { m } , \lambda _ { m } ) , } & { \mathrm { C S T } _ { T } . } \end{array} \right.\tag{5}
$$

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP AND EVALUATION PROTOCOL

We evaluate ActionSplice on minWM–Wan Action2V (Zhao et al., 2026a) and HY-WM1.5 (HunyuanWorld, 2025). Both backbones use a four-step sampler, and we train a separate corrector for each backbone and CST variant. We use the same prompt-disjoint split of 150 scene prompts for both backbones, with 120 for training and 30 reserved for evaluation. Each prompt yields three interruption captures per corrector, giving 360 training and 90 held-out captures. Training captures for both variants come from recurrent rollouts, so later interruptions inherit committed histories produced after earlier corrections. Each $\mathrm { C S T } _ { R }$ training trajectory contains three action updates and four action segments, while $\mathrm { C S T } _ { T }$ captures are balanced over temporal boundaries $m \in \{ 1 , 2 , 3 \}$ . Both partitions are balanced across the six directed transitions among forward, backward, and yaw-left actions. The main evaluation uses receipt step $r = 2$ and contains 30 matched prompt groups for $\mathrm { C S T } _ { R }$ and 90 prompt-boundary groups for $\mathrm { C S T } _ { T }$ . The appendix reports results for $r \in \{ 1 , 2 , 3 \}$

We compare against waiting, condition swapping, partial rollback, re-noising, and a matched rollback oracle. Waiting applies the revised action in the next chunk, while condition swapping changes the action for the remaining evaluations without correcting the interrupted state. Partial rollback restores the preceding solver checkpoint and replays one completed evaluation. Re-noising forms a clean estimate from the interrupted state, re-noises it to the noise level at the start of the final two solver evaluations, and reruns those evaluations under the revised action. The matched oracle uses ordinary full rollback for $\mathrm { C S T } _ { R }$ and prefix-clamped rollback for $\mathrm { C S T } _ { T }$ . Within each comparison group, all methods share the prompt, initial state, committed history, action update, receipt step, and solver schedule. The initial stochastic inputs are matched, while re-noising uses an additional fixed Gaussian draw.

Action following is the percentage of examples in which the revised action appears within the intended editable region. Stale frames count decoded frames in that region before the first visible response. Both measurements use human annotations blinded to method identity and are reported in Table 3. Figure 6 shows the annotation interface. We compute LPIPS and PSNR against the matched counterfactual reference over the full active chunk for $\mathrm { C } \bar { \mathrm { S } } \mathrm { T } _ { R }$ and over the suffix $\tau \geq m$ for $\mathrm { C S T } _ { T }$ Prefix LPIPS measures preservation before m. Boundary error is measured at the history-to-active transition for $\mathrm { C S T } _ { R }$ and between positions $m - 1$ and m for $\mathrm { C S T } _ { T }$ . Latent-ready latency ends when the responsive latent is available. Pixel-ready latency ends when the first responsive frame is decoded. Unless otherwise stated, we average framewise metrics within each example and report the median across matched examples. HY-WorldPlay quality and camera-trajectory metrics follow their benchmark protocols and are reported as arithmetic means across examples. Human action following is reported as a proportion, and stale frames as the mean response delay. Timing is measured on an NVIDIA A100 for minWM and an NVIDIA H100 for HY-WM1.5, with runtime comparisons restricted to the same backbone. Optimization and checkpoint-selection details are provided in the appendix. Code and trained corrector weights are open source.

Table 1: Whole-chunk action-update results after $r = 2$ of $K = 4$ solver evaluations over 30 heldout prompts. Fidelity metrics compare the active chunk with matched full rollback. Speedup is computed from pixel-ready latency relative to waiting. Bold and underlined values indicate best and second-best non-oracle results within each backbone.
<table><tr><td>Model</td><td>Method</td><td>Latent-ready latency (ms) ↓</td><td>Pixel-ready latency (ms) ↓</td><td>Speedup ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>Boundary error ↓</td></tr><tr><td rowspan="5">minWM</td><td>Wait</td><td>5295.8</td><td>6815.0</td><td>1.00×</td><td>0.3159</td><td>12.95</td><td>0.0215</td></tr><tr><td>Condition Swap</td><td>979.2</td><td>2498.1</td><td>2.73×</td><td>0.3155</td><td>12.97</td><td>0.0215</td></tr><tr><td>Partial Rollback-d = 1</td><td>1467.4</td><td>2979.3</td><td>2.29×</td><td>0.3111</td><td>12.95</td><td>0.0212</td></tr><tr><td>Re-noising (rerun 2)</td><td>983.4</td><td>2506.4</td><td>2.72×</td><td>0.3160</td><td>12.92</td><td>0.0214</td></tr><tr><td>Full Rollback</td><td>1961.0</td><td>3481.4</td><td>1.96×</td><td></td><td></td><td></td></tr><tr><td></td><td> $\mathbf { C S T } _ { R }$ </td><td>1016.2</td><td>2498.3</td><td>2.73×</td><td>0.1214</td><td>16.30</td><td>0.0175</td></tr><tr><td rowspan="5">HY-WM1.5</td><td>Wait</td><td>7414.0</td><td>11711.2</td><td>1.00×</td><td>0.3682</td><td>14.94</td><td>0.0379</td></tr><tr><td>Condition Swap</td><td>2820.7</td><td>7052.1</td><td>1.66×</td><td>0.3446</td><td>15.20</td><td>0.0368</td></tr><tr><td>Partial Rollback-d = 1</td><td>3619.0</td><td>7942.8</td><td>1.47×</td><td>0.1674</td><td>18.89</td><td>0.0399</td></tr><tr><td>Re-noising</td><td>2818.7</td><td>7101.2</td><td>1.65×</td><td>0.3480</td><td>15.08</td><td>0.0379</td></tr><tr><td>Full Rollback</td><td>4415.0</td><td>8706.8</td><td>1.34×</td><td></td><td></td><td></td></tr><tr><td></td><td> $\mathbf { C S T } _ { R }$ </td><td>2748.8</td><td>7442.2</td><td>1.57×</td><td>0.0831</td><td>20.51</td><td>0.0196</td></tr></table>

## 4.2 WHOLE-CHUNK ACTION UPDATES

We evaluate whole-chunk retargeting with an action update received after $r \ = \ 2$ of the $K = 4$ solver evaluations. The revised action should control the entire active chunk. Table 1 compares $\mathrm { C S T } _ { R }$ with waiting, condition swapping, partial rollback, re-noising, and matched full rollback. All methods use the matched protocol described above. Full rollback provides the counterfactual reference but replays the completed evaluations. Receipt-step sweeps and results for all six directed action transitions are reported in Appendices B.1 and B.3, respectively.

$\mathrm { C S T } _ { R }$ provides the best rollback-relative fidelity among the non-oracle methods on both backbones. On minWM, it matches the pixel-ready latency of condition swapping while reducing LPIPS by 61.5% and improving PSNR by 3.33 dB. On HY-WM1.5, it reduces LPIPS by 75.9% and improves PSNR by 5.31 dB relative to condition swapping, while remaining 1.57× faster than waiting. It also obtains the lowest non-oracle boundary error on both backbones. These results show that same-step transport approaches rollback fidelity without replaying the completed solver evaluations.

## 4.3 WITHIN-CHUNK ACTION TRANSITIONS

We evaluate action updates intended to take effect at temporal boundary m within the active chunk. Positions before m should retain the previous action, while positions from m onward should follow the revised action. Only CST<sub>T</sub> explicitly restricts its correction to this suffix. Table 2 reports aggregate results, while Appendix B.2 reports results for each boundary.

Table 2: Within-chunk action transitions at receipt step $r \ = \ 2$ with $K = 4 .$ , aggregated over 90 matched groups from 30 prompts and boundaries $\bar { m } \in \{ 1 , 2 , 3 \}$ . Fidelity is measured against prefixclamped rollback, and speedup uses pixel-ready latency relative to waiting. Bold and underlined values indicate the best and second-best non-oracle results within each backbone.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">Latent-ready latency (ms) ↓</td><td rowspan="2">Pixel-ready latency (ms) ↓</td><td rowspan="2">Speedup ↑</td><td rowspan="2">Prefix LPIPS↓</td><td rowspan="2">Suffix LPIPS↓</td><td rowspan="2">Suffix PSNR ↑</td><td rowspan="2">Boundary error ↓</td></tr><tr><td></td></tr><tr><td rowspan="5">minWM</td><td>Wait</td><td>5292.5</td><td>6811.6</td><td>1.00×</td><td>0.0863</td><td>0.2874</td><td>13.15</td><td>0.0552</td></tr><tr><td>Condition Swap</td><td>979.2</td><td>2494.4</td><td>2.73×</td><td>0.0867</td><td>0.2868</td><td>13.16</td><td>0.0548</td></tr><tr><td>Partial Rollback  $\mathbf { \nabla } \cdot d = 1$ </td><td>1468.4</td><td>2986.6</td><td>2.28×</td><td>0.0842</td><td>0.2773</td><td>13.27</td><td>0.0542</td></tr><tr><td>Re-noising (rerun 2)</td><td>982.4</td><td>2502.7</td><td>2.72×</td><td>0.0897</td><td>0.2889</td><td>13.14</td><td>0.0549</td></tr><tr><td>Full Rollback (oracle) CSTT</td><td>1962.6</td><td>3482.4 2496.4</td><td>1.96×</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Wait</td><td>1013.8</td><td></td><td>2.73×</td><td>0.0740</td><td>0.1258</td><td>16.46</td><td>0.0511</td></tr><tr><td rowspan="5">HY-WM1.5</td><td></td><td>7406.3</td><td>11701.7</td><td>1.00×</td><td>0.0715</td><td>0.3612</td><td>15.06</td><td>0.0497</td></tr><tr><td>Condition Swap</td><td>2818.2</td><td>7132.4</td><td>1.64×</td><td>0.0763</td><td>0.2731</td><td>15.53</td><td>0.0451</td></tr><tr><td>Partial Rollback  $\mathbf { \nabla } \cdot d = 1$ </td><td>3618.1</td><td>7901.2</td><td>1.48×</td><td>0.0498</td><td>0.0809</td><td>21.52</td><td>0.0269</td></tr><tr><td>Re-noising (rerun 2)</td><td>2811.8</td><td>7115.5</td><td>1.64×</td><td>0.0933</td><td>0.2790</td><td>15.51</td><td>0.0464</td></tr><tr><td>Full Rollback (oracle)</td><td>6012.7</td><td>10305.8</td><td>1.14×</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>CSTT</td><td>2746.3</td><td>6941.2</td><td>1.69×</td><td>0.0412</td><td>0.0615</td><td>25.05</td><td>0.0215</td></tr></table>

Table 3: Human evaluation of action responsiveness on HY-WM1.5 over 30 matched $\mathrm { C S T } _ { R }$ groups and 90 matched $\mathrm { C S T } _ { T }$ groups. Action following measures responses within the editable region; mean stale frames measures the delay to the first visible response.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2"> $\mathrm { C S T } _ { R }$ </td><td colspan="2"> $\mathrm { C S T } _ { T }$ </td></tr><tr><td>Action following (%) ↑</td><td>Mean stale frames ↓</td><td>Action following (%) ↑</td><td>Mean stale frames ↓</td></tr><tr><td rowspan="6">HY-WM1.5</td><td>Wait</td><td>21.1</td><td>15.37</td><td>26.7</td><td>7.53</td></tr><tr><td>Condition Swap</td><td>84.2</td><td>8.89</td><td>86.7</td><td>3.73</td></tr><tr><td>Partial Rollback  $- d = 1$ </td><td>89.5</td><td>3.00</td><td>93.3</td><td>0.93</td></tr><tr><td>Re-noising</td><td>78.9</td><td>9.89</td><td>86.7</td><td>4.40</td></tr><tr><td>Full Rollback (oracle)</td><td>100.0</td><td>0.00</td><td>100.0</td><td>0.00</td></tr><tr><td>CST</td><td>94.7</td><td>1.05</td><td>100.0</td><td>0.40</td></tr></table>

CST<sub>T</sub> obtains the best non-oracle prefix preservation, suffix fidelity, and boundary continuity on both backbones. On minWM, it matches the pixel-ready latency of condition swapping while reducing suffix LPIPS by 56.1% and improving suffix PSNR by 3.30 dB. On HY-WM1.5, it reduces suffix LPIPS by 77.5% and improves suffix PSNR by 9.52 dB relative to condition swapping, while providing a 1.69× speedup over waiting. Its lower prefix LPIPS and boundary error indicate better prefix preservation and a smoother transition at m.

## 4.4 COMPARISON WITH PRIOR WORK

Table 4 compares $\mathrm { C S T } _ { R }$ with published HY-WorldPlay results (Lu et al., 2026). This complementary benchmark measures global video quality rather than the in-flight action-update fidelity and latency in Tables 1 and 2. Our result uses a reproduced HY-WM1.5 backbone under the same protocol.

Table 4: Global video quality under the HY-WorldPlay protocol. $\mathrm { C S T } _ { R }$ uses reproduced HY-WM1.5. Bold and underlined values indicate the best and second-best reported results.
<table><tr><td rowspan="3">Model</td><td rowspan="3">Method</td><td colspan="3">vs. Original</td><td colspan="3">Self-Comparison</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM↑</td><td>LPIPS ↓</td></tr><tr><td rowspan="6"></td><td>Original (HunyuanWorld, 2025)</td><td></td><td></td><td></td><td>18.60</td><td>0.5678</td><td>0.2051</td></tr><tr><td>SVG (Xi et al., 2025)</td><td>19.48</td><td>0.6028</td><td>0.2209</td><td>17.75</td><td>0.5299</td><td>0.2187</td></tr><tr><td>BSA (Meituan LongCat Team et al., 2025)</td><td>15.94</td><td>0.4639</td><td>0.3755</td><td>15.44</td><td>0.4205</td><td>0.3720</td></tr><tr><td>TeaCache (Liu et al., 2025)</td><td>20.90</td><td>0.6588</td><td>0.1892</td><td>18.86</td><td>0.5743</td><td>0.2054</td></tr><tr><td>Light Interaction (Lu et al., 2026)</td><td>24.81</td><td>0.6500</td><td>0.1788</td><td>18.85</td><td>0.5854</td><td>0.1963</td></tr><tr><td>CSTR</td><td>25.66</td><td>0.6902</td><td>0.1337</td><td>20.38</td><td>0.6267</td><td>0.1587</td></tr></table>

$\mathrm { C S T } _ { R }$ obtains the best reported PSNR, SSIM, and LPIPS in both comparisons, indicating that inflight state transport preserves global video quality under the established benchmark.

## 4.5 CAMERA-ACTION FOLLOWING

We evaluate camera-action following with the rotational and translational trajectory errors $R _ { \mathrm { d i s t } }$ and $T _ { \mathrm { d i s t } }$ from HunyuanWorld (2025). ViPE estimates camera poses from each video. We express the estimated and target poses relative to their first frames and normalize translation scale. We score the active chunk for $\mathrm { C S T } _ { R }$ and only the editable suffix $\tau \geq m$ for $\mathrm { C S T } _ { T }$ . Table 5 reports published short-term references and our in-flight results. The request protocols differ, so the published rows provide context rather than a direct ranking.

## 4.6 REPEATED IN-FLIGHT ACTION UPDATES

We test whether transport error accumulates under repeated updates. Each HY-WM1.5 rollout contains six action segments separated by five updates, all received after solver step $r \ = \ 2$ of $K = 4$ . The $\mathrm { C S T } _ { R }$ corrector is trained on trajectories with three interruptions and evaluated on five-interruption sequences without further training. Condition swapping and sequential Full Rollback use matched prompts, action schedules, initial states, and solver trajectories. Figure 3 reports rotation and translation error, rollback-relative LPIPS, and boundary error after each interruption.

![](images/a67d7c36e8610f4aed94b614f0a113532c367930e8852d0f0ce1d1d145f6539d.jpg)

Table 5: Camera-action following measured by rotation and translation trajectory error. Published short-term references follow HunyuanWorld (2025). Lower is better.
<table><tr><td>Method</td><td> $R _ { \mathrm { d i s t } }$  →  $T _ { \mathrm { d i s t } } \downarrow$ </td></tr><tr><td>CameraCtrl</td><td>0.341</td></tr><tr><td>VMem Matrix-Game 2.0 (He et al., 2025)</td><td>0.219</td></tr><tr><td>0.287 WorldPlay (HunyuanWorld, 2025) 0.031</td><td>0.843 0.121</td></tr><tr><td> $\mathrm { C S T } _ { R }$ </td><td>0.049</td></tr><tr><td></td><td>0.037</td></tr><tr><td> $\mathrm { C S T } _ { T }$ </td><td>0.035 0.120</td></tr></table>

![](images/1b01ed40821d097289ce5a4a5478a00f60311693e35d1e46ea79fc11eba7f9f3.jpg)

![](images/eb537647a958065de6362c39985436c33e3d9a3986cde44699ece1b6c71cd152.jpg)  
Figure 3: Repeated in-flight updates on HY-WM1.5. A $\mathrm { C S T } _ { R }$ corrector trained on three-interruption trajectories is applied at five interruptions without further training. Curves show mean error over the evaluation prompts. Sequential Full Rollback provides the reference.

Across all interruption indices, $\mathrm { C S T } _ { R }$ remains closer to sequential Full Rollback than condition swapping on every metric.

![](images/93b949a556a3c922e895367ccb9c066fafb98d6e173218590d8a7dc0672fceef.jpg)  
Figure 4: Qualitative whole-chunk action updates. The top pair shows HY-WM1.5 changing from forward to yaw left. Condition swapping retains stale forward motion, while $\mathrm { C S T } _ { R }$ responds within the active chunk. The bottom pair shows minWM changing from yaw left to yaw right. Waiting defers the request, while $\mathrm { C S T } _ { R }$ responds in the active chunk. Red and green borders mark stale and responsive frames.

![](images/2827387d0ab6058a58909e021273f878491b2b97a1af40cc53f10e4ba60c7bcd.jpg)  
Figure 5: Qualitative within-chunk transition on HY-WM1.5. The update arrives at $r = 2$ and requests yaw right from $m = 2 .$ . The Full Rollback row follows the original yaw-left action throughout the chunk, while $\mathrm { C S T } _ { T }$ preserves the yaw-left prefix and applies yaw right to the suffix. Both rollouts continue through the following chunk to make the trajectory difference visible. The yaw-right target is outside the corrector’s training action set and is shown only as a qualitative generalization example.

## 4.7 QUALITATIVE COMPARISONS

Figure 4 shows representative whole-chunk updates on both backbones. On HY-WM1.5, condition swapping continues the forward trajectory after a yaw left request, while $\mathrm { C S T } _ { R }$ turns within the active chunk. On minWM, waiting defers a yaw right request, while $\mathrm { C S T } _ { R }$ applies it within the active chunk. In both cases, same-step transport makes the revised action visible without restarting the chunk.

Figure 5 complements the within-chunk results in Table 2. An update received at $r = 2$ requests a yaw-left-to-yaw-right transition at $m = 2$ . A one-action rollback follows yaw left throughout the chunk, whereas $\mathrm { C } \mathrm { \check { S } T } _ { T }$ preserves the prefix and applies yaw right to the suffix. We continue both rollouts through the following chunk to make the trajectory difference visible.

## 4.8 ABLATIONS

We ablate matched counterfactual supervision and the hard output mask under the full-model evaluation protocol. Removing the mask tests whether $\mathrm { C S T } _ { T }$ can learn prefix preservation without an explicit output constraint. Unmatched supervision removes the matched source-teacher pairing. Table 6 reports action error, rollback-relative fidelity, boundary continuity, and prefix preservation.

Table 6: ActionSplice ablations. RB-LPIPS measures fidelity to matched Full Rollback, and prefix LPIPS applies only to $\mathrm { C S T } _ { T }$ . Lower is better for all metrics.
<table><tr><td></td><td colspan="3"> $\mathrm { C S T } _ { R }$ </td><td colspan="4"> $\mathrm { C S T } _ { T }$ </td></tr><tr><td>Configuration</td><td>Action error ↓</td><td>RB-LPIPS ↓</td><td>Boundary error ↓</td><td>Action error ↓</td><td>RB-LPIPS ↓</td><td>Boundary error ↓</td><td>Prefix LPIPS ↓</td></tr><tr><td>Full model</td><td>0.6192</td><td>0.0831</td><td>0.0196</td><td>0.3283</td><td>0.0615</td><td>0.0215</td><td>0.0412</td></tr><tr><td>Without hard mask</td><td></td><td></td><td></td><td>0.4092</td><td>0.0787</td><td>0.0240</td><td>0.0604</td></tr><tr><td>Unmatched supervision</td><td>4.6889</td><td>0.7797</td><td>0.0435</td><td>3.0415</td><td>0.7719</td><td>0.0349</td><td>0.4209</td></tr></table>

The full model performs best on every reported metric for both variants. Removing the hard mask worsens $\mathrm { C S T } _ { T }$ prefix preservation and boundary continuity. Unmatched supervision degrades every metric, supporting the use of paired states that differ only in the requested action schedule.

## 5 CONCLUSION

We introduced ActionSplice for same-step action updates during diffusion sampling. $\mathrm { C S T } _ { R }$ retargets the active chunk, while $\mathrm { C S T } _ { T }$ preserves a temporal prefix and updates only the suffix. Across minWM and HY-WM1.5, matched counterfactual transport improves rollback-relative fidelity over direct condition swapping without replaying completed evaluations. It also remains effective under repeated interruptions and preserves global video quality under the HY-WorldPlay protocol.

## REFERENCES

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and Franc¸ois Fleuret. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024.

Boyuan Chen, Diego Mart´ı Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitz-´ mann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

Xiangbo Gao, Siyuan Yang, Ping He, Mingyang Wu, Yuheng Wu, Yushen Zuo, Jiongze Yu, Ryan Cui, Hongyuan Hua, Devin Ma, et al. Visko orbis 1.0: A live model for real-time interactive long video generation. arXiv preprint arXiv:2607.26694, 2026.

Peyman Gholami and Robert Xiao. Streamlining image editing with layered diffusion brushes. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 17368–17378. IEEE, 2025.

Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al. Matrix-game 2.0: An open-source real-time and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. Advances in Neural Information Processing Systems, 38:167283–167308, 2026.

Team HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025.

Seonsoo Kim, Seongil Hong, and Jun-Gill Kang. Diffusion reroll: Revisable denoising for robotic sequential prediction. arXiv preprint arXiv:2607.19919, 2026.

Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 7353–7363. IEEE, 2025.

Hao Liu, Ye Huang, Chenghuan Huang, Zhenyi Zheng, Jiangsu Du, Ziyang Ma, Jing Lyu, and Yutong Lu. Beyond few-step inference: Accelerating video diffusion transformer model serving with inter-request caching reuse. arXiv preprint arXiv:2604.04451, 2026.

Jiacheng Lu, Haoyi Zhu, Sipei Yi, Enze Xie, Yu Li, and Cheng Zhuo. Light interaction: Trainingfree inference acceleration for interactive video world models. arXiv preprint arXiv:2605.31158, 2026.

Zhengyao Lyu, Chenyang Si, Junhao Song, Zhenyu Yang, Yu Qiao, Ziwei Liu, and Kwan-Yee K Wong. Fastercache: Training-free video diffusion model acceleration with high quality. In International Conference on Learning Representations, volume 2025, pp. 33132–33156, 2025.

Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, and Tong Zhang. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025. URL https://arxiv.org/abs/2510.22200.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15886–15896. IEEE, 2023.

Shaohao Rui, Xiaofeng Mao, Zhanyu Zhang, Peijia Lin, Yansong Zhu, Yibo Zhang, Haibin Wan, Zhangrui Zhao, and Weijie Ma. Biwm: Advancing open-source interactive video world models with bidirectional autoregression. arXiv preprint arXiv:2606.10135, 2026.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

Pardis Taghavi, Reza Langari, and Gaurav Pandey. Partition the support, reconstruct the residual: Training-free sparse attention for video generation and world models. arXiv preprint arXiv:2608.18484, 2026.

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real time game engines. In International Conference on Learning Representations, volume 2025, pp. 73754–73776, 2025.

Bram Wallace, Akash Gokul, and Nikhil Naik. Edict: Exact diffusion inversion via coupled transformations. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 22532–22541. IEEE, 2023.

Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, et al. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

Mingqiang Wu, Weilun Feng, Zhefeng Zhang, Haotong Qin, Yuqi Li, Guoxin Fan, Xiaokun Liu, Zhulin An, Libo Huang, Yongjun Xu, et al. Echo-forcing: A scene memory framework for interactive long video generation. arXiv preprint arXiv:2605.16003, 2026a.

Yuheng Wu, Xiangbo Gao, Tianhao Chen, Xinghao Chen, Qing Yin, Zhengzhong Tu, and Dongman Lee. Delta forcing: Trust region steering for interactive autoregressive video generation. arXiv preprint arXiv:2605.14382, 2026b.

Haocheng Xi, Shuo Yang, Yilong Zhao, Chenfeng Xu, Muyang Li, Xiuyu Li, Yujun Lin, Han Cai, Jintao Zhang, Dacheng Li, et al. Sparse videogen: Accelerating video diffusion transformers with spatial-temporal sparsity. arXiv preprint arXiv:2502.01776, 2025.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

Yang Yang, Tianyi Zhang, Wei Huang, Jinwei Chen, Boxi Wu, Xiaofei He, Deng Cai, Bo Li, and Peng-Tao Jiang. Anchor forcing: Anchor memory and tri-region rope for interactive streaming video diffusion. arXiv preprint arXiv:2603.13405, 2026.

Yixiao Zeng, Jianlei Zheng, Chaoda Zheng, Shijia Chen, Mingdian Liu, Tongping Liu, Tengwei Luo, Yu Zhang, Boyang Wang, Linkun Xu, et al. X-cache: Cross-chunk block caching for fewstep autoregressive world models inference. arXiv preprint arXiv:2604.20289, 2026.

Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande He, Xiao Yang, Chongxuan Li, et al. minwm: A full-stack open-source framework for real-time interactive video world models. arXiv preprint arXiv:2605.30263, 2026a.

Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao Yang, Chongxuan Li, and Jun Zhu. Causal forcing++: Scalable few-step autoregressive diffusion distillation for real-time interactive video generation. arXiv preprint arXiv:2605.15141, 2026b.

Weisen Zhao, Lam Nguyen, Zhicong Lu, and Yuzhang Shang. $C ^ { 3 }$ ache: Accelerating world action models with cross inference chunk cache. arXiv preprint arXiv:2606.08962, 2026c.

Xuanlei Zhao, Xiaolong Jin, Kai Wang, and Yang You. Real-time video generation with pyramid attention broadcast. In International Conference on Learning Representations, volume 2025, pp. 3296–3319, 2025.

Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma. Causal-rcm: A unified teacher-forcing and self-forcing open recipe for autoregressive diffusion distillation in streaming video generation and interactive world models. arXiv preprint arXiv:2606.25473, 2026.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.

## SUPPLEMENTARY MATERIAL

## A IMPLEMENTATION AND TRAINING DETAILS

## A.1 BACKBONE-SPECIFIC TRANSPORT AND RESUMPTION

For minWM–Wan Action2V, the corrector edits the cached clean prediction $x _ { r } = \mathtt { p r e d \_ x 0 }$ . We reconstruct z with the stored transition-noise tensor and the native scheduler. For HY-WM1.5, the corrector edits $z _ { r }$ directly and resumes deterministic Euler sampling at the same solver step.

## A.2 DATASET CONSTRUCTION

We use a fixed prompt-disjoint split of 150 prompt-defined scenes. The split contains 120 training prompts and 30 held-out prompts, with one fixed seed per prompt. The same master manifest and split are used for minWM and HY-WM1.5. Training captures are balanced across the six directed transitions among forward, backward, and yaw-left motion. Each capture stores the same solver trace at receipt steps $r \in \{ 1 , 2 , 3 \}$

For $\mathrm { C S T } _ { R } ,$ each prompt produces one recurrent trajectory with three interruption events and four action segments. Each action persists for three or four chunks, which produces trajectories of 48 to 64 latent frames. The three events yield 450 captures in total, split into 360 training and 90 heldout captures. For $\mathrm { C S T } _ { T }$ , each prompt produces three recurrent captures with temporal boundaries m $\in \ \{ 1 , 2 , 3 \}$ . The split again contains 360 training and 90 held-out captures. The six action transitions and three temporal boundaries are balanced in both partitions.

## A.3 OPTIMIZATION, LOSSES, AND CHECKPOINT SELECTION

We train each corrector for 4,000 steps with AdamW, batch size 1, weight decay $1 0 ^ { - 4 }$ , and gradient clipping at 1.0. The learning rate follows a cosine schedule from $1 0 ^ { - 5 } \mathrm { \ t o \ 1 0 ^ { - 6 } }$ . The state-matching term has unit weight for both backbones. minWM adds a unit-weight residual term, while HY-WM1.5 directly matches the solver state and sets the residual-loss weight to zero. Both backbones use decoded LPIPS, temporal, and history-to-chunk boundary losses with weights 0.05, 0.1, and 0.1. $\mathrm { C S T } _ { T }$ adds an intra-chunk boundary loss with weight 0.1. These auxiliary terms complement the core transport objective in Eq. equation 5.

We initialize the minWM correctors from their corresponding $\mathrm { C S T } _ { R }$ and multi-boundary $\mathrm { C S T } _ { T }$ pilot checkpoints. Both HY-WM1.5 correctors are initialized from the HY-WM1.5 pilot corrector. The world model and sampler remain frozen. We maintain an exponential moving average of the corrector weights with decay 0.999 and evaluate it every 200 steps. The reported checkpoint is the EMA checkpoint with the lowest held-out transport NMSE.

## B ADDITIONAL RESULTS

## B.1 RECEIPT-STEP SENSITIVITY

We vary the $\mathrm { C S T } _ { R }$ receipt step $r \in \{ 1 , 2 , 3 \}$ on HY-WM1.5 while holding the prompt, initial state, action transition, solver schedule, and stochastic inputs fixed. All rows use the same 30 held-out prompts and matched trajectories as the main evaluation. Table 7 measures remaining computation, latency, rollback fidelity, boundary continuity, and camera-trajectory error.

Table 7: Receipt-step sweep for $\mathrm { C S T } _ { R }$ on HY-WM1.5. The sampler uses $K \ = \ 4 ,$ so Action-Splice executes $K - r$ backbone evaluations after correction. LPIPS, PSNR, and boundary error use matched full rollback as the reference. Camera-trajectory errors are measured over the active chunk.
<table><tr><td>Receipt step r</td><td>Post-request backbone NFE ↓</td><td>Latent-ready latency (ms) ↓</td><td>Pixel-ready latency (ms) ↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>Boundary error ↓</td><td> $R _ { \mathrm { d i s t } } \downarrow$ </td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td></tr><tr><td>1</td><td>3</td><td>3535.8</td><td>7578.8</td><td>0.0643</td><td>23.13</td><td>0.0204</td><td>0.108</td><td>0.024</td></tr><tr><td>2</td><td>2</td><td>2748.8</td><td>7442.2</td><td>0.0831</td><td>20.51</td><td>0.0196</td><td>0.048</td><td>0.025</td></tr><tr><td>3</td><td>1</td><td>1980.1</td><td>6026.6</td><td>0.2694</td><td>18.39</td><td>0.0246</td><td>0.180</td><td>0.064</td></tr></table>

Later receipt steps leave fewer backbone evaluations to execute, but the $r = 3$ results show lower rollback fidelity than earlier corrections.

## B.2 TEMPORAL-BOUNDARY SENSITIVITY

We fix receipt step $r \ = \ 2$ and vary the $\mathrm { C S T } _ { T }$ boundary $m \in \{ 1 , 2 , 3 \}$ on HY-WM1.5. The 30 held-out prompts and matched trajectories are shared across boundaries. Table 8 reports prefix preservation, suffix fidelity, boundary continuity, and camera-trajectory error.

Table 8: Temporal-boundary sweep for $\mathrm { C S T } _ { T }$ on HY-WM1.5 with receipt step $r = 2 .$ Prefix LPIPS is measured on frames before m. Suffix metrics and camera-trajectory errors are measured on frames $\tau \geq m$ . Boundary error is measured between frames $m - 1$ and m.
<table><tr><td>Boundary m</td><td>Prefix  $\mathrm { L P I P S } \downarrow$ </td><td>Suffix  $\mathrm { L P I P S } \downarrow$ </td><td>Suffix PSNR↑</td><td>Boundary error ↓</td><td> $R _ { \mathrm { d i s t } } \downarrow$ </td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td></tr><tr><td>1</td><td>0.0403</td><td>0.0701</td><td>24.74</td><td>0.0189</td><td>0.063</td><td>0.047</td></tr><tr><td>2</td><td>0.0443</td><td>0.0597</td><td>25.53</td><td>0.0208</td><td>0.049</td><td>0.093</td></tr><tr><td>3</td><td>0.0395</td><td>0.0578</td><td>24.69</td><td>0.0230</td><td>0.049</td><td>0.023</td></tr></table>

No boundary is best on every metric, and performance remains comparable across $m \in \{ 1 , 2 , 3 \}$

## B.3 DIRECTED ACTION-TRANSITION BREAKDOWN

Table 9: Per-transition results for $\mathrm { C S T } _ { R }$ on HY-WM1.5 at receipt step $r = 2$ . Rotation and translation errors are measured over the active chunk. LPIPS uses matched full rollback as the reference. The final row aggregates the six balanced transition groups.

<table><tr><td>Action transition</td><td> $R _ { \mathrm { d i s t } } \downarrow$ </td><td> $T _ { \mathrm { d i s t } } \downarrow$ </td><td>LPIPS to rollback↓</td></tr><tr><td>Forward → Backward</td><td>0.002</td><td>0.028</td><td>0.0479</td></tr><tr><td>Forward → Yaw-left</td><td>0.078</td><td>0.031</td><td>0.1285</td></tr><tr><td>Backward → Forward</td><td>0.003</td><td>0.043</td><td>0.0593</td></tr><tr><td>Backward → Yaw-left</td><td>0.054</td><td>0.003</td><td>0.1042</td></tr><tr><td>Yaw-left → Forward</td><td>0.074</td><td>0.028</td><td>0.1434</td></tr><tr><td>Yaw-left → Backward</td><td>0.080</td><td>0.019</td><td>0.2654</td></tr><tr><td>All transitions</td><td>0.048</td><td>0.025</td><td>0.0831</td></tr></table>

## B.4 HUMAN ANNOTATION INTERFACE

Figure 6 shows the blinded interface used for action-following and stale-frame annotation.  
![](images/a4e31f2879a67a9098748a6a65a12a2f742adc7204495e91b92c98639ec89360.jpg)  
Figure 6: Blinded interface for action-following and stale-frame annotation. Annotators inspect anonymized videos for a matched prompt and action schedule, score each action segment, and record a group-level preference. Method identities remain hidden.