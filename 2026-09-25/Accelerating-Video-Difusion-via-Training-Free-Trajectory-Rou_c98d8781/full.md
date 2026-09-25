# Accelerating Video Difusion via Training-Free Trajectory Routing

Mustafa Munir, Huy Vu, Shreyas Misra, Rohit Jena, Sajad Norouzi, Ali Taghibakhshi<sup>\*</sup>, Anis   
Ahmad, Anjul Patney, Pavlo Molchanov, Nima Tajbakhsh   
<sup>\*</sup>Project Lead

Abstract: Video difusion is computationally expensive, as it requires executing a large model across many denoising steps. Even with step-distillation, inference remains expensive because every distilled step still requires a costly model evaluation. We present TRACK: TRajectory-Aware Capacity routing via top-K selection, a heterogeneous denoising strategy that switches between compatible large and small models at selected steps, reducing the average cost per denoising evaluation. The switching steps are determined using a calibration process. TRACK first rolls out a reference trajectory with the large model. Then at each step, the small model’s prediction is also collected and compared against the large model’s prediction to obtain a relative disagreement score. Both models receive the same latent, timestep, conditioning, and guidance inputs. Aggregating this signal over a calibration set produces a disagreement score map across difusion steps, which determines a switching policy for an eficient inference process: quality-sensitive steps keep using the large model, while steps with low disagreement scores are routed to the small model. Inference executes only the selected model at each step, requiring no retraining, architecture or scheduler changes, or online dual-model evaluation. Across Wan 2.1, Cosmos 3, TurboDifusion, and FastVideo, TRACK yields 1.95×, 2.04×–2.73×, 2.69×, and 2.17× speedups, respectively, with comparable aggregate quality and high diversity retention. TRACK thereby establishes automated, training-free model switching as a practical acceleration paradigm for video difusion.

## Introduction

Video difusion models synthesize high-fidelity visual content with realistic motion, but inference latency remains a primary bottleneck for practical deployment [7, 8, 11, 13, 31]. Advanced systems such as Wan 2.1 [37] and Cosmos 3 [1] achieve high physical realism and temporal consistency, yet sampling requires repeatedly executing multi-billion-parameter models over high-dimensional spatiotemporal latents. Model families often provide smaller, lightweight checkpoints that run significantly faster, though at the cost of lower generation quality. This raises a natural deployment question: can we selectively combine the speed of the small model with the quality of the large one?

Most difusion acceleration methods reduce the number of denoising evaluations through distillation or specialized few-step objectives [32, 34, 41], or reduce the cost within each evaluation through caching, streaming, model-sharing, and sparse computation [4, 17, 24, 27, 42, 43]. These directions are essential, but they leave a complementary source of redundancy unexploited. After a sampler and step count have been selected, the remaining evaluations are still assigned to one model at uniform cost. The model switching approach instead reduces the average model cost per remaining step. It can therefore multiply the gains from step reduction rather than compete with them.

Allocating heterogeneous model capacity across video difusion trajectories introduces distinct eficiency and quality challenges. Naive capacity allocation (i.e., model switching) based on rigid or heuristic temporal boundaries or simple monotonic handofs fails to capture the complex spatiotemporal dynamics of video generation. For example, two models can exhibit similar aggregate prediction errors while disagreeing on spatial detail, camera motion, or frame-to-frame consistency at specific timesteps. Furthermore, uncalibrated capacity reduction risks altering structural commitment and collapsing sample diversity across seeds. Therefore, a practical eficiency framework must identify model-switchability at step-level for a given checkpoint pair. This helps dynamically protect quality-sensitive steps with a large model while ofloading switchable steps to the smaller model.

We introduce TRACK: TRajectory-Aware Capacity routing via top-K selection, an automated framework that turns the trajectory capacity allocation problem into an ofline calibration problem. Given independently trained large and small checkpoints sharing a latent space and scheduler (e.g., large 14B and small 1.3B checkpoints of the Wan model family), TRACK rolls out an all-large reference trajectory on a calibration set. At each step, both denoisers are evaluated on the exact same reference latent � , timestep, conditioning, and guidance. We then measure their normalized relative disagreement score to determine steps where the small model accurately approximates the large model. Aggregating these scores produces a disagreement map, which leads to an ofline switching policy that routes quality-sensitive steps to the large model and less sensitive, switchable steps to the small model. During inference, exactly one denoiser runs at each step, requiring no retraining, architecture or scheduler modifications, or online dual-model evaluations.

![](images/8650aa7c74e7614d7a038ec9e300180647ac183d27461c93a3904c14029f14a4.jpg)  
Figure 1 | Training-Free Trajectory-Aware Capacity routing. Top: Ofline calibration rolls out the all-large-model reference trajectory and, at every step, evaluates the large and small checkpoints using the same latent �<sub>�</sub>, timestep, conditioning, and guidance inputs. Their guided-prediction disagreement is aggregated across calibration prompts to produce a switching policy that minimizes disagreement scores. Bottom: Online inference applies that policy while preserving the shared latent representation and scheduler update: quality-sensitive steps keep using the large model (green), while low disagreement steps use the small model (purple), and only one denoiser runs at each step. The method requires no retraining, architecture changes, scheduler changes, or online dual-model evaluation.

We evaluate TRACK across four major video difusion pipelines spanning both many-step and step-distilled models. TRACK achieves 1.95× speedup on Wan 2.1 and 2.04×–2.73× on Cosmos 3 while maintaining visual quality. On few-step distilled pipelines, TRACK yields 2.17× speedup on three-step FastVideo and 2.69× on four-step TurboDifusion, showing that our model switching policy is efective with aggressive step reduction to maximize inference speed. Beyond latency reduction gains, our spatial and latenttemporal frequency analyses demonstrate that prediction disagreement is phase-dependent, explaining intuitively why intermediate timesteps safely tolerate reduced model capacity. Finally, multi-seed evaluations confirm that calibrated, disagreement-guided switching preserves the diversity across generated outputs. TRACK thus establishes training-free model switching as a practical acceleration paradigm that optimizes inference eficiency while maintaining quality and diversity. We summarize our contributions as follows:

• We introduce TRACK, a training-free heterogeneous denoising strategy that generalizes prior large–small video difusion switching from a single permanent handof to disagreementcalibrated, fixed-budget allocation at individual denoising steps, enabling contiguous and noncontiguous schedules without modifying weights, architectures, or schedulers.

• We validate TRACK across four video difusion families spanning many-step and distilled fewstep pipelines, obtaining 1.95×–2.73× speedups at comparable visual quality and similar diversity retention as the all-large model baseline.

• We provide spatial and latent-temporal frequency analyses that explain the phase-dependent substitutability of large and small denoisers.

## Related Work

## Difusion Models for Visual Generation

Denoising difusion models learn a reverse process that transforms noise into samples through a sequence of denoising updates [13, 33]. Improved training and guidance made difusion competitive for high-fidelity conditional image synthesis [7, 12], while latent diffusion reduced computation by moving the reverse process from pixel space to a learned latent representation [31]. Video difusion extends this framework with temporal modeling so that appearance, structure, and motion remain coherent across frames [8, 11]. Recent model families such as Wan 2.1 and Cosmos 3 demonstrate the quality and breadth of modern video generation systems [1, 3, 37]. However, their highdimensional spatiotemporal latents and heavy multibillion parameter denoisers (e.g., Wan 14B, Cosmos 3 - Super 64B) make repeated sampling evaluations a primary deployment eficiency bottleneck.

## Eficient Difusion Inference

Difusion acceleration generally acts on either the total number of denoising steps or the cost of each individual step. Step-reduction methods use distillation or specialized training objectives to compress sampling trajectories. Faster ODE solvers such as DPM-Solver [20] reduce the number of required steps without retraining, while distillation-based methods including Distribution Matching Distillation, SD-Turbo, SDXS, and LCM push further toward one- or fewstep generation [21, 32, 34, 41]. While these methods reduce latency, they require additional training and can introduce model- and budget-dependent quality tradeofs.

Other techniques optimize individual forward passes through spatial redundancy reduction, feature caching, and streaming execution [4, 17, 18, 22, 24, 45]. FastVideo and TurboDifusion combine few-step sampling with video-specific model and system optimizations [42, 43]. TRACK is complementary to both categories: it retains the target sampler and step count while reducing the average model evaluation cost across the denoising process. Its application to three-step FastVideo and four-step TurboDifusion directly demonstrates this complementarity with step reduction.

## Adaptive and Multi-Model Difusion

The difusion denoising process exhibits distinct generation characteristics across timesteps. Spectral analyses show that deep networks process spatial frequency bands non-uniformly [15, 23, 28, 30, 38], and difusion trajectories generally progress from coarse structural formation toward finer detail refinement [27]. This timestep dependence has motivated multi-expert and adaptive-computation approaches that allocate specialized denoisers, frequency components, or subnetworks to diferent portions of the sampling trajectory [2, 36, 39]. Such methods, however, typically require specialized training, architectural modifications, auxiliary control mechanisms, or additional expert parameters. Related ideas arise in model stitching, where representations from networks of diferent capacities are connected to obtain flexible eficiency– accuracy trade-ofs [25, 26]. T-Stitch applies model switching to image difusion by executing a single, one-way handof—using a small model for the initial steps before switching to a large model for the remainder of the trajectory [27].

Other methods like SRDifusion [6] accelerate video difusion on Wan [37] and CogVideoX [40] by switching from a large to a small model based on a runtime latent-change signal computed from the large model alone, enforcing a permanent one-way handof. This approach has two limitations. First, it prevents the large model from resuming even when disagreement rises again, which our frequency analysis in Figure 4 shows occurs in the final denoising steps. Second, since the switching criterion is based only on the large model’s predictions, it does not take into account the small model’s predictions; therefore, it cannot directly measure how well the small model approximates the large one at a given step. TRACK addresses both limitations: it computes cross-model disagreement on shared reference latents ofline, and derives a flexible policy to switch back and forth between models when needed.

Other methods like [16] anneal between the score functions of a base model and its reward-fine-tuned counterpart over the denoising trajectory to preserve complementary characteristics of the two models. TRACK difers in both how its routing policy is obtained and the setting in which it is applied. Rather than using prescribed temporal schedules or learning adapters, experts, or control modules, TRACK evaluates compatible, independently trained large and small checkpoints on identical reference latents and uses the disagreement scores from their predictions to derive an ofline, per-step switching policy. At inference, only the selected denoiser is evaluated at each step, requiring no retraining, architectural modification, or online dual-model evaluation.

While prior work has demonstrated training-free large–small switching for video difusion via a single permanent handof, TRACK generalizes this to disagreement-calibrated, fixed-budget allocation at individual denoising steps, enabling contiguous and non-contiguous schedules and compatibility with stepdistilled pipelines.

## Proposed Methodology

TRACK accelerates video difusion by applying a precomputed step-dependent model-capacity schedule without modifying weights or altering schedulers. Figure 1 illustrates the overall approach. The method operates in two stages: an ofline calibration phase that measures step-level model disagreement on a shared reference trajectory and an execution phase that evaluates exactly one denoiser per step according to a calibrated switching policy. The following section 3.1 formalizes the compatible sampling framework, while section 3.2 details reference-trajectory calibration and derives the disagreement-based switching policy.

## Preliminaries

Video difusion sampling. Let $\boldsymbol { x } _ { t _ { i } } \in \mathbb { R } ^ { C \times F \times H \times W }$ denote a noisy video latent at denoising-step index $i ,$ where $C , F , H$ , and � denote the channel, frame, height, and width dimensions, respectively. The sampler follows a decreasing sequence of scheduler timesteps $t _ { 0 } > t _ { 1 } > \dots > t _ { N - 1 }$

Given a large model � and a small model $S ,$ along with prompt conditioning �, negative prompt ∅, and guidance scale �, the denoiser $m \in \{ L , S \}$ yields the guided prediction (with classifier-free guidance):

$$
\begin{array} { r l } & { p _ { m } ( x _ { t _ { i } } , t _ { i } ) = f _ { m } ( x _ { t _ { i } } , t _ { i } , \emptyset ) } \\ & { \qquad + w \big ( f _ { m } ( x _ { t _ { i } } , t _ { i } , \tau ) - f _ { m } ( x _ { t _ { i } } , t _ { i } , \emptyset ) \big ) , } \end{array}\tag{1}
$$

The scheduler maps this prediction to the next latent:

$$
x _ { t _ { i + 1 } } = S ( x _ { t _ { i } } , p _ { m } ( x _ { t _ { i } } , t _ { i } ) , t _ { i } , h _ { i } ) ,\tag{2}
$$

where $h _ { i }$ denotes any scheduler history required by a multistep solver. TRACK preserves the original timestep sequence, prediction target, scheduler, and scheduler-state evolution; it alters only which denoiser $m \in \{ L , S \}$ supplies $p _ { m } ( x _ { t _ { i } } , t _ { i } )$ in Eq. (2).

Compatible model pairs. TRACK operates on pairs of large and small checkpoints that can be seamlessly interchanged within the same sampling trajectory. They must share the latent representation, output shape, conditioning interfaces, prediction target, and scheduler semantics. Although their internal architectures and parameter counts may difer, their guided predictions defined in Eq. (1) must be semantically identical for the scheduler, ensuring either checkpoint can validly advance the shared latent state in Eq. (2). In practice, this applies to difusion model families which have multiple model sizes but are based on a similar training and inference process.

Ofline Per-Step Capacity Scheduling. An ofline-derived switching policy $\pi : \{ 0 , \ldots , N - 1 \} $ $\{ L , S \}$ assigns a model size to each denoising-step index. Inference then follows

$$
x _ { t _ { i + 1 } } = S \big ( x _ { t _ { i } } , p _ { \pi ( i ) } ( x _ { t _ { i } } , t _ { i } ) , t _ { i } , h _ { i } \big ) .\tag{3}
$$

With $C _ { m }$ denoting the wall-clock latency of a single evaluation of model � on the target hardware, the total denoising cost of the policy is approximately $\textstyle \sum _ { i } C _ { \pi ( i ) }$ , excluding shared text encoding, decoding, and scheduler overhead. The objective is therefore to minimize this total denoising cost by deriving a policy � that selectively assigns the small checkpoint to switchable steps while retaining the large checkpoint at quality-sensitive steps. TRACK derives this � ofline from a small one-time calibration set, then inference executes directly from the calibrated ofline policy with no further online overhead.

## Disagreement-Based Switching Policy

## Reference-Trajectory Calibration

Let $\mathcal { D } = \{ d _ { j } \} _ { j = 1 } ^ { M }$ be a small calibration set. For each prompt $d _ { j }$ , we initialize the pipeline normally and roll out the complete trajectory using only the large checkpoint. Immediately before each scheduler update, we evaluate both small and large checkpoints on the same reference latent $\boldsymbol { x } _ { t _ { k } } ^ { ( j ) }$ , timestep $t _ { k } ,$ conditioning, negative conditioning, and guidance scale. The paired predictions therefore difer only in the denoiser that produced them. After recording the pair, the large-model prediction advances the trajectory, ensuring that every subsequent probe remains on the all-large reference path.

This procedure produces one paired large–small prediction at every denoising step and for every calibration prompt. It does not modify either checkpoint or the underlying sampler. Although both denoisers are evaluated during calibration, this cost is incurred only ofline and is not part of the online inference.

## Normalized Prediction Disagreement

For calibration prompt $j$ and denoising step $k ,$ both the small and large model are evaluated on the same all-large reference latent $\boldsymbol { x } _ { t _ { k } } ^ { ( j ) }$ , yielding $p _ { S } ( x _ { t _ { k } } ^ { ( j ) } , t _ { k } )$ and $p _ { L } ( x _ { t _ { k } } ^ { ( j ) } , t _ { k } )$ , respectively. We first compute the normalized guided-prediction disagreement score. Here, the score is normalized by the prediction magnitude of the large model:

$$
r _ { j , k } = \frac { \left\| p _ { S } ( x _ { t _ { k } } ^ { ( j ) } , t _ { k } ) - p _ { L } ( x _ { t _ { k } } ^ { ( j ) } , t _ { k } ) \right\| _ { 2 } } { \operatorname* { m a x } \left( \left\| p _ { L } ( x _ { t _ { k } } ^ { ( j ) } , t _ { k } ) \right\| _ { 2 } , \epsilon \right) }\tag{4}
$$

where $t _ { k }$ is the scheduler timestep at denoising-step index $k ,$ and $\epsilon > 0$ is a small constant for numerical stability that prevents division by zero when the largemodel prediction norm is negligible. Then, TRACK computes the mean normalized score across � calibration prompts as the final disagreement score:

<table><tr><td rowspan="2">Model</td><td rowspan="2">Configuration</td><td rowspan="2">Switched / Total Steps</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Speedup Flicker ↑ Motion ↑ Subject ↑ Background ↑</td></tr><tr><td></td></tr><tr><td>Wan 2.1 [37]</td><td>All-Large Model (14B) TRACK</td><td>0/50 30/50</td><td>1.00× 1.95×</td><td>96.71 96.86</td><td>98.31 98.39</td><td>94.62 94.81</td><td>94.79 94.99</td></tr><tr><td rowspan="3">Cosmos 3 [1]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>All-Large Model (Super)</td><td>0/35</td><td>1.00×</td><td>98.96</td><td>99.41</td><td>96.77</td><td>95.39</td></tr><tr><td>TRACK (Super/Nano) TRACK (Super/Edge)</td><td>24/35 24/35</td><td>2.04× 2.73×</td><td>99.06 98.33</td><td>99.46 98.98</td><td>96.93 97.24</td><td>95.55 95.04</td></tr><tr><td rowspan="2">TurboDiffusion [42]</td><td>All-Large Model (14B)</td><td>0/4</td><td>1.00×</td><td>97.26</td><td>98.64</td><td>94.10</td><td>93.32</td></tr><tr><td>TRACK</td><td>3/4</td><td>2.69×</td><td>96.94</td><td>98.42</td><td>94.10</td><td>93.48</td></tr><tr><td rowspan="2">FastVideo [43]</td><td>All-Large Model (14B)</td><td>0/3</td><td>1.00×</td><td>98.19</td><td>99.15</td><td>95.81</td><td>95.91</td></tr><tr><td>TRACK</td><td>2/3</td><td>2.17×</td><td>97.99</td><td>99.12</td><td>96.07</td><td>96.16</td></tr></table>

Table 1 | Speed–quality tradeofs across video pipelines. Comparison between All-Large (baseline), and the TRACK method (ours) across full model pipelines (Cosmos 3 [1] and Wan 2.1 [37]) and step-distilled pipelines (TurboDifusion’s Wan 2.1 [42], FastVideo’s Wan 2.1 [43]). Performance is evaluated on the same 250 prompts for all configurations using VBench’s temporal flicker, motion smoothness, subject consistency, and background consistency [14].Our method shows significant speed-up at comparable aggregate quality.

$$
q _ { k } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } r _ { j , k } ,\tag{5}
$$

Lower $q _ { k }$ indicates closer large–small agreement and therefore a safer candidate for using small-model execution instead of large-model.

## Constrained Trajectory Divergence Minimization using Top-K Switching

The calibration score provides a non-negative trajectory divergence surrogate using the relative diference between the guided predictions of the small and large models. Let $W _ { k } = q _ { k } ^ { 2 }$ be a squared divergence cost associated with choosing the smaller model, and let $\beta _ { k }$ denote a relaxed switching variable between the small model � and large model � for denoising iteration �. We associate a benefit o $\mathrm { ~ f ~ } - C _ { 0 } \sum _ { k } \beta _ { k }$ with using the small model, where $C _ { 0 }$ is a user-specified constant representing a cost we want to minimize (e.g. latency on specific hardware). This gives us a box-constrained optimization problem:

$$
\operatorname { a r g } _ { \beta _ { 0 } , . . . , \beta _ { N - 1 } \in [ 0 , 1 ] } \sum _ { k } ( W _ { k } - C _ { 0 } ) \beta _ { k }\tag{6}
$$

where � is the total number of denoising steps.

Although Eq. (6) is a continuous optimization problem, its objective is linear and separable in $\beta _ { k }$ . There-

fore, for $W _ { k } \ne C _ { 0 }$ , an optimal solution is

$$
\beta _ { k } ^ { \star } = \left\{ { 1 , \quad W _ { k } < C _ { 0 } } , \right.\tag{7}
$$

When $W _ { k } = C _ { 0 }$ , any value in [0, 1] is optimal; in particular, an integral solution can always be selected.

For a fixed switching budget �, TRACK routes the � steps with the lowest calibrated disagreement scores $q _ { k }$ to the small model and retains the large model at the remaining steps. Since $q _ { k } \ge 0 ;$ ranking by �<sub>�</sub> is equivalent to ranking by $W _ { k } = q _ { k } ^ { 2 }$ , and choosing $C _ { 0 }$ between the �-th and (�+1)-th smallest $W _ { k }$ makes the threshold solution in Eq. (7) select exactly these � steps; specifying $C _ { 0 }$ is therefore equivalent to specifying �. This special case of the continuous relaxation thus admits a discrete optimal solution without a separate mixed-integer formulation, and provides direct control over the compute budget by specifying the number, or equivalently the fraction, of denoising steps assigned to the small model.

## Inference

During inference, generation strictly follows the offline pre-computed policy � via Eq. (3). The pipeline evaluates exactly one checkpoint per step, seamlessly passing the shared latent, conditioning, and scheduler state between the two models. Because the step allocation is determined entirely ofline, inference requires no learned routing modules or simultaneous dual-model evaluations. Assuming a single-pass latency of $C _ { m }$ for checkpoint �, the total denoising cost reduces directly to $\textstyle \sum _ { i } C _ { \pi ( i ) }$ , plus the standard scheduler overhead. Because policy calibration is a one-time ofline setup for a given sampler configuration, it introduces zero overhead during actual deployment.

![](images/b145681443478ee780b1f32255ea340542ef6400cbbdd64c055058d04cbf0488.jpg)  
Figure 2 | Speed–quality tradeofs across video pipelines. Each panel compares the all-large model (baseline) and TRACK (ours) switching policy. All models retain comparable aggregate quality at the selected operating points in most quality dimensions.

## Experimental Results

## Experimental Setup

Pipelines and Switching Budgets. We evaluate TRACK on four text-to-video pipelines using their native inference pipelines. We pair the large and small checkpoints of Wan 2.1 (14B/1.3B, 50 steps) [37], Cosmos 3 (Super/Nano and Super/Edge, 35 steps) [1], FastVideo (14B/1.3B, 3 steps) [43], and TurboDifusion (14B/1.3B, 4 steps) [42]. All pipelines generate 81-frame, 480 × 832 videos. Through an ablation study (Appendix A), we select the operating-point switching budget � as the maximum number of smallmodel steps that maintains quality comparable to the all-large baseline.

Evaluation Protocols. We evaluate speed and quality across 250 text-video prompt pairs from VBench [14], EvalCrafter [19], T2V-CompBench [35], and manually designed prompts. For quality and temporal consistency, we report VBench’s temporal flicker, motion smoothness, subject consistency (DINO ViT-B/16 [5]), and background consistency. Generative diversity is measured via mean pairwise DreamSim distance [9] over ten prompts with eight shared seeds per policy.

![](images/3d0423158dd7056b3ceee079fc01c66813fcc09c07a4c7c416cb1be61d68c43b.jpg)  
Figure 3 | Human evaluation study. Human evaluators compare TRACK outputs against the corresponding all-large model across Wan 2.1, Cosmos 3, TurboDifusion, and FastVideo. Each horizontal bar reports the percentage of comparisons rated as TRACK Win, Tie, or Lose against the all-large baseline.

Human Evaluation. We conducted a pairwise human evaluation comparing TRACK switching against the all-large baseline across all four pipelines. Evaluators were shown side-by-side videos generated from the same prompt and asked to judge which was better on two axes: visual quality and prompt adherence. We collected a minimum of 150 judgments per pipeline (30 prompts × 5 raters each).

## Quantitative and Qualitative Results

Table 1, together with Figure 2, summarizes the selected operating point for each pipeline. TRACK accelerates all four families, spanning 50-, 35-, four-, and three-step trajectories, without changing their native schedulers or step counts. On the 50-step Wan 2.1 pipeline, the calibrated switching policy achieves a 1.95× speedup. For Cosmos 3, the policy yields a 2.04×–2.73× speedup depending on whether Nano or Edge model is chosen for the smaller model. TRACK is also efective with step-distilled pipelines, achieving a 2.17× speedup on 3-step FastVideo and a 2.69× speedup on 4-step TurboDifusion.

In terms of quality, Wan 2.1 preserves subject and background consistency relative to the all-large model. Cosmos 3 retains subject and background consistency as well, while TurboDifusion and FastVideo are similarly comparable across all metrics, demonstrating that our model switching approach remains efective after few-step distillation.

![](images/003dcf8208b97df32526ac2e180c2d9af31b1c595eb14834bd76e2aa65a9e7b8.jpg)

![](images/48a1c334928f2e5a24df5ae456112c6a67109259227426568f00150fa5d56e5c.jpg)

![](images/7eca06ae59e4dfd85331333374ba0933af7f565896f2030ff82410c19ed8dea7.jpg)  
Figure 4 | Analysis of Wan 2.1’s disagreement scores across denoising steps. Disagreement scores along the denoising trajectory for Wan 2.1 (14B vs. 1.3B), averaged across 150 prompts at 50 steps. (A) Scalar disagreement score. The green lines denote top-K steps with lowest disagreement score suitable for switching. (B, C) The same score decomposed over spatial and temporal frequency bands.

These results show that TRACK reduces the cost of the denoising evaluations on both conventional many-step sampling and aggressive step distillation. Crucially, ofline calibration on a small prompt set generalizes seamlessly to the full evaluation benchmark without manual step tuning. This demonstrates that a disagreement-based switching policy provides a stable, prompt-independent criterion for automated ofline model allocation across both many-step and step-distilled video difusion systems.

## User Study

Figure 3 summarizes the results of the human evaluation study. On prompt adherence, evaluators found the two pipelines indistinguishable in the majority of comparisons across all models (67–79% tie rate), with win and lose rates near parity. On visual quality, judgments were more evenly distributed, but the TRACK switching approach matched or edged the full model: TRACK wins are comparable or slightly exceeded full-model wins on TurboDifusion (37.7% vs. 34.7%), FastVideo (38% vs. 35.3%), Cosmos 3 (32.0% vs. 26.0%), and Wan 2.1 (33.3% vs. 38.0%), with Cosmos 3 showing the largest tie share (42.0%). Across both dimensions, no model family showed a consistent preference for the full model, confirming that the speedups reported in Table 1 come without a perceptible quality penalty. Figure 5 illustrates qualitatively the videos generated by our method and baseline. Further qualitative results can be found in Figure 8 in the Appendix.

## Analysis of Disagreement across Denoising Steps

We visualize the disagreement scores across difusion steps to understand at which steps the small model can replace the large one. Figure 4A shows the disagreement curve for Wan 2.1 over a 50-step denoising trajectory, averaged across 150 prompts. The error is high at the initial and final steps and low in between, with the minimum falling in the latter half of the trajectory. We observe the same pattern for Cosmos 3 (Figure 7 in $\operatorname { A p p e n d i x } )$ . For these models, switching to the small model is therefore safest in the middle steps.

To understand what drives this landscape, we decompose the disagreement scores over spatial and temporal frequency bands (low, mid, high) using an orthonormal Fourier transform. Figures 4B and 4C show that the early-step disagreement is dominated by low-frequency components, whereas the late-step error is dominated by mid and high frequencies. This is consistent with the coarse-to-fine nature of the denoising process: early steps commit to global structure, so any disagreement between the two models appears in the low-frequency band, while late steps refine fine-grained detail, shifting the disagreement to higher frequencies. The disagreement is lowest in the middle of the trajectory - mostly on the latter half, where the coarse structure is already fixed and fine detail has not yet been resolved. Hence, these steps are the most promising for switchability without causing significant disagreement between the large and small models. Appendix reports additional disagreement landscapes and frequency analyses for Cosmos 3 [1], TurboDifusion [42], and FastVideo [43].

## Equal-Budget Policy Ablations and Diversity Retention Analysis

To isolate the efect of where model switching occurs from the overall amount of compute reduction, we compare TRACK against three equal-budget switching baselines. This ablation helps us understand how our policy compares to other naive or heuristic approaches at a pre-determined number of switching steps. Specifically, each policy replaces exactly � out of the � denoising steps with the small model: First-� replaces the initial � steps, Last-� replaces the final � steps, and Random-� randomly selects � steps. In contrast, TRACK Top-� selects the � steps with the lowest calibrated disagreement scores �<sub>�</sub>. Here, First-� is aligned with the T-stitch [27] method, in which the first steps use the small model then switch entirely to the large model for the rest. On the other hand, Last-� is conceptually similar to SRDifusion’s [6] approach, in which the first steps use large model while using small model for the rest. One nuanced diference is, Last-� fixes the number of switching steps while SRDifusion dynamically determines the number switching steps based on input prompts. We also conduct additional experiments comparing our method and SRDifusion [6] in the Appendix.

<table><tr><td>Policy</td><td>Flicker ↑</td><td>Motion ↑</td><td>Subject ↑</td><td>Background ↑</td><td>DreamSim Diversity Retention ↑</td></tr><tr><td>All-Large</td><td>96.30</td><td>98.08</td><td>94.28</td><td>94.95</td><td>100.0%</td></tr><tr><td>All-Small</td><td>96.35 (+0.05%)</td><td> $9 7 . 6 3 \ ( - 0 . 4 6 \% )$ </td><td>93.83 (-0.48%)</td><td>94.17 (−0.82%)</td><td>79.4% (-20.6%)</td></tr><tr><td>First-K</td><td> $9 5 . 3 9 \ ( - 0 . 9 5 \% )$ </td><td> $9 6 . 9 0 \ ( - 1 . 2 0 \% )$ </td><td> $9 3 . 7 7 \ ( - 0 . 5 4 \% )$ </td><td> $9 4 . 1 4 \ ( - 0 . 8 5 \% )$ </td><td>84.6% (-15.4%)</td></tr><tr><td>Last-K</td><td> $9 6 . 0 4 \ \dot { ( } - 0 . 2 7 \% )$ </td><td> $9 7 . 8 9 \ ( - 0 . 1 9 \% )$ </td><td>94.05 (−0.24%)</td><td> $9 4 . 5 1 \ ( - 0 . 4 6 \% )$ </td><td>97.8% (-2.2%)</td></tr><tr><td>Random-K</td><td>95.95 (-0.36%)</td><td>97.55 (-0.54%)</td><td>93.91 (−0.39%)</td><td>94.49 (−0.48%)</td><td>85.7% (−14.3%)</td></tr><tr><td>TRACK Top-K</td><td>96.47 (+0.18%)</td><td>98.14 (+0.06%)</td><td>94.41(+0.14%)</td><td>94.88 (-0.07%)</td><td>98.3% (-1.7%)</td></tr></table>

Table 2 | Equal-budget switching policy comparison. We compare TRACK Top-� policy against heuristic policies (First-�, Last-�, and Random-�) on Wan 2.1 with � = 30, with each policy replacing exactly � denoising steps with the small model. All switching policies therefore use the same number of largeand small-model evaluations. Temporal flicker, motion smoothness, subject consistency, and background consistency are evaluated on a 150-prompt evaluation set. DreamSim diversity [9] is evaluated separately using ten prompts with eight shared seeds per policy. TRACK Top-� provides the best performance with the same switching budget.

![](images/6254f9ed11d790b2fac892ef7868dcfb171e142cdf4347423716915e270befbc.jpg)  
Figure 5 | Qualitative comparison. Comparing frames from videos generated with TRACK switching policy and the all-large 14B model for Wan 2.1. Our approach preserves the scene composition, subject identity, and motion pattern of the large-model outputs.

All policies use identical prompts and total numbers of large- and small-model evaluations. Besides quality comparison, we also evaluate generative sample diversity without conflating it with semantic or temporal drift. We measure mean pairwise DreamSim distance [9] over eight seeds, four frames per video, and ten prompts. Following established diversity protocols [10], we adapt five broad prompt categories from their benchmark and supplement them with three

## motion-heavy prompts and two control prompts.

Table 2 summarizes quality and diversity metrics across these switching policies on Wan 2.1 (� = 30/50). We find that in both quality and diversity retention metrics, our approach outperforms all switching policies compared. We outperform First-K by a particularly large margin in diversity retention, likely because generation diversity primarily stems from the large model’s predictions in the initial steps [10], which First-K has replaced with the small model. Compared to Last-K (and additionally SRDifusion in the Appendix Section D), our method also outperforms across quality and diversity metrics. This might be explained by our analysis in section 4.4, which shows that the last difusion steps contain high disagreement between large and small model when finalizing the videos’ high-frequency details - hence, the pipeline should switch back to the large model. This observation once again illustrates our approach’s flexibility when switching models across steps, compared to other methods’ one-time handofs—either small-to-large or large-to-small. Table 2 confirms the efectiveness of the switching policy built on evidencebased calibration analysis over heuristic policies.

Regarding diversity retention, since in our approach, the switchable steps are mostly in the middle difusion steps, all initial steps use the large model. This might explain why our method maintains high diversity retention compared to other methods, confirming the findings in [10]. Further representative multi-seed visual generations, as well as additional evaluations following the diversity evaluation protocol from [10], are reported in Section F of the Appendix.

## Conclusion

We introduced TRACK, a training-free approach for converting compatible, independently trained large and small video difusion checkpoints into a switching denoising system. TRACK calibrates the checkpoints on the same all-large reference latents, uses a normalized guided-prediction disagreement score to identify switchable denoising steps, and executes a switching policy with exactly one model evaluation per step. It therefore changes neither model weights nor scheduler behavior and introduces no online dual-model comparison.

Across Wan 2.1, Cosmos 3, FastVideo, and TurboDifusion, TRACK provides 1.95×–2.73× speedups at selected comparable-quality operating points. The gains on three- and four-step pipelines show that our method complements step-distillation by further lowering the average cost of the distilled steps. Spatial and latent-temporal analyses further show that large– small disagreement is structured across the trajectory, providing an empirical and intuitive explanation for model-specific step selection. We further tested our approach against other heuristic switching policies with the same step budget and demonstrated that our method outperforms these baselines in both quality and diversity metrics. Together, these results establish TRACK’s model-switching policy as a practical additional axis for accelerating video difusion while maintaining quality.

## References

[1] Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical AI. arXiv preprint arXiv:2606.02800, 2026.

[2] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. edif-i: Text-to-image difusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

[3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video difusion: Scaling latent video difusion models to large datasets. In International Conference on Learning Representations, 2024.

[4] Daniel Bolya and Judy Hofman. Token merging for fast stable difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4599–4603, 2023.

[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9650–9660, 2021.

[6] Shenggan Cheng, Yuanxin Wei, Lansong Diao, Yong Liu, Bujiao Chen, Lianghua Huang, Yu Liu, Wenyuan Yu, Jiangsu Du, Wei Lin, and Yang You. Srdiffusion: Accelerate video difusion inference via sketching-rendering cooperation. arXiv preprint arXiv:2505.19151, 2025.

[7] Prafulla Dhariwal and Alexander Nichol. Difusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780– 8794, 2021.

[8] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with difusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023.

[9] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Advances in Neural Information Processing Systems, 2023.

[10] Rohit Gandikota and David Bau. Distilling diversity and control in difusion models. In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1304–1313. IEEE, 2026.

[11] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic video generation with difusion models. arXiv preprint arXiv:2312.06662, 2023.

[12] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

[14] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianx ing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[15] Aapo Hyvärinen, Jarmo Hurri, and Patrick O. Hoyer. Natural Image Statistics: A Probabilistic Approach to Early Computational Vision. Springer, 2009.

[16] Rohit Jena, Ali Taghibakhshi, Sahil Jain, Gerald Shen, Nima Tajbakhsh, and Arash Vahdat. Elucidating optimal reward-diversity tradeofs in text-toimage difusion models. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 232–242. IEEE, 2025.

[17] Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, Masayoshi Tomizuka, et al. Streamdifusion: A pipeline-level solution for real-time interactive generation. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 12371–12380. IEEE, 2025.

[18] Feng Liang, Akio Kodaira, Chenfeng Xu, Masayoshi Tomizuka, Kurt Keutzer, and Diana Marculescu. Looking backward: Streaming video-to-video translation with feature banks. In International Conference on Learning Representations, pages 46425– 46445, 2025.

[19] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22139–22149, 2024.

[20] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chunxiao Li, and Jun Zhu. DPM-Solver: A fast ODE solver for difusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, 2022.

[21] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. In International Conference on Learning Representations, 2024.

[22] Xinyin Ma, Gongfan Fang, and Xinchao Wang. Deep-Cache: Accelerating difusion models for free. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[23] Mustafa Munir, Guihong Li, Md Mostafijur Rahman, Alex Zhang, and Radu Marculescu. From data to design: Leveraging frequency statistics for eficient neural network architectures. In 2025 IEEE/CVF

Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3199–3209. IEEE, 2025.

[24] Mustafa Munir, Sophia Zalewski, Shiqiu Liu, David Tarjan, Sushmitha Belede, Anjul Patney, and Radu Marculescu. Smoothdifusion-ve: Real-time generative video editing using adaptive feature cache. In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 8468–8478. IEEE, 2026.

[25] Zizheng Pan, Jianfei Cai, and Bohan Zhuang. Stitchable neural networks. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16102–16112. IEEE, 2023.

[26] Zizheng Pan, Jing Liu, Haoyu He, Jianfei Cai, and Bohan Zhuang. Stitched vits are flexible vision backbones. In European Conference on Computer Vision, pages 258–274. Springer, 2024.

[27] Zizheng Pan, Bohan Zhuang, De-An Huang, Weili Nie, Zhiding Yu, Chaowei Xiao, Jianfei Cai, and Anima Anandkumar. T-Stitch: Accelerating sampling in pre-trained difusion models with trajectory stitching. In International Conference on Learning Representations, pages 6103–6137, 2025.

[28] Namuk Park and Songkuk Kim. How do vision transformers work? In International Conference on Learning Representations, 2022.

[29] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent difusion models for high-resolution image synthesis. In International Conference on Learning Representations, pages 1862–1874, 2024.

[30] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred A. Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In Proceedings of the 36th International Conference on Machine Learning, pages 5301–5310. PMLR, 2019.

[31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[32] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial difusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.

[33] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising difusion implicit models. In International Conference on Learning Representations, 2021.

[34] Yuda Song, Zehao Sun, and Xuanwu Yin. SDXS: Real-time one-step latent difusion models with image conditions. arXiv preprint arXiv:2403.16627, 2024.

[35] Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2v-compbench: A comprehensive benchmark for compositional text-tovideo generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8406–8416, 2025.

[36] Ali Taghibakhshi, Ruisi Cai, Saurav Muralidharan, Sharath Turuvekere Sreenivas, Ameya Sunil Mahabaleshwarkar, Marcin Chochowski, Akhiad Bercovich, Ran Zilberstein, Ran El-Yaniv, Yonatan Geifman, Daniel Korzekwa, Yoshi Suhara, Oluwatobi Olabiyi, Ashwath Aithal, Nima Tajbakhsh, and Pavlo Molchanov. Star elastic: Many-in-one reasoning LLMs with eficient budget control. In Forty-third International Conference on Machine Learning, 2026.

[37] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. Wan: Open and advanced large-scale video generative mod els. arXiv preprint arXiv:2503.20314, 2025.

[38] Haohan Wang, Xindi Wu, Zeyi Huang, and Eric P. Xing. High-frequency component helps explain the generalization of convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8684– 8694, 2020.

[39] Xingyi Yang, Daquan Zhou, Jiashi Feng, and Xinchao Wang. Difusion probabilistic model made slim. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22552–22562. IEEE, 2023.

[40] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. CogVideoX: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

[41] Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step difusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024.

[42] Jintao Zhang, Kaiwen Zheng, Kai Jiang, Haoxu Wang, Ion Stoica, Joseph E Gonzalez, Jianfei Chen, and Jun Zhu. Turbodifusion: Accelerating video difusion models by 100-200 times. arXiv preprint arXiv:2512.16093, 2025.

[43] Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong Liu, and Hao Zhang. Fast video generation with sliding tile attention. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

[44] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of difusion models. Advances in Neural Information Processing Systems, 36: 49842–49869, 2023.

[45] Xuanlei Zhao, Xiaolong Jin, and Yang You. Realtime video generation with pyramid attention broadcast. In Advances in Neural Information Processing Systems, 2024.

## Accelerating Video Difusion via Training-Free Trajectory Routing Supplementary Material

## Ablation on Determining the Switching Budget (�)

To understand the tradeof between compute reduction and generation quality, which helps determine the optimal switching budget, we conduct an ablation study varying the switching budget � (the number of denoising steps routed to the small model) across all four pipelines. For this analysis, we report the average of temporal flicker, motion smoothness, subject consistency, and background consistency scores, providing a unified metric for visual fidelity. The switching policies are generated using the TRACK Top-� selection criterion derived in Section 3, meaning steps are routed in order of lowest normalized disagreement score.

![](images/0c3f236980a00acae669a73db9dc1406b0644aa537ef27f315bc881915a481c6.jpg)

![](images/65ec9104abfbb42729eac1c90ffda653864f4e29077ab6de838b36a017ce43f5.jpg)

![](images/ba2b3c8171a057f300252c3c08e1fc21998b676f8f799ce7f98c4012ab422c84.jpg)

![](images/3ccb3b07e5551adabf15dd3aa15993f47696359303e29d3230d3cbd944299047.jpg)  
Figure 6 | Efect of trajectory routed budget steps (�) on visual quality. We evaluate the average of the temporal flicker, motion smoothness, subject consistency, and background consistency scores from VBench across varying numbers of small-model steps (�). From this study, we can determine the maximum � used in the TRACK algorithm before the quality starts to degrade.

Figure 6 reports the visual consistency across varying budgets. The results demonstrate a clear “plateau of switchability” for video difusion models. As � increases from 0 (the all-large baseline), visual consistency initially remains stable and can even improve. This confirms that a substantial portion of the denoising trajectory simply does not require the representational capacity of the largest network.

However, as � approaches the total number of sampling steps �, the capacity reduction begins to encroach on quality-sensitive steps, causing a sharp degradation in visual fidelity. For instance, increasing the budget to � = 28 on Cosmos 3 or � = 40 on Wan 2.1 severely penalizes consistency.

![](images/f7bb76973777156e06eb327ed222ea41fcfac86b71109b54ac74752a88d6ca3e.jpg)  
Figure 7 | TRACK disagreement score across denoising steps analysis across models. Disagreement score error for Cosmos 3 (A–C, 35 steps), TurboDifusion (D–F, 4 steps), and FastVideo (G–I, 3 steps), averaged across prompts. Columns in order of: scalar disagreement score; spatial-band frequencies amplitude; temporal-band frequencies amplitude.

## Extended Analysis of Disagreement Scores Across Denoising Steps

Figures 7A–C show that Cosmos 3 closely follows the behavior of Wan 2.1. The disagreement score is high at the initial and final steps and reaches its minimum in the latter half of the trajectory. The band decomposition reproduces the same frequency handof: early-step disagreement is concentrated in the low band, while the late-step rise is carried by the mid and high bands. The U-shaped disagreement landscape is therefore not specific to Wan 2.1, and switching is again best placed in the middle steps. TurboDifusion [42] and FastVideo [43] show a diferent landscape (Figures 7D–I). The score is largest at the first step and decreases monotonically until the last, with no late-step rise. The band decomposition shows that the disagreement remains low-frequency dominated across all steps — the low band carries the majority of the error energy at every step.

## Energy Consumption and Eficiency Gains

Beyond accelerating inference latency, TRACK’s inference eficiency can be translated into substantial reductions in GPU energy consumption. This is a critical metric for the deployment of generative video models at scale, where continuous execution of multibillion parameter denoisers incurs heavy power and thermal costs.

Measurement Protocol. We measure the hardware energy consumption of the denoising loop using the NVIDIA Management Library (NVML). For each generated video �, we read the cumulative GPU energy counter immediately before and after the denoising trajectory. The per-video energy consumption in joules is calculated as:

$$
E _ { i } = { \frac { \mathrm { N V M L } _ { \mathrm { e n d } } - \mathrm { N V M L } _ { \mathrm { s t a r t } } } { 1 0 0 0 } } .\tag{8}
$$

To ensure a rigorous evaluation, we compute the total percentage of energy saved as the ratio of aggregate means across the entire evaluation set, rather than the unweighted average of individual per-video percentages. Let $\bar { E } _ { \mathrm { b a s e } }$ and $\bar { E } _ { \mathrm { T R A C K } }$ denote the mean pervideo energy consumption for the all-large baseline and the TRACK policy, respectively. The aggregate energy savings percentage is defined as:

$$
\mathrm { S a v i n g s \ ( \% ) } = \left( 1 . 0 - \frac { \bar { E } _ { \mathrm { T R A C K } } } { \bar { E } _ { \mathrm { b a s e } } } \right) \times 1 0 0 .\tag{9}
$$

All measurements are recorded on a single NVIDIA A100 GPU, isolating the denoising loop and excluding fixed overheads such as text encoding and VAE decoding.

Results. Table 3 summarizes the energy reductions across all evaluated pipelines. TRACK cuts the GPU energy footprint by approximately half across the board.

For many-step, high-capacity pipelines, the absolute energy savings are significant. On Wan 2.1 (14B to 1.3B), TRACK saves 194.57 kJ per video, translating to 54.05 kWh saved per 1,000 videos generated (a 49.3% reduction). On Cosmos 3 (Super/Edge), TRACK reduces energy consumption by nearly 60%, saving 35.41 kWh per 1,000 videos.

Crucially, TRACK remains highly efective even on heavily distilled pipelines where the total step count is already minimized. On the three-step FastVideo pipeline and four-step TurboDifusion pipeline, TRACK yields 54.4% and 45.6% energy savings, respectively. At scale, this prevents over 1 kWh of energy waste per 1,000 videos without requiring any additional training or distillation. In practice, video difusion pipelines generate millions of videos, and these energy savings scale directly with the number of generated videos.

<table><tr><td>Pipeline</td><td>Steps  $( K / \bar { N } )$ </td><td>Base (kJ)</td><td>TRACK (kJ)</td><td>Saved (kJ)</td><td>Saved (%)</td><td>1K Vids (kWh)</td></tr><tr><td>TurboDiff. [42]</td><td>2/4</td><td>8.31</td><td>4.52</td><td>3.79</td><td>45.6%</td><td>1.05</td></tr><tr><td>FastVideo [43]</td><td>2/3</td><td>9.44</td><td>4.30</td><td>5.14</td><td>54.4%</td><td>1.43</td></tr><tr><td>Cosmos 3 (S/N) [1]</td><td>24/35</td><td>204.22</td><td>97.07</td><td>107.15</td><td>52.5%</td><td>29.77</td></tr><tr><td>Cosmos 3 (S/E) [1]</td><td>24/35</td><td>213.40</td><td>85.93</td><td></td><td>127.4659.7%</td><td>35.41</td></tr><tr><td>Wan 2.1 [37]</td><td>30/50</td><td>394.43</td><td>199.86</td><td></td><td>194.57 49.3%</td><td>54.05</td></tr></table>

Table 3 | GPU Energy Consumption and Savings. Denoising energy is measured via NVML on a single NVIDIA A100 GPU. Percentage savings are calculated using the ratio of the aggregate means. TRACK reduces the energy footprint of video generation by roughly 45% to 60%, saving up to 54 kWh per 1,000 videos on many-step models and over 1 kWh per 1,000 videos on aggressively step-distilled models. For Cosmos 3, S/N and S/E denote the Super/Nano and Super/Edge checkpoint configurations, respectively.

<table><tr><td>Policy</td><td>Flicker ↑</td><td>Motion ↑</td><td>Subject ↑</td><td>Background ↑</td><td>DreamSim Diversity ↑</td></tr><tr><td>All-Large</td><td>96.30</td><td>98.08</td><td>94.28</td><td>94.95</td><td>100.0%</td></tr><tr><td>SRDiffusion (0.03) [6]</td><td>95.84 (−0.48%)</td><td>97.57 (-0.52%)</td><td>93.76 (-0.55%)</td><td>94.35 (-0.63%)</td><td>87.1% (−12.9%)</td></tr><tr><td>SRDiffusion (0.01) [6]</td><td>95.98 (−0.33%)</td><td>97.73 (−0.36%)</td><td>93.94 (−0.36%)</td><td>94.63 (−0.34%)</td><td>92.6% (-7.4%)</td></tr><tr><td>TRACK Top-K (Ours)</td><td>96.47 (+0.18%)</td><td>98.14(+0.06%)</td><td>94.41(+0.14%)</td><td>94.88 (-0.07%)</td><td>98.3% (-1.7%)</td></tr></table>

Table 4 | TRACK vs. SRDifusion Comparison. We compare our TRACK Top-� policy (� = 30) and two threshold variants of SRDifusion [6] on Wan 2.1. Bold indicates the best performing method.

## Additional Quantitative Results

In this section, we provide a direct, isolated comparison between our proposed TRACK switching policy and SRDifusion [6]. SRDifusion determines its switching point using an online, prompt-adaptive threshold based on the latent rate of change. We compare TRACK against two variants of SRDifusion (using threshold values of 0.01 and 0.03).

As shown in Table 4, TRACK outperforms the SRDiffusion baselines across both video consistency metrics and sample diversity retention. Because SRDifusion enforces a permanent, monotonic handof to the small model, it struggles to preserve the high-frequency temporal details in the final denoising steps, leading to loss of DreamSim diversity. By dynamically placing the small model at optimal low-disagreement steps and returning to the large model when necessary, TRACK preserves the large model’s visual fidelity and diversity.

Beyond the numerical metrics, visual inspection confirms that the structural integrity of the generated videos remains intact. As shown in Figure 5, the TRACK switching policy seamlessly preserves the scene composition, subject identity, and motion patterns of the all-large baseline outputs compared to contiguous handof methods.

## Additional Qualitative Results

In Figure 8, we provide additional side-by-side video frame comparisons across all four model families: Wan 2.1, Cosmos 3, TurboDifusion, and FastVideo. For each model, we show frames generated from the same prompt using the all-large baseline and the TRACK switching policy. Across all families, TRACK preserves scene composition, subject identity, and motion patterns of the large-model outputs, while delivering the speedups reported in Table 1.

TRACK (Ours)

![](images/af35cf1d9dc7fbfdae742819f387c484134bfa5dc50595b49db512231108e24c.jpg)  
Figure 8 | Qualitative comparison across all four base models. Four frames at evenly-spaced timestamps are shown for three prompts generated with each model under two configurations: the full large-model pipeline (baseline) and our TRACK pipeline (Ours). Across all four models: Wan 2.1, Cosmos 3, TurboDifusion, and FastVideo, TRACK’s switching policy preserves scene composition, subject identity, and motion patterns of the large-model outputs, while reducing inference latency.

## Extended Diversity Preservation Analysis

When aggressively switching large-model evaluations with a smaller checkpoint, a critical concern is whether the system sufers from mode collapse or loss of sam ple diversity across diferent initial noise seeds. As shown qualitatively in Figure 9, TRACK successfully preserves the varied subjects, compositions, and styles of the all-large baseline across diferent random seeds, whereas naive heuristic policies like First-� can severely shift or narrow the output modes.

Following the diversity evaluation protocol established in recent literature [10], we measure the mean pairwise DreamSim distance across generated outputs. Higher DreamSim distances indicate greater compositional and semantic diversity across seeds.

Evaluation Protocol and Baselines. We compute the DreamSim distance using five standardized prompt categories utilized in prior difusion diversity studies [10] (e.g., sunset beach, puppy, futuristic city, person, and Van Gogh art). To isolate the impact of our switching policy, we compare video generation pipelines against the image-based SDXL baseline reported in [10]. Because video evaluation averages features across multiple frames (four frames per video across eight seeds), absolute DreamSim values difer across modalities; therefore, the primary metric of interest is the retention percentage relative to the all-large baseline. As reported in Table 5, TRACK preserves > 95% of the original large-model diversity across all four video pipelines, successfully mirroring the high retention dynamics observed in image-based models like SDXL [29].

<table><tr><td></td><td colspan="3">Mean Pairwise DreamSim (↑)</td></tr><tr><td>Pipeline</td><td>Base</td><td>Hybrid</td><td>Retention</td></tr><tr><td>SDXL [29]</td><td>0.337</td><td>0.350</td><td>103.9%</td></tr><tr><td>FastVideo [43]</td><td>0.271</td><td>0.284</td><td>104.8%</td></tr><tr><td>TurboDiff. [42]</td><td>0.607</td><td>0.581</td><td>95.7%</td></tr><tr><td>Wan 2.1 [37]</td><td>0.505</td><td>0.502</td><td>99.4%</td></tr><tr><td>Cosmos 3 [1]</td><td>0.550</td><td>0.548</td><td>99.5%</td></tr></table>

Table 5 | Sample Diversity Retention. We report mean pairwise DreamSim distance on a standardized 5-category prompt set. FastVideo, TurboDifusion, Wan 2.1, and Cosmos 3 successfully reproduce the retention dynamics observed in 2D image difusion, preserving > 95% of large-model diversity when the highest capacity model is retained for critical denoising steps.

(a) Wan 2.1  
![](images/adf53616d2ef248f394c24f15e6ffc19ba9b55403cae6e6ae0a9de6510f13529.jpg)  
Figure 9 | Representative multi-seed generations used in the diversity analysis. We show midpoint frames from four seeds per prompt, with each column using the same random seed across rows. Rows compare the all-large reference, a same-budget First-� schedule, and TRACK; both switching policies use the same number of small-model evaluations. The top grid shows two prompts for Wan 2.1 with deterministic UniPC sampling [44], while the lower grids show three-step FastVideo with deterministic UniPC sampling and four-step TurboDifusion with stochastic rCM sampling. TRACK preserves the varied subjects, compositions, and styles of the all-large reference, whereas First-� shifts or narrows its output modes.

## Limitations and Future Work

TRACK, similar to other difusion model switching methods, currently still requires storing both the large and small models simultaneously, increasing deployment memory compared to one single-model pipeline. A natural direction for future work is to integrate TRACK with nested model families as introduced in the language domain [36], where the small denoiser is realized as a subnetwork of the large denoiser sharing the same parameters. Under such a design, the full and reduced capacity models are contained within a single checkpoint, eliminating the memory overhead of maintaining two separate models while preserving the flexibility switching benefits demonstrated here.