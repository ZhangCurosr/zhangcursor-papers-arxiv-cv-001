# GDB-Reward: From Evaluation Metrics to Training Rewards for Graphic Design

Adrienne Deganutti<sup>1,2</sup>, Purvanshi Mehta<sup>1</sup>, Simon Hadfield<sup>2</sup>, Andrew Gilbert<sup>2</sup> <sup>1</sup>Lica World, <sup>2</sup>University of Surrey

## Abstract

Text-to-image models excel at natural image synthesis but struggle with graphic design, where success depends on satisfying precise constraints on typography, layout, color, and visual communication. While prompt optimization offers an attractive alternative to expensive diffusion model fine-tuning, learning prompts for frozen image generators requires informative reward functions despite the entirely non-differentiable generation process. Reinforcement learning does not require differentiable objectives; it requires only scalar rewards capable of ranking candidate outputs. This raises a simple question: can design evaluation metrics themselves become reinforcement learning rewards? Our central contribution is GDB-REWARD, a framework that systematically transforms heterogeneous graphic design evaluation metrics into a unified reinforcement learning reward. Experiments demonstrate that GDB-REWARD provides an effective optimization objective, substantially improving adherence to the design specification in perceptual quality, rendering fidelity, and spatial accuracy while keeping the image generator entirely frozen. More broadly, our results demonstrate that heterogeneous, non-differentiable evaluation metrics can move beyond passive benchmarking to become effective optimization objectives for reinforcement learning in domains where differentiable supervision is unavailable.

## 1 Introduction

Text-to-image (T2I) models generate photorealistic and stylistically rich imagery. However, graphic design requires meeting precise, structured constraints, including typography, layout, color consistency, and communicative intent. This poses a fundamentally different generation challenge. These requirements make faithful reconstruction substantially more difficult than generating natural images. A generated design may appear visually plausible yet fail to reproduce the required text, colors, or visual structure, making graphic design generation a matter of specification adherence rather than purely perceptual realism.

A natural solution could be to fine-tune the image generator itself. However, doing so requires substantial computational resources and large collections of paired, high-quality graphic design data, both of which are expensive to obtain. We instead ask a different question: can we improve specification adherence without modifying the image generator?

![](images/fc0f1b3d959b34f501111c233e0dfcb18859e6eca07a576baec0c3e9dc45f741.jpg)  
Figure 1: Overview of the proposed framework. Standard text-to-image models generate visually plausible images but fail to satisfy explicit design specifications. We optimize a prompt policy with reinforcement learning using GDB-REWARD, a composite reward derived from design-specific evaluation metrics. This improves design-specification adherence while keeping the image generator frozen.

The central challenge is defining an optimization objective for a completely non-differentiable generation pipeline. Unlike supervised fine-tuning, prompt optimization cannot rely on a reconstruction loss because the prompt generator never observes gradients from the image generator. Instead, learning must be driven by feedback that reflects how well the generated design satisfies the intended specification. Defining such a reward is challenging because graphic design quality is inherently multi-dimensional, requiring simultaneous assessment of typography, color, layout, semantics, and visual quality.

Recent work has begun to formalize graphic design evaluation through design-specific metrics. In particular, GraphicDesignBench (Deganutti et al. 2026) proposes a metric taxonomy purpose-built for design evaluation that spans spatial accuracy, typographic fidelity, structural validity, and human-aligned preference, moving beyond the perceptual quality scores (FID, CLIPScore) that dominate existing image benchmarks but fail to capture design-specific requirements. Although these metrics provide meaningful post hoc evaluation, they are not designed to serve as learning objectives because they are heterogeneous, largely nondifferentiable, and combine discrete and continuous signals.

We observe that reinforcement learning does not require a differentiable objective. Instead, it requires only a scalar reward capable of ranking generated outputs. This observation changes the role of evaluation metrics: rather than designing a new reward from scratch, we transform a bank of design evaluation metrics into a unified reward for prompt optimization. This suggests a broader paradigm shift: evaluation metrics can function as optimization objectives rather than solely as passive benchmarks.

We propose GDB-REWARD, a framework for transforming heterogeneous design evaluation metrics into a unified reinforcement learning reward. A lightweight language model rewrites structured layout metadata into natural language prompts for a frozen text-to-image model. Generated designs are scored using GDB-REWARD, and Group Relative Policy Optimization (GRPO) updates only a LoRA adapter on the prompt generator while leaving the image generator and all reward models frozen. In this way, our approach improves adherence to the design specification without requiring expensive diffusion model fine-tuning.

Our contributions are as follows:

• We propose GDB-REWARD, a composite, nondifferentiable reward that converts graphic design evaluation metrics spanning perceptual quality, rendering fidelity, and spatial accuracy into a unified optimization objective.

• We formulate graphic design generation as a prompt optimization problem over frozen text-to-image models, avoiding diffusion model fine-tuning through reinforcement learning.

• We develop a design-intent captioning model that evaluates whether generated designs preserve the underlying communicative goal, providing an independent semantic evaluation complementary to the optimization reward.

• We demonstrate that reinforcement learning with GDB-REWARD substantially improves design-specification adherence, particularly for typography, color fidelity, and layout structure, while training only a lightweight LoRA adapter and leaving the image generator entirely frozen.

## 2 Related Works

Graphic Design Generation. Text-to-image diffusion models excel at natural image synthesis; however, graphic design generation remains fundamentally different: success is judged not by whether an image is visually plausible but by whether it reproduces structured constraints. Multimodal systems address this by generating whole, editable designs (Cheng et al. 2024; Jia et al. 2024; Qu et al. 2025); however, these approaches train or fine-tune specialized diffusion models on paired graphic-design data, which is costly to curate and tied to a single generator. In contrast, we leave the image generator frozen and instead optimize the prompts presented to it.

Evaluation of graphic design. Evaluating graphic design is a complex task because no single metric captures every axis of quality. Image evaluation has long relied on perceptual similarity measures such as SSIM (Wang et al. 2004) and LPIPS (Zhang et al. 2018), while more recent work has introduced learned models of aesthetics and preferences. Generic text-to-image preference models such as HPSv2 (Wu et al. 2023) and PickScore (Kirstain et al. 2023) underperform in assessing design quality because they primarily focus on global image aesthetics and overlook critical dimensions of typography, layout, and structural consistency. Aesthetics++ (Kong et al. 2022) attempts to improve this by using a learned preference model trained on human aesthetic preferences to refine existing designs. More recently, design-specific evaluation frameworks have been proposed to better capture the unique requirements of graphic design. PosterReward (Lai et al. 2026) introduces an LLM-based evaluator that jointly assesses layout, text rendering, and aesthetic expression, while GraphicDesignBench (Deganutti et al. 2026) consolidates deterministic and LLM-based metrics spanning perceptual quality, text fidelity, spatial accuracy, semantic alignment, and structural validity. While prior work uses these metrics solely for evaluation, to the best of our knowledge, no prior work before GDB-REWARD has investigated whether evaluation metrics themselves can serve as optimization objectives for reinforcement learning.

Prompt Optimization and Reinforcement Learning for Text-to-Image Models. Rather than adapting the diffusion model itself, recent work has explored optimizing the prompts presented to frozen generators. Methods range from prompt rewriting using instruction-tuned LLMs such as Promptist (Hao et al. 2023) to reinforcement learning-based approaches such as PromptLoop (Lee and Ye 2025), demonstrating that prompt optimization can substantially improve generation quality without modifying the underlying image model. However, existing prompt optimization methods generally assume the availability of a task-specific reward or feedback signal rather than investigating how such rewards should be constructed. The design of effective reward functions has long been recognized as a central challenge in reinforcement learning, particularly when balancing multiple competing objectives and ensuring that optimization aligns with the desired behaviour (Barto 2021). Importantly, policy-gradient methods optimize policies using scalar rewards without requiring gradients through the environment itself. Algorithms such as Proximal Policy Optimization (PPO) (Schulman et al. 2017) and, more recently, Group Relative Policy Optimization (GRPO) (Shao et al. 2024) exploit only the relative ranking of candidate actions, making them naturally suited to optimization over blackbox, non-differentiable objectives. Our work differs in that it focuses not on the optimization algorithm itself, but on how heterogeneous evaluation signals can be systematically transformed into reward functions for graphic design generation.

![](images/f50d2cb2fb1746562d81d2eb39aaee3f6633ec58602b0879327a821d2906ef62.jpg)  
Figure 2: Overview of the proposed methodology. For each graphic design, the prompt policy maps structured layout metadata into a set of candidate image-generation prompts. A frozen text-to-image model renders each prompt into a design, which is scored by GDB-REWARD using complementary measures of perceptual quality, rendering fidelity, and spatial accuracy. The normalized metrics are combined into a single reward that ranks candidate prompts within each group, enabling GRPO to optimize only the LoRA parameters of the prompt policy while keeping the image generator and reward function fixed.

Recent reinforcement learning approaches address graphic design problems but optimize different components of the pipeline. PosterReward (Lai et al. 2026) fine-tunes the diffusion model itself, LaySPA (Li et al. 2026) reinforces spatial reasoning over structured layout representations, and CreatiParser (Chen et al. 2026) applies GRPO to graphic design decomposition rather than image generation. In contrast, we optimize prompts using a reward systematically constructed from heterogeneous design evaluation metrics computed on the final rendered design, rather than assuming such a reward already exists.

## 3 Methodology

Our goal is to learn a prompt policy that converts structured design representations into natural-language instructions that elicit faithful reconstructions from a frozen text-toimage generator. Figure 2 summarizes the resulting closed loop, which is made up of four core components: the prompt generator, a frozen image generator, the design-intent model, and a composite reward, GDB-REWARD.

## 3.1 Problem Setup

Let a design sample consist of layout metadata ℓ and a rendered image $x _ { \ell }$ . The layout metadata specifies the canvas size, background, an ordered set of text elements (including their content and typographic properties), decorative image elements, and color blocks. The rendered image $x _ { \ell }$ is an exact visualization of the layout metadata, produced using HTML and CSS visualization tools.

A policy π<sub>θ</sub> (the prompt generator) maps the layout metadata ℓ to a textual prompt $p \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mathbf { \mu } | \textit { \ell } )$ . The generated prompt should preserve every on-design text string verbatim while describing the canvas, colors, and design components. A frozen text-to-image generator G then renders an image $x = G ( p )$ . A reward function $R ( \ell , x _ { \ell } , x , p ) \in [ 0 , 1 ]$ measures how faithfully the generated image x matches the target design represented by (ℓ, x<sub>ℓ</sub>). Our objective is:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { \ell } \mathbb { E } _ { p \sim \pi _ { \theta } ( \cdot | \ell ) } \left[ R ( \ell , x _ { \ell } , x , p ) \right]\tag{1}
$$

Because neither the discrete prompt tokens nor the image generator are differentiable, we optimize Eq. (1) using policy-gradient reinforcement learning with the scalar reward provided by GDB-REWARD.

## 3.2 Prompt Policy

The policy’s role is to translate structured layouts into prompts that maximize GDB-REWARD. The prompt generator is an instruction-tuned LLM equipped with a LoRA adapter (Hu et al. 2022). Restricting optimization to LoRA preserves the strong instruction-following capabilities of the underlying language model while providing sufficient flexibility for reward-driven prompt refinement. Given the layout metadata ℓ, it is prompted via a fixed system instruction (see Appendix A) to produce a single, self-contained imagegeneration prompt that describes the layout, composition, palette, style, and mood while reproducing every on-design text string verbatim. As the reinforcement learning policy, the model also provides the primitives required by GRPO: sampling a group of prompts for a given layout, and computing the corresponding log-probabilities under a frozen reference policy for the KL penalty. The reference policy is initialized from the pre-trained instruction-tuned model before optimization and remains fixed throughout training.

## 3.3 GDB-REWARD

GDB-REWARD is a framework for transforming heterogeneous evaluation metrics into unified reinforcement learning rewards. In this work, we instantiate the framework using graphic design evaluation metrics. We combine complementary signals covering three design-native dimensions drawn from the GraphicDesignBench taxonomy (Deganutti et al. 2026): perceptual quality, rendering fidelity, and spatial accuracy. Designing a reward for reinforcement learning requires satisfying three properties: (i) heterogeneous evaluation signals must be brought onto a common scale; (ii) the reward must remain numerically stable across samples; and (iii) it must preserve the relative ordering of candidate generations. Since these metrics produce heterogeneous outputs (e.g., perceptual distances, OCR accuracy, cosine similarity), each score is mapped to a normalized reward in [0, 1], where higher values indicate better performance, and the scores are then combined into a single scalar reward. The selected metrics satisfy two design principles. First, they collectively capture complementary aspects of design quality that a single metric cannot measure. Second, each produces a reliable scalar score that can be normalized and combined into an effective optimization signal for reinforcement learning.

Formally, let M denote the set of metrics with nonnegative weights $\{ w _ { k } \} _ { k \in \mathcal { M } }$ . Each metric is represented by a measurement function $\mu _ { k } : ( \ell , x _ { \ell } , x , p ) $ R which computes the raw score produced by metric k. The resulting measurement is:

$$
m _ { k } = \mu _ { k } ( \ell , x _ { \ell } , x , p )\tag{2}
$$

where ℓ is the layout metadata, $x _ { \ell }$ is the reference rendering, x is the generated image, and $p$ is the generated prompt. Because different metrics produce values in different units (e.g., LPIPS distances, OCR accuracy, or color differences), each measurement is converted into a normalized reward:

$$
r _ { k } = \phi _ { k } ( m _ { k } ) \in [ 0 , 1 ]\tag{3}
$$

where larger values always indicate better design quality. Keeping the raw measurements separate from their normalized rewards allows us to report standard evaluation metrics while presenting the reinforcement-learning algorithm with a consistent reward scale.

We organize the metric bank according to the following design objectives, summarized in Table 1.

Perceptual Quality. Perceptual quality metrics measure the visual similarity between the generated image and the intended design. We use three reference-based measures: DreamSim (Fu et al. 2023), LPIPS (Zhang et al. 2018), and SSIM (Wang et al. 2004). DreamSim and LPIPS produce perceptual distances that are converted into rewards using:

$$
\phi ( m ) = e ^ { - m / \tau } \quad \mathrm { w h e r e } \quad \tau = 2 0\tag{4}
$$

while SSIM is rescaled from [−1, 1] to [0, 1] by:

$$
\phi ( m ) = \frac { m + 1 } { 2 }\tag{5}
$$

To provide reference-free quality signals, we additionally include ImageReward (Xu et al. 2023) and NIMA (Talebi and Milanfar 2018). ImageReward scores are passed through a logistic sigmoid, while NIMA’s predicted aesthetic score is linearly mapped from [1, 10] to [0, 1].

Rendering fidelity. Graphic designs often depend on accurate typography and color reproduction. We therefore evaluate three complementary properties. Color fidelity measures the average CIEDE2000 (Sharma, Wu, and Dalal 2005) distance $( \Delta \bar { E } _ { 0 0 } )$ between the colors specified in the layout metadata and the nearest dominant color extracted from the generated image; we then convert it into a reward using Eq. (4). Typography is evaluated using OCR read-back accuracy, which compares each expected text string with the OCR output bounded in [0, 1].

Spatial Accuracy. Spatial accuracy evaluates whether the correct number of components are rendered. We detect text and image regions using a pre-trained YOLO-OBB detector from the GraphicDesignBench repository (Deganutti et al. 2026) that was trained on graphic design layouts. We compare the detections with both the reference render and the layout metadata. Component count penalizes discrepancies in the number of detected elements: $\phi ( m ) = \operatorname* { m a x } ( 0 , 1 - m )$ .

<table><tr><td>Category</td><td>Metric</td><td>Evaluation Signal</td></tr><tr><td rowspan="5">Perceptual</td><td>DreamSim</td><td>perceptual distance</td></tr><tr><td>LPIPS</td><td>perceptual distance</td></tr><tr><td>SSIM</td><td>structural similarity</td></tr><tr><td>ImageReward</td><td>learned human preference</td></tr><tr><td>NIMA</td><td>aesthetic preference</td></tr><tr><td rowspan="2">Rendering</td><td> $\Delta E _ { 0 0 }$ </td><td>color distance</td></tr><tr><td>OCR</td><td>text reconstruction</td></tr><tr><td>Spatial</td><td>YOLO Count</td><td>component count error</td></tr></table>

Table 1: The metric bank used by GDB-REWARD.

Composite Reward. Not every metric applies to every design; OCR metrics, for example, cannot be computed when no text is present. Let $A ( \ell , x ) \subseteq { \mathcal { M } }$ denote the subset of applicable metrics used to produce measurements for the current sample. The final reward is computed as the weighted arithmetic mean over these metrics. We adopt a weighted arithmetic mean rather than multiplicative or minimumbased aggregation because individual metrics capture complementary, not mandatory, properties of a successful design. Product-based aggregation would excessively penalize isolated weaknesses, while minimum-based aggregation would cause optimization to be dominated by a single poorly performing metric. The weighted average instead provides smooth trade-offs while preserving the contribution of all metrics.

$$
R ( \ell , x _ { \ell } , x , p ) = \frac { \sum _ { k \in A ( \ell , x ) } w _ { k } \phi _ { k } \big ( \mu _ { k } \big ( \ell , x _ { \ell } , x , p \big ) \big ) } { \sum _ { k \in A ( \ell , x ) } w _ { k } }\tag{6}
$$

Finally, GDB-REWARD is designed specifically for Group Relative Policy Optimization. Because GRPO standardizes rewards within each sampled group, only the relative ranking of candidate prompts affects the policy update.

<table><tr><td rowspan="3"></td><td rowspan="3"></td><td colspan="5">Perceptual quality</td><td colspan="2">Rendering fidelity</td><td colspan="2">Spatial accuracy</td></tr><tr><td colspan="3">Reference-based</td><td colspan="2">Reference-free</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>Full DS</td><td></td><td>LP</td><td>SS</td><td>IR</td><td>NM</td><td> $\Delta E _ { 0 0 }$  OCR</td><td> $\mathrm { C } _ { g }$ </td><td> $\mathrm { C } _ { m }$ </td></tr><tr><td>Base (Qwen3.5-9B + FLUX.2-dev)</td><td>0.560</td><td>0.543</td><td>0.314</td><td>0.744</td><td>0.366</td><td>0.418</td><td>0.462</td><td>0.870</td><td>0.591</td><td>0.482</td></tr><tr><td>+GDB-REWARD (trained)</td><td>0.647</td><td>0.565</td><td>0.296</td><td>0.735</td><td>0.571</td><td>0.415</td><td>0.578</td><td>0.869</td><td>0.731</td><td>0.675</td></tr><tr><td>Base (Qwen3.5-9B + FLUX.1-dev)</td><td>0.398</td><td>0.437</td><td>0.336</td><td>0.768</td><td>0.337</td><td>0.387</td><td>0.378</td><td>0.446</td><td>0.353</td><td>0.307</td></tr><tr><td>+GDB-REWARD (trained)</td><td>0.559</td><td>0.513</td><td>0.272</td><td>0.730</td><td>0.545</td><td>0.427</td><td>0.504</td><td>0.643</td><td>0.601</td><td>0.612</td></tr></table>

Table 2: Per-metric performance on the LICA test split. Entries are the mean normalized metric score $\phi _ { k } \in [ 0 , 1 ]$ (higher is better). The composite Full score is upper-bounded at 0.758 rather than 1, since reference-based perceptual metrics cannot saturate even for the ground-truth render; trained scores should be read against this ceiling. Perceptual: DS=DreamSim, LP=LPIPS, SS=SSIM, IR=ImageReward, NM=NIMA; Rendering: ${ \Delta E _ { 0 0 } } \mathrm { { = } C I E D E 2 0 0 0 }$ color distance, OCR=OCR accuracy; Spatial: $\mathrm { C } _ { g } / \mathrm { C } _ { m } { = } \mathrm { Y O L O }$ component count vs. ground-truth render / layout metadata.

Consequently, GDB-REWARD is designed to reliably distinguish better reconstructions from worse ones, rather than to produce perfectly calibrated absolute scores.

## 3.4 Policy Optimization with GRPO

We optimize the prompt generator using GRPO, treating it as the policy and GDB-REWARD as a black-box reward function. Since the generated prompt consists of discrete tokens and the text-to-image generator is non-differentiable, gradients cannot be propagated through the rendering process. Reinforcement learning instead allows the prompt generator to improve by maximizing the scalar reward assigned to the generated image.

For each layout ℓ, the policy samples a group of G candidate prompts:

$$
\{ p _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta } ( \cdot \mid \ell )\tag{7}
$$

where $\pi _ { \theta }$ denotes the prompt generator. The frozen textto-image model renders each prompt to produce an image $x _ { i } = G ( p _ { i } )$ , which is then evaluated using GDB-REWARD from Eq. (6). GRPO computes a group-relative advantage by standardizing rewards within the sampled group:

$$
A _ { i } = \frac { R _ { i } - \mathrm { m e a n } ( \{ R _ { j } \} ) } { \mathrm { s t d } ( \{ R _ { j } \} ) + \epsilon }\tag{8}
$$

This encourages prompts that outperform the group average while discouraging poorer candidates. The prompt generator is updated using the GRPO policy-gradient objective with a KL regularization term that constrains the policy to remain close to the frozen base language model. Throughout training, only the LoRA adapter of the prompt generator is updated. Both the image generator and GDB-REWARD remain frozen and operate entirely in inference mode.

## 4 Experiments

## 4.1 Implementation Details

The prompt policy is Qwen3.5-9B (Qwen Team 2026) finetuned with LoRA adapters $( r ~ = ~ 1 6 , \alpha ~ = ~ 3 2 )$ , optimizing 29.1M parameters while keeping the backbone frozen. We train with GRPO using frozen FLUX.1-dev (Labs 2024) and FLUX.2-dev (Labs 2025) generators and our proposed GDB-REWARD. Further implementation details are provided in Appendix B.

## 4.2 Datasets

Experiments are conducted on the LICA graphic design benchmark (Deganutti et al. 2026). Each example contains structured layout metadata, a corresponding reference image, and a concise textual description of the intended design objective (examples shown in Appendix C). We train on 911 layouts and evaluate on a held-out test split of 100 layouts.

## 4.3 GDB-REWARD: Quantitative Analysis

Table 2 evaluates whether GDB-REWARD functions as an effective optimization objective. We compare the frozen prompt policy with the same policy after reinforcement learning using GDB-REWARD, holding the image generator fixed and repeating the experiment across two generators of differing capability. Improvements therefore isolate the reward’s ability to guide optimization across diffusion models with a range of capabilities. It is worth noting that this evaluation is performed using the same metrics that comprise our training loss. The evaluations in later sections will provide independent verifications of the performance.

GDB-REWARD increases the composite score under both generators, confirming that its optimization signal is consistent across generators of different sizes. The gains are concentrated on the metrics that directly test specification adherence: OCR accuracy, color fidelity, and componentcount agreement. Two properties show these gains are principled rather than incidental. First, the reward targets each backbone’s failure mode: on FLUX.1, where text rendering is weak, OCR jumps 0.446 → 0.643, whereas on FLUX.2 OCR already sits at ceiling. Second, training pulls the two generators toward a common specification-satisfying operating point: the per-metric gap between them shrinks after training, and the trained FLUX.1 policy (0.559) recovers essentially the full quality of the untrained FLUX.2 baseline (0.560) through prompting alone. The perceptual metrics further validate this objective. Reference-free quality improves consistently, while reference-based similarity is mixed, with DreamSim increasing but LPIPS and SSIM decreasing slightly. This is expected because a design brief allows multiple valid realizations; optimizing for the specification preserves semantic fidelity while allowing some variation in typography, spacing, and placement.

![](images/8400232b5556e93a986140ee97af6a861b22ab8f41190dc05af5599adaef4c22.jpg)

![](images/de12d71538456e46cb13753b8ee5a83df1987d290ca08b134d2bab0615c9c276.jpg)  
Figure 3: Qualitative results. Comparison of designs generated by our method and the baseline, both using Qwen3.5-9B and FLUX.1-dev as backbones. The examples highlight improvements across the four evaluation dimensions: rendering fidelity, perceptual quality, and spatial accuracy.

## 4.4 GDB-REWARD: Qualitative Analysis

Figure 3 provides qualitative examples illustrating the behavior observed quantitatively. Rather than consistently producing more aesthetically pleasing images, the optimized prompt policy generates designs that more closely satisfy the intended design specification. Improvements are most evident in three areas. First, generated text is reproduced more accurately and legibly, consistent with the large increase in OCR accuracy. Second, color palettes more closely match the specified design colors, reflecting the improvement in color fidelity. Third, layouts contain a more appropriate number of visual components and better preserve the intended composition, matching the gains in the spatial evaluation metrics. These examples mirror the quantitative evaluation and suggest that reinforcement learning improves specification adherence. Additional qualitative results are given in Appendix E.

## 4.5 Ablations

Task groups. To evaluate the contribution of each reward category, we retrain the prompt policy while removing one of the three groups of reward terms: perceptual quality, rendering fidelity, or spatial accuracy. Table 3 reports the resulting composite scores. Removing any reward group degrades overall performance, indicating that all three contribute to the final policy. The largest reduction occurs when rendering fidelity is removed, suggesting that fine-grained content guidance provides the strongest optimization signal. Nevertheless, combining all three categories consistently achieves the highest overall reward, demonstrating that the different objectives are complementary.

<table><tr><td>Training reward</td><td>Composite Score</td></tr><tr><td>Full GDB-REWARD</td><td>0.530</td></tr><tr><td>Perceptual quality + Rendering fidelity</td><td>0.523</td></tr><tr><td>Rendering fidelity + Spatial accuracy</td><td>0.498</td></tr><tr><td>Spatial accuracy + Perceptual quality</td><td>0.491</td></tr></table>

Table 3: Ablation of the three GDB-REWARD task groups, trained on a 50-sample subset of the training dataset, and evaluated on the full evaluation split. Each row is the Qwen3.5-9B prompt generator trained with one group removed from the reward, paired with FLUX.1-dev.

Reward-reweighting. A separate design question is whether the relative weighting of metrics within GDB-REWARD affects performance. Our default weighting assigns each of the three task groups equal total importance, split evenly across its metrics. Since GRPO normalizes rewards within each sampled group, optimization is primarily driven by the relative ordering of generated samples rather than the absolute scale of individual reward terms. We stress-test this by retraining with a deliberately skewed composite that shifts roughly 90% of the reward mass onto the higher-variance, more discriminating rendering-fidelity (OCR, $\Delta E _ { 0 0 } )$ and spatial (YOLO component-count) metrics. Table 4 demonstrates that despite devoting far more training signal to rendering fidelity and spatial accuracy, the reweighted policy is no better on the very axes it emphasizes. This corroborates the principle that within-group standardization already rescales each metric by its group variance, making hand-tuned per-metric coefficients largely redundant and occasionally harmful, which justifies the simple equalper-concept default.

<table><tr><td rowspan="2"></td><td rowspan="2">Full</td><td rowspan="2"></td><td colspan="5">Perceptual quality</td><td colspan="2">Rendering fidelity</td><td colspan="2">Spatial accuracy</td></tr><tr><td>DS</td><td>LP</td><td>SS</td><td>IR</td><td>NM</td><td> $\Delta E _ { 0 0 }$ </td><td>OCR</td><td> $\mathrm { C } _ { g }$ </td><td> $\mathrm { C } _ { m }$ </td></tr><tr><td rowspan="2">Equal per concept (default)</td><td>Weight</td><td></td><td>0.050</td><td>0.050</td><td>0.050</td><td>0.050</td><td>0.050</td><td>0.125</td><td>0.125</td><td>0.125</td><td>0.125</td></tr><tr><td>Score</td><td>0.530</td><td>0.514</td><td>0.300</td><td>0.738</td><td>0.514</td><td>0.426</td><td>0.468</td><td>0.709</td><td>0.547</td><td>0.464</td></tr><tr><td rowspan="2">Rendering and spatial-heavy</td><td>Weight</td><td></td><td>0.030</td><td>0.020</td><td>0.005</td><td>0.040</td><td>0.005</td><td>0.160</td><td>0.300</td><td>0.220</td><td>0.220</td></tr><tr><td>Score</td><td>0.508</td><td>0.495</td><td>0.316</td><td>0.743</td><td>0.431</td><td>0.433</td><td>0.491</td><td>0.648</td><td>0.507</td><td>0.437</td></tr></table>

Table 4: Reward-reweighting ablation. Both rows train the Qwen3.5-9B + FLUX.1-dev prompt policy with GDB-REWARD under identical GRPO settings and differ only in the training reward weights. Equal per concept is our default (Table 1).

## 4.6 GDB-REWARD vs. Generator Fine-tuning

Our central design decision is to optimize the prompt policy while keeping the image generator frozen. We compare this approach against supervised fine-tuning (SFT) of the image generator itself. The SFT method attaches a LoRA adapter to the image generator and trains the denoising network using a standard flow-matching objective to reconstruct the reference render directly from the layout metadata. In contrast, our approach leaves the generator unchanged and optimizes only the language model that produces the prompt. Table 5 compares both approaches. SFT lands below the frozen baseline on both backbones. We attribute this to an objective-and-conditioning mismatch. SFT conditions the denoiser on serialized layout metadata which is far out of distribution for a generator whose text encoder expects natural-language prompts. This pulls the model off its strong pretrained prior toward exact reconstruction of one target, trading away the perceptual and aesthetic quality the composite score rewards.

<table><tr><td rowspan="2">Method</td><td colspan="2">Trainable Params.</td><td colspan="2">Composite Score</td></tr><tr><td>FL.1</td><td>FL.2</td><td>FL.1</td><td>FL.2</td></tr><tr><td>SFT</td><td>46.7M</td><td>83.4M</td><td>0.342</td><td>0.518</td></tr><tr><td>Baseline</td><td>-</td><td>-</td><td>0.398</td><td>0.560</td></tr><tr><td>GDB-REWARD</td><td>29.1M</td><td>29.1M</td><td>0.559</td><td>0.647</td></tr></table>

Table 5: Comparison of supervised generator fine-tuning (SFT) and GDB-REWARD prompt optimization. FL.1 and FL.2 denote FLUX.1-dev and FLUX.2-dev.

## 4.7 Design-Intent Evaluation

While GDB-REWARD evaluates low-level aspects of graphic design quality, it does not explicitly measure whether a generated design preserves the original communicative goal. We therefore perform a complementary evaluation based on design intent. Given a generated design, a vision–language model predicts a concise textual description of its intended deliverable, message, target audience, and overall tone. This predicted description is compared against the ground-truth design intent provided by the LICA dataset. Details of the evaluator and its training are provided in Appendix D. We evaluate design-intent preservation as a text generation task using both semantic and lexical similarity metrics. Our primary metric is G-Eval (Liu et al. 2023), implemented with DeepEval (Ip and Vongthongsri 2026) using GPT-5.5 (OpenAI 2026), which measures whether the predicted description captures the same underlying design objective as the reference. We additionally report BLEU-1 (Papineni et al. 2002), ME-TEOR (Banerjee and Lavie 2005), ROUGE-L (Lin 2004), and CIDEr (Vedantam, Lawrence Zitnick, and Parikh 2015), which quantify lexical agreement between the predicted and reference intents.

Table 6 shows that design-intent preservation is primarily limited by the image generator rather than the prompt policy: upgrading the frozen backbone from FLUX.1-dev to FLUX.2-dev improves G-Eval from 52.5 to 65.0. Across all metrics, GDB-REWARD predominantly preserves or slightly improves design intent, demonstrating that the reward improves measurable design fidelity without sacrificing the original communicative goal.

<table><tr><td>Prompt policy</td><td>B1↑</td><td>M↑</td><td>R-L↑</td><td>Cr ↑</td><td>G-Eval ↑</td></tr><tr><td>FLUX.1-dev</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>41.28</td><td>16.48</td><td>31.75</td><td>50.86</td><td>52.50</td></tr><tr><td>+GDB-REWARD</td><td>42.19</td><td>17.09</td><td>32.14</td><td>50.92</td><td>52.70</td></tr><tr><td>FLUX.2-dev</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>44.18</td><td>19.27</td><td>35.72</td><td>65.72</td><td>65.00</td></tr><tr><td>+GDB-REWARD</td><td>45.29</td><td>18.97</td><td>35.87</td><td>67.66</td><td>61.60</td></tr></table>

Table 6: Design-intent preservation of the generated reconstructions on the LICA test split, scored by the fine-tuned Qwen3.5-9B captioner against the ground-truth intent.

## 5 Conclusion

We presented GDB-REWARD, a framework that transforms heterogeneous graphic design evaluation metrics into a unified reinforcement learning reward for prompt optimization. By optimizing only a lightweight prompt policy while keeping the text-to-image generator frozen, GDB-REWARD improves adherence to graphic design specifications without diffusion model fine-tuning. More broadly, our results suggest that evaluation metrics can serve not only as benchmarks but also as effective optimization objectives for reinforcement learning in non-differentiable generation tasks.

## References

Banerjee, S.; and Lavie, A. 2005. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measuresfor Machine Translation and/or Summarization, 65–72.

Barto, A. G. 2021. Reinforcement learning: An introduction. by richard’s sutton. SIAM Rev, 6(2): 423.

Chen, W.; Hong, D.; Mao, Z.; Cheng, Y.; Liu, X.; Zhang, L.; and Zhang, Y. 2026. Creatiparser: Generative image parsing of raster graphic designs into editable layers. arXiv preprint arXiv:2604.19632.

Cheng, Y.; Zhang, Z.; Yang, M.; Nie, H.; Li, C.; Wu, X.; and Shao, J. 2024. Graphic design with large multimodal model. arXiv preprint arXiv:2404.14368.

Deganutti, A.; Hirsch, E.; Zhu, H.; Seol, J.; and Mehta, P. 2026. Graphic-Design-Bench: A comprehensive benchmark for evaluating AI on graphic design tasks. arXiv preprint arXiv:2604.04192.

Fu, S.; Tamir, N.; Sundaram, S.; Chai, L.; Zhang, R.; Dekel, T.; and Isola, P. 2023. DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data. In Advances in Neural Information Processing Systems, volume 36, 50742–50768.

Hao, Y.; Chi, Z.; Dong, L.; and Wei, F. 2023. Optimizing prompts for text-to-image generation. Advances in Neural Information Processing Systems, 36: 66923–66939.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; Chen, W.; et al. 2022. Lora: Low-rank adaptation of large language models. International Conference on Learning Representations, 1(2): 3.

Ip, J.; and Vongthongsri, K. 2026. DeepEval: The Open-Source LLM Evaluation Framework. https://github.com/ confident-ai/deepeval. Accessed: 2026-07-09.

Jia, P.; Li, C.; Yuan, Y.; Liu, Z.; Shen, Y.; Chen, B.; Chen, X.; Zheng, Y.; Chen, D.; Li, J.; Xie, X.; Zhang, S.; and Guo, B. 2024. COLE: A Hierarchical Generation Framework for Multi-Layered and Editable Graphic Design. arXiv:2311.16974.

Kirstain, Y.; Polyak, A.; Singer, U.; Matiana, S.; Penna, J.; and Levy, O. 2023. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36: 36652–36663.

Kong, W.; Jiang, Z.; Sun, S.; Guo, Z.; Cui, W.; Liu, T.; Lou, J.; and Zhang, D. 2022. Aesthetics++: Refining graphic designs by exploring design principles and human preference. IEEE Transactions on Visualization and Computer Graphics, 29(6): 3093–3104.

Labs, B. F. 2024. FLUX. https://github.com/black-forestlabs/flux.

Labs, B. F. 2025. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2.

Lai, J.; Chen, S.; Gao, J.; Shi, H.; Liu, Z.; Zhai, F.; Luo, J.; Wei, X.; Wang, L.; and Zhu, L. 2026. PosterReward: Unlocking Accurate Evaluation for High-Quality Graphic Design Generation. arXiv:2603.29855.

Lee, S.; and Ye, J. C. 2025. Plug-and-Play Prompt Refinement via Latent Feedback for Diffusion Model Alignment. arXiv preprint arXiv:2510.00430.

Li, S.; Petrangeli, S.; Shen, Y.; and Chen, X. 2026. From Pixels to Policies: Reinforcing Spatial Reasoning in Language Models for Content-Aware Layout Design. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026), 1509–1518.

Lin, C.-Y. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, 74–81.

Liu, Y.; Iter, D.; Xu, Y.; Wang, S.; Xu, R.; and Zhu, C. 2023. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. arXiv:2303.16634.

OpenAI. 2026. Introducing GPT-5.5. https://openai.com/ index/introducing-gpt-5-5/.

Papineni, K.; Roukos, S.; Ward, T.; and Zhu, W.-J. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Associationfor Computational Linguistics, 311–318.

Qu, Y.; Fang, S.; Wang, Y.; Wang, X.; Chen, Z.; Xie, H.; and Zhang, Y. 2025. Igd: Instructional graphic design with multimodal layer generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 18218–18228.

Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Sharma, G.; Wu, W.; and Dalal, E. N. 2005. The CIEDE2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations. Color Research & Application: Endorsed by Inter-Society Color Council, The Colour Group (Great Britain), Canadian Society for Color, Color Science Association of Japan, Dutch Society for the Study of Color, The Swedish Colour Centre Foundation, Colour Society of Australia, Centre Franc¸ais de la Couleur, 30(1): 21–30.

Talebi, H.; and Milanfar, P. 2018. NIMA: Neural Image Assessment. IEEE Transactions on Image Processing, 27(8): 3998–4011.

Vedantam, R.; Lawrence Zitnick, C.; and Parikh, D. 2015. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 4566–4575.

Wang, Z.; Bovik, A.; Sheikh, H.; and Simoncelli, E. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612.

Wu, X.; Hao, Y.; Sun, K.; Chen, Y.; Zhu, F.; Zhao, R.; and Li, H. 2023. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341.

Xu, J.; Liu, X.; Wu, Y.; Tong, Y.; Li, Q.; Ding, M.; Tang, J.; and Dong, Y. 2023. ImageReward: learning and evaluating human preferences for text-to-image generation. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, 15903–15935.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

## A Additional Methodology Details

Prompt Policy The prompt generator used as our policy is an instruction-tuned LLM equipped with a LoRA adapter. Given the layout metadata, the model is prompted via the fixed system instruction shown in Figure 4.

![](images/3e4e5126d7dfe2efed4b0e115c9047bb594c16d21d37eea6184e44d960ddf81d.jpg)  
Figure 4: System instruction used to condition the prompt policy. The instruction guides the LLM to convert structured graphic design metadata into a single natural-language prompt for the frozen text-to-image generator.

## B Experiments Implementation Details

This section provides the implementation details required to reproduce our experiments, including the prompt policy, frozen image generator, GDB-REWARD configuration, and reinforcement learning hyperparameters.

Prompt policy. The prompt policy is Qwen3.5-9B operating in bfloat16, fine-tuned with LoRA adapters (r = 16, $\alpha = 3 2 )$ inserted into the attention and MLP projection layers while keeping the backbone frozen. In total, 29.1M parameters (∼ 0.2% of the model) are optimized. During training, prompts are generated from the layout metadata using nucleus sampling (temperature 1.0, top- $\cdot p \ = \ 0 . 9 5 )$ with a maximum length of 320 tokens.

Image generator. We report results on both FLUX.1-dev and FLUX.2-dev as the frozen image generator. Images are generated using 20 diffusion steps with guidance scale 3.5 and sequence length 512, producing 768 × 768 images during RL training and 1024 × 1024 images for evaluation.

GDB-REWARD configuration. Our proposed GDB-REWARD uses six dominant palette colors $( K \ : = \ : 6 )$ and color sharpness parameter $\tau = 2 0$ . For layout analysis, we use a fine-tuned YOLO11x-OBB detector taken from the GraphicDesignBench (Deganutti et al. 2026) repository<sup>1</sup>. All reward models operate in inference mode only.

RL hyperparameters. We optimize the prompt policy using GRPO for 1368 optimization steps (three epochs) when using FLUX.1-dev as the backbone, and 912 optimization steps (two epochs) when using FLUX.2-dev. We set the GRPO group size to 12 sampled prompts across two layouts per update. The learning rate is $3 \times 1 0 ^ { - 5 }$ , the KL coefficient is $\beta \ : = \ : 0 . 0 2$ , gradients are clipped to 1.0, and the advantage standard deviation is lower bounded by $1 0 ^ { - 4 }$ . Following prior work, common random numbers (shared diffusion seeds) are used within each group to reduce sampling variance. All experiments use a fixed random seed of 0.

Computing infrastructure. All experiments were conducted on a compute node equipped with eight NVIDIA H100 SXM5 GPUs (80 GB memory each), 168 CPU cores, 907 GB system memory, and 12.2 TB of local storage, running Ubuntu 24.04 LTS. The implementation was built using PyTorch, Hugging Face Transformers, TRL for GRPO optimization, PEFT for LoRA fine-tuning, and Diffusers for image generation. The experiments were performed with CUDA 12.0.

## C Dataset Examples

We provide a full example from the LICA GraphicDesign-Bench dataset (Deganutti et al. 2026). Figure 5 illustrates the dataset structure, showing the design intent description, rendered image, and corresponding layout metadata for each example.

## D Design-Intent Evaluation Model

To evaluate whether generated designs preserve their intended communicative goal, we train a dedicated vision– language model that predicts a textual description of the design intent from a rendered graphic design. Each description summarizes the intended deliverable, message, target audience, and desired emotional tone. The model is used exclusively for evaluation and is never incorporated into the reinforcement-learning reward.

<table><tr><td>Model</td><td>B1↑</td><td>M↑</td><td>R-L↑</td><td>Cr ↑</td><td>G-Eval ↑ (gpt-5.5)</td></tr><tr><td colspan="6">Base model</td></tr><tr><td>Qwen-3.5</td><td>19.60</td><td>13.57</td><td>17.39</td><td>4.69</td><td>55.80</td></tr><tr><td>Llama-4</td><td>17.96</td><td>14.89</td><td>19.35</td><td>3.98</td><td>53.00</td></tr><tr><td>DeepSeek</td><td>6.91</td><td>10.97</td><td>09.61</td><td>0.00</td><td>19.30</td></tr><tr><td colspan="6">Fine-tuned model</td></tr><tr><td>Qwen-3.5</td><td>48.84</td><td>21.90</td><td>39.97</td><td>87.25</td><td>72.90</td></tr><tr><td>Llama-4</td><td>49.34</td><td>21.99</td><td>39.42</td><td>85.73</td><td>72.60</td></tr><tr><td>DeepSeek</td><td>47.92</td><td>20.74</td><td>37.13</td><td>73.69</td><td>67.50</td></tr></table>

Table 7: We finetune four models: Qwen3.5-9B, Llama-4- Scout-17B-16E-Instruct, and DeepSeek-VL-7B-Chat on our design intent captioning task and report both semantic and exact match metrics on the test split.

![](images/e1af881eca064998bd9d1b05c46ef1cf161bf97154eae21cca5fd00b07dd19c4.jpg)  
Figure 5: Example from the LICA GraphicDesignBench dataset. Each sample consists of (a) serialized layout metadata describing the design elements and their attributes, (b) the corresponding rendered graphic design, and (c) a natural-language design intent used to guide generation. Metadata has been truncated for readability.

We fine-tune pretrained vision–language models using LoRA on paired design renders and intent descriptions from the LICA training split. Training follows a standard autoregressive objective, where the model learns to generate the corresponding design-intent description given the rendered design image. During evaluation, the predicted intent $\hat { y } ~ = ~ f _ { \phi } ( \bar { x } )$ is compared with the ground-truth intent y<sub>ℓ</sub>, where x is the generated image.

To select the evaluation model, we first compare several candidate vision–language backbones on the held-out LICA test split. Table 7 reports both the performance of the original pretrained models and their fine-tuned counterparts. We evaluate using G-Eval (Liu et al. 2023) as the primary semantic metric. We provide the prompt and evaluation rubric used by G-Eval to assess the design-intent in Figure 6 and Table 8. Furthermore, we evaluate the VLM using BLEU-1 (Papineni et al. 2002), METEOR (Banerjee and Lavie 2005), ROUGE-L (Lin 2004), and CIDEr (Vedantam, Lawrence Zitnick, and Parikh 2015). Fine-tuning substantially improves performance for all candidate models. Although Qwen3.5 and Llama-4 obtain comparable lexical scores, the fine-tuned Qwen3.5 model achieves the highest G-Eval score and is therefore selected as the design-intent evaluator used throughout our experiments.

Note the FORM of ’expected output’: it is a single, concise design-intent statement (typically one sentence, roughly 20-40 words) that names the deliverable/artifact, its message or purpose, the target audience, and the intended feeling or tone. Identify those core elements (deliverable, message/purpose, audience, feeling/tone) in ’expected output’ and check whether ’actual output’ states the same core goal. Reward semantic agreement even when the wording differs. Judge concision and form: ’actual output’ should be about as brief as ’expected output’ and read as a single-paragraph intent statement. HEAVILY penalize output that is markedly longer or more verbose than ’expected output’ – multiple sentences or paragraphs, exhaustive visual description, enumerating colours/objects/positions, step-by-step or list formatting, or reading like an image caption rather than a concise statement of intent – EVEN IF the extra content is accurate. Heavily penalize contradictions (wrong artifact type, wrong audience, opposite mood) and omission of a core element present in ’expected output’. Do NOT reward extra detail beyond the expected scope; faithfulness to the expected output’s brevity and scope is part of correctness. Minor paraphrasing and stylistic differences are acceptable.

Figure 6: Evaluation instructions provided to the LLM judge for G-Eval. The prompt guides the model to assess semantic agreement between the predicted and reference design intents while penalizing verbosity and mismatched intent.

![](images/ebb1d39b713a67cf93cd7f6d09c5c07d259bc4920235149b8c4079528902dcb5.jpg)  
Figure 7: Additional qualitative results. We compare the ground-truth reference design, the baseline prompt policy with FLUX.1-dev, our prompt policy optimized with GDB-REWARD using the same generator, the baseline prompt policy with FLUX.2-dev, and our optimized prompt policy with FLUX.2-dev.

<table><tr><td>Score</td><td>Expected Outcome</td></tr><tr><td>0-2</td><td>Wrong/contradictory intent, or a verbose descrip- tion/caption that does not read as a concise intent statement.</td></tr><tr><td>3-5</td><td>Captures the goal but is markedly more verbose/long than the expected output, or misses a core element.</td></tr><tr><td>6-8</td><td>Matches the intent and is reasonably concise; only minor verbosity or a minor missing element.</td></tr><tr><td>9-10</td><td>Fully captures the design goal in a single concise statement matching the expected output&#x27;s brevity and scope.</td></tr></table>

Table 8: Scoring rubric used by the G-Eval judge to assess design-intent predictions. Higher scores indicate closer semantic agreement with the reference intent while maintaining comparable brevity.

## E Additional Qualitative Examples

Figure 7 presents additional qualitative comparisons across a diverse range of graphic design tasks, including event posters, planners, promotional flyers, awareness campaigns, and advertisements. We compare the ground-truth reference render against outputs from the baseline prompt policy with FLUX.1-dev, our prompt policy optimized with GDB-

REWARD using the same generator, the baseline prompt policy with FLUX.2-dev, and our optimized prompt policy with FLUX.2-dev.

A remaining limitation of our approach is that the generator does not receive the original photographic assets when they are part of the design. Instead, the prompt policy only provides textual descriptions of these elements, which can lead to larger deviations from the reference render for imagecentric designs. This is particularly evident in examples such as the third column, where the reference contains portraits of individual DJs that cannot be reproduced exactly from text alone.

Despite these limitations, the optimized prompt policy improves alignment with the intended design structure and aesthetics. While the frozen image generator remains a bottleneck for exact reproduction of complex layouts, typography, and embedded imagery, our method effectively steers generation toward outputs that better satisfy the visual constraints captured by the design specification.

## F Additional Quantitative Results

To test whether our GRPO prompt-optimization framework generalizes beyond the FLUX backbone, we repeat the full pipeline with Stable Diffusion XL (stable-diffusion-xl-base-1.0) as the frozen image generator. The setup is deliberately held identical to the FLUX runs to keep the comparison controlled: the same policy LLM (Qwen3.5-9B with an r=16 LoRA adapter), the same GraphicDesignBenchderived composite reward and its per-concept weighting, and the same GRPO hyperparameters (group size 12, two layouts per step, learning rate $3 \times 1 0 ^ { - 5 }$ , KL coefficient 0.02), data split, and random seed. Only the image generator is swapped, and its inference settings are adjusted to SDXL-appropriate values rather than copied from FLUX: a classifier-free guidance scale of 5.0 (versus 3.5 for FLUX, which visibly washes SDXL out) and 30 denoising steps (versus 20), with the generator run at 768×768 to match the FLUX training resolution. We report these results in the supplementary material rather than the main paper for two reasons. First, SDXL’s generations are substantially weaker than FLUX’s across our metrics, and since FLUX is the current state of the art for text-to-image generation we prioritize it in the main paper. Second, and more fundamentally, SDXL encodes prompts with two CLIP text encoders capped at 77 tokens, whereas FLUX uses a T5 encoder that admits far longer inputs; SDXL therefore cannot ingest the long, richly descriptive prompts our policy learns to produce, meaning it is structurally unable to take full advantage of the prompt-optimization system our method is built around. We include SDXL here only to demonstrate that the method is backbone-agnostic in principle, while noting that a backbone with a short text-encoder context is a poor fit for prompt-level optimization. We provide qualitative results of the SDXL backbone in Figure 8.

<table><tr><td></td><td></td><td colspan="5">Perceptual quality</td><td colspan="2">Rendering fidelity</td><td colspan="2">Spatial accuracy</td></tr><tr><td>Model</td><td>Full</td><td>DS</td><td>LP</td><td>SS</td><td>IR</td><td>NM</td><td> $\Delta E _ { 0 0 }$ </td><td>OCR</td><td> $\mathrm { C } _ { g }$ </td><td> $\mathbf { C } _ { m }$ </td></tr><tr><td>GDB-REWARD (FLUX.2)</td><td>0.647</td><td>0.565</td><td>0.296</td><td>0.735</td><td>0.571</td><td>0.415</td><td>0.578</td><td>0.869</td><td>0.731</td><td>0.675</td></tr><tr><td>GDB-REWARD (FLUX.1)</td><td>0.559</td><td>0.513</td><td>0.272</td><td>0.730</td><td>0.545</td><td>0.427</td><td>0.504</td><td>0.643</td><td>0.601</td><td>0.612</td></tr><tr><td>Base (Qwen3.5-9B + SDXL)</td><td>0.334</td><td>0.307</td><td>0.235</td><td>0.691</td><td>0.322</td><td>0.493</td><td>0.526</td><td>0.139</td><td>0.281</td><td>0.232</td></tr><tr><td>+GDB-REWARD</td><td>0.446</td><td>0.359</td><td>0.257</td><td>0.717</td><td>0.436</td><td>0.487</td><td>0.537</td><td>0.255</td><td>0.513</td><td>0.464</td></tr></table>

Table 9: Quantitative comparison across image-generation backbones on the held-out test split. We evaluate the composite GDB-REWARD score (Full) and its constituent metrics. The top block reports our method with the FLUX.2-dev and FLUX.1- dev generators; the bottom block reports the SDXL backbone before (Base) and after (+GDB-REWARD) prompt optimization, holding the policy LLM, reward, and all GRPO hyperparameters fixed. SDXL underperforms the FLUX backbones and, owing to its 77-token CLIP text encoder, cannot ingest the long descriptive prompts our policy learns to produce.

![](images/819ca29f716665388ac02ed58e1d5835d01986dcd0f26818e737ae690dee7c5c.jpg)  
Figure 8: Qualitative results on SDXL backbone. We compare the ground-truth reference design, the baseline prompt policy with SDXL, and our prompt policy optimized with GDB-REWARD using the same generator.