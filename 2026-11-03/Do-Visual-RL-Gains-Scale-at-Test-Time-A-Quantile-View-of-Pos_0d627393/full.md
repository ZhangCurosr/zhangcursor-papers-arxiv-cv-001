# Do Visual RL Gains Scale at Test Time? A Quantile View of Post-Training

Austin Wang1, Lexing Ying1

1ByteDance Seed

## Abstract

Reinforcement learning (RL) post-training improves the single-sample quality of visual generative models, but deployment often provides additional inference compute through Best-of-N sampling. Whether RL gains transfer under such test-time scaling, particularly on difficult held-out tasks, remains poorly understood in the visual domain. Our central contribution is to systematically identify and characterize a failure mode of scaling transfer in visual RL: across multiple RL methods, model backbones, and evaluations spanning compositional generation, text rendering, and preference alignment, higher single-sample reward does not necessarily imply a better qualitycompute frontier. RL gains may attenuate, saturate, or even reverse at large sampling budgets, with the base model eventually outperforming its post-trained counterpart. We explain these regimes through a quantile-transfer framework, showing that Best-of-N places increasing weight on the upper tail of the reward distribution as N grows. Consequently, RL can improve typical samples while compressing or degrading the rare high-quality outcomes that test-time scaling exploits. As a simple intervention motivated by this analysis, we introduce Quantile Utility Policy Optimization (QUPO), which optimizes a controllable mixture of quantile utilities corresponding to different inference budgets. Experiments on SD3.5-M and BAGEL show that QUPO better preserves upper-tail quality and substantially improves scaling on challenging held-out prompts relative to standard RL post-training. Our results suggest that visual generative models should be evaluated and optimized not only for single-sample quality, but for how post-training reshapes the full reward distribution that inference-time scaling exploits.

Correspondence: Austin Wang at austinwang@bytedance.com Code and Model Checkpoints: https://github.com/austin-k-wang/QUP0

## 1 Introduction

Reinforcement learning (RL) post-training has become an effective way to improve visual generative models, including diffusion and flow-based generators [1, 13, 16, 19], and unified multimodal models such as BAGEL [8]. Recent methods improve compositional accuracy, text rendering, and preference alignment [20, 42, 49], but are typically optimized and evaluated primarily through single-sample reward. This leaves open how visual post-training affects a model's ability to benefit from additional inference-time compute.

At deployment, additional compute can be converted into quality through Best-of-N sampling: generate N candidates and select the highest-reward output. We ask: when RL improves Best-of-1, do those gains persist as N increases? We call this property scaling transfer, the extent to which post-training gains are preserved along the Best-of-N quality-compute frontier. This question is especially important on difficult held-out

prompts, where additional sampling is most useful.

We find that scaling transfer is far from guaranteed. Across multiple visual RL methods and tasks, Best-of-1 gains can persist, attenuate, or even reverse as N grows. On held-out compositional benchmarks such as GenEval2 [15] and ConceptMix [41], several post-trained models outperform the base model at small N but are eventually overtaken at larger budgets. Text rendering instead exhibits strongly attenuated but positive transfer, while preference-alignment gains are substantially more persistent. These qualitatively different regimes are obscured by single-sample evaluation.

We explain these behaviors through a quantile view of the per-prompt reward distribution. Let $Q _ { \pi } ( u \mid c )$ denote the reward quantile function of model π on prompt c. Our analysis shows that Best-of-1 weights reward quantiles uniformly, whereas increasing N places progressively greater emphasis on the upper tail. Post-training can therefore improve typical generations while leaving rare high-reward outcomes unchanged or degraded, causing its advantage to shrink or reverse at larger sampling budgets. Empirically, the observed quantile shifts closely match the corresponding scaling-transfer regimes.

Motivated by this analysis, we introduce Quantile Utility Policy Optimization (QUPO) as a simple intervention for the scaling-transfer failure identified in visual RL. QUPO is a drop-in objective that optimizes a controllable mixture of Best-of-k utilities, explicitly emphasizing the reward quantiles that additional inference samples exploit. Related language model work has explored Pass@k and other tail-aware objectives [27, 30, 33]; our main contribution centers on systematically identifying and characterizing scaling-transfer failure in visual generation, with QUPO demonstrating that a simple change to the training utility can mitigate it. Across SD3.5-M and BAGEL, QUPO better preserves upper-tail reward quality and substantially improves Best-of-N scaling relative to standard RL post-training

We summarize our contributions as follows:

• We identify and characterize scaling transfer in visual RL, showing that post-training gains can persist, attenuate, or reverse as test-time sampling compute increases.

• We develop a quantile-transfer framework showing that larger Best-of-N budgets increasingly emphasize the upper reward tail, and empirically connect upper-tail changes to observed scaling behavior.

• We introduce QUPO, a simple quantile-aware post-training objective that improves scaling transfer on both SD3.5-M and BAGEL.

## 2 Related Work

## 2.1 Reward-Based Post-Training of Visual Generators

Diffusion and flow-based generators have been post-trained through direct reward backpropagation [7, 25], reinforcement learning |4, 6, 10, 20, 36, 42, 43, 49], and preference optimization |12, 22, 23, 34, 37]. These methods primarily target single-sample alignment through expected-reward or preference objectives and are typically evaluated at fixed inference budgets. Consequently, how post-training reshapes generation quality across larger sampling budgets remains poorly understood.

Reward-based post-training has also been extended to unified multimodal models (UMMs), which jointly support language, visual understanding, and image generation. Recently, UniGRPO [21| applies GRPO-style optimization to models such as BAGEL [8]. We study both flow-based generators and UMMs to test whether scaling-transfer behavior persists across distinct generative architectures.

## 2.2 Test-Time Scaling of Visual Generators

Best-of-N (BoN) inference improves generation quality by sampling multiple candidates and selecting the highest-scoring output, providing a simple characterization of the quality-compute frontier. More sophisticated approaches use reward-gradient guidance, search and lookahead, or posterior and reward-tilted sampling [3, 14, 29, 32, 35, 44, 45, 47, 48]. Whereas these methods study how to better use inference compute for a fixed model, we study how post-training changes the model's inherent ability to benefit from that compute. We therefore use Best-of-N as a model-agnostic diagnostic of the generator's native reward distribution, avoiding confounding from a particular search or guidance procedure.

## 2.3 Test-Time-Aware Policy Optimization

Recent language model works study how post-training affects multi-sample inference: PKPO [33] optimizes Pass@k, MaxRL [30] introduces compute-indexed objectives, MaxPO [31] improves Max@k estimators, and TEA [18] extrapolates upper-tail reward statistics to deployment budgets exceeding the rollout budget. [5] analyzes how RL reshapes LLM reasoning, linking changes in strategy selection to cross-domain transfer, reasoning coverage, and scaling laws in RL compute. Concurrently, TailRL [27] develops a tail-sensitive objective expressible as a harmonic mixture of Best-of-k utilities. QUPO subsumes this objective as a special case while allowing controllable weighting across inference budgets. In visual generation, Multi-Axis Max@K [24] targets representative diversity through a set-level objective that maximizes scores across samples separately for predefined semantic modes, such as color and lighting

Our primary contribution is the systematic characterization of scaling transfer in visual RL: when Best-of-1 gains persist or fail as inference budgets grow. We connect these failures to reward-quantile shifts and derive QUPO from this quantile view of the scalar Best-of-N quality-compute frontier, demonstrating a simple remedy across visual generation architectures.

## 3 Background

## 3.1 Reward Distributions and Best-of-N

Let $\pi ( \pmb { x } \mid c )$ denote a visual generative model conditioned on prompt c, with scalar reward $r ( { \pmb x } , c )$ . Sampling $\mathbf { x } \sim \pi ( \cdot | c )$ induces a reward random variable $\mathbf { r } = r ( \mathbf { x } , c )$ with CDF

$$
F _ { \pi } ( t \mid c ) = \mathbb { P } ( \mathrm { r } \leq t ) ,
$$

and quantile function

$$
Q _ { \pi } ( u \mid c ) = \operatorname* { i n f } \{ t : F _ { \pi } ( t \mid c ) \geq u \} , \qquad u \in ( 0 , 1 ) .\tag{1}
$$

For N independent generations $\mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { N } \sim \pi ( \cdot \mid c )$ , define the expected Best-of-N reward

$$
B _ { N } ^ { \pi } = \mathbb { E } _ { c } \mathbb { E } _ { \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { N } \sim \pi ( \cdot | c ) } \left[ \operatorname* { m a x } _ { i } r ( \mathbf { x } _ { i } , c ) \right] .\tag{2}
$$

For $N = 1$

$$
B _ { 1 } ^ { \pi } = \mathbb { E } _ { c } \mathbb { E } _ { \mathbf { x } \sim \pi ( \cdot | c ) } [ r ( \mathbf { x } , c ) ] ,\tag{3}
$$

so expected single-sample reward is the first point on the Best-of-N quality-compute frontier.

Given a reference model $\pi _ { \mathrm { r e f } }$ and post-trained model $\pi _ { \mathrm { R L } }$ , we define

$$
\begin{array} { r } { A _ { N } = B _ { N } ^ { \pi _ { \mathrm { R L } } } - B _ { N } ^ { \pi _ { \mathrm { r e f } } } . } \end{array}\tag{4}
$$

A decreasing $A _ { N }$ indicates attenuating scaling transfer, while $A _ { N } < 0$ indicates frontier reversal. In Sec. 5, we relate $A _ { N }$ to changes in the reward quantiles.

## 3.2 Reward-Maximizing Visual Post-Training

Many visual RL methods optimize learning signals derived from rewards assigned to individual generations. The canonical reward objective is

$$
\begin{array} { r } { \mathcal { I } _ { \mathrm { r e w a r d } } ( \theta ) = \mathbb { E } _ { c } \mathbb { E } _ { \mathbf { x } \sim \pi _ { \theta } ( \cdot \vert c ) } [ r ( \mathbf { x } , c ) ] = B _ { 1 } ^ { \pi _ { \theta } } . } \end{array}\tag{5}
$$

Several methods sample groups of G generations and construct normalized advantages such as

$$
\widehat { A } _ { i } = \frac { r _ { i } - \bar { r } } { s _ { r } + \epsilon } , \qquad r _ { i } = r ( { \bf x } _ { i } , c ) ,\tag{6}
$$

where r and sr are the group mean and standard deviation. FlowGRPO [20], GRPO-Guard [38], AWM [42], and UniGRPO [21] use this advantage in GRPO-style [28] optimization, while DiffusionNFT [49] uses a different update rule but similarly derives its signal from per-generation rewards.

Crucially, using multiple rollouts during training does not itself optimize a Best-of-G utility: the group provides normalization or credit assignment, while utility remains defined per sample. These methods are therefore primarily aligned with single-sample reward rather than explicitly optimizing the multi-sample utility realized at deployment.

## 4 Diagnosing Scaling Transfer in Visual RL

We study whether gains from visual RL post-training persist as test-time sampling compute increases on difficult held-out prompts. Across compositional generation, text rendering, and preference alignment, we compare each post-trained SD3.5-M model against its matched base model using up to N = 128 samples per prompt.

## 4.1 Experimental Protocol

We evaluate FlowGRPO [20], DiffusionNFT [49], GRPO-Guard [38], and AWM [42], where applicable. Models are either reproduced using the official implementation or obtained from officially released LoRA adapters (Table 3). Each model is post-trained on a source task and evaluated on harder held-out prompts from the same capability domain.

For each prompt c, we sample G = 128 generations and estimate Best-of-N performance by averaging the maximum reward over all size-N subsets:

$$
\widehat { B } _ { N } ( c ) = \frac { 1 } { { \binom { G } { N } } } \sum _ { S \subseteq \{ 1 , \ldots , G \} } \operatorname* { m a x } _ { i \in S } r _ { i } .\tag{7}
$$

This U-statistic is unbiased for expected Best-of-N reward and uses all available samples. We use oracle Best-of-N, selecting candidates according to the evaluation reward itself. Importantly, all evaluation rewards are reference-free and can also be used for practical inference-time reranking without ground-truth image annotations. Thus, our protocol characterizes both the intrinsic reward distribution and the attainable Best-of-N frontier under the corresponding reward-based selector, excluding scoring overhead. We average $\widehat { B } _ { N } ( c )$ across prompts and compare each post-trained model against its matched base model. Extended experimental details are provided in Section C.1.

## 4.2 Compositional Generation

We evaluate models post-trained on GenEval [11] on the harder held-out GenEval2 benchmark [15]. Diffusion-NFT and AWM use a CFG scale of 1.0, while FlowGRPO and GRPO-Guard use CFG 4.5; we report the base model under both settings.

Figure 1a shows a clear frontier reversal. All post-trained models improve Best-of-1 and small-N performance, but their advantage steadily decreases and eventually becomes negative at large N. Thus, successful compositional post-training does not necessarily preserve the high-reward outcomes exploited by larger sampling budgets.

## 4.3 Text Rendering

We evaluate models trained on the FlowGRPO OCR task on harder held-out TextInVision prompts [9]. Unlike compositional generation, post-trained models remain uniformly stronger, but their advantage decreases substantially as they approach the reward ceiling and the base model catches up under additional sampling. We refer to this regime as positive but decaying scaling transfer.

![](images/77d5b1cf9d1de8d93886cd74d5abc24ef84c72119365dde3bd2f36ecaef2e077.jpg)  
(a) GenEval2.

![](images/bde429089f816b3253834064872eced4060368fb63e8f7fe108791b8adbc82f0.jpg)

(b) TextInVision.  
![](images/171d18974e92e8f175fd4a08ae23d86482dda089d7832b55dbab5e56d2b6bfac.jpg)  
(c) Parti-Prompts.  
Figure 1 Best-of-N scaling on difficult held-out prompts. (a) Compositional generation exhibits frontier reversal. (b) Text-rendering gains remain positive but decay toward saturation. (c) Preference-alignment gains remain approximately persistent as N increases.

## 4.4 Preference Alignment

We train on Pick-a-Pic prompts with PickScore [17] and evaluate on held-out Parti Prompts using HPSv2 [40], testing transfer across both prompts and reward models. Here, post-trained models remain uniformly stronger and maintain a similar advantage as N increases, exhibiting approximately persistent scaling transfer.

## 4.5 Scaling-Transfer Regimes

Across the three domains, scaling transfer ranges from frontier reversal to positive but decaying and approximately persistent transfer. These results indicate that weakened Best-of-N scaling is neither universal nor explained by single-sample performance alone. Instead, the shape of the entire reward distribution must determine how post-training gains evolve with inference compute. In the next section, we develop a quantile-based analysis that makes this connection explicit.

## 5 Understanding Scaling Transfer Through Reward Quantiles

## 5.1 Best-of-N as Upper-Tail Quantile Transfer

To explain the scaling-transfer regimes above, we study how post-training reshapes the per-prompt reward distribution. Define the average quantile lift

$$
\Delta Q ( u ) = \mathbb { E } _ { c } [ Q _ { \pi _ { \mathrm { R L } } } ( u \mid c ) - Q _ { \pi _ { \mathrm { r e f } } } ( u \mid c ) ] , \qquad u \in ( 0 , 1 ) .\tag{8}
$$

Theorem 1 (Best-of-N as quantile transfer). Assuming integrable rewards, for every $N \geq 1$ , the post-training

![](images/893f283c67606c0449c1ec9e635364c0a890b020b4c059426c55e7b47bb92d2a.jpg)  
(a) GenEval2.

![](images/f65b9803f051d9e815ffee5264ba9b2fb8de3851e23cc0af7a93bd9bd4d453d6.jpg)  
(b) TextInVision.

![](images/747bade8ef67355d4775ae0479ee99b86cf7e2a53bdb67c4e4ae174311fcc842.jpg)  
(c) Parti-Prompts.  
Figure 2 Quantile transfer across held-out tasks. GenEval2 exhibits negative upper-tail lift and frontier reversal; TextInVision exhibits positive but strongly decaying upper-tail lift; and Parti-Prompts with HPSv2 exhibits largely persistent positive lift.

advantage at inference budget N $( E q . 4 )$ can be expressed as

$$
A _ { N } = \int _ { 0 } ^ { 1 } \Delta Q ( u ) N u ^ { N - 1 } d u = \mathbb { E } _ { \mathbf { u } _ { N } \sim \mathrm { B e t a } ( N , 1 ) } [ \Delta Q ( \mathrm { u } _ { N } ) ] .\tag{9}
$$

The proof is deferred to Appendix A.1. For $N = 1$ , reward quantiles are weighted uniformly; as N increases, $\mathrm { B e t a } ( N , 1 )$ concentrates near $u = 1$ , placing increasing emphasis on the upper tail. Thus, post-training can improve average reward while degrading large-N performance if its gains do not extend to the highest reward quantiles.

We estimate $Q _ { \pi } ( u \mid c )$ from the order statistics of the $G = 1 2 8$ rewards per prompt, compute paired posttrained-minus-base quantile differences, and average across prompts; 95% confidence intervals are obtained by paired prompt bootstrap. Figure 2 closely matches the regimes predicted by Theorem 1. On GenEval2, lift becomes negative in the upper tail, explaining the eventual frontier reversal. On TextInVision, lift remains positive but contracts sharply near $u = 1$ , matching positive but decaying scaling transfer. On Parti-Prompts, upper-tail lift remains positive and largely stable, consistent with approximately persistent transfer. Thus, the shape of $\Delta Q ( u )$ directly explains how Best-of-1 gains evolve with inference compute.

## 5.2 Prompt Difficulty and Scaling Transfer

For complex prompts, strong solutions may occupy only the extreme upper reward tail. Improving typical generations can therefore raise average reward without preserving the rare outcomes that additional sampling exploits.

Figure 3 stratifies GenEval2 by atomicity. On simpler prompts (atomicity 3–4), all four methods substantially improve Best-of-1, but these gains largely vanish as N grows. On the most complex prompts (atomicity 9–10), Best-of-1 changes little, yet all four methods underperform their matched baselines at $N = 1 2 8$ . As Theorem 1 shows, larger inference budgets emphasize upper reward quantiles, exposing losses obscured at Best-of-1. Together with the quantile-lift results, this suggests that post-training can improve typical outcomes while weakening rare solutions needed for difficult compositions.

## 6 Mitigating Scaling-Transfer Failure

## 6.1 Controlling Quantile Utility

Our analysis suggests that improving scaling transfer requires controlling which reward quantiles are emphasized during training. Consider the quantile-weighted objective

$$
\mathcal { I } _ { \phi } ( \theta ) = \mathbb { E } _ { c } \biggl [ \int _ { 0 } ^ { 1 } Q _ { \pi _ { \theta } } ( u \mid c ) \phi ( u ) d u \biggr ] .\tag{10}
$$

GenEval2 Best-of-N gap versus matched SD3.5-M by prompt atomicity

![](images/3ed83669c2a56e81f4fdeacacec8868543707b87cf8d299e7d0fb997923789c3.jpg)  
Figure 3Scaling transfer by prompt atomicity on GenEval2. Best-of-N advantages relative to matched SD3.5-M baselines: CFG 4.5 for FlowGRPO and GRPO-Guard, and CFG 1.0 for DiffusionNFT and AWM. Lower-atomicity prompts exhibit substantial low-budget gains, whereas the highest-atomicity group exhibits negative high-budget transfer across methods. Shading denotes 95% confidence intervals; vertical scales differ across panels.

Expected-reward optimization uses the uniform weighting $\phi ( u ) = 1$ , while Theorem 1 shows that Best-of-k corresponds to $\phi _ { k } ( u ) = k u ^ { k - 1 }$ , increasingly emphasizing the upper tail as k grows.

We therefore optimize a weighted combination of deployment budgets,

$$
\phi _ { q } ( u ) = \sum _ { k = 1 } ^ { T } q _ { k } k u ^ { k - 1 } , \qquad \mathcal { I } _ { \mathrm { Q U P O } } ( \theta ) = \sum _ { k = 1 } ^ { T } q _ { k } B _ { k } ^ { \pi _ { \theta } } .\tag{11}
$$

In practice, we parameterize the budget weights as $q _ { k } = k ^ { \alpha }$ , giving two intuitive controls. The horizon T determines the largest Best-of-k utility explicitly represented during training, while α controls the relative emphasis placed on smaller versus larger sampling budgets. Notably, setting $T = 1$ recovers the standard expected-reward optimization objective.

We call the resulting objective Quantile Utility Policy Optimization (QuPo). QUPO does not directly regularize diversity or optimize only a single large-k operating point; instead, it exposes the quantile weighting induced by test-time sampling as a controllable training objective.

## 6.2 Practical Advantage Estimation

Given a rollout group of G generations with rewards $r _ { 1 } , \ldots , r _ { G }$ , we measure the contribution of sample i to Best-of-k utility as

$$
a _ { i } ^ { ( k ) } = \mathbb { E } _ { S } \left[ \operatorname* { m a x } _ { j \in S } r _ { j } \mid i \in S , \ | S | = k \right] - \mathbb { E } _ { S } \left[ \operatorname* { m a x } _ { j \in S } r _ { j } \mid i \notin S , \ | S | = k \right] .\tag{12}
$$

![](images/d3fc83cbf9a5951a4569ee7ba385dffce977050d7bf6a8ca777130d98e62cfb2.jpg)  
Figure 4 Multiple generations for the same held-out GenEval2 prompt. Each row shows four independent samples from QUPO, the base BAGEL model, and UniGRPO for the prompt "two flamingos and five pink rabbits", under matched generation seeds. QUPO more consistently produces samples satisfying the requested composition, qualitatively reflecting its strong scaling transfer.

This measures how much including sample i improves the expected Best-of-k reward. Combining budgets gives the QUPO advantage

$$
A _ { i } ^ { \mathrm { Q U P O } } = \sum _ { k = 1 } ^ { T } k q _ { k } a _ { i } ^ { ( k ) } .\tag{13}
$$

For $T \leq G - 1$ , this yields an unbiased policy-gradient estimator for Eq. 11; the full derivation and discussion of its efficient computation are given in Appendix A.2.

QUPO is a drop-in replacement for the scalar rollout advantage. In FlowGRPO [20] and UniGRPO [21], we replace the original group-relative advantage with $A _ { i } ^ { \mathrm { Q U P O } }$ while leaving the remaining optimization pipeline—including rollout generation, likelihood computation, clipping, and regularization—unchanged.

## 7 Experiments

## 7.1 Experimental Setup

We focus on compositional generation, as it is the only diagnostic setting exhibiting frontier reversal. Models are post-trained on GenEval [11] and evaluated on the harder held-out GenEval2 [15] and ConceptMix [41] benchmarks.

We evaluate QUPO on SD3.5-M and BAGEL [8]. For SD3.5-M, we compare against FlowGRPO [20], DiffusionNFT [49], GRPO-Guard [38], and AWM [42], generating 128 images per prompt; for BAGEL, we compare against UniGRPO [21], generating 64 images per prompt to mitigate compute costs. QUPO changes only the rollout advantage in the corresponding training pipeline, with applicable training settings matched. We evaluate average Best-of-N performance using the estimator from Sec. 4.1, ensuring standardized generation seeds for fair comparison across models. GenEval2 (800 prompts) and ConceptMix (2100 prompts) are scored with Soft-TIFA using Qwen3-VL-8B-Instruct [2], following established evaluation protocol [15]. Further implementation details, ablations, and diversity analysis are deferred to the appendix.

(a) GenEval2.
<table><tr><td>Model</td><td>CFG</td><td>GenEval</td><td>Bol</td><td>Bo4</td><td>Bo8</td><td>Bo16</td><td>Bo32</td><td>Bo64</td><td>Bo96</td><td>Bo128</td></tr><tr><td>SD3.5-M</td><td>4.5</td><td>0.63</td><td>0.1804</td><td>0.3365</td><td>0.4193</td><td>0.4978</td><td>0.5679</td><td>0.6295</td><td>0.6625</td><td>0.6850</td></tr><tr><td>QUPO</td><td>4.5</td><td>0.88</td><td>0.2287</td><td>0.3841</td><td>0.4618</td><td>0.5333</td><td>0.5962</td><td>0.6508</td><td>0.6794</td><td>0.7000</td></tr><tr><td>FlowGRPO†</td><td>4.5</td><td>0.91</td><td>0.2190</td><td>0.3469</td><td>0.4135</td><td>0.4793</td><td>0.5412</td><td>0.5971</td><td>0.6269</td><td>0.6471</td></tr><tr><td>FlowGRPO (official)</td><td>4.5</td><td>0.95</td><td>0.2351</td><td>0.3727</td><td>0.4413</td><td>0.5067</td><td>0.5662</td><td>0.6186</td><td>0.6468</td><td>0.6664</td></tr><tr><td>GRPO-Guard†</td><td>4.5</td><td>0.95</td><td>0.2395</td><td>0.3644</td><td>0.4279</td><td>0.4883</td><td>0.5448</td><td>0.5968</td><td>0.6248</td><td>0.6436</td></tr><tr><td>SD3.5-M</td><td>1.0</td><td>0.23</td><td>0.0998</td><td>0.2383</td><td>0.3223</td><td>0.4051</td><td>0.4813</td><td>0.5485</td><td>0.5833</td><td>0.6060</td></tr><tr><td>DiffusionNFT</td><td>1.0</td><td>0.95</td><td>0.1825</td><td>0.2991</td><td>0.3584</td><td>0.4152</td><td>0.4682</td><td>0.5168</td><td>0.5431</td><td>0.5608</td></tr><tr><td>AWM</td><td>1.0</td><td>0.90</td><td>0.2041</td><td>0.3240</td><td>0.3830</td><td>0.4391</td><td>0.4907</td><td>0.5379</td><td>0.5637</td><td>0.5812</td></tr></table>

(b) ConceptMix.
<table><tr><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>SD3.5-M</td><td rowspan=1 colspan=1>QUPO</td><td rowspan=1 colspan=1>FlowGRPO</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.4107</td><td rowspan=1 colspan=1>0.4578</td><td rowspan=1 colspan=1>0.4546</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.6742</td><td rowspan=1 colspan=1>0.6980</td><td rowspan=1 colspan=1>0.6704</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>0.7338</td><td rowspan=1 colspan=1>0.7506</td><td rowspan=1 colspan=1>0.7199</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>0.7823</td><td rowspan=1 colspan=1>0.7923</td><td rowspan=1 colspan=1>0.7607</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.8220</td><td rowspan=1 colspan=1>0.8255</td><td rowspan=1 colspan=1>0.7947</td></tr><tr><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>0.8411</td><td rowspan=1 colspan=1>0.8432</td><td rowspan=1 colspan=1>0.8119</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0.8543</td><td rowspan=1 colspan=1>0.8559</td><td rowspan=1 colspan=1>0.8231</td></tr></table>

† Trained with the official implementation using an optimization budget matched to QUPO.  
Table 1 QUPO improves scaling transfer on SD3.5-M. On GenEval2, standard RL methods eventually fall below the matched base model, whereas QUPO remains above it through Best-of-128. The improvement transfers to ConceptMix.  
GenEval2 Soft-TIFA GM: quantile lift over matched SD3.5-M base

![](images/7e40c866925d814e7d09cf32090444c38bea925c4e5bb75838205e935a37d46c.jpg)  
(a) Reward quantile lift.

![](images/e0ed0848915e9888ef3adeba145f45cc566d0ed9d72b7f36fa7e5aa72f9318c0.jpg)

![](images/fe1ba15101149f1ddb0e0a9d93ada8f30e4666cd802f68aafdcb8e499e5d0685.jpg)  
(b) Large-N scaling advantage.  
Figure 5 QUPO preserves upper-tail reward quality and mitigates scaling-transfer failure on GenEval2. Left: FlowGRPO and GRPO-Guard exhibit negative upper-tail quantile lift relative to the matched base model, whereas QUPO remains positive. Right: Best-of-N post-training advantage relative to the matched base model for N = 64–128, with shaded bands indicating paired prompt-bootstrap 95% confidence intervals. QUPO maintains positive estimated advantage, whereas standard RL methods exhibit frontier reversal.

## 7.2 Scaling Transfer on SD3.5-M

We first evaluate QUPO on SD3.5-M using the GenEval post-training setup from our diagnostic study. On held-out GenEval2 prompts (Table 1), QUPO improves Best-of-1 from 0.1804 to 0.2287 and remains above the base model through Best-of-128 (0.7000 vs. 0.6850). In contrast, FlowGRPO and GRPO-Guard achieve comparable or stronger Best-of-1 and higher in-distribution GenEval scores, yet are overtaken as N increases Figure 5 explains this difference: standard RL exhibits negative upper-tail quantile lift, whereas QUPO maintains positive lift in the region increasingly emphasized at large N.

While Theorem 1 provides an exact relationship between quantile lift and Best-of-N advantage within a fixed prompt distribution, an important question is whether upper-tail behavior measured on one benchmark can also characterize scaling transfer on another. We investigate this on ConceptMix using the quantile analysis obtained independently from GenEval2, without using ConceptMix reward distributions to construct the quantile lifts. As shown in Table 1b, the qualitative scaling regimes transfer across benchmarks: FlowGRPO exhibits negative upper-tail lift on GenEval2 and undergoes frontier reversal on ConceptMix, while QUPO maintains positive upper-tail lift on GenEval2 and approximately preserves the base model's large-N frontier on ConceptMix (0.8559 vs. 0.8543). This cross-benchmark consistency suggests that the upper-tail effects characterized by our quantile analysis can generalize across compositional prompt distributions, providing a useful diagnostic of scaling-transfer behavior beyond the benchmark on which the quantile shifts are measured.

Table 2 Scaling transfer on BAGEL. UniGRPO improves Best-of-1 but degrades the large-N frontier; QUPO largely preserves it.
<table><tr><td>Model</td><td>GenEval</td><td>Bo1</td><td>Bo4</td><td>Bo8</td><td>Bo16</td><td>Bo32</td><td>Bo64</td></tr><tr><td>BAGEL</td><td>0.74</td><td>0.2351</td><td>0.4188</td><td>0.5091</td><td>0.5890</td><td>0.6579</td><td>0.7166</td></tr><tr><td>UniGRPO</td><td>0.85</td><td>0.2420</td><td>0.4063</td><td>0.4892</td><td>0.5653</td><td>0.6321</td><td>0.6891</td></tr><tr><td>QUPO</td><td>0.83</td><td>0.2468</td><td>0.4240</td><td>0.5120</td><td>0.5905</td><td>0.6583</td><td>0.7177</td></tr></table>

## 7.3 Scaling Transfer on BAGEL

We next test whether both the scaling-transfer failure and the benefits of QUPO extend beyond flow-based image generators. We use BAGEL [8], a unified multimodal model with a qualitatively different generative architecture, and compare the base model against UniGRPO [21] and QUPO. All models are post-trained on GenEval and evaluated on held-out GenEval2 prompts at 512 × 512 resolution.

The same scaling-transfer failure appears on BAGEL: UniGRPO improves GenEval and GenEval2 Best-of-1 but falls increasingly below the base model as N grows. QUPO performs best through Best-of-64, remaining slightly ahead of the base while outperforming UniGRPO by nearly 0.03. Thus, both weakened scaling transfer and its mitigation through upper-tail-aware training generalize across qualitatively different visual generators.

## 8 Conclusion

We study scaling transfer in visual generative models: whether gains from RL persist as test-time sampling compute increases on difficult held-out prompts. Across multiple RL methods, tasks, and model families, we show that stronger Best-of-1 performance does not necessarily imply a better quality-compute frontier, with regimes ranging from persistent gains to attenuation and complete frontier reversal. We explain these behaviors through a quantile-transfer analysis showing that larger Best-of-N budgets increasingly depend on the upper tail of the reward distribution. Motivated by this insight, we introduce QUPO, a simple quantile-aware objective that better preserves upper-tail quality and improves scaling transfer on both SD3.5-M and BAGEL. More broadly, our findings expose a limitation of single-sample evaluation and motivate treating the full quality-compute frontier as an essential measure of visual RL post-training. Strong single-sample evaluation need not transfer to strong test-time scalability.

## References

[1] Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal of Machine Learning Research, 26(209):1–80, 2025.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[3] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Roni Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In International Conference on Learning Representations volume 2024, pages 51304–51323, 2024.

[4] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pages 4965–4987, 2024.

[5] Ziheng Cheng, Yixiao Huang, Hanlin Zhu, and Somayeh Sojoudi. How rl reshapes llm reasoning: Transferability, coverage, and scaling laws. arXiv preprint arXiv:2610.04158, 2026.

[6] Jaemoo Choi, Yuchen Zhu, Wei Guo, Petr Molodyk, Bo Yuan, Jinbin Bai, Yi Xin, Molei Tao, and Yongxin Chen. Rethinking the design space of reinforcement learning for diffusion models: On the importance of likelihood estimation beyond loss design. arXiv preprint arXiv:2602.04663, 2026.

[7] Kevin Clark, Paul Vicol, Kevin Swersky, and David Fleet. Directly fine-tuning diffusion models on differentiable rewards. In International Conference on Learning Representations, volume 2024, pages 4793–4822, 2024

[8] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 2025.

[9] Forouzan Fallah, Maitreya Patel, Agneet Chatterjee, Vlad Morariu, Chitta Baral, and Yezhou Yang. Textinvision: Text and prompt complexity driven visual text generation benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 525–534, 2025.

[10] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in neural information processing systems, 36:79858–79885, 2023.

[11] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.

12 Jiaqi Han, Austin Wang, Minkai Xu, Wenda Chu, Meihua Dang, Haotian Ye, Huayu Chen, Yisong Yue, and Stefano Ermon. Discrete diffusion trajectory alignment via stepwise decomposition. In International Conference on Learning Representations, volume 2026, pages 5023–5047, 2026.

[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[14] Jerry Y Huang, Justin Lin, Sheel Shah, Kartik Nair, and Nicholas M Boffi. How to guide your flow: Few-step alignment via flow map reward guidance. arXiv preprint arXiv:2604.27147, 2026.

[15] Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, and Marjan Ghazvininejad. Geneval 2: Addressing benchmark drift in text-to-image evaluation. arXiv preprint arXiv:2512.16853, 2025.

[16] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022.

[17] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

[18] Muheng Li, Jian Qian, and Wenlong Mou. What should post-training optimize? a test-time scaling law perspective. arXiv preprint arXiv:2605.10716, 2026.

[19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022

[20] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. Advances in neural information processing systems, 38:40783–40818, 2026.

[21] Jie Liu, Zilyu Ye, Linxiao Yuan, Shenhan Zhu, Yu Gao, Jie Wu, Kunchang Li, Xionghui Wang, Xiaonan Nie, Weilin Huang, et al. Unigrpo: Unified policy optimization for reasoning-driven visual generation. arXiv preprint arXiv:2603.23500, 2026.

[22] Yunhong Lu, Qichao Wang, Hengyuan Cao, Xierui Wang, Xiaoyin Xu, and Min Zhang. Inpo: Inversion preference optimization with reparametrized ddim for efficient diffusion model alignment. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 28629–28639. IEEE, 2025.

[23] Yunhong Lu, Qichao Wang, Hengyuan Cao, Xiaoyin Xu, and Min Zhang. Smoothed preference optimization via renoise inversion for aligning diffusion models with varied human preferences. arXiv preprint arXiv:2506.02698, 2025.

[24] Ku Onoda, Paavo Parmas, Hiroki Furuta, Soichiro Nishimori, Yuta Oshima, Shohei Taniguchi, and Yutaka Matsuo. Multi-axis max@ k reinforcement learning for representative diversity in text-to-image generation. arXiv preprint arXiv:2607.14962, 2026.

[25] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. 2023.

[26] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

[27] Shrinivas Ramasubramanian, Daman Arora, Fahim Tajwar, Guanning Zeng, Qingyang Wu, Zhongzhu Zhou Chenfeng Xu, Haiwen Feng, Yuda Song, Aarti Singh, et al. Tail-likelihood reinforcement learning. arXiv preprint arXiv:2609.02987, 2026.

[28] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[29] Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen McKeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. arXiv preprint arXiv:2501.06848, 2025.

[30] Fahim Tajwar, Guanning Zeng, Yueer Zhou, Yuda Song, Daman Arora, Yiding Jiang, Jeff Schneider, Ruslan Salakhutdinov, Haiwen Feng, and Andrea Zanette. Maximum likelihood reinforcement learning. arXiv preprint arXiv:2602.02710, 2026.

[31] Shota Takashiro, Soichiro Nishimori, Paavo Parmas, Yongmin Kim, Kohsei Matsutani, Gouki Minegishi, Yusuke Iwasawa, Takeshi Kojima, and Yutaka Matsuo. On advantage estimates for max@ k policy gradients. arXiv preprint arXiv:2606.06080, 2026.

[32] Masatoshi Uehara, Yulai Zhao, Chenyu Wang, Xiner Li, Aviv Regev, Sergey Levine, and Tommaso Biancalani. Inference-time alignment in diffusion models with reward-guided generation: Tutorial and review. arXiv preprint arXiv:2501.09685, 2025.

[33] Christian Walder and Deep Tejas Karkhanis. Pass@ k policy optimization: Solving harder reinforcement learning problems. Advances in Neural Information Processing Systems, 38:152416–152445, 2026.

[34] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8228–8238. IEEE, 2024.

[35] Austin Wang, Hongkai Zheng, Zihui Wu, Ricardo Baptista, Daniel Zhengyu Huang, and Yisong Yue. Ensemble kalman sampling and diffusion prior in tandem: A split gibbs framework. In Frontiers in Probabilistic Inference: Learning meets Sampling, 2025.

[36] Austin Wang, Ziheng Cheng, and Lexing Ying. Aligning one-step generative models with reward-weighted transport distillation. arXiv preprint arXiv:2609.30840, 2026.

[37] Austin Wang, Jiaqi Han, Stefano Ermon, and Yisong Yue. Beyond pairwise preferences: Listwise reward-aware alignment for diffusion models. arXiv preprint arXiv:2605.26491, 2026

[38] Jing Wang, Jiajun Liang, Jie Liu, Henglin Liu, Gongye Liu, Jun Zheng, Wanyuan Pang, Ao Ma, Zhenyu Xie, Xintao Wang, et al. Grpo-guard: Mitigating implicit over-optimization in flow matching via regulated clipping. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5988–5998, 2026.

[39] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[40] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

[41] Xindi Wu, Dingli Yu, Yangsibo Huang, Olga Russakovsky, and Sanjeev Arora. Conceptmix: A compositional image generation benchmark with controllable difficulty. Advances in Neural Information Processing Systems, 37: 86004–86047, 2024

[42] Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025.

[43] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

[44] Po-Hung Yeh, Kuang-Huei Lee, and Jun-Cheng Chen. Training-free diffusion model alignment with sampling demons. In International Conference on Learning Representations, volume 2025, pages 95708–95744, 2025.

[45] Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song. Improving diffusion inverse problem solving with decoupled noise annealing. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20895–20905. IEEE, 2025.

[46] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF conference on computer vision and pattern recognition, pages 586–595. IEEE, 2018.

[47] Hongkai Zheng, Wenda Chu, Austin Wang, Nikola Kovachki, Ricardo Baptista, and Yisong Yue. Ensemble kalman diffusion guidance: A derivative-free method for inverse problems. arXiv preprint arXiv:2409.20175, 2024.

[48] Hongkai Zheng, Austin Wang, Zihui Wu, Zhengyu Huang, Ricardo Baptista, and Yisong Yue. Blade: A derivative-free bayesian inversion method using diffusion priors. arXiv preprint arXiv:2510.10968, 2025.

[49] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. In International Conference on Learning Representations, volume 2026, pages 134129–134150, 2026.

## Appendix

## Appendix Contents

A Proofs and Derivations 14   
A.1 Proof of Theorem 1 14   
A.2 Derivation of the QUPO Policy Gradient 15   
A.3 Implementation of the QUPO Advantage 17   
B Connections to Related Objectives . 18   
B.1 Connection to TailRL 18   
B.2 Connection to Multi-Axis Max@K 18   
Experimental and Implementation Details 19   
C.1 Diagnostic Experiments 19   
C.2 Post-Training Hyperparameters 20   
C.3 Evaluation Hyperparameters 20   
D Additional Ablations 20   
D.1 Effect of the Optimization Horizon T 20   
D.2 Effect of the Weighting Parameter α 21   
D.3 Effect of Normalized Advantage Coefficients 22   
D.4 Scaling-Transfer Robustness Across Evaluators and Selectors 22   
D.5 Comparison with a Single Best-of-k Objective 23   
D.6 Does Early Stopping Benefit RL Baselines? 23   
E Diversity and Qualitative Results 24   
E.1 Perceptual Diversity 24   
E.2 Additional Qualitative Comparisons 24

## A Proofs and Derivations

## A.1 Proof of Theorem 1

Proof. Fix a prompt c and model π. Let

$$
F _ { \pi } ( t \mid c ) = \mathbb { P } ( r ( \mathbf { x } , c ) \leq t ) , \qquad \mathbf { x } \sim \pi ( \cdot \mid c ) ,
$$

with quantile function $Q _ { \pi } ( u \mid c )$ . Let $\mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { N } \overset { \mathrm { i . i . d . } } { \sim }$ Uniform(0, 1). By the inverse-transform representation,

$$
\begin{array} { r } { Q _ { \pi } ( \mathbf { u } _ { i } \mid c ) \stackrel { d } { = } r ( \mathbf { x } _ { i } , c ) , \qquad \mathbf { x } _ { i } \sim \pi ( \cdot \mid c ) . } \end{array}
$$

Since $Q _ { \pi } ( \cdot \mid c )$ is nondecreasing,

$$
\operatorname* { m a x } _ { 1 \leq i \leq N } Q _ { \pi } ( \mathrm { u } _ { i } \mid c ) = Q _ { \pi } \biggl ( \operatorname* { m a x } _ { 1 \leq i \leq N } \mathrm { u } _ { i } \bigg | c \biggr ) .\tag{14}
$$

Define the maximum order statistic

$$
\mathrm { u } _ { ( N ) } = \operatorname* { m a x } _ { 1 \leq i \leq N } \mathrm { u } _ { i } .
$$

For $u \in [ 0 , 1 ]$

$$
\mathbb { P } ( \mathbf { u } _ { ( N ) } \le u ) = \prod _ { i = 1 } ^ { N } \mathbb { P } ( \mathbf { u } _ { i } \le u ) = u ^ { N } ,
$$

SO $\mathrm { u } _ { ( N ) } \sim$ Beta $, ( N , 1 )$ with density

$$
p _ { N } ( u ) = N u ^ { N - 1 } .
$$

Therefore, the conditional Best-of-N reward is

$$
B _ { N } ^ { \pi } ( c ) = \mathbb { E } _ { \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { N } \sim \pi ( \cdot | c ) } \left[ \operatorname* { m a x } _ { i } r ( \mathbf { x } _ { i } , c ) \right]\tag{15}
$$

$$
= \mathbb { E } _ { \mathbf { u } _ { ( N ) } } \left[ Q _ { \pi } ( \mathbf { u } _ { ( N ) } \mid c ) \right]\tag{16}
$$

$$
= \int _ { 0 } ^ { 1 } Q _ { \pi } ( u \mid c ) N u ^ { N - 1 } d u .\tag{17}
$$

Averaging over prompts and using integrability to exchange the prompt expectation and integral gives

$$
B _ { N } ^ { \pi } = \int _ { 0 } ^ { 1 } \mathbb { E } _ { c } [ Q _ { \pi } ( u \mid c ) ] N u ^ { N - 1 } d u .\tag{18}
$$

Applying this identity to $\pi _ { \mathrm { R L } }$ and $\pi _ { \mathrm { r e f } }$ and subtracting,

$$
A _ { N } = B _ { N } ^ { \pi _ { \mathrm { R L } } } - B _ { N } ^ { \pi _ { \mathrm { r e f } } }\tag{19}
$$

$$
= \int _ { 0 } ^ { 1 } \mathbb { E } _ { c } \left[ Q _ { \pi _ { \mathrm { R L } } } ( u \mid c ) - Q _ { \pi _ { \mathrm { r e f } } } ( u \mid c ) \right] N u ^ { N - 1 } d u\tag{20}
$$

$$
= \int _ { 0 } ^ { 1 } \Delta Q ( u ) N u ^ { N - 1 } d u .\tag{21}
$$

Since $N u ^ { N - 1 }$ is the density of Beta(N, 1), this is equivalently

$$
A _ { N } = \mathbb { E } _ { \mathrm { u } _ { N } \sim \mathrm { B e t a } ( N , 1 ) } \left[ \Delta Q ( \mathrm { u } _ { N } ) \right] ,\tag{22}
$$

which proves the result.

## A.2 Derivation of the QUPO Policy Gradient

We derive the policy-gradient estimator used for the QUPO objective

$$
\mathcal { T } _ { \mathrm { Q U P O } } ( \theta ) = \sum _ { k = 1 } ^ { T } q _ { k } B _ { k } ^ { \pi _ { \theta } } .\tag{23}
$$

We first consider a single Best-of-k objective. For k independent generations $\mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { k } \sim \pi _ { \theta } ( \cdot \mid c )$ , define

$$
M _ { k } = \operatorname* { m a x } _ { 1 \leq i \leq k } r ( { \mathbf x } _ { i } , c ) .\tag{24}
$$

Applying the score-function identity gives

$$
\nabla _ { \theta } B _ { k } ^ { \pi _ { \theta } } = \mathbb { E } _ { c } \mathbb { E } _ { \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { k } } \left[ M _ { k } \nabla _ { \theta } \log \prod _ { i = 1 } ^ { k } \pi _ { \theta } ( \mathbf { x } _ { i } \mid c ) \right]\tag{25}
$$

$$
= \mathbb { E } _ { c } \mathbb { E } _ { \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { k } } \left[ M _ { k } \sum _ { i = 1 } ^ { k } \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf { x } _ { i } \mid c ) \right] .\tag{26}
$$

In practice, visual RL methods sample a group of $G \geq k$ generations per prompt. Rather than selecting a single size-k subset, we average Eq. 26 over all $\dot { ( \boldsymbol { { G } } ) }$ subsets. For $S \subseteq \{ 1 , \dots , G \}$ with $| S | = k ,$ let

$$
M _ { S } = \operatorname* { m a x } _ { j \in S } r _ { j } , \qquad r _ { j } = r ( \mathbf { x } _ { j } , c ) .\tag{27}
$$

An unbiased estimator of the gradient is then

$$
\widehat { \nabla _ { \theta } B _ { k } } = \frac { 1 } { { \binom { G } { k } } } \sum _ { { S \subseteq [ G ] } \atop { | S | = k } } M _ { S } \sum _ { i \in S } \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf { x } _ { i } \mid c ) .\tag{28}
$$

Rearranging the sum by generation yields

$$
{ \widehat { \nabla _ { \theta } B _ { k } } } = { \frac { k } { G } } \sum _ { i = 1 } ^ { G } u _ { i } ^ { ( k ) } \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf { x } _ { i } \mid c ) ,\tag{29}
$$

where

$$
u _ { i } ^ { ( k ) } = \frac { 1 } { \binom { G - 1 } { k - 1 } } \sum _ { S \subseteq [ G ] \atop | S | = k , i \in S } M _ { S }\tag{30}
$$

is the average Best-of-k utility over subsets containing sample i. The factor $k / G$ follows from ${ \binom { G - 1 } { k - 1 } } / { \binom { G } { k } } = k / G$ To improve credit assignment, we subtract a leave-one-out baseline defined by the expected Best-of-k utility of subsets that exclude sample i:

$$
v _ { i } ^ { ( k ) } = \frac { 1 } { \binom { G - 1 } { k } } \sum _ { \stackrel { S \subseteq [ G ] } { | S | = k , i \notin S } } M _ { S } , \qquad k \leq G - 1 .\tag{31}
$$

Because $v _ { i } ^ { ( k ) }$ depends only on the other $G - 1$ samples, it is independent of $\mathbf { x } _ { i }$ conditional on those samples. Hence,

$$
\begin{array} { r } { \mathbb { E } \left[ v _ { i } ^ { ( k ) } \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf { x } _ { i } \mid c ) \right] = 0 , } \end{array}\tag{32}
$$

so subtracting this baseline preserves the expected policy gradient. Defining

$$
a _ { i } ^ { ( k ) } = u _ { i } ^ { ( k ) } - v _ { i } ^ { ( k ) } ,\tag{33}
$$

we obtain

$$
{ \widehat { \nabla _ { \theta } B _ { k } } } = { \frac { k } { G } } \sum _ { i = 1 } ^ { G } a _ { i } ^ { ( k ) } \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf { x } _ { i } \mid c ) .\tag{34}
$$

Finally, by linearity of the gradient,

$$
\begin{array} { r l } & { \widehat { \nabla _ { \theta } \mathcal { I } _ { \mathrm { Q U P O } } } = \displaystyle \sum _ { k = 1 } ^ { T } q _ { k } \widehat { \nabla _ { \theta } B _ { k } } } \\ & { \qquad = \displaystyle \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \left( \sum _ { k = 1 } ^ { T } k q _ { k } a _ { i } ^ { ( k ) } \right) \nabla _ { \theta } \log \pi _ { \theta } ( \mathbf { x } _ { i } \mid c ) . } \end{array}\tag{35}
$$

(36)

Thus, defining the QUPO advantage

$$
A _ { i } ^ { \mathrm { Q U P O } } = \sum _ { k = 1 } ^ { T } k q _ { k } a _ { i } ^ { ( k ) } ,\tag{37}
$$

gives the sample-level policy-gradient estimator used in Sec. 6.2. In our implementation, $T \leq G - 1$ , ensuring that the leave-one-out baseline is well-defined for every included budget. For FlowGRPO and UniGRPO, this scalar advantage replaces the original group-relative advantage while the remaining optimization surrogate is unchanged.

For each fixed $k , a _ { i } ^ { ( k ) }$ coincides with the canonical centered finite-batch Max@k advantage used by MaxPO [31]; we include the derivation for completeness and to make the coefficients $k q _ { k }$ in the QUPO mixture explicit. While the per-budget advantage coincides with that of MaxPO, our primary contribution is the systematic empirical characterization of scaling transfer in visual $\mathrm { R L , }$ with QUPO serving as a simple intervention to mitigate the observed failures.

## A.3 Implementation of the QUPO Advantage

Although the definition of $a _ { i } ^ { ( k ) }$ averages over all size-k subsets, it can be computed efficiently without enumerating subsets. For a rollout group with rewards $r _ { 1 } , \ldots , r _ { G }$ , we first sort the samples such that

$$
r _ { 1 } \leq r _ { 2 } \leq \cdot \cdot \cdot \leq r _ { G } .
$$

Consider the sample at sorted position $j .$ Among size-k subsets containing $j ,$ its reward $r _ { j }$ is the maximum when the remaining $k - 1$ elements are chosen from the preceding $j - 1$ samples. If instead a later sample $\ell > j$ is the maximum, the remaining $k - 2$ elements must be selected from the $\ell - 2$ samples preceding l other than $j .$ Hence,

$$
u _ { j } ^ { ( k ) } = \frac { r _ { j } { \binom { j - 1 } { k - 1 } } + \sum _ { \ell = j + 1 } ^ { G } r _ { \ell } { \binom { \ell - 2 } { k - 2 } } } { { \binom { G - 1 } { k - 1 } } } .\tag{38}
$$

Similarly, for subsets excluding $j ,$ a sample $\ell < j$ is the maximum in $\binom { \ell - 1 } { k - 1 }$ subsets, whereas a sample $\ell > j$ is the maximum in $\binom { \ell - 2 } { k - 1 }$ subsets. Therefore,

$$
v _ { j } ^ { ( k ) } = \frac { \sum _ { \ell = 1 } ^ { j - 1 } r _ { \ell } { \binom { \ell - 1 } { k - 1 } } + \sum _ { \ell = j + 1 } ^ { G } r _ { \ell } { \binom { \ell - 2 } { k - 1 } } } { { \binom { G - 1 } { k } } } .\tag{39}
$$

We then compute

$$
a _ { j } ^ { ( k ) } = u _ { j } ^ { ( k ) } - v _ { j } ^ { ( k ) } , \qquad A _ { j } ^ { \mathrm { Q U P O } } = \sum _ { k = 1 } ^ { T } k q _ { k } a _ { j } ^ { ( k ) } ,\tag{40}
$$

and map the resulting advantages back to the original rollout ordering. We use the convention ${ \binom { n } { m } } = 0$ when m < 0 or $m > n$ . After sorting, the sums in Eqs. 38–39 can be evaluated with prefix and suffix accumulations, giving $O ( T G )$ computation after the initial O(G log G) sort. Since rollout groups are small in our experiments, this overhead is negligible relative to generation and likelihood evaluation.

QUPO is implemented as a direct replacement for the scalar rollout advantage in existing visual RL pipelines. For FlowGRPO [20], we replace the original group-relative advantage with $A _ { i } ^ { \mathrm { Q U P O } }$ and otherwise retain the official training procedure. Likewise, for UniGRPO [21], we substitute the QUPO advantage while leaving the remaining optimization pipeline unchanged. QUPO does not modify the model parameterization, rollout procedure, policy surrogate, clipping or regularization terms, or the method used to compute rollout likelihoods. This last point is important for visual RL, where likelihood estimation for diffusion, flow, or unified multimodal generators is itself a central algorithmic component. QUPO operates only on the rewardto-advantage transformation and is therefore compatible with the likelihood computation already provided by each underlying method.

## B Connections to Related Objectives

In this section, we discuss the two objectives most closely related to QUPO. TailRL [27] is closely connected at the objective level through its harmonic mixture of Best-of-k utilities, while Multi-Axis Max@K [24| focuses on Max@K-aware post-training for diversity optimization. We emphasize that QUPO is a simple intervention motivated by the broader, primary contribution of this work: identifying and analyzing scaling-transfer failure in visual RL.

## B.1 Connection to TailRL

TailRL, which was developed concurrently and independently, is closely related to QUPO in its motivation to preserve rare high-reward outcomes under increased inference-time sampling. For rewards in [0, 1], TailRL shows that its finite-order objective can be written as

$$
\mathcal { I } _ { \mathrm { T a i l R L } } ^ { ( T ) } ( \theta ) = \sum _ { k = 1 } ^ { T } \frac { B _ { k } ^ { \pi _ { \theta } } - 1 } { k } .\tag{41}
$$

Thus, TailRL implicitly assigns harmonic weight $1 / k$ to each Best-of-k objective.

QUPO uses the more general objective

$$
\mathcal { I } _ { \mathrm { Q U P O } } ( \boldsymbol { \theta } ) = \sum _ { k = 1 } ^ { T } q _ { k } B _ { k } ^ { \pi _ { \theta } } , \qquad q _ { k } = k ^ { \alpha } .\tag{42}
$$

Setting α = −1 gives

$$
\mathcal { I } _ { \mathrm { Q U P O } } ^ { \alpha = - 1 } ( \theta ) = \sum _ { k = 1 } ^ { T } \frac { B _ { k } ^ { \pi _ { \theta } } } { k } .\tag{43}
$$

Letting $\textstyle H _ { T } = \sum _ { k = 1 } ^ { T } 1 / k$ denote the T-th harmonic number,

$$
{ \mathcal { I } } _ { \mathrm { T a i l R L } } ^ { ( T ) } ( \theta ) = { \mathcal { I } } _ { \mathrm { Q U P O } } ^ { \alpha = - 1 } ( \theta ) - H _ { T } .\tag{44}
$$

Therefore, for a fixed horizon T, TailRL is exactly the $\alpha = - 1$ instance of QUPO up to an additive constant, and the two objectives have identical gradients. This equivalence underscores that the novelty of our work does not rest on the introduction of a new class of Best-of-k objectives. Rather, these objectives provide a convenient means to test and address the scaling-transfer failure identified through our visual RL experiments.

The distinction is primarily in how the objective is motivated and parameterized. TailRL derives the harmonic weighting from a tail-likelihood objective, whereas QUPO begins from the equivalence between quantile weighting and the Best-of-N quality-compute frontier. This perspective treats the weighting over sampling budgets as a controllable design choice: α determines how strongly larger Best-of-k utilities are emphasized, while T determines the largest budget represented explicitly during training. TailRL corresponds to the particular choice $\alpha = - 1$

Our ablations highlight two practical findings distinct from TailRL: the harmonic choice $\alpha = - 1$ need not be optimal for visual generation, and the optimization horizon T need not be coupled to the rollout group size G. In practice, moderate $T < G$ can already provide sufficient upper-tail emphasis to improve scaling at substantially larger deployment budgets. We study both effects in Sec. D.

## B.2 Connection to Multi-Axis Max@K

Multi-Axis Max@K [24] also uses set-level utilities to post-train visual generators, but addresses a different objective. It formalizes diversity as coverage of predefined semantic modes (i.e., color and gender) and maximizes, for each mode, the score of its best representative among a group of samples before aggregating across modes. This allows different samples to receive credit for covering different categories.

Table 3 Diagnostic scaling-transfer settings. Training denotes reproduction using the official implementation; Official LoRA denotes an author-released adapter.
<table><tr><td>Domain</td><td>Method</td><td>Post-training source</td><td>Acquisition</td><td>Held-out evaluation</td></tr><tr><td>Compositionality</td><td>FlowGRPO DiffusionNFT</td><td>GenEval GenEval</td><td>Official LoRA</td><td>GenEval2</td></tr><tr><td></td><td>GRPO-Guard</td><td>GenEval</td><td>Official training Official LoRA</td><td>GenEval2 GenEval2</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>AWM</td><td>GenEval</td><td>Official training</td><td>GenEval2</td></tr><tr><td>Text rendering</td><td>FlowGRPO</td><td>OCR training set</td><td>Official LoRA</td><td>TextInVision</td></tr><tr><td></td><td>DiffusionNFT</td><td>OCR training set</td><td>Official training</td><td>TextInVision</td></tr><tr><td></td><td>GRPO-Guard</td><td>OCR training set</td><td>Official LoRA</td><td>TextInVision</td></tr><tr><td>Preference alignment</td><td>FlowGRPO</td><td>Pick-a-Pic + PickScore</td><td>Official LoRA</td><td>Parti-Prompts + HPSv2</td></tr><tr><td></td><td>DiffusionNFT</td><td>Pick-a-Pic + PickScore</td><td>Official training</td><td>Parti-Prompts + HPSv2</td></tr><tr><td></td><td>GRPO-Guard</td><td>Pick-a-Pic + PickScore</td><td>Official LoRA</td><td>Parti-Prompts + HPSv2</td></tr></table>

QUPO instead considers a scalar reward and is derived from the Best-of-N quality-compute frontier. It optimizes a weighted family of Best-of-k utilities across deployment budgets, with the goal of preserving the reward upper tail that determines test-time scaling. Thus, Multi-Axis Max@K uses set-level credit assignment to improve which modes are represented within a sample set, whereas QUPO uses it to improve how scalar reward quality scales with inference compute. Our primary contribution further lies in identifying and analyzing scaling-transfer failure across visual RL methods, tasks, and model families.

## C Experimental and Implementation Details

## C.1 Diagnostic Experiments

Compositional Generation. We evaluated SD3.5-M models post-trained on GenEval [11] prompts, using a GenEval detector-derived reward, following established protocol from FlowGRPO [20]. The GenEval training prompts are open-sourced in the FlowGRPO codebase.

We evaluated these models on GenEval2 [15], which consists of compositional text-to-image prompts with broader coverage of visual primitives and higher degrees of compositionality than GenEval, making it substantially more challenging for modern generators.

We use the official Soft-TIFA evaluator [15] with Qwen3-VL-8B-Instruct [2] as the VQA scorer. Each GenEval2 prompt is associated with a set of atomic visual questions targeting its constituent objects, attributes, relations, and counts. For each generated image, Qwen3-VL is queried with these questions, and Soft-TIFA uses the model probability assigned to the correct answer as a soft per-atom score. We use the standard Soft-TIFA GM aggregation, taking the geometric mean of the atom-level scores to obtain the reward for each image.

Text Rendering. We evaluated SD3.5-M models post-trained using the open-sourced OCR training prompts and reward from the official FlowGRPO codebase: PaddleOCR is applied to each generated image, and all recognized text is concatenated, lowercased, and stripped of whitespace. Let y denote the target text extracted from the prompt and Î the recognized text. The reward is based on normalized Levenshtein similarity

$$
r _ { \mathrm { O C R } } = 1 - \frac { \operatorname* { m i n } \{ d _ { \mathrm { L e v } } ( \hat { y } , y ) , | y | \} } { | y | } ,\tag{45}
$$

with reward 1 assigned whenever y appears as a substring of ${ \hat { y } } .$ This yields a score in [0, 1], where 1 corresponds to perfect text rendering

For evaluation on TextInVision [9], we use the same normalized-Levenshtein OCR scoring procedure on the benchmark's target text. We specifically evaluated on a subset of 800 prompts from TextInVision, which provides a substantially harder held-out OCR benchmark than the training set. Its target text averages 6.1 words and 36.4 characters—roughly twice the training averages—and includes long sentences, rare or synthetic words, numbers, and punctuation. Moreover, 90% of the prompts place text within complex scenes or constrained surfaces, and all targets and scene contexts were filtered against the training data to prevent exact or near-duplicate leakage

(a) SD3.5-M training.
<table><tr><td>Setting</td><td>FlowGRPO†</td><td>QUPO</td></tr><tr><td>Base / reward</td><td>SD3.5-M / GenEval</td><td>Same</td></tr><tr><td>Resolution / GPUs</td><td> $5 1 2 ^ { 2 ^ { \prime } } / 3 2$ </td><td>Same</td></tr><tr><td>Rollout steps / CFG</td><td> $1 0 ~ / ~ 4 . 5$ </td><td>Same</td></tr><tr><td>Group size G</td><td>24</td><td>Same</td></tr><tr><td>LoRA rank / α</td><td>32  / 64</td><td>Same</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>Same</td></tr><tr><td>AdamW  $\left( \beta _ { 1 } , \beta _ { 2 } \right) / \mathrm { W D }$ </td><td> $\left( 0 . 9 , 0 . 9 9 9 \right) / 1 0 ^ { - 4 }$ </td><td>Same</td></tr><tr><td>Per-GPU batch / grad. accum.</td><td> $9 \ / \ 2 ^ { ' }$ </td><td>Same</td></tr><tr><td>Updates  / epoch</td><td>2</td><td>Same</td></tr><tr><td># Epochs</td><td>3350</td><td>Same</td></tr><tr><td>PPO clip / KL</td><td> $1 0 ^ { - 4 } ~ / ~ 0 . 0 4$ </td><td>Same</td></tr><tr><td>Max grad. norm</td><td>1.0</td><td>Same</td></tr><tr><td>Trained timesteps</td><td>9  / 10</td><td>Same</td></tr><tr><td>Timestep fraction / noise</td><td>0.99 / 0.7</td><td>Same</td></tr><tr><td>Advantage clamp</td><td>[-5,5]</td><td>Same</td></tr><tr><td>Advantage T/α</td><td>Group-relative</td><td>QUPO  $1 6 ^ { - } / - 0 . 5$ </td></tr></table>

(b) BAGEL training.
<table><tr><td>Setting</td><td>UniGRPO</td><td>QUPO</td></tr><tr><td>Base / reward</td><td>BAGEL GenEval</td><td>Same</td></tr><tr><td>Resolution / GPUs</td><td>5122 64</td><td>Same</td></tr><tr><td>Rollout steps / CFG</td><td>25 /1.0</td><td>Same</td></tr><tr><td>Group size G</td><td>24</td><td>Same</td></tr><tr><td>Images / reasoning chain</td><td>1</td><td>Same</td></tr><tr><td>AR sampling</td><td> ${ T } { = } 1 . 0 , \ p { = } 0 . 9 , \ k { = } 1 0 2 4$ </td><td>Same</td></tr><tr><td>Max reasoning tokens</td><td>1024</td><td>Same</td></tr><tr><td>Fine-tuning</td><td> $\mathrm { F u l l \ ( n o \ L o R A ) }$ </td><td>Same</td></tr><tr><td>LR (reason / image)</td><td> $1 0 ^ { - 6 } ~ / ~ 3 { \times } 1 0 ^ { - 5 }$ </td><td>Same</td></tr><tr><td>AdamW  $( \beta _ { 1 } , \beta _ { 2 } ) ~ / \mathrm { \Delta W D }$ </td><td> $\left( 0 . 9 , 0 . 9 9 9 \right) / \mathrm { ~ 0 ~ }$ </td><td>Same</td></tr><tr><td>Batch / grad. accum.</td><td> $^ { \prime } 1 \ / \ 1 2 ^ { \prime }$ </td><td>Same</td></tr><tr><td>Effective batch / updates</td><td> $6 7 2 ~ / ~ 2$ </td><td>Same</td></tr><tr><td>Inner epochs</td><td>1</td><td>Same</td></tr><tr><td># Epochs</td><td> $^ \mathrm { 1 5 0 }$ </td><td>Same</td></tr><tr><td>PPO clip (text / image)</td><td> $1 0 ^ { - 2 } \ / \ 1 0 ^ { - 6 }$ </td><td>Same</td></tr><tr><td>KL (text)</td><td>0</td><td>Same</td></tr><tr><td>Velocity-MSE / ratio norm.</td><td> $1 . 5 { \times } 1 0 ^ { - 5 } ~ / ~ \mathrm { o n }$ </td><td>Same</td></tr><tr><td>Max grad. norm</td><td> $_ { 1 . 0 }$ </td><td>Same</td></tr><tr><td>Trained timesteps</td><td> $3 ~ / ~ 2 5$ </td><td>Same</td></tr><tr><td>Timestep fraction /noise</td><td> $[ 0 , 0 . 2 ] / 0 . 8$ </td><td>Same</td></tr><tr><td>Advantage clamp</td><td> $[ - 5 , 5 ]$ </td><td>Same</td></tr><tr><td>Advantage</td><td>Group-relative</td><td>QUPO</td></tr><tr><td> $T \ / \ \alpha$ </td><td></td><td> $1 6 ~ \AA / - 0 . 8 7 5$ </td></tr></table>

Table 4 Post-training hyperparameters for SD3.5-M and BAGEL. Within each backbone, QUPO uses the same optimization pipeline as the corresponding baseline and changes only the rollout advantage. (†: FlowGRPO trained at matched budget, not the official checkpoint.)

Preference Alignment. We evaluated SD3.5-M models post-trained on the Pick-a-Pic v2 [17] prompt set, using PickScore as the training reward. Pick-a-Pic contains user-written text-to-image prompts collected through a preference-data collection platform, together with human comparisons between generated images.

To evaluate transfer beyond both the training prompts and reward model, we test on the held-out Parti Prompts benchmark and score each generated image with HPSv2 [40], a learned human-preference metric conditioned on the corresponding text prompt. We then compute Best-of-N performance using the HPSv2 scores across the 128 generations for each prompt.

## C.2 Post-Training Hyperparameters

We present the detailed hyperparameter configuration in Tables 4a and 4b. As shown, all applicable hyperparameters are matched between QUPO and the respective baseline, differing only in the per-sample advantage computation. FlowGRPO and UniGRPO use Equation 6 to compute the per-sample advantage, while QUPO uses Equation 13.

## C.3 Evaluation Hyperparameters

We generated 128 images per prompt for SD3.5-M evaluation and 64 images per prompt for BAGEL, using matched seeds across methods to ensure fair comparison. The detailed inference hyperparameters are contained in Table 5. The hyperparameters used for SD3.5-M evaluation are shared across both the diagnostic experiments and the QUPO experiments

## D Additional Ablations

## D.1 Effect of the Optimization Horizon T

We study the effect of the QUPO horizon T, which controls the largest Best-of-k utility explicitly included in the training objective. We fix $\alpha = - 0 . 5$ and train otherwise identical SD3.5-M models to the same optimization step. Results are shown in Table 6, alongside the pretrained SD3.5-M base model for reference.

Table 5 Inference hyperparameters. All Best-of-N evaluations use the same inference configuration within each backbone.
<table><tr><td>Setting</td><td>SD3.5-M</td><td>BAGEL</td></tr><tr><td>Resolution</td><td>512 × 512</td><td>512 × 512</td></tr><tr><td>Generation steps</td><td>40</td><td>50</td></tr><tr><td>CFG</td><td>4.5†</td><td>4.0 / 1.5 (text / image)</td></tr><tr><td>Flow shift</td><td></td><td>3.0</td></tr><tr><td>AR temperature</td><td></td><td>0.3</td></tr><tr><td>Max reasoning tokens</td><td></td><td>1024</td></tr><tr><td># Samples per prompt</td><td>128</td><td>64</td></tr></table>

† DiffusionNFT and AWM are evaluated with CFG = 1.0 following their corresponding inference settings; FlowGRPO, GRPO-Guard, and QUPO use $\mathrm { C F G } = 4 . 5 .$

Table 6Effect of the QUPO horizon T. All post-trained models use $\alpha = - 0 . 5$ and are evaluated at training step 3350. While $T = 1$ improves single-sample quality, it degrades large-N performance relative to the base model. Moderate horizons improve the full held-out Best-of-N frontier, with performance peaking around $T = 1 6$
<table><tr><td>T</td><td>Step</td><td>GenEval |</td><td>Bo1</td><td>Bo4</td><td>Bo8</td><td>Bo16</td><td>Bo32</td><td>Bo64</td><td>Bo96</td><td>Bo128</td></tr><tr><td>Base</td><td></td><td>0.63</td><td>0.1804</td><td>0.3365</td><td>0.4193</td><td>0.4978</td><td>0.5679</td><td>0.6295</td><td>0.6625</td><td>0.6850</td></tr><tr><td>1</td><td>3350</td><td>0.79</td><td>0.2068</td><td>0.3551</td><td>0.4300</td><td>0.5010</td><td>0.5661</td><td>0.6237</td><td>0.6536</td><td>0.6729</td></tr><tr><td>8</td><td>3350</td><td>0.87</td><td>0.2222</td><td>0.3751</td><td>0.4519</td><td>0.5239</td><td>0.5889</td><td>0.6465</td><td>0.6768</td><td>0.6965</td></tr><tr><td>16</td><td>3350</td><td>0.88</td><td>0.2287</td><td>0.3841</td><td>0.4618</td><td>0.5333</td><td>0.5962</td><td>0.6508</td><td>0.6794</td><td>0.7000</td></tr><tr><td>23</td><td>3350</td><td>0.87</td><td>0.2256</td><td>0.3765</td><td>0.4523</td><td>0.5243</td><td>0.5899</td><td>0.6476</td><td>0.6775</td><td>0.6973</td></tr></table>

Increasing T from 1 to 8 and 16 consistently improves the held-out quality-compute frontier, while extending the horizon further to $T = 2 3$ provides no additional benefit. Notably, optimizing only through $T = 1 6$ improves performance even at much larger deployment budgets such as Best-of-64 and Best-of-128, suggesting that directly including very large k is not necessary to obtain stronger upper-tail behavior.

Crucially, the $T = 1$ setting recovers expected-reward optimization and still exhibits scaling-transfer failure. For $k = 1$ , the QUPO advantage reduces to

$$
a _ { i } ^ { ( 1 ) } = r _ { i } - \frac { 1 } { G - 1 } \sum _ { j \neq i } r _ { j } = \frac { G } { G - 1 } ( r _ { i } - \bar { r } ) ,\tag{46}
$$

which is proportional to the mean-centered reward used by FlowGRPO, but without its division by the withingroup reward standard deviation. Thus, both objectives are aligned with Best-of-1, while FlowGRPO generally applies a more aggressively scaled update through Z-score normalization. Consistent with this difference, $T = 1$ improves GenEval and Best-of-1 less than FlowGRPO, but nevertheless exhibits the same qualitative scaling failure: its Best-of-1 improves over the pretrained base (0.2068 vs. 0.1804), while its improvement decays with N and reverses by Best-of-128 (0.6729 vs. 0.6850). In contrast, introducing multi-sample utility with $T = 8$ or $T = 1 6$ preserves a positive advantage throughout the measured frontier, reaching Best-of-128 scores of 0.6965 and 0.7000, respectively. Together, these results suggest that frontier reversal is not specific to FlowGRPO's Z-score advantage or aggressive reward scaling, but arises more fundamentally when post-training is aligned only with expected single-sample reward; explicitly incorporating Best-of-k utilities mitigates this failure.

## D.2 Effect of the Weighting Parameter α

We next vary the QUPO weighting parameter α while fixing the optimization horizon to $T = 1 6$ . Recall that $q _ { k } = k ^ { \alpha }$ , so increasing α places relatively greater weight on larger Best-of-k utilities. Table 7 reports the resulting GenEval and held-out GenEval2 performance.

The results show that the harmonic choice $\alpha = - 1$ is not uniquely optimal for visual generation. Increasing α from -2 toward -0.5 generally improves both in-distribution GenEval performance and most of the held-out quality-compute frontier, with $\alpha = - 0 . 5$ performing best from Best-of-1 through Best-of-128.

This ablation supports treating the Best-of-k weighting as a controllable design choice rather than fixing the

Table 7 Effect of the QUPO weighting parameter $\alpha .$ All runs use $T = 1 6$ . The harmonic weighting $\alpha = - 1$ corresponding to TailRL, is competitive but is not uniformly optimal; $\alpha = - 0 . 5$ achieves the strongest performance across most sampling budgets
<table><tr><td>α</td><td>GenEval</td><td>Bol</td><td>Bo4</td><td>Bo8</td><td>Bo16</td><td>Bo32</td><td>Bo64</td><td>Bo96</td><td>Bo128</td></tr><tr><td>-2.0</td><td>0.79</td><td>0.2119</td><td>0.3661</td><td>0.4434</td><td>0.5160</td><td>0.5817</td><td>0.6401</td><td>0.6710</td><td>0.6914</td></tr><tr><td>-1.0</td><td>0.85</td><td>0.2194</td><td>0.3676</td><td>0.4427</td><td>0.5153</td><td>0.5831</td><td>0.6452</td><td>0.6787</td><td>0.6920</td></tr><tr><td>-0.75</td><td>0.85</td><td>0.2218</td><td>0.3734</td><td>0.4497</td><td>0.5216</td><td>0.5862</td><td>0.6427</td><td>0.6721</td><td>0.6911</td></tr><tr><td>-0.5</td><td>0.88</td><td>0.2287</td><td>0.3841</td><td>0.4618</td><td>0.5333</td><td>0.5962</td><td>0.6508</td><td>0.6794</td><td>0.7000</td></tr></table>

Table 8 Effect of normalized advantage coefficients. All runs use $T = 1 6$ with normalized weights satisfying $\begin{array} { r } { \frac { 1 } { T } \sum _ { k = 1 } ^ { T } k q _ { k } = 1 } \end{array}$ . Increasing α shifts performance from smaller toward larger inference budgets, with $\alpha = 0$ achieving the strongest Best-of-8 through Best-of-128 results
<table><tr><td>α</td><td>Bol</td><td>Bo4</td><td>Bo8</td><td>Bo16</td><td>Bo32</td><td>Bo64</td><td>Bo96</td><td>Bo128</td></tr><tr><td>-1.0</td><td>0.2194</td><td>0.3676</td><td>0.4427</td><td>0.5153</td><td>0.5831</td><td>0.6452</td><td>0.6787</td><td>0.6920</td></tr><tr><td>-0.5</td><td>0.2087</td><td>0.3653</td><td>0.4446</td><td>0.5196</td><td>0.5873</td><td>0.6457</td><td>0.6787</td><td>0.7003</td></tr><tr><td>0.0</td><td>0.2048</td><td>0.3669</td><td>0.4489</td><td>0.5246</td><td>0.5916</td><td>0.6496</td><td>0.6800</td><td>0.7012</td></tr></table>

harmonic weighting a priori. In particular, moderately stronger emphasis on larger k can improve broad scaling performance, while the optimal weighting may depend on the range of inference budgets of interest.

## D.3 Effect of Normalized Advantage Coefficients

In the preceding ablation, varying α changes not only the relative weighting of different Best-of-k utilities, but also the overall magnitude of the effective advantage coefficients $k q _ { k }$ . To disentangle these effects, we additionally study a normalized weighting scheme

$$
q _ { k } = c _ { \alpha } k ^ { \alpha } , \qquad c _ { \alpha } = { \frac { T } { \sum _ { j = 1 } ^ { T } j ^ { \alpha + 1 } } } ,\tag{47}
$$

which ensures that the effective coefficients $w _ { k } = k q _ { k }$ satisfy $\begin{array} { r } { \frac { 1 } { T } \sum _ { k = 1 } ^ { T } w _ { k } = 1 } \end{array}$ . This keeps the total effective coefficient weight fixed across choices of α, while preserving their relative emphasis on different sampling budgets.

Table 8 reports the results on held-out GenEval2 with $T = 1 6$ fixed. Under coefficient normalization, increasing α from -1 to 0 decreases Best-of-1 performance from 0.2194 to 0.2048, while improving Best-of-128 from 0.6920 to 0.7012. More generally, larger α consistently improves performance from Best-of-8 through Best-of-128, demonstrating a tradeoff between single-sample quality and performance at larger inference budgets.

These results further support the interpretation of QUPO as a controllable weighting of the quality-compute frontier. Even when the total effective coefficient weight is held fixed, placing greater relative emphasis on larger Best-of-k utilities shifts the learned frontier toward higher sampling budgets. While this normalization does not directly equalize the variance of the resulting per-sample advantages, it removes the change in total coefficient weight as a confounding factor in the α ablation.

## D.4 Scaling-Transfer Robustness Across Evaluators and Selectors

Our primary GenEval2 compositional evaluation uses oracle Best-of-N selection, where candidates are selected using the same Qwen3 Soft-TIFA score used for evaluation. This allows us to characterize how RL post-training reshapes the per-prompt reward distribution without confounding effects from a mismatch between selection and evaluation rewards. To assess the robustness of our findings to the choice of reward model and selection procedure, we consider two additional GenEval2 settings: (i) oracle Best-of-N using InternVL3.5-8B |39] as the Soft-TIFA scorer, and (ii) non-oracle Best-of-N, where Qwen3.5-9B [26] serves as the Soft-TIFA selector and InternVL3.5-8B as the evaluator.

As shown in Figure 6, scaling-transfer reversal persists under both settings. Although FlowGRPO and GRPO Guard improve Best-of-1 performance relative to the base SD3.5-M model, the advantage diminishes as N

![](images/30f46d793be08fea6ed577bb33a533c9b6818d45279dfcf8819295afa7cc50ba.jpg)  
(a) Oracle Best-of-N evaluation on GenEval2 using InternVL3.5 Soft-TIFA scores.

![](images/9db0f8931eae23d403391752872489ec3a7be0d9af112b1695031b22e157134c.jpg)  
(b) Non-oracle Best-of-N evaluation using Qwen3.5 as the scorer and InternVL3.5 as the evaluator

Figure 6 Additional Best-of-N evaluation on GenEval2 with different VLMs. As shown, the scaling transfer reversal occurs for FlowGRPO and GRPO Guard, while QUPO mitigates this failure.

Table 9 Best-of-N rewards on GenEval2 and GenEval scores for the base SD3.5-M model, single Best-of-16 optimization, QUPO, FlowGRPO, and GRPO Guard. †: FlowGRPO and GRPO Guard evaluation uses earlier checkpoints that match QUPO in GenEval score.
<table><tr><td>Method</td><td>GenEval</td><td>Bo1</td><td>Bo4</td><td>Bo8</td><td>Bo16</td><td>Bo32</td><td>Bo64</td><td>Bo96</td><td>Bo128</td></tr><tr><td>SD3.5-M</td><td>0.63</td><td>0.1804</td><td>0.3365</td><td>0.4193</td><td>0.4978</td><td>0.5679</td><td>0.6295</td><td>0.6625</td><td>0.6850</td></tr><tr><td>Single Bo16</td><td>0.75</td><td>0.1908</td><td>0.3518</td><td>0.4353</td><td>0.5131</td><td>0.5819</td><td>0.6417</td><td>0.6733</td><td>0.6946</td></tr><tr><td>QUPO</td><td>0.88</td><td>0.2287</td><td>0.3841</td><td>0.4618</td><td>0.5333</td><td>0.5962</td><td>0.6508</td><td>0.6794</td><td>0.7000</td></tr><tr><td>FlowGRPO†</td><td>0.88</td><td>0.2039</td><td>0.3256</td><td>0.3897</td><td>0.4550</td><td>0.5187</td><td>0.5783</td><td>0.6106</td><td>0.6322</td></tr><tr><td>GRPO Guard†</td><td>0.88</td><td>0.2013</td><td>0.3150</td><td>0.3740</td><td>0.4329</td><td>0.4901</td><td>0.5456</td><td>0.5773</td><td>0.5991</td></tr></table>

increases and eventually becomes negative. This behavior occurs both when InternVL3.5 is used for selection and evaluation and when Qwen3.5 selects candidates to be evaluated by InternVL3.5. On the other hand, QUPO largely mitigates the scaling transfer failure in both cases, remaining above the RL models and base SD3.5-M as N increases towards 128. Together with our primary Qwen3-VL-based evaluation, these results demonstrate that the observed scaling-transfer failure is not specific to the choice of VLM evaluator or to using identical rewards for Best-of-N selection and evaluation,

## D.5 Comparison with a Single Best-of-k Objective

To assess the benefit of mixing inference budgets, we compare QUPO with a single Best-of-16 objective under matched training conditions. For this baseline, we retain only the Best-of-16 component and set its coefficient to the sum of all QUPO advantage coefficients, yielding $\begin{array} { r } { \dot { A _ { i } } = \left( \sum _ { k = 1 } ^ { 1 6 } k q _ { k } \right) a _ { i } ^ { ( 1 6 ) } \approx 4 4 . 4 7 a _ { i } ^ { ( 1 6 ) } } \end{array}$ . As shown in Table 9, the single-budget baseline remains above the base model through Best-of-128, mitigating frontier reversal. QUPO achieves higher rewards at every evaluated budget, with the largest gains at small N, supporting budget mixtures as a way to improve low-budget quality while retaining strong large-budget performance.

## D.6 Does Early Stopping Benefit RL Baselines?

We next study whether early stopping improves the test-time scaling of RL baselines such as FlowGRPO and GRPO-Guard. We evaluate earlier checkpoints of both methods, selected to match QUPO's in-domain GenEval score of 0.88. As shown in Table 9, early stopping does not prevent frontier reversal: both baselines outperform SD3.5-M at Best-of-1 on GenEval2, but fall below it already at N = 4. In contrast, QUPO maintains a positive estimated advantage through Best-of-128. These results show that weakened test-time scaling persists at matched in-domain performance and cannot be attributed simply to the baselines achieving higher GenEval scores. This supports our quantile-based interpretation: optimizing single-sample reward can improve typical generations without preserving the upper-tail quality exploited by Best-of-N sampling.

Table 10 Within-prompt perceptual diversity on GenEval2. Average per-prompt pairwise LPIPS over all generated images.
<table><tr><td>Model Mean pairwise LPIPS ↑</td></tr><tr><td>SD3.5-M 0.6977</td></tr><tr><td>FlowGRPO (official) 0.5763 0.6451</td></tr><tr><td>QUPO</td></tr><tr><td>BAGEL 0.6307 UniGRPO 0.5481</td></tr><tr><td>QUPO 0.5825</td></tr><tr><td></td></tr></table>

## E Diversity and Qualitative Results

## E.1 Perceptual Diversity

We additionally examine whether weakened scaling transfer can be explained by a loss of perceptual diversity. Using the same G = 128 GenEval2 generations per prompt for SD3.5-M and G = 64 for BAGEL models, we compute the average pairwise LPIPS distance [46]

$$
D ( c ) = \frac { 2 } { G ( G - 1 ) } \sum _ { 1 \leq i < j \leq G } \mathrm { L P I P S } ( \mathbf { x } _ { i } , \mathbf { x } _ { j } ) ,\tag{48}
$$

and report $D ( c )$ averaged across prompts. Higher values indicate greater within-prompt perceptual diversity.

FlowGRPO substantially reduces perceptual diversity relative to the base model, while QUPO preserves considerably more of it. However, diversity alone does not explain Best-of-N scaling: QUPO remains less diverse than the base model despite achieving a stronger large-N reward frontier. Thus, diversity collapse can accompany upper-tail degradation, but perceptual diversity is not the quantity that directly determines scaling transfer. Our quantile analysis provides the more precise explanation: large-N performance depends specifically on preserving rare high-reward outcomes, rather than diversity for its own sake.

## E.2 Additional Qualitative Comparisons

We provide further qualitative examples from each model, across both the GenEval2 and ConceptMix benchmarks. As shown in Figures 7 and 8, QUPO is able to more consistently generate the rare high-reward sample that benefits test-time scaling compared to its FlowGRPO and UniGRPO baselines. Figure 9 compares results of the base SD3.5-M model with the expected-reward RL baselines. As seen, SD3.5-M generates images that can capture the rare high-reward results more consistently, resulting in stronger scaling performance.

![](images/bd518e194f40e051c2558a0d2acc7c52048ef147a4ad16e7843b69da8f668bd5.jpg)  
(a) BAGEL on GenEval2.

![](images/d9052c88c39e30e0e53e3929887c73bff9c1ca63297d28e74b04f9dda0e89c70.jpg)  
(b) SD3.5-M on ConceptMix.  
Figure 7 Qualitative examples of improved scaling transfer with QUPo. Left: Four matched-seed generations from QUPO, base BAGEL, and UniGRPO. QUPO more consistently satisfies the requested composition, reflecting its strong scaling transfer. Right: Four matched-seed generations from QUPO, base SD3.5-M, and FlowGRPO on a held-out ConceptMix prompt. QUPO better preserves the requested composition and style; FlowGRPO often captures the three animals but misses the requested graffiti style.

![](images/da5ad1ea3c93b24c62af903260ea03079afda536d456a37a97fba1f079687c98.jpg)  
(a) BAGEL on GenEval2.

![](images/bbbf2868e398b9ce7602c1ee127c1cf9264e05ac64faf62a4cb1e395f8790a7e.jpg)  
(b) SD3.5-M on GenEval2.  
Figure 8 Qualitative examples of improved scaling transfer with QUPo. Left: Four matched-seed generations from QUPO, base BAGEL, and UniGRPO. QUPO more consistently satisfies the requested composition, reflecting its stronger scaling transfer. Right: Four matched-seed generations from QUPO, base SD3.5-M, and FlowGRPO on a held-out GenEval2 prompt.

![](images/48dd78608db39292aa04540ccfc057ce1f19c66035ad0f965c7f41efc55fd77d.jpg)  
(a) SD3.5-M GenEval2 comparison.

![](images/45a52e0a21d8db9a9b2653b1b6a3ef147585d15a39fce4a650104448c90274a3.jpg)  
(b) Additional SD3.5-M GenEval2 comparison.  
Figure 9 Multiple generations for the same held-out GenEval2 prompts. Each row shows four independent samples from the base SD3.5-M model, FlowGRPO, DiffusionNFT, GRPO Guard, and AWM under matched generation seeds. In both examples, the base SD3.5-M model produces the rarer correct generations for these difficult prompts, illustrating the upper-tail failures that contribute to weakened Best-of-N scaling after standard RL post-training