# Confident First, Reflective Second: Reinforcement Learning of Confidence-Preserving Reflection for Video Understanding

Zihao Lin<sup>1</sup> Jingyuan Liu<sup>2</sup> Yucheng Zhang<sup>2</sup> Yuguang Yao<sup>3</sup> Lifu Huang<sup>1</sup> <sup>1</sup>University of California, Davis <sup>2</sup>Independent Researcher <sup>3</sup>Intuit

## Abstract

Video models answer from sampled frames and may miss the decisive moment. A common remedy is to commit to an answer, reexamine the video, and confirm or revise it, a loop now mainly learned with reinforcement learning (RL). We find that RL erodes the initial answer, a phenomenon we term confidence erosion, where confidence means the label-verified reliability of the initial answer rather than selfreported certainty: because errors can be repaired later, a correct initial answer earns less reward, and the initial answer stalls even as the final answer’s gain over it rises. We propose the principle confident first, reflective second: the initial answer must stay reliable on its own, above a floor set by a reference policy, and reflection may only add to it. Our method, COFIRE, implements this principle with one mechanism for each erosion channel: reward incentives, parameter drift, and gradient interference. COFIRE reaches 67.0% mean accuracy on Video-MME, MLVU, and LongVideoBench, 1.7 points above the strongest compared 7B method, and improves initial-answer and final accuracy over outcome-reward RL by 4.1 and 3.7 points on average: a stronger initial answer carries through to a stronger final one.

## 1 Introduction

Understanding long-form videos requires reasoning over visual evidence that is often incomplete, sparse, or difficult to localize. Under compute and context limits, multimodal large language models process only a subset of frames and may overlook decisive moments, miss fine-grained details, or misinterpret temporal relationships (Fu et al., 2025; Zhou et al., 2025; Wu et al., 2024). Recent video agents therefore give an initial answer, retrieve or re-examine relevant segments, and verify or revise the answer based on new evidence (Wang et al., 2024; Pan et al., 2026; Madaan et al., 2023; Chen et al., 2026). Reinforcement learning (RL) has emerged as a promising way to learn when to seek more evidence and how to use it (Kumar et al., 2025; Wang et al., 2025a; Chen et al., 2026). Unlike reflection within a single reasoning trace (Guo et al., 2025; Feng et al., 2025), an explicit answerand-revision process exposes both the initial and final answers, so their correctness and the outcome of each revision can be evaluated separately.

![](images/86f1d70912b1f2abe4fd687f8117e626c3b944830aaf08977854963dff9c0b95.jpg)  
Figure 1: Reflection gain is not reflective progress. Video-MME policies by initial-answer accuracy and reflection gain ∆; dashed lines are iso-lines of final accuracy. From the same SFT policy, Outcome RL moves up (more gain, same initial answer), while COFIRE moves right to a higher final accuracy. Gray dots: configurations from Table 8.

Existing reflective objectives mainly reward final-answer correctness or improvement through revision, leaving open how optimizing reflection affects initial-answer accuracy. Answering and evidence-conditioned revision are distinct capabilities. The former reasons over available observations, whereas the latter must identify missing or conflicting evidence and judge whether to revise. Reflection can repair initial mistakes but is imperfect: evidence may be unavailable, retrieval may miss relevant moments, and new observations can be misinterpreted or even overturn a correct initial answer. A robust reflective agent should therefore learn not only to correct mistakes but also to answer accurately before correction is needed.

We identify an imbalance between initial-answer learning and correction in reflective RL. As illustrated in Figure 1, outcome-based RL (Outcome RL) raises the reflection gain on Video-MME from 0.3 to +0.5 points while initialanswer accuracy stays nearly unchanged (60.4% 60.3%), and similar patterns emerge on MLVU and LongVideoBench. A larger reflection gain thus need not indicate a stronger system and may instead reflect greater dependence on correction. We refer to this imbalance, particularly when accompanied by degradation of the initial answer, as confidence erosion, where confidence denotes the labelverified reliability of the initial answer rather than self-reported certainty.

This imbalance arises from two challenges in optimizing a shared reflective policy. First, outcomebased rewards can obscure the contribution of the initial answer (reward incentives), since an incorrect initial answer that is later repaired can receive the same credit as a correct one that remains unchanged. As we show in Section 3, under fixed repair and break rates, successful correction reduces the marginal value of initial-answer accuracy, and correction bonuses can further favor repaired trajectories. Stage-specific credit can separate the two (Kumar et al., 2025; Chen et al., 2026), but reward attribution alone does not resolve the problem. Second, answering and reflecting share parameters, so updates that improve retrieval or revision may alter reliable initial-answer decisions (parameter drift), and reflective gradients may conflict with those improving the initial answer (gradient interference).

We therefore propose COFIRE (Confident First, Reflective Second), which protects initial-answer accuracy while encouraging beneficial revision, rather than treating reflection gain as an end in itself. We formalize this principle as maximizing initial-answer reliability and reflective utility subject to a confidencefloor relative to a frozen reference policy, and approximately optimize it with three mechanisms. (1) Initial-answer credit rewards the initial answer for its own correctness, separately rewards corrections and penalizes destructive revisions, and raises its weight through an adaptive multiplier when initial-answer accuracy falls below the floor. (2) Verified-decision anchoring consolidates reference decisions verified as correct through label-guided supervision and decisionlocal regularization, stabilizing reliable predictions while allowing incorrect ones to improve. (3) Asymmetric gradient protection prioritizes the protected objective whenever the initial-answer and reflective

gradients conflict.

We evaluate COFIRE with Qwen2.5-VL-7B (Bai et al., 2025) on Video-MME (Fu et al., 2025), MLVU (Zhou et al., 2025), and LongVideoBench (Wu et al., 2024). Under identical initialization and inference, COFIRE improves initial-answer and final accuracy over Outcome RL by 4.1 and 3.7 points on average, so its initial-answer gains carry through to the final answer, and reflection still helps on all three benchmarks. It also reaches 67.0% average accuracy, 1.7 points above the strongest compared 7B method. Ablations show that the three mechanisms contribute complementarily, with notable gains on long videos. Reflective learning should thus be judged not only by how well it corrects mistakes but also by whether it improves the answers that correction starts from.

Our main contributions are threefold:

• Understanding the limitations of outcomedriven reflective learning. We show that gains from reflective RL can concentrate in revision while initial-answer accuracy stays largely unchanged, and analyze how reward attribution and shared parameters cause this imbalance.

• A unified framework for reliable reflective optimization. We introduce COFIRE, which combines three mechanisms to improve both the initial answer and its correction.

• Empirical validation on video understanding. Experiments on three video benchmarks show gains in both initial-answer and final accuracy, supported by ablations and analyses.

## 2 Related Work

Answering and re-examining video. Outcomereward RL with group-relative advantages is the standard recipe for eliciting reasoning in LLMs (Shao et al., 2024; Guo et al., 2025) and multimodal models (Huang et al., 2025). For video, Video-R1 rewards temporal reasoning (Feng et al., 2025) and VideoChat-R1 reinforces spatio-temporal perception (Li et al., 2025), but both answer in a single pass. A second line lets the model look again: training-free agents seek question-relevant evidence (Wang et al., 2024, 2025b; Ye et al., 2025), TimeSearch-R interleaves temporal search with reasoning under a self-verification reward (Pan et al., 2026), and Reflect-R1 arbitrates an intuitive answer against retrieved evidence (Chen et al., 2026). Most trained variants reward only the final answer, which cannot distinguish an initially correct answer from a repaired one and reduces the initial answer to a draft. Stage-decoupled credit (Reflect-R1) removes this discount but still trains the initial answer only as an input to revision. We instead make initial-answer accuracy an explicit, protected training target.

Learning to revise. Self-Refine and Reflexion improve outputs with generated feedback (Madaan et al., 2023; Shinn et al., 2023), yet intrinsic selfcorrection fails without reliable feedback (Huang et al., 2024). SCoRe and $\mathbb { S } ^ { 2 } \mathbb { R }$ train self-correction with RL (Kumar et al., 2025; Ma et al., 2025), and SCoRe adds a first-attempt KL penalty and a progress reward. $\mathrm { R ^ { 3 } V , }$ VL-Rethinker, SRPO, and ReCoT extend reflection to vision-language models (Cheng et al., 2025; Wang et al., 2025a; Wan et al., 2025; Qu et al., 2025), and Reflect-R1 additionally penalizes overturning correct answers (Chen et al., 2026). These methods optimize whether the revision improves or preserves an answer, whereas we show that this objective erodes the initial answer, an effect that neither per-rollout revision penalties nor fixed first-attempt regularizers correct. $\mathsf { A p - }$ pendix A relates COFIRE to constrained RL, continual learning, and gradient surgery.

## 3 Confidence Erosion in Learning to Re-examine

## 3.1 Answering and Re-examining a Video

Let $x = ( V , q )$ be a video-question input with answer options and correct option $y ^ { \star }$ . As illustrated in Figure 2, a policy $\pi _ { \theta }$ with shared parameters θ runs three stages. The initial-answer stage $S _ { \mathrm { p r e } }$ observes a uniform 2 fps preview of the whole video, whose fixed visual budget renders long videos at low resolution. It therefore requests afirst look of up to eight higher-resolution frames, selected for query relevance and temporal diversity, and commits to an initial answer y<sub>0</sub>.

The gate $S _ { \mathrm { g a t e } }$ inspects the first-look frames, reports a confidence score, and selects a mode m: TRUST (return $y _ { 0 } )$ , REVIEW (re-examine the first look), SEEK (retrieve once), or MULTI-HOP (up to three retrieval rounds), together with retrieval queries and a frame count when needed. A frozen SigLIP encoder (Zhai et al., 2023) ranks frames from a 2 fps cache of the whole video by query similarity and option-discriminating scores, never returning a frame twice. The post-answer stage $S _ { \mathrm { r e v } }$ then takes the initial-answer reasoning, the firstlook and retrieved frames, and a frame-derived option belief (a distribution over options from SigLIP frame–option similarity), and produces a candidate answer $y _ { f }$ , which may confirm $y _ { 0 }$ . The preview is dropped after $S _ { \mathrm { p r e } }$ to bound the context.

At inference, the returned answer is $\hat { y } = y _ { 0 }$ under TRUST and $\hat { y } = y _ { f }$ otherwise; during training, $y _ { f }$ is generated in every mode. We refer to $S _ { \mathrm { g a t e } }$ and $S _ { \mathrm { r e v } }$ together as reflection. Because $y _ { 0 }$ is explicitly committed, the task label alone determines whether the initial answer is correct and whether reflection repairs or preserves it, without step-level annotation.

## 3.2 Why Reflective Training Erodes Confidence

Let $r _ { 0 }$ and $r _ { f }$ indicate whether $y _ { 0 }$ and $y _ { f }$ are correct. A FIX repairs an error $( r _ { 0 } = 0 , r _ { f } = 1 )$ a BREAK overturns a correct answer $( r _ { 0 } ~ = ~ 1$ $r _ { f } = 0 )$ , and a CW switches between incorrect options. We report initial-answer accuracy $\operatorname { A c c } _ { \mathrm { p r e } } =$ $\mathbb { E } [ r _ { 0 } ]$ , final accuracy $\operatorname { A c c } ( \hat { y } )$ , and reflection gain $\Delta = \mathrm { A c c } ( \hat { y } ) - \mathrm { A c c } _ { \mathrm { p r e } }$ . Let $a = \mathbb { E } [ r _ { 0 } ]$ , and let $\rho$ and δ be the rates at which reflection repairs incorrect and breaks correct initial answers.

Observation 1 (Reflection discounts the initial answer). Since $\mathrm { P r } [ \mathrm { F I X } ] ~ = ~ \rho ( 1 - a )$ and $\mathrm { P r } [ \mathrm { B R E A K } ] = \delta a ,$

$$
\begin{array} { c } { { \mathbb { E } [ r _ { f } ] = \rho + \left( 1 - \rho - \delta \right) a , } } \\ { { \mathbb { E } [ r _ { f } - r _ { 0 } ] = \rho - \left( \rho + \delta \right) a . } } \end{array}\tag{1}
$$

For fixed reflective behavior, $\partial \mathbb { E } [ r _ { f } ] / \partial a = 1 -$ $\rho - \delta .$ . An outcome reward thus values a correct initial answer at only $1 - \rho - \delta $ , so the better the reflection, the weaker the pressure to be right. An improvement reward even values it negatively, paying more for an error that is later fixed than for an answer that is correct throughout. Since reflection gain decreases in $^ { a , }$ selecting policies by reflection gain also selects for erosion. Figure 1 confirms this prediction: Outcome RL turns a negative reflection gain positive while initial-answer accuracy stays flat, and Table 2 shows the same pattern on all three benchmarks. The policy learns a better fallback, not a better answer.

Drift and interference. Routing credit by stage (Kumar et al., 2025; Chen et al., 2026) removes this discount but not erosion, because answering and reflecting share parameters. Reflection updates can oppose the initial-answer gradient (interference) and shift answers on unsampled questions $( d r i f t )$

![](images/58551982bbd698daf2189bb9f74e35e359c114a8745a4c2471f91f3780b3198c.jpg)  
Figure 2: Answering and re-examining a Video-MME example. From a 2 fps preview and an eight-frame first look showing only two mugshots, $S _ { \mathrm { p r e } }$ answers $y _ { 0 } = \mathbf { C } , S _ { \mathrm { { g a t e } } }$ reports low confidence and chooses SEEK; frames at 258–261 s reveal a third mugshot, and $S _ { \mathrm { r e v } }$ revises to $y _ { f } = \mathrm { D } ,$ a FIX. More examples are in Appendix G.

and neither a fixed reward weight nor the model’s self-reported confidence (Section 5.4) can detect these effects.

## 4 Confident First, Reflective Second

Figure 3 gives an overview of COFIRE. Section 4.1 formalizes the confidentfirst, reflective second principle as a constrained objective, and Sections 4.2– 4.4 close the incentive, drift, and interference channels in turn.

## 4.1 Confident First, Reflective Second

A frozen pretrained model $\pi _ { \mathrm { b a s e } }$ provides reference answers, and a frozen copy of the supervised initialization, $\pi _ { \mathrm { r e f } }$ , provides KL targets. For each training input x, we cache the direct reference answer $y _ { b }$ and its correctness $r _ { b } = \mathbb { I } [ y _ { b } = y ^ { \star } ]$ , which separate verified reference decisions from reference errors.

Let $\mathcal { D } _ { w }$ denote the training distribution over labeled inputs, including reference-error reweighting when enabled. Define initial-answer reliability $J _ { 0 } ( \theta ) = \mathbb { E } [ r _ { 0 } ]$ , reference reliability $J _ { \mathrm { b a s e } } = \mathbb { E } [ r _ { b } ]$ and reflective utility $J _ { \mathrm { r e v } } ( \theta ) = \mathbb { E } [ u ]$ , where u is the transition utility in Equation 4. Expectations are over inputs from $\mathcal { D } _ { w }$ and, for $J _ { 0 }$ and $J _ { \mathrm { r e v } }$ , over trajectories from $\pi _ { \boldsymbol { \theta } } .$

COFIRE is built on the constrained objective

$$
\begin{array} { r l } { \underset { \theta } { \operatorname* { m a x } } } & { { } J _ { 0 } ( \theta ) + \alpha J _ { \mathrm { r e v } } ( \theta ) } \\ { \mathrm { s u b j e c t  t o } } & { { } J _ { 0 } ( \theta ) \geq J _ { \mathrm { b a s e } } - \epsilon , } \end{array}\tag{2}
$$

where $\alpha \geq 0$ weights reflection and $\epsilon \geq 0$ is the permitted shortfall relative to the reference $( \epsilon =$ 0.10). We call the constraint the confidencefloor.

Equation 2 differs from outcome-driven reflective RL in two ways. First, initial-answer reliability enters as a separate term, so reflection is never credited for the value of the answer. Second, it enters as a floor, so the policy cannot trade it for reflective utility, however profitable reflection becomes. Reflection gain $\Delta$ is deliberately absent, since by Observation 1 optimizing it would reward erosion. We optimize Equation 2 approximately with stage-routed policy surrogates (Appendix C).

## 4.2 Initial-Answer Credit under a Confidence Floor

Credit the initial answer only for itself. For each question, sample G trajectories indexed by $i \ = \ 1 , \ldots , G .$ For a scalar trajectory signal $z _ { i }$ , let the within-group centering be ctr $\left( z _ { i } \right) \mathbf { \Psi } =$ $\begin{array} { r } { z _ { i } - \frac { 1 } { G } \sum _ { j = 1 } ^ { G } z _ { j } } \end{array}$ . Let $\lambda \geq 0$ be an adaptive floor multiplier. We define

$$
A _ { 0 , i } = ( 1 + \lambda ) \operatorname { c t r } ( r _ { 0 , i } ) ,\tag{3}
$$

$$
u _ { i } = \mathbb { I } \big [ \mathrm { F I X } _ { i } \big ] - \beta \mathbb { I } \big [ \mathrm { B R E A K } _ { i } \big ] - \gamma \mathbb { I } \big [ \mathrm { C W } _ { i } \big ] ,\tag{4}
$$

$$
A _ { \mathrm { r e v } , i } = \mathrm { c t r } ( u _ { i } ) .
$$

We assign $A _ { 0 , i }$ to initial-answer tokens and $A _ { \mathrm { r e v } , i }$ to gate and post-answer tokens, and apply α when combining the stage gradients (Section 4.4). Under this routing, reflection no longer pays for initialanswer errors. A trajectory that errs and is later fixed gives its initial answer a below-average advantage, while its reflective stages are credited for the repair. In the notation of Observation 1, the initial answer is valued at 1 + λ per unit of accuracy, independent of $\rho$ and δ. The transition utility is asymmetric. With $\beta = 1 . 5 \mathrm { a n d } \gamma = 0 . 2 , s$ fix earns +1, a break costs 1.5, and a change between incorrect options costs 0.2, so reflection learns to respect a reliable initial answer rather than compete with it. Format rewards and a mode-dependent effort cost further encourage valid and economical verification (Appendix C).

![](images/7c12341b040f2af4b327e8019f10c69d3a5830732c661eda891521ea2c96d8b5.jpg)  
Figure 3: Overview of COFIRE. Top: G rollouts of one policy answer $( S _ { \mathrm { p r e } } )$ , choose how to check $( S _ { \mathrm { g a t e } } )$ , and confirm or revise $( S _ { \mathrm { r e v } } )$ . Middle: stage-routed advantages penalize the initial answer of an err-then-fix rollout but credit the repair. Bottom: one mechanism per erosion channel: λ rises below the initial-answer accuracy floor (incentive), verified reference answers anchor the decision token (drift), and only the conflicting part of the reflection gradient is removed (interference).

Raise the weight when reliability slips. Routed credit removes the explicit discount, but drift and interference can still erode the initial answer, and a fixed credit weight cannot react. We therefore adapt the weight through the Lagrangian of Equation 2,

$$
\begin{array} { c } { { \mathcal { T } _ { \lambda } ( \theta ) = ( 1 + \lambda ) J _ { 0 } ( \theta ) + \alpha J _ { \mathrm { r e v } } ( \theta ) } } \\ { { - \lambda ( J _ { \mathrm { b a s e } } - \epsilon ) , } } \end{array}\tag{5}
$$

which yields the weight $1 + \lambda$ in Equation 3. With ${ \widehat J } _ { 0 }$ and $\widehat { J } _ { \mathrm { b a s e } }$ denoting moving-average estimates of initial-answer and cached reference correctness on training batches (Appendix C.3), we update the multiplier by projected dual ascent,

$$
\begin{array} { r } { \lambda  \mathrm { c l i p } \Big ( \lambda + \eta _ { \lambda } \big ( \widehat { J } _ { \mathrm { b a s e } } - \epsilon - \widehat { J } _ { 0 } \big ) , 0 , \lambda _ { \mathrm { m a x } } \Big ) , } \end{array}\tag{6}
$$

with step size $\eta _ { \lambda } > 0$ and bound $\lambda _ { \mathrm { m a x } }$ , holding λ fixed within each policy update.

Below the floor, the weight on being right the first time grows until the violation is repaired;

above it, the extra weight relaxes and reflection receives a larger share of the update. Unlike a fixed first-attempt regularizer, the protection strength thus adapts to the policy’s own behavior.

## 4.3 Verified-Decision Anchoring

Consolidate verified decisions where they are made. The multiplier reacts to aggregate reliability, whereas drift acts on individual decisions. We therefore anchor the decision itself, and only where its correctness is verified. Let $h _ { t }$ denote the multimodal context preceding generated position t. At the initial-answer option-label position t<sup>⋆</sup>, define $p _ { \theta } ^ { \star } = \pi _ { \theta } ( \cdot \mid h _ { t ^ { \star } } )$ and $p _ { \mathrm { r e f } } ^ { \star } = \pi _ { \mathrm { r e f } } ( \cdot \mid h _ { t ^ { \star } } )$ on the same sampled context. With $y _ { b }$ as the target label and $\pi _ { \mathrm { r e f } }$ as the KL anchor,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { d e c } } = \mathbb { E } [ \mu r _ { b } \operatorname { C E } ( p _ { \theta } ^ { \star } , y _ { b } ) + \kappa _ { 1 } D _ { \mathrm { K L } } ( p _ { \theta } ^ { \star } \| p _ { \mathrm { r e f } } ^ { \star } ) ] , } \end{array}\tag{7}
$$

where $\mu , \kappa _ { 1 } \geq 0$ weight the two terms. The correctness gate $r _ { b }$ makes consolidation selective. The cross-entropy term consolidates the reference decision only where it is correct, so the policy is never anchored to a wrong answer and the correctness reward remains free to find a better one, while the KL term bounds drift. The preserved confidence is thus earned from labels rather than read from self-reported certainty, and no confidence threshold is needed. Because both terms act only at $t ^ { \star }$ they protect the decision without constraining the reasoning that leads to it.

Table 1: Overall performance on three video benchmarks. Video-MME is evaluated without subtitles; MLVU reports m-avg and LongVideoBench validation accuracy. <sup>‡</sup> marks source subtotal discrepancies (Appendix C.5).
<table><tr><td>Model</td><td colspan="4">Video-MME</td><td rowspan="2"></td><td rowspan="2">MLVU LongVideoBench</td><td rowspan="2"> $\operatorname { A v g }$ </td></tr><tr><td></td><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td></tr><tr><td>Without tools</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>71.4</td><td>60.1</td><td>52.3</td><td>61.3</td><td>57.9</td><td>53.9</td><td>57.7</td></tr><tr><td>VL-Rethinker (Wang et al., 2025a)</td><td>70.1</td><td>60.8</td><td>53.0</td><td>61.3</td><td>63.5</td><td>55.6</td><td>60.1</td></tr><tr><td>Vision-R1-7B (Huang et al., 2025)</td><td>71.2</td><td>60.2</td><td>52.8</td><td>61.4</td><td>68.2</td><td>52.8</td><td>60.8</td></tr><tr><td>Video-R1-7B‡ (Feng et al., 2025)</td><td>72.2</td><td>58.1</td><td>52.3</td><td>60.8</td><td>68.5</td><td>60.1</td><td>63.1</td></tr><tr><td>Tool-augmented reasoning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TimeSearch-R-7B (Pan et al., 2026)</td><td>73.5</td><td>61.2</td><td>53.4</td><td>62.7</td><td>69.3</td><td>61.2</td><td>64.4</td></tr><tr><td>Reflect-R1-7B (Chen et al., 2026)</td><td>73.9</td><td>61.0</td><td>55.6</td><td>63.5</td><td>69.8</td><td>62.5</td><td>65.3</td></tr><tr><td>CoFiRE-7B (ours)</td><td>75.0</td><td>61.2</td><td>57.3</td><td>64.5</td><td>71.2</td><td>65.2</td><td>67.0</td></tr></table>

Give reflection a looser leash. For stage $s \in$ gate, rev with generated positions $\mathcal { T } _ { s } ,$ , we regularize later stages with the token-averaged KL $\begin{array} { r l r } { \mathcal { L } _ { \mathrm { l a t e } } } & { = } &  \kappa _ { 2 } \sum _ { s } \mathbb { E } \big [ \frac { 1 } { | T _ { s } | } \sum _ { t \in \mathcal { T } _ { s } } \ \end{array}$ D<sub>KL</sub>(π<sub>θ</sub>( $h _ { t } ) \vert \vert \pi _ { \mathrm { r e f } } ( \cdot ~ \vert ~ h _ { t } ) ) \vert$ . We set $\kappa _ { 2 } ~ < ~ \kappa _ { 1 } :$ the answer decision is held close to the reference, whereas gate and reflection tokens, which must learn new behavior, are only loosely regularized.

## 4.4 Asymmetric Gradient Protection

Initial-answer credit and anchoring define what must be protected, and the interference channel decides whether reflective learning respects it. We therefore make the shared update asymmetric.

Let $\mathcal { L } _ { \mathrm { p r e } } , \mathcal { L } _ { \mathrm { g a t e } } .$ , and ${ \mathcal { L } } _ { \mathrm { r e v } }$ denote the negative clipped GRPO-style policy surrogates for $S _ { \mathrm { p r e } } ,$ $S _ { \mathrm { g a t e } }$ , and $S _ { \mathrm { r e v } } .$ , using the routed advantages above and auxiliary shaping terms (Appendices C and C.4). We define the protected objective ${ \mathcal { L } } _ { \mathrm { p } } \ =$ $\mathcal { L } _ { \mathrm { p r e } } + \mathcal { L } _ { \mathrm { d e c } }$ and the reflective objective $\mathcal { L } _ { \mathrm { r } } ~ =$ ${ \mathcal { L } } _ { \mathrm { g a t e } } + { \mathcal { L } } _ { \mathrm { r e v } } + { \mathcal { L } } _ { \mathrm { l a t e } } ,$ so that ${ \mathcal { L } } _ { \mathrm { p } }$ both learns correct initial answers from current rollouts and consolidates verified reference decisions.

Let $g _ { \mathrm { p } } = \nabla _ { \theta } \mathcal { L } _ { \mathrm { p } }$ and $g _ { \mathrm { r } } = \nabla _ { \theta } { \mathcal { L } } _ { 1 }$ <sub>r</sub> on the same batch and parameters. We compute

$$
\begin{array} { r l r } & { \omega = \operatorname* { m i n } \left( 0 , \frac { \langle g _ { \mathrm { r } } , g _ { \mathrm { p } } \rangle } { \| g _ { \mathrm { p } } \| ^ { 2 } + \varepsilon _ { \mathrm { p r o j } } } \right) , } & \\ & { g _ { \mathrm { r } } ^ { \prime } = g _ { \mathrm { r } } - \omega g _ { \mathrm { p } } , \qquad g = g _ { \mathrm { p } } + \alpha g _ { \mathrm { r } } ^ { \prime } . } & \end{array}\tag{8}
$$

Here $\omega ~ \leq ~ 0$ is the correction coefficient and $\varepsilon _ { \mathrm { p r o j } } > 0$ a numerical stabilizer. When the two gradients agree, reflection is learned unchanged. When they conflict, the opposing component of the reflection gradient is removed while $g _ { \mathrm { p } }$ is kept intact. With exact gradients and $\varepsilon _ { \mathrm { p r o j } } = 0 \nonumber$ , this is the Euclidean projection of g<sub>r</sub> onto the halfspace $\{ v \ : \ \langle g _ { \mathrm { p } } , v \rangle \ \geq \ 0 \}$ , and $\langle g _ { \mathrm { p } } , g \rangle \geq \| g _ { \mathrm { p } } \| ^ { 2 }$ for $\alpha \geq 0$ , so a sufficiently small step decreases the protected objective at least as fast as without reflection (Proposition 1, Appendix F). The geometry follows A-GEM (Chaudhry et al., 2019), but what we protect is the current initial-answer reliability of the same policy rather than a memory of past tasks.

Protection does not freeze wrong answers. The coefficient ω is computed per batch, and $r _ { b }$ gates only the CE anchor, not the projection. Because $\mathrm { c t r } ( r _ { 0 , i } )$ is negative for incorrect initial answers, $g _ { \mathrm { p } }$ both raises the probability of correct initial answers and lowers that of incorrect ones. Asymmetric protection preserves both directions: it forbids reflection from undoing the push away from a wrong initial answer as well as the pull toward a right one, so the protected direction points toward better answers rather than toward the status quo.

Implementation as loss reweighting. Since computing both full-model gradients separately is <sup>expensive, we estimate</sup> ωb <sup>on 16 top-layer parameter</sup> tensors, detach it, and minimize

$$
{ \mathcal { L } } _ { \mathrm { u p d a t e } } = ( 1 - \alpha { \widehat { \omega } } ) { \mathcal { L } } _ { \mathrm { p } } + \alpha { \mathcal { L } } _ { \mathrm { r } } .\tag{9}
$$

<sup>With</sup> ωb <sup>held</sup> <sup>constant,</sup> <sup>the</sup> <sup>gradient</sup> <sup>of</sup> <sup>Equation</sup> <sup>9</sup> <sup>is</sup> $g _ { \mathrm { p } } + \alpha ( g _ { \mathrm { r } } - { \widehat { \omega } } g _ { \mathrm { p } } )$ , exactly the protected update of Equation 8. The projection thus reduces to a single scalar that up-weights the initial-answer objective whenever reflection conflicts with it. The subset only affects the coefficient estimate, and the update still trains all parameters with a single backward pass (Algorithm 1, Appendix F).

## 5 Experiments

We evaluate whether COFIRE improves video understanding (Section 5.2), whether and through which mechanisms it prevents confidence erosion

Table 2: Initial-answer accuracy versus reflection gain under a shared inference protocol. $\mathrm { A c c } _ { \mathrm { p r e } }$ and Final are initial-answer and returned-answer accuracy; ∆ is their difference in points. Duration breakdowns appear in Table 7.
<table><tr><td rowspan="2">Training configuration</td><td colspan="3">Video-MME</td><td colspan="3">MLVU</td><td colspan="3">LongVideoBench</td></tr><tr><td> $\mathrm { A c c } _ { \mathrm { p r e } }$ </td><td>Final</td><td>∆</td><td> $\mathrm { A c c } _ { \mathrm { p r e } }$ </td><td>Final</td><td> $\Delta$ </td><td> $\mathrm { A c c } _ { \mathrm { p r e } }$ </td><td>Final</td><td>∆</td></tr><tr><td>Zero-shot, our protocol</td><td>57.7</td><td>57.2</td><td>-0.5</td><td>56.0</td><td>55.2</td><td>-0.8</td><td>51.1</td><td>49.9</td><td>-1.2</td></tr><tr><td>Supervised initialization</td><td>60.4</td><td>60.1</td><td>-0.3</td><td>65.9</td><td>65.5</td><td>-0.4</td><td>58.3</td><td>57.6</td><td>-0.7</td></tr><tr><td>+ Outcome RL</td><td>60.3</td><td>60.8</td><td>+0.5</td><td>65.9</td><td>67.8</td><td>+1.9</td><td>57.9</td><td>61.2</td><td>+3.3</td></tr><tr><td>+ CoFIRE</td><td>64.2</td><td>64.5</td><td>+0.3</td><td>69.9</td><td>71.2</td><td> $+ 1 . 3$ </td><td>62.3</td><td>65.2</td><td>+2.9</td></tr></table>

Table 3: Contribution of each mechanism on Video-MME. Mechanisms are added cumulatively to Outcome RL and grouped by the channel they close; the last row removes gradient protection from the full configuration. Long is final accuracy on long videos. The complete progression is in Table 8.
<table><tr><td>Channel</td><td>Configuration</td><td> $\mathrm { A c c } _ { \mathrm { p r e } }$ </td><td>Final</td><td> $\Delta$ </td><td>Long</td></tr><tr><td></td><td>Outcome RL</td><td>60.3</td><td>60.8</td><td>+0.5</td><td>51.8</td></tr><tr><td>Incentive</td><td>+ routed initial-answer credit</td><td>61.0</td><td>61.6</td><td>+0.6</td><td>52.7</td></tr><tr><td>Drift</td><td>+ CE anchor at  $t ^ { \star }$ </td><td>62.3</td><td>62.4</td><td>+0.1</td><td>54.0</td></tr><tr><td>Incentive + drift</td><td>+ floor multiplier, KL at  $t ^ { \star }$  , transition utility</td><td>62.0</td><td>62.3</td><td>+0.3</td><td>53.8</td></tr><tr><td>Interference</td><td>+ asymmetric gradient protection</td><td>63.2</td><td>63.4</td><td>+0.2</td><td>55.8</td></tr><tr><td>All</td><td>+ later-stage KL, reference-error sampling (full CoFiRE)</td><td>64.2</td><td>64.5</td><td>+0.3</td><td>57.3</td></tr><tr><td>Interference</td><td>Full CoFiRE w/o gradient protection</td><td>63.5</td><td>64.0</td><td>+0.5</td><td>55.8</td></tr></table>

(Section 5.3), and how reflection behaves on a confident initial answer (Section 5.4).

## 5.1 Experimental Setup

All trained variants start from the same supervised Qwen2.5-VL-7B-Instruct policy (Bai et al., 2025) and train on the 42,602 multiple-choice questions of the Reflect-R1 pool (Chen et al., 2026) for up to 1,000 steps. Appendices B and C detail the data, inference protocol, and hyperparameters. Outcome RL, the standard reflective RL recipe analyzed in Observation 1, trains the same staged policy with final correctness plus a correction bonus applied to the whole trajectory, and otherwise shares the initialization, data, rollout and inference protocol, and hyperparameters of COFIRE (Appendix C). Following Reflect-R1 (Chen et al., 2026), we evaluate on Video-MME without subtitles, MLVU m-avg, and LongVideoBench validation (Fu et al., 2025; Zhou et al., 2025; Wu et al., 2024), reporting initialanswer accuracy $\mathrm { A c c } _ { \mathrm { p r e } } ,$ final accuracy Acc(ˆy), reflection gain $\Delta ,$ and the unweighted benchmark average Avg (Appendix C.5).

## 5.2 Main Results

As shown in Table 1, COFIRE achieves the best results among compared 7B models on every benchmark, with a 67.0% average. It outperforms Reflect-R1, the strongest reflective baseline, by 1.0–2.7 points, with the largest margins on long Video-MME videos and LongVideoBench, where an unreliable first glance is most costly. These results show that protecting the initial answer pays off most in exactly the settings that motivate re-examination.

## 5.3 Does COFIRE Prevent Confidence Erosion?

COFIRE improves the answer, not the fallback. As illustrated in Figure 1, Outcome RL channels its gains into reflection. In contrast, Table 2 shows that COFIRE raises Video-MME initial-answer accuracy by 3.9 points and final accuracy by 3.7 points with a smaller reflection gain (+0.3), so ranking policies by reflection gain would select the wrong one. Reflection still helps on every benchmark, most on LongVideoBench (+2.9), where a single preview is least sufficient. This confirms that reflection gain misrepresents progress, since the improvement of COFIRE comes from a stronger initial answer that carries through to the final answer.

Closing the incentive channel. Routed initialanswer credit directly tests Observation 1. As shown in Table 3, once the initial answer is no longer paid for errors that reflection repairs, initialanswer and final accuracy rise by 0.7 and 0.8 points without lowering the reflection gain. This confirms the reward discount of Observation 1 as a real cause of erosion, but removing it alone is not enough, as the drift and interference mechanisms contribute the rest of the initial-answer gain.

Table 4: Gate-conditioned outcomes on Video-MME (%). Each policy selects its own subsets; SEEK and TRUST are shown. Fix and break frequencies use all SEEK questions as the denominator; the per-error repair rates in the text divide Fix by the initial-answer error rate on SEEK questions.
<table><tr><td></td><td>SEEK rate</td><td>Acc SEEK</td><td>Acc TRUST</td><td>Fix</td><td>Break</td></tr><tr><td>Outcome RL</td><td>14.6</td><td>44.3</td><td>62.8</td><td>16.7</td><td>9.6</td></tr><tr><td>CoFIRE</td><td>15.7</td><td>51.9</td><td>65.8</td><td>13.2</td><td>11.1</td></tr></table>

Closing the drift channel. As shown in Table 3, anchoring verified reference decisions at $t ^ { \star }$ raises initial-answer accuracy by 1.3 points while the reflection gain falls to +0.1, consistent with RL overwriting decisions the reference already gets right even under routed credit. The anchor recovers these decisions without constraining the reasoning, and the floor multiplier, decision-local KL, and transition utility then partly restore the reflection gain. Drift is therefore a distinct erosion channel that reward design alone cannot close, and anchoring verified decisions at the decision token closes it effectively.

Interference is most pronounced on long videos. As shown in Table 3, asymmetric gradient protection adds 1.2 points of initial-answer accuracy, 1.1 of final accuracy, and 2.0 on long videos. Removing it from the full configuration lowers all three while raising the reflection gain, the erosion signature predicted by Observation 1. Gradient interference thus erodes the initial answer even when rewards and anchors are in place, and asymmetric protection is most valuable on long videos, where the initial answer is hardest to get right. Across the progression, the two largest initialanswer gains, from the CE anchor and gradient protection, both carry through to final accuracy, supporting our premise that a stronger initial answer yields a stronger final one.

## 5.4 Reflection on a Confident Initial Answer

Same corrector, more reliable initial answer. As shown in Table 4, both gates send a similar share of questions to retrieval, yet the answers that COFIRE trusts are 3.0 points more accurate. On SEEK questions, COFIRE starts from a far more accurate initial answer (49.8% vs. 37.2%), while both correctors repair a similar fraction of initial errors (26.3% vs. 26.6%), far above the 6.2% of REVIEW, which re-examines the first look without new frames. COFIRE thus applies an equally effective corrector to a far more reliable initial answer, showing that its gains come from avoiding errors upfront rather than from repairing more of them.

![](images/2fe0bbebd8fdfe6da09f10179117d4b3264bc8a225b00a781e00489547a0796b.jpg)  
Figure 4: Penalizing revision does not protect the initial answer. Video-MME SEEK rate, the share of questions routed to SEEK (bars, left) and final accuracy (line, right), varying one penalty from the default $\beta =$ 1.5, $\gamma = 0 . 2 $

Timid reflection and self-reported confidence are no substitute. As demonstrated in Figure 4, penalizing breaks more heavily suppresses reflection without strengthening the initial answer. Raising β from 1.5 to 2.0 cuts the SEEK rate by a third and lowers final accuracy, and a larger $\beta$ or γ nearly eliminates retrieval with further losses. Making reflection timid thus gives up useful retrieval without making the initial answer more reliable. Selfreported confidence is no substitute either. As reported in Table 9, the gate’s confidence predicts initial-answer correctness on Video-MME with an AUC of only 0.68, and the option belief over retrieved frames separates fixes from breaks at 0.53. The model’s own confidence is therefore too weak a signal to decide when to keep or overturn the initial answer. Erosion is thus prevented by protecting the initial answer with verified correctness, not by making reflection timid.

## 6 Conclusion

RL on answer-and-re-examine video loops can erode the initial answer, raising reflection gain while initial-answer accuracy stalls. Following confident first, reflective second, COFIRE closes the incentive, drift, and interference channels with dedicated mechanisms, achieving the best results among compared 7B models on three video benchmarks and improving initial-answer and final accuracy over Outcome RL by 4.1 and 3.7 points on average.

## Limitations

We study one backbone and one answer-and-reexamine loop on long-video multiple-choice benchmarks, with one run per configuration; extending the analysis to other loops trained with finaloutcome rewards, such as agentic temporal search, is future work.

## References

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. 2017. Constrained policy optimization. In International Conference on Machine Learning (ICML).

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923.

Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. 2019. Efficient lifelong learning with A-GEM. In International Conference on Learning Representations (ICLR).

Shuimu Chen, Yuteng Chen, Yuanshen Guan, Zebang Cheng, Zeyu Zhang, Shengqian Qin, Bin Xia, Jiaran Li, Wenming Yang, and Fei Ma. 2026. Reflect-R1: Evidence-driven reflection for self-correction in long video understanding. In European Conference on Computer Vision (ECCV).

Kanzhi Cheng, Yantao Li, Fangzhi Xu, Jianbing Zhang, Hao Zhou, and Yang Liu. 2025. Vision-language models can self-improve reasoning via reflection. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL).

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. 2025. Video-R1: Reinforcing video reasoning in MLLMs. In Advances in Neural Information Processing Systems (NeurIPS), volume 38.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. 2025. Video-MME: The first-ever comprehensive evaluation benchmark of multi-modal LLMs in video analysis. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In International Conference on Machine Learning (ICML).

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai

Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, et al. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645(8081):633–638.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In International Conference on Learning Representations (ICLR).

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Xu Tang, Yao Hu, and Shaohui Lin. 2025. Vision-R1: Incentivizing reasoning capability in multimodal large language models. Preprint, arXiv:2503.06749.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526.

Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D. Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. 2025. Training language models to self-correct via reinforcement learning. In International Conference on Learning Representations (ICLR).

Xinhao Li, Ziang Yan, Desen Meng, Lu Dong, Xiangyu Zeng, Yinan He, Yali Wang, Yu Qiao, Yi Wang, and Limin Wang. 2025. VideoChat-R1: Enhancing spatio-temporal perception via reinforcement finetuning. arXiv preprint arXiv:2504.06958.

David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. In Advances in Neural Information Processing Systems (NeurIPS).

Ruotian Ma, Peisong Wang, Cheng Liu, Xingyan Liu, Jiaqi Chen, Bang Zhang, Xin Zhou, Nan Du, and Jia Li. 2025. S<sup>2</sup>R: Teaching LLMs to self-verify and selfcorrect via reinforcement learning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 22632–22654.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. 2023. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems (NeurIPS).

Junwen Pan, Qizhe Zhang, Rui Zhang, Ming Lu, Xin Wan, Yuan Zhang, Chang Liu, and Qi She. 2026. TimeSearch-R: Adaptive temporal search for longform video understanding via self-verification reinforcement learning. In International Conference on Learning Representations (ICLR).

Mengxue Qu, Yibo Hu, Kunyang Han, Yunchao Wei, and Yao Zhao. 2025. ReCoT: Reflective selfcorrection training for mitigating confirmation bias in large vision-language models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 9147–9157.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS).

Chen Tessler, Daniel J. Mankowitz, and Shie Mannor. 2019. Reward constrained policy optimization. In International Conference on Learning Representations (ICLR).

Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, Chaofan Tao, Yangfan He, Mi Zhang, and Shen Yan. 2025. SRPO: Enhancing multimodal LLM reasoning via reflection-aware reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 38.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. 2025a. VL-Rethinker: Incentivizing self-reflection of visionlanguage models with reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 38.

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. 2024. VideoAgent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision (ECCV).

Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. 2025b. VideoTree: Adaptive tree-based video representation for LLM reasoning on long videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. LongVideoBench: A benchmark for longcontext interleaved video-language understanding. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 28828–28857.

Jinhui Ye, Zihan Wang, Haosen Sun, Keshigeyan Chandrasegaran, Zane Durante, Cristobal Eyzaguirre, Yonatan Bisk, Juan Carlos Niebles, Ehsan Adeli, Li Fei-Fei, et al. 2025. Re-thinking temporal search for long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8579–8591.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. 2020. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems (NeurIPS).

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, et al. 2025. MLVU: Benchmarking multi-task long video understanding. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13691–13701.

## A Extended Related Work

Protecting one capability while learning another. Constrained RL enforces performance requirements with Lagrangian multipliers (Achiam et al., 2017; Tessler et al., 2019), continual learning protects earlier tasks with parameter penalties or gradient constraints (Kirkpatrick et al., 2017; Lopez-Paz and Ranzato, 2017; Chaudhry et al., 2019), PCGrad removes conflicts among task gradients (Yu et al., 2020), and self-reported confidence is known to be miscalibrated (Guo et al., 2017; Kadavath et al., 2022). COFIRE brings these ideas inside a single reflective policy, where answering and reflecting are coupled capabilities, and adapts each to that setting: the constraint is a confidencefloor on the policy’s own initial-answer accuracy, enforced by a dual multiplier on initial-answer credit; consolidation targets only label-verified reference decisions at the decision token, rather than decisions the model is merely confident in; and gradient surgery is asymmetric, so reflection yields to answering but never the reverse.

## B Training Data and Supervised Initialization

## B.1 RL Question Pool

We use the multiple-choice question pool released with Reflect-R1 (Chen et al., 2026). The short split contains 30,974 questions on 15,147 videos from LLaVA-Video-178K, STAR, NeXT-QA, CLEVRER, and PerceptionTest. These videos last 4–180 seconds, with a median duration of 32 seconds. The long split contains 11,628 questions on 3,527 Panda-70M videos lasting 10 seconds to one hour, with a median duration of 17 minutes. Its 21 question types include temporal gaps, action localization, object existence, duration, and temporal order.

Reference cache. Cached answers are greedy direct predictions from pretrained Qwen2.5-VL-7B-Instruct. For 498 long-video questions without released predictions, we recomputed answers from 12 uniformly sampled frames. Cache accuracy is 73.9% on short questions, 15.3% on long questions, and 57.9% overall.

Overlap checks. Comparing training-video YouTube identifiers with the 900 Video-MME videos found no shared identifiers. Exact questionstring matches covered 12 generic templates, including questions about a video’s main focus.

## B.2 Supervised Initialization

Supervised trajectories teach an overview request, an initial answer, a gate decision, and a final answer. Answer turns use <think>. . . </think> followed by <answer>. . . </answer>, and the gate outputs reasoning and the JSON fields confidence, mode, queries, and num\_frames. Visual observations are inserted between assistant turns, and the supervised overview contains at most 60 frames. All RL runs start from the same supervised checkpoint, and gate confidence targets are constructed from correctness labels.

## C Implementation and Objective Details

## C.1 Training Settings

Losses are averaged over generated text positions in each stage, and the four gate effort costs correspond to TRUST, REVIEW, SEEK, and MULTI-HOP, respectively. First-look and retrieved frames come from a separate 2 fps cache of at most 2,048 frames per video. Because the preview shares one pixel budget across its frames, it is coarse for long videos: at the 768-frame evaluation cap, each preview frame has at most 40 units of 28 28 pixels (about 177 177), and at most 192 units for short videos, whereas each cache frame has up to 256 units (about 448 448). The preview is used only by $S _ { \mathrm { p r e } }$ ; the gate sees the first-look frames, and $S _ { \mathrm { r e v } }$ sees the first-look and retrieved frames. Apart from the decision-local and later-stage terms of Section 4.3, no sequence-wide KL is used, and format rewards are routed only to the gate and post-answer stage.

Outcome RL reward. Outcome RL scores each trajectory by its final correctness plus a correction bonus, $R _ { i } = r _ { f , i } + c \mathbb { I } [ \mathrm { F I X } _ { i } ]$ with $c > 0$ , and assigns the group-centered advantage ctr $( R _ { i } )$ to every generated token of the trajectory, without stage routing, floor multiplier, anchoring, or gradient protection; format rewards are the same as for COFIRE.

Inference protocol. The preview is sampled at 2 fps with at most 768 frames, and $S _ { \mathrm { p r e } }$ adds a first look of at most eight cache frames; each retrieval round adds 4–32 gate-requested frames, with one round for SEEK and at most three for MULTI-HOP. SigLIP-so400m option beliefs let MULTI-HOP stop once evidence is sufficient, and evaluation decodes greedily. Outcome RL uses this protocol unchanged, and Reflect-R1 is reproduced with its official pipeline.

Table 5: Training sources and cached reference accuracy. The cache includes 498 long-video predictions recomputed with a 12-frame budget.
<table><tr><td>Split</td><td>Source</td><td>Questions</td><td>Durations</td><td> $\mathbb { E } [ r _ { b } ]$ </td></tr><tr><td rowspan="6">Short</td><td>LLaVA-Video-178K</td><td>20,654</td><td>0–3 min</td><td>0.79</td></tr><tr><td>STAR</td><td>4,197</td><td></td><td>0.38</td></tr><tr><td>NeXT-QA</td><td>2,297</td><td></td><td>0.78</td></tr><tr><td>CLEVRER</td><td>1,919</td><td></td><td>1.00</td></tr><tr><td>PerceptionTest</td><td>1,907</td><td></td><td>0.66</td></tr><tr><td>Panda-70M</td><td>11,628</td><td> $1 0 \mathrm { { s } { - } 1 h }$ </td><td>0.15</td></tr><tr><td>Long Total</td><td></td><td>42,602</td><td></td><td>0.58</td></tr></table>

Answer extraction. The returned answer is read from the initial-answer stage under TRUST and from the post-answer stage otherwise; $\operatorname { A c c } _ { \mathrm { p r e } }$ always reads the initial-answer stage. Within a stage, the option letter is taken from the answer tag, falling back to the last standalone option letter when the tag is missing or malformed; an output without any option letter is scored as incorrect.

## C.2 Reference Policies and the Confidence Floor

The frozen pretrained model $\pi _ { \mathrm { b a s e } }$ supplies the cached direct answers $y _ { b }$ and their correctness $r _ { b } .$ which define $J _ { \mathrm { b a s e } }$ and the cross-entropy targets in Equation 7. The frozen supervised initialization $\pi _ { \mathrm { r e f } }$ supplies the KL targets and is also the starting RL policy. The cached answer is produced with a direct-answer prompt, whereas the initial answer is produced within the staged protocol.

The floor $J _ { \mathrm { b a s e } } - \epsilon$ is defined relative to the reference and is evaluated under the same training sampler $\mathcal { D } _ { w }$ as $J _ { 0 } ,$ so reference-error reweighting changes both quantities consistently (Appendix C.3). We use $\epsilon \ : = \ : 0 . 1 0$ The floor is fixed relative to the reference rather than to the best checkpoint reached during training. In training, λ starts at 0, both moving averages are initialized at their first batch estimates, and λ is updated after every rollout batch by Equation 6.

## C.3 Estimating the Accuracy Signals

Population quantities under the training sampler. For a training pool of N labeled inputs $( x _ { n } , y _ { n } ^ { \star } )$ , let $p _ { n } = \operatorname* { P r } _ { \mathcal { D } _ { w } } ( x _ { n } )$ be the sampling probability of question n, with $\textstyle \sum _ { n = 1 } ^ { N } p _ { n } = 1$ . Let ${ r _ { b , n } }$ denote its cached reference correctness. The two population quantities are

$$
\begin{array} { l } { { \displaystyle J _ { 0 } ( \theta ) = \sum _ { n = 1 } ^ { N } p _ { n } \operatorname* { P r } _ { \pi _ { \theta } } ( y _ { 0 } = y _ { n } ^ { \star } \mid x _ { n } ) } , \ ~ } \\ { { \displaystyle J _ { \mathrm { b a s e } } = \sum _ { n = 1 } ^ { N } p _ { n } r _ { b , n } } . } \end{array}\tag{10}
$$

The first quantity averages over sampled questions and initial-answer generation, and the second averages fixed cache entries under the same question distribution, so reference-error reweighting changes $p _ { n }$ and therefore both quantities.

Batch accuracy estimates. At training step t, sample B questions from $\mathcal { D } _ { w }$ and generate G trajectories per question using the current rollout policy. Let $r _ { 0 , j i } ^ { ( \hat { t } ) }$ be the raw initial-answer correctness of trajectory i for sampled question $j ,$ and let $r _ { b , j } ^ { ( t ) }$ be that question’s cached reference correctness. Compute

$$
\begin{array} { c } { { \bar { J } _ { 0 , t } = \displaystyle \frac { 1 } { B G } \sum _ { j = 1 } ^ { B } \sum _ { i = 1 } ^ { G } r _ { 0 , j i } ^ { ( t ) } , } } \\ { { \bar { J } _ { \mathrm { b a s e } , t } = \displaystyle \frac { 1 } { B } \sum _ { j = 1 } ^ { B } r _ { b , j } ^ { ( t ) } . } } \end{array}\tag{11}
$$

The first estimate uses only the initial answers, before any revision, and excludes the multiplier $1 + \lambda .$ group centering, transition utility, and auxiliary shaping. The reference estimate reads one cached value per question, so no reference-model generation is required, and since questions are already sampled from $\mathcal { D } _ { w } .$ , the reference-error weights are not applied a second time.

Moving averages and update timing. Let ν = 0.99 be the accuracy EMA decay from Table $^ { 6 , }$ and let $\widehat { J } _ { 0 , t }$ and ${ \widehat { J } } _ { \mathrm { b a s e } , t }$ denote the controller states available at the start of step t. After the policy update, incorporate the accuracies from that step’s sampled trajectories:

Table 6: COFIRE training and inference settings.
<table><tr><td>Group</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Rollout</td><td>Preview (train / eval)</td><td>2 fps; ≤ 734 frames, 10,240 visual tokens  $r \leq 7 6 8$  frames, 15,360 tokens</td></tr><tr><td></td><td>First look</td><td>One tool call by  $S _ { \mathrm { p r e } } ; \leq 8$  cache frames selected by query relevance and temporal diversity</td></tr><tr><td></td><td>Retrieval frames per round</td><td>Gate-selected, clipped to [4, 32] (default 8); no frame retrieved twice</td></tr><tr><td></td><td>Retrieval rounds</td><td>TRUST, REVIEW: 0; SEEK: 1; MULTI-HOP: ≤ 3</td></tr><tr><td></td><td>MULTI-HOP stopping</td><td>From round 2, sufficiency score &gt; 0.7 (top belief, margin, entropy)</td></tr><tr><td></td><td>Option belief</td><td>SigLIP-so400m frame-option similarity, temperature 0.05</td></tr><tr><td></td><td>Sampling</td><td>Temperature 1.0, top-p 1.0 (vLLM)</td></tr><tr><td></td><td>Maximum generated length</td><td>512 tokens per stage in training, 1,024 in evaluation (greedy)</td></tr><tr><td>Optimization</td><td>Optimizer</td><td>AdamW (momentum coefficients 0.9 and 0.999), no weight decay,</td></tr><tr><td></td><td>Learning rate</td><td>gradient clipping 1.0  $\bar { 1 } 0 ^ { - 6 } .$  cosine decay, no warmup</td></tr><tr><td></td><td>Batch</td><td>2 questions × 8 trajectories per optimizer step</td></tr><tr><td></td><td>Training steps</td><td>Up to 1,000</td></tr><tr><td></td><td>Replay</td><td>Groups with all-zero advantages are replaced from a 16-group buffer</td></tr><tr><td></td><td>Precision and system</td><td>bf16, DeepSpeed ZeRO-3 with CPU offload, gradient checkpointing,</td></tr><tr><td></td><td>Hardware</td><td>FlashAttention-2 8×H100 per run</td></tr><tr><td>GRPO</td><td>Trajectories per question</td><td>8</td></tr><tr><td></td><td>Advantage normalization</td><td>Group centering without standard-deviation scaling</td></tr><tr><td>Reward</td><td>Break / changed-wrong penalties</td><td> $\beta = 1 . 5 , \gamma = 0 . 2$ </td></tr><tr><td></td><td>Gate effort cost</td><td> $0 . 0 5 e , \mathrm { w i t h } e = 0 , 0 . 2 5 , 0 . 5 , 1$  across the four modes</td></tr><tr><td></td><td>Format reward weight</td><td>0.2</td></tr><tr><td>Confidence floor</td><td>Accuracy tolerance</td><td> $\epsilon = 0 . 1 0$ </td></tr><tr><td></td><td>Dual step / maximum</td><td> $\eta _ { \lambda } = 0 . 1 , \lambda _ { \operatorname* { m a x } } = 5$ </td></tr><tr><td></td><td>Accuracy EMA decay</td><td>0.99, initialized at the first batch estimate</td></tr><tr><td></td><td>Initial multiplier</td><td>λ = 0, updated after every rollout batch</td></tr><tr><td>Anchoring</td><td>Cross-entropy weight</td><td> $\mu = 1 . 0$ </td></tr><tr><td></td><td>Decision / later-stage KL</td><td> $\kappa _ { 1 } = 0 . 0 5 , \kappa _ { 2 } = 0 . 0 0 5 , \mathrm { b o t h t o } \pi _ { \mathrm { r e f } }$ </td></tr><tr><td></td><td>KL estimator</td><td> $k _ { 3 } \colon e ^ { d } - d - 1 , d = \log \pi _ { \mathrm { r e f } } - \log$  πθ clipped to  $[ - 1 0 , 1 0 ]$ </td></tr><tr><td>Gradient</td><td>Reflection weight</td><td> $\alpha = 1 . 0$ </td></tr><tr><td></td><td>Projection estimate</td><td>Last 16 trainable tensors with  $\leq 1 0 ^ { 6 }$  entries (top-layer norms and biases)</td></tr><tr><td></td><td>Stabilizer / clipping</td><td> $\varepsilon _ { \mathrm { p r o j } } \stackrel { \cdot } { = } 1 0 ^ { - 1 2 } ;$  clipped to  $[ - 1 0 , 0 ]$ </td></tr><tr><td>Sampling</td><td>Reference-error question weight</td><td>2 in the full configuration</td></tr></table>

$$
\begin{array} { r } { \widehat { J } _ { 0 , t + 1 } = \nu \widehat { J } _ { 0 , t } + ( 1 - \nu ) \bar { J } _ { 0 , t } , } \\ { \widehat { J } _ { \mathrm { b a s e } , t + 1 } = \nu \widehat { J } _ { \mathrm { b a s e } , t } \qquad } \\ { + ( 1 - \nu ) \bar { J } _ { \mathrm { b a s e } , t } . } \end{array}\tag{12}
$$

The updated values enter Equation 6 to obtain the multiplier for the next batch (Algorithm 1). The statistics come from the batch already generated, so no second rollout is needed, and both EMA states and the multiplier are treated as constants during backpropagation.

## C.4 Stage Surrogates and Candidate Rewards

Stage routing omits the downstream transition utility from the initial-answer score-function terms, so initial-answer priority is implemented through separate surrogates rather than an unbiased estimator of the joint gradient of Equation 2. When all group rewards are equal, the task advantage is zero, while auxiliary terms and anchoring still contribute gradients.

For TRUST trajectories, training still generates a post-answer candidate, and the gate and postanswer advantages use the candidate utility u in every mode, whereas the utility of the returned answer is

$$
u _ { \mathrm { d e p l o y } } = \mathbb { I } [ m \neq \mathrm { T R U S T } ] u .\tag{13}
$$

The gate is thus trained on candidate utility even under TRUST, where inference returns y<sub>0</sub>.

Transition-penalty calibration. Let p<sub>FIX</sub>, p<sub>BREAK</sub>, p<sub>CW</sub> be transition frequencies under the initialization and rollout protocol. For

p<sub>BREAK</sub> $> 0$ , the mean raw transition utility is zero at

$$
\beta ^ { \star } = \frac { p _ { \mathrm { F I X } } - \gamma p _ { \mathrm { C W } } } { p _ { \mathrm { B R E A K } } } .\tag{14}
$$

Under the supervised initialization, $\beta ^ { \star } \approx 1 . 4 3 .$ which motivates the default $\beta \ : = \ : 1 . 5 ;$ Figure 4 reports the sensitivity.

## C.5 Reporting and Reproducibility Notes

Duration subtotals in Table 7 are computed on the same checkpoints as the overall scores, and Video-MME’s three duration groups have equal size, so Overall is their mean. Baseline scores in Table 1 are transcribed from Reflect-R1 (Chen et al., 2026); <sup>‡</sup> marks Video-R1, whose published duration subtotals do not average to its published overall, a discrepancy present in the source table. Baselines use the frame budgets and inference protocols of their original papers, so Table 1 compares complete systems; Table 2 provides the controlled comparison under a single protocol. Our reproduction of Reflect-R1 with its official pipeline reaches 62.6% on Video-MME (Table 7), 0.9 points below its published 63.5%; Table 1 reports the published number.

## D Detailed Experimental Results

## D.1 Video-MME Duration Breakdowns

Table 7 reports Video-MME duration breakdowns for our training configurations and the officialpipeline Reflect-R1 reproduction, complementing Table 2.

## D.2 Complete Training Progression

Table 8 lists all nine progressive configurations underlying Table 3, adding routed initial-answer credit, the CE anchor, the floor multiplier, decisionlocal KL, transition utility, and asymmetric gradient protection in sequence. The final pair adds later-stage KL and reference-error reweighting together.

## D.3 Self-Reported Confidence as a Selection Signal

Table 9 reports how well the model’s own signals separate outcomes that matter for reflection on Video-MME. Neither signal is reliable enough to decide when to keep or overturn the initial answer, which motivates protecting it with label-verified correctness (Section 4.3).

## E Training Algorithm

Algorithm 1 gives the COFIRE update. The projection coefficient is estimated on the tensor subset and detached before forming the full-model loss, and the moving averages and multiplier are updated after each policy step.

## F Analysis of Asymmetric Gradient Protection

Proposition 1 (Fixed-batch protected descent). Assume that ${ \mathcal { L } } _ { \mathrm { p } }$ has a locally Lipschitz gradient, the sampled batch and advantages are fixed, $g _ { \mathrm { p } } \neq 0 ,$ $\varepsilon _ { \mathrm { p r o j } } = 0$ , and both gradients in Equation 8 are exact. For a plain gradient step with size $\eta > 0$ and $\alpha \geq 0$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p } } ( \theta - \eta g ) - \mathcal { L } _ { \mathrm { p } } ( \theta ) \leq - \eta \| g _ { \mathrm { p } } \| ^ { 2 } + O ( \eta ^ { 2 } ) . } \end{array}
$$

Proof. The exact projection satisfies $\langle g _ { \mathrm { p } } , g _ { \mathrm { r } } ^ { \prime } \rangle \geq 0$ A first-order expansion gives

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p } } ( \theta - \eta g ) - \mathcal { L } _ { \mathrm { p } } ( \theta ) } \\ & { = - \eta \| g _ { \mathrm { p } } \| ^ { 2 } - \eta \alpha \langle g _ { \mathrm { p } } , g _ { \mathrm { r } } ^ { \prime } \rangle + O ( \eta ^ { 2 } ) . } \end{array}
$$

The second first-order term is non-positive.

The result follows from the asymmetric projection geometry of A-GEM (Chaudhry et al., 2019) and applies to the protected surrogate on a fixed batch.

Subset coefficient estimation. The coefficient ω<sub>b</sub> is estimated on the last 16 trainable tensors with at most $1 0 ^ { 6 }$ entries each, namely top-layer normalization weights and biases, which makes the two separate gradients cheap to compute; the final update is a single backward pass of Equation 9 through all <sup>parameters.</sup> <sup>With</sup> ωb <sup>held</sup> <sup>constant</sup> <sup>during</sup> <sup>differenti-</sup> ation, the full-space inner product is

$$
\langle g _ { \mathrm { p } } , g _ { \mathrm { r } } - { \widehat { \omega } } g _ { \mathrm { p } } \rangle = \langle g _ { \mathrm { p } } , g _ { \mathrm { r } } \rangle - { \widehat { \omega } } \| g _ { \mathrm { p } } \| ^ { 2 } .
$$

Numerical stabilizer. For $c = \langle { g _ { \mathrm { r } } , g _ { \mathrm { p } } } \rangle \textless 0$ , a positive stabilizer leaves the residual

$$
\langle g _ { \mathrm { p } } , g _ { \mathrm { r } } ^ { \prime } \rangle = \frac { c \varepsilon _ { \mathrm { p r o j } } } { \| g _ { \mathrm { p } } \| ^ { 2 } + \varepsilon _ { \mathrm { p r o j } } } ,\tag{15}
$$

which vanishes as $\varepsilon _ { \mathrm { p r o j } } \ \to \ 0 ;$ we use $\varepsilon _ { \mathrm { p r o j } } =$ $1 0 ^ { - 1 2 }$

Table 7: Video-MME duration breakdowns for training comparisons. All scores are final accuracy without subtitles. Reflect-R1<sup>∗</sup> uses its official pipeline; the other rows use our staged protocol.
<table><tr><td>Configuration</td><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td></tr><tr><td>Zero-shot, our protocol</td><td>67.0</td><td>55.1</td><td>49.4</td><td>57.2</td></tr><tr><td>Reflect-R1 reproduction *</td><td>74.7</td><td>62.3</td><td>50.7</td><td>62.6</td></tr><tr><td>Supervised initialization</td><td>71.3</td><td>59.3</td><td>49.8</td><td>60.1</td></tr><tr><td>+ Outcome RL</td><td>72.7</td><td>57.9</td><td>51.8</td><td>60.8</td></tr><tr><td>+ CoFIRE</td><td>75.0</td><td>61.2</td><td>57.3</td><td>64.5</td></tr></table>

Table 8: Progressive training configurations on Video-MME. Components are added cumulatively except in the explicitly marked no-projection row. <sup>§</sup>The final two configurations add later-stage KL and double the reference-error sampling weight together.
<table><tr><td>Channel</td><td>Configuration</td><td>Overall</td><td>Long</td><td> $\mathrm { A c c } _ { \mathrm { p r e } }$ </td><td> $\Delta$ </td></tr><tr><td></td><td>Outcome RL</td><td>60.8</td><td>51.8</td><td>60.3</td><td>+0.5</td></tr><tr><td>Incentive</td><td>+ routed initial-answer credit  $t ^ { \star }$ </td><td>61.6</td><td>52.7</td><td>61.0</td><td>+0.6</td></tr><tr><td>Drift</td><td>+ cross-entropy anchor at</td><td>62.4</td><td>54.0</td><td>62.3</td><td>+0.1</td></tr><tr><td>Incentive</td><td>+ floor multiplier</td><td>62.2</td><td>53.6</td><td>62.4</td><td>-0.2</td></tr><tr><td>Drift</td><td>+ KL at t*</td><td>62.3</td><td>54.1</td><td>62.3</td><td>0.0</td></tr><tr><td>Incentive</td><td>+ transition utility</td><td>62.3</td><td>53.8</td><td>62.0</td><td>+0.3</td></tr><tr><td>Interference</td><td>+ asymmetric gradient protection</td><td>63.4</td><td>55.8</td><td>63.2</td><td>+0.2</td></tr><tr><td>All</td><td>+ later-stage KL / sampling§, no projection</td><td>64.0</td><td>55.8</td><td>63.5</td><td>+0.5</td></tr><tr><td>All</td><td>Full CoFIRE 8</td><td>64.5</td><td>57.3</td><td>64.2</td><td>+0.3</td></tr></table>

Table 9: Self-reported signals on Video-MME (COFIRE). AUC of each signal for the stated binary outcome; 0.5 is chance.
<table><tr><td>Signal</td><td>Outcome predicted</td><td>AUC</td></tr><tr><td>Gate confidence</td><td>Initial answer correct</td><td>0.68</td></tr><tr><td>Retrieved-frame option belief</td><td>FIX VS. BREAK (SEEK)</td><td>0.53</td></tr></table>

## G Qualitative Examples

Figures 5–8 show four Video-MME trajectories of the checkpoint reported in Table 2, in the format of Figure 2. Model outputs are verbatim and shortened with $\cdots { \ r } ;$ bold marks the key sentence. Thumbnails are the frames at the logged timestamps; $S _ { \mathrm { p r e } }$ thumbnails are first-look frames. Timeline ticks mark first-look and retrieved frames, while the 2 fps preview spans the whole video. Dashed boxes mark the evidence. We selected the three fixes from the 56 FIX cases among the 424 SEEK questions. Each was kept only if the revision cites the newly retrieved frames, and those frames show what the revision claims. We excluded fixes that were correct only by guessing. The fourth example is one of the 47 BREAK cases. Each figure also reports the Outcome RL baseline on the same question.

Fixes. In Figure 5, $S _ { \mathrm { p r e } }$ misreads its own first look: the scoreboard at 44.0 s lists CHN first, but $S _ { \mathrm { p r e } }$ reports KOR and describes the skater at 48.5 s as wearing a “KOREA” jersey, while the frame shows a Chinese skater. The gate re-reads the scoreboard, assigns confidence 0.12, and retrieves dense frames around the finish, which confirm CHN. In Figure 6, the 19-minute video shows the injured athlete only briefly, and no bib is legible in the coarse preview or the first look, so $S _ { \mathrm { p r e } }$ guesses from the uniform color. The gate states exactly what is missing (“I cannot clearly read a bib number”), and all 12 retrieved frames fall in the 33-second treatment segment, where the bib “377” becomes visible. In Figure 7, $S _ { \mathrm { p r e } }$ answers from a language prior, the documentary title Cocainefor Germany. The gate notes that the video concerns Dutch cartels, and retrieval on the 42-minute video returns the full headline of the article that the first look showed only partially. In Figures 5 and 6, the Outcome RL baseline instead trusts its wrong initial answer with confidence 0.90 and 0.70.

A failure. Figure 8 shows a BREAK. $S _ { \mathrm { p r e } }$ infers the Los Angeles–New York route from the map at 44.5 s and answers correctly (3,224 miles), but this answer rests on prior knowledge rather than an on-screen figure, so the gate correctly reports low confidence. The query “total miles traveled by the main character” retrieves a salient graphic that matches the option text: “254,000 MILES” over a map labeled “1916” and “freight and passenger rail”. The revision reads the graphic correctly but attributes the 1916 network length to the traveler and switches to A. The perception is correct, but the referent is wrong. The prompt rule “keep your initial answer unless the new evidence clearly contradicts it” does not prevent this error, because the retrieved number matches an option verbatim and therefore appears decisive. The Outcome RL baseline retrieves the same frames and makes the same error. A grounded initial answer cannot be broken this way, but this initial answer was correct without on-screen grounding. This suggests that the BREAK cost in Equation 4 is a backstop and does not replace checking whether retrieved evidence answers the question that was asked.

Algorithm 1 Confident First, Reflective Second (COFIRE)   
Require: Question pool D with $( y _ { b } , r _ { b } )$ cached by frozen π<sub>base</sub>; initialization θ<sub>0</sub> and its frozen copy π<sub>ref</sub>   
Require: Group size G; tolerance ϵ; weights $\alpha , \mu , \kappa _ { 1 } , \kappa _ { 2 } ;$ dual settings η<sub>λ</sub>, λ<sub>max</sub>   
Require: Projection tensor subset P; stabilizer ε<sub>proj</sub>   
Ensure: Trained shared policy π<sub>θ</sub>   
1: $\theta  \theta _ { 0 } ; \lambda  0 ;$ initialize $\widehat { J } _ { 0 } , \widehat { J } _ { \mathrm { b a s e } }$ at their first batch estimates   
2: for each training batch do   
3: Sample questions $\begin{array} { r } { B \sim \mathcal { D } _ { w } ; } \end{array}$ use reference-error weight 2 in the full configuration   
4: for each question $x \in B$ do   
5: Sample G trajectories $\tau _ { i } = ( S _ { \mathrm { p r e } } ^ { ( i ) } , S _ { \mathrm { g a t e } } ^ { ( i ) } , S _ { \mathrm { r e v } } ^ { ( i ) } )$ from π<sub>θ</sub>   
6: Extract $( y _ { 0 , i } , m _ { i } , y _ { f , i } )$ and initial-answer positions $t _ { i } ^ { \star }$   
7: Compute $r _ { 0 , i }$ and u<sub>i</sub> using task labels and Equation 4   
8: Assign $( 1 + \lambda )$ ctr(r<sub>0,i</sub>) to $S _ { \mathrm { p r e } } ^ { ( i ) }$ and ctr(u<sub>i</sub>) to $S _ { \mathrm { g a t e } } ^ { ( i ) } , S _ { \mathrm { r e v } } ^ { ( i ) }$   
9: end for   
10: Add format and gate-effort shaping; construct clipped stage surrogates   
11: Compute ${ \mathcal { L } } _ { \mathrm { d e c } }$ and $\mathcal { L } _ { \mathrm { l a t e } }$ using π<sub>ref</sub>   
12: $\mathcal { L } _ { \mathrm { p } } \stackrel { \cdot } {  } \mathcal { L } _ { \mathrm { p r e } } + \mathcal { L } _ { \mathrm { d e c } }$   
13: $\mathcal { L } _ { \mathrm { r } }  \dot { \mathcal { L } } _ { \mathrm { g a t e } } + \mathcal { L } _ { \mathrm { r e v } } + \mathcal { L } _ { \mathrm { l a } }$ te   
14: $g _ { \mathrm { p } } ^ { \mathcal { P } }  \nabla _ { \theta _ { \mathcal { P } } } \mathcal { L } _ { \mathrm { p } } ; \mathrm { ~ } g _ { \mathrm { r } } ^ { \mathcal { P } }  \nabla _ { \theta _ { \mathcal { P } } } \mathcal { L } _ { \mathrm { r } }$   
15: $\mathcal { \hat { \omega } } \gets \mathrm { s t o p g r a d } \left[ \operatorname* { m i n } \left( 0 , \frac { \langle g _ { \mathrm { r } } ^ { \mathcal { P } } , g _ { \mathrm { p } } ^ { \mathcal { P } } \rangle } { \| g _ { \mathrm { p } } ^ { \mathcal { P } } \| ^ { 2 } + \varepsilon _ { \mathrm { p r o j } } } \right) \right]$   
16: ${ \mathcal { L } } _ { \mathrm { u p d a t e } } \gets ( 1 - \overline { { \alpha } } \widehat { \omega } ) \dot { \mathcal { L } } _ { \mathrm { p } } + \alpha { \mathcal { L } } _ { \mathrm { r } }$   
17: Update θ with the configured optimizer using $\nabla _ { \theta } \mathcal { L } _ { \mathrm { u p d a t e } }$   
18: Update ${ \widehat { J } } _ { 0 } , { \widehat { J } } _ { \mathrm { b a s e } }$ using Equations 11–12   
19: $\lambda  \mathrm { c l i p } \big ( \lambda + \eta _ { \lambda } ( \widehat { J } _ { \mathrm { b a s e } } - \epsilon - \widehat { J } _ { 0 } ) , 0 , \lambda _ { \mathrm { m a x } } \big )$   
20: end for   
21: return π

![](images/c44770d4f95cc5f64f3518afec250d37eca96beb6d13e4c48ece4b5a0ca5308e.jpg)  
Figure 5: Fix: the gate catches a misread first-look frame (Video-MME 180-3, 66 s video). S<sub>pre</sub> reports KOR from a scoreboard that lists CHN first; the gate re-reads it, and dense retrieval around the finish confirms the revision to C.

![](images/112deb8447805d1a017080f87bd263ed24c0f41b86ee138a5dc07a923f290a24.jpg)  
Figure 6: Fix: localizing a few seconds of evidence in a long video (Video-MME 766-1, 19 min). S<sub>pre</sub> guesses the athlete from the uniform color; the gate requests the bib, and all retrieved frames fall in the treatment segment, where the bib reads 377.

![](images/74c609d2479d001abc7891f2957ae425eb58274e6fffd4cc22789ee20b720755.jpg)  
Figure 7: Fix: replacing a language prior with on-screen text (Video-MME 713-3, 42 min). $S _ { \mathrm { p r e } }$ answers from the documentary title; retrieval returns the full headline attributing the threat to Dutch drug gangs.

![](images/727c0a7f60a9dc2d91af1b093d3bdf8d0457dbbc6433a56ab7b8d88e5c763c26.jpg)  
Figure 8: Failure: a correct initial answer broken by a salient but irrelevant number (Video-MME 568-1, 13 min). $S _ { \mathrm { p r e } }$ is correct but not grounded in an on-screen figure; the query retrieves a 1916 rail-network statistic that matches option A verbatim, and the revision attributes it to the traveler.