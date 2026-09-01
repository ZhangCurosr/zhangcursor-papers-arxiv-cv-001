# CANVAS: Consistency-Aware Navigation via Visual Adaptive Sampling for Long-Context Text-to-SVG Generation

Yichen Wu<sup>1</sup> Haoxuan Qu<sup>2</sup> Yihang Lou<sup>3</sup> Hossein Rahmani<sup>2</sup> Jun Liu<sup>2</sup> <sup>1</sup>University of Nottingham <sup>2</sup>Lancaster University <sup>3</sup>Peking University

Code: github.com/Louis-YW/CANVAS

## Abstract

Autoregressive large models have recently advanced Textto-SVG generation from simple icons to complex, longcontext graphics, yet standard autoregressive decoding often fails to maintain global consistency across geometry, layout, occlusion, and composition. We introduce CAN-VAS (Consistency-Aware Navigation via Visual Adaptive Sampling), a training-free, render-aware inference framework that combines power-sharpened trajectory likelihood with visual feedback from rendered futures and derives a stroke-wise navigation rule. It effectively estimates each candidate stroke’sfuture value under a limited generation and rendering budget and adaptively allocates samples according to candidate uncertainty, decision influence, and rollout cost. Experiments across multiple autoregressive SVG backbones and complementary benchmarks demonstrate improvements in global consistency, which includes sound geometric relationships, spatial layouts, occlusion ordering, and overall composition, without additional training, demonstrating the effectiveness and generalization ability of our framework.

## 1. Introduction

Text-to-SVG generation aims to generate scalable, structurally editable vector graphics from natural-language descriptions, has broad applications in icon design, digital illustration, and web content creation, and has thus attracted increasing research attention [3, 16, 18, 20, 21, 24, 26]. In recent years, with advances in multimodal large language models in code generation and visual understanding, the direct autoregressive generation of SVG code with large models, as a mainstream technical route for this task, has expanded from simple icons to complex graphics containing numerous paths, coordinates, and style attributes [16, 24, 26, 28].

However, the ability to generate longer SVG sequences does not mean that a model can stably produce complete images with good global consistency, which includes sound geometric relationships, spatial layouts, occlusion ordering, and overall composition, all of which are important to the generation quality of complex SVGs [3, 12, 18, 19, 21, 24– 26, 28]. This is because achieving these properties depends on a large number of interdependent drawing decisions. However, standard autoregressive decoding selects only according to the conditional probability at the current position. Therefore, a locally high-probability decision may still lead to a low-quality complete SVG, whereas a globally better generation trajectory may be eliminated early because its early local probabilities are lower and cannot be recovered during subsequent generation.

Such problems do not necessarily mean that the base model has not learned the structure of long-sequence SVGs. Models trained on large-scale SVG data already contain structural priors about path combinations, stylistic relationships, and complete programs, but standard autoregressive decoding may not fully reflect the model’s preference over complete generation trajectories. Therefore, we study how to use these existing capabilities more effectively at inference time, without modifying the model parameters, to improve the consistency of long-context SVG generation.

However, relying only on the base model’s token probabilities remains insufficient, because they mainly reflect the code distribution and generation preferences learned from the training data rather than an explicit evaluation of the final rendering, and thus cannot directly determine whether the shapes, positions, occlusion relationships, and overall composition in the final rendering are consistent. Even if an SVG trajectory has high model probability and can be parsed successfully, its rendering may still exhibit element misalignment, occlusion conflicts, or compositional imbalance. Motivated by prior studies that incorporate rendered visual feedback into SVG generation [12, 19], we retain the structural priors learned by the base model while leveraging rendered outputs to provide direct visual feedback.

To this end, we propose CANVAS (Consistency-Aware Navigation via Visual Adaptive Sampling), a training-free, render-aware decoding framework for long-context Textto-SVG generation. Here, “navigation” refers to selecting generation trajectories in the SVG program space rather than following locally probable token decisions alone. At the complete-sequence level, CANVAS constructs a unified objective that combines the base model’s preference over complete SVG trajectories with the visual quality of the final rendering. Inspired by professional artists, who consider the global composition when deciding how to place each stroke, we reformulate the complete-sequence objective as a stroke-wise future-aware navigation rule: at each complete drawing boundary identified by an incremental parser, CANVAS navigates among candidate strokes (i.e., newly appended SVG drawing units that are parser-complete and directly renderable) according to their selection probabilities and the overall quality of the future complete SVGs they may produce.

Practical SVG applications also demand efficient inference, as users often need to generate, compare, and modify multiple candidate SVGs [3, 28]. However, exactly computing the future value of each stroke requires enumerating a large number of complete SVG continuation trajectories, making the associated computational cost prohibitive. To make this navigation computationally practical, CANVAS uses finite-sample rollouts to obtain conditionally unbiased estimates of finite-horizon future values and adaptively allocates generation and rendering budgets according to the uncertainty, decision influence, and sampling cost of different candidate strokes. Under theoretical guarantees, this improves the reliability of the estimates under a limited computational budget.

Figure 1 compares CANVAS with Best-of-N decoding. Even after the unchanged backbone is sampled N = 5 times and the best completed SVG is selected, CANVAS still exhibits better global consistency, which includes sound geometric relationships, spatial layouts, occlusion ordering, and overall composition.

In summary, our main contributions are as follows: 1) We propose CANVAS, a training-free, render-aware inference framework for consistency-aware navigation in long-context Text-to-SVG generation. 2) We transform the joint modeland-visual objective at the complete-SVG level into a strokewise future-aware navigation rule and propose a reliable visual adaptive sampling method with a finite horizon and adaptive budget allocation. 3) Experiments on multiple representative autoregressive SVG generators and complementary benchmarks verify the effectiveness and generalization ability of our method.

## 2. Related Work

Text-to-SVG generation. Earlier work on this task often adopted optimization-based approaches, synthesizing text-guided vector graphics by directly optimizing vector primitives through differentiable rasterization under visionlanguage or text-to-image diffusion guidance [6, 8, 22, 23].

With the advent of MLLMs, recent works mainly follow autoregressive paradigm, tokenizing SVG code and predicting the resulting tokens sequentially [3, 17, 18, 20, 21, 24, 26, 28]. Earlier autoregressive methods primarily explore representations that make SVG sequences easier to model: IconShop [20] sequentializes SVG paths into uniquely decodable tokens, StrokeNUWA [17] compresses graphics into semantically meaningful stroke tokens, and SVGBuilder [3] generates colored graphics with a component-based representation. Vector Grimoire [4] learns a discrete codebook of vector shapes from raster supervision. Chat2SVG [21] uses an LLM to produce an SVG template and then refines its geometry through diffusion-guided optimization.

More recent Text-to-SVG methods seek to improve generation of complex graphics and longer output sequences. They commonly construct large long-context SVG training datasets [1, 18, 19, 25, 28]. vHector [28] introduces domainspecific tokens. IntroSVG [19] trains a unified generator and critic and performs an iterative generate, critique, and refine loop using rendered feedback. Some also introduce Chainof-Thought (CoT) in supervision [1, 18, 25]. SVGen [18] combines curriculum learning with integrity and path-count rewards, while SVGThinker [1] aligns CoT supervision with incremental rendering, and Reason-SVG [25] introduces Drawing-with-Thought with hybrid-reward reinforcement learning.

Together, these methods modify the training data, representation, supervision, or inference loop.

Different from existing studies, to the best of our knowledge, CANVAS is the first training-free inference framework to improve the global consistency of SVGs generated by diverse autoregressive backbones.

Power-distribution sampling. Power-distribution sampling globally sharpens a model’s sequence distribution, allowing inference to favor trajectories that are probable as complete sequences rather than relying only on local token probabilities. Karan and Du [11] show that this principle can elicit strong reasoning from base LLMs without additional training, and it has subsequently been extended to long-horizon robotic planning [2]. Sampling for Quality [14] augments the base sequence distribution with prefixdependent reward potentials, enabling model likelihood and sequence-level quality to jointly guide decoding. Different from existing works, CANVAS makes decisions at the stroke level by combining each candidate’s power-sharpened model probability with the visual quality of the future complete SVGs it may lead to.

## 3. Method

To improve global consistency in long-context SVG generation during inference time, a key challenge is that standard autoregressive decoding acts only on local token probabilities and cannot anticipate how a current choice will affect the final canvas. To tackle this challenge, we propose CAN-VAS, a novel framework that makes stroke-wise decisions by combining power-sharpened model preferences with visual evidence from the rendered futures that each candidate stroke may produce. Figure 2 provides an overview of CANVAS.

![](images/4fd80e9c278fb0218efd25e360e1e978d62613a2d7af2df09c68f5cdd3295f37.jpg)  
Figure 1. Illustration of SVGs generated with the same vHector-8B backbone by Best-of-N and CANVAS at inference time. CANVAS shows better global consistency, which includes sound geometric relationships, spatial layouts, occlusion ordering, and overall composition. The prompts from left to right are: “The image depicts a cartoon-style illustration featuring a person and a laboratory setup.”; “The image depicts a traffic sign with a green background and white text and symbols.”; “The image depicts a mobile phone with a sleek, modern design.”; and “The image depicts a highly intricate and symmetrical geometric pattern, specifically a star polygon, which is a polygon with multiple vertices and edges that are connected in a star-like shape.”

Sec. 3.1 defines a complete-SVG target that combines trajectory-level model likelihood with rendered visual quality. Sec. 3.2 derives its parser-aligned stroke-wise navigation rule, Sec. 3.3 replaces the intractable candidate and future spaces with finite sampling, and Sec. 3.4 allocates a limited generation and rendering budget across candidates.

## 3.1. Complete-SVG Target

Given a natural-language prompt q, the base model generates a complete SVG program $\boldsymbol { Y } = ( y _ { 1 } , \dots , y _ { T ( Y ) } )$ with probability

$$
p _ { \theta } ( Y \mid q ) = \prod _ { t = 1 } ^ { T ( Y ) } p _ { \theta } ( y _ { t } \mid y _ { < t } , q ) .\tag{1}
$$

Inspired by Context-Aware Power Sampling (CAPS) [2], we first investigate whether power sharpening of this model’s joint trajectory likelihood can improve global consistency in long-context SVG generation. However, as shown by the comparison in Appendix K.3, this model-likelihood-only approach remains insufficient. We attribute this limitation to the fact that model likelihood cannot determine whether the rendered SVG has correct geometry, occlusion relations, and overall composition. Direct visual feedback is therefore necessary; maximizing it alone, however, could exploit evaluator errors and abandon the SVG prior. Effectively combining these two sources of information without discarding either one is the first challenge. Let $p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( Y \mid q ) \propto p _ { \theta } ( Y \mid q ) ^ { \alpha }$ with $\alpha \geq 1$ , denote the power-sharpened model distribution. Inspired by the KL-regularized preference-optimization principle underlying DPO [15], CANVAS retains the powersharpened generator as a frozen prior while incorporating rendered visual quality as an inference-time reward. We seek a visually improved distribution through the following KL-regularized objective [14]:

$$
\begin{array} { r l } & { \Pi _ { \alpha , \beta } ( \cdot \mid q ) = \arg \underset { \pi \in \Delta ( \mathcal { Y } _ { q } ) } { \operatorname* { m a x } } ~ \mathcal { I } _ { \alpha , \beta } ( \pi ) , } \\ & { \quad ~ \mathcal { I } _ { \alpha , \beta } ( \pi ) = \beta \mathbb { E } _ { Y \sim \pi } [ s ( \mathcal { R } ( Y ) , q ) ] } \\ & { \quad \quad \quad - D _ { \mathrm { K L } } ( \pi ( \cdot \mid q ) \| p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( \cdot \mid q ) ) , \qquad \beta \ge 0 . } \end{array}\tag{2}
$$

Here, R(Y) renders program Y, and $s ( \mathcal { R } ( Y ) , q ) \in [ 0 , 1 ]$ measures its visual quality relative to prompt q. The first term favors visually coherent complete renderings, whereas the KL term penalizes departure from the model’s powersharpened trajectory distribution. The objective therefore improves final-render quality while retaining the SVG syntax, structure, and complete-program preferences learned by the base model. Its unique optimizer has the closed form

$$
\begin{array} { r l } { \Pi _ { \alpha , \beta } ( Y \mid q ) = \cfrac { 1 } { \widetilde { Z } _ { \alpha , \beta } ( q ) } p _ { \theta } ( Y \mid q ) ^ { \alpha } } \\ { \times \exp \{ \beta s ( \mathcal { R } ( Y ) , q ) \} , \quad } & { Y \in \mathcal { V } _ { q } , } \end{array}\tag{3}
$$

![](images/6518cbd5d28f0ba22f3032c73c440c4376b16175de17f98b4650eb6af268fa7b.jpg)  
Figure 2. Overview of CANVAS. Long-context SVG decoding is difficult because a locally plausible stroke can lead to globally inconsistent geometry, occlusion, and composition, while exhaustively scoring all complete futures is prohibitive in generation and rendering costs. CANVAS addresses this challenge with a novel training-free, render-aware stroke-wise navigation mechanism: at each parser-complete boundary, it samples candidate strokes, evaluates their future rendered trajectories with a joint model-and-visual score, adaptively concentrates rollout budget on uncertain and decision-critical candidates, and commits one stroke before iterating with the fixed base generator.

where $\widetilde { Z } _ { \alpha , \beta } ( q )$ normalizes the distribution over valid, terminated SVGs $\mathcal { V } _ { q }$ . The factor $p _ { \theta } ( Y \mid q ) ^ { \alpha }$ preserves and strengthens the model’s preference over complete programs, while exp $\{ \beta s ( \mathcal { R } ( Y ) , q ) \}$ reweights them according to finalrender quality. Thus, the target favors complete SVGs that are both plausible to the generator and visually consistent after rendering. The detailed derivation from the KL-regularized objective to the Gibbs distribution above is provided in Appendix C.

## 3.2. Consistency-Aware Stroke-wise Navigation

Eq. 3 specifies how probability mass should be assigned to complete, terminated SVG programs, but it does not directly provide an online decision rule. During decoding, we must decide which candidate generation unit to choose before the future suffix is generated, evaluating the set of possible complete SVG trajectories that each candidate unit may induce. Inspired by the global-to-step-wise reformulation of Scalable Power Sampling [9], but using a distinct renderaware objective, we first consider token-wise decisions; these require prohibitively frequent rollouts (Appendix K.6). A naive way to reduce this cost is to replace token-wise decoding with fixed-length blocks. Although this reduces the number of decisions, block boundaries may fall inside a coordinate, path command, or XML attribute and thus fail to represent a complete visual operation. Inspired by path-level SVG representations and step-wise path-fragment generation [12, 20], CANVAS instead parses the SVG token stream as it is generated and places a decision boundary whenever a complete renderable element (hereafter called a stroke) has been formed. We operationally define a stroke as a parsercomplete SVG drawing unit, either a <path> element or another directly renderable primitive, such as <circle> or <rect>. More details about the mathematical forms for token-wise, fixed-length, and parser-detected blocks are provided in Appendix B.

Each stroke-boundary prefix $H _ { k }$ then corresponds to a visually meaningful partial rendering $\mathcal { R } ( H _ { k } )$ . Let $h = H _ { k - 1 }$ be the current SVG prefix, let b be a candidate next stroke, and write $H _ { k } = h b$ after committing it. To measure only the visual change caused by $b ,$ rather than repeatedly rewarding the existing canvas and thereby favoring programs with more strokes, we define the incremental render potential

$$
\begin{array} { r } { \psi _ { k } ( H _ { k } , q ) = \exp \left\{ \beta \left[ s ( \mathcal { R } ( H _ { k } ) , q ) - s ( \mathcal { R } ( H _ { k - 1 } ) , q ) \right] \right\} _ { \binom { d } { d } } . } \end{array}\tag{4}
$$

Multiplying these incremental factors over all strokes gives

$$
\prod _ { k = 1 } ^ { K ( Y ) } \psi _ { k } ( H _ { k } , q ) = \exp \{ \beta \left[ s ( \mathcal { R } ( Y ) , q ) - s ( \mathcal { R } ( H _ { 0 } ) , q ) \right] \} .\tag{5}
$$

Each factor measures only the visual change introduced by one stroke. Multiplying them cancels the intermediate canvas scores, leaving only the visual-reward term for the completed SVG in Eq. 3, apart from an initial-canvas constant shared by all candidates.

Because CANVAS chooses a candidate stroke based not only on the model’s preference for the current drawing operation but also on the future value of the complete SVG trajectories reachable after that stroke, we divide the remaining target weight of a continuation composed of the current stroke b and future suffix C into two parts: one part is the model preference and immediate visual change determined by b, and the other is the total value of all complete continuations that b may induce. Thus, the complete-trajectory target can be converted into the following online stroke-wise navigation rule:

$$
\boxed { \Pi _ { \alpha , \beta } ( B _ { k } = b \mid H _ { k - 1 } = h , q ) \propto u _ { k } ( b ) V _ { k } ( h b , q ) } .\tag{6}
$$

where:

$$
\begin{array} { c } { { u _ { k } ( b ) = P _ { \theta } ^ { \mathrm { s t r o k e } } ( b \mid h , q ) ^ { \alpha } \psi _ { k } ( h b , q ) , } } \\ { { V _ { k } ( h b , q ) = \displaystyle \sum _ { C : h b C \in \mathcal { V } _ { q } } p _ { \theta } ( C \mid h b , q ) ^ { \alpha } \Phi _ { > k } ( h b , C , q ) , } } \\ { { { } } } \\ { { \Phi _ { > k } ( h b , C , q ) = \displaystyle \prod _ { r = k + 1 } ^ { K ( h b C ) } \psi _ { r } ( H _ { r } , q ) . } } \end{array}\tag{7}
$$

Here, $P _ { \theta } ^ { \mathrm { s t r o k e } } ( b \mid h , q )$ is the probability of generating candidate stroke b under the current condition, while $p _ { \theta } ( C \mid h b , q )$ is the model probability of generating suffix C after b. The current-stroke factor $u _ { k } ( b )$ combines the model’s preference for the current stroke with its immediate visual change, whereas the future value $V _ { k } ( h b , q )$ marginalizes the model preferences and visual effects of all complete continuations after that stroke. More derivation details are provided in Appendix D.

## 3.3. From Exact Expectations to Finite Sampling

However, evaluating the future value $V _ { k } ( h b , q )$ requires considering a large number of possible complete continuations for every candidate stroke. As shown in Appendix K.1, doing so incurs prohibitive generation and rendering costs. Practical SVG applications require users to rapidly generate, compare, and modify multiple candidate results [3, 28], making complete continuation evaluation impractical. To make the future value estimable, we express it as an expectation under the base-model continuation distribution:

$$
\begin{array} { l } { { { \displaystyle V _ { k } ( h b , q ) = \sum _ { C : h b C \in \mathcal { S } _ { q } } p _ { \theta } ( C \mid h b , q ) } } } \\ { { \displaystyle ~ \times \left[ p _ { \theta } ( C \mid h b , q ) ^ { \alpha - 1 } \Phi _ { > k } ( h b , C , q ) \right] } } \\ { { \displaystyle ~ = \mathbb { E } _ { C \sim p _ { \theta } ( \cdot \mid h b , q ) } \Big [ } } \\ { { \displaystyle p _ { \theta } ( C \mid h b , q ) ^ { \alpha - 1 } } } \\ { { \displaystyle ~ \times \Phi _ { > k } ( h b , C , q ) \Big ] . } } \end{array}\tag{8}
$$

This expectation admits an unbiased Monte Carlo estimator. We first sample candidates from the parser-valid next-stroke proposal $q _ { k }$ and compute their importance factors:

$$
\begin{array} { r l } { B _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } q _ { k } ( \cdot \mid h , q ) , } & { { } \quad i = 1 , \ldots , L , } \\ { A _ { i } = \displaystyle \frac { u _ { k } ( B _ { i } ) } { q _ { k } ( B _ { i } \mid h , q ) } . } \end{array}\tag{9}
$$

Here, $q _ { k }$ is the parser-valid proposal induced by $P _ { \theta } ^ { \mathrm { s t r o k e } }$ , L is the number of sampled candidates, and $A _ { i }$ is the importance weight for candidate $B _ { i }$ . The exact first-hit stopping rule is given in Appendix B. The resulting importance-weighted average is an unbiased estimate of the exact unnormalized stroke mass:

$$
\begin{array} { r l } {  { \mathbb { E } [ \frac { 1 } { L } \sum _ { i = 1 } ^ { L } A _ { i } V _ { k } ( h B _ { i } , q ) ] = \mathbb { E } _ { B \sim q _ { k } } [ \frac { u _ { k } ( B ) V _ { k } ( h B , q ) } { q _ { k } ( B \mid h , q ) } ] } \quad } & { } \\ & { = \displaystyle \sum _ { b \in \mathcal { B } ( h ) } u _ { k } ( b ) V _ { k } ( h b , q ) . } \end{array}\tag{10}
$$

At this point, candidate-stroke sampling is complete, but estimating the future value of each candidate remains challenging. As shown in Tabs. 4, the time cost grows with the rollout horizon, whereas rolling out every sample to its end $( \mathrm { i . e . }$ , its EOS token) incurs unacceptable overhead. For each candidate $B _ { i }$ , we therefore draw $M _ { i }$ independent continuations, each containing at most H future strokes. Let $C _ { 1 : H } ^ { ( m ) }$ denote the continuation from rollout $m _ { : }$ , and let $H _ { k + r } ^ { ( m ) }$ denote the prefix after its first r future strokes. In rollout m, $R _ { i } ^ { ( m ) }$ denotes the number of future strokes included in the rollout: it is H if EOS is not reached within the horizon, and a smaller value if EOS is reached earlier. We define

$$
\begin{array} { l } { { \displaystyle Z _ { i , H } ^ { ( m ) } = p _ { \theta } ( C _ { 1 : H } ^ { ( m ) } \mid h B _ { i } , q ) ^ { \alpha - 1 } } } \\ { { \displaystyle \qquad \mathrm { m i n } ( H , R _ { i } ^ { ( m ) } ) } } \\ { { \displaystyle \qquad \sum _ { r = 1 } \qquad \psi _ { k + r } ( H _ { k + r } ^ { ( m ) } , q ) } , } \\ { { \displaystyle \widehat V _ { i , H } = \frac { 1 } { M _ { i } } \sum _ { m = 1 } ^ { M _ { i } } Z _ { i , H } ^ { ( m ) } } , } \\ { { \displaystyle \widehat \rho _ { i } = \frac { A _ { i } \widehat V _ { i , H } } { \sum _ { j = 1 } ^ { L } A _ { j } \widehat V _ { j , H } } . } } \end{array}\tag{11}
$$

Here, $Z _ { i , H } ^ { ( m ) }$ is the contribution of rollout m for candidate $B _ { i }$ $\widehat { V } _ { i , H }$ is the sample-mean estimate of its finite-horizon future value, and $\widehat { \rho } _ { i }$ is the normalized selection weight assigned to $B _ { i }$ . Conditioned on $B _ { i } , \ \hat { V _ { i , H } }$ is an unbiased estimate of the finite-horizon future value when $M _ { i }$ is fixed before the fresh rollouts are observed. Together with candidate importance sampling, this gives an unbiased estimate of the finite-horizon unnormalized target mass. More details about rollout-based future-value estimation are provided in Appendix E, while details about importance approximation over the finite candidate space are provided in Appendix F.

## 3.4. Visual Adaptive Sampling under a Limited Budget

At this stage, candidate strokes differ in both the uncertainty of their future values and the cost of estimating them. Assigning the same rollout count to all candidates wastes computation on stable branches and may undersample candidates whose estimates most affect the normalized stroke decision. To tackle this problem, CANVAS allocates the budget according to rollout uncertainty, decision influence, and per-rollout cost.

To implement this allocation, we first need candidatespecific estimates of future value, rollout variance, and per-rollout generation-and-rendering cost. A naive strategy would use the same rollout samples both for this diagnosis and to form the final future-value estimate. This creates selection dependence because the same random fluctuation can influence both the allocated rollout count and the final estimate. CANVAS avoids this dependence by first using m pilot rollouts per candidate to estimate its future value $\widehat { V } _ { i } ^ { ( 0 ) }$ , rollout variance ${ \widehat { \sigma } } _ { i } ^ { 2 }$ , and per-rollout generation-andrendering cost $\widehat { c } _ { i }$ . Then it freezes the allocation and evaluates each candidate with independent fresh rollouts. More details about the derivation of the adaptive rollout allocation are given in Appendix G, and more implementation details of the CANVAS decoding procedure and its error decomposition are given in Appendix J.

Let $\begin{array} { r } { V _ { i } = V _ { i , H } , \sigma _ { i } ^ { 2 } = \operatorname { V a r } ( Z _ { i , H } \mid B _ { i } ) , D = \sum _ { i } A _ { i } V _ { i } } \end{array}$ and $\rho _ { i } = A _ { i } V _ { i } / D .$ . A first-order delta analysis of the normalized map gives the decision-error approximation and its optimal continuous allocation:

$$
\begin{array} { l } { { \displaystyle { \mathbb E } \| \hat { \rho } - \rho \| _ { 2 } ^ { 2 } \simeq \sum _ { i = 1 } ^ { L } \frac { d _ { i } } { M _ { i } } , } } \\ { { \displaystyle d _ { i } = \frac { A _ { i } ^ { 2 } \sigma _ { i } ^ { 2 } } { D ^ { 2 } } \| { \bf e } _ { i } - \rho \| _ { 2 } ^ { 2 } } , } \\ { { \displaystyle M _ { i } ^ { * } = \frac { C _ { k } \sqrt { d _ { i } / c _ { i } } } { \sum _ { j = 1 } ^ { L } \sqrt { d _ { j } c _ { j } } } . } } \end{array}\tag{12}
$$

Here, $C _ { k }$ is the fresh-rollout budget for the current decision, $c _ { i }$ is candidate $i \ ' _ { \mathrm { { s } } }$ per-rollout cost, and $\mathbf { e } _ { i }$ is its coordinate vector. In practice, we round the continuous allocation to obtain integer rollout counts.

The allocation above minimizes the first-order decision error, but it remains to determine whether adapting the rollout counts provides a real advantage over assigning the same count to every candidate. Under the same generation-andrendering budget, the two error surrogates satisfy

$$
\begin{array} { r l } & { \mathcal { E } _ { \mathrm { a d a p t i v e } } = \frac { \left( \sum _ { i } \sqrt { c _ { i } d _ { i } } \right) ^ { 2 } } { C _ { k } } } \\ & { \qquad \leq \frac { \left( \sum _ { i } d _ { i } \right) \left( \sum _ { i } c _ { i } \right) } { C _ { k } } = \mathcal { E } _ { \mathrm { u n i f o r m } } . } \end{array}\tag{13}
$$

where the inequality follows from Cauchy–Schwarz and is strict unless $d _ { i } / c _ { i }$ is identical across candidates. Thus, under the same budget, our future-uncertainty-aware allocation is theoretically guaranteed to be no worse than uniform allocation.

In addition, a further difficulty remains: even when every $\widehat { V } _ { i , H }$ is conditionally unbiased, the normalized decision in Eq. 11 divides by a random denominator, and generally $\mathbb { E } [ X / Y ] \neq \mathbb { E } [ X ] / \bar { \mathbb { E } } [ Y ]$ . Here, to address this normalization error, we use a second-order expansion of the ratio map. For a bounded candidate statistic $f ,$ let $Y _ { i } = A _ { i } \widehat { V } _ { i , H }$ $X _ { i } ( f ) = f ( B _ { i } ) Y _ { i }$ , and let $\overline { { Y } } , \overline { { X } } _ { f } , s _ { Y } ^ { 2 }$ , and $s _ { X Y } ( f )$ denote their sample means, denominator variance, and numerator– denominator covariance. Using these observed second-order statistics, CANVAS subtracts the leading ratio-bias term:

$$
\widetilde { T } _ { k } ( f ) = \frac { \overline { { X } } _ { f } } { \overline { { Y } } } - \frac { 1 } { L } \left( \frac { \overline { { X } } _ { f } s _ { Y } ^ { 2 } } { \overline { { Y } } ^ { 3 } } - \frac { s _ { X Y } ( f ) } { \overline { { Y } } ^ { 2 } } \right) .\tag{14}
$$

Proposition 1 (Leading bias correction) Let M = min<sub>i</sub> $M _ { i }$

$$
\operatorname* { l i m } _ { L , M \to \infty } L \Big ( \mathbb { E } [ \widetilde { T } _ { k } ( f ) ] - T _ { k , H } ( f ) \Big ) = 0 .\tag{15}
$$

This correction mitigates the leading bias introduced by the random denominator when normalizing noisy estimates.

The proposition applies to any fixed bounded statistic f of a candidate stroke. More details about the regularity conditions for Proposition 1 are provided in Appendix H. More details about the finite-candidate importance approximation and the unbiased estimation of unnormalized masses are provided in Appendix F. More details about the ratio-bias derivation and the unequal fresh-rollout analysis are provided in Appendix H. More details about failed rollouts and numerical fallback rules are provided in Appendix I.

In summary, CANVAS converts a joint model-and-render complete-SVG target into a parser-aligned, future-aware stroke navigation rule, and reliably approximates it under a limited generation and rendering budget using finite-horizon rollouts and influence- and cost-aware visual adaptive sampling.

Table 1. Quantitative comparison on HeisenVec. The upper block augments the original vHector results [28] with our consistency evaluations; the lower block reports CANVAS results. Best and second-best results are bold and underlined.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Type</td><td rowspan="2">Token pred.</td><td colspan="6">Standard metrics</td><td colspan="6">Additional consistency metrics</td></tr><tr><td>CLIP-S↑</td><td>FID↓ HPSv2↑ SSIM↑ LPIPS↓ DINO-S↑</td><td></td><td></td><td></td><td></td><td> $\mathrm { L C I _ { 9 \times 9 } }$ </td><td>↑ CVIU-C↑</td><td>SC↑</td><td>SL↑</td><td></td><td>CE↑ LaDe↑</td></tr><tr><td colspan="10">Original vHector Table 2 baseline results</td><td colspan="7"></td></tr><tr><td>CodeLLaMA [28]</td><td></td><td>LLM 8,192</td><td></td><td>60.82 793.97</td><td></td><td>13.83</td><td>59.51</td><td>67.79</td><td>25.68</td><td>0.437</td><td>0.271</td><td>0.3140.1220.290</td><td></td><td></td><td>2.134</td></tr><tr><td>IconShop [20]</td><td>Dec.</td><td></td><td></td><td>66.09 316.86</td><td></td><td>12.94</td><td>46.78</td><td>68.29</td><td>39.23</td><td>0.737</td><td>0.594</td><td>0.688</td><td>0.283</td><td>0.633</td><td>3.691</td></tr><tr><td>LLM4SVG [24]</td><td>LLM</td><td>2,048</td><td></td><td>62.08830.70</td><td></td><td>14.01</td><td>69.62</td><td>61.85</td><td>35.92</td><td>0.728</td><td>0.190</td><td>0.436</td><td></td><td>0.2090.497</td><td>2.674</td></tr><tr><td>OmniSVG [26]</td><td>VLM</td><td>8,192</td><td></td><td>63.82 309.20</td><td></td><td>14.46</td><td>61.05</td><td>65.93</td><td>37.63</td><td>0.676</td><td>0.457</td><td>0.465</td><td></td><td>0.2350.480</td><td>2.806</td></tr><tr><td>vHector 8B [28]</td><td>LLM</td><td>8,192</td><td></td><td>67.84 105.22</td><td></td><td>15.32</td><td>65.91</td><td>57.54</td><td>49.33</td><td>0.723</td><td></td><td>0.4800.531</td><td></td><td>0.3600.522</td><td>2.950</td></tr><tr><td>vHector 3B [28]</td><td>LLM</td><td>8,192</td><td></td><td>64.30343.90</td><td></td><td>14.34</td><td>63.62</td><td>61.40</td><td>37.76</td><td>0.470</td><td></td><td>0.3100.314</td><td>0.175</td><td>0.280</td><td>2.233</td></tr><tr><td colspan="10">Our plug-and-play CANVAS results</td><td colspan="7"></td></tr><tr><td>CodeLLaMA-7B+CANVAS LLM</td><td></td><td>8,192</td><td></td><td>66.55177.11</td><td></td><td>14.50</td><td>65.48</td><td>63.16</td><td>31.63</td><td>0.454</td><td>0.277</td><td>0.339</td><td>0.1550.317</td><td></td><td>2.231</td></tr><tr><td>IconShop+CANVAS</td><td>Dec.</td><td></td><td></td><td>67.70177.99</td><td></td><td>13.59</td><td>50.97</td><td>65.46</td><td>43.84</td><td>0.746</td><td>0.594</td><td>0.725</td><td>0.357</td><td>0.701</td><td>3.875</td></tr><tr><td>LLM4SVG+CANVAS</td><td>LLM</td><td>2,048</td><td></td><td>67.02146.64</td><td></td><td>14.30</td><td>71.21</td><td>60.59</td><td>38.51</td><td>0.635</td><td>0.172</td><td>0.445</td><td></td><td>0.253 0.503</td><td>2.732</td></tr><tr><td>OmniSVG+CANVAS</td><td>VLM</td><td>8,192</td><td>67.24</td><td>119.76</td><td></td><td>14.80</td><td>63.85</td><td>64.05</td><td>40.76</td><td>0.652</td><td></td><td>0.444</td><td></td><td>0.4830.2680.498</td><td>2.875</td></tr><tr><td>vHector-8B+CANVAS</td><td>LLM</td><td>8,192</td><td>72.32</td><td>151.63</td><td></td><td>20.22</td><td>59.52</td><td>56.07</td><td>59.40</td><td>0.774</td><td></td><td>0.526</td><td></td><td>0.931 0.927 0.916</td><td>4.736</td></tr><tr><td>vHector-3B+CANVAS</td><td>LLM</td><td>8,192</td><td></td><td>65.77150.84</td><td></td><td>15.10</td><td>65.12</td><td>59.91</td><td>40.70</td><td>0.510</td><td></td><td>0.3240.3680.2290.344</td><td></td><td></td><td>2.420</td></tr></table>

Table 2. Quantitative comparison on the IntroSVG evaluation set. The upper block augments the original IntroSVG results [19] with our consistency evaluations; the lower block reports CANVAS results. Best and second-best results are bold and underlined.
<table><tr><td rowspan="2">Backbone</td><td colspan="6">Standard metrics</td><td colspan="4">Additional consistency metrics</td></tr><tr><td>Avg. Token↓</td><td>RSR↑</td><td></td><td>FID↓ CLIP-T2I↑ Aesthetic↑ HPS↑</td><td></td><td></td><td> $\operatorname { L C I } _ { 9 \times 9 } \uparrow$ </td><td>CVIU-C↑ SC↑</td><td>SL↑</td><td>CE↑ LaDe↑</td></tr><tr><td>Original IntroSVG baseline results</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniSVG (Qwen2.5-3B-Instruct) [26]</td><td>2260.54</td><td>75.36</td><td>142.38</td><td>0.2297</td><td></td><td>4.72320.1877</td><td>0.676</td><td>0.466 0.507</td><td>0.293 0.513</td><td>2.808</td></tr><tr><td>SVGen (Qwen2.5-Coder-7B-Instruct) [18]</td><td>1531.42</td><td>84.64</td><td>26.27</td><td>0.2339</td><td></td><td>4.58580.1916</td><td>0.841</td><td>0.6240.774</td><td>0.586 0.798</td><td>3.887</td></tr><tr><td>IntroSVG [19]</td><td>1707.77</td><td>99.26</td><td>26.18</td><td>0.2529</td><td>4.88940.1969</td><td></td><td>0.841</td><td>0.6200.775</td><td>0.5880.802</td><td>3.892</td></tr><tr><td>Our plug-and-play CANVAS result</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OmniSVG+CANVAS</td><td>1337.52</td><td>92.57</td><td>25.73</td><td>0.2491</td><td></td><td>4.59700.1925</td><td>0.839</td><td>0.6250.7730.582 0.804</td><td></td><td>3.858</td></tr><tr><td>SVGen+CANVAS</td><td>1319.20</td><td>81.29</td><td>25.71</td><td>0.2499</td><td></td><td>4.5917 0.1930</td><td>0.847</td><td>0.6340.781</td><td>0.555 0.799</td><td>3.922</td></tr><tr><td>IntroSVG+CANVAS</td><td>1416.04 100.00</td><td></td><td>25.66</td><td>0.2615</td><td></td><td>4.9872 0.2008</td><td>0.844</td><td>0.6240.7910.6130.810</td><td></td><td>3.999</td></tr></table>

Table 3. Ablation of the decision unit. Times are average generation wall-clock per sample on four H200 GPUs: Ours uses the completed strict four-lane runtime, while the two-sample alternatives are linearly extrapolated. Higher is better except for FID, LPIPS, and time. Bes and second-best results are bold and underlined, respectively.
<table><tr><td rowspan="2">Variant</td><td colspan="6">Standard metrics</td><td colspan="4">Additional consistency metrics</td><td rowspan="2">Avg. time/sample (s) 4×H200↓</td></tr><tr><td>CLIP-S↑</td><td>FID↓</td><td>HPSv2↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DINO-S↑</td><td> $\mathrm { L C I _ { 9 \times 9 } } ^ { \mathbf { \alpha } . }$ </td><td>↑ CVIU-C↑ SC↑</td><td>SL↑</td><td>CE↑ LaDe↑</td></tr><tr><td>Token-wise</td><td>68.878</td><td>66.574</td><td>13.753</td><td>69.913</td><td>62.766</td><td>32.981</td><td>0.8801</td><td>0.3121 0.4723</td><td>0.2663</td><td>0.5230 2.6875</td><td>≈7,112.4s</td></tr><tr><td>Fixed block-wise (m = 20)</td><td>71.326</td><td>89.597</td><td>14.395</td><td>62.912</td><td>61.039</td><td>56.761</td><td>0.8195</td><td>0.6021 0.6355</td><td>0.3769</td><td>0.6657 3.7224</td><td>≈165.9 s</td></tr><tr><td>Ours</td><td>72.324</td><td>151.634</td><td>20.219</td><td>59.516</td><td>56.073</td><td>59.400</td><td>0.7736</td><td>0.52560.9312</td><td>0.9274</td><td>0.9159 4.7362≈17.8s</td><td></td></tr></table>

Table 4. Rollout-horizon ablation on vHector-8B. Times are average generation wall-clock per sample on four H200 GPUs. H1(Ours), H2, and H3 use completed-run timings converted to the same four-card basis (excluding user-requested pauses), while HEOS uses a linear throughput extrapolation. Higher is better except for FID, LPIPS, and time. Best and second-best results are bold and underlined, respectively.
<table><tr><td rowspan="2">Variant</td><td colspan="5">Standard metrics</td><td colspan="5">Additional consistency metrics</td><td rowspan="2">Avg. time/sample (s) 4×H200↓</td></tr><tr><td>CLIP-S↑</td><td>FID↓</td><td>HPSv2↑</td><td>SSIM↑ LPIPS↓</td><td>DINO-S↑</td><td> $\mathrm { L C I } _ { 9 \times 9 }$ </td><td>↑CVIU-C↑</td><td>SC↑</td><td>SL↑ CE↑</td><td>LaDe↑</td></tr><tr><td>H2</td><td>69.977</td><td>102.383</td><td>15.806</td><td>62.604</td><td>59.617</td><td>56.048</td><td>0.7976 0.5811</td><td>0.5698</td><td>0.4184</td><td>0.6177 3.1752</td><td>≈38.2 s</td></tr><tr><td>H3</td><td>71.046</td><td>198.614</td><td>17.844</td><td>63.681 54.196</td><td>65.620</td><td>0.8377</td><td>0.6180</td><td>0.6951</td><td>0.5182 0.7338</td><td>3.6844</td><td>≈63.4s</td></tr><tr><td>HEOS</td><td>70.358</td><td>205.196</td><td>17.300</td><td>61.466 55.391</td><td>59.878</td><td>0.8270</td><td>0.6212</td><td>0.6456</td><td>0.4816 0.6971</td><td>3.5956</td><td>≈227.5 s</td></tr><tr><td>H1(Ours)</td><td>72.324</td><td>151.634</td><td>20.219</td><td>59.516 56.073</td><td>59.400</td><td>0.7736</td><td>0.5256</td><td>0.9312</td><td>0.9274 0.9159</td><td>4.7362</td><td>≈17.8s</td></tr></table>

## 4. Experiments

Datasets and evaluation metrics. To evaluate the effectiveness and generalization ability of CANVAS across different SVG generators, we conduct experiments on two complementary Text-to-SVG benchmarks: the HeisenVec benchmark introduced with vHector [28] and the benchmark introduced by IntroSVG [19]. Following their official evaluation protocols, we retain each benchmark’s original metrics. HeisenVec targets complex, long-context Text-to-SVG generation and exposes the difficulty of maintaining coherent SVG structure over extended outputs [28]. For this dataset, we follow vHector [28] and report CLIP Score (CLIP-S), Fréchet Inception Distance (FID), Human Preference Score v2 (HPSv2), Structural Similarity Index Measure (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), and DINO Similarity (DINO-S). IntroSVG is a more recent Textto-SVG framework that explicitly incorporates renderedimage feedback to iteratively improve the visual fidelity and semantic accuracy of complex, multicolor SVGs. Its unified evaluation setting therefore provides a strong complementary testbed for assessing whether CANVAS can further improve global consistency in structurally rich, long-form SVG generation [19]. For this dataset, we follow IntroSVG [19] and report average token count (Avg. Token), Render Success Rate (RSR), FID, CLIP text-to-image similarity (CLIP-T2I), Aesthetic Score, and HPS. Definitions and execution details for the metrics used by both benchmarks are provided in Appendix L.

Beyond these benchmark-standard measures, we introduce six complementary consistency metrics for long-range SVG geometry and composition. They are the Local Connectivity Index $\left( \mathrm { L C I _ { 9 \times 9 } } \right)$ , CVIU Statistical Complexity (CVIU-C), Structural Coherence (SC), Spatial Logic (SL), Character Efficiency (CE), and LaDe Cross-Layer Consistency (LaDe). Their formulas, the intuition behind them, SVG transfer protocols, execution details, and qualitative interpretations are provided in Appendix M.

Implementation details. Main experiments use four NVIDIA H200 GPUs with $\alpha = 2 , \beta = 6 4$ , and $H = 1 ;$ further details are provided in Appendices I and J.

## 4.1. Main Results

We apply CANVAS to mainstream Text-to-SVG generators on HeisenVec [28] and IntroSVG [19] (Tabs. 1 and 2). Across both benchmarks and the large majority of reported metrics, CANVAS yields consistent improvements, supporting its effectiveness and generalization ability in longcontext SVG generation. Comparisons with temperature scaling, beam search, and Best-of-N, together with qualitative visualizations, are provided in Appendix K.

## 4.2. Ablation Studies

We conduct extensive ablation experiments using vHector-8B and the HeisenVec evaluation protocol from Tab. 1. Additional ablation studies are provided in Appendix K. We analyze two choices that are central to making future-aware SVG navigation both effective and computationally practical: the decision unit of CANVAS and the rollout horizon used to estimate the candidates’ future values.

Impact of using strokes as the decision unit. In CAN-VAS, we use parser-complete strokes as the decision unit. To assess the contribution of this choice, we test two variants. In the first variant (token-wise), CANVAS applies its evaluation and selection procedure at every token. In the second variant (fixed block-wise), CANVAS applies the same procedure once per fixed-length token block. As shown in Tab. 3, although these alternatives may lead on individual metrics, their cost–quality tradeoffs are unsatisfactory: token-wise decisions require an unacceptable inference-time cost, while the cost of fixed block-wise decisions can be moved into a practical range by changing the block length m, but their performance is unstable because a fixed block may end inside a coordinate, path command, or XML attribute. We therefore report m = 20 as a representative setting with comparatively strong performance. These results support our parser-aligned decision unit: stroke boundaries provide a visually meaningful and manageable action space, striking a favorable balance between computational cost and generation quality.

Impact of visual adaptive sampling. For each candidate stroke, CANVAS uses visual adaptive sampling with bounded future rollouts to estimate its future value under a fixed inference-time budget. To assess the rollout design, we compare the complete one-stroke method (H1/Ours) with two-stroke (H2), three-stroke (H3), and valid-EOS (HEOS) rollouts. As shown in Tab. 4, CANVAS achieves a favorable balance between computational cost and generation quality. These results demonstrate the effectiveness of our visual adaptive sampling design.

## 5. Conclusion

In this paper, we proposed CANVAS, a novel training-free framework that improves the global consistency of longcontext Text-to-SVG decoding by introducing consistencyaware navigation through the SVG generation space. In CANVAS, we propose several novel designs that enable a fixed autoregressive SVG generator to exploit both its learned preference over complete generation trajectories and direct visual evidence from rendered futures within an acceptable inference-time budget and without any additional training. Extensive experiments demonstrate the effectiveness and generalization ability of CANVAS.

## References

[1] Hanqi Chen, Zhongyin Zhao, Ye Chen, Zhujin Liang, and Bingbing Ni. SVGThinker: Instruction-aligned and reasoningdriven text-to-SVG generation. In Proceedings of the 33rd ACM International Conference on Multimedia, 2025. 2

[2] Kewei Chen, Yayu Long, and Mingsheng Shang. Drift is a sampling error: SNR-aware power distributions for longhorizon robotic planning. arXiv preprint arXiv:2605.09537, 2026. 2, 3

[3] Zehao Chen and Rong Pan. SVGBuilder: Component-based colored SVG generation with text-guided autoregressive transformers. In Proceedings ofthe AAAI Conference on Artificial Intelligence, pages 2358–2366, 2025. 1, 2, 5

[4] Marco Cipriano, Moritz Feuerpfeil, and Gerard De Melo. Vector grimoire: Codebook-based shape generation under raster image supervision. In Proceedings of the 42nd International Conference on Machine Learning, pages 10966–10993, 2025. 2

[5] Haixia Feng, Qingwu Hu, Pengcheng Zhao, Daoyuan Zheng, Mingyao Ai, Siliang Chen, and Xiyu Hu. Automatic generation of chinese mural line drawings via enhanced edge detection and CycleGAN-based denoising. npj Heritage Science, 13:345, 2025. 29

[6] Kevin Frans, Lisa Soros, and Olaf Witkowski. CLIPDraw: Exploring text-to-drawing synthesis through language-image encoders. In Advances in Neural Information Processing Systems, pages 5207–5218, 2022. 2

[7] Javier Gimenez, Jorge Martinez, and Ana Georgina Flesia. Unsupervised edge map scoring: A statistical complexity approach. Computer Vision and Image Understanding, 122: 131–142, 2014. 29

[8] Ajay Jain, Amber Xie, and Pieter Abbeel. VectorFusion: Text-to-SVG by abstracting pixel-based diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1911–1920, 2023. 2

[9] Xiaotong Ji, Rasul Tutunov, Matthieu Zimmer, and Haitham Bou Ammar. Scalable power sampling: Unlocking efficient, training-free reasoning for LLMs via distribution sharpening. arXiv preprint arXiv:2601.21590, 2026. 4

[10] Herman Kahn and Albert W. Marshall. Methods of reducing sample size in monte carlo computations. Journal of the Operations Research Society ofAmerica, 1(5):263–278, 1953. 16

[11] Aayush Karan and Yilun Du. Reasoning with sampling: Your base model is smarter than you think. arXiv preprint arXiv:2510.14901, 2025. 2

[12] Guotao Liang, Zhangcheng Wang, Juncheng Hu, Haitao Zhou, Ziteng Xue, Jing Zhang, Dong Xu, and Qian Yu. Render-inthe-loop: Vector graphics generation via visual self-feedback. arXiv preprint arXiv:2604.20730, 2026. 1, 4

[13] Vlad-Constantin Lungu-Stan, Ionu¸t Mironica, and Mariana-˘ Iuliana Georgescu. LaDe: Unified multi-layered graphic media generation and decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 6031–6040, 2026. 26, 27

[14] Jelena Markovic-Voronov, Wenhui Zhu, Bo Long, Zhipeng Wang, Suyash Gupta, Kayhan Behdin, Bee-Chung Chen, and

Deepak Agarwal. Sampling for quality: Training-free rewardguided LLM decoding via sequential monte carlo. arXiv preprint arXiv:2604.16453, 2026. 2, 3, 13

[15] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023. 3

[16] Juan A. Rodriguez, Abhay Puri, Shubham Agarwal, Issam H. Laradji, Pau Rodriguez, Sai Rajeswar, David Vazquez, Christopher Pal, and Marco Pedersoli. StarVector: Generating scalable vector graphics code from images and text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16175–16186, 2025. 1

[17] Zecheng Tang, Chenfei Wu, Zekai Zhang, Minheng Ni, Shengming Yin, Yu Liu, Zhengyuan Yang, Lijuan Wang, Zicheng Liu, Juntao Li, and Nan Duan. StrokeNUWA: Tokenizing strokes for vector graphic synthesis. In Proceedings of the 41st International Conference on Machine Learning, pages 47830–47845, 2024. 2

[18] Feiyu Wang, Zhiyuan Zhao, Yuandong Liu, Da Zhang, Junyu Gao, Hao Sun, and Xuelong Li. SVGen: Interpretable vector graphics generation with large language models. In Proceedings ofthe 33rd ACM International Conference on Multimedia, 2025. 1, 2, 7

[19] Feiyu Wang, Jiayuan Yang, Zhiyuan Zhao, Da Zhang, Bingyu Li, Peng Liu, and Junyu Gao. IntroSVG: Learning from rendering feedback for text-to-SVG generation via an introspective generator-critic framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 615–625, 2026. 1, 2, 7, 8, 25

[20] Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao. Icon-Shop: Text-guided vector icon synthesis with autoregressive transformers. ACM Transactions on Graphics, 42(6):1–14, 2023. 1, 2, 4, 7, 23

[21] Ronghuan Wu, Wanchao Su, and Jing Liao. Chat2SVG: Vector graphics generation with large language models and image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23690–23700, 2025. 1, 2

[22] Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu. DiffSketcher: Text guided vector sketch synthesis through latent diffusion models. In Advances in Neural Information Processing Systems, pages 15869–15889, 2023. 2

[23] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. SVGDreamer: Text guided SVG generation with diffusion model. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4546–4555, 2024. 2

[24] Ximing Xing, Juncheng Hu, Guotao Liang, Jing Zhang, Dong Xu, and Qian Yu. Empowering LLMs to understand and generate complex vector graphics. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19487–19497, 2025. 1, 2, 7, 23

[25] Ximing Xing, Ziteng Xue, Yandong Guan, Jing Zhang, Dong Xu, and Qian Yu. Reason-SVG: Enhancing structured reasoning for vector graphics generation with reinforcement

learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. 2

[26] Yiying Yang, Wei Cheng, Sijin Chen, Xianfang Zeng, Fukun Yin, Jiaxu Zhang, Liao Wang, Gang Yu, Xingjun Ma, and Yu-Gang Jiang. OmniSVG: A unified scalable vector graphics generation model. In Advances in Neural Information Processing Systems, 2025. 1, 2, 7, 23

[27] Yiren Zheng, Shibo Li, Jiaming Liu, Haofan Wang, and Yiren Song. Unlocking the latent canvas: Eliciting and benchmarking symbolic visual expression in LLMs. arXiv preprint arXiv:2603.14505, 2026. 26, 27

[28] Leonardo Zini, Elia Frigieri, Sebastiano Aloscari, and Lorenzo Baraldi. vHector and HeisenVec: Scalable vector graphics generation through large language models. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2025. 1, 2, 5, 7, 8, 23, 25

# Supplementary Material

## A. How Power Sharpening Concentrates the Complete-Sequence Prior

This section defines the complete-sequence power prior used by the target in the main text and proves its trajectoryconcentration property. Let q be a prompt, and let $p _ { \theta }$ denote the fixed base model with parameters θ. Let Y be a complete SVG program, let $T ( Y )$ denote its number of tokens, and write $\boldsymbol { Y } = ( y _ { 1 } , \dots , y _ { T ( Y ) } )$ . For each $t \in \{ 1 , \ldots , T ( Y ) \}$ , $y _ { t }$ is the token at position t, and $y _ { < t } = \left( y _ { 1 } , \ldots , y _ { t - 1 } \right)$ is the prefix preceding that token. Under prompt $q ,$ the model assigns $Y$ the complete-sequence likelihood $p _ { \theta } ( Y \mid q )$ by multiplying the next-token probabilities $p _ { \theta } ( y _ { t } \mid y _ { < t } , q )$ , as defined in Eq. 1 in the main paper. Let $\mathcal { { D } } _ { q }$ denote the space of valid, completely terminated SVG programs for prompt q; its operational caps are specified in Appendix B. We define the power-sharpened distribution over this space as

$$
\begin{array} { c } { { p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( Y \mid q ) = \displaystyle \frac { p _ { \theta } ( Y \mid q ) ^ { \alpha } } { Z _ { \alpha } ( q ) } , \qquad \alpha \ge 1 , } } \\ { { Z _ { \alpha } ( q ) = \displaystyle \sum _ { Y \in \mathcal { Y } _ { q } } p _ { \theta } ( Y \mid q ) ^ { \alpha } . } } \end{array}\tag{A.1}
$$

Here, α controls how strongly the base model’s preferences are reinforced at the complete-trajectory level. Setting $\alpha = 1$ leaves the relative probabilities over $\mathcal { { D } } _ { q }$ unchanged, whereas a larger α concentrates the distribution; $Z _ { \alpha } ( q )$ is the normalizer that makes the probabilities over $\mathcal { \partial } _ { q }$ sum to one. Fix prompt q and consider any two complete SVG programs $Y _ { a }$ and $Y _ { b }$ with positive probability; if $p _ { \theta } ( Y _ { a } \mid q ) > p _ { \theta } ( Y _ { b } \mid q )$ then for $\alpha > 1$

$$
\frac { p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( Y _ { a } \mid q ) } { p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( Y _ { b } \mid q ) } = \left( \frac { p _ { \theta } ( Y _ { a } \mid q ) } { p _ { \theta } ( Y _ { b } \mid q ) } \right) ^ { \alpha } > \frac { p _ { \theta } ( Y _ { a } \mid q ) } { p _ { \theta } ( Y _ { b } \mid q ) } .\tag{A.2}
$$

Therefore, power sharpening increases the relative weight of a more likely complete program over a less likely one. The same effect can be studied at the distribution level. Because

$$
p _ { \theta } ( Y \mid q ) ^ { \alpha } = \exp \{ \alpha \log p _ { \theta } ( Y \mid q ) \} ,
$$

the base-model log-likelihood is the natural quantity for describing how the power-sharpened distribution changes with α.

Let $\ell ( Y ) = \log p _ { \theta } ( Y \mid q )$ and define

$$
\begin{array} { l } { m ( \alpha ) = \mathbb { E } _ { Y \sim p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( \cdot \vert q ) } [ \ell ( Y ) ] } \\ { \quad = \displaystyle \sum _ { Y \in \mathcal { Y } _ { q } } p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( Y \mid q ) \log p _ { \theta } ( Y \mid q ) . } \end{array}\tag{A.3}
$$

Thus, $m ( \alpha )$ is the average base-model log-likelihood of a complete SVG program drawn from the power-sharpened

distribution. An increase in $m ( \alpha )$ means that the sampled programs are assigned higher base-model log-likelihood on average.

Differentiating with respect to α gives

$$
\frac { \mathrm { d } m ( \alpha ) } { \mathrm { d } \alpha } = \mathrm { V a r } _ { Y \sim p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( \cdot \vert q ) } [ \log p _ { \theta } ( Y \mid q ) ] \geq 0 .\tag{A.4}
$$

Because a variance is always nonnegative, $m ( \alpha )$ cannot decrease as α increases. It increases strictly unless all programs in the support have the same base-model likelihood. Therefore, increasing α shifts probability mass toward complete programs that the base model already considers more likely. This conclusion concerns only model likelihood and does not imply better rendering quality, which motivates the visual reward introduced in the main text.

## B. Unifying Online Decision Units in a Valid SVG Space

General block segmentation. Here we use a deterministic boundary rule Γ to split each generated sequence into blocks, which can describe token-wise, fixed-length, and parserdetected variable-length decision units in a single framework. For any valid complete sequence $\boldsymbol { Y } = \left( y _ { 1 } , \ldots , y _ { T ( Y ) } \right)$ , the rule produces the unique boundaries

$$
\begin{array} { r } { 0 = \tau _ { 0 } < \tau _ { 1 } < \cdot \cdot \cdot < \tau _ { K ( Y ) } = T ( Y ) , \ } \\ { B _ { k } = y _ { \tau _ { k - 1 } + 1 : \tau _ { k } } , \qquad } \\ { H _ { k } = y _ { 1 : \tau _ { k } } . \qquad } \end{array}\tag{B.1}
$$

Here, $K ( Y )$ is the number of blocks in $Y , \ k \in \mathbf { \Sigma }$ $\{ 1 , \ldots , K ( Y ) \}$ is the block index, and $\tau _ { k }$ is the token position at which the k-th block ends, with $\tau _ { 0 } = 0$ . Thus, $B _ { k }$ is the k-th block, and $H _ { k } = H _ { k - 1 } B _ { k }$ is the full sequence prefix after that block has been appended. The boundary rule operates online: after token $y _ { t }$ is generated, it determines whether the current block ends at position t based on the prefix $y _ { 1 : t }$

The three decision units are special cases of this rule. For token-wise decisions, each block contains one token, so the k-th block ends at token k and $\tau _ { k } = k$ . For fixed-length decisions, each block contains m tokens except possibly the final block, so $\tau _ { k } = \operatorname* { m i n } ( k m , T ( Y ) )$ ). For parser-detected decisions, let $h = H _ { k - 1 }$ be the current boundary prefix. Starting from h, suppose the incremental SVG parser first detects either a newly completed renderable drawing element or the completion of the entire SVG program after generating a segment $\boldsymbol { b } = ( b _ { 1 } , \dots , b _ { | b | } )$ . This entire segment, including the token that triggers the event, forms one candidate block. If b is selected and appended to $h ,$ we set $B _ { k } = b , \tau _ { k } =$ $\tau _ { k - 1 } + | b |$ , and $H _ { k } = h b$ . In this case, no shorter prefix of b already completes a new renderable drawing element or the entire SVG program. That is what the first-hit condition means: appending the complete block b to h triggers Γ, whereas appending any strict prefix $b _ { 1 : j }$ , with $1 \leq j < | b |$ does not.

To support boundary-wise decisions at a general block granularity, including the parser-detected stroke-wise granularity used by CANVAS, we aggregate the base model’s token-level probabilities within each candidate block.

Given a current boundary prefix h, let $B _ { \Gamma } ( h )$ denote the candidate blocks whose endpoint is the next boundary selected by Γ. Here, each $b \in B _ { \Gamma } ( h )$ satisfies the first-hit condition defined above. For a candidate block $\boldsymbol { b } = ( b _ { 1 } , \dots , b _ { | b | } )$ let $b _ { < j } = \left( b _ { 1 } , \ldots , b _ { j - 1 } \right)$ . Juxtaposition denotes sequence concatenation, so $h b _ { < j }$ is the complete token prefix before $b _ { j }$ is generated. The raw base-model probability of generating the entire block b from h is

$$
P _ { \theta } ^ { \Gamma } ( b \mid h , q ) = \prod _ { j = 1 } ^ { | b | } p _ { \theta } ( b _ { j } \mid h b _ { < j } , q ) , \qquad b \in \mathcal { B } _ { \Gamma } ( h ) .\tag{B.2}
$$

Here, $P _ { \theta } ^ { \Gamma } ( b \mid h , q )$ is simply the product of their original autoregressive next-token probabilities, with every token conditioned on the full prefix preceding it.

Now consider a valid complete sequence $Y$ . The rule Γ uniquely partitions $Y$ into consecutive, non-overlapping blocks $B _ { 1 } , \ldots , B _ { K ( Y ) }$ that together contain every token exactly once. Since $H _ { k - 1 }$ is the complete prefix before block $B _ { k }$ , the same complete-sequence likelihood can be written as

$$
p _ { \theta } ( Y \mid q ) = \prod _ { t = 1 } ^ { T ( Y ) } p _ { \theta } ( y _ { t } \mid y _ { < t } , q ) = \prod _ { k = 1 } ^ { K ( Y ) } P _ { \theta } ^ { \Gamma } ( B _ { k } \mid H _ { k - 1 } , q ) .\tag{B.3}
$$

The middle expression multiplies the model probabilities one token at a time, whereas the final expression groups exactly the same factors by block. Therefore, changing the boundary rule changes only the decision granularity, not the likelihood assigned to Y . When Γ is specialized to the parser-detected rule, each drawing block is parser-complete, and the final product gives the stroke-wise factorization used by CANVAS.

From the complete target to the stroke-wise conditional. To derive the stroke-wise conditional used by CANVAS, we first write the corresponding conditional for a general boundary rule Γ. Under the parser-detected rule used by CANVAS, each drawing candidate is a parser-complete stroke. A terminating candidate instead completes the SVG by closing all remaining SVG/XML tags and emitting EOS.

Let $h = H _ { k - 1 }$ be the current committed prefix, and let $b \in B _ { \Gamma } ( h )$ be a candidate decision unit. Let $\mathcal { { D } } _ { q }$ denote the set of valid, completely terminated SVG programs for prompt q under the fixed operational caps.

For fixed h and b, let C range over all future token suffixes that complete hb into an element of $\mathcal { { V } } _ { q }$ . Here, hbC denotes the concatenation of $h , b ,$ and $C .$ Thus, $C$ contains every token generated after b until valid SVG completion.

For each candidate $b ,$ we sum the complete-target contributions of all such valid futures and combine this future contribution with the base-model probability of b:

$$
\begin{array} { r l } & { \quad G _ { k } ^ { \Gamma } ( h b , q ) = \displaystyle \sum _ { C : h b C \in \mathcal { V } _ { q } } p _ { \theta } ( C \mid h b , q ) ^ { \alpha } e ^ { \beta s ( \mathcal { R } ( h b C ) , q ) } , } \\ & { \quad W _ { k } ^ { \Gamma } ( b ; h , q ) = P _ { \theta } ^ { \Gamma } ( b \mid h , q ) ^ { \alpha } G _ { k } ^ { \Gamma } ( h b , q ) . } \end{array}\tag{B.4}
$$

The two quantities in Eq. B.4 separate the contribution of the future continuations from the contribution of the current candidate.

For a fixed candidate $b ,$ consider one valid future suffix $C .$ The factor $p _ { \theta } ( C \mid h b , q ) ^ { \alpha }$ measures how strongly the power-sharpened base model favors this continuation, with α controlling the strength of the sharpening. The complete program hbC is rendered by $\mathcal { R }$ , and $s ( \mathcal { R } ( h b C ) , q )$ is its final visual score. The factor $e ^ { \beta s ( \mathcal { R } ( h b \dot { C } ) , \dot { q } ) }$ converts this score into a positive visual reward, with $\beta$ controlling the strength of the visual preference. Their product is the target contribution of this particular complete future. Summing this contribution over all valid suffixes $C$ gives $G _ { k } ^ { \Gamma } ( h b , q )$ the aggregate future score after candidate b.

The factor $P _ { \theta } ^ { \Gamma } ( b \mid h , q ) ^ { \alpha }$ measures the power-sharpened base-model preference for generating the current candidate b from prefix h. Multiplying it by $G _ { k } ^ { \Gamma } ( h b , q )$ gives $W _ { k } ^ { \Gamma } ( b ; h , q )$ , the unnormalized score assigned to candidate b. This score combines how plausible b is under the base model with the likelihoods and visual rewards of all valid complete SVGs that can follow it. It is not yet a probability; it becomes one only after the candidate scores are normalized.

All candidates are compared from the same committed prefix h. If the likelihood of this prefix were written explicitly, the full unnormalized score for candidate b would contain

$$
p _ { \theta } ( h \mid q ) ^ { \alpha } W _ { k } ^ { \Gamma } ( b ; h , q ) .
$$

Because $p _ { \theta } ( h \mid q ) ^ { \alpha }$ is identical for every candidate, it cancels when the candidate scores are normalized. The global normalizer of the complete-SVG target cancels for the same reason. We therefore omit these shared factors.

Normalizing the remaining candidate scores gives the conditional distribution at the current boundary:

$$
\Pi _ { \alpha , \beta } ^ { \Gamma } ( b \mid h , q ) = \frac { W _ { k } ^ { \Gamma } ( b ; h , q ) } { \sum _ { b ^ { \prime } \in \mathcal { B } _ { \Gamma } ( h ) } W _ { k } ^ { \Gamma } ( b ^ { \prime } ; h , q ) } .\tag{B.5}
$$

Thus, a candidate receives high conditional probability when it is both preferred by the base model and leads to high-value complete SVG trajectories.

This derivation holds for any boundary rule Γ. Under the parser-detected rule used by CANVAS, Eq. B.5 becomes the stroke-wise conditional.

A valid stroke follows parser first-hit law. During the generation of candidate strokes, a normal stop occurs when the parser first recognizes either a complete drawing element or the completion of the entire SVG program, with all remaining SVG/XML tags closed and EOS emitted. A failure stop occurs when a stroke-length token cap, program-length cap, or block-count cap is reached, or when the parser enters an unrecoverable malformed state. All such failure outcomes are represented by the single absorbing symbol ⊥.

Let $\overline { { B } } ( h )$ contain every explicit first-hit segment, whether or not it is later accepted, together with ⊥. The stopped generation law is

$$
\begin{array} { l } { { \displaystyle Q _ { \theta } ( b \mid h , q ) = \prod _ { j = 1 } ^ { | b | } p _ { \theta } ( b _ { j } \mid h b _ { < j } , q ) , \qquad b \not = \perp , } } \\ { { \displaystyle Q _ { \theta } ( \perp \mid h , q ) = 1 - \sum _ { b \in \overline { { \mathcal { B } } } ( h ) \atop b \not = \perp } Q _ { \theta } ( b \mid h , q ) . } } \end{array}\tag{B.6}
$$

For an explicit segment $b , Q _ { \theta } ( b \mid h , q )$ is its original autoregressive generation probability. The residual definition of $Q _ { \theta } ( \perp \mid h , q )$ assigns all remaining probability mass to failure, ensuring that $Q _ { \theta }$ sums to one.

Not every stopped outcome should be committed. Let $\nu ( h ) \subset { \overline { { B } } } ( h )$ contain the parser-valid candidate strokes and the terminating candidate that completes the SVG program. Its total stopped-law mass is

$$
Q _ { \theta } ( \mathcal { V } ( h ) \mid h , q ) = \sum _ { b \in \mathcal { V } ( h ) } Q _ { \theta } ( b \mid h , q ) .
$$

Conditioning the stopped law on this accepted set gives the proposal used at decision step k:

$$
q _ { k } ( b \mid h , q ) = \frac { Q _ { \theta } ( b \mid h , q ) } { Q _ { \theta } ( \mathcal { V } ( h ) \mid h , q ) } , \qquad b \in \mathcal { V } ( h ) .\tag{B.7}
$$

This proposal can be sampled by rejecting stopped outcomes outside $\mathcal { V } ( h )$ . The denominator is shared by all accepted candidates at the same step and therefore cancels in their normalized importance weights. If $Q _ { \theta } ( \mathcal { V } ( h ) \mid h , q ) =$ 0, no valid candidate can be generated from $h ,$ so the decoder reports a parser failure rather than committing an invalid segment.

Finite valid program space. To make the completesequence target and its normalizer well defined, we restrict generation to a finite set $\mathcal { { D } } _ { q }$ . This set contains all parservalid, fully terminated SVG programs for prompt q that satisfy three limits: at most $T _ { \mathrm { m a x } }$ tokens in total, at most $K _ { \mathrm { m a x } }$ parser-detected segments, and a stroke-length token cap of $B _ { \mathrm { m a x } }$ . The last cap means that, after one boundary is committed, the parser may read at most $B _ { \mathrm { m a x } }$ additional tokens while waiting for the next boundary. For a complete program $Y , K ( Y )$ denotes its number of segments, including the terminating segment that closes the remaining SVG/XML tags and emits EOS. If generation exceeds any limit or enters an unrecoverable malformed state, it terminates in the failure state ⊥ and does not produce an element of $\mathcal { { D } } _ { q }$ . These limits ensure that every rollout stops after finitely many steps and that sums over complete sequences are well defined.

More details about rendering and scoring. To make the visual score reproducible, we use the same renderer version, viewport, background, raster resolution, SVG canonicalization rule, and evaluator checkpoint throughout an experiment. For a completed program Y, the frozen evaluator’s output is mapped by a fixed calibration to $s ( \mathcal { R } ( Y ) , q ) \in [ 0 , 1 ]$ . Thus, the rendering and score of a given program do not depend on when or during which rollout it is evaluated.

## C. Deriving the Complete-SVG Target from the KL-Regularized Objective

Eq. 2 in the main paper defines the KL-regularized complete-SVG objective [14]. We now derive its closed-form optimizer term by term. To simplify notation, conditioning of all distributions on q is omitted below. Let $\Pi _ { \alpha , \beta }$ denote the following normalized complete-SVG target distribution:

$$
\begin{array} { r l } & { \Pi _ { \alpha , \beta } ( Y ) = \frac { p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( Y ) } { C _ { \alpha , \beta } ( q ) } \exp \{ \beta s ( \mathcal { R } ( Y ) , q ) \} , } \\ & { \quad C _ { \alpha , \beta } ( q ) = \displaystyle \sum _ { Y \in \mathcal { Y } _ { q } } p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( Y ) } \\ & { \quad \quad \quad \times \exp \{ \beta s ( \mathcal { R } ( Y ) , q ) \} . } \end{array}\tag{C.1}
$$

Because $\mathcal { { V } } _ { q }$ is finite and the score is bounded, $C _ { \alpha , \beta } ( q )$ is finite and positive whenever the valid base-model support is nonempty. For any candidate distribution π defined on the same support, write $\mathcal { D } _ { \alpha , \beta } ( \pi ) = D _ { \mathrm { K L } } ( \pi \| \Pi _ { \alpha , \beta } )$ . Expanding this KL divergence gives

$$
\begin{array} { r l } & { \mathcal { D } _ { \alpha , \beta } ( \pi ) = D _ { \mathrm { K L } } \Big ( \pi \Big \| p _ { \theta , \alpha } ^ { \mathrm { p o w } } \Big ) } \\ & { \qquad - \beta \mathbb { E } _ { Y \sim \pi } [ s ( \mathcal { R } ( Y ) , q ) ] } \\ & { \qquad + \log C _ { \alpha , \beta } ( q ) . } \end{array}\tag{C.2}
$$

Therefore, the objective in Eq. 2 of the main paper equals log $C _ { \alpha , \beta } ( q ) - D _ { \mathrm { K L } } ( \pi \| \Pi _ { \alpha , \beta } )$ . The KL divergence is nonnegative and equals zero only when the two distributions are identical, so the unique optimizer is $\pi = \Pi _ { \alpha , \beta }$ . Substituting $p _ { \theta , \alpha } ^ { \mathrm { p o w } } ( Y \mid q ) = p _ { \theta } ( Y \mid q ) ^ { \alpha } / Z _ { \alpha } ( q )$ and absorbing the Y -independent $Z _ { \alpha } ( q )$ into the normalizer gives Eq. 3 in the main paper.

## D. Turning the Complete-SVG Target into an Online Stroke-Wise Navigation Rule

To turn the global complete-SVG target into the online navigation rule used by CANVAS, this section derives the conditional distribution of the next stroke given the current committed prefix. The goal is to show that the stroke-wise rule is the conditional induced by the complete-SVG target rather than a separate local heuristic.

The derivation has four steps. We first decompose the complete-render reward into boundary-level visual changes without altering the normalized target. We then separate the contribution of the current candidate from the contributions of all futures that can follow it, rewrite the future contribution as an expectation under the base-model rollout law, and finally normalize the candidate scores. We carry out these steps for a general deterministic online boundary rule Γ before specializing it to parser-detected strokes.

Step 1: Decomposing the complete-render reward. To assign visual credit at individual boundaries without repeatedly counting the same partial rendering, we express the complete-render reward as a product of incremental visual factors. For the deterministic online boundary rule Γ introduced above, let $H _ { k }$ be the sequence prefix after the k-th decision unit has been appended, and define

$$
\begin{array} { r } { \psi _ { k } ( H _ { k } , q ) = \exp \{ \beta \left[ s ( \mathcal { R } ( H _ { k } ) , q ) - s ( \mathcal { R } ( H _ { k - 1 } ) , q ) \right] \} . } \end{array}\tag{D.1}
$$

The factor $\psi _ { k } ( H _ { k } , q )$ compares the rendered score after the k-th decision with the score before that decision. Multiplying these factors over the complete sequence gives

$$
\prod _ { k = 1 } ^ { K ( Y ) } \psi _ { k } ( H _ { k } , q ) = \exp \{ \beta \left[ s ( \mathcal { R } ( Y ) , q ) - s ( \mathcal { R } ( H _ { 0 } ) , q ) \right] \} .\tag{D.2}
$$

Here, every intermediate score appears once with a positive sign and once with a negative sign, so these scores cancel pairwise. Only the final score $s ( \mathcal { R } ( Y ) , q )$ and the initialcanvas score $s ( \mathcal { R } ( H _ { 0 } ) , q )$ remain. The initial-canvas score is shared by all programs and therefore cancels when the target is normalized. Thus, the decomposition preserves the complete-render factor in Eq. 3 and does not give extra reward to a drawing merely because it is split into more decision units. When $\beta > 0$ , a candidate b appended to prefix h has $\psi _ { k } ( h b , q ) > 1$ if it improves the canvas score, $\psi _ { k } ( h b , q ) < 1$ if it lowers the score, and $\psi _ { k } ( h b , q ) = 1$ if the score is unchanged. When $\beta = 0$ , all visual factors equal 1 and the target has no visual reweighting.

At this point, the global visual reward has been decomposed into one factor for the current decision and a product of factors for later decisions. We next use this decomposition to group complete trajectories according to their next candidate.

Step 2: Separating the current candidate from its futures. To determine the target contribution of a current candidate, we collect the weights of all complete outcomes that begin with that candidate. Let $h = H _ { k - 1 }$ be the current committed prefix, and let $b \in B _ { \Gamma } ( h )$ be a candidate decision unit beginning at h. Appending b produces the new prefix $h b .$ . For a valid continuation, let C contain every token generated after b until valid SVG termination, so $h b C$ is the corresponding complete program. Any factor that depends only on the shared prefix h is identical for all candidates and is omitted because it cancels during normalization.

We first separate the factors determined by the current candidate from the visual factors determined by its future continuation:

$$
\begin{array} { c } { { u _ { k } ( b ) = P _ { \theta } ^ { \Gamma } ( b \mid h , q ) ^ { \alpha } \psi _ { k } ( h b , q ) , } } \\ { { { } } } \\ { { \Phi _ { > k } ( h b , C , q ) = \displaystyle \prod _ { r = k + 1 } ^ { K ( h b C ) } \psi _ { r } ( H _ { r } , q ) . } } \end{array}\tag{D.3}
$$

Here, $u _ { k } ( b )$ is the contribution already determined after b has been generated: $P _ { \theta } ^ { \Gamma } ( b \mid h , q ) ^ { \alpha }$ is the sharpened basemodel preference for b, and $\psi _ { k } ( h b , q )$ is its immediate visual change. The product $\Phi _ { > k } ( h b , C , q )$ contains the visual changes of all later decisions along the valid continuation $C .$ . If b already completes the SVG program, C is empty and there are no later decisions, so this product is empty and equals 1.

At this point, the current and future factors have been separated, but the score of b must still include every future outcome reachable from hb. To include the same capped outcomes used by the decoder, let ${ \overline { { \mathcal { C } } } } ( h b )$ contain all valid terminal suffixes together with the absorbing failure outcome ⊥. Let $\overline { { p } } _ { \theta } ( C \mid h b , q )$ be the normalized capped rollout law that continues generation until either valid termination or ⊥. For a valid suffix, $\Phi _ { > k } ^ { \epsilon }$ is the telescoping product above. For a failure, it is the positive factor defined in Appendix I so that the complete failed trajectory receives reward ϵ. Summing over these outcomes gives

$$
\begin{array} { l } { { U _ { k } ^ { \epsilon } ( b ) = u _ { k } ( b ) \displaystyle \sum _ { C \in \overline { { \mathcal { C } } } ( h b ) } \overline { { p } } _ { \theta } ( C \mid h b , q ) ^ { \alpha } \Phi _ { > k } ^ { \epsilon } ( h b , C , q ) } } \\ { { \mathrm { } } } \\ { { \mathrm { } = u _ { k } ( b ) V _ { k } ^ { \epsilon } ( h b , q ) . } } \end{array}\tag{D.4}
$$

Here, $U _ { k } ^ { \epsilon } ( b )$ is the unnormalized target mass assigned to all outcomes whose next candidate is b. The equality defines $V _ { k } ^ { \epsilon } ( h b , q )$ as the total future contribution after b: each outcome is weighted by its power-sharpened suffix likelihood and its future visual factor. The factorization $U _ { k } ^ { \epsilon } ( b ) = u _ { k } ( b ) V _ { k } ^ { \epsilon } ( h b , q )$ therefore separates what is known about the current candidate from what depends on the futures reachable from it.

Step 3: Making future value estimable with rollouts. To estimate $V _ { k } ^ { \epsilon } ( h b , q )$ using continuations sampled directly from the capped base model, we rewrite its sum as an expectation under $\overline { { p } } _ { \boldsymbol { \theta } } ( \cdot \mid h b , q )$ . Using $\smash { \overline { { p } } _ { \theta } ( C \mid h b , q ) ^ { \alpha } = \overline { { p } } _ { \theta } ( C \mid }$ $h b , q ) \overline { { { p } } } _ { \theta } ( C \mid h b , q ) ^ { \alpha - 1 }$ gives

$$
\begin{array} { r l } & { V _ { k } ^ { \epsilon } ( h b , q ) = \displaystyle \sum _ { C \in \overline { { \mathcal { C } } } ( h b ) } \overline { { p } } _ { \theta } ( C \mid h b , q ) ^ { \alpha } \Phi _ { > k } ^ { \epsilon } ( h b , C , q ) } \\ & { \quad \quad \quad = \mathbb { E } _ { C \sim \overline { { p } } _ { \theta } ( \cdot \mid h b , q ) } \left[ \overline { { p } } _ { \theta } ( C \mid h b , q ) ^ { \alpha - 1 } \Phi _ { > k } ^ { \epsilon } ( h b , C , q ) \right] } \end{array}\tag{D.5}
$$

This expectation is an exact rewriting of the preceding sum, not an additional approximation. One factor of $\overline { { p } } _ { \theta }$ supplies the sampling distribution, while $\overline { { p } } _ { \theta } ^ { \alpha - 1 }$ supplies the remaining power needed to preserve the power-sharpened suffix likelihood. Failure outcomes remain in the expectation through their small, strictly positive $\Phi _ { > k } ^ { \epsilon }$ contribution rather than being silently discarded. For $\alpha \stackrel { \cdot } { = } 1 , \overline { { p } } _ { \theta } ^ { \alpha - 1 }$ is defined as 1 on positive-mass suffixes.

At this point, every candidate b has an exact unnormalized score $u _ { k } ( b ) V _ { k } ^ { \epsilon } ( h b , q )$ that combines its current contribution with the value of all future outcomes. The final step turns these scores into a conditional distribution and chooses the parser rule that defines strokes.

Step 4: Obtaining the stroke-wise selection rule. To obtain the probability of selecting the next candidate, we normalize its unnormalized mass over all valid candidates $b \in B _ { \Gamma } ( h )$

$$
\Pi _ { \alpha , \beta } ^ { \epsilon } ( B _ { k } = b \vert H _ { k - 1 } = h , q ) = \frac { u _ { k } ( b ) V _ { k } ^ { \epsilon } ( h b , q ) } { \sum _ { b ^ { \prime } \in \mathcal { B } _ { \Gamma } ( h ) } u _ { k } ( b ^ { \prime } ) V _ { k } ^ { \epsilon } ( h b ^ { \prime } , q ) } .\tag{D.6}
$$

Eq. D.6 holds for any deterministic online boundary rule Γ that produces a unique segmentation. The main text omits the ϵ superscript for concision; as $\epsilon  0$ , the expression recovers the conditional restricted to valid outcomes.

To obtain the stroke-wise rule used by CANVAS, we specialize Γ to parser-detected stroke boundaries. For nonterminal decisions in the default path-normalized representation, the next boundary is triggered when the parser recognizes a complete path-wise drawing element; if the representation retains other visible primitives, each complete visible drawing element triggers a boundary in the same way. The terminating boundary occurs when the remaining SVG/XML tags are closed and EOS is emitted, as defined above. This specialization changes only where online decisions are made. It does not change the complete-SVG target or the marginalization that produced the conditional. Together with the definitions of $u _ { k }$ and $V _ { k } ^ { \epsilon }$ above, Eq. D.6 yields Eqs. 7 and 6 in the main paper.

## E. Estimating Future Value with Finite-Horizon Rollouts

To estimate a candidate’s future value with bounded computation, we stop each continuation after at most H future strokes and average several rollouts. We first define this finite-horizon target, then distinguish truncation error from finite-sample error. In the main text, we use simplified notation. Here, $\overline { { p } } _ { \theta }$ denotes the rollout distribution after applying the stopping caps, and the superscript ϵ indicates that failed rollouts receive the small positive reward ϵ.

Bounding each rollout with a finite horizon. To bound the cost of one rollout, consider decision step k, let $h =$ $H _ { k - 1 }$ be the current committed prefix, and fix a sampled candidate stroke $B _ { i }$ . Starting from the new prefix $h B _ { i }$ , generate until H additional strokes have been produced or an earlier SVG-termination or failure event occurs. Let $C _ { 1 : H }$ denote the resulting future segment. If the SVG terminates or the rollout fails before reaching H future strokes, $C _ { 1 : H }$ contains only the strokes generated before that stopping event.

Under the capped rollout process, $\overline { { p } } _ { \theta } ( C _ { 1 : H } \mid h B _ { i } , q )$ is the probability that the base model generates the stopped continuation $C _ { 1 : H }$ from $h B _ { i }$ . We associate this continuation with a visual factor $\Phi _ { i , H } ^ { \epsilon }$ . For a valid continuation, this factor is the product of its incremental visual factors ψ. If the rollout fails, we instead use the positive factor corresponding to the small failure reward ϵ defined in Appendix I. We define the contribution of one rollout and its population finite-horizon value by

$$
\begin{array} { c } { { Z _ { i , H } = \overline { { { p } } } _ { \theta } ( C _ { 1 : H } \mid h B _ { i } , q ) ^ { \alpha - 1 } \Phi _ { i , H } ^ { \epsilon } , } } \\ { { V _ { k , H } ^ { \epsilon } ( h B _ { i } , q ) = \mathbb { E } [ Z _ { i , H } \mid B _ { i } ] . } } \end{array}\tag{E.1}
$$

Here, $\alpha \geq 1$ is the likelihood-sharpening exponent. After one factor of $\overline { { p } } _ { \theta }$ is used to sample the rollout, $\bar { p } _ { \theta } ^ { \alpha - 1 }$ supplies the remaining likelihood power and $\Phi _ { i , H } ^ { \epsilon }$ supplies the visual contribution. Thus, $Z _ { i , H }$ is one rollout’s contribution, $V _ { k , H } ^ { \epsilon }$ is its expectation, and $Z _ { i , H } ^ { ( m ) }$ denotes the m-th independent copy. If all continuations stop within H strokes, $V _ { k , H } ^ { \epsilon }$ equals the exact value $V _ { k } ^ { \epsilon } ;$ ; otherwise, it is a truncated surrogate.

Quantifying finite-horizon truncation error. To measure only the approximation introduced by stopping at $H ,$ consider a rollout that has not yet terminated at the horizon. Let $C _ { > H }$ denote all strokes generated after $C _ { 1 : H }$ , and let $\Phi _ { > k + H } ^ { \epsilon }$ be their future visual factor. After fixing the continuation $C _ { 1 : H }$ observed up to the horizon, we average over all possible suffixes that could follow it. We denote their expected additional multiplicative contribution by $W _ { H } ( C _ { 1 : H } )$

$$
\begin{array} { r } { W _ { H } ( C _ { 1 : H } ) = \mathbb { E } \left[ \overline { { p } } _ { \theta } ( C _ { > H } \mid h B _ { i } C _ { 1 : H } , q ) ^ { \alpha - 1 } \Phi _ { > k + H } ^ { \epsilon } \mid C _ { 1 : H } \right] . } \\ { ( \mathrm { E } . 2 ) } \end{array}
$$

The expectation in $W _ { H }$ is taken over the capped base-model law of the tail $C _ { > H }$ conditional on $C _ { 1 : H }$ . The likelihood term supplies the remaining power for this omitted suffix, while $\Phi _ { > k + H } ^ { \epsilon }$ supplies its visual contribution. If the rollout terminates within the horizon, there is no omitted tail and we set $W _ { H } = 1$ . The exact future value can then be recovered by multiplying the observed contribution $Z _ { i , H }$ by this tail multiplier and taking an expectation. By the tower property,

$$
V _ { k } ^ { \epsilon } = \mathbb { E } [ Z _ { i , H } W _ { H } ] , \qquad | V _ { k } ^ { \epsilon } - V _ { k , H } ^ { \epsilon } | \le \mathbb { E } [ Z _ { i , H } | W _ { H } - 1 | ] .\tag{E.3}
$$

where $V _ { k } ^ { \epsilon }$ is the exact untruncated value for the same candidate. The bound shows directly that truncation error is small when the omitted multiplier $W _ { H }$ stays close to 1 on rollouts with substantial contribution. Because truncating the rollout may change the candidate value, we empirically evaluate different rollout horizons in Appendix K.1.

At this point, the effect of truncating a rollout has been isolated through $W _ { H }$ . We next estimate the resulting finitehorizon value without introducing additional selection bias.

Keeping the finite-horizon estimate unbiased. To estimate $V _ { k , H } ^ { \epsilon } ( h B _ { i } , q )$ , we choose the rollout count $M _ { i }$ before observing the fresh rollout outcomes and average $M _ { i }$ independent contributions:

$$
\widehat { V } _ { i , H } = \frac { 1 } { M _ { i } } \sum _ { m = 1 } ^ { M _ { i } } Z _ { i , H } ^ { ( m ) } , \qquad \mathbb { E } [ \widehat { V } _ { i , H } \mid B _ { i } ] = V _ { k , H } ^ { \epsilon } ( h B _ { i } , q ) .\tag{E.4}
$$

Here, $V _ { k , H } ^ { \epsilon } ( h B _ { i } , q )$ is the exact finite-horizon future value of candidate $B _ { i }$ under the capped rollout law. Its sample estimate $\widehat { V } _ { i , H }$ is the average of $M _ { i }$ independent fresh-rollout contributions. Conditional on $B _ { i }$ and the preselected $M _ { i } ,$ this estimate is unbiased. Thus, H controls truncation of the target, whereas $M _ { i }$ controls Monte Carlo variance. As described in Appendix G, pilot rollouts are used to determine $M _ { i }$ . Once $M _ { i }$ is fixed, independent fresh rollouts are generated to compute $\widehat { V } _ { i , H }$

## F. Estimating the Next-Stroke Distribution with Importance Sampling

To avoid enumerating every parser-valid next stroke, we sample finitely many candidates and correct their proposal probabilities with importance weights. We then form unbiased estimates of the unnormalized candidate masses and identify the separate error introduced by normalization. To keep the notation concise, we omit the superscript ϵ in this section. Failed rollouts still receive the small positive reward ϵ defined in Appendix I.

Recovering target weights with importance sampling. To approximate the target without enumerating every parservalid next stroke, we draw candidates from a proposal distribution $q _ { k } ( \cdot \mid h , q )$ . Here, $q _ { k } ( b \mid h , q )$ is the actual probability that the decoder proposes stroke b from the current prefix h. Because candidates with larger proposal probabilities appear more often, we use importance sampling [10] to correct for these unequal sampling probabilities when computing their target weights.

For this correction to be valid, every candidate with positive target weight must also have positive probability under $q _ { k }$ . We then draw L candidates independently, $B _ { i } \stackrel { \mathrm { i i d } } { \sim } q _ { k } ( \cdot | h , q )$ , and define

$$
A _ { i } = \frac { u _ { k } ( B _ { i } ) } { q _ { k } ( B _ { i } \mid h , q ) } , \qquad i = 1 , \ldots , L .\tag{F.1}
$$

We refer to each draw $B _ { i }$ as a candidate particle. The numerator $u _ { k } ( B _ { i } ) \ = \ P _ { \theta } ^ { \mathrm { s t r o k e } } ( B _ { i } \ \mid \ h , q ) ^ { \alpha } \psi _ { k } ( h B _ { i } , q )$ is the current-stroke factor required by the target, while the denominator $q _ { k } ( B _ { i } \mid h , q )$ is the probability with which that particle was sampled. Thus, $A _ { i }$ corrects the proposal frequency so that proposal-weighted averages recover sums under the unnormalized stroke target.

From corrected weights to the next-stroke distribution. To convert the corrected weights into probabilities for choosing the next stroke, we normalize their target contributions over all candidates. We write this normalization using a candidate-level function $f$ whose values are bounded, so that the same formulation covers both individual candidate probabilities and other weighted averages. For a generic candidate B, let $A ( B ) = u _ { k } ( B ) / q _ { k } ( B \mid h , q )$ denote the importance factor defined above. We then define

$$
\begin{array} { c l } { \displaystyle N _ { k , H } ( f ) = \mathbb { E } _ { B \sim q _ { k } } \Big [ } \\ { \displaystyle A ( B ) V _ { k , H } ( h B , q ) f ( B ) \Big ] , } \\ { \displaystyle D _ { k , H } = N _ { k , H } ( 1 ) , } \\ { \displaystyle T _ { k , H } ( f ) = \frac { N _ { k , H } ( f ) } { D _ { k , H } } . } \end{array}\tag{F.2}
$$

Here, $N _ { k , H } ( f )$ is the total candidate weight after multiplying each candidate’s contribution by $f ( B )$ , while $D _ { k , H } =$ $N _ { k , H } ( 1 )$ is the total weight of all candidates. Their ratio $T _ { k , H } ( f ) = N _ { k , H } ( f ) / D _ { k , H }$ is therefore the corresponding normalized quantity. In particular, if $f ( B ) = { \bf 1 } [ B = b ]$ only candidate b contributes to the numerator, so $T _ { k , H } ( f )$ is the probability assigned to b by the finite-horizon next-stroke distribution.

To estimate these quantities from the L sampled particles, we replace each exact future value by its independent fresh-

rollout estimate $\widehat { V } _ { i , H }$

$$
\widehat { N } _ { k , H } ( f ) = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } A _ { i } \widehat { V } _ { i , H } f ( B _ { i } ) , \qquad \widehat { D } _ { k , H } = \widehat { N } _ { k , H } ( 1 ) .\tag{F.3}
$$

Here, $N _ { k , H } ( f )$ and $D _ { k , H }$ are the exact numerator and denominator of the finite-horizon normalized statistic. Their finite-sample estimates, $\widehat { N } _ { k , H } ( f )$ and $\widehat { D } _ { k , H }$ , are obtained by averaging over the $L$ sampled candidates and replacing each exact future value with its independent fresh-rollout estimate $\widehat { V } _ { i , H }$ . The proposal correction in $A _ { i }$ removes candidatesampling bias, while Eq. E.4 removes bias from estimating each finite-horizon future value. Applying iterated expectation therefore gives

$$
\mathbb { E } [ \widehat { N } _ { k , H } ( f ) ] = N _ { k , H } ( f ) , \qquad \mathbb { E } [ \widehat { D } _ { k , H } ] = D _ { k , H } .\tag{F.4}
$$

Thus, candidate sampling and fresh rollouts are unbiased before normalization. Their ratio can nevertheless be biased because $\widehat { D } _ { k , H }$ is random; Appendix H quantifies and corrects its leading bias.

At this point, the finite particle estimator is well defined for a proposal with full target support. We next state two implementation requirements needed to preserve that interpretation.

Ensuring valid importance weights. To make the importance-sampling correction match the actual sampling process, we keep every draw as a separate sample, even when multiple draws produce the same token sequence. The denominator in Eq. F.1 must also use the candidate’s actual sampling probability after top-p, top-K, parser screening, or rejection sampling. If any candidate with positive target weight is assigned zero sampling probability, the estimator no longer represents the full candidate distribution and instead targets only the retained candidates.

Under the default parser-valid proposal in Eq. B.7, every candidate with positive target weight also has positive proposal probability, so the full-support condition holds. Optional top-K or other hard screening instead changes the target to the restricted conditional described next.

Separating support restriction from Monte Carlo error. To separate the effect of intentionally excluding candidates from the error caused by drawing only finitely many samples, let $\mathcal { G } _ { k } ( h )$ denote the retained candidate set, such as a top-K set. Normalizing Eq. D.6 over the complete candidate space gives the exact one-step conditional, whereas normalizing only over $\mathcal { G } _ { k } ( h )$ gives that conditional further conditioned on $B _ { k } \in \mathcal G _ { k } ( h )$ . The difference between these two targets is support-restriction error. It is distinct from finite-L candidate Monte Carlo error, finite-H horizon error, and ratio bias.

This distinction concerns one stroke-wise decision; it does not by itself provide an error bound for the complete generated SVG.

## G. Spending the Rollout Budget Where It Most Improves the Decision

To use a limited rollout budget effectively, CANVAS gives more samples to candidates that are uncertain, influential in the normalized decision, and inexpensive to evaluate. We estimate these quantities with pilot rollouts, derive the corresponding cost-constrained allocation, and then implement it with the pilot estimates. As in Appendix F, we omit the superscript ϵ in this section; all future-value and decision quantities remain defined using the same small positive failure reward.

Using pilot rollouts to plan the fresh-rollout budget. To learn how difficult each candidate is before assigning the main budget, we first identify the population quantities that the pilot stage must estimate. For candidate $B _ { i }$ , let $V _ { i } = V _ { k , H } ( h B _ { i } , q )$ be its exact finite-horizon future value, let $\sigma _ { i } ^ { 2 } = \mathrm { V a r } ( Z _ { i , H } ^ { ( m ) } \mid B _ { i } )$ be the variance of one rollout contribution, and let $c _ { i } = \mathbb { E } [ c _ { i } ^ { ( m ) } \mid B _ { i } ]$ be the expected cost of one rollout. Here, $Z _ { i , H } ^ { ( m ) }$ and $c _ { i } ^ { ( m ) }$ denote the finite-horizon contribution and token-generation-plus-rendering cost of any single rollout $m$ . We estimate these quantities by drawing the same small number $m _ { 0 } \geq 2$ of pilot futures for every candidate and computing

$$
\begin{array} { c } { { \displaystyle \widehat { V } _ { i } ^ { ( 0 ) } = \frac { 1 } { m _ { 0 } } \sum _ { m = 1 } ^ { m _ { 0 } } Z _ { i , H } ^ { ( m ) } , } } \\ { { \displaystyle \widehat { \sigma } _ { i } ^ { 2 } = \frac { 1 } { m _ { 0 } - 1 } \sum _ { m = 1 } ^ { m _ { 0 } } \bigl ( Z _ { i , H } ^ { ( m ) } - \widehat { V } _ { i } ^ { ( 0 ) } \bigr ) ^ { 2 } , } } \\ { { \displaystyle \widehat { c } _ { i } = \frac { 1 } { m _ { 0 } } \sum _ { m = 1 } ^ { m _ { 0 } } c _ { i } ^ { ( m ) } . } } \end{array}\tag{G.1}
$$

The pilot statistics $\widehat { V } _ { i } ^ { ( 0 ) } , \widehat { \sigma } _ { i } ^ { 2 }$ , and $\widehat { c } _ { i }$ estimate $V _ { i } , \ \sigma _ { i } ^ { 2 } .$ , and $c _ { i } ,$ , respectively. The parenthesized superscript (0) labels quantities computed during the preliminary pilot stage; it is a stage index rather than an exponent. In particular, $\widehat { \widehat { V } } _ { i } ^ { ( 0 ) }$ is used only to allocate fresh rollouts, whereas $\widehat { V } _ { i , H }$ is the final estimate computed from those fresh rollouts.

The pilot rollouts are used only to choose the number $M _ { i }$ of fresh rollouts assigned to each candidate; their outcomes are not included in the final estimate $\widehat { V } _ { i , H }$ . After $M _ { i }$ is fixed, independent fresh rollouts are generated to compute the final future-value estimate. This separation prevents the same random pilot fluctuation from both changing the sample count and influencing the final estimate.

Identifying which candidates benefit most from additional rollouts. To decide where an additional rollout is most useful, we measure how an error in each future value changes the normalized candidate probabilities. Condition on the sampled candidates, and let $\begin{array} { r } { D = \sum _ { i } A _ { i } V _ { i } } \end{array}$ be their total unnormalized mass and $\rho _ { j } = A _ { j } V _ { j } / D$ be the ideal normalized probability of candidate $j .$ . Because $\widehat { V } _ { i , H }$ averages $M _ { i }$ independent fresh rollouts, $\mathrm { V a r } ( \widehat { V } _ { i , H } \mid B _ { i } ) = \sigma _ { i } ^ { 2 } / M _ { i }$

The sensitivity of probability $\rho _ { j }$ to a perturbation in $V _ { i }$ is

$$
\frac { \partial \rho _ { j } } { \partial V _ { i } } = \frac { A _ { i } } { D } \left( { \bf 1 } [ i = j ] - \rho _ { j } \right) .\tag{G.2}
$$

where $1 [ i = j ]$ equals 1 when $i = j$ and 0 otherwise. The factor $A _ { i } / D$ measures the scale of candidate i in the normalized decision, while $1 [ i = j ] - \rho _ { j } $ records how changing $V _ { i }$ redistributes probability across coordinate $j$

To summarize the effect over the full probability vector, apply first-order error propagation and sum over all coordinates:

$$
\mathbb { E } \| \widehat { \pmb { \rho } } - \pmb { \rho } \| _ { 2 } ^ { 2 } \simeq \sum _ { i = 1 } ^ { L } \frac { d _ { i } } { M _ { i } } , \qquad d _ { i } = \frac { A _ { i } ^ { 2 } \sigma _ { i } ^ { 2 } } { D ^ { 2 } } \| \mathbf { e } _ { i } - \pmb { \rho } \| _ { 2 } ^ { 2 } .\tag{G.3}
$$

Here, $\rho$ is the ideal candidate-probability vector obtained from the exact finite-horizon values $V _ { i } ,$ with $\rho _ { i } = A _ { i } V _ { i } / D$ The vector $\widehat { \rho }$ estimates $\rho$ by replacing each $V _ { i }$ with its freshrollout estimate $\widehat { V } _ { i , H }$ . The expectation is taken over the fresh-rollout randomness conditional on the sampled candidates, and $\mathbf { e } _ { i }$ is the one-hot vector for candidate i. The coefficient $d _ { i }$ combines rollout variance $\sigma _ { i } ^ { 2 }$ , importance scale $A _ { i } ^ { 2 } / D ^ { 2 }$ , and the candidate’s influence $\lVert \mathbf { e } _ { i } - \boldsymbol { \rho } \rVert _ { 2 } ^ { 2 }$ . Candidate i therefore contributes approximately $d _ { i } / M _ { i }$ to the squared decision error, so variance alone is not sufficient for allocation.

At this point, the statistical value of assigning rollouts to each candidate has been summarized by $d _ { i }$ . We next balance that value against the cost of generating and rendering those rollouts.

Minimizing decision error under a fixed rollout budget. To minimize the decision-error surrogate under a fixed expected budget, let $c _ { i } > 0$ be the expected token-generationplus-rendering cost of one fresh rollout for candidate $i ,$ and let $C _ { k }$ be the fresh-rollout budget for the current stroke-wise decision. If the population values $d _ { i }$ and $c _ { i }$ were known, the continuous allocation would solve

$$
\operatorname* { m i n } _ { M _ { i } > 0 } \sum _ { i } { \frac { d _ { i } } { M _ { i } } } \quad { \mathrm { s . t . } } \quad \sum _ { i } { c _ { i } M _ { i } } = C _ { k } .\tag{G.4}
$$

The objective is the approximate squared decision error, and the constraint charges each of the $M _ { i }$ rollouts its expected cost $c _ { i }$ . Introducing a multiplier λ gives the stationarity

condition $- d _ { i } / M _ { i } ^ { 2 } + \lambda c _ { i } = 0$ . Solving this condition and enforcing the budget yields

$$
M _ { i } ^ { * } = \frac { C _ { k } \sqrt { d _ { i } / c _ { i } } } { \sum _ { j } \sqrt { d _ { j } c _ { j } } } , \qquad \operatorname* { m i n } _ { \{ M _ { i } \} } \sum _ { i } \frac { d _ { i } } { M _ { i } } = \frac { \left( \sum _ { i } \sqrt { c _ { i } d _ { i } } \right) ^ { 2 } } { C _ { k } } .\tag{G.5}
$$

The optimal count satisfies $M _ { i } ^ { * } \propto \sqrt { d _ { i } / c _ { i } } ;$ a candidate receives more rollouts when its decision-error coefficient is larger and fewer when each rollout is more expensive. The second expression is the minimum attainable value of the first-order error surrogate under budget $C _ { k }$

Comparing adaptive and uniform allocation. To compare with equal allocation at the same expected cost, set the shared count to $\begin{array} { r } { M \ = \ C _ { k } / \sum _ { i } c _ { i } } \end{array}$ . Its surrogate error is $\begin{array} { r } { \mathcal { E } _ { \mathrm { u n i f o r m } } ~ = ~ ( \sum _ { i } d _ { i } ) ( \sum _ { i } c _ { i } ) / \langle C _ { k } } \end{array}$ , whereas Eq. G.5 gives $\begin{array} { r } { \mathcal { E } _ { \mathrm { a d a p t i v e } } = ( \sum _ { i } \sqrt { c _ { i } d _ { i } } ) ^ { 2 } / C _ { k } } \end{array}$ . By Cauchy–Schwarz, $\mathcal { E } _ { \mathrm { a d a p t i v e } } \leq \mathcal { E } _ { \mathrm { u n i f o r m } } ,$ with equality only when every candidate has the same ratio $d _ { i } / c _ { i }$ . This proves Eq. 13 in the main paper under the stated first-order surrogate.

Turning the oracle allocation into a practical rule. To use the oracle rule during decoding, replace the unknown $V _ { i }$ $\sigma _ { i } ^ { 2 }$ , and $c _ { i }$ by the pilot estimates from Eq. G.1. First compute

$$
\begin{array} { c } { { D ^ { ( 0 ) } = \displaystyle \sum _ { i } A _ { i } \widehat V _ { i } ^ { ( 0 ) } , } } \\ { { \displaystyle } } \\ { { \rho _ { i } ^ { ( 0 ) } = \frac { A _ { i } \widehat V _ { i } ^ { ( 0 ) } } { D ^ { ( 0 ) } } , } } \\ { { \widehat d _ { i } = \displaystyle \frac { A _ { i } ^ { 2 } \widehat \sigma _ { i } ^ { 2 } } { ( D ^ { ( 0 ) } ) ^ { 2 } } } } \\ { { \displaystyle \qquad \times \| { \bf e } _ { i } - \rho ^ { ( 0 ) } \| _ { 2 } ^ { 2 } . } } \end{array}\tag{G.6}
$$

Here, $d _ { i }$ is the ideal decision-error coefficient defined in Eq. G.3. Replacing the exact future values $V _ { i }$ with their pilot estimates $\widehat { V } _ { i } ^ { ( 0 ) }$ gives the pilot total mass $D ^ { ( 0 ) }$ and the pilot probability vector $\rho ^ { ( 0 ) }$ , whose i-th coordinate is $\rho _ { i } ^ { ( 0 ) }$ . The parenthesized superscript (0) indicates that $D ^ { ( 0 ) }$ and $\rho ^ { ( 0 ) }$ are computed from pilot-stage estimates; it is a stage label rather than an exponent. Combining these pilot quantities with ${ \widehat { \sigma } } _ { i } ^ { 2 }$ gives $\widehat { d } _ { i }$ , the pilot plug-in estimate of $d _ { i }$ . These quantities determine only the allocation; they do not enter the final fresh-rollout future-value estimate.

To prevent any candidate from receiving too few or too many rollouts, we impose $M _ { \operatorname* { m i n } } \le M _ { i } \le M _ { \operatorname* { m a x } }$ . The continuous Karush–Kuhn–Tucker solution is

$$
M _ { i } ( \lambda ) = \mathrm { c l i p } \left( \sqrt { \frac { \widehat { d } _ { i } } { \lambda \widehat { c } _ { i } } } , M _ { \mathrm { m i n } } , M _ { \mathrm { m a x } } \right) ,\tag{G.7}
$$

Here, $\mathrm { c l i p } ( x , M _ { \mathrm { m i n } } , M _ { \mathrm { m a x } } )$ keeps $x$ unchanged when it lies within the allowed interval, sets it to $M _ { \mathrm { m i n } }$ when it is too small, and sets it to $M _ { \mathrm { m a x } }$ when it is too large. The scalar λ controls the total allocation: increasing λ decreases the rollout counts. We choose λ by a one-dimensional monotone search so that the estimated total cost $\sum _ { i } \widehat { c } _ { i } M _ { i } ( \lambda )$ stays within the budget $C _ { k }$ and uses as much of that budget as the bounds allow.

Because rollout counts must be integers, we round the continuous allocation by repeatedly assigning an additional rollout where it gives the largest surrogate reduction per unit cost. If candidate i currently has $M _ { i }$ rollouts, that marginal gain is

$$
\Delta _ { i } ( M _ { i } ) = { \frac { \widehat { d } _ { i } } { \widehat { c } _ { i } } } \left( { \frac { 1 } { M _ { i } } } - { \frac { 1 } { M _ { i } + 1 } } \right) .\tag{G.8}
$$

To obtain integer rollout counts, we start from $M _ { i } = M _ { \operatorname* { m i n } }$ for every candidate and add rollouts one at a time. If candidate i currently has $M _ { i }$ rollouts, then

$$
\widehat { d } _ { i } \left( \frac { 1 } { M _ { i } } - \frac { 1 } { M _ { i } + 1 } \right)
$$

is the estimated reduction in the decision-error surrogate produced by one additional rollout. Dividing this reduction by the estimated rollout cost $\widehat { c } _ { i }$ gives $\Delta _ { i } ( M _ { i } )$ , the estimated benefit per unit cost. At each iteration, we assign the next affordable rollout to the candidate with the largest $\Delta _ { i } ( M _ { i } )$ We stop when the remaining budget cannot fund another rollout or every candidate has reached $M _ { \mathrm { m a x } }$ . Ties are broken randomly without using any fresh-rollout outcomes. To keep the allocation well defined in numerical edge cases, we first compute the minimum required cost $\begin{array} { r } { C _ { \operatorname* { m i n } } = \sum _ { i } \widehat { c } _ { i } M _ { \operatorname* { m i n } } } \end{array}$ . If $C _ { \mathrm { m i n } }$ exceeds the available budget, no allocation can satisfy the lower bound, so the fresh-rollout stage is skipped. Otherwise, every candidate receives at least $M _ { \mathrm { m i n } }$ rollouts, and any zero cost estimate is replaced by a small positive value. If the pilot denominator $D ^ { ( 0 ) }$ is zero or all $\widehat { d } _ { i }$ values are zero, the pilot results provide no usable basis for prioritizing candidates, so we use a uniform allocation instead. The budget is defined in terms of estimated generation-and-rendering cost rather than elapsed wall-clock time, while the rollout caps bound the amount of work in each rollout. Finally, L is fixed before candidate sampling, and all $M _ { i }$ values are fixed from the pilot data before fresh rollouts begin. Fresh-rollout outcomes therefore cannot change their own sample counts, preserving the conditional unbiasedness of the final sample means.

## H. Correcting the Leading Bias from Random Normalization

To turn the unnormalized estimates from Appendix F into probabilities, we divide by an estimated total mass. Because a random ratio generally satisfies $\mathbb { E } [ \widehat { N } / \widehat { D } ] \neq N / D .$ , this normalization introduces bias. We identify its leading $O ( L ^ { - 1 } )$ term and estimate that term from the sampled particles. The resulting correction removes the leading asymptotic bias, but is not exactly unbiased at finite sample sizes. $\mathbf { A } \mathbf { s }$ in $\mathsf { A p - }$ pendix F, we omit the superscript ϵ, although all quantities remain defined under the same failure-handling rule.

The source of normalization bias explicit. To identify where normalization bias comes from, we write each particle’s estimated numerator and denominator contributions explicitly. For a bounded candidate statistic $f ,$ define

$$
\begin{array} { c } { { \displaystyle Y _ { i } = A _ { i } \widehat V _ { i , H } , } } \\ { { \displaystyle X _ { i } ( f ) = f ( B _ { i } ) Y _ { i } , } } \\ { { \displaystyle \overline { { { Y } } } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } Y _ { i } , } } \\ { { \displaystyle \overline { { { X } } } _ { f } = \frac { 1 } { L } \sum _ { i = 1 } ^ { L } X _ { i } ( f ) . } } \end{array}\tag{H.1}
$$

Here, $A _ { i }$ is the candidate importance factor and $\widehat { V } _ { i , H }$ is its fresh-rollout future-value estimate. Their product $Y _ { i }$ is the estimated unnormalized mass of particle $B _ { i }$ . Multiplying by $f ( B _ { i } )$ gives its numerator contribution $X _ { i } ( f )$ . The averages $\overleftarrow { Y }$ and $\overline { { X } } _ { f }$ are therefore the estimated denominator and numerator of the normalized statistic.

To quantify the second-order fluctuations that cause ratio bias, assume $L \geq 2$ and compute the sample variance of the denominator contributions and their sample covariance with the numerator contributions:

$$
\begin{array} { c } { { s _ { Y } ^ { 2 } = \displaystyle \frac { 1 } { L - 1 } \sum _ { i } ( Y _ { i } - \overline { { { Y } } } ) ^ { 2 } , } } \\ { { s _ { X Y } ( f ) = \displaystyle \frac { 1 } { L - 1 } \sum _ { i } ( X _ { i } ( f ) - \overline { { { X } } } _ { f } ) ( Y _ { i } - \overline { { { Y } } } ) . } } \end{array}\tag{H.2}
$$

The quantity $s _ { Y } ^ { 2 }$ measures particle-to-particle variation in the estimated denominator mass, while $s _ { X Y } ( f )$ measures how numerator and denominator fluctuations move together.

Identifying the leading normalization bias. To define the population bias before estimating it, let $\mu _ { X } = \mathbb { E } [ \overline { { X } } _ { f } ]$ and $\mu _ { Y } = \operatorname { \mathbb { E } } [ { \overline { { Y } } } ]$ . Also define the scaled denominator variance $\nu _ { Y , L } = L \operatorname { V a r } ( { \overline { { Y } } } )$ and scaled numerator–denominator covariance $\nu _ { X Y , L } = \dot { L } \operatorname { C o v } ( \overline { { X } } _ { f } , \overline { { Y } } )$ . A second-order delta expansion of the random ratio $\dot { \overline { { X } } } _ { f } / \overline { { Y } }$ gives

$$
\begin{array} { c } { { \displaystyle B _ { L } ( f ) = \frac { 1 } { L } \left( \frac { \mu _ { X } \nu _ { Y , L } } { \mu _ { Y } ^ { 3 } } - \frac { \nu _ { X Y , L } } { \mu _ { Y } ^ { 2 } } \right) , } } \\ { { \displaystyle \mathbb { E } \left[ \frac { \overline { { X } } _ { f } } { \overline { { Y } } } \right] - T _ { k , H } ( f ) = B _ { L } ( f ) + o ( L ^ { - 1 } ) . } } \end{array}\tag{H.3}
$$

Candidate symmetry, conditionally unbiased fresh rollouts, and the pilot/fresh split imply $\mu _ { X } / \mu _ { Y } = T _ { k , H } ( f )$ . Thus,

$T _ { k , H } ( f )$ is the ideal finite-horizon normalized target, while $B _ { L } ( f )$ is the leading $O ( L ^ { - 1 } )$ ) difference between that target and the expected random ratio. The notation $o ( L ^ { - 1 } )$ denotes a smaller-order remainder.

Estimating and removing the leading bias. To estimate the normalized target and its leading bias from the observed particles, write

$$
\begin{array} { l } { \displaystyle \widehat { T } _ { k } ^ { + } ( f ) = \frac { \overline { { X } } _ { f } } { \overline { { Y } } } , } \\ { \displaystyle \widehat { B } _ { L } ( f ) = \frac { 1 } { L } \left( \frac { \overline { { X } } _ { f } s _ { Y } ^ { 2 } } { \overline { { Y } } ^ { 3 } } - \frac { s _ { X Y } ( f ) } { \overline { { Y } } ^ { 2 } } \right) . } \end{array}\tag{H.4}
$$

Here, $\widehat { T } _ { k } ^ { + } ( f )$ is the ordinary self-normalized estimate of $T _ { k , H } ( f )$ . The quantity $\widehat { B } _ { L } ( f )$ estimates the population bias term $B _ { L } ( f )$ by combining the observed denominator variance and numerator–denominator covariance, scaled by $1 / L .$

The observed moments $s _ { Y } ^ { 2 }$ and $s _ { X Y } ( f )$ consistently estimate the corresponding population second-order quantities. Consequently,

$$
{ \widehat { B } } _ { L } ( f ) - B _ { L } ( f ) = o _ { p } ( L ^ { - 1 } ) .\tag{H.5}
$$

Eq. H.5 means that, in probability, the error in the estimated correction is asymptotically smaller than the $L ^ { - 1 }$ bias term being removed. CANVAS therefore uses $\widetilde { T } _ { k } ( f ) = \widehat { T } _ { k } ^ { + } ( f ) -$ $\widehat { B } _ { L } ( f )$ , which is Eq. 14 in the main paper.

At this point, the leading ratio bias caused by finite candidate sampling and its observable correction have been identified. It remains to verify that using different fresh-rollout counts across candidates does not invalidate the expansion.

Preserving the correction under unequal rollout counts. To verify that adaptive, unequal rollout counts are covered by the correction, let $M = \operatorname* { m i n } _ { i } M _ { i }$ . If one rollout for candidate i has variance $\sigma _ { i } ^ { 2 } .$ , then its sample-mean future estimate contributes variance $\sigma _ { i } ^ { 2 } / M _ { i }$ . The resulting freshrollout components of the scaled moments are

$$
\begin{array} { l } { \nu _ { Y , L } ^ { \mathrm { f u t u r e } } = \displaystyle \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \mathbb { E } \bigg [ \frac { A _ { i } ^ { 2 } \sigma _ { i } ^ { 2 } } { M _ { i } } \bigg ] = O ( M ^ { - 1 } ) , } \\ { \nu _ { X Y , L } ^ { \mathrm { f u t u r e } } = \displaystyle \frac { 1 } { L } \sum _ { i = 1 } ^ { L } \mathbb { E } \bigg [ \frac { f ( B _ { i } ) A _ { i } ^ { 2 } \sigma _ { i } ^ { 2 } } { M _ { i } } \bigg ] = O ( M ^ { - 1 } ) . } \end{array}\tag{H.6}
$$

Here, the expectations include the candidate and pilot randomness determining $A _ { i } , \sigma _ { i } ^ { 2 }$ , and $M _ { i } .$ . Because the actual $M _ { i } ^ { - 1 }$ appears in each term and every $M _ { i } \geq M ,$ unequal allocation is retained explicitly and both contributions vanish as $O ( M ^ { - 1 } )$ .

Recovering the main-paper bias guarantee. To obtain the proposition in the main paper, take $L \ \to \ \infty$ and $M = \mathrm { m i n } _ { i } M _ { i }  \infty$ . We assume target-support coverage, bounded $f ,$ suitable bounded higher-order and inversedenominator moments, a permutation-equivariant pilot allocation, conditionally independent fresh batches, and the relevant laws of large numbers and uniform integrability. Zero-denominator and numerical-fallback probabilities must be $o ( L ^ { - 1 } )$ . Under these regularity conditions, Eqs. H.3–H.6 give $L ( \mathbb { E } [ \widetilde { T } _ { k } ( f ) ] - T _ { k , H } ( f ) ) \to 0$ , which is Eq. 15 and proves Proposition 1 in the main paper.

The correction removes only the leading ratio-bias term. As a consistency check, when $f \equiv 1 , \overline { { X } } _ { f } = \overline { { Y } }$ and $s _ { X Y } = s _ { Y } ^ { 2 }$ , so the correction is zero. Coordinate-wise correction of candidate indicators therefore sums to 1 in exact arithmetic; negative or nonfinite coordinates use the fallback in Appendix I.

## I. Handling Failed Rollouts and Numerical Edge Cases

To keep the sampler defined for every outcome, we assign small positive weight to failed rollouts, correct any modified rollout proposal, and specify numerical fallbacks. We also quantify how failure smoothing changes the valid-only target.

Preventing failed rollouts from receiving zero weight. To prevent failed rollouts from creating zero denominators or unstable weights, we define a bounded, strictly positive complete-trajectory reward. For a valid, renderable SVG program $Y ,$ , we use

$$
r _ { \beta } ( Y , q ) = \exp \{ \beta s ( \mathcal { R } ( Y ) , q ) \} > 0 .\tag{I.1}
$$

where $\mathcal { R } ( Y )$ is the final rendering, $s ( \mathcal { R } ( Y ) , q )$ is its bounded score for prompt $q ,$ and $\beta \geq 0$ controls the strength of visual reweighting. The theory does not require a particular scorer architecture; it only requires the evaluator and its calibration to be fixed during inference.

For a malformed, timeout, overflow, or renderer-failure outcome, a valid visual score is unavailable. Instead of assigning zero reward, we set

$$
\begin{array} { r } { r _ { \beta } ( Y , q ) = \epsilon , \qquad 0 < \epsilon \ll 1 , } \end{array}\tag{I.2}
$$

where ϵ is a fixed small positive constant. This choice keeps every capped outcome in the probability space with nonzero reward while strongly downweighting failures.

Showing how failure smoothing changes the target. To make the difference between the ideal valid-only target and the implemented target explicit, let $\mathcal { { D } } _ { q }$ be the valid completeprogram set and let $\mathcal { F } _ { q }$ contain all capped failure outcomes.

Their union $\overline { { \mathcal { Y } } } _ { q } = \mathcal { Y } _ { q } \cup \mathcal { F } _ { q }$ is the operational outcome space. Define

$$
\Pi _ { \alpha , \beta } ^ { \epsilon } ( Y \mid q ) \propto \overline { { { p } } } _ { \theta } ( Y \mid q ) ^ { \alpha } r _ { \beta } ( Y , q ) , \qquad Y \in \overline { { { \mathcal { D } } } } _ { q } .\tag{I.3}
$$

Here, $\overline { { p } } _ { \theta }$ is the probability distribution induced by running the base model under the operational caps. A rollout that produces a valid, fully terminated SVG before reaching any cap is assigned the probability of its generated token sequence. A rollout that reaches a cap or enters an unrecoverable malformed state is not discarded; its probability is assigned to the failure outcome ⊥. Thus, $\overline { { p } } _ { \theta }$ remains a normalized distribution over both valid completions and failed rollouts. The factor ${ \overline { { p } } } _ { \theta } ( Y \mid q ) ^ { \alpha }$ applies power sharpening, and $r _ { \beta } ( Y , q )$ supplies either the valid visual reward or the failure reward ϵ. Thus, Eq. I.3 is the exact target implemented by the capped decoder before normalization.

To quantify how much target probability is assigned to failures, let $Z _ { \mathrm { v a l } }$ and $Z _ { \mathrm { f a i l } }$ be the unnormalized mass of Eq. I.3 on $\mathcal { { D } } _ { q }$ and $\mathcal { F } _ { q } ,$ respectively. Then

$$
\delta _ { \epsilon } = \frac { Z _ { \mathrm { f a i l } } } { Z _ { \mathrm { v a l } } + Z _ { \mathrm { f a i l } } } , \qquad Z _ { \mathrm { f a i l } } = \epsilon \sum _ { Y \in \mathcal { F } _ { q } } \overline { { p } } _ { \theta } ( Y \mid q ) ^ { \alpha } .\tag{I.4}
$$

Here, $\delta _ { \epsilon }$ is the operational failure probability, and $Z _ { \mathrm { f a i l } }$ shows that failure mass is proportional to ϵ. Conditioning the operational target on $\mathcal { { D } } _ { q }$ exactly recovers the valid-only target. Equivalently, after extending the valid-only target by zero mass on $\mathcal { F } _ { q } ,$ its total-variation distance from the operational target is $\delta _ { \epsilon }$ , which vanishes as $\epsilon \to 0$

Keeping failed futures in the future-value expectation. To use the same future-value formula for both valid and failed continuations, we retain the telescoping visual product from Appendix D for a valid suffix C. For a failure suffix $C = \bot$ , we define $\Phi _ { > k } ^ { \epsilon } ( h b , \bot , q ) > 0$ so that the reward already accumulated by prefix $h b ,$ multiplied by this future factor, equals the fixed complete failure reward ϵ. The future value then has the uniform form

$$
\begin{array} { r l } & { V _ { k } ^ { \epsilon } ( h b , q ) = \mathbb { E } _ { C \sim \overline { { p } } _ { \theta } ( \cdot \vert h b , q ) } \left[ \overline { { p } } _ { \theta } ( C \mid h b , q ) ^ { \alpha - 1 } \Phi _ { > k } ^ { \epsilon } ( h b , C , q ) \right] } \\ & { \quad \quad \Phi _ { > k } ^ { \epsilon } > 0 . } \end{array}\tag{I.5}
$$

Here, $\overline { { p } } _ { \theta } ( C \mid h b , q )$ is the capped probability of future outcome C, the exponent α − 1 supplies the likelihood power not already absorbed by sampling, and $\Phi _ { > k } ^ { \epsilon }$ supplies the remaining visual or failure factor. The strict positivity statement ensures that the exact future value and its denominator remain defined. A malformed rollout is retained in the sample mean with a small contribution.

Preserving the target under modified rollout proposals. To preserve the same future-value target when render-guided pruning, early rejection, or other efficiency mechanisms change the rollout distribution, we let $q _ { \mathrm { r o l l } } ( C \mid h b , q )$ denote the actual rollout proposal and use

$$
V _ { k } ^ { \epsilon } ( h b , q ) = \mathbb { E } _ { C \sim q _ { \mathrm { r o l l } } ( \cdot | h b , q ) } \left[ \frac { \overline { { p } } _ { \theta } ( C \mid h b , q ) ^ { \alpha } } { q _ { \mathrm { r o l l } } ( C \mid h b , q ) } \Phi _ { > k } ^ { \epsilon } ( h b , C , q ) \right] .\tag{I.6}
$$

The numerator is the desired unnormalized contribution of $C ,$ and division by the actual proposal probability corrects its sampling frequency. The proposal must cover every positivetarget outcome. When $q _ { \mathrm { r o l l } } = \overline { { p } } _ { \theta }$ , one factor cancels and the expression reduces to Eq. I.5. In particular, averaging only render-filter survivors without this correction would estimate a different, survival-conditioned target.

Keeping selection valid when numerical correction fails. To keep selection defined when finite-precision arithmetic or the signed ratio correction fails, we distinguish three events. Let ${ \mathcal { E } } _ { 0 }$ be the event that the estimated total weight is zero or nonfinite. Let ${ \mathcal E } _ { \mathrm { c o r r } }$ be the event that the corrected probability vector has a negative coordinate, and let $\mathcal { E } _ { \mathrm { n u m } }$ be the event that the correction itself is numerically nonfinite. For a bounded candidate statistic $f ,$ the implemented estimator is

$$
\begin{array} { r l } & { T _ { \mathrm { a l g } } ( f ) = \overline { { f } } \mathbf { 1 } [ \mathcal { E } _ { 0 } ] + \widehat { T } _ { k } ^ { + } ( f ) \mathbf { 1 } [ \mathcal { E } _ { 0 } ^ { c } \cap \left( \mathcal { E } _ { \mathrm { c o r r } } \cup \mathcal { E } _ { \mathrm { n u m } } \right) ] } \\ & { \quad \quad \quad \quad + \widetilde { T } _ { k } ( f ) \mathbf { 1 } [ \mathcal { E } _ { 0 } ^ { c } \cap \mathcal { E } _ { \mathrm { c o r r } } ^ { c } \cap \mathcal { E } _ { \mathrm { n u m } } ^ { c } ] , } \end{array}\tag{I.7}
$$

Here, $\mathbf { 1 } [ \cdot ]$ is an event indicator, the superscript c denotes the complement of an event, and $\begin{array} { r } { \overline { { f } } = \bar { L ^ { - 1 } } \sum _ { i } \overline { { f ( B _ { i } } } ) } \end{array}$ is the statistic under a uniform distribution over the L parser-valid particles. Eq. I.7 implements three branches. If ${ \mathcal { E } } _ { 0 }$ occurs, selection is uniform over valid particles. If the total weight is valid but the correction is negative or nonfinite, the method uses the uncorrected self-normalized estimator $\widehat { T } _ { k } ^ { + }$ . Otherwise, it uses the corrected estimator $\widetilde { T } _ { k }$ and renormalizes the corrected vector in floating-point arithmetic.

Separating fallback error from estimator error. To separate the usual corrected-estimator error from the additional error caused by fallbacks, we define the numerically valid event $\mathcal { G } = \mathcal { E } _ { 0 } ^ { c } \cap \mathcal { E } _ { \mathrm { n u m } } ^ { c }$ and let $\| f \| _ { \infty } = \operatorname* { s u p } _ { b } | f ( b )$ |. Because this section makes failure smoothing explicit, we write the finite-horizon reference target as $T _ { k , H } ^ { \epsilon } ;$ the algorithmic estimators below are computed from the same ϵ-smoothed weights. Then

$$
\begin{array} { r l } & { | \mathbb { E } [ T _ { \mathrm { a l g } } ( f ) - T _ { k , H } ^ { \epsilon } ( f ) ] | \leq \left| \mathbb { E } [ ( \widetilde { T } _ { k } ( f ) - T _ { k , H } ^ { \epsilon } ( f ) ) \mathbf { 1 } [ \mathcal { G } ] ] \right| } \\ & { \phantom { \left| \mathbb { E } [ T _ { \mathrm { a l g } } ( f ) - T _ { k , H } ^ { \epsilon } ( f ) ] \right| } + 2 \| f \| _ { \infty } \{ \operatorname* { P r } ( \mathcal { E } _ { 0 } ) + \operatorname* { P r } ( \mathcal { E } _ { \mathrm { n u m } } ) \} } \\ & { \phantom { \left| \mathbb { E } [ T _ { k } ( f ) - T _ { k } ( f ) ] ^ { 2 } \right| } + \Big ( \mathbb { E } | \widehat { T } _ { k } ^ { + } ( f ) - \widetilde { T } _ { k } ( f ) | ^ { 2 } \Big ) ^ { 1 / 2 } } \\ & { \phantom { \left| \mathbb { E } [ T _ { k } ( f ) - T _ { k } ( f ) ] \right| } \times \operatorname* { P r } ( \mathcal { E } _ { \mathrm { c o r r } } ) ^ { 1 / 2 } . } \end{array}\tag{I.8}
$$

The three terms respectively cover the corrected estimator on valid numerical outcomes, zero or nonfinite weights, and fallback caused by a negative corrected coordinate. Thus, the leading-bias result applies on the regular event, while fallback frequencies remain a separate implementation diagnostic.

Keeping weights stable and visual scores comparable. To avoid numerical underflow, all candidate likelihoods, suffix likelihoods, visual factors, and importance weights are accumulated in log space and normalized with log-sum-exp. Padding tokens are excluded from both the likelihood and the model context.

To keep scores comparable, current candidates and future completions use the same canonicalization, viewport, background, raster resolution, parser, and renderer. Caching and batching do not remove generated tokens, renderer calls, or evaluator calls from the reported cost. Because the visual factors telescope, only the horizon-endpoint visual score is needed for the rollout weight; intermediate parsing still detects boundaries and failures.

## J. Complete CANVAS Procedure and Sources of Approximation

To provide an end-to-end specification of CANVAS, we assemble the preceding components into one navigation procedure, clarify which guarantees are exact and which rely on approximation, and state the shared experimental configuration.

Generating one SVG end to end. To generate one $\mathrm { { S V G } }$ CANVAS receives prompt $q ,$ candidate count $L ,$ pilot count $m _ { 0 } ,$ rollout horizon H, likelihood-sharpening parameter $\alpha ,$ visual-weight parameter $\beta ,$ failure reward $\epsilon ,$ and a rolloutand-rendering budget. At decision step k, let $h = H _ { k - 1 }$ be the current committed prefix. CANVAS repeats the following procedure at every parser-detected stroke boundary.

Step 1: Candidate generation. To approximate the full next-stroke candidate space, independently sample L parsercomplete candidate strokes from the actual valid proposal $q _ { k } ( \cdot , \mid h , q )$ . For each particle, compute the importance factor in Eq. F.1. Repeated token-identical candidates remain separate particles because they arose from separate proposal draws.

Step 2: Estimate where fresh rollouts are needed. To estimate how many fresh rollouts each candidate should receive, generate $m _ { 0 }$ capped pilot futures for every particle. Use Eq. G.1 to estimate its future value $\widehat { V } _ { i } ^ { ( 0 ) }$ , rollout variance ${ \widehat { \sigma } } _ { i } ^ { 2 }$ , and per-rollout cost $\widehat { c } _ { i }$ . A malformed or unrenderable pilot is retained and receives the failure reward ϵ from Eq. I.2.

Step 3: Budget allocation. To spend the fresh-rollout budget where it most reduces decision error, use Eq. G.6 to estimate each candidate’s decision-error coefficient. Determine the integer rollout count $M _ { i }$ under the lower, upper, and total-budget constraints using Eqs. G.7 and G.8. Freeze every $M _ { i }$ before any fresh outcome is generated.

Step 4: Estimate future value with independent rollouts. To obtain the final conditionally unbiased finite-horizon estimate, generate $M _ { i }$ new capped futures for candidate $B _ { i }$ independently of its pilots, and compute $\widehat { V } _ { i , H }$ using Eq. E.4. Pilot outcomes are not reused in this estimate.

Step 5: Combine current and future evidence. To combine the current-stroke factor with its estimated future value, compute the unnormalized particle mass $Y _ { i } = A _ { i } \widehat { V } _ { i , H }$ . Normalize these masses using Eq. 11 in the main paper to obtain the uncorrected particle distribution. Eq. F.4 guarantees unbiasedness of the numerator and denominator masses before this normalization.

Step 6: Reduce normalization bias. To reduce the leading bias introduced by the random normalizing denominator, compute the particle variance and covariance statistics in Eqs. H.1 and H.2. Apply Eq. 14 in the main paper to the indicator of each candidate. The resulting corrected coordinates form the signed vector ${ \widetilde { \rho } } .$

Step 7: Fallback and commit. To ensure that the committed choice always comes from a valid probability distribution, select the corrected, uncorrected, or uniform valid-particle distribution according to Eq. I.7. Sample the next stroke from that distribution and append it to the committed prefix. If an operational program cap is exhausted or the valid currentcandidate proposal has zero mass, record a capped-decoding failure rather than committing a malformed segment.

Step 8: Termination. To complete generation, repeat Steps 1–7 until the parser observes that all remaining SVG/XML tags are closed and EOS is generated. Return the resulting complete SVG program. Every committed prefix remains parser-valid; malformed or unrenderable hypothetical futures influence the lookahead estimate only through their small ϵ reward and are never committed directly.

Clarifying what is exact and what is approximate. To interpret the guarantees correctly, separate four levels. The complete-SVG target and its boundary-wise marginalization are exact. With fixed horizon and proposal support, candidate importance averages and independent fresh-rollout means are unbiased before normalization. Random normalization introduces ratio bias, for which Appendix H removes only the leading asymptotic term. Finite H, restricted support, ϵ-smoothing, integer allocation, and fallback are separate approximations. The visual evaluator also defines the reward being optimized; matching that target does not guarantee unmeasured human preference.

Explaining why correction and allocation are both needed. To clarify why both mechanisms are needed, the ratio correction reduces systematic bias from random normalization, whereas the adaptive counts $M _ { i }$ reduce finite-budget variance. Eq. 13 establishes the allocation result only for the stated first-order decision-error surrogate; downstream metrics also depend on reward fidelity, horizon H, and proposal coverage.

Keeping backbone comparisons reproducible. To make comparisons across backbones reproducible, the main experiments use four NVIDIA H200 GPUs and set $\alpha = 2$ On HeisenVec, we evaluate six representative autoregressive SVG backbones: IconShop [20], LLM4SVG [24], CodeLLaMA-7B, vHector-3B, OmniSVG [26], and vHector-8B [28]. On the unified evaluation set introduced by IntroSVG, we report the three backbones in its main comparison, OmniSVG, SVGen, and IntroSVG.

For every backbone, all model parameters remain frozen; CANVAS changes only the inference procedure and performs no fine-tuning or gradient update. Each backbone retains its native tokenizer, prompt format, and SVG serialization. Stroke boundaries are detected online by the incremental parser in Sec. 3.2, while benchmark prompts, rendering settings are all following original settings from both benchmarks.

## K. Additional Ablation Studies

In this section, we present additional ablation studies of the main design choices in CANVAS. Unless noted otherwise, all experiments use vHector-8B and the HeisenVec evaluation protocol from Tab. 1 in the main paper. Ours denotes the complete CANVAS configuration with H = 1 and α = 2. Arrows in the table headers indicate the preferred metric direction, and bold marks the best result in each table.

## K.1. Impact of the Rollout Horizon

In our framework, the rollout horizon H limits how many future strokes are evaluated after each current candidate, and we use $H = 1$ by default. To evaluate the impact of this choice, we compare three variants. H1(Ours), H2, and H3 allow at most one, two, and three additional parser-complete future strokes, respectively, while HEOS continues to valid EOS subject to the operational caps in Appendix B. Every variant stops earlier if valid EOS is reached. All reported runtimes were measured using four NVIDIA H200 GPUs.

As shown in Tab. 5, H1 achieves the best performance on most metrics while requiring only about 17.8 seconds per sample. Deeper rollouts improve a few individual metrics, but at a substantial runtime cost. These results show that H1 provides the strongest overall quality–efficiency balance, supporting its use as our default horizon.

## K.2. Evaluation on Decoding and Sampling Alternatives

In our framework, CANVAS uses parser-aligned, futureaware decisions rather than a conventional sequence decoder. To determine whether its gains could instead come from a generic decoding or sampling strategy, we compare it with native decoding, four sampling temperatures, beam search with width $B \ = \ 5$ , and Best-of-N selection with $N =$ $5 ,$ all using the same vHector-8B backbone. As shown in Tab. 6, temperature scaling and beam search improve only isolated metrics. Best-of-N is the strongest conventional alternative, but CANVAS improves eight of its 12 metrics and also achieves the best overall result on eight metrics. This comparison shows that CANVAS’s gains do not follow simply from sharpening the decoder, maintaining a beam, or selecting among more complete samples. Figure 3 provides a qualitative comparison on the same sample IDs.

## K.3. Impact of Power Sharpening and Rendered Visual Weighting

In our framework, power sharpening increases the relative weight of trajectories preferred by the base model, while rendered visual weighting favors complete SVGs that match the prompt. To evaluate the contribution of these two components, we compare native decoding, power-only sampling, and full CANVAS. As shown in Tab. 7, power-only sampling improves eight of the 12 metrics over native decoding, but degrades FID, HPSv2, SSIM, and LPIPS. Adding rendered visual weighting improves 11 of the 12 metrics over poweronly sampling and gives the best overall result on 10 metrics. These results support using the two signals together: power sharpening preserves the base model’s trajectory preference, while rendered feedback guides selection toward visually stronger complete SVGs.

## K.4. Impact of the Power Coefficient α

In our framework, the coefficient α controls how strongly the complete-sequence likelihood is sharpened, and we use $\alpha = 2$ by default. To evaluate this choice, we test $\alpha \in \{ 1 , 2 , 3 , 5 , 1 0 \}$ while keeping the remaining CANVAS components fixed. As shown in Tab. 8, $\alpha = 1$ gives the best FID and SSIM, while $\alpha = 3$ gives the best CLIP-S, DINO-S, and CVIU-C. The default $\alpha = 2$ leads the other seven metrics, including all four global-consistency scores SC, SL, CE, and LaDe. It therefore provides the strongest overall balance for the behavior targeted by CANVAS.

Table 5. Evaluation on the rollout horizon and inference time.
<table><tr><td></td><td colspan="6">Benchmark metrics</td><td colspan="6">Consistency metrics</td><td>Avg. time/sample (s)</td></tr><tr><td>Variant</td><td>CLIP-S↑</td><td>FID↓</td><td>HPSv2↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DINO-S↑</td><td> $\operatorname { L C I } _ { 9 \times 9 } \uparrow$ </td><td>CVIU-C↑</td><td>SC↑</td><td>SL↑</td><td>CE↑</td><td>LaDe↑</td><td>4×H200↓</td></tr><tr><td>H1(Ours)</td><td>72.324</td><td>151.634</td><td>20.219</td><td>59.516</td><td>56.073</td><td>59.400</td><td>0.7736</td><td>0.5256</td><td>0.9312</td><td>0.9274</td><td>0.9159</td><td>4.7362</td><td>≈17.8s</td></tr><tr><td>H2</td><td>69.977</td><td>102.383</td><td>15.806</td><td>62.604</td><td>59.617</td><td>56.048</td><td>0.7976</td><td>0.5811</td><td>0.5698</td><td>0.4184</td><td>0.6177</td><td>3.1752</td><td>≈38.2s</td></tr><tr><td>H3</td><td>71.046</td><td>198.614</td><td>17.844</td><td>63.681</td><td>54.196</td><td>65.620</td><td>0.8377</td><td>0.6180</td><td>0.6951</td><td>0.5182</td><td>0.7338</td><td>3.6844</td><td>≈63.4s</td></tr><tr><td>HEOS</td><td>70.358</td><td>205.196</td><td>17.300</td><td>61.466</td><td>55.391</td><td>59.878</td><td>0.8270</td><td>0.6212</td><td>0.6456</td><td>0.4816</td><td>0.6971</td><td>3.5956</td><td>≈227.5 s</td></tr></table>

Table 6. Evaluation on decoding and sampling alternatives.
<table><tr><td></td><td colspan="6">Benchmark metrics</td><td colspan="6">Consistency metrics</td></tr><tr><td>Variant</td><td>CLIP-S↑</td><td>FID↓</td><td>HPSv2↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DINO-S↑</td><td> $\operatorname { L C I } _ { 9 \times 9 } \uparrow$ </td><td>CVIU-C↑</td><td>SC↑</td><td>SL↑</td><td>CE↑</td><td>LaDe↑</td></tr><tr><td>Native decoding</td><td>66.787</td><td>117.741</td><td>15.213</td><td>66.229</td><td>57.599</td><td>49.488</td><td>0.6818</td><td>0.4535</td><td>0.5148</td><td>0.3436</td><td>0.4816</td><td>3.0000</td></tr><tr><td>Temperature (T = 1)</td><td>67.966</td><td>103.390</td><td>14.987</td><td>64.262</td><td>59.305</td><td>52.412</td><td>0.7542</td><td>0.5198</td><td>0.4897</td><td>0.3463</td><td>0.5288</td><td>2.8505</td></tr><tr><td>Temperature (T = 0.5)</td><td>67.252</td><td>110.003</td><td>15.144</td><td>65.867</td><td>57.643</td><td>51.360</td><td>0.7256</td><td>0.4810</td><td>0.5359</td><td>0.3579</td><td>0.5188</td><td>3.0506</td></tr><tr><td>Temperature (T = 0.2)</td><td>66.893</td><td>116.173</td><td>15.202</td><td>66.083</td><td>57.750</td><td>49.867</td><td>0.6896</td><td>0.4570</td><td>0.5239</td><td>0.3490</td><td>0.4896</td><td>3.0684</td></tr><tr><td>Temperature (T = 0.1)</td><td>66.923</td><td>117.194</td><td>15.217</td><td>66.221</td><td>57.597</td><td>49.766</td><td>0.6887</td><td>0.4547</td><td>0.5172</td><td>0.3470</td><td>0.4828</td><td>3.0075</td></tr><tr><td>Beam search (B = 5)</td><td>66.380</td><td>144.842</td><td>15.452</td><td>69.358</td><td>56.884</td><td>46.081</td><td>0.6115</td><td>0.3969</td><td>0.4959</td><td>0.3236</td><td>0.4771</td><td>3.0233</td></tr><tr><td>Best-of-N (N = 5)</td><td>68.582</td><td>102.333</td><td>15.734</td><td>64.739</td><td>57.978</td><td>56.817</td><td>0.7965</td><td>0.5622</td><td>0.6076</td><td>0.4386</td><td>0.6448</td><td>3.3594</td></tr><tr><td>Ours</td><td>72.324</td><td>151.634</td><td>20.219</td><td>59.516</td><td>56.073</td><td>59.400</td><td>0.7736</td><td>0.5256</td><td>0.9312</td><td>0.9274</td><td>0.9159</td><td>4.7362</td></tr></table>

![](images/70d5201827fdf7476d0302fe7a95f28e69b889c14767fe7aacee74d9b7bbff34.jpg)  
Figure 3. Qualitative evaluation on decoding and sampling alternatives. Rows compare Best-of-N (N = 5), beam search (B = 5), and CANVAS (Ours, H1) using the same vHector-8B backbone and sample IDs. From left to right, the prompts describe a laboratory scene, a traffic sign, a mobile phone, and a geometric star pattern. CANVAS more consistently preserves the requested content and global structure

## K.5. Impact of Adaptive Allocation and Estimator Reliability

In our framework, rollout counts depend jointly on estimated uncertainty, influence on the final decision, and rollout cost.

Table 7. Evaluation on power sharpening and rendered visual weighting.
<table><tr><td></td><td colspan="6">Benchmark metrics</td><td colspan="6">Consistency metrics</td></tr><tr><td>Variant</td><td>CLIP-S↑</td><td>FID↓</td><td>HPSv2↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DINO-S↑</td><td> $\mathrm { L C I } _ { 9 \times 9 } \ ^ { . }$  ←</td><td>CVIU-C↑</td><td>SC↑</td><td>SL↑</td><td>CE↑</td><td>LaDe↑</td></tr><tr><td>Power only</td><td>67.116</td><td>174.587</td><td>15.207</td><td>66.054</td><td>57.667</td><td>50.209</td><td>0.7031</td><td>0.4636</td><td>0.5247</td><td>0.3507</td><td>0.4978</td><td>3.0230</td></tr><tr><td>Native decoding</td><td>66.787</td><td>117.741</td><td>15.213</td><td>66.229</td><td>57.599</td><td>49.488</td><td>0.6818</td><td>0.4535</td><td>0.5148</td><td>0.3436</td><td>0.4816</td><td>3.0000</td></tr><tr><td>Full CANVAS(Ours)</td><td>72.324</td><td>151.634</td><td>20.219</td><td>59.516</td><td>56.073</td><td>59.400</td><td>0.7736</td><td>0.5256</td><td>0.9312</td><td>0.9274</td><td>0.9159</td><td>4.7362</td></tr></table>

Table 8. Evaluation on the power coefficient α.
<table><tr><td></td><td colspan="6">Benchmark metrics</td><td colspan="6">Consistency metrics</td></tr><tr><td>α</td><td>CLIP-S↑</td><td>FID↓</td><td>HPSv2↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DINO-S↑</td><td> $\mathrm { L C I } _ { 9 \times 9 }$  个</td><td>CVIU-C↑</td><td>SC↑</td><td>SL↑</td><td>CE↑</td><td>LaDe↑</td></tr><tr><td>1</td><td>69.961</td><td>103.224</td><td>15.799</td><td>66.666</td><td>56.861</td><td>57.000</td><td>0.7543</td><td>0.5212</td><td>0.6370</td><td>0.4841</td><td>0.6337</td><td>3.3971</td></tr><tr><td>2 (Ours)</td><td>72.324</td><td>151.634</td><td>20.219</td><td>59.516</td><td>56.073</td><td>59.400</td><td>0.7736</td><td>0.5256</td><td>0.9312</td><td>0.9274</td><td>0.9159</td><td>4.7362</td></tr><tr><td>3</td><td>74.045</td><td>143.156</td><td>18.278</td><td>54.977</td><td>59.161</td><td>61.718</td><td>0.6924</td><td>0.5401</td><td>0.8396</td><td>0.6892</td><td>0.8094</td><td>4.4342</td></tr><tr><td>5</td><td>71.066</td><td>142.134</td><td>18.420</td><td>57.578</td><td>57.494</td><td>57.055</td><td>0.7062</td><td>0.4452</td><td>0.8145</td><td>0.6979</td><td>0.6661</td><td>4.0851</td></tr><tr><td>10</td><td>72.122</td><td>139.098</td><td>19.051</td><td>58.394</td><td>56.643</td><td>57.541</td><td>0.7095</td><td>0.4420</td><td>0.8365</td><td>0.6805</td><td>0.7462</td><td>4.4342</td></tr></table>

Pilot rollouts determine these counts, fresh rollouts estimate candidate values, and an analytic correction reduces the leading bias from random normalization. To evaluate these choices, we replace the allocation with uniform or variance-only allocation, remove its cost term, reuse pilot outcomes, remove the ratio correction, or remove both the pilot/fresh split and the correction. As shown in Tab. 9, individual variants lead on isolated metrics, but the complete estimator achieves the overall best performance. This result supports combining influence- and cost-aware allocation with fresh evaluation and ratio correction, particularly for global consistency.

## K.6. Impact of the Decision Unit

In our framework, parser-detected stroke boundaries define the decision unit. To evaluate this choice, we compare strokewise decisions with token-wise decisions and fixed-length blocks of m = 20 tokens. As shown in In Tab. 10, tokenwise and fixed-length decisions achieve the best results on a few metrics but incur substantial additional runtime. Stroke-wise CANVAS takes approximately 17.8 seconds and achieves the best result on most of the metrics. Fixed boundaries may also split a coordinate, path command, or XML attribute. These results support parser-detected strokes as a meaningful and efficient decision unit.

## L. Benchmark Datasets and Official Evaluation Metrics

We retain the official evaluation protocols and metric directions of vHector/HeisenVec [28] and IntroSVG [19].

HeisenVec. HeisenVec is a million-scale, richly captioned SVG dataset spanning diverse visual styles and sequence lengths, with samples extending to 32k tokens, and is designed for long-context Text-to-SVG modeling [28]. Following its official benchmark, we report six complementary measures. CLIP Score (CLIP-S) is the cosine similarity between CLIP image and text embeddings and measures alignment between a rendered SVG and its source caption. Fréchet Inception Distance (FID) compares the distributions of Inception-v3 features from generated and reference renders; lower values indicate closer distributional alignment. Human Preference Score v2 (HPSv2) is a learned captionimage preference surrogate trained to approximate which output a human would favor. Structural Similarity Index Measure (SSIM) compares local luminance, contrast, and structure between a generated render and its paired reference. Learned Perceptual Image Patch Similarity (LPIPS) measures perceptual distance between the pair using learned deep image features. DINO Similarity (DINO-S) compares their DINO feature representations and provides a complementary feature-level fidelity measure. Higher values are preferred for CLIP-S, HPSv2, SSIM, and DINO-S, whereas lower values are preferred for FID and LPIPS.

IntroSVG. IntroSVG constructs a unified held-out evaluation set of 1,400 samples, stratified across the data sources used by LLM4SVG, OmniSVG, and SVGen and excluded from its training corpus [19]. Average token count (Avg. Token) is the length of the generated SVG after tokenization by the Qwen2.5 tokenizer and measures code conciseness. Render Success Rate (RSR) is the percentage of generated SVG programs that CairoSVG renders successfully. FID has the distributional interpretation given above. CLIP text-to-image similarity (CLIP-T2I) measures semantic alignment between the prompt and rendered SVG through CLIP embeddings. Aesthetic Score applies a pretrained image-aesthetic predictor to the render, while HPS applies a learned caption-image human-preference predictor. Lower values are preferred for Avg. Token and FID, while higher values are preferred for RSR, CLIP-T2I, Aesthetic Score, and HPS.

Table 9. Evaluation on adaptive allocation and estimator-reliability components.
<table><tr><td></td><td colspan="6">Benchmark metrics</td><td colspan="6">Consistency metrics</td></tr><tr><td>Variant</td><td>CLIP-S↑</td><td>FID↓</td><td>HPSv2↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DINO-S↑</td><td> $\operatorname { L C I } _ { 9 \times 9 } \uparrow$ </td><td>CVIU-C↑</td><td>SC↑</td><td>SL↑</td><td>CE↑</td><td>LaDe↑</td></tr><tr><td>Uniform allocation</td><td>72.913</td><td>135.835</td><td>19.080</td><td>58.664</td><td>55.298</td><td>60.148</td><td>0.6721</td><td>0.4705</td><td>0.8909</td><td>0.7479</td><td>0.8618</td><td>4.5601</td></tr><tr><td>Variance-only allocation</td><td>71.651</td><td>140.309</td><td>19.511</td><td>62.616</td><td>57.320</td><td>67.808</td><td>0.6637</td><td>0.5588</td><td>0.8575</td><td>0.7600</td><td>0.8814</td><td>4.3685</td></tr><tr><td>Influence without cost</td><td>73.173</td><td>139.924</td><td>17.332</td><td>54.226</td><td>61.401</td><td>51.610</td><td>0.6919</td><td>0.4426</td><td>0.7591</td><td>0.6400</td><td>0.7244</td><td>3.9695</td></tr><tr><td>Reuse pilot outcomes</td><td>74.017</td><td>135.622</td><td>18.926</td><td>54.611</td><td>58.873</td><td>61.841</td><td>0.7215</td><td>0.5082</td><td>0.8046</td><td>0.7479</td><td>0.8499</td><td>4.0851</td></tr><tr><td>No ratio correction</td><td>72.859</td><td>136.713</td><td>18.657</td><td>54.807</td><td>62.281</td><td>63.613</td><td>0.6588</td><td>0.4631</td><td>0.8046</td><td>0.7520</td><td>0.7718</td><td>4.0851</td></tr><tr><td>No split or correction</td><td>73.423</td><td>136.300</td><td>17.969</td><td>52.747</td><td>61.687</td><td>56.415</td><td>0.7038</td><td>0.5077</td><td>0.7700</td><td>0.6979</td><td>0.7156</td><td>3.9297</td></tr><tr><td>Ours</td><td>72.324</td><td>151.634</td><td>20.219</td><td>59.516</td><td>56.073</td><td>59.400</td><td>0.7736</td><td>0.5256</td><td>0.9312</td><td>0.9274</td><td>0.9159</td><td>4.7362</td></tr></table>

Table 10. Evaluation on token-wise, fixed-length, and stroke-wise decision units.
<table><tr><td></td><td colspan="6">Benchmark metrics</td><td colspan="6">Consistency metrics</td><td>Avg. time/sample (s)</td></tr><tr><td>Variant</td><td>CLIP-S↑</td><td>FID↓</td><td>HPSv2↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DINO-S↑</td><td> $\mathbf { L C I _ { 9 \times 9 } } \uparrow$ </td><td>CVIU-C↑</td><td>SC↑</td><td>SL↑</td><td>CE↑</td><td>LaDe↑</td><td>4×H200↓</td></tr><tr><td>Token-wise decision</td><td>68.878</td><td>66.574</td><td>13.753</td><td>69.913</td><td>62.766</td><td>32.981</td><td>0.8801</td><td>0.3121</td><td>0.4723</td><td>0.2663</td><td>0.5230</td><td>2.6875</td><td>≈7,112.4s</td></tr><tr><td>Fixed block  $( m = 2 0 )$ </td><td>71.326</td><td>89.597</td><td>14.395</td><td>62.912</td><td>61.039</td><td>56.761</td><td>0.8195</td><td>0.6021</td><td>0.6355</td><td>0.3769</td><td>0.6657</td><td>3.7224</td><td>≈165.9s</td></tr><tr><td>Ours</td><td>72.324</td><td>151.634</td><td>20.219</td><td>59.516</td><td>56.073</td><td>59.400</td><td>0.7736</td><td>0.5256</td><td>0.9312</td><td>0.9274</td><td>0.9159</td><td>4.7362</td><td>≈17.8s</td></tr></table>

## M. Definitions and Qualitative Interpretation of the Consistency Metrics

To make the six additional consistency metrics in the main paper easy to interpret, this section identifies the source of each metric, gives its precise definition and dataset-level aggregation, and then explains what a higher score looks like in practice. The metrics measure consistency in two different ways. SC, SL, CE, and LaDe use visual-judging criteria adapted to our setting, whereas $\mathrm { L C I _ { 9 \times 9 } }$ and CVIU-C compute consistency directly from image edges using fixed deterministic rules. All six are higher-is-better. SC, SL, CE, $\mathrm { L C I _ { 9 \times 9 } }$ , and CVIU-C lie in [0, 1], while LaDe lies in [1, 5].

Two complementary evaluation protocols. We first evaluate SC, SL, CE, and LaDe as VLM-judged metrics. SC, SL, and CE adapt the corresponding symbolic-visual rubrics from Latent Canvas [27], while LaDe adapts the cross-layerconsistency stage of the layered-design judge in Lungu-Stan et al. [13]. For each sample and method variant, a single blinded GPT-5.5 call jointly returns all four scores in one JSON response. The judge receives the exact caption, the final SVG rendered on a white 512 × 512 canvas, SVG source statistics, and at most six pseudo-layers formed by grouping contiguous drawables in z-order. It is also shown a second rendering generated from the same caption solely to calibrate failure severity; this image is not ground truth and is not assumed to be better. Because these four metrics are defined by frozen VLM rubrics rather than closed-form pixel calculations, the corresponding subsections quote their metric-specific prompt clauses separately so that each definition can be read in place. This separation is only for presentation: all four scores are obtained together in the

same judge call.

In contrast, $\mathrm { L C I _ { 9 \times 9 } }$ and CVIU-C are deterministic metrics computed from image edges by explicit formulas, without a VLM judge. We apply the same frozen raster-to-edge adapter to every method: the 512 × 512 white-background rendering is resized to 128 × 128, its RGB distance from white is normalized at the 99.5th percentile, the result is smoothed with a Gaussian kernel of $\sigma = 0 . 8$ , and a Canny edge map is extracted with thresholds 32 and 96. The subsections below define how $\mathrm { L C I _ { 9 \times 9 } }$ and CVIU-C summarize different properties of this common edge map.

For any one of the six metrics, let $m _ { n }$ be its score on evaluation sample n. The value reported in the tables is the arithmetic mean over the N evaluated samples:

$$
\overline { { m } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } m _ { n } .\tag{M.1}
$$

For readability, table headers use the metric name without the overbar.

How to read the qualitative comparisons. Figures 4– 9 show two same-prompt, same-sample-ID pairs for each metric. “Higher” and “Lower” compare only the two outputs within one pair; they do not denote dataset-wide maxima or minima. Every displayed SVG is render-valid and nonempty, and its content is shown unchanged on the same white canvas without cropping. The examples are chosen to make the property targeted by each metric visible.

## M.1. Structural Coherence (SC)

Latent Canvas defines Structural Coherence as a [0, 1] VLMjudge score for whether lines are connected and shapes are closed and logically constructed [27]. We transfer this rubric from ASCII art to SVG while retaining the same score range. SC clause in the frozen joint judge prompt. “latent\_structural\_coherence [0, 1], adapted from Latent Canvas: are contours, lines and shapes connected/closed when they should be, stable, recognizable, andfree ofbroken or disconnected construction?”

This clause asks whether an SVG’s local geometry forms stable, recognizable objects. It checks whether contours and parts join, close, and remain separate where the caption requires, and penalizes broken or unclosed contours, unintended gaps, accidentally fused parts, and unstable shape construction.

A higher SC score therefore means that the intended contours can be followed and the depicted parts form a coherent object; it does not simply reward greater detail. In Figure 4, the higher-scoring star remains one closed shape instead of fragmenting into displaced pieces. Likewise, the higher-scoring human silhouettes remain distinct rather than colliding and merging.

## M.2. Spatial Logic (SL)

Latent Canvas defines Spatial Logic as a [0, 1] VLM-judge score for whether object parts have anatomically or geometrically correct relative positions [27]. We retain this score range when transferring the rubric to SVG.

SL clause in the frozen joint judge prompt. “latent\_spatial\_logic [0, 1], adapted from Latent Canvas: are anatomy, geometry, counts, relative positions, containment, pose and placement logically correctfor this exact caption?”

This clause evaluates object and part counts, relative position, containment, pose, direction, anatomy, and geometric placement. Unlike SC, which focuses on whether shapes are structurally intact, SL asks whether the right entities appear in the right roles and locations specified by the caption.

A higher SL score means that the requested arrangement and component relations are represented correctly. In Figure 5, Sample 0529 tests whether the person is actually placed on the left. Sample 1074 tests whether the components form a bow and arrow rather than a clean but semantically different four-way arrow symbol.

## M.3. Character Efficiency (CE)

Latent Canvas defines Character Efficiency as a [0, 1] VLMjudge score for visual cleanliness, with penalties for character spam, grid filling, and unnecessary artifacts [27]. We retain this score range but transfer the notion of character efficiency to the graphical elements and source structure of an SVG.

CE clause in the frozen joint judge prompt. “latent\_character\_efficiency [0, 1], adapted from Latent Canvas: is the SVG clean and economical, without visual noise, repetitive path spam, duplicated marks or unnecessary artifacts? Use the supplied source statistics as evidence but do not reward an empty or under-detailed graphic.”

This clause evaluates whether the visible structure is produced by purposeful elements rather than redundant code or over-drawing. It receives the common visual inputs together with shape, path, stratum, duplicate, and character counts. It penalizes repeated markers, unnecessary details, visual noise, and local path spam, while explicitly avoiding a trivial preference for blank or under-detailed outputs.

A higher CE score therefore indicates that most generated elements make a useful contribution to the final image. In Figure 6, the higher-scoring outputs use three graphic elements in each pair. Their lower-scoring counterparts use 152 and 167 elements, respectively, to create dense grids or repeated hatching that the prompts do not require.

## M.4. LaDe Cross-Layer Consistency

LaDe defines a 1–5 VLM-judge rubric for layered designs that includes cross-layer consistency among its evaluation stages [13]. We retain this score range while transferring that stage to SVG pseudo-layers.

LaDe clause in the frozen joint judge prompt. “lade\_cross\_layer\_consistency [1, 5], adapted from LaDe: across the ordered pseudo-layers, assess color harmony, style, perspective, scale, occlusion/depth, foreground/background relations and whether componentsform one coherent composite. 1 is severe conflict; 5 isfully consistent.”

This clause measures whether separately drawn components combine into one coherent scene. The original LaDe protocol evaluates semantic RGBA layers as part of a broader layered-design rubric and uses different judges. Our flat-SVG evaluation instead groups contiguous drawables into ordered pseudo-layers, so the resulting score is a transfer proxy rather than the native LaDe score.

![](images/0de3bcd939308a26c3fb585604db5d9bfc9f49a22f82ae8fcb48a464f4dd48a4.jpg)

Figure 4. Structural Coherence (SC). Sample 5756 requests a stylized star on a blue background, and Sample 3546 requests three simple black human silhouettes. The higher-SC outputs preserve closed contours and keep distinct parts separate; the lower-SC outputs exhibit fragmentation or unintended fusion.

![](images/7a2e867d7cb423baf7083ff927a6377d4b2ba1bd4b9e2894bb412e685935b161.jpg)  
Figure 5. Spatial Logic (SL). Sample 0529 requests a person standing on the left, and Sample 1074 requests a line-drawn bow and arrow. The higher-SL outputs satisfy the requested placement and component relations; the lower-SL outputs change object presence, direction, or component identity.

![](images/ad11d7ef2c1e7e101e0b64b95c25c3fef7c574350775ba7da6dc711541c844a7.jpg)  
Figure 6. Character Efficiency (CE). Sample 0653 requests a simple graph with one blue line, and Sample 1541 requests a geometric figure made of several lines and shapes. The higher-CE outputs use a small set of purposeful elements, whereas the lower-CE outputs are dominated by unnecessary grids or repeated strokes. All four outputs are render-valid and nonempty.

A higher transferred LaDe score means that components agree in scale and appearance and follow a sensible front-toback order. In Figure 7, the higher-scoring graduation figure aligns its cap, head, gown, and arms, while the lower-scoring output overlays them at incompatible scales. The building pair similarly reveals differences in roof, facade, entrance, and foreground occlusion.

## M.5. Local Connectivity Index $\mathbf { ( L C I _ { 9 \times 9 } ) }$

To measure local contour connectivity, $\mathrm { L C I _ { 9 \times 9 } }$ adapts the grid-based line-connectivity measure of Feng et al. [5]. Divide the shared edge map into a $9 \times 9$ grid and let $\mathscr { G } _ { + }$ be the set of nonempty tiles. For each $g \in { \mathcal { G } } _ { + }$ , let $n _ { g }$ be its number of edge pixels and let $c _ { g }$ be the number of pixels in its largest 8-connected component. For a nonempty edge map, we compute

$$
\mathrm { L C I } _ { 9 \times 9 } = \frac { 1 } { \vert \mathscr { G } _ { + } \vert } \sum _ { g \in \mathscr { G } _ { + } } \frac { c _ { g } } { n _ { g } } \in [ 0 , 1 ] .\tag{M.2}
$$

Each tile contributes the fraction of its edges belonging to its dominant connected component, and the final score averages these fractions over occupied tiles.

A higher $\mathrm { L C I _ { 9 \times 9 } }$ score means that local regions are usually dominated by one connected contour rather than several detached or competing fragments. It does not directly measure prompt agreement, detail count, or how widely the drawing covers the canvas. Figure 8 illustrates this distinction with a continuous house outline versus detached roof and doorway fragments, and with a simple map-marker contour versus several nearby boundaries.

## M.6. CVIU Complexity (CVIU-C)

To measure whether an SVG combines regular local edge structure with nondegenerate spatial coverage, CVIU-C adapts the reference-free Statistical Complexity Measure of Gimenez et al. [7]. Let $\mathcal { P }$ be the set of edge pixels in the shared edge map, $N _ { p }$ the vectorized $7 \times 7$ neighborhood around edge pixel $p ,$ and $\tau$ the bank of 140 local line patterns from the original measure. Let $\mathcal { U }$ denote the uniform distribution over canvas locations, and let $D _ { \mathrm { 2 D } } ( \mathcal { P } , \mathcal { U } ) \in [ 0 , 1 ]$ be the maximum two-dimensional empirical-CDF discrepancy between the observed edge locations and U. We compute

$$
\begin{array} { r l r } {  { E = \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } \operatorname* { m a x } _ { \tau \in \mathcal { T } } \frac { \langle N _ { p } , \tau \rangle } { \| N _ { p } \| _ { 2 } \| \tau \| _ { 2 } } , } } \\ & { } & { H = 1 - D _ { 2 \mathrm { D } } ( \mathcal { P } , \mathcal { U } ) , \quad \mathrm { ~ C V I U - C = E } H \in [ 0 , 1 ] . } \end{array}\tag{M.3}
$$

The equilibrium term $E$ is the average best match to a regular local line pattern. The information term H decreases when edges collapse onto one axis or a small region.

A higher CVIU-C score therefore requires both clean local edge motifs and sufficiently rich two-dimensional coverage. This differs from $\operatorname { L C I } _ { 9 \times 9 } { \mathrm { : } }$ a small or one-axis drawing may be locally connected yet still receive low CVIU-C because its $H$ term is small. Figure 9 shows this difference through a distributed bar chart versus a concentrated arrow-like axis, and a broad building facade versus a narrow vertical edge pattern.

![](images/e6633a67c5941d3b477c402102f94ba8d27a730c5d3ed800c0cbcb8578c7dafb.jpg)  
Figure 7. LaDe Cross-Layer Consistency. Sample 3293 requests a person wearing a graduation cap and gown, and Sample 2693 requests a two-dimensional building. The higher-LaDe outputs combine their parts at compatible scales and with clearer occlusion order; the lower-LaDe outputs contain conflicting overlays and front-to-back relations.

![](images/4cbe1483a020cde6b0d364484a3b3c454d401c7978eb6c4335c3be2872acf320.jpg)  
Figure 8. Local Connectivity Index (LCI<sub>9×9</sub>). Sample 0543 requests a simple house outline, and Sample 1245 requests a minimalist map marker. The higher-LCI outputs contain one dominant connected contour within most occupied grid cells; the lower-LCI outputs contain detached or competing local edge components.

![](images/9278beb8477131198ae633857662d2f2e1e3634161425a5b7942b4a488f94c4d.jpg)  
Figure 9. CVIU Complexity (CVIU-C). Sample 0557 requests a minimalist vertical bar chart, and Sample 5291 requests a two-dimensional building. The higher-CVIU-C outputs distribute regular edge motifs across both canvas axes; the lower-CVIU-C outputs concentrate most edge information along one axis or within a narrow region.