# BEYOND QUADRATIC ATTENTION: SEPARABLEGAUSSIAN OPERATORS FOR ODE VISION TRANS-FORMERS

Carlos Boned Riera<sup>∗</sup> Universitat Autonoma de Barcelona\` Centre de Visio per Computador (CVC)´ cboned@cvc.uab.cat

David Romero Sanchez´ Universitat Autonoma de Barcelona\` Centre de Recerca Matematica (CRM)\` dromero@crm.cat

Oriol Ramos Terrades Universitat Autonoma de Barcelona\` Centre de Visio per Computador (CVC)´ oriolrt@cvc.uab.cat

CIFAR-100 top-1 accuracy  
![](images/a46f7d0556276e7b8eeef214bcb7f9ea5a88eb835bbd3ef0bfffc8bd5627aa67.jpg)  
Figure 1: Left: throughput–memory trade-off at resolution 1024 as the width of the model increases. Points farther upward provide more throughput and points farther left allocate less memory. Our proposed method dominates multi-head attention, achieving higher throughput at lower memory. Right: comparison of the three model families with our proposed method.

## ABSTRACT

Self-attention carries a quadratic cost in token count, which limits how far Vision Transformers can scale with input resolution. We introduce SSoG, a separable sum-of-Gaussians spatial operator that replaces dense self-attention with a learned mixture of axis-factorized Gaussian kernels, reducing the per-layer cost from $\mathcal { O } ( N ^ { 2 } )$ to subquadratic in N. Unlike dot-product self-attention, whose Lipschitz constant is unbounded outside a compact input domain, SSoG with fixed kernels is globally Lipschitz continuous, which, via the Picard–Lindelof theorem,¨ guarantees that the continuous-depth variant has a unique solution on the whole integration interval, a global guarantee that dense-attention counterparts lack in general. On CIFAR-100 at 224 × 224, in a matched discrete-Transformer setting and in the continuous setting, SSoG improves Top-1 accuracy by 2.2–4.1 points across three model widths while using 8.8–16.6% fewer parameters, with depth and training recipe held fixed. In a looped-Transformer setting, SSoG improves

Top-1 accuracy by 5.4–8.1 points. We further probe inference-time scaling by evaluating on inputs up to 1024 × 1024 at matched batch size: in the continuousdepth (Neural ODE) architecture, SSoG reaches up to 154% higher throughput and over 60% lower peak memory than dense attention, with the gap widening as resolution grows, consistent with the difference in asymptotic cost. In the discrete and looped settings, SSoG also achieves over 30% higher throughput than dense attention at this resolution. Together, these results establish SSoG as a practical, parameter-efficient, and theoretically grounded alternative to dense attention across discrete, weight-shared, and continuous-depth Transformer variants.

## 1 INTRODUCTION

The self-attention mechanism, implemented through scaled dot-product attention (SDPA) Vaswani et al. (2017), has marked a paradigm shift in AI. ViT Dosovitskiy et al. (2021) showed that this SDPA mechanism could scale to images, and this led to remarkably impressive results across a wide range of vision tasks. Foundation models such as DINO Caron et al. (2021), SAM Kirillov et al. (2023) for segmentation and DiT Peebles & Xie (2023) for diffusion, as well as VLMs and LLMs, demonstrated that this attention mechanism could scale to results beyond what seemed possible. Alongside these strengths, however, come several problems: the complexity of self-attention itself, a limited understanding of how a model learns to attend, difficulty training these models due to theo retical issues inherent to SDPA, and the emergence of artifacts Darcet et al. (2024); Qiu et al. (2025), among others. These issues have mostly been studied in isolation. In this paper, we show that reducing the complexity of self-attention and satisfying the theoretical property required for Lipschitz continuity can be achieved by the same operator, and we study the behaviour of this operator when instantiated across discrete, looped, and continuous-depth Transformers.

The complexity of SDPA arises from the token sequence length. For N tokens of dimension d, SDPA needs $\dot { \mathcal { O } } ( N ^ { 2 } )$ memory and $\mathcal { O } ( N ^ { 2 } d )$ computation, and in vision N grows with the square of the image side, so the cost increases with the resolution. Research on context compression Zhang et al. (2025); Liu et al. (2026) reduces the memory needed to store the matrix, reducing the effective sequence length, but the operation is still quadratic. State Space Models Gu & Dao (2024); Zhu et al. (2024) drop pairwise interaction altogether and are linear in N; however, they tend to be harder to train and imply a complete change of architecture. Looped Transformers Byra et al. (2026) and Neural ODEs Zhong et al. (2022) work on a different axis: they share parameters across depth, which reduces model size but not the cost of a single evaluation. Moreover, these models are sensitive to hyperparameters, and noise in the parameter space is carried through all the optimization steps and cannot be isolated.

Beyond the cost of recurrent and continuous models, they inherit a second, more fundamental problem from dense attention itself: attention scores are computed over an unbounded domain. Since the query-key interaction is an inner product between unconstrained vectors, SDPA is not globally Lipschitz continuous Kim et al. (2021). This is part of why the literature argues that Transformers are sensitive to hyperparameters, initialization, and learning rate Qi et al. (2023). These theoretical implications matter most for continuous-depth models. Dense attention is smooth, and hence Lipschitz on bounded sets, so the Picard–Lindelof theorem still guarantees a unique solution locally in¨ time. What global Lipschitz continuity adds is existence of the solution on the whole integration interval for every initial condition, without finite-time blow-up, together with a uniform bound on how much two trajectories can drift apart when their initial conditions differ. Recurrent and continuous architectures, which apply the same block many times, are therefore the ones that benefit most from a property that dense attention lacks.

Pisoni (2026) recently outlined SSoG, a Sum of Separable Gaussians operator that replaces dotproduct attention in standard ViTs with a small mixture of anisotropic, axis-factorized Gaussian kernels, in a fixed and a content-steered variant, and reported preliminary results on image classification. Because the kernel factorizes along the image axes, SSoG never builds the $N \times N$ attention matrix and is subquadratic in N. We study it in a regime for which it was neither designed nor analyzed: as the token mixer of looped and continuous-depth Transformers. In this paper, we show that SSoG with fixed kernels is globally Lipschitz continuous by construction, precisely the property that looped and continuous-depth models require, while its content-steered variant is Lipschitz on sets of bounded token norms; we are the first to study it inside these architectures. Unlike MHA, where tokens compete for a fixed normalization budget through content-dependent scores, fixed SSoG assigns weights from a spatial kernel of the relative offsets, so no token can take mass away from another, leaving no mechanism for an attention sink to form. We confirm this empirically on the trained models: dense attention develops a persistent sink on ImageNet, whereas SSoG does not (Appendix A.2). We benchmark this instantiation of SSoG against dense attention across discrete, looped, and continuous-depth settings, providing the first empirical and theoretical characterization of its behavior as a weight-shared, continuous-depth attention mechanism. We also find that content steering improves the discrete and looped models but not the continuous-depth one, and that the efficiency gains of SSoG grow with input resolution.

## Contributions.

• We prove that fixed SSoG is globally Lipschitz by construction, which makes continuousdepth models built on it well posed on the whole integration interval, and that the contentsteered variant is Lipschitz on bounded sets (Appendix A.6).

• We use SSoG, in fixed and content-steered variants, as a drop-in replacement for dense attention in discrete, looped and continuous-depth vision Transformers, and benchmark it extensively against dense attention in each setting.

• We study the accuracy-compute trade-off of ODE-SSoG in integration depth and number of atoms, and show its sub-quadratic operator gains over dense attention in throughput and memory as resolution grows.

## 2 METHOD

We follow the formulation of Pisoni (2026) and restate it with the notation used in our analysis. In contrast to conventional self-attention, SSoG does not construct query-key similarities or explicitly materialize an $N \times N$ attention matrix. Instead, each head mixes value features through a small collection of learned anisotropic Gaussian spatial kernels. These kernels factorize along the x and y axes, so two-dimensional mixing reduces to successive one-dimensional contractions.

Each Gaussian atom, r, provides a compact geometric description of a spatial interaction pattern. Its center determines where information is gathered from, its spread determines how local or global the interaction is, and its mixture weight determines the contribution of that pattern to the head output. Let $\boldsymbol { X } \in \mathbb { R } ^ { B \times N \times D }$ denote the sequence of input tokens, where the N−1 tokens are patch tokens and one is a class token and let H be the number of heads with head dimension $D ^ { h } = \bar { D ^ { \prime } } H$ . Unlike dotproduct self-attention, SSoG contains no query or key projections. For each head $\dot { h } \in \{ 1 , \dots , H \}$ a learned projection $W _ { V } ^ { h } \in \mathbb { R } ^ { D \times D ^ { h } }$ produces:

$$
V ^ { h } = X W _ { V } ^ { h } ,\tag{1}
$$

Stacking all heads gives $V \in \mathbb { R } ^ { B \times N \times H \times D ^ { h } }$ . The class token is separated before spatial filtering, while the remaining $N - 1 = n _ { y } n _ { x }$ patch tokens are reshaped onto a $n _ { y } \times n _ { x }$ spatial grid.

Gaussian spatial kernels. Each head h contains R Gaussian atoms. Atom r is parameterized by a center $\mu _ { r } = ( \mu _ { r , x } , \mu _ { r , y } )$ , an anisotropic spread $\boldsymbol { \sigma } _ { r } = ( \sigma _ { r , x } , \sigma _ { r , y } )$ , and a mixture logit $\ell _ { r }$ , from which the mixture weights λ are obtained by softmax over atoms. For axis $a \in \{ x , y \}$ and offset $\delta _ { a } = q _ { a } - s _ { a }$ , the unnormalized Gaussian log-kernel is:

$$
\begin{array} { r } { \log G ^ { a } ( \delta _ { a } ) = - \frac { 1 } { 2 } \log ( 2 \pi ) - \log \sigma _ { a } - \frac { ( \delta _ { a } - \mu _ { a } ) ^ { 2 } } { 2 \sigma _ { a } ^ { 2 } } . } \end{array}\tag{2}
$$

The kernel is normalized over source coordinates using a learned positive temperature $\tau _ { \ast }$

$$
A _ { a } ( \delta ) = \mathrm { s o f t m a x } \left( \frac { \log G ^ { a } ( \delta ) } { \tau } \right) .\tag{3}
$$

To ensure valid, non-degenerate kernels, $\sigma _ { a }$ and $\tau$ are obtained from unconstrained learned counterparts $\tilde { \sigma } _ { a }$ and $\tilde { \tau }$ via a positive reparameterization, with $\sigma _ { \operatorname* { m i n } } > 0$ bounding each axis-wise spread

from below:

$$
\sigma _ { a } = \mathrm { s o f t p l u s } ( \tilde { \sigma } _ { a } ) + \sigma _ { \mathrm { m i n } } + \varepsilon
$$

$$
\tau = \mathrm { s o f t p l u s } ( \tilde { \tau } ) + \tau _ { 0 } ,\tag{4}
$$

(5)

$$
\lambda = \operatorname { s o f t m a x } _ { r } ( \ell )\tag{6}
$$

Here $\tau _ { 0 } > 0$ and $\varepsilon > 0$ prevent over-sharpening of the axis-wise softmax and ensure numerical stability of the width constraint, respectively. Under the offset convention $\delta = q - s ,$ a center $\mu$ selects source positions near $s = q - \mu { \mathrm { . } }$ , so the learned centers directly encode the relative spatial locations from which each head gathers information.

We study two variants of this operator: fixed SSoG and content-steered, described below. In fixed SSoG the geometric parameters, centers, spreads, and mixture weights, are shared across all images and query positions. In the content-steered variant, termed LookAt, bounded input-dependent perturbations adapt the geometry per query patch while preserving the separable structure.

## 2.1 FIXED SSOG

In fixed SSoG, the atom geometry and mixture weights are shared across images and query positions. For a fixed head $h ,$ the value vectors are collected into $V \in \mathbb { R } ^ { n _ { y } \times n _ { x } \times D ^ { h } }$ . For each atom r, the axis attention matrices defined entry-wise by equation 3 are $A _ { r , y } \in \mathbb R ^ { n _ { y } \times n _ { y } }$ and $A _ { r , x } \in \mathbb { R } ^ { n _ { x } \times n _ { x } }$ . The head output is:

$$
U _ { r } = A _ { r , y } V ,\tag{7}
$$

$$
Z = \sum _ { r = 1 } ^ { R } \lambda _ { r } U _ { r } A _ { r , x } ^ { \top } .\tag{8}
$$

This separable formulation reduces the asymptotic cost of spatial mixing from quadratic to subquadratic in the number of tokens: for a square n × n token grid with $N - \bar { 1 } = n ^ { 2 }$ and a small, fixed number of atoms R, applying R separable Gaussian atoms through successive horizontal and vertical contractions costs $\mathcal { O } ( \bar { R } \bar { N } ^ { 3 / 2 } D )$ , compared with $\mathcal { O } ( N ^ { 2 } D )$ for dense self-attention; Appendix A.1 gives the full derivation, including storage costs and the effect of query-dependent steering.

## 2.2 CONTENT-STEERED SSOG

Fixed SSoG applies the same learned spatial geometry independently of image content. To introduce content dependence without reintroducing pairwise query-key interactions, we additionally consider the LookAt variant. For each query position $q ,$ with token $\boldsymbol { x _ { q } } ^ { - } \in \mathbb { R } ^ { D }$ (the q-th row of X), the model predicts bounded modifications to the Gaussian centers, spreads, and mixture weights. Three linear predictors generate the following query-dependent perturbations for each head h independently:

$$
\mu _ { r , a } ( x _ { q } ) = \mu _ { r , a } + \alpha _ { \mu } M \operatorname { t a n h } \bigl ( [ W _ { \mu } x _ { q } + b _ { \mu } ] _ { r , a } \bigr ) ,\tag{9}
$$

$$
\sigma _ { r , a } ( x _ { q } ) = \sigma _ { r , a } \exp \Bigl [ \alpha _ { \sigma } \operatorname { t a n h } \bigl ( [ W _ { \sigma } x _ { q } + b _ { \sigma } ] _ { r , a } \bigr ) \Bigr ] ,\tag{10}
$$

$$
\lambda _ { r } ( x _ { q } ) = \operatorname { s o f t m a x } _ { r } \left[ \ell _ { r } + \alpha _ { \lambda } \operatorname { t a n h } \left( [ W _ { \lambda } x _ { q } + b _ { \lambda } ] _ { r } \right) \right] .\tag{11}
$$

where $W _ { z }$ and $b _ { z }$ are learned weights and biases, $[ v ] _ { r , a }$ denotes the $( r , a )$ -th entry of v (indexed over atoms and axes), M is the steering radius, so that each center moves by at most $\alpha _ { \mu } M$ grid cells, and

$$
\alpha _ { z } = \mathrm { s o f t p l u s } ( \tilde { \alpha } _ { z } ) + \varepsilon , \qquad z \in \{ \mu , \sigma , \lambda \} .\tag{12}
$$

The tanh nonlinearities bound the predicted modifications, while the exponential parameterization ensures that the steered spreads remain positive. Query-dependent steering does not remove the separable factorization; the computation continues to use successive one-dimensional contractions. Individual steering paths can be disabled independently for ablation.

The scalar steering parameters are initialized close to zero; the specific values used in each training regime are given in §3. The model therefore starts close to the fixed SSoG geometry and progressively activates content-dependent modulation during training. This initialization is particularly useful in the continuous-depth setting, where the operator is applied repeatedly.

## 2.3 INSTANTIATING SSOG ACROSS ARCHITECTURES

SSoG is designed as a drop-in replacement for MHA: in every architecture we study, the only change relative to the corresponding baseline is replacing its attention block with SSoG, while the surrounding architecture is left unchanged. We instantiate this substitution in three settings: a standard discrete (non-recurrent) ViT, a looped/recurrent Transformer based on Byra et al. (2026) that shares its attention block across depth, and a continuous-depth architecture in which SSoG defines the vector field of a Neural ODE Boned Riera et al. (2025).

All three settings share the same tokenization. Let an input image be $I \in \mathbb { R } ^ { B \times C \times H _ { 0 } \times W _ { 0 } }$ , where B is the batch size. A convolution with kernel size and stride $P$ produces a grid of $n _ { y } \times n _ { x }$ patch tokens, where $n _ { y } = H _ { 0 } / P$ and $n _ { x } = W _ { 0 } / P$ We denote the total sequence length by $N = n _ { y } n _ { x } + 1$ and the embedding dimension by D. With a class token prepended, the initial sequence is

$$
X _ { 0 } = \mathrm { P a t c h E m b e d } ( I ) + E _ { \mathrm { p o s } }\tag{13}
$$

Class-token handling. The class token has no natural two-dimensional coordinate and is therefore excluded from Gaussian spatial filtering. Instead, it receives a lightweight global aggregation of the projected patch values $v _ { p } \in \mathbb { R } ^ { D ^ { h } } , p = 1 , \dots , N - 1$

$$
v _ { \mathrm { c l s } } = \frac { 1 } { N - 1 } \sum _ { p = 1 } ^ { N - 1 } v _ { p } .\tag{14}
$$

This aggregated class-token representation is concatenated with the filtered patch representations and projected through $W _ { O } \in \mathbf { \bar { \mathbb { R } } } ^ { D \times D }$ , producing the SSoG block output. Patch-to-patch mixing therefore remains entirely Gaussian, while no dense class-to-patch attention scores are introduced. Here $v _ { \mathrm { c l s } }$ denotes the per-block value-space aggregate of the class token; in all three settings below, the class-token state accumulated across blocks is denoted $z _ { \mathrm { c l s } }$ .

Discrete ViT and looped Transformer. In the standard discrete setting, L SSoG blocks are stacked, each with its own independent parameters $\theta _ { 1 } , \ldots , \theta _ { L }$ . In the looped setting, a single block is used instead, with parameters θ shared across all L iterations (L is the depth); both follow the recurrence

$$
X _ { l } = X _ { l - 1 } + \mathrm { S S o G } ( X _ { l - 1 } | \theta _ { l } ) , \qquad l = 1 , \ldots , L ,\tag{15}
$$

with $\theta _ { l }$ independent across l in the discrete case, and $\theta _ { l } = \theta$ for all l in the looped case.

ODE-SSoG (continuous-depth). Rather than applying ${ \mathrm { S S o G } } _ { \theta }$ a fixed number of times, ODE-SSoG treats depth as continuous: SSoG serves as the token-mixing operator inside the vector field $F ( \cdot , \cdot | \theta )$ of the Continuous ViT, and the token sequence is evolved by the initial-value problem

$$
\frac { d X ( t ) } { d t } = F ( t , X ( t ) | \theta ) , \qquad X ( 0 ) = X _ { 0 } , \qquad t \in [ 0 , 1 ] ,\tag{16}
$$

integrated numerically from $t = 0$ to $t = 1$ . As in the looped setting, the same parameters θ are reused at every function evaluation; ODE-SSoG can be seen as the continuum limit of the looped variant, where the number of iterations define the horizon in the integration path.

In all three settings, the final class-token state ${ z _ { \mathrm { c l s } } }$ , obtained after L iterations of equation 15 or by integrating equation 16 to t = 1, is normalized and linearly projected to the class logits:

$$
\widehat { \boldsymbol { y } } = { \cal W } _ { \mathrm { c l s } } \mathrm { L N } ( \boldsymbol { z } _ { \mathrm { c l s } } ) + \boldsymbol { b } _ { \mathrm { c l s } } .\tag{17}
$$

## 3 EXPERIMENTS AND RESULTS

In this section, we study different questions, each addressed in its own subsection. The Appendix A.3 reports further supporting ablations that isolate individual design choices without bearing directly on these questions. Unless otherwise stated, classification ablations are conducted on CIFAR-100 for 500 epochs, following the same training recipe as Touvron et al. (2021) and patch size 16. For the continuous models, explicit Euler integration is used; NFE denotes the number of vector field evaluations.

![](images/d577105dd3921d7addbd7fea4fbe5751af865e64fed08507af07955e32db4dac.jpg)

Figure 2: Width scaling on CIFAR-100, dense attention (MHA) versus SSoG, across the three architecture settings. The dotted connector and label give the Acc@1 gain of SSoG at $D = 7 6 8$  
![](images/3ea9c0f95de1a80b9d1bb29489468f180c47e9e2a011ebe45458ab8e787a8328.jpg)  
Figure 3: Ablations of SSoG on CIFAR-100 in the three architecture settings.

## 3.1 IS SSOG SENSITIVE TO DIFFERENT HYPERPARAMETERS?

Figure 2 summarizes the main comparison among models and capacity levels. In all cases the addition of the SSoG improves the results in all the families of models and in the capacity of each model within the family by at least 2.22 Top-1 points (full results in Table 10).

We report ablations over head count, depth, and atom count. Figure 3 shows the behavior of SSoG under different hyperparameters, with the underlying results reported in Appendix A.3 Table 11. The table shows that, at a fixed embedding dimension of 768 and four atoms per head, increasing the number of heads from 6 to 24 improves top-1 accuracy slightly in the Continuous model while there is almost no effect in the looped and discrete setting. The atom-count sweep saturates rapidly: moving from one to two atoms improves top-1 accuracy from 73.42% to 74.13%, whereas eight atoms reach only 73.99%. To assess whether this saturation reflects redundancy among the learned atoms, we computed the entropy of their mixture weights: on CIFAR-100, the eight-atom model has an entropy-derived effective atom count of only 3.381. Appendix A.4 gives the formulation of the entropy used to compute the effective atoms, and Appendix A.3.1 additionally reports the effect of the sigma floor $\sigma _ { \mathrm { m i n } }$ , the remaining capacity constraint on the Gaussian atoms. Two to four atoms per head therefore appear sufficient, while additional atoms are redundant.

We ran the same ablation without SSoG, using dense attention instead; the results are reported in Appendix A.3 Table 12, where the variation across hyperparameters is larger than for SSoG.

## 3.2 STEERING: WHAT SHOULD BE STEERED?

We isolated query-dependent changes to the atom center $\mu ,$ width $\sigma ,$ and mixture weight λ in Table 1. We used full cold steering initialization with 12 NFE as baseline. More runs can be found in the Appendix A.3 Table 7. We focus on the ODE as it yielded the worst results. Width-only steering matches the fixed model in accuracy (74.01% versus 74.22%). Mixture-only steering is also competitive at 73.87%. In contrast, jointly steering centers and widths reduces accuracy to 71.21%, close to full steering at 70.89%. Since center-only and width-only models reach 73.33% and 74.01%, respectively, we conjecture that the main failure mode is their joint query-dependent geometry from the division in equation 2. Appendix A.3.3 reports an ablation showing that a warmer initialization or a larger initialization of the gate $\tilde { \alpha } _ { \mu }$ both cost additional accuracy on top of this interaction, and Appendix A.3.4 shows that constraining the steering radius partially, but not fully, recovers it.Appendix A.3 Table 7 reports results of steering for the other families, where there is an improvement over the fixed SSoG.

Table 1: Steering-component attribution on CIFAR-100 at NFE 12 and R = 4, cold initialization throughout. Checkmarks denote query-dependent components.
<table><tr><td>Variant</td><td>µ</td><td>σ</td><td>λ</td><td>Acc@1</td><td>Acc@3</td><td>Acc@5</td><td>Loss</td><td>Time (h)</td></tr><tr><td>Fixed</td><td>一</td><td>=</td><td>一</td><td>74.22</td><td>88.59</td><td>92.38</td><td>1.12</td><td>47.77</td></tr><tr><td>Full, cold</td><td>√</td><td>√</td><td>V</td><td>70.89</td><td>86.89</td><td>91.30</td><td>1.20</td><td>59.48</td></tr><tr><td>Center only</td><td>V</td><td>-</td><td>1</td><td>73.33</td><td>87.79</td><td>91.61</td><td>1.15</td><td>58.37</td></tr><tr><td>Width only</td><td>–</td><td>√</td><td>一</td><td>74.01</td><td>88.39</td><td>92.16</td><td>1.11</td><td>57.42</td></tr><tr><td>Mixture only</td><td>一</td><td>一</td><td>√</td><td>73.87</td><td>88.20</td><td>91.73</td><td>1.13</td><td>57.08</td></tr><tr><td>Center + width</td><td>√</td><td>√</td><td>–</td><td>71.21</td><td>86.83</td><td>91.13</td><td>1.19</td><td>58.61</td></tr></table>

## 3.3 DOES SSOG OUTPERFORM OPTIMIZED SDPA KERNELS AS BATCH SIZE AND RESOLUTION GROW?

We benchmark randomly initialized models in FP16 on an NVIDIA A40 using ten warm-ups and 50 measured iterations. Figure 4 reports a stress test of measured throughput, FLOPs and peak allocated CUDA memory at different resolutions averaged over batch sizes from 8 to 512. Full throughput, memory and FLOPs results are given in Appendix A.1 (Figure 5, Table 4).

On the one hand, in the continuous-depth setting, the throughput of SSoG already exceeds ODE-ViT at every measured resolution, growing from +11.0% at 224 × 224 to +154.5% at 1024 × 1024. The discrete setting instead starts below parity: SSoG trails ViT-B/16 by up to 22.6% at resolution 224, reaches approximate parity around resolution 512 to 640, and only clearly overtakes it from resolution 640 onward, reaching +30.1% at 1024. The looped setting sits between the two, with a positive throughput delta at every resolution, from a marginal +2.9% at 224 to +33.1% at 1024.

On the other hand, on memory, SSoG uses less memory than ODE-ViT, saving between 27% and over 60%, and less memory than the standard ViT, saving between approximately 6% and 10%, whereas the looped setting is the only one where SSoG costs more than dense attention. A plausible explanation is that the discrete and looped baselines use a highly optimized fused attention kernel with strong absolute memory efficiency, while the separable-Gaussian implementation used by SSoG relies on standard PyTorch ops; at high resolution, this constant-factor gap can outweigh the structural advantage of SSoG, an effect that appears earliest for the looped setting, whose dense baseline also carries less fixed parameter memory due to weight sharing. Appendix A.1.1 and Table 3 report the computation cost of the attention operation in isolation.

An independent sweep over Tiny, Small, Base and Large ViTs (Figure 1, left) shows that SSoG is faster and lighter than MHA at every width.

## 3.4 DOES SSOG GENERALIZE BEYOND SCALING?

Finally, Table 2 reports how the operator scales to a larger benchmark, ImageNet-1k in this case. Each model family was trained under the same experimental setup on the same hardware. SSoG matches or exceeds its dense counterpart in the continuous-depth and discrete settings, by a narrow +0.30-point margin for continuous-depth and a clearer +1.42-point margin for discrete; the looped setting is the only exception, where dense leads SSoG by 0.35 points on top-1 accuracy. The accuracy of SSoG is competitive with dense attention across all three settings rather than specific to any one of them. Appendix A.5 additionally reports results on linear probing and full fine-tuning, as well as the effect of the data augmentation in the transfer setting.

![](images/45e6e111b497cf8431a244fb0e70da6c44750e9e78db722f86a2b9ad1aacc11e.jpg)

![](images/c602339c72d61108972e9d3960d33315e739641982ff2d567695877269666d1e.jpg)

![](images/93232576bf130d5e5f58f4481a59cf495c9bb1fe3e0f54a6001ce9b92b747c9a.jpg)

![](images/b9295be896e6d6371bac961fa8ca5415d00aa032ca426095729ea6a9854c4c18.jpg)  
Figure 4: SSoG versus dense attention, throughput and memory improvement (%) by resolution, for each of the three architecture settings (continuous-depth ODE-ViT, discrete ViT-B/16, looped bViT). Each point is the mean relative difference over batch sizes 8 → 512 at that resolution. Memory is negated (plotted as % memory saved) so both lines share one direction: above the dashed zero line means SSoG wins that metric at that resolution, below means dense wins. The rightmost panel shows the relative improvement in FLOPs for the whole model and for the attention operation alone.

Table 2: ImageNet-1k accuracy at $D = 7 6 8 \colon$ SSoG versus dense attention (MHA).
<table><tr><td rowspan="2">Metric</td><td colspan="2">Continuous</td><td colspan="2">Discrete</td><td colspan="2">Looped</td></tr><tr><td>MHA</td><td>SSoG</td><td>MHA</td><td>SSoG</td><td>MHA</td><td>SSoG</td></tr><tr><td>Acc@1</td><td>63.04</td><td>64.02</td><td>76.47</td><td>77.89</td><td>73.72</td><td>73.37</td></tr><tr><td>Acc@3</td><td>79.04</td><td>79.62</td><td>89.44</td><td>90.59</td><td>88.35</td><td>87.93</td></tr><tr><td>Acc@5</td><td>83.40</td><td>84.45</td><td>92.31</td><td>93.37</td><td>91.90</td><td>91.65</td></tr></table>

## 4 RELATED WORK & DISCUSSION

Approximating attention versus replacing it. Most work on subquadratic attention approximates the softmax kernel instead of abandoning it, using low-rank projections, random features or kernel feature maps Wang et al. (2020); Choromanski et al. (2021); Katharopoulos et al. (2020). What ties these together is the idea that attention acts as a kernel smoother Tsai et al. (2019), which vision models have since adopted in several forms Han et al. (2023); Cai et al. (2023). SSoG goes the other way: it fixes the form of the kernel in advance and learns only the geometry, so the saving comes from structure and not from an approximation. Computationally the closest relatives are separable large-kernel convolutions Guo et al. (2023); Lau et al. (2024), which split a two-dimensional depthwise kernel into cascaded one-dimensional ones for the same reason we do. What we do differently is constrain each axis kernel to a mixture of R Gaussians instead of learning a free weight at every tap, and that constraint turns out to cost little, since accuracy can be saturated at few atoms per head: for instance, the continuous model with eight atoms uses an effective number of only 3.381.

Gaussians as a bias, and Gaussians as the operator. Gaussians appear often as a spatial prior for attention. The attention maps of a trained ViT tend towards a Gaussian shape Kim et al. (2023), and several models add a learned Gaussian term to the attention logits Hajimiri et al. (2026), which keeps the N × N matrix and makes the Gaussian a bias on top of attention rather than the operator itself. Closer to how we parameterize things, Kundu et al. (2026) drop the query and key projections in favour of an RBF kernel with one bandwidth per head but still compare every pair of tokens, and concurrently Huang & Raza (2026) route tokens through Gaussian mixture components in a latent feature space for language. The operator we study is that of Pisoni (2026), who proposed SSoG for standard ViTs; our contribution lies in its theoretical analysis and in its use as a weight-shared and continuous-depth token mixer. We place the Gaussians over the grid of tokens instead, which is what makes the factorization along axes possible, but it also limits how much content dependence the operator can take on. In the continuous-depth model, steering the width alone matches the fixed model (74.01% against 74.22%) and steering the centers alone costs 0.77 points, while steering both drops to 71.21%; equation 2 suggests why: the center and the spread appear together as the ratio $( \delta _ { a } ^ { - } - \mu _ { a } ) ^ { 2 } / 2 \sigma _ { a } ^ { 2 }$ , so when both depend on the query the numerator and the denominator can drift at the same time. We therefore recommend the fixed operator or a limited steering for this model, whereas full steering improves the discrete and looped models (Table 7).

Lipschitz continuity, weight sharing and continuous depth. Kim et al. (2021) showed that dotproduct attention is not globally Lipschitz, and the work that followed recovers the property by constraining the operator after the fact, either by substituting Lipschitz counterparts for the unstable components Qi et al. (2023) or by normalizing the layer and bounding its spectral norm Dasoulas et al. (2021); Yudin et al. (2025). In fixed SSoG it comes out of the parameterization itself, because the mixing weights depend only on bounded geometric parameters and not on inner products between unconstrained token vectors, so nothing has to be added to the loss to obtain it (Proposition 1); the content-steered variant retains this property only on bounded sets (Proposition 2). This matters most when one block is applied many times, which covers both depth recurrence, from the Universal Transformer Dehghani et al. (2019) and ALBERT Lan et al. (2020) to the fully shared single-block ViT of Byra et al. (2026), and continuous depth, from Chen et al. (2018) to recent im age classifiers Zhong et al. (2022); Sander et al. (2022); Boned Riera et al. (2025); Belhasin et al. (2026). This second line has mostly used convolutional vector fields, which, unlike attention, are globally Lipschitz and thus satisfy the Picard–Lindelof theorem on the whole interval.¨

We hypothesize that Lipschitz continuity contributes to the accuracy gains: our three settings differ in how much the property should matter, and the results are consistent with that order. The looped model applies an unconstrained dense block repeatedly and gains the most, 7.02 points on average and 79.80% against 72.30% at D = 768. The discrete model never repeats a block and gains 3.66 points. ODE-ViT also repeats one block, but it already relaxes the requirement to a local bound, since the JaSMin loss of Boned Riera et al. (2025) penalizes the eigenvalue distribution of the attention map and constrains the spectral radius, which may explain why its gain of 3.32 points is closer to that of the discrete setting than to that of the looped one. The regularizer also has a computational cost. Regularizing the spectrum of the attention map requires materializing that matrix during training, which rules out a fused kernel and keeps the full $N \times N$ map in the computational graph at every function evaluation, so the memory and latency overhead grows with the NFE. What ODE-ViT obtains with a regularizer whose cost scales with the matrix it regularizes, fixed SSoG obtains from the form of the operator at no additional cost. Our efficiency benchmarks (§3.3) are measured at inference, where the regularizer is not evaluated; there, the advantage over ODE-ViT reflects the asymptotic cost of the operator together with the fact that the ODE-ViT baseline does not use a fused attention kernel (Appendix A.1.1).

Limitations. The margins narrow on ImageNet, where SSoG is on par with its counterparts while requiring less compute and achieving higher throughput. Part of this is expected, since a fixed spatial prior is an inductive bias and inductive biases are worth less when data is abundant. It also bounds what the Lipschitz argument can claim, because a benefit that shrinks with dataset size behaves like regularization rather than like well-posedness. We compare only against dense attention, while linear-attention variants and state space models Gu & Dao (2024); Zhu et al. (2024) scale subquadratically as well and are the natural competitors in the regime where SSoG wins. SSoG is also $\mathcal { O } \bar { ( N ^ { 3 / 2 } ) }$ rather than linear (Appendix A.1), so a state space model keeps an asymptotic advantage, whereas SSoG drops into an existing model without changing the rest of the architecture or the training recipe. Finally, we evaluate classification only, leaving open other tasks that could benefit from this operator.

## 5 CONCLUSIONS

We presented the first systematic study of SSoG, a separable sum-of-Gaussians alternative to dense self-attention recently outlined by Pisoni (2026), as the token mixer of discrete, looped and continuous-depth Transformers. We proved that its fixed form is globally Lipschitz continuous, which guarantees a unique solution of the continuous-depth model on the whole integration interval, whereas the content-steered form is Lipschitz only on bounded sets; in both forms the cost is subquadratic in the number of tokens. On CIFAR-100, SSoG outperformed dense attention in all three settings with fewer parameters, most markedly in the looped one; on ImageNet-1k it matched or exceeded dense attention in the discrete and continuous-depth settings and trailed it slightly in the looped one. Its throughput advantage widened with resolution, and it used less memory than dense attention in the continuous-depth and discrete settings but not in the looped one, where the dense baseline benefits from a fused kernel. Future work includes a fused implementation of SSoG, tasks beyond classification, and a comparison with linear attention and state space models.

## 6 ACKNOWLEDGMENTS

This work has been partially supported by the Spanish project PID2024-157778OB-I00, Ministerio de Ciencia e Innovacion. Carlos Boned Riera is funded with the support of the FI-STEP predoctoral´ grants program of the Department of Research and Universities of the Generalitat of Catalonia and co-financing by the European Social Fund Plus (2025 STEP 00066)

## AI USE STATEMENT

In this work, we used generative AI tools, namely an LLM-based coding and writing assistant (Claude, Anthropic), to aid and polish writing, to support research ideation and execution, and to assist with mathematical claims, as detailed below.

Writing. The assistant polished the grammar and concision of author-written paragraphs, turned results and specifications provided by the authors into LAT X tables, figure captions and descriptive paragraphs. The authors revised all such text.

Research ideation and execution. The LLM wrote and debugged experiment code: training and evaluation configurations and cluster launch scripts for the ablation and transfer experiments; the throughput and peak-memory benchmark harness; the operator-level attention memory benchmark; the patch-size robustness evaluation; scripts that collect metrics from experiment logs into tables; and the plotting scripts for the figures. While auditing benchmark results, it identified a cross-model measurement interference that led us to re-run every efficiency benchmark with each model in it own process, and all reported efficiency numbers come from these isolated runs. It also proposed an analysis that we adopted: isolating the memory scaling of the attention operator from the rest of the network. The research questions, the method, the choice of experiments to report, the runs and the interpretation of results are the authors’.

Mathematical claims. We used AI to help validate the derivations of the Lipschitz continuity results, providing intuition and suggesting directions that supported the final lemmas and propositions. The LLM also reviewed the notation and its consistency throughout the paper.

Verification. We have reviewed all AI-assisted work. AI-generated code was inspected and tested before any result was reported; for example, models re-loaded by the evaluation code were checked to reproduce their recorded validation accuracy at the training patch size, parameter counts were checked against the saved checkpoints, and efficiency benchmarks were cross-checked by re-running them in isolation. We take responsibility for the final content of this work, including text, claims o artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

We release the complete source code as anonymized supplementary material. Every experiment is specified by a modular, hierarchical Hydra configuration: a base file per dataset and architecture setting (continuous, discrete and looped; SSoG and dense-attention baselines) fixes the full training recipe, and each experiment is a short file that inherits from it and overrides only the varied quantity (width, heads, atoms, depth or NFE, or augmentation), so every reported run corresponds to exactly one versioned configuration file. Each model family has its own entry point on a shared PyTorch Lightning training pipeline, and we provide the cluster launch scripts used for every run. Training is seeded through Lightning’s seed everything (including data-loader workers) and run with deterministic=True; the default seed is 42. Architectures and the training recipes are described in section 2 and section 3, and the per-ablation settings in Appendix A.3–A.5. Efficiency numbers come from a separate benchmark configuration and script, run on an NVIDIA A40 under torch.inference mode(), with each model benchmarked in its own process (section 3.3 and Appendix A.1.1). We use only the public CIFAR-100 and ImageNet-1k datasets with their standard splits; all preprocessing and augmentation is defined in the released configurations and follows the DeiT recipe Touvron et al. (2021). The table-generation and plotting scripts that produce every table and figure from the logged metrics are also included.

## REFERENCES

Omer Belhasin, Shelly Golan, Ran El-Yaniv, and Michael Elad. Advancing image classification with discrete diffusion classification modeling. In Conference on Computer Vision and Pattern Recognition 2026, 2026. URL https://openreview.net/forum?id=qxQcVusjI2.

Carlos Boned Riera, David Romero Sanchez, and Oriol Ramos Terrades. ODE-ViT: Plug & play attention layer from the generalization of the vit as an ordinary differential equation, 2025. URL https://arxiv.org/abs/2511.16501.

Michal Byra, Pawel Olszowiec, Grzegorz Stefanski, Grzegorz Gruszczynski, and Alberto Presta. bvit: Investigating single-block recurrence in vision transformers for image recognition, 2026. URL https://arxiv.org/abs/2605.10661.

Han Cai, Junyan Li, Muyan Hu, Chuang Gan, and Song Han. Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 17302–17313, 2023.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou, Julien Mairal, Piotr Bojanowski, and´ Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9650–9660, October 2021.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In (NeurIPS), 2018. URL https://api.semanticscholar.org/ CorpusID:49310446.

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. Rethinking Attention with Performers. In International Conference on Learning Representations, 2021. URL https: //mlanthology.org/iclr/2021/choromanski2021iclr-rethinking/.

Earl A. Coddington and Norman Levinson. Theory of Ordinary Differential Equations. McGraw-Hill, New York, 1955.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need´ registers. In (ICLR), 2024. URL https://openreview.net/forum?id=2dnO3LLiJ1.

George Dasoulas, Kevin Scaman, and Aladin Virmaux. Lipschitz normalization for self-attention layers with application to graph neural networks. In International Conference on Machine Learning, volume 139, pp. 2456–2466, 2021. URL https://proceedings.mlr.press/ v139/dasoulas21a.html.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019. URL https: //openreview.net/forum?id=HyzdRiR9Y7.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In (ICLR), 2021. URL https://openreview.net/forum?id=YicbFdNTTy.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling (COLM), 2024. URL https://openreview.net/ forum?id=tEYskw1VY2.

Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, and Shi-Min Hu. Visual attention network. Computational Visual Media, 9(4):733–752, 2023. doi: 10.1007/ s41095-023-0364-2.

Sina Hajimiri, Farzad Beizaee, Fereshteh Shakeri, Christian Desrosiers, Ismail Ben Ayed, and Jose Dolz. Locality-Attending Vision Transformer. In International Conference on Learning Representations, 2026. URL https://mlanthology.org/iclr/2026/ hajimiri2026iclr-localityattending/.

Dongchen Han, Xuran Pan, Yizeng Han, Shiji Song, and Gao Huang. Flatten transformer: Vision transformer using focused linear attention. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Yongchao Huang and Hassan Raza. Gaussian mixture attention: Linear-time sequence mixing via probabilistic latent routing, 2026. URL https://arxiv.org/abs/2606.18283.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, volume 119, pp. 5156–5165, 2020. URL https://proceedings.mlr. press/v119/katharopoulos20a.html.

Bum Jun Kim, Hyeyeon Choi, Hyeonah Jang, and Sang Woo Kim. Understanding gaussian attention bias of vision transformers using effective receptive fields, 2023. URL https://arxiv.org/ abs/2305.04722.

Hyunjik Kim, George Papamakarios, and Andriy Mnih. The lipschitz constant of self-attention. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 5562–5571. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/kim21i. html.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick.´ Segment anything, 2023. URL https://arxiv.org/abs/2304.02643.

Debarshi Kundu, Archisman Ghosh, Swaroop Ghosh, and Vasant Honavar. Projection-free transformers via gaussian kernel attention, 2026. URL https://arxiv.org/abs/2605. 02144.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In International Conference on Learning Representations, 2020. URL https://openreview.net/ forum?id=H1eA7AEtvS.

Kin Wai Lau, Lai-Man Po, and Yasar Abbas Ur Rehman. Large separable kernel attention: Rethinking the large kernel attention design in cnn. Expert Systems with Applications, 236:121352, 2024. ISSN 0957-4174. doi: https://doi.org/10.1016/j.eswa.2023.121352. URL https: //www.sciencedirect.com/science/article/pii/S0957417423018547.

Xin Liu, Runsong Zhao, Pengcheng Huang, Xinyu Liu, Junyi Xiao, Chunyang Xiao, Tong Xiao, Shengxiang Gao, Zhengtao Yu, and Jingbo Zhu. Autoencoding-free context compression for LLMs via contextual semantic anchors. In The Fourteenth International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/forum?id= 8Pi6Du0n7F.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 4195–4205, October 2023.

Raphael Pisoni. A few gaussians is all you need: Ssog-attention that steers instead of scores, aug 2026. URL https://pisoni.ai/posts/ssog/. Code available at: https: //github.com/4rtemi5/ssog.

Xianbiao Qi, Jianan Wang, Yihao Chen, Yukai Shi, and Lei Zhang. Lipsformer: Introducing lipschitz continuity to vision transformers. In (ICLR), 2023. URL https://openreview.net/ forum?id=cHf1DcCwcH3.

Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated Attention for Large Language Models: Non-Linearity, Sparsity, and Attention-Sink-Free. In Advances in Neural Information Processing Systems, 2025. URL https://mlanthology.org/neurips/ 2025/qiu2025neurips-gated/.

Michael Sander, Pierre Ablin, and Gabriel Peyre. Do residual neural networks dis-´ cretize neural ordinary differential equations? In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 36520–36532. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-2646. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/ecc38927fe5148c66bee64ee8fed1e76-Paper-Conference.pdf.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J´ egou. Training data-efficient image transformers & amp; distillation through attention.´ In Marina Meila and Tong Zhang (eds.), (ICML), volume 139 of Proceedings ofMachine Learning Research, pp. 10347–10357. PMLR, 18–24 Jul 2021. URL https://proceedings. mlr.press/v139/touvron21a.html.

Yao-Hung Hubert Tsai, Shaojie Bai, Makoto Yamada, Louis-Philippe Morency, and Ruslan Salakhutdinov. Transformer dissection: An unified understanding for transformer’s attention via the lens of kernel. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4344–4353, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1443. URL https://aclanthology.org/D19-1443/.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity, 2020. URL https://arxiv.org/abs/2006.04768.

Nikolay Yudin, Alexander Gaponov, Sergei Kudriashov, and Maxim Rakhuba. Pay attention to attention distribution: A new local lipschitz bound for transformers, 2025. URL https:// arxiv.org/abs/2507.07814.

Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. Long context compression with activation beacon. In The Thirteenth International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id= 1eQT9OzfNQ.

Yaofeng Desmond Zhong, Tongtao Zhang, Amit Chakraborty, and Biswadip Dey. A neural ODE interpretation of transformer layers. In The Symbiosis ofDeep Learning and Differential Equations II, 2022. URL https://openreview.net/forum?id=nA9hvYMQCy.

Lianghui Zhu, Bencheng Liao, Qian Zhang, Xinlong Wang, Wenyu Liu, and Xinggang Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 62429–62442. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/zhu24f.html.

## A APPENDIX

## A.1 NUMERICAL INTEGRATION AND COMPUTATIONAL COST

All costs below are stated per sample; multiply by batch size B to obtain the total cost over a mini-batch. We integrate equation 16 over [0, 1] using torchdiffeq, with a uniform grid of S evaluation points and the explicit Euler solver:

$$
X _ { k + 1 } = X _ { k } + \Delta t F ( t _ { k } , X _ { k } | \theta ) , \qquad \Delta t = { \frac { 1 } { S - 1 } } .\tag{18}
$$

Euler performs $S - 1$ function evaluations, so 13 grid points correspond to NFE 12.

Cost of one fixed SSoG evaluation. We count multiply-accumulate operations, reading the cost of each contraction off its index set as the product of the ranges of all free and all summed indices. The class token adds one token to the projections and is excluded from spatial filtering; since N already counts it (§2.3), all costs below are written in terms of $N$

The value projection of equation 1 costs $\mathcal { O } ( N D ^ { 2 } )$ , and is the only input projection, since SSoG forms no queries and no keys where dense attention needs three. In the vertical contraction $U _ { r } =$ $A _ { r , y } V$ of equation 8, each of the H heads and R atoms multiplies an $n _ { y } \times n _ { y }$ matrix by an $n _ { y } \times$ $( n _ { x } D ^ { h } )$ matrix, giving

$$
H R n _ { y } ^ { 2 } n _ { x } D ^ { h } = \mathcal { O } \big ( D R n _ { y } ^ { 2 } n _ { x } \big )\tag{19}
$$

after recombining $H D ^ { h } = D$ . In the horizontal contraction $\textstyle \sum _ { r } \lambda _ { r } ^ { h } U _ { r } A _ { r , x } ^ { \top } .$ , the weights ${ \lambda } _ { r } ^ { h }$ are applied as the atom index is summed, and each head multiplies, for every atom, an $( n _ { y } D ^ { h } ) \times n _ { x }$ matrix by an $n _ { x } \times n _ { x }$ matrix, giving

$$
H R n _ { y } n _ { x } ^ { 2 } D ^ { h } = { \mathcal O } \big ( D R n _ { y } n _ { x } ^ { 2 } \big ) .\tag{20}
$$

The class-token average of equation 14 costs $\mathcal { O } ( N D )$ and the output projection $\mathcal { O } ( N D ^ { 2 } )$ . Adding the two contractions and using $n _ { y } n _ { x } = N - 1$ , spatial mixing costs

$$
\mathcal O \big ( D R \left( n _ { y } ^ { 2 } n _ { x } + n _ { y } n _ { x } ^ { 2 } \right) \big ) = \mathcal O ( D R N ( n _ { y } + n _ { x } ) ) ,\tag{21}
$$

which for a square grid with $n _ { y } = n _ { x } = n$ and $N - 1 = n ^ { 2 }$ becomes

$$
\mathcal { O } ( D R n ^ { 3 } ) = \mathcal { O } \Big ( D R N ^ { 3 / 2 } \Big ) .\tag{22}
$$

The exponent $3 / 2$ rather than 1 is the price of separability on a two-dimensional grid, since one axis of length $\sqrt { N }$ remains inside each contraction.

Spatial mixing overtakes the projections when $D R N ^ { 3 / 2 } > N D ^ { 2 }$ , that is when $R \sqrt { N } > D$ , while for dense attention the token interaction overtakes them when $D N ^ { 2 } > N D ^ { 2 }$ , that is when $N > D$ $\mathrm { A t } \ D \ = \ 7 6 8$ with $R = 4$ and patch size $^ { 1 6 , }$ dense attention becomes interaction-bound above $N = 7 6 8$ tokens, an input resolution of about 443, whereas SSoG would only become mixing-bound beyond a resolution of 3000. Across the whole range we benchmark, SSoG stays projection-bound while dense attention crosses into the quadratic regime part way through, which is the mechanism behind the crossover reported in $\ S 3 . 3 .$

Storage. Dense self-attention stores an attention matrix of size $\mathcal { O } ( H N ^ { 2 } )$ , and SSoG stores none. Its axis kernels are shared across the batch and cost $\mathcal { O } ( H R ( n _ { y } ^ { 2 } + n _ { x } ^ { 2 } ) )$ independently of B, while the atom-wise intermediates U require O(DRN). The latter is linear in N and explains the memory behaviour of the looped setting in §3.3, where the dense baseline already carries little parameter memory.

Cost of content-steered SSoG. Steering still avoids the $N \times N$ query-key matrix, but its axis kernels depend on the batch index and on the full query position rather than being shared. Building the query-dependent kernel tables costs $\mathcal { O } ( N H R ( n _ { y } + n _ { x } ) )$ , and the three linear predictors of equation 9 cost $\mathcal { O } ( N D H R )$ . Both are dominated by the contraction itself, by a factor $D ^ { h }$ for the tables and $\sqrt { N } / H$ for the predictors, the latter holding at every resolution we test although only narrowly at 224. Steering therefore has the same leading-order complexity as fixed SSoG. What changes is arithmetic intensity: in the fixed variant each kernel entry is reused across the batch and along one spatial axis, while under steering each entry is used once.

Table 3: Peak allocated memory (MiB) of the attention operator alone, at matching token count.
<table><tr><td>Resolution</td><td>Tokens (N)</td><td>SSoG</td><td>Dense (fused MHA)</td><td>Ratio (×)</td></tr><tr><td>224</td><td>197</td><td>47.6</td><td>29.2</td><td>1.63</td></tr><tr><td>384</td><td>577</td><td>113.0</td><td>56.0</td><td>2.02</td></tr><tr><td>512</td><td>1025</td><td>184.5</td><td>88.3</td><td>2.09</td></tr><tr><td>640</td><td>1601</td><td>281.3</td><td>128.1</td><td>2.20</td></tr><tr><td>768</td><td>2305</td><td>395.8</td><td>181.4</td><td>2.18</td></tr><tr><td>1024</td><td>4097</td><td>689.0</td><td>304.3</td><td>2.26</td></tr></table>

Table 4: Forward FLOPs per image (GFLOPs, 2 per multiply-add) at dim768 with 12 block applications. The counts are identical, to within 0.3 GFLOPs, for the continuous (12 ODE steps), discrete (12 blocks) and looped (12 iterations) settings. Attention includes the projections; element-wise operations are not counted for either operator. Reduction is SSoG relative to dense SDPA attention (MHA).
<table><tr><td></td><td></td><td colspan="3">Whole model</td><td colspan="3">Attention</td></tr><tr><td>Resolution</td><td>Tokens</td><td>MHA</td><td>SSoG</td><td>Reduction (%)</td><td>MHA</td><td>SSoG</td><td>Reduction (%)</td></tr><tr><td>224</td><td>197</td><td>35.1</td><td>28.4</td><td>19.2</td><td>12.6</td><td>6.0</td><td>52.6</td></tr><tr><td>384</td><td>577</td><td>111.0</td><td>84.3</td><td>24.0</td><td>44.9</td><td>18.4</td><td>59.1</td></tr><tr><td>512</td><td>1025</td><td>214.1</td><td>151.1</td><td>29.4</td><td>96.8</td><td>33.9</td><td>65.0</td></tr><tr><td>640</td><td>1601</td><td>368.3</td><td>237.9</td><td>35.4</td><td>185.1</td><td>54.9</td><td>70.4</td></tr><tr><td>768</td><td>2305</td><td>590.1</td><td>345.3</td><td>41.5</td><td>326.4</td><td>81.7</td><td>75.0</td></tr><tr><td>1024</td><td>4097</td><td>1319.6</td><td>623.6</td><td>52.7</td><td>850.8</td><td>154.9</td><td>81.8</td></tr></table>

## A.1.1 ISOLATING THE MEMORY SCALING OF ATTENTION OPERATION

Figure 4 shows that the memory of SSoG advantage over dense attention behaves differently across settings: it grows with resolution in the continuous-depth setting but shrinks (discrete) or reverses (looped) as resolution increases. Since the dense baselines of the discrete and looped settings both use the fused SDPA from PyTorch, while the continuous-depth baseline (ODE-ViT) does not. We isolate the attention operator itself stripped of every other confound, including patch embedding, MLP, classifier head, and weight sharing to test whether this implementation difference alone explains the pattern.

We instantiate SSOGAttention (fixed, non-steered) and torch.nn.MultiheadAttention (matching $D = 7 6 8 , H = 1 2$ , the exact configuration used by the dense baselines) in isolation, feed both an identical $[ B , N , D ]$ input at token counts N matching each benchmarked resolution (batch 8, patch size 16), reset the peak memory counter, and measure only that forward call of the module.

Table 3 reports that the ratio grows monotonically with resolution: the attention module of SSoG alone uses 63% more memory than the fused dense kernel at $N { = } 1 9 7 .$ , rising to 126% more at $N { = } 4 0 9 7$ . Fitting a power law (memory $\propto \ d N _ { } ^ { k } )$ to each series gives $k \approx 0 . 8 8$ for SSoG and $k \approx 0 . 7 8$ for the fused dense kernel. We attribute this to an implementation gap rather than an inherent property of the separable formulation: the kernel construction and mixture application of SSoG are built from generic PyTorch broadcast and reshape operations, whereas SDPA dispatches to a fused, memory-optimized CUDA kernel.

Furthermore, in order to isolate the computation beyond the kernels and the hardware optimization Table 4 reports the numbers of the Flops among all settings.

## A.2 ATTENTION SINKS

Dense attention can develop attention sinks: tokens that receive a disproportionate share of attention from every query, largely independent of the input. Fixed SSoG has no mechanism to form them. Its patch-to-patch weights depend only on the relative offset between two patches, so every key receives the same total attention up to renormalization at the image border, and patches never attend to the class token, which only pools. Content steering can move a kernel, but by at most $\alpha _ { \mu } M$ tokens, so it cannot concentrate attention from the whole image onto one token.

We measure this on the trained models (Table 5). A sink requires three things at once: a key that receives much more than uniform attention (sink score), the same key across images (same key: the share of images whose most-attended key is the head’s most frequent one), and persistence across layers, which the residual attention rollout captures because it compounds only concentration that the heads and layers agree on.

Only the ImageNet dense model meets all three. In its worst layer the most-attended key receives $4 0 . 6 \times$ the uniform mass, it is the same token in 91% of images, and the rollout compounds it to $4 1 . 4 \times ; 8 . 1 \%$ of patch attention also flows to the class token. The CIFAR-100 dense models each fail at least one condition. The continuous one has strong single-head peaks (22.9× in the worst layer), but the attended key changes from image to image (same key in 6% of images) and rollout washes it out (1.7×): content-dependent focusing rather than a sink. The discrete one is intermediate (8.1×, same key in 49% of images, 5.0× under rollout). The looped one has a higher same-key rate (64%) but a low worst-layer score (3.5×), so consistency alone does not make a sink: all three conditions must hold together.

Fixed SSoG stays at or below 3.3× in every layer and 1.1-1.5× under rollout in every setting. Its most-attended key is identical across images (100%) only because its content-independent kernel always peaks at an image corner through border renormalization, a geometric effect that no input can move. Steering raises the worst-layer score moderately (5.3× discrete, 2.9× looped) and at the same time lowers the same-key rate from 100% to 60% and 44%, while rollout stays near uniform (1.7× and 1.6×). The added concentration therefore follows the content, as in the continuous dense model, rather than forming a fixed sink.

SSoG does not, however, avoid feature-norm artifacts altogether. On ImageNet, 29% of its final patch tokens have a norm above 3× the image median, against 3.5% for the dense model. These tokens are concentrated at the image border, which suggests a positional rather than content-driven origin, unlike the isolated high-norm tokens associated with registers in dense ViTs Darcet et al. (2024). We therefore claim the absence of attention sinks, not of norm artifacts.

## A.3 ABLATIONS

This appendix reports additional CIFAR-100 ablations that support the design choices made in §3. Unless stated otherwise, all rows use D = 768, H = 12, R = 4, NFE 12, and the training recipe of §3

## A.3.1 SIGMA FLOOR

The lower bound $\sigma _ { \mathrm { m i n } }$ on atom width (equation 4) trades off sharply local kernels against enforced smoothing. Table 6 sweeps this floor for fixed SSoG. The default value 0.25 performs best; lowering it to 0.10 costs 0.26 points and raising it to 0.50 costs 0.97 points, and the learned minimum sigma tracks the constraint in each case, confirming that the floor is active rather than a slack bound.

## A.3.2 STEERING BASELINES

Table 7 reports the results of steered SSoG, which serve as the steering baselines. For the continuousdepth model (ODE-SSoG), steering is evaluated at different NFE; for the discrete and looped models, it is evaluated at 12 blocks and 12 iterations, respectively.

Table 5: Attention-sink diagnostics of trained models on 512 validation images. First 3 settings are on CIFAR100 and the last setting on ImageNet. Sink score: attention received by the most-attended key relative to uniform (1 = no sink), averaged over heads, as the mean over layers, for the worst layer, and for the residual attention rollout $\textstyle \prod _ { l } ( A _ { l } + I ) / 2$ of the head-averaged maps. Same key: share of images whose most-attended key is the head’s most frequent one (averaged over heads; most consistent layer); for fixed SSoG it is 100% by construction, because its content-independent kernel always peaks at an image corner through border renormalization. CLS: attention mass from patch queries to the class token (- where the model has none; SSoG patches never attend to it). High-norm: share of final pre-norm patch tokens with norm above 3× the per-image median.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Operator</td><td colspan="3">Sink score</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Same key (%) CLS (%) High-norm (%)</td></tr><tr><td>Mean</td><td>Worst layer</td><td>Rollout</td></tr><tr><td rowspan="2">Continuous</td><td>MHA</td><td>11.3</td><td>22.9</td><td>1.7</td><td>6</td><td>0.3</td><td>0.0</td></tr><tr><td>SSoG</td><td>2.2</td><td>2.2</td><td>1.3</td><td>100</td><td>-</td><td>0.2</td></tr><tr><td rowspan="3">Discrete</td><td>MHA</td><td>3.8</td><td>8.1</td><td>5.0</td><td>49</td><td>0.9</td><td>0.0</td></tr><tr><td>SSoG</td><td>2.6</td><td>3.3</td><td>1.5</td><td>100</td><td>-</td><td>0.3</td></tr><tr><td>SSoG steered</td><td>3.3</td><td>5.3</td><td>1.7</td><td>60</td><td>-</td><td>0.7</td></tr><tr><td rowspan="3">Looped</td><td>MHA</td><td>2.8</td><td>3.5</td><td>2.8</td><td>64</td><td>1.0</td><td>0.1</td></tr><tr><td>SSoG</td><td>2.0</td><td>2.0</td><td>1.4</td><td>100</td><td>-</td><td>2.6</td></tr><tr><td>SSoG steered</td><td>2.3</td><td>2.9</td><td>1.6</td><td>44</td><td>-</td><td>0.1</td></tr><tr><td rowspan="2">Discrete</td><td>MHA</td><td>20.6</td><td>40.6</td><td>41.4</td><td>91</td><td>8.1</td><td>3.5</td></tr><tr><td>SSoG</td><td>2.0</td><td>2.5</td><td>1.1</td><td>100</td><td></td><td>29.0</td></tr></table>

Table 6: Effect of the sigma floor $\sigma _ { \mathrm { m i n } }$ on fixed SSoG.
<table><tr><td> $\sigma _ { \mathrm { m i n } }$ </td><td>Acc@1</td><td>Val. loss</td><td>Learned σ min</td></tr><tr><td>0.10</td><td>73.73</td><td>1.14</td><td>0.182</td></tr><tr><td>0.25</td><td>74.22</td><td>1.12</td><td>0.366</td></tr><tr><td>0.50</td><td>73.02</td><td>1.14</td><td>0.578</td></tr></table>

## A.3.3 STEERING INITIALIZATION

Table 8 isolates how quickly the content-steered pathway should activate. Warm initialization of full steering underperforms cold initialization by 5.33 points and plateaus early. Within center-only steering, moving the initialization of the gate $\tilde { \alpha } _ { \mu }$ of equation 12 from −8 to −4 costs 1.80 points despite both runs converging to a nearly identical final scale $\alpha _ { \mu }$ (0.127 vs. 0.126), indicating that the optimization trajectory, not only the final steering magnitude, determines the outcome.

## A.3.4 STEERING RADIUS

Table 9 sweeps the steering radius M in the full content-steered model (equation 9). The offset sweep is monotonic over the tested range: constraining M from 7 to 4 gains 1.59 points, and from 4 to 2 gains a further 0.36 points. Smaller M is partially compensated by a larger learned scale $\alpha _ { \mu } .$ but the effective radius $\alpha _ { \mu } M$ still grows with M (0.87, 0.89, 0.94 for $\dot { M } = 2 , \mathsf { \bar { 4 } } , 7 )$ , so the nominal limit remains the binding constraint.

## A.3.5 DISCRETE-SETTING WIDTH SWEEP

Table 10 gives the full width sweep: a 12-block discrete transformer using SSoG at every block, compared against a matched dense-attention transformer of identical depth, width, and training recipe.

## A.3.6 RESOLUTION ROBUSTNESS

In order to validate the robustness of our method whe test the trained models with different patch size. Table 13 reports the results of models trained at patch 16 (224px, 14 × 14 tokens) evaluated a

Table 7: Fixed versus steered SSoG on CIFAR-100 across the three architecture settings. Steps are ODE function evaluations (continuous), blocks (discrete) or loop iterations (looped). Time is the W&B training runtime for 500 epochs.
<table><tr><td>Setting</td><td>Variant</td><td>Steps</td><td>Acc@1</td><td>Acc@3</td><td>Acc@5</td><td>Time (h)</td></tr><tr><td rowspan="6">Continuous</td><td>Fixed</td><td>6</td><td>69.70</td><td>85.74</td><td>90.23</td><td>25.5</td></tr><tr><td>Fixed</td><td>8</td><td>72.16</td><td>87.30</td><td>91.05</td><td>33.2</td></tr><tr><td>Fixed</td><td>12</td><td>74.22</td><td>88.59</td><td>92.38</td><td>47.8</td></tr><tr><td>Steered</td><td>6</td><td>70.36</td><td>86.04</td><td>90.67</td><td>31.5</td></tr><tr><td>Steered</td><td>8</td><td>72.00</td><td>87.16</td><td>91.42</td><td>40.3</td></tr><tr><td>Steered</td><td>12</td><td>70.89</td><td>86.89</td><td>91.30</td><td>59.5</td></tr><tr><td rowspan="2">Discrete</td><td>Fixed</td><td>12</td><td>78.63</td><td>90.12</td><td>93.38</td><td>33.40</td></tr><tr><td>Steered</td><td>12</td><td>81.17</td><td>90.21</td><td>91.67</td><td>57.81</td></tr><tr><td rowspan="2">Looped</td><td>Fixed</td><td>12</td><td>79.80</td><td>90.58</td><td>93.10</td><td>23.90</td></tr><tr><td>Steered</td><td>12</td><td>81.20</td><td>91.12</td><td>93.56</td><td>49.23</td></tr></table>

Table 8: Effect of steering initialization at NFE 12, $R = 4 .$
<table><tr><td>Initialization</td><td>Final Acc@1</td><td>Val. loss</td><td>Instabilities</td></tr><tr><td>Cold, full steering</td><td>70.89</td><td>1.2017</td><td>None</td></tr><tr><td>Warm, full steering</td><td>65.56</td><td>1.3714</td><td>Early plateau</td></tr><tr><td>Center-only,  ${ \tilde { \alpha } } _ { \mu } = - 8$ </td><td>73.33</td><td>1.1546</td><td>None</td></tr><tr><td>Center-only,  $\tilde { \alpha } _ { \mu } = - 4$ </td><td>71.53</td><td>1.2138</td><td>None</td></tr></table>

patch 14 and 8 without retraining. Patch-embedding kernels are PI-resized and position embeddings bicubically interpolated for every model; the rescaled SSoG row also scales $\mu , \sigma$ by the grid ratio so each Gaussian covers the same image.

## A.4 EFFECTIVE ATOMS

To measure how many of the R Gaussian atoms a head actually uses, we take its learned mixture weights $\lambda _ { r } ^ { h } = \operatorname { s o f t m a x } _ { r } ( \ell _ { r } ^ { h } )$ and compute their Shannon entropy and its exponential,

$$
R _ { \mathrm { e f f } } ^ { h } = \exp \Big ( - \sum _ { r = 1 } ^ { R } \lambda _ { r } ^ { h } \log \lambda _ { r } ^ { h } \Big ) , \qquad \overline { { R } } _ { \mathrm { e f f } } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } R _ { \mathrm { e f f } } ^ { h } .\tag{23}
$$

$R _ { \mathrm { e f f } } ^ { h }$ ranges from 1 (all mass on one atom) to R (uniform mixture), and we report its mean over heads. It is computed on the fixed per-head weights, before content-dependent steering, so it shows how the model spends its atom budget rather than any single query’s mixture.

## A.4.1 IMAGENET WIDTH CONFIRMATION

Table 14 reports the ImageNet width sweep for fixed SSoG at constant head dimension $D ^ { h } = 6 4$ Best-epoch Top-1 improves from 55.65% to 62.43% to 64.02% as D increases from 384 to 576 to 768, with diminishing gains from 576 to 768.

## A.5 AUGMENTATION AND IMAGENET-TO-CIFAR-100 TRANSFER

Table 15 sweeps training augmentation strength for fixed SSoG trained from scratch on CIFAR-100: stronger augmentation is preferred when learning the full representation from random initialization. Table 16 instead adapts to CIFAR-100, either through a linear probe or full fine-tuning; here weaker augmentation is preferred, reversing the from-scratch trend, consistent with adapting rather than learning a representation from random initialization.

Table 9: Effect of the steering radius M, full steering, NFE 12.
<table><tr><td>M</td><td>Acc@1</td><td>Val. loss</td><td>Mean |µ|</td></tr><tr><td>2</td><td>72.84</td><td>1.1385</td><td>0.529</td></tr><tr><td>4</td><td>72.48</td><td>1.1643</td><td>0.535</td></tr><tr><td>7</td><td>70.89</td><td>1.2017</td><td>0.475</td></tr></table>

Table 10: Width scaling on CIFAR-100, dense attention versus SSoG, across the three architecture settings. Params are per model; bold marks the smaller parameter count and the better Acc@1 within each row.
<table><tr><td>Setting</td><td>D</td><td>MHA Params(M)</td><td>SSoG Params</td><td>MHA Acc@1</td><td>SSoG Acc@1</td></tr><tr><td rowspan="3">Continuous</td><td>384</td><td>3.07</td><td>2.8</td><td>69.15</td><td>71.37</td></tr><tr><td>576</td><td>6.59</td><td>5.9</td><td>69.36</td><td>73.48</td></tr><tr><td>768</td><td>11.45</td><td>10.3</td><td>70.47</td><td>74.22</td></tr><tr><td rowspan="3">Discrete</td><td>384</td><td>21.69</td><td>18.5</td><td>74.51</td><td>77.56</td></tr><tr><td>576</td><td>48.47</td><td>41.0</td><td>74.20</td><td>78.18</td></tr><tr><td>768</td><td>85.86</td><td>71.6</td><td>74.67</td><td>78.63</td></tr><tr><td rowspan="3">Looped</td><td>384</td><td>2.18</td><td>1.9</td><td>72.04</td><td>77.46</td></tr><tr><td>576</td><td>4.60</td><td>3.9</td><td>71.24</td><td>79.38</td></tr><tr><td>768</td><td>7.91</td><td>6.7</td><td>72.30</td><td>79.80</td></tr></table>

![](images/23b64debffe5e05cff1a66111d6d37645eb4422edd32bd0cfd33978fef9317d4.jpg)  
Figure 5: Throughput and peak memory versus input resolution, dense attention versus SSoG, across all three architecture settings (rows: continuous-depth ODE-ViT, discrete ViT-B/16, looped bViT) and six batch sizes (columns: 8, 32, 64, 128, 256, 512), at the paper’s flagship width $( D = 7 6 8 , H =$ 12; NFE 12 for the continuous-depth setting, depth 12 for the discrete and looped settings). Each panel shares a log-scaled resolution axis (224-1024) with two y-axes: throughput in images/s (blue, left) and peak allocated memory in MiB (magenta, right). Solid circles/lines denote dense attention; dashed triangles denote SSoG. An × (dense) or + (SSoG) marks the first resolution at which that configuration ran out of GPU memory; its vertical position is the last successfully measured value at that batch size, not a measured point at the marked resolution, and no further resolutions are plotted for that curve beyond it. All measurements are FP16 autocast (model parameters and inputs in FP32) on a single NVIDIA A40, randomly initialized weights, a single run per configuration (no repeatedseed or repeated-measurement uncertainty), 10 warmup and 50 measured iterations per point.

Table 11: Capacity and depth ablations on CIFAR-100 for the three SSoG model families. The Depth column gives the number of blocks for the Discrete family, the number of loop iterations for the Looped family, and the number of ODE function evaluations (NFE) for the Continuous family. The Heads and Atoms blocks are evaluated at NFE 12 where applicable, while the Depth block varies depth or NFE at fixed D = 768, H = 12, R = 4. The column highlighted in blue is the parameter varied within each block; bold marks the best metric within each block.
<table><tr><td>Family</td><td>Ablation</td><td>D</td><td>H</td><td>R</td><td>Depth</td><td>Par(M)</td><td>Acc@1</td><td>Acc@3</td><td>Acc@5</td></tr><tr><td rowspan="5">DISCRETE</td><td>Heads</td><td>768</td><td>6 12 24</td><td>4</td><td>12</td><td>71.6</td><td>79.97 78.63 79.52</td><td>89.99 90.12 88.59</td><td>91.94 93.38 90.14</td></tr><tr><td>Atoms</td><td>768</td><td>12</td><td>1 2 4</td><td>12</td><td>71.6</td><td>79.62 80.34</td><td>89.38 89.83</td><td>91.52 91.69</td></tr><tr><td></td><td></td><td></td><td>8</td><td>6</td><td>36.2</td><td>78.63 79.93 79.55</td><td>90.12 88.67</td><td>93.38 90.35 93.89</td></tr><tr><td rowspan="2">Depth</td><td rowspan="2">768</td><td rowspan="2">12</td><td>4</td><td rowspan="2">8</td><td>48.0</td><td>79.70</td><td>90.93 90.24</td><td>92.66</td></tr><tr><td>12</td><td>71.6</td><td>78.63</td><td>90.12</td><td>93.38</td></tr><tr><td rowspan="5">LOOPED</td><td>Heads</td><td>768</td><td>6 12 24</td><td>4</td><td>12</td><td>6.7</td><td>79.32 79.80 79.51</td><td>90.49 90.58 90.59</td><td>92.85 93.10 93.32</td></tr><tr><td>Atoms</td><td>768</td><td>12</td><td>1 2 4</td><td>12</td><td>6.7</td><td>80.31 79.67</td><td>90.68 90.57</td><td>93.28 93.09</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td>8</td><td rowspan="2"></td><td rowspan="2"></td><td>79.80</td><td>90.58</td><td>93.10</td></tr><tr><td></td><td>79.73</td><td>90.60</td><td>93.34</td></tr><tr><td rowspan="2">Depth</td><td rowspan="2">768</td><td rowspan="2">12</td><td></td><td>6 8 12</td><td rowspan="2"></td><td>79.23</td><td>91.18</td><td>93.96</td></tr><tr><td></td><td>4</td><td>6.7</td><td>78.88 79.80</td><td>90.18 90.58</td><td>93.17 93.10</td></tr><tr><td rowspan="5">CONTINUOUS</td><td>Heads</td><td>768</td><td>6 12 24</td><td>4</td><td>12</td><td>10.3</td><td>73.62 74.22</td><td>88.39 88.59</td><td>92.11 92.38</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td>1 2</td><td rowspan="2"></td><td rowspan="2">10.3</td><td>74.40</td><td>88.44</td><td>92.37</td></tr><tr><td></td><td>73.42</td><td>88.25</td><td>92.05</td></tr><tr><td rowspan="2">Atoms</td><td rowspan="2">768</td><td rowspan="2">12</td><td>4</td><td rowspan="2">12</td><td rowspan="2"></td><td>74.13</td><td>88.36</td><td>91.95</td></tr><tr><td></td><td>74.22 73.99</td><td>88.59</td><td>92.38</td></tr><tr><td rowspan="4"></td><td rowspan="2"></td><td rowspan="2"></td><td></td><td>8</td><td rowspan="2"></td><td rowspan="4"></td><td></td><td>88.58</td><td>92.47</td></tr><tr><td></td><td>6 8</td><td>69.70</td><td>85.74</td><td>90.23</td></tr><tr><td rowspan="2">Depth</td><td rowspan="2">768</td><td rowspan="2">12</td><td>4</td><td>10.3</td><td>72.16 74.22</td><td>87.30</td><td>91.05</td></tr><tr><td>12 16</td><td></td><td>74.92</td><td>88.59 89.02</td><td>92.38 92.46</td></tr></table>

Table 12: MHA capacity and depth ablations on CIFAR-100 for the Discrete MHA (ViT) and Looped MHA (bViT) families. The column highlighted in blue is the parameter varied within each block; bold marks the best metric within each block.
<table><tr><td>Family</td><td>Ablation</td><td>D</td><td>H</td><td>Depth</td><td>Params</td><td>Acc@1</td><td>Acc@3</td><td> $\operatorname { A c c } ( { \widehat { \varrho } } 5 $ </td></tr><tr><td rowspan="4">DISCRETE</td><td>Heads</td><td>768</td><td>6 12</td><td>12</td><td>85.9M</td><td>75.63 74.67</td><td>87.51 86.89</td><td>90.55 90.19</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>24</td><td></td><td></td><td>73.70</td><td>86.0</td><td>89.30</td></tr><tr><td></td><td>6</td><td>43.3M</td><td>70.47</td><td>84.50</td><td>88.61</td></tr><tr><td rowspan="2">Depth</td><td rowspan="2">768</td><td rowspan="2">12</td><td>8</td><td>57.5M 85.9M</td><td>72.22</td><td>85.68</td><td>89.60</td></tr><tr><td>12</td><td></td><td>74.67</td><td>86.89</td><td>90.19</td></tr><tr><td rowspan="5">LOOPED</td><td rowspan="2">Heads</td><td rowspan="2">768</td><td>6</td><td>12</td><td rowspan="2">7.9M</td><td>73.18</td><td>85.80</td><td>89.41</td></tr><tr><td>12</td><td></td><td>72.30</td><td>84.37</td><td>88.33</td></tr><tr><td rowspan="2">Depth</td><td rowspan="2">768</td><td>24</td><td>6</td><td rowspan="2">7.9M</td><td>70.07</td><td>83.13</td><td>86.89</td></tr><tr><td>12</td><td></td><td>68.72</td><td>83.44</td><td>87.69</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td>8</td><td rowspan="2"></td><td>70.99</td><td>84.57</td><td>88.60</td></tr><tr><td>12</td><td>72.30</td><td>84.37</td><td>88.33</td></tr></table>

Table 13: Patch-size robustness on CIFAR-100: models trained at patch 16 (224px, $1 4 \times 1 4$ tokens) evaluated at patch 14 and 8 without retraining. Patch-embedding kernels are PI-resized and position embeddings bicubically interpolated for every model; the rescaled SSoG row also scales µ, σ by the grid ratio so each Gaussian covers the same image area. Acc@1 (%), change vs. patch 16 in parentheses.
<table><tr><td>Setting</td><td>Operator</td><td>Patch 16  $( 1 4 \times 1 4 )$ </td><td>Patch 14  $( 1 6 \times 1 6 )$ </td><td>Patch 8 (28 × 28)</td></tr><tr><td rowspan="3">Continuous</td><td>MHA</td><td>70.47</td><td>70.23 (+0.80)</td><td>58.05 (−11.38)</td></tr><tr><td>SSoG</td><td>74.20</td><td>74.85(+0.65)</td><td>57.44 (−16.76)</td></tr><tr><td>SSoG (rescaled  $\mu , \sigma )$ </td><td>74.20</td><td>74.30 (+0.10)</td><td>64.39(-9.81)</td></tr><tr><td rowspan="3">Discrete</td><td>MHA</td><td>74.67</td><td>74.91 (−0.18)</td><td>56.92 (-18.17)</td></tr><tr><td>SSoG</td><td>78.63</td><td>79.34 (+0.71)</td><td>63.49 (-15.53)</td></tr><tr><td>SSoG (rescaled  $\mu , \sigma )$ </td><td>78.63</td><td>79.48(+0.85)</td><td>72.09(-6.93)</td></tr><tr><td rowspan="3">Looped</td><td>MHA</td><td>72.30</td><td>72.21 (−0.09)</td><td>52.24 (-20.06)</td></tr><tr><td>SSoG</td><td>79.80</td><td>80.98(+1.18)</td><td>73.56 (-6.74)</td></tr><tr><td>SSoG (rescaled  $\mu , \sigma )$ </td><td>79.80</td><td>80.12 (+0.42)</td><td>73.59(−-6.71)</td></tr></table>

Table 14: ImageNet width sweep, fixed SSoG, $R = 4 ,$ , NFE 12.
<table><tr><td> $D$ </td><td>H</td><td> $\operatorname { A c c } ( \varnothing 1$ </td><td> $\operatorname { A c c } ( { \widehat { \varrho } } 5 $ </td></tr><tr><td>384</td><td>6</td><td>55.65 (245)</td><td>78.46</td></tr><tr><td>576</td><td>9</td><td>62.43 (273)</td><td>83.37</td></tr><tr><td>768</td><td>12</td><td>64.02 (254)</td><td>84.45</td></tr></table>

Table 15: From-scratch augmentation ablation, $D = 7 6 8 .$ $R = 4$ , NFE 12.
<table><tr><td>Augmentation</td><td>Best Acc@1 (epoch)</td><td>Final Acc@1</td><td>Runtime (h)</td></tr><tr><td>Strong</td><td>74.22 (488)</td><td>74.22</td><td>47.77</td></tr><tr><td>Moderate</td><td>73.09 (487)</td><td>72.72</td><td>34.08</td></tr><tr><td>Weak</td><td>62.70 (452)</td><td>62.27</td><td>33.26</td></tr></table>

Table 16: ImageNet→CIFAR-100 transfer, D = 768 backbone from an ImageNet checkpoint, across all three architecture settings.
<table><tr><td>Setting</td><td>Protocol / Augmentation</td><td></td><td>Best Acc@1</td></tr><tr><td>Continuous</td><td>Linear probe (frozen)</td><td>Hard</td><td>66.26 (43)</td></tr><tr><td></td><td>Linear probe (frozen)</td><td>Weak</td><td>70.01 (45)</td></tr><tr><td></td><td>Full fine-tune</td><td>Hard</td><td>80.25 (91)</td></tr><tr><td></td><td>Full fine-tune</td><td>Moderate</td><td>83.84 (83)</td></tr><tr><td>Discrete</td><td>Linear probe (frozen)</td><td>Hard</td><td>66.77 (48)</td></tr><tr><td></td><td>Linear probe (frozen)</td><td>Weak</td><td>69.05 (43)</td></tr><tr><td></td><td>Full fine-tune</td><td>Hard</td><td>86.71 (92)</td></tr><tr><td></td><td>Full fine-tune</td><td>Moderate</td><td>87.09 (95)</td></tr><tr><td>Looped</td><td>Linear probe (frozen)</td><td>Hard</td><td>68.71 (48)</td></tr><tr><td></td><td>Linear probe (frozen)</td><td>Weak</td><td>70.96 (47)</td></tr><tr><td></td><td>Full fine-tune</td><td>Hard</td><td>82.98 (97)</td></tr><tr><td></td><td>Full fine-tune</td><td>Moderate</td><td>84.32 (99)</td></tr></table>

## A.6 LIPSCHITZ CONTINUITY OF SSOG

In the continuous-depth setting of equation 16, the token matrix $X ( t )$ evolves under the vector field $F ( \cdot , \cdot | \theta )$ . By the Picard–Lindelof theorem Coddington & Levinson (1955), if¨ $F$ is continuous in t and Lipschitz continuous in X with a constant independent of t and of X, i.e. globally Lipschitz, then equation 16 has a unique solution on the whole interval [0, 1] for every initial condition $X _ { 0 }$ . If $F$ is only Lipschitz on bounded sets, the same theorem guarantees existence and uniqueness only on a possibly shorter interval, which extends to [0, 1] provided the trajectory remains bounded.

The vector field $F ( \cdot , \cdot | \theta )$ is built from the SSoG spatial operator, the value and output projections $\boldsymbol { W } _ { V } ^ { h }$ and $W _ { O }$ , the feed-forward block and layer normalization. Since sums and compositions of globally Lipschitz maps are globally Lipschitz, it suffices to bound each component. Linear maps are globally Lipschitz with constant equal to their spectral norm; the feed-forward block is globally Lipschitz with constant at most the product of the spectral norms of its weight matrices and the Lipschitz constant of its activation; and layer normalization with a stabilizing constant $\varepsilon _ { \mathrm { L N } } > 0$ is globally Lipschitz (see the discussion following Proposition 1). The only non-trivial component is therefore the SSoG spatial operator. We show that fixed SSoG is globally Lipschitz (Proposition 1), so that ODE-SSoG with fixed kernels has a unique solution on [0, 1], whereas content-steered SSoG is only Lipschitz on bounded sets (Proposition 2), so that for this variant only the local guarantee applies.

Notation. We recall the notation of §2. For head h, the value projection is $V ^ { h } = X W _ { V } ^ { h }$ (equation 1). For each atom r and axis $a \in \{ x , y \}$ , with $\begin{array} { r } { \delta _ { a } : = q _ { a } - s _ { a } , \operatorname { l e t } g _ { r , a } ( \delta _ { a } ) : = \exp \bigl ( \log G ^ { a } ( \delta _ { a } ) / \tau \bigr ) } \end{array}$ denote the temperature-scaled Gaussian kernel of atom $r ,$ where log $G ^ { a }$ is given by equation 2 and τ is the temperature of equation 3. The normalized axis kernel $A _ { r , a } ( \delta _ { a } )$ coincides with $A _ { a } ( \delta _ { a } )$ from equation 3 evaluated for atom r. The mixture weights $\left( \lambda _ { r } \right) _ { { } }$ <sub>r</sub> lie in the probability simplex (equation 4), and the per-block class-token aggregate is $\begin{array} { r } { v _ { \mathrm { c l s } } = \frac { 1 } { N - 1 } \sum _ { p = 1 } ^ { N - 1 } v _ { p } } \end{array}$ (equation 14). All norms on token matrices are Frobenius; norms on linear maps are spectral.

We drop the batch index, since every operation is separable across the batch.

Lemma 1 (Kernel norm). For each atom r and each axis $a \in \{ x , y \}$ , the axis kernel of equation 3 satisfies $\| A _ { r , a } \| _ { 2 } \le \sqrt { \kappa _ { r , a } }$ , where

$$
\kappa _ { r , a } = \frac { \operatorname* { m a x } _ { s _ { a } } \sum _ { q _ { a } } g _ { r , a } ( \delta _ { a } ) } { \operatorname* { m i n } _ { q _ { a } } \sum _ { s _ { a } } g _ { r , a } ( \delta _ { a } ) } , \quad \delta _ { a } : = q _ { a } - s _ { a } \geq 1\tag{24}
$$

and $g _ { r , a } ( \delta _ { a } )$ is the temperature-scaled Gaussian kernel $g _ { r , a } ( \delta _ { a } ) = \exp \bigl ( \log G ^ { a } ( \delta _ { a } ) / \tau \bigr )$ , with log $G ^ { \bar { a } }$ as in equation 2 evaluatedfor atom r and τ the temperature ofequation 3.

Proof. Since all entries of $A _ { r , a }$ are positive, $\| A _ { r , a } \| _ { \infty }$ and $\Vert A _ { r , a } \Vert .$ are its maximum row sum and maximum column sum, respectively. We bound each in turn.

By equation 3, each row of $A _ { r , a }$ is a softmax over the source coordinate $\begin{array} { r } { s _ { a } , { \bf s } _ { 0 } \sum _ { s _ { a } } A _ { r , a } ( q _ { a } , s _ { a } ) = 1 } \end{array}$ for every $q _ { a }$ . Hence $\| A _ { r , a } \| _ { \infty } = 1$

For a fixed column $s _ { a }$ , each entry $A _ { r , a } ( q _ { a } , s _ { a } )$ equals $g _ { r , a } ( q _ { a } \mathrm { ~ - ~ } s _ { a } )$ divided by the row sum $\begin{array} { r } { \sum _ { s _ { a } ^ { \prime } } g _ { r , a } ( q _ { a } - s _ { a } ^ { \prime } ) } \end{array}$ . Bounding every such denominator from below by its minimum over $q _ { a }$ gives

$$
\sum _ { q _ { a } } A _ { r , a } ( q _ { a } , s _ { a } ) = \sum _ { q _ { a } } \frac { g _ { r , a } ( q _ { a } - s _ { a } ) } { \sum _ { s _ { a } ^ { \prime } } g _ { r , a } ( q _ { a } - s _ { a } ^ { \prime } ) } \leq \frac { \displaystyle \sum _ { q _ { a } } g _ { r , a } ( q _ { a } - s _ { a } ) } { \displaystyle \operatorname* { m i n } _ { q _ { a } } \sum _ { s _ { a } ^ { \prime } } g _ { r , a } ( q _ { a } - s _ { a } ^ { \prime } ) } .
$$

The denominator on the right no longer depends on $s _ { a } ,$ , so taking the maximum over $s _ { a }$ on both sides yields $\| A _ { r , a } \| _ { 1 } \le \kappa _ { r , a } .$ , with $\kappa _ { r , a }$ as in equation 24.

Finally, every real matrix M satisfies $\| M \| _ { 2 } \leq \sqrt { \| M \| _ { 1 } \| M \| _ { \infty } } .$ , since $\| M \| _ { 2 } ^ { 2 } = \rho ( M ^ { \top } M ) \ \leq$ $\begin{array} { r } { \| M ^ { \top } M \| _ { \infty } \leq \| M ^ { \top } \| _ { \infty } \| M \| _ { \infty } = \| M \| _ { 1 } \| M \| _ { \infty } } \end{array}$ . Applying this to $A _ { r , a }$ with the two bounds above gives $\| A _ { r , a } \| _ { 2 } \leq \sqrt { \kappa _ { r , a } } .$ □

Remark. The constant $\kappa _ { r , a }$ is finite for every choice of atom parameters, which is all that Proposition 1 requires; with truncated boundaries, however, it does not approach one as the grid grows. A query $q _ { a }$ at the edge of the axis only retains the part of $g _ { r , a } ( q _ { a } , \cdot )$ that falls inside the grid, whereas an interior column collects the full kernel mass. Consequently $\kappa _ { r , a } \approx 1$ only when the effective width $\sqrt { \tau } \sigma _ { r , a }$ is small compared with the patch spacing and the offset $\mu _ { r , a }$ is close to zero; for a centred kernel of larger width $\kappa _ { r , a }$ is close to $2 ,$ and it grows rapidly with $\left| \mu _ { r , a } \right|$ |, since edge rows then retain only a tail of the kernel. The bound $\sqrt { \kappa _ { r , a } }$ is therefore conservative.

Proposition 1 (Fixed SSoG is globally Lipschitz). Let $\boldsymbol { X } \in \mathbb { R } ^ { N \times D }$ consist of one class token and $N - 1 = n _ { y } n _ { x }$ patch tokens arranged on an $n _ { y } \times n _ { x }$ grid. For each head $h \in \{ 1 , \ldots , H \}$ , assume that the atom parameters $\mu _ { r , a } , \sigma _ { r , a } , \ell _ { r }$ and the temperature τ are fixed, i.e. independent of X. Let $W _ { V } ^ { h } \in \mathbb { R } ^ { D \times D ^ { h } }$ be the value projections of equation 1, $W _ { V } = W _ { V } ^ { 1 } \parallel \perp \perp W _ { V } ^ { H } \in \mathbb { R } ^ { D \times D }$ their concatenation, and $W _ { O } \in \mathbb { R } ^ { \bar { D } \times D }$ the output projection. Let

$$
\mathcal { A } ^ { h } = \sum _ { r = 1 } ^ { R } \lambda _ { r } ^ { h } \left( A _ { r , y } ^ { h } \otimes A _ { r , x } ^ { h } \right) \in \mathbb { R } ^ { \left( N - 1 \right) \times \left( N - 1 \right) }\tag{25}
$$

be the effective spatial kernel of head h on the patch tokens. Then fixed SSoG is a linear map on $\mathbb { R } ^ { N \times D }$ and, with the Frobenius norm on token matrices,

$$
\mathrm { L i p ( S S o G ) } \leq \| W _ { V } \| _ { 2 } \| W _ { O } \| _ { 2 } K _ { \cal A } ,\tag{26}
$$

where

$$
\begin{array} { r } { K _ { \mathcal { A } } : = \displaystyle \operatorname* { m a x } _ { h } \sqrt { \frac { 1 } { N - 1 } + \| \mathcal { A } ^ { h } \| _ { 2 } ^ { 2 } } \leq \displaystyle \operatorname* { m a x } _ { h } \sqrt { \frac { 1 } { N - 1 } + \operatorname* { m a x } _ { r } \kappa _ { r , y } ^ { h } \kappa _ { r , x } ^ { h } } , } \end{array}\tag{27}
$$

and $\kappa _ { r , a } ^ { h }$ is the constant of Lemma 1 for atom r of head h. The bound depends only on the parameters and not on the input.

Proof. Fix a head $h ,$ order the tokens with the class token last, as in $\ S 2 ,$ and write $V ^ { h } = X W _ { V } ^ { h } =$ $\binom { V _ { \mathrm { P } } ^ { h } } { v _ { \mathrm { c l s } } ^ { h } }$ , where $V _ { \mathrm { P } } ^ { h } \in \mathbb { R } ^ { ( N - 1 ) \times D ^ { h } }$ collects the patch rows. Vectorizing the grid in row-major order and using vec $( A U B ^ { \top } ) = ( A \otimes B )$ vec(U), equation 8 gives the filtered patch rows $\mathcal { A } ^ { h } V _ { \mathrm { P } } ^ { h }$ , while by equation 14 the class row is $\frac { 1 } { N - 1 } \mathbf { 1 } ^ { \top } V _ { \mathrm { P } } ^ { h }$ . Hence the output of head h is $M ^ { h } X W _ { V } ^ { h }$ , with

$$
\boldsymbol { M } ^ { h } = \left( \begin{array} { c c } { \mathcal { A } ^ { h } } & { 0 } \\ { \frac { 1 } { N - 1 } { \bf 1 } ^ { \top } } & { 0 } \end{array} \right) \in \mathbb { R } ^ { N \times N } .
$$

In the fixed variant $M ^ { h }$ does not depend on X. Denoting by

$$
Z ^ { h } = M ^ { h } X W _ { V } ^ { h } \in \mathbb { R } ^ { N \times D ^ { h } }
$$

the output of head $h ,$ the SSoG output is obtained by concatenating the heads and applying the output projection,

$$
\operatorname { S S o G } ( X ) = { \big ( } Z ^ { 1 } \parallel \cdots \parallel Z ^ { H } { \big ) } W _ { O } ,
$$

which is linear in $X ;$ its Lipschitz constant is therefore its operator norm. Since the heads occupy disjoint column blocks,

$$
\begin{array} { r l } {  { \| \mathrm { S S o G } ( X ) \| _ { F } \le \| W _ { O } \| _ { 2 } \Big ( \sum _ { h } \| Z ^ { h } \| _ { F } ^ { 2 } \Big ) ^ { 1 / 2 } \le \| W _ { O } \| _ { 2 } \operatorname* { m a x } _ { h } \| M ^ { h } \| _ { 2 } \| X W _ { V } \| _ { F } } } \\ & { \le \| W _ { V } \| _ { 2 } \| W _ { O } \| _ { 2 } \operatorname* { m a x } _ { h } \| M ^ { h } \| _ { 2 } \| X \| _ { F } . } \end{array}
$$

Since the last column of $M ^ { h }$ vanishes, $\| M ^ { h } \| _ { 2 }$ equals the spectral norm of its first $N - 1$ columns; since, moreover, the squared spectral norm of a matrix equals the spectral norm of its Gram matrix,

$$
\begin{array} { r } { \| M ^ { h } \| _ { 2 } ^ { 2 } = \left\| \left( \underset { N - 1 } { \overset { \mathcal { A } ^ { h } } { \longrightarrow } } \mathbf { 1 } ^ { \top } \right) \right\| _ { 2 } ^ { 2 } = \left\| ( \mathcal { A } ^ { h } ) ^ { \top } \mathcal { A } ^ { h } + \frac { 1 } { ( N - 1 ) ^ { 2 } } \mathbf { 1 } \mathbf { 1 } ^ { \top } \right\| _ { 2 } \leq \| \mathcal { A } ^ { h } \| _ { 2 } ^ { 2 } + \frac { 1 } { N - 1 } , } \end{array}
$$

where the inequality follows from the triangle inequality and $\| \mathbf { 1 1 } ^ { \top } \| _ { 2 } ~ = ~ N - 1$ , so that max $\mathit { \bar { t } } _ { h } \| M ^ { h } \| _ { 2 } \leq K \bar { _ { A } }$ , which gives equation 26. For the inequality in equation 27, $\| A \otimes B \| _ { 2 } =$ $\| A \| _ { 2 } \| B \| _ { 2 }$ and $( \lambda _ { r } ^ { h } ) _ { : }$ <sub>r</sub> lies in the simplex by equation $^ { 4 , }$ so equation $2 5$ is a convex combination and, by Lemma $1 , \| A ^ { h } \| _ { 2 } \leq \operatorname* { m a x } _ { r } \| A _ { r , y } ^ { h } \| _ { 2 } \| A _ { r , x } ^ { h } \| _ { 2 } \leq$ max<sub>r</sub> $\sqrt { \kappa _ { r , y } ^ { h } \kappa _ { r , x } ^ { h } }$ □

Proposition 1 has two immediate consequences. For the discrete and looped models, the residual block of equation 15 is globally Lipschitz with constant at most $1 + \| W _ { V } \| _ { 2 } \| W _ { O } \| _ { 2 } K _ { A }$ . Since $K _ { A }$ is finite for any choice of parameters, the only factors not determined by the kernel are the projection norms, which spectral normalization bounds explicitly and weight decay keeps moderate in practice. For ODE-SSoG, the vector field $F ( \cdot , \cdot | \theta )$ of equation 16 composes fixed SSoG with the feed-forward block and layer normalization. The feed-forward block is globally Lipschitz, with constant at most the product of the spectral norms of its weight matrices and the Lipschitz constant of its activation. Layer normalization with gain $\gamma$ and stabilizing constant $\varepsilon _ { \mathrm { L N } } > 0$ added to the variance has Jacobian of spectral norm at most max<sub>i</sub> $| \gamma _ { i } | / \sqrt { \varepsilon _ { \mathrm { L N } } }$ , so it is globally Lipschitz as well. Hence $F ( \cdot , \cdot | \theta )$ is continuous in t and globally Lipschitz in $X$ with a constant independent of $t ,$ and the Picard–Lindelof theorem yields a unique solution of equation 16 on¨ [0, 1].

We now turn to content-steered SSoG. Steering acts on each head independently: the predictors of equation 9 modify only the atom parameters of their own head, and different heads interact only through the output projection $W _ { O } .$ , which is handled exactly as in Proposition 1. We therefore first analyse a single head. In Lemma 2 and its proof we fix one head and, to lighten notation, omit the superscript h from all of its quantities; the superscript is restored in Proposition 2, where all heads are combined.

Intuition and notation for Lemma 2. Steering can only move each atom within a bounded range. Since $| \operatorname { t a n h } | < 1$ , equation 9 shifts the center $\mu _ { r , a }$ by at most $\alpha _ { \mu } M$ and multiplies the spread $\sigma _ { r , a }$ by a factor between $e ^ { - \alpha \sigma }$ and $e ^ { \alpha _ { \sigma } }$ . Together with the floor $\sigma _ { r , a } \geq \sigma _ { \operatorname* { m i n } } ,$ this confines every steered spread to an interval $[ \sigma _ { \star } , \sigma ^ { \star } ]$ , and every distance between a grid offset $\delta _ { a } \in \{ - ( n _ { a } - 1 ) , \ldots , n _ { a } - 1 \}$ and a steered center to at most $\Delta$ , where

$$
\sigma _ { \star } = \sigma _ { \operatorname* { m i n } } e ^ { - \alpha _ { \sigma } } , \qquad \sigma ^ { \star } = \operatorname* { m a x } _ { r , a } ^ { } \sigma _ { r , a } e ^ { \alpha _ { \sigma } } , \qquad \Delta = \operatorname* { m a x } _ { r , a } \left( n _ { a } - 1 + | \mu _ { r , a } | \right) + \alpha _ { \mu } M .\tag{28}
$$

Thus $\sigma _ { \ast }$ is the narrowest spread an atom can reach, $\sigma ^ { \star }$ the widest, and $\Delta$ the largest distance between a steered center and a grid offset. The first and the last bound how fast the logits of the axis softmax, log $g _ { r , a } ( \delta _ { a } )$ , react to the atom parameters: on the logarithmic scale a Gaussian is a parabola, whose slope grows as the spread shrinks and as the offset moves away from the center, so the worst case is a spread $\sigma _ { \star }$ at distance $\Delta$ . The kernel value itself is close to zero far from the center, and the softmax damps changes in small entries, so bounding the kernel through its logits is conservative. The widest spread $\sigma ^ { \star }$ matters for a different reason: steering rescales the spread multiplicatively, so the same change in the predictor output moves a wide atom more than a narrow one. Lemma 2 turns these bounds into a Lipschitz constant, namely how fast the atom parameters follow the query token times how fast the kernel follows its parameters.

Lemma 2 (Steered kernel). Consider a single head, and let its atom parameters be steered by the query token $x _ { q }$ as in equation 9, with steering radius $M$ , predictor weights $W _ { \mu } , W _ { \sigma } , W _ { \lambda }$ and scales $\alpha _ { \mu } , \alpha _ { \sigma } , \alpha _ { \lambda } > 0$ as in equation 12, and with $\sigma _ { r , a } ~ \geq ~ \sigma _ { \operatorname* { m i n } }$ and $\tau \geq \tau _ { 0 }$ as in equation 4. Denote by $A _ { r , a } ( x _ { q } ) \in \mathbb { R } ^ { n _ { a } }$ the row of $A _ { r , a }$ associated with the query position q, and by $A ( x _ { q } ) =$ $\begin{array} { r } { \sum _ { r } \lambda _ { r } ( \dot { x _ { q } } ) A _ { r , y } ( x _ { q } ^ { ' } ) \otimes A _ { r , x } ( x _ { q } ) } \end{array}$ the corresponding row ofthe effective kernel. Let $\sigma _ { \star } , \sigma ^ { \star }$ and $\Delta b e$ as in equation 28. Then:

(i) for every $x _ { q } ,$ , r and $^ { a , }$

$$
\big ( \mu _ { r , a } ( x _ { q } ) , \sigma _ { r , a } ( x _ { q } ) \big ) \in \Theta _ { r , a } : = [ \mu _ { r , a } - \alpha _ { \mu } M , \mu _ { r , a } + \alpha _ { \mu } M ] \times [ \sigma _ { \star } , \sigma ^ { \star } ] ;\tag{29}
$$

(ii) the row $ { \mathcal { A } } ( x _ { q } )$ ofthe effective kernel is Lipschitz in $x _ { q }$ with respect to the Euclidean norm, with constant

$$
\begin{array} { r } { L _ { \mathcal { A } } = \frac { 1 } { 2 } \sqrt { R / 2 } \ \alpha _ { \lambda } \| W _ { \lambda } \| _ { 2 } + \underset { r } { \operatorname* { m a x } } \left( L _ { r , y } + L _ { r , x } \right) , } \end{array}\tag{30}
$$

where

$$
L _ { r , a } = \frac { \sqrt { n _ { a } } } { 2 \tau _ { 0 } } \left( \frac { \Delta } { \sigma _ { \star } ^ { 2 } } \alpha _ { \mu } M \| W _ { \mu } \| _ { 2 } + \Big ( \frac { 1 } { \sigma _ { \star } } + \frac { \Delta ^ { 2 } } { \sigma _ { \star } ^ { 3 } } \Big ) \sigma ^ { \star } \alpha _ { \sigma } \| W _ { \sigma } \| _ { 2 } \right) .\tag{31}
$$

Proof. (i) Since | tanh $\lvert \leq 1$ and $\sigma _ { r , a } \geq \sigma _ { \operatorname* { m i n } } , | \mu _ { r , a } ( x _ { q } ) - \mu _ { r , a } | \leq \alpha _ { \mu } M$ and $\sigma _ { \star } \leq \sigma _ { r , a } ( x _ { q } ) \leq \sigma ^ { \star }$ (ii) Fix two tokens $x _ { q }$ and $x _ { q } ^ { \prime }$ . For the center, write $\boldsymbol { u } = [ W _ { \mu } x _ { q } + b _ { \mu } ] _ { r , a }$ and $u ^ { \prime } = [ W _ { \mu } x _ { q } ^ { \prime } + b _ { \mu } ] _ { r , a }$ for the pre-activations of its predictor, so that $| u - u ^ { \prime } | \leq \| W _ { \mu } \| _ { 2 } \| x _ { q } - x _ { q } ^ { \prime } \| _ { 2 }$ . Since tanh is 1-Lipschitz,

$$
| \mu _ { r , a } ( x _ { q } ) - \mu _ { r , a } ( x _ { q } ^ { \prime } ) | = \alpha _ { \mu } M | \operatorname { t a n h } u - \operatorname { t a n h } u ^ { \prime } | \leq \alpha _ { \mu } M \| W _ { \mu } \| _ { 2 } \| x _ { q } - x _ { q } ^ { \prime } \| _ { 2 } .
$$

□

For the spread, write likewise $v = [ W _ { \sigma } x _ { q } + b _ { \sigma } ] _ { r , a }$ and $v ^ { \prime } = [ W _ { \sigma } x _ { q } ^ { \prime } + b _ { \sigma } ] _ { r , a }$ , and regard the steered spread as the function $v \mapsto \sigma _ { r , a } \exp ( \alpha _ { \sigma }$ tanh v). Its derivative satisfies

$$
0 \leq \frac { \partial \sigma _ { r , a } } { \partial v } = \sigma _ { r , a } e ^ { \alpha _ { \sigma } \operatorname { t a n h } v } \alpha _ { \sigma } \operatorname { t a n h } ^ { \prime } ( v ) \leq \sigma ^ { \star } \alpha _ { \sigma } ,
$$

because the steered spread never exceeds $\sigma ^ { \star }$ and $0 < \operatorname { t a n h } ^ { \prime } \le 1$ . By the mean value theorem,

$$
| \sigma _ { r , a } ( x _ { q } ) - \sigma _ { r , a } ( x _ { q } ^ { \prime } ) | \leq \sigma ^ { \star } \alpha _ { \sigma } | v - v ^ { \prime } | \leq \sigma ^ { \star } \alpha _ { \sigma } \| W _ { \sigma } \| _ { 2 } \| x _ { q } - x _ { q } ^ { \prime } \| _ { 2 } .
$$

The axis row $A _ { r , a } ( x _ { q } )$ is the softmax of $z \ \in \ \mathbb { R } ^ { n _ { a } } , \ z _ { s } \ = \ \log g _ { r , a } ( q _ { a } \ - \ s )$ , where log $\begin{array} { r l } { g _ { r , a } } & { { } = } \end{array}$ $\tau ^ { - 1 } \log G ^ { a }$ , with log $G ^ { a }$ as in equation $^ { 2 , }$ , is evaluated at $\mu _ { r , a } ( x _ { q } )$ and $\sigma _ { r , a } ( x _ { q } )$ ; let $z ^ { \prime }$ be the same vector at $x _ { q } ^ { \prime } .$ . On $\Theta _ { r , a } ,$ every grid offset satisfies $| \delta _ { a } - \mu | \leq \Delta , \operatorname { s o } { \tau } \geq \tau _ { 0 }$ gives

$$
\left| \frac { \partial \log g _ { r , a } } { \partial \mu } \right| \leq \frac { \Delta } { \tau _ { 0 } \sigma _ { \star } ^ { 2 } } , \qquad \left| \frac { \partial \log g _ { r , a } } { \partial \sigma } \right| \leq \frac { 1 } { \tau _ { 0 } } \left( \frac { 1 } { \sigma _ { \star } } + \frac { \Delta ^ { 2 } } { \sigma _ { \star } ^ { 3 } } \right) .\tag{32}
$$

Since $\Theta _ { r , a }$ is convex, the segment joining $\left( \mu _ { r , a } ( x _ { q } ) , \sigma _ { r , a } ( x _ { q } ) \right)$ and $\left( \mu _ { r , a } ( x _ { q } ^ { \prime } ) , \sigma _ { r , a } ( x _ { q } ^ { \prime } ) \right)$ stays in $\Theta _ { r , a } .$ , where the bounds equation 32 hold. The mean value theorem along this segment, together with the two estimates above, gives for every source coordinate s

$$
\begin{array} { r l r } {  { \vert z _ { s } - z _ { s } ^ { \prime } \vert \leq \frac { \Delta } { \tau _ { 0 } \sigma _ { \star } ^ { 2 } }  \mu _ { r , a } ( x _ { q } ) - \mu _ { r , a } ( x _ { q } ^ { \prime } )  } } \\ & { } & { \quad + \frac { 1 } { \tau _ { 0 } } \Big ( \frac { 1 } { \sigma _ { \star } } + \frac { \Delta ^ { 2 } } { \sigma _ { \star } ^ { 3 } } \Big )  \sigma _ { r , a } ( x _ { q } ) - \sigma _ { r , a } ( x _ { q } ^ { \prime } )  \leq \frac { 2 } { \sqrt { n _ { a } } } L _ { r , a } \| x _ { q } - x _ { q } ^ { \prime } \| _ { 2 } , } \end{array}\tag{33}
$$

where the last step is the definition equation 31 of $L _ { r , a }$ . It remains to pass from the logits z to the normalized row. The softmax is ${ \frac { 1 } { 2 } } \cdot$ -Lipschitz in the Euclidean norm, because its Jacobian diag(p) − $p p ^ { \top }$ is positive semidefinite with spectral norm at most $\frac { 1 } { 2 }$ . Hence

$$
\begin{array} { r } { \| A _ { r , a } ( x _ { q } ) - A _ { r , a } ( x _ { q } ^ { \prime } ) \| _ { 2 } \leq \frac { 1 } { 2 } \| z - z ^ { \prime } \| _ { 2 } \leq \frac { 1 } { 2 } \sqrt { n _ { a } } \operatorname* { m a x } _ { s } | z _ { s } - z _ { s } ^ { \prime } | \leq L _ { r , a } \| x _ { q } - x _ { q } ^ { \prime } \| _ { 2 } . } \end{array}\tag{34}
$$

For the effective kernel, write $c _ { r } = A _ { r , y } ( x _ { q } ) \otimes A _ { r , x } ( x _ { q } ) , c _ { r } ^ { \prime }$ for its value at $x _ { q } ^ { \prime } .$ , and $d = \lambda ( x _ { q } ) -$ $\lambda ( x _ { q } ^ { \prime } )$ , so that

$$
\boldsymbol { \mathcal { A } } ( \boldsymbol { x } _ { q } ) - \boldsymbol { \mathcal { A } } ( \boldsymbol { x } _ { q } ^ { \prime } ) = \sum _ { r } d _ { r } \boldsymbol { c } _ { r } + \sum _ { r } \boldsymbol { \lambda } _ { r } ( \boldsymbol { x } _ { q } ^ { \prime } ) ( \boldsymbol { c } _ { r } - \boldsymbol { c } _ { r } ^ { \prime } ) .\tag{35}
$$

We bound the two sums separately.

Second sum. All rows are probability vectors, so their Euclidean norm is at most one. For any vectors $a , a ^ { \prime } , b , b ^ { \prime }$

$$
\| a \otimes b - a ^ { \prime } \otimes b ^ { \prime } \| _ { 2 } \leq \| a - a ^ { \prime } \| _ { 2 } \| b \| _ { 2 } + \| a ^ { \prime } \| _ { 2 } \| b - b ^ { \prime } \| _ { 2 } ,\tag{36}
$$

and applying it with $a = A _ { r , y } ( x _ { q } ) , b = A _ { r , x } ( x _ { q } )$ and their counterparts at $x _ { q } ^ { \prime }$ gives

$$
\| c _ { r } - c _ { r } ^ { \prime } \| _ { 2 } \leq \left( L _ { r , y } + L _ { r , x } \right) \| x _ { q } - x _ { q } ^ { \prime } \| _ { 2 } .\tag{37}
$$

Since the weights $\lambda _ { r } ( x _ { q } ^ { \prime } )$ form a convex combination, the second sum is at most max<sub>r</sub> $( L _ { r , y } +$ $L _ { r , x } ) \| x _ { q } - x _ { q } ^ { \prime } \| _ { 2 }$

First sum. Its coefficients add up to zero, $\textstyle \sum _ { r } d _ { r } = 0$ , so splitting d into its positive and negative parts gives $\begin{array} { r } { \sum _ { r } d _ { r } c _ { r } = \frac { 1 } { 2 } \| d \| _ { 1 } \left( P - Q \right) } \end{array}$ for two probability vectors $P$ and $Q .$ . Since $\| { \cal P } - { \cal Q } \| _ { 2 } ^ { 2 } \le$ $\| P - \mathsf { \overline { { Q } } } \| _ { 1 } \| \overline { { P } } ^ { \prime } - Q \| _ { \infty } \leq 2 \| Q \| _ { \infty } ,$

$$
\Big \| \sum _ { r } d _ { r } c _ { r } \Big \| _ { 2 } \leq \frac { \| d \| _ { 1 } } { \sqrt { 2 } } \leq \sqrt { R / 2 } \| d \| _ { 2 } .\tag{38}
$$

Finally, the mixture weights are a softmax of $\ell _ { r } + \alpha _ { \lambda }$ tanh(·); since the softmax is ${ \frac { 1 } { 2 } } \cdot$ -Lipschitz and tanh is 1-Lipschitz,

$$
\begin{array} { r } { \| d \| _ { 2 } \leq \frac { 1 } { 2 } \alpha _ { \lambda } \| W _ { \lambda } \| _ { 2 } \| x _ { q } - x _ { q } ^ { \prime } \| _ { 2 } . } \end{array}\tag{39}
$$

Adding the bounds on the two sums gives equation 30.

The constants $\boldsymbol { L } _ { r , a }$ and $L _ { A }$ of Lemma 2 are finite because of three design choices of $\ S 2$ and the finiteness of the grid: the floor $\sigma _ { \mathrm { m i n } }$ keeps the spread away from zero, the steering radius M bounds how far the center moves, the offset $\tau _ { 0 }$ keeps the temperature away from zero, and $\delta _ { a }$ ranges over a finite set. Removing any one of them makes the corresponding bound in equation 32, and hence $L _ { r , a }$ , diverge.

Proposition 2 (Steered SSoG is Lipschitz on bounded sets). Let $\rho > 0 ,$ , let $X , W _ { V } ^ { h } , W _ { V }$ and $W _ { O }$ be as in Proposition 1, and let the atom parameters of every head h be steered as in equation ${ \mathit { 9 , } }$ under the assumptions ofLemma 2, so that the effective kernel $\mathbf { \bar { \mathcal { A } } } ^ { h } ( X ) \in \mathbb { R } ^ { ( N - 1 ) \times ( N - 1 ) }$ has rows $A ^ { h } ( x _ { q } )$ as in that lemma. For each head $h ,$ atom r and axis $^ { a , }$ let $\Theta _ { r , a } ^ { h }$ be the set $\Theta _ { r , a }$ ofLemma 2for head $h ,$ and let

$$
\bar { \kappa } _ { r , a } ^ { h } = \frac { \operatorname* { m a x } _ { s _ { a } } \sum _ { q _ { a } } \operatorname* { s u p } _ { \theta \in \Theta _ { r , a } ^ { h } } g _ { \theta } ( q _ { a } - s _ { a } ) } { \operatorname* { m i n } _ { q _ { a } } \operatorname* { i n f } _ { \theta \in \Theta _ { r , a } ^ { h } } \sum _ { s _ { a } } g _ { \theta } ( q _ { a } - s _ { a } ) } ,\tag{40}
$$

where g<sub>θ</sub> denotes the kernel $g _ { r , a }$ of head h evaluated at $\theta \ : = \ : ( \mu , \sigma )$ . Then, for all $X , X ^ { \prime }$ with $\| X \| _ { F } , \| X ^ { \prime } \| _ { F } \leq \rho ,$

$$
\begin{array} { r } { \| \mathrm { S S o G } ( X ) - \mathrm { S S o G } ( X ^ { \prime } ) \| _ { F } \leq \| W _ { V } \| _ { 2 } \| W _ { O } \| _ { 2 } \Big ( \bar { K } _ { A } + \operatorname* { m a x } _ { h } { L _ { A } ^ { h } \rho } \Big ) \| X - X ^ { \prime } \| _ { F } , } \end{array}\tag{41}
$$

where

$$
\begin{array} { r } { \bar { K } _ { \mathcal { A } } : = \operatorname* { m a x } _ { h } \sqrt { \frac { 1 } { N - 1 } + \sum _ { r } \bar { \kappa } _ { r , y } ^ { h } \bar { \kappa } _ { r , x } ^ { h } } } \end{array}\tag{42}
$$

and $L _ { \mathcal { A } } ^ { h }$ is the constant equation $3 0 f o r$ head h. In particular, the Lipschitz bound on the ball of radius $\rho$ grows linearly in $\rho .$

Proof. With the class token last, as in the proof of Proposition 1, let $M ^ { h } ( X ) = \left( \begin{array} { c c } { { A ^ { h } ( X ) } } & { { 0 } } \\ { { \frac { 1 } { N - 1 } { \bf 1 } ^ { \top } } } & { { 0 } } \end{array} \right)$ Denoting by

$$
Z ^ { h } ( X ) = M ^ { h } ( X ) X W _ { V } ^ { h } \in \mathbb { R } ^ { N \times D ^ { h } }
$$

the output of head $h ,$ , the SSoG output is obtained by concatenating the heads and applying the output projection,

$$
\operatorname { S S o G } ( X ) = { \big ( } Z ^ { 1 } ( X ) \parallel \cdots \parallel Z ^ { H } ( X ) { \big ) } W _ { O } .
$$

Since the class row of $M ^ { h }$ does not depend on X,

$$
Z ^ { h } ( X ) - Z ^ { h } ( X ^ { \prime } ) = M ^ { h } ( X ) ( X - X ^ { \prime } ) W _ { V } ^ { h } + \left( \sp { \left( A ^ { h } ( X ) - { A ^ { h } ( X ^ { \prime } ) } \right) X _ { \mathrm { { P } } } ^ { \prime } } W _ { V } ^ { h } \right) ,
$$

where $X _ { \mathrm { P } } ^ { \prime }$ collects the patch rows of $X ^ { \prime }$ . Arguing over the heads as in the proof of Proposition 1 and using the triangle inequality, the first terms contribute at most:

$$
\| W _ { V } \| _ { 2 } \| W _ { O } \| _ { 2 } \operatorname* { m a x } _ { h } \| M ^ { h } ( X ) \| _ { 2 } \| X - X ^ { \prime } \| _ { F }
$$

and, since $\| X _ { \mathrm { P } } ^ { \prime } W _ { V } \| _ { F } \le \| W _ { V } \| _ { 2 } \| X ^ { \prime } \| _ { F }$ , the second at most:

$$
\| W _ { V } \| _ { 2 } \| W _ { O } \| _ { 2 } \operatorname* { m a x } _ { h } \| \boldsymbol { \mathcal { A } } ^ { h } ( \boldsymbol { X } ) - \boldsymbol { \mathcal { A } } ^ { h } ( \boldsymbol { X } ^ { \prime } ) \| _ { 2 } \| \boldsymbol { X } ^ { \prime } \| _ { F } .
$$

By Lemma 2(i), every steered parameter lies in $\Theta _ { r , a } ^ { h } .$ , a compact set on which $g _ { \theta }$ is continuous and positive on the finite grid, so $\bar { \kappa } _ { r , a } ^ { h } \ < \ \infty$ . Every row of ${ \mathcal { A } } ^ { h } ( X )$ is a probability vector, so $\| { \mathcal { A } } ^ { h } ( X ) \| _ { \infty } = 1$ . Each entry of a row of $A _ { r , a } ^ { h }$ is at most

$$
\operatorname* { s u p } _ { \theta \in \Theta _ { r , a } ^ { h } } { g _ { \theta } ( q _ { a } - s _ { a } ) / \operatorname* { m i n } _ { q _ { a } } \operatorname* { i n f } _ { \theta \in \Theta _ { r , a } ^ { h } } \sum _ { s _ { a } } g _ { \theta } ( q _ { a } - s _ { a } ) }
$$

and $\lambda _ { r } ^ { h } \leq 1$ since $\lambda ^ { h }$ is a softmax, so the column of ${ \mathcal { A } } ^ { h } ( X )$ indexed by $\boldsymbol { s } = \left( s _ { y } , s _ { x } \right)$ sums to at most $\begin{array} { r } { \sum _ { r } \bar { \kappa } _ { r , y } ^ { h } \bar { \kappa } _ { r , x } ^ { h } } \end{array}$ . Hence

$$
\| \boldsymbol { \mathcal { A } } ^ { h } ( \boldsymbol { X } ) \| _ { 2 } \leq ( \| \boldsymbol { \mathcal { A } } ^ { h } ( \boldsymbol { X } ) \| _ { 1 } \| \boldsymbol { \mathcal { A } } ^ { h } ( \boldsymbol { X } ) \| _ { \infty } ) ^ { 1 / 2 } \leq \big ( \sum _ { r } \bar { \kappa } _ { r , y } ^ { h } \bar { \kappa } _ { r , x } ^ { h } \big ) ^ { 1 / 2 }
$$

and, exactly as in Proposition 1,

$$
\| M ^ { h } ( { \cal X } ) \| _ { 2 } \le \bigl ( { \textstyle \frac { 1 } { N - 1 } } + \sum _ { r } \bar { \kappa } _ { r , y } ^ { h } \bar { \kappa } _ { r , x } ^ { h } \bigr ) ^ { 1 / 2 } ,
$$

so that max<sub>h</sub> $\| M ^ { h } ( X ) \| _ { 2 } \leq \bar { K } _ { A }$ . By Lemma 2(ii),

$$
\| \mathcal { A } ^ { h } ( X ) - \mathcal { A } ^ { h } ( X ^ { \prime } ) \| _ { 2 } \leq \| \mathcal { A } ^ { h } ( X ) - \mathcal { A } ^ { h } ( X ^ { \prime } ) \| _ { F } = \big ( \sum _ { q } \| \mathcal { A } ^ { h } ( x _ { q } ) - \mathcal { A } ^ { h } ( x _ { q } ^ { \prime } ) \| _ { 2 } ^ { 2 } \big ) ^ { 1 / 2 } \leq L _ { A } ^ { h } \| X - X ^ { \prime } \| _ { F } .
$$

Combining the two contributions,

$$
\| \mathrm { S S o G } ( X ) - \mathrm { S S o G } ( X ^ { \prime } ) \| _ { F } \leq \| W _ { V } \| _ { 2 } \| W _ { O } \| _ { 2 } \Bigl ( \bar { K } _ { A } + \operatorname* { m a x } _ { h } L _ { A } ^ { h } \| X ^ { \prime } \| _ { F } \Bigr ) \| X - X ^ { \prime } \| _ { F } ,
$$

and $\| X ^ { \prime } \| _ { F } \leq \rho$ gives equation 41.

Remark. The dependence on $\rho$ in equation 41 cannot be removed in general. For non-degenerate steering, fix a query position q and its token $x _ { q } .$ , and scale the remaining patch tokens by a factor c: a perturbation of $x _ { q }$ changes the kernel row $A ^ { h } ( x _ { q } )$ at a rate that does not depend on c, while the values it multiplies are of size $O ( c )$ , so the directional derivative of steered SSoG grows linearly in $c .$ When the steering does not act, for instance when $W _ { \mu } = W _ { \sigma } = W _ { \lambda } = 0$ , the operator reduces to fixed SSoG and Proposition 1 applies.

The preceding Remark has the same structure as the obstruction that Kim et al. (2021) identify for dot-product attention, namely a data-dependent mixing matrix multiplying a data-dependent value matrix, but not the same rate. Attention logits are inner products between two token features, so the Lipschitz constant of the mixing matrix itself grows with $\| X \|$ and the available upper bounds on a ball of radius $\rho$ grow as $\rho ^ { 2 }$ ; the steered kernel depends only on the query token and only through a bounded tanh, so $L _ { \mathcal { A } } ^ { h }$ does not depend on X and the bound of Proposition 2 grows as $\rho .$ Relative to dot-product attention, steering thus lowers the exponent of this bound from two to one, and the fixed variant removes the dependence on $\rho$ altogether. We therefore regard Proposition 1 as a further reason to prefer the fixed operator, alongside the accuracy results of $\ S \bar { 3 } . 2$